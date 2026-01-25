"""
算法基类
========

定义调度算法的通用接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import time

from core.data_loader import load_reservoir_data, get_inflow_sequence, get_monthly_limits
from core.curve_fitting import CurveFitters
from core.simulation import SimulationResult, simulate_from_water_level_seq
from config import (
    ReservoirConfig, DispatchConfig, AlgorithmConfig,
    DEFAULT_RESERVOIR, DEFAULT_DISPATCH, DEFAULT_ALGORITHM
)


@dataclass
class DispatchResult:
    """调度算法结果"""
    algorithm_name: str              # 算法名称
    water_level_trajectory: np.ndarray  # 最优水位轨迹
    simulation_result: SimulationResult  # 仿真结果
    convergence_history: np.ndarray = None  # 收敛历史（可选）
    elapsed_time: float = 0.0        # 运行时间 (秒)
    extra_info: Dict[str, Any] = None  # 额外信息


class BaseDispatchAlgorithm(ABC):
    """
    调度算法基类
    
    定义所有调度算法的通用接口和基础功能
    """
    
    algorithm_name: str = "Base"
    
    def __init__(
        self,
        data: Dict[str, Any] = None,
        reservoir_config: ReservoirConfig = None,
        dispatch_config: DispatchConfig = None,
        algorithm_config: AlgorithmConfig = None,
        data_file: str = None
    ):
        """
        初始化调度算法
        
        Parameters
        ----------
        data : Dict[str, Any], optional
            预加载的数据字典，如果为None则自动加载
        reservoir_config : ReservoirConfig, optional
            水库配置
        dispatch_config : DispatchConfig, optional
            调度配置
        algorithm_config : AlgorithmConfig, optional
            算法配置
        data_file : str, optional
            数据文件路径（仅在data为None时使用）
        """
        # 加载数据
        if data is None:
            self.data = load_reservoir_data(data_file)
        else:
            self.data = data
        
        # 设置配置
        self.reservoir_config = reservoir_config or DEFAULT_RESERVOIR
        self.dispatch_config = dispatch_config or DEFAULT_DISPATCH
        self.algorithm_config = algorithm_config or DEFAULT_ALGORITHM
        
        # 创建曲线拟合器
        self.fitters = CurveFitters(self.data)
        
        # 获取入库流量序列
        self.inflow = get_inflow_sequence(self.data)
        
        # 获取水位限制
        self.H_max_full, self.H_dead_full = get_monthly_limits(
            self.data, self.dispatch_config.num_years)
        
        # 时段数
        self.n_stages = self.dispatch_config.n_stages
        
        # 结果
        self.result: Optional[DispatchResult] = None
    
    @abstractmethod
    def _run_algorithm(self) -> np.ndarray:
        """
        运行具体的优化算法
        
        Returns
        -------
        np.ndarray
            最优水位轨迹（长度为n_stages+1，包含初始水位）
        """
        pass
    
    def run(self) -> DispatchResult:
        """
        运行调度算法
        
        Returns
        -------
        DispatchResult
            调度结果
        """
        print(f"开始运行 {self.algorithm_name} 算法...")
        start_time = time.time()
        
        # 运行具体算法获取最优水位轨迹
        water_level_trajectory = self._run_algorithm()
        
        # 对水位轨迹进行仿真
        simulation_result = simulate_from_water_level_seq(
            water_level_trajectory,
            self.inflow,
            self.fitters,
            self.reservoir_config,
            self.dispatch_config
        )
        
        elapsed_time = time.time() - start_time
        
        # 构建结果
        self.result = DispatchResult(
            algorithm_name=self.algorithm_name,
            water_level_trajectory=water_level_trajectory,
            simulation_result=simulation_result,
            elapsed_time=elapsed_time
        )
        
        # 打印统计信息
        self._print_statistics()
        
        return self.result
    
    def _print_statistics(self):
        """打印调度结果统计信息"""
        if self.result is None:
            return
        
        sim = self.result.simulation_result
        
        print(f"\n=== {self.algorithm_name} 调度结果统计 ===")
        print(f"总发电量 = {sim.total_energy / 1000:.2f} GWh")
        
        # 年发电量
        num_years = self.dispatch_config.num_years
        annual_energy = np.zeros(num_years)
        for y in range(num_years):
            idx_start = y * 12
            idx_end = (y + 1) * 12
            annual_energy[y] = np.sum(sim.energy[idx_start:idx_end])
        avg_annual_energy = np.mean(annual_energy)
        print(f"多年平均年发电量 = {avg_annual_energy / 1000:.2f} GWh")
        
        # 保证率
        guaranteed_output = self.dispatch_config.guaranteed_output
        num_lower = np.sum(sim.power < guaranteed_output - self.tolerance)
        guarantee_rate = 1 - num_lower / self.n_stages
        print(f"保证率 = {guarantee_rate * 100:.2f} %")
        
        # 弃水量
        seconds_per_month = self.reservoir_config.seconds_per_month
        total_abandoned = np.sum(sim.abandoned_Q) * seconds_per_month / 1e8
        print(f"总弃水量 = {total_abandoned:.2f} 亿立方米")
        
        # 水量利用率
        water_use_rate = 100 * np.sum(sim.generated_Q) / np.sum(self.inflow)
        print(f"水量利用率 = {water_use_rate:.2f} %")
        
        print(f"运行时间 = {self.result.elapsed_time:.2f} 秒")
    
    def get_annual_statistics(self) -> Dict[str, np.ndarray]:
        """
        获取年度统计数据
        
        Returns
        -------
        Dict[str, np.ndarray]
            包含年发电量、年弃水量等统计数据
        """
        if self.result is None:
            raise ValueError("请先运行算法")
        
        sim = self.result.simulation_result
        num_years = self.dispatch_config.num_years
        seconds_per_month = self.reservoir_config.seconds_per_month
        
        annual_energy = np.zeros(num_years)
        annual_abandoned = np.zeros(num_years)
        annual_inflow = np.zeros(num_years)
        annual_outflow = np.zeros(num_years)
        annual_avg_water_level = np.zeros(num_years)
        
        for y in range(num_years):
            idx_start = y * 12
            idx_end = (y + 1) * 12
            
            annual_energy[y] = np.sum(sim.energy[idx_start:idx_end])
            annual_abandoned[y] = np.sum(sim.abandoned_Q[idx_start:idx_end]) * seconds_per_month / 1e8
            annual_inflow[y] = np.sum(self.inflow[idx_start:idx_end]) * seconds_per_month / 1e8
            annual_outflow[y] = np.sum(sim.outflow[idx_start:idx_end]) * seconds_per_month / 1e8
            annual_avg_water_level[y] = np.mean(sim.water_level[idx_start+1:idx_end+1])
        
        return {
            'annual_energy': annual_energy,           # MWh
            'annual_abandoned': annual_abandoned,     # 亿m³
            'annual_inflow': annual_inflow,           # 亿m³
            'annual_outflow': annual_outflow,         # 亿m³
            'annual_avg_water_level': annual_avg_water_level  # m
        }
