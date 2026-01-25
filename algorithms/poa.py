"""
POA逐步优化算法
===============

Progressive Optimality Algorithm
"""
import numpy as np
from typing import Dict, Any

from .base import BaseDispatchAlgorithm
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
from core.simulation import calculate_stage_energy


class POADispatch(BaseDispatchAlgorithm):
    """
    POA逐步优化算法
    
    从初始水位轨迹开始，逐时段优化调整
    """
    
    algorithm_name = "POA逐步优化"
    
    def __init__(
        self,
        data: Dict[str, Any] = None,
        reservoir_config: ReservoirConfig = None,
        dispatch_config: DispatchConfig = None,
        algorithm_config: AlgorithmConfig = None,
        data_file: str = None,
        initial_trajectory: np.ndarray = None
    ):
        super().__init__(data, reservoir_config, dispatch_config, algorithm_config, data_file)
        
        # POA参数
        self.max_iterations = self.algorithm_config.poa_max_iterations
        self.tolerance = self.algorithm_config.poa_tolerance
        self.d_water_level = self.algorithm_config.d_water_level
        self.penalty_coefficient = self.algorithm_config.penalty_coefficient
        
        # 初始轨迹
        self.initial_trajectory = initial_trajectory
    
    def set_initial_trajectory(self, trajectory: np.ndarray):
        """设置初始水位轨迹"""
        self.initial_trajectory = trajectory
    
    def _calculate_stage_energy(
        self,
        Z_start: float,
        Z_end: float,
        inflow: float,
        stage: int
    ) -> float:
        """计算单阶段发电量"""
        month_idx = stage % 12
        H_dead = self.data['dispatch_limits']['H_dead'][month_idx]
        H_max = self.data['dispatch_limits']['H_max'][month_idx]
        
        return calculate_stage_energy(
            Z_start, Z_end, inflow,
            self.fitters,
            self.reservoir_config,
            self.dispatch_config,
            self.penalty_coefficient,
            H_dead, H_max
        )
    
    def _run_algorithm(self) -> np.ndarray:
        """
        运行POA算法
        
        Returns
        -------
        np.ndarray
            最优水位轨迹（长度为n_stages+1）
        """
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level
        
        # 初始化水位轨迹
        if self.initial_trajectory is None:
            # 使用线性插值作为初始轨迹
            water_level_trajectory = np.linspace(
                initial_water_level, final_water_level, self.n_stages + 1)
        else:
            water_level_trajectory = self.initial_trajectory.copy()
        
        # 确保始末水位固定
        water_level_trajectory[0] = initial_water_level
        water_level_trajectory[-1] = final_water_level
        
        # 存储收敛历史
        total_energy_history = []
        
        print(f"  开始POA优化，最大迭代次数: {self.max_iterations}")
        
        for iter_num in range(self.max_iterations):
            old_trajectory = water_level_trajectory.copy()
            
            # 逐个时段进行优化（不优化第一个和最后一个时段）
            for t in range(1, self.n_stages):
                month_idx = (t - 1) % 12
                
                # 确定当前时段的上下限
                H_dead = self.data['dispatch_limits']['H_dead'][month_idx]
                H_max = self.data['dispatch_limits']['H_max'][month_idx]
                
                # 生成离散的水位候选点
                Z_candidates = np.arange(H_dead, H_max + self.d_water_level, self.d_water_level)
                
                best_energy = -np.inf
                best_Z = water_level_trajectory[t]
                
                # 对每个候选水位进行评估
                for candidate_Z in Z_candidates:
                    # 第一阶段：从前一时段水位到候选水位
                    energy1 = self._calculate_stage_energy(
                        water_level_trajectory[t-1], candidate_Z, self.inflow[t-1], t-1)
                    
                    if energy1 < 0:
                        continue
                    
                    # 第二阶段：从候选水位到下一时段水位
                    if t < self.n_stages:
                        energy2 = self._calculate_stage_energy(
                            candidate_Z, water_level_trajectory[t+1], self.inflow[t], t)
                        
                        if energy2 < 0:
                            continue
                    else:
                        energy2 = 0
                    
                    total_energy = energy1 + energy2
                    
                    if total_energy > best_energy:
                        best_energy = total_energy
                        best_Z = candidate_Z
                
                # 更新当前时段的最优水位
                water_level_trajectory[t] = best_Z
            
            # 确保始末水位保持不变
            water_level_trajectory[0] = initial_water_level
            water_level_trajectory[-1] = final_water_level
            
            # 计算总发电量用于收敛判断
            total_energy = 0
            for t in range(self.n_stages):
                energy = self._calculate_stage_energy(
                    water_level_trajectory[t], water_level_trajectory[t+1],
                    self.inflow[t], t)
                if energy > 0:
                    total_energy += energy
            
            total_energy_history.append(total_energy)
            
            # 检查收敛
            max_change = np.max(np.abs(water_level_trajectory[1:-1] - old_trajectory[1:-1]))
            
            if (iter_num + 1) % 10 == 0:
                print(f"    迭代 {iter_num + 1}: 总发电量 = {total_energy / 1000:.2f} GWh, "
                      f"最大水位变化 = {max_change:.4f} m")
            
            if max_change < self.tolerance:
                print(f"  POA算法在{iter_num + 1}次迭代后收敛")
                break
        
        # 保存收敛历史
        self._convergence_history = np.array(total_energy_history)
        
        return water_level_trajectory
    
    def run(self):
        """重写run方法以保存收敛历史"""
        result = super().run()
        if hasattr(self, '_convergence_history'):
            result.convergence_history = self._convergence_history
        return result
