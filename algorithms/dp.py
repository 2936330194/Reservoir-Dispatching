"""
动态规划算法
============

使用DP求解水库逐月最优调度
"""
import numpy as np
from typing import Dict, Any, Tuple

from .base import BaseDispatchAlgorithm
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
from core.simulation import calculate_stage_energy


class DPDispatch(BaseDispatchAlgorithm):
    """
    动态规划调度算法
    
    离散化水位状态空间，使用DP求解最优水位轨迹
    """
    
    algorithm_name = "动态规划(DP)"
    
    def __init__(
        self,
        data: Dict[str, Any] = None,
        reservoir_config: ReservoirConfig = None,
        dispatch_config: DispatchConfig = None,
        algorithm_config: AlgorithmConfig = None,
        data_file: str = None
    ):
        super().__init__(data, reservoir_config, dispatch_config, algorithm_config, data_file)
        
        # DP参数
        self.d_water_level = self.algorithm_config.d_water_level
        self.penalty_coefficient = self.algorithm_config.penalty_coefficient
        # self.tolerance = self.algorithm_config.tolerance
        self.tolerance = 0
    
    def _generate_state_values(self, H_min: float, H_max: float) -> np.ndarray:
        """生成离散状态值"""
        return np.arange(H_min, H_max + self.d_water_level, self.d_water_level)
    
    def _calculate_transition_energy(
        self, 
        H_prev: float, 
        H_curr: float, 
        inflow: float,
        month_idx: int
    ) -> float:
        """
        计算状态转移的发电量
        
        Parameters
        ----------
        H_prev : float
            前一时段末水位
        H_curr : float
            当前时段末水位
        inflow : float
            当前时段入库流量
        month_idx : int
            月份索引 (0-11)
        
        Returns
        -------
        float
            发电量 (MWh)，负值表示不可行
        """
        H_dead = self.data['dispatch_limits']['H_dead'][month_idx]
        H_max = self.data['dispatch_limits']['H_max'][month_idx]
        
        return calculate_stage_energy(
            H_prev, H_curr, inflow,
            self.fitters,
            self.reservoir_config,
            self.dispatch_config,
            self.penalty_coefficient,
            H_dead, H_max
        )
    
    def _run_algorithm(self) -> np.ndarray:
        """
        运行DP算法
        
        Returns
        -------
        np.ndarray
            最优水位轨迹（长度为n_stages+1）
        """
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level
        
        # 生成全局状态空间
        global_min_H = np.min(self.H_dead_full)
        global_max_H = np.max(self.H_max_full)
        state_values_global = self._generate_state_values(global_min_H, global_max_H)
        n_state = len(state_values_global)
        
        print(f"  状态空间大小: {n_state} 个离散水位")
        
        # 初始化DP表格
        cumulative_energy = np.full((self.n_stages, n_state), -np.inf)
        backtrack = np.zeros((self.n_stages, n_state), dtype=int)
        
        # 第1阶段：从初始水位到所有候选状态
        print("  处理第 1 阶段...")
        month_idx = 0
        H_dead = self.data['dispatch_limits']['H_dead'][month_idx]
        H_max = self.data['dispatch_limits']['H_max'][month_idx]
        
        for j, next_H in enumerate(state_values_global):
            if next_H < H_dead or next_H > H_max:
                continue
            
            energy = self._calculate_transition_energy(
                initial_water_level, next_H, self.inflow[0], month_idx)
            
            if energy >= 0:
                cumulative_energy[0, j] = energy
                backtrack[0, j] = -1  # 初始状态标记
        
        # 中间阶段 (2 to n_stages-1)
        for t in range(1, self.n_stages - 1):
            if (t + 1) % 100 == 0:
                print(f"  处理第 {t + 1} 阶段...")
            
            month_idx = t % 12
            H_dead = self.data['dispatch_limits']['H_dead'][month_idx]
            H_max = self.data['dispatch_limits']['H_max'][month_idx]
            
            prev_month_idx = (t - 1) % 12
            prev_H_dead = self.data['dispatch_limits']['H_dead'][prev_month_idx]
            prev_H_max = self.data['dispatch_limits']['H_max'][prev_month_idx]
            
            for j, curr_H in enumerate(state_values_global):
                if curr_H < H_dead or curr_H > H_max:
                    continue
                
                best_cum = -np.inf
                best_k = 0
                
                for k, prev_H in enumerate(state_values_global):
                    if prev_H < prev_H_dead or prev_H > prev_H_max:
                        continue
                    
                    if cumulative_energy[t-1, k] <= -np.inf:
                        continue
                    
                    energy = self._calculate_transition_energy(
                        prev_H, curr_H, self.inflow[t], month_idx)
                    
                    if energy >= 0:
                        cum = cumulative_energy[t-1, k] + energy
                        if cum > best_cum:
                            best_cum = cum
                            best_k = k
                
                if best_cum > -np.inf:
                    cumulative_energy[t, j] = best_cum
                    backtrack[t, j] = best_k
        
        # 最后阶段：到指定终水位
        print(f"  处理第 {self.n_stages} 阶段...")
        t = self.n_stages - 1
        month_idx = t % 12
        
        prev_month_idx = (t - 1) % 12
        prev_H_dead = self.data['dispatch_limits']['H_dead'][prev_month_idx]
        prev_H_max = self.data['dispatch_limits']['H_max'][prev_month_idx]
        
        best_cum = -np.inf
        best_k = 0
        
        for k, prev_H in enumerate(state_values_global):
            if prev_H < prev_H_dead or prev_H > prev_H_max:
                continue
            
            if cumulative_energy[t-1, k] <= -np.inf:
                continue
            
            energy = self._calculate_transition_energy(
                prev_H, final_water_level, self.inflow[t], month_idx)
            
            if energy >= 0:
                cum = cumulative_energy[t-1, k] + energy
                if cum > best_cum:
                    best_cum = cum
                    best_k = k

                if best_cum > -np.inf:
                    cumulative_energy[t, best_k] = best_cum
                    backtrack[t, best_k] = best_k
        
        # 回溯最优路径
        optimal_trajectory = np.zeros(self.n_stages + 1)
        optimal_trajectory[0] = initial_water_level
        optimal_trajectory[-1] = final_water_level
        
        # 回溯
        optimal_index = np.zeros(self.n_stages, dtype=int)
        optimal_index[-1] = best_k  # 最后一个阶段前的最优状态索引
        
        for t in range(self.n_stages - 2, -1, -1):
            optimal_index[t] = backtrack[t + 1, optimal_index[t + 1]]
        
        # 填充水位轨迹
        for t in range(self.n_stages - 1):
            optimal_trajectory[t + 1] = state_values_global[optimal_index[t]]
        
        print(f"  DP完成，最优总发电量估计: {best_cum / 1000:.2f} GWh")
        
        return optimal_trajectory
