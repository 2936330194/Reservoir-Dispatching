"""
DDDP离散微分动态规划算法
========================

Discrete Differential Dynamic Programming
"""
import numpy as np
from typing import Dict, Any, List

from .base import BaseDispatchAlgorithm
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
from core.simulation import calculate_stage_energy


class DDDPDispatch(BaseDispatchAlgorithm):
    """
    DDDP离散微分动态规划算法
    
    在参考轨迹附近构建走廊，走廊内进行DP，迭代优化
    """
    
    algorithm_name = "DDDP离散微分动态规划"
    
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
        
        # DDDP参数
        self.max_iterations = self.algorithm_config.dddp_max_iterations
        self.tolerance = self.algorithm_config.dddp_tolerance
        self.corridor_width = self.algorithm_config.dddp_corridor_width
        self.d_water_level = self.algorithm_config.dddp_water_level
        self.penalty_coefficient = self.algorithm_config.penalty_coefficient
        
        # 初始轨迹
        self.initial_trajectory = initial_trajectory
    
    def set_initial_trajectory(self, trajectory: np.ndarray):
        """设置初始水位轨迹（参考轨迹）"""
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
    
    def _build_state_corridors(self, reference_trajectory: np.ndarray) -> List[np.ndarray]:
        """
        构建走廊内的状态点
        
        Parameters
        ----------
        reference_trajectory : np.ndarray
            参考轨迹
        
        Returns
        -------
        List[np.ndarray]
            每个时段的状态候选点列表
        """
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level
        
        H_max_monthly = self.data['dispatch_limits']['H_max']
        H_dead_monthly = self.data['dispatch_limits']['H_dead']
        
        state_corridors = []
        
        for t in range(self.n_stages + 1):
            if t == 0:
                # 初始状态固定
                state_corridors.append(np.array([initial_water_level]))
            elif t == self.n_stages:
                # 终止状态固定
                state_corridors.append(np.array([final_water_level]))
            else:
                # 中间状态在参考轨迹附近构建走廊
                month_idx = (t - 1) % 12
                Z_ref = reference_trajectory[t]
                Z_min_corridor = max(H_dead_monthly[month_idx], Z_ref - self.corridor_width)
                Z_max_corridor = min(H_max_monthly[month_idx], Z_ref + self.corridor_width)
                
                # 在走廊内离散状态点
                states = np.arange(Z_min_corridor, Z_max_corridor + self.d_water_level, 
                                   self.d_water_level)
                state_corridors.append(states)
        
        return state_corridors
    
    def _dddp_in_corridor(
        self,
        state_corridors: List[np.ndarray]
    ) -> tuple:
        """
        在走廊内运行动态规划
        
        Parameters
        ----------
        state_corridors : List[np.ndarray]
            每个时段的状态候选点列表
        
        Returns
        -------
        tuple
            (optimal_trajectory, total_energy)
        """
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level
        
        n_max_states = max(len(s) for s in state_corridors)
        
        # DP表格
        cumulative_energy = np.full((self.n_stages + 1, n_max_states), -np.inf)
        backtrack_pointer = np.zeros((self.n_stages + 1, n_max_states), dtype=int)
        
        # 初始阶段
        cumulative_energy[0, 0] = 0
        
        # 逐阶段递推
        for t in range(self.n_stages):
            current_states = state_corridors[t]
            next_states = state_corridors[t + 1]
            n_current = len(current_states)
            n_next = len(next_states)
            
            for j in range(n_next):
                best_energy = -np.inf
                best_prev_index = 0
                
                for i in range(n_current):
                    if cumulative_energy[t, i] <= -np.inf:
                        continue
                    
                    energy = self._calculate_stage_energy(
                        current_states[i], next_states[j], self.inflow[t], t)
                    
                    if energy < 0:
                        continue
                    
                    total_energy_val = cumulative_energy[t, i] + energy
                    
                    if total_energy_val > best_energy:
                        best_energy = total_energy_val
                        best_prev_index = i
                
                cumulative_energy[t + 1, j] = best_energy
                backtrack_pointer[t + 1, j] = best_prev_index
        
        # 回溯最优路径
        optimal_trajectory = np.zeros(self.n_stages + 1)
        
        # 找到最终状态对应的索引
        final_states = state_corridors[-1]
        final_idx = np.argmin(np.abs(final_states - final_water_level))
        optimal_trajectory[-1] = final_states[final_idx]
        
        # 回溯路径
        for t in range(self.n_stages, 0, -1):
            current_state_idx = backtrack_pointer[t, final_idx]
            current_states = state_corridors[t - 1]
            optimal_trajectory[t - 1] = current_states[current_state_idx]
            final_idx = current_state_idx
        
        # 确保始末状态正确
        optimal_trajectory[0] = initial_water_level
        optimal_trajectory[-1] = final_water_level
        
        # 计算总发电量
        final_states = state_corridors[-1]
        final_idx = np.argmin(np.abs(final_states - final_water_level))
        total_energy = cumulative_energy[-1, final_idx]
        
        return optimal_trajectory, total_energy
    
    def _run_algorithm(self) -> np.ndarray:
        """
        运行DDDP算法
        
        Returns
        -------
        np.ndarray
            最优水位轨迹（长度为n_stages+1）
        """
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level
        
        # 初始化参考轨迹
        if self.initial_trajectory is None:
            # 使用线性插值作为初始轨迹
            reference_trajectory = np.linspace(
                initial_water_level, final_water_level, self.n_stages + 1)
        else:
            reference_trajectory = self.initial_trajectory.copy()
        
        # 存储收敛历史
        total_energy_history = []
        convergence_history = []
        
        print(f"  开始DDDP优化，最大迭代次数: {self.max_iterations}")
        
        for iter_num in range(self.max_iterations):
            # 构建当前走廊
            state_corridors = self._build_state_corridors(reference_trajectory)
            
            # 在走廊内运行DP
            new_trajectory, total_energy = self._dddp_in_corridor(state_corridors)
            
            total_energy_history.append(total_energy)
            
            # 计算收敛性（水位变化）
            max_water_level_change = np.max(
                np.abs(new_trajectory[1:-1] - reference_trajectory[1:-1]))
            convergence_history.append(max_water_level_change)
            
            if (iter_num + 1) % 5 == 0:
                print(f"    迭代 {iter_num + 1}: 总发电量 = {total_energy / 1000:.2f} GWh, "
                      f"最大水位变化 = {max_water_level_change:.4f} m")
            
            # 检查收敛
            if max_water_level_change < self.tolerance:
                print(f"  DDDP算法在{iter_num + 1}次迭代后收敛")
                reference_trajectory = new_trajectory
                break
            
            # 更新参考轨迹
            reference_trajectory = new_trajectory
            
            # 动态调整走廊宽度
            if iter_num > 5 and convergence_history[-1] < 0.5 * convergence_history[-2]:
                self.corridor_width = max(0.5, self.corridor_width * 0.8)
        
        # 保存收敛历史
        self._convergence_history = np.array(total_energy_history)
        
        return reference_trajectory
    
    def run(self):
        """重写run方法以保存收敛历史"""
        result = super().run()
        if hasattr(self, '_convergence_history'):
            result.convergence_history = self._convergence_history
        return result
