"""
粒子群优化算法调度
==================

使用PSO优化水库调度
"""
import numpy as np
from typing import Dict, Any

from .base import BaseDispatchAlgorithm
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
from optimizers.pso_optimizer import PSOOptimizer
from utils.helpers import clamp


class PSODispatch(BaseDispatchAlgorithm):
    """
    粒子群优化算法调度
    
    使用PSO求解最优水位序列
    """
    
    algorithm_name = "粒子群优化算法(PSO)"
    
    def __init__(
        self,
        data: Dict[str, Any] = None,
        reservoir_config: ReservoirConfig = None,
        dispatch_config: DispatchConfig = None,
        algorithm_config: AlgorithmConfig = None,
        data_file: str = None
    ):
        super().__init__(data, reservoir_config, dispatch_config, algorithm_config, data_file)
        
        # 算法参数
        self.population_size = self.algorithm_config.population_size
        self.max_iterations = self.algorithm_config.max_iterations
        self.random_seed = self.algorithm_config.random_seed
        
        # 惩罚系数
        self.pc_bounds = self.algorithm_config.penalty_bounds
        self.pc_spill = self.algorithm_config.penalty_spill
        self.pc_minQ = self.algorithm_config.penalty_min_Q
        self.pc_negQ = self.algorithm_config.penalty_neg_Q
        self.pc_guarantee = self.algorithm_config.penalty_guarantee
    
    def _fitness_function(self, H_row: np.ndarray) -> float:
        """
        适应度函数（最小化）
        
        与DholeDispatch相同的适应度函数
        """
        H = H_row.flatten()
        n = len(H)
        
        num_turbines = self.reservoir_config.num_turbines
        max_generation_flow = self.reservoir_config.max_generation_flow
        num_spillway = self.reservoir_config.num_spillway
        height_of_spillway = self.reservoir_config.height_of_spillway
        seconds_per_month = self.reservoir_config.seconds_per_month
        Q_min = self.dispatch_config.Q_min
        guaranteed_output = self.dispatch_config.guaranteed_output
        
        V = np.array([self.fitters.fit_V(h) for h in H])
        total_energy = 0
        penalty = 0
        
        prev_V = self.fitters.fit_V(H[-1])
        
        for t in range(n):
            month_idx = t % 12
            H_dead = self.data['dispatch_limits']['H_dead'][month_idx]
            H_max = self.data['dispatch_limits']['H_max'][month_idx]
            
            if t == 0:
                curr_V_prev = prev_V
            else:
                curr_V_prev = V[t - 1]
            
            curr_V = V[t]
            Qout = self.inflow[t] - (curr_V - curr_V_prev) * 1e8 / seconds_per_month
            
            if t == 0:
                avg_H = (H[-1] + H[t]) / 2
            else:
                avg_H = (H[t - 1] + H[t]) / 2
            
            if H[t] < H_dead or H[t] > H_max:
                penalty += self.pc_bounds * abs(H[t] - clamp(H[t], H_dead, H_max))
            
            if Qout < 0:
                penalty += self.pc_negQ * abs(Qout)
                continue
            
            if avg_H < height_of_spillway:
                maxQ = num_turbines * max_generation_flow
            else:
                maxQ = num_turbines * max_generation_flow + num_spillway * self.fitters.fit_spill_Q(avg_H)
            
            if Qout > maxQ:
                penalty += self.pc_spill * abs(Qout - maxQ)
                Q_power = num_turbines * max_generation_flow
            else:
                if Qout <= num_turbines * max_generation_flow:
                    Q_power = Qout
                else:
                    Q_power = num_turbines * max_generation_flow
            
            if Qout < Q_min:
                penalty += self.pc_minQ * abs(Q_min - Qout)
            
            Hdown = self.fitters.fit_H_downstream(Qout)
            q_unit = max(Q_power / num_turbines, 1e-10)
            head_loss = num_turbines * self.fitters.fit_dH_loss(q_unit)
            net_head = avg_H - Hdown - head_loss
            
            P_inst = num_turbines * self.fitters.fit_output_N(q_unit, net_head)
            
            if P_inst < guaranteed_output:
                penalty += self.pc_guarantee * abs(guaranteed_output - P_inst)
            
            E = P_inst * 30.4 * 24
            total_energy += E
        
        return -total_energy + penalty
    
    def _run_algorithm(self) -> np.ndarray:
        """
        运行PSO算法
        
        Returns
        -------
        np.ndarray
            最优水位轨迹（长度为n_stages+1）
        """
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level
        
        # 构建边界
        state_lb = np.tile(self.data['dispatch_limits']['H_dead'], 
                          self.dispatch_config.num_years)
        state_ub = np.tile(self.data['dispatch_limits']['H_max'], 
                          self.dispatch_config.num_years)
        
        state_lb[-1] = final_water_level
        state_ub[-1] = final_water_level
        
        print(f"  开始PSO优化: pop={self.population_size}, iter={self.max_iterations}, dim={self.n_stages}")
        
        optimizer = PSOOptimizer(
            n_particles=self.population_size,
            max_iterations=self.max_iterations,
            lb=state_lb,
            ub=state_ub,
            dim=self.n_stages,
            objective_func=self._fitness_function,
            random_seed=self.random_seed
        )
        
        best_fitness, best_H_seq, convergence_curve = optimizer.optimize()
        
        print(f"  PSO完成, best fitness = {best_fitness:.4e}")
        
        water_level_trajectory = np.zeros(self.n_stages + 1)
        water_level_trajectory[0] = initial_water_level
        water_level_trajectory[1:] = best_H_seq
        
        self._convergence_history = convergence_curve
        
        return water_level_trajectory
    
    def run(self):
        result = super().run()
        if hasattr(self, '_convergence_history'):
            result.convergence_history = self._convergence_history
        return result
