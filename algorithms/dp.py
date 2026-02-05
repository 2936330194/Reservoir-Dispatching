"""
动态规划调度算法
============

使用 DP 求解水库逐月最优调度。
"""
import numpy as np
from typing import Dict, Any

from .base import BaseDispatchAlgorithm
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig


class DPDispatch(BaseDispatchAlgorithm):
    """
    动态规划调度算法。

    将水位状态离散化后，采用逐阶段递推求最优轨迹。
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

        # DP 参数
        self.d_water_level = self.algorithm_config.d_water_level
        self.penalty_coefficient = self.algorithm_config.penalty_coefficient
        self.tolerance = 0

    def _generate_state_values(self, H_min: float, H_max: float) -> np.ndarray:
        """生成离散状态值。"""
        return np.arange(H_min, H_max + self.d_water_level, self.d_water_level)

    def _calculate_transition_energy_batch(
        self,
        H_prev: np.ndarray,
        H_curr: float,
        inflow: float
    ) -> np.ndarray:
        """
        批量计算 prev->curr 转移发电量，不可行转移返回 -inf。
        """
        H_prev = np.asarray(H_prev, dtype=float)

        num_turbines = self.reservoir_config.num_turbines
        max_generation_flow = self.reservoir_config.max_generation_flow
        num_spillway = self.reservoir_config.num_spillway
        height_of_spillway = self.reservoir_config.height_of_spillway
        seconds_per_month = self.reservoir_config.seconds_per_month
        Q_min = self.dispatch_config.Q_min
        guaranteed_output = self.dispatch_config.guaranteed_output

        V_start = self.fitters.fit_V(H_prev)
        V_end = self.fitters.fit_V(H_curr)
        Q_release = (V_start - V_end) * 1e8 / seconds_per_month + inflow

        Z_avg = 0.5 * (H_prev + H_curr)
        spill_Q = self.fitters.fit_spill_Q(Z_avg)
        Q_max = np.where(
            Z_avg < height_of_spillway,
            num_turbines * max_generation_flow,
            num_turbines * max_generation_flow + num_spillway * spill_Q
        )

        feasible = (Q_release >= Q_min) & (Q_release <= Q_max)
        energy = np.full(H_prev.shape, -np.inf, dtype=float)
        if not np.any(feasible):
            return energy

        Q_release_f = Q_release[feasible]
        H_tail = np.atleast_1d(self.fitters.fit_H_downstream(Q_release_f)).astype(float)
        Q_gen = np.minimum(Q_release_f, num_turbines * max_generation_flow)
        q_unit = Q_gen / num_turbines
        dH_loss = num_turbines * np.atleast_1d(self.fitters.fit_dH_loss(q_unit)).astype(float)
        H_net = Z_avg[feasible] - H_tail - dH_loss

        positive_head = H_net > 0
        if np.any(positive_head):
            q_pos = q_unit[positive_head]
            h_pos = H_net[positive_head]
            N = num_turbines * np.atleast_1d(self.fitters.fit_output_N(q_pos, h_pos)).astype(float)
            low_power = N < guaranteed_output
            N[low_power] = N[low_power] - self.penalty_coefficient * (guaranteed_output - N[low_power])
            N = np.maximum(N, 0.0)

            e_pos = N * 30.4 * 24
            e_full = np.zeros_like(Q_release_f)
            e_full[positive_head] = e_pos
            energy[feasible] = e_full
        else:
            energy[feasible] = 0.0

        return energy

    def _run_algorithm(self) -> np.ndarray:
        """
        运行 DP 算法。

        Returns
        -------
        np.ndarray
            最优水位轨迹（长度 n_stages + 1）。
        """
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level

        # 全局状态空间
        global_min_H = np.min(self.H_dead_full)
        global_max_H = np.max(self.H_max_full)
        state_values_global = self._generate_state_values(global_min_H, global_max_H)
        n_state = len(state_values_global)

        print(f"  状态空间大小: {n_state} 个离散水位")

        month_of_stage = np.mod(np.arange(self.n_stages), 12)
        month_h_dead = self.data['dispatch_limits']['H_dead'][month_of_stage]
        month_h_max = self.data['dispatch_limits']['H_max'][month_of_stage]
        feasible_mask = (state_values_global[None, :] >= month_h_dead[:, None]) & \
                        (state_values_global[None, :] <= month_h_max[:, None])
        feasible_indices = [np.flatnonzero(feasible_mask[t]) for t in range(self.n_stages)]

        # DP 表格
        cumulative_energy = np.full((self.n_stages, n_state), -np.inf)
        backtrack = np.zeros((self.n_stages, n_state), dtype=int)

        # 第 1 阶段
        print("  处理第 1 阶段...")
        stage0_indices = feasible_indices[0]
        if len(stage0_indices) > 0:
            stage0_states = state_values_global[stage0_indices]
            stage0_energy = self._calculate_transition_energy_batch(
                np.full(len(stage0_states), initial_water_level),
                stage0_states,
                self.inflow[0]
            )
            valid = stage0_energy >= 0
            if np.any(valid):
                valid_indices = stage0_indices[valid]
                cumulative_energy[0, valid_indices] = stage0_energy[valid]
                backtrack[0, valid_indices] = -1

        # 中间阶段 (2 to n_stages-1)
        for t in range(1, self.n_stages - 1):
            if (t + 1) % 100 == 0:
                print(f"  处理第 {t + 1} 阶段...")

            prev_candidates = feasible_indices[t - 1]
            curr_candidates = feasible_indices[t]
            if len(prev_candidates) == 0 or len(curr_candidates) == 0:
                continue

            reachable_mask = np.isfinite(cumulative_energy[t - 1, prev_candidates])
            if not np.any(reachable_mask):
                continue

            prev_indices = prev_candidates[reachable_mask]
            prev_states = state_values_global[prev_indices]
            prev_cumulative = cumulative_energy[t - 1, prev_indices]
            inflow_t = self.inflow[t]

            for j in curr_candidates:
                curr_H = state_values_global[j]
                energy = self._calculate_transition_energy_batch(prev_states, curr_H, inflow_t)
                feasible_transitions = np.isfinite(energy) & (energy >= 0)
                if not np.any(feasible_transitions):
                    continue

                total = prev_cumulative[feasible_transitions] + energy[feasible_transitions]
                best_local = np.argmax(total)
                cumulative_energy[t, j] = total[best_local]
                backtrack[t, j] = prev_indices[feasible_transitions][best_local]

        # 最后阶段：终水位固定
        print(f"  处理第 {self.n_stages} 阶段...")
        t = self.n_stages - 1
        best_cum = -np.inf
        best_k = 0

        prev_candidates = feasible_indices[t - 1]
        if len(prev_candidates) > 0:
            reachable_mask = np.isfinite(cumulative_energy[t - 1, prev_candidates])
            if np.any(reachable_mask):
                prev_indices = prev_candidates[reachable_mask]
                prev_states = state_values_global[prev_indices]
                prev_cumulative = cumulative_energy[t - 1, prev_indices]

                energy = self._calculate_transition_energy_batch(
                    prev_states,
                    final_water_level,
                    self.inflow[t]
                )
                feasible_transitions = np.isfinite(energy) & (energy >= 0)
                if np.any(feasible_transitions):
                    total = prev_cumulative[feasible_transitions] + energy[feasible_transitions]
                    best_local = np.argmax(total)
                    best_cum = total[best_local]
                    best_k = prev_indices[feasible_transitions][best_local]
                    cumulative_energy[t, best_k] = best_cum
                    backtrack[t, best_k] = best_k

        # 回溯最优路径
        optimal_trajectory = np.zeros(self.n_stages + 1)
        optimal_trajectory[0] = initial_water_level
        optimal_trajectory[-1] = final_water_level

        optimal_index = np.zeros(self.n_stages, dtype=int)
        optimal_index[-1] = best_k

        for t in range(self.n_stages - 2, -1, -1):
            optimal_index[t] = backtrack[t + 1, optimal_index[t + 1]]

        for t in range(self.n_stages - 1):
            optimal_trajectory[t + 1] = state_values_global[optimal_index[t]]

        print(f"  DP 完成，最优总发电量估计: {best_cum / 1000:.2f} GWh")

        return optimal_trajectory
