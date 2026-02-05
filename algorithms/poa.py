"""
Progressive Optimality Algorithm (POA)
"""
import numpy as np
from typing import Dict, Any

from .base import BaseDispatchAlgorithm
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig


class POADispatch(BaseDispatchAlgorithm):
    """POA 逐步优化算法。"""

    algorithm_name = "POA逐步优化"

    def __init__(
        self,
        data: Dict[str, Any] = None,
        reservoir_config: ReservoirConfig = None,
        dispatch_config: DispatchConfig = None,
        algorithm_config: AlgorithmConfig = None,
        data_file: str = None,
        initial_trajectory: np.ndarray = None,
    ):
        super().__init__(data, reservoir_config, dispatch_config, algorithm_config, data_file)

        self.max_iterations = self.algorithm_config.poa_max_iterations
        self.tolerance = self.algorithm_config.poa_tolerance
        self.d_water_level = self.algorithm_config.poa_water_level
        self.penalty_coefficient = self.algorithm_config.penalty_coefficient
        self.initial_trajectory = initial_trajectory

    def set_initial_trajectory(self, trajectory: np.ndarray):
        self.initial_trajectory = trajectory

    def _calculate_stage_energy_batch(
        self,
        Z_start: np.ndarray,
        Z_end: np.ndarray,
        Q_in: np.ndarray,
        month_idx: int,
        V_start: np.ndarray = None,
        V_end: np.ndarray = None,
    ) -> np.ndarray:
        """批量计算单阶段发电量；不可行返回 -1e6。"""
        Z_start_arr, Z_end_arr = np.broadcast_arrays(
            np.asarray(Z_start, dtype=float),
            np.asarray(Z_end, dtype=float),
        )
        shape = Z_start_arr.shape
        Z_start_flat = Z_start_arr.ravel()
        Z_end_flat = Z_end_arr.ravel()
        Q_in_flat = np.broadcast_to(np.asarray(Q_in, dtype=float), shape).ravel()

        if V_start is None:
            V_start_flat = np.atleast_1d(self.fitters.fit_V(Z_start_flat)).astype(float)
        else:
            V_start_flat = np.broadcast_to(np.asarray(V_start, dtype=float), shape).ravel()

        if V_end is None:
            V_end_flat = np.atleast_1d(self.fitters.fit_V(Z_end_flat)).astype(float)
        else:
            V_end_flat = np.broadcast_to(np.asarray(V_end, dtype=float), shape).ravel()

        num_turbines = self.reservoir_config.num_turbines
        max_generation_flow = self.reservoir_config.max_generation_flow
        num_spillway = self.reservoir_config.num_spillway
        height_of_spillway = self.reservoir_config.height_of_spillway
        seconds_per_month = self.reservoir_config.seconds_per_month
        Q_min = self.dispatch_config.Q_min
        guaranteed_output = self.dispatch_config.guaranteed_output

        H_dead = self.data['dispatch_limits']['H_dead'][month_idx]
        H_max = self.data['dispatch_limits']['H_max'][month_idx]

        energy = np.full(Z_start_flat.shape, -1e6, dtype=float)

        bounds_ok = (
            (Z_start_flat >= H_dead)
            & (Z_start_flat <= H_max)
            & (Z_end_flat >= H_dead)
            & (Z_end_flat <= H_max)
        )
        if not np.any(bounds_ok):
            return energy.reshape(shape)

        start_b = Z_start_flat[bounds_ok]
        end_b = Z_end_flat[bounds_ok]
        q_in_b = Q_in_flat[bounds_ok]
        v_start_b = V_start_flat[bounds_ok]
        v_end_b = V_end_flat[bounds_ok]

        Q_release = (v_start_b - v_end_b) * 1e8 / seconds_per_month + q_in_b
        Z_avg = 0.5 * (start_b + end_b)

        spill_Q = np.atleast_1d(self.fitters.fit_spill_Q(Z_avg)).astype(float)
        Q_max = np.where(
            Z_avg < height_of_spillway,
            num_turbines * max_generation_flow,
            num_turbines * max_generation_flow + num_spillway * spill_Q,
        )

        hydraulic_ok = (Q_release >= Q_min) & (Q_release <= Q_max)
        if not np.any(hydraulic_ok):
            return energy.reshape(shape)

        q_rel_h = Q_release[hydraulic_ok]
        z_avg_h = Z_avg[hydraulic_ok]

        H_tail = np.atleast_1d(self.fitters.fit_H_downstream(q_rel_h)).astype(float)
        Q_gen = np.minimum(q_rel_h, num_turbines * max_generation_flow)
        q_unit = Q_gen / num_turbines
        dH_loss = num_turbines * np.atleast_1d(self.fitters.fit_dH_loss(q_unit)).astype(float)
        H_net = z_avg_h - H_tail - dH_loss

        local_energy = np.zeros_like(q_rel_h)
        positive_head = H_net > 0
        if np.any(positive_head):
            q_pos = q_unit[positive_head]
            h_pos = H_net[positive_head]
            N = num_turbines * np.atleast_1d(self.fitters.fit_output_N(q_pos, h_pos)).astype(float)
            low_power = N < guaranteed_output
            N[low_power] = N[low_power] - self.penalty_coefficient * (guaranteed_output - N[low_power])
            N = np.maximum(N, 0.0)
            local_energy[positive_head] = N * 30.4 * 24

        bounds_indices = np.flatnonzero(bounds_ok)
        feasible_indices = bounds_indices[hydraulic_ok]
        energy[feasible_indices] = local_energy

        return energy.reshape(shape)

    def _calculate_total_energy_for_trajectory(
        self,
        water_level_trajectory: np.ndarray,
        inflow_seq: np.ndarray,
    ) -> float:
        """批量计算当前轨迹总发电量，用于收敛判断。"""
        stage_month_idx = np.mod(np.arange(self.n_stages), 12)
        z_start = water_level_trajectory[:-1]
        z_end = water_level_trajectory[1:]
        total_energy = 0.0

        for month_idx in range(12):
            mask = stage_month_idx == month_idx
            if not np.any(mask):
                continue

            energy = self._calculate_stage_energy_batch(
                z_start[mask],
                z_end[mask],
                inflow_seq[mask],
                month_idx,
            )
            valid = energy > 0
            if np.any(valid):
                total_energy += float(np.sum(energy[valid]))

        return total_energy

    def _run_algorithm(self) -> np.ndarray:
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level

        if self.initial_trajectory is None:
            water_level_trajectory = np.linspace(
                initial_water_level,
                final_water_level,
                self.n_stages + 1,
            )
        else:
            water_level_trajectory = self.initial_trajectory.copy()

        water_level_trajectory[0] = initial_water_level
        water_level_trajectory[-1] = final_water_level

        total_energy_history = []
        inflow_seq = self.inflow[:self.n_stages]

        # 预生成候选状态与对应库容，避免迭代内重复开销
        monthly_candidates = []
        monthly_candidate_volume = []
        for month_idx in range(12):
            H_dead = self.data['dispatch_limits']['H_dead'][month_idx]
            H_max = self.data['dispatch_limits']['H_max'][month_idx]
            candidates = np.arange(H_dead, H_max + self.d_water_level, self.d_water_level)
            monthly_candidates.append(candidates)
            monthly_candidate_volume.append(
                np.atleast_1d(self.fitters.fit_V(candidates)).astype(float)
            )

        print(f"  开始POA优化，最大迭代次数: {self.max_iterations}")

        for iter_num in range(self.max_iterations):
            old_trajectory = water_level_trajectory.copy()

            # Update intermediate states stage-by-stage.
            for t in range(1, self.n_stages):
                prev_month_idx = (t - 1) % 12
                next_month_idx = t % 12

                Z_candidates = monthly_candidates[prev_month_idx]
                V_candidates = monthly_candidate_volume[prev_month_idx]

                energy1 = self._calculate_stage_energy_batch(
                    water_level_trajectory[t - 1],
                    Z_candidates,
                    inflow_seq[t - 1],
                    prev_month_idx,
                    V_end=V_candidates,
                )
                energy2 = self._calculate_stage_energy_batch(
                    Z_candidates,
                    water_level_trajectory[t + 1],
                    inflow_seq[t],
                    next_month_idx,
                    V_start=V_candidates,
                )

                valid = (energy1 >= 0) & (energy2 >= 0)
                if np.any(valid):
                    total_local_energy = energy1[valid] + energy2[valid]
                    best_idx = np.argmax(total_local_energy)
                    water_level_trajectory[t] = Z_candidates[valid][best_idx]

            water_level_trajectory[0] = initial_water_level
            water_level_trajectory[-1] = final_water_level

            total_energy = self._calculate_total_energy_for_trajectory(
                water_level_trajectory,
                inflow_seq,
            )
            total_energy_history.append(total_energy)

            max_change = np.max(np.abs(water_level_trajectory[1:-1] - old_trajectory[1:-1]))
            print(
                f"    迭代 {iter_num + 1}: 总发电量 = {total_energy / 1000:.2f} GWh, "
                f"最大水位变化 = {max_change:.4f} m"
            )

            if max_change < self.tolerance:
                print(f"  POA算法在第 {iter_num + 1} 次迭代后收敛")
                break

        self._convergence_history = np.array(total_energy_history)
        return water_level_trajectory

    def run(self):
        result = super().run()
        if hasattr(self, '_convergence_history'):
            result.convergence_history = self._convergence_history
        return result
