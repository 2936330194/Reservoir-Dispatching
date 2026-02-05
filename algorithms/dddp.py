"""
Discrete Differential Dynamic Programming (DDDP)
"""
import numpy as np
from typing import Dict, Any, List, Tuple

from .base import BaseDispatchAlgorithm
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig


class DDDPDispatch(BaseDispatchAlgorithm):
    """DDDP: DP inside a shrinking corridor around a reference trajectory."""

    algorithm_name = "离散微分动态规划(DDDP)"

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

        self.max_iterations = self.algorithm_config.dddp_max_iterations
        self.tolerance = self.algorithm_config.dddp_tolerance
        self.corridor_width = self.algorithm_config.dddp_corridor_width
        self.d_water_level = self.algorithm_config.dddp_water_level
        self.penalty_coefficient = self.algorithm_config.dddp_penalty_coefficient

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
        """Vectorized stage-energy; infeasible transitions return -1e6."""
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

    def _build_state_corridors(self, reference_trajectory: np.ndarray) -> List[np.ndarray]:
        """Build corridor states for every stage (including endpoints)."""
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level

        H_max_monthly = self.data['dispatch_limits']['H_max']
        H_dead_monthly = self.data['dispatch_limits']['H_dead']

        state_corridors: List[np.ndarray] = []
        for t in range(self.n_stages + 1):
            if t == 0:
                state_corridors.append(np.array([initial_water_level], dtype=float))
            elif t == self.n_stages:
                state_corridors.append(np.array([final_water_level], dtype=float))
            else:
                month_idx = (t - 1) % 12
                z_ref = reference_trajectory[t]
                z_min = max(H_dead_monthly[month_idx], z_ref - self.corridor_width)
                z_max = min(H_max_monthly[month_idx], z_ref + self.corridor_width)
                states = np.arange(z_min, z_max + self.d_water_level, self.d_water_level)
                state_corridors.append(states)

        return state_corridors

    def _dddp_in_corridor(
        self,
        state_corridors: List[np.ndarray],
        inflow_seq: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Run DP inside corridor using vectorized transition evaluation."""
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level

        n_max_states = max(len(s) for s in state_corridors)
        cumulative_energy = np.full((self.n_stages + 1, n_max_states), -np.inf)
        backtrack_pointer = np.zeros((self.n_stages + 1, n_max_states), dtype=int)

        # Precompute stage state volumes to avoid repeated fit_V calls.
        state_volumes = [
            np.atleast_1d(self.fitters.fit_V(states)).astype(float)
            for states in state_corridors
        ]

        cumulative_energy[0, 0] = 0.0

        for t in range(self.n_stages):
            current_states = state_corridors[t]
            next_states = state_corridors[t + 1]
            n_current = len(current_states)
            n_next = len(next_states)

            if n_current == 0 or n_next == 0:
                continue

            prev_cum_all = cumulative_energy[t, :n_current]
            reachable_mask = np.isfinite(prev_cum_all)
            if not np.any(reachable_mask):
                continue

            prev_idx = np.flatnonzero(reachable_mask)
            prev_states = current_states[prev_idx]
            prev_volumes = state_volumes[t][prev_idx]
            prev_cum = prev_cum_all[prev_idx]

            next_volumes = state_volumes[t + 1]
            month_idx = t % 12

            energy_matrix = self._calculate_stage_energy_batch(
                prev_states[:, None],
                next_states[None, :],
                inflow_seq[t],
                month_idx,
                V_start=prev_volumes[:, None],
                V_end=next_volumes[None, :],
            )

            valid = energy_matrix >= 0
            if not np.any(valid):
                continue

            total_matrix = prev_cum[:, None] + energy_matrix
            total_matrix[~valid] = -np.inf

            best_prev_local = np.argmax(total_matrix, axis=0)
            best_vals = total_matrix[best_prev_local, np.arange(n_next)]
            feasible_next = np.isfinite(best_vals)
            if not np.any(feasible_next):
                continue

            cumulative_energy[t + 1, :n_next][feasible_next] = best_vals[feasible_next]
            backtrack_pointer[t + 1, :n_next][feasible_next] = prev_idx[best_prev_local[feasible_next]]

        optimal_trajectory = np.zeros(self.n_stages + 1)

        final_states = state_corridors[-1]
        final_idx = int(np.argmin(np.abs(final_states - final_water_level)))
        total_energy = cumulative_energy[-1, final_idx]

        optimal_trajectory[-1] = final_states[final_idx]
        for t in range(self.n_stages, 0, -1):
            prev_state_idx = backtrack_pointer[t, final_idx]
            prev_states = state_corridors[t - 1]
            optimal_trajectory[t - 1] = prev_states[prev_state_idx]
            final_idx = prev_state_idx

        optimal_trajectory[0] = initial_water_level
        optimal_trajectory[-1] = final_water_level

        return optimal_trajectory, total_energy

    def _run_algorithm(self) -> np.ndarray:
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level

        if self.initial_trajectory is None:
            reference_trajectory = np.linspace(initial_water_level, final_water_level, self.n_stages + 1)
        else:
            reference_trajectory = self.initial_trajectory.copy()

        total_energy_history = []
        convergence_history = []
        inflow_seq = self.inflow[:self.n_stages]

        print(f"  开始DDDP优化，最大迭代次数: {self.max_iterations}")

        for iter_num in range(self.max_iterations):
            state_corridors = self._build_state_corridors(reference_trajectory)
            new_trajectory, total_energy = self._dddp_in_corridor(state_corridors, inflow_seq)

            total_energy_history.append(total_energy)
            max_water_level_change = np.max(np.abs(new_trajectory[1:-1] - reference_trajectory[1:-1]))
            convergence_history.append(max_water_level_change)

            print(
                f"    迭代 {iter_num + 1}: 总发电量 = {total_energy / 1000:.2f} GWh, "
                f"最大水位变化 = {max_water_level_change:.4f} m"
            )

            if max_water_level_change < self.tolerance:
                print(f"  DDDP算法在第 {iter_num + 1} 次迭代后收敛")
                reference_trajectory = new_trajectory
                break

            reference_trajectory = new_trajectory

            if iter_num > 5 and convergence_history[-1] < 0.5 * convergence_history[-2]:
                self.corridor_width = max(self.d_water_level, self.corridor_width * 0.8)

        self._convergence_history = np.array(total_energy_history)
        return reference_trajectory

    def run(self):
        result = super().run()
        if hasattr(self, '_convergence_history'):
            result.convergence_history = self._convergence_history
        return result