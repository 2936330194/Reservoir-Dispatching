"""
Particle Swarm Optimization dispatch algorithm.
"""
import numpy as np
from typing import Dict, Any

from .base import BaseDispatchAlgorithm
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
from optimizers.pso_optimizer import PSOOptimizer


class PSODispatch(BaseDispatchAlgorithm):
    """Use PSO to optimize the reservoir dispatch trajectory."""

    algorithm_name = "Particle Swarm Optimization (PSO)"

    def __init__(
        self,
        data: Dict[str, Any] = None,
        reservoir_config: ReservoirConfig = None,
        dispatch_config: DispatchConfig = None,
        algorithm_config: AlgorithmConfig = None,
        data_file: str = None,
    ):
        super().__init__(data, reservoir_config, dispatch_config, algorithm_config, data_file)

        self.population_size = self.algorithm_config.population_size
        self.max_iterations = self.algorithm_config.max_iterations
        self.random_seed = self.algorithm_config.random_seed

        self.pc_bounds = self.algorithm_config.penalty_bounds
        self.pc_spill = self.algorithm_config.penalty_spill
        self.pc_minQ = self.algorithm_config.penalty_min_Q
        self.pc_negQ = self.algorithm_config.penalty_neg_Q
        self.pc_guarantee = self.algorithm_config.penalty_guarantee
        self.tolerance = self.algorithm_config.tolerance

        # Cache frequently used arrays/constants for faster fitness calls.
        self._inflow_seq = self.inflow[:self.n_stages]
        self._month_idx = np.mod(np.arange(self.n_stages), 12)
        self._h_dead_seq = self.data['dispatch_limits']['H_dead'][self._month_idx]
        self._h_max_seq = self.data['dispatch_limits']['H_max'][self._month_idx]

        self._sec_per_month = self.reservoir_config.seconds_per_month
        self._num_turbines = self.reservoir_config.num_turbines
        self._num_spillway = self.reservoir_config.num_spillway
        self._height_of_spillway = self.reservoir_config.height_of_spillway
        self._q_turbine_max = self._num_turbines * self.reservoir_config.max_generation_flow
        self._q_min = self.dispatch_config.Q_min
        self._guaranteed_output = self.dispatch_config.guaranteed_output

    def _fitness_function(self, H_row: np.ndarray) -> float:
        """
        Minimize objective: fitness = -total_energy + penalties.

        H_row is shape (n_stages,). This implementation is vectorized.
        """
        H = np.asarray(H_row, dtype=float).reshape(-1)

        # Circular previous stage to match original logic at t=0.
        H_prev = np.roll(H, 1)
        V = np.atleast_1d(self.fitters.fit_V(H)).astype(float)
        V_prev = np.roll(V, 1)

        Qout = self._inflow_seq - (V - V_prev) * 1e8 / self._sec_per_month
        avg_H = 0.5 * (H_prev + H)

        penalty = 0.0

        # Water-level bound penalty.
        below = np.maximum(self._h_dead_seq - H, 0.0)
        above = np.maximum(H - self._h_max_seq, 0.0)
        penalty += self.pc_bounds * float(np.sum(below + above))

        # Negative outflow penalty; for these stages skip subsequent terms.
        neg_mask = Qout < 0.0
        if np.any(neg_mask):
            penalty += self.pc_negQ * float(np.sum(-Qout[neg_mask]))

        valid_mask = ~neg_mask
        if not np.any(valid_mask):
            return penalty

        qout_valid = Qout[valid_mask]
        avg_h_valid = avg_H[valid_mask]

        spill_q = np.atleast_1d(self.fitters.fit_spill_Q(avg_h_valid)).astype(float)
        maxQ = np.where(
            avg_h_valid < self._height_of_spillway,
            self._q_turbine_max,
            self._q_turbine_max + self._num_spillway * spill_q,
        )

        # Spill-capacity penalty.
        spill_excess = np.maximum(qout_valid - maxQ, 0.0)
        penalty += self.pc_spill * float(np.sum(spill_excess))

        # Min-release penalty.
        qmin_deficit = np.maximum(self._q_min - qout_valid, 0.0)
        penalty += self.pc_minQ * float(np.sum(qmin_deficit))

        q_power = np.minimum(qout_valid, self._q_turbine_max)
        q_unit = np.maximum(q_power / self._num_turbines, 1e-10)

        hdown = np.atleast_1d(self.fitters.fit_H_downstream(qout_valid)).astype(float)
        head_loss = self._num_turbines * np.atleast_1d(self.fitters.fit_dH_loss(q_unit)).astype(float)
        net_head = avg_h_valid - hdown - head_loss

        p_inst = self._num_turbines * np.atleast_1d(self.fitters.fit_output_N(q_unit, net_head)).astype(float)

        # Guaranteed-output penalty.
        guarantee_deficit = np.maximum(self._guaranteed_output - p_inst, 0.0)
        penalty += self.pc_guarantee * float(np.sum(guarantee_deficit))

        total_energy = float(np.sum(p_inst * 30.4 * 24))
        return -total_energy + penalty

    def _run_algorithm(self) -> np.ndarray:
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level

        state_lb = np.tile(self.data['dispatch_limits']['H_dead'], self.dispatch_config.num_years)
        state_ub = np.tile(self.data['dispatch_limits']['H_max'], self.dispatch_config.num_years)

        # Fix terminal water level.
        state_lb[-1] = final_water_level
        state_ub[-1] = final_water_level

        print(
            f"  Starting PSO optimization: pop={self.population_size}, "
            f"iter={self.max_iterations}, dim={self.n_stages}"
        )

        optimizer = PSOOptimizer(
            n_particles=self.population_size,
            max_iterations=self.max_iterations,
            lb=state_lb,
            ub=state_ub,
            dim=self.n_stages,
            objective_func=self._fitness_function,
            random_seed=self.random_seed,
        )

        best_fitness, best_H_seq, convergence_curve = optimizer.optimize()

        print(f"  PSO done, best fitness = {best_fitness:.4e}")

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