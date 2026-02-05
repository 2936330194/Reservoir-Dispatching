"""
Dhole Optimization Algorithm (DOA) core optimizer.
"""
import numpy as np
from typing import Callable, Tuple


class DholeOptimizer:
    """Population-based optimizer used by DholeDispatch."""

    def __init__(
        self,
        n_population: int,
        max_iterations: int,
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
        objective_func: Callable[[np.ndarray], float],
        random_seed: int = None,
    ):
        self.N = n_population
        self.T = max_iterations
        self.lb = np.atleast_1d(lb)
        self.ub = np.atleast_1d(ub)
        self.dim = dim
        self.fobj = objective_func

        if random_seed is not None:
            np.random.seed(random_seed)

        if len(self.lb) == 1:
            self.lb = np.full(dim, self.lb[0])
        if len(self.ub) == 1:
            self.ub = np.full(dim, self.ub[0])

    def _initialize_population(self) -> np.ndarray:
        """Initialize positions in [lb, ub]."""
        return np.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb

    def _p_obj(self, PWN: int) -> float:
        return ((1.0 / (1.0 + np.exp(-0.5 * (PWN - 25)))) ** 2) * np.random.rand()

    def _evaluate_population(self, X: np.ndarray) -> np.ndarray:
        """Evaluate each individual once."""
        fitness = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            fitness[i] = self.fobj(X[i, :])
        return fitness

    def optimize(self) -> Tuple[float, np.ndarray, np.ndarray]:
        convergence_curve = np.zeros(self.T)
        X = self._initialize_population()

        fitness_f = self._evaluate_population(X)
        best_idx = int(np.argmin(fitness_f))
        Best_fitness = fitness_f[best_idx]
        localBest_position = X[best_idx, :].copy()

        prey_global = localBest_position.copy()
        Xnew = np.zeros_like(X)

        for t in range(self.T):
            C = 1.0 - (t / self.T)
            PWN = int(np.random.rand() * 15 + 5)
            prey = (prey_global + localBest_position) / 2.0
            prey_local = localBest_position.copy()
            prey_local_fitness_abs = max(abs(self.fobj(prey_local)), 1e-10)

            for i in range(self.N):
                if np.random.rand() < 0.5:
                    if PWN < 10:
                        Xnew[i, :] = X[i, :] + C * np.random.rand() * (prey - X[i, :])
                    else:
                        # Sample z != i for each dimension without rejection loop.
                        z = np.random.randint(0, self.N - 1, size=self.dim)
                        z += (z >= i)
                        Xnew[i, :] = X[i, :] - X[z, np.arange(self.dim)] + prey
                else:
                    Q = 3.0 * np.random.rand() * fitness_f[i] / prey_local_fitness_abs
                    p_val = self._p_obj(PWN)

                    if Q > 2:
                        W_prey = np.exp(-1.0 / max(Q, 1e-10)) * prey_local
                        theta = 2.0 * np.pi * np.random.rand(self.dim)
                        Xnew[i, :] = X[i, :] + np.cos(theta) * W_prey * p_val - np.sin(theta) * W_prey * p_val
                    else:
                        Xnew[i, :] = (X[i, :] - prey_global) * p_val + p_val * np.random.rand(self.dim) * X[i, :]

            np.clip(Xnew, self.lb, self.ub, out=Xnew)

            # Evaluate each new individual once.
            new_fitness = self._evaluate_population(Xnew)

            # Update local best from newly generated population.
            new_best_idx = int(np.argmin(new_fitness))
            localBest_position = Xnew[new_best_idx, :].copy()

            # Greedy selection.
            improved_mask = new_fitness < fitness_f
            if np.any(improved_mask):
                fitness_f[improved_mask] = new_fitness[improved_mask]
                X[improved_mask, :] = Xnew[improved_mask, :]

            current_best_idx = int(np.argmin(fitness_f))
            if fitness_f[current_best_idx] < Best_fitness:
                Best_fitness = fitness_f[current_best_idx]
                prey_global = X[current_best_idx, :].copy()

            convergence_curve[t] = Best_fitness

            if (t + 1) % 50 == 0:
                print(f"    DOA迭代 {t + 1}: Best fitness = {Best_fitness:.4e}")

        return Best_fitness, prey_global, convergence_curve