"""
Particle Swarm Optimization (PSO) core optimizer.
"""
import numpy as np
from typing import Callable, Tuple


class PSOOptimizer:
    """Standard PSO optimizer with vectorized state updates."""

    def __init__(
        self,
        n_particles: int,
        max_iterations: int,
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
        objective_func: Callable[[np.ndarray], float],
        random_seed: int = None,
    ):
        self.N = n_particles
        self.max_iter = max_iterations
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

        self.Vmax = 6.0
        self.wMax = 0.9
        self.wMin = 0.6
        self.c1 = 2.0
        self.c2 = 2.0

    def _evaluate_population(self, pos: np.ndarray) -> np.ndarray:
        fitness = np.empty(self.N, dtype=float)
        for i in range(self.N):
            fitness[i] = self.fobj(pos[i, :])
        return fitness

    def optimize(self) -> Tuple[float, np.ndarray, np.ndarray]:
        vel = 0.3 * np.random.rand(self.N, self.dim)
        pos = np.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb

        pBestScore = np.full(self.N, np.inf)
        pBest = np.zeros((self.N, self.dim))
        gBest = np.zeros(self.dim)
        gBestScore = np.inf
        convergence_curve = np.zeros(self.max_iter)

        for l in range(self.max_iter):
            np.clip(pos, self.lb, self.ub, out=pos)

            fitness = self._evaluate_population(pos)

            improved = fitness < pBestScore
            if np.any(improved):
                pBestScore[improved] = fitness[improved]
                pBest[improved, :] = pos[improved, :]

            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < gBestScore:
                gBestScore = fitness[best_idx]
                gBest = pos[best_idx, :].copy()

            w = self.wMax - l * ((self.wMax - self.wMin) / self.max_iter)

            r1 = np.random.rand(self.N, self.dim)
            r2 = np.random.rand(self.N, self.dim)
            vel = (
                w * vel
                + self.c1 * r1 * (pBest - pos)
                + self.c2 * r2 * (gBest[None, :] - pos)
            )
            np.clip(vel, -self.Vmax, self.Vmax, out=vel)
            pos += vel

            convergence_curve[l] = gBestScore

            if (l + 1) % 50 == 0:
                print(f"    PSO iter {l + 1}: Best fitness = {gBestScore:.4e}")

        return gBestScore, gBest, convergence_curve