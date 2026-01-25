"""
粒子群优化算法 (Particle Swarm Optimization)
============================================

元启发式优化器核心实现
"""
import numpy as np
from typing import Callable, Tuple


class PSOOptimizer:
    """
    粒子群优化算法 (PSO)
    """
    
    def __init__(
        self,
        n_particles: int,
        max_iterations: int,
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
        objective_func: Callable[[np.ndarray], float],
        random_seed: int = None
    ):
        """
        初始化PSO优化器
        
        Parameters
        ----------
        n_particles : int
            粒子数量
        max_iterations : int
            最大迭代次数
        lb : np.ndarray
            下界 (dim,)
        ub : np.ndarray
            上界 (dim,)
        dim : int
            问题维度
        objective_func : Callable
            目标函数（最小化）
        random_seed : int, optional
            随机种子
        """
        self.N = n_particles
        self.max_iter = max_iterations
        self.lb = np.atleast_1d(lb)
        self.ub = np.atleast_1d(ub)
        self.dim = dim
        self.fobj = objective_func
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 扩展边界到完整维度
        if len(self.lb) == 1:
            self.lb = np.full(dim, self.lb[0])
        if len(self.ub) == 1:
            self.ub = np.full(dim, self.ub[0])
        
        # PSO参数
        self.Vmax = 6.0        # 最大速度
        self.wMax = 0.9        # 惯性权重最大值
        self.wMin = 0.6        # 惯性权重最小值
        self.c1 = 2.0          # 个体学习因子
        self.c2 = 2.0          # 社会学习因子
    
    def optimize(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        执行优化
        
        Returns
        -------
        tuple
            (best_fitness, best_position, convergence_curve)
        """
        # 初始化
        vel = np.zeros((self.N, self.dim))
        pos = np.zeros((self.N, self.dim))
        pBestScore = np.full(self.N, np.inf)
        pBest = np.zeros((self.N, self.dim))
        gBest = np.zeros(self.dim)
        gBestScore = np.inf
        convergence_curve = np.zeros(self.max_iter)
        
        # 初始化粒子位置和速度
        for i in range(self.N):
            for j in range(self.dim):
                pos[i, j] = np.random.rand() * (self.ub[j] - self.lb[j]) + self.lb[j]
                vel[i, j] = 0.3 * np.random.rand()
        
        # 主循环
        for l in range(self.max_iter):
            # 边界处理
            pos = np.clip(pos, self.lb, self.ub)
            
            # 评估适应度
            for i in range(self.N):
                fitness = self.fobj(pos[i, :])
                
                # 更新个体最优
                if fitness < pBestScore[i]:
                    pBestScore[i] = fitness
                    pBest[i, :] = pos[i, :].copy()
                
                # 更新全局最优
                if fitness < gBestScore:
                    gBestScore = fitness
                    gBest = pos[i, :].copy()
            
            # 更新惯性权重（线性递减）
            w = self.wMax - l * ((self.wMax - self.wMin) / self.max_iter)
            
            # 更新速度和位置
            for i in range(self.N):
                for j in range(self.dim):
                    # 速度更新
                    vel[i, j] = (w * vel[i, j] +
                                 self.c1 * np.random.rand() * (pBest[i, j] - pos[i, j]) +
                                 self.c2 * np.random.rand() * (gBest[j] - pos[i, j]))
                    
                    # 速度限制
                    vel[i, j] = np.clip(vel[i, j], -self.Vmax, self.Vmax)
                    
                    # 位置更新
                    pos[i, j] = pos[i, j] + vel[i, j]
            
            convergence_curve[l] = gBestScore
            
            # 进度显示
            if (l + 1) % 100 == 0:
                print(f"    PSO迭代 {l + 1}: Best fitness = {gBestScore:.4e}")
        
        return gBestScore, gBest, convergence_curve
