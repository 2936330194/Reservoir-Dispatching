"""
豺狼优化算法 (Dhole Optimization Algorithm)
==========================================

元启发式优化器核心实现
"""
import numpy as np
from typing import Callable, Tuple


class DholeOptimizer:
    """
    豺狼优化算法 (DOA)
    
    一种模拟豺狗狩猎行为的群智能优化算法
    """
    
    def __init__(
        self,
        n_population: int,
        max_iterations: int,
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
        objective_func: Callable[[np.ndarray], float],
        random_seed: int = None
    ):
        """
        初始化豺狼优化器
        
        Parameters
        ----------
        n_population : int
            种群大小
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
        self.N = n_population
        self.T = max_iterations
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
    
    def _initialize_population(self) -> np.ndarray:
        """初始化种群位置"""
        positions = np.zeros((self.N, self.dim))
        for i in range(self.dim):
            positions[:, i] = np.random.rand(self.N) * (self.ub[i] - self.lb[i]) + self.lb[i]
        return positions
    
    def _p_obj(self, PWN: int) -> float:
        """计算位置更新参数（公式4）"""
        return ((1 / (1 + np.exp(-0.5 * (PWN - 25)))) ** 2) * np.random.rand()
    
    def optimize(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        执行优化
        
        Returns
        -------
        tuple
            (best_fitness, best_position, convergence_curve)
        """
        # 初始化
        convergence_curve = np.zeros(self.T)
        X = self._initialize_population()
        
        Best_fitness = np.inf
        fitness_f = np.zeros(self.N)
        localBest_position = None
        
        # 计算初始种群适应度
        for i in range(self.N):
            fitness_f[i] = self.fobj(X[i, :])
            if fitness_f[i] < Best_fitness:
                Best_fitness = fitness_f[i]
                localBest_position = X[i, :].copy()
        
        prey_global = localBest_position.copy()
        Xnew = np.zeros_like(X)
        
        # 主循环
        for t in range(self.T):
            C = 1 - (t / self.T)  # 线性递减参数
            PWN = int(np.random.rand() * 15 + 5)  # 随机豺狗数量参数
            prey = (prey_global + localBest_position) / 2  # 平均猎物位置
            prey_local = localBest_position.copy()
            
            # 遍历每个个体
            for i in range(self.N):
                if np.random.rand() < 0.5:  # 探索行为
                    if PWN < 10:
                        # 搜索阶段
                        Xnew[i, :] = X[i, :] + C * np.random.rand() * (prey - X[i, :])
                    else:
                        # 包围阶段
                        for j in range(self.dim):
                            z = np.random.randint(0, self.N)
                            while z == i:
                                z = np.random.randint(0, self.N)
                            Xnew[i, j] = X[i, j] - X[z, j] + prey[j]
                else:  # 开发行为
                    Q = 3 * np.random.rand() * fitness_f[i] / max(abs(self.fobj(prey_local)), 1e-10)
                    
                    if Q > 2:
                        W_prey = np.exp(-1 / max(Q, 1e-10)) * prey_local
                        p_val = self._p_obj(PWN)
                        for j in range(self.dim):
                            Xnew[i, j] = (X[i, j] + 
                                          np.cos(2 * np.pi * np.random.rand()) * W_prey[j] * p_val -
                                          np.sin(2 * np.pi * np.random.rand()) * W_prey[j] * p_val)
                    else:
                        p_val = self._p_obj(PWN)
                        Xnew[i, :] = ((X[i, :] - prey_global) * p_val + 
                                      p_val * np.random.rand(self.dim) * X[i, :])
            
            # 边界处理
            Xnew = np.clip(Xnew, self.lb, self.ub)
            
            # 更新最佳位置
            localBest_position = Xnew[0, :].copy()
            localBest_fitness = self.fobj(localBest_position)
            
            for i in range(self.N):
                local_fitness = self.fobj(Xnew[i, :])
                
                if local_fitness < localBest_fitness:
                    localBest_fitness = local_fitness
                    localBest_position = Xnew[i, :].copy()
                
                # 贪婪策略更新
                if local_fitness < fitness_f[i]:
                    fitness_f[i] = local_fitness
                    X[i, :] = Xnew[i, :].copy()
                    
                    if fitness_f[i] < Best_fitness:
                        Best_fitness = fitness_f[i]
                        prey_global = X[i, :].copy()
            
            convergence_curve[t] = Best_fitness
            
            # 进度显示
            if (t + 1) % 50 == 0:
                print(f"    DOA迭代 {t + 1}: Best fitness = {Best_fitness:.4e}")
        
        return Best_fitness, prey_global, convergence_curve
