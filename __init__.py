"""
水库调度算法库 - Reservoir Dispatching Algorithms
================================================

该库包含多种水库中长期调度优化算法的Python实现，包括：
- 常规调度 (Conventional Dispatching)
- POA逐步优化算法 (Progressive Optimality Algorithm)
- DP动态规划 (Dynamic Programming)
- DDDP离散微分动态规划 (Discrete Differential Dynamic Programming)
- 豺狼优化算法 (Dhole Optimization Algorithm)
- 粒子群优化算法 (Particle Swarm Optimization)
- XGBoost调度函数 (XGBoost-based Dispatching Function)

Author: Converted from MATLAB by Gemini
"""

__version__ = "1.0.0"
__author__ = "Hu Hao"

from .core.data_loader import load_reservoir_data
from .core.curve_fitting import CurveFitters
from .core.simulation import simulate_from_water_level_seq
