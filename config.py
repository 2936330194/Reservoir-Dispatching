"""
水库调度配置参数
================

包含水库物理参数、调度参数和算法参数
"""
import os
from dataclasses import dataclass, field
from typing import List
import numpy as np

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# 数据文件路径
DATA_FILE = os.path.join(DATA_DIR, 'data_ty.xls')


@dataclass
class ReservoirConfig: 
    """水库物理参数配置"""
    # 机组参数
    num_turbines: int = 4                    # 机组数量
    max_generation_flow: float = 301.2       # 单机最大发电流量 (m³/s)
    installed_output: float = 1200.0         # 装机容量 (MW)
    
    # 溢洪道参数
    num_spillway: int = 5                    # 溢洪道数量
    height_of_spillway: float = 760.0        # 溢洪道底坎高程 (m)
    
    # 时间参数
    seconds_per_month: float = 30.4 * 24 * 3600  # 每月秒数


@dataclass
class DispatchConfig:
    """调度参数配置"""
    # 调度期参数
    num_years: int = 73                      # 调度年数
    num_months: int = 12                     # 每年月数
    
    # 水位约束
    initial_water_level: float = 764.2       # 初始水位 (m)
    final_water_level: float = 764.2         # 终止水位 (m)
    
    # 出力约束
    guaranteed_output: float = 405.2         # 保证出力 (MW)
    
    # 流量约束
    Q_min: float = 321.0                     # 最小下泄流量 (m³/s)
    
    @property
    def n_stages(self) -> int:
        """总调度时段数"""
        return self.num_years * self.num_months


@dataclass
class AlgorithmConfig:
    """算法参数配置"""
    # DP/DDDP参数
    d_water_level: float = 0.5               # 水位离散步长 (m)
    penalty_coefficient: float = 60.0        # 保证出力惩罚系数
    
    # POA参数
    poa_max_iterations: int = 100            # POA最大迭代次数
    poa_tolerance: float = 0.01              # POA收敛容差 (m)
    poa_water_level: float = 0.5             # 水位离散步长 (m)
    
    # DDDP参数
    dddp_max_iterations: int = 50            # DDDP最大迭代次数
    dddp_tolerance: float = 0.01             # DDDP收敛容差 (m)
    dddp_corridor_width: float = 5.0         # DDDP初始走廊宽度 (m)
    
    # 元启发式算法参数
    population_size: int = 100               # 种群大小
    max_iterations: int = 3500               # 最大迭代次数
    random_seed: int = 42                     # 随机种子（保证复现性）
    
    # 惩罚系数（元启发式算法）
    penalty_bounds: float = 1e4              # 水位越界惩罚
    penalty_spill: float = 1e3               # 溢洪超能力惩罚
    penalty_min_Q: float = 1e3               # 最小下泄流量惩罚
    penalty_neg_Q: float = 5e3               # 负流量惩罚
    penalty_guarantee: float = 1e3           # 保证出力惩罚

    # 保证率容差
    tolerance: float = 1.0                   # 保证率容差（mW）


@dataclass
class XGBoostConfig:
    """XGBoost参数配置"""
    max_iter: int = 75                       # 最大迭代次数
    booster: str = 'gbtree'                  # 基学习器类型
    objective: str = 'reg:squarederror'      # 目标函数(Python版本用squarederror)
    max_depth: int = 5                       # 树最大深度
    learning_rate: float = 0.01               # 学习率
    min_child_weight: int = 1                # 最小叶子权重
    subsample: float = 0.95                  # 采样比例
    colsample_bytree: float = 1.0            # 特征采样比例
    train_test_split: tuple = (0.7, 0.3)     # 训练集:测试集比例
    random_seed: int = 42                     # 随机种子


# 默认配置实例
DEFAULT_RESERVOIR = ReservoirConfig()
DEFAULT_DISPATCH = DispatchConfig()
DEFAULT_ALGORITHM = AlgorithmConfig()
DEFAULT_XGBOOST = XGBoostConfig()
