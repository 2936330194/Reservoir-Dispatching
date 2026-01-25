"""
XGBoost调度函数算法
===================

使用XGBoost拟合水位、入流与出流之间的关系
"""
import numpy as np
from typing import Dict, Any, Tuple

from .base import BaseDispatchAlgorithm
from config import (
    ReservoirConfig, DispatchConfig, AlgorithmConfig, XGBoostConfig,
    DEFAULT_XGBOOST
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("警告: xgboost未安装，XGBoostDispatch将不可用")


class XGBoostDispatch(BaseDispatchAlgorithm):
    """
    XGBoost调度函数算法
    
    训练XGBoost回归模型拟合最优调度决策
    """
    
    algorithm_name = "XGBoost调度函数"
    
    def __init__(
        self,
        data: Dict[str, Any] = None,
        reservoir_config: ReservoirConfig = None,
        dispatch_config: DispatchConfig = None,
        algorithm_config: AlgorithmConfig = None,
        data_file: str = None,
        xgboost_config: XGBoostConfig = None,
        training_data: Dict[str, np.ndarray] = None
    ):
        """
        初始化XGBoost调度算法
        
        Parameters
        ----------
        training_data : Dict[str, np.ndarray], optional
            训练数据，包含:
            - 'inflow': 入库流量
            - 'water_level': 水位
            - 'outflow': 出库流量（标签）
            如果不提供，需要先设置训练数据
        """
        super().__init__(data, reservoir_config, dispatch_config, algorithm_config, data_file)
        
        if not HAS_XGBOOST:
            raise ImportError("xgboost未安装，请运行: pip install xgboost")
        
        self.xgb_config = xgboost_config or DEFAULT_XGBOOST
        self.training_data = training_data
        self.model = None
        
        # 标准化参数
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
    
    def set_training_data(
        self,
        inflow: np.ndarray,
        water_level: np.ndarray,
        outflow: np.ndarray
    ):
        """
        设置训练数据
        
        Parameters
        ----------
        inflow : np.ndarray
            入库流量序列
        water_level : np.ndarray
            水位序列（长度比inflow多1）
        outflow : np.ndarray
            出库流量序列（标签）
        """
        # 使用时段初水位
        X = np.column_stack([inflow, water_level[:-1]])
        y = outflow
        
        self.training_data = {
            'X': X,
            'y': y
        }
    
    def train_model(self) -> Dict[str, float]:
        """
        训练XGBoost模型
        
        Returns
        -------
        Dict[str, float]
            训练指标（RMSE等）
        """
        if self.training_data is None:
            raise ValueError("请先设置训练数据")
        
        X = self.training_data['X']
        y = self.training_data['y']
        
        # 设置随机种子
        np.random.seed(self.xgb_config.random_seed)
        
        # 数据打乱
        m = len(y)
        indices = np.random.permutation(m)
        
        # 划分训练集和测试集
        train_ratio = self.xgb_config.train_test_split[0]
        train_num = int(train_ratio / sum(self.xgb_config.train_test_split) * m)
        
        train_idx = indices[:train_num]
        test_idx = indices[train_num:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Z-score标准化
        self.x_mean = np.mean(X_train, axis=0)
        self.x_std = np.std(X_train, axis=0)
        self.x_std[self.x_std == 0] = 1  # 避免除零
        
        self.y_mean = np.mean(y_train)
        self.y_std = np.std(y_train)
        if self.y_std == 0:
            self.y_std = 1
        
        X_train_norm = (X_train - self.x_mean) / self.x_std
        y_train_norm = (y_train - self.y_mean) / self.y_std
        X_test_norm = (X_test - self.x_mean) / self.x_std
        
        # 训练XGBoost
        print("  训练XGBoost回归模型...")
        
        dtrain = xgb.DMatrix(X_train_norm, label=y_train_norm)
        dtest = xgb.DMatrix(X_test_norm)
        
        params = {
            'booster': self.xgb_config.booster,
            'objective': self.xgb_config.objective,
            'max_depth': self.xgb_config.max_depth,
            'eta': self.xgb_config.learning_rate,
            'min_child_weight': self.xgb_config.min_child_weight,
            'subsample': self.xgb_config.subsample,
            'colsample_bytree': self.xgb_config.colsample_bytree,
        }
        
        self.model = xgb.train(params, dtrain, num_boost_round=self.xgb_config.max_iter)
        
        # 测试集预测
        y_pred_norm = self.model.predict(dtest)
        y_pred = y_pred_norm * self.y_std + self.y_mean
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        print(f"  测试集RMSE: {rmse:.4f} m³/s")
        
        return {'test_rmse': rmse}
    
    def predict_outflow(self, inflow: float, water_level: float) -> float:
        """
        使用训练好的模型预测出库流量
        
        Parameters
        ----------
        inflow : float
            入库流量 (m³/s)
        water_level : float
            当前水位 (m)
        
        Returns
        -------
        float
            建议出库流量 (m³/s)
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train_model()")
        
        X = np.array([[inflow, water_level]])
        X_norm = (X - self.x_mean) / self.x_std
        
        dtest = xgb.DMatrix(X_norm)
        y_pred_norm = self.model.predict(dtest)
        y_pred = y_pred_norm * self.y_std + self.y_mean
        
        return float(y_pred[0])
    
    def _run_algorithm(self) -> np.ndarray:
        """
        运行XGBoost调度
        
        Returns
        -------
        np.ndarray
            水位轨迹（长度为n_stages+1）
        """
        if self.model is None:
            if self.training_data is None:
                raise ValueError("请先设置训练数据并训练模型")
            self.train_model()
        
        initial_water_level = self.dispatch_config.initial_water_level
        Q_min = self.dispatch_config.Q_min
        seconds_per_month = self.reservoir_config.seconds_per_month
        num_turbines = self.reservoir_config.num_turbines
        max_generation_flow = self.reservoir_config.max_generation_flow
        num_spillway = self.reservoir_config.num_spillway
        height_of_spillway = self.reservoir_config.height_of_spillway
        
        # 初始化
        water_level = np.zeros(self.n_stages + 1)
        water_level[0] = initial_water_level
        
        print("  开始实时调度模拟...")
        
        for t in range(self.n_stages):
            month_idx = t % 12
            H_max = self.data['dispatch_limits']['H_max'][month_idx]
            H_dead = self.data['dispatch_limits']['H_dead'][month_idx]
            
            current_inflow = self.inflow[t]
            current_water_level = water_level[t]
            current_storage = self.fitters.fit_V(current_water_level)
            
            # 使用XGBoost预测出库流量
            suggested_outflow = self.predict_outflow(current_inflow, current_water_level)
            
            # 约束1: 不低于最小下泄流量
            suggested_outflow = max(suggested_outflow, Q_min)
            
            # 约束2: 确保水位不超上限
            max_storage = self.fitters.fit_V(H_max)
            min_required_outflow = current_inflow - (max_storage - current_storage) * 1e8 / seconds_per_month
            suggested_outflow = max(suggested_outflow, min_required_outflow)
            
            # 约束3: 确保水位不低于下限
            min_storage = self.fitters.fit_V(H_dead)
            max_allowed_outflow = current_inflow - (min_storage - current_storage) * 1e8 / seconds_per_month
            suggested_outflow = min(suggested_outflow, max_allowed_outflow)
            
            # 计算末水位
            end_storage = current_storage + (current_inflow - suggested_outflow) * seconds_per_month / 1e8
            end_water_level = self.fitters.fit_H(end_storage)
            
            # 计算最大下泄能力
            avg_water_level = (current_water_level + end_water_level) / 2
            if avg_water_level < height_of_spillway:
                max_limit_Q = num_turbines * max_generation_flow
            else:
                max_limit_Q = num_turbines * max_generation_flow + num_spillway * self.fitters.fit_spill_Q(avg_water_level)
            
            # 约束4: 不超过最大下泄能力
            suggested_outflow = min(suggested_outflow, max_limit_Q)
            
            # 更新末水位
            end_storage = current_storage + (current_inflow - suggested_outflow) * seconds_per_month / 1e8
            end_water_level = self.fitters.fit_H(end_storage)
            
            water_level[t + 1] = end_water_level
            
            if (t + 1) % 100 == 0:
                print(f"    已完成 {t + 1} 个时段调度")
        
        return water_level
