"""
曲线拟合模块
============

包含水库调度所需的各种曲线拟合和插值函数
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, interp1d, NearestNDInterpolator
from scipy.optimize import curve_fit
from typing import Dict, Any, Callable


class CurveFitters:
    """
    曲线拟合器类
    
    封装所有水库调度所需的拟合/插值函数
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        初始化曲线拟合器
        
        Parameters
        ----------
        data : Dict[str, Any]
            load_reservoir_data()返回的数据字典
        """
        self.data = data
        self._build_fitters()
    
    def _build_fitters(self):
        """构建所有拟合器"""
        # 1. 水位-库容关系 (H -> V)
        H_levels = self.data['water_volume']['H_levels']
        V_values = self.data['water_volume']['V_values']
        self._pV_coeffs, self._pV_mean, self._pV_std = self._polyfit_normalized(H_levels, V_values, 3)
        
        # 2. 库容-水位关系 (V -> H) - 使用幂函数拟合
        self._fit_volume_to_height(V_values, H_levels)
        
        # 3. 下游流量-尾水位关系 (Q -> H_downstream)
        Q_downstream = self.data['tail_water']['Q_downstream']
        H_downstream = self.data['tail_water']['H_downstream']
        self._pHdown_coeffs, self._pHdown_mean, self._pHdown_std = self._polyfit_normalized(
            Q_downstream, H_downstream, 2)
        
        # 4. 流量-水头损失关系 (Q -> dH_loss)
        Q_loss = self.data['head_loss']['Q_loss']
        dH_loss = self.data['head_loss']['dH_loss']
        self._pdH_coeffs, self._pdH_mean, self._pdH_std = self._polyfit_normalized(Q_loss, dH_loss, 2)
        
        # 5. 出力特性曲面插值 (Q, H) -> N
        self._build_output_interpolator()
        
        # 6. 溢洪道最大流量插值 (H -> Q_max)
        upper_level_H = self.data['spillway']['upper_level_H']
        max_through_Q = self.data['spillway']['max_through_Q']
        self._spill_H = upper_level_H
        self._spill_Q = max_through_Q
        
        # 7. 闸门开度插值 (H, Q) -> Opening
        self._build_gate_interpolator()
    
    def _polyfit_normalized(self, x: np.ndarray, y: np.ndarray, degree: int):
        """
        标准化多项式拟合 (模拟MATLAB的polyfit带muV输出)
        
        Parameters
        ----------
        x : np.ndarray
            自变量
        y : np.ndarray
            因变量
        degree : int
            多项式阶数
        
        Returns
        -------
        tuple
            (coeffs, mean, std) 多项式系数和标准化参数
        """
        x_mean = np.mean(x)
        x_std = np.std(x)
        x_norm = (x - x_mean) / x_std
        coeffs = np.polyfit(x_norm, y, degree)
        return coeffs, x_mean, x_std
    
    def _polyval_normalized(self, coeffs: np.ndarray, x: np.ndarray, 
                           x_mean: float, x_std: float) -> np.ndarray:
        """
        使用标准化参数进行多项式求值
        """
        x_norm = (x - x_mean) / x_std
        return np.polyval(coeffs, x_norm)
    
    def _fit_volume_to_height(self, V: np.ndarray, H: np.ndarray):
        """
        拟合库容-水位幂函数关系: H = a * V^b + c
        """
        def power2(V, a, b, c):
            return a * np.power(V, b) + c
        
        # 初始参数 (来自MATLAB的StartPoint)
        p0 = [658.6769, 0.0356973, 0.1675196]
        
        try:
            popt, _ = curve_fit(power2, V, H, p0=p0, maxfev=10000)
            self._vh_params = popt
        except:
            # 如果拟合失败，使用备用的多项式拟合
            self._vh_params = None
            self._pVH_coeffs, self._pVH_mean, self._pVH_std = self._polyfit_normalized(V, H, 3)
    
    def _build_output_interpolator(self):
        """
        构建出力特性曲面插值器
        
        模拟MATLAB的griddedInterpolant('linear', 'nearest'):
        - 内部使用线性插值
        - 外推使用最近邻
        """
        through_Q = self.data['power_output']['through_Q']
        through_H = self.data['power_output']['through_H']
        output_N = self.data['power_output']['output_N']
        
        # 使用散点插值（内插使用线性）
        points = np.column_stack([through_Q, through_H])
        self._output_linear_interp = LinearNDInterpolator(points, output_N, fill_value=np.nan)
        
        # 外推使用最近邻
        self._output_nearest_interp = NearestNDInterpolator(points, output_N)
        
        # 存储数据范围用于边界处理
        self._output_Q_range = (through_Q.min(), through_Q.max())
        self._output_H_range = (through_H.min(), through_H.max())
        
        # 存储原始数据用于备用计算
        self._output_Q_data = through_Q
        self._output_H_data = through_H
        self._output_N_data = output_N
    
    def _build_gate_interpolator(self):
        """
        构建闸门开度插值器 (H, Q) -> Opening
        """
        level_spillway = self.data['spillway']['level_spillway']
        gate_opening = self.data['spillway']['gate_opening']
        spillway_flow_table = self.data['spillway']['spillway_flow_table']
        
        # 检查维度：flow_table应为 (L, M)，其中L=水位数，M=开度数
        L = len(level_spillway)
        M = spillway_flow_table.shape[1] if len(spillway_flow_table.shape) > 1 else 0
        
        # 如果gate_opening长度和M不匹配，截断或使用flow_table的列数
        if len(gate_opening) != M:
            # 假设开度从0到最大，线性分布
            gate_opening = np.linspace(0, gate_opening.max() if len(gate_opening) > 0 else 14.0, M)
        
        # 确保水位数匹配
        if len(level_spillway) > spillway_flow_table.shape[0]:
            level_spillway = level_spillway[:spillway_flow_table.shape[0]]
        elif len(level_spillway) < spillway_flow_table.shape[0]:
            spillway_flow_table = spillway_flow_table[:len(level_spillway), :]
        
        L = len(level_spillway)
        M = len(gate_opening)
        
        # 构建网格
        H_grid, O_grid = np.meshgrid(level_spillway, gate_opening, indexing='ij')
        Q_grid = spillway_flow_table[:, :M]  # 确保列数匹配
        
        # 构建 (H, Q) -> Opening 的散点插值
        H_samples = H_grid.flatten()
        Q_samples = Q_grid.flatten()
        O_samples = O_grid.flatten()
        
        # 检查维度
        min_len = min(len(H_samples), len(Q_samples), len(O_samples))
        H_samples = H_samples[:min_len]
        Q_samples = Q_samples[:min_len]
        O_samples = O_samples[:min_len]
        
        # 过滤掉NaN值
        valid_mask = ~(np.isnan(H_samples) | np.isnan(Q_samples) | np.isnan(O_samples))
        H_samples = H_samples[valid_mask]
        Q_samples = Q_samples[valid_mask]
        O_samples = O_samples[valid_mask]
        
        if len(H_samples) > 0:
            # 按 (H, Q) 去重
            data_combined = np.column_stack([H_samples, Q_samples, O_samples])
            _, unique_idx = np.unique(data_combined[:, :2], axis=0, return_index=True)
            H_unique = H_samples[unique_idx]
            Q_unique = Q_samples[unique_idx]
            O_unique = O_samples[unique_idx]
            
            points = np.column_stack([H_unique, Q_unique])
            self._gate_interp = LinearNDInterpolator(points, O_unique, fill_value=0.0)
        else:
            # 创建一个简单的返回0的插值器
            self._gate_interp = lambda x: np.zeros(len(x))
    
    # ==================== 公共接口 ====================
    
    def fit_V(self, H: np.ndarray) -> np.ndarray:
        """
        水位 -> 库容
        
        Parameters
        ----------
        H : np.ndarray or float
            水位 (m)
        
        Returns
        -------
        np.ndarray or float
            库容 (亿m³)
        """
        H = np.atleast_1d(H)
        V = self._polyval_normalized(self._pV_coeffs, H, self._pV_mean, self._pV_std)
        return V if len(V) > 1 else V[0]
    
    def fit_H(self, V: np.ndarray) -> np.ndarray:
        """
        库容 -> 水位
        
        Parameters
        ----------
        V : np.ndarray or float
            库容 (亿m³)
        
        Returns
        -------
        np.ndarray or float
            水位 (m)
        """
        V = np.atleast_1d(V)
        
        if self._vh_params is not None:
            a, b, c = self._vh_params
            H = a * np.power(V, b) + c
        else:
            H = self._polyval_normalized(self._pVH_coeffs, V, self._pVH_mean, self._pVH_std)
        
        return H if len(H) > 1 else H[0]
    
    def fit_H_downstream(self, Q: np.ndarray) -> np.ndarray:
        """
        下泄流量 -> 下游水位
        
        Parameters
        ----------
        Q : np.ndarray or float
            下泄流量 (m³/s)
        
        Returns
        -------
        np.ndarray or float
            下游水位 (m)
        """
        Q = np.atleast_1d(Q)
        H = self._polyval_normalized(self._pHdown_coeffs, Q, self._pHdown_mean, self._pHdown_std)
        return H if len(H) > 1 else H[0]
    
    def fit_dH_loss(self, Q: np.ndarray) -> np.ndarray:
        """
        流量 -> 水头损失
        
        Parameters
        ----------
        Q : np.ndarray or float
            流量 (m³/s)
        
        Returns
        -------
        np.ndarray or float
            水头损失 (m)
        """
        Q = np.atleast_1d(Q)
        dH = self._polyval_normalized(self._pdH_coeffs, Q, self._pdH_mean, self._pdH_std)
        return dH if len(dH) > 1 else dH[0]
    
    def fit_output_N(self, Q: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        (流量, 净水头) -> 单机出力
        
        模拟MATLAB的griddedInterpolant('linear', 'nearest'):
        - 内部点使用线性插值
        - 外部点使用最近邻外推
        
        Parameters
        ----------
        Q : np.ndarray or float
            单机流量 (m³/s)
        H : np.ndarray or float
            净水头 (m)
        
        Returns
        -------
        np.ndarray or float
            单机出力 (MW)
        """
        Q = np.atleast_1d(Q)
        H = np.atleast_1d(H)
        
        points = np.column_stack([Q, H])
        
        # 先尝试线性插值
        N = self._output_linear_interp(points)
        
        # 对于NaN值（外推点）使用最近邻
        nan_mask = np.isnan(N)
        if np.any(nan_mask):
            N[nan_mask] = self._output_nearest_interp(points[nan_mask])
        
        # 确保非负
        N = np.maximum(N, 0.0)
        
        return N if len(N) > 1 else N[0]
    
    def fit_spill_Q(self, H: np.ndarray) -> np.ndarray:
        """
        水位 -> 溢洪道单孔最大流量
        
        模拟MATLAB的griddedInterpolant('spline', 'linear'):
        - 内部使用样条插值
        - 外推使用线性外推
        
        Parameters
        ----------
        H : np.ndarray or float
            上游水位 (m)
        
        Returns
        -------
        np.ndarray or float
            溢洪道单孔最大流量 (m³/s)
        """
        H = np.atleast_1d(H)
        
        # 排序数据（scipy要求）
        sort_idx = np.argsort(self._spill_H)
        spill_H_sorted = self._spill_H[sort_idx]
        spill_Q_sorted = self._spill_Q[sort_idx]
        
        # 创建样条插值器，边界外使用线性外推
        # scipy的interp1d可以设置fill_value='extrapolate'来实现外推
        # 但为了更接近MATLAB的行为，我们手动实现线性外推
        
        Q_result = np.zeros_like(H, dtype=float)
        
        H_min = spill_H_sorted.min()
        H_max = spill_H_sorted.max()
        
        # 内部点：使用样条插值
        inside_mask = (H >= H_min) & (H <= H_max)
        if np.any(inside_mask):
            try:
                spline_interp = interp1d(spill_H_sorted, spill_Q_sorted, kind='cubic', 
                                          bounds_error=False, fill_value=np.nan)
                Q_result[inside_mask] = spline_interp(H[inside_mask])
            except:
                # 如果样条失败，使用线性插值
                Q_result[inside_mask] = np.interp(H[inside_mask], spill_H_sorted, spill_Q_sorted)
        
        # 外推点：使用线性外推
        below_mask = H < H_min
        above_mask = H > H_max
        
        if np.any(below_mask):
            # 使用前两个点的斜率外推
            if len(spill_H_sorted) >= 2:
                slope = (spill_Q_sorted[1] - spill_Q_sorted[0]) / (spill_H_sorted[1] - spill_H_sorted[0])
                Q_result[below_mask] = spill_Q_sorted[0] + slope * (H[below_mask] - H_min)
            else:
                Q_result[below_mask] = spill_Q_sorted[0]
        
        if np.any(above_mask):
            # 使用最后两个点的斜率外推
            if len(spill_H_sorted) >= 2:
                slope = (spill_Q_sorted[-1] - spill_Q_sorted[-2]) / (spill_H_sorted[-1] - spill_H_sorted[-2])
                Q_result[above_mask] = spill_Q_sorted[-1] + slope * (H[above_mask] - H_max)
            else:
                Q_result[above_mask] = spill_Q_sorted[-1]
        
        # 确保非负
        Q_result = np.maximum(Q_result, 0.0)
        
        return Q_result if len(Q_result) > 1 else Q_result[0]
    
    def fit_gate_opening(self, H: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        (水位, 单孔流量) -> 闸门开度
        
        Parameters
        ----------
        H : np.ndarray or float
            上游水位 (m)
        Q : np.ndarray or float
            溢洪道单孔流量 (m³/s)
        
        Returns
        -------
        np.ndarray or float
            闸门开度 (m)
        """
        H = np.atleast_1d(H)
        Q = np.atleast_1d(Q)
        
        points = np.column_stack([H, Q])
        opening = self._gate_interp(points)
        
        # 处理NaN值
        opening = np.nan_to_num(opening, nan=0.0)
        
        return opening if len(opening) > 1 else opening[0]
