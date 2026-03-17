# -*- coding: utf-8 -*-
"""集成多种算法的水库优化调度软件 V1.0"""
import os
from dataclasses import dataclass, field
from typing import List
import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'data_ty.xls')
@dataclass
class ReservoirConfig:
    """水库物理参数配置"""
    num_turbines: int = 4
    max_generation_flow: float = 301.2
    installed_output: float = 1200.0
    num_spillway: int = 5
    height_of_spillway: float = 760.0
    seconds_per_month: float = 30.4 * 24 * 3600
@dataclass
class DispatchConfig:
    """调度参数配置"""
    num_years: int = 73
    num_months: int = 12
    initial_water_level: float = 764.2
    final_water_level: float = 764.2
    guaranteed_output: float = 405.2
    Q_min: float = 321.0
    @property
    def n_stages(self) -> int:
        """总调度时段数"""
        return self.num_years * self.num_months
@dataclass
class AlgorithmConfig:
    """算法参数配置"""
    d_water_level: float = 0.5
    penalty_coefficient: float = 60.0
    poa_max_iterations: int = 100
    poa_tolerance: float = 0.01
    poa_water_level: float = 0.1
    dddp_max_iterations: int = 50
    dddp_tolerance: float = 0.01
    dddp_corridor_width: float = 5.0
    dddp_water_level: float = 0.1
    dddp_penalty_coefficient: float = 60.0
    population_size: int = 50
    max_iterations: int = 3500
    random_seed: int = 42
    penalty_bounds: float = 1e4
    penalty_spill: float = 1e3
    penalty_min_Q: float = 1e3
    penalty_neg_Q: float = 5e3
    penalty_guarantee: float = 1e3
    tolerance: float = 1.0
@dataclass
class XGBoostConfig:
    """XGBoost参数配置"""
    max_iter: int = 75
    booster: str = 'gbtree'
    objective: str = 'reg:squarederror'
    max_depth: int = 5
    learning_rate: float = 0.01
    min_child_weight: int = 1
    subsample: float = 0.95
    colsample_bytree: float = 1.0
    train_test_split: tuple = (0.7, 0.3)
    random_seed: int = 42
DEFAULT_RESERVOIR = ReservoirConfig()
DEFAULT_DISPATCH = DispatchConfig()
DEFAULT_ALGORITHM = AlgorithmConfig()
DEFAULT_XGBOOST = XGBoostConfig()
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
from config import DATA_FILE
def load_reservoir_data(data_file: str = None) -> Dict[str, Any]:
    """
    加载水库调度所需的全部数据
    """
    if data_file is None:
        data_file = DATA_FILE
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    data = {}
    df_wv = pd.read_excel(data_file, sheet_name='水位库容关系', header=None, skiprows=1)
    df_wv = df_wv.dropna(subset=[1, 2])
    data['water_volume'] = {
        'H_levels': pd.to_numeric(df_wv.iloc[:, 1], errors='coerce').dropna().values,
        'V_values': pd.to_numeric(df_wv.iloc[:, 2], errors='coerce').dropna().values,
    }
    df_tw = pd.read_excel(data_file, sheet_name='尾水流量关系', header=None, skiprows=1)
    df_tw = df_tw.dropna(subset=[1, 2])
    data['tail_water'] = {
        'Q_downstream': pd.to_numeric(df_tw.iloc[:, 2], errors='coerce').dropna().values,
        'H_downstream': pd.to_numeric(df_tw.iloc[:, 1], errors='coerce').dropna().values,
    }
    df_sp = pd.read_excel(data_file, sheet_name='泄流特性曲线', header=None)
    water_level_col = pd.to_numeric(df_sp.iloc[:, 0], errors='coerce')
    valid_rows = ~np.isnan(water_level_col)
    data_start_row = np.where(valid_rows)[0][0] if np.any(valid_rows) else 1
    gate_row = df_sp.iloc[0, 1:]
    gate_opening = pd.to_numeric(gate_row, errors='coerce').dropna().values
    if len(gate_opening) == 0:
        gate_row = df_sp.iloc[1, 1:]
        gate_opening = pd.to_numeric(gate_row, errors='coerce').dropna().values
    spillway_H = pd.to_numeric(df_sp.iloc[data_start_row:, 0], errors='coerce').dropna().values
    max_Q = pd.to_numeric(df_sp.iloc[data_start_row:, -1], errors='coerce').dropna().values
    flow_table_df = df_sp.iloc[data_start_row:, 1:-1]
    flow_table = flow_table_df.apply(pd.to_numeric, errors='coerce').values
    valid_flow_rows = ~np.all(np.isnan(flow_table), axis=1)
    flow_table = flow_table[valid_flow_rows]
    spillway_H = spillway_H[:len(flow_table)]
    max_Q = max_Q[:len(flow_table)]
    data['spillway'] = {
        'upper_level_H': spillway_H,
        'max_through_Q': max_Q,
        'level_spillway': spillway_H,
        'gate_opening': gate_opening,
        'spillway_flow_table': flow_table,
    }
    df_po = pd.read_excel(data_file, sheet_name='出力特性曲线', header=None, skiprows=1)
    df_po = df_po.dropna(subset=[1, 2, 3])
    data['power_output'] = {
        'through_H': pd.to_numeric(df_po.iloc[:, 1], errors='coerce').dropna().values,
        'through_Q': pd.to_numeric(df_po.iloc[:, 3], errors='coerce').dropna().values,
        'output_N': pd.to_numeric(df_po.iloc[:, 2], errors='coerce').dropna().values,
    }
    df_hl = pd.read_excel(data_file, sheet_name='水头损失曲线', header=None, skiprows=1)
    df_hl = df_hl.dropna(subset=[1, 2])
    data['head_loss'] = {
        'Q_loss': pd.to_numeric(df_hl.iloc[:, 1], errors='coerce').dropna().values,
        'dH_loss': pd.to_numeric(df_hl.iloc[:, 2], errors='coerce').dropna().values,
    }
    df_dl = pd.read_excel(data_file, sheet_name='天一调度图控制水位', header=None)
    df_dl_data = df_dl.iloc[:, 10:]
    numeric_rows = []
    for idx in range(len(df_dl_data)):
        row = df_dl_data.iloc[idx]
        try:
            float(row.iloc[-2])
            float(row.iloc[-1])
            numeric_rows.append(idx)
        except (ValueError, TypeError):
            continue
    if len(numeric_rows) >= 12:
        df_dl_clean = df_dl_data.iloc[numeric_rows[:12]]
        data['dispatch_limits'] = {
            'H_max': pd.to_numeric(df_dl_clean.iloc[:, -2], errors='coerce').values,
            'H_dead': pd.to_numeric(df_dl_clean.iloc[:, -1], errors='coerce').values,
            'H_upper': pd.to_numeric(df_dl_clean.iloc[:, 2], errors='coerce').values,
            'H_lower': pd.to_numeric(df_dl_clean.iloc[:, 3], errors='coerce').values,
        }
    else:
        data['dispatch_limits'] = {
            'H_max': np.array([790.0] * 6 + [776.4] * 4 + [790.0] * 2),
            'H_dead': np.array([735.0] * 12),
            'H_upper': np.array([764.2] * 12),
            'H_lower': np.array([750.0] * 12),
        }
    df_if = pd.read_excel(data_file, sheet_name='历史径流', header=None, skiprows=2)
    df_if_data = df_if.iloc[:, 6:]
    numeric_inflow_rows = []
    for idx in range(len(df_if_data)):
        row = df_if_data.iloc[idx]
        try:
            val = float(row.iloc[0])
            if not np.isnan(val):
                numeric_inflow_rows.append(idx)
        except (ValueError, TypeError):
            continue
    if len(numeric_inflow_rows) >= 2:
        df_if_clean = df_if_data.iloc[numeric_inflow_rows]
        inflow_matrix = df_if_clean.apply(pd.to_numeric, errors='coerce').values
        valid_rows = ~np.all(np.isnan(inflow_matrix), axis=1)
        data['inflow'] = {
            'Q': inflow_matrix[valid_rows],
        }
    else:
        raise ValueError("无法解析历史径流数据")
    return data
def get_inflow_sequence(data: Dict[str, Any]) -> np.ndarray:
    """
    将入库流量矩阵转换为时序序列
    """
    Q = data['inflow']['Q']
    Q_flatten = Q.flatten('C')
    Q_flatten_clean = Q_flatten[~np.isnan(Q_flatten)]
    return Q_flatten_clean
def get_monthly_limits(data: Dict[str, Any], num_years: int) -> tuple:
    """
    获取全部调度期的月度水位限制
    """
    H_max = data['dispatch_limits']['H_max']
    H_dead = data['dispatch_limits']['H_dead']
    H_max_full = np.tile(H_max, num_years)
    H_dead_full = np.tile(H_dead, num_years)
    return H_max_full, H_dead_full
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
        """
        self.data = data
        self._build_fitters()
    def _build_fitters(self):
        """构建所有拟合器"""
        H_levels = self.data['water_volume']['H_levels']
        V_values = self.data['water_volume']['V_values']
        self._pV_coeffs, self._pV_mean, self._pV_std = self._polyfit_normalized(H_levels, V_values, 3)
        self._fit_volume_to_height(V_values, H_levels)
        Q_downstream = self.data['tail_water']['Q_downstream']
        H_downstream = self.data['tail_water']['H_downstream']
        self._pHdown_coeffs, self._pHdown_mean, self._pHdown_std = self._polyfit_normalized(
            Q_downstream, H_downstream, 2)
        Q_loss = self.data['head_loss']['Q_loss']
        dH_loss = self.data['head_loss']['dH_loss']
        self._pdH_coeffs, self._pdH_mean, self._pdH_std = self._polyfit_normalized(Q_loss, dH_loss, 2)
        self._build_output_interpolator()
        upper_level_H = self.data['spillway']['upper_level_H']
        max_through_Q = self.data['spillway']['max_through_Q']
        self._spill_H = upper_level_H
        self._spill_Q = max_through_Q
        self._build_gate_interpolator()
    def _polyfit_normalized(self, x: np.ndarray, y: np.ndarray, degree: int):
        """
        标准化多项式拟合
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
        p0 = [658.6769, 0.0356973, 0.1675196]
        try:
            popt, _ = curve_fit(power2, V, H, p0=p0, maxfev=10000)
            self._vh_params = popt
        except:
            self._vh_params = None
            self._pVH_coeffs, self._pVH_mean, self._pVH_std = self._polyfit_normalized(V, H, 3)
    def _build_output_interpolator(self):
        """
        构建出力特性曲面插值器
        """
        through_Q = self.data['power_output']['through_Q']
        through_H = self.data['power_output']['through_H']
        output_N = self.data['power_output']['output_N']
        points = np.column_stack([through_Q, through_H])
        self._output_linear_interp = LinearNDInterpolator(points, output_N, fill_value=np.nan)
        self._output_nearest_interp = NearestNDInterpolator(points, output_N)
        self._output_Q_range = (through_Q.min(), through_Q.max())
        self._output_H_range = (through_H.min(), through_H.max())
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
        L = len(level_spillway)
        M = spillway_flow_table.shape[1] if len(spillway_flow_table.shape) > 1 else 0
        if len(gate_opening) != M:
            gate_opening = np.linspace(0, gate_opening.max() if len(gate_opening) > 0 else 14.0, M)
        if len(level_spillway) > spillway_flow_table.shape[0]:
            level_spillway = level_spillway[:spillway_flow_table.shape[0]]
        elif len(level_spillway) < spillway_flow_table.shape[0]:
            spillway_flow_table = spillway_flow_table[:len(level_spillway), :]
        L = len(level_spillway)
        M = len(gate_opening)
        H_grid, O_grid = np.meshgrid(level_spillway, gate_opening, indexing='ij')
        Q_grid = spillway_flow_table[:, :M]
        H_samples = H_grid.flatten()
        Q_samples = Q_grid.flatten()
        O_samples = O_grid.flatten()
        min_len = min(len(H_samples), len(Q_samples), len(O_samples))
        H_samples = H_samples[:min_len]
        Q_samples = Q_samples[:min_len]
        O_samples = O_samples[:min_len]
        valid_mask = ~(np.isnan(H_samples) | np.isnan(Q_samples) | np.isnan(O_samples))
        H_samples = H_samples[valid_mask]
        Q_samples = Q_samples[valid_mask]
        O_samples = O_samples[valid_mask]
        if len(H_samples) > 0:
            data_combined = np.column_stack([H_samples, Q_samples, O_samples])
            _, unique_idx = np.unique(data_combined[:, :2], axis=0, return_index=True)
            H_unique = H_samples[unique_idx]
            Q_unique = Q_samples[unique_idx]
            O_unique = O_samples[unique_idx]
            points = np.column_stack([H_unique, Q_unique])
            self._gate_interp = LinearNDInterpolator(points, O_unique, fill_value=0.0)
        else:
            self._gate_interp = lambda x: np.zeros(len(x))
    def fit_V(self, H: np.ndarray) -> np.ndarray:
        """
        水位 -> 库容
        """
        H = np.atleast_1d(H)
        V = self._polyval_normalized(self._pV_coeffs, H, self._pV_mean, self._pV_std)
        return V if len(V) > 1 else V[0]
    def fit_H(self, V: np.ndarray) -> np.ndarray:
        """
        库容 -> 水位
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
        """
        Q = np.atleast_1d(Q)
        H = self._polyval_normalized(self._pHdown_coeffs, Q, self._pHdown_mean, self._pHdown_std)
        return H if len(H) > 1 else H[0]
    def fit_dH_loss(self, Q: np.ndarray) -> np.ndarray:
        """
        流量 -> 水头损失
        """
        Q = np.atleast_1d(Q)
        dH = self._polyval_normalized(self._pdH_coeffs, Q, self._pdH_mean, self._pdH_std)
        return dH if len(dH) > 1 else dH[0]
    def fit_output_N(self, Q: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        (流量, 净水头) -> 单机出力
        """
        Q = np.atleast_1d(Q)
        H = np.atleast_1d(H)
        points = np.column_stack([Q, H])
        N = self._output_linear_interp(points)
        nan_mask = np.isnan(N)
        if np.any(nan_mask):
            N[nan_mask] = self._output_nearest_interp(points[nan_mask])
        N = np.maximum(N, 0.0)
        return N if len(N) > 1 else N[0]
    def fit_spill_Q(self, H: np.ndarray) -> np.ndarray:
        """
        水位 -> 溢洪道单孔最大流量
        """
        H = np.atleast_1d(H)
        sort_idx = np.argsort(self._spill_H)
        spill_H_sorted = self._spill_H[sort_idx]
        spill_Q_sorted = self._spill_Q[sort_idx]
        Q_result = np.zeros_like(H, dtype=float)
        H_min = spill_H_sorted.min()
        H_max = spill_H_sorted.max()
        inside_mask = (H >= H_min) & (H <= H_max)
        if np.any(inside_mask):
            try:
                spline_interp = interp1d(spill_H_sorted, spill_Q_sorted, kind='cubic',
                                          bounds_error=False, fill_value=np.nan)
                Q_result[inside_mask] = spline_interp(H[inside_mask])
            except:
                Q_result[inside_mask] = np.interp(H[inside_mask], spill_H_sorted, spill_Q_sorted)
        below_mask = H < H_min
        above_mask = H > H_max
        if np.any(below_mask):
            if len(spill_H_sorted) >= 2:
                slope = (spill_Q_sorted[1] - spill_Q_sorted[0]) / (spill_H_sorted[1] - spill_H_sorted[0])
                Q_result[below_mask] = spill_Q_sorted[0] + slope * (H[below_mask] - H_min)
            else:
                Q_result[below_mask] = spill_Q_sorted[0]
        if np.any(above_mask):
            if len(spill_H_sorted) >= 2:
                slope = (spill_Q_sorted[-1] - spill_Q_sorted[-2]) / (spill_H_sorted[-1] - spill_H_sorted[-2])
                Q_result[above_mask] = spill_Q_sorted[-1] + slope * (H[above_mask] - H_max)
            else:
                Q_result[above_mask] = spill_Q_sorted[-1]
        Q_result = np.maximum(Q_result, 0.0)
        return Q_result if len(Q_result) > 1 else Q_result[0]
    def fit_gate_opening(self, H: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        (水位, 单孔流量) -> 闸门开度
        """
        H = np.atleast_1d(H)
        Q = np.atleast_1d(Q)
        points = np.column_stack([H, Q])
        opening = self._gate_interp(points)
        opening = np.nan_to_num(opening, nan=0.0)
        return opening if len(opening) > 1 else opening[0]
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from config import ReservoirConfig, DispatchConfig, DEFAULT_RESERVOIR, DEFAULT_DISPATCH
@dataclass
class SimulationResult:
    """仿真结果数据类"""
    total_energy: float
    water_level: np.ndarray
    storage: np.ndarray
    outflow: np.ndarray
    generated_Q: np.ndarray
    abandoned_Q: np.ndarray
    power: np.ndarray
    energy: np.ndarray
    max_limit_Q: np.ndarray
    gate_opening: np.ndarray
def simulate_from_water_level_seq(
    H_seq: np.ndarray,
    inflow: np.ndarray,
    fitters: CurveFitters,
    reservoir_config: ReservoirConfig = None,
    dispatch_config: DispatchConfig = None,
    Q_min: float = None
) -> SimulationResult:
    """
    根据水位序列进行调度仿真
    """
    if reservoir_config is None:
        reservoir_config = DEFAULT_RESERVOIR
    if dispatch_config is None:
        dispatch_config = DEFAULT_DISPATCH
    if Q_min is None:
        Q_min = dispatch_config.Q_min
    num_turbines = reservoir_config.num_turbines
    max_generation_flow = reservoir_config.max_generation_flow
    num_spillway = reservoir_config.num_spillway
    height_of_spillway = reservoir_config.height_of_spillway
    seconds_per_month = reservoir_config.seconds_per_month
    n_stages = len(H_seq) - 1
    water_level = np.array(H_seq, dtype=float)
    storage = np.zeros(n_stages + 1)
    outflow = np.zeros(n_stages)
    generated_Q = np.zeros(n_stages)
    abandoned_Q = np.zeros(n_stages)
    power = np.zeros(n_stages)
    energy = np.zeros(n_stages)
    max_limit_Q = np.zeros(n_stages)
    gate_opening = np.zeros(n_stages)
    storage[0] = fitters.fit_V(H_seq[0])
    for t in range(n_stages):
        H_start = H_seq[t]
        H_end = H_seq[t + 1]
        V_start = fitters.fit_V(H_start)
        V_end = fitters.fit_V(H_end)
        avg_H = 0.5 * (H_start + H_end)
        Qout = inflow[t] - (V_end - V_start) * 1e8 / seconds_per_month
        if Qout < 0:
            Qout = 0
            print(f"警告: 第{t+1}个时段优化结果不可取（负出流）！")
        if Qout < Q_min:
            print(f"警告: 第{t+1}个时段出流{Qout:.2f}小于最小下泄流量{Q_min}！")
        if avg_H < height_of_spillway:
            maxQ = num_turbines * max_generation_flow
        else:
            maxQ = num_turbines * max_generation_flow + num_spillway * fitters.fit_spill_Q(avg_H)
        if Qout <= num_turbines * max_generation_flow:
            downstream_H = fitters.fit_H_downstream(Qout)
            generated = Qout
            abandoned = 0
            opening = 0
            head_loss = num_turbines * fitters.fit_dH_loss(generated / num_turbines)
            net_head = avg_H - downstream_H - head_loss
            actual_power = num_turbines * fitters.fit_output_N(
                max(generated / num_turbines, 1e-6), net_head)
        elif Qout <= maxQ:
            downstream_H = fitters.fit_H_downstream(Qout)
            generated = num_turbines * max_generation_flow
            abandoned = Qout - generated
            opening = fitters.fit_gate_opening(avg_H, abandoned / num_spillway)
            head_loss = num_turbines * fitters.fit_dH_loss(generated / num_turbines)
            net_head = avg_H - downstream_H - head_loss
            actual_power = num_turbines * fitters.fit_output_N(
                generated / num_turbines, net_head)
        else:
            print(f"警告: 第{t+1}个时段出流{Qout:.2f}大于最大下泄能力{maxQ:.2f}！")
            downstream_H = fitters.fit_H_downstream(maxQ)
            generated = num_turbines * max_generation_flow
            abandoned = maxQ - generated
            opening = fitters.fit_gate_opening(avg_H, abandoned / num_spillway)
            head_loss = num_turbines * fitters.fit_dH_loss(generated / num_turbines)
            net_head = avg_H - downstream_H - head_loss
            actual_power = num_turbines * fitters.fit_output_N(
                generated / num_turbines, net_head)
        if actual_power < 0:
            actual_power = 0
        E = actual_power * 30.4 * 24
        storage[t + 1] = V_end
        outflow[t] = Qout
        generated_Q[t] = generated
        abandoned_Q[t] = abandoned
        power[t] = actual_power
        energy[t] = E
        max_limit_Q[t] = maxQ
        gate_opening[t] = opening
    total_energy = np.sum(energy)
    return SimulationResult(
        total_energy=total_energy,
        water_level=water_level,
        storage=storage,
        outflow=outflow,
        generated_Q=generated_Q,
        abandoned_Q=abandoned_Q,
        power=power,
        energy=energy,
        max_limit_Q=max_limit_Q,
        gate_opening=gate_opening
    )
def calculate_stage_energy(
    Z_start: float,
    Z_end: float,
    Q_in: float,
    fitters: CurveFitters,
    reservoir_config: ReservoirConfig = None,
    dispatch_config: DispatchConfig = None,
    penalty_coefficient: float = 60.0,
    H_dead: float = None,
    H_max: float = None
) -> float:
    """
    计算单个调度阶段的发电量
    """
    if reservoir_config is None:
        reservoir_config = DEFAULT_RESERVOIR
    if dispatch_config is None:
        dispatch_config = DEFAULT_DISPATCH
    num_turbines = reservoir_config.num_turbines
    max_generation_flow = reservoir_config.max_generation_flow
    num_spillway = reservoir_config.num_spillway
    height_of_spillway = reservoir_config.height_of_spillway
    seconds_per_month = reservoir_config.seconds_per_month
    Q_min = dispatch_config.Q_min
    guaranteed_output = dispatch_config.guaranteed_output
    if H_dead is not None and H_max is not None:
        if Z_start < H_dead or Z_start > H_max or Z_end < H_dead or Z_end > H_max:
            return -1e6
    V_start = fitters.fit_V(Z_start)
    V_end = fitters.fit_V(Z_end)
    Q_release = (V_start - V_end) * 1e8 / seconds_per_month + Q_in
    if Q_release < Q_min:
        return -1e6
    Z_avg = (Z_start + Z_end) / 2
    if Z_avg < height_of_spillway:
        Q_max = num_turbines * max_generation_flow
    else:
        Q_max = num_turbines * max_generation_flow + num_spillway * fitters.fit_spill_Q(Z_avg)
    if Q_release > Q_max:
        return -1e6
    H_tail = fitters.fit_H_downstream(Q_release)
    if Q_release <= num_turbines * max_generation_flow:
        Q_gen = Q_release
    else:
        Q_gen = num_turbines * max_generation_flow
    q_unit = Q_gen / num_turbines
    dH_loss = num_turbines * fitters.fit_dH_loss(q_unit)
    H_net = Z_avg - H_tail - dH_loss
    if H_net <= 0:
        return 0
    N = num_turbines * fitters.fit_output_N(q_unit, H_net)
    if N < guaranteed_output:
        N = N - penalty_coefficient * (guaranteed_output - N)
    if N < 0:
        N = 0
    energy = N * 30.4 * 24
    return energy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import time
@dataclass
class DispatchResult:
    """调度算法结果"""
    algorithm_name: str
    water_level_trajectory: np.ndarray
    simulation_result: SimulationResult
    convergence_history: np.ndarray = None
    elapsed_time: float = 0.0
    extra_info: Dict[str, Any] = None
class BaseDispatchAlgorithm(ABC):
    """
    调度算法基类
    定义所有调度算法的通用接口和基础功能
    """
    algorithm_name: str = "Base"
    def __init__(
        self,
        data: Dict[str, Any] = None,
        reservoir_config: ReservoirConfig = None,
        dispatch_config: DispatchConfig = None,
        algorithm_config: AlgorithmConfig = None,
        data_file: str = None
    ):
        """
        初始化调度算法
        """
        if data is None:
            self.data = load_reservoir_data(data_file)
        else:
            self.data = data
        self.reservoir_config = reservoir_config or DEFAULT_RESERVOIR
        self.dispatch_config = dispatch_config or DEFAULT_DISPATCH
        self.algorithm_config = algorithm_config or DEFAULT_ALGORITHM
        self.fitters = CurveFitters(self.data)
        self.inflow = get_inflow_sequence(self.data)
        self.H_max_full, self.H_dead_full = get_monthly_limits(
            self.data, self.dispatch_config.num_years)
        self.n_stages = self.dispatch_config.n_stages
        self.result: Optional[DispatchResult] = None
    @abstractmethod
    def _run_algorithm(self) -> np.ndarray:
        """
        运行具体的优化算法
        """
        pass
    def run(self) -> DispatchResult:
        """
        运行调度算法
        """
        print(f"开始运行 {self.algorithm_name} 算法...")
        start_time = time.time()
        water_level_trajectory = self._run_algorithm()
        simulation_result = simulate_from_water_level_seq(
            water_level_trajectory,
            self.inflow,
            self.fitters,
            self.reservoir_config,
            self.dispatch_config
        )
        elapsed_time = time.time() - start_time
        self.result = DispatchResult(
            algorithm_name=self.algorithm_name,
            water_level_trajectory=water_level_trajectory,
            simulation_result=simulation_result,
            elapsed_time=elapsed_time
        )
        self._print_statistics()
        return self.result
    def _print_statistics(self):
        """打印调度结果统计信息"""
        if self.result is None:
            return
        sim = self.result.simulation_result
        print(f"\n=== {self.algorithm_name} 调度结果统计 ===")
        print(f"总发电量 = {sim.total_energy / 1000:.2f} GWh")
        num_years = self.dispatch_config.num_years
        annual_energy = np.zeros(num_years)
        for y in range(num_years):
            idx_start = y * 12
            idx_end = (y + 1) * 12
            annual_energy[y] = np.sum(sim.energy[idx_start:idx_end])
        avg_annual_energy = np.mean(annual_energy)
        print(f"多年平均年发电量 = {avg_annual_energy / 1000:.2f} GWh")
        guaranteed_output = self.dispatch_config.guaranteed_output
        num_lower = np.sum(sim.power < guaranteed_output - self.tolerance)
        guarantee_rate = 1 - num_lower / self.n_stages
        print(f"保证率 = {guarantee_rate * 100:.2f} %")
        seconds_per_month = self.reservoir_config.seconds_per_month
        total_abandoned = np.sum(sim.abandoned_Q) * seconds_per_month / 1e8
        print(f"总弃水量 = {total_abandoned:.2f} 亿立方米")
        water_use_rate = 100 * np.sum(sim.generated_Q) / np.sum(self.inflow)
        print(f"水量利用率 = {water_use_rate:.2f} %")
        print(f"运行时间 = {self.result.elapsed_time:.2f} 秒")
    def get_annual_statistics(self) -> Dict[str, np.ndarray]:
        """
        获取年度统计数据
        """
        if self.result is None:
            raise ValueError("请先运行算法")
        sim = self.result.simulation_result
        num_years = self.dispatch_config.num_years
        seconds_per_month = self.reservoir_config.seconds_per_month
        annual_energy = np.zeros(num_years)
        annual_abandoned = np.zeros(num_years)
        annual_inflow = np.zeros(num_years)
        annual_outflow = np.zeros(num_years)
        annual_avg_water_level = np.zeros(num_years)
        for y in range(num_years):
            idx_start = y * 12
            idx_end = (y + 1) * 12
            annual_energy[y] = np.sum(sim.energy[idx_start:idx_end])
            annual_abandoned[y] = np.sum(sim.abandoned_Q[idx_start:idx_end]) * seconds_per_month / 1e8
            annual_inflow[y] = np.sum(self.inflow[idx_start:idx_end]) * seconds_per_month / 1e8
            annual_outflow[y] = np.sum(sim.outflow[idx_start:idx_end]) * seconds_per_month / 1e8
            annual_avg_water_level[y] = np.mean(sim.water_level[idx_start+1:idx_end+1])
        return {
            'annual_energy': annual_energy,
            'annual_abandoned': annual_abandoned,
            'annual_inflow': annual_inflow,
            'annual_outflow': annual_outflow,
            'annual_avg_water_level': annual_avg_water_level
        }
import numpy as np
from typing import Dict, Any
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
class ConventionalDispatch(BaseDispatchAlgorithm):
    """
    常规调度算法
    根据调度图分区确定计划出力，迭代求解满足出力的下泄流量
    """
    algorithm_name = "常规调度"
    def __init__(
        self,
        data: Dict[str, Any] = None,
        reservoir_config: ReservoirConfig = None,
        dispatch_config: DispatchConfig = None,
        algorithm_config: AlgorithmConfig = None,
        data_file: str = None
    ):
        super().__init__(data, reservoir_config, dispatch_config, algorithm_config, data_file)
        self.H_upper = self.data['dispatch_limits']['H_upper']
        self.H_lower = self.data['dispatch_limits']['H_lower']
        self.tolerance = 1.0
        self.max_iterations = 100
    def _determine_planned_power(self, water_level: float, month: int) -> float:
        """
        根据调度图确定计划出力
        """
        guaranteed_output = self.dispatch_config.guaranteed_output
        H_max = self.data['dispatch_limits']['H_max'][month]
        H_upper = self.H_upper[month]
        H_lower = self.H_lower[month]
        H_dead = self.data['dispatch_limits']['H_dead'][month]
        if water_level >= H_max:
            return 1.5 * guaranteed_output
        elif water_level >= H_upper:
            return 1.2 * guaranteed_output
        elif water_level >= H_lower:
            return 1.0 * guaranteed_output
        elif water_level >= H_dead:
            return 0.75 * guaranteed_output
        else:
            return 0
    def _run_algorithm(self) -> np.ndarray:
        """
        运行常规调度算法
        """
        num_years = self.dispatch_config.num_years
        num_months = self.dispatch_config.num_months
        initial_water_level = self.dispatch_config.initial_water_level
        guaranteed_output = self.dispatch_config.guaranteed_output
        Q_min = self.dispatch_config.Q_min
        num_turbines = self.reservoir_config.num_turbines
        max_generation_flow = self.reservoir_config.max_generation_flow
        num_spillway = self.reservoir_config.num_spillway
        height_of_spillway = self.reservoir_config.height_of_spillway
        seconds_per_month = self.reservoir_config.seconds_per_month
        average_output_coefficient = 8.6
        water_level_2d = np.zeros((num_years, num_months + 1))
        storage_2d = np.zeros((num_years, num_months + 1))
        H_max = self.data['dispatch_limits']['H_max']
        H_dead = self.data['dispatch_limits']['H_dead']
        water_level_2d[0, 0] = initial_water_level
        storage_2d[0, 0] = self.fitters.fit_V(initial_water_level)
        for year in range(num_years):
            if (year + 1) % 10 == 0:
                print(f"  处理第 {year + 1} 年...")
            for month in range(num_months):
                current_month_idx = month
                current_water_level = water_level_2d[year, month]
                current_storage = storage_2d[year, month]
                t = year * num_months + month
                current_inflow = self.inflow[t]
                planned_power = self._determine_planned_power(current_water_level, month)
                if current_water_level > 0:
                    Q_out_guess = planned_power * 1e3 / (average_output_coefficient * current_water_level)
                else:
                    Q_out_guess = Q_min
                iteration_count = 0
                converged = False
                end_water_level = current_water_level
                end_storage = current_storage
                actual_power = 0
                Q_power = 0
                Q_abandoned = 0
                open_level = 0
                max_limit_Q = num_turbines * max_generation_flow
                while iteration_count < self.max_iterations and not converged:
                    iteration_count += 1
                    end_storage = current_storage + (current_inflow - Q_out_guess) * seconds_per_month / 1e8
                    if end_storage < 0:
                        end_storage = self.data['water_volume']['V_values'][0]
                    end_water_level = self.fitters.fit_H(end_storage)
                    if end_water_level > H_max[current_month_idx]:
                        end_water_level = H_max[current_month_idx]
                        end_storage = self.fitters.fit_V(end_water_level)
                        Q_out_guess = current_inflow - (end_storage - current_storage) * 1e8 / seconds_per_month
                        avg_water_level = (current_water_level + end_water_level) / 2
                        downstream_water_level = self.fitters.fit_H_downstream(Q_out_guess)
                        head_loss = num_turbines * self.fitters.fit_dH_loss(Q_out_guess / num_turbines)
                        net_head = avg_water_level - downstream_water_level - head_loss
                        if avg_water_level < height_of_spillway:
                            max_limit_Q = num_turbines * max_generation_flow
                        else:
                            max_limit_Q = num_turbines * max_generation_flow + num_spillway * self.fitters.fit_spill_Q(avg_water_level)
                        if Q_out_guess > num_turbines * max_generation_flow:
                            actual_power = num_turbines * self.fitters.fit_output_N(max_generation_flow, net_head)
                            Q_power = num_turbines * max_generation_flow
                            Q_abandoned = Q_out_guess - Q_power
                            if avg_water_level <= height_of_spillway:
                                if current_water_level < height_of_spillway and end_water_level < height_of_spillway:
                                    print(f"无法弃水！")
                                    open_level = np.nan
                                elif current_water_level < end_water_level:
                                    Q_abandoned_new = 2 * Q_abandoned / ((end_water_level - height_of_spillway) / (end_water_level - avg_water_level))
                                    avg_water_level_new = (height_of_spillway + end_water_level) / 2
                                    open_level = self.fitters.fit_gate_opening(avg_water_level_new, Q_abandoned_new / num_spillway)
                                else:
                                    Q_abandoned_new = 2 * Q_abandoned / ((current_water_level - height_of_spillway) / (current_water_level - avg_water_level))
                                    avg_water_level_new = (height_of_spillway + current_water_level) / 2
                                    open_level = self.fitters.fit_gate_opening(avg_water_level_new, Q_abandoned_new / num_spillway)
                            else:
                                open_level = self.fitters.fit_gate_opening(avg_water_level, Q_abandoned / num_spillway)
                            break
                        else:
                            actual_power = num_turbines * self.fitters.fit_output_N(Q_out_guess / num_turbines, net_head)
                            if actual_power >= planned_power:
                                Q_power = Q_out_guess
                                Q_abandoned = 0
                                open_level = 0
                                break
                            else:
                                if abs(actual_power - planned_power) < self.tolerance:
                                    converged = True
                                else:
                                    if abs(actual_power) > 1e-6:
                                        Q_out_guess = Q_out_guess * (planned_power / abs(actual_power))
                                    continue
                        continue
                    if end_water_level < H_dead[current_month_idx]:
                        end_water_level = H_dead[current_month_idx]
                        end_storage = self.fitters.fit_V(end_water_level)
                        Q_out_guess = current_inflow - (end_storage - current_storage) * 1e8 / seconds_per_month
                        avg_water_level = (current_water_level + end_water_level) / 2
                        downstream_water_level = self.fitters.fit_H_downstream(Q_out_guess)
                        head_loss = num_turbines * self.fitters.fit_dH_loss(Q_out_guess / num_turbines)
                        net_head = avg_water_level - downstream_water_level - head_loss
                        actual_power = num_turbines * self.fitters.fit_output_N(Q_out_guess / num_turbines, net_head)
                        if avg_water_level < height_of_spillway:
                            max_limit_Q = num_turbines * max_generation_flow
                        else:
                            max_limit_Q = num_turbines * max_generation_flow + num_spillway * self.fitters.fit_spill_Q(avg_water_level)
                        if actual_power <= planned_power:
                            Q_power = Q_out_guess
                            Q_abandoned = 0
                            open_level = 0
                            break
                        else:
                            if abs(actual_power - planned_power) < self.tolerance:
                                converged = True
                            else:
                                if abs(actual_power) > 1e-6:
                                    Q_out_guess = Q_out_guess * (planned_power / abs(actual_power))
                                continue
                        continue
                    avg_water_level = (current_water_level + end_water_level) / 2
                    downstream_water_level = self.fitters.fit_H_downstream(Q_out_guess)
                    head_loss = num_turbines * self.fitters.fit_dH_loss(Q_out_guess / num_turbines)
                    net_head = avg_water_level - downstream_water_level - head_loss
                    actual_power = num_turbines * self.fitters.fit_output_N(Q_out_guess / num_turbines, net_head)
                    if avg_water_level < height_of_spillway:
                        max_limit_Q = num_turbines * max_generation_flow
                    else:
                        max_limit_Q = num_turbines * max_generation_flow + num_spillway * self.fitters.fit_spill_Q(avg_water_level)
                    if abs(actual_power - planned_power) < self.tolerance:
                        converged = True
                        if Q_out_guess <= Q_min:
                            Q_out_guess = Q_min
                            end_storage = current_storage + (current_inflow - Q_out_guess) * seconds_per_month / 1e8
                            end_water_level = self.fitters.fit_H(end_storage)
                            avg_water_level = (current_water_level + end_water_level) / 2
                            downstream_water_level = self.fitters.fit_H_downstream(Q_out_guess)
                            head_loss = num_turbines * self.fitters.fit_dH_loss(Q_out_guess / num_turbines)
                            net_head = avg_water_level - downstream_water_level - head_loss
                            actual_power = num_turbines * self.fitters.fit_output_N(Q_out_guess / num_turbines, net_head)
                            if avg_water_level < height_of_spillway:
                                max_limit_Q = num_turbines * max_generation_flow
                            else:
                                max_limit_Q = num_turbines * max_generation_flow + num_spillway * self.fitters.fit_spill_Q(avg_water_level)
                            Q_power = Q_out_guess
                            Q_abandoned = 0
                            open_level = 0
                            break
                    else:
                        if abs(actual_power) > 1e-6:
                            Q_out_guess = Q_out_guess * (planned_power / abs(actual_power))
                        Q_power = Q_out_guess
                        Q_abandoned = 0
                        open_level = 0
                water_level_2d[year, month + 1] = end_water_level
                storage_2d[year, month + 1] = end_storage
            if year < num_years - 1:
                water_level_2d[year + 1, 0] = water_level_2d[year, num_months]
                storage_2d[year + 1, 0] = storage_2d[year, num_months]
        water_level_seq = np.zeros(self.n_stages + 1)
        water_level_seq[0] = initial_water_level
        for year in range(num_years):
            for month in range(num_months):
                t = year * num_months + month
                water_level_seq[t + 1] = water_level_2d[year, month + 1]
        return water_level_seq
import numpy as np
from typing import Dict, Any
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
class DPDispatch(BaseDispatchAlgorithm):
    """
    动态规划调度算法。
    将水位状态离散化后，采用逐阶段递推求最优轨迹。
    """
    algorithm_name = "动态规划(DP)"
    def __init__(
        self,
        data: Dict[str, Any] = None,
        reservoir_config: ReservoirConfig = None,
        dispatch_config: DispatchConfig = None,
        algorithm_config: AlgorithmConfig = None,
        data_file: str = None
    ):
        super().__init__(data, reservoir_config, dispatch_config, algorithm_config, data_file)
        self.d_water_level = self.algorithm_config.d_water_level
        self.penalty_coefficient = self.algorithm_config.penalty_coefficient
        self.tolerance = 0
    def _generate_state_values(self, H_min: float, H_max: float) -> np.ndarray:
        """生成离散状态值。"""
        return np.arange(H_min, H_max + self.d_water_level, self.d_water_level)
    def _calculate_transition_energy_batch(
        self,
        H_prev: np.ndarray,
        H_curr: float,
        inflow: float
    ) -> np.ndarray:
        """
        批量计算 prev->curr 转移发电量，不可行转移返回 -inf。
        """
        H_prev = np.asarray(H_prev, dtype=float)
        num_turbines = self.reservoir_config.num_turbines
        max_generation_flow = self.reservoir_config.max_generation_flow
        num_spillway = self.reservoir_config.num_spillway
        height_of_spillway = self.reservoir_config.height_of_spillway
        seconds_per_month = self.reservoir_config.seconds_per_month
        Q_min = self.dispatch_config.Q_min
        guaranteed_output = self.dispatch_config.guaranteed_output
        V_start = self.fitters.fit_V(H_prev)
        V_end = self.fitters.fit_V(H_curr)
        Q_release = (V_start - V_end) * 1e8 / seconds_per_month + inflow
        Z_avg = 0.5 * (H_prev + H_curr)
        spill_Q = self.fitters.fit_spill_Q(Z_avg)
        Q_max = np.where(
            Z_avg < height_of_spillway,
            num_turbines * max_generation_flow,
            num_turbines * max_generation_flow + num_spillway * spill_Q
        )
        feasible = (Q_release >= Q_min) & (Q_release <= Q_max)
        energy = np.full(H_prev.shape, -np.inf, dtype=float)
        if not np.any(feasible):
            return energy
        Q_release_f = Q_release[feasible]
        H_tail = np.atleast_1d(self.fitters.fit_H_downstream(Q_release_f)).astype(float)
        Q_gen = np.minimum(Q_release_f, num_turbines * max_generation_flow)
        q_unit = Q_gen / num_turbines
        dH_loss = num_turbines * np.atleast_1d(self.fitters.fit_dH_loss(q_unit)).astype(float)
        H_net = Z_avg[feasible] - H_tail - dH_loss
        positive_head = H_net > 0
        if np.any(positive_head):
            q_pos = q_unit[positive_head]
            h_pos = H_net[positive_head]
            N = num_turbines * np.atleast_1d(self.fitters.fit_output_N(q_pos, h_pos)).astype(float)
            low_power = N < guaranteed_output
            N[low_power] = N[low_power] - self.penalty_coefficient * (guaranteed_output - N[low_power])
            N = np.maximum(N, 0.0)
            e_pos = N * 30.4 * 24
            e_full = np.zeros_like(Q_release_f)
            e_full[positive_head] = e_pos
            energy[feasible] = e_full
        else:
            energy[feasible] = 0.0
        return energy
    def _run_algorithm(self) -> np.ndarray:
        """
        运行 DP 算法。
        """
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level
        global_min_H = np.min(self.H_dead_full)
        global_max_H = np.max(self.H_max_full)
        state_values_global = self._generate_state_values(global_min_H, global_max_H)
        n_state = len(state_values_global)
        print(f"  状态空间大小: {n_state} 个离散水位")
        month_of_stage = np.mod(np.arange(self.n_stages), 12)
        month_h_dead = self.data['dispatch_limits']['H_dead'][month_of_stage]
        month_h_max = self.data['dispatch_limits']['H_max'][month_of_stage]
        feasible_mask = (
            (state_values_global[None, :] >= month_h_dead[:, None])
            & (state_values_global[None, :] <= month_h_max[:, None])
        )
        feasible_indices = [np.flatnonzero(feasible_mask[t]) for t in range(self.n_stages)]
        cumulative_energy = np.full((self.n_stages, n_state), -np.inf)
        backtrack = np.zeros((self.n_stages, n_state), dtype=int)
        print("  处理第 1 阶段...")
        stage0_indices = feasible_indices[0]
        if len(stage0_indices) > 0:
            stage0_states = state_values_global[stage0_indices]
            stage0_energy = self._calculate_transition_energy_batch(
                np.full(len(stage0_states), initial_water_level),
                stage0_states,
                self.inflow[0]
            )
            valid = stage0_energy >= 0
            if np.any(valid):
                valid_indices = stage0_indices[valid]
                cumulative_energy[0, valid_indices] = stage0_energy[valid]
                backtrack[0, valid_indices] = -1
        for t in range(1, self.n_stages - 1):
            if (t + 1) % 50 == 0:
                print(f"  处理第 {t + 1} 阶段...")
            prev_candidates = feasible_indices[t - 1]
            curr_candidates = feasible_indices[t]
            if len(prev_candidates) == 0 or len(curr_candidates) == 0:
                continue
            reachable_mask = np.isfinite(cumulative_energy[t - 1, prev_candidates])
            if not np.any(reachable_mask):
                continue
            prev_indices = prev_candidates[reachable_mask]
            prev_states = state_values_global[prev_indices]
            prev_cumulative = cumulative_energy[t - 1, prev_indices]
            inflow_t = self.inflow[t]
            for j in curr_candidates:
                curr_H = state_values_global[j]
                energy = self._calculate_transition_energy_batch(prev_states, curr_H, inflow_t)
                feasible_transitions = np.isfinite(energy) & (energy >= 0)
                if not np.any(feasible_transitions):
                    continue
                total = prev_cumulative[feasible_transitions] + energy[feasible_transitions]
                best_local = np.argmax(total)
                cumulative_energy[t, j] = total[best_local]
                backtrack[t, j] = prev_indices[feasible_transitions][best_local]
        print(f"  处理第 {self.n_stages} 阶段...")
        t = self.n_stages - 1
        best_cum = -np.inf
        best_k = 0
        prev_candidates = feasible_indices[t - 1]
        if len(prev_candidates) > 0:
            reachable_mask = np.isfinite(cumulative_energy[t - 1, prev_candidates])
            if np.any(reachable_mask):
                prev_indices = prev_candidates[reachable_mask]
                prev_states = state_values_global[prev_indices]
                prev_cumulative = cumulative_energy[t - 1, prev_indices]
                energy = self._calculate_transition_energy_batch(
                    prev_states,
                    final_water_level,
                    self.inflow[t]
                )
                feasible_transitions = np.isfinite(energy) & (energy >= 0)
                if np.any(feasible_transitions):
                    total = prev_cumulative[feasible_transitions] + energy[feasible_transitions]
                    best_local = np.argmax(total)
                    best_cum = total[best_local]
                    best_k = prev_indices[feasible_transitions][best_local]
                    cumulative_energy[t, best_k] = best_cum
                    backtrack[t, best_k] = best_k
        optimal_trajectory = np.zeros(self.n_stages + 1)
        optimal_trajectory[0] = initial_water_level
        optimal_trajectory[-1] = final_water_level
        optimal_index = np.zeros(self.n_stages, dtype=int)
        optimal_index[-1] = best_k
        for t in range(self.n_stages - 2, -1, -1):
            optimal_index[t] = backtrack[t + 1, optimal_index[t + 1]]
        for t in range(self.n_stages - 1):
            optimal_trajectory[t + 1] = state_values_global[optimal_index[t]]
        print(f"  DP 完成，最优总发电量估计: {best_cum / 1000:.2f} GWh")
        return optimal_trajectory
import numpy as np
from typing import Dict, Any
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
class POADispatch(BaseDispatchAlgorithm):
    """POA 逐步优化算法。"""
    algorithm_name = "POA逐步优化"
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
        self.max_iterations = self.algorithm_config.poa_max_iterations
        self.tolerance = self.algorithm_config.poa_tolerance
        self.d_water_level = self.algorithm_config.poa_water_level
        self.penalty_coefficient = self.algorithm_config.penalty_coefficient
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
        """批量计算单阶段发电量；不可行返回 -1e6。"""
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
    def _calculate_total_energy_for_trajectory(
        self,
        water_level_trajectory: np.ndarray,
        inflow_seq: np.ndarray,
    ) -> float:
        """批量计算当前轨迹总发电量，用于收敛判断。"""
        stage_month_idx = np.mod(np.arange(self.n_stages), 12)
        z_start = water_level_trajectory[:-1]
        z_end = water_level_trajectory[1:]
        total_energy = 0.0
        for month_idx in range(12):
            mask = stage_month_idx == month_idx
            if not np.any(mask):
                continue
            energy = self._calculate_stage_energy_batch(
                z_start[mask],
                z_end[mask],
                inflow_seq[mask],
                month_idx,
            )
            valid = energy > 0
            if np.any(valid):
                total_energy += float(np.sum(energy[valid]))
        return total_energy
    def _run_algorithm(self) -> np.ndarray:
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level
        if self.initial_trajectory is None:
            water_level_trajectory = np.linspace(
                initial_water_level,
                final_water_level,
                self.n_stages + 1,
            )
        else:
            water_level_trajectory = self.initial_trajectory.copy()
        water_level_trajectory[0] = initial_water_level
        water_level_trajectory[-1] = final_water_level
        total_energy_history = []
        inflow_seq = self.inflow[:self.n_stages]
        monthly_candidates = []
        monthly_candidate_volume = []
        for month_idx in range(12):
            H_dead = self.data['dispatch_limits']['H_dead'][month_idx]
            H_max = self.data['dispatch_limits']['H_max'][month_idx]
            candidates = np.arange(H_dead, H_max + self.d_water_level, self.d_water_level)
            monthly_candidates.append(candidates)
            monthly_candidate_volume.append(
                np.atleast_1d(self.fitters.fit_V(candidates)).astype(float)
            )
        print(f"  开始POA优化，最大迭代次数: {self.max_iterations}")
        for iter_num in range(self.max_iterations):
            old_trajectory = water_level_trajectory.copy()
            for t in range(1, self.n_stages):
                prev_month_idx = (t - 1) % 12
                next_month_idx = t % 12
                Z_candidates = monthly_candidates[prev_month_idx]
                V_candidates = monthly_candidate_volume[prev_month_idx]
                energy1 = self._calculate_stage_energy_batch(
                    water_level_trajectory[t - 1],
                    Z_candidates,
                    inflow_seq[t - 1],
                    prev_month_idx,
                    V_end=V_candidates,
                )
                energy2 = self._calculate_stage_energy_batch(
                    Z_candidates,
                    water_level_trajectory[t + 1],
                    inflow_seq[t],
                    next_month_idx,
                    V_start=V_candidates,
                )
                valid = (energy1 >= 0) & (energy2 >= 0)
                if np.any(valid):
                    total_local_energy = energy1[valid] + energy2[valid]
                    best_idx = np.argmax(total_local_energy)
                    water_level_trajectory[t] = Z_candidates[valid][best_idx]
            water_level_trajectory[0] = initial_water_level
            water_level_trajectory[-1] = final_water_level
            total_energy = self._calculate_total_energy_for_trajectory(
                water_level_trajectory,
                inflow_seq,
            )
            total_energy_history.append(total_energy)
            max_change = np.max(np.abs(water_level_trajectory[1:-1] - old_trajectory[1:-1]))
            print(
                f"    迭代 {iter_num + 1}: 总发电量 = {total_energy / 1000:.2f} GWh, "
                f"最大水位变化 = {max_change:.4f} m"
            )
            if max_change < self.tolerance:
                print(f"  POA算法在第 {iter_num + 1} 次迭代后收敛")
                break
        self._convergence_history = np.array(total_energy_history)
        return water_level_trajectory
    def run(self):
        result = super().run()
        if hasattr(self, '_convergence_history'):
            result.convergence_history = self._convergence_history
        return result
import numpy as np
from typing import Dict, Any, List, Tuple
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
            new_fitness = self._evaluate_population(Xnew)
            new_best_idx = int(np.argmin(new_fitness))
            localBest_position = Xnew[new_best_idx, :].copy()
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
import numpy as np
from typing import Dict, Any
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
class DholeDispatch(BaseDispatchAlgorithm):
    """Use DOA to optimize the reservoir dispatch trajectory."""
    algorithm_name = "Dhole优化算法(DOA)"
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
        H_prev = np.roll(H, 1)
        V = np.atleast_1d(self.fitters.fit_V(H)).astype(float)
        V_prev = np.roll(V, 1)
        Qout = self._inflow_seq - (V - V_prev) * 1e8 / self._sec_per_month
        avg_H = 0.5 * (H_prev + H)
        penalty = 0.0
        below = np.maximum(self._h_dead_seq - H, 0.0)
        above = np.maximum(H - self._h_max_seq, 0.0)
        penalty += self.pc_bounds * float(np.sum(below + above))
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
        spill_excess = np.maximum(qout_valid - maxQ, 0.0)
        penalty += self.pc_spill * float(np.sum(spill_excess))
        qmin_deficit = np.maximum(self._q_min - qout_valid, 0.0)
        penalty += self.pc_minQ * float(np.sum(qmin_deficit))
        q_power = np.minimum(qout_valid, self._q_turbine_max)
        q_unit = np.maximum(q_power / self._num_turbines, 1e-10)
        hdown = np.atleast_1d(self.fitters.fit_H_downstream(qout_valid)).astype(float)
        head_loss = self._num_turbines * np.atleast_1d(self.fitters.fit_dH_loss(q_unit)).astype(float)
        net_head = avg_h_valid - hdown - head_loss
        p_inst = self._num_turbines * np.atleast_1d(self.fitters.fit_output_N(q_unit, net_head)).astype(float)
        guarantee_deficit = np.maximum(self._guaranteed_output - p_inst, 0.0)
        penalty += self.pc_guarantee * float(np.sum(guarantee_deficit))
        total_energy = float(np.sum(p_inst * 30.4 * 24))
        return -total_energy + penalty
    def _run_algorithm(self) -> np.ndarray:
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level
        state_lb = np.tile(self.data['dispatch_limits']['H_dead'], self.dispatch_config.num_years)
        state_ub = np.tile(self.data['dispatch_limits']['H_max'], self.dispatch_config.num_years)
        state_lb[-1] = final_water_level
        state_ub[-1] = final_water_level
        print(
            f"  开始DOA优化: pop={self.population_size}, "
            f"iter={self.max_iterations}, dim={self.n_stages}"
        )
        optimizer = DholeOptimizer(
            n_population=self.population_size,
            max_iterations=self.max_iterations,
            lb=state_lb,
            ub=state_ub,
            dim=self.n_stages,
            objective_func=self._fitness_function,
            random_seed=self.random_seed,
        )
        best_fitness, best_H_seq, convergence_curve = optimizer.optimize()
        print(f"  DOA完成, best fitness = {best_fitness:.4e}")
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
import numpy as np
from typing import Dict, Any
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
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
        H_prev = np.roll(H, 1)
        V = np.atleast_1d(self.fitters.fit_V(H)).astype(float)
        V_prev = np.roll(V, 1)
        Qout = self._inflow_seq - (V - V_prev) * 1e8 / self._sec_per_month
        avg_H = 0.5 * (H_prev + H)
        penalty = 0.0
        below = np.maximum(self._h_dead_seq - H, 0.0)
        above = np.maximum(H - self._h_max_seq, 0.0)
        penalty += self.pc_bounds * float(np.sum(below + above))
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
        spill_excess = np.maximum(qout_valid - maxQ, 0.0)
        penalty += self.pc_spill * float(np.sum(spill_excess))
        qmin_deficit = np.maximum(self._q_min - qout_valid, 0.0)
        penalty += self.pc_minQ * float(np.sum(qmin_deficit))
        q_power = np.minimum(qout_valid, self._q_turbine_max)
        q_unit = np.maximum(q_power / self._num_turbines, 1e-10)
        hdown = np.atleast_1d(self.fitters.fit_H_downstream(qout_valid)).astype(float)
        head_loss = self._num_turbines * np.atleast_1d(self.fitters.fit_dH_loss(q_unit)).astype(float)
        net_head = avg_h_valid - hdown - head_loss
        p_inst = self._num_turbines * np.atleast_1d(self.fitters.fit_output_N(q_unit, net_head)).astype(float)
        guarantee_deficit = np.maximum(self._guaranteed_output - p_inst, 0.0)
        penalty += self.pc_guarantee * float(np.sum(guarantee_deficit))
        total_energy = float(np.sum(p_inst * 30.4 * 24))
        return -total_energy + penalty
    def _run_algorithm(self) -> np.ndarray:
        initial_water_level = self.dispatch_config.initial_water_level
        final_water_level = self.dispatch_config.final_water_level
        state_lb = np.tile(self.data['dispatch_limits']['H_dead'], self.dispatch_config.num_years)
        state_ub = np.tile(self.data['dispatch_limits']['H_max'], self.dispatch_config.num_years)
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
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
class DispatchPlotter:
    """
    调度结果可视化类
    """
    def __init__(
        self,
        dispatch_config: Any = None,
        reservoir_config: Any = None
    ):
        """
        初始化绘图器
        """
        from config import DEFAULT_DISPATCH, DEFAULT_RESERVOIR
        self.dispatch_config = dispatch_config or DEFAULT_DISPATCH
        self.reservoir_config = reservoir_config or DEFAULT_RESERVOIR
    def plot_water_level_process(
        self,
        result: Any,
        H_max_full: np.ndarray = None,
        H_dead_full: np.ndarray = None,
        figsize: tuple = (12, 8),
        save_path: str = None
    ):
        """
        绘制水位过程线
        """
        sim = result.simulation_result
        n_stages = len(sim.power)
        num_years = self.dispatch_config.num_years
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        ax1 = axes[0]
        ax1.plot(range(n_stages + 1), sim.water_level, 'b-', linewidth=1.5, label='实际水位')
        if H_max_full is not None:
            H_max_plot = np.concatenate([[H_max_full[-1]], H_max_full])
            ax1.plot(range(n_stages + 1), H_max_plot, 'r--', linewidth=1, label='允许最高')
        if H_dead_full is not None:
            H_dead_plot = np.concatenate([[H_dead_full[-1]], H_dead_full])
            ax1.plot(range(n_stages + 1), H_dead_plot, 'g--', linewidth=1, label='死水位')
        ax1.set_xlabel('时间 (月)')
        ax1.set_ylabel('水位 (m)')
        ax1.set_title(f'水库月末水位变化过程 - {result.algorithm_name}')
        ax1.legend(loc='best')
        ax1.grid(True)
        ax2 = axes[1]
        annual_avg = np.zeros(num_years)
        for y in range(num_years):
            idx_start = y * 12 + 1
            idx_end = (y + 1) * 12 + 1
            annual_avg[y] = np.mean(sim.water_level[idx_start:idx_end])
        ax2.bar(range(1, num_years + 1), annual_avg)
        ax2.set_xlabel('年份')
        ax2.set_ylabel('年平均水位 (m)')
        ax2.set_title('年度平均水位')
        ax2.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    def plot_power_energy(
        self,
        result: Any,
        figsize: tuple = (12, 8),
        save_path: str = None
    ):
        """
        绘制出力和发电量
        """
        sim = result.simulation_result
        n_stages = len(sim.power)
        num_years = self.dispatch_config.num_years
        guaranteed_output = self.dispatch_config.guaranteed_output
        installed_output = self.reservoir_config.installed_output
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        ax1 = axes[0]
        ax1.plot(range(1, n_stages + 1), sim.power, 'b-', linewidth=1.5, label='实际出力')
        ax1.axhline(y=guaranteed_output, color='r', linestyle='--', linewidth=1.5, label='保证出力')
        ax1.axhline(y=installed_output, color='g', linestyle='--', linewidth=1.5, label='装机容量')
        ax1.set_xlabel('时间 (月)')
        ax1.set_ylabel('出力 (MW)')
        ax1.set_title(f'实际出力变化过程 - {result.algorithm_name}')
        ax1.legend(loc='best')
        ax1.grid(True)
        ax2 = axes[1]
        annual_energy = np.zeros(num_years)
        for y in range(num_years):
            idx_start = y * 12
            idx_end = (y + 1) * 12
            annual_energy[y] = np.sum(sim.energy[idx_start:idx_end]) / 1000
        avg_annual = np.mean(annual_energy)
        ax2.bar(range(1, num_years + 1), annual_energy, label='年发电量')
        ax2.axhline(y=avg_annual, color='r', linestyle='-', linewidth=2, label=f'多年平均 ({avg_annual:.2f} GWh)')
        ax2.set_xlabel('年份')
        ax2.set_ylabel('年发电量 (GWh)')
        ax2.set_title('年发电量变化过程')
        ax2.legend(loc='best')
        ax2.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    def plot_outflow(
        self,
        result: Any,
        inflow: np.ndarray = None,
        figsize: tuple = (12, 8),
        save_path: str = None
    ):
        """
        绘制出入库流量
        """
        sim = result.simulation_result
        n_stages = len(sim.power)
        num_years = self.dispatch_config.num_years
        Q_min = self.dispatch_config.Q_min
        seconds_per_month = self.reservoir_config.seconds_per_month
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        ax1 = axes[0]
        ax1.plot(range(1, n_stages + 1), sim.outflow, 'b-', linewidth=1.5, label='出库流量')
        ax1.plot(range(1, n_stages + 1), sim.max_limit_Q, 'r--', linewidth=1, label='最大下泄能力')
        ax1.axhline(y=Q_min, color='g', linestyle='--', linewidth=1, label='最小下泄')
        ax1.set_xlabel('时间 (月)')
        ax1.set_ylabel('流量 (m^3/s)')
        ax1.set_title(f'出库流量变化过程 - {result.algorithm_name}')
        ax1.legend(loc='best')
        ax1.grid(True)
        ax2 = axes[1]
        annual_outflow = np.zeros(num_years)
        annual_inflow = np.zeros(num_years)
        for y in range(num_years):
            idx_start = y * 12
            idx_end = (y + 1) * 12
            annual_outflow[y] = np.sum(sim.outflow[idx_start:idx_end]) * seconds_per_month / 1e8
            if inflow is not None:
                annual_inflow[y] = np.sum(inflow[idx_start:idx_end]) * seconds_per_month / 1e8
        x = np.arange(1, num_years + 1)
        width = 0.35
        if inflow is not None:
            ax2.bar(x - width/2, annual_inflow, width, label='入库水量')
            ax2.bar(x + width/2, annual_outflow, width, label='出库水量')
        else:
            ax2.bar(x, annual_outflow, label='出库水量')
        ax2.set_xlabel('年份')
        ax2.set_ylabel('年总水量 (亿m^3)')
        ax2.set_title('年度入库与出库水量对比')
        ax2.legend(loc='best')
        ax2.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    def plot_abandoned_water(
        self,
        result: Any,
        figsize: tuple = (12, 8),
        save_path: str = None
    ):
        """
        绘制弃水分析
        """
        sim = result.simulation_result
        n_stages = len(sim.power)
        num_years = self.dispatch_config.num_years
        seconds_per_month = self.reservoir_config.seconds_per_month
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        ax1 = axes[0]
        ax1.plot(range(1, n_stages + 1), sim.abandoned_Q, 'r-', linewidth=1.5)
        ax1.set_xlabel('时间 (月)')
        ax1.set_ylabel('弃水流量 (m^3/s)')
        ax1.set_title(f'月弃水流量变化 - {result.algorithm_name}')
        ax1.grid(True)
        ax2 = axes[1]
        annual_abandoned = np.zeros(num_years)
        for y in range(num_years):
            idx_start = y * 12
            idx_end = (y + 1) * 12
            annual_abandoned[y] = np.sum(sim.abandoned_Q[idx_start:idx_end]) * seconds_per_month / 1e8
        ax2.bar(range(1, num_years + 1), annual_abandoned)
        ax2.set_xlabel('年份')
        ax2.set_ylabel('年弃水量 (亿m^3)')
        ax2.set_title('年度弃水量统计')
        ax2.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    def plot_convergence(
        self,
        result: Any,
        figsize: tuple = (10, 5),
        save_path: str = None
    ):
        """
        Plot algorithm convergence curve.
        """
        if result.convergence_history is None:
            print("No convergence history available.")
            return
        history = np.asarray(result.convergence_history, dtype=float).ravel()
        finite_mask = np.isfinite(history)
        if not np.any(finite_mask):
            print("Convergence history has no finite values.")
            return
        history = history[finite_mask]
        iterations = np.arange(1, len(history) + 1)
        fig, ax = plt.subplots(figsize=figsize)
        if np.all(history > 0):
            ax.semilogy(iterations, history, 'b-', linewidth=1.5)
            ax.set_ylabel('Fitness (log scale)')
        else:
            ax.plot(iterations, history, 'b-', linewidth=1.5)
            ax.set_ylabel('Fitness')
            if np.any(history <= 0):
                ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
                ax.text(
                    0.02,
                    0.98,
                    'Non-positive values detected; switched to linear scale.',
                    transform=ax.transAxes,
                    va='top',
                    fontsize=9,
                )
        ax.set_xlabel('Iteration')
        ax.set_title(f'{result.algorithm_name} Convergence')
        ax.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    def plot_all(
        self,
        result: Any,
        inflow: np.ndarray = None,
        H_max_full: np.ndarray = None,
        H_dead_full: np.ndarray = None,
        save_dir: str = None
    ):
        """
        绘制所有图表
        """
        import os
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            water_path = os.path.join(save_dir, 'water_level.png')
            power_path = os.path.join(save_dir, 'power_energy.png')
            outflow_path = os.path.join(save_dir, 'outflow.png')
            abandoned_path = os.path.join(save_dir, 'abandoned.png')
            conv_path = os.path.join(save_dir, 'convergence.png')
        else:
            water_path = power_path = outflow_path = abandoned_path = conv_path = None
        self.plot_water_level_process(result, H_max_full, H_dead_full, save_path=water_path)
        self.plot_power_energy(result, save_path=power_path)
        self.plot_outflow(result, inflow, save_path=outflow_path)
        self.plot_abandoned_water(result, save_path=abandoned_path)
        if result.convergence_history is not None:
            self.plot_convergence(result, save_path=conv_path)
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TRAJECTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'trajectories')
def save_trajectory(trajectory: np.ndarray, name: str) -> str:
    """
    保存水位过程线到文件
    """
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    filepath = os.path.join(TRAJECTORY_DIR, f'{name}.npy')
    np.save(filepath, trajectory)
    print(f"水位过程线已保存至: {filepath}")
    return filepath
def load_trajectory(name: str) -> np.ndarray:
    """
    从文件加载水位过程线
    """
    filepath = os.path.join(TRAJECTORY_DIR, f'{name}.npy')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"水位过程线文件不存在: {filepath}")
    trajectory = np.load(filepath)
    print(f"已加载水位过程线: {filepath}")
    return trajectory
def list_saved_trajectories() -> list:
    """列出所有已保存的水位过程线"""
    if not os.path.exists(TRAJECTORY_DIR):
        return []
    files = [f[:-4] for f in os.listdir(TRAJECTORY_DIR) if f.endswith('.npy')]
    return files
def run_conventional_dispatch(show_plot=True, save_result=True):
    """运行常规调度算法
    """
    print("\n" + "=" * 60)
    print("运行常规调度算法")
    print("=" * 60)
    algo = ConventionalDispatch()
    result = algo.run()
    if save_result:
        save_trajectory(result.water_level_trajectory, 'conventional')
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_all(result, algo.inflow, algo.H_max_full, algo.H_dead_full)
    return result
def run_dp_dispatch(show_plot=True, save_result=True):
    """运行动态规划算法
    """
    print("\n" + "=" * 60)
    print("运行动态规划(DP)算法")
    print("=" * 60)
    algo_config = AlgorithmConfig(d_water_level=0.5)
    algo = DPDispatch(algorithm_config=algo_config)
    result = algo.run()
    if save_result:
        save_trajectory(result.water_level_trajectory, 'dp')
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_all(result, algo.inflow, algo.H_max_full, algo.H_dead_full)
    return result
def run_poa_dispatch(initial_trajectory=None, show_plot=True, save_result=True):
    """运行POA逐步优化算法
    """
    print("\n" + "=" * 60)
    print("运行POA逐步优化算法")
    print("=" * 60)
    algo = POADispatch(initial_trajectory=initial_trajectory)
    result = algo.run()
    if save_result:
        save_trajectory(result.water_level_trajectory, 'poa')
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_all(result, algo.inflow, algo.H_max_full, algo.H_dead_full)
    return result
def run_dddp_dispatch(initial_trajectory=None, show_plot=True, save_result=True):
    """运行DDDP离散微分动态规划算法
    """
    print("\n" + "=" * 60)
    print("运行DDDP离散微分动态规划算法")
    print("=" * 60)
    algo = DDDPDispatch(initial_trajectory=initial_trajectory)
    result = algo.run()
    if save_result:
        save_trajectory(result.water_level_trajectory, 'dddp')
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_all(result, algo.inflow, algo.H_max_full, algo.H_dead_full)
    return result
def run_dhole_dispatch(max_iter=500, pop_size=30, show_plot=True, save_result=True):
    """运行豺狼优化算法（演示模式，减少迭代次数）
    """
    print("\n" + "=" * 60)
    print("运行豺狼优化算法(DHOLE)")
    print("=" * 60)
    algo_config = AlgorithmConfig(
        max_iterations=max_iter,
        population_size=pop_size,
        random_seed=1
    )
    algo = DholeDispatch(algorithm_config=algo_config)
    result = algo.run()
    if save_result:
        save_trajectory(result.water_level_trajectory, 'dhole')
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_all(result, algo.inflow, algo.H_max_full, algo.H_dead_full)
    return result
def run_pso_dispatch(max_iter=500, pop_size=30, show_plot=True, save_result=True):
    """运行粒子群优化算法（演示模式，减少迭代次数）
    """
    print("\n" + "=" * 60)
    print("运行粒子群优化算法(PSO)")
    print("=" * 60)
    algo_config = AlgorithmConfig(
        max_iterations=max_iter,
        population_size=pop_size,
        random_seed=1
    )
    algo = PSODispatch(algorithm_config=algo_config)
    result = algo.run()
    if save_result:
        save_trajectory(result.water_level_trajectory, 'pso')
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_all(result, algo.inflow, algo.H_max_full, algo.H_dead_full)
    return result
def compare_algorithms():
    """比较不同算法的结果"""
    print("\n" + "=" * 60)
    print("算法对比")
    print("=" * 60)
    results = {}
    print("\n[1/3] 运行常规调度...")
    algo_conv = ConventionalDispatch()
    results['常规调度'] = algo_conv.run()
    print("\n[2/3] 运行DP...")
    algo_config_dp = AlgorithmConfig(d_water_level=2.0)
    algo_dp = DPDispatch(algorithm_config=algo_config_dp)
    results['DP'] = algo_dp.run()
    print("\n[3/3] 运行POA...")
    algo_poa = POADispatch(
        initial_trajectory=results['常规调度'].water_level_trajectory
    )
    results['POA'] = algo_poa.run()
    print("\n" + "=" * 60)
    print("算法对比结果")
    print("=" * 60)
    print(f"{'算法':<15} {'总发电量(GWh)':<15} {'保证率(%)':<12} {'运行时间(s)':<12}")
    print("-" * 54)
    dispatch_config = DEFAULT_DISPATCH
    n_stages = dispatch_config.n_stages
    guaranteed = dispatch_config.guaranteed_output
    for name, result in results.items():
        sim = result.simulation_result
        total_energy = sim.total_energy / 1000
        num_lower = np.sum(sim.power < guaranteed)
        guarantee_rate = (1 - num_lower / n_stages) * 100
        elapsed = result.elapsed_time
        print(f"{name:<15} {total_energy:<15.2f} {guarantee_rate:<12.2f} {elapsed:<12.2f}")
    return results
def main():
    """主函数"""
    print("=" * 60)
    print("水库调度算法库 - 主程序")
    print("=" * 60)
    if not os.path.exists(DATA_FILE):
        print(f"错误: 数据文件不存在: {DATA_FILE}")
        print("请确保已将 data_ty.xls 复制到 data/ 目录")
        return
    print(f"数据文件: {DATA_FILE}")
    print("\n请选择运行模式:")
    print("1. 运行常规调度算法")
    print("2. 运行动态规划(DP)算法")
    print("3. 运行POA逐步优化算法")
    print("4. 运行DDDP算法")
    print("5. 运行豺狼优化算法(DHOLE)")
    print("6. 运行粒子群优化算法(PSO)")
    print("7. 算法对比")
    print("0. 退出")
    saved = list_saved_trajectories()
    if saved:
        print(f"\n已保存的水位过程线: {', '.join(saved)}")
    try:
        choice = input("\n请输入选择 (默认1): ").strip()
        if choice == "":
            choice = "1"
        choice = int(choice)
    except:
        choice = 1
    if choice == 1:
        run_conventional_dispatch()
    elif choice == 2:
        run_dp_dispatch()
    elif choice == 3:
        if 'dp' in saved:
            use_saved = input("检测到已保存的DP调度轨迹，是否使用? (Y/n): ").strip().lower()
            if use_saved != 'n':
                trajectory = load_trajectory('dp')
                run_poa_dispatch(trajectory)
            else:
                dp_result = run_dp_dispatch(show_plot=False)
                run_poa_dispatch(dp_result.water_level_trajectory)
        else:
            dp_result = run_dp_dispatch(show_plot=False)
            run_poa_dispatch(dp_result.water_level_trajectory)
    elif choice == 4:
        if 'dp' in saved:
            use_saved = input("检测到已保存的DP调度轨迹，是否使用? (Y/n): ").strip().lower()
            if use_saved != 'n':
                trajectory = load_trajectory('dp')
                run_dddp_dispatch(trajectory)
            else:
                dp_result = run_dp_dispatch(show_plot=False)
                run_dddp_dispatch(dp_result.water_level_trajectory)
        else:
            dp_result = run_dp_dispatch(show_plot=False)
            run_dddp_dispatch(dp_result.water_level_trajectory)
    elif choice == 5:
        run_dhole_dispatch(max_iter=3000, pop_size=50)
    elif choice == 6:
        run_pso_dispatch(max_iter=3000, pop_size=50)
    elif choice == 7:
        compare_algorithms()
    else:
        print("再见!")
if __name__ == "__main__":
    main()
