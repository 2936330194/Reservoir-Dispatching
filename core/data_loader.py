"""
数据加载模块
============

从Excel文件加载水库调度所需的各种曲线数据
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import os

from config import DATA_FILE


def load_reservoir_data(data_file: str = None) -> Dict[str, Any]:
    """
    加载水库调度所需的全部数据
    
    Parameters
    ----------
    data_file : str, optional
        数据文件路径，默认使用配置中的路径
    
    Returns
    -------
    Dict[str, Any]
        包含以下键的字典:
        - 'water_volume': 水位-库容关系数据
        - 'tail_water': 尾水-流量关系数据
        - 'spillway': 泄流特性曲线数据
        - 'power_output': 出力特性曲线数据
        - 'head_loss': 水头损失曲线数据
        - 'dispatch_limits': 调度图控制水位数据
        - 'inflow': 历史径流数据
    """
    if data_file is None:
        data_file = DATA_FILE
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    data = {}
    
    # 1. 水位-库容关系 (跳过第一行标题)
    df_wv = pd.read_excel(data_file, sheet_name='水位库容关系', header=None, skiprows=1)
    # 过滤掉非数值行
    df_wv = df_wv.dropna(subset=[1, 2])
    data['water_volume'] = {
        'H_levels': pd.to_numeric(df_wv.iloc[:, 1], errors='coerce').dropna().values,  # 水位 (m)
        'V_values': pd.to_numeric(df_wv.iloc[:, 2], errors='coerce').dropna().values,  # 库容 (亿m³)
    }
    
    # 2. 尾水-流量关系 (跳过第一行标题)
    df_tw = pd.read_excel(data_file, sheet_name='尾水流量关系', header=None, skiprows=1)
    df_tw = df_tw.dropna(subset=[1, 2])
    data['tail_water'] = {
        'Q_downstream': pd.to_numeric(df_tw.iloc[:, 2], errors='coerce').dropna().values,  # 下泄流量 (m³/s)
        'H_downstream': pd.to_numeric(df_tw.iloc[:, 1], errors='coerce').dropna().values,  # 下游水位 (m)
    }
    
    # 3. 泄流特性曲线
    df_sp = pd.read_excel(data_file, sheet_name='泄流特性曲线', header=None)
    # 第一行是闸门开度（可能包含标题），第二行开始是数据，第一列是水位
    # 找出水位列（第一列中的数值数据）
    water_level_col = pd.to_numeric(df_sp.iloc[:, 0], errors='coerce')
    valid_rows = ~np.isnan(water_level_col)
    data_start_row = np.where(valid_rows)[0][0] if np.any(valid_rows) else 1
    
    # 闸门开度在第一行（排除首尾列）
    gate_row = df_sp.iloc[0, 1:]
    gate_opening = pd.to_numeric(gate_row, errors='coerce').dropna().values
    if len(gate_opening) == 0:
        # 尝试从第2行获取
        gate_row = df_sp.iloc[1, 1:]
        gate_opening = pd.to_numeric(gate_row, errors='coerce').dropna().values
    
    # 水位和流量数据
    spillway_H = pd.to_numeric(df_sp.iloc[data_start_row:, 0], errors='coerce').dropna().values
    max_Q = pd.to_numeric(df_sp.iloc[data_start_row:, -1], errors='coerce').dropna().values
    
    # 流量表（中间列）
    flow_table_df = df_sp.iloc[data_start_row:, 1:-1]
    flow_table = flow_table_df.apply(pd.to_numeric, errors='coerce').values
    # 去除全NaN行
    valid_flow_rows = ~np.all(np.isnan(flow_table), axis=1)
    flow_table = flow_table[valid_flow_rows]
    spillway_H = spillway_H[:len(flow_table)]
    max_Q = max_Q[:len(flow_table)]
    
    data['spillway'] = {
        'upper_level_H': spillway_H,      # 上游水位
        'max_through_Q': max_Q,           # 最大下泄流量
        'level_spillway': spillway_H,     # 溢洪道水位
        'gate_opening': gate_opening,     # 闸门开度
        'spillway_flow_table': flow_table,  # L×M流量表
    }
    
    # 4. 出力特性曲线 (跳过标题行)
    df_po = pd.read_excel(data_file, sheet_name='出力特性曲线', header=None, skiprows=1)
    df_po = df_po.dropna(subset=[1, 2, 3])
    data['power_output'] = {
        'through_H': pd.to_numeric(df_po.iloc[:, 1], errors='coerce').dropna().values,  # 净水头 (m)
        'through_Q': pd.to_numeric(df_po.iloc[:, 3], errors='coerce').dropna().values,  # 发电流量 (m³/s)
        'output_N': pd.to_numeric(df_po.iloc[:, 2], errors='coerce').dropna().values,   # 出力 (MW)
    }
    
    # 5. 水头损失曲线 (跳过标题行)
    df_hl = pd.read_excel(data_file, sheet_name='水头损失曲线', header=None, skiprows=1)
    df_hl = df_hl.dropna(subset=[1, 2])
    data['head_loss'] = {
        'Q_loss': pd.to_numeric(df_hl.iloc[:, 1], errors='coerce').dropna().values,     # 流量 (m³/s)
        'dH_loss': pd.to_numeric(df_hl.iloc[:, 2], errors='coerce').dropna().values,    # 水头损失 (m)
    }
    
    # 6. 调度图控制水位
    df_dl = pd.read_excel(data_file, sheet_name='天一调度图控制水位', header=None)
    # 需要找到数据行（通常是数值行）
    # 清理数据：删除前10列，跳过标题行
    df_dl_data = df_dl.iloc[:, 10:]
    # 过滤只保留数值行（12个月的数据）
    numeric_rows = []
    for idx in range(len(df_dl_data)):
        row = df_dl_data.iloc[idx]
        try:
            # 尝试将最后两列转换为数值
            float(row.iloc[-2])
            float(row.iloc[-1])
            numeric_rows.append(idx)
        except (ValueError, TypeError):
            continue
    
    if len(numeric_rows) >= 12:
        df_dl_clean = df_dl_data.iloc[numeric_rows[:12]]
        data['dispatch_limits'] = {
            'H_max': pd.to_numeric(df_dl_clean.iloc[:, -2], errors='coerce').values,     # 允许最高水位/防洪限制水位
            'H_dead': pd.to_numeric(df_dl_clean.iloc[:, -1], errors='coerce').values,    # 允许最低水位/死水位
            'H_upper': pd.to_numeric(df_dl_clean.iloc[:, 2], errors='coerce').values,    # 上基本调度线
            'H_lower': pd.to_numeric(df_dl_clean.iloc[:, 3], errors='coerce').values,    # 下基本调度线
        }
    else:
        # 如果无法解析，使用默认值
        data['dispatch_limits'] = {
            'H_max': np.array([790.0] * 6 + [776.4] * 4 + [790.0] * 2),   # 汛期防洪限制，非汛期正常
            'H_dead': np.array([735.0] * 12),                            # 死水位
            'H_upper': np.array([764.2] * 12),                           # 上基本调度线
            'H_lower': np.array([750.0] * 12),                           # 下基本调度线
        }
    
    # 7. 未来径流数据
    df_if = pd.read_excel(data_file, sheet_name='历史径流', header=None, skiprows=2)
    # 找出数值数据的范围（假设入库流量在第6列之后）
    # 尝试过滤掉非数值行
    df_if_data = df_if.iloc[:, 6:]
    numeric_inflow_rows = []
    for idx in range(len(df_if_data)):
        row = df_if_data.iloc[idx]
        try:
            # 尝试将第一个数据转换为数值
            val = float(row.iloc[0])
            if not np.isnan(val):
                numeric_inflow_rows.append(idx)
        except (ValueError, TypeError):
            continue
    
    if len(numeric_inflow_rows) >= 2:
        df_if_clean = df_if_data.iloc[numeric_inflow_rows]
        inflow_matrix = df_if_clean.apply(pd.to_numeric, errors='coerce').values
        # 过滤掉全NaN的行
        valid_rows = ~np.all(np.isnan(inflow_matrix), axis=1)
        data['inflow'] = {
            'Q': inflow_matrix[valid_rows],  # 73年×12月的入库流量矩阵
        }
    else:
        raise ValueError("无法解析历史径流数据")
    
    return data


def get_inflow_sequence(data: Dict[str, Any]) -> np.ndarray:
    """
    将入库流量矩阵转换为时序序列
    
    Parameters
    ----------
    data : Dict[str, Any]
        load_reservoir_data()返回的数据字典
    
    Returns
    -------
    np.ndarray
        长度为n_years*12的入库流量序列 (m³/s)
    """
    Q = data['inflow']['Q']
    Q_flatten = Q.flatten('C')  # 按行展开 (C-order)
    Q_flatten_clean = Q_flatten[~np.isnan(Q_flatten)]   # 清理NaN
    return Q_flatten_clean  


def get_monthly_limits(data: Dict[str, Any], num_years: int) -> tuple:
    """
    获取全部调度期的月度水位限制
    
    Parameters
    ----------
    data : Dict[str, Any]
        load_reservoir_data()返回的数据字典
    num_years : int
        调度年数
    
    Returns
    -------
    tuple
        (H_max_full, H_dead_full) 两个数组，长度为num_years*12
    """
    H_max = data['dispatch_limits']['H_max']
    H_dead = data['dispatch_limits']['H_dead']
    
    H_max_full = np.tile(H_max, num_years)
    H_dead_full = np.tile(H_dead, num_years)
    
    return H_max_full, H_dead_full
