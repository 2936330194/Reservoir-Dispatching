"""
仿真模块
========

根据水位序列计算发电量、出流量、弃水量等调度结果
"""
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from .curve_fitting import CurveFitters
from config import ReservoirConfig, DispatchConfig, DEFAULT_RESERVOIR, DEFAULT_DISPATCH


@dataclass
class SimulationResult:
    """仿真结果数据类"""
    total_energy: float              # 总发电量 (MWh)
    water_level: np.ndarray          # 水位序列 (n_stages+1,)
    storage: np.ndarray              # 库容序列 (n_stages+1,)
    outflow: np.ndarray              # 出库流量 (n_stages,)
    generated_Q: np.ndarray          # 发电流量 (n_stages,)
    abandoned_Q: np.ndarray          # 弃水流量 (n_stages,)
    power: np.ndarray                # 出力 (n_stages,)
    energy: np.ndarray               # 各时段发电量 (n_stages,)
    max_limit_Q: np.ndarray          # 最大下泄能力 (n_stages,)
    gate_opening: np.ndarray         # 闸门开度 (n_stages,)


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
    
    Parameters
    ----------
    H_seq : np.ndarray
        长度为 n_stages+1 的期末水位序列（包含初始水位）
    inflow : np.ndarray
        长度为 n_stages 的入库流量序列 (m³/s)
    fitters : CurveFitters
        曲线拟合器对象
    reservoir_config : ReservoirConfig, optional
        水库配置，默认使用DEFAULT_RESERVOIR
    dispatch_config : DispatchConfig, optional
        调度配置，默认使用DEFAULT_DISPATCH
    Q_min : float, optional
        最小下泄流量，默认从dispatch_config获取
    
    Returns
    -------
    SimulationResult
        仿真结果
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
    
    # 初始化结果数组
    water_level = np.array(H_seq, dtype=float)
    storage = np.zeros(n_stages + 1)
    outflow = np.zeros(n_stages)
    generated_Q = np.zeros(n_stages)
    abandoned_Q = np.zeros(n_stages)
    power = np.zeros(n_stages)
    energy = np.zeros(n_stages)
    max_limit_Q = np.zeros(n_stages)
    gate_opening = np.zeros(n_stages)
    
    # 计算初始库容
    storage[0] = fitters.fit_V(H_seq[0])
    
    for t in range(n_stages):
        H_start = H_seq[t]
        H_end = H_seq[t + 1]
        V_start = fitters.fit_V(H_start)
        V_end = fitters.fit_V(H_end)
        
        avg_H = 0.5 * (H_start + H_end)
        
        # 水量平衡求总出库 (m³/s)
        Qout = inflow[t] - (V_end - V_start) * 1e8 / seconds_per_month
        
        if Qout < 0:
            Qout = 0
            print(f"警告: 第{t+1}个时段优化结果不可取（负出流）！")
        
        if Qout < Q_min:
            print(f"警告: 第{t+1}个时段出流{Qout:.2f}小于最小下泄流量{Q_min}！")
        
        # 最大下泄能力
        if avg_H < height_of_spillway:
            maxQ = num_turbines * max_generation_flow
        else:
            maxQ = num_turbines * max_generation_flow + num_spillway * fitters.fit_spill_Q(avg_H)
        
        # 计算出力与弃水
        if Qout <= num_turbines * max_generation_flow:
            # 不弃水
            downstream_H = fitters.fit_H_downstream(Qout)
            generated = Qout
            abandoned = 0
            opening = 0
            head_loss = num_turbines * fitters.fit_dH_loss(generated / num_turbines)
            net_head = avg_H - downstream_H - head_loss
            actual_power = num_turbines * fitters.fit_output_N(
                max(generated / num_turbines, 1e-6), net_head)
        elif Qout <= maxQ:
            # 弃水
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
            # 按最大能力计算，弃不了的水就自求多福
            downstream_H = fitters.fit_H_downstream(maxQ)
            generated = num_turbines * max_generation_flow
            abandoned = maxQ - generated
            opening = fitters.fit_gate_opening(avg_H, abandoned / num_spillway)
            head_loss = num_turbines * fitters.fit_dH_loss(generated / num_turbines)
            net_head = avg_H - downstream_H - head_loss
            actual_power = num_turbines * fitters.fit_output_N(
                generated / num_turbines, net_head)
        
        # 确保出力非负
        if actual_power < 0:
            actual_power = 0
        
        # 发电量 (MWh)
        E = actual_power * 30.4 * 24
        
        # 记录结果
        storage[t + 1] = V_end
        outflow[t] = Qout
        generated_Q[t] = generated
        abandoned_Q[t] = abandoned
        power[t] = actual_power
        energy[t] = E
        max_limit_Q[t] = maxQ
        gate_opening[t] = opening
    
    # 总发电量
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
    
    Parameters
    ----------
    Z_start : float
        阶段初水位 (m)
    Z_end : float
        阶段末水位 (m)
    Q_in : float
        入库流量 (m³/s)
    fitters : CurveFitters
        曲线拟合器
    reservoir_config : ReservoirConfig, optional
        水库配置
    dispatch_config : DispatchConfig, optional
        调度配置
    penalty_coefficient : float
        保证出力惩罚系数
    H_dead : float, optional
        死水位限制
    H_max : float, optional
        最高水位限制
    
    Returns
    -------
    float
        发电量 (MWh)，如果违反约束返回负数表示惩罚
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
    
    # 检查水位约束
    if H_dead is not None and H_max is not None:
        if Z_start < H_dead or Z_start > H_max or Z_end < H_dead or Z_end > H_max:
            return -1e6  # 惩罚
    
    # 计算库容
    V_start = fitters.fit_V(Z_start)
    V_end = fitters.fit_V(Z_end)
    
    # 计算下泄流量（水量平衡）
    Q_release = (V_start - V_end) * 1e8 / seconds_per_month + Q_in
    
    # 检查最小下泄流量约束
    if Q_release < Q_min:
        return -1e6  # 惩罚
    
    # 计算平均水位
    Z_avg = (Z_start + Z_end) / 2
    
    # 计算最大下泄能力
    if Z_avg < height_of_spillway:
        Q_max = num_turbines * max_generation_flow
    else:
        Q_max = num_turbines * max_generation_flow + num_spillway * fitters.fit_spill_Q(Z_avg)
    
    # 检查最大下泄流量约束
    if Q_release > Q_max:
        return -1e6  # 惩罚
    
    # 计算尾水位
    H_tail = fitters.fit_H_downstream(Q_release)
    
    # 确定发电流量
    if Q_release <= num_turbines * max_generation_flow:
        Q_gen = Q_release
    else:
        Q_gen = num_turbines * max_generation_flow
    
    # 单机流量
    q_unit = Q_gen / num_turbines
    
    # 计算水头损失
    dH_loss = num_turbines * fitters.fit_dH_loss(q_unit)
    
    # 计算净水头
    H_net = Z_avg - H_tail - dH_loss
    
    if H_net <= 0:
        return 0
    
    # 计算出力
    N = num_turbines * fitters.fit_output_N(q_unit, H_net)
    
    # 保证出力惩罚
    if N < guaranteed_output:
        N = N - penalty_coefficient * (guaranteed_output - N)
    
    if N < 0:
        N = 0
    
    # 计算发电量（MWh）
    energy = N * 30.4 * 24
    
    return energy
