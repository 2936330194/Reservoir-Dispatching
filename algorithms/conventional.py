"""
常规调度算法
============

基于调度图的常规运行调度
"""
import numpy as np
from typing import Dict, Any

from .base import BaseDispatchAlgorithm, DispatchResult
from config import ReservoirConfig, DispatchConfig, AlgorithmConfig
from core.simulation import SimulationResult


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
        
        # 获取调度图水位线
        self.H_upper = self.data['dispatch_limits']['H_upper']  # 上基本调度线
        self.H_lower = self.data['dispatch_limits']['H_lower']  # 下基本调度线
        
        # 迭代参数
        self.tolerance = 1.0  # 出力计算容差 (MW)
        self.max_iterations = 100
    
    def _determine_planned_power(self, water_level: float, month: int) -> float:
        """
        根据调度图确定计划出力
        
        Parameters
        ----------
        water_level : float
            当前水位 (m)
        month : int
            当前月份 (0-11)
        
        Returns
        -------
        float
            计划出力 (MW)
        """
        guaranteed_output = self.dispatch_config.guaranteed_output
        H_max = self.data['dispatch_limits']['H_max'][month]
        H_upper = self.H_upper[month]
        H_lower = self.H_lower[month]
        H_dead = self.data['dispatch_limits']['H_dead'][month]
        
        if water_level >= H_max:
            return 1.5 * guaranteed_output  # 1.5倍保证出力
        elif water_level >= H_upper:
            return 1.2 * guaranteed_output  # 1.2倍保证出力
        elif water_level >= H_lower:
            return 1.0 * guaranteed_output  # 1.0倍保证出力
        elif water_level >= H_dead:
            return 0.75 * guaranteed_output  # 0.75倍保证出力
        else:
            return 0  # 水位低于死水位，不能过流发电
    
    def _run_algorithm(self) -> np.ndarray:
        """
        运行常规调度算法
        
        Returns
        -------
        np.ndarray
            水位轨迹（长度为n_stages+1）
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
        
        # 平均出力系数 (来自MATLAB)
        average_output_coefficient = 8.6
        
        # 初始化结果数组 - 使用二维数组，与MATLAB一致
        water_level_2d = np.zeros((num_years, num_months + 1))
        storage_2d = np.zeros((num_years, num_months + 1))
        
        # 调度图水位限制
        H_max = self.data['dispatch_limits']['H_max']  # 12个月
        H_dead = self.data['dispatch_limits']['H_dead']  # 12个月
        
        # 设置初始条件
        water_level_2d[0, 0] = initial_water_level
        storage_2d[0, 0] = self.fitters.fit_V(initial_water_level)
        
        # 主计算循环
        for year in range(num_years):
            if (year + 1) % 10 == 0:
                print(f"  处理第 {year + 1} 年...")
            
            for month in range(num_months):
                current_month_idx = month  # 0-11
                
                # 1. 获取当前月初水位和库容
                current_water_level = water_level_2d[year, month]
                current_storage = storage_2d[year, month]
                t = year * num_months + month
                current_inflow = self.inflow[t]
                
                # 2. 根据调度图确定计划出力
                planned_power = self._determine_planned_power(current_water_level, month)
                
                # 3. 初始化迭代变量
                if current_water_level > 0:
                    Q_out_guess = planned_power * 1e3 / (average_output_coefficient * current_water_level)
                else:
                    Q_out_guess = Q_min
                iteration_count = 0
                converged = False
                
                # 初始化输出变量
                end_water_level = current_water_level
                end_storage = current_storage
                actual_power = 0
                Q_power = 0
                Q_abandoned = 0
                open_level = 0
                max_limit_Q = num_turbines * max_generation_flow
                
                # 迭代计算，直到找到满足出力要求的出库流量
                while iteration_count < self.max_iterations and not converged:
                    iteration_count += 1
                    
                    # 3.1 计算时段末库容和水位
                    end_storage = current_storage + (current_inflow - Q_out_guess) * seconds_per_month / 1e8
                    if end_storage < 0:
                        end_storage = self.data['water_volume']['V_values'][0]  # 负数库容修正
                    end_water_level = self.fitters.fit_H(end_storage)
                    
                    # 3.2 检查水位约束并调整
                    # 如果超过最高水位，强制设置为最高水位
                    if end_water_level > H_max[current_month_idx]:
                        end_water_level = H_max[current_month_idx]
                        end_storage = self.fitters.fit_V(end_water_level)
                        # 重新计算需要的出库流量
                        Q_out_guess = current_inflow - (end_storage - current_storage) * 1e8 / seconds_per_month
                        # 计算此时均上游水位、下游水位、水头损失、净水头
                        avg_water_level = (current_water_level + end_water_level) / 2
                        downstream_water_level = self.fitters.fit_H_downstream(Q_out_guess)
                        head_loss = num_turbines * self.fitters.fit_dH_loss(Q_out_guess / num_turbines)
                        net_head = avg_water_level - downstream_water_level - head_loss
                        
                        # 计算此时的最大下泄能力
                        if avg_water_level < height_of_spillway:
                            max_limit_Q = num_turbines * max_generation_flow
                        else:
                            max_limit_Q = num_turbines * max_generation_flow + num_spillway * self.fitters.fit_spill_Q(avg_water_level)
                        
                        # 如果下泄流量大于机组最大引用流量，则发生弃水
                        if Q_out_guess > num_turbines * max_generation_flow:
                            # 重算实际功率
                            actual_power = num_turbines * self.fitters.fit_output_N(max_generation_flow, net_head)
                            # 计算满足预想出力所需的发电流量
                            Q_power = num_turbines * max_generation_flow
                            # 计算弃水量
                            Q_abandoned = Q_out_guess - Q_power
                            # 根据平均水位与溢洪道高程的关系调整闸门开度
                            if avg_water_level <= height_of_spillway:
                                if current_water_level < height_of_spillway and end_water_level < height_of_spillway:
                                    # 无法弃水
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
                            # 重算实际功率
                            actual_power = num_turbines * self.fitters.fit_output_N(Q_out_guess / num_turbines, net_head)
                            if actual_power >= planned_power:
                                # 计算满足预想出力所需的发电流量
                                Q_power = Q_out_guess
                                # 计算弃水量
                                Q_abandoned = 0
                                open_level = 0
                                break
                            else:
                                # 检查是否收敛
                                if abs(actual_power - planned_power) < self.tolerance:
                                    converged = True
                                else:
                                    # 调整出库流量猜测值 (基于出力误差)
                                    if abs(actual_power) > 1e-6:
                                        Q_out_guess = Q_out_guess * (planned_power / abs(actual_power))
                                    continue
                        continue
                    
                    # 如果低于死水位，强制设置为死水位
                    if end_water_level < H_dead[current_month_idx]:
                        end_water_level = H_dead[current_month_idx]
                        end_storage = self.fitters.fit_V(end_water_level)
                        # 重新计算需要的出库流量
                        Q_out_guess = current_inflow - (end_storage - current_storage) * 1e8 / seconds_per_month
                        # 计算此时均上游水位、下游水位、水头损失、净水头、实际出力
                        avg_water_level = (current_water_level + end_water_level) / 2
                        downstream_water_level = self.fitters.fit_H_downstream(Q_out_guess)
                        head_loss = num_turbines * self.fitters.fit_dH_loss(Q_out_guess / num_turbines)
                        net_head = avg_water_level - downstream_water_level - head_loss
                        actual_power = num_turbines * self.fitters.fit_output_N(Q_out_guess / num_turbines, net_head)
                        
                        # 计算此时的最大下泄能力
                        if avg_water_level < height_of_spillway:
                            max_limit_Q = num_turbines * max_generation_flow
                        else:
                            max_limit_Q = num_turbines * max_generation_flow + num_spillway * self.fitters.fit_spill_Q(avg_water_level)
                        
                        if actual_power <= planned_power:
                            # 计算满足预想出力所需的发电流量
                            Q_power = Q_out_guess
                            Q_abandoned = 0
                            open_level = 0
                            break
                        else:
                            # 检查是否收敛
                            if abs(actual_power - planned_power) < self.tolerance:
                                converged = True
                            else:
                                # 调整出库流量猜测值 (基于出力误差)
                                if abs(actual_power) > 1e-6:
                                    Q_out_guess = Q_out_guess * (planned_power / abs(actual_power))
                                continue
                        continue
                    
                    # 高于死水位、低于限制水位的正常情况
                    # 计算此时均上游水位、下游水位、水头损失、净水头、实际出力
                    avg_water_level = (current_water_level + end_water_level) / 2
                    downstream_water_level = self.fitters.fit_H_downstream(Q_out_guess)
                    head_loss = num_turbines * self.fitters.fit_dH_loss(Q_out_guess / num_turbines)
                    net_head = avg_water_level - downstream_water_level - head_loss
                    actual_power = num_turbines * self.fitters.fit_output_N(Q_out_guess / num_turbines, net_head)
                    
                    # 计算此时的最大下泄能力
                    if avg_water_level < height_of_spillway:
                        max_limit_Q = num_turbines * max_generation_flow
                    else:
                        max_limit_Q = num_turbines * max_generation_flow + num_spillway * self.fitters.fit_spill_Q(avg_water_level)
                    
                    # 检查是否收敛
                    if abs(actual_power - planned_power) < self.tolerance:
                        converged = True
                        # 应用最小下泄流量约束
                        if Q_out_guess <= Q_min:
                            Q_out_guess = Q_min
                            end_storage = current_storage + (current_inflow - Q_out_guess) * seconds_per_month / 1e8
                            end_water_level = self.fitters.fit_H(end_storage)
                            # 重新计算
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
                        # 调整出库流量猜测值 (基于出力误差)
                        if abs(actual_power) > 1e-6:
                            Q_out_guess = Q_out_guess * (planned_power / abs(actual_power))
                        # 计算满足预想出力所需的发电流量
                        Q_power = Q_out_guess
                        Q_abandoned = 0
                        open_level = 0
                
                # 4. 存储结果
                water_level_2d[year, month + 1] = end_water_level
                storage_2d[year, month + 1] = end_storage
            
            # 设置下一年初的水位和库容
            if year < num_years - 1:
                water_level_2d[year + 1, 0] = water_level_2d[year, num_months]
                storage_2d[year + 1, 0] = storage_2d[year, num_months]
        
        # 将二维数组转换为一维序列（月初水位序列）
        # 构造完整的水位轨迹：初水位 + 各时段末水位
        water_level_seq = np.zeros(self.n_stages + 1)
        water_level_seq[0] = initial_water_level
        
        for year in range(num_years):
            for month in range(num_months):
                t = year * num_months + month
                water_level_seq[t + 1] = water_level_2d[year, month + 1]
        
        return water_level_seq

