"""
可视化模块
==========

调度结果绑定与可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

# 设置中文字体
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
        
        Parameters
        ----------
        dispatch_config : DispatchConfig
            调度配置
        reservoir_config : ReservoirConfig
            水库配置
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
        
        Parameters
        ----------
        result : DispatchResult
            调度结果
        H_max_full : np.ndarray, optional
            各时段最高水位限制
        H_dead_full : np.ndarray, optional
            各时段最低水位限制
        """
        sim = result.simulation_result
        n_stages = len(sim.power)
        num_years = self.dispatch_config.num_years
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 水位过程
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
        
        # 年平均水位
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
        
        # 出力过程
        ax1 = axes[0]
        ax1.plot(range(1, n_stages + 1), sim.power, 'b-', linewidth=1.5, label='实际出力')
        ax1.axhline(y=guaranteed_output, color='r', linestyle='--', linewidth=1.5, label='保证出力')
        ax1.axhline(y=installed_output, color='g', linestyle='--', linewidth=1.5, label='装机容量')
        ax1.set_xlabel('时间 (月)')
        ax1.set_ylabel('出力 (MW)')
        ax1.set_title(f'实际出力变化过程 - {result.algorithm_name}')
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # 年发电量
        ax2 = axes[1]
        annual_energy = np.zeros(num_years)
        for y in range(num_years):
            idx_start = y * 12
            idx_end = (y + 1) * 12
            annual_energy[y] = np.sum(sim.energy[idx_start:idx_end]) / 1000  # GWh
        
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
        
        # 月流量
        ax1 = axes[0]
        ax1.plot(range(1, n_stages + 1), sim.outflow, 'b-', linewidth=1.5, label='出库流量')
        ax1.plot(range(1, n_stages + 1), sim.max_limit_Q, 'r--', linewidth=1, label='最大下泄能力')
        ax1.axhline(y=Q_min, color='g', linestyle='--', linewidth=1, label='最小下泄')
        ax1.set_xlabel('时间 (月)')
        ax1.set_ylabel('流量 (m^3/s)')
        ax1.set_title(f'出库流量变化过程 - {result.algorithm_name}')
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # 年总流量
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
        
        # 月弃水量
        ax1 = axes[0]
        ax1.plot(range(1, n_stages + 1), sim.abandoned_Q, 'r-', linewidth=1.5)
        ax1.set_xlabel('时间 (月)')
        ax1.set_ylabel('弃水流量 (m^3/s)')
        ax1.set_title(f'月弃水流量变化 - {result.algorithm_name}')
        ax1.grid(True)
        
        # 年弃水量
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

        # Log scale cannot display non-positive values.
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
