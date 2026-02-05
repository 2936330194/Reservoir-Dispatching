"""
水库调度算法 - 主程序
====================

演示如何使用各种水库调度算法

作者: hu hao
"""
import sys
import os
import numpy as np

# 将当前目录添加到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import load_reservoir_data, get_inflow_sequence, get_monthly_limits
from core.curve_fitting import CurveFitters
from algorithms.conventional import ConventionalDispatch
from algorithms.dp import DPDispatch
from algorithms.poa import POADispatch
from algorithms.dddp import DDDPDispatch
from algorithms.dhole import DholeDispatch
from algorithms.pso import PSODispatch
from visualization.plots import DispatchPlotter
from config import (
    ReservoirConfig, DispatchConfig, AlgorithmConfig,
    DEFAULT_RESERVOIR, DEFAULT_DISPATCH, DEFAULT_ALGORITHM, DATA_FILE
)

# 水位过程线保存目录
TRAJECTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'trajectories')


def save_trajectory(trajectory: np.ndarray, name: str) -> str:
    """
    保存水位过程线到文件
    
    Parameters
    ----------
    trajectory : np.ndarray
        水位过程线数组
    name : str
        保存名称（不含扩展名）
    
    Returns
    -------
    str
        保存的文件路径
    """
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    filepath = os.path.join(TRAJECTORY_DIR, f'{name}.npy')
    np.save(filepath, trajectory)
    print(f"水位过程线已保存至: {filepath}")
    return filepath


def load_trajectory(name: str) -> np.ndarray:
    """
    从文件加载水位过程线
    
    Parameters
    ----------
    name : str
        保存时使用的名称（不含扩展名）
    
    Returns
    -------
    np.ndarray
        水位过程线数组
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
    
    Parameters
    ----------
    show_plot : bool
        是否显示可视化图表，默认为True
    save_result : bool
        是否保存水位过程线，默认为False
    """
    print("\n" + "=" * 60)
    print("运行常规调度算法")
    print("=" * 60)
    
    algo = ConventionalDispatch()
    result = algo.run()
    
    # 保存水位过程线（可选）
    if save_result:
        save_trajectory(result.water_level_trajectory, 'conventional')
    
    # 可视化（可选）
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
        plotter.plot_power_energy(result)
    
    return result


def run_dp_dispatch(show_plot=True, save_result=True):
    """运行动态规划算法
    
    Parameters
    ----------
    save_result : bool
        是否保存水位过程线，默认为False
    """
    print("\n" + "=" * 60)
    print("运行动态规划(DP)算法")
    print("=" * 60)
    
    # 使用更粗的离散步长以加快计算（生产环境可用0.5m）
    algo_config = AlgorithmConfig(d_water_level=0.5)
    
    algo = DPDispatch(algorithm_config=algo_config)
    result = algo.run()
    
    # 保存水位过程线（可选）
    if save_result:
        save_trajectory(result.water_level_trajectory, 'dp')
    
    # 可视化（可选）
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
        plotter.plot_power_energy(result)
    
    return result


def run_poa_dispatch(initial_trajectory=None, show_plot=True, save_result=True):
    """运行POA逐步优化算法
    
    Parameters
    ----------
    initial_trajectory : np.ndarray, optional
        初始水位轨迹
    show_plot : bool
        是否显示可视化图表，默认为True
    save_result : bool
        是否保存水位过程线，默认为True
    """
    print("\n" + "=" * 60)
    print("运行POA逐步优化算法")
    print("=" * 60)
    
    algo = POADispatch(initial_trajectory=initial_trajectory)
    result = algo.run()
    
    # 保存水位过程线（可选）
    if save_result:
        save_trajectory(result.water_level_trajectory, 'poa')
    
    # 可视化（可选）
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
    
    return result


def run_dddp_dispatch(initial_trajectory=None, show_plot=True, save_result=True):
    """运行DDDP离散微分动态规划算法
    
    Parameters
    ----------
    initial_trajectory : np.ndarray, optional
        初始水位轨迹
    show_plot : bool
        是否显示可视化图表，默认为True
    save_result : bool
        是否保存水位过程线，默认为True
    """
    print("\n" + "=" * 60)
    print("运行DDDP离散微分动态规划算法")
    print("=" * 60)
    
    algo = DDDPDispatch(initial_trajectory=initial_trajectory)
    result = algo.run()
    
    # 保存水位过程线（可选）
    if save_result:
        save_trajectory(result.water_level_trajectory, 'dddp')
    
    # 可视化（可选）
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
        
        if result.convergence_history is not None:
            plotter.plot_convergence(result)
    
    return result


def run_dhole_dispatch(max_iter=500, pop_size=30, show_plot=True, save_result=True):
    """运行豺狼优化算法（演示模式，减少迭代次数）
    
    Parameters
    ----------
    max_iter : int
        最大迭代次数，默认为500
    pop_size : int
        种群大小，默认为30
    show_plot : bool
        是否显示可视化图表，默认为True
    save_result : bool
        是否保存水位过程线，默认为True
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
    
    # 保存水位过程线（可选）
    if save_result:
        save_trajectory(result.water_level_trajectory, 'dhole')
    
    # 可视化（可选）
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
        
        if result.convergence_history is not None:
            plotter.plot_convergence(result)
    
    return result


def run_pso_dispatch(max_iter=500, pop_size=30, show_plot=True, save_result=True):
    """运行粒子群优化算法（演示模式，减少迭代次数）
    
    Parameters
    ----------
    max_iter : int
        最大迭代次数，默认为500
    pop_size : int
        种群大小，默认为30
    show_plot : bool
        是否显示可视化图表，默认为True
    save_result : bool
        是否保存水位过程线，默认为True
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
    
    # 保存水位过程线（可选）
    if save_result:
        save_trajectory(result.water_level_trajectory, 'pso')
    
    # 可视化（可选）
    if show_plot:
        plotter = DispatchPlotter()
        plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
        
        if result.convergence_history is not None:
            plotter.plot_convergence(result)
    
    return result


def compare_algorithms():
    """比较不同算法的结果"""
    print("\n" + "=" * 60)
    print("算法对比")
    print("=" * 60)
    
    results = {}
    
    # 1. 常规调度
    print("\n[1/3] 运行常规调度...")
    algo_conv = ConventionalDispatch()
    results['常规调度'] = algo_conv.run()
    
    # 2. DP（使用粗离散快速演示）
    print("\n[2/3] 运行DP...")
    algo_config_dp = AlgorithmConfig(d_water_level=2.0)
    algo_dp = DPDispatch(algorithm_config=algo_config_dp)
    results['DP'] = algo_dp.run()
    
    # 3. POA（基于常规调度结果）
    print("\n[3/3] 运行POA...")
    algo_poa = POADispatch(
        initial_trajectory=results['常规调度'].water_level_trajectory
    )
    results['POA'] = algo_poa.run()
    
    # 打印比较结果
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
    
    # 检查数据文件
    if not os.path.exists(DATA_FILE):
        print(f"错误: 数据文件不存在: {DATA_FILE}")
        print("请确保已将 data_ty.xls 复制到 data/ 目录")
        return
    
    print(f"数据文件: {DATA_FILE}")
    
    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 运行常规调度算法")
    print("2. 运行动态规划(DP)算法")
    print("3. 运行POA逐步优化算法")
    print("4. 运行DDDP算法")
    print("5. 运行豺狼优化算法(DHOLE)")
    print("6. 运行粒子群优化算法(PSO)")
    print("7. 算法对比")
    print("0. 退出")
    
    # 显示已保存的水位过程线
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
        # POA: 尝试加载已保存的轨迹，否则运行DP调度
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
        # DDDP: 尝试加载已保存的轨迹，否则运行DP调度
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
