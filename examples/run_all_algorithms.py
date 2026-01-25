"""
水库调度算法运行示例
====================

演示如何使用各种水库调度算法
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from reservoir_dispatching.core.data_loader import load_reservoir_data, get_inflow_sequence, get_monthly_limits
from reservoir_dispatching.core.curve_fitting import CurveFitters
from reservoir_dispatching.algorithms.conventional import ConventionalDispatch
from reservoir_dispatching.algorithms.dp import DPDispatch
from reservoir_dispatching.algorithms.poa import POADispatch
from reservoir_dispatching.algorithms.dddp import DDDPDispatch
from reservoir_dispatching.algorithms.dhole import DholeDispatch
from reservoir_dispatching.algorithms.pso import PSODispatch
from reservoir_dispatching.visualization.plots import DispatchPlotter
from reservoir_dispatching.config import (
    ReservoirConfig, DispatchConfig, AlgorithmConfig,
    DEFAULT_RESERVOIR, DEFAULT_DISPATCH, DEFAULT_ALGORITHM
)


def run_conventional_dispatch():
    """运行常规调度算法"""
    print("\n" + "=" * 60)
    print("运行常规调度算法")
    print("=" * 60)
    
    algo = ConventionalDispatch()
    result = algo.run()
    
    # 可视化
    plotter = DispatchPlotter()
    plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
    plotter.plot_power_energy(result)
    
    return result


def run_dp_dispatch():
    """运行动态规划算法"""
    print("\n" + "=" * 60)
    print("运行动态规划(DP)算法")
    print("=" * 60)
    
    # 使用更粗的离散步长以加快计算（生产环境可用0.5m）
    algo_config = AlgorithmConfig(d_water_level=2.0)
    
    algo = DPDispatch(algorithm_config=algo_config)
    result = algo.run()
    
    plotter = DispatchPlotter()
    plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
    plotter.plot_power_energy(result)
    
    return result


def run_poa_dispatch(initial_trajectory=None):
    """运行POA逐步优化算法"""
    print("\n" + "=" * 60)
    print("运行POA逐步优化算法")
    print("=" * 60)
    
    algo = POADispatch(initial_trajectory=initial_trajectory)
    result = algo.run()
    
    plotter = DispatchPlotter()
    plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
    
    return result


def run_dddp_dispatch(initial_trajectory=None):
    """运行DDDP离散微分动态规划算法"""
    print("\n" + "=" * 60)
    print("运行DDDP离散微分动态规划算法")
    print("=" * 60)
    
    algo = DDDPDispatch(initial_trajectory=initial_trajectory)
    result = algo.run()
    
    plotter = DispatchPlotter()
    plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
    
    if result.convergence_history is not None:
        plotter.plot_convergence(result)
    
    return result


def run_dhole_dispatch(max_iter=500, pop_size=30):
    """运行豺狼优化算法（演示模式，减少迭代次数）"""
    print("\n" + "=" * 60)
    print("运行豺狼优化算法(DOA)")
    print("=" * 60)
    
    algo_config = AlgorithmConfig(
        max_iterations=max_iter,
        population_size=pop_size,
        random_seed=1
    )
    
    algo = DholeDispatch(algorithm_config=algo_config)
    result = algo.run()
    
    plotter = DispatchPlotter()
    plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
    plotter.plot_convergence(result)
    
    return result


def run_pso_dispatch(max_iter=500, pop_size=30):
    """运行粒子群优化算法（演示模式，减少迭代次数）"""
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
    
    plotter = DispatchPlotter()
    plotter.plot_water_level_process(result, algo.H_max_full, algo.H_dead_full)
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
    print("水库调度算法库 - 运行示例")
    print("=" * 60)
    
    # 检查数据文件
    from reservoir_dispatching.config import DATA_FILE
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
    print("5. 运行豺狼优化算法(DOA)")
    print("6. 运行粒子群优化算法(PSO)")
    print("7. 算法对比")
    print("0. 退出")
    
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
        # 先运行常规调度获取初始轨迹
        conv_result = run_conventional_dispatch()
        run_poa_dispatch(conv_result.water_level_trajectory)
    elif choice == 4:
        conv_result = run_conventional_dispatch()
        run_dddp_dispatch(conv_result.water_level_trajectory)
    elif choice == 5:
        run_dhole_dispatch()
    elif choice == 6:
        run_pso_dispatch()
    elif choice == 7:
        compare_algorithms()
    else:
        print("再见!")


if __name__ == "__main__":
    main()
