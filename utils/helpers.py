"""
辅助工具函数
============
"""
import numpy as np


def clamp(x: float, a: float, b: float) -> float:
    """
    将值限制在指定范围内
    
    Parameters
    ----------
    x : float
        输入值
    a : float
        下界
    b : float
        上界
    
    Returns
    -------
    float
        限制后的值
    """
    return min(max(x, a), b)


def clamp_array(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    将数组值限制在指定范围内
    
    Parameters
    ----------
    x : np.ndarray
        输入数组
    lb : np.ndarray
        下界数组
    ub : np.ndarray
        上界数组
    
    Returns
    -------
    np.ndarray
        限制后的数组
    """
    return np.clip(x, lb, ub)
