
"""
数据预处理模块：
- 读取原始数据
- 可选：标准化、SNV、滑动平均等预处理
"""

import pandas as pd
import numpy as np
import os

def load_and_concatenate_batches(folder_path):
    """
    将一个目录下所有 NOC 批次的 csv 文件拼接成一个大 DataFrame。
    每个文件为一个批次的光谱矩阵（按时间顺序），列为波长。

    参数：
        folder_path (str): 包含多个 batch 的 csv 文件夹路径

    返回：
        pd.DataFrame: 拼接后的大 DataFrame（所有 batch 按行堆叠）
    """
    all_batches = []

    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('.csv'):
            fpath = os.path.join(folder_path, fname)
            df = pd.read_csv(fpath, header=0)  # 假设第一行是列名（波长）
            all_batches.append(df)

    X_augmented_df = pd.concat(all_batches, axis=0, ignore_index=True)
    return X_augmented_df


def load_augmented_data(filepath):
    """
    从CSV文件中读取拼接后的NOC批次光谱数据，并返回为numpy数组。

    参数：
        filepath (str): 数据文件路径（CSV）

    返回：
        np.ndarray: shape = (n_samples, n_features) 的光谱数据矩阵
    """
    df = pd.read_csv(filepath)
    return df.to_numpy()

def standard_normal_variate(X):
    """
    对光谱矩阵应用SNV（Standard Normal Variate）标准正态变换。

    参数：
        X (np.ndarray): 光谱数据，shape = (n_samples, n_features)

    返回：
        np.ndarray: SNV处理后的矩阵
    """
    return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

def moving_average(X, window_size=3):
    """
    对光谱矩阵按每一列进行滑动平均滤波（降噪）。

    参数：
        X (np.ndarray): 原始光谱矩阵
        window_size (int): 滑动窗口大小（必须为奇数）

    返回：
        np.ndarray: 滤波后的矩阵，shape不变
    """
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(X, size=window_size, axis=0, mode='nearest')
