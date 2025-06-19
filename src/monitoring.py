"""
实时监控模块：
- 对新样本点进行监测，计算其在各local MSPC模型下的Qr值
- 判断该点是否为异常（所有Qr均大于1时）
"""

import numpy as np

def monitor_new_point(x_k, models):
    """
    对新观测点x_k进行监测，判断是否异常，并返回所有local模型下的Qr值。

    参数：
        x_k (np.ndarray): shape = (J,)，新的光谱观测点
        models (List[dict]): 所有local PCA模型，每个模型包含'P', 'X_mean', 'Q_lim'

    返回：
        is_faulty (bool): 如果该点在所有模型下的Qr最小值仍大于1，则为True（异常）
        Qr_all (np.ndarray): 所有模型下的Qr值，shape = (G-1,)
    """
    Qr_all = []

    for model in models:
        P = model['P']
        X_mean = model['X_mean']
        X_std = model['X_std']
        Q_lim = model['Q_lim']

        x_standardized = (x_k - X_mean) / X_std     #标准化
        t = x_standardized @ P                      #投影
        x_hat = (t @ P.T) * X_std + X_mean          #重构观测点x_k (在原始单位下）
        e = x_k - x_hat                             #计算重构误差向量e（原始单位）
        Q = np.sum(e ** 2)                          #计算Q值
        # x_hat_standardized = t @ P.T                              #在标准化下重构观测点x_standardized（在标准化单位下）
        # e_standardized = x_standardized - x_hat_standardized      #在标准单位下计算重构误差向量e_standardized（标准化单位）
        # Q = np.sum(e_standardized ** 2)                           #用标准化的误差计算Q值
        Qr = Q / Q_lim
        Qr_all.append(Qr)

    Qr_all = np.array(Qr_all)
    is_faulty = np.min(Qr_all) > 1

    return is_faulty, Qr_all
