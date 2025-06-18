"""
异常诊断模块：
- 计算新点在某个local模型下的残差向量，用于贡献图
- 支持两种模式：当前点自身的最佳模型 / 最近一次正常点的最佳模型
"""

import numpy as np

def get_contribution_vector(x_k, models, Qr_k_all, use_latest_noc_model=False, latest_valid_model_idx=None):
    """
    获取新观测点x_k的残差向量（用于绘制贡献图）。

    参数：
        x_k (np.ndarray): shape = (J,)，当前光谱观测点
        models (List[dict]): 所有local MSPC模型，含'P', 'X_mean', 'Q_lim'
        Qr_k_all (List[float]): 当前点在所有模型下的Qr值
        use_latest_noc_model (bool): 是否使用最近一个正常点(NOC)的模型（论文中好像是这么设计的）
        latest_valid_model_idx (int or None): 最近正常点使用的模型索引（如果开启上面的选项）

    返回：
        e (np.ndarray): shape = (J,)，该点的残差向量
        model_idx (int): 实际使用的模型编号 (Qr最小的local MSPC模型索引)
    """
    if use_latest_noc_model:
        if latest_valid_model_idx is None:
            raise ValueError("latest_valid_model_idx must be provided if use_latest_noc_model is True.")
        model_idx = latest_valid_model_idx
    else:
        model_idx = int(np.argmin(Qr_k_all))

    model = models[model_idx]
    P = model['P']
    X_mean = model['X_mean']
    X_std = model['X_std']

    x_standardized = (x_k - X_mean) / X_std     # 标准化
    t = x_standardized @ P                      # 投影到主成分
    x_hat = (t @ P.T) * X_std + X_mean          # 重构（到原始单位）
    e = x_k - x_hat                             # 计算重构误差/残差（在原始单位下）

    return e, model_idx

