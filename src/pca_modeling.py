
"""
PCA建模模块：
- 全局或局部PCA拟合
- 使用Jackson & Mudholkar方法计算Q统计量的控制限Q_lim
"""

import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import eigvalsh
from scipy.stats import norm

def compute_q_lim(eigenvals_unused, alpha=0.95):
    """
    使用Jackson & Mudholkar方法计算Q控制限（Q_lim）。

    参数：
        eigenvals_unused (np.ndarray): 未保留的主成分特征值（残差子空间）
        alpha (float): 显著性水平，默认0.95

    返回：
        float: Q统计量的控制限
    """
    λ = eigenvals_unused
    θ1 = np.sum(λ)
    θ2 = np.sum(λ**2)
    θ3 = np.sum(λ**3)

    h0 = 1 - (2 * θ1 * θ3) / (3 * θ2**2)
    z_alpha = norm.ppf(alpha)

    if h0 <= 0 or np.isnan(h0):
        return np.inf

    Q_lim = θ1 * (
        z_alpha * np.sqrt(2 * θ2 * h0**2) / θ1
        + 1
        + (θ2 * h0 * (h0 - 1)) / θ1**2
    ) ** (1 / h0)

    return Q_lim

def train_local_pca(X_local, n_components=3, alpha=0.95):
    """
    拟合局部PCA模型并计算对应的Q_lim控制限。

    参数：
        X_local (np.ndarray): 原始光谱矩阵（一个局部模型的数据）
        n_components (int): PCA保留的主成分数量
        alpha (float): 显著性水平，控制Q_lim

    返回：
        dict: 包含PCA模型主成分、均值、Q_lim等信息的字典
    """
    #若一维则强制二维
    if X_local.ndim == 1:
        X_local = X_local.reshape(1, -1)
    
        
    # 手动中心化数据
    X_mean = np.mean(X_local, axis=0)
    X_centered = X_local - X_mean
    

    # 拟合PCA
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(X_centered)


    # 使用协方差矩阵计算完整特征值
    cov = np.cov(X_centered, rowvar=False)
    eigenvals_full = np.sort(eigvalsh(cov))[::-1]
    eigenvals_unused = eigenvals_full[n_components:]

    # 计算Q_lim
    Q_lim = compute_q_lim(eigenvals_unused, alpha=alpha)

    return {
        'P': pca.components_.T,         # 主成分方向矩阵
        'X_mean': X_mean,               # 数据均值
        'Q_lim': Q_lim,                 # 该local MSPC模型的Q统计量控制限
        'explained_variance': pca.explained_variance_
    }

