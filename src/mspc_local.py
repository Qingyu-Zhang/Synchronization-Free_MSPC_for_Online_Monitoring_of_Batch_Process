"""
local MSPC建模模块：
- 将按顺序排列的相邻两个cluster组合
- 使用这些数据拟合局部PCA模型（local MSPC）
"""

import numpy as np
from src.pca_modeling import train_local_pca

def build_local_models(X_augmented, cluster_path, cluster_index_map, n_components=3, alpha=0.95):
    """
    基于相邻的clusters构建一系列局部PCA模型。

    参数：
        X_augmented (np.ndarray): 拼接后的NOC批次原始光谱数据
        cluster_path (List[int]): 按顺序排列的聚类编号路径
        cluster_index_map (dict): 每个cluster对应的样本索引列表
        n_components (int): PCA保留的主成分数量
        alpha (float): Q统计量的显著性水平（默认0.95）

    返回：
        List[dict]: 每个局部PCA模型的参数字典，包含'P', 'X_mean', 'Q_lim'等
    """
    local_models = []

    for i in range(len(cluster_path) - 1):
        c1, c2 = cluster_path[i], cluster_path[i + 1]

        # 获取两个cluster的所有样本索引
        idx = cluster_index_map[c1] + cluster_index_map[c2]
        X_local = X_augmented[idx]

        # 拟合局部PCA模型
        model = train_local_pca(X_local, n_components=n_components, alpha=alpha)
        local_models.append(model)

    return local_models
