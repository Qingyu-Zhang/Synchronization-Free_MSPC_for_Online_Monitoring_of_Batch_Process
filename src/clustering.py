"""
聚类模块：
- 在PCA得分空间上执行KMeans聚类
- 基于欧氏距离和PC1坐标构建轨迹顺序
- 构造 cluster → 索引 映射
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

def perform_kmeans(T_all, n_clusters=10, random_state=0):
    """
    对PCA得分矩阵进行KMeans聚类。

    参数：
        T_all (np.ndarray): 所有样本在得分空间中的坐标，shape = (n_samples, A)
        n_clusters (int): 聚类数量
        random_state (int): 随机种子，保证可重复

    返回：
        labels (np.ndarray): 每个样本所属的聚类标签
        centers (np.ndarray): 每个聚类的中心点坐标，shape = (n_clusters, A)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(T_all)
    return kmeans.labels_, kmeans.cluster_centers_

def build_cluster_path(centers):
    """
    基于PC1方向 + 欧氏距离的贪心策略，构建聚类顺序路径。

    参数：
        centers (np.ndarray): 各聚类中心的得分空间坐标，shape = (G, A)

    返回：
        List[int]: cluster顺序路径，例如 [3, 0, 5, 2, ...]
    """
    G = len(centers)
    visited = []
    current = np.argmax(centers[:, 0])  # 从PC1最大值开始
    visited.append(current)

    while len(visited) < G:
        remaining = [i for i in range(G) if i not in visited]
        distances = pairwise_distances([centers[current]], centers[remaining])[0]
        next_idx = remaining[np.argmin(distances)]
        visited.append(next_idx)
        current = next_idx

    return visited

def build_cluster_index_map(labels):
    """
    构造 cluster label → 属于该聚类的样本索引列表。

    参数：
        labels (np.ndarray): 每个样本的聚类标签，shape = (n_samples,)

    返回：
        dict: key 为 cluster label，value 为样本索引 list
    """
    cluster_to_indices = {}
    for i, label in enumerate(labels):
        if label not in cluster_to_indices:
            cluster_to_indices[label] = []
        cluster_to_indices[label].append(i)
    return cluster_to_indices
