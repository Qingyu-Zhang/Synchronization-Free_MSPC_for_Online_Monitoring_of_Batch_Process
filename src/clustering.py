"""
聚类模块：
- 在PCA得分空间上执行KMeans聚类
- 基于欧氏距离和PC1坐标构建轨迹顺序
- 构造 cluster → 索引 映射
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from collections import deque


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
    current = np.argmin(centers[:, 0])  # 从PC1最小值开始
    visited.append(current)

    while len(visited) < G:
        remaining = [i for i in range(G) if i not in visited]
        distances = pairwise_distances([centers[current]], centers[remaining])[0]
        next_idx = remaining[np.argmin(distances)]
        visited.append(next_idx)
        current = next_idx

    return visited

def build_cluster_path_mst(centers: np.ndarray) -> list:
    """
    基于最小生成树（MST）+ 最长路径启发 + 最近邻插入 的聚类路径构建方法。

    参数：
        centers (np.ndarray): 每个聚类的中心点坐标，shape = (G, A)

    返回：
        List[int]: 聚类顺序路径，例如 [3, 0, 5, 2, ...]
    """
    G = len(centers)
    if G <= 2:
        return list(range(G))

    # Step 1: 计算欧氏距离矩阵并构建MST（无向图）
    dist_matrix = squareform(pdist(centers))
    mst = minimum_spanning_tree(dist_matrix).toarray()
    graph = mst + mst.T  # 转为无向图

    # Step 2: BFS 找到最长路径的两个端点
    def bfs_farthest(start):
        visited = {start}
        queue = deque([(start, 0)])
        farthest = (start, 0)
        while queue:
            node, dist = queue.popleft()
            if dist > farthest[1]:
                farthest = (node, dist)
            for neighbor in np.where(graph[node] > 0)[0]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + graph[node, neighbor]))
        return farthest

    A, _ = bfs_farthest(0)
    B, _ = bfs_farthest(A)

    # Step 3: 回溯获取 A 到 B 的路径（主干路径）
    def get_path(u, v):
        parent = {u: None}
        queue = deque([u])
        while queue:
            curr = queue.popleft()
            if curr == v:
                break
            for neighbor in np.where(graph[curr] > 0)[0]:
                if neighbor not in parent:
                    parent[neighbor] = curr
                    queue.append(neighbor)
        path = []
        curr = v
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        return path[::-1]

    main_path = get_path(A, B)

    # Step 4: 插入未包含的剩余节点（最近邻插入）
    full_order = main_path.copy()
    unused = set(range(G)) - set(full_order)
    counter=1
    while unused:
        print(f"插值{counter}次")
        counter+=1
        insert_idx = unused.pop()
        # 找到full_order中距离该点最近的位置插入
        dists = {i: np.linalg.norm(centers[insert_idx] - centers[i]) for i in full_order}
        nearest = min(dists, key=dists.get)
        pos = full_order.index(nearest)
        full_order.insert(pos + 1, insert_idx)

    return full_order


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

def get_cluster_path_from_user(G):
    """
    获取用户输入的 cluster_path，并验证其长度是否为 G。

    参数：
        G (int): 聚类个数

    返回：
        List[int] or None: 用户输入的 cluster_path 列表，若失败则返回 None
    """
    user_input = input(f"请输入 cluster_path（共 {G} 个，用逗号分隔，如：3, 0, 2, ...）：")

    try:
        cluster_path = [int(x.strip()) for x in user_input.split(",")]

        if len(cluster_path) != G:
            print(f"❌ 输入的 cluster 数量为 {len(cluster_path)}，但应为 {G} 个。请重新输入。")
            return None

        print(f"✅ 你输入的 cluster_path 是：{cluster_path}")
        return cluster_path

    except ValueError:
        print("❌ 输入格式错误！请确保只输入整数，用英文逗号分隔。")
        return None
