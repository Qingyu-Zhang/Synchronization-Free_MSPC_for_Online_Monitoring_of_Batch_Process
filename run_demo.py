"""
演示脚本：运行全流程
包括：
- 加载数据
- 全局PCA + 聚类
- 构建局部MSPC模型
- 监控一个新批次
- 生成Q曲线与贡献图
"""

import numpy as np
import pandas as pd
from src.preprocessing import load_augmented_data
from src.pca_modeling import PCA
from src.clustering import perform_kmeans, build_cluster_path, build_cluster_index_map
from src.mspc_local import build_local_models
from src.monitoring import monitor_new_point
from src.diagnostics import get_contribution_vector
from src.utils import ensure_dir, save_q_curve, save_contribution_plot


# 全局参数
A = 3         # PCA保留主成分数
G = 10        # KMeans聚类数
ALPHA = 0.95  # Q统计量置信水平

def main():
    # Step 1: 加载数据
    filepath = "data/example_augmented.csv"
    X_augmented = load_augmented_data(filepath)
    print(f"读取训练用NOC批次光谱数据合并矩阵 shape: {X_augmented.shape}")

    # Step 2: 全局PCA并获得得分矩阵
    X_mean = np.mean(X_augmented, axis=0)
    X_centered = X_augmented - X_mean
    pca_global = PCA(n_components=A)
    T_all = pca_global.fit_transform(X_centered)

    # Step 3: KMeans聚类
    labels, centers = perform_kmeans(T_all, n_clusters=G)
    cluster_path = build_cluster_path(centers)
    cluster_index_map = build_cluster_index_map(labels)

    # Step 4: 构建local MSPC模型
    local_models = build_local_models(X_augmented, cluster_path, cluster_index_map,
                                      n_components=A, alpha=ALPHA)

    # Step 5: 模拟一个新批次（这里只取原始数据的前20行作为演示）
    new_batch = X_augmented[:20]
    min_qrs = []
    latest_valid_model_idx = None  # 用于论文忠实复现功能，可选择开启（使用最近一个正常点(NOC)的最小Qr的local MSPC模型）

    ensure_dir("results/Q_curves")
    ensure_dir("results/contributions")

    for k, x_k in enumerate(new_batch):
        is_faulty, Qr_all = monitor_new_point(x_k, local_models)
        min_qrs.append(np.min(Qr_all))

        if is_faulty:
            # 异常 → 画贡献图（默认方式）
            e_k, used_model = get_contribution_vector(
                x_k, local_models, Qr_all,
                use_latest_noc_model=False,
                latest_valid_model_idx=latest_valid_model_idx
            )
            save_contribution_plot(e_k,
                                   f"results/contributions/point{k:02d}_fault.png",
                                   title=f"Contribution Plot - Point {k}")
        else:
            latest_valid_model_idx = int(np.argmin(Qr_all))

    # Step 6: 保存Q曲线
    save_q_curve(min_qrs, "results/Q_curves/demo_q_curve.png", title="Min Qr Monitoring Demo")

if __name__ == "__main__":
    main()
