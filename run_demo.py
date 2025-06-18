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
from src.preprocessing import load_data
from src.pca_modeling import PCA
from src.clustering import perform_kmeans, build_cluster_path, build_cluster_index_map, get_cluster_path_from_user, build_cluster_path_mst
from src.mspc_local import build_local_models
from src.monitoring import monitor_new_point
from src.diagnostics import get_contribution_vector
from src.utils import ensure_dir, save_q_curve, save_contribution_plot
from src.visualization import plot_pca_2d, plot_pca_3d



def run_whole_process(X_augmented, new_batch, n_components=2, G=10, ALPHA=0.99, Use_Last_NOC_Model=False, Mannual_Path=False, MST_Path=True):
    """
    运行全流程
    包括：
    - 加载数据
    - 全局PCA + 聚类
    - 构建局部MSPC模型
    - 监控一个新批次
    - 生成Q曲线与贡献图

    参数：
        X_augmented (DataFrame): shape = (n_samples, n_features)的所有NOC批次合并的增广矩阵
        new_batch (DataFrame): shape = (m_samples, n_features)的新批次数据矩阵
        n_components (int)： PCA保留的主成分个数，接受2或3
        G (int): KMeans聚类个数
        ALPHA (int): Q统计量置信水平
        Use_Last_NOC_Model (bool): 画异常点特征贡献图时是否使用最近的NOC的最小Qr local model
        Mannual_Path (bool): 是否根据可视化轨迹手动输入路径编号顺序

    返回：None
    """
    # Step 1: 加载数据
    feature_names = list(X_augmented.columns)   # DataFrame.columns是pandas.Index类型
    X_augmented = X_augmented.to_numpy().astype(np.float64)

    print(f"读取训练用NOC批次光谱数据合并矩阵 shape: {X_augmented.shape}")

    # Step 2: 全局PCA并获得得分矩阵
    X_mean = np.mean(X_augmented, axis=0)
    X_std = np.std(X_augmented, axis=0)
    X_standardized = (X_augmented - X_mean) / X_std
    pca_global = PCA(n_components=n_components)
    T_all = pca_global.fit_transform(X_standardized)

    # Step 3: KMeans聚类
    labels, centers = perform_kmeans(T_all, n_clusters=G)

    # 可视化步骤
    ensure_dir("results/visualization")
    if n_components == 2:
        plot_pca_2d(T_all, labels=labels, title="PCA Score Plot (2D)",
                    save_path="results/visualization/pca_2d.png")

    if n_components >= 3:
        plot_pca_3d(T_all, labels=labels, title="PCA Score Plot (3D)",
                    save_path="results/visualization/pca_3d.png")

    if Mannual_Path:
        cluster_path = get_cluster_path_from_user(G)
    else:
        if MST_Path:
            cluster_path = build_cluster_path_mst(centers)
        else:
            cluster_path = build_cluster_path(centers)

        print("cluster_path:", cluster_path)

    cluster_index_map = build_cluster_index_map(labels)


    # Step 4: 构建local MSPC模型
    local_models = build_local_models(X_augmented, cluster_path, cluster_index_map,
                                      n_components=n_components, alpha=ALPHA)

    # Step 5: 新批次数据
    new_batch = new_batch.to_numpy().astype(np.float64)
    print("新批次矩阵shape:", new_batch.shape)

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
                use_latest_noc_model=Use_Last_NOC_Model,
                latest_valid_model_idx=latest_valid_model_idx
            )
            save_contribution_plot(e_k, feature_names,
                                   f"results/contributions/point{k:02d}_fault.png",
                                   title=f"Contribution Plot - Point {k}")
        else:
            latest_valid_model_idx = int(np.argmin(Qr_all))

    # Step 6: 保存Q曲线
    save_q_curve(min_qrs, "results/Q_curves/demo_q_curve.png", title="Min Qr Monitoring Demo")

