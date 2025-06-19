"""
PCA得分空间可视化模块：
- 支持二维 (PC1 vs PC2) 和三维 (PC1 vs PC2 vs PC3) 可视化
- 支持label着色（如batch或cluster编号）
- 自动标注起点/终点cluster（若label为连续编号）
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go

def plot_pca_2d(T_all, labels=None, save_path=None, title="PCA Score Plot 2D", label_names=None):
    """
    绘制二维 PCA 得分空间图（PC1 vs PC2）

    参数：
        T_all (np.ndarray): shape = (N, 2)，得分矩阵（PC1, PC2）
        labels (np.ndarray or list): 每个样本的标签（可选）
        save_path (str): 如果指定，将图像保存到该路径
        title (str): 图标题
        label_names (dict): label → 名字 映射（可选）
    """
    plt.figure(figsize=(8, 6))

    if labels is None:
        plt.scatter(T_all[:, 0], T_all[:, 1], color='gray', alpha=0.6)
    else:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap('tab20', len(unique_labels))

        for i, label in enumerate(unique_labels):
            idx = labels == label
            label_str = label_names[label] if label_names and label in label_names else f"Group {label}"
            plt.scatter(T_all[idx, 0], T_all[idx, 1], label=label_str, s=20, alpha=0.7, color=cmap(i))

            # 可选：在每组中心加文字
            center_x = np.mean(T_all[idx, 0])
            center_y = np.mean(T_all[idx, 1])
            plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='center', weight='bold')

        # # 起点/终点 cluster 箭头
        # if len(unique_labels) >= 2:
        #     start_idx = labels == unique_labels[0]
        #     end_idx = labels == unique_labels[-1]
        #     plt.annotate(f"Start ({unique_labels[0]})", xy=T_all[start_idx][0], xytext=(-30, 20),
        #                  textcoords='offset points', arrowprops=dict(arrowstyle="->", color='blue'))
        #     plt.annotate(f"End ({unique_labels[-1]})", xy=T_all[end_idx][-1], xytext=(30, -30),
        #                  textcoords='offset points', arrowprops=dict(arrowstyle="->", color='red'))

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_pca_3d(T_all, labels=None, save_path=None, title="PCA Score Plot 3D", label_names=None):
    """
    绘制三维 PCA 得分空间图（PC1 vs PC2 vs PC3）

    参数同上，T_all shape = (N, 3)
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    if labels is None:
        ax.scatter(T_all[:, 0], T_all[:, 1], T_all[:, 2], color='gray', alpha=0.6)
    else:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap('tab20', len(unique_labels))

        for i, label in enumerate(unique_labels):
            idx = labels == label
            label_str = label_names[label] if label_names and label in label_names else f"Group {label}"
            ax.scatter(T_all[idx, 0], T_all[idx, 1], T_all[idx, 2], label=label_str, s=20, alpha=0.7, color=cmap(i))

            # 中心位置标文字
            cx = np.mean(T_all[idx, 0])
            cy = np.mean(T_all[idx, 1])
            cz = np.mean(T_all[idx, 2])
            ax.text(cx, cy, cz, str(label), fontsize=8, weight='bold')

        # if len(unique_labels) >= 2:
        #     start_idx = labels == unique_labels[0]
        #     end_idx = labels == unique_labels[-1]
        #     ax.text(*T_all[start_idx][0], f"Start ({unique_labels[0]})", color='blue')
        #     ax.text(*T_all[end_idx][-1], f"End ({unique_labels[-1]})", color='red')

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

#
# def plot_pca_3d_interactive(T_all, labels=None, save_path=None, title="PCA Score Plot 3D (Interactive)", label_names=None):
#     """
#     使用 Plotly 绘制可交互的 3D PCA 得分图。
#
#     参数：
#         T_all (np.ndarray): shape = (N, 3)，PCA得分
#         labels (np.ndarray or list): 每个样本的标签（可选）
#         save_path (str): 如果指定，将图保存为HTML（推荐以 .html 结尾）
#         title (str): 图标题
#         label_names (dict): label → 名称映射
#     """
#     fig = go.Figure()
#
#     if labels is None:
#         fig.add_trace(go.Scatter3d(
#             x=T_all[:, 0], y=T_all[:, 1], z=T_all[:, 2],
#             mode='markers',
#             marker=dict(size=4, color='gray', opacity=0.7),
#             name='Data'
#         ))
#     else:
#         labels = np.array(labels)
#         unique_labels = np.unique(labels)
#         for label in unique_labels:
#             idx = labels == label
#             label_str = label_names[label] if label_names and label in label_names else f"Group {label}"
#
#             fig.add_trace(go.Scatter3d(
#                 x=T_all[idx, 0], y=T_all[idx, 1], z=T_all[idx, 2],
#                 mode='markers+text',
#                 marker=dict(size=4, opacity=0.8),
#                 text=[str(label)] * np.sum(idx),
#                 name=label_str
#             ))
#
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='PC1',
#             yaxis_title='PC2',
#             zaxis_title='PC3'
#         ),
#         title=title,
#         width=800,
#         height=700,
#         showlegend=True
#     )
#
#     if save_path:
#         fig.write_html(save_path)
#     else:
#         fig.show()



def plot_pca_3d_interactive(T_all, labels=None, save_path=None, title="PCA Score Plot 3D (Interactive)", label_names=None):
    """
    使用 Plotly 绘制可交互的 3D PCA 得分图，仅在每个 cluster 中心标注一个编号。

    参数：
        T_all (np.ndarray): shape = (N, 3)，PCA得分
        labels (np.ndarray or list): 每个样本的标签（可选）
        save_path (str): 如果指定，将图保存为HTML（推荐以 .html 结尾）
        title (str): 图标题
        label_names (dict): label → 名称映射
    """
    fig = go.Figure()

    if labels is None:
        fig.add_trace(go.Scatter3d(
            x=T_all[:, 0], y=T_all[:, 1], z=T_all[:, 2],
            mode='markers',
            marker=dict(size=4, color='gray', opacity=0.7),
            name='Data'
        ))
    else:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            label_str = label_names[label] if label_names and label in label_names else f"Group {label}"

            # 添加当前 cluster 的点
            fig.add_trace(go.Scatter3d(
                x=T_all[idx, 0], y=T_all[idx, 1], z=T_all[idx, 2],
                mode='markers',
                marker=dict(size=4, opacity=0.8),
                name=label_str
            ))

            # 计算当前 cluster 的中心点，并添加文字标签
            cx = np.mean(T_all[idx, 0])
            cy = np.mean(T_all[idx, 1])
            cz = np.mean(T_all[idx, 2])
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz],
                mode='text',
                text=[str(label)],
                textposition='middle center',
                showlegend=False
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        title=title,
        width=800,
        height=700,
        showlegend=True
    )

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()
