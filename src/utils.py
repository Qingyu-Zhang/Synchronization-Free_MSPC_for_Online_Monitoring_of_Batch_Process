"""
工具模块：
- 包含通用辅助函数，例如路径管理、结果保存等
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(path):
    """
    若路径不存在则创建目录。

    参数：
        path (str): 文件夹路径
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_q_curve(qr_values, save_path, title="Q Statistic Curve"):
    """
    保存一条批次的最小Qr随时间变化曲线。

    参数：
        qr_values (List[float]): 每个时刻观测点的最小Qr值
        save_path (str): 图像保存路径
        title (str): 图标题
    """
    plt.figure()
    plt.plot(qr_values, marker='o')
    plt.axhline(1.0, color='r', linestyle='--', label='Qr = 1 (threshold)')
    plt.title(title)
    plt.xlabel('Time (observation index)')
    plt.ylabel('Min Qr across models')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_contribution_plot(e, feature_names, save_path, title="Contribution Plot"):
    """
    保存贡献图（残差平方）。

    参数：
        e (np.ndarray): 残差向量
        feature_names (List[str]): 特征列名
        save_path (str): 图像保存路径
        title (str): 图标题
    """
    plt.figure(figsize=(12,8))
    plt.bar(feature_names, e)
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Squared Residual')
    plt.xticks(rotation=45, ha='right')  # 避免标签重叠
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
