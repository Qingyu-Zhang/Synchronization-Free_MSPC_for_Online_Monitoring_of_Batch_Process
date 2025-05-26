# Synchronization-Free MSPC Replication

本项目复现论文：

> **Synchronization-Free Multivariate Statistical Process Control for Online Monitoring of Batch Process Evolution**  
> Rodrigo Rocha de Oliveira, Anna de Juan  
> *Frontiers in Analytical Science, 2022*  
> DOI: [10.3389/frans.2021.772844](https://doi.org/10.3389/frans.2021.772844)

---

## 📌 项目简介

本项目基于论文中的同步无关多变量统计过程控制（Synchronization-Free MSPC）方法，实现对非同步批次过程演化的实时监控与故障诊断。

该方法通过以下步骤构建无同步要求的 MSPC 系统：

1. 将多个非同步 NOC（正常运行）批次拼接为一个大矩阵；
2. 对拼接矩阵进行 PCA 建模，获得整体轨迹的得分空间；
3. 在得分空间中进行 KMeans 聚类；
4. 每次使用相邻两个 clusters 构造一个 local MSPC 模型；
5. 在新批次到来时，逐时刻计算其在所有 local 模型下的 Qr 值；
6. 判断是否异常，并绘制贡献图指出主要故障变量。

---

## 🧱 项目结构

```
syncfree_mspc_replication/
├── data/                         # 输入数据（拼接后的 NOC 光谱矩阵）
│   └── example_augmented.csv
├── results/                      # 输出图像
│   ├── Q_curves/
│   └── contributions/
├── src/                          # 模块化代码
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── pca_modeling.py
│   ├── mspc_local.py
│   ├── monitoring.py
│   ├── diagnostics.py
│   └── utils.py
├── run_demo.py                   # 主流程入口脚本（相当于 main.py）
└── README.md
```


## 🚀 使用方法

1. 将拼接后的 NOC 批次数据保存为：
   ```
   data/example_augmented.csv
   ```

2. 运行主流程脚本：
   ```bash
   python run_demo.py
   ```


## 📊 输出结果

执行成功后将在 `results/` 文件夹下生成：

- `Q_curves/demo_q_curve.png`：每时刻最小 Qr 值的变化图；
- `contributions/pointXX_fault.png`：每个被诊断为异常的观测点的变量残差贡献图。


## ⚙️ 参数设置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `A` | 全局与局部 PCA 主成分数 | 3 |
| `G` | KMeans 聚类数量 | 10 |
| `alpha` | Q 控制限显著性水平 | 0.95 |
