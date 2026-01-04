# MHC_time_series

本项目是一个时间序列预测研究项目，主要探索和对比了多种先进的深度学习模型在特定时间序列数据集上的表现。

## 项目简介

本项目实现了以下模型用于时间序列预测：
- **PatchTST**: 作为基准模型。
- **iTransformer**: 原始的 iTransformer 模型。
- **MHC-iTransformer**: 本项目提出的改进模型（Ours）。

项目通过对比不同模型在各项指标（MAPE, RMSE, NRMSE）上的表现，验证 MHC-iTransformer 的有效性。

## 项目结构

```text
.
├── datasets/           # 数据集目录 (闭源，未包含在仓库中)
├── docs/               # 相关文档
├── models/             # 模型定义与数据处理工具
│   ├── data_utils.py   # 数据加载与预处理
│   ├── models.py       # PatchTST 与 iTransformer 定义
│   └── mhc_itransformer.py # MHC-iTransformer 定义
├── results/            # 训练结果与可视化图表保存目录
├── requirements.txt    # 项目依赖
└── train_and_eval.py   # 主训练与评估脚本
```

## 安装指南

1. 确保已安装 Python 3.8+。
2. 安装项目依赖：

```bash
pip install -r requirements.txt
```

## 运行方法

直接运行主脚本进行训练与评估：

```bash
python train_and_eval.py
```

脚本将自动加载 `datasets/` 下的数据，进行模型训练，并在 `results/` 目录下生成对比图表和评估指标。

## 注意事项

- **数据集**: 本项目使用的数据集（位于 `datasets/` 目录）属于闭源内容，未包含在公开代码库中。如需运行代码，请确保您拥有相应的数据文件 `aligned_15m_full.csv`。
- **环境**: 建议在支持 CUDA 的 GPU 环境下运行以加快训练速度。
