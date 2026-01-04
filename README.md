# MHC_time_series

本项目是一个时间序列预测研究项目，主要探索和对比了多种先进的深度学习模型在特定时间序列数据集上的表现。

## 项目简介

本项目实现了以下模型用于时间序列预测：
- **PatchTST**: 基于 Patch 的时间序列 Transformer 基准模型。
- **iTransformer**: 倒置 Transformer (Inverted Transformer) 模型，通过将维度与时间序列对换来捕获多变量相关性。
- **MHC-iTransformer**: **本项目核心改进模型**，将多视图霍普菲尔德计算（Multi-view Hopfield Computing）与 iTransformer 结合。

## MHC-iTransformer 详细说明

MHC-iTransformer 在原始 iTransformer 的基础上引入了 **多视图霍普菲尔德计算 (MHC)** 机制，旨在增强模型对复杂时间序列模式的捕捉能力。

### 1. 核心原理
- **多视图流 (Multi-stream)**: 不同于传统 Transformer 维护单一的状态，MHC 维护 $N$ 个并行的信息流（Streams/Views）。每个流可以捕捉时间序列的不同特征。
- **倒置维度 (Inverted Embedding)**: 继承 iTransformer 的特性，将每个变量的整条序列映射为 Token，使得 Attention 机制在变量维度而非时间维度上运行。
- **Sinkhorn 投影流形**: 使用 Sinkhorn 算法确保流与流之间的转移矩阵 $W$ 是双随机（Doubly Stochastic）的，保证了信息在传递过程中的守恒与多样性。

### 2. 核心公式
MHC Block 的状态更新遵循以下残差逻辑：

$$H_{l+1} = H_l \cdot W + \phi \cdot \text{Sublayer}(\text{Agg}(H_l))$$

其中：
- $H_l$: 第 $l$ 层的多视图流状态。
- $W$: 经过 Sinkhorn 投影的流转移矩阵，负责流间的信息交换。
- $\text{Agg}(H_l)$: 通过注意力加权聚合多个流的信息作为子层输入。
- $\text{Sublayer}$: 代表多头注意力（Attention）或前馈网络（FFN）。
- $\phi$: 可学习的门控参数，控制子层输出对各流的影响。

### 3. 主要改进点
- **增强的鲁棒性**: 通过多视图机制，模型能够同时关注局部细节和全局趋势，对噪声数据具有更强的鲁棒性。
- **非平稳性处理**: 集成了 **RevIN (Reversible Instance Normalization)**，有效解决了时间序列分布随时间偏移的问题。
- **信息交换流形**: 引入 Sinkhorn 投影，使得模型在学习复杂变量间关系时具有更优的几何约束。

## 实验结果

项目在特定季度的数据集上进行了对比实验，以下是 MHC-iTransformer 与基准模型的预测效果对比：

![模型对比结果](results/q2_final_itransformer_comparison.png)

## 项目结构

```text
.
├── datasets/           # 数据集目录 (闭源)
├── docs/               # 相关文档
├── models/             # 模型定义与数据处理工具
│   ├── data_utils.py   # 数据加载与预处理
│   ├── models.py       # PatchTST 与 iTransformer 定义
│   └── mhc_itransformer.py # MHC-iTransformer 定义 (核心)
├── results/            # 训练结果与可视化图表
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

## 注意事项

- **数据集**: 本项目使用的数据集属于闭源内容。如需运行，请将 `aligned_15m_full.csv` 放置于 `datasets/` 目录。
- **计算资源**: 建议在支持 CUDA 的 GPU 环境下运行。
