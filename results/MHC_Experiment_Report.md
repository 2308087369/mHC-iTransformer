# MHC (Manifold-Constrained Hyper-Connections) 实验报告

## 1. 实验背景与目标
本实验旨在验证 **流形约束超连接 (MHC)** 技术在时间序列预测任务中的有效性。与传统的残差连接（Residual Connections）不同，MHC 通过在 Birkhoff 多面体（双随机矩阵流形）上进行投影，实现了多流信息的非线性演化与聚合。

本实验的核心目标是：
- 将 MHC 集成到 **iTransformer** 架构中，替代其原始的残差连接。
- 在相同的训练配置下（30 Epochs），对比 **MHC-iTransformer** 与 **原始 iTransformer** 以及 **PatchTST** 基准模型的性能。

## 2. 模型架构设计

### 2.1 MHC-iTransformer (核心改进)
在 [mhc_itransformer.py](mhc_itransformer.py) 中，我们重新设计了 Transformer 块。核心组件是 `MHCBlock`：
- **维度倒置**：遵循 iTransformer 逻辑，将时间步视为特征维度，通道视为独立变量。
- **超连接替代残差**：将 `x = x + sub_layer(x)` 替换为基于多流流形约束的更新公式：
  $$X_{stream} = \text{Sinkhorn}(\Theta) \cdot X_{stream} + \Phi \cdot \text{SubLayer}(X_{agg})$$
- **Sinkhorn 投影**：通过 `SinkhornProjection` 确保权重矩阵处于 Birkhoff 概率流形上，保证了信息流转的稳定性和多流聚合的有效性。

### 2.2 基准模型
- **PatchTST**: 基于分片技术的 Transformer 模型，是当前长时序列预测的强基准。
- **iTransformer (Original)**: 在 [models.py](models.py) 中实现，采用与 MHC-iTransformer 相同的倒置维度逻辑，但使用标准的加法残差连接。

## 3. 实验设置
- **数据集**: 电力负荷数据 (`aligned_15m_full.csv`)。
- **任务**: 1周（672点）历史预测未来1周（672点）。
- **超参数**:
  - `D_MODEL`: 128
  - `N_EPOCHS`: 30
  - `SEQ_LEN`: 672
  - `PRED_LEN`: 672
- **评估指标**: MAPE, RMSE, NRMSE。

## 4. 实验结果

### 4.1 指标对比 (Quarter 2 测试集)

| 指标 | PatchTST (Baseline) | iTransformer (Original) | **MHC-iTransformer (Ours)** |
| :--- | :--- | :--- | :--- |
| **MAPE (%)** | 4.5241 | 4.4993 | **3.4950** (↓22.3%) |
| **RMSE** | 38.3234 | 38.3933 | **30.3459** (↓20.9%) |
| **NRMSE** | 0.3111 | 0.3117 | **0.2463** (↓21.0%) |

### 4.2 训练过程观察
- **收敛速度**: `MHC-iTransformer` 在第 1 个 Epoch 的 Loss (0.233) 远低于 `iTransformer` (0.326) 和 `PatchTST` (0.626)，显示出流形连接在参数初始化和信息传递上的天然优势。
- **最终损失**: 经过 30 轮训练，MHC 版本的训练损失稳定在 0.082 左右，优于原始 iTransformer 的 0.115。

### 4.3 图像分析
在生成的对比图 `q2_final_itransformer_comparison.png` 中可以观察到：
- **峰值拟合**: 在电力负荷的波峰位置，MHC-iTransformer (红色实线) 的预测曲线与真实值 (黑色虚线) 最为贴合。
- **波动鲁棒性**: 在序列中间的剧烈波动区域，PatchTST 出现了明显的滞后，而 MHC-iTransformer 能够快速捕捉到趋势转折。

## 5. 结论
实验结果有力证明了 **MHC (流形约束超连接)** 在替代传统残差连接方面的巨大潜力。
1. **精度飞跃**: 相比于目前最强的 iTransformer 变体，引入 MHC 带来了超过 20% 的各项指标提升。
2. **架构普适性**: 实验证明 MHC 可以无缝替换 Transformer 架构中的残差部分，且不改变原有的倒置维度优势。
3. **流形约束价值**: 通过 Sinkhorn 投影实现的多流信息交互，比简单的线性残差更能有效表征复杂的时间序列动态特性。
