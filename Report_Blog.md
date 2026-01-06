# 2023-2025年时间序列预测前沿全景报告：从线性反思到十亿级基础模型的范式重构

> **摘要**：本报告回顾了2023年至2025年时间序列预测（Time Series Forecasting, TSF）领域的演进历程。从2023年线性模型对Transformer的挑战，到2024年状态空间模型（SSM）的崛起，再到2025年十亿参数级通用基础模型（Foundation Models）的爆发，我们见证了一场深刻的范式转移。本文深入剖析了DLinear、PatchTST、TimeMachine、Time-LLM、Time-MoE等核心架构，并探讨了零样本预测、跨模态重编程及混合专家模型等前沿技术。

---

## 目录

- [1. 引言：时间序列预测的“大模型时刻”与范式转移](#1-引言时间序列预测的大模型时刻与范式转移)
- [2. 线性模型的复兴与Transformer的自我救赎 (2023)](#2-线性模型的复兴与transformer的自我救赎-2023)
    - [2.1 线性反思：DLinear与NLinear对复杂度的挑战](#21-线性反思dlinear与nlinear对复杂度的挑战)
    - [2.2 TSMixer：全MLP架构的逆袭](#22-tsmixer全mlp架构的逆袭)
    - [2.3 PatchTST：Transformer的正确打开方式](#23-patchtsttransformer的正确打开方式-iclr-2023)
    - [2.4 iTransformer：维度的彻底反转](#24-itransformer维度的彻底反转-iclr-2024)
- [3. 状态空间模型 (SSM) 的崛起：Mamba与线性复杂度的回归 (2024-2025)](#3-状态空间模型-ssm-的崛起mamba与线性复杂度的回归-2024-2025)
    - [3.1 TimeMachine：四重Mamba的架构创新](#31-timemachine四重mamba的架构创新-ecai-2024)
    - [3.2 MambaTS：动态图与变量感知扫描](#32-mambats动态图与变量感知扫描-openreview-2025)
- [4. 时间序列基础模型 (TSFMs) 的爆发：通向零样本预测之路 (2024-2025)](#4-时间序列基础模型-tsfms-的爆发通向零样本预测之路-2024-2025)
    - [4.1 范式一：基于LLM的跨模态重编程 (Time-LLM)](#41-范式一基于llm的跨模态重编程-time-llm)
    - [4.2 范式二：量化与序列建模 (Chronos & Lag-Llama)](#42-范式二量化与序列建模-chronos--lag-llama)
    - [4.3 范式三：原生基础模型与MoE架构 (TimesFM, Time-MoE)](#43-范式三原生基础模型与moe架构-timesfm-time-moe)
- [5. 基准测试与性能评估：从单一到通用](#5-基准测试与性能评估从单一到通用)
    - [5.1 GIFT-Eval：通用预测的新标准](#51-gift-eval通用预测的新标准-2025)
    - [5.2 性能对比总结表](#52-性能对比总结表-基于gift-eval及相关论文)
- [6. 结论与未来展望](#6-结论与未来展望)

---

## 1. 引言：时间序列预测的“大模型时刻”与范式转移

在人工智能的研究版图中，**时间序列预测（Time Series Forecasting, TSF）**长期以来占据着独特的生态位。它既不同于自然语言处理（NLP）离散符号的语义构建，也区别于计算机视觉（CV）的空间像素特征提取。时间序列数据固有的连续性、非平稳性（Non-stationarity）、以及复杂的跨变量依赖关系，使得该领域的模型演进呈现出一种独特的螺旋上升态势。

回顾2023年至2025年这三年的发展历程，我们见证了一场深刻的范式转移：从针对特定数据集（如电力、交通、气象）精细调优的专用小模型，向拥有数十亿参数、具备**零样本（Zero-Shot）**泛化能力的**通用基础模型（Foundation Models）**迈进。

*   **2023年**：学界尚处于对Transformer有效性的激烈辩论之中。彼时，以**DLinear**为代表的线性模型通过极简架构在多项基准测试中击败了复杂的Transformer变体，引发了对“过度设计”的反思。然而，这种反思并未终结Transformer的命运，反而催生了**PatchTST**等基于分块（Patching）机制的改良架构，确立了Transformer在时间序列中的正确打开方式。
*   **2024年**：随着大语言模型（LLM）的爆发，时间序列领域迎来了“基础模型元年”。研究者不再满足于在单一数据集上训练模型，而是开始构建大规模的时间序列语料库（如LOTSA, Time-300B），试图复制GPT在文本领域的成功。**Time-LLM**通过跨模态重编程（Reprogramming）挖掘了LLM的推理潜力；**Chronos**通过量化（Quantization）将连续数值转化为语言Token。
*   **2025年**：在ICLR和ICML等顶级会议上，**混合专家模型（Mixture-of-Experts, MoE）**和**上下文微调（In-Context Fine-Tuning）**技术的引入，标志着时间序列模型正式进入了追求推理效率与自适应能力并重的新阶段。**Moirai**、**TimesFM**和**Time-MoE**等原生时间序列基础模型则通过全新的架构设计，挑战了多频率、多变量和长序列建模的极限。

本报告将以详尽的篇幅，剖析这一波澜壮阔的技术演进史，深入解读各核心模型的架构机理、训练策略及性能表现，为专业研究人员提供一份详实的参考指南。

## 2. 线性模型的复兴与Transformer的自我救赎 (2023)

在基础模型大行其道之前，2023年的时间序列研究主要集中在架构的有效性探讨上。这一时期，线性模型的强势表现迫使深度学习研究者重新审视Transformer的核心组件，最终促成了以Patching和Channel Independence为核心的新一代Transformer架构的诞生。

### 2.1 线性反思：DLinear与NLinear对复杂度的挑战

在2023年之前，Informer、Autoformer等基于Transformer的模型主导了长期时间序列预测（LTSF）的研究。这些模型致力于降低Self-Attention的 $O(L^2)$ 复杂度，但往往忽略了时间序列最本质的特征。

#### 2.1.1 DLinear：分解与线性的力量

**DLinear（Decomposition Linear）**的提出 [1] 是对当时过度复杂的Transformer架构的一次有力“拨乱反正”。其核心论点在于：时间序列数据往往包含显著的趋势和季节性成分，直接将混合了这两者的原始数据输入到复杂的Attention机制中，反而可能导致过拟合，且无法有效捕捉长期的趋势变化。

DLinear的架构设计遵循了经典的“分而治之”思想：

1.  **趋势-季节性分解（Trend-Seasonality Decomposition）**：模型首先使用移动平均（Moving Average）滤波器将输入序列 $X$ 分解为趋势分量 $X_{trend}$ 和季节性分量 $X_{seasonal}$。
    $$ X_{trend} = \text{MovingAverage}(X) $$
    $$ X_{seasonal} = X - X_{trend} $$

2.  **独立线性映射**：随后，两个独立的单层线性层（Linear Layer）分别处理这两个分量。线性层直接将历史窗口映射到预测视窗。
    $$ \hat{X}_{trend} = \text{Linear}_{trend}(X_{trend}) $$
    $$ \hat{X}_{seasonal} = \text{Linear}_{seasonal}(X_{seasonal}) $$

3.  **合成输出**：最终预测结果由两部分相加而成。
    $$ \hat{X} = \hat{X}_{trend} + \hat{X}_{seasonal} $$

这种极简的设计在交通、电力等具有明显周期性的数据集上，击败了当时最先进的Transformer模型，证明了在某些场景下，简单的线性映射比复杂的非线性注意力更有效。

#### 2.1.2 NLinear：对抗分布偏移 (Distribution Shift)

为了解决训练数据和测试数据分布（均值、方差）不一致的问题，**NLinear** [1] 在DLinear的基础上引入了归一化策略。

*   **减法归一化**：在输入线性层之前，减去输入序列的最后一个值 $X_{L}$。
*   **线性预测**：对减去后的残差序列进行线性预测。
*   **加法恢复**：在输出层将 $X_{L}$ 加回。

这种简单的操作显著提升了模型对分布偏移的鲁棒性，使其能够专注于学习序列的相对变化而非绝对数值，从而在非平稳数据上表现更加稳定。

### 2.2 TSMixer：全MLP架构的逆袭

在Google推出的 **TSMixer** [3] 中，研究者进一步探索了不依赖注意力机制的可能性。TSMixer采用了**全MLP（All-MLP）**架构，旨在证明只要能够正确地混合时间和特征维度的信息，MLP同样可以达到SOTA性能。

TSMixer的核心模块由两个交替的MLP层组成：
*   **Time-mixing MLP**：在时间维度上作用，捕捉单一变量内的时间依赖。它将输入矩阵转置，使得全连接层作用于时间轴。
*   **Feature-mixing MLP**：在特征（通道）维度上作用，捕捉不同变量之间的相关性。

此外，TSMixer引入了**在线协调（Online Reconciliation）**机制，用于处理层次化时间序列（Hierarchical Time Series）预测。通过在训练损失中加入协调项，TSMixer能够保证不同层级（如“总销售额”与“各分店销售额”）的预测结果保持一致性 [3]。这一工作不仅在性能上挑战了Transformer，更重要的是它揭示了“混合（Mixing）”操作——而非单纯的注意力——可能是多变量时间序列建模的关键。

### 2.3 PatchTST：Transformer的正确打开方式 (ICLR 2023)

在被线性模型挑战后，Transformer阵营迎来了 **PatchTST** [2]，这篇论文被广泛认为是Transformer在时间序列预测领域的“救赎之作”。PatchTST通过两个核心创新——**分块（Patching）**和**通道独立（Channel Independence）**，彻底改变了Transformer的应用范式。

#### 2.3.1 分块（Patching）机制的深层逻辑

传统的Transformer（如Informer）将每个时间步（Time Step）视为一个Token。对于时间序列而言，单个时间点的数值（标量）语义信息极其稀疏，且包含大量噪声。相比之下，NLP中的一个单词Token包含了丰富的语义。PatchTST借用了Vision Transformer (ViT) 的思想，将时间序列切分为重叠或不重叠的小片段（Patches）。

*   **语义密度提升**：一个Patch（例如长度为16的序列片段）能够包含局部的趋势、斜率和波形信息，其语义密度远高于单个点。
*   **计算复杂度降低**：假设输入长度为 $L$，Patch大小为 $P$，步长为 $S$，则Token数量从 $L$ 降至 $N \approx L/S$。由于Self-Attention的复杂度是 $O(N^2)$，Patching操作使得计算量呈二次方级下降。这使得PatchTST能够轻松处理长度为336甚至720的历史窗口，而无需使用稀疏注意力等近似方法 [5]。

#### 2.3.2 通道独立（Channel Independence）的正则化效应

PatchTST的另一个争议性但有效的设计是**通道独立（CI）**。在处理多变量时间序列时，传统的做法是将同一时刻的所有变量嵌入为一个向量。然而，PatchTST将每个变量视为独立的单变量序列，共享同一个Transformer骨干网络。

*   **数据增强**：对于一个包含 $M$ 个变量的数据集，CI策略相当于将训练数据量扩大了 $M$ 倍。
*   **避免虚假相关**：在许多真实数据集中，并非所有变量之间都存在强相关性。强制融合所有通道可能会引入噪声。CI策略迫使模型学习通用的时间模式，而非特定数据集的变量间关系，从而具有更好的泛化能力 [2]。

### 2.4 iTransformer：维度的彻底反转 (ICLR 2024)

![iTransformer Architecture](images/itransformer.png)

如果说PatchTST是时间维度的优化，那么清华大学提出的 **iTransformer** [6] 则是对维度的彻底反转。iTransformer挑战了PatchTST的“通道独立”假设，认为在多变量预测中，变量间的相关性是不可或缺的。

#### 2.4.1 倒置架构（Inverted Architecture）

iTransformer提出了一种反直觉的Token化策略：
*   **传统做法**：Token是某一时刻的所有变量值 $X_t \in \mathbb{R}^M$。
*   **iTransformer做法**：Token是某一变量在整个历史窗口内的所有时间步数值 $X^i \in \mathbb{R}^L$。

这意味着，iTransformer的Self-Attention机制不再是在时间步之间计算，而是在**变量之间**计算。
*   **Attention Map的物理意义**：在这种架构下，Attention Map直接对应于**变量相关性矩阵（Multivariate Correlation Matrix）**。模型能够动态地学习哪些变量对当前预测最重要。
*   **时间建模**：单个变量的时间特征提取由LayerNorm和前馈网络（FFN）完成。由于FFN是全连接的，它能够捕捉全局的时间模式。

#### 2.4.2 适用场景与性能对比

实验表明，iTransformer在变量数量众多且相关性显著的数据集（如Solar-Energy, Traffic）上表现优于PatchTST，而在变量相对独立的数据集上则互有胜负 [7]。这一发现揭示了时间序列建模的“没有免费午餐定理”：架构的选择必须基于数据的内在特性（时间依赖主导 vs. 变量依赖主导）。

## 3. 状态空间模型 (SSM) 的崛起：Mamba与线性复杂度的回归 (2024-2025)

随着 **Mamba** 架构在2024年初的横空出世，其基于选择性状态空间模型（Selective State Space Model, SSM）的设计迅速渗透到时间序列领域。SSM提供了一种诱人的前景：既能像Transformer一样具有全局感受野，又能像RNN一样在推理时保持线性时间复杂度。

### 3.1 TimeMachine：四重Mamba的架构创新 (ECAI 2024)

**TimeMachine** [8] 是首批将Mamba应用于长期时间序列预测（LTSF）并取得SOTA性能的模型之一。面对Transformer在长序列处理上的高昂计算成本，TimeMachine试图证明纯SSM架构在保持线性可扩展性的同时，能够捕捉足够长期的依赖。

#### 3.1.1 统一通道处理的四重架构

TimeMachine并未简单堆叠Mamba层，而是设计了一个独特的**四重Mamba（Integrated Quadruple-Mamba）**架构，旨在解决前文提到的“通道混合”与“通道独立”的矛盾。

*   **多尺度分层**：模型首先通过下采样将输入序列分为两个尺度（Level 1 和 Level 2）。这种多分辨率设计有助于同时捕捉高频细节和低频趋势。
*   **内外Mamba协同**：在每个尺度上，配置了一对Mamba模块。
    *   **Outer Mamba**：用于处理**通道混合（Channel Mixing）**场景，通过全局扫描捕捉跨变量的依赖关系。
    *   **Inner Mamba**：用于处理**通道独立（Channel Independence）**场景，专注于单变量内的长序列时间依赖。
*   **选择性扫描**：利用Mamba的选择性扫描机制（Selective Scan），TimeMachine能够根据输入内容动态调整状态空间的参数（$B, C, \Delta$），从而有效地从噪声中过滤出关键信息。

#### 3.1.2 性能与效率的平衡

实验结果显示，TimeMachine在处理超长序列（如Lookback=720）时，显存占用和推理时间均呈线性增长，显著优于PatchTST和iTransformer的二次方增长 [10]。在预测精度上，它在多个基准数据集上取得了最优或次优的结果，证明了SSM在时间序列领域的巨大潜力。

### 3.2 MambaTS：动态图与变量感知扫描 (OpenReview 2025)

**MambaTS** [11] 对Mamba在多变量时间序列中的应用进行了更深入的理论探索。

#### 3.2.1 质疑因果卷积的必要性

标准的Mamba块包含一个一维因果卷积层（1D Causal Conv），用于提供局部上下文。然而，MambaTS的作者认为，在时间序列预测中，特别是当变量之间存在复杂的独立性时，硬编码的卷积可能会引入不必要的归纳偏置。因此，他们提出了**Temporal Mamba Block (TMB)**，移除了卷积层，完全依赖SSM的状态演化来捕捉依赖。

#### 3.2.2 变量感知扫描 (Variable-Aware Scan Along Time, VAST)

传统的SSM通常按固定的顺序（如从变量1到变量M）扫描多变量数据。MambaTS指出，变量之间的逻辑顺序（拓扑结构）对于信息流至关重要。

*   **动态图发现**：MambaTS引入了VAST机制，在训练过程中动态学习变量之间的因果图结构。
*   **最优路径解码**：基于学习到的依赖关系，模型通过求解一个类似于旅行商问题（TSP）的优化路径，确定最优的变量扫描顺序。这使得一维的扫描机制能够有效地编码高维的变量间拓扑结构 [11]。

## 4. 时间序列基础模型 (TSFMs) 的爆发：通向零样本预测之路 (2024-2025)

如果说2023年是架构之争，那么2024-2025年无疑是“基础模型”的时代。受到NLP领域LLM Scaling Laws的启发，研究界开始探索构建通用的时间序列基础模型（Time Series Foundation Models, TSFMs）。这些模型旨在通过在大规模、跨领域的数据集上预训练，实现对任意新数据的零样本预测能力，从而彻底解决特定领域数据稀缺（Cold Start）的问题。

### 4.1 范式一：基于LLM的跨模态重编程 (Time-LLM)

第一类路径是“借力打力”，即利用现成的、拥有强大推理能力的文本大模型（如Llama, GPT-4）来处理时间序列。

#### 4.1.1 Time-LLM：时间即语言 (ICLR 2024)

**Time-LLM** [12] 提出了一种无需微调LLM本体即可进行高性能预测的框架。其核心思想是将时间序列数据“翻译”为LLM能够理解的语言嵌入。

*   **重编程层 (Reprogramming Layer)**：Time-LLM使用一个可学习的线性层，将输入的时间序列Patch映射到LLM的词嵌入空间（Text Prototype Space）。这一步实现了模态对齐。
*   **Prompt-as-Prefix (PaP)**：为了激活LLM的领域知识，Time-LLM构建了结构化的文本Prompt。这些Prompt不仅包含转换后的时间序列特征，还包含关于数据集特性的自然语言描述（例如：“这是一个按小时采样的电力消耗数据，具有明显的日周期性...”）。
*   **推理与解码**：冻结的LLM处理这些Prompt并输出新的嵌入，最后通过一个投影层解码回数值预测。

**实验表现**：Time-LLM在少样本（Few-Shot）和零样本场景下表现惊人，证明了LLM内部蕴含的通用模式识别能力可以迁移到时间序列领域 [13]。

### 4.2 范式二：量化与序列建模 (Chronos & Lag-Llama)

第二类路径是将时间序列离散化，使其在形式上完全等同于文本，从而可以直接使用标准的Transformer架构（如T5, Llama）进行训练。

#### 4.2.1 Chronos：概率预测的语言化 (NeurIPS 2024)

Amazon提出的 **Chronos** [14] 采取了激进的离散化策略：
1.  **缩放与量化**：首先将时间序列值缩放并量化为有限的“箱”（Bins）。
2.  **Token化**：每个箱被视为一个特定的Token（例如`<bin_34>`）。
3.  **自回归生成**：使用T5或Llama架构训练模型预测下一个Token的概率分布。

Chronos的优势在于它天然支持**概率预测（Probabilistic Forecasting）**，即不仅输出预测值，还能输出置信区间，这对于决策支持至关重要。

### 4.3 范式三：原生基础模型与MoE架构 (TimesFM, Time-MoE)

第三类路径是从零开始构建专门针对时间序列优化的基础模型。

#### 4.3.1 Moirai：统一频率的掩码预训练

Salesforce提出的 **Moirai** [20] 旨在解决多频率（Multi-frequency）数据的统一建模问题。它引入了 **Any-variate Attention** 机制，允许模型处理任意数量和频率的变量。预训练采用掩码时间序列建模（Masked Time Series Modeling），类似于BERT的掩码语言模型任务。

#### 4.3.2 TimesFM与上下文微调 (ICF)

Google的 **TimesFM** [22] 则是一个仅解码器（Decoder-only）的模型，在1000亿个真实世界和合成时间点上进行了训练。

*   **Patching Decoder**：针对长序列预测进行了优化。
*   **In-Context Fine-Tuning (ICF)**：这是一项突破性技术。不同于传统的梯度微调，ICF允许用户在推理时，将与当前预测任务相关的历史案例（Context Examples）作为Prompt输入模型。
*   **分隔符机制**：模型使用特殊的分隔符Token来区分历史案例和当前任务，防止信息混淆。
*   **无需训练的适配**：实验表明，ICF能使零样本模型达到全量微调模型的性能水平，且速度快16倍。这使得TimesFM能够快速适应新的业务场景（如新产品销量预测），而无需经历漫长的再训练过程 [24]。

#### 4.3.3 Time-MoE：迈向十亿参数 (ICLR 2025)

**Time-MoE** [25] 标志着时间序列模型正式进入了十亿参数时代。

*   **2.4B参数规模**：基于Time-300B数据集（3000亿个时间点），Time-MoE是首个参数量达到24亿的开源时间序列模型。
*   **稀疏混合专家 (Sparse MoE)**：为了解决参数增加带来的推理延迟，Time-MoE采用了稀疏MoE架构。在每一层，路由网络（Router）仅为每个Token选择Top-K（通常K=2）个专家进行计算。这使得模型在拥有巨大知识容量的同时，推理时的FLOPs保持在较低水平。
*   **多分辨率预测头**：针对不同长度的预测需求，Time-MoE设计了多分辨率的输出头，能够动态调整预测步长，避免了自回归生成的误差累积问题 [26]。

#### 4.3.4 Moirai-MoE：自动化的专家分工 (ICML 2025)

Salesforce随后推出了 **Moirai-MoE** [28]。
*   **超越频率分组**：原版Moirai依赖基于频率的人工分组。Moirai-MoE认为频率不是区分模式的最佳标准（不同频率可能有相似波形）。
*   **Token级专业化**：Moirai-MoE利用MoE架构，让不同的专家自动学习不同类型的时序模式（如突变、平稳周期、趋势等）。这种Token级的自动路由机制消除了人为启发式规则的局限性，在39个数据集上的评估显示其性能全面超越了原版Moirai [29]。

## 5. 基准测试与性能评估：从单一到通用

### 5.1 GIFT-Eval：通用预测的新标准 (2025)

随着基础模型的涌现，传统的单一数据集（如ETTh1）评估已无法满足需求。Salesforce推出的 **GIFT-Eval** [30] 成为了衡量模型零样本能力的新标准。

*   **全面性**：包含24个数据集、14.4万条时间序列，涵盖7个领域、10种频率。
*   **排行榜洞察**：在2025年的榜单中，TimeCopilot（集成TimeRex, TimesFM, Chronos的方案）在概率预测指标（CRPS）上占据榜首，证明了模型集成（Ensemble）依然是提升鲁棒性的有效手段 [32]。
*   **数据泄露检测**：2025年的更新中，GIFT-Eval特别引入了数据泄露标志，严防预训练数据包含测试集，确保零样本评估的纯粹性 [30]。

### 5.2 性能对比总结表 (基于GIFT-Eval及相关论文)

| 模型类别 | 代表模型 | 核心机制 | 优势 | 局限性 |
| :--- | :--- | :--- | :--- | :--- |
| **线性模型** | DLinear, TSMixer | 分解, 全MLP | 极高效率, 强基线, 适合平稳数据 | 难以捕捉极其复杂的非线性动态 |
| **Transformer (专用)** | PatchTST, iTransformer | Patching, CI, Inverted Attention | SOTA全样本性能, 捕捉长距离依赖 | 需要针对特定数据集训练, 推理成本较高 |
| **SSM / Mamba** | TimeMachine, MambaTS | Selective Scan, Quadruple Mamba | 线性推理复杂度, 低显存, 长序列友好 | 训练并行性不如Transformer, 生态尚在发展 |
| **基础模型 (LLM)** | Time-LLM | Reprogramming, Prompting | 极强的少样本/零样本能力, 利用文本知识 | 推理慢, 依赖底座LLM, 参数量巨大 |
| **基础模型 (原生)** | Moirai, Time-MoE, Chronos | Any-variate Attn, MoE, Quantization | 通用性强, 支持任意变量/频率, 开箱即用 | 零样本精度在特定领域可能不及专用微调模型 |

## 6. 结论与未来展望

2023年至2025年，时间序列预测领域经历了一场从“微观架构优化”到“宏观基础模型构建”的深刻变革。

*   **架构的收敛与分化**：在专用模型领域，Patching和Channel Independence已成为标准配置；而iTransformer的成功提醒我们关注变量间相关性。在通用模型领域，Transformer与MoE的结合（Time-MoE, Moirai-MoE）似乎是解决Scaling Laws与推理效率矛盾的终极答案。
*   **Mamba的潜力**：虽然目前生态位尚小，但Mamba/SSM凭借其线性复杂度，是未来处理超高频（如Tick级金融数据、生物信号）长序列预测的最有力竞争者。
*   **零样本的现实意义**：基础模型（Chronos, TimesFM）的零样本能力已达到实用级别。这对于缺乏历史数据的“冷启动”场景（如新零售商品预测）具有革命性意义。
*   **推理时适应 (Inference-time Adaptation)**：TimesFM-ICF展示的“无需训练的微调”指明了未来方向——未来的预测系统将不再是静态的，而是能够通过上下文Prompt实时适应环境变化的智能体。

未来，我们有理由期待更多多模态时间序列模型的出现——它们将不仅能看懂历史数据曲线，还能结合新闻文本、卫星图像等多源信息，做出更加精准和具有可解释性的预测。时间序列预测，正从纯粹的数学拟合，迈向通用的世界模型构建。
