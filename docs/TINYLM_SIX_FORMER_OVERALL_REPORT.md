# TinyLM 六种 Former 架构总体实验报告

日期：2026-09-09  
项目：TinyStories / TinyLM 高效 Transformer 比较  
目标架构：Longformer、Memformer、Linformer、Performer、Reformer、Keyformer  
补充因素：Dense FFN、全六层 4-expert Top-1 dropless MoE  
硬件：NVIDIA GeForce RTX 4090 24GB，主要训练设备为单卡 `cuda:0`

## 摘要

本报告整合项目中已经完成的 Dense FFN、Longformer/Memformer 100M-token 扩展、C v2 Full Attention/Reformer/Keyformer，以及最新的全六层 MoE 实验，回答一个核心问题：六种 Former 架构究竟改变了 Transformer 的哪一部分，它们的质量、速度、显存和长历史能力优缺点分别从何而来？

六个名称并不处于完全相同的设计层级：Longformer、Linformer、Performer、Reformer 和 Memformer 是训练时改变 attention 或历史状态处理方式的模型路径；Keyformer 是训练完成后作用于自回归 KV cache 的推理策略。Full Attention 是精确 attention 基线，不计入六种 Former 名称之中。

当前最可靠的结论是：

- Dense 10M screening 中，Memformer 在 512-context、seed=17 的 B/A+B 统一协议下质量最好；Longformer 次之，Linformer 和 Performer 依次落后，Reformer 需要以 C v2 的独立协议解释。
- Longformer 和 Memformer 从 10M 到 100M 的质量排序发生反转：10M 时 Memformer 略优，100M 时 Longformer PPL=6.0967 优于 Memformer PPL=6.4622。这说明训练预算会改变架构结论。
- 本次全层 MoE 将总参数容量增加约 63%–73%，但 Top-1 active parameters/token 基本保持 Dense 水平。MoE 对 Memformer 的提升最明显，对 Longformer/Linformer 的提升较小，对 Performer 几乎没有提升；当前通用 PyTorch dispatch 使训练速度下降、峰值显存上升。
- Keyformer 在 Dense 和 MoE Full-Attention backbone 上都将 50% KV cache 真实压缩到约一半，并保持当前 TinyStories 窗口评估的 PPL；但 token 选择和 gather 开销使 decode 变慢约 16%–17%。
- 没有证据支持一个跨所有协议的绝对总排名。当前结果适合用于分析设计取舍，而不是宣称某个架构在所有规模、上下文和任务上全面优于其他架构。

## 1. 研究对象和设计维度

### 1.1 六种架构分别改变什么

| 架构 | 改变的主要对象 | 核心压缩/近似位置 | 是否训练型网络 |
|---|---|---|---|
| Longformer | attention 连接图 | 将全局 token-to-token attention 限制为固定左侧窗口 | 是 |
| Memformer | segment 间历史状态 | 将上一段压缩为固定数量 recurrent memory slots | 是 |
| Linformer | K/V 的序列维度 | 用低秩投影把 K/V 序列长度压到固定 pool/chunk | 是 |
| Performer | softmax attention kernel | 用 FAVOR+ 随机特征近似 softmax kernel | 是 |
| Reformer | attention 候选集合 | 用 LSH 把可能相关的 token 分桶，只计算候选连接 | 是 |
| Keyformer | 推理 KV cache | 按重要性保留部分历史 K/V，并保留 recent window | 否，推理策略 |

### 1.2 Full Attention 的角色

Full Attention 不属于六个 Former 名称，但它是必要基线。它保留完整的 causal attention 连接，质量上通常最稳，工程上也最容易使用高度优化的 SDPA/Flash kernel。Keyformer 的 backbone 可以是 Full Attention；本次实验中它进一步把六层 FFN 替换成了 MoE，因此“Full-Attention+MoE”中的 Full 只描述 attention 路径。

## 2. 证据层次与可比性

本项目包含多个阶段，不能将所有绝对 PPL 和 tok/s 直接拼成一张排行榜。

| 证据组 | 主要方法 | seed | context | 训练预算 | 作用 |
|---|---|---:|---:|---:|---|
| A+B Dense unified | Memformer、Longformer、Linformer、Performer | 17/29/43 | 512 | 10M | Dense 主横向比较；seed=17 可与 MoE 配对 |
| B historical extension | Longformer、Memformer | 17 | 512 | 100M | 判断训练预算是否改变排序 |
| C v2 Dense | Full Attention、Causal LSH Reformer | 17/29/43 | 512 | 10M | 补齐 Full/Reformer 三 seed 基线 |
| C v2 Keyformer | Dense Full checkpoint 上的 FullKV/75/50 | 17 | 窗口化 decode | N/A | Dense backbone 的推理 cache 评估 |
| 本次 MoE | 五种训练型 backbone + Full-Attention Keyformer backbone | 17 | 512 | 10M | 全六层 4-expert MoE screening |

直接可比的 Dense→MoE 结果只有 Memformer、Longformer、Linformer、Performer 四组：它们使用相同 seed=17、10M predictions 和 4,765,917 个完整 validation predictions。C v2 Full/Reformer 与 MoE 使用不同 runner 或 Reformer 候选规则，因此只能作为补充证据。

## 3. 公共 TinyLM 协议

主要 TinyStories 协议固定为：6 层、hidden size=384、8 heads、FFN=1536、Pre-LN、GELU、RoPE 最大位置 32,768、context=512、tied embeddings、FP32 参数/BF16 autocast、AdamW、有效 batch=8,192 prediction tokens。数据使用固定 TinyStories revision、GPT-2 tokenizer（词表 50,257，EOS=50,256），每篇故事追加 EOS 后按固定顺序连续 packing。

10M 实验使用 10,000,000 个有效 training predictions，完整 validation 使用 4,765,917 个 next-token predictions。训练指标中的 workflow throughput 包含周期性 probe 和 checkpoint 操作；它不是纯 attention kernel 的理论吞吐。

## 4. Dense FFN 10M 结果

### 4.1 A+B unified 的三 seed 主结果

| 方法 | Val NLL 均值±std | Val PPL 均值±std | 训练 tok/s 均值±std | Peak allocated | 参数量 |
|---|---:|---:|---:|---:|---:|
| **Memformer** | **2.849148±0.004651** | **17.273176±0.080427** | 14,669.5±300.5 | **2.633 GiB** | 33,466,752 |
| Longformer | 2.885517±0.004599 | 17.912946±0.082476 | 13,498.8±268.5 | 6.799 GiB | 29,925,504 |
| Linformer | 2.933816±0.006033 | 18.799463±0.113420 | **27,857.8±106.0** | 2.580 GiB | 29,925,504 |
| Performer | 3.450477±0.008266 | 31.516144±0.260635 | 27,079.0±541.2 | 2.704 GiB | 29,925,504 |

10M Dense 质量排序为 Memformer > Longformer > Linformer > Performer。这个排序只适用于 A+B 的统一 context=512 协议，不应外推到 C v2 或 A 组旧的 context=4096 结果。

### 4.2 机制解释

Memformer 的优势来自 recurrent memory 为 segment 之间提供了直接状态路径；它不必等待局部窗口逐层传播历史信息。代价是 memory projection、输出投影和 gate 增加约 11.83% 参数，并且压缩 slot 会丢失细节。

Longformer 的质量接近 Memformer，但其每个 query 只看到当前 token 和最多 128 个左侧 token。它保留局部 token 细节，结构简单，但远距信息必须经过多层局部传播，因此存在明显传播边界。

Linformer 通过序列维低秩池化减少 K/V 表示，当前质量居中。低秩假设在 TinyStories 上可能提供一定正则化，但它也可能丢失局部或高秩信息；pooling、projection 和通用张量操作带来实际开销。

Performer 用随机特征近似 softmax kernel，理论上可获得近似线性复杂度，但当前 features=384 的近似误差和随机特征变换成本仍然明显，导致质量最低。增加 features、延长上下文或使用更成熟 kernel 后结论可能改变。

## 5. Longformer/Memformer 的 100M 扩展

Longformer 和 Memformer 使用冻结配置、seed=17，从随机初始化分别重新训练到 100M predictions，不是从 10M checkpoint 接续。

| 指标 | Longformer window=128 | Memformer segment=128, slots=64 |
|---|---:|---:|
| 完整 Val NLL | **1.807748** | 1.865966 |
| 完整 Val PPL | **6.096701** | 6.462175 |
| 训练 tok/s | 14,743.8 | **16,813.0** |
| 训练时间 | 113.04 min | **99.13 min** |
| Peak allocated | 6.80 GiB | **2.63 GiB** |
| Peak reserved | 7.69 GiB | **3.93 GiB** |

100M 时 Longformer 的 PPL 比 Memformer 低约 5.99%，但 Memformer 吞吐高约 14.03%，训练时间少约 12.31%，peak allocated 少约 61.28%。

这个反转揭示了两个设计事实：第一，Memformer 的压缩 memory 路径在短预算下可能帮助优化，但固定 slots 的信息瓶颈不一定在充分训练后继续占优；第二，Longformer 直接保留局部 token 细节，给足预算后可能更充分地学习局部语言统计。100M 仍只有一个 seed，且曲线仍在下降，不能称为充分收敛结论。

## 6. 全六层 MoE 结果

本次 MoE 在全部 6 个 block 的 FFN 位置替换为 4 experts，每个 expert 的 FFN hidden size 仍为 1,536，采用 token-level Top-1、dropless dispatch，load-balance coefficient=0.01，router z-loss coefficient=0.001。

### 6.1 六个 MoE backbone

| 方法 | 角色 | Val NLL | Val PPL | 训练 tok/s | Peak GiB | 总参数 | 激活参数/token |
|---|---|---:|---:|---:|---:|---:|---:|
| **Memformer** | 训练型 backbone | **2.758941** | **15.7831** | 12,201.3 | 2.960 | 54,709,632 | 33,475,968 |
| Longformer | 训练型 backbone | 2.869545 | 17.6290 | 12,265.3 | 7.234 | 51,168,384 | 29,934,720 |
| Full Attention | Keyformer 共享 backbone | 2.907851 | 18.3174 | 26,503.8 | 2.911 | 51,168,384 | 29,934,720 |
| Linformer | 训练型 backbone | 2.925335 | 18.6405 | 20,340.2 | 2.913 | 51,168,384 | 29,934,720 |
| Reformer | 训练型 backbone | 2.980978 | 19.7071 | **26,821.2** | **2.900** | 50,283,648 | 29,049,984 |
| Performer | 训练型 backbone | 3.457819 | 31.7477 | 20,358.5 | 3.120 | 51,168,384 | 29,934,720 |

### 6.2 Dense FFN→MoE 的配对比较

| 方法 | Dense PPL | MoE PPL | PPL 变化 | 吞吐变化 | Peak 显存变化 |
|---|---:|---:|---:|---:|---:|
| Memformer | 17.2112 | **15.7831** | **-8.30%** | -15.55% | +12.43% |
| Longformer | 17.8841 | **17.6290** | -1.43% | -7.00% | +6.40% |
| Linformer | 18.7987 | **18.6405** | -0.84% | -26.74% | +12.89% |
| Performer | 31.7872 | 31.7477 | -0.12% | -25.79% | +15.41% |

MoE 对 Memformer 的改善最大，但它同时保留了 Memformer 的额外 memory 参数，因此不能把 8.30% 的 PPL 改善单独归因于专家路由。Longformer 和 Linformer 有小幅改善；Performer 基本没有改善，表明扩大 FFN 条件容量不能自动修复其随机特征 attention 的近似误差。

### 6.3 MoE 路由和容量解释

普通 backbone Dense 总参数为 29,925,504，MoE 总参数为 51,168,384，增加约 70.99%；Memformer 则从 33,466,752 增至 54,709,632，增加约 63.47%。但普通 backbone 的 MoE active parameters/token 为 29,934,720，只比 Dense 多约 0.031%。这正是 Top-1 MoE 的“总容量扩大、单 token 激活容量近似不变”特征。

正式训练路由比例大致为：Memformer 21.09/22.54/34.77/21.60%，Longformer 21.66/26.98/28.83/22.52%，Full 20.78/30.50/27.62/21.11%，Linformer 22.46/27.54/27.55/22.44%，Reformer 19.11/30.41/26.59/23.90%，Performer 22.72/20.82/36.03/20.43%。四个 experts 都被使用，但部分模型存在偏向 expert 3 或 expert 2 的现象，仍需更多 seed 和按层路由统计判断是否稳定。

MoE 训练速度没有随 active parameters 保持 Dense 水平，主要原因是当前 runner 使用 token indexing、逐 expert 前向和 `index_add_` 聚合，而不是 fused grouped-GEMM 或 MegaBlocks 类 kernel。dropless 表示没有因 capacity 溢出丢弃 token，不表示每个小 batch 的每个 expert 都一定收到 token。

## 7. Reformer：LSH 设计的收益与缺点

C v2 Dense Reformer 使用 bucket size=32、4 hashes、recent window=32 的因果 LSH 候选规则，三 seed 结果为 NLL `3.024±0.008`、PPL `20.58±0.16`、workflow throughput `46,939±954 tok/s`。本次 MoE Reformer 为 NLL `2.980978`、PPL `19.7071`，但使用 deterministic same-bucket causal mask 和不同 runner，因此不能把差异归因于 MoE。

Reformer 的优点来自候选筛选：如果相关 token 能被哈希到同桶，就不必计算全量 attention。它的缺点同样来自候选不完整：哈希碰撞会漏掉相关 token，多轮 hashing、排序、bucket padding 和不规则访存会增加常数开销。旧版全局排序还出现过 future-invariance 最大差异 `0.645508` 的未来信息泄漏；修复后必须在候选构造阶段施加 `key_position <= query_position` 的因果限制。

因此，Reformer 的优势需要三个条件同时满足：序列足够长、相关性适合 LSH、实现具有高效稀疏 kernel。context=512 的 TinyStories 结果尚未显示出明确收益。

## 8. Keyformer：推理期 KV-cache 压缩

Keyformer 不改变训练 backbone 参数，而是按照 token 重要性保留部分历史 K/V，并保留 recent window。它解决的是自回归 decode 的 cache 容量问题，不是训练 attention 的复杂度问题。

### 8.1 C v2 Dense backbone

| 策略 | Cache ratio | PPL | KV bytes | Decode tok/s |
|---|---:|---:|---:|---:|
| FullKV | 1.00 | 14.91 | 4,709,376 | 219.8 |
| Keyformer-100 | 1.00 | 14.91 | 4,709,376 | 219.9 |
| Keyformer-75 | 0.75 | 14.91 | 3,538,944 | 202.0 |
| Keyformer-50 | 0.50 | 14.89 | 2,359,296 | 184.2 |

### 8.2 本次 MoE Full-Attention backbone

| 策略 | Cache ratio | NLL | PPL | Peak cached tokens | Peak KV bytes | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| FullKV | 1.00 | 2.602441 | 13.4966 | 512 | 4,718,592 | 125.4 |
| Keyformer-50 | 0.50 | 2.601262 | 13.4807 | 256 | 2,359,296 | 103.8 |

MoE Keyformer-50 的 KV bytes 和 cached tokens 都减少 50%，PPL 变化为 `-0.0159`，不应解释为压缩提高了模型能力；吞吐比例为 0.8277，即下降约 17.23%。C v2 Dense 实验也观察到约 16.2% 的 decode 吞吐下降。两套实验共同说明：当前选择/gather 开销抵消了 cache 缩小带来的 attention 节省。

Keyformer 的优点是无需重新训练 backbone、可直接控制 cache ratio、物理 KV tensor 确实变小；缺点是每步选择和重排增加 latency，过度压缩可能丢失远程事实，而且它不减少训练成本。它适合显存受限的长 decode 场景，但要获得速度收益需要 fused selection、cache update 和 attention kernel。

## 9. 六种架构的优缺点总览

| 架构 | 优点 | 优点来自 | 主要缺点 | 缺点来自 |
|---|---|---|---|---|
| Longformer | 保留局部 token 细节；结构直观；长序列候选规模约为 O(NW) | 固定 causal sliding window | 远距传播有层数×窗口边界；当前窗口实现显存高 | 连接图是局部且固定的；通用 unfold/chunk kernel 开销大 |
| Memformer | 跨 segment 有直接历史路径；当前训练显存低；10M 质量好 | recurrent memory slots 和跨段 state | 压缩造成细节衰减；额外参数；推理调度复杂 | 固定容量 bottleneck、memory projection/gate、segment fusion |
| Linformer | K/V 序列维度固定压缩；当前 Dense 质量中等 | 低秩投影/池化 | 可能丢失高秩和局部信息；当前实现训练较慢 | 低秩假设和 projection/pooling 操作 |
| Performer | 理论近似线性；避免显式 softmax attention 矩阵 | FAVOR+ 随机特征 kernel | 当前 PPL 最差；特征数和随机近似敏感 | kernel 近似误差、随机特征变换和 redraw/投影开销 |
| Reformer | LSH 只计算候选相关连接；可与 reversible/chunking 结合 | hash bucket candidate filtering | 哈希碰撞漏边；排序/padding 不规则；因果实现容易出错 | 候选集合不完整，且 sparse kernel 不成熟 |
| Keyformer | 不重训即可将 KV cache 减半；当前 PPL 基本保持 | token importance retention + recent window | 当前 decode 变慢；不改变训练成本；远程信息可能被淘汰 | selection/gather/cache update 的额外路径 |

## 10. 这些优缺点是否确实来自设计？

### 已有较强证据支持的归因

1. Memformer 的跨 segment 信息通路确实存在：history effect 非零，reset 和 batch reorder 检查通过；这与 recurrent state 设计一致。
2. Longformer 的局部传播边界来自固定窗口连接图，而不是训练偶然性；在有限层数下，远距 token 需要逐层传播。
3. Reformer 的未来泄漏风险来自全局 hash 排序改变 bucket 编排；修复 causal candidate construction 后 future-invariance 通过。
4. Keyformer 的 cache 节省来自真实的 token 保留和 gather：ratio=0.5 时物理 KV bytes 直接下降约一半。
5. MoE 总参数增加但 active parameters/token 近似不变，直接来自 Top-1 routing；吞吐下降则与未融合 dispatch 实现一致。

### 仍不能完全归因的现象

1. Memformer 10M 的质量优势部分可能来自额外 3.54M memory 参数，而非纯 attention 机制。
2. MoE 对 Memformer 的 8.30% PPL 改善可能来自 memory 与 conditional FFN 的交互，也可能来自参数容量或单 seed 波动。
3. C v2 Dense Full 与本次 Full+MoE 的绝对 PPL 差异混合了 seed 数量、runner 和训练实现差异。
4. Reformer Dense 与 MoE 的差异混合了 attention 候选规则改变和 FFN 改变。
5. Linformer/Performer 的质量差异可能同时来自低秩/随机特征近似、实现适配层和训练超参数，而不是单一数学近似误差。

## 11. 当前仍存在的缺点和风险

### 11.1 实验设计缺口

- MoE 目前只有 seed=17，缺少 seeds 29/43；Performer、Linformer 等小幅 MoE 改善尚不能判定稳定。
- 尚无同一 MoE runner 下的 Dense FFN 对照；现有四组 Dense→MoE 虽然协议相同，但 runner 不同，严格消融仍应在同一代码路径完成。
- C v2 Reformer 与本次 MoE Reformer 的候选实现不同，当前不能给出严格 Reformer MoE 因果结论。
- 10M 是 screening，100M 只有 Longformer/Memformer 且单 seed；所有结果都不是充分收敛或论文级最终排名。

### 11.2 任务和评估缺口

- context=512 的 TinyStories PPL 主要反映短故事语言建模，不能证明 passkey、copy、associative recall 或跨章节实体保持能力。
- 跨故事 packing 不等于真实长对话、长文档事实依赖。
- 现有机制测试证明了状态路径和因果性，但不证明 memory 保存了正确语义。
- Keyformer 评估使用窗口化增量预测；不同报告的 prefill、decode、prediction limit 和平均/峰值 cache 口径不同，不能比较绝对 PPL。

### 11.3 工程缺口

- 当前 MoE 使用通用 token indexing、逐 expert GEMM 和 `index_add_`，尚未接入 fused grouped-GEMM/MegaBlocks；因此训练速度不能代表优化 kernel 上限。
- 当前 Keyformer selection/gather 未融合，压缩 cache 尚未转化为 decode 加速。
- Longformer 的 unfold/chunk、Memformer 的 segment fusion、Reformer 的 sort/bucket 都存在 Python/PyTorch adapter 常数开销。
- 大量 experts 增加 checkpoint、optimizer state 和参数存储；MoE 不是训练存储压缩方法。

## 12. 结论与推荐

如果目标是短中序列语言建模并追求当前实验质量，Memformer 在 10M screening 和全层 MoE seed=17 结果中最有吸引力，但必须接受额外 memory 参数和固定容量压缩。若目标是结构简单、局部 token 细节和较清晰的传播解释，Longformer 更容易分析；100M 结果还显示它在更高预算下可能反超 Memformer。

如果目标是降低 attention 的序列维表示或 softmax 计算，Linformer 和 Performer 提供了不同的近似路径，但当前小模型和实现尚未体现工程速度优势；应在更长 context、更大 hidden size 和成熟 kernel 下重新评估。Reformer 的潜在收益依赖真正长序列、合适的 hash locality 和严格因果的稀疏 kernel，当前 context=512 结果不足以支持其优势。

如果主要瓶颈是自回归 decode 的 KV cache，Keyformer 是最直接的补充方案。它可以与 Full Attention、MoE、甚至其他训练 backbone 正交组合；当前证据支持“50% cache、PPL 基本不变”，但不支持“50% cache 必然加速”。

综合而言，六种方案不是同一条技术轴上的互斥替代品：Longformer/Memformer 改变历史连接，Linformer/Performer/Reformer 近似 attention 计算，Keyformer 压缩推理状态，MoE 则扩大 FFN 条件容量。最合理的系统设计可能是按瓶颈组合它们，而不是追求单一架构总冠军。

## 13. 下一步实验

1. 在同一 runner 中为六种 MoE backbone 补跑 seeds 29/43，并报告均值、标准差和 paired-seed 差异。
2. 加入同代码路径的 Dense FFN / `num_experts=1` 对照，隔离 MoE 路由本身的因果效果。
3. 在 C v2 的 bucket=32、hashes=4、recent=32 Reformer 路径中只替换 FFN，形成严格 Reformer MoE 消融。
4. 对 Full Attention Dense/MoE 都跑 seeds 17/29/43，并用同一 Keyformer evaluator 测 FullKV、75% 和 50% cache。
5. 训练 passkey、copy、associative recall、实体属性保持和跨 segment 干扰任务，报告 exact match、距离衰减和错误类型。
6. 接入 fused grouped-GEMM、token dispatch 和 cache selection kernel，重新测量训练吞吐、decode latency、峰值显存和真实 energy/token。
7. 绘制质量—吞吐—显存—参数—KV bytes 的 Pareto 曲线，并分别给出训练架构榜和推理策略榜。

## 14. 可追溯文件

### 本次 MoE

- 配置：[experiments/tinystories_tinylm_moe_v1/moe_seed17.yaml](../experiments/tinystories_tinylm_moe_v1/moe_seed17.yaml)
- runner：[experiments/tinystories_tinylm_moe_v1/run_moe_six.py](../experiments/tinystories_tinylm_moe_v1/run_moe_six.py)
- 聚合摘要：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/summary.json`
- 六模型结果：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/SUMMARY.md`
- correctness：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/correctness.json`

### Dense、B 扩展和 C v2

- A+B Dense 三 seed：[PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md](PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md)
- B 原始详细报告：[PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md](PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md)
- C v2：[C_UNIFIED_RESULT_REPORT_V2.md](C_UNIFIED_RESULT_REPORT_V2.md)
- A+B 逐 run 聚合：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/per_run.csv`
- Longformer 100M：`/root/autodl-tmp/26summerBDMI_transformer/runs/person_b_extended/extended100m_longformer_w128_s17/summary.json`
- Memformer 100M：`/root/autodl-tmp/26summerBDMI_transformer/runs/person_b_extended/extended100m_memformer_s128_m64_s17/summary.json`

### 相关协议

- [UNIFIED_EXPERIMENT_PROTOCOL.md](UNIFIED_EXPERIMENT_PROTOCOL.md)
- [TINYLM_EXPERIMENT_QA.md](TINYLM_EXPERIMENT_QA.md)
- [文档索引](README.md)
