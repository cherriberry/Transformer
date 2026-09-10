# Adaptive Gated Memory：当前实验结果与论文式分析

日期：2026-09-10  
实验阶段：TinyStories 骨干 + 结构化选择性事实记忆  
主模型：SWA-sink local attention + adaptive gated fact memory + value-token alignment  
主实验 seed：17

## 摘要

本阶段研究一个具有固定局部 KV 缓存和有界长期语义记忆的 TinyLM。模型使用 4 个 sink token 与 124 个 recent token 组成局部窗口，并通过内容门控 fact memory 保存需要跨越长干扰上下文的事实。针对旧 Gated 模型“能够检索事实但不能稳定输出具体答案 token”的问题，进一步加入 value-span extraction、lexical value state 和 memory-to-token auxiliary loss。

在统一的 selective-memory stress protocol 中，完整模型在 288 条 held-out 样本上达到 100% Exact Match（EM）、100% 目标事实存活率、100% Recall@1 和 1.0000 MRR。最长条件包含 8 个临时事实和 16 个 TinyStories 延迟 segment，仍只激活平均 1 个长期记忆槽位。相对于旧 Gated 模型，EM 从 52.43% 提升到 100%，完整状态仅增加 0.61%。

该结果支持以下论文结论：事实级门控记忆能够在固定状态预算内抵抗长上下文噪声；value-span 与 token-level alignment 能够修复“语义记忆已检索但答案 token 排名不稳定”的结构性缺陷。但当前仍只有 seed 17、16 个封闭颜色值和受控模板，不能直接外推到开放域事实记忆。

## 1. 研究问题

普通 Sliding Window/SWA 只能保留局部 token KV。当重要事实超出窗口后，模型无法依靠局部 KV 恢复它。另一方面，单纯将事实压缩为一个连续向量可能保留语义类别，却无法精确输出 value 的具体 token。

本实验检验两个问题：

1. 内容门控槽位能否在长延迟和临时事实噪声下保留重要事实？
2. 显式 value span 与 memory-to-token 监督能否把检索后的语义记忆转换为精确答案？

## 2. 模型结构

```text
TinyLM
├── local attention：4 sinks + 124 recent
├── fact extractor：fact span + value span
├── semantic key/value encoder
├── lexical value encoder
├── write gate / retention gate / merge-update
├── bounded memory slots：8
├── sparse Top-k reader：k=4
├── semantic + lexical memory fusion
└── tied LM head + memory-to-token auxiliary head
```

局部 KV 负责短期上下文；semantic fact slot 负责长期事实；lexical value 和辅助 token loss 负责精确恢复。辅助 head 只用于训练，推理仍由普通自回归 LM head 生成答案。

## 3. 实验设置

### 3.1 模型与硬件

| 项目 | 设置 |
|---|---:|
| Transformer layers | 6 |
| hidden size | 384 |
| heads | 8 |
| FFN size | 1536 |
| local KV | 4 sinks + 124 recent |
| fact slots | 8 |
| reader Top-k | 4 |
| max write candidates | 1 |
| payload budget | 12 tokens |
| GPU | RTX 4090 24 GB |
| 精度 | CUDA BF16 autocast |

### 3.2 数据协议

每条样本为：

```text
durable user fact
→ temporary user facts
→ TinyStories distractor tokens
→ user query
→ assistant answer
```

训练噪声为 0、1、2、4 条临时事实，延迟为 1、2、4 个 128-token segment。验证噪声为 0、2、4、8，延迟为 1、4、16 个 segment，每个交叉条件 24 条，共 288 条。

训练和验证使用不同事实提示模板；验证包含未见的 durable/noise cue，以检验模板迁移。

### 3.3 优化

| 项目 | 设置 |
|---|---:|
| seed | 17 |
| optimizer | AdamW |
| steps | 2000 |
| batch size | 2 |
| gradient accumulation | 2 |
| backbone LR | 3e-5 |
| memory/new-head LR | 3e-4 |
| weight decay | 0.1 |
| gradient clip | 1.0 |

## 4. 对比条件

| 条件 | 说明 |
|---|---|
| SWA-sink only | 只有 4 sinks + 124 recent，无长期事实槽位 |
| 旧 Gated | adaptive write/retention memory，但无 value-token alignment |
| Fixed-LRU | 所有有效候选都写入，满容量后按 LRU 淘汰 |
| 新 Gated | 旧 Gated + value span + lexical value + memory-to-token auxiliary loss |

## 5. 主要结果

| 条件 | EM | 目标存活率 | Recall@1 | MRR | 平均 active slots | noise accept rate |
|---|---:|---:|---:|---:|---:|---:|
| SWA-sink only | 5.21% | N/A | N/A | N/A | 0.00 | N/A |
| 旧 Gated | 52.43% | 100.00% | 100.00% | 1.0000 | 1.00 | 0.00% |
| Fixed-LRU | 74.65% | 92.01% | 90.28% | 0.9103 | 4.25 | 75.00% |
| **新 Gated + value-token** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.00** | **0.00%** |

### 5.1 长延迟和高噪声

新模型在 12 个验证条件全部达到 100% EM，包括：

- 8 条 temporary user facts；
- 16 个 128-token delay segments，约 2048 个干扰 token；
- 目标事实仍保持 100% survival；
- 平均只使用 1 个 active fact slot。

### 5.2 词级恢复

| 指标 | 新模型 |
|---|---:|
| value start accuracy | 100% |
| value length accuracy | 100% |
| value span exact | 100% |
| memory-token token accuracy | 100% |
| memory-token sequence exact | 100% |
| answer first-token mean rank | 1.0000 |
| answer first-token Top-5 | 100% |
| answer first-token logit margin | +13.5763 |
| teacher-forced answer NLL | 0.0000195 |

288 条生成均以正确答案 token 序列开头并正常生成 EOS；多 token value `navy` 也被正确恢复。

## 6. Checkpoint 稳定性

| Checkpoint | EM | 首 token 平均排名 | Logit margin | Auxiliary value exact |
|---:|---:|---:|---:|---:|
| 500 | 100% | 1.00 | +6.8794 | 94.79% |
| 1000 | 100% | 1.00 | +9.1122 | 100% |
| 1500 | 100% | 1.00 | +11.4877 | 100% |
| 2000 | 100% | 1.00 | +13.5763 | 100% |

主生成在 500 步已经达到 100% EM，继续训练主要增加答案置信度并使多 token value 的辅助头完全收敛。

## 7. 门控机制分析

### 7.1 相比 Fixed-LRU

Fixed-LRU 的噪声接收率为 75%，平均激活 4.25 个槽位，目标 survival 为 92.01%。新 Gated 将噪声接收率降至 0%，平均只保留 1 个槽位，同时 survival 提升至 100%。

这说明门控的价值不是简单增加存储，而是进行内容级选择：重要事实进入长期记忆，临时事实被拒绝。

### 7.2 相比旧 Gated

旧 Gated 已经实现了事实写入、保留和检索：目标 survival、Recall@1 和 MRR 均为 100%，但 EM 只有 52.43%。新模型在不改变 slot 数量和检索结果的情况下，把 EM 提升到 100%。

因此旧模型的主要瓶颈不是“记忆丢失”，而是“检索后的词级答案映射”；value-span 与 memory-to-token alignment 针对性地解决了这个瓶颈。

## 8. 存储和效率

| 项目 | 旧 Gated | 新 Gated | 变化 |
|---|---:|---:|---:|
| 总参数量 | 31,854,750 | 32,160,043 | +0.96% |
| 完整 recurrent state | 2,408,906 B | 2,423,702 B | +0.61% |
| active slot logical bytes | 3,221 B | 4,865 B | +51.0% |
| 2000-step 训练时间 | 1323.0 s | 1380.1 s | +4.31% |
| 训练吞吐 | 6.047 ex/s | 5.797 ex/s | -4.14% |
| 峰值 allocated 显存 | 2.673 GB | 2.682 GB | +0.36% |

完整 recurrent state 包含局部 SWA KV、semantic memory、lexical value、payload IDs、pending state 和 metadata。状态由固定 slot 数封顶，不随历史长度线性增长。

## 9. 结论

当前证据支持以下结论：

1. 单纯 SWA-sink 无法可靠恢复超出局部窗口的事实；
2. adaptive gated fact memory 能以一个槽位保留重要事实并过滤临时事实；
3. value-span 和 memory-to-token alignment 能显著改善精确答案恢复；
4. 新架构在当前受控任务中同时取得长程 recall、低状态占用和精确 EM。

## 10. 局限性

本结果仍不能直接宣称开放域普适性，原因包括：

- 只有 seed 17；
- 答案集合为 16 个颜色值；
- 事实关系和模板较固定；
- value span 使用监督标签；
- 尚未完成 seed 29/43；
- 尚未完成数字、日期、实体名和罕见多 token 字符串测试；
- 尚未完成 value span、lexical fusion、token loss 的独立消融。

## 11. 后续 MoE 实验计划

下一阶段将在相同 TinyLM 骨干、相同数据、相同 seed 和相同训练预算下，为以下四个条件加入同一套 top-2 MoE：

1. SWA-sink only + MoE；
2. Fixed-LRU + MoE；
3. 旧 Gated + MoE；
4. 新 Gated/value-token + MoE。

MoE 只替换 Transformer FFN，不改变局部 KV、memory slots、reader 或 value-token 机制。将记录 EM、NLL、Recall、active slots、MoE expert load、训练吞吐、推理延迟和峰值显存。MoE 默认关闭，以保证旧实验仍可复现。

正式论文结论应同时报告：

- memory-only 的架构效果；
- MoE-enhanced 的架构效果；
- MoE 带来的质量增益是否值得其参数、显存和吞吐开销。

