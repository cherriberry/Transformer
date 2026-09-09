# TinyLM 六变体全层 MoE 统一实验报告

日期：2026-09-09  
数据集：TinyStories  
硬件：NVIDIA GeForce RTX 4090 24GB，单卡 `cuda:0`  
MoE 配置：全部 6 层、4 experts、Top-1 routing、dropless dispatch  
训练预算：每个训练 backbone 10,000,000 prediction tokens，seed=17

## 摘要

本实验在统一 TinyLM 中将全部 6 个 Transformer block 的 Dense FFN 替换为 4-expert、Top-1、dropless MoE，并完成 Memformer、Linformer、Performer、Longformer、Reformer，以及供 Keyformer 使用的 Full-Attention backbone 共六个 seed-17 训练任务。Keyformer 不是独立训练网络，而是在训练完成的 Full-Attention+MoE checkpoint 上执行推理期 KV-cache 压缩。

六个训练任务均完成 10M prediction tokens 和完整 validation。当前 MoE 质量排序为 Memformer、Longformer、Full Attention、Linformer、Reformer、Performer，其中 Memformer 的完整 validation NLL/PPL 最低，为 `2.758941 / 15.7831`。与相同 seed、相同 10M-token Dense FFN 基线直接比较，MoE 将 Memformer、Longformer、Linformer、Performer 的 PPL 分别降低约 `8.30%`、`1.43%`、`0.84%` 和 `0.12%`；与此同时，当前未融合专家 dispatch 实现使训练吞吐下降约 `7%–27%`，峰值 allocated memory 增加约 `6%–15%`。

C v2 提供了此前缺少的三 seed Dense Full-Attention 结果。Full-Attention+MoE 的 seed-17 PPL 为 `18.3174`，低于 C v2 Dense Full Attention 三 seed 均值 `18.95 ± 0.12`，但这是“单 seed 对三 seed 均值”，且 runner 不同，只能视为有利的 screening 信号，不能作为严格显著性结论。

在 Full-Attention+MoE checkpoint 上，Keyformer-50 将物理 KV-cache 从 `4,718,592` bytes 降至 `2,359,296` bytes，恰好减少 `50%`；32,768 个增量预测上的 PPL 从 FullKV 的 `13.4966` 变为 `13.4807`，没有观察到质量损失，但当前选择和 gather 路径使吞吐下降约 `17.23%`。C v2 的 Dense FFN Keyformer 实验也观察到约一半 KV 容量、近乎不变的 PPL 和约 `16.2%` 的 decode 吞吐下降，说明该资源—延迟权衡在 Dense 与 MoE backbone 上方向一致。

## 1. 研究问题

本报告回答以下问题：

1. 将六层 Dense FFN 全部替换为 Top-1 MoE 后，质量、训练吞吐、显存和参数规模如何变化？
2. MoE 的收益是否在不同 attention/memory backbone 上一致？
3. Full-Attention+MoE 是否可以作为 Keyformer 的共享训练 backbone？
4. Keyformer 在 Dense FFN 和 MoE backbone 上是否都能真实减少物理 KV-cache？
5. 哪些结果可以直接比较，哪些只能作为跨报告参考？

## 2. 证据层次与比较口径

本报告使用三组证据。它们不能全部混成一个统计总体。

| 证据 | 方法 | seeds | 预算 | 用途 |
|---|---|---:|---:|---|
| A+B unified Dense | Memformer、Longformer、Linformer、Performer | 17/29/43 | 每 run 10M | 其中 seed=17 与本次 MoE 构成最严格 FFN 消融 |
| C protocol v2 Dense | Full Attention、Causal LSH Reformer；另有 Keyformer cache 评估 | 17/29/43 | 每 run 10M | 补齐 Full Attention/Reformer/Keyformer 证据 |
| 本次 MoE | 五种训练型 attention/memory backbone，加一个 Keyformer 共享 Full-Attention backbone | 17 | 每 backbone 10M | 检验全六层 MoE 的可行性和单 seed screening 效果 |

直接比较规则如下：

- Memformer、Longformer、Linformer、Performer 使用相同 seed=17、模型维度、训练 token 流、预算和完整 validation，可以做 Dense FFN 对 MoE 的直接 screening 对照。
- C v2 Full Attention 当前报告只给出三 seed 均值，本次 MoE 只有 seed=17，因此只能比较量级和方向，不能做配对显著性判断。
- C v2 Reformer 使用 `bucket size=32、4 hashes、recent window=32` 的候选规则；本次 MoE Reformer 使用固定 seed 的严格因果同桶 mask。两者 attention 候选定义不同，不能把差异全部归因于 MoE。
- Keyformer 是 inference-time cache policy。其 NLL/PPL 只能与同一 checkpoint 的 FullKV 对照，不能作为独立训练模型加入训练质量排名。
- C v2 与本次 MoE 的 Keyformer prefill、计时和 cache 统计口径不同，因此只比较相对 KV 节省和相对质量/速度变化，不比较绝对 NLL 或绝对 tok/s。

## 3. 统一数据、模型与训练协议

### 3.1 数据与 token 流

| 项目 | 设置 |
|---|---|
| 数据集 | `roneneldan/TinyStories` |
| 数据 revision | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| tokenizer | `openai-community/gpt2` |
| tokenizer revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| vocabulary / EOS | 50,257 / 50,256 |
| packing | 每篇故事追加 EOS 后按固定顺序连续拼接 |
| train cache | 10,000,385 tokens，实际训练 10,000,000 predictions |
| validation cache | 4,765,918 tokens，完整评估 4,765,917 predictions |
| train token SHA-256 | `a25e6c436b1cfcde1bcee54724563931763b40e034abd99e63d2286dbdf20f17` |
| validation token SHA-256 | `d5e1575253e3489e2af831dee209d5ad56a6bb9a4bf02854d66fc80dddc2a9e2` |

所有方法使用同一固定 token 顺序。跨故事 packing 提高了 token 利用率，但不等价于真实长文档或跨文档事实记忆任务。

### 3.2 公共 TinyLM

| 项目 | 设置 |
|---|---:|
| decoder layers | 6 |
| hidden size | 384 |
| attention heads | 8 |
| head dimension | 48 |
| context | 512 |
| RoPE maximum position | 32,768 |
| Dense FFN hidden size | 1,536 |
| normalization | Pre-LN |
| activation | GELU |
| dropout | 0.1 |
| embedding/output | tied |

“Full Attention”描述的是 attention 路径，而不是 FFN 类型。本次 Keyformer 共享 backbone 使用精确 Full Attention，但其六层 FFN 均为 MoE。

### 3.3 MoE 配置

| 项目 | 设置 |
|---|---|
| 替换层 | block 1–6，全部替换 |
| experts per layer | 4 |
| expert FFN hidden size | 每个 expert 1,536 |
| routing | token-level Top-1 |
| capacity/drop policy | dropless，无 token 因容量上限被丢弃 |
| load-balance coefficient | 0.01 |
| router z-loss coefficient | 0.001 |
| validation objective | 只报告 LM token-weighted cross-entropy，不加入 router auxiliary loss |

Top-1 使每个 token 每层只激活一个 expert。因此总参数量显著增加，但单 token 激活的 FFN 参数规模接近 Dense FFN；router 每个普通 backbone 仅增加 9,216 个激活参数。

### 3.4 优化与精度

| 项目 | 设置 |
|---|---|
| optimizer | AdamW |
| learning rate | peak `3e-4`，3% warmup，cosine decay 至 `3e-5` |
| betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| weight decay | 0.1 |
| gradient clipping | global norm 1.0 |
| parameter / compute dtype | FP32 / BF16 autocast |
| TF32 | disabled |
| micro batch / accumulation | 4 sequences / 4 steps |
| effective batch | 8,192 prediction tokens |
| seed | 17 |

## 4. Correctness 闸门

聚合 correctness 状态为 `passed`。已检查：

- 六层 MoE 前向输出 shape 正确，logits 和 loss 有限；
- 反向传播和梯度数值有限；
- 所有 attention/memory 方法的 future-invariance 最大差异为 0；
- FullKV prefill/decode 与完整前向误差分别不超过 `0.00390625 / 0.001953125`，低于 `0.02` 容差；
- Keyformer cache ratio=1 与 FullKV 的 prefill/decode 最大差异均为 0；
- 正式训练的全局路由统计中，六个 backbone 的四个 experts 均得到非零 token，未出现全局 expert collapse。

在 Longformer 的一个很小 correctness batch 中，第六层 expert 2 恰好没有接收 token，因此该 isolated batch 中对应两个权重没有梯度。这不表示 token 被丢弃：Top-1 dropless 保证每个 token 被路由，但不保证每个小 batch 的每个 expert 都收到 token。正式 10M-token 路由统计显示四个 experts 均被使用。

## 5. 六个 MoE 训练结果

下表均为 seed=17、10M training predictions、4,765,917 个完整 validation predictions。

| 方法 | 角色 | Val NLL | Val PPL | 训练 tok/s | Peak GiB | 总参数 | 激活参数/token |
|---|---|---:|---:|---:|---:|---:|---:|
| **Memformer** | trainable backbone | **2.758941** | **15.7831** | 12,201.3 | 2.960 | 54,709,632 | 33,475,968 |
| **Longformer** | trainable backbone | 2.869545 | 17.6290 | 12,265.3 | 7.234 | 51,168,384 | 29,934,720 |
| **Full Attention** | Keyformer shared backbone | 2.907851 | 18.3174 | 26,503.8 | 2.911 | 51,168,384 | 29,934,720 |
| **Linformer** | trainable backbone | 2.925335 | 18.6405 | 20,340.2 | 2.913 | 51,168,384 | 29,934,720 |
| **Reformer** | trainable backbone | 2.980978 | 19.7071 | **26,821.2** | **2.900** | 50,283,648 | 29,049,984 |
| **Performer** | trainable backbone | 3.457819 | 31.7477 | 20,358.5 | 3.120 | 51,168,384 | 29,934,720 |

这里的训练 tok/s 是训练阶段 workflow throughput，包含周期性 validation probe；不包含训练结束后的完整 validation 与 checkpoint 保存。不同方法的 Python/PyTorch adapter 调度成本也包含在该指标中。

### 5.1 Expert 路由分布

| 方法 | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|---|---:|---:|---:|---:|
| Memformer | 21.09% | 22.54% | 34.77% | 21.60% |
| Longformer | 21.66% | 26.98% | 28.83% | 22.52% |
| Full Attention | 20.78% | 30.50% | 27.62% | 21.11% |
| Linformer | 22.46% | 27.54% | 27.55% | 22.44% |
| Reformer | 19.11% | 30.41% | 26.59% | 23.90% |
| Performer | 22.72% | 20.82% | 36.03% | 20.43% |

所有模型都使用了四个 experts，但分布不是严格均匀。Performer 和 Memformer 的最大 expert 占比分别达到约 `36.03%` 和 `34.77%`，仍未形成单一 expert 垄断全部 token 的 collapse。

## 6. Dense FFN 与 MoE 的直接对照

本节只使用 A+B unified 的 seed=17 Dense run，与本次同 seed MoE run 比较。两侧均训练 10M predictions，并使用相同的 4,765,917-token 完整 validation。

| 方法 | Dense NLL / PPL | MoE NLL / PPL | PPL 相对变化 | Dense → MoE tok/s | Dense → MoE Peak GiB |
|---|---:|---:|---:|---:|---:|
| Memformer | 2.845559 / 17.2112 | **2.758941 / 15.7831** | **-8.30%** | 14,448.6 → 12,201.3（-15.55%） | 2.633 → 2.960（+12.43%） |
| Longformer | 2.883914 / 17.8841 | **2.869545 / 17.6290** | -1.43% | 13,188.8 → 12,265.3（-7.00%） | 6.799 → 7.234（+6.40%） |
| Linformer | 2.933785 / 18.7987 | **2.925335 / 18.6405** | -0.84% | 27,766.0 → 20,340.2（-26.74%） | 2.580 → 2.913（+12.89%） |
| Performer | 3.459063 / 31.7872 | 3.457819 / 31.7477 | -0.12% | 27,434.6 → 20,358.5（-25.79%） | 2.704 → 3.120（+15.41%） |

### 6.1 参数容量与激活计算

| backbone 类型 | Dense 总参数 | MoE 总参数 | 总参数变化 | MoE 激活参数/token | 相对 Dense 激活参数变化 |
|---|---:|---:|---:|---:|---:|
| 普通 backbone | 29,925,504 | 51,168,384 | +70.99% | 29,934,720 | +0.031% |
| Memformer | 33,466,752 | 54,709,632 | +63.47% | 33,475,968 | +0.028% |

Top-1 MoE 的核心现象是“总容量增加、单 token 激活容量近似不变”。但是总参数仍会扩大 checkpoint、optimizer state 和参数/梯度存储；同时当前实现按 expert 做索引、聚合和多个小矩阵计算，尚未使用 fused grouped-GEMM，因此实际吞吐并没有保持 Dense 水平。

### 6.2 直接对照结论

- Memformer 获得最明显的质量改善，NLL 降低 `0.086618`，PPL 降低 `1.4280`。
- Longformer 的 PPL 改善约 `1.43%`，同时吞吐只下降约 `7%`，是四个直接对照中相对温和的资源代价。
- Linformer 的 PPL 改善不足 `1%`，而吞吐下降约 `26.7%`。
- Performer 的 PPL 仅改善约 `0.12%`，在当前单 seed 结果中基本可以视为没有清晰质量收益。
- 只有一个 MoE seed。尤其是 Linformer 和 Performer 的小幅差异，需要 seeds 29/43 才能判断是否超出初始化方差。

## 7. C v2 与本次 MoE 的补充比较

### 7.1 Full Attention

| 配置 | seeds | NLL | PPL | workflow tok/s | Peak GiB | 总参数 | 激活参数/token |
|---|---|---:|---:|---:|---:|---:|---:|
| C v2 Full Attention + Dense FFN | 17/29/43 | 2.942 ± 0.006 | 18.95 ± 0.12 | 53,770 ± 585 | 2.58 | 29,925,504 | 29,925,504 |
| Full Attention + MoE | 17 | 2.907851 | 18.3174 | 26,503.8 | 2.911 | 51,168,384 | 29,934,720 |

相对 C v2 Dense 三 seed 均值，MoE seed-17 的 NLL 低约 `1.16%`、PPL 低约 `3.34%`；总参数增加约 `70.99%`，单 token 激活参数只增加约 `0.031%`。但吞吐下降约 `50.7%`，明显大于 A+B 四种 backbone 的 MoE 降幅，提示 Full-Attention MoE runner 与 C v2 runner 的工程路径/计时实现可能贡献了额外差异。该表不能替代同 runner、同 seed 的配对消融。

### 7.2 Reformer

| 配置 | seeds | NLL | PPL | workflow tok/s | Peak GiB | 总参数 |
|---|---|---:|---:|---:|---:|---:|
| C v2 Dense Causal LSH，b32/h4/recent32 | 17/29/43 | 3.024 ± 0.008 | 20.58 ± 0.16 | 46,939 ± 954 | 2.57 | 29,040,768 |
| 本次 MoE Reformer，deterministic same-bucket causal mask | 17 | 2.980978 | 19.7071 | 26,821.2 | 2.900 | 50,283,648 |

MoE 行的质量数值优于 C v2 三 seed 均值，但 attention 候选规则和 runner 同时改变，因此不能声称这 `4.24%` 的 PPL 差异由 MoE 单独造成。严格 Reformer MoE 消融需要在 C v2 的 b32/h4/recent32 attention 路径中只替换 FFN，并复用 seeds 17/29/43。

## 8. Keyformer：Dense 与 MoE backbone 上的 KV-cache 结果

### 8.1 C v2：Dense Full-Attention backbone

| 策略 | Cache ratio | NLL | PPL | KV 平均 bytes | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| FullKV | 1.00 | 2.702 | 14.91 | 4,709,376 | 219.8 |
| Keyformer-100 | 1.00 | 2.702 | 14.91 | 4,709,376 | 219.9 |
| Keyformer-75 | 0.75 | 2.702 | 14.91 | 3,538,944 | 202.0 |
| Keyformer-50 | 0.50 | 2.701 | 14.89 | 2,359,296 | 184.2 |

Keyformer-100 与 FullKV 的 NLL 差为 0，等价性检查通过。Keyformer-50 将平均 KV bytes 减少约一半，PPL 没有观察到退化，但 decode 吞吐下降约 `16.2%`。

### 8.2 本次实验：MoE Full-Attention backbone

评估使用 32,768 个 validation predictions，prefill=128、context=512。

| 策略 | Cache ratio | NLL | PPL | Peak cached tokens | Peak KV bytes | 评估 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| FullKV | 1.00 | 2.602441 | 13.4966 | 512 | 4,718,592 | 125.4 |
| Keyformer-50 | 0.50 | 2.601262 | 13.4807 | 256 | 2,359,296 | 103.8 |

Keyformer-50 的结果为：

- NLL 变化 `-0.001179`，PPL 变化 `-0.0159`，该微小改善不应解释为压缩提高了模型能力；
- peak cached tokens 和 peak KV bytes 均减少 `50%`；
- 吞吐比为 `0.8277`，即下降约 `17.23%`。

### 8.3 跨 Dense/MoE 的稳定结论

两套协议虽然绝对 NLL、prefill 和计时定义不同，但都支持以下有限结论：

1. cache ratio=1 的 Keyformer 路径可以与 FullKV 数值等价；
2. cache ratio=0.5 能真实把物理 KV 存储降至约一半；
3. 在当前 Python/PyTorch 选择与 gather 实现中，减少 KV 容量没有带来 decode 加速，反而产生约 `16%–17%` 的吞吐损失；
4. 当前 TinyStories 窗口评估没有观察到 50% cache 引起的 PPL 退化；这不代表更长依赖任务也一定无损。

## 9. 结果解释

### 9.1 质量

全层 MoE 并没有在所有 backbone 上产生同等收益。Memformer 改善最大，可能说明 recurrent memory 压缩后的 token 表征更能利用条件专家容量；但 Memformer 同时具有更多方法专属参数，且目前只有一个 MoE seed，不能把该现象直接解释成 memory 与 MoE 的普遍协同规律。Performer 几乎没有改善，说明增加 FFN 总容量不能自动修复 attention 近似带来的误差或当前特征数限制。

### 9.2 训练效率

Top-1 保持了近似 Dense 的 active-parameter count，却没有保持 Dense 吞吐。主要原因是当前 runner 使用通用 PyTorch token indexing、逐 expert 前向和 `index_add_` 聚合，不是专用 grouped-GEMM/megablocks kernel。当前结果反映“算法配置 + 本仓库实现”的端到端成本，而不是优化后 MoE kernel 的上限。

### 9.3 显存与存储

MoE 的训练峰值显存增幅小于总参数增幅，因为峰值还受到 activations、attention workspace、logits 和 allocator 行为影响；但 final checkpoint 需要保存所有 experts 以及 AdamW optimizer state，单个 checkpoint 约为 `0.60–0.66 GB`。MoE 适合扩大模型总容量，不等价于降低训练参数存储。

### 9.4 Memory 与 KV compression

Memformer 与 Keyformer 解决的是两个不同层面的 memory 问题：

- Memformer 是训练期和推理期都参与计算的 recurrent memory architecture；
- Keyformer 是训练后作用于精确 attention backbone 的 KV-cache eviction/retention policy；
- MoE 位于 FFN 子层，与二者正交，可以同时组合，但指标必须分开报告。

## 10. 局限性

1. 本次 MoE 只有 seed=17，Dense A+B/C 基线有三个 seeds；小幅提升缺少方差估计。
2. 10M tokens 是 screening 预算，所有模型的 probe 曲线末端仍可能继续下降，不能视为充分收敛。
3. 不同 attention 方法的总参数并非严格匹配；Memformer 保留额外 memory projection 和 gate。
4. MoE 总参数增加约 63%–73%，因此质量改善不能表述为“同参数量算法净收益”。公平口径是 active parameters/token 近似匹配。
5. C v2 Reformer 与本次 MoE Reformer 的候选规则不同，不能做严格 MoE 因果归因。
6. C v2 与本次 Keyformer 的窗口、prefill、计时和 average/peak cache 统计不同，绝对 NLL 与吞吐不能跨表比较。
7. TinyStories 以短故事为主，context=512 的语言建模 PPL 不能验证 passkey、跨章节实体保持或长期事实检索。
8. Dropless 表示不因 expert capacity 丢弃 token，不表示每个小 batch 中四个 experts 必然都有 token。
9. 当前实现没有 fused MoE kernel，吞吐结论不能外推到 DeepSpeed-MoE、MegaBlocks 或其他优化实现。

## 11. 结论

在本次 TinyLM/TinyStories、seed=17、10M-token screening 中，六层 4-expert Top-1 dropless MoE 已经在全部六个请求变体上稳定运行。最强质量结果来自 Memformer+MoE，完整 validation PPL 为 `15.7831`；相对同 seed Dense Memformer 下降约 `8.30%`。Longformer 和 Linformer 有小幅改善，Performer 基本持平。MoE 通过将总参数容量扩大约 63%–73%，在保持单 token 激活参数近似不变的同时改善了部分 backbone 的质量，但当前非融合实现带来训练吞吐下降和小幅峰值显存增加。

Full-Attention+MoE 可以作为 Keyformer 的共享 backbone。Dense C v2 与本次 MoE 两套 Keyformer 实验都证明 50% cache 能真实减少约一半 KV 存储，并在当前评估窗口上保持 PPL；但当前实现均因 token 选择与 gather 开销而变慢。因此现阶段最稳妥的结论是：

> 全层 Top-1 MoE 为部分 memory/attention backbone 提供了额外质量容量，其中 Memformer 的单 seed 收益最明显；Keyformer 与 MoE 可以正交组合并将 KV-cache 减半，但要获得可推广的质量结论和实际速度收益，仍需补齐多 seed、严格同 runner 消融、真实长依赖任务和 fused expert/cache kernel。

## 12. 下一步实验建议

1. 为全部 MoE backbone 补跑 seeds 29/43，报告均值、标准差和 paired-seed 差异。
2. 在同一 MoE runner 中加入 `num_experts=1` 或原 Dense FFN 开关，形成完全同代码路径的严格消融。
3. 在 C v2 b32/h4/recent32 Reformer 上只替换 FFN，隔离 attention 实现差异。
4. 对 Full Attention 同时运行 Dense/MoE seeds 17/29/43，并使用同一 Keyformer evaluator。
5. 加入 passkey、copy、associative recall、实体属性恢复和跨 segment 干扰任务。
6. 接入 fused grouped-GEMM 或 MegaBlocks 类 dropless kernel，重新测量训练吞吐和峰值显存。
7. 将 Keyformer 的 selection/gather 融合，分别报告 prefill、decode、KV bytes、选择开销和端到端 latency。

## 13. 可追溯文件

### 本次 MoE

- 配置：[`experiments/tinystories_tinylm_moe_v1/moe_seed17.yaml`](../experiments/tinystories_tinylm_moe_v1/moe_seed17.yaml)
- runner：[`experiments/tinystories_tinylm_moe_v1/run_moe_six.py`](../experiments/tinystories_tinylm_moe_v1/run_moe_six.py)
- 聚合报告：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/SUMMARY.md`
- 完整 JSON：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/summary.json`
- correctness：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/correctness.json`
- Keyformer 评估：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/keyformer_evaluation.json`
- checkpoints：`/root/autodl-tmp/26summerBDMI_transformer/runs/tinystories_tinylm_moe_v1/`
- runner SHA-256：`423428daef532c510580ccf91ebc17a36724b9749fe9ae623ec41a54570a623c`

### Dense 基线与 C v2

- C v2 报告：[`C_UNIFIED_RESULT_REPORT_V2.md`](C_UNIFIED_RESULT_REPORT_V2.md)
- A+B 三 seed Dense 报告：[`PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md`](PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md)
- A+B 逐 run CSV：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/per_run.csv`
- A+B 聚合 JSON：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/summary.json`

## 14. 复现命令

在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  experiments/tinystories_tinylm_moe_v1/run_moe_six.py remaining \
  --device cuda:0 \
  --seed 17 \
  --train-tokens 10000000 \
  --validation-probe-tokens 262144 \
  --full-validation \
  --save-checkpoint

python -u experiments/tinystories_tinylm_moe_v1/run_moe_six.py \
  aggregate --seed 17 --train-tokens 10000000
```
