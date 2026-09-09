# B 角色实验结果报告（统一三 seed 版本）

日期：2026-09-06  
实验角色：B（Longformer + Memformer）  
硬件：NVIDIA GeForce RTX 4090 24GB，实际运行设备为 `cuda:0`  
数据：TinyStories + TinyLM  
报告性质：统一验证集筛选、机制诊断与工程效率分析；不是原论文严格复现

> 本文件是新建的统一版本报告。原有的 [PERSON_B_TINYLM_REPORT.md](PERSON_B_TINYLM_REPORT.md) 和 [PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md](PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md) 未修改，保留为历史 B 角色结果记录。

## 摘要与结论先行

本报告沿用原 B 角色报告的分析思路，重点回答：固定局部窗口的 Longformer 与固定容量 recurrent memory 的 Memformer，在相同 TinyLM/TinyStories 条件下分别牺牲和获得了什么。

本次证据分为两层：

1. **B 角色主结论**：Longformer 和 Memformer 的机制、长序列效率、跨 segment 影响以及 100M-token 扩展。
2. **统一 A+B 横向结果**：在同一 runner 下加入 Linformer 和 Performer，四种训练型模型各使用 seed=17、29、43，作为当前 10M-token 主比较。

统一三 seed、10M-token、context=512 的结果如下：

| 方法 | Validation NLL（均值 ± std） | Validation PPL（均值 ± std） | 训练吞吐 tok/s（均值 ± std） | Peak allocated | 参数量 |
|---|---:|---:|---:|---:|---:|
| **Memformer** | **2.849148 ± 0.004651** | **17.273176 ± 0.080427** | 14,669.5 ± 300.5 | **2.633 GiB** | 33,466,752 |
| **Longformer** | 2.885517 ± 0.004599 | 17.912946 ± 0.082476 | 13,498.8 ± 268.5 | 6.799 GiB | 29,925,504 |
| **Linformer** | 2.933816 ± 0.006033 | 18.799463 ± 0.113420 | **27,857.8 ± 106.0** | 2.580 GiB | 29,925,504 |
| **Performer** | 3.450477 ± 0.008266 | 31.516144 ± 0.260635 | 27,079.0 ± 541.2 | 2.704 GiB | 29,925,504 |

在当前固定配置下：

- B 角色内部的质量排序为 **Memformer > Longformer**；Memformer PPL 低约 3.57%，训练吞吐高约 8.67%，peak allocated 低约 61.28%。
- Memformer 比 Longformer 多 3,541,248 个参数，即约 **11.83%**；所以不能把质量差异直接归因于 recurrent memory 机制。
- 四模型质量排序为 **Memformer、Longformer、Linformer、Performer**；训练吞吐排序为 **Linformer、Performer、Memformer、Longformer**。
- B 的 100M-token 扩展改变了质量排序：Longformer 完整 validation PPL 为 **6.0967**，优于 Memformer 的 **6.4622**；但 Memformer 仍保持更高训练吞吐和更低训练峰值显存。
- 当前 batch=1 的完整前向中，Longformer 比 Memformer 快，但二者都没有超过高度优化的 Full SDPA 参考。这反映的是当前 Python/PyTorch adapter 的工程代价，不是否定理论复杂度。
- 早期 token 扰动实验显示：Longformer 的远距影响受层数×窗口限制；Memformer 可以跨越该局部传播上限，但影响会随反复压缩而衰减。

最稳妥的总体结论是：

> Longformer 用较少参数保存局部 token 级细节，结构简单、当前推理路径较快，但远距传播有硬边界；Memformer 用额外参数和固定槽压缩历史，能建立跨 segment 路径并降低当前训练峰值显存，但有压缩信息瓶颈、较高推理调度成本，且在更长训练预算下质量不一定占优。

这支持把 Memformer 作为“固定容量记忆基线”，并继续研究动态重要性保留；不支持宣称固定 Memformer 已经实现语义精确找回，也不支持任一方法无条件全面优于另一方法。

## 1. 研究问题与证据层次

### 1.1 B 角色研究问题

本实验围绕四个问题展开：

1. **质量**：在相同公共 TinyLM、数据和训练预算下，Longformer 与 Memformer 的 validation NLL/PPL 如何变化？质量排序是否依赖训练预算？
2. **训练资源**：两种机制的训练流程吞吐、墙钟时间和峰值显存有何差异？
3. **长度扩展**：冻结 checkpoint 在 512–32,768 token 前向时，端到端 latency 和 attention-only latency/memory 如何增长？
4. **信息通路**：替换最早一段输入后，末端 logits 是否仍发生变化？结果是否符合局部传播上限和 recurrent state 的结构预期？

### 1.2 结果层次

| 结果层 | 内容 | 本报告中的作用 |
|---|---|---|
| B 历史 10M main | Longformer/Memformer，seed=17，原 B runner | 复核原 B 结论、pilot 和机制诊断 |
| B 历史 100M extension | 两模型各 seed=17，从初始化训练 100M | 分析训练预算导致的质量排序变化 |
| A+B unified rerun | Linformer/Performer/Longformer/Memformer，各 3 seed | 当前最严格的四模型 10M 横向比较 |

统一 runner 的 Longformer/Memformer seed=17 质量结果与原 B main 一致；但不同 runner 的计时和审计字段不同，因此本报告不把两套吞吐数字混成一条统计序列。100M extension 只用于训练预算分析，不与 10M 三 seed 均值合并。

Reformer 和 Keyformer 尚未按当前 unified runner 重训：Reformer不能进入本报告的四模型主表；Keyformer是推理期 KV-cache 策略，不应与训练型 backbone 直接排同一质量榜。

## 2. 统一实验协议

### 2.1 数据集和 token stream

| 项目 | 固定设置 |
|---|---|
| 数据集 | `roneneldan/TinyStories` |
| 数据 revision | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| train stories | 2,119,719 |
| validation stories | 21,990 |
| tokenizer | `openai-community/gpt2` |
| tokenizer revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| vocabulary | 50,257 |
| EOS token | 50,256 |
| packing | 每篇故事末尾追加 EOS，按固定 parquet 顺序串接，再切成 512-token blocks |
| 训练 cache | 10,000,385 tokens，供严格 10,000,000 个有效 predictions 使用 |
| 完整 validation | 4,765,917 个有效 next-token predictions |
| 官方 test split | 无 |

所有方法读取相同固定 token 顺序，三 seed 没有独立 shuffle。因此本报告中的 seed 标准差主要反映初始化差异，以及 Performer 的随机特征投影差异，不包含数据顺序方差。跨故事 packing 提高了 token 利用率，但不等于构造了真实的跨文档长期事实记忆任务。

### 2.2 公共 TinyLM

| 模块 | 设置 |
|---|---:|
| layers | 6 |
| hidden size | 384 |
| heads | 8 |
| head dimension | 48 |
| FFN size | 1,536 |
| normalization | Pre-LN |
| activation | GELU |
| position | RoPE，最大位置 32,768 |
| residual dropout | 0.1 |
| attention dropout | 0.0 |
| embedding/output | tied |
| Linear bias | 关闭 |
| LayerNorm affine | 保留 weight 和 bias |

公共主干参数量为 **29,925,504**。比较采用“公共主干匹配、方法专属参数保留”，而不是强行让总参数量相等：Memformer 的 memory Q/K/V、输出投影和 gate 保留为其方法成本。

### 2.3 优化、精度和预算

| 项目 | 设置 |
|---|---|
| optimizer | AdamW |
| learning rate | `3e-4` |
| betas | `(0.9, 0.95)` |
| epsilon | `1e-8` |
| weight decay | `0.1` |
| schedule | 3% warmup + cosine decay，最低 `3e-5` |
| gradient clip | global norm=1.0 |
| compute | BF16 autocast；参数 FP32；TF32 关闭 |
| micro batch | 4 sequences |
| accumulation | 4 |
| effective batch | 8,192 prediction tokens |
| unified budget | 10,000,000 predictions |
| unified seeds | 17、29、43 |

### 2.4 指标定义

- **Validation NLL**：所有有效 next-token prediction 的 token-weighted cross-entropy，越低越好。
- **PPL**：`exp(NLL)`，从完整 validation NLL 派生，越低越好。
- **训练流程吞吐**：完成训练 token 数除以 run elapsed time；包含定期 validation probe 和 checkpoint 操作，不是纯 forward/backward kernel 吞吐。
- **Peak allocated/reserved**：从初始 validation probe 前重置 CUDA peak 统计，覆盖整个 run 的峰值。
- **效率 latency**：batch=1、BF16、10 次 warmup 后 30 次 CUDA Event 计时的中位数。
- **标准差**：三个 seed 的 run-level sample standard deviation，不是置信区间。

## 3. 方法机制与参数公平性

### 3.1 Longformer

本实验使用 decoder-only causal sliding-window adapter，而不是完整复刻原始 Longformer 的 encoder/global-token 任务结构。冻结配置为：

| 参数 | 值 |
|---|---:|
| causal left window | 128 |
| total window size | 257 |
| causal max visible tokens | 当前 token + 左侧最多 128 个 |
| global tokens | 0 |
| query chunk size | 128 |

固定窗口下，attention 候选规模随序列长度近似为 `O(NW)`。当前实现使用 padded `unfold`、query chunk 和通用 PyTorch 算子；理论 `O(NW)` 不等同于已融合 CUDA kernel 的实际速度。

### 3.2 Memformer

Memformer 将每个 128-token segment 的信息融合到固定数量的 memory slots，并把更新后的 state 传给后续 segment。冻结配置为：

| 参数 | 值 |
|---|---:|
| segment length | 128 |
| memory slots | 64 |
| segment 间 detach | 否 |
| 独立 packed example/forward reset | 是 |
| dynamic active slots | 否，64 槽始终参与计算 |

64 slots 不是可学习 slot embedding 参数，而是运行时 state。batch=1、BF16、6 层时，原始 state 大小为：

```text
6 × 64 × 384 × 2 bytes = 294,912 bytes = 288 KiB
```

实际 run peak 还包括参数、autograd 激活、临时张量和 allocator 行为，所以不能用 288 KiB 单独解释 2.63 GiB 的峰值显存。

### 3.3 参数匹配原则

统一 runner 对公共 Q/K/V/output 路径进行参数名规范化，保证 embedding、MLP、LayerNorm 和公共 attention 参数可确定性匹配初始化；方法专属参数不被抹除。correctness 检查确认公共非 attention 参数最大初始化差异为 0。

| 方法 | 总参数 | 相对公共主干 |
|---|---:|---:|
| Longformer | 29,925,504 | 基准 |
| Memformer | 33,466,752 | +3,541,248，+11.83% |

因此质量比较是“同公共主干、不同方法容量”的结果，而不是严格等参数因果实验。

## 4. Correctness 与 pilot 冻结

### 4.1 Correctness 闸门

统一四模型 correctness suite 状态为 `passed`：

| 检查 | 实测 | 结果 |
|---|---:|---|
| 输出 shape 与 finite logits | 4/4 | 通过 |
| 有限梯度 | 4/4 | 通过 |
| future-invariance | 最大差异 0 | 通过 |
| Linformer 满窗等价 | 最大差异 0 | 通过 |
| Longformer 满窗与 dense reference | `1.778e-4` | 低于 `5e-4` 容差 |
| Memformer history effect | 非零 | 通过 |

B 专属检查还得到：Longformer full-window RoPE 等价误差 `2.384e-7`，Memformer history effect `0.035115`，reset 差异 0，batch reorder 差异 0。通过 correctness 只能证明因果性、数值稳定性和预期状态路径，不能证明语义记忆正确。

### 4.2 Pilot 选择

选择顺序预先固定为：correctness → finite loss/gradient → validation NLL → 在 5% 质量带内比较吞吐 → 在 10% 吞吐带内比较峰值显存。

| 方法 | 候选 | Probe NLL | Probe PPL | 吞吐 tok/s | Peak allocated | 是否冻结 |
|---|---|---:|---:|---:|---:|---|
| Longformer | left=128 | 4.912294 | 135.9510 | 14,130 | 6.80 GiB | **是** |
| Longformer | left=256 | 4.872144 | 130.6006 | 7,997 | 12.16 GiB | 否 |
| Memformer | slots=32 | 4.580344 | 97.5480 | 15,508 | 2.64 GiB | 否 |
| Memformer | slots=64 | 4.532153 | 92.9585 | 16,093 | 2.63 GiB | **是** |

window=256 的质量改善在预设 5% 带内，但速度和显存代价明显，因此冻结 window=128。64 slots 同时改善 pilot NLL 和吞吐，峰值显存基本不变，因此冻结 slots=64。

## 5. 统一 10M-token 三 seed 质量结果

### 5.1 四模型主表

| 方法 | 配置 | 参数量 | Val NLL（均值 ± std） | Val PPL（均值 ± std） | 吞吐 tok/s（均值 ± std） | Peak allocated |
|---|---|---:|---:|---:|---:|---:|
| **Memformer** | segment=128，slots=64 | 33,466,752 | **2.849148 ± 0.004651** | **17.273176 ± 0.080427** | 14,669.5 ± 300.5 | **2.633 GiB** |
| **Longformer** | left window=128 | 29,925,504 | 2.885517 ± 0.004599 | 17.912946 ± 0.082476 | 13,498.8 ± 268.5 | 6.799 GiB |
| Linformer | pool/chunk=128 | 29,925,504 | 2.933816 ± 0.006033 | 18.799463 ± 0.113420 | **27,857.8 ± 106.0** | 2.580 GiB |
| Performer | features=384，redraw disabled | 29,925,504 | 3.450477 ± 0.008266 | 31.516144 ± 0.260635 | 27,079.0 ± 541.2 | 2.704 GiB |

质量排序在三个 seed 中完全一致。Memformer 相对 Longformer：

- NLL 低 `0.036369`，相对低约 **1.26%**；
- PPL 低 `0.639770`，相对低约 **3.57%**；
- 训练吞吐高约 **8.67%**；
- peak allocated 低约 **61.28%**；
- 参数多约 **11.83%**。

因此，10M 结果支持“Memformer 在当前短预算筛选中较好”，但不支持“严格等参数下 memory 机制一定更好”。

### 5.2 逐 seed validation

| 方法 | seed=17 NLL / PPL | seed=29 NLL / PPL | seed=43 NLL / PPL |
|---|---:|---:|---:|
| Linformer | 2.933785 / 18.7987 | 2.939865 / 18.9133 | 2.927799 / 18.6865 |
| Performer | 3.459063 / 31.7872 | 3.442574 / 31.2673 | 3.449795 / 31.4939 |
| Longformer | 2.883914 / 17.8841 | 2.890703 / 18.0060 | 2.881933 / 17.8487 |
| Memformer | 2.845559 / 17.2112 | 2.854402 / 17.3641 | 2.847481 / 17.2443 |

方法内 PPL 波动较小：Longformer 的范围为 17.8487–18.0060，Memformer 为 17.2112–17.3641；方法间差异大于对应 seed 波动。由于只有三个 seed，不能据此给出论文级显著性结论。

### 5.3 Validation probe 轨迹

三 seed probe NLL 均值如下：

| 训练 predictions | Linformer | Performer | Longformer | Memformer |
|---:|---:|---:|---:|---:|
| 1,048,576 | 4.4318 ± 0.0296 | 4.5941 ± 0.0648 | 4.3911 ± 0.0218 | **4.1428 ± 0.0122** |
| 2,097,152 | 3.7835 ± 0.0126 | 4.1162 ± 0.0058 | 3.7384 ± 0.0195 | **3.6329 ± 0.0075** |
| 4,194,304 | 3.3983 ± 0.0095 | 3.8164 ± 0.0005 | 3.3550 ± 0.0037 | **3.2878 ± 0.0104** |
| 6,291,456 | 3.1219 ± 0.0061 | 3.6362 ± 0.0070 | 3.0863 ± 0.0057 | **3.0366 ± 0.0054** |
| 8,388,608 | 3.0079 ± 0.0058 | 3.5170 ± 0.0071 | 2.9557 ± 0.0041 | **2.9188 ± 0.0040** |
| 10,000,000 | 2.9589 ± 0.0058 | 3.4807 ± 0.0061 | 2.9104 ± 0.0037 | **2.8757 ± 0.0044** |

所有 run 的最佳 probe 都出现在最终 step 1,221，曲线末端仍在下降。这说明 10M 是稳定可比的 screening 预算，不是充分收敛或全局最佳训练状态。

## 6. B 的 100M-token 扩展

Longformer 和 Memformer 的冻结配置分别从随机初始化训练到 100M predictions，seed=17；不是从 10M checkpoint 接续。由于 100M 使用 12,208 steps，其 cosine schedule 与 10M 不同，不能把两个预算视为同一训练轨迹。

| 指标 | Longformer w=128 | Memformer s=128,m=64 | Memformer 相对 Longformer |
|---|---:|---:|---:|
| 完整 Val NLL | **1.807748** | 1.865966 | 高 3.22% |
| 完整 Val PPL | **6.096701** | 6.462175 | 高 5.99% |
| 训练吞吐 | 14,743.8 tok/s | **16,813.0 tok/s** | 高 14.03% |
| 训练时间 | 113.04 min | **99.13 min** | 少 12.31% |
| Peak allocated | 6.80 GiB | **2.63 GiB** | 少 61.28% |
| Peak reserved | 7.69 GiB | **3.93 GiB** | 少约 48.9% |

从约 20M probe predictions 开始，Longformer 在每个观测点都优于 Memformer，100M 时差距仍存在。这个反转现象说明：10M screening 的 Memformer 质量优势不能外推为充分训练后的普遍优势；训练预算本身是架构比较的重要变量。100M 仍只有一个 seed，且 validation 曲线末端仍在下降，因此也不能称为充分收敛。

## 7. 训练资源、效率与长度边界

### 7.1 统一训练墙钟与显存

| 方法 | 10M elapsed（约） | 训练吞吐 | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| Linformer | 5.98 min | 27,857.8 tok/s | 2.580 GiB | 3.781 GiB |
| Performer | 6.16 min | 27,079.0 tok/s | 2.704 GiB | 3.895 GiB |
| Longformer | 12.35 min | 13,498.8 tok/s | 6.799 GiB | 7.672 GiB |
| Memformer | 11.36 min | 14,669.5 tok/s | 2.633 GiB | 3.926 GiB |

Memformer 的 run-level 显存优势反映当前实现的 activation/intermediate 形态，而不只是 memory state 大小；Longformer 的局部窗口展开和 chunked score 路径在训练反向中保存了较大的中间张量。训练吞吐包含 probe/checkpoint，因此不能直接当作单个 attention kernel 的速度。

### 7.2 完整 TinyLM forward

这是冻结 checkpoint 的 batch=1、BF16、eval-mode、完整 6 层 TinyLM forward，包含 tied vocabulary projection。每格为“中位 latency ms / tokens/s”：

| 长度 | Full SDPA 参考 | Longformer | Memformer |
|---:|---:|---:|---:|
| 512 | **3.873 / 132,205** | 14.520 / 35,261 | 26.362 / 19,422 |
| 1,024 | **4.078 / 251,099** | 23.295 / 43,958 | 50.841 / 20,141 |
| 2,048 | **4.319 / 474,215** | 39.875 / 51,360 | 100.013 / 20,477 |
| 4,096 | **5.473 / 748,363** | 74.896 / 54,689 | 197.842 / 20,703 |
| 8,192 | **10.136 / 808,244** | 143.740 / 56,992 | 393.868 / 20,799 |
| 16,384 | **23.523 / 696,514** | 280.707 / 58,367 | 783.836 / 20,902 |
| 32,768 | **67.832 / 483,077** | 557.841 / 58,741 | 1,509.627 / 21,706 |

当前实现中 Full SDPA 在所有已测长度都更快。32K 时 Longformer 延迟约为 Full 的 8.22 倍，Memformer 约为 Full 的 22.26 倍、Longformer 的 2.71 倍。不能据此否定稀疏/记忆方法的理论复杂度；当前 Python chunk/segment 调度、通用 kernel 和大词表 logits 投影会遮蔽算法收益。

### 7.3 Attention-only forward

为隔离词表投影，只运行一个训练层的 attention，输入为确定性 Gaussian hidden states：

| 长度 | Full SDPA ms / MiB | Longformer ms / MiB | Memformer ms / MiB |
|---:|---:|---:|---:|
| 512 | **0.503 / 3.9** | 2.174 / 38.2 | 3.950 / 4.4 |
| 1,024 | **0.502 / 7.9** | 3.614 / 32.9 | 7.876 / 5.2 |
| 2,048 | **0.486 / 12.8** | 6.290 / 40.4 | 15.484 / 7.1 |
| 4,096 | **0.666 / 25.5** | 11.906 / 51.5 | 30.748 / 10.3 |
| 8,192 | **1.117 / 51.6** | 22.968 / 74.8 | 60.712 / 18.1 |
| 16,384 | **2.547 / 102.1** | 44.237 / 120.5 | 121.048 / 36.2 |
| 32,768 | **8.174 / 204.2** | 88.378 / 233.3 | 242.225 / **72.3** |

32K 时 Memformer 的 attention incremental memory 比 Longformer 低约 69%，但 latency 约为 Longformer 的 2.74 倍。当前结果体现“更低 attention 工作内存、较高调度延迟”的工程权衡；要判断理论复杂度能否转化为速度，需要 fused segment-memory 和 fused sliding-window kernel。

### 7.4 长度边界

三种路径在长度 32,768 成功运行；长度 32,769 均返回：

```text
ValueError: position exceeds configured RoPE maximum
```

这是 `rope_max_position=32768` 的显式边界，不是 OOM，也不是长上下文质量验证。模型只在 context=512 上训练和 validation。

## 8. 跨 segment 信息通路诊断

### 8.1 诊断方法

在固定 validation 前缀中替换最早 128 个 token，观察末端 logits 的变化。该测试回答“早期输入是否仍可影响末端”，不回答“模型能否正确恢复被替换的事实”。

### 8.2 结果

每格为“末 128 个位置最大 logit 差 / 最后一个位置最大 logit 差”：

| 序列长度 | Longformer | Memformer |
|---:|---:|---:|
| 512 | 0.12500 / 0.06250 | **1.09375 / 0.265625** |
| 768 | 0.06250 / 0.03125 | **0.37500 / 0.125000** |
| 896 | 0.06250 / 0.06250 | **0.18750 / 0.062500** |
| 1,024 | 0 / 0 | **0.12500 / 0.062500** |
| 2,048 | 0 / 0 | **0.06250 / 0.062500** |
| 4,096 | 0 / 0 | **0.06250 / 0.031250** |

Longformer 的 6 层、left window=128 给出约 `6×128=768` 的单 token 理论传播上限；长度 1,024 时早期扰动已无法影响末端。Memformer 在 4,096 仍保留非零影响，说明 recurrent state 提供了跨 segment 路径，但影响随反复压缩而衰减。

因此本实验支持：

- Longformer 保留近期 token 级细节，但远距可达范围受层数和窗口硬限制；
- Memformer 可跨越局部窗口上限，但依赖固定槽摘要，不能保证远距细节无损。

本实验不支持：

- 非零 logit delta 等同于有用语义记忆；
- Memformer 能准确找回任意早期事实；
- 当前固定槽机制已经验证了动态重要性记忆创新。

## 9. 方法优缺点与目标架构启示

| 方法 | 本轮证据中的优点 | 本轮证据中的代价 | 后续重点 |
|---|---|---|---|
| Longformer | 参数少；局部路径简单可解释；100M 质量更好；当前前向明显快于 Memformer | 训练峰值显存高；无 global token 时有明确远距传播上限；当前未融合实现慢于 Full SDPA | fused sliding-window kernel、global token 消融、窗口质量曲线 |
| Memformer | 10M 三 seed 质量略好；训练吞吐更高；训练峰值显存更低；存在跨 segment 路径 | 参数多 11.83%；100M 质量落后；固定槽压缩衰减；当前前向慢；独立 forward 不持久化 state | dynamic active slots、重要性选择、事实检索、fused memory kernel |

对拟议“模型自主决定保留重要内容”的目标架构，本轮结果提供三点支持：

1. 固定局部窗口在足够远距离后会失去信息通路，因此仅靠局部 attention 不足以覆盖目标场景。
2. 固定 memory state 能跨越局部传播上限，并改善当前训练显存/吞吐，但反复压缩会衰减细节。
3. Memformer 已经包含固定槽、跨 segment state 和 learned gate；这些机制本身不能再作为新颖性声明。创新应放在内容重要性估计、动态 active slots、slot survival/merge/eviction、语义 payload 和真实存储/计算节省上。

下一步应加入 passkey、copy、associative recall、实体属性保持和话题切换后的细节恢复任务，报告 exact match、距离衰减、slot 使用率、state bytes 和端到端延迟。

## 10. 结论边界

1. 统一四模型结果是 10M-token、context=512、三个 seed 的 screening，不是充分收敛或超参数最优结果。
2. 三个 seed 使用固定数据顺序，std 不包含数据顺序方差。
3. Memformer 总参数多 11.83%，质量结果不是严格等参数因果比较。
4. Linformer/Performer 使用 A 下载源码适配层，Longformer/Memformer 使用 B 实现；不同实现仍可能贡献一部分差异。
5. A 原始 context=4096、effective batch=32768 的旧结果不能直接并入统一主榜。
6. B 100M extension 只有 seed=17，使用不同总步数的 cosine schedule，只用于训练预算趋势。
7. 长度 512–32,768 的结果是 frozen forward/机制诊断，不能称为长上下文语言建模质量。
8. Full SDPA 是效率参考，没有独立质量 checkpoint。
9. TinyStories 主要由短故事组成，跨故事 packing 不能替代真实长对话或跨文档事实记忆。
10. unified seed=17 的 config 文件生成早于 cache 审计字段补充，缺少内嵌 train/validation cache metadata；实际使用的 cache 与 seed=29/43 相同。下一轮应先冻结 runner 并写入 runner hash。

## 11. 可追溯文件

### B 历史实验

- 原详细协议：[PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md](PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md)
- 原 B 报告：[PERSON_B_TINYLM_REPORT.md](PERSON_B_TINYLM_REPORT.md)
- 原简略报告：[PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md](PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md)
- pilot：[person_b_pilot_selection.json](../experiments/tinystories_tinylm_v1/aggregate/person_b_pilot_selection.json)
- correctness：[person_b_correctness.json](../experiments/tinystories_tinylm_v1/aggregate/person_b_correctness.json)
- B 10M summary：[person_b_final_summary.json](../experiments/tinystories_tinylm_v1/aggregate/person_b_final_summary.json)
- 端到端效率：[person_b_efficiency.json](../experiments/tinystories_tinylm_v1/aggregate/person_b_efficiency.json)
- attention-only 效率：[person_b_attention_efficiency.json](../experiments/tinystories_tinylm_v1/aggregate/person_b_attention_efficiency.json)
- 跨 segment 诊断：[person_b_cross_segment_influence.json](../experiments/tinystories_tinylm_v1/aggregate/person_b_cross_segment_influence.json)
- 长度边界：[person_b_length_boundary.json](../experiments/tinystories_tinylm_v1/aggregate/person_b_length_boundary.json)
- 100M Longformer：[summary.json](/root/autodl-tmp/26summerBDMI_transformer/runs/person_b_extended/extended100m_longformer_w128_s17/summary.json)
- 100M Memformer：[summary.json](/root/autodl-tmp/26summerBDMI_transformer/runs/person_b_extended/extended100m_memformer_s128_m64_s17/summary.json)

### A+B unified rerun

- runner：[run_unified_ab.py](../experiments/tinystories_tinylm_v1/run_unified_ab.py)
- 三 seed 聚合：[summary.csv](/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/summary.csv)
- 逐 run 结果：[per_run.csv](/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/per_run.csv)
- 聚合 JSON：[summary.json](/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/summary.json)
- correctness：[correctness.json](/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/correctness.json)
- 统一重训说明：[UNIFIED_AB_RERUN_REPORT.md](UNIFIED_AB_RERUN_REPORT.md)

### 数据与源码审计

- A `models.py` SHA-256：`d8aef33738476c8422cd14ae119d5303d67896dc34abdfd05ec5f476b8161f4b`
- train cache SHA-256：`a25e6c436b1cfcde1bcee54724563931763b40e034abd99e63d2286dbdf20f17`
- validation cache SHA-256：`d5e1575253e3489e2af831dee209d5ad56a6bb9a4bf02854d66fc80dddc2a9e2`

