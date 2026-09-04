# B 角色实验报告：Longformer + Memformer

日期：2026-09-04  
硬件：NVIDIA GeForce RTX 4090 24GB（本次实际环境可见 1 张卡）  
角色主题：固定局部窗口与跨 segment recurrent memory 的长上下文信息保留。

## 结论先行

详细实验文档见 [PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md](PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md)。

在统一 TinyLM/TinyStories、10M training-token、seed=17、context=512 的 screening 条件下，两个冻结配置都完成了完整 validation。Memformer（segment=128、64 slots）取得略低的 validation NLL/PPL，并且训练吞吐更高、训练峰值显存更低；Longformer（left window=128、global=0）参数更少、结构更简单。效率矩阵显示：当前 Python/PyTorch 实现的 Longformer 和 Memformer 都没有超过高度优化的 SDPA Full-Attention 参考的端到端前向速度，因此结果同时揭示了算法复杂度与工程 kernel 实现之间的差距。

这些结论是本项目 TinyLM-long-v1 的短预算 validation screening，不是原论文复现，也不是充分收敛或论文级排行榜。

## 1. 固定实验协议

- 数据：固定 revision 的 `roneneldan/TinyStories`；GPT-2 tokenizer；每个故事追加 EOS，按固定 parquet 顺序串接并切成 512-token blocks；validation 仅用于 probe、checkpoint 选择和最终报告，无官方 test split。
- 公共主干：6 layers、hidden=384、8 heads、head dim=48、FFN=1536、Pre-LN、GELU、RoPE(max=32768)、dropout=0.1、tied embeddings、Linear bias 关闭、LayerNorm affine 保留。
- 公共参数：29,925,504（约 29.93M）。Longformer 保持该数量；Memformer 因额外 memory Q/K/V、输出投影和 gate 为 33,466,752（约 33.47M）。
- 训练：AdamW，lr=3e-4，betas=(0.9,0.95)，weight decay=0.1，3% warmup，cosine 到 3e-5，gradient clip=1.0，BF16 autocast、FP32 参数；micro-batch=4、gradient accumulation=4，有效 batch=8,192 tokens；seed=17。
- pilot：每个候选 1,048,576 tokens；主实验：10,000,000 tokens；固定 probe：262,144 predictions；最终 validation：4,765,917 predictions。

## 2. 候选筛选与冻结

选择顺序严格按预注册协议：correctness → finite loss/gradients → pilot validation NLL → 在 5% 质量带内比较吞吐 → 在 10% 吞吐带内比较显存。

| 方法/候选 | probe NLL | probe PPL | 训练 tok/s | peak allocated | 结果 |
|---|---:|---:|---:|---:|---|
| Longformer left=128 | 4.9123 | 135.95 | 14,130 | 6.80 GiB | **冻结** |
| Longformer left=256 | 4.8721 | 130.60 | 7,997 | 12.16 GiB | 不冻结 |
| Memformer slots=32 | 4.5803 | 97.55 | 15,508 | 2.64 GiB | 不冻结 |
| Memformer slots=64 | 4.5322 | 92.96 | 16,093 | 2.63 GiB | **冻结** |

Longformer 的 256-window PPL 低 3.94%，但仍在预定 5% 质量带内；window=128 吞吐约高 76.7%、峰值 allocated 低约 44.1%，故冻结 128。Memformer 的 64 slots 同时改善 probe NLL/PPL 和吞吐，显存基本不变，故冻结 64。完整选择依据见 [person_b_pilot_selection.json](experiments/tinystories_tinylm_v1/aggregate/person_b_pilot_selection.json)。

## 3. 主实验质量结果

| 方法 | train tokens | steps | 训练 tok/s | 最终 probe NLL / PPL | 完整 validation NLL / PPL | peak allocated / reserved |
|---|---:|---:|---:|---:|---:|---:|
| Longformer w=128 | 10,000,000 | 1,221 | 14,004 | 2.9097 / 18.351 | 2.8839 / 17.884 | 6.80 / 7.68 GiB |
| Memformer s=128,m=64 | 10,000,000 | 1,221 | 15,734 | 2.8722 / 17.677 | 2.8456 / 17.211 | 2.63 / 3.92 GiB |

相对 Longformer，Memformer 完整 validation NLL 低约 1.33%，PPL 低约 3.76%；训练吞吐高约 12.4%；peak allocated 低约 61.3%，peak reserved 低约 48.8%。同时其参数量高约 11.8%，所以“质量—显存更好”不能简化为“全面更优”。

每个训练步和 validation probe 的原始曲线保存在：

- [Longformer metrics.jsonl](experiments/tinystories_tinylm_v1/runs/person_b/main_b_longformer_w128_s17/metrics.jsonl)
- [Memformer metrics.jsonl](experiments/tinystories_tinylm_v1/runs/person_b/main_b_memformer_s128_m64_s17/metrics.jsonl)

## 4. 长度效率矩阵

这是完整 TinyLM 前向（包含 tied vocabulary projection）的 batch=1、BF16、warmup=10、timed=30 测量；Full-Attention 参考使用 Longformer checkpoint 的公共 Q/K/V/output 权重，仅作效率参照，不代表独立 Full-Attention 质量训练。完整原始记录见 [person_b_efficiency.json](experiments/tinystories_tinylm_v1/aggregate/person_b_efficiency.json)。

| 长度 | Full ref ms | Longformer ms | Memformer ms | Longformer tok/s | Memformer tok/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 3.873 | 14.520 | 26.362 | 35,261 | 19,422 |
| 1,024 | 4.078 | 23.295 | 50.841 | 43,958 | 20,141 |
| 2,048 | 4.319 | 39.875 | 100.013 | 51,360 | 20,477 |
| 4,096 | 5.473 | 74.896 | 197.842 | 54,689 | 20,703 |
| 8,192 | 10.136 | 143.740 | 393.868 | 56,992 | 20,799 |
| 16,384 | 23.523 | 280.707 | 783.836 | 58,367 | 20,902 |
| 32,768 | 67.832 | 557.841 | 1,509.627 | 58,741 | 21,706 |

端到端 benchmark 已按协议使用 BF16 autocast。在本实现中，Full SDPA 参考在所有测量长度都更快；32,768 时 Longformer 约为 Full 的 8.22×，Memformer 约为 Full 的 22.26×、约为 Longformer 的 2.71×。这不是对 Longformer/Memformer 理论复杂度的否定，而是说明 Python/chunk 调度、recurrent segment 更新和统一词表 logits 投影会显著影响实际端到端速度。旧版表格曾将未启用 autocast 的结果标成 BF16，已重测并以当前 `person_b_efficiency.json` 为准。

为隔离 attention 路径，另有 [person_b_attention_efficiency.json](experiments/tinystories_tinylm_v1/aggregate/person_b_attention_efficiency.json)：单层 attention-only 在 32,768 时 Longformer 88.38ms、Memformer 242.23ms、Full 8.17ms；attention-only 峰值增量约为 233.3、72.3、204.2 MiB。Memformer 的固定 state 路径确实节省状态规模，但当前实现仍有额外 segment fusion 开销。

对应图表：[训练/validation 曲线](experiments/tinystories_tinylm_v1/aggregate/person_b_training_validation_curves.png)、[端到端效率曲线](experiments/tinystories_tinylm_v1/aggregate/person_b_end_to_end_efficiency_curves.png)、[attention-only 效率曲线](experiments/tinystories_tinylm_v1/aggregate/person_b_attention_only_efficiency_curves.png)。

32768 是 RoPE 配置的最大位置；长度 32769 的三种方法均明确失败并返回 `ValueError: position exceeds configured RoPE maximum`，不是 OOM。边界记录见 [person_b_length_boundary.json](experiments/tinystories_tinylm_v1/aggregate/person_b_length_boundary.json)。

## 5. 跨 segment 影响机制诊断

为直接检验“局部传播”与“跨 segment state”而不引入额外训练任务，在相同 validation 前缀上替换最早 128 个 token，比较末端 logits 的变化。Longformer 6 层、left window=128 的单 token 传播上限约为 `6×128=768` 个位置；由于这里替换的是一个 128-token 前缀，长度 768/896 仍可观测到末端影响，而长度 1,024 及以上的末端影响降为 0。Memformer 在 512、768、896、1,024、2,048、4,096 上均保留非零末端影响（分别约 `0.2656、0.1250、0.0625、0.0625、0.0625、0.03125` 的最大 logit 绝对差），但影响随 recurrent 压缩和 segment 数增加而衰减。该结果支持两种信息通路的机制差异：Longformer 的远距可达性受层数×窗口限制，Memformer 可以跨任意已处理 segment 传递压缩影响；它不是事实检索准确率，也不证明无损记忆。

原始记录：[person_b_cross_segment_influence.json](experiments/tinystories_tinylm_v1/aggregate/person_b_cross_segment_influence.json)。

## 6. 正确性与可复现性

统一 runner 的 correctness 结果为 passed：

- Longformer causal future-invariance：0；full-window RoPE 等价误差 `2.38e-7`；
- Memformer causal future-invariance：0；跨段 history effect `0.0351`；reset 与 batch reorder 误差均为 0；
- 两模型反向传播梯度有限；公共非 attention 参数初始化最大差异为 0；
- 结果文件：[person_b_correctness.json](experiments/tinystories_tinylm_v1/aggregate/person_b_correctness.json)。

训练前发现并修复了一个流程问题：`evaluate()` 会切换 `eval()`，而原 runner 未在验证后恢复 `train()`，会错误关闭 dropout。修复后的主实验在初始 validation、每次 probe 和最终 validation 后都显式恢复 `model.train()`。修复前的 `pilot_seq_longformer_w128_s17` 不纳入结果。

## 7. 证据边界与解释

本轮支持：

- 在相同 TinyLM/TinyStories 预算下，Longformer 的固定局部连接与 Memformer 的跨 segment 状态路径都能稳定训练并达到 validation；
- Memformer 在本配置和短预算上取得稍好的 PPL、较高训练吞吐和较低训练峰值显存，但使用约 11.8% 更多参数；
- Longformer 的窗口参数对质量、速度和显存有明显折中；Memformer 的 slots 参数在 pilot 中体现容量—质量趋势；
- 长度扩展到 32K 在当前 RoPE 和实现下可运行，结构状态与 attention-only 内存趋势可被单独观察。

本轮不支持：

- 原论文 text8/enwik8 或 MemBART 的严格复现；
- 充分收敛、三 seed 统计显著性或跨数据集普遍结论；
- 仅凭 TinyStories packed validation 证明真正的跨文档事实记忆；
- 把当前 Python 实现的速度直接等同于 fused Longformer/Memformer kernel 的理论复杂度。

## 8. 可追溯文件

- 训练 runner：[run_person_b.py](experiments/tinystories_tinylm_v1/run_person_b.py)
- 效率 runner：[benchmark_person_b.py](experiments/tinystories_tinylm_v1/benchmark_person_b.py)
- attention-only runner：[benchmark_attention_person_b.py](experiments/tinystories_tinylm_v1/benchmark_attention_person_b.py)
- 机制影响诊断：[mechanism_person_b.py](experiments/tinystories_tinylm_v1/mechanism_person_b.py)
- 主实验聚合：[person_b_runs.json](experiments/tinystories_tinylm_v1/aggregate/person_b_runs.json)
- pilot 冻结依据：[person_b_pilot_selection.json](experiments/tinystories_tinylm_v1/aggregate/person_b_pilot_selection.json)
- 配置：[three_day_validation_screening.yaml](experiments/tinystories_tinylm_v1/three_day_validation_screening.yaml)
- 计划：[THREE_DAY_EXECUTION_PLAN.md](experiments/tinystories_tinylm_v1/THREE_DAY_EXECUTION_PLAN.md)
