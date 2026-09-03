# 高效 Transformer 原论文配置与本项目推荐参数

> 版本：`paper_config_review_v1`  
> 目的：整理 Longformer、Performer、Reformer、Linformer、Memformer、Keyformer 的原论文或官方实现配置，并给出适合 RTX 5070 以上 GPU、约一周实验周期的推荐方案。  
> 重要说明：原论文的任务、模型和训练目标不同，不存在一套能同时称为“所有论文原始参数”的统一配置。

## 1. 三种实验标签

| 标签 | 含义 |
|---|---|
| `paper_core` | 原论文明确提出并验证的机制和关键参数 |
| `official_config` | 作者代码、官方 checkpoint 或官方配置中可直接核对的设置 |
| `matched_tinylm` | 为本项目公平比较设计的统一配置，不冒充原论文复现 |

最终报告应把以下结果分表：

1. `paper_reported`：原论文公开数字和实验设置；
2. `paper_aligned`：尽量按原论文或官方配置做的复现检查；
3. `matched_tinylm`：所有方法使用相同主干、数据和训练预算；
4. `innovation_validation`：本项目局部、全局近似、MemoryState 和动态控制的消融。

## 2. Longformer

论文：[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

### 原论文配置

Longformer 论文包含两条实验线。

**Character-level autoregressive LM：**

| 参数 | 原论文设置 |
|---|---|
| 数据 | text8、enwik8；各约 100M 字符，90M/5M/5M train/dev/test |
| small model | 12 layers、hidden 512、8 heads |
| large model | 30 layers、hidden 512、8 heads |
| 位置编码 | Transformer-XL 风格 relative + sinusoidal |
| LayerNorm/激活 | pre-LN、GeLU |
| 优化器 | AdamW，weight decay 0.01，gradient clip 0.25 |
| dropout | small 0.2；large 0.4 |
| 训练 | 5 phases，逐步增加长度/窗口并降低学习率 |
| 长度 | 2,048 起步，最后阶段 23,040 |
| batch | 32、32、16、16、16 |
| attention pattern | 低层小窗口，高层增大窗口；2 个 heads 使用 dilation |

**选择理由：**低层学习局部连续性，高层扩大感受野；dilation 增加可见距离而不按比例增加连接数；分阶段训练比直接训练超长序列更稳定。

RoBERTa/长文档线通常使用 RoBERTa-base（12 层、768 hidden、12 heads）或 large（约 24 层、1024 hidden、16 heads），bidirectional MLM，常见长度 4,096；LED 可扩展到 16,384。该实验线不能直接和 causal LM PPL 比较。

### 推荐配置

```yaml
layers: 6
hidden_size: 384
heads: 8
ffn_size: 1536
position: RoPE
dropout: 0.1
causal: true
window: [64, 128, 256, 512]
global_tokens: [0, 1, 4]
main_window: 128
main_global_tokens: 1
```

## 3. Performer / FAVOR+

论文：[Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)

### 原论文配置

| 参数 | 原论文/官方附录设置 |
|---|---|
| 长文本数据 | PG-19；SentencePiece unigram，32,768 vocabulary |
| 小型 Transformer | 使用过 1–3 layers、`d=256` 的小型模型 |
| optimizer | Adam，β1=0.9、β2=0.98、ε=1e-9 |
| learning rate | 1e-3 fixed |
| weight decay/dropout | 0.1/0.1 |
| gradient clipping | 0.5 |
| softmax FAVOR+ features | 256 |
| renormalization/stabilizer | true / 1e-6 |
| orthogonal features | true |
| generalized attention | 256 features、ReLU kernel、epsilon=1e-3 |
| causal计算 | prefix-sum，不构造完整下三角矩阵 |

**选择理由：**256 features 是质量和计算的折中；正交特征降低随机估计方差；positive features 保证近似权重非负；prefix-sum 保持 causal 计算近似线性。

### 推荐配置

```yaml
features: [64, 128, 256, 512]
main_features: 256
feature_type: positive_orthogonal_random_features
renormalize_attention: true
stabilizer: 1.0e-6
causal: true
feature_seeds: [101, 202, 303, 404, 505]
```

## 4. Reformer

论文：[Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)

### 原论文和官方实现核心配置

| 参数 | 常见论文/官方实现值 |
|---|---|
| 长序列 LM 规模 | 约 12 layers、dim 1,024、8 heads；具体任务会变化 |
| head dimension | 常见 64 |
| attention | LSH attention，多轮 hashing |
| bucket size | 64 是官方推荐起点 |
| n_hashes | 4 是质量/速度折中；8 通常更准但更慢 |
| causal | autoregressive LM 使用 true |
| position | 长序列推荐 axial positional embedding |
| residual | reversible residual layers |
| FFN/attention | chunked feedforward 和 attention chunking |
| persistent memory KV | 官方 PyTorch 实现常见 128 |
| 长度约束 | 通常要求长度为 `2 × bucket_size` 的倍数，或显式 autopad |

**选择理由：**bucket size 决定候选集合；n_hashes 增加相遇概率但增加 hashing 成本；reversible layers 节省训练激活显存；axial position 避免长位置的完整绝对 embedding。

### 推荐配置

```yaml
layers: 6
hidden_size: 384
heads: 8
dim_head: 48
causal: true
bucket_size: [32, 64, 128]
n_hashes: [2, 4, 8]
axial_position: true
reversible: true
ff_chunks: 8
main_bucket_size: 64
main_n_hashes: 4
```

## 5. Linformer

论文：[Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)

### 原论文核心配置

Linformer 沿序列维度把 K/V 投影到低秩维度 `k`。论文在 BERT/理解、语言和视觉任务中使用不同主干，不能把一个任务的参数当成全论文默认值。常见对照是 BERT-base 风格主干：12 layers、hidden 768、12 heads、长度 512；`k=256` 是常见默认级别，projection 通常为 learned。

**选择理由：**长度 512 时 `k=256` 将序列方向压缩约一半；长度到 2K–8K 时固定 256 可能过度压缩，因此必须测试固定 k 和随长度增长的 k。

### 推荐配置

```yaml
rank_k: [64, 128, 256, 384, 512]
main_rank_k: 256
projection: learned
share_projection_across_heads: [true, false]
causal: true
```

另加两条长度策略：`fixed-k: k=256` 和 `ratio-k: k=min(512,max(64,length/2))`，两者分表报告。

## 6. Memformer / MemBART

论文：[Memformer: A Memory-Augmented Transformer for Sequence Modeling](https://arxiv.org/abs/2010.06891)  
官方实现：[qywu/memformers](https://github.com/qywu/memformers)

### 官方公开配置

| 参数 | MemBART-base | MemBART-large |
|---|---:|---:|
| backbone | BART-base | BART-large |
| vocab size | 50,265 | 50,265 |
| d_model | 768 | 1,024 |
| encoder/decoder layers | 6/6 | 12/12 |
| attention heads | 12 | 16 |
| FFN dim | 3,072 | 4,096 |
| max position | 1,024 | 1,024 |
| memory length | 64 | 128 |
| activation | GELU | GELU |
| official dropout | 0.0 | 0.0 |
| init std | 0.01 | 0.001 |

**选择理由：**memory length 是固定状态容量；BART 主干便于使用预训练 checkpoint；memory 必须通过 recurrent update 跨片段传递，不能只是静态可学习 token。

### 推荐配置

```yaml
layers: 6
hidden_size: 384
heads: 8
ffn_size: 1536
segment_length: [128, 256, 512]
memory_slots: [16, 32, 64, 128]
main_segment_length: 256
main_memory_slots: 64
num_segments: [2, 4, 8, 16]
reset_between_samples: true
detach_memory: [true, false]
```

## 7. Keyformer

论文：[Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference](https://arxiv.org/abs/2403.09054)  
作者代码：[d-matrix-ai/keyformer-llm](https://github.com/d-matrix-ai/keyformer-llm)

Keyformer 是生成阶段的 KV cache 选择方法，不是需要从头预训练的独立 backbone。论文对齐重点是 decoder-only checkpoint、tokenizer、prompt、prefill/decode 长度、cache budget 和 FullKV baseline。计时必须包含 score、selection、gather、attention 和 cache update；100% cache 必须与 FullKV token/质量等价。

### 推荐配置

```yaml
backbone: GPT-2 Medium；资源允许时增加 1B–3B decoder-only model
batch_size: 1
prompt_length: [256, 512, 1024, 2048]
generation_length: [32, 128, 256]
cache_ratio: [0.25, 0.50, 0.75, 1.00]
recent_ratio_of_budget: 0.50
policies: [full_kv, keyformer, recent_window, random_plus_recent]
seeds: [17, 29, 43]
```

## 8. xFormers / PyTorch SDPA

xFormers/SDPA 是精确 attention kernel/backend，不是独立模型架构，没有单独的论文模型参数。应与 TinyLM 或 GPT-2 使用相同主干，测试：

```yaml
backend: math, memory_efficient, flash, xformers_if_available
dtype: [bf16, fp16, optional_fp32]
head_dim: [32, 64, 128]
length: [128, 512, 1024, 2048, 4096, 8192, 16384]
```

必须记录 requested backend 和 actual backend。

## 9. RTX 5070 以上的推荐统一配置

### 9.1 TinyLM 主模型

```yaml
tokenizer: GPT-2 BPE
layers: 6
hidden_size: 384
heads: 8
head_dim: 48
ffn_size: 1536
activation: GELU
normalization: Pre-LN
position: RoPE
max_position: 16384
dropout: 0.1
dtype: BF16
```

相比原来的 4 层、256 hidden，这一档更适合做有学习能力的机制实验，同时仍远小于原论文的大模型。

### 9.2 训练和数据

```yaml
optimizer: AdamW
learning_rate: 3e-4
betas: [0.9, 0.95]
eps: 1e-8
weight_decay: 0.1
warmup_ratio: 0.03
lr_schedule: cosine
grad_clip: 1.0
effective_batch_tokens: 32768
train_context: [512, 1024]
eval_context: [128, 512, 1024, 2048, 4096]
seeds: [17, 29, 43]
```

数据建议：

```text
Pilot: WikiText-2 raw-v1
主语言建模: WikiText-103
长文扩展: PG-19（时间允许时）
机制任务: Local Copy + In-Segment Retrieval + Passkey/Associative Recall
```

推荐固定 token budget：smoke 0.5–2M，pilot 5–20M，主实验 50–200M；所有方法使用相同实际训练 token 数。

### 9.3 各方法主配置

| 方法 | 主配置 | 消融范围 |
|---|---|---|
| SDPA/full attention | exact causal | math/memory-efficient/Flash |
| Longformer | window=128，global=1 | window=64/256/512；global=0/4 |
| Performer | features=256 | 64/128/512；5 feature seeds |
| Reformer | bucket=64，n_hashes=4 | bucket=32/128；hashes=2/8 |
| Linformer | k=256 | k=64/128/384/512；共享/不共享 |
| Memformer | segment=256，slots=64 | slots=16/32/128；segment=128/512 |
| Keyformer | GPT-2 Medium，cache=50% | 25/50/75/100%；recent ratio 0/50/100% |

## 10. 论文参数与推荐参数如何并存

### 论文对齐组

- 只在能核对任务、数据、模型和关键超参数时使用 `paper_aligned`；
- 原论文没有披露的设置标为 unknown，不自行补写；
- 若模型参数差异超过约 1%，或训练 token 差异超过约 5%，不称“严格复现”；
- 原论文数字、当地复现数字和统一 TinyLM 数字分表。

### 统一比较组

- 所有方法使用同一 TinyLM、tokenizer、数据 split、训练 token、seeds 和评估指标；
- 保留各方法的核心架构旋钮；
- 质量、速度和显存均由各负责人独立记录；
- 结论限定在本协议、任务、长度和硬件范围内。

## 11. RTX 5070 以上的一周执行方案

| 天数 | 任务 |
|---|---|
| 第 1 天 | 整理论文参数来源；冻结 TinyLM；完成环境和 50–100 step smoke |
| 第 2 天 | 完成 paper-aligned sanity check、mask、padding、shape 和 backward 检查 |
| 第 3–4 天 | WikiText-2 pilot；WikiText-103 固定 token budget；3 seeds |
| 第 5 天 | 128–16K forward/OOM benchmark；20 warmup + 50 timed samples |
| 第 6 天 | 各方法关键消融和机制任务 |
| 第 7 天 | 交叉复核、自动生成图表、整理三份独立报告 |

## 12. 参考来源

## 12.1 这些参数是否可以直接互相比较

不能把架构专属参数按数字大小直接比较。例如：

```text
Longformer window=128
Performer features=256
Linformer k=256
Memformer slots=64
```

这里的 128、256、256、64 不是同一种单位：

| 参数 | 实际表示 | 能否与其他方法的同名数字直接比较 |
|---|---|---|
| Longformer `window` | 每个 token 的局部候选位置数量/窗口范围 | 不能与 features、rank 或 slots 直接比较 |
| Longformer `global_tokens` | 具有全局可见性的特殊位置数量 | 只能在 Longformer 或相同 global-token 机制内比较 |
| Performer `features` | FAVOR+ 随机特征映射维度 | 不能等同于 attention 连接数；应看近似误差和实际耗时 |
| Reformer `bucket_size` | 每个 hash bucket 的候选规模 | 需要和 `n_hashes` 一起看，不能只比较 bucket 数字 |
| Reformer `n_hashes` | 独立 hash rounds 数量 | 可在 Reformer 内比较质量—速度折中 |
| Linformer `k` | K/V 序列投影的低秩维度 | 不能等同于 Performer features 或 Longformer window |
| Memformer `memory_slots` | 跨片段保留的压缩状态槽数量 | 不能等同于局部窗口；应结合保留距离和状态 bytes |
| Memformer `chunk/segment` | 每次处理和更新 memory 的片段长度 | 是时间边界和训练路径参数，不是 memory 容量本身 |
| Keyformer `cache_ratio` | 保留 KV token 的比例 | 是推理缓存预算，不是训练 attention 结构参数 |
| xFormers/SDPA `backend` | 精确 attention 的实现路径 | 是 kernel 条件，不是模型容量参数 |

### 三种可比性

**第一层：可以直接统一的参数。**

这些应在三人主实验中保持一致：

- 层数、hidden size、heads、head dimension、FFN；
- tokenizer、词表、位置编码、LayerNorm 和输出头；
- dropout、dtype、训练 token 数、optimizer 和 seeds；
- 输入长度、batch、任务语义和评估指标。

它们决定的是“模型主干和实验条件”，可以直接做配对比较。

**第二层：只能在同一方法内部直接比较的参数。**

例如 Longformer 的 `window=64/128/256/512`，或 Performer 的 `features=64/128/256/512`。这些参数可以在同一种方法内部画出质量—速度曲线，但不能据此断言 `window=128` 比 `features=256` 更小、更快或更强。

**第三层：跨方法需要经过归一化后比较的结果。**

跨方法应比较以下可观测或派生量：

- 相同输入下的实际 attention/候选交互数量；
- 实际 feature、bucket、rank 或 memory 状态的元素数量；
- attention 和状态的理论 bytes；
- 实测 median latency、tokens/s、peak allocated/reserved memory；
- 相同质量约束下的速度和显存；
- 相同延迟或显存预算下的 NLL/PPL/accuracy。

因此，推荐的跨方法表不是“专属参数大小规律”，而是：

| 比较表 | 固定条件 | 主要结论 |
|---|---|---|
| 质量表 | 相同主干、数据、训练 token、seeds | 哪种机制质量更好 |
| 速度表 | 同 GPU、dtype、长度、batch、backend | 哪种实现更快 |
| 显存表 | 同 GPU、长度和运行模式 | 哪种方法更省显存 |
| 等预算表 | 匹配实测 latency、attention elements 或 memory bytes | 在相同成本下哪种质量更好 |
| Pareto 表 | 不预先固定单一成本 | 哪些配置位于质量—效率前沿 |

### 推荐的公平比较方式

对你们的项目，建议采用以下顺序：

1. **统一主干比较**：所有方法使用 `6 layers, hidden=384, heads=8, FFN=1536`；
2. **方法内部消融**：分别改变 window、features、rank、hashes、slots 等；
3. **跨方法实测归一化**：报告实际 latency、显存、OOM 和质量，而不是比较专属参数数字；
4. **等预算对照**：选择若干配置，使它们的实测 latency 或状态 bytes 接近，再比较 PPL/accuracy；
5. **创新组合对照**：固定 causal 语义，比较 local-only、global-only、memory-only、三分支静态和三分支动态。

### 对本项目主配置的正确解释

```text
Longformer window=128：局部路径的邻域预算
Performer features=256：全局近似的特征预算
Memformer slots=64：历史压缩状态预算
```

它们可以共同构成“总计算预算”的不同组成部分，但不能说三者的数字在数学上相等。应使用实验测得的 latency、显存、交互数和质量来判断哪种预算分配更合理。

### 原论文参数的可比性

不同原论文的参数通常不能直接横向比较，因为它们可能同时改变了：

- 模型规模和 tokenizer；
- causal/bidirectional 语义；
- 数据集和训练目标；
- 序列长度和 batch；
- 硬件、kernel 和训练预算。

原论文参数适合回答“该论文方法在其原始条件下如何”，统一 TinyLM 参数适合回答“在同一条件下机制如何”。两类结果必须分表。

- Longformer: <https://arxiv.org/abs/2004.05150>
- Performer: <https://arxiv.org/abs/2009.14794>
- Reformer: <https://arxiv.org/abs/2001.04451>
- Linformer: <https://arxiv.org/abs/2006.04768>
- Memformer: <https://arxiv.org/abs/2010.06891>
- Memformer official implementation: <https://github.com/qywu/memformers>
- Reformer PyTorch implementation: <https://github.com/lucidrains/reformer-pytorch>
- Keyformer: <https://arxiv.org/abs/2403.09054>
- Keyformer author implementation: <https://github.com/d-matrix-ai/keyformer-llm>
