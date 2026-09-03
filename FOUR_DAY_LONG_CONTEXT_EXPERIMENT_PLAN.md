# 四天长文本高效 Transformer 架构比较执行计划

> 版本：`long_context_10m_v1`  
> 日期：2026-09-03  
> 适用硬件：RTX 4090 24GB 或更高；推荐 4 张同型号 GPU  
> 时间上限：96 小时  
> 目的：比较不同长文本注意力架构的质量、收敛、长距离能力、速度和显存，不复现原论文数值。

## 0. 给执行 AI 的任务定义

本计划是执行规范，不是背景材料。执行 AI 应按照阶段闸门推进，不得跳过正确性测试直接提交大规模训练。

研究问题固定为：

> 在相同长文本数据、causal 语言建模目标、TinyLM 主干、训练 token 数和评估协议下，Full Attention、Longformer、Performer、Memformer 以及可选混合结构各自有什么质量、收敛速度、长距离信息利用、训练效率和显存方面的优缺点？

四天内必须交付：

1. 一套冻结的 `protocol_version=long_context_10m_v1` 配置；
2. 每个核心架构 3 个训练 seed 的正式结果；
3. PG-19 自然长文本质量结果；
4. 固定目标、不同历史长度的受控长上下文结果；
5. Local Copy、In-Segment Retrieval、Cross-Segment Passkey 机制结果；
6. 1K 到 16K 的速度、吞吐、显存和 OOM 结果；
7. 固定训练预算表、等成本表和质量-效率 Pareto 图；
8. 所有原始 JSONL、日志、配置、环境信息和失败记录。

明确不做：

- 不复现任何原论文的完整模型、数据或论文表格；
- 不训练 Keyformer；Keyformer 是推理期 KV cache 方法，不进入本次训练架构主表；
- 不把 xFormers、Flash 或 SDPA 当成独立模型架构；它们是 Full Attention 的实现后端；
- 本计划内任何正式架构都不超过约 10M training tokens；更长训练必须另建协议，不能混入本计划主表；
- 不运行所有超参数的笛卡尔积；
- 不跨不同 GPU 型号比较绝对 latency；
- 不把 WikiText-103 结果当作 8K/16K 长距离能力证据。

## 1. 实验结构和优先级

### P0：必须完成的核心架构

| 方法 ID | 方法 | 研究作用 |
|---|---|---|
| `full_sdpa` | 精确 causal Full Attention，优先 Flash SDPA | 质量和系统基线 |
| `longformer` | causal 左侧滑动窗口注意力 | 局部精确稀疏架构 |
| `performer` | causal FAVOR+ | 当前上下文全局近似架构 |
| `memformer` | 固定槽 recurrent memory | 跨 segment 历史压缩架构 |

P0 正式训练数：

```text
4 methods x 3 seeds = 12 runs
```

### P1：条件满足后加入

| 方法 ID | 加入条件 | 作用 |
|---|---|---|
| `hybrid_static` | local/global/memory 三个分支均通过正确性和 pilot | 检查三类信息能否互补 |
| `hybrid_dynamic` | 动态控制器 A 已实现且训练稳定 | 验证动态计算预算创新 |

P1 每加入一个方法，需要增加 3 次正式训练。若只有 3 张 GPU，先保证 P0 完成，再运行 P1。

### P2：有空余 GPU 时追加

| 方法 | 加入条件 |
|---|---|
| Reformer | causal LSH 通过 future-invariance；4K 吞吐不低于 500 tokens/s |
| Linformer | 序列投影不会混入未来 token；通过 future-invariance |

P2 失败不能阻塞 P0/P1。P2 不完整时，在报告中标记 `not_run` 或 `ineligible_causal`，不得伪造为 0。

## 2. 阶段定义：筛选、消融和主实验

执行时必须区分以下三类任务：

| 类型 | 数据和预算 | seed | 用途 |
|---|---|---:|---|
| 参数筛选 | PG-19 train 固定子集，约 2M tokens | 17 | 为每种方法选一个正式配置 |
| 探索性消融 | 合成任务或 PG-19 小子集，约 2M tokens | 17 | 判断组件是否能工作 |
| 正式主实验 | PG-19，10,027,008 tokens | 17/29/43 | 形成 10M 短预算下的架构比较结论 |

Longformer 的 window sweep、Performer 的 feature sweep、Memformer 的 slots/segment sweep属于参数筛选，不叫消融。

消融专指对项目混合结构删除组件：

```text
local_only
global_only
memory_only
local_plus_global
local_plus_memory
global_plus_memory
full_static
```

第一天的消融是探索性证据，用于决定混合结构是否进入 P1。若最终报告要声称某组件带来确定提升，至少将 `full_static`、`no_local`、`no_global`、`no_memory` 在冻结配置下补齐正式预算或 3 seeds。时间不足时必须把第一天结果标为 `exploratory_ablation`。

## 3. 数据集和数据处理

### 3.1 主数据集：PG-19

PG-19 是本次自然长文本训练和评估主数据集。使用官方 split 或可追溯的镜像；运行前记录数据来源、版本或 revision、文件 hash。若首选来源不可用，执行 AI 必须报告并选择内容等价的 PG-19 镜像，不得静默替换成其他语料。

固定设置：

```yaml
dataset: PG-19
splits: official_train_validation_test
tokenizer: GPT-2 BPE
tokenizer_name: gpt2
vocab_size: 50257
add_eos_at_document_end: true
cross_document_packing: false
train_context: 4096
tokens_per_training_example: 4096
required_source_tokens_per_example: 4097
```

处理规则：

1. tokenizer、特殊 token、文本规范化在所有方法之间完全相同；
2. 一个训练样本只能来自一篇文档；
3. 不把两篇书籍拼接成一个 4K 样本；
4. 每个 seed 预生成共享 manifest，内容为 `document_id, start_offset, input_length, target_length, order_index`；
5. 同一个 seed 下，所有方法读取相同样本和相同顺序；
6. validation/test manifest 对所有 seed 和方法固定；
7. Memformer 在新文档、样本替换和 batch 重排时重置 memory；
8. 保存原始文件 hash、tokenized cache hash 和 manifest hash；
9. 训练、validation、test 不得共享文档；
10. test 只能在配置冻结后使用。

### 3.2 WikiText-103 的定位

WikiText-103 不再是长文本主实验。它只用于以下目的：

- 数据和训练 runner 的快速控制测试；
- 检查不同架构能否在普通自然语言上稳定降低 loss；
- PG-19 下载或预处理期间并行完成 smoke；
- 可选的跨语料泛化检查。

推荐设置：

```yaml
dataset: Salesforce/wikitext
subset: wikitext-103-raw-v1
context: 1024
budget: 5M tokens maximum
seed: 17
result_label: control_only
```

WikiText-103 结果不进入 8K/16K 长文本主结论，也不需要为所有方法跑 3 seeds。

### 3.3 受控长历史评估集

从 PG-19 validation/test 中选取足够长的完整文档，创建固定目标评估集：

```yaml
history_lengths: [512, 1024, 2048, 4096, 8192, 16384]
target_length: 512
samples_per_history_length: 256
same_target_across_history_lengths: true
minimum_document_tokens: 16896
```

每个样本固定最后 512 个 target token，只改变 target 之前提供的历史长度。所有架构必须预测完全相同的 target token。

记录：

```text
NLL(L)
PPL(L)
delta_NLL(L) = NLL(L) - NLL(512)
```

8K 和 16K 高于 4K 训练长度，必须标记为 `length_extrapolation`。Memformer 可以分 segment 流式处理，但总可见历史不得超过当前 L；Full Attention 和其他方法也使用相同 L。

### 3.4 合成长依赖数据

建立统一生成器，生成以下三类 causal token 任务：

| 任务 | 训练长度 | 测试长度 | 变量 | 指标 |
|---|---|---|---|---|
| Local Copy | 1K/2K/4K | 1K/2K/4K/8K | copy distance=32/128/512/2048 | token accuracy、exact match |
| In-Segment Retrieval | 1K/2K/4K | 1K/2K/4K/8K | distractors=16/32/64/128 | answer exact match |
| Cross-Segment Passkey | 2K/4K | 2K/4K/8K/16K | key-query distance=256/1K/2K/4K/8K | passkey exact match |

统一规则：

- train/test 使用不同生成 seed；
- 所有架构读取相同生成 manifest；
- 训练分布最长 4K，8K/16K 是长度泛化；
- 不把 PG-19 训练后的模型直接当作能零样本理解任务指令；
- 使用共同的 causal 格式对每个模型进行约 5M synthetic tokens 的短期训练或微调；
- 四天时间不足时，先完成每个方法 seed 17；支撑强结论时再补齐 3 seeds。

## 4. 统一 TinyLM 主干

除 attention 和该方法必需的状态外，所有方法固定：

```yaml
model_family: TinyLM-long-v1
vocab_size: 50257
layers: 6
hidden_size: 384
heads: 8
head_dim: 48
ffn_size: 1536
activation: GELU
normalization: Pre-LN
position_encoding: RoPE
rope_max_position: 32768
dropout: 0.1
attention_dropout: 0.0
tie_input_output_embeddings: true
use_bias: false
causal: true
dtype: bfloat16
```

公平性规则：

1. embedding、FFN、LayerNorm、RoPE 和 LM head 从同一基础 state dict 复制；
2. 相同形状的 Q/K/V/O 参数也从同一 state dict 复制；
3. 方法新增参数使用同一个初始化函数和 seed 派生规则；
4. 报告总参数量、可训练参数量、attention 参数量和 checkpoint bytes；
5. 参数量差异超过 1% 时，在主表显著标记；
6. 不为了让某个方法更好而单独改变层数、hidden size、FFN、dropout 或 tokenizer。

## 5. 统一训练参数

```yaml
optimizer: AdamW
learning_rate: 3.0e-4
betas: [0.9, 0.95]
epsilon: 1.0e-8
weight_decay: 0.1
warmup_ratio: 0.03
lr_schedule: cosine
minimum_learning_rate: 3.0e-5
gradient_clip_norm: 1.0
train_context: 4096
effective_batch_tokens: 32768
main_optimizer_steps: 306
actual_main_training_tokens: 10027008
pilot_optimizer_steps: 62
actual_pilot_tokens: 2031616
seeds: [17, 29, 43]
validation_interval_steps: 61
checkpoint_interval_steps: 61
keep_checkpoints: [best_validation, final, latest_recovery]
checkpoint_validation_tokens: 262144
final_validation_tokens: 1048576
final_test_tokens: 2097152
```

建议单卡设置：

```yaml
micro_batch_sequences: 1
gradient_accumulation_steps: 8
```

如果显存允许增加 micro-batch，应同步减少 accumulation，保持 32,768 effective batch tokens。不得因为某方法显存更省就让它看到更多训练 token 或使用更大 effective batch。

AMP 和数值规则：

- 主训练使用 BF16；
- loss accumulation 使用 FP32；
- 梯度裁剪发生在 unscale 之后；
- 记录 TF32、deterministic 和 compile 设置；
- NaN/Inf 立即保存诊断并将 run 标记为失败；
- 不允许静默降低长度、batch 或切换为 FP16 后继续写入同一 run。

## 6. 各架构候选参数和主配置

### 6.1 Full Attention

```yaml
method: full_sdpa
attention: exact_causal
requested_backend: flash_sdpa
fallback_backend: memory_efficient_sdpa
```

执行时必须记录 actual backend。若 Flash 回退为 math，不得仍写成 Flash。

### 6.2 Longformer

参数筛选：

```yaml
left_window: [128, 512, 1024, 2048]
global_tokens: 0
```

预设主配置：

```yaml
left_window: 512
global_tokens: 0
```

本次 causal 主表只使用左侧窗口。标准 bidirectional global token 可能读取未来位置，因此不得直接加入 causal 主实验。若实现 causal-safe global token，必须单独命名、通过 future-invariance，并作为扩展配置报告。

### 6.3 Performer

参数筛选：

```yaml
features: [128, 256, 384, 512]
feature_type: positive_orthogonal_random_features
causal_prefix_sum: true
renormalize_attention: true
stabilizer: 1.0e-6
```

预设主配置：

```yaml
features: 256
feature_redraw_interval: 1000
```

每个 run 的 feature seed 必须由训练 seed 确定性派生并记录。评估期间禁止 redraw。

### 6.4 Memformer

参数筛选候选保持 4 个，与其他方法相同：

```yaml
candidates:
  - {segment_length: 256, memory_slots: 64}
  - {segment_length: 512, memory_slots: 32}
  - {segment_length: 512, memory_slots: 64}
  - {segment_length: 512, memory_slots: 128}
```

预设主配置：

```yaml
segment_length: 512
memory_slots: 64
segments_per_training_example: 8
detach_memory_within_example: false
reset_between_documents: true
reset_on_batch_reorder: true
```

若完整 8-segment BPTT 在 24GB GPU 上 OOM，可将主协议升级为：

```yaml
detach_every_segments: 4
```

但所有 Memformer 正式 runs 必须统一使用新设置并升级 protocol version；旧结果不能混入。

### 6.5 混合结构

探索性固定配置：

```yaml
local_span: 512
global_features: 256
memory_slots: 64
segment_length: 512
local_enabled: true
global_enabled: true
memory_enabled: true
fusion: normalized_static_weights
fusion_weights: [1.0, 1.0, 1.0]
```

只有下列条件全部通过，`hybrid_static` 才进入 P1 正式主实验：

- 三个分支单独运行均能降低 loss；
- full static 无 NaN，梯度可到达所有分支；
- future-invariance 通过；
- 4K 吞吐不低于 500 tokens/s，或已租用足够 GPU；
- 参数量和状态 bytes 已记录。

动态 A 控制器与静态 B 执行模块必须分成两个 method ID。不得把动态控制器的收益归因于 B 模块本身。

## 7. 参数筛选规则

每种核心方法最多运行 4 个候选，统一使用：

```yaml
dataset: PG-19 train manifest seed 17
training_tokens: 2031616
train_context: 4096
validation_manifest: fixed
```

候选选择顺序：

1. 淘汰 causal、padding、梯度、状态重置测试失败者；
2. 淘汰 NaN、无法恢复 OOM 或平均吞吐低于执行闸门者；
3. 在剩余候选中选择 validation NLL 最低者；
4. 若两个候选 validation PPL 相对差小于 1%，选择吞吐更高者；
5. 若吞吐差小于 5%，选择 peak allocated memory 更低者；
6. 保存全部候选结果，不只保存胜者；
7. 配置冻结后，不允许查看 test 后重新选择。

由于 2M-token pilot 非常短，候选排序可能随训练继续而改变。报告中必须将筛选结果称为“本预算下的代表配置”，不能称为每种方法的全局最优配置。

## 8. 正确性闸门

每个进入正式训练的方法都必须通过：

### 8.1 Future-invariance

固定输入前缀，只改变未来 token。位置 `t` 及之前的 logits 必须保持一致：

```text
max_abs_delta <= 1e-5 in FP32 reference
max_abs_delta <= 5e-3 in BF16
```

### 8.2 Full-window 等价性

在小序列上将局部窗口扩大到覆盖完整合法历史，Longformer 输出应与共享权重的 causal Full Attention 对齐。

### 8.3 Padding invariance

增加被 mask 的 padding 不得改变有效 token logits。

### 8.4 梯度测试

所有预期可训练参数必须得到有限梯度；没有梯度的参数需要显式说明其设计原因。

### 8.5 Memory 生命周期

Memformer 和混合模型必须验证：

- 同一文档的下一 segment 会受到上一状态影响；
- reset 后结果恢复为无历史状态；
- batch 重排时 state 使用相同 permutation；
- 一个样本结束后不会污染另一个样本；
- memory 大小不随历史长度增长。

### 8.6 保存和恢复

从 recovery checkpoint 恢复后，下一批 loss、optimizer step、scheduler 和已训练 token 计数与未中断路径一致，允许 BF16 合理误差。

任何未通过项都要写入 `status=failed_correctness`，该方法不得进入 causal 主表。

## 9. 正式主实验矩阵

### 9.1 固定训练预算比较

每个冻结方法运行：

```yaml
dataset: PG-19
train_context: 4096
training_tokens: 10027008
seeds: [17, 29, 43]
```

每 61 steps 记录：

- train token-weighted NLL；
- validation token-weighted NLL/PPL；
- 累计训练 token；
- 累计墙钟和纯训练时间；
- training tokens/s；
- learning rate；
- gradient norm；
- peak allocated/reserved memory；
- NaN、OOM、backend 回退和数据加载等待时间。

训练期间每次 checkpoint validation 固定评分 262,144 个 token；训练结束后在固定 manifest 上评分 1,048,576 个 validation token 和 2,097,152 个 test token。如果官方 split 可用 token 少于该数量，则使用该 split 的全部有效 token 并记录实际数量。所有方法必须评分相同 target token。

主质量表报告：

```text
final validation/test NLL and PPL
best validation checkpoint test NLL and PPL
mean, standard deviation, 95% CI across seeds
PPL relative delta versus full_sdpa
total parameters and attention parameters
```

### 9.2 优化效果比较

比较的横轴必须同时提供：

- 已训练 token 数；
- optimizer steps；
- GPU 训练时间。

生成：

1. validation NLL vs trained tokens；
2. validation NLL vs GPU hours；
3. 达到预先冻结 NLL 阈值所需 token 数；
4. 达到相同 NLL 所需 GPU 时间；
5. 未达到阈值的方法标记 `not_reached`，不得外推。

NLL 阈值在 Full Attention seed 17 pilot 完成后、其他正式 test 结果产生前冻结，并写入协议。

### 9.3 等成本比较

固定主干和固定训练 token 回答“同样数据下谁学得更好”，但各架构实际计算成本可能不同。因此增加两个等成本表：

- equal-latency：4K training step median latency 差异在正负 10% 内；
- equal-memory：4K peak allocated memory 差异在正负 10% 内。

等成本配置只能从第一天已运行的候选配置中选择，不允许 test 后新增只对某个方法有利的候选。若无法匹配，报告最近配置和实际偏差，不宣称严格等预算。

## 10. 质量指标定义

语言建模采用 token-weighted 聚合：

```text
total_nll = sum(loss over all valid target tokens)
mean_nll = total_nll / number_of_valid_target_tokens
ppl = exp(mean_nll)
```

禁止先计算各 batch PPL 再做普通平均。

必须记录：

- train/validation/test `total_nll`；
- `valid_token_count`；
- token-weighted mean NLL/PPL；
- best 和 final checkpoint；
- 按 history length 分组的 NLL/PPL；
- synthetic accuracy 和 exact match；
- 三个 seed 的单独值、均值、标准差和 95% CI；
- 受控长历史样本级结果，便于 bootstrap。

3 seeds 的 CI 只能用于描述不确定性，不能仅凭区间重叠宣称完全等价。

## 11. 性能 benchmark

正式 benchmark 必须在同一张、同型号、独占 GPU 上依次重跑所有方法。训练可分布到多张同型号 GPU，但性能表不能混用 GPU 型号。

固定矩阵：

```yaml
sequence_lengths: [1024, 2048, 4096, 8192, 16384]
batch_size: 1
dtype: bfloat16
warmup_runs: 20
timed_runs: 50
modes:
  - inference_prefill
  - forward_backward
```

4K 额外测量完整 training step：

```yaml
mode: forward_backward_optimizer
gradient_accumulation: excluded_from_single_step_latency
```

计时要求：

- 使用 CUDA Event；
- 每次测量前后正确同步；
- 保存全部 50 个原始 samples；
- 报告 median、mean、P5、P95；
- 报告 tokens/s；
- 记录 peak allocated 和 peak reserved；
- 记录 requested backend 和 actual backend；
- OOM 保留为正式结果；
- 方法运行顺序随机化，完整矩阵至少重复一次检查顺序偏差；
- benchmark 期间禁止同卡运行其他进程。

派生指标：

```text
speedup = full_sdpa_median_latency / method_median_latency
memory_ratio = method_peak_allocated / full_sdpa_peak_allocated
quality_cost = test_NLL, latency, peak_memory three-dimensional Pareto
```

## 12. 结果 schema 和目录

执行 AI 应建立以下目标目录。现有 legacy 文件不得删除。

```text
experiments/long_context_10m_v1/
  protocol.yaml
  configs/
    base.yaml
    full_sdpa.yaml
    longformer.yaml
    performer.yaml
    memformer.yaml
    hybrid_static.yaml
  manifests/
    pg19_train_seed17.jsonl
    pg19_train_seed29.jsonl
    pg19_train_seed43.jsonl
    pg19_validation.jsonl
    pg19_test.jsonl
    controlled_history.jsonl
    synthetic_train.jsonl
    synthetic_test.jsonl
  jobs/
    jobs.csv
  runs/
    <run_id>/
      config.resolved.yaml
      environment.json
      metrics.jsonl
      stdout.log
      checkpoints/
  aggregate/
  figures/
  reports/
```

每个结果至少包含：

```json
{
  "run_id": "long_context_10m_v1_method_seed_timestamp",
  "protocol_version": "long_context_10m_v1",
  "result_label": "pilot_or_main_or_exploratory_ablation",
  "method": "longformer",
  "seed": 17,
  "dataset": "PG-19",
  "split": "train_or_validation_or_test",
  "causal": true,
  "train_context": 4096,
  "trained_tokens": 10027008,
  "config_hash": "...",
  "git_commit": "...",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "torch_version": "...",
  "cuda_version": "...",
  "dtype": "bfloat16",
  "requested_backend": "flash_sdpa",
  "actual_backend": "...",
  "status": "ok_or_oom_or_error_or_not_reached",
  "metrics": {}
}
```

作业状态只允许：

```text
queued
running
ok
oom
error
failed_correctness
needs_rerun
not_run
ineligible_causal
```

自动重试最多 1 次。相同错误再次出现后停止重试，保留日志并进入人工或 AI 诊断队列。

## 13. 目标 CLI 和实现任务

以下命令是执行 AI 应实现或适配的目标接口；当前仓库未必已经具有这些入口，不能假设命令现已可用。

```powershell
python -m experiments.long_context.prepare_data --config experiments/long_context_10m_v1/protocol.yaml
python -m experiments.long_context.validate_manifests --root experiments/long_context_10m_v1/manifests
python -m experiments.long_context.run_correctness --method all --config experiments/long_context_10m_v1/protocol.yaml
python -m experiments.long_context.run_train --config <resolved-config> --device cuda:0
python -m experiments.long_context.run_eval --run-dir <run-dir> --suite natural,history,synthetic
python -m experiments.long_context.run_benchmark --methods all --device cuda:0
python -m experiments.long_context.aggregate --root experiments/long_context_10m_v1/runs
python -m experiments.long_context.make_report --root experiments/long_context_10m_v1
```

实现要求：

1. 优先复用现有 `models/`、各架构目录和 `common/`，不要复制第二套 attention 数学实现；
2. 数据、模型、训练、评估和 benchmark 分层；
3. 所有命令接受配置文件并保存 resolved config；
4. 支持断点恢复；
5. 结果使用追加写 JSONL，进程崩溃时保留已完成记录；
6. 聚合和绘图只读取结构化原始结果；
7. 不在图表脚本中手工填写实验数字；
8. 任何 backend 回退、OOM、NaN 和缺失 checkpoint 都写入状态。

## 14. 硬件选择和 GPU 调度

### 14.1 推荐硬件

最低推荐：

```text
3 x RTX 4090 24GB
```

更稳妥：

```text
4 x RTX 4090 24GB，或
4 x RTX 5090 32GB，或
4 x A100 40/80GB
```

本 TinyLM 的 4K 主训练不要求 RTX 50 系。RTX 4090 足够；RTX 5090 的主要价值是吞吐和 32GB 显存。租用 Blackwell/50 系服务器前必须确认驱动、CUDA、PyTorch、Triton/FlexAttention 与该计算能力兼容。稳定的软件栈优先于理论峰值。

不建议对 TinyLM 使用 DDP。默认每张 GPU 运行一个独立 method/seed 作业。

### 14.2 四张 GPU 分配

第一轮固定：

| GPU | 队列 |
|---|---|
| GPU 0 | Full Attention seeds 17/29/43；之后正式 benchmark |
| GPU 1 | Longformer seeds 17/29/43 |
| GPU 2 | Performer seeds 17/29/43 |
| GPU 3 | Memformer seeds 17/29/43 |

P1 hybrid 在最早空闲 GPU 上运行，正式 latency 最终仍回到 GPU 0 重测。

如有 5 张 GPU，GPU 4 专门运行 hybrid。第 1 天参数筛选时四张卡按候选配置分发，不固定方法。

### 14.3 三张 GPU 分配

使用中央队列运行 12 个 P0 jobs，每张卡任何时刻只运行一个 job。先保证每个方法 seed 17 成功，再补 seed 29/43，避免某个实现错误占满两天。

### 14.4 三人负责七种方法时的分工和关键路径

若人员固定为以下分工：

| 人员 | 方法所有权 | 主要职责 |
|---|---|---|
| 人员 A | Performer、Reformer、Linformer | 三种近似架构的实现、筛选、训练、评估和小报告 |
| 人员 B | Longformer、Memformer | 局部稀疏与跨 segment memory 的实现、训练、评估和小报告 |
| 人员 C | Full Attention、hybrid | 统一数据/基线、混合结构、正式 benchmark、聚合和总报告 |

这里的“所有权”是代码、配置、结果检查和报告责任，不表示每个人只能使用一张 GPU。人员 A 拥有三种方法，是整个计划的关键路径，必须给其方法分配多张卡或允许中央队列动态调度。

正式主训练数为：

```text
人员 A：3 methods x 3 seeds = 9 runs
人员 B：2 methods x 3 seeds = 6 runs
人员 C：2 methods x 3 seeds = 6 runs
总计：7 methods x 3 seeds = 21 runs
```

四天完整预算还包括：

```text
参数筛选：
  人员 A：3 methods x 4 candidates x 2M = 24M tokens
  人员 B：2 methods x 4 candidates x 2M = 16M tokens
探索性 hybrid 消融：约 7 variants x 2M = 14M tokens
synthetic 训练：7 methods x 5M = 35M tokens
正式主训练：21 runs x 10M = 210M tokens
合计训练量：约 299M tokens
```

若每个人只使用一张 GPU，则人员 A 的串行训练时间为最大瓶颈。仅计算 9 次正式主训练：

| 人员 A 平均吞吐 | 9 次 10M 主训练 | 加筛选和 synthetic 后约需 |
|---:|---:|---:|
| 1,000 tokens/s | 25.0 h | 35.8 h |
| 1,500 tokens/s | 16.7 h | 23.9 h |
| 2,000 tokens/s | 12.5 h | 17.9 h |
| 2,500 tokens/s | 10.0 h | 14.3 h |
| 3,000 tokens/s | 8.3 h | 11.9 h |

10M 预算下三人各一张 GPU 已经可行，人员 A 仍是关键路径。其三种实现平均持续吞吐最好不低于 1,000 tokens/s；低于 500 tokens/s 时应增加 GPU 或优化实现。

推荐使用 4 张同型号 GPU；资源充足时使用 6 张进一步隔离方法队列：

| GPU | 主训练队列 |
|---|---|
| GPU 0 | Performer seeds 17/29/43 |
| GPU 1 | Reformer seeds 17/29/43 |
| GPU 2 | Linformer seeds 17/29/43 |
| GPU 3 | Longformer seeds 17/29/43 |
| GPU 4 | Memformer seeds 17/29/43 |
| GPU 5 | Full Attention seeds 17/29/43，随后 hybrid seeds 17/29/43 |

负责人可以同时监控自己名下的多张 GPU。GPU 0-4 完成主队列后，立即通过中央队列接手 hybrid、失败重跑、synthetic 或评估作业。

按总训练量约 299M tokens、GPU 有效利用率 75% 估算，不含最后统一 benchmark 的预计墙钟时间为：

| 平均持续吞吐/每张卡 | 3 GPUs | 4 GPUs | 6 GPUs |
|---:|---:|---:|---:|
| 500 tokens/s | 73.8 h | 55.4 h | 36.9 h |
| 750 tokens/s | 49.2 h | 36.9 h | 24.6 h |
| 1,000 tokens/s | 36.9 h | 27.7 h | 18.5 h |
| 1,500 tokens/s | 24.6 h | 18.5 h | 12.3 h |
| 2,000 tokens/s | 18.5 h | 13.8 h | 9.2 h |

最后的受控历史评估、PG-19 test、1K-16K benchmark、异常重测和汇总还应预留 12-20 小时。由此得到硬件建议：

- 3 GPUs：平均吞吐达到约 1,000 tokens/s 时，训练加最终评估预计约 49-57 小时，可行；
- 4 GPUs：即使平均吞吐约 750 tokens/s，预计约 49-57 小时，推荐；
- 6 GPUs：适合实现低于 750 tokens/s、需要大量失败重跑，或希望给每种方法独立队列的情况；
- 7 GPUs：可以将七种方法完全分卡，但对 10M 预算通常没有必要。

若 hybrid 未通过 P1 闸门，人员 C 只运行 Full Attention，正式主训练从 21 次降为 18 次；此时必须在结果中明确 hybrid 未进入主实验的原因。

## 15. 吞吐量和时间闸门

第一天为每个冻结候选运行至少 1M tokens 的计时探针。估算公式：

```text
estimated_10M_hours = measured_1M_hours x 10
estimated_hours = 10027008 / measured_tokens_per_second / 3600
```

决策规则：

| 最慢正式方法的持续吞吐 | 动作 |
|---:|---|
| `>= 1,000 tokens/s` | 3 张 GPU 可完成 P0/P1；4 张 GPU 有充分重跑余量 |
| `750-1,000` | 推荐 4 张 GPU；保持 10M 和 3 seeds |
| `500-750` | 租 5-6 张同型号 GPU，或先优化最慢实现 |
| `< 500` | 暂停正式训练，定位 Python loop、dense fallback、数据加载或同步瓶颈 |

禁止首先删除 seeds。若必须缩减，顺序为：

1. 删除 P2；
2. 删除 dynamic hybrid；
3. 删除非关键 synthetic 扩展；
4. 保留 P0 的 10M 和 3 seeds；
5. 只有 P0 仍无法在期限内完成时，统一降为 5M，并升级 protocol version，所有方法一起重跑或截取同一 token checkpoint。

不得让某些方法训练 5M、另一些训练 10M 后放进同一最终质量列。

## 16. 四天详细时间表

### 第 1 天 00:00-06:00：环境、数据和统一接口

- 记录 GPU、driver、CUDA、PyTorch、Triton/Flex/xFormers 环境；
- 下载或挂载 PG-19；
- 构建 GPT-2 BPE token cache；
- 生成 train/validation/test manifests；
- 实现或适配统一 runner 和结果 schema；
- 同时用 WikiText-103 跑控制 smoke；
- 不启动 10M 正式训练。

完成标准：随机抽查 manifest 无 split 泄漏、无跨文档样本，配置 hash 和环境文件可生成。

### 第 1 天 06:00-12:00：正确性闸门

- 运行 future-invariance；
- 运行 padding、梯度、保存恢复；
- 运行 Longformer full-window 等价；
- 运行 Memformer state/reset/reorder；
- 所有方法执行 50-100 steps smoke；
- 修复失败后从头重测。

完成标准：P0 全部 `passed`；失败方法不能进入后续 GPU 队列。

### 第 1 天 12:00-24:00：参数筛选和探索性消融

- 每种方法最多 4 个 2M-token 候选；
- 候选可在多 GPU 并行；
- 每个候选保存 validation NLL、吞吐、显存和失败状态；
- 混合结构运行 2M-token 的 7 项探索性消融；
- 对拟进入正式训练的配置运行 1M-token 持续吞吐探针；
- 当天结束前冻结主配置和 NLL threshold。

若数据准备或代码修复超过 12 小时，优先使用预设主配置，停止完整参数筛选，以保护正式训练时间。

### 第 2 天：P0 正式训练第一轮

- 先并行运行所有方法 seed 17；
- seed 17 首次 validation 正常后提交 seed 29/43；
- 每 61 steps validation 和 checkpoint；
- 聚合程序持续检查缺失字段、NaN、OOM 和吞吐下降；
- 不根据中途 test 结果更改超参数。

### 第 3 天：完成 P0，运行 P1

- 完成 P0 剩余 seeds；
- 对失败作业最多重试一次；
- GPU 空闲后运行 hybrid_static 3 seeds；
- 并行构建受控历史和 synthetic test manifests；
- 开始 best checkpoint 的 PG-19 validation/test。

### 第 4 天 00:00-12:00：质量和机制评估

- 固定目标 history-length 评估；
- PG-19 best/final checkpoint test；
- synthetic 5M-token 训练或微调和长度泛化测试；
- 生成单 seed 和 3-seed 汇总；
- 检查每个主表单元格可追溯到 run_id。

### 第 4 天 12:00-20:00：性能 benchmark

- 固定一张 GPU，清空其他进程；
- 随机顺序运行全部方法的 1K/2K/4K/8K/16K；
- 保存 20 warmup、50 timed samples；
- 记录 actual backend、显存和 OOM；
- 对异常高波动点重测一次。

### 第 4 天 20:00-24:00：汇总和冻结

- 生成固定预算表、等成本表、长历史曲线和 Pareto 图；
- 生成缺失、失败、OOM 和协议偏差清单；
- 写每种架构的优点、缺点、适用范围和证据 run_id；
- 保存代码 commit、配置 hash、数据 manifest hash；
- 将未完成项目明确标为 future work，不伪装完成。

## 17. 阶段闸门和停止条件

### 进入参数筛选前

- PG-19 manifest 验证通过；
- causal、padding、梯度、reset 测试通过；
- runner 能写合法 JSONL；
- recovery checkpoint 可恢复。

### 进入正式训练前

- 每个方法已冻结唯一主配置；
- 每个方法完成 1M-token 吞吐探针；
- 预计总 GPU 时间不超过剩余资源的 75%；
- NLL threshold 已冻结；
- test 未用于选择配置。

### 主实验完成条件

- 每个 P0 方法具有 3 个成功 seeds，或明确列出无法恢复的失败；
- 每个 run 达到同一 training token budget；
- PG-19 best checkpoint test 完成；
- 受控 history-length 评估完成；
- 1K-16K benchmark 完成到 OOM；
- 原始结果通过 schema validator；
- 主表没有混合不同 protocol version。

立即停止单个 run 的条件：

- loss 或 gradient 出现 NaN/Inf；
- 连续两次 validation 明显异常且相同配置 baseline 正常；
- backend 静默回退导致方法语义改变；
- 数据样本跨 split 或跨文档；
- recovery 后 token 计数或数据顺序不一致；
- 超过预计时间 2 倍且吞吐持续下降。

## 18. 最终表格和图形

至少输出：

### 表 A：统一设置

方法、主干、专属参数、总参数、训练 token、seed、GPU、actual backend。

### 表 B：固定训练预算质量

| 方法 | Test NLL | Test PPL | 相对 Full PPL | 3-seed std | 最佳 step |
|---|---:|---:|---:|---:|---:|

### 表 C：优化效率

| 方法 | 达到阈值的 tokens | 达到阈值的 GPU h | Train tokens/s | 未达到 |
|---|---:|---:|---:|---|

### 表 D：长历史收益

| 方法 | 512 NLL | 2K NLL | 4K NLL | 8K NLL | 16K NLL | 16K delta |
|---|---:|---:|---:|---:|---:|---:|

### 表 E：机制任务

按任务、长度和依赖距离报告 accuracy/exact match，不只报告总平均。

### 表 F：系统性能

| 方法 | 长度 | median latency | tokens/s | peak allocated | peak reserved | OOM | actual backend |
|---|---:|---:|---:|---:|---:|---|---|

### 图形

- validation NLL vs trained tokens；
- validation NLL vs GPU hours；
- NLL/PPL vs available history；
- mechanism accuracy vs dependency distance；
- latency vs sequence length；
- peak memory vs sequence length；
- quality-latency-memory Pareto；
- hybrid leave-one-out ablation 图，如果正式消融完成。

## 19. 如何写架构优缺点

每个优缺点必须绑定至少一个质量证据和一个成本证据，建议采用：

```text
观察：在 8K history 的固定目标评估中，方法 X 的 delta NLL 为 ...。
代价：其 8K median latency 和 peak memory 分别为 ...。
机制证据：在 distance=4K 的 Passkey 上 exact match 为 ...。
结论边界：该结论只适用于 TinyLM-long-v1、PG-19、causal、当前实现和记录的 GPU。
```

禁止仅根据理论复杂度写“更快”，或仅根据单 seed 写“质量更好”。

推荐最终回答以下问题：

1. Longformer 的局部精确性是否以丢失远程信息为代价？窗口扩大后收益和成本如何变化？
2. Performer 是否在较低显存下获得当前 4K/8K 全局信息？随机近似是否导致质量或 seed 波动？
3. Memformer 是否能跨 8/16/32 个 segment 保留关键信息？固定槽容量何时饱和？
4. Full Attention 在 4K 内是否仍是质量或速度强基线？其 OOM 或成本交叉点在哪里？
5. 混合结构是否位于单一架构无法达到的 Pareto 区域？提升来自哪个分支？

## 20. 四天后的扩展顺序

只有 P0 完整后，按下列顺序扩展：

1. hybrid_static 3 seeds；
2. 正式 `no_local/no_global/no_memory` 消融；
3. Reformer causal；
4. Linformer causal；
5. 如后续确实需要研究训练规模，另建新的扩展训练协议；该结果不混入本计划 10M 主表；
6. train context 从 4K 增加到 8K；
7. 在第二种同型号或数据中心 GPU 上复测硬件敏感性。

## 21. 执行摘要

最小可信四天方案是：

```text
数据：PG-19
任务：causal long-context LM
主干：6 layers, hidden 384, 8 heads, FFN 1536
训练长度：4096
训练预算：10,027,008 tokens
方法：Full SDPA, Longformer, Performer, Memformer
随机种子：17, 29, 43
GPU：推荐 4 x RTX 4090 24GB 或更高
质量评估：PG-19 PPL + 固定 512-token 目标的 512-16K history sweep，RoPE 容量 32K
机制评估：Local Copy + Retrieval + Cross-Segment Passkey
性能评估：1K/2K/4K/8K/16K latency, tokens/s, memory, OOM
主结论：固定训练预算、等成本和 Pareto 三种视角共同解释架构优缺点
```

执行优先级始终为：正确性 > P0 三个 seeds > 长历史评估 > 性能 benchmark > hybrid > P2 扩展。10M 结果只代表短预算早期训练表现；任何缩减都必须同时作用于可比较方法，并通过新的 protocol version 留痕。
