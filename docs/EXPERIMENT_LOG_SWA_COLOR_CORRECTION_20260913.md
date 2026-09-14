# SWA 随机颜色纠正实验日志

日期：2026-09-13  
主计划：[PLAN_SWA_COLOR_CORRECTION_20260913.md](PLAN_SWA_COLOR_CORRECTION_20260913.md)

## Milestone 1：Full-Attention path 与代码回归

状态：完成

### 实现

- 新增 `FullCausalAttention`，复用现有 SWA attention 的 Q/K/V/out 参数命名；
- `AdaptiveFactMemoryLM(..., attention_mode="full")` 支持 unbounded causal KV；
- `train_memory_stress.py` 新增 `--attention-mode {swa,full}`；
- 默认 `attention_mode="swa"`，历史接口和默认实验行为保持不变；
- Full Attention cache 使用动态增长的 `LocalKVCache.recent_*`，不使用 sink path。

### 验证

```text
PYTHONPATH=adaptive_gated_fact_memory/src:adaptive_gated_fact_memory/scripts \
python -m unittest discover -s adaptive_gated_fact_memory/tests -v
```

结果：`25/25 passed`。

新增测试：

1. Full Attention 整段 forward 与分块 forward 最大误差在 `2e-5` 内；
2. Full Attention cache 可增长到 20 tokens，超过 SWA 的 8-token test budget；
3. Full Attention 与 SWA 的 projection state-dict 命名和形状兼容。

### 尚未完成

- 尚未训练正式 FA/SWA color task；
- 尚未启用 `answer_leading_space` 的正式评估；
- 尚未执行 out-of-window stress 或多 seed。

## Milestone 2：in-window seed=17

状态：进行中。300-step sanity 对照已完成；由于两种 attention 均未达到高 EM，
将继续用相同协议做 1000-step 训练确认是否只是训练不足。

实验目的：事实保持在 124-token recent window 内时，验证 FA-task 和 SWA-task
是否都能学习随机 name→color 复制。

共同设置：

```text
parent = tinystories_backbone_s17_10m_20260908/checkpoint.final.pt
seed = 17
answer_leading_space = true
segment_length = 32
delay = 1
noise = 0
```

正式 run 目录：

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/
  adaptive_gated_fact_memory_color_corrected/

### 300-step sanity 对照（已完成）

共同设置：`seed=17`、`memory_policy=none`、`answer_leading_space=true`、
`segment_length=32`、`delay=1`、`noise=0`、`batch_size=2`、
`gradient_accumulation_steps=2`、`steps=300`、`eval_examples=24`。

| run | attention | elapsed (s) | in-window EM | answer NLL | first-token NLL | mean rank | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `color_corrected_swa_inwindow_s17_m2` | SWA(128,4) | 106.07 | 5/24 = 20.83% | 1.3892 | 2.7779 | 5.333 | 66.67% |
| `color_corrected_fa_inwindow_s17_m2` | Full causal | 79.85 | 5/24 = 20.83% | 1.4051 | 2.8096 | 5.542 | 66.67% |

解释：两者均主要输出少数高频颜色，且没有达到 in-window control 的预期，
因此这组 300-step 结果只能作为 sanity 基线，不能作为 FA/SWA 能力差异结论。
对应完整 `summary.json`、checkpoint 和 `metrics.jsonl` 均保存在上述 run 目录。

### 1000-step 训练确认（seed=17）

Full-Attention 运行已完整结束：

| run | attention | steps | elapsed (s) | in-window EM | answer NLL | first-token NLL | mean rank | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `color_corrected_fa_inwindow_s17_m2_1000` | Full causal | 1000 | 254.85 | 24/24 = 100% | 0.00636 | 0.01271 | 1.000 | 100% |
| `color_corrected_swa_inwindow_s17_m2_1000_retry2` | SWA(128,4) | 1000 | 350.45 | 24/24 = 100% | 0.00837 | 0.01674 | 1.000 | 100% |

SWA 1000-step retry 因用户要求扩容而暂停。首次运行
`color_corrected_swa_inwindow_s17_m2_1000` 在写 checkpoint 时因目标盘空间耗尽
失败；已将三个中间 checkpoint（step 250/500/750）原样移至
`.failed_run_archive/color_corrected_swa_inwindow_s17_m2_1000/`，未删除。
重试 run `color_corrected_swa_inwindow_s17_m2_1000_retry1` 在 step 179 收到
`KeyboardInterrupt`，当前只保留 `config.resolved.json` 和截至 step 179 的
`metrics.jsonl`，没有生成 final checkpoint。扩容后应从头重新执行该 SWA 对照，
或在实现 resume 选项后从有效 checkpoint 继续；不要把该暂停 run 当作结果。
```

SWA retry2 已于扩容前的清理后完整完成。结论：在事实仍处于 SWA 的
124-token recent window 内时，Full-Attention 与 SWA-task 都能学会随机
name→color 复制；此前 300-step 的 20.83% 是训练不足，不是 SWA 的信息下界。
两者的正式 1000-step in-window control 均为 100% EM。

### Milestone 3：out-of-window FA/SWA diagnostic（seed=17）

共同设置：`answer_leading_space=true`、`segment_length=32`、
`train_delays=1,4,16`、`train_noise=0`、`eval_delays=1,4,16`、
`eval_noise=0`、`steps=1000`、`batch_size=2`、梯度累积 2、每条件 24 条。
delay=16 对 SWA 已明显超过 124-token recent window；Full Attention 不受该窗口
限制。

| run | attention | elapsed (s) | delay=1 EM | delay=4 EM | delay=16 EM | overall EM | overall answer NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `color_corrected_fa_outwindow_s17_m3` | Full causal | 263.36 | 23/24 = 95.83% | 22/24 = 91.67% | 20/24 = 83.33% | 65/72 = 90.28% | 0.1673 |
| `color_corrected_swa_outwindow_s17_m3` | SWA(128,4) | 343.32 | 23/24 = 95.83% | 3/24 = 12.50% | 1/24 = 4.17% | 27/72 = 37.50% | 1.0683 |

该结果验证了预期：窗口内两者相当；事实离开 SWA 窗口后，纯 SWA 迅速退化，
而 Full Attention 仍保持较高准确率。这是信息可见性差异，不是随机颜色的固定
词表命中问题。两个 run 的完整结果记录和 final checkpoint 均已保留。

### Milestone 4：Fixed-LRU（seed=17）

运行：`color_corrected_fixed_lru_outwindow_s17_m4`，SWA(128,4) + fixed LRU，
其余数据、tokenizer、seed、loss、训练预算与 Milestone 3 完全相同。

| delay | EM | target survival | Recall@1 | answer NLL | first-token rank |
|---:|---:|---:|---:|---:|---:|
| 1 | 79.17% | 100% | 100% | 0.4482 | 1.250 |
| 4 | 12.50% | 100% | 100% | 1.0662 | 3.958 |
| 16 | 29.17% | 100% | 100% | 1.1121 | 3.625 |

整体 EM：`29/72 = 40.28%`；整体 answer NLL：`0.8755`。
解释：Fixed-LRU 已稳定保存并检索目标事实，但生成端尚未可靠利用 retrieved
value，故 EM 低于 retrieval 指标；该结果支持把写入/读取与答案利用分开报告。

### Gated Memory（seed=17，同一 Milestone 4 条件）

运行：`color_corrected_gated_outwindow_s17_m4`，SWA(128,4) + gated memory。

| delay | EM | target survival | Recall@1 | answer NLL | first-token rank |
|---:|---:|---:|---:|---:|---:|
| 1 | 95.83% | 100% | 100% | 0.2286 | 1.042 |
| 4 | 29.17% | 100% | 100% | 0.9847 | 3.125 |
| 16 | 33.33% | 100% | 100% | 0.9437 | 3.333 |

整体 EM：`38/72 = 52.78%`；整体 answer NLL：`0.7190`。

### Value-token Gated（seed=17，同一 Milestone 4 条件）

运行：`color_corrected_value_token_outwindow_s17_m4`，SWA(128,4) + gated
memory + `value_token_alignment`。

| delay | EM | target survival | Recall@1 | value-span exact | memory-token exact | answer NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100% | 100% | 100% | 100% | 100% | 0.00297 |
| 4 | 100% | 100% | 100% | 100% | 100% | 0.00374 |
| 16 | 100% | 100% | 100% | 100% | 100% | 0.00304 |

整体：`72/72 = 100% EM`；整体 answer NLL：`0.00325`。
这表明 value-token alignment 修复了旧 Gated 的“已检索但不能稳定生成具体
颜色 token”瓶颈。该结果仍属于颜色任务微调，不是 training-free SWA。

### 磁盘清理记录

为恢复训练所需空间，已清理数据盘 runs 目录下所有正式 run 的
`checkpoint.step_*.pt` 中间快照（共 55 个，约 23.75 GiB）。每个 run 的
`checkpoint.final.pt`、`summary.json`、`metrics.jsonl` 和 `config.resolved.json`
均保留；TinyStories backbone 和当前颜色任务 run 也保留。中间快照不是结果，
如需重新生成，应使用相同配置从 parent 重新训练。

### 第 6 步：Full-Attention TinyStories parent 与 training-free SWA（seed=17）

新增训练 run：
`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_tinystories/`
`tinystories_full_parent_s17_step6_20260913/`

训练配置：TinyStories 10,000,000 tokens、context=512、有效 batch=8192 tokens、
1221 optimizer steps、Full causal attention、仅 next-token LM loss；未启用 episodic
fact memory，也没有颜色任务微调。训练过程曲线：
`training_accuracy_curve.png`。验证 probe 的 token accuracy 从 `0.0008%`（step 0）
提升到 `41.87%`（step 1221），full validation 为 `41.96%`，NLL=`2.9463`，
PPL=`19.0348`。

之后冻结 `checkpoint.final.pt`，只在推理时切换 attention cache，未更新任何参数：

| 模式 | budget | token accuracy | NLL | PPL | tokens/s |
|---|---:|---:|---:|---:|---:|
| Full reference | unbounded | 41.9609% | 2.9463 | 19.0348 | 125,029 |
| SWA | 64 + 4 sinks | 41.7793% | 2.9573 | 19.2455 | 108,501 |
| SWA | 128 + 4 sinks | 42.0221% | 2.9435 | 18.9831 | 95,281 |
| SWA | 256 + 4 sinks | 42.0009% | 2.9438 | 18.9875 | 75,830 |
| SWA | 512 + 4 sinks | 41.9625% | 2.9463 | 19.0350 | 52,972 |

结果文件：`training_free_swa_results.json`。该实验是论文式 inference-only SWA，
与前面的颜色任务微调 SWA-task 分开统计。

### Milestone 7：Memformer stateful adapter 正确性闸门（seed=17 protocol）

状态：完成；未启动长训练。

本 milestone 只验证新的 `MemformerDecoderLM`（decoder adapter）的状态语义，
不包含颜色任务训练，也没有加入 memory-to-token reconstruction、
value-token alignment、value-span supervision 或 pointer/copy head。

实现修复：

1. attention 的多头结果在 `MemformerSegmentAttention` 中显式合并为
   `[batch, tokens, hidden]` 后再进入 memory/output projection。此前遗漏该
   reshape，会在协议的 4/8-head 配置下触发矩阵维度错误；
2. 完全 padding 的 segment 现在是 recurrent state 的 no-op，不会改写 memory
   或增加 `segment_count`，从而避免 packed batch 中较短样本被 padding 污染；
3. `MemformerState` 增加层数、shape、device、dtype、counter 和跨层一致性检查，
   并导出 `MemformerState`、`MemformerOutput`、`MemformerDecoderLM` 公共接口。

验证命令和结果：

```text
PYTHONPATH=adaptive_gated_fact_memory/src:adaptive_gated_fact_memory/scripts \
python3 -m unittest discover -s adaptive_gated_fact_memory/tests -v
结果：35/35 passed（其中新增 Memformer 测试 10 项）。

PYTHONPATH=adaptive_gated_fact_memory/src python3 -m compileall -q \
  adaptive_gated_fact_memory/src adaptive_gated_fact_memory/tests
结果：通过。
```

新增闸门覆盖：state shape/device/dtype/bytes、reset 等价新样本、batch reorder、
one-shot 与逐 segment 一致、future-token invariance、当前 segment 新 state 不
反哺当前 logits、state zero/swap 的因果影响、全 padding no-op、有限梯度和普通
tied LM head 输出接口。CUDA RTX 4090 BF16 smoke 亦通过，输出 shape=`[2,9,97]`。

在正式配置（6 layers、hidden=384、64 slots、batch=1）下，state bytes 为：
`294,928 B`（BF16，含两个 long counters；memory tensors 占 `294,912 B`），
FP32 为 `589,840 B`。代码 hash：
`memformer.py`=`102d28ec970070f9966ad6bb12faa98a951f4833c159266d4969a1d0cf68490a`。

边界说明：该实现是本项目的 decoder-only Memformer-style adapter，不声称完整
复现论文的 encoder-decoder MemBART；后续颜色实验仍需单独验证 parent 权重映射、
segment 梯度规则和训练曲线。

### Milestone 8：共同 parent 的短程 sanity（seed=17）

状态：完成；可进入 M9 正式共同目标训练。

新增 runner：
`adaptive_gated_fact_memory/scripts/train_color_m8.py`。三种条件使用同一
`SelectiveMemoryStressGenerator`、GPT-2 tokenizer、`answer_leading_space=true`、
同一 Full-Attention TinyStories parent、同一 `L_full_causal_lm + L_answer`
（`lambda_answer=1.0`）目标和 AdamW 设置。M8 固定 `delay=1`、`noise=0`、
`segment_length=32`，训练集为 24 个固定样本，验证集为 24 个独立随机映射的
held-out 样本；Memformer 样本内保留跨 round 梯度，FA/SWA cache 在 round 间
detach。没有任何事实标签、重建、value-token alignment 或 pointer/copy head。

正式短程 run（500 steps，batch=2）位于：
`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_color_m8/`

| 条件 | 训练集 answer EM | held-out answer EM | held-out answer NLL | 状态 |
|---|---:|---:|---:|---|
| FA-task (`m8_fa_task_s17_v2`) | 24/24 = 100% | 0/24 = 0% | 3.5100 | ok |
| SWA-task (`m8_swa_task_s17_v2`) | 24/24 = 100% | 0/24 = 0% | 3.6303 | ok |
| Memformer (`m8_memformer_s17_v3`) | 24/24 = 100% | 0/24 = 0% | 4.0977 | ok |

三个 run 均无 NaN/OOM，训练 objective 明显下降（FA 17.1466→约 0.51，
SWA 17.0435→约 0.51，Memformer 16.5830→约 0.61 的最低 step loss）。
Memformer 的 v2（`m8_memformer_s17_v2`）曾在 round 间 detach，训练集 EM 仅
75%；改为保留 query loss 到 recurrent write 的梯度后，v3 达到 100%。这条
对照保留为训练规则诊断，不能与 v3 混为同一正式结果。

held-out 0% 不表示模型或 parent 失效：M8 的 24 个固定训练映射主要用于确认
过拟合、答案 tokenization、state 传播和 loss 数值；短程预算尚不足以学习从
训练模板到独立验证模板的算法式复制。正式泛化结论留给 M9 的 2,000-step、
多映射训练协议，不能把 M8 held-out 数字当作架构下界。

Memformer 正式配置（6 layers、hidden=384、64 slots、BF16、batch=2）的
recurrent state bytes 为 `589,856 B`（memory tensors `589,824 B`，另含两个
long counters）。M8 runner 已通过一、三方法 CUDA smoke；其 parent 映射明确
加载 embedding、final norm、每层 token Q/K/V/O、FFN norm/in/out，只有
Memformer-specific memory projections 和 update gate 随机初始化。

### Milestone 9：正式共同目标比较（seed=17、noise=0）

状态：完成。三个方法均完成 `2,000` optimizer steps，均保存
`checkpoint.final.pt`、`checkpoint.best.pt`、`metrics.jsonl`、`summary.json` 和
解析后的配置。训练没有出现 NaN/OOM；训练曲线见：

`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_color_m9/m9_training_accuracy_objective_curves.png`

共同设置为 M9 冻结协议：同一个 Full-Attention TinyStories parent、GPT-2
tokenizer、随机 name -> color 映射、`answer_leading_space=true`、segment length
32、delay=`1,4,16`、noise=0、`L_full_causal_lm + L_answer`（lambda=1.0）、
batch=2、梯度累积=2、AdamW 和 2,000 steps。Memformer 不使用 reconstruction、
value-token alignment、事实标签或 pointer/copy head。

验证集为 72 个 held-out examples（每个 delay 24 个）：

| 条件 | delay=1 EM | delay=4 EM | delay=16 EM | overall EM | answer NLL | first-token rank |
|---|---:|---:|---:|---:|---:|---:|
| FA-task | 95.83% | 87.50% | 4.17% | 62.50% | 0.6791 | 3.708 |
| SWA-task | 79.17% | 0% | 4.17% | 27.78% | 1.4591 | 34.833 |
| Memformer | 100% | 100% | 100% | 100% | 0.0170 | 1.000 |

FA-task 的 delay=16 下降不是窗口上限，而是该 2,000-step run 在长 filler
条件上的泛化不足；SWA-task 的 delay=4/16 则同时受局部可见性限制。Memformer
的共同主任务目标在该协议下恢复了随机映射，并且训练 answer-token accuracy
在最后 step 为 100%。因此 M9 结果可以进入 M10 的状态因果诊断，但不能把
Memformer 的 100% 归因于额外 lexical supervision。

### Milestone 10：Memformer recurrent-state 因果诊断

状态：完成。脚本和结果：

`adaptive_gated_fact_memory/scripts/diagnose_memformer_m10.py`

`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_color_m9/m10_memformer_state_diagnostics_s17.json`

诊断使用 M9 的 `checkpoint.final.pt`、相同 72 个 held-out examples 和普通 tied
LM head；干预不训练任何新参数，也保留每个 query 自己的 RoPE 位置计数。结果：

| state 条件 | delay=1 EM | delay=4 EM | delay=16 EM | overall answer NLL | 说明 |
|---|---:|---:|---:|---:|---|
| normal | 100% | 100% | 100% | 0.018 | prefix state 传入 query |
| zero memory | 8.33% | 8.33% | 8.33% | 2.845 | 清空所有 memory vectors |
| swapped memory | 0% | 0% | 0% | 3.729 | 两样本 memory vectors 互换 |
| per-segment reset | 8.33% | 8.33% | 8.33% | 2.845 | 每个显式 32-token prefix segment 后清空 |
| last slot only | 8.33% | 8.33% | 8.33% | 2.843 | 仅保留 slot 63 |

normal state 的 prefix RMS norm 随 delay 为 `0.0590 / 0.0633 / 0.0724`，而
zero/per-segment reset 的 query memory norm 为 0。zero 相对 normal 的 EM 下降
为 `91.67` 个百分点，swapped 为 `100` 个百分点；这和 state delta、延迟无关
的 zero baseline 一致，说明模型使用的是样本特定 recurrent memory，而不是仅
依赖 query 模板或位置计数。

对 state 缩放的补充结果（EM，delay=1/4/16）：

| scale | delay=1 | delay=4 | delay=16 |
|---:|---:|---:|---:|
| 0.0 | 8.33% | 8.33% | 8.33% |
| 0.25 | 12.50% | 16.67% | 8.33% |
| 0.5 | 62.50% | 58.33% | 50.00% |
| 1.0 | 100% | 100% | 100% |
| 2.0 | 100% | 100% | 95.83% |

state noise（噪声标准差为 state RMS 的 0.01、0.1、0.5、1.0 倍）在本 checkpoint
上仍保持 100% EM（delay=1/4/16 均相同）；这表示当前答案 margin 较大，不能
用小幅随机噪声作为比 zero 更敏感的必要性测试。完整 NLL、rank、state norm 和
每个样本行级结果保存在 JSON 中。

M9/M10 的解释边界：Memformer 在本项目中是 decoder-only Memformer-style
adapter，不是论文的完整 encoder-decoder MemBART 复现。当前证据支持“该 adapter
通过跨 segment recurrent state 使用了样本信息”，不支持把它写成完整论文复现。

### Milestone 11：Fixed-LRU 与 Gated Memory 原始分组对照

状态：完成。两者均在与 M9 完全相同的新 Full-Attention TinyStories parent、
seed=17、GPT-2 tokenizer、随机 name -> color 数据、`answer_leading_space=true`、
segment length=32、delay=`1,4,16`、noise=0、batch=2、梯度累积=2、AdamW、
2,000 steps 下训练。两次运行均 `status=ok`，无 NaN/OOM；训练曲线见：

`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_color_m11/m11_training_accuracy_objective_curves.png`

| 条件 | delay=1 EM | delay=4 EM | delay=16 EM | overall EM | answer/teacher NLL | target survival | read hit@k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-LRU | 95.83% | 58.33% | 58.33% | 70.83% | 0.4113 | 100% | 100% |
| Gated Memory | 95.83% | 54.17% | 45.83% | 65.28% | 0.5961 | 100% | 100% |

两种方法在验证集均只激活约 1 个 user slot，预分配 state bytes 均为
`2,423,702 B`（8 slots；该数包含语义/词法/metadata 张量，不能与 Memformer
latent-only state bytes 直接称为相同 memory budget）。Fixed-LRU 的最终训练
total 为 `0.1748`，Gated Memory 为 `0.4865`；各自最优训练 total 分别为
`0.1338`（step 未另行作为泛化选择标准）和 `0.1311`。Gated 的 native
结构项（fact start/length、write、key alignment、retention、budget）保持启用，
而 `value_token_alignment=false`、reconstruction=false、pointer/copy=false；
因此该表属于原生系统对照，不是与 FA-task/SWA-task/Memformer 相同监督的严格
排名。两者 target survival/read hit 均为 100%，但生成端 EM 仍低于检索指标，
说明 lexical value 到普通 LM head 的利用仍是独立瓶颈。

### Milestone 12：noise stress（seed=17，eval-only）

状态：完成。脚本
`adaptive_gated_fact_memory/scripts/evaluate_color_m12.py` 不调用 optimizer，
对 M9/M11 的 final checkpoint 使用相同的 held-out generator，评估
noise=`0,2,4,8`、delay=`1,4,16`，每个 cell 24 条。结果文件：

`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_color_m12/m12_noise_stress_s17.json`

下表为每个 noise 桶中三个 delay 的 EM 平均；括号内为三个 delay 的平均
teacher-forced answer NLL。noise=0 与 M9/M11 正式结果一致，后续 noise 桶只作
鲁棒性 stress，不替代主 leaderboard。

| 条件 | noise=0 | noise=2 | noise=4 | noise=8 |
|---|---:|---:|---:|---:|
| FA-task | 62.50% (0.679) | 13.89% (1.896) | 9.72% (2.054) | 2.78% (2.242) |
| SWA-task | 27.78% (1.459) | 5.56% (1.929) | 19.44% (1.686) | 6.94% (2.096) |
| Memformer | 100.00% (0.017) | 5.56% (2.417) | 2.78% (2.862) | 1.39% (3.241) |
| Fixed-LRU | 70.83% (0.411) | 22.22% (1.683) | 9.72% (2.331) | 8.33% (2.599) |
| Gated Memory | 65.28% (0.596) | 8.33% (1.968) | 13.89% (2.081) | 5.56% (2.124) |

在 noise=0 时，Fixed-LRU/Gated 的 target survival 均为 100%、read hit@k 均为
100%，但答案 EM 只有 70.83%/65.28%；这再次区分了“事实仍在记忆中”和“普通
LM head 能否把 lexical value 生成出来”。噪声增大后，Fixed-LRU 的平均
target survival/read hit@k 从 noise=0 的 100%/100% 降至 noise=8 的
93.06%/0%，Gated Memory 则为 100%/58.33%。Memformer 的 noise=0 正常状态
结果保持 100%，但在 noise>=2 时迅速退化，说明其 recurrent state 对 filler
扰动敏感。

### Milestone 14：value-token alignment 配对消融（执行协议）

M14 计划使用新 parent 重跑两个此前没有严格配对的条件：

1. `Value-token Gated` = `memory_policy=gated` + `value_token_alignment=true`；
2. `Fixed-LRU + alignment` = `memory_policy=fixed_lru` +
   `value_token_alignment=true`。

两者均使用 M11 的 seed=17、随机 name→color、GPT-2 tokenizer、
`answer_leading_space=true`、segment length=32、delay=`1,4,16`、noise=0、
2,000 steps、batch=2、gradient accumulation=2、AdamW 和同一
Full-Attention TinyStories parent。alignment 权重固定为 start/length/candidate/
retrieved=`0.50/0.25/0.50/0.50`；答案仍由普通 decoder LM head 生成。

验收标准：无 NaN/OOM；训练 objective 和 answer accuracy 曲线在 2,000 steps
内稳定下降/收敛；同时记录 answer EM/NLL、target survival/read hit、
value-span exact、memory-token exact、active slots 和 state bytes。完成后以
`Value-token Gated − Gated Memory` 与 `Fixed-LRU + alignment − Fixed-LRU`
作为创新点的主要 paired delta，并在 M15 对两个新 checkpoint 做 noise stress。

### Milestone 13：效率、状态和最终 seed=17 汇总

状态：完成。统一 benchmark 脚本
`adaptive_gated_fact_memory/scripts/benchmark_color_m13.py` 在 RTX 4090、CUDA
BF16 autocast、batch=1、held-out `noise=0, delay=16`、523-token prefix 上运行，
warmup=2、timed repeats=8。prefill 只计 durable/noise/filler prefix；decode
计 query prompt 加最多两个 greedy answer tokens。结果文件：

`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_color_m13/m13_inference_benchmark_s17.json`

| 条件 | 参数量 (M) | prefix state (B) | prefill tok/s | decode tok/s | peak allocated (MiB) |
|---|---:|---:|---:|---:|---:|
| FA-task | 31.855 | 4,885,106 | 19,674 | 189.2 | 250.9 |
| SWA-task | 31.855 | 1,186,376 | 12,522 | 139.8 | 247.4 |
| Memformer | 33.476 | 294,928 | 2,376 | 177.5 | 256.1 |
| Fixed-LRU | 31.855 | 1,242,516 | 8,444 | 87.8 | 250.4 |
| Gated Memory | 31.855 | 1,242,514 | 8,094 | 84.7 | 251.0 |
| Value-token Gated | 32.160 | 1,242,514 | 7,976 | 83.5 | 252.4 |
| Fixed-LRU + alignment | 32.160 | 1,242,516 | 8,148 | 84.4 | 251.9 |

state bytes 是 batch=1、prefix 完成后的持久状态，不是参数或 CUDA allocator
显存；FA 的增长来自完整历史 KV，SWA 的 local KV 固定，Memformer 只有 latent
slots，fact-memory 条件还包含语义/词法/metadata slot。M11 表中的
`2,423,702 B` 来自 FP32 state 的旧 eval helper；M13 按 CUDA BF16 推理协议，
所以对应 fact-memory state 约为 `1,242,5xx B`，这是 dtype 差异而不是状态结构
变化。alignment 仅增加约 0.305M 参数，不改变 slot state bytes。该 benchmark
是当前 Python/PyTorch 实现的端到端测量，不能直接当作理论复杂度结论。

### Milestone 14：value-token alignment 配对消融（seed=17）

状态：完成。两个 run 均使用新 Full-Attention TinyStories parent、SWA(128,4)、
同一随机颜色数据、GPT-2 tokenizer、`answer_leading_space=true`、delay=`1,4,16`、
noise=0、batch=2、gradient accumulation=2、AdamW 和 2,000 steps；无 NaN/OOM，
均保存 final checkpoint、resolved config、metrics 和 summary。

训练曲线：

- [M14 value-token alignment accuracy/objective](/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_color_m14/m14_alignment_training_accuracy_objective_curves.png)
- [M14 paired objective curves, Gated baseline vs alignment](/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_color_m14/m14_gated_vs_value_token_gated_training_curves.png)
- [M14 paired objective curves, Fixed-LRU baseline vs alignment](/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_color_m14/m14_fixed_lru_vs_alignment_training_curves.png)

Native M11 baseline logs predate per-step answer-token accuracy logging, so the paired
baseline figures use their recorded objective; the M14 alignment figure uses the
recorded candidate memory-token accuracy. Final held-out answer EM/NLL below is the
authoritative answer-generation metric.

| 条件 | delay=1 EM | delay=4 EM | delay=16 EM | overall EM | answer NLL | value-span exact | memory-token exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gated Memory（M11） | 95.83% | 54.17% | 45.83% | 65.28% | 0.5961 | N/A | N/A |
| Value-token Gated（M14） | 100% | 100% | 100% | 100% | 0.000408 | 100% | 100% |
| Fixed-LRU（M11） | 95.83% | 58.33% | 58.33% | 70.83% | 0.4113 | N/A | N/A |
| Fixed-LRU + alignment（M14） | 100% | 100% | 100% | 100% | 0.000276 | 100% | 100% |

paired delta（alignment 减去同架构 baseline）：

- Value-token Gated − Gated Memory：`+34.72` 个百分点 EM，answer NLL
  `-0.5957`；target survival/read hit 从 `100%/100%` 保持为 `100%/100%`，
  因此提升主要发生在 retrieved lexical value 到普通 LM head 的利用。
- Fixed-LRU + alignment − Fixed-LRU：`+29.17` 个百分点 EM，answer NLL
  `-0.4110`；target survival/read hit 从 `100%/100%` 保持为 `100%/100%`。

两个 alignment run 的 state bytes 都是 `2,423,702 B`（8 个预分配 slots），
noise=0 平均 active slots 都是 1；alignment 没有改变 memory 容量，只改变了
词级 value 读出训练信号。答案仍由普通 decoder LM head 生成，alignment decoder
不参与最终 greedy answer。

### Milestone 15：alignment noise stress（seed=17，eval-only）

状态：完成。脚本
`adaptive_gated_fact_memory/scripts/evaluate_color_m15.py` 不更新参数，结果文件：

`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_color_m15/m15_alignment_noise_stress_s17.json`

下表为每个 noise 桶中 delay=`1,4,16` 三格的平均；每格 24 个 held-out examples。

| 条件 | noise=0 EM / read hit | noise=2 EM / read hit | noise=4 EM / read hit | noise=8 EM / read hit |
|---|---:|---:|---:|---:|
| Value-token Gated | 100.00% / 100.00% | 43.06% / 100.00% | 15.28% / 68.06% | 16.67% / 48.61% |
| Fixed-LRU + alignment | 100.00% / 100.00% | 20.83% / 100.00% | 5.56% / 62.50% | 11.11% / 6.94% |

两种 alignment 方法在所有 stress 桶的 `value-span exact` 都是 100%。
Value-token Gated 的 `memory-token exact` 在 noise=`0,2,4,8` 仍为 100%，说明
它保存的 lexical value 仍可被 alignment head 重建；答案下降来自 noisy query
下的读出/融合。Fixed-LRU + alignment 在 noise=8 的 memory-token exact 与
target survival 一起降至平均 `40.28%`，表明固定替换策略先丢失目标 slot。
因此 M15 不应把高噪声 EM 下降解释为 alignment 失效，而应分别报告 survival、
read hit 和普通 LM head 的答案利用。
