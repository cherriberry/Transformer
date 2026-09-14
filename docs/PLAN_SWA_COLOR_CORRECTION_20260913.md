# 随机颜色长期记忆实验计划（修订版）

更新时间：2026-09-14
当前范围：seed=17；暂不扩展 seed=29/43
本版核心变更：不为 Memformer 增加 reconstruction；同时补充显式 fact-memory
架构内部的 value-token alignment 配对消融。

## 1. 本次修订的结论

Memformer 的主实验只使用它原生的 recurrent latent memory 和普通
next-token/answer language-model loss。计划中不加入：

- Memformer value-token reconstruction head；
- 从 Memformer memory state 直接预测颜色 token 的辅助 CE；
- value-span 标签、pointer/copy head；
- 用辅助 head 的输出替代普通 decoder LM head。

这样测到的是：

> 在没有事实字段标签、没有词级重建辅助的条件下，Memformer 能否仅靠训练任务本身，
> 把当前样本的随机 name -> color 映射压缩到跨 segment 状态，并在查询时使用它。

Value-token Memory 的既有结果仍保留为历史诊断，但不进入本版
Memformer 主比较表。它使用了额外的 value-span 和 token alignment 监督，不能
与原生 Memformer 的结果作单一架构结论。

## 2. 已完成的证据（不重跑）

### 2.1 SWA 纠错和 in-window 检查

Full Attention 与 SWA(128,4) 在事实仍位于局部窗口内时，1000 steps 都达到
24/24 = 100% EM。因此早期 SWA-only=5.21% 不是普遍的训练失败，而主要是
随机映射超出 SWA recent window 后没有长期状态。

### 2.2 out-of-window 结果

旧的统一颜色协议（seed=17、noise=0）已经得到：

| 条件 | delay=1 | delay=4 | delay=16 | overall |
|---|---:|---:|---:|---:|
| Full Attention | 95.83% | 91.67% | 83.33% | 90.28% |
| SWA(128,4) | 95.83% | 12.50% | 4.17% | 37.50% |
| Fixed-LRU | 79.17% | 12.50% | 29.17% | 40.28% |
| Gated Memory（旧 vanilla） | 95.83% | 29.17% | 33.33% | 52.78% |
| Value-token Memory（额外 alignment） | 100% | 100% | 100% | 100% |

这张表只作为历史 screening 证据。新 Memformer 实验必须使用新的 Full-Attention
TinyStories parent 和本版冻结的训练协议，不能把旧 parent、旧 loss 和新结果混在一起。

### 2.3 新的 Full-Attention TinyStories parent

共同 parent：

/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_tinystories/tinystories_full_parent_s17_step6_20260913/checkpoint.final.pt

配置为 TinyStories、Full causal attention、10M training tokens、context=512、
6 layers/hidden=384/8 heads/FFN=1536。验证 token accuracy 为 41.9609%，
NLL=2.9463，PPL=19.0348。它是新颜色任务的共同初始化，不是颜色任务结果。

该 parent 上的 training-free SWA 结果已经记录在 training_free_swa_results.json，
不需要因为本版去掉 reconstruction 而重做。

## 3. 新实验要回答的问题

### Q1：Memformer 是否能记住随机事实？

在每个样本的颜色映射都重新随机生成、样本之间严格 reset 的情况下，比较
不同 delay 的答案 EM。若 Memformer 在 delay=16 仍显著高于 1/16=6.25% 的
随机命中率，说明 recurrent state 至少保留了部分样本特定信息。

### Q2：它是否真的使用了 recurrent state？

对同一个已训练 checkpoint 做状态干预：

1. 正常传递 state；
2. 查询前将 state 清零；
3. 将 batch 内其他样本的 state 交换过来；
4. 每个 segment 都 reset state。

记录 EM、first-token accuracy、answer NLL 的变化。正常状态相对清零/交换状态
的性能下降，是“模型使用了跨 segment 状态”的证据；这不需要训练任何
reconstruction head。

### Q3：Gated Memory 的优势来自架构还是额外监督？

分别报告：

- 共同主任务 loss：FA-task、SWA-task 和 Memformer 使用相同的 causal LM + answer
  loss；
- 原始 memory 分组：Fixed-LRU 和 Gated Memory 保留各自必要的候选/读取或事实
  结构监督，但明确记录该额外监督。

不把两种条件合并成一个排名。

## 4. 实验分组

### 4.1 正式实验分组

| 原始分组名称 | 注意力/状态 | 训练目标/监督 | 用途 |
|---|---|---|---|
| FA-task | Full causal attention | 共同 causal LM + answer loss | 可访问完整历史的任务参考 |
| SWA-task | SWA(128,4) | 共同 causal LM + answer loss | 局部窗口参考 |
| Fixed-LRU | SWA + fixed LRU memory | 共同主 loss + Fixed-LRU 所需的候选/读取监督 | 固定记忆基线 |
| Gated Memory | SWA + adaptive gated fact memory | 共同主 loss + 原有 fact start/length、write、read、key、retention、budget loss | 自适应事实记忆 |
| Memformer | recurrent latent memory slots | 共同 causal LM + answer loss；不加事实标签或 reconstruction | 无事实标签的循环记忆基线 |

五个名称中，FA-task、SWA-task 和 Memformer 构成严格的共同目标对照；Fixed-LRU
和 Gated Memory 是你原来分组中的 memory 条件，并在报告中单独列出额外结构监督。

### 4.2 监督口径

Fixed-LRU 和 Gated Memory 的结构辅助项不能伪装成与 Memformer 相同的 loss。每个
run 的配置必须记录主任务 loss、结构辅助 loss、权重、是否使用 hard/soft threshold
以及 answer loss。最终报告同时给出共同主任务指标和完整 native-training 指标，
不把两种口径合并为单一架构排名。

### 4.3 明确排除的条件

以下条件不在本版主实验中训练：

- Memformer + memory-to-token reconstruction；
- Memformer + value-token alignment；
- Memformer + pointer/copy；
- 将 Value-token Memory 的 100% EM 与 Memformer 直接作公平优劣结论。

已有 Value-token Memory 结果只作为“Gated Memory 的词级读出瓶颈已被额外监督
修复”的独立历史消融。

## 5. 共同数据、parent 和训练协议

所有新 run 固定：

| 项目 | 设置 |
|---|---|
| seed | 17 |
| parent | 上述 Full-Attention TinyStories parent |
| tokenizer | GPT-2，vocab=50,257 |
| 答案格式 | answer_leading_space=true |
| 数据 | 每样本独立随机 name -> color |
| segment length | 32 tokens |
| train delays | 1、4、16 segments |
| eval delays | 1、4、16 segments |
| train noise | 第一轮 0；主实验通过后再做 0、2、4、8 |
| eval noise | 第一轮 0；stress 阶段 0、2、4、8 |
| physical batch | 2 |
| gradient accumulation | 2 |
| optimizer | AdamW，betas=(0.9,0.95)，weight decay=0.1 |
| steps | 2,000；所有 matched-loss 方法相同 |
| precision | CUDA BF16，TF32 关闭 |
| validation | 每个 delay/noise cell 24 个 held-out examples |

共同主损失固定为：

L_common = L_full_causal_lm + lambda_answer * L_answer

其中 lambda_answer 必须在首个 run 前写入 config.resolved.json，之后所有
matched-loss 方法保持不变。结构辅助项只能在 native-training 组出现，并逐项
记录其权重。

## 6. Memformer 实现要求（无 reconstruction 版本）

当前历史 MemformerSegmentAttention 是可复用的 attention-level adapter，但
颜色任务需要一个真正 stateful 的 wrapper。实现必须满足：

1. 每层维护自己的 recurrent memory slots；
2. 一个 segment 结束后把新 state 传给下一个 segment；
3. query 只能读取已经完成的前序 segment state；
4. 当前 segment 产生的新 state 从下一个 segment 才生效；
5. 独立样本、packed example 和 batch reorder 时正确 reset/重排 state；
6. 不在 Memformer 中加入事实标签、value span 或 token reconstruction；
7. 训练时在一个样本内部按预先声明的规则处理跨 segment 梯度；
8. 记录 memory_slots、detach/TBPTT 规则、state bytes 和每段更新时间。

本实验中的名称统一写成 Memformer（decoder adapter）或
Memformer-style recurrent memory。它与原论文完整的 encoder-decoder MemBART
不是同一模型；若以后做论文级复现，应另开独立 protocol。

## 7. 按 milestone 执行

### M7：stateful adapter 和因果性闸门

只实现和测试，不开始长训练。

必须通过：

- state shape/device/dtype 正确；
- state reset 后与新样本等价；
- batch state reorder 后样本对应关系不变；
- future-token invariance；
- chunked segment 与逐段调用结果一致；
- query 看不到当前 segment 尚未写入的新 state；
- state swap/zero 干预确实改变后续输出；
- 与普通 LM head 的接口一致；Memformer 配置不实例化、不调用
  memory-to-token head。

产物：测试日志、adapter code hash、最小 state bytes 报告。

### M8：短程过拟合和 in-window sanity

执行状态（2026-09-13）：已完成。FA-task、SWA-task 和 Memformer 均在 24 个
固定训练映射上达到 100% answer EM，loss 明显下降且无 NaN/OOM；Memformer
必须在样本内保留 query loss 到 recurrent write 的梯度，详见实验日志中的
`m8_memformer_s17_v2`/`v3` 规则诊断。独立 held-out 映射在该短程固定样本预算
下为 0% EM，因此只作为泛化待办，不作为 M8 架构结论。正式泛化比较转入 M9。

使用 8–24 条固定结构样本以及独立随机映射的 held-out 样本，运行
FA-task、SWA-task、Memformer。先只用 noise=0。

接受条件：

- FA/SWA in-window 达到接近 100% EM；
- Memformer 不出现 NaN、state 不恒为零；
- 每个方法训练曲线在预算内明显下降；
- answer tokenization 与 GPT-2 token 数记录完整。

若 FA/SWA 失败，停止后续实验并先修协议；若只有 Memformer 失败，先检查
state 传递、mask、segment 边界和梯度规则，不添加 reconstruction。

### M9：seed=17、noise=0 的正式共同目标比较

运行：

FA-task / SWA-task / Memformer

每个方法 2,000 steps，保存 metrics、best checkpoint 和 final checkpoint。
中间 checkpoint 只保留必要节点，避免占满数据盘。

主指标：

- answer EM；
- first-token Top-1/Top-5；
- teacher-forced answer NLL；
- delay 曲线；
- 训练 loss/accuracy 曲线。

### M10：recurrent-state 因果诊断

对 M9 的 Memformer checkpoint（必要时也对 Gated）执行：

- normal state；
- query 前 zero state；
- cross-example swapped state；
- per-segment reset；
- 只保留最后一个 state slot；
- state noise/scale sweep。

报告 Delta EM、Delta NLL、first-token rank 和 state norm。该 milestone 不
训练任何 probe 或 reconstruction head；所有输出均来自原普通 LM head。

### M11：Fixed-LRU 与 Gated Memory 对照

在相同 parent、数据、seed、steps 和 eval 网格下运行 Fixed-LRU 与
Gated Memory。允许既有结构辅助损失，但在配置和报告中逐项列出：

- fact start/length；
- write/read/key alignment；
- retention/budget；
- 是否使用 hard/soft threshold；
- 是否使用 answer loss。

Value-token alignment 保持关闭。

### M12：noise 和容量 stress

只有 M9–M11 的实现闸门通过后才进行：

1. noise=0、2、4、8；
2. delay=1、4、16；
3. Memformer slots 进行预注册的小 sweep（例如 4、8、16、32）；
4. Gated Memory slots 保持 8，并另行记录其共享槽状态。

不把“相同 slot 数”称为“相同 memory budget”。同时报告实际 state bytes、
active slots 和每段更新次数。

### M14：value-token alignment 配对消融（当前执行阶段）

M14 不改变任务定义，也不重新使用旧 parent。四个条件必须从同一个新的
Full-Attention TinyStories parent、同一随机颜色数据、tokenizer、seed=17、
loss 主项、batch、优化器和 2,000 steps 开始：

| 原始分组名称 | memory policy | value-token alignment | 对应关系 |
|---|---|---:|---|
| Gated Memory | gated | false | M11 baseline |
| Value-token Gated | gated | true | Gated Memory 的配对改进 |
| Fixed-LRU | fixed_lru | false | M11 baseline |
| Fixed-LRU + alignment | fixed_lru | true | Fixed-LRU 的配对改进 |

Value-token alignment 的固定辅助权重为：value-span start `0.50`、length
`0.25`、candidate memory-to-token `0.50`、retrieved memory-to-token `0.50`。
答案仍由普通 decoder LM head 生成；alignment head 只提供训练/诊断信号，不能
用其输出替代模型答案。该 alignment 是显式 fact-memory 方法的合法配对消融，
不是给 FA-task、SWA-task 或 Memformer 强行添加事实标签。

M14 每个 run 必须保存 resolved config、代码/parent hash、metrics、final
checkpoint、summary，并生成 objective/answer-accuracy 曲线。报告以下配对差值：

- Value-token Gated − Gated Memory；
- Fixed-LRU + alignment − Fixed-LRU；
- answer EM、first-token Top-1/Top-5、answer NLL；
- target survival、Recall/read hit、value-span exact、memory-token exact；
- state bytes、active slots、训练末期与最佳 objective。

M14 的四个条件可以放在同一张完整 leaderboard 中，但解释分为两层：
FA-task/SWA-task/Memformer 是共同 causal-LM 目标对照；Fixed-LRU/Gated Memory
及其 alignment 版本是带 native fact-memory supervision 的系统对照。创新效果
以同一架构的 paired delta 为主证据，不把额外 alignment 监督伪装成 matched-loss
架构结论。

### M15：alignment stress 与最终报告

M14 两个新 checkpoint 通过训练充分性检查后，使用与 M12 完全相同的 held-out
网格（delay=`1,4,16`，noise=`0,2,4,8`，每格 24 条）做 eval-only noise
stress。随后补齐四种显式 memory 条件的训练曲线、吞吐/峰值显存/持久 state
bytes，并生成最终 leaderboard。除非用户另行要求，不启动 seed=29/43。

### M13：效率和最终 seed=17 报告（已完成）

为使最终效率表同时包含 M14 的两个 alignment checkpoint，本次执行顺序是先完成
M14/M15，再完成编号为 M13 的统一 benchmark；这不改变各 milestone 的训练协议。

在已选定 checkpoint 上测：

- prefill/decode throughput；
- 峰值 allocated/reserved memory；
- persistent state bytes；
- 参数量；
- 不同 delay 的 EM/NLL；
- 训练过程 accuracy 曲线。

执行结果记录在实验日志的 M13；统一 benchmark 使用 held-out
`noise=0, delay=16`、batch=1、warmup=2、repeats=8，并额外包含 M14 的两个
alignment checkpoint。

只有 M13 完成并确认没有训练不足后，才决定是否启动 seed=29/43。

## 8. 指标和解释规则

### 8.1 所有方法共有

- EM；
- first-token accuracy 和 rank；
- teacher-forced answer NLL；
- full causal LM loss；
- 训练/验证 accuracy 曲线；
- 参数量、训练步数、吞吐和显存。

### 8.2 只有显式 fact-memory 方法报告

- target survival；
- Recall@1/Recall@4/MRR；
- active slots；
- noise accept；
- memory state bytes。

Memformer 没有可对应的 fact slot，因此这些字段写 N/A，不能把
N/A 解释成“没有记忆”。

### 8.3 Memformer 的核心诊断

Memformer 不使用 reconstruction 时，最重要的证据链是：

normal state EM > zero/swapped state EM

并且 state delta、跨 segment influence 和 delay 曲线均为非零。若 normal、
zero 和 swapped 几乎相同，则不能声称它在该任务中使用了长期 memory，即使
语言模型整体 loss 正常。

## 9. 结果判定

in-window FA 高、SWA 高
→ 局部 attention 和答案 tokenization 正常

out-of-window FA 高、SWA 低
→ 纯 SWA 的可见性上限

Memformer normal 高、zero/swapped 明显下降
→ 原生 recurrent state 记住并使用了样本信息

Memformer normal 低、zero/swapped 无差异
→ 没有证据表明它学会了事实记忆；不加入 reconstruction 掩盖结果

Gated Memory 高、Memformer 低
→ 需要结合两者的监督项和 state 干预结果，不能只归因于架构

## 10. 报告组织

### 表 1：共同目标的架构对照

FA-task / SWA-task / Memformer

只比较共同 causal LM + answer loss，报告 EM、NLL、delay 曲线和训练充分性。

### 表 2：原生系统对照

Fixed-LRU / Gated Memory / Memformer

增加 supervision、state bytes、active slots 和检索诊断；明确这不是严格
matched-supervision 排名。

### 表 3：Value-token 配对消融

列出 Gated Memory ↔ Value-token Gated 和 Fixed-LRU ↔ Fixed-LRU + alignment
的 paired delta。旧 parent 上的 `color_corrected_value_token_outwindow_s17_m4`
只作为历史证据单独标注，不能替代 M14。

### 表 4：seed=17 noise=0 最终 leaderboard

| 原始分组名称 | delay=1 EM | delay=4 EM | delay=16 EM | overall EM | 监督口径 |
|---|---:|---:|---:|---:|---|
| FA-task | 95.83% | 87.50% | 4.17% | 62.50% | common LM + answer |
| SWA-task | 79.17% | 0% | 4.17% | 27.78% | common LM + answer |
| Memformer | 100% | 100% | 100% | 100% | native recurrent LM + answer |
| Fixed-LRU | 95.83% | 58.33% | 58.33% | 70.83% | native fact-memory supervision |
| Gated Memory | 95.83% | 54.17% | 45.83% | 65.28% | native fact-memory supervision |
| Value-token Gated | 100% | 100% | 100% | 100% | Gated + value-token alignment |
| Fixed-LRU + alignment | 100% | 100% | 100% | 100% | Fixed-LRU + value-token alignment |

该表用于完整呈现结果，不把不同监督口径解释成单一公平排名；创新点的主要
因果证据是同一 memory policy 的 paired delta。

## 11. 当前不做的事情

- 不重新训练 Memformer reconstruction；
- 不为 Memformer 增加 value-token alignment 或事实标签；
- 不把 frozen probe 或任何后处理分类器的准确率当作模型 EM；
- 不在 M9 前增加 noise、额外 seed 或大规模 slot sweep；
- 不把旧 SWA parent 的颜色结果与新 Full-Attention parent 的结果混表；
- 不把当前 decoder adapter 的结果写成完整 Memformer 论文复现。

本版执行顺序的核心是：先证明 Memformer state 是否被使用，再比较
Gated Memory 的 native supervision 价值；如果 Memformer 不能恢复随机
颜色，结论应如实记录为压缩状态/训练目标的限制，而不是通过 reconstruction
head 改变问题定义。

## 12. 执行状态（2026-09-14）

- M7：完成，stateful Memformer adapter 闸门 `35/35 passed`。
- M8：完成，三方法固定样本过拟合 sanity 通过。
- M9：完成，FA-task、SWA-task、Memformer 均完成 2,000 steps；共同目标结果
  已写入实验日志，训练曲线已生成。
- M10：完成，normal/zero/swapped/per-segment-reset/last-slot 以及 scale/noise
  状态干预已完成；normal 与 zero/swapped 的显著差异证明 recurrent state 被使用。
- M11：完成，在同一 parent、数据、seed、预算下运行原始分组的 Fixed-LRU 和
  Gated Memory；value-token alignment 保持关闭，native 结构监督已逐项记录。
- M12：完成 M9/M11 final checkpoint 的 noise=`0,2,4,8` eval-only stress；
- M14：完成，按完全相同协议重跑 Value-token Gated 与 Fixed-LRU + alignment；
  两者 noise=0 的三种 delay 均为 100% EM，paired delta 已写入实验日志。
- M15：完成，对两个新 checkpoint 做 noise=`0,2,4,8` stress；高噪声下的主要
  退化来自 memory survival/read，而非 value-token span 解码。
- M13：完成，已记录最终 checkpoint 的效率、显存、state bytes、参数量和训练
  曲线汇总；不增加 seed=29/43。
