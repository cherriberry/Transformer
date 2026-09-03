# 高效 Transformer 三人协作统一实验协议

> 文档用途：三人分工前共同讨论、确认并冻结实验口径。  
> 推荐版本：`protocol_v1`  
> 适用对象：Longformer、Performer、Memformer，以及后续完整混合模型和其他 xxFormer 架构。  
> 核心原则：每个人独立完成自己的架构实验，但所有影响公平比较、结果解释和复现的条件必须统一。

## 1. 为什么需要共同协议

三个人分别跑不同架构时，一次运行可以同时产生质量、速度和显存数据。因此，不能把“质量、速度、显存”拆成三个人的工作；每个人都要对自己的架构完成一套完整实验。

真正需要共同讨论的是实验的“尺子”：

- 什么任务算有效；
- 不同架构是否在相同条件下比较；
- 什么叫 latency、显存节省和质量损失；
- 哪些结果可以放进同一张表；
- 失败、OOM 和不适用情况如何记录；
- 什么证据才足以支持最终结论。

协议冻结后，三个人可以使用不同代码、不同 GPU 和不同架构超参数，但不能在实验进行中各自改变上述尺子。

## 2. 需要三人共同讨论并冻结的事项

以下事项属于强制统一项。建议在阶段 0 的一次会议中逐项确认，并写入 `shared_contract/PROTOCOL.md`、`result_schema.json` 和 `metric_definitions.md`。

### 2.1 研究问题和结论边界

必须先确定项目要回答的问题，而不是实验结束后再根据最好看的数字改写问题。

建议冻结为：

1. 不同高效注意力结构在相同任务和模型规模下的质量—效率权衡是什么？
2. 局部精确注意力、当前 segment 全局近似和跨 segment memory 各自解决哪类依赖？
3. 完整混合模型是否比单分支或固定策略更有价值？
4. 优势是来自算法结构、显存占用、GPU kernel，还是仅来自实现差异？

同时明确不能预先承诺“无条件更快”“质量不损失”“线性复杂度”或“学术上首次”。这些表述必须等实验和文献核查后再决定。

### 2.2 任务语义和 mask 规则

这是最容易造成无效比较的地方，必须全员确认。

推荐标准：

- 语言建模统一采用 causal attention，位置 `t` 不能读取 `t+1` 及之后的 token；
- 编码或分类任务可以采用 bidirectional attention，但必须单独分表，不和 causal LM 结果混合；
- 所有方法都使用同一套 padding、attention mask 和 segment 边界定义；
- 长序列切成多个 segment 时，必须明确当前 segment 能看到什么、历史 segment 通过什么状态传递；
- 每种实现都必须通过 future-invariance 测试：改变未来 token 不应改变 causal 位置之前的输出；
- 如果某方法无法合法支持 causal 语义，只能标记为 bidirectional/fidelity 或实现级结果，不能放入 causal LM 主表。

会议必须最终确认：主任务是 causal 还是 bidirectional、segment 是否跨 batch、memory 是否在样本间 reset，以及是否允许跨 segment 的梯度。

### 2.3 数据集、划分和预处理

推荐标准：

- 主质量任务统一使用一个明确版本的 WikiText 数据集；会议中必须写清是 WikiText-2、WikiText-103 还是其他版本；
- train、validation、test 严格分离；test 只在最终配置冻结后使用；
- tokenizer、词表、特殊 token、最大长度、截断、padding 和 packing 方式统一；
- 评估 token 数固定，使用 token-weighted NLL，而不是按 batch loss 的简单平均；
- 固定评估样本顺序和数据加载规则；
- 所有数据预处理生成版本号或 hash，避免三个人使用不同缓存；
- 机制任务可以不同，但必须写明其目的，不得把不同任务的分数直接合并成总排名。

建议设置两层任务：

| 层次 | 统一要求 | 用途 |
|---|---|---|
| 公共主任务 | 三个人使用相同数据、tokenizer、评估集和指标 | 进行基本横向比较 |
| 机制任务 | 根据架构特点分别设计局部、当前 segment 远程、跨 segment memory 任务 | 证明对应机制是否真的发挥作用 |

### 2.4 模型主干和参数规模

推荐统一：

- embedding、位置编码、层数、hidden size、heads、head dimension、FFN、LayerNorm 和输出头；
- 模型 dtype、激活函数和 dropout 设置；
- 模型输入输出 shape、权重命名和 state dict 结构；
- 训练和评估时的 batch 组织方式。

建议同时保留两种比较模式，并在报告中分表：

1. **相同主干模式**：所有方法使用相同 hidden size、层数和输出头，突出 attention 结构差异；
2. **参数量匹配模式**：调整 FFN 或其他非 attention 部分，使总参数量尽量接近，建议差异不超过 1%。

每条正式结果必须自动记录总参数量、可训练参数量、attention 参数量和 checkpoint 大小。

### 2.5 初始化、训练和调参预算

必须共同确认：

- 公共参数是否从同一个基础 `state_dict` 复制；
- 无法共享的投影、随机特征或 memory 参数如何初始化；
- optimizer 类型，建议统一 AdamW；
- 初始学习率、warmup、schedule、weight decay、dropout 和 gradient clipping；
- batch size、gradient accumulation、训练步数或训练 token 数；
- 混合精度类型，如 FP16 或 BF16；
- checkpoint 保存间隔、选择最佳 checkpoint 的规则；
- 是否允许 early stopping；
- 每个方法可使用的调参配置数量和总 GPU 预算。

推荐正式质量实验使用 seeds `17、29、43`。先用短 pilot 筛选配置，再对候选配置运行全部 seeds。不能让一个方法进行大量调参，另一个方法只跑默认配置后直接比较。

### 2.6 基线的层级和用途

三人必须统一 baseline 名称和使用场景：

| 基线 | 用途 | 是否用于最终速度主表 |
|---|---|---|
| Standard reference | 数值正确性、输出误差和语义校验 | 通常不作为最快工程实现 |
| PyTorch SDPA/Flash | 同机工程性能比较 | 是 |
| FullKV | 自回归生成的完整 KV cache 参考 | 只用于生成/KV cache 表 |
| 无 memory / 完整历史 | Memformer 或 MemoryState 的机制对照 | 视任务放入对应表 |

每个人必须在自己的 GPU、自己的 runner 中重新运行相关 baseline。不能把人员 A 的 SDPA 延迟直接借给人员 B 使用。

### 2.7 正确性和数值校验

全员统一最小正确性测试：

- 输入输出 shape 和 dtype 正确；
- padding mask 正确；
- causal future-invariance 通过；
- 可反向传播，梯度无 NaN；
- 短序列与 Standard reference 的误差在预先约定阈值内；
- 100% cache 或等价配置与 FullKV 对齐（适用于 Keyformer 等生成方法）；
- segment 和 batch 重排后结果不串样本；
- MemoryState 在新样本开始时正确 reset；
- AMP、不同长度和非整齐长度不产生未说明的错误。

数值 fidelity 只能在 attention 语义、权重、输入和 mask 一致时计算。若两个方法已经训练成不同模型，不能把输出差异简单称为 fidelity 误差。

### 2.8 Benchmark 输入和执行模式

会议必须确定正式矩阵，推荐：

- sequence length：`128、512、1K、2K、4K、8K、16K`，显存不足时记录 OOM，不随意删掉；
- batch size：至少 `1` 和一个代表性训练 batch；
- dtype：FP16 或 BF16，必要时增加代表性的 FP32 数值校验；
- 模式：forward、training step；生成方法增加 prefill 和 decode；
- 输入：使用固定输入集合，同时至少包含多种长度和 padding 比例；
- GPU：正式比较时同一张卡上必须运行 baseline 和方法，测试期间不运行其他 GPU 任务。

正式计时建议采用：

1. 预热至少 20 次；
2. 正式记录至少 50 次 timed samples；
3. 使用 CUDA event 或同步计时；
4. 随机化不同方法的运行顺序；
5. 保存每一次原始 latency，而不只保存平均值；
6. 报告 median、mean、p10/p90、标准差和 95% CI；
7. 明确计时是否包含 token 选择、gather、KV eviction、memory update 和数据搬运。

不同 GPU 型号之间不能直接比较绝对毫秒数。若租用相同型号 GPU，才可以进行跨人员绝对 latency 比较；否则统一报告相对各自同机 baseline 的 speedup 和 memory ratio。

### 2.9 指标定义

统一定义如下：

**质量指标**

- token-weighted NLL；
- PPL，由完整评估集 NLL 计算；
- 分类或检索任务 accuracy/F1；
- 仅在语义一致时报告 relative L2、cosine、logits KL、top-k overlap 等 fidelity 指标。

**效率指标**

- median latency；
- tokens/s；
- training step time；
- prefill latency 和 decode latency；
- 相对同机 baseline 的 speedup。

**显存和状态指标**

- peak allocated memory；
- peak reserved memory；
- OOM 的最大可运行长度；
- KV cache bytes；
- MemoryState bytes；
- 实际 attention 或状态的元素数量。

所有指标必须写出单位和计算公式。无法适用的字段写 `N/A`，不能用 0 代替。

### 2.10 随机性和统计规则

三人需要共同确认：

- 正式训练 seeds；
- Performer random feature、Reformer hash、Keyformer 随机选择等内部随机源是否单独固定；
- latency 的重复次数和异常值处理；
- 质量结果报告均值、标准差或置信区间；
- speedup 使用 ratio-of-means、median ratio 还是 paired speedup；
- 是否使用 bootstrap，以及 bootstrap 的样本单位。

推荐保留所有原始样本。不能只报告最有利的 seed、最快的一次 latency 或最小显存记录。

### 2.11 软件、硬件和实际 backend

每条正式结果必须记录：

- GPU 型号、显存和数量；
- driver、CUDA、PyTorch、Python、Triton、xFormers 版本；
- 操作系统和 commit；
- dtype、TF32 设置和确定性设置；
- 请求使用的 backend；
- 实际执行的 backend，以及发生回退时的原因；
- 是否使用编译、缓存或预热后的 kernel。

“请求 Flash”不能自动写成“使用 Flash”。必须从框架日志、profile 或 backend 诊断中确认实际路径。

### 2.12 结果 schema、目录和追溯

建议每个 run 至少包含：

```json
{
  "run_id": "project_method_task_seed_length_timestamp",
  "protocol_version": "protocol_v1",
  "method": "Performer",
  "task": "wikitext_causal_lm",
  "seed": 17,
  "config_hash": "...",
  "git_commit": "...",
  "gpu": "...",
  "dtype": "bf16",
  "requested_backend": "sdpa",
  "actual_backend": "flash",
  "status": "ok",
  "metrics": {},
  "raw_samples_path": "..."
}
```

目录建议为：

```text
independent_projects/
  shared_contract/
    PROTOCOL.md
    result_schema.json
    metric_definitions.md
  project_a/results/
  project_b/results/
  project_c/results/
```

三个人可以复制一份已经冻结的 baseline adapter，但不能共同修改一个仍在频繁变化的万能 runner。每个人的结果写入自己的目录；图表只能由原始结构化结果自动生成。

### 2.13 失败、OOM 和结果剔除规则

共同约定：

- OOM 记录发生的长度、batch、dtype、backend 和显存状态；
- NaN、非法 mask、梯度爆炸和 backend 回退不能静默重试后删除；
- 硬件或环境故障可以重跑，但必须保留失败记录和重跑原因；
- 只有预先定义的无效运行才可以从主统计中剔除，并在报告中列出；
- 不把某方法没有实现的模式伪装成 0 或成功结果；
- 失败配置和边界案例必须进入各自小报告。

### 2.14 报告、图表和结论口径

三个人的报告必须使用相同的基本结构：

1. 研究问题和假设；
2. 方法与数据流图；
3. 实验设置和基线；
4. 正确性结果；
5. 质量结果；
6. 速度、吞吐和显存结果；
7. 至少一个关键消融；
8. 一个失败或适用边界案例；
9. 优点、缺点和结论；
10. 原始结果位置和复现命令。

统一图表规则：横轴、单位、颜色、方法名称、OOM 标记、误差条和图例含义保持一致。结论必须写成“在本协议、任务、长度、硬件和质量约束下……”，不能宣布无条件总冠军。

## 3. 每个人可以独立决定的事项

以下事项不必由三个人设成相同数值，但必须在自己的配置和报告中公开：

- Longformer 的 window size、global token 数量和位置；
- Performer 的 random feature 数、特征类型和 redraw interval；
- Memformer 的 memory slots、chunk size、detach/BPTT 和 update 方式；
- Linformer 的 rank；
- Reformer 的 bucket size、hash 次数和 hash seed；
- Keyformer 的 cache ratio、recent ratio、score 和 budget 分配；
- 方法专属的 kernel 优化、缓存策略和实现语言；
- 机制任务的具体输入构造，但要说明该任务验证的能力；
- 在统一调参预算内选择的候选配置。

个人可以改变“架构旋钮”，不能改变“实验尺子”。例如，Performer 可以选择 64 或 256 个 random features，但不能因为某个 features 数效果不好就改 tokenizer、训练 token 数或 PPL 计算方式。

## 3.1 论文参数复现与统一公平比较必须分开

本协议推荐的 TinyLM 配置不是所有原论文的原始参数，因此不能把它直接写成“复现原论文”。它的用途是让三个人在一套相同主干、相同数据和相同训练预算下比较注意力机制。

推荐在报告中明确区分三种实验标签：

| 标签 | 参数原则 | 可以支持的结论 |
|---|---|---|
| `paper_aligned` | 尽量使用原论文的模型规模、位置编码、任务、数据、训练设置和方法超参数 | 是否大致重现该论文报告的现象；通常只能在相同任务和实现条件下比较 |
| `matched_tinylm` | 统一 tokenizer、embedding、层数、hidden size、heads、FFN、LayerNorm、位置编码、训练预算和评估指标 | 在受控小模型条件下，不同方法的质量—速度—显存权衡 |
| `innovation_validation` | 使用与本项目 B 模块和 A/C 接口相容的配置，并加入分支消融和机制任务 | 局部、全局近似、跨 segment memory 及动态控制是否各有贡献 |

三种标签不能混用。例如，Longformer 原论文常使用 BERT/RoBERTa 风格的大模型和 bidirectional 任务；把它改成 4 层、hidden size 256、causal LM 后，仍然可以作为受控 Longformer 实验，但不能声称“按 Longformer 原论文配置复现”。同理，Performer、Reformer 和 Memformer 的论文实验也分别使用不同任务、模型、位置编码、序列长度和训练设置，并不存在一套适用于所有方法的共同“原论文参数”。xFormers/SDPA 更是 kernel/backend 对照，本身没有一套独立的论文模型参数。

### 推荐的参数使用方式

在本项目只有一周的情况下，建议：

1. 主表使用 `matched_tinylm`：4 层、hidden size 256、8 heads、FFN 1024、context 512/1024，seeds=17/29/43；
2. 每种方法保留其核心架构旋钮，例如 Longformer window、Performer feature 数和 Memformer slots；
3. 每个负责人额外做一个低成本 `paper_aligned` sanity check，只核对关键默认值和语义，不要求完整重现所有原论文训练；
4. 最终完整混合模型使用 `innovation_validation` 标签，不能和原论文复现实验混在同一张主表。

报告中建议使用如下表述：

> 本实验采用统一 TinyLM 主干进行受控比较，并保留各方法的核心结构参数；该设置用于比较机制和系统代价，不等同于对各原论文完整训练配方的复现。论文参数对齐结果作为补充 sanity check 单独报告。

如果确实要宣称“复现原论文结果”，必须另外记录原论文的任务、数据集、tokenizer、模型规模、位置编码、优化器、训练 token、序列长度、batch、评估指标和实现版本，并按方法分别建立 `paper_aligned` 配置。不同论文之间不能因为都叫 xxFormer 就强行使用同一套原始参数。

## 4. 推荐的三人会议流程

### 第一次会议：冻结研究范围

共同决定：

- 三人各自负责的架构；
- 主研究问题和不支持的结论；
- causal/bidirectional 分组；
- 公共主任务和机制任务；
- 统一模型尺寸和参数比较模式。

### 第二次会议：冻结实验口径

共同决定：

- tokenizer、数据 split 和预处理版本；
- seeds、训练预算和调参预算；
- Standard、SDPA/Flash、FullKV 的用途；
- sequence length、batch、dtype、warmup、repeats；
- PPL、latency、显存、OOM 和 fidelity 公式。

### 第三次会议：冻结交付接口

共同决定：

- 结果 JSON schema；
- 目录和文件所有权；
- 报告和图表模板；
- 阶段闸门和失败记录规则；
- `protocol_v1` 的版本号和冻结日期。

三次会议可以合并为一次半天会议，但必须留下书面记录，而不是只在聊天中口头约定。

## 5. 阶段闸门

进入正式实验前，三个人都必须确认：

- 协议文件已经提交并带版本号；
- 主任务、mask、tokenizer 和数据 split 已固定；
- 自己的 baseline 可以运行并通过正确性测试；
- 结果 schema 校验通过；
- pilot 没有未解释的 NaN 或数据泄漏；
- GPU 时间和显存预算可接受；
- 每个人都知道自己的目录、提交格式和截止时间。

阶段结束时，每个人独立提交自己的结果和报告，不要求三个人共同编辑一个实时变化的结果文件。

## 6. 协议变更规则

协议冻结后，如果必须修改，遵守以下规则：

1. 任何人提出修改都要说明原因、影响范围和需要重跑的实验；
2. 三人确认后升级版本，例如 `protocol_v2`；
3. 旧结果保留，不覆盖、不自动混入新版本统计；
4. 受影响的 baseline 和方法必须使用新协议重跑；
5. 最终报告按 protocol version 分组，不能把不同口径的数字放在同一主表。

## 7. 三人签字确认清单

在正式实验开始前，每个人都应逐项确认：

| 确认项 | 人员 A | 人员 B | 人员 C |
|---|---:|---:|---:|
| 已阅读并同意 `protocol_v1` | [ ] | [ ] | [ ] |
| 了解 causal/bidirectional 分组 | [ ] | [ ] | [ ] |
| 使用相同 tokenizer 和数据 split | [ ] | [ ] | [ ] |
| 使用统一模型和参数报告方式 | [ ] | [ ] | [ ] |
| 使用统一 seeds、训练和调参预算 | [ ] | [ ] | [ ] |
| 已在自己的 GPU 重跑 baseline | [ ] | [ ] | [ ] |
| 正确性测试通过 | [ ] | [ ] | [ ] |
| latency、显存和 OOM 定义一致 | [ ] | [ ] | [ ] |
| 结果包含 run_id、配置 hash 和环境信息 | [ ] | [ ] | [ ] |
| 承诺保留失败结果和原始 timed samples | [ ] | [ ] | [ ] |
| 报告使用统一模板和结论边界 | [ ] | [ ] | [ ] |

## 8. 最终推荐

三个人真正需要共同讨论并冻结的是：

> 任务语义、数据和 tokenizer、模型主干、参数与初始化、训练预算、基线层级、正确性标准、benchmark 输入和计时、质量/效率/显存指标、随机性统计、软件硬件记录、结果 schema、失败处理和报告结论口径。

三个人不需要统一的是：

> 各自架构的内部超参数、实现细节、专属消融和机制任务设计。

这样分工后，每个人仍然是一个完整的独立实验负责人，同时三个人的结果又能在明确边界内进行可信比较。
