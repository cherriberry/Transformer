# TinyLM + TinyStories 实验设置与问答记录

更新时间：2026-09-04

## 1. 当前决定

- 主模型：仓库自建的 `TinyLM-long-v1`，不是外部 TinyLM checkpoint。
- 模型类型：从头训练的 decoder-only causal language model。
- 主训练数据：`roneneldan/TinyStories`。
- 数据版本：`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`。
- tokenizer：`openai-community/gpt2`，只下载 tokenizer，不下载 GPT-2 权重。
- tokenizer 版本：`607a30d783dfa663caf39e06633721c8d4cfcd7e`。
- PG-19：不再用于新的主实验；旧实验和结果保留原标签，不回写成 TinyStories。
- 当前协议：`experiments/tinystories_tinylm_v1/protocol.yaml`。
- 三天版硬件：三人各使用一张 NVIDIA RTX 4090，三个 worker 并行执行。
- 公共 TinyLM 预期参数量：29,925,504（标准 LayerNorm affine）；Linear bias 关闭、输入/输出 embedding tied。

## 2. TinyLM 主干草案

```yaml
family: TinyLM-long-v1
architecture: decoder-only causal LM
layers: 6
hidden_size: 384
heads: 8
head_dim: 48
ffn_size: 1536
normalization: Pre-LN
position_encoding: RoPE
rope_max_position: 32768
dropout: 0.1
vocab_size: 50257
tokenizer: GPT-2 BPE
initialization: from scratch, matched across methods
seeds: [17, 29, 43]
```

除候选方法必需的 attention 或 memory 参数外，各模型应复制同一个基础 state dict，并使用相同数据顺序、训练 token 数、优化器和 seed。

## 3. TinyStories 数据说明

TinyStories 是英文短故事语料，适合低成本训练小型生成模型。当前下载官方 parquet 形式的 `train` 和 `validation` split；上游没有官方 `test` split。

这项替换会改变实验结论的边界：TinyStories 适合比较小模型的学习效率和短文本生成质量，但多数故事较短，不能天然证明 4K、8K 或 16K 的长距离建模能力。若把多个故事拼成一个长序列，得到的是计算长度，不等于同一文档内的长距离语义依赖。

## 4. 尚未冻结的问题

以下设置会实质影响实验结果，应在开始正式训练前逐项确定：

1. 训练上下文长度采用 256、512、1024，还是多长度课程；
2. 是否允许跨故事 packing，以及 attention/loss 是否跨越文档边界；
3. 在没有官方 test split 时，如何建立独立 validation/test；
4. pilot 和正式训练分别使用多少实际 token；
5. 各方法是严格匹配总参数量，还是只固定公共主干；
6. 主指标只使用 token-weighted causal NLL/PPL，还是加入生成质量评估；
7. 长上下文机制是否另用 synthetic passkey/copy/needle 任务验证。

在以上问题解决前，协议状态保持 `draft_pending_experiment_questions`，不能把结果称为冻结后的正式主实验。

## 5. 数据资产与复现

本地数据位于 `data/raw/tinystories/`，tokenizer 位于 `data/tokenizer/gpt2/`。大文件不提交 Git；`data/tinystories_manifest.json` 保存 split 行数、上游 commit、文件大小、SHA-256 和依赖版本。

复现命令：

```bash
python -m pip install -r requirements-tinylm.txt
HF_ENDPOINT=https://hf-mirror.com \
python experiments/tinystories_tinylm_v1/prepare_data.py
```

## 6. 问答记录

### Q1：采用 TinyLM 和 BERT 作为基座有区别吗？

有。这里的 TinyLM 是 causal decoder-only next-token LM；BERT 是 bidirectional encoder，通常用 MLM。二者的注意力可见范围、训练目标、位置编码、模型规模和评估指标不同，PPL 不能直接混表比较。

### Q2：主训练数据是否继续使用 PG-19？

不使用。新的实验线采用 TinyStories，PG-19 目录与历史结果只用于保留实验来源，不纳入新的 TinyStories 主实验。

### Q3：本实验的研究目的是什么？

当前工作定义（待研究者继续确认）：

> 第一阶段在统一 TinyLM、TinyStories、训练预算和评估协议下，系统比较不同 Transformer/X-former 架构设计带来的质量、效率、记忆能力和工程实现方面的优缺点。第二阶段把比较结果作为“门控记忆与线性注意力”研究的前期证据，分析现有架构为什么难以同时保留局部精确信息、利用远距离历史，并按输入需要控制计算和记忆预算，进而为新的混合记忆架构确定基线、组件选择和可证伪的改进目标。

这里暂不使用含义模糊的“长期记忆和短期记忆转换”，而将问题描述为：

> 现有架构缺少长期压缩记忆与短期局部上下文之间自适应的信息流转和动态调度机制。

该问题至少包含四个可分别测量的过程：

1. **写入/巩固（write/consolidation）**：判断哪些短期信息值得压缩并保存为长期状态；
2. **读取/检索（read/retrieval）**：判断当前 token 或 segment 何时需要调用更早历史；
3. **融合/更新（fusion/update）**：将检索出的长期信息与当前局部表示结合，并更新记忆状态；
4. **遗忘/替换（forget/eviction）**：在固定记忆容量下删除或覆盖价值较低的旧信息。

如果强调短期信息进入长期记忆，最准确的词是“巩固”；如果强调长期信息回到当前上下文，则是“检索”或“再激活”；若概括整个双向过程，建议使用“信息流转与动态调度”，而不是单独使用“转换”。

这一工作定义还没有冻结。需要继续确认最终研究贡献究竟以“现有架构比较”为主，还是以“提出新的门控混合记忆架构”为主；前者决定第一阶段本身就是核心成果，后者则意味着第一阶段主要承担基线、诊断和设计依据的作用。

### Q4：时间有限时，希望最终得到什么核心结论？

研究者确认，本项目不以在当前阶段完成整篇论文或穷尽所有实验为目标。期望到达的核心结论是：

> 我们设计并实现了一种自适应记忆 Transformer。该架构能够根据输入和当前状态，自主调节所使用的记忆容量与有效记忆跨度，并在短文本和长文本条件下兼顾输出质量与计算效率。

这里将原表述中的“记忆量于记忆长度”理解为“记忆容量与有效记忆跨度”，但后者的确切含义仍待确认：它可能指可读取的历史 token/segment 范围，也可能指一条信息在记忆中被保留的持续时间。

考虑到时间限制，应区分两种结论强度：

- **目标结论**：上述架构在长短文本上都形成优于固定记忆方案的质量—效率折中；
- **当前最低可交付结论**：完成可运行原型，并通过受控 TinyLM + TinyStories 实验初步证明门控会随输入改变记忆使用量，且相较匹配的固定记忆基线，在至少一种质量—效率指标上取得改善，同时明确尚未验证的适用范围。

“表现较好”不能只依赖一个 PPL 或一次速度测试。至少应拆为：

1. 输出质量：token-weighted causal NLL/PPL；条件允许时补充固定生成样例或任务准确率；
2. 输出效率：推理吞吐、延迟、峰值显存和实际启用的记忆比例；
3. 短文本与长文本：预先固定两档或多档长度，不在看完结果后重新定义；
4. 自主决策：记录每个样本/segment 的 read、write、forget gate 和有效记忆使用量，证明模型不是始终全开或全关。

该回答已作为当前研究方向记录；其中“有效记忆跨度”的操作定义、短/长文本边界以及“较好”的判定规则仍需继续确认。

### Q5：“全文历史”和“模型决定保留多久”是否会失去窗口式记忆的优点？

研究者当前意图是：历史信息的潜在来源覆盖全文，但模型自主决定一条信息在有限记忆中保留多久。这里的“覆盖全文”不应解释为每一步重新对全部历史 token 做精确注意力；否则计算和 KV cache 仍会随文本长度增长，窗口方法的效率优势也会消失。

建议采用双时间尺度结构：

1. **短期精确通道**：始终保留固定大小的 causal sliding window，精确处理最近 token；
2. **长期压缩通道**：每个 segment 结束时，从即将离开窗口的信息中选择少量内容写入固定上限的 memory bank；
3. **保留/遗忘门**：为记忆槽提供 retention/forget gate，决定已有信息继续保留、衰减或被替换；
4. **读取与融合门**：当前 segment 仅在需要时读取长期记忆，并控制长期表示对局部表示的影响；
5. **可选动态容量**：在固定最大槽数 `M_max` 内，模型决定实际激活多少槽，而不是动态创建无限状态。

因此，更严谨的性质是：

> 模型对全文历史具有递归的信息传递路径，但只精确保留局部窗口，并对更早历史进行有损、选择性的压缩。

不能把它表述成“能够无损访问全文”。被 gate 遗忘的信息无法再恢复；是否保留了任务所需信息必须通过不同延迟距离的检索任务验证。

推荐的最小可实现原型暂定为：

```text
当前 segment
  -> 局部滑动窗口精确注意力 -------------------+
  -> read gate -> 长期 memory bank -> fusion gate -> 输出
                  ^
                  |
        write gate + retention/forget gate
```

在时间受限时，第一版只实现“局部窗口 + recurrent memory + segment 级软门控”；FAVOR+/线性注意力分支作为后续扩展，不同时引入，以免无法判断收益来自哪一个组件。软 gate 可以验证模型是否学会调度记忆，但只有进一步实现硬 gate、active-slot packing 或跳过读取计算，才能声称获得实际速度/显存收益。

最低必要对照应包括：固定局部窗口、固定容量且始终读写的 memory、以及自适应门控 memory。TinyStories 用于自然语言 NLL/PPL；不同延迟的 Passkey/Associative Recall 用于验证信息能否在离开局部窗口后继续保留。短文本长度不超过窗口时，预期 memory gate 低开启；长文本且存在远距离依赖时，预期相关记忆保留时间和读取率上升。

### Q6：采用压缩记忆槽是否与 Performer 类似？

研究者选择长期通道保存**压缩记忆槽**，而不是选中的原始 token K/V。

它与 Performer 有一层抽象上的相似性：两者都把无限增长的 token 历史变成大小有界的中间状态，从而避免每一步对全部历史做二次复杂度的注意力。但二者压缩的对象、更新规则和研究目的不同。

| 维度 | Performer/FAVOR+ | 本项目压缩记忆槽方案 |
|---|---|---|
| 核心目的 | 用随机特征近似 softmax attention | 选择哪些跨 segment 信息长期保留 |
| 状态形式 | 特征映射后的累计统计量，例如 `sum(phi(K)V)` 与 `sum(phi(K))` | 可寻址、可更新的 `M_max x d` 记忆槽 |
| 写入方式 | 通常每个 token 按固定线性注意力公式累计 | learned write gate 选择性写入/覆盖 |
| 遗忘方式 | 基础 Performer 通常没有针对某条记忆的显式保留时长 | retention/forget gate 控制槽的保留、衰减和替换 |
| 读取方式 | 每个 query 通过核特征读取累计统计量 | read gate 决定是否读取，cross-attention/检索决定读取哪些槽 |
| 动态计算 | random-feature 维度通常固定 | 目标是在 `M_max` 内动态改变实际激活槽数并跳过无效计算 |
| 可解释诊断 | 分析随机特征数、近似误差和 feature seed | 分析槽使用率、保留时长、读写门和记忆内容 |

因此，可以把 causal Performer 的累计统计量看作一种**隐式快速权重记忆**，但不能把它等同于显式的门控压缩槽。项目中的 Performer 应作为重要基线：它回答“固定线性全历史汇总是否已经足够”；本项目方法则要回答“内容相关的选择性写入、保留和读取是否进一步改善质量—效率折中”。

同时需要降低创新性表述风险：压缩槽、循环记忆和读写门控本身都有大量既有工作。仅将它们组合起来不自动构成新贡献。当前更可能成立的贡献点是：

> 在精确局部窗口之外，联合学习记忆槽的实际使用量与内容保留时长，并将门控决策落实为可跳过的计算路径；验证该调度是否随输入长度和依赖需求变化，并优于固定窗口、固定槽 memory 与 Performer 基线的质量—效率折中。

第一版不加入 Performer 分支。先完成 local + gated memory；Performer 作为独立对照。如果基础原型成立且时间允许，再测试线性当前-segment 分支是否提供额外收益。

### Q7：自主记忆是否应当由内容重要性驱动，而不只是由文本长度驱动？

是。研究者确认：对于长度相同的两段文本，模型也应能够根据内容重要性和潜在的远距离依赖，使用不同数量的记忆槽，并让关键信息保留更久。由此，长度只能作为可用上下文和计算预算的一个输入因素，不能成为唯一的记忆决策规则。

当前正式目标更新为：

> 模型根据当前内容、已有记忆状态和潜在的后续依赖，自主决定哪些信息值得写入长期压缩记忆、使用多少记忆容量、何时读取以及何时遗忘；该决策在短文本和长文本上都应形成合理的质量—效率折中。

这意味着实验不能只报告“序列越长，启用槽越多”。还必须构造**相同长度、不同信息重要性**的样本，检查 gate 是否产生不同决策。例如，一段故事中后文会再次提到的人物、地点、目标或因果关系，应该比无关的修饰句更可能被写入并保留。

建议将自主决策拆成两个层次：

- **segment 级容量决策**：当前 segment 是否需要访问长期记忆，以及激活多少槽；
- **slot 级保留决策**：每个已有槽继续保留、衰减、合并还是替换。

第一版仍建议使用连续 soft gate，以保证梯度稳定；同时加入轻量的 memory-use budget penalty，防止模型无条件打开全部槽。最终只有当 gate 决策被编译为 active-slot 选择、跳过无效读取/写入时，才能把“自主减少计算”作为实测效率结论。否则只能称为自适应信息路由，而不是已经实现的动态计算节省。

因此，最低验证集合应包含：

1. 同长度、低依赖样本：记忆读取率和激活槽数应较低；
2. 同长度、强远距离依赖样本：相关槽的写入、读取和保留时长应上升；
3. 不同长度、相同依赖类型样本：区分长度带来的预算变化与内容带来的决策变化；
4. 固定窗口、固定槽 memory、内容门控 memory 三组对照。

这一点是当前研究目标的核心，而不是附加的可视化分析。

### Q8：长期记忆主要服务故事一致性，还是任意早期事实的延迟检索？

研究者优先选择第二种：模型应尽可能记住早期出现的事实，并在经过较长文本和干扰信息后被询问时准确检索。TinyStories 的自然语言建模仍用于检查基础生成质量，但长期记忆能力的主要证据应来自可控的延迟事实检索任务。

需要明确一个因果边界：如果问题直到文本末尾才出现，模型在读到早期事实时并不知道未来会询问哪一条。在容量有限时，任何模型都无法保证无损保存任意多的未知事实。因此，实验必须区分：

1. **指令引导的选择性记忆**：提前告诉模型要关注哪个对象或哪类事实，测试模型能否只写入相关信息；
2. **未知后续查询的关联记忆**：先出现多个 key-value 事实，最后才给 query，测试有限槽容量下能保存多少可检索事实；
3. **干扰和保留距离**：固定事实数量，增加事实与查询之间的无关内容和 segment 间隔，测试记忆保留时长；
4. **动态容量**：固定总文本长度，改变需要记忆的事实数量，检查实际激活槽数是否随记忆需求而不是随长度变化。

推荐的最小任务矩阵是：

| 控制变量 | 建议取值 | 主要回答的问题 |
|---|---|---|
| 需记事实数 | 1、2、4、8 | 模型是否自主增加有效记忆量 |
| 延迟距离 | 1、2、4、8 segments | 相关槽能保留多久 |
| 干扰事实数 | 0、4、16 | 是否会被无关内容覆盖 |
| query 时机 | 提前指定、末尾出现 | 内容选择与容量压缩的区别 |
| 总序列长度 | 配对固定 | 排除仅根据长度开关 gate 的解释 |

主要指标应使用 query answer 的 exact-match accuracy；同时记录 active slots、相关槽存活时间、read/write/forget gate、延迟和峰值显存。PPL 仍然报告，但不能代替检索准确率。

为了控制工作量，第一版可先完成“结构化 key-value 事实 + 自然语言干扰 + 末尾 query”，再增加提前指令条件。该任务可以与 TinyStories 混合：使用 TinyStories 片段作为干扰文本，而不是另做一个完全脱离自然语言的数据环境。

### Q9：现有的 X-former 理论上是否已经能实现这些功能？

**能实现其中很多功能，但当前比较集合中没有一种方法同时满足全部目标。** 必须区分“早期信息理论上能影响后续输出”和“模型显式、自适应地管理有限记忆，并因此获得实际资源收益”。前者很多架构都能做到，后者才是本项目希望验证的组合目标。

| 架构 | 早期信息可达 | 历史表示 | 内容相关选择 | 自主决定实际记忆量/保留时长 | 主要缺口 |
|---|---|---|---|---|---|
| Full Attention | 上下文窗口内直接可达 | 保存全部 token K/V | attention 权重按 query 变化 | 否 | 记忆和计算随长度增长，超出窗口后不可达 |
| Longformer | 经多层传播或 global token 可达 | 最近窗口 + 固定 global 通路 | attention 内容相关，但连接范围预设 | 通常否 | 窗口/global 位置通常固定，不负责学习写入和遗忘 |
| Performer | 通过 causal 累计统计量覆盖整个前缀 | 随机特征的固定维汇总 | query 对汇总的读取内容相关 | 基础版本否 | 所有 token 按固定公式累计，缺少可寻址槽和显式选择性遗忘 |
| Linformer | 在固定序列范围内通过低秩投影可达 | `k` 个序列投影分量 | learned projection，但容量通常固定 | 否 | 不是跨 segment 的持久状态，因果化和长度外推也更困难 |
| Reformer | 相关 token 哈希到相邻 bucket 时可达 | 通常仍保存 token，只稀疏读取 | LSH 是内容相关候选选择 | 否 | 不提供固定容量的跨段压缩记忆或 learned retention |
| Keyformer | 被选中的历史 KV 可直接读取 | 原始 token K/V 的子集 | 是，按重要性选择 | 部分；预算比例通常预设 | 主要是预训练生成模型的推理期 cache 淘汰，不是可训练压缩槽 |
| Memformer/MemBART | 通过 recurrent memory 跨 segment 可达 | 固定数量压缩槽 | 是，attention 更新且有 update gate | 部分；槽数固定且通常全部计算 | 最接近本方案，但没有证明动态 active-slot 容量和物理跳算 |

尤其需要正视 Memformer 的重叠。仓库中的官方 MemBART 实现已经：

- 建立固定 `memory_len x d_model` 的跨 segment 状态；
- 让 memory 与当前 hidden states 共同参加 attention；
- 通过 `memory_gate_net` 在新旧 memory 之间进行学习式插值更新。

因此，“压缩记忆槽 + 门控更新 + 跨段传播”本身属于已有思路。本项目若只做到这一层，应诚实表述为 Memformer 风格原型或扩展，而不是全新架构。

当前仍可能形成明确研究差异的目标是：

1. 固定最大容量 `M_max`，但按样本和 segment 内容决定实际 active slots；
2. 对不同槽学习不同的保留/替换决策，并测量事实的实际存活时长；
3. 局部精确窗口始终处理短期信息，长期槽只承担窗口外依赖；
4. 将 soft gate 转成 hard routing/active-slot packing，真实跳过未启用槽的读取和写入计算；
5. 在匹配总长度但记忆需求不同的输入上，证明资源分配由内容而非长度规则驱动；
6. 相对固定槽 Memformer、Performer 和固定窗口，在相同质量或相同资源预算下形成更好的折中。

即使实现上述目标，也不能在完成系统文献检索前声称“首次提出”。门控循环记忆、adaptive computation、动态 token/KV 选择等方向均已有相关研究。时间有限时，更稳妥的项目定位是：

> 构建并验证一种内容驱动的动态容量记忆原型，系统说明它与典型 X-former 基线的差异、有效条件和失败边界。

此外，“任意事实检索”应改成“容量约束下的关联事实检索”。Full Attention 在窗口内保留原文，是质量上界；所有固定容量压缩方法都可能发生信息碰撞或遗忘，无法保证无损保存无限多的未知事实。

### Q10：定位为 Memformer 的动态容量扩展后，还可能有哪些创新点？

研究者接受“对 Memformer 的内容驱动动态容量扩展”这一定位。除动态 active slots 外，还有以下候选贡献。它们是**待验证的研究点**，不是已经确认的首创；正式声称创新前仍需系统检索相关工作。

#### 候选 A：联合分配短期窗口与长期记忆预算（优先推荐）

常见方法分别固定 local window、memory slots 或 random features。本项目可以给定最大计算预算，让同一个控制器联合选择：

- 当前 segment 的精确局部窗口 `w_active`；
- 本次读取的长期槽数 `m_read`；
- 本次写入/保留的长期槽数 `m_keep`。

这样模型处理短文本或纯局部依赖时，可以减少长期记忆；遇到跨段依赖时，再把预算从局部计算分配给长期槽。贡献不再只是“动态槽数”，而是“局部精度与长期容量之间的内容驱动预算分配”。

可检验假设：在相同 FLOPs、延迟或显存约束下，联合调度优于固定窗口 + 固定槽；或者在相同准确率约束下，平均实际计算更低。

#### 候选 B：显式的记忆生存期/风险率

Memformer 的 update gate 能在新旧状态之间插值，但“信息保留多久”通常只是隐含结果。本项目可为每个槽维护 `age`、`last_access` 和 learned survival probability，并在每个 segment 决定继续存活、合并或淘汰。

这会把研究者提出的“自主决定保留多久”变成可观察量，而不只是事后查看一个 gate 值。评价包括事实存活 segment 数、不同内容类型的生存曲线，以及正确答案丢失前的淘汰行为。

风险：TTL、遗忘门和 learned eviction 都存在相关研究，创新点必须落在具体联合机制和实验证据上，不能声称“首次让记忆遗忘”。

#### 候选 C：反事实记忆需求蒸馏（推荐的训练方法）

仅用 next-token loss 时，门控容易全开或全关。训练阶段可以同时运行：

1. Full Attention teacher，代表可使用完整历史；
2. local-only 路径，代表没有长期记忆；
3. gated-memory student。

当 Full teacher 相比 local-only 明显降低目标 token 的损失时，将差异作为“这个位置确实需要历史”的软监督；student 同时蒸馏 teacher 的输出，并预测何时开启记忆。部署时只运行 student。

这比人工用文本长度监督 gate 更符合内容驱动目标，也能回答“模型如何知道现在需要长期记忆”。风险是训练成本增加，且相关蒸馏工作需要检索；第一版可以只在合成事实检索任务上使用该信号。

#### 候选 D：预算可控的单模型 Pareto 前沿

训练时随机提供目标预算 `b`，让同一个 checkpoint 在推理时接受“低/中/高记忆预算”，并在每个预算下自主选择槽与窗口。这样不需要为每种资源限制重新训练模型。

可检验假设：同一模型随预算增加获得单调的准确率提升，并覆盖多个固定配置模型的质量—效率折中。该方向实用性强，但训练稳定性和单调性约束会增加工作量。

#### 候选 E：配对控制的动态记忆评测协议

构造总长度完全相同、只有“需要保留的事实数量、延迟和干扰程度”不同的样本。若 gate 随记忆需求变化而不是只随长度变化，就能排除最简单的伪解释。该项更偏评测方法，不足以单独证明架构创新，但对核心主张非常重要且实现成本较低。

#### 时间受限时的推荐组合

不建议同时实现全部候选。最合理的最小组合是：

1. **核心机制**：动态 active slots + 显式 slot survival/eviction；
2. **架构扩展**：如果基础版本稳定，再加入局部窗口/长期槽联合预算；
3. **训练支持**：先在检索任务上使用反事实记忆需求信号；
4. **证据设计**：必须完成配对的同长度、不同记忆需求评测；
5. **工程结论**：soft gate 后必须实现 hard selection，才报告实际效率收益。

暂缓 FAVOR+ 分支、多级记忆、动态分段和预算条件化单模型。这些可以列为后续工作，避免有限时间内变量过多。

当前推荐的一句话贡献为：

> 在 Memformer 风格固定上限的压缩记忆中，引入内容驱动的槽激活与显式生存期决策，并与精确局部窗口共享计算预算；通过反事实记忆需求信号训练控制器，使模型在相同长度但不同记忆需求的输入上执行不同的记忆策略。

### Q11：最终希望长期记忆解决什么实际场景？

研究者进一步明确，目标场景是**长时间、多话题对话中的情景细节记忆**：用户先讨论话题 A，随后切换到话题 B 并持续较长时间；当用户再次提及 A 时，模型仍能回忆 A 中的具体细节，而不仅是大意。

这里的长期记忆特指当前对话流中的非参数化状态，不是预训练后写入模型权重的知识。目标能力应表述为：

> 模型能够从早期对话中选择性巩固话题相关细节，在无关话题和长间隔期间抑制干扰并持续保存；当后续对话重新出现相应的话题、实体或语义线索时，模型能够检索和重新激活正确记忆，从而生成事实一致的回答。

“细节记忆”至少包括人物/实体名称、偏好、数字、日期、地点、事件及其属性关系。它不同于摘要：摘要保留主题大意即可，而该任务要求恢复可核验的具体事实。

#### 对架构设计的影响

单纯的时间衰减或固定 TTL 不适合这一目标。话题 B 持续时间长，不代表话题 A 的信息已经失去价值。保留/淘汰应主要依据内容价值、唯一性、冲突、容量压力和未来使用概率，而不是只依据年龄。

推荐将压缩槽设计为主题可寻址的 key-value memory：

- `memory key`：表示话题、实体和检索线索；
- `memory value`：压缩保存对应的事实细节；
- `metadata`：记录置信度、写入时间、最近访问、访问次数和冲突状态；
- `write/consolidation gate`：决定一句话是否形成长期记忆；
- `top-k read gate`：根据当前话题只激活少量相关槽；
- `retention/eviction gate`：在容量不足时保护独特且重要的旧事实，合并冗余信息，淘汰低价值内容；
- `update gate`：用户纠正旧信息时更新对应槽，而不是同时保留互相矛盾的事实。

局部窗口继续保存最近若干轮对话的原始细节；只有即将离开窗口且被判断为有未来价值的信息才写入长期槽。这保留了窗口式注意力对最近上下文的精确建模优势。

压缩槽对姓名、数字和日期等精确字符串存在风险：连续向量可能保存语义却丢失字面值。第一版可以使用“每个槽对应一个原子事实”的 key-value 表示，限制每槽承担的事实数量；如果仍无法稳定恢复细节，再考虑给槽附加少量离散 token payload。后者属于混合记忆，不再是纯向量压缩，应单独消融和报告存储成本。

#### 训练与数据的影响

TinyStories 是单篇故事语料，不能单独训练或验证话题切换后的对话记忆。建议采用两阶段、低成本方案：

1. 用 TinyStories 训练 TinyLM 的基础语言建模能力；
2. 用可控的合成多轮对话训练和评估 memory/controller，其中包含 A -> 长 B -> 回到 A 的结构。

合成对话可将 A 中的事实写成自然语言，将 TinyStories 片段用作 B 的干扰文本，并在末尾以不同措辞重新询问 A。训练集和测试集必须使用不同实体、属性组合与模板，防止模型只记住固定句式。

#### 最小评测矩阵

| 变量 | 建议设置 |
|---|---|
| 话题数 | 1、2、4、8 |
| A-B 间隔 | 1、2、4、8、16 segments |
| 每个话题的细节数 | 1、2、4 |
| 细节类型 | 姓名、偏好、数字、日期、地点、事件 |
| 干扰类型 | 无关话题、相似实体、冲突事实 |
| 返回线索 | 原词、同义改写、仅实体提示 |
| 总长度控制 | 构造同长度、不同记忆需求的配对样本 |

主要指标是细节 exact match / token F1、冲突更新准确率和不同间隔下的遗忘曲线；同时记录写入率、active slots、top-k 检索命中率、槽存活时间、延迟和峰值显存。

#### 更新后的项目定位

> 构建一个面向多话题长对话的内容驱动动态情景记忆原型：近期内容由精确局部窗口处理，早期对话细节被选择性压缩到主题可寻址的有限记忆槽；模型在话题切换后按语义线索重新激活相关槽，并动态控制写入、读取、更新和淘汰成本。

这一场景比泛化的“任意全文事实”更具体，也更容易设计可证伪实验。但在给定有限容量下仍不能承诺记住所有细节，结论必须限定为训练分布、容量和测试间隔内的记忆准确率。

### Q12：记忆范围是同一次对话，还是跨会话持久记忆？

研究者确认：**只要求同一次连续对话内的记忆**。关闭对话或开始新会话后，不要求继续保留此前细节。

因此，本项目研究的是模型运行期间的 recurrent episodic state，而不是用户画像、外部向量数据库或写入模型参数的长期知识。状态生命周期定义为：

```text
conversation start -> 初始化空 memory
每个 turn/segment -> 读取、写入、更新或淘汰槽
conversation end -> 清空 memory
```

这一范围带来以下实现要求：

- 同一对话的 turn 必须按原顺序送入模型，并携带正确的 memory state；
- batch 重排时，memory 必须随对应 conversation ID 一起重排；
- 新对话必须显式 reset，不能把其他样本或用户的细节带入；
- 训练截断反向传播时可以 detach 梯度，但不能误清空前向记忆状态；
- checkpoint 恢复是否保存“进行中的对话状态”应单独定义；第一版可只恢复训练状态，不承诺恢复真实用户会话；
- 评估必须加入跨样本污染测试：在对话 A 写入一个独特事实后开启对话 B，B 不得检索到该事实。

该约束显著缩小了工作范围：不实现跨会话检索、持久化数据库、用户身份绑定、删除合规和隐私权限系统。

### Q13：话题切换由模型自主识别，还是由输入提供主题标签？

研究者确认：**模型应自主识别话题变化和话题返回，推理时不提供 topic ID 或显式主题边界标签。**

为控制第一版复杂度，不建议先训练一个独立的大型话题分类器。推荐让每个对话 turn/segment 产生一个语义 key，并复用 memory 寻址完成隐式话题识别：

1. 对当前内容池化得到 `current_key`；
2. 将 `current_key` 与已有 `memory_key` 做相似度匹配；
3. 高相似度表示旧话题或旧实体重新出现，读取并更新相应槽；
4. 低相似度表示新话题，申请空槽或触发合并/淘汰；
5. read/write/retention gate 同时参考当前表示、匹配相似度和槽 metadata。

该方案中的“话题”不是固定分类标签，而是模型学习出的语义簇。一个自然语言话题可能对应多个事实槽，一个槽也必须尽量只保存一个可核验的原子事实，避免主题级摘要覆盖具体细节。

合成数据生成器可以保存真实 `topic_id`、`entity_id` 和事实关系，但这些字段只能用于：

- 计算话题切换/返回检测指标；
- 判断检索出的槽是否属于正确话题；
- 分析错误来自未写入、被覆盖还是读取失败；
- 构造 oracle-topic 上界。

这些标签不得拼接到模型输入。主要实验使用无标签输入；oracle-topic 只能作为诊断上界，不能当作本方法结果。

自主识别至少需要三类测试：

1. **词面返回**：使用与话题 A 相同的关键词重新提及；
2. **改写返回**：不复用原关键词，通过同义或释义线索返回 A；
3. **相似话题干扰**：A 与 B 共享部分实体或词汇，验证模型不会读取错误槽。

建议额外报告 topic-return retrieval Recall@k、错误槽读取率和新槽误分配率。仅在相同关键词条件成功，不能充分证明模型识别了语义话题。

### Q14：每轮对话结束后更新记忆有什么缺点？

如果“对话结束”指**每个完整 turn 结束**，这是当前场景下合理的默认更新边界；如果指整场 conversation 结束，则记忆无法在当前会话中发挥作用，不符合目标。

按 turn 更新的优点是边界自然、不会切断一句话、更新次数远少于逐 token 写入，并且易于区分用户和助手来源。不过存在以下问题：

1. **超长单轮的信息可能来不及写入。** 如果一轮输入超过局部窗口，开头细节可能在轮末提交前已经离开窗口；因此超过 `max_segment_tokens` 的 turn 仍需按完整句子或 token 上限建立中间检查点。
2. **短轮次会造成过度更新。** “好的”“继续”等内容没有长期价值，若每轮无条件执行全量 memory attention，会增加延迟并污染槽；write gate 应允许整轮跳过提交。
3. **一个 turn 不一定只有一个话题。** 用户可能在一条消息中同时谈 A 和 B，不能强制把整轮压入同一槽；需要在轮内产生多个候选原子事实，再在轮末分别寻址。
4. **事实可能尚未表达完整。** 逐句立即永久写入可能产生碎片和重复；轮末应执行合并、去重和冲突检查。
5. **助手输出可能形成错误反馈。** 若模型把自己的幻觉当作事实重新写入，错误会在后续不断强化；memory 必须保存来源角色和置信度，不能无条件把 assistant 内容当成用户事实。
6. **可变长度不利于批处理。** 不同 turn 长度使训练 batch 和状态更新时机不一致，需要 attention mask、conversation ID 和独立 reset/reorder 管理。
7. **训练时容易发生未来泄漏。** 当前 turn 形成的持久记忆只能影响其后的 turn，不能反过来影响当前 turn 中更早 token 的预测。

推荐采用两阶段的 turn-level consolidation：

```text
turn 内：局部 causal attention + 临时候选缓冲区
turn 末：抽取原子事实 -> 匹配已有槽 -> 合并/更新/分配/淘汰
下一 turn：按当前语义 top-k 读取已提交的长期槽
```

对超长 turn，使用“完整句子优先、固定 token 上限兜底”的子段处理；子段只更新临时缓冲区或执行安全提交。这样保留自然轮次边界，同时避免开头内容在轮末前丢失。

第一版推荐：

- 默认在每个完整 turn 末调用 memory controller；
- controller 可以输出 `no-op`，并真实跳过写入计算；
- 每个 turn 可以产生 0 到多个事实候选；
- user、assistant、system 内容携带不同的 source metadata；
- 用户明确纠正旧事实时，优先执行 update/conflict resolution；
- 单独比较“每 turn 更新”和“固定 token 更新”，但不把两者同时做成复杂动态策略。

### Q15：是否确定在每一轮完整对话结束后提交记忆？

可以，研究者确认采用**完整轮次结束后提交**。在本协议中，一轮明确指：

```text
一轮 = 一条用户输入 + 随后生成完成的一条助手回答
```

时序固定为：

```text
第 t 轮开始：读取截至第 t-1 轮已经提交的长期记忆
处理用户输入：使用已有长期记忆 + 当前局部上下文
生成助手回答：继续使用相同记忆和本轮局部上下文
第 t 轮结束：汇总本轮候选事实，执行一次提交
第 t+1 轮：可以读取第 t 轮新提交的内容
```

这样不会产生未来信息泄漏：第 t 轮结束时形成的记忆不能反向改变第 t 轮已经生成的 token，只能服务后续轮次。本轮用户刚提供的信息不需要先进入长期槽才能回答，因为它仍在局部窗口内。

轮末提交不是把整轮压成一个槽，而是先从用户和助手内容中形成 0 到多个原子事实候选，再分别执行去重、来源标记、话题寻址、冲突更新以及槽分配。没有长期价值时允许整轮 `no-op`。

超长用户输入或助手回答仍需轮内临时缓冲/安全检查点，否则开头可能在整轮结束前离开局部窗口；这些检查点不改变正式的“轮末一次提交”语义。

### Q16：Memformer 会保存助手自己提出的内容吗？

**架构本身不区分事实来源；是否保存取决于输入数据流。** 在仓库附带的 Memformer 对话微调示例中，`DialogDataAgent` 将双方每个 turn 的 token 都追加到 `history`。当前助手 turn 先作为生成目标，完成后也被加入 history，因此在预测后续助手 turn 时，之前的助手回复会进入 encoder，并可能写入 recurrent memory。

Memformer 的 memory update 只接收 hidden/memory states，通过 attention 和 `memory_gate_net` 更新固定槽。它没有显式的以下机制：

- user/assistant 来源 metadata；
- “用户事实比助手推测更可信”的硬规则；
- 原子事实抽取和事实级写入；
- 对助手幻觉禁止写入；
- 用户确认后才提高助手内容置信度。

角色 token 出现在输入中，因此模型理论上可能从数据中隐式学会区别说话者，但这不是可靠的来源控制，也无法直接审计某个槽保存了谁的陈述。

这正是本项目可以与固定 Memformer 基线形成的一个机制差异：在轮末 consolidation 时保留 `source_role` 与 `confidence`，让写入、冲突解决和淘汰策略参考来源。公平比较时应保留两条路径：

1. **原始 Memformer 基线**：双方历史都进入 memory，不增加显式来源策略；
2. **本项目方法**：使用相同的对话文本，但在事实级记忆中显式记录来源并执行确定的可信度规则。

当前推荐但尚待研究者确认的规则是：用户陈述和用户明确纠正可作为高优先级情景事实；助手提出的计划、承诺或阶段结论可低置信度保存；助手对外部世界的推测或未经用户确认的细节不自动提升为事实。无论是否写入，助手原文仍可在局部窗口内正常参与后续短期对话。

### Q17：是否记录助手内容；可以为此使用固定记忆槽吗？

研究者确认助手相关内容也应记录；如果将其纳入动态事实记忆不便，可以使用固定记忆槽。第一版先按以下四类实现，不继续增加助手记忆类别。

推荐采用**来源分区的双记忆池**，避免助手生成内容与用户提供的细节竞争同一容量：

1. **用户情景记忆池**：固定最大容量、动态 active slots，保存用户提供或明确确认的原子事实，是本项目主要研究对象；
2. **助手状态记忆池**：少量固定语义槽，保存助手在当前对话中的计划、承诺和双方已经确认的阶段结论，是工程辅助状态。

助手固定槽不应保存每一条普通回复。第一版建议使用按职责定义的槽，例如：

- `current_goal`：当前共同目标；
- `agreed_constraints`：双方已经确认的限制和设置；
- `pending_actions`：尚未完成的下一步；
- `assistant_commitments`：助手明确承诺后续要执行或遵守的事项。

轮末提交时，助手候选内容只能更新这些固定槽；未经确认的知识性推测和普通措辞不写入。每个槽保留 `source_role=assistant`、置信度与最后更新时间。用户明确修改或否定某项计划时，以用户指令更新对应槽。

该设计的优点是实现简单、状态可解释，并降低助手幻觉污染用户事实槽的风险；缺点是固定槽的类别由人工定义，不属于“自主决定记忆量”的核心创新。论文或报告中应把它标为来源控制/工程设计，不能拿它证明动态容量机制有效。

做质量和效率对比时，助手固定槽的参数量、状态 bytes 和读写成本必须计入本方法总成本。建议通过消融比较：不保存助手状态、固定助手槽、助手内容与用户内容共用动态槽。

## 7. 变更记录

| 日期 | 变更 | 状态 |
|---|---|---|
| 2026-09-04 | 建立 TinyLM + TinyStories 独立实验协议 | 已完成 |
| 2026-09-04 | 固定 TinyStories 与 GPT-2 tokenizer 上游版本 | 已完成 |
| 2026-09-04 | 下载数据、tokenizer 并生成校验清单 | 已完成 |
| 2026-09-04 | 冻结 packing、split、context 和 token budget | 待讨论 |
| 2026-09-04 | 记录实验目的及“转换”的工作定义 | 待研究者确认 |
| 2026-09-04 | 记录时间约束下的目标结论与最低可交付结论 | 已确认方向，指标待冻结 |
| 2026-09-04 | 明确全文历史、保留时长与局部窗口可以并存 | 已形成推荐方案，待确认记忆载体 |
| 2026-09-04 | 选择压缩记忆槽，并明确其与 Performer 的差异 | 已确认 |
| 2026-09-04 | 确认记忆决策必须由内容重要性驱动 | 已确认核心目标 |
| 2026-09-04 | 将任意早期事实的延迟检索设为主要记忆目标 | 已确认，查询时机待确认 |
| 2026-09-04 | 分析现有 X-former 是否已覆盖目标能力 | 已完成；Memformer 是最接近基线 |
| 2026-09-04 | 接受 Memformer 动态容量扩展定位并整理候选创新 | 已确认定位，候选待选择 |
| 2026-09-04 | 将目标场景明确为话题切换后的对话细节记忆 | 已确认 |
| 2026-09-04 | 将记忆生命周期限定为同一次连续对话 | 已确认 |
| 2026-09-04 | 确认话题切换和返回由模型自主识别 | 已确认 |
| 2026-09-04 | 分析 turn-level 记忆更新及风险 | 推荐采用轮内暂存、轮末提交；来源策略待确认 |
| 2026-09-04 | 确认每个用户—助手完整轮次结束后提交记忆 | 已确认 |
| 2026-09-04 | 核对 Memformer 是否保存助手内容 | 已完成；原实现无显式来源控制 |
| 2026-09-04 | 确认记录助手内容并采用独立固定槽 | 第一版四类槽已冻结 |
| 2026-09-04 | 确认三人各有一张 RTX 4090，并固定三 GPU 并行执行方式 | 已确认；每人一张卡，数据只读共享、结果隔离 |

## 8. 后续研究设计访谈记录

### Q18：新框架的主要目标和机制重点是什么？

研究者进一步确认，最终希望优先实现的目标是：

> 在长时间、多话题对话中，模型能够在话题切换和较长干扰之后找回早期对话中的具体细节。

这对应目标 C（话题切换后的细节检索），但研究者同时指出：在有限保存时间和压缩存储条件下，要求模型无损找回所有细节可能并不现实。因此后续需要把目标明确为一种质量—保留时长—存储成本之间的折中，而不是无条件保证全文事实可恢复。

机制方面，研究者最初的设想是目标 A：模型自主决定何时写入、读取和遗忘记忆。目标 B（自主决定实际使用多少记忆槽）尚未形成明确方案，需要单独评估其可行性。目标 D（局部窗口与长期记忆之间的预算分配）暂不视为已确认创新点；Memformer 已经让当前表示和固定 memory 共同参与跨 segment 计算，但是否具备内容驱动的预算重分配、active-slot 选择和物理跳算能力，仍需通过实现和文献核对确认。

当前暂定的完整框架方向是：以 A 的读写/遗忘控制为核心，若 B 的动态容量机制可行则纳入，并将最终确认的机制组合成一个完整的对话情景记忆系统；各部分贡献必须通过独立消融区分。

### Q19：长期记忆中的保留内容是否由模型根据重要性自主决定？语义精度是否足够？

研究者确认：保留内容应由模型根据内容重要性自主决定；检索评价以语义精确为主，不要求所有内容逐字恢复。

该机制在技术上可行，但不能把它解释成模型能够预知未来并无损保存所有可能有用的信息。当前更准确的目标是：模型根据当前内容、已有记忆状态、槽容量、语义重复、冲突和潜在检索线索，为候选事实产生重要性分数，并自主决定是否写入、继续保留、读取、合并或淘汰。

训练和评测需要同时约束质量与记忆成本。仅使用普通 next-token causal loss 可能导致 gate 全开、全关，或无法学习未知未来查询下的长期价值。因此建议组合使用 TinyStories causal NLL、对话记忆检索损失、记忆预算/槽使用惩罚，并在必要时加入反事实记忆需求信号：比较有长期历史与仅局部历史时目标位置的损失差异，将其作为记忆需求的辅助监督。

“语义精确”应按事实类型定义可核验标准。姓名、数字、日期、地点等字段优先使用结构化解析或归一化后的 exact match；改写性回答可使用 token F1 或语义等价判定，但不能把主题相近、细节缺失的答案算作正确。主结果必须报告事实检索正确率，同时报告 active slots、槽存活时间、写入/读取率和存储成本，形成质量—保留—压缩效率折中。

### Q20：重要性和读写/保留门应如何训练？

研究者选择混合方案。

推理时不提供 topic ID、未来查询或显式的重要性标签；模型必须根据当前内容、已有 memory state 和语义匹配自主产生重要性分数以及 write/read/retention/eviction 决策。训练时允许使用合成对话生成器中已知的后续检索关系，以及有长期历史和仅局部历史时目标位置的损失差异，构造事实级或 segment 级的软辅助监督。

辅助标签不得拼接到模型输入，也不能作为正式测试时的决策条件。训练目标至少包含：TinyStories causal NLL、对话记忆检索损失、记忆预算/槽使用惩罚，以及可选的反事实 memory-need 蒸馏损失。正式测试必须使用未见过的实体、属性组合、改写方式和干扰结构，检验模型是否真正根据内容需求调度记忆，而不是记忆固定模板或依赖长度规则。

### Q21：长期记忆采用什么表示？

研究者选择原子事实 Key–Value 槽，而不是纯连续摘要向量。

长期记忆应尽量将对话内容拆成可核验的原子事实，每个槽原则上只保存一个事实。例如：

```text
key:   用户 — 出发日期
value: 9 月 18 日
```

槽还应保存来源角色、置信度、写入时间、最近访问、访问次数和冲突状态。key 用于话题、实体和属性的语义寻址；value 用于恢复事实内容。具体 value 是连续表示、可复制的原文 span、离散 token payload，还是它们的组合，仍需在下一步确定。

这一选择使本方案相对标准 Memformer 的差异更明确：Memformer 主要维护连续 hidden-state memory，而本方案还需要进行事实候选生成、语义寻址、重复合并、冲突更新和事实级淘汰。但 recurrent memory、固定上限槽和门控更新仍属于共同的结构基础，因此不能仅凭 Key–Value 表示宣称完全独立于 Memformer。

### Q22：原子事实由谁抽取？

研究者同意采用混合训练方案：训练和合成数据中可以使用结构化事实字段对事实候选、key/value、写入门和检索结果提供辅助监督；但推理时不使用外部事实抽取器、topic ID 或未来查询标签，模型必须从当前 turn 自主形成事实候选并决定是否写入。

因此，结构化字段只用于训练信号、错误诊断和评测，不属于部署时输入，也不能被用来替代模型的自主记忆决策。正式测试应使用未见过的实体、属性组合、表述模板和话题干扰，以验证模型是否学会了可迁移的事实抽取和寻址能力。

### Q23：事实 value 是否允许保留少量原始文本或 token payload？

研究者确认允许。为兼顾语义寻址和姓名、数字、日期等细节的恢复，长期槽采用“语义 key + 紧凑 value payload”的混合表示：key 和 value embedding 用于语义匹配与改写查询，payload 保留归一化事实文本或少量 token，避免纯连续向量压缩丢失可核验字面信息。

payload 的最大长度、每槽存储字节数、是否直接复制输入 span、是否使用归一化 token，以及 payload 解码成本，都必须纳入压缩效率和质量—存储折中统计。纯连续向量槽应作为消融项，以判断离散事实载荷是否是细节检索提升的主要来源。因此，本方案应明确称为“向量寻址的压缩事实记忆”，而不是纯向量 recurrent memory。

### Q24：第一版如何生成事实 payload？

研究者选择 A：第一版由模型自主判断是否保留事实，并通过指针/边界选择从当前输入中复制一个短原文 span 作为 value payload。外部抽取器不替模型做部署时的事实选择；结构化事实字段只可用于训练辅助监督和评测。

因此，第一版的压缩主要来自只保存少量高价值原子事实，而不是把每条事实进一步生成成更短的新文本。模型仍需自主完成候选事实识别、重要性判断、span 定位、槽寻址和后续读取。模型生成归一化 payload（B）以及原文 span 与生成 payload 的混合表示（C）暂列为后续扩展/消融方向，待基础机制稳定后再研究更高压缩率与可能的生成误差。

### Q25：动态 active slots 和真实跳过计算的实现复杂度如何？

需要区分三个层次。soft gate 只为每个槽赋予连续权重，实现较简单，但全部槽仍参加计算，不能证明实际加速。hard top-k/threshold routing 可以只选择部分槽，但离散决策带来梯度估计和训练稳定性问题。要在 GPU 上得到真实吞吐或显存收益，还需处理逐样本不同的 active-slot 数、变长 packing、gather/scatter、槽 metadata 重排和批内 padding；选择开销也可能抵消小规模 memory attention 节省的计算。

此外，逻辑上只激活部分槽不等于降低物理峰值存储：若仍预分配 `M_max`，显存不会自动下降。真正减少状态存储需要稀疏或变长布局。当前建议按阶段推进：先以 soft gate 验证内容驱动选择，再用 hard top-k 测量准确率—槽数折中，最后实现 active-slot packing 并单独验证真实效率收益。在第三阶段完成前，只能声称自适应信息选择和有效 payload 压缩，不能声称已经实现动态计算或峰值显存节省。

### Q26：是否接受动态容量机制的分阶段实现？

研究者接受分阶段安排：

1. 第一版必须完成 soft gate 和内容驱动的槽选择，并统计实际保存的事实数、payload token 数及相对原始历史的压缩率；
2. hard top-k/threshold routing 作为优先扩展，用于测量准确率—槽数折中；
3. active-slot packing、物理跳过未使用槽的计算以及真实吞吐/显存收益作为后续阶段，不阻塞第一版的记忆能力结论。

在第三阶段完成前，结果只能表述为自适应信息选择和事实 payload 压缩，不能声称已经实现动态计算节省或峰值显存降低。

### Q27：创新记忆实验与六种既有架构的比较如何组织？

研究者确认：除验证新框架外，还需要让其余六种 former 在统一 TinyLM + TinyStories 条件下进行对比研究。

因此最终实验应拆成两条关联但不可混淆的线：

1. **公共架构比较线**：所有方法使用同一 TinyLM 主干、TinyStories 数据、tokenizer、数据顺序、训练 token 预算、优化器、评估规则和随机种子，比较 causal NLL/PPL、吞吐、延迟、峰值显存以及适用长度。该线回答不同架构在统一语言建模条件下的质量—效率差异。
2. **创新记忆线**：在基础语言能力建立后，使用合成多轮对话记忆数据训练/微调新框架的 controller 和 Key–Value memory，并在 `A → 长 B → 回到 A`、事实冲突、不同延迟和自然语言改写上验证话题切换后的细节检索。TinyStories 片段可以作为自然干扰，但不能把只接受 TinyStories 训练的基线与接受额外记忆数据的新模型直接混入同一质量排名。

创新线可以为具有跨 segment 状态的基线提供机制对照，但必须明确标注训练数据和能力范围；所有比较都要保留公共 TinyStories 质量结果作为共同参照。

### Q28：Keyformer 的原始架构是什么？TinyStories 是否可以用于它？

Keyformer 原始上不是一种替换 Transformer 主干的训练架构，而是附着在已有 decoder-only causal LM 上的**推理期 KV-cache 压缩/淘汰算法**。论文目标是在不微调模型的情况下减少自回归生成阶段的 KV cache、内存带宽和 decode 延迟。

其基本流程是：prompt 经过普通 causal attention 后形成 K/V cache；在每层和每个 head 上根据 attention logits 计算 token 重要性，使用带温度的 Gumbel 扰动/softmax 得到选择分数，并跨 decoding 步骤累计；固定 cache 预算 `k`，保留最近窗口中的 `w` 个 token，再从更早历史中保留分数最高的 `k-w` 个 token；其余 K/V 被物理删除，同时保留原始绝对位置。后续生成只访问留下的 K/V。仓库中的 `models/keyformer.py` 是这一算法的受控实现，不是完整 GPT-J/MPT/Cerebras 集成。

原始 Keyformer 不提供跨 turn recurrent 压缩状态、语义原子事实 Key–Value 槽、事实级写入/读取/冲突更新或显式生存期管理。它保存的是原始 token K/V，重点是 token 级 cache 选择；本项目方案保存的是经过选择的原子事实及其 payload，重点是对话情景记忆。二者可以共享“内容相关选择”和“固定预算”这一抽象，但机制和目标不同。

TinyStories 完全可以用于 Keyformer，限制不在数据集。推荐做法是先用 TinyStories 从头训练统一的 TinyLM/Full-Attention checkpoint，再在生成阶段比较 FullKV、Keyformer、RecentWindow 和 Random+Recent 的质量、KV bytes、decode latency 和吞吐。由于单篇 TinyStory 通常较短，应另外构造较长 prompt、多故事拼接或多轮对话序列，否则难以观察 cache 压缩优势。

若将 Keyformer 改成训练期 soft mask 并从头训练，则应命名为 `Keyformer-style TinyLM` 或训练期 token-selection adapter，而不是原始 Keyformer；它可以作为探索性分支，但不能与原始推理期 Keyformer 的结果混称。

### Q29：六种框架比较中，Keyformer 应如何保持定义和公平性？（待确认）

六种比较对象已确认是 Longformer、Performer、Linformer、Reformer、Memformer 和 Keyformer，Standard/Full Attention 作为参考基线。

这里存在一个必须显式处理的协议分支：原始 Keyformer 是已有 decoder-only 模型上的推理期 KV-cache 选择策略，不是独立的训练期 attention backbone。若强行让它像其他方法一样在 TinyStories 上从头训练，就必须引入训练期 token-selection/soft mask，这会形成新的 `Keyformer-style` 变体，不能再称为原始 Keyformer。

当前推荐但尚待研究者确认的分层比较是：

1. Longformer、Performer、Linformer、Reformer、Memformer 使用统一 TinyLM 主干和 TinyStories 从头训练，进行训练/建模架构比较；
2. 原始 Keyformer 使用同一 TinyLM/Full-Attention checkpoint，在 TinyStories 的长 prompt、teacher-forced sequential evaluation 和生成阶段比较 FullKV、Keyformer、RecentWindow、Random+Recent 的质量—KV cache—延迟折中；
3. 若另做训练期 token selection，则单独命名为 `Keyformer-style TinyLM`，作为探索性消融，不与原始 Keyformer 结果混表。

该安排不代表 TinyStories 不适合 Keyformer；相反，TinyStories 用于训练统一 backbone 和构造长 prompt/多轮序列都可行，只是单篇故事偏短，需要通过固定规则的多故事 packing 或合成多轮序列显现 cache 压缩效果。

### Q30：是否能够比较六种架构设计在不同方面的优缺点？

可以，但不能用一个总分或一次 PPL/速度测试概括优缺点。六个方法优化的对象不同，必须按维度和适用范围比较，并把 Keyformer 的推理期性质单独标注。

建议的比较维度如下：

| 维度 | 核心问题 | 主要指标 |
|---|---|---|
| 基础语言质量 | 在统一 TinyLM + TinyStories 上学得好不好 | token-weighted causal NLL/PPL，按上下文长度分层 |
| 长距离依赖 | 早期信息经过干扰后还能否被利用 | passkey/copy/associative recall，多话题对话细节语义检索准确率 |
| 训练效率 | 达到同等质量需要多少训练成本 | tokens/s、step time、达到目标 NLL 所需训练 token 数 |
| 推理效率 | 运行时是否更快 | prefill latency、decode latency、tokens/s |
| 存储效率 | 长序列状态占多少资源 | peak VRAM、激活内存、KV bytes、MemoryState bytes、OOM 边界 |
| 质量—资源折中 | 节省资源后损失多少质量 | PPL/检索准确率对 latency、显存和状态 bytes 的 Pareto 曲线 |
| 稳定性和鲁棒性 | 换长度、seed、padding 或随机设置是否退化 | 多 seed 方差、长度外推、随机特征/哈希敏感性、失败边界 |
| 实现代价 | 理论复杂度是否真正转化为工程收益 | 实测 backend、kernel、索引/排序开销、物理节省与理论节省差异 |

六种方法的预期设计取舍是研究假设，不是预先写死的实验结论：Longformer 偏向局部依赖和可预测稀疏计算；Performer 偏向线性复杂度但承担随机特征近似误差；Linformer 依赖低秩假设及 rank/长度选择；Reformer 依赖 LSH 候选连接并承担碰撞和排序开销；Memformer 以固定 recurrent state 支持跨 segment 历史但存在压缩瓶颈；Keyformer 主要压缩生成阶段 KV cache，长 prompt 可能受益而短 prompt 可能被选择开销抵消；本项目方法重点验证内容驱动事实记忆和质量—存储折中。

最终结果建议拆成三张主表：

1. **统一 TinyLM + TinyStories 语言建模表**：基础质量、训练成本、prefill 和显存；原始 Keyformer 使用共享 Full-Attention checkpoint，并明确其不是独立训练 backbone。
2. **长距离/记忆能力表**：不同延迟、干扰和话题返回条件下的事实检索；新模型重点报告语义细节恢复。
3. **质量—资源 Pareto 表/图**：以 latency、peak VRAM 或状态 bytes 为资源轴，以 PPL 或记忆准确率为质量轴。

TinyStories 单篇故事偏短，不能单独充分揭示长距离差异；需用固定规则的多故事 packing 和独立的 `A → 长 B → 回到 A` 合成对话任务。合成记忆结果不能直接替代 TinyStories PPL。

建议将基础质量、长距离检索、推理/存储效率和质量—资源 Pareto 设为主结果；稳定性和实现代价作为辅助结果；dynamic gate、active slots 和 payload 压缩作为新模型的机制结果。不同性质的方法不应被强行合并成一个无意义的总排名。

### Q31：三天内六种框架如何分给三个人？（已澄清）

研究者澄清：六个框架全部执行，每个人各负责两个框架；不是三个人合计只做两个模型。六个框架为 Linformer、Performer、Longformer、Reformer、Memformer 和 Keyformer。

推荐按机制相近性和现有实现成熟度分组：

| 人员 | 负责框架 | 共同研究主题 | 最低交付 |
|---|---|---|---|
| A | Linformer + Performer | 低秩投影与随机特征近似：近似误差、容量参数和长度扩展 | 两个模型的 TinyLM/TinyStories pilot、正确性、质量、延迟、显存和至少一项容量消融 |
| B | Longformer + Reformer | 结构化稀疏连接：局部窗口与 LSH 候选的质量—效率取舍 | 两个模型的 causal mask/稀疏正确性、质量、长度—稀疏参数曲线、延迟、显存和 OOM 记录 |
| C | Memformer + Keyformer | 历史状态或 KV cache 压缩：固定 recurrent memory 与 token 级 cache 选择 | Memformer 的跨 segment state、Keyformer 的 FullKV 等价/预算测试，以及两者各自的质量和效率结果 |

该分组的理由是：A 的两个方法都用近似计算替代 dense attention，便于复用同一套 fidelity 和 feature/rank 消融；B 的两个方法都改变 token 连接图，便于统一测试 causal 稀疏 mask、长度扩展和不规则访问开销；C 的两个方法都围绕“有限状态保存历史”，但必须明确 Memformer 是训练/跨 segment memory，Keyformer 是推理期 KV-cache policy，不能把二者混成同一种架构或直接合并成单一排行榜。Keyformer 仍使用统一 TinyLM/Full-Attention checkpoint 做 cache 压缩评估，而不是强行改造成原始论文没有的训练 backbone。

每个人都要对自己负责的两个框架完成完整子实验，不能把某个模型的质量交给一人、速度交给另一人；每个人也必须在自己的运行环境中重跑 Standard/SDPA 或 FullKV 对照，不能借用他人的延迟数字。三人可以指定一名**协议协调人**负责合并 schema、检查配置 hash 和生成汇总图，但协调职责不替代其两个模型的实验责任。

三天排程建议为：

1. 第 1 天上午冻结 TinyLM/TinyStories 数据、context、token budget、seed、结果 schema；三人并行完成六个 adapter 的 shape、causal/future-invariance 和反向传播 smoke test。
2. 第 1 天下午至第 2 天完成最小 pilot 和容量筛选：Linformer `k`、Performer feature 数、Longformer window、Reformer bucket/hash 数、Memformer slots、Keyformer cache ratio。只保留预先约定的 2–3 个候选值，不在正式结果阶段反复调参。
3. 第 2 天晚至第 3 天运行冻结配置的最小正式矩阵：TinyStories token-weighted NLL/PPL、context `512/1024`（资源允许加 `2048`）、至少一个长程检索/跨 segment 任务、prefill/forward latency、显存/状态 bytes；最后统一生成方法卡片和质量—资源表。

三天内的最低结果不要求每个模型都有完整论文级多 seed 曲线，但必须保留 raw metrics、失败/OOM 记录和配置 hash。时间优先级为：正确性 > 一套可比质量结果 > 一套可比效率结果 > 第二个 seed/更长 context > 细致消融。正式报告应把 Keyformer 的推理缓存结果单独分表，把 Memformer/其他训练 backbone 的结果放在另一张表。

### Q32：三天内的训练结果是否至少要达到 TinyStories validation？

研究者表示最好能够达到验证集。当前将其落实为：六个框架都必须完成统一 TinyStories `train → validation` 流程，并至少产出一个可比较的 validation token-weighted causal NLL/PPL；不能只报告 attention fidelity、随机输入 smoke test 或单纯模块速度。这里暂不预设某个绝对 PPL 门槛，“达到验证集”暂按完成验证评估并获得相对稳定 checkpoint 理解，是否需要额外的收敛阈值仍待确认。

对于 Longformer、Performer、Linformer、Reformer 和 Memformer，使用统一 TinyLM 主干、相同训练 token budget、数据顺序、优化器和 seed 在 TinyStories train split 上训练，在固定 checkpoint 间隔计算 validation NLL/PPL，并按照预先声明的 validation 规则选择 checkpoint。另训练一个共享的 Full-Attention TinyLM reference checkpoint；Keyformer 原始版本不单独从头训练，而是使用该 checkpoint，在 TinyStories validation 的顺序评估或生成阶段比较 FullKV、Keyformer、RecentWindow 和 Random+Recent；其结果单独列为推理期 KV-cache 质量—效率实验。

三天内的优先级分层为：

1. **必须完成**：六个方法各有至少一个冻结配置、一个 seed、一次 validation NLL/PPL 和完整配置/失败记录；
2. **优先补充**：对关键方法增加第二个 seed、第二个 context 或 validation checkpoint 曲线；
3. **有余力再做**：三 seed、更多容量消融、长程检索和完整 Pareto 曲线。

validation 用于配置/ checkpoint 选择时，不能再把同一结果称为独立 test；三天内若没有额外独立 test split，应明确报告为 validation 结果，并把最终 test 留到协议冻结后的后续阶段。

### Q33：验证集结果的最低可接受标准是什么？

研究者确认采用以下最低标准：每个训练型框架必须完成至少一个统一 TinyStories training run，并在 validation split 上计算 token-weighted causal NLL/PPL；训练 loss 和固定 validation probe loss 总体应呈下降趋势，最终 validation loss 应明显低于随机初始化基线；必须保存 checkpoint、配置、原始曲线和环境信息。暂不设定跨架构统一的绝对 PPL 门槛，因为三天版的目标是得到可比较的 validation 证据，而不是声称所有模型已经充分收敛。

如果某个模型未收敛、OOM、NaN 或只完成部分 token，仍须保留并标记 `incomplete`、`oom` 或 `failed`，不能静默删除，也不能把不完整结果当作正常 validation 结果。Keyformer 原始版本不单独训练，按共享 Full-Attention checkpoint 的 validation/cache 协议记录。

### Q34：三天六框架验证集实验的具体设置、配置和时间预算是什么？

研究者要求据此形成可执行的三天方案。详细配置保存于 `experiments/tinystories_tinylm_v1/three_day_validation_screening.yaml`，人类可读计划保存于 `experiments/tinystories_tinylm_v1/THREE_DAY_EXECUTION_PLAN.md`。核心设置如下：

- 数据：固定 revision 的 TinyStories train（2,119,719 rows）和 validation（21,990 rows），GPT-2 tokenizer 固定 revision；每篇追加 EOS，按固定 parquet 顺序串接，切成无 padding 的 512-token blocks；train 上限 10M tokens；训练中使用固定 262,144-token validation probe，最终对最后 checkpoint 和 probe 最佳 checkpoint 完整扫描 validation stream（当前约 4,765,918 tokens）；无官方 test，因此只报告 validation。
- 公共 TinyLM：6 layers、hidden 384、8 heads、head dim 48、FFN 1536、Pre-LN、GELU、RoPE、dropout 0.1、tied embeddings、无 bias、causal，约 30M 公共参数。
- 训练：AdamW，lr `3e-4`，betas `(0.9,0.95)`，weight decay `0.1`，3% warmup，cosine 至 `3e-5`，gradient clip 1.0，BF16；context 512，effective batch 8192 tokens（推荐 micro-batch 4、accumulation 4）；每候选 pilot 1,048,576 tokens（约128 optimizer steps），主 run 10M tokens；必跑 seed17，可选 seed29。
- 主配置：Longformer left window 128（总窗口257）；Performer features128；Linformer rank128；Reformer bucket64、hash4；Memformer segment128、slots64；Keyformer cache ratio 50/75/100%，recent ratio0.5，在共享 Full checkpoint 上评估。
- 必测：train/validation 曲线、validation NLL/PPL、tokens/s、step time、prefill/decode latency、peak allocated/reserved memory、方法特有 state/cache bytes、参数量、OOM/失败边界以及 correctness 闸门。
- 分工：A=Linformer+Performer，B=Longformer+Reformer，C=Memformer+Keyformer；每人对两个模型完成完整闭环。
- 时间：数据缓存约20–90分钟；每个候选 pilot约15–60分钟，未融合稀疏实现可能1–2小时；五个训练型模型和一个共享 Full-Attention reference 的 10M main 通常各30–120分钟，未融合的 Longformer/Reformer 可能2–4小时；validation约10–40分钟；Keyformer sweep约20–60分钟。三张GPU时总墙钟约12–24小时；一张共享GPU时约24–48小时。单run预计超过6小时则降至5M tokens并记录偏差。

这是一份“时间受限 validation screening”配置，不是最终冻结的论文主实验。它优先保证六个框架都有可追溯 validation 结果；三 seed、完整参数曲线、长上下文和新模型的多话题记忆任务留到下一阶段。

补充执行风险：仓库已有六种方法的模块、历史微基准和 WikiText 脚本，但尚无直接符合这份协议的统一 TinyStories `train → validation` runner。第 1 天必须完成统一数据流、TinyLM wrapper、方法 adapter、checkpoint/validation probe 和结果 schema；历史结果不能直接冒充本轮 validation 结果。若第 1 天结束仍未形成统一 runner，应立即启用 5M-token fallback，并将缺失项标记为实现风险。

### Q35：三个人各有一张 RTX 4090 时，模型规模和执行方式是否需要调整？

研究者确认：三个人各有一张 RTX 4090。因此三名 worker 可以真正并行，不需要为了共享显存而缩小公共 TinyLM。若三张卡在同一台主机上，应通过不同的 `CUDA_VISIBLE_DEVICES` 隔离；若在不同主机上，每个 worker 使用本机的 `cuda:0` 即可。TinyStories/tokenizer cache 可以在准备完成后只读共享，但 checkpoint、日志和结果目录必须独立，避免相互覆盖。

当前公共主干的参数量约为 **29.93M**。在标准 LayerNorm affine（包含 weight 和 bias）、Linear bias 关闭、输入/输出 embedding tied 的假设下，精确值为 `29,925,504`；如果 LayerNorm bias 也关闭，则为 `29,920,512`。两者差异仅 4,992 个参数，正式 runner 必须用 `sum(p.numel() for p in model.parameters())` 记录实际值，并同时报告方法特有参数。

RTX 4090 的 24GB 显存足以运行当前 `context=512` 配置。推荐每个 worker 使用 BF16、`micro_batch=4`、`gradient_accumulation=4`（有效 batch 为 8,192 tokens）；若某个未融合实现产生 OOM，改为 `micro_batch=2`、`gradient_accumulation=8`，不改变有效 batch。参数权重本身只占约 57MiB BF16，显存主要消耗来自激活、词表 logits、attention 中间张量和方法特有状态。当前机器的单步校准显示，Full Attention 在 batch=4、context=512 时峰值约 1.6–1.7GiB；仓库现有未融合 Longformer 实现约 6.3GiB，仍低于 24GB，但正式结果应以统一 runner 的峰值显存为准。

三张卡并行后，三天版的总墙钟估计维持约 12–24 小时：A 负责 Linformer+Performer，B 负责 Longformer+Reformer，C 先训练共享 Full-Attention reference，再评估 Memformer+Keyformer。若某个单 run 在首轮 100-step calibration 后预计超过 6 小时，仍按既定规则降至 5M tokens并记录偏差；不通过删除模型或改变模型宽度来“适配”时间。
