# 多种高效 Transformer 架构与效率对比实验总结报告

> 审阅日期：2026-08-31  
> 审阅范围：本仓库的说明文档、模型/适配器代码、测试、JSON 原始结果和已生成图表  
> 重要说明：本报告区分“原始结果”“由结果支持的结论”和“尚待实验验证的架构推断”，避免把不同口径的实验直接排名。

## 1. 执行摘要

本项目围绕标准 Transformer 自注意力在长序列上的二次时间与空间开销，收集、实现或适配了七类高效方法：Linformer、Longformer、Reformer、Performer、Memformer、Keyformer，以及 xFormers/SDPA 内存高效注意力内核。项目已经具备以下成果：

1. 为 Memformer、Performer、Reformer 和标准注意力建立了统一的 PyTorch 注意力模块微基准；
2. 为 Longformer 编写了局部与全局 token 注意力实现，并提供了较完整的单元测试；
3. 为 Linformer、Longformer、Reformer、Performer、Memformer 和 xFormers 增加了 WikiText-2 上的注意力输出保真度或小模型训练实验；
4. 将 Keyformer 真正接入预训练 GPT-2 Medium，在 RTX 5060 Laptop GPU 上完成 KV cache、解码延迟和 WikiText-2 困惑度实验，并设置 FullKV、RecentWindow、Random+Recent 对照；
5. 保存了多数实验的 JSON、PNG、配置和复现脚本，具备继续统一实验的良好基础。

当前最可靠的实验结论来自 Keyformer：在 GPT-2 Medium、512-token 评估上下文中，保留 75% 和 50% KV cache 时，PPL 分别从 27.903 增至 28.111（+0.75%）和 28.697（+2.84%），而物理 KV 容量按预算近似线性下降；长 prompt 下可以出现解码加速，但短 prompt 下选 token 和 gather 的额外开销会使其变慢。该结论受到真实预训练模型、物理 cache 裁剪、FullKV 等价性测试和同预算基线的共同支持。

其余方法的现有结果更适合作为“机制验证”和“工程可运行性验证”，暂时不足以形成七种方法的统一优劣排名。主要原因是：实验来自至少两台不同 GPU；模型规模、精度、因果掩码、训练初始化、数据量和测量方式不统一；部分结果是随机初始化注意力模块或仅训练 120 步的 TinyLM；Linformer/Longformer/xFormers 的历史图没有配套结构化原始数据与完整环境元数据；部分内存图出现负值或异常跳变。

因此，本项目目前完成的是一个有价值的“高效注意力机制实验集合 + Keyformer 深度案例”，而不是完整、严格、同口径的多架构论文级复现。下一阶段最重要的工作不是继续增加零散模型，而是建立统一、可审计的性能—显存—质量三维实验矩阵。

## 2. 背景问题

### 2.1 标准注意力的瓶颈

对长度为 \(n\)、隐藏维度为 \(d\) 的序列，标准自注意力需要形成 \(n\times n\) 的注意力分数矩阵。忽略常数后：

- 时间复杂度约为 \(O(n^2d)\)；
- 注意力矩阵显存约为 \(O(n^2)\)；
- 自回归生成虽然每一步只产生一个 token，但需要保存各层历史 K/V，KV cache 随上下文长度线性增长；
- 当上下文扩展到数千或数万 token 时，训练显存、prefill、decode latency 和可支持 batch size 都会受到限制。

“高效 Transformer”并不是单一方案。不同方法压缩的是不同对象，也对应不同的误差与工程成本：

- 稀疏注意力只连接局部或少量全局位置；
- 低秩方法压缩 K/V 的序列维；
- 核方法用特征映射重排 softmax 注意力计算；
- 哈希方法先寻找可能相关的 token，再在桶内计算；
- 记忆方法用固定容量状态跨块传递历史；
- KV cache 方法只优化自回归推理时的历史缓存；
- 高效内核保持注意力数学定义不变，只改变 GPU 上的计算与内存访问方式。

这些方法的目标、适用阶段和误差来源不同，因此不能只用一个“速度”数字评价，也不能把 xFormers 内核优化与 Linformer 等架构近似视为完全同类的方法。

### 2.2 本项目覆盖的方法

| 方法 | 核心思想 | 主要优化阶段 | 典型复杂度或资源目标 | 是否改变注意力语义 |
|---|---|---|---|---|
| Standard | 全量 token 两两注意 | 训练/推理 | \(O(n^2)\) 注意力存储 | 否，基线 |
| Linformer | 将 K/V 的长度维投影到低秩 \(k\) | 训练/整段推理 | \(O(nk)\) | 是，低秩近似 |
| Longformer | 滑动窗口局部注意力 + 少量全局 token | 长文编码 | \(O(nw+ng)\) | 是，结构化稀疏 |
| Reformer | LSH 将相似 token 分桶，桶内注意力；论文还使用可逆层 | 长序列训练/推理 | 常表述为约 \(O(n\log n)\) | 是，哈希近似 |
| Performer | FAVOR+ 随机特征近似 softmax kernel | 训练/推理 | \(O(nr)\) | 是，核近似 |
| Memformer | 固定数量的外部记忆槽跨块传递 | 超长/流式序列 | 历史状态容量受控 | 是，信息瓶颈变为记忆槽 |
| Keyformer | 解码时保留重要历史 K/V 与近期窗口 | 自回归 decode | KV cache 与预算近似线性 | 是，丢弃历史 K/V |
| xFormers/SDPA | 融合、分块或 Flash 风格精确内核 | 训练/prefill/部分 decode | 降低中间显存和访存 | 通常否；属于实现优化 |

其中，Keyformer 不减少模型训练阶段的标准注意力成本；xFormers 也不是一个新的 Transformer 架构。把二者单列出来有助于避免“所有 xxFormer 都在解决同一个问题”的误解。

## 3. 实验目标

从仓库说明、代码和产物可以归纳出四层目标：

1. **可运行性目标**：在本地 PyTorch/CUDA 环境中运行各方法或其代表性注意力模块；
2. **效率目标**：比较序列长度增加时的前向延迟和峰值显存，观察理论复杂度是否在实际硬件上转化为收益；
3. **质量目标**：检查高效方法相对全注意力的输出偏差，以及在 WikiText-2 上的语言建模质量；
4. **深度案例目标**：以 Keyformer 为例，回答 KV cache 压缩率、PPL 损失、解码速度和基线策略之间的权衡。

若要进一步回答“哪种架构设计更好”，还必须增加一个第五层目标：在同一任务、同一模型容量、同一训练预算、同一硬件和同一测量协议下，获得带随机种子与置信区间的 Pareto 前沿，而不是只比较单次或异构实验中的绝对数字。

## 4. 仓库工作与实验计划复原

### 4.1 代码与资料组织

项目根目录中的关键材料包括：

| 内容 | 位置 | 作用 |
|---|---|---|
| 项目总览 | `README.md`、`STATUS.md` | 方法范围与完成状态 |
| 来源说明 | `SOURCES.md` | 论文、官方实现与本地可运行实现映射 |
| 通用微基准 | `universal_benchmark.py`、`run_benchmark.py` | 统一输入输出接口、计时、峰值内存、JSON/PNG |
| 通用质量工具 | `common/wikitext_quality.py` | WikiText-2、输出误差、TinyLM 训练与 PPL |
| Longformer 实现 | `models/longformer_attention.py` | 局部窗口与全局 token 注意力 |
| Keyformer 实现 | `models/keyformer.py`、`models/gpt2_keyformer.py` | 算法级策略与 GPT-2 Medium 集成 |
| Keyformer 深度报告 | `KEYFORMER_REPORT.md` | 协议、结果和复现说明 |
| 原始结果 | 各方法的 `results/*.json` | 机器可读实验数据 |
| 图表 | 各方法的 `figures/*.png` | 性能和质量可视化 |
| 测试 | `tests/` | Longformer、Keyformer 与 benchmark 正确性检查 |

此外，`performer-pytorch/`、`reformer-pytorch/`、`memformers/` 保存了第三方可运行源码快照。来源文件说明 Memformer 使用论文作者仓库，Performer/Reformer 的本地 PyTorch 实现来自社区重实现，而官方实现分别位于 Google Research 与 Trax。后续论文写作应继续区分“论文官方实现”和“本实验采用实现”。

### 4.2 已执行实验的分层

#### A. 统一注意力微基准

- 输入：随机张量，batch=1，hidden size=256，8 heads；
- 序列长度：128、256、512；
- 对象：Standard、Memformer 注意力适配器、Performer、Reformer；
- 环境：RTX 4070 Laptop GPU；
- 指标：推理前向延迟、PyTorch CUDA peak allocated memory；
- 运行次数：每个长度 3 次。

该实验测量“当前实现的一个注意力风格模块在短序列上的工程成本”，没有训练完整模型，也没有参数量匹配。

#### B. 各方法历史性能脚本/图

Linformer、Longformer 和 xFormers 目录包含独立脚本与长序列图，序列长度可到 8192。它们能展示随长度增长的趋势，但并非由当前统一 runner 生成，缺少统一 JSON 环境记录，部分使用进程 RSS 差分作为内存指标。适合做探索性参考，不宜与 A 组数字直接合并。

#### C. WikiText-2 注意力保真度

大多数方法使用 WikiText-2 token 的随机 embedding、固定随机 Q/K/V 投影，比较高效注意力输出与全 softmax 输出的 MSE、relative L2 和 cosine similarity。该实验隔离了注意力近似的即时数值影响，但没有使用经过训练的表示或真实下游层，因此不是模型准确率。

#### D. WikiText-2 TinyLM

Linformer、xFormers、Memformer、Performer、Reformer 使用 hidden size=256、2 层、8 heads、sequence length=128 的小语言模型，保存结果显示训练 120 步，在 15 个窗口上评估 PPL。每个架构单独训练，主要用于确认模型能够学习，而非可靠排名。

#### E. 预训练 Longformer MLM

使用 `allenai/longformer-base-4096` 在 256 tokens 的单个遮盖窗口上得到 MLM loss/PPL。该结果说明预训练模型可以运行，但没有同模型全注意力或不同窗口的对照。

#### F. Keyformer 真实生成与质量实验

- 模型：`openai-community/gpt2-medium`（355M）；
- GPU：RTX 5060 Laptop GPU 8GB；
- 精度：FP16，batch=1，greedy decode；
- prompt：256/512/768，生成 128 tokens；
- cache ratio：25%/50%/75%/100%；
- 策略：FullKV、RecentWindow、Random+Recent、Keyformer；
- seeds：0/1/2；
- 质量：WikiText-2 测试集前 2048 tokens，context=512，teacher-forced NLL/PPL/top-1；
- 指标：prefill、decode ms/token、tokens/s、物理 KV bytes、CUDA peak memory、PPL、top-1。

该组实验是仓库中控制最完整、可解释性最强的部分。

## 5. 实验结果

### 5.1 Standard、Memformer、Performer、Reformer 短序列微基准

来源：`benchmark_results.json`。延迟单位为 ms，显存为 CUDA peak allocated MiB。

| 方法 | L=128 | L=256 | L=512 |
|---|---:|---:|---:|
| Standard | 0.378 / 11.88 MiB | 0.368 / 15.63 MiB | 0.664 / 29.13 MiB |
| Memformer adapter | 0.789 / 14.32 MiB | 0.623 / 18.82 MiB | 1.060 / 33.82 MiB |
| Performer | 1.254 / 16.80 MiB | 1.116 / 21.43 MiB | 2.328 / 30.69 MiB |
| Reformer | 1.943 / 17.41 MiB | 2.013 / 22.93 MiB | 2.317 / 33.98 MiB |

在 128–512 这一范围内，Standard 都最快，并且峰值显存最低或接近最低。这个结果不反驳高效架构的渐近复杂度，而是表明在短序列、小 hidden size 和高度优化的 GPU dense kernel 上，哈希、随机特征、记忆拼接及额外数据变换的常数开销占主导。要观察理论优势，需要将长度扩展到 1k–32k，记录 OOM 边界，并增加更稳定的重复测量。

Memformer 项这里只是带 64 个持久 memory slots 的 `MemBartEncoderAttention` 适配器，不是训练好的完整 MemBART；Performer 使用全局 FAVOR+、无 local heads；Reformer 使用 bucket size=64、2 hashes。三者不是参数量完全一致的端到端模型。

### 5.2 历史长序列性能图能说明什么

#### Linformer

历史图中，Linformer 从 256 到 8192 的耗时始终低于脚本中的手写全注意力，且随长度增长的斜率更低；k 从 64 增至 1024 时耗时上升，符合 \(O(nk)\) 的预期。这是“低秩投影可能在长序列带来时间收益”的正向信号。

但同图的内存曲线出现负 RSS 增量（最低约 -500 MB），说明该测量不能用于内存节省结论；图中的合成分类训练也使用随机数据且样本很少，不能作为任务质量证据。

#### Longformer

历史图中，window=256/512 的 Python/当前实现比手写 Standard baseline 慢，尽管稀疏结构理论上减少了注意力项。可能原因包括窗口展开、索引和 kernel 融合不足。内存曲线有剧烈非单调跳变，因此只可说明脚本运行过，不能用于定量排名。

Longformer 自编实现的测试价值更高：测试覆盖输出形状、padding、边界、全局 token 可达性、梯度、全窗口等价性，并检查未构造完整 `sequence × sequence` einsum。这支持“实现确实采用窗口存储结构”，但仍需 GPU 长序列 benchmark 证明实际收益。

#### xFormers/SDPA

历史图显示，短序列时高效内核的启动成本可能略高，长度增加后明显快于脚本中的普通实现；注意力保真度也几乎为数值等价。这符合 xFormers/Flash/SDPA 的设计：不牺牲注意力定义，通过融合和分块降低 HBM 读写与中间矩阵存储。

但当前代码注释说明 RTX 50 系列上 xFormers kernel 可能因 SM 兼容性回退到 PyTorch SDPA，因此“xFormers available”不等于每次实际调用了同一个 xFormers kernel。后续必须记录实际 backend 名称。

### 5.3 注意力输出保真度

| 方法 | seq/chunks | Relative L2 | Cosine similarity | 解释边界 |
|---|---:|---:|---:|---|
| xFormers/efficient | 128 / 4 | 3.78e-7 | 0.99999997 | 与全注意力数值等价，符合精确内核预期 |
| Performer | 128 / 4 | 0.00223 | 0.9999975 | 当前随机投影与 64 features 下近似很接近 |
| Longformer local w=64 | 128 / 4 | 0.725 | 0.809 | 局部窗口丢失远距离连接，输出明显变化 |
| Linformer random projection k=64 | 128 / 4 | 1.152 | 0.201 | 未训练的随机低秩投影并不保真 |
| Memformer-like vs Standard | 128 / 1 | 1.374 | 0.013 | 两者目标语义不同，不应解释成近似误差 |
| Reformer LSH vs Standard module | 128 / 1 | 1.519 | -0.098 | 未共享完整投影/输出权重，不能归因于 LSH 本身 |

这组数据最明确的结论是：xFormers/SDPA 的主要价值是工程优化而非近似；Longformer 等结构近似会改变单层输出；未经学习的 Linformer 随机投影不能代表训练后低秩投影的质量。Performer 的当前单层结果很接近，但仍需跨 feature 数、长度、输入分布和 seed 验证随机特征方差。

### 5.4 TinyLM 与预训练 Longformer 质量结果

| 方法实验 | Standard PPL | Efficient PPL | 表面差异 |
|---|---:|---:|---:|
| Linformer | 29.682 | 27.974 | Linformer 较低 5.8% |
| xFormers/SDPA | 29.405 | 28.170 | efficient 较低 4.2% |
| Memformer-like | 31.231 | 27.805 | memory-like 较低 11.0% |
| Performer | 30.224 | 32.127 | Performer 较高 6.3% |
| Reformer | 29.076 | 30.720 | Reformer 较高 5.7% |

这些数字不能被解释为架构真实优劣，原因包括：

1. 结果只有单次训练，没有固定并成对复用除注意力外的初始化，也没有多 seed 均值和方差；
2. 只训练 120 步、仅评估 15 个高度相关窗口，训练与评估样本很小；
3. 参数量、注意力实现内部投影数及优化难度不匹配；
4. 多个 `StandardAttn` 实现实际没有因果 mask，而 TinyLM 名称和目标是 causal LM；
5. xFormers 的 Standard 路径设置 `is_causal=True`，efficient 路径却设置 `is_causal=False`，形成未来 token 泄漏的不公平对照；
6. Performer/Reformer 的 Standard 路径调用未加 causal mask 的 `full_attention`，而 efficient 路径配置 causal，方向相反；
7. 当前脚本默认 steps=150，但保存 JSON 为 120 steps，运行参数没有写入统一实验配置。

因此，这一组只能证明各脚本可训练并产生有限数值，不应放入论文主结论或架构排行榜。建议修复因果性和配对初始化后重新运行。

Longformer 的预训练 MLM 得到 loss=2.142、masked-token PPL=8.514（256 tokens）。它使用真实预训练模型，但只有一个固定遮盖模式和一个窗口，也没有 RoBERTa/全注意力对照，故只能作为运行 smoke test。

### 5.5 Keyformer：KV cache、质量与速度

#### 5.5.1 正确性证据

Keyformer 部分具有以下关键门槛：

- FullKV prefill logits 与原始 GPT-2 对齐；
- FullKV greedy tokens 与 Hugging Face 生成一致；
- Keyformer 在 100% budget 时与 FullKV token 一致；
- 压缩策略满足 `cached_tokens <= budget`；
- 原始绝对位置在裁剪后仍被保存；
- Random+Recent 的 seed 可复现；
- 100% budget 下各策略的 WikiText-2 PPL 均为 27.903。

这些测试使压缩结果不太可能来自 cache 没有真正参与后续 logits、位置编号回退或只 mask 不 gather 等常见实现错误。

#### 5.5.2 质量—压缩权衡

三 seed 均值如下：

| 策略 | Cache ratio | PPL | 相对 FullKV | Top-1 |
|---|---:|---:|---:|---:|
| FullKV | 100% | 27.903 | 基线 | 41.14% |
| Keyformer | 75% | 28.111 | +0.75% | 40.93% |
| Keyformer | 50% | 28.697 | +2.84% | 40.64% |
| Keyformer | 25% | 32.822 | +17.6% | 38.96% |
| Random+Recent | 75% | 30.709 | +10.1% | 39.81% |
| Random+Recent | 50% | 129.946 | +365.7% | 28.60% |
| Random+Recent | 25% | 1951.644 | 极大退化 | 14.45% |
| RecentWindow | 75% | 76.660 | +174.7% | 33.37% |
| RecentWindow | 50% | 530.219 | +1800% | 21.62% |
| RecentWindow | 25% | 3757.343 | 灾难性退化 | 11.40% |

在相同 cache budget 下，Keyformer 显著优于只保留近期 token 和随机保留旧 token 的基线。最实用的区域是 50%–75% budget：物理 KV 减少 25%–50%，PPL 只增加约 0.75%–2.84%。25% budget 仍可运行，但质量损失已经明显。

#### 5.5.3 物理 KV 容量

| Prompt + 128 decode | FullKV | 75% | 50% | 25% |
|---|---:|---:|---:|---:|
| 256 | 36 MiB | 27 MiB | 18 MiB | 9 MiB |
| 512 | 60 MiB | 45 MiB | 30 MiB | 15 MiB |
| 768 | 84 MiB | 63 MiB | 42 MiB | 21 MiB |

物理 K/V tensor 大小基本严格随预算变化，这是最直接、最稳定的内存结论。CUDA allocator peak 还受到模型权重、临时 tensor、缓存和碎片影响，不能期待与 KV bytes 同比例下降。

#### 5.5.4 解码延迟

以 50% cache 为例：

| Prompt | FullKV ms/token | Keyformer ms/token | 由均值计算的 speedup |
|---|---:|---:|---:|
| 256 | 22.07 | 43.73 | 0.50×，明显变慢 |
| 512 | 36.77 | 35.77 | 1.03×，近似持平 |
| 768 | 63.74 | 44.37 | 1.44×，出现加速 |

结果支持一个清晰的交叉点：短上下文时 score 累积、top-k 和 gather 开销大于少算的注意力；上下文更长后，减少 K/V 数量才可能抵消选择开销。

需要谨慎解读 768/25% 的 speedup。原始三 seed 的 FullKV 延迟约为 90.77、33.83、66.64 ms/token，波动很大；Keyformer 对应为 36.95、35.99、72.77。JSON 中逐 seed speedup 的算术平均约为 1.44×，但“FullKV 总均值 / Keyformer 总均值”为 63.74/48.57≈1.31×，而且只有 seed 0 显著加速。这不是算法错误，而是统计汇总方式与高方差共同造成的差异。正式结论应增加 20–50 次重复、随机化运行顺序，并报告 median、bootstrap 95% CI 和 paired speedup。

## 6. 当前能支持的架构优缺点

下表把理论设计、仓库证据和仍未验证部分分开：

| 方法 | 设计优势 | 设计代价/风险 | 本仓库证据强度 |
|---|---|---|---|
| Linformer | 低秩投影使复杂度随 \(nk\) 增长；长序列图显示更好的时间斜率 | 固定投影 rank 是容量瓶颈；对长度外推和自回归因果处理更复杂 | 性能趋势中等；质量证据弱，随机 projection fidelity 不能代表训练后模型 |
| Longformer | 局部窗口适合长文局部依赖；全局 token 可传播远程信息；存储结构可线性扩展 | 全局 token 选择依赖任务；窗口外依赖可能丢失；未融合实现可能比 dense 更慢 | 正确性测试较强；性能/任务收益证据弱 |
| Reformer | LSH 让相似 token 跨距离聚合；理论上适合很长序列；可逆层可省激活内存 | hash/bucket/n_hashes 引入随机性和不连续性；排序分桶常数开销高；短序列不占优 | 可运行性中等；当前 fidelity 不公平，任务证据弱 |
| Performer | 线性 attention；无需人工规定稀疏图；可支持因果形式 | feature 数决定速度—方差；随机投影稳定性、数值稳定性和重绘策略需调参 | 单层 fidelity 是正向信号；短序列性能不占优；任务证据弱 |
| Memformer | 固定容量记忆可服务流式/超长历史；计算与保存状态有上界 | 记忆槽是强信息瓶颈；写入、遗忘和跨块训练难；并行性下降 | 当前只是注意力适配器/类 Memformer；没有真实跨块状态质量实验 |
| Keyformer | 无需重训练即可压缩生成 KV；保留重要旧 token 明显优于纯 recent/random；物理 cache 可控 | 只优化 decode；选择/gather 有额外成本；丢弃后不可恢复；长上下文和 RoPE 模型尚未验证 | 当前最强：真实预训练模型、正确性门槛、同预算质量基线均具备 |
| xFormers/SDPA | 通常保持精确注意力质量；能利用融合 kernel 降低显存和访存；工程部署直接 | 收益依赖 GPU、dtype、shape 和 backend；不改变 \(n^2\) 算术量；kernel 可能回退 | 数值等价证据强；历史性能趋势正向；缺实际 backend 与统一峰值显存记录 |

从现有证据看，不应宣布某一种方法“总体最佳”。更合理的选择原则是：

- 需要保持原注意力语义并追求工程效率：优先 xFormers/SDPA/Flash 类内核；
- 长文编码且任务具有局部结构：Longformer 值得验证；
- 希望训练/整段推理的线性近似：对 Performer 与 Linformer 做 feature/rank 扫描；
- 流式、跨块状态：验证真正的 Memformer 记忆更新；
- GPT 类自回归 decode 的 KV 受限：Keyformer 当前证据最直接；
- 超长内容寻址且可接受哈希随机性：Reformer 需要在远长于 512 的长度上再判断。

## 7. 审计发现与实验局限

### 7.1 跨实验不可直接比较

- RTX 4070 与 RTX 5060 两套硬件混用；
- FP32 微基准与 FP16 Keyformer 不同；
- 有的是单注意力层，有的是 2 层 TinyLM，有的是 355M 预训练 GPT-2；
- 有的是双向注意力，有的是因果生成；
- 性能图、JSON 和脚本并非全部来自同一 runner。

### 7.2 TinyLM 的因果性与公平性问题

这是质量实验中优先级最高的问题。当前多条 Standard 路径没有 causal mask；xFormers 两条路径的 `is_causal` 甚至不同。存在未来 token 泄漏时，PPL 不再衡量合法自回归预测，差异也会被掩码差异而不是架构能力主导。

### 7.3 未控制初始化、参数量和随机种子

两个模型分别构造会消耗不同随机数，embedding、MLP、layer norm 和输出头并不共享。单 seed 下几 PPL 的差距完全可能来自初始化与短训练噪声。应对非注意力参数使用相同 state dict，并使参数量或训练 FLOPs 可比。

### 7.4 注意力 fidelity 的适用范围

随机 embedding 和随机 QKV 有利于机制隔离，但分布不等于训练模型内部激活。Linformer 的随机长度投影尤其不能代表学习后的投影；Memformer 的输出本来就不应等于无记忆全注意力；Reformer 当前对照没有共享内部投影权重。

### 7.5 内存测量不统一

历史脚本用进程 RSS 的前后差值，受到 Python allocator、垃圾回收和 CUDA 异步行为影响，已经产生负值。统一实验应至少记录：

- `torch.cuda.max_memory_allocated()`；
- `torch.cuda.max_memory_reserved()`；
- 模型参数 bytes、输入 bytes、KV/activation bytes 的解析值；
- 运行前 baseline 与 OOM 状态；
- 训练时 forward+backward+optimizer 的峰值，而不只 forward。

### 7.6 性能统计不足

3 次运行难以覆盖 GPU 动态频率、Windows 调度、首次 kernel、温度和后台任务。Keyformer 768 长度的 seed 间大波动已经显示该问题。模型执行顺序也固定，可能把热机/降频效应混入架构差异。

### 7.7 数据规模与任务覆盖不足

- TinyLM 只使用最多 2048 tokens、120 steps；
- Keyformer PPL 只评估 2044 个 target tokens，而非完整 WikiText-2 test；
- 没有长程检索、复制、跨块记忆、长文分类/问答等能区分架构归纳偏置的任务；
- 没有生成质量或事实保持指标；
- 没有训练吞吐、反向显存和收敛到固定质量的总成本。

### 7.8 可复现性细节

优点是 Keyformer 保存了较完整配置和环境，通用 runner 也能写环境 metadata。仍需改进：锁定依赖版本/commit；所有 JSON 写入 git commit、命令行参数、seed、dtype、backend、GPU power state；历史图补机器可读原始数据；图表避免缺失中文字体；删除或标记过时配置（例如某测试仍期待 prompt 1024，但最终实验因 GPT-2 最大位置改为 768）。

## 8. 要分析架构设计优缺点，必须补充的实验

### 8.1 最高优先级：统一且正确的端到端语言建模对比

目标是回答“在同等质量下谁更快/更省，或在同等资源下谁质量更好”。

建议协议：

1. 任务使用 WikiText-103 或 OpenWebText 子集，另设完整 WikiText-2 快速版；
2. 使用相同 tokenizer、hidden size、层数、heads、FFN、embedding tying、dropout；
3. 所有方法使用严格相同 causal mask 语义；
4. 对非注意力权重做成对初始化；
5. 同时报告参数量匹配与训练 FLOPs 匹配两套设置；
6. 至少 3–5 个训练 seed；
7. 训练到相同 token 数，并另报告达到固定验证 PPL 所需时间/FLOPs；
8. sequence length 至少覆盖 512、1k、2k、4k、8k；
9. 每个 checkpoint 在同一完整验证集上评估 NLL/PPL；
10. 汇报均值、标准差/95% CI，而不是最佳单次结果。

这项实验能消除当前 TinyLM 最严重的因果与初始化混淆。

### 8.2 统一性能扩展曲线与交叉点

在同一台 GPU、同一 PyTorch/CUDA、同一 dtype 下，对 Standard、SDPA/Flash、Linformer、Longformer、Performer、Reformer 和可合理适配的 Memformer 运行：

- sequence length：128 到 OOM，建议 2 的幂扩展至 32k/64k；
- batch：1、4，以及“填满显存的最大 batch”；
- dtype：FP32、BF16/FP16；
- forward inference、forward+backward、autoregressive prefill/decode 分开；
- warmup ≥20，timed repeats ≥50；
- 随机化方法执行顺序；
- 报告 median、p10/p90、95% CI、tokens/s、peak allocated/reserved、OOM 长度。

重点不是只报“8192 时谁最快”，而是求每种方法相对强 baseline（PyTorch SDPA/Flash）的速度和显存交叉点。手写 einsum Standard 可以保留用于教学，但不能作为唯一工程 baseline。

### 8.3 每种架构的关键消融

| 方法 | 必做消融 | 需要回答的问题 |
|---|---|---|
| Linformer | rank k={32,64,128,256,512}；共享/不共享 K/V projection；训练长度外推 | 质量是否由低秩假设限制？k 增加到何处收益饱和？固定长度 projection 能否泛化？ |
| Longformer | window={64,128,256,512,1024}；global token 数/位置/学习选择；层间交错连接 | 局部窗口多大才足够？全局 token 是否真正传递远程信息？速度收益是否被实现开销抵消？ |
| Performer | random features r={32,64,128,256,512}；redraw interval；orthogonal features；seed | 近似方差如何随 r/长度变化？质量—速度 Pareto 点在哪里？ |
| Reformer | bucket size、n_hashes={1,2,4,8}、hash seed；相同权重 full-attn 对照 | 增加 hashes 能否恢复质量？排序/分桶成本何时摊薄？结果稳定性如何？ |
| Memformer | memory slots={16,32,64,128,256}；chunk size；detach/BPTT 长度；memory update/forget | 固定记忆能保留多远的信息？槽位增多的收益和成本？记忆是否塌缩或只记近期？ |
| Keyformer | ratio、recent ratio、tau_init/tau_delta、score 规则、逐层/逐头预算 | 质量来自重要 token 选择还是 recent window？选择开销如何减少？不同层/头是否需要不同预算？ |
| xFormers/SDPA | backend、dtype、head dim、causal/noncausal、dropout、batch/length | 何时使用 Flash、memory-efficient 或 math backend？回退条件和收益边界是什么？ |

### 8.4 长程依赖和归纳偏置任务

仅用短上下文 PPL 难以区分设计优势。建议加入：

- **Associative Recall / Passkey Retrieval**：测试远距离键值检索；
- **Copy / Selective Copy**：测试精确长期记忆；
- **Needle-in-a-Haystack**：随上下文长度和 needle 位置画准确率热图；
- **Long Range Arena**：ListOps、Text、Retrieval、Pathfinder 等；
- **长文分类/问答**：验证 Longformer 全局 token 设计；
- **分块流式任务**：每块只见局部输入，专门验证 Memformer 跨块记忆；
- **长文本生成**：验证 Keyformer 对早期事实、实体一致性和引用保持。

这些任务必须按架构机制设计。例如，若没有跨 chunk 评估，Memformer 的核心优势根本没有被测试；若没有远距离目标，Longformer 的全局连接和 Reformer 的内容寻址也难以区分。

### 8.5 近似误差的层级传播实验

建议在同一个预训练 Transformer 中替换单层、前半层、后半层或全部注意力，并共享 Q/K/V/O 权重，记录：

- 单层 output MSE/relative L2/cosine；
- attention distribution KL/JS（仅语义可比的方法）；
- hidden-state CKA/cosine 随层深变化；
- logits KL、top-k overlap、PPL；
- 误差随 sequence length、head、token 距离的变化。

这能回答“单层近似看似很小，堆叠后是否放大”，也能找到更适合近似或压缩的层与 head。

### 8.6 Keyformer 的必要扩展

Keyformer 当前最接近可发表结论，但仍需：

1. 更长上下文且支持 RoPE 的现代模型，如 Pythia、Llama/TinyLlama 或 Qwen 小模型；
2. context 至少 2k、4k、8k，确认速度交叉点与质量趋势；
3. 完整 WikiText-2/PG-19，以及 LongBench/Needle/生成一致性；
4. 与 H2O、StreamingLLM、SnapKV、PyramidKV 等更强 KV eviction baseline 比较；
5. 20–50 次性能重复与随机运行顺序；
6. 单独 profile attention、score update、top-k、gather、position handling 的耗时；
7. fused/批量化 selection kernel，减少 Python/eager 开销；
8. 多 batch、beam search、不同 generation length；
9. 层级/头级预算和动态预算策略；
10. 质量约束下的最佳 speedup，而不是孤立报告最低 ratio。

## 9. 建议的新统一实验矩阵

为了控制工作量，可分为三个阶段。

### 阶段 1：修正与可信最小集

- 修复所有 TinyLM causal mask；
- 统一 seed、配对初始化、参数统计和 JSON schema；
- 在 WikiText-2 上运行 3 seeds、至少 2k–10k steps；
- 使用同一台 GPU 对长度 128–8192 做统一 forward benchmark；
- 把 PyTorch SDPA/Flash 设为主要 baseline；
- 为每个结果保存命令、环境、backend 和原始 samples。

这是把当前“探索性结果”升级为“可信课程实验”的最低成本路径。

### 阶段 2：架构机制验证

- 完成各方法关键超参数消融；
- 加入 LRA/Passkey/Copy/跨块 memory 等机制匹配任务；
- 报告 quality—latency—memory Pareto frontier；
- 使用 Nsight/PyTorch Profiler 解释 kernel 与访存瓶颈。

### 阶段 3：规模化与现代模型

- 在 100M–1B 参数模型上验证长度扩展；
- Keyformer 接入 RoPE 模型并加入强 KV baseline；
- 完整训练或标准数据集 fine-tuning；
- 多 GPU/不同 GPU 代际验证结论的硬件依赖。

## 10. 推荐的统一结果表结构

后续每条实验记录至少应包含：

```text
run_id, git_commit, timestamp
method, implementation_source, backend
task, dataset, split, tokenizer
model_params, trainable_params, layers, dim, heads
sequence_length, batch_size, dtype, causal
method_hyperparameters
seed, train_tokens, optimizer, learning_rate
gpu, torch, cuda, driver
latency_samples, throughput, peak_allocated, peak_reserved
loss, perplexity/accuracy, status, error
```

所有图都从这一份长表自动生成，避免图、文档和 JSON 的汇总口径不一致。

## 11. 工作质量评价

### 已做得较好的部分

- 方法覆盖面广，能够展示高效 Transformer 的多条技术路线；
- 保留了来源和第三方代码 provenance；
- 通用 benchmark 已改为 inference mode、CUDA event 和结构化 record；
- Longformer 的结构与边界测试较全面；
- Keyformer 有真实模型集成、物理 KV gather、位置跟踪和同预算基线；
- 多数实验有 JSON 原始结果而不仅是截图；
- Keyformer 报告主动说明没有复现论文 A100/大模型 headline，结论较克制。

### 最需要改进的部分

- 统一实验协议与主 baseline；
- 修复 TinyLM 因果性和初始化公平性；
- 用真实、完整、可比较的任务质量取代单次短训练 PPL；
- 废弃 RSS 差分内存结论；
- 增加长序列、重复次数、误差条和 OOM 边界；
- 区分架构近似、记忆机制、KV eviction 和精确 kernel 优化；
- 将散落在各目录的实验汇总到同一机器可读表。

## 12. 最终结论

本项目已经成功搭建了多个高效 Transformer 方法的代码与实验入口，并完成了从模块级性能、注意力输出误差到预训练模型 KV cache 压缩的多层次探索。它最有说服力的成果是 Keyformer：在 GPT-2 Medium 上，50%–75% KV budget 提供了明确的内存—质量折中，且重要 token 选择显著优于同预算 recent/random eviction；当上下文足够长时可以出现解码加速，但短上下文和当前 eager 实现会被选择开销反噬。

对 Linformer、Longformer、Reformer、Performer、Memformer 和 xFormers，现有工作已经能够说明实现是否可运行、某些理论趋势是否出现以及单层输出如何变化，但还不能可靠回答总体优劣。当前 TinyLM、历史内存图和异构硬件结果中存在足以改变排名的混杂因素。最科学的下一步是先修复因果性与配对初始化，再以强 SDPA/Flash baseline、统一长序列性能协议、机制匹配任务和多 seed 质量实验建立 Pareto 前沿。

如果上述补充实验完成，报告就可以从“多方法实验汇编”升级为真正回答架构设计问题的比较研究：不同方法在哪个长度、任务结构、硬件与质量容忍度下更合适，以及它们的理论优势何时能转化成可测的系统收益。

## 13. 主要证据索引

- 项目状态：`README.md`、`STATUS.md`
- 论文与实现来源：`SOURCES.md`
- 通用微基准：`run_benchmark.py`、`universal_benchmark.py`、`benchmark_results.json`
- 通用质量实现：`common/wikitext_quality.py`
- 各方法结果：`linformer/results/`、`longformer/results/`、`xformer/results/`、`memformer/results/`、`performer/results/`、`reformer/results/`
- Longformer 实现与测试：`models/longformer_attention.py`、`tests/test_longformer.py`
- Keyformer 实现、测试与结果：`models/keyformer.py`、`models/gpt2_keyformer.py`、`tests/test_keyformer.py`、`tests/test_gpt2_keyformer.py`、`keyformer/results/`
- Keyformer 原报告：`KEYFORMER_REPORT.md`

