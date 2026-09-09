# Efficient Transformer Design Trade-offs in TinyLM: A Unified Study of Six Former Architectures and Full-Layer MoE

## 摘要

高效 Transformer 方法通常从不同位置降低计算或存储成本：限制 attention 连接、压缩序列维、近似 softmax kernel、筛选稀疏候选、维护 recurrent memory，或在推理时压缩 KV cache。本文整合 TinyStories/TinyLM 项目中的 Dense FFN 10M-token screening、Longformer/Memformer 100M-token 扩展、Causal-LSH Reformer 与 Full-Attention protocol v2 结果，以及一组将全部六层 FFN 替换为 4-expert、Top-1、dropless MoE 的 seed-17 实验，系统分析六种 Former 架构的质量、训练效率、显存、KV 存储和设计归因。

在 A+B unified Dense 10M protocol（context=512、三 seed）中，Memformer 的平均 validation PPL 最低（17.2732），其次为 Longformer（17.9129）、Linformer（18.7995）和 Performer（31.5161）。Longformer 与 Memformer 延长到 100M tokens 后质量排序反转：Longformer PPL=6.0967，优于 Memformer 的 6.4622，但 Memformer 训练吞吐高 14.03%，peak allocated memory 低 61.28%。全层 MoE 在 seed=17 下进一步改善了 Memformer、Longformer 和 Linformer 的 PPL，改善幅度分别为 8.30%、1.43% 和 0.84%；Performer 仅改善 0.12%。MoE 总参数增加约 63%–73%，但 Top-1 active parameters/token 基本不变，当前通用 PyTorch dispatch 仍使训练吞吐下降 7%–27%。Keyformer 在 Dense 和 MoE Full-Attention backbone 上均将 50% KV cache 物理压缩到约一半，并在当前窗口评估中保持 PPL；但选择、gather 和 cache update 开销使 decode 吞吐下降约 16%–17%。

这些结果表明，六种方法并非同一技术层面的互斥替代品：Longformer 和 Memformer改变历史信息通路，Linformer、Performer 和 Reformer近似训练时 attention，Keyformer压缩推理状态，而 MoE扩大 FFN 的条件容量。当前证据支持“不同设计对应不同资源—质量取舍”，不支持跨协议的绝对总排名。要形成更强的因果结论，还需要同一 runner 的 Dense/MoE 配对消融、多 seed MoE、真实长依赖任务以及 fused attention/MoE/cache kernel。

**关键词：** Efficient Transformer；Longformer；Memformer；Linformer；Performer；Reformer；Keyformer；Mixture-of-Experts；KV cache；TinyStories

## 1. 引言

标准 causal self-attention 为长度为 (N) 的序列构造近似 (O(N^2)) 的 token-pair 交互，并在自回归生成中累积随上下文增长的 K/V cache。高效 Transformer 的目标并不唯一：有的方法减少训练时连接数量，有的方法压缩 K/V 的表示，有的方法以近似 kernel 替代 softmax，有的方法把历史压缩成 recurrent state，还有的方法只在推理时淘汰不重要的缓存 token。

如果只比较最终 PPL，容易忽略三类问题。第一，不同方法优化的是不同阶段，训练型 backbone 与推理型 cache policy 不能直接排名。第二，渐近复杂度不等于实际速度；通用 PyTorch adapter、排序、gather 和 kernel 常数可能完全改变排序。第三，增加模型容量（例如 MoE）可能改善质量，但这种改善不等于 attention 机制本身的净收益。

本文以同一 TinyLM/TinyStories 研究线为基础，整合已有 Dense、100M 扩展、Causal-LSH 和全层 MoE 结果，回答：

1. 六种 Former 架构分别改变了 Transformer 的哪个设计部分？
2. 它们的质量、吞吐、显存和 KV 存储现象是否与设计预期一致？
3. 全层 Top-1 dropless MoE 对不同 attention/memory backbone 的影响是否一致？
4. 当前观察到的优势和缺点，哪些可以归因于结构设计，哪些仍混合了容量、实现和协议差异？

本文的主要贡献是：

- 建立训练型 attention/memory 方法与推理型 KV-cache 方法的统一分类和比较边界；
- 汇总 Dense 10M、Longformer/Memformer 100M 和六变体 MoE 结果；
- 给出 Dense FFN→MoE 的 seed-17 配对 screening；
- 将实验现象映射到固定窗口、recurrent slots、低秩投影、随机特征、LSH 候选和 cache eviction 等具体设计；
- 明确当前证据不能支持的结论与下一步严格实验。

## 2. 方法分类与设计假设

### 2.1 六种架构

| 架构 | 训练/推理阶段 | 被改变的对象 | 设计假设 |
|---|---|---|---|
| Longformer | 训练型 | causal attention 连接图 | 近邻 token 是主要有用上下文，远距信息可通过层叠窗口传播 |
| Memformer | 训练型 | segment 间历史状态 | 历史可压缩为固定数量 memory slots，并由 recurrent state 跨段传递 |
| Linformer | 训练型 | K/V 序列维 | K/V 在序列方向具有较低有效秩 |
| Performer | 训练型 | softmax attention kernel | softmax kernel 可由有限随机特征近似 |
| Reformer | 训练型 | attention 候选集合 | 相关 query/key 更可能落在相同 LSH bucket |
| Keyformer | 推理型 | 自回归 KV cache | 少数历史 token 对下一步预测更重要，可安全淘汰其余 token |

Full Attention 是精确 causal attention 基线，不计入六种 Former 名称。MoE 是与上述 attention/memory 设计正交的 FFN 容量因素。

### 2.2 设计—现象对应关系

- Longformer 的固定窗口带来近似 (O(NW)) 候选规模，但产生层数×窗口的远距传播边界。
- Memformer 用固定 slots 将 segment 历史压缩为状态，绕过局部传播边界，但引入压缩瓶颈和额外 memory projection/gate。
- Linformer 通过低秩投影固定 K/V 序列维，可能获得正则化，也可能丢失高秩局部结构。
- Performer 将显式 softmax attention 转换为随机特征 kernel 计算，理论上可近似线性，但误差依赖 feature 数与随机投影。
- Reformer 用 LSH 筛选候选连接，收益依赖相关性与 hash locality，质量风险来自 hash collision、漏边和排序/分桶成本。
- Keyformer 不改变已训练参数，只在 decode 时选择和重排历史 K/V；它直接节省 cache bytes，但选择路径可能抵消速度收益。

## 3. 实验协议

### 3.1 公共模型和数据

主要 TinyStories/TinyLM 协议使用 6 层、hidden size=384、8 heads、FFN hidden size=1536、Pre-LN、GELU、RoPE 最大位置 32,768、tied embeddings 和 context=512。数据固定为 `roneneldan/TinyStories`，GPT-2 tokenizer 词表 50,257、EOS=50,256，每篇故事追加 EOS 后按固定顺序连续 packing。

训练使用 FP32 参数、BF16 autocast、AdamW（peak learning rate (3\times10^{-4})、3% warmup、cosine decay、weight decay=0.1），effective batch=8,192 prediction tokens。统一 10M 实验实际完成 10,000,000 个 training predictions，完整 validation 使用 4,765,917 个 next-token predictions。

### 3.2 证据组与可比性

| 证据组 | 方法 | seed | budget/context | 比较用途 |
|---|---|---:|---|---|
| A+B Dense unified | Memformer、Longformer、Linformer、Performer | 17/29/43 | 10M / 512 | Dense 主横向结果；seed=17 与 MoE 做 screening 对照 |
| B 100M extension | Longformer、Memformer | 17 | 100M / 512 | 训练预算对质量排序的影响 |
| C protocol v2 | Full Attention、Causal-LSH Reformer | 17/29/43 | 10M / 512 | 补齐 Full/Reformer 基线 |
| C Keyformer | Dense Full checkpoint | 17 | windowed decode | Dense backbone 的 KV cache 评估 |
| MoE v1 | Memformer、Longformer、Linformer、Performer、Reformer、Full Attention | 17 | 10M / 512 | 全六层 4-expert Top-1 dropless screening |

四种 A+B Dense backbone 的 seed=17 结果与 MoE 结果使用相同完整 validation 规模，可做直接 screening 对照。C v2 与 MoE 的 Full/Reformer 结果包含 runner 或候选规则差异，只能做补充比较。Keyformer 的 PPL 只在同一 checkpoint 的 FullKV 与压缩策略之间解释。

## 4. Dense FFN 结果

### 4.1 10M 三 seed 结果

| 方法 | Validation NLL | Validation PPL | workflow tok/s | Peak allocated | 参数量 |
|---|---:|---:|---:|---:|---:|
| **Memformer** | **2.849148±0.004651** | **17.273176±0.080427** | 14,669.5±300.5 | **2.633 GiB** | 33,466,752 |
| Longformer | 2.885517±0.004599 | 17.912946±0.082476 | 13,498.8±268.5 | 6.799 GiB | 29,925,504 |
| Linformer | 2.933816±0.006033 | 18.799463±0.113420 | **27,857.8±106.0** | 2.580 GiB | 29,925,504 |
| Performer | 3.450477±0.008266 | 31.516144±0.260635 | 27,079.0±541.2 | 2.704 GiB | 29,925,504 |

在该统一协议下，Memformer 的质量最好，Performer 明显落后；训练 workflow 吞吐则由 Linformer 和 Performer 领先。质量和吞吐排序不同，说明 attention 近似的渐近形式不能直接预测端到端训练速度。

### 4.2 100M 训练预算效应

Longformer 与 Memformer 从随机初始化分别训练到 100M predictions。Longformer 的完整 validation PPL 为 6.0967，Memformer 为 6.4622；Memformer 的训练吞吐为 16,813 tok/s，高于 Longformer 的 14,743.8 tok/s，peak allocated 为 2.63 GiB，低于 Longformer 的 6.80 GiB。

10M 时 Memformer 略优、100M 时 Longformer 反超，说明固定 recurrent compression 的短预算收益不能外推为充分训练后的普遍优势。Longformer 直接保留窗口内 token 细节，给足训练预算后可能更充分地学习局部统计；Memformer 则用有限 slots 换取跨 segment 路径和较低资源。

## 5. 全层 MoE 结果

### 5.1 MoE 设置

每个 Transformer block 的 FFN 替换为 4 个 expert，每个 expert hidden size=1536。router 使用 token-level Top-1、dropless dispatch，load-balance coefficient=0.01，router z-loss coefficient=0.001。Top-1 使每个 token 每层只激活一个 expert，但 checkpoint 和 optimizer 仍需保存全部 experts。

### 5.2 六个 MoE backbone

| 方法 | Val NLL | Val PPL | 训练 tok/s | Peak GiB | 总参数 | 激活参数/token |
|---|---:|---:|---:|---:|---:|---:|
| **Memformer** | **2.758941** | **15.7831** | 12,201.3 | 2.960 | 54,709,632 | 33,475,968 |
| Longformer | 2.869545 | 17.6290 | 12,265.3 | 7.234 | 51,168,384 | 29,934,720 |
| Full Attention | 2.907851 | 18.3174 | 26,503.8 | 2.911 | 51,168,384 | 29,934,720 |
| Linformer | 2.925335 | 18.6405 | 20,340.2 | 2.913 | 51,168,384 | 29,934,720 |
| Reformer | 2.980978 | 19.7071 | **26,821.2** | **2.900** | 50,283,648 | 29,049,984 |
| Performer | 3.457819 | 31.7477 | 20,358.5 | 3.120 | 51,168,384 | 29,934,720 |

该表是 MoE seed=17 的 screening，不是三 seed 统计排名。Full Attention 行是 Keyformer 的共享训练 backbone，而不是 Keyformer 独立训练网络。

### 5.3 Dense FFN 到 MoE 的配对 screening

| 方法 | Dense PPL | MoE PPL | PPL 变化 | 吞吐变化 | Peak 显存变化 |
|---|---:|---:|---:|---:|---:|
| Memformer | 17.2112 | **15.7831** | **-8.30%** | -15.55% | +12.43% |
| Longformer | 17.8841 | **17.6290** | -1.43% | -7.00% | +6.40% |
| Linformer | 18.7987 | **18.6405** | -0.84% | -26.74% | +12.89% |
| Performer | 31.7872 | 31.7477 | -0.12% | -25.79% | +15.41% |

MoE 的收益高度依赖 backbone。Memformer 的改善最大，可能反映 recurrent memory 表征与条件 FFN 的交互，但也可能部分来自额外容量和单 seed 波动。Performer 几乎没有改善，说明增加 FFN 条件容量不能自动修复随机特征 attention 的近似误差。

普通 Dense backbone 的参数量为 29,925,504，MoE 为 51,168,384，增加 70.99%；但 MoE active parameters/token 为 29,934,720，仅增加约 0.031%。因此 MoE 实现的是“扩大总容量、保持单 token 激活容量”，不是参数存储压缩。

## 6. Keyformer KV-cache 结果

Keyformer 是推理期策略，不改变训练 backbone。它按照 token 重要性保留部分历史 K/V，并保留 recent window。

### 6.1 Dense Full-Attention backbone（C v2）

| 策略 | Cache ratio | PPL | KV bytes | Decode tok/s |
|---|---:|---:|---:|---:|
| FullKV | 1.00 | 14.91 | 4,709,376 | 219.8 |
| Keyformer-100 | 1.00 | 14.91 | 4,709,376 | 219.9 |
| Keyformer-75 | 0.75 | 14.91 | 3,538,944 | 202.0 |
| Keyformer-50 | 0.50 | 14.89 | 2,359,296 | 184.2 |

### 6.2 MoE Full-Attention backbone

| 策略 | Cache ratio | NLL | PPL | Peak cached tokens | Peak KV bytes | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| FullKV | 1.00 | 2.602441 | 13.4966 | 512 | 4,718,592 | 125.4 |
| Keyformer-50 | 0.50 | 2.601262 | 13.4807 | 256 | 2,359,296 | 103.8 |

在 MoE backbone 上，Keyformer-50 将 cached tokens 和 KV bytes 都减少 50%，PPL 变化为 -0.0159；评估吞吐下降约 17.23%。Dense C v2 也观察到约 16.2% 的 decode 吞吐下降。两套实验支持“缓存节省稳定、速度收益尚未实现”，但不能跨协议比较绝对 PPL。

## 7. 机制归因：优势与缺点如何产生

### 7.1 Longformer

优势来自固定局部连接图：窗口内 token 的细粒度 K/V 被直接保留，候选规模约为 (O(NW))，结构和因果性容易解释。缺点也来自同一设计：超过层数×窗口的远距依赖不能直接连接，只能逐层传播；当前 padded unfold、query chunk 和通用 PyTorch 算子还带来明显显存和调度成本。100M 结果显示，直接保留局部细节在更高预算下可能优于固定压缩 memory。

### 7.2 Memformer

优势来自 recurrent memory slots：segment 之间存在直接状态路径，能够跨越 Longformer 的有限局部传播边界；当前实验中训练显存最低，10M PPL 也最好。缺点来自 fixed-capacity compression：历史 token 必须被投影和压缩到有限 slots，细节可能衰减；memory Q/K/V、输出投影和 gate 使参数比普通 backbone 多约 11.83%，segment fusion 也增加工程复杂度。100M 反转说明短预算质量优势并不保证长期训练后继续存在。

### 7.3 Linformer

优势来自低秩 K/V 序列投影，理论上避免对完整序列维进行 attention；当前 Dense 质量居中。缺点来自低秩假设：若真实 K/V 结构高秩或局部敏感，投影会丢信息；pooling/projection 在小模型上还可能抵消理论复杂度优势。当前实现中其质量不差，但训练 workflow 吞吐不如 Full Attention。

### 7.4 Performer

优势来自 FAVOR+ 随机特征，将 softmax attention 改写为近似 kernel 计算，理论上适合超长序列。缺点直接来自有限随机特征近似：误差随 feature 数、序列和数据分布变化；随机特征变换还有额外张量成本。当前 10M Dense PPL=31.5161，明显低于其他 A+B 方法；MoE 后几乎不变，说明 FFN 容量不是其主要瓶颈。

### 7.5 Reformer

优势来自 LSH 候选筛选：只对可能相关的 query/key 做 attention，理论上可降低候选数量，并可与 reversible/chunking 结合。缺点来自候选集合不完整和不规则执行：hash collision 会漏掉相关 token，sorting、bucket padding 和 gather 增加常数开销。C v2 还显示，若先做全局 hash 排序再构造候选，未来 token 可能改变过去 bucket，导致 future-invariance 泄漏；严格 causal candidate construction 是必要条件。当前 context=512 结果未显示其质量或速度优势。

### 7.6 Keyformer

优势来自阶段定位：无需重新训练 backbone，即可按 cache ratio 压缩推理状态，并真实减少 KV bytes。缺点来自 token importance 计算、选择、gather 和 cache update；在当前实现中，50% cache 没有带来速度提升，反而使 decode 吞吐下降约 16%–17%。Keyformer 不降低训练 attention 成本，过度压缩还可能丢失远程事实。

### 7.7 MoE 的正交作用

MoE 改变的是 FFN 条件容量，而不是 attention 连接。Top-1 routing 解释了 active parameters/token 近似 Dense；全量 experts 解释了 checkpoint 和 optimizer 存储增加；逐 expert indexing 与 `index_add_` 解释了当前吞吐下降。MoE 与 Longformer、Memformer、Linformer、Performer、Reformer 和 Keyformer 均可组合，但质量收益不能自动归因给 attention 结构。

## 8. 综合讨论

### 8.1 质量与效率不存在单一排序

Dense 10M 结果中，Memformer质量最好，但 Linformer/Performer workflow 吞吐更高；100M 时 Longformer质量反超 Memformer；MoE 后 Memformer 仍最好，但不同 backbone 的收益差异很大。由此可见，“最优架构”取决于目标函数：短预算质量、长期训练、训练显存、decode cache 或工程吞吐会产生不同选择。

### 8.2 理论复杂度与工程实现之间存在间隙

近似线性或稀疏候选并不保证小模型上更快。Full Attention 可以调用高度优化的 SDPA；而 Linformer、Performer、Reformer 需要额外 projection、random feature、sorting 或 gather。MoE 也展示了同样现象：active parameters/token 接近 Dense，但未融合 dispatch 仍带来速度损失。实际系统比较必须报告 workflow throughput、kernel 路径和内存，而不能只报告 Big-O。

### 8.3 训练期 memory 与推理期 memory 是不同问题

Memformer 通过 recurrent state 改变模型在 segment 之间携带历史的方式；Keyformer 在训练结束后淘汰部分 K/V。前者可能改变表示学习和训练激活，后者不改变主干参数。二者可以和 MoE 同时使用，但实验指标必须拆为训练质量、训练资源、KV bytes 和 decode latency 四组。

## 9. 局限性和威胁

1. MoE 只有 seed=17；其小幅改善尚无 seeds 29/43 的方差支持。
2. 现有 Dense→MoE 配对来自不同 runner，虽共享数据、模型维度和验证规模，但不是完全同代码路径的因果消融。
3. C v2 Reformer 与 MoE Reformer 使用不同候选规则，无法把差异单独归因于 FFN MoE。
4. 10M 是 screening 预算，100M 只覆盖 Longformer/Memformer，且单 seed 曲线仍可能继续下降。
5. C v2 与 MoE Keyformer 的 prediction limit、prefill/decode 流程和统计字段不同，绝对 PPL/tok/s 不应跨表排名。
6. TinyStories 主要由短故事组成；context=512 PPL 不能证明 passkey、copy、associative recall、跨章节实体保持或长对话记忆。
7. 不同方法的参数量并不完全相等，Memformer 和 MoE 都有额外容量成本。
8. 当前实现未使用 fused sliding-window、LSH、grouped-GEMM 或 cache-selection kernel，速度结果代表本仓库 adapter，而非理论硬件上限。

## 10. 结论

本文整合了六种 Former 架构的 Dense、100M 扩展、C v2 和全层 MoE 结果。实验支持以下结论：

1. Longformer、Memformer、Linformer、Performer、Reformer 和 Keyformer分别从连接图、历史状态、低秩表示、kernel近似、候选筛选和推理缓存六个位置改变 Transformer。
2. Memformer 的 recurrent state 在短预算下带来最佳 Dense/MoE PPL 和较低训练显存，但 fixed-slot compression、额外参数和长期训练反转是其主要风险。
3. Longformer 结构简单且保留局部细节；100M 结果显示它在更高预算下可能获得更好质量，但局部传播边界和当前实现显存较高。
4. Linformer 和 Performer 的理论复杂度优势在当前小模型/adapter 中没有转化为普遍端到端速度优势；Performer 还受到明显质量近似误差影响。
5. Reformer 的收益依赖真正长序列、合适的 hash locality 和严格因果的稀疏实现；context=512 当前证据不足以证明其优势。
6. 全层 Top-1 dropless MoE 扩大了总 FFN 容量，在 Memformer 上收益最明显，但代价是 checkpoint/optimizer 存储增加和未融合 dispatch 的速度损失。
7. Keyformer 与任何训练 backbone（包括 Full-Attention+MoE）正交组合，可以将 KV cache 减半并保持当前窗口 PPL；但当前实现尚未将容量节省转化为 decode 加速。

因此，最合理的系统结论不是选出一个绝对冠军，而是根据瓶颈组合设计：用 Memformer/Longformer处理训练或历史路径，用 Linformer/Performer/Reformer探索长序列 attention 压缩，用 MoE扩大条件 FFN 容量，并在 decode 显存成为瓶颈时叠加 Keyformer。要将这些结论提升为更强的论文证据，必须补齐统一 runner、多 seed、长依赖任务和融合 kernel。

## 11. 可复现性与数据来源

### 本次 MoE

- 配置：[experiments/tinystories_tinylm_moe_v1/moe_seed17.yaml](../experiments/tinystories_tinylm_moe_v1/moe_seed17.yaml)
- runner：[experiments/tinystories_tinylm_moe_v1/run_moe_six.py](../experiments/tinystories_tinylm_moe_v1/run_moe_six.py)
- 聚合结果：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/summary.json`
- correctness：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/correctness.json`

### Dense、100M 和 C v2

- A+B Dense 统一报告：[PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md](PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md)
- B 详细实验文档：[PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md](PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md)
- C v2：[C_UNIFIED_RESULT_REPORT_V2.md](C_UNIFIED_RESULT_REPORT_V2.md)
- Longformer 100M：`/root/autodl-tmp/26summerBDMI_transformer/runs/person_b_extended/extended100m_longformer_w128_s17/summary.json`
- Memformer 100M：`/root/autodl-tmp/26summerBDMI_transformer/runs/person_b_extended/extended100m_memformer_s128_m64_s17/summary.json`
- Dense A+B 逐 run：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/per_run.csv`

### 复现命令

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  experiments/tinystories_tinylm_moe_v1/run_moe_six.py remaining \
  --device cuda:0 --seed 17 --train-tokens 10000000 \
  --validation-probe-tokens 262144 --full-validation --save-checkpoint

python -u experiments/tinystories_tinylm_moe_v1/run_moe_six.py \
  aggregate --seed 17 --train-tokens 10000000
```

报告索引：[docs/README.md](README.md)
