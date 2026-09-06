# 两个云盘实验包的 TinyStories 高效 Transformer 对比总结

**审阅日期：2026-09-06**  
**数据来源：**

- [链接 A：partA_Linformer + Performer](https://cloud.tsinghua.edu.cn/d/ace43d623eca4a56a3f1/)
- [链接 C：Reformer + Keyformer](https://cloud.tsinghua.edu.cn/d/5cd5cc8533324eda9db0/)

## 1. 结论先行

### 能否对比？

**可以，但只能分层对比，不能把六个结果直接组成一张无条件总排行榜。**

两个链接中实际出现的是六个“结果条目”：

1. A 组 `full_sdpa`
2. A 组 `linformer`
3. A 组 `performer`
4. C 组 `full_attention`
5. C 组 `reformer`
6. C 组 `keyformer`

但其中 A 组和 C 组的 Full Attention 是同类基线，Keyformer 也不是独立训练的 Transformer backbone，而是作用在 Full Attention/GPT 风格模型上的**推理期 KV-cache 策略**。因此更准确的说法是：

- **四种独立注意力/模型路径：**Full Attention、Linformer、Performer、Reformer；
- **一个推理策略：**Keyformer；
- **Full Attention 在两个实验包中各出现一次。**

### 最可靠的可比结论

- 在 A 组内部，3 seeds、10M TinyStories token 的结果显示：Linformer 的 test PPL 最低（32.37±0.16），Full SDPA 为 36.69±1.01，Performer 为 56.30±0.83；但 Full SDPA 的吞吐远高于两个近似实现。
- 在 C 组内部，Full Attention 与因果修复版 Reformer 共享 TinyLM 主干、数据和训练协议：Full PPL=23.735，Reformer PPL=77.895，Reformer 约为 Full 的 3.28 倍；最终训练 probe 吞吐也低约 13.8%。
- Keyformer 不应与其他模型按训练 PPL 直接排名。它共享 Full Attention checkpoint，在 50% KV cache 时 PPL 从 18.782 降到 18.708，KV bytes 从 4.71 MB 降到 2.36 MB，但 decode 吞吐由 218.38 降到 151.15 tok/s，说明本次实现主要获得显存收益，没有获得即时速度收益。
- 两组 Full Attention 的绝对 PPL 不能直接比较：A 组 context=4096、3 seeds、4×RTX 4090；C 组 context=512、1 seed、RTX 4090 D，且有效 batch token 数也不同。

## 2. 两个链接包含什么

### 链接 A：Linformer / Performer

主要文件：`DELIVERABLES_README.md`、`REPORT.md`、`code/`、`results/`。

该包采用 TinyStories 和统一 TinyLM-long 风格主干，比较：

- `full_sdpa`：精确全注意力基线；
- `linformer`：沿序列维将 K/V 投影到较低维度；
- `performer`：FAVOR+ 随机特征近似 softmax attention。

核心协议：6 层、hidden=384、8 heads、head dimension=48、FFN=1536、RoPE、GPT-2 BPE、BF16、context=4096、主训练约 10M token、seeds=17/29/43。报告注明使用 4×RTX 4090，训练期间每张 GPU 独立运行任务。

冻结的架构参数为：Linformer `lin_pool=128`，Performer `n_features=384`。

### 链接 C：Reformer / Keyformer

主要文件：`c_role_tinystories_results/reports/c_role_report.md`、`aggregate/c_summary.json`、`runs/`。

该包采用相同风格的 TinyLM 主干，比较：

- `full_attention`：Full Attention reference；
- `reformer`：因果 LSH attention；
- `keyformer`：在 Full checkpoint 上进行 KV cache 选择。

核心协议：6 层、hidden=384、8 heads、FFN=1536、BF16、context=512、AdamW + cosine、10M token、seed=17、RTX 4090 D。Reformer 主配置为 bucket size=32、4 次 hashing、recent window=32、因果候选约束。

Keyformer 评估使用共享 Full checkpoint，测试 FullKV、100%、75% 和 50% cache 保留策略。

## 3. 实验协议可比性审计

| 比较对象 | 可比性 | 原因 |
|---|---|---|
| A 组 Full SDPA vs Linformer | 高 | 同一 TinyStories 数据、主干、训练预算和 3 seeds |
| A 组 Full SDPA vs Performer | 高 | 同上；只改变 attention 路径 |
| C 组 Full Attention vs Reformer | 高 | 同一 TinyLM、数据、10M token、seed 和验证流；Reformer 通过因果正确性检查 |
| C 组 FullKV vs Keyformer | 很高（推理任务内） | 同一 Full checkpoint，只改变 cache policy；100% 等价性通过 |
| A 组 Full vs C 组 Full | 中/低 | context、有效 batch、seed 数、GPU 型号和评估规模不同 |
| Linformer/Performer vs Reformer | 低 | 来自不同实验包，context=4096 vs 512，训练配置和硬件不同 |
| 所有条目按绝对 PPL 排名 | 不成立 | Keyformer 是推理策略，且 PPL 评估窗口和协议不一致 |

因此，本报告将结果分为三类：

1. **训练质量比较：**A 组内部、C 组内部；
2. **架构机制比较：**低秩、随机特征、LSH、精确注意力的优缺点；
3. **推理缓存比较：**Keyformer 与 FullKV 的质量—显存—延迟权衡。

## 4. A 组：Linformer、Performer 与 Full SDPA

### 4.1 10M token 训练结果

数据来自链接 A 的 `results/comparison_summary.csv`，为 test split、3 seeds 的均值和标准差。

| 方法 | Test NLL | Test PPL | PPL std | 训练吞吐（tok/s，报告值） |
|---|---:|---:|---:|---:|
| Full SDPA | 3.6021 | 36.6863 | 1.0129 | 165,521 |
| Linformer | **3.4772** | **32.3684** | **0.1615** | 24,617 |
| Performer | 4.0307 | 56.3035 | 0.8346 | 51,163 |

相对 A 组 Full SDPA：

- Linformer PPL 低约 11.8%，NLL 低约 0.125；
- Performer PPL 高约 53.5%，NLL 高约 0.429；
- Linformer 吞吐约为 Full SDPA 的 14.9%；
- Performer 吞吐约为 Full SDPA 的 30.9%。

这里的“Linformer 质量更好”不能解释为 Linformer 在所有任务上都优于精确注意力。它可能来自 TinyStories 的数据分布、低秩池化带来的正则化、训练随机性，以及实现层面的差异。该结果能支持的准确表述是：

> 在链接 A 的 TinyStories、TinyLM、context=4096、10M token 和 3-seed 协议下，当前 Linformer 配置获得了最低的 test PPL；但它的实测训练吞吐明显低于 Full SDPA。

### 4.2 长度吞吐趋势

链接 A 的 `throughput_fwd_bwd.csv` 给出前向+反向吞吐：

| 长度 L | Full SDPA | Performer | Linformer |
|---:|---:|---:|---:|
| 1,024 | 58,035 | 15,068 | 11,046 |
| 2,048 | 115,049 | 18,768 | 11,968 |
| 4,096 | 173,707 | 20,435 | 12,191 |
| 8,192 | 166,818 | 16,495 | 12,360 |
| 16,384 | 146,740 | 10,233 | 14,060 |

在当前 hidden size=384 和实现方式下，理论上的线性复杂度没有转化成 wall-clock 优势。原因主要是：

- Full SDPA/Flash 类 dense kernel 已经高度优化；
- Linformer 的池化和 Performer 的随机特征变换存在 Python/张量操作开销；
- 小模型下，矩阵乘法本身不是唯一瓶颈；
- 近似方法的优势需要更长序列、更大模型或更受限的显存环境才能体现。

因此，A 组同时说明了一个重要事实：**渐近复杂度更低，不等于当前硬件和当前实现一定更快。**

### 4.3 Fidelity 与质量的关系

共享初始化的 attention fidelity 为：

- Performer 在 T=1024 时 cosine=0.9155；
- Linformer 在 T=1024 时 cosine=0.6871。

训练后两者与 Full 的输出 cosine 都约为 0.994，但 PPL 仍有明显差别：Performer 56.30，Linformer 32.37，Full 36.69。这说明单层输出相似度不能替代完整语言建模质量。训练会让不同 attention 路径适应自身的近似误差，但误差仍可能在长序列 token 上累积。

A 组的正确性闸门为 11/11 PASS，包含 future invariance、padding、gradient、save/restore 和 full-window equivalence；因此其训练结果具备基本实现可信度。

## 5. C 组：Reformer 与 Full Attention

### 5.1 正确性与实现修复

C 组特别记录了 Reformer 的一次重要失败：旧版“全局哈希排序”实现的 future-invariance 最大偏差为 0.645508，超过 0.005 阈值。原因是未来 token 会改变全局排序和 bucket 边界，造成过去位置的候选集合变化。

修复版采用：

- 先限制 `key_position <= query_position`；
- 再在同一 LSH bucket 或 recent window 中选择候选；
- 保持 causal LSH mask。

修复版 future-invariance delta=0，通过有限梯度检查。这说明 Reformer 类方法的正确性不只是“有没有 mask”，还取决于 hashing、排序和候选构造是否本身具有因果性。

### 5.2 训练质量和吞吐

C 组使用相同 TinyLM、TinyStories、10M training tokens 和同一 validation stream：

| 方法 | Validation NLL | PPL | 最终训练 probe 吞吐 |
|---|---:|---:|---:|
| Full Attention | 3.16696 | **23.735** | 87,852 tok/s |
| Reformer causal LSH | 4.35536 | 77.895 | 75,769 tok/s |

Reformer 相对 Full：

- PPL 增加约 228.2%，约为 Full 的 3.28 倍；
- 最终 probe 吞吐约为 Full 的 86.2%，低约 13.8%；
- peak allocated memory 基本相同（约 2.53 GB）；
- peak reserved memory 反而略高（约 4.29 GB vs 4.14 GB）。

该结果并不否定 Reformer 的理论目标，而是说明当前实现和 context=512 的 TinyStories 任务还处在“哈希/排序开销大于稀疏收益”的区域。LSH 还会丢失未落入同一 bucket 的相关 token，导致明显质量损失。

### 5.3 Reformer 的优缺点来源

**优点：**通过 LSH 将可能相似的 query/key 放入同一 bucket，只计算候选集合，理论上减少全量两两连接；可结合 reversible layers 和 chunking 节省训练激活。

**缺点：**哈希碰撞会漏掉相关 token；多轮 hashing 增加计算；排序、bucket、padding 和不规则访存带来常数开销；全局排序如果不做因果约束会直接造成未来信息泄漏。

**设计来源：**Reformer 把“所有 token 两两比较”改成“先哈希筛选候选，再做局部 attention”。它的收益依赖序列足够长且相关性确实能被哈希捕获；它的质量代价则来自候选集合的不完整。

## 6. Keyformer：推理期 KV-cache 策略

Keyformer 不重新训练一个独立 backbone，而是在 C 组 Full checkpoint 上进行推理期缓存压缩。因此它应该单独回答：

> 在不大幅损失语言质量的情况下，能否减少自回归解码所需的 KV cache？

### 6.1 质量—显存结果

来自 `c_summary.json` 的 10,240-token、40-window 评估：

| 策略 | Cache ratio | PPL | KV bytes | Decode tok/s |
|---|---:|---:|---:|---:|
| FullKV | 100% | 18.782 | 4,709,376 | 218.38 |
| Keyformer-100 | 100% | 18.782 | 4,709,376 | 187.10 |
| Keyformer-75 | 75% | 18.760 | 3,538,944 | 166.82 |
| Keyformer-50 | 50% | 18.708 | 2,359,296 | 151.15 |

Keyformer-100 与 FullKV 的 NLL delta=0，100% 等价性通过。50% cache 时：

- KV bytes 减少 50%；
- PPL 反而低约 0.40%，属于当前评估噪声或缓存选择对该窗口的偶然正则化，不能解读为压缩必然提升质量；
- decode 吞吐下降约 30.8%；
- prefill median 从约 5.3 ms 增加到约 39.1 ms。

因此本次 Keyformer 实验的主要结论是：

> Keyformer 已证明可以在当前 Full checkpoint 上把 KV cache 物理容量降到 50%，同时保持近乎相同的 TinyStories 评估 PPL；但当前选择、gather 和 cache update 开销抵消了 decode 速度收益。

### 6.2 Keyformer 的优缺点来源

**优点：**只改推理阶段，不需要重新训练 backbone；利用 token 重要性选择高价值历史，同时保留 recent window；物理 gather 确实降低 KV tensor 大小。

**缺点：**每一步需要计算重要性、选择 token 和重排 cache；短或中等上下文时选择开销可能超过注意力节省；过度压缩会丢失远程事实；它不降低训练阶段的全注意力成本。

**设计来源：**Keyformer 假设历史 token 的价值分布是不均匀的，少数 token 对下一步预测更重要。因此它压缩的是 KV cache，而不是重新定义训练 attention。

## 7. 六个结果条目的综合对照

| 条目 | 任务阶段 | 质量证据 | 效率证据 | 当前最稳妥结论 |
|---|---|---|---|---|
| A-Full SDPA | 训练 | PPL=36.69±1.01 | A 组吞吐最高 | 精确语义和工程吞吐基线 |
| A-Linformer | 训练 | PPL=32.37±0.16 | 4K 约 12.2K tok/s | 当前 TinyStories 配置质量好，但实现较慢 |
| A-Performer | 训练 | PPL=56.30±0.83 | 4K 约 20.4K tok/s | 近似误差和实现开销都较明显 |
| C-Full Attention | 训练 | PPL=23.735 | 10M probe 约 87.9K tok/s | C 组精确参考，不与 A-Full 绝对 PPL 直比 |
| C-Reformer | 训练 | PPL=77.895 | 约 75.8K tok/s | 因果修复后可运行，但 context=512 下质量和速度均未占优 |
| C-Keyformer | 推理 cache | 50% cache PPL=18.708 | 151.2 tok/s；KV -50% | 主要节省显存，当前实现不保证加速 |

### 可以形成的局部排名

**A 组训练质量：**Linformer > Full SDPA > Performer（PPL 越低越好）。  
**A 组训练吞吐：**Full SDPA > Performer > Linformer（以 4K 报告值）。  
**C 组训练质量：**Full Attention > Reformer。  
**C 组训练吞吐：**Full Attention > Reformer。  
**C 组 KV 显存：**Keyformer-50 < Keyformer-75 < FullKV。  
**C 组 Keyformer 质量保持：**Keyformer-50/75 与 FullKV 基本持平，但这是同一 checkpoint 的 policy 对照，不是模型架构排名。

## 8. 为什么不能做六模型绝对总排名

### 8.1 context 不一致

A 组训练 context=4096，C 组训练 context=512。长上下文越长，近似 attention 的候选裁剪和信息损失越可能显现；短 context 则可能让 Full Attention 的二次矩阵尚未成为瓶颈。

### 8.2 训练预算不一致

A 组有效 batch token 约为 32,768，C 组约为 8,192；A 组使用 3 seeds，C 组主要结果为 seed=17。训练动态和最终方差因此不同。

### 8.3 硬件和软件不一致

A 组记录为 4×RTX 4090，C 组为 RTX 4090 D；软件栈分别以 A 包的训练说明和 C 包的 `environment.json` 为准。绝对 tok/s 不应跨包直接比较。

### 8.4 Keyformer 不是同类对象

Linformer、Performer、Reformer 改变了训练时 attention 计算；Keyformer 使用已经训练好的 Full checkpoint，仅在生成时压缩历史 K/V。它不能与其他方法按“训练 PPL/训练吞吐”排在同一列。

### 8.5 质量评估规模不同

A 组报告 test split 约 1,048,576 token 的 3-seed 汇总；C 组 Reformer/Full 使用约 4,763,648 valid tokens，而 Keyformer 使用 8,192 或 10,240 token 的窗口化评估。PPL 数值的统计稳定性和含义不同。

## 9. 架构选择建议

| 需求 | 推荐 | 原因 |
|---|---|---|
| 短中序列、要求精确语义、追求实际吞吐 | Full SDPA/Flash | 当前实验中 kernel 常数优势明显，质量最稳 |
| 固定长度或可充分训练低秩投影 | Linformer | A 组 TinyStories PPL 最低，但需接受当前实现吞吐损失 |
| 超长序列、允许随机特征近似 | Performer | 理论上近似线性，但应增加 feature 数和长度后再评估 |
| 超长序列、相关性适合哈希分桶 | Reformer | 需要严格 causal LSH、足够长序列和更成熟 kernel |
| 自回归生成、KV cache 是主要显存瓶颈 | Keyformer | 50%–75% cache 可保持质量，但需优化选择/gather 才可能加速 |

## 10. 这两组实验已经证明和没有证明的事情

### 已经证明

- Linformer、Performer 和修复后的 Reformer 都可以在 TinyStories/TinyLM 框架下运行并通过基本正确性检查；
- A 组中 Linformer 在当前配置下取得了最佳 test PPL；
- Reformer 的因果实现需要专门约束，旧版全局排序确实会导致未来信息泄漏；
- Keyformer 的物理 cache 压缩和 100% FullKV 等价性已经得到端到端验证；
- 当前实现中，Full SDPA 的实际训练吞吐明显高于近似 attention；
- Keyformer 的 KV bytes 可以近似按 cache ratio 线性下降。

### 尚未证明

- 没有证明 Linformer 在所有数据集、模型规模和 context 下都优于 Full Attention；
- 没有证明 Performer 或 Reformer 在足够长序列上一定更快；
- 没有证明 Keyformer 一定降低 decode latency；
- 没有证明各方法在 8K/16K 长故事中具有可靠远距离记忆能力；
- 没有形成严格的六路统一 Pareto 前沿。

A 组的 Copy/Passkey/Retrieve 零样本长度外推 exact-match 均为 0，说明只在 TinyStories 正文上训练的模型没有自动学会随机合成任务。这不是架构失败，而是任务没有进行对应的机制微调。要评价长距离能力，必须单独训练或微调包含远距离复制、passkey、跨章节人物属性保持的任务。

## 11. 建议的统一复现实验

如果希望最终得到真正可发表的六路（或五种独立方法）横向比较，建议重新冻结以下协议：

1. 所有方法使用同一 TinyStories revision、GPT-2 tokenizer manifest、相同 train/validation/test 故事边界；
2. 统一 TinyLM：6 层、hidden=384、8 heads、FFN=1536、RoPE、causal、BF16；
3. 统一 context，例如 512、2048、4096、8192；
4. 统一有效 batch token、训练 token 数和 seeds=17/29/43；
5. Full Attention 使用同一张 GPU 重新运行，且同时记录 SDPA backend；
6. 训练方法只比较 Full、Linformer、Performer、Reformer；Keyformer 单独作为 inference cache 实验；
7. 统一记录 token-weighted NLL/PPL、median/p90 latency、fwd+bwd throughput、peak allocated/reserved、OOM 和实际候选/状态元素数；
8. 增加 local copy、passkey、cross-segment retrieval 和长故事人物一致性任务；
9. 生成结果时把 `trained_attention` 与 `inference_cache_policy` 分成两张主表。

## 12. 总结

两个云盘实验包可以组成一份有价值的 TinyStories 高效 Transformer 总结，但正确的结论不是“六个模型谁绝对第一”，而是：

> 不同方法从不同设计维度换取效率：Linformer 压缩序列维、Performer 近似 softmax kernel、Reformer 用 LSH 筛选候选连接、Full SDPA 优化精确 kernel，而 Keyformer 压缩自回归 KV cache。当前 TinyStories 结果显示，Full SDPA 在实际吞吐上最强，Linformer 在 A 组质量上最好，Reformer 在 context=512 下尚未体现收益，Keyformer 则以约 50% 的 KV 容量换取近乎不变的评估 PPL，但当前实现仍有额外选择开销。

因此，这两组结果适合支持“架构设计—效率—质量代价”的机制性分析；若要宣称六路统一排名，还需要相同 context、相同 batch、相同 seeds、相同硬件、相同评估规模和严格分离训练架构与推理策略。

## 附录：原始材料

- 链接 A 报告：`REPORT.md`
- 链接 A 交付说明：`DELIVERABLES_README.md`
- 链接 A 结果：`results/comparison_summary.csv`、`results/throughput_fwd_bwd.csv`、`results/fidelity_shared_init.csv`、`results/fidelity_trained.csv`、`results/correctness.jsonl`
- 链接 C 报告：`c_role_tinystories_results/reports/c_role_report.md`
- 链接 C 汇总：`c_role_tinystories_results/aggregate/c_summary.json`
- 链接 C Reformer 训练记录：`c_role_tinystories_results/runs/reformer_seed17_main_causal_lsh_b32h4r32/metrics.jsonl`
- 链接 C Full 训练记录：`c_role_tinystories_results/runs/full_attention_seed17_main/metrics.jsonl`
