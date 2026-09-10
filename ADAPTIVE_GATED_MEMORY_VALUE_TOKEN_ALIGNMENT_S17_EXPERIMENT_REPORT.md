# Adaptive Gated Memory 的 value-span 与 memory-to-token 对齐实验报告

更新时间：2026-09-10  
实验阶段：第一阶段结构改进（尚未加入 pointer/copy head）  
实验对象：TinyLM + SWA-sink + adaptive gated fact memory，seed 17

## 摘要

旧 Gated 模型能以 1 个槽位完整保留并检索目标事实，但 held-out 验证 Exact Match（EM）只有 52.43%。已有诊断显示，正确答案首 token 往往已位于词表前五，瓶颈主要在“被检索的语义记忆如何精确映射回答案 token”。

本实验保留原语义记忆与门控控制器，增加四项最小改动：精确 value-span 标签、value 起点/长度预测头、独立 lexical value 表示，以及 position-conditioned memory-to-token 辅助头。主模型仍通过原 LM head 自回归生成，辅助头不直接替换输出，也没有使用 pointer/copy。

在与旧实验相同的 seed、TinyStories 骨干、训练步数、噪声/延迟划分和验证规模下，新模型在全部 288 条 held-out 验证样本上取得 100% EM；目标存活率、Recall@1、Recall@4、MRR、value-span exact 和 memory-token exact 也均为 100%。与旧 Gated 相比，EM 提高 47.57 个百分点，完整状态增加 14,796 B（0.61%），参数增加 305,293（0.96%），训练耗时增加 4.31%。该结果强烈支持“词级对齐是 seed-17 旧模型的主要瓶颈”，但由于目前只有一个 seed、16 个封闭词汇值和合成模板，不能据此宣称已解决开放域精确记忆。

## 一、研究问题与假设

### 1.1 旧模型的问题

旧模型的正式结果为：

| 指标 | 旧 Gated，seed 17 |
|---|---:|
| EM | 52.43% |
| 目标事实存活率 | 100% |
| Recall@1 | 100% |
| Recall@4 | 100% |
| MRR | 1.0000 |
| 平均 active slots | 1.00 |

因此，失败发生在已经正确写入和检索之后。旧路径把完整事实 span 平均为一个语义向量，同时把完整 payload 的 token embedding 再次平均。它能表达“事实大意”，但没有告诉模型哪个子串是必须精确回答的 value/object，也没有直接施加 `memory representation → answer token` 的监督。

### 1.2 可检验假设

若词级映射确实是主要瓶颈，则在不改变槽位预算、门控目标、语义 reader 和局部注意力预算的前提下：

1. 显式监督 value 起点与长度应能把目标词从完整事实中分离；
2. 对 value 表示施加 token CE 应使正确答案首 token 从“前五但不稳定”变成稳定第一；
3. EM 应提高，而存活率、Recall 和 active slots 不应恶化；
4. 状态与训练开销应保持有界。

## 二、实现机制

### 2.1 保留不变的路径

下列部分与旧 Gated 实验相同：

```text
TinyLM backbone
  ├── local attention：4 sinks + 124 recent
  ├── 完整事实 span → semantic key
  ├── 完整事实 span → semantic value
  ├── write / retention / merge / bounded slots
  ├── semantic Top-k reader
  └── 原自回归 LM head
```

其中完整事实表示仍负责关系级语义与检索，避免将模型退化成只保存答案 token 的查表器。

### 2.2 新增 value-span 标签

例如：

```text
Store: Bob likes [blue].
```

生成器先在原始字符串中定位 `blue` 的字符范围，再使用 GPT-2 fast tokenizer 的 offset mapping 映射到 token 起点和长度。这样不会依赖容易误配的 token 子序列搜索，并能处理 token 边界差异。

训练字段新增：

- `value_start_targets`：value 的 token 起点；
- `value_length_targets`：value 的 token 长度。

### 2.3 新增 value-span extractor

`AtomicFactExtractor` 新增：

- `value_start_head`：在当前事实 span 内选择 value 起点；
- `value_length_head`：预测从起点开始的 value 长度；
- `lexical_projection`：把 value span 的 hidden states 平均后投影成独立 lexical value。

新路径为：

```text
完整事实 hidden states
  → value 起点/长度
  → 只聚合 value token
  → lexical projection
  → lexical value
```

完整 fact span 和 value span 的职责不同：前者用于理解与检索，后者用于精确词汇恢复。

### 2.4 槽位状态的变化

每个用户事实槽位保留原字段，并增加：

- `lexical_values`：一个 hidden-size 连续向量；
- `value_payload_ids`：value span 的可审计 token IDs；
- `value_payload_mask`：value span 的有效位置。

这些字段与原 payload 一同经历写入、覆盖、batch 重排、detach 与 reset。它们不增长随时间累积的 KV；开销仍由固定槽位数和固定 payload 上限决定。

### 2.5 新的 fusion 输入

本实验使用 `semantic_lexical`：

```text
memory value sent to fusion
  = 原 semantic value + 新 lexical value
```

原来的 `combined`、`payload_only` 和 `value_only` 路径仍保留。不开启 `value_token_alignment` 时模型保持旧默认行为。

### 2.6 memory-to-token 辅助头

辅助头接收 lexical value，并为最多 `payload_tokens=12` 个答案位置形成位置化表示：

```text
lexical value
  + learned answer-position embedding
  → linear projection
  → layer norm
  → 与 TinyLM token embedding 共享的词表投影
  → token cross-entropy
```

加入位置 embedding 是为了支持多 token value。当前词表中大多数答案在 query 中是不带前导空格的一个 token，但 `navy` 为两个 token；模型必须按顺序恢复每个目标 token。

辅助监督从两个位置施加：

1. 写入轮 candidate-local loss：使用人工标注的 value span，直接把梯度传给 lexical encoder；
2. 查询前 retrieved-memory loss：使用槽位中保存的 lexical value，监督从真实记忆状态到 token 的映射。

第二项在跨轮 TBPTT detach 后主要训练 token decoder；第一项保证早期 value encoder 也能获得直接梯度。主输出仍来自普通 LM logits，因此最终 EM 不是直接读取辅助头的预测。

## 三、代码保留与可复现性

用户要求只保留本次会修改的部分。修改前只备份了以下 7 个文件：

```text
adaptive_gated_fact_memory/src/adaptive_fact_memory/memory.py
adaptive_gated_fact_memory/src/adaptive_fact_memory/model.py
adaptive_gated_fact_memory/src/adaptive_fact_memory/state.py
adaptive_gated_fact_memory/scripts/synthetic_memory.py
adaptive_gated_fact_memory/scripts/synthetic_memory_stress.py
adaptive_gated_fact_memory/scripts/train_memory_stress.py
adaptive_gated_fact_memory/tests/test_prototype.py
```

快照位于：

```text
code_snapshots/pre_value_token_alignment_20260910_utc/
```

其中包含创建时间、commit `78fed2baf53e3756ba52979b15d0ec3ed50aa5bc` 和逐文件 SHA-256。上述 7 个文件在修改前相对该 commit 没有 diff。已有报告、旧 checkpoints 和其他工作区改动均未覆盖或回退。

正式 run：

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_stress/
selective_stress_v3_gated_value_token_s17_20260910/
```

最终 checkpoint SHA-256：

```text
affe5ad1463c69148df37c9d058336fbf25a898cae65d379864796f1adf17fc5
```

配置哈希：

```text
50635848b3fcf290223cae2eafacf6ea959ca9aae6235566d76d04c747c95dba
```

## 四、实验设置

### 4.1 硬件与软件

| 项目 | 设置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| PyTorch | 2.12.1+cu130 |
| CUDA runtime | 13.0 |
| 计算精度 | CUDA bfloat16 autocast |
| TF32 | 关闭 |

### 4.2 TinyLM 和记忆配置

| 项目 | 设置 |
|---|---:|
| Transformer layers | 6 |
| hidden size | 384 |
| attention heads | 8 |
| FFN size | 1536 |
| local KV budget | 128 = 4 sinks + 124 recent |
| user memory slots | 8 |
| memory read Top-k | 4 |
| max candidates/user round | 1 |
| payload token budget | 12 |
| merge threshold | 0.999 |
| train write/retention threshold | 0.1 / 0.1 |
| eval write/retention threshold | 0.5 / 0.5 |
| memory fusion | semantic value + lexical value |

父模型是相同的 TinyStories seed-17 骨干：

```text
tinystories_backbone_s17_10m_20260908/checkpoint.final.pt
SHA-256: d995248587a5e535af59d850f36a5bc01925c54bd9dff213f0f18bfb26f2b78f
```

### 4.3 数据协议

每条样本遵循：

```text
durable user fact
→ 0/1/2/4 个训练期 temporary user facts
→ TinyStories token delay
→ query and answer
```

- 训练 durable cues：`Remember`、`Keep`、`Save`；
- 验证 durable cue：held-out `Store`；
- 训练 noise cues：`Temporary`、`Skip`、`Ignore`；
- 验证 noise cue：held-out `Discard`；
- value 集合：16 个颜色词；
- TinyStories 只作为自然文本延迟/干扰，不为答案提供标签。

训练条件：

- delays：1、2、4 个 segment；
- noise rounds：0、1、2、4；
- segment length：128 tokens。

验证条件：

- delays：1、4、16 个 segment；
- noise rounds：0、2、4、8；
- 12 个交叉条件；
- 每个条件 24 条，共 288 条。

最长验证延迟约为 `16 × 128 = 2048` 个 TinyStories tokens，明显超过 124-token recent window。

### 4.4 优化设置

| 项目 | 设置 |
|---|---:|
| seed | 17 |
| optimizer | AdamW |
| betas | (0.9, 0.95) |
| weight decay | 0.1 |
| backbone LR | 3e-5 |
| memory/new heads LR | 3e-4 |
| optimizer steps | 2000 |
| physical batch | 2 |
| gradient accumulation | 2 |
| effective batch | 4 |
| gradient clip | 1.0 |
| value-start weight | 0.50 |
| value-length weight | 0.25 |
| candidate memory-token weight | 0.50 |
| retrieved memory-token weight | 0.50 |

其他旧损失权重保持不变：LM 1.0、fact start 0.75、fact length 0.25、write 0.75、read 1.0、key alignment 0.50、retention 0.10、budget 0.01。

## 五、测试逻辑与结果

### 5.1 单元测试

测试命令：

```bash
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 -m unittest discover \
  -s adaptive_gated_fact_memory/tests -p 'test_*.py' -v
```

结果：18/18 通过。新增测试分别验证：

1. value span 的 token 边界正确且 padding 不进入 value；
2. lexical value、value IDs 和 mask 能随槽位写入；
3. 新字段能随 batch 重排并被 reset；
4. memory-token CE 能向 `lexical_projection` 和 token decoder 回传非零有限梯度；
5. 默认不开开关时仍走旧 value/payload 路径。

### 5.2 GPU 冒烟实验

正式训练前运行了独立 2-step smoke run：

```text
selective_stress_v3_value_token_smoke_s17_20260910
```

它完整通过训练、保存和 12 条最小评测，峰值 allocated 显存约 1.28 GB。该 run 只用于程序正确性检查，不作为科学结果。

### 5.3 正式训练曲线

训练共处理 8,000 个样本实例。早期随机新头使第一步总损失和梯度较大，但所有值有限；之后稳定收敛：

| Step | LM loss | candidate token CE | retrieved token CE | token exact（当前批） | active slots |
|---:|---:|---:|---:|---:|---:|
| 113 | 2.8570 | 3.6907 | 3.6907 | 100% | 1.00 |
| 479 | 0.2174 | 0.3609 | 0.3609 | 100% | 1.00 |
| 1013 | 0.00339 | 0.00432 | 0.00432 | 100% | 1.00 |
| 1541 | 0.000142 | 0.000059 | 0.000059 | 100% | 1.00 |
| 2000 | 0.0000163 | 0.00000438 | 0.00000438 | 100% | 1.00 |

在约第 595 步，包含困难分词的一个批次出现 token accuracy 83.3%、value exact 75%，随后继续收敛。最近 100 步在第 1222 步时 value exact 平均已为 100%。这说明多 token value 确实被训练，而非被评测忽略。

## 六、正式验证结果

### 6.1 总体结果

| 指标 | V3 value-token Gated |
|---|---:|
| EM | **100.00%** |
| 目标事实存活率 | **100.00%** |
| Recall@1 | **100.00%** |
| Recall@4 | **100.00%** |
| MRR | **1.0000** |
| value start accuracy | **100.00%** |
| value length accuracy | **100.00%** |
| value span exact | **100.00%** |
| memory-token token accuracy | **100.00%** |
| memory-token sequence exact | **100.00%** |
| teacher-forced answer NLL | 0.0000195 |
| answer first-token NLL | 0.0000297 |
| answer first-token mean rank | **1.0000** |
| answer first-token Top-5 | **100.00%** |
| answer first-token logit margin | **+13.5763** |
| 平均 active slots | **1.00** |
| noise accept rate | **0.00%** |

全部 288 条生成都以正确答案 token 序列开头，随后立即生成 EOS。目标值覆盖全部 16 个 value；其中 `navy` 的答案分词为两个 token，也全部正确。

### 6.2 分条件结果

| Noise rounds | Delay segments | 样本数 | EM | Survival | Recall@1 | Active slots |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 24 | 100% | 100% | 100% | 1.00 |
| 0 | 4 | 24 | 100% | 100% | 100% | 1.00 |
| 0 | 16 | 24 | 100% | 100% | 100% | 1.00 |
| 2 | 1 | 24 | 100% | 100% | 100% | 1.00 |
| 2 | 4 | 24 | 100% | 100% | 100% | 1.00 |
| 2 | 16 | 24 | 100% | 100% | 100% | 1.00 |
| 4 | 1 | 24 | 100% | 100% | 100% | 1.00 |
| 4 | 4 | 24 | 100% | 100% | 100% | 1.00 |
| 4 | 16 | 24 | 100% | 100% | 100% | 1.00 |
| 8 | 1 | 24 | 100% | 100% | 100% | 1.00 |
| 8 | 4 | 24 | 100% | 100% | 100% | 1.00 |
| 8 | 16 | 24 | 100% | 100% | 100% | 1.00 |

没有观察到随延迟或 noise 增长的退化。特别是 8 个 temporary facts + 16 个 delay segments 的最难条件仍保持 1 个槽位与 100% EM。

### 6.3 Checkpoint 回扫

| Step | EM | answer first-token rank | logit margin | teacher-forced NLL | auxiliary value exact |
|---:|---:|---:|---:|---:|---:|
| 500 | 100% | 1.00 | +6.8794 | 0.133411 | 94.79% |
| 1000 | 100% | 1.00 | +9.1122 | 0.001828 | 100% |
| 1500 | 100% | 1.00 | +11.4877 | 0.000148 | 100% |
| 2000 | 100% | 1.00 | +13.5763 | 0.000019 | 100% |

主生成在 500 步已全部正确；继续训练主要提高答案置信度并使辅助头对多 token value 完全收敛。因此，本任务上 2000 步不是获得 EM 的必要条件，但提供了更大的 logit margin。

## 七、与原架构和基线对比

| 条件 | EM | Survival | Recall@1 | MRR | Active slots | Noise accept |
|---|---:|---:|---:|---:|---:|---:|
| SWA-sink only | 5.21% | N/A | N/A | N/A | 0.00 | N/A |
| 旧 Gated | 52.43% | 100.00% | 100.00% | 1.0000 | 1.00 | 0.00% |
| Fixed-LRU | 74.65% | 92.01% | 90.28% | 0.9103 | 4.25 | 75.00% |
| **V3 value-token Gated** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.00** | **0.00%** |

相对变化：

- 对旧 Gated：EM +47.57 个百分点；存活、检索和槽位数不变；
- 对 Fixed-LRU：EM +25.35 个百分点；平均少用 3.25 个槽位；
- 对 SWA-only：EM +94.79 个百分点。

这组比较把两个问题分开了：

1. 旧 Gated 已经解决“保留什么、如何检索”；
2. 新 value-token 路径解决“检索后如何精确回答”。

Fixed-LRU 旧结果在低噪声时有较高 EM，但在 8-noise 条件下因容量竞争而丢失目标。新模型同时保留 Gated 的选择性压缩和精确输出。

## 八、效率与存储开销

### 8.1 参数与状态

| 项目 | 旧 Gated | V3 | 变化 |
|---|---:|---:|---:|
| 总参数量 | 31,854,750 | 32,160,043 | +305,293（+0.96%） |
| 完整 state bytes | 2,408,906 | 2,423,702 | +14,796（+0.61%） |
| 1 个 active user slot logical bytes | 3,221 | 4,865 | +1,644 |
| 平均 active slots | 1.00 | 1.00 | 0 |

单 active slot 的逻辑字节增幅为 51.0%，因为增加了一个 384 维 lexical vector 和一组 value IDs/mask；但整个流式状态还包含六层固定 SWA cache，所以总状态只增加 0.61%。更重要的是，开销由 8 个固定槽位封顶，不随对话长度线性增长。

### 8.2 训练效率

| 项目 | 旧 Gated | V3 | 变化 |
|---|---:|---:|---:|
| 2000-step 训练时间 | 1323.0 s | 1380.1 s | +57.1 s（+4.31%） |
| 训练吞吐 | 6.047 examples/s | 5.797 examples/s | -4.14% |
| 峰值 allocated 显存 | 2.673 GB | 2.682 GB | +0.0096 GB（+0.36%） |
| 峰值 reserved 显存 | 6.648 GB | 6.702 GB | +0.0545 GB（+0.82%） |

### 8.3 推理延迟补测

另以 final checkpoint 做 96 条单样本端到端计时，每个 12 条件各 8 条；排除模型加载并先 warm-up 一条。计时包含 durable/noise/filler/query 编码、记忆操作和 greedy answer：

| 指标 | V3 |
|---|---:|
| 平均时间/样本 | 0.2670 s |
| 中位时间/样本 | 0.2627 s |
| P95 时间/样本 | 0.4012 s |
| 吞吐 | 3.746 examples/s |
| 推理峰值 allocated | 0.596 GB |
| 推理峰值 reserved | 1.734 GB |

该补测不是与旧模型同一进程中的配对 latency benchmark，因此只能作为 V3 的绝对运行记录，不能据此宣称推理加速或减速。

## 九、结果解释

### 9.1 对根因假设的支持

证据链完整一致：

```text
旧模型：Survival 100%, Recall@1 100%, EM 52.43%
       正确 token 平均 rank 2.04，Top-5 97.57%

加入 value span + token alignment：
       Survival/Recall/slots 不变
       正确 token rank 变为 1.00，margin +13.58
       EM 变为 100%
```

因此，在当前 seed-17 合成任务内，主要瓶颈确实是从压缩语义记忆到具体答案 token 的映射，而不是目标事实保留、reader 或融合门过高/过低。

### 9.2 对门控架构目标的支持

新模型没有靠接受更多 temporary facts 换取回答准确率：noise accept 仍是 0%，平均槽位仍是 1。它在最强噪声和最长延迟下同时达到：

- 只保留一个重要事实；
- 丢弃八个临时事实；
- 精确恢复答案；
- 状态开销固定。

这直接支持“模型自主选择重要内容 + 有界语义记忆 + 精确值恢复”的完整设计方向。

### 9.3 为什么辅助头不是答案泄漏

- value 标签只出现在早期 durable fact 中，来自本来就应被记忆的输入；
- query round 的主 LM 不接收答案标签作为检索 key；
- auxiliary head 只参与训练损失，推理时主答案仍由原自回归 LM head 生成；
- 验证使用未见的 `Store`/`Discard` cues、最长约 2048-token delay；
- 每条生成都在正确答案之后正常生成 EOS。

不过，这仍是强监督方案：训练器明确知道事实中的 value 边界与目标答案。论文应把它描述为 supervised value extraction/alignment，不能声称模型在无标注开放文本中自动发现任意 value。

## 十、局限性与论文表述边界

1. **只有一个 seed。** 100% 是 seed-17 的点估计，尚无均值 ± 标准差。
2. **封闭答案集合。** 只有 16 个颜色值，模型可能学习颜色分类与模板映射；还未证明对实体、数字、日期或任意字符串泛化。
3. **模板规模较小。** 虽然验证 cue held-out，但句法关系固定为 `name likes value`。
4. **value span 使用人工监督。** 实际对话没有天然 offset 标签，需要自动标注、弱监督或联合学习。
5. **没有新模型内部消融。** 本次同时加入 value span、lexical fusion 和 token auxiliary loss，不能单独量化每项贡献。
6. **尚未加入 pointer/copy。** 当前 100% 表明本任务不需要 copy head；面对 OOV、长数字或罕见多 token 字符串时仍可能需要显式 copy。
7. **旧基线缺少新诊断指标。** 旧 run 没记录 teacher-forced NLL、logit margin 和 value-span 指标，因此这些指标只能在 V3 内或与既有离线诊断比较。
8. **推理时使用 argmax。** 尚未研究采样温度、beam search 或更开放生成设置。

因此目前可写的结论是：

> 在受控的 seed-17 selective-memory stress task 中，显式 value-span 表示与 memory-to-token 辅助监督消除了已识别的词级映射瓶颈，同时保持了 Gated memory 的单槽位选择性压缩。

目前不能写：

> 该机制已经解决所有长对话中的精确事实恢复，或已在统计意义上普遍优于所有基线。

## 十一、下一步实验

优先级建议如下：

1. 以完全相同配置训练 seed 29 和 43，报告 3-seed 均值 ± 标准差；
2. 做最关键的结构消融：
   - 旧 Gated；
   - value span + lexical fusion，但 token-loss weight=0；
   - value span + token loss，但 fusion 仍用旧 combined；
   - 完整 V3；
3. 扩展 value 类型与未见词测试：随机人名、日期、数字、罕见多 token 字符串；
4. 扩展关系和 query paraphrase，验证不是颜色分类器；
5. 仅当上述测试仍表现出精确恢复瓶颈时，再加入 pointer/copy head，并单独量化状态、延迟和准确率开销。

## 十二、实验产物

正式 run 目录包含：

- `config.resolved.json`：全部配置、环境和代码哈希；
- `metrics.jsonl`：2000 步逐步训练指标；
- `checkpoint.step_{500,1000,1500,2000}.pt`；
- `checkpoint.final.pt`；
- `summary.json`：288 条完整结果与逐条件聚合；
- `diagnostics_checkpoint_sweep.json`：四个 checkpoint 的回扫结果；
- `inference_benchmark.json`：96 条推理计时结果。

本报告为新增文件，没有修改原结果报告或根因诊断报告。
