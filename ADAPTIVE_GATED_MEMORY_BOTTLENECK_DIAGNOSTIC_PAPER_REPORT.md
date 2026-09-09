# 自适应门控事实记忆的瓶颈诊断实验报告

## 摘要

本报告记录一组围绕 seed=17 的诊断实验。研究目标不是简单比较哪个模型的 Exact
Match（EM）最高，而是回答一个更基础的问题：**当模型已经找到正确的长期事实，
但最终答案仍然经常错误时，问题究竟发生在哪一层？**

实验先确认长期记忆是否正常工作，再分别检查门控权重、训练进程、提示模板迁移、
答案 tokenization，以及记忆融合强度。结果显示：

1. Gated 模型在 288 条验证样本上目标事实存活率、Recall@1、Recall@4 和 MRR
   均为 100%，所以“没有记住”或“没有检索到”不是主要瓶颈。
2. 写入门和保留门确实接近 0/1 饱和，但在本任务中没有产生错误写入或错误保留。
3. 融合门固定扫描的最佳点约为 0.5；然而重新训练固定门=0.5 的模型只把 EM 从
   52.43% 提高到 55.90%，说明门值过高可能有轻微影响，但不是根本原因。
4. 同一 checkpoint 在训练提示模板上的 EM 为 78.13%，在验证改写模板上为
   52.43%，提示模板/词汇迁移是重要因素。
5. 把答案统一改成带前导空格的 token 后，EM 反而降到 20.14%。因此 tokenization
   确实值得研究，但“简单地加一个前导空格”不是有效修复。
6. 500→1000→1500→2000 step 的 EM 持续上升，说明当前 2,000 step 还没有明显
   过早过拟合；不过最终答案生成仍未训练到足够稳定。

综合判断：当前最可信的根因是**记忆内容被压缩并融合后，没有稳定地映射到具体答案
词的输出 logit**，同时受到验证模板迁移和 GPT-2 tokenization 的影响。门控饱和是
需要监控的风险，但目前没有证据表明它是主要退化来源。

---

## 1. 研究问题和因果分解

一次答案生成可以分成五个环节：

```text
输入事实
  → 写入门决定是否保存
  → 保留门决定是否继续保存
  → 语义 reader 找到相关槽位
  → fusion 把槽位内容送入 decoder
  → LM head 把 decoder 状态转换为答案 token
```

如果最终 EM 低，至少有以下几种不同解释：

- **写入失败**：目标事实根本没有进入记忆；
- **保留失败**：事实在长文本或噪声中被清除；
- **检索失败**：事实存在，但查询没有找到它；
- **融合失败**：事实找到了，但没有有效影响 decoder；
- **表示/解码失败**：事实影响了 decoder，但具体答案词的 logit 仍然输给了
  其他词；
- **泛化失败**：训练时学会了某种模板，验证时换模板后不能使用同一机制；
- **优化失败**：训练步数或多任务损失尚未让答案路径充分收敛。

本报告的每个测试都只改变一个主要因素，尽量把这些解释分开。

---

## 2. 所有实验共同使用的协议

### 2.1 模型

三个主条件和两个重训变体都以同一个 TinyStories seed-17 backbone 为起点：

| 项目 | 设置 |
|---|---|
| Transformer layers | 6 |
| hidden size | 384 |
| attention heads | 8 |
| FFN size | 1,536 |
| 总参数 | 31,854,750，约 31.85M |
| tokenizer | GPT-2，词表 50,257 |
| 局部 KV | 128：4 sinks + 124 recent |
| 长期用户槽位 | 8 |
| reader | semantic top-k=4 |
| 每轮最大候选 | 1 |
| merge threshold | 0.999 |
| 数值精度 | RTX 4090 BF16 autocast |

父 checkpoint：

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_tinystories/
tinystories_backbone_s17_10m_20260908/checkpoint.final.pt
```

父 checkpoint SHA-256：

```text
d995248587a5e535af59d850f36a5bc01925c54bd9dff213f0f18bfb26f2b78f
```

### 2.2 结构化压力数据

每个样本遵循：

```text
耐久事实 → 临时事实噪声 → TinyStories filler → 查询 → 答案
```

训练条件：

- 临时事实数量：0、1、2、4；
- filler delay：1、2、4 个 128-token 段；
- 总训练步数：2,000；
- batch size=2，gradient accumulation=2；
- 每个优化 step 处理 4 个样本，共 8,000 个训练样本。

验证条件：

- 临时事实数量：0、2、4、8；
- filler delay：1、4、16 个 128-token 段；
- 每个条件 24 条样本；
- 共 4×3×24=288 条验证样本。

训练与验证使用不同提示词：

| 语义 | 训练提示 | 验证提示 |
|---|---|---|
| 耐久事实 | `Remember`、`Keep`、`Save` | `Store` |
| 临时事实 | `Temporary`、`Skip`、`Ignore` | `Discard` |

这个设计一方面测试了门控能否区分相同事实形式下的不同重要性，另一方面也测试了
从训练提示到未见提示的迁移。

### 2.3 资源和公平性

所有完整重训均使用 2,000 step、相同父 checkpoint、相同数据协议和同一 RTX 4090
规格。两次重训并行运行在物理 GPU 1 和 GPU 0；进程内由于 `CUDA_VISIBLE_DEVICES`
映射，均显示为 `cuda:0`。所有新实验使用独立 run ID，没有覆盖旧结果。

---

## 3. 实验 A：三种记忆架构主比较

### 3.1 目的

先确定长期记忆是否必要，并比较“学习式选择性写入”和“固定写入+LRU”的差别。

### 3.2 唯一变量

| 条件 | 长期记忆策略 |
|---|---|
| SWA-only | 不使用事实记忆；只有 4 sinks + 124 recent |
| Fixed-LRU | 所有有效候选均写入；容量满时 LRU 淘汰 |
| Gated | 学习式 write/retention gate；拒绝临时事实 |

### 3.3 结果

| 条件 | EM | 目标存活率 | Recall@1 | Recall@4 | MRR | 平均活动槽位 |
|---|---:|---:|---:|---:|---:|---:|
| Gated | 52.43% | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.00** |
| Fixed-LRU | **74.65%** | 92.01% | 90.28% | 92.01% | 0.9103 | 4.25 |
| SWA-only | 5.21% | 不适用 | 不适用 | 不适用 | 不适用 | 0.00 |

在 8 条临时事实时：

| 条件 | EM | 目标存活率 | Recall@1 | 活动槽位 | 临时事实接受率 |
|---|---:|---:|---:|---:|---:|
| Gated | 55.56% | **100%** | **100%** | **1** | 0% |
| Fixed-LRU | 61.11% | 68.06% | 61.11% | 8 | 100% |
| SWA-only | 6.94% | 不适用 | 不适用 | 0 | 不适用 |

### 3.4 解释

主比较已经验证了门控机制最核心的容量管理能力：Gated 拒绝所有临时事实，始终只
占用一个槽位；Fixed-LRU 接受所有临时事实，噪声超过槽位容量后开始遗忘目标。

但是 Fixed-LRU 的 EM 比 Gated 高 22.22 个百分点。这说明：

- 门控在“记住什么”方面更有选择性；
- Fixed-LRU 在当前训练条件下更容易把记忆内容转成最终答案；
- 不能把 EM 较低简单解释成 Gated 没有找到目标，因为 Gated 的检索指标是满分。

主实验提供了“记忆选择成功、答案生成未成功”的研究动机。

---

## 4. 实验 B：全验证集门控权重审计

### 4.1 目的

判断门控输出是否存在明显的过高、过低或饱和，并检查饱和是否与错误样本相关。

### 4.2 逻辑

不重新训练。加载正式 Gated final checkpoint，逐条重放全部 288 条验证样本，并分别
记录：

1. 耐久事实候选的 write probability；
2. 临时事实候选的 write probability；
3. 目标槽位的 retention probability；
4. 查询位置两个 memory-fusion layer 的 gate；
5. 答对和答错样本的融合门分布。

这样可以判断“门控权重很极端”与“门控导致答案错误”是否是同一件事。

### 4.3 结果

#### 写入门

| 候选类型 | 数量 | 最小值 | 平均值 | 最大值 | 硬决策 |
|---|---:|---:|---:|---:|---|
| 耐久事实 | 288 | 1.000000 | 1.000000 | 1.000000 | 全部写入 |
| 临时事实 | 1,008 | 3.23e-12 | 5.69e-7 | 2.62e-5 | 全部拒绝 |

写入门高度饱和，但没有错误写入：耐久事实都保留，临时事实都拒绝。

#### 保留门

目标事实 retention probability：

| 最小值 | 5%分位 | 平均值 | 95%分位 | 最大值 |
|---:|---:|---:|---:|---:|
| 0.999117 | 0.999189 | 0.999239 | 0.999281 | 0.999296 |

保留门几乎完全打开，目标事实存活率为 100%。本任务没有设置“曾经重要但后来
过期”的事实，所以不能判断它是否能够主动释放旧记忆。

#### 融合门

| 融合层 | 全部均值 | 最小值 | 最大值 | 答对均值 | 答错均值 |
|---|---:|---:|---:|---:|---:|
| 第一个融合层 | 0.999295 | 0.997367 | 0.999797 | 0.999259 | 0.999335 |
| 第二个融合层 | 0.984094 | 0.920799 | 0.999731 | 0.989297 | 0.978358 |

答对和答错样本的门值高度重叠。第一层中答错样本的平均 gate 甚至略高，而不是更
低。因此不能把当前错误归因于融合门普遍关闭。

### 4.4 结论

本实验支持以下表述：

> Gated 的 write/retention/fusion gate 存在接近 0 或 1 的饱和，但在当前明确提示
> 的压力任务上没有造成错误存储或检索。门控饱和是未来开放域泛化的风险，而不是
> 当前 EM 退化的已证实主因。

---

## 5. 实验 C：固定融合门扫描

### 5.1 目的

直接检验融合强度是否影响最终答案。这里不改变写入门、保留门、reader 或模型
参数，只在推理时把两个融合层的 gate 强行设为固定值。

### 5.2 设置

使用同一个 Gated final checkpoint、同一个 288 条验证集，扫描：

```text
gate = 0.00, 0.25, 0.50, 0.75, 1.00
```

这是 inference-only intervention，不是重新训练结果。因此它用于定位因果敏感性，
不能直接当作一个新模型的最终性能。

### 5.3 结果

| 固定融合门 | EM | 目标存活率 | Recall@1 | MRR |
|---:|---:|---:|---:|---:|
| 0.00 | 5.21% | 100% | 100% | 1.0000 |
| 0.25 | 14.58% | 100% | 100% | 1.0000 |
| 0.50 | **57.99%** | 100% | 100% | 1.0000 |
| 0.75 | 55.21% | 100% | 100% | 1.0000 |
| 1.00 | 52.43% | 100% | 100% | 1.0000 |

### 5.4 解释

融合门为 0 时，EM 退化到 5.21%，与没有事实记忆的 SWA-only 基线相近。这证明
记忆必须真正进入 decoder，不能只停留在 state 或 reader 中。

从 0.25 增至 0.5 时性能快速提高，说明融合过弱确实会损失信息。0.5 达到 57.99%，
略高于自然 gate 的 52.43%；继续增大到 0.75 和 1.0 后性能下降，但下降幅度有限。

所以当前结论是：

- gate 太低会明显退化；
- gate 太高可能存在轻微过融合或干扰；
- 自然 gate 不是完全失效，但可能没有落在最优强度；
- 仅凭这项扫描不能解释 22.22 个百分点的 Gated/Fixed EM 差距。

### 5.5 实现改动

增加了 `gate_override` 参数：

- `SharedMemoryFusion` 可接受固定 gate；
- `DecoderBlock` 和 `AdaptiveFactMemoryLM` 传递该参数；
- `train_memory_stress.py` 增加 `--fusion-gate-override`；
- 新增诊断脚本 `scripts/diagnose_memory_bottleneck.py`。

默认值仍为 `None`，因此不影响原有模型和旧 run。

---

## 6. 实验 D：训练 checkpoint 轨迹扫描

### 6.1 目的

判断 EM 低是否主要因为训练过早停止，或者模型在中途已经达到最佳点、之后发生过拟合。

### 6.2 逻辑

不重新训练。对同一个 Gated run 的四个 checkpoint 逐一评估：

```text
checkpoint.step_500.pt
checkpoint.step_1000.pt
checkpoint.step_1500.pt
checkpoint.step_2000.pt
```

验证集和其他评估设置保持不变。

### 6.3 结果

| checkpoint | EM | 目标存活率 | Recall@1 | MRR |
|---:|---:|---:|---:|---:|
| 500 | 11.46% | 100% | 100% | 1.0000 |
| 1000 | 19.79% | 100% | 100% | 1.0000 |
| 1500 | 42.36% | 100% | 100% | 1.0000 |
| 2000 | 52.43% | 100% | 100% | 1.0000 |

### 6.4 解释

记忆写入、保留和读取从早期 checkpoint 起就已经成功；随着训练继续，主要提升发生
在最终答案生成。因此 2,000 step 不是“门控刚学会就停止”的情况，但 EM 仍在上升，
说明继续训练或专门训练 answer/fusion path 可能有效。

这个结果也说明“Recall 已满分”并不代表整条生成链已经收敛。当前优化过程先学会了
记忆选择，再逐步学习如何把记忆转化为词表输出。

---

## 7. 实验 E：训练模板与验证模板对照

### 7.1 目的

测量验证模板改写造成了多少性能损失，避免把模板泛化问题误判为门控机制失败。

### 7.2 逻辑

加载同一个 Gated final checkpoint，分别用：

- `split=train`：耐久提示为 `Remember/Keep/Save`，噪声提示为
  `Temporary/Skip/Ignore`；
- `split=validation`：耐久提示为 `Store`，噪声提示为 `Discard`。

其他 delay/noise 网格、样本数量和生成流程保持一致。

### 7.3 结果

| 数据模板 | EM | 目标存活率 | Recall@1 | MRR |
|---|---:|---:|---:|---:|
| 训练模板 | **78.13%** | 100% | 100% | 1.0000 |
| 验证改写模板 | 52.43% | 100% | 100% | 1.0000 |

EM 差距为 25.69 个百分点，但所有记忆内部指标仍为满分。

### 7.4 解释

这说明模型对记忆机制本身的执行是稳定的，但对训练中没有见过的提示词组合，答案
生成能力明显下降。当前合成任务仍然含有相当强的 lexical cue：模型可能学会了
“哪些词通常对应保留/丢弃”，而不是完全抽象地理解重要性。

因此，后续论文主结果必须把模板内性能和模板外性能分开报告；不能只报告一个混合 EM。

---

## 8. 实验 F：统一答案前导空格的重训

### 8.1 动机

GPT-2 的 tokenization 对前导空格敏感。例如：

| 文本 | token |
|---|---|
| `blue` | 一个 token，ID 17585 |
| ` blue` | 另一个 token，ID 4171 |
| `navy` | 两个 token：`n`、`avy` |
| ` navy` | 一个 token |

事实句中的颜色通常出现在单词之间，接近 ` blue`；当前答案目标原本编码为 `blue`。
因此尝试把答案目标也改为带前导空格的形式，使事实 payload 和答案 token 更一致。

### 8.2 唯一修改

新增 `--answer-leading-space`，只改变答案 target 的编码：

```python
answer_text = " " + value
answer_ids = encode(answer_text) + (eos,)
```

写入规则、reader、fusion、训练步数和验证网格均不变。

### 8.3 设置与结果

| 项目 | 设置 |
|---|---|
| run ID | `selective_stress_v2_gated_s17_answer_space_20260909` |
| seed | 17 |
| steps | 2,000 |
| GPU | 物理 GPU 1 |
| 训练时间 | 1,364.63 s（22.74 min） |
| 吞吐 | 5.86 examples/s |
| EM | **20.14%** |
| 目标存活率 | 100% |
| Recall@1 | 100% |
| Recall@4 | 100% |
| MRR | 1.0000 |
| 平均活动槽位 | 1.00 |

错误输出主要变成带前导空格的 ` gold`、` navy`、` red` 等词。记忆内部指标没有
变化，但答案生成显著下降。

### 8.4 解释

这个结果排除了一个过于简单的修复假设：**只要把答案改成和事实一样的前导空格，
问题就会解决。** 实际上，这个改动同时改变了 decoder 在查询结束位置需要预测的
目标 token 分布；如果查询和答案之间没有明确的格式分隔，带空格目标可能与当前
生成上下文不匹配。

因此 tokenization 仍可能是影响因素，但需要更谨慎地设计：应统一“事实、问题、
答案”的完整文本格式，并配合 token-level 对齐分析，而不是只修改 target 字符串。

### 8.5 实现改动

- `synthetic_memory_stress.py` 新增 `answer_leading_space` 选项；
- `train_memory_stress.py` 新增 `--answer-leading-space`；
- 默认仍为关闭，不改变原有协议。

---

## 9. 实验 G：固定融合门=0.5 的重训

### 9.1 目的

固定门扫描显示 0.5 的推理 EM 最高。这里进一步检查：如果从训练开始就使用固定
融合门=0.5，能否稳定改善最终模型，而不是只在已有 checkpoint 上做推理干预。

### 9.2 唯一修改

训练和验证都通过：

```text
--fusion-gate-override 0.5
```

写入门、保留门、reader、payload 表示、训练数据和优化器不变。

### 9.3 设置与结果

| 项目 | 设置 |
|---|---|
| run ID | `selective_stress_v2_gated_s17_fusion05_20260909` |
| seed | 17 |
| steps | 2,000 |
| GPU | 物理 GPU 0 |
| 训练时间 | 1,338.73 s（22.31 min） |
| 吞吐 | 5.98 examples/s |
| EM | **55.90%** |
| 目标存活率 | 100% |
| Recall@1 | 100% |
| Recall@4 | 100% |
| MRR | 1.0000 |
| 平均活动槽位 | 1.00 |

### 9.4 解释

相对于原始 Gated 的 52.43%，固定门=0.5 提高 3.47 个百分点；检索指标和容量使用
完全不变。

这说明融合强度确实影响答案生成，但改善幅度有限。它支持“自然门值可能略偏高”
这一弱结论，不支持“融合门是根本瓶颈”这一强结论。即使把融合强度调整到当前
扫描得到的较优位置，EM 仍远低于 Fixed-LRU 的 74.65%。

---

## 10. 训练损失和资源补充

### 10.1 最后 100 step 的 LM loss

| 条件 | 最后 100 step LM loss | 训练时间 |
|---|---:|---:|
| 原始 Gated | 0.764 | 1,323.02 s |
| Fixed-LRU | 0.416 | 1,200.59 s |
| SWA-only | 1.356 | 880.05 s |
| Gated + fusion=0.5 | 需以该 run 日志为准，EM 已报告 |
| Gated + answer leading space | 需以该 run 日志为准，EM 已报告 |

原始 Gated 的 LM loss 明显高于 Fixed-LRU，说明两者不只是“保存内容不同”。Fixed-LRU
始终打开融合门，且不承担 write/retention/budget 门控损失；Gated 同时训练多个辅助
目标，存在多任务梯度竞争的可能。

### 10.2 状态和逻辑槽位

原始 Gated 和 Fixed-LRU 的完整预分配 `state_bytes` 都为 2,408,906 bytes；Gated
平均逻辑活动槽位为 1，Fixed-LRU 为 4.25。Gated 的逻辑活动槽位数据量平均少
76.47%，但当前实现仍预分配全部槽位，因此不能把它直接称为物理显存压缩。

### 10.3 训练资源

| 条件 | 训练时间 | 吞吐 | peak allocated | peak reserved |
|---|---:|---:|---:|---:|
| 原始 Gated | 1,323.02 s | 6.05 ex/s | 2.673 GB | 6.648 GB |
| Fixed-LRU | 1,200.59 s | 6.66 ex/s | 2.666 GB | 6.568 GB |
| SWA-only | 880.05 s | 9.09 ex/s | 2.871 GB | 6.881 GB |

这些数字来自当前未融合的 PyTorch 原型，主要用于记录实验成本，不代表优化 kernel
后的理论极限。

---

## 11. 综合根因分析

### 11.1 已基本排除的原因

#### 不是目标事实完全没有写入

Gated 目标存活率为 100%，耐久事实 write probability 为 1.0。

#### 不是语义 reader 找错了槽位

Gated 在所有 288 条验证样本上 Recall@1、Recall@4 和 MRR 都是满分。

#### 不是融合门普遍关闭

两个融合层 gate 均较高，答错样本的第一层 gate 甚至略高于答对样本。

#### 不是明显的训练过拟合

500 到 2,000 step 的 EM 持续上升，没有出现“中间最好、后面崩溃”的轨迹。

### 11.2 仍然最可疑的原因

#### 原因一：记忆表示压缩后不够适合精确词汇解码

当前 fusion 使用：

```text
memory value + payload token embedding 的平均向量
```

这有利于语义检索，却可能把 `blue`、`green`、`white` 等具体值压成过于相似的
“颜色语义”表示。输出层需要从这个模糊表示恢复具体词，容易被更强的语言先验打败。

#### 原因二：输出头是 tied embedding，要求隐藏状态精确对准 token embedding

当前 LM head 与输入 embedding 共享权重。若 memory fusion 只提供“这是某种颜色”
的方向，而没有提供足够精确的 token 方向，最终 logit 可能稳定偏向某个高先验词，
例如 `yellow`。

#### 原因三：训练和验证提示迁移造成答案生成下降

训练模板 EM 78.13%，验证模板 EM 52.43%，差距很大；但两者的记忆内部指标都满分。
这说明模型可能已学会机制，却没有学会对未见的语言提示进行稳健抽象。

#### 原因四：tokenization 增加了不必要的映射难度

事实内的带空格颜色 token、答案起始位置的无空格颜色 token，以及 `navy` 的单词
拆分差异，都可能让精确复制变成重新分类问题。简单加空格的重训没有成功，说明需要
统一完整序列格式，而不是局部修改 target。

#### 原因五：Gated 的辅助损失与答案 LM 目标存在竞争

Gated 同时优化写入、读取、key alignment、retention、budget 和答案 LM；Fixed-LRU
不训练同样的门控目标，且固定融合门全开。原始 Gated 的最终 LM loss 高于 Fixed-LRU，
说明答案路径可能没有获得同样强的优化资源。

### 11.3 当前最重要的证据链

```text
目标槽位存在：100%
目标槽位排第一：100%
正确 token 排词表第一：52.43%
正确 token 进入词表前五：97.57%
```

因此问题发生在：

```text
正确记忆
  → decoder 隐藏状态
  → 词表 logit 排序
```

而不是：

```text
事实是否存活
或
reader 是否找对
```

---

## 12. 本轮代码改动清单

### 已用于正式结果的改动

- 新增 `scripts/diagnose_memory_bottleneck.py`：门值扫描、checkpoint 扫描、模板
  对照；
- `SharedMemoryFusion` 新增 `gate_override`；
- `DecoderBlock`、`AdaptiveFactMemoryLM` 传递固定融合门；
- `train_memory_stress.py` 新增：
  - `--fusion-gate-override`；
  - `--answer-leading-space`；
  - `--memory-value-mode` 参数接口；
- `synthetic_memory_stress.py` 新增 `answer_leading_space`。

### 已加入但尚未纳入正式结果的接口

`memory_value_mode` 支持三种 fusion 表示：

- `combined`：当前默认的 `selection.values + payload_summary`；
- `payload_only`：只使用 payload token 平均表示；
- `value_only`：只使用 memory value。

这一接口已经通过 15 项单元测试，但本报告没有把它们当成已完成的正式对比，因为
尚未运行对应的完整评估/重训。这样可以避免把“代码已支持”误写成“实验已证明”。

---

## 13. 可复核产物

### 主实验目录

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_stress/
selective_stress_v2_gated_s17_20260909/

/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_stress/
selective_stress_v2_fixed_lru_s17_20260909/

/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_stress/
selective_stress_v2_swa_only_s17_20260909/
```

### 诊断输出

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_stress/
diagnostics_gate_scan_s17_20260909.json

/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_stress/
diagnostics_checkpoint_sweep_s17_20260909.json

/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_stress/
diagnostics_template_comparison_s17_20260909.json
```

### 两个重训变体

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_stress/
selective_stress_v2_gated_s17_answer_space_20260909/

/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_stress/
selective_stress_v2_gated_s17_fusion05_20260909/
```

每个重训目录包含完整 `config.resolved.json`、`metrics.jsonl`、每 500 step 的
checkpoint、`checkpoint.final.pt` 和 `summary.json`。

### 软件测试

```bash
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 -m unittest discover \
  -s adaptive_gated_fact_memory/tests -p 'test_*.py' -v
```

结果：15 项测试全部通过。

---

## 14. 论文中应如何表述

推荐的严谨表述是：

> 在 seed-17 的 TinyLM 结构化压力 pilot 中，门控记忆始终保留并检索目标事实，
> 即 target survival、Recall@1 和 Recall@4 均达到 100%，同时拒绝全部临时事实并
> 将平均活动槽位限制为 1。相较之下，固定写入的 LRU 记忆在超容量噪声下目标存活率
> 降至 68.06%。然而，Gated 的端到端 EM 为 52.43%，低于 Fixed-LRU 的 74.65%，
> 表明当前主要挑战是将已检索的事实稳定映射到具体答案 token，而不是事实存储或
> 语义检索本身。固定融合门扫描和重训显示门值校准只能带来有限改善；模板迁移和
> 记忆表示到输出层的对齐仍是后续工作的重点。

不推荐的表述是：

- “门控导致了 EM 退化”：目前没有因果证据；
- “门控已经实现物理存储压缩”：当前只是逻辑活动槽位稀疏；
- “模型已经自主理解信息重要性”：当前数据有明确耐久/临时提示；
- “这是论文官方 SWA 代码结果”：当前是 source-policy adapter；
- “2,000 step 已达到最佳状态”：尚未进行独立验证集 early stopping。

---

## 15. 后续最有价值的实验

按优先级建议：

1. 对 `combined/payload_only/value_only` 做同一 checkpoint 的推理干预和完整重训；
2. 加入正确槽位 oracle，比较正常 fusion 与直接 value 注入的 EM 上限；
3. 把 teacher-forced answer NLL、首 token rank、logit margin 与自由生成 EM 同时
   报告；
4. 使用统一的文本格式设计 tokenization 消融，而不是只加前导空格；
5. 分离 write gate、retention gate、fusion gate 的训练消融；
6. 训练更多 step 并按独立验证 EM 选择 best checkpoint；
7. 补 seed=29、43，报告均值±标准差；
8. 加入多条真正重要事实、过期事实和冲突更新，测试门控是否会合理遗忘。

## 16. 最终结论

本轮实验没有找到“门控权重过高或过低导致当前退化”的直接证据。门控输出确实很
饱和，但它正确地写入了耐久事实、拒绝了临时事实并保护了目标槽位。融合门强度对
EM 有影响，最佳固定值约为 0.5，但重新训练后的收益只有 3.47 个百分点。

当前最根本的问题更可能是：**模型已经检索到正确记忆，却没有把压缩后的记忆表示
稳定转换成正确的词表输出**。验证模板迁移、前导空格 tokenization、tied LM head、
payload 平均池化和 Gated 的多任务优化共同增加了这一步的难度。下一阶段应优先
研究 memory-to-logit 对齐，而不是继续扩大门控复杂度。
