# 从压缩记忆到答案 token：根因诊断与解决方案

更新时间：2026-09-09  
实验对象：seed-17 Gated memory TinyLM  
目的：解释“目标事实已被正确保存和检索，但最终答案仍经常错误”的原因，并确定可行的修复方向。

## 1. 结论先行

当前模型并不是没有记住事实，而是没有把事实中“真正要回答的值”稳定转换为词表中的答案 token。

现有证据把问题定位在下面这条链路：

```text
完整事实句
  → 平均池化成一个 span 表示
  → value projection 压缩
  → memory fusion 变成 decoder residual
  → tied LM head 排序整个词表
  → 具体答案词没有稳定排第一
```

最重要的事实是：

- Gated 的目标事实存活率：100%；
- Recall@1、Recall@4、MRR：全部 100%；
- 但最终 EM：52.43%；
- 正确答案 token 的平均词表排名：2.04；
- 正确答案 token 进入词表前五的比例：97.57%；
- 错误通常是另一个颜色词，而不是完全无意义的输出。

这表示模型大致知道“答案属于某个颜色集合”，但还不能可靠区分应当输出哪个具体颜色。

## 2. 当前机制到底保存了什么

当前 `AtomicFactExtractor` 从整句事实中选择一个 span。它做的核心操作是：

```text
span_hidden = 事实 span 中所有 token 的 hidden state
span_representation = 所有有效 token 的平均值
key   = key_projection(span_representation)
value = value_projection(span_representation)
```

同时，系统还保存了 `payload_ids`，但在 `SharedMemoryFusion` 中并没有逐 token 使用它们，而是：

```text
payload_summary = 平均(payload token embeddings)
memory_values   = value + payload_summary
```

因此当前保存的是两种压缩后的信息：

1. `value`：由整句 span 的平均 hidden 经过线性投影得到的连续向量；
2. `payload_summary`：整句 payload token embedding 的平均向量。

它们非常适合表达“这条记忆大概讲了什么”，却不天然等价于“请输出其中第几个 token”。

例如事实：

```text
Store: Bob likes blue.
```

当前压缩表示没有显式标记：

```text
blue 是需要在答案中复制的 object/value span
```

它只知道整句话和 Bob 的查询相关。

## 3. 诊断一：记忆检索是否真的成功

### 3.1 目的

先排除“目标事实没写进去”与“reader 找错槽位”这两个更早阶段的问题。

### 3.2 设置

- 使用正式 Gated final checkpoint；
- 验证集 4 个噪声强度 × 3 个延迟长度；
- 每个条件 24 条，共 288 条；
- 不改变模型参数。

### 3.3 结果

| 指标 | 结果 |
|---|---:|
| 目标事实存活率 | 100% |
| Recall@1 | 100% |
| Recall@4 | 100% |
| MRR | 1.0000 |
| 最终 EM | 52.43% |

### 3.4 结论

错误不是由“记忆消失”或“reader 找错”造成的。目标槽位已经被正确找到，问题发生在检索之后。

## 4. 诊断二：逐层检查记忆表示到输出表示的变化

### 4.1 目的

比较以下表示与正确答案 token embedding 的关系：

1. payload 平均向量；
2. 压缩后的 memory value；
3. 二者相加后的 combined memory；
4. 第一、第二个融合层产生的 residual；
5. 最终 decoder hidden。

这里使用 cosine similarity 作为方向性诊断。它不是严格的概率，但可以判断表示是否仍然携带目标词方向。

### 4.2 设置

- 正式 Gated final checkpoint；
- 全部 288 条验证样本；
- 分别统计答对样本与答错样本；
- 不训练、不修改权重。

### 4.3 结果

| 表示 | 全部样本 | 答对样本 | 答错样本 |
|---|---:|---:|---:|
| payload 平均向量与目标词 cosine | 0.520 | 0.547 | 0.490 |
| memory value 与目标词 cosine | -0.036 | -0.031 | -0.041 |
| combined memory 与目标词 cosine | -0.008 | 0.000 | -0.017 |
| 第一个融合层 residual cosine | 0.108 | 0.141 | 0.072 |
| 第二个融合层 residual cosine | 0.287 | 0.338 | 0.230 |
| 最终 hidden 与目标词 cosine | 0.255 | 0.274 | 0.233 |

最终答案 token 的词表排名为：

| 指标 | 结果 |
|---|---:|
| 答案 token 排名第一 | 52.43% |
| 答案 token 排名前五 | 97.57% |
| 平均排名 | 2.04 |
| 中位排名 | 1 |
| 错误样本中目标词与最高 logit 的平均差 | -0.5853 |

### 4.4 结论

表示在融合后并没有完全丢失答案信息：答错样本的最终 hidden 仍比随机方向更接近目标词，而且正确词通常在前五名。

但是它没有形成足够强的、稳定的词级方向，来压过其他颜色词的语言模型先验。这是“语义可用、词汇不精确”的典型表现。

需要注意：`memory value` 与答案 embedding 的 cosine 为负，并不意味着 value 必须直接等于答案 embedding。value 经过 `to_v`、attention 和 `to_out` 后才进入 decoder；这个数值主要说明当前 value 本身没有被直接训练成“答案 token 表示”。

## 5. 诊断三：改变 fusion 输入，判断哪部分信息有用

### 5.1 目的

检查 `selection.values` 和 `payload_summary` 各自对答案的贡献。

### 5.2 设置

使用同一个正式 Gated checkpoint，只在推理时替换 fusion 的输入：

- `combined`：当前实现，`value + payload_summary`；
- `payload_only`：只使用 payload 平均 embedding；
- `value_only`：只使用压缩后的 value。

这是一项 inference-only 干预，不是重新训练后的独立模型。

### 5.3 结果

| Fusion 输入 | EM | 目标存活率 | Recall@1 | MRR |
|---|---:|---:|---:|---:|
| combined | 52.43% | 100% | 100% | 1.0000 |
| payload_only | 21.53% | 100% | 100% | 1.0000 |
| value_only | 27.78% | 100% | 100% | 1.0000 |

### 5.4 结论

两种表示都携带有用信息，简单删除任意一部分都会降低 EM。当前问题不是“value 完全无用”，而是：

```text
连续 value 和平均 payload 都有部分信息
但没有任何一项被直接训练成可复制的答案 token 表示
```

这也说明不能把修复简单地写成“只使用 payload”或“只使用 value”。需要增加词级表示或显式复制路径。

## 6. 诊断四：fusion gate 是否是主要原因

### 6.1 设置

在同一个 Gated final checkpoint 上，把两个融合层的 gate 固定为：

```text
0.00、0.25、0.50、0.75、1.00
```

写入、保留、reader 和 decoder 参数均不变。

### 6.2 结果

| 固定 fusion gate | EM | Recall@1 |
|---:|---:|---:|
| 0.00 | 5.21% | 100% |
| 0.25 | 14.58% | 100% |
| 0.50 | 57.99% | 100% |
| 0.75 | 55.21% | 100% |
| 1.00 | 52.43% | 100% |

另外，固定 gate=0.5 从头训练的 Gated 变体结果为：

```text
EM = 55.90%
目标存活率 = 100%
Recall@1 = 100%
```

### 6.3 结论

- gate=0 会使 EM 退化到 SWA-only 水平，说明记忆必须真正进入 decoder；
- gate 太低确实会损失记忆信号；
- gate=0.5 略优于当前自然 gate；
- 但重新训练只提高 3.47 个百分点，不能解释全部退化。

因此 fusion gate 是次要校准因素，不是根本原因。

## 7. 诊断五：训练是否太早结束

### 7.1 设置

评估同一 Gated run 的：

```text
step 500、1000、1500、2000
```

### 7.2 结果

| Checkpoint | EM | Recall@1 |
|---:|---:|---:|
| 500 | 11.46% | 100% |
| 1000 | 19.79% | 100% |
| 1500 | 42.36% | 100% |
| 2000 | 52.43% | 100% |

### 7.3 结论

EM 仍在持续提高，没有出现明显的中途最佳、之后崩溃。因此不能说当前模型已经严重过拟合；更准确地说，2,000 step 时记忆机制已经学会，但答案解码还没有完全收敛。

## 8. 诊断六：提示模板迁移和 tokenization

### 8.1 模板迁移

同一个 final checkpoint：

| 模板 | EM | Recall@1 |
|---|---:|---:|
| 训练模板 | 78.13% | 100% |
| 验证改写模板 | 52.43% | 100% |

这说明模型对 `Remember/Keep/Save` 等训练词的适应好于未见过的 `Store`/`Discard` 组合。它进一步增加了答案生成难度，但不会解释记忆 reader 的失败，因为 Recall 仍是满分。

### 8.2 Tokenization

GPT-2 对前导空格敏感：

| 文本 | token 示例 |
|---|---|
| `blue` | 一个 token |
| ` blue` | 另一个 token |
| `navy` | 可能拆成两个 token |
| ` navy` | 可能是一个 token |

尝试把答案全部改成带前导空格并重新训练 2,000 step 后：

```text
EM = 20.14%
目标存活率 = 100%
Recall@1 = 100%
```

因此 tokenization 是真实影响因素，但简单加前导空格不是解决方案。需要同时统一问题末尾格式、答案起始位置和训练目标，而不是只修改 `answer_ids`。

## 9. 根本原因的综合判断

### 原因一：完整事实 span 被平均池化，value 部分没有单独标记

事实句中包含命令词、人物、关系词、答案值和标点。平均池化会把这些信息混在一起。

当前模型被要求从：

```text
“Bob likes blue” 的整体语义向量
```

恢复：

```text
“blue” 这个具体词的词表 logit
```

这本身就是一个比检索更难的任务。

### 原因二：没有 memory-to-vocabulary 的直接监督

当前答案 LM loss 只监督最终 query answer token：

```text
query hidden → decoder → tied LM head → answer token
```

没有单独的损失要求：

```text
retrieved memory → 正确 value token
```

于是 value projection 只需要成为“可用于语义回答的连续特征”，不需要成为可复制的词级表示。

### 原因三：`payload_ids` 被平均为一个向量，顺序和 token 身份被弱化

虽然状态中已经保存了 payload token IDs，但 fusion 只使用它们的平均 embedding。平均操作丢失：

- token 顺序；
- 哪个 token 是 value；
- value 的精确离散 ID；
- 多 token value 的边界。

### 原因四：tied LM head 放大了表示误差

当前输出层与输入 embedding 共享权重。最终 hidden 必须落在正确答案 token embedding 的方向上，才能让目标词 logit 超过竞争词。若融合只提供模糊的颜色语义，输出层会忠实地选择语言先验更强的颜色词。

### 原因五：跨轮 detach 和多任务损失削弱了 value 学习

事实写入轮与查询答案轮之间会 detach state，因此答案 LM loss 不能完整地反向传播到早期 value encoder。与此同时 Gated 同时优化写入、读取、key alignment、retention、budget 和答案 LM，可能出现答案路径与记忆路径的梯度竞争。

## 10. 推荐的解决方案

### 方案 A：增加显式 value-copy 路径（最推荐）

在结构化任务中，事实通常具有：

```text
subject / relation / object(value)
```

应当在槽位中单独保存：

```text
value_token_ids
value_token_mask
```

查询后由 reader 选择槽位，再通过 pointer/copy head 把 value token 直接复制到答案分布中：

```text
P(answer token)
  = (1 - α) P(LM token)
  + α P(copied value token)
```

优点：

- 不要求连续向量重新发明原始 token；
- 多 token value 的顺序可以保留；
- 能直接利用当前 state 已有的 payload ID 存储机制；
- 最符合“记忆系统保存事实、回答时复制 value”的设计目标。

注意：不能直接复制整个事实 span，必须先保存 object/value 子span，或增加 value-span head。

### 方案 B：对 payload 做 token-level cross-attention

把当前：

```text
平均(payload token embeddings)
```

改为：

```text
query hidden 对 payload token 序列做 cross-attention
```

这样 decoder 可以分别看到 `Store`、`Bob`、`likes`、`blue`，并学习关注 value token，而不是接收一个平均向量。

建议保留一个很小的 payload 序列，例如 12 token，以免状态开销失控。

### 方案 C：加入直接的 memory-to-token 辅助损失

新增一个 memory answer head：

```text
memory_answer_logits = answer_projection(retrieved_memory)
```

训练时直接对正确 value token 做 cross-entropy：

```text
L = L_LM + λ_value * CE(memory_answer_logits, value_token_ids)
```

它的作用是把 value projection 从“语义特征”拉向“可解码 value 表示”。

如果不想增加完整词表 head，可以使用 tied embedding 的 token contrastive loss，只在候选 value 词和若干 hard negative 颜色词之间计算。

### 方案 D：给事实抽取器增加 value-span 监督

当前只有整句 fact span 的 start/length。应增加：

```text
value_start
value_length
```

在本合成任务中这些标签可以由生成器直接提供。开放域场景则需要关系抽取器或 object-span head。

### 方案 E：分阶段训练 value 路径

建议不要一开始同时强烈优化全部门控目标：

1. 冻结 TinyLM 主干，只训练 value encoder、fusion 和 answer head；
2. 让 memory-to-token CE 先收敛；
3. 再打开 write/retention gate，并降低辅助损失权重；
4. 最后联合微调。

这可以减少 Gated 的多任务梯度竞争。

### 方案 F：减少或局部解除跨轮 detach

不建议直接对整段长 filler 做完整 BPTT，因为显存和稳定性成本很高。更安全的做法是：

- 只对“事实写入 → 查询答案”的短 value 路径保留梯度；
- filler 仍然 detach；
- 或增加局部 value reconstruction loss，让 value encoder 在写入轮就获得监督。

### 方案 G：统一答案格式和 tokenization

应统一：

- query 末尾是否有空格或换行；
- answer 是否带前导空格；
- 训练和验证的答案 token 边界；
- 多 token value 的结束标记。

单独加前导空格已经失败，因此它只能作为完整格式协议的一部分，不能单独作为修复。

### 方案 H：输出层消融

可以增加一个 untied LM head 作为对照：

```text
输入 embedding 与输出分类矩阵不共享
```

如果 untied head 明显提升，说明 tied embedding 的几何约束确实限制了 memory-to-token 映射；如果没有提升，则主要问题在 fusion/value 表示，而不是输出矩阵本身。

## 11. 推荐的最小验证实验

为了用最少实验确定解决方向，建议按顺序做以下四项，每项都使用 seed=17、相同 2,000 step 和相同 288 条验证集：

### 实验 1：value-span + token-level cross-attention

- 不改变 write/retention gate；
- 从事实中保存 value 子span；
- fusion 对 value token 序列做 cross-attention；
- 记录 EM、首 token rank、teacher-forced NLL。

如果明显提升，说明平均池化是主要原因。

### 实验 2：memory-to-token auxiliary CE

- 保持现有 combined fusion；
- 增加 retrieved value token 的直接 CE；
- 比较 `λ_value=0.1/0.5/1.0`。

如果提升，说明缺少词级监督是主要原因。

### 实验 3：copy head

- 正确 reader 槽位后，直接对 value token 做 pointer/copy；
- 与普通 LM logits 混合；
- 测试 oracle reader 和正常 reader 两种情况。

如果 copy head 大幅提升，说明连续 value 解码不是合适的路径，应保留离散 token memory。

### 实验 4：untied output head

- 只将 tied LM head 改为独立输出矩阵；
- 其他设置完全不变。

如果只有这个实验提升，才有理由把主要问题归因于输出层参数化。

## 12. 当前应如何表述

推荐的论文表述：

> The proposed memory module achieved perfect target-slot survival and retrieval recall, but its end-to-end answer accuracy remained limited. Layer-wise analysis showed that the compressed memory representation preserved semantic information while providing insufficient token-level alignment for exact answer generation. This gap is attributable to span-level average pooling, the absence of an explicit memory-to-vocabulary objective, and the use of a tied output head. A token-level value pathway or an explicit copy head is therefore a principled next step.

中文可表述为：

> 实验表明，模型的长期事实存储和语义检索已经成功，但压缩记忆主要保留了事实的语义，而没有被直接训练成可复制的答案 token 表示。整句 span 平均池化、缺少 memory-to-vocabulary 监督以及 tied LM head 共同造成了从“找到事实”到“输出具体值”的性能间隙。下一步应增加 value 子span 的 token-level pathway 或显式 copy head，而不是继续复杂化门控本身。

## 13. 可复核文件

诊断结果位于数据盘：

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/diagnostics_gate_scan_s17_20260909.json
/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/diagnostics_checkpoint_sweep_s17_20260909.json
/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/diagnostics_template_comparison_s17_20260909.json
/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/diagnostics_representation_mapping_s17_20260909.json
```

重训变体：

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/selective_stress_v2_gated_s17_answer_space_20260909/
/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/selective_stress_v2_gated_s17_fusion05_20260909/
```

本报告没有覆盖任何旧实验结果；当前 15 项单元测试全部通过。
