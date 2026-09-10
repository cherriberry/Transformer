# 自适应事实记忆架构演进与实验设置说明

更新时间：2026-09-11（UTC）  
项目：Adaptive Gated Fact Memory + TinyLM + MoE  
实验主协议：seed 17 selective-memory stress  

## 1. 先给出最重要的结论

这个项目实际上有两个相互独立的改动方向：

```text
方向 A：事实记忆如何保存和恢复
        旧 Gated  →  Value-token Gated

方向 B：Transformer 的 FFN 如何计算
        Dense FFN  →  Top-2 MoE FFN
```

因此，实验不是把“旧 Gated 改成 MoE Gated”，而是形成一个 2×2 对照：

| 事实记忆路径 | Dense FFN | Top-2 MoE FFN |
|---|---|---|
| 旧 Gated | 旧 Gated + Dense | 旧 Gated + MoE |
| Value-token Gated | Value-token Gated + Dense | Value-token Gated + MoE |

最重要的解释是：

- **Value-token 修改**解决的是“已经检索到事实，但不能稳定生成正确 value token”的问题。
- **MoE 修改**解决的是“Transformer FFN 的条件容量和 expert 路由”的问题。
- MoE router 不是事实记忆 gate；它不决定事实是否写入、保留或读取。
- 在当前受控任务上，最明显的质量提升来自 Value-token alignment，而不是 MoE。

## 2. 整体架构：模型同时维护两个时间尺度

模型是一个 decoder-only causal Transformer。它同时使用两种记忆：

```text
短时间尺度：最近 token 的精确表示
    4 个 attention sink + 最近 124 个 token 的 per-layer KV

长时间尺度：经过选择的事实记忆
    semantic key/value + 可选 lexical value + 原始 payload token
```

完整数据流可以理解为：

```text
输入 token
   │
   ▼
Token embedding
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ Decoder block × 6                                       │
│                                                         │
│  Local causal attention                                 │
│    └─ 4 sink KV + 124 recent KV                         │
│                                                         │
│  Memory fusion（在第 3、6 层，代码下标 2、5）            │
│    └─ 当前 prompt 读取的 top-k 事实                      │
│                                                         │
│  FFN：Dense 或 Top-2 MoE                                │
└─────────────────────────────────────────────────────────┘
   │
   ▼
Final LayerNorm
   │
   ▼
Tied token embedding LM head
   │
   ▼
next-token logits
```

这里的“事实记忆”不是把所有历史 token 永久保存下来，而是从用户消息中选择少量
短事实，将它们放进固定数量的 memory slots。旧 token 如果没有被写成事实，最终会
随着 local KV 窗口滑动而消失；这正是实验要测量的压力点。

## 3. 修改前：旧 Gated + Dense 架构

### 3.1 局部注意力路径

每一层使用有界的 causal sink/sliding attention：

```text
attention budget = 128 个唯一 key
                 = 4 个 sink token
                 + 124 个最近 token
```

sink token 是对话开头的少数关键位置，始终保留；recent window 保存最新的非 sink
token。每层都有自己的 KV cache，因此缓存会随层数增加，但每一层的缓存长度都有
同样的上限。

这样设置有两个目的：

1. 模拟真实的有界上下文系统，不让模型依靠无限增长的 KV cache；
2. 让较早的事实确实需要进入长期 memory 才能在长 filler 之后被恢复。

### 3.2 事实提取

旧版本的 `AtomicFactExtractor` 从 user token 中提出一个短 span：

```text
user span
   ├─ start head：选择事实起点
   ├─ length head：预测事实长度
   ├─ key projection：生成 semantic key
   ├─ value projection：生成 semantic value
   └─ payload：保存原始 token IDs，便于审计
```

例如：

```text
Bob likes blue
```

旧模型大致会保存：

```text
key   = “这个事实是什么”的向量
value = “这个事实内容”的语义向量
payload = 原始 span 的 token IDs
```

旧路径的主要问题是：完整事实 span 被压成一个语义向量后，模型知道“Bob 喜欢某个
颜色”，但不一定能稳定地从这个向量恢复出精确答案 token `blue`。

### 3.3 旧版 memory gates

旧 Gated 的核心 gate 包括：

```text
write gate       是否把候选事实写入 memory
retention gate   旧 slot 在下一次 commit 后是否继续保留
update gate      新事实与已有相似事实如何插值更新
merge/eviction   是否合并到相似 slot，或使用空 slot / 淘汰 slot
reader top-k     当前 prompt 读取哪些事实
fusion gate      读取的事实注入 decoder 的强度
```

memory controller 的工作顺序是：

```text
提出候选 → retention → 选择目标 slot → write/update → 更新 metadata
```

槽位中除了 key/value，还记录 active、age、last access、confidence、conflict、
source role 和 payload 等信息。

### 3.4 轮次级别的因果语义

模型把一次 user message + assistant response 看成一个 round：

1. user prompt 开始时，reader 选择旧事实；
2. 选中的事实只在 prompt 末端及后续 assistant continuation 可见；
3. 当前 round 的 user span 先进入 pending buffer；
4. 当前 round 的 logits 全部计算完后才 commit 新事实；
5. 新写入的事实只能影响下一轮，不能泄漏到创建它的当前回答。

这保证了训练和推理的语义一致，也避免“答案 token 先被写入、再帮助自己预测”的
信息泄漏。

## 4. 第一次主要修改：Value-token Gated

### 4.1 为什么需要修改

旧 Gated 的典型结果是：

```text
target survival = 100%
Recall@1        = 100%
MRR             = 1.0
但 EM           = 52.43%
```

这说明事实已经被写入，也被正确读取；主要失败发生在最后的“向 token 生成”阶段，
而不是“写入/检索”阶段。

### 4.2 新增的 value-span 路径

Value-token 版本保留旧的 semantic path，同时额外标注和提取 value：

```text
完整事实 span
   ├─ semantic path
   │    ├─ semantic key/value
   │    └─ 用于检索和关系理解
   │
   └─ lexical value path
        ├─ value_start_head
        ├─ value_length_head
        ├─ lexical_projection
        ├─ value_payload_ids
        └─ value_payload_mask
```

例如：

```text
Store: Bob likes [blue]
                 ↑
             明确的 value span
```

新状态同时保存：

- `semantic value`：用于表达事实语义；
- `lexical_values`：只聚合 value span 的 hidden states；
- `value_payload_ids`：value 的原始 token IDs；
- `value_payload_mask`：哪些 value token 有效。

### 4.3 memory-to-token 辅助头

新模型增加一个位置条件的辅助 token head：

```text
lexical value
   + answer position embedding
   → linear projection
   → LayerNorm
   → tied token embedding projection
   → value token logits
```

它提供 value token 的辅助监督，但**不替换正式生成路径**。正式自回归生成仍然使用
普通 LM head，因此这不是 pointer/copy head。

训练时增加了：

- value start loss；
- value length loss；
- candidate memory-token loss；
- retrieved memory-token loss。

Value-token Gated 的核心变化可以概括为：

```text
旧 Gated：保存“事实的语义”
新 Gated：保存“事实的语义” + “答案 value 的词级路径”
```

write、retention、merge、reader 和 memory fusion gate 的基本设计没有因为这个修改
而被替换。

## 5. 第二次主要修改：Dense FFN → Top-2 MoE FFN

### 5.1 Dense 版本

每个 decoder block 的 FFN 是：

```text
LayerNorm
  → Linear(384 → 1536)
  → GELU
  → Linear(1536 → 384)
  → residual add
```

所有 token 都经过同一个 FFN。

### 5.2 MoE 版本

MoE 只替换 FFN，attention 和事实 memory 路径保持不变：

```text
LayerNorm
   │
   ▼
Router: Linear(384 → 4)
   │
   ├─ Expert 0: Linear(384→1536) → GELU → Linear(1536→384)
   ├─ Expert 1: Linear(384→1536) → GELU → Linear(1536→384)
   ├─ Expert 2: Linear(384→1536) → GELU → Linear(1536→384)
   └─ Expert 3: Linear(384→1536) → GELU → Linear(1536→384)
   │
   ▼
Top-2 expert 输出按 router 权重加权合并
   │
   ▼
residual add
```

正式配置为：

```text
experts = 4
top-k   = 2
dropless dispatch
```

每一层都有独立的 4 个 experts 和 router。每个 token 在每一层选择两个 expert，
两个输出按归一化后的路由权重相加。

### 5.3 MoE router 与 memory gate 的区别

两者都可能被口语称为“门控”，但作用完全不同：

| 机制 | 输入 | 决定什么 | 所在位置 |
|---|---|---|---|
| memory write gate | user fact representation | 事实是否写入 | fact extractor/controller |
| retention gate | slot representation + metadata | 事实是否继续保留 | memory controller |
| memory fusion gate | decoder hidden + memory context | memory 注入强度 | decoder memory fusion |
| MoE router | 每个 token 的 hidden state | 使用哪个 FFN expert | decoder FFN |

MoE router 不知道某个事实是否重要，也不决定 memory slot 是否被淘汰。

### 5.4 为什么保留旧 Dense FFN 参数

代码中保留 dense `ffn_in/ffn_out` 有两个工程目的：

1. 让旧 Dense checkpoint 的 state-dict 仍然可以加载；
2. 启动 MoE 时把 Dense FFN 权重复制到所有 expert，作为稳定初始化。

初始化后，Dense FFN 权重被冻结，不参与 MoE active computation。它们的保留是兼容
和初始化策略，不代表 MoE 前向同时执行了 Dense FFN。

## 6. 四种架构版本的关系

### 6.1 修改关系图

```text
旧 Gated + Dense
       │
       ├── 只替换 FFN → 旧 Gated + MoE
       │
       └── 增加 value-token alignment → Value-token Gated + Dense
                                      │
                                      └── 再替换 FFN → Value-token Gated + MoE
```

更严格地说，实验矩阵是：

```text
                    FFN 路径
                 Dense        Top-2 MoE
记忆路径
旧 Gated       52.43% EM     32.99% EM
Value-token   100.00% EM    100.00% EM
```

因此，Value-token Gated + MoE 并不是为了适配 MoE 才重新设计 memory gate；它是
已经完成的 Value-token Gated 路径，再叠加一个独立的 FFN 替换。

## 7. 实验参数与设置理由

### 7.1 Backbone 参数

| 参数 | 设置 | 设置理由 |
|---|---:|---|
| layers | 6 | 固定 TinyLM 主干，控制实验成本，并让每层都能测试 memory/MoE 路径 |
| hidden size | 384 | 与既有 seed-17 TinyStories backbone 对齐，避免换主干带来混杂变量 |
| attention heads | 8 | hidden/head=48，维度整除且适合小型 decoder |
| dense FFN size | 1536 | 4× hidden size，是原 Dense backbone 的固定 FFN 宽度 |
| vocabulary | 50,257 | GPT-2 tokenizer 词表，保证与父 checkpoint 兼容 |
| embedding/output | tied | 减少参数，并让 memory-to-token 辅助头直接使用同一词表空间 |

这些参数不是本轮优化出来的超参数，而是为了严格继承已训练的 TinyStories 父模型。

### 7.2 局部 KV 参数

| 参数 | 设置 | 设置理由 |
|---|---:|---|
| attention budget | 128 | 给每层物理 KV cache 一个明确上限 |
| sink tokens | 4 | 保留对话开头的稳定锚点 |
| recent window | 124 | `128 - 4`，保证总 unique keys 不超过 128 |

实验中的 filler segment 长度是 128 token。这样可以明确制造“事实离开 recent window、
必须依靠长期 memory 恢复”的情况，而不是让答案偶然从局部 KV 直接复制出来。

### 7.3 Fact memory 参数

| 参数 | 设置 | 设置理由 |
|---|---:|---|
| user memory slots | 8 | 故意设置为有限容量，让临时事实与耐久事实竞争 |
| reader top-k | 4 | 允许读取少量候选，同时保留可审计的检索压力 |
| max write candidates | 1 | 每个 user round 最多一个候选，避免同一句产生多个重叠 span，便于归因 |
| merge threshold | 0.999 | 几乎不合并不同事实，让噪声真正占用 slot，而不是被相似度提前合并 |
| payload tokens | 12 | 覆盖短事实/value，同时限制每个 slot 的状态开销 |
| memory fusion layers | 代码下标 2、5（第 3、6 层） | 在中层和高层各注入一次 memory，兼顾表征变换和最终解码 |

8 个 slot 不是为了声称 8 是最优容量，而是为了构造可观察的容量压力：当耐久事实
加上 8 条临时事实时，Fixed-LRU 会进入超容量区间，而 Gated 应该拒绝临时事实。

### 7.4 训练/验证压力轴

| 轴 | 训练值 | 验证值 | 设置理由 |
|---|---|---|---|
| filler delay | 1、2、4 segments | 1、4、16 segments | 验证 16 测试长度外推 |
| temporary noise | 0、1、2、4 | 0、2、4、8 | 验证 8 测试容量外推 |
| examples/cell | — | 24 | 每个 cell 可审计且总量可控 |
| total validation | — | 12×24=288 | 同时覆盖 4 个 noise × 3 个 delay |

训练和验证使用不同的 prompt 模板；验证使用未直接出现的提示形式，避免模型只是
记住 `Remember`、`Temporary` 等固定关键词。

### 7.5 优化参数

| 参数 | 设置 | 设置理由 |
|---|---:|---|
| steps | 2,000 | 固定 screening 预算，确保八个条件可直接比较 |
| batch size | 2 | 受 GPU 显存和 memory state 计算成本限制 |
| gradient accumulation | 2 | 有效每 step 4 个 examples，同时保持 micro-batch 较小 |
| backbone LR | 3e-5 | 小幅调整父 TinyStories 表征，避免破坏语言先验 |
| memory/new-head LR | 3e-4 | 新增 gate、value 和 token head 从随机初始化开始，需要更快学习 |
| optimizer | AdamW | 对小型 Transformer 和新增控制器都稳定、常用 |
| betas | 0.9、0.95 | 与既有 TinyLM 协议保持一致 |
| weight decay | 0.1 | 抑制新增控制器和 backbone 的过拟合 |
| precision | BF16 autocast | 在 RTX 4090 上降低训练显存并提高吞吐，同时保留 FP32 router/loss 计算需要的稳定性 |

所有条件使用相同 seed=17、父 checkpoint、数据生成顺序和训练步数。这样做不是为了
得到统计显著性，而是先完成一个可复核的机制 screening。

### 7.6 Value-token 参数

| 参数 | 设置 | 设置理由 |
|---|---:|---|
| value start loss weight | 0.5 | 帮助模型定位 value 的起始 token，但不压过 LM loss |
| value length loss weight | 0.25 | 预测 value 边界，权重低于起点和 token 对齐 |
| candidate token loss | 0.5 | 训练写入前的 lexical value 表示 |
| retrieved token loss | 0.5 | 训练读取后的 lexical value 表示 |
| memory value mode | `semantic_lexical` | 同时保留语义检索能力和词级恢复能力 |
| auxiliary head | tied embedding | 使用与 LM head 相同的 token 空间，减少额外词表参数 |

这些 loss 是辅助监督，不是把模型改成 copy/pointer generator。

### 7.7 MoE 参数

| 参数 | 设置 | 设置理由 |
|---|---:|---|
| experts | 4 | 提供足够的条件容量，同时保持小模型路由可审计 |
| top-k | 2 | 允许每个 token 使用两个 expert，降低单一路由错误的风险 |
| dispatch | dropless | 不因 capacity 截断丢 token，便于把质量差异归因于模型而非丢弃 |
| load-balance weight | 0.01 | 防止单 expert collapse，同时不让路由损失主导 memory 任务 |
| padding handling | 排除 padding | padding 不应被计入 expert 使用率或 load balance |

Top-2 的理论均衡参考是：每个 expert 的 token fraction 约 0.5、dispatch fraction 约
0.25、load-balance 数值约 2。当前实现没有 fused grouped-GEMM，因此这个配置主要用于
机制验证，不代表生产级 MoE 的速度上限。

## 8. 实验结果如何解释

### 8.1 四个 Dense 条件

| 条件 | EM | target survival | Recall@1 | Recall@4 | MRR |
|---|---:|---:|---:|---:|---:|
| SWA-only | 5.21% | N/A | N/A | N/A | N/A |
| Fixed-LRU | 74.65% | 92.01% | 90.28% | 92.01% | 0.9103 |
| 旧 Gated | 52.43% | 100.00% | 100.00% | 100.00% | 1.0000 |
| Value-token Gated | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |

Dense 对照说明：

- 旧 Gated 已能选择性保留和检索事实，但词级恢复不稳定；
- Value-token alignment 把 EM 从 52.43% 提升到 100%；
- SWA-only 没有长期 memory，所以 survival/recall 应写 N/A，不是“记忆得分为 0”。

### 8.2 四个 MoE 条件

| 条件 | EM | target survival | Recall@1 | Recall@4 | MRR |
|---|---:|---:|---:|---:|---:|
| SWA-only + MoE | 5.21% | N/A | N/A | N/A | N/A |
| Fixed-LRU + MoE | 65.28% | 93.06% | 93.06% | 93.06% | 0.9306 |
| 旧 Gated + MoE | 32.99% | 100.00% | 100.00% | 100.00% | 1.0000 |
| Value-token Gated + MoE | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |

MoE 对照说明：

- MoE 没有破坏 Value-token Gated 的选择性记忆和词级恢复；
- MoE 没有自动修复旧 Gated 的答案生成问题；
- 当前任务在 Dense Value-token Gated 上已经达到 100% EM，因此 MoE 没有额外质量空间。

### 8.3 资源变化

| 版本 | 总参数 | active parameters/token | 训练 examples/s | peak allocated |
|---|---:|---:|---:|---:|
| Dense 旧 Gated | 31.85M | 31.85M | 6.05 | 2.673 GB |
| MoE 旧 Gated | 60.18M | 38.94M | 4.20 | 3.201 GB |
| Dense Value-token Gated | 32.16M | 32.16M | 5.80 | 2.682 GB |
| MoE Value-token Gated | 60.48M | 39.25M | 4.15 | 3.205 GB |

MoE 的总容量显著增加，但当前 Top-2 实现每个 token 仍要执行两个 expert，并且使用
普通 PyTorch indexing、逐 expert forward 和 `index_add_`。因此训练吞吐下降、峰值
显存增加是预期现象。MoE 总参数增加也不等于 checkpoint 或 optimizer storage 降低，
因为所有 experts 都必须保存。

## 9. 当前设计的优点与边界

### 已经被当前实验支持的部分

1. 有界 local KV 能制造真实的长期事实压力；
2. write/retention/read gate 能拒绝临时事实并保护耐久事实；
3. Value-token path 能修复旧 Gated 的词级答案恢复瓶颈；
4. MoE 可以作为 FFN 的独立替换，和事实 memory 组合运行；
5. dropless Top-2 路由在当前小模型中没有出现明显 expert collapse。

### 还不能从当前实验推出的结论

1. 只有 seed=17，不能报告统计显著性或跨 seed 稳定性；
2. 任务使用受控模板和有限颜色值，不能外推为开放域自主记忆；
3. logical active-slot bytes 不等于物理显存已经压缩，因为 slot tensor 仍预分配；
4. 当前 MoE dispatch 不是优化 kernel，吞吐不能代表生产级 MoE；
5. 四个 expert 被使用不等于已经证明了清晰、可解释的 expert specialization；
6. Value-token auxiliary head 不是 pointer/copy head。

## 10. 一句话总结

最准确的架构描述是：

> 一个带有有界 sink+recent causal attention 的 decoder-only TinyLM，在其上增加
> content-dependent 的事实写入、保留、合并、读取和融合机制；Value-token 版本
> 再增加词级 value 对齐与辅助 token 监督；MoE 版本则只把每个 Transformer block
> 的 Dense FFN 替换为 4-expert Top-2 dropless FFN。两种 gate 分属不同子系统，
> 不应混为一个“MoE gate”。

## 11. 代码和结果位置

主要实现：

- [`model.py`](../adaptive_gated_fact_memory/src/adaptive_fact_memory/model.py)：decoder block、`SparseMoE`、memory-to-token head；
- [`memory.py`](../adaptive_gated_fact_memory/src/adaptive_fact_memory/memory.py)：事实提取、write/retention/update/reader/fusion；
- [`attention.py`](../adaptive_gated_fact_memory/src/adaptive_fact_memory/attention.py)：sink+recent causal KV；
- [`state.py`](../adaptive_gated_fact_memory/src/adaptive_fact_memory/state.py)：会话、KV、fact slots 和 pending state；
- [`train_memory_stress.py`](../adaptive_gated_fact_memory/scripts/train_memory_stress.py)：固定 selective-stress 训练和评估协议。

结果汇总：

- [`ADAPTIVE_GATED_MEMORY_MOE_COMPARISON_S17_REPORT.md`](ADAPTIVE_GATED_MEMORY_MOE_COMPARISON_S17_REPORT.md)：四 Dense、四 MoE 和 Dense→MoE 对照；
- [`ADAPTIVE_GATED_MEMORY_VALUE_TOKEN_ALIGNMENT_S17_EXPERIMENT_REPORT.md`](../ADAPTIVE_GATED_MEMORY_VALUE_TOKEN_ALIGNMENT_S17_EXPERIMENT_REPORT.md)：Value-token 机制专项报告；
- `/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/`：所有 checkpoint、metrics 和 summary。

正确性测试：

```bash
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 -m unittest discover \
  -s adaptive_gated_fact_memory/tests -p 'test_*.py' -v
```

当前测试结果：23/23 passed。
