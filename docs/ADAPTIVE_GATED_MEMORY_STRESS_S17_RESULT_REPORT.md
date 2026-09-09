# 门控事实记忆容量压力实验报告（seed 17）

更新时间：2026-09-09  
实验协议：`adaptive_fact_memory_selective_stress_v2`  
状态：三个条件均完成 2,000/2,000 step，验证完成，15/15 项测试通过

## 1. 结论摘要

本实验回答的是一个具体问题：在局部注意力窗口相同、长期事实槽位上限固定为
8 的情况下，学习式门控能否拒绝临时事实，从而保护真正需要长期保留的事实？

seed 17 的结果支持“门控具有选择性写入和容量保护能力”，但尚不支持“门控模型
已经获得最高的最终回答准确率”。

- `gated memory` 拒绝了全部验证临时事实，始终只使用 1 个有效槽位；目标事实的
  存活率、Recall@1、Recall@4 和 MRR 均为 100%。
- `fixed-LRU memory` 接收全部临时事实；噪声达到 8 条、超过可用容量时，目标事实
  存活率降至 68.06%，Recall@1 降至 61.11%。
- `SWA+sink only` 没有长期事实槽位，在事实被推出 128-token 局部缓存后，总体
  Exact Match 仅为 5.21%。这说明本任务确实需要局部窗口之外的状态。
- 但是最终回答 Exact Match 由 fixed-LRU 取得最高值：74.65%，高于 gated 的
  52.43%。gated 的检索指标已经满分而答案仍会生成错误，说明当前主要瓶颈从
  “存什么、能否读到”转移到了“如何把读出的事实稳定地产生为答案”。
- gated 的平均逻辑用户槽位字节数比 fixed-LRU 少 76.47%；在 8 条噪声下少
  87.50%。当前实现仍预分配完整槽位张量，因此这是逻辑稀疏性，不是已经实现的
  物理显存压缩。

因此，最准确的当前表述是：**门控机制的容量选择优势已经得到 seed-17 pilot
支持；端到端准确率优势、物理内存优势和跨 seed 稳定性尚未得到证明。**

## 2. 三个比较条件

| 条件 | 局部注意力 | 长期记忆写入策略 | 容量满时行为 |
|---|---|---|---|
| SWA-only | 4 sinks + 124 recent | 不使用长期事实记忆 | 不适用 |
| Fixed-LRU | 4 sinks + 124 recent | 所有有效事实候选均写入 | 明确的 LRU 淘汰 |
| Gated | 4 sinks + 124 recent | 学习式 write/retention gate 决定是否保留 | 门控保留、合并或淘汰 |

SWA-only 使用的是接入同一 TinyLM 的 `StreamingLLM-source` sink+recent 缓存
规则。它是可审计的上游机制适配基线，不应称为论文作者的官方复现；截至实验日，
未找到 *Sliding-window beats linear attention* 作者公开的原始代码仓库。

Fixed-LRU 不依赖未训练或冻结的 retention head 做淘汰，因此它是明确且可解释的
“固定写入”消融。语义读取会刷新 LRU，所以容量满时目标并非必然被淘汰；报告的是
实际测得的存活率。

## 3. 公平性与实验配置

三个条件从同一个 seed-17 TinyStories 骨干 checkpoint 开始，并使用同一数据生成
顺序、训练步数、验证样本、TinyLM 主干、tokenizer、局部 KV 预算和 GPU 型号。
机制不同所必需的训练目标也不同：fixed-LRU 不优化 write/retention/budget gate
损失，gated 会优化这些损失，SWA-only 只优化答案 LM 损失。因此这是“相同资源下
分别训练各机制”的比较，不是逐算子完全相同的损失函数比较。

| 项目 | 设置 |
|---|---|
| seed | 17 |
| 父 checkpoint | `tinystories_backbone_s17_10m_20260908/checkpoint.final.pt` |
| 父 checkpoint SHA-256 | `d995248587a5e535af59d850f36a5bc01925c54bd9dff213f0f18bfb26f2b78f` |
| 模型 | 6 layers, hidden 384, 8 heads, FFN 1536，约 31.85M 参数 |
| 局部 KV | 总预算 128：4 sinks + 124 recent |
| 用户事实槽位 | 8 |
| 语义读取 | top-k = 4 |
| 每轮写候选 | 最多 1 个 |
| merge threshold | 0.999 |
| 训练步数 | 2,000 |
| micro batch / 梯度累积 | 2 / 2；每 step 4 个训练样本 |
| 每个 run 的训练样本数 | 8,000 |
| 优化器 | AdamW，betas=(0.9, 0.95)，weight decay=0.1 |
| 主干学习率 | 3e-5 |
| 记忆模块学习率 | 3e-4 |
| 数值精度 | RTX 4090 上 BF16 autocast |
| 物理设备 | `CUDA_VISIBLE_DEVICES=1`，进程内显示为 `cuda:0` |
| checkpoint 间隔 | 500 step，并保存 final checkpoint |

### 3.1 结构化数据

每条样本的轮次结构为：

```text
耐久事实 -> 0/若干条临时事实 -> TinyStories 长 filler -> 查询 -> 答案
```

训练耐久提示使用 `Remember`、`Keep`、`Save`，训练临时提示使用 `Temporary`、
`Skip`、`Ignore`。验证改用未在训练模板中出现的 `Store` 与 `Discard`，因此结果
至少包含提示词改写泛化，而不是对同一模板的复述。临时事实与耐久事实都使用
user role、相同的 `NAME likes VALUE` 关系形式；fixed-LRU 必须接收这些有效候选，
而 gated 接受“保留/丢弃”监督。

需要明确：重要性标签由这些显式语言提示和训练监督定义。本实验还没有证明模型能
在无提示、开放域对话中自主推断任意内容的重要性。

### 3.2 训练与验证分布

| 轴 | 训练取值 | 验证取值 | 外推条件 |
|---|---|---|---|
| TinyStories delay segments | 1, 2, 4 | 1, 4, 16 | 16 |
| 临时事实数 | 0, 1, 2, 4 | 0, 2, 4, 8 | 8 |

每个验证网格单元使用 24 条样本，共 `4 noise × 3 delay × 24 = 288` 条/模型。
每个网格单元由不同的确定性索引生成，并不是完全相同事实的逐 delay 配对，因此
小幅、非单调的 delay 波动不能解释为严格的因果退化曲线。

## 4. 总体结果

| 条件 | Exact Match | 目标存活率 | Recall@1 | Recall@4 | MRR | 平均有效槽位 | 平均逻辑用户槽位 bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gated | 52.43% | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.00** | **3,221** |
| Fixed-LRU | **74.65%** | 92.01% | 90.28% | 92.01% | 0.9103 | 4.25 | 13,689.25 |
| SWA-only | 5.21% | N/A | N/A | N/A | N/A | 0.00 | 0 |

这里的 Recall@1 是目标槽位在语义读取排序中为第 1 名；Recall@4 是目标槽位被
top-4 读出的比例。SWA-only 没有显式事实槽位，所以这些记忆内部指标是“不适用”，
而不是可与另外两者等价解释的 0 分。

总体上：

- Gated 相对 SWA-only 的 EM 提高 47.22 个百分点。
- Fixed-LRU 相对 SWA-only 的 EM 提高 69.44 个百分点。
- Fixed-LRU 的 EM 比 Gated 高 22.22 个百分点。
- Gated 相对 Fixed-LRU 平均减少 3.25 个活动槽位，逻辑活动槽位字节减少
  10,468.25 bytes，即 76.47%。

## 5. 按临时事实数量分析

每一行聚合 3 个 delay，共 72 条验证样本/模型。

| 临时事实数 | Gated EM | Gated 存活/R@1 | Gated 槽位 | Fixed EM | Fixed 存活 | Fixed R@1 | Fixed 槽位 | SWA EM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48.61% | 100% / 100% | 1 | 73.61% | 100% | 100% | 1 | 2.78% |
| 2 | 51.39% | 100% / 100% | 1 | 83.33% | 100% | 100% | 3 | 6.94% |
| 4 | 54.17% | 100% / 100% | 1 | 80.56% | 100% | 100% | 5 | 4.17% |
| 8 | 55.56% | **100% / 100%** | **1** | 61.11% | **68.06%** | **61.11%** | **8** | 6.94% |

在存在临时事实的条件中，Gated 的临时事实接受率为 0%，Fixed-LRU 为 100%。
前 7 个临时事实加 1 个目标事实尚能装入 8 槽容量；第 8 个临时事实使系统进入
超容量区间。正是在这个区间，Fixed-LRU 的目标存活和读取开始明显下降，而
Gated 仍维持单槽位和完整检索。

在 8 条噪声下，Gated 使用 3,221 个逻辑用户槽位 bytes，Fixed-LRU 使用
25,768 bytes，前者少 87.50%。这直接支持“选择性写入避免无关事实耗尽固定容量”。

但是 Gated 在所有 noise 桶中的 EM 都低于 Fixed-LRU；即使 noise=0，差距仍为
25.00 个百分点。这说明当前 EM 差异并非仅由过容量噪声造成，不能用本实验声称
门控已经提升整体答案质量。

## 6. 按延迟长度分析

每一行聚合 4 个 noise 条件，共 96 条验证样本/模型。

| delay segments | Gated EM | Gated 存活/R@1 | Fixed EM | Fixed 存活 | Fixed R@1 | SWA EM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 47.92% | 100% / 100% | 78.13% | 94.79% | 91.67% | 6.25% |
| 4 | 58.33% | 100% / 100% | 69.79% | 88.54% | 87.50% | 3.13% |
| 16 | 51.04% | **100% / 100%** | 76.04% | 92.71% | 91.67% | 6.25% |

Gated 在训练最大 delay=4 之外的 delay=16 仍保持 100% 存活和读取，说明显式
事实状态没有随 filler 长度衰减。Fixed-LRU 的主要失败发生在噪声超容量时；按
delay 聚合后没有单调下降。SWA-only 一直接近偶然命中水平。

## 7. 效率与状态开销

| 条件 | 训练时间 | 训练吞吐 | peak allocated | peak reserved | `state_bytes` |
|---|---:|---:|---:|---:|---:|
| Gated | 1,323.02 s（22.05 min） | 6.05 examples/s | 2.673 GB | 6.648 GB | 2,408,906 |
| Fixed-LRU | 1,200.59 s（20.01 min） | 6.66 examples/s | 2.666 GB | 6.568 GB | 2,408,906 |
| SWA-only | 880.05 s（14.67 min） | 9.09 examples/s | 2.871 GB | 6.881 GB | 2,408,906 |

训练时间只计 2,000 个优化 step；显存峰值计数在训练前清零、在验证结束后读取，
因此包含训练和验证阶段可能出现的最高值。单次 allocator 峰值受执行路径和缓存
影响，不能因为 SWA-only 的该数值略高就推断长期记忆节省了 GPU 显存。

Gated 比 Fixed-LRU 训练时间长约 10.20%，吞吐低约 9.25%；比 SWA-only 训练
时间长约 50.34%。这是当前未融合的 PyTorch 原型成本，而不是优化 kernel 后的
理论复杂度结论。

`state_bytes=2,408,906` 对 Gated 和 Fixed-LRU 完全相同，因为当前状态对象为全部
8 个用户槽位预分配张量。Gated 的 3,221 bytes 是“当前活动槽位对应的逻辑数据
量”；除非后续加入 packed/ragged storage 或物理回收，它不会自动变成进程显存
下降。因此当前结果不能表述为已经实现物理 KV/状态压缩。

## 8. 对目标架构的支持与不支持

### 已获得支持

1. **选择性写入可行**：门控能把耐久事实与形式相近的临时 user facts 分开。
2. **容量保护有效**：在超过 8 槽容量的验证条件下，门控保持 100% 目标存活，
   固定写入+LRU 降至 68.06%。
3. **语义读取稳定**：Gated 的 Recall@1、Recall@4、MRR 在所有 12 个条件都满分。
4. **长度外推成立于当前任务**：训练到 4 段，验证 16 段仍能保留和读出事实。
5. **逻辑状态稀疏**：相同上限下只激活必要槽位，不让临时事实线性占满容量。
6. **长期状态确有必要**：SWA-only 无法可靠回答已经离开局部缓存的事实。

### 尚未获得支持

1. **端到端准确率领先**：Fixed-LRU 的 EM 当前显著高于 Gated。
2. **物理内存或 KV 显存压缩**：槽位张量仍完整预分配，峰值显存也没有下降证据。
3. **完全自主的重要性判断**：当前训练有明确的 durable/temporary 监督和语言提示。
4. **跨 seed 稳定性**：只有 seed 17，不能给出均值±标准差或显著性结论。
5. **最优训练状态**：运行固定 2,000 step 并评估 final checkpoint，没有基于独立
   validation EM 的 early stopping 或 best-checkpoint 选择。
6. **完整效率结论**：尚缺统一的单 token 解码延迟、长序列长度扫描和优化 kernel。
7. **论文官方复现**：SWA 是 source-policy adapter，不是论文作者代码结果。

## 9. 为什么“检索 100%”但“EM 52.43%”

Gated 的目标事实始终存在、始终排在读取第 1 位，也始终进入 top-4，但只有约一半
答案完全匹配。这排除了“目标已经被遗忘”或“语义检索没有找到目标”作为主要原因。
当前最可能需要继续检查的是记忆 value 到 decoder 的融合强度、答案 token 的生成
监督，以及生成阶段是否能稳定利用已经读出的 payload。

现阶段不应仅凭一次运行指定唯一原因。一个直接的诊断闭环是同时报告：

- oracle：把正确目标槽位强制送入 decoder 后的 EM；
- 正常 gated read 的 EM；
- 禁用 memory fusion 的 EM；
- 在“检索命中”子集上的条件 EM；
- teacher-forced answer NLL 与自由生成 EM。

这能把问题分解为“写入—保留—读取—融合—生成”，而不是把满分 Recall 误当成
满分任务准确率。

## 10. 建议的后续优先级

1. **先修复/增强 answer utilization，再重复同协议。** 当前最有价值的工作不是扩大
   噪声范围，而是让已正确读出的事实可靠影响答案；用上述 oracle/fusion 诊断定位。
2. **补 seeds 29、43。** 三个策略都应使用各自对应 seed 的 TinyStories 骨干，
   形成 `3 policies × 3 seeds` 的均值±标准差；不要只改变结构化阶段的随机数。
3. **加入更弱提示的重要性测试。** 去除 `Remember/Discard` 这类直接标签，改用
   后续是否真正被查询、跨轮冲突或任务相关性来监督，验证更接近“自主重要性”。
4. **做容量扫描。** 至少测试 slots=4/8/16 与 noise/slots 比例，画质量—逻辑状态
   Pareto 曲线，检验门控优势是否随容量压力稳定扩大。
5. **实现物理压缩后再做效率主张。** 使用紧凑活动槽位存储，并统一测量 state bytes、
   peak memory、prefill/decode latency 和 tokens/s。

## 11. 可复核产物

三个正式 run 均位于数据盘，且没有覆盖旧实验：

- Gated：`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/selective_stress_v2_gated_s17_20260909/`
- Fixed-LRU：`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/selective_stress_v2_fixed_lru_s17_20260909/`
- SWA-only：`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/selective_stress_v2_swa_only_s17_20260909/`

每个目录都包含：

- `config.resolved.json`：完整配置、环境、父 checkpoint 哈希和代码哈希；
- `metrics.jsonl`：2,000 个 step 的训练记录；
- `checkpoint.step_{500,1000,1500,2000}.pt`；
- `checkpoint.final.pt`；
- `summary.json`：288 条逐样本验证记录、分条件聚合及资源指标。

第一次诊断用的 `selective_stress_gated_s17_20260909` 存在模板长度泛化和错误 retention
清除问题，保留用于审计，但不属于本报告的正式比较，也不应引用为最终结果。

测试命令：

```bash
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 -m unittest discover \
  -s adaptive_gated_fact_memory/tests -p 'test_*.py' -v
```

结果：15 tests，全部通过。
