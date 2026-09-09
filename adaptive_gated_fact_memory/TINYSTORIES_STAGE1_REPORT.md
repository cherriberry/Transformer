# TinyStories 第一阶段训练报告：Adaptive Gated Fact Memory

日期：2026-09-08  
运行 ID：`tinystories_backbone_s17_10m_20260908`  
硬件：NVIDIA RTX 4090 24 GB，BF16 autocast

## 结论先行

新架构已经在固定 TinyStories 协议上完成 10M-token 骨干训练，并完整跑完验证集。完整验证结果为：

| 指标 | 结果 |
|---|---:|
| 训练 token | 10,000,000 |
| 验证 token | 4,765,917 |
| 验证 NLL | **2.889852** |
| 验证 PPL | **17.9906** |
| 训练时间 | 430.9 s（约 7.2 min） |
| 平均训练吞吐 | 23,207 token/s |
| 峰值已分配显存 | 4.66 GB |
| 参数量 | 31,854,750 |

这证明模型的 TinyLM 局部语言建模骨干可以正常学习，且数值稳定、能达到验证集。它还不能证明事实记忆、动态写入或跨话题召回能力，因为 TinyStories 没有事实跨度、对话轮次和记忆需求标签，本次训练明确关闭了 episodic memory path。

## 固定设置

- 数据：`roneneldan/TinyStories`，revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`。
- tokenizer：固定 GPT-2 tokenizer，vocab 50,257，revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`。
- 数据顺序：固定 parquet 顺序，故事末尾加入 EOS，跨文档连续打包为 512-token block。
- 数据校验依据：仓库中的 [`data/tinystories_manifest.json`](/root/BDMI/26summerBDMI_transformer/data/tinystories_manifest.json)。
- TinyLM：6 层、hidden 384、8 heads、FFN 1536、RoPE、GELU、权重绑定输出头。
- 局部通道：4 个 sink + 124 个 recent KV，单层物理 KV 上限 128。
- 优化器：AdamW，lr `3e-4`，betas `(0.9, 0.95)`，weight decay `0.1`，3% warmup，cosine decay，梯度裁剪 1.0。
- 有效 batch：8192 token（4 个 512-token 序列 × 4 个梯度累积步）。
- seed：17。

原型的初始化也在本次训练前修正为 TinyLM 风格的 0.02 正态初始化。否则 PyTorch `Embedding` 的默认单位方差会让 tied LM head 初始 logits 饱和，不能作为有效训练起点。

## 训练过程

验证探针（262,144 token）随训练下降：

| 已训练 token | NLL | PPL |
|---:|---:|---:|
| 0 | 10.9253 | 55,562.6 |
| 1,048,576 | 4.3567 | 78.00 |
| 2,097,152 | 3.7293 | 41.65 |
| 4,194,304 | 3.3466 | 28.40 |
| 6,291,456 | 3.0851 | 21.87 |
| 8,388,608 | 2.9582 | 19.26 |
| 10,000,000 | 2.9146 | 18.44 |

完整验证集的 NLL 为 2.889852，PPL 为 17.9906。最后一步也是探针最优点，因此最终 checkpoint 与 best-probe checkpoint 的验证结果一致。

## 与既有统一 TinyStories 结果的关系

同一仓库中 seed 17 的统一重跑结果如下，仅作量级参照：

| 方法 | full-val NLL | full-val PPL |
|---|---:|---:|
| Memformer | 2.845559 | 17.2112 |
| Longformer | 2.883914 | 17.8841 |
| **本架构（本次阶段一）** | **2.889852** | **17.9906** |

本架构比该 seed 的 Longformer 高约 0.00594 NLL（约 0.6% PPL），处于相近量级；但局部预算定义并不完全相同（本架构 128 个唯一 KV 中含 4 个 sink，Longformer runner 使用 left window 128），而且本架构此阶段没有计算长期记忆分支。因此不能把这张表解读为正式速度或最终架构优劣结论。

## 记忆参数是否被训练

没有。`forward_tinystories` 为本次阶段的显式入口：

1. 每个 packed block 使用全新的 conversation state；
2. 不把故事 token 当作 user/assistant 轮次；
3. 跳过事实提取、读写、保留、融合路径；
4. 只对 token-level LM loss 更新共享 embedding、局部 attention 和 FFN 骨干。

独立核查显示，最终 checkpoint 中所有记忆控制器参数与 seed 17 的初始值最大绝对差为 0，而骨干参数发生了更新。这正是预期行为，避免用无标签短故事训练出伪记忆策略。

## 产物

所有大文件在数据盘，不覆盖仓库已有的 B 角色报告：

- 运行目录：`/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_tinystories/tinystories_backbone_s17_10m_20260908/`
- [resolved config](/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_tinystories/tinystories_backbone_s17_10m_20260908/config.resolved.json)
- [raw metrics](/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_tinystories/tinystories_backbone_s17_10m_20260908/metrics.jsonl)
- [summary](/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_tinystories/tinystories_backbone_s17_10m_20260908/summary.json)
- `checkpoint.final.pt`
- `checkpoint.tokens_10000000.pt`
- `checkpoint.tokens_5005312.pt`

训练入口：`scripts/train_tinystories.py`。后续结构化对话训练应从 `checkpoint.final.pt` 初始化，并重新打开记忆模块，加入 span start/length、write/read、retention/update 和 query-answer recovery 的监督；那一阶段才评价事实召回和动态容量。
