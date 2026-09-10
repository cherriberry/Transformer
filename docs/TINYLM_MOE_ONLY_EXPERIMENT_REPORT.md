# TinyLM 全层 MoE 实验报告

日期：2026-09-09  
实验协议：`tinystories_tinylm_moe_v1`  
训练设备：NVIDIA GeForce RTX 4090 24GB，`cuda:0`  
随机种子：17  
训练预算：每个 backbone 10,000,000 prediction tokens

## 摘要

本实验在统一的六层 TinyLM 中，将每一层原有的 Dense FFN 替换为 4-expert、Top-1、dropless sparse MoE。实验覆盖 Memformer、Linformer、Performer、Longformer、Reformer 和 Full Attention 六种 attention/backbone 配置。其中 Full Attention+MoE checkpoint 作为 Keyformer 的共享训练 backbone；Keyformer 本身是推理期 KV-cache policy，不是独立训练模型。

六个 MoE backbone 均完成 10M-token 训练和完整 validation。Memformer+MoE 获得最低 validation PPL（15.7831），Longformer、Full Attention、Linformer、Reformer 和 Performer 的 PPL 分别为 17.6290、18.3174、18.6405、19.7071 和 31.7477。所有模型都使用了四个 expert，但路由比例存在一定偏向，尚未观察到全局 expert collapse。

本实验的 correctness 状态为 `passed`：前向、loss、梯度均有限，因果 future-invariance 通过，FullKV 和 Keyformer ratio=1 数值等价检查通过。当前实现是通用 PyTorch token dispatch，并非 fused MoE kernel；因此结果同时反映 MoE 配置和当前实现的工程成本。

## 1. 实验目标

本实验只研究以下问题：

1. 将全部六层 Dense FFN 替换为统一的全层 MoE 后，六种 backbone 是否都能稳定训练？
2. 4-expert、Top-1、dropless routing 下，各 backbone 的质量、吞吐和显存如何？
3. expert 是否被有效使用，是否出现明显路由塌缩？
4. Full-Attention+MoE 是否可以与 Keyformer 推理期 KV-cache 压缩组合？

本报告不把 Dense FFN、100M 扩展和其他协议的结果混入主表；需要跨协议比较时，请参考 [总体报告](TINYLM_SIX_FORMER_OVERALL_REPORT.md)。

## 2. MoE 模型结构

### 2.1 公共 TinyLM 骨架

所有六个 MoE 配置共享以下基础结构：

| 项目 | 设置 |
|---|---:|
| Transformer layers | 6 |
| hidden size | 384 |
| attention heads | 8 |
| head dimension | 48 |
| context | 512 |
| RoPE maximum position | 32,768 |
| expert FFN hidden size | 1,536 |
| normalization | Pre-LN |
| activation | GELU |
| residual dropout | 0.1 |
| embedding/output | tied |

六个模型共用 embedding、RoPE、LayerNorm、residual、输出 projection、数据流和训练流程；仅 attention/memory adapter 不同。MoE 位于每个 Transformer block 的 FFN 子层，不修改 attention 结构。

### 2.2 每层 MoE

每一层的 FFN 由一个 router 和四个独立 expert 组成：

```text
hidden state
    ↓
router: 384 → 4
    ↓ Top-1 assignment
Expert 1/2/3/4:
    384 → 1536 → GELU → 384
    ↓
residual connection
```

四个 expert 的网络结构完全相同，但参数独立、初始化独立、训练独立。每个 Transformer 层有自己的一组四个 expert；不同层和不同 backbone 之间不共享 expert 权重。

### 2.3 Routing 配置

| 项目 | 设置 |
|---|---|
| experts per layer | 4 |
| routing | token-level Top-1 |
| capacity | 无容量截断 |
| dispatch | dropless，不丢弃 token |
| load-balance coefficient | 0.01 |
| router z-loss coefficient | 0.001 |
| MoE layers | 6/6 层 |

Top-1 表示每个 token 在每层只执行一个 expert。Dropless 表示 token 不会因为 expert capacity 溢出而被丢弃，但不保证每个小 batch 的每个 expert 都收到 token。

## 3. 数据与训练协议

| 项目 | 设置 |
|---|---|
| dataset | `roneneldan/TinyStories` |
| tokenizer | GPT-2，vocab=50,257，EOS=50,256 |
| packing | 每篇故事追加 EOS 后连续 packing |
| train cache | 10,000,385 tokens |
| 实际训练 | 10,000,000 prediction tokens |
| validation cache | 4,765,918 tokens |
| 完整 validation | 4,765,917 next-token predictions |
| optimizer | AdamW |
| peak learning rate | `3e-4` |
| schedule | 3% warmup + cosine decay 至 `3e-5` |
| betas | `(0.9, 0.95)` |
| weight decay | 0.1 |
| gradient clip | 1.0 |
| parameter dtype | FP32 |
| compute dtype | BF16 autocast |
| effective batch | 8,192 prediction tokens |
| seed | 17 |

验证指标只报告 token-weighted language-model cross-entropy 及其 PPL，不将 router auxiliary loss 加入 validation NLL。

## 4. 六个 MoE backbone 结果

下表全部来自同一 MoE protocol、seed=17、10M training predictions 和完整 validation。

| backbone | 角色 | Val NLL | Val PPL | 训练 tok/s | Peak allocated | 总参数 | active params/token |
|---|---|---:|---:|---:|---:|---:|---:|
| **Memformer** | trainable backbone | **2.758941** | **15.7831** | 12,201.3 | 2.960 GiB | 54,709,632 | 33,475,968 |
| Longformer | trainable backbone | 2.869545 | 17.6290 | 12,265.3 | 7.234 GiB | 51,168,384 | 29,934,720 |
| Full Attention | Keyformer shared backbone | 2.907851 | 18.3174 | 26,503.8 | 2.911 GiB | 51,168,384 | 29,934,720 |
| Linformer | trainable backbone | 2.925335 | 18.6405 | 20,340.2 | 2.913 GiB | 51,168,384 | 29,934,720 |
| Reformer | trainable backbone | 2.980978 | 19.7071 | **26,821.2** | **2.900 GiB** | 50,283,648 | 29,049,984 |
| Performer | trainable backbone | 3.457819 | 31.7477 | 20,358.5 | 3.120 GiB | 51,168,384 | 29,934,720 |

训练 tok/s 是 training workflow throughput，包含周期性 validation probe；不包括训练结束后的完整 validation。该指标包含当前 attention adapter 和 MoE dispatch 的实际工程开销。

### 4.1 结果排序

按 validation PPL 从低到高：

```text
Memformer < Longformer < Full Attention < Linformer < Reformer < Performer
```

按训练 workflow throughput 从高到低：

```text
Reformer > Full Attention > Performer ≈ Linformer > Longformer ≈ Memformer
```

这两个排序不同，说明 MoE 质量和训练速度受 attention/memory adapter 以及 dispatch 实现共同影响。

## 5. Expert 路由行为

正式训练期间累计路由比例如下：

| backbone | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|---|---:|---:|---:|---:|
| Memformer | 21.09% | 22.54% | 34.77% | 21.60% |
| Longformer | 21.66% | 26.98% | 28.83% | 22.52% |
| Full Attention | 20.78% | 30.50% | 27.62% | 21.11% |
| Linformer | 22.46% | 27.54% | 27.55% | 22.44% |
| Reformer | 19.11% | 30.41% | 26.59% | 23.90% |
| Performer | 22.72% | 20.82% | 36.03% | 20.43% |

观察结果：

- 六个 backbone 的四个 expert 都获得了正式训练 token；
- 没有出现所有 token 集中到一个 expert 的全局 collapse；
- Memformer 和 Performer 对 Expert 3 有较明显偏向；
- Full Attention 和 Reformer 对 Expert 2 有较明显偏向；
- 这些比例是全局聚合值，不代表每一层、每个 batch 都均匀；
- 当前实验没有提供按层、按 token 类型的专家语义解释。

## 6. MoE 的参数与计算特征

普通 backbone 的 Dense/MoE 参数规模对照如下：

| 项目 | Dense FFN | 4-expert MoE |
|---|---:|---:|
| 总参数 | 29,925,504 | 51,168,384 |
| 总参数变化 | — | +70.99% |
| active parameters/token | 约 29.93M | 29,934,720 |
| active 参数变化 | — | 约 +0.031% |

MoE 的核心特征是：

```text
总模型容量增加很多
每个 token 每层只激活一个 expert
单 token 激活参数近似 Dense
```

但是总参数仍然会增加 checkpoint、optimizer state 和参数存储成本。当前实现使用 token indexing、逐 expert 前向和 `index_add_` 聚合，因此 active parameters/token 接近 Dense 不代表实际 wall-clock 速度也接近 Dense。

## 7. Full-Attention+MoE 与 Keyformer

Full Attention+MoE 是一个独立训练的 MoE backbone。Keyformer 在该 checkpoint 上执行推理期 KV-cache 压缩，不重新训练 attention 或 expert。

### 7.1 评估设置

- validation predictions：32,768
- context：512
- prefill：128 tokens
- FullKV cache ratio：1.0
- Keyformer cache ratio：0.5
- recent ratio：0.5

### 7.2 结果

| 策略 | Cache ratio | NLL | PPL | Peak cached tokens | Peak KV bytes | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| FullKV | 1.00 | 2.602441 | 13.4966 | 512 | 4,718,592 | 125.4 |
| Keyformer-50 | 0.50 | 2.601262 | 13.4807 | 256 | 2,359,296 | 103.8 |

Keyformer-50 相对 FullKV：

- KV bytes 减少 50%；
- peak cached tokens 减少 50%；
- NLL 变化为 `-0.001179`；
- PPL 变化为 `-0.0159`；
- 吞吐比例为 `0.8277`，下降约 17.23%。

这里轻微的 PPL 改善不能解释为压缩提高了模型能力，更合理的解释是当前窗口评估波动或 token 保留策略的偶然影响。当前实现证明了物理 cache 节省，但没有证明 decode 加速。

## 8. Correctness 与稳定性

聚合 correctness 状态为 `passed`，包括：

- 六层 MoE 前向输出 shape 正确；
- logits、loss 和 gradients 均有限；
- 所有方法的 future-invariance 最大差异为 0；
- FullKV prefill 最大误差：`0.00390625`；
- FullKV decode 最大误差：`0.001953125`；
- Keyformer ratio=1 与 FullKV 的 prefill/decode 最大差异：均为 0；
- 六个正式训练 run 的四个 expert 均获得非零全局路由量。

需要区分“dropless”和“均匀使用”：dropless 只表示不因容量上限丢弃 token；它不保证每个 batch 的四个 expert 都收到 token，也不保证四个 expert 的路由比例完全相等。

## 9. 仅针对本次 MoE 的结论

1. 全六层 4-expert、Top-1、dropless MoE 可以稳定嵌入六种 TinyLM backbone。
2. MoE 只替换 FFN 子层；各 backbone 的 attention/memory 结构保持独立。
3. 四个 expert 结构相同但参数独立；不同层、不同 backbone 之间不共享 expert 权重。
4. Memformer+MoE 在本次 seed=17 screening 中获得最低 PPL；Performer+MoE 仍然最弱。
5. 四个 expert 都被使用，但路由存在 backbone-specific 偏向，尚不能解释为专家已经学习了明确语义分工。
6. Top-1 使 active parameters/token 接近 Dense，但当前非融合 dispatch 仍导致实际训练吞吐和显存成本上升。
7. Full-Attention+MoE 可以作为 Keyformer 的训练 backbone；Keyformer-50 能把物理 KV cache 减半，但当前实现产生额外 decode 开销。

## 10. 局限性

1. MoE 目前只有 seed=17，没有 seeds 29/43 的均值、标准差和 paired-seed 检验。
2. 训练预算为 10M tokens，属于 screening，不是充分收敛训练。
3. 当前 runner 的 DenseFFN 仅作为备用模块，正式六模型结果没有在同一 runner 中完成 Dense/MoE 双模式配对。
4. Reformer 使用当前 MoE runner 的 deterministic same-bucket causal mask；不能将其结果直接视为其他 Reformer 实现的普遍表现。
5. 路由统计是全局聚合，缺少按层、按 token 类型、按训练阶段的 expert 分析。
6. 当前没有 fused grouped-GEMM、专业 sparse dispatch 或 fused cache-selection kernel，吞吐结果代表当前 PyTorch 实现。
7. Keyformer 只在 32,768 个 validation predictions、context=512 的窗口协议上评估，不能证明长事实依赖下的无损压缩。

## 11. 下一步 MoE 实验

1. 补跑 seeds 29/43，报告每个 backbone 的均值、标准差和路由稳定性。
2. 在同一 runner 中增加 `ffn_type=dense/moe`，形成严格 Dense–MoE 消融。
3. 增加 `num_experts=1/2/4/8`，绘制质量—容量—速度曲线。
4. 做参数匹配实验，区分专家容量收益与 routing 收益。
5. 记录每层 expert load、router entropy、top probability 和 token 类别分布。
6. 接入 fused grouped-GEMM/dispatch，重新测训练吞吐和峰值显存。
7. 在 Full-Attention+MoE 上评估 Keyformer-75、Keyformer-50 和更长 decode。
8. 增加 passkey、copy、associative recall 和跨 segment 检索任务。

## 12. 可追溯文件

- 配置：[moe_seed17.yaml](../experiments/tinystories_tinylm_moe_v1/moe_seed17.yaml)
- runner：[run_moe_six.py](../experiments/tinystories_tinylm_moe_v1/run_moe_six.py)
- 聚合摘要：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/SUMMARY.md`
- 完整 JSON：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/summary.json`
- correctness：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/correctness.json`
- Keyformer evaluation：`/root/autodl-tmp/26summerBDMI_transformer/aggregate/tinystories_tinylm_moe_v1/keyformer_evaluation.json`
- checkpoints：`/root/autodl-tmp/26summerBDMI_transformer/runs/tinystories_tinylm_moe_v1/`
- runner SHA-256：`423428daef532c510580ccf91ebc17a36724b9749fe9ae623ec41a54570a623c`

文档索引：[docs/README.md](README.md)
