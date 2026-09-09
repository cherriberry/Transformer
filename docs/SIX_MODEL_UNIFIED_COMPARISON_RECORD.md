# A/B/C 三部分六方案统一比较记录

记录日期：2026-09-06

## 1. 比较对象

本项目需要合并比较三个实验部分中的六个方案：

| 实验部分 | 方案 |
|---|---|
| 链接 A | Linformer、Performer |
| 本地 B | Longformer、Memformer |
| 链接 C | Reformer、Keyformer |

Full Attention 不占用六个方案名额，而是作为所有方法共享的质量、训练和推理基线。

需要特别区分：Linformer、Performer、Reformer、Longformer、Memformer是训练型架构；Keyformer是作用于已训练模型的推理期 KV-cache 压缩策略。

## 2. 当前实验的主要差异

| 实验部分 | context | 训练 token | seed | effective batch | 备注 |
|---|---:|---:|---|---:|---|
| 链接 A | 4096 | 约 10M | 17/29/43 | 32768 | Linformer、Performer、Full |
| 本地 B | 512 | 10M | 主要为17 | 8192 | TinyStories 统一 manifest 下的 Longformer、Memformer |
| 链接 C | 512 | 10M | 主要为17 | 8192 | Reformer；Keyformer为推理缓存实验 |

其他影响可比性的差异：

1. A、B、C 的 TinyStories revision、EOS/packing 规则和 tokenizer manifest 没有完全由同一份原始文件证明。
2. A 组曾记录 vocab size=50258，而统一协议应为 GPT-2 vocab size=50257。
3. A、B、C 使用的 GPU、PyTorch/CUDA 版本和计时对象不同。
4. A、B、C 的 PPL 评估 token 数不同，Keyformer使用窗口化推理评估。
5. Memformer包含额外 memory 投影和 gate，参数量高于 Longformer 公共主干。
6. 旧的 `experiments/long_context_10m_v1` synthetic fallback 使用 vocab=64，不得进入 TinyStories 质量主榜。
7. B 组已有 100M-token 扩展，但不能和其他方法的 10M-token结果混入同一主榜。

## 3. 当前结果可以支持的局部结论

- 链接 A 内部：Linformer 的 PPL 最低，Full Attention 的训练吞吐最高，Performer 的当前配置质量较差但吞吐高于 Linformer。
- 本地 B 的 10M、seed=17：Memformer 的 validation PPL 约为 17.21，Longformer 约为 17.88；Memformer训练吞吐更高、峰值显存更低，但参数更多。
- 本地 B 的 100M 扩展：Longformer PPL 约 6.10，反超 Memformer 的约 6.46，说明训练预算会改变质量排序。
- 链接 C 内部：Reformer 在 context=512 下明显劣于 Full Attention；Keyformer-50 可减少约 50% KV容量，但当前实现不保证 decode 加速。

这些局部结果不能直接组成六方案绝对总排名。

## 4. 推荐的统一主实验协议

建议以当前 `experiments/tinystories_tinylm_v1/protocol.yaml` 为基础，冻结为：

```text
dataset: 同一 TinyStories revision
tokenizer: 同一 GPT-2 tokenizer revision
vocab_size: 50257
context: 512
training_tokens: 10M
effective_batch_tokens: 8192
seeds: 17, 29, 43
layers: 6
hidden_size: 384
heads: 8
ffn_size: 1536
causal: true
position: RoPE
dtype: BF16
optimizer: AdamW
```

所有方法必须使用同一数据顺序、EOS处理、packing规则、validation token流、PPL计算方式和 benchmark 计时规则。

为了体现长上下文能力，另设 1024、2048、4096、8192 的效率和机制测试，但不能把不同 context 的质量结果混成一个主榜。

## 5. 重训和补训清单

### 严格方案

五个训练型架构全部按统一协议重跑 3 个 seed：

- Linformer：必须在 context=512 下重训；A 组 4096 结果保留为附录。
- Performer：必须在 context=512 下重训；A 组 4096 结果保留为附录。
- Longformer：已有 seed=17 可作为候选，但严格方案仍建议统一 runner 重跑 17/29/43。
- Memformer：已有 seed=17 可作为候选，但严格方案仍建议统一 runner 重跑 17/29/43，并固定 memory reset、detach 和 batch reorder 规则。
- Reformer：建议统一 runner 重跑 17/29/43，并只使用修复后的 causal LSH 实现。
- Full Attention：必须重新训练统一的共享 baseline，建议使用 17/29/43 三个 seed。
- Keyformer：不重新训练 backbone；在新的统一 Full checkpoint 上重新评估 FullKV、Keyformer-75、Keyformer-50。

### 最低计算量方案

- 保留 Longformer/Memformer 的 seed=17，补训 seed=29、43。
- Reformer只有在确认数据 manifest、tokenizer、vocab和 causal LSH实现一致后，才可保留已有 seed=17；否则三个 seed 全部重跑。
- Linformer和Performer仍需在 context=512 下重新训练。
- Full Attention至少要训练统一 seed=17版本，最好使用三个 seed。
- Keyformer只重新评估，不训练独立模型。

## 6. 排行榜定义

### 六方案总表

六个名称全部列出：

```text
Linformer / Performer / Longformer / Memformer / Reformer / Keyformer
```

Keyformer的训练字段应标记为“继承 Full Attention”或“N/A”，不能伪装成独立训练结果。

### 训练架构排行榜

比较 Full Attention、Linformer、Performer、Longformer、Memformer、Reformer，指标为：

- 质量：token-weighted validation NLL/PPL；
- 训练效率：fwd+bwd tokens/s、step time；
- 资源：peak allocated/reserved memory、参数量；
- 建议权重：质量 50%、训练吞吐 30%、显存 20%。

### 推理缓存排行榜

比较 FullKV、Keyformer-75、Keyformer-50，指标为：

- PPL保持率；
- KV cache bytes；
- prefill latency；
- decode ms/token 和 tokens/s；
- peak inference memory。

如果课程要求一个单一总分，只能将其命名为“系统方案综合分”，并明确 Keyformer 的训练成本继承自 Full Attention；该分数依赖预先声明的权重，不代表六种训练架构的绝对优劣。

## 7. 最终结论

A、B、C 三部分的六个方案可以合并成一个统一实验项目，但当前结果不能直接横向总排名。最少需要统一 context、batch、训练 token、seed、数据 manifest、评估 token 流、硬件/计时口径，并把训练型架构与 Keyformer 推理策略分开解释。

推荐主榜使用 context=512、10M token、3 seeds；长上下文长度作为独立效率附表。这样既能保留六个方案的共同比较，又不会把不同实验协议或不同物理阶段的指标误当成同一种结果。
