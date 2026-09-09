# C v2：Reformer 与 Keyformer 统一复现实验结果

## 结论

本报告只汇总 protocol v2。它固定 TinyStories token 流、模型规模、训练预算、FP32 参数/BF16 autocast 和三随机种子，因此 Full Attention 与因果 LSH Reformer 的训练结果可直接比较。Keyformer 不作为独立训练网络参与这一排名；它在同一个 Full Attention 检查点上比较 KV 缓存压缩带来的速度、显存与困惑度变化。旧 v1 报告保持不变。

## 统一实验条件

- 数据：TinyStories，revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`。训练流固定为前 10,000,000 个预测 token，验证流固定为完整保存的连续 token 流。
- 分词器：GPT-2，revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`，词表 50,257，EOS=50,256；每篇故事末尾追加一个 EOS 后连续拼接。
- 共同网络：6 层、hidden size 384、8 个注意力头、FFN 1536、RoPE、causal decoder、权重绑定。
- 训练：FP32 参数 + CUDA BF16 autocast，AdamW（β1=0.9、β2=0.95、weight decay=0.1），峰值学习率 3e-4，global batch=8,192 token，训练 10M predictions；seed=17/29/43。TF32 关闭。
- 精确计数：训练 cache 10,000,385 tokens，实际训练 prediction 10,000,000；验证 cache 4,765,918 tokens，完整验证 prediction 4,765,917。
- Reformer：bucket size=32，4 次哈希，recent window=32。候选键须满足“任一哈希同桶或位于最近窗口”且键位置不晚于查询位置。
- 数据清单：训练 token SHA256 `a25e6c436b1cfcde1bcee54724563931763b40e034abd99e63d2286dbdf20f17`；验证 token SHA256 `d5e1575253e3489e2af831dee209d5ad56a6bb9a4bf02854d66fc80dddc2a9e2`。

## 训练网络的统一结果

最终验证使用各 seed 的 best-probe checkpoint，并在固定完整验证流上重新计算。主表吞吐是 workflow throughput，包含训练、周期性 probe 和 checkpoint；表后另列纯 step throughput，只统计去掉前 20 步后的优化步骤。显存为训练过程的 peak allocated memory。

| 方法 | 参数量 | seed | 验证 NLL | 验证 PPL | workflow 吞吐（token/s） | 平均 step 秒 / 峰值显存 GiB |
|---|---:|---|---:|---:|---:|---:|
| Full Attention | 29925504 | 17 / 29 / 43 | 2.942 ± 0.006 | 18.95 ± 0.12 | 53770 ± 585 | 0.134 / 2.58 |
| Causal LSH Reformer | 29040768 | 17 / 29 / 43 | 3.024 ± 0.008 | 20.58 ± 0.16 | 46939 ± 954 | 0.155 / 2.57 |

纯 step 吞吐（用于诊断，不作为主表排名）：Full Attention 61193 ± 601 token/s, Causal LSH Reformer 52901 ± 906 token/s.

## Keyformer 缓存评测

使用 Full Attention seed=17 的 best-probe checkpoint。每个窗口先 prefill 256 token，再自回归 decode 256 token；前 10 个窗口预热，剩余窗口取中位数计时。Keyformer-100 用于核对压缩路径与 FullKV 的数值等价性。

| 策略 | 缓存比例 | NLL | PPL | KV cache 平均字节 | Prefill 中位数 s | Decode ms/token | Decode token/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| FullKV | 1.00 | 2.702 | 14.91 | 4709376 | 0.0053 | 4.549 | 219.8 |
| Keyformer-100 | 1.00 | 2.702 | 14.91 | 4709376 | 0.0053 | 4.547 | 219.9 |
| Keyformer-75 | 0.75 | 2.702 | 14.91 | 3538944 | 0.0053 | 4.950 | 202.0 |
| Keyformer-50 | 0.50 | 2.701 | 14.89 | 2359296 | 0.0053 | 5.430 | 184.2 |

FullKV 与 Keyformer-100 的 NLL 绝对差为 0.000000，等价性检查：通过。

## 检查与边界

- 每个训练 run 在开始前均已检查长度 64、128、512 的前向有限性、未来 token 不影响历史位置、有限梯度和检查点恢复；详细数值保存在各 run 的 `metrics.jsonl` 与 `final.json`。
- Reformer 的实现采用可审计的因果 LSH 候选掩码，便于验证语义正确性；它不是专用 CUDA 稀疏注意力内核。因此本次吞吐结果衡量的是该统一实现，不能外推为所有 LSH 内核的极限性能。
- Keyformer 的结论属于推理缓存策略：它不改变训练后的主干参数，不能与 Full Attention、Reformer 当作三个可训练结构用单一总分排序。

## 结果文件

- `runs/<method>_seed<seed>_10m/config.json`：固定配置与配置哈希。
- `runs/<method>_seed<seed>_10m/environment.json`：软件与硬件环境。
- `runs/<method>_seed<seed>_10m/metrics.jsonl`：探测、最终验证和正确性记录。
- `runs/full_attention_seed17_10m/keyformer_eval.json`：Keyformer 的逐策略数据。
- `aggregate/c_unified_summary.json`：汇总结果。
