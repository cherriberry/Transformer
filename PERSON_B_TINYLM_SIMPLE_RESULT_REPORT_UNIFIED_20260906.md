# B 角色实验结果速览（统一三 seed 版本）

日期：2026-09-06  
任务：TinyLM + TinyStories validation screening  
硬件：单张 NVIDIA GeForce RTX 4090 24GB

> 本文件是新建的简略版，不覆盖原有 [PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md](PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md)。

## 一句话结论

在统一的 10M-token、context=512、三 seed 实验中，Memformer 的 validation 略好、训练吞吐更高、训练峰值显存更低；但它多 11.83% 参数。训练到 100M tokens 后，Longformer 的 validation 反超 Memformer，而 Memformer 的资源优势仍保留。因此不存在“一个方法全面胜出”的结论。

## 1. 实验条件

| 项目 | 设置 |
|---|---|
| 数据集 | `roneneldan/TinyStories`，固定 revision |
| tokenizer | GPT-2，vocab=50,257，EOS=50,256 |
| TinyLM | 6 层 / hidden=384 / 8 heads / FFN=1536 / RoPE |
| context | 512 |
| 训练预算 | 10,000,000 next-token predictions |
| effective batch | 8,192 tokens（micro=4，accumulation=4） |
| 优化 | AdamW，lr=3e-4，3% warmup，cosine，BF16 |
| seeds | 17、29、43 |
| 完整 validation | 4,765,917 predictions |
| Longformer | left window=128，global=0 |
| Memformer | segment=128，slots=64 |

所有 12 个正式 run（4 方法 × 3 seeds）均完成 10M 训练 predictions 和完整 validation，状态为 `ok`；无 OOM/NaN。

## 2. 四模型统一结果

箭头表示指标方向：NLL/PPL/显存越低越好，吞吐越高越好。

| 方法 | Val NLL ↓ | PPL ↓ | 训练吞吐 ↑ | Peak allocated ↓ | 参数量 |
|---|---:|---:|---:|---:|---:|
| **Memformer** | **2.8491 ± 0.0047** | **17.2732 ± 0.0804** | 14,669.5 ± 300.5 | **2.633 GiB** | 33.47M |
| **Longformer** | 2.8855 ± 0.0046 | 17.9129 ± 0.0825 | 13,498.8 ± 268.5 | 6.799 GiB | 29.93M |
| **Linformer** | 2.9338 ± 0.0060 | 18.7995 ± 0.1134 | **27,857.8 ± 106.0** | 2.580 GiB | 29.93M |
| **Performer** | 3.4505 ± 0.0083 | 31.5161 ± 0.2606 | 27,079.0 ± 541.2 | 2.704 GiB | 29.93M |

质量排序：**Memformer > Longformer > Linformer > Performer**。  
训练吞吐排序：**Linformer > Performer > Memformer > Longformer**。

## 3. B 角色重点：Longformer vs Memformer

| 指标 | Longformer | Memformer | Memformer 相对 Longformer |
|---|---:|---:|---:|
| 参数量 | 29,925,504 | 33,466,752 | +11.83% |
| Validation NLL | 2.8855 | **2.8491** | 低约 1.26% |
| Validation PPL | 17.9129 | **17.2732** | 低约 3.57% |
| 训练吞吐 | 13,498.8 tok/s | **14,669.5 tok/s** | 高约 8.67% |
| Peak allocated | 6.799 GiB | **2.633 GiB** | 低约 61.28% |

含义：10M screening 中 Memformer 的表现更好，但它使用更多参数，因此不能把差异完全归因于 memory 机制。

## 4. 100M-token 扩展

两个 B 模型使用同一冻结配置、seed=17，从初始化重新训练到 100M tokens：

| 指标 | Longformer | Memformer | 当前占优 |
|---|---:|---:|---|
| 完整 Val NLL | **1.807748** | 1.865966 | Longformer |
| 完整 Val PPL | **6.096701** | 6.462175 | Longformer |
| 训练吞吐 | 14,743.8 tok/s | **16,813.0 tok/s** | Memformer |
| 训练时间 | 113.04 min | **99.13 min** | Memformer |
| Peak allocated | 6.80 GiB | **2.63 GiB** | Memformer |

结论：训练预算会改变质量排序；Memformer 的资源优势在 100M 仍存在。100M 只有一个 seed，且曲线末端仍在下降，不能称为充分收敛。

## 5. 长序列与机制结果

### 长序列前向

冻结 checkpoint 的完整 TinyLM forward（batch=1、BF16、含词表 logits）：

| 长度 | Full SDPA | Longformer | Memformer |
|---:|---:|---:|---:|
| 512 | **3.873 ms** | 14.520 ms | 26.362 ms |
| 4,096 | **5.473 ms** | 74.896 ms | 197.842 ms |
| 32,768 | **67.832 ms** | 557.841 ms | 1,509.627 ms |

当前 Python/PyTorch 实现中，Full SDPA 仍最快；Longformer 比 Memformer 快；这反映工程 kernel 和调度开销，不代表理论复杂度失效。

### 早期信息影响末端

替换最早 128 个 token 后，观察最后 token 的最大 logit 差：

| 长度 | Longformer | Memformer |
|---:|---:|---:|
| 512 | 0.0625 | 0.2656 |
| 1,024 | **0** | 0.0625 |
| 2,048 | **0** | 0.0625 |
| 4,096 | **0** | 0.03125 |

Longformer 的约 `6×128=768` 层叠传播上限得到体现；Memformer 可以跨 segment 传递影响，但影响随固定槽压缩而衰减。非零影响不等于正确找回事实。

## 6. 方法优缺点

| 方法 | 优点 | 代价 |
|---|---|---|
| Longformer | 参数少；局部细节路径清晰；100M 质量更好；当前前向较快 | 训练峰值显存高；远距传播有硬边界；未融合实现仍慢于 Full SDPA |
| Memformer | 10M 质量略好；训练更省峰值显存；训练吞吐更高；有跨 segment 路径 | 参数多；固定槽会压缩/遗忘细节；当前前向调度较慢 |
| Linformer | 当前训练吞吐最高；规则低秩路径 | 质量低于 B 两模型；rank 需要进一步消融 |
| Performer | 训练吞吐高；参数量不增加 | 当前 features=384 配置 PPL 最差；随机特征稳定性需继续研究 |

## 7. 对目标架构的支持

结果支持以下设计判断：

1. 只用固定局部窗口，远距信息最终会不可达。
2. 固定容量 memory 能跨越局部窗口，但会出现压缩衰减。
3. “模型自主选择保留重要内容”仍有明确空间：当前 Memformer 没有动态 active slots、hard top-k、slot eviction/merge 或语义 payload。

因此 Memformer 适合作为固定槽基线；下一阶段应加入 passkey/copy/associative recall、动态槽使用率、等参数对照和真实 state/latency 测量。

## 8. 必须保留的限制

- 统一质量结果只对应 context=512 和 10M tokens；不是充分收敛结果。
- Memformer 参数多 11.83%，不是严格等参数比较。
- 三 seed 使用固定数据顺序，std 不包含数据顺序方差。
- 100M 扩展只有 seed=17。
- 长度 512–32K 是 forward/机制诊断，不是长上下文训练质量。
- TinyStories 没有官方 test split，表中 PPL 是 validation PPL。
- Reformer、Keyformer 尚未按本 unified protocol 纳入；不能把旧结果直接并入六模型主榜。

## 9. 结果文件

- 详细新报告：[PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md](PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md)
- 原详细报告：[PERSON_B_TINYLM_REPORT.md](PERSON_B_TINYLM_REPORT.md)
- 原简略报告：[PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md](PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md)
- 统一聚合表：[summary.csv](/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/summary.csv)
- 逐 run 结果：[per_run.csv](/root/autodl-tmp/26summerBDMI_transformer/aggregate/unified_ab_rerun/per_run.csv)
- 统一 runner：[run_unified_ab.py](experiments/tinystories_tinylm_v1/run_unified_ab.py)

