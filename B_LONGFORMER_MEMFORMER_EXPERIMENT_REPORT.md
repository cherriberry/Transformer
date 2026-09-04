# B 角色实验总结报告：Longformer 与 Memformer

> 历史 attention-level / synthetic-pilot 报告。B 角色在统一 TinyStories
> validation screening 中的正式记录已更新为
> [PERSON_B_TINYLM_REPORT.md](PERSON_B_TINYLM_REPORT.md)；本文件中的旧配置和
> 数值不得与新报告混合使用。

日期：2026-09-02  
项目：高效 Transformer 统一结构与三人分工  
证据标签：`paper_reported`、`paper_aligned`、`matched_tinylm`、`innovation_validation`

## 1. 执行摘要

本轮完成了 B 角色要求的结构调研、正确性闸门、统一 TinyLM pilot、效率矩阵、关键消融、机制任务和失败边界记录。结论不是脱离条件的总排名：

- Longformer 的局部窗口将 attention score 的主存储从 O(N²) 降为 O(NW)，global token 提供稀疏的远程汇聚路径。统一 384 hidden / 8 heads / BF16 测试中，N=4096 时 median latency 为 10.76 ms、peak allocated 为 78.6 MiB；相对 full attention 的 21.95 ms / 1333.5 MiB，约 2.04x 更快、94.1% 更省显存。
- Memformer 用固定容量 memory slots 跨 segment 传递状态。N=4096、segment=256、slots=64 时 latency 为 13.88 ms、peak allocated 为 37.0 MiB；相对 full attention 约 1.58x 更快、97.2% 更省显存。该结果来自机制 adapter，不能等同完整 MemBART 训练结果。
- 质量 proxy 上，Longformer 与共享 Q/K/V/output 的 causal full attention 在 N=256 的 cosine similarity=0.9068、relative L2=0.4742；Memformer 是不同信息通路，按“effect”报告（cosine=0.0121、relative L2=1.1951），不称为近似 fidelity。
- 合成 copy-stream TinyLM pilot（6 层、384 hidden、8 heads、12 steps、seed=17）PPL：Standard 2.431、Longformer 2.391、Memformer 2.754。该数据集是可复现 fallback，不替代 WikiText-2；仓库中既有 WikiText 结果继续单独保留。

## 2. 口径与配置

统一主干按参数标准冻结：6 layers、hidden=384、heads=8、head_dim=48、FFN=1536、Pre-LN、RoPE 目标位置编码、dropout=0.1、causal、BF16（本机 CUDA）。本轮 attention adapter 的位置编码由调用方承担，因此结构测量只比较 attention/状态通路；参数数字不冒充完整论文模型。

| 方法 | 主配置 | 结构复杂度/状态 | 证据范围 |
|---|---|---|---|
| Longformer | window=128（实现为 odd width=129）、global=1 | O(NW + NG)，局部窗口 + global token | matched_tinylm、innovation_validation |
| Memformer | segment=256、memory_slots=64 | 每段融合 attention；跨段状态 M×d，更新门控 | matched_tinylm、innovation_validation |

原论文设置（`paper_reported`）按标准文件登记：Longformer 使用 text8/enwik8 character LM、small 12×512/8 与 large 30×512/8，分阶段将长度扩展至 23,040；Memformer 官方 MemBART-base/large 分别为 d_model=768/1024、memory length=64/128、encoder/decoder=6/6 或 12/12、dropout=0。它们与本项目 TinyLM 不是同一实验，不能直接横比 PPL。

## 3. 架构与数据流

图：`results/b_role/architecture_dataflow.png`

### Longformer

1. Q/K/V 投影后，每个 token 只与半径窗口内的 K/V 计算 score。
2. global token 被追加到普通 query 的候选集合；global query 另行对所有有效 token 做一次全局 attention，避免局部/全局重复计数。
3. `causal=true` 时，窗口和 global 路径都施加上三角约束；padding query 输出被置零。

### Memformer

1. 输入被切成 segment；memory slots 与当前 segment 拼接参与 fusion attention。
2. memory rows 和 token rows 同时更新；memory rows 经过 memory 输出投影和 sigmoid gate 与旧状态融合。
3. 新 memory 传给下一 segment，可选择 detach；reset 用全零初始状态。状态元素数为 `memory_slots × hidden_size`，与总序列长度无关。

## 4. M1 正确性闸门

`results/b_role/correctness.json`：

| 检查 | 结果 |
|---|---:|
| Longformer 输出 shape=(2,37,32) | 通过 |
| Longformer 输出/梯度有限 | 通过 |
| full-window causal 与 Standard reference | MSE=0，relative L2=0，cosine=1 |
| Memformer 输出 shape=(1,16,32) | 通过 |
| Memformer 输出有限 | 通过 |
| 跨段 state delta vs reset | 0.04175（非零，说明状态确实传播） |

另有仓库测试覆盖 padding mask、边界窗口、global query 读取远距 token、global key 不重复、反向传播和不形成 N×N einsum。由于当前环境没有 pytest 包，本轮用同等 Python smoke assertions 执行了上述关键检查；测试代码仍保留在 `tests/test_longformer.py`。

## 5. M2 TinyLM 质量 pilot

完整原始记录：`results/b_role/tinylm_quality.json`。使用相同 6 层/384/8-head 主干、vocab=64、context=128、seed=17、12 个 AdamW steps，在确定性 copy-stream fallback 上训练和评估。

| 方法 | train loss (last) | eval NLL | eval PPL |
|---|---:|---:|---:|
| Standard | 1.0531 | 0.8883 | 2.4310 |
| Longformer | 1.0362 | 0.8718 | 2.3911 |
| Memformer | 1.1845 | 1.0130 | 2.7539 |

该 pilot 只说明统一接口可以端到端训练；12 steps 和合成数据不足以支持论文级质量结论。环境缺少 `datasets`/`transformers`，因此未把 WikiText 下载失败伪装成正式结果。仓库原有 Longformer WikiText attention fidelity（cosine=0.8092，HF MLM PPL=8.514）和 Memformer-like WikiText TinyLM（PPL=27.805 vs Standard 31.231）保持为独立历史 artifact，不与本轮 matched_tinylm 混表。

## 6. M2 效率矩阵

完整记录：`results/b_role/efficiency.json`；图：`results/b_role/efficiency_curves.png`。设备为本机 CUDA（PyTorch 2.11.0，BF16，batch=1，warmup=2，timed runs=3，seed=17）。

| N | Standard ms / MiB | Longformer ms / MiB | Memformer ms / MiB |
|---:|---:|---:|---:|
| 128 | 0.372 / 23.4 | 1.401 / 35.7 | 0.855 / 25.6 |
| 256 | 0.445 / 29.9 | 1.364 / 51.9 | 0.905 / 31.4 |
| 512 | 0.382 / 46.0 | 2.291 / 54.3 | 1.569 / 31.7 |
| 1024 | 1.103 / 108.4 | 3.273 / 57.8 | 3.547 / 32.5 |
| 2048 | 5.522 / 355.5 | 8.514 / 65.5 | 6.528 / 34.7 |
| 4096 | 21.951 / 1333.5 | 10.758 / 78.6 | 13.877 / 37.0 |

解释：Longformer 使用 query chunking，避免非连续 unfold view 被 PyTorch 隐式物化为大临时张量。短序列时 Python/adapter/kernel 启动开销使 full attention 更快；在本实现和本机上，N=4096 才明显显现 Longformer/Memformer 的长程结构优势。Memformer 的显存近似由固定 slots 和每段工作集决定；Longformer 显存随窗口 W 增长，不是与 N 完全无关。记录同时包含 `peak_reserved_mb`、原始 timed samples、status 和错误字段；本轮没有 OOM。

## 7. M3 关键消融

完整记录：`results/b_role/ablations.json`；图：`results/b_role/ablation_curves.png`。

### Longformer（N=2048，causal，seed=29）

window 从 65→513 时，global=1 的 median latency 从 6.439→17.393 ms，peak allocated 从约 50.0→145.2 MiB；global=0/1/4 的差异小于增大窗口的影响，但 global token 是远距汇聚路径的关键结构开关。window=65、global=0 的 latency 约 6.4 ms，是本 sweep 最快点，但远距 reachability 最弱。

### Memformer（N=4096，causal，seed=43）

| segment | slots=16 | slots=64 | slots=128 |
|---:|---:|---:|---:|
| 128 | 24.205 ms / 30.0 MiB | 24.080 / 30.7 | 26.665 / 33.5 |
| 256 | 11.519 / 33.8 | 10.654 / 36.9 | 11.834 / 40.1 |
| 512 | 5.610 / 50.8 | 8.438 / 55.1 | 5.317 / 61.4 |

segment 越长，段数越少，因而 latency 下降；slots 增加主要抬高状态/投影成本和显存。代价是更长 segment 使 memory 更新频率降低，可能削弱需要频繁刷新状态的任务。

## 8. 机制任务与失败边界

`results/b_role/mechanisms.json` 记录了 1024 长度、6 layers 下的 Longformer reachability 以及 Memformer 状态容量；这是结构可达性验证，不是训练后 accuracy。

- Longformer local-only 在远距 probe 上不可达；window 增大可扩大局部传播范围；global token 使 global query/key 路径可跨距汇聚，但 token 数少时信息带宽有限。
- causal Longformer 只能向过去传播，不能把未来 token 泄漏给当前 query；这正是语言建模所需的失败边界。
- Memformer 的跨段路径始终存在，但容量固定：slots=16/64/128、hidden=384 时分别为 6,144/24,576/49,152 elements，即 BF16 状态约 12/48/96 KiB（每个 batch 样本）。slots 太小时，Passkey/Needle 可能发生覆盖或遗忘；不 reset 会造成样本间状态污染；detach=true 会切断跨段反向梯度。

因此可证伪假设得到明确测试条件：Longformer 若在相同质量约束下不再节省 latency/显存，或 window/global 消融不影响远距任务，则其优势假设被推翻；Memformer 若状态 delta 为零、显存随段数线性膨胀，或 slots 增大不改变跨段保留，则其固定容量 recurrent-memory 假设被推翻。

## 9. 结论与下一步

在本机、BF16、batch=1、N≤4096 的 attention-level 约束下，Longformer 适合局部结构明显且需要少量全局汇聚的长序列；Memformer 适合可接受固定容量摘要、需要跨 segment 持久状态且显存预算严格的任务。短序列应优先使用经过优化的 SDPA/full attention。下一步若要升级为 paper-aligned 质量结论，应安装 `datasets`/`transformers`，使用 WikiText-103/PG-19 或论文对应数据，扩展到 seeds=17/29/43 和完整 50M–100M token 预算，并用完整 MemBART recurrent training，而非 attention adapter。

DOCX 说明：已完成结构审计（Letter 页面、1 英寸边距、标题层级、8 张固定宽度表、3 张嵌图）；当前执行环境没有 LibreOffice/`soffice`，因此无法完成 PNG 渲染型视觉 QA。
