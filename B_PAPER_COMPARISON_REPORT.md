# B 角色：与 Longformer / Memformer 原论文条件对比报告

> 本文件是原论文条件对齐的历史报告，不是本次 TinyStories validation
> screening 的结果汇总。统一 TinyLM/TinyStories 的 B 角色正式结果见
> [PERSON_B_TINYLM_REPORT.md](PERSON_B_TINYLM_REPORT.md)。

日期：2026-09-03  
标准：`PAPER_COMPARISON_STANDARD.md`（`paper_comparison_v1`）

## 结论先行

基础参数实验已经完成；原论文对比实验也可以运行，但当前属于 **等级 B：method_aligned**，不是等级 A 的严格论文复现。原因是当前环境没有 `datasets`/`transformers`，没有原论文训练语料、完整训练预算和 checkpoint。所有结果都已单独标注，未将本地数字与论文数字直接相减或声称超过论文。

本轮新增三条证据线：

1. Longformer 论文配置的 d=512、8 heads、causal local/global 长度 sanity check；
2. Memformer MemBART-base 配置（d_model=768、heads=12、memory length=64）的官方 attention 对齐检查；
3. C1/C2/C3 causal 配对：同一 TinyLM、同一数据/初始化/优化器、3 seeds，仅改变 bidirectional/causal mask 或加入 Longformer/Memformer 结构。

## 1. 论文公开条件（`paper_reported`）

| 方法 | 论文任务/数据 | 论文模型与关键参数 | 本项目可直接复现性 |
|---|---|---|---|
| Longformer | text8、enwik8 character-level autoregressive LM；100M chars train/dev/test | small 12 layers、hidden=512、8 heads；large 30 layers、hidden=512、8 heads；Transformer-XL relative + sinusoidal；AdamW wd=0.01、grad clip=0.25；5 phases，长度从 2,048 到 23,040 | 原始数据与完整 5-phase 训练未安装，降级 method_aligned |
| Memformer / MemBART | Memformer sequence modeling；官方实现基于 BART | base：d_model=768、encoder/decoder=6/6、heads=12、FFN=3072、max position=1024、memory length=64、GELU、dropout=0；large：1024/12/12/4096/128 | 完整 BART checkpoint/训练未运行；官方 attention 单层对齐，降级 method_aligned |

来源：Longformer <https://arxiv.org/abs/2004.05150>；Memformer <https://arxiv.org/abs/2010.06891>；官方 Memformer 实现 <https://github.com/qywu/memformers>。论文配置整理自 `PAPER_CONFIGS_AND_RECOMMENDATIONS.md`，未知项未自行补齐。

## 2. Longformer 论文配置对齐检查

原始记录：`results/paper_comparison_b/longformer_paper_aligned.json`；图：`results/paper_comparison_b/paper_alignment_latency.png`。

实现使用 d=512、8 heads、causal mask、window width=513（近似论文窗口 512），比较 full reference、local-only、local+global(1)。这是 attention-level 长度机制检查，不是 text8/enwik8 PPL 复现。

| N | full reference ms / MiB | local-only ms / MiB | local+global ms / MiB |
|---:|---:|---:|---:|
| 2048 | 6.329 / 347.1 | 25.656 / 169.8 | 26.780 / 177.5 |
| 4096 | 22.044 / 1327.1 | 48.748 / 185.9 | 51.386 / 202.3 |
| 8192 | 87.007 / 5231.1 | 97.201 / 218.0 | 103.398 / 249.5 |

解释：分块滑窗实现把峰值显存降到约 186–249 MiB；在 N=8192 时 latency 97.201 ms，接近 full reference 的 87.007 ms，但 N≤4096 时 Python/chunk 开销更明显。该结果说明结构的 O(NW) 存储趋势，不支持“本实现绝对更快”的论文级结论。论文中的 fused sliding-window kernel、5-phase 长度课程和完整模型尚未复现。

## 3. Memformer MemBART-base 对齐检查

原始记录：`results/paper_comparison_b/memformer_paper_aligned.json`；同一图：`paper_alignment_latency.png`。

使用官方 `MemBartEncoderAttention`，d_model=768、12 heads、memory length=64、bidirectional encoder 语义；比较 full reference 与官方 attention 单层 adapter。

| N | full reference ms / MiB | Memformer attention ms / MiB |
|---:|---:|---:|
| 128 | 0.360 / 25.9 | 0.541 / 26.4 |
| 256 | 0.310 / 31.6 | 0.564 / 29.5 |
| 512 | 0.354 / 57.1 | 0.617 / 43.0 |

官方 memory state 检查：shape=`[1,64,768]`，state elements=49,152，BF16 state bytes=98,304，连续两个 segment 的 state delta=0.01312（非零）。这验证了固定容量状态和跨段更新路径；它不等价于完整 MemBART 的 encoder/decoder 训练质量。

## 4. C1/C2/C3 causal 配对（3 seeds）

原始记录：`results/paper_comparison_b/causal_pair.json`；图：`results/paper_comparison_b/causal_pair_ppl.png`。

设置：4 layers、hidden=256、8 heads、context=128、vocab=64、AdamW 3e-4、8 steps、synthetic copy-stream fallback；seeds=17/29/43。C1/C2 使用相同 Standard TinyLM 初始化，C1 为 bidirectional，C2 为 causal；C3 在 causal 语义下加入 Longformer 或 Memformer。

| 变体 | mask/结构 | mean PPL | SD | 95% CI | n |
|---|---|---:|---:|---:|---:|
| C1_bidirectional | bidirectional baseline | 25.071 | 1.225 | [22.028, 28.113] | 3 |
| C2_causal | causal baseline | 23.498 | 1.204 | [20.506, 26.490] | 3 |
| C3_longformer_causal | causal + local/global | 22.878 | 1.134 | [20.062, 25.694] | 3 |
| C3_memformer_causal | causal + recurrent memory | 23.382 | 1.023 | [20.841, 25.922] | 3 |

预先定义的差值：

- causal 影响 `C2-C1` = **-1.573 PPL**（合成 causal copy-stream 上 causal 反而更好）；
- Longformer 结构贡献 `C3-C2` = **-0.620 PPL**；
- Memformer 结构贡献 `C3-C2` = **-0.116 PPL**。

这些差值只说明在该合成任务、短训练预算和小模型下未观察到 causal 质量退化；不能外推为 WikiText/原论文任务上的普遍结论。由于任务是 causal stream，C1/C2 的 mask 公平性满足接口要求，但不替代 bidirectional MLM 任务对比。

## 5. 与等级标准的对应关系

| 证据线 | 等级 | 可以支持的表述 | 不可以支持的表述 |
|---|---|---|---|
| 论文公开表格与设置 | `paper_reported` | “原论文在其条件下报告……” | “我们比论文提升……” |
| Longformer d=512 / Memformer d=768 attention 对齐 | `method_aligned` | “保留核心机制后观察到 O(NW) 存储/固定 memory state 趋势” | “完全复现原论文” |
| C1/C2/C3 三种配对 | `method_aligned` + `innovation_validation` | “在本 synthetic task 和 3 seeds 下，causal 未造成明显退化” | “causal 与 bidirectional 等价” |

## 6. 如何升级到等级 A

1. 安装并锁定 `datasets`、`transformers`，获取 text8/enwik8、WikiText-103 或 Memformer 原论文对应数据，保存数据 hash 与 split。
2. Longformer 按论文 small/large 主干、relative position、5-phase 长度课程、batch 和训练 token 数运行；记录 checkpoint 与 PPL。
3. Memformer 使用完整 MemBART-base/large encoder-decoder 与 recurrent memory training，而不是单层 attention adapter。
4. 每个条件至少 seeds=17/29/43，报告均值、标准差和 95% CI；将 paper_aligned、matched_tinylm、causal pair 分开成三张表。
5. 只有在任务、数据、模型、预算、mask 和指标都对齐时，才可升级为等级 A 并讨论论文数值差异。

运行命令：

```powershell
python paper_comparison_b.py --out-dir results/paper_comparison_b --runs 3 --steps 8 --longformer-lengths 2048 4096 8192
```
