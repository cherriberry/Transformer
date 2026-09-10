# Adaptive Gated Fact Memory + MoE 对比报告（seed 17）

更新时间：2026-09-10（UTC）  
实验协议：`adaptive_fact_memory_selective_stress_v2`；Value-token 条件使用
`adaptive_fact_memory_selective_stress_v3_value_token`
状态：四个 Dense 条件和四个 MoE 条件均完成 2,000/2,000 step，验证完成

## 1. 结论摘要

本报告完成了交接文档要求的四条件 `4 experts / top-2 / dropless` MoE selective-stress
比较。新增的两个正式 run 是：

- `selective_stress_moe_gated_s17_20260910_retry1`
- `selective_stress_moe_value_token_gated_s17_20260910_retry1`

此前已经完成的 SWA-only + MoE 和 Fixed-LRU + MoE 也纳入统一表格。历史失败目录
`selective_stress_moe_gated_s17_20260910` 保留为审计资产，没有被覆盖。

最重要的结果是：

- Value-token Gated + MoE 在 288/288 验证样本上达到 **100% EM、100% target
  survival、Recall@1、Recall@4、MRR**，value-span 和 memory-token exact 也均为
  100%。它与非 MoE Value-token Gated 的 100% 结果一致。
- 旧 Gated + MoE 保持 **100% target survival/Recall/MRR**，但 EM 为 32.99%，低于
  非 MoE 旧 Gated 的 52.43%；这说明 MoE 本身没有修复旧语义值到输出 token 的瓶颈。
- Fixed-LRU + MoE 的 EM 为 65.28%，低于 Dense Fixed-LRU 的 74.65%，但目标存活率
  从 92.01% 提升到 93.06%。
- SWA-only + MoE 的 EM 为 5.21%，与 Dense SWA-only 的偶然词表命中水平相同；它没有
  长期事实槽位，因此 target survival/recall 均为 N/A。
- MoE 将总参数从约 31.85M 增加到约 60.18M（旧 value 路径），但 Top-2 每 token
  只激活约 38.94M 参数；Value-token 路径对应 32.16M → 60.48M，总容量增加而
  单 token 激活容量仍明显低于完整 MoE。
- 四个 MoE 条件均未出现明显 expert collapse。最终 checkpoint 的只读路由审计中，
  四个 expert 的 token fraction 均为非零且大致在 0.43–0.57；训练日志中的
  `moe_load_balance` 约为 2.01–2.06，接近 Top-2 完全均衡时的理论参考值 2。

这些结果是 **seed-17 screening**，不是跨 seed 的统计显著性结论。当前 MoE dispatch
使用普通 PyTorch token indexing、逐 expert forward 和 `index_add_`，没有 fused
grouped-GEMM/MegaBlocks，因此吞吐反映原型实现成本。

## 2. 固定协议与模型

所有正式 run 从同一个 seed-17 TinyStories backbone 开始：

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/
adaptive_gated_fact_memory_tinystories/
tinystories_backbone_s17_10m_20260908/checkpoint.final.pt
```

父 checkpoint SHA-256：

```text
d995248587a5e535af59d850f36a5bc01925c54bd9dff213f0f18bfb26f2b78f
```

| 项目 | 设置 |
|---|---:|
| TinyLM | 6 layers, hidden 384, heads 8, FFN 1536 |
| local KV | 4 sink tokens + 124 recent tokens |
| user memory | 8 slots，reader semantic top-k=4 |
| max write candidates | 1 |
| merge threshold | 0.999 |
| train/eval delay | 训练 1/2/4，验证 1/4/16 segments |
| train/eval noise | 训练 0/1/2/4，验证 0/2/4/8 temporary facts |
| validation | 12 个条件 × 24 = 288 examples |
| optimizer | AdamW，betas=(0.9, 0.95)，weight decay=0.1 |
| learning rates | backbone 3e-5，memory/new heads 3e-4 |
| batch / accumulation | 2 / 2，每 step 4 examples |
| precision | RTX 4090 BF16 autocast |
| steps | 2,000 |

MoE 正式配置：

```text
4 experts / top-k=2 / dropless / moe_load_balance_weight=0.01
```

MoE 只替换每个 decoder block 的 FFN；local attention、事实 memory、reader、value-token
路径不变。每个 token 通过线性 router 选择两个 expert，两个输出按 router 权重归一化
后相加。padding token 不参与 dispatch 或 load 统计。旧 dense FFN 参数保留以兼容
checkpoint，并在 MoE 初始化后冻结；active computation 只经过 selected experts。

## 3. 正式 run 与产物

所有目录位于：

```text
/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_stress/
```

| 条件 | run ID | 状态 |
|---|---|---|
| Dense SWA-only | `selective_stress_v2_swa_only_s17_20260909` | ok |
| Dense Fixed-LRU | `selective_stress_v2_fixed_lru_s17_20260909` | ok |
| Dense old Gated | `selective_stress_v2_gated_s17_20260909` | ok |
| Dense Value-token Gated | `selective_stress_v3_gated_value_token_s17_20260910` | ok |
| MoE SWA-only | `selective_stress_moe_swa_only_s17_20260910` | ok |
| MoE Fixed-LRU | `selective_stress_moe_fixed_lru_s17_20260910` | ok |
| MoE old Gated | `selective_stress_moe_gated_s17_20260910_retry1` | ok |
| MoE Value-token Gated | `selective_stress_moe_value_token_gated_s17_20260910_retry1` | ok |

历史 step-0 失败目录 `selective_stress_moe_gated_s17_20260910` 的
`summary.json` 未修改。当前源码测试为 23/23 passed；Value-token MoE 另完成了
2-step smoke run。

## 4. 四个 Dense 条件

Dense Value-token run 的 `teacher-forced answer NLL` 等 value-token 指标来自其正式
summary。旧 Dense v2 run 的 summary 没有保存 answer NLL；为统一比较，使用当前只读
评估器加载其 final checkpoint 复算 NLL 和状态统计，EM/存活/Recall 与原 summary 一致。
旧 v2 状态张量不含后来加入的 value-token 字段，因此本节的状态 bytes 使用统一当前
schema 的复算值；这不是物理显存下降。

| 条件 | EM | answer NLL | survival | Recall@1 | Recall@4 | MRR | active slots | noise accept | logical active bytes | state bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWA-only | 5.21% | 1.38947 | N/A | N/A | N/A | N/A | 0.00 | N/A | 0 | 2,423,702 |
| Fixed-LRU | 74.65% | 0.50323 | 92.01% | 90.28% | 92.01% | 0.9103 | 4.25 | 75.00% | 20,676 | 2,423,702 |
| old Gated | 52.43% | 0.63089 | 100.00% | 100.00% | 100.00% | 1.0000 | 1.00 | 0.00% | 4,865 | 2,423,702 |
| Value-token Gated | 100.00% | 0.00001946 | 100.00% | 100.00% | 100.00% | 1.0000 | 1.00 | 0.00% | 4,865 | 2,423,702 |

SWA-only 的记忆 survival/recall 是 N/A，而不是记忆机制的“0 分”；它没有长期事实
槽位。SWA-only 的 5.21% EM 是偶然 token 命中水平。

## 5. 四个 MoE 条件

下表的质量指标直接来自各 run 的 `summary.json`。`mean active slots`、logical bytes
和 noise accept rate 是 288 个验证样本的等权平均。路由列为 final checkpoint 的
只读验证审计；早期 SWA-only/Fixed-LRU 的训练日志只保存了 load-balance，没有保存
完整 expert fraction，所以不能把训练日志缺失误写成 0。

| 条件 | EM | answer NLL | survival | Recall@1 | Recall@4 | MRR | active slots | noise accept | logical active bytes | state bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWA-only + MoE | 5.21% | 1.38736 | N/A | N/A | N/A | N/A | 0.00 | N/A | 0 | 2,423,702 |
| Fixed-LRU + MoE | 65.28% | 0.57731 | 93.06% | 93.06% | 93.06% | 0.9306 | 4.25 | 75.00% | 20,659 | 2,423,702 |
| old Gated + MoE | 32.99% | 0.85217 | 100.00% | 100.00% | 100.00% | 1.0000 | 1.00 | 0.00% | 4,865 | 2,423,702 |
| Value-token Gated + MoE | 100.00% | 0.00002212 | 100.00% | 100.00% | 100.00% | 1.0000 | 1.00 | 0.00% | 4,865 | 2,423,702 |

旧 Gated + MoE 的 teacher-forced NLL 明显优于 SWA-only，但自由生成 EM 仍只有
32.99%；它确实能读到目标事实，却没有稳定把语义值转成正确答案 token。Value-token
辅助路径消除了这一瓶颈，但它仍是 auxiliary memory-to-token head，不是 pointer/copy
head；正式生成仍使用普通 LM head。

### 5.1 MoE 路由统计

以下 fraction 为 token fraction（某 token 是否被该 expert 选中）和 dispatch fraction
（按 Top-2 权重计的加权 dispatch）。Top-2 下 token fraction 之和约为 2，单 expert
的均匀参考值约为 0.5；dispatch fraction 的均匀参考值约为 0.25。

| 条件 | token fractions (E0/E1/E2/E3) | dispatch fractions (E0/E1/E2/E3) | routing entropy | load-balance |
|---|---|---|---:|---:|
| SWA-only + MoE | 0.536 / 0.434 / 0.574 / 0.456 | 0.251 / 0.242 / 0.273 / 0.233 | 1.230 | 2.240 |
| Fixed-LRU + MoE | 0.532 / 0.527 / 0.533 / 0.408 | 0.297 / 0.238 / 0.262 / 0.202 | 1.282 | 2.275 |
| old Gated + MoE | 0.545 / 0.457 / 0.569 / 0.429 | 0.275 / 0.232 / 0.270 / 0.223 | 1.312 | 2.210 |
| Value-token Gated + MoE | 0.471 / 0.508 / 0.511 / 0.511 | 0.239 / 0.259 / 0.244 / 0.258 | 1.355 | 2.111 |

这些是独立的 final-checkpoint validation routing audit，和训练日志末步的统计用途
不同。训练日志末步 `moe_load_balance` 分别约为 SWA 2.013、Fixed-LRU 2.031、旧
Gated 2.022、Value-token Gated 2.008；两组统计均没有显示单 expert collapse。由于
当前实现是 dropless，不存在因 capacity 截断而丢 token；小批次中某 expert 偶尔没有
token 也不等于 token 丢失。

## 6. Dense → MoE 配对变化

### 6.1 质量与记忆行为

| 条件 | EM Dense → MoE | answer NLL Dense → MoE | survival 变化 | Recall@1 变化 | Recall@4 变化 | MRR 变化 |
|---|---:|---:|---:|---:|---:|---:|
| SWA-only | 5.21% → 5.21% (0.00 pp) | 1.38947 → 1.38736 (-0.00211) | N/A | N/A | N/A | N/A |
| Fixed-LRU | 74.65% → 65.28% (-9.38 pp) | 0.50323 → 0.57731 (+0.07408) | +1.04 pp | +2.78 pp | +1.04 pp | +0.0203 |
| old Gated | 52.43% → 32.99% (-19.44 pp) | 0.63089 → 0.85217 (+0.22128) | 0.00 pp | 0.00 pp | 0.00 pp | 0.0000 |
| Value-token Gated | 100.00% → 100.00% (0.00 pp) | 0.00001946 → 0.00002212 (+0.00000266) | 0.00 pp | 0.00 pp | 0.00 pp | 0.0000 |

解释上不能把 Fixed-LRU 的 survival 小幅提升理解为 MoE 改善了记忆算法：两者仍受
单 seed、训练随机性和模型容量变化影响。更稳妥的结论是，MoE 没有破坏 value-token
模型已经获得的选择性记忆行为；对旧 Gated，MoE 也没有修复“检索成功但生成失败”。

### 6.2 资源变化

| 条件 | Dense 总参数 | MoE 总参数 | Dense active/token | MoE active/token | 训练时间 Dense → MoE | examples/s Dense → MoE | peak allocated Dense → MoE | peak reserved Dense → MoE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SWA-only | 31,854,750 | 60,175,542 | 31,854,750 | 38,941,878 | 880.1 → 1479.8 s (+68.2%) | 9.09 → 5.41 (-40.5%) | 2.871 → 3.433 GB (+19.5%) | 6.881 → 7.176 GB (+4.3%) |
| Fixed-LRU | 31,854,750 | 60,175,542 | 31,854,750 | 38,941,878 | 1200.6 → 1804.0 s (+50.3%) | 6.66 → 4.43 (-33.4%) | 2.666 → 3.190 GB (+19.7%) | 6.568 → 6.960 GB (+6.0%) |
| old Gated | 31,854,750 | 60,175,542 | 31,854,750 | 38,941,878 | 1323.0 → 1905.0 s (+44.0%) | 6.05 → 4.20 (-30.6%) | 2.673 → 3.201 GB (+19.8%) | 6.648 → 7.021 GB (+5.6%) |
| Value-token Gated | 32,160,043 | 60,480,835 | 32,160,043 | 39,247,171 | 1380.1 → 1927.1 s (+39.6%) | 5.80 → 4.15 (-28.4%) | 2.682 → 3.205 GB (+19.5%) | 6.702 → 6.984 GB (+4.2%) |

GB 表示十进制近似 `bytes / 1e9`；原始 run 保存了 bytes。MoE 总参数约增加 89%，
但 active parameters/token 只激活 Top-2 expert 路径，且仍包括共享 backbone、router
和一个有效的 expert 子集。总参数增加并不意味着 checkpoint 或 optimizer state
存储下降；所有 experts 仍需保存。当前 memory slot 仍是固定预分配张量，因此
logical active-slot bytes 也不等于物理显存已经下降。

## 7. 按噪声与延迟的 MoE 结果

每个单元 24 examples。MoE Value-token Gated 在全部 12 个条件均为 100% EM；旧
Gated + MoE 在全部条件 survival/Recall 仍为 100%，但 EM 介于 20.8%–45.8%。
Fixed-LRU + MoE 在 8 条噪声（超过有效容量）时 target survival 降至 58.3%–83.3%，
而旧 Gated 和 Value-token Gated 保持 100%。

| MoE 条件 | noise=0，delay=1/4/16 | noise=2，delay=1/4/16 | noise=4，delay=1/4/16 | noise=8，delay=1/4/16 |
|---|---|---|---|---|
| SWA-only | 0.0/4.2/4.2% | 8.3/4.2/8.3% | 4.2/4.2/4.2% | 12.5/0.0/8.3% |
| Fixed-LRU | 58.3/70.8/66.7% | 66.7/75.0/75.0% | 54.2/79.2/66.7% | 66.7/45.8/58.3% |
| old Gated | 29.2/20.8/37.5% | 33.3/41.7/33.3% | 33.3/29.2/45.8% | 29.2/33.3/29.2% |
| Value-token Gated | 100/100/100% | 100/100/100% | 100/100/100% | 100/100/100% |

SWA-only 的百分比只是词表偶然命中，不应当解释为长期记忆能力。

## 8. 正确性、失败处理与可复核性

本次正式实验前执行了：

```bash
PYTHONPATH=adaptive_gated_fact_memory/src \
python3 -m unittest discover \
  -s adaptive_gated_fact_memory/tests -p 'test_*.py' -v
```

结果：**23/23 tests passed**。Value-token Gated + MoE 的 2-step smoke run 也成功，
随后才启动正式 2,000-step run。两个新增正式 run 均包含：

- `config.resolved.json`
- `metrics.jsonl`
- `checkpoint.step_{500,1000,1500,2000}.pt`
- `checkpoint.final.pt`
- `summary.json`

历史失败 run 保留，失败原因是旧版本的 MoE diagnostics 聚合形状错误；当前代码已
通过 route fraction 对 layer 维度求均值，并对 reader query/key dtype 做一致化。没有
删除失败 summary，也没有覆盖旧 checkpoint。

## 9. 限制与结论边界

1. 只有 seed=17，不能报告均值、标准差、显著性或跨初始化稳定性。
2. 任务是 16 个封闭颜色值和受控模板的选择性事实压力测试，不能外推为开放域长期
   记忆能力。
3. Value-token auxiliary head 不是 pointer/copy head，正式生成仍由普通 tied LM head
   完成。
4. MoE 增加总参数容量，不等于降低 checkpoint/optimizer 存储；active parameters/token
   也不等于物理显存占用。
5. 当前 dropless dispatch 是普通 PyTorch 原型实现，吞吐不能代表 fused MoE kernel
   的理论上限。
6. Dense v2 与后续 value-token/MoE 代码的 state schema 不完全相同；报告中的 Dense
   旧模型 bytes 已用当前 schema 只读复算以便配对，但历史 summary 仍保持原样。
7. 训练和评估使用 final checkpoint，没有跨 checkpoint 选择独立 validation 最优点。

综合而言，本次实验支持以下 screening 结论：**Top-2 MoE 可以与有界事实记忆和
value-token 对齐路径稳定组合；它没有破坏门控的目标事实保留能力，而 Value-token
Gated + MoE 在当前受控任务上保持 100% 端到端 EM。旧 Gated + MoE 的失败说明，MoE
本身不是词级答案恢复的替代方案。** 要形成更强结论，下一步应补 seeds 29/43、
更弱的重要性提示、容量扫描，以及 fused dispatch 下的效率复测。
