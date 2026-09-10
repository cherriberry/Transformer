# 文档索引

本目录收纳原先散落在仓库根目录的项目级 Markdown。分类表示推荐阅读顺序，不表示删除或覆盖历史材料。

## 当前结果与推荐入口

| 文档 | 内容 | 状态 |
|---|---|---|
| [TINYLM_MOE_UNIFIED_EXPERIMENT_REPORT.md](TINYLM_MOE_UNIFIED_EXPERIMENT_REPORT.md) | 六种变体的全六层 4-expert Top-1 dropless MoE 结果，以及 Dense 对照 | 最新 MoE 主报告 |
| [TINYLM_MOE_ONLY_EXPERIMENT_REPORT.md](TINYLM_MOE_ONLY_EXPERIMENT_REPORT.md) | 仅包含本次六模型 MoE 训练、路由、correctness 和 Keyformer+MoE 评估 | MoE 专项报告 |
| [TINYLM_SIX_FORMER_OVERALL_REPORT.md](TINYLM_SIX_FORMER_OVERALL_REPORT.md) | Dense、100M 扩展、MoE 与 Keyformer 的六架构总体分析 | 最新总体报告 |
| [TINYLM_SIX_FORMER_PAPER_STYLE_REPORT.md](TINYLM_SIX_FORMER_PAPER_STYLE_REPORT.md) | 按论文结构整理的六架构与 MoE 总体研究报告 | 论文式版本 |
| [C_UNIFIED_RESULT_REPORT_V2.md](C_UNIFIED_RESULT_REPORT_V2.md) | Full Attention、Causal LSH Reformer、Keyformer 的 protocol v2 结果 | C 当前结果 |
| [PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md](PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md) | A+B 四种 Dense backbone、三 seed 统一结果 | Dense 主报告 |
| [PERSON_B_TINYLM_SIMPLE_RESULT_REPORT_UNIFIED_20260906.md](PERSON_B_TINYLM_SIMPLE_RESULT_REPORT_UNIFIED_20260906.md) | 上述 A+B 结果的简版 | 快速阅读 |
| [UNIFIED_AB_RERUN_REPORT.md](UNIFIED_AB_RERUN_REPORT.md) | A+B 统一重训的实现和审计说明 | 配套报告 |
| [ADAPTIVE_GATED_MEMORY_STRESS_S17_RESULT_REPORT.md](ADAPTIVE_GATED_MEMORY_STRESS_S17_RESULT_REPORT.md) | 门控事实记忆容量压力实验 | 当前专项结果 |

建议先读 MoE 主报告；需要核对 Dense 基线时，再读 C v2 和 A+B 三 seed 报告。

## 跨模型汇总与设计分析

| 文档 | 内容 |
|---|---|
| [ARCHITECTURE_EFFICIENCY_COMPARISON_REPORT.md](ARCHITECTURE_EFFICIENCY_COMPARISON_REPORT.md) | 高效 Transformer 架构、效率和设计来源 |
| [LINKED_TINYSTORIES_SIX_MODEL_COMPARISON_REPORT.md](LINKED_TINYSTORIES_SIX_MODEL_COMPARISON_REPORT.md) | 两组 TinyStories 实验包的六模型对比 |
| [SIX_MODEL_UNIFIED_COMPARISON_RECORD.md](SIX_MODEL_UNIFIED_COMPARISON_RECORD.md) | A/B/C 六方案统一比较记录 |
| [PROJECT_SUMMARY_REPORT.md](PROJECT_SUMMARY_REPORT.md) | 项目级历史总结 |
| [B_LONGFORMER_MEMFORMER_EXPERIMENT_REPORT.md](B_LONGFORMER_MEMFORMER_EXPERIMENT_REPORT.md) | Longformer/Memformer 结构化实验总结 |
| [B_PAPER_COMPARISON_REPORT.md](B_PAPER_COMPARISON_REPORT.md) | B 角色结果与原论文条件对齐 |
| [KEYFORMER_REPORT.md](KEYFORMER_REPORT.md) | GPT-2 Medium 阶段 Keyformer 深度报告 |

这些报告来自不同阶段和协议。跨报告引用数值时，应优先遵循各报告中的“比较口径”和“局限性”。

## 协议、计划与研究资料

| 文档 | 内容 |
|---|---|
| [UNIFIED_EXPERIMENT_PROTOCOL.md](UNIFIED_EXPERIMENT_PROTOCOL.md) | 三人协作统一实验协议 |
| [TINYLM_EXPERIMENT_QA.md](TINYLM_EXPERIMENT_QA.md) | TinyLM 实验设置、决策和问答记录 |
| [PAPER_CONFIGS_AND_RECOMMENDATIONS.md](PAPER_CONFIGS_AND_RECOMMENDATIONS.md) | 原论文配置与本项目参数建议 |
| [SOURCES.md](SOURCES.md) | 论文和源码来源映射 |
| [PLAN_Future.md](PLAN_Future.md) | 后续研究计划 |
| [FOUR_DAY_LONG_CONTEXT_EXPERIMENT_PLAN.md](FOUR_DAY_LONG_CONTEXT_EXPERIMENT_PLAN.md) | 历史四天长文本执行计划 |

`PLANS/` 和各 `experiments/*/` 目录中还保留更贴近具体任务的计划与复现说明。

## B 角色历史报告

| 文档 | 内容 |
|---|---|
| [PERSON_B_TINYLM_REPORT.md](PERSON_B_TINYLM_REPORT.md) | 原 B 角色完整报告 |
| [PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md](PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md) | 原 B 角色结果速览 |
| [PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md](PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md) | 原 B 角色详细实验文档 |

这些文件为可追溯性保留；当前 10M Dense 横向结论优先参考带 `UNIFIED_20260906` 的版本。

## 交接与历史状态

| 文档 | 内容 |
|---|---|
| [HANDOFF_ROLES_A_C.md](HANDOFF_ROLES_A_C.md) | A/C 角色执行交接 |
| [HANDOFF.md](HANDOFF.md) | 历史微基准交接 |
| [KEYFORMER_HANDOFF.md](KEYFORMER_HANDOFF.md) | GPT-2 Medium Keyformer 阶段交接 |
| [STATUS.md](STATUS.md) | 2026-08-17 历史完成状态 |

## 文件维护约定

- 仓库根目录只保留项目入口 `README.md`。
- 项目级结果、协议、计划和交接文档放在 `docs/`。
- 与某个实现强绑定的 README/报告放在对应模块或实验目录。
- 新报告应注明日期、protocol、seed、训练预算、验证范围和原始结果路径。
- 不覆盖历史报告；新协议使用新文件，并在本索引中标注推荐版本。
