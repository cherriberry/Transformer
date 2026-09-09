# Efficient Transformer / TinyLM experiments

本仓库比较 Longformer、Memformer、Linformer、Performer、Reformer、
Full Attention 与 Keyformer，并包含 Dense FFN 和全六层 MoE 实验。

## 文档入口

- [文档总索引](docs/README.md)：按当前结果、基线报告、设计资料和历史材料分类。
- [最新 MoE 统一实验报告](docs/TINYLM_MOE_UNIFIED_EXPERIMENT_REPORT.md)：六层均为 4-expert、Top-1、dropless MoE，seed=17。
- [C v2 统一结果](docs/C_UNIFIED_RESULT_REPORT_V2.md)：Full Attention、Causal LSH Reformer 与 Dense Keyformer 评估。
- [A+B Dense 三 seed 报告](docs/PERSON_B_TINYLM_REPORT_UNIFIED_20260906.md)：Linformer、Performer、Longformer、Memformer。
- [实验协议](docs/UNIFIED_EXPERIMENT_PROTOCOL.md)与[实验问答](docs/TINYLM_EXPERIMENT_QA.md)。

根目录只保留本入口文件；其他项目级 Markdown 已统一移入 `docs/`。各模型目录中的 README、源码说明和实验目录内文档仍保留在原位置。

## 主要目录

| 目录 | 内容 |
|---|---|
| `experiments/` | TinyStories/TinyLM 训练、评估与 MoE runner |
| `adaptive_gated_fact_memory/` | 自适应门控事实记忆实验 |
| `keyformer/` | Keyformer KV-cache compression |
| `longformer/` | Longformer local/sliding-window attention |
| `memformer/` | Memformer-style recurrent memory |
| `linformer/` | Linformer low-rank attention |
| `performer/` | Performer FAVOR+ attention |
| `reformer/` | Reformer LSH attention |
| `common/`、`models/` | 公共训练和模型组件 |
| `results/` | 仓库内保存的历史结果与图表 |

大体积 TinyLM checkpoints 和聚合结果保存在数据盘：
`/root/autodl-tmp/26summerBDMI_transformer/`。

## 当前 TinyLM 协议

当前数据协议位于 `experiments/tinystories_tinylm_v1/protocol.yaml`：从头训练的 causal TinyLM，使用固定 revision 的 TinyStories 和 GPT-2 tokenizer。最新 MoE 配置及 runner 位于：

- `experiments/tinystories_tinylm_moe_v1/moe_seed17.yaml`
- `experiments/tinystories_tinylm_moe_v1/run_moe_six.py`

`experiments/long_context_10m_v1/` 中的 PG-19 文件仅作为历史来源保留，不应重新标记为 TinyStories 结果。

Public repository: <https://github.com/lzr20082024/26summerBDMI_transformer>
