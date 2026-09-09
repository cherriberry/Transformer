# Completion status (historical inventory, 2026-08-17)

> 本表记录的是旧版 WikiText/微基准资产，不是当前 TinyStories/TinyLM
> validation screening 的完成状态。角色 A/C 的执行入口、待办和验收标准见
> [HANDOFF_ROLES_A_C.md](HANDOFF_ROLES_A_C.md)；角色 B 的当前结果见本目录的
> `PERSON_B_TINYLM_*` 报告和 `../experiments/tinystories_tinylm_v1/aggregate/`。

| Method | Perf baseline | WikiText quality | Location |
|---|---|---|---|
| Keyformer | yes (GPT-2 Medium sweep) | yes (PPL/NLL) | `keyformer/` |
| Longformer | yes (teammate figures) | yes (local-window fidelity + HF MLM) | `longformer/` |
| Linformer | yes (teammate figures) | yes (fidelity + tiny LM PPL) | `linformer/` |
| xFormer | yes (teammate figures) | yes (SDPA/efficient fidelity + tiny LM PPL) | `xformer/` |
| Memformer | yes (shared microbench) | yes (memory-slot effect + tiny LM PPL) | `memformer/` |
| Performer | yes (shared microbench) | yes (FAVOR+ fidelity + tiny LM PPL) | `performer/` |
| Reformer | yes (shared microbench) | yes (LSH vs standard + tiny LM PPL) | `reformer/` |

Teammate sources incorporated:
- https://github.com/cherriberry/Transformer (Mem/Per/Re microbenchmarks)
- https://cloud.tsinghua.edu.cn/d/6e749f61c39f4772b400/ (Long/Lin/X benchmarks)
