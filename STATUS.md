# Completion status (2026-08-17)

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
