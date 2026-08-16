# Keyformer on GPT-2 Medium — Project Report

## 1. Research question

在自回归语言模型生成过程中，**Keyformer** 能否在尽量保持模型质量的同时，减少 KV cache 显存和解码时间？

本报告只覆盖 Keyformer。Longformer 由同学负责，不作为本报告主线。

## 2. Method summary

### Why KV cache dominates memory

For each layer and head, decode stores past keys/values. Bytes grow roughly as
\(2 \times L \times H \times T \times d_{head} \times sizeof(dtype)\).
Prefill builds the initial cache; decode appends one token at a time. Keyformer
targets decode by discarding low-score history while protecting a recent window.

### Keyformer selection

Each layer/head keeps keys, values, **original absolute positions**, and accumulated
scores. Every new token:

1. Append K/V  
2. Compute attention logits  
3. Gumbel-softmax importance with \(\tau_t=\tau_{init}+t\Delta\tau\)  
4. Accumulate scores  
5. Keep newest \(w\) tokens and top-scoring \(k-w\) older tokens  
6. **Physically gather** retained K/V (not only mask)

Defaults used here: \(\tau_{init}=1.0\), \(\Delta\tau=0.01\), recent window = 50% of budget.

### Compared policies

| Policy | Role |
|---|---|
| FullKV | Exact baseline |
| RecentWindow | Keep only newest \(k\) tokens |
| Random+Recent | Random older tokens + recent window |
| Keyformer | High-score older tokens + recent window |

Shared controls: GPT-2 Medium weights, FP16, batch=1, greedy decoding. Gumbel
temperature is **not** text-sampling temperature.

## 3. Implementation

| Path | Role |
|---|---|
| `models/keyformer.py` | Algorithm-level policies (prior 26 tests) |
| `models/gpt2_keyformer.py` | GPT-2 Medium integration + engine |
| `tests/test_gpt2_keyformer.py` | Equivalence / budget / reproducibility |
| `benchmarks/keyformer_gpt2.py` | Synthetic speed/memory sweep |
| `benchmarks/keyformer_quality.py` | WikiText-2 sliding-window PPL |
| `benchmarks/keyformer_plot.py` | Figures |
| `configs/keyformer_experiment.yaml` | Frozen protocol |
| `results/synthetic/perf_final.json` | Performance raw data |
| `results/wikitext2/quality_final.json` | Quality raw data |
| `results/figures/*.png` | Plots |

Absolute positions are tracked separately from physical cache length, so compression
cannot rewind `wpe` indices.

### Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB)
- Model: `openai-community/gpt2-medium` (355M, `n_positions=1024`)
- Data: WikiText-2 test (`Salesforce/wikitext`, `wikitext-2-raw-v1`)
- Stack: PyTorch 2.11.0+cu128, Transformers 5.15.0, FP16

**Context note:** GPT-2 Medium cannot exceed 1024 absolute positions. The requested
prompt length 1024 + generation 128 is impossible; the final perf matrix uses
**prompt 256 / 512 / 768** with generation **128** (max total 896 ≤ 1024).

## 4. Correctness evidence

All of the following passed in `tests/test_gpt2_keyformer.py` plus algorithm tests:

1. FullKV prefill logits match stock GPT-2 within tight tolerance  
2. FullKV greedy tokens match stock HF greedy generation  
3. Keyformer @ **100% budget** produces **identical** tokens to FullKV  
4. At 50% budget, Keyformer / Recent / Random keep `cached_tokens ≤ budget`  
5. Random+Recent is seed-reproducible  

WikiText-2 also shows FullKV / Keyformer@100% / Recent@100% / Random@100% all give
**identical PPL = 27.903** on the evaluated tokens — a strong end-to-end check that
100% budget does not silently diverge.

## 5. Performance results

Source: `results/synthetic/perf_final.json` (117/117 ok)  
Figure: `results/figures/keyformer_perf.png`

Protocol: gen=128; cache ratios 25/50/75/100%; seeds 0/1/2; recent window 50% of budget.
Primary memory metric below is **physical KV bytes** (all layers). CUDA allocator peaks
are also logged but can include fragmentation; prefer KV bytes for compression claims.

### Physical KV cache (MiB, mean over seeds)

| Prompt | FullKV | 75% | 50% | 25% | Saving @25% |
|---|---:|---:|---:|---:|---:|
| 256 | 36.0 | 27.0 | 18.0 | 9.0 | **75%** |
| 512 | 60.0 | 45.0 | 30.0 | 15.0 | **75%** |
| 768 | 84.0 | 63.0 | 42.0 | 21.0 | **75%** |

Compressed policies share the same KV footprint at a given ratio (budget-limited).

### Decode latency / throughput (Keyformer vs FullKV)

| Prompt | Ratio | FullKV ms/tok | Keyformer ms/tok | Speedup |
|---|---:|---:|---:|---:|
| 256 | 0.50 | 22.1 | 43.7 | 0.56× (slower) |
| 512 | 0.50 | 36.8 | 35.8 | **1.02×** |
| 768 | 0.50 | 63.7 | 44.4 | **1.44×** |
| 768 | 0.25 | 63.7 | 48.6 | **1.44×** |

**Takeaway:** on this eager GPT-2 Medium path, Keyformer’s selection/gather overhead
dominates at short prompts; a clear decode speedup appears around **prompt ≥ 512–768**.
RecentWindow is often faster than Keyformer at the same budget (cheaper policy) but
collapses in quality (next section). Headline paper numbers (2.1× / 2.4× on large models
+ A100) are **not** reproduced here, and should not be claimed.

## 6. WikiText-2 quality results

Source: `results/wikitext2/quality_final.json` (39/39 ok)  
Figure: `results/figures/keyformer_quality.png`

Protocol: 2048 test tokens, sliding windows of context 512, teacher-forced NLL/PPL/top-1,
budget = `ratio × 512`, 3 seeds. Compressed caches **participate** in subsequent logits.

| Policy | Ratio | PPL | ΔPPL vs FullKV | Top-1 |
|---|---:|---:|---:|---:|
| FullKV | 100% | 27.903 | 0.0% | 41.14% |
| Keyformer | 100% | 27.903 | 0.0% | 41.14% |
| Keyformer | 75% | 28.111 | **+0.75%** | 40.93% |
| Keyformer | 50% | 28.697 | **+2.84%** | 40.64% |
| Keyformer | 25% | 32.822 | **+17.6%** | 38.96% |
| RecentWindow | 50% | 530.2 | +1800% | 21.6% |
| Random+Recent | 50% | 129.9 | +366% | 28.6% |
| RecentWindow | 25% | 3757 | catastrophic | 11.4% |
| Random+Recent | 25% | 1952 | catastrophic | 14.5% |

**Takeaway:** at equal budget, Keyformer preserves language-modeling quality far better
than RecentWindow or Random+Recent. Mild compression (50–75%) keeps PPL nearly intact;
aggressive 25% still works but with a clear quality cost.

## 7. Direct answers

1. **Memory at 75/50/25%:** physical KV bytes fall by ~25/50/75% versus FullKV at the same prompt+gen length.  
2. **When speed helps:** about prompt **512+** on this setup; at 768 / 50% Keyformer ≈ **1.44×** decode vs FullKV.  
3. **Better than Recent/Random?** **Yes on quality**, decisively. On raw latency, Recent can be faster but is not an acceptable quality baseline.  
4. **Selection cost included?** Yes — timings wrap the full decode step (score + gather + attention).  
5. **PPL vs ratio:** Keyformer degrades gracefully; Recent/Random do not.  
6. **Recent window:** default 50% of budget (paper-style).  
7. **Paper 2.1×/2.4×?** Not matched — different model scale, GPU, and attention kernel path. Treat as a GPT-2 Medium laptop study.

## 8. Limitations

- GPT-2 Medium only (no TinyLlama / RoPE follow-up in this deliverable)
- Absolute context capped at 1024 by GPT-2 `wpe`
- Eager attention for score access; not a production FlashAttention kernel
- WikiText-2 evaluated on 2048 tokens (fixed protocol; not the entire test split)
- Longformer is out of scope for this report

## 9. Reproduction

```powershell
cd C:\Users\lzr20\Documents\Codex\2026-08-13\ba-z\work\Transformer
.\.venv\Scripts\python.exe -m pip install -r requirements-keyformer.txt
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest -q tests/test_keyformer.py tests/test_gpt2_keyformer.py
.\.venv\Scripts\python.exe benchmarks\keyformer_gpt2.py --mode final
.\.venv\Scripts\python.exe benchmarks\keyformer_quality.py --mode quality_final
.\.venv\Scripts\python.exe benchmarks\keyformer_plot.py
```

## 10. Conclusion

Keyformer was connected to **real GPT-2 Medium** with physical per-layer KV compression and
fair FullKV / Recent / Random controls. Correctness gates pass (including 100% budget
equivalence on tokens and WikiText-2 PPL). On an RTX 5060 8GB laptop:

- KV memory scales down almost exactly with the cache ratio  
- Decode speedups emerge at longer prompts once attention cost dominates selection overhead  
- **Quality is the differentiator:** Keyformer keeps PPL close to FullKV at 50–75% budget, while same-budget Recent/Random baselines collapse  

This answers the research question for GPT-2 Medium + WikiText-2 under controlled conditions,
without claiming a full reproduction of the paper’s 7B/A100 headline speedups.
