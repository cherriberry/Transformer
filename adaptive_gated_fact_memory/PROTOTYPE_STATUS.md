# Prototype status — 2026-09-09

## Completed in this pass

- Downloaded and pinned all five reference implementations listed in
  `SOURCE_MANIFEST.md`; the Sliding-window paper source audit records that no
  author repository is publicly available as of 2026-09-09.
- Added a source-backed `StreamingSWASourcePolicy` adapter that delegates to
  the frozen upstream `StartRecentKVCache` implementation.  It is a baseline
  policy adapter, not a claim of reproducing the paper author's code.
- Implemented an independent PyTorch TinyLM with sink-based local attention,
  physically bounded local KV, shared fact memory and explicit dialogue state.
- Implemented user span candidates, copied payloads, write/update/retention/
  eviction control, top-k read routing, gated fusion and assistant-state slots.
- Implemented a bounded within-round candidate buffer so one-shot training and
  streamed user-prefill/assistant-generation have the same commit semantics.
- Added state reset, batch reorder, detach and full byte accounting.

## Automated verification

Command:

```bash
PYTHONPATH=adaptive_gated_fact_memory/src \
  python3 -m unittest discover -s adaptive_gated_fact_memory/tests -v
```

Result: **14/14 tests passed**. The tests cover:

1. one-shot vs chunked local attention;
2. causal prefix invariance;
3. physical local-cache bound;
4. commit cannot alter current-round logits;
5. memory is hidden before the answer boundary;
6. retrieval is reused rather than repeated during assistant continuation;
7. one-shot vs streamed round and memory-state equality;
8. selective conversation reset;
9. state-safe batch reorder;
10. padding behavior;
11. write-gate gradient and effective retention eviction.
12. source-backed sink/recent cache retention and no-op behavior (the adapter
    exposes the frozen upstream URL and revision).

The CUDA smoke test also completed a forward, commit, next-round read and
backward pass.

## Default-config RTX 4090 audit

Configuration: 6 layers, hidden 384, 8 heads, FFN 1536, BF16, batch 1,
160 tokens, local budget 128, 4 sinks, 64 user slots, 4 assistant slots.

```text
total parameters                  31,854,750
approximate non-memory parameters 29,927,040
logits shape                      [1, 160, 50,257]
local cache entries per layer     128
explicit runtime state            1,416,149 bytes (1.351 MiB)
single cold forward               660.3 ms
peak allocated during audit       88.5 MiB
```

The cold-forward and allocator measurements are only execution checks. They
are not a speed comparison: the prototype uses unfused Python/PyTorch control
flow, includes the vocabulary projection, and was not warmed or benchmarked
under the unified protocol.

## TinyStories stage-1 result

The explicit TinyStories backbone entry point is
`scripts/train_tinystories.py`.  A seed-17, 10M-token run completed on one
RTX 4090 with full validation:

- validation NLL: **2.889852**;
- validation PPL: **17.9906**;
- training time: **430.9 s**;
- peak allocated memory: **4.66 GB**.

This stage uses a fresh state for each packed block and disables the episodic
path.  A TinyStories corpus has no fact-span or dialogue-round labels, so the
memory controller parameters remain at their seed-17 initialization.  The
result is a learned local-language-model backbone, not evidence of memory
recall.  Full details and data-disk artifact paths are in
`TINYSTORIES_STAGE1_REPORT.md`.

## Structured memory stage-2 result

The seed-17 backbone has now completed a 2,000-step structured
`fact -> long distractor -> query` curriculum. Strict hard-threshold validation
over 120 examples produced:

- Exact Match: **86.67% (104/120)**;
- strict target-slot read hit: **100% (120/120)**;
- mean active user slots: **1.0**;
- state size: **2,598,885 bytes**, invariant across tested delays;
- 16-segment delay Exact Match/read hit: **87.50% / 100%**, although training
  used delays only through 8 segments.

See `STRUCTURED_MEMORY_STAGE2_S17_REPORT.md` for the complete configuration,
artifact paths and limitations.

## Not completed yet

- No structured-memory baseline or quality–state Pareto comparison exists yet.
- Only seed 17 has completed structured-memory training; three-seed variance
  is not available.
- Multi-fact capacity, conflict/update behavior and label-free extraction have
  not been validated.
- Span proposal currently needs structured auxiliary supervision.
- Dynamic user slots are logically selected but still preallocated to
  `M_max`; active-slot packing is not implemented.
- TriAttention scoring and Performer are references/baselines, not silently
  mixed into this first model.
- Production fused local attention and packed memory kernels remain future
  engineering work.
