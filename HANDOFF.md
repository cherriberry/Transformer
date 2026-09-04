# Handoff: Efficient Transformer Benchmark

## Objective and current state

The workspace contains source snapshots and a runnable GPU microbenchmark for
Memformer, Performer, and Reformer, with standard full attention as a baseline.
The user then asked whether inference accuracy should be measured. The correct
answer is: it is not needed for a performance-only microbenchmark, but it is
required before describing the work as a paper-level reproduction or comparing
model quality.

The benchmark has been run successfully on the existing `Summer` Conda
environment and RTX 4070 Laptop GPU. All three adapters have passed a CUDA
forward smoke test with output shape `(1, 128, 256)`.

## Source provenance

See `SOURCES.md` for links and provenance.

- `memformers`: source by the Memformer paper's first author, Qingyang Wu.
- `performer-pytorch`: runnable PyTorch implementation of the Performer paper.
  The official source is `google-research/google-research/performer`.
- `reformer-pytorch`: runnable PyTorch implementation of Reformer. The author
  implementation is in `google/trax`.

Direct `git clone` was blocked by the sandbox network policy. The runnable
projects were downloaded through GitHub codeload archives, so they do not have
`.git` metadata. The original zip files are retained at the workspace root.

## Benchmark implementation

- `universal_benchmark.py` was updated to run inference under
  `torch.inference_mode()` and to report CUDA peak allocated memory using
  `torch.cuda.max_memory_allocated()`. On CPU it still reports process RSS.
- `run_benchmark.py` imports and wraps the three implementations around the
  `(batch, sequence, dim) -> same shape` interface expected by the benchmark.
- `MemformerAdapter` wraps the author implementation's
  `MemBartEncoderAttention` with 64 persistent external memory slots. It is
  **not** a loaded or trained full MemBART model.
- `PerformerAdapter` uses the PyTorch `Performer` configuration with global
  FAVOR+ attention and no local-attention heads.
- `ReformerAdapter` uses the PyTorch `Reformer` with LSH attention, a 64-token
  bucket and two hashes. Its sequence lengths must be divisible by 64.
- `compat_deps/` contains minimal compatibility modules used only when the
  active environment lacks old dependencies (`einops`, `local_attention`,
  `axial_positional_embedding`, and `product_key_memory`). The selected model
  configurations do not enter local-attention, axial-position, or PKM paths.
  If installing genuine dependencies, `run_benchmark.py` will prefer them.

## Verified environment and commands

The default `python` in this shell does not contain PyTorch. Use:

```powershell
$env:MPLBACKEND='Agg'
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\fyh_1\.conda\envs\Summer\python.exe' run_benchmark.py `
  --seq-lengths 128 256 512 --dim 256 --heads 8 --runs 3 `
  --output benchmark_results.png --json-output benchmark_results.json
```

`Summer` has PyTorch 2.11.0 with CUDA available. It does not have the old
auxiliary packages, so the built-in compatibility directory is needed there.

Static syntax checks and CUDA smoke test have passed. The plot emits warnings
because the active Matplotlib font lacks CJK glyphs; the PNG and JSON are still
written successfully.

## Current results

Configuration: inference mode, batch size 1, `dim=256`, 8 heads, three timed
runs per length, RTX 4070 Laptop GPU. Values are latency in milliseconds and
CUDA peak allocated memory in MiB.

| Model | L=128 | L=256 | L=512 |
|---|---|---|---|
| Standard full attention | 0.378 ms / 11.88 MiB | 0.368 ms / 15.63 MiB | 0.664 ms / 29.13 MiB |
| Memformer attention adapter | 0.789 ms / 14.32 MiB | 0.623 ms / 18.82 MiB | 1.060 ms / 33.82 MiB |
| Performer | 1.254 ms / 16.80 MiB | 1.116 ms / 21.43 MiB | 2.328 ms / 30.69 MiB |
| Reformer | 1.943 ms / 17.41 MiB | 2.013 ms / 22.93 MiB | 2.317 ms / 33.98 MiB |

Full machine-readable results are in `benchmark_results.json`, with the plot
in `benchmark_results.png`. These lengths are too short to demonstrate the
asymptotic advantages of Performer/Reformer; standard CUDA attention is faster
at this scale.

## Important limitations

1. This is an inference performance microbenchmark, not a full paper
   reproduction and not an accuracy study.
2. Random inputs and randomly initialized models mean no task metric (accuracy,
   F1, BLEU, or perplexity) can be reported.
3. The baseline and adapters are not parameter-matched complete models. The
   test fairly compares their callable attention-style implementations under
   the shared input/output interface, not their training quality.
4. `max_memory_allocated` is PyTorch's allocated-memory high water mark. It
   does not represent all CUDA reserved memory or total process GPU footprint.

## Longformer extension

The repository now contains a Longformer-style attention microbenchmark in
`models/longformer_attention.py`, including local-only and local-plus-global
variants. The implementation uses a strided sliding-window view and creates
attention scores of shape `[batch, heads, sequence, window]`, not a quadratic
sequence-by-sequence score matrix. `tests/test_longformer.py` verifies shape,
padding, boundary behavior, global-token reachability, gradients, full-window
equivalence, and the storage strategy.

`universal_benchmark.py` now preserves one structured record per requested
sequence length, reports mean/median/std and raw timings, records OOM/errors
without losing alignment, and uses CUDA events on GPU. `run_benchmark.py`
includes Longformer and stores configuration plus environment metadata.

## Recommended next work

1. Add `accuracy_benchmark.py` with module-level numerical checks:
   - Performer: share projection/output weights with a full-attention reference;
     report relative L2 error, MSE, and cosine similarity across seeds,
     sequence lengths, and feature counts.
   - Reformer: use a same-weight full-attention reference where feasible;
     report the same metrics over varying `n_hashes` and bucket sizes. State
     clearly that LSH is an approximation.
   - Memformer: test deterministic state propagation and output shapes across
     chunks; do not claim equivalence to full attention, as external memory
     changes the architecture.
2. Run longer performance lengths such as `1024 2048 4096` with `dim=256` and
   perhaps fewer timed runs. Reformer requires multiples of 128 for the current
   64-token bucket configuration. OOM is recorded per model/length instead of
   ending the entire run.
3. For a task-level reproduction, choose a specific original-paper dataset and
   metric first. Then load or train the full architectures with matched model
   budgets and report the appropriate metric. For Memformer, use its complete
   MemBART/training workflow, not the attention adapter used in this benchmark.
4. Consider installing the real dependencies from `requirements-benchmark.txt`
   into a dedicated environment before an extended study. The current fallback
   is sufficient only for the selected microbenchmark paths.
