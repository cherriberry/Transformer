# 26summer BDMI — Transformer / Keyformer

Course deliverable focused on **Keyformer KV-cache compression** with GPT-2 Medium
and WikiText-2. Longformer microbenchmarks in this tree are secondary / teammate scope.

## Keyformer (main deliverable)

Research question: during autoregressive generation, can Keyformer keep language-model
quality while reducing KV-cache memory and decode latency?

| Item | Path |
|---|---|
| Report | [`KEYFORMER_REPORT.md`](KEYFORMER_REPORT.md) |
| GPT-2 integration | `models/gpt2_keyformer.py` |
| Algorithm policies | `models/keyformer.py` |
| Tests | `tests/test_keyformer.py`, `tests/test_gpt2_keyformer.py` |
| Perf / quality / plots | `benchmarks/` |
| Config | `configs/keyformer_experiment.yaml` |
| Results JSON | `results/synthetic/perf_final.json`, `results/wikitext2/quality_final.json` |
| Figures | `results/figures/` |

### Quick reproduce

```powershell
python -m pip install -r requirements-keyformer.txt
$env:PYTHONPATH = (Get-Location)
python -m pytest -q tests/test_keyformer.py tests/test_gpt2_keyformer.py
python benchmarks/keyformer_gpt2.py --mode final
python benchmarks/keyformer_quality.py --mode quality_final
python benchmarks/keyformer_plot.py
```

Hardware used for reported numbers: RTX 5060 Laptop 8GB, GPT-2 Medium FP16.

## Other attention microbenchmarks

The repository also contains adapters for Memformer / Performer / Reformer / Longformer
style modules used earlier as performance microbenchmarks. See the sections below and
`HANDOFF.md` / `SOURCES.md` for provenance. These are **not** the Keyformer quality study.

### Run (legacy microbenchmark)

```powershell
python -m pip install -r requirements-benchmark.txt
python run_benchmark.py --seq-lengths 128 256 512 --dim 256 --heads 8 --runs 3
```

### Test

```powershell
python -m pytest -q
```
