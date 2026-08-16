# Keyformer handoff (GPT-2 Medium stage)

## Status

Algorithm-level Keyformer (steps 7–10) remains in `models/keyformer.py`.

The GPT-2 Medium integration, synthetic performance sweep, WikiText-2 perplexity
evaluation, figures, and report are complete:

- `models/gpt2_keyformer.py`
- `tests/test_gpt2_keyformer.py`
- `benchmarks/keyformer_*.py`
- `configs/keyformer_experiment.yaml`
- `results/synthetic/perf_final.json`
- `results/wikitext2/quality_final.json`
- `results/figures/keyformer_perf.png`
- `results/figures/keyformer_quality.png`
- `KEYFORMER_REPORT.md`

## Not in scope here

- Longformer (teammate)
- TinyLlama / RoPE transfer (optional enhancement, not required)
- Claiming GPT-J 6B / A100 paper speedups

## Quick verify

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest -q tests/test_keyformer.py tests/test_gpt2_keyformer.py
```
