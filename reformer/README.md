# reformer

## Contents
- `benchmark_perf.py` / figures: performance study (teammate baseline where applicable)
- `quality_wikitext.py`: WikiText-2 real-data quality evaluation
- `results/`: JSON outputs
- `figures/`: plots

## Run quality
```powershell
$env:PYTHONPATH = (Resolve-Path ..)
python quality_wikitext.py
```
