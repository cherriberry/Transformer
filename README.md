# Project overview

Course study of efficient Transformer variants. Each method has its own folder:

| Dir | Method | Owner focus |
|---|---|---|
| `keyformer/` | Keyformer KV-cache compression | This repo primary deliverable |
| `longformer/` | Longformer local+global attention | Teammate perf + quality added here |
| `linformer/` | Linformer low-rank attention | Teammate perf + quality added here |
| `xformer/` | xFormers memory-efficient attention | Teammate perf + quality added here |
| `memformer/` | Memformer-style memory attention | Teammate perf + quality added here |
| `performer/` | Performer FAVOR+ attention | Teammate perf + quality added here |
| `reformer/` | Reformer LSH attention | Teammate perf + quality added here |

Shared helpers: `common/wikitext_quality.py`

## Reproduce quality (WikiText-2)

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe longformer\quality_wikitext.py
.\.venv\Scripts\python.exe linformer\quality_wikitext.py
.\.venv\Scripts\python.exe xformer\quality_wikitext.py
.\.venv\Scripts\python.exe memformer\quality_wikitext.py
.\.venv\Scripts\python.exe performer\quality_wikitext.py
.\.venv\Scripts\python.exe reformer\quality_wikitext.py
```

Keyformer results are already under `keyformer/results/` and `keyformer/REPORT.md`.

Public repo: https://github.com/lzr20082024/26summerBDMI_transformer


codex resume 01a062a1-0fda-75a3-8b71-c84eb36d5d17