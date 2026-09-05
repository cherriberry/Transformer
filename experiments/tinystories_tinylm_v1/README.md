# TinyStories + TinyLM experiment

This is the active time-boxed screening protocol for using the repository's from-scratch
`TinyLM-long-v1` with `roneneldan/TinyStories`. The older
`long_context_10m_v1` directory is retained as historical PG-19/synthetic-pilot
provenance and must not be relabeled as TinyStories.

## Prepare local assets

```bash
python -m pip install -r requirements-tinylm.txt
HF_ENDPOINT=https://hf-mirror.com \
python experiments/tinystories_tinylm_v1/prepare_data.py
```

The script downloads only TinyStories parquet data and GPT-2 tokenizer files.
It does not download GPT-2 model weights. Files are stored under `data/`, and a
checksum manifest is written to `data/tinystories_manifest.json`.

## Important dataset constraint

TinyStories is made of short, independent stories and has no official test
split. The Person-B Longformer/Memformer execution has now been completed under
the resolved screening configuration. In addition to the 10M-token screening,
the same frozen configurations were extended to 100M tokens on the writable
data disk. The results remain validation-screening/extension evidence rather
than a fully converged or paper-faithful reproduction; the 100M run uses one
seed and its validation curve is still descending.

## Person-B result entry points

- Quick-look result report: `../../PERSON_B_TINYLM_SIMPLE_RESULT_REPORT.md`
- Detailed experiment document: `../../PERSON_B_TINYLM_DETAILED_EXPERIMENT_DOCUMENT.md`
- Compact result report: `../../PERSON_B_TINYLM_REPORT.md`
- Correctness: `aggregate/person_b_correctness.json`
- Pilot selection: `aggregate/person_b_pilot_selection.json`
- Main-run summaries: `aggregate/person_b_runs.json` and
  `aggregate/person_b_final_summary.json`
- 100M extension summaries: `/root/autodl-tmp/26summerBDMI_transformer/runs/person_b_extended/extended100m_longformer_w128_s17/summary.json`
  and `/root/autodl-tmp/26summerBDMI_transformer/runs/person_b_extended/extended100m_memformer_s128_m64_s17/summary.json`
- End-to-end and attention-only efficiency: `aggregate/person_b_efficiency.json`
  and `aggregate/person_b_attention_efficiency.json`

The training runner is `run_person_b.py`; its `correctness`, `train`, and
`aggregate` subcommands reproduce the checks and summary schema. The two
benchmark scripts measure frozen-checkpoint efficiency separately from quality
training.

## Extended runs on the writable data disk

The completed 10M-token screening remains in the repository paths above. For
longer or multi-seed runs, redirect generated cache/checkpoint/log artifacts to
the writable data disk (`/root/autodl-tmp`, about 50GB available in this
environment):

```bash
export TINYSTORIES_CACHE_DIR=/root/autodl-tmp/26summerBDMI_transformer/data/cache/tinystories_tinylm_v1
export PERSON_B_RUNS_DIR=/root/autodl-tmp/26summerBDMI_transformer/runs/person_b_extended
export PERSON_B_AGGREGATE_DIR=/root/autodl-tmp/26summerBDMI_transformer/aggregate
mkdir -p "$TINYSTORIES_CACHE_DIR" "$PERSON_B_RUNS_DIR" "$PERSON_B_AGGREGATE_DIR"
```

`TINYSTORIES_DATA_DIR` and `TINYSTORIES_TOKENIZER_DIR` can also be set when
those read-only assets are mirrored elsewhere. Defaults still point to the
repository, so existing formal results are not changed. For a 50M/100M run,
use a new `run-id`, set `--checkpoint-interval-tokens 10000000`, and retain the
best and final checkpoints rather than writing one checkpoint every 1M tokens.
