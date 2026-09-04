# TinyStories + TinyLM experiment

This is the active draft protocol for using the repository's from-scratch
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
split. Context length, cross-document packing, and a leakage-safe evaluation
split must be decided before the protocol is frozen. Until then this protocol
is `draft_pending_experiment_questions`.

