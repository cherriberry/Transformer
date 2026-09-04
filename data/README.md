# Local experiment data

Large dataset and tokenizer assets in this directory are intentionally ignored
by Git. Recreate and verify them with:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python experiments/tinystories_tinylm_v1/prepare_data.py
```

The generated `tinystories_manifest.json` records pinned upstream revisions,
file sizes, and SHA-256 checksums. TinyStories has upstream `train` and
`validation` splits but no official `test` split.

