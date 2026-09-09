# Reference source manifest

These repositories are read-only design references. The prototype under
`src/` is an independent implementation and does not import code from them.
The directories are ignored by the parent Git repository because they are
reproducible upstream downloads with their own licenses.

| Reference | Upstream | Frozen revision | Local path | License | Mechanism inspected |
|---|---|---|---|---|---|
| Performer | `https://github.com/google-research/google-research/tree/master/performer` | `952b8d85a0389ba59d27de382046a6dd9c01b0e5` | `references/google-research/performer/` | Apache-2.0 | FAVOR+ random-feature attention and causal prefix accumulation |
| Google Gemma PyTorch | `https://github.com/google/gemma_pytorch` | `014acb7ac4563a5f77c76d7ff98f31b568c16508` | `references/gemma_pytorch/` | Apache-2.0 | `LOCAL_SLIDING` mask construction, RoPE and KV-cache layout |
| StreamingLLM | `https://github.com/mit-han-lab/streaming-llm` | `2e5042606d69933d88fbf909bd77907456b9b4dd` | `references/streaming-llm/` | MIT | physical `start_size + recent_size` cache eviction / attention sinks |
| TriAttention | `https://github.com/WeianMao/triattention` | `a4bc3c8f709db60f016ef42c3feb290fd0c00c1b` | `references/triattention/` | Apache-2.0 | pre-RoPE trigonometric scoring, calibration and physical KV compaction |
| Memformer / MemBART | `https://github.com/qywu/memformers` | `4575d14a76284eb32d2f5e2f52cbf2378ad829f2` | `references/memformers/` | MIT | recurrent memory state, attention fusion and learned update gate |
| *Sliding-window beats linear attention* (arXiv:2608.28444v1) | **No public author repository found as of 2026-09-09** | N/A | N/A | N/A | paper-aligned SWA definition; use StreamingLLM source for the executable sink-cache baseline |

Archive SHA-256 values used for the non-Git downloads:

```text
gemma_pytorch  551974218cbd7a2323109111c592c8faac4dcf07a2a1eaf47e5595744b0215b2
streaming-llm  cc72a91243c3505cbb4023ed996aeead1e1d5824c68834cb5d4e41e887080ceb
triattention   4f06fafce1e1356d54c47486f213a00df2c0ce1d58adf72d6a7cd006bc472c06
memformers     89d7b44ae36dbf479cbb59ee99ac6936e6d546c27562900c5f9838c47eb8e121
```

Important source locations:

- Gemma sliding mask: `references/gemma_pytorch/gemma/model.py`
- Streaming sink eviction: `references/streaming-llm/streaming_llm/kv_cache.py`
- TriAttention equation-level scoring:
  `references/triattention/triattention/methods/pruning_utils.py`
- TriAttention physical runtime compressor:
  `references/triattention/triattention/vllm/core/compressor.py`
- Memformer gate:
  `references/memformers/memformers/models/membart/membart_encoder.py`
- Google Performer JAX implementation:
  `references/google-research/performer/fast_attention/jax/fast_attention.py`

Run `scripts/fetch_references.sh` from this directory to reconstruct missing
reference trees. Existing directories are never overwritten.

The absence of an author repository for arXiv:2608.28444v1 and the resulting
baseline naming rules are documented in
[`SWA_BASELINE_SOURCE_AUDIT.md`](SWA_BASELINE_SOURCE_AUDIT.md).
