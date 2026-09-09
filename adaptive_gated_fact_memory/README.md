# Adaptive gated fact memory prototype

This separate research directory implements the first executable version of
the proposed architecture described in `../TINYLM_EXPERIMENT_QA.md`:

> exact sink-based local attention plus a shared, content-addressed episodic
> fact memory with explicit read, write, update, retention and eviction gates.

It does not modify the six-model runners, checkpoints or result reports.

## Current status

Implemented and tested:

- causal streaming sliding-window attention;
- four persistent attention-sink KV positions;
- physical local KV cache bounded by the configured attention budget;
- shared user fact slots with copied token payloads and metadata;
- bounded pending candidates across streamed chunks;
- per-round semantic top-k retrieval reused during assistant generation;
- gated memory fusion in selected decoder layers;
- straight-through write/update and effective retention/eviction paths;
- four fixed assistant-state slots;
- mixed-batch reset, reorder and detach support;
- explicit state-byte accounting and trainable auxiliary outputs.

The TinyStories backbone stage has now been trained once under the frozen
10M-token screening protocol. See
[`TINYSTORIES_STAGE1_REPORT.md`](TINYSTORIES_STAGE1_REPORT.md) for the result.
The same seed-17 checkpoint has also completed the first supervised structured
memory curriculum, reaching 86.67% answer Exact Match and 100% strict target
slot read hit over 120 validation examples. See
[`STRUCTURED_MEMORY_STAGE2_S17_REPORT.md`](STRUCTURED_MEMORY_STAGE2_S17_REPORT.md).
This is not yet a multi-seed or baseline-comparative result. See `DESIGN.md`
for exact boundaries and remaining work.

## Layout

```text
src/adaptive_fact_memory/
  attention.py   bounded sink + recent causal KV
  memory.py      fact extraction, read/write/retention/update controllers
  model.py       decoder blocks and round lifecycle
  state.py       explicit recurrent and pending state
configs/
  tinystories_stage1.yaml  resolved first-stage training settings
tests/
  test_prototype.py
scripts/
  smoke_test.py
  train_tinystories.py
  fetch_references.sh
references/      downloaded upstream code; ignored by the parent repository
```

Reference provenance and frozen revisions are recorded in
`SOURCE_MANIFEST.md`.

The paper-specific code audit is in
[`SWA_BASELINE_SOURCE_AUDIT.md`](SWA_BASELINE_SOURCE_AUDIT.md).  The paper has
no publicly identifiable author repository as of 2026-09-09, so the executable
SWA baseline delegates to the cited StreamingLLM source:
`adaptive_fact_memory.StreamingSWASourcePolicy`.  It accepts Hugging Face-style
`past_key_values`; converting the current TinyLM `LocalKVCache` to that format
must remain explicit in any experiment runner.

## Verify

No package installation is required when PyTorch is already available:

```bash
cd /root/BDMI/26summerBDMI_transformer
PYTHONPATH=adaptive_gated_fact_memory/src \
  python3 -m unittest discover -s adaptive_gated_fact_memory/tests -v

PYTHONPATH=adaptive_gated_fact_memory/src \
  python3 adaptive_gated_fact_memory/scripts/smoke_test.py
```

Alternatively:

```bash
python3 -m pip install -e adaptive_gated_fact_memory
```

## Minimal API

```python
import torch
from adaptive_fact_memory import AdaptiveFactMemoryLM, SourceRole

model = AdaptiveFactMemoryLM().eval().cuda()
state = model.initial_state(batch_size=1, device="cuda")

# User prefill opens the round, selects old memory and buffers fact candidates.
user_ids = torch.tensor([[101, 102, 103]], device="cuda")
user_roles = torch.full_like(user_ids, int(SourceRole.USER))
user = model.forward_round(
    user_ids, state=state, source_roles=user_roles, commit=False
)

# Assistant chunks reuse the same retrieval. Only the final chunk commits.
assistant_ids = torch.tensor([[201, 202]], device="cuda")
assistant_roles = torch.full_like(assistant_ids, int(SourceRole.ASSISTANT))
assistant = model.forward_round(
    assistant_ids,
    state=user.state,
    source_roles=assistant_roles,
    commit=True,
)
state = assistant.state

# At a new conversation boundary, reset the corresponding batch row.
state = state.reset(torch.tensor([True], device="cuda"))
```

During training, pass `hard_memory=False` and consume the tensors under
`output.supervision`. During evaluation, hard routing is the default.
