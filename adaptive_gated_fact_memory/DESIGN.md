# Architecture design and current implementation boundary

## Working definition

The prototype is a decoder-only causal Transformer with two time scales:

```text
current tokens -> 4 sinks + exact recent sliding KV ------------------+
                                                                    |
old committed facts -> semantic top-k -> gated memory fusion -------+-> logits

completed round -> bounded candidate buffer -> write / merge / update / evict
                                                    |
                                                    +-> shared fact bank
```

The goal is not to claim that every old token remains recoverable. Recent
tokens retain exact per-layer KV. Older user information survives only if the
learned controller copies a short payload and commits a semantic fact slot.

## Default TinyLM shape

| Component | Default |
|---|---:|
| layers / hidden / heads / FFN | 6 / 384 / 8 / 1536 |
| local attention budget | 128 unique keys |
| attention sinks | 4 |
| recent non-sink KV | 124 |
| user fact slots | 64 shared slots |
| assistant state slots | 4 fixed-role slots |
| retrieved facts | top 8 per round |
| fact candidates | at most 4 per round |
| copied payload | at most 12 token IDs per fact |
| fusion layers | layers 2 and 5, zero-indexed |

These are implementation defaults, not experimentally frozen optimum values.

## Causal round lifecycle

One round is one user message followed by one completed assistant response.

1. A user/system prompt opens a round and selects the top-k facts once.
2. Selected facts become visible at the last prompt token, whose logits predict
   the first assistant token. They are reused during assistant continuation.
3. User spans are accumulated in a bounded pending buffer. Assistant hidden
   states are accumulated as a bounded running sum and count.
4. `commit=True` writes the buffered user facts and assistant summary only
   after current logits have been computed.
5. The new facts can affect the next round, never an earlier/current token.
6. `ConversationState.reset()` clears all local KV, facts, pending candidates,
   cached retrieval choices and position counters for selected batch rows.

This lets one-shot training and streamed inference use the same semantics. A
test verifies equality between a full-round call and user-prefill plus multiple
assistant chunks.

## Runtime state

`ConversationState` contains every tensor that must follow a conversation when
a batch is reordered:

- per-layer sink and recent KV caches;
- shared user fact keys, values, copied payloads and active weights;
- age, last access, access count, confidence, conflict and source metadata;
- four assistant-state slots;
- pending user candidates and a running assistant summary;
- cached top-k retrieval indices for the open round;
- absolute next position and round-open flag.

`AdaptiveFactMemoryLM.state_bytes()` counts all of these tensors. It does not
pretend that a logical zero gate saves physical bytes.

## Implemented controllers

- **Span proposal:** scores user-token starts, predicts payload length and
  copies an auditable source span.
- **Write gate:** independently predicts whether each proposed span is worth
  retaining. The soft mode uses a straight-through threshold.
- **Slot addressing:** merges into a sufficiently similar fact; otherwise uses
  an empty slot or evicts the lowest-retention slot.
- **Retention gate:** controls whether an existing active slot survives the
  next commit.
- **Update gate:** interpolates old and new representations on merge/update.
- **Read routing:** gathers only top-k shared facts once per round.
- **Fusion gate:** injects the selected memory at two decoder layers.
- **Assistant pool:** maps the round summary into one of four fixed slots. Its
  role classifier needs explicit training supervision.

Trainable tensors needed for structured auxiliary objectives are returned in
`output.supervision`; budget and write-entropy terms are exposed in
`output.auxiliary_losses`. The model does not silently add arbitrary weights
to the language-model loss.

## What is deliberately not claimed

- The random initialization does not yet know what a fact is.
- Hard top-k candidate starts are not differentiable; structured span labels or
  a later soft-span relaxation are required.
- The slot conflict flag is only a conservative payload-difference diagnostic,
  not a complete entity/attribute contradiction resolver.
- The local attention implementation has bounded cache and bounded logical
  attention, but uses ordinary PyTorch tensor operations rather than a fused
  production kernel.
- The reference Performer branch is not mixed into v1. It remains a baseline.
- TriAttention is not mixed into v1. Its score should only be added after
  measuring pre-RoPE concentration and reconstruction correlation in TinyLM.
- A seed-17, 10M-token TinyStories backbone result now exists (full-validation
  NLL 2.889852, PPL 17.9906); see `TINYSTORIES_STAGE1_REPORT.md`.  This is
  only a local-language-model result: no fact-retrieval or dynamic-memory
  quality result exists until structured dialogue training is run.

## Next implementation gates

1. Add a synthetic A -> long B -> A dialogue generator with structured fact
   span, query and conflict labels.
2. Add losses for start/length, write, read, retention, assistant role and
   query-answer recovery; then train the controller curriculum.
3. Compare fixed memory, learned soft gates and hard routing at matched state
   bytes.
4. Add a fused or SDPA-backed sliding-window path before reporting speed.
5. Calibrate TinyLM pre-RoPE Q/K concentration before testing a TriAttention
   auxiliary feature.
6. Implement physically packed variable active slots before claiming dynamic
   peak-memory reduction. The current user slot tensor remains `M_max`-sized.
