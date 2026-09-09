# Structured memory stage-2 report — seed 17

## Outcome

The seed-17 TinyStories backbone was trained end to end on the structured
conversation curriculum:

```text
early user fact A
-> long TinyStories/system distractor B
-> user query asking for A
```

The formal run completed all 2,000 steps without OOM, NaN or a non-finite
gradient. Under the ordinary hard inference threshold (`0.5`), strict
full-payload matching found the correct memory slot in all 120 validation
examples. Greedy answer Exact Match was 104/120 (86.67%).

## Parent checkpoint

- checkpoint:
  `/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_tinystories/tinystories_backbone_s17_10m_20260908/checkpoint.final.pt`
- seed: 17
- parent TinyStories validation NLL/PPL: 2.889852 / 17.9906
- parent checkpoint SHA-256:
  `d995248587a5e535af59d850f36a5bc01925c54bd9dff213f0f18bfb26f2b78f`

The parent checkpoint trained only the local-language-model backbone. This
stage trained the previously initialized fact extraction, memory control,
semantic read and gated fusion paths together with a smaller backbone learning
rate.

## Formal training configuration

| Item | Value |
|---|---:|
| Steps | 2,000 |
| Micro-batch | 2 examples |
| Gradient accumulation | 2 |
| Effective batch | 4 examples |
| Examples seen | 8,000 |
| Optimizer | AdamW, betas=(0.9, 0.95), eps=1e-8 |
| Memory/controller learning rate | 3e-4 |
| Backbone learning rate | 3e-5 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Training write/retention threshold | 0.1 / 0.1 |
| Ordinary inference threshold | 0.5 / 0.5 |
| Training delay buckets | 1, 2, 4, 8 segments |
| Validation delay buckets | 1, 2, 4, 8, 16 segments |
| Segment length | 128 tokens plus a small deterministic variation |
| Local KV budget | 4 sink + 124 recent = 128 keys |
| User memory capacity | 64 slots, 12 copied payload tokens per slot |
| Read routing | semantic top-k, k=8 |
| Precision/device | BF16 autocast, one RTX 4090 |

Each example contains one short fact such as `Bob likes red.`, a natural-text
TinyStories distractor in the system role, and a query such as
`What does Bob like?`. Validation uses a deterministic shifted name/value
pairing and never contributes gradients.

## Training details needed for interpretation

- The labelled fact start and length are supervised.
- The span-level write decision is trained by marking all tokens in the
  labelled fact span as positive. Supervising only the first token was found
  to drive the averaged span logit below the write threshold.
- During training only, a one- or two-token fact prefix may identify the target
  slot while the length head is still learning. All reported evaluation uses
  strict equality with the complete copied payload.
- Long recurrent backpropagation is truncated at round boundaries. Round A
  trains fact extraction/write/key alignment, round B trains the local path and
  retention behavior, and the query round trains semantic read/fusion and
  answer generation. Consequently this run validates the mechanism under a
  supervised curriculum; it is not evidence that the full hard controller can
  be learned without structured labels.
- Assistant-state writes are suppressed in this one-fact first curriculum.

## Validation result

The two threshold evaluations were identical. Therefore the result is not an
artifact that exists only under the permissive training threshold.

| Delay segments | Approx. distractor scale | Examples | Exact Match | Strict read hit | Mean active user slots | State bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128+ tokens | 24 | 87.50% | 100% | 1.0 | 2,598,885 |
| 2 | 256+ tokens | 24 | 87.50% | 100% | 1.0 | 2,598,885 |
| 4 | 512+ tokens | 24 | 87.50% | 100% | 1.0 | 2,598,885 |
| 8 | 1,024+ tokens | 24 | 83.33% | 100% | 1.0 | 2,598,885 |
| 16 | 2,048+ tokens | 24 | 87.50% | 100% | 1.0 | 2,598,885 |
| **Overall** | — | **120** | **86.67% (104/120)** | **100% (120/120)** | **1.0** | **2,598,885** |

The 16-segment bucket is outside the training delay range. Its unchanged read
hit rate and comparable Exact Match provide an initial positive result for
distance extrapolation. The answer errors in the 16-segment bucket (and, in
fact, all answer errors in the table) occurred after the correct slot had been
retrieved: retrieval and answer realization are therefore separate bottlenecks
in this run.

## Training and resource statistics

| Metric | Result |
|---|---:|
| Training time | 1,127.11 s (18 min 47 s) |
| Mean training rate | 7.10 examples/s |
| Best observed batch total loss | 0.178628 |
| Final batch total loss | 0.339365 |
| Final answer/assistant LM loss | 0.169443 |
| Final supervised read rows | 2/2 |
| Final read target score | 1.000 |
| Final active slots | 1.0 |
| Final gradient norm before clipping | 4.642 |
| Peak CUDA allocated | 4,770,630,144 bytes (4.44 GiB) |
| Peak CUDA reserved | 9,722,396,672 bytes (9.06 GiB) |

## Artifact locations

- run directory:
  `/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_structured/structured_memory_s17_20260909`
- final checkpoint: `checkpoint.final.pt`
- intermediate checkpoints: `checkpoint.step_500.pt`,
  `checkpoint.step_1000.pt`, `checkpoint.step_1500.pt`,
  `checkpoint.step_2000.pt`
- resolved configuration: `config.resolved.json`
- per-step metrics: `metrics.jsonl`
- complete validation rows and aggregate result: `summary.json`
- console log:
  `/root/autodl-tmp/26summerBDMI_transformer/runs/adaptive_gated_fact_memory_structured/structured_memory_s17_20260909.log`

The run configuration hash is
`12d5495fc9b912ef30bdaa9ed5dc44582d5fa61d8975fc6a7783ef29b6192f74`.

## Scope of the conclusion

This run establishes that the seed-17 backbone plus the proposed supervised
fact-memory path can write one complete fact, preserve a single active slot,
retrieve it after distractors much longer than the local KV budget, and use it
to generate the correct answer in most held-out examples. It does not yet
establish multi-fact capacity, conflict/update behavior, robustness without
span labels, three-seed variance, or superiority to Sliding Window and
Memformer on this structured task. Those require the remaining baseline,
ablation and multi-seed runs.
