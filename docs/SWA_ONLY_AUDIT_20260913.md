# SWA-only baseline audit (2026-09-13)

This note records the audit of the `memory_policy=none` stress baseline.  The
existing checkpoints and reports are historical artifacts and are not changed.

## Finding

The 5.21% Exact Match result is not evidence of a broken SWA cache.  It is the
combination of a deliberately out-of-window task, an answer-tokenization issue,
and a very small/imbalanced training signal.

## Cache and paper alignment

`StreamingSinkSlidingAttention` retains four initial sink KV pairs and the
newest 124 non-sink KV pairs, for at most 128 unique visible keys.  The source
policy in `baselines.py` has the same retention rule.  Existing causal,
chunking, bounded-cache, reset/reorder, and commit-order tests pass (23/23 in
the audit run); one-shot versus chunked logits differ only at floating-point
noise (`~1.6e-7`).  No cache-position or future-token leak was found.

This is still not the paper's exact experiment: the paper describes a
training-free inference-time replacement on an already trained large model.
The repository baseline starts from a TinyStories model already trained with
SWA and then performs 2,000 structured stress-training steps.  It should be
called a *trained TinyLM SWA-sink floor*, not a training-free paper reproduction.

## Why 5.21% is expected for the current task

The stress generator uses a minimum filler length of `128 + extra` tokens even
for `delay=1`.  The durable fact is six tokens long, so it is necessarily
evicted from the 124-token recent window before the query.  With
`memory_policy=none`, there is no alternate state that can retain it.

Names and colors are deterministically randomized per example.  The 16-color
answer is therefore not parameter-memory knowledge; the model must copy a
random value from outside its local window.  The final checkpoint emits
`[77, 2830]`, i.e. `navy`, for many examples.  Its 5.21% EM is consequently
close to the 1/16 random color rate, not a meaningful long-memory score.

The parent TinyStories checkpoint, without stress fine-tuning, obtains no
useful answer signal (first-token NLL about 15.7 on a small validation probe).
Conversely, a fixed-example SWA-only model overfits an in-window example to
100% EM in about 300 updates.  These two checks bracket the issue: the cache
can copy, but the formal random out-of-window task supplies no usable memory
path and is not trained to convergence for this algorithmic behavior.

## Problems in the experimental protocol

1. **Not paper-comparable.**  The stress run is a task-specific fine-tune, not
   inference-only SWA conversion of a full-attention checkpoint.
2. **Answer boundary/tokenization.**  `answer_leading_space` defaults to false.
   In GPT-2, `navy` is `[77, 2830]`, while ` navy` is the single token
   `[23956]`; analogous differences occur for the other colors.  The answer
   is concatenated after `?`, so the default target is not natural GPT-2
   continuation text and gives unequal target lengths.  The option already
   exists in `synthetic_memory_stress.py` and should be enabled for a corrected
   run.
3. **Weak/short SWA-only training signal.**  In the `none` branch,
   `stress_training_batch_loss` computes loss only on the assistant answer
   positions.  Each example contributes only the answer token(s); fact and
   filler language-model positions are not trained.  The run therefore sees
   8,000 examples but only roughly 16,000 supervised answer positions, while
   the random name-to-color mapping requires learning a copy procedure.
4. **No validation-based checkpoint selection.**  The final checkpoint is
   always evaluated.  On the saved SWA-only checkpoints, the `noise=0,
   delay=1` probe was 8.33% EM at step 500/1000, 4.17% at step 1500, and 0%
   at step 2000.  Training loss alone did not select the best answer behavior.

The memory modules being present in the `none` model and receiving only weight
decay is harmless for outputs because `forward_round` disables the memory path;
it is bookkeeping overhead, not the cause of the low EM.

## Corrected follow-up protocol

Keep the historical run labelled as the floor, then run separate, explicitly
named controls:

- `answer_leading_space=True` for all stress conditions;
- an in-window control (`segment_length` below 124) to establish a local-copy
  upper bound;
- a full-attention or sufficiently large-window control with the same training
  objective;
- periodic held-out answer evaluation and best-checkpoint selection;
- a longer/curriculum SWA-only run, or a standard full-sequence LM objective
  plus an explicitly reported answer-loss weight;
- a true paper-style control: full-attention pretrained checkpoint, no
  structured fine-tuning, then inference-time SWA(128,4).

Only after these controls should SWA-only be compared with gated memory.  The
existing 5.21% number should not be interpreted as a failed implementation or
as a paper-level SWA result.
