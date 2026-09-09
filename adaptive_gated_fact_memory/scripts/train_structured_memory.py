"""Train and evaluate the seed-17 structured fact-memory curriculum.

The parent TinyStories checkpoint supplies the language-model backbone.  This
stage adds a controlled conversation task:

    early user fact -> long system/TinyStories distractor -> user query

The query answer is teacher-forced during training and greedily generated at
evaluation.  Auxiliary losses supervise fact-span starts/lengths and write
probabilities; a differentiable contrastive loss trains the semantic reader
against the slot containing the known payload.  The generator uses TinyStories
tokens only as distractors, so its exact-match result is kept separate from
the TinyStories NLL/PPL table.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "adaptive_gated_fact_memory" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from adaptive_fact_memory import (  # noqa: E402
    AdaptiveFactMemoryLM,
    FactMemoryConfig,
    SourceRole,
)
from experiments.tinystories_tinylm_v1 import run_person_b as protocol  # noqa: E402
from synthetic_memory import (  # noqa: E402
    CollatedRound,
    MemoryExample,
    SyntheticMemoryGenerator,
)


TOKENIZER_DIR = Path(
    os.environ.get("TINYSTORIES_TOKENIZER_DIR", str(DATA_DISK / "data/tokenizer/gpt2"))
)
CACHE_DIR = Path(
    os.environ.get(
        "TINYSTORIES_CACHE_DIR",
        str(DATA_DISK / "data/cache/tinystories_tinylm_v1"),
    )
)
FILLER_PATH = CACHE_DIR / "train_100000257.int32.bin"
BACKBONE_CHECKPOINT = Path(
    os.environ.get(
        "ADAPTIVE_TINYSTORIES_CHECKPOINT",
        str(
            DATA_DISK
            / "runs/adaptive_gated_fact_memory_tinystories"
            / "tinystories_backbone_s17_10m_20260908"
            / "checkpoint.final.pt"
        ),
    )
)
RUNS_DIR = Path(
    os.environ.get(
        "ADAPTIVE_MEMORY_RUNS_DIR",
        str(DATA_DISK / "runs/adaptive_gated_fact_memory_structured"),
    )
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def environment_record(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu": props.name,
                "gpu_total_memory_bytes": props.total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return result


def masked_loss_sum(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    valid_positions: torch.Tensor,
) -> torch.Tensor:
    """Causal CE where ``valid_positions[i]`` scores input token i+1."""

    targets = torch.roll(input_ids, shifts=-1, dims=1)
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    return (losses * valid_positions.reshape(-1)).sum()


def model_parameter_groups(model: torch.nn.Module) -> list[dict[str, Any]]:
    memory_markers = (
        "fact_extractor",
        "memory_controller",
        "memory_reader",
        "memory_norm",
        "memory_fusion",
    )
    backbone: list[torch.nn.Parameter] = []
    memory: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(marker in name for marker in memory_markers):
            memory.append(parameter)
        else:
            backbone.append(parameter)
    # A smaller backbone LR retains the TinyStories language prior while the
    # newly initialized controller/fusion learns the synthetic task quickly.
    groups: list[dict[str, Any]] = []
    if backbone:
        groups.append({"params": backbone, "lr": 3.0e-5, "name": "backbone"})
    if memory:
        groups.append({"params": memory, "lr": 3.0e-4, "name": "memory"})
    return groups


def configure_memory_policy(
    model: AdaptiveFactMemoryLM,
    memory_policy: str,
) -> None:
    """Freeze parameters whose behavior is disabled by a baseline policy."""

    if memory_policy == "none":
        for name, parameter in model.named_parameters():
            if any(
                marker in name
                for marker in (
                    "fact_extractor",
                    "memory_controller",
                    "memory_reader",
                    "memory_norm",
                    "memory_fusion",
                )
            ):
                parameter.requires_grad_(False)
        return

    if memory_policy == "fixed":
        # Fixed memory uses the same candidate/key/value and fusion projections
        # as the proposed model, but these gate heads are ignored by the
        # controller/fusion implementation and must not receive optimizer
        # updates that would have no effect on the forward path.
        frozen_markers = (
            "fact_extractor.write_head",
            "fact_extractor.confidence_head",
            "memory_controller.retention_head",
            "memory_controller.update_head",
            "memory_controller.assistant_write_head",
            "memory_controller.assistant_slot_head",
            "memory_controller.assistant_key",
            "memory_controller.assistant_value",
            ".memory_fusion.gate",
        )
        for name, parameter in model.named_parameters():
            if any(marker in name for marker in frozen_markers):
                parameter.requires_grad_(False)


def load_stage1_model(
    device: torch.device,
    *,
    training: bool,
    memory_policy: str = "gated",
) -> AdaptiveFactMemoryLM:
    config = FactMemoryConfig(
        # A low curriculum threshold lets an initially sparse write gate
        # expose a slot to the reader.  Evaluation below also reports the
        # default-threshold result separately when requested.
        write_threshold=0.1 if training else 0.5,
        retention_threshold=0.1 if training else 0.5,
    )
    model = AdaptiveFactMemoryLM(config, memory_policy=memory_policy).to(device)
    checkpoint = torch.load(BACKBONE_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    # Assistant-state memory is not part of this first single-fact curriculum;
    # suppress accidental writes while retaining the module for later stages.
    with torch.no_grad():
        model.memory_controller.assistant_write_head.bias.fill_(-10.0)
    configure_memory_policy(model, memory_policy)
    return model


def payload_slot(
    state,
    payload: tuple[int, ...],
    *,
    active_threshold: float,
    relaxed: bool = False,
) -> int | None:
    """Find the slot containing ``payload``.

    During the early curriculum the length head may commit only a prefix of
    the labelled fact.  ``relaxed=True`` therefore accepts a one- or
    two-token prefix, which gives the reader a training target before span
    length prediction has converged.  Evaluation keeps the default strict
    full-payload match so reported recall cannot be inflated by partial
    writes.
    """
    target = torch.tensor(payload, device=state.memory.payload_ids.device, dtype=torch.long)
    for slot in range(state.memory.payload_ids.shape[1]):
        if float(state.memory.active[0, slot].detach()) <= active_threshold:
            continue
        mask = state.memory.payload_mask[0, slot]
        stored = state.memory.payload_ids[0, slot][mask]
        if not relaxed:
            if int(stored.numel()) != len(payload):
                continue
            matches = torch.equal(stored, target)
        else:
            # Require as much of the prefix as is available, up to two tokens.
            # A one-token prefix is useful at initialization; two tokens are
            # preferred once the extractor starts predicting longer spans.
            prefix_length = min(int(stored.numel()), len(payload), 2)
            if prefix_length < 1:
                continue
            matches = torch.equal(stored[:prefix_length], target[:prefix_length])
        if matches:
            return slot
    return None


def all_memory_read_loss(
    model: AdaptiveFactMemoryLM,
    state,
    query_round: CollatedRound,
    examples: list[MemoryExample],
) -> tuple[torch.Tensor, list[int], list[float]]:
    """Contrast query keys against all active user slots.

    The model's hard top-k index itself is nondifferentiable.  This auxiliary
    loss uses the same normalized query/key score before top-k and supplies a
    gradient to both semantic projections.
    """

    batch = len(examples)
    hidden = model.token_embedding(query_round.input_ids)
    query, _, _ = model._retrieval_boundary(
        hidden, query_round.attention_mask, query_round.roles
    )
    projected_query = F.normalize(
        model.memory_reader.query_projection(query), dim=-1, eps=1e-6
    )
    keys = F.normalize(state.memory.keys, dim=-1, eps=1e-6)
    scores = torch.einsum("bd,bmd->bm", projected_query, keys)
    active = state.memory.active > model.config.active_threshold
    losses: list[torch.Tensor] = []
    target_slots: list[int] = []
    target_scores: list[float] = []
    for row, example in enumerate(examples):
        target = payload_slot(
            state.index_select(torch.tensor([row], device=state.next_position.device)),
            example.fact_payload,
            active_threshold=model.config.active_threshold,
            relaxed=True,
        )
        if target is None:
            continue
        row_scores = scores[row].masked_fill(~active[row], -1.0e4)
        losses.append(F.cross_entropy(row_scores[None], torch.tensor([target], device=row_scores.device)))
        target_slots.append(target)
        target_scores.append(float(scores[row, target].detach()))
    if not losses:
        return scores.sum() * 0.0, target_slots, target_scores
    return torch.stack(losses).mean(), target_slots, target_scores


def candidate_key_alignment_loss(
    model: AdaptiveFactMemoryLM,
    fact_output,
    query_round: CollatedRound,
) -> torch.Tensor:
    """Align the query key with the candidate whose labelled start is zero.

    This bypasses recurrent-state BPTT while still training the fact-key
    projection.  The candidate is selected from the extractor output, not an
    oracle slot, and the loss is only used during the supervised curriculum.
    """

    hidden = model.token_embedding(query_round.input_ids)
    query, _, _ = model._retrieval_boundary(
        hidden, query_round.attention_mask, query_round.roles
    )
    query_key = F.normalize(model.memory_reader.query_projection(query), dim=-1, eps=1e-6)
    candidate_keys = F.normalize(
        fact_output.supervision["candidate_keys"], dim=-1, eps=1e-6
    )
    candidate_starts = fact_output.supervision["candidate_starts"]
    candidate_valid = fact_output.diagnostics["candidate_payload_mask"].any(dim=-1)
    terms: list[torch.Tensor] = []
    for row in range(query_key.shape[0]):
        matches = torch.nonzero(
            (candidate_starts[row] == 0) & candidate_valid[row],
            as_tuple=False,
        ).flatten()
        if matches.numel() == 0:
            continue
        target = int(matches[0])
        scores = torch.einsum("kd,d->k", candidate_keys[row], query_key[row])
        terms.append(
            F.cross_entropy(
                scores[None],
                torch.tensor([target], device=scores.device),
            )
        )
    if not terms:
        return query_key.sum() * 0.0
    return torch.stack(terms).mean()


def auxiliary_losses(
    model: AdaptiveFactMemoryLM,
    output,
    collated: CollatedRound,
    *,
    fact_round: bool,
) -> dict[str, torch.Tensor]:
    supervision = output.supervision
    user_mask = collated.attention_mask & (
        collated.roles == int(SourceRole.USER)
    )
    if fact_round:
        # The positive class is rare even in this compact fact sentence.
        start_loss = F.binary_cross_entropy_with_logits(
            supervision["fact_start_logits"][user_mask],
            collated.fact_start_targets[user_mask].to(torch.float32),
            pos_weight=torch.tensor(4.0, device=output.logits.device),
        )
        write_loss = F.binary_cross_entropy_with_logits(
            supervision["fact_token_write_logits"][user_mask],
            collated.fact_token_write_targets[user_mask].to(torch.float32),
            pos_weight=torch.tensor(4.0, device=output.logits.device),
        )
        length_terms: list[torch.Tensor] = []
        for row, targets in enumerate(collated.fact_length_targets):
            for start, length in targets:
                length_terms.append(
                    F.cross_entropy(
                        supervision["fact_length_logits"][row, start][None],
                        torch.tensor([length - 1], device=output.logits.device),
                    )
                )
        length_loss = (
            torch.stack(length_terms).mean()
            if length_terms
            else output.logits.sum() * 0.0
        )
    else:
        # Query tokens should not become persistent facts.  This is a weak
        # negative term because query commit is disabled in this curriculum.
        start_loss = F.binary_cross_entropy_with_logits(
            supervision["fact_start_logits"][user_mask],
            torch.zeros_like(supervision["fact_start_logits"][user_mask]),
        ) if bool(user_mask.any()) else output.logits.sum() * 0.0
        write_loss = F.binary_cross_entropy_with_logits(
            supervision["fact_token_write_logits"][user_mask],
            torch.zeros_like(supervision["fact_token_write_logits"][user_mask]),
        ) if bool(user_mask.any()) else output.logits.sum() * 0.0
        length_loss = output.logits.sum() * 0.0
    lm_loss = masked_loss_sum(
        output.logits,
        collated.input_ids,
        collated.answer_target_mask,
    ) / collated.answer_target_mask.sum().clamp_min(1)
    return {
        "lm": lm_loss,
        "start": start_loss,
        "length": length_loss,
        "write": write_loss,
    }


def training_batch_loss(
    model: AdaptiveFactMemoryLM,
    examples: list[MemoryExample],
    rounds: list[CollatedRound],
    device: torch.device,
    memory_policy: str = "gated",
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = len(examples)
    state = model.initial_state(
        batch_size,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    round_outputs = []
    losses: dict[str, torch.Tensor] = {}
    state_after_distractor = None
    for round_index, collated in enumerate(rounds):
        output = model.forward_round(
            collated.input_ids,
            state=state,
            attention_mask=collated.attention_mask,
            source_roles=collated.roles,
            commit=(round_index < 2) and memory_policy != "none",
            hard_memory=False,
        )
        state = output.state
        round_outputs.append(output)
        if memory_policy == "none":
            answer_count = collated.answer_target_mask.sum().clamp_min(1)
            current = {
                "lm": masked_loss_sum(
                    output.logits,
                    collated.input_ids,
                    collated.answer_target_mask,
                )
                / answer_count,
            }
        else:
            current = auxiliary_losses(
                model,
                output,
                collated,
                fact_round=round_index == 0,
            )
        for name, value in current.items():
            losses[name] = losses.get(name, value * 0.0) + value
        if round_index in (0, 1):
            # Backpropagating through recurrent local-KV and memory updates
            # across a long distractor creates extremely large Jacobians in
            # this first prototype.  Truncate at each round boundary: the
            # fact/key/write objectives train round A, the distractor losses
            # train round B, and the query losses train the reader/fusion.
            # This is the intended stable TBPTT boundary for long dialogue.
            if round_index == 1:
                state_after_distractor = state
            state = state.detach()

    if memory_policy == "none":
        total = losses["lm"]
        scalar = {
            name: float(value.detach().cpu()) for name, value in losses.items()
        }
        scalar.update(
            {
                "total": float(total.detach().cpu()),
                "active_slots": 0.0,
                "read_supervised_rows": 0.0,
                "read_target_score": 0.0,
            }
        )
        return total, scalar

    # The answer round is the only one requiring long-range retrieval.  Its
    # contrastive reader loss uses the state immediately before that round.
    if state_after_distractor is None:  # pragma: no cover - fixed 3-round generator
        raise RuntimeError("structured example did not produce a distractor round")
    state_before_query = state_after_distractor
    query_state = state_after_distractor.detach()
    # The query forward was intentionally run from ``query_state`` above.  A
    # separate local variable is retained for the differentiable controller
    # losses on the pre-detach state.
    read_loss, target_slots, target_scores = all_memory_read_loss(
        model, query_state, rounds[2], examples
    )
    key_align_loss = candidate_key_alignment_loss(
        model, round_outputs[0], rounds[2]
    )
    retention = model.memory_controller.retention_scores(state_before_query.memory)
    retention_terms: list[torch.Tensor] = []
    for row, example in enumerate(examples):
        row_state = state_before_query.index_select(
            torch.tensor([row], device=device)
        )
        slot = payload_slot(
            row_state,
            example.fact_payload,
            active_threshold=model.config.active_threshold,
            relaxed=True,
        )
        if slot is not None:
            # Positive target for the known fact; discourage retaining active
            # accidental spans at this first-stage one-fact setting.
            active = row_state.memory.active[0] > model.config.active_threshold
            if bool(active.any()):
                target = torch.zeros_like(retention[row])
                target[slot] = 1.0
                # ``retention_scores`` already applies sigmoid; BCE itself is
                # intentionally evaluated outside CUDA autocast.
                with torch.autocast(
                    device_type=device.type,
                    enabled=False,
                ):
                    retention_terms.append(
                        F.binary_cross_entropy(
                            retention[row][active].float(),
                            target[active].float(),
                        )
                    )
    retention_loss = (
        torch.stack(retention_terms).mean()
        if retention_terms
        else state_before_query.memory.active.sum() * 0.0
    )
    active_count = state_before_query.memory.active.sum(dim=1)
    budget_loss = F.relu(active_count - 1.0).mean()
    losses["read"] = read_loss
    losses["key_align"] = key_align_loss
    losses["retention"] = retention_loss
    losses["budget"] = budget_loss

    if memory_policy == "fixed":
        # Keep span extraction and semantic addressing trainable, but remove
        # losses for gate heads whose decisions are intentionally fixed.
        total = (
            losses["lm"]
            + 0.75 * losses["start"]
            + 0.25 * losses["length"]
            + 1.0 * losses["read"]
            + 0.50 * losses["key_align"]
        )
    else:
        total = (
            losses["lm"]
            + 0.75 * losses["start"]
            + 0.25 * losses["length"]
            + 0.75 * losses["write"]
            + 1.0 * losses["read"]
            + 0.50 * losses["key_align"]
            + 0.10 * losses["retention"]
            + 0.01 * losses["budget"]
        )
    scalar = {
        name: float(value.detach().cpu()) for name, value in losses.items()
    }
    scalar.update(
        {
            "total": float(total.detach().cpu()),
            "active_slots": float(state_before_query.memory.active.detach().sum(dim=1).mean().cpu()),
            "read_supervised_rows": float(len(target_slots)),
            "read_target_score": statistics.fmean(target_scores) if target_scores else 0.0,
        }
    )
    return total, scalar


def _round_tensors(
    example: MemoryExample,
    round_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample = example.rounds[round_index]
    ids = torch.tensor(sample.input_ids, dtype=torch.long, device=device)[None]
    roles = torch.tensor(sample.roles, dtype=torch.long, device=device)[None]
    mask = torch.ones_like(ids, dtype=torch.bool)
    return ids, roles, mask


@torch.no_grad()
def generate_one(
    model: AdaptiveFactMemoryLM,
    example: MemoryExample,
    device: torch.device,
    *,
    max_answer_tokens: int = 8,
) -> dict[str, Any]:
    model.eval()
    state = None
    for round_index in (0, 1):
        ids, roles, mask = _round_tensors(example, round_index, device)
        output = model.forward_round(
            ids,
            state=state,
            attention_mask=mask,
            source_roles=roles,
            commit=True,
            hard_memory=True,
        )
        state = output.state

    query_sample = example.rounds[2]
    query_length = int(query_sample.answer_start or 0)
    query_ids = torch.tensor(query_sample.input_ids[:query_length], device=device)[None]
    query_roles = torch.full_like(query_ids, int(SourceRole.USER))
    query_mask = torch.ones_like(query_ids, dtype=torch.bool)
    before_query = state
    query_output = model.forward_round(
        query_ids,
        state=state,
        attention_mask=query_mask,
        source_roles=query_roles,
        commit=False,
        hard_memory=True,
    )
    state = query_output.state
    generated: list[int] = []
    current = query_output.logits[:, -1].argmax(dim=-1)
    for index in range(max_answer_tokens):
        token = int(current[0])
        generated.append(token)
        token_ids = current[:, None]
        token_roles = torch.full_like(token_ids, int(SourceRole.ASSISTANT))
        token_mask = torch.ones_like(token_ids, dtype=torch.bool)
        continuation = model.forward_round(
            token_ids,
            state=state,
            attention_mask=token_mask,
            source_roles=token_roles,
            commit=index == max_answer_tokens - 1,
            hard_memory=True,
        )
        state = continuation.state
        if token == model.config.pad_token_id or token == 50256:
            break
        current = continuation.logits[:, -1].argmax(dim=-1)

    target = list(example.answer_ids)
    if target and target[-1] == 50256:
        target = target[:-1]
    exact = generated[: len(target)] == target
    target_slot = payload_slot(
        before_query,
        example.fact_payload,
        active_threshold=model.config.active_threshold,
    )
    read_indices = query_output.diagnostics["read_indices"][0].tolist()
    read_valid = query_output.diagnostics["read_valid"][0].tolist()
    read_hit = target_slot is not None and any(
        valid and int(index) == target_slot for index, valid in zip(read_indices, read_valid)
    )
    return {
        "memory_policy": model.memory_policy,
        "exact_match": bool(exact),
        "generated_ids": generated,
        "target_ids": target,
        "delay_segments": example.delay_segments,
        "target_slot": target_slot,
        "read_hit": bool(read_hit),
        "read_valid_count": int(sum(bool(value) for value in read_valid)),
        "active_slots_after_distractor": int(
            (before_query.memory.active[0] > model.config.active_threshold).sum()
        ),
        "state_bytes_after_distractor": int(model.state_bytes(before_query)),
        "effective_state_bytes_after_distractor": int(
            model.effective_state_bytes(before_query)
        ),
    }


@torch.no_grad()
def evaluate_memory(
    model: AdaptiveFactMemoryLM,
    generator: SyntheticMemoryGenerator,
    device: torch.device,
    *,
    examples_per_delay: int,
    start_index: int,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for delay in generator.delays:
        for offset in range(examples_per_delay):
            example = generator.make(
                start_index + delay * 10_000 + offset,
                delay_segments=delay,
            )
            rows.append(generate_one(model, example, device))
    by_delay: dict[str, dict[str, Any]] = {}
    for delay in generator.delays:
        subset = [row for row in rows if row["delay_segments"] == delay]
        by_delay[str(delay)] = {
            "examples": len(subset),
            "exact_match_accuracy": statistics.fmean(
                float(row["exact_match"]) for row in subset
            ) if subset else 0.0,
            "read_hit_rate": statistics.fmean(
                float(row["read_hit"]) for row in subset
            ) if subset else 0.0,
            "mean_active_slots": statistics.fmean(
                row["active_slots_after_distractor"] for row in subset
            ) if subset else 0.0,
            "mean_state_bytes": statistics.fmean(
                row["state_bytes_after_distractor"] for row in subset
            ) if subset else 0.0,
            "mean_effective_state_bytes": statistics.fmean(
                row["effective_state_bytes_after_distractor"] for row in subset
            ) if subset else 0.0,
        }
    return {
        "examples_per_delay": examples_per_delay,
        "by_delay": by_delay,
        "overall_exact_match_accuracy": statistics.fmean(
            float(row["exact_match"]) for row in rows
        ) if rows else 0.0,
        "overall_read_hit_rate": statistics.fmean(
            float(row["read_hit"]) for row in rows
        ) if rows else 0.0,
        "rows": rows,
    }


def run_training(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_all(args.seed)
    if not FILLER_PATH.exists():
        required = 100_000_257
        protocol.build_token_cache("train", required)
    train_generator = SyntheticMemoryGenerator(
        TOKENIZER_DIR,
        FILLER_PATH,
        context=args.context,
        segment_length=128,
        split="train",
        seed=args.seed,
        delay_buckets=(1, 2, 4, 8),
    )
    valid_generator = SyntheticMemoryGenerator(
        TOKENIZER_DIR,
        FILLER_PATH,
        context=args.context,
        segment_length=128,
        split="validation",
        seed=args.seed + 10_000,
        delay_buckets=(1, 2, 4, 8, 16),
    )
    model = load_stage1_model(
        device,
        training=True,
        memory_policy=args.memory_policy,
    )
    model.train()
    optimizer = torch.optim.AdamW(
        model_parameter_groups(model),
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    resolved = {
        "protocol_version": "adaptive_fact_memory_structured_v1",
        "stage": "stage2_structured_memory_training",
        "run_id": args.run_id,
        "seed": args.seed,
        "memory_policy": args.memory_policy,
        "parent_checkpoint": str(BACKBONE_CHECKPOINT),
        "parent_checkpoint_sha256": sha256_file(BACKBONE_CHECKPOINT),
        "architecture": {
            "config": {
                key: value
                for key, value in vars(model.config).items()
                if not key.startswith("_")
            },
            "tiny_lm_backbone": "6 layers / hidden 384 / 8 heads / FFN 1536",
            "local_kv": "4 sinks + 124 recent (128 unique keys)",
        },
        "data": {
            "task": "early_fact -> long_system_tinystories_distractor -> return_query",
            "train_split": "synthetic_train_combinations",
            "validation_split": "synthetic_validation_combinations",
            "tiny_stories_filler_cache": str(FILLER_PATH),
            "delays_train": [1, 2, 4, 8],
            "delays_eval": [1, 2, 4, 8, 16],
            "answer_metric": "exact_match_before_EOS",
        },
        "optimization": {
            "memory_learning_rate": 3.0e-4,
            "backbone_learning_rate": 3.0e-5,
            "optimizer": "AdamW",
            "gradient_clip_norm": 1.0,
            "hard_memory": False,
            "write_threshold_curriculum": 0.1,
            "retention_threshold_curriculum": 0.1,
        },
        "request": vars(args),
        "environment": environment_record(device),
        "code_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__),
                ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "model.py",
                ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "memory.py",
                Path(__file__).with_name("synthetic_memory.py"),
            )
            if path.exists()
        },
    }
    resolved["config_hash"] = stable_hash(resolved)
    atomic_json(run_dir / "config.resolved.json", resolved)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    best_loss = math.inf
    best_state_cpu: dict[str, torch.Tensor] | None = None
    step_records: list[dict[str, Any]] = []
    status = "ok"
    failure: str | None = None
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            aggregate: dict[str, float] = {}
            step_started = time.perf_counter()
            try:
                for accumulation in range(args.gradient_accumulation_steps):
                    first = ((step - 1) * args.gradient_accumulation_steps + accumulation) * args.batch_size
                    indices = range(first, first + args.batch_size)
                    examples, rounds = train_generator.batch(indices, device)
                    with autocast_context(device):
                        loss, scalars = training_batch_loss(
                            model,
                            examples,
                            rounds,
                            device,
                            memory_policy=args.memory_policy,
                        )
                    (loss / args.gradient_accumulation_steps).backward()
                    for name, value in scalars.items():
                        aggregate[name] = aggregate.get(name, 0.0) + value / args.gradient_accumulation_steps
                    del loss, examples, rounds
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm: {grad_norm}")
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except torch.cuda.OutOfMemoryError as exc:
                status = "oom"
                failure = repr(exc)
                break
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                failure = repr(exc)
                break

            elapsed_step = time.perf_counter() - step_started
            aggregate.update(
                {
                    "event": "train_step",
                    "step": step,
                    "gradient_norm": grad_norm,
                    "step_seconds": elapsed_step,
                    "examples_seen": step * args.batch_size * args.gradient_accumulation_steps,
                }
            )
            if aggregate.get("total", math.inf) < best_loss:
                best_loss = aggregate["total"]
                best_state_cpu = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            metrics_file.write(json.dumps(aggregate) + "\n")
            metrics_file.flush()
            step_records.append(aggregate)

            if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config_hash": resolved["config_hash"],
                    },
                    run_dir / f"checkpoint.step_{step}.pt",
                )

    if status == "ok":
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": len(step_records),
                "config_hash": resolved["config_hash"],
            },
            run_dir / "checkpoint.final.pt",
        )
    elapsed = time.perf_counter() - started

    # Evaluate both the curriculum threshold and the default hard threshold.
    # Keeping the two records prevents an apparent recall gain from being
    # hidden by the training-only threshold choice.
    memory_eval = None
    default_eval = None
    if status == "ok":
        model.eval()
        memory_eval = evaluate_memory(
            model,
            valid_generator,
            device,
            examples_per_delay=args.eval_examples_per_delay,
            start_index=1_000_000,
        )
        default_model = load_stage1_model(
            device,
            training=False,
            memory_policy=args.memory_policy,
        )
        default_model.load_state_dict(model.state_dict())
        default_model.eval()
        default_eval = evaluate_memory(
            default_model,
            valid_generator,
            device,
            examples_per_delay=args.eval_examples_per_delay,
            start_index=1_000_000,
        )
        model = default_model

    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    summary = {
        "protocol_version": "adaptive_fact_memory_structured_v1",
        "stage": "stage2_structured_memory_training",
        "run_id": args.run_id,
        "seed": args.seed,
        "memory_policy": args.memory_policy,
        "status": status,
        "failure": failure,
        "steps_completed": len(step_records),
        "elapsed_training_seconds": elapsed,
        "mean_examples_per_second": (
            args.batch_size * args.gradient_accumulation_steps * len(step_records)
            / max(elapsed, 1e-9)
        ),
        "best_training_total_loss": None if best_state_cpu is None else best_loss,
        "last_training": step_records[-1] if step_records else None,
        "memory_eval_curriculum_threshold": memory_eval,
        "memory_eval_default_threshold": default_eval,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "config_hash": resolved["config_hash"],
        "run_dir": str(run_dir),
    }
    atomic_json(run_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--memory-policy",
        choices=AdaptiveFactMemoryLM.VALID_MEMORY_POLICIES,
        default="gated",
        help="gated proposal, fixed always-on memory, or SWA-only (none)",
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--eval-examples-per-delay", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cuda()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    summary = run_training(args, device)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
