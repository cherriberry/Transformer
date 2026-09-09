"""Train/evaluate the selective-memory capacity stress pilot.

This is a separate protocol from ``train_structured_memory.py``.  It keeps
the original results immutable and introduces user-role temporary facts that
compete with one durable fact for a deliberately small memory.  The primary
comparison is gated memory versus a deterministic fixed-LRU baseline; ``none``
is the SWA+sink-only floor.
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


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "adaptive_gated_fact_memory" / "src"
SCRIPT_DIR = ROOT / "adaptive_gated_fact_memory" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from adaptive_fact_memory import (  # noqa: E402
    AdaptiveFactMemoryLM,
    FactMemoryConfig,
    SourceRole,
)
from adaptive_gated_fact_memory.scripts import train_structured_memory as base  # noqa: E402
from adaptive_gated_fact_memory.scripts.synthetic_memory_stress import (  # noqa: E402
    SelectiveMemoryStressGenerator,
    StressMemoryExample,
)


DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
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
        "ADAPTIVE_MEMORY_STRESS_RUNS_DIR",
        str(DATA_DISK / "runs/adaptive_gated_fact_memory_stress"),
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
    record: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record.update(
            {
                "gpu": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return record


def load_stress_model(
    device: torch.device,
    *,
    training: bool,
    memory_policy: str,
    memory_slots: int,
    memory_read_top_k: int,
    fusion_gate_override: float | None = None,
) -> AdaptiveFactMemoryLM:
    config = FactMemoryConfig(
        memory_slots=memory_slots,
        memory_read_top_k=memory_read_top_k,
        # One atomic candidate per user turn makes the capacity accounting
        # auditable: one durable message consumes one slot and each temporary
        # message can consume at most one slot.  With four candidates, four
        # overlapping spans from one sentence would confound the stress axis.
        max_write_candidates=1,
        # A high merge threshold makes distinct temporary facts compete for
        # slots instead of silently collapsing into one another.
        merge_threshold=0.999,
        write_threshold=0.1 if training else 0.5,
        retention_threshold=0.1 if training else 0.5,
    )
    model = AdaptiveFactMemoryLM(
        config,
        memory_policy=memory_policy,
        fusion_gate_override=fusion_gate_override,
    ).to(device)
    checkpoint = torch.load(BACKBONE_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    # The stress protocol does not study assistant-state writes.
    with torch.no_grad():
        model.memory_controller.assistant_write_head.bias.fill_(-10.0)
    base.configure_memory_policy(model, memory_policy)
    return model


def _is_fixed(memory_policy: str) -> bool:
    return memory_policy in {"fixed", "fixed_lru"}


def _stress_auxiliary_losses(
    model: AdaptiveFactMemoryLM,
    output,
    collated,
    *,
    durable_round: bool,
) -> dict[str, torch.Tensor]:
    """Use the original span objectives, with negative user-noise rounds."""

    # Reuse the audited implementation so the stress and original protocols
    # have identical token-level loss semantics.
    return base.auxiliary_losses(
        model,
        output,
        collated,
        fact_round=durable_round,
    )


def stress_training_batch_loss(
    model: AdaptiveFactMemoryLM,
    examples: list[StressMemoryExample],
    rounds,
    device: torch.device,
    *,
    memory_policy: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train one variable-length durable/noise/delay/query conversation batch."""

    batch_size = len(examples)
    state = model.initial_state(
        batch_size,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    round_outputs = []
    durable_losses: dict[str, torch.Tensor] = {}
    negative_losses: dict[str, list[torch.Tensor]] = {"start": [], "write": []}
    state_before_query = None
    tracked_target_slots: list[int | None] = [None for _ in examples]
    retention_terms_over_time: list[torch.Tensor] = []
    query_index = len(rounds) - 1

    for round_index, collated in enumerate(rounds):
        kind = examples[0].round_kinds[round_index]
        if any(example.round_kinds[round_index] != kind for example in examples):
            raise ValueError("mixed round kinds in one stress batch")
        is_query = kind == "query"
        output = model.forward_round(
            collated.input_ids,
            state=state,
            attention_mask=collated.attention_mask,
            source_roles=collated.roles,
            commit=(not is_query) and memory_policy != "none",
            hard_memory=False,
        )
        round_outputs.append(output)
        state = output.state
        if memory_policy == "none":
            # Only the answer round has assistant targets in this protocol.
            count = collated.answer_target_mask.sum().clamp_min(1)
            value = base.masked_loss_sum(
                output.logits,
                collated.input_ids,
                collated.answer_target_mask,
            ) / count
            durable_losses["lm"] = durable_losses.get("lm", value * 0.0) + value
        else:
            current = _stress_auxiliary_losses(
                model,
                output,
                collated,
                durable_round=kind == "durable",
            )
            if kind == "durable":
                for name, value in current.items():
                    durable_losses[name] = durable_losses.get(name, value * 0.0) + value
            elif kind == "noise" or is_query:
                # Average over all negative user rounds rather than allowing a
                # larger noise bucket to change the overall loss scale.
                negative_losses["start"].append(current["start"])
                negative_losses["write"].append(current["write"])
                if is_query:
                    durable_losses["lm"] = durable_losses.get(
                        "lm", current["lm"] * 0.0
                    ) + current["lm"]

            # Keep a stable target-slot identity through the distractor.  The
            # original one-fact curriculum supervised retention only at its
            # final boundary; under many user-noise rounds that can let the
            # target disappear before the loss is evaluated.  Supervising the
            # known slot after each commit makes this stress test measure
            # retention rather than an accidental threshold mismatch.
            if kind == "durable":
                for row, example in enumerate(examples):
                    row_state = state.index_select(torch.tensor([row], device=device))
                    tracked_target_slots[row] = base.payload_slot(
                        row_state,
                        example.fact_payload,
                        active_threshold=model.config.active_threshold,
                        relaxed=True,
                    )
            # Retention is evaluated *before the next commit*.  Supervising
            # the differentiable post-write recurrent state across several
            # commits creates an unstable higher-order path through the
            # straight-through slot assignment.  Supervise the detached state
            # instead: gradients train the retention head at every boundary,
            # while write/key objectives train their own local round paths.
            if kind != "query":
                retention_state = state.detach()
                for row, example in enumerate(examples):
                    row_state = retention_state.index_select(
                        torch.tensor([row], device=device)
                    )
                    tracked_target_slots[row] = base.payload_slot(
                        row_state,
                        example.fact_payload,
                        active_threshold=model.config.active_threshold,
                        relaxed=True,
                    )
                retention_scores = model.memory_controller.retention_scores(
                    retention_state.memory
                )
                target = torch.zeros_like(retention_scores)
                supervised = torch.zeros_like(retention_scores, dtype=torch.bool)
                for row, slot in enumerate(tracked_target_slots):
                    if slot is None:
                        continue
                    target[row, slot] = 1.0
                    supervised[row] = True
                if bool(supervised.any()):
                    # Include the target slot and all currently active slots;
                    # non-target slots are explicitly discouraged from being
                    # retained in this single-durable-fact stress task.
                    active = state.memory.active > model.config.active_threshold
                    mask = active | torch.zeros_like(active)
                    for row, slot in enumerate(tracked_target_slots):
                        if slot is not None:
                            mask[row, slot] = True
                    with torch.autocast(device_type=device.type, enabled=False):
                        retention_terms_over_time.append(
                            F.binary_cross_entropy(
                                retention_scores[mask].float().clamp(1.0e-6, 1.0 - 1.0e-6),
                                target[mask].float(),
                            )
                        )

        if round_index == query_index - 1:
            state_before_query = state
        if not is_query:
            # Keep the same round-boundary TBPTT rule as the original stage.
            state = state.detach()

    if memory_policy == "none":
        total = durable_losses["lm"]
        scalar = {name: float(value.detach().cpu()) for name, value in durable_losses.items()}
        scalar.update(
            {
                "total": float(total.detach().cpu()),
                "active_slots": 0.0,
                "read_supervised_rows": 0.0,
                "read_target_score": 0.0,
            }
        )
        return total, scalar

    if state_before_query is None:
        raise RuntimeError("stress example has no pre-query state")
    query_state = state_before_query.detach()
    query_round = rounds[query_index]
    read_loss, read_target_slots, target_scores = base.all_memory_read_loss(
        model, query_state, query_round, examples
    )
    key_align_loss = base.candidate_key_alignment_loss(
        model, round_outputs[0], query_round
    )

    # The time-local terms above include the target even when a hard threshold
    # would otherwise deactivate it.  Retain the final-state term as a small
    # safety net for batches in which a slot was not identified early.
    retention = model.memory_controller.retention_scores(state_before_query.memory)
    retention_terms: list[torch.Tensor] = list(retention_terms_over_time)
    for row, example in enumerate(examples):
        slot = tracked_target_slots[row]
        if slot is None:
            row_state = state_before_query.index_select(torch.tensor([row], device=device))
            slot = base.payload_slot(
                row_state,
                example.fact_payload,
                active_threshold=model.config.active_threshold,
                relaxed=True,
            )
        if slot is not None:
            with torch.autocast(device_type=device.type, enabled=False):
                retention_terms.append(
                    F.binary_cross_entropy(
                        retention[row, slot].float().clamp(1.0e-6, 1.0 - 1.0e-6),
                        torch.ones_like(retention[row, slot]).float(),
                    )
                )
    retention_loss = (
        torch.stack(retention_terms).mean()
        if retention_terms
        else state_before_query.memory.active.sum() * 0.0
    )
    active_count = state_before_query.memory.active.sum(dim=1)
    budget_loss = F.relu(active_count - 1.0).mean()
    neg_start = (
        torch.stack(negative_losses["start"]).mean()
        if negative_losses["start"]
        else read_loss * 0.0
    )
    neg_write = (
        torch.stack(negative_losses["write"]).mean()
        if negative_losses["write"]
        else read_loss * 0.0
    )
    # The durable span loss is positive supervision; negative user rounds are
    # added separately so the write gate learns the durable/temporary cue.
    durable_losses["start"] = durable_losses.get("start", read_loss * 0.0) + neg_start
    durable_losses["write"] = durable_losses.get("write", read_loss * 0.0) + neg_write
    durable_losses.update(
        {
            "read": read_loss,
            "key_align": key_align_loss,
            "retention": retention_loss,
            "budget": budget_loss,
        }
    )

    if _is_fixed(memory_policy):
        total = (
            durable_losses.get("lm", read_loss * 0.0)
            + 0.75 * durable_losses["start"]
            + 0.25 * durable_losses["length"]
            + 1.0 * read_loss
            + 0.50 * key_align_loss
        )
    else:
        total = (
            durable_losses.get("lm", read_loss * 0.0)
            + 0.75 * durable_losses["start"]
            + 0.25 * durable_losses["length"]
            + 0.75 * durable_losses["write"]
            + 1.0 * read_loss
            + 0.50 * key_align_loss
            + 0.10 * retention_loss
            + 0.01 * budget_loss
        )
    scalar = {name: float(value.detach().cpu()) for name, value in durable_losses.items()}
    scalar.update(
        {
            "total": float(total.detach().cpu()),
            "active_slots": float(active_count.detach().mean().cpu()),
            "read_supervised_rows": float(len(read_target_slots)),
            "read_target_score": statistics.fmean(target_scores) if target_scores else 0.0,
        }
    )
    return total, scalar


def _round_tensor(example: StressMemoryExample, index: int, device: torch.device):
    sample = example.rounds[index]
    ids = torch.tensor(sample.input_ids, dtype=torch.long, device=device)[None]
    roles = torch.tensor(sample.roles, dtype=torch.long, device=device)[None]
    mask = torch.ones_like(ids, dtype=torch.bool)
    return ids, roles, mask


def _active_user_slot_bytes(model: AdaptiveFactMemoryLM, state) -> tuple[int, int]:
    """Return (active logical bytes, full user-slot bytes).

    ``state_bytes`` intentionally includes preallocated tensors.  This helper
    reports the additional logical quantity needed to discuss dynamic
    compaction; it is not a claim about current allocator behavior.
    """

    names = (
        "keys", "values", "active", "payload_ids", "payload_mask", "age",
        "last_access", "access_count", "confidence", "conflict", "source_role",
    )
    per_slot = 0
    for name in names:
        tensor = getattr(state.memory, name)
        per_slot += tensor[0, 0].numel() * tensor.element_size()
    active = int((state.memory.active[0] > model.config.active_threshold).sum())
    return active * per_slot, model.config.memory_slots * per_slot


@torch.no_grad()
def generate_stress_one(
    model: AdaptiveFactMemoryLM,
    example: StressMemoryExample,
    device: torch.device,
    *,
    max_answer_tokens: int = 8,
) -> dict[str, Any]:
    model.eval()
    state = None
    cumulative_noise_accepts = 0
    cumulative_noise_candidates = 0
    survival_curve: list[bool] = []
    prequery_diagnostics: list[dict[str, Any]] = []

    for round_index, kind in enumerate(example.round_kinds[:-1]):
        ids, roles, mask = _round_tensor(example, round_index, device)
        output = model.forward_round(
            ids,
            state=state,
            attention_mask=mask,
            source_roles=roles,
            commit=model.memory_policy != "none",
            hard_memory=True,
        )
        state = output.state
        diagnostic = output.diagnostics
        if kind == "noise" and model.memory_policy != "none":
            accepted = diagnostic.get("write_accepted")
            valid = diagnostic.get("candidate_payload_mask")
            if accepted is not None:
                cumulative_noise_accepts += int(accepted.sum())
            if valid is not None:
                cumulative_noise_candidates += int(valid.any(dim=-1).sum())
        target_slot = base.payload_slot(
            state,
            example.fact_payload,
            active_threshold=model.config.active_threshold,
        )
        survival_curve.append(target_slot is not None)
        prequery_diagnostics.append(
            {
                "kind": kind,
                "active_slots": int(
                    (state.memory.active[0] > model.config.active_threshold).sum()
                ),
                "target_survived": target_slot is not None,
            }
        )

    before_query = state
    query_sample = example.rounds[-1]
    query_length = int(query_sample.answer_start or 0)
    query_ids = torch.tensor(query_sample.input_ids[:query_length], device=device)[None]
    query_roles = torch.full_like(query_ids, int(SourceRole.USER))
    query_mask = torch.ones_like(query_ids, dtype=torch.bool)

    target_slot = base.payload_slot(
        before_query,
        example.fact_payload,
        active_threshold=model.config.active_threshold,
    )
    # Compute rank among all active slots, not only the reader's top-k output.
    if model.memory_policy == "none":
        target_rank = None
        target_mrr = 0.0
        target_top1 = False
    else:
        hidden = model.token_embedding(query_ids)
        turn_query, _, _ = model._retrieval_boundary(
            hidden, query_mask, query_roles
        )
        query_key = F.normalize(
            model.memory_reader.query_projection(turn_query), dim=-1, eps=1e-6
        )
        keys = F.normalize(before_query.memory.keys, dim=-1, eps=1e-6)
        scores = torch.einsum("bd,bmd->bm", query_key, keys)[0]
        active = before_query.memory.active[0] > model.config.active_threshold
        if target_slot is None:
            target_rank = None
            target_mrr = 0.0
            target_top1 = False
        else:
            target_score = scores[target_slot]
            target_rank = 1 + int(((scores > target_score) & active).sum())
            target_mrr = 1.0 / target_rank
            target_top1 = target_rank == 1

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
        continuation = model.forward_round(
            token_ids,
            state=state,
            attention_mask=torch.ones_like(token_ids, dtype=torch.bool),
            source_roles=token_roles,
            commit=index == max_answer_tokens - 1,
            hard_memory=True,
        )
        state = continuation.state
        if token in {model.config.pad_token_id, 50_256}:
            break
        current = continuation.logits[:, -1].argmax(dim=-1)

    target = list(example.answer_ids)
    if target and target[-1] == 50_256:
        target = target[:-1]
    exact = generated[: len(target)] == target
    read_indices = query_output.diagnostics.get("read_indices")
    read_valid = query_output.diagnostics.get("read_valid")
    read_hit = False
    if target_slot is not None and read_indices is not None and read_valid is not None:
        read_hit = any(
            bool(valid) and int(index) == target_slot
            for index, valid in zip(read_indices[0].tolist(), read_valid[0].tolist())
        )
    active_bytes, full_slot_bytes = _active_user_slot_bytes(model, before_query)
    return {
        "memory_policy": model.memory_policy,
        "noise_rounds": example.noise_rounds,
        "delay_segments": example.delay_segments,
        "exact_match": bool(exact),
        "generated_ids": generated,
        "target_ids": target,
        "target_slot": target_slot,
        "target_survived": target_slot is not None,
        "target_rank": target_rank,
        "target_mrr": target_mrr,
        "target_top1": target_top1,
        "read_hit_at_k": bool(read_hit),
        "read_valid_count": int(read_valid.sum().item()) if read_valid is not None else 0,
        "active_slots": int(
            (before_query.memory.active[0] > model.config.active_threshold).sum()
        ),
        "active_user_slot_bytes": active_bytes,
        "full_user_slot_bytes": full_slot_bytes,
        "state_bytes": int(model.state_bytes(before_query)),
        "effective_state_bytes": int(model.effective_state_bytes(before_query)),
        "noise_candidates_seen": cumulative_noise_candidates,
        "noise_accepts": cumulative_noise_accepts,
        "noise_accept_rate": (
            cumulative_noise_accepts / cumulative_noise_candidates
            if cumulative_noise_candidates
            else 0.0
        ),
        "survival_curve": survival_curve,
        "prequery_diagnostics": prequery_diagnostics,
    }


@torch.no_grad()
def evaluate_stress(
    model: AdaptiveFactMemoryLM,
    generator: SelectiveMemoryStressGenerator,
    device: torch.device,
    *,
    examples_per_condition: int,
    noise_buckets: tuple[int, ...],
    delay_buckets: tuple[int, ...],
    start_index: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for noise_index, noise in enumerate(noise_buckets):
        for delay_index, delay in enumerate(delay_buckets):
            for offset in range(examples_per_condition):
                index = start_index + noise_index * 100_000 + delay_index * 10_000 + offset
                example = generator.make(
                    index,
                    delay_segments=delay,
                    noise_rounds=noise,
                )
                rows.append(generate_stress_one(model, example, device))

    by_condition: dict[str, dict[str, Any]] = {}
    for noise in noise_buckets:
        for delay in delay_buckets:
            subset = [
                row
                for row in rows
                if row["noise_rounds"] == noise and row["delay_segments"] == delay
            ]
            mean = lambda key: statistics.fmean(float(row[key]) for row in subset) if subset else 0.0
            by_condition[f"noise_{noise}_delay_{delay}"] = {
                "noise_rounds": noise,
                "delay_segments": delay,
                "examples": len(subset),
                "exact_match": mean("exact_match"),
                "target_survival": mean("target_survived"),
                "target_top1": mean("target_top1"),
                "target_mrr": mean("target_mrr"),
                "read_hit_at_k": mean("read_hit_at_k"),
                "mean_active_slots": mean("active_slots"),
                "mean_active_user_slot_bytes": mean("active_user_slot_bytes"),
                "mean_state_bytes": mean("state_bytes"),
                "mean_noise_accept_rate": mean("noise_accept_rate"),
            }
    return {
        "examples_per_condition": examples_per_condition,
        "noise_buckets": list(noise_buckets),
        "delay_buckets": list(delay_buckets),
        "by_condition": by_condition,
        "overall_exact_match": statistics.fmean(float(row["exact_match"]) for row in rows)
        if rows
        else 0.0,
        "overall_target_survival": statistics.fmean(float(row["target_survived"]) for row in rows)
        if rows
        else 0.0,
        "overall_target_top1": statistics.fmean(float(row["target_top1"]) for row in rows)
        if rows
        else 0.0,
        "overall_target_mrr": statistics.fmean(float(row["target_mrr"]) for row in rows)
        if rows
        else 0.0,
        "overall_read_hit_at_k": statistics.fmean(float(row["read_hit_at_k"]) for row in rows)
        if rows
        else 0.0,
        "rows": rows,
    }


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated non-empty list")
    return result


def run_training(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_all(args.seed)
    if not BACKBONE_CHECKPOINT.exists():
        raise FileNotFoundError(f"backbone checkpoint is missing: {BACKBONE_CHECKPOINT}")
    if not FILLER_PATH.exists():
        raise FileNotFoundError(f"TinyStories filler cache is missing: {FILLER_PATH}")

    run_dir = Path(args.runs_dir) / args.run_id
    if (run_dir / "summary.json").exists() or (run_dir / "checkpoint.final.pt").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_delays = args.train_delays
    train_noise = args.train_noise
    eval_delays = args.eval_delays
    eval_noise = args.eval_noise
    train_generator = SelectiveMemoryStressGenerator(
        TOKENIZER_DIR,
        FILLER_PATH,
        split="train",
        seed=args.seed,
        delay_buckets=train_delays,
        noise_buckets=train_noise,
        segment_length=args.segment_length,
        payload_tokens=12,
        answer_leading_space=args.answer_leading_space,
    )
    valid_generator = SelectiveMemoryStressGenerator(
        TOKENIZER_DIR,
        FILLER_PATH,
        split="validation",
        seed=args.seed + 10_000,
        delay_buckets=eval_delays,
        noise_buckets=eval_noise,
        segment_length=args.segment_length,
        payload_tokens=12,
        answer_leading_space=args.answer_leading_space,
    )
    model = load_stress_model(
        device,
        training=True,
        memory_policy=args.memory_policy,
        memory_slots=args.memory_slots,
        memory_read_top_k=args.memory_read_top_k,
        fusion_gate_override=args.fusion_gate_override,
    )
    model.train()
    optimizer = torch.optim.AdamW(
        base.model_parameter_groups(model),
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    resolved: dict[str, Any] = {
        "protocol_version": "adaptive_fact_memory_selective_stress_v2",
        "stage": "stage2_selective_memory_stress",
        "run_id": args.run_id,
        "seed": args.seed,
        "memory_policy": args.memory_policy,
        "parent_checkpoint": str(BACKBONE_CHECKPOINT),
        "parent_checkpoint_sha256": sha256_file(BACKBONE_CHECKPOINT),
        "architecture": {
            "layers": model.config.layers,
            "hidden_size": model.config.hidden_size,
            "heads": model.config.heads,
            "ffn_size": model.config.ffn_size,
            "memory_slots": model.config.memory_slots,
            "memory_read_top_k": model.config.memory_read_top_k,
            "max_write_candidates": model.config.max_write_candidates,
            "merge_threshold": model.config.merge_threshold,
            "local_kv": "4 sinks + 124 recent",
        },
        "data": {
            "task": "durable_user_fact -> user_temporary_fact_noise -> TinyStories_delay -> query",
            "durable_cue": "train templates and held-out validation paraphrase",
            "train_delays": list(train_delays),
            "train_noise_rounds": list(train_noise),
            "eval_delays": list(eval_delays),
            "eval_noise_rounds": list(eval_noise),
            "examples_per_condition": args.eval_examples_per_condition,
            "filler": str(FILLER_PATH),
        },
        "optimization": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "memory_learning_rate": 3.0e-4,
            "backbone_learning_rate": 3.0e-5,
            "optimizer": "AdamW",
            "hard_memory_training": False,
            "write_threshold_training": 0.1,
            "retention_threshold_training": 0.1,
            "fusion_gate_override": args.fusion_gate_override,
            "answer_leading_space": args.answer_leading_space,
        },
        "request": vars(args),
        "environment": environment_record(device),
        "code_hashes": {},
    }
    code_paths = (
        Path(__file__),
        Path(__file__).with_name("synthetic_memory_stress.py"),
        ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "model.py",
        ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "memory.py",
        ROOT / "adaptive_gated_fact_memory" / "scripts" / "train_structured_memory.py",
    )
    resolved["code_hashes"] = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths if path.exists()
    }
    resolved["config_hash"] = stable_hash(resolved)
    atomic_json(run_dir / "config.resolved.json", resolved)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best_loss = math.inf
    step_records: list[dict[str, Any]] = []
    status = "ok"
    failure: str | None = None
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            aggregate: dict[str, float] = {}
            step_started = time.perf_counter()
            try:
                for accumulation in range(args.gradient_accumulation_steps):
                    logical_index = (step - 1) * args.gradient_accumulation_steps + accumulation
                    first = logical_index * args.batch_size
                    noise = train_noise[logical_index % len(train_noise)]
                    delay = train_delays[(logical_index // len(train_noise)) % len(train_delays)]
                    examples, rounds = train_generator.batch(
                        range(first, first + args.batch_size),
                        device,
                        delay_segments=delay,
                        noise_rounds=noise,
                    )
                    with autocast_context(device):
                        loss, scalars = stress_training_batch_loss(
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
            best_loss = min(best_loss, aggregate.get("total", math.inf))
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

    evaluation = None
    if status == "ok" and not args.skip_eval:
        eval_model = load_stress_model(
            device,
            training=False,
            memory_policy=args.memory_policy,
            memory_slots=args.memory_slots,
            memory_read_top_k=args.memory_read_top_k,
            fusion_gate_override=args.fusion_gate_override,
        )
        eval_model.load_state_dict(model.state_dict())
        eval_model.eval()
        evaluation = evaluate_stress(
            eval_model,
            valid_generator,
            device,
            examples_per_condition=args.eval_examples_per_condition,
            noise_buckets=eval_noise,
            delay_buckets=eval_delays,
            start_index=1_000_000,
        )
        del eval_model

    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    summary = {
        "protocol_version": resolved["protocol_version"],
        "stage": resolved["stage"],
        "run_id": args.run_id,
        "seed": args.seed,
        "memory_policy": args.memory_policy,
        "status": status,
        "failure": failure,
        "steps_completed": len(step_records),
        "elapsed_training_seconds": elapsed,
        "mean_examples_per_second": (
            args.batch_size * args.gradient_accumulation_steps * len(step_records)
            / max(elapsed, 1.0e-9)
        ),
        "best_training_total_loss": None if not step_records else best_loss,
        "last_training": step_records[-1] if step_records else None,
        "evaluation": evaluation,
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
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--memory-policy",
        choices=AdaptiveFactMemoryLM.VALID_MEMORY_POLICIES,
        default="gated",
    )
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--memory-read-top-k", type=int, default=4)
    parser.add_argument(
        "--fusion-gate-override",
        type=float,
        default=None,
        help="Use a fixed memory-fusion gate during training and evaluation.",
    )
    parser.add_argument(
        "--answer-leading-space",
        action="store_true",
        help="Encode answer values with a leading space, matching fact payload tokenization.",
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--train-delays", type=parse_int_tuple, default=(1, 2, 4))
    parser.add_argument("--train-noise", type=parse_int_tuple, default=(0, 1, 2, 4))
    parser.add_argument("--eval-delays", type=parse_int_tuple, default=(1, 4, 16))
    parser.add_argument("--eval-noise", type=parse_int_tuple, default=(0, 2, 4, 8))
    parser.add_argument("--eval-examples-per-condition", type=int, default=24)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cuda()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    summary = run_training(args, device)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "run_id",
                    "memory_policy",
                    "status",
                    "steps_completed",
                    "elapsed_training_seconds",
                    "evaluation",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
                if key in summary
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
