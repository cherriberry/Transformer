"""M10 causal diagnostics for the trained Memformer colour task.

The script evaluates one final M9 checkpoint under state interventions.  It
does not train a probe or add a reconstruction/alignment objective: answers
always come from the tied ordinary vocabulary projection.
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
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "adaptive_gated_fact_memory" / "src"
SCRIPT_DIR = ROOT / "adaptive_gated_fact_memory" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from adaptive_fact_memory import MemformerDecoderLM, MemformerState  # noqa: E402
from adaptive_gated_fact_memory.scripts.synthetic_memory_stress import (  # noqa: E402
    SelectiveMemoryStressGenerator,
)
from train_color_m8 import (  # noqa: E402
    FILLER_PATH,
    PARENT_CHECKPOINT,
    TOKENIZER_DIR,
    autocast_context,
    configure_cuda,
    load_common_parent_into_memformer,
    load_parent_state,
)
from adaptive_fact_memory import FactMemoryConfig  # noqa: E402


DEFAULT_RUN = Path(
    "/root/autodl-tmp/26summerBDMI_transformer/runs/"
    "adaptive_gated_fact_memory_color_m9/m9_memformer_s17_noise0"
)
DEFAULT_OUTPUT = DEFAULT_RUN.parent / "m10_memformer_state_diagnostics_s17.json"


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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(run_dir: Path, device: torch.device) -> MemformerDecoderLM:
    checkpoint_path = run_dir / "checkpoint.final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing M9 checkpoint: {checkpoint_path}")
    parent_state, _ = load_parent_state()
    config = FactMemoryConfig()
    model = MemformerDecoderLM(
        config,
        memory_slots=64,
        segment_length=32,
        detach_memory=False,
    ).to(device)
    load_common_parent_into_memformer(model, parent_state)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(checkpoint_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def state_norm(state: MemformerState) -> float:
    values = [memory.detach().float().square().mean().sqrt() for memory in state.memories]
    return float(torch.stack(values).mean().cpu())


def state_max_abs(state: MemformerState) -> float:
    return max(float(memory.detach().float().abs().max().cpu()) for memory in state.memories)


def zero_memory(state: MemformerState) -> MemformerState:
    """Clear recurrent vectors while preserving each example's position."""

    result = MemformerState(
        memories=tuple(torch.zeros_like(memory) for memory in state.memories),
        next_position=state.next_position,
        segment_count=state.segment_count,
    )
    result._validate()
    return result


def swap_memory(state: MemformerState) -> MemformerState:
    """Exchange memory vectors but retain the query batch's own positions."""

    if state.batch_size != 2:
        raise ValueError("state swap requires a two-example batch")
    result = MemformerState(
        memories=tuple(memory.index_select(0, torch.tensor([1, 0], device=memory.device)) for memory in state.memories),
        next_position=state.next_position,
        segment_count=state.segment_count,
    )
    result._validate()
    return result


def last_slot_state(state: MemformerState) -> MemformerState:
    memories = []
    for memory in state.memories:
        mask = torch.zeros_like(memory)
        mask[..., -1:] = 1
        memories.append(memory * mask)
    result = MemformerState(
        memories=tuple(memories),
        next_position=state.next_position,
        segment_count=state.segment_count,
    )
    result._validate()
    return result


def scaled_state(state: MemformerState, scale: float) -> MemformerState:
    result = MemformerState(
        memories=tuple(memory * float(scale) for memory in state.memories),
        next_position=state.next_position,
        segment_count=state.segment_count,
    )
    result._validate()
    return result


def noisy_state(state: MemformerState, noise_scale: float, generator: torch.Generator) -> MemformerState:
    memories = []
    for memory in state.memories:
        std = memory.float().std().to(dtype=memory.dtype)
        noise = torch.randn(
            memory.shape,
            device=memory.device,
            dtype=memory.dtype,
            generator=generator,
        )
        memories.append(memory + noise * std * float(noise_scale))
    result = MemformerState(
        memories=tuple(memories),
        next_position=state.next_position,
        segment_count=state.segment_count,
    )
    result._validate()
    return result


def process_prefix(
    model: MemformerDecoderLM,
    rounds,
    device: torch.device,
    *,
    segment_length: int,
    reset_after_each_segment: bool = False,
) -> MemformerState:
    """Encode all pre-query rounds, optionally resetting every segment."""

    state: MemformerState | None = None
    for collated in rounds[:-1]:
        if not reset_after_each_segment:
            with autocast_context(device):
                output = model(
                    collated.input_ids,
                    state=state,
                    attention_mask=collated.attention_mask,
                    segment_length=segment_length,
                )
            state = output.state
            continue

        # Explicit chunks make the intervention a true per-segment reset,
        # including long filler rounds that contain several 32-token chunks.
        tokens = collated.input_ids.shape[1]
        for start in range(0, tokens, segment_length):
            end = min(tokens, start + segment_length)
            chunk_ids = collated.input_ids[:, start:end]
            chunk_mask = collated.attention_mask[:, start:end]
            if state is None:
                state = model.initial_state(
                    chunk_ids.shape[0],
                    device=device,
                    dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                )
            with autocast_context(device):
                output = model.forward_segment(
                    chunk_ids,
                    state=state,
                    attention_mask=chunk_mask,
                )
            state = zero_memory(output.state)

    if state is None:
        raise RuntimeError("prefix did not contain any rounds")
    return state


def row_metrics(output, collated) -> list[dict[str, float]]:
    logits = output.logits.float()
    rows: list[dict[str, float]] = []
    for row in range(collated.input_ids.shape[0]):
        positions = torch.nonzero(
            collated.answer_target_mask[row, :-1], as_tuple=False
        ).flatten()
        if positions.numel() == 0:
            raise RuntimeError("query has no answer target")
        targets = collated.input_ids[row, positions + 1]
        prediction = logits[row, positions].argmax(dim=-1)
        first_logits = logits[row, positions[0]]
        first_target = int(targets[0])
        log_probs = F.log_softmax(first_logits, dim=-1)
        rank = 1 + int((first_logits > first_logits[first_target]).sum())
        token_nll = -F.log_softmax(logits[row, positions], dim=-1).gather(
            1, targets[:, None]
        ).squeeze(1)
        rows.append(
            {
                "answer_exact": float(torch.equal(prediction, targets)),
                "first_token_top1": float(int(prediction[0]) == first_target),
                "first_token_top5": float(rank <= 5),
                "first_token_rank": float(rank),
                "first_token_nll": float(-log_probs[first_target].cpu()),
                "answer_nll": float(token_nll.mean().cpu()),
                "answer_token_count": float(targets.numel()),
            }
        )
    return rows


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty rows")
    keys = (
        "answer_exact",
        "first_token_top1",
        "first_token_top5",
        "first_token_rank",
        "first_token_nll",
        "answer_nll",
    )
    return {key: statistics.fmean(float(row[key]) for row in rows) for key in keys}


def evaluate_mode(
    model: MemformerDecoderLM,
    generator: SelectiveMemoryStressGenerator,
    device: torch.device,
    *,
    mode: str,
    delay: int,
    examples_count: int,
    batch_size: int,
    segment_length: int,
    seed: int,
    scale: float | None = None,
    noise_scale: float | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    prefix_norms: list[float] = []
    prefix_maxima: list[float] = []
    query_state_norms: list[float] = []
    state_deltas: list[float] = []
    for start in range(0, examples_count, batch_size):
        count = min(batch_size, examples_count - start)
        # A pair is intentional: swapped-state intervention needs a distinct
        # other example while keeping all other batching decisions fixed.
        indices = range(1_000_000 + start, 1_000_000 + start + count)
        _, rounds = generator.batch(
            indices,
            device,
            delay_segments=delay,
            noise_rounds=0,
        )
        prefix_state = process_prefix(
            model,
            rounds,
            device,
            segment_length=segment_length,
            reset_after_each_segment=mode == "per_segment_reset",
        )
        prefix_norms.append(state_norm(prefix_state))
        prefix_maxima.append(state_max_abs(prefix_state))

        if mode == "normal" or mode == "per_segment_reset":
            query_state = prefix_state
        elif mode == "zero":
            query_state = zero_memory(prefix_state)
        elif mode == "swapped":
            if count != 2:
                raise RuntimeError("swapped mode requires batch size two")
            query_state = swap_memory(prefix_state)
        elif mode == "last_slot":
            query_state = last_slot_state(prefix_state)
        elif mode == "scale":
            if scale is None:
                raise ValueError("scale mode requires scale")
            query_state = scaled_state(prefix_state, scale)
        elif mode == "noise":
            if noise_scale is None:
                raise ValueError("noise mode requires noise_scale")
            noise_generator = torch.Generator(device=device)
            noise_generator.manual_seed(seed + start + int(noise_scale * 10_000))
            query_state = noisy_state(prefix_state, noise_scale, noise_generator)
        else:
            raise ValueError(f"unknown mode: {mode}")

        query_state_norms.append(state_norm(query_state))
        state_deltas.append(
            statistics.fmean(
                float((a.float() - b.float()).square().mean().sqrt().cpu())
                for a, b in zip(prefix_state.memories, query_state.memories)
            )
        )
        query_collated = rounds[-1]
        with autocast_context(device):
            output = model(
                query_collated.input_ids,
                state=query_state,
                attention_mask=query_collated.attention_mask,
                segment_length=segment_length,
            )
        rows.extend(row_metrics(output, query_collated))

    metrics = aggregate(rows)
    metrics.update(
        {
            "mode": mode,
            "delay_segments": delay,
            "examples": len(rows),
            "prefix_state_norm": statistics.fmean(prefix_norms),
            "prefix_state_max_abs": statistics.fmean(prefix_maxima),
            "query_state_norm": statistics.fmean(query_state_norms),
            "state_delta_rms": statistics.fmean(state_deltas),
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--examples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--segment-length", type=int, default=32)
    parser.add_argument("--delays", default="1,4,16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cuda()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.batch_size != 2:
        raise ValueError("M10 swapped-state diagnostic requires --batch-size 2")
    seed_all(args.seed)
    delays = tuple(int(piece) for piece in args.delays.split(",") if piece.strip())
    model = load_model(args.run_dir, device)
    generator = SelectiveMemoryStressGenerator(
        TOKENIZER_DIR,
        FILLER_PATH,
        split="validation",
        seed=args.seed + 10_000,
        delay_buckets=delays,
        noise_buckets=(0,),
        segment_length=args.segment_length,
        answer_leading_space=True,
    )

    started = time.perf_counter()
    conditions: dict[str, dict[str, Any]] = {}
    base_modes = ("normal", "zero", "swapped", "per_segment_reset", "last_slot")
    for delay in delays:
        entries: dict[str, Any] = {}
        for mode in base_modes:
            entries[mode] = evaluate_mode(
                model,
                generator,
                device,
                mode=mode,
                delay=delay,
                examples_count=args.examples,
                batch_size=args.batch_size,
                segment_length=args.segment_length,
                seed=args.seed,
            )
        scales = (0.0, 0.25, 0.5, 1.0, 2.0)
        entries["scale"] = {
            str(scale): evaluate_mode(
                model,
                generator,
                device,
                mode="scale",
                delay=delay,
                examples_count=args.examples,
                batch_size=args.batch_size,
                segment_length=args.segment_length,
                seed=args.seed,
                scale=scale,
            )
            for scale in scales
        }
        noise_scales = (0.01, 0.1, 0.5, 1.0)
        entries["noise"] = {
            str(noise): evaluate_mode(
                model,
                generator,
                device,
                mode="noise",
                delay=delay,
                examples_count=args.examples,
                batch_size=args.batch_size,
                segment_length=args.segment_length,
                seed=args.seed,
                noise_scale=noise,
            )
            for noise in noise_scales
        }
        conditions[str(delay)] = entries

    normal = {
        delay: conditions[str(delay)]["normal"] for delay in delays
    }
    result = {
        "protocol_version": "adaptive_color_m10_state_intervention_v1",
        "milestone": "M10",
        "status": "ok",
        "seed": args.seed,
        "method": "Memformer",
        "run_dir": str(args.run_dir),
        "checkpoint": str(args.run_dir / "checkpoint.final.pt"),
        "checkpoint_sha256": sha256_file(args.run_dir / "checkpoint.final.pt"),
        "parent_checkpoint": str(PARENT_CHECKPOINT),
        "parent_checkpoint_sha256": sha256_file(PARENT_CHECKPOINT),
        "tokenizer": str(TOKENIZER_DIR),
        "data": {
            "generator": "SelectiveMemoryStressGenerator",
            "split": "validation",
            "random_name_to_color": True,
            "examples_per_delay": args.examples,
            "batch_size": args.batch_size,
            "delay_segments": list(delays),
            "noise_rounds": [0],
            "answer_leading_space": True,
            "segment_length": args.segment_length,
        },
        "objective": {
            "answer_metrics": "ordinary tied LM head",
            "memory_to_token_reconstruction": False,
            "value_token_alignment": False,
            "fact_labels": False,
            "pointer_copy_head": False,
        },
        "interventions": {
            "normal": "prefix recurrent state passed to query",
            "zero": "all memory slots and counters reset before query",
            "swapped": "two prefix states exchanged before query",
            "per_segment_reset": "state reset after every explicit 32-token prefix segment",
            "last_slot": "all memory slots zeroed except slot 63",
            "scale": [0.0, 0.25, 0.5, 1.0, 2.0],
            "noise": [0.01, 0.1, 0.5, 1.0],
        },
        "conditions": conditions,
        "normal_by_delay": normal,
        "elapsed_seconds": time.perf_counter() - started,
        "code_hashes": {
            "diagnostic_script": sha256_file(Path(__file__)),
            "memformer": sha256_file(SRC / "adaptive_fact_memory" / "memformer.py"),
            "stress_generator": sha256_file(SCRIPT_DIR / "synthetic_memory_stress.py"),
        },
    }
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
