"""M12 eval-only noise stress for the random-colour checkpoints.

This script never calls an optimizer.  It loads final checkpoints produced by
M9/M11 and evaluates the same held-out generator over noise and delay cells.
Memformer uses the common-objective evaluator; Fixed-LRU/Gated Memory use the
native stress evaluator and retain their retrieval diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "adaptive_gated_fact_memory" / "src"
SCRIPTS = ROOT / "adaptive_gated_fact_memory" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from adaptive_fact_memory import FactMemoryConfig, MemformerDecoderLM  # noqa: E402
from adaptive_gated_fact_memory.scripts.synthetic_memory_stress import (  # noqa: E402
    SelectiveMemoryStressGenerator,
)
import train_color_m8 as color  # noqa: E402
import train_memory_stress as native  # noqa: E402


DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
DEFAULT_OUTPUT = DATA_DISK / (
    "runs/adaptive_gated_fact_memory_color_m12/m12_noise_stress_s17.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_common(evaluation: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "examples",
        "answer_exact",
        "first_token_top1",
        "first_token_top5",
        "first_token_rank",
        "first_token_nll",
        "answer_nll",
        "lm_nll",
        "objective",
    )
    return {
        key: evaluation.get(key)
        for key in keys
    } | {
        "conditions": {
            key: {
                field: value.get(field)
                for field in (
                    "examples",
                    "answer_exact",
                    "first_token_top1",
                    "first_token_top5",
                    "first_token_rank",
                    "first_token_nll",
                    "answer_nll",
                    "lm_nll",
                    "objective",
                )
            }
            for key, value in evaluation["conditions"].items()
        }
    }


def compact_native(evaluation: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "examples_per_condition",
        "noise_buckets",
        "delay_buckets",
        "overall_exact_match",
        "overall_target_survival",
        "overall_target_top1",
        "overall_target_mrr",
        "overall_read_hit_at_k",
        "overall_teacher_forced_answer_nll",
        "overall_answer_first_token_rank",
        "overall_value_span_exact",
        "overall_memory_token_exact",
    )
    condition_fields = (
        "examples",
        "exact_match",
        "target_survival",
        "target_top1",
        "target_mrr",
        "read_hit_at_k",
        "teacher_forced_answer_nll",
        "answer_first_token_rank",
        "mean_active_slots",
        "mean_state_bytes",
        "mean_noise_accept_rate",
        "value_span_exact",
        "memory_token_exact",
    )
    return {
        key: evaluation.get(key) for key in keys
    } | {
        "by_condition": {
            key: {field: value.get(field) for field in condition_fields}
            for key, value in evaluation["by_condition"].items()
        }
    }


def load_memformer(checkpoint: Path, device: torch.device) -> MemformerDecoderLM:
    parent_state, _ = color.load_parent_state()
    model = MemformerDecoderLM(
        FactMemoryConfig(),
        memory_slots=64,
        segment_length=32,
        detach_memory=False,
    ).to(device)
    color.load_common_parent_into_memformer(model, parent_state)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Memformer checkpoint mismatch: {missing}, {unexpected}")
    model.eval()
    return model


def evaluate_memformer(
    checkpoint: Path,
    device: torch.device,
    *,
    examples: int,
    batch_size: int,
    delays: tuple[int, ...],
    noises: tuple[int, ...],
) -> dict[str, Any]:
    model = load_memformer(checkpoint, device)
    generator = SelectiveMemoryStressGenerator(
        color.TOKENIZER_DIR,
        color.FILLER_PATH,
        split="validation",
        seed=17 + 10_000,
        delay_buckets=delays,
        noise_buckets=noises,
        segment_length=32,
        answer_leading_space=True,
    )
    evaluation = color.evaluate(
        model,
        "Memformer",
        generator,
        device,
        examples_count=examples,
        batch_size=batch_size,
        segment_length=32,
        lambda_answer=1.0,
        start_index=1_000_000,
        split="validation_heldout",
        delay_buckets=delays,
        noise_buckets=noises,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return compact_common(evaluation)


def evaluate_native(
    checkpoint: Path,
    method: str,
    device: torch.device,
    *,
    examples: int,
    delays: tuple[int, ...],
    noises: tuple[int, ...],
    value_token_alignment: bool = False,
) -> dict[str, Any]:
    policy = "fixed_lru" if method.startswith("Fixed-LRU") else "gated"
    model = native.load_stress_model(
        device,
        training=False,
        memory_policy=policy,
        attention_mode="swa",
        memory_slots=8,
        memory_read_top_k=4,
        value_token_alignment=value_token_alignment,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{method} checkpoint mismatch: {missing}, {unexpected}")
    model.eval()
    generator = SelectiveMemoryStressGenerator(
        native.TOKENIZER_DIR,
        native.FILLER_PATH,
        split="validation",
        seed=17 + 10_000,
        delay_buckets=delays,
        noise_buckets=noises,
        segment_length=32,
        payload_tokens=12,
        answer_leading_space=True,
    )
    evaluation = native.evaluate_stress(
        model,
        generator,
        device,
        examples_per_condition=examples,
        noise_buckets=noises,
        delay_buckets=delays,
        start_index=1_000_000,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return compact_native(evaluation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--delays", default="1,4,16")
    parser.add_argument("--noises", default="0,2,4,8")
    parser.add_argument(
        "--memformer-checkpoint",
        type=Path,
        default=DATA_DISK / "runs/adaptive_gated_fact_memory_color_m9/m9_memformer_s17_noise0/checkpoint.final.pt",
    )
    parser.add_argument(
        "--fa-checkpoint",
        type=Path,
        default=DATA_DISK / "runs/adaptive_gated_fact_memory_color_m9/m9_fa_task_s17_noise0/checkpoint.final.pt",
    )
    parser.add_argument(
        "--swa-checkpoint",
        type=Path,
        default=DATA_DISK / "runs/adaptive_gated_fact_memory_color_m9/m9_swa_task_s17_noise0/checkpoint.final.pt",
    )
    parser.add_argument(
        "--fixed-checkpoint",
        type=Path,
        default=DATA_DISK / "runs/adaptive_gated_fact_memory_color_m11/m11_fixed_lru_s17_noise0/checkpoint.final.pt",
    )
    parser.add_argument(
        "--gated-checkpoint",
        type=Path,
        default=DATA_DISK / "runs/adaptive_gated_fact_memory_color_m11/m11_gated_memory_s17_noise0/checkpoint.final.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.examples <= 0 or args.batch_size <= 0:
        raise ValueError("examples and batch size must be positive")
    configure = native.configure_cuda
    configure()
    # The native runner's historical module default points at the old
    # TinyStories parent.  Override it in-process so every M12 method uses the
    # exact M9/M11 parent, independent of shell environment variables.
    native.BACKBONE_CHECKPOINT = color.PARENT_CHECKPOINT
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    delays = tuple(int(x) for x in args.delays.split(",") if x.strip())
    noises = tuple(int(x) for x in args.noises.split(",") if x.strip())
    started = time.perf_counter()
    checkpoints = {
        "FA-task": args.fa_checkpoint,
        "SWA-task": args.swa_checkpoint,
        "Memformer": args.memformer_checkpoint,
        "Fixed-LRU": args.fixed_checkpoint,
        "Gated Memory": args.gated_checkpoint,
    }
    for name, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name} checkpoint: {path}")
    results: dict[str, Any] = {}
    # Native evaluator can only represent the two memory methods; FA/SWA use
    # the common evaluator exactly like M9.
    for name in ("FA-task", "SWA-task"):
        parent_state, _ = color.load_parent_state()
        mode = "full" if name == "FA-task" else "swa"
        model = color.AdaptiveFactMemoryLM(
            FactMemoryConfig(), memory_policy="none", attention_mode=mode
        ).to(device)
        payload = torch.load(checkpoints[name], map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"{name} checkpoint mismatch: {missing}, {unexpected}")
        model.eval()
        generator = SelectiveMemoryStressGenerator(
            color.TOKENIZER_DIR,
            color.FILLER_PATH,
            split="validation",
            seed=17 + 10_000,
            delay_buckets=delays,
            noise_buckets=noises,
            segment_length=32,
            answer_leading_space=True,
        )
        results[name] = compact_common(
            color.evaluate(
                model,
                name,
                generator,
                device,
                examples_count=args.examples,
                batch_size=args.batch_size,
                segment_length=32,
                lambda_answer=1.0,
                start_index=1_000_000,
                split="validation_heldout",
                delay_buckets=delays,
                noise_buckets=noises,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["Memformer"] = evaluate_memformer(
        checkpoints["Memformer"], device, examples=args.examples,
        batch_size=args.batch_size, delays=delays, noises=noises
    )
    for name in ("Fixed-LRU", "Gated Memory"):
        results[name] = evaluate_native(
            checkpoints[name], name, device, examples=args.examples,
            delays=delays, noises=noises
        )
    output = {
        "protocol_version": "adaptive_color_m12_noise_stress_v1",
        "milestone": "M12",
        "status": "ok",
        "seed": 17,
        "device": str(device),
        "data": {
            "split": "validation",
            "examples_per_condition": args.examples,
            "batch_size": args.batch_size,
            "delay_segments": list(delays),
            "noise_rounds": list(noises),
            "segment_length": 32,
            "answer_leading_space": True,
            "random_name_to_color": True,
        },
        "objective": {
            "eval_only": True,
            "parameters_updated": False,
            "value_token_alignment": False,
            "reconstruction": False,
        },
        "checkpoints": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in checkpoints.items()
        },
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
        "script_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": "ok", "output": str(args.output), "elapsed_seconds": output["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
