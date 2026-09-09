"""Offline diagnostics for the seed-17 memory-stress bottleneck.

This script intentionally does not train or overwrite an existing run.  It
reuses the frozen Gated stress checkpoint and evaluates:

* fixed memory-fusion gate values (write/retention stay unchanged);
* checkpoint trajectory (500/1000/1500/2000 steps);
* train-template versus held-out validation-template performance.

The diagnostics help separate memory selection from downstream answer
generation.  Results are written to a new JSON file on the data disk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "adaptive_gated_fact_memory" / "src"
SCRIPT_DIR = ROOT / "adaptive_gated_fact_memory" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from adaptive_gated_fact_memory.scripts.synthetic_memory_stress import (  # noqa: E402
    SelectiveMemoryStressGenerator,
)
from adaptive_gated_fact_memory.scripts.train_memory_stress import (  # noqa: E402
    BACKBONE_CHECKPOINT,
    FILLER_PATH,
    TOKENIZER_DIR,
    evaluate_stress,
    load_stress_model,
)


DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
DEFAULT_RUN = DATA_DISK / (
    "runs/adaptive_gated_fact_memory_stress/"
    "selective_stress_v2_gated_s17_20260909"
)


class ConstantGate(nn.Module):
    """Return a constant sigmoid gate while preserving the expected shape."""

    def __init__(self, probability: float):
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("gate probability must be in [0, 1]")
        self.probability = float(probability)
        clipped = min(max(self.probability, 1.0e-6), 1.0 - 1.0e-6)
        self.logit = math.log(clipped / (1.0 - clipped))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (*values.shape[:-1], 1),
            self.logit,
            dtype=values.dtype,
            device=values.device,
        )


def load_checkpoint_model(
    device: torch.device,
    checkpoint: Path,
) -> Any:
    model = load_stress_model(
        device,
        training=False,
        memory_policy="gated",
        memory_slots=8,
        memory_read_top_k=4,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model", payload))
    model.eval()
    return model


def set_fusion_gate(model: Any, probability: float) -> None:
    count = 0
    for block in model.blocks:
        if block.memory_fusion is not None:
            block.memory_fusion.gate = ConstantGate(probability).to(
                next(block.parameters()).device
            )
            count += 1
    if count == 0:
        raise RuntimeError("model has no memory fusion layers")


def generator(
    split: str,
    *,
    seed: int,
) -> SelectiveMemoryStressGenerator:
    return SelectiveMemoryStressGenerator(
        TOKENIZER_DIR,
        FILLER_PATH,
        split=split,
        seed=seed,
        delay_buckets=(1, 4, 16),
        noise_buckets=(0, 2, 4, 8),
        segment_length=128,
        payload_tokens=12,
    )


def compact_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Keep aggregate fields and omit 288 verbose rows from the report."""
    keys = (
        "overall_exact_match",
        "overall_target_survival",
        "overall_target_top1",
        "overall_target_mrr",
        "overall_read_hit_at_k",
    )
    result: dict[str, Any] = {key: evaluation[key] for key in keys}
    result["by_condition"] = evaluation["by_condition"]
    return result


@torch.no_grad()
def run_gate_scan(device: torch.device, checkpoint: Path) -> dict[str, Any]:
    values = (0.0, 0.25, 0.5, 0.75, 1.0)
    valid = generator("validation", seed=10_017)
    output: dict[str, Any] = {}
    for value in values:
        model = load_checkpoint_model(device, checkpoint)
        set_fusion_gate(model, value)
        evaluation = evaluate_stress(
            model,
            valid,
            device,
            examples_per_condition=24,
            noise_buckets=(0, 2, 4, 8),
            delay_buckets=(1, 4, 16),
            start_index=1_000_000,
        )
        output[str(value)] = compact_evaluation(evaluation)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


@torch.no_grad()
def run_checkpoint_sweep(device: torch.device, run_dir: Path) -> dict[str, Any]:
    valid = generator("validation", seed=10_017)
    output: dict[str, Any] = {}
    for step in (500, 1000, 1500, 2000):
        checkpoint = run_dir / f"checkpoint.step_{step}.pt"
        model = load_checkpoint_model(device, checkpoint)
        evaluation = evaluate_stress(
            model,
            valid,
            device,
            examples_per_condition=24,
            noise_buckets=(0, 2, 4, 8),
            delay_buckets=(1, 4, 16),
            start_index=1_000_000,
        )
        output[str(step)] = compact_evaluation(evaluation)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


@torch.no_grad()
def run_template_comparison(device: torch.device, checkpoint: Path) -> dict[str, Any]:
    model = load_checkpoint_model(device, checkpoint)
    output: dict[str, Any] = {}
    for split, seed in (("train", 17), ("validation", 10_017)):
        data = generator(split, seed=seed)
        evaluation = evaluate_stress(
            model,
            data,
            device,
            examples_per_condition=24,
            noise_buckets=(0, 2, 4, 8),
            delay_buckets=(1, 4, 16),
            start_index=1_000_000,
        )
        output[split] = compact_evaluation(evaluation)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("gate_scan", "checkpoint_sweep", "template_comparison", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    final_checkpoint = run_dir / "checkpoint.final.pt"
    result: dict[str, Any] = {
        "checkpoint": str(final_checkpoint),
        "mode": args.mode,
        "device": str(device),
    }
    if args.mode in {"gate_scan", "all"}:
        result["gate_scan"] = run_gate_scan(device, final_checkpoint)
    if args.mode in {"checkpoint_sweep", "all"}:
        result["checkpoint_sweep"] = run_checkpoint_sweep(device, run_dir)
    if args.mode in {"template_comparison", "all"}:
        result["template_comparison"] = run_template_comparison(device, final_checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "mode": args.mode}, ensure_ascii=False))


if __name__ == "__main__":
    main()
