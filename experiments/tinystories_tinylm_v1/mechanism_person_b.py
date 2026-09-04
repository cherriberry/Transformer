"""Measure cross-segment causal influence of the frozen Person-B models.

This is a mechanism diagnostic, not a passkey/copy accuracy benchmark.  The
same validation-token prefix is evaluated twice; only the first 128 token IDs
are replaced.  We report the change in final-token logits and hidden-independent
summary statistics.  A zero/near-zero Longformer result beyond its finite
layered local receptive field is expected, while Memformer can transmit an
effect through its recurrent state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_person_b import load_model, validation_tokens
from run_person_b import autocast_context, environment_record


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
AGG = EXP / "aggregate"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_config():
    from run_person_b import ModelConfig

    return ModelConfig(context=512, longformer_query_chunk=128, memformer_segment_length=128)


def run(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config = model_config()
    specs = [
        ("longformer", 128, "main_b_longformer_w128_s17"),
        ("memformer", 64, "main_b_memformer_s128_m64_s17"),
    ]
    records: list[dict[str, Any]] = []
    for method, method_value, run_id in specs:
        model, checkpoint = load_model(method, method_value, run_id, device)
        try:
            for length in args.lengths:
                original = validation_tokens(length, device)
                changed = original.clone()
                generator = torch.Generator(device="cpu").manual_seed(91000 + length)
                replacement = torch.randint(
                    0, 50257, (1, min(args.perturb_tokens, length)), generator=generator
                ).to(device)
                changed[:, : replacement.shape[1]] = replacement
                with torch.inference_mode():
                    with autocast_context(device):
                        original_logits = model(original)
                        changed_logits = model(changed)
                delta = (original_logits.float() - changed_logits.float()).abs()
                tail = delta[:, -min(args.tail_tokens, length) :, :]
                record = {
                    "benchmark_type": "cross_segment_causal_influence",
                    "method": method,
                    "method_value": method_value,
                    "run_id": run_id,
                    "source_checkpoint": str(checkpoint.relative_to(ROOT)),
                    "sequence_length": length,
                    "perturb_tokens": int(replacement.shape[1]),
                    "tail_tokens_measured": int(tail.shape[1]),
                    "max_abs_logit_delta_all_positions": float(delta.max().cpu()),
                    "max_abs_logit_delta_tail": float(tail.max().cpu()),
                    "mean_abs_logit_delta_tail": float(tail.mean().cpu()),
                    "last_token_max_abs_logit_delta": float(delta[:, -1, :].max().cpu()),
                    "last_token_mean_abs_logit_delta": float(delta[:, -1, :].mean().cpu()),
                    "finite": bool(torch.isfinite(original_logits).all() and torch.isfinite(changed_logits).all()),
                    "expected_local_receptive_distance_upper_bound": (
                        config.layers * method_value if method == "longformer" else None
                    ),
                    "cross_segment_state_path": method == "memformer",
                    "config_hash": stable_hash(
                        {
                            "method": method,
                            "method_value": method_value,
                            "length": length,
                            "perturb_tokens": int(replacement.shape[1]),
                            "tail_tokens": args.tail_tokens,
                            "layers": config.layers,
                            "segment_length": config.memformer_segment_length,
                        }
                    ),
                }
                records.append(record)
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {
        "protocol_version": "tinystories_tinylm_v1",
        "execution_profile": "three_day_validation_screening_v1",
        "role": "person_b",
        "benchmark_type": "cross_segment_causal_influence",
        "quality_claim": "none_mechanism_diagnostic_only",
        "configuration": {
            "lengths": args.lengths,
            "perturb_tokens": args.perturb_tokens,
            "tail_tokens": args.tail_tokens,
            "model_config": config.__dict__,
            "input": "pinned TinyStories validation prefix; first token block replaced deterministically",
        },
        "environment": environment_record(device),
        "records": records,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--perturb-tokens", type=int, default=128)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--output", default=str(AGG / "person_b_cross_segment_influence.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected CUDA device does not support BF16")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = False
    payload = run(args, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(payload["records"])}, indent=2))


if __name__ == "__main__":
    main()
