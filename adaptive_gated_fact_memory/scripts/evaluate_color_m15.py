"""M15 eval-only noise stress for the value-token alignment pair.

This evaluates the two new M14 checkpoints with the same held-out grid used by
M12.  Both models use the ordinary decoder LM head for answer generation; the
alignment losses are only used during training and for diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

import evaluate_color_m12 as m12
import train_memory_stress as native


DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
DEFAULT_OUTPUT = DATA_DISK / (
    "runs/adaptive_gated_fact_memory_color_m15/m15_alignment_noise_stress_s17.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> m12.argparse.Namespace:
    parser = m12.argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--examples", type=int, default=24)
    parser.add_argument("--delays", default="1,4,16")
    parser.add_argument("--noises", default="0,2,4,8")
    parser.add_argument(
        "--value-token-gated-checkpoint",
        type=Path,
        default=DATA_DISK
        / "runs/adaptive_gated_fact_memory_color_m14"
        / "m14_value_token_gated_s17_noise0/checkpoint.final.pt",
    )
    parser.add_argument(
        "--fixed-lru-alignment-checkpoint",
        type=Path,
        default=DATA_DISK
        / "runs/adaptive_gated_fact_memory_color_m14"
        / "m14_fixed_lru_alignment_s17_noise0/checkpoint.final.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.examples <= 0:
        raise ValueError("examples must be positive")
    native.configure_cuda()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    delays = tuple(int(x) for x in args.delays.split(",") if x.strip())
    noises = tuple(int(x) for x in args.noises.split(",") if x.strip())
    checkpoints = {
        "Value-token Gated": args.value_token_gated_checkpoint,
        "Fixed-LRU + alignment": args.fixed_lru_alignment_checkpoint,
    }
    for name, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name} checkpoint: {path}")
    native.BACKBONE_CHECKPOINT = m12.color.PARENT_CHECKPOINT
    started = time.perf_counter()
    results = {
        name: m12.evaluate_native(
            path,
            name,
            device,
            examples=args.examples,
            delays=delays,
            noises=noises,
            value_token_alignment=True,
        )
        for name, path in checkpoints.items()
    }
    output = {
        "protocol_version": "adaptive_color_m15_alignment_noise_stress_v1",
        "milestone": "M15",
        "status": "ok",
        "seed": 17,
        "device": str(device),
        "data": {
            "split": "validation",
            "examples_per_condition": args.examples,
            "delay_segments": list(delays),
            "noise_rounds": list(noises),
            "segment_length": 32,
            "answer_leading_space": True,
            "random_name_to_color": True,
        },
        "objective": {
            "eval_only": True,
            "parameters_updated": False,
            "value_token_alignment": True,
            "reconstruction": False,
            "answer_head": "ordinary decoder LM head",
        },
        "checkpoints": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in checkpoints.items()
        },
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
        "script_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "elapsed_seconds": output["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
