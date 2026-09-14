"""Evaluate a frozen Full-Attention TinyStories parent under inference-only SWA.

The parent parameters are loaded once and never optimized.  Each SWA budget is
an independently constructed bounded-attention model with the same weights.
Validation metrics include token accuracy and token-weighted NLL.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
os.environ.setdefault("TINYSTORIES_DATA_DIR", str(DATA_DISK / "data/raw/tinystories/data"))
os.environ.setdefault("TINYSTORIES_TOKENIZER_DIR", str(DATA_DISK / "data/tokenizer/gpt2"))
os.environ.setdefault("TINYSTORIES_CACHE_DIR", str(DATA_DISK / "data/cache/tinystories_tinylm_v1"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adaptive_gated_fact_memory" / "src"))
from experiments.tinystories_tinylm_v1 import run_person_b as protocol  # noqa: E402
from adaptive_fact_memory import AdaptiveFactMemoryLM, FactMemoryConfig  # noqa: E402


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model: AdaptiveFactMemoryLM, stream, device: torch.device, batch_sequences: int) -> dict[str, Any]:
    model.eval()
    consumed = 0
    block_cursor = 0
    total_loss = 0.0
    total_correct = 0
    started = time.perf_counter()
    while consumed < stream.prediction_count:
        remaining = stream.prediction_count - consumed
        valid_predictions = min(batch_sequences * stream.context, remaining)
        sequences = math.ceil(valid_predictions / stream.context)
        inputs, targets, valid = stream.batch(block_cursor, sequences, valid_predictions, device)
        block_cursor += sequences
        with autocast_context(device):
            logits = model.forward_tinystories(inputs)
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(valid)
        total_loss += float((losses * valid).sum().detach().cpu())
        total_correct += int(((logits.argmax(dim=-1) == targets) & valid).sum().detach().cpu())
        consumed += valid_predictions
    elapsed = time.perf_counter() - started
    nll = total_loss / max(consumed, 1)
    return {
        "tokens": consumed,
        "token_weighted_nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "token_accuracy": total_correct / max(consumed, 1),
        "elapsed_seconds": elapsed,
        "tokens_per_second": consumed / max(elapsed, 1e-9),
    }


def load_state(parent: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint does not contain a model state dict: {parent}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--budgets", default="64,128,256,512")
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--validation-batch-sequences", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    budgets = tuple(int(item.strip()) for item in args.budgets.split(",") if item.strip())
    if not budgets or any(budget <= args.sink_tokens for budget in budgets):
        raise ValueError("budgets must be larger than sink-tokens")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_path, validation_metadata = protocol.build_token_cache("validation", None)
    validation_stream = protocol.PackedTokenStream(
        validation_path, int(validation_metadata["token_count"]), 512
    )
    state = load_state(args.parent_checkpoint)
    results: list[dict[str, Any]] = []
    # The same frozen state dict is copied into each attention configuration.
    for budget in (None, *budgets):
        attention_mode = "full" if budget is None else "swa"
        model_budget = budgets[-1] if budget is None else budget
        config = replace(
            FactMemoryConfig(),
            attention_budget=model_budget,
            sink_tokens=args.sink_tokens,
        )
        model = AdaptiveFactMemoryLM(
            config,
            memory_policy="none",
            attention_mode=attention_mode,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"parent state mismatch for {attention_mode}/{budget}: "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )
        model.to(device)
        result = evaluate(model, validation_stream, device, args.validation_batch_sequences)
        result.update(
            {
                "attention_mode": attention_mode,
                "attention_budget": budget,
                "sink_tokens": args.sink_tokens,
            }
        )
        results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "protocol_version": "training_free_swa_v1",
        "parent_checkpoint": str(args.parent_checkpoint),
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "parent_frozen": True,
        "data": {
            "validation_cache": validation_metadata,
            "context": 512,
            "tokenizer": str(protocol.TOKENIZER_DIR),
        },
        "results": results,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_runtime": torch.version.cuda,
        },
    }
    output = args.output_dir / "training_free_swa_results.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
