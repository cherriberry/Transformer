"""Benchmark the frozen Person-B TinyLM checkpoints.

This benchmark is deliberately separate from training quality.  It loads the
last checkpoint for the selected Longformer and Memformer runs, measures a
complete TinyLM forward (including the tied vocabulary projection), and
records raw CUDA timings, peak memory, and failures over several sequence
lengths.  A Full-Attention reference uses the Longformer checkpoint's common
weights so that the timing comparison is not confounded by different random
weights; it is an efficiency reference, not an independently trained quality
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_person_b import (
    CACHE_DIR,
    ModelConfig,
    PackedTokenStream,
    TinyLM,
    autocast_context,
    build_token_cache,
    environment_record,
    parameter_record,
    seed_all,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = Path(
    os.environ.get("PERSON_B_RUNS_DIR", str(EXPERIMENT_DIR / "runs" / "person_b"))
)
AGGREGATE_DIR = Path(
    os.environ.get("PERSON_B_AGGREGATE_DIR", str(EXPERIMENT_DIR / "aggregate"))
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def path_for_record(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def checkpoint_path(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    candidates = list(run_dir.glob("checkpoint.tokens_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found for run {run_id}")

    def token_count(path: Path) -> int:
        match = re.search(r"checkpoint\.tokens_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=token_count)


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    # ``weights_only`` is available in the current torch, but keep a fallback
    # for older installations used by collaborators.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"checkpoint does not contain a model state: {path}")
    return checkpoint["model"]


def make_config() -> ModelConfig:
    return ModelConfig(context=512, longformer_query_chunk=128, memformer_segment_length=128)


def load_model(
    method: str,
    method_value: int,
    run_id: str,
    device: torch.device,
) -> tuple[TinyLM, Path]:
    model = TinyLM(make_config(), method, method_value, seed=17).to(device)
    path = checkpoint_path(run_id)
    state = load_checkpoint_state(path)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, path


def load_full_reference(device: torch.device, source_path: Path) -> tuple[TinyLM, Path]:
    """Load a Full-Attention model with the common Longformer checkpoint state."""

    model = TinyLM(make_config(), "full_attention", 0, seed=17).to(device)
    state = load_checkpoint_state(source_path)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, source_path


def validation_tokens(length: int, device: torch.device) -> torch.Tensor:
    """Use the pinned validation stream when available, with deterministic fallback."""

    try:
        path, metadata = build_token_cache("validation", None)
        stream = PackedTokenStream(path, int(metadata["token_count"]), context=length)
        values = np.asarray(stream.tokens[:length], dtype=np.int64)
        if values.size == length:
            return torch.from_numpy(values[None, :]).to(device)
    except Exception:  # noqa: BLE001 - benchmark should still produce a record
        pass
    generator = torch.Generator(device="cpu").manual_seed(17000 + length)
    return torch.randint(0, 50257, (1, length), generator=generator, device=device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def timed_forward(
    model: TinyLM,
    tokens: torch.Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    model.eval()
    try:
        for _ in range(warmup):
            with autocast_context(device):
                output = model(tokens)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        samples: list[float] = []
        for _ in range(runs):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with autocast_context(device):
                    output = model(tokens)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)) / 1000.0)
            else:
                started = time.perf_counter()
                with autocast_context(device):
                    output = model(tokens)
                samples.append(time.perf_counter() - started)
        synchronize(device)
        if device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = None
            peak_reserved = None
        median_seconds = float(statistics.median(samples))
        return {
            "status": "ok",
            "output_shape": list(output.shape),
            "output_finite": bool(torch.isfinite(output).all().item()),
            "median_latency_seconds": median_seconds,
            "mean_latency_seconds": float(statistics.mean(samples)),
            "stdev_latency_seconds": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
            "tokens_per_second": float(tokens.shape[1] / max(median_seconds, 1e-12)),
            "timed_samples_seconds": samples,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"status": "oom", "error": repr(exc)}
    except Exception as exc:  # noqa: BLE001
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"status": "error", "error": repr(exc)}


def method_metadata(method: str, method_value: int, config: ModelConfig) -> dict[str, Any]:
    if method == "longformer":
        return {
            "left_window": method_value,
            "total_window_size": 2 * method_value + 1,
            "global_tokens": 0,
            "causal_max_visible_tokens": method_value + 1,
        }
    if method == "memformer":
        state_bytes = config.layers * method_value * config.hidden_size * 2
        return {
            "segment_length": config.memformer_segment_length,
            "memory_slots": method_value,
            "detach_memory_between_segments": False,
            "state_bytes_bf16_batch1": state_bytes,
        }
    return {"reference": "full_attention", "global_tokens": 0}


def benchmark(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config = make_config()
    longformer_path = checkpoint_path("main_b_longformer_w128_s17")
    memformer_path = checkpoint_path("main_b_memformer_s128_m64_s17")
    # Load and benchmark one model at a time.  Keeping all three models alive
    # would make the allocator's peak include unrelated parameter weights and
    # would erase the memory difference this matrix is intended to measure.
    model_specs: list[tuple[str, str, int, str, Path, dict[str, Any]]] = [
        (
            "longformer",
            "longformer",
            128,
            "main_b_longformer_w128_s17",
            longformer_path,
            method_metadata("longformer", 128, config),
        ),
        (
            "memformer",
            "memformer",
            64,
            "main_b_memformer_s128_m64_s17",
            memformer_path,
            method_metadata("memformer", 64, config),
        ),
        (
            "full_attention_reference",
            "full_attention",
            0,
            "main_b_longformer_w128_s17",
            longformer_path,
            method_metadata("full", 0, config),
        ),
    ]
    records: list[dict[str, Any]] = []
    for name, method, method_value, run_id, source_path, method_config in model_specs:
        if method == "full_attention":
            model, loaded_path = load_full_reference(device, source_path)
        else:
            model, loaded_path = load_model(method, method_value, run_id, device)
        try:
            for length in args.lengths:
                tokens = validation_tokens(length, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    baseline_allocated = int(torch.cuda.memory_allocated(device))
                else:
                    baseline_allocated = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                result = timed_forward(model, tokens, device, args.warmup, args.runs)
                payload = {
                    "method": name,
                    "sequence_length": length,
                    "batch_sequences": 1,
                    "dtype": "bfloat16" if device.type == "cuda" else "float32",
                    "source_checkpoint": path_for_record(loaded_path),
                    "method_config": method_config,
                    "parameter_record": parameter_record(model),
                    "baseline_allocated_bytes": baseline_allocated,
                    "config_hash": stable_hash(
                        {
                            "method": name,
                            "sequence_length": length,
                            "warmup": args.warmup,
                            "runs": args.runs,
                            "model_config": config.__dict__,
                            "method_config": method_config,
                            "compute_dtype": "bfloat16" if device.type == "cuda" else "float32",
                            "autocast": device.type == "cuda",
                        }
                    ),
                    **result,
                }
                if result.get("peak_allocated_bytes") is not None and baseline_allocated is not None:
                    payload["peak_incremental_bytes"] = max(
                        0, int(result["peak_allocated_bytes"]) - baseline_allocated
                    )
                records.append(payload)
                if result.get("status") == "oom":
                    # Keep testing the other lengths, but clear the allocator
                    # before the next attempt for this model.
                    torch.cuda.empty_cache()
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

    return {
        "protocol_version": "tinystories_tinylm_v1",
        "execution_profile": "three_day_validation_screening_v1",
        "role": "person_b",
        "benchmark_type": "frozen_checkpoint_forward_efficiency",
        "quality_claim": "none_efficiency_only",
        "configuration": {
            "lengths": args.lengths,
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "batch_sequences": 1,
            "context_trained": 512,
            "compute_dtype": "bfloat16" if device.type == "cuda" else "float32",
            "autocast": device.type == "cuda",
            "full_reference_weight_source": "main_b_longformer_w128_s17 common state",
            "model_config": config.__dict__,
        },
        "environment": environment_record(device),
        "records": records,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--output",
        default=str(AGGREGATE_DIR / "person_b_efficiency.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected CUDA device does not support BF16")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(17)
    payload = benchmark(args, device)
    output = Path(args.output)
    atomic_json(output, payload)
    print(json.dumps({"output": str(output), "records": len(payload["records"])}, indent=2))


if __name__ == "__main__":
    main()
