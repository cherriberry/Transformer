"""Attention-only efficiency matrix for the frozen Person-B methods.

The complete TinyLM benchmark includes the tied vocabulary projection, whose
cost can dominate and hide the attention-path difference.  This companion
benchmark isolates one trained layer's attention operation while keeping the
same hidden size, heads, RoPE and method settings.  It is reported as
``attention_only`` and must not be read as end-to-end generation latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from benchmark_person_b import checkpoint_path, load_checkpoint_state, make_config
from run_person_b import (
    FullCausalAttention,
    LongformerSelfAttention,
    MemformerSegmentAttention,
    RotaryEmbedding,
    autocast_context,
    environment_record,
    parameter_record,
    seed_all,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
AGGREGATE_DIR = Path(
    os.environ.get("PERSON_B_AGGREGATE_DIR", str(EXPERIMENT_DIR / "aggregate"))
)


def path_for_record(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class AttentionLayerWrapper(nn.Module):
    def __init__(self, method: str, attention: nn.Module, rope: RotaryEmbedding, segment_length: int = 128):
        super().__init__()
        self.method = method
        self.attention = attention
        self.rope = rope
        self.segment_length = segment_length

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        if self.method == "longformer":
            return self.attention(
                hidden, rotary_emb=self.rope, position_ids=positions
            )
        if self.method == "memformer":
            output, _ = self.attention(
                hidden,
                segment_length=self.segment_length,
                memory=None,
                rotary_emb=self.rope,
                position_offset=0,
            )
            return output
        return self.attention(hidden, self.rope, positions)


def load_attention(
    method: str, device: torch.device
) -> tuple[AttentionLayerWrapper, Path, dict[str, Any]]:
    config = make_config()
    if method == "longformer":
        run_id = "main_b_longformer_w128_s17"
        checkpoint = checkpoint_path(run_id)
        attention = LongformerSelfAttention(
            config.hidden_size,
            config.context,
            heads=config.heads,
            window_size=257,
            global_tokens=0,
            dropout=0.0,
            bias=False,
            output_bias=False,
            causal=True,
            query_chunk_size=config.longformer_query_chunk,
        )
        state = load_checkpoint_state(checkpoint)
        attention.load_state_dict(
            {key[len("blocks.0.attn.") :]: value for key, value in state.items() if key.startswith("blocks.0.attn.")},
            strict=True,
        )
        metadata = {
            "left_window": 128,
            "total_window_size": 257,
            "global_tokens": 0,
            "source_run": run_id,
        }
    elif method == "memformer":
        run_id = "main_b_memformer_s128_m64_s17"
        checkpoint = checkpoint_path(run_id)
        attention = MemformerSegmentAttention(
            config.hidden_size,
            config.heads,
            memory_slots=64,
            causal=True,
            detach_memory=False,
            bias=False,
        )
        state = load_checkpoint_state(checkpoint)
        attention.load_state_dict(
            {key[len("blocks.0.attn.") :]: value for key, value in state.items() if key.startswith("blocks.0.attn.")},
            strict=True,
        )
        metadata = {
            "segment_length": 128,
            "memory_slots": 64,
            "state_bytes_bf16_batch1": config.layers * 64 * config.hidden_size * 2,
            "source_run": run_id,
        }
    elif method == "full_attention_reference":
        run_id = "main_b_longformer_w128_s17"
        checkpoint = checkpoint_path(run_id)
        attention = FullCausalAttention(config.hidden_size, config.heads)
        state = load_checkpoint_state(checkpoint)
        source = {
            "to_q.weight": state["blocks.0.attn.to_q.weight"],
            "to_k.weight": state["blocks.0.attn.to_k.weight"],
            "to_v.weight": state["blocks.0.attn.to_v.weight"],
            "to_out.weight": state["blocks.0.attn.to_out.weight"],
        }
        attention.load_state_dict(source, strict=True)
        metadata = {"reference": "full_attention", "source_run": run_id}
    else:
        raise ValueError(method)

    wrapper = AttentionLayerWrapper(
        "memformer" if method == "memformer" else "longformer" if method == "longformer" else "full",
        attention,
        RotaryEmbedding(config.hidden_size // config.heads, config.rope_max_position),
        segment_length=config.memformer_segment_length,
    ).to(device)
    wrapper.eval()
    return wrapper, checkpoint, metadata


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def timed(
    model: nn.Module,
    hidden: torch.Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    try:
        for _ in range(warmup):
            with autocast_context(device):
                output = model(hidden)
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
                    output = model(hidden)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)) / 1000.0)
            else:
                started = time.perf_counter()
                with autocast_context(device):
                    output = model(hidden)
                samples.append(time.perf_counter() - started)
        synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        median_seconds = float(statistics.median(samples))
        return {
            "status": "ok",
            "output_shape": list(output.shape),
            "output_finite": bool(torch.isfinite(output).all().item()),
            "median_latency_seconds": median_seconds,
            "mean_latency_seconds": float(statistics.mean(samples)),
            "stdev_latency_seconds": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
            "tokens_per_second": float(hidden.shape[1] / max(median_seconds, 1e-12)),
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


def run(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config = make_config()
    records: list[dict[str, Any]] = []
    for method in ("longformer", "memformer", "full_attention_reference"):
        model, checkpoint, method_config = load_attention(method, device)
        try:
            for length in args.lengths:
                generator = torch.Generator(device="cpu").manual_seed(17000 + length)
                hidden = torch.randn(
                    (1, length, config.hidden_size), generator=generator, dtype=torch.float32
                ).to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    baseline_allocated = int(torch.cuda.memory_allocated(device))
                    torch.cuda.empty_cache()
                else:
                    baseline_allocated = None
                result = timed(model, hidden, device, args.warmup, args.runs)
                record = {
                    "benchmark_scope": "attention_only_single_layer",
                    "method": method,
                    "sequence_length": length,
                    "batch_sequences": 1,
                    "dtype": "bfloat16" if device.type == "cuda" else "float32",
                    "source_checkpoint": path_for_record(checkpoint),
                    "method_config": method_config,
                    "parameter_record": parameter_record(model.attention),
                    "baseline_allocated_bytes": baseline_allocated,
                    "config_hash": stable_hash(
                        {
                            "scope": "attention_only_single_layer",
                            "method": method,
                            "length": length,
                            "warmup": args.warmup,
                            "runs": args.runs,
                            "model_config": config.__dict__,
                            "method_config": method_config,
                        }
                    ),
                    **result,
                }
                if result.get("peak_allocated_bytes") is not None and baseline_allocated is not None:
                    record["peak_incremental_bytes"] = max(
                        0, int(result["peak_allocated_bytes"]) - baseline_allocated
                    )
                records.append(record)
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    return {
        "protocol_version": "tinystories_tinylm_v1",
        "execution_profile": "three_day_validation_screening_v1",
        "role": "person_b",
        "benchmark_type": "frozen_checkpoint_attention_only_efficiency",
        "quality_claim": "none_efficiency_only",
        "configuration": {
            "lengths": args.lengths,
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "batch_sequences": 1,
            "model_config": config.__dict__,
            "input": "deterministic Gaussian hidden states; one trained layer's attention weights",
        },
        "environment": environment_record(device),
        "records": records,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", default=str(AGGREGATE_DIR / "person_b_attention_efficiency.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected CUDA device does not support BF16")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(17)
    payload = run(args, device)
    atomic_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "records": len(payload["records"])}, indent=2))


if __name__ == "__main__":
    main()
