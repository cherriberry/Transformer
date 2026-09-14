"""M13 inference benchmark for the seed-17 random-colour checkpoints.

The benchmark measures the same held-out prefix/query example for every final
checkpoint.  It separates prefix prefill from query+greedy decode, reports the
tokens included in each timed region, and records persistent state bytes after
the prefix.  No optimizer or checkpoint write is used.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import statistics
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

from adaptive_fact_memory import (  # noqa: E402
    AdaptiveFactMemoryLM,
    FactMemoryConfig,
    MemformerDecoderLM,
    SourceRole,
)
from adaptive_gated_fact_memory.scripts import train_color_m8 as color  # noqa: E402
from adaptive_gated_fact_memory.scripts import train_memory_stress as native  # noqa: E402
from adaptive_gated_fact_memory.scripts.synthetic_memory_stress import (  # noqa: E402
    SelectiveMemoryStressGenerator,
)


DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
DEFAULT_OUTPUT = DATA_DISK / (
    "runs/adaptive_gated_fact_memory_color_m13/m13_inference_benchmark_s17.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")


def autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_payload(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state dict: {path}")
    return state


def load_model(name: str, checkpoint: Path, device: torch.device):
    parent_state, _ = color.load_parent_state()
    state = load_payload(checkpoint)
    if name == "FA-task":
        model = AdaptiveFactMemoryLM(
            FactMemoryConfig(), memory_policy="none", attention_mode="full"
        ).to(device)
        missing, unexpected = model.load_state_dict(state, strict=True)
    elif name == "SWA-task":
        model = AdaptiveFactMemoryLM(
            FactMemoryConfig(), memory_policy="none", attention_mode="swa"
        ).to(device)
        missing, unexpected = model.load_state_dict(state, strict=True)
    elif name == "Memformer":
        model = MemformerDecoderLM(
            FactMemoryConfig(), memory_slots=64, segment_length=32, detach_memory=False
        ).to(device)
        color.load_common_parent_into_memformer(model, parent_state)
        missing, unexpected = model.load_state_dict(state, strict=True)
    else:
        policy = "fixed_lru" if name.startswith("Fixed-LRU") else "gated"
        alignment = "alignment" in name or name == "Value-token Gated"
        model = native.load_stress_model(
            device,
            training=False,
            memory_policy=policy,
            attention_mode="swa",
            memory_slots=8,
            memory_read_top_k=4,
            value_token_alignment=alignment,
        )
        missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{name} checkpoint mismatch: {missing}, {unexpected}")
    model.eval()
    return model


def round_tensors(sample, device: torch.device, *, limit: int | None = None):
    ids = list(sample.input_ids if limit is None else sample.input_ids[:limit])
    roles = list(sample.roles if limit is None else sample.roles[:limit])
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)[None]
    role_ids = torch.tensor(roles, dtype=torch.long, device=device)[None]
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, role_ids, mask


def initial_state(model, batch: int, device: torch.device):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return model.initial_state(batch, device=device, dtype=dtype)


def forward_round(model, name: str, ids, roles, mask, state, device, *, commit: bool):
    if name == "Memformer":
        return model(
            ids,
            state=state,
            attention_mask=mask,
            segment_length=32,
        )
    return model.forward_round(
        ids,
        state=state,
        attention_mask=mask,
        source_roles=roles,
        commit=commit,
        hard_memory=True,
    )


@torch.no_grad()
def prefix(model, name: str, example, device: torch.device):
    state = initial_state(model, 1, device)
    token_count = 0
    with autocast(device):
        for sample in example.rounds[:-1]:
            ids, roles, mask = round_tensors(sample, device)
            output = forward_round(
                model, name, ids, roles, mask, state, device,
                commit=name not in {"FA-task", "SWA-task", "Memformer"},
            )
            state = output.state
            token_count += ids.shape[1]
    return state, token_count


@torch.no_grad()
def decode(model, name: str, example, state, device: torch.device) -> int:
    query = example.rounds[-1]
    prompt_len = int(query.answer_start or 0)
    ids, roles, mask = round_tensors(query, device, limit=prompt_len)
    with autocast(device):
        output = forward_round(model, name, ids, roles, mask, state, device, commit=False)
        current = output.logits[:, -1].argmax(dim=-1)
        generated = 0
        for index in range(2):
            token_ids = current[:, None]
            token_roles = torch.full_like(token_ids, int(SourceRole.ASSISTANT))
            token_mask = torch.ones_like(token_ids, dtype=torch.bool)
            output = forward_round(
                model, name, token_ids, token_roles, token_mask, output.state,
                device, commit=False,
            )
            generated += 1
            if int(current[0]) in {50_256, model.config.pad_token_id}:
                break
            current = output.logits[:, -1].argmax(dim=-1)
    return prompt_len + generated


def state_bytes(model, name: str, state) -> int:
    if name == "Memformer":
        return int(state.state_bytes())
    return int(model.effective_state_bytes(state))


def timed_method(
    name: str,
    checkpoint: Path,
    example,
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    model = load_model(name, checkpoint, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    for _ in range(warmup):
        prefix(model, name, example, device)
        state, _ = prefix(model, name, example, device)
        decode(model, name, example, copy.deepcopy(state), device)
    synchronize(device)

    prefill_seconds: list[float] = []
    prefill_tokens = 0
    decode_seconds: list[float] = []
    decode_tokens = 0
    persistent_bytes = 0
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        state, prefill_tokens = prefix(model, name, example, device)
        synchronize(device)
        prefill_seconds.append(time.perf_counter() - started)
        persistent_bytes = state_bytes(model, name, state)

        state_for_decode = state.detach()
        synchronize(device)
        started = time.perf_counter()
        decode_tokens = decode(model, name, example, state_for_decode, device)
        synchronize(device)
        decode_seconds.append(time.perf_counter() - started)

    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mean_prefill = statistics.fmean(prefill_seconds)
    mean_decode = statistics.fmean(decode_seconds)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameters": parameters,
        "persistent_state_bytes_after_prefix": persistent_bytes,
        "prefix_tokens": prefill_tokens,
        "decode_tokens_including_query": decode_tokens,
        "warmup": warmup,
        "repeats": repeats,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "mean_prefill_seconds": mean_prefill,
        "mean_decode_seconds": mean_decode,
        "prefill_tokens_per_second": prefill_tokens / mean_prefill,
        "decode_tokens_per_second": decode_tokens / mean_decode,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=int, default=1_000_123)
    parser.add_argument("--delay", type=int, default=16)
    parser.add_argument("--noise", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    configure_cuda()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    native.BACKBONE_CHECKPOINT = color.PARENT_CHECKPOINT
    generator = SelectiveMemoryStressGenerator(
        color.TOKENIZER_DIR,
        color.FILLER_PATH,
        split="validation",
        seed=17 + 10_000,
        delay_buckets=(args.delay,),
        noise_buckets=(args.noise,),
        segment_length=32,
        payload_tokens=12,
        answer_leading_space=True,
    )
    example = generator.make(args.index, delay_segments=args.delay, noise_rounds=args.noise)
    checkpoints = {
        "FA-task": DATA_DISK / "runs/adaptive_gated_fact_memory_color_m9/m9_fa_task_s17_noise0/checkpoint.final.pt",
        "SWA-task": DATA_DISK / "runs/adaptive_gated_fact_memory_color_m9/m9_swa_task_s17_noise0/checkpoint.final.pt",
        "Memformer": DATA_DISK / "runs/adaptive_gated_fact_memory_color_m9/m9_memformer_s17_noise0/checkpoint.final.pt",
        "Fixed-LRU": DATA_DISK / "runs/adaptive_gated_fact_memory_color_m11/m11_fixed_lru_s17_noise0/checkpoint.final.pt",
        "Gated Memory": DATA_DISK / "runs/adaptive_gated_fact_memory_color_m11/m11_gated_memory_s17_noise0/checkpoint.final.pt",
        "Value-token Gated": DATA_DISK / "runs/adaptive_gated_fact_memory_color_m14/m14_value_token_gated_s17_noise0/checkpoint.final.pt",
        "Fixed-LRU + alignment": DATA_DISK / "runs/adaptive_gated_fact_memory_color_m14/m14_fixed_lru_alignment_s17_noise0/checkpoint.final.pt",
    }
    for name, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name} checkpoint: {path}")
    started = time.perf_counter()
    results: dict[str, Any] = {}
    for name, checkpoint in checkpoints.items():
        results[name] = timed_method(
            name,
            checkpoint,
            example,
            device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    output = {
        "protocol_version": "adaptive_color_m13_inference_benchmark_v1",
        "milestone": "M13",
        "status": "ok",
        "seed": 17,
        "device": str(device),
        "protocol": {
            "split": "validation",
            "index": args.index,
            "delay_segments": args.delay,
            "noise_rounds": args.noise,
            "segment_length": 32,
            "answer_leading_space": True,
            "batch_size": 1,
            "precision": "CUDA BF16 autocast / FP32 parameters"
            if device.type == "cuda"
            else "FP32",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "prefill": "all durable/noise/filler rounds before query",
            "decode": "query prompt plus up to two greedy answer tokens",
        },
        "example": {
            "name": example.name,
            "value": example.value,
            "prefix_tokens": sum(len(round_.input_ids) for round_ in example.rounds[:-1]),
            "query_prompt_tokens": int(example.rounds[-1].answer_start or 0),
            "answer_tokens": len(example.answer_ids),
        },
        "parent_checkpoint": str(color.PARENT_CHECKPOINT),
        "parent_checkpoint_sha256": sha256_file(color.PARENT_CHECKPOINT),
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
