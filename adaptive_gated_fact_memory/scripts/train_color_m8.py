"""M8 short sanity runs for FA-task, SWA-task and Memformer.

The three methods consume exactly the same deterministic colour examples and
the same next-token/answer objective.  This runner is intentionally separate
from the native fact-memory stress runner: Memformer has no fact labels,
lexical payloads, reconstruction head, or value-token alignment loss.

Only a final checkpoint is written.  The runner refuses to overwrite an
existing run directory so an interrupted experiment remains auditable.
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
    MemformerDecoderLM,
    SourceRole,
)
from adaptive_gated_fact_memory.scripts.synthetic_memory_stress import (  # noqa: E402
    SelectiveMemoryStressGenerator,
)


DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
TOKENIZER_DIR = Path(
    os.environ.get("TINYSTORIES_TOKENIZER_DIR", str(DATA_DISK / "data/tokenizer/gpt2"))
)
FILLER_PATH = Path(
    os.environ.get(
        "TINYSTORIES_FILLER_PATH",
        str(DATA_DISK / "data/cache/tinystories_tinylm_v1/train_100000257.int32.bin"),
    )
)
PARENT_CHECKPOINT = Path(
    os.environ.get(
        "ADAPTIVE_TINYSTORIES_FULL_PARENT",
        str(
            DATA_DISK
            / "runs/adaptive_gated_fact_memory_tinystories"
            / "tinystories_full_parent_s17_step6_20260913/checkpoint.final.pt"
        ),
    )
)
RUNS_DIR = Path(
    os.environ.get(
        "ADAPTIVE_COLOR_M8_RUNS_DIR",
        str(DATA_DISK / "runs/adaptive_gated_fact_memory_color_m8"),
    )
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


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
    except Exception:  # pragma: no cover - numpy is present in the runner env
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
    result: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu": props.name,
                "gpu_total_memory_bytes": props.total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return result


def load_parent_state() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not PARENT_CHECKPOINT.exists():
        raise FileNotFoundError(f"parent checkpoint is missing: {PARENT_CHECKPOINT}")
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise ValueError("parent checkpoint does not contain a model state dict")
    return state, payload


def _copy_checked(
    destination: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    destination_key: str,
    source_key: str,
) -> None:
    if source_key not in source:
        raise KeyError(f"parent checkpoint is missing {source_key}")
    if destination_key not in destination:
        raise KeyError(f"Memformer model is missing {destination_key}")
    if destination[destination_key].shape != source[source_key].shape:
        raise ValueError(
            f"shape mismatch for {destination_key}: "
            f"{tuple(destination[destination_key].shape)} vs "
            f"{tuple(source[source_key].shape)}"
        )
    destination[destination_key].copy_(source[source_key])


def load_common_parent_into_memformer(
    model: MemformerDecoderLM, parent_state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Load only the shared backbone; leave recurrent projections fresh."""

    destination = model.state_dict()
    with torch.no_grad():
        _copy_checked(destination, parent_state, "token_embedding.weight", "token_embedding.weight")
        _copy_checked(destination, parent_state, "final_norm.weight", "final_norm.weight")
        _copy_checked(destination, parent_state, "final_norm.bias", "final_norm.bias")
        for layer in range(model.config.layers):
            prefix = f"blocks.{layer}"
            common = (
                "local_norm.weight",
                "local_norm.bias",
                "local_attention.to_q.weight",
                "local_attention.to_k.weight",
                "local_attention.to_v.weight",
                "local_attention.to_out.weight",
                "ffn_norm.weight",
                "ffn_norm.bias",
                "ffn_in.weight",
                "ffn_out.weight",
            )
            for suffix in common:
                _copy_checked(
                    destination,
                    parent_state,
                    f"{prefix}.{suffix}",
                    f"{prefix}.{suffix}",
                )
    # The recurrent memory projections and update gate are intentionally not
    # copied: the Full-Attention parent has no corresponding parameters.
    recurrent_keys = [
        key
        for key in destination
        if any(
            marker in key
            for marker in (
                "mem_q_proj",
                "mem_k_proj",
                "mem_v_proj",
                "mem_out_proj",
                "update_gate",
            )
        )
    ]
    return {
        "shared_parent_keys_loaded": len(destination) - len(recurrent_keys),
        "recurrent_keys_randomly_initialized": len(recurrent_keys),
        "recurrent_key_names": recurrent_keys,
    }


def build_model(method: str, device: torch.device, parent_state: dict[str, torch.Tensor], *, segment_length: int, memory_slots: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = FactMemoryConfig()
    if method == "FA-task":
        model = AdaptiveFactMemoryLM(
            config, memory_policy="none", attention_mode="full"
        ).to(device)
        missing, unexpected = model.load_state_dict(parent_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"FA parent load mismatch: missing={missing}, unexpected={unexpected}")
        # The episodic controller is not part of the common task objective.
        for name, parameter in model.named_parameters():
            if name.startswith(("fact_extractor.", "memory_controller.", "memory_reader.")):
                parameter.requires_grad_(False)
        return model, {
            "attention": "full causal",
            "parent_load": "strict exact",
            "adapter_parameters": 0,
        }
    if method == "SWA-task":
        model = AdaptiveFactMemoryLM(
            config, memory_policy="none", attention_mode="swa"
        ).to(device)
        missing, unexpected = model.load_state_dict(parent_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"SWA parent load mismatch: missing={missing}, unexpected={unexpected}")
        for name, parameter in model.named_parameters():
            if name.startswith(("fact_extractor.", "memory_controller.", "memory_reader.")):
                parameter.requires_grad_(False)
        return model, {
            "attention": "SWA(128,4)",
            "parent_load": "strict exact projection-compatible",
            "adapter_parameters": 0,
        }
    if method == "Memformer":
        model = MemformerDecoderLM(
            config,
            memory_slots=memory_slots,
            segment_length=segment_length,
            detach_memory=False,
        ).to(device)
        parent_metadata = load_common_parent_into_memformer(model, parent_state)
        return model, {
            "attention": "Memformer-style recurrent segment attention",
            "parent_load": "shared embedding/norm/token QKV/O/FFN mapped; recurrent projections fresh",
            "adapter_parameters": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if any(
                    marker in name
                    for marker in (
                        "mem_q_proj",
                        "mem_k_proj",
                        "mem_v_proj",
                        "mem_out_proj",
                        "update_gate",
                    )
                )
            ),
            **parent_metadata,
        }
    raise ValueError(f"unknown method: {method}")


def _round_forward(
    model: torch.nn.Module,
    method: str,
    collated,
    state,
    device: torch.device,
    segment_length: int,
):
    state_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if method in {"FA-task", "SWA-task"}:
        if state is None:
            state = model.initial_state(
                collated.input_ids.shape[0], device=device, dtype=state_dtype
            )
        output = model.forward_round(
            collated.input_ids,
            state=state,
            attention_mask=collated.attention_mask,
            source_roles=collated.roles,
            commit=False,
            hard_memory=False,
        )
    else:
        if state is None:
            state = model.initial_state(
                collated.input_ids.shape[0], device=device, dtype=state_dtype
            )
        output = model(
            collated.input_ids,
            state=state,
            attention_mask=collated.attention_mask,
            segment_length=segment_length,
        )
    return output


def shifted_task_losses(output, collated) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return full causal CE, answer CE, valid-token count and answer count."""

    logits = output.logits[:, :-1].float()
    targets = collated.input_ids[:, 1:]
    valid = collated.attention_mask[:, 1:]
    answer = collated.answer_target_mask[:, :-1] & valid
    if not bool(valid.any()):
        zero = output.logits.sum() * 0.0
        return zero, zero, 0, 0
    lm_loss = F.cross_entropy(logits[valid], targets[valid], reduction="mean")
    if bool(answer.any()):
        answer_loss = F.cross_entropy(logits[answer], targets[answer], reduction="mean")
        answer_count = int(answer.sum())
    else:
        answer_loss = lm_loss * 0.0
        answer_count = 0
    return lm_loss, answer_loss, int(valid.sum()), answer_count


def detach_state(state):
    return None if state is None else state.detach()


def run_train_step(
    model: torch.nn.Module,
    method: str,
    generator: SelectiveMemoryStressGenerator,
    indices: range,
    device: torch.device,
    *,
    segment_length: int,
    lambda_answer: float,
    delay_segments: int = 1,
    noise_rounds: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    examples, rounds = generator.batch(
        indices,
        device,
        delay_segments=delay_segments,
        noise_rounds=noise_rounds,
    )
    state = None
    lm_terms: list[torch.Tensor] = []
    answer_terms: list[torch.Tensor] = []
    valid_tokens = 0
    answer_tokens = 0
    answer_correct = 0
    answer_exact_rows = 0
    for collated in rounds:
        output = _round_forward(model, method, collated, state, device, segment_length)
        lm_loss, answer_loss, n_valid, n_answer = shifted_task_losses(output, collated)
        lm_terms.append(lm_loss)
        if n_answer:
            answer_terms.append(answer_loss)
            answer_positions = collated.answer_target_mask[:, :-1] & collated.attention_mask[:, 1:]
            predictions = output.logits[:, :-1].float().argmax(dim=-1)
            answer_correct += int((predictions[answer_positions] == collated.input_ids[:, 1:][answer_positions]).sum())
            for row in range(collated.input_ids.shape[0]):
                row_mask = answer_positions[row]
                if bool(row_mask.any()):
                    answer_exact_rows += int(
                        torch.equal(
                            predictions[row, row_mask],
                            collated.input_ids[row, 1:][row_mask],
                        )
                    )
        valid_tokens += n_valid
        answer_tokens += n_answer
        # The answer loss must be able to train how Memformer writes the
        # durable prefix into recurrent state.  Detaching here would leave
        # the query with a fixed, untrained state encoder.  FA/SWA have no
        # learned cross-round state, so their cache can be detached safely.
        state = output.state if method == "Memformer" else detach_state(output.state)
    lm_loss = torch.stack(lm_terms).mean()
    answer_loss = torch.stack(answer_terms).mean() if answer_terms else lm_loss * 0.0
    total = lm_loss + lambda_answer * answer_loss
    return total, {
        "lm_loss": float(lm_loss.detach().cpu()),
        "answer_loss": float(answer_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
        "valid_tokens": float(valid_tokens),
        "answer_tokens": float(answer_tokens),
        "answer_token_accuracy": float(answer_correct / max(answer_tokens, 1)),
        "answer_exact_batch": float(answer_exact_rows / max(len(examples), 1)),
        "state_bytes": float(state.state_bytes()) if method == "Memformer" else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    method: str,
    generator: SelectiveMemoryStressGenerator,
    device: torch.device,
    *,
    examples_count: int,
    batch_size: int,
    segment_length: int,
    lambda_answer: float,
    start_index: int = 1_000_000,
    split: str = "validation",
    delay_buckets: tuple[int, ...] = (1,),
    noise_buckets: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    model.eval()
    if examples_count <= 0:
        raise ValueError("evaluation set is empty")
    if not delay_buckets or not noise_buckets:
        raise ValueError("evaluation delay/noise buckets must be non-empty")

    all_rows: list[dict[str, float]] = []
    conditions: dict[str, dict[str, Any]] = {}
    for noise_index, noise in enumerate(noise_buckets):
        for delay_index, delay in enumerate(delay_buckets):
            rows: list[dict[str, float]] = []
            answer_losses: list[float] = []
            lm_losses: list[float] = []
            condition_start = start_index + noise_index * 100_000 + delay_index * 10_000
            for start in range(0, examples_count, batch_size):
                count = min(batch_size, examples_count - start)
                examples, rounds = generator.batch(
                    range(condition_start + start, condition_start + start + count),
                    device,
                    delay_segments=delay,
                    noise_rounds=noise,
                )
                state = None
                query_output = None
                query_collated = None
                for collated in rounds:
                    # Match the training dtype contract: states are allocated
                    # in BF16 on CUDA, so evaluation must enter the same
                    # autocast context before projecting FP32 parameters into
                    # the cache.
                    with autocast_context(device):
                        output = _round_forward(
                            model, method, collated, state, device, segment_length
                        )
                    lm_loss, answer_loss, _, n_answer = shifted_task_losses(output, collated)
                    lm_losses.append(float(lm_loss.cpu()))
                    if n_answer:
                        answer_losses.append(float(answer_loss.cpu()))
                    state = detach_state(output.state)
                    query_output = output
                    query_collated = collated
                assert query_output is not None and query_collated is not None
                logits = query_output.logits.float()
                for row, _example in enumerate(examples):
                    positions = torch.nonzero(
                        query_collated.answer_target_mask[row, :-1], as_tuple=False
                    ).flatten()
                    targets = query_collated.input_ids[row, positions + 1]
                    predictions = logits[row, positions].argmax(dim=-1)
                    exact = bool(torch.equal(predictions, targets))
                    first_logits = logits[row, positions[0]]
                    first_target = int(targets[0])
                    log_probs = F.log_softmax(first_logits, dim=-1)
                    rank = 1 + int((first_logits > first_logits[first_target]).sum())
                    rows.append(
                        {
                            "index": float(condition_start + start + row),
                            "delay_segments": float(delay),
                            "noise_rounds": float(noise),
                            "answer_exact": float(exact),
                            "first_token_top1": float(int(predictions[0]) == first_target),
                            "first_token_top5": float(rank <= 5),
                            "first_token_rank": float(rank),
                            "first_token_nll": float(-log_probs[first_target]),
                            "answer_token_count": float(targets.numel()),
                        }
                    )
            if not rows:
                raise ValueError("evaluation condition produced no rows")
            mean = lambda key: statistics.fmean(float(row[key]) for row in rows)
            key = f"noise_{noise}_delay_{delay}"
            conditions[key] = {
                "delay_segments": delay,
                "noise_rounds": noise,
                "examples": len(rows),
                "answer_exact": mean("answer_exact"),
                "first_token_top1": mean("first_token_top1"),
                "first_token_top5": mean("first_token_top5"),
                "first_token_rank": mean("first_token_rank"),
                "first_token_nll": mean("first_token_nll"),
                "answer_nll": statistics.fmean(answer_losses) if answer_losses else None,
                "lm_nll": statistics.fmean(lm_losses) if lm_losses else None,
                "objective": (
                    statistics.fmean(lm_losses)
                    + lambda_answer * statistics.fmean(answer_losses)
                    if lm_losses and answer_losses
                    else None
                ),
                "rows": rows,
            }
            all_rows.extend(rows)
    if not all_rows:
        raise ValueError("evaluation set is empty")
    mean = lambda key: statistics.fmean(float(row[key]) for row in all_rows)
    all_answer_losses = [
        float(condition["answer_nll"])
        for condition in conditions.values()
        if condition["answer_nll"] is not None
    ]
    all_lm_losses = [
        float(condition["lm_nll"])
        for condition in conditions.values()
        if condition["lm_nll"] is not None
    ]
    # Keep scalar aliases for the one-cell M8 output while exposing the full
    # condition matrix required by M9.
    one_cell = len(conditions) == 1
    return {
        "examples": len(all_rows),
        "split": split,
        "delay_segments": delay_buckets[0] if one_cell else list(delay_buckets),
        "noise_rounds": noise_buckets[0] if one_cell else list(noise_buckets),
        "answer_exact": mean("answer_exact"),
        "first_token_top1": mean("first_token_top1"),
        "first_token_top5": mean("first_token_top5"),
        "first_token_rank": mean("first_token_rank"),
        "first_token_nll": mean("first_token_nll"),
        "answer_nll": statistics.fmean(all_answer_losses) if all_answer_losses else None,
        "lm_nll": statistics.fmean(all_lm_losses) if all_lm_losses else None,
        "objective": (
            statistics.fmean(all_lm_losses) + lambda_answer * statistics.fmean(all_answer_losses)
            if all_lm_losses and all_answer_losses
            else None
        ),
        "conditions": conditions,
        "rows": all_rows,
    }


def run(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_all(args.seed)
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"tokenizer directory is missing: {TOKENIZER_DIR}")
    if not FILLER_PATH.exists():
        raise FileNotFoundError(f"filler cache is missing: {FILLER_PATH}")
    run_dir = Path(args.runs_dir) / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    parent_state, parent_payload = load_parent_state()
    model, model_metadata = build_model(
        args.method,
        device,
        parent_state,
        segment_length=args.segment_length,
        memory_slots=args.memory_slots,
    )
    train_generator = SelectiveMemoryStressGenerator(
        TOKENIZER_DIR,
        FILLER_PATH,
        split="train",
        seed=args.seed,
        delay_buckets=args.train_delays,
        noise_buckets=args.train_noise,
        segment_length=args.segment_length,
        answer_leading_space=True,
    )
    valid_generator = SelectiveMemoryStressGenerator(
        TOKENIZER_DIR,
        FILLER_PATH,
        split="validation",
        seed=args.seed + 10_000,
        delay_buckets=args.eval_delays,
        noise_buckets=args.eval_noise,
        segment_length=args.segment_length,
        answer_leading_space=True,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )

    resolved: dict[str, Any] = {
        "protocol_version": "adaptive_color_common_objective_v2",
        "milestone": args.milestone,
        "run_id": args.run_id,
        "method": args.method,
        "seed": args.seed,
        "parent_checkpoint": str(PARENT_CHECKPOINT),
        "parent_checkpoint_sha256": sha256_file(PARENT_CHECKPOINT),
        "architecture": {
            "layers": model.config.layers,
            "hidden_size": model.config.hidden_size,
            "heads": model.config.heads,
            "ffn_size": model.config.ffn_size,
            "segment_length": args.segment_length,
            "memory_slots": args.memory_slots if args.method == "Memformer" else None,
            **model_metadata,
        },
        "data": {
            "generator": "SelectiveMemoryStressGenerator",
            "random_name_to_color": True,
            "train_examples_fixed": args.train_examples,
            "validation_examples": args.eval_examples,
            "train_delays": list(args.train_delays),
            "eval_delays": list(args.eval_delays),
            "train_noise_rounds": list(args.train_noise),
            "eval_noise_rounds": list(args.eval_noise),
            "answer_leading_space": True,
            "tokenizer": str(TOKENIZER_DIR),
            "filler": str(FILLER_PATH),
        },
        "objective": {
            "formula": "L_full_causal_lm + lambda_answer * L_answer",
            "lambda_answer": args.lambda_answer,
            "state_detach_between_rounds": args.method != "Memformer",
            "memory_to_token_reconstruction": False,
            "value_token_alignment": False,
            "fact_labels": False,
            "pointer_copy_head": False,
        },
        "optimization": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "optimizer": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "gradient_clip_norm": 1.0,
            "precision": "CUDA BF16 autocast / FP32 parameters" if device.type == "cuda" else "FP32",
        },
        "request": vars(args),
        "environment": environment_record(device),
        "parent_payload_step": parent_payload.get("step"),
        "code_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__),
                ROOT / "adaptive_gated_fact_memory" / "scripts" / "synthetic_memory_stress.py",
                ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "memformer.py",
                ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "model.py",
                ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "attention.py",
            )
            if path.exists()
        },
    }
    resolved["config_hash"] = stable_hash(resolved)
    atomic_json(run_dir / "config.resolved.json", resolved)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    status = "ok"
    failure: str | None = None
    best_total = math.inf
    best_step = None
    best_state_cpu: dict[str, torch.Tensor] | None = None
    model.train()
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as stream:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            aggregate: dict[str, float] = {}
            try:
                for accumulation in range(args.gradient_accumulation_steps):
                    logical_index = (
                        (step - 1) * args.gradient_accumulation_steps + accumulation
                    )
                    first = (logical_index * args.batch_size) % args.train_examples
                    indices = [
                        (first + offset) % args.train_examples
                        for offset in range(args.batch_size)
                    ]
                    delay = args.train_delays[logical_index % len(args.train_delays)]
                    noise = args.train_noise[
                        (logical_index // len(args.train_delays)) % len(args.train_noise)
                    ]
                    with autocast_context(device):
                        loss, scalar = run_train_step(
                            model,
                            args.method,
                            train_generator,
                            indices,
                            device,
                            segment_length=args.segment_length,
                            lambda_answer=args.lambda_answer,
                            delay_segments=delay,
                            noise_rounds=noise,
                        )
                    (loss / args.gradient_accumulation_steps).backward()
                    for name, value in scalar.items():
                        aggregate[name] = aggregate.get(name, 0.0) + value / args.gradient_accumulation_steps
                    del loss
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                if not math.isfinite(grad_norm) or not math.isfinite(aggregate.get("total", math.inf)):
                    raise FloatingPointError(
                        f"non-finite loss/gradient: loss={aggregate.get('total')}, grad={grad_norm}"
                    )
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
            record = {
                "event": "train_step",
                "step": step,
                "gradient_norm": grad_norm,
                "step_seconds": time.perf_counter() - step_started,
                "examples_seen": step * args.batch_size * args.gradient_accumulation_steps,
                **aggregate,
            }
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if float(record["total"]) < best_total:
                best_total = float(record["total"])
                best_step = int(record["step"])
                best_state_cpu = {
                    name: parameter.detach().cpu()
                    for name, parameter in model.state_dict().items()
                }

    if status == "ok":
        model_cpu = {name: parameter.detach().cpu() for name, parameter in model.state_dict().items()}
        torch.save(
            {
                "model": model_cpu,
                "optimizer": optimizer.state_dict(),
                "step": len(records),
                "config_hash": resolved["config_hash"],
            },
            run_dir / "checkpoint.final.pt",
        )
        if best_state_cpu is not None:
            torch.save(
                {
                    "model": best_state_cpu,
                    "step": best_step,
                    "config_hash": resolved["config_hash"],
                },
                run_dir / "checkpoint.best.pt",
            )
    elapsed = time.perf_counter() - started
    evaluation = None
    train_evaluation = None
    if status == "ok":
        train_evaluation = evaluate(
            model,
            args.method,
            train_generator,
            device,
            examples_count=args.train_examples,
            batch_size=args.batch_size,
            segment_length=args.segment_length,
            lambda_answer=args.lambda_answer,
            start_index=0,
            split="train_fixed",
            delay_buckets=args.train_delays,
            noise_buckets=args.train_noise,
        )
        evaluation = evaluate(
            model,
            args.method,
            valid_generator,
            device,
            examples_count=args.eval_examples,
            batch_size=args.batch_size,
            segment_length=args.segment_length,
            lambda_answer=args.lambda_answer,
            start_index=1_000_000,
            split="validation_heldout",
            delay_buckets=args.eval_delays,
            noise_buckets=args.eval_noise,
        )
    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    summary = {
        "protocol_version": resolved["protocol_version"],
        "milestone": args.milestone,
        "run_id": args.run_id,
        "method": args.method,
        "seed": args.seed,
        "status": status,
        "failure": failure,
        "steps_completed": len(records),
        "elapsed_training_seconds": elapsed,
        "best_training_total_loss": min(
            (float(record["total"]) for record in records), default=None
        ),
        "best_training_step": best_step,
        "first_training_total_loss": records[0]["total"] if records else None,
        "last_training": records[-1] if records else None,
        "evaluation": evaluation,
        "train_evaluation": train_evaluation,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "config_hash": resolved["config_hash"],
        "run_dir": str(run_dir),
    }
    atomic_json(run_dir / "summary.json", summary)
    return summary


def parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("FA-task", "SWA-task", "Memformer"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--milestone", default="M8")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--train-examples", type=int, default=8)
    parser.add_argument("--eval-examples", type=int, default=24)
    parser.add_argument("--train-delays", type=parse_int_tuple, default=(1,))
    parser.add_argument("--train-noise", type=parse_int_tuple, default=(0,))
    parser.add_argument("--eval-delays", type=parse_int_tuple, default=(1,))
    parser.add_argument("--eval-noise", type=parse_int_tuple, default=(0,))
    parser.add_argument("--segment-length", type=int, default=32)
    parser.add_argument("--memory-slots", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--lambda-answer", type=float, default=1.0)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.batch_size <= 0
        or args.gradient_accumulation_steps <= 0
        or args.train_examples < args.batch_size
    ):
        parser.error(
            "steps/batch-size/gradient accumulation must be positive and "
            "train-examples >= batch-size"
        )
    return args


def main() -> None:
    args = parse_args()
    configure_cuda()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    summary = run(args, device)
    print(
        json.dumps(
            {
                key: summary.get(key)
                for key in (
                    "run_id",
                    "method",
                    "status",
                    "steps_completed",
                    "elapsed_training_seconds",
                    "evaluation",
                    "peak_allocated_bytes",
                )
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
