"""Train the adaptive fact-memory backbone on the pinned TinyStories stream.

This is the first (language-model) stage of the proposed architecture.  The
dataset contains independent stories and no user/assistant or fact-span labels,
so the run uses :meth:`AdaptiveFactMemoryLM.forward_tinystories`: every packed
block starts with a fresh bounded KV state, the episodic memory path is disabled,
and only the shared embedding/local attention/MLP backbone receives the LM
loss.  The resulting checkpoint is intended as initialization for the later
structured-dialogue memory stage; it must not be reported as evidence of fact
retrieval quality.

Example (RTX 4090, the frozen screening budget)::

    PYTHONPATH=adaptive_gated_fact_memory/src \
      python adaptive_gated_fact_memory/scripts/train_tinystories.py \
      --run-id tinystories_backbone_s17 \
      --train-tokens 10000000 --full-validation

Generated caches, metrics and checkpoints default to the writable data disk
under ``/root/autodl-tmp/26summerBDMI_transformer``.  Set the corresponding
environment variables to override those locations.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

# The existing protocol helper reads these variables at import time.
os.environ.setdefault(
    "TINYSTORIES_DATA_DIR",
    str(DATA_DISK / "data/raw/tinystories/data"),
)
os.environ.setdefault(
    "TINYSTORIES_TOKENIZER_DIR",
    str(DATA_DISK / "data/tokenizer/gpt2"),
)
os.environ.setdefault(
    "TINYSTORIES_CACHE_DIR",
    str(DATA_DISK / "data/cache/tinystories_tinylm_v1"),
)
os.environ.setdefault(
    "ADAPTIVE_TINYSTORIES_RUNS_DIR",
    str(DATA_DISK / "runs/adaptive_gated_fact_memory_tinystories"),
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adaptive_gated_fact_memory" / "src"))
from experiments.tinystories_tinylm_v1 import run_person_b as protocol  # noqa: E402
from adaptive_fact_memory import AdaptiveFactMemoryLM, FactMemoryConfig  # noqa: E402


RUNS_DIR = Path(
    os.environ.get(
        "ADAPTIVE_TINYSTORIES_RUNS_DIR",
        str(DATA_DISK / "runs/adaptive_gated_fact_memory_tinystories"),
    )
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is optional for the prototype
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_argument(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    # Match the existing TinyStories screening protocol.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def environment_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        record.update(
            {
                "gpu": props.name,
                "gpu_total_memory_bytes": props.total_memory,
                "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return record


def parameter_record(model: torch.nn.Module) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    memory_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if any(
            marker in name
            for marker in (
                "fact_extractor",
                "memory_controller",
                "memory_reader",
                "memory_norm",
                "memory_fusion",
            )
        )
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "memory_controller_parameters": memory_parameters,
        "backbone_parameters": total - memory_parameters,
        "parameter_dtype": "float32",
        "fp32_parameter_bytes": total * 4,
        "bf16_parameter_bytes": total * 2,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def masked_loss_sum(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    return (losses * valid.reshape(-1)).sum()


@torch.no_grad()
def evaluate(
    model: AdaptiveFactMemoryLM,
    stream,
    prediction_limit: int,
    batch_sequences: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    prediction_limit = min(int(prediction_limit), stream.prediction_count)
    consumed = 0
    block_cursor = 0
    total_loss = 0.0
    started = time.perf_counter()
    while consumed < prediction_limit:
        remaining = prediction_limit - consumed
        valid_predictions = min(batch_sequences * stream.context, remaining)
        sequences = math.ceil(valid_predictions / stream.context)
        inputs, targets, valid = stream.batch(
            block_cursor, sequences, valid_predictions, device
        )
        block_cursor += sequences
        with autocast_context(device):
            logits = model.forward_tinystories(inputs)
        total_loss += float(masked_loss_sum(logits, targets, valid).detach().cpu())
        consumed += valid_predictions
    elapsed = time.perf_counter() - started
    nll = total_loss / max(consumed, 1)
    model.train()
    return {
        "tokens": consumed,
        "token_weighted_nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "elapsed_seconds": elapsed,
        "tokens_per_second": consumed / max(elapsed, 1e-9),
    }


def learning_rate_for_step(
    step: int,
    total_steps: int,
    base_lr: float,
    minimum_lr: float,
    warmup_ratio: float,
) -> float:
    warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))
    if step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return minimum_lr + (base_lr - minimum_lr) * cosine


@dataclass(frozen=True)
class TrainRequest:
    run_id: str
    seed: int
    train_tokens: int
    context: int
    micro_batch_sequences: int
    gradient_accumulation_steps: int
    validation_probe_tokens: int
    validation_batch_sequences: int
    validation_interval_tokens: int
    checkpoint_interval_tokens: int
    full_validation: bool
    save_checkpoint: bool


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    config_hash: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "config_hash": config_hash,
        },
        temporary,
    )
    temporary.replace(path)


def run_training(request: TrainRequest, device: torch.device) -> dict[str, Any]:
    if request.context <= 1:
        raise ValueError("context must be greater than one")
    effective_batch_tokens = (
        request.context
        * request.micro_batch_sequences
        * request.gradient_accumulation_steps
    )
    if effective_batch_tokens != 8192:
        raise ValueError(
            "this first run follows the frozen protocol: context * micro_batch "
            f"* accumulation must equal 8192, got {effective_batch_tokens}"
        )

    required_train_cache_tokens = (
        math.ceil(request.train_tokens / request.context) * request.context + 1
    )
    train_path, train_metadata = protocol.build_token_cache(
        "train", required_train_cache_tokens
    )
    validation_path, validation_metadata = protocol.build_token_cache(
        "validation", None
    )
    train_stream = protocol.PackedTokenStream(
        train_path, int(train_metadata["token_count"]), request.context
    )
    validation_stream = protocol.PackedTokenStream(
        validation_path, int(validation_metadata["token_count"]), request.context
    )

    seed_all(request.seed)
    config = FactMemoryConfig()
    model = AdaptiveFactMemoryLM(config).to(device)
    parameters = parameter_record(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    total_steps = math.ceil(request.train_tokens / effective_batch_tokens)
    run_dir = RUNS_DIR / request.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    code_paths = [
        Path(__file__),
        ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "model.py",
        ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "attention.py",
        ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "memory.py",
        ROOT / "adaptive_gated_fact_memory" / "src" / "adaptive_fact_memory" / "config.py",
    ]
    code_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths}
    manifest_path = ROOT / "data" / "tinystories_manifest.json"
    config_payload: dict[str, Any] = {
        "protocol_version": "tinystories_tinylm_v1",
        "stage": "stage1_tinystories_backbone_pretraining",
        "run": asdict(request),
        "architecture": asdict(config),
        "memory_training_status": {
            "episodic_path_enabled": False,
            "fact_supervision_available": False,
            "memory_parameters_updated": False,
            "reason": "TinyStories has no round/fact annotations",
            "future_stage": "structured_dialogue_fact_memory_training",
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 3.0e-4,
            "minimum_learning_rate": 3.0e-5,
            "betas": [0.9, 0.95],
            "epsilon": 1.0e-8,
            "weight_decay": 0.1,
            "warmup_ratio": 0.03,
            "schedule": "cosine",
            "gradient_clip_norm": 1.0,
            "effective_batch_tokens": effective_batch_tokens,
            "compute_dtype": "bfloat16" if device.type == "cuda" else "float32",
            "parameter_dtype": "float32",
        },
        "parameters": parameters,
        "data": {
            "train_cache": train_metadata,
            "validation_cache": validation_metadata,
            "data_order": "pinned_parquet_order_sequential_512_token_blocks",
            "cross_document_packing": True,
            "eos_at_document_end": True,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path)
            if manifest_path.exists()
            else None,
        },
        "storage": {
            "runs_dir": str(RUNS_DIR),
            "data_dir": str(protocol.DATA_DIR),
            "tokenizer_dir": str(protocol.TOKENIZER_DIR),
            "cache_dir": str(protocol.CACHE_DIR),
        },
        "environment": environment_record(device),
        "code_hashes": code_hashes,
    }
    config_payload["config_hash"] = stable_hash(config_payload)
    atomic_json(run_dir / "config.resolved.json", config_payload)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    initial_validation = evaluate(
        model,
        validation_stream,
        request.validation_probe_tokens,
        request.validation_batch_sequences,
        device,
    )
    model.train()
    best_probe = dict(initial_validation)
    best_probe["step"] = 0
    best_state_cpu = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    best_checkpoint_step = 0
    tokens_seen = 0
    block_cursor = 0
    next_validation = request.validation_interval_tokens
    next_checkpoint = request.checkpoint_interval_tokens
    started = time.perf_counter()
    status = "ok"
    failure: str | None = None
    last_step = 0

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        metrics_file.write(
            json.dumps(
                {
                    "event": "initial_validation_probe",
                    "step": 0,
                    "tokens_seen": 0,
                    **initial_validation,
                }
            )
            + "\n"
        )
        metrics_file.flush()

        for step in range(1, total_steps + 1):
            last_step = step
            step_target = min(effective_batch_tokens, request.train_tokens - tokens_seen)
            optimizer.zero_grad(set_to_none=True)
            step_loss_sum = 0.0
            step_predictions = 0
            step_started = time.perf_counter()
            try:
                while step_predictions < step_target:
                    micro_limit = request.micro_batch_sequences * request.context
                    valid_predictions = min(micro_limit, step_target - step_predictions)
                    sequences = math.ceil(valid_predictions / request.context)
                    inputs, targets, valid = train_stream.batch(
                        block_cursor, sequences, valid_predictions, device
                    )
                    block_cursor += sequences
                    with autocast_context(device):
                        logits = model.forward_tinystories(inputs)
                    loss_sum = masked_loss_sum(logits, targets, valid)
                    (loss_sum / step_target).backward()
                    step_loss_sum += float(loss_sum.detach().cpu())
                    step_predictions += valid_predictions
                    del logits, loss_sum, inputs, targets, valid

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    )
                )
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm: {grad_norm}")
                learning_rate = learning_rate_for_step(
                    step, total_steps, 3.0e-4, 3.0e-5, 0.03
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
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

            tokens_seen += step_target
            step_seconds = time.perf_counter() - step_started
            train_nll = step_loss_sum / max(step_target, 1)
            record: dict[str, Any] = {
                "event": "train_step",
                "step": step,
                "tokens_seen": tokens_seen,
                "train_nll": train_nll,
                "train_ppl": math.exp(min(train_nll, 20.0)),
                "gradient_norm": grad_norm,
                "learning_rate": learning_rate,
                "step_seconds": step_seconds,
                "tokens_per_second": step_target / max(step_seconds, 1e-9),
            }

            should_validate = (
                tokens_seen >= next_validation or tokens_seen == request.train_tokens
            )
            if should_validate:
                probe = evaluate(
                    model,
                    validation_stream,
                    request.validation_probe_tokens,
                    request.validation_batch_sequences,
                    device,
                )
                model.train()
                record["validation_probe"] = probe
                if probe["token_weighted_nll"] < best_probe["token_weighted_nll"]:
                    best_probe = dict(probe)
                    best_probe["step"] = step
                    best_checkpoint_step = step
                    best_state_cpu = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                while next_validation <= tokens_seen:
                    next_validation += request.validation_interval_tokens

            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

            should_checkpoint = (
                tokens_seen >= next_checkpoint or tokens_seen == request.train_tokens
            )
            if should_checkpoint and request.save_checkpoint:
                save_checkpoint(
                    run_dir / f"checkpoint.tokens_{tokens_seen}.pt",
                    model,
                    optimizer,
                    step,
                    tokens_seen,
                    config_payload["config_hash"],
                )
                while next_checkpoint <= tokens_seen:
                    next_checkpoint += request.checkpoint_interval_tokens

    elapsed = time.perf_counter() - started
    final_probe = evaluate(
        model,
        validation_stream,
        request.validation_probe_tokens,
        request.validation_batch_sequences,
        device,
    )
    model.train()
    full_validation = None
    best_full_validation = None
    if request.full_validation and status == "ok":
        full_validation = evaluate(
            model,
            validation_stream,
            validation_stream.prediction_count,
            request.validation_batch_sequences,
            device,
        )
        model.train()
        final_state_cpu = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        if best_checkpoint_step == last_step:
            best_full_validation = dict(full_validation)
        else:
            model.load_state_dict(best_state_cpu)
            best_full_validation = evaluate(
                model,
                validation_stream,
                validation_stream.prediction_count,
                request.validation_batch_sequences,
                device,
            )
            model.train()
            model.load_state_dict(final_state_cpu)
            model.train()

    # Always retain explicit final and probe-selected weights, even if the
    # interval was larger than the requested run.
    if request.save_checkpoint:
        save_checkpoint(
            run_dir / "checkpoint.final.pt",
            model,
            optimizer,
            last_step,
            tokens_seen,
            config_payload["config_hash"],
        )
        if best_checkpoint_step != last_step:
            model.load_state_dict(best_state_cpu)
            save_checkpoint(
                run_dir / "checkpoint.best_probe.pt",
                model,
                optimizer,
                best_checkpoint_step,
                int(best_probe.get("tokens_seen", 0))
                if "tokens_seen" in best_probe
                else 0,
                config_payload["config_hash"],
            )
            model.load_state_dict(final_state_cpu if "final_state_cpu" in locals() else best_state_cpu)

    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    summary = {
        "protocol_version": "tinystories_tinylm_v1",
        "stage": "stage1_tinystories_backbone_pretraining",
        "run_id": request.run_id,
        "status": status,
        "failure": failure,
        "seed": request.seed,
        "train_tokens_requested": request.train_tokens,
        "train_tokens_completed": tokens_seen,
        "optimizer_steps_completed": last_step,
        "elapsed_training_seconds": elapsed,
        "mean_training_tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        "initial_validation_probe": initial_validation,
        "final_validation_probe": final_probe,
        "best_validation_probe": best_probe,
        "best_checkpoint_step": best_checkpoint_step,
        "full_validation": full_validation,
        "best_probe_full_validation": best_full_validation,
        "parameters": parameters,
        "memory_training_status": config_payload["memory_training_status"],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "config_hash": config_payload["config_hash"],
        "run_dir": str(run_dir),
    }
    atomic_json(run_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-tokens", type=int, default=10_000_000)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--micro-batch-sequences", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--validation-probe-tokens", type=int, default=262_144)
    parser.add_argument("--validation-batch-sequences", type=int, default=8)
    parser.add_argument("--validation-interval-tokens", type=int, default=1_048_576)
    parser.add_argument("--checkpoint-interval-tokens", type=int, default=5_000_000)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_argument(args.device)
    configure_cuda()
    request = TrainRequest(
        run_id=args.run_id,
        seed=args.seed,
        train_tokens=args.train_tokens,
        context=args.context,
        micro_batch_sequences=args.micro_batch_sequences,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        validation_probe_tokens=args.validation_probe_tokens,
        validation_batch_sequences=args.validation_batch_sequences,
        validation_interval_tokens=args.validation_interval_tokens,
        checkpoint_interval_tokens=args.checkpoint_interval_tokens,
        full_validation=args.full_validation,
        save_checkpoint=not args.no_checkpoint,
    )
    summary = run_training(request, device)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
