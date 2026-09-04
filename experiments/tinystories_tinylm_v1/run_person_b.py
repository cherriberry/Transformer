"""Run the Person-B TinyLM experiments: causal Longformer and Memformer.

This runner is intentionally scoped to the frozen three-day TinyStories
screening protocol.  It uses the same decoder-only TinyLM backbone, GPT-2
tokenizer stream, optimizer, token order, and validation rules for both
methods.  Historical PG-19/synthetic pilots are never loaded as quality data.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROFILE_PATH = EXPERIMENT_DIR / "three_day_validation_screening.yaml"
DATA_DIR = ROOT / "data" / "raw" / "tinystories" / "data"
TOKENIZER_DIR = ROOT / "data" / "tokenizer" / "gpt2"
CACHE_DIR = ROOT / "data" / "cache" / "tinystories_tinylm_v1"
RUNS_DIR = EXPERIMENT_DIR / "runs" / "person_b"
AGGREGATE_DIR = EXPERIMENT_DIR / "aggregate"

sys.path.insert(0, str(ROOT))
from b_role_experiments import MemformerSegmentAttention  # noqa: E402
from models.longformer_attention import LongformerSelfAttention  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_profile() -> dict[str, Any]:
    with PROFILE_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
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


def cache_paths(split: str, requested_tokens: int | None) -> tuple[Path, Path]:
    suffix = "all" if requested_tokens is None else str(requested_tokens)
    binary = CACHE_DIR / f"{split}_{suffix}.int32.bin"
    return binary, binary.with_suffix(binary.suffix + ".json")


def ordered_parquet_files(split: str) -> list[Path]:
    files = sorted(DATA_DIR.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no local TinyStories parquet files for split={split}")
    return files


def build_token_cache(
    split: str,
    requested_tokens: int | None,
    *,
    batch_rows: int = 512,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Tokenize in pinned parquet order, append EOS per story, and cache int32."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    binary, metadata_path = cache_paths(split, requested_tokens)
    if binary.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_bytes = int(metadata["token_count"]) * np.dtype(np.int32).itemsize
        if binary.stat().st_size == expected_bytes:
            return binary, metadata

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    if len(tokenizer) != 50257 or tokenizer.eos_token_id != 50256:
        raise RuntimeError("local tokenizer does not match the pinned GPT-2 tokenizer")

    temporary = binary.with_suffix(binary.suffix + ".tmp")
    count = 0
    rows = 0
    started = time.perf_counter()
    with temporary.open("wb") as output:
        stop = False
        for parquet_path in ordered_parquet_files(split):
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches(batch_size=batch_rows, columns=["text"]):
                texts = batch.column(0).to_pylist()
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                for token_ids in encoded:
                    rows += 1
                    values = np.asarray([*token_ids, tokenizer.eos_token_id], dtype=np.int32)
                    if requested_tokens is not None:
                        values = values[: max(0, requested_tokens - count)]
                    values.tofile(output)
                    count += int(values.size)
                    if requested_tokens is not None and count >= requested_tokens:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
    if requested_tokens is not None and count != requested_tokens:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"TinyStories {split} ended at {count:,} tokens; requested {requested_tokens:,}"
        )
    temporary.replace(binary)
    metadata = {
        "protocol_version": "tinystories_tinylm_v1",
        "split": split,
        "requested_tokens": requested_tokens,
        "token_count": count,
        "rows_consumed": rows,
        "dtype": "int32",
        "bytes": binary.stat().st_size,
        "parquet_files": [str(path.relative_to(ROOT)) for path in ordered_parquet_files(split)],
        "tokenizer": "openai-community/gpt2",
        "tokenizer_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "eos_token_id": tokenizer.eos_token_id,
        "add_eos_at_document_end": True,
        "cross_document_packing": True,
        "elapsed_seconds": time.perf_counter() - started,
        "created_at_utc": utc_now(),
    }
    atomic_json(metadata_path, metadata)
    return binary, metadata


class PackedTokenStream:
    def __init__(self, binary_path: Path, token_count: int, context: int, eos: int = 50256):
        self.path = binary_path
        self.tokens = np.memmap(binary_path, mode="r", dtype=np.int32, shape=(token_count,))
        self.context = context
        self.eos = eos

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def prediction_count(self) -> int:
        return max(0, self.token_count - 1)

    def batch(
        self,
        block_start: int,
        batch_sequences: int,
        valid_predictions: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = self.context
        packed = np.full((batch_sequences, length + 1), self.eos, dtype=np.int64)
        for row in range(batch_sequences):
            offset = (block_start + row) * length
            available = max(0, min(length + 1, self.token_count - offset))
            if available:
                packed[row, :available] = self.tokens[offset : offset + available]
        inputs = torch.from_numpy(packed[:, :-1]).to(device=device, non_blocking=True)
        targets = torch.from_numpy(packed[:, 1:]).to(device=device, non_blocking=True)
        valid = torch.arange(batch_sequences * length, device=device) < valid_predictions
        return inputs, targets, valid.view(batch_sequences, length)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 32768, base: float = 10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    @staticmethod
    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        left = value[..., ::2]
        right = value[..., 1::2]
        return torch.stack((-right, left), dim=-1).flatten(-2)

    def apply_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.numel() and int(position_ids.max()) >= self.max_position:
            raise ValueError("position exceeds configured RoPE maximum")
        frequencies = torch.outer(position_ids.float(), self.inv_freq)
        frequencies = torch.repeat_interleave(frequencies, repeats=2, dim=-1)
        cosine = frequencies.cos().to(dtype=query.dtype)[None, None, :, :]
        sine = frequencies.sin().to(dtype=query.dtype)[None, None, :, :]
        return (
            query * cosine + self.rotate_half(query) * sine,
            key * cosine + self.rotate_half(key) * sine,
        )


class FullCausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        value: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, _ = value.shape
        reshape = lambda tensor: tensor.view(  # noqa: E731
            batch, tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        query, key, projected_value = map(
            reshape, (self.to_q(value), self.to_k(value), self.to_v(value))
        )
        query, key = rotary_emb.apply_qk(query, key, position_ids)
        attended = F.scaled_dot_product_attention(
            query, key, projected_value, dropout_p=0.0, is_causal=True
        )
        return self.to_out(attended.transpose(1, 2).reshape(batch, tokens, self.dim))


class TinyBlock(nn.Module):
    def __init__(self, config: "ModelConfig", method: str, method_value: int):
        super().__init__()
        self.method = method
        self.ln1 = nn.LayerNorm(config.hidden_size, elementwise_affine=True, bias=True)
        if method == "longformer":
            self.attn = LongformerSelfAttention(
                config.hidden_size,
                config.context,
                heads=config.heads,
                window_size=2 * method_value + 1,
                global_tokens=0,
                dropout=0.0,
                bias=False,
                output_bias=False,
                causal=True,
                query_chunk_size=config.longformer_query_chunk,
            )
        elif method == "memformer":
            self.attn = MemformerSegmentAttention(
                config.hidden_size,
                config.heads,
                memory_slots=method_value,
                causal=True,
                detach_memory=False,
                bias=False,
            )
        elif method == "full_attention":
            self.attn = FullCausalAttention(config.hidden_size, config.heads)
        else:
            raise ValueError(f"unsupported method: {method}")
        self.ln2 = nn.LayerNorm(config.hidden_size, elementwise_affine=True, bias=True)
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.fc2 = nn.Linear(config.ffn_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def residual_mlp(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.dropout(self.fc2(F.gelu(self.fc1(self.ln2(hidden)))))


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    layers: int = 6
    hidden_size: int = 384
    heads: int = 8
    ffn_size: int = 1536
    context: int = 512
    rope_max_position: int = 32768
    dropout: float = 0.1
    longformer_query_chunk: int = 128
    memformer_segment_length: int = 128


class TinyLM(nn.Module):
    def __init__(self, config: ModelConfig, method: str, method_value: int, seed: int):
        super().__init__()
        self.config = config
        self.method = method
        self.method_value = method_value
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RotaryEmbedding(config.hidden_size // config.heads, config.rope_max_position)
        self.blocks = nn.ModuleList(
            [TinyBlock(config, method, method_value) for _ in range(config.layers)]
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size, elementwise_affine=True, bias=True
        )
        self.reset_protocol_parameters(seed)

    @staticmethod
    def canonical_parameter_name(name: str) -> str:
        replacements = {
            ".attn.q_proj.": ".attn.to_q.",
            ".attn.k_proj.": ".attn.to_k.",
            ".attn.v_proj.": ".attn.to_v.",
            ".attn.out_proj.": ".attn.to_out.",
        }
        for old, new in replacements.items():
            name = name.replace(old, new)
        return name

    def reset_protocol_parameters(self, seed: int) -> None:
        for name, parameter in self.named_parameters():
            canonical = self.canonical_parameter_name(name)
            derived = int(hashlib.sha256(f"{seed}:{canonical}".encode()).hexdigest()[:16], 16)
            generator = torch.Generator(device="cpu").manual_seed(derived % (2**63 - 1))
            if "ln" in name or name.startswith("final_norm"):
                if name.endswith("weight"):
                    nn.init.ones_(parameter)
                else:
                    nn.init.zeros_(parameter)
            elif parameter.ndim >= 2:
                nn.init.normal_(parameter, mean=0.0, std=0.02, generator=generator)
            else:
                nn.init.zeros_(parameter)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(token_ids)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        for block in self.blocks:
            normalized = block.ln1(hidden)
            if self.method == "longformer":
                attended = block.attn(
                    normalized,
                    rotary_emb=self.rope,
                    position_ids=positions,
                )
            elif self.method == "memformer":
                attended, _ = block.attn(
                    normalized,
                    segment_length=self.config.memformer_segment_length,
                    memory=None,
                    rotary_emb=self.rope,
                    position_offset=0,
                )
            else:
                attended = block.attn(normalized, self.rope, positions)
            hidden = hidden + block.dropout(attended)
            hidden = block.residual_mlp(hidden)
        hidden = self.final_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight)


def parameter_record(model: nn.Module) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    attention = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".attn." in name
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "attention_parameters": attention,
        "fp32_parameter_bytes": total * 4,
        "bf16_parameter_bytes": total * 2,
    }


def model_parameter_digest(model: nn.Module, *, common_only: bool = False) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if common_only and ".attn." in name:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def common_parameter_delta(left: TinyLM, right: TinyLM) -> tuple[float, list[str]]:
    maximum = 0.0
    mismatches: list[str] = []
    right_parameters = dict(right.named_parameters())
    for name, value in left.named_parameters():
        if ".attn." in name:
            continue
        other = right_parameters.get(name)
        if other is None or other.shape != value.shape:
            mismatches.append(name)
            continue
        delta = float((value.detach() - other.detach()).abs().max())
        maximum = max(maximum, delta)
        if delta != 0.0:
            mismatches.append(name)
    return maximum, mismatches


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def masked_loss_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    return (losses * valid.reshape(-1)).sum()


@torch.no_grad()
def evaluate(
    model: TinyLM,
    stream: PackedTokenStream,
    prediction_limit: int,
    batch_sequences: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    prediction_limit = min(prediction_limit, stream.prediction_count)
    total_loss = 0.0
    consumed = 0
    block = 0
    started = time.perf_counter()
    while consumed < prediction_limit:
        remaining = prediction_limit - consumed
        valid_predictions = min(batch_sequences * stream.context, remaining)
        sequences = math.ceil(valid_predictions / stream.context)
        inputs, targets, valid = stream.batch(
            block, sequences, valid_predictions, device
        )
        with autocast_context(device):
            logits = model(inputs)
        total_loss += float(masked_loss_sum(logits, targets, valid).detach().cpu())
        consumed += valid_predictions
        block += sequences
    elapsed = time.perf_counter() - started
    nll = total_loss / max(1, consumed)
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


@dataclass
class TrainRequest:
    method: str
    method_value: int
    run_id: str
    seed: int
    train_tokens: int
    context: int
    micro_batch_sequences: int
    gradient_accumulation_steps: int
    validation_probe_tokens: int
    validation_batch_sequences: int
    full_validation: bool
    save_checkpoint: bool


def run_training(request: TrainRequest, device: torch.device) -> dict[str, Any]:
    profile = load_profile()
    training_cfg = profile["training"]
    model_cfg = ModelConfig(
        context=request.context,
        longformer_query_chunk=int(
            profile["method_configs"]["longformer"]["query_chunk_size"]
        ),
        memformer_segment_length=int(
            profile["method_configs"]["memformer"]["main_segment_length"]
        ),
    )
    effective_tokens = (
        request.context
        * request.micro_batch_sequences
        * request.gradient_accumulation_steps
    )
    if effective_tokens != int(training_cfg["effective_batch_tokens"]):
        raise ValueError(
            f"effective batch is {effective_tokens}, expected "
            f"{training_cfg['effective_batch_tokens']}"
        )

    required_train_cache_tokens = math.ceil(request.train_tokens / request.context) * request.context + 1
    train_path, train_metadata = build_token_cache(
        "train", required_train_cache_tokens
    )
    validation_path, validation_metadata = build_token_cache("validation", None)
    train_stream = PackedTokenStream(
        train_path, int(train_metadata["token_count"]), request.context
    )
    validation_stream = PackedTokenStream(
        validation_path, int(validation_metadata["token_count"]), request.context
    )

    seed_all(request.seed)
    model = TinyLM(model_cfg, request.method, request.method_value, request.seed).to(device)
    parameters = parameter_record(model)
    expected_common = int(profile["model"]["expected_common_parameter_count"])
    if request.method == "longformer" and parameters["total_parameters"] != expected_common:
        raise RuntimeError(
            f"Longformer parameter mismatch: {parameters['total_parameters']} != {expected_common}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        betas=tuple(float(value) for value in training_cfg["betas"]),
        eps=float(training_cfg["epsilon"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    total_steps = math.ceil(request.train_tokens / effective_tokens)
    run_dir = RUNS_DIR / request.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    config_payload = {
        "protocol_version": "tinystories_tinylm_v1",
        "execution_profile": profile["execution_profile"],
        "role": "person_b",
        "request": asdict(request),
        "model": asdict(model_cfg),
        "parameters": parameters,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(training_cfg["learning_rate"]),
            "minimum_learning_rate": float(training_cfg["minimum_learning_rate"]),
            "betas": training_cfg["betas"],
            "epsilon": float(training_cfg["epsilon"]),
            "weight_decay": float(training_cfg["weight_decay"]),
            "warmup_ratio": float(training_cfg["warmup_ratio"]),
            "gradient_clip_norm": float(training_cfg["gradient_clip_norm"]),
        },
        "effective_batch_tokens": effective_tokens,
        "train_cache": train_metadata,
        "validation_cache": validation_metadata,
        "data_order": "pinned_parquet_order_sequential_512_token_blocks",
        "compute_dtype": "bfloat16" if device.type == "cuda" else "float32",
        "parameter_dtype": "float32",
        "environment": environment_record(device),
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
    # ``evaluate`` switches the module to eval mode.  Explicitly restore
    # training mode before the first optimizer step so the configured
    # dropout remains active during training.
    model.train()
    best_probe = dict(initial_validation)
    best_probe["step"] = 0
    best_checkpoint_step = 0
    # Keep a CPU copy of the best-probe weights so the required full
    # validation can be run even when the best probe is not the last step (or
    # when checkpoint writing was disabled for a smoke run).  The TinyLM
    # state is only about 120 MB in FP32 and this avoids conflating the last
    # model with the selected validation checkpoint.
    best_model_state_cpu = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    tokens_seen = 0
    block_cursor = 0
    started = time.perf_counter()
    validation_interval = int(training_cfg["validation_interval_tokens"])
    next_validation = validation_interval
    checkpoint_interval = int(training_cfg["checkpoint_interval_tokens"])
    next_checkpoint = checkpoint_interval
    status = "ok"
    failure: str | None = None

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
            step_target = min(effective_tokens, request.train_tokens - tokens_seen)
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
                        logits = model(inputs)
                    loss_sum = masked_loss_sum(logits, targets, valid)
                    (loss_sum / step_target).backward()
                    step_loss_sum += float(loss_sum.detach().cpu())
                    step_predictions += valid_predictions

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(training_cfg["gradient_clip_norm"])
                    )
                )
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm: {grad_norm}")
                learning_rate = learning_rate_for_step(
                    step,
                    total_steps,
                    float(training_cfg["learning_rate"]),
                    float(training_cfg["minimum_learning_rate"]),
                    float(training_cfg["warmup_ratio"]),
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
            train_nll = step_loss_sum / step_target
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

            should_validate = tokens_seen >= next_validation or tokens_seen == request.train_tokens
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
                    best_model_state_cpu = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                while next_validation <= tokens_seen:
                    next_validation += validation_interval

            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

            should_checkpoint = tokens_seen >= next_checkpoint or tokens_seen == request.train_tokens
            if should_checkpoint and request.save_checkpoint:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "request": asdict(request),
                    "config_hash": config_payload["config_hash"],
                }
                checkpoint_path = run_dir / f"checkpoint.tokens_{tokens_seen}.pt"
                temporary = checkpoint_path.with_suffix(".pt.tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(checkpoint_path)
                while next_checkpoint <= tokens_seen:
                    next_checkpoint += checkpoint_interval

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
    best_probe_full_validation = None
    if request.full_validation and status == "ok":
        full_validation = evaluate(
            model,
            validation_stream,
            validation_stream.prediction_count,
            request.validation_batch_sequences,
            device,
        )
        model.train()

        # Evaluate the probe-selected weights separately.  Restore the final
        # weights afterwards so the summary's model digest and final probe
        # continue to refer to the last training step.
        if best_checkpoint_step == (step if "step" in locals() else 0):
            best_probe_full_validation = dict(full_validation)
        else:
            final_model_state_cpu = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            model.load_state_dict(best_model_state_cpu)
            best_probe_full_validation = evaluate(
                model,
                validation_stream,
                validation_stream.prediction_count,
                request.validation_batch_sequences,
                device,
            )
            model.train()
            model.load_state_dict(final_model_state_cpu)
            model.train()

    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    summary = {
        "protocol_version": "tinystories_tinylm_v1",
        "execution_profile": profile["execution_profile"],
        "role": "person_b",
        "run_id": request.run_id,
        "method": request.method,
        "method_value": request.method_value,
        "status": status,
        "failure": failure,
        "seed": request.seed,
        "train_tokens_requested": request.train_tokens,
        "train_tokens_completed": tokens_seen,
        "optimizer_steps_completed": step if "step" in locals() else 0,
        "elapsed_training_seconds": elapsed,
        "mean_training_tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        "initial_validation_probe": initial_validation,
        "final_validation_probe": final_probe,
        "best_validation_probe": best_probe,
        "best_checkpoint_step": best_checkpoint_step,
        "full_validation": full_validation,
        "best_probe_full_validation": best_probe_full_validation,
        "parameters": parameters,
        "model_parameter_digest": model_parameter_digest(model),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "config_hash": config_payload["config_hash"],
        "completed_at_utc": utc_now(),
    }
    if request.method == "memformer":
        summary["method_specific_state_bytes_bf16_batch1"] = (
            model_cfg.layers * request.method_value * model_cfg.hidden_size * 2
        )
    atomic_json(run_dir / "summary.json", summary)
    return summary


def correctness(device: torch.device) -> dict[str, Any]:
    seed_all(17)
    tiny_config = ModelConfig(
        vocab_size=128,
        layers=2,
        hidden_size=64,
        heads=4,
        ffn_size=256,
        context=32,
        dropout=0.0,
        longformer_query_chunk=16,
        memformer_segment_length=8,
    )
    longformer = TinyLM(tiny_config, "longformer", 31, 17).to(device)
    memformer = TinyLM(tiny_config, "memformer", 8, 17).to(device)
    full = TinyLM(tiny_config, "full_attention", 0, 17).to(device)

    common_delta, common_mismatches = common_parameter_delta(longformer, memformer)
    for layer in range(tiny_config.layers):
        source = full.blocks[layer].attn
        target = longformer.blocks[layer].attn
        for name in ("to_q", "to_k", "to_v", "to_out"):
            getattr(target, name).load_state_dict(getattr(source, name).state_dict())

    sample = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.context), device=device)
    changed = sample.clone()
    changed[:, 21:] = torch.randint(
        0, tiny_config.vocab_size, changed[:, 21:].shape, device=device
    )
    longformer.eval()
    memformer.eval()
    full.eval()
    with torch.no_grad():
        long_logits = longformer(sample)
        long_changed = longformer(changed)
        mem_logits = memformer(sample)
        mem_changed = memformer(changed)
        full_logits = full(sample)
    long_future = float((long_logits[:, :21] - long_changed[:, :21]).abs().max())
    mem_future = float((mem_logits[:, :21] - mem_changed[:, :21]).abs().max())
    full_equivalence = float((long_logits - full_logits).abs().max())

    # Direct state tests use the attention module to expose recurrent memory.
    memory_module = MemformerSegmentAttention(
        64, 4, memory_slots=8, causal=True, detach_memory=False, bias=False
    ).to(device)
    first = torch.randn(2, 8, 64, device=device)
    second = torch.randn(2, 8, 64, device=device)
    zero = torch.zeros(2, 8, 64, device=device)
    _, state = memory_module.forward_segment(first, zero)
    history_output, history_state = memory_module.forward_segment(second, state)
    reset_output, reset_state = memory_module.forward_segment(second, zero)
    history_effect = float((history_output - reset_output).detach().abs().max())
    permutation = torch.tensor([1, 0], device=device)
    permuted_output, permuted_state = memory_module.forward_segment(
        second[permutation], state[permutation]
    )
    reorder_delta = max(
        float((history_output[permutation] - permuted_output).detach().abs().max()),
        float((history_state[permutation] - permuted_state).detach().abs().max()),
    )
    reset_output_2, reset_state_2 = memory_module.forward_segment(second, zero)
    reset_delta = max(
        float((reset_output - reset_output_2).detach().abs().max()),
        float((reset_state - reset_state_2).detach().abs().max()),
    )

    for model in (longformer, memformer):
        model.train()
        model.zero_grad(set_to_none=True)
        logits = model(sample)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), sample[:, 1:].reshape(-1))
        loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError(f"{model.method} has missing or non-finite gradients")

    tests = {
        "common_non_attention_parameter_max_delta": {"value": common_delta, "limit": 0.0},
        "common_non_attention_parameter_mismatches": common_mismatches,
        "longformer_future_invariance": {"value": long_future, "limit": 1e-5},
        "longformer_full_window_rope_equivalence": {
            "value": full_equivalence,
            "limit": 1e-5,
        },
        "memformer_future_invariance": {"value": mem_future, "limit": 1e-5},
        "memformer_history_effect": {"value": history_effect, "minimum": 1e-8},
        "memformer_reset": {"value": reset_delta, "limit": 1e-5},
        "memformer_batch_reorder": {"value": reorder_delta, "limit": 1e-5},
        "longformer_parameters": parameter_record(
            TinyLM(ModelConfig(), "longformer", 128, 17)
        ),
        "memformer_parameters": parameter_record(
            TinyLM(ModelConfig(), "memformer", 64, 17)
        ),
    }
    passed = not common_mismatches
    for value in tests.values():
        if not isinstance(value, dict) or "value" not in value:
            continue
        if "limit" in value:
            passed &= math.isfinite(value["value"]) and value["value"] <= value["limit"]
        if "minimum" in value:
            passed &= math.isfinite(value["value"]) and value["value"] >= value["minimum"]
    payload = {
        "protocol_version": "tinystories_tinylm_v1",
        "role": "person_b",
        "status": "passed" if passed else "failed_correctness",
        "environment": environment_record(device),
        "tests": tests,
    }
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(AGGREGATE_DIR / "person_b_correctness.json", payload)
    if not passed:
        raise SystemExit(1)
    return payload


def aggregate() -> dict[str, Any]:
    summaries = []
    for path in sorted(RUNS_DIR.glob("*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    selection_path = AGGREGATE_DIR / "person_b_pilot_selection.json"
    excluded_ids: set[str] = set()
    screening_ids: set[str] = {
        "main_b_longformer_w128_s17",
        "main_b_memformer_s128_m64_s17",
    }
    selection = None
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        excluded_ids.update(
            str(row["run_id"]) for row in selection.get("excluded_runs", [])
        )
        screening_ids.update(
            str(row["run_id"]) for row in selection.get("candidates", [])
        )
    screening_runs = [
        row
        for row in summaries
        if row.get("run_id") in screening_ids and row.get("run_id") not in excluded_ids
    ]
    excluded_runs = [row for row in summaries if row.get("run_id") in excluded_ids]
    other_runs = [
        row
        for row in summaries
        if row.get("run_id") not in screening_ids and row.get("run_id") not in excluded_ids
    ]
    payload = {
        "protocol_version": "tinystories_tinylm_v1",
        "role": "person_b",
        "created_at_utc": utc_now(),
        "runs": screening_runs,
        "excluded_runs": excluded_runs,
        "other_runs": other_runs,
        "selection_manifest": str(selection_path.relative_to(ROOT))
        if selection_path.exists()
        else None,
    }
    atomic_json(AGGREGATE_DIR / "person_b_runs.json", payload)
    return payload


def device_from_argument(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-cache")
    prepare.add_argument("--train-loss-tokens", type=int, default=10_000_000)
    prepare.add_argument("--context", type=int, default=512)
    prepare.add_argument("--force", action="store_true")

    subparsers.add_parser("correctness")

    train = subparsers.add_parser("train")
    train.add_argument("--method", choices=["longformer", "memformer"], required=True)
    train.add_argument("--method-value", type=int, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--train-tokens", type=int, default=1_048_576)
    train.add_argument("--context", type=int, default=512)
    train.add_argument("--micro-batch-sequences", type=int, default=4)
    train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train.add_argument("--validation-probe-tokens", type=int, default=262_144)
    train.add_argument("--validation-batch-sequences", type=int, default=8)
    train.add_argument("--full-validation", action="store_true")
    train.add_argument("--no-checkpoint", action="store_true")

    subparsers.add_parser("aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_argument(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    if args.command == "prepare-cache":
        required = math.ceil(args.train_loss_tokens / args.context) * args.context + 1
        train_path, train_metadata = build_token_cache(
            "train", required, force=args.force
        )
        validation_path, validation_metadata = build_token_cache(
            "validation", None, force=args.force
        )
        print(
            json.dumps(
                {
                    "train_path": str(train_path),
                    "train": train_metadata,
                    "validation_path": str(validation_path),
                    "validation": validation_metadata,
                },
                indent=2,
            )
        )
    elif args.command == "correctness":
        print(json.dumps(correctness(device), indent=2))
    elif args.command == "train":
        request = TrainRequest(
            method=args.method,
            method_value=args.method_value,
            run_id=args.run_id,
            seed=args.seed,
            train_tokens=args.train_tokens,
            context=args.context,
            micro_batch_sequences=args.micro_batch_sequences,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            validation_probe_tokens=args.validation_probe_tokens,
            validation_batch_sequences=args.validation_batch_sequences,
            full_validation=args.full_validation,
            save_checkpoint=not args.no_checkpoint,
        )
        print(json.dumps(run_training(request, device), indent=2))
    else:
        print(json.dumps(aggregate(), indent=2))


if __name__ == "__main__":
    main()
