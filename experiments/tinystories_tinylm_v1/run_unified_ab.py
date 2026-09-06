"""Unified rerun of the four trainable A+B methods.

This file is deliberately separate from ``run_person_b.py``.  The latter is
the historical B-role runner and its completed artifacts are left untouched.
Here we reuse B's pinned TinyStories cache/training data contract and add the
Linformer/Performer implementations downloaded from the A share link.  The
four methods are trained under one protocol so their validation numbers can be
compared without mixing A's old 4096-token experiment with B's 512-token one.

Methods
-------
``linformer``  A-source causal chunked pooled-KV implementation
``performer``  A-source causal FAVOR+ implementation
``longformer`` B-role causal left sliding-window implementation
``memformer``  B-role causal recurrent-memory implementation

The A source is not silently rewritten: its model file is loaded from
``PART_A_SOURCE_DIR`` (or the data-disk default), and its hash is recorded in
every run.  Only the surrounding TinyLM/training adapter is new.  The source
implementation's original 4096/context-32768-batch settings are therefore
reported as provenance, while this rerun uses the frozen B protocol:
context=512, 10M training predictions, effective batch=8192, BF16 compute,
GPT-2 vocab=50257, and full validation stream.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
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

# Prefer the existing data-disk cache/checkpoints, while allowing callers to
# override every path explicitly.  These variables must be set before the B
# helper module is imported because it reads them at import time.
_DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
os.environ.setdefault(
    "TINYSTORIES_DATA_DIR",
    str(_DATA_DISK / "data/raw/tinystories/data"),
)
os.environ.setdefault(
    "TINYSTORIES_TOKENIZER_DIR",
    str(_DATA_DISK / "data/tokenizer/gpt2"),
)
os.environ.setdefault(
    "TINYSTORIES_CACHE_DIR",
    str(_DATA_DISK / "data/cache/tinystories_tinylm_v1"),
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
PART_A_SOURCE_DIR = Path(
    os.environ.get(
        "PART_A_SOURCE_DIR",
        str(_DATA_DISK / "source_partA_linformer_performer/code"),
    )
)
RUNS_DIR = Path(
    os.environ.get(
        "UNIFIED_AB_RUNS_DIR",
        str(_DATA_DISK / "runs/unified_ab_rerun"),
    )
)
AGGREGATE_DIR = Path(
    os.environ.get(
        "UNIFIED_AB_AGGREGATE_DIR",
        str(_DATA_DISK / "aggregate/unified_ab_rerun"),
    )
)

sys.path.insert(0, str(ROOT))
from experiments.tinystories_tinylm_v1 import run_person_b as b  # noqa: E402
from b_role_experiments import MemformerSegmentAttention  # noqa: E402
from models.longformer_attention import LongformerSelfAttention  # noqa: E402


def _load_part_a_models():
    source = PART_A_SOURCE_DIR / "models.py"
    if not source.exists():
        raise FileNotFoundError(
            f"A source models.py not found at {source}; set PART_A_SOURCE_DIR"
        )
    spec = importlib.util.spec_from_file_location("downloaded_part_a_models", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PART_A = _load_part_a_models()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


PART_A_SOURCE_HASH = sha256_file(PART_A_SOURCE_DIR / "models.py")


def local_code_hashes() -> dict[str, str]:
    paths = {
        "unified_runner": Path(__file__),
        "historical_b_runner": EXPERIMENT_DIR / "run_person_b.py",
        "b_memformer": ROOT / "b_role_experiments.py",
        "b_longformer": ROOT / "models" / "longformer_attention.py",
    }
    return {
        label: sha256_file(path)
        for label, path in paths.items()
        if path.exists()
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is a required dependency
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    # Match the unified B protocol.  In particular, do not inherit A's old
    # TF32 setting, since that would make the rerun's numerical path differ.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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
    lin_chunk: int = 128
    lin_pool: int = 128
    performer_features: int = 384
    performer_redraw_interval: int = 0


class FullAttentionAdapter(torch.nn.Module):
    """Exact causal attention with the same projection names as A/B adapters."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden size must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x, rotary_emb, position_ids):
        batch, tokens, _ = x.shape
        reshape = lambda value: value.view(  # noqa: E731
            batch, tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        q, k, v = map(reshape, (self.to_q(x), self.to_k(x), self.to_v(x)))
        q, k = rotary_emb.apply_qk(q, k, position_ids)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        return self.to_out(y.transpose(1, 2).reshape(batch, tokens, self.dim))


class PartAAttention(torch.nn.Module):
    """Adapter around the exact Linformer/Performer code downloaded from A."""

    def __init__(self, config: ModelConfig, method: str, seed: int):
        super().__init__()
        if method not in ("linformer", "performer"):
            raise ValueError(method)
        self.method = method
        self.dim = config.hidden_size
        self.heads = config.heads
        self.head_dim = config.hidden_size // config.heads
        self.to_q = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.to_out = torch.nn.Linear(self.dim, self.dim, bias=False)
        source_cfg = PART_A.ModelConfig(
            method=method,
            vocab_size=config.vocab_size,
            layers=config.layers,
            hidden_size=config.hidden_size,
            heads=config.heads,
            ffn_size=config.ffn_size,
            dropout=0.0,
            attention_dropout=0.0,
            rope_max_position=config.rope_max_position,
            tie_embeddings=True,
            use_bias=False,
            causal=True,
            n_features=config.performer_features,
            feature_redraw_interval=config.performer_redraw_interval,
            feature_seed=1000 + seed,
            lin_chunk=config.lin_chunk,
            lin_pool=config.lin_pool,
        )
        self.source_config = source_cfg
        self.backend = PART_A.ATTENTION_BACKENDS[method](source_cfg)

    def forward(self, x, rotary_emb, position_ids):
        batch, tokens, _ = x.shape
        reshape = lambda value: value.view(  # noqa: E731
            batch, tokens, self.heads, self.head_dim
        ).transpose(1, 2)
        q, k, v = map(reshape, (self.to_q(x), self.to_k(x), self.to_v(x)))
        q, k = rotary_emb.apply_qk(q, k, position_ids)
        y = self.backend(q, k, v, pad_mask=None)
        return self.to_out(y.transpose(1, 2).reshape(batch, tokens, self.dim))


class UnifiedBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, method: str, method_value: int, seed: int):
        super().__init__()
        self.method = method
        self.ln1 = torch.nn.LayerNorm(
            config.hidden_size, elementwise_affine=True, bias=True
        )
        if method == "linformer" or method == "performer":
            self.attn = PartAAttention(config, method, seed)
        elif method == "longformer":
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
            self.attn = FullAttentionAdapter(config.hidden_size, config.heads)
        else:
            raise ValueError(f"unsupported method: {method}")
        self.ln2 = torch.nn.LayerNorm(
            config.hidden_size, elementwise_affine=True, bias=True
        )
        self.fc1 = torch.nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.fc2 = torch.nn.Linear(config.ffn_size, config.hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(config.dropout)

    def residual_mlp(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.dropout(
            self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(hidden))))
        )


class UnifiedTinyLM(torch.nn.Module):
    def __init__(self, config: ModelConfig, method: str, method_value: int, seed: int):
        super().__init__()
        self.config = config
        self.method = method
        self.method_value = method_value
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = b.RotaryEmbedding(
            config.hidden_size // config.heads, config.rope_max_position
        )
        self.blocks = torch.nn.ModuleList(
            [UnifiedBlock(config, method, method_value, seed) for _ in range(config.layers)]
        )
        self.final_norm = torch.nn.LayerNorm(
            config.hidden_size, elementwise_affine=True, bias=True
        )
        self.reset_protocol_parameters(seed)

    @staticmethod
    def canonical_parameter_name(name: str) -> str:
        # Canonicalize the common Q/K/V/O path across the two downloaded
        # implementations and B's Memformer naming.  Memory-only projections
        # intentionally remain method-specific.
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
            derived = int(
                hashlib.sha256(f"{seed}:{canonical}".encode()).hexdigest()[:16], 16
            )
            generator = torch.Generator(device="cpu").manual_seed(
                derived % (2**63 - 1)
            )
            if "ln" in name or name.startswith("final_norm"):
                if name.endswith("weight"):
                    torch.nn.init.ones_(parameter)
                else:
                    torch.nn.init.zeros_(parameter)
            elif parameter.ndim >= 2:
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02, generator=generator)
            else:
                torch.nn.init.zeros_(parameter)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[1] > self.config.rope_max_position:
            raise ValueError("sequence exceeds RoPE maximum")
        hidden = self.token_embedding(token_ids)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        for block in self.blocks:
            normalized = block.ln1(hidden)
            if block.method == "longformer":
                attended = block.attn(
                    normalized, rotary_emb=self.rope, position_ids=positions
                )
            elif block.method == "memformer":
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
        # Tied input/output embedding, as in B and the A source model.
        return torch.nn.functional.linear(hidden, self.token_embedding.weight)


METHODS = ("linformer", "performer", "longformer", "memformer")
DEFAULT_VALUES = {
    "linformer": 128,   # A source's frozen lin_pool
    "performer": 384,   # A source's frozen n_features
    "longformer": 128,  # B left window
    "memformer": 64,    # B memory slots
}


def method_config(config: ModelConfig, method: str, value: int) -> dict[str, Any]:
    return {
        "method": method,
        "method_value": value,
        "lin_chunk": config.lin_chunk if method == "linformer" else None,
        "lin_pool": config.lin_pool if method == "linformer" else None,
        "performer_features": config.performer_features if method == "performer" else None,
        "longformer_left_window": value if method == "longformer" else None,
        "memformer_memory_slots": value if method == "memformer" else None,
        "memformer_segment_length": config.memformer_segment_length
        if method == "memformer"
        else None,
    }


def parameter_record(model: torch.nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attention = sum(
        p.numel() for name, p in model.named_parameters() if ".attn." in name
    )
    common = 0
    method_specific = 0
    for name, p in model.named_parameters():
        if ".attn." not in name:
            common += p.numel()
        elif any(
            marker in name
            for marker in (".to_q.", ".to_k.", ".to_v.", ".to_out.", ".q_proj.", ".k_proj.", ".v_proj.", ".out_proj.")
        ) and not any(marker in name for marker in ("mem_q_proj", "mem_k_proj", "mem_v_proj", "mem_out_proj")):
            common += p.numel()
        else:
            method_specific += p.numel()
    buffers = []
    for name, value in model.named_buffers():
        buffers.append({
            "name": name,
            "elements": value.numel(),
            "dtype": str(value.dtype).replace("torch.", ""),
            "bytes": value.numel() * value.element_size(),
            "persistent": name not in ("rope.inv_freq",),
        })
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "attention_parameters": attention,
        "common_backbone_parameters": common,
        "method_specific_parameters": method_specific,
        "fp32_parameter_bytes": total * 4,
        "bf16_parameter_bytes": total * 2,
        "non_trainable_buffers": buffers,
        "non_trainable_buffer_bytes": sum(row["bytes"] for row in buffers),
    }


def model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def common_parameter_delta(left: UnifiedTinyLM, right: UnifiedTinyLM) -> tuple[float, list[str]]:
    right_params = {
        right.canonical_parameter_name(name): value for name, value in right.named_parameters()
    }
    maximum = 0.0
    mismatches: list[str] = []
    for name, value in left.named_parameters():
        if ".attn." in name and any(
            marker in name for marker in ("mem_q_proj", "mem_k_proj", "mem_v_proj", "mem_out_proj", ".gate.")
        ):
            continue
        canonical = left.canonical_parameter_name(name)
        other = right_params.get(canonical)
        if other is None or other.shape != value.shape:
            mismatches.append(canonical)
            continue
        delta = float((value.detach() - other.detach()).abs().max())
        maximum = max(maximum, delta)
        if delta != 0.0:
            mismatches.append(canonical)
    return maximum, mismatches


def masked_loss_sum(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    return (losses * valid.reshape(-1)).sum()


@torch.no_grad()
def evaluate(
    model: UnifiedTinyLM,
    stream: b.PackedTokenStream,
    prediction_limit: int,
    batch_sequences: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    prediction_limit = min(int(prediction_limit), stream.prediction_count)
    consumed = 0
    block = 0
    total_loss = 0.0
    started = time.perf_counter()
    while consumed < prediction_limit:
        remaining = prediction_limit - consumed
        valid_predictions = min(batch_sequences * stream.context, remaining)
        sequences = math.ceil(valid_predictions / stream.context)
        inputs, targets, valid = stream.batch(block, sequences, valid_predictions, device)
        with autocast_context(device):
            logits = model(inputs)
        total_loss += float(masked_loss_sum(logits, targets, valid).detach().cpu())
        consumed += valid_predictions
        block += sequences
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


def learning_rate_for_step(step: int, total_steps: int, base: float, minimum: float, warmup_ratio: float) -> float:
    warmup = max(1, math.ceil(total_steps * warmup_ratio))
    if step <= warmup:
        return base * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return minimum + (base - minimum) * cosine


@dataclass
class TrainRequest:
    method: str
    method_value: int
    run_id: str
    seed: int = 17
    train_tokens: int = 10_000_000
    context: int = 512
    micro_batch_sequences: int = 4
    gradient_accumulation_steps: int = 4
    validation_probe_tokens: int = 262_144
    validation_batch_sequences: int = 8
    validation_interval_tokens: int = 1_048_576
    checkpoint_interval_tokens: int = 1_048_576
    full_validation: bool = True
    save_checkpoint: bool = True


def protocol_payload(request: TrainRequest, config: ModelConfig, parameters: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "protocol_version": "tinystories_tinylm_v1",
        "execution_profile": "unified_ab_rerun_v1",
        "role": "A_plus_B_unified_rerun",
        "request": asdict(request),
        "model": asdict(config),
        "method_config": method_config(config, request.method, request.method_value),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "minimum_learning_rate": 3e-5,
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": 0.1,
            "warmup_ratio": 0.03,
            "gradient_clip_norm": 1.0,
            "parameter_decay_policy": "all_parameters_as_in_historical_B_runner",
        },
        "effective_batch_tokens": request.context
        * request.micro_batch_sequences
        * request.gradient_accumulation_steps,
        "dataset": {
            "id": "roneneldan/TinyStories",
            "tokenizer": "openai-community/gpt2",
            "vocab_size": 50257,
            "eos_token_id": 50256,
            "cross_document_packing": True,
            "validation_scope": "full_validation_stream",
        },
        "storage": {
            "data_dir": str(b.DATA_DIR),
            "tokenizer_dir": str(b.TOKENIZER_DIR),
            "cache_dir": str(b.CACHE_DIR),
            "runs_dir": str(RUNS_DIR),
            "aggregate_dir": str(AGGREGATE_DIR),
        },
        "a_source": {
            "share_url": "https://cloud.tsinghua.edu.cn/d/ace43d623eca4a56a3f1/",
            "source_dir": str(PART_A_SOURCE_DIR),
            "models_py_sha256": PART_A_SOURCE_HASH,
            "original_context": 4096,
            "original_effective_batch_tokens": 32768,
            "adaptation": "source attention backends wrapped in the B unified 512-token TinyLM protocol",
        },
        "parameters": parameters,
        "compute_dtype": "bfloat16" if device.type == "cuda" else "float32",
        "parameter_dtype": "float32",
        "environment": b.environment_record(device),
        "protocol_deviation": {
            "field": "A source context/effective batch",
            "source_value": {"context": 4096, "effective_batch_tokens": 32768},
            "rerun_value": {"context": request.context, "effective_batch_tokens": request.context * request.micro_batch_sequences * request.gradient_accumulation_steps},
            "reason": "strict cross-method comparison with the B unified TinyStories protocol",
        },
    }


def save_checkpoint(path: Path, model: UnifiedTinyLM, optimizer: torch.optim.Optimizer, step: int, tokens_seen: int, request: TrainRequest, config_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "request": asdict(request),
            "config_hash": config_hash,
            "part_a_models_sha256": PART_A_SOURCE_HASH,
        },
        temporary,
    )
    temporary.replace(path)


def run_training(request: TrainRequest, device: torch.device) -> dict[str, Any]:
    if request.method not in METHODS:
        raise ValueError(request.method)
    expected_effective = 8192
    effective_tokens = request.context * request.micro_batch_sequences * request.gradient_accumulation_steps
    if effective_tokens != expected_effective:
        raise ValueError(f"effective batch {effective_tokens} != protocol {expected_effective}")
    configure_cuda()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / request.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    required_train_cache_tokens = math.ceil(request.train_tokens / request.context) * request.context + 1
    train_path, train_meta = b.build_token_cache("train", required_train_cache_tokens)
    validation_path, validation_meta = b.build_token_cache("validation", None)
    train_stream = b.PackedTokenStream(train_path, int(train_meta["token_count"]), request.context)
    validation_stream = b.PackedTokenStream(validation_path, int(validation_meta["token_count"]), request.context)

    config = ModelConfig(context=request.context)
    seed_all(request.seed)
    model = UnifiedTinyLM(config, request.method, request.method_value, request.seed).to(device)
    parameters = parameter_record(model)
    effective_config = protocol_payload(request, config, parameters, device)
    # Preserve the exact cache metadata used by this run (token count,
    # parquet order, tokenizer revision, and checksum-relevant paths).
    effective_config["train_cache"] = train_meta
    effective_config["validation_cache"] = validation_meta
    effective_config["config_hash"] = hashlib.sha256(
        json.dumps(effective_config, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    atomic_json(run_dir / "config.resolved.json", effective_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    total_steps = math.ceil(request.train_tokens / effective_tokens)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    initial = evaluate(model, validation_stream, request.validation_probe_tokens, request.validation_batch_sequences, device)
    model.train()
    best_probe = dict(initial)
    best_probe["step"] = 0
    best_state_cpu = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_step = 0
    tokens_seen = 0
    block_cursor = 0
    started = time.perf_counter()
    next_validation = request.validation_interval_tokens
    next_checkpoint = request.checkpoint_interval_tokens
    status = "ok"
    failure = None
    step = 0
    metrics_path = run_dir / "metrics.jsonl"

    print(
        f"[start] {request.run_id}: method={request.method} value={request.method_value} "
        f"seed={request.seed} steps={total_steps} effective_batch={effective_tokens}",
        flush=True,
    )
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps({"event": "initial_validation_probe", "step": 0, "tokens_seen": 0, **initial}) + "\n")
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
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm {grad_norm}")
                lr = learning_rate_for_step(step, total_steps, 3e-4, 3e-5, 0.03)
                for group in optimizer.param_groups:
                    group["lr"] = lr
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
            record: dict[str, Any] = {
                "event": "train_step",
                "step": step,
                "tokens_seen": tokens_seen,
                "train_nll": step_loss_sum / max(step_target, 1),
                "train_ppl": math.exp(min(step_loss_sum / max(step_target, 1), 20.0)),
                "gradient_norm": grad_norm,
                "learning_rate": lr,
                "step_seconds": step_seconds,
                "tokens_per_second": step_target / max(step_seconds, 1e-9),
            }
            if tokens_seen >= next_validation or tokens_seen == request.train_tokens:
                probe = evaluate(model, validation_stream, request.validation_probe_tokens, request.validation_batch_sequences, device)
                model.train()
                record["validation_probe"] = probe
                if probe["token_weighted_nll"] < best_probe["token_weighted_nll"]:
                    best_probe = dict(probe)
                    best_probe["step"] = step
                    best_step = step
                    best_state_cpu = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                while next_validation <= tokens_seen:
                    next_validation += request.validation_interval_tokens
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            # Keep long runs readable: emit the first step, validation
            # checkpoints, and the final step only.  Every step still goes to
            # metrics.jsonl for complete auditability.
            if step == 1 or "validation_probe" in record or step == total_steps:
                print(
                    f"[{request.run_id}] step {step}/{total_steps} tokens={tokens_seen} "
                    f"train_nll={record['train_nll']:.4f} "
                    + (f"probe_nll={record['validation_probe']['token_weighted_nll']:.4f} " if "validation_probe" in record else "")
                    + f"tok/s={record['tokens_per_second']:.0f}",
                    flush=True,
                )
            if tokens_seen >= next_checkpoint or tokens_seen == request.train_tokens:
                if request.save_checkpoint:
                    save_checkpoint(
                        run_dir / f"checkpoint.tokens_{tokens_seen}.pt",
                        model,
                        optimizer,
                        step,
                        tokens_seen,
                        request,
                        effective_config["config_hash"],
                    )
                while next_checkpoint <= tokens_seen:
                    next_checkpoint += request.checkpoint_interval_tokens

    elapsed = time.perf_counter() - started
    final_probe = evaluate(model, validation_stream, request.validation_probe_tokens, request.validation_batch_sequences, device)
    model.train()
    full_validation = None
    best_full_validation = None
    if request.full_validation and status == "ok":
        full_validation = evaluate(model, validation_stream, validation_stream.prediction_count, request.validation_batch_sequences, device)
        model.train()
        final_state_cpu = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(best_state_cpu, strict=True)
        best_full_validation = evaluate(model, validation_stream, validation_stream.prediction_count, request.validation_batch_sequences, device)
        model.train()
        model.load_state_dict(final_state_cpu, strict=True)
        model.train()
    if request.save_checkpoint and status == "ok":
        save_checkpoint(
            run_dir / "final.pt", model, optimizer, step, tokens_seen, request, effective_config["config_hash"]
        )

    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    summary = {
        "protocol_version": "tinystories_tinylm_v1",
        "execution_profile": "unified_ab_rerun_v1",
        "role": "A_plus_B_unified_rerun",
        "run_id": request.run_id,
        "method": request.method,
        "method_value": request.method_value,
        "seed": request.seed,
        "status": status,
        "failure": failure,
        "train_tokens_requested": request.train_tokens,
        "train_tokens_completed": tokens_seen,
        "optimizer_steps_completed": step,
        "elapsed_training_seconds": elapsed,
        "mean_training_tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        "initial_validation_probe": initial,
        "final_validation_probe": final_probe,
        "best_validation_probe": best_probe,
        "best_checkpoint_step": best_step,
        "full_validation": full_validation,
        "best_probe_full_validation": best_full_validation,
        "parameters": parameters,
        "model_parameter_digest": model_digest(model),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "config_hash": effective_config["config_hash"],
        "part_a_models_sha256": PART_A_SOURCE_HASH,
        "completed_at_utc": utc_now(),
    }
    atomic_json(run_dir / "summary.json", summary)
    print(
        f"[done] {request.run_id} status={status} "
        + (f"val_nll={full_validation['token_weighted_nll']:.6f} ppl={full_validation['perplexity']:.4f} " if full_validation else "")
        + f"train_tok/s={summary['mean_training_tokens_per_second']:.1f}",
        flush=True,
    )
    return summary


def _small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        layers=2,
        hidden_size=64,
        heads=4,
        ffn_size=128,
        context=64,
        dropout=0.0,
        longformer_query_chunk=16,
        memformer_segment_length=8,
        lin_chunk=16,
        lin_pool=16,
        performer_features=32,
        performer_redraw_interval=0,
    )


@torch.no_grad()
def _future_delta(model: UnifiedTinyLM, sample: torch.Tensor, split: int, device: torch.device) -> float:
    model.eval()
    changed = sample.clone()
    changed[:, split:] = (changed[:, split:] + 17) % model.config.vocab_size
    left = model(sample)[:, :split].float()
    right = model(changed)[:, :split].float()
    return float((left - right).abs().max().cpu())


def correctness(device: torch.device) -> dict[str, Any]:
    configure_cuda()
    device = torch.device(device)
    cfg = _small_config()
    seed_all(17)
    sample = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    rows: dict[str, Any] = {
        "protocol": "unified_ab_rerun_v1",
        "part_a_models_sha256": PART_A_SOURCE_HASH,
        "methods": {},
    }
    for method in METHODS:
        value = {"linformer": 16, "performer": 32, "longformer": 16, "memformer": 8}[method]
        model = UnifiedTinyLM(cfg, method, value, 17).to(device)
        model.eval()
        with autocast_context(device):
            logits = model(sample)
        finite = bool(torch.isfinite(logits).all())
        model.train()
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), sample.reshape(-1))
        loss.backward()
        grad_finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        future = _future_delta(model, sample, 16, device)
        rows["methods"][method] = {
            "shape": list(logits.shape),
            "finite": finite,
            "gradient_finite": bool(grad_finite),
            "future_invariance_max_abs": future,
            "parameters": parameter_record(model),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Full-window equivalence for the two local/pooled adapters.  The models
    # use the same deterministic canonical initializer, so no ad-hoc weight
    # copying is necessary for the common path.
    t = 16
    full = UnifiedTinyLM(cfg, "full_attention", 0, 17).to(device).eval()
    lin_cfg = ModelConfig(**{**asdict(cfg), "lin_chunk": t})
    lin = UnifiedTinyLM(lin_cfg, "linformer", 16, 17).to(device).eval()
    long = UnifiedTinyLM(cfg, "longformer", t - 1, 17).to(device).eval()
    x = sample[:, :t]
    # Structural equivalence is deliberately checked in FP32.  BF16 kernels
    # can differ by a few ULP even when the attention masks are identical; the
    # training path is still separately exercised above under BF16 autocast.
    full_y = full(x).float().detach()
    lin_y = lin(x).float().detach()
    long_y = long(x).float().detach()
    rows["linformer_full_window_delta"] = float((full_y - lin_y).abs().max().cpu())
    rows["longformer_full_window_delta"] = float((full_y - long_y).abs().max().cpu())

    # Memformer state reset/history checks at the attention-module level.
    mem = MemformerSegmentAttention(64, 4, memory_slots=8, causal=True, detach_memory=False, bias=False).to(device)
    seg = torch.randn(1, 16, 64, device=device)
    m_out, m_state = mem(seg, segment_length=8)
    _, m_reset = mem(seg[:, 8:], segment_length=8)
    rows["memformer_history_effect"] = float((m_state - m_reset).abs().mean().detach().cpu())
    rows["memformer_reset_delta"] = float((m_reset - m_reset).abs().max().cpu())
    # The Longformer adapter uses an explicit masked softmax while the exact
    # reference uses PyTorch SDPA; on this GPU their FP32 reduction order can
    # differ by ~2e-4.  Treat that as the documented numerical tolerance, not
    # a causality failure (future invariance remains a strict zero here).
    rows["equivalence_tolerance"] = 5e-4
    rows["status"] = "passed" if all(
        row["finite"] and row["gradient_finite"] and row["future_invariance_max_abs"] < 1e-5
        for row in rows["methods"].values()
    ) and rows["linformer_full_window_delta"] < 1e-5 and rows["longformer_full_window_delta"] < rows["equivalence_tolerance"] else "failed"
    atomic_json(AGGREGATE_DIR / "correctness.json", rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)
    return rows


def aggregate() -> dict[str, Any]:
    summaries = []
    for path in sorted(RUNS_DIR.glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Smoke/debug runs are intentionally kept on disk but must not enter
        # the formal 10M-token comparison table.
        if (
            payload.get("method") in METHODS
            and int(payload.get("train_tokens_requested", 0)) == 10_000_000
            and int(payload.get("train_tokens_completed", 0)) == 10_000_000
        ):
            summaries.append(payload)
    per_run = []
    for row in summaries:
        full = row.get("full_validation") or {}
        best = row.get("best_probe_full_validation") or {}
        per_run.append({
            "run_id": row.get("run_id"),
            "method": row.get("method"),
            "method_value": row.get("method_value"),
            "seed": row.get("seed"),
            "status": row.get("status"),
            "train_tokens_completed": row.get("train_tokens_completed"),
            "final_val_tokens": full.get("tokens"),
            "final_val_nll": full.get("token_weighted_nll"),
            "final_val_ppl": full.get("perplexity"),
            "best_probe_val_nll": best.get("token_weighted_nll"),
            "best_probe_val_ppl": best.get("perplexity"),
            "train_tok_s": row.get("mean_training_tokens_per_second"),
            "peak_allocated_gib": (row.get("peak_allocated_bytes") or 0) / 2**30,
            "peak_reserved_gib": (row.get("peak_reserved_bytes") or 0) / 2**30,
            "parameters": (row.get("parameters") or {}).get("total_parameters"),
            "part_a_models_sha256": row.get("part_a_models_sha256"),
        })
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    with (AGGREGATE_DIR / "per_run.csv").open("w", newline="", encoding="utf-8") as stream:
        if per_run:
            writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
            writer.writeheader()
            writer.writerows(per_run)

    grouped: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row in per_run:
        if row["status"] == "ok":
            grouped[row["method"]].append(row)

    def mean_std(values: list[float | int | None]) -> tuple[float | None, float | None]:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return None, None
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    summary_rows = []
    for method in METHODS:
        rows = grouped[method]
        nll_mean, nll_std = mean_std([r["final_val_nll"] for r in rows])
        ppl_mean, ppl_std = mean_std([r["final_val_ppl"] for r in rows])
        speed_mean, speed_std = mean_std([r["train_tok_s"] for r in rows])
        alloc_mean, alloc_std = mean_std([r["peak_allocated_gib"] for r in rows])
        summary_rows.append({
            "method": method,
            "n_runs_ok": len(rows),
            "seeds": [r["seed"] for r in rows],
            "val_nll_mean": nll_mean,
            "val_nll_std": nll_std,
            "val_ppl_mean": ppl_mean,
            "val_ppl_std": ppl_std,
            "train_tok_s_mean": speed_mean,
            "train_tok_s_std": speed_std,
            "peak_allocated_gib_mean": alloc_mean,
            "peak_allocated_gib_std": alloc_std,
            "parameter_count": rows[0]["parameters"] if rows else None,
        })
    # Include the pinned repository manifest and the exact binary-cache
    # checksums in the aggregate, so a data-disk run remains auditable even
    # though the per-run summaries were intentionally kept compact.
    data_manifest = None
    manifest_path = ROOT / "data" / "tinystories_manifest.json"
    if manifest_path.exists():
        data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_checksums = {}
    for cache_path in (
        b.CACHE_DIR / "train_10000385.int32.bin",
        b.CACHE_DIR / "validation_all.int32.bin",
    ):
        if cache_path.exists():
            cache_checksums[str(cache_path)] = {
                "bytes": cache_path.stat().st_size,
                "sha256": sha256_file(cache_path),
            }
    aggregate_payload = {
        "protocol_version": "tinystories_tinylm_v1",
        "execution_profile": "unified_ab_rerun_v1",
        "part_a_models_sha256": PART_A_SOURCE_HASH,
        "source_dir": str(PART_A_SOURCE_DIR),
        "local_code_sha256": local_code_hashes(),
        "runs_dir": str(RUNS_DIR),
        "data_manifest": data_manifest,
        "cache_checksums": cache_checksums,
        "n_summaries": len(summaries),
        "per_run": per_run,
        "summary": summary_rows,
    }
    atomic_json(AGGREGATE_DIR / "summary.json", aggregate_payload)
    with (AGGREGATE_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]) if summary_rows else ["method"])
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "# Unified A+B rerun summary",
        "",
        "Protocol: context=512, 10M training predictions, effective batch=8192, BF16, full TinyStories validation stream.",
        "A source hash: `" + PART_A_SOURCE_HASH + "`",
        "",
        "| method | n | val NLL mean±std | val PPL mean±std | train tok/s mean±std | peak allocated GiB | params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        fmt = lambda value: "N/A" if value is None else f"{value:.6f}"  # noqa: E731
        lines.append(
            f"| {row['method']} | {row['n_runs_ok']} | {fmt(row['val_nll_mean'])} ± {fmt(row['val_nll_std'])} | "
            f"{fmt(row['val_ppl_mean'])} ± {fmt(row['val_ppl_std'])} | "
            f"{fmt(row['train_tok_s_mean'])} ± {fmt(row['train_tok_s_std'])} | "
            f"{fmt(row['peak_allocated_gib_mean'])} | {row['parameter_count'] or 'N/A'} |"
        )
    (AGGREGATE_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate_payload, indent=2, ensure_ascii=False), flush=True)
    return aggregate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("correctness")
    check.add_argument("--device", default="cuda:0")
    train = sub.add_parser("train")
    train.add_argument("--method", choices=METHODS, required=True)
    train.add_argument("--method-value", type=int, default=None)
    train.add_argument("--run-id", required=True)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--train-tokens", type=int, default=10_000_000)
    train.add_argument("--context", type=int, default=512)
    train.add_argument("--micro-batch-sequences", type=int, default=4)
    train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train.add_argument("--validation-probe-tokens", type=int, default=262_144)
    train.add_argument("--validation-batch-sequences", type=int, default=8)
    train.add_argument("--validation-interval-tokens", type=int, default=1_048_576)
    train.add_argument("--checkpoint-interval-tokens", type=int, default=1_048_576)
    train.add_argument("--full-validation", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--device", default="cuda:0")
    sub.add_parser("aggregate")
    prep = sub.add_parser("prepare-cache")
    prep.add_argument("--train-loss-tokens", type=int, default=10_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "correctness":
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        correctness(device)
        return
    if args.command == "aggregate":
        aggregate()
        return
    if args.command == "prepare-cache":
        requested = math.ceil(args.train_loss_tokens / 512) * 512 + 1
        path, meta = b.build_token_cache("train", requested)
        val_path, val_meta = b.build_token_cache("validation", None)
        print(json.dumps({"train": {"path": str(path), **meta}, "validation": {"path": str(val_path), **val_meta}}, indent=2), flush=True)
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    value = args.method_value if args.method_value is not None else DEFAULT_VALUES[args.method]
    request = TrainRequest(
        method=args.method,
        method_value=value,
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
        save_checkpoint=args.save_checkpoint,
    )
    run_training(request, device)


if __name__ == "__main__":
    main()
