"""Run six long-context variants with a shared sparse MoE feed-forward design.

The historical dense runners and their results are intentionally left untouched.
Five trainable attention backbones use the same six-layer TinyLM, with every
dense FFN replaced by a four-expert, top-1, dropless MoE.  Keyformer remains an
inference-time KV-cache policy and is evaluated on a separately trained
Full-Attention + MoE checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DATA_DISK = Path("/root/autodl-tmp/26summerBDMI_transformer")
RUNS_DIR = Path(
    os.environ.get(
        "MOE_SIX_RUNS_DIR",
        str(DATA_DISK / "runs" / "tinystories_tinylm_moe_v1"),
    )
)
AGGREGATE_DIR = Path(
    os.environ.get(
        "MOE_SIX_AGGREGATE_DIR",
        str(DATA_DISK / "aggregate" / "tinystories_tinylm_moe_v1"),
    )
)

sys.path.insert(0, str(ROOT))
from experiments.tinystories_tinylm_v1 import run_unified_ab as dense  # noqa: E402
from models.keyformer import (  # noqa: E402
    FullKVPolicy,
    KeyformerPolicy,
    KVCacheState,
)

# The vendored Reformer source has optional imports.  The repository contains
# minimal compatibility modules for configurations that disable those paths.
sys.path.insert(0, str(ROOT / "reformer-pytorch"))
sys.path.insert(0, str(ROOT / "compat_deps"))
from reformer_pytorch import LSHSelfAttention  # noqa: E402


TRAINABLE_METHODS = (
    "memformer",
    "linformer",
    "performer",
    "longformer",
    "reformer",
)
BASE_METHOD = "full_attention"
ALL_TRAINING_METHODS = TRAINABLE_METHODS + (BASE_METHOD,)
DEFAULT_VALUES = {
    "memformer": 64,
    "linformer": 128,
    "performer": 384,
    "longformer": 128,
    "reformer": 64,
    "full_attention": 0,
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int = 4
    top_k: int = 1
    dropless: bool = True
    # Zero-based indices.  The requested setup replaces all six layers.
    moe_layers: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    load_balance_coefficient: float = 0.01
    router_z_loss_coefficient: float = 0.001

    def validate(self, model_layers: int) -> None:
        if self.num_experts < 2:
            raise ValueError("num_experts must be at least two")
        if self.top_k != 1:
            raise ValueError("this audited runner currently implements top_k=1 only")
        if not self.dropless:
            raise ValueError("this experiment requires dropless routing")
        if tuple(self.moe_layers) != tuple(range(model_layers)):
            raise ValueError(
                "the requested protocol requires MoE in every Transformer layer"
            )


class ExpertFFN(torch.nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden, bias=False)
        self.fc2 = torch.nn.Linear(hidden, dim, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(value)))


class SparseTop1MoE(torch.nn.Module):
    """Top-1 token router with no capacity truncation and no token dropping."""

    def __init__(self, dim: int, hidden: int, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.router = torch.nn.Linear(dim, config.num_experts, bias=False)
        self.experts = torch.nn.ModuleList(
            [ExpertFFN(dim, hidden) for _ in range(config.num_experts)]
        )

    def forward(
        self,
        value: torch.Tensor,
        return_router: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        original_shape = value.shape
        flat = value.reshape(-1, original_shape[-1])
        router_logits = self.router(flat).float()
        probabilities = torch.softmax(router_logits, dim=-1)
        top_probability, assignments = probabilities.max(dim=-1)

        combined = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_indices = torch.nonzero(
                assignments == expert_index, as_tuple=False
            ).flatten()
            if token_indices.numel() == 0:
                continue
            expert_input = flat.index_select(0, token_indices)
            expert_output = expert(expert_input)
            gate = top_probability.index_select(0, token_indices).to(
                expert_output.dtype
            )
            combined.index_add_(
                0,
                token_indices,
                expert_output * gate.unsqueeze(-1),
            )

        output = combined.view(original_shape)
        if not return_router:
            return output, None

        counts = torch.bincount(assignments, minlength=self.num_experts)
        route_fraction = counts.float() / max(assignments.numel(), 1)
        probability_mass = probabilities.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(
            route_fraction.detach() * probability_mass
        )
        router_z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1).square())
        entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-9)), dim=-1
        ).mean()
        return output, {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
            "counts": counts,
            "probability_mass": probability_mass,
            "entropy": entropy,
            "mean_top_probability": top_probability.mean(),
        }


class DenseFFN(torch.nn.Module):
    """Available for small ablations; unused by the all-six-layer protocol."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden, bias=False)
        self.fc2 = torch.nn.Linear(hidden, dim, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(value)))


class ReformerAttentionAdapter(torch.nn.Module):
    """Causal LSH attention adapter using the vendored Reformer implementation."""

    def __init__(
        self,
        config: dense.ModelConfig,
        bucket_size: int,
        seed: int,
        layer_index: int,
    ):
        super().__init__()
        if config.context % (2 * bucket_size):
            raise ValueError(
                f"context={config.context} must be divisible by 2*bucket={2 * bucket_size}"
            )
        self.hash_seed = 50_000 + seed * 101 + layer_index
        self.backend = LSHSelfAttention(
            dim=config.hidden_size,
            heads=config.heads,
            bucket_size=bucket_size,
            n_hashes=4,
            causal=True,
            dim_head=config.hidden_size // config.heads,
            attn_chunks=1,
            random_rotations_per_head=False,
            attend_across_buckets=True,
            allow_duplicate_attention=True,
            num_mem_kv=0,
            one_value_head=False,
            use_full_attn=False,
            full_attn_thres=0,
            post_attn_dropout=0.0,
            dropout=0.0,
            n_local_attn_heads=0,
        )

    def forward(
        self,
        value: torch.Tensor,
        rotary_emb: dense.b.RotaryEmbedding,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        frequencies = torch.outer(position_ids.float(), rotary_emb.inv_freq)
        reformer_rope = torch.cat(
            (frequencies.sin(), frequencies.cos()), dim=-1
        ).unsqueeze(0)
        devices: list[int] = []
        if value.device.type == "cuda":
            devices = [
                value.device.index
                if value.device.index is not None
                else torch.cuda.current_device()
            ]
        # Fixed per-layer rotations make hashes reproducible across validation
        # calls and preserve the declared seed contract.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.hash_seed)
            if value.device.type == "cuda":
                torch.cuda.manual_seed(self.hash_seed)
            return self.backend(value, pos_emb=reformer_rope)


class MoEBlock(torch.nn.Module):
    def __init__(
        self,
        config: dense.ModelConfig,
        method: str,
        method_value: int,
        seed: int,
        layer_index: int,
        moe_config: MoEConfig,
    ):
        super().__init__()
        self.method = method
        self.layer_index = layer_index
        self.ln1 = torch.nn.LayerNorm(
            config.hidden_size, elementwise_affine=True, bias=True
        )
        if method in ("linformer", "performer"):
            self.attn = dense.PartAAttention(config, method, seed)
        elif method == "longformer":
            self.attn = dense.LongformerSelfAttention(
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
            self.attn = dense.MemformerSegmentAttention(
                config.hidden_size,
                config.heads,
                memory_slots=method_value,
                causal=True,
                detach_memory=False,
                bias=False,
            )
        elif method == "reformer":
            self.attn = ReformerAttentionAdapter(
                config, method_value, seed, layer_index
            )
        elif method == "full_attention":
            self.attn = dense.FullAttentionAdapter(
                config.hidden_size, config.heads
            )
        else:
            raise ValueError(f"unsupported method: {method}")
        self.ln2 = torch.nn.LayerNorm(
            config.hidden_size, elementwise_affine=True, bias=True
        )
        self.is_moe = layer_index in moe_config.moe_layers
        self.ffn: torch.nn.Module
        if self.is_moe:
            self.ffn = SparseTop1MoE(
                config.hidden_size, config.ffn_size, moe_config
            )
        else:
            self.ffn = DenseFFN(config.hidden_size, config.ffn_size)
        self.dropout = torch.nn.Dropout(config.dropout)

    def feed_forward(
        self,
        hidden: torch.Tensor,
        return_router: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        normalized = self.ln2(hidden)
        if self.is_moe:
            output, diagnostics = self.ffn(normalized, return_router=return_router)
        else:
            output = self.ffn(normalized)
            diagnostics = None
        return hidden + self.dropout(output), diagnostics


class MoETinyLM(torch.nn.Module):
    def __init__(
        self,
        config: dense.ModelConfig,
        method: str,
        method_value: int,
        seed: int,
        moe_config: MoEConfig,
    ):
        super().__init__()
        moe_config.validate(config.layers)
        self.config = config
        self.method = method
        self.method_value = method_value
        self.seed = seed
        self.moe_config = moe_config
        self.token_embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.rope = dense.b.RotaryEmbedding(
            config.hidden_size // config.heads, config.rope_max_position
        )
        self.blocks = torch.nn.ModuleList(
            [
                MoEBlock(
                    config,
                    method,
                    method_value,
                    seed,
                    layer_index,
                    moe_config,
                )
                for layer_index in range(config.layers)
            ]
        )
        self.final_norm = torch.nn.LayerNorm(
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
            derived = int(
                hashlib.sha256(f"{seed}:{canonical}".encode()).hexdigest()[:16],
                16,
            )
            generator = torch.Generator(device="cpu").manual_seed(
                derived % (2**63 - 1)
            )
            if "ln" in name or name.startswith("final_norm"):
                if name.endswith("weight"):
                    torch.nn.init.ones_(parameter)
                else:
                    torch.nn.init.zeros_(parameter)
            elif name.endswith("router.weight"):
                torch.nn.init.normal_(
                    parameter, mean=0.0, std=0.005, generator=generator
                )
            elif parameter.ndim >= 2:
                torch.nn.init.normal_(
                    parameter, mean=0.0, std=0.02, generator=generator
                )
            else:
                torch.nn.init.zeros_(parameter)

    def forward(
        self,
        token_ids: torch.Tensor,
        return_router: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if token_ids.shape[1] > self.config.rope_max_position:
            raise ValueError("sequence exceeds configured RoPE maximum")
        hidden = self.token_embedding(token_ids)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        router_rows: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            normalized = block.ln1(hidden)
            if block.method == "longformer":
                attended = block.attn(
                    normalized,
                    rotary_emb=self.rope,
                    position_ids=positions,
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
            hidden, diagnostics = block.feed_forward(
                hidden, return_router=return_router
            )
            if diagnostics is not None:
                router_rows.append(diagnostics)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.token_embedding.weight)
        if not return_router:
            return logits
        if not router_rows:
            raise RuntimeError("router diagnostics requested without MoE layers")
        return logits, {
            "load_balance_loss": torch.stack(
                [row["load_balance_loss"] for row in router_rows]
            ).mean(),
            "router_z_loss": torch.stack(
                [row["router_z_loss"] for row in router_rows]
            ).mean(),
            "counts": torch.stack([row["counts"] for row in router_rows]).sum(0),
            "probability_mass": torch.stack(
                [row["probability_mass"] for row in router_rows]
            ).mean(0),
            "entropy": torch.stack([row["entropy"] for row in router_rows]).mean(),
            "mean_top_probability": torch.stack(
                [row["mean_top_probability"] for row in router_rows]
            ).mean(),
        }


def moe_parameter_record(model: MoETinyLM) -> dict[str, Any]:
    record = dense.parameter_record(model)
    expert = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".ffn.experts." in name
    )
    router = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".ffn.router." in name
    )
    per_expert_per_layer = 2 * model.config.hidden_size * model.config.ffn_size
    active = (
        record["total_parameters"]
        - expert
        + per_expert_per_layer * len(model.moe_config.moe_layers)
    )
    record.update(
        {
            "expert_parameters": expert,
            "router_parameters": router,
            "active_parameters_per_token_top1": active,
            "moe_layers": list(model.moe_config.moe_layers),
            "num_experts": model.moe_config.num_experts,
            "top_k": model.moe_config.top_k,
            "dropless": model.moe_config.dropless,
        }
    )
    return record


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
    full_validation: bool = True
    save_checkpoint: bool = True


def run_id_for(method: str, seed: int, train_tokens: int) -> str:
    budget = "10m" if train_tokens == 10_000_000 else str(train_tokens)
    role = "keyformer_base" if method == "full_attention" else method
    return f"moe4_top1_dropless_all6_{role}_s{seed}_{budget}"


def method_payload(
    model_config: dense.ModelConfig,
    request: TrainRequest,
) -> dict[str, Any]:
    return {
        "method": request.method,
        "method_value": request.method_value,
        "linformer_pool": request.method_value
        if request.method == "linformer"
        else None,
        "performer_features": request.method_value
        if request.method == "performer"
        else None,
        "longformer_left_window": request.method_value
        if request.method == "longformer"
        else None,
        "memformer_memory_slots": request.method_value
        if request.method == "memformer"
        else None,
        "memformer_segment_length": model_config.memformer_segment_length
        if request.method == "memformer"
        else None,
        "reformer_bucket_size": request.method_value
        if request.method == "reformer"
        else None,
        "reformer_n_hashes": 4 if request.method == "reformer" else None,
        "keyformer_role": "shared_full_attention_moe_backbone"
        if request.method == "full_attention"
        else None,
    }


def protocol_payload(
    request: TrainRequest,
    model_config: dense.ModelConfig,
    moe_config: MoEConfig,
    parameters: dict[str, Any],
    device: torch.device,
    train_meta: dict[str, Any],
    validation_meta: dict[str, Any],
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    reformer_path = ROOT / "reformer-pytorch/reformer_pytorch/reformer_pytorch.py"
    return {
        "protocol_version": "tinystories_tinylm_moe_v1",
        "execution_profile": "moe4_top1_dropless_all6_seed17",
        "request": asdict(request),
        "model": asdict(model_config),
        "moe": asdict(moe_config),
        "method_config": method_payload(model_config, request),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "minimum_learning_rate": 3e-5,
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": 0.1,
            "warmup_ratio": 0.03,
            "gradient_clip_norm": 1.0,
            "lm_loss": "token_weighted_cross_entropy",
            "load_balance_coefficient": moe_config.load_balance_coefficient,
            "router_z_loss_coefficient": moe_config.router_z_loss_coefficient,
            "validation_excludes_router_auxiliary_losses": True,
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
        "train_cache": train_meta,
        "validation_cache": validation_meta,
        "storage": {
            "data_dir": str(dense.b.DATA_DIR),
            "cache_dir": str(dense.b.CACHE_DIR),
            "runs_dir": str(RUNS_DIR),
            "aggregate_dir": str(AGGREGATE_DIR),
        },
        "parameters": parameters,
        "compute_dtype": "bfloat16" if device.type == "cuda" else "float32",
        "parameter_dtype": "float32",
        "environment": dense.b.environment_record(device),
        "source_sha256": {
            "moe_runner": sha256_file(runner_path),
            "unified_dense_runner": dense.sha256_file(Path(dense.__file__)),
            "part_a_models": dense.PART_A_SOURCE_HASH,
            "reformer": sha256_file(reformer_path),
        },
        "comparison_note": (
            "MoE is a new orthogonal factor. Results must be compared against "
            "the retained dense baselines, not merged into their statistics."
        ),
    }


def save_checkpoint(
    path: Path,
    model: MoETinyLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    request: TrainRequest,
    config_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "request": asdict(request),
            "model_config": asdict(model.config),
            "moe_config": asdict(model.moe_config),
            "config_hash": config_hash,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        temporary,
    )
    temporary.replace(path)


def completed_summary(request: TrainRequest) -> dict[str, Any] | None:
    path = RUNS_DIR / request.run_id / "summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") == "ok"
        and payload.get("method") == request.method
        and int(payload.get("seed", -1)) == request.seed
        and int(payload.get("train_tokens_completed", -1)) == request.train_tokens
        and (payload.get("moe_config") or {}).get("moe_layers")
        == list(range(6))
    ):
        return payload
    return None


def run_training(
    request: TrainRequest,
    device: torch.device,
    moe_config: MoEConfig,
) -> dict[str, Any]:
    if request.method not in ALL_TRAINING_METHODS:
        raise ValueError(request.method)
    cached = completed_summary(request)
    if cached is not None:
        print(f"[skip complete] {request.run_id}", flush=True)
        return cached
    effective_tokens = (
        request.context
        * request.micro_batch_sequences
        * request.gradient_accumulation_steps
    )
    if effective_tokens != 8192:
        raise ValueError(
            f"effective batch {effective_tokens} does not match protocol 8192"
        )
    dense.configure_cuda()
    device = torch.device(device)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / request.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    required_train_tokens = (
        math.ceil(request.train_tokens / request.context) * request.context + 1
    )
    train_path, train_meta = dense.b.build_token_cache(
        "train", required_train_tokens
    )
    validation_path, validation_meta = dense.b.build_token_cache(
        "validation", None
    )
    train_stream = dense.b.PackedTokenStream(
        train_path, int(train_meta["token_count"]), request.context
    )
    validation_stream = dense.b.PackedTokenStream(
        validation_path,
        int(validation_meta["token_count"]),
        request.context,
    )

    model_config = dense.ModelConfig(context=request.context)
    moe_config.validate(model_config.layers)
    seed_all(request.seed)
    model = MoETinyLM(
        model_config,
        request.method,
        request.method_value,
        request.seed,
        moe_config,
    ).to(device)
    parameters = moe_parameter_record(model)
    effective_config = protocol_payload(
        request,
        model_config,
        moe_config,
        parameters,
        device,
        train_meta,
        validation_meta,
    )
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

    initial = dense.evaluate(
        model,
        validation_stream,
        request.validation_probe_tokens,
        request.validation_batch_sequences,
        device,
    )
    model.train()
    best_probe = dict(initial)
    best_probe["step"] = 0
    best_state_cpu = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    best_step = 0
    tokens_seen = 0
    block_cursor = 0
    next_validation = request.validation_interval_tokens
    status = "ok"
    failure = None
    step = 0
    started = time.perf_counter()
    router_counts_total = torch.zeros(moe_config.num_experts, dtype=torch.long)
    router_probability_sum = torch.zeros(moe_config.num_experts, dtype=torch.float64)
    router_observations = 0
    entropy_weighted_sum = 0.0
    top_probability_weighted_sum = 0.0
    metrics_path = run_dir / "metrics.jsonl"

    print(
        f"[start] {request.run_id}: method={request.method} "
        f"seed={request.seed} steps={total_steps} params={parameters['total_parameters']}",
        flush=True,
    )
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        metrics_file.write(
            json.dumps(
                {
                    "event": "initial_validation_probe",
                    "step": 0,
                    "tokens_seen": 0,
                    **initial,
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
            step_balance = 0.0
            step_z_loss = 0.0
            step_entropy = 0.0
            step_top_probability = 0.0
            step_router_counts = torch.zeros(
                moe_config.num_experts, dtype=torch.long
            )
            step_started = time.perf_counter()
            try:
                while step_predictions < step_target:
                    micro_limit = request.micro_batch_sequences * request.context
                    valid_predictions = min(
                        micro_limit, step_target - step_predictions
                    )
                    sequences = math.ceil(valid_predictions / request.context)
                    inputs, targets, valid = train_stream.batch(
                        block_cursor,
                        sequences,
                        valid_predictions,
                        device,
                    )
                    block_cursor += sequences
                    with dense.autocast_context(device):
                        logits, router = model(inputs, return_router=True)
                    loss_sum = dense.masked_loss_sum(logits, targets, valid)
                    micro_weight = valid_predictions / step_target
                    auxiliary = (
                        moe_config.load_balance_coefficient
                        * router["load_balance_loss"]
                        + moe_config.router_z_loss_coefficient
                        * router["router_z_loss"]
                    )
                    (loss_sum / step_target + micro_weight * auxiliary).backward()
                    step_loss_sum += float(loss_sum.detach().cpu())
                    step_predictions += valid_predictions
                    step_balance += micro_weight * float(
                        router["load_balance_loss"].detach().cpu()
                    )
                    step_z_loss += micro_weight * float(
                        router["router_z_loss"].detach().cpu()
                    )
                    step_entropy += micro_weight * float(
                        router["entropy"].detach().cpu()
                    )
                    step_top_probability += micro_weight * float(
                        router["mean_top_probability"].detach().cpu()
                    )
                    counts_cpu = router["counts"].detach().cpu()
                    step_router_counts += counts_cpu
                    observations = int(counts_cpu.sum())
                    router_counts_total += counts_cpu
                    router_probability_sum += (
                        router["probability_mass"].detach().double().cpu()
                        * observations
                    )
                    router_observations += observations
                    entropy_weighted_sum += float(
                        router["entropy"].detach().cpu()
                    ) * observations
                    top_probability_weighted_sum += float(
                        router["mean_top_probability"].detach().cpu()
                    ) * observations

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                )
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"non-finite gradient norm {grad_norm}"
                    )
                learning_rate = dense.learning_rate_for_step(
                    step, total_steps, 3e-4, 3e-5, 0.03
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
            route_total = max(int(step_router_counts.sum()), 1)
            record: dict[str, Any] = {
                "event": "train_step",
                "step": step,
                "tokens_seen": tokens_seen,
                "train_nll": step_loss_sum / max(step_target, 1),
                "train_ppl": math.exp(
                    min(step_loss_sum / max(step_target, 1), 20.0)
                ),
                "router_load_balance_loss": step_balance,
                "router_z_loss": step_z_loss,
                "router_entropy": step_entropy,
                "router_mean_top_probability": step_top_probability,
                "router_token_counts": step_router_counts.tolist(),
                "router_route_fractions": (
                    step_router_counts.double() / route_total
                ).tolist(),
                "gradient_norm": grad_norm,
                "learning_rate": learning_rate,
                "step_seconds": step_seconds,
                "tokens_per_second": step_target / max(step_seconds, 1e-9),
            }
            if (
                tokens_seen >= next_validation
                or tokens_seen == request.train_tokens
            ):
                probe = dense.evaluate(
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
                    best_step = step
                    best_state_cpu = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                while next_validation <= tokens_seen:
                    next_validation += request.validation_interval_tokens
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            if step == 1 or "validation_probe" in record or step == total_steps:
                probe_text = ""
                if "validation_probe" in record:
                    probe_text = (
                        f"probe_nll={record['validation_probe']['token_weighted_nll']:.4f} "
                    )
                print(
                    f"[{request.run_id}] step {step}/{total_steps} "
                    f"tokens={tokens_seen} train_nll={record['train_nll']:.4f} "
                    f"{probe_text}tok/s={record['tokens_per_second']:.0f} "
                    f"routes={record['router_route_fractions']}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_probe = dense.evaluate(
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
        full_validation = dense.evaluate(
            model,
            validation_stream,
            validation_stream.prediction_count,
            request.validation_batch_sequences,
            device,
        )
        final_state_cpu = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(best_state_cpu, strict=True)
        best_full_validation = dense.evaluate(
            model,
            validation_stream,
            validation_stream.prediction_count,
            request.validation_batch_sequences,
            device,
        )
        model.load_state_dict(final_state_cpu, strict=True)
        model.train()
    if request.save_checkpoint and status == "ok":
        save_checkpoint(
            run_dir / "final.pt",
            model,
            optimizer,
            step,
            tokens_seen,
            request,
            effective_config["config_hash"],
        )

    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    total_routes = max(int(router_counts_total.sum()), 1)
    summary = {
        "protocol_version": "tinystories_tinylm_moe_v1",
        "execution_profile": "moe4_top1_dropless_all6_seed17",
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
        "moe_config": asdict(moe_config),
        "router_training": {
            "token_assignments": router_counts_total.tolist(),
            "route_fractions": (
                router_counts_total.double() / total_routes
            ).tolist(),
            "mean_probability_mass": (
                router_probability_sum / max(router_observations, 1)
            ).tolist(),
            "mean_entropy": entropy_weighted_sum
            / max(router_observations, 1),
            "mean_top_probability": top_probability_weighted_sum
            / max(router_observations, 1),
        },
        "model_parameter_digest": dense.model_digest(model),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "config_hash": effective_config["config_hash"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "completed_at_utc": now_utc(),
    }
    atomic_json(run_dir / "summary.json", summary)
    validation_text = ""
    if full_validation is not None:
        validation_text = (
            f"val_nll={full_validation['token_weighted_nll']:.6f} "
            f"ppl={full_validation['perplexity']:.4f} "
        )
    print(
        f"[done] {request.run_id} status={status} {validation_text}"
        f"train_tok/s={summary['mean_training_tokens_per_second']:.1f}",
        flush=True,
    )
    del model, optimizer, best_state_cpu
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary
