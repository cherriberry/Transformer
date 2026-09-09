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
            expert_output = expert(expert_input).to(combined.dtype)
            gate = top_probability.index_select(0, token_indices).to(
                combined.dtype
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
    """Strictly causal fixed-hash LSH attention with shared query/key weights.

    The commonly used sort-and-chunk implementation can let future-token bucket
    occupancy change the candidate set for an earlier query.  Here every token
    is independently hashed with fixed seeded rotations, and the attention mask
    only admits same-bucket keys at positions not later than the query.  The
    implementation materializes the mask at the 512-token training length; it
    prioritizes causal correctness over a fused long-sequence kernel benchmark.
    """

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
        self.dim = config.hidden_size
        self.heads = config.heads
        self.head_dim = config.hidden_size // config.heads
        self.bucket_size = bucket_size
        self.n_hashes = 4
        self.to_qk = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.to_out = torch.nn.Linear(self.dim, self.dim, bias=False)
        maximum_buckets = config.context // bucket_size
        if maximum_buckets % 2:
            raise ValueError("Reformer requires an even number of buckets")
        generator = torch.Generator(device="cpu").manual_seed(
            50_000 + seed * 101 + layer_index
        )
        rotations = torch.randn(
            self.n_hashes,
            self.head_dim,
            maximum_buckets // 2,
            generator=generator,
        )
        self.register_buffer("hash_rotations", rotations, persistent=True)

    def forward(
        self,
        value: torch.Tensor,
        rotary_emb: dense.b.RotaryEmbedding,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, _ = value.shape
        if tokens % self.bucket_size:
            raise ValueError(
                f"sequence length {tokens} must divide bucket size {self.bucket_size}"
            )
        n_buckets = tokens // self.bucket_size
        if n_buckets < 2 or n_buckets % 2:
            raise ValueError("the number of Reformer buckets must be positive and even")

        def split(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(
                batch, tokens, self.heads, self.head_dim
            ).transpose(1, 2)

        query_key = split(self.to_qk(value))
        projected_value = split(self.to_v(value))
        query_key, _ = rotary_emb.apply_qk(
            query_key, query_key, position_ids
        )
        rotations = self.hash_rotations[..., : n_buckets // 2].to(
            query_key.dtype
        )
        projected_hashes = torch.einsum(
            "bhnd,rdk->bhrnk", query_key, rotations
        )
        projected_hashes = torch.cat(
            (projected_hashes, -projected_hashes), dim=-1
        )
        buckets = projected_hashes.argmax(dim=-1)
        allowed = torch.zeros(
            batch,
            self.heads,
            tokens,
            tokens,
            dtype=torch.bool,
            device=value.device,
        )
        for hash_index in range(self.n_hashes):
            identifiers = buckets[:, :, hash_index]
            allowed |= identifiers.unsqueeze(-1) == identifiers.unsqueeze(-2)
        causal = torch.ones(
            tokens, tokens, dtype=torch.bool, device=value.device
        ).tril()
        allowed &= causal.view(1, 1, tokens, tokens)
        scores = torch.matmul(
            query_key, query_key.transpose(-1, -2)
        ) * (self.head_dim**-0.5)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1).to(
            projected_value.dtype
        )
        attended = torch.matmul(probabilities, projected_value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch, tokens, self.dim
        )
        return self.to_out(attended)


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
        "reformer_implementation": "strict_causal_fixed_hash_same_bucket_mask"
        if request.method == "reformer"
        else None,
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
        "reformer_adapter_note": (
            "The local runner uses a deterministic same-bucket causal mask. "
            "The vendored sort-and-chunk causal path failed future-invariance "
            "under future-token perturbation and is retained only as provenance."
        ),
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


def build_cache_policy(
    name: str,
    budget: int,
    recent_window: int,
    seed: int,
) -> FullKVPolicy | KeyformerPolicy:
    if name == "full_kv":
        return FullKVPolicy()
    if name == "keyformer":
        return KeyformerPolicy(
            budget=budget,
            recent_window=recent_window,
            tau_init=1.0,
            tau_delta=0.01,
            gumbel_noise=True,
            seed=seed,
        )
    raise ValueError(name)


class MoEKeyformerEngine:
    """Incremental Full-Attention + MoE inference with physical KV pruning."""

    def __init__(
        self,
        model: MoETinyLM,
        policy_name: str,
        total_context: int,
        cache_ratio: float,
        recent_ratio: float,
        seed: int,
    ):
        if model.method != "full_attention":
            raise ValueError("Keyformer requires a Full-Attention backbone")
        if not 0.0 < cache_ratio <= 1.0:
            raise ValueError("cache_ratio must be in (0, 1]")
        self.model = model.eval()
        self.policy_name = policy_name
        self.total_context = total_context
        self.cache_ratio = cache_ratio
        self.recent_ratio = recent_ratio
        self.seed = seed
        self.budget = (
            total_context
            if policy_name == "full_kv"
            else max(1, math.ceil(cache_ratio * total_context))
        )
        self.recent_window = min(
            self.budget, math.floor(self.budget * recent_ratio)
        )
        self.reset()

    def reset(self, seed_offset: int = 0) -> None:
        self.policies = [
            build_cache_policy(
                self.policy_name,
                self.budget,
                self.recent_window,
                self.seed + seed_offset * 997 + layer_index,
            )
            for layer_index in range(len(self.model.blocks))
        ]
        self.caches: list[KVCacheState | None] = [
            None for _ in self.model.blocks
        ]
        self.seen_tokens = 0

    def _split(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        heads = self.model.config.heads
        head_dim = self.model.config.hidden_size // heads
        return value.view(batch, tokens, heads, head_dim).transpose(1, 2)

    def _merge(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, tokens, _ = value.shape
        return value.transpose(1, 2).contiguous().view(
            batch, tokens, self.model.config.hidden_size
        )

    def _attention(
        self,
        block: MoEBlock,
        hidden: torch.Tensor,
        layer_index: int,
        position_ids: torch.Tensor,
        prefill: bool,
    ) -> torch.Tensor:
        attention = block.attn
        query = self._split(attention.to_q(hidden))
        key = self._split(attention.to_k(hidden))
        value = self._split(attention.to_v(hidden))
        query, key = self.model.rope.apply_qk(
            query, key, position_ids
        )
        scale = (self.model.config.hidden_size // self.model.config.heads) ** -0.5
        policy = self.policies[layer_index]
        state = self.caches[layer_index]

        if prefill or state is None:
            tokens = hidden.shape[1]
            logits = torch.matmul(query, key.transpose(-1, -2)) * scale
            causal = torch.ones(
                tokens, tokens, dtype=torch.bool, device=hidden.device
            ).tril()
            masked = logits.masked_fill(
                ~causal, torch.finfo(logits.dtype).min
            )
            probabilities = torch.softmax(masked.float(), dim=-1).to(value.dtype)
            context = torch.matmul(probabilities, value)
            score_probabilities = policy.score_probabilities(
                masked.float(), decode_step=0
            )
            scores = score_probabilities.sum(dim=-2).float()
            positions = position_ids.view(1, 1, tokens).expand(
                hidden.shape[0], self.model.config.heads, tokens
            )
            state = KVCacheState(
                keys=key,
                values=value,
                positions=positions,
                scores=scores,
                decode_step=0,
                seen_tokens=tokens,
            )
            self.caches[layer_index] = policy.select(state)
        else:
            if hidden.shape[1] != 1:
                raise ValueError("decode path accepts exactly one token")
            keys = torch.cat((state.keys, key), dim=2)
            values = torch.cat((state.values, value), dim=2)
            next_position = position_ids.view(1, 1, 1).expand(
                hidden.shape[0], self.model.config.heads, 1
            )
            positions = torch.cat((state.positions, next_position), dim=-1)
            logits = torch.matmul(query, keys.transpose(-1, -2)) * scale
            probabilities = torch.softmax(logits.float(), dim=-1).to(
                values.dtype
            )
            context = torch.matmul(probabilities, values)
            decode_step = state.decode_step + 1
            score_row = policy.score_probabilities(
                logits.float(), decode_step=decode_step
            ).squeeze(-2)
            scores = torch.cat(
                (
                    state.scores,
                    torch.zeros_like(score_row[..., :1]),
                ),
                dim=-1,
            )
            scores = scores + score_row.to(scores.dtype)
            updated = KVCacheState(
                keys=keys,
                values=values,
                positions=positions,
                scores=scores,
                decode_step=decode_step,
                seen_tokens=state.seen_tokens + 1,
            )
            self.caches[layer_index] = policy.select(updated)

        return attention.to_out(self._merge(context))

    def _block(
        self,
        block: MoEBlock,
        hidden: torch.Tensor,
        layer_index: int,
        position_ids: torch.Tensor,
        prefill: bool,
    ) -> torch.Tensor:
        attended = self._attention(
            block,
            block.ln1(hidden),
            layer_index,
            position_ids,
            prefill,
        )
        hidden = hidden + block.dropout(attended)
        hidden, _ = block.feed_forward(hidden, return_router=False)
        return hidden

    def prefill(
        self,
        token_ids: torch.Tensor,
        seed_offset: int = 0,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("Keyformer engine currently requires batch size one")
        self.reset(seed_offset=seed_offset)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.model.token_embedding(token_ids)
        for layer_index, block in enumerate(self.model.blocks):
            hidden = self._block(
                block,
                hidden,
                layer_index,
                positions,
                prefill=True,
            )
        self.seen_tokens = token_ids.shape[1]
        hidden = self.model.final_norm(hidden)
        return F.linear(hidden, self.model.token_embedding.weight)

    def decode_step(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape != (1, 1):
            raise ValueError("decode_step requires token_ids shape [1, 1]")
        if self.seen_tokens <= 0 or any(
            state is None for state in self.caches
        ):
            raise RuntimeError("prefill must run before decode_step")
        positions = torch.tensor(
            [self.seen_tokens], device=token_ids.device, dtype=torch.long
        )
        hidden = self.model.token_embedding(token_ids)
        for layer_index, block in enumerate(self.model.blocks):
            hidden = self._block(
                block,
                hidden,
                layer_index,
                positions,
                prefill=False,
            )
        self.seen_tokens += 1
        hidden = self.model.final_norm(hidden)
        return F.linear(hidden, self.model.token_embedding.weight)

    def cache_summary(self) -> dict[str, Any]:
        states = [state for state in self.caches if state is not None]
        if len(states) != len(self.caches):
            raise RuntimeError("cache is not initialized")
        return {
            "seen_tokens": self.seen_tokens,
            "cached_tokens_max": max(state.num_tokens for state in states),
            "kv_bytes_total": sum(state.storage_bytes for state in states),
            "layers": len(states),
            "budget": self.budget,
            "recent_window": self.recent_window,
        }


def load_full_moe_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MoETinyLM, dict[str, Any]]:
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = dense.ModelConfig(**payload["model_config"])
    moe_payload = dict(payload["moe_config"])
    moe_payload["moe_layers"] = tuple(moe_payload["moe_layers"])
    moe_config = MoEConfig(**moe_payload)
    request = payload["request"]
    if request["method"] != "full_attention":
        raise ValueError("checkpoint is not a Full-Attention + MoE backbone")
    model = MoETinyLM(
        model_config,
        "full_attention",
        int(request["method_value"]),
        int(request["seed"]),
        moe_config,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def evaluate_keyformer_policy(
    model: MoETinyLM,
    policy_name: str,
    validation_stream: dense.b.PackedTokenStream,
    prediction_limit: int,
    context: int,
    prefill_tokens: int,
    cache_ratio: float,
    recent_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    prediction_limit = min(prediction_limit, validation_stream.prediction_count)
    consumed = 0
    block_index = 0
    loss_sum = 0.0
    top1_correct = 0
    peak_cache_bytes = 0
    peak_cached_tokens = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    while consumed < prediction_limit:
        valid_predictions = min(context, prediction_limit - consumed)
        inputs, targets, _ = validation_stream.batch(
            block_index,
            1,
            valid_predictions,
            device,
        )
        prompt = min(prefill_tokens, valid_predictions)
        engine = MoEKeyformerEngine(
            model,
            policy_name,
            total_context=context,
            cache_ratio=cache_ratio,
            recent_ratio=recent_ratio,
            seed=seed,
        )
        with dense.autocast_context(device):
            logits = engine.prefill(
                inputs[:, :prompt], seed_offset=block_index
            )
        prompt_logits = logits[:, :prompt].float()
        prompt_targets = targets[:, :prompt]
        loss_sum += float(
            F.cross_entropy(
                prompt_logits.reshape(-1, prompt_logits.shape[-1]),
                prompt_targets.reshape(-1),
                reduction="sum",
            ).cpu()
        )
        top1_correct += int(
            (prompt_logits.argmax(dim=-1) == prompt_targets).sum().cpu()
        )

        for token_index in range(prompt, valid_predictions):
            with dense.autocast_context(device):
                logits = engine.decode_step(
                    inputs[:, token_index : token_index + 1]
                )
            target = targets[:, token_index]
            row = logits[:, -1].float()
            loss_sum += float(F.cross_entropy(row, target, reduction="sum").cpu())
            top1_correct += int((row.argmax(dim=-1) == target).sum().cpu())

        cache = engine.cache_summary()
        peak_cache_bytes = max(peak_cache_bytes, int(cache["kv_bytes_total"]))
        peak_cached_tokens = max(
            peak_cached_tokens, int(cache["cached_tokens_max"])
        )
        consumed += valid_predictions
        block_index += 1
        if block_index == 1 or consumed == prediction_limit or block_index % 16 == 0:
            print(
                f"[keyformer:{policy_name}] tokens={consumed}/{prediction_limit} "
                f"cached={cache['cached_tokens_max']}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    nll = loss_sum / max(consumed, 1)
    peak_allocated = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    return {
        "status": "ok",
        "policy": policy_name,
        "seed": seed,
        "context": context,
        "prefill_tokens": prefill_tokens,
        "cache_ratio": 1.0 if policy_name == "full_kv" else cache_ratio,
        "recent_ratio": recent_ratio,
        "tokens_evaluated": consumed,
        "token_weighted_nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "top1_accuracy": top1_correct / max(consumed, 1),
        "elapsed_seconds": elapsed,
        "tokens_per_second": consumed / max(elapsed, 1e-9),
        "peak_cached_tokens": peak_cached_tokens,
        "peak_kv_bytes": peak_cache_bytes,
        "peak_allocated_bytes": peak_allocated,
    }


def run_keyformer_suite(
    device: torch.device,
    seed: int,
    train_tokens: int,
    prediction_limit: int,
    context: int = 512,
    prefill_tokens: int = 128,
) -> dict[str, Any]:
    base_run_id = run_id_for("full_attention", seed, train_tokens)
    checkpoint_path = RUNS_DIR / base_run_id / "final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"train the Full-Attention + MoE base first: {checkpoint_path}"
        )
    output_path = AGGREGATE_DIR / "keyformer_evaluation.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "ok"
            and existing.get("seed") == seed
            and existing.get("prediction_limit") == prediction_limit
            and existing.get("base_checkpoint") == str(checkpoint_path)
        ):
            print("[skip complete] Keyformer evaluation", flush=True)
            return existing

    model, checkpoint = load_full_moe_checkpoint(checkpoint_path, device)
    validation_path, validation_meta = dense.b.build_token_cache(
        "validation", None
    )
    validation_stream = dense.b.PackedTokenStream(
        validation_path,
        int(validation_meta["token_count"]),
        context,
    )
    rows = []
    for policy_name, ratio in (("full_kv", 1.0), ("keyformer", 0.5)):
        row = evaluate_keyformer_policy(
            model,
            policy_name,
            validation_stream,
            prediction_limit,
            context,
            prefill_tokens,
            ratio,
            0.5,
            seed,
            device,
        )
        rows.append(row)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    full_row = next(row for row in rows if row["policy"] == "full_kv")
    keyformer_row = next(row for row in rows if row["policy"] == "keyformer")
    payload = {
        "protocol_version": "tinystories_tinylm_moe_v1",
        "status": "ok",
        "seed": seed,
        "prediction_limit": prediction_limit,
        "base_checkpoint": str(checkpoint_path),
        "base_config_hash": checkpoint["config_hash"],
        "base_runner_sha256": checkpoint.get("runner_sha256"),
        "evaluation_runner_sha256": sha256_file(Path(__file__).resolve()),
        "validation_cache": validation_meta,
        "rows": rows,
        "keyformer_minus_full": {
            "nll": keyformer_row["token_weighted_nll"]
            - full_row["token_weighted_nll"],
            "ppl": keyformer_row["perplexity"] - full_row["perplexity"],
            "kv_bytes_saved_fraction": 1.0
            - keyformer_row["peak_kv_bytes"] / full_row["peak_kv_bytes"],
            "throughput_ratio": keyformer_row["tokens_per_second"]
            / full_row["tokens_per_second"],
        },
        "completed_at_utc": now_utc(),
    }
    atomic_json(output_path, payload)
    print(
        f"[done] keyformer nll={keyformer_row['token_weighted_nll']:.6f} "
        f"full_nll={full_row['token_weighted_nll']:.6f} "
        f"kv_saved={payload['keyformer_minus_full']['kv_bytes_saved_fraction']:.2%}",
        flush=True,
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def small_model_config() -> dense.ModelConfig:
    return dense.ModelConfig(
        vocab_size=128,
        layers=6,
        hidden_size=64,
        heads=4,
        ffn_size=128,
        context=32,
        rope_max_position=256,
        dropout=0.0,
        longformer_query_chunk=8,
        memformer_segment_length=8,
        lin_chunk=8,
        lin_pool=16,
        performer_features=32,
        performer_redraw_interval=0,
    )


@torch.no_grad()
def future_invariance_delta(
    model: MoETinyLM,
    sample: torch.Tensor,
    split: int,
) -> float:
    model.eval()
    changed = sample.clone()
    changed[:, split:] = (changed[:, split:] + 19) % model.config.vocab_size
    left = model(sample)[:, :split].float()
    right = model(changed)[:, :split].float()
    return float((left - right).abs().max().cpu())


def correctness(device: torch.device) -> dict[str, Any]:
    dense.configure_cuda()
    device = torch.device(device)
    config = small_model_config()
    moe_config = MoEConfig(moe_layers=tuple(range(config.layers)))
    seed_all(17)
    sample = torch.randint(
        0, config.vocab_size, (2, config.context), device=device
    )
    values = {
        "memformer": 8,
        "linformer": 16,
        "performer": 32,
        "longformer": 8,
        "reformer": 8,
        "full_attention": 0,
    }
    rows: dict[str, Any] = {}
    for method in ALL_TRAINING_METHODS:
        seed_all(17)
        model = MoETinyLM(
            config, method, values[method], 17, moe_config
        ).to(device)
        model.train()
        model.zero_grad(set_to_none=True)
        with dense.autocast_context(device):
            logits, router = model(sample, return_router=True)
        objective = F.cross_entropy(
            logits.float().reshape(-1, config.vocab_size),
            sample.reshape(-1),
        )
        objective = (
            objective
            + moe_config.load_balance_coefficient
            * router["load_balance_loss"]
            + moe_config.router_z_loss_coefficient
            * router["router_z_loss"]
        )
        objective.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        grad_finite = bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        unrouted = [
            name
            for name, parameter in model.named_parameters()
            if ".ffn.experts." in name and parameter.grad is None
        ]
        future_delta = future_invariance_delta(model, sample, 16)
        rows[method] = {
            "shape": list(logits.shape),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "loss_finite": bool(torch.isfinite(objective)),
            "gradients_finite": grad_finite,
            "unrouted_expert_parameters": unrouted,
            "future_invariance_max_abs": future_delta,
            "router_counts": router["counts"].detach().cpu().tolist(),
            "router_probability_mass": router[
                "probability_mass"
            ].detach().cpu().tolist(),
            "parameters": moe_parameter_record(model),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_all(17)
    full_model = MoETinyLM(
        config, "full_attention", 0, 17, moe_config
    ).to(device).eval()
    prefix = sample[:1, :16]
    next_token = sample[:1, 16:17]
    with torch.no_grad(), dense.autocast_context(device):
        direct_prefill = full_model(prefix)
        direct_decode = full_model(sample[:1, :17])[:, -1:]
        full_engine = MoEKeyformerEngine(
            full_model,
            "full_kv",
            total_context=32,
            cache_ratio=1.0,
            recent_ratio=0.5,
            seed=17,
        )
        engine_prefill = full_engine.prefill(prefix)
        engine_decode = full_engine.decode_step(next_token)
        keyformer_ratio_one = MoEKeyformerEngine(
            full_model,
            "keyformer",
            total_context=32,
            cache_ratio=1.0,
            recent_ratio=0.5,
            seed=17,
        )
        key_prefill = keyformer_ratio_one.prefill(prefix)
        key_decode = keyformer_ratio_one.decode_step(next_token)
    keyformer_checks = {
        "fullkv_prefill_max_abs": float(
            (direct_prefill.float() - engine_prefill.float()).abs().max().cpu()
        ),
        "fullkv_decode_max_abs": float(
            (direct_decode.float() - engine_decode.float()).abs().max().cpu()
        ),
        "ratio_one_prefill_max_abs": float(
            (engine_prefill.float() - key_prefill.float()).abs().max().cpu()
        ),
        "ratio_one_decode_max_abs": float(
            (engine_decode.float() - key_decode.float()).abs().max().cpu()
        ),
        "ratio_one_cached_tokens": keyformer_ratio_one.cache_summary()[
            "cached_tokens_max"
        ],
    }
    tolerance = 0.02 if device.type == "cuda" else 1e-5
    method_ok = all(
        row["logits_finite"]
        and row["loss_finite"]
        and row["gradients_finite"]
        and row["future_invariance_max_abs"] < tolerance
        for row in rows.values()
    )
    keyformer_ok = (
        keyformer_checks["fullkv_prefill_max_abs"] < tolerance
        and keyformer_checks["fullkv_decode_max_abs"] < tolerance
        and keyformer_checks["ratio_one_prefill_max_abs"] == 0.0
        and keyformer_checks["ratio_one_decode_max_abs"] == 0.0
    )
    payload = {
        "protocol_version": "tinystories_tinylm_moe_v1",
        "status": "passed" if method_ok and keyformer_ok else "failed",
        "seed": 17,
        "moe_config": asdict(moe_config),
        "methods": rows,
        "keyformer": keyformer_checks,
        "tolerance": tolerance,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "completed_at_utc": now_utc(),
    }
    atomic_json(AGGREGATE_DIR / "correctness.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    del full_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def aggregate(seed: int, train_tokens: int) -> dict[str, Any]:
    training_rows = []
    for method in ALL_TRAINING_METHODS:
        run_id = run_id_for(method, seed, train_tokens)
        path = RUNS_DIR / run_id / "summary.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        validation = row.get("full_validation") or {}
        training_rows.append(
            {
                "method": method,
                "role": "keyformer_shared_backbone"
                if method == "full_attention"
                else "trainable_backbone",
                "run_id": run_id,
                "status": row.get("status"),
                "seed": row.get("seed"),
                "train_tokens": row.get("train_tokens_completed"),
                "validation_tokens": validation.get("tokens"),
                "validation_nll": validation.get("token_weighted_nll"),
                "validation_ppl": validation.get("perplexity"),
                "training_tokens_per_second": row.get(
                    "mean_training_tokens_per_second"
                ),
                "peak_allocated_gib": (
                    (row.get("peak_allocated_bytes") or 0) / 2**30
                ),
                "parameters": (row.get("parameters") or {}).get(
                    "total_parameters"
                ),
                "active_parameters_per_token": (
                    row.get("parameters") or {}
                ).get("active_parameters_per_token_top1"),
                "route_fractions": (row.get("router_training") or {}).get(
                    "route_fractions"
                ),
            }
        )
    keyformer_path = AGGREGATE_DIR / "keyformer_evaluation.json"
    keyformer = (
        json.loads(keyformer_path.read_text(encoding="utf-8"))
        if keyformer_path.exists()
        else None
    )
    correctness_path = AGGREGATE_DIR / "correctness.json"
    correctness_result = (
        json.loads(correctness_path.read_text(encoding="utf-8"))
        if correctness_path.exists()
        else None
    )
    expected = set(ALL_TRAINING_METHODS)
    completed = {
        row["method"]
        for row in training_rows
        if row["status"] == "ok" and row["train_tokens"] == train_tokens
    }
    status = (
        "complete"
        if completed == expected
        and keyformer is not None
        and keyformer.get("status") == "ok"
        and correctness_result is not None
        and correctness_result.get("status") == "passed"
        else "partial"
    )
    payload = {
        "protocol_version": "tinystories_tinylm_moe_v1",
        "status": status,
        "seed": seed,
        "train_tokens_per_backbone": train_tokens,
        "moe_config": asdict(MoEConfig()),
        "training_rows": training_rows,
        "keyformer_evaluation": keyformer,
        "correctness": correctness_result,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "generated_at_utc": now_utc(),
    }
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(AGGREGATE_DIR / "summary.json", payload)

    lines = [
        "# TinyLM MoE seed-17 results",
        "",
        "All six Transformer layers use four experts, top-1 routing, and dropless dispatch.",
        "",
        "## Trainable backbones and the shared Keyformer base",
        "",
        "| method | role | status | val NLL | val PPL | train tok/s | peak GiB | params | active params/token |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in training_rows:
        def show(value: Any, digits: int = 4) -> str:
            if value is None:
                return "N/A"
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)

        lines.append(
            f"| {row['method']} | {row['role']} | {row['status']} | "
            f"{show(row['validation_nll'], 6)} | {show(row['validation_ppl'], 4)} | "
            f"{show(row['training_tokens_per_second'], 1)} | "
            f"{show(row['peak_allocated_gib'], 3)} | {show(row['parameters'])} | "
            f"{show(row['active_parameters_per_token'])} |"
        )
    lines.extend(["", "## Keyformer KV-cache evaluation", ""])
    if keyformer is None:
        lines.append("Keyformer evaluation has not completed.")
    else:
        lines.extend(
            [
                "| policy | tokens | NLL | PPL | tok/s | peak KV bytes |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in keyformer.get("rows", []):
            lines.append(
                f"| {row['policy']} | {row['tokens_evaluated']} | "
                f"{row['token_weighted_nll']:.6f} | {row['perplexity']:.4f} | "
                f"{row['tokens_per_second']:.1f} | {row['peak_kv_bytes']} |"
            )
    (AGGREGATE_DIR / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return payload


def make_request(
    method: str,
    seed: int,
    train_tokens: int,
    validation_probe_tokens: int,
    full_validation: bool,
    save_checkpoint_flag: bool,
) -> TrainRequest:
    return TrainRequest(
        method=method,
        method_value=DEFAULT_VALUES[method],
        run_id=run_id_for(method, seed, train_tokens),
        seed=seed,
        train_tokens=train_tokens,
        validation_probe_tokens=validation_probe_tokens,
        full_validation=full_validation,
        save_checkpoint=save_checkpoint_flag,
    )


def run_priority(
    device: torch.device,
    seed: int,
    train_tokens: int,
    validation_probe_tokens: int,
    full_validation: bool,
    save_checkpoint_flag: bool,
    keyformer_predictions: int,
) -> None:
    moe_config = MoEConfig()
    for method in ("memformer", "full_attention"):
        request = make_request(
            method,
            seed,
            train_tokens,
            validation_probe_tokens,
            full_validation,
            save_checkpoint_flag,
        )
        result = run_training(request, device, moe_config)
        if result.get("status") != "ok":
            raise RuntimeError(f"{method} training failed: {result.get('failure')}")
    if not save_checkpoint_flag:
        print(
            "[skip] Keyformer evaluation requires a saved Full-Attention checkpoint",
            flush=True,
        )
        return
    run_keyformer_suite(
        device,
        seed,
        train_tokens,
        keyformer_predictions,
    )


def run_remaining(
    device: torch.device,
    seed: int,
    train_tokens: int,
    validation_probe_tokens: int,
    full_validation: bool,
    save_checkpoint_flag: bool,
) -> None:
    moe_config = MoEConfig()
    for method in ("linformer", "performer", "longformer", "reformer"):
        request = make_request(
            method,
            seed,
            train_tokens,
            validation_probe_tokens,
            full_validation,
            save_checkpoint_flag,
        )
        result = run_training(request, device, moe_config)
        if result.get("status") != "ok":
            raise RuntimeError(f"{method} training failed: {result.get('failure')}")


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-tokens", type=int, default=10_000_000)
    parser.add_argument(
        "--validation-probe-tokens", type=int, default=262_144
    )
    parser.add_argument(
        "--full-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keyformer-predictions", type=int, default=32_768
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("correctness")
    check.add_argument("--device", default="cuda:0")

    train = subparsers.add_parser("train")
    add_run_arguments(train)
    train.add_argument("--method", choices=ALL_TRAINING_METHODS, required=True)

    priority = subparsers.add_parser("priority")
    add_run_arguments(priority)

    remaining = subparsers.add_parser("remaining")
    add_run_arguments(remaining)

    all_runs = subparsers.add_parser("all")
    add_run_arguments(all_runs)

    keyformer = subparsers.add_parser("keyformer")
    keyformer.add_argument("--device", default="cuda:0")
    keyformer.add_argument("--seed", type=int, default=17)
    keyformer.add_argument("--train-tokens", type=int, default=10_000_000)
    keyformer.add_argument("--predictions", type=int, default=32_768)

    aggregation = subparsers.add_parser("aggregate")
    aggregation.add_argument("--seed", type=int, default=17)
    aggregation.add_argument("--train-tokens", type=int, default=10_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "aggregate":
        aggregate(args.seed, args.train_tokens)
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.command == "correctness":
        result = correctness(device)
        if result["status"] != "passed":
            raise SystemExit("correctness checks failed")
        return
    if args.command == "keyformer":
        run_keyformer_suite(
            device,
            args.seed,
            args.train_tokens,
            args.predictions,
        )
        return
    if args.command == "train":
        request = make_request(
            args.method,
            args.seed,
            args.train_tokens,
            args.validation_probe_tokens,
            args.full_validation,
            args.save_checkpoint,
        )
        result = run_training(request, device, MoEConfig())
        if result.get("status") != "ok":
            raise SystemExit(result.get("failure") or "training failed")
        return
    if args.command in ("priority", "all"):
        run_priority(
            device,
            args.seed,
            args.train_tokens,
            args.validation_probe_tokens,
            args.full_validation,
            args.save_checkpoint,
            args.keyformer_predictions,
        )
    if args.command in ("remaining", "all"):
        run_remaining(
            device,
            args.seed,
            args.train_tokens,
            args.validation_probe_tokens,
            args.full_validation,
            args.save_checkpoint,
        )
    aggregate(args.seed, args.train_tokens)


if __name__ == "__main__":
    main()
