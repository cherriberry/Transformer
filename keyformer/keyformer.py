"""Reference KV-cache policies and causal attention for Keyformer experiments.

This module is an algorithm-level implementation for controlled benchmarks. It
models one causal self-attention layer and keeps token selection independent for
every batch item and attention head. Keyformer scores are accumulated from a
temperature-scaled Gumbel-softmax distribution, following Algorithm 1 of the
paper and the authors' released GPT-J implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class KVCacheState:
    """Per-head KV cache with original token positions and accumulated scores."""

    keys: torch.Tensor  # [batch, heads, tokens, head_dim]
    values: torch.Tensor  # [batch, heads, tokens, head_dim]
    positions: torch.Tensor  # [batch, heads, tokens]
    scores: torch.Tensor  # [batch, heads, tokens]
    decode_step: int = 0
    seen_tokens: int = 0

    @property
    def num_tokens(self) -> int:
        return self.keys.shape[-2]

    @property
    def storage_bytes(self) -> int:
        """Bytes used by K and V tensors, excluding small policy metadata."""
        return self.keys.numel() * self.keys.element_size() + self.values.numel() * self.values.element_size()

    def validate(self) -> None:
        if self.keys.ndim != 4 or self.values.shape != self.keys.shape:
            raise ValueError("keys and values must have matching [B, H, N, D] shapes")
        expected = self.keys.shape[:3]
        if self.positions.shape != expected or self.scores.shape != expected:
            raise ValueError("positions and scores must have shape [B, H, N]")


def _gather_cache(state: KVCacheState, indices: torch.Tensor) -> KVCacheState:
    """Gather independently selected token indices for each batch item/head."""
    state.validate()
    if indices.ndim != 3 or indices.shape[:2] != state.keys.shape[:2]:
        raise ValueError("indices must have shape [B, H, kept_tokens]")
    vector_indices = indices.unsqueeze(-1).expand(*indices.shape, state.keys.shape[-1])
    return KVCacheState(
        keys=torch.gather(state.keys, 2, vector_indices),
        values=torch.gather(state.values, 2, vector_indices),
        positions=torch.gather(state.positions, 2, indices),
        scores=torch.gather(state.scores, 2, indices),
        decode_step=state.decode_step,
        seen_tokens=state.seen_tokens,
    )


class KVCachePolicy:
    """Base cache policy. Subclasses choose which tokens survive each step."""

    name = "base"

    def temperature(self, decode_step: int) -> float:
        return 1.0

    def score_probabilities(self, logits: torch.Tensor, decode_step: int) -> torch.Tensor:
        return torch.softmax(logits / self.temperature(decode_step), dim=-1)

    def select(self, state: KVCacheState) -> KVCacheState:
        return state


class FullKVPolicy(KVCachePolicy):
    """Exact baseline that never discards a cached token."""

    name = "full_kv"


class RecentWindowPolicy(KVCachePolicy):
    """Baseline retaining only the newest ``budget`` original positions."""

    name = "recent_window"

    def __init__(self, budget: int) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.budget = budget

    def select(self, state: KVCacheState) -> KVCacheState:
        if state.num_tokens <= self.budget:
            return state
        indices = torch.argsort(state.positions, dim=-1)[..., -self.budget :]
        indices = torch.gather(indices, -1, torch.argsort(torch.gather(state.positions, -1, indices), dim=-1))
        return _gather_cache(state, indices)


class RandomKVPolicy(KVCachePolicy):
    """Random-old-token baseline with the same protected recent window."""

    name = "random_plus_recent"

    def __init__(self, budget: int, recent_window: int, seed: int = 0) -> None:
        _validate_budget(budget, recent_window)
        self.budget = budget
        self.recent_window = recent_window
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def select(self, state: KVCacheState) -> KVCacheState:
        if state.num_tokens <= self.budget:
            return state
        return _select_old_and_recent(
            state,
            budget=self.budget,
            recent_window=self.recent_window,
            old_ranker=self._random_ranker,
        )

    def _random_ranker(self, old_scores: torch.Tensor) -> torch.Tensor:
        random_cpu = torch.rand(old_scores.shape, generator=self.generator, device="cpu")
        return random_cpu.to(old_scores.device)


class KeyformerPolicy(KVCachePolicy):
    """Accumulated Gumbel-score selection with a protected recent window."""

    name = "keyformer"

    def __init__(
        self,
        budget: int,
        recent_window: int,
        tau_init: float = 1.0,
        tau_delta: float = 0.01,
        gumbel_noise: bool = True,
        seed: int = 0,
    ) -> None:
        _validate_budget(budget, recent_window)
        if tau_init <= 0 or tau_delta < 0:
            raise ValueError("tau_init must be positive and tau_delta non-negative")
        self.budget = budget
        self.recent_window = recent_window
        self.tau_init = tau_init
        self.tau_delta = tau_delta
        self.gumbel_noise = gumbel_noise
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def temperature(self, decode_step: int) -> float:
        return self.tau_init + decode_step * self.tau_delta

    def score_probabilities(self, logits: torch.Tensor, decode_step: int) -> torch.Tensor:
        if self.gumbel_noise:
            uniform = torch.rand(logits.shape, generator=self.generator, device="cpu")
            epsilon = max(float(torch.finfo(logits.dtype).eps), 1e-6)
            uniform = uniform.to(device=logits.device, dtype=logits.dtype).clamp_(epsilon, 1 - epsilon)
            logits = logits - torch.log(-torch.log(uniform))
        return torch.softmax(logits / self.temperature(decode_step), dim=-1)

    def select(self, state: KVCacheState) -> KVCacheState:
        if state.num_tokens <= self.budget:
            return state
        return _select_old_and_recent(
            state,
            budget=self.budget,
            recent_window=self.recent_window,
            old_ranker=lambda old_scores: old_scores,
        )


def _validate_budget(budget: int, recent_window: int) -> None:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if recent_window < 0 or recent_window > budget:
        raise ValueError("recent_window must satisfy 0 <= recent_window <= budget")


def _select_old_and_recent(state: KVCacheState, budget: int, recent_window: int, old_ranker) -> KVCacheState:
    """Select ranked old tokens plus newest tokens, preserving chronological order."""
    chronological = torch.argsort(state.positions, dim=-1)
    ordered_positions = torch.gather(state.positions, -1, chronological)
    ordered_scores = torch.gather(state.scores, -1, chronological)
    old_count = budget - recent_window
    old_limit = state.num_tokens - recent_window

    if old_count:
        old_scores = ordered_scores[..., :old_limit]
        ranks = old_ranker(old_scores)
        old_relative = torch.topk(ranks, k=old_count, dim=-1, largest=True).indices
        old_indices = torch.gather(chronological[..., :old_limit], -1, old_relative)
    else:
        old_indices = chronological[..., :0]

    recent_indices = chronological[..., -recent_window:] if recent_window else chronological[..., :0]
    selected = torch.cat((old_indices, recent_indices), dim=-1)
    selected_positions = torch.gather(state.positions, -1, selected)
    order = torch.argsort(selected_positions, dim=-1)
    return _gather_cache(state, torch.gather(selected, -1, order))


class CausalSelfAttentionWithKV(nn.Module):
    """Single-layer causal attention supporting prefill and token-by-token decode."""

    def __init__(self, dim: int, heads: int = 8, policy: Optional[KVCachePolicy] = None, bias: bool = False) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.policy = policy or FullKVPolicy()
        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_k = nn.Linear(dim, dim, bias=bias)
        self.to_v = nn.Linear(dim, dim, bias=bias)
        self.to_out = nn.Linear(dim, dim, bias=True)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def _merge(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, tokens, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(batch, tokens, self.dim)

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, KVCacheState]:
        """Process a prompt causally, score its tokens, then apply the cache policy."""
        if x.ndim != 3 or x.shape[-1] != self.dim or x.shape[1] == 0:
            raise ValueError(f"x must have non-empty shape [batch, tokens, {self.dim}]")
        q, k, v = (self._split(layer(x)) for layer in (self.to_q, self.to_k, self.to_v))
        tokens = x.shape[1]
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).tril()
        masked_logits = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)
        attention = torch.softmax(masked_logits, dim=-1)
        output = self.to_out(self._merge(torch.matmul(attention, v)))

        score_probabilities = self.policy.score_probabilities(masked_logits, decode_step=0)
        scores = score_probabilities.sum(dim=-2)
        positions = torch.arange(tokens, device=x.device).view(1, 1, tokens).expand(x.shape[0], self.heads, tokens)
        state = KVCacheState(k, v, positions, scores, decode_step=0, seen_tokens=tokens)
        return output, self.policy.select(state)

    def decode_step(self, x: torch.Tensor, state: KVCacheState) -> tuple[torch.Tensor, KVCacheState]:
        """Attend with one new token, update scores, then compress for the next step."""
        state.validate()
        if x.ndim != 3 or x.shape[1:] != (1, self.dim):
            raise ValueError(f"x must have shape [batch, 1, {self.dim}]")
        if x.shape[0] != state.keys.shape[0]:
            raise ValueError("x and cache batch sizes must match")
        q, new_k, new_v = (self._split(layer(x)) for layer in (self.to_q, self.to_k, self.to_v))
        keys = torch.cat((state.keys, new_k), dim=2)
        values = torch.cat((state.values, new_v), dim=2)
        next_position = torch.full(
            (*state.positions.shape[:2], 1),
            state.seen_tokens,
            dtype=state.positions.dtype,
            device=state.positions.device,
        )
        positions = torch.cat((state.positions, next_position), dim=-1)

        logits = torch.matmul(q, keys.transpose(-1, -2)) * self.scale
        attention = torch.softmax(logits, dim=-1)
        output = self.to_out(self._merge(torch.matmul(attention, values)))
        step = state.decode_step + 1
        score_probabilities = self.policy.score_probabilities(logits, decode_step=step).squeeze(-2)
        scores = torch.cat((state.scores, torch.zeros_like(score_probabilities[..., :1])), dim=-1)
        scores = scores + score_probabilities
        updated = KVCacheState(
            keys,
            values,
            positions,
            scores,
            decode_step=step,
            seen_tokens=state.seen_tokens + 1,
        )
        return output, self.policy.select(updated)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Exact causal reference output without returning or compressing a cache."""
        output, _ = CausalSelfAttentionWithKV.prefill(self, x)
        return output


def cache_metrics(state: KVCacheState, full_token_count: int) -> dict[str, float | int]:
    """Report physical cache size and token compression for one layer."""
    if full_token_count <= 0 or state.num_tokens > full_token_count:
        raise ValueError("full_token_count must be positive and at least the stored token count")
    return {
        "cached_tokens": state.num_tokens,
        "full_tokens": full_token_count,
        "compression_ratio": 1.0 - state.num_tokens / full_token_count,
        "kv_bytes": state.storage_bytes,
    }


def output_error_metrics(reference: torch.Tensor, approximation: torch.Tensor) -> dict[str, float]:
    """Numerical distortion metrics for compressed-cache attention outputs."""
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation shapes must match")
    ref = reference.float().reshape(-1)
    approx = approximation.float().reshape(-1)
    difference = approx - ref
    mse = torch.mean(difference.square())
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(ref).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(ref.unsqueeze(0), approx.unsqueeze(0), dim=-1)[0]
    return {"mse": mse.item(), "relative_l2": relative_l2.item(), "cosine_similarity": cosine.item()}
