"""GPT-2 Medium integration for Keyformer KV-cache policies.

This module reuses Hugging Face GPT-2 weights but drives attention with the
algorithm-level policies in ``models.keyformer``. Absolute token positions are
tracked separately from the physical cache length so compression cannot corrupt
``wpe`` indexing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

try:
    from .keyformer import (
        FullKVPolicy,
        KeyformerPolicy,
        KVCachePolicy,
        KVCacheState,
        RandomKVPolicy,
        RecentWindowPolicy,
        cache_metrics,
    )
except ImportError:  # script-style execution inside keyformer/
    from keyformer import (
        FullKVPolicy,
        KeyformerPolicy,
        KVCachePolicy,
        KVCacheState,
        RandomKVPolicy,
        RecentWindowPolicy,
        cache_metrics,
    )


PolicyName = str


def compute_budget(total_tokens: int, cache_ratio: float, recent_ratio: float = 0.5) -> tuple[int, int]:
    """Map a cache ratio to a fixed token budget and protected recent window."""
    if not 0.0 < cache_ratio <= 1.0:
        raise ValueError("cache_ratio must be in (0, 1]")
    if not 0.0 <= recent_ratio <= 1.0:
        raise ValueError("recent_ratio must be in [0, 1]")
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    budget = max(1, int(math.ceil(cache_ratio * total_tokens)))
    recent_window = min(budget, int(math.floor(recent_ratio * budget)))
    return budget, recent_window


def build_policy(
    name: PolicyName,
    budget: int,
    recent_window: int,
    seed: int = 0,
    tau_init: float = 1.0,
    tau_delta: float = 0.01,
) -> KVCachePolicy:
    key = name.lower().replace("-", "_").replace("+", "_")
    if key in {"full", "full_kv", "fullkv"}:
        return FullKVPolicy()
    if key in {"recent", "recent_window", "recentwindow"}:
        return RecentWindowPolicy(budget=budget)
    if key in {"random", "random_recent", "random_plus_recent", "random+recent"}:
        return RandomKVPolicy(budget=budget, recent_window=recent_window, seed=seed)
    if key in {"keyformer", "kf"}:
        return KeyformerPolicy(
            budget=budget,
            recent_window=recent_window,
            tau_init=tau_init,
            tau_delta=tau_delta,
            gumbel_noise=True,
            seed=seed,
        )
    raise ValueError(f"Unknown policy: {name}")


@dataclass
class EngineConfig:
    model_name: str = "openai-community/gpt2-medium"
    policy: PolicyName = "full_kv"
    cache_ratio: float = 1.0
    recent_ratio: float = 0.5
    seed: int = 0
    tau_init: float = 1.0
    tau_delta: float = 0.01
    dtype: str = "float16"
    device: Optional[str] = None


@dataclass
class StepTiming:
    prefill_ms: float = 0.0
    decode_ms: list[float] | None = None


class GPT2KeyformerEngine:
    """Manual GPT-2 forward with physically compressed per-layer KV caches."""

    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2TokenizerFast,
        policy_factory: Callable[[], KVCachePolicy],
        device: torch.device,
        dtype: torch.dtype,
        total_tokens_hint: int,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.policy_factory = policy_factory
        self.device = device
        self.dtype = dtype
        self.total_tokens_hint = total_tokens_hint
        self.transformer = model.transformer
        self.num_layers = len(self.transformer.h)
        self.num_heads = model.config.n_head
        self.head_dim = model.config.n_embd // model.config.n_head
        self.reset()

    @classmethod
    def from_pretrained_shared(
        cls,
        model: GPT2LMHeadModel,
        tokenizer: GPT2TokenizerFast,
        config: EngineConfig,
        total_tokens: int,
    ) -> "GPT2KeyformerEngine":
        device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = model.dtype
        budget, recent_window = compute_budget(total_tokens, config.cache_ratio, config.recent_ratio)

        def factory() -> KVCachePolicy:
            return build_policy(
                config.policy,
                budget=budget,
                recent_window=recent_window,
                seed=config.seed,
                tau_init=config.tau_init,
                tau_delta=config.tau_delta,
            )

        engine = cls(model, tokenizer, factory, device, dtype, total_tokens)
        engine.budget = budget
        engine.recent_window = recent_window
        engine.policy_name = config.policy
        engine.cache_ratio = config.cache_ratio
        return engine

    @classmethod
    def from_config(cls, config: EngineConfig, total_tokens: int) -> "GPT2KeyformerEngine":
        device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            config.dtype.lower()
        ]
        tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(config.model_name, torch_dtype=dtype)
        model.to(device)
        model.config._attn_implementation = "eager"
        return cls.from_pretrained_shared(model, tokenizer, config, total_tokens)

    def reset(self) -> None:
        self.policies = [self.policy_factory() for _ in range(self.num_layers)]
        self.caches: list[Optional[KVCacheState]] = [None] * self.num_layers
        self.seen_tokens = 0

    def _attention(
        self,
        attn_module,
        hidden_states: torch.Tensor,
        layer_idx: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        policy = self.policies[layer_idx]
        query, key, value = attn_module.c_attn(hidden_states).split(attn_module.split_size, dim=2)
        batch, tokens, _ = query.shape
        shape = (batch, tokens, self.num_heads, self.head_dim)
        query = query.view(*shape).transpose(1, 2)
        key = key.view(*shape).transpose(1, 2)
        value = value.view(*shape).transpose(1, 2)
        scale = float(attn_module.scaling)

        state = self.caches[layer_idx]
        if is_prefill or state is None:
            logits = torch.matmul(query, key.transpose(-1, -2)) * scale
            causal = torch.ones(tokens, tokens, dtype=torch.bool, device=hidden_states.device).tril()
            masked = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)
            probs = torch.softmax(masked.float(), dim=-1).to(value.dtype)
            context = torch.matmul(probs, value)
            score_probabilities = policy.score_probabilities(masked.float(), decode_step=0)
            scores = score_probabilities.sum(dim=-2).to(dtype=torch.float32)
            positions = (
                torch.arange(tokens, device=hidden_states.device)
                .view(1, 1, tokens)
                .expand(batch, self.num_heads, tokens)
            )
            state = KVCacheState(
                keys=key,
                values=value,
                positions=positions,
                scores=scores,
                decode_step=0,
                seen_tokens=tokens,
            )
            self.caches[layer_idx] = policy.select(state)
        else:
            if tokens != 1:
                raise ValueError("decode path expects q_len=1; use prefill for longer chunks")
            keys = torch.cat((state.keys, key), dim=2)
            values = torch.cat((state.values, value), dim=2)
            next_position = torch.full(
                (batch, self.num_heads, 1),
                state.seen_tokens,
                dtype=state.positions.dtype,
                device=state.positions.device,
            )
            positions = torch.cat((state.positions, next_position), dim=-1)
            logits = torch.matmul(query, keys.transpose(-1, -2)) * scale
            probs = torch.softmax(logits.float(), dim=-1).to(values.dtype)
            context = torch.matmul(probs, values)
            step = state.decode_step + 1
            score_row = policy.score_probabilities(logits.float(), decode_step=step).squeeze(-2)
            scores = torch.cat(
                (state.scores, torch.zeros(batch, self.num_heads, 1, device=state.scores.device, dtype=state.scores.dtype)),
                dim=-1,
            )
            scores = scores + score_row.to(scores.dtype)
            updated = KVCacheState(
                keys=keys,
                values=values,
                positions=positions,
                scores=scores,
                decode_step=step,
                seen_tokens=state.seen_tokens + 1,
            )
            self.caches[layer_idx] = policy.select(updated)

        context = context.transpose(1, 2).contiguous().view(batch, tokens, -1)
        return attn_module.resid_dropout(attn_module.c_proj(context))

    def _block(self, block, hidden_states: torch.Tensor, layer_idx: int, is_prefill: bool) -> torch.Tensor:
        residual = hidden_states
        hidden_states = block.ln_1(hidden_states)
        attn_output = self._attention(block.attn, hidden_states, layer_idx, is_prefill=is_prefill)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = block.ln_2(hidden_states)
        return residual + block.mlp(hidden_states)

    def _embed(self, input_ids: torch.Tensor, start_position: int) -> torch.Tensor:
        max_pos = int(self.transformer.wpe.weight.shape[0])
        end_position = start_position + input_ids.shape[1]
        if end_position > max_pos:
            raise ValueError(
                f"Absolute positions [{start_position}, {end_position}) exceed GPT-2 wpe size {max_pos}. "
                "Reduce prompt_len + gen_len so the total stays within n_positions."
            )
        position_ids = torch.arange(
            start_position,
            end_position,
            device=input_ids.device,
        ).unsqueeze(0)
        return self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("engine currently supports batch_size=1")
        self.reset()
        input_ids = input_ids.to(self.device)
        hidden = self._embed(input_ids, start_position=0)
        for idx, block in enumerate(self.transformer.h):
            hidden = self._block(block, hidden, idx, is_prefill=True)
        hidden = self.transformer.ln_f(hidden)
        self.seen_tokens = input_ids.shape[1]
        return self.model.lm_head(hidden)

    def decode_step(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape != (1, 1):
            raise ValueError("decode_step expects shape [1, 1]")
        if self.seen_tokens <= 0 or any(cache is None for cache in self.caches):
            raise RuntimeError("call prefill before decode_step")
        token_ids = token_ids.to(self.device)
        hidden = self._embed(token_ids, start_position=self.seen_tokens)
        for idx, block in enumerate(self.transformer.h):
            hidden = self._block(block, hidden, idx, is_prefill=False)
        hidden = self.transformer.ln_f(hidden)
        self.seen_tokens += 1
        return self.model.lm_head(hidden)

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        logits = self.prefill(input_ids)
        generated = [input_ids.to(self.device)]
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for _ in range(max_new_tokens):
            generated.append(next_token)
            step_logits = self.decode_step(next_token)
            next_token = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)
        return torch.cat(generated, dim=1)

    def cache_summary(self) -> dict[str, float | int]:
        if any(cache is None for cache in self.caches):
            raise RuntimeError("no active caches")
        layer_metrics = [cache_metrics(cache, max(self.seen_tokens, cache.num_tokens)) for cache in self.caches]
        kv_bytes = sum(int(item["kv_bytes"]) for item in layer_metrics)
        cached_tokens = max(int(item["cached_tokens"]) for item in layer_metrics)
        return {
            "seen_tokens": self.seen_tokens,
            "cached_tokens_max": cached_tokens,
            "kv_bytes_total": kv_bytes,
            "layers": self.num_layers,
            "budget": getattr(self, "budget", -1),
            "recent_window": getattr(self, "recent_window", -1),
        }


def stock_greedy_tokens(model: GPT2LMHeadModel, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Reference greedy generation using the unmodified Hugging Face model."""
    model.eval()
    generated = input_ids
    past = None
    for _ in range(max_new_tokens):
        if past is None:
            out = model(generated, use_cache=True)
        else:
            out = model(generated[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    return generated


def max_abs_logit_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return (reference.float() - actual.float()).abs().max().item()
