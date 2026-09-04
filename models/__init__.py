"""Attention implementations with the common benchmark interface."""

from .longformer_attention import LongformerSelfAttention
from .keyformer import (
    CausalSelfAttentionWithKV,
    FullKVPolicy,
    KeyformerPolicy,
    KVCacheState,
    RandomKVPolicy,
    RecentWindowPolicy,
    cache_metrics,
    output_error_metrics,
)
try:  # Optional dependency: Keyformer GPT-2 path is not needed for B-role runs.
    from .gpt2_keyformer import EngineConfig, GPT2KeyformerEngine, build_policy, compute_budget
except ModuleNotFoundError:  # pragma: no cover - depends on environment extras
    EngineConfig = GPT2KeyformerEngine = build_policy = compute_budget = None

__all__ = [
    "CausalSelfAttentionWithKV",
    "EngineConfig",
    "FullKVPolicy",
    "GPT2KeyformerEngine",
    "KeyformerPolicy",
    "KVCacheState",
    "LongformerSelfAttention",
    "RandomKVPolicy",
    "RecentWindowPolicy",
    "build_policy",
    "cache_metrics",
    "compute_budget",
    "output_error_metrics",
]
