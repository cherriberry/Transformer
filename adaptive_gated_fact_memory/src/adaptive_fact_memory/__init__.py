"""Adaptive gated fact-memory Transformer prototype."""

from .config import FactMemoryConfig, SourceRole
from .baselines import StreamingSWASourcePolicy
from .model import AdaptiveFactMemoryLM, AdaptiveFactMemoryOutput
from .state import ConversationState, FactMemoryState, LocalKVCache, PendingRoundState

__all__ = [
    "AdaptiveFactMemoryLM",
    "AdaptiveFactMemoryOutput",
    "ConversationState",
    "FactMemoryConfig",
    "FactMemoryState",
    "LocalKVCache",
    "PendingRoundState",
    "SourceRole",
    "StreamingSWASourcePolicy",
]
