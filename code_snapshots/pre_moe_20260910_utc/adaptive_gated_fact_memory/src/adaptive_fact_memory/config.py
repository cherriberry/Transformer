"""Configuration for the adaptive fact-memory prototype."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SourceRole(IntEnum):
    """Token and memory provenance used by the first prototype."""

    PAD = 0
    USER = 1
    ASSISTANT = 2
    SYSTEM = 3


@dataclass(frozen=True)
class FactMemoryConfig:
    """Small decoder-only LM with bounded local KV and shared fact memory.

    ``attention_budget`` counts unique local keys per query after the cache is
    full.  It is split into ``sink_tokens`` preserved conversation-initial
    keys and ``recent_window`` recent non-sink keys.
    """

    vocab_size: int = 50_257
    layers: int = 6
    hidden_size: int = 384
    heads: int = 8
    ffn_size: int = 1_536
    dropout: float = 0.1
    rope_max_position: int = 32_768

    attention_budget: int = 128
    sink_tokens: int = 4

    memory_slots: int = 64
    memory_read_top_k: int = 8
    memory_fusion_layers: tuple[int, ...] = (2, 5)
    max_write_candidates: int = 4
    payload_tokens: int = 12
    write_threshold: float = 0.5
    retention_threshold: float = 0.5
    merge_threshold: float = 0.78
    active_threshold: float = 0.05
    assignment_temperature: float = 0.5

    assistant_slots: int = 4
    pad_token_id: int = 0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if self.attention_budget <= self.sink_tokens:
            raise ValueError("attention_budget must be larger than sink_tokens")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be non-negative")
        if self.memory_slots <= 0:
            raise ValueError("memory_slots must be positive")
        if not 0 < self.memory_read_top_k <= self.memory_slots + self.assistant_slots:
            raise ValueError("memory_read_top_k exceeds available memory slots")
        if self.max_write_candidates <= 0 or self.payload_tokens <= 0:
            raise ValueError("candidate and payload counts must be positive")
        if any(layer < 0 or layer >= self.layers for layer in self.memory_fusion_layers):
            raise ValueError("memory_fusion_layers contains an invalid layer index")
        if len(set(self.memory_fusion_layers)) != len(self.memory_fusion_layers):
            raise ValueError("memory_fusion_layers must be unique")
        if not 0.0 <= self.write_threshold <= 1.0:
            raise ValueError("write_threshold must be in [0, 1]")
        if not 0.0 <= self.retention_threshold <= 1.0:
            raise ValueError("retention_threshold must be in [0, 1]")
        if not -1.0 <= self.merge_threshold <= 1.0:
            raise ValueError("merge_threshold must be in [-1, 1]")
        if self.assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.heads

    @property
    def recent_window(self) -> int:
        return self.attention_budget - self.sink_tokens
