"""Explicit, inspectable recurrent state for the prototype."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .config import FactMemoryConfig


def _index_dataclass(instance, indices: Tensor):
    values = {field.name: getattr(instance, field.name).index_select(0, indices) for field in fields(instance)}
    return type(instance)(**values)


def _detach_dataclass(instance):
    values = {field.name: getattr(instance, field.name).detach() for field in fields(instance)}
    return type(instance)(**values)


@dataclass
class LocalKVCache:
    """Per-layer cache containing only sinks plus recent non-sink KV."""

    sink_k: Tensor
    sink_v: Tensor
    sink_valid: Tensor
    recent_k: Tensor
    recent_v: Tensor
    recent_positions: Tensor
    recent_valid: Tensor

    @classmethod
    def empty(
        cls,
        config: FactMemoryConfig,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "LocalKVCache":
        head_shape = (batch_size, config.heads, config.head_dim)
        return cls(
            sink_k=torch.zeros(*head_shape[:2], config.sink_tokens, head_shape[-1], device=device, dtype=dtype),
            sink_v=torch.zeros(*head_shape[:2], config.sink_tokens, head_shape[-1], device=device, dtype=dtype),
            sink_valid=torch.zeros(batch_size, config.sink_tokens, device=device, dtype=torch.bool),
            recent_k=torch.zeros(*head_shape[:2], config.recent_window, head_shape[-1], device=device, dtype=dtype),
            recent_v=torch.zeros(*head_shape[:2], config.recent_window, head_shape[-1], device=device, dtype=dtype),
            recent_positions=torch.full((batch_size, config.recent_window), -1, device=device, dtype=torch.long),
            recent_valid=torch.zeros(batch_size, config.recent_window, device=device, dtype=torch.bool),
        )

    def index_select(self, indices: Tensor) -> "LocalKVCache":
        return _index_dataclass(self, indices)

    def detach(self) -> "LocalKVCache":
        return _detach_dataclass(self)

    def reset(self, reset_mask: Tensor) -> "LocalKVCache":
        keep = ~reset_mask.to(device=self.sink_k.device, dtype=torch.bool)
        return LocalKVCache(
            sink_k=self.sink_k * keep[:, None, None, None],
            sink_v=self.sink_v * keep[:, None, None, None],
            sink_valid=self.sink_valid & keep[:, None],
            recent_k=self.recent_k * keep[:, None, None, None],
            recent_v=self.recent_v * keep[:, None, None, None],
            recent_positions=torch.where(keep[:, None], self.recent_positions, -1),
            recent_valid=self.recent_valid & keep[:, None],
        )


@dataclass
class FactMemoryState:
    """Shared user fact slots and the small fixed assistant-state pool."""

    keys: Tensor
    values: Tensor
    active: Tensor
    payload_ids: Tensor
    payload_mask: Tensor
    age: Tensor
    last_access: Tensor
    access_count: Tensor
    confidence: Tensor
    conflict: Tensor
    source_role: Tensor
    assistant_keys: Tensor
    assistant_values: Tensor
    assistant_active: Tensor
    assistant_last_update: Tensor
    turn_index: Tensor

    @classmethod
    def empty(
        cls,
        config: FactMemoryConfig,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "FactMemoryState":
        slot_shape = (batch_size, config.memory_slots)
        assistant_shape = (batch_size, config.assistant_slots)
        return cls(
            keys=torch.zeros(*slot_shape, config.hidden_size, device=device, dtype=dtype),
            values=torch.zeros(*slot_shape, config.hidden_size, device=device, dtype=dtype),
            active=torch.zeros(slot_shape, device=device, dtype=dtype),
            payload_ids=torch.full((*slot_shape, config.payload_tokens), config.pad_token_id, device=device, dtype=torch.long),
            payload_mask=torch.zeros(*slot_shape, config.payload_tokens, device=device, dtype=torch.bool),
            age=torch.zeros(slot_shape, device=device, dtype=torch.long),
            last_access=torch.full(slot_shape, -1, device=device, dtype=torch.long),
            access_count=torch.zeros(slot_shape, device=device, dtype=torch.long),
            confidence=torch.zeros(slot_shape, device=device, dtype=dtype),
            conflict=torch.zeros(slot_shape, device=device, dtype=torch.bool),
            source_role=torch.zeros(slot_shape, device=device, dtype=torch.long),
            assistant_keys=torch.zeros(*assistant_shape, config.hidden_size, device=device, dtype=dtype),
            assistant_values=torch.zeros(*assistant_shape, config.hidden_size, device=device, dtype=dtype),
            assistant_active=torch.zeros(assistant_shape, device=device, dtype=dtype),
            assistant_last_update=torch.full(assistant_shape, -1, device=device, dtype=torch.long),
            turn_index=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def index_select(self, indices: Tensor) -> "FactMemoryState":
        return _index_dataclass(self, indices)

    def detach(self) -> "FactMemoryState":
        return _detach_dataclass(self)

    def reset(self, reset_mask: Tensor) -> "FactMemoryState":
        mask = reset_mask.to(device=self.keys.device, dtype=torch.bool)
        keep_slot = (~mask)[:, None]
        keep_value = keep_slot[..., None]
        keep_payload = keep_slot[..., None]
        keep_assistant = (~mask)[:, None]
        keep_assistant_value = keep_assistant[..., None]
        return FactMemoryState(
            keys=self.keys * keep_value,
            values=self.values * keep_value,
            active=self.active * keep_slot,
            payload_ids=torch.where(keep_payload, self.payload_ids, 0),
            payload_mask=self.payload_mask & keep_payload,
            age=torch.where(keep_slot, self.age, 0),
            last_access=torch.where(keep_slot, self.last_access, -1),
            access_count=torch.where(keep_slot, self.access_count, 0),
            confidence=self.confidence * keep_slot,
            conflict=self.conflict & keep_slot,
            source_role=torch.where(keep_slot, self.source_role, 0),
            assistant_keys=self.assistant_keys * keep_assistant_value,
            assistant_values=self.assistant_values * keep_assistant_value,
            assistant_active=self.assistant_active * keep_assistant,
            assistant_last_update=torch.where(keep_assistant, self.assistant_last_update, -1),
            turn_index=torch.where(~mask, self.turn_index, 0),
        )


@dataclass
class PendingRoundState:
    """Bounded candidates retained until a streamed round is committed."""

    keys: Tensor
    values: Tensor
    payload_ids: Tensor
    payload_mask: Tensor
    write_probability: Tensor
    confidence: Tensor
    valid: Tensor
    assistant_sum: Tensor
    assistant_count: Tensor

    @classmethod
    def empty(
        cls,
        config: FactMemoryConfig,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "PendingRoundState":
        candidates = (batch_size, config.max_write_candidates)
        return cls(
            keys=torch.zeros(*candidates, config.hidden_size, device=device, dtype=dtype),
            values=torch.zeros(*candidates, config.hidden_size, device=device, dtype=dtype),
            payload_ids=torch.full(
                (*candidates, config.payload_tokens),
                config.pad_token_id,
                device=device,
                dtype=torch.long,
            ),
            payload_mask=torch.zeros(
                *candidates, config.payload_tokens, device=device, dtype=torch.bool
            ),
            write_probability=torch.zeros(candidates, device=device, dtype=dtype),
            confidence=torch.zeros(candidates, device=device, dtype=dtype),
            valid=torch.zeros(candidates, device=device, dtype=torch.bool),
            assistant_sum=torch.zeros(
                batch_size, config.hidden_size, device=device, dtype=dtype
            ),
            assistant_count=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def index_select(self, indices: Tensor) -> "PendingRoundState":
        return _index_dataclass(self, indices)

    def detach(self) -> "PendingRoundState":
        return _detach_dataclass(self)

    def reset(self, reset_mask: Tensor) -> "PendingRoundState":
        mask = reset_mask.to(device=self.keys.device, dtype=torch.bool)
        keep_candidate = (~mask)[:, None]
        keep_value = keep_candidate[..., None]
        return PendingRoundState(
            keys=self.keys * keep_value,
            values=self.values * keep_value,
            payload_ids=torch.where(
                keep_value, self.payload_ids, torch.zeros_like(self.payload_ids)
            ),
            payload_mask=self.payload_mask & keep_value,
            write_probability=self.write_probability * keep_candidate,
            confidence=self.confidence * keep_candidate,
            valid=self.valid & keep_candidate,
            assistant_sum=self.assistant_sum * (~mask)[:, None],
            assistant_count=torch.where(mask, 0, self.assistant_count),
        )


@dataclass
class ConversationState:
    """Everything that must follow a conversation during streaming/batching."""

    memory: FactMemoryState
    pending: PendingRoundState
    layer_caches: tuple[LocalKVCache, ...]
    next_position: Tensor
    retrieved_indices: Tensor
    retrieved_valid: Tensor
    round_open: Tensor

    @classmethod
    def empty(
        cls,
        config: FactMemoryConfig,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "ConversationState":
        return cls(
            memory=FactMemoryState.empty(config, batch_size, device=device, dtype=dtype),
            pending=PendingRoundState.empty(config, batch_size, device=device, dtype=dtype),
            layer_caches=tuple(
                LocalKVCache.empty(config, batch_size, device=device, dtype=dtype)
                for _ in range(config.layers)
            ),
            next_position=torch.zeros(batch_size, device=device, dtype=torch.long),
            retrieved_indices=torch.zeros(
                batch_size,
                config.memory_read_top_k,
                device=device,
                dtype=torch.long,
            ),
            retrieved_valid=torch.zeros(
                batch_size,
                config.memory_read_top_k,
                device=device,
                dtype=torch.bool,
            ),
            round_open=torch.zeros(batch_size, device=device, dtype=torch.bool),
        )

    @property
    def batch_size(self) -> int:
        return int(self.next_position.shape[0])

    def index_select(self, indices: Tensor) -> "ConversationState":
        indices = indices.to(device=self.next_position.device, dtype=torch.long)
        return ConversationState(
            memory=self.memory.index_select(indices),
            pending=self.pending.index_select(indices),
            layer_caches=tuple(cache.index_select(indices) for cache in self.layer_caches),
            next_position=self.next_position.index_select(0, indices),
            retrieved_indices=self.retrieved_indices.index_select(0, indices),
            retrieved_valid=self.retrieved_valid.index_select(0, indices),
            round_open=self.round_open.index_select(0, indices),
        )

    def detach(self) -> "ConversationState":
        return ConversationState(
            memory=self.memory.detach(),
            pending=self.pending.detach(),
            layer_caches=tuple(cache.detach() for cache in self.layer_caches),
            next_position=self.next_position.detach(),
            retrieved_indices=self.retrieved_indices.detach(),
            retrieved_valid=self.retrieved_valid.detach(),
            round_open=self.round_open.detach(),
        )

    def reset(self, reset_mask: Tensor) -> "ConversationState":
        mask = reset_mask.to(device=self.next_position.device, dtype=torch.bool)
        if mask.shape != self.next_position.shape:
            raise ValueError(f"reset_mask must have shape {tuple(self.next_position.shape)}")
        return ConversationState(
            memory=self.memory.reset(mask),
            pending=self.pending.reset(mask),
            layer_caches=tuple(cache.reset(mask) for cache in self.layer_caches),
            next_position=torch.where(mask, 0, self.next_position),
            retrieved_indices=torch.where(
                mask[:, None], 0, self.retrieved_indices
            ),
            retrieved_valid=self.retrieved_valid & ~mask[:, None],
            round_open=self.round_open & ~mask,
        )
