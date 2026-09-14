"""Stateful decoder-only Memformer-style recurrent memory.

This module is deliberately separate from the adaptive fact-memory model.
It provides the native latent-state baseline used by the random-colour
experiment: memory is a fixed collection of continuous vectors, there are no
fact labels, lexical payloads, or memory-to-token reconstruction heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .attention import RotaryEmbedding
from .config import FactMemoryConfig


@dataclass
class MemformerState:
    """Per-example recurrent state carried across Memformer segments."""

    memories: tuple[Tensor, ...]
    next_position: Tensor
    segment_count: Tensor

    @classmethod
    def empty(
        cls,
        config: FactMemoryConfig,
        batch_size: int,
        memory_slots: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "MemformerState":
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if memory_slots <= 0:
            raise ValueError("memory_slots must be positive")
        memories = tuple(
            torch.zeros(
                batch_size,
                memory_slots,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.layers)
        )
        return cls(
            memories=memories,
            next_position=torch.zeros(batch_size, device=device, dtype=torch.long),
            segment_count=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    @property
    def batch_size(self) -> int:
        return int(self.next_position.shape[0])

    @property
    def memory_slots(self) -> int:
        if not self.memories:
            return 0
        return int(self.memories[0].shape[1])

    def _validate(self) -> None:
        if not self.memories:
            raise ValueError("MemformerState must contain at least one layer")
        if self.next_position.ndim != 1 or self.segment_count.ndim != 1:
            raise ValueError("position and segment counters must have shape [batch]")
        if self.next_position.dtype != torch.long or self.segment_count.dtype != torch.long:
            raise ValueError("position and segment counters must use torch.long")
        batch = self.batch_size
        slots = self.memory_slots
        if slots <= 0:
            raise ValueError("memory slots must be positive")
        if self.next_position.device != self.segment_count.device:
            raise ValueError("state counters must be on the same device")
        for memory in self.memories:
            if memory.ndim != 3:
                raise ValueError("each memory tensor must have shape [batch, slots, hidden]")
            if memory.shape[0] != batch or memory.shape[1] != slots:
                raise ValueError("all memory tensors must have matching batch/slot dimensions")
            if memory.shape[2] <= 0:
                raise ValueError("memory hidden dimension must be positive")
            if not memory.is_floating_point() and not memory.is_complex():
                raise ValueError("memory tensors must use a floating-point dtype")
            if memory.device != self.next_position.device:
                raise ValueError("all state tensors must be on the same device")
        first = self.memories[0]
        for memory in self.memories[1:]:
            if memory.shape != first.shape:
                raise ValueError("all memory tensors must have the same shape")
            if memory.dtype != first.dtype or memory.device != first.device:
                raise ValueError("all memory tensors must have the same dtype and device")
        if self.segment_count.shape != self.next_position.shape:
            raise ValueError("segment_count and next_position shape mismatch")

    def index_select(self, indices: Tensor) -> "MemformerState":
        if indices.ndim != 1:
            raise ValueError("indices must have shape [batch]")
        indices = indices.to(device=self.next_position.device, dtype=torch.long)
        result = MemformerState(
            memories=tuple(memory.index_select(0, indices) for memory in self.memories),
            next_position=self.next_position.index_select(0, indices),
            segment_count=self.segment_count.index_select(0, indices),
        )
        result._validate()
        return result

    def detach(self) -> "MemformerState":
        result = MemformerState(
            memories=tuple(memory.detach() for memory in self.memories),
            next_position=self.next_position.detach(),
            segment_count=self.segment_count.detach(),
        )
        result._validate()
        return result

    def reset(self, reset_mask: Tensor) -> "MemformerState":
        mask = reset_mask.to(device=self.next_position.device, dtype=torch.bool)
        if mask.shape != self.next_position.shape:
            raise ValueError(
                f"reset_mask must have shape {tuple(self.next_position.shape)}"
            )
        keep = (~mask).to(dtype=self.memories[0].dtype)
        result = MemformerState(
            memories=tuple(memory * keep[:, None, None] for memory in self.memories),
            next_position=torch.where(
                mask, torch.zeros_like(self.next_position), self.next_position
            ),
            segment_count=torch.where(
                mask, torch.zeros_like(self.segment_count), self.segment_count
            ),
        )
        result._validate()
        return result

    def state_bytes(self) -> int:
        total = 0
        for memory in self.memories:
            total += memory.numel() * memory.element_size()
        total += self.next_position.numel() * self.next_position.element_size()
        total += self.segment_count.numel() * self.segment_count.element_size()
        return int(total)


@dataclass
class MemformerOutput:
    logits: Tensor
    state: MemformerState
    diagnostics: dict[str, Tensor]


class MemformerSegmentAttention(nn.Module):
    """One causal segment-level Memformer fusion/update step.

    Memory queries can read their own old slot and valid tokens in the current
    segment. Token queries can read every old slot and a causal prefix of the
    current segment. The newly written memory is not fed back to token queries
    until the next segment, which prevents same-segment future leakage.
    """

    def __init__(
        self,
        config: FactMemoryConfig,
        memory_slots: int = 8,
        *,
        detach_memory: bool = False,
        attention_dropout: float | None = None,
    ):
        super().__init__()
        if memory_slots <= 0:
            raise ValueError("memory_slots must be positive")
        self.config = config
        self.memory_slots = int(memory_slots)
        self.detach_memory = bool(detach_memory)
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mem_q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.mem_k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.mem_v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.mem_out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.update_gate = nn.Linear(config.hidden_size, 1, bias=True)
        probability = (
            config.dropout if attention_dropout is None else float(attention_dropout)
        )
        self.dropout = nn.Dropout(probability)

    def _heads(self, value: Tensor) -> Tensor:
        batch, tokens, width = value.shape
        if width != self.config.hidden_size:
            raise ValueError("hidden size does not match configuration")
        return value.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def _memory_heads(self, value: Tensor) -> Tensor:
        batch, slots, width = value.shape
        if width != self.config.hidden_size or slots != self.memory_slots:
            raise ValueError("memory shape does not match configuration")
        return value.view(batch, slots, self.heads, self.head_dim).transpose(1, 2)

    def _allowed_mask(self, token_mask: Tensor) -> Tensor:
        """Build [B, M+T, M+T] allowed-attention mask."""

        batch, tokens = token_mask.shape
        slots = self.memory_slots
        total = slots + tokens
        allowed = torch.zeros(
            batch, total, total, device=token_mask.device, dtype=torch.bool
        )

        slot_indices = torch.arange(slots, device=token_mask.device)
        allowed[:, slot_indices, slot_indices] = True

        # Memory rows read valid current-segment tokens, but not other slots.
        allowed[:, :slots, slots:] = token_mask[:, None, :].expand(
            batch, slots, tokens
        )

        # Token rows read all old memory slots.
        allowed[:, slots:, :slots] = True

        causal = torch.ones(
            tokens, tokens, device=token_mask.device, dtype=torch.bool
        ).tril()
        token_to_token = (
            causal[None]
            & token_mask[:, :, None]
            & token_mask[:, None, :]
        )
        allowed[:, slots:, slots:] = token_to_token

        # Fully padded query rows would otherwise have an all-false softmax
        # row. Give them one harmless memory key and zero their output later.
        if slots:
            padded = ~token_mask
            allowed[:, slots:, 0] |= padded
        return allowed

    def forward_segment(
        self,
        hidden: Tensor,
        memory: Tensor,
        rotary: RotaryEmbedding,
        position_ids: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, tokens, hidden]")
        if memory.ndim != 3:
            raise ValueError("memory must have shape [batch, slots, hidden]")
        batch, tokens, width = hidden.shape
        if position_ids.shape != (batch, tokens):
            raise ValueError("position_ids shape mismatch")
        if token_mask.shape != (batch, tokens):
            raise ValueError("token_mask shape mismatch")
        if memory.shape != (batch, self.memory_slots, width):
            raise ValueError("memory shape mismatch")

        token_q = self._heads(self.to_q(hidden))
        token_k = self._heads(self.to_k(hidden))
        token_v = self._heads(self.to_v(hidden))
        token_q, token_k = rotary.apply(token_q, token_k, position_ids)

        memory_q = self._memory_heads(self.mem_q_proj(memory))
        memory_k = self._memory_heads(self.mem_k_proj(memory))
        memory_v = self._memory_heads(self.mem_v_proj(memory))

        query = torch.cat((memory_q, token_q), dim=2)
        key = torch.cat((memory_k, token_k), dim=2)
        value = torch.cat((memory_v, token_v), dim=2)
        scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * self.scale
        allowed = self._allowed_mask(token_mask)
        scores = scores.masked_fill(
            ~allowed[:, None],
            torch.finfo(scores.dtype).min,
        )
        probabilities = torch.softmax(scores.float(), dim=-1).to(value.dtype)
        probabilities = self.dropout(probabilities)
        attended = torch.einsum("bhqk,bhkd->bhqd", probabilities, value)
        # Merge the head and head-dimension axes before applying the output
        # projections.  Keeping ``[B, total, H, D]`` here happens to work for
        # a one-head toy model but fails for the protocol's multi-head model
        # (and would silently feed the wrong feature layout to the FFN).
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch, self.memory_slots + tokens, self.config.hidden_size)
        )

        memory_candidate = self.mem_out_proj(attended[:, : self.memory_slots])
        update_probability = torch.sigmoid(self.update_gate(memory_candidate))
        new_memory = (
            update_probability * memory_candidate
            + (1.0 - update_probability) * memory
        )
        if self.detach_memory:
            new_memory = new_memory.detach()

        token_output = self.to_out(attended[:, self.memory_slots :])
        token_output = token_output * token_mask[..., None]
        diagnostics = {
            "update_gate_mean": update_probability.mean().detach(),
            "memory_attention_to_tokens": probabilities[
                :, :, : self.memory_slots, self.memory_slots :
            ].sum(dim=-1).mean().detach(),
            "token_attention_to_memory": probabilities[
                :, :, self.memory_slots :, : self.memory_slots
            ].sum(dim=-1).mean().detach(),
            "valid_token_count": token_mask.sum().detach(),
        }
        return token_output, new_memory, diagnostics


class MemformerDecoderBlock(nn.Module):
    """Pre-LN decoder block with one recurrent memory state."""

    def __init__(
        self,
        config: FactMemoryConfig,
        memory_slots: int,
        *,
        detach_memory: bool = False,
    ):
        super().__init__()
        self.local_norm = nn.LayerNorm(config.hidden_size)
        self.local_attention = MemformerSegmentAttention(
            config,
            memory_slots=memory_slots,
            detach_memory=detach_memory,
            attention_dropout=0.0,
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_in = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.ffn_out = nn.Linear(config.ffn_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        rotary: RotaryEmbedding,
        position_ids: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        attended, new_memory, diagnostics = self.local_attention.forward_segment(
            self.local_norm(hidden),
            memory,
            rotary,
            position_ids,
            token_mask,
        )
        hidden = hidden + self.dropout(attended)
        ffn_input = self.ffn_norm(hidden)
        ffn = self.ffn_out(F.gelu(self.ffn_in(ffn_input)))
        hidden = hidden + self.dropout(ffn) * token_mask[..., None]
        return hidden, new_memory, diagnostics


class MemformerDecoderLM(nn.Module):
    """Tiny decoder LM with persistent segment-level latent memory.

    This is the task adapter for the current experiment, not a claim of a
    complete encoder-decoder MemBART reproduction.
    """

    def __init__(
        self,
        config: FactMemoryConfig = FactMemoryConfig(),
        *,
        memory_slots: int = 8,
        segment_length: int = 32,
        detach_memory: bool = False,
    ):
        super().__init__()
        if segment_length <= 0:
            raise ValueError("segment_length must be positive")
        self.config = config
        self.memory_slots = int(memory_slots)
        self.segment_length = int(segment_length)
        self.detach_memory = bool(detach_memory)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_max_position)
        self.blocks = nn.ModuleList(
            [
                MemformerDecoderBlock(
                    config,
                    self.memory_slots,
                    detach_memory=detach_memory,
                )
                for _ in range(config.layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.constant_(block.local_attention.update_gate.bias, -2.0)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> MemformerState:
        if device is None:
            device = self.token_embedding.weight.device
        if dtype is None:
            dtype = self.token_embedding.weight.dtype
        return MemformerState.empty(
            self.config,
            batch_size,
            self.memory_slots,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _token_mask(
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids")
        mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        invalid_then_valid = (~mask).cumsum(dim=1).gt(0) & mask
        if bool(invalid_then_valid.any()):
            raise ValueError("attention_mask must use right padding")
        return mask

    def _state_dtype(self) -> torch.dtype:
        if self.token_embedding.weight.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return self.token_embedding.weight.dtype

    def forward(
        self,
        input_ids: Tensor,
        state: MemformerState | None = None,
        *,
        attention_mask: Tensor | None = None,
        segment_length: int | None = None,
        detach_state: bool = False,
    ) -> MemformerOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        batch, tokens = input_ids.shape
        if tokens <= 0:
            raise ValueError("input_ids must contain at least one token")
        token_mask = self._token_mask(input_ids, attention_mask)
        if state is None:
            state = self.initial_state(
                batch,
                device=input_ids.device,
                dtype=self._state_dtype(),
            )
        state._validate()
        if state.batch_size != batch:
            raise ValueError("state batch size does not match input_ids")
        if len(state.memories) != self.config.layers:
            raise ValueError("state layer count does not match model configuration")
        if state.memory_slots != self.memory_slots:
            raise ValueError("state memory slot count does not match model")
        if state.memories[0].shape[-1] != self.config.hidden_size:
            raise ValueError("state hidden size does not match model configuration")
        if state.memories[0].device != input_ids.device:
            raise ValueError("state and input_ids must be on the same device")
        if state.memories[0].dtype != self._state_dtype() and not self.training:
            # Under autocast, a caller may pass a state created in the model
            # parameter dtype. Casting is safe for inference and keeps the
            # public state contract explicit.
            state = MemformerState(
                memories=tuple(memory.to(dtype=self._state_dtype()) for memory in state.memories),
                next_position=state.next_position,
                segment_count=state.segment_count,
            )

        chunk_size = self.segment_length if segment_length is None else int(segment_length)
        if chunk_size <= 0:
            raise ValueError("segment_length must be positive")
        hidden_chunks: list[Tensor] = []
        update_gate_values: list[Tensor] = []
        memory_to_token_values: list[Tensor] = []
        token_to_memory_values: list[Tensor] = []
        start = 0
        while start < tokens:
            end = min(tokens, start + chunk_size)
            chunk_hidden = self.token_embedding(input_ids[:, start:end])
            chunk_mask = token_mask[:, start:end]
            offsets = chunk_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
            positions = state.next_position[:, None] + offsets
            new_memories: list[Tensor] = []
            layer_diags: list[dict[str, Tensor]] = []
            for layer_index, block in enumerate(self.blocks):
                chunk_hidden, new_memory, diagnostics = block(
                    chunk_hidden,
                    state.memories[layer_index],
                    self.rotary,
                    positions,
                    chunk_mask,
                )
                new_memories.append(new_memory)
                layer_diags.append(diagnostics)
            hidden_chunks.append(chunk_hidden)
            update_gate_values.extend(
                diagnostics["update_gate_mean"] for diagnostics in layer_diags
            )
            memory_to_token_values.extend(
                diagnostics["memory_attention_to_tokens"] for diagnostics in layer_diags
            )
            token_to_memory_values.extend(
                diagnostics["token_attention_to_memory"] for diagnostics in layer_diags
            )
            has_tokens = chunk_mask.any(dim=1)
            # A completely padded chunk is a no-op for recurrent state.  This
            # is important for packed batches with unequal example lengths:
            # padding must not age, rewrite, or count as a segment.
            state = MemformerState(
                memories=tuple(
                    torch.where(
                        has_tokens[:, None, None],
                        new_memory,
                        old_memory,
                    )
                    for old_memory, new_memory in zip(state.memories, new_memories)
                ),
                next_position=state.next_position + chunk_mask.sum(dim=1),
                segment_count=state.segment_count + has_tokens.to(torch.long),
            )
            start = end

        hidden = torch.cat(hidden_chunks, dim=1)
        final_hidden = self.final_norm(hidden) * token_mask[..., None]
        logits = F.linear(final_hidden, self.token_embedding.weight)
        if detach_state:
            state = state.detach()
        diagnostics = {
            "update_gate_mean": torch.stack(update_gate_values).mean()
            if update_gate_values
            else logits.sum() * 0.0,
            "memory_attention_to_tokens": torch.stack(memory_to_token_values).mean()
            if memory_to_token_values
            else logits.sum() * 0.0,
            "token_attention_to_memory": torch.stack(token_to_memory_values).mean()
            if token_to_memory_values
            else logits.sum() * 0.0,
            "segments_processed": state.segment_count.detach(),
            "state_bytes": torch.tensor(
                state.state_bytes(), device=input_ids.device, dtype=torch.long
            ),
        }
        return MemformerOutput(logits=logits, state=state, diagnostics=diagnostics)

    def forward_segment(
        self,
        input_ids: Tensor,
        state: MemformerState | None = None,
        *,
        attention_mask: Tensor | None = None,
        detach_state: bool = False,
    ) -> MemformerOutput:
        """Process one explicit segment and return its updated state."""

        if input_ids.shape[1] > self.segment_length:
            raise ValueError("forward_segment input exceeds configured segment_length")
        return self.forward(
            input_ids,
            state=state,
            attention_mask=attention_mask,
            segment_length=input_ids.shape[1],
            detach_state=detach_state,
        )
