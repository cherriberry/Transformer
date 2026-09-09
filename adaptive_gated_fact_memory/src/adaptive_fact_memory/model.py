"""End-to-end TinyLM prototype for adaptive, content-driven fact memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .attention import RotaryEmbedding, StreamingSinkSlidingAttention
from .config import FactMemoryConfig, SourceRole
from .memory import (
    AtomicFactExtractor,
    FactCandidates,
    FactMemoryController,
    FactMemoryReader,
    MemorySelection,
    SharedMemoryFusion,
)
from .state import ConversationState, PendingRoundState


@dataclass
class AdaptiveFactMemoryOutput:
    logits: Tensor
    state: ConversationState
    diagnostics: dict[str, Any]
    auxiliary_losses: dict[str, Tensor]
    supervision: dict[str, Tensor]


class DecoderBlock(nn.Module):
    def __init__(self, config: FactMemoryConfig, use_memory: bool):
        super().__init__()
        self.local_norm = nn.LayerNorm(config.hidden_size)
        self.local_attention = StreamingSinkSlidingAttention(config)
        self.memory_norm = nn.LayerNorm(config.hidden_size) if use_memory else None
        self.memory_fusion = SharedMemoryFusion(config) if use_memory else None
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_in = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.ffn_out = nn.Linear(config.ffn_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: Tensor,
        cache,
        rotary: RotaryEmbedding,
        position_ids: Tensor,
        token_mask: Tensor,
        memory: MemorySelection | None,
        token_embedding: nn.Embedding,
        memory_visibility: Tensor,
        memory_enabled: bool = True,
        fixed_memory_fusion: bool = False,
        fusion_gate_override: float | None = None,
    ):
        local, new_cache, attention_diagnostics = self.local_attention.forward_chunk(
            self.local_norm(hidden), cache, rotary, position_ids, token_mask
        )
        hidden = hidden + self.dropout(local)
        memory_gate = None
        if memory_enabled and self.memory_fusion is not None and self.memory_norm is not None:
            memory_residual, memory_gate = self.memory_fusion(
                self.memory_norm(hidden),
                memory,
                token_embedding,
                memory_visibility,
                fixed_gate=fixed_memory_fusion,
                gate_override=fusion_gate_override,
            )
            hidden = hidden + self.dropout(memory_residual)
        ffn = self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(hidden))))
        hidden = hidden + self.dropout(ffn) * token_mask[..., None]
        return hidden, new_cache, attention_diagnostics, memory_gate


class AdaptiveFactMemoryLM(nn.Module):
    """A bounded-KV decoder plus a shared, inspectable fact memory.

    The public unit of state update is a completed dialogue round.  Memory is
    read before the round's answer region and committed only after all current
    logits have been produced, so a newly written fact cannot influence the
    round that created it.

    ``source_roles`` should mark a contiguous user/system prompt followed by
    assistant tokens.  Memory retrieval pools that known prompt and is exposed
    only at its final token and later positions.  This mirrors inference, where
    the complete user message is known before answer generation, without
    leaking assistant target tokens into the retrieval decision.
    """

    VALID_MEMORY_POLICIES = ("gated", "fixed", "fixed_lru", "none")

    def __init__(
        self,
        config: FactMemoryConfig = FactMemoryConfig(),
        *,
        memory_policy: str = "gated",
        fusion_gate_override: float | None = None,
    ):
        super().__init__()
        if memory_policy not in self.VALID_MEMORY_POLICIES:
            raise ValueError(
                f"memory_policy must be one of {self.VALID_MEMORY_POLICIES}, "
                f"got {memory_policy!r}"
            )
        self.config = config
        self.memory_policy = memory_policy
        self.fusion_gate_override = fusion_gate_override
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_max_position)
        fusion_layers = set(config.memory_fusion_layers)
        self.blocks = nn.ModuleList(
            DecoderBlock(config, layer_index in fusion_layers)
            for layer_index in range(config.layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.fact_extractor = AtomicFactExtractor(config)
        self.memory_controller = FactMemoryController(config)
        self.memory_reader = FactMemoryReader(config)
        self.lm_head = (
            None
            if config.tie_word_embeddings
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the TinyLM backbone with the protocol's small scale.

        ``nn.Embedding`` otherwise defaults to unit-variance weights.  That
        makes the tied output head produce saturated logits at step zero
        (and an unusable initial NLL).  Keep the controller priors explicit:
        write gates start sparse, retention starts permissive, and fusion
        gates start mostly closed until a memory task provides a signal.
        """

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

        # Restore deliberate controller/fusion priors after the generic
        # linear initialization above.
        for module in self.modules():
            if hasattr(module, "write_head"):
                nn.init.constant_(module.write_head.bias, -2.0)
            if hasattr(module, "assistant_write_head"):
                nn.init.constant_(module.assistant_write_head.bias, -2.0)
            if hasattr(module, "retention_head"):
                nn.init.constant_(module.retention_head.bias, 2.0)
            if hasattr(module, "gate") and isinstance(module.gate, nn.Linear):
                nn.init.constant_(module.gate.bias, -2.0)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ConversationState:
        if device is None:
            device = self.token_embedding.weight.device
        if dtype is None:
            dtype = self.token_embedding.weight.dtype
        return ConversationState.empty(
            self.config, batch_size, device=device, dtype=dtype
        )

    @staticmethod
    def _normalize_commit_mask(
        commit: bool | Tensor, batch_size: int, device: torch.device
    ) -> Tensor:
        if isinstance(commit, bool):
            return torch.full((batch_size,), commit, device=device, dtype=torch.bool)
        if commit.shape != (batch_size,):
            raise ValueError(f"commit tensor must have shape {(batch_size,)}")
        return commit.to(device=device, dtype=torch.bool)

    def _validate_roles(self, roles: Tensor, token_mask: Tensor) -> None:
        valid_roles = (
            (roles == int(SourceRole.USER))
            | (roles == int(SourceRole.ASSISTANT))
            | (roles == int(SourceRole.SYSTEM))
            | ~token_mask
        )
        if not bool(valid_roles.all()):
            raise ValueError("source_roles contains an unsupported value")
        # An assistant target followed by another prompt token would make a
        # single retrieval boundary ambiguous.  Split such data into rounds.
        seen_assistant = (roles == int(SourceRole.ASSISTANT)).cumsum(dim=1) > 0
        later_prompt = seen_assistant & token_mask & (roles != int(SourceRole.ASSISTANT))
        if bool(later_prompt.any()):
            raise ValueError("one call must contain prompt tokens followed by assistant tokens")

    def _positions(self, state: ConversationState, token_mask: Tensor) -> Tensor:
        # The cache packer assumes ordinary right padding within each call.
        invalid_then_valid = (~token_mask).cumsum(dim=1).gt(0) & token_mask
        if bool(invalid_then_valid.any()):
            raise ValueError("attention_mask must use right padding")
        offsets = token_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        return state.next_position[:, None] + offsets

    def _retrieval_boundary(
        self, embeddings: Tensor, token_mask: Tensor, source_roles: Tensor
    ) -> tuple[Tensor, Tensor]:
        prompt_mask = token_mask & (
            (source_roles == int(SourceRole.USER))
            | (source_roles == int(SourceRole.SYSTEM))
        )
        has_prompt = prompt_mask.any(dim=1)
        fallback = torch.zeros_like(prompt_mask)
        first_valid = token_mask.to(torch.int64).argmax(dim=1)
        fallback.scatter_(1, first_valid[:, None], has_prompt.logical_not()[:, None])
        query_mask = torch.where(has_prompt[:, None], prompt_mask, fallback)
        denominator = query_mask.sum(dim=1, keepdim=True).clamp_min(1).to(embeddings.dtype)
        query = (embeddings * query_mask[..., None]).sum(dim=1) / denominator

        token_indices = torch.arange(embeddings.shape[1], device=embeddings.device)[None]
        last_prompt = torch.where(
            prompt_mask,
            token_indices,
            torch.full_like(token_indices, -1),
        ).max(dim=1).values
        boundary = torch.where(has_prompt, last_prompt, first_valid)
        visibility = token_mask & (token_indices >= boundary[:, None])
        return query, visibility, has_prompt

    @staticmethod
    def _mix_selection(
        fresh_mask: Tensor,
        fresh: MemorySelection,
        cached: MemorySelection,
    ) -> MemorySelection:
        def choose(left: Tensor, right: Tensor) -> Tensor:
            shape = (fresh_mask.shape[0],) + (1,) * (left.ndim - 1)
            return torch.where(fresh_mask.view(shape), left, right)

        return MemorySelection(
            keys=choose(fresh.keys, cached.keys),
            values=choose(fresh.values, cached.values),
            payload_ids=choose(fresh.payload_ids, cached.payload_ids),
            payload_mask=choose(fresh.payload_mask, cached.payload_mask),
            indices=choose(fresh.indices, cached.indices),
            valid=choose(fresh.valid, cached.valid),
            scores=choose(fresh.scores, cached.scores),
        )

    def _update_pending_round(
        self,
        pending: PendingRoundState,
        candidates: FactCandidates,
        hidden: Tensor,
        token_mask: Tensor,
        source_roles: Tensor,
    ) -> tuple[PendingRoundState, FactCandidates]:
        candidate_count = self.config.max_write_candidates
        keys = torch.cat((pending.keys, candidates.keys), dim=1)
        values = torch.cat((pending.values, candidates.values), dim=1)
        payload_ids = torch.cat((pending.payload_ids, candidates.payload_ids), dim=1)
        payload_mask = torch.cat((pending.payload_mask, candidates.payload_mask), dim=1)
        write_probability = torch.cat(
            (pending.write_probability, candidates.write_probability), dim=1
        )
        confidence = torch.cat((pending.confidence, candidates.confidence), dim=1)
        valid = torch.cat((pending.valid, candidates.valid), dim=1)
        ranking_score = write_probability.masked_fill(~valid, -torch.inf)
        _, selected = torch.topk(ranking_score, k=candidate_count, dim=1)

        def gather(source: Tensor) -> Tensor:
            index = selected
            while index.ndim < source.ndim:
                index = index.unsqueeze(-1)
            index = index.expand(*selected.shape, *source.shape[2:])
            return torch.gather(source, 1, index)

        selected_keys = gather(keys)
        selected_values = gather(values)
        selected_payload_ids = gather(payload_ids)
        selected_payload_mask = gather(payload_mask)
        selected_probability = gather(write_probability)
        selected_confidence = gather(confidence)
        selected_valid = gather(valid)

        assistant_mask = token_mask & (source_roles == int(SourceRole.ASSISTANT))
        assistant_sum = pending.assistant_sum + (
            hidden * assistant_mask[..., None]
        ).sum(dim=1)
        assistant_count = pending.assistant_count + assistant_mask.sum(dim=1)
        updated = PendingRoundState(
            keys=selected_keys,
            values=selected_values,
            payload_ids=selected_payload_ids,
            payload_mask=selected_payload_mask,
            write_probability=selected_probability,
            confidence=selected_confidence,
            valid=selected_valid,
            assistant_sum=assistant_sum,
            assistant_count=assistant_count,
        )
        buffered_candidates = FactCandidates(
            keys=selected_keys,
            values=selected_values,
            payload_ids=selected_payload_ids,
            payload_mask=selected_payload_mask,
            starts=torch.zeros_like(candidates.starts),
            lengths=selected_payload_mask.sum(dim=-1),
            write_probability=selected_probability,
            confidence=selected_confidence,
            valid=selected_valid,
            start_logits=candidates.start_logits,
            length_logits=candidates.length_logits,
            token_write_logits=candidates.token_write_logits,
        )
        return updated, buffered_candidates

    def forward_round(
        self,
        input_ids: Tensor,
        state: ConversationState | None = None,
        *,
        attention_mask: Tensor | None = None,
        source_roles: Tensor | None = None,
        commit: bool | Tensor = False,
        hard_memory: bool | None = None,
        detach_state: bool = False,
    ) -> AdaptiveFactMemoryOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        batch, tokens = input_ids.shape
        if tokens == 0:
            raise ValueError("a round must contain at least one token")
        if attention_mask is None:
            token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask shape must match input_ids")
            token_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if source_roles is None:
            source_roles = torch.full_like(input_ids, int(SourceRole.USER))
        else:
            if source_roles.shape != input_ids.shape:
                raise ValueError("source_roles shape must match input_ids")
            source_roles = source_roles.to(device=input_ids.device, dtype=torch.long)
        source_roles = torch.where(
            token_mask, source_roles, torch.full_like(source_roles, int(SourceRole.PAD))
        )
        self._validate_roles(source_roles, token_mask)

        if state is None:
            state = self.initial_state(batch, device=input_ids.device)
        if state.batch_size != batch:
            raise ValueError("state batch size does not match input_ids")
        if hard_memory is None:
            hard_memory = not self.training
        commit_mask = self._normalize_commit_mask(commit, batch, input_ids.device)
        position_ids = self._positions(state, token_mask)

        hidden = self.token_embedding(input_ids)
        memory_enabled = self.memory_policy != "none"
        if memory_enabled:
            turn_query, memory_visibility, has_prompt = self._retrieval_boundary(
                hidden, token_mask, source_roles
            )
            fresh_retrieval = has_prompt | ~state.round_open
            fresh_selection, accessed_memory = self.memory_reader(
                state.memory, turn_query, update_mask=fresh_retrieval
            )
            cached_selection = self.memory_reader.from_indices(
                state.memory, state.retrieved_indices, state.retrieved_valid
            )
            selection = self._mix_selection(
                fresh_retrieval, fresh_selection, cached_selection
            )
        else:
            memory_visibility = torch.zeros_like(token_mask)
            fresh_retrieval = torch.zeros(batch, device=input_ids.device, dtype=torch.bool)
            accessed_memory = state.memory
            selection = None
        new_caches = []
        layer_diagnostics: list[dict[str, Tensor]] = []
        gate_means: list[Tensor] = []
        for layer_index, (block, cache) in enumerate(zip(self.blocks, state.layer_caches)):
            hidden, new_cache, attention_diagnostics, memory_gate = block(
                hidden,
                cache,
                self.rotary,
                position_ids,
                token_mask,
                selection,
                self.token_embedding,
                memory_visibility,
                memory_enabled=memory_enabled,
                fixed_memory_fusion=self.memory_policy in {"fixed", "fixed_lru"},
                fusion_gate_override=self.fusion_gate_override,
            )
            new_caches.append(new_cache)
            layer_diagnostics.append(attention_diagnostics)
            if memory_gate is not None:
                visible_count = memory_visibility.sum().clamp_min(1)
                gate_means.append(
                    (memory_gate.squeeze(-1) * memory_visibility).sum() / visible_count
                )

        final_hidden = self.final_norm(hidden) * token_mask[..., None]
        logits = (
            F.linear(final_hidden, self.token_embedding.weight)
            if self.lm_head is None
            else self.lm_head(final_hidden)
        )

        if not memory_enabled:
            next_position = state.next_position + token_mask.sum(dim=1)
            new_state = ConversationState(
                memory=state.memory,
                pending=state.pending.reset(commit_mask),
                layer_caches=tuple(new_caches),
                next_position=next_position,
                retrieved_indices=torch.zeros_like(state.retrieved_indices),
                retrieved_valid=torch.zeros_like(state.retrieved_valid),
                round_open=torch.zeros_like(state.round_open),
            )
            if detach_state:
                new_state = new_state.detach()
            zero = logits.sum() * 0.0
            read_shape = (batch, self.config.memory_read_top_k)
            diagnostics: dict[str, Any] = {
                "memory_policy": self.memory_policy,
                "read_indices": torch.zeros(
                    read_shape, device=input_ids.device, dtype=torch.long
                ),
                "read_valid": torch.zeros(
                    read_shape, device=input_ids.device, dtype=torch.bool
                ),
                "read_scores": torch.full(
                    read_shape,
                    -torch.inf,
                    device=input_ids.device,
                    dtype=hidden.dtype,
                ),
                "memory_gate_means": torch.empty(0, device=input_ids.device),
                "active_slots": torch.zeros(
                    batch, device=input_ids.device, dtype=torch.long
                ),
                "layer_attention": layer_diagnostics,
            }
            return AdaptiveFactMemoryOutput(
                logits=logits,
                state=new_state,
                diagnostics=diagnostics,
                auxiliary_losses={"memory_budget": zero, "write_entropy": zero},
                supervision={},
            )

        candidates = self.fact_extractor(
            final_hidden, input_ids, token_mask, source_roles
        )
        if self.memory_policy in {"fixed", "fixed_lru"}:
            # A fixed-memory ablation receives the same proposed spans and
            # slot capacity, but every valid proposal is written.  This also
            # removes the write probability from pending-candidate ranking.
            candidates = replace(
                candidates,
                write_probability=candidates.valid.to(candidates.keys.dtype),
            )
        pending, buffered_candidates = self._update_pending_round(
            state.pending, candidates, final_hidden, token_mask, source_roles
        )
        assistant_denominator = pending.assistant_count[:, None].clamp_min(1).to(
            pending.assistant_sum.dtype
        )
        assistant_summary = pending.assistant_sum / assistant_denominator
        committed_memory, assistant_diagnostics = self.memory_controller.commit_assistant_state(
            accessed_memory,
            assistant_summary,
            pending.assistant_count > 0,
            commit_mask,
            hard=hard_memory,
        )
        committed_memory, write_diagnostics = self.memory_controller.commit_facts(
            committed_memory,
            buffered_candidates,
            commit_mask,
            hard=hard_memory,
            fixed_policy=self.memory_policy in {"fixed", "fixed_lru"},
            fixed_eviction_policy="lru" if self.memory_policy == "fixed_lru" else None,
        )
        pending = pending.reset(commit_mask)
        next_position = state.next_position + token_mask.sum(dim=1)
        round_open = (state.round_open | fresh_retrieval) & ~commit_mask
        retrieved_indices = torch.where(
            fresh_retrieval[:, None], selection.indices, state.retrieved_indices
        )
        retrieved_valid = torch.where(
            fresh_retrieval[:, None], selection.valid, state.retrieved_valid
        )
        retrieved_valid = retrieved_valid & round_open[:, None]
        new_state = ConversationState(
            memory=committed_memory,
            pending=pending,
            layer_caches=tuple(new_caches),
            next_position=next_position,
            retrieved_indices=retrieved_indices,
            retrieved_valid=retrieved_valid,
            round_open=round_open,
        )
        if detach_state:
            new_state = new_state.detach()

        active_fraction = committed_memory.active.mean()
        write_probability = candidates.write_probability.clamp(1e-6, 1 - 1e-6)
        write_entropy = -(
            write_probability * write_probability.log()
            + (1 - write_probability) * (1 - write_probability).log()
        ).mean()
        diagnostics: dict[str, Any] = {
            "memory_policy": self.memory_policy,
            "read_indices": selection.indices.detach(),
            "read_valid": selection.valid.detach(),
            "read_scores": selection.scores.detach(),
            "write_probabilities": candidates.write_probability.detach(),
            "candidate_starts": candidates.starts.detach(),
            "candidate_lengths": candidates.lengths.detach(),
            "candidate_payload_ids": candidates.payload_ids.detach(),
            "candidate_payload_mask": candidates.payload_mask.detach(),
            "pending_fact_count": pending.valid.sum(dim=1).detach(),
            "pending_assistant_tokens": pending.assistant_count.detach(),
            "memory_gate_means": torch.stack(gate_means).detach()
            if gate_means
            else torch.empty(0, device=input_ids.device),
            "layer_attention": layer_diagnostics,
            **write_diagnostics,
            **assistant_diagnostics,
        }
        auxiliary_losses = {
            "memory_budget": active_fraction,
            # Minimize a negative coefficient during warm-up if gate collapse
            # to all-off is observed; it is exposed rather than silently added.
            "write_entropy": write_entropy,
        }
        supervision = {
            "fact_start_logits": candidates.start_logits,
            "fact_length_logits": candidates.length_logits,
            "fact_token_write_logits": candidates.token_write_logits,
            "candidate_keys": candidates.keys,
            "candidate_values": candidates.values,
            "candidate_starts": candidates.starts,
            "candidate_write_probability": candidates.write_probability,
            "candidate_confidence": candidates.confidence,
            "retention_probability": write_diagnostics["retention_scores"],
        }
        return AdaptiveFactMemoryOutput(
            logits=logits,
            state=new_state,
            diagnostics=diagnostics,
            auxiliary_losses=auxiliary_losses,
            supervision=supervision,
        )

    def forward_tinystories(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        state_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Run the TinyStories language-model stage without memory writes.

        TinyStories contains independent short stories rather than annotated
        user/assistant rounds.  This stage trains the shared embedding, local
        sink+recent attention and feed-forward backbone on each packed block
        while starting from a fresh conversation state and disabling the
        episodic path.  The fact controller is intentionally not updated here;
        its training belongs to the later structured-dialogue stage.  A fresh
        bounded cache also prevents memory or KV leakage across packed
        examples.

        Returns logits with shape ``[batch, tokens, vocab_size]``.
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        batch, tokens = input_ids.shape
        if tokens == 0:
            raise ValueError("input_ids must contain at least one token")
        if attention_mask is None:
            token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask shape must match input_ids")
            token_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        invalid_then_valid = (~token_mask).cumsum(dim=1).gt(0) & token_mask
        if bool(invalid_then_valid.any()):
            raise ValueError("attention_mask must use right padding")

        if state_dtype is None:
            # Matching cache and K/V dtypes avoids an unintended float32
            # promotion in the sliding-window concatenation under autocast.
            if input_ids.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
                state_dtype = torch.get_autocast_dtype("cuda")
            else:
                state_dtype = self.token_embedding.weight.dtype
        state = self.initial_state(batch, device=input_ids.device, dtype=state_dtype)
        position_ids = self._positions(state, token_mask)
        hidden = self.token_embedding(input_ids)
        memory_visibility = torch.zeros_like(token_mask)
        for block, cache in zip(self.blocks, state.layer_caches):
            hidden, _, _, _ = block(
                hidden,
                cache,
                self.rotary,
                position_ids,
                token_mask,
                memory=None,
                token_embedding=self.token_embedding,
                memory_visibility=memory_visibility,
                memory_enabled=False,
            )
        final_hidden = self.final_norm(hidden) * token_mask[..., None]
        return (
            F.linear(final_hidden, self.token_embedding.weight)
            if self.lm_head is None
            else self.lm_head(final_hidden)
        )

    def forward(self, input_ids: Tensor, **kwargs) -> AdaptiveFactMemoryOutput:
        return self.forward_round(input_ids, **kwargs)

    @staticmethod
    def state_bytes(state: ConversationState) -> int:
        """Count every tensor byte, including metadata and copied payloads."""

        tensors: list[Tensor] = [
            state.next_position,
            state.retrieved_indices,
            state.retrieved_valid,
            state.round_open,
        ]
        tensors.extend(vars(state.memory).values())
        tensors.extend(vars(state.pending).values())
        for cache in state.layer_caches:
            tensors.extend(vars(cache).values())
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @staticmethod
    def local_state_bytes(state: ConversationState) -> int:
        """Count state required by a sink-SWA-only deployment."""

        tensors: list[Tensor] = [state.next_position]
        for cache in state.layer_caches:
            tensors.extend(vars(cache).values())
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def effective_state_bytes(self, state: ConversationState) -> int:
        """Count tensors required by the selected architecture policy."""

        return (
            self.local_state_bytes(state)
            if self.memory_policy == "none"
            else self.state_bytes(state)
        )
