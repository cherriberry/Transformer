"""Content-driven fact extraction, retrieval, fusion, and slot updates."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import FactMemoryConfig, SourceRole
from .state import FactMemoryState


@dataclass
class FactCandidates:
    keys: Tensor
    values: Tensor
    payload_ids: Tensor
    payload_mask: Tensor
    starts: Tensor
    lengths: Tensor
    write_probability: Tensor
    confidence: Tensor
    valid: Tensor
    start_logits: Tensor
    length_logits: Tensor
    token_write_logits: Tensor


@dataclass
class MemorySelection:
    keys: Tensor
    values: Tensor
    payload_ids: Tensor
    payload_mask: Tensor
    indices: Tensor
    valid: Tensor
    scores: Tensor


class AtomicFactExtractor(nn.Module):
    """Select short source spans and encode them as fact Key--Value pairs.

    The discrete top-k/span choice is intentional: structured labels can
    supervise ``start_logits`` and ``length_logits`` during training, while the
    copied payload remains auditable at inference.  This module does not use an
    external fact extractor.
    """

    def __init__(self, config: FactMemoryConfig):
        super().__init__()
        self.config = config
        self.start_head = nn.Linear(config.hidden_size, 1)
        self.length_head = nn.Linear(config.hidden_size, config.payload_tokens)
        self.key_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.write_head = nn.Linear(config.hidden_size, 1)
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        # Sparse by default; auxiliary retrieval supervision must earn writes.
        nn.init.constant_(self.write_head.bias, -2.0)
        nn.init.constant_(self.confidence_head.bias, 0.0)

    def forward(
        self,
        hidden: Tensor,
        input_ids: Tensor,
        token_mask: Tensor,
        source_roles: Tensor,
    ) -> FactCandidates:
        batch, tokens, width = hidden.shape
        candidate_count = self.config.max_write_candidates
        payload_length = self.config.payload_tokens
        start_logits = self.start_head(hidden).squeeze(-1)
        length_logits = self.length_head(hidden)
        token_write_logits = self.write_head(hidden).squeeze(-1)
        allowed = token_mask & (source_roles == int(SourceRole.USER))
        masked_scores = start_logits.masked_fill(~allowed, -torch.inf)

        selected_count = min(tokens, candidate_count)
        top_scores, starts = torch.topk(masked_scores, k=selected_count, dim=1)
        selected_valid = torch.isfinite(top_scores)
        if selected_count < candidate_count:
            padding = candidate_count - selected_count
            starts = F.pad(starts, (0, padding), value=0)
            top_scores = F.pad(top_scores, (0, padding), value=-torch.inf)
            selected_valid = F.pad(selected_valid, (0, padding), value=False)

        gather_length = starts[..., None].expand(-1, -1, payload_length)
        selected_length_logits = torch.gather(length_logits, 1, gather_length)
        lengths = selected_length_logits.argmax(dim=-1) + 1

        offsets = torch.arange(payload_length, device=hidden.device).view(1, 1, -1)
        raw_indices = starts[..., None] + offsets
        safe_indices = raw_indices.clamp(max=max(0, tokens - 1))
        batch_indices = torch.arange(batch, device=hidden.device).view(batch, 1, 1)
        span_hidden = hidden[batch_indices, safe_indices]
        payload_ids = input_ids[batch_indices, safe_indices]
        span_token_valid = token_mask[batch_indices, safe_indices]
        span_roles = source_roles[batch_indices, safe_indices]
        payload_mask = (
            selected_valid[..., None]
            & (raw_indices < tokens)
            & (offsets < lengths[..., None])
            & span_token_valid
            & (span_roles == int(SourceRole.USER))
        )
        payload_ids = torch.where(payload_mask, payload_ids, self.config.pad_token_id)
        denominator = payload_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(hidden.dtype)
        span_representation = (
            span_hidden * payload_mask[..., None].to(hidden.dtype)
        ).sum(dim=2) / denominator
        span_representation = span_representation * selected_valid[..., None]

        keys = F.normalize(self.key_projection(span_representation), dim=-1, eps=1e-6)
        values = self.value_projection(span_representation)
        write_probability = torch.sigmoid(self.write_head(span_representation).squeeze(-1))
        write_probability = write_probability * selected_valid
        confidence = torch.sigmoid(self.confidence_head(span_representation).squeeze(-1))
        confidence = confidence * selected_valid
        return FactCandidates(
            keys=keys,
            values=values,
            payload_ids=payload_ids,
            payload_mask=payload_mask,
            starts=starts,
            lengths=lengths,
            write_probability=write_probability,
            confidence=confidence,
            valid=selected_valid,
            start_logits=start_logits,
            length_logits=length_logits,
            token_write_logits=token_write_logits,
        )


class FactMemoryController(nn.Module):
    """A straight-through atomic slot controller with explicit eviction."""

    def __init__(self, config: FactMemoryConfig):
        super().__init__()
        self.config = config
        metadata_width = 5
        self.retention_head = nn.Linear(2 * config.hidden_size + metadata_width, 1)
        self.update_head = nn.Linear(2 * config.hidden_size + 1, 1)
        self.assistant_key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.assistant_value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.assistant_slot_head = nn.Linear(config.hidden_size, config.assistant_slots)
        self.assistant_write_head = nn.Linear(config.hidden_size, 1)
        nn.init.constant_(self.assistant_write_head.bias, -2.0)
        nn.init.constant_(self.retention_head.bias, 2.0)

    def retention_scores(self, state: FactMemoryState) -> Tensor:
        metadata = torch.stack(
            (
                state.active,
                state.confidence,
                state.age.to(state.keys.dtype) / 32.0,
                torch.log1p(state.access_count.to(state.keys.dtype)) / 4.0,
                state.conflict.to(state.keys.dtype),
            ),
            dim=-1,
        )
        features = torch.cat((state.keys, state.values, metadata), dim=-1)
        return torch.sigmoid(self.retention_head(features).squeeze(-1))

    def _choose_targets(
        self,
        state: FactMemoryState,
        candidate_keys: Tensor,
        *,
        fixed_eviction_policy: str | None = None,
    ) -> Tensor:
        active = state.active > self.config.active_threshold
        similarity = torch.einsum(
            "bmd,bd->bm", F.normalize(state.keys, dim=-1, eps=1e-6), candidate_keys
        )
        retention = self.retention_scores(state)
        targets = torch.zeros(candidate_keys.shape[0], device=candidate_keys.device, dtype=torch.long)
        for batch_index in range(candidate_keys.shape[0]):
            active_indices = torch.nonzero(active[batch_index], as_tuple=False).flatten()
            empty_indices = torch.nonzero(~active[batch_index], as_tuple=False).flatten()
            if active_indices.numel():
                active_similarity = similarity[batch_index, active_indices]
                best_offset = int(active_similarity.argmax())
                best_slot = int(active_indices[best_offset])
                if (
                    float(active_similarity[best_offset].detach())
                    >= self.config.merge_threshold
                ):
                    targets[batch_index] = best_slot
                    continue
            if empty_indices.numel():
                targets[batch_index] = empty_indices[0]
            elif fixed_eviction_policy == "lru":
                # The fixed-memory stress baseline must not depend on the
                # randomly initialized, frozen retention head once capacity is
                # exhausted.  Evict the least recently accessed active slot;
                # ``argmin`` gives a deterministic lowest-index tie break.
                targets[batch_index] = active_indices[
                    state.last_access[batch_index, active_indices].argmin()
                ]
            else:
                targets[batch_index] = retention[batch_index].argmin()
        return targets

    def _write_strength(self, probability: Tensor, hard: bool) -> Tensor:
        discrete = (probability >= self.config.write_threshold).to(probability.dtype)
        if hard:
            return discrete
        # Hard behavior in the forward pass, sigmoid gradient in backward.
        return discrete + probability - probability.detach()

    def _apply_retention(
        self,
        state: FactMemoryState,
        commit_mask: Tensor,
        *,
        hard: bool,
        fixed_policy: bool = False,
    ) -> tuple[FactMemoryState, Tensor, Tensor]:
        learned_probability = self.retention_scores(state)
        probability = (
            torch.ones_like(learned_probability)
            if fixed_policy
            else learned_probability
        )
        discrete = (probability >= self.config.retention_threshold).to(probability.dtype)
        strength = (
            discrete
            if hard or fixed_policy
            else discrete + probability - probability.detach()
        )
        strength = torch.where(commit_mask[:, None], strength, torch.ones_like(strength))
        was_active = state.active > self.config.active_threshold
        active = state.active * strength
        is_active = active > self.config.active_threshold
        evicted = was_active & ~is_active & commit_mask[:, None]
        age = torch.where(
            is_active & commit_mask[:, None], state.age + 1, state.age
        )
        return replace(state, active=active, age=age), probability, evicted

    def commit_facts(
        self,
        state: FactMemoryState,
        candidates: FactCandidates,
        commit_mask: Tensor,
        *,
        hard: bool,
        fixed_policy: bool = False,
        fixed_eviction_policy: str | None = None,
    ) -> tuple[FactMemoryState, dict[str, Tensor]]:
        batch, candidate_count, _ = candidates.keys.shape
        commit_mask = commit_mask.to(device=state.keys.device, dtype=torch.bool)
        state, retention_before_write, retention_evicted = self._apply_retention(
            state, commit_mask, hard=hard, fixed_policy=fixed_policy
        )
        target_history: list[Tensor] = []
        accepted_history: list[Tensor] = []
        evicted_history: list[Tensor] = []

        for candidate_index in range(candidate_count):
            candidate_key = candidates.keys[:, candidate_index]
            candidate_value = candidates.values[:, candidate_index]
            probability = candidates.write_probability[:, candidate_index]
            candidate_valid = candidates.valid[:, candidate_index] & commit_mask
            write_strength = (
                candidate_valid.to(probability.dtype)
                if fixed_policy
                else self._write_strength(probability, hard) * candidate_valid
            )
            targets = self._choose_targets(
                state,
                candidate_key,
                fixed_eviction_policy=fixed_eviction_policy,
            )

            normalized_keys = F.normalize(state.keys, dim=-1, eps=1e-6)
            similarity = torch.einsum("bmd,bd->bm", normalized_keys, candidate_key)
            empty_bonus = 2.0 * (1.0 - state.active.clamp(0, 1))
            soft_assignment = torch.softmax(
                (similarity + empty_bonus) / self.config.assignment_temperature, dim=-1
            )
            hard_assignment = F.one_hot(targets, self.config.memory_slots).to(state.keys.dtype)
            if hard or fixed_policy:
                assignment = hard_assignment
            else:
                assignment = hard_assignment + soft_assignment - soft_assignment.detach()

            batch_indices = torch.arange(batch, device=state.keys.device)
            old_target_values = state.values[batch_indices, targets]
            old_target_active = state.active[batch_indices, targets] > self.config.active_threshold
            target_similarity = similarity[batch_indices, targets]
            update_probability = (
                torch.ones(batch, device=state.keys.device, dtype=state.keys.dtype)
                if fixed_policy
                else torch.sigmoid(
                    self.update_head(
                        torch.cat(
                            (
                                candidate_value,
                                old_target_values,
                                target_similarity[:, None],
                            ),
                            dim=-1,
                        )
                    ).squeeze(-1)
                )
            )
            update_probability = torch.where(
                old_target_active, update_probability, torch.ones_like(update_probability)
            )
            alpha = write_strength[:, None] * assignment * update_probability[:, None]
            keys = F.normalize(
                state.keys * (1.0 - alpha[..., None])
                + candidate_key[:, None, :] * alpha[..., None],
                dim=-1,
                eps=1e-6,
            )
            values = (
                state.values * (1.0 - alpha[..., None])
                + candidate_value[:, None, :] * alpha[..., None]
            )
            activation_alpha = write_strength[:, None] * assignment
            active = state.active + activation_alpha * (1.0 - state.active)
            confidence = (
                state.confidence * (1.0 - activation_alpha)
                + candidates.confidence[:, candidate_index, None] * activation_alpha
            )

            payload_ids = state.payload_ids.clone()
            payload_mask = state.payload_mask.clone()
            source_role = state.source_role.clone()
            last_access = state.last_access.clone()
            conflict = state.conflict.clone()
            new_age = state.age.clone()
            accepted = (
                candidate_valid
                if fixed_policy
                else candidate_valid & (probability >= self.config.write_threshold)
            )
            evicted = torch.zeros(batch, device=state.keys.device, dtype=torch.bool)
            for batch_index in range(batch):
                if not bool(accepted[batch_index]):
                    continue
                slot = int(targets[batch_index])
                was_active = bool(old_target_active[batch_index])
                was_similar = (
                    float(target_similarity[batch_index].detach())
                    >= self.config.merge_threshold
                )
                evicted[batch_index] = was_active and not was_similar
                if was_active:
                    old_ids = payload_ids[batch_index, slot]
                    old_mask = payload_mask[batch_index, slot]
                    new_ids = candidates.payload_ids[batch_index, candidate_index]
                    new_mask = candidates.payload_mask[batch_index, candidate_index]
                    overlap = old_mask & new_mask
                    different = bool(overlap.any() and (old_ids[overlap] != new_ids[overlap]).any())
                    conflict[batch_index, slot] |= different
                payload_ids[batch_index, slot] = candidates.payload_ids[batch_index, candidate_index]
                payload_mask[batch_index, slot] = candidates.payload_mask[batch_index, candidate_index]
                source_role[batch_index, slot] = int(SourceRole.USER)
                last_access[batch_index, slot] = state.turn_index[batch_index]
                new_age[batch_index, slot] = 0

            state = replace(
                state,
                keys=keys,
                values=values,
                active=active,
                payload_ids=payload_ids,
                payload_mask=payload_mask,
                age=new_age,
                last_access=last_access,
                confidence=confidence,
                conflict=conflict,
                source_role=source_role,
            )
            target_history.append(targets)
            accepted_history.append(accepted)
            evicted_history.append(evicted)

        state = replace(
            state,
            turn_index=state.turn_index + commit_mask.to(state.turn_index.dtype),
        )
        diagnostics = {
            "write_targets": torch.stack(target_history, dim=1).detach(),
            "write_accepted": torch.stack(accepted_history, dim=1).detach(),
            "slot_evicted": torch.stack(evicted_history, dim=1).detach(),
            "retention_evicted": retention_evicted.detach(),
            "active_slots": (state.active > self.config.active_threshold).sum(dim=1).detach(),
            "soft_active_slots": state.active.sum(dim=1),
            "retention_scores_before_write": retention_before_write,
            "retention_scores": (
                torch.ones_like(state.active)
                if fixed_policy
                else self.retention_scores(state)
            ),
            "fixed_policy": torch.full(
                (batch,), fixed_policy, device=state.keys.device, dtype=torch.bool
            ),
        }
        return state, diagnostics

    def commit_assistant_state(
        self,
        state: FactMemoryState,
        pooled: Tensor,
        has_content: Tensor,
        commit_mask: Tensor,
        *,
        hard: bool,
    ) -> tuple[FactMemoryState, dict[str, Tensor]]:
        has_content = has_content & commit_mask
        probability = torch.sigmoid(self.assistant_write_head(pooled).squeeze(-1))
        strength = self._write_strength(probability, hard) * has_content
        slot_probabilities = torch.softmax(self.assistant_slot_head(pooled), dim=-1)
        targets = slot_probabilities.argmax(dim=-1)
        hard_assignment = F.one_hot(targets, self.config.assistant_slots).to(pooled.dtype)
        assignment = hard_assignment if hard else hard_assignment + slot_probabilities - slot_probabilities.detach()
        alpha = strength[:, None] * assignment
        candidate_keys = F.normalize(self.assistant_key(pooled), dim=-1, eps=1e-6)
        candidate_values = self.assistant_value(pooled)
        assistant_keys = F.normalize(
            state.assistant_keys * (1.0 - alpha[..., None])
            + candidate_keys[:, None] * alpha[..., None],
            dim=-1,
            eps=1e-6,
        )
        assistant_values = (
            state.assistant_values * (1.0 - alpha[..., None])
            + candidate_values[:, None] * alpha[..., None]
        )
        assistant_active = state.assistant_active + alpha * (1.0 - state.assistant_active)
        assistant_last_update = state.assistant_last_update.clone()
        accepted = has_content & (probability >= self.config.write_threshold)
        for batch_index in range(pooled.shape[0]):
            if bool(accepted[batch_index]):
                assistant_last_update[batch_index, targets[batch_index]] = state.turn_index[batch_index]
        return replace(
            state,
            assistant_keys=assistant_keys,
            assistant_values=assistant_values,
            assistant_active=assistant_active,
            assistant_last_update=assistant_last_update,
        ), {
            "assistant_write_probability": probability.detach(),
            "assistant_write_accepted": accepted.detach(),
            "assistant_write_target": targets.detach(),
        }


class FactMemoryReader(nn.Module):
    """Retrieve a fixed small top-k set once per processed round."""

    def __init__(self, config: FactMemoryConfig):
        super().__init__()
        self.config = config
        self.query_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _combined(self, state: FactMemoryState) -> tuple[Tensor, ...]:
        assistant_payload_ids = torch.full(
            (
                state.keys.shape[0],
                self.config.assistant_slots,
                self.config.payload_tokens,
            ),
            self.config.pad_token_id,
            device=state.keys.device,
            dtype=torch.long,
        )
        assistant_payload_mask = torch.zeros_like(assistant_payload_ids, dtype=torch.bool)
        keys = torch.cat((state.keys, state.assistant_keys), dim=1)
        values = torch.cat((state.values, state.assistant_values), dim=1)
        active = torch.cat((state.active, state.assistant_active), dim=1)
        payload_ids = torch.cat((state.payload_ids, assistant_payload_ids), dim=1)
        payload_mask = torch.cat((state.payload_mask, assistant_payload_mask), dim=1)
        return keys, values, active, payload_ids, payload_mask

    def from_indices(
        self, state: FactMemoryState, indices: Tensor, valid: Tensor
    ) -> MemorySelection:
        keys, values, _, payload_ids, payload_mask = self._combined(state)
        safe_indices = indices[..., None].expand(-1, -1, self.config.hidden_size)
        selected_keys = torch.gather(keys, 1, safe_indices) * valid[..., None]
        selected_values = torch.gather(values, 1, safe_indices) * valid[..., None]
        payload_indices = indices[..., None].expand(-1, -1, self.config.payload_tokens)
        selected_payload_ids = torch.gather(payload_ids, 1, payload_indices)
        selected_payload_mask = torch.gather(payload_mask, 1, payload_indices) & valid[..., None]
        scores = torch.zeros(indices.shape, device=keys.device, dtype=keys.dtype)
        scores = scores.masked_fill(~valid, -torch.inf)
        return MemorySelection(
            keys=selected_keys,
            values=selected_values,
            payload_ids=selected_payload_ids,
            payload_mask=selected_payload_mask,
            indices=indices,
            valid=valid,
            scores=scores,
        )

    def forward(
        self,
        state: FactMemoryState,
        turn_query: Tensor,
        *,
        update_mask: Tensor | None = None,
    ) -> tuple[MemorySelection, FactMemoryState]:
        keys, values, active, payload_ids, payload_mask = self._combined(state)

        query = F.normalize(self.query_projection(turn_query), dim=-1, eps=1e-6)
        scores = torch.einsum("bd,bmd->bm", query, F.normalize(keys, dim=-1, eps=1e-6))
        active_mask = active > self.config.active_threshold
        scores = scores.masked_fill(~active_mask, -torch.inf)
        top_scores, indices = torch.topk(scores, k=self.config.memory_read_top_k, dim=-1)
        valid = torch.isfinite(top_scores)
        safe_indices = indices[..., None].expand(-1, -1, self.config.hidden_size)
        selected_keys = torch.gather(keys, 1, safe_indices)
        selected_values = torch.gather(values, 1, safe_indices)
        payload_indices = indices[..., None].expand(-1, -1, self.config.payload_tokens)
        selected_payload_ids = torch.gather(payload_ids, 1, payload_indices)
        selected_payload_mask = torch.gather(payload_mask, 1, payload_indices)
        selected_keys = selected_keys * valid[..., None]
        selected_values = selected_values * valid[..., None]
        selected_payload_mask = selected_payload_mask & valid[..., None]

        access_count = state.access_count.clone()
        last_access = state.last_access.clone()
        for batch_index in range(indices.shape[0]):
            for selected_index in range(indices.shape[1]):
                if not bool(valid[batch_index, selected_index]):
                    continue
                if update_mask is not None and not bool(update_mask[batch_index]):
                    continue
                slot = int(indices[batch_index, selected_index])
                if slot >= self.config.memory_slots:
                    continue
                access_count[batch_index, slot] += 1
                last_access[batch_index, slot] = state.turn_index[batch_index]
        updated_state = replace(state, access_count=access_count, last_access=last_access)
        return MemorySelection(
            keys=selected_keys,
            values=selected_values,
            payload_ids=selected_payload_ids,
            payload_mask=selected_payload_mask,
            indices=indices,
            valid=valid,
            scores=top_scores,
        ), updated_state


class SharedMemoryFusion(nn.Module):
    """Cross-attend to the retrieved shared memory and gate its residual."""

    def __init__(self, config: FactMemoryConfig):
        super().__init__()
        self.config = config
        self.to_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gate = nn.Linear(2 * config.hidden_size, 1)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(
        self,
        hidden: Tensor,
        selection: MemorySelection,
        token_embedding: nn.Embedding,
        token_mask: Tensor,
        *,
        fixed_gate: bool = False,
        gate_override: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        payload_embeddings = token_embedding(selection.payload_ids)
        payload_denominator = selection.payload_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        payload_summary = (
            payload_embeddings * selection.payload_mask[..., None]
        ).sum(dim=2) / payload_denominator.to(payload_embeddings.dtype)
        memory_values = selection.values + payload_summary
        query = self.to_q(hidden)
        key = self.to_k(selection.keys)
        value = self.to_v(memory_values)
        scores = torch.einsum("btd,bkd->btk", query, key) * (self.config.hidden_size**-0.5)
        scores = scores.masked_fill(~selection.valid[:, None, :], torch.finfo(scores.dtype).min)
        probability = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
        probability = probability * selection.valid[:, None, :]
        denominator = probability.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        probability = probability / denominator
        context = torch.einsum("btk,bkd->btd", probability, value)
        has_memory = selection.valid.any(dim=-1)[:, None, None]
        context = context * has_memory
        learned_gate = torch.sigmoid(self.gate(torch.cat((hidden, context), dim=-1)))
        if gate_override is not None:
            if not 0.0 <= gate_override <= 1.0:
                raise ValueError("gate_override must be in [0, 1]")
            gate = torch.full_like(learned_gate, float(gate_override))
        else:
            gate = torch.ones_like(learned_gate) if fixed_gate else learned_gate
        fused = self.to_out(context) * gate * token_mask[..., None]
        return fused, gate * has_memory
