"""Bounded causal sliding-window attention with preserved sink tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import FactMemoryConfig
from .state import LocalKVCache


class RotaryEmbedding(nn.Module):
    """RoPE supporting different absolute offsets for each batch element."""

    def __init__(self, head_dim: int, max_position: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    @staticmethod
    def _rotate_half(value: Tensor) -> Tensor:
        even, odd = value[..., ::2], value[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def apply(self, query: Tensor, key: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, tokens]")
        if position_ids.numel() and int(position_ids.max()) >= self.max_position:
            raise ValueError("position exceeds configured RoPE maximum")
        frequencies = position_ids.float().unsqueeze(-1) * self.inv_freq
        frequencies = torch.repeat_interleave(frequencies, repeats=2, dim=-1)
        cosine = frequencies.cos().to(query.dtype)[:, None, :, :]
        sine = frequencies.sin().to(query.dtype)[:, None, :, :]
        return (
            query * cosine + self._rotate_half(query) * sine,
            key * cosine + self._rotate_half(key) * sine,
        )


class StreamingSinkSlidingAttention(nn.Module):
    """Causal SWA whose physical cache is bounded by one attention budget.

    The first ``sink_tokens`` conversation positions are preserved as ordinary
    KV pairs.  They are keys/values only: unlike Longformer global tokens, sink
    queries do not gain bidirectional or full-sequence access.  The remaining
    cache stores recent non-sink positions.  Each query therefore observes at
    most ``attention_budget`` unique keys once the cache is full.
    """

    def __init__(self, config: FactMemoryConfig):
        super().__init__()
        self.config = config
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.to_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def _heads(self, value: Tensor) -> Tensor:
        batch, tokens, _ = value.shape
        return value.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def _update_sinks(
        self,
        cache: LocalKVCache,
        key: Tensor,
        value: Tensor,
        position_ids: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sink_k = cache.sink_k.clone()
        sink_v = cache.sink_v.clone()
        sink_valid = cache.sink_valid.clone()
        for sink_position in range(self.config.sink_tokens):
            matches = (position_ids == sink_position) & token_mask
            if not bool(matches.any()):
                continue
            # A valid conversation position occurs at most once.  The weighted
            # selection keeps this operation differentiable with respect to KV.
            weights = matches.to(key.dtype)[:, None, :, None]
            selected_k = (key * weights).sum(dim=2)
            selected_v = (value * weights).sum(dim=2)
            batch_matches = matches.any(dim=1)
            sink_k[:, :, sink_position] = torch.where(
                batch_matches[:, None, None], selected_k, sink_k[:, :, sink_position]
            )
            sink_v[:, :, sink_position] = torch.where(
                batch_matches[:, None, None], selected_v, sink_v[:, :, sink_position]
            )
            sink_valid[:, sink_position] |= batch_matches
        return sink_k, sink_v, sink_valid

    def _recent_windows(
        self,
        cache: LocalKVCache,
        key: Tensor,
        value: Tensor,
        position_ids: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return [B,H,T,R,D] windows and their positions/validity."""

        recent = self.config.recent_window
        all_k = torch.cat((cache.recent_k, key), dim=2)
        all_v = torch.cat((cache.recent_v, value), dim=2)
        all_positions = torch.cat((cache.recent_positions, position_ids), dim=1)
        all_valid = torch.cat((cache.recent_valid, token_mask), dim=1)

        padded_k = F.pad(all_k, (0, 0, recent - 1, 0))
        padded_v = F.pad(all_v, (0, 0, recent - 1, 0))
        padded_positions = F.pad(all_positions, (recent - 1, 0), value=-1)
        padded_valid = F.pad(all_valid, (recent - 1, 0), value=False)

        # unfold appends the window dimension last: [B,H,N,D,R].
        k_windows = padded_k.unfold(2, recent, 1).permute(0, 1, 2, 4, 3)
        v_windows = padded_v.unfold(2, recent, 1).permute(0, 1, 2, 4, 3)
        position_windows = padded_positions.unfold(1, recent, 1)
        valid_windows = padded_valid.unfold(1, recent, 1)

        tokens = key.shape[2]
        return (
            k_windows[:, :, -tokens:],
            v_windows[:, :, -tokens:],
            position_windows[:, -tokens:],
            valid_windows[:, -tokens:],
        )

    def _pack_new_recent(
        self,
        cache: LocalKVCache,
        key: Tensor,
        value: Tensor,
        position_ids: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Physically retain only the newest R non-sink KV pairs."""

        recent = self.config.recent_window
        all_k = torch.cat((cache.recent_k, key), dim=2)
        all_v = torch.cat((cache.recent_v, value), dim=2)
        all_positions = torch.cat((cache.recent_positions, position_ids), dim=1)
        all_valid = torch.cat((cache.recent_valid, token_mask), dim=1)
        all_valid = all_valid & (all_positions >= self.config.sink_tokens)

        new_k = torch.zeros_like(cache.recent_k)
        new_v = torch.zeros_like(cache.recent_v)
        new_positions = torch.full_like(cache.recent_positions, -1)
        new_valid = torch.zeros_like(cache.recent_valid)
        for batch_index in range(key.shape[0]):
            selected = torch.nonzero(all_valid[batch_index], as_tuple=False).flatten()
            selected = selected[-recent:]
            count = int(selected.numel())
            if not count:
                continue
            destination = slice(recent - count, recent)
            new_k[batch_index, :, destination] = all_k[batch_index, :, selected]
            new_v[batch_index, :, destination] = all_v[batch_index, :, selected]
            new_positions[batch_index, destination] = all_positions[batch_index, selected]
            new_valid[batch_index, destination] = True
        return new_k, new_v, new_positions, new_valid

    def forward_chunk(
        self,
        hidden: Tensor,
        cache: LocalKVCache,
        rotary: RotaryEmbedding,
        position_ids: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, LocalKVCache, dict[str, Tensor]]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, tokens, hidden]")
        batch, tokens, width = hidden.shape
        if width != self.config.hidden_size:
            raise ValueError("hidden size does not match configuration")
        if position_ids.shape != (batch, tokens) or token_mask.shape != (batch, tokens):
            raise ValueError("position_ids/token_mask shape mismatch")

        query = self._heads(self.to_q(hidden))
        key = self._heads(self.to_k(hidden))
        value = self._heads(self.to_v(hidden))
        query, key = rotary.apply(query, key, position_ids)

        sink_k, sink_v, sink_valid = self._update_sinks(
            cache, key, value, position_ids, token_mask
        )
        local_k, local_v, local_positions, local_valid = self._recent_windows(
            cache, key, value, position_ids, token_mask
        )
        query_positions = position_ids[:, :, None]
        # Sink positions are represented separately; exclude them from the
        # recent path to guarantee a unique-key budget.
        local_valid = (
            local_valid
            & (local_positions >= self.config.sink_tokens)
            & (local_positions <= query_positions)
        )
        sink_positions = torch.arange(
            self.config.sink_tokens, device=hidden.device
        ).view(1, 1, -1)
        sink_query_valid = sink_valid[:, None, :] & (sink_positions <= query_positions)

        local_scores = torch.einsum("bhtd,bhtrd->bhtr", query, local_k) * self.scale
        sink_scores = torch.einsum("bhtd,bhsd->bhts", query, sink_k) * self.scale
        scores = torch.cat((sink_scores, local_scores), dim=-1)
        score_mask = torch.cat((sink_query_valid, local_valid), dim=-1)[:, None]
        score_mask = score_mask & token_mask[:, None, :, None]
        scores = scores.masked_fill(~score_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        probabilities = self.dropout(probabilities) * token_mask[:, None, :, None]

        sink_probability = probabilities[..., : self.config.sink_tokens]
        local_probability = probabilities[..., self.config.sink_tokens :]
        attended = torch.einsum("bhts,bhsd->bhtd", sink_probability, sink_v)
        attended = attended + torch.einsum("bhtr,bhtrd->bhtd", local_probability, local_v)
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        output = self.to_out(attended) * token_mask.unsqueeze(-1)

        recent_k, recent_v, recent_positions, recent_valid = self._pack_new_recent(
            cache, key, value, position_ids, token_mask
        )
        new_cache = LocalKVCache(
            sink_k=sink_k,
            sink_v=sink_v,
            sink_valid=sink_valid,
            recent_k=recent_k,
            recent_v=recent_v,
            recent_positions=recent_positions,
            recent_valid=recent_valid,
        )
        diagnostics = {
            "sink_probability_mean": sink_probability.sum(dim=-1).mean().detach(),
            "visible_keys_max": score_mask.sum(dim=-1).max().detach(),
            "cache_entries": (sink_valid.sum(dim=-1) + recent_valid.sum(dim=-1)).detach(),
        }
        return output, new_cache, diagnostics
