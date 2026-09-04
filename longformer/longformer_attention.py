"""Memory-efficient sliding-window attention inspired by Longformer.

The module intentionally implements only the attention operation, matching the
``(batch, sequence, dim) -> same shape`` interface used by this repository's
microbenchmark.  It does not load pretrained Longformer weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LongformerSelfAttention(nn.Module):
    """Sliding-window self-attention with optional leading global tokens.

    Local attention is computed from an ``unfold`` view of padded keys and
    values, so the largest score tensor is ``[B, H, N, W]`` rather than
    ``[B, H, N, N]``.  Global keys are appended to the local window for normal
    queries, while global queries attend to the complete sequence.

    Args:
        dim: Input and output embedding size.
        seq_len: Accepted for the repository's common model interface.  The
            implementation itself supports dynamic sequence lengths.
        heads: Number of attention heads.
        window_size: Total local window size.  It must be a positive odd
            number so each token has an equal radius on both sides.
        global_tokens: Number of leading positions with global attention.
        dropout: Dropout probability for attention weights.
        bias: Whether linear projections use bias.
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        heads: int = 8,
        window_size: int = 65,
        global_tokens: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd number")
        if global_tokens < 0:
            raise ValueError("global_tokens must be non-negative")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window_size = window_size
        self.window_radius = window_size // 2
        self.global_tokens = global_tokens
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_k = nn.Linear(dim, dim, bias=bias)
        self.to_v = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim, dim, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        return x.reshape(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

    def _local_windows(self, x: torch.Tensor) -> torch.Tensor:
        """Return a strided ``[B, H, N, W, D]`` view without N-by-N storage."""
        radius = self.window_radius
        padded = F.pad(x, (0, 0, radius, radius))
        # Tensor.unfold appends the window dimension after the untouched dims:
        # [B, H, N, D, W] -> [B, H, N, W, D].
        return padded.unfold(2, self.window_size, 1).permute(0, 1, 2, 4, 3)

    def _local_valid_mask(self, sequence: int, device: torch.device) -> torch.Tensor:
        offsets = torch.arange(-self.window_radius, self.window_radius + 1, device=device)
        positions = torch.arange(sequence, device=device).unsqueeze(-1) + offsets
        return (positions >= 0) & (positions < sequence)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, sequence, dim], got shape {tuple(x.shape)}")
        batch, sequence, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"expected dim={self.dim}, got dim={dim}")
        if sequence == 0:
            return x
        global_count = min(self.global_tokens, sequence)

        if attention_mask is None:
            valid_tokens = torch.ones(batch, sequence, dtype=torch.bool, device=x.device)
        else:
            if attention_mask.shape != (batch, sequence):
                raise ValueError(
                    f"attention_mask must have shape {(batch, sequence)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            valid_tokens = attention_mask.to(device=x.device, dtype=torch.bool)

        q = self._split_heads(self.to_q(x))
        k = self._split_heads(self.to_k(x))
        v = self._split_heads(self.to_v(x))

        k_windows = self._local_windows(k)
        v_windows = self._local_windows(v)
        local_scores = torch.einsum("bhnd,bhnwd->bhnw", q, k_windows) * self.scale

        boundary_mask = self._local_valid_mask(sequence, x.device)
        padded_valid = F.pad(valid_tokens, (self.window_radius, self.window_radius), value=False)
        valid_windows = padded_valid.unfold(1, self.window_size, 1)
        local_valid = valid_windows & boundary_mask.unsqueeze(0)

        if global_count:
            offsets = torch.arange(
                -self.window_radius, self.window_radius + 1, device=x.device
            )
            local_positions = torch.arange(sequence, device=x.device).unsqueeze(-1) + offsets
            # Global keys are appended explicitly below, so exclude them from
            # local windows to prevent double-counting nearby global tokens.
            local_valid = local_valid & ~(
                (local_positions >= 0) & (local_positions < global_count)
            ).unsqueeze(0)
            global_k = k[:, :, :global_count, :]
            global_v = v[:, :, :global_count, :]
            global_scores = torch.einsum("bhnd,bhgd->bhng", q, global_k) * self.scale
            global_valid = valid_tokens[:, :global_count]
            scores = torch.cat((local_scores, global_scores), dim=-1)
            score_valid = torch.cat(
                (
                    local_valid[:, None, :, :],
                    global_valid[:, None, None, :].expand(batch, 1, sequence, global_count),
                ),
                dim=-1,
            )
        else:
            global_v = None
            scores = local_scores
            score_valid = local_valid[:, None, :, :]

        scores = scores.masked_fill(~score_valid, torch.finfo(scores.dtype).min)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        local_attention = attention[..., : self.window_size]
        output = torch.einsum("bhnw,bhnwd->bhnd", local_attention, v_windows)

        if global_count and global_v is not None:
            global_attention = attention[..., self.window_size :]
            output = output + torch.einsum("bhng,bhgd->bhnd", global_attention, global_v)

            # Global queries attend exactly once to every valid token.  Replacing
            # these rows also removes duplicate local/global keys from their path.
            global_q = q[:, :, :global_count, :]
            full_scores = torch.einsum("bhgd,bhnd->bhgn", global_q, k) * self.scale
            full_scores = full_scores.masked_fill(
                ~valid_tokens[:, None, None, :], torch.finfo(full_scores.dtype).min
            )
            full_attention = self.dropout(torch.softmax(full_scores, dim=-1))
            output[:, :, :global_count, :] = torch.einsum(
                "bhgn,bhnd->bhgd", full_attention, v
            )

        # Avoid arbitrary outputs from projection bias at padded query positions.
        output = output * valid_tokens[:, None, :, None]
        output = output.transpose(1, 2).reshape(batch, sequence, dim)
        output = self.to_out(output)
        return output * valid_tokens.unsqueeze(-1)
