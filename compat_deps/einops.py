"""Minimal einops compatibility for the vendored Performer/Reformer sources."""

from __future__ import annotations

import torch


def rearrange(tensor: torch.Tensor, pattern: str, **axes) -> torch.Tensor:
    pattern = " ".join(pattern.split())
    if pattern == "b n (h d) -> b h n d":
        heads = axes["h"]
        batch, length, width = tensor.shape
        return tensor.reshape(batch, length, heads, width // heads).permute(0, 2, 1, 3)
    if pattern == "b h n d -> b n (h d)":
        batch, heads, length, width = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(batch, length, heads * width)
    if pattern == "... (d j) -> ... d j":
        factor = axes["j"]
        return tensor.reshape(*tensor.shape[:-1], tensor.shape[-1] // factor, factor)
    if pattern == "... d j -> ... (d j)":
        return tensor.reshape(*tensor.shape[:-2], tensor.shape[-2] * tensor.shape[-1])
    if pattern == "() n (j d) -> n j d":
        factor = axes["j"]
        tensor = tensor.squeeze(0)
        return tensor.reshape(tensor.shape[0], factor, tensor.shape[-1] // factor)
    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")


def repeat(tensor: torch.Tensor, pattern: str, **axes) -> torch.Tensor:
    pattern = " ".join(pattern.split())
    if pattern == "j d -> b h j d":
        return tensor.unsqueeze(0).unsqueeze(0).expand(
            axes["b"], axes["h"], tensor.shape[0], tensor.shape[1]
        )
    if pattern in {"b n -> b (n j)", "n d -> n (d j)"}:
        factor = axes["j"]
        return tensor.unsqueeze(-1).expand(*tensor.shape, factor).reshape(
            *tensor.shape[:-1], tensor.shape[-1] * factor
        )
    raise NotImplementedError(f"Unsupported repeat pattern: {pattern}")
