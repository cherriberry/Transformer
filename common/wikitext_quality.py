"""Shared WikiText helpers and attention-quality metrics for all methods."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load_wikitext_tokens(tokenizer, max_tokens: int = 2048) -> torch.Tensor:
    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in dataset if row["text"] and row["text"].strip())
    approx_chars = max_tokens * 6
    tokens = tokenizer(text[:approx_chars], return_tensors="pt", truncation=False).input_ids[0]
    return tokens[:max_tokens]


def full_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q/k/v: [B, H, N, D] -> [B, H, N, D]."""
    scale = q.size(-1) ** -0.5
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(logits.float(), dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def output_metrics(reference: torch.Tensor, approx: torch.Tensor) -> dict[str, float]:
    ref = reference.float().reshape(-1)
    ap = approx.float().reshape(-1)
    diff = ap - ref
    mse = torch.mean(diff.square()).item()
    rel = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref).clamp_min(1e-12)).item()
    cos = F.cosine_similarity(ref.unsqueeze(0), ap.unsqueeze(0), dim=-1)[0].item()
    return {"mse": mse, "relative_l2": rel, "cosine_similarity": cos}


@torch.no_grad()
def attention_fidelity_on_tokens(
    tokens: torch.Tensor,
    embed: torch.nn.Embedding,
    make_qkv,
    efficient_attn,
    seq_len: int = 256,
    batch_chunks: int = 4,
) -> dict[str, float]:
    """Compare efficient attention vs full softmax on real WikiText embeddings."""
    device = embed.weight.device
    totals = {"mse": 0.0, "relative_l2": 0.0, "cosine_similarity": 0.0}
    count = 0
    tokens = tokens.to(device)
    for start in range(0, max(1, tokens.numel() - seq_len), seq_len):
        if count >= batch_chunks:
            break
        chunk = tokens[start : start + seq_len]
        if chunk.numel() < seq_len:
            break
        x = embed(chunk.view(1, -1))
        q, k, v = make_qkv(x)
        ref = full_attention(q, k, v)
        approx = efficient_attn(q, k, v)
        m = output_metrics(ref, approx)
        for key in totals:
            totals[key] += m[key]
        count += 1
    if count == 0:
        raise RuntimeError("no WikiText chunks evaluated")
    return {k: v / count for k, v in totals.items()} | {"chunks": count, "seq_len": seq_len}


def nll_to_ppl(nll: float) -> float:
    return math.exp(nll)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TinyCausalLM(torch.nn.Module):
    """Small causal LM with a swappable self-attention module factory."""

    def __init__(self, vocab: int, dim: int, layers: int, heads: int, attn_factory, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.tok = torch.nn.Embedding(vocab, dim)
        self.pos = torch.nn.Embedding(seq_len, dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                torch.nn.ModuleDict(
                    {
                        "ln1": torch.nn.LayerNorm(dim),
                        "attn": attn_factory(dim=dim, heads=heads, seq_len=seq_len),
                        "ln2": torch.nn.LayerNorm(dim),
                        "mlp": torch.nn.Sequential(
                            torch.nn.Linear(dim, 4 * dim),
                            torch.nn.GELU(),
                            torch.nn.Linear(4 * dim, dim),
                            torch.nn.Dropout(dropout),
                        ),
                    }
                )
            )
        self.ln_f = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)
        self.seq_len = seq_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, n = idx.shape
        if n > self.seq_len:
            raise ValueError("sequence longer than model seq_len")
        pos = torch.arange(n, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for block in self.layers:
            a = block["attn"](block["ln1"](x))
            if isinstance(a, tuple):
                a = a[0]
            x = x + a
            x = x + block["mlp"](block["ln2"](x))
        return self.head(self.ln_f(x))


def train_and_eval_ppl(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    seq_len: int,
    steps: int = 200,
    batch_size: int = 8,
    lr: float = 3e-4,
    device: str = "cuda",
) -> dict[str, float]:
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tokens = tokens.to(device)
    usable = tokens.numel() - seq_len - 1
    if usable < batch_size:
        raise RuntimeError("not enough tokens for training windows")

    losses: list[float] = []
    for step in range(steps):
        starts = torch.randint(0, usable, (batch_size,), device=device)
        batch = torch.stack([tokens[s : s + seq_len] for s in starts], dim=0)
        target = torch.stack([tokens[s + 1 : s + seq_len + 1] for s in starts], dim=0)
        logits = model(batch)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        eval_nll = []
        for s in range(0, min(usable, seq_len * 20), seq_len):
            batch = tokens[s : s + seq_len].view(1, -1)
            target = tokens[s + 1 : s + seq_len + 1].view(1, -1)
            if target.numel() != batch.numel():
                continue
            logits = model(batch)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            eval_nll.append(float(nll.item()))
    mean_nll = sum(eval_nll) / max(1, len(eval_nll))
    return {
        "train_steps": steps,
        "train_loss_last": losses[-1] if losses else float("nan"),
        "eval_nll": mean_nll,
        "eval_perplexity": nll_to_ppl(mean_nll),
        "eval_windows": len(eval_nll),
        "seq_len": seq_len,
    }
