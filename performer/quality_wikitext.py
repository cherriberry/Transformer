"""Performer WikiText quality using local performer-pytorch package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "performer-pytorch"))
from common.wikitext_quality import (  # noqa: E402
    TinyCausalLM,
    attention_fidelity_on_tokens,
    full_attention,
    load_wikitext_tokens,
    output_metrics,
    save_json,
    train_and_eval_ppl,
)
from performer_pytorch.performer_pytorch import FastAttention, SelfAttention  # noqa: E402


class StandardAttn(nn.Module):
    def __init__(self, dim, heads, seq_len, **kwargs):
        super().__init__()
        self.attn = SelfAttention(dim=dim, heads=heads, causal=True, nb_features=None)
        # Force exact attention path by monkeypatching? SelfAttention uses FastAttention when feature based.
        self.heads = heads
        self.dim_head = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, **kwargs):
        b, n, c = x.shape
        h, d = self.heads, self.dim_head
        q = self.to_q(x).view(b, n, h, d).transpose(1, 2)
        k = self.to_k(x).view(b, n, h, d).transpose(1, 2)
        v = self.to_v(x).view(b, n, h, d).transpose(1, 2)
        out = full_attention(q, k, v)
        # causal mask
        return self.to_out(out.transpose(1, 2).reshape(b, n, c))


class PerformerAttn(nn.Module):
    def __init__(self, dim, heads, seq_len, **kwargs):
        super().__init__()
        self.inner = SelfAttention(dim=dim, heads=heads, causal=True, nb_features=64)

    def forward(self, x, **kwargs):
        return self.inner(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device
    out_dir = Path(__file__).resolve().parent / "results"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = load_wikitext_tokens(tokenizer, max_tokens=args.max_tokens)
    vocab, dim, heads = tokenizer.vocab_size, 256, 8

    embed = nn.Embedding(vocab, dim).to(device)
    nn.init.normal_(embed.weight, std=0.02)
    gen = torch.Generator(device=device).manual_seed(0)
    w_q = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    w_k = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    w_v = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    fast = FastAttention(dim_heads=dim // heads, nb_features=64, causal=False).to(device)

    def make_qkv(x):
        b, n, c = x.shape
        dh = c // heads
        q = (x @ w_q).view(b, n, heads, dh).transpose(1, 2)
        k = (x @ w_k).view(b, n, heads, dh).transpose(1, 2)
        v = (x @ w_v).view(b, n, heads, dh).transpose(1, 2)
        return q, k, v

    def performer_attn(q, k, v):
        # FastAttention expects [B,H,N,D]
        return fast(q, k, v)

    fidelity = attention_fidelity_on_tokens(
        tokens, embed, make_qkv, performer_attn, seq_len=args.seq_len, batch_chunks=4
    )

    std = TinyCausalLM(vocab, dim, 2, heads, StandardAttn, args.seq_len)
    perf = TinyCausalLM(vocab, dim, 2, heads, PerformerAttn, args.seq_len)
    std_ppl = train_and_eval_ppl(std, tokens, args.seq_len, steps=args.steps, device=device)
    perf_ppl = train_and_eval_ppl(perf, tokens, args.seq_len, steps=args.steps, device=device)

    payload = {
        "method": "performer",
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "attention_fidelity_vs_full": fidelity,
        "tiny_lm_standard": std_ppl,
        "tiny_lm_performer": perf_ppl,
        "device": device,
    }
    save_json(out_dir / "quality_wikitext.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
