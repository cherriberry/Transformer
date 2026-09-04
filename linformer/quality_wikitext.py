"""Linformer WikiText quality: attention fidelity + tiny LM PPL vs standard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from linformer import LinformerSelfAttention
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.wikitext_quality import (  # noqa: E402
    TinyCausalLM,
    attention_fidelity_on_tokens,
    load_wikitext_tokens,
    save_json,
    train_and_eval_ppl,
)


class StandardAttn(nn.Module):
    def __init__(self, dim, heads, seq_len, **kwargs):
        super().__init__()
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
        scale = d**-0.5
        attn = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.to_out(out)


class LinformerAttnWrapper(nn.Module):
    def __init__(self, dim, heads, seq_len, k=64, **kwargs):
        super().__init__()
        self.inner = LinformerSelfAttention(dim=dim, seq_len=seq_len, heads=heads, k=k)

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
    vocab = tokenizer.vocab_size
    dim, heads = 256, 8

    # Fidelity: Linformer projection attention vs full attention on shared QKV
    embed = nn.Embedding(vocab, dim).to(device)
    nn.init.normal_(embed.weight, std=0.02)
    gen = torch.Generator(device=device).manual_seed(0)
    w_q = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    w_k = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    w_v = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    # Learn-free low-rank K/V projection (Linformer idea)
    e_k = torch.randn(args.seq_len, 64, generator=gen, device=device) * (args.seq_len**-0.5)
    e_v = torch.randn(args.seq_len, 64, generator=gen, device=device) * (args.seq_len**-0.5)

    def make_qkv(x):
        b, n, c = x.shape
        dh = c // heads
        q = (x @ w_q).view(b, n, heads, dh).transpose(1, 2)
        k = (x @ w_k).view(b, n, heads, dh).transpose(1, 2)
        v = (x @ w_v).view(b, n, heads, dh).transpose(1, 2)
        return q, k, v

    def linformer_attn(q, k, v):
        # project length dim of K,V: [B,H,N,D] -> [B,H,k,D]
        # k^T E : use einsum over sequence
        # E: [N,k]
        kp = torch.einsum("bhnd,nk->bhkd", k, e_k)
        vp = torch.einsum("bhnd,nk->bhkd", v, e_v)
        scale = q.size(-1) ** -0.5
        attn = torch.softmax((q @ kp.transpose(-1, -2)) * scale, dim=-1)
        return attn @ vp

    fidelity = attention_fidelity_on_tokens(
        tokens, embed, make_qkv, linformer_attn, seq_len=args.seq_len, batch_chunks=4
    )

    std_model = TinyCausalLM(vocab, dim, layers=2, heads=heads, attn_factory=StandardAttn, seq_len=args.seq_len)
    lin_model = TinyCausalLM(
        vocab, dim, layers=2, heads=heads, attn_factory=LinformerAttnWrapper, seq_len=args.seq_len
    )
    std_ppl = train_and_eval_ppl(std_model, tokens, args.seq_len, steps=args.steps, device=device)
    lin_ppl = train_and_eval_ppl(lin_model, tokens, args.seq_len, steps=args.steps, device=device)

    payload = {
        "method": "linformer",
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "attention_fidelity_vs_full": fidelity,
        "tiny_lm_standard": std_ppl,
        "tiny_lm_linformer": lin_ppl,
        "device": device,
    }
    save_json(out_dir / "quality_wikitext.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
