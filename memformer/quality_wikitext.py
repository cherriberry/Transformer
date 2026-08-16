"""Memformer WikiText quality using local memformer attention adapter ideas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "memformers"))
from common.wikitext_quality import (  # noqa: E402
    TinyCausalLM,
    attention_fidelity_on_tokens,
    full_attention,
    load_wikitext_tokens,
    output_metrics,
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
        out = full_attention(q, k, v)
        return self.to_out(out.transpose(1, 2).reshape(b, n, c))


class MemformerLikeAttn(nn.Module):
    """Persistent memory slots concatenated into K/V (Memformer-style external memory)."""

    def __init__(self, dim, heads, seq_len, mem_slots: int = 32, **kwargs):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.mem_slots = mem_slots
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.memory = nn.Parameter(torch.randn(mem_slots, dim) * 0.02)

    def forward(self, x, **kwargs):
        b, n, c = x.shape
        h, d = self.heads, self.dim_head
        mem = self.memory.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(b, -1, -1)
        x_mem = torch.cat((mem, x), dim=1)
        q = self.to_q(x).view(b, n, h, d).transpose(1, 2)
        k = self.to_k(x_mem).view(b, n + self.mem_slots, h, d).transpose(1, 2)
        v = self.to_v(x_mem).view(b, n + self.mem_slots, h, d).transpose(1, 2)
        out = full_attention(q, k, v)
        return self.to_out(out.transpose(1, 2).reshape(b, n, c))


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
    std_attn = StandardAttn(dim, heads, args.seq_len).to(device)
    mem_attn = MemformerLikeAttn(dim, heads, args.seq_len, mem_slots=32).to(device)
    with torch.no_grad():
        chunk = tokens[: args.seq_len].view(1, -1).to(device)
        x = embed(chunk)
        ref = std_attn(x)
        approx = mem_attn(x)
        fidelity = output_metrics(ref, approx) | {
            "chunks": 1,
            "seq_len": args.seq_len,
            "mode": "memory_augmented_vs_standard",
            "note": "Not an approximation of full attention; measures effect of persistent memory slots on WikiText embeddings.",
        }

    std = TinyCausalLM(vocab, dim, 2, heads, StandardAttn, args.seq_len)
    mem = TinyCausalLM(vocab, dim, 2, heads, MemformerLikeAttn, args.seq_len)
    std_ppl = train_and_eval_ppl(std, tokens, args.seq_len, steps=args.steps, device=device)
    mem_ppl = train_and_eval_ppl(mem, tokens, args.seq_len, steps=args.steps, device=device)

    payload = {
        "method": "memformer",
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "attention_effect_on_wikitext": fidelity,
        "tiny_lm_standard": std_ppl,
        "tiny_lm_memformer_like": mem_ppl,
        "device": device,
        "source_note": "Performance microbenchmark artifacts retained from teammate Memformer/Performer/Reformer suite; quality uses Memformer-style memory slots on WikiText.",
    }
    save_json(out_dir / "quality_wikitext.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
