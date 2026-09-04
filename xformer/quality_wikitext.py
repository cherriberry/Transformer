"""xFormer / memory-efficient attention WikiText quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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

try:
    import xformers.ops as xops

    HAS_XFORMERS = True
except Exception:  # noqa: BLE001
    HAS_XFORMERS = False


class StandardAttn(nn.Module):
    def __init__(self, dim, heads, seq_len, **kwargs):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, **kwargs):
        b, n, c = x.shape
        h, d = self.heads, self.dim_head
        qkv = self.to_qkv(x).view(b, n, 3, h, d)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.to_out(out.transpose(1, 2).reshape(b, n, c))


class EfficientAttn(nn.Module):
    def __init__(self, dim, heads, seq_len, **kwargs):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, **kwargs):
        b, n, c = x.shape
        h, d = self.heads, self.dim_head
        qkv = self.to_qkv(x).view(b, n, 3, h, d)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]
        # SDPA is the portable "memory-efficient attention" path on Windows / new GPUs.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
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
    gen = torch.Generator(device=device).manual_seed(0)
    w_q = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    w_k = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    w_v = torch.randn(dim, dim, generator=gen, device=device) * 0.02

    def make_qkv(x):
        b, n, c = x.shape
        dh = c // heads
        q = (x @ w_q).view(b, n, heads, dh)
        k = (x @ w_k).view(b, n, heads, dh)
        v = (x @ w_v).view(b, n, heads, dh)
        # fidelity helper expects [B,H,N,D]
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def efficient(q, k, v):
        # Prefer PyTorch SDPA on RTX 50-series: many xFormers kernels lack sm_120 support.
        try:
            if HAS_XFORMERS:
                qq, kk, vv = [t.transpose(1, 2).contiguous().half() for t in (q, k, v)]
                out = xops.memory_efficient_attention(qq, kk, vv)
                return out.transpose(1, 2).to(q.dtype)
        except Exception:
            pass
        return F.scaled_dot_product_attention(q, k, v)

    fidelity = attention_fidelity_on_tokens(
        tokens, embed, make_qkv, efficient, seq_len=args.seq_len, batch_chunks=4
    )

    std = TinyCausalLM(vocab, dim, 2, heads, StandardAttn, args.seq_len)
    eff = TinyCausalLM(vocab, dim, 2, heads, EfficientAttn, args.seq_len)
    std_ppl = train_and_eval_ppl(std, tokens, args.seq_len, steps=args.steps, device=device)
    eff_ppl = train_and_eval_ppl(eff, tokens, args.seq_len, steps=args.steps, device=device)

    payload = {
        "method": "xformer",
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "xformers_available": HAS_XFORMERS,
        "attention_fidelity_vs_full": fidelity,
        "tiny_lm_standard_sdpa_causal": std_ppl,
        "tiny_lm_memory_efficient": eff_ppl,
        "device": device,
    }
    save_json(out_dir / "quality_wikitext.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
