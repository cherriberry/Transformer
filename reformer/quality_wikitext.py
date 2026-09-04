"""Reformer WikiText quality using local reformer-pytorch package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reformer-pytorch"))
from common.wikitext_quality import (  # noqa: E402
    TinyCausalLM,
    attention_fidelity_on_tokens,
    full_attention,
    load_wikitext_tokens,
    save_json,
    train_and_eval_ppl,
)
from reformer_pytorch.reformer_pytorch import LSHSelfAttention  # noqa: E402


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


class ReformerAttn(nn.Module):
    def __init__(self, dim, heads, seq_len, **kwargs):
        super().__init__()
        # bucket_size must divide sequence length
        bucket = 32 if seq_len % 32 == 0 else 16
        self.inner = LSHSelfAttention(dim=dim, heads=heads, bucket_size=bucket, n_hashes=4, causal=True)

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

    # Fidelity: compare LSH attention output vs full on same projections is noisy;
    # still report cosine/MSE for documentation, then focus on LM PPL.
    embed = nn.Embedding(vocab, dim).to(device)
    nn.init.normal_(embed.weight, std=0.02)
    gen = torch.Generator(device=device).manual_seed(0)
    w_q = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    w_k = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    w_v = torch.randn(dim, dim, generator=gen, device=device) * 0.02
    lsh = LSHSelfAttention(dim=dim, heads=heads, bucket_size=32, n_hashes=4, causal=False).to(device)

    def make_qkv(x):
        b, n, c = x.shape
        dh = c // heads
        q = (x @ w_q).view(b, n, heads, dh).transpose(1, 2)
        k = (x @ w_k).view(b, n, heads, dh).transpose(1, 2)
        v = (x @ w_v).view(b, n, heads, dh).transpose(1, 2)
        return q, k, v

    def reformer_like(q, k, v):
        # Use module on reconstituted x-less path: approximate by attending with LSH on v-shaped input
        # Fallback: full attention placeholder if LSH API needs x
        return full_attention(q, k, v)  # structural placeholder; LM path is primary for Reformer

    fidelity = attention_fidelity_on_tokens(
        tokens, embed, make_qkv, reformer_like, seq_len=args.seq_len, batch_chunks=2
    )
    # Better fidelity: run LSHSelfAttention on embeddings vs standard attn module
    std_attn = StandardAttn(dim, heads, args.seq_len).to(device)
    with torch.no_grad():
        chunk = tokens[: args.seq_len].view(1, -1).to(device)
        x = embed(chunk)
        ref = std_attn(x)
        approx = lsh(x)
        from common.wikitext_quality import output_metrics

        fidelity = output_metrics(ref, approx) | {"chunks": 1, "seq_len": args.seq_len, "mode": "lsh_vs_standard_module"}

    std = TinyCausalLM(vocab, dim, 2, heads, StandardAttn, args.seq_len)
    refm = TinyCausalLM(vocab, dim, 2, heads, ReformerAttn, args.seq_len)
    std_ppl = train_and_eval_ppl(std, tokens, args.seq_len, steps=args.steps, device=device)
    ref_ppl = train_and_eval_ppl(refm, tokens, args.seq_len, steps=args.steps, device=device)

    payload = {
        "method": "reformer",
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "attention_fidelity_vs_full": fidelity,
        "tiny_lm_standard": std_ppl,
        "tiny_lm_reformer": ref_ppl,
        "device": device,
    }
    save_json(out_dir / "quality_wikitext.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
