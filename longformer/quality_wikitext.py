"""Longformer WikiText quality: attention fidelity + HF MLM loss if available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.wikitext_quality import (  # noqa: E402
    attention_fidelity_on_tokens,
    load_wikitext_tokens,
    save_json,
)


class LongformerLocalAttention(nn.Module):
    """Sliding-window approximation used for fidelity vs full attention."""

    def __init__(self, window: int = 64):
        super().__init__()
        self.window = window

    def forward(self, q, k, v):
        # q/k/v: [B,H,N,D]
        b, h, n, d = q.shape
        scale = d**-0.5
        out = torch.zeros_like(v)
        half = self.window // 2
        for i in range(n):
            left = max(0, i - half)
            right = min(n, i + half + 1)
            logits = torch.matmul(q[:, :, i : i + 1], k[:, :, left:right].transpose(-1, -2)) * scale
            probs = torch.softmax(logits.float(), dim=-1).to(v.dtype)
            out[:, :, i : i + 1] = torch.matmul(probs, v[:, :, left:right])
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device
    out_dir = Path(__file__).resolve().parent / "results"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = load_wikitext_tokens(tokenizer, max_tokens=args.max_tokens)
    embed = nn.Embedding(tokenizer.vocab_size, 256).to(device)
    nn.init.normal_(embed.weight, std=0.02)

    # Fixed random projections for fair fidelity test
    gen = torch.Generator(device=device).manual_seed(0)
    w_q = torch.randn(256, 256, generator=gen, device=device) * 0.02
    w_k = torch.randn(256, 256, generator=gen, device=device) * 0.02
    w_v = torch.randn(256, 256, generator=gen, device=device) * 0.02

    def make_qkv(x):
        b, n, c = x.shape
        heads = 8
        dh = c // heads
        q = (x @ w_q).view(b, n, heads, dh).transpose(1, 2)
        k = (x @ w_k).view(b, n, heads, dh).transpose(1, 2)
        v = (x @ w_v).view(b, n, heads, dh).transpose(1, 2)
        return q, k, v

    local = LongformerLocalAttention(window=64)
    fidelity = attention_fidelity_on_tokens(
        tokens, embed, make_qkv, local, seq_len=args.seq_len, batch_chunks=4
    )

    mlm = {"status": "skipped", "reason": "optional heavy model"}
    try:
        model_name = "allenai/longformer-base-4096"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        text_tokens = load_wikitext_tokens(tok, max_tokens=512)
        # MLM on one window
        input_ids = text_tokens[:256].view(1, -1).to(device)
        labels = input_ids.clone()
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[:, 10::5] = True
        input_ids = input_ids.clone()
        input_ids[mask] = tok.mask_token_id
        labels = labels.masked_fill(~mask, -100)
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels).loss
        mlm = {
            "status": "ok",
            "model": model_name,
            "masked_lm_loss": float(loss.item()),
            "masked_lm_ppl": float(torch.exp(loss).item()),
            "tokens": int(input_ids.numel()),
        }
    except Exception as exc:  # noqa: BLE001
        mlm = {"status": "error", "error": repr(exc)}

    payload = {
        "method": "longformer",
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "attention_fidelity_vs_full": fidelity,
        "hf_mlm": mlm,
        "device": device,
        "note": "Fidelity compares local-window attention to full softmax on WikiText embeddings; HF MLM uses pretrained Longformer when downloadable.",
    }
    save_json(out_dir / "quality_wikitext.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
