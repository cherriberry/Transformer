"""Run the B-role 2,031,616-token parameter-screening pilot.

The runner keeps the protocol's TinyLM width/depth and candidate attention
settings.  When the Hugging Face stack/PG-19 is unavailable it uses a
deterministic copy stream and records that fact in every result; this is a
controlled implementation smoke, not a PG-19 claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.longformer_attention import LongformerSelfAttention  # noqa: E402
from b_role_experiments import MemformerSegmentAttention, StandardAttention  # noqa: E402

PROTOCOL = "long_context_10m_v1"
PILOT_TOKENS = 2_031_616


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


class PilotLM(nn.Module):
    def __init__(self, method: str, vocab: int, dim: int, layers: int, heads: int,
                 seq_len: int, left_window: int = 128, segment_length: int = 256,
                 memory_slots: int = 64):
        super().__init__()
        self.method, self.seq_len, self.vocab = method, seq_len, vocab
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            if method == "longformer":
                attn = LongformerSelfAttention(
                    dim, seq_len, heads=heads, window_size=2 * left_window + 1,
                    global_tokens=0, causal=True, query_chunk_size=256,
                )
            elif method == "memformer":
                attn = MemformerSegmentAttention(
                    dim, heads, memory_slots=memory_slots, causal=True,
                    detach_memory=False,
                )
            else:
                attn = StandardAttention(dim, heads, causal=True)
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(dim), attn, nn.LayerNorm(dim),
                nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)),
            ]))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, n = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(n, device=idx.device))[None]
        memories: list[torch.Tensor | None] = [None] * len(self.blocks)
        for i, (ln1, attn, ln2, mlp) in enumerate(self.blocks):
            if self.method == "memformer":
                a, memories[i] = attn(
                    ln1(x), segment_length=self._segment_length,
                    memory=memories[i],
                )
            else:
                a = attn(ln1(x))
            x = x + a
            x = x + mlp(ln2(x))
        return self.head(self.norm(x))

    def set_segment_length(self, value: int) -> None:
        self._segment_length = value


def make_batch(batch: int, seq_len: int, vocab: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(vocab, (batch, seq_len), generator=gen)
    # A deterministic next-token stream gives stable pilot signal without
    # silently replacing PG-19 in the recorded dataset field.
    y = (x + 3) % vocab
    return x, y


def train_candidate(cfg: dict[str, Any], args: argparse.Namespace, device: torch.device,
                    dtype: torch.dtype, out_root: Path) -> dict[str, Any]:
    seed = int(cfg["seed"])
    seed_all(seed)
    run_id = cfg["run_id"]
    run_dir = out_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = run_dir / "config.resolved.json"
    if args.skip_existing and existing.exists():
        try:
            cached = json.loads(existing.read_text(encoding="utf-8"))
            if cached.get("status") == "ok" and cached.get("trained_tokens", 0) >= PILOT_TOKENS:
                print(f"SKIP {run_id} (already complete)", flush=True)
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    model = PilotLM(
        cfg["method"], vocab=args.vocab, dim=args.hidden_size, layers=args.layers,
        heads=args.heads, seq_len=args.seq_len,
        left_window=cfg.get("left_window", 128),
        segment_length=cfg.get("segment_length", 256),
        memory_slots=cfg.get("memory_slots", 64),
    ).to(device=device, dtype=dtype)
    if cfg["method"] == "memformer":
        model.set_segment_length(int(cfg["segment_length"]))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1)
    batch = args.batch
    steps = math.ceil(PILOT_TOKENS / (batch * args.seq_len))
    metrics_path = run_dir / "metrics.jsonl"
    start = time.perf_counter()
    best = float("inf")
    last_loss = float("nan")
    status = "ok"
    with metrics_path.open("w", encoding="utf-8") as log:
        for step in range(1, steps + 1):
            idx, target = make_batch(batch, args.seq_len, args.vocab, seed + step)
            idx, target = idx.to(device), target.to(device)
            try:
                logits = model(idx)
                loss = F.cross_entropy(logits.float().reshape(-1, args.vocab), target.reshape(-1))
                if not torch.isfinite(loss):
                    status = "error"
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                if not math.isfinite(grad_norm):
                    status = "error"
                    break
                opt.step()
            except torch.cuda.OutOfMemoryError:
                status = "oom"
                break
            last_loss = float(loss.detach().cpu())
            best = min(best, last_loss)
            if step == 1 or step % args.log_interval == 0 or step == steps:
                rec = {"step": step, "trained_tokens": min(step * batch * args.seq_len, PILOT_TOKENS),
                       "train_nll": last_loss, "train_ppl": math.exp(min(last_loss, 20.0)),
                       "grad_norm": grad_norm, "wall_time_s": time.perf_counter() - start}
                log.write(json.dumps(rec) + "\n"); log.flush()
    elapsed = time.perf_counter() - start
    trained_tokens = min((step if 'step' in locals() else 0) * batch * args.seq_len, PILOT_TOKENS)
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved = torch.cuda.max_memory_reserved() / 2**20
    else:
        peak_alloc = peak_reserved = None
    resolved = {**cfg, "protocol_version": PROTOCOL, "result_label": "parameter_screening",
                "dataset": "synthetic_copy_stream_fallback", "requested_dataset": "PG-19",
                "seq_len": args.seq_len, "vocab_size": args.vocab, "hidden_size": args.hidden_size,
                "layers": args.layers, "heads": args.heads, "training_tokens_target": PILOT_TOKENS,
                "trained_tokens": trained_tokens, "status": status,
                "config_hash": stable_hash(cfg), "dtype": str(dtype).replace("torch.", ""),
                "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "elapsed_s": elapsed, "tokens_per_s": trained_tokens / max(elapsed, 1e-9),
                "best_train_nll": best if best != float("inf") else None,
                "final_train_nll": last_loss if math.isfinite(last_loss) else None,
                "peak_allocated_mb": peak_alloc, "peak_reserved_mb": peak_reserved,
                "protocol_deviations": ["requested PG-19 but used synthetic fallback",
                                        "sequence length 512 instead of 4096",
                                        "vocabulary 64 instead of GPT-2 50257",
                                        "learned absolute positions instead of RoPE"],
                "note": "Implementation/throughput smoke only; not usable for natural-language candidate selection."}
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}, run_dir / "checkpoint.final.pt")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return resolved


def candidates() -> list[dict[str, Any]]:
    rows = []
    for w in (128, 512, 1024, 2048):
        rows.append({"run_id": f"b_longformer_w{w}_s17", "method": "longformer", "seed": 17,
                     "left_window": w, "window_size": 2 * w + 1, "global_tokens": 0})
    for seg, slots in ((256, 64), (512, 32), (512, 64), (512, 128)):
        rows.append({"run_id": f"b_memformer_seg{seg}_m{slots}_s17", "method": "memformer", "seed": 17,
                     "segment_length": seg, "memory_slots": slots})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=ROOT / "experiments" / PROTOCOL)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--log-interval", type=int, default=64)
    p.add_argument("--max-runs", type=int, default=8)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--run-id", action="append", default=[])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    out_root = args.out_root.resolve(); (out_root / "runs").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    all_rows = []
    selected = [c for c in candidates() if not args.run_id or c["run_id"] in set(args.run_id)]
    for cfg in selected[:args.max_runs]:
        print(f"START {cfg['run_id']}", flush=True)
        row = train_candidate(cfg, args, device, dtype, out_root)
        all_rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("run_id", "status", "trained_tokens", "tokens_per_s", "best_train_nll", "peak_allocated_mb")}), flush=True)
    summary = {"protocol_version": PROTOCOL, "scope": "role_b_longformer_memformer",
               "result_label": "parameter_screening", "target_tokens_per_run": PILOT_TOKENS,
               "records": all_rows, "environment": {"python": platform.python_version(),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
               "note": "All rows use deterministic synthetic_copy_stream_fallback until PG-19 dependencies are available."}
    (out_root / "aggregate" / "pilot_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (out_root / "aggregate" / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"completed": len(all_rows), "summary": str(out_root / 'aggregate' / 'pilot_summary.json')}), flush=True)


if __name__ == "__main__":
    main()
