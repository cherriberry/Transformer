"""Plot Keyformer performance and WikiText-2 quality results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_perf(perf_path: Path, out_dir: Path) -> None:
    data = _load(perf_path)
    rows = [r for r in data["rows"] if r.get("status") == "ok"]
    if not rows:
        print("No successful perf rows")
        return

    # Decode latency vs prompt length at cache_ratio=0.5, gen fixed
    gen_lens = sorted({r["gen_len"] for r in rows})
    gen = gen_lens[0]
    ratios = sorted({r["cache_ratio"] for r in rows})
    ratio = 0.5 if 0.5 in ratios else ratios[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for policy in sorted({r["policy"] for r in rows}):
        xs, ys = [], []
        for prompt in sorted({r["prompt_len"] for r in rows}):
            vals = [
                r["decode_ms_mean"]
                for r in rows
                if r["policy"] == policy and r["prompt_len"] == prompt and r["gen_len"] == gen and r["cache_ratio"] == ratio
            ]
            if vals:
                xs.append(prompt)
                ys.append(sum(vals) / len(vals))
        if xs:
            ax.plot(xs, ys, marker="o", label=policy)
    ax.set_title(f"Decode latency/token (gen={gen}, ratio={ratio})")
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("ms / token")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for policy in sorted({r["policy"] for r in rows}):
        xs, ys = [], []
        for prompt in sorted({r["prompt_len"] for r in rows}):
            vals = [
                r["decode_tokens_per_s"]
                for r in rows
                if r["policy"] == policy and r["prompt_len"] == prompt and r["gen_len"] == gen and r["cache_ratio"] == ratio
            ]
            if vals:
                xs.append(prompt)
                ys.append(sum(vals) / len(vals))
        if xs:
            ax.plot(xs, ys, marker="o", label=policy)
    ax.set_title(f"Decode throughput (gen={gen}, ratio={ratio})")
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("tokens / s")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for policy in sorted({r["policy"] for r in rows}):
        xs, ys = [], []
        for cr in sorted({r["cache_ratio"] for r in rows}):
            vals = [
                r["kv_bytes_total"] / (1024**2)
                for r in rows
                if r["policy"] == policy and r["cache_ratio"] == cr and r["gen_len"] == gen
            ]
            if vals:
                xs.append(cr)
                ys.append(sum(vals) / len(vals))
        if xs:
            ax.plot(xs, ys, marker="o", label=policy)
    ax.set_title("KV cache bytes vs cache ratio")
    ax.set_xlabel("Cache ratio")
    ax.set_ylabel("MiB (all layers)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for policy in sorted({r["policy"] for r in rows}):
        xs, ys = [], []
        for cr in sorted({r["cache_ratio"] for r in rows}):
            vals = [
                r["cuda_peak_bytes"] / (1024**2)
                for r in rows
                if r["policy"] == policy and r["cache_ratio"] == cr and r["gen_len"] == gen
            ]
            if vals:
                xs.append(cr)
                ys.append(sum(vals) / len(vals))
        if xs:
            ax.plot(xs, ys, marker="o", label=policy)
    ax.set_title("CUDA peak memory vs cache ratio")
    ax.set_xlabel("Cache ratio")
    ax.set_ylabel("MiB")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "keyformer_perf.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_quality(quality_path: Path, out_dir: Path) -> None:
    data = _load(quality_path)
    rows = [r for r in data["rows"] if r.get("status") == "ok"]
    if not rows:
        print("No successful quality rows")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for policy in sorted({r["policy"] for r in rows}):
        xs, ys = [], []
        for cr in sorted({r["cache_ratio"] for r in rows}):
            vals = [r["perplexity"] for r in rows if r["policy"] == policy and r["cache_ratio"] == cr]
            if vals:
                xs.append(cr)
                ys.append(sum(vals) / len(vals))
        if xs:
            ax.plot(xs, ys, marker="o", label=policy)
    ax.set_title("WikiText-2 perplexity vs cache ratio")
    ax.set_xlabel("Cache ratio")
    ax.set_ylabel("Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for policy in sorted({r["policy"] for r in rows}):
        xs, ys = [], []
        for cr in sorted({r["cache_ratio"] for r in rows}):
            vals = [
                100.0 * r.get("ppl_increase_vs_fullkv", 0.0)
                for r in rows
                if r["policy"] == policy and r["cache_ratio"] == cr
            ]
            if vals:
                xs.append(cr)
                ys.append(sum(vals) / len(vals))
        if xs:
            ax.plot(xs, ys, marker="o", label=policy)
    ax.set_title("PPL increase vs FullKV")
    ax.set_xlabel("Cache ratio")
    ax.set_ylabel("% increase")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "keyformer_quality.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", type=Path, default=ROOT / "results" / "synthetic" / "perf_final.json")
    parser.add_argument("--quality", type=Path, default=ROOT / "results" / "wikitext2" / "quality_quality_final.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.perf.exists():
        plot_perf(args.perf, args.out_dir)
    else:
        quick = ROOT / "results" / "synthetic" / "perf_quick.json"
        if quick.exists():
            plot_perf(quick, args.out_dir)
    if args.quality.exists():
        plot_quality(args.quality, args.out_dir)
    else:
        quick_q = ROOT / "results" / "wikitext2" / "quality_quality_quick.json"
        if quick_q.exists():
            plot_quality(quick_q, args.out_dir)


if __name__ == "__main__":
    main()
