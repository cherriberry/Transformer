"""Render compact plots from the Person-B raw JSONL/JSON records."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("PERSON_B_RUNS_DIR", str(EXP / "runs" / "person_b")))
AGG = Path(os.environ.get("PERSON_B_AGGREGATE_DIR", str(EXP / "aggregate")))


def load_metrics(run_id: str) -> list[dict]:
    path = RUNS / run_id / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    long_rows = load_metrics("main_b_longformer_w128_s17")
    mem_rows = load_metrics("main_b_memformer_s128_m64_s17")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for label, rows, color in (
        ("Longformer w=128", long_rows, "#2E74B5"),
        ("Memformer s=128,m=64", mem_rows, "#D97706"),
    ):
        train = [r for r in rows if r.get("event") == "train_step"]
        probes = [r for r in train if "validation_probe" in r]
        axes[0].plot(
            [r["tokens_seen"] / 1e6 for r in train],
            [r["train_nll"] for r in train],
            color=color,
            alpha=0.35,
            linewidth=1,
            label=f"{label} train",
        )
        axes[1].plot(
            [r["tokens_seen"] / 1e6 for r in probes],
            [r["validation_probe"]["token_weighted_nll"] for r in probes],
            "o-",
            color=color,
            label=label,
        )
    axes[0].set(xlabel="Training tokens (M)", ylabel="Train NLL", title="Person-B training curves")
    axes[1].set(xlabel="Training tokens (M)", ylabel="Validation probe NLL", title="Validation probe curves")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(AGG / "person_b_training_validation_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    efficiency = json.loads((AGG / "person_b_efficiency.json").read_text(encoding="utf-8"))
    attention = json.loads((AGG / "person_b_attention_efficiency.json").read_text(encoding="utf-8"))
    for data, output, title in (
        (efficiency, "person_b_end_to_end_efficiency_curves.png", "End-to-end TinyLM forward"),
        (attention, "person_b_attention_only_efficiency_curves.png", "Single-layer attention-only"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for method, label, color in (
            ("full_attention_reference", "Full-Attention reference", "#555555"),
            ("longformer", "Longformer", "#2E74B5"),
            ("memformer", "Memformer", "#D97706"),
        ):
            rows = [r for r in data["records"] if r["method"] == method and r["status"] == "ok"]
            rows.sort(key=lambda r: r["sequence_length"])
            axes[0].plot(
                [r["sequence_length"] for r in rows],
                [r["median_latency_seconds"] * 1000 for r in rows],
                "o-",
                label=label,
                color=color,
            )
            axes[1].plot(
                [r["sequence_length"] for r in rows],
                [r["peak_incremental_bytes"] / 2**20 for r in rows],
                "o-",
                label=label,
                color=color,
            )
        axes[0].set(xscale="log", yscale="log", xlabel="Sequence length", ylabel="Median latency (ms)", title=f"{title}: latency")
        axes[1].set(xscale="log", xlabel="Sequence length", ylabel="Peak incremental memory (MiB)", title=f"{title}: memory")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(AGG / output, dpi=180, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
