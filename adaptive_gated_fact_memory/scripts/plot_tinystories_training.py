"""Plot TinyStories validation accuracy and NLL from a training run.

The training runner writes validation records to ``metrics.jsonl``.  This
utility produces a compact PNG with token accuracy and token-weighted NLL so a
parent checkpoint's convergence can be inspected without loading the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_validation_records(metrics_path: Path) -> list[dict]:
    records: list[dict] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        probe = row.get("validation_probe")
        if probe is None and row.get("event") == "initial_validation_probe":
            probe = row
        if probe is None:
            continue
        step = int(row.get("step", probe.get("step", 0)))
        records.append(
            {
                "step": step,
                "tokens_seen": int(row.get("tokens_seen", probe.get("tokens", 0))),
                "token_accuracy": float(probe["token_accuracy"]),
                "token_weighted_nll": float(probe["token_weighted_nll"]),
                "perplexity": float(probe["perplexity"]),
            }
        )
    records.sort(key=lambda item: (item["step"], item["tokens_seen"]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    metrics_path = args.run_dir / "metrics.jsonl"
    records = read_validation_records(metrics_path)
    if not records:
        raise SystemExit(f"no validation records found in {metrics_path}")
    output = args.output or (args.run_dir / "training_accuracy_curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    x = [record["tokens_seen"] for record in records]
    accuracy = [record["token_accuracy"] * 100.0 for record in records]
    nll = [record["token_weighted_nll"] for record in records]
    fig, left = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    left.plot(x, accuracy, marker="o", linewidth=2, label="validation token accuracy")
    left.set_xlabel("training tokens seen")
    left.set_ylabel("token accuracy (%)")
    left.grid(True, alpha=0.25)
    left.set_ylim(0, 100)
    right = left.twinx()
    right.plot(x, nll, color="#d95f02", marker="s", linewidth=1.6, label="validation NLL")
    right.set_ylabel("token-weighted NLL")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    left.set_title("TinyStories parent training validation curve")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

    print(json.dumps({"output": str(output), "validation_points": len(records)}, indent=2))


if __name__ == "__main__":
    main()
