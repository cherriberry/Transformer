"""Plot answer accuracy and objective curves for colour-task runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Random-colour task training curve")
    parser.add_argument(
        "--accuracy-key",
        default=None,
        help=(
            "Metric key for the left panel. Defaults to answer_token_accuracy; "
            "use candidate_memory_token_accuracy for alignment runs."
        ),
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=160)
    accuracy_key = args.accuracy_key or "answer_token_accuracy"
    for run_dir in args.run_dirs:
        rows = read_metrics(run_dir)
        if not rows:
            raise SystemExit(f"no metrics in {run_dir}")
        label = run_dir.name
        steps = [int(row["step"]) for row in rows]
        accuracy = [100.0 * float(row.get(accuracy_key, float("nan"))) for row in rows]
        total = [float(row["total"]) for row in rows]
        left.plot(steps, accuracy, linewidth=1.5, label=label)
        right.plot(steps, total, linewidth=1.5, label=label)
    left.set_xlabel("optimizer step")
    left.set_ylabel(f"training {accuracy_key} (%)")
    left.set_ylim(0, 100)
    left.grid(True, alpha=0.25)
    right.set_xlabel("optimizer step")
    right.set_ylabel("common objective")
    right.grid(True, alpha=0.25)
    left.legend(fontsize=8, loc="best")
    right.legend(fontsize=8, loc="best")
    fig.suptitle(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "runs": [str(path) for path in args.run_dirs]}, indent=2))


if __name__ == "__main__":
    main()
