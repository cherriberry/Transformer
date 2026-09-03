"""Validate the B-role pilot configuration without optional YAML dependencies.

The protocol YAML is the human-readable source of truth. This validator checks
the machine-facing jobs.csv and records hashes/environment metadata so a pilot
run can be reproduced even when the optional Hugging Face stack is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def validate_jobs(path: Path) -> dict[str, object]:
    required = {
        "run_id", "protocol_version", "result_label", "method", "seed",
        "requested_dataset", "actual_dataset", "split", "training_tokens",
        "trained_tokens", "status", "status_reason",
    }
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    errors: list[str] = []
    if not rows:
        errors.append("jobs.csv contains no rows")
    missing_columns = sorted(required - set(rows[0])) if rows else sorted(required)
    if missing_columns:
        errors.append(f"jobs.csv is missing required columns: {missing_columns}")
    ids = [r.get("run_id", "") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("run_id values are not unique")
    if any(r.get("protocol_version") != "long_context_10m_v1" for r in rows):
        errors.append("all jobs must use long_context_10m_v1")
    if any(r.get("result_label") != "parameter_screening" for r in rows):
        errors.append("pilot jobs must use result_label=parameter_screening")
    if any(r.get("seed") != "17" for r in rows):
        errors.append("all screening jobs must use seed 17")
    if any(r.get("requested_dataset") != "PG-19" or r.get("split") != "train" for r in rows):
        errors.append("screening jobs must target PG-19 train")
    tokens = {r.get("training_tokens") for r in rows}
    if tokens != {"2031616"}:
        errors.append(f"pilot token budget mismatch: {sorted(tokens)}")
    methods = {r.get("method") for r in rows}
    if methods != {"longformer", "memformer"}:
        errors.append(f"unexpected methods: {sorted(methods)}")
    lf = [r for r in rows if r.get("method") == "longformer"]
    mf = [r for r in rows if r.get("method") == "memformer"]
    if len(lf) != 4 or len(mf) != 4:
        errors.append(f"expected 4 Longformer + 4 Memformer jobs, got {len(lf)} + {len(mf)}")
    allowed_status = {"queued", "running", "ok", "oom", "error", "failed_correctness",
                      "needs_rerun", "not_run", "ineligible_causal"}
    if any(r.get("status") not in allowed_status for r in rows):
        errors.append("jobs.csv contains a status outside the protocol vocabulary")
    for row in lf:
        try:
            w = int(row["window_size"])
            if w % 2 != 1:
                errors.append(f"Longformer window_size must be odd: {row['run_id']}")
            if w != 2 * int(row["left_window"]) + 1:
                errors.append(f"Longformer window_size/left_window mismatch: {row['run_id']}")
            if int(row["global_tokens"]) != 0:
                errors.append(f"causal pilot cannot use global tokens: {row['run_id']}")
        except (KeyError, ValueError):
            errors.append(f"invalid Longformer candidate fields: {row.get('run_id')}")
    for row in mf:
        try:
            if int(row["segment_length"]) <= 0 or int(row["memory_slots"]) <= 0:
                errors.append(f"Memformer dimensions must be positive: {row['run_id']}")
        except (KeyError, ValueError):
            errors.append(f"invalid Memformer candidate fields: {row.get('run_id')}")
    return {"status": "ok" if not errors else "error", "rows": len(rows), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = root / "protocol.yaml"
    jobs = root / "jobs" / "jobs.csv"
    check = validate_jobs(jobs)
    payload = {
        "protocol_version": "long_context_10m_v1",
        "scope": "role_b_longformer_memformer",
        "config_status": check,
        "files": {
            "protocol_yaml": {"path": str(protocol), "sha256": sha256(protocol)},
            "jobs_csv": {"path": str(jobs), "sha256": sha256(jobs)},
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git_commit": git_commit(root.parents[1]),
        },
    }
    out = root / "config_validation.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if check["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
