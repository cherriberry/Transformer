"""Aggregate every completed B-role pilot run without manual numbers."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = ROOT / "aggregate"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    records = []
    for cfg_path in sorted(RUNS.glob("*/config.resolved.json")):
        row = read_json(cfg_path)
        records.append(row)
    records.sort(key=lambda row: row.get("run_id", ""))
    summary = {
        "protocol_version": "long_context_10m_v1",
        "scope": "role_b_longformer_memformer",
        "result_label": "parameter_screening",
        "target_tokens_per_run": 2_031_616,
        "records": records,
        "completed_ok": sum(row.get("status") == "ok" for row in records),
        "completed_full_budget": sum(
            row.get("status") == "ok" and row.get("trained_tokens", 0) >= 2_031_616
            for row in records
        ),
        "note": "Aggregate includes all run directories; data is synthetic fallback unless PG-19 is explicitly available.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    jobs_path = ROOT / "jobs" / "jobs.csv"
    if jobs_path.exists():
        with jobs_path.open(encoding="utf-8", newline="") as f:
            jobs = list(csv.DictReader(f))
        by_id = {row.get("run_id"): row for row in records}
        for job in jobs:
            result = by_id.get(job.get("run_id"))
            if result:
                job["actual_dataset"] = result.get("dataset", "")
                job["trained_tokens"] = str(result.get("trained_tokens", 0))
                job["status"] = result.get("status", "error")
                if not job.get("status_reason"):
                    job["status_reason"] = "implementation_smoke_only_pg19_unavailable"
            elif job.get("run_id") == "b_longformer_w1024_s17":
                job["status"] = "needs_rerun"
                job["status_reason"] = "interrupted_after_time_gate"
            elif job.get("run_id") == "b_longformer_w2048_s17":
                job["status"] = "not_run"
                job["status_reason"] = "time_gate_and_8gb_vram_risk"
        fields = list(jobs[0]) if jobs else []
        with jobs_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(jobs)
    print(json.dumps({
        "records": len(records),
        "ok": summary["completed_ok"],
        "full_budget": summary["completed_full_budget"],
        "summary": str(OUT / "pilot_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
