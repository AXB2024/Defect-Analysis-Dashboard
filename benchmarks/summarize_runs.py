"""Aggregate run logs: API calls per run (billable vs 304) and creation->queryable latency.

python -m benchmarks.summarize_runs --store data --since 2026-11-14T20:00:00+00:00
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from common.runlog import read_run_logs
from common.storage import open_store

ROOT = Path(__file__).resolve().parents[1]


def summarize(runs: list[dict]) -> dict:
    calls = [r["github"]["calls"] for r in runs]
    billable = [r["github"]["billable_calls"] for r in runs]
    n304 = [r["github"]["not_modified"] for r in runs]
    lags = [x for r in runs for d in r["repos"].values() for x in (d.get("latency") or {}).get("lags_s", [])]
    per_repo_304 = sum(1 for r in runs for d in r["repos"].values() if d.get("not_modified"))
    per_repo_req = sum(len(r["repos"]) for r in runs)
    out = {
        "runs": len(runs),
        "first_run": runs[0]["started_at"] if runs else None,
        "last_run": runs[-1]["started_at"] if runs else None,
        "statuses": {s: sum(1 for r in runs if r["status"] == s) for s in {r["status"] for r in runs}},
        "api_calls_per_run": {"mean": statistics.mean(calls), "min": min(calls), "max": max(calls)} if calls else None,
        "billable_calls_per_run": {"mean": statistics.mean(billable), "min": min(billable), "max": max(billable)}
        if billable
        else None,
        "not_modified_total": sum(n304),
        "repo_polls_answered_304": f"{per_repo_304}/{per_repo_req}",
        "records_ingested_total": sum(r["records_ingested"] for r in runs),
        "run_duration_s": {
            "median": statistics.median(r["duration_s"] for r in runs),
            "max": max(r["duration_s"] for r in runs),
        }
        if runs
        else None,
    }
    if lags:
        a = np.array(lags)
        out["latency_created_to_queryable_s"] = {
            "n_new_issues": len(lags),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
            "mean": float(a.mean()),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--store", default="data")
    p.add_argument("--since", help="only runs started at/after this ISO timestamp")
    p.add_argument("--out", default=str(ROOT / "benchmarks/results/scheduled_runs.json"))
    args = p.parse_args()
    runs = read_run_logs(open_store(args.store))
    if args.since:
        runs = [r for r in runs if r["started_at"] >= args.since]
    s = summarize(runs)
    Path(args.out).write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
