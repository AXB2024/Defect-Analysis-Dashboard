"""Simulate the EventBridge schedule locally: run the ingestion job every N seconds and
record per-run API usage and creation->queryable latency (from the run logs the job writes).

    python -m benchmarks.schedule_local --store data --interval 900 --runs 8

Afterwards, `python -m benchmarks.summarize_runs --store data --since <ISO>` aggregates.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime

from common.storage import open_store
from ingestion.github_client import GitHubClient
from ingestion.handler import run_ingestion


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--store", default="data")
    p.add_argument("--interval", type=int, default=900)
    p.add_argument("--runs", type=int, default=8)
    args = p.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    store = open_store(args.store)
    print(json.dumps({"started": datetime.now(UTC).isoformat(), "interval_s": args.interval}), flush=True)
    for i in range(args.runs):
        t0 = time.monotonic()
        run = run_ingestion(store, GitHubClient(token=os.environ.get("GITHUB_TOKEN")))
        print(
            json.dumps(
                {
                    "i": i,
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "records": run["records_ingested"],
                    "duration_s": run["duration_s"],
                    "github": run["github"],
                    "latency": {r: d.get("latency") for r, d in run["repos"].items()},
                }
            ),
            flush=True,
        )
        if i < args.runs - 1:
            time.sleep(max(0, args.interval - (time.monotonic() - t0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
