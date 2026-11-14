"""Local / CI entry point for the ingestion job (same code path as the Lambda).

python -m ingestion.run --store data                      # one incremental run
python -m ingestion.run --store data --until-caught-up    # backfill in capped runs
python -m ingestion.run --store s3://my-bucket            # against the real lake
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from common.storage import open_store
from ingestion.github_client import GitHubClient
from ingestion.handler import run_ingestion, summarise


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default=os.environ.get("DATA_URI", "data"))
    p.add_argument("--repos", help="comma-separated owner/name list (default: config.DEFAULT_REPOS)")
    p.add_argument("--max-pages", type=int, default=None, help="pages per repo per run")
    p.add_argument("--backfill-days", type=int, default=None)
    p.add_argument("--until-caught-up", action="store_true", help="repeat runs until no repo has more pages")
    p.add_argument("--max-runs", type=int, default=50)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    store = open_store(args.store)
    repos = [r.strip() for r in args.repos.split(",")] if args.repos else None
    status = "success"
    for i in range(args.max_runs if args.until_caught_up else 1):
        client = GitHubClient(token=os.environ.get("GITHUB_TOKEN") or None)
        run = run_ingestion(store, client, repos=repos, max_pages=args.max_pages, backfill_days=args.backfill_days)
        status = run["status"]
        print(summarise(run).to_string(index=False))
        print(
            json.dumps(
                {k: run[k] for k in ("run_id", "status", "records_ingested", "duration_s", "github", "failures")}
            )
        )
        more = any(r.get("has_more") for r in run["repos"].values())
        if not (args.until_caught_up and more) or status == "failed":
            break
        logging.info("run %d done, more pages pending - continuing", i + 1)
    return 0 if status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
