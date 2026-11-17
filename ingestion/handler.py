"""Incremental ingestion job. Runs as an AWS Lambda (EventBridge, every 15 min) or locally
via `python -m ingestion.run`.

Per repo:
  1. GET /issues?since=<watermark - lookback>&sort=updated&direction=asc  (If-None-Match)
  2. 304 -> nothing changed, done (free call)
  3. 200 -> write every page untouched to raw/ (gzipped JSON, replayable)
  4. clean -> data-quality checks -> quarantine bad rows -> upsert good rows to processed/
  5. advance the watermark only after processed/ is written (at-least-once + idempotent upsert)
Finally a run log (records, duration, API calls, DQ results, failures) is written to logs/.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

from common import config
from common.runlog import write_run_log
from common.storage import Store, open_store
from ingestion.github_client import GitHubClient
from ingestion.state import RepoState, load_state, save_state
from transform.clean import clean_issues, is_pull_request, to_utc
from transform.lake import upsert, write_quarantine
from transform.quality import run_checks

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

LOOKBACK = timedelta(minutes=5)  # re-read a small overlap; the upsert dedupes it


def _iso(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def since_for(state: RepoState, now: datetime, backfill_days: int) -> str:
    if state.watermark:
        wm = datetime.fromisoformat(state.watermark.replace("Z", "+00:00"))
        return _iso(wm - LOOKBACK)
    return _iso(now - timedelta(days=backfill_days))


def next_watermark(items: list[dict], has_more: bool, current: str | None, seen_max: str | None = None) -> str | None:
    """Where the next fresh (non-cursor) read should start.

    GitHub's `sort=updated&direction=asc` listing is not strictly ordered by `updated_at`:
    an item can sit among others updated months earlier (seen live: #307470, updated
    2026-11-03, between items updated 2026-04-30). The old rule - max(updated_at) after
    every run - therefore jumped the watermark past everything a page-capped run had not
    read yet, silently dropping months of issues.

    So the watermark only advances once the stream has been read to its end (no `next`
    link); a capped run keeps it and resumes from GitHub's cursor instead. When the end is
    reached, everything updated before the newest `updated_at` seen has been read, so that
    is a safe resume point. It never moves backwards.
    """
    if has_more:
        return current
    candidates = [it["updated_at"] for it in items] + ([seen_max] if seen_max else [])
    if not candidates:
        return current
    newest = max(candidates)
    return newest if not current or newest > current else current


def ingest_repo(
    store: Store, client: GitHubClient, repo: str, run_id: str, now: datetime, max_pages: int, backfill_days: int
) -> dict:
    t0 = time.perf_counter()
    state = load_state(store, repo)
    resuming = bool(state.next_url)
    since = state.cursor_since if resuming and state.cursor_since else since_for(state, now, backfill_days)
    etag = None if resuming else (state.etag if state.etag_since == since else None)
    calls_before = client.stats.calls

    cursor_expired = False
    if resuming:
        try:
            res = client.fetch_issues_since(repo, since, max_pages=max_pages, resume_url=state.next_url)
        except requests.HTTPError as e:
            # Expired/invalid cursor: restart from the same `since` (re-reads, never skips).
            log.warning("cursor for %s rejected (%s); restarting from since=%s", repo, e, since)
            cursor_expired, resuming = True, False
            state.next_url, state.cursor_max_updated = None, None
            res = client.fetch_issues_since(repo, since, max_pages=max_pages)
    else:
        res = client.fetch_issues_since(repo, since, etag=etag, max_pages=max_pages)
    out: dict = {
        "since": since,
        "conditional_request": etag is not None,
        "resumed_from_cursor": resuming,
        "cursor_expired": cursor_expired,
        "not_modified": res.not_modified,
        "pages": len(res.pages),
        "has_more": res.has_more,
        "stopped_reason": res.stopped_reason,
        "watermark_before": state.watermark,
    }

    if res.not_modified:
        state.last_success_at = _iso(now)
        state.caught_up = True
        save_state(store, state)
        out.update(
            items_fetched=0,
            issues=0,
            prs_skipped=0,
            records_ingested=0,
            api_calls=client.stats.calls - calls_before,
            duration_s=round(time.perf_counter() - t0, 3),
            watermark_after=state.watermark,
        )
        return out

    # 1. raw zone: every page, untouched
    slug = config.repo_slug(repo)
    for i, page in enumerate(res.pages):
        store.put_json(f"{config.RAW_PREFIX}/repo={slug}/ingest_date={now:%Y-%m-%d}/{run_id}-p{i:04d}.json.gz", page)

    items = res.items
    df = clean_issues(items, repo, now)
    report = run_checks(df)
    quarantine_key = write_quarantine(store, report.bad, f"{run_id}-{slug}", now)
    new_ids: set = set()
    write_stats = upsert(store, report.good, inserted_ids=new_ids)
    visible_at = datetime.now(UTC)

    # Ingestion latency: for issues first written by THIS run and created after the previous
    # successful run started, (visible_at - created_at) is their end-to-end lag. Both conditions
    # matter: the 5-min lookback re-fetches issues an earlier run already ingested.
    latency = {}
    if state.caught_up and state.last_success_at and not report.good.empty:
        prev = to_utc(state.last_success_at)
        good = report.good
        fresh = good[(good["created_at"] > prev) & good["issue_id"].isin(new_ids)]
        if not fresh.empty:
            lag = (to_utc(visible_at) - fresh["created_at"]).dt.total_seconds()
            latency = {
                "new_issues": int(len(lag)),
                "p50_s": round(float(lag.median()), 1),
                "p95_s": round(float(lag.quantile(0.95)), 1),
                "max_s": round(float(lag.max()), 1),
                "lags_s": [round(float(x), 1) for x in lag],
            }

    # 2. advance state only after processed/ is durable
    page_max = max((it["updated_at"] for it in items), default=None)
    seen_max = max(filter(None, [state.cursor_max_updated, page_max]), default=None)
    if res.has_more:  # capped: remember exactly where to continue
        state.next_url, state.cursor_since, state.cursor_max_updated = res.next_url, since, seen_max
    else:  # stream read to its end: safe to advance
        state.watermark = next_watermark(items, False, state.watermark, seen_max)
        state.next_url = state.cursor_since = state.cursor_max_updated = None
    state.etag, state.etag_since = res.etag, since
    state.last_success_at = _iso(now)
    state.caught_up = not res.has_more
    save_state(store, state)

    out.update(
        items_fetched=len(items),
        prs_skipped=sum(1 for it in items if is_pull_request(it)),
        issues=len(df),
        records_ingested=report.rows_out,
        quality=report.summary(),
        quarantine_key=quarantine_key,
        upsert=write_stats,
        latency=latency,
        api_calls=client.stats.calls - calls_before,
        duration_s=round(time.perf_counter() - t0, 3),
        watermark_after=state.watermark,
    )
    return out


def run_ingestion(
    store: Store,
    client: GitHubClient,
    repos: list[str] | None = None,
    now: datetime | None = None,
    max_pages: int | None = None,
    backfill_days: int | None = None,
) -> dict:
    now = now or datetime.now(UTC)
    repos = repos or config.repos()
    max_pages = max_pages or config.max_pages_per_run()
    backfill_days = backfill_days or config.backfill_days()
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    t0 = time.perf_counter()

    run = {"run_id": run_id, "started_at": now.isoformat(), "repos": {}, "failures": []}
    for repo in repos:
        try:
            run["repos"][repo] = ingest_repo(store, client, repo, run_id, now, max_pages, backfill_days)
        except Exception as e:  # one repo failing must not block the others
            log.exception("ingestion failed for %s", repo)
            run["failures"].append({"repo": repo, "error": f"{type(e).__name__}: {e}"[:500]})

    run["finished_at"] = datetime.now(UTC).isoformat()
    run["duration_s"] = round(time.perf_counter() - t0, 3)
    run["records_ingested"] = sum(r.get("records_ingested", 0) for r in run["repos"].values())
    run["github"] = client.stats.as_dict()
    run["status"] = "failed" if len(run["failures"]) == len(repos) else ("partial" if run["failures"] else "success")
    run["log_key"] = write_run_log(store, run)
    return run


# ---------------------------------------------------------------- Lambda entry point
_TOKEN_CACHE: dict[str, str] = {}


def _github_token() -> str | None:
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]
    param = os.environ.get("GITHUB_TOKEN_PARAM")
    if not param:
        return None
    if param not in _TOKEN_CACHE:
        import boto3

        _TOKEN_CACHE[param] = boto3.client("ssm").get_parameter(Name=param, WithDecryption=True)["Parameter"]["Value"]
    return _TOKEN_CACHE[param]


def emf_metrics(run: dict, function_name: str) -> str:
    """CloudWatch Embedded Metric Format line: CloudWatch turns it into metrics on ingest."""
    return json.dumps(
        {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": "DefectDashboard",
                        "Dimensions": [["FunctionName"]],
                        "Metrics": [
                            {"Name": "FailedRepos", "Unit": "Count"},
                            {"Name": "RecordsIngested", "Unit": "Count"},
                        ],
                    }
                ],
            },
            "FunctionName": function_name,
            "FailedRepos": len(run["failures"]),
            "RecordsIngested": run["records_ingested"],
        }
    )


def lambda_handler(event, context):  # noqa: ARG001 - Lambda signature
    store = open_store(os.environ["DATA_URI"])
    run = run_ingestion(store, GitHubClient(token=_github_token()))
    log.info(
        "run %s status=%s records=%s calls=%s", run["run_id"], run["status"], run["records_ingested"], run["github"]
    )
    print(emf_metrics(run, os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "local")), flush=True)
    if run["status"] == "failed":
        # Surface as a Lambda error so the CloudWatch alarm fires.
        raise RuntimeError(f"ingestion failed for all repos: {run['failures']}")
    return {k: run[k] for k in ("run_id", "status", "records_ingested", "duration_s", "github", "failures")}


def summarise(run: dict) -> pd.DataFrame:
    rows = [{"repo": r, **{k: v for k, v in d.items() if not isinstance(v, dict)}} for r, d in run["repos"].items()]
    return pd.DataFrame(rows)
