"""Pipeline run log: one JSON document per run plus a `latest` pointer for the dashboard."""

from __future__ import annotations

from datetime import datetime

from common.config import RUNLOG_PREFIX, STATE_PREFIX
from common.storage import Store

LATEST_RUN_KEY = f"{STATE_PREFIX}/latest_run.json"


def write_run_log(store: Store, log: dict) -> str:
    started = datetime.fromisoformat(log["started_at"])
    key = f"{RUNLOG_PREFIX}/date={started:%Y-%m-%d}/{log['run_id']}.json"
    store.put_json(key, log)
    store.put_json(LATEST_RUN_KEY, log)
    return key


def read_run_logs(store: Store, limit: int | None = None) -> list[dict]:
    keys = sorted(k for k in store.list_keys(RUNLOG_PREFIX + "/") if k.endswith(".json"))
    if limit:
        keys = keys[-limit:]
    return [d for d in (store.get_json(k) for k in keys) if d]


def latest_run(store: Store) -> dict | None:
    return store.get_json(LATEST_RUN_KEY)
