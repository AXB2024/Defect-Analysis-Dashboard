"""Processed-zone writer: idempotent upsert into Hive-partitioned Parquet.

Layout:  processed/issues/repo=<owner__name>/created_month=YYYY-MM/part-0.parquet

An issue's created_month never changes, so every version of an issue lands in the same
partition and an upsert is: read partition -> concat -> keep latest updated_at -> rewrite.
Re-running the same batch is a no-op (idempotent), which makes Lambda retries safe.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common.config import PROCESSED_PREFIX, QUARANTINE_PREFIX, repo_slug
from common.storage import Store
from transform.clean import SCHEMA, TS_COLS, conform, dedupe_latest

PART_FILE = "part-0.parquet"


def partition_key(repo: str, created_month: str) -> str:
    return f"{PROCESSED_PREFIX}/repo={repo_slug(repo)}/created_month={created_month}/{PART_FILE}"


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Parquet with microsecond UTC timestamps (Athena/Glue can't read ns timestamps)."""
    df = df.copy()
    for c in TS_COLS:
        df[c] = df[c].dt.tz_convert("UTC").astype("datetime64[us, UTC]")
    for c in ("issue_id", "number", "num_comments"):
        df[c] = df[c].astype("Int64")
    table = pa.Table.from_pandas(df[list(SCHEMA)], preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


def read_partition(store: Store, key: str) -> pd.DataFrame | None:
    raw = store.get_bytes(key)
    if raw is None:
        return None
    df = pd.read_parquet(io.BytesIO(raw))
    df["label_names"] = df["label_names"].apply(lambda x: list(x) if x is not None else [])
    return conform(df)


def upsert(store: Store, df: pd.DataFrame) -> dict:
    """Merge `df` into the processed zone. Returns per-run write statistics."""
    stats = {"partitions_written": 0, "rows_inserted": 0, "rows_updated": 0, "rows_unchanged": 0}
    if df.empty:
        return stats
    for (repo, month), batch in df.groupby(["repo", "created_month"], sort=True):
        key = partition_key(repo, month)
        existing = read_partition(store, key)
        if existing is None or existing.empty:
            merged = dedupe_latest(batch)
            stats["rows_inserted"] += len(merged)
        else:
            known = existing.set_index("issue_id")["updated_at"]
            incoming = dedupe_latest(batch)
            prev = incoming["issue_id"].map(known)
            stats["rows_inserted"] += int(prev.isna().sum())
            stats["rows_updated"] += int((prev.notna() & (incoming["updated_at"] > prev)).sum())
            stats["rows_unchanged"] += int((prev.notna() & (incoming["updated_at"] <= prev)).sum())
            merged = dedupe_latest(pd.concat([existing, incoming], ignore_index=True))
        store.put_bytes(key, to_parquet_bytes(merged), "application/vnd.apache.parquet")
        stats["partitions_written"] += 1
    return stats


def write_quarantine(store: Store, bad: pd.DataFrame, run_id: str, now: datetime) -> str | None:
    if bad is None or bad.empty:
        return None
    key = f"{QUARANTINE_PREFIX}/date={now:%Y-%m-%d}/{run_id}.json"
    store.put_bytes(key, bad.to_json(orient="records", date_format="iso").encode(), "application/json")
    return key


def read_processed(store: Store) -> pd.DataFrame:
    """Read the full processed zone (used by the trainer and the dashboard)."""
    keys = [k for k in store.list_keys(PROCESSED_PREFIX + "/") if k.endswith(".parquet")]
    frames = [read_partition(store, k) for k in keys]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        from transform.clean import empty_frame

        return empty_frame()
    return pd.concat(frames, ignore_index=True)
