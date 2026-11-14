"""raw GitHub issue JSON  ->  typed, flat, analysis-ready records."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from transform.labels import map_category, map_severity

BODY_MAX_CHARS = 5000  # raw/ keeps the full body; processed/ keeps enough for NLP

# Column order == Glue table column order (partition columns `repo_part`/`created_month` are
# derived from the S3 path in Athena but also kept in the file for direct Parquet reads).
SCHEMA: dict[str, str] = {
    "repo": "string",
    "issue_id": "int64",
    "number": "int64",
    "title": "string",
    "body": "string",
    "state": "string",
    "state_reason": "string",
    "author_association": "string",
    "user_login": "string",
    "label_names": "object",  # list[str]
    "num_comments": "int64",
    "created_at": "datetime64[ns, UTC]",
    "updated_at": "datetime64[ns, UTC]",
    "closed_at": "datetime64[ns, UTC]",
    "resolution_days": "float64",
    "category": "string",
    "severity": "string",
    "module": "string",
    "created_month": "string",
    "first_ingested_at": "datetime64[ns, UTC]",
    "last_ingested_at": "datetime64[ns, UTC]",
}

TS_COLS = ["created_at", "updated_at", "closed_at", "first_ingested_at", "last_ingested_at"]


def to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def is_pull_request(item: dict) -> bool:
    return "pull_request" in item


def clean_issues(items: list[dict], repo: str, ingested_at: datetime) -> pd.DataFrame:
    """Flatten raw issues for one repo. PRs are dropped (the /issues endpoint mixes them in)."""
    rows = []
    for it in items:
        if is_pull_request(it):
            continue
        labels = [lbl["name"] if isinstance(lbl, dict) else str(lbl) for lbl in it.get("labels") or []]
        body = it.get("body") or ""
        rows.append(
            {
                "repo": repo,
                "issue_id": it.get("id"),
                "number": it.get("number"),
                "title": (it.get("title") or "").strip(),
                "body": body[:BODY_MAX_CHARS],
                "state": it.get("state"),
                "state_reason": it.get("state_reason"),
                "author_association": it.get("author_association"),
                "user_login": (it.get("user") or {}).get("login"),
                "label_names": labels,
                "num_comments": int(it.get("comments") or 0),
                "created_at": it.get("created_at"),
                "updated_at": it.get("updated_at"),
                "closed_at": it.get("closed_at"),
            }
        )
    if not rows:
        return empty_frame()

    df = pd.DataFrame(rows)
    for c in ("created_at", "updated_at", "closed_at"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    ts = to_utc(ingested_at)
    df["first_ingested_at"] = ts
    df["last_ingested_at"] = ts
    return enrich(df)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["resolution_days"] = (df["closed_at"] - df["created_at"]).dt.total_seconds() / 86400.0
    df["category"] = df["label_names"].apply(map_category)
    df["severity"] = df["label_names"].apply(map_severity)
    df["module"] = df["repo"].str.split("/").str[-1]
    df["created_month"] = df["created_at"].dt.strftime("%Y-%m")
    return conform(df)


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with exactly SCHEMA's columns, order and dtypes."""
    out = pd.DataFrame(index=df.index)
    for col, dtype in SCHEMA.items():
        s = df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index)
        if dtype.startswith("datetime64"):
            s = pd.to_datetime(s, utc=True, errors="coerce")
        elif dtype == "int64":
            s = pd.to_numeric(s, errors="coerce").astype("Int64")
        elif dtype == "float64":
            s = pd.to_numeric(s, errors="coerce").astype("float64")
        elif dtype == "string":
            s = s.astype("string")
        out[col] = s
    return out.reset_index(drop=True)


def empty_frame() -> pd.DataFrame:
    return conform(pd.DataFrame(columns=list(SCHEMA)))


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recently updated version of each (repo, issue_id), while preserving
    the earliest first_ingested_at seen for that issue (needed for latency metrics)."""
    if df.empty:
        return df
    first_seen = df.groupby(["repo", "issue_id"])["first_ingested_at"].min()
    df = df.sort_values(["updated_at", "last_ingested_at"]).drop_duplicates(["repo", "issue_id"], keep="last")
    df = df.set_index(["repo", "issue_id"])
    df["first_ingested_at"] = first_seen.reindex(df.index)
    return df.reset_index()[list(SCHEMA)].reset_index(drop=True)
