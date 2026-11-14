from datetime import UTC, datetime, timedelta

import pandas as pd

from tests.conftest import make_raw_issue
from transform.clean import BODY_MAX_CHARS, SCHEMA, clean_issues, dedupe_latest, empty_frame

T0 = datetime(2026, 1, 10, 12, tzinfo=UTC)
NOW = datetime(2026, 3, 1, tzinfo=UTC)


def test_drops_pull_requests_and_conforms_schema():
    items = [make_raw_issue(1, T0), make_raw_issue(2, T0, pr=True)]
    df = clean_issues(items, "o/r", NOW)
    assert list(df.columns) == list(SCHEMA)
    assert df["number"].tolist() == [1]
    assert str(df["created_at"].dtype).endswith("UTC]")


def test_resolution_days_and_derived_columns():
    items = [
        make_raw_issue(1, T0, closed=T0 + timedelta(days=2, hours=12), labels=["terminal", "bug"]),
        make_raw_issue(2, T0, labels=["feature-request"]),
    ]
    df = clean_issues(items, "microsoft/vscode", NOW).set_index("number")
    assert df.loc[1, "resolution_days"] == 2.5
    assert pd.isna(df.loc[2, "resolution_days"])
    assert df.loc[1, "category"] == "Integration"
    assert df.loc[1, "severity"] == "High"
    assert df.loc[1, "module"] == "vscode"
    assert df.loc[1, "created_month"] == "2026-01"
    assert df.loc[1, "label_names"] == ["terminal", "bug"]
    assert df.loc[1, "first_ingested_at"] == pd.Timestamp(NOW)


def test_body_truncated_and_nulls_tolerated():
    item = make_raw_issue(1, T0, body="x" * (BODY_MAX_CHARS + 100))
    item["user"] = None
    item["labels"] = None
    df = clean_issues([item], "o/r", NOW)
    assert len(df.loc[0, "body"]) == BODY_MAX_CHARS
    assert df.loc[0, "user_login"] is pd.NA or pd.isna(df.loc[0, "user_login"])
    assert df.loc[0, "label_names"] == []


def test_naive_ingest_time_is_treated_as_utc():
    df = clean_issues([make_raw_issue(1, T0)], "o/r", datetime(2026, 3, 1))
    assert df.loc[0, "last_ingested_at"] == pd.Timestamp("2026-03-01", tz="UTC")


def test_empty_input_gives_typed_empty_frame():
    df = clean_issues([make_raw_issue(1, T0, pr=True)], "o/r", NOW)
    assert df.empty and list(df.columns) == list(SCHEMA)
    assert list(empty_frame().columns) == list(SCHEMA)


def test_dedupe_keeps_latest_update_but_earliest_first_seen():
    v1 = clean_issues([make_raw_issue(1, T0, updated=T0)], "o/r", NOW)
    v2 = clean_issues(
        [make_raw_issue(1, T0, updated=T0 + timedelta(days=1), title="edited")], "o/r", NOW + timedelta(hours=1)
    )
    out = dedupe_latest(pd.concat([v2, v1], ignore_index=True))
    assert len(out) == 1
    assert out.loc[0, "title"] == "edited"
    assert out.loc[0, "first_ingested_at"] == pd.Timestamp(NOW)
    assert out.loc[0, "last_ingested_at"] == pd.Timestamp(NOW + timedelta(hours=1))
