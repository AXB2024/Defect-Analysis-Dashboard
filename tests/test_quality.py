from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tests.conftest import make_raw_issue
from transform.clean import clean_issues
from transform.quality import SchemaError, run_checks

T0 = datetime(2026, 1, 10, tzinfo=UTC)
NOW = datetime(2026, 3, 1, tzinfo=UTC)


def _df(items):
    return clean_issues(items, "o/r", NOW)


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def test_clean_batch_passes_everything():
    rep = run_checks(_df([make_raw_issue(i, T0, closed=T0 + timedelta(days=i)) for i in range(1, 6)]))
    assert rep.passed
    assert rep.rows_in == rep.rows_out == 5
    assert rep.rows_quarantined == 0
    assert all(c.passed for c in rep.checks)


def test_missing_column_is_fatal():
    with pytest.raises(SchemaError, match="missing"):
        run_checks(_df([make_raw_issue(1, T0)]).drop(columns=["title"]))


def test_extra_column_is_fatal():
    with pytest.raises(SchemaError, match="extra"):
        run_checks(_df([make_raw_issue(1, T0)]).assign(surprise=1))


def test_wrong_dtype_is_fatal():
    df = _df([make_raw_issue(1, T0)])
    df["created_at"] = df["created_at"].astype(str)
    with pytest.raises(SchemaError, match="dtype"):
        run_checks(df)


def test_null_keys_are_quarantined():
    df = _df([make_raw_issue(1, T0), make_raw_issue(2, T0)])
    df.loc[1, "created_at"] = pd.NaT
    rep = run_checks(df)
    assert _check(rep, "not_null:created_at").failed == 1
    assert rep.rows_quarantined == 1 and rep.rows_out == 1
    assert "null_created_at" in rep.bad.iloc[0]["dq_reasons"]
    assert not rep.passed


def test_closed_before_created_is_quarantined():
    bad = make_raw_issue(1, T0, closed=T0 - timedelta(days=1))
    rep = run_checks(_df([bad, make_raw_issue(2, T0)]))
    assert _check(rep, "closed_at>=created_at").failed == 1
    assert rep.bad["number"].tolist() == [1]
    assert rep.good["number"].tolist() == [2]


def test_invalid_state_is_quarantined():
    df = _df([make_raw_issue(1, T0)])
    df.loc[0, "state"] = "merged"
    rep = run_checks(df)
    assert _check(rep, "valid_state").failed == 1 and rep.rows_out == 0


def test_duplicates_are_counted_then_resolved_to_latest():
    v1 = make_raw_issue(1, T0, updated=T0)
    v2 = make_raw_issue(1, T0, updated=T0 + timedelta(hours=3), title="newer")
    rep = run_checks(_df([v1, v2]))
    assert _check(rep, "duplicates_resolved").failed == 1
    assert _check(rep, "unique:(repo,issue_id)").passed
    assert rep.rows_out == 1 and rep.good.loc[0, "title"] == "newer"
    assert rep.passed  # duplicates are info-level, not a failure


def test_warn_checks_keep_rows():
    item = make_raw_issue(1, T0, title="")
    item["state"] = "closed"  # closed but no closed_at
    rep = run_checks(_df([item]))
    assert _check(rep, "closed_has_closed_at").failed == 1
    assert _check(rep, "non_empty_title").failed == 1
    assert rep.rows_out == 1 and rep.passed


def test_summary_is_json_serialisable():
    import json

    rep = run_checks(_df([make_raw_issue(1, T0)]))
    s = json.loads(json.dumps(rep.summary()))
    assert s["rows_in"] == 1 and s["passed"] is True
    assert {"name", "severity", "failed", "passed"} <= set(s["checks"][0])
