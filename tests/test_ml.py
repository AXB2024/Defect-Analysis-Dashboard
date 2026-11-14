from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.evaluate import classification_metrics, improvement_pct, majority_baseline, time_split
from ml.features import NUMERIC, build_features
from ml.labels import (
    BUCKETS,
    bucket_of,
    bucket_target,
    eligible_for_bucket,
    eligible_for_regression,
    regression_target,
)
from tests.conftest import make_raw_issue
from transform.clean import clean_issues

AS_OF = pd.Timestamp("2026-06-01", tz="UTC")


@pytest.mark.parametrize(
    "days, expected",
    [
        (0.0, "<1d"),
        (0.999, "<1d"),
        (1.0, "1-7d"),  # edges are left-closed
        (6.99, "1-7d"),
        (7.0, "7-30d"),
        (29.99, "7-30d"),
        (30.0, ">30d"),
        (400.0, ">30d"),
        (np.nan, None),
        (None, None),
        (-1.0, None),  # corrupt (closed before created) never gets a label
    ],
)
def test_bucket_edges(days, expected):
    assert bucket_of(days) == expected


def _frame(rows):
    """rows: (created_days_before_as_of, resolution_days or None)."""
    items = []
    for i, (age, res) in enumerate(rows, start=1):
        created = (AS_OF - pd.Timedelta(days=age)).to_pydatetime()
        closed = created + timedelta(days=res) if res is not None else None
        items.append(make_raw_issue(i, created, closed))
    return clean_issues(items, "o/r", AS_OF.to_pydatetime())


def test_bucket_target_handles_censoring():
    df = _frame([(100, 0.5), (100, 3), (100, 10), (100, 45), (100, None), (10, None), (40, 2)])
    y = bucket_target(df, AS_OF)
    assert y.tolist()[:5] == ["<1d", "1-7d", "7-30d", ">30d", ">30d"]  # open & 100d old -> certainly >30d
    assert pd.isna(y.iloc[5])  # open & only 10d old -> unknown
    assert y.iloc[6] == "1-7d"
    assert eligible_for_bucket(df, AS_OF).tolist() == [True] * 5 + [False, True]


def test_regression_target_caps_and_censors():
    df = _frame([(200, 2.0), (200, 150.0), (200, None), (50, None), (50, 1.0)])
    y = regression_target(df, AS_OF, cap=90)
    assert y.iloc[0] == 2.0 and y.iloc[1] == 90 and y.iloc[2] == 90
    assert pd.isna(y.iloc[3])  # open, younger than the cap -> unknown
    assert y.iloc[4] == 1.0
    assert eligible_for_regression(df, AS_OF, cap=90).tolist() == [True, True, True, False, False]


def test_time_split_is_chronological_with_no_overlap():
    s = pd.Series(pd.date_range("2026-01-01", periods=100, freq="D", tz="UTC")).sample(frac=1, random_state=0)
    tr, te, cutoff = time_split(s, test_frac=0.2)
    assert len(tr) == 80 and len(te) == 20
    assert s.loc[tr].max() < cutoff <= s.loc[te].min()
    assert set(tr).isdisjoint(te)
    with pytest.raises(ValueError):
        time_split(s, test_frac=1.0)


def test_baselines_and_metrics():
    majority, pred = majority_baseline(pd.Series(["a", "a", "b"]), 4)
    assert majority == "a" and list(pred) == ["a"] * 4
    m = classification_metrics(["a", "b", "b", "a"], ["a", "a", "b", "a"], ["a", "b"])
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"] == [[2, 0], [1, 1]]
    assert m["per_class"]["b"]["recall"] == 0.5
    assert improvement_pct(10.0, 7.5) == 25.0
    assert np.isnan(improvement_pct(0.0, 1.0))


def test_features_use_only_creation_time_signals():
    body = "<!-- template -->\nSteps to reproduce\n```\ncode\n```\nTraceback (most recent call last)\n see https://x.y ![img](a.png)"
    df = clean_issues(
        [make_raw_issue(1, datetime(2026, 1, 5, 14, tzinfo=UTC), title="Why?", body=body, labels=["bug"])],
        "o/r",
        datetime(2026, 2, 1, tzinfo=UTC),
    )
    X = build_features(df)
    row = X.iloc[0]
    assert row["has_repro"] == 1 and row["has_stack_trace"] == 1 and row["n_code_blocks"] == 1
    assert row["n_links"] == 1 and row["n_images"] == 1 and row["title_has_question"] == 1
    assert row["created_hour"] == 14 and row["created_dow"] == 0
    assert "template" not in row["text"] and "URL" in row["text"]
    assert not {"num_labels", "has_bug_label", "num_comments"} & set(NUMERIC)  # leaky features not default
    assert BUCKETS == ["<1d", "1-7d", "7-30d", ">30d"]
