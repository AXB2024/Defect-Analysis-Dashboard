"""Targets for the resolution-time models, with right-censoring handled explicitly.

An open issue has an *unknown* resolution time, but not an uninformative one: an issue
that has been open for 45 days certainly has resolution > 30 days. Dropping open issues
(as the legacy app did) biases the newest data toward quickly-closed issues, which is
exactly the time window a time-based test set uses. So each target is only defined for
issues old enough that it is fully known at the snapshot time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = ["<1d", "1-7d", "7-30d", ">30d"]
BUCKET_EDGES = [0.0, 1.0, 7.0, 30.0, np.inf]  # left-closed: [0,1) [1,7) [7,30) [30,inf)
BUCKET_HORIZON_DAYS = 30.0  # the >30d bucket is known once an issue is 30 days old
REGRESSION_CAP_DAYS = 90.0  # regression target = min(resolution, 90); known after 90 days


def bucket_of(days: float) -> str | None:
    if days is None or pd.isna(days) or days < 0:
        return None
    for name, lo, hi in zip(BUCKETS, BUCKET_EDGES[:-1], BUCKET_EDGES[1:], strict=True):
        if lo <= days < hi:
            return name
    return None


def age_days(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    return (as_of - df["created_at"]).dt.total_seconds() / 86400.0


def bucket_target(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Resolution bucket, or NaN where it is not yet knowable.

    * closed  -> bucket of its resolution time
    * open and age >= 30d -> '>30d' (censored but certain)
    * open and age < 30d  -> unknown
    Callers should additionally restrict to issues with age >= 30d so that the closed
    and open populations are sampled identically (see `eligible_for_bucket`).
    """
    closed = df["closed_at"].notna()
    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[closed] = df.loc[closed, "resolution_days"].map(bucket_of)
    old_open = ~closed & (age_days(df, as_of) >= BUCKET_HORIZON_DAYS)
    out[old_open] = ">30d"
    return out


def eligible_for_bucket(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    return age_days(df, as_of) >= BUCKET_HORIZON_DAYS


def regression_target(df: pd.DataFrame, as_of: pd.Timestamp, cap: float = REGRESSION_CAP_DAYS) -> pd.Series:
    """min(resolution_days, cap); open issues older than `cap` contribute `cap`."""
    closed = df["closed_at"].notna()
    y = pd.Series(np.nan, index=df.index, dtype="float64")
    y[closed] = df.loc[closed, "resolution_days"].clip(lower=0, upper=cap)
    y[~closed & (age_days(df, as_of) >= cap)] = cap
    return y


def eligible_for_regression(df: pd.DataFrame, as_of: pd.Timestamp, cap: float = REGRESSION_CAP_DAYS) -> pd.Series:
    return age_days(df, as_of) >= cap
