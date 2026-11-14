"""Time-based splitting, baselines and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


def time_split(created_at: pd.Series, test_frac: float = 0.2) -> tuple[pd.Index, pd.Index, pd.Timestamp]:
    """Train on the oldest (1-test_frac) of issues, test on the newest. No shuffling, so no
    future information leaks into training. Ties at the cutoff go to test."""
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be in (0, 1)")
    order = created_at.sort_values(kind="stable")
    cutoff = order.iloc[int(len(order) * (1 - test_frac))]
    train = order.index[order < cutoff]
    test = order.index[order >= cutoff]
    return train, test, cutoff


def period(s: pd.Series) -> list[str]:
    return [str(s.min().date()), str(s.max().date())] if len(s) else [None, None]


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def improvement_pct(baseline: float, model: float) -> float:
    return float((baseline - model) / baseline * 100.0) if baseline else float("nan")


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),  # rows=true, cols=pred
        "per_class": {
            lbl: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lbl in enumerate(labels)
        },
    }


def majority_baseline(y_train: pd.Series, n: int) -> tuple[str, np.ndarray]:
    majority = y_train.value_counts().idxmax()
    return majority, np.full(n, majority, dtype=object)
