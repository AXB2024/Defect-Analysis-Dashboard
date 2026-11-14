"""Inference helpers for the dashboard: load the latest artifacts and score issues.
No training happens here."""

from __future__ import annotations

import io

import joblib
import pandas as pd

from common.config import MODELS_PREFIX
from common.storage import Store
from ml.features import build_features
from ml.models import LogTargetRegressor  # noqa: F401  (needed to unpickle the regressor)

LATEST_KEY = f"{MODELS_PREFIX}/latest.json"
PRED_COLS = ["predicted_days", "predicted_bucket", "bucket_confidence", "predicted_category"]
ARTIFACTS = ("regressor", "bucket_clf", "category_clf")


def load_latest(store: Store) -> tuple[dict, dict] | None:
    """Return ({name: model}, metrics) for the latest version, or None if never trained."""
    pointer = store.get_json(LATEST_KEY)
    if not pointer:
        return None
    prefix = pointer["prefix"]
    models = {}
    for name in ARTIFACTS:
        raw = store.get_bytes(f"{prefix}/{name}.joblib")
        if raw is not None:
            models[name] = joblib.load(io.BytesIO(raw))
    metrics = store.get_json(f"{prefix}/metrics.json") or {}
    return models, metrics


def score(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add predicted_category (all issues) and predicted_days / predicted_bucket /
    bucket_confidence (open issues only - a closed issue's resolution time is known)."""
    if df.empty:
        return df
    X = build_features(df)
    out = df.copy()
    is_open = (out["state"] == "open").to_numpy()
    Xo = X[is_open]
    if "regressor" in models:
        out["predicted_days"] = float("nan")
        if len(Xo):
            out.loc[is_open, "predicted_days"] = models["regressor"].predict(Xo)
    if "bucket_clf" in models:
        out["predicted_bucket"] = None
        out["bucket_confidence"] = float("nan")
        if len(Xo):
            proba = models["bucket_clf"].predict_proba(Xo)
            out.loc[is_open, "predicted_bucket"] = models["bucket_clf"].classes_[proba.argmax(axis=1)]
            out.loc[is_open, "bucket_confidence"] = proba.max(axis=1)
    if "category_clf" in models:
        out["predicted_category"] = models["category_clf"].predict(X)
    return out


def predictions_frame(models: dict, df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Batch-score `df` (written by the training job next to the model artifacts)."""
    scored = score(models, df)
    out = scored[["repo", "issue_id"] + [c for c in PRED_COLS if c in scored.columns]].copy()
    out["model_version"] = version
    return out


def load_predictions(store: Store, prefix: str) -> pd.DataFrame | None:
    return store.get_parquet(f"{prefix}/predictions.parquet")


def attach_predictions(models: dict, issues: pd.DataFrame, batch: pd.DataFrame | None, fetch_full) -> pd.DataFrame:
    """Join precomputed predictions; score on the fly only issues the batch hasn't seen
    (ingested since the last training run). `fetch_full(missing_df)` must return those rows
    with every column (incl. body), since `issues` may have been read without it."""
    if issues.empty:
        return issues
    if batch is None or batch.empty:
        return score(models, fetch_full(issues))
    merged = issues.merge(batch.drop(columns=["model_version"]), on=["repo", "issue_id"], how="left")
    # New since the batch run, or reopened after it (no bucket prediction for an open issue).
    missing = merged["predicted_category"].isna() | ((merged["state"] == "open") & merged["predicted_bucket"].isna())
    if missing.any():
        delta = score(models, fetch_full(merged.loc[missing, issues.columns]))
        merged = pd.concat([merged[~missing], delta], ignore_index=True)
    return merged
