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
    """Add predicted_days / predicted_bucket / bucket_confidence / predicted_category."""
    if df.empty:
        return df
    X = build_features(df)
    out = df.copy()
    if "regressor" in models:
        out["predicted_days"] = models["regressor"].predict(X)
    if "bucket_clf" in models:
        proba = models["bucket_clf"].predict_proba(X)
        classes = models["bucket_clf"].classes_
        out["predicted_bucket"] = classes[proba.argmax(axis=1)]
        out["bucket_confidence"] = proba.max(axis=1)
    if "category_clf" in models:
        out["predicted_category"] = models["category_clf"].predict(X)
    return out
