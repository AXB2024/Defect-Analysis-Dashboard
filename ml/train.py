"""Offline training job (runs in GitHub Actions on a schedule, or locally).

    python -m ml.train --store data            # local lake
    python -m ml.train --store s3://bucket     # real lake

Reads processed Parquet, trains three models with a time-based split, evaluates each
against a naive baseline on the *same* held-out (newest) issues, refits on all eligible
data, and writes versioned artifacts:

    models/v=<UTC timestamp>/{regressor,bucket_clf,category_clf}.joblib
    models/v=<UTC timestamp>/metrics.json
    models/latest.json   (pointer read by the dashboard)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common.config import MODELS_PREFIX
from common.storage import Store, open_store
from ml.evaluate import (
    classification_metrics,
    improvement_pct,
    majority_baseline,
    period,
    regression_metrics,
    time_split,
)
from ml.features import CATEGORICAL, LEAKY_NUMERIC, NUMERIC, build_features, preprocessor
from ml.labels import (
    BUCKETS,
    REGRESSION_CAP_DAYS,
    bucket_target,
    eligible_for_bucket,
    eligible_for_regression,
    regression_target,
)
from ml.models import LogTargetRegressor, SafeSVD
from transform.labels import OTHER
from transform.lake import read_processed

log = logging.getLogger("ml.train")
SEED = 42
LATEST_KEY = f"{MODELS_PREFIX}/latest.json"


# ----------------------------------------------------------------------------- models
def make_regressor(leaky: bool = False) -> Pipeline:
    numeric = NUMERIC + (LEAKY_NUMERIC if leaky else [])
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            (
                "text",
                Pipeline(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(max_features=20000, min_df=3, sublinear_tf=True, stop_words="english"),
                        ),
                        ("svd", SafeSVD(n_components=64, random_state=SEED)),
                    ]
                ),
                "text",
            ),
        ]
    )
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=SEED)
    return Pipeline([("pre", pre), ("rf", rf)])


def make_classifier(class_weight=None, leaky: bool = False, C: float = 1.0) -> Pipeline:
    clf = LogisticRegression(max_iter=2000, C=C, class_weight=class_weight, random_state=SEED)
    return Pipeline([("pre", preprocessor(text=True, leaky=leaky)), ("clf", clf)])


def make_category_classifier(class_weight=None) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(
                    max_features=20000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english"
                ),
                "text",
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["repo"]),
        ]
    )
    clf = LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


# ----------------------------------------------------------------------------- tasks
def _val_split(idx_train: pd.Index, created: pd.Series, frac: float = 0.2):
    tr, va, _ = time_split(created.loc[idx_train], test_frac=frac)
    return tr, va


def train_regression(X: pd.DataFrame, df: pd.DataFrame, as_of: pd.Timestamp) -> tuple[object, dict]:
    elig = eligible_for_regression(df, as_of)
    y = regression_target(df, as_of)[elig]
    Xe, created = X[elig], df.loc[elig, "created_at"]
    tr, te, cutoff = time_split(created)
    if len(tr) < 50 or len(te) < 20:
        raise ValueError(f"not enough regression data: train={len(tr)} test={len(te)}")

    # Model selection on a validation slice carved from the *end* of the training window.
    vtr, vva = _val_split(tr, created)
    val = {}
    for name, factory in {
        "raw_days": make_regressor,
        "log1p_days": lambda: LogTargetRegressor(make_regressor()),
    }.items():
        m = factory().fit(Xe.loc[vtr], y.loc[vtr])
        val[name] = regression_metrics(y.loc[vva], m.predict(Xe.loc[vva]))
    chosen = min(val, key=lambda k: val[k]["mae"])
    factory = (
        (lambda leaky=False: make_regressor(leaky))
        if chosen == "raw_days"
        else (lambda leaky=False: LogTargetRegressor(make_regressor(leaky)))
    )

    model = factory().fit(Xe.loc[tr], y.loc[tr])
    pred = model.predict(Xe.loc[te])
    median = float(y.loc[tr].median())
    base = regression_metrics(y.loc[te], np.full(len(te), median))
    mod = regression_metrics(y.loc[te], pred)

    leaky_model = factory(leaky=True).fit(Xe.loc[tr], y.loc[tr])
    leaky = regression_metrics(y.loc[te], leaky_model.predict(Xe.loc[te]))

    metrics = {
        "target": f"min(resolution_days, {REGRESSION_CAP_DAYS:g}); open issues older than {REGRESSION_CAP_DAYS:g}d = {REGRESSION_CAP_DAYS:g}",
        "population": f"issues created >= {REGRESSION_CAP_DAYS:g} days before snapshot",
        "n_train": len(tr),
        "n_test": len(te),
        "train_period": period(created.loc[tr]),
        "test_period": period(created.loc[te]),
        "cutoff": str(cutoff),
        "validation": val,
        "chosen_variant": chosen,
        "test": {
            "baseline_median": base | {"value_days": median},
            "random_forest": mod,
            "mae_improvement_pct": improvement_pct(base["mae"], mod["mae"]),
            "rmse_improvement_pct": improvement_pct(base["rmse"], mod["rmse"]),
        },
        "ablation_with_leaky_features": leaky
        | {"features_added": LEAKY_NUMERIC, "note": "post-hoc features; NOT used by the served model"},
        "test_target_distribution": {
            "mean": float(y.loc[te].mean()),
            "median": float(y.loc[te].median()),
            "share_at_cap": float((y.loc[te] >= REGRESSION_CAP_DAYS).mean()),
        },
    }
    final = factory().fit(Xe, y)  # refit on all eligible data for serving
    return final, metrics


def train_bucket(X: pd.DataFrame, df: pd.DataFrame, as_of: pd.Timestamp) -> tuple[object, dict]:
    elig = eligible_for_bucket(df, as_of)
    y = bucket_target(df, as_of)[elig].dropna().astype(str)
    Xe, created = X.loc[y.index], df.loc[y.index, "created_at"]
    tr, te, cutoff = time_split(created)
    vtr, vva = _val_split(tr, created)

    val = {}
    for cw in (None, "balanced"):
        m = make_classifier(class_weight=cw).fit(Xe.loc[vtr], y.loc[vtr])
        val[str(cw)] = classification_metrics(y.loc[vva], m.predict(Xe.loc[vva]), BUCKETS)["macro_f1"]
    chosen = max(val, key=val.get)
    cw = None if chosen == "None" else chosen

    model = make_classifier(class_weight=cw).fit(Xe.loc[tr], y.loc[tr])
    majority, base_pred = majority_baseline(y.loc[tr], len(te))
    leaky = make_classifier(class_weight=cw, leaky=True).fit(Xe.loc[tr], y.loc[tr])
    lm = classification_metrics(y.loc[te], leaky.predict(Xe.loc[te]), BUCKETS)

    metrics = {
        "population": "issues created >= 30 days before snapshot (bucket fully known)",
        "n_train": len(tr),
        "n_test": len(te),
        "train_period": period(created.loc[tr]),
        "test_period": period(created.loc[te]),
        "cutoff": str(cutoff),
        "validation_macro_f1_by_class_weight": val,
        "chosen_class_weight": chosen,
        "train_distribution": y.loc[tr].value_counts().reindex(BUCKETS, fill_value=0).astype(int).to_dict(),
        "test_distribution": y.loc[te].value_counts().reindex(BUCKETS, fill_value=0).astype(int).to_dict(),
        "test": {
            "baseline_majority": classification_metrics(y.loc[te], base_pred, BUCKETS) | {"majority_class": majority},
            "logreg": classification_metrics(y.loc[te], model.predict(Xe.loc[te]), BUCKETS),
        },
        "ablation_with_leaky_features": {"accuracy": lm["accuracy"], "macro_f1": lm["macro_f1"]},
    }
    final = make_classifier(class_weight=cw).fit(Xe, y)
    return final, metrics


def train_category(X: pd.DataFrame, df: pd.DataFrame) -> tuple[object, dict]:
    mask = df["category"].astype(str) != OTHER
    y = df.loc[mask, "category"].astype(str)
    Xe, created = X.loc[y.index], df.loc[y.index, "created_at"]
    tr, te, cutoff = time_split(created)
    labels = sorted(y.unique())
    vtr, vva = _val_split(tr, created)
    val = {}
    for cw in (None, "balanced"):
        m = make_category_classifier(cw).fit(Xe.loc[vtr], y.loc[vtr])
        val[str(cw)] = classification_metrics(y.loc[vva], m.predict(Xe.loc[vva]), labels)["macro_f1"]
    chosen = max(val, key=val.get)
    cw = None if chosen == "None" else chosen

    model = make_category_classifier(cw).fit(Xe.loc[tr], y.loc[tr])
    majority, base_pred = majority_baseline(y.loc[tr], len(te))
    metrics = {
        "population": "issues whose labels map to a category (Other excluded); weak labels from transform.labels",
        "n_train": len(tr),
        "n_test": len(te),
        "train_period": period(created.loc[tr]),
        "test_period": period(created.loc[te]),
        "cutoff": str(cutoff),
        "validation_macro_f1_by_class_weight": val,
        "chosen_class_weight": chosen,
        "test_distribution": y.loc[te].value_counts().astype(int).to_dict(),
        "test": {
            "baseline_majority": classification_metrics(y.loc[te], base_pred, labels) | {"majority_class": majority},
            "tfidf_logreg": classification_metrics(y.loc[te], model.predict(Xe.loc[te]), labels),
        },
    }
    final = make_category_classifier(cw).fit(Xe, y)
    return final, metrics


# ----------------------------------------------------------------------------- orchestration
def _git_sha() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def _dump(obj) -> bytes:
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    return buf.getvalue()


def train_all(store: Store, window_days: int = 365, as_of: pd.Timestamp | None = None) -> dict:
    t0 = time.perf_counter()
    df = read_processed(store)
    if df.empty:
        raise ValueError(f"no processed data in {store.uri}")
    as_of = as_of or df["last_ingested_at"].max()  # the lake's snapshot time
    # Issues created before the backfill window only exist in the lake if they were
    # updated recently - a biased sample - so the population is bounded to the window.
    df = df[df["created_at"] >= as_of - pd.Timedelta(days=window_days)].reset_index(drop=True)
    X = build_features(df)
    log.info("training on %d issues (as_of=%s)", len(df), as_of)

    reg, reg_m = train_regression(X, df, as_of)
    bkt, bkt_m = train_bucket(X, df, as_of)
    cat, cat_m = train_category(X, df)

    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    metrics = {
        "version": version,
        "trained_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "sklearn_version": sklearn.__version__,
        "data": {
            "store": store.uri,
            "as_of": str(as_of),
            "window_days": window_days,
            "n_issues": len(df),
            "by_repo": df["repo"].value_counts().to_dict(),
            "created_period": period(df["created_at"]),
        },
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL, "text": "tfidf(title x2 + body)"},
        "regression": reg_m,
        "bucket": bkt_m,
        "category": cat_m,
        "duration_s": round(time.perf_counter() - t0, 2),
    }
    prefix = f"{MODELS_PREFIX}/v={version}"
    store.put_bytes(f"{prefix}/regressor.joblib", _dump(reg))
    store.put_bytes(f"{prefix}/bucket_clf.joblib", _dump(bkt))
    store.put_bytes(f"{prefix}/category_clf.joblib", _dump(cat))
    store.put_json(f"{prefix}/metrics.json", metrics)
    store.put_json(LATEST_KEY, {"version": version, "prefix": prefix, "trained_at": metrics["trained_at"]})
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--store", default=os.environ.get("DATA_URI", "data"))
    p.add_argument("--window-days", type=int, default=365)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = train_all(open_store(args.store), window_days=args.window_days)
    print(json.dumps({k: m[k] for k in ("version", "duration_s", "data")}, indent=2, default=str))
    print("regression:", json.dumps(m["regression"]["test"], indent=2))
    print("bucket:", {k: {kk: v[kk] for kk in ("accuracy", "macro_f1")} for k, v in m["bucket"]["test"].items()})
    print("category:", {k: {kk: v[kk] for kk in ("accuracy", "macro_f1")} for k, v in m["category"]["test"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
