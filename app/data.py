"""Read-only data access for the dashboard. Nothing here calls GitHub or trains a model."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from common.runlog import latest_run, read_run_logs
from common.storage import Store, open_store
from ml.predict import load_latest, score
from transform.lake import read_processed

DATA_TTL_S = 300  # dashboard re-reads the lake at most every 5 min (ingest runs every 15)


def _configure_from_secrets() -> None:
    """On Streamlit Community Cloud, credentials come from st.secrets; locally from env."""
    try:
        secrets = dict(st.secrets)
    except Exception:
        return
    for k in ("DATA_URI", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
        if k in secrets and not os.environ.get(k):
            os.environ[k] = str(secrets[k])


def data_uri() -> str:
    _configure_from_secrets()
    return os.environ.get("DATA_URI", "data")


# Every cached loader takes the URI so caches are keyed by data location.
@st.cache_resource
def get_store(uri: str) -> Store:
    return open_store(uri)


@st.cache_data(ttl=DATA_TTL_S, show_spinner="Reading processed Parquet...")
def load_issues(uri: str) -> pd.DataFrame:
    return read_processed(get_store(uri))


@st.cache_resource(ttl=DATA_TTL_S, show_spinner="Loading models...")
def load_models(uri: str):
    return load_latest(get_store(uri))


@st.cache_data(ttl=DATA_TTL_S, show_spinner="Scoring issues...")
def load_scored(uri: str, model_version: str | None) -> pd.DataFrame:  # version is part of the cache key
    df = load_issues(uri)
    loaded = load_models(uri)
    if loaded is None or df.empty:
        return df
    models, _ = loaded
    return score(models, df)


@st.cache_data(ttl=60)
def load_runs(uri: str, limit: int = 300) -> list[dict]:
    return read_run_logs(get_store(uri), limit=limit)


@st.cache_data(ttl=60)
def load_latest_run(uri: str) -> dict | None:
    return latest_run(get_store(uri))
