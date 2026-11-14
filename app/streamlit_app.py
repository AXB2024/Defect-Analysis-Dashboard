"""Defect Analysis Dashboard (Streamlit). Run: DATA_URI=s3://bucket streamlit run app/streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Community Cloud puts only this file's directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Defect Analysis Dashboard", layout="wide")  # must be the first st call

from app import views  # noqa: E402
from app.data import data_uri, get_store, load_latest_run, load_models, load_runs, load_scored  # noqa: E402


def main() -> None:
    st.title("Defect Analysis Dashboard")
    uri = data_uri()
    loaded = load_models(uri)
    metrics = loaded[1] if loaded else {}
    df = load_scored(uri, metrics.get("version"))
    st.caption(
        f"Data: `{get_store(uri).uri}` · {len(df):,} issues · model `{metrics.get('version', 'none')}` · "
        "ingested every 15 min by AWS Lambda; this page only reads Parquet and saved models."
    )
    if st.sidebar.button("Refresh now"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    if df.empty:
        st.warning("No processed data yet. Run `python -m ingestion.run` (or wait for the scheduled Lambda).")
        return

    overview, predictions, health, model_tab = st.tabs(["Overview", "Predictions", "Pipeline health", "Model metrics"])
    filtered = views.sidebar_filters(df)
    with overview:
        views.overview(filtered)
    with predictions:
        views.predictions(filtered, metrics)
    with health:
        views.pipeline_health(df, load_runs(uri), load_latest_run(uri))
    with model_tab:
        views.model_metrics(metrics)


main()
