"""Dashboard tabs. Pure rendering over already-loaded DataFrames / dicts."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from ml.labels import BUCKETS

WIDE = {"width": "stretch"}


# ----------------------------------------------------------------------------- filters
def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    has_pred = "predicted_category" in df.columns
    mode = st.sidebar.radio(
        "Category source",
        ["Label rules", "NLP model"] if has_pred else ["Label rules"],
        help="Label rules map area labels to a category; the NLP model predicts it from title+body "
        "(useful for untriaged issues that have no area label yet).",
    )
    df = df.assign(category_view=df["predicted_category"] if mode == "NLP model" else df["category"])

    def multi(label: str, col: str) -> list:
        opts = sorted(df[col].dropna().astype(str).unique())
        return st.sidebar.multiselect(label, opts, default=opts)

    cats, sevs, repos = multi("Category", "category_view"), multi("Severity", "severity"), multi("Repo", "repo")
    states = st.sidebar.multiselect("State", ["open", "closed"], default=["open", "closed"])
    lo, hi = df["created_at"].min().date(), df["created_at"].max().date()
    rng = st.sidebar.date_input("Created between", [lo, hi], min_value=lo, max_value=hi)
    start, end = (rng[0], rng[1]) if isinstance(rng, list | tuple) and len(rng) == 2 else (lo, hi)
    created = df["created_at"].dt.date
    mask = (
        df["category_view"].astype(str).isin(cats)
        & df["severity"].astype(str).isin(sevs)
        & df["repo"].astype(str).isin(repos)
        & df["state"].astype(str).isin(states)
        & (created >= start)
        & (created <= end)
    )
    return df[mask]


# ----------------------------------------------------------------------------- overview
def overview(df: pd.DataFrame) -> None:
    closed = df[df["closed_at"].notna()]
    k = st.columns(5)
    k[0].metric("Issues", f"{len(df):,}")
    k[1].metric("Open", f"{(df['state'] == 'open').sum():,}")
    k[2].metric("Median resolution (closed)", f"{closed['resolution_days'].median():.1f} d" if len(closed) else "-")
    k[3].metric("Closed within 1 day", f"{(closed['resolution_days'] < 1).mean():.0%}" if len(closed) else "-")
    k[4].metric("Critical", f"{(df['severity'] == 'Critical').sum():,}")

    c1, c2 = st.columns(2)
    with c1:
        sev = df.groupby(["severity", "repo"]).size().reset_index(name="issues")
        st.plotly_chart(px.bar(sev, x="severity", y="issues", color="repo", title="Issues by severity"), **WIDE)
    with c2:
        weekly = (
            pd.concat(
                [
                    df.set_index("created_at").resample("W").size().rename("opened"),
                    closed.set_index("closed_at").resample("W").size().rename("closed"),
                ],
                axis=1,
            )
            .fillna(0)
            .reset_index(names="week")
            .melt("week", var_name="event", value_name="issues")
        )
        st.plotly_chart(px.line(weekly, x="week", y="issues", color="event", title="Opened vs closed per week"), **WIDE)

    heat = df.groupby(["category_view", "module"]).size().reset_index(name="issues")
    st.plotly_chart(
        px.density_heatmap(
            heat, x="category_view", y="module", z="issues", color_continuous_scale="Reds", title="Category x module"
        ),
        **WIDE,
    )
    if len(closed):
        res = closed.assign(days=closed["resolution_days"].clip(upper=120))
        st.plotly_chart(
            px.histogram(res, x="days", color="repo", nbins=60, title="Resolution time (days, clipped at 120)"), **WIDE
        )

    st.subheader("Issues")
    cols = ["repo", "number", "title", "state", "category", "severity", "created_at", "closed_at", "resolution_days"]
    page_size = 200
    pages = max(1, -(-len(df) // page_size))
    page = st.number_input("Page", 1, pages, 1)
    st.dataframe(
        df.sort_values("created_at", ascending=False)[cols].iloc[(page - 1) * page_size : page * page_size], **WIDE
    )


# ----------------------------------------------------------------------------- predictions
def predictions(df: pd.DataFrame, metrics: dict) -> None:
    if "predicted_bucket" not in df.columns:
        st.info("No trained model found. Run `python -m ml.train`.")
        return
    open_df = df[df["state"] == "open"].copy()
    st.markdown(
        "Predictions for **currently open** issues, using only information available when the issue was opened "
        "(title, body, author association, repo, time). "
        f"Bucket model test macro-F1: **{metrics['bucket']['test']['logreg']['macro_f1']:.3f}** "
        f"vs majority baseline {metrics['bucket']['test']['baseline_majority']['macro_f1']:.3f} - treat as a triage hint, "
        "not a promise. See *Model metrics*."
    )
    if open_df.empty:
        st.write("No open issues match the filters.")
        return
    dist = open_df["predicted_bucket"].value_counts().reindex(BUCKETS, fill_value=0).reset_index()
    dist.columns = ["bucket", "issues"]
    st.plotly_chart(px.bar(dist, x="bucket", y="issues", title="Open issues by predicted resolution bucket"), **WIDE)
    show = open_df.assign(age_days=(datetime.now(UTC) - open_df["created_at"]).dt.total_seconds() / 86400)
    cols = ["repo", "number", "title", "age_days", "predicted_bucket", "bucket_confidence", "predicted_days"]
    cols += ["predicted_category"] if "predicted_category" in show.columns else []
    st.dataframe(
        show.sort_values("created_at", ascending=False)[cols].round(
            {"age_days": 1, "bucket_confidence": 2, "predicted_days": 1}
        ),
        **WIDE,
    )


# ----------------------------------------------------------------------------- pipeline health
def _ts(s: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(s).tz_convert("UTC") if s else None


def pipeline_health(df: pd.DataFrame, runs: list[dict], latest: dict | None) -> None:
    now = pd.Timestamp.now(tz="UTC")
    if not latest:
        st.info("No pipeline runs recorded yet.")
        return
    last_end = _ts(latest["finished_at"])
    lag_min = (now - last_end).total_seconds() / 60
    status = "🟢 fresh" if lag_min <= 20 else ("🟡 late" if lag_min <= 60 else "🔴 stale")
    k = st.columns(5)
    k[0].metric("Last ingest (UTC)", f"{last_end:%Y-%m-%d %H:%M}")
    k[1].metric("Pipeline lag", f"{lag_min:.0f} min", help="now - last successful ingestion run; schedule is 15 min")
    k[2].metric("Status", status)
    k[3].metric("Newest issue update in lake", f"{(now - df['updated_at'].max()).total_seconds() / 60:.0f} min ago")
    k[4].metric("Last run status", latest.get("status", "?"))

    st.subheader("Record counts")
    counts = df.groupby("repo").agg(
        issues=("issue_id", "size"),
        open=("state", lambda s: int((s == "open").sum())),
        months=("created_month", "nunique"),
        first_created=("created_at", "min"),
        last_updated=("updated_at", "max"),
    )
    st.dataframe(counts, **WIDE)

    if runs:
        hist = pd.DataFrame(
            [
                {
                    "started_at": _ts(r["started_at"]),
                    "status": r["status"],
                    "records_ingested": r.get("records_ingested", 0),
                    "duration_s": r.get("duration_s"),
                    "api_calls": r.get("github", {}).get("calls", 0),
                    "not_modified_304": r.get("github", {}).get("not_modified", 0),
                    "failures": len(r.get("failures", [])),
                }
                for r in runs
            ]
        ).sort_values("started_at")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(hist, x="started_at", y="records_ingested", color="status", title="Records ingested per run"),
                **WIDE,
            )
        with c2:
            calls = hist.melt("started_at", ["api_calls", "not_modified_304"], var_name="kind", value_name="n")
            st.plotly_chart(
                px.line(calls, x="started_at", y="n", color="kind", title="GitHub API calls per run (304s are free)"),
                **WIDE,
            )
        lat = [
            {"run": r["run_id"], "repo": repo, **d["latency"]}
            for r in runs
            for repo, d in r.get("repos", {}).items()
            if d.get("latency")
        ]
        if lat:
            st.subheader("Ingestion latency (issue created → queryable in lake)")
            st.dataframe(pd.DataFrame(lat).tail(20), **WIDE)
        st.subheader("Run history")
        st.dataframe(hist.sort_values("started_at", ascending=False), **WIDE)

    st.subheader("Data-quality checks (latest run)")
    rows = []
    for repo, d in latest.get("repos", {}).items():
        for c in d.get("quality", {}).get("checks", []):
            rows.append({"repo": repo, **c})
    if rows:
        st.dataframe(pd.DataFrame(rows), **WIDE)
    else:
        st.write("Latest run fetched nothing new (HTTP 304), so no checks ran.")
    if latest.get("failures"):
        st.error(latest["failures"])


# ----------------------------------------------------------------------------- model metrics
def _cm_figure(cm: list[list[int]], labels: list[str], title: str):
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", title=title)
    fig.update_layout(xaxis_title="predicted", yaxis_title="actual")
    return fig


def model_metrics(m: dict) -> None:
    if not m:
        st.info("No trained model found. Run `python -m ml.train`.")
        return
    st.caption(
        f"Version `{m['version']}` · trained {m['trained_at'][:19]} UTC · git `{m.get('git_sha')}` · "
        f"{m['data']['n_issues']:,} issues · time-based split (train=older, test=newer)"
    )
    r = m["regression"]
    st.subheader("Resolution time (RandomForestRegressor)")
    st.write(f"Target: {r['target']}. Test period {r['test_period'][0]} → {r['test_period'][1]} (n={r['n_test']}).")
    st.table(
        pd.DataFrame(
            {
                "median baseline": r["test"]["baseline_median"],
                "random forest": r["test"]["random_forest"],
            }
        )
        .loc[["mae", "rmse"]]
        .round(2)
    )
    st.write(
        f"MAE improvement {r['test']['mae_improvement_pct']:.1f}% · RMSE improvement {r['test']['rmse_improvement_pct']:.1f}%"
    )

    b = m["bucket"]
    st.subheader("Resolution bucket (LogisticRegression, TF-IDF + numeric)")
    bt = pd.DataFrame({k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for k, v in b["test"].items()}).round(
        3
    )
    st.table(bt)
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        _cm_figure(b["test"]["logreg"]["confusion_matrix"], b["test"]["logreg"]["labels"], "LogReg"), **WIDE
    )
    c2.plotly_chart(
        _cm_figure(
            b["test"]["baseline_majority"]["confusion_matrix"], b["test"]["baseline_majority"]["labels"], "Majority"
        ),
        **WIDE,
    )

    c = m["category"]
    st.subheader("Category (TF-IDF + LogisticRegression)")
    st.table(
        pd.DataFrame({k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for k, v in c["test"].items()}).round(3)
    )
    with st.expander("Raw metrics.json"):
        st.json(m)
