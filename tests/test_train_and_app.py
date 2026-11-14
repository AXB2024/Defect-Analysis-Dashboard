"""Training smoke test on synthetic data + a headless render of the dashboard."""

from datetime import UTC, datetime

import pytest

from common.runlog import write_run_log
from ml.predict import load_latest, score
from ml.train import train_all
from tests.conftest import synthetic_issues
from transform.clean import clean_issues
from transform.lake import read_processed, upsert

NOW = datetime(2026, 11, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def trained_lake(tmp_path_factory):
    from common.storage import LocalStore

    store = LocalStore(tmp_path_factory.mktemp("lake"))
    upsert(store, clean_issues(synthetic_issues(800, NOW), "o/r", NOW))
    write_run_log(
        store,
        {
            "run_id": "r1",
            "started_at": NOW.isoformat(),
            "finished_at": NOW.isoformat(),
            "status": "success",
            "records_ingested": 800,
            "duration_s": 1.0,
            "github": {"calls": 8, "not_modified": 0},
            "failures": [],
            "repos": {"o/r": {"latency": {"new_issues": 1, "p50_s": 60.0}, "quality": {"checks": []}}},
        },
    )
    metrics = train_all(store, window_days=365)
    return store, metrics


def test_training_beats_baselines_on_learnable_data(trained_lake):
    store, m = trained_lake
    reg, bkt, cat = m["regression"], m["bucket"], m["category"]
    # time-based split: every test issue is newer than every train issue
    assert reg["train_period"][1] <= reg["test_period"][0]
    assert reg["test"]["random_forest"]["mae"] < reg["test"]["baseline_median"]["mae"]
    assert bkt["test"]["logreg"]["macro_f1"] > bkt["test"]["baseline_majority"]["macro_f1"]
    assert cat["test"]["tfidf_logreg"]["macro_f1"] > cat["test"]["baseline_majority"]["macro_f1"]
    assert sum(map(sum, bkt["test"]["logreg"]["confusion_matrix"])) == bkt["n_test"]


def test_artifacts_versioned_and_loadable(trained_lake):
    store, m = trained_lake
    prefix = f"models/v={m['version']}"
    assert {
        f"{prefix}/{n}" for n in ("regressor.joblib", "bucket_clf.joblib", "category_clf.joblib", "metrics.json")
    } <= set(store.list_keys("models/"))
    models, metrics = load_latest(store)
    assert metrics["version"] == m["version"]
    scored = score(models, read_processed(store).head(25))
    assert {"predicted_days", "predicted_bucket", "bucket_confidence", "predicted_category"} <= set(scored.columns)
    assert (scored["predicted_days"] >= 0).all()


def test_load_latest_none_when_untrained(local_store):
    assert load_latest(local_store) is None


def test_dashboard_renders_without_errors(trained_lake, monkeypatch):
    from streamlit.testing.v1 import AppTest

    store, _ = trained_lake
    monkeypatch.setenv("DATA_URI", store.uri)
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Predictions", "Pipeline health", "Model metrics"]
    assert any("800" in m.value for m in at.metric)


def test_dashboard_empty_lake_shows_warning(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DATA_URI", str(tmp_path / "empty"))
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60).run()
    assert not at.exception
    assert at.warning and "No processed data" in at.warning[0].value
