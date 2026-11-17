"""End-to-end ingestion runs with GitHub mocked by `responses` and S3/SSM mocked by moto."""

from datetime import UTC, datetime, timedelta

import boto3
import pytest
import responses

from common.runlog import latest_run, read_run_logs
from ingestion import handler
from ingestion.github_client import GitHubClient
from ingestion.state import load_state
from tests.conftest import BUCKET, iso, make_raw_issue
from transform.lake import read_processed

URL = "https://api.github.com/repos/o/r/issues"
NOW = datetime(2026, 3, 1, 12, tzinfo=UTC)
T0 = NOW - timedelta(days=3)


def github_mock(rsps, items_by_call, etag='"v1"'):
    """Serve `items_by_call[i]` on the i-th 200; honour If-None-Match with a 304."""
    calls = {"n": 0}

    def cb(request):
        if request.headers.get("If-None-Match") == etag:
            return 304, {}, ""
        i = min(calls["n"], len(items_by_call) - 1)
        calls["n"] += 1
        import json

        return 200, {"ETag": etag, "X-RateLimit-Remaining": "4999"}, json.dumps(items_by_call[i])

    rsps.add_callback(responses.GET, URL, callback=cb, content_type="application/json")


def client():
    return GitHubClient(token="t", sleep=lambda s: None)


def test_backfill_then_incremental_then_304(s3_store):
    items = [
        make_raw_issue(1, T0, closed=T0 + timedelta(days=1)),
        make_raw_issue(2, T0 + timedelta(hours=5)),
        make_raw_issue(3, T0, pr=True),
    ]
    with responses.RequestsMock() as rsps:
        github_mock(rsps, [items])

        run1 = handler.run_ingestion(s3_store, client(), repos=["o/r"], now=NOW, backfill_days=30)
        r = run1["repos"]["o/r"]
        assert run1["status"] == "success"
        assert r["items_fetched"] == 3 and r["prs_skipped"] == 1 and r["records_ingested"] == 2
        assert r["since"] == iso(NOW - timedelta(days=30))
        assert r["quality"]["passed"] is True
        assert len([k for k in s3_store.list_keys("raw/") if k.endswith(".json.gz")]) == 1
        assert len(read_processed(s3_store)) == 2

        st = load_state(s3_store, "o/r")
        assert st.watermark == max(i["updated_at"] for i in items) and st.caught_up and st.etag == '"v1"'

        # Run 2: watermark moved, so `since` differs and no conditional header can be used.
        run2 = handler.run_ingestion(s3_store, client(), repos=["o/r"], now=NOW + timedelta(minutes=15))
        r2 = run2["repos"]["o/r"]
        assert r2["conditional_request"] is False
        assert r2["since"] == iso(datetime.fromisoformat(st.watermark.replace("Z", "+00:00")) - timedelta(minutes=5))
        assert r2["upsert"]["rows_unchanged"] == 2 and r2["upsert"]["rows_inserted"] == 0

        # Run 3: same since + stored ETag -> 304, zero billable calls, nothing written.
        raw_before = len(s3_store.list_keys("raw/"))
        run3 = handler.run_ingestion(s3_store, client(), repos=["o/r"], now=NOW + timedelta(minutes=30))
        r3 = run3["repos"]["o/r"]
        assert r3["conditional_request"] and r3["not_modified"]
        assert run3["github"]["billable_calls"] == 0 and run3["github"]["not_modified"] == 1
        assert len(s3_store.list_keys("raw/")) == raw_before

    logs = read_run_logs(s3_store)
    assert [log["run_id"] for log in logs] == sorted([run1["run_id"], run2["run_id"], run3["run_id"]])
    assert latest_run(s3_store)["run_id"] == run3["run_id"]


def test_latency_measured_for_issues_created_after_previous_run(local_store):
    old = make_raw_issue(1, T0)
    fresh = make_raw_issue(2, NOW + timedelta(minutes=10))
    with responses.RequestsMock() as rsps:
        github_mock(rsps, [[old], [old, fresh]], etag='"x"')
        handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW)
        run = handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW + timedelta(minutes=15))
    lat = run["repos"]["o/r"]["latency"]
    assert lat["new_issues"] == 1
    assert lat["p50_s"] > 0  # visible_at (wall clock) - created_at


def test_dq_failures_are_quarantined_not_written(local_store):
    bad = make_raw_issue(1, T0, closed=T0 - timedelta(days=2))
    good = make_raw_issue(2, T0)
    with responses.RequestsMock() as rsps:
        github_mock(rsps, [[bad, good]])
        run = handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW)
    r = run["repos"]["o/r"]
    assert r["records_ingested"] == 1 and r["quality"]["rows_quarantined"] == 1
    assert local_store.get_json(r["quarantine_key"])[0]["number"] == 1
    assert read_processed(local_store)["number"].tolist() == [2]


def test_one_repo_failing_is_partial_and_does_not_advance_its_state(local_store):
    with responses.RequestsMock() as rsps:
        github_mock(rsps, [[make_raw_issue(1, T0)]])
        rsps.get("https://api.github.com/repos/bad/repo/issues", status=500)
        run = handler.run_ingestion(local_store, client(), repos=["o/r", "bad/repo"], now=NOW)
    assert run["status"] == "partial"
    assert run["failures"][0]["repo"] == "bad/repo" and "HTTPError" in run["failures"][0]["error"]
    assert load_state(local_store, "bad/repo").watermark is None
    assert load_state(local_store, "o/r").watermark is not None


def test_capped_backfill_resumes_from_github_cursor(local_store):
    p1 = [make_raw_issue(1, T0, updated=T0)]
    p2 = [make_raw_issue(2, T0, updated=T0 + timedelta(hours=1))]
    nxt = f"{URL}?after=CURSOR&page=2"
    with responses.RequestsMock() as rsps:
        rsps.get(URL, json=p1, headers={"Link": f'<{nxt}>; rel="next"', "ETag": '"a"'})
        run = handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW, max_pages=1)
    assert run["repos"]["o/r"]["has_more"]
    st = load_state(local_store, "o/r")
    assert not st.caught_up and st.watermark is None  # watermark untouched while capped
    assert st.next_url == nxt and st.cursor_since == run["repos"]["o/r"]["since"]
    with responses.RequestsMock() as rsps:
        rsps.get(URL, json=p2, match=[responses.matchers.query_param_matcher({"after": "CURSOR", "page": "2"})])
        run2 = handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW, max_pages=1)
        assert rsps.calls[0].request.url == nxt  # resumed at the exact cursor
    assert run2["repos"]["o/r"]["resumed_from_cursor"]
    st = load_state(local_store, "o/r")
    assert st.caught_up and st.next_url is None and st.watermark == p2[0]["updated_at"]
    assert sorted(read_processed(local_store)["number"].tolist()) == [1, 2]


def test_expired_cursor_restarts_from_same_since_without_skipping(local_store):
    nxt = f"{URL}?after=OLD&page=2"
    with responses.RequestsMock() as rsps:
        rsps.get(URL, json=[make_raw_issue(1, T0)], headers={"Link": f'<{nxt}>; rel="next"'})
        first = handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW, max_pages=1)
    since = first["repos"]["o/r"]["since"]
    with responses.RequestsMock() as rsps:
        rsps.get(
            URL,
            status=422,
            json={"message": "cursor expired"},
            match=[responses.matchers.query_param_matcher({"after": "OLD", "page": "2"})],
        )
        rsps.get(URL, json=[make_raw_issue(1, T0), make_raw_issue(2, T0)])
        run = handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW + timedelta(days=1), max_pages=1)
    r = run["repos"]["o/r"]
    assert r["cursor_expired"] and r["since"] == since  # same since, not "now - backfill_days"
    assert sorted(read_processed(local_store)["number"].tolist()) == [1, 2]


def test_lambda_handler_reads_token_from_ssm(s3_store, env):
    ssm = boto3.client("ssm", region_name="us-east-1")
    ssm.put_parameter(Name="/dd/token", Value="secret-token", Type="SecureString")
    env(DATA_URI=f"s3://{BUCKET}", REPOS="o/r", GITHUB_TOKEN_PARAM="/dd/token")
    handler._TOKEN_CACHE.clear()
    with responses.RequestsMock() as rsps:
        github_mock(rsps, [[make_raw_issue(1, T0)]])
        out = handler.lambda_handler({}, None)
        assert rsps.calls[0].request.headers["Authorization"] == "Bearer secret-token"
    assert out["status"] == "success" and out["records_ingested"] == 1


def test_lambda_handler_raises_when_every_repo_fails(s3_store, env):
    env(DATA_URI=f"s3://{BUCKET}", REPOS="o/r", GITHUB_TOKEN="t")
    with responses.RequestsMock() as rsps:
        rsps.get(URL, status=404)
        with pytest.raises(RuntimeError, match="failed for all repos"):
            handler.lambda_handler({}, None)
    assert latest_run(s3_store)["status"] == "failed"  # run log still written


def test_summarise_flattens_run():
    run = {"repos": {"o/r": {"pages": 1, "quality": {"x": 1}}}}
    df = handler.summarise(run)
    assert df.columns.tolist() == ["repo", "pages"]


def test_latency_ignores_issues_refetched_by_lookback(local_store):
    # Issue 2 is created after run 1 *started* but run 1 already fetched it; run 2's
    # 5-minute lookback re-fetches it. It must not be counted as a new arrival in run 2.
    early = make_raw_issue(2, NOW + timedelta(seconds=1))
    with responses.RequestsMock() as rsps:
        github_mock(rsps, [[make_raw_issue(1, T0)], [make_raw_issue(1, T0), early]], etag='"y"')
        handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW - timedelta(minutes=15))
        r1 = handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW)
        assert r1["repos"]["o/r"]["latency"]["new_issues"] == 1
    with responses.RequestsMock() as rsps:
        github_mock(rsps, [[early]], etag='"z"')
        r2 = handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW + timedelta(minutes=15))
    assert r2["repos"]["o/r"]["latency"] == {}


def test_emf_metrics_reports_failed_repos():
    import json

    line = json.loads(handler.emf_metrics({"failures": [{"repo": "a/b", "error": "x"}], "records_ingested": 7}, "fn"))
    spec = line["_aws"]["CloudWatchMetrics"][0]
    assert spec["Namespace"] == "DefectDashboard" and spec["Dimensions"] == [["FunctionName"]]
    assert {m["Name"] for m in spec["Metrics"]} == {"FailedRepos", "RecordsIngested"}
    assert line["FailedRepos"] == 1 and line["RecordsIngested"] == 7 and line["FunctionName"] == "fn"


def test_lambda_handler_emits_metric_on_partial_failure(s3_store, env, capsys):
    import json

    env(DATA_URI=f"s3://{BUCKET}", REPOS="o/r,bad/repo", GITHUB_TOKEN="t")
    with responses.RequestsMock() as rsps:
        github_mock(rsps, [[make_raw_issue(1, T0)]])
        rsps.get("https://api.github.com/repos/bad/repo/issues", status=500)
        out = handler.lambda_handler({}, None)  # partial: no exception...
    assert out["status"] == "partial"
    emf = [json.loads(x) for x in capsys.readouterr().out.splitlines() if '"_aws"' in x]
    assert emf and emf[-1]["FailedRepos"] == 1  # ...but the metric still lets the alarm fire


# --------------------------------------------------------------------------- watermark / out-of-order
def _it(n, updated):
    return make_raw_issue(n, T0 - timedelta(days=30), updated=updated)


def test_next_watermark_only_advances_when_stream_fully_read():
    items = [_it(1, T0), _it(2, T0 + timedelta(days=90)), _it(3, T0 + timedelta(minutes=1))]
    # capped run: a stray newer item must NOT move the watermark
    assert handler.next_watermark(items, has_more=True, current=iso(T0 - timedelta(days=1))) == iso(
        T0 - timedelta(days=1)
    )
    # fully read: newest updated_at is safe, including what earlier capped runs saw
    assert handler.next_watermark(items, has_more=False, current=None) == iso(T0 + timedelta(days=90))
    assert handler.next_watermark([], has_more=False, current=None, seen_max=iso(T0)) == iso(T0)


def test_next_watermark_never_moves_backwards():
    older = [_it(1, T0 - timedelta(days=1))]
    assert handler.next_watermark(older, has_more=False, current=iso(T0)) == iso(T0)
    assert handler.next_watermark([], has_more=False, current=iso(T0)) == iso(T0)


def _fake_github(rsps, stream):
    """Serve `stream` [(sort_key, item)] like GitHub: filter by since on the sort key,
    order by the sort key (which can disagree with item updated_at), 1 item per page."""
    import json
    from urllib.parse import parse_qs, urlparse

    def cb(request):
        q = parse_qs(urlparse(request.url).query)
        since = q["since"][0]
        page = int(q.get("page", ["1"])[0])
        rows = [it for key, it in sorted(stream, key=lambda kv: kv[0]) if key >= since]
        chunk = rows[page - 1 : page]
        headers = {"ETag": f'"{since}-{page}"'}
        if page < len(rows):
            headers["Link"] = (
                f'<{URL}?state=all&sort=updated&direction=asc&since={since}&per_page=100&page={page + 1}>; rel="next"'
            )
        return 200, headers, json.dumps(chunk)

    rsps.add_callback(responses.GET, URL, callback=cb, content_type="application/json")


def test_capped_backfill_with_out_of_order_item_ingests_everything(local_store):
    """Regression: max(updated_at) after a capped run jumped past unread issues."""
    base = NOW - timedelta(days=10)
    stray = _it(2, NOW - timedelta(hours=1))  # sort key says day 0, updated_at says today
    stream = [(iso(base), _it(1, base)), (iso(base + timedelta(seconds=1)), stray)]
    stream += [(iso(base + timedelta(days=d)), _it(10 + d, base + timedelta(days=d))) for d in range(1, 6)]
    with responses.RequestsMock() as rsps:
        _fake_github(rsps, stream)
        for _ in range(20):
            run = handler.run_ingestion(local_store, client(), repos=["o/r"], now=NOW, max_pages=2, backfill_days=30)
            if not run["repos"]["o/r"]["has_more"]:
                break
    got = sorted(read_processed(local_store)["number"].tolist())
    assert got == [1, 2, 10 + 1, 10 + 2, 10 + 3, 10 + 4, 10 + 5]
