from datetime import UTC, datetime

import pytest
import requests
import responses

from ingestion.github_client import GitHubClient, RateLimited
from tests.conftest import make_raw_issue

URL = "https://api.github.com/repos/o/r/issues"
T0 = datetime(2026, 1, 10, tzinfo=UTC)
NO_SLEEP = lambda s: None  # noqa: E731


def page(n0, n):
    return [make_raw_issue(i, T0) for i in range(n0, n0 + n)]


@responses.activate
def test_follows_link_pagination_and_sends_since():
    responses.get(URL, json=page(1, 2), headers={"ETag": '"e1"', "Link": f'<{URL}?page=2>; rel="next"'})
    responses.get(f"{URL}?page=2", json=page(3, 1))
    res = GitHubClient(token="t", sleep=NO_SLEEP).fetch_issues_since("o/r", "2026-01-01T00:00:00Z")
    assert [len(p) for p in res.pages] == [2, 1]
    assert res.etag == '"e1"' and not res.has_more
    first = responses.calls[0].request
    assert "since=2026-01-01T00%3A00%3A00Z" in first.url
    assert "sort=updated" in first.url and "direction=asc" in first.url and "state=all" in first.url
    assert first.headers["Authorization"] == "Bearer t"
    assert "If-None-Match" not in first.headers


@responses.activate
def test_304_is_not_modified_and_not_billable():
    responses.get(URL, status=304)
    client = GitHubClient(sleep=NO_SLEEP)
    res = client.fetch_issues_since("o/r", "2026-01-01T00:00:00Z", etag='"e1"')
    assert res.not_modified and res.pages == [] and res.etag == '"e1"'
    assert responses.calls[0].request.headers["If-None-Match"] == '"e1"'
    assert client.stats.as_dict() | {} == {
        "calls": 1,
        "not_modified": 1,
        "billable_calls": 0,
        "retries": 0,
        "rate_remaining": None,
    }


@responses.activate
def test_max_pages_sets_has_more():
    responses.get(URL, json=page(1, 1), headers={"Link": f'<{URL}?page=2>; rel="next"'})
    res = GitHubClient(sleep=NO_SLEEP).fetch_issues_since("o/r", "x", max_pages=1)
    assert res.has_more and res.stopped_reason == "max_pages" and len(responses.calls) == 1


@responses.activate
def test_stops_when_rate_budget_low():
    responses.get(URL, json=page(1, 1), headers={"Link": f'<{URL}?page=2>; rel="next"', "X-RateLimit-Remaining": "3"})
    client = GitHubClient(sleep=NO_SLEEP, min_rate_remaining=10)
    res = client.fetch_issues_since("o/r", "x")
    assert res.has_more and res.stopped_reason == "rate_budget"
    assert client.stats.rate_remaining == 3


@responses.activate
def test_retries_5xx_then_succeeds():
    responses.get(URL, status=502)
    responses.get(URL, json=page(1, 1))
    slept = []
    client = GitHubClient(sleep=slept.append)
    res = client.fetch_issues_since("o/r", "x")
    assert len(res.items) == 1 and client.stats.retries == 1 and slept == [1]


@responses.activate
def test_secondary_rate_limit_short_retry_after_is_retried():
    responses.get(URL, status=403, headers={"Retry-After": "2"})
    responses.get(URL, json=page(1, 1))
    slept = []
    res = GitHubClient(sleep=slept.append).fetch_issues_since("o/r", "x")
    assert len(res.items) == 1 and slept == [2.0]


@responses.activate
def test_long_rate_limit_raises_instead_of_sleeping():
    responses.get(URL, status=403, headers={"Retry-After": "3600"})
    with pytest.raises(RateLimited):
        GitHubClient(sleep=NO_SLEEP).fetch_issues_since("o/r", "x")


@responses.activate
def test_403_without_retry_hint_raises_rate_limited():
    responses.get(URL, status=403, json={"message": "forbidden"})
    with pytest.raises(RateLimited):
        GitHubClient(sleep=NO_SLEEP).fetch_issues_since("o/r", "x")


@responses.activate
def test_404_raises_http_error():
    responses.get(URL, status=404, json={"message": "Not Found"})
    with pytest.raises(requests.HTTPError):
        GitHubClient(sleep=NO_SLEEP, max_retries=0).fetch_issues_since("o/r", "x")


@responses.activate
def test_non_list_payload_fails_loudly():
    # The legacy app iterated over error dicts like {"message": ...} as if they were issues.
    responses.get(URL, json={"message": "Moved Permanently"})
    with pytest.raises(ValueError, match="expected a JSON list"):
        GitHubClient(sleep=NO_SLEEP).fetch_issues_since("o/r", "x")
