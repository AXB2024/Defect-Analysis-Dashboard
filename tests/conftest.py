from __future__ import annotations

import os
import random
from datetime import UTC, datetime, timedelta

import boto3
import pytest
from moto import mock_aws

from common.storage import LocalStore, S3Store

BUCKET = "test-lake"


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch):
    # Never let a test touch real AWS or real GitHub credentials.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("DATA_URI", raising=False)


@pytest.fixture
def s3_store():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield S3Store(BUCKET, client=client)


@pytest.fixture
def local_store(tmp_path):
    return LocalStore(tmp_path / "lake")


def iso(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_raw_issue(
    number: int,
    created: datetime,
    closed: datetime | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
    body: str = "",
    updated: datetime | None = None,
    pr: bool = False,
    author_association: str = "NONE",
) -> dict:
    """Shape matches GET /repos/{owner}/{repo}/issues items."""
    item = {
        "id": 1_000_000 + number,
        "number": number,
        "title": title if title is not None else f"Issue {number}",
        "body": body,
        "state": "closed" if closed else "open",
        "state_reason": "completed" if closed else None,
        "author_association": author_association,
        "user": {"login": f"user{number % 7}"},
        "labels": [{"name": n} for n in (labels or [])],
        "comments": number % 5,
        "created_at": iso(created),
        "updated_at": iso(updated or closed or created),
        "closed_at": iso(closed) if closed else None,
    }
    if pr:
        item["pull_request"] = {"url": "https://api.github.com/..."}
    return item


TOPIC_WORDS = {
    "UI": ("ui", "button layout theme icon menu render misaligned sidebar panel"),
    "Performance": ("perf", "slow memory freeze cpu lag startup leak hang"),
    "Integration": ("terminal", "ssh remote container extension git wsl proxy"),
    "AI/Chat": ("chat", "agent copilot model prompt mcp response session"),
}


def synthetic_issues(n: int = 800, now: datetime | None = None, seed: int = 7) -> list[dict]:
    """Issues spread over ~300 days where text predicts both category and speed of
    resolution, so trained models must beat the baselines."""
    rng = random.Random(seed)
    now = now or datetime(2026, 11, 1, tzinfo=UTC)
    items = []
    cats = list(TOPIC_WORDS)
    for i in range(n):
        cat = cats[i % len(cats)]
        label, words = TOPIC_WORDS[cat]
        words = words.split()
        created = now - timedelta(days=rng.uniform(1, 300), hours=rng.uniform(0, 23))
        fast = rng.random() < 0.5
        text = " ".join(rng.choice(words) for _ in range(12)) + (" typo trivial quick" if fast else " crash complex")
        days = rng.uniform(0.05, 0.9) if fast else rng.uniform(8, 120)
        closed = created + timedelta(days=days)
        if closed > now:
            closed = None
        items.append(
            make_raw_issue(
                i + 1,
                created,
                closed,
                labels=[label] + (["bug"] if i % 3 == 0 else []),
                title=f"{' '.join(rng.choice(words) for _ in range(4))} problem",
                body=text,
                author_association=rng.choice(["NONE", "MEMBER", "CONTRIBUTOR"]),
            )
        )
    return items


@pytest.fixture
def env(monkeypatch):
    def _set(**kw):
        for k, v in kw.items():
            monkeypatch.setenv(k, v)

    return _set


os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
