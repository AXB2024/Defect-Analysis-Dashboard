"""Minimal GitHub REST client for incremental issue ingestion.

Key behaviours:
* `since` + `sort=updated&direction=asc` so each run resumes from a watermark and a
  capped run (max_pages) can pick up exactly where it stopped.
* Conditional requests: page 1 is sent with `If-None-Match`. A 304 response does not
  count against the primary rate limit.
* Follows `Link: rel="next"` pagination and HTTP redirects (repos get renamed).
* Backs off on 5xx / secondary rate limits, and stops early (instead of crashing)
  when the primary rate limit is nearly exhausted.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

log = logging.getLogger(__name__)

API = "https://api.github.com"


class RateLimited(Exception):
    pass


@dataclass
class CallStats:
    calls: int = 0  # every HTTP request made
    not_modified: int = 0  # 304s (free w.r.t. the rate limit)
    retries: int = 0
    rate_remaining: int | None = None

    @property
    def billable_calls(self) -> int:
        return self.calls - self.not_modified

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "not_modified": self.not_modified,
            "billable_calls": self.billable_calls,
            "retries": self.retries,
            "rate_remaining": self.rate_remaining,
        }


@dataclass
class FetchResult:
    pages: list[list[dict]] = field(default_factory=list)
    not_modified: bool = False
    etag: str | None = None  # ETag of page 1 (for the next conditional request)
    has_more: bool = False  # stopped because of max_pages / rate-limit budget
    stopped_reason: str | None = None

    @property
    def items(self) -> list[dict]:
        return [it for page in self.pages for it in page]


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        session: requests.Session | None = None,
        api: str = API,
        max_retries: int = 3,
        min_rate_remaining: int = 25,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 30.0,
    ):
        self.api = api.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "defect-analysis-pipeline",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.max_retries = max_retries
        self.min_rate_remaining = min_rate_remaining
        self.sleep = sleep
        self.timeout = timeout
        self.stats = CallStats()

    def _get(self, url: str, params: dict | None = None, etag: str | None = None) -> requests.Response:
        headers = {"If-None-Match": etag} if etag else {}
        for attempt in range(self.max_retries + 1):
            self.stats.calls += 1
            resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            remaining = resp.headers.get("X-RateLimit-Remaining")
            if remaining is not None:
                self.stats.rate_remaining = int(remaining)

            if resp.status_code == 304:
                self.stats.not_modified += 1
                return resp
            if resp.status_code == 200:
                return resp

            retry_after = _retry_after_seconds(resp)
            retryable = resp.status_code >= 500 or (resp.status_code in (403, 429) and retry_after is not None)
            if resp.status_code in (403, 429) and retry_after is not None and retry_after > 60:
                raise RateLimited(f"rate limited for {retry_after:.0f}s on {url}")
            if not retryable or attempt == self.max_retries:
                if resp.status_code in (403, 429):
                    raise RateLimited(f"HTTP {resp.status_code} on {url}: {resp.text[:200]}")
                resp.raise_for_status()
            self.stats.retries += 1
            self.sleep(retry_after if retry_after is not None else 2**attempt)
        raise RuntimeError("unreachable")

    def fetch_issues_since(
        self,
        repo: str,
        since_iso: str,
        etag: str | None = None,
        max_pages: int = 40,
        per_page: int = 100,
    ) -> FetchResult:
        """Fetch issues (and PRs, which the endpoint mixes in) updated at/after `since_iso`,
        oldest-update first."""
        url = f"{self.api}/repos/{repo}/issues"
        params = {
            "state": "all",
            "sort": "updated",
            "direction": "asc",
            "since": since_iso,
            "per_page": per_page,
        }
        result = FetchResult()
        resp = self._get(url, params=params, etag=etag)
        if resp.status_code == 304:
            result.not_modified = True
            result.etag = etag
            return result
        result.etag = resp.headers.get("ETag")
        result.pages.append(_json_list(resp))

        next_url = resp.links.get("next", {}).get("url")
        while next_url:
            if len(result.pages) >= max_pages:
                result.has_more, result.stopped_reason = True, "max_pages"
                break
            if self.stats.rate_remaining is not None and self.stats.rate_remaining < self.min_rate_remaining:
                result.has_more, result.stopped_reason = True, "rate_budget"
                break
            resp = self._get(next_url)
            result.pages.append(_json_list(resp))
            next_url = resp.links.get("next", {}).get("url")
        return result


def _json_list(resp: requests.Response) -> list[dict]:
    data = resp.json()
    if not isinstance(data, list):
        # The legacy app iterated over error dicts as if they were issues; fail loudly.
        raise ValueError(f"expected a JSON list from {resp.url}, got {type(data).__name__}: {str(data)[:200]}")
    return data


def _retry_after_seconds(resp: requests.Response) -> float | None:
    if "Retry-After" in resp.headers:
        try:
            return float(resp.headers["Retry-After"])
        except ValueError:
            return None
    if resp.headers.get("X-RateLimit-Remaining") == "0" and "X-RateLimit-Reset" in resp.headers:
        return max(0.0, float(resp.headers["X-RateLimit-Reset"]) - time.time())
    return None
