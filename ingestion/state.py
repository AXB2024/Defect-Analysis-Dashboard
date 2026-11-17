"""Per-repo ingestion state (watermark + ETag), stored next to the data in the lake."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from common.config import STATE_PREFIX, repo_slug
from common.storage import Store


@dataclass
class RepoState:
    repo: str
    watermark: str | None = None  # max updated_at seen (ISO-8601, UTC)
    etag: str | None = None  # ETag for page 1 of the query with since=etag_since
    etag_since: str | None = None
    last_success_at: str | None = None
    caught_up: bool = False  # False while the initial backfill is still paging
    # Set while a page-capped read is in progress: GitHub's cursor URL for the next page,
    # the `since` that read started from, and the newest updated_at seen so far.
    next_url: str | None = None
    cursor_since: str | None = None
    cursor_max_updated: str | None = None


def _key(repo: str) -> str:
    return f"{STATE_PREFIX}/github/{repo_slug(repo)}.json"


def load_state(store: Store, repo: str) -> RepoState:
    data = store.get_json(_key(repo))
    if not data:
        return RepoState(repo=repo)
    known = {f for f in RepoState.__dataclass_fields__}
    return RepoState(**{k: v for k, v in data.items() if k in known})


def save_state(store: Store, state: RepoState) -> None:
    store.put_json(_key(state.repo), asdict(state))
