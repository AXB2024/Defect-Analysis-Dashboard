"""Central configuration. Everything is overridable by environment variable so the
same code runs in Lambda, GitHub Actions, Streamlit Cloud and locally."""

from __future__ import annotations

import os

DEFAULT_REPOS = ("microsoft/vscode", "react/react")  # facebook/react now 301-redirects here

# Lake layout (keys are relative to the store root)
RAW_PREFIX = "raw/issues"
PROCESSED_PREFIX = "processed/issues"
QUARANTINE_PREFIX = "quarantine/issues"
STATE_PREFIX = "state"
RUNLOG_PREFIX = "logs/runs"
MODELS_PREFIX = "models"


def repos() -> list[str]:
    raw = os.environ.get("REPOS")
    if not raw:
        return list(DEFAULT_REPOS)
    return [r.strip() for r in raw.split(",") if r.strip()]


def repo_slug(repo: str) -> str:
    """'microsoft/vscode' -> 'microsoft__vscode' (safe as a Hive partition value)."""
    return repo.replace("/", "__")


def backfill_days() -> int:
    return int(os.environ.get("BACKFILL_DAYS", "365"))


def max_pages_per_run() -> int:
    return int(os.environ.get("MAX_PAGES_PER_RUN", "40"))
