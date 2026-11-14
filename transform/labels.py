"""Rule-based mapping from GitHub label names to an analysis category and severity.

These rules produce the *weak labels* the category classifier learns from. They only
look at area labels; process labels (bug, info-needed, *duplicate, verified, stale, ...)
say nothing about which part of the product an issue concerns, so they are ignored.

Rules are checked in order; the first match wins.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

OTHER = "Other"

# (category, keywords) checked in priority order; keywords match label *tokens*
# (labels split on non-alphanumerics), so "agents-window" matches "agents".
CATEGORY_RULES: list[tuple[str, frozenset[str]]] = [
    (cat, frozenset(words.split()))
    for cat, words in [
        ("Performance", "perf performance slow memory freeze leak startup cpu"),
        ("AI/Chat", "chat agent agents copilot mcp model ai llm codex claude byok"),
        (
            "UI",
            "ui ux frontend layout menus menu theme themes icon icons workbench accessibility a11y tree quick dom css notebook editor",
        ),
        (
            "Integration",
            "integration compatibility remote containers container ssh wsl network proxy extension extensions marketplace git scm terminal debug devtools developer tasks",
        ),
        (
            "Core",
            "backend api core server compiler reconciler hooks fiber rendering languages search workspace files settings config build suspense",
        ),
    ]
]

# Labels that describe process/triage state, never area.
PROCESS_LABELS = frozenset(
    "bug feature-request info-needed verified insiders-released unreleased triage-needed invalid spam stale "
    "under-discussion *duplicate duplicate *caused-by-extension *as-designed *out-of-scope ai-generated "
    "ai-analyzed candidate engineering".split()
) | {"good first issue", "help wanted"}

_TOKEN = re.compile(r"[a-z0-9]+")


def _normalise(label: str) -> str:
    label = label.lower().strip()
    # React uses "Component: X", "Type: Bug", "Status: ...", "Resolution: ..."
    for prefix in ("component:", "type:", "status:", "resolution:"):
        if label.startswith(prefix):
            if prefix != "component:":
                return ""  # Type/Status/Resolution labels are process labels
            label = label[len(prefix) :].strip()
    return label


def map_category(label_names: Iterable[str]) -> str:
    area = [n for n in (_normalise(lbl) for lbl in label_names) if n and n not in PROCESS_LABELS]
    if not area:
        return OTHER
    tokens = [set(_TOKEN.findall(lbl)) for lbl in area]
    for category, keywords in CATEGORY_RULES:
        if any(keywords & toks for toks in tokens):
            return category
    return OTHER


CRITICAL_MARKERS = ("critical", "crash", "data-loss", "security", "important")
HIGH_MARKERS = ("bug", "regression")


def map_severity(label_names: Iterable[str]) -> str:
    names = [n.lower() for n in label_names]
    if any(m in n for n in names for m in CRITICAL_MARKERS):
        return "Critical"
    if any(m in n for n in names for m in HIGH_MARKERS):
        return "High"
    return "Medium"
