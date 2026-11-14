"""Feature engineering. Only information available *when the issue is opened* is used by
default; label/comment features are post-hoc (added during triage/discussion) and are
kept behind `LEAKY_NUMERIC` for an ablation that shows how much they would inflate scores."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)  # issue templates are full of these
_CODE_FENCE = re.compile(r"```")
_URL = re.compile(r"https?://\S+")
_IMAGE = re.compile(r"!\[[^\]]*\]\(|<img\s", re.I)
_STACK = re.compile(r"(?:Traceback \(most recent call last\)|^\s+at .+[:(]\d+|Error: .+\n\s+at )", re.M)
_REPRO = re.compile(r"steps to reproduce|repro steps|reproduction|to reproduce", re.I)

NUMERIC = [
    "title_len",
    "body_len",
    "n_code_blocks",
    "n_links",
    "n_images",
    "has_stack_trace",
    "has_repro",
    "title_has_question",
    "created_hour",
    "created_dow",
]
CATEGORICAL = ["repo", "author_association"]
LEAKY_NUMERIC = ["num_labels", "has_bug_label", "num_comments"]  # ablation only


def clean_text(s: str | None) -> str:
    s = _HTML_COMMENT.sub(" ", s or "")
    return _URL.sub(" URL ", s)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    body = df["body"].fillna("").astype(str)
    title = df["title"].fillna("").astype(str)
    out = pd.DataFrame(index=df.index)
    out["text"] = (title + " \n " + title + " \n " + body.map(clean_text)).str.slice(0, 6000)  # title weighted x2
    out["title_len"] = title.str.len()
    out["body_len"] = np.log1p(body.str.len())
    out["n_code_blocks"] = body.str.count(_CODE_FENCE) // 2
    out["n_links"] = body.str.count(_URL)
    out["n_images"] = body.str.count(_IMAGE)
    out["has_stack_trace"] = body.str.contains(_STACK).astype(int)
    out["has_repro"] = body.str.contains(_REPRO).astype(int)
    out["title_has_question"] = title.str.contains(r"\?").astype(int)
    out["created_hour"] = df["created_at"].dt.hour
    out["created_dow"] = df["created_at"].dt.dayofweek
    out["repo"] = df["repo"].astype(str)
    out["author_association"] = df["author_association"].fillna("NONE").astype(str)
    labels = df["label_names"].apply(lambda x: list(x) if x is not None else [])
    out["num_labels"] = labels.apply(len)
    out["has_bug_label"] = labels.apply(lambda ls: int(any("bug" in str(n).lower() for n in ls)))
    out["num_comments"] = df["num_comments"].fillna(0).astype(int)
    return out


def preprocessor(text: bool = True, leaky: bool = False, max_features: int = 20000) -> ColumnTransformer:
    numeric = NUMERIC + (LEAKY_NUMERIC if leaky else [])
    parts = [
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ]
    if text:
        parts.append(
            (
                "text",
                TfidfVectorizer(
                    max_features=max_features, ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english"
                ),
                "text",
            )
        )
    return ColumnTransformer(parts, sparse_threshold=1.0 if text else 0.0)
