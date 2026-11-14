"""Data-quality checks run on every batch before it reaches processed/.

* Schema problems are fatal for the batch (SchemaError) - writing them would corrupt the
  Glue/Athena table.
* Row-level problems with severity "error" are quarantined (written to quarantine/ with
  the reason) and excluded; severity "warn" problems are counted but the rows are kept.
* Duplicates are counted, then resolved by keeping the latest `updated_at`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from transform.clean import SCHEMA, dedupe_latest

REQUIRED_NOT_NULL = ["repo", "issue_id", "number", "created_at", "updated_at", "state"]
VALID_STATES = {"open", "closed"}


class SchemaError(ValueError):
    pass


@dataclass
class CheckResult:
    name: str
    severity: str  # "error" (rows quarantined) | "warn" (rows kept) | "info"
    failed: int
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.failed == 0


@dataclass
class QualityReport:
    checks: list[CheckResult] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    rows_quarantined: int = 0
    good: pd.DataFrame | None = None
    bad: pd.DataFrame | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    def summary(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_quarantined": self.rows_quarantined,
            "passed": self.passed,
            "checks": [asdict(c) | {"passed": c.passed} for c in self.checks],
        }


def check_schema(df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA if c not in df.columns]
    extra = [c for c in df.columns if c not in SCHEMA]
    if missing or extra:
        raise SchemaError(f"schema mismatch: missing={missing} extra={extra}")
    bad_types = []
    for col, dtype in SCHEMA.items():
        actual = str(df[col].dtype)
        if dtype.startswith("datetime64"):
            ok = actual.startswith("datetime64") and "UTC" in actual
        elif dtype == "int64":
            ok = actual in ("Int64", "int64")
        elif dtype == "float64":
            ok = actual == "float64"
        elif dtype == "string":
            ok = actual in ("string", "object")
        else:
            ok = True
        if not ok:
            bad_types.append(f"{col}:{actual}!={dtype}")
    if bad_types:
        raise SchemaError(f"dtype mismatch: {bad_types}")


def run_checks(df: pd.DataFrame) -> QualityReport:
    report = QualityReport(rows_in=len(df))
    check_schema(df)
    report.checks.append(CheckResult("schema", "error", 0, f"{len(SCHEMA)} columns"))

    reasons = pd.Series([""] * len(df), index=df.index, dtype="object")

    def flag(mask: pd.Series, reason: str) -> int:
        mask = mask.fillna(False).astype(bool)
        reasons[mask] = reasons[mask] + reason + ";"
        return int(mask.sum())

    # 1. Nulls in key fields
    for col in REQUIRED_NOT_NULL:
        n = flag(df[col].isna(), f"null_{col}")
        report.checks.append(CheckResult(f"not_null:{col}", "error", n))

    # 2. Domain checks
    n = flag(~df["state"].isin(VALID_STATES) & df["state"].notna(), "invalid_state")
    report.checks.append(CheckResult("valid_state", "error", n, f"allowed={sorted(VALID_STATES)}"))

    # 3. Temporal integrity: closed_at >= created_at
    bad_close = df["closed_at"].notna() & df["created_at"].notna() & (df["closed_at"] < df["created_at"])
    n = flag(bad_close, "closed_before_created")
    report.checks.append(CheckResult("closed_at>=created_at", "error", n))

    # Warn-level checks (kept, but surfaced)
    upd = int((df["updated_at"] < df["created_at"]).fillna(False).sum())
    report.checks.append(CheckResult("updated_at>=created_at", "warn", upd))
    closed_no_ts = int(((df["state"] == "closed") & df["closed_at"].isna()).sum())
    report.checks.append(CheckResult("closed_has_closed_at", "warn", closed_no_ts))
    empty_title = int((df["title"].fillna("").str.len() == 0).sum())
    report.checks.append(CheckResult("non_empty_title", "warn", empty_title))

    bad_mask = reasons != ""
    bad = df[bad_mask].copy()
    bad["dq_reasons"] = reasons[bad_mask]
    good = df[~bad_mask]

    # 4. Duplicates: count, then resolve
    dup = int(good.duplicated(["repo", "issue_id"]).sum())
    report.checks.append(CheckResult("duplicates_resolved", "info", dup, "kept latest updated_at"))
    good = dedupe_latest(good)
    post_dup = int(good.duplicated(["repo", "issue_id"]).sum())
    report.checks.append(CheckResult("unique:(repo,issue_id)", "error", post_dup))

    report.good, report.bad = good, bad
    report.rows_out, report.rows_quarantined = len(good), len(bad)
    return report
