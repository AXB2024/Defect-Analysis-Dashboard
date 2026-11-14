# Results

Every number below was measured on **2026-11-14**. Each one either comes from a JSON/log
artifact in [`benchmarks/results/`](benchmarks/results/) and the lake's run logs, or is
pasted from the output of a command listed next to it. ML and cold-load tables are generated
by `python -m benchmarks.report_tables` from `metrics.json` and the benchmark JSON files.
Values are shown to 2–3 decimals; the JSON files hold full precision.

**Environment.** Apple M1 (8 cores, 8 GB), macOS 15.0, Python 3.12.15, scikit-learn 1.8.0,
pandas 2.3.3, Streamlit 1.55.0. Live GitHub API with an authenticated token.

**Important scope note.** No AWS credentials were available in this environment, and
deploying is an outward-facing step that needs your decision. So **the AWS stack has not been
deployed and nothing here was measured on AWS**. The pipeline, training and dashboard code
that would run on AWS was executed unchanged against a *local* lake (`--store data`). The S3
code path is exercised only by moto-mocked tests. Numbers that need a real deployment are
listed under [Not yet measured](#not-yet-measured).

---

## 1. Tests and coverage

```bash
make lint      # ruff check . && ruff format --check .  -> "All checks passed!", "40 files already formatted"
make test      # pytest --cov --cov-report=term-missing
```

| result | value |
|---|---|
| tests | **99 passed** in 15.24 s |
| line coverage (`common`, `ingestion`, `transform`, `ml`, `app`; argparse CLIs excluded) | **95.90%** |
| CI gate | `--cov-fail-under=85` |

Full per-file report: [`benchmarks/results/pytest_coverage.txt`](benchmarks/results/pytest_coverage.txt).
GitHub is mocked with `responses` and S3/SSM with moto. The suite covers transforms, label
mapping, bucket/censoring logic and every DQ check. It also runs end-to-end ingestion (200 →
304, capped backfill resume, partial failure, quarantine, SSM token), trains on synthetic
data, and renders the Streamlit app headlessly (`AppTest`).

`sam validate --lint` passes on `infra/template.yaml` and `infra/bootstrap.yaml`, and
`sam build` succeeds (SAM CLI 1.172.1): `make validate && make build`.

## 2. Pipeline: backfill and lake

```bash
python -m ingestion.run --store data --until-caught-up --max-pages 100   # log: benchmarks/results/backfill.log
```

| metric | value |
|---|---|
| backfill window | 365 days (`since` = 2025-11-14) |
| capped runs until caught up | 4 (100 pages per repo per run) |
| wall time | 346.19 s |
| GitHub API calls | 339 (135 + 101 + 101 + 2); 2 of them were 304s, so 337 billable; 0 retries |
| raw pages stored | 337 gzipped JSON pages, 30 MB |
| processed lake | **24,055 unique issues** (microsoft/vscode 23,008, react/react 1,047) in 219 Parquet partitions, 21 MB |
| DQ | 0 rows quarantined, 0 duplicate `(repo, issue_id)` after upsert |

The lake also contains issues created before the window but updated inside it. Training
restricts itself to issues *created* in the window (11,894), because older issues are a
biased sample.

**ETag / 304 check** ([`etag_304_check.txt`](benchmarks/results/etag_304_check.txt)). Against
the live API, GitHub's `core.used` counter rose by 1 for the 200 and by **0 for each of two
authenticated 304 responses**. An unauthenticated 304 during the early smoke test *did*
decrement `X-RateLimit-Remaining` (55 → 54). The exemption needs a token.

## 3. ML baselines (same chronological test set per task)

Command: `python -m ml.train --store data` (log: [`train.log`](benchmarks/results/train.log)).
The whole process took 136.70 s wall time.

Model version `20261114T193712Z` · snapshot (as_of) 2026-11-14 19:30:07.493404+00:00 · 11894 issues created 2025-11-14 → 2026-11-14 (microsoft/vscode: 11245, react/react: 649) · train+evaluate+refit 119.6 s · then batch scoring 24056 rows in 12.71 s

### Regression: resolution time (days)

Target: min(resolution_days, 90); open issues older than 90d = 90. Train n=9319 (created 2025-11-14 → 2026-03-31), test n=2330 (created 2026-03-31 → 2026-08-13). Variant chosen on validation: `log1p_days` (val MAE: raw_days 14.801, log1p_days 7.486).

| model | MAE (days) | RMSE (days) |
|---|---|---|
| median baseline (predict 7.020 d) | 35.242 | 51.286 |
| RandomForestRegressor | 33.026 | 48.524 |
| *ablation: + leaky post-hoc features (not served)* | 31.178 | 46.577 |

Improvement over baseline: MAE **6.29%**, RMSE **5.38%**. Test target: mean 36.00 d, median 5.65 d, 37.3% at the 90-day cap.

### Resolution bucket classification

Train n=9437 (2025-11-14 → 2026-04-01), test n=2360 (2026-04-01 → 2026-10-14). class_weight chosen on validation: `balanced` (val macro-F1 None: 0.367, balanced: 0.411).

| model | accuracy | macro-F1 |
|---|---|---|
| majority baseline (`<1d`) | 0.353 | 0.130 |
| LogisticRegression (TF-IDF + numeric) | 0.480 | 0.402 |
| *ablation: + leaky post-hoc features (not served)* | 0.504 | 0.438 |

Class distribution: train {'<1d': 3537, '1-7d': 1175, '7-30d': 2830, '>30d': 1895}, test {'<1d': 832, '1-7d': 320, '7-30d': 190, '>30d': 1018}.

LogReg confusion matrix (test):

| actual \ predicted | <1d | 1-7d | 7-30d | >30d |
|---|---|---|---|---|
| **<1d** | 409 | 163 | 111 | 149 |
| **1-7d** | 65 | 118 | 56 | 81 |
| **7-30d** | 69 | 29 | 47 | 45 |
| **>30d** | 92 | 244 | 123 | 559 |

Per-class (LogReg): `<1d` P=0.64 R=0.49 F1=0.56; `1-7d` P=0.21 R=0.37 F1=0.27; `7-30d` P=0.14 R=0.25 F1=0.18; `>30d` P=0.67 R=0.55 F1=0.60

Majority-baseline confusion matrix (test):

| actual \ predicted | <1d | 1-7d | 7-30d | >30d |
|---|---|---|---|---|
| **<1d** | 832 | 0 | 0 | 0 |
| **1-7d** | 320 | 0 | 0 | 0 |
| **7-30d** | 190 | 0 | 0 | 0 |
| **>30d** | 1018 | 0 | 0 | 0 |

### Category classification (weak labels from area labels)

Train n=2445 (2025-11-14 → 2026-03-30), test n=612 (2026-03-30 → 2026-11-10). Test distribution {'AI/Chat': 406, 'Integration': 120, 'UI': 62, 'Performance': 18, 'Core': 6}.

| model | accuracy | macro-F1 |
|---|---|---|
| majority baseline (`AI/Chat`) | 0.663 | 0.160 |
| TF-IDF + LogisticRegression | 0.735 | 0.540 |

Per-class F1: `AI/Chat` 0.81 (n=406); `Core` 0.29 (n=6); `Integration` 0.76 (n=120); `Performance` 0.31 (n=18); `UI` 0.52 (n=62)

**How to read these numbers.** Each split is chronological: training issues are older than
every test issue, and nothing is shuffled. Model selection (target transform, class weights)
used only a validation slice cut from the end of the training window; the test set was
scored once. Targets are censoring-aware:
- An open issue older than 30 days counts as `>30d`.
- Regression only uses issues at least 90 days old, with targets capped at 90.

The "leaky" rows add post-hoc features (label count, has-bug-label, comment count) to show
what the legacy feature set would inflate. The served models don't use them.

## 4. Systems benchmarks: before vs after

### 4a. Dashboard cold load

```bash
python -m benchmarks.cold_load --trials 5 --data-uri data   # -> benchmarks/results/cold_load.json
```

Each trial is a fresh Python process running the app headlessly via
`streamlit.testing.AppTest`. That includes imports and empty caches, but not browser
rendering. "Legacy" is the original `app.py`, frozen at
[`benchmarks/legacy_app.py`](benchmarks/legacy_app.py).

| app | cold load median (min–max), s | warm rerun median, s | GitHub API calls per load |
|---|---|---|---|
| legacy (fetch + train on load) | 13.76 (13.69–14.20) | 0.09 | 15 |
| new (read lake + saved models) | 4.81 (4.62–5.66) | 0.36 | 0 |
| *new, first version (re-scored all issues on load)* | 17.60 (16.13–18.42) | 0.42 | 0 |

Speed-up (legacy median / new median): **2.86×**; 5 trials each.

- **The first version of the new app was slower than the legacy one** (17.60 s vs ~14 s).
  Profiling (`build_features` 3.83 s and category prediction over 24k issues 5.82 s, on top
  of a 1.99 s Parquet read) showed it was re-scoring every issue on every load. Moving that
  to batch scoring in the training job, delta-scoring only new issues and skipping the
  `body` column brought it to 4.81 s. The `cold_load_before_batch_scoring.json` row was
  measured about 40 minutes before the final run; the legacy app measured 14.49 s median then.
- The comparison isn't like-for-like on data volume. The new app works on **24,055 issues**,
  while the legacy app ended up showing **78 issues**
  ([`legacy_kpis.txt`](benchmarks/results/legacy_kpis.txt)). It fetches 5 pages per repo
  (mostly React PRs), then keeps only closed issues.
- Warm reruns are slower in the new app (0.36 s vs 0.09 s) because it renders far more data
  and four tabs.
- The new app was measured reading a **local** lake. Reading from S3 adds network transfer
  (the lake is 21 MB of Parquet) and hasn't been measured.

### 4b. GitHub API calls per refresh

| | legacy app | new architecture |
|---|---|---|
| calls when the dashboard (re)loads | **15** per cache miss (5 pages × vscode, plus 5 pages × react where each page is a 301 redirect followed by the real request) | **0**: the dashboard never calls GitHub |
| calls to keep data fresh | none: data is frozen until someone clicks Refresh, which re-fetches all 15 | the ingestion job, see 4c: 2 calls per 15-min run, mean 1.375 billable (8 runs) |

### 4c. Ingestion latency: issue created → queryable in the lake

```bash
python -m benchmarks.schedule_local --store data --interval 900 --runs 8      # log: benchmarks/results/schedule_local.log
python -m benchmarks.summarize_runs --store data --since 2026-11-14T19:30:06+00:00   # -> scheduled_runs.json
```

This simulates the deployed schedule locally: the same `run_ingestion` code runs every 900 s
against the live GitHub API and the local lake, 8 runs in total. **All 8 succeeded.**

**Caveat: the laptop slept during the simulation.** A sleeping Mac pauses the 15-minute
wait, so 3 of the 7 gaps between runs were longer than 16 minutes. One of them was 2 h 3.5 min,
before run 8. An issue created during a gap waits for the next run, so latency samples
from late runs measure the laptop's sleep, not the pipeline. A real EventBridge schedule
doesn't drift this way.

| run | started (UTC) | gap since previous | API calls | of which 304 | records upserted | run duration (s) | new-issue latency samples (s) |
|---|---|---|---|---|---|---|---|
| 1 | 19:30:07 | - | 2 | 1 | 12 | 1.47 | - |
| 2 | 19:45:07 | 15.0 min | 2 | 1 | 16 | 1.328 | 591.5 |
| 3 | 20:00:07 | 15.0 min | 2 | 1 | 13 | 1.299 | 344.4 |
| 4 | 20:23:25 | 23.3 min | 2 | 0 | 24 | 1.721 | - |
| 5 | 20:38:25 | 15.0 min | 2 | 0 | 11 | 1.633 | - |
| 6 | 21:01:42 | 23.3 min | 2 | 1 | 0 | 2.428 | - |
| 7 | 21:16:41 | 15.0 min | 2 | 1 | 14 | 1.384 | 602.8 |
| 8 | 23:20:11 | 123.5 min | 2 | 0 | 36 | 1.575 | 4112.5, 1731.5, 6258.5 |

| metric (all 8 runs) | value |
|---|---|
| status | success: 8 |
| API calls per run (both repos) | 2 every run |
| billable calls per run | mean 1.375 (min 1, max 2) |
| repo polls answered 304 (free) | 5/16 |
| records upserted | 126 |
| run duration | median 1.5225 s, max 2.428 s |

**Latency (created_at → Parquet write finished):**
- **On-schedule runs** (gap ≤ 16 min): 3 samples, 344.4, 591.5, 602.8 s. All are under
  the 900 s schedule interval, as expected: an issue waits at most one interval plus the run time.
- **Late runs**: 3 samples, 1731.5, 4112.5, 6258.5 s. These are inflated by
  the sleep gap and are not a pipeline measurement.
- All 6 samples together (what `summarize_runs` reports): p50
  1167.15 s, max 6258.5 s.
  The late runs dominate this, so it isn't representative.

3 on-schedule samples are too few for reliable percentiles. The next step is to measure
on the deployed Lambda schedule, where a laptop can't sleep, with
`python -m benchmarks.summarize_runs --store s3://<bucket>`. The dashboard adds up to
`DATA_TTL_S = 300` s of cache by configuration; that part wasn't measured.

## 5. Weak results and next steps

1. **Resolution-time regression is weak: only a 6.29% MAE improvement over predicting the
   median.**
   - **Bimodal target.** 37.3% of test issues sit at the 90-day cap, while the median is
     5.65 days. A single point estimate is the wrong output for that.
   - **Drift.** Validation MAE was 7.49 but test MAE is 33.03. The test window (Apr–Aug
     2026) has many more long-lived issues than the training window.
   - **Next steps:**
     - Fit a survival / accelerated-failure-time model that uses censoring natively instead
       of capping.
     - Predict quantiles or reuse the bucket probabilities instead of a mean.
     - Retrain on a rolling window or with recency weighting.
     - Add a "triage-time" model whose features come from the first hour (first maintainer
       response, labels at T+1h via the issue timeline API).
2. **The bucket classifier beats the baseline on macro-F1 (0.402 vs 0.130), but its accuracy
   is only 0.480.**
   - The middle buckets are poor: `1-7d` F1 0.27 and `7-30d` F1 0.18.
   - The class mix shifted a lot: `7-30d` was 30% of training data and 8% of test data, and
     `>30d` went from 20% to 43%.
   - **Next steps:** an ordinal model (buckets are ordered), probability calibration, and
     the drift fixes above.
3. **Category classification works for big classes and fails for small ones.**
   - F1 is 0.81 for `AI/Chat` (n=406) and 0.76 for `Integration` (n=120). It's 0.29 for
     `Core` (n=6) and 0.31 for `Performance` (n=18), where per-class numbers are too noisy
     to trust.
   - Labels are weak (derived from area labels), and 67.8% of all issues had no mappable
     area label at backfill time (`Other`, 16,317 of 24,055).
   - **Next steps:** a longer window or more repos for minority classes, a small
     hand-labelled evaluation set, and sentence embeddings instead of TF-IDF.
4. **The leakage ablation shows modest inflation.** Post-hoc features would have raised bucket
   macro-F1 from 0.402 to 0.438 and cut MAE from 33.03 to 31.18. The legacy app's in-sample
   evaluation was much worse: its "Predicted Resolution" equals the actual average (0.42 vs
   0.42) because the forest is scored on its own training rows.

## Not yet measured

These need the AWS deployment (`make deploy`) and will replace the local equivalents above:

- Lambda duration and cold start per scheduled run, from CloudWatch `Duration`/`InitDuration`.
- Dashboard cold load reading from S3:
  `python -m benchmarks.cold_load --trials 5 --data-uri s3://<bucket> --apps new`.
- End-to-end latency on the deployed schedule: the same run-log fields, summarised with
  `python -m benchmarks.summarize_runs --store s3://<bucket>`.
- Athena query latency and bytes scanned for the README query.
- Actual monthly cost, from Cost Explorer after a full month.
- The Streamlit Community Cloud deployment (manual UI step; see README).
