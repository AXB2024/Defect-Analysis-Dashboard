# Results

Every number below was measured between **2026-11-14 and 2026-11-17**. Each one either comes from a JSON/log
artifact in [`benchmarks/results/`](benchmarks/results/) and the lake's run logs, or is
pasted from the output of a command listed next to it. ML and cold-load tables are generated
by `python -m benchmarks.report_tables` from `metrics.json` and the benchmark JSON files.
Values are shown to 2–3 decimals; the JSON files hold full precision.

**Environment.** Apple M1 (8 cores, 8 GB), macOS 15.0, Python 3.12.15, scikit-learn 1.8.0,
pandas 2.3.3, Streamlit 1.55.0. Live GitHub API with an authenticated token.

**Scope.** The AWS stack has been deployed to us-east-2 since 2026-11-15 (Lambda every
15 minutes, S3 lake, Glue/Athena). Pipeline and ML numbers below come from running the same
code against both a local lake (`--store data`) and the S3 lake. The dashboard cold-load and
latency benchmarks were run against the local lake; the S3 equivalents are listed under
[Not yet measured](#not-yet-measured).

> **Correction (2026-11-17): the original backfill silently skipped months of issues, and
> every ML number first published here was computed on that incomplete data.**
> GitHub's `sort=updated&direction=asc` issue listing is not strictly ordered by `updated_at`
> (seen live: #307470, updated 2026-11-03, served among items updated 2026-04-30). After each
> page-capped run the ingestion moved its watermark to the newest `updated_at` it had seen,
> so a single stray item jumped it months ahead. The first local backfill went
> 2025-11-14 → 2026-01-03 → 2026-04-30 → 2026-11-14 in three capped runs. It stored
> 24,055 issues from 337 pages, when the window needs ~911 pages, and kept e.g. 63 vscode
> issues created in June 2026 instead of 2,524. The fix (commit `099a84f`) resumes capped runs
> from GitHub's cursor and only advances the watermark once the stream is fully read. Both
> lakes were rebuilt from scratch and every model retrained. **All numbers below are from the
> rebuilt data.** The superseded numbers are in git history, with the before/after
> comparison in [§2](#2-pipeline-backfill-and-lake). Pre-fix artifacts are kept as
> `*_gappy_lake*` / `backfill.log` in `benchmarks/results/`.

---

## 1. Tests and coverage

```bash
make lint      # ruff check . && ruff format --check .  -> "All checks passed!", "43 files already formatted"
make test      # pytest --cov --cov-report=term-missing
```

| result | value |
|---|---|
| tests | **107 passed** in 17.24 s |
| line coverage (`common`, `ingestion`, `transform`, `ml`, `app`; argparse CLIs excluded) | **96.02%** |
| CI gate | `--cov-fail-under=85` |

Full per-file report: [`benchmarks/results/pytest_coverage.txt`](benchmarks/results/pytest_coverage.txt).
GitHub is mocked with `responses` and S3/SSM with moto. The suite covers transforms, label
mapping, bucket/censoring logic and every DQ check. It also runs end-to-end ingestion (200 →
304, cursor-based resume of capped runs, expired cursors, partial failure, quarantine, SSM
token), trains on synthetic data, and renders the Streamlit app headlessly (`AppTest`).
Regression tests cover both production bugs: a simulated out-of-order GitHub stream (the
old watermark logic ingests 2 of 7 issues) and merging microsecond with nanosecond timestamps.

`sam validate --lint` passes on `infra/template.yaml` and `infra/bootstrap.yaml`, and
`sam build` succeeds (SAM CLI 1.172.1): `make validate && make build`.

## 2. Pipeline: backfill and lake

```bash
python -m ingestion.run --store data --until-caught-up --max-pages 100 --max-runs 200
#   log: benchmarks/results/backfill_fixed.log
python -m ingestion.run --store s3://defect-dashboard-lake-444301506063-us-east-2 --until-caught-up --max-pages 100 --max-runs 200
#   log: benchmarks/results/backfill_fixed_s3.log
```

| metric | local lake | S3 lake |
|---|---|---|
| backfill window | 365 days (`since` = 2025-11-17 04:27 UTC) | same |
| capped runs until caught up | 9 (100 pages per repo per run) | 9 |
| wall time | 1,023.19 s | 1,711.57 s |
| GitHub API calls | 919; 7 were free 304s, so 912 billable; 0 retries | 919; 0 retries |
| failures / rejected cursors | 0 / 0 | 0 / 0 |
| processed lake | **58,985 unique issues** (vscode 57,935, react 1,050) in 228 Parquet partitions, 52 MB | 58,986 (the extra issue was created after the local run finished) |
| raw zone | 912 gzipped API pages, 90 MB | — |
| DQ | 0 rows quarantined, 0 duplicate `(repo, issue_id)` | 0 duplicates |

The two lakes agree issue for issue: all 58,985 local issues are in S3 with identical state.
GitHub's search API reports 87,590 vscode and 3,459 react issues+PRs updated in the window,
which is ~911 pages, matching the 912 billable pages fetched.

**Before vs after the fix** (vscode issues created per month, inside the window):

| created | 2025-12 | 2026-01 | 2026-02 | 2026-04 | 2026-05 | 2026-06 | 2026-07 | 2026-08 | 2026-09 | 2026-10 |
|---|---|---|---|---|---|---|---|---|---|---|
| original backfill | 2,626 | 652 | 442 | 1,457 | 610 | 63 | 14 | 15 | 4 | 0 |
| rebuilt | 3,927 | 4,704 | 4,424 | 3,631 | 3,375 | 2,524 | 2,556 | 2,714 | 2,672 | 2,959 |

The original backfill took 346.19 s and 339 calls only because it skipped most of the
stream. 418 issues from it are absent from the rebuild: all were last updated before the
new window starts (3 days later), so they're outside it, not lost. React was unaffected,
because it never needed more than one capped run.

The lake also contains issues created before the window but updated inside it. Training
restricts itself to issues *created* in the window (43,592), because older issues are a
biased sample.

**ETag / 304 check** ([`etag_304_check.txt`](benchmarks/results/etag_304_check.txt)). Against
the live API, GitHub's `core.used` counter rose by 1 for the 200 and by **0 for each of two
authenticated 304 responses**. An unauthenticated 304 during the early smoke test *did*
decrement `X-RateLimit-Remaining` (55 → 54). The exemption needs a token.

## 3. ML baselines (same chronological test set per task)

Command: `python -m ml.train --store data` (log: [`train.log`](benchmarks/results/train.log)).
The whole process took 477.75 s wall time.

Model version `20261117T045223Z` · snapshot (as_of) 2026-11-17 04:43:02.715853+00:00 · 43592 issues created 2025-11-17 → 2026-11-17 (microsoft/vscode: 42942, react/react: 650) · train+evaluate+refit 439.14 s · then batch scoring 58985 rows in 32.21 s

### Regression: resolution time (days)

Target: min(resolution_days, 90); open issues older than 90d = 90. Train n=28054 (created 2025-11-17 → 2026-05-28), test n=7014 (created 2026-05-28 → 2026-08-19). Variant chosen on validation: `raw_days` (val MAE: raw_days 33.462, log1p_days 33.734).

| model | MAE (days) | RMSE (days) |
|---|---|---|
| median baseline (predict 7.458 d) | 35.881 | 50.972 |
| RandomForestRegressor | 34.332 | 37.766 |
| *ablation: + leaky post-hoc features (not served)* | 30.419 | 34.873 |

Improvement over baseline: MAE **4.32%**, RMSE **25.91%**. Test target: mean 38.17 d, median 9.78 d, 34.1% at the 90-day cap.

### Resolution bucket classification

Train n=32524 (2025-11-17 → 2026-07-20), test n=8131 (2026-07-20 → 2026-10-18). class_weight chosen on validation: `None` (val macro-F1 None: 0.434, balanced: 0.432).

| model | accuracy | macro-F1 |
|---|---|---|
| majority baseline (`<1d`) | 0.262 | 0.104 |
| LogisticRegression (TF-IDF + numeric) | 0.532 | 0.398 |
| *ablation: + leaky post-hoc features (not served)* | 0.531 | 0.413 |

Class distribution: train {'<1d': 10663, '1-7d': 3736, '7-30d': 7630, '>30d': 10495}, test {'<1d': 2127, '1-7d': 1020, '7-30d': 1161, '>30d': 3823}.

LogReg confusion matrix (test):

| actual \ predicted | <1d | 1-7d | 7-30d | >30d |
|---|---|---|---|---|
| **<1d** | 765 | 85 | 190 | 1087 |
| **1-7d** | 137 | 181 | 47 | 655 |
| **7-30d** | 171 | 29 | 216 | 745 |
| **>30d** | 338 | 87 | 234 | 3164 |

Per-class (LogReg): `<1d` P=0.54 R=0.36 F1=0.43; `1-7d` P=0.47 R=0.18 F1=0.26; `7-30d` P=0.31 R=0.19 F1=0.23; `>30d` P=0.56 R=0.83 F1=0.67

Majority-baseline confusion matrix (test):

| actual \ predicted | <1d | 1-7d | 7-30d | >30d |
|---|---|---|---|---|
| **<1d** | 2127 | 0 | 0 | 0 |
| **1-7d** | 1020 | 0 | 0 | 0 |
| **7-30d** | 1161 | 0 | 0 | 0 |
| **>30d** | 3823 | 0 | 0 | 0 |

### Category classification (weak labels from area labels)

Train n=9376 (2025-11-17 → 2026-07-27), test n=2344 (2026-07-27 → 2026-11-17). Test distribution {'AI/Chat': 1588, 'Integration': 289, 'UI': 288, 'Performance': 125, 'Core': 54}.

| model | accuracy | macro-F1 |
|---|---|---|
| majority baseline (`AI/Chat`) | 0.677 | 0.162 |
| TF-IDF + LogisticRegression | 0.767 | 0.606 |

Per-class F1: `AI/Chat` 0.85 (n=1588); `Core` 0.28 (n=54); `Integration` 0.66 (n=289); `Performance` 0.63 (n=125); `UI` 0.61 (n=288)

**S3 lake (production).** The same training job run against the rebuilt S3 lake
(`python -m ml.train --store s3://defect-dashboard-lake-444301506063-us-east-2`, log
[`train_s3.log`](benchmarks/results/train_s3.log)) produced model `20261117T053422Z` with
**identical test metrics** to every decimal. That's expected, not a copy: its snapshot is
16 minutes later (as_of 2026-11-17 04:59:06), and its one extra issue is too new to fall in
any model's train or test population. So both runs see the same 28,054/7,014 regression,
32,524/8,131 bucket and 9,376/2,344 category train/test issues, and the random seeds are fixed.
Train+evaluate took 451.53 s on S3 data vs 439.14 s locally. The S3 job's wall time was
3,203.39 s, mostly reading 228 partitions over the network while another job used the CPU.

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
| legacy (fetch + train on load) | 16.49 (15.42–17.95) | 0.06 | 15 |
| new (read lake + saved models) | 3.58 (3.28–4.00) | 0.43 | 0 |
| *new, first version (re-scored all issues on load)* | 17.60 (16.13–18.42) | 0.42 | 0 |

Speed-up (legacy median / new median): **4.61×**; 5 trials each.

- **The first version of the new app was slower than the legacy one** (17.60 s vs ~14 s,
  measured on the original 24,055-issue lake).
  Profiling (`build_features` 3.83 s and category prediction over 24k issues 5.82 s, on top
  of a 1.99 s Parquet read) showed it was re-scoring every issue on every load. Moving that
  to batch scoring in the training job, delta-scoring only new issues and skipping the
  `body` column brought it to 4.81 s on that lake
  ([`cold_load_gappy_lake.json`](benchmarks/results/cold_load_gappy_lake.json)). The table
  above is a fresh side-by-side run on the rebuilt 58,985-issue lake.
- **Compare apps only within one run.** The legacy app's median moved from 13.76 s to 16.49 s
  between sessions, because its GitHub fetches depend on network conditions. So the
  speed-up is quoted from a single side-by-side run, not across sessions. The new app did not
  get slower on 2.4× more data (3.58 s vs 4.81 s); I haven't investigated why it got faster
  and don't attribute it to anything.
- **One earlier run hung** for 8 h 41 min: the legacy app calls GitHub with no network
  timeout. The harness now kills and retries a trial after 300 s and records discarded
  attempts in the JSON; the run above discarded none.
- The comparison isn't like-for-like on data volume. The new app works on **58,985 issues**,
  while the legacy app ended up showing **78 issues**
  ([`legacy_kpis.txt`](benchmarks/results/legacy_kpis.txt)). It fetches 5 pages per repo
  (mostly React PRs), then keeps only closed issues.
- Warm reruns are slower in the new app (0.36 s vs 0.09 s) because it renders far more data
  and four tabs.
- The new app was measured reading a **local** lake. Reading from S3 adds network transfer
  (the lake is 52 MB of Parquet) and hasn't been measured.

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

1. **Resolution-time regression is weak on MAE: only 4.32% better than predicting the
   median (34.33 vs 35.88 days).** RMSE improves much more (25.91%: 37.77 vs 50.97), so the
   forest mainly avoids the median baseline's large misses on long-lived issues, while
   typical errors barely shrink.
   - **Bimodal target.** 34.1% of test issues sit at the 90-day cap while the median is
     9.78 days. A single point estimate is the wrong output for that.
   - **Drift was mostly a data artifact.** Before the fix, validation MAE (7.49) and test
     MAE (33.03) were far apart, which I had attributed to drift. On complete data they're
     close (33.46 vs 34.33), so the gap came largely from the missing months.
   - **Next steps:**
     - Fit a survival / accelerated-failure-time model that uses censoring natively instead
       of capping.
     - Predict quantiles or reuse the bucket probabilities instead of a mean.
     - Add a "triage-time" model whose features come from the first hour (first maintainer
       response, labels at T+1h via the issue timeline API).
2. **The bucket classifier beats the baseline clearly on macro-F1 (0.398 vs 0.104) and on
   accuracy (0.532 vs 0.262), but the middle buckets are poor.**
   - Per-class F1: `<1d` 0.43, `1-7d` 0.26, `7-30d` 0.23, `>30d` 0.67. The model predicts
     `>30d` for 69.5% of test issues (recall 0.83 for that class), so the confusion is
     mostly short-lived issues being called long-lived.
   - The class mix still shifts between periods: `>30d` is 32.3% of training issues and
     47.0% of test issues, and `<1d` drops from 32.8% to 26.2%. That's why the majority
     baseline (`<1d`, from training) scores only 0.262.
   - **Next steps:** an ordinal model (buckets are ordered), probability calibration, and
     recency-weighted or rolling-window retraining.
3. **Category classification is solid for large classes and weak for the smallest.**
   - F1 is 0.85 for `AI/Chat` (n=1,588), 0.66 for `Integration` (n=289), 0.63 for
     `Performance` (n=125) and 0.61 for `UI` (n=288), but 0.28 for `Core` (n=54).
   - Labels are weak (derived from area labels), and 68.5% of all issues have no mappable
     area label (`Other`, 40,396 of 58,985).
   - **Next steps:** more repos for minority classes, a small hand-labelled evaluation set,
     and sentence embeddings instead of TF-IDF.
4. **The leakage ablation shows real inflation.** Post-hoc features would have raised bucket
   macro-F1 from 0.398 to 0.413 and cut MAE from 34.33 to 30.42. The legacy app's in-sample
   evaluation was much worse: its "Predicted Resolution" equals the actual average (0.42 vs
   0.42) because the forest is scored on its own training rows.
5. **Two production bugs reached the deployed pipeline before tests caught them.** Both now
   have regression tests:
   - the out-of-order GitHub stream skipping issues (see the correction at the top);
   - the Lambda failing every run from 2026-11-15 00:49 UTC: its pandas 2.1.4 couldn't merge
     microsecond timestamps written locally with new nanosecond ones. The fix normalises
     timestamps and pins the Lambda's pandas layer to the project's version (2.3.3). The
     first fixed run caught up all 191 missed records in 9.84 s. A new alarm now also fires
     when a single repo fails, which went unalerted for about 6 hours.

## Not yet measured

- Lambda duration and cold start per scheduled run, from CloudWatch `Duration`/`InitDuration`.
- Dashboard cold load reading from S3:
  `python -m benchmarks.cold_load --trials 5 --data-uri s3://<bucket> --apps new`.
- End-to-end latency on the deployed schedule (now running every 15 minutes), summarised with
  `python -m benchmarks.summarize_runs --store s3://defect-dashboard-lake-444301506063-us-east-2`.
- Athena query latency and bytes scanned for the README query.
- Actual monthly cost, from Cost Explorer after a full month.
- The Streamlit Community Cloud deployment (manual UI step; see README).
