"""Dashboard cold-load benchmark: legacy app (fetch + train on load) vs new app (read lake).

Each trial is a fresh Python subprocess (cold imports, empty Streamlit caches) that runs the
app headlessly with streamlit.testing.AppTest and reports wall time and the number of HTTP
requests sent to api.github.com. A second `run()` in the same process measures the warm
(cached) rerun.

    python -m benchmarks.cold_load --trials 5 --data-uri data
    python -m benchmarks.cold_load --trials 5 --data-uri s3://<bucket>      # after deploying

Writes benchmarks/results/cold_load.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPS = {"legacy": "benchmarks/legacy_app.py", "new": "app/streamlit_app.py"}


def one_trial(which: str) -> dict:
    import requests

    counts = {"github": 0}
    orig = requests.Session.send

    def counting_send(self, request, **kw):
        if "api.github.com" in request.url:
            counts["github"] += 1
        return orig(self, request, **kw)

    requests.Session.send = counting_send
    t_import = time.perf_counter()
    from streamlit.testing.v1 import AppTest

    os.chdir(ROOT)
    at = AppTest.from_file(APPS[which], default_timeout=900)
    t0 = time.perf_counter()
    at.run()
    cold = time.perf_counter() - t0
    if at.exception:
        raise RuntimeError(f"{which} app raised: {at.exception}")
    calls_cold = counts["github"]
    t1 = time.perf_counter()
    at.run()
    warm = time.perf_counter() - t1
    return {
        "app": which,
        "cold_s": cold,
        "cold_incl_streamlit_import_s": time.perf_counter() - t_import - warm,
        "warm_rerun_s": warm,
        "github_calls_cold": calls_cold,
        "github_calls_warm": counts["github"] - calls_cold,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--data-uri", default="data")
    p.add_argument("--apps", default="legacy,new")
    p.add_argument("--one", choices=list(APPS), help=argparse.SUPPRESS)
    p.add_argument("--out", default=str(ROOT / "benchmarks/results/cold_load.json"))
    p.add_argument("--trial-timeout", type=int, default=300, help="seconds before a trial is killed and retried")
    args = p.parse_args()

    if args.one:
        print(json.dumps(one_trial(args.one)))
        return 0

    env = os.environ | {"DATA_URI": args.data_uri, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
    try:
        from dotenv import dotenv_values

        env |= {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}  # GITHUB_TOKEN for legacy app
    except ImportError:
        pass
    results: dict = {"data_uri": args.data_uri, "trials": args.trials, "runs": {}, "discarded_attempts": []}
    for app in args.apps.split(","):
        runs = []
        for i in range(args.trials):
            # The legacy app calls GitHub with no network timeout, so one stalled connection can
            # hang a trial forever (seen: 8h+). Time-box each attempt; retry once; record discards.
            r = None
            for attempt in range(2):
                try:
                    out = subprocess.run(
                        [sys.executable, "-m", "benchmarks.cold_load", "--one", app],
                        cwd=ROOT,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=args.trial_timeout,
                    )
                except subprocess.TimeoutExpired:
                    reason = f"timeout after {args.trial_timeout}s"
                else:
                    if out.returncode == 0:
                        r = json.loads(out.stdout.strip().splitlines()[-1])
                        break
                    reason = "error: " + out.stderr.strip().splitlines()[-1][:200]
                results["discarded_attempts"].append(
                    {"app": app, "trial": i + 1, "attempt": attempt + 1, "reason": reason}
                )
                print(f"{app} trial {i + 1} attempt {attempt + 1} discarded ({reason})", flush=True)
            if r is None:
                raise SystemExit(f"{app} trial {i + 1} failed twice")
            runs.append(r)
            print(
                f"{app} trial {i + 1}: cold={r['cold_s']:.2f}s warm={r['warm_rerun_s']:.2f}s "
                f"github_calls={r['github_calls_cold']}",
                flush=True,
            )
        results["runs"][app] = runs
        cold = [r["cold_s"] for r in runs]
        results[app] = {
            "cold_median_s": statistics.median(cold),
            "cold_min_s": min(cold),
            "cold_max_s": max(cold),
            "warm_median_s": statistics.median(r["warm_rerun_s"] for r in runs),
            "github_calls_per_cold_load": statistics.median(r["github_calls_cold"] for r in runs),
        }
    if "legacy" in results and "new" in results:
        results["speedup_cold_median_x"] = results["legacy"]["cold_median_s"] / results["new"]["cold_median_s"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
