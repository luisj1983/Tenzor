#!/usr/bin/env python3
"""
Regenerate the performance baseline JSON consumed by
tests/backend_parity/test_performance_regression.cpp.

Workflow:
    1. Build `test_performance_regression`.
    2. Run the DISABLED_BaselineRegressionCheck test — it prints per-backend
       timings to stdout in lines like:
            [backend=cpu] MatMul_512x512 perf: median=0.84ms p99=1.10ms
       (The print format is what this script parses.)
    3. Parse stdout, write the per-host JSON in tests/backend_parity/baselines/.

The baseline is a HOST-KEYED map so a CI self-hosted runner and a developer
laptop can both commit their own numbers without clobbering each other:

    { "hosts": { "<hostname>": { "ops": { "<op>_<size>": { "<backend>": {
        "median_ms": X, "p99_ms": Y } } } } } }

Only the CURRENT host's entry is rewritten; other hosts' entries are
preserved. The regression check (DISABLED_BaselineRegressionCheck) reads this
JSON and enforces when TENZOR_PERF_ENFORCE=1 or the running host has an entry,
failing when a measured value exceeds the threshold (env-tunable):
    TENZOR_PERF_REGRESSION_RTOL      — default 1.15 on median
    TENZOR_PERF_REGRESSION_P99_RTOL  — default 1.50 on p99

Usage:
    cd <repo_root>
    # Run the (disabled) enforcement test so all ops are measured:
    python tools/regen_perf_baseline.py [--build-dir build]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

# A line we emit from the perf test looks like:
#   [backend=cuda:0] MatMul_512x512 perf: median=0.84ms p99=1.10ms
# The op key (already in "<op>_<size>" form) is captured whole.
LINE_RE = re.compile(
    r"^\[backend=(?P<backend>[A-Za-z0-9_:]+)\]\s+"
    r"(?P<key>[A-Za-z0-9_]+)\s+perf:\s+"
    r"median=(?P<median>[\d.]+)ms\s+"
    r"p99=(?P<p99>[\d.]+)ms\s*$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", default="build",
                   help="CMake build directory (default: build)")
    p.add_argument("--baseline",
                   default="tests/backend_parity/baselines/perf_baseline.json",
                   help="Path to the baseline JSON to write")
    # --executable defaults to None so main() can derive it from --build-dir
    # (<build-dir>/bin/test_performance_regression) -- a hardcoded
    # "bin/test_performance_regression" default silently ignored --build-dir
    # entirely, so `--build-dir build` (the exact form this script's own
    # docstring/help text recommends) looked in <repo_root>/bin instead of
    # <repo_root>/build/bin and always failed with "binary not found".
    p.add_argument("--executable", default=None,
                   help="Path to the test binary, relative to repo root "
                        "(default: <build-dir>/bin/test_performance_regression)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    executable = args.executable or f"{args.build_dir}/bin/test_performance_regression"
    bin_path = repo_root / executable
    baseline_path = repo_root / args.baseline

    if not bin_path.exists():
        print(f"perf binary not found at {bin_path}; build it first with"
              f" `ninja -C {args.build_dir} test_performance_regression`",
              file=sys.stderr)
        return 1

    # Run only the enforcement test, and run it even though it is DISABLED_.
    # The test prints timings for every op on every available backend.
    print(f"Running {bin_path} (BaselineRegressionCheck) ...", flush=True)
    completed = subprocess.run(
        [str(bin_path),
         "--gtest_filter=*BaselineRegressionCheck*",
         "--gtest_also_run_disabled_tests"],
        capture_output=True, text=True, check=False)
    print(completed.stdout)
    # NOTE: a non-zero exit is EXPECTED during regeneration. When the current
    # host already has a baseline entry, BaselineRegressionCheck enforces the
    # OLD numbers and fails if the freshly-measured timings drift — which is
    # exactly what we are here to overwrite (chicken-and-egg). The timing lines
    # are printed regardless of the enforcement verdict, so we parse stdout and
    # rewrite the baseline anyway. We only abort if NO timing lines were parsed
    # (handled below), i.e. the binary actually failed to run / changed format.
    if completed.returncode != 0:
        print(f"perf binary exited with {completed.returncode} (expected during "
              f"regen if this host already has a baseline); parsing timings anyway.",
              file=sys.stderr)

    ops: dict[str, dict[str, dict[str, float]]] = {}
    for line in completed.stdout.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        key = m.group("key")
        ops.setdefault(key, {})[m.group("backend")] = {
            "median_ms": float(m.group("median")),
            "p99_ms":   float(m.group("p99")),
        }

    if not ops:
        print("No timing lines parsed — perf test output format may have changed.",
              file=sys.stderr)
        print("Expected format:", LINE_RE.pattern, file=sys.stderr)
        return 1

    # Read existing baseline (preserves _comment fields and other hosts).
    if baseline_path.exists():
        try:
            existing = json.loads(baseline_path.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}

    host = platform.node()
    hosts = existing.get("hosts")
    if not isinstance(hosts, dict):
        hosts = {}
    hosts[host] = {
        "tenzor_version": os.environ.get("TENZOR_VERSION", "unknown"),
        "captured_at_iso8601": _dt.datetime.utcnow().isoformat() + "Z",
        "ops": ops,
    }
    existing["hosts"] = hosts

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(existing, indent=4) + "\n")
    print(f"Wrote {sum(len(b) for b in ops.values())} (op, backend) entries "
          f"for host '{host}' to {baseline_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
