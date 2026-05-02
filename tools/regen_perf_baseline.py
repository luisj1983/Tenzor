#!/usr/bin/env python3
"""
Regenerate the performance baseline JSON consumed by
tests/backend_parity/test_performance_regression.cpp.

Workflow:
    1. Build `test_performance_regression`.
    2. Run it — the existing TEST(PerformanceRegression, ...) cases print
       per-backend timings to stdout in lines like:
            [backend=cpu] MatMul 256x256: median=0.84ms p99=1.10ms
       (The print format is what this script parses.)
    3. Parse stdout, write the JSON in tests/backend_parity/baselines/.

The regression check (BaselineRegression_MatMul, etc.) reads this JSON,
compares the current run's timings to the recorded ones, and fails when
the current run exceeds the threshold (env-tunable):
    TENZOR_PERF_REGRESSION_RTOL      — default 1.25 on median
    TENZOR_PERF_REGRESSION_P99_RTOL  — default 1.50 on p99

Usage:
    cd <repo_root>
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
#   [backend=cuda:0] MatMul 256x256: median=0.84ms p99=1.10ms
LINE_RE = re.compile(
    r"^\[backend=(?P<backend>[A-Za-z0-9_:]+)\]\s+"
    r"(?P<op>[A-Za-z0-9_]+)\s+"
    r"(?P<size>[A-Za-z0-9_]+):\s+"
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
    p.add_argument("--executable",
                   default="bin/test_performance_regression",
                   help="Path to the test binary, relative to repo root")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / args.executable
    baseline_path = repo_root / args.baseline

    if not bin_path.exists():
        print(f"perf binary not found at {bin_path}; build it first with"
              f" `ninja -C {args.build_dir} test_performance_regression`",
              file=sys.stderr)
        return 1

    print(f"Running {bin_path} ...", flush=True)
    completed = subprocess.run([str(bin_path)],
                               capture_output=True, text=True, check=False)
    print(completed.stdout)
    if completed.returncode != 0:
        print(f"perf binary exited with {completed.returncode}; baseline NOT regenerated.",
              file=sys.stderr)
        return completed.returncode

    ops: dict[str, dict[str, dict[str, float]]] = {}
    for line in completed.stdout.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        key = f"{m.group('op')}_{m.group('size')}"
        ops.setdefault(key, {})[m.group('backend')] = {
            "median_ms": float(m.group('median')),
            "p99_ms":   float(m.group('p99')),
        }

    if not ops:
        print("No timing lines parsed — perf test output format may have changed.",
              file=sys.stderr)
        print("Expected format:", LINE_RE.pattern, file=sys.stderr)
        return 1

    # Read existing baseline (preserves _comment fields).
    if baseline_path.exists():
        try:
            existing = json.loads(baseline_path.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}

    existing["host"] = platform.node()
    existing["tenzor_version"] = os.environ.get("TENZOR_VERSION", "unknown")
    existing["captured_at_iso8601"] = _dt.datetime.utcnow().isoformat() + "Z"
    existing["ops"] = ops

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(existing, indent=4) + "\n")
    print(f"Wrote {sum(len(b) for b in ops.values())} (op, backend) entries to "
          f"{baseline_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
