#!/usr/bin/env python3
"""Compare two benchmark JSON files and report regressions.

Usage:
    python scripts/compare_benchmarks.py baseline.json current.json

Compares each benchmark by name, computing speedup/regression percentage.
Flags >5% regressions with WARNING and exits with code 1 if any >7%
regression is found (override with --threshold).

Output format: Markdown table suitable for CI comments.
"""

import json
import sys
from pathlib import Path


WARNING_THRESHOLD = 5.0    # Percentage regression to flag as WARNING
FAILURE_THRESHOLD = 7.0    # Percentage regression to fail the build


def load_benchmarks(filepath: str) -> dict[str, dict]:
    """Load benchmark JSON and return dict keyed by benchmark name."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(2)

    with open(path) as f:
        data = json.load(f)

    benchmarks = {}

    # Handle merged all.json format (has "suites" array)
    if "suites" in data:
        for suite in data["suites"]:
            suite_name = suite.get("suite_name", "unknown")
            for bench in suite.get("benchmarks", []):
                key = f"{suite_name}/{bench['name']}"
                benchmarks[key] = bench
    # Handle single suite format (has "benchmarks" array)
    elif "benchmarks" in data:
        suite_name = data.get("suite_name", "unknown")
        for bench in data["benchmarks"]:
            key = f"{suite_name}/{bench['name']}"
            benchmarks[key] = bench
    # Handle benchmarks/python run_benchmarks.py output: {"system_info": ...,
    # "results": [{"name": ..., "category": ..., "device": ..., "framework": ...,
    # "mean_ms": ...}, ...]}. mean_ms is flat (no "stats" subdict), so wrap
    # each entry in a synthetic {"stats": {"mean_ms": ...}} so get_mean_ms()
    # works uniformly.
    elif "results" in data and isinstance(data["results"], list):
        for bench in data["results"]:
            name = bench.get("name", "")
            category = bench.get("category", "")
            device = bench.get("device", "")
            framework = bench.get("framework", "")
            key_parts = [p for p in (category, name, device, framework) if p]
            key = "/".join(key_parts)
            wrapped = dict(bench)
            wrapped["stats"] = {"mean_ms": bench.get("mean_ms", 0.0)}
            benchmarks[key] = wrapped
    # Handle flat array of results
    elif isinstance(data, list):
        for bench in data:
            benchmarks[bench["name"]] = bench
    else:
        print(f"Error: Unrecognized JSON format in {filepath}", file=sys.stderr)
        sys.exit(2)

    return benchmarks


def get_mean_ms(bench: dict) -> float:
    """Extract mean time in milliseconds from a benchmark entry."""
    if "stats" in bench:
        return bench["stats"].get("mean_ms", 0.0)
    return 0.0


def compare(baseline: dict, current: dict) -> tuple[list[dict], bool]:
    """Compare baseline vs current benchmarks.

    Returns:
        Tuple of (comparison results list, has_failure bool)
    """
    results = []
    has_failure = False

    # Find all benchmark names (union of both sets)
    all_names = sorted(set(baseline.keys()) | set(current.keys()))

    for name in all_names:
        entry = {"name": name}

        if name not in baseline:
            entry["status"] = "NEW"
            entry["current_ms"] = get_mean_ms(current[name])
            entry["baseline_ms"] = None
            entry["change_pct"] = None
            results.append(entry)
            continue

        if name not in current:
            entry["status"] = "REMOVED"
            entry["baseline_ms"] = get_mean_ms(baseline[name])
            entry["current_ms"] = None
            entry["change_pct"] = None
            results.append(entry)
            continue

        base_ms = get_mean_ms(baseline[name])
        curr_ms = get_mean_ms(current[name])

        entry["baseline_ms"] = base_ms
        entry["current_ms"] = curr_ms

        if base_ms <= 0:
            entry["status"] = "N/A"
            entry["change_pct"] = None
            results.append(entry)
            continue

        # Positive change_pct means regression (slower)
        change_pct = ((curr_ms - base_ms) / base_ms) * 100.0
        entry["change_pct"] = change_pct

        if change_pct > FAILURE_THRESHOLD:
            entry["status"] = "REGRESSION"
            has_failure = True
        elif change_pct > WARNING_THRESHOLD:
            entry["status"] = "WARNING"
        elif change_pct < -WARNING_THRESHOLD:
            entry["status"] = "IMPROVED"
        else:
            entry["status"] = "OK"

        results.append(entry)

    return results, has_failure


def format_markdown(results: list[dict], has_failure: bool) -> str:
    """Format comparison results as a Markdown table."""
    lines = []

    if has_failure:
        lines.append(
            f"**Performance regression detected (>{FAILURE_THRESHOLD:g}%).** "
            "Review the benchmarks below.\n"
        )

    lines.append("| Benchmark | Baseline (ms) | Current (ms) | Change | Status |")
    lines.append("|:----------|-------------:|------------:|---------:|:-------|")

    for entry in results:
        name = entry["name"]

        if entry["baseline_ms"] is not None:
            base_str = f"{entry['baseline_ms']:.3f}"
        else:
            base_str = "-"

        if entry["current_ms"] is not None:
            curr_str = f"{entry['current_ms']:.3f}"
        else:
            curr_str = "-"

        if entry["change_pct"] is not None:
            pct = entry["change_pct"]
            sign = "+" if pct > 0 else ""
            change_str = f"{sign}{pct:.1f}%"
        else:
            change_str = "-"

        status = entry["status"]
        if status == "REGRESSION":
            status_str = "REGRESSION"
        elif status == "WARNING":
            status_str = "WARNING"
        elif status == "IMPROVED":
            status_str = "IMPROVED"
        elif status == "NEW":
            status_str = "new"
        elif status == "REMOVED":
            status_str = "removed"
        else:
            status_str = "ok"

        lines.append(
            f"| {name} | {base_str} | {curr_str} | {change_str} | {status_str} |"
        )

    # Summary
    total = len(results)
    regressions = sum(1 for r in results if r["status"] == "REGRESSION")
    warnings = sum(1 for r in results if r["status"] == "WARNING")
    improved = sum(1 for r in results if r["status"] == "IMPROVED")

    lines.append("")
    lines.append(
        f"**Summary:** {total} benchmarks | "
        f"{regressions} regressions | "
        f"{warnings} warnings | "
        f"{improved} improved"
    )

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("baseline", help="Baseline benchmark JSON file")
    parser.add_argument("current", help="Current benchmark JSON file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override failure threshold (percent regression)")
    args = parser.parse_args()

    if args.threshold is not None:
        global FAILURE_THRESHOLD
        FAILURE_THRESHOLD = args.threshold

    baseline = load_benchmarks(args.baseline)
    current = load_benchmarks(args.current)

    if not baseline:
        print("Warning: Baseline has no benchmarks.", file=sys.stderr)
    if not current:
        print("Warning: Current has no benchmarks.", file=sys.stderr)

    results, has_failure = compare(baseline, current)
    markdown = format_markdown(results, has_failure)

    print(markdown)

    if has_failure:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
