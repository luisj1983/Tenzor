#!/usr/bin/env python3
"""Compare benchmark results against a baseline to detect performance regressions.

Usage:
    python compare_results.py baseline.json current.json [--threshold 10]

Exits with code 1 if any benchmark regresses beyond the threshold.
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(filepath: str) -> dict:
    """Load benchmark results as a dict keyed by (name, device, category)."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    results = {}
    for r in data.get("results", []):
        key = (r["name"], r.get("device", ""), r.get("category", ""))
        results[key] = r
    return results


def compare(baseline_path: str, current_path: str, threshold: float) -> bool:
    """Compare current results against baseline. Returns True if all pass."""
    baseline = load_results(baseline_path)
    current = load_results(current_path)

    regressions = []
    improvements = []
    unchanged = []
    missing = []

    for key, base_result in baseline.items():
        name, device, category = key
        label = f"{name} [{device}/{category}]"

        if key not in current:
            missing.append(label)
            continue

        curr_result = current[key]
        base_ms = base_result["mean_ms"]
        curr_ms = curr_result["mean_ms"]

        if base_ms <= 0:
            continue

        delta_pct = ((curr_ms - base_ms) / base_ms) * 100.0

        entry = {
            "label": label,
            "baseline_ms": base_ms,
            "current_ms": curr_ms,
            "delta_pct": delta_pct,
        }

        if delta_pct > threshold:
            regressions.append(entry)
        elif delta_pct < -threshold:
            improvements.append(entry)
        else:
            unchanged.append(entry)

    # Print report
    print(f"Benchmark Comparison Report")
    print(f"  Baseline: {baseline_path}")
    print(f"  Current:  {current_path}")
    print(f"  Threshold: {threshold:.1f}%")
    print(f"  Benchmarks compared: {len(baseline)}")
    print()

    if regressions:
        print(f"REGRESSIONS ({len(regressions)}):")
        for r in sorted(regressions, key=lambda x: -x["delta_pct"]):
            print(f"  {r['label']}: {r['baseline_ms']:.3f}ms -> {r['current_ms']:.3f}ms "
                  f"(+{r['delta_pct']:.1f}%)")
        print()

    if improvements:
        print(f"Improvements ({len(improvements)}):")
        for r in sorted(improvements, key=lambda x: x["delta_pct"]):
            print(f"  {r['label']}: {r['baseline_ms']:.3f}ms -> {r['current_ms']:.3f}ms "
                  f"({r['delta_pct']:.1f}%)")
        print()

    if missing:
        print(f"Missing from current ({len(missing)}):")
        for label in missing:
            print(f"  {label}")
        print()

    print(f"Summary: {len(unchanged)} unchanged, {len(improvements)} improved, "
          f"{len(regressions)} regressed, {len(missing)} missing")

    if regressions:
        print(f"\nFAILED: {len(regressions)} benchmark(s) regressed beyond {threshold:.1f}% threshold")
        return False

    print("\nPASSED: No regressions detected")
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results against baseline")
    parser.add_argument("baseline", help="Path to baseline results JSON")
    parser.add_argument("current", help="Path to current results JSON")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Regression threshold in percent (default: 10)")
    args = parser.parse_args()

    if not Path(args.baseline).exists():
        print(f"Error: Baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.current).exists():
        print(f"Error: Current file not found: {args.current}", file=sys.stderr)
        sys.exit(1)

    passed = compare(args.baseline, args.current, args.threshold)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
