#!/usr/bin/env python3
"""
Coverage summarizer for Tenzor.

Walks a coverage-instrumented build directory, runs gcov in JSON mode on
every .gcda file, aggregates per-source line coverage, filters per the
project's .codecov.yml ignore list, and prints a concise report:

  - Overall line coverage %
  - Per-subsystem breakdown (src/core, src/nn, src/ops, src/backends/*, ...)
  - Top-N files by uncovered-line count (where the work is)
  - Files that have a .gcno but no .gcda (compiled but never executed)

A JSON dump of the per-file data is written next to the report for
downstream tooling. Designed to run with the bare `gcov` Arch installs;
no lcov / gcovr dependency.

Usage:
  scripts/coverage_summary.py [--build-dir build-cov] [--out audit/...]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .codecov.yml ignore: tests/**, benchmarks/**, examples/**, docs/**, build/**
IGNORE_DIRS = ("tests/", "benchmarks/", "examples/", "docs/", "build/", "build-cov/", "_deps/")


def is_ignored(rel: str) -> bool:
    if rel.startswith("/usr/") or rel.startswith("/opt/"):
        return True
    return any(rel.startswith(p) for p in IGNORE_DIRS)


def subsystem(rel: str) -> str:
    """Bucket a source path under src/ into a coarse subsystem."""
    if not rel.startswith("src/"):
        return "(other)"
    parts = rel.split("/")
    # src/backends/cpu/kernels/foo.cpp -> src/backends/cpu
    if len(parts) >= 4 and parts[1] == "backends":
        return "src/backends/" + parts[2]
    # src/nn/layers/foo.cpp -> src/nn
    if len(parts) >= 3:
        return "src/" + parts[1]
    return "src"


def run_gcov_json(gcda: Path) -> Path | None:
    """Run gcov -j on a .gcda; returns path to the .gcov.json.gz produced.

    gcov writes the JSON file into the cwd, named after the source file.
    """
    cwd = gcda.parent
    # gcov can take the .gcda or just the basename without extension.
    # No -r: that filters out sources not under cwd, which excludes
    # everything since CMake puts gcda files under build/CMakeFiles/...
    cmd = ["gcov", "-j", "-b", gcda.name]
    try:
        subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=False, timeout=30
        )
    except subprocess.TimeoutExpired:
        return None
    return cwd  # multiple .gcov.json.gz may appear; caller scans cwd


def parse_gcov_json_gz(path: Path):
    """Yield (source_file, executable_lines, executed_lines) tuples."""
    try:
        with gzip.open(path, "rt") as f:
            data = json.load(f)
    except Exception:
        return
    for f_entry in data.get("files", []):
        src = f_entry.get("file", "")
        executable = 0
        executed = 0
        for ln in f_entry.get("lines", []):
            # gcc-16 schema: {"line_number", "count", "unexecuted_block", ...}
            if ln.get("count", 0) >= 0 and not ln.get("gcc_only", False):
                executable += 1
                if ln.get("count", 0) > 0:
                    executed += 1
        yield src, executable, executed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build-cov", type=Path)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "audit" / "coverage_summary.md")
    ap.add_argument("--json", type=Path, default=None,
                    help="JSON dump path (default: <out>.json)")
    args = ap.parse_args()

    build = (PROJECT_ROOT / args.build_dir).resolve() if not args.build_dir.is_absolute() else args.build_dir
    if not build.is_dir():
        print(f"ERROR: build dir not found: {build}", file=sys.stderr)
        return 2

    json_out = args.json or args.out.with_suffix(".json")

    # --- collect .gcda files --------------------------------------------------
    gcda_files = list(build.rglob("*.gcda"))
    gcno_files = list(build.rglob("*.gcno"))
    print(f"[coverage] found {len(gcda_files)} .gcda files in {build}", file=sys.stderr)
    print(f"[coverage] found {len(gcno_files)} .gcno files in {build}", file=sys.stderr)

    if not gcda_files:
        print("ERROR: no .gcda files; did you run any tests?", file=sys.stderr)
        return 3

    # --- run gcov in JSON mode on each ----------------------------------------
    visited_dirs = set()
    for i, gcda in enumerate(gcda_files):
        if gcda.parent in visited_dirs:
            continue
        # Run gcov per-directory rather than per-file to amortize cost
        # gcov picks up all .gcda in cwd by default if you pass them.
        visited_dirs.add(gcda.parent)
        # Run gcov on every .gcda in this directory in one invocation
        names = [g.name for g in gcda.parent.glob("*.gcda")]
        subprocess.run(
            ["gcov", "-j", "-b", *names],
            cwd=gcda.parent, capture_output=True, text=True, check=False,
        )
        if (i + 1) % 50 == 0:
            print(f"[coverage] processed {i+1}/{len(gcda_files)}", file=sys.stderr)

    # --- parse all .gcov.json.gz ----------------------------------------------
    # Aggregate per source file across all object dirs (a single .cpp can
    # appear in multiple .gcda when used by multiple targets).
    src_executable: dict[str, int] = {}
    src_executed: dict[str, int] = {}
    json_files = list(build.rglob("*.gcov.json.gz"))
    print(f"[coverage] parsing {len(json_files)} gcov json files", file=sys.stderr)
    for jf in json_files:
        for src, executable, executed in parse_gcov_json_gz(jf):
            # Normalize to project-relative path
            src_path = Path(src)
            try:
                rel = str(src_path.resolve().relative_to(PROJECT_ROOT))
            except ValueError:
                rel = src
            if is_ignored(rel):
                continue
            # Use the MAX coverage seen for this source file across all
            # objects (different objects may instantiate different
            # template specializations with different reachable lines).
            prev_executable = src_executable.get(rel, 0)
            prev_executed = src_executed.get(rel, 0)
            if executable > prev_executable or (
                executable == prev_executable and executed > prev_executed
            ):
                src_executable[rel] = executable
                src_executed[rel] = executed

    # --- compute per-file and overall ----------------------------------------
    per_file = []
    total_exec = 0
    total_run = 0
    for rel, ex in src_executable.items():
        ru = src_executed[rel]
        total_exec += ex
        total_run += ru
        pct = (ru / ex * 100.0) if ex else 0.0
        per_file.append((rel, ex, ru, pct))
    overall_pct = (total_run / total_exec * 100.0) if total_exec else 0.0

    # --- per-subsystem --------------------------------------------------------
    sub_exec: dict[str, int] = defaultdict(int)
    sub_run: dict[str, int] = defaultdict(int)
    sub_files: dict[str, int] = defaultdict(int)
    for rel, ex, ru, _ in per_file:
        s = subsystem(rel)
        sub_exec[s] += ex
        sub_run[s] += ru
        sub_files[s] += 1

    # --- gcno-without-gcda (compiled but never executed) ---------------------
    # Match by stem under build dir.
    gcda_stems = {g.with_suffix("").name for g in gcda_files}
    never_run = []
    for gcno in gcno_files:
        if gcno.with_suffix("").name not in gcda_stems:
            # Best-effort: derive source file from the .gcno path.
            never_run.append(str(gcno.relative_to(build)))

    # --- write report ---------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Tenzor Coverage Summary")
    lines.append("")
    lines.append(f"Build directory: `{build.relative_to(PROJECT_ROOT) if str(build).startswith(str(PROJECT_ROOT)) else build}`")
    lines.append(f"Source files measured: {len(per_file)}")
    lines.append(f"`.gcda` files (executed objects): {len(gcda_files)}")
    lines.append(f"`.gcno` files (compiled objects): {len(gcno_files)}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"**Line coverage: {total_run} / {total_exec} = {overall_pct:.2f}%**")
    lines.append("")
    lines.append("## Per-subsystem")
    lines.append("")
    lines.append("| Subsystem | Files | Executable | Executed | Coverage |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in sorted(sub_exec.keys()):
        ex, ru, n = sub_exec[s], sub_run[s], sub_files[s]
        pct = (ru / ex * 100.0) if ex else 0.0
        lines.append(f"| `{s}` | {n} | {ex} | {ru} | {pct:.1f}% |")
    lines.append("")

    # Sort by uncovered-line count descending — that's where the work is.
    per_file_uncov = sorted(per_file, key=lambda r: -(r[1] - r[2]))
    lines.append(f"## Top {args.top} files by uncovered-line count")
    lines.append("")
    lines.append("| File | Executable | Executed | Coverage | Uncovered |")
    lines.append("|---|---:|---:|---:|---:|")
    for rel, ex, ru, pct in per_file_uncov[: args.top]:
        lines.append(f"| `{rel}` | {ex} | {ru} | {pct:.1f}% | {ex - ru} |")
    lines.append("")

    # Files with 0% coverage
    zero_cov = [r for r in per_file if r[2] == 0 and r[1] > 0]
    zero_cov.sort(key=lambda r: -r[1])
    lines.append(f"## Files with 0% line coverage ({len(zero_cov)})")
    lines.append("")
    if zero_cov:
        lines.append("| File | Executable lines |")
        lines.append("|---|---:|")
        for rel, ex, _, _ in zero_cov[: args.top]:
            lines.append(f"| `{rel}` | {ex} |")
        if len(zero_cov) > args.top:
            lines.append(f"| _… and {len(zero_cov) - args.top} more_ |  |")
    else:
        lines.append("_(none)_")
    lines.append("")

    # Compiled but never executed (no .gcda)
    if never_run:
        lines.append(f"## Compiled objects with no execution data ({len(never_run)})")
        lines.append("")
        lines.append("These have a `.gcno` from compilation but no `.gcda` — the linker pulled them in but no test exercised the code.")
        lines.append("")
        for rel in sorted(never_run)[:30]:
            lines.append(f"- `{rel}`")
        if len(never_run) > 30:
            lines.append(f"- _… and {len(never_run) - 30} more_")
        lines.append("")

    args.out.write_text("\n".join(lines))
    print(f"[coverage] wrote {args.out}", file=sys.stderr)

    # JSON dump for downstream tools
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w") as f:
        json.dump(
            {
                "build_dir": str(build),
                "overall": {
                    "executable_lines": total_exec,
                    "executed_lines": total_run,
                    "percent": overall_pct,
                },
                "per_subsystem": {
                    s: {
                        "files": sub_files[s],
                        "executable": sub_exec[s],
                        "executed": sub_run[s],
                        "percent": (sub_run[s] / sub_exec[s] * 100.0) if sub_exec[s] else 0.0,
                    }
                    for s in sub_exec
                },
                "per_file": [
                    {"path": r, "executable": ex, "executed": ru, "percent": pct}
                    for r, ex, ru, pct in per_file
                ],
                "never_executed": never_run,
            },
            f,
            indent=2,
        )
    print(f"[coverage] wrote {json_out}", file=sys.stderr)

    # Print one-line summary to stdout
    print(f"OVERALL: {total_run}/{total_exec} = {overall_pct:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
