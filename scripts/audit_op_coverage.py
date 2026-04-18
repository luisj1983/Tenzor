#!/usr/bin/env python3
"""
Audit OpId × test coverage.

Parses `include/tenzor/ops/op_id.hpp` to discover every enum member of
`enum class OpId`, then greps `tests/` for each name. Emits a coverage
matrix (stdout and optional JSON) and can fail CI with --require-all when
any OpId has zero test references.

Usage:
  scripts/audit_op_coverage.py
  scripts/audit_op_coverage.py --json coverage.json
  scripts/audit_op_coverage.py --require-all
  scripts/audit_op_coverage.py --list-untested
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OP_ID_HPP = REPO_ROOT / "include" / "tenzor" / "ops" / "op_id.hpp"
TESTS_DIR = REPO_ROOT / "tests"

# Match enum members:
#   Name,
#   Name = 123,
# while ignoring comments and the `OP_COUNT` sentinel.
ENUM_MEMBER_RE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9_]*)\s*(?:=\s*[0-9]+)?\s*,\s*(?://.*)?$"
)


def parse_opids() -> list[str]:
    text = OP_ID_HPP.read_text()
    # Bound scan between `enum class OpId` and its closing brace.
    m = re.search(r"enum class OpId\b[^{]*\{(.*?)\};", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not find `enum class OpId` in {OP_ID_HPP}")
    body = m.group(1)
    out = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            continue
        hit = ENUM_MEMBER_RE.match(line)
        if hit:
            name = hit.group(1)
            if name == "OP_COUNT":
                continue
            out.append(name)
    return out


def grep_tests(name: str) -> int:
    """Count references to an OpId name in tests/ (cheap word-boundary grep)."""
    try:
        # Use ripgrep if available, else grep -r.
        proc = subprocess.run(
            ["grep", "-r", "-l", "-w", f"OpId::{name}\\|{name}", str(TESTS_DIR)],
            capture_output=True, text=True, timeout=30,
        )
        files = [l for l in proc.stdout.splitlines() if l]
        return len(files)
    except Exception:
        return 0


def grep_tests_word(name: str) -> int:
    """Word-boundary grep for bare identifier. False positives possible for
    common names (e.g. `Add`, `Mul`) but still useful as a coarse signal."""
    proc = subprocess.run(
        ["grep", "-r", "-l", "-w", name, str(TESTS_DIR)],
        capture_output=True, text=True, timeout=30,
    )
    return len([l for l in proc.stdout.splitlines() if l])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write coverage JSON to this path")
    ap.add_argument("--require-all", action="store_true",
                    help="exit nonzero if any OpId has zero test references")
    ap.add_argument("--list-untested", action="store_true",
                    help="only print OpIds with zero references")
    args = ap.parse_args()

    ops = parse_opids()
    ops = sorted(set(ops))
    coverage: dict[str, int] = {}
    for name in ops:
        coverage[name] = grep_tests_word(name)

    untested = [n for n, c in coverage.items() if c == 0]
    total = len(ops)
    covered = total - len(untested)

    if args.json:
        args.json.write_text(json.dumps(
            {"total": total, "covered": covered, "untested": untested, "coverage": coverage},
            indent=2,
        ))

    if args.list_untested:
        for n in untested:
            print(n)
    else:
        print(f"OpId coverage: {covered}/{total} ({covered*100//total}%)")
        if untested:
            print(f"\nUntested OpIds ({len(untested)}):")
            for n in untested:
                print(f"  {n}")

    if args.require_all and untested:
        print(f"\nFAIL: {len(untested)} OpIds have no test references", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
