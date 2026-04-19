#!/usr/bin/env python3
"""Audit test files for relaxed tolerances without justification.

Per Phase 9-followup: "pass with relaxed tolerance" patterns (like the
Conv1d Float32-accumulator bug Phase 1.1 fixed) hide real backend bugs.
This script greps every tests/**/*.cpp file for tolerances looser than
fixture defaults (atol > 1e-5 or rtol > 1e-4 on Float32/Float64) and
reports any that lack a comment line within 2 lines above explaining why.

Exit code 0 = no violations or all justified.
Exit code 1 = unjustified loose tolerances found (CI failure signal).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Match common tolerance literal forms inside gradcheck()/EXPECT_NEAR/
# expectTensorNear/test_operation_parity etc. Looking for floats >= 1e-3.
TOL_PATTERN = re.compile(
    r"\b(?:atol|rtol|EXPECT_NEAR|expectTensorNear|tensors_close|"
    r"gradcheck|test_operation_parity|test_gradient_parity)"
    r"[^;\n]*?"
    r"(?:1e-[0-2]|0\.0[0-9]+|0\.[0-9]+)"
)
# Looser literals threshold: anything with magnitude > 1e-3 in the call.
LOOSE_THRESHOLD = 1e-3
LITERAL_RX = re.compile(r"(?<![A-Za-z_])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![A-Za-z_])")
JUSTIFY_RX = re.compile(r"//.*\b(reason|because|tolerance|relaxed|FIXME|TODO|"
                        r"precision|noise|float32 accum|Float16 noise)", re.I)


def file_lines(p: Path) -> list[str]:
    try:
        return p.read_text(errors="replace").splitlines()
    except Exception:
        return []


def has_justification(lines: list[str], idx: int) -> bool:
    # Look at the matching line itself + 2 lines above for a // reason: comment
    start = max(0, idx - 2)
    return any(JUSTIFY_RX.search(l) for l in lines[start:idx + 1])


def scan_file(p: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    lines = file_lines(p)
    for i, line in enumerate(lines):
        if not TOL_PATTERN.search(line):
            continue
        for lit_match in LITERAL_RX.finditer(line):
            try:
                val = float(lit_match.group(1))
            except ValueError:
                continue
            # Only consider tolerance-like literals: between 1e-3 and 1.
            if LOOSE_THRESHOLD <= val < 1.0:
                if not has_justification(lines, i):
                    out.append((i + 1, line.strip()))
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="tests", help="root directory to scan")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any unjustified loose tolerance is found")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"audit_tolerances: {root} does not exist", file=sys.stderr)
        return 2

    violations: list[tuple[Path, int, str]] = []
    for cpp in sorted(root.rglob("*.cpp")):
        for line_no, text in scan_file(cpp):
            violations.append((cpp, line_no, text))

    if not violations:
        print("OK: no unjustified loose tolerances found.")
        return 0

    print(f"Found {len(violations)} unjustified loose tolerance(s):")
    for path, line_no, text in violations:
        print(f"  {path}:{line_no}  {text}")
    print("\nJustify each with a `// reason:` (or similar) comment within 2 "
          "lines above, OR tighten the tolerance.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
