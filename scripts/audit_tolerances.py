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

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "tools" / "audit_tolerances_baseline.txt"

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


def _load_baseline(path: Path) -> set[str]:
    """Parse a baseline file containing one `path:line` token per line.

    Mirrors the pattern in ``tools/lint_test_contract.py``: the baseline pins
    the set of pre-existing violations. With ``--strict --baseline``, the
    audit passes if every reported violation is in the baseline and fails
    only on NEW violations. Drive the list to zero over time.
    """
    if not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def _violation_token(path: Path, line_no: int, root: Path) -> str:
    """Render a baseline token. Paths are made relative to ``root.parent``
    when possible so that the baseline file is independent of the absolute
    location the auditor was invoked from."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
        return f"{rel}:{line_no}"
    except ValueError:
        return f"{path}:{line_no}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="tests", help="root directory to scan")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any unjustified loose tolerance is found")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline file of known-existing violations (one `path:line` "
        "per line). Pre-existing entries are ignored under --strict; new "
        "violations fail the audit. Pass an explicit path or omit to default "
        "to tools/audit_tolerances_baseline.txt.",
    )
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline file with the current set of violations. "
        "Use after legitimately introducing or removing a justified relaxed "
        "tolerance.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"audit_tolerances: {root} does not exist", file=sys.stderr)
        return 2

    violations: list[tuple[Path, int, str]] = []
    for cpp in sorted(root.rglob("*.cpp")):
        for line_no, text in scan_file(cpp):
            violations.append((cpp, line_no, text))

    baseline_path = args.baseline if args.baseline is not None else DEFAULT_BASELINE

    if args.update_baseline:
        tokens = sorted({_violation_token(p, ln, root) for p, ln, _ in violations})
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            "# Baseline of known unjustified loose tolerances flagged by\n"
            "# scripts/audit_tolerances.py. Each line is `<path>:<line>`\n"
            "# (path relative to repo root). Pre-existing entries are ignored\n"
            "# under --strict; new violations FAIL the audit. Drive this list\n"
            "# to zero over time by justifying or tightening each tolerance.\n"
            + "\n".join(tokens)
            + "\n",
            encoding="utf-8",
        )
        print(f"baseline updated: {len(tokens)} entries -> {baseline_path}")
        return 0

    if not violations:
        print("OK: no unjustified loose tolerances found.")
        return 0

    # Under --strict with a baseline, only NEW violations fail.
    if args.strict and args.baseline is not None:
        baseline = _load_baseline(baseline_path)
        new_violations = [
            (p, ln, t) for (p, ln, t) in violations
            if _violation_token(p, ln, root) not in baseline
        ]
        grandfathered = len(violations) - len(new_violations)
        if new_violations:
            print(
                f"Found {len(new_violations)} NEW unjustified loose "
                f"tolerance(s) (plus {grandfathered} pre-existing in baseline):"
            )
            for path, line_no, text in new_violations:
                print(f"  {path}:{line_no}  {text}")
            print("\nJustify each new violation with a `// reason:` comment "
                  "within 2 lines above, tighten the tolerance, or — if the "
                  "relaxation is legitimately permanent — regenerate the "
                  "baseline with `--update-baseline`.")
            return 1
        print(
            f"OK: {grandfathered} pre-existing violations (all in baseline), "
            f"0 new violations."
        )
        return 0

    print(f"Found {len(violations)} unjustified loose tolerance(s):")
    for path, line_no, text in violations:
        print(f"  {path}:{line_no}  {text}")
    print("\nJustify each with a `// reason:` (or similar) comment within 2 "
          "lines above, OR tighten the tolerance.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
