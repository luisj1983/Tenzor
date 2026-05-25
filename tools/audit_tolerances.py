#!/usr/bin/env python3
"""Audit test files for relaxed tolerances without justification.

Per Phase 9-followup: "pass with relaxed tolerance" patterns (like the
Conv1d Float32-accumulator bug Phase 1.1 fixed) hide real backend bugs.
This script greps every tests/**/*.cpp file for tolerances looser than
fixture defaults (atol > 1e-5 or rtol > 1e-4 on Float32/Float64) and
reports any that lack a comment line within 2 lines above explaining why.

EE.18: baseline keys are normalised to ``path|function|value`` rather than
``path:line``. Test-file edits that only shift line numbers no longer
invalidate the baseline; only adding a new unjustified relaxed tolerance
(or changing its value) registers as a new violation.

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
    r"\b(?:atol|rtol|EXPECT_NEAR|ASSERT_NEAR|expectTensorNear|tensors_close|"
    r"gradcheck|test_operation_parity|test_gradient_parity)"
    r"[^;\n]*?"
    r"(?:1e-[0-2]|0\.0[0-9]+|0\.[0-9]+)"
)
# Looser literals threshold: anything with magnitude > 1e-3 in the call.
LOOSE_THRESHOLD = 1e-3
LITERAL_RX = re.compile(r"(?<![A-Za-z_])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![A-Za-z_])")
JUSTIFY_RX = re.compile(r"//.*\b(reason|because|tolerance|relaxed|FIXME|TODO|"
                        r"precision|noise|float32 accum|Float16 noise)", re.I)

# EE.18: extract the enclosing TEST(...) / TEST_P(...) / TEST_F(...) /
# TYPED_TEST(...) / TYPED_TEST_P(...) macro to attribute each violation
# to a stable test name. Falling back to "<file-scope>" when no enclosing
# macro is found (rare; usually helper-function bodies).
TEST_MACRO_RX = re.compile(
    r"^\s*(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P|TEST_CASE)\s*\(\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)


def file_lines(p: Path) -> list[str]:
    try:
        return p.read_text(errors="replace").splitlines()
    except Exception:
        return []


def has_justification(lines: list[str], idx: int) -> bool:
    # Look at the matching line itself + 2 lines above for a // reason: comment
    start = max(0, idx - 2)
    return any(JUSTIFY_RX.search(l) for l in lines[start:idx + 1])


def nearest_test_name(lines: list[str], idx: int) -> str:
    """Walk back from ``idx`` to find the enclosing TEST(...) / TEST_P(...)
    / TEST_F(...) / TYPED_TEST(...) macro and return ``Fixture.Case`` as the
    key component. Returns ``<file-scope>`` if no macro is found above.

    EE.18: this is the stable identifier we key the baseline on, so a small
    edit to a test (adding or removing lines above the EXPECT_NEAR) does not
    invalidate the entry the way ``file:line`` did.
    """
    for back in range(idx, -1, -1):
        m = TEST_MACRO_RX.match(lines[back])
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    return "<file-scope>"


_NEAR_CALL_RX = re.compile(
    r"\b(EXPECT_NEAR|ASSERT_NEAR|DOUBLES_EQUAL)\s*\("
)


def _split_top_level_args(arglist: str) -> list[str]:
    """Split a comma-separated argument list, respecting parens / brackets /
    template depth. Used to pick out arg[2] (the tolerance) from
    EXPECT_NEAR(value, expected, atol).
    """
    args: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in arglist:
        if ch in "([{<":
            depth += 1
            current.append(ch)
        elif ch in ")]}>":
            depth -= 1
            current.append(ch)
            if depth < 0:
                break
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    return args


def _literal_in(s: str) -> float | None:
    """Pick the first loose-magnitude numeric literal in ``s``."""
    for lit_match in LITERAL_RX.finditer(s):
        try:
            val = float(lit_match.group(1))
        except ValueError:
            continue
        if LOOSE_THRESHOLD <= val < 1.0:
            return val
    return None


def _loose_literal(line: str) -> float | None:
    """Return the numeric literal in ``line`` that represents the tolerance
    threshold (not the expected value), if it exceeds ``LOOSE_THRESHOLD``.

    HH.24: previously returned the first loose literal anywhere on the line,
    which for ``EXPECT_NEAR(x, expected, atol)`` mistakenly picked ``expected``
    when it happened to fall in [1e-3, 1.0). For EXPECT_NEAR / ASSERT_NEAR /
    DOUBLES_EQUAL we now parse the call's top-level arguments and inspect the
    3rd argument (the tolerance). For other matched macros (EXPECT_LT,
    gradcheck, ...) we keep the original first-literal heuristic.
    """
    m = _NEAR_CALL_RX.search(line)
    if m:
        # Walk forward from the opening paren to find the matching close,
        # then split the inner arg list at the top level.
        start = m.end()  # position right after the '('
        depth = 1
        i = start
        while i < len(line) and depth > 0:
            c = line[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        inner = line[start:i]
        args = _split_top_level_args(inner)
        if len(args) >= 3:
            return _literal_in(args[2])
        # If the call spans multiple lines we won't see args[2] here.
        # Fall through to the legacy first-literal heuristic so we still
        # emit *some* signal rather than silently dropping the line.
    return _literal_in(line)


def _format_value(value: float) -> str:
    """Normalise the tolerance literal so reformatting (e.g. ``0.01`` vs
    ``1e-2``) does not invalidate the baseline. Uses ``%.6g`` which gives a
    canonical short representation."""
    return f"{value:.6g}"


def scan_file(p: Path) -> list[tuple[int, str, str, float]]:
    """Return ``(line_no, text, test_name, value)`` for every unjustified
    loose-tolerance hit in ``p``. ``test_name`` comes from
    ``nearest_test_name``; ``value`` is the first loose literal on the line.
    """
    out: list[tuple[int, str, str, float]] = []
    lines = file_lines(p)
    for i, line in enumerate(lines):
        if not TOL_PATTERN.search(line):
            continue
        val = _loose_literal(line)
        if val is None:
            continue
        if has_justification(lines, i):
            continue
        out.append((i + 1, line.strip(), nearest_test_name(lines, i), val))
    return out


def _load_baseline(path: Path) -> set[str]:
    """Parse a baseline file containing one ``path|function|value`` token per
    line. Falls back to accepting legacy ``path:line`` entries when the new
    key cannot be parsed (defensive; CI regenerates the baseline anyway).
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


def _violation_token(path: Path, test_name: str, value: float) -> str:
    """Render a baseline token keyed on ``rel-path|test|value``. EE.18: this
    replaces the previous ``path:line`` token so editing the test file no
    longer drifts the baseline — only the test name and the literal value
    matter for stability."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
        rel_str = str(rel)
    except ValueError:
        rel_str = str(path)
    return f"{rel_str}|{test_name}|{_format_value(value)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="tests", help="root directory to scan")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any unjustified loose tolerance is found")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline file of known-existing violations (one "
        "`path|function|value` per line). Pre-existing entries are ignored "
        "under --strict; new violations fail the audit. Pass an explicit "
        "path or omit to default to tools/audit_tolerances_baseline.txt.",
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

    violations: list[tuple[Path, int, str, str, float]] = []
    for cpp in sorted(root.rglob("*.cpp")):
        for line_no, text, test_name, value in scan_file(cpp):
            violations.append((cpp, line_no, text, test_name, value))

    baseline_path = args.baseline if args.baseline is not None else DEFAULT_BASELINE

    if args.update_baseline:
        tokens = sorted({_violation_token(p, tn, v)
                         for p, _, _, tn, v in violations})
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            "# Baseline of known unjustified loose tolerances flagged by\n"
            "# tools/audit_tolerances.py. EE.18: each line is\n"
            "# `<rel-path>|<TestFixture.TestCase>|<tolerance-value>`\n"
            "# (path relative to repo root, value formatted via %.6g).\n"
            "# Pre-existing entries are ignored under --strict; new\n"
            "# violations FAIL the audit. Drive this list to zero over time\n"
            "# by justifying or tightening each tolerance.\n"
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
            (p, ln, t, tn, v) for (p, ln, t, tn, v) in violations
            if _violation_token(p, tn, v) not in baseline
        ]
        grandfathered = len(violations) - len(new_violations)
        if new_violations:
            print(
                f"Found {len(new_violations)} NEW unjustified loose "
                f"tolerance(s) (plus {grandfathered} pre-existing in baseline):"
            )
            for path, line_no, text, test_name, value in new_violations:
                print(f"  {path}:{line_no}  [{test_name} @ "
                      f"{_format_value(value)}]  {text}")
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
    for path, line_no, text, test_name, value in violations:
        print(f"  {path}:{line_no}  [{test_name} @ "
              f"{_format_value(value)}]  {text}")
    print("\nJustify each with a `// reason:` (or similar) comment within 2 "
          "lines above, OR tighten the tolerance.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
