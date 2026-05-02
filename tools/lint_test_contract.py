#!/usr/bin/env python3
"""
Lint gate enforcing the TESTING.md contract on test files.

Phase 10 of the test-coverage campaign. Run via `ctest -L lint` or directly:
    python3 tools/lint_test_contract.py

Exits non-zero (and prints offending lines) if any test file violates:

  1. Defines `struct BackendDTypeParam` outside multi_backend_dtype_fixture.hpp.
     Use `MultiBackendDTypeTest` from that header instead.
  2. Contains `EXPECT_NO_THROW(...backward...)` without a matching
     `EXPECT_GRAD_FLOWS` in the same TEST_F/TEST_P body.
  3. Contains `if (!Device::*_available()) GTEST_SKIP()` outside SetUp().
     Use the canonical `SKIP_IF_NO_<BACKEND>` macros from
     `tests/backend_parity/parity_test_utils.hpp`, or
     `REQUIRE_MULTI_BACKEND_OR_SKIP` for parity tests.
  4. Contains `DISABLED_*` test names (banned per TESTING.md — fix the
     implementation or delete the test).

Allowed exceptions (enumerated explicitly to avoid silent drift):
  - tests/multi_backend_dtype_fixture.hpp defines the canonical type alias.
  - tests/backend_test_fixture.hpp defines the canonical macros.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests"

# Files that are allowed to mention BackendDTypeParam (the canonical alias
# itself lives here). Everything else under tests/ is checked.
ALLOWED_BACKEND_DTYPE_PARAM_FILES = {
    TESTS_DIR / "multi_backend_dtype_fixture.hpp",
}


def _iter_test_sources(test_dir: Path) -> Iterable[Path]:
    for p in test_dir.rglob("*.cpp"):
        yield p
    for p in test_dir.rglob("*.hpp"):
        yield p


def _find_test_p_blocks(text: str) -> list[tuple[int, int, str]]:
    """Return (start_offset, end_offset, body) for every TEST_F/TEST_P macro.

    Uses brace counting so nested braces in the test body don't trip us up.
    """
    blocks: list[tuple[int, int, str]] = []
    pat = re.compile(r"\bTEST(?:_F|_P)?\s*\([^)]+\)\s*\{", re.MULTILINE)
    for m in pat.finditer(text):
        body_start = m.end()
        depth = 1
        i = body_start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        blocks.append((m.start(), i, text[body_start:i]))
    return blocks


def check_struct_backend_dtype_param(path: Path, text: str) -> list[str]:
    if path in ALLOWED_BACKEND_DTYPE_PARAM_FILES:
        return []
    errors: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if re.search(r"^\s*struct\s+BackendDTypeParam\b", line):
            errors.append(
                f"{path}:{i}: defines `struct BackendDTypeParam` — use "
                f"MultiBackendDTypeTest from "
                f"tests/multi_backend_dtype_fixture.hpp instead."
            )
    return errors


def check_weak_backward(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    for start, end, body in _find_test_p_blocks(text):
        # Only flag if the test calls EXPECT_NO_THROW on .backward() AND
        # never asserts gradient flow / non-zero gradient afterwards.
        no_throw_backward = re.search(
            r"EXPECT_NO_THROW\s*\(\s*[^)]*\.backward\s*\(", body
        )
        if not no_throw_backward:
            continue
        # Acceptable strengthenings:
        if re.search(
            r"EXPECT_GRAD_FLOWS|EXPECT_GRAD_FLOWS_REL|"
            r"\.has_grad\s*\(\s*\)|\.grad\s*\(\s*\)\s*\.value",
            body,
        ):
            continue
        # Compute the line number of the EXPECT_NO_THROW.
        line_no = text[: start + no_throw_backward.start()].count("\n") + 1
        errors.append(
            f"{path}:{line_no}: EXPECT_NO_THROW(.backward(...)) without "
            f"EXPECT_GRAD_FLOWS or has_grad/grad-value check. Strengthen "
            f"with EXPECT_GRAD_FLOWS(<var>) from tests/grad_flow_helpers.hpp."
        )
    return errors


def check_inline_skip(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    # Find any TEST_F/TEST_P body with `if (!Device::*_available()) GTEST_SKIP()`
    pat = re.compile(
        r"if\s*\(\s*!?\s*Device::\w+_available\s*\(\s*\)\s*\)\s*"
        r"(?:\{?\s*)?GTEST_SKIP\s*\("
    )
    # Also catch is_*_available() variants and Device::Type::CPU comparisons
    # used as backend-availability gates outside SetUp.
    test_blocks = _find_test_p_blocks(text)
    for start, end, body in test_blocks:
        m = pat.search(body)
        if m:
            line_no = text[: start + m.start()].count("\n") + 1
            errors.append(
                f"{path}:{line_no}: inline `if (!Device::*_available()) "
                f"GTEST_SKIP()` — use SKIP_IF_NO_<BACKEND> from "
                f"tests/backend_parity/parity_test_utils.hpp."
            )
    return errors


def check_disabled_tests(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    pat = re.compile(r"\bTEST(?:_F|_P)?\s*\([^,]+,\s*(DISABLED_\w+)")
    for m in pat.finditer(text):
        line_no = text[: m.start()].count("\n") + 1
        errors.append(
            f"{path}:{line_no}: DISABLED_ test `{m.group(1)}` is banned. "
            f"Fix the implementation or delete the test (see TESTING.md)."
        )
    return errors


def lint_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    errors: list[str] = []
    errors.extend(check_struct_backend_dtype_param(path, text))
    errors.extend(check_weak_backward(path, text))
    errors.extend(check_inline_skip(path, text))
    errors.extend(check_disabled_tests(path, text))
    return errors


def _load_baseline(path: Path) -> set[str]:
    """Parse a baseline file containing one `path:line:` token per line.

    The baseline pins the set of pre-existing violations: the lint passes if
    every reported violation is in the baseline, and fails on any NEW one.
    The intent is to ratchet down — never introduce new violations, fix
    existing ones over time.
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


def _violation_token(err: str) -> str:
    """Extract `path:line:` from an error string."""
    m = re.match(r"(.+?:\d+):", err)
    return m.group(1) + ":" if m else err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=TESTS_DIR,
        help="Path to tests/ directory (default: project tests/).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "tools" / "lint_test_contract_baseline.txt",
        help="Baseline file of known-existing violations (one `path:line:` "
        "per line). Pre-existing entries are ignored; new violations fail "
        "the lint.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Ignore the baseline; fail on every violation.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline file with the current set of violations. "
        "Use after fixing a violation to remove it from the baseline.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors; suppress 'OK' summary on success.",
    )
    args = parser.parse_args()

    all_errors: list[str] = []
    file_count = 0
    for path in _iter_test_sources(args.tests_dir):
        file_count += 1
        all_errors.extend(lint_file(path))

    if args.update_baseline:
        tokens = sorted({_violation_token(e) for e in all_errors})
        args.baseline.write_text(
            "# Baseline of known TESTING.md contract violations.\n"
            "# Each line is `<path>:<line>:` matching the lint script's output.\n"
            "# Pre-existing entries are ignored by the lint; new violations FAIL.\n"
            "# Drive this list to zero over time.\n"
            + "\n".join(tokens)
            + "\n",
            encoding="utf-8",
        )
        print(f"baseline updated: {len(tokens)} entries → {args.baseline}")
        return 0

    baseline = set() if args.strict else _load_baseline(args.baseline)

    new_errors = [e for e in all_errors if _violation_token(e) not in baseline]
    grandfathered = len(all_errors) - len(new_errors)

    if new_errors:
        print(
            f"TESTING.md contract: {len(new_errors)} NEW violations "
            f"(plus {grandfathered} pre-existing in baseline):",
            file=sys.stderr,
        )
        for err in new_errors:
            print(f"  {err}", file=sys.stderr)
        print(
            f"\nFix the new violations or, if intentional, regenerate the "
            f"baseline with `python3 tools/lint_test_contract.py --update-baseline`.",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        print(
            f"OK: {file_count} test files, {len(all_errors)} pre-existing "
            f"violations (all in baseline), 0 new violations."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
