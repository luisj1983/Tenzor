#!/usr/bin/env python3
"""
check_layer_tests.py — flag layer tests calling .backward() without an
explicit non-zero gradient assertion.

The bug class this guards against (raw-tensor-op breaks autograd graph)
silently zeros gradients. Tests that only check `EXPECT_NO_THROW(...)` or
`ASSERT_TRUE(input.grad().has_value())` pass for an all-zero grad tensor
and so don't catch the regression. The repo standard is to use the
`EXPECT_GRAD_FLOWS` macro from `tests/grad_flow_helpers.hpp` (or an
equivalent magnitude check like `EXPECT_GT(max(abs(grad)).item(), 0)`).

This script scans every `tests/nn/layers/test_*.cpp` file. For each that
calls `.backward()` (or `loss.backward()`), it requires at least one of:

  - `EXPECT_GRAD_FLOWS(` macro call
  - `numerical_gradient(` finite-difference helper
  - An explicit max-magnitude / per-element non-zero pattern
  - A `// grad-check-exempt:` opt-out comment with reason

If any file fails, the script exits with code 1 and prints the offenders.

Run as a CI step or pre-merge check:
    python3 tools/check_layer_tests.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LAYER_TEST_DIR = REPO_ROOT / "tests" / "nn" / "layers"

# Patterns that count as a real grad-flow assertion.
STRONG_PATTERNS = [
    re.compile(r"EXPECT_GRAD_FLOWS\b"),
    re.compile(r"EXPECT_GRAD_FLOWS_REL\b"),
    re.compile(r"numerical_gradient\b"),  # finite-difference helper
    # `max(abs(grad...))` followed (typically a few lines later) by an
    # `EXPECT_GT(...item..., 0)` is the canonical manual non-zero check.
    # Use a multiline-aware regex so the two parts can sit on different
    # lines.
    re.compile(r"max\(abs\(.*?grad", re.DOTALL),
    re.compile(r"EXPECT_GT\([^,;]*\.item<[^,;]*0\.0", re.DOTALL),
    re.compile(r"max_abs\b"),
    re.compile(r"local_max\b"),
    # Element-wise expected-value checks (Identity, Alibi, etc).
    re.compile(r"EXPECT_NEAR\([^)]*g\.data<"),
    re.compile(r"EXPECT_NEAR\([^)]*grad_data\b"),
]

OPT_OUT = re.compile(r"//\s*grad-check-exempt:\s*\S+")
BACKWARD = re.compile(r"\.backward\(\)|loss\.backward\b|->backward\b")


def file_has_strong_assertion(text: str) -> bool:
    return any(p.search(text) for p in STRONG_PATTERNS)


def file_has_opt_out(text: str) -> bool:
    return bool(OPT_OUT.search(text))


def file_calls_backward(text: str) -> bool:
    return bool(BACKWARD.search(text))


def main() -> int:
    if not LAYER_TEST_DIR.is_dir():
        print(f"check_layer_tests: directory not found: {LAYER_TEST_DIR}",
              file=sys.stderr)
        return 2

    offenders: list[Path] = []
    for path in sorted(LAYER_TEST_DIR.glob("test_*.cpp")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not file_calls_backward(text):
            continue
        if file_has_strong_assertion(text):
            continue
        if file_has_opt_out(text):
            continue
        offenders.append(path)

    if offenders:
        print(
            "check_layer_tests: the following layer tests call .backward() "
            "without an EXPECT_GRAD_FLOWS / equivalent non-zero assertion:",
            file=sys.stderr,
        )
        for p in offenders:
            rel = p.relative_to(REPO_ROOT)
            print(f"  {rel}", file=sys.stderr)
        print(
            "\nFix: add `#include \"../../grad_flow_helpers.hpp\"` and call\n"
            "  EXPECT_GRAD_FLOWS(input);\n"
            "after every loss.backward() in each affected test.\n"
            "Or use a stronger explicit check (EXPECT_GT max abs, "
            "numerical_gradient, EXPECT_NEAR per element).\n"
            "If a non-zero check is genuinely inappropriate, mark with\n"
            "  // grad-check-exempt: <one-line reason>\n"
            "anywhere in the file.",
            file=sys.stderr,
        )
        return 1

    print(f"check_layer_tests: OK ({len(list(LAYER_TEST_DIR.glob('test_*.cpp')))} "
          f"files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
