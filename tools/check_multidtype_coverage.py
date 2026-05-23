#!/usr/bin/env python3
"""
audit-3 T.14: ratchet for multi-dtype companion coverage.

For every operations/autograd/nn test file (test_*.cpp under tests/) we
expect a companion test_*_multidtype.cpp that runs the same surface area
through the `MultiBackendDTypeTest` fixture (5 backends × supported dtypes).

This script lists every non-multidtype test_*.cpp, checks whether a sibling
multidtype companion exists somewhere under tests/, and reports any missing
companion that is NOT on the known-intentional allowlist in
`tests/backend_parity/MULTIDTYPE_COVERAGE_GAPS.md`.

Exits 1 on any new gap, 0 otherwise. Wired into ctest as the
`MultidtypeCoverage` lint test (see top-level CMakeLists.txt).

Subtrees deliberately excluded (see MULTIDTYPE_COVERAGE_GAPS.md "Known-
intentional (kept CPU-only, no action)" section):
    backend_parity/, jit/, backends/, distributed/, serving/, lite/,
    nested/, utils/, examples/, benchmarks/
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
GAPS_DOC = TESTS_DIR / "backend_parity" / "MULTIDTYPE_COVERAGE_GAPS.md"

# Subtrees inside tests/ whose contents are CPU-only by design (parity-
# parameterized themselves, infrastructure, JIT, distributed RPC, lite/serving
# runtime, examples). These do not participate in the multidtype-companion
# convention.
EXCLUDED_DIR_PREFIXES = (
    "tests/backend_parity/",
    "tests/jit/",
    "tests/backends/",
    "tests/distributed/",
    "tests/serving/",
    "tests/lite/",
    "tests/nested/",
    "tests/utils/",
    "tests/examples/",
    "tests/benchmarks/",
)

# Filename suffixes / patterns that don't participate (gradcheck per-op,
# performance, contract regression files).
EXCLUDED_NAME_PATTERNS = (
    re.compile(r"^test_gradcheck.*\.cpp$"),
    re.compile(r"^test_simd_.*\.cpp$"),
    re.compile(r"^test_performance.*\.cpp$"),
    re.compile(r"^test_allocator.*\.cpp$"),
    re.compile(r"^test_memory.*\.cpp$"),
    re.compile(r"^test_pinned_.*\.cpp$"),
    re.compile(r"^test_caching_.*\.cpp$"),
)


def load_known_intentional(doc_path: Path) -> set[str]:
    """Parse the machine-readable block in MULTIDTYPE_COVERAGE_GAPS.md."""
    if not doc_path.exists():
        return set()
    text = doc_path.read_text(encoding="utf-8")
    m = re.search(
        r"<!--\s*KNOWN-INTENTIONAL-START\s*-->\s*\n(.*?)\n\s*<!--\s*KNOWN-INTENTIONAL-END\s*-->",
        text,
        re.DOTALL,
    )
    if not m:
        return set()
    entries: set[str] = set()
    for raw in m.group(1).splitlines():
        # Strip trailing inline comments and whitespace.
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        entries.add(line)
    return entries


def is_excluded(rel_path: str) -> bool:
    if any(rel_path.startswith(p) for p in EXCLUDED_DIR_PREFIXES):
        return True
    name = Path(rel_path).name
    return any(p.match(name) for p in EXCLUDED_NAME_PATTERNS)


def main() -> int:
    if not TESTS_DIR.is_dir():
        print(f"ERROR: tests dir not found at {TESTS_DIR}", file=sys.stderr)
        return 2

    allow = load_known_intentional(GAPS_DOC)

    # Index every multidtype companion by its base name (without the
    # _multidtype.cpp suffix) so a companion can live anywhere under tests/.
    multidtype_bases: set[str] = set()
    for p in TESTS_DIR.rglob("test_*_multidtype.cpp"):
        base = p.name[: -len("_multidtype.cpp")]  # "test_foo"
        multidtype_bases.add(base)

    missing: list[str] = []
    for p in TESTS_DIR.rglob("test_*.cpp"):
        rel = p.relative_to(REPO_ROOT).as_posix()
        name = p.name
        if name.endswith("_multidtype.cpp"):
            continue
        if is_excluded(rel):
            continue
        base = name[: -len(".cpp")]  # "test_foo"
        if base in multidtype_bases:
            continue
        if rel in allow:
            continue
        missing.append(rel)

    missing.sort()
    if missing:
        print("NEW multi-dtype companion gaps detected:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print(
            f"\nEither (a) author a sibling test_<name>_multidtype.cpp using "
            f"tests/multi_backend_dtype_fixture.hpp, or (b) accept the gap as "
            f"intentional by adding the path to the KNOWN-INTENTIONAL block in "
            f"{GAPS_DOC.relative_to(REPO_ROOT)}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
