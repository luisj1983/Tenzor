#!/usr/bin/env python3
"""
parity_coverage_matrix.py — emit an op × backend × dtype coverage matrix
for the Tenzor backend-parity test suite.

Lists every parity test from `ctest --show-only=json-v1`, parses test names
to extract op + backend + dtype, and prints:

  - A per-op summary: which (backend, dtype) combinations have coverage.
  - A grand-total: total parity tests, count by backend, count by dtype.
  - A "gaps" section: ops that are tested on CPU only (potential parity gap).

Run from the build directory:
    cd build && python3 ../tools/parity_coverage_matrix.py

Optional flag:
    --json   emit machine-readable JSON instead of human-readable tables.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


# Common backend names that appear in test parameter strings.
BACKENDS = {"cpu", "cuda", "rocm", "vulkan", "oneapi", "mps"}

# DType strings GoogleTest emits for the typed-test param. The MultiBackendDType
# fixture parameters tuples like ("cuda", <DType>); GoogleTest renders the DType
# enum as a 1-byte object — so we infer dtype from the test-suite name when
# possible (e.g., "FusedConv2dSigmoidShape" → dtype unknown, "Float32Add" → fp32).
DTYPE_HINTS = {
    "float32": "Float32",
    "float64": "Float64",
    "float16": "Float16",
    "bfloat16": "BFloat16",
    "fp32": "Float32",
    "fp64": "Float64",
    "fp16": "Float16",
    "bf16": "BFloat16",
}


@dataclass
class TestEntry:
    full_name: str
    op_name: str
    backend: str | None = None
    dtype: str | None = None
    parameterized: bool = False


PARAM_RE = re.compile(r'\("(?P<backend>[a-z]+)"\s*,\s*(?P<rest>.+)\)\s*$')


def parse_test_name(test_name: str) -> TestEntry:
    """
    Parse a GoogleTest test name into (op, backend, dtype).

    Examples:
      MultiBackendDType/FooBar.MyOp/("cuda", 1-byte object <00>)
        → op=MyOp, backend=cuda, dtype=Float32 (00 → first dtype)
      VisionFusedParity.FusedConv2dSigmoid
        → op=FusedConv2dSigmoid, parameterized=False
    """
    parameterized = "/" in test_name
    name_only = test_name.split("/")[-1] if parameterized else test_name
    suite_op = name_only.split(".")[-1] if "." in name_only else name_only
    op_name = re.sub(r"_*[A-Z][a-z]+$", "", suite_op) or suite_op

    backend = None
    dtype = None
    m = PARAM_RE.search(test_name)
    if m:
        backend = m.group("backend")
        rest = m.group("rest").lower()
        for hint, canonical in DTYPE_HINTS.items():
            if hint in rest:
                dtype = canonical
                break
        # Fallback: GoogleTest renders enums as "1-byte object <NN>".
        if dtype is None:
            byte_m = re.search(r"<(?P<hex>[0-9A-Fa-f]+)>", rest)
            if byte_m:
                idx = int(byte_m.group("hex"), 16)
                dtype_order = ["Float32", "Float64", "Float16", "BFloat16"]
                if 0 <= idx < len(dtype_order):
                    dtype = dtype_order[idx]

    # Detect backend keywords in the test/suite name (not all parity tests are
    # parameterized — many use direct helper calls and the backend is implicit).
    if backend is None:
        for b in BACKENDS:
            if b in test_name.lower():
                backend = b
                break

    return TestEntry(
        full_name=test_name,
        op_name=op_name,
        backend=backend,
        dtype=dtype,
        parameterized=parameterized,
    )


def list_ctest_tests(label: str = "backend_parity") -> list[str]:
    """Invoke `ctest --show-only=json-v1` and return all matching test names."""
    try:
        out = subprocess.check_output(
            ["ctest", "--show-only=json-v1", "-L", label],
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"ctest failed: {e}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(out)
    return [t["name"] for t in data.get("tests", [])]


def list_gtest_tests_in_binary(binary_path: str) -> list[str]:
    """
    Run `<binary> --gtest_list_tests` and return the full test names.

    Output format (per Google Test docs):
        SuiteName.
          TestName1
          TestName2
        ParamSuite/.
          ParamTest/("cuda", 1-byte object <00>)
          ...

    Each TestName is indented under its suite. We join them as Suite.Test.
    """
    import os.path
    if not os.path.exists(binary_path):
        return []
    try:
        out = subprocess.check_output(
            [binary_path, "--gtest_list_tests"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []
    tests: list[str] = []
    suite = None
    for line in out.splitlines():
        if not line.strip():
            continue
        if line.endswith("."):
            suite = line.strip().rstrip(".")
        elif line.startswith(" ") and suite is not None:
            test = line.strip()
            tests.append(f"{suite}.{test}")
    return tests


def discover_all_tests(label: str = "backend_parity",
                       bin_dir: str = "bin") -> list[str]:
    """
    Combine ctest binary names with each binary's gtest test enumeration to
    produce a list of fully-qualified GoogleTest test names.
    """
    binaries = list_ctest_tests(label)
    all_tests: list[str] = []
    for b in binaries:
        path = f"{bin_dir}/{b}"
        gtests = list_gtest_tests_in_binary(path)
        if gtests:
            all_tests.extend(gtests)
        else:
            # Fall back to the binary name so it still appears in the matrix.
            all_tests.append(b)
    return all_tests


@dataclass
class OpCoverage:
    backends: set[str] = field(default_factory=set)
    dtypes: set[str] = field(default_factory=set)
    pairs: set[tuple[str, str]] = field(default_factory=set)  # (backend, dtype)
    total_tests: int = 0


def aggregate(entries: Iterable[TestEntry]) -> dict[str, OpCoverage]:
    cov: dict[str, OpCoverage] = defaultdict(OpCoverage)
    for e in entries:
        c = cov[e.op_name]
        c.total_tests += 1
        if e.backend:
            c.backends.add(e.backend)
        if e.dtype:
            c.dtypes.add(e.dtype)
        if e.backend and e.dtype:
            c.pairs.add((e.backend, e.dtype))
    return cov


def print_human(cov: dict[str, OpCoverage]) -> None:
    backend_totals: dict[str, int] = defaultdict(int)
    dtype_totals: dict[str, int] = defaultdict(int)
    for op, c in cov.items():
        for b in c.backends:
            backend_totals[b] += 1
        for d in c.dtypes:
            dtype_totals[d] += 1

    print("=" * 70)
    print(f"BACKEND PARITY COVERAGE — {len(cov)} ops, {sum(c.total_tests for c in cov.values())} total tests")
    print("=" * 70)

    print("\nPer-backend coverage (ops with at least one test on this backend):")
    for b in sorted(BACKENDS):
        n = backend_totals.get(b, 0)
        marker = "✓" if n > 0 else "✗"
        print(f"  {marker} {b:<8} {n:>5} ops")

    print("\nPer-dtype coverage:")
    for d in ("Float32", "Float64", "Float16", "BFloat16"):
        n = dtype_totals.get(d, 0)
        marker = "✓" if n > 0 else "✗"
        print(f"  {marker} {d:<10} {n:>5} ops")

    print("\nOps tested only on CPU (potential gap — should be parameterized):")
    cpu_only = sorted(op for op, c in cov.items() if c.backends == {"cpu"})
    for op in cpu_only:
        print(f"  - {op}")
    if not cpu_only:
        print("  (none — every parameterized op covers at least one non-CPU backend)")

    print("\nOps with no detected backend (likely use helper-fn pattern, not TEST_P):")
    no_backend = sorted(op for op, c in cov.items() if not c.backends)
    print(f"  {len(no_backend)} ops — these are good candidates for the Phase 4.1 refactor.")
    for op in no_backend[:25]:
        print(f"  - {op}")
    if len(no_backend) > 25:
        print(f"  ... and {len(no_backend) - 25} more.")


def print_json(cov: dict[str, OpCoverage]) -> None:
    out = {
        op: {
            "backends": sorted(c.backends),
            "dtypes": sorted(c.dtypes),
            "pairs": sorted(list(p) for p in c.pairs),
            "total_tests": c.total_tests,
        }
        for op, c in sorted(cov.items())
    }
    print(json.dumps(out, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of human-readable tables.")
    parser.add_argument("--label", default="backend_parity",
                        help="ctest label to filter by (default: backend_parity).")
    args = parser.parse_args()

    test_names = discover_all_tests(args.label)
    entries = [parse_test_name(name) for name in test_names]
    cov = aggregate(entries)

    if args.json:
        print_json(cov)
    else:
        print_human(cov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
