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

# DType labels emitted by the MultiBackendDType fixture's param-namer.
# A test name like ``ZerosBasicShape/cuda0_Float16`` carries both axes
# explicitly in the printable suffix; the fixture builds this in
# tests/multi_backend_dtype_fixture.hpp via the BackendDTypeNamer helper.
DTYPE_LABELS = ("Float32", "Float64", "Float16", "BFloat16",
                "Int8", "Int16", "Int32", "Int64",
                "UInt8", "Bool",
                "Complex64", "Complex128")


@dataclass
class TestEntry:
    full_name: str
    op_name: str
    backend: str | None = None
    dtype: str | None = None
    parameterized: bool = False


# Audit-4 W.25: the printable param suffix is ``<backend>[N]_<Dtype>``:
#   ZerosBasicShape/cpu_Float32
#   ZerosBasicShape/cuda0_Float16
#   ZerosBasicShape/oneapi0_BFloat16
# The optional trailing digits come from device-index suffixes (cuda0, vulkan1).
PARAM_TAIL_RE = re.compile(
    r"/(?P<backend>cpu|cuda|rocm|vulkan|oneapi|mps)\d*_(?P<dtype>"
    + "|".join(re.escape(d) for d in DTYPE_LABELS) +
    r")\b"
)

# Backup: ctest renders the param tuple verbatim, e.g.
#   ``MultiBackendDType/Foo.Op/("cuda", 1-byte object <01>)``
# The hex byte indexes into the DType enum order
# (kept in sync with include/tenzor/core/dtype.hpp).
ENUM_DTYPE_ORDER = ("Float32", "Float64", "Float16", "BFloat16",
                    "Int8", "Int16", "Int32", "Int64",
                    "UInt8", "Bool", "Complex64", "Complex128")
BYTE_PARAM_RE = re.compile(
    r'\(\s*"(?P<backend>[a-z]+)"\s*,'
    r'[^<]*<(?P<idx>[0-9A-Fa-f]+)>'
)


def parse_test_name(test_name: str) -> TestEntry:
    """
    Parse a GoogleTest test name into (op, backend, dtype).

    Examples:
      MultiBackendDType/CreationOpsMultiDTypeTest.ZerosBasicShape/cuda0_Float16
        → op=ZerosBasicShape, backend=cuda, dtype=Float16
      VisionFusedParity.FusedConv2dSigmoid
        → op=FusedConv2dSigmoid, parameterized=False
    """
    parameterized = "/" in test_name
    # Op name is the leaf inside the SuiteName.OpName segment (the slash
    # divides between TEST_P suites and the parameter suffix).
    head = test_name.split("/")[1] if "/" in test_name else test_name
    suite_op = head.split(".")[-1] if "." in head else head
    op_name = suite_op  # keep the full op name — the prior regex strip
                       # mangled names like "ZerosBasicShape" → "ZerosBasic".

    backend = None
    dtype = None

    # Primary: pull (backend, dtype) from the printable param suffix.
    tail = PARAM_TAIL_RE.search(test_name)
    if tail:
        backend = tail.group("backend")
        dtype = tail.group("dtype")
    else:
        # Secondary: ctest renders the raw param tuple ``("cuda", 1-byte
        # object <01>)`` — decode the hex byte against the DType enum.
        cm = BYTE_PARAM_RE.search(test_name)
        if cm:
            backend = cm.group("backend")
            try:
                idx = int(cm.group("idx"), 16)
            except ValueError:
                idx = -1
            if 0 <= idx < len(ENUM_DTYPE_ORDER):
                dtype = ENUM_DTYPE_ORDER[idx]

        # Tertiary: the dtype is sometimes baked into the op name itself,
        # e.g. ``ExtendedParity.MatMul_Float64/"cpu"`` — pluck it out so
        # the dtype tally isn't an undercount.
        if dtype is None:
            for d in DTYPE_LABELS:
                if re.search(rf"(?<![A-Za-z0-9]){re.escape(d)}(?![A-Za-z0-9])",
                             test_name):
                    dtype = d
                    break

    # Last resort: scan the non-param portion of the name for a backend
    # keyword as a *word*, not a substring (substring match used to pick up
    # things like "cpu" inside arbitrary identifiers and ended up adding a
    # bogus "cpu" entry as an op name).
    if backend is None:
        for b in BACKENDS:
            if re.search(rf"(?<![A-Za-z0-9])(?:{b})(?![A-Za-z0-9])",
                         test_name, flags=re.IGNORECASE):
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
    except FileNotFoundError:
        # ctest itself is missing (PATH / not installed).
        print("ERROR: ctest not found on PATH — run from build/ directory",
              file=sys.stderr)
        sys.exit(2)
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

    Audit-4 W.25: when called for the default ``backend_parity`` label we
    also pull in the ``multi`` (MultiBackendDType) label so the per-dtype
    counts reflect both the dtype-typed parity tests and the per-axis
    multidtype tests. Without this, the dtype tally for the default report
    was always zero — the multidtype tests live under a separate label.
    """
    binaries = list_ctest_tests(label)
    if label == "backend_parity":
        # Merge in the multidtype binaries (label "multi") — dedupe by name
        # since the same binary often carries both labels.
        binaries = list({*binaries, *list_ctest_tests("multi")})
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
    parser.add_argument(
        "--require-real-ctest",
        action="store_true",
        help=(
            "FF.30: raise RuntimeError when ctest enumerates zero tests "
            "(i.e. the tool was run outside build/, or the build step was "
            "skipped). Without this flag the default Y.28 behaviour is "
            "preserved — print an explanatory error and exit 2."
        ),
    )
    args = parser.parse_args()

    test_names = discover_all_tests(args.label)
    # Y.28 / FF.30: ctest --show-only=json-v1 exits 0 with an empty test list
    # when the tool is invoked outside of build/ (no CTestTestfile.cmake to
    # scan). The legacy behaviour was to emit a silent empty matrix with exit
    # 0 — the exact same output we'd produce for a perfectly green run. The
    # default still treats the empty case as exit 2 with a human-readable
    # diagnostic. CI passes --require-real-ctest to escalate the same case
    # into a RuntimeError so a CI step that silently skipped the build
    # cannot mask the failure as a clean "no parity work to do" pass.
    if not test_names:
        if args.require_real_ctest:
            raise RuntimeError(
                "ctest enumerated zero tests — was the build step skipped?"
            )
        print(
            "ERROR: ctest enumerated zero tests for label "
            f"'{args.label}' — run this tool from the build/ directory "
            "(e.g. `cd build && python3 ../tools/parity_coverage_matrix.py`).",
            file=sys.stderr,
        )
        return 2
    entries = [parse_test_name(name) for name in test_names]
    cov = aggregate(entries)

    if args.json:
        print_json(cov)
    else:
        print_human(cov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
