#!/usr/bin/env python3
"""
Operation coverage matrix for the Tenzor project.

Parses OpIds from op_id.hpp and kernel registrations from each backend's
registry file, then outputs a table showing which ops are registered where.

Usage:
    python op_coverage_matrix.py                     # Full markdown table
    python op_coverage_matrix.py --missing-only      # Only ops with gaps
    python op_coverage_matrix.py --format csv        # CSV output
    python op_coverage_matrix.py --reserved          # Include reserved/unused ops
"""

import argparse
import re
import sys
from pathlib import Path
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OP_ID_HPP = PROJECT_ROOT / "include" / "tenzor" / "ops" / "op_id.hpp"

BACKEND_REGISTRIES = OrderedDict([
    ("CPU",    PROJECT_ROOT / "src" / "backends" / "cpu"    / "cpu_kernel_registry.cpp"),
    ("CUDA",   PROJECT_ROOT / "src" / "backends" / "cuda"   / "cuda_kernel_registry.cpp"),
    ("ROCm",   PROJECT_ROOT / "src" / "backends" / "rocm"   / "rocm_kernel_registry.cpp"),
    ("Vulkan", PROJECT_ROOT / "src" / "backends" / "vulkan" / "vulkan_kernel_registry.cpp"),
    ("OneAPI", PROJECT_ROOT / "src" / "backends" / "oneapi"  / "oneapi_kernel_registry.cpp"),
    ("MPS",    PROJECT_ROOT / "src" / "backends" / "mps"    / "mps_kernel_registry.mm"),
])

# OpIds that are explicitly documented as reserved/unused in op_id.hpp
RESERVED_OPS = {
    "LSTMBackward", "GRUBackward", "BiLSTMBackward", "RNNForward",
    "SparseSpMM", "SparseSpMV", "SparseToDense", "DenseToSparse", "SparseAdd",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_op_ids(path: Path) -> list[tuple[str, int]]:
    """Parse enum class OpId from op_id.hpp. Returns [(name, value), ...]."""
    text = path.read_text()

    # Extract the enum body between the opening { and the closing };
    match = re.search(r"enum\s+class\s+OpId\s*:\s*\w+\s*\{(.*?)\};", text, re.DOTALL)
    if not match:
        print(f"ERROR: Could not find OpId enum in {path}", file=sys.stderr)
        sys.exit(1)

    body = match.group(1)
    ops = []
    current_value = 0

    for line in body.splitlines():
        # Strip comments (but preserve for context)
        line_stripped = re.sub(r"//.*", "", line).strip()
        if not line_stripped:
            continue

        # Match: Name = Value,  or  Name,  (with optional trailing comma)
        m = re.match(r"^(\w+)\s*(?:=\s*(\d+))?\s*,?\s*$", line_stripped)
        if not m:
            continue

        name = m.group(1)
        if name == "OP_COUNT":
            continue

        if m.group(2) is not None:
            current_value = int(m.group(2))

        ops.append((name, current_value))
        current_value += 1

    return ops


def parse_registry(path: Path) -> dict[str, str]:
    """
    Parse a backend kernel registry file.
    Returns {op_name: registration_type} where registration_type is one of:
      'kernel', 'single', 'inplace', 'fallback'
    """
    if not path.exists():
        return {}

    text = path.read_text()
    registered: dict[str, str] = {}

    # Pattern 1: table.register_kernel(OpId::XXX, ...)
    for m in re.finditer(r"table\.register_kernel\(OpId::(\w+)", text):
        registered.setdefault(m.group(1), "kernel")

    # Pattern 2: table.register_single_output_kernel(OpId::XXX, ...)
    for m in re.finditer(r"table\.register_single_output_kernel\(OpId::(\w+)", text):
        registered.setdefault(m.group(1), "single")

    # Pattern 3: table.register_inplace_kernel(OpId::XXX, ...)
    for m in re.finditer(r"table\.register_inplace_kernel\(OpId::(\w+)", text):
        registered.setdefault(m.group(1), "inplace")

    # Pattern 4: TENZOR_REGISTER_*_KERNEL(table, XXX, ...) — CPU macros
    for m in re.finditer(r"TENZOR_REGISTER_\w+_KERNEL\(table,\s*(\w+)", text):
        registered.setdefault(m.group(1), "kernel")

    # Pattern 5: VULKAN_CPU_FALLBACK(XXX) and VULKAN_FFT_CPU_FALLBACK(XXX)
    for m in re.finditer(r"VULKAN_(?:FFT_)?CPU_FALLBACK\((\w+)\)", text):
        registered[m.group(1)] = "fallback"  # Override — mark as fallback

    # Pattern 6: ROCM_SINGLE_UNARY_NATIVE(OpName, fn)
    for m in re.finditer(r"ROCM_SINGLE_UNARY_NATIVE\((\w+)", text):
        registered.setdefault(m.group(1), "single")

    # Pattern 7: VK_REGISTER_UNARY_SPECIAL(OpName, opcode)
    for m in re.finditer(r"VK_REGISTER_UNARY_SPECIAL\((\w+)", text):
        registered.setdefault(m.group(1), "single")

    # Pattern 8: ONEAPI_REGISTER_UNARY_SPECIAL(OpName, fn)
    for m in re.finditer(r"ONEAPI_REGISTER_UNARY_SPECIAL\((\w+)", text):
        registered.setdefault(m.group(1), "single")

    # Pattern 9: ONEAPI_REGISTER_BINARY_SPECIAL(OpName, fn)
    for m in re.finditer(r"ONEAPI_REGISTER_BINARY_SPECIAL\((\w+)", text):
        registered.setdefault(m.group(1), "kernel")

    return registered


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def status_symbol(reg_type: str | None) -> str:
    if reg_type is None:
        return "-"
    if reg_type == "fallback":
        return "F"
    return "Y"


def format_markdown(
    ops: list[tuple[str, int]],
    backends: OrderedDict,
    registrations: dict[str, dict[str, str]],
    missing_only: bool,
    show_reserved: bool,
) -> str:
    backend_names = list(backends.keys())
    lines = []

    # Header
    header = f"| {'OpId':<35} | {'#':>3} | " + " | ".join(f"{b:^6}" for b in backend_names) + " |"
    sep    = f"|{'-' * 37}|{'-' * 5}|" + "|".join("-" * 8 for _ in backend_names) + "|"
    lines.append(header)
    lines.append(sep)

    for name, value in ops:
        if not show_reserved and name in RESERVED_OPS:
            continue

        row_statuses = []
        for bname in backend_names:
            reg = registrations[bname].get(name)
            row_statuses.append(status_symbol(reg))

        if missing_only and all(s == "Y" for s in row_statuses):
            continue

        cells = " | ".join(f"{s:^6}" for s in row_statuses)
        reserved_mark = " *" if name in RESERVED_OPS else ""
        lines.append(f"| {name + reserved_mark:<35} | {value:>3} | {cells} |")

    return "\n".join(lines)


def format_csv(
    ops: list[tuple[str, int]],
    backends: OrderedDict,
    registrations: dict[str, dict[str, str]],
    missing_only: bool,
    show_reserved: bool,
) -> str:
    backend_names = list(backends.keys())
    lines = []
    lines.append(",".join(["OpId", "Value"] + backend_names))

    for name, value in ops:
        if not show_reserved and name in RESERVED_OPS:
            continue

        row_statuses = []
        for bname in backend_names:
            reg = registrations[bname].get(name)
            row_statuses.append(status_symbol(reg))

        if missing_only and all(s == "Y" for s in row_statuses):
            continue

        lines.append(",".join([name, str(value)] + row_statuses))

    return "\n".join(lines)


def print_summary(
    ops: list[tuple[str, int]],
    backends: OrderedDict,
    registrations: dict[str, dict[str, str]],
    show_reserved: bool,
) -> str:
    backend_names = list(backends.keys())
    filtered_ops = [
        (name, val) for name, val in ops
        if show_reserved or name not in RESERVED_OPS
    ]
    total = len(filtered_ops)

    lines = [
        "",
        "## Summary",
        "",
        f"Total operations: {total}" + ("" if show_reserved else f" (excluding {len(RESERVED_OPS)} reserved)"),
        "",
    ]

    header = f"| {'Backend':<10} | {'Registered':>10} | {'Fallback':>8} | {'Missing':>7} | {'Coverage':>8} |"
    sep    = f"|{'-' * 12}|{'-' * 12}|{'-' * 10}|{'-' * 9}|{'-' * 10}|"
    lines.append(header)
    lines.append(sep)

    for bname in backend_names:
        regs = registrations[bname]
        native = sum(1 for name, _ in filtered_ops if regs.get(name) in ("kernel", "single", "inplace"))
        fallback = sum(1 for name, _ in filtered_ops if regs.get(name) == "fallback")
        missing = total - native - fallback
        pct = (native + fallback) / total * 100 if total else 0
        lines.append(
            f"| {bname:<10} | {native:>10} | {fallback:>8} | {missing:>7} | {pct:>7.1f}% |"
        )

    lines.append("")
    lines.append("Legend: Y = native kernel, F = CPU fallback, - = missing, * = reserved/unused")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate an operation coverage matrix for Tenzor backends."
    )
    parser.add_argument(
        "--format", choices=["markdown", "csv"], default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--missing-only", action="store_true",
        help="Show only operations that have gaps (not registered on all backends)",
    )
    parser.add_argument(
        "--reserved", action="store_true",
        help="Include reserved/unused OpIds in the output",
    )
    args = parser.parse_args()

    # Parse OpIds
    if not OP_ID_HPP.exists():
        print(f"ERROR: {OP_ID_HPP} not found", file=sys.stderr)
        sys.exit(1)

    ops = parse_op_ids(OP_ID_HPP)
    if not ops:
        print("ERROR: No OpIds parsed from op_id.hpp", file=sys.stderr)
        sys.exit(1)

    # Parse each backend registry
    registrations: dict[str, dict[str, str]] = {}
    for bname, path in BACKEND_REGISTRIES.items():
        if not path.exists():
            print(f"WARNING: {path} not found, treating {bname} as empty", file=sys.stderr)
        registrations[bname] = parse_registry(path)

    # Output
    if args.format == "markdown":
        print(format_markdown(ops, BACKEND_REGISTRIES, registrations, args.missing_only, args.reserved))
        print(print_summary(ops, BACKEND_REGISTRIES, registrations, args.reserved))
    elif args.format == "csv":
        print(format_csv(ops, BACKEND_REGISTRIES, registrations, args.missing_only, args.reserved))


if __name__ == "__main__":
    main()
