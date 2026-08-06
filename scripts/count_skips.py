#!/usr/bin/env python3
"""
Tally test skips in tests/ by category.

Scans every tests/**/*.{cpp,py} for:
  - GTEST_SKIP / GTEST_SKIP_ (C++) — the skip message is inspected for a
    [SkipReason::XXX] tag emitted by SKIP_WITH_REASON in
    multi_backend_dtype_fixture.hpp. Untagged skips are counted as UNTAGGED
    so they can be chased down over time.
  - pytest.skip / @pytest.mark.skipif / unittest.skip* (Python)

Produces:
  - human-readable summary on stdout (default)
  - machine-readable JSON with --json
  - CI gate: non-zero exit if UNTAGGED C++ skip count exceeds a threshold
    passed via --max-untagged.

Usage:
  scripts/count_skips.py
  scripts/count_skips.py --json report.json
  scripts/count_skips.py --max-untagged 500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

# Known SkipReason enum values — keep in sync with multi_backend_dtype_fixture.hpp.
KNOWN_REASONS = {
    "BackendUnavailable",
    "BackendExcludedByEnv",
    "NumericalDivergence",
    "DtypeUnsupportedOnBackend",
    "ComplexFP16Unrepresentable",
    "GradcheckFDPrecision",
    "KernelNotImplemented",
    "RequiresMultiGPU",
    "KnownBug",
    "MissingPerfBaseline",
    "NotApplicable",
    # Heuristic-only labels below (no matching C++ SkipReason enum value —
    # these classify *bare* GTEST_SKIP() call sites by message text, distinct
    # from the explicitly-tagged SKIP_WITH_REASON enum in
    # multi_backend_dtype_fixture.hpp).
    "ResourceConstraint",
}

# Match GTEST_SKIP() / GTEST_SKIP_(..) with optional << "message" sequences
# until the terminating semicolon. We keep everything on the logical line
# after GTEST_SKIP for reason extraction.
CXX_SKIP_RE = re.compile(
    r"\bGTEST_SKIP(?:_F|\s*\(\s*\))?[^;]*?(?:<<[^;]*)?;", re.DOTALL
)
CXX_SKIP_WITH_REASON_RE = re.compile(
    r"\bSKIP_WITH_REASON\s*\(\s*(?:::)?(?:\w+::)*SkipReason::(\w+)\s*,", re.MULTILINE
)
CXX_REASON_IN_MSG_RE = re.compile(r"\[SkipReason::(\w+)\]")

# Heuristic — when a raw GTEST_SKIP message text matches one of these regexes we
# auto-classify without requiring source edits. Keeps the "UNTAGGED" bucket
# focused on actually-ambiguous skips. Order matters (first hit wins).
MSG_HEURISTICS = [
    (re.compile(r"\bTENZOR_SKIP_BACKENDS\b|excluded via env", re.I), "BackendExcludedByEnv"),
    (re.compile(r"not available|no .* devices? (?:available|found)|backend .* unavailable", re.I),
     "BackendUnavailable"),
    (re.compile(r"requires? .* multi[- ]?gpu|multiple? gpus|2\+? gpus|world_size\s*>\s*1", re.I),
     "RequiresMultiGPU"),
    (re.compile(r"not implemented|TODO|FIXME|KernelNotImplemented", re.I), "KernelNotImplemented"),
    (re.compile(r"Float16 .*complex|complex.* Float16|no Float16 complex", re.I),
     "ComplexFP16Unrepresentable"),
    (re.compile(r"Need at least 2 backends", re.I), "RequiresMultiGPU"),
    # Structural pattern: "if (backends.size() < 2) GTEST_SKIP()" — a
    # multi-backend parity test that can't run on single-backend hosts.
    (re.compile(r"backends\.size\s*\(\s*\)\s*<\s*2", re.I), "RequiresMultiGPU"),
    # "if (!has_cuda()) GTEST_SKIP()" and similar device probes without a
    # human-readable message in the skip argument. Matches both function-call
    # syntax (has_cuda()) and a local bool variable of the same name
    # (bool has_cuda = ...; if (!has_cuda) ...) — the latter evaded this
    # heuristic when it required a literal trailing '(', which was the #1
    # cause of the UNTAGGED bucket (test_quantized_linear_int4.cpp,
    # test_fft.cpp both probe availability into a local bool).
    (re.compile(r"!\s*has_(?:cuda|rocm|vulkan|oneapi|mps)\b", re.I),
     "BackendUnavailable"),
    (re.compile(r"!is_(?:cuda|rocm|vulkan|oneapi|mps)_available|"
                r"device(?:_count)?\s*\(\s*\)\s*==\s*0",
                re.I), "BackendUnavailable"),
    # "device_count < N" — requires ≥N devices; tag as RequiresMultiGPU
    # (but BackendUnavailable when N==1 since that's "no device").
    (re.compile(r"device_count\s*<\s*[2-9]", re.I), "RequiresMultiGPU"),
    # "Less than N devices available" as a message.
    (re.compile(r"[Ll]ess\s+than\s+\d+\s+(?:devices?|GPUs?)",
                re.I), "RequiresMultiGPU"),
    # "HIP not available" / HIP toolchain missing — BackendUnavailable.
    (re.compile(r"HIP\s+not\s+available|no\s+HIP\s+devices?",
                re.I), "BackendUnavailable"),
    # "needs >=N backends" / "only one successful backend" — generic multi-
    # backend requirement messages that slipped past the earlier heuristics.
    (re.compile(r"needs?\s*(?:>=|at\s+least)\s*\d+\s*backends?|"
                r"only\s+one\s+successful\s+backend",
                re.I), "RequiresMultiGPU"),
    # "reshape returned the same Tensor object" / impl()-pointer aliasing
    # degenerate paths — implementation-defined, not a bug (as of the
    # 2026-08 skip audit these call sites are now explicitly tagged
    # SKIP_WITH_REASON(SkipReason::NotApplicable, ...), so this heuristic
    # only fires for any future untagged occurrences of the same pattern).
    (re.compile(r"impl\(\)\.get\(\)\s*==|same\s+Tensor\s+object",
                re.I), "NotApplicable"),
    # "Not enough memory for ..." — allocator / physical-memory resource
    # constraints (host doesn't have enough free VRAM/RAM right now), not a
    # missing kernel. Was previously mis-tagged KernelNotImplemented, which
    # made test_cuda_caching_allocator.cpp / test_rocm_caching_allocator.cpp
    # look like feature gaps instead of environment-capacity skips.
    (re.compile(r"[Nn]ot\s+enough\s+memory|insufficient\s+memory|out\s+of\s+memory",
                re.I), "ResourceConstraint"),
    # "Need at least N GPUs" — stricter than generic RequiresMultiGPU test.
    (re.compile(r"[Nn]eed\s+at\s+least\s+\d+\s+GPUs?",
                re.I), "RequiresMultiGPU"),
    # "X tests require non-CPU multi-device backend" — covers FSDP, TP, PP,
    # offload, DataParallel family.
    (re.compile(r"require\s+non-?CPU\s+(?:multi-)?device|non-?CPU\s+backend",
                re.I), "RequiresMultiGPU"),
    # "as_strided is CPU metadata op" — op is a no-op on GPU by design.
    (re.compile(r"[Cc]PU\s+metadata\s+op|metadata-?only",
                re.I), "KernelNotImplemented"),
    # "Float16 not supported on X" / "BFloat16 not supported on X" — backend
    # doesn't register the dtype.
    (re.compile(r"(?:Float16|BFloat16)\s+not\s+supported\s+on",
                re.I), "DtypeUnsupportedOnBackend"),
    # "create_backend symbol not found" — the backend .so was built but
    # doesn't export the loader symbol.
    (re.compile(r"symbol\s+not\s+found|dlsym\s+failed",
                re.I), "BackendUnavailable"),
    # "First init failed" / "rpc init failed" — distributed bootstrap.
    (re.compile(r"[Ii]nit\s+failed|distributed\s+bootstrap",
                re.I), "RequiresMultiGPU"),
    # "Unknown backend: X" — caller provided an unrecognized backend name.
    (re.compile(r"[Uu]nknown\s+backend",
                re.I), "BackendUnavailable"),
    # "if (dtype != Float32 && dtype != Float64 ... ) GTEST_SKIP()" — test
    # body is only defined for floating-point dtypes.
    (re.compile(r"dtype(?:\s*\(\s*\))?\s*!=\s*DType::(?:Float|BFloat)", re.I),
     "DtypeUnsupportedOnBackend"),
    # Integer-only or symmetric-dtype opt-outs like
    # "if (dtype == Int32 || dtype == Int64) GTEST_SKIP()".
    (re.compile(r"dtype(?:\s*\(\s*\))?\s*==\s*DType::(?:Int|UInt|Bool)", re.I),
     "DtypeUnsupportedOnBackend"),
    # Test probes for BLAS/LAPACK availability before doing linalg.
    (re.compile(r"!\s*has_(?:blas|lapack|mkl|cudnn|cublas|cusparse|cusolver)",
                re.I), "KernelNotImplemented"),
    # Dtype capability guards surfaced via the multi_backend fixture flags
    # ("config_.dtype.is_floating", "supports_inf_nan", etc.).
    (re.compile(r"!\s*config_\.dtype\.(?:is_floating|is_signed|supports_inf_nan|is_complex)",
                re.I), "DtypeUnsupportedOnBackend"),
    (re.compile(r"(?:requires|only supports?|only supported).*"
                r"(?:floating|float\s*point|Int|integer|complex)",
                re.I), "DtypeUnsupportedOnBackend"),
    (re.compile(r"unsupported\s+dtype|dtype\s+not\s+supported",
                re.I), "DtypeUnsupportedOnBackend"),
    # Backend-info struct guards commonly seen in test fixtures.
    (re.compile(r"!\s*config_\.backend\.(?:is_available|available)",
                re.I), "BackendUnavailable"),
    # Unfinished / TODO skip messages.
    (re.compile(r"not yet implemented|TODO:|FIXME:|not implemented yet",
                re.I), "KernelNotImplemented"),
    # "if (dtype != DType::Int8) GTEST_SKIP()" and similar single-dtype
    # specialisation gates — mirrors the int-only subtests.
    (re.compile(r"dtype(?:\s*\(\s*\))?\s*!=\s*DType::(?:Int8|Int16|Int32|Int64|UInt8|UInt16|UInt32|UInt64|Bool)",
                re.I), "DtypeUnsupportedOnBackend"),
    # "if (dtype != DType::Complex64) GTEST_SKIP()" — complex-only tests.
    (re.compile(r"dtype(?:\s*\(\s*\))?\s*!=\s*DType::Complex",
                re.I), "DtypeUnsupportedOnBackend"),
    # Tests reserved for a specific dtype mentioned in the skip message.
    (re.compile(r"only\s+for\s+(?:Int|UInt|Float|BFloat|Complex|Bool)",
                re.I), "DtypeUnsupportedOnBackend"),
    # "only meaningful for Float32/Float64" — similar pattern, broader wording.
    (re.compile(r"only\s+meaningful\s+for\s+(?:Int|UInt|Float|BFloat|Complex|Bool)",
                re.I), "DtypeUnsupportedOnBackend"),
    # "Test only meaningful for dtype X" — captures phrasing that doesn't name
    # specific dtype but still gates on floating-point-ness.
    (re.compile(r"(?:Inf|NaN)\s+only\s+(?:meaningful|valid)\s+for",
                re.I), "DtypeUnsupportedOnBackend"),
    # "skip if device.type != CPU" — op is CPU-only. This is usually a bug
    # (GPU impl missing) but tagging it surfaces the gap for triage.
    (re.compile(r"device\.type\s*!=\s*Device::Type::CPU",
                re.I), "KernelNotImplemented"),
    # Fixture-level availability flags (rocm_available, cuda_available, etc.)
    # set once in SetUp and consulted per test.
    (re.compile(r"!\s*(?:rocm|cuda|vulkan|oneapi|mps|gpu)_available",
                re.I), "BackendUnavailable"),
    # Skip when the test is gated behind a specific device being present.
    (re.compile(r"device_count\s*\(\s*\)\s*<\s*\d+",
                re.I), "RequiresMultiGPU"),
    # "Need 2 GPUs for test" / "Requires N GPUs"
    (re.compile(r"need\s*\d+\+?\s*(?:GPU|device)|requires\s+\d+\+?\s*(?:GPU|device)",
                re.I), "RequiresMultiGPU"),
    # "device.type == CPU" exclusion — test wants GPU specifically.
    (re.compile(r"device\.type\s*==\s*Device::Type::CPU",
                re.I), "BackendUnavailable"),
    # Integer gradients / non-differentiable dtype skips.
    (re.compile(r"gradient.*(?:Int|Integer)|(?:Int|Integer).*gradient|"
                r"not differentiable",
                re.I), "DtypeUnsupportedOnBackend"),
    # "CPU reference failed" / "reference compute failed" — a fallback
    # where the in-test reference computation itself threw. Tag as
    # NumericalDivergence since the test can't assert anything useful.
    (re.compile(r"(CPU|reference)\s+(?:compute|call)?\s*failed",
                re.I), "NumericalDivergence"),
    # Optional feature probe: AWQ / GPTQ / other quantization feature not
    # compiled in. Tag as KernelNotImplemented.
    (re.compile(r"(?:AWQ|GPTQ|[Ff]eature)\s+\w+\s+unavailable",
                re.I), "KernelNotImplemented"),
    # Dtype-literal if-check preceding an empty GTEST_SKIP()  — covers
    # "if (dtype == DType::Float32) { ... } else { GTEST_SKIP(); }" style
    # guards where the message was omitted. The 120-char context catches
    # the preceding if-block.
    (re.compile(r"dtype(?:\s*\(\s*\))?\s*==\s*DType::(?:Float|BFloat|Int|UInt|Bool|Complex)",
                re.I), "DtypeUnsupportedOnBackend"),
    # Distributed environment probes: RANK / WORLD_SIZE / MASTER_ADDR.
    (re.compile(r"(?:RANK|WORLD_SIZE|MASTER_ADDR|MASTER_PORT|rank_env|world_size_env)",
                re.I), "RequiresMultiGPU"),
    (re.compile(r"requires\s+(?:exactly\s+)?\d+\s+rank|need\s+\d+\+?\s+(?:processes|ranks)",
                re.I), "RequiresMultiGPU"),
    (re.compile(r"world_size_\s*(?:!=|<|<=)\s*\d+",
                re.I), "RequiresMultiGPU"),
    # catch-clause skip with the exception message — treat as KernelNotImplemented
    # since the op raised before we could run it.
    (re.compile(r"catch\s*\([^)]*\)\s*\{[^}]*GTEST_SKIP\s*\(\s*\)\s*<<\s*e\.what",
                re.I | re.DOTALL), "KernelNotImplemented"),
    # Broader catch-and-skip that the restrictive regex above misses because
    # the `catch` line sits >120 chars before GTEST_SKIP (the window size in
    # this scanner). Use the `e.what()` hint directly in the skip body as
    # the signal — only a caught exception carries that call.
    (re.compile(r"GTEST_SKIP\s*\(\s*\)\s*<<[^;]*e\.what\s*\(\s*\)",
                re.I), "KernelNotImplemented"),
    # Explicit "device not available" / "not available, skipping" messages.
    (re.compile(r"device\s+not\s+available|not\s+available,\s*skipping",
                re.I), "BackendUnavailable"),
    # "GPU path currently only wired for X" — feature gated to specific GPU
    # backends, equivalent to KernelNotImplemented for other devices.
    (re.compile(r"only\s+wired\s+for|only\s+implemented\s+for|"
                r"GPU\s+path\s+(?:currently\s+)?only",
                re.I), "KernelNotImplemented"),
    # "device_.type != Device::Type::CUDA" and similar equality chains
    # gating by specific GPU presence.
    (re.compile(r"device_?\.?type\s*!=\s*Device::Type::(?:CUDA|ROCm|Vulkan|OneAPI|MPS)",
                re.I), "KernelNotImplemented"),
    # Cross-device / multi-device tests that degenerate on single-GPU hosts.
    (re.compile(r"cross[- ]device|multiple\s+devices|requires\s+multiple\s+(?:devices|GPUs)",
                re.I), "RequiresMultiGPU"),
    # Tests that require a distributed process group but don't have one set.
    (re.compile(r"requires\s+distributed\s+process\s+group|process_group",
                re.I), "RequiresMultiGPU"),
    # Inverted-expectation tests: "DeviceNotAvailable_CUDA" fires a skip when
    # CUDA *is* available. Tag as KnownBug so they surface during review.
    (re.compile(r"is\s+available,\s+cannot\s+test\s+unavailability",
                re.I), "KnownBug"),
    # "gradcheck needs complex-output support" and similar blocked-by-phase
    # markers — tests parked behind a follow-up implementation.
    (re.compile(r"gradcheck\s+needs|needs\s+complex-output|\(Phase\s*\d+\)",
                re.I), "KernelNotImplemented"),
    # "Complex types may not be supported" / "complex unsupported" phrases.
    (re.compile(r"[Cc]omplex\s+types?\s+(?:may\s+not\s+be\s+supported|unsupported|not\s+supported)",
                re.I), "DtypeUnsupportedOnBackend"),
    # "No OneAPI devices available" / "No CUDA devices" — device-count=0
    # variants not yet covered by the earlier "not available" heuristic.
    (re.compile(r"No\s+(?:OneAPI|CUDA|ROCm|Vulkan|GPU|MPS)\s+(?:devices?|backend)",
                re.I), "BackendUnavailable"),
    # "API not available" / feature-function missing from the build.
    (re.compile(r"API\s+not\s+available|function\s+(?:not\s+)?available|API\s+unavailable",
                re.I), "KernelNotImplemented"),
    # "No backends accepted the kernel" — custom-op parity tests gate on
    # successful registration on at least one non-CPU backend.
    (re.compile(r"No\s+backends?\s+accepted|no\s+backend\s+supports",
                re.I), "KernelNotImplemented"),
    # DataParallel / DDP requires a GPU backend.
    (re.compile(r"DataParallel\s+requires|DDP\s+requires\s+(?:a\s+)?GPU|requires\s+(?:a\s+)?GPU\s+backend",
                re.I), "BackendUnavailable"),
    # "Cannot allocate on X" from catch clauses.
    (re.compile(r"Cannot\s+allocate\s+on|allocation\s+failed",
                re.I), "KernelNotImplemented"),
    # Gradcheck / FD precision — test uses finite differences that can't survive FP16 rounding.
    (re.compile(r"gradcheck.*precision|finite[- ]?diff.*precision|higher precision|FP32\+?\s*precision|Float32\+?\s*precision", re.I),
     "GradcheckFDPrecision"),
    # Precision-bound numerical tests that aren't gradcheck.
    (re.compile(r"precision.*Float16|Float16.*(?:precision|imprecise)|numerical.*Float16|Float16.*numerical", re.I),
     "NumericalDivergence"),
    # Dtype unsupported for this kernel.
    (re.compile(r"not supported .* (?:for|on) (?:Float16|BFloat16|Int|dtype)|dtype.*unsupported|unsupported dtype", re.I),
     "DtypeUnsupportedOnBackend"),
    # Known bug markers.
    (re.compile(r"known bug|issue\s*#\d+|xfail", re.I), "KnownBug"),
]

# Python skip patterns.
PY_SKIP_RE = re.compile(
    r"\b(?:pytest\.skip\s*\(|pytest\.mark\.skip(?:if)?\s*\(|unittest\.skip(?:If|Unless)?\s*\(|@pytest\.mark\.skip)",
    re.MULTILINE,
)

# Matches (in priority order) string literals, char literals, //-line
# comments, and /* */ block comments. Used to blank out comment text before
# scanning for GTEST_SKIP/SKIP_WITH_REASON, so prose describing a removed or
# historical skip (e.g. "previously wrapped in try{...}catch(...){GTEST_SKIP(
# ...)}") isn't counted as a live skip. String/char literals are matched (and
# left untouched) rather than skipped over, so a `//` or `/*` inside a string
# literal doesn't get misread as the start of a comment.
_CXX_COMMENT_OR_STRING_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
    re.DOTALL,
)


def _strip_cxx_comments(text: str) -> str:
    def repl(m: re.Match) -> str:
        s = m.group(0)
        if s.startswith("//") or s.startswith("/*"):
            # Blank out comment text but keep newlines, so line-relative
            # offsets used elsewhere (e.g. the 240-char context window)
            # still land in equivalent positions.
            return re.sub(r"[^\n]", " ", s)
        return s  # string/char literal — keep as-is

    return _CXX_COMMENT_OR_STRING_RE.sub(repl, text)


def scan_cxx(path: Path) -> dict:
    text = path.read_text(errors="replace")
    text = _strip_cxx_comments(text)
    reasons = Counter()
    # Every SKIP_WITH_REASON invocation is a tagged skip.
    for m in CXX_SKIP_WITH_REASON_RE.finditer(text):
        reasons[m.group(1)] += 1
    # Raw GTEST_SKIP callsites: check for explicit [SkipReason::X] marker
    # first, then apply message-text heuristics, then inspect a trailing
    # context window for structural hints (e.g. the preceding predicate),
    # and finally fall back to UNTAGGED.
    for m in CXX_SKIP_RE.finditer(text):
        match = m.group(0)
        # Include ~240 chars of preceding context so we can detect bare
        # "if (backends.size() < 2) GTEST_SKIP();" forms without messages,
        # and catch dtype-if-elif chains where the bare `else { GTEST_SKIP(); }`
        # sits several lines below the triggering `if (dtype == DType::...)`.
        ctx_start = max(0, m.start() - 240)
        ctx = text[ctx_start:m.end()]
        msg_tag = CXX_REASON_IN_MSG_RE.search(match)
        if msg_tag:
            reasons[msg_tag.group(1)] += 1
            continue
        heur_hit = None
        for pattern, label in MSG_HEURISTICS:
            if pattern.search(ctx):
                heur_hit = label
                break
        reasons[heur_hit or "UNTAGGED"] += 1
    return dict(reasons)


def scan_py(path: Path) -> int:
    text = path.read_text(errors="replace")
    return len(PY_SKIP_RE.findall(text))


def walk() -> dict:
    cxx_by_file: dict[str, dict] = {}
    py_by_file: dict[str, int] = {}
    for p in TESTS_DIR.rglob("*.cpp"):
        counts = scan_cxx(p)
        if counts:
            cxx_by_file[str(p.relative_to(REPO_ROOT))] = counts
    for p in TESTS_DIR.rglob("*.py"):
        n = scan_py(p)
        if n:
            py_by_file[str(p.relative_to(REPO_ROOT))] = n
    return {"cxx": cxx_by_file, "py": py_by_file}


def aggregate(report: dict) -> dict:
    cxx_totals: Counter = Counter()
    for counts in report["cxx"].values():
        for reason, n in counts.items():
            cxx_totals[reason] += n
    py_total = sum(report["py"].values())
    return {"cxx_by_reason": dict(cxx_totals), "py_total": py_total}


def print_summary(report: dict, totals: dict) -> None:
    print("== Skip tally ==")
    print()
    print("C++ GTEST_SKIP by reason:")
    for reason, n in sorted(totals["cxx_by_reason"].items(), key=lambda kv: -kv[1]):
        tag = "" if reason in KNOWN_REASONS or reason == "UNTAGGED" else "  [unknown reason]"
        print(f"  {reason:32s} {n:6d}{tag}")
    print()
    print(f"Python test skips: {totals['py_total']}")
    print()

    untagged_files = [
        (f, counts["UNTAGGED"])
        for f, counts in report["cxx"].items()
        if counts.get("UNTAGGED")
    ]
    if untagged_files:
        untagged_files.sort(key=lambda kv: -kv[1])
        print(f"Top files with UNTAGGED skips (first 20):")
        for f, n in untagged_files[:20]:
            print(f"  {n:4d}  {f}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write full report JSON to this path")
    ap.add_argument("--max-untagged", type=int, default=None,
                    help="fail if UNTAGGED C++ skip count exceeds this threshold")
    args = ap.parse_args()

    report = walk()
    totals = aggregate(report)

    if args.json:
        args.json.write_text(json.dumps({"report": report, "totals": totals}, indent=2))
    print_summary(report, totals)

    untagged = totals["cxx_by_reason"].get("UNTAGGED", 0)
    if args.max_untagged is not None and untagged > args.max_untagged:
        print(f"FAIL: {untagged} untagged C++ skips exceeds threshold {args.max_untagged}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
