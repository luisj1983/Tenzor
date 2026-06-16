#!/usr/bin/env bash
#
# test_all_backends.sh — canonical LOCAL full-matrix backend runner.
#
# This machine is assumed to have ALL backends installed (CUDA, ROCm, Vulkan,
# OneAPI). The script:
#   1. Configures + builds every backend and the tests.
#   2. Runs the cross-backend parity suite with TENZOR_REQUIRE_MULTI_BACKEND=1
#      so a missing/broken backend is a hard FAILURE, not a silent skip.
#   3. (--record-goldens) re-records committed goldens for CPU-only CI.
#   4. (--perf-baseline)   regenerates the per-host perf baseline.
#
# Per CLAUDE.md testing constraints the AMD/ROCm driver is fragile under
# parallel test processes, so parity tests run -j1 (sequential), ONE shell at
# a time. Honors TENZOR_SKIP_BACKENDS (e.g. "cuda,rocm") to drop a known-bad
# backend without editing this file — it disables both the build flag and the
# ctest exclusion for that backend.
#
# Usage:
#   scripts/test_all_backends.sh                 # build + parity suite
#   scripts/test_all_backends.sh --record-goldens
#   scripts/test_all_backends.sh --perf-baseline
#   scripts/test_all_backends.sh --build-dir build-all --record-goldens
#
# Idempotent: re-running reconfigures in place and rebuilds incrementally.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="build"
DO_RECORD_GOLDENS=0
DO_PERF_BASELINE=0
DO_PARITY=1   # default action: run the parity suite

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)      BUILD_DIR="$2"; shift 2 ;;
        --record-goldens) DO_RECORD_GOLDENS=1; DO_PARITY=0; shift ;;
        --perf-baseline)  DO_PERF_BASELINE=1;  DO_PARITY=0; shift ;;
        --parity)         DO_PARITY=1; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

BUILD_PATH="$PROJECT_ROOT/$BUILD_DIR"

# ---------------------------------------------------------------------------
# Resolve which backends to enable, honoring TENZOR_SKIP_BACKENDS.
# ---------------------------------------------------------------------------
SKIP="${TENZOR_SKIP_BACKENDS:-}"
backend_enabled() {
    # $1 = backend name (cuda|rocm|vulkan|oneapi). Returns 0 unless listed in
    # the comma-separated TENZOR_SKIP_BACKENDS.
    local b="$1"
    IFS=',' read -ra _skips <<< "$SKIP"
    for s in "${_skips[@]}"; do
        [[ "$(echo "$s" | tr '[:upper:]' '[:lower:]' | tr -d ' ')" == "$b" ]] && return 1
    done
    return 0
}

flag() { backend_enabled "$1" && echo ON || echo OFF; }

CUDA=$(flag cuda)
ROCM=$(flag rocm)
VULKAN=$(flag vulkan)
ONEAPI=$(flag oneapi)

echo "================================================"
echo "Tenzor full-matrix backend runner"
echo "  build dir : $BUILD_PATH"
echo "  CUDA=$CUDA ROCm=$ROCM Vulkan=$VULKAN OneAPI=$ONEAPI"
[[ -n "$SKIP" ]] && echo "  TENZOR_SKIP_BACKENDS=$SKIP"
echo "================================================"

# Build a ctest -E exclusion for skipped backends so their tests don't run.
CTEST_EXCLUDE=""
for b in cuda rocm vulkan oneapi; do
    if ! backend_enabled "$b"; then
        CTEST_EXCLUDE="${CTEST_EXCLUDE:+$CTEST_EXCLUDE|}$b"
    fi
done

# ---------------------------------------------------------------------------
# (1) Configure + build.
# ---------------------------------------------------------------------------
echo "=== Configuring ($BUILD_DIR) ==="
cmake -B "$BUILD_PATH" -S "$PROJECT_ROOT" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENZOR_BUILD_CUDA="$CUDA" \
    -DTENZOR_BUILD_ROCM="$ROCM" \
    -DTENZOR_BUILD_VULKAN="$VULKAN" \
    -DTENZOR_BUILD_ONEAPI="$ONEAPI" \
    -DTENZOR_BUILD_PYTHON=OFF \
    -DTENZOR_BUILD_TESTS=ON \
    -DTENZOR_BUILD_BENCHMARKS=OFF \
    -DTENZOR_BUILD_EXAMPLES=OFF \
    -DTENZOR_BUILD_MODEL_HUB=OFF

echo "=== Building (this may take a while) ==="
# -j1 keeps memory/driver pressure sane on machines juggling all toolchains;
# bump with NINJA_JOBS if your box can take it.
ninja -C "$BUILD_PATH" -j"${NINJA_JOBS:-1}"

# ---------------------------------------------------------------------------
# (3) --record-goldens : refresh tests/backend_parity/golden/
# ---------------------------------------------------------------------------
if [[ "$DO_RECORD_GOLDENS" == 1 ]]; then
    echo "=== Recording goldens (TENZOR_RECORD_GOLDENS=1) ==="
    echo "Goldens land in: $PROJECT_ROOT/tests/backend_parity/golden/"
    ( cd "$BUILD_PATH" && \
      TENZOR_RECORD_GOLDENS=1 \
      TENZOR_GOLDEN_DIR="$PROJECT_ROOT/tests/backend_parity/golden" \
      ctest -L backend_parity \
            ${CTEST_EXCLUDE:+-E "$CTEST_EXCLUDE"} \
            -j1 --output-on-failure --timeout 900 )
    echo "Goldens recorded. Review & commit the .gold files under"
    echo "  tests/backend_parity/golden/"
    exit 0
fi

# ---------------------------------------------------------------------------
# (4) --perf-baseline : regenerate the per-host perf baseline
# ---------------------------------------------------------------------------
if [[ "$DO_PERF_BASELINE" == 1 ]]; then
    echo "=== Regenerating per-host perf baseline ==="
    ninja -C "$BUILD_PATH" test_performance_regression
    python3 "$PROJECT_ROOT/tools/regen_perf_baseline.py" --build-dir "$BUILD_DIR"
    echo "Per-host baseline updated in"
    echo "  tests/backend_parity/baselines/perf_baseline.json"
    echo "Review & commit the change for this host."
    exit 0
fi

# ---------------------------------------------------------------------------
# (2) Default: run the cross-backend parity suite (gating semantics).
# ---------------------------------------------------------------------------
if [[ "$DO_PARITY" == 1 ]]; then
    echo "=== Running cross-backend parity suite (-j1, multi-backend required) ==="
    ( cd "$BUILD_PATH" && \
      TENZOR_REQUIRE_MULTI_BACKEND=1 \
      ctest -L backend_parity \
            ${CTEST_EXCLUDE:+-E "$CTEST_EXCLUDE"} \
            -j1 --output-on-failure --timeout 900 )
    echo "================================================"
    echo "Cross-backend parity suite PASSED."
    echo "================================================"
fi
