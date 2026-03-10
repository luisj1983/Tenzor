#!/usr/bin/env bash
# ci_benchmark.sh - Build and run benchmarks, output JSON results
#
# Usage: ./scripts/ci_benchmark.sh [--build-dir DIR]
#
# Builds benchmark targets, runs them with --json, and merges results
# into benchmark_results/all.json

set -euo pipefail

BUILD_DIR="build"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_PATH="$PROJECT_ROOT/$BUILD_DIR"

if [[ ! -d "$BUILD_PATH" ]]; then
    echo "Error: Build directory '$BUILD_PATH' does not exist." >&2
    echo "Run 'cmake -B $BUILD_DIR -G Ninja' first." >&2
    exit 1
fi

# Step 1: Build benchmark targets
echo "=== Building benchmark targets ==="
cd "$BUILD_PATH"
ninja -j4 benchmark_ops benchmark_convolutions benchmark_training
echo "Build complete."

# Step 2: Create results directory
RESULTS_DIR="$BUILD_PATH/benchmark_results"
mkdir -p "$RESULTS_DIR"

# Step 3: Run benchmarks with --json output
BENCHMARKS=(
    "benchmark_ops:ops"
    "benchmark_convolutions:convolutions"
    "benchmark_training:training"
)

FAILED=0

for entry in "${BENCHMARKS[@]}"; do
    IFS=':' read -r binary name <<< "$entry"
    BINARY_PATH="$BUILD_PATH/bin/$binary"

    if [[ ! -x "$BINARY_PATH" ]]; then
        echo "Warning: $BINARY_PATH not found or not executable, skipping." >&2
        FAILED=1
        continue
    fi

    echo "=== Running $binary ==="
    if "$BINARY_PATH" --json > "$RESULTS_DIR/${name}.json" 2>/dev/null; then
        echo "  Saved: benchmark_results/${name}.json"
    else
        echo "  Error: $binary failed (exit code $?)" >&2
        # Write empty result on failure so merge doesn't break
        echo '{"suite_name": "'"$name"'", "benchmarks": []}' > "$RESULTS_DIR/${name}.json"
        FAILED=1
    fi
done

# Step 4: Merge all JSON results into all.json
echo "=== Merging results ==="
{
    echo '{'
    echo '  "timestamp": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",'
    echo '  "suites": ['

    FIRST=true
    for entry in "${BENCHMARKS[@]}"; do
        IFS=':' read -r _ name <<< "$entry"
        JSON_FILE="$RESULTS_DIR/${name}.json"

        if [[ ! -f "$JSON_FILE" ]]; then
            continue
        fi

        if [[ "$FIRST" == true ]]; then
            FIRST=false
        else
            echo '    ,'
        fi

        # Indent each line of the suite JSON by 4 spaces
        while IFS= read -r line; do
            echo "    $line"
        done < "$JSON_FILE"
    done

    echo '  ]'
    echo '}'
} > "$RESULTS_DIR/all.json"

echo "  Saved: benchmark_results/all.json"

if [[ "$FAILED" -ne 0 ]]; then
    echo ""
    echo "Warning: Some benchmarks failed. Check output above." >&2
    exit 1
fi

echo ""
echo "=== All benchmarks completed successfully ==="
