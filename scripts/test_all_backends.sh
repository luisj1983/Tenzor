#!/bin/bash
# Test all existing test suites on all available backends
# This script runs each test binary multiple times, once per backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$PROJECT_ROOT/bin"
RESULTS_DIR="$PROJECT_ROOT/test_results/backend_parity"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

# Available backends
BACKENDS=("cpu" "cuda" "vulkan" "oneapi")

# Test binaries to run (add more as needed)
TEST_BINARIES=(
    "tenzor_unit_tests"
    "tenzor_integration_tests"
    "test_ciou_loss"
    "test_slice_backend_parity"
    "test_phase11_backends"
)

echo "================================================"
echo "Backend Parity Test Suite"
echo "================================================"
echo "Testing ${#TEST_BINARIES[@]} test suites on ${#BACKENDS[@]} backends"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Summary tracking
declare -A RESULTS

# Function to run a test on a specific backend
run_test() {
    local test_name=$1
    local backend=$2
    local test_binary="$BIN_DIR/$test_name"

    if [[ ! -f "$test_binary" ]]; then
        echo -e "${YELLOW}[SKIP]${NC} $test_name not found"
        return
    fi

    echo -e "${YELLOW}[RUN]${NC} $test_name on $backend backend"

    # Set environment variable to prefer this backend
    export TENZOR_DEFAULT_BACKEND=$backend

    # Run the test
    local output_file="$RESULTS_DIR/${test_name}_${backend}.log"
    local xml_file="$RESULTS_DIR/${test_name}_${backend}.xml"

    if timeout 300 "$test_binary" --gtest_output=xml:"$xml_file" > "$output_file" 2>&1; then
        echo -e "${GREEN}[PASS]${NC} $test_name on $backend"
        RESULTS["${test_name}_${backend}"]="PASS"

        # Extract test counts
        local passed=$(grep -oP '\[\s+PASSED\s+\]\s+\K\d+' "$output_file" | tail -1 || echo "0")
        echo "       Tests passed: $passed"
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            echo -e "${RED}[TIMEOUT]${NC} $test_name on $backend (300s)"
            RESULTS["${test_name}_${backend}"]="TIMEOUT"
        else
            echo -e "${RED}[FAIL]${NC} $test_name on $backend"
            RESULTS["${test_name}_${backend}"]="FAIL"

            # Show last few lines of error
            echo "       Last 5 lines:"
            tail -5 "$output_file" | sed 's/^/       /'
        fi
    fi
    echo ""
}

# Run all tests on all backends
for test_name in "${TEST_BINARIES[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        run_test "$test_name" "$backend"
    done
done

# Generate summary report
echo "================================================"
echo "SUMMARY REPORT"
echo "================================================"
echo ""

# Create markdown report
REPORT_FILE="$RESULTS_DIR/parity_report.md"
cat > "$REPORT_FILE" <<EOF
# Backend Parity Test Report
**Generated**: $(date)
**Backends Tested**: ${BACKENDS[*]}

## Test Results

| Test Suite | CPU | CUDA | Vulkan | OneAPI |
|------------|-----|------|--------|--------|
EOF

# Fill in the table
for test_name in "${TEST_BINARIES[@]}"; do
    if [[ ! -f "$BIN_DIR/$test_name" ]]; then
        continue
    fi

    echo -n "| $test_name " >> "$REPORT_FILE"

    for backend in "${BACKENDS[@]}"; do
        result="${RESULTS[${test_name}_${backend}]:-SKIP}"
        case "$result" in
            PASS)
                echo -n "| ✅ PASS " >> "$REPORT_FILE"
                ;;
            FAIL)
                echo -n "| ❌ FAIL " >> "$REPORT_FILE"
                ;;
            TIMEOUT)
                echo -n "| ⏱️ TIMEOUT " >> "$REPORT_FILE"
                ;;
            SKIP)
                echo -n "| ⏭️ SKIP " >> "$REPORT_FILE"
                ;;
        esac
    done
    echo "|" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" <<EOF

## Legend
- ✅ PASS: All tests passed
- ❌ FAIL: One or more tests failed
- ⏱️ TIMEOUT: Tests exceeded 300 second timeout
- ⏭️ SKIP: Backend not available or test not found

## Detailed Logs
See individual log files in: \`$RESULTS_DIR/\`

## Backend Parity Analysis

### Full Parity (all backends pass)
EOF

# Analyze results
for test_name in "${TEST_BINARIES[@]}"; do
    if [[ ! -f "$BIN_DIR/$test_name" ]]; then
        continue
    fi

    all_pass=true
    for backend in "${BACKENDS[@]}"; do
        if [[ "${RESULTS[${test_name}_${backend}]}" != "PASS" ]]; then
            all_pass=false
            break
        fi
    done

    if $all_pass; then
        echo "- ✅ $test_name" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" <<EOF

### Partial Parity (some backends pass)
EOF

for test_name in "${TEST_BINARIES[@]}"; do
    if [[ ! -f "$BIN_DIR/$test_name" ]]; then
        continue
    fi

    pass_count=0
    total_count=0

    for backend in "${BACKENDS[@]}"; do
        result="${RESULTS[${test_name}_${backend}]:-SKIP}"
        if [[ "$result" != "SKIP" ]]; then
            ((total_count++))
            if [[ "$result" == "PASS" ]]; then
                ((pass_count++))
            fi
        fi
    done

    if [[ $pass_count -gt 0 && $pass_count -lt $total_count ]]; then
        echo "- ⚠️ $test_name ($pass_count/$total_count backends passing)" >> "$REPORT_FILE"
    fi
done

# Display summary to console
cat "$REPORT_FILE"

echo ""
echo "================================================"
echo "Full report saved to: $REPORT_FILE"
echo "================================================"
