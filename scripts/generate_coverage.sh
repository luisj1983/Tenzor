#!/bin/bash
# Coverage report generation script for Tenzor
set -e

PROJECT_ROOT="/home/lee/Projects/Tenzor"
BUILD_DIR="$PROJECT_ROOT/build"
COVERAGE_DIR="$PROJECT_ROOT/coverage_report"

echo "=== Tenzor Coverage Report Generator ==="
echo ""

# Step 1: Clean previous coverage data (keep .gcno files - they're created at compile time)
echo "1. Cleaning previous coverage data..."
find "$BUILD_DIR" -name "*.gcda" -delete
find "$BUILD_DIR" -name "*.gcov" -delete
rm -rf "$COVERAGE_DIR"
mkdir -p "$COVERAGE_DIR"

# Step 2: Run all tests to generate coverage data
echo ""
echo "2. Running tests to generate coverage data..."

cd "$PROJECT_ROOT/bin"

# Run unit tests
echo "  - Running unit tests..."
./tenzor_unit_tests --gtest_output=xml:unit_results.xml 2>&1 | grep -E "(PASSED|FAILED|tests from)" || true

# Run integration tests
echo "  - Running integration tests..."
./tenzor_integration_tests --gtest_output=xml:integration_results.xml 2>&1 | grep -E "(PASSED|FAILED|tests from)" || true

# Run quantization tests
echo "  - Running quantization tests..."
./test_quantization --gtest_output=xml:quantization_results.xml 2>&1 | grep -E "(PASSED|FAILED|tests from)" || true

# Run backend tests
echo "  - Running backend tests..."
./test_phase11_backends --gtest_filter="-*ROCm*" --gtest_output=xml:phase11_results.xml 2>&1 | grep -E "(PASSED|FAILED|tests from)" || true

# Step 3: Generate coverage with gcov
echo ""
echo "3. Generating coverage data with gcov..."

cd "$BUILD_DIR"

# Find all .gcda files and run gcov on them
find . -name "*.gcda" | while read gcda_file; do
    dir=$(dirname "$gcda_file")
    base=$(basename "$gcda_file" .gcda)

    # Run gcov
    gcov -o "$dir" "$dir/$base" > /dev/null 2>&1 || true
done

# Step 4: Collect and summarize coverage
echo ""
echo "4. Collecting coverage statistics..."

# Count total .gcov files
total_files=$(find . -name "*.gcov" | wc -l)

echo "  Generated $total_files .gcov files"

# Simple coverage summary using gcov output
echo ""
echo "=== Coverage Summary ==="

# Parse gcov files for line coverage
total_lines=0
covered_lines=0

for gcov_file in $(find "$BUILD_DIR/src" -name "*.gcov" 2>/dev/null | grep -v "/usr/include" | head -100); do
    if [ -f "$gcov_file" ]; then
        # Count executed and total lines
        lines=$(grep -c "^  *[0-9]" "$gcov_file" 2>/dev/null || echo 0)
        executed=$(grep -c "^  *[1-9]" "$gcov_file" 2>/dev/null || echo 0)

        total_lines=$((total_lines + lines))
        covered_lines=$((covered_lines + executed))
    fi
done

if [ $total_lines -gt 0 ]; then
    coverage_pct=$(awk "BEGIN {printf \"%.1f\", ($covered_lines / $total_lines) * 100}")
    echo "Lines covered: $covered_lines / $total_lines ($coverage_pct%)"
else
    echo "No coverage data found. Make sure to build with -DCMAKE_BUILD_TYPE=Coverage"
fi

echo ""
echo "Coverage data generated in: $BUILD_DIR"
echo "To view detailed coverage, examine .gcov files in the build directory"
echo ""
echo "=== Coverage generation complete ==="
