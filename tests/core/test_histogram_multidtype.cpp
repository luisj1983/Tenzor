/**
 * @file test_histogram_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for histogram computation
 *
 * Note: Float16 is skipped for value-checking tests due to precision.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class HistogramMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(HistogramMultiDTypeTest, UniformDataShape) {
    // Build input in Float32 on CPU, then move
    auto input = tenzor::arange(0.0f, 10.0f, 1.0f, DType::Float32, Device::cpu());
    input = input.to(device());
    if (dtype() != DType::Float32) {
        input = input.to(dtype());
    }

    auto [counts, edges] = tenzor::histogram(input, /*bins=*/5, /*min=*/0.0, /*max=*/10.0);

    EXPECT_EQ(counts.shape()[0], 5);
    EXPECT_EQ(edges.shape()[0], 6);  // bins + 1 edges
}

TEST_P(HistogramMultiDTypeTest, UniformDataCounts) {
    // Skip Float16 due to precision
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "Skipping value check for low-precision dtype";
    }

    auto input = tenzor::arange(0.0f, 10.0f, 1.0f, DType::Float32, Device::cpu());
    input = input.to(device());
    if (dtype() != DType::Float32) {
        input = input.to(dtype());
    }

    auto [counts, edges] = tenzor::histogram(input, /*bins=*/5, /*min=*/0.0, /*max=*/10.0);

    // Each bin should have 2 elements
    auto c_cpu = counts.to(Device::cpu());
    auto* c = c_cpu.data<int64_t>();
    for (int64_t i = 0; i < 5; i++) {
        EXPECT_EQ(c[i], 2);
    }
}

TEST_P(HistogramMultiDTypeTest, AutoRangeTotalCount) {
    // Skip Float16 due to precision
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "Skipping value check for low-precision dtype";
    }

    auto input = tenzor::arange(0.0f, 5.0f, 1.0f, DType::Float32, Device::cpu());
    input = input.to(device());
    if (dtype() != DType::Float32) {
        input = input.to(dtype());
    }

    auto [counts, edges] = tenzor::histogram(input, /*bins=*/5);

    EXPECT_EQ(counts.shape()[0], 5);
    EXPECT_EQ(edges.shape()[0], 6);

    // Total count should equal input size
    auto c_cpu = counts.to(Device::cpu());
    auto* c = c_cpu.data<int64_t>();
    int64_t total = 0;
    for (int64_t i = 0; i < 5; i++) {
        total += c[i];
    }
    EXPECT_EQ(total, 5);
}

TEST_P(HistogramMultiDTypeTest, SingleBinContainsAll) {
    auto input = createRandn({100});
    auto [counts, edges] = tenzor::histogram(input, /*bins=*/1);

    EXPECT_EQ(counts.shape()[0], 1);
    auto c_cpu = counts.to(Device::cpu());
    EXPECT_EQ(c_cpu.data<int64_t>()[0], 100);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HistogramMultiDTypeTest);
