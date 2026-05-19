/**
 * @file test_simd_elementwise_half.cpp
 * @brief Tests for F16/BF16 SIMD widen-convert-narrow in simd_elementwise.hpp
 *
 * Task 4.3: Vectorized elementwise ops for Float16 (F16C+AVX2) and
 * BFloat16 (AVX2 bit-shifts).
 */

#include <gtest/gtest.h>
#include <chrono>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor { void initialize(); }
namespace tz = ::tenzor;

class SimdElementwiseHalfEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const g_env =
    ::testing::AddGlobalTestEnvironment(new SimdElementwiseHalfEnv);

TEST(SimdElementwiseHalf, Float16AddCorrectness) {
    auto a = tz::full({1024}, 2.0, tz::DType::Float16);
    auto b = tz::full({1024}, 3.0, tz::DType::Float16);
    auto c = a + b;
    auto p = c.cpu().to(tz::DType::Float64).data<double>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 5.0, 1e-3);
}

TEST(SimdElementwiseHalf, BFloat16MulCorrectness) {
    auto a = tz::full({1024}, 2.0, tz::DType::BFloat16);
    auto b = tz::full({1024}, 3.0, tz::DType::BFloat16);
    auto c = a * b;
    auto p = c.cpu().to(tz::DType::Float64).data<double>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 6.0, 1e-2);
}

TEST(SimdElementwiseHalf, Float16SubCorrectness) {
    auto a = tz::full({1024}, 7.0, tz::DType::Float16);
    auto b = tz::full({1024}, 4.0, tz::DType::Float16);
    auto c = a - b;
    auto p = c.cpu().to(tz::DType::Float64).data<double>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 3.0, 1e-3);
}

TEST(SimdElementwiseHalf, BFloat16AddCorrectness) {
    auto a = tz::full({1024}, 1.5, tz::DType::BFloat16);
    auto b = tz::full({1024}, 2.5, tz::DType::BFloat16);
    auto c = a + b;
    auto p = c.cpu().to(tz::DType::Float64).data<double>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 4.0, 1e-2);
}

TEST(SimdElementwiseHalf, Float16NonMultiple8) {
    // Test that scalar tail (n not divisible by 8) works correctly
    auto a = tz::full({13}, 2.0, tz::DType::Float16);
    auto b = tz::full({13}, 3.0, tz::DType::Float16);
    auto c = a + b;
    auto p = c.cpu().to(tz::DType::Float64).data<double>();
    for (int i = 0; i < 13; ++i) EXPECT_NEAR(p[i], 5.0, 1e-3);
}

TEST(SimdElementwiseHalf, BFloat16NonMultiple8) {
    auto a = tz::full({13}, 2.0, tz::DType::BFloat16);
    auto b = tz::full({13}, 3.0, tz::DType::BFloat16);
    auto c = a * b;
    auto p = c.cpu().to(tz::DType::Float64).data<double>();
    for (int i = 0; i < 13; ++i) EXPECT_NEAR(p[i], 6.0, 1e-2);
}

TEST(SimdElementwiseHalf, PerfFloat16Add) {
    const int N = 1024 * 1024;
    auto a = tz::full({N}, 1.5, tz::DType::Float16);
    auto b = tz::full({N}, 2.5, tz::DType::Float16);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto c = a + b;
    (void)c.cpu();  // force materialization
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_LT(ms, 50.0) << "F16 elementwise add too slow: " << ms
                        << " ms (scalar would be ~hundreds).";
}

TEST(SimdElementwiseHalf, PerfBFloat16Mul) {
    const int N = 1024 * 1024;
    auto a = tz::full({N}, 1.5, tz::DType::BFloat16);
    auto b = tz::full({N}, 2.5, tz::DType::BFloat16);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto c = a * b;
    (void)c.cpu();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_LT(ms, 50.0) << "BF16 elementwise mul too slow: " << ms << " ms";
}
