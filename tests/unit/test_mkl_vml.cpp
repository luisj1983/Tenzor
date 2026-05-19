/**
 * @file test_mkl_vml.cpp
 * @brief Tests for MKL VML transcendental routing in math.cpp
 *
 * Task 6.1: Verify exp/log/sin/cos/erf/etc. route through MKL VML
 * when TENZOR_USE_MKL is defined, with correctness + perf checks.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor { void initialize(); }
namespace tz = ::tenzor;

class MklVmlEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const g_env =
    ::testing::AddGlobalTestEnvironment(new MklVmlEnv);

// ============================================================================
// Correctness tests
// ============================================================================

TEST(MklVml, Float32ExpCorrectness) {
    auto a = tz::full({1024}, 1.0, tz::DType::Float32);
    auto y = tz::exp(a);
    auto p = y.cpu().data<float>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], std::exp(1.0f), 1e-5f);
}

TEST(MklVml, Float64ExpCorrectness) {
    auto a = tz::full({1024}, 1.0, tz::DType::Float64);
    auto y = tz::exp(a);
    auto p = y.cpu().data<double>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], std::exp(1.0), 1e-12);
}

TEST(MklVml, Float32LogCorrectness) {
    auto a = tz::full({1024}, std::exp(1.0f), tz::DType::Float32);
    auto y = tz::log(a);
    auto p = y.cpu().data<float>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 1.0f, 1e-5f);
}

TEST(MklVml, Float32SinCorrectness) {
    // sin(pi/6) = 0.5
    float pi_over_6 = static_cast<float>(M_PI / 6.0);
    auto a = tz::full({1024}, static_cast<double>(pi_over_6), tz::DType::Float32);
    auto y = tz::sin(a);
    auto p = y.cpu().data<float>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 0.5f, 1e-5f);
}

TEST(MklVml, Float32CosCorrectness) {
    // cos(0) = 1
    auto a = tz::zeros({1024}, tz::DType::Float32);
    auto y = tz::cos(a);
    auto p = y.cpu().data<float>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 1.0f, 1e-5f);
}

TEST(MklVml, Float32ErfCorrectness) {
    // erf(0) = 0, erf(large) ≈ 1
    auto a = tz::zeros({1024}, tz::DType::Float32);
    auto y = tz::erf(a);
    auto p = y.cpu().data<float>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 0.0f, 1e-6f);
}

TEST(MklVml, Float32Log2Correctness) {
    auto a = tz::full({1024}, 8.0, tz::DType::Float32);
    auto y = tz::log2(a);
    auto p = y.cpu().data<float>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 3.0f, 1e-5f);
}

TEST(MklVml, Float32Log10Correctness) {
    auto a = tz::full({1024}, 1000.0, tz::DType::Float32);
    auto y = tz::log10(a);
    auto p = y.cpu().data<float>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 3.0f, 1e-5f);
}

TEST(MklVml, Float32Expm1Correctness) {
    // expm1(0) = 0
    auto a = tz::zeros({1024}, tz::DType::Float32);
    auto y = tz::expm1(a);
    auto p = y.cpu().data<float>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 0.0f, 1e-6f);
}

TEST(MklVml, Float32Log1pCorrectness) {
    // log1p(0) = 0
    auto a = tz::zeros({1024}, tz::DType::Float32);
    auto y = tz::log1p(a);
    auto p = y.cpu().data<float>();
    for (int i = 0; i < 1024; ++i) EXPECT_NEAR(p[i], 0.0f, 1e-6f);
}

// ============================================================================
// Performance test
// ============================================================================

TEST(MklVml, Float32ExpPerf) {
    const int N = 1024 * 1024;
    auto a = tz::full({N}, 1.0, tz::DType::Float32);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto y = tz::exp(a);
    (void)y.cpu();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_LT(ms, 30.0) << "F32 exp(1M) too slow: " << ms << " ms";
}

TEST(MklVml, Float64ExpPerf) {
    const int N = 1024 * 1024;
    auto a = tz::full({N}, 1.0, tz::DType::Float64);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto y = tz::exp(a);
    (void)y.cpu();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_LT(ms, 60.0) << "F64 exp(1M) too slow: " << ms << " ms";
}
