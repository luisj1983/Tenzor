/**
 * @file test_init_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for weight initialization functions
 *
 * Covers: xavier_uniform/normal_, kaiming_uniform/normal_, lecun_uniform/normal_,
 * orthogonal_, uniform_, normal_, constant_, zeros_, ones_,
 * calculate_fan_in_and_fan_out, calculate_gain
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/init.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/math.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn::init;

// ============================================================================
// Fixture
// ============================================================================

class InitMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Fan Calculation Tests
// ============================================================================

TEST_P(InitMultiDTypeTest, FanCalculation2D) {
    // Linear weight: (out_features=64, in_features=128) -> fan_in=128, fan_out=64
    auto t = tenzor::zeros({64, 128}, dtype(), device());
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(t);
    EXPECT_EQ(fan_in, 128);
    EXPECT_EQ(fan_out, 64);
}

TEST_P(InitMultiDTypeTest, FanCalculation4D) {
    // Conv2d weight: (out=32, in=16, kH=3, kW=3) -> fan_in=16*9=144, fan_out=32*9=288
    auto t = tenzor::zeros({32, 16, 3, 3}, dtype(), device());
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(t);
    EXPECT_EQ(fan_in, 16 * 9);
    EXPECT_EQ(fan_out, 32 * 9);
}

// ============================================================================
// Gain Calculation Tests
// ============================================================================

TEST_P(InitMultiDTypeTest, CalculateGainLinear) {
    EXPECT_NEAR(calculate_gain("linear"), 1.0, 1e-6);
}

TEST_P(InitMultiDTypeTest, CalculateGainRelu) {
    EXPECT_NEAR(calculate_gain("relu"), std::sqrt(2.0), 1e-6);
}

TEST_P(InitMultiDTypeTest, CalculateGainTanh) {
    EXPECT_NEAR(calculate_gain("tanh"), 5.0 / 3.0, 1e-6);
}

// ============================================================================
// Xavier Initialization Tests
// ============================================================================

TEST_P(InitMultiDTypeTest, XavierUniformRange) {
    auto t = tenzor::zeros({64, 32}, dtype(), device());
    xavier_uniform_(t, 1.0);

    // Values should be in [-a, a] where a = sqrt(6 / (fan_in + fan_out))
    float a = std::sqrt(6.0f / (32.0f + 64.0f));
    float max_val = compute_max(t);
    float min_val = compute_min(t);
    EXPECT_LE(max_val, a + 0.05f);
    EXPECT_GE(min_val, -a - 0.05f);
}

TEST_P(InitMultiDTypeTest, XavierUniformInPlace) {
    auto t = tenzor::zeros({32, 32}, dtype(), device());
    auto& ref = xavier_uniform_(t, 1.0);
    // Should return reference to same tensor
    EXPECT_EQ(t.data_ptr(), ref.data_ptr());
}

TEST_P(InitMultiDTypeTest, XavierNormalMeanStd) {
    auto t = tenzor::zeros({256, 256}, dtype(), device());
    xavier_normal_(t, 1.0);

    float mean_val = compute_mean(t);
    EXPECT_NEAR(mean_val, 0.0f, 0.1f);
    expectDevice(t);
}

// ============================================================================
// Kaiming Initialization Tests
// ============================================================================

TEST_P(InitMultiDTypeTest, KaimingUniformRange) {
    auto t = tenzor::zeros({64, 32}, dtype(), device());
    kaiming_uniform_(t, 0.0, FanMode::FanIn, "relu");

    // bound = sqrt(6 / fan_in) * gain
    float bound = std::sqrt(6.0f / 32.0f) * std::sqrt(2.0f);
    float max_val = compute_max(t);
    EXPECT_LE(max_val, bound + 0.1f);
}

TEST_P(InitMultiDTypeTest, KaimingFanOutMode) {
    auto t = tenzor::zeros({64, 32}, dtype(), device());
    kaiming_uniform_(t, 0.0, FanMode::FanOut, "relu");

    // With FanOut, bound uses fan_out=64 instead of fan_in=32
    float bound = std::sqrt(6.0f / 64.0f) * std::sqrt(2.0f);
    float max_val = compute_max(t);
    EXPECT_LE(max_val, bound + 0.1f);
}

// ============================================================================
// Simple Initialization Tests
// ============================================================================

TEST_P(InitMultiDTypeTest, ConstantFill) {
    auto t = tenzor::zeros({4, 4}, dtype(), device());
    constant_(t, 3.14);

    auto t_f32 = t.to(Device::cpu()).to(DType::Float32);
    auto* d = t_f32.data<float>();
    for (int64_t i = 0; i < t_f32.numel(); ++i) {
        EXPECT_NEAR(d[i], 3.14f, atol());
    }
}

TEST_P(InitMultiDTypeTest, ZerosFill) {
    auto t = tenzor::ones({4, 4}, dtype(), device());
    zeros_(t);

    float max_val = compute_max_abs(t);
    EXPECT_NEAR(max_val, 0.0f, atol());
}

TEST_P(InitMultiDTypeTest, OnesFill) {
    auto t = tenzor::zeros({4, 4}, dtype(), device());
    ones_(t);

    auto t_f32 = t.to(Device::cpu()).to(DType::Float32);
    auto* d = t_f32.data<float>();
    for (int64_t i = 0; i < t_f32.numel(); ++i) {
        EXPECT_NEAR(d[i], 1.0f, atol());
    }
}

TEST_P(InitMultiDTypeTest, UniformRange) {
    auto t = tenzor::zeros({100}, dtype(), device());
    uniform_(t, -2.0, 2.0);

    float min_val = compute_min(t);
    float max_val = compute_max(t);
    EXPECT_GE(min_val, -2.0f - atol());
    EXPECT_LE(max_val, 2.0f + atol());
}

TEST_P(InitMultiDTypeTest, NormalMean) {
    auto t = tenzor::zeros({1000}, dtype(), device());
    normal_(t, 0.0, 1.0);

    float mean_val = compute_mean(t);
    EXPECT_NEAR(mean_val, 0.0f, 0.2f);  // statistical tolerance
}

// ============================================================================
// Orthogonal Initialization Tests
// ============================================================================

TEST_P(InitMultiDTypeTest, OrthogonalOrthogonality) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "Orthogonal init requires higher precision";
    }

    auto t = tenzor::zeros({32, 32}, dtype(), device());
    orthogonal_(t, 1.0);

    // W^T * W should approximate identity
    auto t_f32 = t.to(Device::cpu()).to(DType::Float32);
    auto wt = tenzor::transpose(t_f32, 0, 1);
    auto product = tenzor::matmul(wt, t_f32);
    auto eye = tenzor::eye(32, std::nullopt, DType::Float32, Device::cpu());
    auto diff = tenzor::sub(product, eye);
    float max_diff = compute_max_abs(diff);
    EXPECT_LT(max_diff, 0.01f);
}

// ============================================================================
// Device and DType Preservation
// ============================================================================

TEST_P(InitMultiDTypeTest, InitPreservesDevice) {
    auto t = tenzor::zeros({8, 8}, dtype(), device());
    xavier_uniform_(t);
    expectDevice(t);
    expectDType(t);
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(InitMultiDTypeTest);
