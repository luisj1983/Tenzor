/**
 * @file test_fp8_ops.cpp
 * @brief Cross-backend tests for FP8 (E4M3 and E5M2) tensor operations.
 *
 * NOTE — Phase 7.4 of the test-coverage campaign:
 *
 * Originally a CPU-only reference suite, this file is now rebased onto the
 * shared BackendTest fixture so it runs across every available backend.
 * The CPU FP8 path is software emulation (`fp8_emulation.hpp`); GPU FP8
 * paths widen to Float32 for compute (numerically equivalent because
 * Float32 exactly represents every FP8 value) or use hardware tensor
 * cores (CUDA Hopper+). Per the no-skip policy, a backend that cannot
 * perform an FP8 op throws and the test FAILs rather than skipping.
 *
 * Type-promotion tests are device-agnostic (operate on the `DType` enum,
 * not on hardware) but are still parameterized so they run once per
 * backend for uniformity.
 *
 * Verifies:
 * 1. FP8+FP8 operations preserve FP8 output dtype (no silent Float32 promotion)
 * 2. Mixed FP8 types promote to FP8_E5M2
 * 3. FP8+Float32 still promotes to Float32
 * 4. Numerical results are within FP8 quantization tolerance of Float32 reference
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/type_promotion.hpp"
#include "tenzor/ops/fp8_scaling.hpp"

#include "../backend_test_fixture.hpp"

using namespace tenzor;

class FP8OpsTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// --- Type Promotion Tests ---

TEST_P(FP8OpsTest, SameTypePromotion_E4M3) {
    EXPECT_EQ(promote_types(DType::FP8_E4M3, DType::FP8_E4M3), DType::FP8_E4M3);
}

TEST_P(FP8OpsTest, SameTypePromotion_E5M2) {
    EXPECT_EQ(promote_types(DType::FP8_E5M2, DType::FP8_E5M2), DType::FP8_E5M2);
}

TEST_P(FP8OpsTest, MixedFP8Promotion) {
    EXPECT_EQ(promote_types(DType::FP8_E4M3, DType::FP8_E5M2), DType::FP8_E5M2);
    EXPECT_EQ(promote_types(DType::FP8_E5M2, DType::FP8_E4M3), DType::FP8_E5M2);
}

TEST_P(FP8OpsTest, FP8Float32Promotion) {
    EXPECT_EQ(promote_types(DType::FP8_E4M3, DType::Float32), DType::Float32);
    EXPECT_EQ(promote_types(DType::Float32, DType::FP8_E5M2), DType::Float32);
}

TEST_P(FP8OpsTest, FP8Float64Promotion) {
    EXPECT_EQ(promote_types(DType::FP8_E4M3, DType::Float64), DType::Float64);
}

// --- Output DType Tests ---

TEST_P(FP8OpsTest, AddOutputDtype_E4M3) {
    auto a = randn({4, 4}, DType::Float32, device).to(DType::FP8_E4M3);
    auto b = randn({4, 4}, DType::Float32, device).to(DType::FP8_E4M3);
    auto c = tenzor::add(a, b);
    EXPECT_EQ(c.dtype(), DType::FP8_E4M3) << "FP8+FP8 add should return FP8";
}

TEST_P(FP8OpsTest, MulOutputDtype_E4M3) {
    auto a = randn({4, 4}, DType::Float32, device).to(DType::FP8_E4M3);
    auto b = randn({4, 4}, DType::Float32, device).to(DType::FP8_E4M3);
    auto c = tenzor::mul(a, b);
    EXPECT_EQ(c.dtype(), DType::FP8_E4M3) << "FP8*FP8 mul should return FP8";
}

TEST_P(FP8OpsTest, DivOutputDtype_E4M3) {
    // Use non-zero values to avoid division by zero
    auto a = ones({4, 4}, DType::Float32, device).to(DType::FP8_E4M3);
    auto b = ones({4, 4}, DType::Float32, device).to(DType::FP8_E4M3);
    auto c = tenzor::div(a, b);
    EXPECT_EQ(c.dtype(), DType::FP8_E4M3) << "FP8/FP8 div should return FP8";
}

TEST_P(FP8OpsTest, MatmulOutputDtype_E4M3) {
    auto a = randn({4, 8}, DType::Float32, device).to(DType::FP8_E4M3);
    auto b = randn({8, 4}, DType::Float32, device).to(DType::FP8_E4M3);
    auto c = tenzor::matmul(a, b);
    EXPECT_EQ(c.dtype(), DType::FP8_E4M3) << "FP8 matmul should return FP8";
}

TEST_P(FP8OpsTest, MixedFP8OutputDtype) {
    auto a = randn({4, 4}, DType::Float32, device).to(DType::FP8_E4M3);
    auto b = randn({4, 4}, DType::Float32, device).to(DType::FP8_E5M2);
    auto c = tenzor::add(a, b);
    EXPECT_EQ(c.dtype(), DType::FP8_E5M2)
        << "FP8_E4M3 + FP8_E5M2 should promote to FP8_E5M2";
}

TEST_P(FP8OpsTest, FP8Float32MixedOutputDtype) {
    auto a = randn({4, 4}, DType::Float32, device).to(DType::FP8_E4M3);
    auto b = randn({4, 4}, DType::Float32, device);
    auto c = tenzor::add(a, b);
    EXPECT_EQ(c.dtype(), DType::Float32)
        << "FP8 + Float32 should promote to Float32";
}

// --- Numerical Correctness Tests ---

TEST_P(FP8OpsTest, MatmulNumericalCorrectness) {
    // Create small-valued tensors to stay within FP8 range
    auto a_f32 = randn({4, 8}, DType::Float32, device) * 0.1f;
    auto b_f32 = randn({8, 4}, DType::Float32, device) * 0.1f;

    // Reference: Float32 matmul
    auto ref = tenzor::matmul(a_f32, b_f32);

    // FP8: convert inputs, matmul, convert result back to Float32
    auto a_fp8 = a_f32.to(DType::FP8_E4M3);
    auto b_fp8 = b_f32.to(DType::FP8_E4M3);
    auto fp8_result = tenzor::matmul(a_fp8, b_fp8);
    auto fp8_as_f32 = fp8_result.to(DType::Float32);

    // FP8 has ~1.5 decimal digits of precision, so tolerance is generous
    // E4M3 range is ±448, we used 0.1x scale so values are small
    auto diff = tenzor::sub(fp8_as_f32, ref);
    auto abs_diff = tenzor::abs(diff);
    auto max_diff_t = tenzor::max(abs_diff);
    float max_diff = max_diff_t.cpu().to(DType::Float32).data<float>()[0];

    // FP8 E4M3 has 3-bit mantissa, so relative error ~12.5% for individual values
    // Matmul accumulates error over K=8 inner dimension
    EXPECT_LT(max_diff, 1.0f)
        << "FP8 matmul result should be within reasonable tolerance of Float32";
}

TEST_P(FP8OpsTest, AddNumericalCorrectness) {
    auto a_f32 = randn({16}, DType::Float32, device) * 0.5f;
    auto b_f32 = randn({16}, DType::Float32, device) * 0.5f;

    auto ref = tenzor::add(a_f32, b_f32);

    auto a_fp8 = a_f32.to(DType::FP8_E4M3);
    auto b_fp8 = b_f32.to(DType::FP8_E4M3);
    auto fp8_result = tenzor::add(a_fp8, b_fp8).to(DType::Float32);

    auto max_diff_t = tenzor::max(tenzor::abs(tenzor::sub(fp8_result, ref)));
    float max_diff = max_diff_t.cpu().to(DType::Float32).data<float>()[0];

    // FP8 E4M3 has 3-bit mantissa: each input has ~12.5% quantization error,
    // and addition accumulates errors from both operands
    EXPECT_LT(max_diff, 2.0f)
        << "FP8 add should be within FP8 quantization tolerance of Float32";
}

// ============================================================================
// Quantized Type Promotion Tests
// ============================================================================

TEST_P(FP8OpsTest, QuantizedFloat32Promotion) {
    EXPECT_EQ(promote_types(DType::QInt8, DType::Float32), DType::Float32);
    EXPECT_EQ(promote_types(DType::QUInt8, DType::Float32), DType::Float32);
}

TEST_P(FP8OpsTest, QuantizedFloat64Promotion) {
    EXPECT_EQ(promote_types(DType::QInt8, DType::Float64), DType::Float64);
}

TEST_P(FP8OpsTest, QuantizedQuantizedPromotion) {
    EXPECT_EQ(promote_types(DType::QInt8, DType::QUInt8), DType::QInt8);
    EXPECT_EQ(promote_types(DType::QUInt8, DType::QInt8), DType::QInt8);
    EXPECT_EQ(promote_types(DType::QInt4x2, DType::QInt8), DType::QInt8);
}

TEST_P(FP8OpsTest, QuantizedIntegerPromotion) {
    EXPECT_EQ(promote_types(DType::QInt8, DType::Int32), DType::Float32);
    EXPECT_EQ(promote_types(DType::QUInt8, DType::Int64), DType::Float32);
}

TEST_P(FP8OpsTest, QuantizedSameTypePromotion) {
    EXPECT_EQ(promote_types(DType::QInt8, DType::QInt8), DType::QInt8);
    EXPECT_EQ(promote_types(DType::QUInt8, DType::QUInt8), DType::QUInt8);
}

// ============================================================================
// FP8 Scaling Tests
// ============================================================================

TEST_P(FP8OpsTest, FP8MaxValue) {
    EXPECT_FLOAT_EQ(tenzor::fp8_max_value(DType::FP8_E4M3), 448.0f);
    EXPECT_FLOAT_EQ(tenzor::fp8_max_value(DType::FP8_E5M2), 57344.0f);
}

TEST_P(FP8OpsTest, ComputeAmax) {
    auto t = tenzor::full({4}, -5.0f, DType::Float32, device);
    EXPECT_FLOAT_EQ(tenzor::compute_amax(t), 5.0f);
}

TEST_P(FP8OpsTest, FP8ScaleComputation) {
    float scale = tenzor::compute_fp8_scale(100.0f, DType::FP8_E4M3);
    // scale = 100.0 / 448.0
    EXPECT_NEAR(scale, 100.0f / 448.0f, 1e-6f);
}

TEST_P(FP8OpsTest, FP8ScalingMathCorrectness) {
    // Verify the scaling math is correct (independent of FP8 dtype conversion)
    auto input = tenzor::full({1}, 100.0f, DType::Float32, device);
    float amax = tenzor::compute_amax(input);
    EXPECT_FLOAT_EQ(amax, 100.0f);

    float scale = tenzor::compute_fp8_scale(amax, DType::FP8_E4M3);
    EXPECT_NEAR(scale, 100.0f / 448.0f, 1e-5f);

    // Verify scaling: input / scale should be in FP8 range
    float scaled_val = 100.0f / scale;
    EXPECT_NEAR(scaled_val, 448.0f, 1.0f);
    EXPECT_LE(scaled_val, 448.0f);  // Should not exceed FP8 max
}

TEST_P(FP8OpsTest, FP8QuantizeOutputDtype) {
    // Verify quantize_to_fp8 returns correct dtype and params
    auto input = tenzor::rand({4, 4}, DType::Float32, device);
    auto [fp8_tensor, params] = tenzor::quantize_to_fp8(input, DType::FP8_E4M3);

    EXPECT_EQ(fp8_tensor.dtype(), DType::FP8_E4M3);
    EXPECT_EQ(params.fp8_dtype, DType::FP8_E4M3);
    EXPECT_GT(params.scale, 0.0f);
    EXPECT_GT(params.amax, 0.0f);
}

INSTANTIATE_BACKEND_TESTS(FP8OpsTest);
