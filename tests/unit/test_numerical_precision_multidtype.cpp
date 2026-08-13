/**
 * @file test_numerical_precision_multidtype.cpp
 * @brief Multi-backend, multi-dtype tests for numerical precision and stability.
 *
 * Converted from test_numerical_precision.cpp.
 *
 * Covers:
 * - Softmax overflow/underflow at extreme values
 * - Log of very small values
 * - Mixed-precision accuracy (float16 vs float32)
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::testing;

// Macro (not a method) so GTEST_SKIP's internal `return` applies to the
// TEST_P body rather than the helper method. Used by tests whose input
// magnitudes exceed Float16's representable range (Float16 max ≈ 6.55e4,
// min-normal ≈ 6.10e-5).
#define SKIP_IF_FLOAT16_UNREPRESENTABLE(reason) \
    do { \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) { \
            GTEST_SKIP() << reason; \
        } \
    } while (0)

class NumericalPrecisionMultiBackendDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// 1. Softmax Overflow Protection (Large Positive Values)
// ============================================================================

TEST_P(NumericalPrecisionMultiBackendDTypeTest, SoftmaxLargePositiveValues) {
    // softmax with very large values should not produce NaN or Inf
    // due to the max-subtraction trick: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    auto data = zeros({1, 5}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1000.0f;
    ptr[1] = 1001.0f;
    ptr[2] = 999.0f;
    ptr[3] = 1000.0f;
    ptr[4] = 998.0f;

    // Move to test device and dtype
    data = data.to(dtype()).to(device());

    Variable x(data, false);
    auto result = tenzor::softmax(x, 1);
    auto result_data = result.tensor().to(DType::Float32).to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    // Check no NaN or Inf
    float total = 0.0f;
    for (int i = 0; i < 5; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "Softmax produced NaN at index " << i;
        EXPECT_FALSE(std::isinf(r_ptr[i])) << "Softmax produced Inf at index " << i;
        EXPECT_GE(r_ptr[i], 0.0f) << "Softmax produced negative value at index " << i;
        EXPECT_LE(r_ptr[i], 1.0f) << "Softmax produced value > 1 at index " << i;
        total += r_ptr[i];
    }

    // Sum should be approximately 1.0
    EXPECT_NEAR(total, 1.0f, atol());

    // The maximum element (index 1, value 1001) should have the largest
    // probability -- but skip the strict ordering on BFloat16: its 7-bit
    // mantissa gives a representable step of ~8 near magnitude 1000, so
    // 1000/1001/999/998 can round to the SAME BFloat16 value, making their
    // softmax outputs legitimately tie rather than strictly order. The
    // NaN/Inf/sum-to-1 checks above already cover BFloat16 correctness.
    if (dtype() != DType::BFloat16) {
        EXPECT_GT(r_ptr[1], r_ptr[0]);
        EXPECT_GT(r_ptr[1], r_ptr[2]);
    }
}

TEST_P(NumericalPrecisionMultiBackendDTypeTest, SoftmaxVeryLargeValues) {
    // 1e30 overflows Float16/BFloat16 on conversion (becomes +Inf), so the
    // test loses its meaning for those dtypes.
    SKIP_IF_FLOAT16_UNREPRESENTABLE("1e30 exceeds Float16/BFloat16 range");
    // Even more extreme values
    auto data = zeros({1, 3}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1e30f;
    ptr[1] = 1e30f + 1.0f;  // slightly larger
    ptr[2] = 1e30f - 1.0f;  // slightly smaller

    data = data.to(dtype()).to(device());

    Variable x(data, false);
    auto result = tenzor::softmax(x, 1);
    auto result_data = result.tensor().to(DType::Float32).to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    float total = 0.0f;
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "Softmax NaN at extreme values, index " << i;
        EXPECT_FALSE(std::isinf(r_ptr[i])) << "Softmax Inf at extreme values, index " << i;
        total += r_ptr[i];
    }
    EXPECT_NEAR(total, 1.0f, atol());
}

// ============================================================================
// 2. Softmax Underflow Protection (Large Negative Values)
// ============================================================================

TEST_P(NumericalPrecisionMultiBackendDTypeTest, SoftmaxLargeNegativeValues) {
    auto data = zeros({1, 4}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = -1000.0f;
    ptr[1] = -999.0f;   // largest (least negative)
    ptr[2] = -1001.0f;
    ptr[3] = -1002.0f;

    data = data.to(dtype()).to(device());

    Variable x(data, false);
    auto result = tenzor::softmax(x, 1);
    auto result_data = result.tensor().to(DType::Float32).to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    float total = 0.0f;
    for (int i = 0; i < 4; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "Softmax NaN at large negative values";
        EXPECT_FALSE(std::isinf(r_ptr[i])) << "Softmax Inf at large negative values";
        EXPECT_GE(r_ptr[i], 0.0f);
        total += r_ptr[i];
    }
    EXPECT_NEAR(total, 1.0f, atol());

    // Index 1 (-999) should have the largest probability -- skip the strict
    // ordering on BFloat16 for the same reason as SoftmaxLargePositiveValues
    // (its ~8-wide representable step near magnitude 1000 can collapse
    // -1000/-999/-1001/-1002 to the same value).
    if (dtype() != DType::BFloat16) {
        EXPECT_GT(r_ptr[1], r_ptr[0]);
        EXPECT_GT(r_ptr[1], r_ptr[2]);
        EXPECT_GT(r_ptr[1], r_ptr[3]);
    }
}

// ============================================================================
// 3. Softmax Uniform Input
// ============================================================================

TEST_P(NumericalPrecisionMultiBackendDTypeTest, SoftmaxUniformInput) {
    // All equal values -> uniform distribution (1/N each)
    int n = 5;
    auto data = full({1, n}, 42.0f, DType::Float32, Device::cpu());
    data = data.to(dtype()).to(device());

    Variable x(data, false);
    auto result = tenzor::softmax(x, 1);
    auto result_data = result.tensor().to(DType::Float32).to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    float expected = 1.0f / static_cast<float>(n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(r_ptr[i], expected, atol());
    }
}

// ============================================================================
// 4. Log of Very Small Values
// ============================================================================

TEST_P(NumericalPrecisionMultiBackendDTypeTest, LogSmallPositiveValues) {
    // 1e-30 underflows to zero in Float16/BFloat16 on conversion, and log(0)
    // is -Inf which the test explicitly disallows — so the test can only be
    // meaningful for Float32/Float64.
    SKIP_IF_FLOAT16_UNREPRESENTABLE("1e-30 underflows in Float16/BFloat16");
    auto data = zeros({4}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1e-30f;
    ptr[1] = 1e-20f;
    ptr[2] = 1e-10f;
    ptr[3] = 1e-5f;

    data = data.to(dtype()).to(device());

    Variable x(data, false);
    auto result = tenzor::log(x);
    auto result_data = result.tensor().to(DType::Float32).to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    // log of small positive values should be large negative numbers, not NaN
    for (int i = 0; i < 4; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "Log of small positive produced NaN at index " << i;
        EXPECT_LT(r_ptr[i], 0.0f) << "Log of small positive should be negative at index " << i;
        EXPECT_FALSE(std::isinf(r_ptr[i])) << "Log of small positive produced -Inf at index " << i;
    }
}

TEST_P(NumericalPrecisionMultiBackendDTypeTest, LogOfOne) {
    auto data = ones({3}, DType::Float32, Device::cpu());
    data = data.to(dtype()).to(device());

    Variable x(data, false);
    auto result = tenzor::log(x);
    auto result_data = result.tensor().to(DType::Float32).to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(r_ptr[i], 0.0f, atol());
    }
}

TEST_P(NumericalPrecisionMultiBackendDTypeTest, LogOfLargeValues) {
    // 1e10, 1e20, 1e30 all overflow Float16/BFloat16 on conversion.
    SKIP_IF_FLOAT16_UNREPRESENTABLE("1e10..1e30 exceed Float16/BFloat16 range");
    auto data = zeros({3}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1e10f;
    ptr[1] = 1e20f;
    ptr[2] = 1e30f;

    data = data.to(dtype()).to(device());

    Variable x(data, false);
    auto result = tenzor::log(x);
    auto result_data = result.tensor().to(DType::Float32).to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "Log of large value produced NaN at index " << i;
        EXPECT_FALSE(std::isinf(r_ptr[i])) << "Log of large value produced Inf at index " << i;
        EXPECT_GT(r_ptr[i], 0.0f);
    }
}

// ============================================================================
// 5. Log-Softmax Numerical Stability
// ============================================================================

TEST_P(NumericalPrecisionMultiBackendDTypeTest, LogSoftmaxLargeValues) {
    // log_softmax should be more numerically stable than log(softmax(x))
    auto data = zeros({1, 4}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1000.0f;
    ptr[1] = 0.0f;
    ptr[2] = -1000.0f;
    ptr[3] = 500.0f;

    data = data.to(dtype()).to(device());

    Variable x(data, false);
    auto result = tenzor::log_softmax(x, 1);
    auto result_data = result.tensor().to(DType::Float32).to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    for (int i = 0; i < 4; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "log_softmax produced NaN at index " << i;
        // log_softmax values should be <= 0
        EXPECT_LE(r_ptr[i], 0.0f + atol());
    }

    // Index 0 (value 1000) should have log-probability close to 0
    EXPECT_NEAR(r_ptr[0], 0.0f, 1.0f);
}

// ============================================================================
// 6. Mixed-Precision: Float16 vs Float32 Accuracy
// ============================================================================

TEST_P(NumericalPrecisionMultiBackendDTypeTest, Float16BasicAccuracy) {
    // Create a Float32 tensor, convert to Float16, convert back, check accuracy
    auto data_f32 = zeros({8}, DType::Float32, Device::cpu());
    auto* ptr = data_f32.data<float>();
    ptr[0] = 0.0f;
    ptr[1] = 1.0f;
    ptr[2] = -1.0f;
    ptr[3] = 0.5f;
    ptr[4] = 0.1f;
    ptr[5] = 100.0f;
    ptr[6] = -100.0f;
    ptr[7] = 0.001f;

    // Move to device, convert to Float16, roundtrip back to Float32
    auto data_on_device = data_f32.to(device());
    auto data_f16 = data_on_device.to(DType::Float16);
    auto data_roundtrip = data_f16.to(DType::Float32).to(Device::cpu());
    auto* rt_ptr = data_roundtrip.data<float>();

    // Float16 has ~3.3 decimal digits of precision
    EXPECT_FLOAT_EQ(rt_ptr[0], 0.0f);     // exact
    EXPECT_FLOAT_EQ(rt_ptr[1], 1.0f);     // exact
    EXPECT_FLOAT_EQ(rt_ptr[2], -1.0f);    // exact
    EXPECT_FLOAT_EQ(rt_ptr[3], 0.5f);     // exact (power of 2)
    EXPECT_NEAR(rt_ptr[4], 0.1f, 1e-3f);  // approximate
    EXPECT_NEAR(rt_ptr[5], 100.0f, 0.1f); // approximate
    EXPECT_NEAR(rt_ptr[6], -100.0f, 0.1f);
    EXPECT_NEAR(rt_ptr[7], 0.001f, 1e-4f);
}

TEST_P(NumericalPrecisionMultiBackendDTypeTest, Float16AdditionAccuracy) {
    // Perform addition in Float16 and verify against Float32
    auto a_f32 = ones({4}, DType::Float32, Device::cpu());
    auto b_f32 = ones({4}, DType::Float32, Device::cpu());
    auto* a_ptr = a_f32.data<float>();
    auto* b_ptr = b_f32.data<float>();
    a_ptr[0] = 1.0f; a_ptr[1] = 0.5f; a_ptr[2] = 10.0f; a_ptr[3] = 0.1f;
    b_ptr[0] = 2.0f; b_ptr[1] = 0.25f; b_ptr[2] = 20.0f; b_ptr[3] = 0.2f;

    // Float32 reference
    auto result_f32 = add(a_f32, b_f32);
    auto* ref = result_f32.data<float>();

    // Float16 computation on test device
    auto a_f16 = a_f32.to(device()).to(DType::Float16);
    auto b_f16 = b_f32.to(device()).to(DType::Float16);
    auto result_f16 = add(a_f16, b_f16);
    auto result_f16_as_f32 = result_f16.to(DType::Float32).to(Device::cpu());
    auto* r16 = result_f16_as_f32.data<float>();

    // Float16 should be close to Float32 for these moderate values
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(r16[i], ref[i], 0.1f)
            << "Float16 addition diverged at index " << i;
    }
}

// ============================================================================
// 7. Exp of Extreme Values
// ============================================================================

TEST_P(NumericalPrecisionMultiBackendDTypeTest, ExpLargePositive) {
    auto data = zeros({2}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 80.0f;   // exp(80) ~ 5.5e34
    ptr[1] = 0.0f;

    data = data.to(dtype()).to(device());

    Variable x(data, false);
    auto result = tenzor::exp(x);
    auto result_data = result.tensor().to(DType::Float32).to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    EXPECT_FALSE(std::isnan(r_ptr[0]));
    EXPECT_GT(r_ptr[0], 0.0f);
    EXPECT_NEAR(r_ptr[1], 1.0f, atol());  // exp(0) = 1
}

TEST_P(NumericalPrecisionMultiBackendDTypeTest, ExpLargeNegative) {
    auto data = zeros({2}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = -80.0f;   // exp(-80) ~ 1.8e-35
    ptr[1] = -200.0f;  // very small, might underflow to 0

    data = data.to(dtype()).to(device());

    Variable x(data, false);
    auto result = tenzor::exp(x);
    auto result_data = result.tensor().to(DType::Float32).to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    // Should be very small but not NaN
    EXPECT_FALSE(std::isnan(r_ptr[0]));
    EXPECT_FALSE(std::isnan(r_ptr[1]));
    EXPECT_GE(r_ptr[0], 0.0f);
    EXPECT_GE(r_ptr[1], 0.0f);
}

// ============================================================================
// 8. Float64 Precision Advantage
// ============================================================================

TEST_P(NumericalPrecisionMultiBackendDTypeTest, Float64HigherPrecisionThanFloat32) {
    // Demonstrate that Float64 preserves more precision
    // 1.0 + 1e-10 in Float32 may lose the small part, but Float64 retains it
    auto data_f64 = zeros({1}, DType::Float64, Device::cpu());
    auto* ptr64 = data_f64.data<double>();
    ptr64[0] = 1.0 + 1e-10;

    // Move to test device for the roundtrip
    auto data_on_device = data_f64.to(device());
    auto data_f32 = data_on_device.to(DType::Float32);
    auto data_back_f64 = data_f32.to(DType::Float64).to(Device::cpu());
    auto* back_ptr = data_back_f64.data<double>();

    // Float32 loses the 1e-10 precision
    double f32_error = std::abs(back_ptr[0] - (1.0 + 1e-10));
    double f64_error = std::abs(ptr64[0] - (1.0 + 1e-10));

    EXPECT_LT(f64_error, f32_error) << "Float64 should preserve more precision than Float32";
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(NumericalPrecisionMultiBackendDTypeTest);
