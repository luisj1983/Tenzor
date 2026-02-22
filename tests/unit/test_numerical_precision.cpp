/**
 * @file test_numerical_precision.cpp
 * @brief Tests for numerical precision and stability
 *
 * Covers:
 * - Softmax overflow/underflow at extreme values
 * - Log of very small values
 * - Mixed-precision accuracy (float16 vs float32)
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;

class NumericalPrecisionTest : public ::testing::Test {
protected:
    static bool initialized;
    void SetUp() override {
        if (!initialized) {
            tenzor::initialize();
            initialized = true;
        }
    }
};

bool NumericalPrecisionTest::initialized = false;

// ============================================================================
// 1. Softmax Overflow Protection (Large Positive Values)
// ============================================================================

TEST_F(NumericalPrecisionTest, SoftmaxLargePositiveValues) {
    // softmax with very large values should not produce NaN or Inf
    // due to the max-subtraction trick: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    auto data = zeros({1, 5}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1000.0f;
    ptr[1] = 1001.0f;
    ptr[2] = 999.0f;
    ptr[3] = 1000.0f;
    ptr[4] = 998.0f;

    Variable x(data, false);
    auto result = tenzor::softmax(x, 1);
    auto result_data = result.tensor().to(Device::cpu());
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
    EXPECT_NEAR(total, 1.0f, 1e-5f);

    // The maximum element (index 1, value 1001) should have the largest probability
    EXPECT_GT(r_ptr[1], r_ptr[0]);
    EXPECT_GT(r_ptr[1], r_ptr[2]);
}

TEST_F(NumericalPrecisionTest, SoftmaxVeryLargeValues) {
    // Even more extreme values
    auto data = zeros({1, 3}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1e30f;
    ptr[1] = 1e30f + 1.0f;  // slightly larger
    ptr[2] = 1e30f - 1.0f;  // slightly smaller

    Variable x(data, false);
    auto result = tenzor::softmax(x, 1);
    auto result_data = result.tensor().to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    float total = 0.0f;
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "Softmax NaN at extreme values, index " << i;
        EXPECT_FALSE(std::isinf(r_ptr[i])) << "Softmax Inf at extreme values, index " << i;
        total += r_ptr[i];
    }
    EXPECT_NEAR(total, 1.0f, 1e-5f);
}

// ============================================================================
// 2. Softmax Underflow Protection (Large Negative Values)
// ============================================================================

TEST_F(NumericalPrecisionTest, SoftmaxLargeNegativeValues) {
    auto data = zeros({1, 4}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = -1000.0f;
    ptr[1] = -999.0f;   // largest (least negative)
    ptr[2] = -1001.0f;
    ptr[3] = -1002.0f;

    Variable x(data, false);
    auto result = tenzor::softmax(x, 1);
    auto result_data = result.tensor().to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    float total = 0.0f;
    for (int i = 0; i < 4; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "Softmax NaN at large negative values";
        EXPECT_FALSE(std::isinf(r_ptr[i])) << "Softmax Inf at large negative values";
        EXPECT_GE(r_ptr[i], 0.0f);
        total += r_ptr[i];
    }
    EXPECT_NEAR(total, 1.0f, 1e-5f);

    // Index 1 (-999) should have the largest probability
    EXPECT_GT(r_ptr[1], r_ptr[0]);
    EXPECT_GT(r_ptr[1], r_ptr[2]);
    EXPECT_GT(r_ptr[1], r_ptr[3]);
}

// ============================================================================
// 3. Softmax Uniform Input
// ============================================================================

TEST_F(NumericalPrecisionTest, SoftmaxUniformInput) {
    // All equal values -> uniform distribution (1/N each)
    int n = 5;
    auto data = full({1, n}, 42.0f, DType::Float32, Device::cpu());

    Variable x(data, false);
    auto result = tenzor::softmax(x, 1);
    auto result_data = result.tensor().to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    float expected = 1.0f / static_cast<float>(n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(r_ptr[i], expected, 1e-5f);
    }
}

// ============================================================================
// 4. Log of Very Small Values
// ============================================================================

TEST_F(NumericalPrecisionTest, LogSmallPositiveValues) {
    auto data = zeros({4}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1e-30f;
    ptr[1] = 1e-20f;
    ptr[2] = 1e-10f;
    ptr[3] = 1e-5f;

    Variable x(data, false);
    auto result = tenzor::log(x);
    auto result_data = result.tensor().to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    // log of small positive values should be large negative numbers, not NaN
    for (int i = 0; i < 4; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "Log of small positive produced NaN at index " << i;
        EXPECT_LT(r_ptr[i], 0.0f) << "Log of small positive should be negative at index " << i;
        EXPECT_FALSE(std::isinf(r_ptr[i])) << "Log of small positive produced -Inf at index " << i;
    }

    // Check approximate values
    EXPECT_NEAR(r_ptr[0], std::log(1e-30f), 1.0f);  // ~-69.08
    EXPECT_NEAR(r_ptr[1], std::log(1e-20f), 1.0f);  // ~-46.05
    EXPECT_NEAR(r_ptr[2], std::log(1e-10f), 0.1f);   // ~-23.03
    EXPECT_NEAR(r_ptr[3], std::log(1e-5f), 0.01f);    // ~-11.51
}

TEST_F(NumericalPrecisionTest, LogOfOne) {
    auto data = ones({3}, DType::Float32, Device::cpu());
    Variable x(data, false);
    auto result = tenzor::log(x);
    auto result_data = result.tensor().to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(r_ptr[i], 0.0f, 1e-6f);
    }
}

TEST_F(NumericalPrecisionTest, LogOfLargeValues) {
    auto data = zeros({3}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1e10f;
    ptr[1] = 1e20f;
    ptr[2] = 1e30f;

    Variable x(data, false);
    auto result = tenzor::log(x);
    auto result_data = result.tensor().to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "Log of large value produced NaN at index " << i;
        EXPECT_FALSE(std::isinf(r_ptr[i])) << "Log of large value produced Inf at index " << i;
        EXPECT_GT(r_ptr[i], 0.0f);
    }

    EXPECT_NEAR(r_ptr[0], std::log(1e10f), 0.1f);   // ~23.03
    EXPECT_NEAR(r_ptr[1], std::log(1e20f), 0.1f);   // ~46.05
    EXPECT_NEAR(r_ptr[2], std::log(1e30f), 1.0f);   // ~69.08
}

// ============================================================================
// 5. Log-Softmax Numerical Stability
// ============================================================================

TEST_F(NumericalPrecisionTest, LogSoftmaxLargeValues) {
    // log_softmax should be more numerically stable than log(softmax(x))
    auto data = zeros({1, 4}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1000.0f;
    ptr[1] = 0.0f;
    ptr[2] = -1000.0f;
    ptr[3] = 500.0f;

    Variable x(data, false);
    auto result = tenzor::log_softmax(x, 1);
    auto result_data = result.tensor().to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    for (int i = 0; i < 4; ++i) {
        EXPECT_FALSE(std::isnan(r_ptr[i])) << "log_softmax produced NaN at index " << i;
        // log_softmax values should be <= 0
        EXPECT_LE(r_ptr[i], 0.0f + 1e-5f);
    }

    // Index 0 (value 1000) should have log-probability close to 0
    EXPECT_NEAR(r_ptr[0], 0.0f, 1.0f);
}

// ============================================================================
// 6. Mixed-Precision: Float16 vs Float32 Accuracy
// ============================================================================

TEST_F(NumericalPrecisionTest, Float16BasicAccuracy) {
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

    auto data_f16 = data_f32.to(DType::Float16);
    auto data_roundtrip = data_f16.to(DType::Float32);
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

TEST_F(NumericalPrecisionTest, Float16AdditionAccuracy) {
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

    // Float16 computation
    auto a_f16 = a_f32.to(DType::Float16);
    auto b_f16 = b_f32.to(DType::Float16);
    auto result_f16 = add(a_f16, b_f16);
    auto result_f16_as_f32 = result_f16.to(DType::Float32);
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

TEST_F(NumericalPrecisionTest, ExpLargePositive) {
    auto data = zeros({2}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 80.0f;   // exp(80) ~ 5.5e34
    ptr[1] = 0.0f;

    Variable x(data, false);
    auto result = tenzor::exp(x);
    auto result_data = result.tensor().to(Device::cpu());
    auto* r_ptr = result_data.data<float>();

    EXPECT_FALSE(std::isnan(r_ptr[0]));
    EXPECT_GT(r_ptr[0], 0.0f);
    EXPECT_NEAR(r_ptr[1], 1.0f, 1e-5f);  // exp(0) = 1
}

TEST_F(NumericalPrecisionTest, ExpLargeNegative) {
    auto data = zeros({2}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = -80.0f;   // exp(-80) ~ 1.8e-35
    ptr[1] = -200.0f;  // very small, might underflow to 0

    Variable x(data, false);
    auto result = tenzor::exp(x);
    auto result_data = result.tensor().to(Device::cpu());
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

TEST_F(NumericalPrecisionTest, Float64HigherPrecisionThanFloat32) {
    // Demonstrate that Float64 preserves more precision
    // 1.0 + 1e-10 in Float32 may lose the small part, but Float64 retains it
    auto data_f64 = zeros({1}, DType::Float64, Device::cpu());
    auto* ptr64 = data_f64.data<double>();
    ptr64[0] = 1.0 + 1e-10;

    auto data_f32 = data_f64.to(DType::Float32);
    auto data_back_f64 = data_f32.to(DType::Float64);
    auto* back_ptr = data_back_f64.data<double>();

    // Float32 loses the 1e-10 precision
    double f32_error = std::abs(back_ptr[0] - (1.0 + 1e-10));
    double f64_error = std::abs(ptr64[0] - (1.0 + 1e-10));

    EXPECT_LT(f64_error, f32_error) << "Float64 should preserve more precision than Float32";
}
