/**
 * @file test_dtype_conversion_chains.cpp
 * @brief Tests for dtype conversion chains and round-trip accuracy
 *
 * Covers:
 * - F32 -> F16 -> F32 round-trips
 * - F32 -> BF16 -> F32 round-trips
 * - Edge values (NaN, Inf, max)
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;

class DTypeConversionChainTest : public ::testing::Test {
protected:
    static bool initialized;
    void SetUp() override {
        if (!initialized) {
            tenzor::initialize();
            initialized = true;
        }
    }
};

bool DTypeConversionChainTest::initialized = false;

// ============================================================================
// 1. F32 -> F16 -> F32 Round-Trips
// ============================================================================

TEST_F(DTypeConversionChainTest, F32_F16_F32_Exact_Representable) {
    // Values that are exactly representable in Float16
    auto data = zeros({6}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 0.0f;
    ptr[1] = 1.0f;
    ptr[2] = -1.0f;
    ptr[3] = 0.5f;
    ptr[4] = 2.0f;
    ptr[5] = -0.25f;

    auto f16 = data.to(DType::Float16);
    auto roundtrip = f16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    EXPECT_FLOAT_EQ(rt[0], 0.0f);
    EXPECT_FLOAT_EQ(rt[1], 1.0f);
    EXPECT_FLOAT_EQ(rt[2], -1.0f);
    EXPECT_FLOAT_EQ(rt[3], 0.5f);
    EXPECT_FLOAT_EQ(rt[4], 2.0f);
    EXPECT_FLOAT_EQ(rt[5], -0.25f);
}

TEST_F(DTypeConversionChainTest, F32_F16_F32_Approximate_Values) {
    // Values that lose some precision in Float16
    auto data = zeros({5}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 0.1f;
    ptr[1] = 0.3f;
    ptr[2] = 3.14159f;
    ptr[3] = 100.0f;
    ptr[4] = -42.5f;

    auto f16 = data.to(DType::Float16);
    auto roundtrip = f16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    // Float16 has ~3.3 decimal digits of precision (10-bit mantissa)
    EXPECT_NEAR(rt[0], 0.1f, 1e-3f);
    EXPECT_NEAR(rt[1], 0.3f, 1e-3f);
    EXPECT_NEAR(rt[2], 3.14159f, 2e-2f);
    EXPECT_NEAR(rt[3], 100.0f, 0.2f);
    EXPECT_NEAR(rt[4], -42.5f, 0.1f);
}

TEST_F(DTypeConversionChainTest, F32_F16_F32_SmallValues) {
    // Very small values that might lose precision or become zero
    auto data = zeros({4}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1e-4f;
    ptr[1] = 1e-5f;
    ptr[2] = 6.1e-5f;  // smallest normal float16 ~6.1e-5
    ptr[3] = 5.96e-8f; // smallest subnormal float16 ~5.96e-8

    auto f16 = data.to(DType::Float16);
    auto roundtrip = f16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    EXPECT_NEAR(rt[0], 1e-4f, 1e-5f);
    // Smaller values might lose precision but shouldn't be NaN
    for (int i = 0; i < 4; ++i) {
        EXPECT_FALSE(std::isnan(rt[i])) << "NaN after F16 round-trip at index " << i;
    }
}

TEST_F(DTypeConversionChainTest, F32_F16_F32_LargeValues) {
    // Float16 max is ~65504
    auto data = zeros({3}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 65504.0f;   // Float16 max
    ptr[1] = 65000.0f;
    ptr[2] = 60000.0f;

    auto f16 = data.to(DType::Float16);
    auto roundtrip = f16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    EXPECT_NEAR(rt[0], 65504.0f, 1.0f);
    EXPECT_NEAR(rt[1], 65000.0f, 100.0f);
    EXPECT_NEAR(rt[2], 60000.0f, 100.0f);
}

// ============================================================================
// 2. F32 -> BF16 -> F32 Round-Trips
// ============================================================================

TEST_F(DTypeConversionChainTest, F32_BF16_F32_Exact_Representable) {
    auto data = zeros({5}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 0.0f;
    ptr[1] = 1.0f;
    ptr[2] = -1.0f;
    ptr[3] = 2.0f;
    ptr[4] = -0.5f;

    auto bf16 = data.to(DType::BFloat16);
    auto roundtrip = bf16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    EXPECT_FLOAT_EQ(rt[0], 0.0f);
    EXPECT_FLOAT_EQ(rt[1], 1.0f);
    EXPECT_FLOAT_EQ(rt[2], -1.0f);
    EXPECT_FLOAT_EQ(rt[3], 2.0f);
    EXPECT_FLOAT_EQ(rt[4], -0.5f);
}

TEST_F(DTypeConversionChainTest, F32_BF16_F32_Approximate_Values) {
    // BFloat16 has 8-bit exponent (same range as float32) but only 7-bit mantissa
    // (~2.3 decimal digits of precision)
    auto data = zeros({4}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 0.1f;
    ptr[1] = 3.14159f;
    ptr[2] = 100.0f;
    ptr[3] = -42.5f;

    auto bf16 = data.to(DType::BFloat16);
    auto roundtrip = bf16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    // BFloat16 is less precise than Float16 in mantissa but has larger range
    EXPECT_NEAR(rt[0], 0.1f, 1e-2f);
    EXPECT_NEAR(rt[1], 3.14159f, 5e-2f);
    EXPECT_NEAR(rt[2], 100.0f, 1.0f);
    EXPECT_NEAR(rt[3], -42.5f, 1.0f);
}

TEST_F(DTypeConversionChainTest, F32_BF16_F32_LargeRange) {
    // BFloat16 has same range as Float32 (8-bit exponent), so large values are fine
    auto data = zeros({3}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1e10f;
    ptr[1] = 1e20f;
    ptr[2] = -1e15f;

    auto bf16 = data.to(DType::BFloat16);
    auto roundtrip = bf16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    // Should maintain order of magnitude
    EXPECT_NEAR(rt[0], 1e10f, 1e8f);   // ~1% relative error
    EXPECT_NEAR(rt[1], 1e20f, 1e18f);
    EXPECT_NEAR(rt[2], -1e15f, 1e13f);
}

// ============================================================================
// 3. Edge Values: NaN
// ============================================================================

TEST_F(DTypeConversionChainTest, NaN_F32_F16_F32) {
    auto data = zeros({2}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = std::numeric_limits<float>::quiet_NaN();
    ptr[1] = 1.0f;  // normal value alongside NaN

    auto f16 = data.to(DType::Float16);
    auto roundtrip = f16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    EXPECT_TRUE(std::isnan(rt[0])) << "NaN should survive F16 round-trip";
    EXPECT_FLOAT_EQ(rt[1], 1.0f) << "Normal value should survive alongside NaN";
}

TEST_F(DTypeConversionChainTest, NaN_F32_BF16_F32) {
    auto data = zeros({2}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = std::numeric_limits<float>::quiet_NaN();
    ptr[1] = 2.0f;

    auto bf16 = data.to(DType::BFloat16);
    auto roundtrip = bf16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    EXPECT_TRUE(std::isnan(rt[0])) << "NaN should survive BF16 round-trip";
    EXPECT_FLOAT_EQ(rt[1], 2.0f);
}

// ============================================================================
// 4. Edge Values: Infinity
// ============================================================================

TEST_F(DTypeConversionChainTest, Inf_F32_F16_F32) {
    auto data = zeros({3}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = std::numeric_limits<float>::infinity();
    ptr[1] = -std::numeric_limits<float>::infinity();
    ptr[2] = 42.0f;

    auto f16 = data.to(DType::Float16);
    auto roundtrip = f16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    EXPECT_TRUE(std::isinf(rt[0]) && rt[0] > 0) << "+Inf should survive F16 round-trip";
    EXPECT_TRUE(std::isinf(rt[1]) && rt[1] < 0) << "-Inf should survive F16 round-trip";
    EXPECT_FLOAT_EQ(rt[2], 42.0f);
}

TEST_F(DTypeConversionChainTest, Inf_F32_BF16_F32) {
    auto data = zeros({3}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = std::numeric_limits<float>::infinity();
    ptr[1] = -std::numeric_limits<float>::infinity();
    ptr[2] = 42.0f;

    auto bf16 = data.to(DType::BFloat16);
    auto roundtrip = bf16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    EXPECT_TRUE(std::isinf(rt[0]) && rt[0] > 0) << "+Inf should survive BF16 round-trip";
    EXPECT_TRUE(std::isinf(rt[1]) && rt[1] < 0) << "-Inf should survive BF16 round-trip";
    EXPECT_FLOAT_EQ(rt[2], 42.0f);
}

// ============================================================================
// 5. Edge Values: Max/Min Representable
// ============================================================================

TEST_F(DTypeConversionChainTest, MaxFloat32_to_F16_Overflow) {
    // Float32 max (3.4e38) overflows Float16 max (65504)
    auto data = zeros({2}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = std::numeric_limits<float>::max();  // 3.4e38
    ptr[1] = 65504.0f;  // Float16 max

    auto f16 = data.to(DType::Float16);
    auto roundtrip = f16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    // Overflow might clamp to Inf or max Float16 value
    EXPECT_TRUE(std::isinf(rt[0]) || rt[0] >= 65504.0f)
        << "Float32 max should overflow to Inf or clamp in Float16";
    EXPECT_NEAR(rt[1], 65504.0f, 1.0f)
        << "Float16 max should round-trip correctly";
}

TEST_F(DTypeConversionChainTest, MaxFloat32_to_BF16_Preserves_Range) {
    // BFloat16 has same exponent range as Float32, so large values survive
    auto data = zeros({2}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1e38f;
    ptr[1] = -1e38f;

    auto bf16 = data.to(DType::BFloat16);
    auto roundtrip = bf16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    // Should preserve order of magnitude (but lose mantissa precision)
    EXPECT_FALSE(std::isnan(rt[0]));
    EXPECT_FALSE(std::isnan(rt[1]));
    EXPECT_GT(rt[0], 0.0f);
    EXPECT_LT(rt[1], 0.0f);
    // Relative error within BF16 precision (~0.8%)
    EXPECT_NEAR(rt[0] / 1e38f, 1.0f, 0.01f);
    EXPECT_NEAR(rt[1] / -1e38f, 1.0f, 0.01f);
}

// ============================================================================
// 6. Negative Zero
// ============================================================================

TEST_F(DTypeConversionChainTest, NegativeZero_F32_F16_F32) {
    auto data = zeros({2}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = -0.0f;
    ptr[1] = 0.0f;

    auto f16 = data.to(DType::Float16);
    auto roundtrip = f16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    // Both -0.0 and +0.0 should compare equal to 0.0
    EXPECT_FLOAT_EQ(rt[0], 0.0f);
    EXPECT_FLOAT_EQ(rt[1], 0.0f);
}

TEST_F(DTypeConversionChainTest, NegativeZero_F32_BF16_F32) {
    auto data = zeros({2}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = -0.0f;
    ptr[1] = 0.0f;

    auto bf16 = data.to(DType::BFloat16);
    auto roundtrip = bf16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    EXPECT_FLOAT_EQ(rt[0], 0.0f);
    EXPECT_FLOAT_EQ(rt[1], 0.0f);
}

// ============================================================================
// 7. Multi-Step Conversion Chains
// ============================================================================

TEST_F(DTypeConversionChainTest, F32_F16_BF16_F32) {
    // Float32 -> Float16 -> BFloat16 -> Float32
    auto data = zeros({4}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 0.5f;
    ptr[2] = 10.0f;
    ptr[3] = -3.0f;

    auto f16 = data.to(DType::Float16);
    auto bf16 = f16.to(DType::BFloat16);
    auto roundtrip = bf16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    // Each conversion loses some precision, but powers of 2 survive exactly
    EXPECT_FLOAT_EQ(rt[0], 1.0f);
    EXPECT_FLOAT_EQ(rt[1], 0.5f);
    EXPECT_NEAR(rt[2], 10.0f, 0.5f);
    EXPECT_NEAR(rt[3], -3.0f, 0.5f);
}

TEST_F(DTypeConversionChainTest, F32_F64_F32) {
    // Float32 -> Float64 -> Float32 should be lossless for Float32-representable values
    auto data = zeros({4}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    ptr[0] = 0.1f;
    ptr[1] = 3.14159f;
    ptr[2] = 1e20f;
    ptr[3] = -42.5f;

    auto f64 = data.to(DType::Float64);
    auto roundtrip = f64.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    // Should be exactly equal (Float64 can represent all Float32 values)
    EXPECT_FLOAT_EQ(rt[0], ptr[0]);
    EXPECT_FLOAT_EQ(rt[1], ptr[1]);
    EXPECT_FLOAT_EQ(rt[2], ptr[2]);
    EXPECT_FLOAT_EQ(rt[3], ptr[3]);
}

// ============================================================================
// 8. Batch Conversion Consistency
// ============================================================================

TEST_F(DTypeConversionChainTest, BatchConversionConsistency) {
    // Converting a batch of values should give the same results as converting individually
    int n = 16;
    auto data = zeros({n}, DType::Float32, Device::cpu());
    auto* ptr = data.data<float>();
    for (int i = 0; i < n; ++i) {
        ptr[i] = static_cast<float>(i) * 0.7f - 3.0f;
    }

    auto f16 = data.to(DType::Float16);
    auto roundtrip = f16.to(DType::Float32);
    auto* rt = roundtrip.data<float>();

    // Verify each element independently
    for (int i = 0; i < n; ++i) {
        auto single = zeros({1}, DType::Float32, Device::cpu());
        single.data<float>()[0] = ptr[i];
        auto single_rt = single.to(DType::Float16).to(DType::Float32);
        EXPECT_FLOAT_EQ(rt[i], single_rt.data<float>()[0])
            << "Batch and single conversion differ at index " << i;
    }
}
