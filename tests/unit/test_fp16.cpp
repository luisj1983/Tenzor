/**
 * @file test_fp16.cpp
 * @brief Unit tests for Float16 and BFloat16 data types
 *
 * Tests conversion accuracy, arithmetic operations, and mixed precision training support
 */

#include <gtest/gtest.h>
#include "tenzor/core/dtype.hpp"
#include "tenzor/tenzor.hpp"
#include <cmath>
#include <limits>

// Forward declare tenzor::initialize
namespace tenzor {
    void initialize();
}

using namespace tenzor;

// Global test environment to initialize Tenzor library
class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

// ============================================================================
// Float16 Tests
// ============================================================================

TEST(Float16Test, BasicConversion) {
    // Test basic float to Float16 conversion
    Float16 f16_zero(0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(f16_zero), 0.0f);

    Float16 f16_one(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(f16_one), 1.0f);

    Float16 f16_neg(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(f16_neg), -1.0f);
}

TEST(Float16Test, SmallValues) {
    // Test small positive and negative values
    Float16 f16_small(0.00006103515625f);  // Smallest normal Float16
    float recovered = static_cast<float>(f16_small);
    EXPECT_NEAR(recovered, 0.00006103515625f, 1e-5f);

    Float16 f16_tiny(0.0001f);
    EXPECT_NEAR(static_cast<float>(f16_tiny), 0.0001f, 1e-5f);
}

TEST(Float16Test, LargeValues) {
    // Test large values approaching Float16 max
    Float16 f16_large(65504.0f);  // Max Float16 value
    EXPECT_NEAR(static_cast<float>(f16_large), 65504.0f, 1.0f);

    // Test overflow
    Float16 f16_overflow(100000.0f);
    float result = static_cast<float>(f16_overflow);
    EXPECT_TRUE(std::isinf(result));
}

TEST(Float16Test, SpecialValues) {
    // Test infinity
    Float16 f16_inf(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isinf(static_cast<float>(f16_inf)));

    Float16 f16_neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isinf(static_cast<float>(f16_neg_inf)));

    // Test NaN
    Float16 f16_nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(std::isnan(static_cast<float>(f16_nan)));
}

TEST(Float16Test, Precision) {
    // Test that Float16 maintains approximately 3 decimal digits of precision
    std::vector<float> test_values = {
        1.234f, 12.34f, 123.4f, 1234.0f, 
        0.001234f, 0.01234f, 0.1234f
    };

    for (float val : test_values) {
        Float16 f16(val);
        float recovered = static_cast<float>(f16);
        float relative_error = std::abs((recovered - val) / val);
        EXPECT_LT(relative_error, 0.01f) << "Value: " << val;  // < 1% error
    }
}

TEST(Float16Test, Comparison) {
    Float16 f16_a(1.5f);
    Float16 f16_b(1.5f);
    Float16 f16_c(2.5f);

    EXPECT_EQ(f16_a, f16_b);
    EXPECT_NE(f16_a, f16_c);
}

// ============================================================================
// BFloat16 Tests
// ============================================================================

TEST(BFloat16Test, BasicConversion) {
    // Test basic float to BFloat16 conversion
    BFloat16 bf16_zero(0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(bf16_zero), 0.0f);

    BFloat16 bf16_one(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(bf16_one), 1.0f);

    BFloat16 bf16_neg(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(bf16_neg), -1.0f);
}

TEST(BFloat16Test, DynamicRange) {
    // BFloat16 should have same range as Float32
    BFloat16 bf16_large(3.0e38f);
    EXPECT_GT(static_cast<float>(bf16_large), 1e37f);

    BFloat16 bf16_small(1.0e-38f);
    float recovered = static_cast<float>(bf16_small);
    EXPECT_LT(recovered, 1e-37f);
    EXPECT_GT(recovered, 0.0f);
}

TEST(BFloat16Test, SpecialValues) {
    // Test infinity
    BFloat16 bf16_inf(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isinf(static_cast<float>(bf16_inf)));

    // Test NaN
    BFloat16 bf16_nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(std::isnan(static_cast<float>(bf16_nan)));
}

TEST(BFloat16Test, Precision) {
    // BFloat16 has less precision than Float16 but wider range
    // Expect approximately 2 decimal digits of precision
    std::vector<float> test_values = {
        1.23f, 12.3f, 123.0f, 1230.0f,
        0.0123f, 0.123f, 1.23f
    };

    for (float val : test_values) {
        BFloat16 bf16(val);
        float recovered = static_cast<float>(bf16);
        float relative_error = std::abs((recovered - val) / val);
        EXPECT_LT(relative_error, 0.02f) << "Value: " << val;  // < 2% error
    }
}

TEST(BFloat16Test, Comparison) {
    BFloat16 bf16_a(1.5f);
    BFloat16 bf16_b(1.5f);
    BFloat16 bf16_c(2.5f);

    EXPECT_EQ(bf16_a, bf16_b);
    EXPECT_NE(bf16_a, bf16_c);
}

// ============================================================================
// Tensor Tests with Float16/BFloat16
// ============================================================================

TEST(TensorFP16Test, CreationAndAccess) {
    // Create Float16 tensor
    Tensor t_fp16({2, 3}, DType::Float16, Device::cpu());
    EXPECT_EQ(t_fp16.dtype(), DType::Float16);
    EXPECT_EQ(t_fp16.numel(), 6);

    // Create BFloat16 tensor
    Tensor t_bf16({2, 3}, DType::BFloat16, Device::cpu());
    EXPECT_EQ(t_bf16.dtype(), DType::BFloat16);
    EXPECT_EQ(t_bf16.numel(), 6);
}

TEST(TensorFP16Test, DTypeSize) {
    // Verify dtype sizes
    EXPECT_EQ(dtype_size(DType::Float16), 2);
    EXPECT_EQ(dtype_size(DType::BFloat16), 2);
    EXPECT_EQ(dtype_size(DType::Float32), 4);
}

TEST(TensorFP16Test, DTypeName) {
    // Verify dtype names
    EXPECT_EQ(dtype_name(DType::Float16), "float16");
    EXPECT_EQ(dtype_name(DType::BFloat16), "bfloat16");
}

// ============================================================================
// Conversion Tests
// ============================================================================

TEST(ConversionTest, Float16ToFloat32) {
    std::vector<float> test_values = {
        0.0f, 1.0f, -1.0f, 3.14159f, -2.71828f,
        0.001f, 1000.0f, -500.0f
    };

    for (float val : test_values) {
        Float16 f16(val);
        float recovered = static_cast<float>(f16);
        
        // Allow 0.1% relative error or 0.001 absolute error for small values
        if (std::abs(val) > 0.01f) {
            float relative_error = std::abs((recovered - val) / val);
            EXPECT_LT(relative_error, 0.001f) << "Value: " << val;
        } else {
            EXPECT_NEAR(recovered, val, 0.001f) << "Value: " << val;
        }
    }
}

TEST(ConversionTest, BFloat16ToFloat32) {
    std::vector<float> test_values = {
        0.0f, 1.0f, -1.0f, 3.14159f, -2.71828f,
        0.001f, 1000.0f, -500.0f, 1e10f, -1e10f
    };

    for (float val : test_values) {
        BFloat16 bf16(val);
        float recovered = static_cast<float>(bf16);
        
        // BFloat16 has less precision, allow 1% relative error
        if (std::abs(val) > 0.01f) {
            float relative_error = std::abs((recovered - val) / val);
            EXPECT_LT(relative_error, 0.01f) << "Value: " << val;
        } else {
            EXPECT_NEAR(recovered, val, 0.01f) << "Value: " << val;
        }
    }
}

TEST(ConversionTest, RoundTrip) {
    float original = 42.0f;
    
    // Float16 round-trip
    Float16 f16(original);
    float f16_result = static_cast<float>(f16);
    Float16 f16_again(f16_result);
    EXPECT_EQ(f16.bits, f16_again.bits);
    
    // BFloat16 round-trip
    BFloat16 bf16(original);
    float bf16_result = static_cast<float>(bf16);
    BFloat16 bf16_again(bf16_result);
    EXPECT_EQ(bf16.bits, bf16_again.bits);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(EdgeCaseTest, Float16Subnormals) {
    // Test subnormal (denormalized) numbers
    float subnormal = 5.96046448e-8f;  // Very small but non-zero
    Float16 f16(subnormal);
    float recovered = static_cast<float>(f16);
    
    // May underflow to zero or become subnormal
    EXPECT_GE(recovered, 0.0f);
    EXPECT_LE(recovered, subnormal * 2.0f);
}

TEST(EdgeCaseTest, ZeroPreservation) {
    // Positive zero
    Float16 f16_zero(0.0f);
    EXPECT_EQ(static_cast<float>(f16_zero), 0.0f);
    
    // Negative zero
    Float16 f16_neg_zero(-0.0f);
    EXPECT_EQ(static_cast<float>(f16_neg_zero), -0.0f);
    
    // BFloat16
    BFloat16 bf16_zero(0.0f);
    EXPECT_EQ(static_cast<float>(bf16_zero), 0.0f);
    
    BFloat16 bf16_neg_zero(-0.0f);
    EXPECT_EQ(static_cast<float>(bf16_neg_zero), -0.0f);
}

// ============================================================================
// Performance Characteristics
// ============================================================================

TEST(PerformanceTest, Float16VsBFloat16Precision) {
    // Compare precision of Float16 vs BFloat16 for typical neural network values
    std::vector<float> nn_values = {
        0.01f, 0.1f, 0.5f, 1.0f, 2.0f, 10.0f,
        -0.01f, -0.1f, -0.5f, -1.0f, -2.0f, -10.0f
    };
    
    for (float val : nn_values) {
        Float16 f16(val);
        BFloat16 bf16(val);
        
        float f16_error = std::abs(static_cast<float>(f16) - val);
        float bf16_error = std::abs(static_cast<float>(bf16) - val);
        
        // Both should have reasonable accuracy for these values
        EXPECT_LT(f16_error, 0.01f) << "Float16 error too large for value: " << val;
        EXPECT_LT(bf16_error, 0.02f) << "BFloat16 error too large for value: " << val;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
