/**
 * @file test_dtype_conversion.cpp
 * @brief Test dtype conversion functionality
 */

#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>

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

class DTypeConversionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test-specific setup if needed
    }
};

// Test Float32 to Float64 conversion
TEST_F(DTypeConversionTest, Float32ToFloat64) {
    // Create Float32 tensor
    auto t_f32 = tenzor::ones({2, 3}, DType::Float32, Device::cpu());
    t_f32.data<float>()[0] = 1.5f;
    t_f32.data<float>()[1] = 2.5f;
    t_f32.data<float>()[2] = 3.5f;

    // Convert to Float64
    auto t_f64 = t_f32.to(DType::Float64);

    ASSERT_EQ(t_f64.dtype(), DType::Float64);
    ASSERT_EQ(t_f64.numel(), 6);
    EXPECT_DOUBLE_EQ(t_f64.data<double>()[0], 1.5);
    EXPECT_DOUBLE_EQ(t_f64.data<double>()[1], 2.5);
    EXPECT_DOUBLE_EQ(t_f64.data<double>()[2], 3.5);
}

// Test Float32 to Int32 conversion
TEST_F(DTypeConversionTest, Float32ToInt32) {
    auto t_f32 = tenzor::zeros({2, 2}, DType::Float32, Device::cpu());
    t_f32.data<float>()[0] = 1.9f;
    t_f32.data<float>()[1] = 2.1f;
    t_f32.data<float>()[2] = -3.7f;
    t_f32.data<float>()[3] = 4.5f;

    auto t_i32 = t_f32.to(DType::Int32);

    ASSERT_EQ(t_i32.dtype(), DType::Int32);
    EXPECT_EQ(t_i32.data<int32_t>()[0], 1);
    EXPECT_EQ(t_i32.data<int32_t>()[1], 2);
    EXPECT_EQ(t_i32.data<int32_t>()[2], -3);
    EXPECT_EQ(t_i32.data<int32_t>()[3], 4);
}

// Test Int64 to Float32 conversion
TEST_F(DTypeConversionTest, Int64ToFloat32) {
    auto t_i64 = tenzor::zeros({3}, DType::Int64, Device::cpu());
    t_i64.data<int64_t>()[0] = 100;
    t_i64.data<int64_t>()[1] = -200;
    t_i64.data<int64_t>()[2] = 300;

    auto t_f32 = t_i64.to(DType::Float32);

    ASSERT_EQ(t_f32.dtype(), DType::Float32);
    EXPECT_FLOAT_EQ(t_f32.data<float>()[0], 100.0f);
    EXPECT_FLOAT_EQ(t_f32.data<float>()[1], -200.0f);
    EXPECT_FLOAT_EQ(t_f32.data<float>()[2], 300.0f);
}

// Test UInt8 to Float32 conversion
TEST_F(DTypeConversionTest, UInt8ToFloat32) {
    auto t_u8 = tenzor::zeros({4}, DType::UInt8, Device::cpu());
    t_u8.data<uint8_t>()[0] = 0;
    t_u8.data<uint8_t>()[1] = 127;
    t_u8.data<uint8_t>()[2] = 255;
    t_u8.data<uint8_t>()[3] = 64;

    auto t_f32 = t_u8.to(DType::Float32);

    ASSERT_EQ(t_f32.dtype(), DType::Float32);
    EXPECT_FLOAT_EQ(t_f32.data<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(t_f32.data<float>()[1], 127.0f);
    EXPECT_FLOAT_EQ(t_f32.data<float>()[2], 255.0f);
    EXPECT_FLOAT_EQ(t_f32.data<float>()[3], 64.0f);
}

// Test Bool to Float32 conversion
TEST_F(DTypeConversionTest, BoolToFloat32) {
    auto t_bool = tenzor::zeros({3}, DType::Bool, Device::cpu());
    t_bool.data<bool>()[0] = true;
    t_bool.data<bool>()[1] = false;
    t_bool.data<bool>()[2] = true;

    auto t_f32 = t_bool.to(DType::Float32);

    ASSERT_EQ(t_f32.dtype(), DType::Float32);
    EXPECT_FLOAT_EQ(t_f32.data<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(t_f32.data<float>()[1], 0.0f);
    EXPECT_FLOAT_EQ(t_f32.data<float>()[2], 1.0f);
}

// Test same dtype (should return as-is)
TEST_F(DTypeConversionTest, SameDType) {
    auto t_f32 = tenzor::ones({2, 2}, DType::Float32, Device::cpu());
    auto t_same = t_f32.to(DType::Float32);

    ASSERT_EQ(t_same.dtype(), DType::Float32);
    EXPECT_EQ(t_same.data_ptr(), t_f32.data_ptr()); // Should be same tensor
}

// Test shape preservation
TEST_F(DTypeConversionTest, ShapePreservation) {
    auto t_src = tenzor::ones({2, 3, 4}, DType::Float32, Device::cpu());
    auto t_dst = t_src.to(DType::Int32);

    ASSERT_EQ(t_dst.ndim(), 3);
    ASSERT_EQ(t_dst.shape()[0], 2);
    ASSERT_EQ(t_dst.shape()[1], 3);
    ASSERT_EQ(t_dst.shape()[2], 4);
}

// Test requires_grad preservation
TEST_F(DTypeConversionTest, RequiresGradPreservation) {
    auto t_src = tenzor::ones({2, 2}, DType::Float32, Device::cpu());
    t_src.set_requires_grad(true);

    auto t_dst = t_src.to(DType::Float64);

    EXPECT_TRUE(t_dst.requires_grad());
}

// Test Float32 to Int8 conversion with clamping
TEST_F(DTypeConversionTest, Float32ToInt8) {
    auto t_f32 = tenzor::zeros({4}, DType::Float32, Device::cpu());
    t_f32.data<float>()[0] = -128.0f;
    t_f32.data<float>()[1] = 127.0f;
    t_f32.data<float>()[2] = 0.0f;
    t_f32.data<float>()[3] = 50.5f;

    auto t_i8 = t_f32.to(DType::Int8);

    ASSERT_EQ(t_i8.dtype(), DType::Int8);
    EXPECT_EQ(t_i8.data<int8_t>()[0], -128);
    EXPECT_EQ(t_i8.data<int8_t>()[1], 127);
    EXPECT_EQ(t_i8.data<int8_t>()[2], 0);
    EXPECT_EQ(t_i8.data<int8_t>()[3], 50);
}

// Test multiple conversions
TEST_F(DTypeConversionTest, MultipleConversions) {
    auto t1 = tenzor::ones({2, 2}, DType::Float32, Device::cpu());
    t1.data<float>()[0] = 3.14f;

    auto t2 = t1.to(DType::Float64);
    auto t3 = t2.to(DType::Int32);
    auto t4 = t3.to(DType::Float32);

    ASSERT_EQ(t4.dtype(), DType::Float32);
    EXPECT_FLOAT_EQ(t4.data<float>()[0], 3.0f); // Lost fractional part in conversion to int
}

// ============================================================================
// FP8 Conversion Tests
// ============================================================================

TEST_F(DTypeConversionTest, Float32ToFP8E4M3Roundtrip) {
    auto t = tenzor::ones({4}, DType::Float32, Device::cpu());
    t.data<float>()[0] = 0.0f;
    t.data<float>()[1] = 1.0f;
    t.data<float>()[2] = -1.0f;
    t.data<float>()[3] = 42.0f;

    auto t_fp8 = t.to(DType::FP8_E4M3);
    ASSERT_EQ(t_fp8.dtype(), DType::FP8_E4M3);
    ASSERT_EQ(t_fp8.numel(), 4);

    auto t_back = t_fp8.to(DType::Float32);
    ASSERT_EQ(t_back.dtype(), DType::Float32);
    EXPECT_FLOAT_EQ(t_back.data<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(t_back.data<float>()[1], 1.0f);
    EXPECT_FLOAT_EQ(t_back.data<float>()[2], -1.0f);
    // 42.0 should roundtrip within E4M3 precision (3-bit mantissa)
    EXPECT_NEAR(t_back.data<float>()[3], 42.0f, 2.0f);
}

TEST_F(DTypeConversionTest, Float32ToFP8E5M2Roundtrip) {
    auto t = tenzor::ones({4}, DType::Float32, Device::cpu());
    t.data<float>()[0] = 0.0f;
    t.data<float>()[1] = 1.0f;
    t.data<float>()[2] = -1.0f;
    t.data<float>()[3] = 256.0f;

    auto t_fp8 = t.to(DType::FP8_E5M2);
    ASSERT_EQ(t_fp8.dtype(), DType::FP8_E5M2);

    auto t_back = t_fp8.to(DType::Float32);
    ASSERT_EQ(t_back.dtype(), DType::Float32);
    EXPECT_FLOAT_EQ(t_back.data<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(t_back.data<float>()[1], 1.0f);
    EXPECT_FLOAT_EQ(t_back.data<float>()[2], -1.0f);
    // 256.0 should roundtrip within E5M2 precision (2-bit mantissa)
    EXPECT_NEAR(t_back.data<float>()[3], 256.0f, 64.0f);
}

TEST_F(DTypeConversionTest, FP8E4M3ToFloat16) {
    auto t = tenzor::ones({2}, DType::Float32, Device::cpu());
    t.data<float>()[0] = 2.0f;
    t.data<float>()[1] = -3.0f;

    auto t_fp8 = t.to(DType::FP8_E4M3);
    auto t_f16 = t_fp8.to(DType::Float16);
    ASSERT_EQ(t_f16.dtype(), DType::Float16);

    auto t_f32 = t_f16.to(DType::Float32);
    EXPECT_NEAR(t_f32.data<float>()[0], 2.0f, 0.1f);
    EXPECT_NEAR(t_f32.data<float>()[1], -3.0f, 0.5f);
}

TEST_F(DTypeConversionTest, Int32ToFP8E4M3) {
    auto t = tenzor::ones({3}, DType::Int32, Device::cpu());
    t.data<int32_t>()[0] = 0;
    t.data<int32_t>()[1] = 5;
    t.data<int32_t>()[2] = -10;

    auto t_fp8 = t.to(DType::FP8_E4M3);
    ASSERT_EQ(t_fp8.dtype(), DType::FP8_E4M3);

    auto t_back = t_fp8.to(DType::Float32);
    EXPECT_FLOAT_EQ(t_back.data<float>()[0], 0.0f);
    EXPECT_NEAR(t_back.data<float>()[1], 5.0f, 1.0f);
    EXPECT_NEAR(t_back.data<float>()[2], -10.0f, 2.0f);
}

TEST_F(DTypeConversionTest, FP8E4M3ToInt32) {
    auto t = tenzor::ones({2}, DType::Float32, Device::cpu());
    t.data<float>()[0] = 7.0f;
    t.data<float>()[1] = -3.0f;

    auto t_fp8 = t.to(DType::FP8_E4M3);
    auto t_int = t_fp8.to(DType::Int32);
    ASSERT_EQ(t_int.dtype(), DType::Int32);
    EXPECT_NEAR(t_int.data<int32_t>()[0], 7, 1);
    EXPECT_NEAR(t_int.data<int32_t>()[1], -3, 1);
}

TEST_F(DTypeConversionTest, FP8E4M3ZeroValues) {
    // Ensure zero converts correctly (not garbage)
    auto t = tenzor::zeros({8}, DType::Float32, Device::cpu());
    auto t_fp8 = t.to(DType::FP8_E4M3);
    auto t_back = t_fp8.to(DType::Float32);

    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(t_back.data<float>()[i], 0.0f)
            << "Element " << i << " should be zero after FP8 roundtrip";
    }
}

TEST_F(DTypeConversionTest, FP8CrossConversion) {
    // E4M3 -> E5M2 conversion
    auto t = tenzor::ones({2}, DType::Float32, Device::cpu());
    t.data<float>()[0] = 1.5f;
    t.data<float>()[1] = -4.0f;

    auto t_e4m3 = t.to(DType::FP8_E4M3);
    auto t_e5m2 = t_e4m3.to(DType::FP8_E5M2);
    ASSERT_EQ(t_e5m2.dtype(), DType::FP8_E5M2);

    auto t_back = t_e5m2.to(DType::Float32);
    EXPECT_NEAR(t_back.data<float>()[0], 1.5f, 0.5f);
    EXPECT_NEAR(t_back.data<float>()[1], -4.0f, 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
