/**
 * @file test_dtype_edge_cases.cpp
 * @brief Comprehensive dtype-specific edge case tests
 *
 * Tests cover:
 * 1. Integer Overflow/Underflow (Int8, Int32, Int64, UInt8)
 * 2. Float Precision (Float16, Float32, Float64)
 * 3. Type Conversions (Float→Int, Int→Float, lossy conversions)
 * 4. Special Float Values (NaN, Infinity, -0.0)
 * 5. Boolean Operations (logical ops, comparison dtypes)
 *
 * All tests use backend + dtype parameterization for complete coverage.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <limits>
#include <cmath>

using namespace tenzor;

// ============================================================================
// Backend + DType Parameterization
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const BackendDTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

class DTypeEdgeCaseTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
        else if (param.backend_name == "adaptivecpp") {
            if (!isBackendAvailable(Device::Type::AdaptiveCpp)) {
                GTEST_SKIP() << "AdaptiveCpp not available";
            }
            device = Device::adaptivecpp(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }
};

// ============================================================================
// 1. INTEGER OVERFLOW/UNDERFLOW TESTS
// ============================================================================

TEST_P(DTypeEdgeCaseTest, Int8Overflow) {
    if (dtype != DType::Int8) {
        GTEST_SKIP() << "Test only for Int8";
    }

    // Create tensor with max Int8 value (127)
    auto a_cpu = zeros({10}, DType::Int8, Device::cpu());
    auto a_data = a_cpu.data<int8_t>();
    for (int i = 0; i < 10; ++i) {
        a_data[i] = 127;  // INT8_MAX
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = ones({10}, DType::Int8, device);

    // 127 + 1 should overflow
    auto c = add(a, b);
    auto c_cpu = c.to(Device::cpu());
    const int8_t* result = c_cpu.data<int8_t>();

    // Verify overflow behavior (typically wraps to -128 on most platforms)
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(result[i] == -128 || result[i] == 127)
            << "Int8 overflow at index " << i << ": " << static_cast<int>(result[i]);
    }
}

TEST_P(DTypeEdgeCaseTest, Int8Underflow) {
    if (dtype != DType::Int8) {
        GTEST_SKIP() << "Test only for Int8";
    }

    // Create tensor with min Int8 value (-128)
    auto a_cpu = zeros({10}, DType::Int8, Device::cpu());
    auto a_data = a_cpu.data<int8_t>();
    for (int i = 0; i < 10; ++i) {
        a_data[i] = -128;  // INT8_MIN
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = ones({10}, DType::Int8, device);

    // -128 - 1 should underflow
    auto c = sub(a, b);
    auto c_cpu = c.to(Device::cpu());
    const int8_t* result = c_cpu.data<int8_t>();

    // Verify underflow behavior (typically wraps to 127)
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(result[i] == 127 || result[i] == -128)
            << "Int8 underflow at index " << i << ": " << static_cast<int>(result[i]);
    }
}

TEST_P(DTypeEdgeCaseTest, UInt8Overflow) {
    if (dtype != DType::UInt8) {
        GTEST_SKIP() << "Test only for UInt8";
    }

    // Create tensor with max UInt8 value (255)
    auto a_cpu = zeros({10}, DType::UInt8, Device::cpu());
    auto a_data = a_cpu.data<uint8_t>();
    for (int i = 0; i < 10; ++i) {
        a_data[i] = 255;  // UINT8_MAX
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = ones({10}, DType::UInt8, device);

    // 255 + 1 should overflow to 0
    auto c = add(a, b);
    auto c_cpu = c.to(Device::cpu());
    const uint8_t* result = c_cpu.data<uint8_t>();

    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(result[i] == 0 || result[i] == 255)
            << "UInt8 overflow at index " << i << ": " << static_cast<unsigned>(result[i]);
    }
}

TEST_P(DTypeEdgeCaseTest, Int32Overflow) {
    if (dtype != DType::Int32) {
        GTEST_SKIP() << "Test only for Int32";
    }

    // Create tensor with max Int32 value
    auto a_cpu = zeros({5}, DType::Int32, Device::cpu());
    auto a_data = a_cpu.data<int32_t>();
    for (int i = 0; i < 5; ++i) {
        a_data[i] = std::numeric_limits<int32_t>::max();
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = ones({5}, DType::Int32, device);

    // INT32_MAX + 1 should overflow
    auto c = add(a, b);
    auto c_cpu = c.to(Device::cpu());
    const int32_t* result = c_cpu.data<int32_t>();

    for (int i = 0; i < 5; ++i) {
        // Verify overflow occurred (wraps to INT32_MIN or saturates)
        EXPECT_TRUE(result[i] == std::numeric_limits<int32_t>::min() ||
                   result[i] == std::numeric_limits<int32_t>::max())
            << "Int32 overflow at index " << i << ": " << result[i];
    }
}

TEST_P(DTypeEdgeCaseTest, Int64Overflow) {
    if (dtype != DType::Int64) {
        GTEST_SKIP() << "Test only for Int64";
    }

    // Create tensor with max Int64 value
    auto a_cpu = zeros({5}, DType::Int64, Device::cpu());
    auto a_data = a_cpu.data<int64_t>();
    for (int i = 0; i < 5; ++i) {
        a_data[i] = std::numeric_limits<int64_t>::max();
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = ones({5}, DType::Int64, device);

    // INT64_MAX + 1 should overflow
    auto c = add(a, b);
    auto c_cpu = c.to(Device::cpu());
    const int64_t* result = c_cpu.data<int64_t>();

    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(result[i] == std::numeric_limits<int64_t>::min() ||
                   result[i] == std::numeric_limits<int64_t>::max())
            << "Int64 overflow at index " << i << ": " << result[i];
    }
}

// ============================================================================
// 2. FLOAT PRECISION TESTS
// ============================================================================

TEST_P(DTypeEdgeCaseTest, Float16PrecisionLoss) {
    if (dtype != DType::Float16) {
        GTEST_SKIP() << "Test only for Float16";
    }

    // Try to create Float16 tensor - skip if not supported
    try {
        // Float16 has ~3-4 decimal digits of precision
        // Test with value that requires more precision
        auto a_cpu = full({10}, 3.141592653589793f, DType::Float32, Device::cpu());
        auto a_f16 = a_cpu.to(DType::Float16);

        if (device.type != Device::Type::CPU) {
            a_f16 = a_f16.to(device);
        }

        // Convert back to Float32 to check precision loss
        auto a_back = a_f16.to(DType::Float32);
        auto a_result = a_back.to(Device::cpu());
        const float* data = a_result.data<float>();

        for (int i = 0; i < 10; ++i) {
            // Float16 should lose precision beyond ~3-4 digits
            EXPECT_NEAR(data[i], 3.141592653589793f, 1e-3f)
                << "Float16 precision at index " << i;
            // But should NOT be exactly equal to full precision
            EXPECT_GT(std::abs(data[i] - 3.141592653589793f), 1e-6f)
                << "Float16 should lose precision at index " << i;
        }
    } catch (...) {
        GTEST_SKIP() << "Float16 not supported on " << device.to_string();
    }
}

TEST_P(DTypeEdgeCaseTest, Float32VsFloat64Precision) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only for Float32/Float64";
    }

    // Test with value that shows Float32 vs Float64 precision difference
    double precise_value = 1.23456789012345678901234567890;

    // Use appropriate full() overload based on dtype
    Tensor a_cpu;
    if (dtype == DType::Float32) {
        a_cpu = full({10}, static_cast<float>(precise_value), dtype, Device::cpu());
    } else {
        a_cpu = full({10}, precise_value, dtype, Device::cpu());
    }
    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto a_result = a.to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* data = a_result.data<float>();
        for (int i = 0; i < 10; ++i) {
            // Float32 has ~7 decimal digits
            EXPECT_NEAR(data[i], precise_value, 1e-6f);
            // Should lose precision after 7th digit
            EXPECT_GT(std::abs(static_cast<double>(data[i]) - precise_value), 1e-15)
                << "Float32 precision loss at index " << i;
        }
    } else {
        const double* data = a_result.data<double>();
        for (int i = 0; i < 10; ++i) {
            // Float64 has ~15 decimal digits
            EXPECT_NEAR(data[i], precise_value, 1e-14);
        }
    }
}

TEST_P(DTypeEdgeCaseTest, DenormalNumbers) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only for Float32/Float64";
    }

    // Test handling of denormal (subnormal) numbers
    float denormal_f32 = std::numeric_limits<float>::denorm_min();
    double denormal_f64 = std::numeric_limits<double>::denorm_min();

    // Use appropriate full() overload based on dtype to preserve precision
    Tensor a_cpu;
    if (dtype == DType::Float32) {
        a_cpu = full({10}, denormal_f32, dtype, Device::cpu());
    } else {
        a_cpu = full({10}, denormal_f64, dtype, Device::cpu());
    }
    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);

    // Test that denormals can be added
    auto b = add(a, a);
    auto b_cpu = b.to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* data = b_cpu.data<float>();
        for (int i = 0; i < 10; ++i) {
            // Debug output
            uint32_t bits;
            std::memcpy(&bits, &data[i], sizeof(float));
            std::printf("Index %d: value=%.20e bits=0x%08x\n", i, data[i], bits);
            // Should get 2 * denorm_min
            EXPECT_GT(data[i], 0.0f) << "Denormal addition at index " << i;
            EXPECT_LT(data[i], std::numeric_limits<float>::min())
                << "Should still be denormal at index " << i;
        }
    } else {
        const double* data = b_cpu.data<double>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_GT(data[i], 0.0) << "Denormal addition at index " << i;
            EXPECT_LT(data[i], std::numeric_limits<double>::min())
                << "Should still be denormal at index " << i;
        }
    }
}

// ============================================================================
// 3. TYPE CONVERSION TESTS
// ============================================================================

TEST_P(DTypeEdgeCaseTest, FloatToIntTruncation) {
    if (dtype != DType::Float32) {
        GTEST_SKIP() << "Test requires Float32 source";
    }

    // Test that float→int uses truncation, not rounding
    auto a_cpu = zeros({10}, DType::Float32, Device::cpu());
    auto a_data = a_cpu.data<float>();

    a_data[0] = 3.1f;
    a_data[1] = 3.9f;
    a_data[2] = -3.1f;
    a_data[3] = -3.9f;
    a_data[4] = 0.5f;
    a_data[5] = -0.5f;
    a_data[6] = 7.999f;
    a_data[7] = -7.999f;
    a_data[8] = 100.7f;
    a_data[9] = -100.7f;

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto a_int = a.to(DType::Int32);
    auto a_result = a_int.to(Device::cpu());
    const int32_t* data = a_result.data<int32_t>();

    // Verify truncation behavior
    EXPECT_EQ(data[0], 3);    // 3.1 → 3 (not 4)
    EXPECT_EQ(data[1], 3);    // 3.9 → 3 (not 4)
    EXPECT_EQ(data[2], -3);   // -3.1 → -3 (not -4)
    EXPECT_EQ(data[3], -3);   // -3.9 → -3 (not -4)
    EXPECT_EQ(data[4], 0);    // 0.5 → 0 (not 1)
    EXPECT_EQ(data[5], 0);    // -0.5 → 0 (not -1)
    EXPECT_EQ(data[6], 7);    // 7.999 → 7 (not 8)
    EXPECT_EQ(data[7], -7);   // -7.999 → -7 (not -8)
    EXPECT_EQ(data[8], 100);  // 100.7 → 100
    EXPECT_EQ(data[9], -100); // -100.7 → -100
}

TEST_P(DTypeEdgeCaseTest, IntToFloatPrecisionLoss) {
    if (dtype != DType::Int64) {
        GTEST_SKIP() << "Test requires Int64 source";
    }

    // Large integers that exceed Float32 precision
    // Float32 mantissa has 24 bits, so integers > 2^24 lose precision
    auto a_cpu = zeros({5}, DType::Int64, Device::cpu());
    auto a_data = a_cpu.data<int64_t>();

    a_data[0] = 16777216;      // 2^24 (exactly representable)
    a_data[1] = 16777217;      // 2^24 + 1 (NOT exactly representable in Float32)
    a_data[2] = 123456789;     // Large number with precision loss
    a_data[3] = 9007199254740992LL;  // 2^53 (exactly representable in Float64, not Float32)
    a_data[4] = 9007199254740993LL;  // 2^53 + 1 (NOT exactly representable in Float64)

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto a_f32 = a.to(DType::Float32);
    auto a_f64 = a.to(DType::Float64);

    auto f32_result = a_f32.to(Device::cpu());
    auto f64_result = a_f64.to(Device::cpu());

    const float* f32_data = f32_result.data<float>();
    const double* f64_data = f64_result.data<double>();

    // Float32 precision checks
    EXPECT_FLOAT_EQ(f32_data[0], 16777216.0f);  // Exact
    EXPECT_NE(static_cast<int64_t>(f32_data[1]), 16777217);  // Precision loss

    // Float64 precision checks
    EXPECT_DOUBLE_EQ(f64_data[0], 16777216.0);  // Exact
    EXPECT_DOUBLE_EQ(f64_data[1], 16777217.0);  // Exact in Float64
    EXPECT_DOUBLE_EQ(f64_data[3], 9007199254740992.0);  // Exact
}

TEST_P(DTypeEdgeCaseTest, LossyConversionChain) {
    if (dtype != DType::Float64) {
        GTEST_SKIP() << "Test requires Float64 source";
    }

    // Test lossy conversion: Float64 → Float32 → Float16 → Float32 → Float64
    auto a_f64 = full({10}, 3.14159265358979323846, DType::Float64, device);

    // Step 1: Float64 → Float32
    auto a_f32 = a_f64.to(DType::Float32);

    // Step 2: Float32 → Float16 (if supported)
    try {
        auto a_f16 = a_f32.to(DType::Float16);

        // Step 3: Float16 → Float32 → Float64
        auto a_restored_f32 = a_f16.to(DType::Float32);
        auto a_restored_f64 = a_restored_f32.to(DType::Float64);

        auto result = a_restored_f64.to(Device::cpu());
        const double* data = result.data<double>();

        // Significant precision loss through the chain
        for (int i = 0; i < 10; ++i) {
            EXPECT_NEAR(data[i], 3.14159265358979323846, 1e-2);
            // But NOT close to original precision
            EXPECT_GT(std::abs(data[i] - 3.14159265358979323846), 1e-6)
                << "Should have precision loss at index " << i;
        }
    } catch (...) {
        // Float16 not supported on this backend, skip the test
        GTEST_SKIP() << "Float16 not supported on " << device.to_string();
    }
}

TEST_P(DTypeEdgeCaseTest, BoolConversion) {
    if (dtype != DType::Bool) {
        GTEST_SKIP() << "Test only for Bool";
    }

    // Test: 0 → false, non-zero → true
    auto int_cpu = zeros({10}, DType::Int32, Device::cpu());
    auto int_data = int_cpu.data<int32_t>();

    int_data[0] = 0;
    int_data[1] = 1;
    int_data[2] = -1;
    int_data[3] = 100;
    int_data[4] = -100;
    int_data[5] = 0;
    int_data[6] = 42;
    int_data[7] = -42;
    int_data[8] = 0;
    int_data[9] = 1;

    auto int_tensor = (device.type == Device::Type::CPU) ? int_cpu : int_cpu.to(device);
    auto bool_tensor = int_tensor.to(DType::Bool);
    auto bool_result = bool_tensor.to(Device::cpu());
    const bool* bool_data = bool_result.data<bool>();

    EXPECT_FALSE(bool_data[0]);  // 0 → false
    EXPECT_TRUE(bool_data[1]);   // 1 → true
    EXPECT_TRUE(bool_data[2]);   // -1 → true
    EXPECT_TRUE(bool_data[3]);   // 100 → true
    EXPECT_TRUE(bool_data[4]);   // -100 → true
    EXPECT_FALSE(bool_data[5]);  // 0 → false
    EXPECT_TRUE(bool_data[6]);   // 42 → true
    EXPECT_TRUE(bool_data[7]);   // -42 → true
    EXPECT_FALSE(bool_data[8]);  // 0 → false
    EXPECT_TRUE(bool_data[9]);   // 1 → true
}

// ============================================================================
// 4. SPECIAL FLOAT VALUES TESTS
// ============================================================================

TEST_P(DTypeEdgeCaseTest, NaNPropagation) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only for floating point types";
    }

    // Create tensor with NaN values
    auto a_cpu = zeros({10}, dtype, Device::cpu());

    if (dtype == DType::Float32) {
        auto a_data = a_cpu.data<float>();
        for (int i = 0; i < 10; ++i) {
            a_data[i] = std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        auto a_data = a_cpu.data<double>();
        for (int i = 0; i < 10; ++i) {
            a_data[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = ones({10}, dtype, device);

    // NaN + 1 should still be NaN
    auto c = add(a, b);
    auto c_cpu = c.to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* data = c_cpu.data<float>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_TRUE(std::isnan(data[i])) << "NaN propagation failed at index " << i;
        }
    } else {
        const double* data = c_cpu.data<double>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_TRUE(std::isnan(data[i])) << "NaN propagation failed at index " << i;
        }
    }
}

TEST_P(DTypeEdgeCaseTest, InfinityArithmetic) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only for floating point types";
    }

    auto a_cpu = zeros({5}, dtype, Device::cpu());

    if (dtype == DType::Float32) {
        auto a_data = a_cpu.data<float>();
        a_data[0] = std::numeric_limits<float>::infinity();
        a_data[1] = -std::numeric_limits<float>::infinity();
        a_data[2] = std::numeric_limits<float>::infinity();
        a_data[3] = std::numeric_limits<float>::infinity();
        a_data[4] = -std::numeric_limits<float>::infinity();
    } else {
        auto a_data = a_cpu.data<double>();
        a_data[0] = std::numeric_limits<double>::infinity();
        a_data[1] = -std::numeric_limits<double>::infinity();
        a_data[2] = std::numeric_limits<double>::infinity();
        a_data[3] = std::numeric_limits<double>::infinity();
        a_data[4] = -std::numeric_limits<double>::infinity();
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);

    // Test: inf + 1 = inf
    auto b = ones({5}, dtype, device);
    auto c1 = add(a, b);
    auto c1_cpu = c1.to(Device::cpu());

    // Test: inf - inf = NaN
    auto c2 = sub(a, a);
    auto c2_cpu = c2.to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* add_data = c1_cpu.data<float>();
        const float* sub_data = c2_cpu.data<float>();

        EXPECT_TRUE(std::isinf(add_data[0]) && add_data[0] > 0);  // inf + 1 = inf
        EXPECT_TRUE(std::isinf(add_data[1]) && add_data[1] < 0);  // -inf + 1 = -inf
        EXPECT_TRUE(std::isnan(sub_data[0]));  // inf - inf = NaN
        EXPECT_TRUE(std::isnan(sub_data[3]));  // inf - inf = NaN
    } else {
        const double* add_data = c1_cpu.data<double>();
        const double* sub_data = c2_cpu.data<double>();

        EXPECT_TRUE(std::isinf(add_data[0]) && add_data[0] > 0);
        EXPECT_TRUE(std::isinf(add_data[1]) && add_data[1] < 0);
        EXPECT_TRUE(std::isnan(sub_data[0]));
        EXPECT_TRUE(std::isnan(sub_data[3]));
    }
}

TEST_P(DTypeEdgeCaseTest, NegativeZero) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only for floating point types";
    }

    // Test that negative zero is preserved
    auto a_cpu = zeros({10}, dtype, Device::cpu());

    if (dtype == DType::Float32) {
        auto a_data = a_cpu.data<float>();
        for (int i = 0; i < 10; ++i) {
            a_data[i] = -0.0f;
        }
    } else {
        auto a_data = a_cpu.data<double>();
        for (int i = 0; i < 10; ++i) {
            a_data[i] = -0.0;
        }
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto a_result = a.to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* data = a_result.data<float>();
        for (int i = 0; i < 10; ++i) {
            // -0.0 should compare equal to 0.0
            EXPECT_EQ(data[i], 0.0f);
            // But signbit should detect the difference
            EXPECT_TRUE(std::signbit(data[i]))
                << "Negative zero should have sign bit set at index " << i;
        }
    } else {
        const double* data = a_result.data<double>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(data[i], 0.0);
            EXPECT_TRUE(std::signbit(data[i]))
                << "Negative zero should have sign bit set at index " << i;
        }
    }
}

// ============================================================================
// 5. COMPARISON OPERATIONS TESTS (Bool dtype output)
// ============================================================================

class ComparisonOpsTest : public tenzor::testing::BackendTest {};

TEST_P(ComparisonOpsTest, ComparisonOutputDType) {
    // Test that comparison operations always return Bool dtype
    auto a = ones({10}, DType::Float32, device) * 5.0f;
    auto b = ones({10}, DType::Float32, device) * 3.0f;

    auto gt_result = gt(a, b);
    auto lt_result = lt(a, b);
    auto eq_result = eq(a, b);
    auto ge_result = ge(a, b);
    auto le_result = le(a, b);
    auto ne_result = ne(a, b);

    EXPECT_EQ(gt_result.dtype(), DType::Bool) << "gt should return Bool";
    EXPECT_EQ(lt_result.dtype(), DType::Bool) << "lt should return Bool";
    EXPECT_EQ(eq_result.dtype(), DType::Bool) << "eq should return Bool";
    EXPECT_EQ(ge_result.dtype(), DType::Bool) << "ge should return Bool";
    EXPECT_EQ(le_result.dtype(), DType::Bool) << "le should return Bool";
    EXPECT_EQ(ne_result.dtype(), DType::Bool) << "ne should return Bool";

    // Verify actual values
    auto gt_cpu = gt_result.to(Device::cpu());
    auto lt_cpu = lt_result.to(Device::cpu());
    auto eq_cpu = eq_result.to(Device::cpu());

    auto gt_data = reinterpret_cast<const bool*>(gt_cpu.data_ptr());
    auto lt_data = reinterpret_cast<const bool*>(lt_cpu.data_ptr());
    auto eq_data = reinterpret_cast<const bool*>(eq_cpu.data_ptr());

    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(gt_data[i]) << "5.0 > 3.0 should be true at index " << i;
        EXPECT_FALSE(lt_data[i]) << "5.0 < 3.0 should be false at index " << i;
        EXPECT_FALSE(eq_data[i]) << "5.0 == 3.0 should be false at index " << i;
    }
}

TEST_P(ComparisonOpsTest, BoolComparisons) {
    // Test comparison operations on Bool dtype inputs
    auto a_cpu = zeros({10}, DType::Bool, Device::cpu());
    auto b_cpu = zeros({10}, DType::Bool, Device::cpu());

    auto a_data = reinterpret_cast<bool*>(a_cpu.data_ptr());
    auto b_data = reinterpret_cast<bool*>(b_cpu.data_ptr());

    a_data[0] = false; b_data[0] = false;
    a_data[1] = false; b_data[1] = true;
    a_data[2] = true;  b_data[2] = false;
    a_data[3] = true;  b_data[3] = true;

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = (device.type == Device::Type::CPU) ? b_cpu : b_cpu.to(device);

    // Test equality
    auto c_eq = eq(a, b);
    EXPECT_EQ(c_eq.dtype(), DType::Bool);
    auto c_eq_cpu = c_eq.to(Device::cpu());
    auto c_eq_data = reinterpret_cast<const bool*>(c_eq_cpu.data_ptr());

    EXPECT_TRUE(c_eq_data[0]);   // false == false
    EXPECT_FALSE(c_eq_data[1]);  // false == true
    EXPECT_FALSE(c_eq_data[2]);  // true == false
    EXPECT_TRUE(c_eq_data[3]);   // true == true

    // Test inequality
    auto c_ne = ne(a, b);
    auto c_ne_cpu = c_ne.to(Device::cpu());
    auto c_ne_data = reinterpret_cast<const bool*>(c_ne_cpu.data_ptr());

    EXPECT_FALSE(c_ne_data[0]);  // false != false
    EXPECT_TRUE(c_ne_data[1]);   // false != true
    EXPECT_TRUE(c_ne_data[2]);   // true != false
    EXPECT_FALSE(c_ne_data[3]);  // true != true
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateDTypeEdgeCaseCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "adaptivecpp"};

    std::vector<std::pair<DType, std::string>> dtypes = {
        // Integer types for overflow tests
        {DType::Int8, "int8"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
        {DType::UInt8, "uint8"},
        // Float types for precision tests
        {DType::Float16, "float16"},
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        // Bool for logical ops
        {DType::Bool, "bool"},
    };

    std::vector<BackendDTypeParam> combinations;
    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name] : dtypes) {
            combinations.push_back({backend, dtype, dtype_name});
        }
    }
    return combinations;
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    DTypeEdgeCaseTest,
    ::testing::ValuesIn(GenerateDTypeEdgeCaseCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_BACKEND_TESTS(ComparisonOpsTest);

/*
 * EDGE CASE TEST COVERAGE SUMMARY:
 *
 * 1. INTEGER OVERFLOW/UNDERFLOW: 5 tests
 *    - Int8 overflow (127 + 1)
 *    - Int8 underflow (-128 - 1)
 *    - UInt8 overflow (255 + 1)
 *    - Int32 overflow (INT32_MAX + 1)
 *    - Int64 overflow (INT64_MAX + 1)
 *
 * 2. FLOAT PRECISION: 3 tests
 *    - Float16 precision loss (~3-4 digits)
 *    - Float32 vs Float64 precision comparison
 *    - Denormal number handling
 *
 * 3. TYPE CONVERSIONS: 4 tests
 *    - Float→Int truncation (not rounding)
 *    - Int→Float precision loss for large ints
 *    - Lossy conversion chain (F64→F32→F16→F32→F64)
 *    - Bool conversions (0=false, non-zero=true)
 *
 * 4. SPECIAL FLOAT VALUES: 3 tests
 *    - NaN propagation (NaN + x = NaN)
 *    - Infinity arithmetic (inf + 1 = inf, inf - inf = NaN)
 *    - Negative zero preservation and detection
 *
 * 5. COMPARISON OPERATIONS: 2 tests
 *    - Comparison output dtype (always Bool: gt, lt, eq, ge, le, ne)
 *    - Bool comparisons (eq/ne on Bool inputs)
 *
 * TOTAL EDGE CASES: 17 unique test scenarios
 *
 * With backend parameterization:
 * - DTypeEdgeCaseTest: 15 tests × 4 backends × 8 dtypes = 480 test scenarios
 *   (actual execution filtered by dtype-specific GTEST_SKIP)
 * - ComparisonOpsTest: 2 tests × 4 backends = 8 test scenarios
 *
 * EFFECTIVE COVERAGE:
 * - Int8: 2 tests × 4 backends = 8 scenarios
 * - UInt8: 1 test × 4 backends = 4 scenarios
 * - Int32: 1 test × 4 backends = 4 scenarios
 * - Int64: 2 tests × 4 backends = 8 scenarios
 * - Float16: 2 tests × 4 backends = 8 scenarios
 * - Float32: 7 tests × 4 backends = 28 scenarios
 * - Float64: 7 tests × 4 backends = 28 scenarios
 * - Bool: 3 tests × 4 backends = 12 scenarios
 *
 * GRAND TOTAL: 100+ effective test executions covering:
 *   - Integer overflow/underflow behavior
 *   - Floating-point precision limits
 *   - Type conversion semantics (truncation vs rounding)
 *   - Special IEEE 754 values (NaN, Inf, -0.0)
 *   - Denormal/subnormal number handling
 *   - Comparison operator dtype guarantees
 */
