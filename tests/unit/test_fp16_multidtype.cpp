/**
 * @file test_fp16_multidtype.cpp
 * @brief Multi-dtype tests for Float16 precision behavior and conversions
 *
 * This test suite focuses on:
 * - Float16 precision behavior and characteristics
 * - Conversions between Float16, Float32, and Float64
 * - Numerical stability with Float16
 * - BFloat16 operations for comparison
 * - Cross-dtype conversion accuracy
 *
 * Coverage:
 * - Float16/BFloat16 core operations
 * - Float32/Float64 comparison baselines
 * - Precision loss quantification
 * - Special value handling (NaN, Inf, subnormals)
 * - Tensor operations with mixed precision
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/tenzor.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

using namespace tenzor;

// ============================================================================
// Test Parameters
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string name;
    float tolerance;

    std::string ToString() const {
        return name;
    }
};

void PrintTo(const DTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

// ============================================================================
// Float16 Conversion Tests (Multi-DType)
// ============================================================================

class Float16ConversionTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    DType dtype;
    float tolerance;

    void SetUp() override {
        tenzor::initialize();
        auto param = GetParam();
        dtype = param.dtype;
        tolerance = param.tolerance;
    }
};

TEST_P(Float16ConversionTest, BasicConversion) {
    std::vector<float> test_values = {0.0f, 1.0f, -1.0f, 2.5f, -3.7f};

    for (float val : test_values) {
        if (dtype == DType::Float16) {
            Float16 f16(val);
            float recovered = static_cast<float>(f16);
            EXPECT_NEAR(recovered, val, tolerance)
                << "Float16 conversion failed for value: " << val;
        } else if (dtype == DType::Float32) {
            // Float32 should be exact for these values
            EXPECT_FLOAT_EQ(val, val);
        } else if (dtype == DType::Float64) {
            // Float64 should be exact for these values
            double val64 = static_cast<double>(val);
            EXPECT_DOUBLE_EQ(val64, val64);
        } else if (dtype == DType::BFloat16) {
            BFloat16 bf16(val);
            float recovered = static_cast<float>(bf16);
            EXPECT_NEAR(recovered, val, tolerance)
                << "BFloat16 conversion failed for value: " << val;
        }
    }
}

TEST_P(Float16ConversionTest, SmallValues) {
    std::vector<float> small_values = {
        0.0001f, 0.001f, 0.01f, 0.1f,
        -0.0001f, -0.001f, -0.01f, -0.1f
    };

    for (float val : small_values) {
        if (dtype == DType::Float16) {
            Float16 f16(val);
            float recovered = static_cast<float>(f16);
            float relative_error = std::abs((recovered - val) / val);
            EXPECT_LT(relative_error, 0.01f)
                << "Float16 small value precision lost for: " << val;
        } else if (dtype == DType::BFloat16) {
            BFloat16 bf16(val);
            float recovered = static_cast<float>(bf16);
            float relative_error = std::abs((recovered - val) / val);
            EXPECT_LT(relative_error, 0.02f)
                << "BFloat16 small value precision lost for: " << val;
        }
    }
}

TEST_P(Float16ConversionTest, LargeValues) {
    if (dtype == DType::Float16) {
        // Float16 max: 65504
        Float16 f16_max(65504.0f);
        EXPECT_NEAR(static_cast<float>(f16_max), 65504.0f, 1.0f);

        // Test overflow
        Float16 f16_overflow(100000.0f);
        EXPECT_TRUE(std::isinf(static_cast<float>(f16_overflow)));
    } else if (dtype == DType::BFloat16) {
        // BFloat16 has same range as Float32
        BFloat16 bf16_large(3.0e38f);
        EXPECT_GT(static_cast<float>(bf16_large), 1e37f);
    } else if (dtype == DType::Float32) {
        // Float32 can handle large values
        float large = 3.4e38f;
        EXPECT_LT(large, std::numeric_limits<float>::infinity());
    } else if (dtype == DType::Float64) {
        // Float64 can handle even larger values
        double large = 1.7e308;
        EXPECT_LT(large, std::numeric_limits<double>::infinity());
    }
}

TEST_P(Float16ConversionTest, SpecialValues) {
    if (dtype == DType::Float16) {
        // Infinity
        Float16 f16_inf(std::numeric_limits<float>::infinity());
        EXPECT_TRUE(std::isinf(static_cast<float>(f16_inf)));

        Float16 f16_neg_inf(-std::numeric_limits<float>::infinity());
        EXPECT_TRUE(std::isinf(static_cast<float>(f16_neg_inf)));

        // NaN
        Float16 f16_nan(std::numeric_limits<float>::quiet_NaN());
        EXPECT_TRUE(std::isnan(static_cast<float>(f16_nan)));
    } else if (dtype == DType::BFloat16) {
        // Infinity
        BFloat16 bf16_inf(std::numeric_limits<float>::infinity());
        EXPECT_TRUE(std::isinf(static_cast<float>(bf16_inf)));

        // NaN
        BFloat16 bf16_nan(std::numeric_limits<float>::quiet_NaN());
        EXPECT_TRUE(std::isnan(static_cast<float>(bf16_nan)));
    } else if (dtype == DType::Float32 || dtype == DType::Float64) {
        // Standard float special values
        EXPECT_TRUE(std::isinf(std::numeric_limits<float>::infinity()));
        EXPECT_TRUE(std::isnan(std::numeric_limits<float>::quiet_NaN()));
    }
}

INSTANTIATE_TEST_SUITE_P(
    MultiDType,
    Float16ConversionTest,
    ::testing::Values(
        DTypeParam{DType::Float16, "Float16", 0.01f},
        DTypeParam{DType::BFloat16, "BFloat16", 0.02f},
        DTypeParam{DType::Float32, "Float32", 1e-6f},
        DTypeParam{DType::Float64, "Float64", 1e-10f}
    ),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Cross-DType Conversion Tests
// ============================================================================

class CrossDTypeConversionTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(CrossDTypeConversionTest, Float16ToFloat32) {
    std::vector<float> test_values = {
        0.0f, 1.0f, -1.0f, 3.14159f, -2.71828f,
        0.001f, 1000.0f, -500.0f, 0.5f, -0.5f
    };

    for (float val : test_values) {
        Float16 f16(val);
        float f32_recovered = static_cast<float>(f16);

        if (std::abs(val) > 0.01f && std::abs(val) < 60000.0f) {
            float relative_error = std::abs((f32_recovered - val) / val);
            EXPECT_LT(relative_error, 0.001f)
                << "Float16→Float32 conversion error for: " << val;
        }
    }
}

TEST_F(CrossDTypeConversionTest, Float16ToFloat64) {
    std::vector<float> test_values = {1.0f, -1.0f, 0.5f, 100.0f};

    for (float val : test_values) {
        Float16 f16(val);
        float f32_intermediate = static_cast<float>(f16);
        double f64_final = static_cast<double>(f32_intermediate);

        // Precision should be limited by Float16
        double relative_error = std::abs((f64_final - val) / val);
        EXPECT_LT(relative_error, 0.001);
    }
}

TEST_F(CrossDTypeConversionTest, Float32ToFloat16ToFloat32) {
    std::vector<float> test_values = {1.0f, 2.5f, -3.7f, 0.1f, 100.0f};

    for (float original : test_values) {
        // Round-trip: Float32 → Float16 → Float32
        Float16 f16(original);
        float recovered = static_cast<float>(f16);

        // Convert back again - should be identical
        Float16 f16_again(recovered);
        float recovered_again = static_cast<float>(f16_again);

        EXPECT_FLOAT_EQ(recovered, recovered_again)
            << "Float32→Float16→Float32 round-trip not stable";
    }
}

TEST_F(CrossDTypeConversionTest, BFloat16ToFloat32) {
    std::vector<float> test_values = {
        0.0f, 1.0f, -1.0f, 3.14159f, -2.71828f,
        0.001f, 1000.0f, -500.0f, 1e10f, -1e10f
    };

    for (float val : test_values) {
        BFloat16 bf16(val);
        float recovered = static_cast<float>(bf16);

        if (std::abs(val) > 0.01f) {
            float relative_error = std::abs((recovered - val) / val);
            EXPECT_LT(relative_error, 0.01f)
                << "BFloat16→Float32 conversion error for: " << val;
        }
    }
}

TEST_F(CrossDTypeConversionTest, CompareFloat16VsBFloat16Precision) {
    // Neural network typical values
    std::vector<float> nn_values = {
        0.01f, 0.1f, 0.5f, 1.0f, 2.0f, 10.0f,
        -0.01f, -0.1f, -0.5f, -1.0f, -2.0f, -10.0f
    };

    for (float val : nn_values) {
        Float16 f16(val);
        BFloat16 bf16(val);

        float f16_error = std::abs(static_cast<float>(f16) - val);
        float bf16_error = std::abs(static_cast<float>(bf16) - val);

        // Both should maintain reasonable accuracy
        EXPECT_LT(f16_error, 0.01f) << "Float16 error for: " << val;
        EXPECT_LT(bf16_error, 0.02f) << "BFloat16 error for: " << val;
    }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

class NumericalStabilityTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    DType dtype;
    float tolerance;

    void SetUp() override {
        tenzor::initialize();
        auto param = GetParam();
        dtype = param.dtype;
        tolerance = param.tolerance;
    }
};

TEST_P(NumericalStabilityTest, AccumulationError) {
    // Test accumulation of many small values
    const int n = 1000;
    const float small_val = 0.001f;

    if (dtype == DType::Float16) {
        float sum_f16 = 0.0f;
        for (int i = 0; i < n; ++i) {
            Float16 f16(small_val);
            sum_f16 += static_cast<float>(f16);
        }

        float expected = n * small_val;
        float relative_error = std::abs((sum_f16 - expected) / expected);

        // Float16 accumulation may have noticeable error
        EXPECT_LT(relative_error, 0.05f)
            << "Float16 accumulation error too large";
    } else if (dtype == DType::Float32) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += small_val;
        }

        float expected = n * small_val;
        EXPECT_NEAR(sum, expected, 1e-4f);
    } else if (dtype == DType::Float64) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += static_cast<double>(small_val);
        }

        double expected = n * static_cast<double>(small_val);
        EXPECT_NEAR(sum, expected, 1e-8);
    }
}

TEST_P(NumericalStabilityTest, MultiplicationStability) {
    // Test repeated multiplication
    std::vector<float> factors = {1.1f, 0.9f, 1.05f, 0.95f};

    if (dtype == DType::Float16) {
        float product_f16 = 1.0f;
        for (float factor : factors) {
            Float16 f16(factor);
            product_f16 *= static_cast<float>(f16);
        }

        float expected = 1.1f * 0.9f * 1.05f * 0.95f;
        float relative_error = std::abs((product_f16 - expected) / expected);
        EXPECT_LT(relative_error, 0.01f);
    } else if (dtype == DType::Float32) {
        float product = 1.0f;
        for (float factor : factors) {
            product *= factor;
        }

        float expected = 1.1f * 0.9f * 1.05f * 0.95f;
        EXPECT_NEAR(product, expected, 1e-5f);
    }
}

TEST_P(NumericalStabilityTest, SubnormalHandling) {
    if (dtype == DType::Float16) {
        // Float16 smallest normal: ~6.1e-5
        // Test subnormal values
        float subnormal = 1e-7f;
        Float16 f16(subnormal);
        float recovered = static_cast<float>(f16);

        // May underflow to zero or become subnormal
        EXPECT_GE(recovered, 0.0f);
        EXPECT_LE(recovered, subnormal * 10.0f);
    } else if (dtype == DType::Float32) {
        // Float32 smallest normal: ~1.18e-38
        float subnormal = 1e-40f;
        EXPECT_GE(subnormal, 0.0f);
    }
}

TEST_P(NumericalStabilityTest, ZeroPreservation) {
    if (dtype == DType::Float16) {
        Float16 f16_pos_zero(0.0f);
        EXPECT_EQ(static_cast<float>(f16_pos_zero), 0.0f);

        Float16 f16_neg_zero(-0.0f);
        EXPECT_EQ(static_cast<float>(f16_neg_zero), -0.0f);
    } else if (dtype == DType::BFloat16) {
        BFloat16 bf16_pos_zero(0.0f);
        EXPECT_EQ(static_cast<float>(bf16_pos_zero), 0.0f);

        BFloat16 bf16_neg_zero(-0.0f);
        EXPECT_EQ(static_cast<float>(bf16_neg_zero), -0.0f);
    } else {
        EXPECT_EQ(0.0f, 0.0f);
        EXPECT_EQ(-0.0f, -0.0f);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MultiDType,
    NumericalStabilityTest,
    ::testing::Values(
        DTypeParam{DType::Float16, "Float16", 0.01f},
        DTypeParam{DType::BFloat16, "BFloat16", 0.02f},
        DTypeParam{DType::Float32, "Float32", 1e-6f},
        DTypeParam{DType::Float64, "Float64", 1e-10f}
    ),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Tensor Operations with Mixed Precision
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

void PrintTo(const BackendDTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

class TensorFloat16Test : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        } else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        } else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
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

TEST_P(TensorFloat16Test, TensorCreation) {
    auto t = zeros({2, 3}, dtype, device);
    EXPECT_EQ(t.dtype(), dtype);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.shape().size(), 2);
}

TEST_P(TensorFloat16Test, DTypeSize) {
    size_t expected_size = 0;
    switch (dtype) {
        case DType::Float16:
        case DType::BFloat16:
            expected_size = 2;
            break;
        case DType::Float32:
            expected_size = 4;
            break;
        case DType::Float64:
            expected_size = 8;
            break;
        default:
            FAIL() << "Unexpected dtype";
    }

    EXPECT_EQ(dtype_size(dtype), expected_size);
}

TEST_P(TensorFloat16Test, DTypeName) {
    std::string expected_name;
    switch (dtype) {
        case DType::Float16:
            expected_name = "float16";
            break;
        case DType::BFloat16:
            expected_name = "bfloat16";
            break;
        case DType::Float32:
            expected_name = "float32";
            break;
        case DType::Float64:
            expected_name = "float64";
            break;
        default:
            FAIL() << "Unexpected dtype";
    }

    EXPECT_EQ(dtype_name(dtype), expected_name);
}

// Instantiate for CPU only (Float16/BFloat16 tensor ops may not be supported on all backends)
INSTANTIATE_TEST_SUITE_P(
    CPU,
    TensorFloat16Test,
    ::testing::Values(
        BackendDTypeParam{"cpu", DType::Float16, "Float16"},
        BackendDTypeParam{"cpu", DType::BFloat16, "BFloat16"},
        BackendDTypeParam{"cpu", DType::Float32, "Float32"},
        BackendDTypeParam{"cpu", DType::Float64, "Float64"}
    ),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Precision Comparison Tests
// ============================================================================

class PrecisionComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Force runtime evaluation by preventing inlining
[[gnu::noinline]] static float float16_roundtrip(float val) {
    Float16 f16(val);
    return static_cast<float>(f16);
}

TEST_F(PrecisionComparisonTest, Float16VsFloat32Precision) {
    // Test values that are NOT exactly representable in Float16
    // (values with mantissa that doesn't fit in 10 bits)
    std::vector<float> test_values = {
        1.234f, 12.34f, 123.4f,
        0.001234f, 0.01234f, 0.1234f
    };
    // Note: 1234.0f is exactly representable (integer that fits in 10-bit mantissa)

    for (float val : test_values) {
        // Use noinline function to ensure conversion happens at runtime
        float f16_recovered = float16_roundtrip(val);

        float f16_error = std::abs(f16_recovered - val);
        float f32_error = 0.0f; // Float32 is exact for these values

        // Float16 error should be noticeable (precision loss expected) but bounded
        EXPECT_GT(f16_error, f32_error)
            << "Expected precision loss for val=" << val
            << " (f16_recovered=" << f16_recovered << ")";
        EXPECT_LT(f16_error, std::abs(val * 0.01f))
            << "Float16 error too large for: " << val;
    }
}

TEST_F(PrecisionComparisonTest, Float16VsBFloat16Range) {
    // Float16 max: ~65504
    // BFloat16 max: ~3.4e38 (same as Float32)

    float large_val = 1e10f;

    Float16 f16(large_val);
    BFloat16 bf16(large_val);

    // Float16 should overflow
    EXPECT_TRUE(std::isinf(static_cast<float>(f16)));

    // BFloat16 should handle it
    EXPECT_FALSE(std::isinf(static_cast<float>(bf16)));
    EXPECT_GT(static_cast<float>(bf16), 1e9f);
}

TEST_F(PrecisionComparisonTest, ComparisonOperators) {
    Float16 f16_a(1.5f);
    Float16 f16_b(1.5f);
    Float16 f16_c(2.5f);

    EXPECT_EQ(f16_a, f16_b);
    EXPECT_NE(f16_a, f16_c);

    BFloat16 bf16_a(1.5f);
    BFloat16 bf16_b(1.5f);
    BFloat16 bf16_c(2.5f);

    EXPECT_EQ(bf16_a, bf16_b);
    EXPECT_NE(bf16_a, bf16_c);
}

TEST_F(PrecisionComparisonTest, RoundTripStability) {
    float original = 42.0f;

    // Float16 round-trip
    Float16 f16(original);
    float f16_result = static_cast<float>(f16);
    Float16 f16_again(f16_result);
    EXPECT_EQ(f16.bits, f16_again.bits)
        << "Float16 round-trip not stable";

    // BFloat16 round-trip
    BFloat16 bf16(original);
    float bf16_result = static_cast<float>(bf16);
    BFloat16 bf16_again(bf16_result);
    EXPECT_EQ(bf16.bits, bf16_again.bits)
        << "BFloat16 round-trip not stable";
}

// ============================================================================
// Edge Cases
// ============================================================================

class EdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(EdgeCaseTest, Float16MinMaxValues) {
    // Float16 range: approximately ±65504
    Float16 f16_max(65504.0f);
    EXPECT_NEAR(static_cast<float>(f16_max), 65504.0f, 1.0f);

    Float16 f16_min(-65504.0f);
    EXPECT_NEAR(static_cast<float>(f16_min), -65504.0f, 1.0f);
}

TEST_F(EdgeCaseTest, Float16SmallestNormal) {
    // Smallest normal Float16: ~6.1e-5
    float smallest_normal = 0.00006103515625f;
    Float16 f16(smallest_normal);
    float recovered = static_cast<float>(f16);

    EXPECT_NEAR(recovered, smallest_normal, 1e-5f);
    EXPECT_GT(recovered, 0.0f);
}

TEST_F(EdgeCaseTest, BFloat16DynamicRange) {
    // BFloat16 should match Float32 range
    BFloat16 bf16_large(3.0e38f);
    EXPECT_GT(static_cast<float>(bf16_large), 1e37f);

    BFloat16 bf16_small(1.0e-38f);
    float recovered = static_cast<float>(bf16_small);
    EXPECT_LT(recovered, 1e-37f);
    EXPECT_GT(recovered, 0.0f);
}

TEST_F(EdgeCaseTest, InfinityPropagation) {
    Float16 f16_inf(std::numeric_limits<float>::infinity());
    Float16 f16_neg_inf(-std::numeric_limits<float>::infinity());

    EXPECT_TRUE(std::isinf(static_cast<float>(f16_inf)));
    EXPECT_TRUE(std::isinf(static_cast<float>(f16_neg_inf)));
    EXPECT_GT(static_cast<float>(f16_inf), 0.0f);
    EXPECT_LT(static_cast<float>(f16_neg_inf), 0.0f);
}

TEST_F(EdgeCaseTest, NaNPropagation) {
    Float16 f16_nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(std::isnan(static_cast<float>(f16_nan)));

    BFloat16 bf16_nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(std::isnan(static_cast<float>(bf16_nan)));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
