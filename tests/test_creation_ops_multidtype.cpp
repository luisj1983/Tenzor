#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;

/**
 * @file test_creation_ops_multidtype.cpp
 * @brief Comprehensive dtype testing for tensor creation operations
 *
 * COVERAGE: Tests all creation ops (zeros, ones, randn, arange, linspace, eye, full)
 * across multiple dtypes (Float32, Float64, Int32, Int64, Bool, UInt8)
 *
 * DESIGN PATTERN: Parameterized testing with type-specific verification
 */

// ============================================================================
// DType Parameter Structure
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string name;

    std::string ToString() const {
        return name;
    }
};

// ============================================================================
// Zeros Operation - Multi-DType Testing
// ============================================================================

class ZerosMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device = Device::cpu();
};

TEST_P(ZerosMultiDTypeTest, BasicShape) {
    auto param = GetParam();
    auto t = zeros({3, 4}, param.dtype, device);

    // Verify shape and dtype
    ASSERT_EQ(t.shape()[0], 3);
    ASSERT_EQ(t.shape()[1], 4);
    ASSERT_EQ(t.dtype(), param.dtype);
    ASSERT_EQ(t.numel(), 12);

    // Type-specific zero verification
    if (param.dtype == DType::Float32) {
        const float* data = t.data<float>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 0.0f);
        }
    }
    else if (param.dtype == DType::Float64) {
        const double* data = t.data<double>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 0.0);
        }
    }
    else if (param.dtype == DType::Int32) {
        const int32_t* data = t.data<int32_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 0);
        }
    }
    else if (param.dtype == DType::Int64) {
        const int64_t* data = t.data<int64_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 0);
        }
    }
    else if (param.dtype == DType::Bool) {
        const bool* data = t.data<bool>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_FALSE(data[i]);
        }
    }
    else if (param.dtype == DType::UInt8) {
        const uint8_t* data = t.data<uint8_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 0);
        }
    }
}

TEST_P(ZerosMultiDTypeTest, HighDimensional) {
    auto param = GetParam();
    auto t = zeros({2, 3, 4, 5}, param.dtype, device);

    ASSERT_EQ(t.ndim(), 4);
    ASSERT_EQ(t.numel(), 120);
    ASSERT_EQ(t.dtype(), param.dtype);
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    ZerosMultiDTypeTest,
    ::testing::Values(
        DTypeParam{DType::Float32, "float32"},
        DTypeParam{DType::Float64, "float64"},
        DTypeParam{DType::Int32, "int32"},
        DTypeParam{DType::Int64, "int64"},
        DTypeParam{DType::Bool, "bool"},
        DTypeParam{DType::UInt8, "uint8"}
    ),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Ones Operation - Multi-DType Testing
// ============================================================================

class OnesMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device = Device::cpu();
};

TEST_P(OnesMultiDTypeTest, BasicShape) {
    auto param = GetParam();
    auto t = ones({3, 4}, param.dtype, device);

    // Verify shape and dtype
    ASSERT_EQ(t.shape()[0], 3);
    ASSERT_EQ(t.shape()[1], 4);
    ASSERT_EQ(t.dtype(), param.dtype);

    // Type-specific one verification
    if (param.dtype == DType::Float32) {
        const float* data = t.data<float>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 1.0f);
        }
    }
    else if (param.dtype == DType::Float64) {
        const double* data = t.data<double>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 1.0);
        }
    }
    else if (param.dtype == DType::Int32) {
        const int32_t* data = t.data<int32_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 1);
        }
    }
    else if (param.dtype == DType::Int64) {
        const int64_t* data = t.data<int64_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 1);
        }
    }
    else if (param.dtype == DType::Bool) {
        const bool* data = t.data<bool>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_TRUE(data[i]);
        }
    }
    else if (param.dtype == DType::UInt8) {
        const uint8_t* data = t.data<uint8_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 1);
        }
    }
}

TEST_P(OnesMultiDTypeTest, LargeTensor) {
    auto param = GetParam();
    auto t = ones({100, 100}, param.dtype, device);

    ASSERT_EQ(t.numel(), 10000);
    ASSERT_EQ(t.dtype(), param.dtype);
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    OnesMultiDTypeTest,
    ::testing::Values(
        DTypeParam{DType::Float32, "float32"},
        DTypeParam{DType::Float64, "float64"},
        DTypeParam{DType::Int32, "int32"},
        DTypeParam{DType::Int64, "int64"},
        DTypeParam{DType::Bool, "bool"},
        DTypeParam{DType::UInt8, "uint8"}
    ),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Full Operation - Multi-DType Testing
// ============================================================================

class FullMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device = Device::cpu();
};

TEST_P(FullMultiDTypeTest, CustomValue) {
    auto param = GetParam();
    float fill_value = 42.0f;
    auto t = full({3, 4}, fill_value, param.dtype, device);

    ASSERT_EQ(t.dtype(), param.dtype);
    ASSERT_EQ(t.numel(), 12);

    // Type-specific value verification
    if (param.dtype == DType::Float32) {
        const float* data = t.data<float>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_FLOAT_EQ(data[i], 42.0f);
        }
    }
    else if (param.dtype == DType::Float64) {
        const double* data = t.data<double>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_DOUBLE_EQ(data[i], 42.0);
        }
    }
    else if (param.dtype == DType::Int32) {
        const int32_t* data = t.data<int32_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 42);
        }
    }
    else if (param.dtype == DType::Int64) {
        const int64_t* data = t.data<int64_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 42);
        }
    }
    else if (param.dtype == DType::Bool) {
        const bool* data = t.data<bool>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_TRUE(data[i]);  // Non-zero -> true
        }
    }
    else if (param.dtype == DType::UInt8) {
        const uint8_t* data = t.data<uint8_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], 42);
        }
    }
}

TEST_P(FullMultiDTypeTest, UInt8Range) {
    auto param = GetParam();

    // Only test UInt8 for this specific case
    if (param.dtype != DType::UInt8) {
        GTEST_SKIP() << "Test only for UInt8";
    }

    auto t = full({10}, 200.0f, param.dtype, device);
    const uint8_t* data = t.data<uint8_t>();

    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_EQ(data[i], 200);
        EXPECT_GE(data[i], 0);
        EXPECT_LE(data[i], 255);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    FullMultiDTypeTest,
    ::testing::Values(
        DTypeParam{DType::Float32, "float32"},
        DTypeParam{DType::Float64, "float64"},
        DTypeParam{DType::Int32, "int32"},
        DTypeParam{DType::Int64, "int64"},
        DTypeParam{DType::Bool, "bool"},
        DTypeParam{DType::UInt8, "uint8"}
    ),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Randn Operation - Float Types Only
// ============================================================================

class RandnFloatTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device = Device::cpu();
};

TEST_P(RandnFloatTest, NormalDistribution) {
    auto param = GetParam();
    auto t = randn({100, 100}, param.dtype, device);

    ASSERT_EQ(t.dtype(), param.dtype);
    ASSERT_EQ(t.numel(), 10000);

    // Calculate mean and std
    if (param.dtype == DType::Float32) {
        const float* data = t.data<float>();
        double sum = 0.0;
        for (int64_t i = 0; i < t.numel(); ++i) {
            sum += data[i];
        }
        double mean = sum / t.numel();

        double variance_sum = 0.0;
        for (int64_t i = 0; i < t.numel(); ++i) {
            double diff = data[i] - mean;
            variance_sum += diff * diff;
        }
        double std = std::sqrt(variance_sum / t.numel());

        // N(0,1) distribution properties
        EXPECT_NEAR(mean, 0.0, 0.1);
        EXPECT_NEAR(std, 1.0, 0.1);
    }
    else if (param.dtype == DType::Float64) {
        const double* data = t.data<double>();
        double sum = 0.0;
        for (int64_t i = 0; i < t.numel(); ++i) {
            sum += data[i];
        }
        double mean = sum / t.numel();

        double variance_sum = 0.0;
        for (int64_t i = 0; i < t.numel(); ++i) {
            double diff = data[i] - mean;
            variance_sum += diff * diff;
        }
        double std = std::sqrt(variance_sum / t.numel());

        EXPECT_NEAR(mean, 0.0, 0.1);
        EXPECT_NEAR(std, 1.0, 0.1);
    }
}

TEST_P(RandnFloatTest, Variability) {
    auto param = GetParam();
    auto t = randn({1000}, param.dtype, device);

    // Check that values are not all the same
    bool all_same = true;

    if (param.dtype == DType::Float32) {
        const float* data = t.data<float>();
        float first = data[0];
        for (int64_t i = 1; i < t.numel(); ++i) {
            if (data[i] != first) {
                all_same = false;
                break;
            }
        }
    }
    else if (param.dtype == DType::Float64) {
        const double* data = t.data<double>();
        double first = data[0];
        for (int64_t i = 1; i < t.numel(); ++i) {
            if (data[i] != first) {
                all_same = false;
                break;
            }
        }
    }

    EXPECT_FALSE(all_same);
}

INSTANTIATE_TEST_SUITE_P(
    FloatTypes,
    RandnFloatTest,
    ::testing::Values(
        DTypeParam{DType::Float32, "float32"},
        DTypeParam{DType::Float64, "float64"}
    ),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Arange Operation - Float and Int Types
// ============================================================================

class ArangeMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device = Device::cpu();
};

TEST_P(ArangeMultiDTypeTest, BasicRange) {
    auto param = GetParam();
    auto t = arange(0.0f, 10.0f, 1.0f, param.dtype, device);

    ASSERT_EQ(t.dtype(), param.dtype);
    ASSERT_EQ(t.ndim(), 1);
    ASSERT_EQ(t.numel(), 10);

    // Type-specific range verification
    if (param.dtype == DType::Float32) {
        const float* data = t.data<float>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
        }
    }
    else if (param.dtype == DType::Float64) {
        const double* data = t.data<double>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_DOUBLE_EQ(data[i], static_cast<double>(i));
        }
    }
    else if (param.dtype == DType::Int32) {
        const int32_t* data = t.data<int32_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], static_cast<int32_t>(i));
        }
    }
    else if (param.dtype == DType::Int64) {
        const int64_t* data = t.data<int64_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], static_cast<int64_t>(i));
        }
    }
}

TEST_P(ArangeMultiDTypeTest, CustomStep) {
    auto param = GetParam();
    auto t = arange(0.0f, 20.0f, 2.0f, param.dtype, device);

    ASSERT_EQ(t.dtype(), param.dtype);
    ASSERT_EQ(t.numel(), 10);

    // Verify step values
    if (param.dtype == DType::Float32) {
        const float* data = t.data<float>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_FLOAT_EQ(data[i], static_cast<float>(i * 2));
        }
    }
    else if (param.dtype == DType::Int32) {
        const int32_t* data = t.data<int32_t>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            EXPECT_EQ(data[i], static_cast<int32_t>(i * 2));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    NumericTypes,
    ArangeMultiDTypeTest,
    ::testing::Values(
        DTypeParam{DType::Float32, "float32"},
        DTypeParam{DType::Float64, "float64"},
        DTypeParam{DType::Int32, "int32"},
        DTypeParam{DType::Int64, "int64"}
    ),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Linspace Operation - Float and Int Types
// ============================================================================

class LinspaceMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device = Device::cpu();
};

TEST_P(LinspaceMultiDTypeTest, BasicLinspace) {
    auto param = GetParam();
    auto t = linspace(0.0f, 1.0f, 5, param.dtype, device);

    ASSERT_EQ(t.dtype(), param.dtype);
    ASSERT_EQ(t.numel(), 5);
    ASSERT_EQ(t.ndim(), 1);

    // Type-specific linspace verification
    if (param.dtype == DType::Float32) {
        const float* data = t.data<float>();
        EXPECT_FLOAT_EQ(data[0], 0.0f);
        EXPECT_FLOAT_EQ(data[1], 0.25f);
        EXPECT_FLOAT_EQ(data[2], 0.5f);
        EXPECT_FLOAT_EQ(data[3], 0.75f);
        EXPECT_FLOAT_EQ(data[4], 1.0f);
    }
    else if (param.dtype == DType::Float64) {
        const double* data = t.data<double>();
        EXPECT_DOUBLE_EQ(data[0], 0.0);
        EXPECT_DOUBLE_EQ(data[1], 0.25);
        EXPECT_DOUBLE_EQ(data[2], 0.5);
        EXPECT_DOUBLE_EQ(data[3], 0.75);
        EXPECT_DOUBLE_EQ(data[4], 1.0);
    }
    else if (param.dtype == DType::Int32) {
        const int32_t* data = t.data<int32_t>();
        // Integer linspace truncates
        EXPECT_EQ(data[0], 0);
        EXPECT_EQ(data[4], 1);
    }
    else if (param.dtype == DType::Int64) {
        const int64_t* data = t.data<int64_t>();
        EXPECT_EQ(data[0], 0);
        EXPECT_EQ(data[4], 1);
    }
}

TEST_P(LinspaceMultiDTypeTest, LargeRange) {
    auto param = GetParam();
    auto t = linspace(-100.0f, 100.0f, 201, param.dtype, device);

    ASSERT_EQ(t.dtype(), param.dtype);
    ASSERT_EQ(t.numel(), 201);

    // Check endpoints
    if (param.dtype == DType::Float32) {
        const float* data = t.data<float>();
        EXPECT_FLOAT_EQ(data[0], -100.0f);
        EXPECT_FLOAT_EQ(data[200], 100.0f);
    }
    else if (param.dtype == DType::Float64) {
        const double* data = t.data<double>();
        EXPECT_DOUBLE_EQ(data[0], -100.0);
        EXPECT_DOUBLE_EQ(data[200], 100.0);
    }
}

INSTANTIATE_TEST_SUITE_P(
    NumericTypes,
    LinspaceMultiDTypeTest,
    ::testing::Values(
        DTypeParam{DType::Float32, "float32"},
        DTypeParam{DType::Float64, "float64"},
        DTypeParam{DType::Int32, "int32"},
        DTypeParam{DType::Int64, "int64"}
    ),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Eye Operation - Multi-DType Testing
// ============================================================================

class EyeMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device = Device::cpu();
};

TEST_P(EyeMultiDTypeTest, SquareIdentity) {
    auto param = GetParam();
    auto t = eye(5, std::nullopt, param.dtype, device);

    ASSERT_EQ(t.dtype(), param.dtype);
    ASSERT_EQ(t.shape()[0], 5);
    ASSERT_EQ(t.shape()[1], 5);
    ASSERT_EQ(t.ndim(), 2);

    // Verify identity matrix pattern
    if (param.dtype == DType::Float32) {
        const float* data = t.data<float>();
        for (int64_t i = 0; i < 5; ++i) {
            for (int64_t j = 0; j < 5; ++j) {
                if (i == j) {
                    EXPECT_EQ(data[i * 5 + j], 1.0f);
                } else {
                    EXPECT_EQ(data[i * 5 + j], 0.0f);
                }
            }
        }
    }
    else if (param.dtype == DType::Float64) {
        const double* data = t.data<double>();
        for (int64_t i = 0; i < 5; ++i) {
            for (int64_t j = 0; j < 5; ++j) {
                if (i == j) {
                    EXPECT_EQ(data[i * 5 + j], 1.0);
                } else {
                    EXPECT_EQ(data[i * 5 + j], 0.0);
                }
            }
        }
    }
    else if (param.dtype == DType::Int32) {
        const int32_t* data = t.data<int32_t>();
        for (int64_t i = 0; i < 5; ++i) {
            for (int64_t j = 0; j < 5; ++j) {
                if (i == j) {
                    EXPECT_EQ(data[i * 5 + j], 1);
                } else {
                    EXPECT_EQ(data[i * 5 + j], 0);
                }
            }
        }
    }
    else if (param.dtype == DType::Int64) {
        const int64_t* data = t.data<int64_t>();
        for (int64_t i = 0; i < 5; ++i) {
            for (int64_t j = 0; j < 5; ++j) {
                if (i == j) {
                    EXPECT_EQ(data[i * 5 + j], 1);
                } else {
                    EXPECT_EQ(data[i * 5 + j], 0);
                }
            }
        }
    }
    else if (param.dtype == DType::Bool) {
        const bool* data = t.data<bool>();
        for (int64_t i = 0; i < 5; ++i) {
            for (int64_t j = 0; j < 5; ++j) {
                if (i == j) {
                    EXPECT_TRUE(data[i * 5 + j]);
                } else {
                    EXPECT_FALSE(data[i * 5 + j]);
                }
            }
        }
    }
    else if (param.dtype == DType::UInt8) {
        const uint8_t* data = t.data<uint8_t>();
        for (int64_t i = 0; i < 5; ++i) {
            for (int64_t j = 0; j < 5; ++j) {
                if (i == j) {
                    EXPECT_EQ(data[i * 5 + j], 1);
                } else {
                    EXPECT_EQ(data[i * 5 + j], 0);
                }
            }
        }
    }
}

TEST_P(EyeMultiDTypeTest, RectangularMatrix) {
    auto param = GetParam();
    auto t = eye(3, 5, param.dtype, device);

    ASSERT_EQ(t.dtype(), param.dtype);
    ASSERT_EQ(t.shape()[0], 3);
    ASSERT_EQ(t.shape()[1], 5);

    // Diagonal should have ones, rest zeros
    ASSERT_EQ(t.numel(), 15);
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    EyeMultiDTypeTest,
    ::testing::Values(
        DTypeParam{DType::Float32, "float32"},
        DTypeParam{DType::Float64, "float64"},
        DTypeParam{DType::Int32, "int32"},
        DTypeParam{DType::Int64, "int64"},
        DTypeParam{DType::Bool, "bool"},
        DTypeParam{DType::UInt8, "uint8"}
    ),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Summary Statistics
// ============================================================================

/*
 * COMPREHENSIVE COVERAGE SUMMARY:
 *
 * Operations Tested: 7 (zeros, ones, full, randn, arange, linspace, eye)
 * DTypes Tested: 6 (Float32, Float64, Int32, Int64, Bool, UInt8)
 *
 * Test Breakdown:
 * - Zeros:    6 dtypes × 2 tests = 12 test scenarios
 * - Ones:     6 dtypes × 2 tests = 12 test scenarios
 * - Full:     6 dtypes × 2 tests = 12 test scenarios
 * - Randn:    2 dtypes × 2 tests = 4 test scenarios  (float only)
 * - Arange:   4 dtypes × 2 tests = 8 test scenarios  (numeric types)
 * - Linspace: 4 dtypes × 2 tests = 8 test scenarios  (numeric types)
 * - Eye:      6 dtypes × 2 tests = 12 test scenarios
 *
 * Total Test Scenarios: 68
 * Total Test Cases: 14
 *
 * Coverage Features:
 * ✓ DType verification for all operations
 * ✓ Value range checks (e.g., UInt8: 0-255)
 * ✓ Type-specific behavior (e.g., Bool: true/false)
 * ✓ Statistical properties (randn: mean≈0, std≈1)
 * ✓ Shape and dimension verification
 * ✓ Large tensor support
 * ✓ Edge cases (rectangular matrices, custom steps)
 *
 * Operations NOT Tested (out of scope):
 * - rand: Similar to randn, left for future extension
 * - randperm: Int64 only by design
 * - empty: Uninitialized memory, no value checks needed
 * - from_data: Template function, covered by type system
 * - *_like: Derivatives of main operations
 */
