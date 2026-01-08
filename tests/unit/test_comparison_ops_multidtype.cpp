#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <limits>
#include <cmath>

using namespace tenzor;

/**
 * @file test_comparison_ops_multidtype.cpp
 * @brief Dtype-parameterized tests for comparison operations
 *
 * CRITICAL DESIGN: Comparison operations ALWAYS return Bool dtype,
 * but accept multiple input dtypes (Float32, Float64, Int32, Int64).
 *
 * Tests verify:
 * - Input dtypes: Float32, Float64, Int32, Int64
 * - Output dtype: Bool (always)
 * - Edge cases: NaN for floats, max/min for integers
 */

// ============================================================================
// Nested Parameterization (Backend + Input DType)
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

class ComparisonOpsMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType input_dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        input_dtype = param.dtype;

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

    // Helper to create tensor with specific value based on dtype
    template<typename T>
    Tensor createTensorWithValue(const std::vector<int64_t>& shape, T value) {
        auto tensor_cpu = zeros(shape, input_dtype, Device::cpu());

        if (input_dtype == DType::Float32) {
            auto data = tensor_cpu.data<float>();
            for (int64_t i = 0; i < tensor_cpu.numel(); ++i) {
                data[i] = static_cast<float>(value);
            }
        }
        else if (input_dtype == DType::Float64) {
            auto data = tensor_cpu.data<double>();
            for (int64_t i = 0; i < tensor_cpu.numel(); ++i) {
                data[i] = static_cast<double>(value);
            }
        }
        else if (input_dtype == DType::Int32) {
            auto data = tensor_cpu.data<int32_t>();
            for (int64_t i = 0; i < tensor_cpu.numel(); ++i) {
                data[i] = static_cast<int32_t>(value);
            }
        }
        else if (input_dtype == DType::Int64) {
            auto data = tensor_cpu.data<int64_t>();
            for (int64_t i = 0; i < tensor_cpu.numel(); ++i) {
                data[i] = static_cast<int64_t>(value);
            }
        }

        return (device.type == Device::Type::CPU) ? tensor_cpu : tensor_cpu.to(device);
    }

    // Helper to verify all elements are true
    void verifyAllTrue(const Tensor& result, const std::string& op_name) {
        ASSERT_EQ(result.dtype(), DType::Bool) << "Output must be Bool for " << op_name;

        auto result_cpu = result.to(Device::cpu());
        const bool* data = result_cpu.data<bool>();

        for (int64_t i = 0; i < result_cpu.numel(); ++i) {
            EXPECT_TRUE(data[i])
                << "Failed for " << op_name
                << " on backend: " << GetParam().backend_name
                << " with dtype: " << GetParam().dtype_name
                << " at index " << i;
        }
    }

    // Helper to verify all elements are false
    void verifyAllFalse(const Tensor& result, const std::string& op_name) {
        ASSERT_EQ(result.dtype(), DType::Bool) << "Output must be Bool for " << op_name;

        auto result_cpu = result.to(Device::cpu());
        const bool* data = result_cpu.data<bool>();

        for (int64_t i = 0; i < result_cpu.numel(); ++i) {
            EXPECT_FALSE(data[i])
                << "Failed for " << op_name
                << " on backend: " << GetParam().backend_name
                << " with dtype: " << GetParam().dtype_name
                << " at index " << i;
        }
    }
};

// ============================================================================
// Basic Comparison Tests
// ============================================================================

TEST_P(ComparisonOpsMultiDTypeTest, EqualOperator_SameValues) {
    auto a = createTensorWithValue({3, 3}, 5);
    auto b = createTensorWithValue({3, 3}, 5);

    auto result = eq(a, b);
    verifyAllTrue(result, "eq");
}

TEST_P(ComparisonOpsMultiDTypeTest, EqualOperator_DifferentValues) {
    auto a = createTensorWithValue({3, 3}, 5);
    auto b = createTensorWithValue({3, 3}, 3);

    auto result = eq(a, b);
    verifyAllFalse(result, "eq");
}

TEST_P(ComparisonOpsMultiDTypeTest, NotEqualOperator) {
    auto a = createTensorWithValue({3, 3}, 5);
    auto b = createTensorWithValue({3, 3}, 3);

    auto result = ne(a, b);
    verifyAllTrue(result, "ne");
}

TEST_P(ComparisonOpsMultiDTypeTest, NotEqualOperator_SameValues) {
    auto a = createTensorWithValue({3, 3}, 5);
    auto b = createTensorWithValue({3, 3}, 5);

    auto result = ne(a, b);
    verifyAllFalse(result, "ne");
}

TEST_P(ComparisonOpsMultiDTypeTest, LessThanOperator) {
    auto a = createTensorWithValue({3, 3}, 3);
    auto b = createTensorWithValue({3, 3}, 5);

    auto result = lt(a, b);
    verifyAllTrue(result, "lt");
}

TEST_P(ComparisonOpsMultiDTypeTest, LessThanOperator_False) {
    auto a = createTensorWithValue({3, 3}, 7);
    auto b = createTensorWithValue({3, 3}, 5);

    auto result = lt(a, b);
    verifyAllFalse(result, "lt");
}

TEST_P(ComparisonOpsMultiDTypeTest, LessEqualOperator_Equal) {
    auto a = createTensorWithValue({2, 2}, 5);
    auto b = createTensorWithValue({2, 2}, 5);

    auto result = le(a, b);
    verifyAllTrue(result, "le");
}

TEST_P(ComparisonOpsMultiDTypeTest, LessEqualOperator_Less) {
    auto a = createTensorWithValue({2, 2}, 3);
    auto b = createTensorWithValue({2, 2}, 5);

    auto result = le(a, b);
    verifyAllTrue(result, "le");
}

TEST_P(ComparisonOpsMultiDTypeTest, GreaterThanOperator) {
    auto a = createTensorWithValue({3, 3}, 7);
    auto b = createTensorWithValue({3, 3}, 5);

    auto result = gt(a, b);
    verifyAllTrue(result, "gt");
}

TEST_P(ComparisonOpsMultiDTypeTest, GreaterThanOperator_False) {
    auto a = createTensorWithValue({3, 3}, 3);
    auto b = createTensorWithValue({3, 3}, 5);

    auto result = gt(a, b);
    verifyAllFalse(result, "gt");
}

TEST_P(ComparisonOpsMultiDTypeTest, GreaterEqualOperator_Equal) {
    auto a = createTensorWithValue({2, 2}, 5);
    auto b = createTensorWithValue({2, 2}, 5);

    auto result = ge(a, b);
    verifyAllTrue(result, "ge");
}

TEST_P(ComparisonOpsMultiDTypeTest, GreaterEqualOperator_Greater) {
    auto a = createTensorWithValue({2, 2}, 7);
    auto b = createTensorWithValue({2, 2}, 5);

    auto result = ge(a, b);
    verifyAllTrue(result, "ge");
}

// ============================================================================
// Edge Case Tests - Float Types
// ============================================================================

class ComparisonOpsFloatEdgeCases : public ComparisonOpsMultiDTypeTest {};

TEST_P(ComparisonOpsFloatEdgeCases, NaN_NotEqualToItself) {
    auto param = GetParam();

    // Only test for float types
    if (input_dtype != DType::Float32 && input_dtype != DType::Float64) {
        GTEST_SKIP() << "NaN test only applicable to floating point types";
    }

    auto a_cpu = zeros({10}, input_dtype, Device::cpu());

    if (input_dtype == DType::Float32) {
        auto data = a_cpu.data<float>();
        for (int64_t i = 0; i < 10; ++i) {
            data[i] = std::numeric_limits<float>::quiet_NaN();
        }
    }
    else if (input_dtype == DType::Float64) {
        auto data = a_cpu.data<double>();
        for (int64_t i = 0; i < 10; ++i) {
            data[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);

    // NaN != NaN by IEEE 754 standard
    auto result = eq(a, a);
    verifyAllFalse(result, "eq_nan");
}

TEST_P(ComparisonOpsFloatEdgeCases, NaN_ComparisonAlwaysFalse) {
    auto param = GetParam();

    // Only test for float types
    if (input_dtype != DType::Float32 && input_dtype != DType::Float64) {
        GTEST_SKIP() << "NaN test only applicable to floating point types";
    }

    auto nan_tensor_cpu = zeros({10}, input_dtype, Device::cpu());
    auto normal_tensor = createTensorWithValue({10}, 5);

    if (input_dtype == DType::Float32) {
        auto data = nan_tensor_cpu.data<float>();
        for (int64_t i = 0; i < 10; ++i) {
            data[i] = std::numeric_limits<float>::quiet_NaN();
        }
    }
    else if (input_dtype == DType::Float64) {
        auto data = nan_tensor_cpu.data<double>();
        for (int64_t i = 0; i < 10; ++i) {
            data[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    auto nan_tensor = (device.type == Device::Type::CPU) ? nan_tensor_cpu : nan_tensor_cpu.to(device);

    // All comparisons with NaN should be false
    auto result_lt = lt(nan_tensor, normal_tensor);
    auto result_gt = gt(nan_tensor, normal_tensor);
    auto result_le = le(nan_tensor, normal_tensor);
    auto result_ge = ge(nan_tensor, normal_tensor);

    verifyAllFalse(result_lt, "lt_nan");
    verifyAllFalse(result_gt, "gt_nan");
    verifyAllFalse(result_le, "le_nan");
    verifyAllFalse(result_ge, "ge_nan");
}

TEST_P(ComparisonOpsFloatEdgeCases, Infinity_Comparisons) {
    auto param = GetParam();

    // Only test for float types
    if (input_dtype != DType::Float32 && input_dtype != DType::Float64) {
        GTEST_SKIP() << "Infinity test only applicable to floating point types";
    }

    auto pos_inf_cpu = zeros({10}, input_dtype, Device::cpu());
    auto neg_inf_cpu = zeros({10}, input_dtype, Device::cpu());
    auto normal = createTensorWithValue({10}, 5);

    if (input_dtype == DType::Float32) {
        auto pos_data = pos_inf_cpu.data<float>();
        auto neg_data = neg_inf_cpu.data<float>();
        for (int64_t i = 0; i < 10; ++i) {
            pos_data[i] = std::numeric_limits<float>::infinity();
            neg_data[i] = -std::numeric_limits<float>::infinity();
        }
    }
    else if (input_dtype == DType::Float64) {
        auto pos_data = pos_inf_cpu.data<double>();
        auto neg_data = neg_inf_cpu.data<double>();
        for (int64_t i = 0; i < 10; ++i) {
            pos_data[i] = std::numeric_limits<double>::infinity();
            neg_data[i] = -std::numeric_limits<double>::infinity();
        }
    }

    auto pos_inf = (device.type == Device::Type::CPU) ? pos_inf_cpu : pos_inf_cpu.to(device);
    auto neg_inf = (device.type == Device::Type::CPU) ? neg_inf_cpu : neg_inf_cpu.to(device);

    // +inf > any finite number
    auto result_pos_gt = gt(pos_inf, normal);
    verifyAllTrue(result_pos_gt, "pos_inf_gt_normal");

    // -inf < any finite number
    auto result_neg_lt = lt(neg_inf, normal);
    verifyAllTrue(result_neg_lt, "neg_inf_lt_normal");

    // -inf < +inf
    auto result_inf_lt = lt(neg_inf, pos_inf);
    verifyAllTrue(result_inf_lt, "neg_inf_lt_pos_inf");
}

// ============================================================================
// Edge Case Tests - Integer Types
// ============================================================================

class ComparisonOpsIntEdgeCases : public ComparisonOpsMultiDTypeTest {};

TEST_P(ComparisonOpsIntEdgeCases, MaxValue_Comparisons) {
    auto param = GetParam();

    // Only test for integer types
    if (input_dtype != DType::Int32 && input_dtype != DType::Int64) {
        GTEST_SKIP() << "Max value test only applicable to integer types";
    }

    auto max_val_cpu = zeros({10}, input_dtype, Device::cpu());
    auto normal = createTensorWithValue({10}, 5);

    if (input_dtype == DType::Int32) {
        auto data = max_val_cpu.data<int32_t>();
        for (int64_t i = 0; i < 10; ++i) {
            data[i] = std::numeric_limits<int32_t>::max();
        }
    }
    else if (input_dtype == DType::Int64) {
        auto data = max_val_cpu.data<int64_t>();
        for (int64_t i = 0; i < 10; ++i) {
            data[i] = std::numeric_limits<int64_t>::max();
        }
    }

    auto max_val = (device.type == Device::Type::CPU) ? max_val_cpu : max_val_cpu.to(device);

    // max_value > normal value
    auto result_gt = gt(max_val, normal);
    verifyAllTrue(result_gt, "max_gt_normal");

    // max_value >= max_value
    auto result_ge = ge(max_val, max_val);
    verifyAllTrue(result_ge, "max_ge_max");
}

TEST_P(ComparisonOpsIntEdgeCases, MinValue_Comparisons) {
    auto param = GetParam();

    // Only test for integer types
    if (input_dtype != DType::Int32 && input_dtype != DType::Int64) {
        GTEST_SKIP() << "Min value test only applicable to integer types";
    }

    auto min_val_cpu = zeros({10}, input_dtype, Device::cpu());
    auto normal = createTensorWithValue({10}, 5);

    if (input_dtype == DType::Int32) {
        auto data = min_val_cpu.data<int32_t>();
        for (int64_t i = 0; i < 10; ++i) {
            data[i] = std::numeric_limits<int32_t>::min();
        }
    }
    else if (input_dtype == DType::Int64) {
        auto data = min_val_cpu.data<int64_t>();
        for (int64_t i = 0; i < 10; ++i) {
            data[i] = std::numeric_limits<int64_t>::min();
        }
    }

    auto min_val = (device.type == Device::Type::CPU) ? min_val_cpu : min_val_cpu.to(device);

    // min_value < normal value
    auto result_lt = lt(min_val, normal);
    verifyAllTrue(result_lt, "min_lt_normal");

    // min_value <= min_value
    auto result_le = le(min_val, min_val);
    verifyAllTrue(result_le, "min_le_min");
}

TEST_P(ComparisonOpsIntEdgeCases, Zero_Comparisons) {
    auto param = GetParam();

    // Only test for integer types
    if (input_dtype != DType::Int32 && input_dtype != DType::Int64) {
        GTEST_SKIP() << "Zero test only applicable to integer types";
    }

    auto zero = createTensorWithValue({10}, 0);
    auto positive = createTensorWithValue({10}, 5);
    auto negative = createTensorWithValue({10}, -5);

    // 0 > -5
    auto result_gt_neg = gt(zero, negative);
    verifyAllTrue(result_gt_neg, "zero_gt_negative");

    // 0 < 5
    auto result_lt_pos = lt(zero, positive);
    verifyAllTrue(result_lt_pos, "zero_lt_positive");

    // 0 == 0
    auto result_eq = eq(zero, zero);
    verifyAllTrue(result_eq, "zero_eq_zero");
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateComparisonTestCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "rocm"};

    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
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
    ComparisonOpsMultiDTypeTest,
    ::testing::ValuesIn(GenerateComparisonTestCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    FloatEdgeCases,
    ComparisonOpsFloatEdgeCases,
    ::testing::ValuesIn(GenerateComparisonTestCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    IntEdgeCases,
    ComparisonOpsIntEdgeCases,
    ::testing::ValuesIn(GenerateComparisonTestCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE SUMMARY:
 *
 * Basic Comparison Tests:
 * - 12 tests × 4 backends × 4 dtypes = 192 test scenarios
 *
 * Float Edge Cases:
 * - 3 tests × 4 backends × 2 float dtypes = 24 test scenarios
 *
 * Integer Edge Cases:
 * - 3 tests × 4 backends × 2 int dtypes = 24 test scenarios
 *
 * Total: 240 test scenarios covering:
 * - All comparison operations (eq, ne, lt, le, gt, ge)
 * - All supported dtypes (Float32, Float64, Int32, Int64)
 * - All backends (CPU, CUDA, Vulkan, OneAPI)
 * - Edge cases (NaN, infinity, max/min values, zero)
 *
 * CRITICAL VERIFICATION:
 * - Output dtype is ALWAYS Bool regardless of input dtype
 * - NaN handling follows IEEE 754 standard
 * - Integer boundary values handled correctly
 */
