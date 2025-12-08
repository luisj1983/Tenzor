#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>

using namespace tenzor;

/**
 * @file test_multi_param_example.cpp
 * @brief Example: Testing with BOTH backend AND dtype parameterization
 *
 * CRITICAL: This shows how to achieve COMPLETE coverage by testing:
 * - All backends (CPU, CUDA, Vulkan, OneAPI, ROCm)
 * - All relevant dtypes (Float32, Float64, Int32, etc.)
 *
 * Single test → 5 backends × N dtypes = massive coverage
 */

// ============================================================================
// Method 1: Nested Parameterization (Backend + DType)
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

class MathOpsMultiParamTest : public ::testing::TestWithParam<BackendDTypeParam> {
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

// Test addition across ALL backends AND dtypes
TEST_P(MathOpsMultiParamTest, AddBasic) {
    auto param = GetParam();

    // Create tensors with the parameterized dtype
    auto a = ones({100, 100}, dtype, device);
    auto b = ones({100, 100}, dtype, device);

    auto c = add(a, b);

    // Verify
    auto c_cpu = c.to(Device::cpu());

    // Type-specific verification
    if (dtype == DType::Float32) {
        const float* data = c_cpu.data<float>();
        for (int i = 0; i < 100; ++i) {
            EXPECT_FLOAT_EQ(data[i], 2.0f);
        }
    }
    else if (dtype == DType::Float64) {
        const double* data = c_cpu.data<double>();
        for (int i = 0; i < 100; ++i) {
            EXPECT_DOUBLE_EQ(data[i], 2.0);
        }
    }
    else if (dtype == DType::Int32) {
        const int32_t* data = c_cpu.data<int32_t>();
        for (int i = 0; i < 100; ++i) {
            EXPECT_EQ(data[i], 2);
        }
    }
}

// Generate all combinations of backend × dtype
std::vector<BackendDTypeParam> GenerateBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "adaptivecpp"};

    // Define dtypes to test per operation type
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
        // Add more as needed
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
    MathOpsMultiParamTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Method 2: DType-Specific Operation Tests
// ============================================================================

class DTypeSpecificOpsTest : public tenzor::testing::BackendTest {};

// Test operations that ONLY make sense for specific dtypes
TEST_P(DTypeSpecificOpsTest, BooleanLogic) {
    // Bool dtype for logical operations
    auto a_cpu = zeros({100}, DType::Bool, Device::cpu());
    auto b_cpu = ones({100}, DType::Bool, Device::cpu());

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = (device.type == Device::Type::CPU) ? b_cpu : b_cpu.to(device);

    // Logical OR: false | true = true
    auto c = logical_or(a, b);

    auto c_cpu = c.to(Device::cpu());
    const bool* data = c_cpu.data<bool>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

TEST_P(DTypeSpecificOpsTest, IntegerDivision) {
    // Int32 for integer division behavior
    auto a_cpu = ones({100}, DType::Int32, Device::cpu());
    auto a_data = a_cpu.data<int32_t>();

    for (int i = 0; i < 100; ++i) {
        a_data[i] = 7;
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = ones({100}, DType::Int32, device) * 3;

    auto c = div(a, b);  // 7 / 3 = 2 (integer division)

    auto c_cpu = c.to(Device::cpu());
    const int32_t* data = c_cpu.data<int32_t>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(data[i], 2);  // NOT 2.333...
    }
}

TEST_P(DTypeSpecificOpsTest, Float16Precision) {
    // Float16 for mixed precision training
    if (!device.supports_dtype(DType::Float16)) {
        GTEST_SKIP() << "Float16 not supported on " << device.to_string();
    }

    auto a = ones({100, 100}, DType::Float16, device);
    auto b = ones({100, 100}, DType::Float16, device);

    auto c = add(a, b);

    // Verify it's still Float16
    EXPECT_EQ(c.dtype(), DType::Float16);

    // Convert to Float32 for verification
    auto c_f32 = c.to(DType::Float32);
    auto c_cpu = c_f32.to(Device::cpu());
    const float* data = c_cpu.data<float>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(data[i], 2.0f, 1e-3f);  // Float16 has less precision
    }
}

TEST_P(DTypeSpecificOpsTest, IntegerOverflow) {
    // Test Int8 overflow behavior
    auto a_cpu = zeros({10}, DType::Int8, Device::cpu());
    auto a_data = a_cpu.data<int8_t>();

    for (int i = 0; i < 10; ++i) {
        a_data[i] = 127;  // Max Int8 value
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = ones({10}, DType::Int8, device);

    auto c = add(a, b);  // 127 + 1 = overflow

    auto c_cpu = c.to(Device::cpu());
    const int8_t* data = c_cpu.data<int8_t>();

    // Document overflow behavior (may be -128 or saturate to 127)
    for (int i = 0; i < 10; ++i) {
        // Platform-dependent behavior
        EXPECT_TRUE(data[i] == -128 || data[i] == 127);
    }
}

INSTANTIATE_BACKEND_TESTS(DTypeSpecificOpsTest);

// ============================================================================
// Method 3: Type Conversion Tests
// ============================================================================

class TypeConversionTest : public tenzor::testing::BackendTest {};

TEST_P(TypeConversionTest, Float32ToFloat64) {
    auto a_f32 = ones({100, 100}, DType::Float32, device);
    auto a_f64 = a_f32.to(DType::Float64);

    EXPECT_EQ(a_f64.dtype(), DType::Float64);

    auto a_cpu = a_f64.to(Device::cpu());
    const double* data = a_cpu.data<double>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(data[i], 1.0);
    }
}

TEST_P(TypeConversionTest, Float32ToInt32) {
    auto a_cpu = zeros({100}, DType::Float32, Device::cpu());
    auto a_data = a_cpu.data<float>();

    for (int i = 0; i < 100; ++i) {
        a_data[i] = 3.7f;
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto a_int = a.to(DType::Int32);

    auto int_cpu = a_int.to(Device::cpu());
    const int32_t* data = int_cpu.data<int32_t>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(data[i], 3);  // Truncation, not rounding
    }
}

TEST_P(TypeConversionTest, Int32ToFloat32) {
    auto a = ones({100, 100}, DType::Int32, device) * 5;
    auto a_float = a.to(DType::Float32);

    auto a_cpu = a_float.to(Device::cpu());
    const float* data = a_cpu.data<float>();

    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(data[i], 5.0f);
    }
}

INSTANTIATE_BACKEND_TESTS(TypeConversionTest);

/*
 * COVERAGE IMPACT:
 *
 * Method 1 (Multi-param):
 * - 1 test × 4 backends × 4 dtypes = 16 test scenarios
 *
 * Method 2 (DType-specific):
 * - 4 tests × 4 backends = 16 test scenarios
 *
 * Method 3 (Type conversion):
 * - 3 tests × 4 backends = 12 test scenarios
 *
 * Total: 44 test scenarios from just 8 test cases!
 *
 * FULL COVERAGE would be:
 * - 100 operations × 5 backends × 8 common dtypes = 4,000 test scenarios
 *
 * RECOMMENDATION:
 * - High priority ops: Test with Float32, Float64, Int32, Bool
 * - Medium priority: Add Float16, Int64
 * - Low priority: Add remaining dtypes as needed
 */
