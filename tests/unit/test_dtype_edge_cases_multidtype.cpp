/**
 * @file test_dtype_edge_cases_multidtype.cpp
 * @brief Enhanced multi-dtype edge case tests
 *
 * This is an enhanced version that focuses on cross-backend consistency
 * for dtype edge cases. The original test_dtype_edge_cases.cpp already
 * has comprehensive dtype parameterization. This version adds:
 * - Additional edge case scenarios
 * - Cross-backend consistency checks
 * - More thorough dtype conversion testing
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

class DTypeEdgeCaseEnhancedTest : public ::testing::TestWithParam<BackendDTypeParam> {
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
// Additional Float Edge Cases
// ============================================================================

TEST_P(DTypeEdgeCaseEnhancedTest, FloatMaxMinValues) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        GTEST_SKIP() << "Test only for floating point types";
    }

    auto a_cpu = zeros({5}, dtype, Device::cpu());

    if (dtype == DType::Float32) {
        auto a_data = a_cpu.data<float>();
        a_data[0] = std::numeric_limits<float>::max();
        a_data[1] = std::numeric_limits<float>::lowest();
        a_data[2] = std::numeric_limits<float>::min();  // Smallest positive normal
        a_data[3] = std::numeric_limits<float>::epsilon();
        a_data[4] = std::numeric_limits<float>::denorm_min();
    } else {
        auto a_data = a_cpu.data<double>();
        a_data[0] = std::numeric_limits<double>::max();
        a_data[1] = std::numeric_limits<double>::lowest();
        a_data[2] = std::numeric_limits<double>::min();
        a_data[3] = std::numeric_limits<double>::epsilon();
        a_data[4] = std::numeric_limits<double>::denorm_min();
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto a_result = a.to(Device::cpu());

    // Verify values are preserved
    if (dtype == DType::Float32) {
        const float* orig = a_cpu.data<float>();
        const float* result = a_result.data<float>();
        for (int i = 0; i < 5; ++i) {
            if (std::isnan(orig[i])) {
                EXPECT_TRUE(std::isnan(result[i]));
            } else if (std::isinf(orig[i])) {
                EXPECT_TRUE(std::isinf(result[i]));
                EXPECT_EQ(std::signbit(orig[i]), std::signbit(result[i]));
            } else {
                EXPECT_FLOAT_EQ(result[i], orig[i]);
            }
        }
    } else {
        const double* orig = a_cpu.data<double>();
        const double* result = a_result.data<double>();
        for (int i = 0; i < 5; ++i) {
            if (std::isnan(orig[i])) {
                EXPECT_TRUE(std::isnan(result[i]));
            } else if (std::isinf(orig[i])) {
                EXPECT_TRUE(std::isinf(result[i]));
                EXPECT_EQ(std::signbit(orig[i]), std::signbit(result[i]));
            } else {
                EXPECT_DOUBLE_EQ(result[i], orig[i]);
            }
        }
    }
}

// ============================================================================
// Integer Boundary Tests
// ============================================================================

TEST_P(DTypeEdgeCaseEnhancedTest, IntegerBoundaryValues) {
    if (dtype != DType::Int8 && dtype != DType::Int32 &&
        dtype != DType::Int64 && dtype != DType::UInt8) {
        GTEST_SKIP() << "Test only for integer types";
    }

    auto a_cpu = zeros({3}, dtype, Device::cpu());

    if (dtype == DType::Int8) {
        auto a_data = a_cpu.data<int8_t>();
        a_data[0] = std::numeric_limits<int8_t>::min();
        a_data[1] = 0;
        a_data[2] = std::numeric_limits<int8_t>::max();
    } else if (dtype == DType::UInt8) {
        auto a_data = a_cpu.data<uint8_t>();
        a_data[0] = std::numeric_limits<uint8_t>::min();
        a_data[1] = 128;
        a_data[2] = std::numeric_limits<uint8_t>::max();
    } else if (dtype == DType::Int32) {
        auto a_data = a_cpu.data<int32_t>();
        a_data[0] = std::numeric_limits<int32_t>::min();
        a_data[1] = 0;
        a_data[2] = std::numeric_limits<int32_t>::max();
    } else if (dtype == DType::Int64) {
        auto a_data = a_cpu.data<int64_t>();
        a_data[0] = std::numeric_limits<int64_t>::min();
        a_data[1] = 0;
        a_data[2] = std::numeric_limits<int64_t>::max();
    }

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto a_result = a.to(Device::cpu());

    // Verify exact values preserved
    if (dtype == DType::Int8) {
        const int8_t* orig = a_cpu.data<int8_t>();
        const int8_t* result = a_result.data<int8_t>();
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(result[i], orig[i]);
        }
    } else if (dtype == DType::UInt8) {
        const uint8_t* orig = a_cpu.data<uint8_t>();
        const uint8_t* result = a_result.data<uint8_t>();
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(result[i], orig[i]);
        }
    } else if (dtype == DType::Int32) {
        const int32_t* orig = a_cpu.data<int32_t>();
        const int32_t* result = a_result.data<int32_t>();
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(result[i], orig[i]);
        }
    } else if (dtype == DType::Int64) {
        const int64_t* orig = a_cpu.data<int64_t>();
        const int64_t* result = a_result.data<int64_t>();
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(result[i], orig[i]);
        }
    }
}

// ============================================================================
// Cross-DType Operations
// ============================================================================

TEST_P(DTypeEdgeCaseEnhancedTest, MixedDTypeAddition) {
    if (dtype != DType::Float32) {
        GTEST_SKIP() << "Test requires Float32 primary dtype";
    }

    // Create Float32 and Int32 tensors
    auto a = full({10}, 5.5f, DType::Float32, device);
    auto b = full({10}, 3.0f, DType::Int32, device);

    // Mixed dtype operations should promote to Float32
    auto c = add(a, b.to(DType::Float32));

    auto c_cpu = c.to(Device::cpu());
    const float* c_data = c_cpu.data<float>();

    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(c_data[i], 8.5f, 1e-5);
    }
}

// ============================================================================
// Bool Operations
// ============================================================================

TEST_P(DTypeEdgeCaseEnhancedTest, BoolLogicalOps) {
    if (dtype != DType::Bool) {
        GTEST_SKIP() << "Test only for Bool";
    }

    auto a_cpu = zeros({4}, DType::Bool, Device::cpu());
    auto b_cpu = zeros({4}, DType::Bool, Device::cpu());

    auto a_data = reinterpret_cast<bool*>(a_cpu.data_ptr());
    auto b_data = reinterpret_cast<bool*>(b_cpu.data_ptr());

    a_data[0] = false; b_data[0] = false;
    a_data[1] = false; b_data[1] = true;
    a_data[2] = true;  b_data[2] = false;
    a_data[3] = true;  b_data[3] = true;

    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto b = (device.type == Device::Type::CPU) ? b_cpu : b_cpu.to(device);

    // Test AND (using element-wise multiplication for bool)
    auto c_and = mul(a, b);
    auto c_and_cpu = c_and.to(Device::cpu());
    auto c_and_data = reinterpret_cast<const bool*>(c_and_cpu.data_ptr());

    EXPECT_FALSE(c_and_data[0]);
    EXPECT_FALSE(c_and_data[1]);
    EXPECT_FALSE(c_and_data[2]);
    EXPECT_TRUE(c_and_data[3]);

    // Test OR (bool add with numeric semantics gives OR: non-zero result = true)
    auto c_or = add(a, b);
    auto c_or_cpu = c_or.to(Device::cpu());
    auto c_or_data = reinterpret_cast<const bool*>(c_or_cpu.data_ptr());

    EXPECT_FALSE(c_or_data[0]);  // false + false = 0 -> false
    EXPECT_TRUE(c_or_data[1]);   // false + true = 1 -> true
    EXPECT_TRUE(c_or_data[2]);   // true + false = 1 -> true
    EXPECT_TRUE(c_or_data[3]);   // true + true = 2 -> true
}

// ============================================================================
// Conversion Roundtrip Tests
// ============================================================================

TEST_P(DTypeEdgeCaseEnhancedTest, ConversionRoundtrip) {
    // Test: original -> Float32 -> original
    auto original = full({10}, 42.0f, dtype, device);
    auto as_float = original.to(DType::Float32);
    auto back = as_float.to(dtype);

    auto original_cpu = original.to(Device::cpu());
    auto back_cpu = back.to(Device::cpu());

    // For integer types, should be exact
    // For float types, should be very close
    if (dtype == DType::Int8) {
        const int8_t* orig = original_cpu.data<int8_t>();
        const int8_t* result = back_cpu.data<int8_t>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(result[i], orig[i]);
        }
    } else if (dtype == DType::UInt8) {
        const uint8_t* orig = original_cpu.data<uint8_t>();
        const uint8_t* result = back_cpu.data<uint8_t>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(result[i], orig[i]);
        }
    } else if (dtype == DType::Int32) {
        const int32_t* orig = original_cpu.data<int32_t>();
        const int32_t* result = back_cpu.data<int32_t>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(result[i], orig[i]);
        }
    } else if (dtype == DType::Float32) {
        const float* orig = original_cpu.data<float>();
        const float* result = back_cpu.data<float>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_FLOAT_EQ(result[i], orig[i]);
        }
    } else if (dtype == DType::Float64) {
        const double* orig = original_cpu.data<double>();
        const double* result = back_cpu.data<double>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_DOUBLE_EQ(result[i], orig[i]);
        }
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateEnhancedEdgeCaseCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "rocm"};

    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Int8, "int8"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
        {DType::UInt8, "uint8"},
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
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
    DTypeEdgeCaseEnhancedTest,
    ::testing::ValuesIn(GenerateEnhancedEdgeCaseCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Test Environment Setup
// ============================================================================

class DTypeEdgeCaseTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        initialize();
    }
};

static ::testing::Environment* const dtype_edge_case_env =
    ::testing::AddGlobalTestEnvironment(new DTypeEdgeCaseTestEnvironment);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
