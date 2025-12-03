#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

/**
 * @file test_autograd_multidtype.cpp
 * @brief Multi-dtype parameterized tests for autograd operations
 *
 * This file refactors test_autograd.cpp to test autograd operations across multiple data types:
 * - Float32 (single precision)
 * - Float64 (double precision)
 *
 * Autograd primarily works with floating-point types for gradient computation.
 *
 * Coverage improvement: All tests now run with both Float32 and Float64
 */

// ============================================================================
// Multi-DType Test Fixture
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

class AutogradMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
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

    static std::string dtype_to_string(DType dtype) {
        switch(dtype) {
            case DType::Float32: return "Float32";
            case DType::Float64: return "Float64";
            default: return "Unknown";
        }
    }

    // Helper to verify gradient data based on dtype
    template<typename T>
    void VerifyGradient(const Tensor& t, T expected_value, size_t count) {
        auto t_cpu = t.to(Device::cpu());
        const T* data = t_cpu.data<T>();
        for (size_t i = 0; i < count; i++) {
            if constexpr (std::is_same_v<T, float>) {
                EXPECT_FLOAT_EQ(data[i], expected_value)
                    << "Failed at index " << i << " on " << device.to_string();
            } else if constexpr (std::is_same_v<T, double>) {
                EXPECT_DOUBLE_EQ(data[i], expected_value)
                    << "Failed at index " << i << " on " << device.to_string();
            }
        }
    }

    void VerifyGradientGeneric(const Tensor& t, double expected_value, size_t count) {
        switch(dtype) {
            case DType::Float32:
                VerifyGradient<float>(t, static_cast<float>(expected_value), count);
                break;
            case DType::Float64:
                VerifyGradient<double>(t, expected_value, count);
                break;
            default:
                FAIL() << "Unsupported dtype for gradient verification";
        }
    }

    // Helper to create scalar tensors with proper dtype
    Tensor scalar(double value) {
        if (dtype == DType::Float32) {
            return ones({2, 2}, dtype, device) * static_cast<float>(value);
        } else {
            return ones({2, 2}, dtype, device) * value;
        }
    }
};

// ============================================================================
// Variable Creation and Manipulation Tests
// ============================================================================

TEST_P(AutogradMultiDTypeTest, VariableCreation) {
    auto data = ones({2, 2}, dtype, device);
    auto var = Variable(data, true);

    EXPECT_TRUE(var.requires_grad()) << "Failed on " << device.to_string();
    EXPECT_FALSE(var.has_grad()) << "Failed on " << device.to_string();
}

TEST_P(AutogradMultiDTypeTest, Detach) {
    auto var = Variable(ones({2, 2}, dtype, device), true);
    auto detached = var.detach();

    EXPECT_FALSE(detached.requires_grad()) << "Failed on " << device.to_string();
}

// ============================================================================
// Basic Backward Operations Tests
// ============================================================================

TEST_P(AutogradMultiDTypeTest, SimpleAddBackward) {
    // Test: c = a + b
    // dc/da = 1, dc/db = 1
    auto a = Variable(ones({2, 2}, dtype, device), true);
    auto b = Variable(ones({2, 2}, dtype, device), true);
    auto c = a + b;

    // Backward with gradient of ones
    c.backward(ones({2, 2}, dtype, device));

    // Check gradients exist
    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    // Check gradient values
    auto a_grad = a.grad().value();
    auto b_grad = b.grad().value();

    VerifyGradientGeneric(a_grad, 1.0, 4);
    VerifyGradientGeneric(b_grad, 1.0, 4);
}

TEST_P(AutogradMultiDTypeTest, SimpleSubBackward) {
    // Test: c = a - b
    // dc/da = 1, dc/db = -1
    auto a = Variable(scalar(3.0), true);
    auto b = Variable(scalar(2.0), true);
    auto c = a - b;

    c.backward(ones({2, 2}, dtype, device));

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    auto a_grad = a.grad().value();
    auto b_grad = b.grad().value();

    VerifyGradientGeneric(a_grad, 1.0, 4);
    VerifyGradientGeneric(b_grad, -1.0, 4);
}

TEST_P(AutogradMultiDTypeTest, SimpleMulBackward) {
    // Test: c = a * b
    // dc/da = b, dc/db = a
    auto a_data = scalar(2.0);
    auto b_data = scalar(3.0);
    auto a = Variable(a_data, true);
    auto b = Variable(b_data, true);
    auto c = a * b;

    c.backward(ones({2, 2}, dtype, device));

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    auto a_grad = a.grad().value();
    auto b_grad = b.grad().value();

    // da should be b (3.0), db should be a (2.0)
    VerifyGradientGeneric(a_grad, 3.0, 4);
    VerifyGradientGeneric(b_grad, 2.0, 4);
}

TEST_P(AutogradMultiDTypeTest, SimpleDivBackward) {
    // Test: c = a / b
    // dc/da = 1/b, dc/db = -a/(b^2)
    auto a_data = scalar(6.0);
    auto b_data = scalar(2.0);
    auto a = Variable(a_data, true);
    auto b = Variable(b_data, true);
    auto c = a / b;

    c.backward(ones({2, 2}, dtype, device));

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    auto a_grad = a.grad().value();
    auto b_grad = b.grad().value();

    // da should be 1/2 = 0.5
    // db should be -6/(2^2) = -1.5
    VerifyGradientGeneric(a_grad, 0.5, 4);
    VerifyGradientGeneric(b_grad, -1.5, 4);
}

// ============================================================================
// Chained Operations Tests
// ============================================================================

TEST_P(AutogradMultiDTypeTest, ChainedOperations) {
    // Test: d = (a + b) * c
    auto a = Variable(scalar(2.0), true);
    auto b = Variable(scalar(3.0), true);
    auto c = Variable(scalar(4.0), true);

    auto d = (a + b) * c;
    d.backward(ones({2, 2}, dtype, device));

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(c.has_grad()) << "Failed on " << device.to_string();

    auto a_grad = a.grad().value();
    auto b_grad = b.grad().value();
    auto c_grad = c.grad().value();

    // d = (a + b) * c = (2 + 3) * 4 = 20
    // dd/da = c = 4
    // dd/db = c = 4
    // dd/dc = (a + b) = 5
    VerifyGradientGeneric(a_grad, 4.0, 4);
    VerifyGradientGeneric(b_grad, 4.0, 4);
    VerifyGradientGeneric(c_grad, 5.0, 4);
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi"};

    // Test with floating-point dtypes (autograd works with floating-point)
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
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
    AutogradMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE IMPACT SUMMARY:
 *
 * Original test_autograd.cpp:
 * - 7 tests × 4 backends × 1 dtype (Float32) = 28 test scenarios
 *
 * Refactored test_autograd_multidtype.cpp:
 * - 7 tests × 4 backends × 2 dtypes (Float32, Float64) = 56 test scenarios
 *
 * Coverage increase: 2x improvement
 *
 * Tests converted: 7/7 (100%)
 * - VariableCreation
 * - Detach
 * - SimpleAddBackward
 * - SimpleSubBackward
 * - SimpleMulBackward
 * - SimpleDivBackward
 * - ChainedOperations
 *
 * DTypes added:
 * - Float32 (original)
 * - Float64 (NEW - adds double precision gradient testing)
 *
 * Benefits:
 * - Ensures autograd works correctly with both single and double precision
 * - Tests numerical stability across different precision levels
 * - Validates gradient computation accuracy for Float64
 * - Maintains all original test logic with dtype parameterization
 *
 * Estimated coverage increase: From ~50% (Float32 only) to ~100% (both float types)
 */
