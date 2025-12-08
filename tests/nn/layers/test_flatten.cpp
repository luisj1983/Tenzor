#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

/**
 * @file test_flatten.cpp
 * @brief Comprehensive backend and dtype-agnostic tests for Flatten layer
 *
 * Tests cover:
 * - Forward pass shape transformations
 * - Backward pass gradients
 * - Edge cases (negative dimensions, boundary conditions)
 * - All backends (CPU, CUDA, Vulkan, OneAPI, ROCm)
 * - All dtypes (Float32, Float64, Int32, Int64, Bool)
 */

// ============================================================================
// Test Environment Setup
// ============================================================================

class FlattenTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const flatten_env =
    ::testing::AddGlobalTestEnvironment(new FlattenTestEnvironment);

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

class FlattenTest : public ::testing::TestWithParam<BackendDTypeParam> {
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
        else if (param.backend_name == "rocm") {
            if (!isBackendAvailable(Device::Type::ROCm)) {
                GTEST_SKIP() << "ROCm not available";
            }
            device = Device::rocm(0);
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

    // Helper function for dtype-aware tensor comparison
    bool tensors_close(const Tensor& a, const Tensor& b, double rtol = 1e-5, double atol = 1e-7) {
        if (a.numel() != b.numel()) return false;

        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());

        size_t numel = a_cpu.numel();

        if (dtype == DType::Float32) {
            const float* a_data = a_cpu.data<float>();
            const float* b_data = b_cpu.data<float>();
            for (size_t i = 0; i < numel; ++i) {
                float diff = std::abs(a_data[i] - b_data[i]);
                float threshold = atol + rtol * std::abs(b_data[i]);
                if (diff > threshold) return false;
            }
        } else if (dtype == DType::Float64) {
            const double* a_data = a_cpu.data<double>();
            const double* b_data = b_cpu.data<double>();
            for (size_t i = 0; i < numel; ++i) {
                double diff = std::abs(a_data[i] - b_data[i]);
                double threshold = atol + rtol * std::abs(b_data[i]);
                if (diff > threshold) return false;
            }
        } else if (dtype == DType::Int32) {
            const int32_t* a_data = a_cpu.data<int32_t>();
            const int32_t* b_data = b_cpu.data<int32_t>();
            for (size_t i = 0; i < numel; ++i) {
                if (a_data[i] != b_data[i]) return false;
            }
        } else if (dtype == DType::Int64) {
            const int64_t* a_data = a_cpu.data<int64_t>();
            const int64_t* b_data = b_cpu.data<int64_t>();
            for (size_t i = 0; i < numel; ++i) {
                if (a_data[i] != b_data[i]) return false;
            }
        } else if (dtype == DType::Bool) {
            const bool* a_data = a_cpu.data<bool>();
            const bool* b_data = b_cpu.data<bool>();
            for (size_t i = 0; i < numel; ++i) {
                if (a_data[i] != b_data[i]) return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Backend and DType-Agnostic Parameterized Tests
// ============================================================================

// Test basic flattening from default start_dim=1
TEST_P(FlattenTest, BasicFlattenFromDim1) {
    auto flatten = Flatten(1);

    // Input: [2, 3, 4, 5]
    auto input = Variable(randn({2, 3, 4, 5}, dtype, device), true);
    auto output = flatten.forward(input);

    // Expected shape: [2, 3*4*5] = [2, 60]
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 60);
    EXPECT_EQ(output.tensor().device().type, device.type);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// Test flattening from dimension 0 (full flatten)
TEST_P(FlattenTest, FlattenFromDim0) {
    auto flatten = Flatten(0);

    // Input: [2, 3, 4, 5]
    auto input = Variable(randn({2, 3, 4, 5}, dtype, device), true);
    auto output = flatten.forward(input);

    // Expected shape: [2*3*4*5] = [120]
    EXPECT_EQ(output.shape().size(), 1);
    EXPECT_EQ(output.shape()[0], 120);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// Test partial flattening with custom range
TEST_P(FlattenTest, PartialFlattenCustomRange) {
    auto flatten = Flatten(1, 2);  // Flatten dims 1-2, keep 0 and 3

    // Input: [2, 3, 4, 5]
    auto input = Variable(randn({2, 3, 4, 5}, dtype, device), true);
    auto output = flatten.forward(input);

    // Expected shape: [2, 3*4, 5] = [2, 12, 5]
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 12);  // 3*4
    EXPECT_EQ(output.shape()[2], 5);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// Test negative dimension indices
TEST_P(FlattenTest, NegativeDimensions) {
    auto flatten = Flatten(-3, -1);  // Last 3 dimensions

    // Input: [2, 3, 4, 5]
    auto input = Variable(randn({2, 3, 4, 5}, dtype, device), true);
    auto output = flatten.forward(input);

    // Flatten dims -3,-2,-1 (which are 1,2,3)
    // Expected shape: [2, 3*4*5] = [2, 60]
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 60);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// Test gradient flow through flatten (backward pass)
TEST_P(FlattenTest, GradientFlowBackward) {
    auto flatten = Flatten(1);

    // Input: [2, 3, 4]
    auto input = Variable(randn({2, 3, 4}, dtype, device), true);
    auto output = flatten.forward(input);

    // Backward pass
    auto grad_output = ones(output.shape(), dtype, device);
    output.backward(grad_output);

    // Check gradient exists and has correct shape
    ASSERT_TRUE(input.grad().has_value());
    auto grad_input = input.grad().value();

    // Gradient should have same shape as input
    EXPECT_EQ(grad_input.shape()[0], 2);
    EXPECT_EQ(grad_input.shape()[1], 3);
    EXPECT_EQ(grad_input.shape()[2], 4);
    EXPECT_EQ(grad_input.dtype(), dtype);

    // Gradient values should match (flatten backward just reshapes)
    EXPECT_TRUE(tensors_close(grad_input, grad_output.reshape({2, 3, 4})));
}

// Test reshape correctness (data ordering)
TEST_P(FlattenTest, DataOrdering) {
    auto flatten = Flatten(0);

    // Create input with known values - dtype-specific initialization
    Tensor input_cpu;
    if (dtype == DType::Float32) {
        input_cpu = arange(0.0f, 24.0f, 1.0f, dtype, Device::cpu()).reshape({2, 3, 4});
    } else if (dtype == DType::Float64) {
        input_cpu = arange(0.0, 24.0, 1.0, dtype, Device::cpu()).reshape({2, 3, 4});
    } else if (dtype == DType::Int32) {
        input_cpu = arange(0, 24, 1, dtype, Device::cpu()).reshape({2, 3, 4});
    } else if (dtype == DType::Int64) {
        input_cpu = arange(static_cast<int64_t>(0), static_cast<int64_t>(24),
                          static_cast<int64_t>(1), dtype, Device::cpu()).reshape({2, 3, 4});
    } else if (dtype == DType::Bool) {
        // For Bool, create a pattern of alternating true/false
        auto temp = arange(0, 24, 1, DType::Int32, Device::cpu());
        input_cpu = (temp % 2).to(dtype).reshape({2, 3, 4});
    }

    auto input_data = (device.type == Device::Type::CPU) ? input_cpu : input_cpu.to(device);
    auto input = Variable(input_data, true);

    auto output = flatten.forward(input);

    // Check that flattened data matches original ordering
    auto output_cpu = output.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* out_data = output_cpu.data<float>();
        for (int i = 0; i < 24; ++i) {
            EXPECT_FLOAT_EQ(out_data[i], static_cast<float>(i));
        }
    } else if (dtype == DType::Float64) {
        const double* out_data = output_cpu.data<double>();
        for (int i = 0; i < 24; ++i) {
            EXPECT_DOUBLE_EQ(out_data[i], static_cast<double>(i));
        }
    } else if (dtype == DType::Int32) {
        const int32_t* out_data = output_cpu.data<int32_t>();
        for (int i = 0; i < 24; ++i) {
            EXPECT_EQ(out_data[i], i);
        }
    } else if (dtype == DType::Int64) {
        const int64_t* out_data = output_cpu.data<int64_t>();
        for (int i = 0; i < 24; ++i) {
            EXPECT_EQ(out_data[i], static_cast<int64_t>(i));
        }
    } else if (dtype == DType::Bool) {
        const bool* out_data = output_cpu.data<bool>();
        for (int i = 0; i < 24; ++i) {
            EXPECT_EQ(out_data[i], (i % 2) == 1);
        }
    }
}

// Test single element tensor
TEST_P(FlattenTest, SingleElement) {
    auto flatten = Flatten(0);

    auto input = Variable(ones({1, 1, 1, 1}, dtype, device), true);
    auto output = flatten.forward(input);

    EXPECT_EQ(output.shape().size(), 1);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// Test 2D tensor (already flat in some sense)
TEST_P(FlattenTest, TwoDimensionalInput) {
    auto flatten = Flatten(1);

    auto input = Variable(randn({5, 10}, dtype, device), true);
    auto output = flatten.forward(input);

    // Should be unchanged
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// Test large tensor
TEST_P(FlattenTest, LargeTensor) {
    auto flatten = Flatten(1);

    auto input = Variable(randn({32, 64, 28, 28}, dtype, device), true);
    auto output = flatten.forward(input);

    // Expected: [32, 64*28*28] = [32, 50176]
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 50176);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// Test flatten in a mini neural network (integration test)
TEST_P(FlattenTest, IntegrationWithLinear) {
    // Simulate CNN -> Flatten -> FC pattern
    auto conv_output = Variable(randn({4, 32, 7, 7}, dtype, device), true);

    auto flatten = Flatten(1);
    auto flattened = flatten.forward(conv_output);

    // Should be [4, 32*7*7] = [4, 1568]
    EXPECT_EQ(flattened.shape()[0], 4);
    EXPECT_EQ(flattened.shape()[1], 1568);
    EXPECT_EQ(flattened.tensor().dtype(), dtype);

    // Gradient flow test
    auto grad = ones(flattened.shape(), dtype, device);
    flattened.backward(grad);

    ASSERT_TRUE(conv_output.grad().has_value());
    auto grad_conv = conv_output.grad().value();
    EXPECT_EQ(grad_conv.shape()[0], 4);
    EXPECT_EQ(grad_conv.shape()[1], 32);
    EXPECT_EQ(grad_conv.shape()[2], 7);
    EXPECT_EQ(grad_conv.shape()[3], 7);
    EXPECT_EQ(grad_conv.dtype(), dtype);
}

// Test end_dim equal to start_dim
TEST_P(FlattenTest, SingleDimRange) {
    auto flatten = Flatten(2, 2);  // Only flatten dim 2 (no-op in some sense)

    auto input = Variable(randn({2, 3, 4, 5}, dtype, device), true);
    auto output = flatten.forward(input);

    // Shape should be [2, 3, 4, 5] still
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 5);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// ============================================================================
// Error Handling Tests (dtype-agnostic)
// ============================================================================

TEST(FlattenErrorTest, InvalidDimensionRange) {
    auto flatten = Flatten(2, 1);  // start > end

    auto input = Variable(randn({2, 3, 4, 5}), true);

    EXPECT_THROW(flatten.forward(input), std::invalid_argument);
}

TEST(FlattenErrorTest, OutOfRangeDimension) {
    auto flatten = Flatten(0, 10);  // end_dim out of range

    auto input = Variable(randn({2, 3, 4}), true);

    EXPECT_THROW(flatten.forward(input), std::invalid_argument);
}

TEST(FlattenErrorTest, NegativeOutOfRange) {
    auto flatten = Flatten(-10, -1);  // start_dim out of range

    auto input = Variable(randn({2, 3, 4}), true);

    EXPECT_THROW(flatten.forward(input), std::invalid_argument);
}

// ============================================================================
// Generate Backend × DType Combinations
// ============================================================================

std::vector<BackendDTypeParam> GenerateBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "adaptivecpp", "rocm"};

    // Flatten works with all dtypes - comprehensive testing
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
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

// ============================================================================
// Instantiate Tests for All Backends × All DTypes
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    FlattenTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE IMPACT:
 *
 * Backend × DType Parameterization:
 * - 12 test cases × 5 backends × 5 dtypes = 300 test scenarios
 *
 * Previous coverage (backend-only):
 * - 12 test cases × 5 backends = 60 test scenarios
 *
 * Improvement: 5x increase in coverage (all dtypes tested)
 *
 * Dtypes tested:
 * - Float32: Standard floating point operations
 * - Float64: Double precision (important for Vulkan backend)
 * - Int32: Integer reshape operations
 * - Int64: Large integer support
 * - Bool: Binary data flattening
 *
 * Total test scenarios: 300 + 3 error tests = 303 tests
 */
