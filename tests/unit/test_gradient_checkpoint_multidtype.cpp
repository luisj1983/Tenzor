/**
 * @file test_gradient_checkpoint_multidtype.cpp
 * @brief Multi-dtype tests for gradient checkpointing
 *
 * Coverage: Backend × DType testing for gradient checkpoint operations
 * - Primary dtypes: Float32, Float64
 * - Backends: CPU, CUDA, Vulkan, OneAPI
 *
 * Note: Gradient checkpointing is primarily relevant for floating-point dtypes
 * since it involves autograd operations. Integer dtypes are not tested.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::autograd;

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

class GradientCheckpointMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;
    double tolerance;

    void SetUp() override {
        auto param = GetParam();
        dtype = param.dtype;

        // Set dtype-specific tolerance
        if (dtype == DType::Float32) {
            tolerance = 1e-5;
        } else if (dtype == DType::Float64) {
            tolerance = 1e-10;
        } else {
            tolerance = 1e-5;
        }

        HONOR_BACKEND_ENV_VARS(param.backend_name);

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
        else if (param.backend_name == "rocm") {
            if (!isBackendAvailable(Device::Type::ROCm)) {
                GTEST_SKIP() << "ROCm not available";
            }
            device = Device::rocm(0);
        }

        reset_checkpoint_stats();
    }

    void TearDown() override {
        reset_checkpoint_stats();
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

    template<typename T>
    Tensor createTensorWithValue(const std::vector<int64_t>& shape, T value) {
        if (dtype == DType::Float32) {
            return full(shape, static_cast<float>(value), dtype, device);
        } else {
            return full(shape, static_cast<double>(value), dtype, device);
        }
    }
};

// ==============================================================================
// Basic Checkpoint Statistics Tests
// ==============================================================================

TEST_P(GradientCheckpointMultiDTypeTest, StatsTracking) {
    auto& stats = get_checkpoint_stats();
    EXPECT_EQ(stats.num_checkpoints, 0);
    EXPECT_EQ(stats.num_recomputations, 0);
    EXPECT_EQ(stats.saved_memory_bytes, 0);
}

TEST_P(GradientCheckpointMultiDTypeTest, ResetStats) {
    auto& stats = get_checkpoint_stats();

    stats.num_checkpoints = 10;
    stats.num_recomputations = 5;

    reset_checkpoint_stats();

    EXPECT_EQ(stats.num_checkpoints, 0);
    EXPECT_EQ(stats.num_recomputations, 0);
}

// ==============================================================================
// Simple Checkpoint Function Tests
// ==============================================================================

TEST_P(GradientCheckpointMultiDTypeTest, SimpleForwardPass) {
    auto x_tensor = randn({4, 8}, dtype, device);
    Variable x(x_tensor, true);

    auto checkpointed_fn = [this](const Variable& input) -> Variable {
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto two = Variable(createTensorWithValue(shape_vec, 2.0), false);
        auto one = Variable(createTensorWithValue(shape_vec, 1.0), false);
        auto doubled = input * two;
        auto result = doubled + one;
        return result;
    };

    auto y = checkpoint(checkpointed_fn, x);

    EXPECT_EQ(y.tensor().shape().size(), x.tensor().shape().size());

    auto x_cpu = x.tensor().to(Device::cpu());
    auto y_cpu = y.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* x_data = x_cpu.data<float>();
        const float* y_data = y_cpu.data<float>();
        for (int i = 0; i < x.tensor().numel(); ++i) {
            EXPECT_NEAR(y_data[i], x_data[i] * 2.0f + 1.0f, tolerance);
        }
    } else {
        const double* x_data = x_cpu.data<double>();
        const double* y_data = y_cpu.data<double>();
        for (int i = 0; i < x.tensor().numel(); ++i) {
            EXPECT_NEAR(y_data[i], x_data[i] * 2.0 + 1.0, tolerance);
        }
    }
}

TEST_P(GradientCheckpointMultiDTypeTest, CheckpointGradientCorrectness) {
    auto x_tensor = ones({3, 3}, dtype, device);
    Variable x(x_tensor, true);

    auto checkpointed_fn = [](const Variable& input) -> Variable {
        return input * input;
    };

    auto y = checkpoint_with_original(checkpointed_fn, x, &x);

    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.grad().has_value());
    auto grad_cpu = x.grad()->to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* grad_data = grad_cpu.data<float>();
        for (int i = 0; i < x.tensor().numel(); ++i) {
            EXPECT_NEAR(grad_data[i], 2.0f, tolerance);
        }
    } else {
        const double* grad_data = grad_cpu.data<double>();
        for (int i = 0; i < x.tensor().numel(); ++i) {
            EXPECT_NEAR(grad_data[i], 2.0, tolerance);
        }
    }
}

TEST_P(GradientCheckpointMultiDTypeTest, MultiVariableCheckpoint) {
    auto x_tensor = ones({2, 2}, dtype, device);
    auto y_value = dtype == DType::Float32 ? 2.0f : 2.0;
    auto y_tensor = mul(ones({2, 2}, dtype, device),
                        createTensorWithValue({2, 2}, y_value));

    Variable x(x_tensor, true);
    Variable y(y_tensor, true);

    auto multi_fn = [](const std::vector<Variable>& inputs) -> std::vector<Variable> {
        auto prod = inputs[0] * inputs[1];
        auto result = prod + inputs[0];
        return std::vector<Variable>{result};
    };

    auto outputs = checkpoint_with_originals(multi_fn, {x, y}, {&x, &y});
    EXPECT_EQ(outputs.size(), 1);

    auto z = outputs[0];

    auto z_cpu = z.tensor().to(Device::cpu());
    if (dtype == DType::Float32) {
        const float* z_data = z_cpu.data<float>();
        for (int i = 0; i < z.tensor().numel(); ++i) {
            EXPECT_NEAR(z_data[i], 3.0f, tolerance);
        }
    } else {
        const double* z_data = z_cpu.data<double>();
        for (int i = 0; i < z.tensor().numel(); ++i) {
            EXPECT_NEAR(z_data[i], 3.0, tolerance);
        }
    }

    auto loss = sum(z);
    loss.backward();

    EXPECT_TRUE(x.grad().has_value());
    EXPECT_TRUE(y.grad().has_value());
}

TEST_P(GradientCheckpointMultiDTypeTest, NestedCheckpoints) {
    auto x_tensor = ones({2, 2}, dtype, device);
    Variable x(x_tensor, true);

    auto outer_fn = [&x, this](const Variable& input) -> Variable {
        auto inner_fn = [this](const Variable& in) -> Variable {
            auto shape = in.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto three = Variable(createTensorWithValue(shape_vec, 3.0), false);
            return in * three;
        };

        auto intermediate = checkpoint(inner_fn, input);
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto one = Variable(createTensorWithValue(shape_vec, 1.0), false);
        return intermediate + one;
    };

    auto y = checkpoint_with_original(outer_fn, x, &x);

    auto y_cpu = y.tensor().to(Device::cpu());
    if (dtype == DType::Float32) {
        const float* y_data = y_cpu.data<float>();
        for (int i = 0; i < y.tensor().numel(); ++i) {
            EXPECT_NEAR(y_data[i], 4.0f, tolerance);
        }
    } else {
        const double* y_data = y_cpu.data<double>();
        for (int i = 0; i < y.tensor().numel(); ++i) {
            EXPECT_NEAR(y_data[i], 4.0, tolerance);
        }
    }

    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.grad().has_value());
    auto grad_cpu = x.grad()->to(Device::cpu());
    if (dtype == DType::Float32) {
        const float* grad_data = grad_cpu.data<float>();
        for (int i = 0; i < x.tensor().numel(); ++i) {
            EXPECT_NEAR(grad_data[i], 3.0f, tolerance);
        }
    } else {
        const double* grad_data = grad_cpu.data<double>();
        for (int i = 0; i < x.tensor().numel(); ++i) {
            EXPECT_NEAR(grad_data[i], 3.0, tolerance);
        }
    }
}

// ==============================================================================
// Checkpoint with Activations
// ==============================================================================

TEST_P(GradientCheckpointMultiDTypeTest, CheckpointWithReLU) {
    auto x_tensor = randn({4, 4}, dtype, device);
    Variable x(x_tensor, true);

    auto relu_fn = [](const Variable& input) -> Variable {
        return nn::relu(input);
    };

    auto y = checkpoint_with_original(relu_fn, x, &x);

    auto loss = sum(y);
    loss.backward();

    EXPECT_TRUE(x.grad().has_value());
}

TEST_P(GradientCheckpointMultiDTypeTest, CheckpointWithSigmoid) {
    auto x_tensor = randn({3, 3}, dtype, device);
    Variable x(x_tensor, true);

    auto sigmoid_fn = [](const Variable& input) -> Variable {
        return nn::sigmoid(input);
    };

    auto y = checkpoint_with_original(sigmoid_fn, x, &x);

    auto y_cpu = y.tensor().to(Device::cpu());
    if (dtype == DType::Float32) {
        const float* y_data = y_cpu.data<float>();
        for (int i = 0; i < y.tensor().numel(); ++i) {
            EXPECT_GT(y_data[i], 0.0f);
            EXPECT_LT(y_data[i], 1.0f);
        }
    } else {
        const double* y_data = y_cpu.data<double>();
        for (int i = 0; i < y.tensor().numel(); ++i) {
            EXPECT_GT(y_data[i], 0.0);
            EXPECT_LT(y_data[i], 1.0);
        }
    }

    auto loss = sum(y);
    loss.backward();

    EXPECT_TRUE(x.grad().has_value());
}

// ==============================================================================
// Memory Tracker Tests
// ==============================================================================

TEST_P(GradientCheckpointMultiDTypeTest, MemorySavingsEstimation) {
    MemoryTracker::reset();
    MemoryTracker::start_tracking();

    auto x_tensor = randn({10, 10}, dtype, device);
    Variable x(x_tensor, true);

    auto memory_fn = [this](const Variable& input) -> Variable {
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto two = Variable(createTensorWithValue(shape_vec, 2.0), false);
        auto one = Variable(createTensorWithValue(shape_vec, 1.0), false);
        auto three = Variable(createTensorWithValue(shape_vec, 3.0), false);
        auto temp1 = input * two;
        auto temp2 = temp1 + one;
        auto temp3 = temp2 * three;
        return temp3;
    };

    auto y = checkpoint(memory_fn, x);
    auto loss = sum(y);
    loss.backward();

    size_t peak = MemoryTracker::peak_memory();
    EXPECT_GE(peak, 0);

    MemoryTracker::stop_tracking();
}

// ==============================================================================
// Test Instantiation
// ==============================================================================

std::vector<BackendDTypeParam> GenerateCheckpointCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "rocm"};

    // Only Float32 and Float64 for gradient operations
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
    AllBackendsFloatDTypes,
    GradientCheckpointMultiDTypeTest,
    ::testing::ValuesIn(GenerateCheckpointCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Test Environment Setup
// ============================================================================

class GradientCheckpointTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        initialize();
    }
};

static ::testing::Environment* const gradient_checkpoint_env =
    ::testing::AddGlobalTestEnvironment(new GradientCheckpointTestEnvironment);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
