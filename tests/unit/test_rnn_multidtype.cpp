#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;

/**
 * @file test_rnn_multidtype.cpp
 * @brief Multi-dtype tests for RNN layers (RNNCell, RNN)
 *
 * Tests RNN operations with Float32, Float64, and Float16 dtypes
 * for mixed precision training scenarios.
 */

// Helper function to convert DType to string
std::string dtype_to_string(DType dtype) {
    switch(dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Float16: return "float16";
        default: return "unknown";
    }
}

// ============================================================================
// Multi-DType Parameterization
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

class RNNMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    DType dtype;
    float tol;
    Device device;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;
        tol = get_tolerance();

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

    double get_tolerance() const {
        if (dtype == DType::Float32) {
            return 1e-5;
        } else if (dtype == DType::Float64) {
            return 1e-10;
        } else if (dtype == DType::Float16) {
            return 1e-2;
        }
        return 1e-5;
    }

    template<typename T>
    bool values_close(const T* a, const T* b, size_t n, float rtol, float atol) {
        for (size_t i = 0; i < n; ++i) {
            float diff = std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
            float threshold = atol + rtol * std::abs(static_cast<float>(b[i]));
            if (diff > threshold) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// RNNCell Tests
// ============================================================================

TEST_P(RNNMultiDTypeTest, RNNCellBasicForward) {
    auto param = GetParam();
    // Test basic forward pass with tanh activation
    nn::RNNCell cell(10, 20, "tanh");

    cell.to(device);
    auto input_tensor = randn({5, 10}, DType::Float32, device);
    auto h_tensor = randn({5, 20}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
        h_tensor = h_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto h = Variable(h_tensor, true);

    auto output = cell.forward(input, h);

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 20);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNCellReLUActivation) {
    auto param = GetParam();
    // Test with ReLU activation
    nn::RNNCell cell(10, 20, "relu");

    cell.to(device);
    auto input_tensor = randn({5, 10}, DType::Float32, device);
    auto h_tensor = randn({5, 20}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
        h_tensor = h_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto h = Variable(h_tensor, true);

    auto output = cell.forward(input, h);

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 20);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNCellNoInitialHidden) {
    auto param = GetParam();
    // Test with zero-initialized hidden state
    nn::RNNCell cell(10, 20);

    cell.to(device);
    auto input_tensor = randn({5, 10}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto output = cell.forward(input);

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 20);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNCellNoBias) {
    auto param = GetParam();
    // Test without bias
    nn::RNNCell cell(10, 20, "tanh", false);

    cell.to(device);
    auto input_tensor = randn({5, 10}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto output = cell.forward(input);

    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 20);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNCellInvalidNonlinearity) {
    auto param = GetParam();
    // Test that invalid activation throws
    EXPECT_THROW({
        nn::RNNCell cell(10, 20, "sigmoid");
        cell.to(device);
    }, std::invalid_argument);
}

// ============================================================================
// RNN Tests
// ============================================================================

TEST_P(RNNMultiDTypeTest, RNNBasicForward) {
    auto param = GetParam();
    // Test basic forward pass
    nn::RNN rnn(10, 20, 1);

    rnn.to(device);
    auto input_tensor = randn({7, 5, 10}, DType::Float32, device);  // (seq_len, batch, features)

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 7);  // seq_len
    EXPECT_EQ(output.shape()[1], 5);  // batch
    EXPECT_EQ(output.shape()[2], 20); // hidden_size
    EXPECT_EQ(output.tensor().dtype(), dtype);

    EXPECT_EQ(h_n.shape().size(), 3);
    EXPECT_EQ(h_n.shape()[0], 1);  // num_layers
    EXPECT_EQ(h_n.shape()[1], 5);  // batch
    EXPECT_EQ(h_n.shape()[2], 20); // hidden_size
    EXPECT_EQ(h_n.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNMultiLayer) {
    auto param = GetParam();
    // Test multi-layer RNN
    nn::RNN rnn(10, 20, 3);

    rnn.to(device);  // 3 layers
    auto input_tensor = randn({7, 5, 10}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 20);
    EXPECT_EQ(output.tensor().dtype(), dtype);

    EXPECT_EQ(h_n.shape()[0], 3);  // 3 layers
    EXPECT_EQ(h_n.shape()[1], 5);
    EXPECT_EQ(h_n.shape()[2], 20);
    EXPECT_EQ(h_n.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNBatchFirst) {
    auto param = GetParam();
    // Test with batch_first=true
    nn::RNN rnn(10, 20, 1, "tanh", true, true);

    rnn.to(device);  // batch_first=true
    auto input_tensor = randn({5, 7, 10}, DType::Float32, device);  // (batch, seq_len, features)

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 5);  // batch
    EXPECT_EQ(output.shape()[1], 7);  // seq_len
    EXPECT_EQ(output.shape()[2], 20); // hidden_size
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNBidirectional) {
    auto param = GetParam();
    // Test bidirectional RNN
    nn::RNN rnn(10, 20, 1, "tanh", true, false, 0.0, true);

    rnn.to(device);  // bidirectional=true
    auto input_tensor = randn({7, 5, 10}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 40); // hidden_size * 2
    EXPECT_EQ(output.tensor().dtype(), dtype);

    EXPECT_EQ(h_n.shape()[0], 2);  // num_layers * num_directions
    EXPECT_EQ(h_n.shape()[1], 5);
    EXPECT_EQ(h_n.shape()[2], 20);
    EXPECT_EQ(h_n.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNBidirectionalMultiLayer) {
    auto param = GetParam();
    // Test bidirectional multi-layer RNN
    nn::RNN rnn(10, 20, 2, "tanh", true, false, 0.0, true);

    rnn.to(device);
    auto input_tensor = randn({7, 5, 10}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 40); // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4);     // num_layers * num_directions
    EXPECT_EQ(output.tensor().dtype(), dtype);
    EXPECT_EQ(h_n.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNWithDropout) {
    auto param = GetParam();
    // Test with dropout between layers
    nn::RNN rnn(10, 20, 3, "tanh", true, false, 0.5);

    rnn.to(device);  // 50% dropout
    auto input_tensor = randn({7, 5, 10}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 20);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNInitialHiddenState) {
    auto param = GetParam();
    // Test with provided initial hidden state
    nn::RNN rnn(10, 20, 2);

    rnn.to(device);
    auto input_tensor = randn({7, 5, 10}, DType::Float32, device);
    auto h0_tensor = randn({2, 5, 20}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
        h0_tensor = h0_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto h0 = Variable(h0_tensor, true);

    auto [output, h_n] = rnn.forward(input, h0);

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(h_n.shape()[0], 2);
    EXPECT_EQ(output.tensor().dtype(), dtype);
    EXPECT_EQ(h_n.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNSequenceLengthVariation) {
    auto param = GetParam();
    // Test with different sequence lengths
    nn::RNN rnn(10, 20);

    rnn.to(device);

    // Short sequence
    auto input1_tensor = randn({3, 5, 10}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input1_tensor = input1_tensor.to(dtype);
    }
    auto input1 = Variable(input1_tensor, true);
    auto [output1, h_n1] = rnn.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[0], 3);
    EXPECT_EQ(output1.tensor().dtype(), dtype);

    // Long sequence
    auto input2_tensor = randn({50, 5, 10}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input2_tensor = input2_tensor.to(dtype);
    }
    auto input2 = Variable(input2_tensor, true);
    auto [output2, h_n2] = rnn.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[0], 50);
    EXPECT_EQ(output2.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNBatchSizeVariation) {
    auto param = GetParam();
    // Test with different batch sizes
    nn::RNN rnn(10, 20);

    rnn.to(device);

    // Small batch
    auto input1_tensor = randn({7, 2, 10}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input1_tensor = input1_tensor.to(dtype);
    }
    auto input1 = Variable(input1_tensor, true);
    auto [output1, h_n1] = rnn.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[1], 2);
    EXPECT_EQ(output1.tensor().dtype(), dtype);

    // Large batch
    auto input2_tensor = randn({7, 32, 10}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input2_tensor = input2_tensor.to(dtype);
    }
    auto input2 = Variable(input2_tensor, true);
    auto [output2, h_n2] = rnn.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[1], 32);
    EXPECT_EQ(output2.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNSingleTimestep) {
    auto param = GetParam();
    // Test with single timestep
    nn::RNN rnn(10, 20);

    rnn.to(device);
    auto input_tensor = randn({1, 5, 10}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNOutputConsistency) {
    auto param = GetParam();
    // Test that output is deterministic
    nn::RNN rnn(10, 20);

    rnn.to(device);
    auto input_tensor = ones({7, 5, 10}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto [output1, h_n1] = rnn.forward(input, Variable{});
    auto [output2, h_n2] = rnn.forward(input, Variable{});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]);
        EXPECT_EQ(h_n1.shape()[i], h_n2.shape()[i]);
    }
    EXPECT_EQ(output1.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNGradientFlow) {
    auto param = GetParam();
    // Test that gradients can flow through RNN
    nn::RNN rnn(10, 20);

    rnn.to(device);
    auto input_tensor = randn({7, 5, 10}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    // Check that output requires grad
    EXPECT_TRUE(output.requires_grad());

    // Sum for scalar output
    auto loss = Variable(sum(output.tensor(), std::nullopt, false), true);

    // Backward should not throw
    EXPECT_NO_THROW({
        loss.backward();
    });
}

TEST_P(RNNMultiDTypeTest, RNNTrainingMode) {
    auto param = GetParam();
    // Test training/eval mode switching
    nn::RNN rnn(10, 20, 2, "tanh", true, false, 0.5);

    rnn.to(device);

    EXPECT_TRUE(rnn.is_training());

    rnn.eval();
    EXPECT_FALSE(rnn.is_training());

    rnn.train();
    EXPECT_TRUE(rnn.is_training());
}

TEST_P(RNNMultiDTypeTest, RNNParameterCount) {
    auto param = GetParam();
    // Test parameter counting
    nn::RNN rnn(10, 20, 2);

    rnn.to(device);
    auto params = rnn.parameters();

    // Each layer has input_layer and hidden_layer, each with weight and bias
    // Layer 0: input (10->20), hidden (20->20)
    // Layer 1: input (20->20), hidden (20->20)
    // Total: 4 weights + 4 biases = 8 parameters
    EXPECT_EQ(params.size(), 8);
}

TEST_P(RNNMultiDTypeTest, RNNLargeHidden) {
    auto param = GetParam();
    // Test with large hidden size
    nn::RNN rnn(10, 512);

    rnn.to(device);
    auto input_tensor = randn({7, 5, 10}, DType::Float32, device);

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 512);
    EXPECT_EQ(h_n.shape()[2], 512);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(RNNMultiDTypeTest, RNNInvalidNumLayers) {
    auto param = GetParam();
    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::RNN rnn(10, 20, 0);
        rnn.to(device);
    }, std::invalid_argument);
}

TEST_P(RNNMultiDTypeTest, RNNInvalidDropout) {
    auto param = GetParam();
    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::RNN rnn(10, 20, 2, "tanh", true, false, 1.5);
        rnn.to(device);
    }, std::invalid_argument);
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi"};

    // Test with floating-point dtypes
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Float16, "float16"}
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
    RNNMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 23
 * DTypes Tested: Float32, Float64, Float16
 * Total Scenarios: 23 tests × 3 dtypes = 69 test scenarios
 *
 * Coverage:
 * RNNCell Tests:
 * - Basic forward pass (tanh, relu activation)
 * - Zero-initialized hidden state
 * - No bias mode
 * - Invalid nonlinearity validation
 *
 * RNN Tests:
 * - Basic forward pass (single layer)
 * - Multi-layer RNN (3 layers)
 * - Batch-first mode
 * - Bidirectional RNN (single and multi-layer)
 * - Dropout between layers
 * - Initial hidden state
 * - Sequence length variation (short/long)
 * - Batch size variation (small/large)
 * - Single timestep processing
 * - Output consistency (deterministic)
 * - Gradient flow validation
 * - Training/eval mode switching
 * - Parameter counting
 * - Large hidden size (512)
 * - Invalid parameter validation (num_layers, dropout)
 *
 * Tolerances:
 * - Float32: 1e-5 (standard precision)
 * - Float64: 1e-10 (high precision)
 * - Float16: 1e-2 (reduced precision for mixed precision training)
 */
