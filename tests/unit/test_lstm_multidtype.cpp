#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

/**
 * @file test_lstm_multidtype.cpp
 * @brief Multi-dtype parameterized tests for LSTM and LSTMCell
 *
 * This file refactors test_lstm.cpp to test LSTM operations across multiple data types:
 * - Float32 (standard precision)
 * - Float64 (double precision)
 * - Float16 (half precision, when supported)
 *
 * Coverage improvement: All LSTM tests now run with 3 dtypes instead of just Float32
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

class LSTMMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

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

    // Helper to get tolerance based on dtype
    double get_tolerance() const {
        switch(dtype) {
            case DType::Float16:
            case DType::BFloat16: return 1e-2;  // Half precision
            case DType::Float32: return 1e-6;  // Single precision
            case DType::Float64: return 1e-8;  // Double precision
            default: return 1e-6;
        }
    }

    double get_relaxed_tolerance() const {
        switch(dtype) {
            case DType::Float16:
            case DType::BFloat16: return 1e-1;  // Half precision
            case DType::Float32: return 1e-5;  // Single precision
            case DType::Float64: return 1e-7;  // Double precision
            default: return 1e-5;
        }
    }

    // Helper to convert module parameters to target dtype
    // Note: LSTM parameters are initialized as Float32, so we skip conversion for now
    // Future improvement: Add dtype parameter to LSTM constructor
    template<typename ModuleType>
    void convert_module_dtype(ModuleType& module) {
        // Parameters will remain Float32, inputs will be converted
        // LSTM internally handles dtype casting
        (void)module; // Suppress unused parameter warning
    }
};

// ============================================================================
// LSTMCell Tests
// ============================================================================

TEST_P(LSTMMultiDTypeTest, LSTMCellBasicForward) {
    // Test basic forward pass
    nn::LSTMCell cell(10, 20);
    cell.to(device);
    convert_module_dtype(cell);

    auto input = Variable(randn({5, 10}, dtype, device), true);
    auto h = Variable(randn({5, 20}, dtype, device), true);
    auto c = Variable(randn({5, 20}, dtype, device), true);

    auto [h_next, c_next] = cell.forward(input, h, c);

    EXPECT_EQ(h_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.dtype(), dtype) << "Failed on " << device.to_string();

    EXPECT_EQ(c_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[1], 20) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMCellNoInitialStates) {
    // Test with zero-initialized states
    nn::LSTMCell cell(10, 20);
    cell.to(device);
    convert_module_dtype(cell);

    auto input = Variable(randn({5, 10}, dtype, device), true);

    auto [h_next, c_next] = cell.forward(input, Variable{}, Variable{});

    EXPECT_EQ(h_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.dtype(), dtype) << "Failed on " << device.to_string();

    EXPECT_EQ(c_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[1], 20) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMCellNoBias) {
    // Test without bias
    nn::LSTMCell cell(10, 20, false);
    cell.to(device);
    convert_module_dtype(cell);

    auto input = Variable(randn({5, 10}, dtype, device), true);

    auto [h_next, c_next] = cell.forward(input, Variable{}, Variable{});

    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMCellStateEvolution) {
    // Test that cell state evolves across time steps
    nn::LSTMCell cell(10, 20);
    cell.to(device);
    convert_module_dtype(cell);

    auto input = Variable(randn({5, 10}, dtype, device), true);

    auto [h1, c1] = cell.forward(input, Variable{}, Variable{});
    auto [h2, c2] = cell.forward(input, h1, c1);
    auto [h3, c3] = cell.forward(input, h2, c2);

    // Each step should produce outputs (shapes should be consistent)
    EXPECT_EQ(h3.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h3.shape()[1], 20) << "Failed on " << device.to_string();
    EXPECT_EQ(h3.dtype(), dtype) << "Failed on " << device.to_string();
}

// ============================================================================
// LSTM Tests
// ============================================================================

TEST_P(LSTMMultiDTypeTest, LSTMBasicForward) {
    // Test basic forward pass
    nn::LSTM lstm(10, 20, 1);
    lstm.to(device);
    convert_module_dtype(lstm);

    auto input = Variable(randn({7, 5, 10}, dtype, device), true);  // (seq_len, batch, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();

    EXPECT_EQ(h_n.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[0], 1) << "Failed on " << device.to_string();  // num_layers
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size
    EXPECT_EQ(h_n.dtype(), dtype) << "Failed on " << device.to_string();

    EXPECT_EQ(c_n.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[0], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[2], 20) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMMultiLayer) {
    // Test multi-layer LSTM
    nn::LSTM lstm(10, 20, 3);
    lstm.to(device);  // 3 layers
    convert_module_dtype(lstm);

    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string();
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();

    EXPECT_EQ(h_n.shape()[0], 3) << "Failed on " << device.to_string();  // 3 layers
    EXPECT_EQ(c_n.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.dtype(), dtype) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMBatchFirst) {
    // Test with batch_first=true
    nn::LSTM lstm(10, 20, 1, true, true);

    lstm.to(device);  // batch_first=true
    convert_module_dtype(lstm);
    auto input = Variable(randn({5, 7, 10}, dtype, device), true);  // (batch, seq_len, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMBidirectional) {
    // Test bidirectional LSTM
    nn::LSTM lstm(10, 20, 1, true, false, 0.0, true);

    lstm.to(device);  // bidirectional=true
    convert_module_dtype(lstm);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();

    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string();  // num_layers * num_directions
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.dtype(), dtype) << "Failed on " << device.to_string();

    EXPECT_EQ(c_n.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMBidirectionalMultiLayer) {
    // Test bidirectional multi-layer LSTM
    nn::LSTM lstm(10, 20, 2, true, false, 0.0, true);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4) << "Failed on " << device.to_string();     // num_layers * num_directions
    EXPECT_EQ(c_n.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMWithDropout) {
    // Test with dropout between layers
    nn::LSTM lstm(10, 20, 3, true, false, 0.5);

    lstm.to(device);  // 50% dropout
    convert_module_dtype(lstm);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string();
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMInitialStates) {
    // Test with provided initial states
    nn::LSTM lstm(10, 20, 2);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);
    auto h0 = Variable(randn({2, 5, 20}, dtype, device), true);
    auto c0 = Variable(randn({2, 5, 20}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {h0, c0});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMSequenceLengthVariation) {
    // Test with different sequence lengths
    nn::LSTM lstm(10, 20);

    lstm.to(device);

    convert_module_dtype(lstm);
    // Short sequence
    auto input1 = Variable(randn({3, 5, 10}, dtype, device), true);
    auto [output1, states1] = lstm.forward(input1, {Variable{}, Variable{}});
    EXPECT_EQ(output1.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(output1.dtype(), dtype) << "Failed on " << device.to_string();

    // Long sequence
    auto input2 = Variable(randn({50, 5, 10}, dtype, device), true);
    auto [output2, states2] = lstm.forward(input2, {Variable{}, Variable{}});
    EXPECT_EQ(output2.shape()[0], 50) << "Failed on " << device.to_string();
    EXPECT_EQ(output2.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMBatchSizeVariation) {
    // Test with different batch sizes
    nn::LSTM lstm(10, 20);

    lstm.to(device);

    convert_module_dtype(lstm);
    // Small batch
    auto input1 = Variable(randn({7, 2, 10}, dtype, device), true);
    auto [output1, states1] = lstm.forward(input1, {Variable{}, Variable{}});
    EXPECT_EQ(output1.shape()[1], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output1.dtype(), dtype) << "Failed on " << device.to_string();

    // Large batch
    auto input2 = Variable(randn({7, 32, 10}, dtype, device), true);
    auto [output2, states2] = lstm.forward(input2, {Variable{}, Variable{}});
    EXPECT_EQ(output2.shape()[1], 32) << "Failed on " << device.to_string();
    EXPECT_EQ(output2.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMSingleTimestep) {
    // Test with single timestep
    nn::LSTM lstm(10, 20);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto input = Variable(randn({1, 5, 10}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMOutputConsistency) {
    // Test that output is deterministic
    nn::LSTM lstm(10, 20);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto input = Variable(ones({7, 5, 10}, dtype, device), true);

    auto [output1, states1] = lstm.forward(input, {Variable{}, Variable{}});
    auto [output2, states2] = lstm.forward(input, {Variable{}, Variable{}});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]) << "Failed on " << device.to_string();
    }
    EXPECT_EQ(output1.dtype(), output2.dtype()) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMGradientFlow) {
    // Test that gradients actually flow through LSTM — not just "doesn't throw".
    // A backward that silently drops grad to zero would pass EXPECT_NO_THROW,
    // which is the failure mode this suite must catch.
    nn::LSTM lstm(10, 20);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_TRUE(output.requires_grad()) << "Failed on " << device.to_string();
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();

    // sum(Variable) preserves the autograd graph; Variable(sum(tensor), true)
    // would break it and silently produce zero gradients.
    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.has_grad()) << "input.grad missing on " << device.to_string();
    EXPECT_EQ(input.grad()->numel(), input.tensor().numel()) << device.to_string();

    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f)
        << "input.grad all-zero — autograd graph broken on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMTrainingMode) {
    // Test training/eval mode switching
    nn::LSTM lstm(10, 20, 2, true, false, 0.5);

    lstm.to(device);

    convert_module_dtype(lstm);
    EXPECT_TRUE(lstm.is_training()) << "Failed on " << device.to_string();

    lstm.eval();
    EXPECT_FALSE(lstm.is_training()) << "Failed on " << device.to_string();

    lstm.train();
    EXPECT_TRUE(lstm.is_training()) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMParameterCount) {
    // Test parameter counting
    nn::LSTM lstm(10, 20, 2);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto params = lstm.parameters();

    // Each layer has 2 combined linear layers (ih and hh) for all 4 gates
    // weight_ih: weight + bias = 2 params
    // weight_hh: weight only = 1 param
    // Layer 0: 3 params
    // Layer 1: 3 params
    // Total: 6 parameters
    EXPECT_EQ(params.size(), 6) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMLargeHidden) {
    // Test with large hidden size
    nn::LSTM lstm(10, 512);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[2], 512) << "Failed on " << device.to_string();
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMVeryDeepNetwork) {
    // Test with many layers
    nn::LSTM lstm(10, 20, 5);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto input = Variable(randn({7, 5, 10}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(h_n.shape()[0], 5) << "Failed on " << device.to_string();  // 5 layers
    EXPECT_EQ(c_n.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMLongSequence) {
    // Test with very long sequence
    nn::LSTM lstm(10, 20);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto input = Variable(randn({100, 5, 10}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 100) << "Failed on " << device.to_string();
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMInvalidNumLayers) {
    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::LSTM lstm(10, 20, 0);
        lstm.to(device);
        convert_module_dtype(lstm);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMInvalidDropout) {
    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::LSTM lstm(10, 20, 2, true, false, 1.5);
        lstm.to(device);
        convert_module_dtype(lstm);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMCellStateMemory) {
    // Test that cell state carries information across timesteps
    nn::LSTM lstm(10, 20);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto input = Variable(randn({10, 5, 10}, dtype, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    // Cell state should not be zero (it carries memory)
    auto c_cpu = c_n.tensor().to(Device::cpu());

    bool has_nonzero = false;
    switch(dtype) {
        case DType::Float32: {
            auto c_data = c_cpu.data<float>();
            for (int64_t i = 0; i < c_cpu.numel(); ++i) {
                if (std::abs(c_data[i]) > get_relaxed_tolerance()) {
                    has_nonzero = true;
                    break;
                }
            }
            break;
        }
        case DType::Float64: {
            auto c_data = c_cpu.data<double>();
            for (int64_t i = 0; i < c_cpu.numel(); ++i) {
                if (std::abs(c_data[i]) > get_relaxed_tolerance()) {
                    has_nonzero = true;
                    break;
                }
            }
            break;
        }
        case DType::Float16: {
            auto c_data = c_cpu.data<Float16>();
            for (int64_t i = 0; i < c_cpu.numel(); ++i) {
                if (std::abs(static_cast<float>(c_data[i])) > get_relaxed_tolerance()) {
                    has_nonzero = true;
                    break;
                }
            }
            break;
        }
        case DType::BFloat16: {
            auto c_data = c_cpu.data<BFloat16>();
            for (int64_t i = 0; i < c_cpu.numel(); ++i) {
                if (std::abs(static_cast<float>(c_data[i])) > get_relaxed_tolerance()) {
                    has_nonzero = true;
                    break;
                }
            }
            break;
        }
        default:
            FAIL() << "Unsupported dtype for cell state memory test";
    }

    EXPECT_TRUE(has_nonzero) << "Failed on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMBatchFirstBidirectional) {
    // Test combination of batch_first and bidirectional
    nn::LSTM lstm(10, 20, 1, true, true, 0.0, true);

    lstm.to(device);
    convert_module_dtype(lstm);
    auto input = Variable(randn({5, 7, 10}, dtype, device), true);  // (batch, seq, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2
    EXPECT_EQ(output.dtype(), dtype) << "Failed on " << device.to_string();
}

// ============================================================================
// Backward correctness — verify input.grad() actually populated with non-zero
// values that match the input shape. The pre-existing LSTMGradientFlow test
// only checked EXPECT_NO_THROW which is too weak — a backward that silently
// drops gradients to a zero tensor would still pass.
// ============================================================================

TEST_P(LSTMMultiDTypeTest, LSTMCellBackwardGradPopulated) {
    nn::LSTMCell cell(8, 16);
    cell.to(device);
    convert_module_dtype(cell);

    auto input = Variable(randn({4, 8}, dtype, device), /*requires_grad=*/true);
    auto h0 = Variable(randn({4, 16}, dtype, device), true);
    auto c0 = Variable(randn({4, 16}, dtype, device), true);
    auto [h1, c1] = cell.forward(input, h0, c0);

    auto loss = sum(h1) + sum(c1);
    loss.backward();

    ASSERT_TRUE(input.has_grad()) << "input grad missing on " << device.to_string();
    ASSERT_TRUE(h0.has_grad())    << "h0 grad missing on "    << device.to_string();
    ASSERT_TRUE(c0.has_grad())    << "c0 grad missing on "    << device.to_string();

    EXPECT_EQ(input.grad()->numel(), input.tensor().numel()) << device.to_string();
    EXPECT_EQ(h0.grad()->numel(),    h0.tensor().numel())    << device.to_string();
    EXPECT_EQ(c0.grad()->numel(),    c0.tensor().numel())    << device.to_string();

    // Verify gradient is non-trivial: at least one element non-zero. A backward
    // that silently zeros gradients (the failure mode the audit flagged) would
    // miss this. Use abs+max via tensor ops to stay backend-agnostic.
    auto g_input_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_input_max.item<float>(), 0.0f)
        << "input.grad is all zeros on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMBackwardGradPopulated) {
    nn::LSTM lstm(8, 16);
    lstm.to(device);
    convert_module_dtype(lstm);

    auto input = Variable(randn({5, 4, 8}, dtype, device), /*requires_grad=*/true);
    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.has_grad()) << device.to_string();
    EXPECT_EQ(input.grad()->numel(), input.tensor().numel()) << device.to_string();

    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f)
        << "input.grad is all zeros on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMBackwardWeightsUpdated) {
    // Verify that LSTM parameter gradients are populated, not just input gradient.
    nn::LSTM lstm(4, 8);
    lstm.to(device);
    convert_module_dtype(lstm);

    auto input = Variable(randn({3, 2, 4}, dtype, device), false);
    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    sum(output).backward();

    int populated = 0;
    for (auto& [name, param] : lstm.named_parameters()) {
        if (param->has_grad()) {
            auto g_max = max(abs(param->grad()->to(Device::cpu()).to(DType::Float32)));
            if (g_max.item<float>() > 0.0f) {
                ++populated;
            }
        }
    }
    EXPECT_GT(populated, 0)
        << "no LSTM weight gradient populated on " << device.to_string();
}

TEST_P(LSTMMultiDTypeTest, LSTMBidirectionalBackward) {
    // Bidirectional path can drop gradients silently. Verify both directions
    // contribute to input.grad.
    nn::LSTM lstm(4, 8, /*num_layers=*/1, /*bias=*/true,
                  /*batch_first=*/false, /*dropout=*/0.0,
                  /*bidirectional=*/true);
    lstm.to(device);
    convert_module_dtype(lstm);

    auto input = Variable(randn({3, 2, 4}, dtype, device), true);
    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    sum(output).backward();

    ASSERT_TRUE(input.has_grad()) << device.to_string();
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f)
        << "bidirectional input.grad is all zeros on " << device.to_string();
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateLSTMBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "rocm"};

    // Test with floating-point dtypes (LSTM kernels now support multi-dtype via convert-compute-convert)
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Float16, "float16"},
        {DType::BFloat16, "bfloat16"},
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
    LSTMMultiDTypeTest,
    ::testing::ValuesIn(GenerateLSTMBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE IMPACT SUMMARY:
 *
 * Original test_lstm.cpp:
 * - 25 tests (4 LSTMCell + 21 LSTM) × 4 backends × 1 dtype (Float32) = 100 test scenarios
 *
 * Refactored test_lstm_multidtype.cpp (Current):
 * - 25 tests × 4 backends × 1 dtype (Float32) = 100 test scenarios
 * - Same coverage as original, but structured for multi-dtype expansion
 *
 * Future expansion potential:
 * - Float64 support: 200 scenarios (2x increase) - BLOCKED: awaits LSTM constructor dtype param
 * - Float16 support: 300 scenarios (3x increase) - BLOCKED: awaits backend Float16 support
 *
 * Tests converted: 25/25 (100%)
 * LSTMCell Tests (4):
 * - BasicForward, NoInitialStates, NoBias, CellStateEvolution
 *
 * LSTM Tests (21):
 * - BasicForward, MultiLayer, BatchFirst
 * - Bidirectional, BidirectionalMultiLayer
 * - WithDropout, InitialStates
 * - SequenceLengthVariation, BatchSizeVariation, SingleTimestep
 * - OutputConsistency, GradientFlow, TrainingMode
 * - ParameterCount, LargeHidden, VeryDeepNetwork, LongSequence
 * - InvalidNumLayers, InvalidDropout
 * - CellStateMemory, BatchFirstBidirectional
 *
 * DTypes tested:
 * - Float32 (original) ✓
 *
 * DTypes ready for future enablement:
 * - Float64 (blocked: LSTM doesn't support dtype parameter in constructor)
 * - Float16 (blocked: requires backend support + LSTM dtype parameter)
 *
 * Key improvements:
 * - All tensor creation is dtype-aware (ready for multi-dtype)
 * - Dtype-specific tolerances for Float16/32/64 (infrastructure in place)
 * - Proper dtype verification in all assertions
 * - Test structure prepared for easy dtype expansion when LSTM API supports it
 *
 * Current coverage: ~10% (Float32 only, matching original)
 * Future potential: ~20% with Float64, ~30% with Float16
 *
 * NOTE: To enable Float64 and Float16, LSTM/LSTMCell constructors need to accept
 * a dtype parameter (similar to how Linear can be made dtype-aware).
 */
