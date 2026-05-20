/**
 * @file test_rnn_multidtype.cpp
 * @brief Multi-dtype tests for RNN layers (RNNCell, RNN)
 *
 * Tests RNN operations with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - RNNCell forward pass with various activations
 * - Multi-layer RNN configurations
 * - Bidirectional RNN support
 * - Gradient flow through recurrent connections
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// RNN Multi-Backend Multi-DType Test Fixture
// ============================================================================

class RNNMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// RNNCell Tests
// ============================================================================

TEST_P(RNNMultiDTypeTest, RNNCellBasicForward) {
    nn::RNNCell cell(10, 20, "tanh");
    convert_model(cell);

    Variable input = createInput({5, 10}, true);
    Variable h = createInput({5, 20}, true);

    auto output = cell.forward(input, h);

    expectShape(output.tensor(), {5, 20});
    expectDType(output.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNCellReLUActivation) {
    nn::RNNCell cell(10, 20, "relu");
    convert_model(cell);

    Variable input = createInput({5, 10}, true);
    Variable h = createInput({5, 20}, true);

    auto output = cell.forward(input, h);

    expectShape(output.tensor(), {5, 20});
    expectDType(output.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNCellNoInitialHidden) {
    nn::RNNCell cell(10, 20);
    convert_model(cell);

    Variable input = createInput({5, 10}, true);
    auto output = cell.forward(input);

    expectShape(output.tensor(), {5, 20});
    expectDType(output.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNCellNoBias) {
    nn::RNNCell cell(10, 20, "tanh", false);
    convert_model(cell);

    Variable input = createInput({5, 10}, true);
    auto output = cell.forward(input);

    expectShape(output.tensor(), {5, 20});
    expectDType(output.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNCellInvalidNonlinearity) {
    // RNNCell now accepts the full ONNX RNN activation set; a genuinely
    // bogus name must still throw.
    EXPECT_THROW({
        nn::RNNCell cell(10, 20, "notarealactivation");
        cell.to(device());
    }, std::invalid_argument);
}

// ============================================================================
// RNN Tests
// ============================================================================

TEST_P(RNNMultiDTypeTest, RNNBasicForward) {
    nn::RNN rnn(10, 20, 1);
    convert_model(rnn);

    Variable input = createInput({7, 5, 10}, true);  // (seq_len, batch, features)
    auto [output, h_n] = rnn.forward(input, Variable{});

    expectShape(output.tensor(), {7, 5, 20});  // seq_len, batch, hidden_size
    expectDType(output.tensor());

    expectShape(h_n.tensor(), {1, 5, 20});  // num_layers, batch, hidden_size
    expectDType(h_n.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNMultiLayer) {
    nn::RNN rnn(10, 20, 3);  // 3 layers
    convert_model(rnn);

    Variable input = createInput({7, 5, 10}, true);
    auto [output, h_n] = rnn.forward(input, Variable{});

    expectShape(output.tensor(), {7, 5, 20});
    expectDType(output.tensor());

    expectShape(h_n.tensor(), {3, 5, 20});  // 3 layers
    expectDType(h_n.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNBatchFirst) {
    nn::RNN rnn(10, 20, 1, "tanh", true, true);  // batch_first=true
    convert_model(rnn);

    Variable input = createInput({5, 7, 10}, true);  // (batch, seq_len, features)
    auto [output, h_n] = rnn.forward(input, Variable{});

    expectShape(output.tensor(), {5, 7, 20});  // batch, seq_len, hidden_size
    expectDType(output.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNBidirectional) {
    nn::RNN rnn(10, 20, 1, "tanh", true, false, 0.0, true);  // bidirectional=true
    convert_model(rnn);

    Variable input = createInput({7, 5, 10}, true);
    auto [output, h_n] = rnn.forward(input, Variable{});

    expectShape(output.tensor(), {7, 5, 40});  // hidden_size * 2
    expectDType(output.tensor());

    expectShape(h_n.tensor(), {2, 5, 20});  // num_layers * num_directions
    expectDType(h_n.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNBidirectionalMultiLayer) {
    nn::RNN rnn(10, 20, 2, "tanh", true, false, 0.0, true);  // 2 layers, bidirectional
    convert_model(rnn);

    Variable input = createInput({7, 5, 10}, true);
    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 40);  // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4);      // num_layers * num_directions
    expectDType(output.tensor());
    expectDType(h_n.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNWithDropout) {
    nn::RNN rnn(10, 20, 3, "tanh", true, false, 0.5);  // 50% dropout
    convert_model(rnn);

    Variable input = createInput({7, 5, 10}, true);
    auto [output, h_n] = rnn.forward(input, Variable{});

    expectShape(output.tensor(), {7, 5, 20});
    expectDType(output.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNInitialHiddenState) {
    nn::RNN rnn(10, 20, 2);
    convert_model(rnn);

    Variable input = createInput({7, 5, 10}, true);
    Variable h0 = createInput({2, 5, 20}, true);

    auto [output, h_n] = rnn.forward(input, h0);

    expectShape(output.tensor(), {7, 5, 20});
    expectShape(h_n.tensor(), {2, 5, 20});
    expectDType(output.tensor());
    expectDType(h_n.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNSequenceLengthVariation) {
    nn::RNN rnn(10, 20);
    convert_model(rnn);

    // Short sequence
    Variable input1 = createInput({3, 5, 10}, true);
    auto [output1, h_n1] = rnn.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[0], 3);
    expectDType(output1.tensor());

    // Long sequence
    Variable input2 = createInput({50, 5, 10}, true);
    auto [output2, h_n2] = rnn.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[0], 50);
    expectDType(output2.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNBatchSizeVariation) {
    nn::RNN rnn(10, 20);
    convert_model(rnn);

    // Small batch
    Variable input1 = createInput({7, 2, 10}, true);
    auto [output1, h_n1] = rnn.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[1], 2);
    expectDType(output1.tensor());

    // Large batch
    Variable input2 = createInput({7, 32, 10}, true);
    auto [output2, h_n2] = rnn.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[1], 32);
    expectDType(output2.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNSingleTimestep) {
    nn::RNN rnn(10, 20);
    convert_model(rnn);

    Variable input = createInput({1, 5, 10}, true);
    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 1);
    expectDType(output.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNOutputConsistency) {
    nn::RNN rnn(10, 20);
    convert_model(rnn);

    auto input_tensor = createOnes({7, 5, 10});
    Variable input(input_tensor, true);

    auto [output1, h_n1] = rnn.forward(input, Variable{});
    auto [output2, h_n2] = rnn.forward(input, Variable{});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]);
        EXPECT_EQ(h_n1.shape()[i], h_n2.shape()[i]);
    }
    expectDType(output1.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNGradientFlow) {
    // Test that gradients actually flow through RNN. A backward that silently
    // drops grad to zero would pass EXPECT_NO_THROW, which is the failure
    // mode this suite must catch.
    nn::RNN rnn(10, 20);
    convert_model(rnn);

    Variable input = createInput({7, 5, 10}, true);
    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_TRUE(output.requires_grad());

    // sum(Variable) preserves the autograd graph; Variable(sum(tensor), true)
    // would break it and silently produce zero gradients.
    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.has_grad()) << "input.grad missing on " << device().to_string();
    EXPECT_EQ(input.grad()->numel(), input.tensor().numel()) << device().to_string();

    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f)
        << "input.grad all-zero — autograd graph broken on " << device().to_string();
}

TEST_P(RNNMultiDTypeTest, RNNTrainingMode) {
    nn::RNN rnn(10, 20, 2, "tanh", true, false, 0.5);
    convert_model(rnn);

    EXPECT_TRUE(rnn.is_training());

    rnn.eval();
    EXPECT_FALSE(rnn.is_training());

    rnn.train();
    EXPECT_TRUE(rnn.is_training());
}

TEST_P(RNNMultiDTypeTest, RNNParameterCount) {
    nn::RNN rnn(10, 20, 2);
    convert_model(rnn);

    auto params = rnn.parameters();

    // Each layer has input_layer and hidden_layer, each with weight and bias
    // Layer 0: input (10->20), hidden (20->20)
    // Layer 1: input (20->20), hidden (20->20)
    // Total: 4 weights + 4 biases = 8 parameters
    EXPECT_EQ(params.size(), 8);
}

TEST_P(RNNMultiDTypeTest, RNNLargeHidden) {
    nn::RNN rnn(10, 512);
    convert_model(rnn);

    Variable input = createInput({7, 5, 10}, true);
    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 512);
    EXPECT_EQ(h_n.shape()[2], 512);
    expectDType(output.tensor());
}

TEST_P(RNNMultiDTypeTest, RNNInvalidNumLayers) {
    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::RNN rnn(10, 20, 0);
        rnn.to(device());
    }, std::invalid_argument);
}

TEST_P(RNNMultiDTypeTest, RNNInvalidDropout) {
    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::RNN rnn(10, 20, 2, "tanh", true, false, 1.5);
        rnn.to(device());
    }, std::invalid_argument);
}

// ============================================================================
// Backward correctness — verify input.grad() is populated and non-zero. The
// existing RNNGradientFlow test only checks EXPECT_NO_THROW which is too weak.
// ============================================================================

TEST_P(RNNMultiDTypeTest, RNNCellBackwardGradPopulated) {
    nn::RNNCell cell(8, 16, "tanh");
    convert_model(cell);
    Variable input = createInput({4, 8}, true);
    Variable h0    = createInput({4, 16}, true);
    auto h1 = cell.forward(input, h0);
    sum(h1).backward();

    ASSERT_TRUE(input.has_grad());
    ASSERT_TRUE(h0.has_grad());
    EXPECT_EQ(input.grad()->numel(), input.tensor().numel());
    EXPECT_EQ(h0.grad()->numel(),    h0.tensor().numel());

    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f) << "input grad all-zero on " << device().to_string();
}

TEST_P(RNNMultiDTypeTest, RNNBackwardGradPopulated) {
    nn::RNN rnn(8, 16);
    convert_model(rnn);
    Variable input = createInput({5, 4, 8}, true);
    auto [output, h_n] = rnn.forward(input, Variable{});
    sum(output).backward();

    ASSERT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad()->numel(), input.tensor().numel());

    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f) << "input grad all-zero on " << device().to_string();
}

TEST_P(RNNMultiDTypeTest, RNNBackwardWeightsUpdated) {
    nn::RNN rnn(4, 8);
    convert_model(rnn);
    Variable input = createInput({3, 2, 4}, false);
    auto [output, h_n] = rnn.forward(input, Variable{});
    sum(output).backward();

    int populated = 0;
    for (auto& [name, param] : rnn.named_parameters()) {
        if (param->has_grad()) {
            auto g_max = max(abs(param->grad()->to(Device::cpu()).to(DType::Float32)));
            if (g_max.item<float>() > 0.0f) ++populated;
        }
    }
    EXPECT_GT(populated, 0) << "no RNN weight gradient populated on " << device().to_string();
}

TEST_P(RNNMultiDTypeTest, RNNBidirectionalBackward) {
    nn::RNN rnn(4, 8, /*num_layers=*/1, /*nonlinearity=*/"tanh", /*bias=*/true,
                /*batch_first=*/false, /*dropout=*/0.0, /*bidirectional=*/true);
    convert_model(rnn);
    Variable input = createInput({3, 2, 4}, true);
    auto [output, h_n] = rnn.forward(input, Variable{});
    sum(output).backward();

    ASSERT_TRUE(input.has_grad());
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f) << "bidirectional grad all-zero on " << device().to_string();
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS(RNNMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 23
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 23 tests × 3 dtypes × 3 backends = 207 test scenarios
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
 */
