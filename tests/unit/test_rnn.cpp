#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// RNNCell Tests
// ============================================================================

class RNNCellTestFixture : public BackendTest {};

TEST_P(RNNCellTestFixture, BasicForward) {
    // Test basic forward pass with tanh activation
    nn::RNNCell cell(10, 20, "tanh");

    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);
    auto h = Variable(randn({5, 20}, DType::Float32, device), true);

    auto output = cell.forward(input, h);

    EXPECT_EQ(output.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(RNNCellTestFixture, ReLUActivation) {
    // Test with ReLU activation
    nn::RNNCell cell(10, 20, "relu");

    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);
    auto h = Variable(randn({5, 20}, DType::Float32, device), true);

    auto output = cell.forward(input, h);

    EXPECT_EQ(output.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(RNNCellTestFixture, NoInitialHidden) {
    // Test with zero-initialized hidden state
    nn::RNNCell cell(10, 20);

    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto output = cell.forward(input);

    EXPECT_EQ(output.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(RNNCellTestFixture, NoBias) {
    // Test without bias
    nn::RNNCell cell(10, 20, "tanh", false);

    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto output = cell.forward(input);

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(RNNCellTestFixture, InvalidNonlinearity) {
    // Test that invalid activation throws
    EXPECT_THROW({
        nn::RNNCell cell(10, 20, "sigmoid");

        cell.to(device);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

// forward_with_precomputed_ih took a raw Tensor for the input-side gate and
// previously did `Variable(precomputed_ih + h_out.tensor(), true)` — a raw
// Tensor add wrapped in a fresh Variable with no grad_fn. That silently
// severed the hidden-side autograd chain: hx and weight_hh produced zero
// gradients. Rewritten to use Variable-level arithmetic; this test locks the
// fix in.
TEST_P(RNNCellTestFixture, PrecomputedIhPropagatesHiddenGradient) {
    nn::RNNCell cell(10, 20, "tanh");
    cell.to(device);

    // Precomputed input-side gate as raw Tensor (caller opted out of input
    // autograd — that's the contract of this API).
    auto precomputed_ih = randn({5, 20}, DType::Float32, device);
    Variable hx(randn({5, 20}, DType::Float32, device), /*requires_grad=*/true);

    auto out = cell.forward_with_precomputed_ih(precomputed_ih, hx);
    auto loss = tenzor::sum(out);
    loss.backward();

    ASSERT_TRUE(hx.has_grad())
        << "forward_with_precomputed_ih must propagate grad back to hx on "
        << device.to_string();
    auto g = hx.grad().value().to(Device::cpu()).contiguous();
    const float* gp = g.data<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < g.numel(); ++i) {
        EXPECT_FALSE(std::isnan(gp[i])) << "grad NaN at " << i
                                         << " on " << device.to_string();
        max_abs = std::max(max_abs, std::abs(gp[i]));
    }
    EXPECT_GT(max_abs, 0.0f)
        << "hx grad identically zero — hidden-side autograd severed on "
        << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(RNNCellTestFixture);

// ============================================================================
// RNN Tests
// ============================================================================

class RNNTestFixture : public BackendTest {};

TEST_P(RNNTestFixture, BasicForward) {
    // Test basic forward pass
    nn::RNN rnn(10, 20, 1);

    rnn.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);  // (seq_len, batch, features)

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size

    EXPECT_EQ(h_n.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[0], 1) << "Failed on " << device.to_string();  // num_layers
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size
}

TEST_P(RNNTestFixture, MultiLayer) {
    // Test multi-layer RNN
    nn::RNN rnn(10, 20, 3);

    rnn.to(device);  // 3 layers
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string();

    EXPECT_EQ(h_n.shape()[0], 3) << "Failed on " << device.to_string();  // 3 layers
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, BatchFirst) {
    // Test with batch_first=true
    nn::RNN rnn(10, 20, 1, "tanh", true, true);

    rnn.to(device);  // batch_first=true
    auto input = Variable(randn({5, 7, 10}, DType::Float32, device), true);  // (batch, seq_len, features)

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size
}

TEST_P(RNNTestFixture, Bidirectional) {
    // Test bidirectional RNN
    nn::RNN rnn(10, 20, 1, "tanh", true, false, 0.0, true);

    rnn.to(device);  // bidirectional=true
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2

    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string();  // num_layers * num_directions
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, BidirectionalMultiLayer) {
    // Test bidirectional multi-layer RNN
    nn::RNN rnn(10, 20, 2, "tanh", true, false, 0.0, true);

    rnn.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4) << "Failed on " << device.to_string();     // num_layers * num_directions
}

TEST_P(RNNTestFixture, WithDropout) {
    // Test with dropout between layers
    nn::RNN rnn(10, 20, 3, "tanh", true, false, 0.5);

    rnn.to(device);  // 50% dropout
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, InitialHiddenState) {
    // Test with provided initial hidden state
    nn::RNN rnn(10, 20, 2);

    rnn.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);
    auto h0 = Variable(randn({2, 5, 20}, DType::Float32, device), true);

    auto [output, h_n] = rnn.forward(input, h0);

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, SequenceLengthVariation) {
    // Test with different sequence lengths
    nn::RNN rnn(10, 20);

    rnn.to(device);

    // Short sequence
    auto input1 = Variable(randn({3, 5, 10}, DType::Float32, device), true);
    auto [output1, h_n1] = rnn.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[0], 3) << "Failed on " << device.to_string();

    // Long sequence
    auto input2 = Variable(randn({50, 5, 10}, DType::Float32, device), true);
    auto [output2, h_n2] = rnn.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[0], 50) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, BatchSizeVariation) {
    // Test with different batch sizes
    nn::RNN rnn(10, 20);

    rnn.to(device);

    // Small batch
    auto input1 = Variable(randn({7, 2, 10}, DType::Float32, device), true);
    auto [output1, h_n1] = rnn.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[1], 2) << "Failed on " << device.to_string();

    // Large batch
    auto input2 = Variable(randn({7, 32, 10}, DType::Float32, device), true);
    auto [output2, h_n2] = rnn.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[1], 32) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, SingleTimestep) {
    // Test with single timestep
    nn::RNN rnn(10, 20);

    rnn.to(device);
    auto input = Variable(randn({1, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, OutputConsistency) {
    // Test that output is deterministic
    nn::RNN rnn(10, 20);

    rnn.to(device);
    auto input = Variable(ones({7, 5, 10}, DType::Float32, device), true);

    auto [output1, h_n1] = rnn.forward(input, Variable{});
    auto [output2, h_n2] = rnn.forward(input, Variable{});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]) << "Failed on " << device.to_string();
        EXPECT_EQ(h_n1.shape()[i], h_n2.shape()[i]) << "Failed on " << device.to_string();
    }
}

TEST_P(RNNTestFixture, GradientFlow) {
    // Test that gradients can flow through RNN
    nn::RNN rnn(10, 20);

    rnn.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    // Check that output requires grad
    EXPECT_TRUE(output.requires_grad()) << "Failed on " << device.to_string();

    // Sum for scalar output
    auto loss = Variable(sum(output.tensor(), std::nullopt, false), true);

    // Backward should not throw
    EXPECT_NO_THROW({
        loss.backward();
    }) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, TrainingMode) {
    // Test training/eval mode switching
    nn::RNN rnn(10, 20, 2, "tanh", true, false, 0.5);

    rnn.to(device);

    EXPECT_TRUE(rnn.is_training()) << "Failed on " << device.to_string();

    rnn.eval();
    EXPECT_FALSE(rnn.is_training()) << "Failed on " << device.to_string();

    rnn.train();
    EXPECT_TRUE(rnn.is_training()) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, ParameterCount) {
    // Test parameter counting
    nn::RNN rnn(10, 20, 2);

    rnn.to(device);
    auto params = rnn.parameters();

    // Each layer has input_layer and hidden_layer, each with weight and bias
    // Layer 0: input (10->20), hidden (20->20)
    // Layer 1: input (20->20), hidden (20->20)
    // Total: 4 weights + 4 biases = 8 parameters
    EXPECT_EQ(params.size(), 8) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, LargeHidden) {
    // Test with large hidden size
    nn::RNN rnn(10, 512);

    rnn.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 512) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[2], 512) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, InvalidNumLayers) {
    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::RNN rnn(10, 20, 0);

        rnn.to(device);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(RNNTestFixture, InvalidDropout) {
    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::RNN rnn(10, 20, 2, "tanh", true, false, 1.5);

        rnn.to(device);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(RNNTestFixture);
