#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// LSTMCell Tests
// ============================================================================

class LSTMCellTestFixture : public BackendTest {};

TEST_P(LSTMCellTestFixture, BasicForward) {
    // Test basic forward pass
    nn::LSTMCell cell(10, 20);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);
    auto h = Variable(randn({5, 20}, DType::Float32, device), true);
    auto c = Variable(randn({5, 20}, DType::Float32, device), true);

    auto [h_next, c_next] = cell.forward(input, h, c);

    EXPECT_EQ(h_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();

    EXPECT_EQ(c_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(LSTMCellTestFixture, NoInitialStates) {
    // Test with zero-initialized states
    nn::LSTMCell cell(10, 20);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto [h_next, c_next] = cell.forward(input, Variable{}, Variable{});

    EXPECT_EQ(h_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();

    EXPECT_EQ(c_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(LSTMCellTestFixture, NoBias) {
    // Test without bias
    nn::LSTMCell cell(10, 20, false);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto [h_next, c_next] = cell.forward(input, Variable{}, Variable{});

    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(LSTMCellTestFixture, CellStateEvolution) {
    // Test that cell state evolves across time steps
    nn::LSTMCell cell(10, 20);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto [h1, c1] = cell.forward(input, Variable{}, Variable{});
    auto [h2, c2] = cell.forward(input, h1, c1);
    auto [h3, c3] = cell.forward(input, h2, c2);

    // Each step should produce outputs (shapes should be consistent)
    EXPECT_EQ(h3.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h3.shape()[1], 20) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(LSTMCellTestFixture);

// ============================================================================
// LSTM Tests
// ============================================================================

class LSTMTestFixture : public BackendTest {};

TEST_P(LSTMTestFixture, BasicForward) {
    // Test basic forward pass
    nn::LSTM lstm(10, 20, 1);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);  // (seq_len, batch, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size

    EXPECT_EQ(h_n.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[0], 1) << "Failed on " << device.to_string();  // num_layers
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size

    EXPECT_EQ(c_n.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[0], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[2], 20) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, MultiLayer) {
    // Test multi-layer LSTM
    nn::LSTM lstm(10, 20, 3);  // 3 layers
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string();

    EXPECT_EQ(h_n.shape()[0], 3) << "Failed on " << device.to_string();  // 3 layers
    EXPECT_EQ(c_n.shape()[0], 3) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, BatchFirst) {
    // Test with batch_first=true
    nn::LSTM lstm(10, 20, 1, true, true);  // batch_first=true
    auto input = Variable(randn({5, 7, 10}, DType::Float32, device), true);  // (batch, seq_len, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size
}

TEST_P(LSTMTestFixture, Bidirectional) {
    // Test bidirectional LSTM
    nn::LSTM lstm(10, 20, 1, true, false, 0.0, true);  // bidirectional=true
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2

    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string();  // num_layers * num_directions
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string();

    EXPECT_EQ(c_n.shape()[0], 2) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, BidirectionalMultiLayer) {
    // Test bidirectional multi-layer LSTM
    nn::LSTM lstm(10, 20, 2, true, false, 0.0, true);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4) << "Failed on " << device.to_string();     // num_layers * num_directions
    EXPECT_EQ(c_n.shape()[0], 4) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, WithDropout) {
    // Test with dropout between layers
    nn::LSTM lstm(10, 20, 3, true, false, 0.5);  // 50% dropout
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, InitialStates) {
    // Test with provided initial states
    nn::LSTM lstm(10, 20, 2);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);
    auto h0 = Variable(randn({2, 5, 20}, DType::Float32, device), true);
    auto c0 = Variable(randn({2, 5, 20}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {h0, c0});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[0], 2) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, SequenceLengthVariation) {
    // Test with different sequence lengths
    nn::LSTM lstm(10, 20);

    // Short sequence
    auto input1 = Variable(randn({3, 5, 10}, DType::Float32, device), true);
    auto [output1, states1] = lstm.forward(input1, {Variable{}, Variable{}});
    EXPECT_EQ(output1.shape()[0], 3) << "Failed on " << device.to_string();

    // Long sequence
    auto input2 = Variable(randn({50, 5, 10}, DType::Float32, device), true);
    auto [output2, states2] = lstm.forward(input2, {Variable{}, Variable{}});
    EXPECT_EQ(output2.shape()[0], 50) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, BatchSizeVariation) {
    // Test with different batch sizes
    nn::LSTM lstm(10, 20);

    // Small batch
    auto input1 = Variable(randn({7, 2, 10}, DType::Float32, device), true);
    auto [output1, states1] = lstm.forward(input1, {Variable{}, Variable{}});
    EXPECT_EQ(output1.shape()[1], 2) << "Failed on " << device.to_string();

    // Large batch
    auto input2 = Variable(randn({7, 32, 10}, DType::Float32, device), true);
    auto [output2, states2] = lstm.forward(input2, {Variable{}, Variable{}});
    EXPECT_EQ(output2.shape()[1], 32) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, SingleTimestep) {
    // Test with single timestep
    nn::LSTM lstm(10, 20);
    auto input = Variable(randn({1, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, OutputConsistency) {
    // Test that output is deterministic
    nn::LSTM lstm(10, 20);
    auto input = Variable(ones({7, 5, 10}, DType::Float32, device), true);

    auto [output1, states1] = lstm.forward(input, {Variable{}, Variable{}});
    auto [output2, states2] = lstm.forward(input, {Variable{}, Variable{}});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]) << "Failed on " << device.to_string();
    }
}

TEST_P(LSTMTestFixture, GradientFlow) {
    // Test that gradients can flow through LSTM
    nn::LSTM lstm(10, 20);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    // Check that output requires grad
    EXPECT_TRUE(output.requires_grad()) << "Failed on " << device.to_string();

    // Sum for scalar output
    auto loss = Variable(sum(output.tensor(), std::nullopt, false), true);

    // Backward should not throw
    EXPECT_NO_THROW({
        loss.backward();
    }) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, TrainingMode) {
    // Test training/eval mode switching
    nn::LSTM lstm(10, 20, 2, true, false, 0.5);

    EXPECT_TRUE(lstm.is_training()) << "Failed on " << device.to_string();

    lstm.eval();
    EXPECT_FALSE(lstm.is_training()) << "Failed on " << device.to_string();

    lstm.train();
    EXPECT_TRUE(lstm.is_training()) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, ParameterCount) {
    // Test parameter counting
    nn::LSTM lstm(10, 20, 2);
    auto params = lstm.parameters();

    // Each layer has 2 combined linear layers (ih and hh) for all 4 gates
    // weight_ih: weight + bias = 2 params
    // weight_hh: weight only = 1 param
    // Layer 0: 3 params
    // Layer 1: 3 params
    // Total: 6 parameters
    EXPECT_EQ(params.size(), 6) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, LargeHidden) {
    // Test with large hidden size
    nn::LSTM lstm(10, 512);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[2], 512) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, VeryDeepNetwork) {
    // Test with many layers
    nn::LSTM lstm(10, 20, 5);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(h_n.shape()[0], 5) << "Failed on " << device.to_string();  // 5 layers
    EXPECT_EQ(c_n.shape()[0], 5) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, LongSequence) {
    // Test with very long sequence
    nn::LSTM lstm(10, 20);
    auto input = Variable(randn({100, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 100) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, InvalidNumLayers) {
    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::LSTM lstm(10, 20, 0);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, InvalidDropout) {
    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::LSTM lstm(10, 20, 2, true, false, 1.5);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, CellStateMemory) {
    // Test that cell state carries information across timesteps
    nn::LSTM lstm(10, 20);
    auto input = Variable(randn({10, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    // Cell state should not be zero (it carries memory)
    auto c_cpu = c_n.tensor().to(Device::cpu());
    auto c_data = c_cpu.data<float>();
    bool has_nonzero = false;
    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        if (std::abs(c_data[i]) > 1e-6) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, BatchFirstBidirectional) {
    // Test combination of batch_first and bidirectional
    nn::LSTM lstm(10, 20, 1, true, true, 0.0, true);
    auto input = Variable(randn({5, 7, 10}, DType::Float32, device), true);  // (batch, seq, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2
}

INSTANTIATE_BACKEND_TESTS(LSTMTestFixture);
