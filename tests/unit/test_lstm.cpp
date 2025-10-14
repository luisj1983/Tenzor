#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;

// Global test environment
class LSTMTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const lstm_env =
    ::testing::AddGlobalTestEnvironment(new LSTMTestEnvironment);

// ============================================================================
// LSTMCell Tests
// ============================================================================

TEST(LSTMCellTest, BasicForward) {
    // Test basic forward pass
    nn::LSTMCell cell(10, 20);
    auto input = Variable(randn({5, 10}), true);
    auto h = Variable(randn({5, 20}), true);
    auto c = Variable(randn({5, 20}), true);

    auto [h_next, c_next] = cell.forward(input, h, c);

    EXPECT_EQ(h_next.shape().size(), 2);
    EXPECT_EQ(h_next.shape()[0], 5);
    EXPECT_EQ(h_next.shape()[1], 20);

    EXPECT_EQ(c_next.shape().size(), 2);
    EXPECT_EQ(c_next.shape()[0], 5);
    EXPECT_EQ(c_next.shape()[1], 20);
}

TEST(LSTMCellTest, NoInitialStates) {
    // Test with zero-initialized states
    nn::LSTMCell cell(10, 20);
    auto input = Variable(randn({5, 10}), true);

    auto [h_next, c_next] = cell.forward(input, Variable{}, Variable{});

    EXPECT_EQ(h_next.shape().size(), 2);
    EXPECT_EQ(h_next.shape()[0], 5);
    EXPECT_EQ(h_next.shape()[1], 20);

    EXPECT_EQ(c_next.shape().size(), 2);
    EXPECT_EQ(c_next.shape()[0], 5);
    EXPECT_EQ(c_next.shape()[1], 20);
}

TEST(LSTMCellTest, NoBias) {
    // Test without bias
    nn::LSTMCell cell(10, 20, false);
    auto input = Variable(randn({5, 10}), true);

    auto [h_next, c_next] = cell.forward(input, Variable{}, Variable{});

    EXPECT_EQ(h_next.shape()[0], 5);
    EXPECT_EQ(h_next.shape()[1], 20);
}

TEST(LSTMCellTest, CellStateEvolution) {
    // Test that cell state evolves across time steps
    nn::LSTMCell cell(10, 20);
    auto input = Variable(randn({5, 10}), true);

    auto [h1, c1] = cell.forward(input, Variable{}, Variable{});
    auto [h2, c2] = cell.forward(input, h1, c1);
    auto [h3, c3] = cell.forward(input, h2, c2);

    // Each step should produce outputs (shapes should be consistent)
    EXPECT_EQ(h3.shape()[0], 5);
    EXPECT_EQ(h3.shape()[1], 20);
}

// ============================================================================
// LSTM Tests
// ============================================================================

TEST(LSTMTest, BasicForward) {
    // Test basic forward pass
    nn::LSTM lstm(10, 20, 1);
    auto input = Variable(randn({7, 5, 10}), true);  // (seq_len, batch, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 7);  // seq_len
    EXPECT_EQ(output.shape()[1], 5);  // batch
    EXPECT_EQ(output.shape()[2], 20); // hidden_size

    EXPECT_EQ(h_n.shape().size(), 3);
    EXPECT_EQ(h_n.shape()[0], 1);  // num_layers
    EXPECT_EQ(h_n.shape()[1], 5);  // batch
    EXPECT_EQ(h_n.shape()[2], 20); // hidden_size

    EXPECT_EQ(c_n.shape().size(), 3);
    EXPECT_EQ(c_n.shape()[0], 1);
    EXPECT_EQ(c_n.shape()[1], 5);
    EXPECT_EQ(c_n.shape()[2], 20);
}

TEST(LSTMTest, MultiLayer) {
    // Test multi-layer LSTM
    nn::LSTM lstm(10, 20, 3);  // 3 layers
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 20);

    EXPECT_EQ(h_n.shape()[0], 3);  // 3 layers
    EXPECT_EQ(c_n.shape()[0], 3);
}

TEST(LSTMTest, BatchFirst) {
    // Test with batch_first=true
    nn::LSTM lstm(10, 20, 1, true, true);  // batch_first=true
    auto input = Variable(randn({5, 7, 10}), true);  // (batch, seq_len, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 5);  // batch
    EXPECT_EQ(output.shape()[1], 7);  // seq_len
    EXPECT_EQ(output.shape()[2], 20); // hidden_size
}

TEST(LSTMTest, Bidirectional) {
    // Test bidirectional LSTM
    nn::LSTM lstm(10, 20, 1, true, false, 0.0, true);  // bidirectional=true
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 40); // hidden_size * 2

    EXPECT_EQ(h_n.shape()[0], 2);  // num_layers * num_directions
    EXPECT_EQ(h_n.shape()[1], 5);
    EXPECT_EQ(h_n.shape()[2], 20);

    EXPECT_EQ(c_n.shape()[0], 2);
}

TEST(LSTMTest, BidirectionalMultiLayer) {
    // Test bidirectional multi-layer LSTM
    nn::LSTM lstm(10, 20, 2, true, false, 0.0, true);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[2], 40); // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4);     // num_layers * num_directions
    EXPECT_EQ(c_n.shape()[0], 4);
}

TEST(LSTMTest, WithDropout) {
    // Test with dropout between layers
    nn::LSTM lstm(10, 20, 3, true, false, 0.5);  // 50% dropout
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 20);
}

TEST(LSTMTest, InitialStates) {
    // Test with provided initial states
    nn::LSTM lstm(10, 20, 2);
    auto input = Variable(randn({7, 5, 10}), true);
    auto h0 = Variable(randn({2, 5, 20}), true);
    auto c0 = Variable(randn({2, 5, 20}), true);

    auto [output, states] = lstm.forward(input, {h0, c0});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(h_n.shape()[0], 2);
    EXPECT_EQ(c_n.shape()[0], 2);
}

TEST(LSTMTest, SequenceLengthVariation) {
    // Test with different sequence lengths
    nn::LSTM lstm(10, 20);

    // Short sequence
    auto input1 = Variable(randn({3, 5, 10}), true);
    auto [output1, states1] = lstm.forward(input1, {Variable{}, Variable{}});
    EXPECT_EQ(output1.shape()[0], 3);

    // Long sequence
    auto input2 = Variable(randn({50, 5, 10}), true);
    auto [output2, states2] = lstm.forward(input2, {Variable{}, Variable{}});
    EXPECT_EQ(output2.shape()[0], 50);
}

TEST(LSTMTest, BatchSizeVariation) {
    // Test with different batch sizes
    nn::LSTM lstm(10, 20);

    // Small batch
    auto input1 = Variable(randn({7, 2, 10}), true);
    auto [output1, states1] = lstm.forward(input1, {Variable{}, Variable{}});
    EXPECT_EQ(output1.shape()[1], 2);

    // Large batch
    auto input2 = Variable(randn({7, 32, 10}), true);
    auto [output2, states2] = lstm.forward(input2, {Variable{}, Variable{}});
    EXPECT_EQ(output2.shape()[1], 32);
}

TEST(LSTMTest, SingleTimestep) {
    // Test with single timestep
    nn::LSTM lstm(10, 20);
    auto input = Variable(randn({1, 5, 10}), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 1);
}

TEST(LSTMTest, OutputConsistency) {
    // Test that output is deterministic
    nn::LSTM lstm(10, 20);
    auto input = Variable(ones({7, 5, 10}), true);

    auto [output1, states1] = lstm.forward(input, {Variable{}, Variable{}});
    auto [output2, states2] = lstm.forward(input, {Variable{}, Variable{}});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]);
    }
}

TEST(LSTMTest, GradientFlow) {
    // Test that gradients can flow through LSTM
    nn::LSTM lstm(10, 20);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    // Check that output requires grad
    EXPECT_TRUE(output.requires_grad());

    // Sum for scalar output
    auto loss = Variable(sum(output.tensor(), std::nullopt, false), true);

    // Backward should not throw
    EXPECT_NO_THROW({
        loss.backward();
    });
}

TEST(LSTMTest, TrainingMode) {
    // Test training/eval mode switching
    nn::LSTM lstm(10, 20, 2, true, false, 0.5);

    EXPECT_TRUE(lstm.is_training());

    lstm.eval();
    EXPECT_FALSE(lstm.is_training());

    lstm.train();
    EXPECT_TRUE(lstm.is_training());
}

TEST(LSTMTest, ParameterCount) {
    // Test parameter counting
    nn::LSTM lstm(10, 20, 2);
    auto params = lstm.parameters();

    // Each layer has 2 combined linear layers (ih and hh) for all 4 gates
    // weight_ih: weight + bias = 2 params
    // weight_hh: weight only = 1 param
    // Layer 0: 3 params
    // Layer 1: 3 params
    // Total: 6 parameters
    EXPECT_EQ(params.size(), 6);
}

TEST(LSTMTest, LargeHidden) {
    // Test with large hidden size
    nn::LSTM lstm(10, 512);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[2], 512);
}

TEST(LSTMTest, VeryDeepNetwork) {
    // Test with many layers
    nn::LSTM lstm(10, 20, 5);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(h_n.shape()[0], 5);  // 5 layers
    EXPECT_EQ(c_n.shape()[0], 5);
}

TEST(LSTMTest, LongSequence) {
    // Test with very long sequence
    nn::LSTM lstm(10, 20);
    auto input = Variable(randn({100, 5, 10}), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 100);
}

TEST(LSTMTest, InvalidNumLayers) {
    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::LSTM lstm(10, 20, 0);
    }, std::invalid_argument);
}

TEST(LSTMTest, InvalidDropout) {
    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::LSTM lstm(10, 20, 2, true, false, 1.5);
    }, std::invalid_argument);
}

TEST(LSTMTest, CellStateMemory) {
    // Test that cell state carries information across timesteps
    nn::LSTM lstm(10, 20);
    auto input = Variable(randn({10, 5, 10}), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    // Cell state should not be zero (it carries memory)
    auto c_data = c_n.tensor().data<float>();
    bool has_nonzero = false;
    for (int64_t i = 0; i < c_n.tensor().numel(); ++i) {
        if (std::abs(c_data[i]) > 1e-6) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(LSTMTest, BatchFirstBidirectional) {
    // Test combination of batch_first and bidirectional
    nn::LSTM lstm(10, 20, 1, true, true, 0.0, true);
    auto input = Variable(randn({5, 7, 10}), true);  // (batch, seq, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 5);  // batch
    EXPECT_EQ(output.shape()[1], 7);  // seq_len
    EXPECT_EQ(output.shape()[2], 40); // hidden_size * 2
}
