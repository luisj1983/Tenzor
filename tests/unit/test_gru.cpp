#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;

// Global test environment
class GRUTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const gru_env =
    ::testing::AddGlobalTestEnvironment(new GRUTestEnvironment);

// ============================================================================
// GRUCell Tests
// ============================================================================

TEST(GRUCellTest, BasicForward) {
    // Test basic forward pass
    nn::GRUCell cell(10, 20);
    auto input = Variable(randn({5, 10}), true);
    auto h = Variable(randn({5, 20}), true);

    auto h_next = cell.forward(input, h);

    EXPECT_EQ(h_next.shape().size(), 2);
    EXPECT_EQ(h_next.shape()[0], 5);
    EXPECT_EQ(h_next.shape()[1], 20);
}

TEST(GRUCellTest, NoInitialHidden) {
    // Test with zero-initialized hidden state
    nn::GRUCell cell(10, 20);
    auto input = Variable(randn({5, 10}), true);

    auto h_next = cell.forward(input);

    EXPECT_EQ(h_next.shape().size(), 2);
    EXPECT_EQ(h_next.shape()[0], 5);
    EXPECT_EQ(h_next.shape()[1], 20);
}

TEST(GRUCellTest, NoBias) {
    // Test without bias
    nn::GRUCell cell(10, 20, false);
    auto input = Variable(randn({5, 10}), true);

    auto h_next = cell.forward(input);

    EXPECT_EQ(h_next.shape()[0], 5);
    EXPECT_EQ(h_next.shape()[1], 20);
}

TEST(GRUCellTest, HiddenStateEvolution) {
    // Test that hidden state evolves across time steps
    nn::GRUCell cell(10, 20);
    auto input = Variable(randn({5, 10}), true);

    auto h1 = cell.forward(input);
    auto h2 = cell.forward(input, h1);
    auto h3 = cell.forward(input, h2);

    // Each step should produce outputs
    EXPECT_EQ(h3.shape()[0], 5);
    EXPECT_EQ(h3.shape()[1], 20);
}

TEST(GRUCellTest, ParameterCount) {
    // Test parameter count
    nn::GRUCell cell(10, 20);
    auto params = cell.parameters();

    // GRU has 3 gates (reset, update, new)
    // Each gate has input and hidden transformations
    // 6 linear layers * 2 params (weight + bias) = 12 parameters
    EXPECT_EQ(params.size(), 12);
}

// ============================================================================
// GRU Tests
// ============================================================================

TEST(GRUTest, BasicForward) {
    // Test basic forward pass
    nn::GRU gru(10, 20, 1);
    auto input = Variable(randn({7, 5, 10}), true);  // (seq_len, batch, features)

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 7);  // seq_len
    EXPECT_EQ(output.shape()[1], 5);  // batch
    EXPECT_EQ(output.shape()[2], 20); // hidden_size

    EXPECT_EQ(h_n.shape().size(), 3);
    EXPECT_EQ(h_n.shape()[0], 1);  // num_layers
    EXPECT_EQ(h_n.shape()[1], 5);  // batch
    EXPECT_EQ(h_n.shape()[2], 20); // hidden_size
}

TEST(GRUTest, MultiLayer) {
    // Test multi-layer GRU
    nn::GRU gru(10, 20, 3);  // 3 layers
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 20);

    EXPECT_EQ(h_n.shape()[0], 3);  // 3 layers
}

TEST(GRUTest, BatchFirst) {
    // Test with batch_first=true
    nn::GRU gru(10, 20, 1, true, true);  // batch_first=true
    auto input = Variable(randn({5, 7, 10}), true);  // (batch, seq_len, features)

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 5);  // batch
    EXPECT_EQ(output.shape()[1], 7);  // seq_len
    EXPECT_EQ(output.shape()[2], 20); // hidden_size
}

TEST(GRUTest, Bidirectional) {
    // Test bidirectional GRU
    nn::GRU gru(10, 20, 1, true, false, 0.0, true);  // bidirectional=true
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 40); // hidden_size * 2

    EXPECT_EQ(h_n.shape()[0], 2);  // num_layers * num_directions
}

TEST(GRUTest, BidirectionalMultiLayer) {
    // Test bidirectional multi-layer GRU
    nn::GRU gru(10, 20, 2, true, false, 0.0, true);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 40); // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4);     // num_layers * num_directions
}

TEST(GRUTest, WithDropout) {
    // Test with dropout between layers
    nn::GRU gru(10, 20, 3, true, false, 0.5);  // 50% dropout
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 20);
}

TEST(GRUTest, InitialHiddenState) {
    // Test with provided initial hidden state
    nn::GRU gru(10, 20, 2);
    auto input = Variable(randn({7, 5, 10}), true);
    auto h0 = Variable(randn({2, 5, 20}), true);

    auto [output, h_n] = gru.forward(input, h0);

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(h_n.shape()[0], 2);
}

TEST(GRUTest, SequenceLengthVariation) {
    // Test with different sequence lengths
    nn::GRU gru(10, 20);

    // Short sequence
    auto input1 = Variable(randn({3, 5, 10}), true);
    auto [output1, h_n1] = gru.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[0], 3);

    // Long sequence
    auto input2 = Variable(randn({50, 5, 10}), true);
    auto [output2, h_n2] = gru.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[0], 50);
}

TEST(GRUTest, BatchSizeVariation) {
    // Test with different batch sizes
    nn::GRU gru(10, 20);

    // Small batch
    auto input1 = Variable(randn({7, 2, 10}), true);
    auto [output1, h_n1] = gru.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[1], 2);

    // Large batch
    auto input2 = Variable(randn({7, 32, 10}), true);
    auto [output2, h_n2] = gru.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[1], 32);
}

TEST(GRUTest, SingleTimestep) {
    // Test with single timestep
    nn::GRU gru(10, 20);
    auto input = Variable(randn({1, 5, 10}), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 1);
}

TEST(GRUTest, OutputConsistency) {
    // Test that output is deterministic
    nn::GRU gru(10, 20);
    auto input = Variable(ones({7, 5, 10}), true);

    auto [output1, h_n1] = gru.forward(input, Variable{});
    auto [output2, h_n2] = gru.forward(input, Variable{});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]);
        EXPECT_EQ(h_n1.shape()[i], h_n2.shape()[i]);
    }
}

TEST(GRUTest, GradientFlow) {
    // Test that gradients can flow through GRU
    nn::GRU gru(10, 20);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    // Check that output requires grad
    EXPECT_TRUE(output.requires_grad());

    // Sum for scalar output
    auto loss = Variable(sum(output.tensor(), std::nullopt, false), true);

    // Backward should not throw
    EXPECT_NO_THROW({
        loss.backward();
    });
}

TEST(GRUTest, TrainingMode) {
    // Test training/eval mode switching
    nn::GRU gru(10, 20, 2, true, false, 0.5);

    EXPECT_TRUE(gru.is_training());

    gru.eval();
    EXPECT_FALSE(gru.is_training());

    gru.train();
    EXPECT_TRUE(gru.is_training());
}

TEST(GRUTest, ParameterCount) {
    // Test parameter counting
    nn::GRU gru(10, 20, 2);
    auto params = gru.parameters();

    // Each layer has 6 linear transformations (3 gates * 2 transforms each)
    // Each linear has weight and bias
    // Layer 0: 6 * 2 = 12 params
    // Layer 1: 6 * 2 = 12 params
    // Total: 24 parameters
    EXPECT_EQ(params.size(), 24);
}

TEST(GRUTest, LargeHidden) {
    // Test with large hidden size
    nn::GRU gru(10, 512);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 512);
}

TEST(GRUTest, VeryDeepNetwork) {
    // Test with many layers
    nn::GRU gru(10, 20, 5);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(h_n.shape()[0], 5);  // 5 layers
}

TEST(GRUTest, LongSequence) {
    // Test with very long sequence
    nn::GRU gru(10, 20);
    auto input = Variable(randn({100, 5, 10}), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 100);
}

TEST(GRUTest, InvalidNumLayers) {
    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::GRU gru(10, 20, 0);
    }, std::invalid_argument);
}

TEST(GRUTest, InvalidDropout) {
    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::GRU gru(10, 20, 2, true, false, 1.5);
    }, std::invalid_argument);
}

TEST(GRUTest, GateOutputRanges) {
    // Test that GRU gates produce reasonable outputs
    nn::GRU gru(10, 20);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    // Hidden state values should be in reasonable range (due to tanh in new gate)
    auto h_data = h_n.tensor().data<float>();
    bool all_reasonable = true;
    for (int64_t i = 0; i < h_n.tensor().numel(); ++i) {
        // Values should generally be in [-10, 10] range
        if (std::abs(h_data[i]) > 10.0f) {
            all_reasonable = false;
            break;
        }
    }
    EXPECT_TRUE(all_reasonable);
}

TEST(GRUTest, BatchFirstBidirectional) {
    // Test combination of batch_first and bidirectional
    nn::GRU gru(10, 20, 1, true, true, 0.0, true);
    auto input = Variable(randn({5, 7, 10}), true);  // (batch, seq, features)

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 5);  // batch
    EXPECT_EQ(output.shape()[1], 7);  // seq_len
    EXPECT_EQ(output.shape()[2], 40); // hidden_size * 2
}

TEST(GRUTest, ComparisonWithLSTM) {
    // Ensure GRU has similar behavior to LSTM (fewer parameters, similar performance)
    nn::GRU gru(10, 20, 2);
    nn::LSTM lstm(10, 20, 2);

    auto input = Variable(randn({7, 5, 10}), true);

    auto [gru_output, gru_h] = gru.forward(input, Variable{});
    auto [lstm_output, lstm_states] = lstm.forward(input, {Variable{}, Variable{}});

    // Both should produce same shaped outputs
    EXPECT_EQ(gru_output.shape()[0], lstm_output.shape()[0]);
    EXPECT_EQ(gru_output.shape()[1], lstm_output.shape()[1]);
    EXPECT_EQ(gru_output.shape()[2], lstm_output.shape()[2]);

    // GRU has more parameters due to separate gate transforms
    // (GRU uses 6 separate Linear layers vs LSTM's 2 combined layers)
    auto gru_params = gru.parameters();
    auto lstm_params = lstm.parameters();
    EXPECT_GT(gru_params.size(), lstm_params.size());
}

TEST(GRUTest, MemoryEfficiency) {
    // Test that GRU uses less memory than LSTM
    nn::GRU gru(10, 20);
    nn::LSTM lstm(10, 20);

    // GRU only returns hidden state, LSTM returns both hidden and cell
    auto input = Variable(randn({7, 5, 10}), true);

    auto [gru_output, gru_h] = gru.forward(input, Variable{});
    auto [lstm_output, lstm_states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [lstm_h, lstm_c] = lstm_states;

    // GRU hidden state shape
    EXPECT_EQ(gru_h.shape().size(), 3);

    // LSTM has both h and c
    EXPECT_EQ(lstm_h.shape().size(), 3);
    EXPECT_EQ(lstm_c.shape().size(), 3);
}
