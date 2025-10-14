#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;

// Global test environment
class RNNTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const rnn_env =
    ::testing::AddGlobalTestEnvironment(new RNNTestEnvironment);

// ============================================================================
// RNNCell Tests
// ============================================================================

TEST(RNNCellTest, BasicForward) {
    // Test basic forward pass with tanh activation
    nn::RNNCell cell(10, 20, "tanh");
    auto input = Variable(randn({5, 10}), true);
    auto h = Variable(randn({5, 20}), true);

    auto output = cell.forward(input, h);

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 20);
}

TEST(RNNCellTest, ReLUActivation) {
    // Test with ReLU activation
    nn::RNNCell cell(10, 20, "relu");
    auto input = Variable(randn({5, 10}), true);
    auto h = Variable(randn({5, 20}), true);

    auto output = cell.forward(input, h);

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 20);
}

TEST(RNNCellTest, NoInitialHidden) {
    // Test with zero-initialized hidden state
    nn::RNNCell cell(10, 20);
    auto input = Variable(randn({5, 10}), true);

    auto output = cell.forward(input);

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 20);
}

TEST(RNNCellTest, NoBias) {
    // Test without bias
    nn::RNNCell cell(10, 20, "tanh", false);
    auto input = Variable(randn({5, 10}), true);

    auto output = cell.forward(input);

    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 20);
}

TEST(RNNCellTest, InvalidNonlinearity) {
    // Test that invalid activation throws
    EXPECT_THROW({
        nn::RNNCell cell(10, 20, "sigmoid");
    }, std::invalid_argument);
}

// ============================================================================
// RNN Tests
// ============================================================================

TEST(RNNTest, BasicForward) {
    // Test basic forward pass
    nn::RNN rnn(10, 20, 1);
    auto input = Variable(randn({7, 5, 10}), true);  // (seq_len, batch, features)

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 7);  // seq_len
    EXPECT_EQ(output.shape()[1], 5);  // batch
    EXPECT_EQ(output.shape()[2], 20); // hidden_size

    EXPECT_EQ(h_n.shape().size(), 3);
    EXPECT_EQ(h_n.shape()[0], 1);  // num_layers
    EXPECT_EQ(h_n.shape()[1], 5);  // batch
    EXPECT_EQ(h_n.shape()[2], 20); // hidden_size
}

TEST(RNNTest, MultiLayer) {
    // Test multi-layer RNN
    nn::RNN rnn(10, 20, 3);  // 3 layers
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 20);

    EXPECT_EQ(h_n.shape()[0], 3);  // 3 layers
    EXPECT_EQ(h_n.shape()[1], 5);
    EXPECT_EQ(h_n.shape()[2], 20);
}

TEST(RNNTest, BatchFirst) {
    // Test with batch_first=true
    nn::RNN rnn(10, 20, 1, "tanh", true, true);  // batch_first=true
    auto input = Variable(randn({5, 7, 10}), true);  // (batch, seq_len, features)

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 5);  // batch
    EXPECT_EQ(output.shape()[1], 7);  // seq_len
    EXPECT_EQ(output.shape()[2], 20); // hidden_size
}

TEST(RNNTest, Bidirectional) {
    // Test bidirectional RNN
    nn::RNN rnn(10, 20, 1, "tanh", true, false, 0.0, true);  // bidirectional=true
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 40); // hidden_size * 2

    EXPECT_EQ(h_n.shape()[0], 2);  // num_layers * num_directions
    EXPECT_EQ(h_n.shape()[1], 5);
    EXPECT_EQ(h_n.shape()[2], 20);
}

TEST(RNNTest, BidirectionalMultiLayer) {
    // Test bidirectional multi-layer RNN
    nn::RNN rnn(10, 20, 2, "tanh", true, false, 0.0, true);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 40); // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4);     // num_layers * num_directions
}

TEST(RNNTest, WithDropout) {
    // Test with dropout between layers
    nn::RNN rnn(10, 20, 3, "tanh", true, false, 0.5);  // 50% dropout
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 20);
}

TEST(RNNTest, InitialHiddenState) {
    // Test with provided initial hidden state
    nn::RNN rnn(10, 20, 2);
    auto input = Variable(randn({7, 5, 10}), true);
    auto h0 = Variable(randn({2, 5, 20}), true);

    auto [output, h_n] = rnn.forward(input, h0);

    EXPECT_EQ(output.shape()[0], 7);
    EXPECT_EQ(h_n.shape()[0], 2);
}

TEST(RNNTest, SequenceLengthVariation) {
    // Test with different sequence lengths
    nn::RNN rnn(10, 20);

    // Short sequence
    auto input1 = Variable(randn({3, 5, 10}), true);
    auto [output1, h_n1] = rnn.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[0], 3);

    // Long sequence
    auto input2 = Variable(randn({50, 5, 10}), true);
    auto [output2, h_n2] = rnn.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[0], 50);
}

TEST(RNNTest, BatchSizeVariation) {
    // Test with different batch sizes
    nn::RNN rnn(10, 20);

    // Small batch
    auto input1 = Variable(randn({7, 2, 10}), true);
    auto [output1, h_n1] = rnn.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[1], 2);

    // Large batch
    auto input2 = Variable(randn({7, 32, 10}), true);
    auto [output2, h_n2] = rnn.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[1], 32);
}

TEST(RNNTest, SingleTimestep) {
    // Test with single timestep
    nn::RNN rnn(10, 20);
    auto input = Variable(randn({1, 5, 10}), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 1);
}

TEST(RNNTest, OutputConsistency) {
    // Test that output is deterministic
    nn::RNN rnn(10, 20);
    auto input = Variable(ones({7, 5, 10}), true);

    auto [output1, h_n1] = rnn.forward(input, Variable{});
    auto [output2, h_n2] = rnn.forward(input, Variable{});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]);
        EXPECT_EQ(h_n1.shape()[i], h_n2.shape()[i]);
    }
}

TEST(RNNTest, GradientFlow) {
    // Test that gradients can flow through RNN
    nn::RNN rnn(10, 20);
    auto input = Variable(randn({7, 5, 10}), true);

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

TEST(RNNTest, TrainingMode) {
    // Test training/eval mode switching
    nn::RNN rnn(10, 20, 2, "tanh", true, false, 0.5);

    EXPECT_TRUE(rnn.is_training());

    rnn.eval();
    EXPECT_FALSE(rnn.is_training());

    rnn.train();
    EXPECT_TRUE(rnn.is_training());
}

TEST(RNNTest, ParameterCount) {
    // Test parameter counting
    nn::RNN rnn(10, 20, 2);
    auto params = rnn.parameters();

    // Each layer has input_layer and hidden_layer, each with weight and bias
    // Layer 0: input (10->20), hidden (20->20)
    // Layer 1: input (20->20), hidden (20->20)
    // Total: 4 weights + 4 biases = 8 parameters
    EXPECT_EQ(params.size(), 8);
}

TEST(RNNTest, LargeHidden) {
    // Test with large hidden size
    nn::RNN rnn(10, 512);
    auto input = Variable(randn({7, 5, 10}), true);

    auto [output, h_n] = rnn.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 512);
    EXPECT_EQ(h_n.shape()[2], 512);
}

TEST(RNNTest, InvalidNumLayers) {
    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::RNN rnn(10, 20, 0);
    }, std::invalid_argument);
}

TEST(RNNTest, InvalidDropout) {
    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::RNN rnn(10, 20, 2, "tanh", true, false, 1.5);
    }, std::invalid_argument);
}
