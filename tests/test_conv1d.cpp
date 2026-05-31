#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>
#include "backend_test_fixture.hpp"

using namespace tenzor;

class Conv1dTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        // Set random seed for reproducibility
        std::srand(42);
    }
};

// Test basic Conv1d forward pass with simple values
TEST_P(Conv1dTest, BasicForward) {
    // Create simple 1D input: [1, 2, 5] (batch=1, channels=2, length=5)
    auto input_tensor = randn({1, 2, 5}, DType::Float32, device);
    auto input = Variable(input_tensor, true);

    // Create Conv1d layer: 2 input channels, 3 output channels, kernel size 3
    auto conv = nn::Conv1d(2, 3, 3, 1, 0, 1, 1, false); // no bias for simplicity
    conv.to(device);

    // Forward pass
    auto output = conv.forward(input);

    // Expected output shape: [1, 3, 3]
    // L_out = floor((5 + 2*0 - 1*(3-1) - 1) / 1 + 1) = floor((5 - 2) / 1 + 1) = 3
    auto output_shape = output.shape();
    ASSERT_EQ(output_shape.size(), 3);
    EXPECT_EQ(output_shape[0], 1);  // batch
    EXPECT_EQ(output_shape[1], 3);  // out_channels
    EXPECT_EQ(output_shape[2], 3);  // length_out
}

// Test Conv1d with padding
TEST_P(Conv1dTest, WithPadding) {
    auto input_tensor = randn({2, 4, 10}, DType::Float32, device); // batch=2, channels=4, length=10
    auto input = Variable(input_tensor, true);

    auto conv = nn::Conv1d(4, 8, 3, 1, 1, 1, 1, true); // padding=1
    conv.to(device);

    auto output = conv.forward(input);

    // L_out = floor((10 + 2*1 - 1*(3-1) - 1) / 1 + 1) = floor((10 + 2 - 2) / 1 + 1) = 10
    auto output_shape = output.shape();
    ASSERT_EQ(output_shape.size(), 3);
    EXPECT_EQ(output_shape[0], 2);  // batch
    EXPECT_EQ(output_shape[1], 8);  // out_channels
    EXPECT_EQ(output_shape[2], 10); // length_out (same as input due to padding=1)
}

// Test Conv1d with stride
TEST_P(Conv1dTest, WithStride) {
    auto input_tensor = randn({1, 3, 16}, DType::Float32, device); // batch=1, channels=3, length=16
    auto input = Variable(input_tensor, true);

    auto conv = nn::Conv1d(3, 6, 3, 2, 0, 1, 1, false); // stride=2
    conv.to(device);

    auto output = conv.forward(input);

    // L_out = floor((16 + 2*0 - 1*(3-1) - 1) / 2 + 1) = floor((16 - 2) / 2 + 1) = 7 + 1 = 8
    auto output_shape = output.shape();
    ASSERT_EQ(output_shape.size(), 3);
    EXPECT_EQ(output_shape[0], 1);  // batch
    EXPECT_EQ(output_shape[1], 6);  // out_channels
    EXPECT_EQ(output_shape[2], 7);  // length_out (reduced due to stride)
}

// Test Conv1d with dilation
TEST_P(Conv1dTest, WithDilation) {
    auto input_tensor = randn({1, 2, 20}, DType::Float32, device); // batch=1, channels=2, length=20
    auto input = Variable(input_tensor, true);

    auto conv = nn::Conv1d(2, 4, 3, 1, 0, 2, 1, false); // dilation=2
    conv.to(device);

    auto output = conv.forward(input);

    // L_out = floor((20 + 2*0 - 2*(3-1) - 1) / 1 + 1) = floor((20 - 4) / 1 + 1) = 16 + 1 = 17
    auto output_shape = output.shape();
    ASSERT_EQ(output_shape.size(), 3);
    EXPECT_EQ(output_shape[0], 1);  // batch
    EXPECT_EQ(output_shape[1], 4);  // out_channels
    EXPECT_EQ(output_shape[2], 16); // length_out (reduced due to dilation)
}

// Test Conv1d with groups
TEST_P(Conv1dTest, WithGroups) {
    auto input_tensor = randn({2, 6, 12}, DType::Float32, device); // batch=2, channels=6, length=12
    auto input = Variable(input_tensor, true);

    auto conv = nn::Conv1d(6, 12, 3, 1, 1, 1, 2, true); // groups=2
    conv.to(device);

    auto output = conv.forward(input);

    // L_out = floor((12 + 2*1 - 1*(3-1) - 1) / 1 + 1) = floor((12 + 2 - 2) / 1 + 1) = 12
    auto output_shape = output.shape();
    ASSERT_EQ(output_shape.size(), 3);
    EXPECT_EQ(output_shape[0], 2);  // batch
    EXPECT_EQ(output_shape[1], 12); // out_channels
    EXPECT_EQ(output_shape[2], 12); // length_out
}

// Test Conv1d backward pass (gradient computation)
TEST_P(Conv1dTest, BackwardPass) {
    auto input_tensor = randn({2, 4, 10}, DType::Float32, device); // batch=2, channels=4, length=10
    auto input = Variable(input_tensor, true);

    auto conv = nn::Conv1d(4, 8, 3, 1, 1, 1, 1, true);
    conv.to(device);

    auto output = conv.forward(input);

    // Create gradient for output
    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec, DType::Float32, device);

    // Backward pass
    output.backward(grad_output);

    // Check that gradients were computed
    ASSERT_TRUE(input.grad().has_value());
    auto input_grad = input.grad().value();

    // Gradient should have same shape as input
    auto grad_shape = input_grad.shape();
    ASSERT_EQ(grad_shape.size(), 3);
    EXPECT_EQ(grad_shape[0], 2);  // batch
    EXPECT_EQ(grad_shape[1], 4);  // in_channels
    EXPECT_EQ(grad_shape[2], 10); // length

    // Check gradient values are not all zeros
    auto input_grad_cpu = input_grad.cpu();
    auto grad_data = input_grad_cpu.data<float>();
    bool has_nonzero = false;
    for (int64_t i = 0; i < input_grad_cpu.numel(); ++i) {
        if (std::abs(grad_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// Test Conv1d with bias
TEST_P(Conv1dTest, WithBias) {
    auto input_tensor = randn({1, 3, 8}, DType::Float32, device);
    auto input = Variable(input_tensor, false);

    auto conv = nn::Conv1d(3, 5, 3, 1, 1, 1, 1, true); // with bias
    conv.to(device);

    auto output = conv.forward(input);

    auto output_shape = output.shape();
    ASSERT_EQ(output_shape.size(), 3);
    EXPECT_EQ(output_shape[0], 1);  // batch
    EXPECT_EQ(output_shape[1], 5);  // out_channels
    EXPECT_EQ(output_shape[2], 8);  // length_out

    // Verify output is not empty
    EXPECT_GT(output.tensor().numel(), 0);
}

// Test Conv1d parameter shapes
TEST_P(Conv1dTest, ParameterShapes) {
    int64_t in_channels = 4;
    int64_t out_channels = 8;
    int64_t kernel_size = 5;

    auto conv = nn::Conv1d(in_channels, out_channels, kernel_size, 1, 0, 1, 1, true);
    conv.to(device);

    auto params = conv.parameters();

    // Should have weight and bias
    EXPECT_EQ(params.size(), 2);

    // Check weight shape: [out_channels, in_channels, kernel_size]
    auto weight_shape = params[0]->shape();
    EXPECT_EQ(weight_shape.size(), 3);
    EXPECT_EQ(weight_shape[0], out_channels);
    EXPECT_EQ(weight_shape[1], in_channels);
    EXPECT_EQ(weight_shape[2], kernel_size);

    // Check bias shape: [out_channels]
    auto bias_shape = params[1]->shape();
    EXPECT_EQ(bias_shape.size(), 1);
    EXPECT_EQ(bias_shape[0], out_channels);
}

// Test Conv1d without bias
TEST_P(Conv1dTest, WithoutBias) {
    auto conv = nn::Conv1d(3, 6, 3, 1, 0, 1, 1, false); // bias=false
    conv.to(device);

    auto params = conv.parameters();

    // Should have weight but no bias
    EXPECT_EQ(params.size(), 1);
}

// Test Conv1d input validation
TEST_P(Conv1dTest, InputValidation) {
    auto conv = nn::Conv1d(4, 8, 3, 1, 0, 1, 1, false);
    conv.to(device);

    // Test with wrong number of dimensions (should be 3D)
    auto wrong_input_2d = Variable(randn({4, 10}, DType::Float32, device), false);
    EXPECT_THROW(conv.forward(wrong_input_2d), std::invalid_argument);

    auto wrong_input_4d = Variable(randn({1, 4, 10, 10}, DType::Float32, device), false);
    EXPECT_THROW(conv.forward(wrong_input_4d), std::invalid_argument);

    // Test with wrong number of input channels
    auto wrong_channels = Variable(randn({1, 3, 10}, DType::Float32, device), false); // should be 4 channels
    EXPECT_THROW(conv.forward(wrong_channels), std::invalid_argument);
}

// Test Conv1d batch processing
TEST_P(Conv1dTest, BatchProcessing) {
    auto conv = nn::Conv1d(3, 6, 3, 1, 1, 1, 1, true);
    conv.to(device);

    // Test with different batch sizes
    for (int64_t batch_size : {1, 2, 4, 8}) {
        auto input = Variable(randn({batch_size, 3, 10}, DType::Float32, device), false);
        auto output = conv.forward(input);

        auto output_shape = output.shape();
        EXPECT_EQ(output_shape[0], batch_size);
        EXPECT_EQ(output_shape[1], 6);
        EXPECT_EQ(output_shape[2], 10);
    }
}

// Test Conv1d parameter count
TEST_P(Conv1dTest, ParameterCount) {
    auto conv_with_bias = nn::Conv1d(3, 6, 3, 1, 0, 1, 1, true);
    conv_with_bias.to(device);
    auto params_with = conv_with_bias.parameters();
    EXPECT_EQ(params_with.size(), 2);  // weight and bias

    auto conv_no_bias = nn::Conv1d(3, 6, 3, 1, 0, 1, 1, false);
    conv_no_bias.to(device);
    auto params_without = conv_no_bias.parameters();
    EXPECT_EQ(params_without.size(), 1);  // only weight
}

// Test Conv1d gradient computation for weight
TEST_P(Conv1dTest, WeightGradient) {
    auto input_tensor = randn({2, 3, 8}, DType::Float32, device);
    auto input = Variable(input_tensor, true);

    auto conv = nn::Conv1d(3, 5, 3, 1, 0, 1, 1, false);
    conv.to(device);

    auto output = conv.forward(input);
    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec, DType::Float32, device);
    output.backward(grad_output);

    // Check weight gradient
    auto params = conv.parameters();
    ASSERT_GE(params.size(), 1);
    auto weight_grad = params[0]->grad();
    ASSERT_TRUE(weight_grad.has_value());

    auto weight_grad_shape = weight_grad.value().shape();
    EXPECT_EQ(weight_grad_shape[0], 5);  // out_channels
    EXPECT_EQ(weight_grad_shape[1], 3);  // in_channels
    EXPECT_EQ(weight_grad_shape[2], 3);  // kernel_size
}

// Test Conv1d with sequence data (typical NLP use case)
TEST_P(Conv1dTest, SequenceData) {
    // Simulate word embeddings: batch=4, embedding_dim=128, sequence_length=50
    int64_t batch_size = 4;
    int64_t embedding_dim = 128;
    int64_t sequence_length = 50;

    auto input = Variable(randn({batch_size, embedding_dim, sequence_length}, DType::Float32, device), false);

    // Apply 1D convolution (common in text classification)
    auto conv = nn::Conv1d(embedding_dim, 256, 3, 1, 1, 1, 1, true);
    conv.to(device);
    auto output = conv.forward(input);

    auto output_shape = output.shape();
    EXPECT_EQ(output_shape[0], batch_size);
    EXPECT_EQ(output_shape[1], 256);
    EXPECT_EQ(output_shape[2], sequence_length); // same length due to padding=1
}

INSTANTIATE_BACKEND_TESTS(Conv1dTest);
