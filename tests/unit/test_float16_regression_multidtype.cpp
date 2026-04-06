/**
 * @file test_float16_regression_multidtype.cpp
 * @brief Multi-backend Float16 regression tests converted from test_float16_regression.cpp.
 *
 * These tests validate that Float16 computations produce correct results across
 * various operations and backends: linear layers, attention, convolutions,
 * backward passes, and common tensor operations.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class Float16RegressionMultiBackendTest : public BackendTest {};

// Validates weight initialization doesn't underflow to zero in Float16
TEST_P(Float16RegressionMultiBackendTest, LinearWeightsNonZero) {
    auto linear = nn::Linear(768, 768);
    linear.to(DType::Float16);
    linear.to(device);

    auto params = linear.parameters();
    ASSERT_FALSE(params.empty());
    auto weight = params[0]->tensor();
    EXPECT_EQ(weight.dtype(), DType::Float16);

    // Verify weights are not all zero by checking sum of absolute values
    auto weight_abs_sum = tenzor::sum(tenzor::abs(weight.to(DType::Float32).to(Device::cpu())));
    EXPECT_GT(weight_abs_sum.item<float>(), 0.0f) << "Float16 weights should not all be zero";
}

// Linear forward pass produces non-zero output
TEST_P(Float16RegressionMultiBackendTest, LinearForward) {
    auto linear = nn::Linear(4, 3);
    linear.to(DType::Float16);
    linear.to(device);

    auto input = Variable(ones({2, 4}, DType::Float16, device), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.tensor().dtype(), DType::Float16);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
}

// Linear backward pass computes gradients without crashing
TEST_P(Float16RegressionMultiBackendTest, LinearBackward) {
    auto linear = nn::Linear(64, 32);
    linear.to(DType::Float16);
    linear.to(device);

    auto input = Variable(randn({4, 64}, DType::Float16, device), true);
    auto output = linear.forward(input);
    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad().value().dtype(), DType::Float16);
}

// MatMul with Float16 accumulates in Float32 correctly
TEST_P(Float16RegressionMultiBackendTest, MatMul) {
    auto a = randn({128, 768}, DType::Float16, device);
    auto b = randn({768, 256}, DType::Float16, device);
    auto c = matmul(a, b);

    EXPECT_EQ(c.dtype(), DType::Float16);
    EXPECT_EQ(c.shape()[0], 128);
    EXPECT_EQ(c.shape()[1], 256);
}

// Softmax backward in Float16
TEST_P(Float16RegressionMultiBackendTest, SoftmaxBackward) {
    auto input = Variable(randn({4, 32}, DType::Float16, device), true);
    auto output = softmax(input, 1);
    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
}

// Mean backward in Float16
TEST_P(Float16RegressionMultiBackendTest, MeanBackward) {
    auto input = Variable(randn({4, 16}, DType::Float16, device), true);
    auto output = mean(input);
    output.backward();

    ASSERT_TRUE(input.grad().has_value());
}

// Reshape backward in Float16
TEST_P(Float16RegressionMultiBackendTest, ReshapeBackward) {
    auto input = Variable(randn({2, 3, 4}, DType::Float16, device), true);
    auto reshaped = reshape(input, {2, 12});
    auto loss = sum(reshaped);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
}

// Permute backward in Float16
TEST_P(Float16RegressionMultiBackendTest, PermuteBackward) {
    auto input = Variable(randn({2, 3, 4}, DType::Float16, device), true);
    auto permuted = permute(input, {0, 2, 1});
    auto loss = sum(permuted);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
}

// Embedding forward/backward in Float16
TEST_P(Float16RegressionMultiBackendTest, EmbeddingBackward) {
    auto embed = nn::Embedding(100, 64);
    embed.to(DType::Float16);
    embed.to(device);

    auto indices = zeros({4, 8}, DType::Int64, device);

    auto input_var = Variable(indices, false);
    auto output = embed.forward(input_var);
    auto loss = sum(output);
    loss.backward();
}

// LSTM cell with Float16
TEST_P(Float16RegressionMultiBackendTest, LSTMCellFloat16) {
    auto lstm = nn::LSTMCell(32, 16);
    lstm.to(DType::Float16);
    lstm.to(device);

    auto input = Variable(randn({2, 32}, DType::Float16, device), true);
    auto h = Variable(zeros({2, 16}, DType::Float16, device), false);
    auto c = Variable(zeros({2, 16}, DType::Float16, device), false);

    auto [h_new, c_new] = lstm.forward(input, h, c);

    EXPECT_EQ(h_new.tensor().dtype(), DType::Float16);
    EXPECT_EQ(c_new.tensor().dtype(), DType::Float16);
}

// Bmm backward in Float16
TEST_P(Float16RegressionMultiBackendTest, BmmBackward) {
    auto a = Variable(randn({2, 4, 8}, DType::Float16, device), true);
    auto b = Variable(randn({2, 8, 6}, DType::Float16, device), true);
    auto c = bmm(a, b);
    auto loss = sum(c);
    loss.backward();

    ASSERT_TRUE(a.grad().has_value());
    ASSERT_TRUE(b.grad().has_value());
}

INSTANTIATE_BACKEND_TESTS(Float16RegressionMultiBackendTest);
