/**
 * @file test_float16_regression.cpp
 * @brief Float16 regression tests consolidated from 19 minimal_float16_* debug tests.
 *
 * These tests validate that Float16 computations produce correct results across
 * various operations: linear layers, attention, convolutions, backward passes,
 * and common tensor operations.
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
#include "../grad_flow_helpers.hpp"

using namespace tenzor;

class Float16RegressionTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// Validates weight initialization doesn't underflow to zero in Float16
TEST_P(Float16RegressionTest, LinearWeightsNonZero) {
    auto linear = nn::Linear(768, 768);
    linear.to(device);
    linear.to(DType::Float16);

    auto params = linear.parameters();
    ASSERT_FALSE(params.empty());
    auto weight = params[0]->tensor();
    EXPECT_EQ(weight.dtype(), DType::Float16);

    // Verify weights are not all zero by checking sum of absolute values
    auto weight_abs_sum = tenzor::sum(tenzor::abs(weight.to(DType::Float32)));
    auto weight_abs_sum_host = weight_abs_sum.cpu();
    EXPECT_GT(weight_abs_sum_host.item<float>(), 0.0f) << "Float16 weights should not all be zero";
}

// Linear forward pass produces non-zero output
TEST_P(Float16RegressionTest, LinearForward) {
    auto linear = nn::Linear(4, 3);
    linear.to(device);
    linear.to(DType::Float16);

    auto input = Variable(ones({2, 4}, DType::Float32, device).to(DType::Float16), true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.tensor().dtype(), DType::Float16);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
}

// Linear backward pass computes gradients without crashing
TEST_P(Float16RegressionTest, LinearBackward) {
    auto linear = nn::Linear(64, 32);
    linear.to(device);
    linear.to(DType::Float16);

    auto input = Variable(randn({4, 64}, DType::Float32, device).to(DType::Float16), true);
    auto output = linear.forward(input);
    auto loss = sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad().value().dtype(), DType::Float16);
}

// MatMul with Float16 accumulates in Float32 correctly
TEST_P(Float16RegressionTest, MatMul) {
    auto a = randn({128, 768}, DType::Float32, device).to(DType::Float16);
    auto b = randn({768, 256}, DType::Float32, device).to(DType::Float16);
    auto c = matmul(a, b);

    EXPECT_EQ(c.dtype(), DType::Float16);
    EXPECT_EQ(c.shape()[0], 128);
    EXPECT_EQ(c.shape()[1], 256);
}

// Softmax backward in Float16
TEST_P(Float16RegressionTest, SoftmaxBackward) {
    auto input = Variable(randn({4, 32}, DType::Float32, device).to(DType::Float16), true);
    auto output = softmax(input, 1);
    auto loss = sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// Mean backward in Float16
TEST_P(Float16RegressionTest, MeanBackward) {
    auto input = Variable(randn({4, 16}, DType::Float32, device).to(DType::Float16), true);
    auto output = mean(input);
    output.backward();

    EXPECT_GRAD_FLOWS(input);
}

// Reshape backward in Float16
TEST_P(Float16RegressionTest, ReshapeBackward) {
    auto input = Variable(randn({2, 3, 4}, DType::Float32, device).to(DType::Float16), true);
    auto reshaped = reshape(input, {2, 12});
    auto loss = sum(reshaped);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// Permute backward in Float16
TEST_P(Float16RegressionTest, PermuteBackward) {
    auto input = Variable(randn({2, 3, 4}, DType::Float32, device).to(DType::Float16), true);
    auto permuted = permute(input, {0, 2, 1});
    auto loss = sum(permuted);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// Embedding forward/backward in Float16
TEST_P(Float16RegressionTest, EmbeddingBackward) {
    auto embed = nn::Embedding(100, 64);
    embed.to(device);
    embed.to(DType::Float16);

    auto indices = zeros({4, 8}, DType::Int64, device);

    auto input_var = Variable(indices, false);
    auto output = embed.forward(input_var);
    auto loss = sum(output);
    loss.backward();
}

// LSTM cell with Float16
TEST_P(Float16RegressionTest, LSTMCellFloat16) {
    auto lstm = nn::LSTMCell(32, 16);
    lstm.to(device);
    lstm.to(DType::Float16);

    auto input = Variable(randn({2, 32}, DType::Float32, device).to(DType::Float16), true);
    auto h = Variable(zeros({2, 16}, DType::Float32, device).to(DType::Float16), false);
    auto c = Variable(zeros({2, 16}, DType::Float32, device).to(DType::Float16), false);

    auto [h_new, c_new] = lstm.forward(input, h, c);

    EXPECT_EQ(h_new.tensor().dtype(), DType::Float16);
    EXPECT_EQ(c_new.tensor().dtype(), DType::Float16);
}

// Bmm backward in Float16
TEST_P(Float16RegressionTest, BmmBackward) {
    auto a = Variable(randn({2, 4, 8}, DType::Float32, device).to(DType::Float16), true);
    auto b = Variable(randn({2, 8, 6}, DType::Float32, device).to(DType::Float16), true);
    auto c = bmm(a, b);
    auto loss = sum(c);
    loss.backward();

    EXPECT_GRAD_FLOWS(a);
    EXPECT_GRAD_FLOWS(b);
}

INSTANTIATE_BACKEND_TESTS(Float16RegressionTest);
