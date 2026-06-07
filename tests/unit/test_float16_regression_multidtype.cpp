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
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// audit-2 P.9 — This file uses BackendTest (not MultiBackendDTypeTest)
// deliberately: Float16 regression tests are dtype-specific by design — every
// case exercises a known Float16 numerical issue caught in
// audit-1 phase F. Parameterising over Float32/Float64 would
// mask the regression assertion.
class Float16RegressionMultiBackendTest : public BackendTest {};

// audit-3 T.1: per-element finiteness + at-least-one non-zero (BackendTest
// has no expectFiniteNonZero — local copy).
static void expect_finite_nonzero(const Tensor& t) {
    auto cpu = t.to(Device::cpu()).to(DType::Float32);
    const float* d = cpu.data<float>();
    bool any_nonzero = false;
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        ASSERT_TRUE(std::isfinite(d[i])) << "Non-finite at " << i;
        if (std::fabs(d[i]) > 1e-4f) any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero);
}

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
    expect_finite_nonzero(output.tensor());
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

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad().value().dtype(), DType::Float16);
    expect_finite_nonzero(input.grad().value());
}

// MatMul with Float16 accumulates in Float32 correctly
TEST_P(Float16RegressionMultiBackendTest, MatMul) {
    auto a = randn({128, 768}, DType::Float16, device);
    auto b = randn({768, 256}, DType::Float16, device);
    auto c = matmul(a, b);

    EXPECT_EQ(c.dtype(), DType::Float16);
    EXPECT_EQ(c.shape()[0], 128);
    EXPECT_EQ(c.shape()[1], 256);
    expect_finite_nonzero(c);
}

// Softmax backward in Float16
TEST_P(Float16RegressionMultiBackendTest, SoftmaxBackward) {
    auto input = Variable(randn({4, 32}, DType::Float16, device), true);
    auto output = softmax(input, 1);
    // sum(softmax(x)) == 1 identically, so its true input-grad is ZERO — a
    // backend computing the backward accurately fails the nonzero check on
    // pure round-off (oneAPI did). Square the output so the true gradient is
    // genuinely nonzero while still exercising the Float16 softmax backward.
    auto loss = sum(output * output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expect_finite_nonzero(input.grad().value());
}

// Mean backward in Float16
TEST_P(Float16RegressionMultiBackendTest, MeanBackward) {
    auto input = Variable(randn({4, 16}, DType::Float16, device), true);
    auto output = mean(input);
    output.backward();

    EXPECT_GRAD_FLOWS(input);
    expect_finite_nonzero(input.grad().value());
}

// Reshape backward in Float16
TEST_P(Float16RegressionMultiBackendTest, ReshapeBackward) {
    auto input = Variable(randn({2, 3, 4}, DType::Float16, device), true);
    auto reshaped = reshape(input, {2, 12});
    auto loss = sum(reshaped);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expect_finite_nonzero(input.grad().value());
}

// Permute backward in Float16
TEST_P(Float16RegressionMultiBackendTest, PermuteBackward) {
    auto input = Variable(randn({2, 3, 4}, DType::Float16, device), true);
    auto permuted = permute(input, {0, 2, 1});
    auto loss = sum(permuted);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expect_finite_nonzero(input.grad().value());
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
    // Embedding weights must receive a non-zero gradient; embedding output
    // must be finite (Float16 underflow has historically zeroed the weight
    // grad on this backend).
    expect_finite_nonzero(output.tensor());
    auto params = embed.parameters();
    ASSERT_FALSE(params.empty());
    EXPECT_TRUE(params[0]->has_grad());
    if (params[0]->has_grad()) {
        expect_finite_nonzero(params[0]->grad().value());
    }
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
    expect_finite_nonzero(h_new.tensor());
    expect_finite_nonzero(c_new.tensor());
}

// Bmm backward in Float16
TEST_P(Float16RegressionMultiBackendTest, BmmBackward) {
    auto a = Variable(randn({2, 4, 8}, DType::Float16, device), true);
    auto b = Variable(randn({2, 8, 6}, DType::Float16, device), true);
    auto c = bmm(a, b);
    auto loss = sum(c);
    loss.backward();

    EXPECT_GRAD_FLOWS(a);
    EXPECT_GRAD_FLOWS(b);
    expect_finite_nonzero(a.grad().value());
    expect_finite_nonzero(b.grad().value());
}

INSTANTIATE_BACKEND_TESTS(Float16RegressionMultiBackendTest);
