/**
 * @file test_severance_sweep.cpp
 * @brief Stream-11 (S11) severance sweep — regression coverage for the
 *        raw-tensor-op-inside-forward grad_fn-severance pattern.
 *
 * For each NN layer the audit flagged as carrying a `Variable(v.tensor().X,
 * v.requires_grad())` rewrap in its forward path, we exercise the autograd
 * round-trip end-to-end:
 *   1. Build the layer.
 *   2. Wrap a `requires_grad=true` input.
 *   3. Run forward.
 *   4. Sum, backward.
 *   5. Assert `EXPECT_GRAD_FLOWS(input)` — i.e. the chain reached the leaf
 *      with a non-zero gradient.
 *
 * Sites covered:
 *   - MaxPool2d, AvgPool2d (Pattern A: contig rewrap)
 *   - LayerNorm                (Pattern A: contig rewrap)
 *   - Conv2d                   (Pattern A: device-mismatch rewrap)
 *   - BatchNorm2d              (Pattern A: higher-order saved-variable rewrap)
 *   - LSTM eval-mode backward  (Pattern D: widened fast-path guard)
 *   - GRU eval-mode backward   (Pattern D: pre-existing strong guard — covered
 *                               for symmetry / regression)
 *   - LazyLinear               (Pattern C: backward stub is higher-order-only;
 *                               first-order grad flow still tested)
 *   - MultiheadAttention       (Pattern C: backward stub on type-cast; first-
 *                               order grad flow still tested)
 *
 * Sites in Pattern C (lazy_linear.cpp:42, attention.cpp:53) are intentional
 * severances inside `backward_with_variables` of a Function whose
 * `is_higher_order_stub()` returns true. They are *not* exercised here; the
 * higher-order-grad-mode machinery covers them.
 *
 * This is a grad-flow sweep parameterized across all backends via BackendTest:
 * every tensor is created on the fixture `device` and nn modules are moved
 * onto it. Inputs are randn (non-zero) so gradients are genuinely non-zero.
 */

#include <gtest/gtest.h>

#include "../grad_flow_helpers.hpp"
#include "../backend_test_fixture.hpp"

#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/lazy_linear.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"

using namespace tenzor;

class SeveranceSweep : public ::tenzor::testing::BackendTest {};

// ---------------------------------------------------------------------------
// MaxPool2d — Pattern A site (pooling.cpp:151 contig rewrap)
// ---------------------------------------------------------------------------
TEST_P(SeveranceSweep, MaxPool2dGradFlows) {
    nn::MaxPool2d layer(/*kernel_size=*/2, /*stride=*/2);
    layer.to(device);
    auto input = Variable(randn({2, 3, 8, 8}, DType::Float32, device), true);
    auto output = layer.forward_impl(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ---------------------------------------------------------------------------
// AvgPool2d — Pattern A site (pooling.cpp:436 contig rewrap)
// ---------------------------------------------------------------------------
TEST_P(SeveranceSweep, AvgPool2dGradFlows) {
    nn::AvgPool2d layer(/*kernel_size=*/2, /*stride=*/2);
    layer.to(device);
    auto input = Variable(randn({2, 3, 8, 8}, DType::Float32, device), true);
    auto output = layer.forward_impl(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ---------------------------------------------------------------------------
// LayerNorm — Pattern A site (normalization.cpp:830 contig rewrap)
// ---------------------------------------------------------------------------
TEST_P(SeveranceSweep, LayerNormGradFlows) {
    nn::LayerNorm layer({16});
    layer.to(device);
    auto input = Variable(randn({4, 16}, DType::Float32, device), true);
    auto output = layer.forward_impl(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ---------------------------------------------------------------------------
// Conv2d — Pattern A site (conv.cpp:308/317 device-mismatch rewrap).
// ---------------------------------------------------------------------------
TEST_P(SeveranceSweep, Conv2dGradFlows) {
    nn::Conv2d layer(/*in_channels=*/3, /*out_channels=*/4,
                     /*kernel_size=*/3, /*stride=*/1,
                     /*padding=*/1);
    layer.to(device);
    auto input = Variable(randn({2, 3, 8, 8}, DType::Float32, device), true);
    auto output = layer.forward_impl(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ---------------------------------------------------------------------------
// BatchNorm2d — Pattern A site (batchnorm.cpp:782/785 saved-variable rewrap
// for higher-order graphs).
// ---------------------------------------------------------------------------
TEST_P(SeveranceSweep, BatchNorm2dGradFlows) {
    nn::BatchNorm2d layer(/*num_features=*/4);
    layer.to(device);
    layer.train();
    auto input = Variable(randn({2, 4, 6, 6}, DType::Float32, device), true);
    auto output = layer.forward_impl(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ---------------------------------------------------------------------------
// LSTM eval-mode backward — Pattern D site.
//
// Before S11, `can_use_fused` only required `!is_training()`. An `eval()`
// network whose input still tracked grad would silently drop through the
// fast-path leaf-Variable rewrap; the user's backward call would then
// populate input.grad with zeros (or never run at all). S11 widens the
// guard to also require `!is_grad_enabled() && !input.requires_grad() &&
// all-params-no-grad`. The test below puts the layer in eval() but with a
// requires_grad input — the autograd-aware standard path must now be taken.
// ---------------------------------------------------------------------------
TEST_P(SeveranceSweep, LSTM_EvalMode_BackwardGradFlows) {
    int64_t seq_len = 4, batch = 2, input_size = 8, hidden_size = 8;
    nn::LSTM lstm(input_size, hidden_size, /*num_layers=*/1);
    lstm.to(device);
    lstm.eval();

    auto input = Variable(
        randn({seq_len, batch, input_size}, DType::Float32, device),
        true);

    Variable h0(zeros({1, batch, hidden_size}, DType::Float32, device),
                false);
    Variable c0(zeros({1, batch, hidden_size}, DType::Float32, device),
                false);

    auto [output, hc] = lstm.forward(input, {h0, c0});
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// ---------------------------------------------------------------------------
// GRU eval-mode backward — Pattern D site (already strong-guarded; pinned
// for regression).
// ---------------------------------------------------------------------------
TEST_P(SeveranceSweep, GRU_EvalMode_BackwardGradFlows) {
    int64_t seq_len = 4, batch = 2, input_size = 8, hidden_size = 8;
    nn::GRU gru(input_size, hidden_size, /*num_layers=*/1);
    gru.to(device);
    gru.eval();

    auto input = Variable(
        randn({seq_len, batch, input_size}, DType::Float32, device),
        true);

    Variable h0(zeros({1, batch, hidden_size}, DType::Float32, device),
                false);

    auto [output, h_final] = gru.forward(input, h0);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// ---------------------------------------------------------------------------
// LazyLinear — exercises the materialise-then-forward autograd path. The
// severance site at lazy_linear.cpp:42 lives in a higher-order stub
// backward (Pattern C); first-order grad flow must still work.
// ---------------------------------------------------------------------------
TEST_P(SeveranceSweep, LazyLinearGradFlows) {
    nn::LazyLinear layer(/*out_features=*/8);
    layer.to(device);
    auto input = Variable(randn({2, 16}, DType::Float32, device), true);
    auto output = layer.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ---------------------------------------------------------------------------
// MultiheadAttention — Pattern C site is in the backward stub; first-order
// grad flow through the layer's forward must still reach the input.
// ---------------------------------------------------------------------------
TEST_P(SeveranceSweep, MultiheadAttentionGradFlows) {
    int64_t embed_dim = 16, num_heads = 4, batch = 2, seq_len = 8;
    nn::MultiheadAttention mha(embed_dim, num_heads, /*dropout=*/0.0,
                               /*bias=*/true,
                               /*add_bias_kv=*/false,
                               /*add_zero_attn=*/false,
                               /*kdim=*/0, /*vdim=*/0,
                               /*batch_first=*/true,
                               /*is_causal=*/false);
    mha.to(device);
    mha.train(false);  // eval mode so the BMM path runs (per
                       // attention.cpp's grad_path_safe gating).

    auto x = Variable(
        randn({batch, seq_len, embed_dim}, DType::Float32, device),
        true);
    auto [out, weights] = mha.forward(x, x, x);
    auto loss = tenzor::sum(out);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);
}

INSTANTIATE_BACKEND_TESTS(SeveranceSweep);
