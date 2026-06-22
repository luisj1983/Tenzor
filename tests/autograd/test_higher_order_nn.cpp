/**
 * @file test_higher_order_nn.cpp
 * @brief Higher-order gradient tests for neural network layers
 *
 * Tests double-backward (gradient of gradient) through Conv2d, LSTM, and GRU.
 * Essential for meta-learning algorithms (MAML, Reptile) and WGAN-GP.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/nn/layers/sync_batchnorm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class HigherOrderNNTest : public BackendTest {};

// ============================================================================
// Conv2d Double Backward
// ============================================================================

TEST_P(HigherOrderNNTest, Conv2d_DoubleBackward) {
    // Compute gradient of the gradient norm w.r.t. input — requires a
    // second-order derivative through Conv2d. Reads input.grad_variable()
    // (populated by create_graph=true) rather than wrapping .grad() in a
    // fresh Variable, which would sever the backward graph.
    auto input = Variable(randn({1, 1, 4, 4}, DType::Float32, device), true);

    nn::Conv2d conv(1, 1, 3, 1, 1);

    auto output = conv.forward(input);
    auto loss = tenzor::sum(output);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    EXPECT_GRAD_FLOWS(input);
    ASSERT_TRUE(input.grad_variable().has_value())
        << "create_graph=true must populate grad_variable()";

    Variable grad_var = input.grad_variable().value();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();

    // The first-backward graph should be followable: back-propagating
    // through grad_norm touches input via the chain that backward_with_variables
    // built, so input ends up with a second-order gradient accumulated on it.
    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// LSTM Inner Loop (MAML-style)
// ============================================================================

TEST_P(HigherOrderNNTest, LSTM_InnerLoop_Gradient) {
    // MAML-style: compute gradient through an inner optimisation step. The
    // LSTM standard-path forward composes only Variable-level ops (matmul,
    // slice, sigmoid, tanh, mul, add, cat, unsqueeze) — each has a real
    // `backward_with_variables`, so `create_graph=true` produces a fully
    // connected second-order graph through the per-timestep cell.
    //
    // Force the training path (the fused/inference fast-paths in lstm.cpp
    // detach to raw Tensors and would mask the higher-order check).
    int64_t input_size = 4;
    int64_t hidden_size = 8;
    int64_t seq_len = 3;
    int64_t batch = 1;

    // The fused cuDNN LSTM train path (CudnnLSTMTrainBackward) does not implement
    // backward_with_variables and so cannot supply second derivatives. Force the
    // per-timestep Variable cell loop — the library's documented mechanism for the
    // double-backward case (see lstm.cpp) — so this higher-order test exercises a
    // real second-order graph on every backend (on CPU it already did).
    setenv("TENZOR_DISABLE_FUSED_LSTM_TRAIN", "1", 1);
    struct LstmEnvGuard { ~LstmEnvGuard() { unsetenv("TENZOR_DISABLE_FUSED_LSTM_TRAIN"); } } lstm_env_guard;

    auto input = Variable(randn({seq_len, batch, input_size}, DType::Float32, device), true);

    nn::LSTM lstm(input_size, hidden_size, 1);
    lstm.to(device);  // move weights/biases onto the test device (matches siblings)
    lstm.train();  // selects the Variable-graph path (see lstm.cpp can_use_fused)

    auto output = lstm.forward_impl(input);
    auto loss = tenzor::sum(output);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    EXPECT_GRAD_FLOWS(input);
    ASSERT_TRUE(input.grad_variable().has_value())
        << "LSTM create_graph=true must populate grad_variable()";

    // Second backward through the grad-variable. grad_norm depends on the
    // first-order gradient, so this exercises the LSTM cell's per-step
    // backward_with_variables chain.
    Variable grad_var = input.grad_variable().value();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// GRU Inner Loop (MAML-style)
// ============================================================================

TEST_P(HigherOrderNNTest, GRU_InnerLoop_Gradient) {
    // Same structure as LSTM_InnerLoop_Gradient — GRU's standard forward path
    // also stays on Variable ops end-to-end, so create_graph=true produces a
    // real second-order graph.
    int64_t input_size = 4;
    int64_t hidden_size = 8;
    int64_t seq_len = 3;
    int64_t batch = 1;

    // GRU's fused cuDNN train path likewise cannot supply second derivatives;
    // force the Variable cell loop (documented double-backward mechanism, see
    // gru.cpp) so the higher-order graph is real on every backend.
    setenv("TENZOR_DISABLE_FUSED_GRU_TRAIN", "1", 1);
    struct GruEnvGuard { ~GruEnvGuard() { unsetenv("TENZOR_DISABLE_FUSED_GRU_TRAIN"); } } gru_env_guard;

    auto input = Variable(randn({seq_len, batch, input_size}, DType::Float32, device), true);

    nn::GRU gru(input_size, hidden_size, 1);
    gru.to(device);  // move weights/biases onto the test device (matches siblings)
    gru.train();  // selects the Variable-graph forward (mirrors LSTM)

    auto output = gru.forward_impl(input);
    auto loss = tenzor::sum(output);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    EXPECT_GRAD_FLOWS(input);
    ASSERT_TRUE(input.grad_variable().has_value())
        << "GRU create_graph=true must populate grad_variable()";

    Variable grad_var = input.grad_variable().value();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// SyncBatchNorm Double Backward (single-process path)
// ============================================================================

TEST_P(HigherOrderNNTest, SyncBatchNorm_DoubleBackward) {
    // Single-process SyncBN (world_size=1) now has a Variable-level backward
    // so create_graph=true produces a real second-order graph — distributed
    // path still disconnects (flagged via is_higher_order_stub).
    auto identity_all_reduce = [](const Tensor& t) { return t; };
    nn::SyncBatchNorm sbn(4, identity_all_reduce, /*world_size=*/1);
    sbn.train();
    sbn.to(device);  // move running stats + weight/bias onto the test device

    auto input = Variable(randn({2, 4, 3, 3}, DType::Float32, device), true);
    auto output = sbn.forward(input);
    auto loss = tenzor::sum(output);

    // Audit: previously wrapped in try/catch -> GTEST_SKIP("SyncBatchNorm
    // unavailable"), which buried a real second-order SyncBN backward break
    // as a clean skip. Single-process SyncBN now has a Variable-level backward,
    // so this must produce a real second-order graph; let exceptions propagate.
    loss.backward(std::nullopt, false, /*create_graph=*/true);
    ASSERT_TRUE(input.grad_variable().has_value())
        << "SyncBatchNorm with create_graph=true must populate grad_variable()";
    Variable grad_var = input.grad_variable().value();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    EXPECT_GRAD_FLOWS(input);
}

INSTANTIATE_BACKEND_TESTS(HigherOrderNNTest);
