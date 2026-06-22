/**
 * @file test_higher_order_nn_multidtype.cpp
 * @brief Multi-backend + multi-dtype higher-order gradient tests for NN layers
 *
 * Multi-backend port of test_higher_order_nn.cpp. Tests double-backward
 * (gradient of gradient) through Conv2d, LSTM, and GRU across all available
 * backends and data types.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class HigherOrderNNMultiDTypeTest : public MultiBackendDTypeTest {};

// Higher-order gradient tests are numerically demanding — skip Float16
#define SKIP_IF_LOW_PRECISION() \
    do { \
        if (dtype() == DType::Float16) \
            GTEST_SKIP() << "Higher-order grads require Float32+ precision"; \
    } while (0)

// ============================================================================
// Conv2d Double Backward
// ============================================================================

TEST_P(HigherOrderNNMultiDTypeTest, Conv2d_DoubleBackward) {
    SKIP_IF_LOW_PRECISION();

    // Test: compute gradient of the gradient norm w.r.t. input
    auto input = Variable(createRandn({1, 1, 4, 4}), true);

    nn::Conv2d conv(1, 1, 3, 1, 1);  // 1->1 channel, 3x3 kernel, stride 1, pad 1
    convert_model(conv);

    // Forward
    auto output = conv.forward(input);
    auto loss = tenzor::sum(output);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    ASSERT_TRUE(input.grad_variable().has_value())
        << "create_graph=true must populate grad_variable()";
    Variable grad_var = input.grad_variable().value();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// LSTM Inner Loop (MAML-style)
// ============================================================================

TEST_P(HigherOrderNNMultiDTypeTest, LSTM_InnerLoop_Gradient) {
    SKIP_IF_LOW_PRECISION();

    int64_t input_size = 4;
    int64_t hidden_size = 8;
    int64_t seq_len = 3;
    int64_t batch = 1;

    // The fused cuDNN LSTM train path cannot supply second derivatives; force the
    // Variable cell loop (documented double-backward mechanism, see lstm.cpp) so
    // the Float32/CUDA case exercises a real second-order graph like the others.
    setenv("TENZOR_DISABLE_FUSED_LSTM_TRAIN", "1", 1);
    struct LstmEnvGuard { ~LstmEnvGuard() { unsetenv("TENZOR_DISABLE_FUSED_LSTM_TRAIN"); } } lstm_env_guard;

    auto input = Variable(createRandn({seq_len, batch, input_size}), true);

    nn::LSTM lstm(input_size, hidden_size, 1);
    convert_model(lstm);

    // Forward through LSTM
    auto output = lstm.forward_impl(input);

    // Simple loss: sum of output
    auto loss = tenzor::sum(output);

    // Audit: previously wrapped in try/catch -> GTEST_SKIP("LSTM double
    // backward not supported"), which reclassified a real second-order
    // autograd break as a clean skip. Let exceptions propagate so a missing
    // LSTM double-backward surfaces as a failure.
    loss.backward(std::nullopt, false, true);
    ASSERT_TRUE(input.grad_variable().has_value())
        << "LSTM create_graph=true must populate grad_variable()";
    Variable grad_var = input.grad_variable().value();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// GRU Inner Loop (MAML-style)
// ============================================================================

TEST_P(HigherOrderNNMultiDTypeTest, GRU_InnerLoop_Gradient) {
    SKIP_IF_LOW_PRECISION();

    int64_t input_size = 4;
    int64_t hidden_size = 8;
    int64_t seq_len = 3;
    int64_t batch = 1;

    // GRU's fused cuDNN train path cannot supply second derivatives; force the
    // Variable cell loop (documented double-backward mechanism, see gru.cpp).
    setenv("TENZOR_DISABLE_FUSED_GRU_TRAIN", "1", 1);
    struct GruEnvGuard { ~GruEnvGuard() { unsetenv("TENZOR_DISABLE_FUSED_GRU_TRAIN"); } } gru_env_guard;

    auto input = Variable(createRandn({seq_len, batch, input_size}), true);

    nn::GRU gru(input_size, hidden_size, 1);
    convert_model(gru);

    auto output = gru.forward_impl(input);
    auto loss = tenzor::sum(output);

    // Audit: previously wrapped in try/catch -> GTEST_SKIP("GRU double
    // backward not supported"), burying a real second-order autograd break
    // as a clean skip. Let exceptions propagate so a missing GRU
    // double-backward surfaces as a failure.
    loss.backward(std::nullopt, false, true);
    ASSERT_TRUE(input.grad_variable().has_value())
        << "GRU create_graph=true must populate grad_variable()";
    Variable grad_var = input.grad_variable().value();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    EXPECT_GRAD_FLOWS(input);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HigherOrderNNMultiDTypeTest);
