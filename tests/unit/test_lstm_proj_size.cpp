// test_lstm_proj_size.cpp
//
// Audit G1: nn::LSTM with `proj_size > 0` (PyTorch LSTMP) must
//   (a) construct without throwing,
//   (b) produce h_n with trailing dim == proj_size (not hidden_size),
//   (c) accept h0 with proj_size trailing dim,
//   (d) reject proj_size >= hidden_size (PyTorch convention),
//   (e) flow gradients back to the input,
//   (f) work multi-layer (inter-layer dim == proj_size).

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include "../grad_flow_helpers.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class G1Test : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(G1Test, Construct_NoThrow_WhenProjSizePositive) {
    EXPECT_NO_THROW(
        nn::LSTM lstm(/*input_size=*/4, /*hidden_size=*/8, /*num_layers=*/1,
                       /*bias=*/true, /*batch_first=*/false, /*dropout=*/0.0,
                       /*bidirectional=*/false, /*proj_size=*/3));
}

TEST_P(G1Test, Reject_ProjSize_GE_HiddenSize) {
    EXPECT_THROW(
        nn::LSTM(4, 8, 1, true, false, 0.0, false, /*proj_size=*/8),
        std::invalid_argument);
    EXPECT_THROW(
        nn::LSTM(4, 8, 1, true, false, 0.0, false, /*proj_size=*/12),
        std::invalid_argument);
}

TEST_P(G1Test, Reject_ProjSize_Negative) {
    EXPECT_THROW(
        nn::LSTM(4, 8, 1, true, false, 0.0, false, /*proj_size=*/-1),
        std::invalid_argument);
}

TEST_P(G1Test, Forward_OutputShape_TrailingDimIsProjSize) {
    const int64_t input_size = 4;
    const int64_t hidden_size = 8;
    const int64_t proj_size = 3;
    const int64_t seq_len = 5;
    const int64_t batch_size = 2;

    nn::LSTM lstm(input_size, hidden_size, /*num_layers=*/1, /*bias=*/true,
                   /*batch_first=*/false, /*dropout=*/0.0, /*bidirectional=*/false,
                   /*proj_size=*/proj_size);
    lstm.to(device);

    // Build input on CPU, inject signal, then move to device.
    auto x_cpu = zeros({seq_len, batch_size, input_size}, DType::Float32, Device::cpu());
    auto* xp = x_cpu.data<float>();
    for (int64_t i = 0; i < x_cpu.numel(); ++i) xp[i] = (i % 7) * 0.1f;
    Variable x(x_cpu.to(device), /*requires_grad=*/true);

    auto [output, hc] = lstm.forward(x, {Variable{}, Variable{}});
    auto& [h_n, c_n] = hc;

    auto out_shape = output.shape();
    ASSERT_EQ(out_shape.size(), 3u);
    EXPECT_EQ(out_shape[0], seq_len);
    EXPECT_EQ(out_shape[1], batch_size);
    EXPECT_EQ(out_shape[2], proj_size);  // output trailing dim is projection

    auto h_shape = h_n.shape();
    ASSERT_EQ(h_shape.size(), 3u);
    EXPECT_EQ(h_shape[0], 1);                // num_layers * num_directions
    EXPECT_EQ(h_shape[1], batch_size);
    EXPECT_EQ(h_shape[2], proj_size);        // h_n trailing dim is proj_size

    auto c_shape = c_n.shape();
    ASSERT_EQ(c_shape.size(), 3u);
    EXPECT_EQ(c_shape[2], hidden_size);      // c_n trailing dim stays hidden_size
}

TEST_P(G1Test, Forward_AcceptsH0_WithProjSizeShape) {
    const int64_t input_size = 4;
    const int64_t hidden_size = 8;
    const int64_t proj_size = 3;
    const int64_t batch_size = 2;
    const int64_t seq_len = 3;

    nn::LSTM lstm(input_size, hidden_size, 1, true, false, 0.0, false, proj_size);
    lstm.to(device);

    Variable x(zeros({seq_len, batch_size, input_size}, DType::Float32, device), false);
    Variable h0(zeros({1, batch_size, proj_size}, DType::Float32, device), false);
    Variable c0(zeros({1, batch_size, hidden_size}, DType::Float32, device), false);

    auto run_forward = [&]() {
        auto [out, _] = lstm.forward(x, {h0, c0});
        (void)out;
    };
    EXPECT_NO_THROW(run_forward());
}

TEST_P(G1Test, Forward_RejectsH0_WithHiddenSizeShapeWhenProjEnabled) {
    nn::LSTM lstm(4, 8, 1, true, false, 0.0, false, /*proj_size=*/3);
    lstm.to(device);
    Variable x(zeros({3, 2, 4}, DType::Float32, device), false);
    // h0 is wrong shape: hidden_size instead of proj_size on trailing dim.
    Variable h0(zeros({1, 2, 8}, DType::Float32, device), false);
    Variable c0(zeros({1, 2, 8}, DType::Float32, device), false);
    EXPECT_THROW(lstm.forward(x, {h0, c0}), std::runtime_error);
}

TEST_P(G1Test, GradientFlowsToInput) {
    const int64_t input_size = 4;
    const int64_t hidden_size = 8;
    const int64_t proj_size = 3;
    const int64_t batch_size = 2;
    const int64_t seq_len = 3;

    nn::LSTM lstm(input_size, hidden_size, 1, true, false, 0.0, false, proj_size);
    lstm.to(device);
    lstm.train();

    // Build input on CPU, inject signal, then move to device.
    auto x_cpu = zeros({seq_len, batch_size, input_size}, DType::Float32, Device::cpu());
    auto* xp = x_cpu.data<float>();
    for (int64_t i = 0; i < x_cpu.numel(); ++i) xp[i] = ((i * 13) % 11) * 0.1f;
    Variable x(x_cpu.to(device), true);

    auto [out, _] = lstm.forward(x, {Variable{}, Variable{}});
    auto loss = tenzor::sum(out);
    loss.backward();
    EXPECT_GRAD_FLOWS(x);
    // Gradient should be non-zero somewhere.
    bool any_nonzero = false;
    auto grad_cpu = x.grad().value().cpu();
    auto* gp = grad_cpu.data<float>();
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        if (std::abs(gp[i]) > 1e-8f) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero) << "input gradient should flow through projection";
}

TEST_P(G1Test, MultiLayer_InterLayerDimIsProjSize) {
    // With proj_size, layer-2's input dim should be proj_size (not hidden_size).
    // Construct succeeds and forward runs without shape errors.
    const int64_t input_size = 4;
    const int64_t hidden_size = 8;
    const int64_t proj_size = 3;
    nn::LSTM lstm(input_size, hidden_size, /*num_layers=*/3, true, false, 0.0, false, proj_size);
    lstm.to(device);
    Variable x(zeros({5, 2, input_size}, DType::Float32, device), false);
    auto [out, _] = lstm.forward(x, {Variable{}, Variable{}});
    auto out_shape = out.shape();
    EXPECT_EQ(out_shape[2], proj_size);
}

INSTANTIATE_BACKEND_TESTS(G1Test);
