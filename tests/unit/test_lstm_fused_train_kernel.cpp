#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "../../src/backends/cpu/kernels/fused_lstm.hpp"
#include <vector>
#include <array>
#include <cmath>

using namespace tenzor;

namespace {

// Reference single-timestep LSTM cell in plain scalar C++, independent of
// fused_lstm.hpp, used only to check lstm_cell_fused's *existing* behavior
// is unchanged by the new optional reserve parameters.
void reference_lstm_cell(
    const std::vector<float>& gates_ih, const std::vector<float>& h,
    const std::vector<float>& c, const std::vector<float>& W_hh,
    const std::vector<float>& bias,
    std::vector<float>& h_out, std::vector<float>& c_out,
    int64_t batch, int64_t hidden) {
    int64_t gate_size = 4 * hidden;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t d = 0; d < hidden; ++d) {
            float i_pre = gates_ih[b * gate_size + d] + bias[d];
            float f_pre = gates_ih[b * gate_size + hidden + d] + bias[hidden + d];
            float g_pre = gates_ih[b * gate_size + 2 * hidden + d] + bias[2 * hidden + d];
            float o_pre = gates_ih[b * gate_size + 3 * hidden + d] + bias[3 * hidden + d];
            for (int64_t k = 0; k < hidden; ++k) {
                float hv = h[b * hidden + k];
                i_pre += hv * W_hh[(0 * hidden + d) * hidden + k];
                f_pre += hv * W_hh[(1 * hidden + d) * hidden + k];
                g_pre += hv * W_hh[(2 * hidden + d) * hidden + k];
                o_pre += hv * W_hh[(3 * hidden + d) * hidden + k];
            }
            float i = 1.0f / (1.0f + std::exp(-i_pre));
            float f = 1.0f / (1.0f + std::exp(-f_pre));
            float g = std::tanh(g_pre);
            float o = 1.0f / (1.0f + std::exp(-o_pre));
            float c_new = f * c[b * hidden + d] + i * g;
            c_out[b * hidden + d] = c_new;
            h_out[b * hidden + d] = o * std::tanh(c_new);
        }
    }
}

// Runs the ReserveCapturingCellMatchesExistingCellExactly checks for a given
// `hidden` size, so callers can exercise the AVX512 (hidden >= 16) and AVX2
// (hidden >= 8) SIMD reserve-store code paths in lstm_cell_fused, not just
// the scalar remainder loop that hidden = 5 alone would hit.
void run_reserve_capturing_cell_check(int64_t batch, int64_t hidden) {
    const int64_t gate_size = 4 * hidden;

    std::vector<float> gates_ih(batch * gate_size), W_hh(gate_size * hidden),
        bias(gate_size), h(batch * hidden), c(batch * hidden);
    for (auto& v : gates_ih) v = 0.01f * static_cast<float>(&v - &gates_ih[0]) - 0.3f;
    for (auto& v : W_hh) v = 0.001f * static_cast<float>(&v - &W_hh[0]) - 0.05f;
    for (auto& v : bias) v = 0.02f * static_cast<float>(&v - &bias[0]) - 0.1f;
    for (auto& v : h) v = 0.05f * static_cast<float>(&v - &h[0]) - 0.2f;
    for (auto& v : c) v = 0.03f * static_cast<float>(&v - &c[0]) - 0.15f;

    // Existing path (no reserve capture).
    std::vector<float> h_out_existing(batch * hidden), c_out_existing(batch * hidden);
    std::vector<float> workspace(batch * gate_size);
    tenzor::cpu::lstm::lstm_cell_fused(
        gates_ih.data(), h.data(), c.data(), W_hh.data(), bias.data(),
        h_out_existing.data(), c_out_existing.data(), workspace.data(), batch, hidden);

    // New path (with reserve capture) must produce bit-identical h_out/c_out
    // since it must call the exact same SIMD/scalar code paths, only with
    // additional stores appended.
    std::vector<float> h_out_new(batch * hidden), c_out_new(batch * hidden);
    std::vector<float> i_res(batch * hidden), f_res(batch * hidden),
        g_res(batch * hidden), o_res(batch * hidden), tanh_c_res(batch * hidden);
    tenzor::cpu::lstm::lstm_cell_fused(
        gates_ih.data(), h.data(), c.data(), W_hh.data(), bias.data(),
        h_out_new.data(), c_out_new.data(), workspace.data(), batch, hidden,
        i_res.data(), f_res.data(), g_res.data(), o_res.data(), tanh_c_res.data());

    for (int64_t i = 0; i < batch * hidden; ++i) {
        EXPECT_EQ(h_out_existing[i], h_out_new[i]) << "h_out diverged at " << i;
        EXPECT_EQ(c_out_existing[i], c_out_new[i]) << "c_out diverged at " << i;
    }

    // The reserve buffers must match an independent scalar reference
    // implementation of the LSTM cell equations, and h_out == o*tanh(c_out).
    std::vector<float> h_ref(batch * hidden), c_ref(batch * hidden);
    reference_lstm_cell(gates_ih, h, c, W_hh, bias, h_ref, c_ref, batch, hidden);
    for (int64_t i = 0; i < batch * hidden; ++i) {
        EXPECT_NEAR(c_ref[i], c_out_new[i], 1e-5f) << "c mismatch at " << i;
        EXPECT_NEAR(std::tanh(c_ref[i]), tanh_c_res[i], 1e-5f) << "tanh(c) reserve mismatch at " << i;
        EXPECT_NEAR(o_res[i] * std::tanh(c_ref[i]), h_out_new[i], 1e-5f) << "h_out vs o*tanh(c) mismatch at " << i;
    }
}

}  // namespace

TEST(LSTMFusedTrainKernel, ReserveCapturingCellMatchesExistingCellExactly) {
    // hidden = 5: below the AVX2 (>=8) and AVX512 (>=16) loop guards, so
    // every lane goes through the scalar remainder loop only.
    run_reserve_capturing_cell_check(/*batch=*/3, /*hidden=*/5);
}

TEST(LSTMFusedTrainKernel, ReserveCapturingCellMatchesExistingCellExactlyAvx2Width) {
    // hidden = 10: exercises the AVX2 loop guard (`d + 8 <= hidden`, one
    // 8-wide iteration) with a 2-element scalar remainder, while staying
    // below the AVX512 guard (`d + 16 <= hidden`) so the AVX512 block is not
    // exercised here.
    run_reserve_capturing_cell_check(/*batch=*/3, /*hidden=*/10);
}

TEST(LSTMFusedTrainKernel, ReserveCapturingCellMatchesExistingCellExactlyAvx512Width) {
    // hidden = 20: exercises the AVX512 loop guard (`d + 16 <= hidden`, one
    // 16-wide iteration) followed by a 4-element remainder that in turn
    // exercises the scalar loop (falls below the AVX2 8-wide guard).
    run_reserve_capturing_cell_check(/*batch=*/3, /*hidden=*/20);
}

TEST(LSTMFusedTrainKernel, ForwardTrainingMatchesInferenceForward) {
    const int64_t seq_len = 4, batch = 2, input_size = 3, hidden = 5;
    const int64_t gate_size = 4 * hidden;

    std::vector<float> input(seq_len * batch * input_size), W_ih(gate_size * input_size),
        W_hh(gate_size * hidden), bias(gate_size), h0(batch * hidden), c0(batch * hidden);
    for (auto& v : input) v = 0.01f * static_cast<float>(&v - &input[0]) - 0.2f;
    for (auto& v : W_ih) v = 0.002f * static_cast<float>(&v - &W_ih[0]) - 0.05f;
    for (auto& v : W_hh) v = 0.001f * static_cast<float>(&v - &W_hh[0]) - 0.03f;
    for (auto& v : bias) v = 0.01f * static_cast<float>(&v - &bias[0]);
    for (auto& v : h0) v = 0.02f * static_cast<float>(&v - &h0[0]) - 0.1f;
    for (auto& v : c0) v = 0.02f * static_cast<float>(&v - &c0[0]) - 0.1f;

    std::vector<float> out_ref(seq_len * batch * hidden), hn_ref(batch * hidden), cn_ref(batch * hidden);
    tenzor::cpu::lstm::lstm_forward(
        input.data(), W_ih.data(), W_hh.data(), bias.data(), h0.data(), c0.data(),
        out_ref.data(), hn_ref.data(), cn_ref.data(), seq_len, batch, input_size, hidden);

    std::vector<float> out_new(seq_len * batch * hidden), hn_new(batch * hidden), cn_new(batch * hidden);
    std::vector<float> gates_reserve(4 * seq_len * batch * hidden);
    std::vector<float> cell_reserve(seq_len * batch * hidden), tanh_cell_reserve(seq_len * batch * hidden);
    tenzor::cpu::lstm::lstm_forward_training(
        input.data(), W_ih.data(), W_hh.data(), bias.data(), h0.data(), c0.data(),
        out_new.data(), hn_new.data(), cn_new.data(),
        gates_reserve.data(), cell_reserve.data(), tanh_cell_reserve.data(),
        seq_len, batch, input_size, hidden);

    for (size_t i = 0; i < out_ref.size(); ++i) EXPECT_EQ(out_ref[i], out_new[i]) << "output diverged at " << i;
    for (size_t i = 0; i < hn_ref.size(); ++i) EXPECT_EQ(hn_ref[i], hn_new[i]) << "h_n diverged at " << i;
    for (size_t i = 0; i < cn_ref.size(); ++i) EXPECT_EQ(cn_ref[i], cn_new[i]) << "c_n diverged at " << i;

    // cell_reserve at the last timestep must equal c_n exactly.
    for (int64_t i = 0; i < batch * hidden; ++i) {
        EXPECT_EQ(cn_new[i], cell_reserve[(seq_len - 1) * batch * hidden + i]);
    }
}

namespace {

struct LstmProblem {
    int64_t seq_len, batch, input_size, hidden;
    std::vector<float> input, W_ih, W_hh, bias, h0, c0;
};

LstmProblem make_problem(int64_t seq_len, int64_t batch, int64_t input_size, int64_t hidden) {
    LstmProblem p{seq_len, batch, input_size, hidden, {}, {}, {}, {}, {}, {}};
    int64_t gate_size = 4 * hidden;
    p.input.resize(seq_len * batch * input_size);
    p.W_ih.resize(gate_size * input_size);
    p.W_hh.resize(gate_size * hidden);
    p.bias.resize(gate_size);
    p.h0.resize(batch * hidden);
    p.c0.resize(batch * hidden);
    auto fill = [](std::vector<float>& v, float scale, float offset) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = scale * static_cast<float>((i * 37 + 11) % 101) / 101.0f - offset;
        }
    };
    fill(p.input, 0.4f, 0.2f);
    fill(p.W_ih, 0.3f, 0.15f);
    fill(p.W_hh, 0.25f, 0.12f);
    fill(p.bias, 0.1f, 0.05f);
    fill(p.h0, 0.2f, 0.1f);
    fill(p.c0, 0.2f, 0.1f);
    return p;
}

// Runs forward_training and returns the scalar loss L = sum(output) + sum(c_n),
// matching the two independent autograd nodes the real LSTM wiring creates
// (one for `output`, one for `c_n`) collapsed into one scalar for this
// finite-difference check.
float run_loss(const LstmProblem& p) {
    int64_t gate_size = 4 * p.hidden;
    std::vector<float> output(p.seq_len * p.batch * p.hidden), h_n(p.batch * p.hidden), c_n(p.batch * p.hidden);
    std::vector<float> gates_reserve(4 * p.seq_len * p.batch * p.hidden);
    std::vector<float> cell_reserve(p.seq_len * p.batch * p.hidden), tanh_cell_reserve(p.seq_len * p.batch * p.hidden);
    tenzor::cpu::lstm::lstm_forward_training(
        p.input.data(), p.W_ih.data(), p.W_hh.data(), p.bias.data(), p.h0.data(), p.c0.data(),
        output.data(), h_n.data(), c_n.data(),
        gates_reserve.data(), cell_reserve.data(), tanh_cell_reserve.data(),
        p.seq_len, p.batch, p.input_size, p.hidden);
    (void)gate_size;
    float loss = 0.0f;
    for (float v : output) loss += v;
    for (float v : c_n) loss += v;
    return loss;
}

}  // namespace

TEST(LSTMFusedTrainKernel, BackwardMatchesFiniteDifference) {
    auto p = make_problem(/*seq_len=*/4, /*batch=*/2, /*input_size=*/3, /*hidden=*/4);
    int64_t gate_size = 4 * p.hidden;

    // Forward + reserve capture.
    std::vector<float> output(p.seq_len * p.batch * p.hidden), h_n(p.batch * p.hidden), c_n(p.batch * p.hidden);
    std::vector<float> gates_reserve(4 * p.seq_len * p.batch * p.hidden);
    std::vector<float> cell_reserve(p.seq_len * p.batch * p.hidden), tanh_cell_reserve(p.seq_len * p.batch * p.hidden);
    tenzor::cpu::lstm::lstm_forward_training(
        p.input.data(), p.W_ih.data(), p.W_hh.data(), p.bias.data(), p.h0.data(), p.c0.data(),
        output.data(), h_n.data(), c_n.data(),
        gates_reserve.data(), cell_reserve.data(), tanh_cell_reserve.data(),
        p.seq_len, p.batch, p.input_size, p.hidden);

    // grad_output = all-ones (matches d(sum(output))/d(output)), grad_cy = all-ones.
    std::vector<float> grad_output(p.seq_len * p.batch * p.hidden, 1.0f);
    std::vector<float> grad_cy(p.batch * p.hidden, 1.0f);

    std::vector<float> grad_input(p.seq_len * p.batch * p.input_size);
    std::vector<float> grad_h0(p.batch * p.hidden), grad_c0(p.batch * p.hidden);
    std::vector<float> grad_W_ih(gate_size * p.input_size), grad_W_hh(gate_size * p.hidden);
    std::vector<float> grad_bias(gate_size);

    tenzor::cpu::lstm::lstm_backward_training(
        grad_output.data(), grad_cy.data(),
        p.input.data(), p.h0.data(), p.c0.data(), p.W_ih.data(), p.W_hh.data(),
        gates_reserve.data(), cell_reserve.data(), tanh_cell_reserve.data(),
        grad_input.data(), grad_h0.data(), grad_c0.data(),
        grad_W_ih.data(), grad_W_hh.data(), grad_bias.data(),
        p.seq_len, p.batch, p.input_size, p.hidden);

    const float eps = 1e-3f;
    const float tol = 3e-2f;  // finite-difference tolerance at eps=1e-3, float32

    // Spot-check grad_input at a few (t,b,f) coordinates.
    for (auto [t, b, f] : std::vector<std::array<int64_t, 3>>{{0, 0, 0}, {2, 1, 2}, {3, 0, 1}}) {
        auto p_plus = p, p_minus = p;
        size_t idx = static_cast<size_t>((t * p.batch + b) * p.input_size + f);
        p_plus.input[idx] += eps;
        p_minus.input[idx] -= eps;
        float numeric = (run_loss(p_plus) - run_loss(p_minus)) / (2 * eps);
        float analytic = grad_input[idx];
        EXPECT_NEAR(numeric, analytic, tol) << "grad_input mismatch at t=" << t << " b=" << b << " f=" << f;
    }

    // Spot-check grad_h0 / grad_c0.
    for (int64_t idx : {int64_t{0}, int64_t{3}, int64_t{7}}) {
        auto p_plus = p, p_minus = p;
        p_plus.h0[idx] += eps;
        p_minus.h0[idx] -= eps;
        float numeric = (run_loss(p_plus) - run_loss(p_minus)) / (2 * eps);
        EXPECT_NEAR(numeric, grad_h0[idx], tol) << "grad_h0 mismatch at " << idx;

        p_plus = p; p_minus = p;
        p_plus.c0[idx] += eps;
        p_minus.c0[idx] -= eps;
        numeric = (run_loss(p_plus) - run_loss(p_minus)) / (2 * eps);
        EXPECT_NEAR(numeric, grad_c0[idx], tol) << "grad_c0 mismatch at " << idx;
    }

    // Spot-check grad_W_ih / grad_W_hh / grad_bias.
    for (int64_t idx : {int64_t{0}, int64_t{5}, int64_t{9}}) {
        auto p_plus = p, p_minus = p;
        p_plus.W_ih[idx] += eps;
        p_minus.W_ih[idx] -= eps;
        float numeric = (run_loss(p_plus) - run_loss(p_minus)) / (2 * eps);
        EXPECT_NEAR(numeric, grad_W_ih[idx], tol) << "grad_W_ih mismatch at " << idx;

        p_plus = p; p_minus = p;
        p_plus.W_hh[idx] += eps;
        p_minus.W_hh[idx] -= eps;
        numeric = (run_loss(p_plus) - run_loss(p_minus)) / (2 * eps);
        EXPECT_NEAR(numeric, grad_W_hh[idx], tol) << "grad_W_hh mismatch at " << idx;

        p_plus = p; p_minus = p;
        p_plus.bias[idx] += eps;
        p_minus.bias[idx] -= eps;
        numeric = (run_loss(p_plus) - run_loss(p_minus)) / (2 * eps);
        EXPECT_NEAR(numeric, grad_bias[idx], tol) << "grad_bias mismatch at " << idx;
    }
}
