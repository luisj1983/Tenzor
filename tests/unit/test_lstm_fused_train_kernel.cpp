#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "../../src/backends/cpu/kernels/fused_lstm.hpp"
#include <vector>
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
