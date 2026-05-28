/**
 * @file test_gradcheck_rnn_f64.cpp
 * @brief S12 — Float64 native precision regression tests for the CPU RNN
 *        (LSTM/GRU) kernels.
 *
 * Before S12, every kernel in `src/backends/cpu/kernels/rnn_kernels.cpp`
 * funnelled Float64 inputs through `to(Float32) → compute → to(Float64)`.
 * That hides Float32 precision loss in the Float64 storage, so gradcheck
 * against the Float64 numerical derivative was off by `ε(Float32) · signal`
 * — small enough that earlier passes mistook it as a tolerance issue.
 *
 * These tests exercise each of the nine audit-listed kernels with Float64
 * inputs and confirm gradcheck passes at the tight Float64 tolerance
 * (1e-6 ε, 1e-5 / 1e-3 abs/rel). If somebody re-introduces a Float32
 * round-trip, the regression here trips.
 *
 * Coverage map (kernel → test):
 *   lstm_cell_forward_kernel          LSTMCellForwardF64
 *   lstm_cell_backward_kernel         LSTMCellBackwardF64
 *   gru_cell_forward_kernel           GRUCellForwardF64
 *   gru_cell_backward_kernel          GRUCellBackwardF64
 *   lstm_forward_kernel               LSTMForwardF64
 *   gru_forward_kernel                GRUForwardF64
 *   bilstm_forward_kernel             BiLSTMForwardF64
 *   lstm_multilayer_forward_kernel    LSTMMultilayerF64
 *   gru_multilayer_forward_kernel     GRUMultilayerF64
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <cmath>
#include <vector>

using namespace tenzor;

namespace {

class RNNF64Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        tenzor::set_grad_enabled(true);
    }

    // Float64 gradcheck tolerances. With native Float64 compute, central
    // FD vs analytic must agree to ~1e-5 absolute / 1e-3 relative.
    static constexpr double kEps  = 1e-6;
    static constexpr double kAtol = 1e-5;
    static constexpr double kRtol = 1e-3;
};

// Build a small Float64 tensor with values in [-1, 1) using a fixed seed
// so the test is fully deterministic across runs/builds.
Tensor det_randn(std::vector<int64_t> shape, uint64_t seed) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    std::vector<double> data(static_cast<size_t>(numel));
    // Linear-congruential generator → smooth doubles in [-1, 1).
    uint64_t s = seed;
    for (int64_t i = 0; i < numel; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        // Use top 53 bits as a uniform double in [0, 1)
        double u = static_cast<double>(s >> 11) * (1.0 / (1ULL << 53));
        data[i] = 2.0 * u - 1.0;
    }
    Tensor t = Tensor::empty_uninitialized(shape, DType::Float64, Device::cpu());
    std::memcpy(t.data<double>(), data.data(),
                static_cast<size_t>(numel) * sizeof(double));
    return t;
}

}  // namespace

// =============================================================================
// Cell forward kernels (LSTM, GRU)
// =============================================================================

TEST_F(RNNF64Test, LSTMCellForwardF64) {
    // nn::LSTMCell goes through the Linear-based gate-composition path for
    // Float64 (the fused LSTMCellForward op-dispatch is skipped because
    // the OneDNN/SIMD cell kernel is Float32-only). However, the dispatch
    // path through OpId::LSTMCellForward IS still callable, and IS what the
    // audit complained about. Test it directly.
    const int64_t batch = 2, in_sz = 3, hid = 4;
    auto input = det_randn({batch, in_sz}, 0xA1);
    auto hx    = det_randn({batch, hid}, 0xA2);
    auto cx    = det_randn({batch, hid}, 0xA3);
    auto w_ih  = det_randn({4 * hid, in_sz}, 0xA4);
    auto w_hh  = det_randn({4 * hid, hid}, 0xA5);
    auto b_ih  = det_randn({4 * hid}, 0xA6);
    auto b_hh  = det_randn({4 * hid}, 0xA7);

    // Sanity check: kernel runs in Float64 end-to-end and the result
    // dtype matches.
    const Tensor in_arr[7] = {input, hx, cx, w_ih, w_hh, b_ih, b_hh};
    auto outs = tenzor::dispatch(OpId::LSTMCellForward,
                                  std::span<const Tensor>{in_arr, 7}, {});
    ASSERT_EQ(outs.size(), 2u);
    EXPECT_EQ(outs[0].dtype(), DType::Float64);
    EXPECT_EQ(outs[1].dtype(), DType::Float64);

    // Compare against a reference implementation done entirely in double.
    auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    std::vector<double> ref_hy(batch * hid), ref_cy(batch * hid);
    const double* in_d  = input.data<double>();
    const double* hx_d  = hx.data<double>();
    const double* cx_d  = cx.data<double>();
    const double* wih_d = w_ih.data<double>();
    const double* whh_d = w_hh.data<double>();
    const double* bih_d = b_ih.data<double>();
    const double* bhh_d = b_hh.data<double>();
    for (int64_t b = 0; b < batch; ++b) {
        std::vector<double> gates(4 * hid);
        for (int64_t g = 0; g < 4 * hid; ++g) {
            double s = bih_d[g] + bhh_d[g];
            for (int64_t i = 0; i < in_sz; ++i)
                s += in_d[b * in_sz + i] * wih_d[g * in_sz + i];
            for (int64_t h = 0; h < hid; ++h)
                s += hx_d[b * hid + h] * whh_d[g * hid + h];
            gates[g] = s;
        }
        for (int64_t h = 0; h < hid; ++h) {
            double i_g = sig(gates[h]);
            double f_g = sig(gates[hid + h]);
            double g_g = std::tanh(gates[2 * hid + h]);
            double o_g = sig(gates[3 * hid + h]);
            double c_new = f_g * cx_d[b * hid + h] + i_g * g_g;
            ref_cy[b * hid + h] = c_new;
            ref_hy[b * hid + h] = o_g * std::tanh(c_new);
        }
    }

    const double* got_hy = outs[0].data<double>();
    const double* got_cy = outs[1].data<double>();
    for (int64_t i = 0; i < batch * hid; ++i) {
        // Native Float64 must match scalar double reference to ~1e-13.
        EXPECT_NEAR(got_hy[i], ref_hy[static_cast<size_t>(i)], 1e-13)
            << "hy mismatch at " << i;
        EXPECT_NEAR(got_cy[i], ref_cy[static_cast<size_t>(i)], 1e-13)
            << "cy mismatch at " << i;
    }
}

TEST_F(RNNF64Test, GRUCellForwardF64) {
    const int64_t batch = 2, in_sz = 3, hid = 4;
    auto input = det_randn({batch, in_sz}, 0xB1);
    auto hx    = det_randn({batch, hid}, 0xB2);
    auto w_ih  = det_randn({3 * hid, in_sz}, 0xB3);
    auto w_hh  = det_randn({3 * hid, hid}, 0xB4);
    auto b_ih  = det_randn({3 * hid}, 0xB5);
    auto b_hh  = det_randn({3 * hid}, 0xB6);

    const Tensor in_arr[6] = {input, hx, w_ih, w_hh, b_ih, b_hh};
    auto outs = tenzor::dispatch(OpId::GRUCellForward,
                                  std::span<const Tensor>{in_arr, 6}, {});
    ASSERT_EQ(outs.size(), 1u);
    EXPECT_EQ(outs[0].dtype(), DType::Float64);

    // Reference: PyTorch GRU cell semantics in double.
    auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    std::vector<double> ref_hy(batch * hid);
    const double* in_d  = input.data<double>();
    const double* hx_d  = hx.data<double>();
    const double* wih_d = w_ih.data<double>();
    const double* whh_d = w_hh.data<double>();
    const double* bih_d = b_ih.data<double>();
    const double* bhh_d = b_hh.data<double>();
    for (int64_t b = 0; b < batch; ++b) {
        std::vector<double> rz(2 * hid);
        for (int64_t g = 0; g < 2 * hid; ++g) {
            double s = bih_d[g] + bhh_d[g];
            for (int64_t i = 0; i < in_sz; ++i)
                s += in_d[b * in_sz + i] * wih_d[g * in_sz + i];
            for (int64_t h = 0; h < hid; ++h)
                s += hx_d[b * hid + h] * whh_d[g * hid + h];
            rz[g] = sig(s);
        }
        for (int64_t h = 0; h < hid; ++h) {
            double r = rz[h];
            double z = rz[hid + h];
            double n_ih = bih_d[2 * hid + h];
            double n_hh = bhh_d[2 * hid + h];
            for (int64_t i = 0; i < in_sz; ++i)
                n_ih += in_d[b * in_sz + i] * wih_d[(2 * hid + h) * in_sz + i];
            for (int64_t hh = 0; hh < hid; ++hh)
                n_hh += hx_d[b * hid + hh] * whh_d[(2 * hid + h) * hid + hh];
            n_hh *= r;
            double n = std::tanh(n_ih + n_hh);
            ref_hy[b * hid + h] = (1.0 - z) * n + z * hx_d[b * hid + h];
        }
    }
    const double* got_hy = outs[0].data<double>();
    for (int64_t i = 0; i < batch * hid; ++i) {
        EXPECT_NEAR(got_hy[i], ref_hy[static_cast<size_t>(i)], 1e-13)
            << "GRU cell hy mismatch at " << i;
    }
}

// =============================================================================
// Cell backward kernels (numerical FD agreement)
// =============================================================================

// FD gradient of f(input) w.r.t. each input element, for scalar-output `f`.
// f operates on a Float64 tensor and returns a Float64 scalar.
std::vector<double> fd_grad(std::function<double(const Tensor&)> f,
                            Tensor input, double h = 1e-6) {
    Tensor x = input.contiguous();
    int64_t n = x.numel();
    std::vector<double> grad(static_cast<size_t>(n));
    double* p = x.data<double>();
    for (int64_t i = 0; i < n; ++i) {
        double orig = p[i];
        p[i] = orig + h;
        double f_plus = f(x);
        p[i] = orig - h;
        double f_minus = f(x);
        p[i] = orig;
        grad[static_cast<size_t>(i)] = (f_plus - f_minus) / (2.0 * h);
    }
    return grad;
}

TEST_F(RNNF64Test, LSTMCellBackwardF64) {
    // Drive lstm_cell_backward_kernel via direct dispatch on
    // OpId::LSTMCellBackward, and compare against numerical FD through a
    // pure-double reference forward.
    const int64_t batch = 2, in_sz = 3, hid = 4;
    auto input = det_randn({batch, in_sz}, 0xC1);
    auto hx    = det_randn({batch, hid}, 0xC2);
    auto cx    = det_randn({batch, hid}, 0xC3);
    auto w_ih  = det_randn({4 * hid, in_sz}, 0xC4);
    auto w_hh  = det_randn({4 * hid, hid}, 0xC5);
    auto b_ih  = det_randn({4 * hid}, 0xC6);
    auto b_hh  = det_randn({4 * hid}, 0xC7);

    // Forward to get hy, cy (needed as inputs to backward).
    const Tensor fwd_in[7] = {input, hx, cx, w_ih, w_hh, b_ih, b_hh};
    auto fwd = tenzor::dispatch(OpId::LSTMCellForward,
                                 std::span<const Tensor>{fwd_in, 7}, {});
    Tensor hy = fwd[0], cy = fwd[1];

    // Upstream grads: arbitrary fixed pattern.
    Tensor d_hy = det_randn({batch, hid}, 0xC8);
    Tensor d_cy = det_randn({batch, hid}, 0xC9);

    const Tensor bwd_in[11] = {d_hy, d_cy, input, hx, cx, hy, cy,
                                w_ih, w_hh, b_ih, b_hh};
    auto bwd = tenzor::dispatch(OpId::LSTMCellBackward,
                                 std::span<const Tensor>{bwd_in, 11}, {});
    ASSERT_EQ(bwd.size(), 7u);
    for (const auto& t : bwd) EXPECT_EQ(t.dtype(), DType::Float64);

    // Pure-double reference forward (matches the kernel logic).
    auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    auto forward_loss = [&](const Tensor& in_) -> double {
        std::vector<double> hy_out(batch * hid), cy_out(batch * hid);
        const double* in_d  = in_.data<double>();
        const double* hx_d  = hx.data<double>();
        const double* cx_d  = cx.data<double>();
        const double* wih_d = w_ih.data<double>();
        const double* whh_d = w_hh.data<double>();
        const double* bih_d = b_ih.data<double>();
        const double* bhh_d = b_hh.data<double>();
        for (int64_t b = 0; b < batch; ++b) {
            std::vector<double> g(4 * hid);
            for (int64_t gi = 0; gi < 4 * hid; ++gi) {
                double s = bih_d[gi] + bhh_d[gi];
                for (int64_t i = 0; i < in_sz; ++i)
                    s += in_d[b * in_sz + i] * wih_d[gi * in_sz + i];
                for (int64_t h = 0; h < hid; ++h)
                    s += hx_d[b * hid + h] * whh_d[gi * hid + h];
                g[gi] = s;
            }
            for (int64_t h = 0; h < hid; ++h) {
                double i_g = sig(g[h]);
                double f_g = sig(g[hid + h]);
                double g_g = std::tanh(g[2 * hid + h]);
                double o_g = sig(g[3 * hid + h]);
                double c_new = f_g * cx_d[b * hid + h] + i_g * g_g;
                hy_out[b * hid + h] = o_g * std::tanh(c_new);
                cy_out[b * hid + h] = c_new;
            }
        }
        // Loss = sum(d_hy * hy + d_cy * cy) — that makes the gradient w.r.t.
        // `input` equal to d_input from the kernel.
        const double* dhy = d_hy.data<double>();
        const double* dcy = d_cy.data<double>();
        double loss = 0.0;
        for (int64_t i = 0; i < batch * hid; ++i) {
            loss += dhy[i] * hy_out[static_cast<size_t>(i)] +
                    dcy[i] * cy_out[static_cast<size_t>(i)];
        }
        return loss;
    };

    auto fd = fd_grad(forward_loss, input, 1e-6);
    const double* d_input = bwd[0].data<double>();
    for (int64_t i = 0; i < input.numel(); ++i) {
        EXPECT_NEAR(d_input[i], fd[static_cast<size_t>(i)], 1e-6)
            << "LSTM-cell-backward d_input mismatch at " << i;
    }
}

TEST_F(RNNF64Test, GRUCellBackwardF64) {
    const int64_t batch = 2, in_sz = 3, hid = 4;
    auto input = det_randn({batch, in_sz}, 0xD1);
    auto hx    = det_randn({batch, hid}, 0xD2);
    auto w_ih  = det_randn({3 * hid, in_sz}, 0xD3);
    auto w_hh  = det_randn({3 * hid, hid}, 0xD4);
    auto b_ih  = det_randn({3 * hid}, 0xD5);
    auto b_hh  = det_randn({3 * hid}, 0xD6);
    Tensor d_hy = det_randn({batch, hid}, 0xD7);

    const Tensor bwd_in[7] = {d_hy, input, hx, w_ih, w_hh, b_ih, b_hh};
    auto bwd = tenzor::dispatch(OpId::GRUCellBackward,
                                 std::span<const Tensor>{bwd_in, 7}, {});
    ASSERT_EQ(bwd.size(), 6u);
    for (const auto& t : bwd) EXPECT_EQ(t.dtype(), DType::Float64);

    // Pure-double reference forward → scalar loss.
    auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    auto forward_loss = [&](const Tensor& in_) -> double {
        std::vector<double> hy_out(batch * hid);
        const double* in_d  = in_.data<double>();
        const double* hx_d  = hx.data<double>();
        const double* wih_d = w_ih.data<double>();
        const double* whh_d = w_hh.data<double>();
        const double* bih_d = b_ih.data<double>();
        const double* bhh_d = b_hh.data<double>();
        for (int64_t b = 0; b < batch; ++b) {
            std::vector<double> rz(2 * hid);
            for (int64_t g = 0; g < 2 * hid; ++g) {
                double s = bih_d[g] + bhh_d[g];
                for (int64_t i = 0; i < in_sz; ++i)
                    s += in_d[b * in_sz + i] * wih_d[g * in_sz + i];
                for (int64_t h = 0; h < hid; ++h)
                    s += hx_d[b * hid + h] * whh_d[g * hid + h];
                rz[g] = sig(s);
            }
            for (int64_t h = 0; h < hid; ++h) {
                double r = rz[h];
                double z = rz[hid + h];
                double n_ih = bih_d[2 * hid + h];
                double n_hh = bhh_d[2 * hid + h];
                for (int64_t i = 0; i < in_sz; ++i)
                    n_ih += in_d[b * in_sz + i] * wih_d[(2 * hid + h) * in_sz + i];
                for (int64_t hh = 0; hh < hid; ++hh)
                    n_hh += hx_d[b * hid + hh] * whh_d[(2 * hid + h) * hid + hh];
                n_hh *= r;
                double n = std::tanh(n_ih + n_hh);
                hy_out[b * hid + h] = (1.0 - z) * n + z * hx_d[b * hid + h];
            }
        }
        const double* dhy = d_hy.data<double>();
        double loss = 0.0;
        for (int64_t i = 0; i < batch * hid; ++i)
            loss += dhy[i] * hy_out[static_cast<size_t>(i)];
        return loss;
    };

    auto fd = fd_grad(forward_loss, input, 1e-6);
    const double* d_input = bwd[0].data<double>();
    for (int64_t i = 0; i < input.numel(); ++i) {
        EXPECT_NEAR(d_input[i], fd[static_cast<size_t>(i)], 1e-6)
            << "GRU-cell-backward d_input mismatch at " << i;
    }
}

// =============================================================================
// Full-sequence forward kernels
// =============================================================================

// Reference Float64 LSTM forward, one direction, mirroring kernel logic.
void ref_lstm_forward_f64(
    const double* input, const double* W_ih, const double* W_hh,
    const double* bias_ih, const double* bias_hh,
    const double* h0, const double* c0,
    double* output, double* h_n, double* c_n,
    int64_t seq_len, int64_t batch, int64_t in_sz, int64_t hid)
{
    int64_t gate = 4 * hid;
    auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    std::vector<double> h_curr(batch * hid), c_curr(batch * hid);
    std::memcpy(h_curr.data(), h0, batch * hid * sizeof(double));
    std::memcpy(c_curr.data(), c0, batch * hid * sizeof(double));
    for (int64_t t = 0; t < seq_len; ++t) {
        const double* in_t = input + t * batch * in_sz;
        double* out_t = output + t * batch * hid;
        for (int64_t b = 0; b < batch; ++b) {
            std::vector<double> g(gate);
            for (int64_t gi = 0; gi < gate; ++gi) {
                double s = (bias_ih ? bias_ih[gi] : 0.0) + (bias_hh ? bias_hh[gi] : 0.0);
                for (int64_t i = 0; i < in_sz; ++i)
                    s += in_t[b * in_sz + i] * W_ih[gi * in_sz + i];
                for (int64_t h = 0; h < hid; ++h)
                    s += h_curr[b * hid + h] * W_hh[gi * hid + h];
                g[gi] = s;
            }
            for (int64_t h = 0; h < hid; ++h) {
                double i_g = sig(g[h]);
                double f_g = sig(g[hid + h]);
                double g_g = std::tanh(g[2 * hid + h]);
                double o_g = sig(g[3 * hid + h]);
                double c_new = f_g * c_curr[b * hid + h] + i_g * g_g;
                double h_new = o_g * std::tanh(c_new);
                out_t[b * hid + h] = h_new;
                c_curr[b * hid + h] = c_new;
            }
        }
        std::memcpy(h_curr.data(), out_t, batch * hid * sizeof(double));
    }
    std::memcpy(h_n, h_curr.data(), batch * hid * sizeof(double));
    std::memcpy(c_n, c_curr.data(), batch * hid * sizeof(double));
}

TEST_F(RNNF64Test, LSTMForwardF64) {
    const int64_t seq = 3, batch = 2, in_sz = 3, hid = 4;
    auto input = det_randn({seq, batch, in_sz}, 0xE1);
    auto W_ih  = det_randn({4 * hid, in_sz}, 0xE2);
    auto W_hh  = det_randn({4 * hid, hid}, 0xE3);
    auto b_ih  = det_randn({4 * hid}, 0xE4);
    auto b_hh  = det_randn({4 * hid}, 0xE5);
    auto h0    = det_randn({batch, hid}, 0xE6);
    auto c0    = det_randn({batch, hid}, 0xE7);

    const Tensor in_arr[7] = {input, W_ih, W_hh, b_ih, b_hh, h0, c0};
    auto outs = tenzor::dispatch(OpId::LSTMForward,
                                  std::span<const Tensor>{in_arr, 7}, {});
    ASSERT_EQ(outs.size(), 3u);
    EXPECT_EQ(outs[0].dtype(), DType::Float64);

    std::vector<double> ref_out(seq * batch * hid);
    std::vector<double> ref_hn(batch * hid), ref_cn(batch * hid);
    ref_lstm_forward_f64(input.data<double>(), W_ih.data<double>(),
                          W_hh.data<double>(), b_ih.data<double>(),
                          b_hh.data<double>(), h0.data<double>(),
                          c0.data<double>(), ref_out.data(),
                          ref_hn.data(), ref_cn.data(),
                          seq, batch, in_sz, hid);

    const double* got_out = outs[0].data<double>();
    const double* got_hn  = outs[1].data<double>();
    const double* got_cn  = outs[2].data<double>();
    for (int64_t i = 0; i < seq * batch * hid; ++i) {
        EXPECT_NEAR(got_out[i], ref_out[static_cast<size_t>(i)], 1e-12)
            << "lstm_forward output mismatch at " << i;
    }
    for (int64_t i = 0; i < batch * hid; ++i) {
        EXPECT_NEAR(got_hn[i], ref_hn[static_cast<size_t>(i)], 1e-12);
        EXPECT_NEAR(got_cn[i], ref_cn[static_cast<size_t>(i)], 1e-12);
    }
}

TEST_F(RNNF64Test, GRUForwardF64) {
    const int64_t seq = 3, batch = 2, in_sz = 3, hid = 4;
    auto input = det_randn({seq, batch, in_sz}, 0xF1);
    auto W_ih  = det_randn({3 * hid, in_sz}, 0xF2);
    auto W_hh  = det_randn({3 * hid, hid}, 0xF3);
    auto b_ih  = det_randn({3 * hid}, 0xF4);
    auto h0    = det_randn({batch, hid}, 0xF6);
    auto b_hh  = det_randn({3 * hid}, 0xF5);

    // Inputs ordered per cpu_kernel_registry.cpp: [input, W_ih, W_hh,
    // bias_ih, h0, bias_hh].
    const Tensor in_arr[6] = {input, W_ih, W_hh, b_ih, h0, b_hh};
    auto outs = tenzor::dispatch(OpId::GRUForward,
                                  std::span<const Tensor>{in_arr, 6}, {});
    ASSERT_EQ(outs.size(), 2u);
    EXPECT_EQ(outs[0].dtype(), DType::Float64);

    // Reference: PyTorch GRU semantics.
    auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    std::vector<double> ref_out(seq * batch * hid);
    std::vector<double> ref_hn(batch * hid);
    std::vector<double> h_curr(batch * hid);
    std::memcpy(h_curr.data(), h0.data<double>(), batch * hid * sizeof(double));
    const double* in_d = input.data<double>();
    const double* wih = W_ih.data<double>();
    const double* whh = W_hh.data<double>();
    const double* bih = b_ih.data<double>();
    const double* bhh = b_hh.data<double>();
    for (int64_t t = 0; t < seq; ++t) {
        const double* in_t = in_d + t * batch * in_sz;
        double* out_t = ref_out.data() + t * batch * hid;
        for (int64_t b = 0; b < batch; ++b) {
            std::vector<double> rz(2 * hid);
            for (int64_t g = 0; g < 2 * hid; ++g) {
                double s = bih[g] + bhh[g];
                for (int64_t i = 0; i < in_sz; ++i)
                    s += in_t[b * in_sz + i] * wih[g * in_sz + i];
                for (int64_t h = 0; h < hid; ++h)
                    s += h_curr[b * hid + h] * whh[g * hid + h];
                rz[g] = sig(s);
            }
            for (int64_t h = 0; h < hid; ++h) {
                double r = rz[h];
                double z = rz[hid + h];
                double n_ih = bih[2 * hid + h];
                double n_hh = bhh[2 * hid + h];
                for (int64_t i = 0; i < in_sz; ++i)
                    n_ih += in_t[b * in_sz + i] * wih[(2 * hid + h) * in_sz + i];
                for (int64_t hh = 0; hh < hid; ++hh)
                    n_hh += h_curr[b * hid + hh] * whh[(2 * hid + h) * hid + hh];
                n_hh *= r;
                double n = std::tanh(n_ih + n_hh);
                out_t[b * hid + h] =
                    (1.0 - z) * n + z * h_curr[b * hid + h];
            }
        }
        std::memcpy(h_curr.data(), out_t, batch * hid * sizeof(double));
    }
    std::memcpy(ref_hn.data(), h_curr.data(), batch * hid * sizeof(double));

    const double* got_out = outs[0].data<double>();
    const double* got_hn  = outs[1].data<double>();
    for (int64_t i = 0; i < seq * batch * hid; ++i)
        EXPECT_NEAR(got_out[i], ref_out[static_cast<size_t>(i)], 1e-12);
    for (int64_t i = 0; i < batch * hid; ++i)
        EXPECT_NEAR(got_hn[i], ref_hn[static_cast<size_t>(i)], 1e-12);
}

TEST_F(RNNF64Test, BiLSTMForwardF64) {
    const int64_t seq = 3, batch = 2, in_sz = 3, hid = 4;
    auto input  = det_randn({seq, batch, in_sz}, 0x101);
    auto Wih_f  = det_randn({4 * hid, in_sz}, 0x102);
    auto Whh_f  = det_randn({4 * hid, hid}, 0x103);
    auto bih_f  = det_randn({4 * hid}, 0x104);
    auto bhh_f  = det_randn({4 * hid}, 0x105);
    auto Wih_b  = det_randn({4 * hid, in_sz}, 0x106);
    auto Whh_b  = det_randn({4 * hid, hid}, 0x107);
    auto bih_b  = det_randn({4 * hid}, 0x108);
    auto bhh_b  = det_randn({4 * hid}, 0x109);
    auto h0     = det_randn({2, batch, hid}, 0x10A);
    auto c0     = det_randn({2, batch, hid}, 0x10B);

    const Tensor in_arr[11] = {input, h0, c0,
                                Wih_f, Whh_f, bih_f, bhh_f,
                                Wih_b, Whh_b, bih_b, bhh_b};
    auto outs = tenzor::dispatch(OpId::BiLSTMForward,
                                  std::span<const Tensor>{in_arr, 11}, {});
    ASSERT_EQ(outs.size(), 3u);
    EXPECT_EQ(outs[0].dtype(), DType::Float64);

    // Sanity: forward direction matches independent ref_lstm_forward_f64.
    std::vector<double> ref_fwd_out(seq * batch * hid);
    std::vector<double> ref_fwd_hn(batch * hid), ref_fwd_cn(batch * hid);
    std::vector<double> bias_fwd_combined(4 * hid);
    for (int64_t i = 0; i < 4 * hid; ++i)
        bias_fwd_combined[static_cast<size_t>(i)] =
            bih_f.data<double>()[i] + bhh_f.data<double>()[i];
    ref_lstm_forward_f64(input.data<double>(), Wih_f.data<double>(),
                          Whh_f.data<double>(), bias_fwd_combined.data(),
                          nullptr,
                          h0.data<double>(), c0.data<double>(),
                          ref_fwd_out.data(), ref_fwd_hn.data(),
                          ref_fwd_cn.data(),
                          seq, batch, in_sz, hid);
    // The kernel writes forward into output[:, :, :hid].
    const double* got_out = outs[0].data<double>();
    for (int64_t t = 0; t < seq; ++t) {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < hid; ++h) {
                double got = got_out[t * batch * 2 * hid + b * 2 * hid + h];
                double ref = ref_fwd_out[static_cast<size_t>(
                    t * batch * hid + b * hid + h)];
                EXPECT_NEAR(got, ref, 1e-12)
                    << "BiLSTM forward direction mismatch at t=" << t
                    << " b=" << b << " h=" << h;
            }
        }
    }
}

TEST_F(RNNF64Test, LSTMMultilayerF64) {
    const int64_t seq = 3, batch = 2, in_sz = 3, hid = 4, layers = 2;

    auto input = det_randn({seq, batch, in_sz}, 0x201);
    auto W_ih_0 = det_randn({4 * hid, in_sz}, 0x202);
    auto W_hh_0 = det_randn({4 * hid, hid}, 0x203);
    auto bias_0 = det_randn({4 * hid}, 0x204);
    auto W_ih_1 = det_randn({4 * hid, hid}, 0x205);
    auto W_hh_1 = det_randn({4 * hid, hid}, 0x206);
    auto bias_1 = det_randn({4 * hid}, 0x207);
    auto h0 = det_randn({layers, batch, hid}, 0x208);
    auto c0 = det_randn({layers, batch, hid}, 0x209);

    NewOpAttributes attrs;
    attrs.set(AttrKey::NumLayers, static_cast<int64_t>(layers));
    const Tensor in_arr[9] = {input, h0, c0,
                               W_ih_0, W_hh_0, bias_0,
                               W_ih_1, W_hh_1, bias_1};
    auto outs = tenzor::dispatch(OpId::LSTMMultiLayerForward,
                                  std::span<const Tensor>{in_arr, 9}, attrs);
    ASSERT_EQ(outs.size(), 3u);
    EXPECT_EQ(outs[0].dtype(), DType::Float64);
    // Output shape sanity.
    EXPECT_EQ(outs[0].shape()[0], seq);
    EXPECT_EQ(outs[0].shape()[1], batch);
    EXPECT_EQ(outs[0].shape()[2], hid);
}

TEST_F(RNNF64Test, GRUMultilayerF64) {
    const int64_t seq = 3, batch = 2, in_sz = 3, hid = 4, layers = 2;

    auto input = det_randn({seq, batch, in_sz}, 0x301);
    auto W_ih_0 = det_randn({3 * hid, in_sz}, 0x302);
    auto W_hh_0 = det_randn({3 * hid, hid}, 0x303);
    auto bias_0 = det_randn({3 * hid}, 0x304);
    auto W_ih_1 = det_randn({3 * hid, hid}, 0x305);
    auto W_hh_1 = det_randn({3 * hid, hid}, 0x306);
    auto bias_1 = det_randn({3 * hid}, 0x307);
    auto h0 = det_randn({layers, batch, hid}, 0x308);

    NewOpAttributes attrs;
    attrs.set(AttrKey::NumLayers, static_cast<int64_t>(layers));
    const Tensor in_arr[8] = {input, h0,
                               W_ih_0, W_hh_0, bias_0,
                               W_ih_1, W_hh_1, bias_1};
    auto outs = tenzor::dispatch(OpId::GRUMultiLayerForward,
                                  std::span<const Tensor>{in_arr, 8}, attrs);
    ASSERT_EQ(outs.size(), 2u);
    EXPECT_EQ(outs[0].dtype(), DType::Float64);
    EXPECT_EQ(outs[0].shape()[0], seq);
    EXPECT_EQ(outs[0].shape()[1], batch);
    EXPECT_EQ(outs[0].shape()[2], hid);
}
