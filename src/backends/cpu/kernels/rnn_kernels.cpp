/**
 * @file rnn_kernels.cpp
 * @brief CPU RNN kernel implementations (LSTM, GRU)
 *
 * Includes both cell-level and full-sequence implementations:
 * - Cell-level: lstm_cell_forward_kernel, gru_cell_forward_kernel
 * - Full-sequence: lstm_forward_kernel, gru_forward_kernel (fused, SIMD-optimized)
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "fused_lstm.hpp"  // SIMD-optimized fused kernels
#include "rnn_onednn.hpp"  // oneDNN-accelerated LSTM/GRU (re-enabled with fixes)
#include <cmath>

namespace tenzor {
namespace cpu {

// Sigmoid activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// =============================================================================
// S12: Float64 native full-sequence implementations
// =============================================================================
// The fused_lstm.hpp SIMD helpers (lstm::lstm_forward, lstm::gru_forward,
// lstm::lstm_forward_bidirectional) are Float32-only. To honour the "no
// Float64 → Float32 round-trip" rule, full-sequence kernels need parallel
// Float64 implementations that operate directly on `double`. These are
// scalar/OpenMP loops — the inner GEMMs are not BLAS-routed, which is fine
// for the small RNN shapes our tests/users hit. If a performance regression
// shows up for large Float64 RNN workloads, the right next step is to swap
// these inner loops for cblas_dgemm — the BLAS contract is identical to
// what the Float32 path uses.
namespace {

// audit C1: register the RNN oneDNN cache clear-callback for clear_dnnl_cache().
struct RnnOneDnnCacheClearRegistrar {
    RnnOneDnnCacheClearRegistrar() {
        ::tenzor::cpu::register_dnnl_cache_clear_callback(
            &::tenzor::cpu::rnn_onednn::clear_rnn_onednn_caches);
    }
};
static RnnOneDnnCacheClearRegistrar g_rnn_onednn_cache_clear_registrar;

inline double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// LSTM single-direction forward, Float64. Same semantics as
// lstm::lstm_forward (gates ordered i, f, g, o, applied as per PyTorch).
inline void lstm_forward_f64(
    const double* input,     // (seq_len, batch, input_size)
    const double* W_ih,      // (4*hidden, input_size)
    const double* W_hh,      // (4*hidden, hidden)
    const double* bias,      // (4*hidden) or nullptr (combined ih+hh)
    const double* h0,        // (batch, hidden)
    const double* c0,        // (batch, hidden)
    double* output,          // (seq_len, batch, hidden)
    double* h_n,             // (batch, hidden)
    double* c_n,             // (batch, hidden)
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden
) {
    int64_t gate_size = 4 * hidden;
    std::vector<double> h_curr(batch * hidden);
    std::vector<double> c_curr(batch * hidden);
    std::memcpy(h_curr.data(), h0, batch * hidden * sizeof(double));
    std::memcpy(c_curr.data(), c0, batch * hidden * sizeof(double));

    for (int64_t t = 0; t < seq_len; ++t) {
        const double* in_t = input + t * batch * input_size;
        double* out_t = output + t * batch * hidden;

        #pragma omp parallel for if(batch > 1)
        for (int64_t b = 0; b < batch; ++b) {
            std::vector<double> gates(gate_size);
            for (int64_t g = 0; g < gate_size; ++g) {
                double sum = bias ? bias[g] : 0.0;
                for (int64_t i = 0; i < input_size; ++i) {
                    sum += in_t[b * input_size + i] * W_ih[g * input_size + i];
                }
                for (int64_t h = 0; h < hidden; ++h) {
                    sum += h_curr[b * hidden + h] * W_hh[g * hidden + h];
                }
                gates[g] = sum;
            }
            for (int64_t h = 0; h < hidden; ++h) {
                double i_gate = sigmoid_d(gates[h]);
                double f_gate = sigmoid_d(gates[hidden + h]);
                double g_gate = std::tanh(gates[2 * hidden + h]);
                double o_gate = sigmoid_d(gates[3 * hidden + h]);
                double c_new = f_gate * c_curr[b * hidden + h] + i_gate * g_gate;
                double h_new = o_gate * std::tanh(c_new);
                out_t[b * hidden + h] = h_new;
                c_curr[b * hidden + h] = c_new;
            }
        }
        // h_curr := out_t (last timestep's output)
        std::memcpy(h_curr.data(), out_t, batch * hidden * sizeof(double));
    }

    std::memcpy(h_n, h_curr.data(), batch * hidden * sizeof(double));
    std::memcpy(c_n, c_curr.data(), batch * hidden * sizeof(double));
}

// Bidirectional LSTM, Float64. output is (seq_len, batch, 2*hidden) —
// forward direction occupies the first `hidden` columns, backward the next.
inline void lstm_forward_bidirectional_f64(
    const double* input,
    const double* W_ih_fwd, const double* W_hh_fwd, const double* bias_fwd,
    const double* W_ih_bwd, const double* W_hh_bwd, const double* bias_bwd,
    const double* h0_fwd, const double* c0_fwd,
    const double* h0_bwd, const double* c0_bwd,
    double* output,        // (seq_len, batch, 2*hidden)
    double* h_n_fwd, double* c_n_fwd,
    double* h_n_bwd, double* c_n_bwd,
    int64_t seq_len, int64_t batch, int64_t input_size, int64_t hidden
) {
    int64_t gate_size = 4 * hidden;

    // Forward direction
    {
        std::vector<double> h_curr(batch * hidden);
        std::vector<double> c_curr(batch * hidden);
        std::memcpy(h_curr.data(), h0_fwd, batch * hidden * sizeof(double));
        std::memcpy(c_curr.data(), c0_fwd, batch * hidden * sizeof(double));
        for (int64_t t = 0; t < seq_len; ++t) {
            const double* in_t = input + t * batch * input_size;
            double* out_t = output + t * batch * 2 * hidden;  // forward goes into [:, :, :hidden]
            #pragma omp parallel for if(batch > 1)
            for (int64_t b = 0; b < batch; ++b) {
                std::vector<double> gates(gate_size);
                for (int64_t g = 0; g < gate_size; ++g) {
                    double sum = bias_fwd ? bias_fwd[g] : 0.0;
                    for (int64_t i = 0; i < input_size; ++i) {
                        sum += in_t[b * input_size + i] * W_ih_fwd[g * input_size + i];
                    }
                    for (int64_t h = 0; h < hidden; ++h) {
                        sum += h_curr[b * hidden + h] * W_hh_fwd[g * hidden + h];
                    }
                    gates[g] = sum;
                }
                for (int64_t h = 0; h < hidden; ++h) {
                    double i_g = sigmoid_d(gates[h]);
                    double f_g = sigmoid_d(gates[hidden + h]);
                    double g_g = std::tanh(gates[2 * hidden + h]);
                    double o_g = sigmoid_d(gates[3 * hidden + h]);
                    double c_new = f_g * c_curr[b * hidden + h] + i_g * g_g;
                    double h_new = o_g * std::tanh(c_new);
                    // Write into forward slot [b, :hidden]
                    out_t[b * 2 * hidden + h] = h_new;
                    c_curr[b * hidden + h] = c_new;
                }
            }
            // Update h_curr to the just-written forward outputs
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t h = 0; h < hidden; ++h) {
                    h_curr[b * hidden + h] = out_t[b * 2 * hidden + h];
                }
            }
        }
        std::memcpy(h_n_fwd, h_curr.data(), batch * hidden * sizeof(double));
        std::memcpy(c_n_fwd, c_curr.data(), batch * hidden * sizeof(double));
    }

    // Backward direction
    {
        std::vector<double> h_curr(batch * hidden);
        std::vector<double> c_curr(batch * hidden);
        std::memcpy(h_curr.data(), h0_bwd, batch * hidden * sizeof(double));
        std::memcpy(c_curr.data(), c0_bwd, batch * hidden * sizeof(double));
        for (int64_t t = seq_len - 1; t >= 0; --t) {
            const double* in_t = input + t * batch * input_size;
            double* out_t = output + t * batch * 2 * hidden;  // backward goes into [:, :, hidden:]
            #pragma omp parallel for if(batch > 1)
            for (int64_t b = 0; b < batch; ++b) {
                std::vector<double> gates(gate_size);
                for (int64_t g = 0; g < gate_size; ++g) {
                    double sum = bias_bwd ? bias_bwd[g] : 0.0;
                    for (int64_t i = 0; i < input_size; ++i) {
                        sum += in_t[b * input_size + i] * W_ih_bwd[g * input_size + i];
                    }
                    for (int64_t h = 0; h < hidden; ++h) {
                        sum += h_curr[b * hidden + h] * W_hh_bwd[g * hidden + h];
                    }
                    gates[g] = sum;
                }
                for (int64_t h = 0; h < hidden; ++h) {
                    double i_g = sigmoid_d(gates[h]);
                    double f_g = sigmoid_d(gates[hidden + h]);
                    double g_g = std::tanh(gates[2 * hidden + h]);
                    double o_g = sigmoid_d(gates[3 * hidden + h]);
                    double c_new = f_g * c_curr[b * hidden + h] + i_g * g_g;
                    double h_new = o_g * std::tanh(c_new);
                    out_t[b * 2 * hidden + hidden + h] = h_new;
                    c_curr[b * hidden + h] = c_new;
                }
            }
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t h = 0; h < hidden; ++h) {
                    h_curr[b * hidden + h] = out_t[b * 2 * hidden + hidden + h];
                }
            }
        }
        std::memcpy(h_n_bwd, h_curr.data(), batch * hidden * sizeof(double));
        std::memcpy(c_n_bwd, c_curr.data(), batch * hidden * sizeof(double));
    }
}

// GRU single-direction forward, Float64. Matches PyTorch contract:
//   n = tanh((W_in @ x + b_in) + r * (W_hn @ h + b_hn))
//   h_new = (1 - z) * n + z * h_prev
// Caller passes bias_ih and bias_hh separately (either may be null).
inline void gru_forward_f64(
    const double* input,     // (seq_len, batch, input_size)
    const double* W_ih,      // (3*hidden, input_size)  layout: r, z, n
    const double* W_hh,      // (3*hidden, hidden)
    const double* bias_ih,   // (3*hidden) or nullptr
    const double* bias_hh,   // (3*hidden) or nullptr
    const double* h0,        // (batch, hidden)
    double* output,          // (seq_len, batch, hidden)
    double* h_n,             // (batch, hidden)
    int64_t seq_len,
    int64_t batch,
    int64_t input_size,
    int64_t hidden
) {
    std::vector<double> h_curr(batch * hidden);
    std::memcpy(h_curr.data(), h0, batch * hidden * sizeof(double));

    for (int64_t t = 0; t < seq_len; ++t) {
        const double* in_t = input + t * batch * input_size;
        double* out_t = output + t * batch * hidden;

        #pragma omp parallel for if(batch > 1)
        for (int64_t b = 0; b < batch; ++b) {
            // r, z gates: sigmoid of (W_ih @ x + b_ih + W_hh @ h + b_hh)
            std::vector<double> rz_pre(2 * hidden);
            for (int64_t g = 0; g < 2 * hidden; ++g) {
                double sum = (bias_ih ? bias_ih[g] : 0.0) + (bias_hh ? bias_hh[g] : 0.0);
                for (int64_t i = 0; i < input_size; ++i) {
                    sum += in_t[b * input_size + i] * W_ih[g * input_size + i];
                }
                for (int64_t h = 0; h < hidden; ++h) {
                    sum += h_curr[b * hidden + h] * W_hh[g * hidden + h];
                }
                rz_pre[g] = sum;
            }

            for (int64_t h = 0; h < hidden; ++h) {
                double r = sigmoid_d(rz_pre[h]);
                double z = sigmoid_d(rz_pre[hidden + h]);

                double n_ih = bias_ih ? bias_ih[2 * hidden + h] : 0.0;
                double n_hh = bias_hh ? bias_hh[2 * hidden + h] : 0.0;
                for (int64_t i = 0; i < input_size; ++i) {
                    n_ih += in_t[b * input_size + i] * W_ih[(2 * hidden + h) * input_size + i];
                }
                for (int64_t hh = 0; hh < hidden; ++hh) {
                    n_hh += h_curr[b * hidden + hh] * W_hh[(2 * hidden + h) * hidden + hh];
                }
                n_hh *= r;
                double n = std::tanh(n_ih + n_hh);
                out_t[b * hidden + h] = (1.0 - z) * n + z * h_curr[b * hidden + h];
            }
        }
        std::memcpy(h_curr.data(), out_t, batch * hidden * sizeof(double));
    }

    std::memcpy(h_n, h_curr.data(), batch * hidden * sizeof(double));
}

}  // anonymous namespace

auto lstm_cell_forward_kernel(const Tensor& input, const Tensor& hx, const Tensor& cx,
                               const Tensor& weight_ih, const Tensor& weight_hh,
                               const Tensor& bias_ih, const Tensor& bias_hh)
    -> std::vector<Tensor> {
    // Contiguity guard (mirrors the full-sequence lstm_forward_kernel ~:852).
    // The typed body below indexes input/hx/cx/weights with shape-derived
    // strides (in_data[b*input_size+i], w_ih_data[g*input_size+i], ...), which
    // assumes contiguous storage; a strided view silently produces wrong cells.
    if (!input.is_contiguous() || !hx.is_contiguous() || !cx.is_contiguous() ||
        !weight_ih.is_contiguous() || !weight_hh.is_contiguous() ||
        (bias_ih.numel() > 0 && !bias_ih.is_contiguous()) ||
        (bias_hh.numel() > 0 && !bias_hh.is_contiguous())) {
        return lstm_cell_forward_kernel(
            input.contiguous(), hx.contiguous(), cx.contiguous(),
            weight_ih.contiguous(), weight_hh.contiguous(),
            bias_ih.numel() > 0 ? bias_ih.contiguous() : bias_ih,
            bias_hh.numel() > 0 ? bias_hh.contiguous() : bias_hh);
    }
    // S12: Float32 native, Float64 native (avoids Float32 round-trip that
    // corrupted Float64 gradcheck by an ε(F32)·signal margin). Float16/
    // BFloat16 stay on the widen-narrow path — half-precision compute is
    // not safe natively and round-tripping through Float32 is the
    // documented PyTorch behaviour for half-precision RNN cells.
    auto run = [&]<typename T>() -> std::vector<Tensor> {
        // input: [batch, input_size]
        // hx, cx: [batch, hidden_size]
        // weight_ih: [4 * hidden_size, input_size]
        // weight_hh: [4 * hidden_size, hidden_size]
        // bias_ih, bias_hh: [4 * hidden_size] (each may be empty)

        auto in_shape = input.shape();
        auto hx_shape = hx.shape();
        int64_t batch_size = in_shape[0];
        int64_t input_size = in_shape[1];
        int64_t hidden_size = hx_shape[1];

        auto hy = Tensor::empty_uninitialized({batch_size, hidden_size}, input.dtype(), input.device());
        auto cy = Tensor::empty_uninitialized({batch_size, hidden_size}, input.dtype(), input.device());

        const T* in_data = input.data<T>();
        const T* hx_data = hx.data<T>();
        const T* cx_data = cx.data<T>();
        const T* w_ih_data = weight_ih.data<T>();
        const T* w_hh_data = weight_hh.data<T>();
        const T* b_ih_data = bias_ih.numel() > 0 ? bias_ih.data<T>() : nullptr;
        const T* b_hh_data = bias_hh.numel() > 0 ? bias_hh.data<T>() : nullptr;
        T* hy_data = hy.data<T>();
        T* cy_data = cy.data<T>();

        auto sigmoid_t = [](T x) -> T {
            return T(1) / (T(1) + std::exp(-x));
        };

        #pragma omp parallel for
        for (int64_t b = 0; b < batch_size; ++b) {
            std::vector<T> gates(4 * hidden_size);

            for (int64_t g = 0; g < 4 * hidden_size; ++g) {
                T sum = (b_ih_data ? b_ih_data[g] : T(0)) + (b_hh_data ? b_hh_data[g] : T(0));
                for (int64_t i = 0; i < input_size; ++i) {
                    sum += in_data[b * input_size + i] * w_ih_data[g * input_size + i];
                }
                for (int64_t h = 0; h < hidden_size; ++h) {
                    sum += hx_data[b * hidden_size + h] * w_hh_data[g * hidden_size + h];
                }
                gates[g] = sum;
            }

            for (int64_t h = 0; h < hidden_size; ++h) {
                T i_gate = sigmoid_t(gates[h]);
                T f_gate = sigmoid_t(gates[hidden_size + h]);
                T g_gate = std::tanh(gates[2 * hidden_size + h]);
                T o_gate = sigmoid_t(gates[3 * hidden_size + h]);

                T c_new = f_gate * cx_data[b * hidden_size + h] + i_gate * g_gate;
                T h_new = o_gate * std::tanh(c_new);

                cy_data[b * hidden_size + h] = c_new;
                hy_data[b * hidden_size + h] = h_new;
            }
        }

        return {hy, cy};
    };

    if (input.dtype() == DType::Float32) {
        return run.template operator()<float>();
    } else if (input.dtype() == DType::Float64) {
        return run.template operator()<double>();
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Widen-narrow: half precision is not safe natively. Round-trip
        // through Float32 matches PyTorch behaviour for half RNN cells.
        DType orig = input.dtype();
        auto results = lstm_cell_forward_kernel(
            input.to(DType::Float32), hx.to(DType::Float32), cx.to(DType::Float32),
            weight_ih.to(DType::Float32), weight_hh.to(DType::Float32),
            bias_ih.to(DType::Float32), bias_hh.to(DType::Float32));
        for (auto& t : results) t = t.to(orig);
        return results;
    } else {
        throw std::runtime_error("lstm_cell_forward_kernel: unsupported dtype");
    }
}

auto lstm_cell_backward_kernel(const Tensor& grad_hy, const Tensor& grad_cy,
                                const Tensor& input, const Tensor& hx, const Tensor& cx,
                                const Tensor& hy, const Tensor& cy,
                                const Tensor& weight_ih, const Tensor& weight_hh,
                                const Tensor& bias_ih, const Tensor& bias_hh)
    -> std::vector<Tensor> {
    // Contiguity guard (mirrors the full-sequence kernels). The typed body
    // indexes every input with shape-derived strides, which assumes contiguous
    // storage; a strided view silently produces wrong gradients.
    if (!grad_hy.is_contiguous() || !grad_cy.is_contiguous() ||
        !input.is_contiguous() || !hx.is_contiguous() || !cx.is_contiguous() ||
        !hy.is_contiguous() || !cy.is_contiguous() ||
        !weight_ih.is_contiguous() || !weight_hh.is_contiguous() ||
        (bias_ih.numel() > 0 && !bias_ih.is_contiguous()) ||
        (bias_hh.numel() > 0 && !bias_hh.is_contiguous())) {
        return lstm_cell_backward_kernel(
            grad_hy.contiguous(), grad_cy.contiguous(),
            input.contiguous(), hx.contiguous(), cx.contiguous(),
            hy.contiguous(), cy.contiguous(),
            weight_ih.contiguous(), weight_hh.contiguous(),
            bias_ih.numel() > 0 ? bias_ih.contiguous() : bias_ih,
            bias_hh.numel() > 0 ? bias_hh.contiguous() : bias_hh);
    }
    // S12: native Float32 / Float64. See lstm_cell_forward_kernel comment.
    auto run = [&]<typename T>() -> std::vector<Tensor> {
        auto shape = grad_hy.shape();
        int64_t batch_size = shape[0];
        int64_t hidden_size = shape[1];
        int64_t input_size = input.shape()[1];

        auto grad_input = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                input.dtype(), input.device());
        auto grad_hx = zeros(std::vector<int64_t>(hx.shape().begin(), hx.shape().end()),
                             hx.dtype(), hx.device());
        auto grad_cx_out = zeros(std::vector<int64_t>(cx.shape().begin(), cx.shape().end()),
                                 cx.dtype(), cx.device());
        auto grad_weight_ih = zeros(std::vector<int64_t>(weight_ih.shape().begin(), weight_ih.shape().end()),
                                    weight_ih.dtype(), weight_ih.device());
        auto grad_weight_hh = zeros(std::vector<int64_t>(weight_hh.shape().begin(), weight_hh.shape().end()),
                                    weight_hh.dtype(), weight_hh.device());
        auto grad_bias_ih = zeros({4 * hidden_size}, input.dtype(), input.device());
        auto grad_bias_hh = zeros({4 * hidden_size}, input.dtype(), input.device());

        const T* in_data = input.data<T>();
        const T* hx_data = hx.data<T>();
        const T* cx_data = cx.data<T>();
        const T* w_ih_data = weight_ih.data<T>();
        const T* w_hh_data = weight_hh.data<T>();
        const T* dhy_data = grad_hy.data<T>();
        const T* dcy_data = grad_cy.data<T>();
        const T* b_ih_data = bias_ih.numel() > 0 ? bias_ih.data<T>() : nullptr;
        const T* b_hh_data = bias_hh.numel() > 0 ? bias_hh.data<T>() : nullptr;

        T* d_input = grad_input.data<T>();
        T* d_hx = grad_hx.data<T>();
        T* d_cx = grad_cx_out.data<T>();
        T* d_w_ih = grad_weight_ih.data<T>();
        T* d_w_hh = grad_weight_hh.data<T>();
        T* d_b_ih = grad_bias_ih.data<T>();
        T* d_b_hh = grad_bias_hh.data<T>();

        auto sigmoid_t = [](T x) -> T { return T(1) / (T(1) + std::exp(-x)); };

        #pragma omp parallel if(batch_size > 4)
        {
            std::vector<T> t_d_w_ih(4 * hidden_size * input_size, T(0));
            std::vector<T> t_d_w_hh(4 * hidden_size * hidden_size, T(0));
            std::vector<T> t_d_b(4 * hidden_size, T(0));

            #pragma omp for
            for (int64_t b = 0; b < batch_size; ++b) {
                std::vector<T> gates(4 * hidden_size);
                for (int64_t g = 0; g < 4 * hidden_size; ++g) {
                    T sum = (b_ih_data ? b_ih_data[g] : T(0)) + (b_hh_data ? b_hh_data[g] : T(0));
                    for (int64_t i = 0; i < input_size; ++i) {
                        sum += in_data[b * input_size + i] * w_ih_data[g * input_size + i];
                    }
                    for (int64_t h = 0; h < hidden_size; ++h) {
                        sum += hx_data[b * hidden_size + h] * w_hh_data[g * hidden_size + h];
                    }
                    gates[g] = sum;
                }

                std::vector<T> i_gate(hidden_size), f_gate(hidden_size);
                std::vector<T> g_gate(hidden_size), o_gate(hidden_size);
                std::vector<T> c_new(hidden_size), tanh_c(hidden_size);

                for (int64_t h = 0; h < hidden_size; ++h) {
                    i_gate[h] = sigmoid_t(gates[h]);
                    f_gate[h] = sigmoid_t(gates[hidden_size + h]);
                    g_gate[h] = std::tanh(gates[2 * hidden_size + h]);
                    o_gate[h] = sigmoid_t(gates[3 * hidden_size + h]);
                    c_new[h] = f_gate[h] * cx_data[b * hidden_size + h] + i_gate[h] * g_gate[h];
                    tanh_c[h] = std::tanh(c_new[h]);
                }

                std::vector<T> d_gates(4 * hidden_size, T(0));

                for (int64_t h = 0; h < hidden_size; ++h) {
                    T dh = dhy_data[b * hidden_size + h];
                    T dc = dcy_data[b * hidden_size + h];

                    T d_o = dh * tanh_c[h];
                    dc += dh * o_gate[h] * (T(1) - tanh_c[h] * tanh_c[h]);

                    T d_f = dc * cx_data[b * hidden_size + h];
                    T d_i = dc * g_gate[h];
                    T d_g = dc * i_gate[h];
                    d_cx[b * hidden_size + h] = dc * f_gate[h];

                    d_gates[h] = d_i * i_gate[h] * (T(1) - i_gate[h]);
                    d_gates[hidden_size + h] = d_f * f_gate[h] * (T(1) - f_gate[h]);
                    d_gates[2 * hidden_size + h] = d_g * (T(1) - g_gate[h] * g_gate[h]);
                    d_gates[3 * hidden_size + h] = d_o * o_gate[h] * (T(1) - o_gate[h]);
                }

                for (int64_t i = 0; i < input_size; ++i) {
                    T sum = T(0);
                    for (int64_t g = 0; g < 4 * hidden_size; ++g) {
                        sum += d_gates[g] * w_ih_data[g * input_size + i];
                    }
                    d_input[b * input_size + i] = sum;
                }

                for (int64_t h = 0; h < hidden_size; ++h) {
                    T sum = T(0);
                    for (int64_t g = 0; g < 4 * hidden_size; ++g) {
                        sum += d_gates[g] * w_hh_data[g * hidden_size + h];
                    }
                    d_hx[b * hidden_size + h] = sum;
                }

                for (int64_t g = 0; g < 4 * hidden_size; ++g) {
                    for (int64_t i = 0; i < input_size; ++i) {
                        t_d_w_ih[g * input_size + i] += d_gates[g] * in_data[b * input_size + i];
                    }
                    for (int64_t h = 0; h < hidden_size; ++h) {
                        t_d_w_hh[g * hidden_size + h] += d_gates[g] * hx_data[b * hidden_size + h];
                    }
                    t_d_b[g] += d_gates[g];
                }
            }

            #pragma omp critical
            {
                for (int64_t i = 0; i < 4 * hidden_size * input_size; ++i) d_w_ih[i] += t_d_w_ih[i];
                for (int64_t i = 0; i < 4 * hidden_size * hidden_size; ++i) d_w_hh[i] += t_d_w_hh[i];
                for (int64_t i = 0; i < 4 * hidden_size; ++i) {
                    d_b_ih[i] += t_d_b[i];
                    d_b_hh[i] += t_d_b[i];
                }
            }
        }

        return {grad_input, grad_hx, grad_cx_out, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh};
    };

    if (grad_hy.dtype() == DType::Float32) {
        return run.template operator()<float>();
    } else if (grad_hy.dtype() == DType::Float64) {
        return run.template operator()<double>();
    } else if (grad_hy.dtype() == DType::Float16 || grad_hy.dtype() == DType::BFloat16) {
        DType orig = grad_hy.dtype();
        auto results = lstm_cell_backward_kernel(
            grad_hy.to(DType::Float32), grad_cy.to(DType::Float32),
            input.to(DType::Float32), hx.to(DType::Float32), cx.to(DType::Float32),
            hy.to(DType::Float32), cy.to(DType::Float32),
            weight_ih.to(DType::Float32), weight_hh.to(DType::Float32),
            bias_ih.to(DType::Float32), bias_hh.to(DType::Float32));
        for (auto& t : results) t = t.to(orig);
        return results;
    } else {
        throw std::runtime_error("lstm_cell_backward_kernel: unsupported dtype");
    }
}

auto gru_cell_forward_kernel(const Tensor& input, const Tensor& hx,
                              const Tensor& weight_ih, const Tensor& weight_hh,
                              const Tensor& bias_ih, const Tensor& bias_hh) -> Tensor {
    // Contiguity guard (mirrors the full-sequence gru_forward_kernel ~:1200).
    // The typed body indexes input/hx/weights with shape-derived strides, which
    // assumes contiguous storage; a strided view silently produces wrong cells.
    if (!input.is_contiguous() || !hx.is_contiguous() ||
        !weight_ih.is_contiguous() || !weight_hh.is_contiguous() ||
        (bias_ih.numel() > 0 && !bias_ih.is_contiguous()) ||
        (bias_hh.numel() > 0 && !bias_hh.is_contiguous())) {
        return gru_cell_forward_kernel(
            input.contiguous(), hx.contiguous(),
            weight_ih.contiguous(), weight_hh.contiguous(),
            bias_ih.numel() > 0 ? bias_ih.contiguous() : bias_ih,
            bias_hh.numel() > 0 ? bias_hh.contiguous() : bias_hh);
    }
    // S12: native Float32 / Float64.
    auto run = [&]<typename T>() -> Tensor {
        auto in_shape = input.shape();
        auto hx_shape = hx.shape();
        int64_t batch_size = in_shape[0];
        int64_t input_size = in_shape[1];
        int64_t hidden_size = hx_shape[1];

        auto hy = Tensor::empty_uninitialized({batch_size, hidden_size}, input.dtype(), input.device());

        const T* in_data = input.data<T>();
        const T* hx_data = hx.data<T>();
        const T* w_ih_data = weight_ih.data<T>();
        const T* w_hh_data = weight_hh.data<T>();
        const T* b_ih_data = bias_ih.numel() > 0 ? bias_ih.data<T>() : nullptr;
        const T* b_hh_data = bias_hh.numel() > 0 ? bias_hh.data<T>() : nullptr;
        T* hy_data = hy.data<T>();

        auto sigmoid_t = [](T x) -> T { return T(1) / (T(1) + std::exp(-x)); };

        #pragma omp parallel for
        for (int64_t b = 0; b < batch_size; ++b) {
            std::vector<T> gates_rz(2 * hidden_size);
            for (int64_t g = 0; g < 2 * hidden_size; ++g) {
                T sum = (b_ih_data ? b_ih_data[g] : T(0)) +
                        (b_hh_data ? b_hh_data[g] : T(0));
                for (int64_t i = 0; i < input_size; ++i) {
                    sum += in_data[b * input_size + i] * w_ih_data[g * input_size + i];
                }
                for (int64_t h = 0; h < hidden_size; ++h) {
                    sum += hx_data[b * hidden_size + h] * w_hh_data[g * hidden_size + h];
                }
                gates_rz[g] = sigmoid_t(sum);
            }

            for (int64_t h = 0; h < hidden_size; ++h) {
                T r = gates_rz[h];
                T z = gates_rz[hidden_size + h];

                T n_ih = b_ih_data ? b_ih_data[2 * hidden_size + h] : T(0);
                T n_hh = b_hh_data ? b_hh_data[2 * hidden_size + h] : T(0);

                for (int64_t i = 0; i < input_size; ++i) {
                    n_ih += in_data[b * input_size + i] * w_ih_data[(2 * hidden_size + h) * input_size + i];
                }
                for (int64_t hh = 0; hh < hidden_size; ++hh) {
                    n_hh += hx_data[b * hidden_size + hh] *
                            w_hh_data[(2 * hidden_size + h) * hidden_size + hh];
                }
                n_hh *= r;

                T n = std::tanh(n_ih + n_hh);
                hy_data[b * hidden_size + h] = (T(1) - z) * n + z * hx_data[b * hidden_size + h];
            }
        }

        return hy;
    };

    if (input.dtype() == DType::Float32) {
        return run.template operator()<float>();
    } else if (input.dtype() == DType::Float64) {
        return run.template operator()<double>();
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto result = gru_cell_forward_kernel(
            input.to(DType::Float32), hx.to(DType::Float32),
            weight_ih.to(DType::Float32), weight_hh.to(DType::Float32),
            bias_ih.to(DType::Float32), bias_hh.to(DType::Float32));
        return result.to(orig);
    } else {
        throw std::runtime_error("gru_cell_forward_kernel: unsupported dtype");
    }
}

auto gru_cell_backward_kernel(const Tensor& grad_hy, const Tensor& input, const Tensor& hx,
                               const Tensor& weight_ih, const Tensor& weight_hh,
                               const Tensor& bias_ih, const Tensor& bias_hh)
    -> std::vector<Tensor> {
    // Contiguity guard (mirrors the full-sequence kernels). The typed body
    // indexes every input with shape-derived strides, which assumes contiguous
    // storage; a strided view silently produces wrong gradients.
    if (!grad_hy.is_contiguous() || !input.is_contiguous() ||
        !hx.is_contiguous() || !weight_ih.is_contiguous() ||
        !weight_hh.is_contiguous() ||
        (bias_ih.numel() > 0 && !bias_ih.is_contiguous()) ||
        (bias_hh.numel() > 0 && !bias_hh.is_contiguous())) {
        return gru_cell_backward_kernel(
            grad_hy.contiguous(), input.contiguous(), hx.contiguous(),
            weight_ih.contiguous(), weight_hh.contiguous(),
            bias_ih.numel() > 0 ? bias_ih.contiguous() : bias_ih,
            bias_hh.numel() > 0 ? bias_hh.contiguous() : bias_hh);
    }
    // S12: native Float32 / Float64.
    auto run = [&]<typename T>() -> std::vector<Tensor> {
        auto shape = grad_hy.shape();
        int64_t batch_size = shape[0];
        int64_t hidden_size = shape[1];
        int64_t input_size = input.shape()[1];

        auto grad_input = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                input.dtype(), input.device());
        auto grad_hx = zeros(std::vector<int64_t>(hx.shape().begin(), hx.shape().end()),
                             hx.dtype(), hx.device());
        auto grad_weight_ih = zeros(std::vector<int64_t>(weight_ih.shape().begin(), weight_ih.shape().end()),
                                    weight_ih.dtype(), weight_ih.device());
        auto grad_weight_hh = zeros(std::vector<int64_t>(weight_hh.shape().begin(), weight_hh.shape().end()),
                                    weight_hh.dtype(), weight_hh.device());
        auto grad_bias_ih = zeros({3 * hidden_size}, input.dtype(), input.device());
        auto grad_bias_hh = zeros({3 * hidden_size}, input.dtype(), input.device());

        const T* in_data = input.data<T>();
        const T* hx_data = hx.data<T>();
        const T* w_ih_data = weight_ih.data<T>();
        const T* w_hh_data = weight_hh.data<T>();
        const T* dhy_data = grad_hy.data<T>();
        const T* b_ih_data = bias_ih.numel() > 0 ? bias_ih.data<T>() : nullptr;
        const T* b_hh_data = bias_hh.numel() > 0 ? bias_hh.data<T>() : nullptr;

        T* d_input = grad_input.data<T>();
        T* d_hx = grad_hx.data<T>();
        T* d_w_ih = grad_weight_ih.data<T>();
        T* d_w_hh = grad_weight_hh.data<T>();
        T* d_b_ih = grad_bias_ih.data<T>();
        T* d_b_hh = grad_bias_hh.data<T>();

        auto sigmoid_t = [](T x) -> T { return T(1) / (T(1) + std::exp(-x)); };

        #pragma omp parallel if(batch_size > 4)
        {
            std::vector<T> t_d_w_ih(3 * hidden_size * input_size, T(0));
            std::vector<T> t_d_w_hh(3 * hidden_size * hidden_size, T(0));
            std::vector<T> t_d_b_ih(3 * hidden_size, T(0));
            std::vector<T> t_d_b_hh(3 * hidden_size, T(0));

            #pragma omp for
            for (int64_t b = 0; b < batch_size; ++b) {
                std::vector<T> rz_pre(2 * hidden_size);
                for (int64_t g = 0; g < 2 * hidden_size; ++g) {
                    T sum = (b_ih_data ? b_ih_data[g] : T(0)) + (b_hh_data ? b_hh_data[g] : T(0));
                    for (int64_t i = 0; i < input_size; ++i) {
                        sum += in_data[b * input_size + i] * w_ih_data[g * input_size + i];
                    }
                    for (int64_t h = 0; h < hidden_size; ++h) {
                        sum += hx_data[b * hidden_size + h] * w_hh_data[g * hidden_size + h];
                    }
                    rz_pre[g] = sum;
                }

                std::vector<T> r_gate(hidden_size), z_gate(hidden_size);
                for (int64_t h = 0; h < hidden_size; ++h) {
                    r_gate[h] = sigmoid_t(rz_pre[h]);
                    z_gate[h] = sigmoid_t(rz_pre[hidden_size + h]);
                }

                std::vector<T> n_ih(hidden_size);
                std::vector<T> n_hh(hidden_size);
                // Pre-reset hidden-hidden term (b_hn + W_hn @ hx), before the
                // *= r_gate[h] scaling. Saved here so the backward block below can
                // reuse it for the dr_from_n term instead of recomputing the
                // O(hidden) inner product per output unit (O(hidden^2) per batch).
                std::vector<T> n_hh_pre(hidden_size);
                std::vector<T> n_gate(hidden_size);

                for (int64_t h = 0; h < hidden_size; ++h) {
                    n_ih[h] = b_ih_data ? b_ih_data[2 * hidden_size + h] : T(0);
                    n_hh[h] = b_hh_data ? b_hh_data[2 * hidden_size + h] : T(0);
                    for (int64_t i = 0; i < input_size; ++i) {
                        n_ih[h] += in_data[b * input_size + i] * w_ih_data[(2 * hidden_size + h) * input_size + i];
                    }
                    for (int64_t hh = 0; hh < hidden_size; ++hh) {
                        n_hh[h] += hx_data[b * hidden_size + hh] *
                                   w_hh_data[(2 * hidden_size + h) * hidden_size + hh];
                    }
                    n_hh_pre[h] = n_hh[h];
                    n_hh[h] *= r_gate[h];
                    n_gate[h] = std::tanh(n_ih[h] + n_hh[h]);
                }

                std::vector<T> d_gates_ih(3 * hidden_size, T(0));
                std::vector<T> d_gates_hh(3 * hidden_size, T(0));

                for (int64_t h = 0; h < hidden_size; ++h) {
                    T dh = dhy_data[b * hidden_size + h];

                    T dn = dh * (T(1) - z_gate[h]);
                    T dz = dh * (hx_data[b * hidden_size + h] - n_gate[h]);
                    d_hx[b * hidden_size + h] = dh * z_gate[h];

                    T dn_pre = dn * (T(1) - n_gate[h] * n_gate[h]);

                    T dr_from_n = dn_pre * n_hh_pre[h];
                    for (int64_t hh = 0; hh < hidden_size; ++hh) {
                        d_hx[b * hidden_size + hh] += dn_pre * r_gate[h] *
                                                       w_hh_data[(2 * hidden_size + h) * hidden_size + hh];
                    }

                    T dz_pre = dz * z_gate[h] * (T(1) - z_gate[h]);
                    T dr = dr_from_n;
                    T dr_pre = dr * r_gate[h] * (T(1) - r_gate[h]);

                    d_gates_ih[h] = dr_pre;
                    d_gates_ih[hidden_size + h] = dz_pre;
                    d_gates_ih[2 * hidden_size + h] = dn_pre;

                    d_gates_hh[h] = dr_pre;
                    d_gates_hh[hidden_size + h] = dz_pre;
                    d_gates_hh[2 * hidden_size + h] = dn_pre * r_gate[h];
                }

                for (int64_t i = 0; i < input_size; ++i) {
                    T sum = T(0);
                    for (int64_t g = 0; g < 3 * hidden_size; ++g) {
                        sum += d_gates_ih[g] * w_ih_data[g * input_size + i];
                    }
                    d_input[b * input_size + i] = sum;
                }

                for (int64_t h = 0; h < hidden_size; ++h) {
                    T sum = T(0);
                    for (int64_t g = 0; g < 2 * hidden_size; ++g) {
                        sum += d_gates_hh[g] * w_hh_data[g * hidden_size + h];
                    }
                    d_hx[b * hidden_size + h] += sum;
                }

                for (int64_t g = 0; g < 3 * hidden_size; ++g) {
                    for (int64_t i = 0; i < input_size; ++i) {
                        t_d_w_ih[g * input_size + i] += d_gates_ih[g] * in_data[b * input_size + i];
                    }
                    t_d_b_ih[g] += d_gates_ih[g];
                }

                for (int64_t g = 0; g < 2 * hidden_size; ++g) {
                    for (int64_t h = 0; h < hidden_size; ++h) {
                        t_d_w_hh[g * hidden_size + h] += d_gates_hh[g] * hx_data[b * hidden_size + h];
                    }
                    t_d_b_hh[g] += d_gates_hh[g];
                }
                for (int64_t h_out = 0; h_out < hidden_size; ++h_out) {
                    // d_gates_hh[2*hidden + h_out] already carries one factor of
                    // r (set at line 721: dn_pre * r_gate[h_out]). The forward
                    //   n = tanh(n_ih + r * (W_hn @ hx + b_hn))
                    // means grad W_hn carries exactly r^1, so do NOT multiply by
                    // r_gate[h_out] again here (that yielded r^2).
                    T dn_pre = d_gates_hh[2 * hidden_size + h_out];
                    for (int64_t h = 0; h < hidden_size; ++h) {
                        t_d_w_hh[(2 * hidden_size + h_out) * hidden_size + h] +=
                            dn_pre * hx_data[b * hidden_size + h];
                    }
                    t_d_b_hh[2 * hidden_size + h_out] += dn_pre;
                }
            }

            #pragma omp critical
            {
                for (int64_t i = 0; i < 3 * hidden_size * input_size; ++i) d_w_ih[i] += t_d_w_ih[i];
                for (int64_t i = 0; i < 3 * hidden_size * hidden_size; ++i) d_w_hh[i] += t_d_w_hh[i];
                for (int64_t i = 0; i < 3 * hidden_size; ++i) {
                    d_b_ih[i] += t_d_b_ih[i];
                    d_b_hh[i] += t_d_b_hh[i];
                }
            }
        }

        return {grad_input, grad_hx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh};
    };

    if (grad_hy.dtype() == DType::Float32) {
        return run.template operator()<float>();
    } else if (grad_hy.dtype() == DType::Float64) {
        return run.template operator()<double>();
    } else if (grad_hy.dtype() == DType::Float16 || grad_hy.dtype() == DType::BFloat16) {
        DType orig = grad_hy.dtype();
        auto results = gru_cell_backward_kernel(
            grad_hy.to(DType::Float32), input.to(DType::Float32), hx.to(DType::Float32),
            weight_ih.to(DType::Float32), weight_hh.to(DType::Float32),
            bias_ih.to(DType::Float32), bias_hh.to(DType::Float32));
        for (auto& t : results) t = t.to(orig);
        return results;
    } else {
        throw std::runtime_error("gru_cell_backward_kernel: unsupported dtype");
    }
}

// =============================================================================
// Full-Sequence Kernels (SIMD-Optimized via fused_lstm.hpp)
// =============================================================================

/**
 * @brief Fused LSTM forward pass for entire sequence
 *
 * Uses SIMD-accelerated gate computations and batched input transformation.
 *
 * @param input Input sequence (seq_len, batch, input_size)
 * @param W_ih Input-to-hidden weights (4*hidden, input_size)
 * @param W_hh Hidden-to-hidden weights (4*hidden, hidden)
 * @param bias Combined bias (4*hidden) or empty tensor
 * @param h0 Initial hidden state (batch, hidden)
 * @param c0 Initial cell state (batch, hidden)
 * @return vector of [output, h_n, c_n]
 */
auto lstm_forward_kernel(
    const Tensor& input,
    const Tensor& W_ih,
    const Tensor& W_hh,
    const Tensor& bias_ih,
    const Tensor& bias_hh,
    const Tensor& h0,
    const Tensor& c0
) -> std::vector<Tensor> {
    // S12: native Float32 (SIMD+oneDNN), native Float64 (scalar/OpenMP),
    // widen-narrow for Float16/BFloat16.
    DType dt = input.dtype();

    if (dt == DType::Float16 || dt == DType::BFloat16) {
        DType orig = dt;
        auto results = lstm_forward_kernel(
            input.to(DType::Float32), W_ih.to(DType::Float32), W_hh.to(DType::Float32),
            bias_ih.numel() > 0 ? bias_ih.to(DType::Float32) : bias_ih,
            bias_hh.numel() > 0 ? bias_hh.to(DType::Float32) : bias_hh,
            h0.to(DType::Float32), c0.to(DType::Float32));
        for (auto& t : results) t = t.to(orig);
        return results;
    }
    if (dt != DType::Float32 && dt != DType::Float64) {
        throw std::runtime_error("lstm_forward_kernel: unsupported dtype");
    }

    // Get dimensions
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden = h0.shape()[1];

    // Make tensors contiguous - these are likely already contiguous, so no-op
    Tensor input_contig = input.contiguous();
    Tensor W_ih_contig = W_ih.contiguous();
    Tensor W_hh_contig = W_hh.contiguous();
    Tensor h0_contig = h0.contiguous();
    Tensor c0_contig = c0.contiguous();

    bool has_bias_ih = bias_ih.numel() > 0;
    bool has_bias_hh = bias_hh.numel() > 0;

    // Allocate output tensors
    Tensor output = empty({seq_len, batch, hidden}, dt, input.device());
    Tensor h_n = empty({batch, hidden}, dt, input.device());
    Tensor c_n = empty({batch, hidden}, dt, input.device());

    if (dt == DType::Float32) {
        // Combine bias_ih + bias_hh for oneDNN (Float32 combined-bias contract)
        std::vector<float> combined_bias_buffer;
        const float* bias_ptr = nullptr;
        if (has_bias_ih || has_bias_hh) {
            int64_t bias_size = 4 * hidden;
            combined_bias_buffer.resize(bias_size);
            const float* bih_ptr = has_bias_ih ? bias_ih.data<float>() : nullptr;
            const float* bhh_ptr = has_bias_hh ? bias_hh.data<float>() : nullptr;
            if (has_bias_ih && has_bias_hh) {
                #pragma omp simd
                for (int64_t i = 0; i < bias_size; ++i) {
                    combined_bias_buffer[i] = bih_ptr[i] + bhh_ptr[i];
                }
            } else if (has_bias_ih) {
                std::memcpy(combined_bias_buffer.data(), bih_ptr, bias_size * sizeof(float));
            } else {
                std::memcpy(combined_bias_buffer.data(), bhh_ptr, bias_size * sizeof(float));
            }
            bias_ptr = combined_bias_buffer.data();
        }

#ifdef TENZOR_USE_ONEDNN
        bool onednn_success = rnn_onednn::lstm_forward_onednn(
            input_contig.data<float>(),
            W_ih_contig.data<float>(),
            W_hh_contig.data<float>(),
            bias_ptr,
            h0_contig.data<float>(),
            c0_contig.data<float>(),
            output.data<float>(),
            h_n.data<float>(),
            c_n.data<float>(),
            seq_len, batch, input_size, hidden
        );
        if (onednn_success) {
            return {output, h_n, c_n};
        }
#endif

        // SIMD-optimized fallback (Float32 only)
        lstm::lstm_forward(
            input_contig.data<float>(),
            W_ih_contig.data<float>(),
            W_hh_contig.data<float>(),
            bias_ptr,
            h0_contig.data<float>(),
            c0_contig.data<float>(),
            output.data<float>(),
            h_n.data<float>(),
            c_n.data<float>(),
            seq_len, batch, input_size, hidden
        );

        return {output, h_n, c_n};
    }

    // Float64 native path. Combine biases into a single double buffer to
    // match the (b_ih + b_hh) semantics the float path uses.
    std::vector<double> combined_bias_d;
    const double* bias_ptr_d = nullptr;
    if (has_bias_ih || has_bias_hh) {
        int64_t bias_size = 4 * hidden;
        combined_bias_d.resize(bias_size);
        const double* bih = has_bias_ih ? bias_ih.data<double>() : nullptr;
        const double* bhh = has_bias_hh ? bias_hh.data<double>() : nullptr;
        if (has_bias_ih && has_bias_hh) {
            for (int64_t i = 0; i < bias_size; ++i) combined_bias_d[i] = bih[i] + bhh[i];
        } else if (has_bias_ih) {
            std::memcpy(combined_bias_d.data(), bih, bias_size * sizeof(double));
        } else {
            std::memcpy(combined_bias_d.data(), bhh, bias_size * sizeof(double));
        }
        bias_ptr_d = combined_bias_d.data();
    }

    lstm_forward_f64(
        input_contig.data<double>(),
        W_ih_contig.data<double>(),
        W_hh_contig.data<double>(),
        bias_ptr_d,
        h0_contig.data<double>(),
        c0_contig.data<double>(),
        output.data<double>(),
        h_n.data<double>(),
        c_n.data<double>(),
        seq_len, batch, input_size, hidden
    );

    return {output, h_n, c_n};
}

/**
 * @brief Bidirectional LSTM forward pass
 *
 * Uses oneDNN-accelerated forward and backward LSTM passes.
 *
 * @param input Input sequence (seq_len, batch, input_size)
 * @param W_ih_fwd Forward input-hidden weights (4*hidden, input_size)
 * @param W_hh_fwd Forward hidden-hidden weights (4*hidden, hidden)
 * @param bias_ih_fwd Forward input bias (4*hidden) or empty
 * @param bias_hh_fwd Forward hidden bias (4*hidden) or empty
 * @param W_ih_bwd Backward input-hidden weights (4*hidden, input_size)
 * @param W_hh_bwd Backward hidden-hidden weights (4*hidden, hidden)
 * @param bias_ih_bwd Backward input bias (4*hidden) or empty
 * @param bias_hh_bwd Backward hidden bias (4*hidden) or empty
 * @param h0 Initial hidden states (2, batch, hidden) - [forward, backward]
 * @param c0 Initial cell states (2, batch, hidden) - [forward, backward]
 * @return vector of [output (seq, batch, 2*hidden), h_n (2, batch, hidden), c_n (2, batch, hidden)]
 */
auto bilstm_forward_kernel(
    const Tensor& input,
    const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
    const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
    const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
    const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd,
    const Tensor& h0,
    const Tensor& c0
) -> std::vector<Tensor> {
    // S12: native Float32 (SIMD+oneDNN), native Float64 (scalar/OpenMP).
    DType dt = input.dtype();

    if (dt == DType::Float16 || dt == DType::BFloat16) {
        DType orig = dt;
        auto to_f32 = [](const Tensor& t) { return t.numel() > 0 ? t.to(DType::Float32) : t; };
        auto results = bilstm_forward_kernel(
            input.to(DType::Float32),
            W_ih_fwd.to(DType::Float32), W_hh_fwd.to(DType::Float32),
            to_f32(bias_ih_fwd), to_f32(bias_hh_fwd),
            W_ih_bwd.to(DType::Float32), W_hh_bwd.to(DType::Float32),
            to_f32(bias_ih_bwd), to_f32(bias_hh_bwd),
            h0.to(DType::Float32), c0.to(DType::Float32));
        for (auto& t : results) t = t.to(orig);
        return results;
    }
    if (dt != DType::Float32 && dt != DType::Float64) {
        throw std::runtime_error("bilstm_forward_kernel: unsupported dtype");
    }

    // Get dimensions
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden = h0.shape()[2];  // h0 is (2, batch, hidden)

    // Make tensors contiguous
    Tensor input_contig = input.contiguous();
    Tensor W_ih_fwd_contig = W_ih_fwd.contiguous();
    Tensor W_hh_fwd_contig = W_hh_fwd.contiguous();
    Tensor W_ih_bwd_contig = W_ih_bwd.contiguous();
    Tensor W_hh_bwd_contig = W_hh_bwd.contiguous();
    Tensor h0_contig = h0.contiguous();
    Tensor c0_contig = c0.contiguous();

    bool has_bias_fwd = bias_ih_fwd.numel() > 0 || bias_hh_fwd.numel() > 0;
    bool has_bias_bwd = bias_ih_bwd.numel() > 0 || bias_hh_bwd.numel() > 0;

    // Allocate output tensors
    Tensor output = empty({seq_len, batch, 2 * hidden}, dt, input.device());
    Tensor h_n = empty({2, batch, hidden}, dt, input.device());
    Tensor c_n = empty({2, batch, hidden}, dt, input.device());

    if (dt == DType::Float32) {
        // Combine biases for forward direction
        std::vector<float> bias_fwd_buffer;
        const float* bias_fwd_ptr = nullptr;
        if (has_bias_fwd) {
            int64_t bias_size = 4 * hidden;
            bias_fwd_buffer.resize(bias_size);
            if (bias_ih_fwd.numel() > 0 && bias_hh_fwd.numel() > 0) {
                const float* bih = bias_ih_fwd.contiguous().data<float>();
                const float* bhh = bias_hh_fwd.contiguous().data<float>();
                for (int64_t i = 0; i < bias_size; ++i) bias_fwd_buffer[i] = bih[i] + bhh[i];
            } else if (bias_ih_fwd.numel() > 0) {
                std::memcpy(bias_fwd_buffer.data(), bias_ih_fwd.contiguous().data<float>(), bias_size * sizeof(float));
            } else {
                std::memcpy(bias_fwd_buffer.data(), bias_hh_fwd.contiguous().data<float>(), bias_size * sizeof(float));
            }
            bias_fwd_ptr = bias_fwd_buffer.data();
        }

        // Combine biases for backward direction
        std::vector<float> bias_bwd_buffer;
        const float* bias_bwd_ptr = nullptr;
        if (has_bias_bwd) {
            int64_t bias_size = 4 * hidden;
            bias_bwd_buffer.resize(bias_size);
            if (bias_ih_bwd.numel() > 0 && bias_hh_bwd.numel() > 0) {
                const float* bih = bias_ih_bwd.contiguous().data<float>();
                const float* bhh = bias_hh_bwd.contiguous().data<float>();
                for (int64_t i = 0; i < bias_size; ++i) bias_bwd_buffer[i] = bih[i] + bhh[i];
            } else if (bias_ih_bwd.numel() > 0) {
                std::memcpy(bias_bwd_buffer.data(), bias_ih_bwd.contiguous().data<float>(), bias_size * sizeof(float));
            } else {
                std::memcpy(bias_bwd_buffer.data(), bias_hh_bwd.contiguous().data<float>(), bias_size * sizeof(float));
            }
            bias_bwd_ptr = bias_bwd_buffer.data();
        }

        const float* h0_data = h0_contig.data<float>();
        const float* c0_data = c0_contig.data<float>();
        const float* h0_fwd = h0_data;
        const float* c0_fwd = c0_data;
        const float* h0_bwd = h0_data + batch * hidden;
        const float* c0_bwd = c0_data + batch * hidden;

        float* h_n_fwd = h_n.data<float>();
        float* c_n_fwd = c_n.data<float>();
        float* h_n_bwd = h_n.data<float>() + batch * hidden;
        float* c_n_bwd = c_n.data<float>() + batch * hidden;

#ifdef TENZOR_USE_ONEDNN
        bool onednn_success = rnn_onednn::bilstm_forward_onednn(
            input_contig.data<float>(),
            W_ih_fwd_contig.data<float>(), W_hh_fwd_contig.data<float>(), bias_fwd_ptr,
            W_ih_bwd_contig.data<float>(), W_hh_bwd_contig.data<float>(), bias_bwd_ptr,
            h0_fwd, c0_fwd, h0_bwd, c0_bwd,
            output.data<float>(),
            h_n_fwd, c_n_fwd, h_n_bwd, c_n_bwd,
            seq_len, batch, input_size, hidden
        );
        if (onednn_success) {
            return {output, h_n, c_n};
        }
#endif

        lstm::lstm_forward_bidirectional(
            input_contig.data<float>(),
            W_ih_fwd_contig.data<float>(), W_hh_fwd_contig.data<float>(), bias_fwd_ptr,
            W_ih_bwd_contig.data<float>(), W_hh_bwd_contig.data<float>(), bias_bwd_ptr,
            h0_fwd, c0_fwd, h0_bwd, c0_bwd,
            output.data<float>(),
            h_n_fwd, c_n_fwd, h_n_bwd, c_n_bwd,
            seq_len, batch, input_size, hidden
        );

        return {output, h_n, c_n};
    }

    // Float64 native path.
    std::vector<double> bias_fwd_d, bias_bwd_d;
    const double* bias_fwd_ptr_d = nullptr;
    const double* bias_bwd_ptr_d = nullptr;
    auto combine_d = [&](const Tensor& bih, const Tensor& bhh, std::vector<double>& buf,
                         const double*& ptr) {
        bool has = bih.numel() > 0 || bhh.numel() > 0;
        if (!has) return;
        int64_t bias_size = 4 * hidden;
        buf.resize(bias_size);
        if (bih.numel() > 0 && bhh.numel() > 0) {
            const double* a = bih.contiguous().data<double>();
            const double* b = bhh.contiguous().data<double>();
            for (int64_t i = 0; i < bias_size; ++i) buf[i] = a[i] + b[i];
        } else if (bih.numel() > 0) {
            std::memcpy(buf.data(), bih.contiguous().data<double>(), bias_size * sizeof(double));
        } else {
            std::memcpy(buf.data(), bhh.contiguous().data<double>(), bias_size * sizeof(double));
        }
        ptr = buf.data();
    };
    combine_d(bias_ih_fwd, bias_hh_fwd, bias_fwd_d, bias_fwd_ptr_d);
    combine_d(bias_ih_bwd, bias_hh_bwd, bias_bwd_d, bias_bwd_ptr_d);

    const double* h0_data = h0_contig.data<double>();
    const double* c0_data = c0_contig.data<double>();
    const double* h0_fwd = h0_data;
    const double* c0_fwd = c0_data;
    const double* h0_bwd = h0_data + batch * hidden;
    const double* c0_bwd = c0_data + batch * hidden;

    double* h_n_fwd = h_n.data<double>();
    double* c_n_fwd = c_n.data<double>();
    double* h_n_bwd = h_n.data<double>() + batch * hidden;
    double* c_n_bwd = c_n.data<double>() + batch * hidden;

    lstm_forward_bidirectional_f64(
        input_contig.data<double>(),
        W_ih_fwd_contig.data<double>(), W_hh_fwd_contig.data<double>(), bias_fwd_ptr_d,
        W_ih_bwd_contig.data<double>(), W_hh_bwd_contig.data<double>(), bias_bwd_ptr_d,
        h0_fwd, c0_fwd, h0_bwd, c0_bwd,
        output.data<double>(),
        h_n_fwd, c_n_fwd, h_n_bwd, c_n_bwd,
        seq_len, batch, input_size, hidden
    );

    return {output, h_n, c_n};
}

/**
 * @brief Fused GRU forward pass for entire sequence
 *
 * Uses SIMD-accelerated gate computations and batched input transformation.
 *
 * @param input Input sequence (seq_len, batch, input_size)
 * @param W_ih Input-to-hidden weights (3*hidden, input_size)
 * @param W_hh Hidden-to-hidden weights (3*hidden, hidden)
 * @param bias Combined bias (3*hidden) or empty tensor
 * @param h0 Initial hidden state (batch, hidden)
 * @return vector of [output, h_n]
 */
auto gru_forward_kernel(
    const Tensor& input,
    const Tensor& W_ih,
    const Tensor& W_hh,
    const Tensor& bias_ih,   // F.2: PyTorch convention — two biases.
    const Tensor& bias_hh,   //       Empty tensor means "no bias".
    const Tensor& h0
) -> std::vector<Tensor> {
    // S12: native Float32 (SIMD+oneDNN), native Float64 (scalar/OpenMP).
    DType dt = input.dtype();

    if (dt == DType::Float16 || dt == DType::BFloat16) {
        DType orig = dt;
        auto results = gru_forward_kernel(
            input.to(DType::Float32), W_ih.to(DType::Float32), W_hh.to(DType::Float32),
            bias_ih.numel() > 0 ? bias_ih.to(DType::Float32) : bias_ih,
            bias_hh.numel() > 0 ? bias_hh.to(DType::Float32) : bias_hh,
            h0.to(DType::Float32));
        for (auto& t : results) t = t.to(orig);
        return results;
    }
    if (dt != DType::Float32 && dt != DType::Float64) {
        throw std::runtime_error("gru_forward_kernel: unsupported dtype");
    }

    // Get dimensions
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden = h0.shape()[1];

    // Make tensors contiguous
    Tensor input_contig = input.contiguous();
    Tensor W_ih_contig = W_ih.contiguous();
    Tensor W_hh_contig = W_hh.contiguous();
    Tensor h0_contig = h0.contiguous();

    // Bias contig handles (kept alive for both dtypes)
    Tensor bias_ih_contig, bias_hh_contig;
    if (bias_ih.numel() > 0) bias_ih_contig = bias_ih.contiguous();
    if (bias_hh.numel() > 0) bias_hh_contig = bias_hh.contiguous();

    // Allocate output tensors
    Tensor output = empty({seq_len, batch, hidden}, dt, input.device());
    Tensor h_n = empty({batch, hidden}, dt, input.device());

    if (dt == DType::Float32) {
        const float* bias_ih_ptr = bias_ih.numel() > 0 ? bias_ih_contig.data<float>() : nullptr;
        const float* bias_hh_ptr = bias_hh.numel() > 0 ? bias_hh_contig.data<float>() : nullptr;

#ifdef TENZOR_USE_ONEDNN
        // The oneDNN GRU path uses the LBR (linear_before_reset=1) primitive and
        // maps bias_ih / bias_hh into oneDNN's 4-term LBR bias, so it is exact
        // for biased GRUs too — no need to gate on the no-bias case anymore.
        {
            bool onednn_success = rnn_onednn::gru_forward_onednn(
                input_contig.data<float>(),
                W_ih_contig.data<float>(),
                W_hh_contig.data<float>(),
                bias_ih_ptr,
                bias_hh_ptr,
                h0_contig.data<float>(),
                output.data<float>(),
                h_n.data<float>(),
                seq_len, batch, input_size, hidden
            );
            if (onednn_success) {
                return {output, h_n};
            }
        }
#endif

        lstm::gru_forward(
            input_contig.data<float>(),
            W_ih_contig.data<float>(),
            W_hh_contig.data<float>(),
            bias_ih_ptr,
            bias_hh_ptr,
            h0_contig.data<float>(),
            output.data<float>(),
            h_n.data<float>(),
            seq_len, batch, input_size, hidden
        );

        return {output, h_n};
    }

    // Float64 native path.
    const double* bias_ih_ptr_d = bias_ih.numel() > 0 ? bias_ih_contig.data<double>() : nullptr;
    const double* bias_hh_ptr_d = bias_hh.numel() > 0 ? bias_hh_contig.data<double>() : nullptr;

    gru_forward_f64(
        input_contig.data<double>(),
        W_ih_contig.data<double>(),
        W_hh_contig.data<double>(),
        bias_ih_ptr_d,
        bias_hh_ptr_d,
        h0_contig.data<double>(),
        output.data<double>(),
        h_n.data<double>(),
        seq_len, batch, input_size, hidden
    );

    return {output, h_n};
}

/**
 * @brief Fused multi-layer LSTM forward pass
 *
 * Uses oneDNN's native multi-layer support for optimal performance.
 * When input_size != hidden_size, processes layer 0 separately and fuses layers 1+.
 *
 * @param input Input sequence (seq_len, batch, input_size)
 * @param W_ih_list Input-to-hidden weights for each layer
 * @param W_hh_list Hidden-to-hidden weights for each layer
 * @param bias_list Bias for each layer (empty if no bias)
 * @param h0 Initial hidden states (num_layers, batch, hidden)
 * @param c0 Initial cell states (num_layers, batch, hidden)
 * @return vector of [output, h_n, c_n]
 */
auto lstm_multilayer_forward_kernel(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0,
    const Tensor& c0
) -> std::vector<Tensor> {
    // S12: native Float32 (oneDNN fused or fallback per-layer) and native
    // Float64 (per-layer single-layer kernel — which is now itself native
    // Float64). Float16/BFloat16 still widen-narrow.
    DType dt = input.dtype();

    if (dt == DType::Float16 || dt == DType::BFloat16) {
        DType orig = dt;
        std::vector<Tensor> W_ih_f32, W_hh_f32, bias_f32;
        for (const auto& t : W_ih_list) W_ih_f32.push_back(t.to(DType::Float32));
        for (const auto& t : W_hh_list) W_hh_f32.push_back(t.to(DType::Float32));
        for (const auto& t : bias_list) bias_f32.push_back(t.numel() > 0 ? t.to(DType::Float32) : t);
        auto results = lstm_multilayer_forward_kernel(
            input.to(DType::Float32), W_ih_f32, W_hh_f32, bias_f32,
            h0.to(DType::Float32), c0.to(DType::Float32));
        for (auto& t : results) t = t.to(orig);
        return results;
    }
    if (dt != DType::Float32 && dt != DType::Float64) {
        throw std::runtime_error("lstm_multilayer_forward_kernel: unsupported dtype");
    }

    // Get dimensions
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    int64_t hidden = h0.shape()[2];  // h0 is (num_layers, batch, hidden)

    Tensor input_contig = input.contiguous();
    Tensor h0_contig = h0.contiguous();
    Tensor c0_contig = c0.contiguous();

    std::vector<Tensor> W_ih_contig, W_hh_contig, bias_contig;

    for (int64_t l = 0; l < num_layers; ++l) {
        W_ih_contig.push_back(W_ih_list[l].contiguous());
        W_hh_contig.push_back(W_hh_list[l].contiguous());
        if (!bias_list.empty() && bias_list[l].numel() > 0) {
            bias_contig.push_back(bias_list[l].contiguous());
        }
    }

    Tensor output = empty({seq_len, batch, hidden}, dt, input.device());
    Tensor h_n = empty({num_layers, batch, hidden}, dt, input.device());
    Tensor c_n = empty({num_layers, batch, hidden}, dt, input.device());

    if (dt == DType::Float32) {
        std::vector<const float*> W_ih_ptrs, W_hh_ptrs;
        for (int64_t l = 0; l < num_layers; ++l) {
            W_ih_ptrs.push_back(W_ih_contig[l].data<float>());
            W_hh_ptrs.push_back(W_hh_contig[l].data<float>());
        }
        // Build a length-num_layers bias pointer vector that is index-aligned to
        // the layer number: nullptr entries for bias-less layers (do NOT compact,
        // otherwise lstm_multilayer_forward_onednn's bias_list[l] selects the
        // wrong layer's bias for mixed-bias configs). bias_contig holds only the
        // non-empty bias tensors in layer order, so walk it with a running index.
        std::vector<const float*> bias_ptrs_f;
        if (!bias_list.empty()) {
            bias_ptrs_f.assign(num_layers, nullptr);
            size_t bi = 0;
            for (int64_t l = 0; l < num_layers; ++l) {
                if (bias_list[l].numel() > 0) {
                    bias_ptrs_f[l] = bias_contig[bi].data<float>();
                    ++bi;
                }
            }
        }

#ifdef TENZOR_USE_ONEDNN
        bool onednn_success = rnn_onednn::lstm_multilayer_forward_onednn(
            input_contig.data<float>(),
            W_ih_ptrs,
            W_hh_ptrs,
            bias_ptrs_f,
            h0_contig.data<float>(),
            c0_contig.data<float>(),
            output.data<float>(),
            h_n.data<float>(),
            c_n.data<float>(),
            num_layers, seq_len, batch, input_size, hidden
        );
        if (onednn_success) {
            return {output, h_n, c_n};
        }
#endif
    }

    // Fallback for Float32 / native path for Float64: process each layer
    // sequentially using the single-layer kernel (which is dtype-aware).
    Tensor layer_input = input_contig;
    std::vector<Tensor> h_states, c_states;

    for (int64_t l = 0; l < num_layers; ++l) {
        Tensor h0_layer = h0_contig.slice(0, l, l + 1).reshape({batch, hidden}).contiguous();
        Tensor c0_layer = c0_contig.slice(0, l, l + 1).reshape({batch, hidden}).contiguous();

        Tensor bias_ih_tensor = (!bias_list.empty() && bias_list[l].numel() > 0)
                                ? bias_list[l]
                                : empty({0}, dt, input.device());
        Tensor bias_hh_empty = empty({0}, dt, input.device());

        auto layer_output = lstm_forward_kernel(
            layer_input, W_ih_list[l], W_hh_list[l], bias_ih_tensor, bias_hh_empty, h0_layer, c0_layer
        );

        h_states.push_back(layer_output[1]);
        c_states.push_back(layer_output[2]);

        layer_input = layer_output[0];
    }

    int64_t elem_size = (dt == DType::Float32) ? sizeof(float) : sizeof(double);

    std::memcpy(static_cast<char*>(output.data_ptr()),
                static_cast<const char*>(layer_input.data_ptr()),
                seq_len * batch * hidden * elem_size);

    for (int64_t l = 0; l < num_layers; ++l) {
        std::memcpy(static_cast<char*>(h_n.data_ptr()) + l * batch * hidden * elem_size,
                    static_cast<const char*>(h_states[l].data_ptr()),
                    batch * hidden * elem_size);
        std::memcpy(static_cast<char*>(c_n.data_ptr()) + l * batch * hidden * elem_size,
                    static_cast<const char*>(c_states[l].data_ptr()),
                    batch * hidden * elem_size);
    }

    return {output, h_n, c_n};
}

/**
 * @brief Fused multi-layer GRU forward pass
 *
 * Uses oneDNN's native multi-layer support for optimal performance.
 * When input_size != hidden_size, processes layer 0 separately and fuses layers 1+.
 *
 * @param input Input sequence (seq_len, batch, input_size)
 * @param W_ih_list Input-to-hidden weights for each layer
 * @param W_hh_list Hidden-to-hidden weights for each layer
 * @param bias_list Bias for each layer (empty if no bias)
 * @param h0 Initial hidden states (num_layers, batch, hidden)
 * @return vector of [output, h_n]
 */
auto gru_multilayer_forward_kernel(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0
) -> std::vector<Tensor> {
    // S12: native Float32 / Float64. Half precision widen-narrow.
    DType dt = input.dtype();

    if (dt == DType::Float16 || dt == DType::BFloat16) {
        DType orig = dt;
        std::vector<Tensor> W_ih_f32, W_hh_f32, bias_f32;
        for (const auto& t : W_ih_list) W_ih_f32.push_back(t.to(DType::Float32));
        for (const auto& t : W_hh_list) W_hh_f32.push_back(t.to(DType::Float32));
        for (const auto& t : bias_list) bias_f32.push_back(t.numel() > 0 ? t.to(DType::Float32) : t);
        auto results = gru_multilayer_forward_kernel(
            input.to(DType::Float32), W_ih_f32, W_hh_f32, bias_f32,
            h0.to(DType::Float32));
        for (auto& t : results) t = t.to(orig);
        return results;
    }
    if (dt != DType::Float32 && dt != DType::Float64) {
        throw std::runtime_error("gru_multilayer_forward_kernel: unsupported dtype");
    }

    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    int64_t hidden = h0.shape()[2];  // h0 is (num_layers, batch, hidden)

    Tensor input_contig = input.contiguous();
    Tensor h0_contig = h0.contiguous();

    std::vector<Tensor> W_ih_contig, W_hh_contig, bias_contig;

    for (int64_t l = 0; l < num_layers; ++l) {
        W_ih_contig.push_back(W_ih_list[l].contiguous());
        W_hh_contig.push_back(W_hh_list[l].contiguous());
        if (!bias_list.empty() && bias_list[l].numel() > 0) {
            bias_contig.push_back(bias_list[l].contiguous());
        }
    }

    Tensor output = empty({seq_len, batch, hidden}, dt, input.device());
    Tensor h_n = empty({num_layers, batch, hidden}, dt, input.device());

    if (dt == DType::Float32) {
        std::vector<const float*> W_ih_ptrs, W_hh_ptrs, bias_ptrs;
        for (int64_t l = 0; l < num_layers; ++l) {
            W_ih_ptrs.push_back(W_ih_contig[l].data<float>());
            W_hh_ptrs.push_back(W_hh_contig[l].data<float>());
        }
        // Build a length-num_layers, index-aligned bias pointer vector with
        // nullptr entries for bias-less layers (do NOT compact, otherwise
        // gru_multilayer_forward_onednn's bias_list[src_layer] indexing selects
        // the wrong layer's bias and reads OOB for mixed-bias configs).
        if (!bias_list.empty()) {
            bias_ptrs.assign(num_layers, nullptr);
            size_t bi = 0;
            for (int64_t l = 0; l < num_layers; ++l) {
                if (bias_list[l].numel() > 0) {
                    bias_ptrs[l] = bias_contig[bi].data<float>();
                    ++bi;
                }
            }
        }

#ifdef TENZOR_USE_ONEDNN
        bool onednn_success = rnn_onednn::gru_multilayer_forward_onednn(
            input_contig.data<float>(),
            W_ih_ptrs,
            W_hh_ptrs,
            bias_ptrs,
            h0_contig.data<float>(),
            output.data<float>(),
            h_n.data<float>(),
            num_layers, seq_len, batch, input_size, hidden
        );
        if (onednn_success) {
            return {output, h_n};
        }
#endif
    }

    // Fallback for Float32 / native for Float64: per-layer via single-layer kernel.
    Tensor layer_input = input_contig;
    std::vector<Tensor> h_states;

    for (int64_t l = 0; l < num_layers; ++l) {
        Tensor h0_layer = h0_contig.slice(0, l, l + 1).reshape({batch, hidden}).contiguous();

        Tensor bias_ih_tensor = (!bias_list.empty() && bias_list[l].numel() > 0)
                                ? bias_list[l]
                                : empty({0}, dt, input.device());
        Tensor bias_hh_empty = empty({0}, dt, input.device());

        auto layer_output = gru_forward_kernel(
            layer_input, W_ih_list[l], W_hh_list[l],
            bias_ih_tensor, bias_hh_empty, h0_layer
        );

        h_states.push_back(layer_output[1]);
        layer_input = layer_output[0];
    }

    int64_t elem_size = (dt == DType::Float32) ? sizeof(float) : sizeof(double);

    std::memcpy(static_cast<char*>(output.data_ptr()),
                static_cast<const char*>(layer_input.data_ptr()),
                seq_len * batch * hidden * elem_size);

    for (int64_t l = 0; l < num_layers; ++l) {
        std::memcpy(static_cast<char*>(h_n.data_ptr()) + l * batch * hidden * elem_size,
                    static_cast<const char*>(h_states[l].data_ptr()),
                    batch * hidden * elem_size);
    }

    return {output, h_n};
}

} // namespace cpu
} // namespace tenzor
