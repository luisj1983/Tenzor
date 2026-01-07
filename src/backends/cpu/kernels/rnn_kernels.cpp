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
// TEMPORARILY disabled to debug crash: #include "rnn_onednn.hpp"
#include <cmath>

namespace tenzor {
namespace cpu {

// Sigmoid activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

auto lstm_cell_forward_kernel(const Tensor& input, const Tensor& hx, const Tensor& cx,
                               const Tensor& weight_ih, const Tensor& weight_hh,
                               const Tensor& bias_ih, const Tensor& bias_hh)
    -> std::vector<Tensor> {
    // input: [batch, input_size]
    // hx: [batch, hidden_size]
    // cx: [batch, hidden_size]
    // weight_ih: [4 * hidden_size, input_size]
    // weight_hh: [4 * hidden_size, hidden_size]
    // bias_ih, bias_hh: [4 * hidden_size]

    auto in_shape = input.shape();
    auto hx_shape = hx.shape();

    int64_t batch_size = in_shape[0];
    int64_t input_size = in_shape[1];
    int64_t hidden_size = hx_shape[1];

    // Output: hy, cy
    auto hy = Tensor::empty_uninitialized({batch_size, hidden_size}, input.dtype(), input.device());
    auto cy = Tensor::empty_uninitialized({batch_size, hidden_size}, input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* hx_data = hx.data<float>();
    const float* cx_data = cx.data<float>();
    const float* w_ih_data = weight_ih.data<float>();
    const float* w_hh_data = weight_hh.data<float>();
    const float* b_ih_data = bias_ih.data<float>();
    const float* b_hh_data = bias_hh.data<float>();
    float* hy_data = hy.data<float>();
    float* cy_data = cy.data<float>();

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        // Compute gates: i, f, g, o (4 * hidden_size total)
        std::vector<float> gates(4 * hidden_size);

        // gates = input @ weight_ih^T + hx @ weight_hh^T + bias_ih + bias_hh
        for (int64_t g = 0; g < 4 * hidden_size; ++g) {
            float sum = b_ih_data[g] + b_hh_data[g];

            for (int64_t i = 0; i < input_size; ++i) {
                sum += in_data[b * input_size + i] * w_ih_data[g * input_size + i];
            }
            for (int64_t h = 0; h < hidden_size; ++h) {
                sum += hx_data[b * hidden_size + h] * w_hh_data[g * hidden_size + h];
            }
            gates[g] = sum;
        }

        // Apply activations and compute outputs
        for (int64_t h = 0; h < hidden_size; ++h) {
            float i_gate = sigmoid(gates[h]);
            float f_gate = sigmoid(gates[hidden_size + h]);
            float g_gate = std::tanh(gates[2 * hidden_size + h]);
            float o_gate = sigmoid(gates[3 * hidden_size + h]);

            float c_new = f_gate * cx_data[b * hidden_size + h] + i_gate * g_gate;
            float h_new = o_gate * std::tanh(c_new);

            cy_data[b * hidden_size + h] = c_new;
            hy_data[b * hidden_size + h] = h_new;
        }
    }

    return {hy, cy};
}

auto lstm_cell_backward_kernel(const Tensor& grad_hy, const Tensor& grad_cy,
                                const Tensor& input, const Tensor& hx, const Tensor& cx,
                                const Tensor& hy, const Tensor& cy,
                                const Tensor& weight_ih, const Tensor& weight_hh)
    -> std::vector<Tensor> {
    auto shape = grad_hy.shape();
    int64_t batch_size = shape[0];
    int64_t hidden_size = shape[1];
    int64_t input_size = input.shape()[1];

    // Gradients
    auto grad_input = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                            input.dtype(), input.device());
    auto grad_hx = zeros(std::vector<int64_t>(hx.shape().begin(), hx.shape().end()),
                         hx.dtype(), hx.device());
    auto grad_cx = zeros(std::vector<int64_t>(cx.shape().begin(), cx.shape().end()),
                         cx.dtype(), cx.device());
    auto grad_weight_ih = zeros(std::vector<int64_t>(weight_ih.shape().begin(), weight_ih.shape().end()),
                                weight_ih.dtype(), weight_ih.device());
    auto grad_weight_hh = zeros(std::vector<int64_t>(weight_hh.shape().begin(), weight_hh.shape().end()),
                                weight_hh.dtype(), weight_hh.device());
    auto grad_bias_ih = zeros({4 * hidden_size}, input.dtype(), input.device());
    auto grad_bias_hh = zeros({4 * hidden_size}, input.dtype(), input.device());

    // Simplified backward pass (would need gate caching for full implementation)
    // For now, return zero gradients as placeholder
    return {grad_input, grad_hx, grad_cx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh};
}

auto gru_cell_forward_kernel(const Tensor& input, const Tensor& hx,
                              const Tensor& weight_ih, const Tensor& weight_hh,
                              const Tensor& bias_ih, const Tensor& bias_hh) -> Tensor {
    // input: [batch, input_size]
    // hx: [batch, hidden_size]
    // weight_ih: [3 * hidden_size, input_size] (r, z, n gates)
    // weight_hh: [3 * hidden_size, hidden_size]

    auto in_shape = input.shape();
    auto hx_shape = hx.shape();

    int64_t batch_size = in_shape[0];
    int64_t input_size = in_shape[1];
    int64_t hidden_size = hx_shape[1];

    auto hy = Tensor::empty_uninitialized({batch_size, hidden_size}, input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* hx_data = hx.data<float>();
    const float* w_ih_data = weight_ih.data<float>();
    const float* w_hh_data = weight_hh.data<float>();
    const float* b_ih_data = bias_ih.data<float>();
    const float* b_hh_data = bias_hh.data<float>();
    float* hy_data = hy.data<float>();

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        // Compute r, z gates (2 * hidden_size)
        std::vector<float> gates_rz(2 * hidden_size);
        for (int64_t g = 0; g < 2 * hidden_size; ++g) {
            float sum = b_ih_data[g] + b_hh_data[g];
            for (int64_t i = 0; i < input_size; ++i) {
                sum += in_data[b * input_size + i] * w_ih_data[g * input_size + i];
            }
            for (int64_t h = 0; h < hidden_size; ++h) {
                sum += hx_data[b * hidden_size + h] * w_hh_data[g * hidden_size + h];
            }
            gates_rz[g] = sigmoid(sum);
        }

        // Compute n gate with reset gate applied
        for (int64_t h = 0; h < hidden_size; ++h) {
            float r = gates_rz[h];
            float z = gates_rz[hidden_size + h];

            float n_ih = b_ih_data[2 * hidden_size + h];
            float n_hh = b_hh_data[2 * hidden_size + h];

            for (int64_t i = 0; i < input_size; ++i) {
                n_ih += in_data[b * input_size + i] * w_ih_data[(2 * hidden_size + h) * input_size + i];
            }
            for (int64_t hh = 0; hh < hidden_size; ++hh) {
                n_hh += (r * hx_data[b * hidden_size + hh]) *
                        w_hh_data[(2 * hidden_size + h) * hidden_size + hh];
            }

            float n = std::tanh(n_ih + n_hh);
            hy_data[b * hidden_size + h] = (1.0f - z) * n + z * hx_data[b * hidden_size + h];
        }
    }

    return hy;
}

auto gru_cell_backward_kernel(const Tensor& grad_hy, const Tensor& input, const Tensor& hx,
                               const Tensor& weight_ih, const Tensor& weight_hh)
    -> std::vector<Tensor> {
    auto shape = grad_hy.shape();
    int64_t hidden_size = shape[1];
    int64_t input_size = input.shape()[1];

    // Gradients (placeholder implementation)
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

    return {grad_input, grad_hx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh};
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
    const Tensor& bias,
    const Tensor& h0,
    const Tensor& c0
) -> std::vector<Tensor> {
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
    Tensor c0_contig = c0.contiguous();

    // Get bias pointer (may be null)
    const float* bias_ptr = nullptr;
    Tensor bias_contig;
    if (bias.numel() > 0) {
        bias_contig = bias.contiguous();
        bias_ptr = bias_contig.data<float>();
    }

    // Allocate output tensors
    Tensor output = empty({seq_len, batch, hidden}, DType::Float32, input.device());
    Tensor h_n = empty({batch, hidden}, DType::Float32, input.device());
    Tensor c_n = empty({batch, hidden}, DType::Float32, input.device());

    // oneDNN LSTM disabled - using SIMD path only

    // Fallback to SIMD-optimized implementation
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
    const Tensor& bias,
    const Tensor& h0
) -> std::vector<Tensor> {
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

    // Get bias pointer (may be null)
    const float* bias_ptr = nullptr;
    Tensor bias_contig;
    if (bias.numel() > 0) {
        bias_contig = bias.contiguous();
        bias_ptr = bias_contig.data<float>();
    }

    // Allocate output tensors
    Tensor output = empty({seq_len, batch, hidden}, DType::Float32, input.device());
    Tensor h_n = empty({batch, hidden}, DType::Float32, input.device());

    // oneDNN GRU disabled - using SIMD path only
    // SIMD-optimized implementation
    lstm::gru_forward(
        input_contig.data<float>(),
        W_ih_contig.data<float>(),
        W_hh_contig.data<float>(),
        bias_ptr,
        h0_contig.data<float>(),
        output.data<float>(),
        h_n.data<float>(),
        seq_len, batch, input_size, hidden
    );

    return {output, h_n};
}

} // namespace cpu
} // namespace tenzor
