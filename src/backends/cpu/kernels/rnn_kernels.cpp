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
    const Tensor& bias_ih,
    const Tensor& bias_hh,
    const Tensor& h0,
    const Tensor& c0
) -> std::vector<Tensor> {
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

    // Combine bias_ih + bias_hh for oneDNN
    // Use thread-local cache with bias pointer fingerprint to avoid re-combining
    static thread_local std::vector<float> combined_bias_buffer;
    static thread_local const float* cached_bias_ih_ptr = nullptr;
    static thread_local const float* cached_bias_hh_ptr = nullptr;
    static thread_local int64_t cached_bias_hidden = 0;
    const float* bias_ptr = nullptr;

    bool has_bias_ih = bias_ih.numel() > 0;
    bool has_bias_hh = bias_hh.numel() > 0;

    if (has_bias_ih || has_bias_hh) {
        int64_t bias_size = 4 * hidden;
        const float* bih_ptr = has_bias_ih ? bias_ih.data<float>() : nullptr;
        const float* bhh_ptr = has_bias_hh ? bias_hh.data<float>() : nullptr;

        // Check if we can reuse cached combined bias
        bool need_recombine = (cached_bias_ih_ptr != bih_ptr) ||
                              (cached_bias_hh_ptr != bhh_ptr) ||
                              (cached_bias_hidden != hidden);

        if (need_recombine) {
            if (combined_bias_buffer.size() < static_cast<size_t>(bias_size)) {
                combined_bias_buffer.resize(bias_size);
            }

            // Combine biases: combined = bias_ih + bias_hh
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

            cached_bias_ih_ptr = bih_ptr;
            cached_bias_hh_ptr = bhh_ptr;
            cached_bias_hidden = hidden;
        }
        bias_ptr = combined_bias_buffer.data();
    }

    // Allocate output tensors
    Tensor output = empty({seq_len, batch, hidden}, DType::Float32, input.device());
    Tensor h_n = empty({batch, hidden}, DType::Float32, input.device());
    Tensor c_n = empty({batch, hidden}, DType::Float32, input.device());

#ifdef TENZOR_USE_ONEDNN
    // Try oneDNN - provides fused LSTM primitive
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

    // SIMD-optimized fallback
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

    // Combine biases for forward direction
    static thread_local std::vector<float> bias_fwd_buffer;
    const float* bias_fwd_ptr = nullptr;
    bool has_bias_fwd = bias_ih_fwd.numel() > 0 || bias_hh_fwd.numel() > 0;
    if (has_bias_fwd) {
        int64_t bias_size = 4 * hidden;
        if (bias_fwd_buffer.size() < static_cast<size_t>(bias_size)) {
            bias_fwd_buffer.resize(bias_size);
        }
        if (bias_ih_fwd.numel() > 0 && bias_hh_fwd.numel() > 0) {
            const float* bih = bias_ih_fwd.contiguous().data<float>();
            const float* bhh = bias_hh_fwd.contiguous().data<float>();
            for (int64_t i = 0; i < bias_size; ++i) {
                bias_fwd_buffer[i] = bih[i] + bhh[i];
            }
        } else if (bias_ih_fwd.numel() > 0) {
            std::memcpy(bias_fwd_buffer.data(), bias_ih_fwd.contiguous().data<float>(), bias_size * sizeof(float));
        } else {
            std::memcpy(bias_fwd_buffer.data(), bias_hh_fwd.contiguous().data<float>(), bias_size * sizeof(float));
        }
        bias_fwd_ptr = bias_fwd_buffer.data();
    }

    // Combine biases for backward direction
    static thread_local std::vector<float> bias_bwd_buffer;
    const float* bias_bwd_ptr = nullptr;
    bool has_bias_bwd = bias_ih_bwd.numel() > 0 || bias_hh_bwd.numel() > 0;
    if (has_bias_bwd) {
        int64_t bias_size = 4 * hidden;
        if (bias_bwd_buffer.size() < static_cast<size_t>(bias_size)) {
            bias_bwd_buffer.resize(bias_size);
        }
        if (bias_ih_bwd.numel() > 0 && bias_hh_bwd.numel() > 0) {
            const float* bih = bias_ih_bwd.contiguous().data<float>();
            const float* bhh = bias_hh_bwd.contiguous().data<float>();
            for (int64_t i = 0; i < bias_size; ++i) {
                bias_bwd_buffer[i] = bih[i] + bhh[i];
            }
        } else if (bias_ih_bwd.numel() > 0) {
            std::memcpy(bias_bwd_buffer.data(), bias_ih_bwd.contiguous().data<float>(), bias_size * sizeof(float));
        } else {
            std::memcpy(bias_bwd_buffer.data(), bias_hh_bwd.contiguous().data<float>(), bias_size * sizeof(float));
        }
        bias_bwd_ptr = bias_bwd_buffer.data();
    }

    // Extract h0/c0 for each direction
    // h0 is (2, batch, hidden), h0[0] is forward, h0[1] is backward
    const float* h0_data = h0_contig.data<float>();
    const float* c0_data = c0_contig.data<float>();
    const float* h0_fwd = h0_data;
    const float* c0_fwd = c0_data;
    const float* h0_bwd = h0_data + batch * hidden;
    const float* c0_bwd = c0_data + batch * hidden;

    // Allocate output tensors
    Tensor output = empty({seq_len, batch, 2 * hidden}, DType::Float32, input.device());
    Tensor h_n = empty({2, batch, hidden}, DType::Float32, input.device());
    Tensor c_n = empty({2, batch, hidden}, DType::Float32, input.device());

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

    // Fallback: not implemented for BiLSTM without oneDNN
    throw std::runtime_error("BiLSTM requires oneDNN support");
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

#ifdef TENZOR_USE_ONEDNN
    // Try oneDNN first - 2-4x faster than SIMD for larger sequences
    bool onednn_success = rnn_onednn::gru_forward_onednn(
        input_contig.data<float>(),
        W_ih_contig.data<float>(),
        W_hh_contig.data<float>(),
        bias_ptr,
        h0_contig.data<float>(),
        output.data<float>(),
        h_n.data<float>(),
        seq_len, batch, input_size, hidden
    );

    if (onednn_success) {
        return {output, h_n};
    }
#endif

    // SIMD-optimized implementation (fallback)
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
    // Get dimensions
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    int64_t hidden = h0.shape()[2];  // h0 is (num_layers, batch, hidden)

    // Make input contiguous
    Tensor input_contig = input.contiguous();
    Tensor h0_contig = h0.contiguous();
    Tensor c0_contig = c0.contiguous();

    // Make weight tensors contiguous and extract pointers
    std::vector<Tensor> W_ih_contig, W_hh_contig, bias_contig;
    std::vector<const float*> W_ih_ptrs, W_hh_ptrs, bias_ptrs;

    for (int64_t l = 0; l < num_layers; ++l) {
        W_ih_contig.push_back(W_ih_list[l].contiguous());
        W_hh_contig.push_back(W_hh_list[l].contiguous());
        W_ih_ptrs.push_back(W_ih_contig.back().data<float>());
        W_hh_ptrs.push_back(W_hh_contig.back().data<float>());

        if (!bias_list.empty() && bias_list[l].numel() > 0) {
            bias_contig.push_back(bias_list[l].contiguous());
            bias_ptrs.push_back(bias_contig.back().data<float>());
        }
    }

    // Allocate output tensors
    Tensor output = empty({seq_len, batch, hidden}, DType::Float32, input.device());
    Tensor h_n = empty({num_layers, batch, hidden}, DType::Float32, input.device());
    Tensor c_n = empty({num_layers, batch, hidden}, DType::Float32, input.device());

#ifdef TENZOR_USE_ONEDNN
    // Try oneDNN fused multi-layer kernel
    bool onednn_success = rnn_onednn::lstm_multilayer_forward_onednn(
        input_contig.data<float>(),
        W_ih_ptrs,
        W_hh_ptrs,
        bias_ptrs,
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

    // Fallback: process each layer sequentially using single-layer kernel
    Tensor layer_input = input_contig;
    std::vector<Tensor> h_states, c_states;

    for (int64_t l = 0; l < num_layers; ++l) {
        int64_t layer_input_size = (l == 0) ? input_size : hidden;

        // Get initial states for this layer
        Tensor h0_layer = h0_contig.slice(0, l, l + 1).reshape({batch, hidden}).contiguous();
        Tensor c0_layer = c0_contig.slice(0, l, l + 1).reshape({batch, hidden}).contiguous();

        // Get bias - multilayer uses combined bias, pass as bias_ih with empty bias_hh
        Tensor bias_ih_tensor = (!bias_list.empty() && bias_list[l].numel() > 0)
                                ? bias_list[l]
                                : empty({0}, DType::Float32, input.device());
        Tensor bias_hh_empty = empty({0}, DType::Float32, input.device());

        // Call single-layer kernel (bias already combined for multilayer case)
        auto layer_output = lstm_forward_kernel(
            layer_input, W_ih_list[l], W_hh_list[l], bias_ih_tensor, bias_hh_empty, h0_layer, c0_layer
        );

        h_states.push_back(layer_output[1]);
        c_states.push_back(layer_output[2]);

        layer_input = layer_output[0];
    }

    // Copy final output
    std::memcpy(output.data<float>(), layer_input.data<float>(),
               seq_len * batch * hidden * sizeof(float));

    // Stack final states
    for (int64_t l = 0; l < num_layers; ++l) {
        std::memcpy(h_n.data<float>() + l * batch * hidden,
                   h_states[l].data<float>(), batch * hidden * sizeof(float));
        std::memcpy(c_n.data<float>() + l * batch * hidden,
                   c_states[l].data<float>(), batch * hidden * sizeof(float));
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
    // Get dimensions
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    int64_t hidden = h0.shape()[2];  // h0 is (num_layers, batch, hidden)

    // Make input contiguous
    Tensor input_contig = input.contiguous();
    Tensor h0_contig = h0.contiguous();

    // Make weight tensors contiguous and extract pointers
    std::vector<Tensor> W_ih_contig, W_hh_contig, bias_contig;
    std::vector<const float*> W_ih_ptrs, W_hh_ptrs, bias_ptrs;

    for (int64_t l = 0; l < num_layers; ++l) {
        W_ih_contig.push_back(W_ih_list[l].contiguous());
        W_hh_contig.push_back(W_hh_list[l].contiguous());
        W_ih_ptrs.push_back(W_ih_contig.back().data<float>());
        W_hh_ptrs.push_back(W_hh_contig.back().data<float>());

        if (!bias_list.empty() && bias_list[l].numel() > 0) {
            bias_contig.push_back(bias_list[l].contiguous());
            bias_ptrs.push_back(bias_contig.back().data<float>());
        }
    }

    // Allocate output tensors
    Tensor output = empty({seq_len, batch, hidden}, DType::Float32, input.device());
    Tensor h_n = empty({num_layers, batch, hidden}, DType::Float32, input.device());

#ifdef TENZOR_USE_ONEDNN
    // Try oneDNN fused multi-layer kernel
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

    // Fallback: process each layer sequentially using single-layer kernel
    Tensor layer_input = input_contig;
    std::vector<Tensor> h_states;

    for (int64_t l = 0; l < num_layers; ++l) {
        int64_t layer_input_size = (l == 0) ? input_size : hidden;

        // Get initial state for this layer
        Tensor h0_layer = h0_contig.slice(0, l, l + 1).reshape({batch, hidden}).contiguous();

        // Get bias or empty tensor
        Tensor bias_tensor = (!bias_list.empty() && bias_list[l].numel() > 0)
                             ? bias_list[l]
                             : empty({0}, DType::Float32, input.device());

        // Call single-layer kernel
        auto layer_output = gru_forward_kernel(
            layer_input, W_ih_list[l], W_hh_list[l], bias_tensor, h0_layer
        );

        h_states.push_back(layer_output[1]);
        layer_input = layer_output[0];
    }

    // Copy final output
    std::memcpy(output.data<float>(), layer_input.data<float>(),
               seq_len * batch * hidden * sizeof(float));

    // Stack final states
    for (int64_t l = 0; l < num_layers; ++l) {
        std::memcpy(h_n.data<float>() + l * batch * hidden,
                   h_states[l].data<float>(), batch * hidden * sizeof(float));
    }

    return {output, h_n};
}

} // namespace cpu
} // namespace tenzor
