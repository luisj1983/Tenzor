/**
 * @file rnn.cpp
 * @brief OneAPI/SYCL kernels for full-sequence RNN operations
 *
 * Implements multi-step LSTM/GRU forward passes (single layer, multi-layer,
 * bidirectional) using the existing cell-level kernels. Each step iterates
 * over the sequence length and calls the fused cell kernel.
 */

#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace oneapi {

// Forward declarations of cell-level kernels (from lstm.cpp / gru.cpp)
auto lstm_cell_forward_kernel(
    const Tensor& gates, const Tensor& c_prev,
    int64_t batch_size, int64_t hidden_size,
    sycl::queue& queue) -> std::pair<Tensor, Tensor>;

auto gru_cell_forward_kernel(
    const Tensor& reset_gates, const Tensor& update_gates,
    const Tensor& new_gates_input, const Tensor& new_gates_hidden,
    const Tensor& h_prev, int64_t batch_size, int64_t hidden_size,
    sycl::queue& queue) -> Tensor;

// Forward declarations of math kernels
auto matmul_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
auto add_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, sycl::queue& queue) -> Tensor;
auto clone_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

// Helper: slice tensor along dim=0 at position t (returns view via narrow-like copy)
static auto slice_at(const Tensor& input, int64_t t, sycl::queue& queue) -> Tensor {
    // input shape: (seq_len, batch, features)
    auto shape = input.shape();
    int64_t batch = shape[1];
    int64_t features = shape[2];
    int64_t elem_size = input.dtype_size();

    Tensor out({batch, features}, input.dtype(), input.device());
    const auto* src = static_cast<const uint8_t*>(input.data_ptr()) + t * batch * features * elem_size;
    auto* dst = const_cast<void*>(out.data_ptr());
    queue.memcpy(dst, src, batch * features * elem_size);
    return out;
}

// Helper: copy into output at time t
static void copy_to_time(Tensor& output, const Tensor& src, int64_t t, sycl::queue& queue) {
    auto shape = output.shape();
    int64_t batch = shape[1];
    int64_t features = shape[2];
    int64_t elem_size = output.dtype_size();

    auto* dst = static_cast<uint8_t*>(const_cast<void*>(output.data_ptr())) + t * batch * features * elem_size;
    const auto* s = static_cast<const uint8_t*>(src.data_ptr());
    queue.memcpy(dst, s, batch * features * elem_size);
}

// ============================================================================
// LSTM Forward (single layer, full sequence)
// ============================================================================
/**
 * @brief Full-sequence LSTM forward
 *
 * inputs: input (seq, batch, input_size), W_ih (4H, input_size), W_hh (4H, H),
 *         bias_ih (4H), bias_hh (4H), h0 (batch, H), c0 (batch, H)
 *
 * @return {output (seq, batch, H), h_n (batch, H), c_n (batch, H)}
 */
auto lstm_forward_kernel(
    const Tensor& input,     // (seq_len, batch, input_size)
    const Tensor& W_ih,      // (4*hidden_size, input_size)
    const Tensor& W_hh,      // (4*hidden_size, hidden_size)
    const Tensor& bias_ih,   // (4*hidden_size) or empty
    const Tensor& bias_hh,   // (4*hidden_size) or empty
    const Tensor& h0,        // (batch, hidden_size)
    const Tensor& c0,        // (batch, hidden_size)
    sycl::queue& queue
) -> std::vector<Tensor> {
    auto in_shape = input.shape();
    int64_t seq_len = in_shape[0];
    int64_t batch_size = in_shape[1];
    int64_t hidden_size = h0.shape()[1];

    // Transpose weights for matmul: W_ih^T (input_size, 4H), W_hh^T (H, 4H)
    // Actually we need input @ W_ih^T, so compute (batch, input_size) @ (input_size, 4H)
    // We'll use the standard matmul which does (M,K) @ (K,N) -> (M,N)

    // Pre-transpose weights: W_ih is (4H, input_size), we need (input_size, 4H)
    // For simplicity, compute gates = x @ W_ih^T + h @ W_hh^T + bias
    // matmul_kernel does (M,K) @ (K,N), but W_ih is (4H, I), so we use the transpose

    // Combine biases (handle case where one may be empty)
    bool has_bias_ih = (bias_ih.numel() > 0);
    bool has_bias_hh = (bias_hh.numel() > 0);
    bool has_bias = has_bias_ih || has_bias_hh;
    Tensor combined_bias;
    if (has_bias_ih && has_bias_hh) {
        combined_bias = add_kernel(bias_ih, bias_hh, queue);
    } else if (has_bias_ih) {
        combined_bias = bias_ih.contiguous();
    } else if (has_bias_hh) {
        combined_bias = bias_hh.contiguous();
    }

    Tensor output({seq_len, batch_size, hidden_size}, input.dtype(), input.device());
    Tensor h_t = clone_kernel(h0, queue);
    Tensor c_t = clone_kernel(c0, queue);

    // Transpose W_ih and W_hh for correct matmul dimensions
    // W_ih: (4H, I) -> need to compute x_t @ W_ih^T = (B, I) @ (I, 4H) = (B, 4H)
    // So we transpose W_ih to (I, 4H) -- but matmul_kernel expects contiguous (M,K)@(K,N)
    // Actually the standard approach: gates_ih = x_t @ W_ih^T
    // matmul_kernel(x_t, W_ih^T) where W_ih^T is (I, 4H)

    // Create transposed weights on host (once)
    // W_ih: (4H, I) -> (I, 4H)
    auto w_ih_shape = W_ih.shape();
    int64_t gate_size = w_ih_shape[0];  // 4*H
    int64_t input_size = w_ih_shape[1];

    // Transpose via permute-like copy
    Tensor W_ih_T({input_size, gate_size}, W_ih.dtype(), W_ih.device());
    {
        int64_t numel = gate_size * input_size;
        int64_t gs = gate_size;
        int64_t is = input_size;
        if (W_ih.dtype() == DType::Float32) {
            const float* src = static_cast<const float*>(W_ih.data_ptr());
            float* dst = static_cast<float*>(const_cast<void*>(W_ih_T.data_ptr()));
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
                int64_t r = idx / is;  // row in W_ih = gate dim
                int64_t c = idx % is;  // col in W_ih = input dim
                dst[c * gs + r] = src[r * is + c];
            });
        } else if (W_ih.dtype() == DType::Float64) {
            const double* src = static_cast<const double*>(W_ih.data_ptr());
            double* dst = static_cast<double*>(const_cast<void*>(W_ih_T.data_ptr()));
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
                int64_t r = idx / is;
                int64_t c = idx % is;
                dst[c * gs + r] = src[r * is + c];
            });
        } else {
            // BFloat16: treat as uint16_t
            const uint16_t* src = static_cast<const uint16_t*>(W_ih.data_ptr());
            uint16_t* dst = static_cast<uint16_t*>(const_cast<void*>(W_ih_T.data_ptr()));
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
                int64_t r = idx / is;
                int64_t c = idx % is;
                dst[c * gs + r] = src[r * is + c];
            });
        }
    }

    // Transpose W_hh: (4H, H) -> (H, 4H)
    auto w_hh_shape = W_hh.shape();
    Tensor W_hh_T({hidden_size, gate_size}, W_hh.dtype(), W_hh.device());
    {
        int64_t numel = gate_size * hidden_size;
        int64_t gs = gate_size;
        int64_t hs = hidden_size;
        if (W_hh.dtype() == DType::Float32) {
            const float* src = static_cast<const float*>(W_hh.data_ptr());
            float* dst = static_cast<float*>(const_cast<void*>(W_hh_T.data_ptr()));
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
                int64_t r = idx / hs;
                int64_t c = idx % hs;
                dst[c * gs + r] = src[r * hs + c];
            });
        } else if (W_hh.dtype() == DType::Float64) {
            const double* src = static_cast<const double*>(W_hh.data_ptr());
            double* dst = static_cast<double*>(const_cast<void*>(W_hh_T.data_ptr()));
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
                int64_t r = idx / hs;
                int64_t c = idx % hs;
                dst[c * gs + r] = src[r * hs + c];
            });
        } else {
            const uint16_t* src = static_cast<const uint16_t*>(W_hh.data_ptr());
            uint16_t* dst = static_cast<uint16_t*>(const_cast<void*>(W_hh_T.data_ptr()));
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
                int64_t r = idx / hs;
                int64_t c = idx % hs;
                dst[c * gs + r] = src[r * hs + c];
            });
        }
    }

    for (int64_t t = 0; t < seq_len; ++t) {
        Tensor x_t = slice_at(input, t, queue);

        // gates = x_t @ W_ih^T + h_t @ W_hh^T
        Tensor gates_ih = matmul_kernel(x_t, W_ih_T, queue);
        Tensor gates_hh = matmul_kernel(h_t, W_hh_T, queue);
        Tensor gates = add_kernel(gates_ih, gates_hh, queue);

        if (has_bias) {
            gates = add_kernel(gates, combined_bias, queue);
        }

        auto [h_new, c_new] = lstm_cell_forward_kernel(gates, c_t, batch_size, hidden_size, queue);
        h_t = h_new;
        c_t = c_new;

        copy_to_time(output, h_t, t, queue);
    }

    return {output, h_t, c_t};
}

// ============================================================================
// GRU Forward (single layer, full sequence)
// ============================================================================
auto gru_forward_kernel(
    const Tensor& input,     // (seq_len, batch, input_size)
    const Tensor& W_ih,      // (3*hidden_size, input_size)
    const Tensor& W_hh,      // (3*hidden_size, hidden_size)
    const Tensor& bias,      // (3*hidden_size) or empty -- combined ih+hh
    const Tensor& h0,        // (batch, hidden_size)
    sycl::queue& queue
) -> std::vector<Tensor> {
    auto in_shape = input.shape();
    int64_t seq_len = in_shape[0];
    int64_t batch_size = in_shape[1];
    int64_t hidden_size = h0.shape()[1];
    int64_t input_size = in_shape[2];
    int64_t gate3 = 3 * hidden_size;

    bool has_bias = (bias.numel() > 0);

    // For GRU, the 3 gates (reset, update, new) have separate input/hidden projections
    // gates_ih = x @ W_ih^T   shape (B, 3H)
    // gates_hh = h @ W_hh^T   shape (B, 3H)
    // Then split into 3 chunks of H each

    // Transpose weights
    auto transpose_2d = [&](const Tensor& w, int64_t rows, int64_t cols) -> Tensor {
        Tensor w_t({cols, rows}, w.dtype(), w.device());
        int64_t numel = rows * cols;
        if (w.dtype() == DType::Float32) {
            const float* src = static_cast<const float*>(w.data_ptr());
            float* dst = static_cast<float*>(const_cast<void*>(w_t.data_ptr()));
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
                int64_t r = idx / cols;
                int64_t c = idx % cols;
                dst[c * rows + r] = src[r * cols + c];
            });
        } else if (w.dtype() == DType::Float64) {
            const double* src = static_cast<const double*>(w.data_ptr());
            double* dst = static_cast<double*>(const_cast<void*>(w_t.data_ptr()));
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
                int64_t r = idx / cols;
                int64_t c = idx % cols;
                dst[c * rows + r] = src[r * cols + c];
            });
        } else {
            const uint16_t* src = static_cast<const uint16_t*>(w.data_ptr());
            uint16_t* dst = static_cast<uint16_t*>(const_cast<void*>(w_t.data_ptr()));
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
                int64_t r = idx / cols;
                int64_t c = idx % cols;
                dst[c * rows + r] = src[r * cols + c];
            });
        }
        return w_t;
    };

    Tensor W_ih_T = transpose_2d(W_ih, gate3, input_size);
    Tensor W_hh_T = transpose_2d(W_hh, gate3, hidden_size);

    // Split bias into ih and hh halves if present
    // For GRU, bias is combined: bias_ih + bias_hh = (3H) each concatenated -> (6H) or just (3H)?
    // Per CPU pattern, bias is a single (3H) tensor (pre-combined)

    Tensor output({seq_len, batch_size, hidden_size}, input.dtype(), input.device());
    Tensor h_t = clone_kernel(h0, queue);

    // Helper to slice (B, 3H) into 3 chunks of (B, H)
    auto slice_gate = [&](const Tensor& gates, int64_t gate_idx) -> Tensor {
        Tensor chunk({batch_size, hidden_size}, gates.dtype(), gates.device());
        int64_t elem_size = gates.dtype_size();
        const auto* src = static_cast<const uint8_t*>(gates.data_ptr()) +
                          gate_idx * hidden_size * elem_size;
        // Need to handle row-major layout: gates is (B, 3H), each row has 3H elements
        // gate_idx-th chunk: elements [gate_idx*H .. (gate_idx+1)*H) in each row
        if (gates.dtype() == DType::Float32) {
            const float* g = static_cast<const float*>(gates.data_ptr());
            float* c = static_cast<float*>(const_cast<void*>(chunk.data_ptr()));
            queue.parallel_for(sycl::range<1>(batch_size * hidden_size), [=](sycl::id<1> idx) {
                int64_t b = idx / hidden_size;
                int64_t h = idx % hidden_size;
                c[b * hidden_size + h] = g[b * gate3 + gate_idx * hidden_size + h];
            });
        } else if (gates.dtype() == DType::Float64) {
            const double* g = static_cast<const double*>(gates.data_ptr());
            double* c = static_cast<double*>(const_cast<void*>(chunk.data_ptr()));
            queue.parallel_for(sycl::range<1>(batch_size * hidden_size), [=](sycl::id<1> idx) {
                int64_t b = idx / hidden_size;
                int64_t h = idx % hidden_size;
                c[b * hidden_size + h] = g[b * gate3 + gate_idx * hidden_size + h];
            });
        } else {
            const uint16_t* g = static_cast<const uint16_t*>(gates.data_ptr());
            uint16_t* c = static_cast<uint16_t*>(const_cast<void*>(chunk.data_ptr()));
            queue.parallel_for(sycl::range<1>(batch_size * hidden_size), [=](sycl::id<1> idx) {
                int64_t b = idx / hidden_size;
                int64_t h = idx % hidden_size;
                c[b * hidden_size + h] = g[b * gate3 + gate_idx * hidden_size + h];
            });
        }
        return chunk;
    };

    for (int64_t t = 0; t < seq_len; ++t) {
        Tensor x_t = slice_at(input, t, queue);

        Tensor gates_ih = matmul_kernel(x_t, W_ih_T, queue);
        Tensor gates_hh = matmul_kernel(h_t, W_hh_T, queue);

        if (has_bias) {
            gates_ih = add_kernel(gates_ih, bias, queue);
        }

        // Split into reset, update, new for ih and hh
        Tensor reset_ih = slice_gate(gates_ih, 0);
        Tensor update_ih = slice_gate(gates_ih, 1);
        Tensor new_input = slice_gate(gates_ih, 2);

        Tensor reset_hh = slice_gate(gates_hh, 0);
        Tensor update_hh = slice_gate(gates_hh, 1);
        Tensor new_hidden = slice_gate(gates_hh, 2);

        // GRU cell: reset_gates = reset_ih + reset_hh, update_gates = update_ih + update_hh
        Tensor reset_gates = add_kernel(reset_ih, reset_hh, queue);
        Tensor update_gates = add_kernel(update_ih, update_hh, queue);

        h_t = gru_cell_forward_kernel(reset_gates, update_gates, new_input, new_hidden,
                                       h_t, batch_size, hidden_size, queue);

        copy_to_time(output, h_t, t, queue);
    }

    return {output, h_t};
}

// ============================================================================
// LSTM Multi-Layer Forward
// ============================================================================
auto lstm_multilayer_forward_kernel(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0,   // (num_layers, batch, hidden_size)
    const Tensor& c0,   // (num_layers, batch, hidden_size)
    sycl::queue& queue
) -> std::vector<Tensor> {
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto h0_shape = h0.shape();
    int64_t batch_size = h0_shape[1];
    int64_t hidden_size = h0_shape[2];

    Tensor current_input = clone_kernel(input, queue);
    Tensor h_n({num_layers, batch_size, hidden_size}, input.dtype(), input.device());
    Tensor c_n({num_layers, batch_size, hidden_size}, input.dtype(), input.device());

    for (int64_t l = 0; l < num_layers; ++l) {
        // Extract h0_l and c0_l for this layer
        Tensor h0_l = slice_at(h0, l, queue);
        Tensor c0_l = slice_at(c0, l, queue);

        // bias_list[l] may be a combined (8H) or split (4H + 4H) or empty
        // Per CPU pattern: bias is combined per layer
        // Split into bias_ih and bias_hh
        Tensor bias_ih, bias_hh;
        if (bias_list[l].numel() > 0) {
            int64_t gate_size = 4 * hidden_size;
            if (bias_list[l].numel() == 2 * gate_size) {
                // Combined: first 4H is bias_ih, second 4H is bias_hh
                bias_ih = Tensor({gate_size}, bias_list[l].dtype(), bias_list[l].device());
                bias_hh = Tensor({gate_size}, bias_list[l].dtype(), bias_list[l].device());
                int64_t elem_size = bias_list[l].dtype_size();
                queue.memcpy(const_cast<void*>(bias_ih.data_ptr()),
                            bias_list[l].data_ptr(), gate_size * elem_size);
                queue.memcpy(const_cast<void*>(bias_hh.data_ptr()),
                            static_cast<const uint8_t*>(bias_list[l].data_ptr()) + gate_size * elem_size,
                            gate_size * elem_size);
            } else {
                bias_ih = bias_list[l];
                bias_hh = Tensor();
            }
        } else {
            bias_ih = Tensor();
            bias_hh = Tensor();
        }

        auto results = lstm_forward_kernel(current_input, W_ih_list[l], W_hh_list[l],
                                           bias_ih, bias_hh, h0_l, c0_l, queue);
        current_input = results[0]; // output becomes input for next layer

        // Store final h and c for this layer
        copy_to_time(h_n, results[1], l, queue);
        copy_to_time(c_n, results[2], l, queue);
    }

    return {current_input, h_n, c_n};
}

// ============================================================================
// GRU Multi-Layer Forward
// ============================================================================
auto gru_multilayer_forward_kernel(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0,   // (num_layers, batch, hidden_size)
    sycl::queue& queue
) -> std::vector<Tensor> {
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto h0_shape = h0.shape();
    int64_t batch_size = h0_shape[1];
    int64_t hidden_size = h0_shape[2];

    Tensor current_input = clone_kernel(input, queue);
    Tensor h_n({num_layers, batch_size, hidden_size}, input.dtype(), input.device());

    for (int64_t l = 0; l < num_layers; ++l) {
        Tensor h0_l = slice_at(h0, l, queue);
        Tensor bias_l = bias_list[l];

        auto results = gru_forward_kernel(current_input, W_ih_list[l], W_hh_list[l],
                                          bias_l, h0_l, queue);
        current_input = results[0];
        copy_to_time(h_n, results[1], l, queue);
    }

    return {current_input, h_n};
}

// ============================================================================
// Bidirectional LSTM Forward
// ============================================================================
auto bilstm_forward_kernel(
    const Tensor& input,        // (seq_len, batch, input_size)
    const Tensor& h0,           // (2, batch, hidden_size)
    const Tensor& c0,           // (2, batch, hidden_size)
    const Tensor& W_ih_fwd,     // (4H, input_size)
    const Tensor& W_hh_fwd,     // (4H, hidden_size)
    const Tensor& bias_ih_fwd,  // (4H)
    const Tensor& bias_hh_fwd,  // (4H)
    const Tensor& W_ih_bwd,     // (4H, input_size)
    const Tensor& W_hh_bwd,     // (4H, hidden_size)
    const Tensor& bias_ih_bwd,  // (4H)
    const Tensor& bias_hh_bwd,  // (4H)
    sycl::queue& queue
) -> std::vector<Tensor> {
    auto in_shape = input.shape();
    int64_t seq_len = in_shape[0];
    int64_t batch_size = in_shape[1];
    int64_t hidden_size = h0.shape()[2];

    // Extract initial states for forward and backward directions
    Tensor h0_fwd = slice_at(h0, 0, queue);
    Tensor c0_fwd = slice_at(c0, 0, queue);
    Tensor h0_bwd = slice_at(h0, 1, queue);
    Tensor c0_bwd = slice_at(c0, 1, queue);

    // Forward direction
    auto fwd_result = lstm_forward_kernel(input, W_ih_fwd, W_hh_fwd,
                                          bias_ih_fwd, bias_hh_fwd,
                                          h0_fwd, c0_fwd, queue);
    Tensor fwd_output = fwd_result[0]; // (seq_len, batch, H)
    Tensor h_n_fwd = fwd_result[1];
    Tensor c_n_fwd = fwd_result[2];

    // Create reversed input for backward direction
    Tensor input_reversed({seq_len, batch_size, in_shape[2]}, input.dtype(), input.device());
    for (int64_t t = 0; t < seq_len; ++t) {
        Tensor x_t = slice_at(input, seq_len - 1 - t, queue);
        copy_to_time(input_reversed, x_t, t, queue);
    }

    auto bwd_result = lstm_forward_kernel(input_reversed, W_ih_bwd, W_hh_bwd,
                                          bias_ih_bwd, bias_hh_bwd,
                                          h0_bwd, c0_bwd, queue);
    Tensor bwd_output_reversed = bwd_result[0];
    Tensor h_n_bwd = bwd_result[1];
    Tensor c_n_bwd = bwd_result[2];

    // Reverse backward output to match original time order
    Tensor bwd_output({seq_len, batch_size, hidden_size}, input.dtype(), input.device());
    for (int64_t t = 0; t < seq_len; ++t) {
        Tensor x_t = slice_at(bwd_output_reversed, seq_len - 1 - t, queue);
        copy_to_time(bwd_output, x_t, t, queue);
    }

    // Concatenate forward and backward outputs along feature dim
    Tensor combined_output({seq_len, batch_size, 2 * hidden_size}, input.dtype(), input.device());
    int64_t elem_size = input.dtype_size();
    for (int64_t t = 0; t < seq_len; ++t) {
        Tensor fwd_t = slice_at(fwd_output, t, queue);
        Tensor bwd_t = slice_at(bwd_output, t, queue);
        // Copy fwd to first H columns, bwd to second H columns
        for (int64_t b = 0; b < batch_size; ++b) {
            auto* dst = static_cast<uint8_t*>(const_cast<void*>(combined_output.data_ptr())) +
                        (t * batch_size * 2 * hidden_size + b * 2 * hidden_size) * elem_size;
            const auto* fwd_src = static_cast<const uint8_t*>(fwd_t.data_ptr()) +
                                  b * hidden_size * elem_size;
            const auto* bwd_src = static_cast<const uint8_t*>(bwd_t.data_ptr()) +
                                  b * hidden_size * elem_size;
            queue.memcpy(dst, fwd_src, hidden_size * elem_size);
            queue.memcpy(dst + hidden_size * elem_size, bwd_src, hidden_size * elem_size);
        }
    }

    // Stack h_n and c_n: (2, batch, hidden_size)
    Tensor h_n({2, batch_size, hidden_size}, input.dtype(), input.device());
    Tensor c_n({2, batch_size, hidden_size}, input.dtype(), input.device());
    copy_to_time(h_n, h_n_fwd, 0, queue);
    copy_to_time(h_n, h_n_bwd, 1, queue);
    copy_to_time(c_n, c_n_fwd, 0, queue);
    copy_to_time(c_n, c_n_bwd, 1, queue);

    return {combined_output, h_n, c_n};
}

} // namespace oneapi
} // namespace tenzor
