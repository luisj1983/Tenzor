/**
 * @file rnn_sequence.cu
 * @brief Full-sequence LSTM/GRU forward kernels for CUDA
 *
 * Implements single-layer, multi-layer, and bidirectional variants.
 * Uses cuBLAS for weight projections and existing fused cell kernels
 * for element-wise gate computation. All ops stay on a single CUDA
 * stream with no per-timestep synchronization.
 */

#ifdef TENZOR_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "cuda_common.cuh"
#include <cublas_v2.h>
#include <vector>
#include <stdexcept>

namespace tenzor {
namespace cuda {

// Forward declarations from other CUDA kernels
auto cublas_matmul(const Tensor& a, const Tensor& b) -> Tensor;
auto add_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;

// From lstm.cu
auto lstm_cell_forward_kernel(
    const Tensor& gates,
    const Tensor& c_prev,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream) -> std::pair<Tensor, Tensor>;

// ============================================================================
// Bias addition kernel (row-wise broadcast)
// ============================================================================

template<typename T>
__global__ void add_bias_kernel(
    T* __restrict__ output,
    const T* __restrict__ bias,
    int64_t batch_size,
    int64_t feature_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * feature_size;
    if (idx < total) {
        int64_t col = idx % feature_size;
        output[idx] += bias[col];
    }
}

template<typename T>
static void launch_add_bias(T* output, const T* bias,
                            int64_t batch, int64_t features, cudaStream_t stream) {
    int64_t total = batch * features;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_kernel<<<grid, block, 0, stream>>>(output, bias, batch, features);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

// ============================================================================
// GRU fused cell kernel for sequence ops
// ============================================================================

template<typename T>
__global__ void gru_cell_fused_kernel(
    const T* __restrict__ gates_ih, // (batch, 3*hidden) — W_ih @ x + bias_ih
    const T* __restrict__ gates_hh, // (batch, 3*hidden) — W_hh @ h + bias_hh
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    int64_t batch_size,
    int64_t hidden_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * hidden_size;
    if (idx >= total) return;

    int64_t b = idx / hidden_size;
    int64_t h = idx % hidden_size;

    // Gate offsets
    int64_t base_ih = b * 3 * hidden_size + h;
    int64_t base_hh = b * 3 * hidden_size + h;

    T r_ih = gates_ih[base_ih];
    T z_ih = gates_ih[base_ih + hidden_size];
    T n_ih = gates_ih[base_ih + 2 * hidden_size];

    T r_hh = gates_hh[base_hh];
    T z_hh = gates_hh[base_hh + hidden_size];
    T n_hh = gates_hh[base_hh + 2 * hidden_size];

    // Reset gate: r = sigmoid(r_ih + r_hh)
    T r = T(1) / (T(1) + exp(-(r_ih + r_hh)));
    // Update gate: z = sigmoid(z_ih + z_hh)
    T z = T(1) / (T(1) + exp(-(z_ih + z_hh)));
    // New gate: n = tanh(n_ih + r * n_hh)
    T n = tanh(n_ih + r * n_hh);

    T hp = h_prev[idx];
    h_out[idx] = (T(1) - z) * n + z * hp;
}

// ============================================================================
// Helper: compute gates = x @ W^T  (using cublas_matmul with transposed weight)
// We store weights as (out_features, in_features) and matmul as (batch, in) @ (in, out)
// So we need to transpose W first, or use cublas directly.
// For simplicity, use cublas_matmul which handles 2D matmul.
// W is (out, in), W^T is (in, out), x is (batch, in), result is (batch, out)
// ============================================================================

static Tensor matmul_weight_t(const Tensor& x, const Tensor& W, cudaStream_t /*stream*/) {
    // W is (out_features, in_features)
    // We need x @ W^T = x @ W.transpose(0,1)
    // cublas_matmul(a, b) computes a @ b
    // So we need to transpose W first
    Tensor W_t = W.transpose(0, 1).contiguous();
    return cublas_matmul(x, W_t);
}

// ============================================================================
// LSTM Forward (single layer, full sequence)
// ============================================================================

auto lstm_forward_cuda(
    const Tensor& input,     // (seq_len, batch, input_size)
    const Tensor& W_ih,      // (4*hidden, input_size)
    const Tensor& W_hh,      // (4*hidden, hidden)
    const Tensor& bias_ih,   // (4*hidden) or empty
    const Tensor& bias_hh,   // (4*hidden) or empty
    const Tensor& h0,        // (batch, hidden)
    const Tensor& c0         // (batch, hidden)
) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t input_size = shape[2];
    int64_t hidden = h0.shape()[1];
    int64_t gate_size = 4 * hidden;

    cudaStream_t stream = nullptr;

    // Output: (seq_len, batch, hidden)
    Tensor output({seq_len, batch, hidden}, input.dtype(), input.device());

    Tensor h_prev = h0.contiguous();
    Tensor c_prev = c0.contiguous();

    bool has_bias_ih = bias_ih.numel() > 0;
    bool has_bias_hh = bias_hh.numel() > 0;

    // Pre-transpose weights once (avoids per-timestep transpose)
    Tensor W_ih_t = W_ih.transpose(0, 1).contiguous();  // (input_size, 4*hidden)
    Tensor W_hh_t = W_hh.transpose(0, 1).contiguous();  // (hidden, 4*hidden)

    // Pre-compute all input gates: (seq_len*batch, input_size) @ (input_size, 4*hidden) -> (seq_len*batch, 4*hidden)
    Tensor input_2d = input.reshape({seq_len * batch, input_size});
    Tensor all_gates_ih = cublas_matmul(input_2d, W_ih_t);  // (seq_len*batch, 4*hidden)

    // Add input bias to all gates at once
    if (has_bias_ih) {
        if (input.dtype() == DType::Float32) {
            launch_add_bias(all_gates_ih.data<float>(), bias_ih.data<float>(), seq_len * batch, gate_size, stream);
        } else if (input.dtype() == DType::Float64) {
            launch_add_bias(all_gates_ih.data<double>(), bias_ih.data<double>(), seq_len * batch, gate_size, stream);
        }
    }

    // Reshape to (seq_len, batch, gate_size) for zero-copy per-timestep slicing
    Tensor all_gates_ih_3d = all_gates_ih.reshape({seq_len, batch, gate_size});

    size_t hidden_step_bytes = batch * hidden * dtype_size(input.dtype());

    // Pre-allocate hidden-to-hidden gate buffer to avoid per-timestep allocation
    Tensor gates_hh_buf({batch, gate_size}, input.dtype(), input.device());

    for (int64_t t = 0; t < seq_len; ++t) {
        // Zero-copy view into pre-computed input gates for this timestep: (batch, 4*hidden)
        Tensor gates_ih_t = all_gates_ih_3d.slice(0, t, t + 1).squeeze(0);

        // Hidden-to-hidden: h_prev @ W_hh_t (pre-transposed)
        Tensor gates_hh = cublas_matmul(h_prev, W_hh_t);

        // Combine input and hidden gates
        Tensor gates = add_kernel(gates_ih_t, gates_hh, stream);

        // Add hidden bias
        if (has_bias_hh && input.dtype() == DType::Float32) {
            launch_add_bias(gates.data<float>(), bias_hh.data<float>(), batch, gate_size, stream);
        }
        if (has_bias_hh && input.dtype() == DType::Float64) {
            launch_add_bias(gates.data<double>(), bias_hh.data<double>(), batch, gate_size, stream);
        }

        // LSTM cell: apply activations and compute h, c
        auto [h_out, c_out] = lstm_cell_forward_kernel(gates, c_prev, batch, hidden, stream);

        // Copy h_out to output[t] (async D2D on same stream, no sync needed)
        cudaMemcpyAsync(
            static_cast<char*>(output.data_ptr()) + t * hidden_step_bytes,
            h_out.data_ptr(),
            hidden_step_bytes,
            cudaMemcpyDeviceToDevice,
            stream);

        h_prev = h_out;
        c_prev = c_out;
    }

    return {output, h_prev, c_prev};
}

// ============================================================================
// GRU Forward (single layer, full sequence)
// ============================================================================

auto gru_forward_cuda(
    const Tensor& input,     // (seq_len, batch, input_size)
    const Tensor& W_ih,      // (3*hidden, input_size)
    const Tensor& W_hh,      // (3*hidden, hidden)
    const Tensor& bias,      // (3*hidden) or empty — combined bias
    const Tensor& h0         // (batch, hidden)
) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t input_size = shape[2];
    int64_t hidden = h0.shape()[1];
    int64_t gate_size = 3 * hidden;

    cudaStream_t stream = nullptr;

    Tensor output({seq_len, batch, hidden}, input.dtype(), input.device());
    Tensor h_prev = h0.contiguous();

    bool has_bias = bias.numel() > 0;

    // Pre-transpose weights once
    Tensor W_ih_t = W_ih.transpose(0, 1).contiguous();  // (input_size, 3*hidden)
    Tensor W_hh_t = W_hh.transpose(0, 1).contiguous();  // (hidden, 3*hidden)

    // Pre-compute all input gates: (seq_len*batch, input_size) @ (input_size, 3*hidden)
    Tensor input_2d = input.reshape({seq_len * batch, input_size});
    Tensor all_gates_ih = cublas_matmul(input_2d, W_ih_t);  // (seq_len*batch, 3*hidden)

    // Add bias to all input gates at once
    if (has_bias) {
        if (input.dtype() == DType::Float32) {
            launch_add_bias(all_gates_ih.data<float>(), bias.data<float>(), seq_len * batch, gate_size, stream);
        } else if (input.dtype() == DType::Float64) {
            launch_add_bias(all_gates_ih.data<double>(), bias.data<double>(), seq_len * batch, gate_size, stream);
        }
    }

    // Reshape to (seq_len, batch, gate_size) for zero-copy per-timestep slicing
    Tensor all_gates_ih_3d = all_gates_ih.reshape({seq_len, batch, gate_size});

    // Pre-allocate reusable h_out buffer (avoids per-timestep allocation)
    Tensor h_out({batch, hidden}, input.dtype(), input.device());
    size_t hidden_step_bytes = batch * hidden * dtype_size(input.dtype());

    // Pre-compute GRU cell launch config (constant across timesteps)
    int64_t total = batch * hidden;
    int block = 256;
    int grid = (total + block - 1) / block;

    for (int64_t t = 0; t < seq_len; ++t) {
        // Zero-copy view into pre-computed input gates for this timestep
        Tensor gates_ih_t = all_gates_ih_3d.slice(0, t, t + 1).squeeze(0);

        // Hidden-to-hidden: h_prev @ W_hh_t (pre-transposed)
        Tensor gates_hh = cublas_matmul(h_prev, W_hh_t);

        // GRU cell — write into pre-allocated h_out buffer
        if (input.dtype() == DType::Float32) {
            gru_cell_fused_kernel<float><<<grid, block, 0, stream>>>(
                gates_ih_t.data<float>(), gates_hh.data<float>(),
                h_prev.data<float>(), h_out.data<float>(),
                batch, hidden);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else if (input.dtype() == DType::Float64) {
            gru_cell_fused_kernel<double><<<grid, block, 0, stream>>>(
                gates_ih_t.data<double>(), gates_hh.data<double>(),
                h_prev.data<double>(), h_out.data<double>(),
                batch, hidden);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }

        cudaMemcpyAsync(
            static_cast<char*>(output.data_ptr()) + t * hidden_step_bytes,
            h_out.data_ptr(),
            hidden_step_bytes,
            cudaMemcpyDeviceToDevice,
            stream);

        // Swap: h_prev now points to current output, h_out gets a fresh buffer for next step
        std::swap(h_prev, h_out);
    }

    // After the loop, h_prev holds the final hidden state (due to the swap at end of last iteration)
    // But we need to return the actual last h — which was written to output and also to h_out before swap
    // After last iteration's swap, h_prev = last h_out (the result), h_out = previous h_prev
    // So h_prev is correct.

    return {output, h_prev};
}

// ============================================================================
// Multi-layer LSTM Forward
// ============================================================================

auto lstm_multi_layer_forward_cuda(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0,    // (num_layers, batch, hidden)
    const Tensor& c0     // (num_layers, batch, hidden)
) -> std::vector<Tensor> {
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t hidden = h0.shape()[2];

    // Final hidden/cell states: (num_layers, batch, hidden)
    Tensor h_n({num_layers, batch, hidden}, input.dtype(), input.device());
    Tensor c_n({num_layers, batch, hidden}, input.dtype(), input.device());

    Tensor layer_input = input;
    size_t layer_bytes = batch * hidden * dtype_size(input.dtype());

    for (int64_t l = 0; l < num_layers; ++l) {
        // Extract h0/c0 for this layer
        Tensor h_l = h0.slice(0, l, l + 1).squeeze(0).contiguous();
        Tensor c_l = c0.slice(0, l, l + 1).squeeze(0).contiguous();

        // Split bias into bias_ih and bias_hh if present
        Tensor bias_ih, bias_hh;
        if (bias_list[l].numel() > 0) {
            // bias_list[l] is combined (8*hidden for LSTM: 4*hidden bias_ih + 4*hidden bias_hh)
            int64_t half = bias_list[l].numel() / 2;
            bias_ih = bias_list[l].slice(0, 0, half).contiguous();
            bias_hh = bias_list[l].slice(0, half, half + half).contiguous();
        }

        auto result = lstm_forward_cuda(
            layer_input, W_ih_list[l], W_hh_list[l],
            bias_ih, bias_hh, h_l, c_l);

        layer_input = result[0];  // output becomes input for next layer

        // Copy final h, c to h_n[l], c_n[l]
        cudaMemcpyAsync(
            static_cast<char*>(h_n.data_ptr()) + l * layer_bytes,
            result[1].data_ptr(), layer_bytes,
            cudaMemcpyDeviceToDevice, nullptr);
        cudaMemcpyAsync(
            static_cast<char*>(c_n.data_ptr()) + l * layer_bytes,
            result[2].data_ptr(), layer_bytes,
            cudaMemcpyDeviceToDevice, nullptr);
    }

    return {layer_input, h_n, c_n};
}

// ============================================================================
// Multi-layer GRU Forward
// ============================================================================

auto gru_multi_layer_forward_cuda(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0    // (num_layers, batch, hidden)
) -> std::vector<Tensor> {
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t hidden = h0.shape()[2];

    Tensor h_n({num_layers, batch, hidden}, input.dtype(), input.device());
    Tensor layer_input = input;
    size_t layer_bytes = batch * hidden * dtype_size(input.dtype());

    for (int64_t l = 0; l < num_layers; ++l) {
        Tensor h_l = h0.slice(0, l, l + 1).squeeze(0).contiguous();

        auto result = gru_forward_cuda(
            layer_input, W_ih_list[l], W_hh_list[l],
            bias_list[l], h_l);

        layer_input = result[0];

        cudaMemcpyAsync(
            static_cast<char*>(h_n.data_ptr()) + l * layer_bytes,
            result[1].data_ptr(), layer_bytes,
            cudaMemcpyDeviceToDevice, nullptr);
    }

    return {layer_input, h_n};
}

// ============================================================================
// Bidirectional LSTM Forward
// ============================================================================

// Kernel to concatenate forward and backward LSTM outputs along the hidden dimension
template<typename T>
__global__ void bilstm_concat_kernel(
    const T* __restrict__ fwd_output,   // (seq_len, batch, hidden)
    const T* __restrict__ bwd_output,   // (seq_len, batch, hidden)
    T* __restrict__ output,             // (seq_len, batch, 2*hidden)
    int64_t seq_len,
    int64_t batch,
    int64_t hidden
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = seq_len * batch * hidden;
    if (idx >= total) return;

    int64_t h = idx % hidden;
    int64_t rem = idx / hidden;
    int64_t b = rem % batch;
    int64_t t = rem / batch;

    int64_t src_offset = (t * batch + b) * hidden + h;
    int64_t dst_base = (t * batch + b) * 2 * hidden;

    output[dst_base + h] = fwd_output[src_offset];
    output[dst_base + hidden + h] = bwd_output[src_offset];
}

// Kernel to reverse a sequence along dim 0: output[t] = input[seq_len-1-t]
template<typename T>
__global__ void reverse_sequence_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t seq_len,
    int64_t batch,
    int64_t hidden
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = seq_len * batch * hidden;
    if (idx >= total) return;

    int64_t step_size = batch * hidden;
    int64_t t = idx / step_size;
    int64_t offset = idx % step_size;
    int64_t t_rev = seq_len - 1 - t;

    output[t_rev * step_size + offset] = input[t * step_size + offset];
}

auto bilstm_forward_cuda(
    const Tensor& input,
    const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
    const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
    const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
    const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd,
    const Tensor& h0,    // (2, batch, hidden)
    const Tensor& c0     // (2, batch, hidden)
) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t hidden = h0.shape()[2];

    cudaStream_t stream = nullptr;

    // Forward direction
    Tensor h0_fwd = h0.slice(0, 0, 1).squeeze(0).contiguous();
    Tensor c0_fwd = c0.slice(0, 0, 1).squeeze(0).contiguous();
    auto fwd_result = lstm_forward_cuda(
        input, W_ih_fwd, W_hh_fwd, bias_ih_fwd, bias_hh_fwd, h0_fwd, c0_fwd);

    // Backward direction: reverse input, run LSTM, reverse output
    Tensor input_rev({seq_len, batch, shape[2]}, input.dtype(), input.device());
    int64_t total = seq_len * batch * shape[2];
    int block = 256;
    int grid = (total + block - 1) / block;

    if (input.dtype() == DType::Float32) {
        reverse_sequence_kernel<float><<<grid, block, 0, stream>>>(
            input.data<float>(), input_rev.data<float>(),
            seq_len, batch, shape[2]);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        reverse_sequence_kernel<double><<<grid, block, 0, stream>>>(
            input.data<double>(), input_rev.data<double>(),
            seq_len, batch, shape[2]);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    Tensor h0_bwd = h0.slice(0, 1, 2).squeeze(0).contiguous();
    Tensor c0_bwd = c0.slice(0, 1, 2).squeeze(0).contiguous();
    auto bwd_result = lstm_forward_cuda(
        input_rev, W_ih_bwd, W_hh_bwd, bias_ih_bwd, bias_hh_bwd, h0_bwd, c0_bwd);

    // Reverse backward output
    Tensor bwd_output_rev({seq_len, batch, hidden}, input.dtype(), input.device());
    int64_t total_out = seq_len * batch * hidden;
    grid = (total_out + block - 1) / block;

    if (input.dtype() == DType::Float32) {
        reverse_sequence_kernel<float><<<grid, block, 0, stream>>>(
            bwd_result[0].data<float>(), bwd_output_rev.data<float>(),
            seq_len, batch, hidden);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        reverse_sequence_kernel<double><<<grid, block, 0, stream>>>(
            bwd_result[0].data<double>(), bwd_output_rev.data<double>(),
            seq_len, batch, hidden);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    // Concatenate forward and backward outputs along hidden dim: (seq_len, batch, 2*hidden)
    Tensor output({seq_len, batch, 2 * hidden}, input.dtype(), input.device());

    {
        int64_t concat_total = seq_len * batch * hidden;
        int concat_block = 256;
        int concat_grid = (concat_total + concat_block - 1) / concat_block;

        if (input.dtype() == DType::Float32) {
            bilstm_concat_kernel<float><<<concat_grid, concat_block, 0, stream>>>(
                fwd_result[0].data<float>(), bwd_output_rev.data<float>(),
                output.data<float>(), seq_len, batch, hidden);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else if (input.dtype() == DType::Float64) {
            bilstm_concat_kernel<double><<<concat_grid, concat_block, 0, stream>>>(
                fwd_result[0].data<double>(), bwd_output_rev.data<double>(),
                output.data<double>(), seq_len, batch, hidden);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }
    }

    // Stack h_n: (2, batch, hidden)
    Tensor h_n({2, batch, hidden}, input.dtype(), input.device());
    Tensor c_n({2, batch, hidden}, input.dtype(), input.device());
    size_t state_bytes = batch * hidden * dtype_size(input.dtype());

    cudaMemcpyAsync(h_n.data_ptr(), fwd_result[1].data_ptr(),
                    state_bytes, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(static_cast<char*>(h_n.data_ptr()) + state_bytes,
                    bwd_result[1].data_ptr(),
                    state_bytes, cudaMemcpyDeviceToDevice, stream);

    cudaMemcpyAsync(c_n.data_ptr(), fwd_result[2].data_ptr(),
                    state_bytes, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(static_cast<char*>(c_n.data_ptr()) + state_bytes,
                    bwd_result[2].data_ptr(),
                    state_bytes, cudaMemcpyDeviceToDevice, stream);

    return {output, h_n, c_n};
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_CUDA_AVAILABLE
