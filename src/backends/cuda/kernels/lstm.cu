#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include "../cuda_error.hpp"

namespace tenzor {
namespace cuda {

#define CUDA_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return (n + block_size - 1) / block_size;
}

// ==================================================================
/**
 * @brief Fused LSTM cell forward kernel
 *
 * Computes all 4 gates (input, forget, cell, output) in a single kernel.
 * This reduces memory bandwidth and improves performance.
 *
 * LSTM equations:
 *   i_t = σ(W_ii @ x_t + W_hi @ h_{t-1} + b_i)
 *   f_t = σ(W_if @ x_t + W_hf @ h_{t-1} + b_f)
 *   g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)
 *   o_t = σ(W_io @ x_t + W_ho @ h_{t-1} + b_o)
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
 *   h_t = o_t ⊙ tanh(c_t)
 *
 * @param x Input at time t: (batch, input_size)
 * @param h_prev Hidden state at t-1: (batch, hidden_size)
 * @param c_prev Cell state at t-1: (batch, hidden_size)
 * @param gates Pre-computed gate values: (batch, 4 * hidden_size)
 *              [input_gate | forget_gate | cell_gate | output_gate]
 * @param h_out Output hidden state: (batch, hidden_size)
 * @param c_out Output cell state: (batch, hidden_size)
 * @param batch_size Number of sequences
 * @param hidden_size Hidden dimension
 */
template<typename T>
__global__ void lstm_cell_forward_fused(
    const T* __restrict__ gates,
    const T* __restrict__ c_prev,
    T* __restrict__ h_out,
    T* __restrict__ c_out,
    int64_t batch_size,
    int64_t hidden_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * hidden_size;

    if (idx < total_elements) {
        int64_t batch_idx = idx / hidden_size;
        int64_t hidden_idx = idx % hidden_size;

        // Offsets for each gate
        int64_t gate_offset = batch_idx * (4 * hidden_size) + hidden_idx;
        int64_t i_offset = gate_offset;                      // Input gate
        int64_t f_offset = gate_offset + hidden_size;        // Forget gate
        int64_t g_offset = gate_offset + 2 * hidden_size;    // Cell gate
        int64_t o_offset = gate_offset + 3 * hidden_size;    // Output gate

        // Load pre-computed gate values (W @ [x, h] + b)
        T i_gate = gates[i_offset];
        T f_gate = gates[f_offset];
        T g_gate = gates[g_offset];
        T o_gate = gates[o_offset];

        // Apply activations
        // Sigmoid for i, f, o gates
        T i_t = T(1) / (T(1) + exp(-i_gate));
        T f_t = T(1) / (T(1) + exp(-f_gate));
        T o_t = T(1) / (T(1) + exp(-o_gate));

        // Tanh for cell gate
        T g_t = tanh(g_gate);

        // Load previous cell state
        T c_prev_val = c_prev[idx];

        // Update cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        T c_t = f_t * c_prev_val + i_t * g_t;

        // Update hidden state: h_t = o_t ⊙ tanh(c_t)
        T h_t = o_t * tanh(c_t);

        // Store outputs
        c_out[idx] = c_t;
        h_out[idx] = h_t;
    }
}

/**
 * @brief Fused LSTM cell backward kernel
 *
 * Computes gradients for all gates in a single kernel.
 *
 * @param grad_h Gradient w.r.t. hidden output: (batch, hidden_size)
 * @param grad_c Gradient w.r.t. cell output: (batch, hidden_size)
 * @param gates Gate values from forward: (batch, 4 * hidden_size)
 * @param c_prev Previous cell state: (batch, hidden_size)
 * @param c_out Current cell state: (batch, hidden_size)
 * @param grad_gates Output gradients for gates: (batch, 4 * hidden_size)
 * @param grad_c_prev Output gradient for previous cell: (batch, hidden_size)
 */
template<typename T>
__global__ void lstm_cell_backward_fused(
    const T* __restrict__ grad_h,
    const T* __restrict__ grad_c,
    const T* __restrict__ gates,
    const T* __restrict__ c_prev,
    const T* __restrict__ c_out,
    T* __restrict__ grad_gates,
    T* __restrict__ grad_c_prev,
    int64_t batch_size,
    int64_t hidden_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * hidden_size;

    if (idx < total_elements) {
        int64_t batch_idx = idx / hidden_size;
        int64_t hidden_idx = idx % hidden_size;

        // Offsets
        int64_t gate_offset = batch_idx * (4 * hidden_size) + hidden_idx;
        int64_t i_offset = gate_offset;
        int64_t f_offset = gate_offset + hidden_size;
        int64_t g_offset = gate_offset + 2 * hidden_size;
        int64_t o_offset = gate_offset + 3 * hidden_size;

        // Load gate values and recompute activations
        T i_gate = gates[i_offset];
        T f_gate = gates[f_offset];
        T g_gate = gates[g_offset];
        T o_gate = gates[o_offset];

        T i_t = T(1) / (T(1) + exp(-i_gate));
        T f_t = T(1) / (T(1) + exp(-f_gate));
        T g_t = tanh(g_gate);
        T o_t = T(1) / (T(1) + exp(-o_gate));

        // Load states
        T c_prev_val = c_prev[idx];
        T c_t = c_out[idx];
        T tanh_c_t = tanh(c_t);

        // Load incoming gradients
        T dh = grad_h[idx];
        T dc = grad_c[idx];

        // Gradient through h = o * tanh(c)
        dc += dh * o_t * (T(1) - tanh_c_t * tanh_c_t);  // tanh derivative
        T do_t = dh * tanh_c_t;

        // Gradient through c = f * c_prev + i * g
        T df_t = dc * c_prev_val;
        T di_t = dc * g_t;
        T dg_t = dc * i_t;
        T dc_prev = dc * f_t;

        // Gradient through gate activations
        // Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
        T di_gate = di_t * i_t * (T(1) - i_t);
        T df_gate = df_t * f_t * (T(1) - f_t);
        T do_gate = do_t * o_t * (T(1) - o_t);

        // Tanh derivative: tanh'(x) = 1 - tanh^2(x)
        T dg_gate = dg_t * (T(1) - g_t * g_t);

        // Store gradients
        grad_gates[i_offset] = di_gate;
        grad_gates[f_offset] = df_gate;
        grad_gates[g_offset] = dg_gate;
        grad_gates[o_offset] = do_gate;
        grad_c_prev[idx] = dc_prev;
    }
}

// ============================================================================
// Host Interface Functions
// ============================================================================

extern "C" {
    /**
     * @brief LSTM cell forward pass (Float32)
     */
    void lstm_cell_forward_float(
        const float* gates,
        const float* c_prev,
        float* h_out,
        float* c_out,
        int64_t batch_size,
        int64_t hidden_size) {

        int64_t total = batch_size * hidden_size;
        int num_blocks = get_num_blocks(total);

        lstm_cell_forward_fused<float><<<num_blocks, BLOCK_SIZE>>>(
            gates, c_prev, h_out, c_out, batch_size, hidden_size);

        CUDA_CHECK(cudaGetLastError());
    }

    /**
     * @brief LSTM cell forward pass (Float64)
     */
    void lstm_cell_forward_double(
        const double* gates,
        const double* c_prev,
        double* h_out,
        double* c_out,
        int64_t batch_size,
        int64_t hidden_size) {

        int64_t total = batch_size * hidden_size;
        int num_blocks = get_num_blocks(total);

        lstm_cell_forward_fused<double><<<num_blocks, BLOCK_SIZE>>>(
            gates, c_prev, h_out, c_out, batch_size, hidden_size);

        CUDA_CHECK(cudaGetLastError());
    }

    /**
     * @brief LSTM cell backward pass (Float32)
     */
    void lstm_cell_backward_float(
        const float* grad_h,
        const float* grad_c,
        const float* gates,
        const float* c_prev,
        const float* c_out,
        float* grad_gates,
        float* grad_c_prev,
        int64_t batch_size,
        int64_t hidden_size) {

        int64_t total = batch_size * hidden_size;
        int num_blocks = get_num_blocks(total);

        lstm_cell_backward_fused<float><<<num_blocks, BLOCK_SIZE>>>(
            grad_h, grad_c, gates, c_prev, c_out,
            grad_gates, grad_c_prev, batch_size, hidden_size);

        CUDA_CHECK(cudaGetLastError());
    }

    /**
     * @brief LSTM cell backward pass (Float64)
     */
    void lstm_cell_backward_double(
        const double* grad_h,
        const double* grad_c,
        const double* gates,
        const double* c_prev,
        const double* c_out,
        double* grad_gates,
        double* grad_c_prev,
        int64_t batch_size,
        int64_t hidden_size) {

        int64_t total = batch_size * hidden_size;
        int num_blocks = get_num_blocks(total);

        lstm_cell_backward_fused<double><<<num_blocks, BLOCK_SIZE>>>(
            grad_h, grad_c, gates, c_prev, c_out,
            grad_gates, grad_c_prev, batch_size, hidden_size);

        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Tensor Wrapper Functions
// ============================================================================

/**
 * @brief LSTM cell forward wrapper for Tensor API
 *
 * @param gates Pre-computed gate values (batch, 4 * hidden_size)
 * @param c_prev Previous cell state (batch, hidden_size)
 * @param batch_size Number of sequences
 * @param hidden_size Hidden dimension
 * @return Pair of (h_out, c_out) tensors
 */
auto lstm_cell_forward_kernel(
    const Tensor& gates,
    const Tensor& c_prev,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream) -> std::pair<Tensor, Tensor> {

    std::vector<int64_t> output_shape = {batch_size, hidden_size};
    Tensor h_out(output_shape, gates.dtype(), gates.device());
    Tensor c_out(output_shape, gates.dtype(), gates.device());

    int64_t total = batch_size * hidden_size;
    int num_blocks = get_num_blocks(total);

    if (gates.dtype() == DType::Float32) {
        lstm_cell_forward_fused<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            gates.data<float>(),
            c_prev.data<float>(),
            h_out.data<float>(),
            c_out.data<float>(),
            batch_size,
            hidden_size);
            CUDA_CHECK(cudaGetLastError());
    } else if (gates.dtype() == DType::Float64) {
        lstm_cell_forward_fused<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            gates.data<double>(),
            c_prev.data<double>(),
            h_out.data<double>(),
            c_out.data<double>(),
            batch_size,
            hidden_size);
            CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("LSTM only supports Float32 and Float64");
    }

    CUDA_CHECK(cudaGetLastError());

    return {h_out, c_out};
}

/**
 * @brief LSTM cell backward wrapper for Tensor API
 */
auto lstm_cell_backward_kernel(
    const Tensor& grad_h,
    const Tensor& grad_c,
    const Tensor& gates,
    const Tensor& c_prev,
    const Tensor& c_out,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream) -> std::pair<Tensor, Tensor> {

    std::vector<int64_t> gate_shape = {batch_size, 4 * hidden_size};
    std::vector<int64_t> state_shape = {batch_size, hidden_size};

    Tensor grad_gates(gate_shape, gates.dtype(), gates.device());
    Tensor grad_c_prev(state_shape, gates.dtype(), gates.device());

    int64_t total = batch_size * hidden_size;
    int num_blocks = get_num_blocks(total);

    if (gates.dtype() == DType::Float32) {
        lstm_cell_backward_fused<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_h.data<float>(),
            grad_c.data<float>(),
            gates.data<float>(),
            c_prev.data<float>(),
            c_out.data<float>(),
            grad_gates.data<float>(),
            grad_c_prev.data<float>(),
            batch_size,
            hidden_size);
            CUDA_CHECK(cudaGetLastError());
    } else if (gates.dtype() == DType::Float64) {
        lstm_cell_backward_fused<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_h.data<double>(),
            grad_c.data<double>(),
            gates.data<double>(),
            c_prev.data<double>(),
            c_out.data<double>(),
            grad_gates.data<double>(),
            grad_c_prev.data<double>(),
            batch_size,
            hidden_size);
            CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("LSTM backward only supports Float32 and Float64");
    }

    CUDA_CHECK(cudaGetLastError());

    return {grad_gates, grad_c_prev};
}

} // namespace cuda
} // namespace tenzor
