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

// ============================================================================
// GRU Cell Kernels
// ============================================================================

/**
 * @brief Fused GRU cell forward kernel
 *
 * Computes all 3 gates (reset, update, new) in a single kernel.
 *
 * GRU equations:
 *   r_t = σ(W_ir @ x_t + W_hr @ h_{t-1} + b_r)
 *   z_t = σ(W_iz @ x_t + W_hz @ h_{t-1} + b_z)
 *   n_t = tanh(W_in @ x_t + r_t ⊙ (W_hn @ h_{t-1} + b_hn))
 *   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
 *
 * @param reset_gates Pre-computed reset gate values: (batch, hidden_size)
 * @param update_gates Pre-computed update gate values: (batch, hidden_size)
 * @param new_gates_input Pre-computed new gate input part: (batch, hidden_size)
 * @param new_gates_hidden Pre-computed new gate hidden part: (batch, hidden_size)
 * @param h_prev Previous hidden state: (batch, hidden_size)
 * @param h_out Output hidden state: (batch, hidden_size)
 * @param batch_size Number of sequences
 * @param hidden_size Hidden dimension
 */
template<typename T>
__global__ void gru_cell_forward_fused(
    const T* __restrict__ reset_gates,
    const T* __restrict__ update_gates,
    const T* __restrict__ new_gates_input,
    const T* __restrict__ new_gates_hidden,
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    int64_t batch_size,
    int64_t hidden_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * hidden_size;

    if (idx < total_elements) {
        // Load pre-computed gate values
        T r_gate = reset_gates[idx];
        T z_gate = update_gates[idx];
        T n_input = new_gates_input[idx];
        T n_hidden = new_gates_hidden[idx];
        T h_prev_val = h_prev[idx];

        // Apply sigmoid activation to reset and update gates
        T r_t = T(1) / (T(1) + exp(-r_gate));
        T z_t = T(1) / (T(1) + exp(-z_gate));

        // Apply reset gate to hidden part and compute new gate
        T n_combined = n_input + r_t * n_hidden;
        T n_t = tanh(n_combined);

        // Compute new hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        T h_t = (T(1) - z_t) * n_t + z_t * h_prev_val;

        // Store output
        h_out[idx] = h_t;
    }
}

/**
 * @brief Fused GRU cell backward kernel
 *
 * Computes gradients for all 3 gates in a single kernel.
 *
 * @param grad_h Gradient w.r.t. hidden output: (batch, hidden_size)
 * @param reset_gates Reset gate values from forward: (batch, hidden_size)
 * @param update_gates Update gate values from forward: (batch, hidden_size)
 * @param new_gates_input New gate input part from forward: (batch, hidden_size)
 * @param new_gates_hidden New gate hidden part from forward: (batch, hidden_size)
 * @param h_prev Previous hidden state: (batch, hidden_size)
 * @param grad_reset Output gradients for reset gates: (batch, hidden_size)
 * @param grad_update Output gradients for update gates: (batch, hidden_size)
 * @param grad_new_input Output gradients for new gate input: (batch, hidden_size)
 * @param grad_new_hidden Output gradients for new gate hidden: (batch, hidden_size)
 * @param grad_h_prev Output gradient for previous hidden: (batch, hidden_size)
 */
template<typename T>
__global__ void gru_cell_backward_fused(
    const T* __restrict__ grad_h,
    const T* __restrict__ reset_gates,
    const T* __restrict__ update_gates,
    const T* __restrict__ new_gates_input,
    const T* __restrict__ new_gates_hidden,
    const T* __restrict__ h_prev,
    T* __restrict__ grad_reset,
    T* __restrict__ grad_update,
    T* __restrict__ grad_new_input,
    T* __restrict__ grad_new_hidden,
    T* __restrict__ grad_h_prev,
    int64_t batch_size,
    int64_t hidden_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * hidden_size;

    if (idx < total_elements) {
        // Recompute forward pass values
        T r_gate = reset_gates[idx];
        T z_gate = update_gates[idx];
        T n_input = new_gates_input[idx];
        T n_hidden = new_gates_hidden[idx];
        T h_prev_val = h_prev[idx];

        T r_t = T(1) / (T(1) + exp(-r_gate));
        T z_t = T(1) / (T(1) + exp(-z_gate));

        T n_combined = n_input + r_t * n_hidden;
        T n_t = tanh(n_combined);

        // Load incoming gradient
        T dh = grad_h[idx];

        // Gradient through h_t = (1 - z_t) * n_t + z_t * h_prev
        T dn_t = dh * (T(1) - z_t);
        T dz_t = dh * (h_prev_val - n_t);
        T dh_prev = dh * z_t;

        // Gradient through n_t = tanh(n_combined)
        // tanh'(x) = 1 - tanh^2(x)
        T dn_combined = dn_t * (T(1) - n_t * n_t);

        // Gradient through n_combined = n_input + r_t * n_hidden
        T dn_input = dn_combined;
        T dr_t = dn_combined * n_hidden;
        T dn_hidden = dn_combined * r_t;

        // Gradient through sigmoid activations
        // σ'(x) = σ(x) * (1 - σ(x))
        T dr_gate = dr_t * r_t * (T(1) - r_t);
        T dz_gate = dz_t * z_t * (T(1) - z_t);

        // Store gradients
        grad_reset[idx] = dr_gate;
        grad_update[idx] = dz_gate;
        grad_new_input[idx] = dn_input;
        grad_new_hidden[idx] = dn_hidden;

        // Gradient for previous hidden also comes from new gate hidden part
        dh_prev += dn_hidden;  // This will be added to dh_prev from z_t path
        grad_h_prev[idx] = dh_prev;
    }
}

// ============================================================================
// Host Interface Functions
// ============================================================================

extern "C" {
    /**
     * @brief GRU cell forward pass (Float32)
     */
    void gru_cell_forward_float(
        const float* reset_gates,
        const float* update_gates,
        const float* new_gates_input,
        const float* new_gates_hidden,
        const float* h_prev,
        float* h_out,
        int64_t batch_size,
        int64_t hidden_size) {

        int64_t total = batch_size * hidden_size;
        int num_blocks = get_num_blocks(total);

        gru_cell_forward_fused<float><<<num_blocks, BLOCK_SIZE>>>(
            reset_gates, update_gates, new_gates_input, new_gates_hidden,
            h_prev, h_out, batch_size, hidden_size);

        CUDA_CHECK(cudaGetLastError());
    }

    /**
     * @brief GRU cell forward pass (Float64)
     */
    void gru_cell_forward_double(
        const double* reset_gates,
        const double* update_gates,
        const double* new_gates_input,
        const double* new_gates_hidden,
        const double* h_prev,
        double* h_out,
        int64_t batch_size,
        int64_t hidden_size) {

        int64_t total = batch_size * hidden_size;
        int num_blocks = get_num_blocks(total);

        gru_cell_forward_fused<double><<<num_blocks, BLOCK_SIZE>>>(
            reset_gates, update_gates, new_gates_input, new_gates_hidden,
            h_prev, h_out, batch_size, hidden_size);

        CUDA_CHECK(cudaGetLastError());
    }

    /**
     * @brief GRU cell backward pass (Float32)
     */
    void gru_cell_backward_float(
        const float* grad_h,
        const float* reset_gates,
        const float* update_gates,
        const float* new_gates_input,
        const float* new_gates_hidden,
        const float* h_prev,
        float* grad_reset,
        float* grad_update,
        float* grad_new_input,
        float* grad_new_hidden,
        float* grad_h_prev,
        int64_t batch_size,
        int64_t hidden_size) {

        int64_t total = batch_size * hidden_size;
        int num_blocks = get_num_blocks(total);

        gru_cell_backward_fused<float><<<num_blocks, BLOCK_SIZE>>>(
            grad_h, reset_gates, update_gates, new_gates_input, new_gates_hidden,
            h_prev, grad_reset, grad_update, grad_new_input, grad_new_hidden,
            grad_h_prev, batch_size, hidden_size);

        CUDA_CHECK(cudaGetLastError());
    }

    /**
     * @brief GRU cell backward pass (Float64)
     */
    void gru_cell_backward_double(
        const double* grad_h,
        const double* reset_gates,
        const double* update_gates,
        const double* new_gates_input,
        const double* new_gates_hidden,
        const double* h_prev,
        double* grad_reset,
        double* grad_update,
        double* grad_new_input,
        double* grad_new_hidden,
        double* grad_h_prev,
        int64_t batch_size,
        int64_t hidden_size) {

        int64_t total = batch_size * hidden_size;
        int num_blocks = get_num_blocks(total);

        gru_cell_backward_fused<double><<<num_blocks, BLOCK_SIZE>>>(
            grad_h, reset_gates, update_gates, new_gates_input, new_gates_hidden,
            h_prev, grad_reset, grad_update, grad_new_input, grad_new_hidden,
            grad_h_prev, batch_size, hidden_size);

        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Tensor Wrapper Functions
// ============================================================================

/**
 * @brief GRU cell forward wrapper for Tensor API
 *
 * @param reset_gates Pre-computed reset gate values (batch, hidden_size)
 * @param update_gates Pre-computed update gate values (batch, hidden_size)
 * @param new_gates_input New gate input part (batch, hidden_size)
 * @param new_gates_hidden New gate hidden part (batch, hidden_size)
 * @param h_prev Previous hidden state (batch, hidden_size)
 * @param batch_size Number of sequences
 * @param hidden_size Hidden dimension
 * @return Output hidden state tensor
 */
auto gru_cell_forward_kernel(
    const Tensor& reset_gates,
    const Tensor& update_gates,
    const Tensor& new_gates_input,
    const Tensor& new_gates_hidden,
    const Tensor& h_prev,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream) -> Tensor {

    std::vector<int64_t> output_shape = {batch_size, hidden_size};
    Tensor h_out(output_shape, reset_gates.dtype(), reset_gates.device());

    int64_t total = batch_size * hidden_size;
    int num_blocks = get_num_blocks(total);

    if (reset_gates.dtype() == DType::Float32) {
        gru_cell_forward_fused<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reset_gates.data<float>(),
            update_gates.data<float>(),
            new_gates_input.data<float>(),
            new_gates_hidden.data<float>(),
            h_prev.data<float>(),
            h_out.data<float>(),
            batch_size,
            hidden_size);
            CUDA_CHECK(cudaGetLastError());
    } else if (reset_gates.dtype() == DType::Float64) {
        gru_cell_forward_fused<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reset_gates.data<double>(),
            update_gates.data<double>(),
            new_gates_input.data<double>(),
            new_gates_hidden.data<double>(),
            h_prev.data<double>(),
            h_out.data<double>(),
            batch_size,
            hidden_size);
            CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("GRU only supports Float32 and Float64");
    }

    CUDA_CHECK(cudaGetLastError());

    return h_out;
}

/**
 * @brief GRU cell backward wrapper for Tensor API
 */
struct GRUBackwardOutputs {
    Tensor grad_reset;
    Tensor grad_update;
    Tensor grad_new_input;
    Tensor grad_new_hidden;
    Tensor grad_h_prev;
};

auto gru_cell_backward_kernel(
    const Tensor& grad_h,
    const Tensor& reset_gates,
    const Tensor& update_gates,
    const Tensor& new_gates_input,
    const Tensor& new_gates_hidden,
    const Tensor& h_prev,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream) -> GRUBackwardOutputs {

    std::vector<int64_t> state_shape = {batch_size, hidden_size};

    GRUBackwardOutputs outputs;
    outputs.grad_reset = Tensor(state_shape, grad_h.dtype(), grad_h.device());
    outputs.grad_update = Tensor(state_shape, grad_h.dtype(), grad_h.device());
    outputs.grad_new_input = Tensor(state_shape, grad_h.dtype(), grad_h.device());
    outputs.grad_new_hidden = Tensor(state_shape, grad_h.dtype(), grad_h.device());
    outputs.grad_h_prev = Tensor(state_shape, grad_h.dtype(), grad_h.device());

    int64_t total = batch_size * hidden_size;
    int num_blocks = get_num_blocks(total);

    if (grad_h.dtype() == DType::Float32) {
        gru_cell_backward_fused<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_h.data<float>(),
            reset_gates.data<float>(),
            update_gates.data<float>(),
            new_gates_input.data<float>(),
            new_gates_hidden.data<float>(),
            h_prev.data<float>(),
            outputs.grad_reset.data<float>(),
            outputs.grad_update.data<float>(),
            outputs.grad_new_input.data<float>(),
            outputs.grad_new_hidden.data<float>(),
            outputs.grad_h_prev.data<float>(),
            batch_size,
            hidden_size);
            CUDA_CHECK(cudaGetLastError());
    } else if (grad_h.dtype() == DType::Float64) {
        gru_cell_backward_fused<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_h.data<double>(),
            reset_gates.data<double>(),
            update_gates.data<double>(),
            new_gates_input.data<double>(),
            new_gates_hidden.data<double>(),
            h_prev.data<double>(),
            outputs.grad_reset.data<double>(),
            outputs.grad_update.data<double>(),
            outputs.grad_new_input.data<double>(),
            outputs.grad_new_hidden.data<double>(),
            outputs.grad_h_prev.data<double>(),
            batch_size,
            hidden_size);
            CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("GRU backward only supports Float32 and Float64");
    }

    CUDA_CHECK(cudaGetLastError());

    return outputs;
}

} // namespace cuda
} // namespace tenzor
