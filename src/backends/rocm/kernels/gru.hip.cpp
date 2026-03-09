#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>

namespace tenzor {
namespace rocm {

// ============================================================================
// HIP Helper Functions
// ============================================================================

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define HIP_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return (n + block_size - 1) / block_size;
}

// RAII wrapper for rocBLAS handle to prevent leaks on exceptions
class RocBLASHandleGuard {
public:
    RocBLASHandleGuard() {
        rocblas_status status = rocblas_create_handle(&handle_);
        if (status != rocblas_status_success) {
            throw std::runtime_error("Failed to create rocBLAS handle");
        }
    }
    ~RocBLASHandleGuard() {
        if (handle_) {
            rocblas_destroy_handle(handle_);
        }
    }
    RocBLASHandleGuard(const RocBLASHandleGuard&) = delete;
    RocBLASHandleGuard& operator=(const RocBLASHandleGuard&) = delete;

    rocblas_handle get() const { return handle_; }

    void set_stream(hipStream_t stream) {
        rocblas_set_stream(handle_, stream);
    }
private:
    rocblas_handle handle_ = nullptr;
};

// Kernel to add bias (broadcasted across batch dimension)
template<typename T>
__global__ void add_bias_kernel(const T* __restrict__ bias, T* __restrict__ gates,
                                 int64_t batch, int64_t gate_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * gate_size) {
        int64_t g = idx % gate_size;
        gates[idx] += bias[g];
    }
}

// Kernel to combine r and z gates from ih and hh parts
template<typename T>
__global__ void combine_rz_gates_kernel(const T* __restrict__ gates_ih,
                                         const T* __restrict__ gates_hh,
                                         T* __restrict__ rz_gates,
                                         int64_t batch,
                                         int64_t hidden) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * 2 * hidden) {
        rz_gates[idx] = gates_ih[idx] + gates_hh[idx];
    }
}

// Kernel to compute GRU from pre-computed gate values
// For full sequence GRU, we compute: gates = [r_ih + r_hh, z_ih + z_hh, n_ih, n_hh]
// This kernel applies the GRU equations given these pre-computed values
template<typename T>
__global__ void gru_sequence_step_kernel(
    const T* __restrict__ rz_gates,    // (batch, 2*hidden) - combined r,z gates
    const T* __restrict__ n_ih_gates,  // (batch, hidden) - new gate input part
    const T* __restrict__ n_hh_gates,  // (batch, hidden) - new gate hidden part
    const T* __restrict__ h_prev,
    T* __restrict__ h_out,
    int64_t batch,
    int64_t hidden) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * hidden) {
        int64_t b = idx / hidden;
        int64_t h = idx % hidden;

        // Load gate values
        T r_gate = rz_gates[b * 2 * hidden + h];
        T z_gate = rz_gates[b * 2 * hidden + hidden + h];
        T n_ih = n_ih_gates[idx];
        T n_hh = n_hh_gates[idx];
        T h_prev_val = h_prev[idx];

        // Apply sigmoid to r and z
        T r_t = T(1) / (T(1) + exp(-r_gate));
        T z_t = T(1) / (T(1) + exp(-z_gate));

        // Compute new gate: n_t = tanh(n_ih + r_t * n_hh)
        T n_t = tanh(n_ih + r_t * n_hh);

        // Compute new hidden state: h_t = (1 - z_t) * n_t + z_t * h_prev
        h_out[idx] = (T(1) - z_t) * n_t + z_t * h_prev_val;
    }
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
// Float16 GRU Cell Kernels
// ============================================================================

/**
 * @brief Float16 GRU cell forward kernel
 *
 * Uses float computation internally for numerical accuracy.
 */
__global__ void gru_cell_forward_fused_fp16(
    const __half* __restrict__ reset_gates,
    const __half* __restrict__ update_gates,
    const __half* __restrict__ new_gates_input,
    const __half* __restrict__ new_gates_hidden,
    const __half* __restrict__ h_prev,
    __half* __restrict__ h_out,
    int64_t batch_size,
    int64_t hidden_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * hidden_size;

    if (idx < total_elements) {
        // Load and convert to float for computation
        float r_gate = __half2float(reset_gates[idx]);
        float z_gate = __half2float(update_gates[idx]);
        float n_input = __half2float(new_gates_input[idx]);
        float n_hidden = __half2float(new_gates_hidden[idx]);
        float h_prev_val = __half2float(h_prev[idx]);

        // Apply sigmoid activation to reset and update gates
        float r_t = 1.0f / (1.0f + expf(-r_gate));
        float z_t = 1.0f / (1.0f + expf(-z_gate));

        // Apply reset gate to hidden part and compute new gate
        float n_combined = n_input + r_t * n_hidden;
        float n_t = tanhf(n_combined);

        // Compute new hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        float h_t = (1.0f - z_t) * n_t + z_t * h_prev_val;

        // Store output
        h_out[idx] = __float2half(h_t);
    }
}

/**
 * @brief Float16 GRU cell backward kernel
 *
 * Uses float computation internally for numerical accuracy.
 */
__global__ void gru_cell_backward_fused_fp16(
    const __half* __restrict__ grad_h,
    const __half* __restrict__ reset_gates,
    const __half* __restrict__ update_gates,
    const __half* __restrict__ new_gates_input,
    const __half* __restrict__ new_gates_hidden,
    const __half* __restrict__ h_prev,
    __half* __restrict__ grad_reset,
    __half* __restrict__ grad_update,
    __half* __restrict__ grad_new_input,
    __half* __restrict__ grad_new_hidden,
    __half* __restrict__ grad_h_prev,
    int64_t batch_size,
    int64_t hidden_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * hidden_size;

    if (idx < total_elements) {
        // Recompute forward pass values (in float)
        float r_gate = __half2float(reset_gates[idx]);
        float z_gate = __half2float(update_gates[idx]);
        float n_input = __half2float(new_gates_input[idx]);
        float n_hidden = __half2float(new_gates_hidden[idx]);
        float h_prev_val = __half2float(h_prev[idx]);

        float r_t = 1.0f / (1.0f + expf(-r_gate));
        float z_t = 1.0f / (1.0f + expf(-z_gate));

        float n_combined = n_input + r_t * n_hidden;
        float n_t = tanhf(n_combined);

        // Load incoming gradient
        float dh = __half2float(grad_h[idx]);

        // Gradient through h_t = (1 - z_t) * n_t + z_t * h_prev
        float dn_t = dh * (1.0f - z_t);
        float dz_t = dh * (h_prev_val - n_t);
        float dh_prev_val = dh * z_t;

        // Gradient through n_t = tanh(n_combined)
        // tanh'(x) = 1 - tanh^2(x)
        float dn_combined = dn_t * (1.0f - n_t * n_t);

        // Gradient through n_combined = n_input + r_t * n_hidden
        float dn_input = dn_combined;
        float dr_t = dn_combined * n_hidden;
        float dn_hidden_val = dn_combined * r_t;

        // Gradient through sigmoid activations
        // σ'(x) = σ(x) * (1 - σ(x))
        float dr_gate = dr_t * r_t * (1.0f - r_t);
        float dz_gate = dz_t * z_t * (1.0f - z_t);

        // Store gradients
        grad_reset[idx] = __float2half(dr_gate);
        grad_update[idx] = __float2half(dz_gate);
        grad_new_input[idx] = __float2half(dn_input);
        grad_new_hidden[idx] = __float2half(dn_hidden_val);

        // Gradient for previous hidden also comes from new gate hidden part
        dh_prev_val += dn_hidden_val;
        grad_h_prev[idx] = __float2half(dh_prev_val);
    }
}

/**
 * @brief Float16 GRU sequence step kernel
 */
__global__ void gru_sequence_step_kernel_fp16(
    const __half* __restrict__ rz_gates,
    const __half* __restrict__ n_ih_gates,
    const __half* __restrict__ n_hh_gates,
    const __half* __restrict__ h_prev,
    __half* __restrict__ h_out,
    int64_t batch,
    int64_t hidden) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * hidden) {
        int64_t b = idx / hidden;
        int64_t h = idx % hidden;

        // Load gate values and convert to float
        float r_gate = __half2float(rz_gates[b * 2 * hidden + h]);
        float z_gate = __half2float(rz_gates[b * 2 * hidden + hidden + h]);
        float n_ih = __half2float(n_ih_gates[idx]);
        float n_hh = __half2float(n_hh_gates[idx]);
        float h_prev_val = __half2float(h_prev[idx]);

        // Apply sigmoid to r and z
        float r_t = 1.0f / (1.0f + expf(-r_gate));
        float z_t = 1.0f / (1.0f + expf(-z_gate));

        // Compute new gate: n_t = tanh(n_ih + r_t * n_hh)
        float n_t = tanhf(n_ih + r_t * n_hh);

        // Compute new hidden state: h_t = (1 - z_t) * n_t + z_t * h_prev
        h_out[idx] = __float2half((1.0f - z_t) * n_t + z_t * h_prev_val);
    }
}

/**
 * @brief Float16 kernel to add bias (broadcasted across batch dimension)
 */
__global__ void add_bias_kernel_fp16(const __half* __restrict__ bias, __half* __restrict__ gates,
                                      int64_t batch, int64_t gate_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * gate_size) {
        int64_t g = idx % gate_size;
        float gate_val = __half2float(gates[idx]);
        float bias_val = __half2float(bias[g]);
        gates[idx] = __float2half(gate_val + bias_val);
    }
}

/**
 * @brief Float16 kernel to combine r and z gates from ih and hh parts
 */
__global__ void combine_rz_gates_kernel_fp16(const __half* __restrict__ gates_ih,
                                              const __half* __restrict__ gates_hh,
                                              __half* __restrict__ rz_gates,
                                              int64_t batch,
                                              int64_t hidden) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * 2 * hidden) {
        float ih_val = __half2float(gates_ih[idx]);
        float hh_val = __half2float(gates_hh[idx]);
        rz_gates[idx] = __float2half(ih_val + hh_val);
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

        hipLaunchKernelGGL(gru_cell_forward_fused<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          reset_gates, update_gates, new_gates_input, new_gates_hidden,
                          h_prev, h_out, batch_size, hidden_size);

        HIP_CHECK(hipGetLastError());
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

        hipLaunchKernelGGL(gru_cell_forward_fused<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          reset_gates, update_gates, new_gates_input, new_gates_hidden,
                          h_prev, h_out, batch_size, hidden_size);

        HIP_CHECK(hipGetLastError());
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

        hipLaunchKernelGGL(gru_cell_backward_fused<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_h, reset_gates, update_gates, new_gates_input, new_gates_hidden,
                          h_prev, grad_reset, grad_update, grad_new_input, grad_new_hidden,
                          grad_h_prev, batch_size, hidden_size);

        HIP_CHECK(hipGetLastError());
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

        hipLaunchKernelGGL(gru_cell_backward_fused<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_h, reset_gates, update_gates, new_gates_input, new_gates_hidden,
                          h_prev, grad_reset, grad_update, grad_new_input, grad_new_hidden,
                          grad_h_prev, batch_size, hidden_size);

        HIP_CHECK(hipGetLastError());
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
    hipStream_t stream) -> Tensor {

    std::vector<int64_t> output_shape = {batch_size, hidden_size};
    Tensor h_out(output_shape, reset_gates.dtype(), reset_gates.device());

    int64_t total = batch_size * hidden_size;
    int num_blocks = get_num_blocks(total);

    if (reset_gates.dtype() == DType::Float32) {
        hipLaunchKernelGGL(gru_cell_forward_fused<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reset_gates.data<float>(),
                          update_gates.data<float>(),
                          new_gates_input.data<float>(),
                          new_gates_hidden.data<float>(),
                          h_prev.data<float>(),
                          h_out.data<float>(),
                          batch_size,
                          hidden_size);
    } else if (reset_gates.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gru_cell_forward_fused<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reset_gates.data<double>(),
                          update_gates.data<double>(),
                          new_gates_input.data<double>(),
                          new_gates_hidden.data<double>(),
                          h_prev.data<double>(),
                          h_out.data<double>(),
                          batch_size,
                          hidden_size);
    } else if (reset_gates.dtype() == DType::Float16) {
        hipLaunchKernelGGL(gru_cell_forward_fused_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(reset_gates.data<Float16>()),
                          reinterpret_cast<const __half*>(update_gates.data<Float16>()),
                          reinterpret_cast<const __half*>(new_gates_input.data<Float16>()),
                          reinterpret_cast<const __half*>(new_gates_hidden.data<Float16>()),
                          reinterpret_cast<const __half*>(h_prev.data<Float16>()),
                          reinterpret_cast<__half*>(h_out.data<Float16>()),
                          batch_size,
                          hidden_size);
    } else if (reset_gates.dtype() == DType::BFloat16) {
        auto rg_f32 = reset_gates.to(DType::Float32);
        auto ug_f32 = update_gates.to(DType::Float32);
        auto ngi_f32 = new_gates_input.to(DType::Float32);
        auto ngh_f32 = new_gates_hidden.to(DType::Float32);
        auto hp_f32 = h_prev.to(DType::Float32);
        auto result = gru_cell_forward_kernel(rg_f32, ug_f32, ngi_f32, ngh_f32, hp_f32,
                                               batch_size, hidden_size, stream);
        return result.to(DType::BFloat16);
    } else {
        throw std::runtime_error("GRU only supports Float32, Float64, Float16, and BFloat16");
    }

    HIP_CHECK(hipGetLastError());

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
    hipStream_t stream) -> GRUBackwardOutputs {

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
        hipLaunchKernelGGL(gru_cell_backward_fused<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
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
    } else if (grad_h.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gru_cell_backward_fused<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
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
    } else if (grad_h.dtype() == DType::Float16) {
        hipLaunchKernelGGL(gru_cell_backward_fused_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_h.data<Float16>()),
                          reinterpret_cast<const __half*>(reset_gates.data<Float16>()),
                          reinterpret_cast<const __half*>(update_gates.data<Float16>()),
                          reinterpret_cast<const __half*>(new_gates_input.data<Float16>()),
                          reinterpret_cast<const __half*>(new_gates_hidden.data<Float16>()),
                          reinterpret_cast<const __half*>(h_prev.data<Float16>()),
                          reinterpret_cast<__half*>(outputs.grad_reset.data<Float16>()),
                          reinterpret_cast<__half*>(outputs.grad_update.data<Float16>()),
                          reinterpret_cast<__half*>(outputs.grad_new_input.data<Float16>()),
                          reinterpret_cast<__half*>(outputs.grad_new_hidden.data<Float16>()),
                          reinterpret_cast<__half*>(outputs.grad_h_prev.data<Float16>()),
                          batch_size,
                          hidden_size);
    } else if (grad_h.dtype() == DType::BFloat16) {
        auto gh_f32 = grad_h.to(DType::Float32);
        auto rg_f32 = reset_gates.to(DType::Float32);
        auto ug_f32 = update_gates.to(DType::Float32);
        auto ngi_f32 = new_gates_input.to(DType::Float32);
        auto ngh_f32 = new_gates_hidden.to(DType::Float32);
        auto hp_f32 = h_prev.to(DType::Float32);
        auto result = gru_cell_backward_kernel(gh_f32, rg_f32, ug_f32, ngi_f32, ngh_f32, hp_f32,
                                                batch_size, hidden_size, stream);
        result.grad_reset = result.grad_reset.to(DType::BFloat16);
        result.grad_update = result.grad_update.to(DType::BFloat16);
        result.grad_new_input = result.grad_new_input.to(DType::BFloat16);
        result.grad_new_hidden = result.grad_new_hidden.to(DType::BFloat16);
        result.grad_h_prev = result.grad_h_prev.to(DType::BFloat16);
        return result;
    } else {
        throw std::runtime_error("GRU backward only supports Float32, Float64, Float16, and BFloat16");
    }

    HIP_CHECK(hipGetLastError());

    return outputs;
}

// ============================================================================
// Full Sequence GRU Forward
// ============================================================================

/**
 * @brief Full sequence GRU forward pass
 *
 * Processes entire sequence using rocBLAS for efficient matrix operations.
 *
 * @param input Input sequence (seq_len, batch, input_size)
 * @param W_ih Input-to-hidden weights (3*hidden, input_size)
 * @param W_hh Hidden-to-hidden weights (3*hidden, hidden)
 * @param bias Combined bias (3*hidden) or empty tensor
 * @param h0 Initial hidden state (batch, hidden)
 * @param stream HIP stream for async execution
 * @return vector of [output, h_n]
 */
auto gru_forward_kernel(
    const Tensor& input,
    const Tensor& W_ih,
    const Tensor& W_hh,
    const Tensor& bias,
    const Tensor& h0,
    hipStream_t stream) -> std::vector<Tensor> {

    // BFloat16 upcast: convert to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto W_ih_f32 = W_ih.to(DType::Float32);
        auto W_hh_f32 = W_hh.to(DType::Float32);
        auto bias_f32 = (bias.numel() > 0) ? bias.to(DType::Float32) : bias;
        auto h0_f32 = h0.to(DType::Float32);
        auto results = gru_forward_kernel(input_f32, W_ih_f32, W_hh_f32, bias_f32, h0_f32, stream);
        results[0] = results[0].to(DType::BFloat16);  // output
        results[1] = results[1].to(DType::BFloat16);  // h_n
        return results;
    }

    // Get dimensions
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden = h0.shape()[1];

    // Allocate output tensors
    Tensor output({seq_len, batch, hidden}, input.dtype(), input.device());
    Tensor h_n({batch, hidden}, input.dtype(), input.device());

    // Temporary tensors for gate computations
    // GRU has 3 gates: r(reset), z(update), n(new)
    // We compute gates_ih (from input) and gates_hh (from hidden) separately for n gate
    Tensor gates_ih({batch, 3 * hidden}, input.dtype(), input.device());  // r_ih, z_ih, n_ih
    Tensor gates_hh({batch, 3 * hidden}, input.dtype(), input.device());  // r_hh, z_hh, n_hh
    Tensor rz_gates({batch, 2 * hidden}, input.dtype(), input.device());  // combined r, z

    // Current hidden state
    Tensor h_t({batch, hidden}, input.dtype(), input.device());

    // Copy h0 to h_t
    hipMemcpyAsync(h_t.data<void>(), h0.data<void>(),
                   batch * hidden * dtype_size(input.dtype()),
                   hipMemcpyDeviceToDevice, stream);

    if (input.dtype() == DType::Float32) {
        RocBLASHandleGuard handle_guard;
        rocblas_handle handle = handle_guard.get();
        handle_guard.set_stream(stream);

        const float alpha = 1.0f;
        const float beta_zero = 0.0f;
        const float beta_one = 1.0f;

        const float* W_ih_ptr = W_ih.data<float>();
        const float* W_hh_ptr = W_hh.data<float>();
        const float* input_ptr = input.data<float>();
        const float* bias_ptr = bias.numel() > 0 ? bias.data<float>() : nullptr;
        float* gates_ih_ptr = gates_ih.data<float>();
        float* gates_hh_ptr = gates_hh.data<float>();
        float* rz_gates_ptr = rz_gates.data<float>();
        float* h_t_ptr = h_t.data<float>();
        float* output_ptr = output.data<float>();

        // Process each timestep
        for (int64_t t = 0; t < seq_len; ++t) {
            const float* x_t = input_ptr + t * batch * input_size;

            // Compute input contribution: gates_ih = x_t @ W_ih^T
            rocblas_sgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                         3 * hidden, batch, input_size,
                         &alpha,
                         W_ih_ptr, input_size,
                         x_t, input_size,
                         &beta_zero,
                         gates_ih_ptr, 3 * hidden);

            // Compute hidden contribution: gates_hh = h_t @ W_hh^T
            rocblas_sgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                         3 * hidden, batch, hidden,
                         &alpha,
                         W_hh_ptr, hidden,
                         h_t_ptr, hidden,
                         &beta_zero,
                         gates_hh_ptr, 3 * hidden);

            // Add bias if present (split between ih and hh parts)
            if (bias_ptr != nullptr) {
                // Add bias to gates_ih
                int64_t total = batch * 3 * hidden;
                int num_blocks = get_num_blocks(total);
                hipLaunchKernelGGL(add_bias_kernel<float>,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    bias_ptr, gates_ih_ptr, batch, 3 * hidden);
            }

            // Combine r and z gates: rz = gates_ih[:, :2*hidden] + gates_hh[:, :2*hidden]
            // n_ih = gates_ih[:, 2*hidden:]
            // n_hh = gates_hh[:, 2*hidden:]
            // Apply GRU step kernel
            int64_t total = batch * hidden;
            int num_blocks = get_num_blocks(total);

            // Combine the rz gates
            hipLaunchKernelGGL(combine_rz_gates_kernel<float>,
                dim3(get_num_blocks(batch * 2 * hidden)), dim3(BLOCK_SIZE), 0, stream,
                gates_ih_ptr, gates_hh_ptr, rz_gates_ptr, batch, hidden);

            // Apply GRU step
            hipLaunchKernelGGL(gru_sequence_step_kernel<float>,
                              dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                              rz_gates_ptr,
                              gates_ih_ptr + 2 * hidden,  // n_ih part
                              gates_hh_ptr + 2 * hidden,  // n_hh part
                              h_t_ptr,
                              h_t_ptr,
                              batch, hidden);

            // Copy h_t to output[t]
            hipMemcpyAsync(output_ptr + t * batch * hidden, h_t_ptr,
                          batch * hidden * sizeof(float),
                          hipMemcpyDeviceToDevice, stream);
        }

        // Copy final state
        hipMemcpyAsync(h_n.data<float>(), h_t_ptr,
                      batch * hidden * sizeof(float),
                      hipMemcpyDeviceToDevice, stream);
    } else if (input.dtype() == DType::Float64) {
        RocBLASHandleGuard handle_guard2;
        rocblas_handle handle = handle_guard2.get();
        handle_guard2.set_stream(stream);

        const double alpha = 1.0;
        const double beta_zero = 0.0;
        const double beta_one = 1.0;

        const double* W_ih_ptr = W_ih.data<double>();
        const double* W_hh_ptr = W_hh.data<double>();
        const double* input_ptr = input.data<double>();
        const double* bias_ptr = bias.numel() > 0 ? bias.data<double>() : nullptr;
        double* gates_ih_ptr = gates_ih.data<double>();
        double* gates_hh_ptr = gates_hh.data<double>();
        double* rz_gates_ptr = rz_gates.data<double>();
        double* h_t_ptr = h_t.data<double>();
        double* output_ptr = output.data<double>();

        // Process each timestep
        for (int64_t t = 0; t < seq_len; ++t) {
            const double* x_t = input_ptr + t * batch * input_size;

            // Compute input contribution: gates_ih = x_t @ W_ih^T
            rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                         3 * hidden, batch, input_size,
                         &alpha,
                         W_ih_ptr, input_size,
                         x_t, input_size,
                         &beta_zero,
                         gates_ih_ptr, 3 * hidden);

            // Compute hidden contribution: gates_hh = h_t @ W_hh^T
            rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                         3 * hidden, batch, hidden,
                         &alpha,
                         W_hh_ptr, hidden,
                         h_t_ptr, hidden,
                         &beta_zero,
                         gates_hh_ptr, 3 * hidden);

            // Add bias if present
            if (bias_ptr != nullptr) {
                int64_t total = batch * 3 * hidden;
                int num_blocks = get_num_blocks(total);
                hipLaunchKernelGGL(add_bias_kernel<double>,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    bias_ptr, gates_ih_ptr, batch, 3 * hidden);
            }

            // Combine rz gates and apply GRU step
            int64_t total = batch * hidden;
            int num_blocks = get_num_blocks(total);

            hipLaunchKernelGGL(combine_rz_gates_kernel<double>,
                dim3(get_num_blocks(batch * 2 * hidden)), dim3(BLOCK_SIZE), 0, stream,
                gates_ih_ptr, gates_hh_ptr, rz_gates_ptr, batch, hidden);

            hipLaunchKernelGGL(gru_sequence_step_kernel<double>,
                              dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                              rz_gates_ptr,
                              gates_ih_ptr + 2 * hidden,
                              gates_hh_ptr + 2 * hidden,
                              h_t_ptr,
                              h_t_ptr,
                              batch, hidden);

            // Copy h_t to output[t]
            hipMemcpyAsync(output_ptr + t * batch * hidden, h_t_ptr,
                          batch * hidden * sizeof(double),
                          hipMemcpyDeviceToDevice, stream);
        }

        // Copy final state
        hipMemcpyAsync(h_n.data<double>(), h_t_ptr,
                      batch * hidden * sizeof(double),
                      hipMemcpyDeviceToDevice, stream);
    } else if (input.dtype() == DType::Float16) {
        RocBLASHandleGuard handle_guard3;
        rocblas_handle handle = handle_guard3.get();
        handle_guard3.set_stream(stream);

        const rocblas_half alpha = rocblas_half(1.0f);
        const rocblas_half beta_zero = rocblas_half(0.0f);

        const __half* W_ih_ptr = reinterpret_cast<const __half*>(W_ih.data<Float16>());
        const __half* W_hh_ptr = reinterpret_cast<const __half*>(W_hh.data<Float16>());
        const __half* input_ptr = reinterpret_cast<const __half*>(input.data<Float16>());
        const __half* bias_ptr = bias.numel() > 0 ? reinterpret_cast<const __half*>(bias.data<Float16>()) : nullptr;
        __half* gates_ih_ptr = reinterpret_cast<__half*>(gates_ih.data<Float16>());
        __half* gates_hh_ptr = reinterpret_cast<__half*>(gates_hh.data<Float16>());
        __half* rz_gates_ptr = reinterpret_cast<__half*>(rz_gates.data<Float16>());
        __half* h_t_ptr = reinterpret_cast<__half*>(h_t.data<Float16>());
        __half* output_ptr = reinterpret_cast<__half*>(output.data<Float16>());

        // Process each timestep
        for (int64_t t = 0; t < seq_len; ++t) {
            const __half* x_t = input_ptr + t * batch * input_size;

            // Compute input contribution: gates_ih = x_t @ W_ih^T
            rocblas_hgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                         3 * hidden, batch, input_size,
                         &alpha,
                         reinterpret_cast<const rocblas_half*>(W_ih_ptr), input_size,
                         reinterpret_cast<const rocblas_half*>(x_t), input_size,
                         &beta_zero,
                         reinterpret_cast<rocblas_half*>(gates_ih_ptr), 3 * hidden);

            // Compute hidden contribution: gates_hh = h_t @ W_hh^T
            rocblas_hgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                         3 * hidden, batch, hidden,
                         &alpha,
                         reinterpret_cast<const rocblas_half*>(W_hh_ptr), hidden,
                         reinterpret_cast<const rocblas_half*>(h_t_ptr), hidden,
                         &beta_zero,
                         reinterpret_cast<rocblas_half*>(gates_hh_ptr), 3 * hidden);

            // Add bias if present
            if (bias_ptr != nullptr) {
                int64_t total = batch * 3 * hidden;
                int num_blocks = get_num_blocks(total);
                hipLaunchKernelGGL(add_bias_kernel_fp16,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    bias_ptr, gates_ih_ptr, batch, 3 * hidden);
            }

            // Combine rz gates and apply GRU step
            int64_t total = batch * hidden;
            int num_blocks = get_num_blocks(total);

            hipLaunchKernelGGL(combine_rz_gates_kernel_fp16,
                dim3(get_num_blocks(batch * 2 * hidden)), dim3(BLOCK_SIZE), 0, stream,
                gates_ih_ptr, gates_hh_ptr, rz_gates_ptr, batch, hidden);

            hipLaunchKernelGGL(gru_sequence_step_kernel_fp16,
                              dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                              rz_gates_ptr,
                              gates_ih_ptr + 2 * hidden,
                              gates_hh_ptr + 2 * hidden,
                              h_t_ptr,
                              h_t_ptr,
                              batch, hidden);

            // Copy h_t to output[t]
            hipMemcpyAsync(output_ptr + t * batch * hidden, h_t_ptr,
                          batch * hidden * sizeof(__half),
                          hipMemcpyDeviceToDevice, stream);
        }

        // Copy final state
        hipMemcpyAsync(reinterpret_cast<__half*>(h_n.data<Float16>()), h_t_ptr,
                      batch * hidden * sizeof(__half),
                      hipMemcpyDeviceToDevice, stream);
    } else {
        throw std::runtime_error("GRU forward only supports Float32, Float64, Float16, and BFloat16");
    }

    HIP_CHECK(hipGetLastError());

    return {output, h_n};
}

// ============================================================================
// Multi-layer GRU Forward
// ============================================================================

auto gru_multi_layer_forward_kernel(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0,    // (num_layers, batch, hidden)
    hipStream_t stream) -> std::vector<Tensor> {

    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t hidden = h0.shape()[2];

    Tensor h_n({num_layers, batch, hidden}, input.dtype(), input.device());
    Tensor layer_input = input;
    size_t layer_bytes = batch * hidden * dtype_size(input.dtype());

    for (int64_t l = 0; l < num_layers; ++l) {
        // Extract h0 for this layer
        Tensor h_l({batch, hidden}, input.dtype(), input.device());
        hipMemcpyAsync(h_l.data<void>(),
                       static_cast<const char*>(h0.data<void>()) + l * layer_bytes,
                       layer_bytes, hipMemcpyDeviceToDevice, stream);

        auto result = gru_forward_kernel(
            layer_input, W_ih_list[l], W_hh_list[l],
            bias_list[l], h_l, stream);

        layer_input = result[0];

        hipMemcpyAsync(
            static_cast<char*>(h_n.data<void>()) + l * layer_bytes,
            result[1].data<void>(), layer_bytes,
            hipMemcpyDeviceToDevice, stream);
    }

    HIP_CHECK(hipGetLastError());

    return {layer_input, h_n};
}

} // namespace rocm
} // namespace tenzor
