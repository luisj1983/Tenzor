#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cmath>
#include <cstdio>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include <string>

namespace tenzor {
namespace rocm {

// ============================================================================
// HIP Helper Functions
// ============================================================================

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error(std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + ": " + hipGetErrorString(err)); \
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
        rocblas_status status = rocblas_set_stream(handle_, stream);
        if (status != rocblas_status_success) {
            throw std::runtime_error("rocblas_set_stream failed: " + std::to_string(status));
        }
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
 * @param gates Pre-computed gate values: (batch, 4 * hidden_size)
 *              [input_gate | forget_gate | cell_gate | output_gate]
 * @param c_prev Cell state at t-1: (batch, hidden_size)
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

// Float16 LSTM cell forward kernel
__global__ void lstm_cell_forward_fused_fp16(
    const __half* __restrict__ gates,
    const __half* __restrict__ c_prev,
    __half* __restrict__ h_out,
    __half* __restrict__ c_out,
    int64_t batch_size,
    int64_t hidden_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * hidden_size;

    if (idx < total_elements) {
        int64_t batch_idx = idx / hidden_size;
        int64_t hidden_idx = idx % hidden_size;

        int64_t gate_offset = batch_idx * (4 * hidden_size) + hidden_idx;
        int64_t i_offset = gate_offset;
        int64_t f_offset = gate_offset + hidden_size;
        int64_t g_offset = gate_offset + 2 * hidden_size;
        int64_t o_offset = gate_offset + 3 * hidden_size;

        // Load and convert to float for computation
        float i_gate = __half2float(gates[i_offset]);
        float f_gate = __half2float(gates[f_offset]);
        float g_gate = __half2float(gates[g_offset]);
        float o_gate = __half2float(gates[o_offset]);

        // Apply activations
        float i_t = 1.0f / (1.0f + expf(-i_gate));
        float f_t = 1.0f / (1.0f + expf(-f_gate));
        float o_t = 1.0f / (1.0f + expf(-o_gate));
        float g_t = tanhf(g_gate);

        float c_prev_val = __half2float(c_prev[idx]);

        // Update cell state
        float c_t = f_t * c_prev_val + i_t * g_t;

        // Update hidden state
        float h_t = o_t * tanhf(c_t);

        // Store outputs
        c_out[idx] = __float2half(c_t);
        h_out[idx] = __float2half(h_t);
    }
}

// Float16 LSTM cell backward kernel
__global__ void lstm_cell_backward_fused_fp16(
    const __half* __restrict__ grad_h,
    const __half* __restrict__ grad_c,
    const __half* __restrict__ gates,
    const __half* __restrict__ c_prev,
    const __half* __restrict__ c_out,
    __half* __restrict__ grad_gates,
    __half* __restrict__ grad_c_prev,
    int64_t batch_size,
    int64_t hidden_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * hidden_size;

    if (idx < total_elements) {
        int64_t batch_idx = idx / hidden_size;
        int64_t hidden_idx = idx % hidden_size;

        int64_t gate_offset = batch_idx * (4 * hidden_size) + hidden_idx;
        int64_t i_offset = gate_offset;
        int64_t f_offset = gate_offset + hidden_size;
        int64_t g_offset = gate_offset + 2 * hidden_size;
        int64_t o_offset = gate_offset + 3 * hidden_size;

        // Load and convert to float
        float i_gate = __half2float(gates[i_offset]);
        float f_gate = __half2float(gates[f_offset]);
        float g_gate = __half2float(gates[g_offset]);
        float o_gate = __half2float(gates[o_offset]);

        float i_t = 1.0f / (1.0f + expf(-i_gate));
        float f_t = 1.0f / (1.0f + expf(-f_gate));
        float g_t = tanhf(g_gate);
        float o_t = 1.0f / (1.0f + expf(-o_gate));

        float c_prev_val = __half2float(c_prev[idx]);
        float c_t = __half2float(c_out[idx]);
        float tanh_c_t = tanhf(c_t);

        float dh = __half2float(grad_h[idx]);
        float dc = __half2float(grad_c[idx]);

        // Gradients
        dc += dh * o_t * (1.0f - tanh_c_t * tanh_c_t);
        float do_t = dh * tanh_c_t;

        float df_t = dc * c_prev_val;
        float di_t = dc * g_t;
        float dg_t = dc * i_t;
        float dc_prev = dc * f_t;

        float di_gate = di_t * i_t * (1.0f - i_t);
        float df_gate = df_t * f_t * (1.0f - f_t);
        float do_gate = do_t * o_t * (1.0f - o_t);
        float dg_gate = dg_t * (1.0f - g_t * g_t);

        // Store gradients
        grad_gates[i_offset] = __float2half(di_gate);
        grad_gates[f_offset] = __float2half(df_gate);
        grad_gates[g_offset] = __float2half(dg_gate);
        grad_gates[o_offset] = __float2half(do_gate);
        grad_c_prev[idx] = __float2half(dc_prev);
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

        hipLaunchKernelGGL(lstm_cell_forward_fused<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          gates, c_prev, h_out, c_out, batch_size, hidden_size);

        HIP_CHECK(hipGetLastError());
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

        hipLaunchKernelGGL(lstm_cell_forward_fused<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          gates, c_prev, h_out, c_out, batch_size, hidden_size);

        HIP_CHECK(hipGetLastError());
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

        hipLaunchKernelGGL(lstm_cell_backward_fused<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_h, grad_c, gates, c_prev, c_out,
                          grad_gates, grad_c_prev, batch_size, hidden_size);

        HIP_CHECK(hipGetLastError());
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

        hipLaunchKernelGGL(lstm_cell_backward_fused<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_h, grad_c, gates, c_prev, c_out,
                          grad_gates, grad_c_prev, batch_size, hidden_size);

        HIP_CHECK(hipGetLastError());
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
    hipStream_t stream) -> std::pair<Tensor, Tensor> {

    std::vector<int64_t> output_shape = {batch_size, hidden_size};
    Tensor h_out(output_shape, gates.dtype(), gates.device());
    Tensor c_out(output_shape, gates.dtype(), gates.device());

    int64_t total = batch_size * hidden_size;
    int num_blocks = get_num_blocks(total);

    if (gates.dtype() == DType::Float32) {
        hipLaunchKernelGGL(lstm_cell_forward_fused<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          gates.data<float>(),
                          c_prev.data<float>(),
                          h_out.data<float>(),
                          c_out.data<float>(),
                          batch_size,
                          hidden_size);
    } else if (gates.dtype() == DType::Float64) {
        hipLaunchKernelGGL(lstm_cell_forward_fused<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          gates.data<double>(),
                          c_prev.data<double>(),
                          h_out.data<double>(),
                          c_out.data<double>(),
                          batch_size,
                          hidden_size);
    } else if (gates.dtype() == DType::Float16) {
        hipLaunchKernelGGL(lstm_cell_forward_fused_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(gates.data<Float16>()),
                          reinterpret_cast<const __half*>(c_prev.data<Float16>()),
                          reinterpret_cast<__half*>(h_out.data<Float16>()),
                          reinterpret_cast<__half*>(c_out.data<Float16>()),
                          batch_size,
                          hidden_size);
    } else if (gates.dtype() == DType::BFloat16) {
        auto gates_f32 = gates.to(DType::Float32);
        auto c_prev_f32 = c_prev.to(DType::Float32);
        auto [h_out_f32, c_out_f32] = lstm_cell_forward_kernel(gates_f32, c_prev_f32,
                                                                 batch_size, hidden_size, stream);
        return {h_out_f32.to(DType::BFloat16), c_out_f32.to(DType::BFloat16)};
    } else {
        throw std::runtime_error("LSTM only supports Float32, Float64, Float16, and BFloat16");
    }

    HIP_CHECK(hipGetLastError());

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
    hipStream_t stream) -> std::pair<Tensor, Tensor> {

    std::vector<int64_t> gate_shape = {batch_size, 4 * hidden_size};
    std::vector<int64_t> state_shape = {batch_size, hidden_size};

    Tensor grad_gates(gate_shape, gates.dtype(), gates.device());
    Tensor grad_c_prev(state_shape, gates.dtype(), gates.device());

    int64_t total = batch_size * hidden_size;
    int num_blocks = get_num_blocks(total);

    if (gates.dtype() == DType::Float32) {
        hipLaunchKernelGGL(lstm_cell_backward_fused<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_h.data<float>(),
                          grad_c.data<float>(),
                          gates.data<float>(),
                          c_prev.data<float>(),
                          c_out.data<float>(),
                          grad_gates.data<float>(),
                          grad_c_prev.data<float>(),
                          batch_size,
                          hidden_size);
    } else if (gates.dtype() == DType::Float64) {
        hipLaunchKernelGGL(lstm_cell_backward_fused<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_h.data<double>(),
                          grad_c.data<double>(),
                          gates.data<double>(),
                          c_prev.data<double>(),
                          c_out.data<double>(),
                          grad_gates.data<double>(),
                          grad_c_prev.data<double>(),
                          batch_size,
                          hidden_size);
    } else if (gates.dtype() == DType::Float16) {
        hipLaunchKernelGGL(lstm_cell_backward_fused_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_h.data<Float16>()),
                          reinterpret_cast<const __half*>(grad_c.data<Float16>()),
                          reinterpret_cast<const __half*>(gates.data<Float16>()),
                          reinterpret_cast<const __half*>(c_prev.data<Float16>()),
                          reinterpret_cast<const __half*>(c_out.data<Float16>()),
                          reinterpret_cast<__half*>(grad_gates.data<Float16>()),
                          reinterpret_cast<__half*>(grad_c_prev.data<Float16>()),
                          batch_size,
                          hidden_size);
    } else if (gates.dtype() == DType::BFloat16) {
        auto grad_h_f32 = grad_h.to(DType::Float32);
        auto grad_c_f32 = grad_c.to(DType::Float32);
        auto gates_f32 = gates.to(DType::Float32);
        auto c_prev_f32 = c_prev.to(DType::Float32);
        auto c_out_f32 = c_out.to(DType::Float32);
        auto [gg_f32, gcp_f32] = lstm_cell_backward_kernel(grad_h_f32, grad_c_f32, gates_f32,
                                                             c_prev_f32, c_out_f32,
                                                             batch_size, hidden_size, stream);
        return {gg_f32.to(DType::BFloat16), gcp_f32.to(DType::BFloat16)};
    } else {
        throw std::runtime_error("LSTM backward only supports Float32, Float64, Float16, and BFloat16");
    }

    HIP_CHECK(hipGetLastError());

    return {grad_gates, grad_c_prev};
}

// ============================================================================
// Full Sequence LSTM Forward
// ============================================================================

/**
 * @brief Full sequence LSTM forward pass
 *
 * Processes entire sequence using rocBLAS for efficient matrix operations.
 *
 * @param input Input sequence (seq_len, batch, input_size)
 * @param W_ih Input-to-hidden weights (4*hidden, input_size)
 * @param W_hh Hidden-to-hidden weights (4*hidden, hidden)
 * @param bias Combined bias (4*hidden) or empty tensor
 * @param h0 Initial hidden state (batch, hidden)
 * @param c0 Initial cell state (batch, hidden)
 * @param stream HIP stream for async execution
 * @return vector of [output, h_n, c_n]
 */
auto lstm_forward_kernel(
    const Tensor& input,
    const Tensor& W_ih,
    const Tensor& W_hh,
    const Tensor& bias,
    const Tensor& h0,
    const Tensor& c0,
    hipStream_t stream) -> std::vector<Tensor> {

    // Get dimensions
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden = h0.shape()[1];

    // Allocate output tensors
    Tensor output({seq_len, batch, hidden}, input.dtype(), input.device());
    Tensor h_n({batch, hidden}, input.dtype(), input.device());
    Tensor c_n({batch, hidden}, input.dtype(), input.device());

    // Temporary tensors for gate computations
    // gates shape: (batch, 4 * hidden)
    Tensor gates({batch, 4 * hidden}, input.dtype(), input.device());

    // Copy initial states to current states
    Tensor h_t({batch, hidden}, input.dtype(), input.device());
    Tensor c_t({batch, hidden}, input.dtype(), input.device());

    // Copy h0 and c0 to h_t and c_t
    HIP_CHECK(hipMemcpyAsync(h_t.data_ptr(), h0.data_ptr(),
                   batch * hidden * dtype_size(input.dtype()),
                   hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(c_t.data_ptr(), c0.data_ptr(),
                   batch * hidden * dtype_size(input.dtype()),
                   hipMemcpyDeviceToDevice, stream));

    if (input.dtype() == DType::Float32) {
        // Get rocBLAS handle (RAII ensures cleanup on exceptions)
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
        float* gates_ptr = gates.data<float>();
        float* h_t_ptr = h_t.data<float>();
        float* c_t_ptr = c_t.data<float>();
        float* output_ptr = output.data<float>();

        // Process each timestep
        for (int64_t t = 0; t < seq_len; ++t) {
            const float* x_t = input_ptr + t * batch * input_size;

            // Compute input contribution: gates = x_t @ W_ih^T
            // x_t: (batch, input_size), W_ih: (4*hidden, input_size)
            // Result: (batch, 4*hidden)
            rocblas_sgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                         4 * hidden, batch, input_size,
                         &alpha,
                         W_ih_ptr, input_size,
                         x_t, input_size,
                         &beta_zero,
                         gates_ptr, 4 * hidden);

            // Add hidden contribution: gates += h_t @ W_hh^T
            rocblas_sgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                         4 * hidden, batch, hidden,
                         &alpha,
                         W_hh_ptr, hidden,
                         h_t_ptr, hidden,
                         &beta_one,
                         gates_ptr, 4 * hidden);

            // Add bias if present
            if (bias_ptr != nullptr) {
                int64_t total = batch * 4 * hidden;
                int num_blocks = get_num_blocks(total);
                hipLaunchKernelGGL(add_bias_kernel<float>,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    bias_ptr, gates_ptr, batch, 4 * hidden);
            }

            // Apply LSTM cell operations using fused kernel
            hipLaunchKernelGGL(lstm_cell_forward_fused<float>,
                              dim3(get_num_blocks(batch * hidden)), dim3(BLOCK_SIZE), 0, stream,
                              gates_ptr, c_t_ptr, h_t_ptr, c_t_ptr,
                              batch, hidden);

            // Copy h_t to output[t]
            HIP_CHECK(hipMemcpyAsync(output_ptr + t * batch * hidden, h_t_ptr,
                          batch * hidden * sizeof(float),
                          hipMemcpyDeviceToDevice, stream));
        }

        // Copy final states
        HIP_CHECK(hipMemcpyAsync(h_n.data<float>(), h_t_ptr,
                      batch * hidden * sizeof(float),
                      hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(c_n.data<float>(), c_t_ptr,
                      batch * hidden * sizeof(float),
                      hipMemcpyDeviceToDevice, stream));
    } else if (input.dtype() == DType::Float64) {
        // Get rocBLAS handle (RAII ensures cleanup on exceptions)
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
        double* gates_ptr = gates.data<double>();
        double* h_t_ptr = h_t.data<double>();
        double* c_t_ptr = c_t.data<double>();
        double* output_ptr = output.data<double>();

        // Process each timestep
        for (int64_t t = 0; t < seq_len; ++t) {
            const double* x_t = input_ptr + t * batch * input_size;

            // Compute input contribution: gates = x_t @ W_ih^T
            rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                         4 * hidden, batch, input_size,
                         &alpha,
                         W_ih_ptr, input_size,
                         x_t, input_size,
                         &beta_zero,
                         gates_ptr, 4 * hidden);

            // Add hidden contribution: gates += h_t @ W_hh^T
            rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none,
                         4 * hidden, batch, hidden,
                         &alpha,
                         W_hh_ptr, hidden,
                         h_t_ptr, hidden,
                         &beta_one,
                         gates_ptr, 4 * hidden);

            // Add bias if present
            if (bias_ptr != nullptr) {
                int64_t total = batch * 4 * hidden;
                int num_blocks = get_num_blocks(total);
                hipLaunchKernelGGL(add_bias_kernel<double>,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    bias_ptr, gates_ptr, batch, 4 * hidden);
            }

            // Apply LSTM cell operations using fused kernel
            hipLaunchKernelGGL(lstm_cell_forward_fused<double>,
                              dim3(get_num_blocks(batch * hidden)), dim3(BLOCK_SIZE), 0, stream,
                              gates_ptr, c_t_ptr, h_t_ptr, c_t_ptr,
                              batch, hidden);

            // Copy h_t to output[t]
            HIP_CHECK(hipMemcpyAsync(output_ptr + t * batch * hidden, h_t_ptr,
                          batch * hidden * sizeof(double),
                          hipMemcpyDeviceToDevice, stream));
        }

        // Copy final states
        HIP_CHECK(hipMemcpyAsync(h_n.data<double>(), h_t_ptr,
                      batch * hidden * sizeof(double),
                      hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(c_n.data<double>(), c_t_ptr,
                      batch * hidden * sizeof(double),
                      hipMemcpyDeviceToDevice, stream));
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto W_ih_f32 = W_ih.to(DType::Float32);
        auto W_hh_f32 = W_hh.to(DType::Float32);
        auto bias_f32 = bias.numel() > 0 ? bias.to(DType::Float32) : bias;
        auto h0_f32 = h0.to(DType::Float32);
        auto c0_f32 = c0.to(DType::Float32);
        auto results = lstm_forward_kernel(input_f32, W_ih_f32, W_hh_f32, bias_f32,
                                            h0_f32, c0_f32, stream);
        for (auto& t : results) t = t.to(DType::BFloat16);
        return results;
    } else {
        throw std::runtime_error("LSTM forward only supports Float32, Float64, and BFloat16");
    }

    HIP_CHECK(hipGetLastError());

    return {output, h_n, c_n};
}

// ============================================================================
// Multi-layer LSTM Forward
// ============================================================================

auto lstm_multi_layer_forward_kernel(
    const Tensor& input,
    const std::vector<Tensor>& W_ih_list,
    const std::vector<Tensor>& W_hh_list,
    const std::vector<Tensor>& bias_list,
    const Tensor& h0,    // (num_layers, batch, hidden)
    const Tensor& c0,    // (num_layers, batch, hidden)
    hipStream_t stream) -> std::vector<Tensor> {

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
        Tensor h_l({batch, hidden}, input.dtype(), input.device());
        Tensor c_l({batch, hidden}, input.dtype(), input.device());

        HIP_CHECK(hipMemcpyAsync(h_l.data_ptr(),
                       static_cast<const char*>(h0.data_ptr()) + l * layer_bytes,
                       layer_bytes, hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(c_l.data_ptr(),
                       static_cast<const char*>(c0.data_ptr()) + l * layer_bytes,
                       layer_bytes, hipMemcpyDeviceToDevice, stream));

        // Split bias into bias_ih and bias_hh if present
        Tensor bias_combined = bias_list[l];
        Tensor effective_bias;
        if (bias_combined.numel() > 0) {
            // Combined bias: first half is bias_ih, second half is bias_hh
            // For simplicity, pass the full combined bias to lstm_forward_kernel
            effective_bias = bias_combined;
        }

        auto result = lstm_forward_kernel(
            layer_input, W_ih_list[l], W_hh_list[l],
            effective_bias, h_l, c_l, stream);

        layer_input = result[0];  // output becomes input for next layer

        // Copy final h, c to h_n[l], c_n[l]
        HIP_CHECK(hipMemcpyAsync(
            static_cast<char*>(h_n.data_ptr()) + l * layer_bytes,
            result[1].data_ptr(), layer_bytes,
            hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(
            static_cast<char*>(c_n.data_ptr()) + l * layer_bytes,
            result[2].data_ptr(), layer_bytes,
            hipMemcpyDeviceToDevice, stream));
    }

    HIP_CHECK(hipGetLastError());

    return {layer_input, h_n, c_n};
}

// ============================================================================
// Bidirectional LSTM Forward
// ============================================================================

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

auto bilstm_forward_kernel(
    const Tensor& input,
    const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
    const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
    const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
    const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd,
    const Tensor& h0,    // (2, batch, hidden)
    const Tensor& c0,    // (2, batch, hidden)
    hipStream_t stream) -> std::vector<Tensor> {

    // BFloat16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto wif_f32 = W_ih_fwd.to(DType::Float32);
        auto whf_f32 = W_hh_fwd.to(DType::Float32);
        auto bif_f32 = bias_ih_fwd.numel() > 0 ? bias_ih_fwd.to(DType::Float32) : bias_ih_fwd;
        auto bhf_f32 = bias_hh_fwd.numel() > 0 ? bias_hh_fwd.to(DType::Float32) : bias_hh_fwd;
        auto wib_f32 = W_ih_bwd.to(DType::Float32);
        auto whb_f32 = W_hh_bwd.to(DType::Float32);
        auto bib_f32 = bias_ih_bwd.numel() > 0 ? bias_ih_bwd.to(DType::Float32) : bias_ih_bwd;
        auto bhb_f32 = bias_hh_bwd.numel() > 0 ? bias_hh_bwd.to(DType::Float32) : bias_hh_bwd;
        auto h0_f32 = h0.to(DType::Float32);
        auto c0_f32 = c0.to(DType::Float32);
        auto results = bilstm_forward_kernel(input_f32, wif_f32, whf_f32, bif_f32, bhf_f32,
                                              wib_f32, whb_f32, bib_f32, bhb_f32,
                                              h0_f32, c0_f32, stream);
        for (auto& t : results) t = t.to(DType::BFloat16);
        return results;
    }

    auto shape = input.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];
    int64_t input_size = shape[2];
    int64_t hidden = h0.shape()[2];

    size_t state_bytes = batch * hidden * dtype_size(input.dtype());

    // Extract forward direction states
    Tensor h0_fwd({batch, hidden}, input.dtype(), input.device());
    Tensor c0_fwd({batch, hidden}, input.dtype(), input.device());
    HIP_CHECK(hipMemcpyAsync(h0_fwd.data_ptr(), h0.data_ptr(), state_bytes, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(c0_fwd.data_ptr(), c0.data_ptr(), state_bytes, hipMemcpyDeviceToDevice, stream));

    // Combine biases for forward
    Tensor bias_fwd;
    if (bias_ih_fwd.numel() > 0 && bias_hh_fwd.numel() > 0) {
        // For the ROCm lstm_forward_kernel, bias is a single combined bias
        // We'll create a temporary combined bias by adding them
        bias_fwd = Tensor({bias_ih_fwd.numel()}, input.dtype(), input.device());
        // Simple addition kernel
        int64_t total_bias = bias_ih_fwd.numel();
        int block = 256;
        int grid = (total_bias + block - 1) / block;
        if (input.dtype() == DType::Float32) {
            hipLaunchKernelGGL(add_bias_kernel<float>,
                dim3(grid), dim3(block), 0, stream,
                bias_ih_fwd.data<float>(), bias_fwd.data<float>(), 1, total_bias);
            hipLaunchKernelGGL(add_bias_kernel<float>,
                dim3(grid), dim3(block), 0, stream,
                bias_hh_fwd.data<float>(), bias_fwd.data<float>(), 1, total_bias);
        } else if (input.dtype() == DType::Float64) {
            hipLaunchKernelGGL(add_bias_kernel<double>,
                dim3(grid), dim3(block), 0, stream,
                bias_ih_fwd.data<double>(), bias_fwd.data<double>(), 1, total_bias);
            hipLaunchKernelGGL(add_bias_kernel<double>,
                dim3(grid), dim3(block), 0, stream,
                bias_hh_fwd.data<double>(), bias_fwd.data<double>(), 1, total_bias);
        }
    }

    auto fwd_result = lstm_forward_kernel(input, W_ih_fwd, W_hh_fwd, bias_fwd, h0_fwd, c0_fwd, stream);

    // Backward direction: reverse input, run LSTM, reverse output
    Tensor input_rev({seq_len, batch, input_size}, input.dtype(), input.device());
    int64_t total_input = seq_len * batch * input_size;
    int block = 256;
    int grid = (total_input + block - 1) / block;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(reverse_sequence_kernel<float>,
            dim3(grid), dim3(block), 0, stream,
            input.data<float>(), input_rev.data<float>(),
            seq_len, batch, input_size);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(reverse_sequence_kernel<double>,
            dim3(grid), dim3(block), 0, stream,
            input.data<double>(), input_rev.data<double>(),
            seq_len, batch, input_size);
    }

    Tensor h0_bwd({batch, hidden}, input.dtype(), input.device());
    Tensor c0_bwd({batch, hidden}, input.dtype(), input.device());
    HIP_CHECK(hipMemcpyAsync(h0_bwd.data_ptr(),
                   static_cast<const char*>(h0.data_ptr()) + state_bytes,
                   state_bytes, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(c0_bwd.data_ptr(),
                   static_cast<const char*>(c0.data_ptr()) + state_bytes,
                   state_bytes, hipMemcpyDeviceToDevice, stream));

    Tensor bias_bwd;
    if (bias_ih_bwd.numel() > 0 && bias_hh_bwd.numel() > 0) {
        bias_bwd = Tensor({bias_ih_bwd.numel()}, input.dtype(), input.device());
        int64_t total_bias = bias_ih_bwd.numel();
        int bgrid = (total_bias + block - 1) / block;
        if (input.dtype() == DType::Float32) {
            hipLaunchKernelGGL(add_bias_kernel<float>,
                dim3(bgrid), dim3(block), 0, stream,
                bias_ih_bwd.data<float>(), bias_bwd.data<float>(), 1, total_bias);
            hipLaunchKernelGGL(add_bias_kernel<float>,
                dim3(bgrid), dim3(block), 0, stream,
                bias_hh_bwd.data<float>(), bias_bwd.data<float>(), 1, total_bias);
        } else if (input.dtype() == DType::Float64) {
            hipLaunchKernelGGL(add_bias_kernel<double>,
                dim3(bgrid), dim3(block), 0, stream,
                bias_ih_bwd.data<double>(), bias_bwd.data<double>(), 1, total_bias);
            hipLaunchKernelGGL(add_bias_kernel<double>,
                dim3(bgrid), dim3(block), 0, stream,
                bias_hh_bwd.data<double>(), bias_bwd.data<double>(), 1, total_bias);
        }
    }

    auto bwd_result = lstm_forward_kernel(input_rev, W_ih_bwd, W_hh_bwd, bias_bwd, h0_bwd, c0_bwd, stream);

    // Reverse backward output
    Tensor bwd_output_rev({seq_len, batch, hidden}, input.dtype(), input.device());
    int64_t total_out = seq_len * batch * hidden;
    grid = (total_out + block - 1) / block;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(reverse_sequence_kernel<float>,
            dim3(grid), dim3(block), 0, stream,
            bwd_result[0].data<float>(), bwd_output_rev.data<float>(),
            seq_len, batch, hidden);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(reverse_sequence_kernel<double>,
            dim3(grid), dim3(block), 0, stream,
            bwd_result[0].data<double>(), bwd_output_rev.data<double>(),
            seq_len, batch, hidden);
    }

    // Concatenate forward and backward outputs: (seq_len, batch, 2*hidden)
    Tensor output({seq_len, batch, 2 * hidden}, input.dtype(), input.device());
    int64_t concat_total = seq_len * batch * hidden;
    int concat_grid = (concat_total + block - 1) / block;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(bilstm_concat_kernel<float>,
            dim3(concat_grid), dim3(block), 0, stream,
            fwd_result[0].data<float>(), bwd_output_rev.data<float>(),
            output.data<float>(), seq_len, batch, hidden);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(bilstm_concat_kernel<double>,
            dim3(concat_grid), dim3(block), 0, stream,
            fwd_result[0].data<double>(), bwd_output_rev.data<double>(),
            output.data<double>(), seq_len, batch, hidden);
    }

    // Stack h_n: (2, batch, hidden)
    Tensor h_n({2, batch, hidden}, input.dtype(), input.device());
    Tensor c_n({2, batch, hidden}, input.dtype(), input.device());

    HIP_CHECK(hipMemcpyAsync(h_n.data_ptr(), fwd_result[1].data_ptr(),
                   state_bytes, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(static_cast<char*>(h_n.data_ptr()) + state_bytes,
                   bwd_result[1].data_ptr(),
                   state_bytes, hipMemcpyDeviceToDevice, stream));

    HIP_CHECK(hipMemcpyAsync(c_n.data_ptr(), fwd_result[2].data_ptr(),
                   state_bytes, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(static_cast<char*>(c_n.data_ptr()) + state_bytes,
                   bwd_result[2].data_ptr(),
                   state_bytes, hipMemcpyDeviceToDevice, stream));

    HIP_CHECK(hipGetLastError());

    return {output, h_n, c_n};
}

} // namespace rocm
} // namespace tenzor
