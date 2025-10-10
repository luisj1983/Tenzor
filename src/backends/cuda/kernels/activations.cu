#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace cuda {

// ============================================================================
// CUDA Helper Functions
// ============================================================================

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Grid-stride loop helper
#define CUDA_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// Optimal block size for element-wise operations
constexpr int BLOCK_SIZE = 256;

// Calculate grid size for element-wise operations
inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return (n + block_size - 1) / block_size;
}

// ============================================================================
// ReLU Activation
// ============================================================================

// Forward: max(0, x)
template<typename T>
__global__ void relu_forward_kernel(const T* input, T* output, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = input[idx] > T(0) ? input[idx] : T(0);
    }
}

// Backward: grad_out * (x > 0)
template<typename T>
__global__ void relu_backward_kernel(const T* grad_output, const T* input,
                                     T* grad_input, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        grad_input[idx] = grad_output[idx] * (input[idx] > T(0) ? T(1) : T(0));
    }
}

// Host functions
extern "C" {
    void relu_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        relu_forward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void relu_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        relu_forward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void relu_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        relu_backward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void relu_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        relu_backward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Sigmoid Activation
// ============================================================================

// Forward: 1 / (1 + exp(-x))
// Numerically stable version
template<typename T>
__device__ __forceinline__ T sigmoid_stable(T x) {
    if (x >= T(0)) {
        return T(1) / (T(1) + exp(-x));
    } else {
        T exp_x = exp(x);
        return exp_x / (T(1) + exp_x);
    }
}

template<typename T>
__global__ void sigmoid_forward_kernel(const T* input, T* output, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = sigmoid_stable(input[idx]);
    }
}

// Backward: grad_out * sigmoid(x) * (1 - sigmoid(x))
template<typename T>
__global__ void sigmoid_backward_kernel(const T* grad_output, const T* input,
                                       T* grad_input, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        T sigmoid_x = sigmoid_stable(input[idx]);
        grad_input[idx] = grad_output[idx] * sigmoid_x * (T(1) - sigmoid_x);
    }
}

// Host functions
extern "C" {
    void sigmoid_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        sigmoid_forward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void sigmoid_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        sigmoid_forward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void sigmoid_backward_float(const float* grad_output, const float* input,
                               float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        sigmoid_backward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void sigmoid_backward_double(const double* grad_output, const double* input,
                                double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        sigmoid_backward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Tanh Activation
// ============================================================================

// Forward: tanh(x)
template<typename T>
__global__ void tanh_forward_kernel(const T* input, T* output, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = tanh(input[idx]);
    }
}

// Backward: grad_out * (1 - tanh(x)^2)
template<typename T>
__global__ void tanh_backward_kernel(const T* grad_output, const T* input,
                                    T* grad_input, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        T tanh_x = tanh(input[idx]);
        grad_input[idx] = grad_output[idx] * (T(1) - tanh_x * tanh_x);
    }
}

// Host functions
extern "C" {
    void tanh_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        tanh_forward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void tanh_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        tanh_forward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void tanh_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        tanh_backward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void tanh_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        tanh_backward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Leaky ReLU Activation
// ============================================================================

// Forward: x if x > 0 else alpha * x
template<typename T>
__global__ void leaky_relu_forward_kernel(const T* input, T* output,
                                         int64_t n, T alpha) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = input[idx] > T(0) ? input[idx] : alpha * input[idx];
    }
}

// Backward: grad_out * (1 if x > 0 else alpha)
template<typename T>
__global__ void leaky_relu_backward_kernel(const T* grad_output, const T* input,
                                          T* grad_input, int64_t n, T alpha) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        grad_input[idx] = grad_output[idx] * (input[idx] > T(0) ? T(1) : alpha);
    }
}

// Host functions
extern "C" {
    void leaky_relu_forward_float(const float* input, float* output,
                                 int64_t n, float alpha) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_forward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(
            input, output, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void leaky_relu_forward_double(const double* input, double* output,
                                  int64_t n, double alpha) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_forward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(
            input, output, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void leaky_relu_backward_float(const float* grad_output, const float* input,
                                  float* grad_input, int64_t n, float alpha) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_backward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void leaky_relu_backward_double(const double* grad_output, const double* input,
                                   double* grad_input, int64_t n, double alpha) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_backward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Softmax Activation
// ============================================================================

// Shared memory size for reductions
constexpr int SOFTMAX_BLOCK_SIZE = 256;

// Warp-level reduction using shuffle instructions
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template<typename T>
__device__ T block_reduce_max(T val, T* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

template<typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// Softmax forward: exp(x_i - max) / sum(exp(x_j - max))
// Each block handles one row
template<typename T>
__global__ void softmax_forward_kernel(const T* input, T* output,
                                      int64_t batch_size, int64_t dim_size) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* input_row = input + row * dim_size;
    T* output_row = output + row * dim_size;

    // Step 1: Find max value for numerical stability
    T max_val = -FLT_MAX;
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        max_val = max(max_val, input_row[i]);
    }
    max_val = block_reduce_max(max_val, shared);
    __syncthreads();

    // Broadcast max to all threads
    if (threadIdx.x == 0) {
        shared[0] = max_val;
    }
    __syncthreads();
    max_val = shared[0];

    // Step 2: Compute exp(x - max) and sum
    T sum_exp = T(0);
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        T exp_val = exp(input_row[i] - max_val);
        output_row[i] = exp_val;
        sum_exp += exp_val;
    }
    sum_exp = block_reduce_sum(sum_exp, shared);
    __syncthreads();

    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        shared[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = shared[0];

    // Step 3: Normalize
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        output_row[i] /= sum_exp;
    }
}

// Softmax backward: softmax[i] * (grad_output[i] - sum(grad_output * softmax))
template<typename T>
__global__ void softmax_backward_kernel(const T* grad_output, const T* output,
                                       T* grad_input,
                                       int64_t batch_size, int64_t dim_size) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* grad_out_row = grad_output + row * dim_size;
    const T* out_row = output + row * dim_size;
    T* grad_in_row = grad_input + row * dim_size;

    // Compute sum(grad_output * softmax)
    T sum = T(0);
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum += grad_out_row[i] * out_row[i];
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();

    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        shared[0] = sum;
    }
    __syncthreads();
    sum = shared[0];

    // Compute gradient: softmax[i] * (grad_output[i] - sum)
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        grad_in_row[i] = out_row[i] * (grad_out_row[i] - sum);
    }
}

// Host functions
extern "C" {
    void softmax_forward_float(const float* input, float* output,
                              int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        softmax_forward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size>>>(
            input, output, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void softmax_forward_double(const double* input, double* output,
                               int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        softmax_forward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size>>>(
            input, output, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void softmax_backward_float(const float* grad_output, const float* output,
                               float* grad_input,
                               int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        softmax_backward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size>>>(
            grad_output, output, grad_input, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void softmax_backward_double(const double* grad_output, const double* output,
                                double* grad_input,
                                int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        softmax_backward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size>>>(
            grad_output, output, grad_input, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// LogSoftmax Activation
// ============================================================================

// LogSoftmax forward: x - max - log(sum(exp(x - max)))
// More numerically stable than log(softmax(x))
template<typename T>
__global__ void log_softmax_forward_kernel(const T* input, T* output,
                                          int64_t batch_size, int64_t dim_size) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* input_row = input + row * dim_size;
    T* output_row = output + row * dim_size;

    // Step 1: Find max value for numerical stability
    T max_val = -FLT_MAX;
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        max_val = max(max_val, input_row[i]);
    }
    max_val = block_reduce_max(max_val, shared);
    __syncthreads();

    // Broadcast max to all threads
    if (threadIdx.x == 0) {
        shared[0] = max_val;
    }
    __syncthreads();
    max_val = shared[0];

    // Step 2: Compute sum(exp(x - max))
    T sum_exp = T(0);
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum_exp += exp(input_row[i] - max_val);
    }
    sum_exp = block_reduce_sum(sum_exp, shared);
    __syncthreads();

    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        shared[0] = log(sum_exp);
    }
    __syncthreads();
    T log_sum_exp = shared[0];

    // Step 3: Compute log_softmax = x - max - log_sum_exp
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        output_row[i] = input_row[i] - max_val - log_sum_exp;
    }
}

// LogSoftmax backward: grad_output - exp(log_softmax) * sum(grad_output)
template<typename T>
__global__ void log_softmax_backward_kernel(const T* grad_output, const T* output,
                                           T* grad_input,
                                           int64_t batch_size, int64_t dim_size) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* grad_out_row = grad_output + row * dim_size;
    const T* out_row = output + row * dim_size;
    T* grad_in_row = grad_input + row * dim_size;

    // Compute sum(grad_output)
    T sum_grad = T(0);
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum_grad += grad_out_row[i];
    }
    sum_grad = block_reduce_sum(sum_grad, shared);
    __syncthreads();

    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        shared[0] = sum_grad;
    }
    __syncthreads();
    sum_grad = shared[0];

    // Compute gradient: grad_output - exp(log_softmax) * sum_grad
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        grad_in_row[i] = grad_out_row[i] - exp(out_row[i]) * sum_grad;
    }
}

// Host functions
extern "C" {
    void log_softmax_forward_float(const float* input, float* output,
                                   int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        log_softmax_forward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size>>>(
            input, output, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void log_softmax_forward_double(const double* input, double* output,
                                    int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        log_softmax_forward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size>>>(
            input, output, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void log_softmax_backward_float(const float* grad_output, const float* output,
                                    float* grad_input,
                                    int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        log_softmax_backward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size>>>(
            grad_output, output, grad_input, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void log_softmax_backward_double(const double* grad_output, const double* output,
                                     double* grad_input,
                                     int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        log_softmax_backward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size>>>(
            grad_output, output, grad_input, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Tensor Wrapper Functions
// ============================================================================

// ReLU wrapper
auto relu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;  // Handle empty tensors
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        relu_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        relu_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
    } else {
        throw std::runtime_error("ReLU only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in relu_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// ReLU backward wrapper
auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;  // Handle empty tensors
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        relu_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        relu_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else {
        throw std::runtime_error("ReLU backward only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in relu_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Sigmoid wrapper
auto sigmoid_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        sigmoid_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        sigmoid_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
    } else {
        throw std::runtime_error("Sigmoid only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in sigmoid_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Sigmoid backward wrapper
auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        sigmoid_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        sigmoid_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else {
        throw std::runtime_error("Sigmoid backward only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in sigmoid_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Tanh wrapper
auto tanh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        tanh_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        tanh_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
    } else {
        throw std::runtime_error("Tanh only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in tanh_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Tanh backward wrapper
auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        tanh_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        tanh_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else {
        throw std::runtime_error("Tanh backward only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in tanh_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Leaky ReLU wrapper
auto leaky_relu_kernel(const Tensor& input, float alpha, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(), n, alpha);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n, static_cast<double>(alpha));
    } else {
        throw std::runtime_error("Leaky ReLU only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in leaky_relu_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Leaky ReLU backward wrapper
auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), result.data<float>(), n, alpha);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n, static_cast<double>(alpha));
    } else {
        throw std::runtime_error("Leaky ReLU backward only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in leaky_relu_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Softmax wrapper
auto softmax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor {
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (input.numel() == 0) {
        return result;
    }

    // Handle negative dimension
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Dimension out of range for softmax");
    }

    // For simplicity, assume softmax over last dimension (reshape if needed)
    // Calculate batch size and dimension size
    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        batch_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        batch_size *= shape[i];
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        softmax_forward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<float>(), result.data<float>(), batch_size, dim_size);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        softmax_forward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<double>(), result.data<double>(), batch_size, dim_size);
    } else {
        throw std::runtime_error("Softmax only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in softmax_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Softmax backward wrapper
auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor {
    std::vector<int64_t> shape(output.shape().begin(), output.shape().end());
    Tensor result(shape, output.dtype(), output.device());

    if (output.numel() == 0) {
        return result;
    }

    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        batch_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        batch_size *= shape[i];
    }

    if (output.dtype() == DType::Float32) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        softmax_backward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<float>(), output.data<float>(), result.data<float>(), batch_size, dim_size);
    } else if (output.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        softmax_backward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<double>(), output.data<double>(), result.data<double>(), batch_size, dim_size);
    } else {
        throw std::runtime_error("Softmax backward only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in softmax_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Log Softmax wrapper
auto log_softmax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor {
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (input.numel() == 0) {
        return result;
    }

    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        batch_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        batch_size *= shape[i];
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        log_softmax_forward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<float>(), result.data<float>(), batch_size, dim_size);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        log_softmax_forward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<double>(), result.data<double>(), batch_size, dim_size);
    } else {
        throw std::runtime_error("Log Softmax only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in log_softmax_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Log Softmax backward wrapper
auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor {
    std::vector<int64_t> shape(output.shape().begin(), output.shape().end());
    Tensor result(shape, output.dtype(), output.device());

    if (output.numel() == 0) {
        return result;
    }

    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        batch_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        batch_size *= shape[i];
    }

    if (output.dtype() == DType::Float32) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        log_softmax_backward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<float>(), output.data<float>(), result.data<float>(), batch_size, dim_size);
    } else if (output.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        log_softmax_backward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<double>(), output.data<double>(), result.data<double>(), batch_size, dim_size);
    } else {
        throw std::runtime_error("Log Softmax backward only supports Float32 and Float64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in log_softmax_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

} // namespace cuda
} // namespace tenzor
