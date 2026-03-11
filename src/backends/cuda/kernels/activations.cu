#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include "tenzor/core/tensor.hpp"
#include <curand_kernel.h>
#include <chrono>
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/backend.hpp"  // For OpAttributes (dispatch wrappers)
#include "cuda_launch_utils.cuh"
#include "cuda_common.cuh"
#include <stdexcept>
#include <vector>
#include <charconv>  // For std::from_chars (dispatch wrappers)
#include <span>      // For std::span (dispatch wrappers)

namespace tenzor {
namespace cuda {

// ============================================================================
// CUDA Helper Functions
// ============================================================================

// Centralized error checking
#include "../cuda_error.hpp"

// Grid-stride loop helper
#define CUDA_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// Default block size for element-wise operations (used as fallback)
constexpr int BLOCK_SIZE = 256;

// Calculate grid size for element-wise operations
inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return compute_grid_size(n, block_size);
}

// Occupancy-based kernel launch: replaces hardcoded BLOCK_SIZE with per-kernel optimal config
#define LAUNCH_KERNEL(kernel, n, stream, ...) \
    do { \
        auto [grid_, block_] = optimal_launch_config(kernel, n); \
        kernel<<<grid_, block_, 0, stream>>>(__VA_ARGS__); \
        CUDA_CHECK(cudaGetLastError()); \
    } while(0)

// Variant with dynamic shared memory
#define LAUNCH_KERNEL_SMEM(kernel, n, smem, stream, ...) \
    do { \
        auto [grid_, block_] = optimal_launch_config(kernel, n, smem); \
        kernel<<<grid_, block_, smem, stream>>>(__VA_ARGS__); \
        CUDA_CHECK(cudaGetLastError()); \
    } while(0)

// ============================================================================
// Half-precision (Float16) Device Helper Functions
// ============================================================================

// Generic device max function
template<typename T>
__device__ __forceinline__ T device_max(T a, T b) {
    return a > b ? a : b;
}

// Specialization for __half
template<>
__device__ __forceinline__ __half device_max(__half a, __half b) {
    return __hgt(a, b) ? a : b;
}

// Generic device exp function
template<typename T>
__device__ __forceinline__ T device_exp(T x) {
    return exp(x);
}

// Specialization for __half
template<>
__device__ __forceinline__ __half device_exp(__half x) {
    return hexp(x);
}

// Generic device log function
template<typename T>
__device__ __forceinline__ T device_log(T x) {
    return log(x);
}

// Specialization for __half
template<>
__device__ __forceinline__ __half device_log(__half x) {
    return hlog(x);
}

// ============================================================================
// BFloat16 Device Helper Functions
// ============================================================================

// Specialization for __nv_bfloat16
template<>
__device__ __forceinline__ __nv_bfloat16 device_max(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hgt(a, b) ? a : b;
}

template<>
__device__ __forceinline__ __nv_bfloat16 device_exp(__nv_bfloat16 x) {
    return __float2bfloat16(expf(__bfloat162float(x)));
}

template<>
__device__ __forceinline__ __nv_bfloat16 device_log(__nv_bfloat16 x) {
    return __float2bfloat16(logf(__bfloat162float(x)));
}

// BFloat16 numerically stable sigmoid
__device__ __forceinline__ __nv_bfloat16 sigmoid_stable(__nv_bfloat16 x) {
    float xf = __bfloat162float(x);
    if (xf >= 0.0f) {
        return __float2bfloat16(1.0f / (1.0f + expf(-xf)));
    } else {
        float exp_x = expf(xf);
        return __float2bfloat16(exp_x / (1.0f + exp_x));
    }
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

// Vectorized ReLU backward using float4 for 4x memory throughput
__global__ void relu_backward_vectorized_kernel(const float4* grad_output, const float4* input,
                                                float4* grad_input, int64_t n4) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 g = grad_output[idx];
        float4 x = input[idx];
        float4 result;
        result.x = g.x * (x.x > 0.0f ? 1.0f : 0.0f);
        result.y = g.y * (x.y > 0.0f ? 1.0f : 0.0f);
        result.z = g.z * (x.z > 0.0f ? 1.0f : 0.0f);
        result.w = g.w * (x.w > 0.0f ? 1.0f : 0.0f);
        grad_input[idx] = result;
    }
}

// Handle remainder elements
__global__ void relu_backward_remainder_kernel(const float* grad_output, const float* input,
                                               float* grad_input, int64_t start, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * (input[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

// ============================================================================
// Vectorized Forward Kernels (float4 for 4x memory bandwidth)
// ============================================================================
// These kernels use float4 vectorized loads/stores for significantly improved
// memory bandwidth utilization on large tensors.

// Minimum tensor size to use vectorized kernels (threshold for overhead)
constexpr int64_t VECTORIZED_THRESHOLD = 1024;

// Vectorized ReLU forward using float4
__global__ void relu_forward_vectorized_kernel(const float4* __restrict__ input,
                                                float4* __restrict__ output, int64_t n4) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 x = __ldg(&input[idx]);
        float4 result;
        result.x = fmaxf(0.0f, x.x);
        result.y = fmaxf(0.0f, x.y);
        result.z = fmaxf(0.0f, x.z);
        result.w = fmaxf(0.0f, x.w);
        output[idx] = result;
    }
}

// Handle remainder elements for ReLU forward
__global__ void relu_forward_remainder_kernel(const float* input, float* output,
                                               int64_t start, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Vectorized Sigmoid forward using float4
__global__ void sigmoid_forward_vectorized_kernel(const float4* __restrict__ input,
                                                   float4* __restrict__ output, int64_t n4) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 x = __ldg(&input[idx]);
        float4 result;
        // Numerically stable sigmoid
        result.x = (x.x >= 0.0f) ? (1.0f / (1.0f + expf(-x.x))) : (expf(x.x) / (1.0f + expf(x.x)));
        result.y = (x.y >= 0.0f) ? (1.0f / (1.0f + expf(-x.y))) : (expf(x.y) / (1.0f + expf(x.y)));
        result.z = (x.z >= 0.0f) ? (1.0f / (1.0f + expf(-x.z))) : (expf(x.z) / (1.0f + expf(x.z)));
        result.w = (x.w >= 0.0f) ? (1.0f / (1.0f + expf(-x.w))) : (expf(x.w) / (1.0f + expf(x.w)));
        output[idx] = result;
    }
}

// Handle remainder elements for Sigmoid forward
__global__ void sigmoid_forward_remainder_kernel(const float* input, float* output,
                                                  int64_t start, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < n) {
        float x = input[idx];
        output[idx] = (x >= 0.0f) ? (1.0f / (1.0f + expf(-x))) : (expf(x) / (1.0f + expf(x)));
    }
}

// Vectorized Tanh forward using float4
__global__ void tanh_forward_vectorized_kernel(const float4* __restrict__ input,
                                                float4* __restrict__ output, int64_t n4) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 x = __ldg(&input[idx]);
        float4 result;
        result.x = tanhf(x.x);
        result.y = tanhf(x.y);
        result.z = tanhf(x.z);
        result.w = tanhf(x.w);
        output[idx] = result;
    }
}

// Handle remainder elements for Tanh forward
__global__ void tanh_forward_remainder_kernel(const float* input, float* output,
                                               int64_t start, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

// Helper device function for GELU computation
__device__ __forceinline__ float gelu_scalar(float x) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    return x * 0.5f * (1.0f + tanhf(tanh_arg));
}

// Vectorized GELU forward using float4
__global__ void gelu_forward_vectorized_kernel(const float4* __restrict__ input,
                                                float4* __restrict__ output, int64_t n4) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 x = __ldg(&input[idx]);
        float4 result;
        result.x = gelu_scalar(x.x);
        result.y = gelu_scalar(x.y);
        result.z = gelu_scalar(x.z);
        result.w = gelu_scalar(x.w);
        output[idx] = result;
    }
}

// Handle remainder elements for GELU forward
__global__ void gelu_forward_remainder_kernel(const float* input, float* output,
                                               int64_t start, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < n) {
        output[idx] = gelu_scalar(input[idx]);
    }
}

// Helper device function for Swish computation
__device__ __forceinline__ float swish_scalar(float x) {
    float sigmoid_x = (x >= 0.0f) ? (1.0f / (1.0f + expf(-x))) : (expf(x) / (1.0f + expf(x)));
    return x * sigmoid_x;
}

// Vectorized Swish forward using float4
__global__ void swish_forward_vectorized_kernel(const float4* __restrict__ input,
                                                 float4* __restrict__ output, int64_t n4) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 x = __ldg(&input[idx]);
        float4 result;
        result.x = swish_scalar(x.x);
        result.y = swish_scalar(x.y);
        result.z = swish_scalar(x.z);
        result.w = swish_scalar(x.w);
        output[idx] = result;
    }
}

// Handle remainder elements for Swish forward
__global__ void swish_forward_remainder_kernel(const float* input, float* output,
                                                int64_t start, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < n) {
        output[idx] = swish_scalar(input[idx]);
    }
}

// Vectorized Leaky ReLU forward using float4
__global__ void leaky_relu_forward_vectorized_kernel(const float4* __restrict__ input,
                                                      float4* __restrict__ output,
                                                      int64_t n4, float alpha) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 x = __ldg(&input[idx]);
        float4 result;
        result.x = (x.x > 0.0f) ? x.x : alpha * x.x;
        result.y = (x.y > 0.0f) ? x.y : alpha * x.y;
        result.z = (x.z > 0.0f) ? x.z : alpha * x.z;
        result.w = (x.w > 0.0f) ? x.w : alpha * x.w;
        output[idx] = result;
    }
}

// Handle remainder elements for Leaky ReLU forward
__global__ void leaky_relu_forward_remainder_kernel(const float* input, float* output,
                                                     int64_t start, int64_t n, float alpha) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < n) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? x : alpha * x;
    }
}

// ============================================================================
// Vectorized Backward Kernels (float4 for 4x memory bandwidth)
// ============================================================================

// Vectorized Sigmoid backward using float4
__global__ void sigmoid_backward_vectorized_kernel(const float4* __restrict__ grad_output,
                                                    const float4* __restrict__ input,
                                                    float4* __restrict__ grad_input, int64_t n4) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 g = __ldg(&grad_output[idx]);
        float4 x = __ldg(&input[idx]);
        float4 result;

        // sigmoid(x) * (1 - sigmoid(x)) * grad
        #define SIGMOID_BACKWARD(comp) \
            { \
                float s = (x.comp >= 0.0f) ? (1.0f / (1.0f + expf(-x.comp))) : (expf(x.comp) / (1.0f + expf(x.comp))); \
                result.comp = g.comp * s * (1.0f - s); \
            }

        SIGMOID_BACKWARD(x)
        SIGMOID_BACKWARD(y)
        SIGMOID_BACKWARD(z)
        SIGMOID_BACKWARD(w)

        #undef SIGMOID_BACKWARD

        grad_input[idx] = result;
    }
}

// Vectorized Tanh backward using float4
__global__ void tanh_backward_vectorized_kernel(const float4* __restrict__ grad_output,
                                                 const float4* __restrict__ input,
                                                 float4* __restrict__ grad_input, int64_t n4) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 g = __ldg(&grad_output[idx]);
        float4 x = __ldg(&input[idx]);
        float4 result;

        float tanh_x = tanhf(x.x);
        result.x = g.x * (1.0f - tanh_x * tanh_x);
        tanh_x = tanhf(x.y);
        result.y = g.y * (1.0f - tanh_x * tanh_x);
        tanh_x = tanhf(x.z);
        result.z = g.z * (1.0f - tanh_x * tanh_x);
        tanh_x = tanhf(x.w);
        result.w = g.w * (1.0f - tanh_x * tanh_x);

        grad_input[idx] = result;
    }
}

// Vectorized GELU backward using float4
__global__ void gelu_backward_vectorized_kernel(const float4* __restrict__ grad_output,
                                                 const float4* __restrict__ input,
                                                 float4* __restrict__ grad_input, int64_t n4) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;

    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 g = __ldg(&grad_output[idx]);
        float4 x = __ldg(&input[idx]);
        float4 result;

        #define GELU_BACKWARD(comp) \
            { \
                float x_sq = x.comp * x.comp; \
                float x_cubed = x_sq * x.comp; \
                float z = sqrt_2_over_pi * (x.comp + coeff * x_cubed); \
                float tanh_z = tanhf(z); \
                float dz_dx = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_sq); \
                float sech2_z = 1.0f - tanh_z * tanh_z; \
                result.comp = g.comp * (0.5f * (1.0f + tanh_z) + 0.5f * x.comp * sech2_z * dz_dx); \
            }

        GELU_BACKWARD(x)
        GELU_BACKWARD(y)
        GELU_BACKWARD(z)
        GELU_BACKWARD(w)

        #undef GELU_BACKWARD

        grad_input[idx] = result;
    }
}

// Vectorized Leaky ReLU backward using float4
__global__ void leaky_relu_backward_vectorized_kernel(const float4* __restrict__ grad_output,
                                                       const float4* __restrict__ input,
                                                       float4* __restrict__ grad_input,
                                                       int64_t n4, float alpha) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 g = __ldg(&grad_output[idx]);
        float4 x = __ldg(&input[idx]);
        float4 result;
        result.x = g.x * (x.x > 0.0f ? 1.0f : alpha);
        result.y = g.y * (x.y > 0.0f ? 1.0f : alpha);
        result.z = g.z * (x.z > 0.0f ? 1.0f : alpha);
        result.w = g.w * (x.w > 0.0f ? 1.0f : alpha);
        grad_input[idx] = result;
    }
}

// Vectorized Swish backward using float4
__global__ void swish_backward_vectorized_kernel(const float4* __restrict__ grad_output,
                                                  const float4* __restrict__ input,
                                                  float4* __restrict__ grad_input, int64_t n4) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 g = __ldg(&grad_output[idx]);
        float4 x = __ldg(&input[idx]);
        float4 result;

        #define SWISH_BACKWARD(comp) \
            { \
                float s = (x.comp >= 0.0f) ? (1.0f / (1.0f + expf(-x.comp))) : (expf(x.comp) / (1.0f + expf(x.comp))); \
                result.comp = g.comp * s * (1.0f + x.comp * (1.0f - s)); \
            }

        SWISH_BACKWARD(x)
        SWISH_BACKWARD(y)
        SWISH_BACKWARD(z)
        SWISH_BACKWARD(w)

        #undef SWISH_BACKWARD

        grad_input[idx] = result;
    }
}

// Host functions
extern "C" {
    void relu_forward_float(const float* input, float* output, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(relu_forward_kernel<float>, n);
        relu_forward_kernel<float><<<grid_size, block_size>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void relu_forward_double(const double* input, double* output, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(relu_forward_kernel<double>, n);
        relu_forward_kernel<double><<<grid_size, block_size>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void relu_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(relu_backward_kernel<float>, n);
        relu_backward_kernel<float><<<grid_size, block_size>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void relu_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(relu_backward_kernel<double>, n);
        relu_backward_kernel<double><<<grid_size, block_size>>>(
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

// Specialization for __half
__device__ __forceinline__ __half sigmoid_stable(__half x) {
    if (__hge(x, __float2half(0.0f))) {
        return __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(x))));
    } else {
        __half exp_x = hexp(x);
        return __hdiv(exp_x, __hadd(__float2half(1.0f), exp_x));
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

// Swish activation: swish(x) = x * sigmoid(x)
template<typename T>
__global__ void swish_forward_kernel(const T* input, T* output, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        output[idx] = x * sigmoid_stable(x);
    }
}

// Swish backward: grad_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
//                = grad_out * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
template<typename T>
__global__ void swish_backward_cuda_kernel(const T* grad_output, const T* input,
                                            T* grad_input, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T sigmoid_x = sigmoid_stable(x);
        // d/dx swish(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //               = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        T grad = sigmoid_x * (T(1) + x * (T(1) - sigmoid_x));
        grad_input[idx] = grad_output[idx] * grad;
    }
}

// Host functions
extern "C" {
    void sigmoid_forward_float(const float* input, float* output, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(sigmoid_forward_kernel<float>, n);
        sigmoid_forward_kernel<float><<<grid_size, block_size>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void sigmoid_forward_double(const double* input, double* output, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(sigmoid_forward_kernel<double>, n);
        sigmoid_forward_kernel<double><<<grid_size, block_size>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void sigmoid_backward_float(const float* grad_output, const float* input,
                               float* grad_input, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(sigmoid_backward_kernel<float>, n);
        sigmoid_backward_kernel<float><<<grid_size, block_size>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void sigmoid_backward_double(const double* grad_output, const double* input,
                                double* grad_input, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(sigmoid_backward_kernel<double>, n);
        sigmoid_backward_kernel<double><<<grid_size, block_size>>>(
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

// Specialization for __half
template<>
__global__ void tanh_forward_kernel<__half>(const __half* input, __half* output, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = __float2half(tanhf(__half2float(input[idx])));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void tanh_forward_kernel<__nv_bfloat16>(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = __float2bfloat16(tanhf(__bfloat162float(input[idx])));
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

// Specialization for __half
template<>
__global__ void tanh_backward_kernel<__half>(const __half* grad_output, const __half* input,
                                            __half* grad_input, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float tanh_x = tanhf(__half2float(input[idx]));
        grad_input[idx] = __float2half(__half2float(grad_output[idx]) * (1.0f - tanh_x * tanh_x));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void tanh_backward_kernel<__nv_bfloat16>(const __nv_bfloat16* grad_output, const __nv_bfloat16* input,
                                                   __nv_bfloat16* grad_input, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float tanh_x = tanhf(__bfloat162float(input[idx]));
        grad_input[idx] = __float2bfloat16(__bfloat162float(grad_output[idx]) * (1.0f - tanh_x * tanh_x));
    }
}

// Host functions
extern "C" {
    void tanh_forward_float(const float* input, float* output, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(tanh_forward_kernel<float>, n);
        tanh_forward_kernel<float><<<grid_size, block_size>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void tanh_forward_double(const double* input, double* output, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(tanh_forward_kernel<double>, n);
        tanh_forward_kernel<double><<<grid_size, block_size>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void tanh_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(tanh_backward_kernel<float>, n);
        tanh_backward_kernel<float><<<grid_size, block_size>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void tanh_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(tanh_backward_kernel<double>, n);
        tanh_backward_kernel<double><<<grid_size, block_size>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// GELU Activation
// ============================================================================

// Forward: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
template<typename T>
__global__ void gelu_forward_kernel(const T* input, T* output, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T x_cubed = x * x * x;
        T tanh_arg = T(0.7978845608) * (x + T(0.044715) * x_cubed);
        output[idx] = x * T(0.5) * (T(1.0) + tanh(tanh_arg));
    }
}

// Specialization for __half
template<>
__global__ void gelu_forward_kernel<__half>(const __half* input, __half* output, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float x = __half2float(input[idx]);
        float x_cubed = x * x * x;
        float tanh_arg = 0.7978845608f * (x + 0.044715f * x_cubed);
        output[idx] = __float2half(x * 0.5f * (1.0f + tanhf(tanh_arg)));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void gelu_forward_kernel<__nv_bfloat16>(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float x = __bfloat162float(input[idx]);
        float x_cubed = x * x * x;
        float tanh_arg = 0.7978845608f * (x + 0.044715f * x_cubed);
        output[idx] = __float2bfloat16(x * 0.5f * (1.0f + tanhf(tanh_arg)));
    }
}

// Backward: grad_out * df/dx
// where df/dx = 0.5 * (1 + tanh(z)) + 0.5 * x * sech²(z) * dz/dx
// z = sqrt(2/π) * (x + 0.044715 * x³)
// dz/dx = sqrt(2/π) * (1 + 0.134145 * x²)
// sech²(z) = 1 - tanh²(z)
template<typename T>
__global__ void gelu_backward_kernel(const T* grad_output, const T* input,
                                     T* grad_input, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T x_squared = x * x;
        T x_cubed = x_squared * x;

        // Constants
        constexpr T sqrt_2_over_pi = T(0.7978845608);
        constexpr T coeff = T(0.044715);

        // Compute z and tanh(z)
        T z = sqrt_2_over_pi * (x + coeff * x_cubed);
        T tanh_z = tanh(z);

        // Compute dz/dx
        T dz_dx = sqrt_2_over_pi * (T(1.0) + T(3.0) * coeff * x_squared);

        // Compute sech²(z) = 1 - tanh²(z)
        T sech2_z = T(1.0) - tanh_z * tanh_z;

        // Compute df/dx
        T df_dx = T(0.5) * (T(1.0) + tanh_z) + T(0.5) * x * sech2_z * dz_dx;

        grad_input[idx] = grad_output[idx] * df_dx;
    }
}

// Specialization for __half
template<>
__global__ void gelu_backward_kernel<__half>(const __half* grad_output, const __half* input,
                                              __half* grad_input, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float x = __half2float(input[idx]);
        float x_squared = x * x;
        float x_cubed = x_squared * x;

        // Constants
        constexpr float sqrt_2_over_pi = 0.7978845608f;
        constexpr float coeff = 0.044715f;

        // Compute z and tanh(z)
        float z = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_z = tanhf(z);

        // Compute dz/dx
        float dz_dx = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_squared);

        // Compute sech²(z) = 1 - tanh²(z)
        float sech2_z = 1.0f - tanh_z * tanh_z;

        // Compute df/dx
        float df_dx = 0.5f * (1.0f + tanh_z) + 0.5f * x * sech2_z * dz_dx;

        grad_input[idx] = __float2half(__half2float(grad_output[idx]) * df_dx);
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void gelu_backward_kernel<__nv_bfloat16>(const __nv_bfloat16* grad_output, const __nv_bfloat16* input,
                                                    __nv_bfloat16* grad_input, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float x = __bfloat162float(input[idx]);
        float x_squared = x * x;
        float x_cubed = x_squared * x;

        // Constants
        constexpr float sqrt_2_over_pi = 0.7978845608f;
        constexpr float coeff = 0.044715f;

        // Compute z and tanh(z)
        float z = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_z = tanhf(z);

        // Compute dz/dx
        float dz_dx = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_squared);

        // Compute sech²(z) = 1 - tanh²(z)
        float sech2_z = 1.0f - tanh_z * tanh_z;

        // Compute df/dx
        float df_dx = 0.5f * (1.0f + tanh_z) + 0.5f * x * sech2_z * dz_dx;

        grad_input[idx] = __float2bfloat16(__bfloat162float(grad_output[idx]) * df_dx);
    }
}

// Host functions
extern "C" {
    void gelu_forward_float(const float* input, float* output, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(gelu_forward_kernel<float>, n);
        gelu_forward_kernel<float><<<grid_size, block_size>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void gelu_forward_double(const double* input, double* output, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(gelu_forward_kernel<double>, n);
        gelu_forward_kernel<double><<<grid_size, block_size>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void gelu_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(gelu_backward_kernel<float>, n);
        gelu_backward_kernel<float><<<grid_size, block_size>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void gelu_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n) {
        auto [grid_size, block_size] = optimal_launch_config(gelu_backward_kernel<double>, n);
        gelu_backward_kernel<double><<<grid_size, block_size>>>(
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

// Float16 forward kernel: compute in Float32 for numerical stability
__global__ void leaky_relu_forward_fp16_kernel(const __half* input, __half* output,
                                                int64_t n, float alpha) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        output[idx] = __float2half(val > 0.0f ? val : alpha * val);
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

// Float16 backward kernel: compute in Float32 for numerical stability
__global__ void leaky_relu_backward_fp16_kernel(const __half* grad_output, const __half* input,
                                                 __half* grad_input, int64_t n, float alpha) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float grad = __half2float(grad_output[idx]);
        float val = __half2float(input[idx]);
        grad_input[idx] = __float2half(grad * (val > 0.0f ? 1.0f : alpha));
    }
}

// Host functions
extern "C" {
    void leaky_relu_forward_float(const float* input, float* output,
                                 int64_t n, float alpha) {
        LAUNCH_KERNEL(leaky_relu_forward_kernel<float>, n, nullptr,
            input, output, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void leaky_relu_forward_double(const double* input, double* output,
                                  int64_t n, double alpha) {
        LAUNCH_KERNEL(leaky_relu_forward_kernel<double>, n, nullptr,
            input, output, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void leaky_relu_backward_float(const float* grad_output, const float* input,
                                  float* grad_input, int64_t n, float alpha) {
        LAUNCH_KERNEL(leaky_relu_backward_kernel<float>, n, nullptr,
            grad_output, input, grad_input, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void leaky_relu_backward_double(const double* grad_output, const double* input,
                                   double* grad_input, int64_t n, double alpha) {
        LAUNCH_KERNEL(leaky_relu_backward_kernel<double>, n, nullptr,
            grad_output, input, grad_input, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// ELU Activation: f(x) = x if x > 0 else alpha * (exp(x) - 1)
// ============================================================================

template<typename T>
__global__ void elu_forward_kernel(const T* input, T* output, int64_t n, float alpha) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T x = input[idx];
        output[idx] = (x > T(0)) ? x : T(alpha * (exp(float(x)) - 1.0f));
    }
}

template<typename T>
__global__ void elu_backward_kernel(const T* grad_output, const T* input,
                                     T* grad_input, int64_t n, float alpha) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T x = input[idx];
        T grad = (x > T(0)) ? T(1) : T(alpha * exp(float(x)));
        grad_input[idx] = grad_output[idx] * grad;
    }
}

extern "C" {
    void elu_forward_float(const float* input, float* output, int64_t n, float alpha) {
        LAUNCH_KERNEL(elu_forward_kernel<float>, n, nullptr, input, output, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void elu_forward_double(const double* input, double* output, int64_t n, float alpha) {
        LAUNCH_KERNEL(elu_forward_kernel<double>, n, nullptr, input, output, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void elu_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n, float alpha) {
        LAUNCH_KERNEL(elu_backward_kernel<float>, n, nullptr,
            grad_output, input, grad_input, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void elu_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n, float alpha) {
        LAUNCH_KERNEL(elu_backward_kernel<double>, n, nullptr,
            grad_output, input, grad_input, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// SELU Activation: f(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
// ============================================================================

constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;

template<typename T>
__global__ void selu_forward_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = float(input[idx]);
        float result = (x > 0.0f) ? x : SELU_ALPHA * (expf(x) - 1.0f);
        output[idx] = T(SELU_SCALE * result);
    }
}

template<typename T>
__global__ void selu_backward_kernel(const T* grad_output, const T* input,
                                      T* grad_input, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = float(input[idx]);
        float grad = (x > 0.0f) ? SELU_SCALE : SELU_SCALE * SELU_ALPHA * expf(x);
        grad_input[idx] = T(float(grad_output[idx]) * grad);
    }
}

extern "C" {
    void selu_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        selu_forward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void selu_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        selu_forward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void selu_backward_float(const float* grad_output, const float* input,
                             float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        selu_backward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void selu_backward_double(const double* grad_output, const double* input,
                              double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        selu_backward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Mish Activation: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
// ============================================================================

template<typename T>
__global__ void mish_forward_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = float(input[idx]);
        // Numerically stable softplus
        float softplus;
        if (x > 20.0f) {
            softplus = x;
        } else if (x < -20.0f) {
            softplus = expf(x);
        } else {
            softplus = log1pf(expf(x));
        }
        output[idx] = T(x * tanhf(softplus));
    }
}

template<typename T>
__global__ void mish_backward_kernel(const T* grad_output, const T* input,
                                      T* grad_input, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = float(input[idx]);
        float softplus;
        if (x > 20.0f) {
            softplus = x;
        } else if (x < -20.0f) {
            softplus = expf(x);
        } else {
            softplus = log1pf(expf(x));
        }
        float tanh_sp = tanhf(softplus);
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float sech2 = 1.0f - tanh_sp * tanh_sp;
        float grad = tanh_sp + x * sech2 * sigmoid_x;
        grad_input[idx] = T(float(grad_output[idx]) * grad);
    }
}

extern "C" {
    void mish_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        mish_forward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void mish_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        mish_forward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void mish_backward_float(const float* grad_output, const float* input,
                             float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        mish_backward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void mish_backward_double(const double* grad_output, const double* input,
                              double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        mish_backward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Softplus Activation: f(x) = ln(1 + exp(beta * x)) / beta
// ============================================================================

template<typename T>
__global__ void softplus_forward_kernel(const T* input, T* output, int64_t n,
                                         float beta, float threshold) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = float(input[idx]) * beta;
        float result;
        if (x > threshold) {
            result = float(input[idx]);
        } else if (x < -threshold) {
            result = expf(x) / beta;
        } else {
            result = log1pf(expf(x)) / beta;
        }
        output[idx] = T(result);
    }
}

template<typename T>
__global__ void softplus_backward_kernel(const T* grad_output, const T* input,
                                          T* grad_input, int64_t n,
                                          float beta, float threshold) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = float(input[idx]) * beta;
        float sigmoid_x;
        if (x > threshold) {
            sigmoid_x = 1.0f;
        } else if (x < -threshold) {
            sigmoid_x = expf(x);
        } else {
            sigmoid_x = 1.0f / (1.0f + expf(-x));
        }
        grad_input[idx] = T(float(grad_output[idx]) * sigmoid_x);
    }
}

extern "C" {
    void softplus_forward_float(const float* input, float* output, int64_t n,
                                float beta, float threshold) {
        int num_blocks = get_num_blocks(n);
        softplus_forward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(input, output, n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    void softplus_forward_double(const double* input, double* output, int64_t n,
                                 float beta, float threshold) {
        int num_blocks = get_num_blocks(n);
        softplus_forward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(input, output, n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    void softplus_backward_float(const float* grad_output, const float* input,
                                 float* grad_input, int64_t n,
                                 float beta, float threshold) {
        int num_blocks = get_num_blocks(n);
        softplus_backward_kernel<float><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    void softplus_backward_double(const double* grad_output, const double* input,
                                  double* grad_input, int64_t n,
                                  float beta, float threshold) {
        int num_blocks = get_num_blocks(n);
        softplus_backward_kernel<double><<<num_blocks, BLOCK_SIZE>>>(
            grad_output, input, grad_input, n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// Softmax Activation
// ============================================================================

// Shared memory size for reductions
constexpr int SOFTMAX_BLOCK_SIZE = 256;

// Type-appropriate minimum value for max reduction initialization
template<typename T>
__device__ __forceinline__ T numeric_min() {
    return -FLT_MAX;  // Default for float
}

template<>
__device__ __forceinline__ double numeric_min<double>() {
    return -DBL_MAX;
}

template<>
__device__ __forceinline__ __half numeric_min<__half>() {
    return __float2half(-65504.0f);
}

template<>
__device__ __forceinline__ __nv_bfloat16 numeric_min<__nv_bfloat16>() {
    return __float2bfloat16(-3.38e38f);  // Close to -FLT_MAX but within BFloat16 range
}

// warp_reduce_sum, block_reduce_sum, warp_reduce_max are in cuda_common.cuh

// Block-level max reduction using shared memory (activations-specific with numeric_min identity)
template<typename T>
__device__ T block_reduce_max(T val, T* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : numeric_min<T>();
    if (wid == 0) {
        val = warp_reduce_max(val);
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
    T max_val = numeric_min<T>();
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        max_val = device_max(max_val, input_row[i]);
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
        T exp_val = device_exp(input_row[i] - max_val);
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
    T max_val = numeric_min<T>();
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        max_val = device_max(max_val, input_row[i]);
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
        sum_exp += device_exp(input_row[i] - max_val);
    }
    sum_exp = block_reduce_sum(sum_exp, shared);
    __syncthreads();

    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        shared[0] = device_log(sum_exp);
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
        grad_in_row[i] = grad_out_row[i] - device_exp(out_row[i]) * sum_grad;
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
// In-Place Activation CUDA Kernels
// ============================================================================

// In-place ReLU: x = max(0, x)
template<typename T>
__global__ void relu_inplace_cuda_kernel(T* data, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        data[idx] = data[idx] > T(0) ? data[idx] : T(0);
    }
}

// Vectorized in-place ReLU using float4
__global__ void relu_inplace_vectorized_kernel(float4* __restrict__ data, int64_t n4) {
    CUDA_GRID_STRIDE_LOOP(idx, n4) {
        float4 x = data[idx];
        x.x = fmaxf(0.0f, x.x);
        x.y = fmaxf(0.0f, x.y);
        x.z = fmaxf(0.0f, x.z);
        x.w = fmaxf(0.0f, x.w);
        data[idx] = x;
    }
}

__global__ void relu_inplace_remainder_kernel(float* data, int64_t start, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < n) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// In-place Sigmoid: x = 1 / (1 + exp(-x))
template<typename T>
__global__ void sigmoid_inplace_cuda_kernel(T* data, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        data[idx] = T(1) / (T(1) + device_exp(-data[idx]));
    }
}

// In-place Tanh: x = tanh(x)
template<typename T>
__global__ void tanh_inplace_cuda_kernel(T* data, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        data[idx] = tanh(data[idx]);
    }
}

// Specialization for __half
template<>
__global__ void tanh_inplace_cuda_kernel<__half>(__half* data, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        // Convert to float, apply tanh, convert back
        float val = __half2float(data[idx]);
        data[idx] = __float2half(tanhf(val));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void tanh_inplace_cuda_kernel<__nv_bfloat16>(__nv_bfloat16* data, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        // Convert to float, apply tanh, convert back
        float val = __bfloat162float(data[idx]);
        data[idx] = __float2bfloat16(tanhf(val));
    }
}

// In-place LeakyReLU: x = max(alpha * x, x)
template<typename T>
__global__ void leaky_relu_inplace_cuda_kernel(T* data, T alpha, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        T val = data[idx];
        data[idx] = val > T(0) ? val : alpha * val;
    }
}

// Float16 inplace kernel: compute in Float32 for numerical stability
__global__ void leaky_relu_inplace_fp16_kernel(__half* data, float alpha, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float val = __half2float(data[idx]);
        data[idx] = __float2half(val > 0.0f ? val : alpha * val);
    }
}

// In-place GELU: x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
template<typename T>
__global__ void gelu_inplace_cuda_kernel(T* data, int64_t n) {
    constexpr T sqrt_2_over_pi = T(0.7978845608028654);
    constexpr T coeff = T(0.044715);

    CUDA_GRID_STRIDE_LOOP(idx, n) {
        T x = data[idx];
        T x_cubed = x * x * x;
        T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        data[idx] = T(0.5) * x * (T(1) + tanh(inner));
    }
}

// Specialization for __half
template<>
__global__ void gelu_inplace_cuda_kernel<__half>(__half* data, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float x = __half2float(data[idx]);
        float x_cubed = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x_cubed);
        data[idx] = __float2half(0.5f * x * (1.0f + tanhf(inner)));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void gelu_inplace_cuda_kernel<__nv_bfloat16>(__nv_bfloat16* data, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        float x = __bfloat162float(data[idx]);
        float x_cubed = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x_cubed);
        data[idx] = __float2bfloat16(0.5f * x * (1.0f + tanhf(inner)));
    }
}

// ============================================================================
// In-Place Activation Tensor Wrapper Functions
// ============================================================================

// In-place ReLU wrapper
auto relu_inplace_kernel(Tensor& input, cudaStream_t stream) -> void {
    int64_t n = input.numel();
    if (n == 0) return;

    if (input.dtype() == DType::Float32) {
        float* data_ptr = input.data<float>();

        // Use vectorized kernel for large tensors with aligned pointers
        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(data_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                relu_inplace_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<float4*>(data_ptr), n4);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                relu_inplace_remainder_kernel<<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    data_ptr, start, n);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            relu_inplace_cuda_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(data_ptr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        relu_inplace_cuda_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        relu_inplace_cuda_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half*>(input.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        relu_inplace_cuda_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("ReLU inplace only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in relu_inplace_kernel: ") + cudaGetErrorString(err));
    }
}

// In-place Sigmoid wrapper
auto sigmoid_inplace_kernel(Tensor& input, cudaStream_t stream) -> void {
    int64_t n = input.numel();
    if (n == 0) return;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        sigmoid_inplace_cuda_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        sigmoid_inplace_cuda_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        sigmoid_inplace_cuda_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half*>(input.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        sigmoid_inplace_cuda_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Sigmoid inplace only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in sigmoid_inplace_kernel: ") + cudaGetErrorString(err));
    }
}

// In-place Tanh wrapper
auto tanh_inplace_kernel(Tensor& input, cudaStream_t stream) -> void {
    int64_t n = input.numel();
    if (n == 0) return;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        tanh_inplace_cuda_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        tanh_inplace_cuda_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        tanh_inplace_cuda_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half*>(input.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        tanh_inplace_cuda_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Tanh inplace only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in tanh_inplace_kernel: ") + cudaGetErrorString(err));
    }
}

// In-place LeakyReLU wrapper
auto leaky_relu_inplace_kernel(Tensor& input, float alpha, cudaStream_t stream) -> void {
    int64_t n = input.numel();
    if (n == 0) return;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        leaky_relu_inplace_cuda_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), alpha, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        leaky_relu_inplace_cuda_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), static_cast<double>(alpha), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        leaky_relu_inplace_fp16_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half*>(input.data_ptr()), alpha, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        leaky_relu_inplace_cuda_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr()), __float2bfloat16(alpha), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("LeakyReLU inplace only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in leaky_relu_inplace_kernel: ") + cudaGetErrorString(err));
    }
}

// In-place GELU wrapper
auto gelu_inplace_kernel(Tensor& input, cudaStream_t stream) -> void {
    int64_t n = input.numel();
    if (n == 0) return;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        gelu_inplace_cuda_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        gelu_inplace_cuda_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        gelu_inplace_cuda_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half*>(input.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        gelu_inplace_cuda_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("GELU inplace only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in gelu_inplace_kernel: ") + cudaGetErrorString(err));
    }
}

// ============================================================================
// Tensor Wrapper Functions
// ============================================================================

// ReLU wrapper - uses vectorized float4 for 4x memory throughput on Float32
auto relu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;  // Handle empty tensors
    }

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        // Use vectorized kernel for large tensors with aligned pointers
        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                relu_forward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                relu_forward_remainder_kernel<<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    input_ptr, result_ptr, start, n);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            // Fallback to scalar kernel
            int num_blocks = get_num_blocks(n);
            relu_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input_ptr, result_ptr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        relu_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        relu_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        relu_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("ReLU only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in relu_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// ReLU backward wrapper - uses vectorized float4 for 4x memory throughput
auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        // NOTE: Removed sync - no operations for empty tensors
        return result;  // Handle empty tensors
    }

    if (input.dtype() == DType::Float32) {
        // Use vectorized kernel for Float32 - 4x memory throughput
        const float* grad_ptr = grad_output.data<float>();
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        int64_t n4 = n / 4;  // Number of float4 elements
        int64_t remainder = n % 4;

        if (n4 > 0) {
            int num_blocks = get_num_blocks(n4);
            relu_backward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const float4*>(grad_ptr),
                reinterpret_cast<const float4*>(input_ptr),
                reinterpret_cast<float4*>(result_ptr), n4);
            CUDA_CHECK(cudaGetLastError());
        }

        // Handle remainder elements
        if (remainder > 0) {
            int64_t start = n4 * 4;
            int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
            relu_backward_remainder_kernel<<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                grad_ptr, input_ptr, result_ptr, start, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        relu_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        relu_backward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        relu_backward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("ReLU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in relu_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Sigmoid wrapper - uses vectorized float4 for 4x memory throughput on Float32
auto sigmoid_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        // Use vectorized kernel for large tensors with aligned pointers
        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                sigmoid_forward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                sigmoid_forward_remainder_kernel<<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    input_ptr, result_ptr, start, n);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            sigmoid_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input_ptr, result_ptr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        sigmoid_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        sigmoid_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        sigmoid_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Sigmoid only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in sigmoid_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Sigmoid backward wrapper - uses vectorized float4 for 4x memory throughput
auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        const float* grad_ptr = grad_output.data<float>();
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(grad_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                sigmoid_backward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(grad_ptr),
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                sigmoid_backward_kernel<float><<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    grad_ptr + start, input_ptr + start, result_ptr + start, remainder);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            sigmoid_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                grad_ptr, input_ptr, result_ptr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        sigmoid_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        sigmoid_backward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        sigmoid_backward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Sigmoid backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in sigmoid_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Swish wrapper - uses vectorized float4 for 4x memory throughput on Float32
auto swish_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                swish_forward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                swish_forward_remainder_kernel<<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    input_ptr, result_ptr, start, n);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            swish_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input_ptr, result_ptr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        swish_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        swish_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        swish_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Swish only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in swish_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Swish backward wrapper - uses vectorized float4 for 4x memory throughput
auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        const float* grad_ptr = grad_output.data<float>();
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(grad_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                swish_backward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(grad_ptr),
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                swish_backward_cuda_kernel<float><<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    grad_ptr + start, input_ptr + start, result_ptr + start, remainder);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            swish_backward_cuda_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                grad_ptr, input_ptr, result_ptr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        swish_backward_cuda_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        swish_backward_cuda_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        swish_backward_cuda_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Swish backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in swish_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Tanh wrapper - uses vectorized float4 for 4x memory throughput on Float32
auto tanh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                tanh_forward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                tanh_forward_remainder_kernel<<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    input_ptr, result_ptr, start, n);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            tanh_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input_ptr, result_ptr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        tanh_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        tanh_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        tanh_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Tanh only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in tanh_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Tanh backward wrapper - uses vectorized float4 for 4x memory throughput
auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        const float* grad_ptr = grad_output.data<float>();
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(grad_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                tanh_backward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(grad_ptr),
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                tanh_backward_kernel<float><<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    grad_ptr + start, input_ptr + start, result_ptr + start, remainder);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            tanh_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                grad_ptr, input_ptr, result_ptr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        tanh_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        tanh_backward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        tanh_backward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Tanh backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in tanh_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// GELU wrapper - uses vectorized float4 for 4x memory throughput on Float32
auto gelu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                gelu_forward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gelu_forward_remainder_kernel<<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    input_ptr, result_ptr, start, n);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            gelu_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input_ptr, result_ptr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        gelu_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        gelu_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        gelu_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("GELU only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in gelu_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// GELU backward wrapper - uses vectorized float4 for 4x memory throughput
auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        const float* grad_ptr = grad_output.data<float>();
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(grad_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                gelu_backward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(grad_ptr),
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gelu_backward_kernel<float><<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    grad_ptr + start, input_ptr + start, result_ptr + start, remainder);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            gelu_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                grad_ptr, input_ptr, result_ptr, n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        gelu_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        gelu_backward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        gelu_backward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("GELU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in gelu_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Leaky ReLU wrapper - uses vectorized float4 for 4x memory throughput on Float32
auto leaky_relu_kernel(const Tensor& input, float alpha, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                leaky_relu_forward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4, alpha);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                leaky_relu_forward_remainder_kernel<<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    input_ptr, result_ptr, start, n, alpha);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            leaky_relu_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input_ptr, result_ptr, n, alpha);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n, static_cast<double>(alpha));
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_forward_fp16_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n, alpha);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n, __float2bfloat16(alpha));
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Leaky ReLU only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
        // NOTE: Removed sync - no operations for empty tensors
        return result;
    }

    if (input.dtype() == DType::Float32) {
        const float* grad_ptr = grad_output.data<float>();
        const float* input_ptr = input.data<float>();
        float* result_ptr = result.data<float>();

        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(grad_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

            int64_t n4 = n / 4;
            int64_t remainder = n % 4;

            if (n4 > 0) {
                int num_blocks = get_num_blocks(n4);
                leaky_relu_backward_vectorized_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float4*>(grad_ptr),
                    reinterpret_cast<const float4*>(input_ptr),
                    reinterpret_cast<float4*>(result_ptr), n4, alpha);
                CUDA_CHECK(cudaGetLastError());
            }

            if (remainder > 0) {
                int64_t start = n4 * 4;
                int num_blocks_rem = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
                leaky_relu_backward_kernel<float><<<num_blocks_rem, BLOCK_SIZE, 0, stream>>>(
                    grad_ptr + start, input_ptr + start, result_ptr + start, remainder, alpha);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            int num_blocks = get_num_blocks(n);
            leaky_relu_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                grad_ptr, input_ptr, result_ptr, n, alpha);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n, static_cast<double>(alpha));
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_backward_fp16_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n, alpha);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        leaky_relu_backward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n, __float2bfloat16(alpha));
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Leaky ReLU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in leaky_relu_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// ELU wrapper
auto elu_kernel(const Tensor& input, float alpha, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        // NOTE: Removed sync - no operations for empty tensors
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        elu_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(), n, alpha);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        elu_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n, alpha);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        elu_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n, alpha);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        elu_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n, alpha);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("ELU only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in elu_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// ELU backward wrapper
auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        // NOTE: Removed sync - no operations for empty tensors
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        elu_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), result.data<float>(), n, alpha);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        elu_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n, alpha);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        elu_backward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n, alpha);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        elu_backward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n, alpha);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("ELU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in elu_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// SELU wrapper
auto selu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        // NOTE: Removed sync - no operations for empty tensors
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        selu_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        selu_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        selu_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        selu_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("SELU only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in selu_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// SELU backward wrapper
auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        // NOTE: Removed sync - no operations for empty tensors
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        selu_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        selu_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        selu_backward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        selu_backward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("SELU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in selu_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Mish wrapper
auto mish_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        // NOTE: Removed sync - no operations for empty tensors
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        mish_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        mish_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        mish_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        mish_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Mish only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in mish_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Mish backward wrapper
auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        // NOTE: Removed sync - no operations for empty tensors
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        mish_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        mish_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        mish_backward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        mish_backward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Mish backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in mish_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Softplus wrapper
auto softplus_kernel(const Tensor& input, float beta, float threshold, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        // NOTE: Removed sync - no operations for empty tensors
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        softplus_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(), n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        softplus_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        softplus_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        softplus_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Softplus only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in softplus_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Softplus backward wrapper
auto softplus_backward_kernel(const Tensor& grad_output, const Tensor& input, float beta, float threshold, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        // NOTE: Removed sync - no operations for empty tensors
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        softplus_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), result.data<float>(), n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        softplus_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), result.data<double>(), n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        softplus_backward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = get_num_blocks(n);
        softplus_backward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Softplus backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in softplus_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// Softmax wrapper
auto softmax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor {
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    // Initialize result to zero
    if (input.dtype() == DType::Float32) {
        cudaMemsetAsync(result.data_ptr(), 0, result.numel() * sizeof(float), stream);
    } else if (input.dtype() == DType::Float64) {
        cudaMemsetAsync(result.data_ptr(), 0, result.numel() * sizeof(double), stream);
    } else if (input.dtype() == DType::Float16) {
        cudaMemsetAsync(result.data_ptr(), 0, result.numel() * sizeof(__half), stream);
    }

    if (input.numel() == 0) {
        // NOTE: Removed sync - no operations for empty tensors
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
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        softmax_forward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<double>(), result.data<double>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__half);
        softmax_forward_kernel<__half><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__nv_bfloat16);
        softmax_forward_kernel<__nv_bfloat16><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Softmax only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in softmax_kernel: ") + cudaGetErrorString(err));
    }

    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Softmax backward wrapper
auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor {
    std::vector<int64_t> shape(output.shape().begin(), output.shape().end());
    Tensor result(shape, output.dtype(), output.device());

    if (output.numel() == 0) {
        // NOTE: Removed sync - no operations for empty tensors
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
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        softmax_backward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<double>(), output.data<double>(), result.data<double>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::Float16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__half);
        softmax_backward_kernel<__half><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(output.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::BFloat16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__nv_bfloat16);
        softmax_backward_kernel<__nv_bfloat16><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Softmax backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
        // NOTE: Removed sync - no operations for empty tensors
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
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        log_softmax_forward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<double>(), result.data<double>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__half);
        log_softmax_forward_kernel<__half><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__nv_bfloat16);
        log_softmax_forward_kernel<__nv_bfloat16><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Log Softmax only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
        // NOTE: Removed sync - no operations for empty tensors
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
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        log_softmax_backward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<double>(), output.data<double>(), result.data<double>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::Float16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__half);
        log_softmax_backward_kernel<__half><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(output.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::BFloat16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__nv_bfloat16);
        log_softmax_backward_kernel<__nv_bfloat16><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Log Softmax backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in log_softmax_backward_kernel: ") + cudaGetErrorString(err));
    }

    return result;
}

// ============================================================================
// Dispatch-Conformant Wrappers (SingleOutputKernelFn signature)
// ============================================================================
// These wrappers match Tensor(*)(std::span<const Tensor>, const OpAttributes&)
// for direct registration with register_single_output_kernel()

namespace {
// Helper to extract stream from attrs (inlined for performance)
inline cudaStream_t get_stream(const OpAttributes& attrs) {
    if (attrs.empty()) return nullptr;
    if (!attrs.has(AttrKey::Stream)) return nullptr;
    auto val = static_cast<uint64_t>(attrs.get_int(AttrKey::Stream, 0));
    return reinterpret_cast<cudaStream_t>(val);
}
} // anonymous namespace

// ReLU dispatch wrappers
Tensor relu_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return relu_kernel(inputs[0], get_stream(attrs));
}

Tensor relu_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return relu_backward_kernel(inputs[0], inputs[1], get_stream(attrs));
}

// Sigmoid dispatch wrappers
Tensor sigmoid_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return sigmoid_kernel(inputs[0], get_stream(attrs));
}

Tensor sigmoid_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return sigmoid_backward_kernel(inputs[0], inputs[1], get_stream(attrs));
}

// Tanh dispatch wrappers
Tensor tanh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return tanh_kernel(inputs[0], get_stream(attrs));
}

Tensor tanh_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return tanh_backward_kernel(inputs[0], inputs[1], get_stream(attrs));
}

// GELU dispatch wrappers
Tensor gelu_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return gelu_kernel(inputs[0], get_stream(attrs));
}

Tensor gelu_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return gelu_backward_kernel(inputs[0], inputs[1], get_stream(attrs));
}

// Swish dispatch wrappers
Tensor swish_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return swish_kernel(inputs[0], get_stream(attrs));
}

Tensor swish_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return swish_backward_kernel(inputs[0], inputs[1], get_stream(attrs));
}

// SELU dispatch wrappers
Tensor selu_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return selu_kernel(inputs[0], get_stream(attrs));
}

Tensor selu_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return selu_backward_kernel(inputs[0], inputs[1], get_stream(attrs));
}

// Mish dispatch wrappers
Tensor mish_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return mish_kernel(inputs[0], get_stream(attrs));
}

Tensor mish_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return mish_backward_kernel(inputs[0], inputs[1], get_stream(attrs));
}


// ============================================================================
// Dropout Forward/Backward CUDA Kernels
// ============================================================================

template<typename T>
__global__ void dropout_forward_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    uint8_t* __restrict__ mask,
    int64_t n,
    float p,
    float scale,
    uint64_t seed) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Initialize cuRAND state per thread
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float rand_val = curand_uniform(&state);
    bool keep = rand_val >= p;
    mask[idx] = keep ? 1 : 0;
    output[idx] = keep ? static_cast<T>(static_cast<float>(input[idx]) * scale) : T(0);
}

__global__ void dropout_forward_kernel_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    uint8_t* __restrict__ mask,
    int64_t n,
    float p,
    float scale,
    uint64_t seed) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float rand_val = curand_uniform(&state);
    bool keep = rand_val >= p;
    mask[idx] = keep ? 1 : 0;
    float val = keep ? __half2float(input[idx]) * scale : 0.0f;
    output[idx] = __float2half(val);
}

template<typename T>
__global__ void dropout_backward_kernel_impl(
    const T* __restrict__ grad_output,
    const uint8_t* __restrict__ mask,
    T* __restrict__ grad_input,
    int64_t n,
    float scale) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    grad_input[idx] = mask[idx] ? static_cast<T>(static_cast<float>(grad_output[idx]) * scale) : T(0);
}

__global__ void dropout_backward_kernel_f16(
    const __half* __restrict__ grad_output,
    const uint8_t* __restrict__ mask,
    __half* __restrict__ grad_input,
    int64_t n,
    float scale) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = mask[idx] ? __half2float(grad_output[idx]) * scale : 0.0f;
    grad_input[idx] = __float2half(val);
}

// Dropout forward: returns {output, mask}
auto dropout_forward_kernel(const Tensor& input, float p, bool training, cudaStream_t stream)
    -> std::pair<Tensor, Tensor> {

    if (!training || p == 0.0f) {
        // No dropout during inference or p=0
        return {input, Tensor(std::vector<int64_t>{}, DType::UInt8, input.device())};
    }

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (p >= 1.0f) {
        // Drop everything
        Tensor output(shape, input.dtype(), input.device());
        cudaMemsetAsync(output.data_ptr(), 0, output.numel() * dtype_size(output.dtype()), stream);
        Tensor mask(shape, DType::UInt8, input.device());
        cudaMemsetAsync(mask.data_ptr(), 0, mask.numel(), stream);
        return {output, mask};
    }

    int64_t n = input.numel();
    float scale = 1.0f / (1.0f - p);

    Tensor output(shape, input.dtype(), input.device());
    Tensor mask(shape, DType::UInt8, input.device());

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // Generate seed from system entropy
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    switch (input.dtype()) {
        case DType::Float32:
            dropout_forward_kernel_impl<float><<<num_blocks, block_size, 0, stream>>>(
                input.data<float>(), output.data<float>(), mask.data<uint8_t>(),
                n, p, scale, seed);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Float64:
            dropout_forward_kernel_impl<double><<<num_blocks, block_size, 0, stream>>>(
                input.data<double>(), output.data<double>(), mask.data<uint8_t>(),
                n, p, scale, seed);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Float16:
            dropout_forward_kernel_f16<<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                reinterpret_cast<__half*>(output.data_ptr()),
                mask.data<uint8_t>(),
                n, p, scale, seed);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            dropout_forward_kernel_impl<__nv_bfloat16><<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                mask.data<uint8_t>(),
                n, p, scale, seed);
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("dropout_forward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return {output, mask};
}

// Dropout backward: grad_input = grad_output * mask * scale
auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p, cudaStream_t stream)
    -> Tensor {

    if (p == 0.0f) {
        return grad_output;
    }

    int64_t n = grad_output.numel();
    float scale = 1.0f / (1.0f - p);

    std::vector<int64_t> grad_shape(grad_output.shape().begin(), grad_output.shape().end());
    Tensor grad_input(grad_shape, grad_output.dtype(), grad_output.device());

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    switch (grad_output.dtype()) {
        case DType::Float32:
            dropout_backward_kernel_impl<float><<<num_blocks, block_size, 0, stream>>>(
                grad_output.data<float>(), mask.data<uint8_t>(), grad_input.data<float>(),
                n, scale);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Float64:
            dropout_backward_kernel_impl<double><<<num_blocks, block_size, 0, stream>>>(
                grad_output.data<double>(), mask.data<uint8_t>(), grad_input.data<double>(),
                n, scale);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Float16:
            dropout_backward_kernel_f16<<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data_ptr()),
                mask.data<uint8_t>(),
                reinterpret_cast<__half*>(grad_input.data_ptr()),
                n, scale);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            dropout_backward_kernel_impl<__nv_bfloat16><<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
                mask.data<uint8_t>(),
                reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()),
                n, scale);
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("dropout_backward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}
} // namespace cuda
} // namespace tenzor
