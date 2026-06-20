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
#include "tenzor/backend/fast_dispatch.hpp"  // dispatch-table types for kernel registration (device-only; no CPU fallback)
#include "cuda_launch_utils.cuh"
#include "cuda_common.cuh"
#include <stdexcept>
#include <vector>
#include <atomic>    // For per-call dropout seed counter
#include <charconv>  // For std::from_chars (dispatch wrappers)
#include <span>      // For std::span (dispatch wrappers)
#include "tenzor/ops/creation.hpp"  // For tenzor::get_global_seed (manual_seed reproducibility)

namespace tenzor {
namespace cuda {

// ============================================================================
// CUDA Helper Functions
// ============================================================================

// Centralized error checking
#include "../cuda_error.hpp"

// Default block size for element-wise operations (used as fallback)
constexpr int BLOCK_SIZE = 256;

// Block size for reduction operations (matches reduction.cu)
constexpr int REDUCTION_BLOCK_SIZE = 256;

// DimMeta for dim-specific reductions (matches reduction.cu).
//
// Audit F3: lifted from rank 8 to rank 16. The previous cap silently
// truncated higher-rank tensors — `make_dim_meta` filled only the first 8
// dims and the kernel iterated only those slots, so dimensions >8 were
// ignored. Modern transformer + 6D-attention models hit rank 9-12 routinely
// (e.g. block-sparse attention with batch + heads + Q-blocks + K-blocks +
// per-block H/W). 16 dims matches the maximum supported in any other
// backend (CPU + ROCm).
constexpr int DIM_META_MAX_RANK = 16;
struct DimMeta {
    int64_t shape[DIM_META_MAX_RANK];
    int64_t strides[DIM_META_MAX_RANK];
};

static DimMeta make_dim_meta(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
    if (shape.size() > DIM_META_MAX_RANK) {
        throw std::runtime_error(
            "CUDA DimMeta: tensor rank " + std::to_string(shape.size()) +
            " exceeds maximum " + std::to_string(DIM_META_MAX_RANK) +
            " (raise DIM_META_MAX_RANK if needed).");
    }
    DimMeta meta{};
    for (size_t i = 0; i < shape.size(); ++i) {
        meta.shape[i] = shape[i];
        meta.strides[i] = strides[i];
    }
    return meta;
}

// Accumulation type: use float for half/bfloat16 to prevent overflow
template<typename T> struct AccumType { using type = T; };
template<> struct AccumType<__half> { using type = float; };
template<> struct AccumType<__nv_bfloat16> { using type = float; };

// One-block-per-row grid (used by softmax / log_softmax forward+backward).
// `batch_size` is int64_t; a plain `int num_blocks = batch_size` would
// truncate to a wrong/negative value once the number of rows exceeds INT_MAX
// (~2.1e9), silently leaving most rows uncomputed. CUDA caps gridDim.x at
// 2^31-1, so validate before launch and return an unsigned grid extent.
static unsigned int softmax_grid_blocks(int64_t batch_size) {
    if (batch_size < 0) {
        throw std::runtime_error("softmax: negative batch size");
    }
    constexpr int64_t kMaxGridDimX = 2147483647LL;  // CUDA gridDim.x limit
    if (batch_size > kMaxGridDimX) {
        throw std::runtime_error(
            "softmax: number of rows " + std::to_string(batch_size) +
            " exceeds maximum CUDA grid dimension " + std::to_string(kMaxGridDimX));
    }
    return static_cast<unsigned int>(batch_size);
}

static auto compute_reduction_shape(
    const std::vector<int64_t>& input_shape,
    int64_t dim,
    bool keepdim
) -> std::vector<int64_t> {
    if (dim == INT64_MIN) {
        if (keepdim) return std::vector<int64_t>(input_shape.size(), 1);
        return {};
    }
    int64_t normalized_dim = dim;
    if (dim < 0) normalized_dim = static_cast<int64_t>(input_shape.size()) + dim;
    std::vector<int64_t> output_shape = input_shape;
    if (keepdim) {
        output_shape[normalized_dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + normalized_dim);
    }
    return output_shape;
}

// ============================================================================
// FP16 Saturation Tracking (opt-in via TENZOR_TRACK_SATURATION)
// ============================================================================
// When enabled, counts how many float→half conversions saturate to ±65504.
// This helps diagnose silent gradient clipping in mixed-precision training.
#ifdef TENZOR_TRACK_SATURATION
__device__ uint32_t g_fp16_saturation_count = 0;

__device__ __forceinline__ __half float2half_tracked(float val) {
    __half result = __float2half_rn(val);
    if (fabsf(val) > 65504.0f && !isinf(val) && !isnan(val)) {
        atomicAdd(&g_fp16_saturation_count, 1);
    }
    return result;
}

// Replace float2half_sat with tracked version when enabled
#define float2half_sat(x) float2half_tracked(x)
#endif // TENZOR_TRACK_SATURATION

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
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx] > T(0) ? input[idx] : T(0);
    }
}

// Backward: grad_out * (x > 0)
template<typename T>
__global__ void relu_backward_kernel(const T* grad_output, const T* input,
                                     T* grad_input, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        grad_input[idx] = grad_output[idx] * (input[idx] > T(0) ? T(1) : T(0));
    }
}

// Vectorized ReLU backward using float4 for 4x memory throughput
__global__ void relu_backward_vectorized_kernel(const float4* grad_output, const float4* input,
                                                float4* grad_input, int64_t n4) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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

// Helper device function for GELU computation — exact erf form
// (0.5 * x * (1 + erf(x / sqrt(2)))), matching PyTorch's default and the CPU
// backend (approximate='none').
__device__ __forceinline__ float gelu_scalar(float x) {
    constexpr float inv_sqrt2 = 0.70710678f;
    return x * 0.5f * (1.0f + erff(x * inv_sqrt2));
}

// Vectorized GELU forward using float4
__global__ void gelu_forward_vectorized_kernel(const float4* __restrict__ input,
                                                float4* __restrict__ output, int64_t n4) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    // Exact erf GELU derivative: 0.5*(1+erf(x/sqrt(2))) + x*(1/sqrt(2*pi))*exp(-x^2/2)
    constexpr float inv_sqrt2 = 0.70710678f;
    constexpr float pdf_coeff = 0.39894228f;  // 1/sqrt(2*pi)

    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
        float4 g = __ldg(&grad_output[idx]);
        float4 x = __ldg(&input[idx]);
        float4 result;

        #define GELU_BACKWARD(comp) \
            { \
                float cdf = 0.5f * (1.0f + erff(x.comp * inv_sqrt2)); \
                float pdf = pdf_coeff * expf(-0.5f * x.comp * x.comp); \
                result.comp = g.comp * (cdf + x.comp * pdf); \
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    void relu_forward_float(const float* input, float* output, int64_t n,
                            cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(relu_forward_kernel<float>, n);
        relu_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void relu_forward_double(const double* input, double* output, int64_t n,
                             cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(relu_forward_kernel<double>, n);
        relu_forward_kernel<double><<<grid_size, block_size, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void relu_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n, cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(relu_backward_kernel<float>, n);
        relu_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void relu_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n, cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(relu_backward_kernel<double>, n);
        relu_backward_kernel<double><<<grid_size, block_size, 0, stream>>>(
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = sigmoid_stable(input[idx]);
    }
}

// Backward: grad_out * sigmoid(x) * (1 - sigmoid(x))
template<typename T>
__global__ void sigmoid_backward_kernel(const T* grad_output, const T* input,
                                       T* grad_input, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        T sigmoid_x = sigmoid_stable(input[idx]);
        grad_input[idx] = grad_output[idx] * sigmoid_x * (T(1) - sigmoid_x);
    }
}

// Swish activation: swish(x) = x * sigmoid(x)
template<typename T>
__global__ void swish_forward_kernel(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        T x = input[idx];
        output[idx] = x * sigmoid_stable(x);
    }
}

// Swish backward: grad_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
//                = grad_out * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
template<typename T>
__global__ void swish_backward_cuda_kernel(const T* grad_output, const T* input,
                                            T* grad_input, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
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
    void sigmoid_forward_float(const float* input, float* output, int64_t n,
                               cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(sigmoid_forward_kernel<float>, n);
        sigmoid_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void sigmoid_forward_double(const double* input, double* output, int64_t n,
                                cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(sigmoid_forward_kernel<double>, n);
        sigmoid_forward_kernel<double><<<grid_size, block_size, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void sigmoid_backward_float(const float* grad_output, const float* input,
                               float* grad_input, int64_t n, cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(sigmoid_backward_kernel<float>, n);
        sigmoid_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void sigmoid_backward_double(const double* grad_output, const double* input,
                                double* grad_input, int64_t n, cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(sigmoid_backward_kernel<double>, n);
        sigmoid_backward_kernel<double><<<grid_size, block_size, 0, stream>>>(
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = tanh(input[idx]);
    }
}

// Specialization for __half
template<>
__global__ void tanh_forward_kernel<__half>(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(tanhf(__half2float(input[idx])));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void tanh_forward_kernel<__nv_bfloat16>(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2bfloat16(tanhf(__bfloat162float(input[idx])));
    }
}

// Backward: grad_out * (1 - tanh(x)^2)
template<typename T>
__global__ void tanh_backward_kernel(const T* grad_output, const T* input,
                                    T* grad_input, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        T tanh_x = tanh(input[idx]);
        grad_input[idx] = grad_output[idx] * (T(1) - tanh_x * tanh_x);
    }
}

// Specialization for __half
template<>
__global__ void tanh_backward_kernel<__half>(const __half* grad_output, const __half* input,
                                            __half* grad_input, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float tanh_x = tanhf(__half2float(input[idx]));
        grad_input[idx] = float2half_sat(__half2float(grad_output[idx]) * (1.0f - tanh_x * tanh_x));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void tanh_backward_kernel<__nv_bfloat16>(const __nv_bfloat16* grad_output, const __nv_bfloat16* input,
                                                   __nv_bfloat16* grad_input, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float tanh_x = tanhf(__bfloat162float(input[idx]));
        grad_input[idx] = __float2bfloat16(__bfloat162float(grad_output[idx]) * (1.0f - tanh_x * tanh_x));
    }
}

// Host functions
extern "C" {
    void tanh_forward_float(const float* input, float* output, int64_t n,
                            cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(tanh_forward_kernel<float>, n);
        tanh_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void tanh_forward_double(const double* input, double* output, int64_t n,
                             cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(tanh_forward_kernel<double>, n);
        tanh_forward_kernel<double><<<grid_size, block_size, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void tanh_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n, cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(tanh_backward_kernel<float>, n);
        tanh_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void tanh_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n, cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(tanh_backward_kernel<double>, n);
        tanh_backward_kernel<double><<<grid_size, block_size, 0, stream>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// GELU Activation
// ============================================================================

// Forward: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// Exact erf GELU: 0.5 * x * (1 + erf(x / sqrt(2))) — matches PyTorch default
// (approximate='none') and the CPU backend.
template<typename T>
__global__ void gelu_forward_kernel(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        T x = input[idx];
        output[idx] = x * T(0.5) * (T(1.0) + erf(x * T(0.70710678118654752)));
    }
}

// Specialization for __half
template<>
__global__ void gelu_forward_kernel<__half>(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(x * 0.5f * (1.0f + erff(x * 0.70710678f)));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void gelu_forward_kernel<__nv_bfloat16>(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(x * 0.5f * (1.0f + erff(x * 0.70710678f)));
    }
}

// Backward: grad_out * gelu'(x), exact erf derivative:
//   gelu'(x) = 0.5*(1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
template<typename T>
__global__ void gelu_backward_kernel(const T* grad_output, const T* input,
                                     T* grad_input, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        T x = input[idx];
        constexpr T inv_sqrt2 = T(0.70710678118654752);
        constexpr T pdf_coeff = T(0.39894228040143268);  // 1/sqrt(2*pi)
        T cdf = T(0.5) * (T(1.0) + erf(x * inv_sqrt2));
        T pdf = pdf_coeff * exp(T(-0.5) * x * x);
        grad_input[idx] = grad_output[idx] * (cdf + x * pdf);
    }
}

// Specialization for __half
template<>
__global__ void gelu_backward_kernel<__half>(const __half* grad_output, const __half* input,
                                              __half* grad_input, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __half2float(input[idx]);
        float cdf = 0.5f * (1.0f + erff(x * 0.70710678f));
        float pdf = 0.39894228f * expf(-0.5f * x * x);
        grad_input[idx] = float2half_sat(__half2float(grad_output[idx]) * (cdf + x * pdf));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void gelu_backward_kernel<__nv_bfloat16>(const __nv_bfloat16* grad_output, const __nv_bfloat16* input,
                                                    __nv_bfloat16* grad_input, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __bfloat162float(input[idx]);
        float cdf = 0.5f * (1.0f + erff(x * 0.70710678f));
        float pdf = 0.39894228f * expf(-0.5f * x * x);
        grad_input[idx] = __float2bfloat16(__bfloat162float(grad_output[idx]) * (cdf + x * pdf));
    }
}

// Host functions
extern "C" {
    void gelu_forward_float(const float* input, float* output, int64_t n,
                            cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(gelu_forward_kernel<float>, n);
        gelu_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void gelu_forward_double(const double* input, double* output, int64_t n,
                             cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(gelu_forward_kernel<double>, n);
        gelu_forward_kernel<double><<<grid_size, block_size, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void gelu_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n, cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(gelu_backward_kernel<float>, n);
        gelu_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void gelu_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n, cudaStream_t stream) {
        auto [grid_size, block_size] = optimal_launch_config(gelu_backward_kernel<double>, n);
        gelu_backward_kernel<double><<<grid_size, block_size, 0, stream>>>(
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx] > T(0) ? input[idx] : alpha * input[idx];
    }
}

// Float16 forward kernel: compute in Float32 for numerical stability
__global__ void leaky_relu_forward_fp16_kernel(const __half* input, __half* output,
                                                int64_t n, float alpha) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        output[idx] = __float2half(val > 0.0f ? val : alpha * val);
    }
}

// BFloat16 forward kernel: compute in Float32 so alpha is not pre-narrowed to
// bf16 (matches CPU and the Float16 GPU path).
__global__ void leaky_relu_forward_bf16_kernel(const __nv_bfloat16* input, __nv_bfloat16* output,
                                               int64_t n, float alpha) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(val > 0.0f ? val : alpha * val);
    }
}

// Backward: grad_out * (1 if x > 0 else alpha)
template<typename T>
__global__ void leaky_relu_backward_kernel(const T* grad_output, const T* input,
                                          T* grad_input, int64_t n, T alpha) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        grad_input[idx] = grad_output[idx] * (input[idx] > T(0) ? T(1) : alpha);
    }
}

// Float16 backward kernel: compute in Float32 for numerical stability
__global__ void leaky_relu_backward_fp16_kernel(const __half* grad_output, const __half* input,
                                                 __half* grad_input, int64_t n, float alpha) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float grad = __half2float(grad_output[idx]);
        float val = __half2float(input[idx]);
        grad_input[idx] = float2half_sat(grad * (val > 0.0f ? 1.0f : alpha));
    }
}

// BFloat16 backward kernel: compute in Float32 so alpha is not pre-narrowed to
// bf16 (matches CPU and the Float16 GPU path).
__global__ void leaky_relu_backward_bf16_kernel(const __nv_bfloat16* grad_output, const __nv_bfloat16* input,
                                                __nv_bfloat16* grad_input, int64_t n, float alpha) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float grad = __bfloat162float(grad_output[idx]);
        float val = __bfloat162float(input[idx]);
        grad_input[idx] = __float2bfloat16(grad * (val > 0.0f ? 1.0f : alpha));
    }
}

// Host functions
extern "C" {
    void leaky_relu_forward_float(const float* input, float* output,
                                 int64_t n, float alpha, cudaStream_t stream) {
        LAUNCH_KERNEL(leaky_relu_forward_kernel<float>, n, stream,
            input, output, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void leaky_relu_forward_double(const double* input, double* output,
                                  int64_t n, double alpha, cudaStream_t stream) {
        LAUNCH_KERNEL(leaky_relu_forward_kernel<double>, n, stream,
            input, output, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void leaky_relu_backward_float(const float* grad_output, const float* input,
                                  float* grad_input, int64_t n, float alpha,
                                  cudaStream_t stream) {
        LAUNCH_KERNEL(leaky_relu_backward_kernel<float>, n, stream,
            grad_output, input, grad_input, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void leaky_relu_backward_double(const double* grad_output, const double* input,
                                   double* grad_input, int64_t n, double alpha,
                                   cudaStream_t stream) {
        LAUNCH_KERNEL(leaky_relu_backward_kernel<double>, n, stream,
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
        if constexpr (std::is_same_v<T, double>) {
            output[idx] = (x > 0.0) ? x : static_cast<double>(alpha) * (exp(x) - 1.0);
        } else if constexpr (std::is_same_v<T, float>) {
            output[idx] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
        } else {
            output[idx] = (x > T(0)) ? x : T(alpha * (expf(float(x)) - 1.0f));
        }
    }
}

template<typename T>
__global__ void elu_backward_kernel(const T* grad_output, const T* input,
                                     T* grad_input, int64_t n, float alpha) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T x = input[idx];
        // Keep exp() in the native precision T — prior code narrowed to float
        // before the exp, silently dropping Float64 precision and failing
        // gradcheck at double-precision tolerances.
        T grad;
        if constexpr (std::is_same_v<T, double>) {
            grad = (x > T(0)) ? T(1) : T(static_cast<double>(alpha) * exp(x));
        } else if constexpr (std::is_same_v<T, float>) {
            grad = (x > T(0)) ? T(1) : T(alpha * expf(x));
        } else {
            // Half / BFloat16: widen to float for numeric stability.
            grad = (x > T(0)) ? T(1) : T(alpha * expf(float(x)));
        }
        grad_input[idx] = grad_output[idx] * grad;
    }
}

extern "C" {
    void elu_forward_float(const float* input, float* output, int64_t n, float alpha,
                           cudaStream_t stream) {
        LAUNCH_KERNEL(elu_forward_kernel<float>, n, stream, input, output, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void elu_forward_double(const double* input, double* output, int64_t n, float alpha,
                            cudaStream_t stream) {
        LAUNCH_KERNEL(elu_forward_kernel<double>, n, stream, input, output, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void elu_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n, float alpha,
                            cudaStream_t stream) {
        LAUNCH_KERNEL(elu_backward_kernel<float>, n, stream,
            grad_output, input, grad_input, n, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    void elu_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n, float alpha,
                             cudaStream_t stream) {
        LAUNCH_KERNEL(elu_backward_kernel<double>, n, stream,
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
        if constexpr (std::is_same_v<T, double>) {
            constexpr double ALPHA = 1.6732632423543772848170429916717;
            constexpr double SCALE = 1.0507009873554804934193349852946;
            double x = static_cast<double>(input[idx]);
            double result = (x > 0.0) ? x : ALPHA * (exp(x) - 1.0);
            output[idx] = SCALE * result;
        } else {
            float x = float(input[idx]);
            float result = (x > 0.0f) ? x : SELU_ALPHA * (expf(x) - 1.0f);
            output[idx] = T(SELU_SCALE * result);
        }
    }
}

template<typename T>
__global__ void selu_backward_kernel(const T* grad_output, const T* input,
                                      T* grad_input, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Compute in native precision T where possible so Float64 gradcheck
        // doesn't silently lose precision through a Float32 intermediate.
        if constexpr (std::is_same_v<T, double>) {
            double x = static_cast<double>(input[idx]);
            constexpr double ALPHA = 1.6732632423543772848170429916717;
            constexpr double SCALE = 1.0507009873554804934193349852946;
            double grad = (x > 0.0) ? SCALE : SCALE * ALPHA * exp(x);
            grad_input[idx] = grad_output[idx] * grad;
        } else {
            float x = float(input[idx]);
            float grad = (x > 0.0f) ? SELU_SCALE : SELU_SCALE * SELU_ALPHA * expf(x);
            grad_input[idx] = T(float(grad_output[idx]) * grad);
        }
    }
}

extern "C" {
    void selu_forward_float(const float* input, float* output, int64_t n,
                            cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        selu_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void selu_forward_double(const double* input, double* output, int64_t n,
                             cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        selu_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void selu_backward_float(const float* grad_output, const float* input,
                             float* grad_input, int64_t n, cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        selu_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void selu_backward_double(const double* grad_output, const double* input,
                              double* grad_input, int64_t n, cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        selu_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
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
        if constexpr (std::is_same_v<T, double>) {
            double x = static_cast<double>(input[idx]);
            double softplus;
            if (x > 20.0) softplus = x;
            else if (x < -20.0) softplus = exp(x);
            else softplus = log1p(exp(x));
            output[idx] = x * tanh(softplus);
        } else {
            float x = float(input[idx]);
            float softplus;
            if (x > 20.0f) softplus = x;
            else if (x < -20.0f) softplus = expf(x);
            else softplus = log1pf(expf(x));
            output[idx] = T(x * tanhf(softplus));
        }
    }
}

template<typename T>
__global__ void mish_backward_kernel(const T* grad_output, const T* input,
                                      T* grad_input, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<T, double>) {
            double x = static_cast<double>(input[idx]);
            double softplus;
            if (x > 20.0) {
                softplus = x;
            } else if (x < -20.0) {
                softplus = exp(x);
            } else {
                softplus = log1p(exp(x));
            }
            double tanh_sp = tanh(softplus);
            double sigmoid_x = 1.0 / (1.0 + exp(-x));
            double sech2 = 1.0 - tanh_sp * tanh_sp;
            double grad = tanh_sp + x * sech2 * sigmoid_x;
            grad_input[idx] = grad_output[idx] * grad;
        } else {
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
}

extern "C" {
    void mish_forward_float(const float* input, float* output, int64_t n,
                            cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        mish_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void mish_forward_double(const double* input, double* output, int64_t n,
                             cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        mish_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void mish_backward_float(const float* grad_output, const float* input,
                             float* grad_input, int64_t n, cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        mish_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output, input, grad_input, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void mish_backward_double(const double* grad_output, const double* input,
                              double* grad_input, int64_t n, cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        mish_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
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
        if constexpr (std::is_same_v<T, double>) {
            double in = static_cast<double>(input[idx]);
            double b = static_cast<double>(beta);
            double th = static_cast<double>(threshold);
            double x = in * b;
            double result;
            if (x > th) result = in;
            else if (x < -th) result = exp(x) / b;
            else result = log1p(exp(x)) / b;
            output[idx] = result;
        } else {
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
}

template<typename T>
__global__ void softplus_backward_kernel(const T* grad_output, const T* input,
                                          T* grad_input, int64_t n,
                                          float beta, float threshold) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<T, double>) {
            double x = static_cast<double>(input[idx]) * static_cast<double>(beta);
            double sigmoid_x;
            if (x > static_cast<double>(threshold)) {
                sigmoid_x = 1.0;
            } else if (x < -static_cast<double>(threshold)) {
                sigmoid_x = exp(x);
            } else {
                sigmoid_x = 1.0 / (1.0 + exp(-x));
            }
            grad_input[idx] = grad_output[idx] * sigmoid_x;
        } else {
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
}

extern "C" {
    void softplus_forward_float(const float* input, float* output, int64_t n,
                                float beta, float threshold, cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        softplus_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    void softplus_forward_double(const double* input, double* output, int64_t n,
                                 float beta, float threshold, cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        softplus_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    void softplus_backward_float(const float* grad_output, const float* input,
                                 float* grad_input, int64_t n,
                                 float beta, float threshold, cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        softplus_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output, input, grad_input, n, beta, threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    void softplus_backward_double(const double* grad_output, const double* input,
                                  double* grad_input, int64_t n,
                                  float beta, float threshold, cudaStream_t stream) {
        int num_blocks = get_num_blocks(n);
        softplus_backward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
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

    val = (threadIdx.x < blockDim.x / 32) ? shared[threadIdx.x] : numeric_min<T>();
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

    // Step 2: Compute exp(x - max) and sum.
    // Accumulate the exp-sum in AccumType<T> (float for half/bf16) so that the
    // reduction does not saturate or lose precision in narrow dtypes; mirrors
    // nansum_along_dim_kernel. A dedicated Acc-typed shared buffer is used so we
    // do not alias the T-typed `shared` array (which has a smaller element size
    // for half/bf16). blockDim.x <= 1024 -> at most 32 warps.
    using Acc = typename AccumType<T>::type;
    __shared__ Acc acc_shared[32];
    Acc sum_exp = Acc(0);
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        Acc exp_val = device_exp(static_cast<Acc>(input_row[i]) - static_cast<Acc>(max_val));
        output_row[i] = static_cast<T>(exp_val);
        sum_exp += exp_val;
    }
    sum_exp = block_reduce_sum<Acc>(sum_exp, acc_shared);
    __syncthreads();

    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        acc_shared[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = acc_shared[0];

    // Step 3: Normalize
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        output_row[i] = static_cast<T>(static_cast<Acc>(output_row[i]) / sum_exp);
    }
}

// Softmax backward: softmax[i] * (grad_output[i] - sum(grad_output * softmax))
template<typename T>
__global__ void softmax_backward_kernel(const T* grad_output, const T* output,
                                       T* grad_input,
                                       int64_t batch_size, int64_t dim_size) {
    // The reduction below uses a statically-allocated Acc-typed shared buffer
    // (acc_shared); the dynamic shared-mem launch param is unused here.
    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* grad_out_row = grad_output + row * dim_size;
    const T* out_row = output + row * dim_size;
    T* grad_in_row = grad_input + row * dim_size;

    // Compute sum(grad_output * softmax) in AccumType<T> (float for half/bf16)
    // so the per-row reduction does not saturate or lose precision in narrow
    // dtypes; mirrors softmax_forward_kernel's Acc handling. A dedicated
    // Acc-typed shared buffer avoids aliasing the (smaller) T-typed `shared`.
    using Acc = typename AccumType<T>::type;
    __shared__ Acc acc_shared[32];
    Acc sum = Acc(0);
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum += static_cast<Acc>(grad_out_row[i]) * static_cast<Acc>(out_row[i]);
    }
    sum = block_reduce_sum<Acc>(sum, acc_shared);
    __syncthreads();

    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        acc_shared[0] = sum;
    }
    __syncthreads();
    sum = acc_shared[0];

    // Compute gradient: softmax[i] * (grad_output[i] - sum)
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        grad_in_row[i] = static_cast<T>(static_cast<Acc>(out_row[i]) *
                                        (static_cast<Acc>(grad_out_row[i]) - sum));
    }
}

// Host functions
extern "C" {
    void softmax_forward_float(const float* input, float* output,
                              int64_t batch_size, int64_t dim_size,
                              cudaStream_t stream) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        softmax_forward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input, output, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void softmax_forward_double(const double* input, double* output,
                               int64_t batch_size, int64_t dim_size,
                               cudaStream_t stream) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        softmax_forward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input, output, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void softmax_backward_float(const float* grad_output, const float* output,
                               float* grad_input,
                               int64_t batch_size, int64_t dim_size,
                               cudaStream_t stream) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        softmax_backward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output, output, grad_input, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void softmax_backward_double(const double* grad_output, const double* output,
                                double* grad_input,
                                int64_t batch_size, int64_t dim_size,
                                cudaStream_t stream) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        softmax_backward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
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

    // Step 2: Compute sum(exp(x - max)).
    // Accumulate and reduce in AccumType<T> (float for half/bf16) to avoid
    // narrow-dtype saturation/precision loss; mirrors nansum_along_dim_kernel.
    using Acc = typename AccumType<T>::type;
    __shared__ Acc acc_shared[32];
    Acc sum_exp = Acc(0);
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum_exp += device_exp(static_cast<Acc>(input_row[i]) - static_cast<Acc>(max_val));
    }
    sum_exp = block_reduce_sum<Acc>(sum_exp, acc_shared);
    __syncthreads();

    // Broadcast log-sum to all threads (computed in Acc precision)
    if (threadIdx.x == 0) {
        acc_shared[0] = (sum_exp > Acc(0)) ? device_log(sum_exp) : Acc(-1e30);
    }
    __syncthreads();
    Acc log_sum_exp = acc_shared[0];

    // Step 3: Compute log_softmax = x - max - log_sum_exp
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        output_row[i] = static_cast<T>(static_cast<Acc>(input_row[i]) - static_cast<Acc>(max_val) - log_sum_exp);
    }
}

// LogSoftmax backward: grad_output - exp(log_softmax) * sum(grad_output)
template<typename T>
__global__ void log_softmax_backward_kernel(const T* grad_output, const T* output,
                                           T* grad_input,
                                           int64_t batch_size, int64_t dim_size) {
    // The reduction below uses a statically-allocated Acc-typed shared buffer
    // (acc_shared); the dynamic shared-mem launch param is unused here.
    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* grad_out_row = grad_output + row * dim_size;
    const T* out_row = output + row * dim_size;
    T* grad_in_row = grad_input + row * dim_size;

    // Compute sum(grad_output) in AccumType<T> (float for half/bf16) so the
    // grad-sum reduction and the exp() are done in widened precision, avoiding
    // narrow-dtype saturation/precision loss on large dims; mirrors
    // log_softmax_forward_kernel's Acc handling.
    using Acc = typename AccumType<T>::type;
    __shared__ Acc acc_shared[32];
    Acc sum_grad = Acc(0);
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum_grad += static_cast<Acc>(grad_out_row[i]);
    }
    sum_grad = block_reduce_sum<Acc>(sum_grad, acc_shared);
    __syncthreads();

    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        acc_shared[0] = sum_grad;
    }
    __syncthreads();
    sum_grad = acc_shared[0];

    // Compute gradient: grad_output - exp(log_softmax) * sum_grad
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        grad_in_row[i] = static_cast<T>(static_cast<Acc>(grad_out_row[i]) -
                                        device_exp(static_cast<Acc>(out_row[i])) * sum_grad);
    }
}

// Host functions
extern "C" {
    void log_softmax_forward_float(const float* input, float* output,
                                   int64_t batch_size, int64_t dim_size,
                                   cudaStream_t stream) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        log_softmax_forward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input, output, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void log_softmax_forward_double(const double* input, double* output,
                                    int64_t batch_size, int64_t dim_size,
                                    cudaStream_t stream) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        log_softmax_forward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input, output, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void log_softmax_backward_float(const float* grad_output, const float* output,
                                    float* grad_input,
                                    int64_t batch_size, int64_t dim_size,
                                    cudaStream_t stream) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        log_softmax_backward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output, output, grad_input, batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    }

    void log_softmax_backward_double(const double* grad_output, const double* output,
                                     double* grad_input,
                                     int64_t batch_size, int64_t dim_size,
                                     cudaStream_t stream) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        log_softmax_backward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = data[idx] > T(0) ? data[idx] : T(0);
    }
}

// Vectorized in-place ReLU using float4
__global__ void relu_inplace_vectorized_kernel(float4* __restrict__ data, int64_t n4) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n4) {
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
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = T(1) / (T(1) + device_exp(-data[idx]));
    }
}

// In-place Tanh: x = tanh(x)
template<typename T>
__global__ void tanh_inplace_cuda_kernel(T* data, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = tanh(data[idx]);
    }
}

// Specialization for __half
template<>
__global__ void tanh_inplace_cuda_kernel<__half>(__half* data, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        // Convert to float, apply tanh, convert back
        float val = __half2float(data[idx]);
        data[idx] = __float2half(tanhf(val));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void tanh_inplace_cuda_kernel<__nv_bfloat16>(__nv_bfloat16* data, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        // Convert to float, apply tanh, convert back
        float val = __bfloat162float(data[idx]);
        data[idx] = __float2bfloat16(tanhf(val));
    }
}

// In-place LeakyReLU: x = max(alpha * x, x)
template<typename T>
__global__ void leaky_relu_inplace_cuda_kernel(T* data, T alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        T val = data[idx];
        data[idx] = val > T(0) ? val : alpha * val;
    }
}

// Float16 inplace kernel: compute in Float32 for numerical stability
__global__ void leaky_relu_inplace_fp16_kernel(__half* data, float alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = __half2float(data[idx]);
        data[idx] = __float2half(val > 0.0f ? val : alpha * val);
    }
}

// BFloat16 inplace kernel: take float alpha and compute in Float32 so the
// negative-slope multiply is not done with a bf16-rounded alpha (mirrors the
// Float16 inplace kernel and the CPU reference).
__global__ void leaky_relu_inplace_bf16_kernel(__nv_bfloat16* data, float alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = __bfloat162float(data[idx]);
        data[idx] = __float2bfloat16(val > 0.0f ? val : alpha * val);
    }
}

// In-place GELU — exact erf form 0.5 * x * (1 + erf(x / sqrt(2))), matching
// PyTorch's default ('none'), the out-of-place CUDA kernel above, and every
// other backend. The previous tanh approximation differed from the exact
// form by up to ~5e-4, which broke cross-backend parity (InplaceOpsParity).
template<typename T>
__global__ void gelu_inplace_cuda_kernel(T* data, int64_t n) {
    constexpr T inv_sqrt2 = T(0.7071067811865475);

    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        T x = data[idx];
        data[idx] = T(0.5) * x * (T(1) + erf(x * inv_sqrt2));
    }
}

// Specialization for __half
template<>
__global__ void gelu_inplace_cuda_kernel<__half>(__half* data, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __half2float(data[idx]);
        data[idx] = __float2half(0.5f * x * (1.0f + erff(x * 0.7071067811865475f)));
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void gelu_inplace_cuda_kernel<__nv_bfloat16>(__nv_bfloat16* data, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __bfloat162float(data[idx]);
        data[idx] = __float2bfloat16(0.5f * x * (1.0f + erff(x * 0.7071067811865475f)));
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
        leaky_relu_inplace_bf16_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr()), alpha, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("LeakyReLU inplace only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
}

// ============================================================================
// Tensor Wrapper Functions
// ============================================================================

// ReLU wrapper - uses vectorized float4 for 4x memory throughput on Float32
auto relu_kernel(const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    // Kernels index linearly (scalar or float4) assuming a packed layout, so
    // non-contiguous (transposed/strided/view) input must be materialized first
    // or we read the wrong elements. Mirrors sigmoid_kernel (audit-2026-05-03 #15).
    auto input = input_raw.contiguous();
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
auto relu_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    // Linear-indexing kernels require packed layouts; contiguify both operands
    // so strided views read correctly. Mirrors sigmoid_kernel.
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
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

        // Only take the float4 path when all three pointers are 16-byte
        // aligned (sub-tensor views/offsets can break alignment); otherwise
        // a float4 access faults. Mirrors sigmoid_backward / relu forward.
        if (n >= VECTORIZED_THRESHOLD &&
            reinterpret_cast<uintptr_t>(grad_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(result_ptr) % 16 == 0) {

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
        } else {
            // Fallback to scalar kernel for small or misaligned tensors
            int num_blocks = get_num_blocks(n);
            relu_backward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                grad_ptr, input_ptr, result_ptr, n);
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
auto sigmoid_kernel(const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure contiguous input.
    auto input = input_raw.contiguous();
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
auto sigmoid_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    // Materialize contiguous copies: the kernels index flat, so a strided/
    // transposed grad_output/input would otherwise read the wrong elements.
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
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
auto swish_kernel(const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    auto input = input_raw.contiguous();
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
auto swish_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    // Materialize contiguous copies: the kernels index flat, so a strided/
    // transposed grad_output/input would otherwise read the wrong elements.
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
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
auto tanh_kernel(const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure contiguous input.
    auto input = input_raw.contiguous();
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
auto tanh_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    // Materialize contiguous copies: the kernels index flat, so a strided/
    // transposed grad_output/input would otherwise read the wrong elements.
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
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
auto gelu_kernel(const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    auto input = input_raw.contiguous();
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
auto gelu_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    // Materialize contiguous copies: the kernels index flat, so a strided/
    // transposed grad_output/input would otherwise read the wrong elements.
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
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
auto leaky_relu_kernel(const Tensor& input_raw, double alpha, cudaStream_t stream) -> Tensor {
    auto input = input_raw.contiguous();
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
        leaky_relu_forward_bf16_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n, static_cast<float>(alpha));
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
auto leaky_relu_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, double alpha, cudaStream_t stream) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
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
        leaky_relu_backward_bf16_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n, static_cast<float>(alpha));
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
auto elu_kernel(const Tensor& input_raw, float alpha, cudaStream_t stream) -> Tensor {
    auto input = input_raw.contiguous();
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
auto elu_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, float alpha, cudaStream_t stream) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
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
auto selu_kernel(const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    auto input = input_raw.contiguous();
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
auto selu_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    // Materialize contiguous copies: the kernels index flat, so a strided/
    // transposed grad_output/input would otherwise read the wrong elements.
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
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
auto mish_kernel(const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    auto input = input_raw.contiguous();
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
auto mish_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    // Materialize contiguous copies: the kernels index flat, so a strided/
    // transposed grad_output/input would otherwise read the wrong elements.
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
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
auto softplus_kernel(const Tensor& input_raw, float beta, float threshold, cudaStream_t stream) -> Tensor {
    auto input = input_raw.contiguous();
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
auto softplus_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, float beta, float threshold, cudaStream_t stream) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
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
        TENZOR_CUDA_CHECK(cudaMemsetAsync(result.data_ptr(), 0, result.numel() * sizeof(float), stream));
    } else if (input.dtype() == DType::Float64) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(result.data_ptr(), 0, result.numel() * sizeof(double), stream));
    } else if (input.dtype() == DType::Float16) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(result.data_ptr(), 0, result.numel() * sizeof(__half), stream));
    } else if (input.dtype() == DType::BFloat16) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(result.data_ptr(), 0, result.numel() * sizeof(__nv_bfloat16), stream));
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

    // The kernel below flattens to a contiguous [batch, dim_size] layout with
    // the softmax axis innermost, which is only valid for a CONTIGUOUS tensor
    // whose softmax axis is the last dimension. For a non-contiguous input
    // (e.g. a transposed view) or a non-last axis it read the wrong elements
    // (the stride-from-shape audit bug). Normalise by moving `dim` to the last
    // axis and materialising a contiguous copy, then transpose the result back.
    const int64_t last = ndim - 1;
    if (dim != last || !input.is_contiguous()) {
        Tensor norm = (dim == last) ? input.contiguous()
                                    : input.transpose(dim, last).contiguous();
        Tensor out = softmax_kernel(norm, last, stream);
        return (dim == last) ? out : out.transpose(dim, last).contiguous();
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
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        softmax_forward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<float>(), result.data<float>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        softmax_forward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<double>(), result.data<double>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__half);
        softmax_forward_kernel<__half><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
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

    // Stride-from-shape normalisation (see softmax_kernel): the flattened
    // [batch, dim_size] backward kernel requires both grad_output and output to
    // be contiguous with the softmax axis last.
    const int64_t last = ndim - 1;
    if (dim != last || !grad_output.is_contiguous() || !output.is_contiguous()) {
        Tensor g = (dim == last) ? grad_output.contiguous()
                                 : grad_output.transpose(dim, last).contiguous();
        Tensor o = (dim == last) ? output.contiguous()
                                 : output.transpose(dim, last).contiguous();
        Tensor gi = softmax_backward_kernel(g, o, last, stream);
        return (dim == last) ? gi : gi.transpose(dim, last).contiguous();
    }

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        batch_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        batch_size *= shape[i];
    }

    if (output.dtype() == DType::Float32) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        softmax_backward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<float>(), output.data<float>(), result.data<float>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::Float64) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        softmax_backward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<double>(), output.data<double>(), result.data<double>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::Float16) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__half);
        softmax_backward_kernel<__half><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(output.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::BFloat16) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
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

    // Same stride-from-shape normalisation as softmax_kernel: the flattened
    // [batch, dim_size] kernel is only valid for a contiguous, last-axis input.
    const int64_t last = ndim - 1;
    if (dim != last || !input.is_contiguous()) {
        Tensor norm = (dim == last) ? input.contiguous()
                                    : input.transpose(dim, last).contiguous();
        Tensor out = log_softmax_kernel(norm, last, stream);
        return (dim == last) ? out : out.transpose(dim, last).contiguous();
    }

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        batch_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        batch_size *= shape[i];
    }

    if (input.dtype() == DType::Float32) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        log_softmax_forward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<float>(), result.data<float>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        log_softmax_forward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<double>(), result.data<double>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__half);
        log_softmax_forward_kernel<__half><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
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

    // Stride-from-shape normalisation (see softmax_kernel): the flattened
    // [batch, dim_size] backward kernel requires both grad_output and output to
    // be contiguous with the softmax axis last.
    const int64_t last = ndim - 1;
    if (dim != last || !grad_output.is_contiguous() || !output.is_contiguous()) {
        Tensor g = (dim == last) ? grad_output.contiguous()
                                 : grad_output.transpose(dim, last).contiguous();
        Tensor o = (dim == last) ? output.contiguous()
                                 : output.transpose(dim, last).contiguous();
        Tensor gi = log_softmax_backward_kernel(g, o, last, stream);
        return (dim == last) ? gi : gi.transpose(dim, last).contiguous();
    }

    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        batch_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        batch_size *= shape[i];
    }

    if (output.dtype() == DType::Float32) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        log_softmax_backward_kernel<float><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<float>(), output.data<float>(), result.data<float>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::Float64) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        log_softmax_backward_kernel<double><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<double>(), output.data<double>(), result.data<double>(), batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::Float16) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(__half);
        log_softmax_backward_kernel<__half><<<num_blocks, SOFTMAX_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(output.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            batch_size, dim_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (output.dtype() == DType::BFloat16) {
        unsigned int num_blocks = softmax_grid_blocks(batch_size);
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
// Hardswish / Hardsigmoid — forward-only single-output kernels.
//   hardswish(x)   = x * clamp(x+3, 0, 6) / 6
//   hardsigmoid(x) = clamp(x+3, 0, 6) / 6
// Backward is autograd-composed (clamp + mul chain), matching the CPU backend;
// there is no dedicated *Backward OpId. Float32/Float64 native; Float16/BF16
// widen to Float32 for the math then narrow on store.
// ============================================================================
template<typename T>
__global__ void hardswish_forward_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<T, double>) {
            double x = static_cast<double>(input[idx]);
            double t = fmin(fmax(x + 3.0, 0.0), 6.0);
            output[idx] = x * t * (1.0 / 6.0);
        } else {
            float x = float(input[idx]);
            float t = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
            output[idx] = T(x * t * (1.0f / 6.0f));
        }
    }
}

template<typename T>
__global__ void hardsigmoid_forward_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<T, double>) {
            double x = static_cast<double>(input[idx]);
            double t = fmin(fmax(x + 3.0, 0.0), 6.0);
            output[idx] = t * (1.0 / 6.0);
        } else {
            float x = float(input[idx]);
            float t = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
            output[idx] = T(t * (1.0f / 6.0f));
        }
    }
}

auto hardswish_kernel(const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    auto input = input_raw.contiguous();
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    int num_blocks = get_num_blocks(n);
    if (input.dtype() == DType::Float32) {
        hardswish_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hardswish_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hardswish_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
    } else if (input.dtype() == DType::BFloat16) {
        hardswish_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
    } else {
        throw std::runtime_error("Hardswish only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

auto hardsigmoid_kernel(const Tensor& input_raw, cudaStream_t stream) -> Tensor {
    auto input = input_raw.contiguous();
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    int num_blocks = get_num_blocks(n);
    if (input.dtype() == DType::Float32) {
        hardsigmoid_forward_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hardsigmoid_forward_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hardsigmoid_forward_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()), n);
    } else if (input.dtype() == DType::BFloat16) {
        hardsigmoid_forward_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()), n);
    } else {
        throw std::runtime_error("Hardsigmoid only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor hardswish_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return hardswish_kernel(inputs[0], get_stream(attrs));
}

Tensor hardsigmoid_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    return hardsigmoid_kernel(inputs[0], get_stream(attrs));
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
    output[idx] = float2half_sat(val);
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
    grad_input[idx] = float2half_sat(val);
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
        CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, output.numel() * dtype_size(output.dtype()), stream));
        Tensor mask(shape, DType::UInt8, input.device());
        CUDA_CHECK(cudaMemsetAsync(mask.data_ptr(), 0, mask.numel(), stream));
        return {output, mask};
    }

    int64_t n = input.numel();
    float scale = 1.0f / (1.0f - p);

    Tensor output(shape, input.dtype(), input.device());
    Tensor mask(shape, DType::UInt8, input.device());

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // Seed from the library's global RNG so manual_seed is honored / dropout is
    // reproducible. Fold in a process-wide monotonic counter so two invocations
    // within the same clock tick (the un-seeded, time-based case) cannot collide
    // and produce identical masks. SplitMix64-style mixing keeps the counter
    // from merely perturbing the low bits.
    static std::atomic<uint64_t> dropout_call_counter{0};
    uint64_t mix = dropout_call_counter.fetch_add(1, std::memory_order_relaxed);
    mix = (mix + 0x9E3779B97F4A7C15ULL);
    mix = (mix ^ (mix >> 30)) * 0xBF58476D1CE4E5B9ULL;
    mix = (mix ^ (mix >> 27)) * 0x94D049BB133111EBULL;
    mix = mix ^ (mix >> 31);
    uint64_t seed = ::tenzor::get_global_seed() ^ mix;

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
// ============================================================================
// FP16 Saturation Tracking - Host Query Functions
// ============================================================================
// RReLU: Randomized Leaky ReLU
// ============================================================================

__global__ void rrelu_eval_f32(const float* input, float* output, int64_t n, float slope) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = input[idx];
        output[idx] = (x >= 0.0f) ? x : slope * x;
    }
}

__global__ void rrelu_train_f32(const float* input, float* output, int64_t n,
                                 float lower, float upper, unsigned long long seed) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = input[idx];
        if (x >= 0.0f) {
            output[idx] = x;
        } else {
            unsigned long long state = seed + static_cast<unsigned long long>(idx) * 6364136223846793005ULL;
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
            float slope = lower + u * (upper - lower);
            output[idx] = slope * x;
        }
    }
}

__global__ void rrelu_backward_f32(const float* grad, const float* input, float* output,
                                    int64_t n, float slope) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = (input[idx] >= 0.0f) ? grad[idx] : slope * grad[idx];
    }
}

Tensor rrelu_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
    float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
    bool training = attrs.get_bool(AttrKey::Training, false);
    float mid = (lower + upper) / 2.0f;

    int64_t n = inputs[0].numel();
    std::vector<int64_t> shape(inputs[0].shape().begin(), inputs[0].shape().end());
    Tensor result(shape, inputs[0].dtype(), inputs[0].device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);

    if (inputs[0].dtype() == DType::Float32) {
        if (training) {
            // Seed from the library's global RNG so manual_seed is honored /
            // RReLU is reproducible. Fold in a process-wide monotonic counter
            // with SplitMix64 mixing so two invocations within the same clock
            // tick cannot collide, exactly as dropout_forward_kernel does.
            static std::atomic<uint64_t> rrelu_call_counter{0};
            uint64_t mix = rrelu_call_counter.fetch_add(1, std::memory_order_relaxed);
            mix = (mix + 0x9E3779B97F4A7C15ULL);
            mix = (mix ^ (mix >> 30)) * 0xBF58476D1CE4E5B9ULL;
            mix = (mix ^ (mix >> 27)) * 0x94D049BB133111EBULL;
            mix = mix ^ (mix >> 31);
            unsigned long long seed =
                static_cast<unsigned long long>(::tenzor::get_global_seed() ^ mix);
            rrelu_train_f32<<<grid, block, 0, stream>>>(
                inputs[0].data<float>(), result.data<float>(), n, lower, upper, seed);
        } else {
            rrelu_eval_f32<<<grid, block, 0, stream>>>(
                inputs[0].data<float>(), result.data<float>(), n, mid);
        }
    } else {
        auto f32 = inputs[0].to(DType::Float32);
        std::array<Tensor, 1> tmp = {f32};
        auto r = rrelu_dispatch(tmp, attrs);
        return r.to(inputs[0].dtype());
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor rrelu_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
    float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
    float mid = (lower + upper) / 2.0f;

    int64_t n = inputs[0].numel();
    std::vector<int64_t> shape(inputs[0].shape().begin(), inputs[0].shape().end());
    Tensor result(shape, inputs[0].dtype(), inputs[0].device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);

    if (inputs[0].dtype() == DType::Float32) {
        rrelu_backward_f32<<<grid, block, 0, stream>>>(
            inputs[0].data<float>(), inputs[1].data<float>(), result.data<float>(), n, mid);
    } else {
        auto f32_g = inputs[0].to(DType::Float32);
        auto f32_in = inputs[1].to(DType::Float32);
        std::array<Tensor, 2> tmp = {f32_g, f32_in};
        auto r = rrelu_backward_dispatch(tmp, attrs);
        return r.to(inputs[0].dtype());
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// LogSigmoid backward
__global__ void log_sigmoid_backward_f32(const float* grad, const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = input[idx];
        float sig_neg_x = (x >= 0.0f) ? expf(-x) / (1.0f + expf(-x)) : 1.0f / (1.0f + expf(x));
        output[idx] = grad[idx] * sig_neg_x;
    }
}

Tensor log_sigmoid_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    int64_t n = inputs[0].numel();
    std::vector<int64_t> shape(inputs[0].shape().begin(), inputs[0].shape().end());
    Tensor result(shape, inputs[0].dtype(), inputs[0].device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (inputs[0].dtype() == DType::Float32) {
        log_sigmoid_backward_f32<<<grid, block, 0, stream>>>(
            inputs[0].data<float>(), inputs[1].data<float>(), result.data<float>(), n);
    } else {
        auto f32_g = inputs[0].to(DType::Float32);
        auto f32_in = inputs[1].to(DType::Float32);
        std::array<Tensor, 2> tmp = {f32_g, f32_in};
        auto r = log_sigmoid_backward_dispatch(tmp, attrs);
        return r.to(inputs[0].dtype());
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// NaN-aware reductions
__global__ void count_nonzero_all_f32(const float* input, int64_t* output, int64_t n) {
    __shared__ int64_t scount;
    if (threadIdx.x == 0) scount = 0;
    __syncthreads();
    int64_t local_count = 0;
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        if (input[idx] != 0.0f) local_count++;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&scount), static_cast<unsigned long long>(local_count));
    __syncthreads();
    if (threadIdx.x == 0) output[0] = scount;
}

// Native (no-downcast) full-reduction count-nonzero so Float64/Int32/Int64 do
// not lose precision via a Float32 cast (a tiny double would underflow to 0.0f).
template <typename T>
__global__ void count_nonzero_all_native(const T* input, int64_t* output, int64_t n) {
    __shared__ int64_t scount;
    if (threadIdx.x == 0) scount = 0;
    __syncthreads();
    int64_t local_count = 0;
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        if (input[idx] != T(0)) local_count++;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&scount), static_cast<unsigned long long>(local_count));
    __syncthreads();
    if (threadIdx.x == 0) output[0] = scount;
}

// ============================================================================
// Dim-specific count_nonzero reduction kernel (eliminates CPU fallback)
// ============================================================================
template<typename T>
__global__ void count_nonzero_along_dim_kernel(
    const T* input,
    int64_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    int64_t count = 0;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        if (static_cast<float>(input[in_idx]) != 0.0f) {
            count++;
        }
    }
    output[out_idx] = count;
}

template<typename T>
static void launch_dim_count_nonzero(
    const T* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) output_size *= input_shape[i];
    }
    if (output_size == 0 || dim_size == 0) return;
    DimMeta meta = make_dim_meta(input_shape, input_strides);
    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    count_nonzero_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Dim-specific nansum reduction kernel (eliminates CPU fallback)
// ============================================================================
template<typename T>
__global__ void nansum_along_dim_kernel(
    const T* input,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    using Acc = typename AccumType<T>::type;
    Acc sum = Acc(0);
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        Acc v = Acc(input[in_idx]);
        if (!isnan(static_cast<float>(v))) {
            sum = sum + v;
        }
    }
    output[out_idx] = T(sum);
}

template<typename T>
static void launch_dim_nansum(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) output_size *= input_shape[i];
    }
    if (output_size == 0 || dim_size == 0) return;
    DimMeta meta = make_dim_meta(input_shape, input_strides);
    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    nansum_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

Tensor count_nonzero_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    if (dim < 0) {
        // Full reduction. Keep Float64/Int32/Int64 native so a value below the
        // float subnormal range is not underflowed to 0.0f by a downcast; only
        // upcast genuinely lower-precision dtypes to Float32. Mirrors the
        // dim-specific branch below.
        Tensor input = inputs[0];
        int64_t n = input.numel();
        Tensor result({1}, DType::Int64, input.device());
        switch (input.dtype()) {
            case DType::Float64:
                count_nonzero_all_native<double><<<1, 256, 0, stream>>>(
                    input.data<double>(), result.data<int64_t>(), n);
                break;
            case DType::Int32:
                count_nonzero_all_native<int32_t><<<1, 256, 0, stream>>>(
                    input.data<int32_t>(), result.data<int64_t>(), n);
                break;
            case DType::Int64:
                count_nonzero_all_native<int64_t><<<1, 256, 0, stream>>>(
                    input.data<int64_t>(), result.data<int64_t>(), n);
                break;
            case DType::Float32:
                count_nonzero_all_f32<<<1, 256, 0, stream>>>(
                    input.data<float>(), result.data<int64_t>(), n);
                break;
            default: {
                Tensor f32 = input.to(DType::Float32);
                count_nonzero_all_f32<<<1, 256, 0, stream>>>(
                    f32.data<float>(), result.data<int64_t>(), n);
                break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    // Dim-specific reduction: native CUDA kernel
    Tensor input = inputs[0];
    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    int64_t normalized_dim = dim;
    if (dim < 0) normalized_dim = ndim + dim;

    // Compute output shape (remove reduced dim)
    std::vector<int64_t> output_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != normalized_dim) output_shape.push_back(input_shape[d]);
    }
    if (output_shape.empty()) output_shape.push_back(1);

    Tensor result(output_shape, DType::Int64, input.device());

    // Upcast to Float32 for comparison if needed
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64 &&
        input.dtype() != DType::Int32 && input.dtype() != DType::Int64) {
        input = input.to(DType::Float32);
    }

    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto strides_vec = std::vector<int64_t>(input.strides().begin(), input.strides().end());

    if (input.dtype() == DType::Float32) {
        launch_dim_count_nonzero(input.data<float>(), result.data<int64_t>(), shape_vec, strides_vec, normalized_dim);
    } else if (input.dtype() == DType::Float64) {
        launch_dim_count_nonzero(input.data<double>(), result.data<int64_t>(), shape_vec, strides_vec, normalized_dim);
    } else if (input.dtype() == DType::Int32) {
        launch_dim_count_nonzero(input.data<int32_t>(), result.data<int64_t>(), shape_vec, strides_vec, normalized_dim);
    } else if (input.dtype() == DType::Int64) {
        launch_dim_count_nonzero(input.data<int64_t>(), result.data<int64_t>(), shape_vec, strides_vec, normalized_dim);
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

__global__ void nansum_all_f32(const float* input, float* output, int64_t n) {
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = 0.0f;
    __syncthreads();
    float local_sum = 0.0f;
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = input[idx];
        if (!isnan(v)) local_sum += v;
    }
    atomicAdd(&ssum, local_sum);
    __syncthreads();
    if (threadIdx.x == 0) output[0] = ssum;
}

// Native double full-reduction nansum so Float64 does not lose ~8 significant
// digits via a Float32 downcast.
__global__ void nansum_all_f64(const double* input, double* output, int64_t n) {
    __shared__ double ssum;
    if (threadIdx.x == 0) ssum = 0.0;
    __syncthreads();
    double local_sum = 0.0;
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double v = input[idx];
        if (!isnan(v)) local_sum += v;
    }
    atomicAdd(&ssum, local_sum);
    __syncthreads();
    if (threadIdx.x == 0) output[0] = ssum;
}

Tensor nansum_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    if (dim < 0) {
        // Full reduction. Preserve Float64 natively (a Float32 downcast would
        // drop ~8 significant digits); only upcast genuinely lower-precision
        // dtypes to Float32. Mirrors the dim-specific branch below.
        DType orig_dtype = inputs[0].dtype();
        Tensor input = inputs[0];
        if (input.dtype() == DType::Float64) {
            Tensor result({1}, DType::Float64, input.device());
            nansum_all_f64<<<1, 256, 0, stream>>>(
                input.data<double>(), result.data<double>(), input.numel());
            CUDA_CHECK(cudaGetLastError());
            return result;
        }
        if (input.dtype() != DType::Float32) {
            input = input.to(DType::Float32);
        }
        Tensor result({1}, DType::Float32, input.device());
        nansum_all_f32<<<1, 256, 0, stream>>>(input.data<float>(), result.data<float>(), input.numel());
        CUDA_CHECK(cudaGetLastError());
        return (orig_dtype != DType::Float32) ? result.to(orig_dtype) : result;
    }
    // Dim-specific reduction: native CUDA kernel
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    DType orig_dtype = inputs[0].dtype();
    Tensor input = inputs[0];
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
        input = input.to(DType::Float32);
    }

    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    int64_t normalized_dim = dim;
    if (dim < 0) normalized_dim = ndim + dim;

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        normalized_dim, keepdim
    );

    Tensor result(output_shape, input.dtype(), input.device());

    auto shape_vec = std::vector<int64_t>(input_shape.begin(), input_shape.end());
    auto strides_vec = std::vector<int64_t>(input.strides().begin(), input.strides().end());

    if (input.dtype() == DType::Float32) {
        launch_dim_nansum(input.data<float>(), result.data<float>(), shape_vec, strides_vec, normalized_dim);
    } else if (input.dtype() == DType::Float64) {
        launch_dim_nansum(input.data<double>(), result.data<double>(), shape_vec, strides_vec, normalized_dim);
    }
    CUDA_CHECK(cudaGetLastError());
    return (orig_dtype != input.dtype()) ? result.to(orig_dtype) : result;
}

__global__ void count_non_nan_all_f32(const float* input, int64_t* output, int64_t n) {
    __shared__ int64_t scount;
    if (threadIdx.x == 0) scount = 0;
    __syncthreads();
    int64_t local_count = 0;
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        if (!isnan(input[idx])) local_count++;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&scount), static_cast<unsigned long long>(local_count));
    __syncthreads();
    if (threadIdx.x == 0) output[0] = scount;
}

__global__ void nanmean_div_f32(const float* sum, const int64_t* count, float* output) {
    if (threadIdx.x == 0) {
        int64_t c = count[0];
        output[0] = (c > 0) ? sum[0] / static_cast<float>(c) : 0.0f;
    }
}

// ============================================================================
// Dim-specific nanmean reduction kernel (mirrors nansum_along_dim_kernel but
// divides the NaN-ignoring sum by the per-slice count of non-NaN elements).
// ============================================================================
template<typename T>
__global__ void nanmean_along_dim_kernel(
    const T* input,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    using Acc = typename AccumType<T>::type;
    Acc sum = Acc(0);
    int64_t count = 0;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        Acc v = Acc(input[in_idx]);
        if (!isnan(static_cast<float>(v))) {
            sum = sum + v;
            count++;
        }
    }
    output[out_idx] = (count > 0) ? T(sum / Acc(count)) : T(Acc(0));
}

template<typename T>
static void launch_dim_nanmean(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) output_size *= input_shape[i];
    }
    if (output_size == 0 || dim_size == 0) return;
    DimMeta meta = make_dim_meta(input_shape, input_strides);
    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    nanmean_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

Tensor nanmean_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    if (dim < 0) {
        // Full reduction over the whole tensor -> {1} scalar.
        DType orig_dtype = inputs[0].dtype();
        Tensor input = inputs[0];
        if (input.dtype() != DType::Float32) {
            input = input.to(DType::Float32);
        }
        int64_t n = input.numel();

        // Compute nansum on GPU
        Tensor sum_result({1}, DType::Float32, input.device());
        nansum_all_f32<<<1, 256, 0, stream>>>(input.data<float>(), sum_result.data<float>(), n);

        // Count non-NaN elements on GPU
        Tensor count_result({1}, DType::Int64, input.device());
        count_non_nan_all_f32<<<1, 256, 0, stream>>>(input.data<float>(), count_result.data<int64_t>(), n);

        // Divide sum by count on GPU
        Tensor result({1}, DType::Float32, input.device());
        nanmean_div_f32<<<1, 1, 0, stream>>>(sum_result.data<float>(), count_result.data<int64_t>(), result.data<float>());
        CUDA_CHECK(cudaGetLastError());

        return (orig_dtype != DType::Float32) ? result.to(orig_dtype) : result;
    }

    // Dim-specific reduction: native CUDA kernel (mirrors nansum_dispatch).
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    DType orig_dtype = inputs[0].dtype();
    Tensor input = inputs[0];
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
        input = input.to(DType::Float32);
    }

    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    int64_t normalized_dim = dim;
    if (dim < 0) normalized_dim = ndim + dim;

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        normalized_dim, keepdim
    );

    Tensor result(output_shape, input.dtype(), input.device());

    auto shape_vec = std::vector<int64_t>(input_shape.begin(), input_shape.end());
    auto strides_vec = std::vector<int64_t>(input.strides().begin(), input.strides().end());

    if (input.dtype() == DType::Float32) {
        launch_dim_nanmean(input.data<float>(), result.data<float>(), shape_vec, strides_vec, normalized_dim);
    } else if (input.dtype() == DType::Float64) {
        launch_dim_nanmean(input.data<double>(), result.data<double>(), shape_vec, strides_vec, normalized_dim);
    }
    CUDA_CHECK(cudaGetLastError());
    return (orig_dtype != input.dtype()) ? result.to(orig_dtype) : result;
}

// Aminmax: compute min and max in a single pass
__global__ void aminmax_all_f32(const float* input, float* out_min, float* out_max, int64_t n) {
    __shared__ float smin[256];
    __shared__ float smax[256];
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = input[idx];
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }
    smin[threadIdx.x] = local_min;
    smax[threadIdx.x] = local_max;
    __syncthreads();
    // Block-level tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smin[threadIdx.x + stride] < smin[threadIdx.x])
                smin[threadIdx.x] = smin[threadIdx.x + stride];
            if (smax[threadIdx.x + stride] > smax[threadIdx.x])
                smax[threadIdx.x] = smax[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_min[0] = smin[0];
        out_max[0] = smax[0];
    }
}

// ============================================================================
// Dim-specific aminmax reduction kernel: per-slice min and max along `dim`.
// Mirrors nansum_along_dim_kernel's stride-based addressing so it is correct
// for any reduced axis (the public API passes a contiguous input).
// ============================================================================
template<typename T>
__global__ void aminmax_along_dim_kernel(
    const T* input,
    T* out_min,
    T* out_max,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    auto offset_at = [&](int64_t i) -> int64_t {
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            int64_t coord = (d == dim) ? i : indices[d];
            in_idx += coord * meta.strides[d];
        }
        return in_idx;
    };

    T mn = input[offset_at(0)];
    T mx = mn;
    for (int64_t i = 1; i < dim_size; i++) {
        T v = input[offset_at(i)];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    out_min[out_idx] = mn;
    out_max[out_idx] = mx;
}

template<typename T>
static void launch_dim_aminmax(
    const T* d_input,
    T* d_min,
    T* d_max,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) output_size *= input_shape[i];
    }
    if (output_size == 0 || dim_size == 0) return;
    DimMeta meta = make_dim_meta(input_shape, input_strides);
    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    aminmax_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_min, d_max, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

std::vector<Tensor> aminmax_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    if (dim < 0) {
        // Full reduction over the whole tensor -> two {1} scalars.
        DType orig_dtype = inputs[0].dtype();
        Tensor input = inputs[0];
        if (input.dtype() != DType::Float32) {
            input = input.to(DType::Float32);
        }
        int64_t n = input.numel();
        Tensor min_result({1}, DType::Float32, input.device());
        Tensor max_result({1}, DType::Float32, input.device());
        aminmax_all_f32<<<1, 256, 0, stream>>>(
            input.data<float>(), min_result.data<float>(), max_result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
        if (orig_dtype != DType::Float32) {
            min_result = min_result.to(orig_dtype);
            max_result = max_result.to(orig_dtype);
        }
        return {min_result, max_result};
    }

    // Dim-specific reduction: native CUDA kernel.
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    DType orig_dtype = inputs[0].dtype();
    Tensor input = inputs[0];
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
        input = input.to(DType::Float32);
    }

    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    int64_t normalized_dim = dim;
    if (dim < 0) normalized_dim = ndim + dim;

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        normalized_dim, keepdim
    );

    Tensor min_result(output_shape, input.dtype(), input.device());
    Tensor max_result(output_shape, input.dtype(), input.device());

    auto shape_vec = std::vector<int64_t>(input_shape.begin(), input_shape.end());
    auto strides_vec = std::vector<int64_t>(input.strides().begin(), input.strides().end());

    if (input.dtype() == DType::Float32) {
        launch_dim_aminmax(input.data<float>(), min_result.data<float>(), max_result.data<float>(),
                           shape_vec, strides_vec, normalized_dim);
    } else if (input.dtype() == DType::Float64) {
        launch_dim_aminmax(input.data<double>(), min_result.data<double>(), max_result.data<double>(),
                           shape_vec, strides_vec, normalized_dim);
    }
    CUDA_CHECK(cudaGetLastError());
    if (orig_dtype != input.dtype()) {
        min_result = min_result.to(orig_dtype);
        max_result = max_result.to(orig_dtype);
    }
    return {min_result, max_result};
}

// IndexAdd, IndexCopy, IndexFill CUDA kernels
__global__ void index_add_f32(float* output, const float* source, const int64_t* index,
                               int64_t outer, int64_t dim_size, int64_t idx_n, int64_t inner,
                               int* error_flag) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * idx_n * inner;
    if (tid >= total) return;
    int64_t j = tid % inner;
    int64_t k = (tid / inner) % idx_n;
    int64_t o = tid / (inner * idx_n);
    int64_t ix = index[k];
    if (ix < 0) ix += dim_size;
    if (ix < 0 || ix >= dim_size) { atomicExch(error_flag, 1); return; }
    atomicAdd(&output[(o * dim_size + ix) * inner + j],
              source[(o * idx_n + k) * inner + j]);
}

__global__ void index_copy_f32(float* output, const float* source, const int64_t* index,
                                int64_t outer, int64_t dim_size, int64_t idx_n, int64_t inner,
                                int* error_flag) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * idx_n * inner;
    if (tid >= total) return;
    int64_t j = tid % inner;
    int64_t k = (tid / inner) % idx_n;
    int64_t o = tid / (inner * idx_n);
    int64_t ix = index[k];
    if (ix < 0) ix += dim_size;
    if (ix < 0 || ix >= dim_size) { atomicExch(error_flag, 1); return; }
    output[(o * dim_size + ix) * inner + j] = source[(o * idx_n + k) * inner + j];
}

__global__ void index_fill_f32(float* output, const int64_t* index, float value,
                                int64_t outer, int64_t dim_size, int64_t idx_n, int64_t inner,
                                int* error_flag) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * idx_n * inner;
    if (tid >= total) return;
    int64_t j = tid % inner;
    int64_t k = (tid / inner) % idx_n;
    int64_t o = tid / (inner * idx_n);
    int64_t ix = index[k];
    if (ix < 0) ix += dim_size;
    if (ix < 0 || ix >= dim_size) { atomicExch(error_flag, 1); return; }
    output[(o * dim_size + ix) * inner + j] = value;
}

Tensor index_add_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    auto output = inputs[0].clone();
    auto shape = output.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t dim_size = shape[dim], idx_n = inputs[1].numel();
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];
    int64_t total = outer * idx_n * inner;
    if (total > 0 && output.dtype() == DType::Float32) {
        // Device-side OOB index error flag (matches the CPU std::out_of_range
        // contract; avoids out-of-bounds device writes on malformed indices).
        CudaBuffer error_buf(sizeof(int));
        CUDA_CHECK(cudaMemsetAsync(error_buf.as<int>(), 0, sizeof(int), stream));
        dim3 grid((total + 255) / 256), block(256);
        index_add_f32<<<grid, block, 0, stream>>>(output.data<float>(), inputs[2].data<float>(), inputs[1].data<int64_t>(), outer, dim_size, idx_n, inner, error_buf.as<int>());
        CUDA_CHECK(cudaGetLastError());
        int host_error = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host_error, error_buf.as<int>(), sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (host_error) {
            throw std::out_of_range("index_add: index out of range for dim of size " +
                                    std::to_string(dim_size));
        }
    } else if (total > 0) {
        auto dev = output.device();
        auto f32_out = inputs[0].to(DType::Float32);
        auto f32_src = inputs[2].to(DType::Float32);
        std::array<Tensor, 3> f32_inputs = {f32_out, inputs[1], f32_src};
        return index_add_dispatch(f32_inputs, attrs).to(inputs[0].dtype());
    }
    return output;
}

// ============================================================================
// ScatterReduce CUDA kernels
// ============================================================================

// Scatter-reduce modes: 0=sum, 1=prod, 2=mean, 3=amax, 4=amin
__global__ void scatter_reduce_f32(float* output, const float* source, const int64_t* index,
                                    int64_t outer, int64_t dim_size, int64_t idx_n, int64_t inner,
                                    int mode, unsigned long long* counts) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * idx_n * inner;
    if (tid >= total) return;
    int64_t j = tid % inner;
    int64_t k = (tid / inner) % idx_n;
    int64_t o = tid / (inner * idx_n);
    // Phase 7.6 2D scatter fix: index has shape matching src (outer, idx_n,
    // inner), so the per-thread index lookup must include the outer/inner
    // offsets. Previous `index[k]` only worked for 1D scatters where the
    // index lookup happens to be flat-equivalent.
    int64_t idx_pos = (o * idx_n + k) * inner + j;
    int64_t out_pos = (o * dim_size + index[idx_pos]) * inner + j;
    int64_t src_pos = idx_pos;
    float val = source[src_pos];

    if (mode == 0 || mode == 2) {
        // sum / mean: atomic add
        atomicAdd(&output[out_pos], val);
        if (mode == 2 && counts) atomicAdd(&counts[out_pos], 1ULL);
    } else if (mode == 1) {
        // prod: CAS loop
        unsigned int* addr = (unsigned int*)&output[out_pos];
        unsigned int old_bits = *addr;
        unsigned int assumed;
        do {
            assumed = old_bits;
            float old_val = __uint_as_float(assumed);
            unsigned int new_bits = __float_as_uint(old_val * val);
            old_bits = atomicCAS(addr, assumed, new_bits);
        } while (assumed != old_bits);
    } else if (mode == 3) {
        // amax: CAS loop. Propagate NaN to match CPU/PyTorch semantics: a NaN
        // candidate always wins, and once the stored value is NaN nothing
        // overwrites it. For finite values, replace only when val > old_val.
        unsigned int* addr = (unsigned int*)&output[out_pos];
        unsigned int old_bits = *addr;
        unsigned int assumed;
        do {
            assumed = old_bits;
            float old_val = __uint_as_float(assumed);
            // Stop replacing if: stored is NaN (sticky), or candidate is finite
            // and not strictly greater. A NaN candidate (val) always proceeds.
            if (!isnan(val) && (isnan(old_val) || val <= old_val)) break;
            unsigned int new_bits = __float_as_uint(val);
            old_bits = atomicCAS(addr, assumed, new_bits);
        } while (assumed != old_bits);
    } else if (mode == 4) {
        // amin: CAS loop. Propagate NaN to match CPU/PyTorch semantics: a NaN
        // candidate always wins, and once the stored value is NaN nothing
        // overwrites it. For finite values, replace only when val < old_val.
        unsigned int* addr = (unsigned int*)&output[out_pos];
        unsigned int old_bits = *addr;
        unsigned int assumed;
        do {
            assumed = old_bits;
            float old_val = __uint_as_float(assumed);
            if (!isnan(val) && (isnan(old_val) || val >= old_val)) break;
            unsigned int new_bits = __float_as_uint(val);
            old_bits = atomicCAS(addr, assumed, new_bits);
        } while (assumed != old_bits);
    }
}

// Initialize output positions that will be touched to identity values (!include_self)
__global__ void scatter_reduce_init_f32(float* output, const int64_t* index,
                                         int64_t outer, int64_t dim_size, int64_t idx_n, int64_t inner,
                                         int mode) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * idx_n * inner;
    if (tid >= total) return;
    int64_t j = tid % inner;
    int64_t k = (tid / inner) % idx_n;
    int64_t o = tid / (inner * idx_n);
    // Phase 7.6 2D scatter fix: index lookup must include outer/inner offsets.
    int64_t idx_pos = (o * idx_n + k) * inner + j;
    int64_t out_pos = (o * dim_size + index[idx_pos]) * inner + j;

    float identity;
    if (mode == 0 || mode == 2) identity = 0.0f;       // sum/mean
    else if (mode == 1) identity = 1.0f;                // prod
    else if (mode == 3) identity = -3.402823466e+38f;   // amax (FLT_MIN → -FLT_MAX)
    else identity = 3.402823466e+38f;                   // amin (FLT_MAX)

    output[out_pos] = identity;
}

// Divide by counts for mean mode.
//
// `counts[tid]` is the number of scatter operations that touched this output
// position (incremented in scatter_reduce_f32; does not include the self
// contribution). With include_self=true the accumulator already contains
// `input + sum(scatters)` so the divisor must be `count + 1` (the +1 is the
// self). With include_self=false the accumulator is just `sum(scatters)`
// and the divisor is `count`. Untouched positions (count=0) keep their
// initial value and are skipped.
__global__ void scatter_reduce_mean_div_f32(float* output, const unsigned long long* counts, int64_t numel, int include_self) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    unsigned long long c = counts[tid];
    if (c == 0ULL) return;  // untouched — leave the original value alone
    float divisor = include_self ? static_cast<float>(c + 1ULL) : static_cast<float>(c);
    output[tid] /= divisor;
}

Tensor scatter_reduce_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    std::string reduce = std::string(attrs.get_string(AttrKey::Reduction, "sum"));
    bool include_self = attrs.get_bool(AttrKey::IncludeSelf, true);

    // Upcast non-Float32
    if (inputs[0].dtype() != DType::Float32) {
        DType orig_dtype = inputs[0].dtype();
        auto f32_in = inputs[0].to(DType::Float32);
        auto f32_src = inputs[2].to(DType::Float32);
        NewOpAttributes f32_attrs;
        f32_attrs.set(AttrKey::Dim, dim);
        f32_attrs.set(AttrKey::Reduction, reduce);
        f32_attrs.set(AttrKey::IncludeSelf, include_self);
        std::array<Tensor, 3> f32_inputs = {f32_in, inputs[1], f32_src};
        return scatter_reduce_dispatch(f32_inputs, f32_attrs).to(orig_dtype);
    }

    auto output = inputs[0].clone();
    auto shape = output.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    // Phase 7.6 2D scatter fix: idx_n must be the index tensor's size along
    // the scatter dim, NOT the total numel. For 1D scatter these are equal;
    // for >1D they differ and the previous `numel()` would over-iterate and
    // miscompute index offsets.
    int64_t dim_size = shape[dim];
    int64_t idx_n = inputs[1].shape().size() > static_cast<size_t>(dim)
                  ? inputs[1].shape()[dim]
                  : inputs[1].numel();
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];
    int64_t total = outer * idx_n * inner;
    if (total == 0) return output;

    int mode;
    if (reduce == "sum") mode = 0;
    else if (reduce == "prod") mode = 1;
    else if (reduce == "mean") mode = 2;
    else if (reduce == "amax") mode = 3;
    else if (reduce == "amin") mode = 4;
    else throw std::invalid_argument("scatter_reduce: unknown reduce mode '" + reduce + "'");

    dim3 grid((total + 255) / 256), block(256);

    // If !include_self, initialize touched positions to identity
    if (!include_self) {
        scatter_reduce_init_f32<<<grid, block, 0, stream>>>(
            output.data<float>(), inputs[1].data<int64_t>(),
            outer, dim_size, idx_n, inner, mode);
        CUDA_CHECK(cudaGetLastError());
    }

    // Allocate count tensor for mean mode. 64-bit to avoid overflow when the
    // per-position fan-in exceeds 2^31 (extreme but possible for large scatters).
    Tensor count_tensor;
    unsigned long long* count_ptr = nullptr;
    if (mode == 2) {
        int64_t out_numel = output.numel();
        count_tensor = Tensor({out_numel}, DType::UInt64, output.device());
        // Zero-initialize counts. counts[] holds ONLY the number of scattered
        // contributions per position; the +1 for include_self is applied in
        // scatter_reduce_mean_div_f32's divisor (c+1 vs c), so counts must NOT
        // be pre-seeded to 1 here (that would divide by c+2).
        CUDA_CHECK(cudaMemsetAsync(count_tensor.data<uint64_t>(), 0,
                                   out_numel * sizeof(uint64_t), stream));
        count_ptr = reinterpret_cast<unsigned long long*>(count_tensor.data<uint64_t>());
    }

    scatter_reduce_f32<<<grid, block, 0, stream>>>(
        output.data<float>(), inputs[2].data<float>(), inputs[1].data<int64_t>(),
        outer, dim_size, idx_n, inner, mode, count_ptr);
    CUDA_CHECK(cudaGetLastError());

    // For mean: divide by counts
    if (mode == 2) {
        int64_t out_numel = output.numel();
        dim3 div_grid((out_numel + 255) / 256);
        scatter_reduce_mean_div_f32<<<div_grid, block, 0, stream>>>(
            output.data<float>(), count_ptr, out_numel, include_self ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
    }

    return output;
}

Tensor index_copy_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    auto output = inputs[0].clone();
    auto shape = output.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t dim_size = shape[dim], idx_n = inputs[1].numel();
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];
    int64_t total = outer * idx_n * inner;
    if (total > 0 && output.dtype() == DType::Float32) {
        // Device-side OOB index error flag (matches CPU std::out_of_range).
        CudaBuffer error_buf(sizeof(int));
        CUDA_CHECK(cudaMemsetAsync(error_buf.as<int>(), 0, sizeof(int), stream));
        dim3 grid((total + 255) / 256), block(256);
        index_copy_f32<<<grid, block, 0, stream>>>(output.data<float>(), inputs[2].data<float>(), inputs[1].data<int64_t>(), outer, dim_size, idx_n, inner, error_buf.as<int>());
        CUDA_CHECK(cudaGetLastError());
        int host_error = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host_error, error_buf.as<int>(), sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (host_error) {
            throw std::out_of_range("index_copy: index out of range for dim of size " +
                                    std::to_string(dim_size));
        }
    } else if (total > 0) {
        auto dev = output.device();
        auto f32_out = inputs[0].to(DType::Float32);
        auto f32_src = inputs[2].to(DType::Float32);
        std::array<Tensor, 3> f32_inputs = {f32_out, inputs[1], f32_src};
        return index_copy_dispatch(f32_inputs, attrs).to(inputs[0].dtype());
    }
    return output;
}

Tensor index_fill_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto stream = get_stream(attrs);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    double value = attrs.get_float(AttrKey::Value, 0.0);
    auto output = inputs[0].clone();
    auto shape = output.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t dim_size = shape[dim], idx_n = inputs[1].numel();
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];
    int64_t total = outer * idx_n * inner;
    if (total > 0 && output.dtype() == DType::Float32) {
        // Device-side OOB index error flag (matches CPU std::out_of_range).
        CudaBuffer error_buf(sizeof(int));
        CUDA_CHECK(cudaMemsetAsync(error_buf.as<int>(), 0, sizeof(int), stream));
        dim3 grid((total + 255) / 256), block(256);
        index_fill_f32<<<grid, block, 0, stream>>>(output.data<float>(), inputs[1].data<int64_t>(), static_cast<float>(value), outer, dim_size, idx_n, inner, error_buf.as<int>());
        CUDA_CHECK(cudaGetLastError());
        int host_error = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host_error, error_buf.as<int>(), sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (host_error) {
            throw std::out_of_range("index_fill: index out of range for dim of size " +
                                    std::to_string(dim_size));
        }
    } else if (total > 0) {
        auto dev = output.device();
        auto f32_out = inputs[0].to(DType::Float32);
        std::array<Tensor, 2> f32_inputs = {f32_out, inputs[1]};
        return index_fill_dispatch(f32_inputs, attrs).to(inputs[0].dtype());
    }
    return output;
}

// ============================================================================
#ifdef TENZOR_TRACK_SATURATION
uint32_t get_and_reset_fp16_saturation_count() {
    uint32_t count = 0;
    cudaMemcpyFromSymbol(&count, g_fp16_saturation_count, sizeof(count));
    uint32_t zero = 0;
    cudaMemcpyToSymbol(g_fp16_saturation_count, &zero, sizeof(zero));
    return count;
}
#endif

} // namespace cuda
} // namespace tenzor
