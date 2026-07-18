#include "rocm_nan_helpers.hip.h"  // E.2: safe_f2h / safe_h2f / safe_f2bf / safe_bf2f
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"  // for tenzor::get_global_seed
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <cstdint>
#include "fp16_saturate.h"

namespace tenzor {
namespace rocm {

// ============================================================================
// HIP Helper Functions
// ============================================================================

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error(std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + ": " + hipGetErrorString(err)); \
        } \
    } while(0)

// Grid-stride loop helper
#define HIP_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// Optimal block size for element-wise operations
constexpr int BLOCK_SIZE = 256;

// Calculate grid size for element-wise operations.
// Clamp to at least 1 block: HIP rejects zero-grid launches with
// hipErrorInvalidConfiguration, so for n==0 we still launch a single block.
// The grid-stride loop bodies do no work when n==0, yielding a correct
// empty result instead of throwing on legitimately-empty tensors.
inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    int blocks = static_cast<int>((n + block_size - 1) / block_size);
    return blocks > 0 ? blocks : 1;
}

// ============================================================================
// ReLU Activation
// ============================================================================

// Forward: max(0, x)
template<typename T>
__global__ void relu_forward_kernel(const T* input, T* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = input[idx] > T(0) ? input[idx] : T(0);
    }
}

// Forward: max(0, x) for Float16
__global__ void relu_forward_kernel_fp16(const __half* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        output[idx] = val > 0.0f ? input[idx] : tenzor::rocm::safe_f2h(0.0f);
    }
}

// Backward: grad_out * (x > 0)
template<typename T>
__global__ void relu_backward_kernel(const T* grad_output, const T* input,
                                     T* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        grad_input[idx] = grad_output[idx] * (input[idx] > T(0) ? T(1) : T(0));
    }
}

// Backward: relu for Float16
__global__ void relu_backward_kernel_fp16(const __half* grad_output, const __half* input,
                                          __half* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        grad_input[idx] = val > 0.0f ? grad_output[idx] : tenzor::rocm::safe_f2h(0.0f);
    }
}

// Host functions
extern "C" {
    void relu_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(relu_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void relu_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(relu_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void relu_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(relu_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }

    void relu_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(relu_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
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
    HIP_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = sigmoid_stable(input[idx]);
    }
}

// Backward: grad_out * sigmoid(x) * (1 - sigmoid(x))
template<typename T>
__global__ void sigmoid_backward_kernel(const T* grad_output, const T* input,
                                       T* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T sigmoid_x = sigmoid_stable(input[idx]);
        grad_input[idx] = grad_output[idx] * sigmoid_x * (T(1) - sigmoid_x);
    }
}

// Float16 sigmoid forward - compute in float for precision
__global__ void sigmoid_forward_kernel_fp16(const __half* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float sigmoid_x;
        if (x >= 0.0f) {
            sigmoid_x = 1.0f / (1.0f + expf(-x));
        } else {
            float exp_x = expf(x);
            sigmoid_x = exp_x / (1.0f + exp_x);
        }
        output[idx] = tenzor::rocm::safe_f2h(sigmoid_x);
    }
}

// Float16 sigmoid backward
__global__ void sigmoid_backward_kernel_fp16(const __half* grad_output, const __half* input,
                                             __half* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float sigmoid_x;
        if (x >= 0.0f) {
            sigmoid_x = 1.0f / (1.0f + expf(-x));
        } else {
            float exp_x = expf(x);
            sigmoid_x = exp_x / (1.0f + exp_x);
        }
        float grad = tenzor::rocm::safe_h2f(grad_output[idx]) * sigmoid_x * (1.0f - sigmoid_x);
        grad_input[idx] = tenzor::rocm::safe_f2h(grad);
    }
}

// Host functions
extern "C" {
    void sigmoid_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void sigmoid_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void sigmoid_backward_float(const float* grad_output, const float* input,
                               float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }

    void sigmoid_backward_double(const double* grad_output, const double* input,
                                double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }
}

// ============================================================================
// Tanh Activation
// ============================================================================

// Forward: tanh(x)
template<typename T>
__global__ void tanh_forward_kernel(const T* input, T* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = tanh(input[idx]);
    }
}

// Forward: tanh(x) for Float16 - compute in float for precision
__global__ void tanh_forward_kernel_fp16(const __half* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        output[idx] = tenzor::rocm::safe_f2h(tanhf(val));
    }
}

// Backward: grad_out * (1 - tanh(x)^2)
template<typename T>
__global__ void tanh_backward_kernel(const T* grad_output, const T* input,
                                    T* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T tanh_x = tanh(input[idx]);
        grad_input[idx] = grad_output[idx] * (T(1) - tanh_x * tanh_x);
    }
}

// Backward: tanh for Float16 - compute in float for precision
__global__ void tanh_backward_kernel_fp16(const __half* grad_output, const __half* input,
                                          __half* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        float tanh_x = tanhf(val);
        float grad = tenzor::rocm::safe_h2f(grad_output[idx]) * (1.0f - tanh_x * tanh_x);
        grad_input[idx] = tenzor::rocm::safe_f2h(grad);
    }
}

// Host functions
extern "C" {
    void tanh_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void tanh_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void tanh_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }

    void tanh_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }
}

// ============================================================================
// GELU Activation (Gaussian Error Linear Unit)
// ============================================================================

// GELU: exact erf form 0.5 * x * (1 + erf(x / sqrt(2))) — matches PyTorch's
// default (approximate='none') and the CPU backend.
template<typename T>
__device__ __forceinline__ T gelu_forward_impl(T x) {
    const T inv_sqrt2 = T(0.70710678118654752);
    return T(0.5) * x * (T(1) + erf(x * inv_sqrt2));
}

// GELU backward derivative (exact):
//   gelu'(x) = 0.5*(1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
template<typename T>
__device__ __forceinline__ T gelu_backward_impl(T x) {
    const T inv_sqrt2 = T(0.70710678118654752);
    const T pdf_coeff = T(0.39894228040143268);  // 1/sqrt(2*pi)
    T cdf = T(0.5) * (T(1) + erf(x * inv_sqrt2));
    T pdf = pdf_coeff * exp(T(-0.5) * x * x);
    return cdf + x * pdf;
}

template<typename T>
__global__ void gelu_forward_kernel(const T* input, T* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = gelu_forward_impl(input[idx]);
    }
}

template<typename T>
__global__ void gelu_backward_kernel(const T* grad_output, const T* input,
                                    T* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        grad_input[idx] = grad_output[idx] * gelu_backward_impl(input[idx]);
    }
}

// GELU forward for Float16 - compute in float32 for precision
__global__ void gelu_forward_kernel_fp16(const __half* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        output[idx] = tenzor::rocm::safe_f2h(gelu_forward_impl(val));
    }
}

// GELU backward for Float16 - compute in float32 for precision
__global__ void gelu_backward_kernel_fp16(const __half* grad_output, const __half* input,
                                          __half* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float grad = tenzor::rocm::safe_h2f(grad_output[idx]);
        float val = tenzor::rocm::safe_h2f(input[idx]);
        grad_input[idx] = tenzor::rocm::safe_f2h(grad * gelu_backward_impl(val));
    }
}

// Host functions
extern "C" {
    void gelu_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void gelu_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void gelu_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }

    void gelu_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }
}

// ============================================================================
// Leaky ReLU Activation
// ============================================================================

// Forward: x if x > 0 else alpha * x
template<typename T>
__global__ void leaky_relu_forward_kernel(const T* input, T* output,
                                         int64_t n, T alpha) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = input[idx] > T(0) ? input[idx] : alpha * input[idx];
    }
}

// Backward: grad_out * (1 if x > 0 else alpha)
template<typename T>
__global__ void leaky_relu_backward_kernel(const T* grad_output, const T* input,
                                          T* grad_input, int64_t n, T alpha) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        grad_input[idx] = grad_output[idx] * (input[idx] > T(0) ? T(1) : alpha);
    }
}

// Float16 Leaky ReLU forward
__global__ void leaky_relu_forward_kernel_fp16(const __half* input, __half* output,
                                               int64_t n, float alpha) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        output[idx] = tenzor::rocm::safe_f2h(val > 0.0f ? val : alpha * val);
    }
}

// Float16 Leaky ReLU backward
__global__ void leaky_relu_backward_kernel_fp16(const __half* grad_output, const __half* input,
                                                __half* grad_input, int64_t n, float alpha) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        float grad = tenzor::rocm::safe_h2f(grad_output[idx]) * (val > 0.0f ? 1.0f : alpha);
        grad_input[idx] = tenzor::rocm::safe_f2h(grad);
    }
}

// Host functions
extern "C" {
    void leaky_relu_forward_float(const float* input, float* output,
                                 int64_t n, float alpha) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n, alpha);
        HIP_CHECK(hipGetLastError());
    }

    void leaky_relu_forward_double(const double* input, double* output,
                                  int64_t n, double alpha) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n, alpha);
        HIP_CHECK(hipGetLastError());
    }

    void leaky_relu_backward_float(const float* grad_output, const float* input,
                                  float* grad_input, int64_t n, float alpha) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n, alpha);
        HIP_CHECK(hipGetLastError());
    }

    void leaky_relu_backward_double(const double* grad_output, const double* input,
                                   double* grad_input, int64_t n, double alpha) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n, alpha);
        HIP_CHECK(hipGetLastError());
    }
}

// ============================================================================
// ELU Activation (Exponential Linear Unit)
// ============================================================================

// ELU: x if x > 0 else alpha * (exp(x) - 1)
template<typename T>
__global__ void elu_forward_kernel(const T* input, T* output, int64_t n, T alpha) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        output[idx] = x > T(0) ? x : alpha * (exp(x) - T(1));
    }
}

// ELU backward: grad_out * (1 if x > 0 else alpha * exp(x))
template<typename T>
__global__ void elu_backward_kernel(const T* grad_output, const T* input,
                                   T* grad_input, int64_t n, T alpha) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        grad_input[idx] = grad_output[idx] * (x > T(0) ? T(1) : alpha * exp(x));
    }
}

// Float16 ELU forward
__global__ void elu_forward_kernel_fp16(const __half* input, __half* output, int64_t n, float alpha) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        output[idx] = tenzor::rocm::safe_f2h(x > 0.0f ? x : alpha * (expf(x) - 1.0f));
    }
}

// Float16 ELU backward
__global__ void elu_backward_kernel_fp16(const __half* grad_output, const __half* input,
                                         __half* grad_input, int64_t n, float alpha) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float grad = tenzor::rocm::safe_h2f(grad_output[idx]) * (x > 0.0f ? 1.0f : alpha * expf(x));
        grad_input[idx] = tenzor::rocm::safe_f2h(grad);
    }
}

// Host functions
extern "C" {
    void elu_forward_float(const float* input, float* output, int64_t n, float alpha) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(elu_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n, alpha);
        HIP_CHECK(hipGetLastError());
    }

    void elu_forward_double(const double* input, double* output, int64_t n, double alpha) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(elu_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n, alpha);
        HIP_CHECK(hipGetLastError());
    }

    void elu_backward_float(const float* grad_output, const float* input,
                           float* grad_input, int64_t n, float alpha) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(elu_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n, alpha);
        HIP_CHECK(hipGetLastError());
    }

    void elu_backward_double(const double* grad_output, const double* input,
                            double* grad_input, int64_t n, double alpha) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(elu_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n, alpha);
        HIP_CHECK(hipGetLastError());
    }
}

// ============================================================================
// SELU Activation (Scaled Exponential Linear Unit)
// ============================================================================

// SELU: scale * (x if x > 0 else alpha * (exp(x) - 1))
// Standard values: alpha = 1.67326, scale = 1.0507
template<typename T>
__global__ void selu_forward_kernel(const T* input, T* output, int64_t n) {
    const T alpha = T(1.6732632423543772848170429916717);
    const T scale = T(1.0507009873554804934193349852946);

    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        output[idx] = scale * (x > T(0) ? x : alpha * (exp(x) - T(1)));
    }
}

// SELU backward
template<typename T>
__global__ void selu_backward_kernel(const T* grad_output, const T* input,
                                    T* grad_input, int64_t n) {
    const T alpha = T(1.6732632423543772848170429916717);
    const T scale = T(1.0507009873554804934193349852946);

    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        grad_input[idx] = grad_output[idx] * scale * (x > T(0) ? T(1) : alpha * exp(x));
    }
}

// Float16 SELU forward
__global__ void selu_forward_kernel_fp16(const __half* input, __half* output, int64_t n) {
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;

    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        output[idx] = tenzor::rocm::safe_f2h(scale * (x > 0.0f ? x : alpha * (expf(x) - 1.0f)));
    }
}

// Float16 SELU backward
__global__ void selu_backward_kernel_fp16(const __half* grad_output, const __half* input,
                                          __half* grad_input, int64_t n) {
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;

    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float grad = tenzor::rocm::safe_h2f(grad_output[idx]) * scale * (x > 0.0f ? 1.0f : alpha * expf(x));
        grad_input[idx] = tenzor::rocm::safe_f2h(grad);
    }
}

// Host functions
extern "C" {
    void selu_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(selu_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void selu_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(selu_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void selu_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(selu_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }

    void selu_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(selu_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }
}

// ============================================================================
// Swish Activation (SiLU - Sigmoid Linear Unit)
// ============================================================================

// Swish/SiLU: x * sigmoid(x)
template<typename T>
__global__ void swish_forward_kernel(const T* input, T* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T sigmoid_x = sigmoid_stable(x);
        output[idx] = x * sigmoid_x;
    }
}

// Swish backward: grad_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
template<typename T>
__global__ void swish_backward_kernel(const T* grad_output, const T* input,
                                     T* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T sigmoid_x = sigmoid_stable(x);
        grad_input[idx] = grad_output[idx] * (sigmoid_x + x * sigmoid_x * (T(1) - sigmoid_x));
    }
}

// Float16 Swish forward
__global__ void swish_forward_kernel_fp16(const __half* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float sigmoid_x = x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
        output[idx] = tenzor::rocm::safe_f2h(x * sigmoid_x);
    }
}

// Float16 Swish backward
__global__ void swish_backward_kernel_fp16(const __half* grad_output, const __half* input,
                                           __half* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float sigmoid_x = x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
        float grad = tenzor::rocm::safe_h2f(grad_output[idx]) * (sigmoid_x + x * sigmoid_x * (1.0f - sigmoid_x));
        grad_input[idx] = tenzor::rocm::safe_f2h(grad);
    }
}

// Host functions
extern "C" {
    void swish_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(swish_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void swish_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(swish_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void swish_backward_float(const float* grad_output, const float* input,
                             float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(swish_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }

    void swish_backward_double(const double* grad_output, const double* input,
                              double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(swish_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }
}

// ============================================================================
// Mish Activation
// ============================================================================

// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
// Numerically stable implementation
template<typename T>
__device__ __forceinline__ T softplus_stable(T x) {
    if (x > T(20)) {
        return x;  // For large x, softplus(x) ≈ x
    } else if (x < T(-20)) {
        return exp(x);  // For very negative x, softplus(x) ≈ exp(x)
    } else {
        return log(T(1) + exp(x));
    }
}

template<typename T>
__global__ void mish_forward_kernel(const T* input, T* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T softplus_x = softplus_stable(x);
        output[idx] = x * tanh(softplus_x);
    }
}

// Mish backward: grad_out * (tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x))
template<typename T>
__global__ void mish_backward_kernel(const T* grad_output, const T* input,
                                    T* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T softplus_x = softplus_stable(x);
        T tanh_softplus = tanh(softplus_x);
        T sech_squared = T(1) - tanh_softplus * tanh_softplus;
        T sigmoid_x = sigmoid_stable(x);

        grad_input[idx] = grad_output[idx] * (tanh_softplus + x * sech_squared * sigmoid_x);
    }
}

// Float16 Mish forward
__global__ void mish_forward_kernel_fp16(const __half* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float softplus_x = x > 20.0f ? x : logf(1.0f + expf(x));
        output[idx] = tenzor::rocm::safe_f2h(x * tanhf(softplus_x));
    }
}

// Float16 Mish backward
__global__ void mish_backward_kernel_fp16(const __half* grad_output, const __half* input,
                                          __half* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float softplus_x = x > 20.0f ? x : logf(1.0f + expf(x));
        float tanh_softplus = tanhf(softplus_x);
        float sech_squared = 1.0f - tanh_softplus * tanh_softplus;
        float sigmoid_x = x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
        float grad = tenzor::rocm::safe_h2f(grad_output[idx]) * (tanh_softplus + x * sech_squared * sigmoid_x);
        grad_input[idx] = tenzor::rocm::safe_f2h(grad);
    }
}

// Host functions
extern "C" {
    void mish_forward_float(const float* input, float* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(mish_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void mish_forward_double(const double* input, double* output, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(mish_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n);
        HIP_CHECK(hipGetLastError());
    }

    void mish_backward_float(const float* grad_output, const float* input,
                            float* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(mish_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }

    void mish_backward_double(const double* grad_output, const double* input,
                             double* grad_input, int64_t n) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(mish_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n);
        HIP_CHECK(hipGetLastError());
    }
}

// ============================================================================
// Softplus Activation
// ============================================================================

// Softplus: log(1 + exp(x))
// Uses numerically stable implementation
template<typename T>
__global__ void softplus_forward_kernel(const T* input, T* output, int64_t n, T beta, T threshold) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T bx = beta * x;
        if (bx > threshold) {
            output[idx] = x;  // For large x, softplus(x) ≈ x
        } else if (bx < -threshold) {
            output[idx] = exp(bx) / beta;  // For very negative x
        } else {
            output[idx] = log(T(1) + exp(bx)) / beta;
        }
    }
}

// Softplus backward: grad_out * sigmoid(beta * x)
template<typename T>
__global__ void softplus_backward_kernel(const T* grad_output, const T* input,
                                         T* grad_input, int64_t n, T beta, T threshold) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T bx = beta * x;
        if (bx > threshold) {
            grad_input[idx] = grad_output[idx];
        } else if (bx < -threshold) {
            grad_input[idx] = grad_output[idx] * exp(bx);
        } else {
            T sig = T(1) / (T(1) + exp(-bx));
            grad_input[idx] = grad_output[idx] * sig;
        }
    }
}

// Float16 Softplus forward
__global__ void softplus_forward_kernel_fp16(const __half* input, __half* output, int64_t n, float beta, float threshold) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float bx = beta * x;
        float result;
        if (bx > threshold) {
            result = x;
        } else if (bx < -threshold) {
            result = expf(bx) / beta;
        } else {
            result = logf(1.0f + expf(bx)) / beta;
        }
        output[idx] = tenzor::rocm::safe_f2h(result);
    }
}

// Float16 Softplus backward
__global__ void softplus_backward_kernel_fp16(const __half* grad_output, const __half* input,
                                              __half* grad_input, int64_t n, float beta, float threshold) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float bx = beta * x;
        float grad;
        if (bx > threshold) {
            grad = tenzor::rocm::safe_h2f(grad_output[idx]);
        } else if (bx < -threshold) {
            grad = tenzor::rocm::safe_h2f(grad_output[idx]) * expf(bx);
        } else {
            float sig = 1.0f / (1.0f + expf(-bx));
            grad = tenzor::rocm::safe_h2f(grad_output[idx]) * sig;
        }
        grad_input[idx] = tenzor::rocm::safe_f2h(grad);
    }
}

// Host functions
extern "C" {
    void softplus_forward_float(const float* input, float* output, int64_t n, float beta, float threshold) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(softplus_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n, beta, threshold);
        HIP_CHECK(hipGetLastError());
    }

    void softplus_forward_double(const double* input, double* output, int64_t n, double beta, double threshold) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(softplus_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          input, output, n, beta, threshold);
        HIP_CHECK(hipGetLastError());
    }

    void softplus_backward_float(const float* grad_output, const float* input,
                                 float* grad_input, int64_t n, float beta, float threshold) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(softplus_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n, beta, threshold);
        HIP_CHECK(hipGetLastError());
    }

    void softplus_backward_double(const double* grad_output, const double* input,
                                  double* grad_input, int64_t n, double beta, double threshold) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(softplus_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                          grad_output, input, grad_input, n, beta, threshold);
        HIP_CHECK(hipGetLastError());
    }
}

// ============================================================================
// Softmax Activation with AMD GPU Optimizations
// ============================================================================

// Shared memory size for reductions
constexpr int SOFTMAX_BLOCK_SIZE = 256;

// NOTE: Do NOT hardcode wavefront size — RDNA 3/4 GPUs use wave32, CDNA uses wave64.
// Use the HIP built-in `warpSize` in device code instead.

// Warp-level reduction using shuffle instructions
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down(val, offset));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Lowest representable value for the element type, used to seed max reductions.
// -FLT_MAX (~-3.4e38) is a valid *finite* double, so seeding a Float64 reduction
// with it would mishandle rows whose values are all below -3.4e38; pick the
// true lower bound per type.
template<typename T>
__device__ __forceinline__ T lowest_value() {
    if constexpr (sizeof(T) == sizeof(double)) {
        return static_cast<T>(-DBL_MAX);
    } else {
        return static_cast<T>(-FLT_MAX);
    }
}

// Block-level reduction using shared memory
// Optimized for AMD GPU memory hierarchy (LDS - Local Data Share)
template<typename T>
__device__ T block_reduce_max(T val, T* shared) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : lowest_value<T>();
    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

template<typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// Softmax forward with temperature scaling: exp((x_i - max) / T) / sum(exp((x_j - max) / T))
// Each block handles one row
template<typename T>
__global__ void softmax_forward_kernel(const T* input, T* output,
                                      int64_t batch_size, int64_t dim_size,
                                      T temperature) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* input_row = input + row * dim_size;
    T* output_row = output + row * dim_size;

    // Step 1: Find max value for numerical stability
    T max_val = lowest_value<T>();
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

    // Step 2: Compute exp((x - max) / temperature) and sum
    T sum_exp = T(0);
    T inv_temp = T(1) / temperature;
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        T exp_val = exp((input_row[i] - max_val) * inv_temp);
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

// Softmax forward for Float16 - compute in float32 for numerical stability
__global__ void softmax_forward_kernel_fp16(const __half* input, __half* output,
                                            int64_t batch_size, int64_t dim_size,
                                            float temperature) {
    extern __shared__ __align__(sizeof(float)) unsigned char shared_mem[];
    float* shared = reinterpret_cast<float*>(shared_mem);

    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const __half* input_row = input + row * dim_size;
    __half* output_row = output + row * dim_size;

    // Step 1: Find max value for numerical stability
    float max_val = -FLT_MAX;
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        max_val = fmaxf(max_val, tenzor::rocm::safe_h2f(input_row[i]));
    }
    max_val = block_reduce_max(max_val, shared);
    __syncthreads();

    // Broadcast max to all threads
    if (threadIdx.x == 0) {
        shared[0] = max_val;
    }
    __syncthreads();
    max_val = shared[0];

    // Step 2: Compute exp((x - max) / temperature) and sum
    float sum_exp = 0.0f;
    float inv_temp = 1.0f / temperature;
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = tenzor::rocm::safe_h2f(input_row[i]);
        float exp_val = expf((val - max_val) * inv_temp);
        // Temporarily store in output (as float bits, will overwrite)
        output_row[i] = tenzor::rocm::safe_f2h(exp_val);
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
        float exp_val = tenzor::rocm::safe_h2f(output_row[i]);
        output_row[i] = tenzor::rocm::safe_f2h(exp_val / sum_exp);
    }
}

// Softmax backward for Float16 - compute in float32 for numerical stability
__global__ void softmax_backward_kernel_fp16(const __half* grad_output, const __half* output,
                                             __half* grad_input,
                                             int64_t batch_size, int64_t dim_size) {
    extern __shared__ __align__(sizeof(float)) unsigned char shared_mem[];
    float* shared = reinterpret_cast<float*>(shared_mem);

    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const __half* grad_out_row = grad_output + row * dim_size;
    const __half* out_row = output + row * dim_size;
    __half* grad_in_row = grad_input + row * dim_size;

    // Compute sum(grad_output * softmax)
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum += tenzor::rocm::safe_h2f(grad_out_row[i]) * tenzor::rocm::safe_h2f(out_row[i]);
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
        float out_val = tenzor::rocm::safe_h2f(out_row[i]);
        float grad_val = tenzor::rocm::safe_h2f(grad_out_row[i]);
        grad_in_row[i] = tenzor::rocm::safe_f2h(out_val * (grad_val - sum));
    }
}

// Host functions
extern "C" {
    void softmax_forward_float(const float* input, float* output,
                              int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        hipLaunchKernelGGL(softmax_forward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, 0, input, output, batch_size, dim_size, 1.0f);
        HIP_CHECK(hipGetLastError());
    }

    void softmax_forward_double(const double* input, double* output,
                               int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(softmax_forward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, 0, input, output, batch_size, dim_size, 1.0);
        HIP_CHECK(hipGetLastError());
    }

    void softmax_forward_float_temperature(const float* input, float* output,
                                          int64_t batch_size, int64_t dim_size, float temperature) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        hipLaunchKernelGGL(softmax_forward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, 0, input, output, batch_size, dim_size, temperature);
        HIP_CHECK(hipGetLastError());
    }

    void softmax_forward_double_temperature(const double* input, double* output,
                                           int64_t batch_size, int64_t dim_size, double temperature) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(softmax_forward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, 0, input, output, batch_size, dim_size, temperature);
        HIP_CHECK(hipGetLastError());
    }

    void softmax_backward_float(const float* grad_output, const float* output,
                               float* grad_input,
                               int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        hipLaunchKernelGGL(softmax_backward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, 0, grad_output, output, grad_input, batch_size, dim_size);
        HIP_CHECK(hipGetLastError());
    }

    void softmax_backward_double(const double* grad_output, const double* output,
                                double* grad_input,
                                int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(softmax_backward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, 0, grad_output, output, grad_input, batch_size, dim_size);
        HIP_CHECK(hipGetLastError());
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
    T max_val = lowest_value<T>();
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

// LogSoftmax forward for Float16 - compute in float32 for numerical stability
__global__ void log_softmax_forward_kernel_fp16(const __half* input, __half* output,
                                                int64_t batch_size, int64_t dim_size) {
    extern __shared__ __align__(sizeof(float)) unsigned char shared_mem[];
    float* shared = reinterpret_cast<float*>(shared_mem);

    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const __half* input_row = input + row * dim_size;
    __half* output_row = output + row * dim_size;

    // Step 1: Find max value for numerical stability
    float max_val = -FLT_MAX;
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        max_val = fmaxf(max_val, tenzor::rocm::safe_h2f(input_row[i]));
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
    float sum_exp = 0.0f;
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum_exp += expf(tenzor::rocm::safe_h2f(input_row[i]) - max_val);
    }
    sum_exp = block_reduce_sum(sum_exp, shared);
    __syncthreads();

    // Broadcast log_sum_exp to all threads
    if (threadIdx.x == 0) {
        shared[0] = logf(sum_exp);
    }
    __syncthreads();
    float log_sum_exp = shared[0];

    // Step 3: Compute log_softmax = x - max - log_sum_exp
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = tenzor::rocm::safe_h2f(input_row[i]);
        output_row[i] = tenzor::rocm::safe_f2h(val - max_val - log_sum_exp);
    }
}

// LogSoftmax backward for Float16 - compute in float32 for numerical stability
__global__ void log_softmax_backward_kernel_fp16(const __half* grad_output, const __half* output,
                                                 __half* grad_input,
                                                 int64_t batch_size, int64_t dim_size) {
    extern __shared__ __align__(sizeof(float)) unsigned char shared_mem[];
    float* shared = reinterpret_cast<float*>(shared_mem);

    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const __half* grad_out_row = grad_output + row * dim_size;
    const __half* out_row = output + row * dim_size;
    __half* grad_in_row = grad_input + row * dim_size;

    // Compute sum(grad_output)
    float sum_grad = 0.0f;
    for (int64_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum_grad += tenzor::rocm::safe_h2f(grad_out_row[i]);
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
        float grad_val = tenzor::rocm::safe_h2f(grad_out_row[i]);
        float out_val = tenzor::rocm::safe_h2f(out_row[i]);
        grad_in_row[i] = tenzor::rocm::safe_f2h(grad_val - expf(out_val) * sum_grad);
    }
}

// Host functions
extern "C" {
    void log_softmax_forward_float(const float* input, float* output,
                                   int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        hipLaunchKernelGGL(log_softmax_forward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, 0, input, output, batch_size, dim_size);
        HIP_CHECK(hipGetLastError());
    }

    void log_softmax_forward_double(const double* input, double* output,
                                    int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(log_softmax_forward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, 0, input, output, batch_size, dim_size);
        HIP_CHECK(hipGetLastError());
    }

    void log_softmax_backward_float(const float* grad_output, const float* output,
                                    float* grad_input,
                                    int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        hipLaunchKernelGGL(log_softmax_backward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, 0, grad_output, output, grad_input, batch_size, dim_size);
        HIP_CHECK(hipGetLastError());
    }

    void log_softmax_backward_double(const double* grad_output, const double* output,
                                     double* grad_input,
                                     int64_t batch_size, int64_t dim_size) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(log_softmax_backward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, 0, grad_output, output, grad_input, batch_size, dim_size);
        HIP_CHECK(hipGetLastError());
    }
}

// ============================================================================
// Tensor Wrapper Functions
// ============================================================================

// ReLU wrapper
auto relu_kernel(const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: input.data<T>() is a raw pointer with NO stride handling --
    // a non-contiguous tensor silently reads the wrong elements via straight
    // linear indexing.
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;  // Handle empty tensors
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(relu_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(relu_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(relu_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = relu_kernel(input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("ReLU only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in relu_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ReLU backward wrapper
auto relu_backward_kernel(const Tensor& grad_output_in, const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: grad_output.data<T>()/input.data<T>() are raw pointers with NO
    // stride handling -- a non-contiguous tensor silently reads the wrong
    // elements via straight linear indexing.
    Tensor grad_output_cont = grad_output_in.is_contiguous() ? grad_output_in : grad_output_in.contiguous();
    const Tensor& grad_output = grad_output_cont;
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;  // Handle empty tensors
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(relu_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(relu_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(relu_backward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = relu_backward_kernel(grad_output_f32, input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("ReLU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in relu_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Sigmoid wrapper
auto sigmoid_kernel(const Tensor& input_raw, hipStream_t stream) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure contiguous input.
    auto input = input_raw.contiguous();
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = sigmoid_kernel(input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Sigmoid only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in sigmoid_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Sigmoid backward wrapper
auto sigmoid_backward_kernel(const Tensor& grad_output_in, const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: grad_output.data<T>()/input.data<T>() are raw pointers with NO
    // stride handling -- a non-contiguous tensor silently reads the wrong
    // elements via straight linear indexing.
    Tensor grad_output_cont = grad_output_in.is_contiguous() ? grad_output_in : grad_output_in.contiguous();
    const Tensor& grad_output = grad_output_cont;
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_backward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = sigmoid_backward_kernel(grad_output_f32, input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Sigmoid backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in sigmoid_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Tanh wrapper
auto tanh_kernel(const Tensor& input_raw, hipStream_t stream) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure contiguous input.
    auto input = input_raw.contiguous();
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = tanh_kernel(input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Tanh only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in tanh_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Tanh backward wrapper
auto tanh_backward_kernel(const Tensor& grad_output_in, const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: grad_output.data<T>()/input.data<T>() are raw pointers with NO
    // stride handling -- a non-contiguous tensor silently reads the wrong
    // elements via straight linear indexing.
    Tensor grad_output_cont = grad_output_in.is_contiguous() ? grad_output_in : grad_output_in.contiguous();
    const Tensor& grad_output = grad_output_cont;
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_backward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = tanh_backward_kernel(grad_output_f32, input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Tanh backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in tanh_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// GELU wrapper
auto gelu_kernel(const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: input.data<T>() is a raw pointer with NO stride handling --
    // a non-contiguous input silently reads the wrong elements.
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = gelu_kernel(input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("GELU only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in gelu_kernel: ") + hipGetErrorString(err));
    }

    if (input.dtype() == DType::Float16) {
        fp16_saturate(result.data_ptr(), result.numel(), stream);
    }

    return result;
}

// GELU backward wrapper
auto gelu_backward_kernel(const Tensor& grad_output_in, const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: grad_output.data<T>()/input.data<T>() are raw pointers with NO
    // stride handling -- a non-contiguous tensor silently reads the wrong
    // elements via straight linear indexing.
    Tensor grad_output_cont = grad_output_in.is_contiguous() ? grad_output_in : grad_output_in.contiguous();
    const Tensor& grad_output = grad_output_cont;
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_backward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = gelu_backward_kernel(grad_output_f32, input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("GELU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in gelu_backward_kernel: ") + hipGetErrorString(err));
    }

    if (input.dtype() == DType::Float16) {
        fp16_saturate(result.data_ptr(), result.numel(), stream);
    }

    return result;
}

// Leaky ReLU wrapper
auto leaky_relu_kernel(const Tensor& input_in, double alpha, hipStream_t stream) -> Tensor {
    // JIT-R163: input.data<T>() is a raw pointer with NO stride handling --
    // a non-contiguous input silently reads the wrong elements.
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n, alpha);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n, static_cast<double>(alpha));
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n, alpha);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = leaky_relu_kernel(input_f32, alpha, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Leaky ReLU only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in leaky_relu_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Leaky ReLU backward wrapper
auto leaky_relu_backward_kernel(const Tensor& grad_output_in, const Tensor& input_in, double alpha, hipStream_t stream) -> Tensor {
    // JIT-R163: grad_output.data<T>()/input.data<T>() are raw pointers with NO
    // stride handling -- a non-contiguous tensor silently reads the wrong
    // elements via straight linear indexing.
    Tensor grad_output_cont = grad_output_in.is_contiguous() ? grad_output_in : grad_output_in.contiguous();
    const Tensor& grad_output = grad_output_cont;
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        return result;
    }

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n, alpha);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n, static_cast<double>(alpha));
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_backward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()), reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n, alpha);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = leaky_relu_backward_kernel(grad_output_f32, input_f32, alpha, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Leaky ReLU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in leaky_relu_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Softmax wrapper with temperature scaling
auto softmax_kernel(const Tensor& input_orig, int64_t dim, hipStream_t stream, float temperature = 1.0f) -> Tensor {
    // The kernel below assumes the softmax axis is contiguous and treats
    // the rest of memory as flat batches. That mapping is only correct when
    // `dim` is the LAST axis. For other dims, transpose to put the softmax
    // axis last, recurse, transpose back. Without this, softmax(dim=0) on a
    // (16,16) input reads rows where it should read columns.
    auto orig_shape_span = input_orig.shape();
    int64_t orig_ndim = static_cast<int64_t>(orig_shape_span.size());
    int64_t norm_dim = (dim < 0) ? orig_ndim + dim : dim;
    int64_t last = orig_ndim - 1;
    if (orig_ndim > 0 && norm_dim != last) {
        Tensor transposed = input_orig.transpose(norm_dim, last).contiguous();
        Tensor result = softmax_kernel(transposed, /*dim=*/last, stream, temperature);
        return result.transpose(norm_dim, last).contiguous();
    }

    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
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

    // After the transpose-to-last normalization above, dim is always the
    // last axis. Calculate batch size and dim size accordingly.
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
        hipLaunchKernelGGL(softmax_forward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, input.data<float>(), result.data<float>(),
                          batch_size, dim_size, temperature);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(softmax_forward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, input.data<double>(), result.data<double>(),
                          batch_size, dim_size, static_cast<double>(temperature));
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);  // Compute in float for stability
        hipLaunchKernelGGL(softmax_forward_kernel_fp16, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()),
                          batch_size, dim_size, temperature);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = softmax_kernel(input_f32, dim, stream, temperature);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Softmax only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in softmax_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Softmax backward wrapper
auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, hipStream_t stream) -> Tensor {
    // The device kernel addresses memory as flat batch_size rows of dim_size
    // via row*dim_size+i, which is only valid when the softmax axis is the
    // contiguous innermost axis. Mirror softmax_kernel's normalization: when
    // dim is not the last axis, transpose both grad_output and output so the
    // softmax axis becomes last (and .contiguous() them), recurse with the
    // last dim, then transpose the result back. Without this, softmax_backward
    // over a non-last dim reduces over the wrong group of elements.
    auto orig_shape_span = output.shape();
    int64_t orig_ndim = static_cast<int64_t>(orig_shape_span.size());
    int64_t norm_dim = (dim < 0) ? orig_ndim + dim : dim;
    int64_t last = orig_ndim - 1;
    if (orig_ndim > 0 && norm_dim != last) {
        Tensor grad_t = grad_output.transpose(norm_dim, last).contiguous();
        Tensor out_t = output.transpose(norm_dim, last).contiguous();
        Tensor res = softmax_backward_kernel(grad_t, out_t, /*dim=*/last, stream);
        return res.transpose(norm_dim, last).contiguous();
    }

    // Ensure contiguous inputs since the kernel reads flat row-major memory.
    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor output_c = output.is_contiguous() ? output : output.contiguous();

    std::vector<int64_t> shape(output_c.shape().begin(), output_c.shape().end());
    Tensor result(shape, output_c.dtype(), output_c.device());

    if (output_c.numel() == 0) {
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

    if (output_c.dtype() == DType::Float32) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        hipLaunchKernelGGL(softmax_backward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, grad_output_c.data<float>(), output_c.data<float>(),
                          result.data<float>(), batch_size, dim_size);
    } else if (output_c.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(softmax_backward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, grad_output_c.data<double>(), output_c.data<double>(),
                          result.data<double>(), batch_size, dim_size);
    } else if (output_c.dtype() == DType::Float16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);  // Compute in float for stability
        hipLaunchKernelGGL(softmax_backward_kernel_fp16, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(grad_output_c.data<Float16>()),
                          reinterpret_cast<const __half*>(output_c.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()),
                          batch_size, dim_size);
    } else if (output_c.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output_c.to(DType::Float32);
        auto output_f32 = output_c.to(DType::Float32);
        auto result_f32 = softmax_backward_kernel(grad_output_f32, output_f32, dim, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Softmax backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in softmax_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Log Softmax wrapper
auto log_softmax_kernel(const Tensor& input_orig, int64_t dim, hipStream_t stream) -> Tensor {
    // Same dim-must-be-last assumption as softmax_kernel above. Normalize
    // by transposing dim to last when needed.
    auto orig_shape_span = input_orig.shape();
    int64_t orig_ndim = static_cast<int64_t>(orig_shape_span.size());
    int64_t norm_dim = (dim < 0) ? orig_ndim + dim : dim;
    int64_t last_axis = orig_ndim - 1;
    if (orig_ndim > 0 && norm_dim != last_axis) {
        Tensor transposed = input_orig.transpose(norm_dim, last_axis).contiguous();
        Tensor result = log_softmax_kernel(transposed, /*dim=*/last_axis, stream);
        return result.transpose(norm_dim, last_axis).contiguous();
    }

    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
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
        hipLaunchKernelGGL(log_softmax_forward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, input.data<float>(), result.data<float>(),
                          batch_size, dim_size);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(log_softmax_forward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, input.data<double>(), result.data<double>(),
                          batch_size, dim_size);
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);  // Compute in float for stability
        hipLaunchKernelGGL(log_softmax_forward_kernel_fp16, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()),
                          batch_size, dim_size);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = log_softmax_kernel(input_f32, dim, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Log Softmax only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in log_softmax_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Log Softmax backward wrapper
auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, hipStream_t stream) -> Tensor {
    // Same dim-must-be-last assumption as the device kernel (flat row*dim_size+i
    // indexing). Normalize by transposing the softmax axis to the last position
    // (and .contiguous()) when it is not already last, recurse, transpose back.
    // Mirrors log_softmax_kernel forward; without it, a non-last dim reduces
    // over the wrong group and produces wrong gradients.
    auto orig_shape_span = output.shape();
    int64_t orig_ndim = static_cast<int64_t>(orig_shape_span.size());
    int64_t norm_dim = (dim < 0) ? orig_ndim + dim : dim;
    int64_t last_axis = orig_ndim - 1;
    if (orig_ndim > 0 && norm_dim != last_axis) {
        Tensor grad_t = grad_output.transpose(norm_dim, last_axis).contiguous();
        Tensor out_t = output.transpose(norm_dim, last_axis).contiguous();
        Tensor res = log_softmax_backward_kernel(grad_t, out_t, /*dim=*/last_axis, stream);
        return res.transpose(norm_dim, last_axis).contiguous();
    }

    // Ensure contiguous inputs since the kernel reads flat row-major memory.
    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor output_c = output.is_contiguous() ? output : output.contiguous();

    std::vector<int64_t> shape(output_c.shape().begin(), output_c.shape().end());
    Tensor result(shape, output_c.dtype(), output_c.device());

    if (output_c.numel() == 0) {
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

    if (output_c.dtype() == DType::Float32) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);
        hipLaunchKernelGGL(log_softmax_backward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, grad_output_c.data<float>(), output_c.data<float>(),
                          result.data<float>(), batch_size, dim_size);
    } else if (output_c.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(log_softmax_backward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, grad_output_c.data<double>(), output_c.data<double>(),
                          result.data<double>(), batch_size, dim_size);
    } else if (output_c.dtype() == DType::Float16) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(float);  // Compute in float for stability
        hipLaunchKernelGGL(log_softmax_backward_kernel_fp16, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(grad_output_c.data<Float16>()),
                          reinterpret_cast<const __half*>(output_c.data<Float16>()),
                          reinterpret_cast<__half*>(result.data<Float16>()),
                          batch_size, dim_size);
    } else if (output_c.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output_c.to(DType::Float32);
        auto output_f32 = output_c.to(DType::Float32);
        auto result_f32 = log_softmax_backward_kernel(grad_output_f32, output_f32, dim, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Log Softmax backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in log_softmax_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ============================================================================
// ELU wrapper
// ============================================================================
auto elu_kernel(const Tensor& input_in, float alpha, hipStream_t stream) -> Tensor {
    // JIT-R163: input.data<T>() is a raw pointer with NO stride handling --
    // a non-contiguous input silently reads the wrong elements.
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();

    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(elu_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n, alpha);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(elu_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n, static_cast<double>(alpha));
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(elu_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n, alpha);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = elu_kernel(input_f32, alpha, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("ELU only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in elu_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ELU backward wrapper
auto elu_backward_kernel(const Tensor& grad_output_in, const Tensor& input_in, float alpha, hipStream_t stream) -> Tensor {
    // JIT-R163: grad_output.data<T>()/input.data<T>() are raw pointers with NO
    // stride handling -- a non-contiguous tensor silently reads the wrong
    // elements via straight linear indexing.
    Tensor grad_output_cont = grad_output_in.is_contiguous() ? grad_output_in : grad_output_in.contiguous();
    const Tensor& grad_output = grad_output_cont;
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();

    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(elu_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n, alpha);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(elu_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n, static_cast<double>(alpha));
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(elu_backward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()), reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n, alpha);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = elu_backward_kernel(grad_output_f32, input_f32, alpha, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("ELU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in elu_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ============================================================================
// SELU wrapper
// ============================================================================
auto selu_kernel(const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: input.data<T>() is a raw pointer with NO stride handling --
    // a non-contiguous input silently reads the wrong elements.
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();

    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(selu_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(selu_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(selu_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = selu_kernel(input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("SELU only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in selu_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// SELU backward wrapper
auto selu_backward_kernel(const Tensor& grad_output_in, const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: grad_output.data<T>()/input.data<T>() are raw pointers with NO
    // stride handling -- a non-contiguous tensor silently reads the wrong
    // elements via straight linear indexing.
    Tensor grad_output_cont = grad_output_in.is_contiguous() ? grad_output_in : grad_output_in.contiguous();
    const Tensor& grad_output = grad_output_cont;
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();

    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(selu_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(selu_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(selu_backward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()), reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = selu_backward_kernel(grad_output_f32, input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("SELU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in selu_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ============================================================================
// Swish wrapper
// ============================================================================
auto swish_kernel(const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: input.data<T>() is a raw pointer with NO stride handling --
    // a non-contiguous input silently reads the wrong elements.
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();

    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(swish_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(swish_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(swish_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = swish_kernel(input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Swish only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in swish_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Swish backward wrapper
auto swish_backward_kernel(const Tensor& grad_output_in, const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: grad_output.data<T>()/input.data<T>() are raw pointers with NO
    // stride handling -- a non-contiguous tensor silently reads the wrong
    // elements via straight linear indexing.
    Tensor grad_output_cont = grad_output_in.is_contiguous() ? grad_output_in : grad_output_in.contiguous();
    const Tensor& grad_output = grad_output_cont;
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();

    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(swish_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(swish_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(swish_backward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()), reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = swish_backward_kernel(grad_output_f32, input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Swish backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in swish_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ============================================================================
// Mish wrapper
// ============================================================================
auto mish_kernel(const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: input.data<T>() is a raw pointer with NO stride handling --
    // a non-contiguous input silently reads the wrong elements.
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();

    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(mish_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(mish_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(mish_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = mish_kernel(input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Mish only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in mish_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ============================================================================
// Hardswish / Hardsigmoid — forward-only single-output kernels.
//   hardswish(x)   = x * clamp(x+3, 0, 6) / 6
//   hardsigmoid(x) = clamp(x+3, 0, 6) / 6
// Backward is autograd-composed (clamp + mul chain), matching the CPU backend.
// Float32/Float64 native, Float16 native (widen helpers), BFloat16 via widen.
// ============================================================================
template<typename T>
__global__ void hardswish_forward_kernel(const T* input, T* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T t = x + T(3);
        t = t < T(0) ? T(0) : (t > T(6) ? T(6) : t);
        output[idx] = x * t * (T(1) / T(6));
    }
}

template<typename T>
__global__ void hardsigmoid_forward_kernel(const T* input, T* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T t = x + T(3);
        t = t < T(0) ? T(0) : (t > T(6) ? T(6) : t);
        output[idx] = t * (T(1) / T(6));
    }
}

__global__ void hardswish_forward_kernel_fp16(const __half* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float t = x + 3.0f;
        t = t < 0.0f ? 0.0f : (t > 6.0f ? 6.0f : t);
        output[idx] = tenzor::rocm::safe_f2h(x * t * (1.0f / 6.0f));
    }
}

__global__ void hardsigmoid_forward_kernel_fp16(const __half* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(input[idx]);
        float t = x + 3.0f;
        t = t < 0.0f ? 0.0f : (t > 6.0f ? 6.0f : t);
        output[idx] = tenzor::rocm::safe_f2h(t * (1.0f / 6.0f));
    }
}

auto hardswish_kernel(const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: input.data<T>() is a raw pointer with NO stride handling --
    // a non-contiguous input silently reads the wrong elements.
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();
    if (n == 0) return result;
    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(hardswish_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(hardswish_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(hardswish_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = hardswish_kernel(input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Hardswish only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in hardswish_kernel: ") + hipGetErrorString(err));
    }
    return result;
}

auto hardsigmoid_kernel(const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: input.data<T>() is a raw pointer with NO stride handling --
    // a non-contiguous input silently reads the wrong elements.
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();
    if (n == 0) return result;
    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(hardsigmoid_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(hardsigmoid_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(hardsigmoid_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = hardsigmoid_kernel(input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Hardsigmoid only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in hardsigmoid_kernel: ") + hipGetErrorString(err));
    }
    return result;
}

// Mish backward wrapper
auto mish_backward_kernel(const Tensor& grad_output_in, const Tensor& input_in, hipStream_t stream) -> Tensor {
    // JIT-R163: grad_output.data<T>()/input.data<T>() are raw pointers with NO
    // stride handling -- a non-contiguous tensor silently reads the wrong
    // elements via straight linear indexing.
    Tensor grad_output_cont = grad_output_in.is_contiguous() ? grad_output_in : grad_output_in.contiguous();
    const Tensor& grad_output = grad_output_cont;
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();

    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(mish_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(mish_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(mish_backward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()), reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = mish_backward_kernel(grad_output_f32, input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Mish backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in mish_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ============================================================================
// Softplus wrapper
// ============================================================================
auto softplus_kernel(const Tensor& input_in, float beta, float threshold, hipStream_t stream) -> Tensor {
    // JIT-R163: input.data<T>() is a raw pointer with NO stride handling --
    // a non-contiguous input silently reads the wrong elements.
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();

    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(softplus_forward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<float>(), result.data<float>(), n, beta, threshold);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(softplus_forward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          input.data<double>(), result.data<double>(), n, static_cast<double>(beta), static_cast<double>(threshold));
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(softplus_forward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n, beta, threshold);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = softplus_kernel(input_f32, beta, threshold, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Softplus only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in softplus_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Softplus backward wrapper
auto softplus_backward_kernel(const Tensor& grad_output_in, const Tensor& input_in, float beta, float threshold, hipStream_t stream) -> Tensor {
    // JIT-R163: grad_output.data<T>()/input.data<T>() are raw pointers with NO
    // stride handling -- a non-contiguous tensor silently reads the wrong
    // elements via straight linear indexing.
    Tensor grad_output_cont = grad_output_in.is_contiguous() ? grad_output_in : grad_output_in.contiguous();
    const Tensor& grad_output = grad_output_cont;
    Tensor input_cont = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor& input = input_cont;
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    int64_t n = input.numel();

    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(softplus_backward_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n, beta, threshold);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(softplus_backward_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n, static_cast<double>(beta), static_cast<double>(threshold));
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(softplus_backward_kernel_fp16, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()), reinterpret_cast<const __half*>(input.data<Float16>()), reinterpret_cast<__half*>(result.data<Float16>()), n, beta, threshold);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = softplus_backward_kernel(grad_output_f32, input_f32, beta, threshold, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Softplus backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in softplus_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ============================================================================
// True in-place activation wrappers (Phase 7.1)
//
// These launch the same forward kernels used by the out-of-place path but
// alias input and output to the target tensor's storage. Eliminates the
// D2D-copy + hipStreamSynchronize dance that the registry was previously
// doing around temporary result tensors.
//
// Covers the pointwise activations (ReLU, Sigmoid, Tanh, LeakyReLU, Gelu).
// Float16/BFloat16 aren't straightforward to alias because the reference
// implementation uses a __half* path and BFloat16 is handled via a cast to
// Float32; for those dtypes the inplace wrapper falls back to the existing
// temporary-result pattern (but without the unnecessary sync — the temporary
// result's destruction is safe to defer to ordinary dtor order because the
// Tensor implementation's storage release schedules its own dealloc).
// ============================================================================

void relu_inplace_kernel(Tensor& target, hipStream_t stream) {
    int64_t n = target.numel();
    if (n == 0) return;
    if (target.dtype() == DType::Float32) {
        int nb = get_num_blocks(n);
        hipLaunchKernelGGL(relu_forward_kernel<float>, dim3(nb), dim3(BLOCK_SIZE), 0, stream,
                          target.data<float>(), target.data<float>(), n);
    } else if (target.dtype() == DType::Float64) {
        int nb = get_num_blocks(n);
        hipLaunchKernelGGL(relu_forward_kernel<double>, dim3(nb), dim3(BLOCK_SIZE), 0, stream,
                          target.data<double>(), target.data<double>(), n);
    } else if (target.dtype() == DType::Float16) {
        int nb = get_num_blocks(n);
        auto* p = reinterpret_cast<__half*>(target.data<Float16>());
        hipLaunchKernelGGL(relu_forward_kernel_fp16, dim3(nb), dim3(BLOCK_SIZE), 0, stream, p, p, n);
    } else if (target.dtype() == DType::BFloat16) {
        // BFloat16 uses a promote-to-f32 path; can't alias losslessly. Fall back to a
        // read-modify-write through a temporary but avoid the explicit sync.
        auto tmp = relu_kernel(target, stream);
        HIP_CHECK(hipMemcpyAsync(target.data_ptr(), tmp.data_ptr(),
            target.numel() * dtype_size(target.dtype()), hipMemcpyDeviceToDevice, stream));
        return;
    } else {
        throw std::runtime_error("ReLU in-place: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
}

void sigmoid_inplace_kernel(Tensor& target, hipStream_t stream) {
    int64_t n = target.numel();
    if (n == 0) return;
    if (target.dtype() == DType::Float32) {
        int nb = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_forward_kernel<float>, dim3(nb), dim3(BLOCK_SIZE), 0, stream,
                          target.data<float>(), target.data<float>(), n);
    } else if (target.dtype() == DType::Float64) {
        int nb = get_num_blocks(n);
        hipLaunchKernelGGL(sigmoid_forward_kernel<double>, dim3(nb), dim3(BLOCK_SIZE), 0, stream,
                          target.data<double>(), target.data<double>(), n);
    } else if (target.dtype() == DType::Float16) {
        int nb = get_num_blocks(n);
        auto* p = reinterpret_cast<__half*>(target.data<Float16>());
        hipLaunchKernelGGL(sigmoid_forward_kernel_fp16, dim3(nb), dim3(BLOCK_SIZE), 0, stream, p, p, n);
    } else if (target.dtype() == DType::BFloat16) {
        auto tmp = sigmoid_kernel(target, stream);
        HIP_CHECK(hipMemcpyAsync(target.data_ptr(), tmp.data_ptr(),
            target.numel() * dtype_size(target.dtype()), hipMemcpyDeviceToDevice, stream));
        return;
    } else {
        throw std::runtime_error("Sigmoid in-place: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
}

void tanh_inplace_kernel(Tensor& target, hipStream_t stream) {
    int64_t n = target.numel();
    if (n == 0) return;
    if (target.dtype() == DType::Float32) {
        int nb = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_forward_kernel<float>, dim3(nb), dim3(BLOCK_SIZE), 0, stream,
                          target.data<float>(), target.data<float>(), n);
    } else if (target.dtype() == DType::Float64) {
        int nb = get_num_blocks(n);
        hipLaunchKernelGGL(tanh_forward_kernel<double>, dim3(nb), dim3(BLOCK_SIZE), 0, stream,
                          target.data<double>(), target.data<double>(), n);
    } else if (target.dtype() == DType::Float16) {
        int nb = get_num_blocks(n);
        auto* p = reinterpret_cast<__half*>(target.data<Float16>());
        hipLaunchKernelGGL(tanh_forward_kernel_fp16, dim3(nb), dim3(BLOCK_SIZE), 0, stream, p, p, n);
    } else if (target.dtype() == DType::BFloat16) {
        auto tmp = tanh_kernel(target, stream);
        HIP_CHECK(hipMemcpyAsync(target.data_ptr(), tmp.data_ptr(),
            target.numel() * dtype_size(target.dtype()), hipMemcpyDeviceToDevice, stream));
        return;
    } else {
        throw std::runtime_error("Tanh in-place: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
}

void leaky_relu_inplace_kernel(Tensor& target, float alpha, hipStream_t stream) {
    int64_t n = target.numel();
    if (n == 0) return;
    if (target.dtype() == DType::Float32) {
        int nb = get_num_blocks(n);
        // Kernel signature is (input, output, n, alpha) — see leaky_relu_forward_kernel
        // at the top of this file. Earlier code passed (alpha, n) here, so n was
        // truncated from a float to 0 and the loop never ran (the input was
        // returned unchanged). Caught by tests/backend_parity/test_inplace_ops_parity.
        hipLaunchKernelGGL(leaky_relu_forward_kernel<float>, dim3(nb), dim3(BLOCK_SIZE), 0, stream,
                          target.data<float>(), target.data<float>(), n, alpha);
    } else if (target.dtype() == DType::Float64) {
        int nb = get_num_blocks(n);
        hipLaunchKernelGGL(leaky_relu_forward_kernel<double>, dim3(nb), dim3(BLOCK_SIZE), 0, stream,
                          target.data<double>(), target.data<double>(), n, static_cast<double>(alpha));
    } else if (target.dtype() == DType::Float16) {
        int nb = get_num_blocks(n);
        auto* p = reinterpret_cast<__half*>(target.data<Float16>());
        hipLaunchKernelGGL(leaky_relu_forward_kernel_fp16, dim3(nb), dim3(BLOCK_SIZE), 0, stream, p, p, n, alpha);
    } else if (target.dtype() == DType::BFloat16) {
        auto tmp = leaky_relu_kernel(target, alpha, stream);
        HIP_CHECK(hipMemcpyAsync(target.data_ptr(), tmp.data_ptr(),
            target.numel() * dtype_size(target.dtype()), hipMemcpyDeviceToDevice, stream));
        return;
    } else {
        throw std::runtime_error("LeakyReLU in-place: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
}

void gelu_inplace_kernel(Tensor& target, hipStream_t stream) {
    int64_t n = target.numel();
    if (n == 0) return;
    if (target.dtype() == DType::Float32) {
        int nb = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_forward_kernel<float>, dim3(nb), dim3(BLOCK_SIZE), 0, stream,
                          target.data<float>(), target.data<float>(), n);
    } else if (target.dtype() == DType::Float64) {
        int nb = get_num_blocks(n);
        hipLaunchKernelGGL(gelu_forward_kernel<double>, dim3(nb), dim3(BLOCK_SIZE), 0, stream,
                          target.data<double>(), target.data<double>(), n);
    } else if (target.dtype() == DType::Float16) {
        int nb = get_num_blocks(n);
        auto* p = reinterpret_cast<__half*>(target.data<Float16>());
        hipLaunchKernelGGL(gelu_forward_kernel_fp16, dim3(nb), dim3(BLOCK_SIZE), 0, stream, p, p, n);
    } else if (target.dtype() == DType::BFloat16) {
        auto tmp = gelu_kernel(target, stream);
        HIP_CHECK(hipMemcpyAsync(target.data_ptr(), tmp.data_ptr(),
            target.numel() * dtype_size(target.dtype()), hipMemcpyDeviceToDevice, stream));
        return;
    } else {
        throw std::runtime_error("Gelu in-place: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
}

// ============================================================================
// RReLU: Randomized Leaky ReLU
// ============================================================================

// Eval mode: slope = (lower + upper) / 2
template<typename T>
__global__ void rrelu_eval_kernel(const T* input, T* output, int64_t n, T slope) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        output[idx] = (x >= T(0)) ? x : slope * x;
    }
}

// Philox4x32-10 counter-based PRNG (HIP device port — same algorithm as the
// CPU/CUDA implementations so per-element slopes are well-distributed i.i.d.
// Uniform samples rather than the near-ramp output of a single LCG step).
__device__ __forceinline__ void rrelu_philox_round(uint32_t ctr[4], const uint32_t key[2]) {
    constexpr uint64_t M0 = 0xD2511F53ULL;
    constexpr uint64_t M1 = 0xCD9E8D57ULL;
    uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
    uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);
    uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
    uint32_t lo0 = static_cast<uint32_t>(prod0);
    uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
    uint32_t lo1 = static_cast<uint32_t>(prod1);
    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
}

// Returns a Uniform[0,1) sample keyed on (seed, idx) via Philox4x32-10.
__device__ __forceinline__ float rrelu_philox_uniform(unsigned long long seed, int64_t idx) {
    uint32_t ctr[4] = {
        static_cast<uint32_t>(idx),
        static_cast<uint32_t>(static_cast<uint64_t>(idx) >> 32),
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32)
    };
    uint32_t k[2] = {static_cast<uint32_t>(seed),
                     static_cast<uint32_t>(seed >> 32) ^ 0x1BD11BDAU};
    constexpr uint32_t W0 = 0x9E3779B9U;
    constexpr uint32_t W1 = 0xBB67AE85U;
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        rrelu_philox_round(ctr, k);
        if (r < 9) { k[0] += W0; k[1] += W1; }
    }
    // 24-bit mantissa precision Uniform[0,1).
    return static_cast<float>(ctr[0] >> 8) * (1.0f / 16777216.0f);
}

// Train mode: per-element random slope via counter-based Philox RNG keyed on
// (seed, idx). The seed honors tenzor::manual_seed (folded with a process-wide
// SplitMix64 counter host-side), so results are reproducible and two launches in
// the same clock tick do not collide.
__global__ void rrelu_train_kernel_f32(const float* input, float* output, int64_t n,
                                       float lower, float upper, unsigned long long seed) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = input[idx];
        if (x >= 0.0f) {
            output[idx] = x;
        } else {
            float u = rrelu_philox_uniform(seed, idx);
            float slope = lower + u * (upper - lower);
            output[idx] = slope * x;
        }
    }
}

// Type-generic RReLU training kernel: the Philox uniform is computed in float
// (slope precision is not the concern) but the multiply happens in T, so a
// Float64 model gets the defining per-element random negative slope instead of
// silently falling back to the deterministic eval-mode midpoint.
template<typename T>
__global__ void rrelu_train_kernel(const T* input, T* output, int64_t n,
                                   float lower, float upper, unsigned long long seed) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        if (x >= T(0)) {
            output[idx] = x;
        } else {
            float u = rrelu_philox_uniform(seed, idx);
            T slope = static_cast<T>(lower + u * (upper - lower));
            output[idx] = slope * x;
        }
    }
}

// RReLU backward
template<typename T>
__global__ void rrelu_backward_kernel_impl(const T* grad, const T* input, T* output,
                                           int64_t n, T slope) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = (input[idx] >= T(0)) ? grad[idx] : slope * grad[idx];
    }
}

// LogSigmoid backward: grad * sigmoid(-x)
template<typename T>
__global__ void log_sigmoid_backward_kernel_impl(const T* grad, const T* input, T* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T x = input[idx];
        T sig_neg_x = (x >= T(0)) ? exp(-x) / (T(1) + exp(-x)) : T(1) / (T(1) + exp(x));
        output[idx] = grad[idx] * sig_neg_x;
    }
}

// Float32 specialisation for LogSigmoid backward (use expf)
__global__ void log_sigmoid_backward_kernel_f32(const float* grad, const float* input,
                                                 float* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float x = input[idx];
        float sig_neg_x = (x >= 0.0f) ? expf(-x) / (1.0f + expf(-x)) : 1.0f / (1.0f + expf(x));
        output[idx] = grad[idx] * sig_neg_x;
    }
}

// ---- dispatch wrappers ----

auto rrelu_kernel(const Tensor& input, float lower, float upper, bool training, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;

    float mid = (lower + upper) / 2.0f;
    int num_blocks = get_num_blocks(n);

    // Seed from the library's global RNG so manual_seed is honored / RReLU is
    // reproducible. Fold in a process-wide monotonic counter with SplitMix64
    // mixing so two invocations within the same clock tick cannot collide,
    // matching the CUDA RReLU implementation.
    auto make_rrelu_seed = []() -> unsigned long long {
        static std::atomic<uint64_t> rrelu_call_counter{0};
        uint64_t mix = rrelu_call_counter.fetch_add(1, std::memory_order_relaxed);
        mix = (mix + 0x9E3779B97F4A7C15ULL);
        mix = (mix ^ (mix >> 30)) * 0xBF58476D1CE4E5B9ULL;
        mix = (mix ^ (mix >> 27)) * 0x94D049BB133111EBULL;
        mix = mix ^ (mix >> 31);
        return static_cast<unsigned long long>(::tenzor::get_global_seed() ^ mix);
    };

    if (input.dtype() == DType::Float32) {
        if (training) {
            unsigned long long seed = make_rrelu_seed();
            hipLaunchKernelGGL(rrelu_train_kernel_f32, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                              input.data<float>(), result.data<float>(), n, lower, upper, seed);
        } else {
            hipLaunchKernelGGL(rrelu_eval_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                              input.data<float>(), result.data<float>(), n, mid);
        }
    } else if (input.dtype() == DType::Float64) {
        if (training) {
            // Native Float64 stochastic slope (the Philox uniform is computed in
            // float, the slope*x multiply in double) — RReLU training must apply
            // a per-element random negative slope, not the deterministic midpoint.
            unsigned long long seed = make_rrelu_seed();
            hipLaunchKernelGGL(rrelu_train_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                              input.data<double>(), result.data<double>(), n, lower, upper, seed);
        } else {
            double mid_d = static_cast<double>(mid);
            hipLaunchKernelGGL(rrelu_eval_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                              input.data<double>(), result.data<double>(), n, mid_d);
        }
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rrelu_kernel(input_f32, lower, upper, training, stream);
        return result_f32.to(input.dtype());
    } else {
        throw std::runtime_error("RReLU only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto rrelu_backward_kernel(const Tensor& grad_output, const Tensor& input,
                           float lower, float upper, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;

    float mid = (lower + upper) / 2.0f;
    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(rrelu_backward_kernel_impl<float>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n, mid);
    } else if (input.dtype() == DType::Float64) {
        double mid_d = static_cast<double>(mid);
        hipLaunchKernelGGL(rrelu_backward_kernel_impl<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n, mid_d);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = rrelu_backward_kernel(grad_f32, input_f32, lower, upper, stream);
        return result_f32.to(input.dtype());
    } else {
        throw std::runtime_error("RReLU backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto log_sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;

    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(log_sigmoid_backward_kernel_f32, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<float>(), input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(log_sigmoid_backward_kernel_impl<double>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_output.data<double>(), input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = log_sigmoid_backward_kernel(grad_f32, input_f32, stream);
        return result_f32.to(input.dtype());
    } else {
        throw std::runtime_error("LogSigmoid backward only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// IndexAdd, IndexCopy, IndexFill HIP kernels
// ============================================================================

// Atomic add helper local to this TU (mirrors the pattern in indexing.hip.cpp):
// HIP's built-in atomicAdd natively covers float/double/int32_t; int64_t has
// no native HIP atomicAdd overload and needs a CAS loop.
template<typename T>
__device__ __forceinline__ T index_atomic_add(T* address, T val) {
    return atomicAdd(address, val);
}

template<>
__device__ __forceinline__ int64_t index_atomic_add<int64_t>(int64_t* address, int64_t val) {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                         static_cast<unsigned long long int>(static_cast<int64_t>(assumed) + val));
    } while (assumed != old);
    return static_cast<int64_t>(old);
}

template<typename T>
__global__ void index_add_kernel_impl(T* output, const T* source, const int64_t* index,
                                     int64_t outer, int64_t dim_size, int64_t idx_n, int64_t inner,
                                     int* error_flag) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * idx_n * inner;
    if (tid >= total) return;
    int64_t j = tid % inner;
    int64_t k = (tid / inner) % idx_n;
    int64_t o = tid / (inner * idx_n);
    // Bounds-check the gathered index before using it as a device offset; an
    // out-of-range index would otherwise be an OOB read/write. The host throws
    // after synchronizing (matching the CPU reference, which throws
    // std::out_of_range).
    int64_t ix = index[k];
    if (ix < 0) ix += dim_size;  // F080: wrap negatives (matches CPU/CUDA)
    if (ix < 0 || ix >= dim_size) {
        atomicOr(error_flag, 1);
        return;
    }
    index_atomic_add(&output[(o * dim_size + ix) * inner + j],
                      source[(o * idx_n + k) * inner + j]);
}

template<typename T>
__global__ void index_copy_kernel_impl(T* output, const T* source, const int64_t* index,
                                      int64_t outer, int64_t dim_size, int64_t idx_n, int64_t inner,
                                      int* error_flag) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * idx_n * inner;
    if (tid >= total) return;
    int64_t j = tid % inner;
    int64_t k = (tid / inner) % idx_n;
    int64_t o = tid / (inner * idx_n);
    // Bounds-check the gathered index before using it as a device offset; an
    // out-of-range index would otherwise be an OOB write. The host throws after
    // synchronizing (matching the CPU reference, which throws std::out_of_range).
    int64_t ix = index[k];
    if (ix < 0) ix += dim_size;  // F080: wrap negatives (matches CPU/CUDA)
    if (ix < 0 || ix >= dim_size) {
        atomicOr(error_flag, 1);
        return;
    }
    output[(o * dim_size + ix) * inner + j] = source[(o * idx_n + k) * inner + j];
}

template<typename T>
__global__ void index_fill_kernel_impl(T* output, const int64_t* index, T value,
                                      int64_t outer, int64_t dim_size, int64_t idx_n, int64_t inner,
                                      int* error_flag) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * idx_n * inner;
    if (tid >= total) return;
    int64_t j = tid % inner;
    int64_t k = (tid / inner) % idx_n;
    int64_t o = tid / (inner * idx_n);
    // Bounds-check the gathered index before using it as a device offset; an
    // out-of-range index would otherwise be an OOB write. The host throws after
    // synchronizing (matching the CPU reference, which throws std::out_of_range).
    int64_t ix = index[k];
    if (ix < 0) ix += dim_size;  // F080: wrap negatives (matches CPU/CUDA)
    if (ix < 0 || ix >= dim_size) {
        atomicOr(error_flag, 1);
        return;
    }
    output[(o * dim_size + ix) * inner + j] = value;
}

// Shared launch+bounds-check helper for the IndexAdd/IndexCopy/IndexFill native
// dtype paths below. Device flag for out-of-range indices is checked on the
// host after sync so we can throw (matching the CPU reference) instead of an
// OOB write.
template<typename T, typename Launch>
static void index_scatter_launch(Tensor& output, int64_t total, hipStream_t stream,
                                  const char* op_name, Launch&& launch) {
    int num_blocks = get_num_blocks(total);
    Tensor err_tensor(std::vector<int64_t>{1}, DType::Int32, output.device());
    HIP_CHECK(hipMemsetAsync(err_tensor.data_ptr(), 0, sizeof(int32_t), stream));
    int* err_ptr = err_tensor.data<int32_t>();
    launch(num_blocks, err_ptr);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    if (err_tensor.to(Device::cpu()).data<int32_t>()[0] != 0) {
        throw std::out_of_range(std::string(op_name) + " ROCm: index out of range");
    }
}

auto index_add_kernel(const Tensor& self, const Tensor& index, const Tensor& source,
                      int64_t dim, hipStream_t stream) -> Tensor {
    auto output = self.clone();
    auto shape = output.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    int64_t idx_n = index.numel();
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];
    int64_t total = outer * idx_n * inner;

    if (total == 0) return output;

    // Native fast paths: Float32/Float64 use HIP's built-in atomicAdd, Int32
    // likewise (native atomicAdd(int*, int)), Int64 via the CAS-based
    // index_atomic_add specialization above. Everything else (Float16,
    // BFloat16, and any other dtype) round-trips through Float32.
    auto dispatch = [&]<typename T>() {
        index_scatter_launch<T>(output, total, stream, "index_add",
            [&](int num_blocks, int* err_ptr) {
                hipLaunchKernelGGL(index_add_kernel_impl<T>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                                  output.data<T>(), source.data<T>(), index.data<int64_t>(),
                                  outer, dim_size, idx_n, inner, err_ptr);
            });
    };
    // UInt32/UInt64 must NOT take the Float32-widen fallback below: unlike
    // Int8/16/UInt8/16/Bool (always exact through a 24-bit Float32 mantissa),
    // UInt32/UInt64 magnitudes routinely exceed 2^24, so round-tripping the
    // WHOLE self/source tensors through Float32 would silently corrupt
    // untouched elements too, not just the ones being added into. Reuse the
    // native Int32/Int64 kernels via a raw bit-pattern reinterpret instead:
    // two's-complement addition (including atomicAdd's CAS-based wraparound
    // for int64) produces the identical result bit pattern whether the
    // operand is interpreted as signed or unsigned, so this is exact — not
    // an approximation. Mirrors index_fill_kernel's dispatch_reinterpret
    // below for the same reasoning.
    auto dispatch_reinterpret = [&]<typename T>() {
        index_scatter_launch<T>(output, total, stream, "index_add",
            [&](int num_blocks, int* err_ptr) {
                hipLaunchKernelGGL(index_add_kernel_impl<T>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                                  reinterpret_cast<T*>(output.data_ptr()),
                                  reinterpret_cast<const T*>(source.data_ptr()),
                                  index.data<int64_t>(),
                                  outer, dim_size, idx_n, inner, err_ptr);
            });
    };
    switch (output.dtype()) {
        case DType::Float32: dispatch.template operator()<float>(); return output;
        case DType::Float64: dispatch.template operator()<double>(); return output;
        case DType::Int32:   dispatch.template operator()<int32_t>(); return output;
        case DType::Int64:   dispatch.template operator()<int64_t>(); return output;
        case DType::UInt32:  dispatch_reinterpret.template operator()<int32_t>(); return output;
        case DType::UInt64:  dispatch_reinterpret.template operator()<int64_t>(); return output;
        default: {
            auto f32_self = self.to(DType::Float32);
            auto f32_src = source.to(DType::Float32);
            auto result = index_add_kernel(f32_self, index, f32_src, dim, stream);
            return result.to(self.dtype());
        }
    }
}

auto index_copy_kernel(const Tensor& self, const Tensor& index, const Tensor& source,
                       int64_t dim, hipStream_t stream) -> Tensor {
    auto output = self.clone();
    auto shape = output.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    int64_t idx_n = index.numel();
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];
    int64_t total = outer * idx_n * inner;

    if (total == 0) return output;

    // Native fast paths for Float32/Float64/Int32/Int64 (plain assignment, no
    // atomics required); Float16/BFloat16/other dtypes round-trip through
    // Float32.
    auto dispatch = [&]<typename T>() {
        index_scatter_launch<T>(output, total, stream, "index_copy",
            [&](int num_blocks, int* err_ptr) {
                hipLaunchKernelGGL(index_copy_kernel_impl<T>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                                  output.data<T>(), source.data<T>(), index.data<int64_t>(),
                                  outer, dim_size, idx_n, inner, err_ptr);
            });
    };
    // UInt32/UInt64 must NOT take the Float32-widen fallback below (same
    // reasoning as index_add_kernel above): magnitudes routinely exceed
    // Float32's 24-bit mantissa, so widening the WHOLE self/source tensors
    // would silently corrupt untouched elements. index_copy is a plain
    // scalar overwrite with no signedness-dependent arithmetic at all, so
    // reusing the native Int32/Int64 kernel via a raw bit-pattern
    // reinterpret is exact. Mirrors index_fill_kernel's dispatch_reinterpret
    // below.
    auto dispatch_reinterpret = [&]<typename T>() {
        index_scatter_launch<T>(output, total, stream, "index_copy",
            [&](int num_blocks, int* err_ptr) {
                hipLaunchKernelGGL(index_copy_kernel_impl<T>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                                  reinterpret_cast<T*>(output.data_ptr()),
                                  reinterpret_cast<const T*>(source.data_ptr()),
                                  index.data<int64_t>(),
                                  outer, dim_size, idx_n, inner, err_ptr);
            });
    };
    switch (output.dtype()) {
        case DType::Float32: dispatch.template operator()<float>(); return output;
        case DType::Float64: dispatch.template operator()<double>(); return output;
        case DType::Int32:   dispatch.template operator()<int32_t>(); return output;
        case DType::Int64:   dispatch.template operator()<int64_t>(); return output;
        case DType::UInt32:  dispatch_reinterpret.template operator()<int32_t>(); return output;
        case DType::UInt64:  dispatch_reinterpret.template operator()<int64_t>(); return output;
        default: {
            auto f32_self = self.to(DType::Float32);
            auto f32_src = source.to(DType::Float32);
            auto result = index_copy_kernel(f32_self, index, f32_src, dim, stream);
            return result.to(self.dtype());
        }
    }
}

auto index_fill_kernel(const Tensor& self, const Tensor& index,
                       int64_t dim, double value, hipStream_t stream) -> Tensor {
    auto output = self.clone();
    auto shape = output.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    int64_t idx_n = index.numel();
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];
    int64_t total = outer * idx_n * inner;

    if (total == 0) return output;

    // Native fast paths for Float32/Float64/Int32/Int64; Float16/BFloat16 and
    // narrow integers (Int8/16, UInt8/16, Bool) round-trip through Float32
    // (always exact for those widths -- no magnitude can exceed Float32's
    // 24-bit mantissa). UInt32/UInt64 must NOT take that fallback: they reuse
    // the Int32/Int64 shaders via a raw bit-pattern reinterpret instead (a
    // plain scalar write has no signedness-dependent arithmetic, so the
    // int32_t/int64_t shader is bit-identical for the unsigned types) --
    // otherwise the ENTIRE self tensor (not just the fill positions) would
    // round-trip through Float32 and silently lose precision above 2^24.
    auto dispatch = [&]<typename T>() {
        index_scatter_launch<T>(output, total, stream, "index_fill",
            [&](int num_blocks, int* err_ptr) {
                hipLaunchKernelGGL(index_fill_kernel_impl<T>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                                  output.data<T>(), index.data<int64_t>(), static_cast<T>(value),
                                  outer, dim_size, idx_n, inner, err_ptr);
            });
    };
    auto dispatch_reinterpret = [&]<typename T>(T bit_value) {
        index_scatter_launch<T>(output, total, stream, "index_fill",
            [&](int num_blocks, int* err_ptr) {
                hipLaunchKernelGGL(index_fill_kernel_impl<T>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                                  reinterpret_cast<T*>(output.data_ptr()), index.data<int64_t>(), bit_value,
                                  outer, dim_size, idx_n, inner, err_ptr);
            });
    };
    switch (output.dtype()) {
        case DType::Float32: dispatch.template operator()<float>(); return output;
        case DType::Float64: dispatch.template operator()<double>(); return output;
        case DType::Int32:   dispatch.template operator()<int32_t>(); return output;
        case DType::Int64:   dispatch.template operator()<int64_t>(); return output;
        case DType::UInt32:
            dispatch_reinterpret.template operator()<int32_t>(
                static_cast<int32_t>(static_cast<uint32_t>(value)));
            return output;
        case DType::UInt64:
            dispatch_reinterpret.template operator()<int64_t>(
                static_cast<int64_t>(static_cast<uint64_t>(value)));
            return output;
        default: {
            auto f32_self = self.to(DType::Float32);
            auto result = index_fill_kernel(f32_self, index, dim, value, stream);
            return result.to(self.dtype());
        }
    }
}

// ============================================================================
// ScatterReduce HIP kernels
// ============================================================================

// Scatter-reduce modes: 0=sum, 1=prod, 2=mean, 3=amax, 4=amin
__global__ void scatter_reduce_f32_kernel(float* output, const float* source, const int64_t* index,
                                           int64_t outer, int64_t dim_size, int64_t idx_n, int64_t inner,
                                           int mode, int* counts, int* error_flag) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * idx_n * inner;
    if (tid >= total) return;
    int64_t j = tid % inner;
    int64_t k = (tid / inner) % idx_n;
    int64_t o = tid / (inner * idx_n);
    // Phase 7.6 2D scatter fix: index lookup must include outer/inner offsets.
    int64_t idx_pos = (o * idx_n + k) * inner + j;
    // Bounds-check the gathered index before using it as a device offset; an
    // out-of-range index would otherwise be an OOB atomic write. The host throws
    // after synchronizing (matching the CPU reference, which throws
    // std::out_of_range).
    int64_t ix = index[idx_pos];
    if (ix < 0 || ix >= dim_size) {
        atomicOr(error_flag, 1);
        return;
    }
    int64_t out_pos = (o * dim_size + ix) * inner + j;
    int64_t src_pos = idx_pos;
    float val = source[src_pos];

    if (mode == 0 || mode == 2) {
        atomicAdd(&output[out_pos], val);
        if (mode == 2 && counts) atomicAdd(&counts[out_pos], 1);
    } else if (mode == 1) {
        // prod: CAS loop
        unsigned int* addr = reinterpret_cast<unsigned int*>(&output[out_pos]);
        unsigned int old_bits = __atomic_load_n(addr, __ATOMIC_RELAXED), new_bits;
        do {
            float old_val = __uint_as_float(old_bits);
            new_bits = __float_as_uint(old_val * val);
        } while (!__atomic_compare_exchange_n(addr, &old_bits, new_bits, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    } else if (mode == 3) {
        // amax: CAS loop
        unsigned int* addr = reinterpret_cast<unsigned int*>(&output[out_pos]);
        unsigned int old_bits = __atomic_load_n(addr, __ATOMIC_RELAXED), new_bits;
        do {
            float old_val = __uint_as_float(old_bits);
            if (val <= old_val) break;
            new_bits = __float_as_uint(val);
        } while (!__atomic_compare_exchange_n(addr, &old_bits, new_bits, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    } else if (mode == 4) {
        // amin: CAS loop
        unsigned int* addr = reinterpret_cast<unsigned int*>(&output[out_pos]);
        unsigned int old_bits = __atomic_load_n(addr, __ATOMIC_RELAXED), new_bits;
        do {
            float old_val = __uint_as_float(old_bits);
            if (val >= old_val) break;
            new_bits = __float_as_uint(val);
        } while (!__atomic_compare_exchange_n(addr, &old_bits, new_bits, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    }
}

__global__ void scatter_reduce_init_f32_kernel(float* output, const int64_t* index,
                                                int64_t outer, int64_t dim_size, int64_t idx_n, int64_t inner,
                                                int mode, int* error_flag) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * idx_n * inner;
    if (tid >= total) return;
    int64_t j = tid % inner;
    int64_t k = (tid / inner) % idx_n;
    int64_t o = tid / (inner * idx_n);
    // Phase 7.6 2D scatter fix: index lookup must include outer/inner offsets.
    int64_t idx_pos = (o * idx_n + k) * inner + j;
    // Bounds-check the gathered index before using it as a device offset; an
    // out-of-range index would otherwise be an OOB write. The host throws after
    // synchronizing (matching the CPU reference, which throws std::out_of_range).
    int64_t ix = index[idx_pos];
    if (ix < 0 || ix >= dim_size) {
        atomicOr(error_flag, 1);
        return;
    }
    int64_t out_pos = (o * dim_size + ix) * inner + j;

    float identity;
    if (mode == 0 || mode == 2) identity = 0.0f;
    else if (mode == 1) identity = 1.0f;
    else if (mode == 3) identity = -3.402823466e+38f;
    else identity = 3.402823466e+38f;

    output[out_pos] = identity;
}

// Mirror of CUDA scatter_reduce_mean_div_f32 (Phase 7.6 mean fix). counts
// = number of scatters touching this position; self is NOT counted in the
// kernel. With include_self=true the accumulator is
// `input + sum(scatters)` so the divisor is `count + 1`. Untouched
// positions keep their initial input value.
__global__ void scatter_reduce_mean_div_f32_kernel(float* output, const int* counts, int64_t numel, int include_self) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    int c = counts[tid];
    if (c <= 0) return;
    float divisor = include_self ? static_cast<float>(c + 1) : static_cast<float>(c);
    output[tid] /= divisor;
}

auto scatter_reduce_kernel(const Tensor& self, const Tensor& index, const Tensor& source,
                           int64_t dim, const std::string& reduce, bool include_self,
                           hipStream_t stream) -> Tensor {
    // Non-Float32: convert to Float32 and recurse
    if (self.dtype() != DType::Float32) {
        auto f32_self = self.to(DType::Float32);
        auto f32_src = source.to(DType::Float32);
        auto r = scatter_reduce_kernel(f32_self, index, f32_src, dim, reduce, include_self, stream);
        return r.to(self.dtype());
    }

    auto output = self.clone();
    auto shape = output.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    // Phase 7.6 2D scatter fix: idx_n must be the index tensor's size along
    // the scatter dim, NOT the total numel. For 1D scatter these are equal.
    int64_t idx_n = (index.shape().size() > static_cast<size_t>(dim))
                  ? static_cast<int64_t>(index.shape()[dim])
                  : index.numel();
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

    int num_blocks = get_num_blocks(total);

    // Device flag for out-of-range indices; checked on the host after sync so we
    // can throw (matching the CPU reference, which throws std::out_of_range)
    // instead of an OOB atomic write. Shared by the init and reduce kernels.
    Tensor err_tensor(std::vector<int64_t>{1}, DType::Int32, output.device());
    HIP_CHECK(hipMemsetAsync(err_tensor.data_ptr(), 0, sizeof(int32_t), stream));
    int* err_ptr = err_tensor.data<int32_t>();

    if (!include_self) {
        hipLaunchKernelGGL(scatter_reduce_init_f32_kernel, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                          output.data<float>(), index.data<int64_t>(),
                          outer, dim_size, idx_n, inner, mode, err_ptr);
    }

    // Allocate count tensor for mean mode
    Tensor count_tensor;
    int* count_ptr = nullptr;
    if (mode == 2) {
        int64_t out_numel = output.numel();
        count_tensor = Tensor({out_numel}, DType::Int32, output.device());
        HIP_CHECK(hipMemsetAsync(count_tensor.data<int>(), 0, out_numel * sizeof(int), stream));
        count_ptr = count_tensor.data<int>();
    }

    hipLaunchKernelGGL(scatter_reduce_f32_kernel, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                      output.data<float>(), source.data<float>(), index.data<int64_t>(),
                      outer, dim_size, idx_n, inner, mode, count_ptr, err_ptr);

    if (mode == 2) {
        int64_t out_numel = output.numel();
        int div_blocks = get_num_blocks(out_numel);
        hipLaunchKernelGGL(scatter_reduce_mean_div_f32_kernel, dim3(div_blocks), dim3(BLOCK_SIZE), 0, stream,
                          output.data<float>(), count_ptr, out_numel, include_self ? 1 : 0);
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    if (err_tensor.to(Device::cpu()).data<int32_t>()[0] != 0) {
        throw std::out_of_range("scatter_reduce ROCm: index out of range");
    }
    return output;
}

} // namespace rocm
} // namespace tenzor
