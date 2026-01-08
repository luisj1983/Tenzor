#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include <vector>

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
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Grid-stride loop helper
#define HIP_GRID_STRIDE_LOOP(i, n) \
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
    HIP_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = input[idx] > T(0) ? input[idx] : T(0);
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

// Backward: grad_out * (1 - tanh(x)^2)
template<typename T>
__global__ void tanh_backward_kernel(const T* grad_output, const T* input,
                                    T* grad_input, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        T tanh_x = tanh(input[idx]);
        grad_input[idx] = grad_output[idx] * (T(1) - tanh_x * tanh_x);
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

// GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Approximation commonly used in transformers
template<typename T>
__device__ __forceinline__ T gelu_forward_impl(T x) {
    const T sqrt_2_over_pi = T(0.7978845608028654); // sqrt(2/pi)
    const T coeff = T(0.044715);
    T x_cubed = x * x * x;
    T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return T(0.5) * x * (T(1) + tanh(inner));
}

// GELU backward derivative
template<typename T>
__device__ __forceinline__ T gelu_backward_impl(T x) {
    const T sqrt_2_over_pi = T(0.7978845608028654);
    const T coeff = T(0.044715);
    T x_squared = x * x;
    T x_cubed = x_squared * x;
    T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    T tanh_inner = tanh(inner);
    T sech_inner_squared = T(1) - tanh_inner * tanh_inner;

    T d_inner_dx = sqrt_2_over_pi * (T(1) + T(3) * coeff * x_squared);
    return T(0.5) * (T(1) + tanh_inner) + T(0.5) * x * sech_inner_squared * d_inner_dx;
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

// Warp-level reduction using shuffle instructions
// AMD GPUs have 64-wide wavefronts (vs NVIDIA's 32-wide warps)
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    // AMD wavefront size is 64
    for (int offset = 32; offset > 0; offset /= 2) {
        val = max(val, __shfl_down(val, offset));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    // AMD wavefront size is 64
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
// Optimized for AMD GPU memory hierarchy (LDS - Local Data Share)
template<typename T>
__device__ T block_reduce_max(T val, T* shared) {
    int lane = threadIdx.x % 64;  // AMD wavefront size
    int wid = threadIdx.x / 64;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 64) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

template<typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % 64;  // AMD wavefront size
    int wid = threadIdx.x / 64;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 64) ? shared[lane] : T(0);
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
auto relu_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("ReLU only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in relu_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ReLU backward wrapper
auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("ReLU backward only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in relu_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Sigmoid wrapper
auto sigmoid_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Sigmoid only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in sigmoid_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Sigmoid backward wrapper
auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Sigmoid backward only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in sigmoid_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Tanh wrapper
auto tanh_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Tanh only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in tanh_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Tanh backward wrapper
auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Tanh backward only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in tanh_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// GELU wrapper
auto gelu_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("GELU only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in gelu_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// GELU backward wrapper
auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("GELU backward only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in gelu_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Leaky ReLU wrapper
auto leaky_relu_kernel(const Tensor& input, float alpha, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Leaky ReLU only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in leaky_relu_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Leaky ReLU backward wrapper
auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Leaky ReLU backward only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in leaky_relu_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Softmax wrapper with temperature scaling
auto softmax_kernel(const Tensor& input, int64_t dim, hipStream_t stream, float temperature = 1.0f) -> Tensor {
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
        hipLaunchKernelGGL(softmax_forward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, input.data<float>(), result.data<float>(),
                          batch_size, dim_size, temperature);
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(softmax_forward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, input.data<double>(), result.data<double>(),
                          batch_size, dim_size, static_cast<double>(temperature));
    } else {
        throw std::runtime_error("Softmax only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in softmax_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Softmax backward wrapper
auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, hipStream_t stream) -> Tensor {
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
        hipLaunchKernelGGL(softmax_backward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, grad_output.data<float>(), output.data<float>(),
                          result.data<float>(), batch_size, dim_size);
    } else if (output.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(softmax_backward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, grad_output.data<double>(), output.data<double>(),
                          result.data<double>(), batch_size, dim_size);
    } else {
        throw std::runtime_error("Softmax backward only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in softmax_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Log Softmax wrapper
auto log_softmax_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Log Softmax only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in log_softmax_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Log Softmax backward wrapper
auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, hipStream_t stream) -> Tensor {
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
        hipLaunchKernelGGL(log_softmax_backward_kernel<float>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, grad_output.data<float>(), output.data<float>(),
                          result.data<float>(), batch_size, dim_size);
    } else if (output.dtype() == DType::Float64) {
        int num_blocks = batch_size;
        int shared_mem_size = SOFTMAX_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(log_softmax_backward_kernel<double>, dim3(num_blocks), dim3(SOFTMAX_BLOCK_SIZE),
                          shared_mem_size, stream, grad_output.data<double>(), output.data<double>(),
                          result.data<double>(), batch_size, dim_size);
    } else {
        throw std::runtime_error("Log Softmax backward only supports Float32 and Float64 dtypes");
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
auto elu_kernel(const Tensor& input, float alpha, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("ELU only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in elu_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// ELU backward wrapper
auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("ELU backward only supports Float32 and Float64 dtypes");
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
auto selu_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("SELU only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in selu_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// SELU backward wrapper
auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("SELU backward only supports Float32 and Float64 dtypes");
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
auto swish_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Swish only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in swish_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Swish backward wrapper
auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Swish backward only supports Float32 and Float64 dtypes");
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
auto mish_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Mish only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in mish_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Mish backward wrapper
auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Mish backward only supports Float32 and Float64 dtypes");
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
auto softplus_kernel(const Tensor& input, float beta, float threshold, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Softplus only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in softplus_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

// Softplus backward wrapper
auto softplus_backward_kernel(const Tensor& grad_output, const Tensor& input, float beta, float threshold, hipStream_t stream) -> Tensor {
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
    } else {
        throw std::runtime_error("Softplus backward only supports Float32 and Float64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in softplus_backward_kernel: ") + hipGetErrorString(err));
    }

    return result;
}

} // namespace rocm
} // namespace tenzor
