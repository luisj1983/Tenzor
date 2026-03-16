#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "tenzor/core/tensor.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <vector>
#include <span>

namespace tenzor {
namespace rocm {

// Helper to create zero-initialized tensor on HIP device
inline Tensor create_hip_zeros(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream = nullptr) {
    Tensor t(shape, dtype, device);
    size_t bytes = t.numel() * dtype_size(dtype);
    if (bytes > 0) {
        hipMemsetAsync(t.data_ptr(), 0, bytes, stream);
    }
    return t;
}

// Helper to convert span to vector
inline std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return std::vector<int64_t>(s.begin(), s.end());
}

// Simple reduction: sum all elements to a scalar tensor (host-side for small tensors)
inline Tensor reduce_sum_hip(const Tensor& t) {
    int64_t n = t.numel();
    Tensor result({}, t.dtype(), t.device());
    if (t.dtype() == DType::Float32) {
        std::vector<float> host(n);
        hipMemcpy(host.data(), t.data<float>(), n * sizeof(float), hipMemcpyDeviceToHost);
        float s = 0;
        for (auto v : host) s += v;
        hipMemcpy(result.data<float>(), &s, sizeof(float), hipMemcpyHostToDevice);
    } else if (t.dtype() == DType::Float64) {
        std::vector<double> host(n);
        hipMemcpy(host.data(), t.data<double>(), n * sizeof(double), hipMemcpyDeviceToHost);
        double s = 0;
        for (auto v : host) s += v;
        hipMemcpy(result.data<double>(), &s, sizeof(double), hipMemcpyHostToDevice);
    }
    return result;
}

inline Tensor reduce_mean_hip(const Tensor& t) {
    int64_t n = t.numel();
    Tensor result({}, t.dtype(), t.device());
    if (t.dtype() == DType::Float32) {
        std::vector<float> host(n);
        hipMemcpy(host.data(), t.data<float>(), n * sizeof(float), hipMemcpyDeviceToHost);
        float s = 0;
        for (auto v : host) s += v;
        s /= static_cast<float>(n);
        hipMemcpy(result.data<float>(), &s, sizeof(float), hipMemcpyHostToDevice);
    } else if (t.dtype() == DType::Float64) {
        std::vector<double> host(n);
        hipMemcpy(host.data(), t.data<double>(), n * sizeof(double), hipMemcpyDeviceToHost);
        double s = 0;
        for (auto v : host) s += v;
        s /= static_cast<double>(n);
        hipMemcpy(result.data<double>(), &s, sizeof(double), hipMemcpyHostToDevice);
    }
    return result;
}

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + hipGetErrorString(error) \
            ); \
        } \
    } while(0)

// ==============================================================================
// Fused Linear + ReLU HIP Kernel
// ==============================================================================

/**
 * @brief HIP kernel for fused linear + ReLU
 *
 * Computes: out = max(0, input @ weight.T + bias)
 * Uses grid-stride loop for large tensors.
 */
template<typename T>
__global__ void fused_linear_relu_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    bool has_bias
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * out_features;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total_elements; idx += stride) {
        int64_t b = idx / out_features;
        int64_t o = idx % out_features;

        T sum = 0;
        for (int64_t i = 0; i < in_features; ++i) {
            sum += input[b * in_features + i] * weight[o * in_features + i];
        }

        if (has_bias) {
            sum += bias[o];
        }

        // ReLU
        output[idx] = (sum > T(0)) ? sum : T(0);
    }
}

/**
 * @brief Fused linear + ReLU host function
 */
auto fused_linear_relu_hip(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    hipStream_t stream
) -> Tensor {
    // Non-Float32: upcast to Float32, compute, convert back
    if (input.dtype() != DType::Float32) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        Tensor bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &bias_f32;
        }
        auto result = fused_linear_relu_hip(input_f32, weight_f32, bias_f32_ptr, stream);
        return result.to(orig_dtype);
    }

    // Flatten input to 2D
    auto input_shape = input.shape();
    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
        batch_size *= input_shape[i];
    }
    int64_t in_features = input_shape[input_shape.size() - 1];
    int64_t out_features = weight.shape()[0];

    // Create output tensor
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
    output_shape.push_back(out_features);
    Tensor output = create_hip_zeros(output_shape, input.dtype(), input.device());

    // Launch kernel
    int64_t total_elements = batch_size * out_features;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    if (input.dtype() == DType::Float32) {
        const float* bias_ptr = bias ? bias->data<float>() : nullptr;
        hipLaunchKernelGGL(fused_linear_relu_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            weight.data<float>(),
            bias_ptr,
            output.data<float>(),
            batch_size,
            in_features,
            out_features,
            bias != nullptr
        );
    } else {
        throw std::runtime_error("fused_linear_relu_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return output;
}

// ==============================================================================
// Fused BatchNorm + ReLU HIP Kernel
// ==============================================================================

template<typename T>
__global__ void fused_batchnorm_relu_kernel(
    const T* input,
    const T* mean,
    const T* var,
    const T* gamma,
    const T* beta,
    T* output,
    int64_t batch_size,
    int64_t num_features,
    int64_t spatial_size,
    T eps
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * num_features * spatial_size;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total_elements; idx += stride) {
        int64_t s = idx % spatial_size;
        int64_t c = (idx / spatial_size) % num_features;
        int64_t n = idx / (spatial_size * num_features);

        T normalized = (input[idx] - mean[c]) * rsqrtf(var[c] + eps);
        T scaled = normalized * gamma[c] + beta[c];

        // ReLU
        output[idx] = (scaled > T(0)) ? scaled : T(0);
    }
}

auto fused_batchnorm_relu_hip(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    // Non-Float32: upcast to Float32, compute, convert back
    if (input.dtype() != DType::Float32) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto rm_f32 = running_mean.to(DType::Float32);
        auto rv_f32 = running_var.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto b_f32 = bias.to(DType::Float32);
        auto result = fused_batchnorm_relu_hip(input_f32, rm_f32, rv_f32, w_f32, b_f32, eps);
        return result.to(orig_dtype);
    }

    int64_t batch_size = input.shape()[0];
    int64_t num_features = input.shape()[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < input.shape().size(); ++i) {
        spatial_size *= input.shape()[i];
    }

    Tensor output = create_hip_zeros(to_vec(input.shape()), input.dtype(), input.device());

    int64_t total_elements = input.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fused_batchnorm_relu_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            running_mean.data<float>(),
            running_var.data<float>(),
            weight.data<float>(),
            bias.data<float>(),
            output.data<float>(),
            batch_size,
            num_features,
            spatial_size,
            eps
        );
    } else {
        throw std::runtime_error("fused_batchnorm_relu_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return output;
}

// ==============================================================================
// Fused Softmax + CrossEntropy HIP Kernel
// ==============================================================================

template<typename T, int BLOCK_SIZE>
__global__ void fused_softmax_cross_entropy_kernel(
    const T* logits,
    const int64_t* targets,
    T* losses,
    int64_t batch_size,
    int64_t num_classes
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* row = logits + b * num_classes;
    int64_t target = targets[b];

    // Shared memory for reduction
    __shared__ T shared_data[BLOCK_SIZE];

    // Find max (for numerical stability)
    T max_val = -INFINITY;
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }

    // Block-wide max reduction
    shared_data[threadIdx.x] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + s]);
        }
        __syncthreads();
    }

    T global_max = shared_data[0];
    __syncthreads();

    // Compute sum(exp(x - max))
    T sum_exp = 0;
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        sum_exp += expf(row[i] - global_max);
    }

    shared_data[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Compute loss
    if (threadIdx.x == 0) {
        T log_sum_exp = logf(shared_data[0]) + global_max;
        losses[b] = log_sum_exp - row[target];
    }
}

auto fused_softmax_cross_entropy_hip(
    const Tensor& logits,
    const Tensor& targets,
    const std::string& reduction
) -> Tensor {
    // Non-Float32: upcast to Float32, compute (loss stays Float32)
    if (logits.dtype() != DType::Float32) {
        auto logits_f32 = logits.to(DType::Float32);
        return fused_softmax_cross_entropy_hip(logits_f32, targets, reduction);
    }

    int64_t batch_size = logits.shape()[0];
    int64_t num_classes = logits.shape()[1];

    Tensor losses = create_hip_zeros({batch_size}, logits.dtype(), logits.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (logits.dtype() == DType::Float32) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_softmax_cross_entropy_kernel<float, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
            logits.data<float>(),
            targets.data<int64_t>(),
            losses.data<float>(),
            batch_size,
            num_classes
        );
    } else {
        throw std::runtime_error("fused_softmax_cross_entropy_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    // Apply reduction
    if (reduction == "mean") {
        return reduce_mean_hip(losses);
    } else if (reduction == "sum") {
        return reduce_sum_hip(losses);
    } else {
        return losses;
    }
}

// ==============================================================================
// Fused Add + ReLU HIP Kernel
// ==============================================================================

template<typename T>
__global__ void fused_add_relu_kernel(
    const T* a,
    const T* b,
    T* output,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        T sum = a[i] + b[i];
        output[i] = (sum > T(0)) ? sum : T(0);
    }
}

auto fused_add_relu_hip(const Tensor& a, const Tensor& b) -> Tensor {
    // Non-Float32: upcast to Float32, compute, convert back
    if (a.dtype() != DType::Float32) {
        DType orig_dtype = a.dtype();
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto result = fused_add_relu_hip(a_f32, b_f32);
        return result.to(orig_dtype);
    }

    Tensor result = create_hip_zeros(to_vec(a.shape()), a.dtype(), a.device());

    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fused_add_relu_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            a.data<float>(),
            b.data<float>(),
            result.data<float>(),
            n
        );
    } else {
        throw std::runtime_error("fused_add_relu_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return result;
}

// ==============================================================================
// Fused GELU HIP Kernel
// ==============================================================================

template<typename T>
__global__ void fused_gelu_kernel(
    const T* input,
    T* output,
    int64_t n
) {
    constexpr T sqrt_2_over_pi = 0.7978845608f;
    constexpr T coeff = 0.044715f;

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        T x = input[i];
        T x_cubed = x * x * x;
        T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        T tanh_val = tanhf(inner);
        output[i] = T(0.5) * x * (T(1.0) + tanh_val);
    }
}

auto fused_gelu_hip(const Tensor& input) -> Tensor {
    // Non-Float32: upcast to Float32, compute, convert back
    if (input.dtype() != DType::Float32) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto result = fused_gelu_hip(input_f32);
        return result.to(orig_dtype);
    }

    Tensor output = create_hip_zeros(to_vec(input.shape()), input.dtype(), input.device());

    int64_t n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fused_gelu_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            output.data<float>(),
            n
        );
    } else {
        throw std::runtime_error("fused_gelu_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return output;
}

// ==============================================================================
// Fused Layer Norm HIP Kernel
// ==============================================================================

template<typename T, int BLOCK_SIZE>
__global__ void fused_layer_norm_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int64_t batch_size,
    int64_t norm_size,
    T eps
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_in = input + b * norm_size;
    T* batch_out = output + b * norm_size;

    __shared__ T shared_data[BLOCK_SIZE];

    // Compute mean
    T sum = 0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        sum += batch_in[i];
    }

    shared_data[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    T mean = shared_data[0] / norm_size;
    __syncthreads();

    // Compute variance
    T var_sum = 0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T diff = batch_in[i] - mean;
        var_sum += diff * diff;
    }

    shared_data[threadIdx.x] = var_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    T variance = shared_data[0] / norm_size;
    T inv_std = rsqrtf(variance + eps);

    // Normalize and scale
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - mean) * inv_std;
        batch_out[i] = normalized * weight[i] + bias[i];
    }
}

auto fused_layer_norm_hip(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    // Non-Float32: upcast to Float32, compute, convert back
    if (input.dtype() != DType::Float32) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto b_f32 = bias.to(DType::Float32);
        auto result = fused_layer_norm_hip(input_f32, normalized_shape, w_f32, b_f32, eps);
        return result.to(orig_dtype);
    }

    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    Tensor output = create_hip_zeros(to_vec(input.shape()), input.dtype(), input.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_layer_norm_kernel<float, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
            input.data<float>(),
            weight.data<float>(),
            bias.data<float>(),
            output.data<float>(),
            batch_size,
            norm_size,
            eps
        );
    } else {
        throw std::runtime_error("fused_layer_norm_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return output;
}

// ==============================================================================
// Fused Conv + BatchNorm + ReLU HIP Kernel (Simplified)
// ==============================================================================

template<typename T>
__global__ void fused_conv_batchnorm_relu_kernel(
    const T* conv_output,
    const T* mean,
    const T* var,
    const T* gamma,
    const T* beta,
    T* output,
    int64_t batch_size,
    int64_t num_features,
    int64_t spatial_size,
    T eps
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * num_features * spatial_size;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total_elements; idx += stride) {
        int64_t s = idx % spatial_size;
        int64_t c = (idx / spatial_size) % num_features;
        int64_t n = idx / (spatial_size * num_features);

        // BatchNorm
        T normalized = (conv_output[idx] - mean[c]) * rsqrtf(var[c] + eps);
        T scaled = normalized * gamma[c] + beta[c];

        // ReLU
        output[idx] = (scaled > T(0)) ? scaled : T(0);
    }
}

auto fused_conv_batchnorm_relu_hip(
    const Tensor& conv_output,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    // Non-Float32: upcast to Float32, compute, convert back
    if (conv_output.dtype() != DType::Float32) {
        DType orig_dtype = conv_output.dtype();
        auto co_f32 = conv_output.to(DType::Float32);
        auto rm_f32 = running_mean.to(DType::Float32);
        auto rv_f32 = running_var.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto b_f32 = bias.to(DType::Float32);
        auto result = fused_conv_batchnorm_relu_hip(co_f32, rm_f32, rv_f32, w_f32, b_f32, eps);
        return result.to(orig_dtype);
    }

    int64_t batch_size = conv_output.shape()[0];
    int64_t num_features = conv_output.shape()[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < conv_output.shape().size(); ++i) {
        spatial_size *= conv_output.shape()[i];
    }

    Tensor output = create_hip_zeros(to_vec(conv_output.shape()), conv_output.dtype(), conv_output.device());

    int64_t total_elements = conv_output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    if (conv_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fused_conv_batchnorm_relu_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            conv_output.data<float>(),
            running_mean.data<float>(),
            running_var.data<float>(),
            weight.data<float>(),
            bias.data<float>(),
            output.data<float>(),
            batch_size,
            num_features,
            spatial_size,
            eps
        );
    } else {
        throw std::runtime_error("fused_conv_batchnorm_relu_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return output;
}

// ==============================================================================
// Fused MatMul + Add (Bias) HIP Kernel
// ==============================================================================

template<typename T, int TILE_SIZE = 16>
__global__ void fused_matmul_add_kernel(
    const T* A,
    const T* B,
    const T* bias,
    T* C,
    int64_t M,
    int64_t N,
    int64_t K,
    bool has_bias
) {
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    T sum = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (has_bias) {
            C[row * N + col] = sum + bias[col];
        } else {
            C[row * N + col] = sum;
        }
    }
}

auto fused_matmul_add_hip(
    const Tensor& A,
    const Tensor& B,
    const Tensor* bias
) -> Tensor {
    // Non-Float32: upcast to Float32, compute, convert back
    if (A.dtype() != DType::Float32) {
        DType orig_dtype = A.dtype();
        auto a_f32 = A.to(DType::Float32);
        auto b_f32 = B.to(DType::Float32);
        Tensor bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &bias_f32;
        }
        auto result = fused_matmul_add_hip(a_f32, b_f32, bias_f32_ptr);
        return result.to(orig_dtype);
    }

    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];

    Tensor C = create_hip_zeros({M, N}, A.dtype(), A.device());

    constexpr int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    if (A.dtype() == DType::Float32) {
        const float* bias_ptr = bias ? bias->data<float>() : nullptr;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_matmul_add_kernel<float, TILE_SIZE>),
            blocks, threads, 0, 0,
            A.data<float>(),
            B.data<float>(),
            bias_ptr,
            C.data<float>(),
            M,
            N,
            K,
            bias != nullptr
        );
    } else {
        throw std::runtime_error("fused_matmul_add_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return C;
}

// ==============================================================================
// Fused Element-wise Chain HIP Kernel
// ==============================================================================

template<typename T>
__global__ void fused_elementwise_chain_kernel(
    const T* a,
    const T* b,
    const T* c,
    T* output,
    int64_t n,
    int op_type
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        T result;
        switch (op_type) {
            case 0:  // (a + b) * c + relu
                result = (a[i] + b[i]) * c[i];
                result = (result > T(0)) ? result : T(0);
                break;
            case 1:  // (a * b) + c + relu
                result = a[i] * b[i] + c[i];
                result = (result > T(0)) ? result : T(0);
                break;
            default:
                result = a[i];
        }
        output[i] = result;
    }
}

auto fused_elementwise_chain_hip(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    int op_type
) -> Tensor {
    // Non-Float32: upcast to Float32, compute, convert back
    if (a.dtype() != DType::Float32) {
        DType orig_dtype = a.dtype();
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto c_f32 = c.to(DType::Float32);
        auto result = fused_elementwise_chain_hip(a_f32, b_f32, c_f32, op_type);
        return result.to(orig_dtype);
    }

    Tensor output = create_hip_zeros(to_vec(a.shape()), a.dtype(), a.device());

    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fused_elementwise_chain_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            a.data<float>(),
            b.data<float>(),
            c.data<float>(),
            output.data<float>(),
            n,
            op_type
        );
    } else {
        throw std::runtime_error("fused_elementwise_chain_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return output;
}

// ==============================================================================
// Fused Attention HIP Kernel
// ==============================================================================

template<typename T, int BLOCK_SIZE>
__global__ void fused_attention_kernel(
    const T* Q,
    const T* K,
    const T* V,
    T* output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t d_k,
    int64_t d_v,
    T scale
) {
    int64_t batch = blockIdx.z;
    int64_t row = blockIdx.y;

    if (batch >= batch_size || row >= seq_len) return;

    __shared__ T shared_scores[BLOCK_SIZE];
    __shared__ T shared_sum[BLOCK_SIZE];

    const T* q_row = Q + (batch * seq_len + row) * d_k;

    // Compute attention scores and find max
    T max_score = -INFINITY;
    for (int64_t col = threadIdx.x; col < seq_len; col += blockDim.x) {
        const T* k_row = K + (batch * seq_len + col) * d_k;
        T score = 0;
        for (int64_t i = 0; i < d_k; ++i) {
            score += q_row[i] * k_row[i];
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
        shared_scores[threadIdx.x] = score;
    }

    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < seq_len) {
            max_score = fmaxf(max_score, shared_scores[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Compute softmax
    T sum_exp = 0;
    for (int64_t col = threadIdx.x; col < seq_len; col += blockDim.x) {
        const T* k_row = K + (batch * seq_len + col) * d_k;
        T score = 0;
        for (int64_t i = 0; i < d_k; ++i) {
            score += q_row[i] * k_row[i];
        }
        score = expf(score * scale - max_score);
        shared_scores[threadIdx.x] = score;
        sum_exp += score;
    }

    shared_sum[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    T sum_total = shared_sum[0];

    // Compute attention @ V
    for (int64_t d = threadIdx.x; d < d_v; d += blockDim.x) {
        T result = 0;
        for (int64_t col = 0; col < seq_len; ++col) {
            const T* k_row = K + (batch * seq_len + col) * d_k;
            T score = 0;
            for (int64_t i = 0; i < d_k; ++i) {
                score += q_row[i] * k_row[i];
            }
            T attention_weight = expf(score * scale - max_score) / sum_total;

            const T* v_row = V + (batch * seq_len + col) * d_v;
            result += attention_weight * v_row[d];
        }
        output[(batch * seq_len + row) * d_v + d] = result;
    }
}

/**
 * @brief Compute row-wise logsumexp of attention scores for backward pass
 *
 * LSE_i = log(sum_j exp(Q_i @ K_j^T * scale - max_j)) + max_j
 * Grid: (batch_size, seq_len), Block: (BLOCK_SIZE)
 */
template<typename T, int BLOCK_SIZE>
__global__ void compute_attention_lse_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    T* __restrict__ lse_out,
    int64_t batch_size,
    int64_t seq_len,
    int64_t d_k,
    T scale
) {
    int64_t batch = blockIdx.x;
    int64_t row = blockIdx.y;
    if (batch >= batch_size || row >= seq_len) return;

    extern __shared__ char shared_bytes[];
    T* shared_reduce = reinterpret_cast<T*>(shared_bytes);

    const T* q_row = Q + (batch * seq_len + row) * d_k;
    const T* k_base = K + batch * seq_len * d_k;

    // Find max score
    T thread_max = -INFINITY;
    for (int64_t col = threadIdx.x; col < seq_len; col += blockDim.x) {
        const T* k_row = k_base + col * d_k;
        T score = 0;
        for (int64_t i = 0; i < d_k; ++i) {
            score += q_row[i] * k_row[i];
        }
        score *= scale;
        thread_max = fmaxf(thread_max, score);
    }

    shared_reduce[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_reduce[threadIdx.x] = fmaxf(shared_reduce[threadIdx.x],
                                                 shared_reduce[threadIdx.x + s]);
        }
        __syncthreads();
    }
    T max_val = shared_reduce[0];

    // Compute sum of exp(score - max)
    T thread_sum = 0;
    for (int64_t col = threadIdx.x; col < seq_len; col += blockDim.x) {
        const T* k_row = k_base + col * d_k;
        T score = 0;
        for (int64_t i = 0; i < d_k; ++i) {
            score += q_row[i] * k_row[i];
        }
        score *= scale;
        thread_sum += expf(score - max_val);
    }

    shared_reduce[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_reduce[threadIdx.x] += shared_reduce[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        lse_out[batch * seq_len + row] = max_val + logf(fmaxf(shared_reduce[0], 1e-10f));
    }
}

// ==============================================================================
// Flash Attention Backward HIP Kernel (Tiled, Memory-Efficient)
// ==============================================================================

/**
 * @brief Tiled Flash Attention backward kernel (ported from CUDA)
 *
 * Recomputes attention scores in tiles using saved logsumexp from the forward pass,
 * avoiding materialization of the full NxN attention matrix.
 *
 * Each thread block processes one KV tile (column block of size Bc) across all Q tiles.
 * dK and dV are accumulated directly in registers (one block per KV tile, no race).
 * dQ is accumulated via atomicAdd since multiple KV tiles contribute to each Q row.
 *
 * Grid: (num_kv_tiles, batch_heads)
 * Block: (BLOCK_SIZE) threads
 *
 * Shared memory layout (fits in 48KB for HEAD_DIM <= 128):
 *   K_tile[Bc][HEAD_DIM], V_tile[Bc][HEAD_DIM],
 *   Q_tile[Br][HEAD_DIM], dO_tile[Br][HEAD_DIM],
 *   S_tile[Br][Bc], l_tile[Br], D_tile[Br]
 */
template<int HEAD_DIM, int Br, int Bc, int BLOCK_SIZE>
__global__ void flash_attention_backward_kernel_hip(
    const float* __restrict__ Q,     // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ K,     // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ V,     // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ O,     // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ dO,    // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ L,     // [batch_heads, seq_len] logsumexp
    float* __restrict__ dQ,          // [batch_heads, seq_len, HEAD_DIM] (atomicAdd)
    float* __restrict__ dK,          // [batch_heads, seq_len, HEAD_DIM] (one block per KV tile)
    float* __restrict__ dV,          // [batch_heads, seq_len, HEAD_DIM] (one block per KV tile)
    const int seq_len,
    const float scale,
    const bool causal
) {
    const int kv_tile_idx = blockIdx.x;  // which KV tile (column block)
    const int batch_head = blockIdx.y;
    const int tid = threadIdx.x;

    const int kv_start = kv_tile_idx * Bc;
    if (kv_start >= seq_len) return;
    const int actual_Bc = min(Bc, seq_len - kv_start);

    // Base pointers for this batch-head
    const float* Q_base  = Q  + batch_head * seq_len * HEAD_DIM;
    const float* K_base  = K  + batch_head * seq_len * HEAD_DIM;
    const float* V_base  = V  + batch_head * seq_len * HEAD_DIM;
    const float* O_base  = O  + batch_head * seq_len * HEAD_DIM;
    const float* dO_base = dO + batch_head * seq_len * HEAD_DIM;
    const float* L_base  = L  + batch_head * seq_len;
    float* dQ_base = dQ + batch_head * seq_len * HEAD_DIM;
    float* dK_base = dK + batch_head * seq_len * HEAD_DIM;
    float* dV_base = dV + batch_head * seq_len * HEAD_DIM;

    // Shared memory layout (no dK/dV tiles - those go directly to global)
    extern __shared__ float smem[];
    float* K_tile  = smem;                                          // [Bc][HEAD_DIM]
    float* V_tile  = K_tile  + Bc * HEAD_DIM;                      // [Bc][HEAD_DIM]
    float* Q_tile  = V_tile  + Bc * HEAD_DIM;                      // [Br][HEAD_DIM]
    float* dO_tile = Q_tile  + Br * HEAD_DIM;                      // [Br][HEAD_DIM]
    float* S_tile  = dO_tile + Br * HEAD_DIM;                      // [Br][Bc]
    float* l_tile  = S_tile  + Br * Bc;                             // [Br]
    float* D_tile  = l_tile  + Br;                                  // [Br]

    // Load K_j and V_j tiles into shared memory
    for (int i = tid; i < actual_Bc * HEAD_DIM; i += BLOCK_SIZE) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        K_tile[row * HEAD_DIM + col] = K_base[(kv_start + row) * HEAD_DIM + col];
        V_tile[row * HEAD_DIM + col] = V_base[(kv_start + row) * HEAD_DIM + col];
    }
    // Zero-pad if actual_Bc < Bc
    for (int i = tid + actual_Bc * HEAD_DIM; i < Bc * HEAD_DIM; i += BLOCK_SIZE) {
        K_tile[i] = 0.0f;
        V_tile[i] = 0.0f;
    }
    __syncthreads();

    // Per-thread accumulators for dK and dV
    // Max elements per thread: ceil(Bc * HEAD_DIM / BLOCK_SIZE)
    constexpr int MAX_ELEMS_PER_THREAD = (Bc * HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float dk_acc[MAX_ELEMS_PER_THREAD];
    float dv_acc[MAX_ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
        dk_acc[e] = 0.0f;
        dv_acc[e] = 0.0f;
    }

    // Iterate over Q tiles (row blocks)
    const int num_q_tiles = (seq_len + Br - 1) / Br;

    for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
        const int q_start = q_tile_idx * Br;
        if (q_start >= seq_len) break;
        const int actual_Br = min(Br, seq_len - q_start);

        // For causal masking: skip if all Q rows come before all K cols
        if (causal && (q_start + actual_Br - 1) < kv_start) {
            continue;
        }

        // Load Q_i and dO_i tiles into shared memory
        for (int i = tid; i < actual_Br * HEAD_DIM; i += BLOCK_SIZE) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            Q_tile[row * HEAD_DIM + col]  = Q_base[(q_start + row) * HEAD_DIM + col];
            dO_tile[row * HEAD_DIM + col] = dO_base[(q_start + row) * HEAD_DIM + col];
        }
        // Zero-pad
        for (int i = tid + actual_Br * HEAD_DIM; i < Br * HEAD_DIM; i += BLOCK_SIZE) {
            Q_tile[i] = 0.0f;
            dO_tile[i] = 0.0f;
        }

        // Load l_i (logsumexp) and compute D_i = rowsum(dO_i * O_i)
        for (int row = tid; row < actual_Br; row += BLOCK_SIZE) {
            l_tile[row] = L_base[q_start + row];

            float d_sum = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                d_sum += dO_base[(q_start + row) * HEAD_DIM + d]
                       * O_base[(q_start + row) * HEAD_DIM + d];
            }
            D_tile[row] = d_sum;
        }
        for (int row = tid + actual_Br; row < Br; row += BLOCK_SIZE) {
            l_tile[row] = -INFINITY;
            D_tile[row] = 0.0f;
        }
        __syncthreads();

        // Compute S_ij = Q_i @ K_j^T * scale  [Br x Bc]
        for (int idx = tid; idx < actual_Br * actual_Bc; idx += BLOCK_SIZE) {
            int i = idx / actual_Bc;
            int j = idx % actual_Bc;
            float dot = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += Q_tile[i * HEAD_DIM + d] * K_tile[j * HEAD_DIM + d];
            }
            S_tile[i * Bc + j] = dot * scale;
        }
        // Set out-of-bounds entries to -inf
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            if (i >= actual_Br || j >= actual_Bc) {
                S_tile[idx] = -INFINITY;
            }
        }
        __syncthreads();

        // Compute P_ij = exp(S_ij - l_i)  [Br x Bc]
        // Apply causal mask: P_ij = 0 where query_pos < key_pos
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            float p = 0.0f;
            if (i < actual_Br && j < actual_Bc) {
                if (causal && (q_start + i) < (kv_start + j)) {
                    p = 0.0f;
                } else {
                    p = expf(S_tile[i * Bc + j] - l_tile[i]);
                }
            }
            S_tile[i * Bc + j] = p;  // Reuse S_tile for P_ij
        }
        __syncthreads();

        // Accumulate dV_j += P_ij^T @ dO_i  [Bc x HEAD_DIM]
        {
            int e = 0;
            for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE, ++e) {
                int j = idx / HEAD_DIM;
                int d = idx % HEAD_DIM;
                if (j < actual_Bc) {
                    float sum = 0.0f;
                    for (int i = 0; i < actual_Br; ++i) {
                        sum += S_tile[i * Bc + j] * dO_tile[i * HEAD_DIM + d];
                    }
                    dv_acc[e] += sum;
                }
            }
        }
        __syncthreads();

        // Compute dS_ij = P_ij * (dP_ij - D_i)  where dP_ij = dO_i . V_j
        // Overwrites S_tile (P_ij) with dS_ij
        for (int idx = tid; idx < actual_Br * actual_Bc; idx += BLOCK_SIZE) {
            int i = idx / actual_Bc;
            int j = idx % actual_Bc;
            float dp = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; ++d) {
                dp += dO_tile[i * HEAD_DIM + d] * V_tile[j * HEAD_DIM + d];
            }
            float p_ij = S_tile[i * Bc + j];
            S_tile[i * Bc + j] = p_ij * (dp - D_tile[i]);
        }
        __syncthreads();

        // Accumulate dK_j += dS_ij^T @ Q_i * scale  [Bc x HEAD_DIM]
        {
            int e = 0;
            for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE, ++e) {
                int j = idx / HEAD_DIM;
                int d = idx % HEAD_DIM;
                if (j < actual_Bc) {
                    float sum = 0.0f;
                    for (int i = 0; i < actual_Br; ++i) {
                        sum += S_tile[i * Bc + j] * Q_tile[i * HEAD_DIM + d];
                    }
                    dk_acc[e] += sum * scale;
                }
            }
        }

        // dQ_i += dS_ij @ K_j * scale  [Br x HEAD_DIM]
        // Accumulate via atomicAdd since multiple KV tiles contribute
        for (int idx = tid; idx < actual_Br * HEAD_DIM; idx += BLOCK_SIZE) {
            int i = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            float sum = 0.0f;
            for (int j = 0; j < actual_Bc; ++j) {
                sum += S_tile[i * Bc + j] * K_tile[j * HEAD_DIM + d];
            }
            atomicAdd(&dQ_base[(q_start + i) * HEAD_DIM + d], sum * scale);
        }
        __syncthreads();
    }

    // Write accumulated dK and dV from registers to global memory
    {
        int e = 0;
        for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE, ++e) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            if (row < actual_Bc) {
                dK_base[(kv_start + row) * HEAD_DIM + col] = dk_acc[e];
                dV_base[(kv_start + row) * HEAD_DIM + col] = dv_acc[e];
            }
        }
    }
}

// Host wrapper for fused flash attention backward
auto flash_attention_backward_hip(
    const Tensor& dO,    // [batch_heads, seq_len, head_dim]
    const Tensor& Q,     // [batch_heads, seq_len, head_dim]
    const Tensor& K,     // [batch_heads, seq_len, head_dim]
    const Tensor& V,     // [batch_heads, seq_len, head_dim]
    const Tensor& O,     // [batch_heads, seq_len, head_dim]
    const Tensor& L,     // [batch_heads, seq_len] logsumexp
    float scale,
    bool causal
) -> std::vector<Tensor> {
    const auto dtype = Q.dtype();

    // Float16: upcast to Float32, compute, convert back
    if (dtype == DType::Float16) {
        auto dO_f32 = dO.to(DType::Float32);
        auto Q_f32  = Q.to(DType::Float32);
        auto K_f32  = K.to(DType::Float32);
        auto V_f32  = V.to(DType::Float32);
        auto O_f32  = O.to(DType::Float32);
        // L is already Float32 from the forward pass
        auto [dQ, dK, dV] = [&]() {
            auto result = flash_attention_backward_hip(dO_f32, Q_f32, K_f32, V_f32, O_f32, L, scale, causal);
            return std::make_tuple(std::move(result[0]), std::move(result[1]), std::move(result[2]));
        }();
        return {dQ.to(DType::Float16), dK.to(DType::Float16), dV.to(DType::Float16)};
    }

    // BFloat16: upcast to Float32, compute, convert back
    if (dtype == DType::BFloat16) {
        auto dO_f32 = dO.to(DType::Float32);
        auto Q_f32  = Q.to(DType::Float32);
        auto K_f32  = K.to(DType::Float32);
        auto V_f32  = V.to(DType::Float32);
        auto O_f32  = O.to(DType::Float32);
        auto [dQ, dK, dV] = [&]() {
            auto result = flash_attention_backward_hip(dO_f32, Q_f32, K_f32, V_f32, O_f32, L, scale, causal);
            return std::make_tuple(std::move(result[0]), std::move(result[1]), std::move(result[2]));
        }();
        return {dQ.to(DType::BFloat16), dK.to(DType::BFloat16), dV.to(DType::BFloat16)};
    }

    int64_t batch_heads = Q.shape()[0];
    int64_t seq_len = Q.shape()[1];
    int64_t head_dim = Q.shape()[2];

    if (dtype != DType::Float32) {
        throw std::runtime_error(
            "flash_attention_backward_hip: Unsupported dtype. "
            "Supported: Float32, Float16, BFloat16");
    }

    Tensor dQ = create_hip_zeros({batch_heads, seq_len, head_dim}, Q.dtype(), Q.device());
    Tensor dK = create_hip_zeros({batch_heads, seq_len, head_dim}, K.dtype(), K.device());
    Tensor dV = create_hip_zeros({batch_heads, seq_len, head_dim}, V.dtype(), V.device());

    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int BLOCK_SIZE = 256;

    int num_kv_tiles = (seq_len + Bc - 1) / Bc;
    dim3 grid(num_kv_tiles, batch_heads);
    dim3 threads(BLOCK_SIZE);

    // Shared memory: K_tile[Bc*HD] + V_tile[Bc*HD] + Q_tile[Br*HD] + dO_tile[Br*HD]
    //              + S_tile[Br*Bc] + l_tile[Br] + D_tile[Br]
    auto compute_bwd_smem = [&](int hd) -> size_t {
        return (2 * Bc * hd + 2 * Br * hd + Br * Bc + Br + Br) * sizeof(float);
    };

    const float* q_ptr  = Q.data<float>();
    const float* k_ptr  = K.data<float>();
    const float* v_ptr  = V.data<float>();
    const float* o_ptr  = O.data<float>();
    const float* do_ptr = dO.data<float>();
    const float* l_ptr  = L.data<float>();
    float* dq_ptr = dQ.data<float>();
    float* dk_ptr = dK.data<float>();
    float* dv_ptr = dV.data<float>();
    int seq_len_int = static_cast<int>(seq_len);

    // Dispatch based on head_dim for optimal unrolling
    if (head_dim == 32) {
        size_t smem = compute_bwd_smem(32);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(flash_attention_backward_kernel_hip<32, Br, Bc, BLOCK_SIZE>),
            grid, threads, smem, 0,
            q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_ptr, dk_ptr, dv_ptr,
            seq_len_int, scale, causal);
        HIP_CHECK(hipGetLastError());
    } else if (head_dim == 64) {
        size_t smem = compute_bwd_smem(64);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(flash_attention_backward_kernel_hip<64, Br, Bc, BLOCK_SIZE>),
            grid, threads, smem, 0,
            q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_ptr, dk_ptr, dv_ptr,
            seq_len_int, scale, causal);
        HIP_CHECK(hipGetLastError());
    } else if (head_dim == 128) {
        size_t smem = compute_bwd_smem(128);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(flash_attention_backward_kernel_hip<128, Br, Bc, BLOCK_SIZE>),
            grid, threads, smem, 0,
            q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_ptr, dk_ptr, dv_ptr,
            seq_len_int, scale, causal);
        HIP_CHECK(hipGetLastError());
    } else {
        throw std::runtime_error(
            "flash_attention_backward_hip: Unsupported head_dim " + std::to_string(head_dim) +
            ". Fused backward supports 32, 64, 128.");
    }

    HIP_CHECK(hipGetLastError());

    return {dQ, dK, dV};
}

auto fused_attention_hip(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    float scale
) -> std::pair<Tensor, Tensor> {
    // Non-Float32: upcast to Float32, compute, convert back
    if (Q.dtype() != DType::Float32) {
        DType orig_dtype = Q.dtype();
        auto q_f32 = Q.to(DType::Float32);
        auto k_f32 = K.to(DType::Float32);
        auto v_f32 = V.to(DType::Float32);
        auto [result, lse] = fused_attention_hip(q_f32, k_f32, v_f32, scale);
        return {result.to(orig_dtype), lse};
    }

    int64_t batch_size = Q.shape()[0];
    int64_t seq_len = Q.shape()[1];
    int64_t d_k = Q.shape()[2];
    int64_t d_v = V.shape()[2];

    Tensor output = create_hip_zeros({batch_size, seq_len, d_v}, Q.dtype(), Q.device());
    Tensor lse = create_hip_zeros({batch_size, seq_len}, Q.dtype(), Q.device());

    constexpr int BLOCK_SIZE = 256;
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(1, seq_len, batch_size);

    if (Q.dtype() == DType::Float32) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_attention_kernel<float, BLOCK_SIZE>),
            blocks, threads, 0, 0,
            Q.data<float>(),
            K.data<float>(),
            V.data<float>(),
            output.data<float>(),
            batch_size,
            seq_len,
            d_k,
            d_v,
            scale
        );

        // Compute logsumexp separately for backward pass compatibility
        // The naive attention kernel doesn't produce it inline, so we compute
        // LSE = log(sum(exp(scores - max))) + max per query row
        // For now, use a simple kernel to compute it from Q and K
        // (this is acceptable overhead since the naive kernel is already O(N^2))
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(compute_attention_lse_kernel<float, BLOCK_SIZE>),
            dim3(batch_size, seq_len), dim3(BLOCK_SIZE), BLOCK_SIZE * sizeof(float), 0,
            Q.data<float>(),
            K.data<float>(),
            lse.data<float>(),
            batch_size,
            seq_len,
            d_k,
            scale
        );
    } else {
        throw std::runtime_error("fused_attention_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return {output, lse};
}

// ==============================================================================
// Fused RMSNorm HIP Kernel
// ==============================================================================

template<typename T, int BLOCK_SIZE>
__global__ void fused_rms_norm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    T* __restrict__ rrms_out,
    int64_t batch_size,
    int64_t norm_size,
    T eps
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_in = input + b * norm_size;
    T* batch_out = output + b * norm_size;

    __shared__ T shared_data[BLOCK_SIZE];

    // Compute sum of squares
    T sum_sq = 0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T val = batch_in[i];
        sum_sq += val * val;
    }

    shared_data[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Compute reciprocal RMS
    __shared__ T shared_rrms;
    if (threadIdx.x == 0) {
        T mean_sq = shared_data[0] / norm_size;
        shared_rrms = rsqrtf(mean_sq + eps);
        rrms_out[b] = shared_rrms;
    }
    __syncthreads();
    T rrms = shared_rrms;

    // Apply normalization
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        batch_out[i] = batch_in[i] * rrms * weight[i];
    }
}

auto fused_rms_norm_hip(
    const Tensor& input,
    const Tensor& weight,
    float eps
) -> std::pair<Tensor, Tensor> {
    // Non-Float32: upcast to Float32, compute, convert back
    if (input.dtype() != DType::Float32) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto [result, rrms] = fused_rms_norm_hip(input_f32, weight_f32, eps);
        return {result.to(orig_dtype), rrms};
    }

    auto shape = input.shape();
    int64_t norm_size = shape[shape.size() - 1];

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }

    Tensor output = create_hip_zeros(to_vec(input.shape()), input.dtype(), input.device());
    Tensor rrms = create_hip_zeros({batch_size}, input.dtype(), input.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_rms_norm_kernel<float, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
            input.data<float>(),
            weight.data<float>(),
            output.data<float>(),
            rrms.data<float>(),
            batch_size,
            norm_size,
            eps
        );
    } else {
        throw std::runtime_error("fused_rms_norm_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return {output, rrms};
}

// ==============================================================================
// Fused Conv2D + BatchNorm + ReLU HIP Kernel (Full: conv + BN + ReLU)
// ==============================================================================

template<typename T>
__global__ void fused_conv2d_bn_relu_full_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    const T* bn_mean,
    const T* bn_var,
    const T* bn_gamma,
    const T* bn_beta,
    T* output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    T eps,
    bool has_bias
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * out_channels * out_h * out_w;
    int64_t stride_loop = blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total_elements; idx += stride_loop) {
        int64_t w_out = idx % out_w;
        int64_t h_out = (idx / out_w) % out_h;
        int64_t c_out = (idx / (out_w * out_h)) % out_channels;
        int64_t n = idx / (out_w * out_h * out_channels);

        // Compute convolution
        T conv_sum = 0;
        for (int64_t c_in = 0; c_in < in_channels; ++c_in) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh;
                    int64_t w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        int64_t input_idx = ((n * in_channels + c_in) * in_h + h_in) * in_w + w_in;
                        int64_t weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                        conv_sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        if (has_bias) {
            conv_sum += bias[c_out];
        }

        // Apply batch normalization
        T normalized = (conv_sum - bn_mean[c_out]) * rsqrtf(bn_var[c_out] + eps);
        T bn_out = normalized * bn_gamma[c_out] + bn_beta[c_out];

        // Apply ReLU
        output[idx] = (bn_out > T(0)) ? bn_out : T(0);
    }
}

auto fused_conv2d_bn_relu_full_hip(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    const Tensor& bn_mean,
    const Tensor& bn_var,
    const Tensor& bn_gamma,
    const Tensor& bn_beta,
    int64_t stride,
    int64_t padding,
    float eps
) -> Tensor {
    // Non-Float32: upcast to Float32, compute, convert back
    if (input.dtype() != DType::Float32) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        Tensor bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &bias_f32;
        }
        auto bn_mean_f32 = bn_mean.to(DType::Float32);
        auto bn_var_f32 = bn_var.to(DType::Float32);
        auto bn_gamma_f32 = bn_gamma.to(DType::Float32);
        auto bn_beta_f32 = bn_beta.to(DType::Float32);
        auto result = fused_conv2d_bn_relu_full_hip(input_f32, weight_f32, bias_f32_ptr,
                                                     bn_mean_f32, bn_var_f32, bn_gamma_f32,
                                                     bn_beta_f32, stride, padding, eps);
        return result.to(orig_dtype);
    }

    int64_t batch_size = input.shape()[0];
    int64_t in_channels = input.shape()[1];
    int64_t in_h = input.shape()[2];
    int64_t in_w = input.shape()[3];

    int64_t out_channels = weight.shape()[0];
    int64_t kernel_h = weight.shape()[2];
    int64_t kernel_w = weight.shape()[3];

    int64_t out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    int64_t out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    Tensor output = create_hip_zeros({batch_size, out_channels, out_h, out_w}, input.dtype(), input.device());

    int64_t total_elements = batch_size * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    if (input.dtype() == DType::Float32) {
        const float* bias_ptr = bias ? bias->data<float>() : nullptr;
        hipLaunchKernelGGL(fused_conv2d_bn_relu_full_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            weight.data<float>(),
            bias_ptr,
            bn_mean.data<float>(),
            bn_var.data<float>(),
            bn_gamma.data<float>(),
            bn_beta.data<float>(),
            output.data<float>(),
            batch_size,
            in_channels,
            out_channels,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
            eps,
            bias != nullptr
        );
    } else {
        throw std::runtime_error("fused_conv2d_bn_relu_full_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());

    return output;
}

// ==============================================================================
// Fused SGD with Momentum HIP Kernel
// ==============================================================================

template<typename T>
__global__ void fused_sgd_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ momentum_buffer,
    int64_t numel,
    float lr,
    float momentum,
    float weight_decay,
    float dampening,
    bool nesterov,
    bool has_momentum_buffer
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    T g = grad[idx];
    T p = param[idx];

    // Apply weight decay
    if (weight_decay > 0.0f) {
        g = g + T(weight_decay) * p;
    }

    if (has_momentum_buffer && momentum > 0.0f) {
        T v = momentum_buffer[idx];

        // Update momentum buffer
        v = T(momentum) * v + T(1.0f - dampening) * g;
        momentum_buffer[idx] = v;

        if (nesterov) {
            g = g + T(momentum) * v;
        } else {
            g = v;
        }
    }

    // Update parameter
    param[idx] = p - T(lr) * g;
}

auto fused_sgd_step_hip(
    Tensor& param,
    const Tensor& grad,
    Tensor* momentum_buffer,
    float lr,
    float momentum,
    float weight_decay,
    float dampening,
    bool nesterov,
    hipStream_t stream
) -> void {
    // BFloat16: upcast to Float32, compute, convert back
    if (param.dtype() == DType::BFloat16) {
        auto param_f32 = param.to(DType::Float32);
        auto grad_f32 = grad.to(DType::Float32);
        Tensor mom_f32;
        Tensor* mom_f32_ptr = nullptr;
        if (momentum_buffer) {
            mom_f32 = momentum_buffer->to(DType::Float32);
            mom_f32_ptr = &mom_f32;
        }
        fused_sgd_step_hip(param_f32, grad_f32, mom_f32_ptr, lr, momentum,
                           weight_decay, dampening, nesterov, stream);
        param = param_f32.to(DType::BFloat16);
        if (momentum_buffer) *momentum_buffer = mom_f32.to(DType::BFloat16);
        return;
    }

    int64_t numel = param.numel();
    constexpr int BLOCK_SIZE = 256;
    int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = std::min(blocks, 65535);

    if (param.dtype() == DType::Float32) {
        float* momentum_ptr = momentum_buffer ? momentum_buffer->data<float>() : nullptr;

        hipLaunchKernelGGL(fused_sgd_kernel<float>,
            dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
            param.data<float>(),
            grad.data<float>(),
            momentum_ptr,
            numel, lr, momentum, weight_decay, dampening,
            nesterov, momentum_buffer != nullptr
        );
    } else if (param.dtype() == DType::Float64) {
        double* momentum_ptr = momentum_buffer ? momentum_buffer->data<double>() : nullptr;

        hipLaunchKernelGGL(fused_sgd_kernel<double>,
            dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
            param.data<double>(),
            grad.data<double>(),
            momentum_ptr,
            numel, lr, momentum, weight_decay, dampening,
            nesterov, momentum_buffer != nullptr
        );
    } else {
        throw std::runtime_error("fused_sgd_step_hip: Only Float32 and Float64 supported");
    }

    HIP_CHECK(hipGetLastError());
}

// ==============================================================================
// Fused Adam Optimizer HIP Kernel
// ==============================================================================

template<typename T>
__global__ void fused_adam_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ exp_avg,
    T* __restrict__ exp_avg_sq,
    T* __restrict__ max_exp_avg_sq,
    int64_t numel,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double bias_correction1,
    double bias_correction2,
    bool amsgrad,
    bool decoupled_weight_decay
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    T g = grad[idx];
    T p = param[idx];
    T m = exp_avg[idx];
    T v = exp_avg_sq[idx];

    double step_size = lr / bias_correction1;
    double bc2_inv = 1.0 / bias_correction2;

    // L2 regularization (added to grad)
    if (weight_decay > 0.0 && !decoupled_weight_decay) {
        g = g + T(weight_decay) * p;
    }

    // Update biased first moment estimate
    m = T(beta1) * m + T(1.0 - beta1) * g;

    // Update biased second raw moment estimate
    v = T(beta2) * v + T(1.0 - beta2) * g * g;

    // Bias-corrected second moment
    T v_hat = v * T(bc2_inv);

    if (amsgrad && max_exp_avg_sq) {
        T max_v = max_exp_avg_sq[idx];
        if (v_hat > max_v) max_v = v_hat;
        max_exp_avg_sq[idx] = max_v;
        v_hat = max_v;
    }

    // Decoupled weight decay (AdamW)
    if (weight_decay > 0.0 && decoupled_weight_decay) {
        p = p * T(1.0 - lr * weight_decay);
    }

    // Update parameter
    p = p - T(step_size) * m / (sqrt(v_hat) + T(eps));

    // Store
    param[idx] = p;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
}

auto fused_adam_step_hip(
    Tensor& param,
    const Tensor& grad,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step,
    bool decoupled_weight_decay,
    hipStream_t stream,
    Tensor* max_exp_avg_sq,
    bool amsgrad
) -> void {
    // BFloat16: upcast to Float32, compute, convert back
    if (param.dtype() == DType::BFloat16) {
        auto param_f32 = param.to(DType::Float32);
        auto grad_f32 = grad.to(DType::Float32);
        auto ea_f32 = exp_avg.to(DType::Float32);
        auto eas_f32 = exp_avg_sq.to(DType::Float32);
        Tensor meas_f32;
        Tensor* meas_f32_ptr = nullptr;
        if (max_exp_avg_sq) {
            meas_f32 = max_exp_avg_sq->to(DType::Float32);
            meas_f32_ptr = &meas_f32;
        }
        fused_adam_step_hip(param_f32, grad_f32, ea_f32, eas_f32, lr, beta1, beta2, eps,
                            weight_decay, step, decoupled_weight_decay, stream, meas_f32_ptr, amsgrad);
        param = param_f32.to(DType::BFloat16);
        exp_avg = ea_f32.to(DType::BFloat16);
        exp_avg_sq = eas_f32.to(DType::BFloat16);
        if (max_exp_avg_sq) *max_exp_avg_sq = meas_f32.to(DType::BFloat16);
        return;
    }

    int64_t numel = param.numel();
    constexpr int BLOCK_SIZE = 256;
    int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = std::min(blocks, 65535);

    double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));

    if (param.dtype() == DType::Float32) {
        float* max_sq_ptr = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<float>() : nullptr;

        hipLaunchKernelGGL(fused_adam_kernel<float>,
            dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
            param.data<float>(),
            grad.data<float>(),
            exp_avg.data<float>(),
            exp_avg_sq.data<float>(),
            max_sq_ptr,
            numel, lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2,
            amsgrad, decoupled_weight_decay
        );
    } else if (param.dtype() == DType::Float64) {
        double* max_sq_ptr = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<double>() : nullptr;

        hipLaunchKernelGGL(fused_adam_kernel<double>,
            dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
            param.data<double>(),
            grad.data<double>(),
            exp_avg.data<double>(),
            exp_avg_sq.data<double>(),
            max_sq_ptr,
            numel, lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2,
            amsgrad, decoupled_weight_decay
        );
    } else {
        throw std::runtime_error("fused_adam_step_hip: Only Float32 and Float64 supported");
    }

    HIP_CHECK(hipGetLastError());
}

// ==============================================================================
// Fused RMSProp Optimizer HIP Kernel
// ==============================================================================

template<typename T>
__global__ void fused_rmsprop_step_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ square_avg,
    T* __restrict__ grad_avg,
    T* __restrict__ momentum_buffer,
    float lr, float alpha, float eps,
    float weight_decay, float momentum,
    bool centered,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T g = grad[idx];

    if (weight_decay != 0.0f) {
        g = g + T(weight_decay) * param[idx];
    }

    T sq = square_avg[idx];
    sq = T(alpha) * sq + T(1.0f - alpha) * g * g;
    square_avg[idx] = sq;

    T avg;
    if (centered && grad_avg) {
        T ga = grad_avg[idx];
        ga = T(alpha) * ga + T(1.0f - alpha) * g;
        grad_avg[idx] = ga;
        avg = sqrt(sq - ga * ga + T(eps));
    } else {
        avg = sqrt(sq + T(eps));
    }

    if (momentum > 0.0f && momentum_buffer) {
        T buf = momentum_buffer[idx];
        buf = T(momentum) * buf + g / avg;
        momentum_buffer[idx] = buf;
        param[idx] = param[idx] - T(lr) * buf;
    } else {
        param[idx] = param[idx] - T(lr) * g / avg;
    }
}

auto fused_rmsprop_step_hip(
    Tensor& param,
    const Tensor& grad,
    Tensor& square_avg,
    Tensor* grad_avg,
    Tensor* momentum_buffer,
    float lr, float alpha, float eps,
    float weight_decay, float momentum,
    bool centered,
    hipStream_t stream
) -> void {
    // BFloat16: upcast to Float32, compute, convert back
    if (param.dtype() == DType::BFloat16) {
        auto param_f32 = param.to(DType::Float32);
        auto grad_f32 = grad.to(DType::Float32);
        auto sa_f32 = square_avg.to(DType::Float32);
        Tensor ga_f32, mb_f32;
        Tensor* ga_f32_ptr = nullptr;
        Tensor* mb_f32_ptr = nullptr;
        if (grad_avg) { ga_f32 = grad_avg->to(DType::Float32); ga_f32_ptr = &ga_f32; }
        if (momentum_buffer) { mb_f32 = momentum_buffer->to(DType::Float32); mb_f32_ptr = &mb_f32; }
        fused_rmsprop_step_hip(param_f32, grad_f32, sa_f32, ga_f32_ptr, mb_f32_ptr,
                               lr, alpha, eps, weight_decay, momentum, centered, stream);
        param = param_f32.to(DType::BFloat16);
        square_avg = sa_f32.to(DType::BFloat16);
        if (grad_avg) *grad_avg = ga_f32.to(DType::BFloat16);
        if (momentum_buffer) *momentum_buffer = mb_f32.to(DType::BFloat16);
        return;
    }

    int64_t n = param.numel();
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    if (param.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fused_rmsprop_step_kernel<float>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            param.data<float>(), grad.data<float>(), square_avg.data<float>(),
            (centered && grad_avg) ? grad_avg->data<float>() : nullptr,
            (momentum > 0.0f && momentum_buffer) ? momentum_buffer->data<float>() : nullptr,
            lr, alpha, eps, weight_decay, momentum, centered, n);
    } else if (param.dtype() == DType::Float64) {
        hipLaunchKernelGGL(fused_rmsprop_step_kernel<double>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            param.data<double>(), grad.data<double>(), square_avg.data<double>(),
            (centered && grad_avg) ? grad_avg->data<double>() : nullptr,
            (momentum > 0.0f && momentum_buffer) ? momentum_buffer->data<double>() : nullptr,
            lr, alpha, eps, weight_decay, momentum, centered, n);
    } else {
        throw std::runtime_error("fused_rmsprop_step_hip: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
}

// ==============================================================================
// Fused Adadelta Optimizer HIP Kernel
// ==============================================================================

template<typename T>
__global__ void fused_adadelta_step_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ square_avg,
    T* __restrict__ acc_delta,
    float rho, float eps, float lr, float weight_decay,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T g = grad[idx];
    if (weight_decay != 0.0f) {
        g = g + T(weight_decay) * param[idx];
    }

    T sq = square_avg[idx];
    sq = T(rho) * sq + T(1.0f - rho) * g * g;
    square_avg[idx] = sq;

    T std_val = sqrt(sq + T(eps));
    T delta = sqrt(acc_delta[idx] + T(eps)) / std_val * g;

    acc_delta[idx] = T(rho) * acc_delta[idx] + T(1.0f - rho) * delta * delta;

    param[idx] = param[idx] - T(lr) * delta;
}

auto fused_adadelta_step_hip(
    Tensor& param,
    const Tensor& grad,
    Tensor& square_avg,
    Tensor& acc_delta,
    float rho, float eps, float lr, float weight_decay,
    hipStream_t stream
) -> void {
    // BFloat16: upcast to Float32, compute, convert back
    if (param.dtype() == DType::BFloat16) {
        auto param_f32 = param.to(DType::Float32);
        auto grad_f32 = grad.to(DType::Float32);
        auto sa_f32 = square_avg.to(DType::Float32);
        auto ad_f32 = acc_delta.to(DType::Float32);
        fused_adadelta_step_hip(param_f32, grad_f32, sa_f32, ad_f32,
                                rho, eps, lr, weight_decay, stream);
        param = param_f32.to(DType::BFloat16);
        square_avg = sa_f32.to(DType::BFloat16);
        acc_delta = ad_f32.to(DType::BFloat16);
        return;
    }

    int64_t n = param.numel();
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    if (param.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fused_adadelta_step_kernel<float>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            param.data<float>(), grad.data<float>(), square_avg.data<float>(), acc_delta.data<float>(),
            rho, eps, lr, weight_decay, n);
    } else if (param.dtype() == DType::Float64) {
        hipLaunchKernelGGL(fused_adadelta_step_kernel<double>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            param.data<double>(), grad.data<double>(), square_avg.data<double>(), acc_delta.data<double>(),
            rho, eps, lr, weight_decay, n);
    } else {
        throw std::runtime_error("fused_adadelta_step_hip: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
}

// ==============================================================================
// Fused Adagrad Optimizer HIP Kernel
// ==============================================================================

template<typename T>
__global__ void fused_adagrad_step_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ sum_sq,
    float lr, float lr_decay, float eps, float weight_decay,
    int64_t step, int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T g = grad[idx];
    if (weight_decay != 0.0f) {
        g = g + T(weight_decay) * param[idx];
    }

    float clr = lr / (T(1) + T(step - 1) * T(lr_decay));

    T sq = sum_sq[idx] + g * g;
    sum_sq[idx] = sq;

    param[idx] = param[idx] - T(clr) * g / (sqrt(sq) + T(eps));
}

auto fused_adagrad_step_hip(
    Tensor& param,
    const Tensor& grad,
    Tensor& sum_sq,
    float lr, float lr_decay, float eps, float weight_decay,
    int64_t step,
    hipStream_t stream
) -> void {
    // BFloat16: upcast to Float32, compute, convert back
    if (param.dtype() == DType::BFloat16) {
        auto param_f32 = param.to(DType::Float32);
        auto grad_f32 = grad.to(DType::Float32);
        auto ss_f32 = sum_sq.to(DType::Float32);
        fused_adagrad_step_hip(param_f32, grad_f32, ss_f32, lr, lr_decay, eps,
                               weight_decay, step, stream);
        param = param_f32.to(DType::BFloat16);
        sum_sq = ss_f32.to(DType::BFloat16);
        return;
    }

    int64_t n = param.numel();
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    if (param.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fused_adagrad_step_kernel<float>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            param.data<float>(), grad.data<float>(), sum_sq.data<float>(),
            lr, lr_decay, eps, weight_decay, step, n);
    } else if (param.dtype() == DType::Float64) {
        hipLaunchKernelGGL(fused_adagrad_step_kernel<double>,
            dim3(num_blocks), dim3(block_size), 0, stream,
            param.data<double>(), grad.data<double>(), sum_sq.data<double>(),
            lr, lr_decay, eps, weight_decay, step, n);
    } else {
        throw std::runtime_error("fused_adagrad_step_hip: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
}

// ==============================================================================
// Fused Adam-Atan2 Optimizer HIP Kernel
// ==============================================================================

template<typename T>
__global__ void fused_adam_atan2_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ exp_avg,
    T* __restrict__ exp_avg_sq,
    T* __restrict__ max_exp_avg_sq,
    int64_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    bool amsgrad
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    T g = grad[idx];
    T p = param[idx];
    T m = exp_avg[idx];
    T v = exp_avg_sq[idx];

    m = T(beta1) * m + T(1.0f - beta1) * g;
    v = T(beta2) * v + T(1.0f - beta2) * g * g;

    T m_hat = m / T(bias_correction1);
    T v_hat = v / T(bias_correction2);

    if (amsgrad && max_exp_avg_sq != nullptr) {
        T max_v = max_exp_avg_sq[idx];
        if (v_hat > max_v) max_v = v_hat;
        max_exp_avg_sq[idx] = max_v;
        v_hat = max_v;
    }

    if (weight_decay > 0.0f) {
        p = p * (T(1) - T(lr) * T(weight_decay));
    }

    T denom = sqrt(v_hat) + T(eps);
    T update = atan2(m_hat, denom);

    p = p - T(lr) * update;

    param[idx] = p;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
}

auto fused_adam_atan2_step_hip(
    Tensor& param,
    const Tensor& grad,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    Tensor* max_exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int64_t step,
    bool amsgrad,
    hipStream_t stream
) -> void {
    // BFloat16: upcast to Float32, compute, convert back
    if (param.dtype() == DType::BFloat16) {
        auto param_f32 = param.to(DType::Float32);
        auto grad_f32 = grad.to(DType::Float32);
        auto ea_f32 = exp_avg.to(DType::Float32);
        auto eas_f32 = exp_avg_sq.to(DType::Float32);
        Tensor meas_f32;
        Tensor* meas_f32_ptr = nullptr;
        if (max_exp_avg_sq) {
            meas_f32 = max_exp_avg_sq->to(DType::Float32);
            meas_f32_ptr = &meas_f32;
        }
        fused_adam_atan2_step_hip(param_f32, grad_f32, ea_f32, eas_f32, meas_f32_ptr,
                                  lr, beta1, beta2, eps, weight_decay, step, amsgrad, stream);
        param = param_f32.to(DType::BFloat16);
        exp_avg = ea_f32.to(DType::BFloat16);
        exp_avg_sq = eas_f32.to(DType::BFloat16);
        if (max_exp_avg_sq) *max_exp_avg_sq = meas_f32.to(DType::BFloat16);
        return;
    }

    int64_t numel = param.numel();
    constexpr int BLOCK_SIZE = 256;
    int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = std::min(blocks, 65535);

    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    if (param.dtype() == DType::Float32) {
        float* max_sq_ptr = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<float>() : nullptr;

        hipLaunchKernelGGL(fused_adam_atan2_kernel<float>,
            dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
            param.data<float>(), grad.data<float>(),
            exp_avg.data<float>(), exp_avg_sq.data<float>(),
            max_sq_ptr, numel,
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2, amsgrad
        );
    } else if (param.dtype() == DType::Float64) {
        double* max_sq_ptr = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<double>() : nullptr;

        hipLaunchKernelGGL(fused_adam_atan2_kernel<double>,
            dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
            param.data<double>(), grad.data<double>(),
            exp_avg.data<double>(), exp_avg_sq.data<double>(),
            max_sq_ptr, numel,
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2, amsgrad
        );
    } else {
        throw std::runtime_error("fused_adam_atan2_step_hip: Only Float32 and Float64 supported");
    }

    HIP_CHECK(hipGetLastError());
}

// ==============================================================================
// Fused RMSNorm Backward HIP Kernel
// ==============================================================================

/**
 * @brief Fused RMSNorm backward kernel.
 *
 * Computes gradients for input and weight.
 * grad_input = weight * rrms * (grad_out - x * rrms^2 * mean(grad_out * x * weight))
 */
template<typename T, int BLOCK_SZ>
__global__ void fused_rms_norm_backward_kernel_hip(
    const T* grad_output,
    const T* input,
    const T* weight,
    const T* rrms,
    T* grad_input,
    T* grad_weight,
    int64_t batch_size,
    int64_t norm_size
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_grad_out = grad_output + b * norm_size;
    const T* batch_in = input + b * norm_size;
    T* batch_grad_in = grad_input + b * norm_size;

    T batch_rrms = rrms[b];

    __shared__ T shared_sum[BLOCK_SZ];

    // Compute sum(grad_out * x * weight) / norm_size for input gradient
    T sum_grad_x_w = 0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        sum_grad_x_w += batch_grad_out[i] * batch_in[i] * weight[i];
    }

    shared_sum[threadIdx.x] = sum_grad_x_w;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    T mean_grad_x_w = shared_sum[0] / norm_size;

    // Compute input gradient and accumulate weight gradient
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T x_i = batch_in[i];
        T w_i = weight[i];
        T grad_out_i = batch_grad_out[i];

        // grad_input = rrms * (grad_out * weight - x * rrms^2 * mean_grad_x_w)
        batch_grad_in[i] = batch_rrms * (grad_out_i * w_i - x_i * batch_rrms * batch_rrms * mean_grad_x_w);

        // grad_weight accumulation (atomic for thread safety across batches)
        atomicAdd(&grad_weight[i], grad_out_i * x_i * batch_rrms);
    }
}

auto fused_rms_norm_backward_hip(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& rrms
) -> std::tuple<Tensor, Tensor> {
    // BFloat16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        auto go_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto rrms_f32 = rrms.to(DType::Float32);
        auto [gi, gw] = fused_rms_norm_backward_hip(go_f32, input_f32, w_f32, rrms_f32);
        return {gi.to(DType::BFloat16), gw.to(DType::BFloat16)};
    }

    auto shape = input.shape();
    int64_t norm_size = shape.back();

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }

    // Create zero-initialized output tensors
    std::vector<int64_t> input_shape(input.shape().begin(), input.shape().end());
    Tensor grad_input(input_shape, input.dtype(), input.device());
    Tensor grad_weight({norm_size}, input.dtype(), input.device());

    // Zero-initialize
    HIP_CHECK(hipMemset(grad_input.data_ptr(), 0,
        grad_input.numel() * dtype_size(input.dtype())));
    HIP_CHECK(hipMemset(grad_weight.data_ptr(), 0,
        norm_size * dtype_size(input.dtype())));

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(
            (fused_rms_norm_backward_kernel_hip<float, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
            grad_output.data<float>(),
            input.data<float>(),
            weight.data<float>(),
            rrms.data<float>(),
            grad_input.data<float>(),
            grad_weight.data<float>(),
            batch_size,
            norm_size
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(
            (fused_rms_norm_backward_kernel_hip<double, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
            grad_output.data<double>(),
            input.data<double>(),
            weight.data<double>(),
            rrms.data<double>(),
            grad_input.data<double>(),
            grad_weight.data<double>(),
            batch_size,
            norm_size
        );
    } else {
        throw std::runtime_error("fused_rms_norm_backward_hip: Only Float32 and Float64 supported");
    }

    HIP_CHECK(hipGetLastError());

    return std::make_tuple(grad_input, grad_weight);
}

// ==============================================================================
// Fused LayerNorm Backward HIP Kernel
// ==============================================================================

/**
 * @brief HIP kernel for LayerNorm backward pass.
 *
 * Computes gradients for input, weight, and bias given output gradients.
 * Uses efficient parallel reduction for batch-wise operations.
 */
template<typename T, int BLOCK_SZ>
__global__ void fused_layer_norm_backward_kernel_hip(
    const T* grad_output,
    const T* input,
    const T* weight,
    const T* mean,
    const T* inv_std,
    T* grad_input,
    T* grad_weight,
    T* grad_bias,
    int64_t batch_size,
    int64_t norm_size
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_grad_out = grad_output + b * norm_size;
    const T* batch_in = input + b * norm_size;
    T* batch_grad_in = grad_input + b * norm_size;

    T batch_mean = mean[b];
    T batch_inv_std = inv_std[b];

    __shared__ T shared_sum1[BLOCK_SZ];  // For sum(grad_out * weight)
    __shared__ T shared_sum2[BLOCK_SZ];  // For sum(grad_out * weight * normalized)

    // Compute sums needed for input gradient
    T sum_grad_out = 0;
    T sum_grad_out_normalized = 0;

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        T grad_out_weighted = batch_grad_out[i] * weight[i];

        sum_grad_out += grad_out_weighted;
        sum_grad_out_normalized += grad_out_weighted * normalized;

        // Accumulate weight and bias gradients atomically
        atomicAdd(&grad_weight[i], batch_grad_out[i] * normalized);
        atomicAdd(&grad_bias[i], batch_grad_out[i]);
    }

    shared_sum1[threadIdx.x] = sum_grad_out;
    shared_sum2[threadIdx.x] = sum_grad_out_normalized;
    __syncthreads();

    // Parallel reduction for sums
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum1[threadIdx.x] += shared_sum1[threadIdx.x + s];
            shared_sum2[threadIdx.x] += shared_sum2[threadIdx.x + s];
        }
        __syncthreads();
    }

    T mean_grad_out = shared_sum1[0] / static_cast<T>(norm_size);
    T mean_grad_out_normalized = shared_sum2[0] / static_cast<T>(norm_size);

    // Compute input gradients
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        T grad_out_weighted = batch_grad_out[i] * weight[i];

        batch_grad_in[i] = (grad_out_weighted - mean_grad_out -
                           normalized * mean_grad_out_normalized) * batch_inv_std;
    }
}

auto fused_layer_norm_backward_hip(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& mean,
    const Tensor& inv_std,
    const std::vector<int64_t>& normalized_shape
) -> std::tuple<Tensor, Tensor, Tensor> {
    // BFloat16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        auto go_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto is_f32 = inv_std.to(DType::Float32);
        auto [gi, gw, gb] = fused_layer_norm_backward_hip(go_f32, input_f32, w_f32,
                                                            mean_f32, is_f32, normalized_shape);
        return {gi.to(DType::BFloat16), gw.to(DType::BFloat16), gb.to(DType::BFloat16)};
    }

    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    // Create zero-initialized output tensors
    std::vector<int64_t> input_shape(input.shape().begin(), input.shape().end());
    Tensor grad_input(input_shape, input.dtype(), input.device());
    Tensor grad_weight({norm_size}, input.dtype(), input.device());
    Tensor grad_bias({norm_size}, input.dtype(), input.device());

    // Zero-initialize
    HIP_CHECK(hipMemset(grad_input.data_ptr(), 0,
        grad_input.numel() * dtype_size(input.dtype())));
    HIP_CHECK(hipMemset(grad_weight.data_ptr(), 0,
        norm_size * dtype_size(input.dtype())));
    HIP_CHECK(hipMemset(grad_bias.data_ptr(), 0,
        norm_size * dtype_size(input.dtype())));

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(
            (fused_layer_norm_backward_kernel_hip<float, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
            grad_output.data<float>(),
            input.data<float>(),
            weight.data<float>(),
            mean.data<float>(),
            inv_std.data<float>(),
            grad_input.data<float>(),
            grad_weight.data<float>(),
            grad_bias.data<float>(),
            batch_size,
            norm_size
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(
            (fused_layer_norm_backward_kernel_hip<double, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
            grad_output.data<double>(),
            input.data<double>(),
            weight.data<double>(),
            mean.data<double>(),
            inv_std.data<double>(),
            grad_input.data<double>(),
            grad_weight.data<double>(),
            grad_bias.data<double>(),
            batch_size,
            norm_size
        );
    } else {
        throw std::runtime_error("fused_layer_norm_backward_hip: Only Float32 and Float64 supported");
    }

    HIP_CHECK(hipGetLastError());

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace rocm
} // namespace tenzor
