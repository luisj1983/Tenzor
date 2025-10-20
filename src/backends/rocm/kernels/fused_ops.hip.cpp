#ifdef TENZOR_ROCM_AVAILABLE

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tenzor/core/tensor.hpp"
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace rocm {

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
    const Tensor* bias
) -> Tensor {
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
    Tensor output = zeros(output_shape, input.dtype(), input.device());

    // Launch kernel
    int64_t total_elements = batch_size * out_features;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

    if (input.dtype() == DType::Float32) {
        const float* bias_ptr = bias ? bias->data<float>() : nullptr;
        hipLaunchKernelGGL(fused_linear_relu_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
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
    HIP_CHECK(hipDeviceSynchronize());

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
    int64_t batch_size = input.shape()[0];
    int64_t num_features = input.shape()[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < input.shape().size(); ++i) {
        spatial_size *= input.shape()[i];
    }

    Tensor output = zeros(input.shape(), input.dtype(), input.device());

    int64_t total_elements = input.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

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
    HIP_CHECK(hipDeviceSynchronize());

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
    int64_t batch_size = logits.shape()[0];
    int64_t num_classes = logits.shape()[1];

    Tensor losses = zeros({batch_size}, logits.dtype(), logits.device());

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
    HIP_CHECK(hipDeviceSynchronize());

    // Apply reduction
    if (reduction == "mean") {
        return mean(losses);
    } else if (reduction == "sum") {
        return sum(losses);
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
    Tensor result = zeros(a.shape(), a.dtype(), a.device());

    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

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
    HIP_CHECK(hipDeviceSynchronize());

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
    Tensor output = zeros(input.shape(), input.dtype(), input.device());

    int64_t n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

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
    HIP_CHECK(hipDeviceSynchronize());

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
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    Tensor output = zeros(input.shape(), input.dtype(), input.device());

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
    HIP_CHECK(hipDeviceSynchronize());

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
    int64_t batch_size = conv_output.shape()[0];
    int64_t num_features = conv_output.shape()[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < conv_output.shape().size(); ++i) {
        spatial_size *= conv_output.shape()[i];
    }

    Tensor output = zeros(conv_output.shape(), conv_output.dtype(), conv_output.device());

    int64_t total_elements = conv_output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

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
    HIP_CHECK(hipDeviceSynchronize());

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
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];

    Tensor C = zeros({M, N}, A.dtype(), A.device());

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
    HIP_CHECK(hipDeviceSynchronize());

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
    Tensor output = zeros(a.shape(), a.dtype(), a.device());

    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

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
    HIP_CHECK(hipDeviceSynchronize());

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

auto fused_attention_hip(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    float scale
) -> Tensor {
    int64_t batch_size = Q.shape()[0];
    int64_t seq_len = Q.shape()[1];
    int64_t d_k = Q.shape()[2];
    int64_t d_v = V.shape()[2];

    Tensor output = zeros({batch_size, seq_len, d_v}, Q.dtype(), Q.device());

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
    } else {
        throw std::runtime_error("fused_attention_hip: Only Float32 supported");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return output;
}

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_ROCM_AVAILABLE
