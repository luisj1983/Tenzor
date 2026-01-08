#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <hiprand/hiprand.h>
#include <cmath>
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
            throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
        } \
    } while(0)

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return std::min(static_cast<int64_t>((n + block_size - 1) / block_size), static_cast<int64_t>(65535));
}

// ============================================================================
// Helper Kernels
// ============================================================================

template<typename T>
__global__ void add_bias_to_output_kernel(const T* __restrict__ bias, T* __restrict__ output,
                                          int64_t batch, int64_t features) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * features) {
        int64_t f = idx % features;
        output[idx] += bias[f];
    }
}

template<typename T>
__global__ void sum_over_batch_kernel(const T* __restrict__ grad, T* __restrict__ grad_bias,
                                       int64_t batch_offset, int64_t features) {
    int64_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < features) {
        atomicAdd(&grad_bias[f], grad[batch_offset + f]);
    }
}

// ============================================================================
// Embedding Kernels
// ============================================================================

template<typename T>
__global__ void embedding_kernel_hip(
    const T* __restrict__ weight,
    const int64_t* __restrict__ indices,
    T* __restrict__ output,
    int64_t num_indices,
    int64_t embedding_dim) {

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = num_indices * embedding_dim;

    if (tid < total_elements) {
        int64_t idx = tid / embedding_dim;
        int64_t dim = tid % embedding_dim;
        int64_t embedding_idx = indices[idx];
        output[tid] = weight[embedding_idx * embedding_dim + dim];
    }
}

template<typename T>
__global__ void embedding_backward_kernel_hip(
    const T* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    T* __restrict__ grad_weight,
    int64_t num_indices,
    int64_t embedding_dim) {

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = num_indices * embedding_dim;

    if (tid < total_elements) {
        int64_t idx = tid / embedding_dim;
        int64_t dim = tid % embedding_dim;
        int64_t embedding_idx = indices[idx];
        // Atomic add for accumulating gradients
        atomicAdd(&grad_weight[embedding_idx * embedding_dim + dim], grad_output[tid]);
    }
}

auto embedding_kernel(const Tensor& weight, const Tensor& indices, hipStream_t stream) -> Tensor {
    // weight: [num_embeddings, embedding_dim]
    // indices: [*] (any shape of int64 indices)
    // output: [*, embedding_dim]

    auto w_shape = weight.shape();
    auto idx_shape = indices.shape();

    int64_t embedding_dim = w_shape[1];
    int64_t num_indices = indices.numel();

    std::vector<int64_t> out_shape(idx_shape.begin(), idx_shape.end());
    out_shape.push_back(embedding_dim);

    Tensor output(out_shape, weight.dtype(), weight.device());

    int64_t total_elements = num_indices * embedding_dim;
    int num_blocks = get_num_blocks(total_elements);

    if (weight.dtype() == DType::Float32) {
        hipLaunchKernelGGL(embedding_kernel_hip<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            weight.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            num_indices,
            embedding_dim);
    } else if (weight.dtype() == DType::Float64) {
        hipLaunchKernelGGL(embedding_kernel_hip<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            weight.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            num_indices,
            embedding_dim);
    } else {
        throw std::runtime_error("Embedding only supports Float32 and Float64");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                               int64_t num_embeddings, hipStream_t stream) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t embedding_dim = grad_shape[grad_shape.size() - 1];
    int64_t num_indices = indices.numel();

    // Initialize grad_weight to zeros
    Tensor grad_weight({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());
    HIP_CHECK(hipMemsetAsync(grad_weight.data<void>(), 0,
        num_embeddings * embedding_dim * dtype_size(grad_output.dtype()), stream));

    int64_t total_elements = num_indices * embedding_dim;
    int num_blocks = get_num_blocks(total_elements);

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(embedding_backward_kernel_hip<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            grad_output.data<float>(),
            indices.data<int64_t>(),
            grad_weight.data<float>(),
            num_indices,
            embedding_dim);
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(embedding_backward_kernel_hip<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            grad_output.data<double>(),
            indices.data<int64_t>(),
            grad_weight.data<double>(),
            num_indices,
            embedding_dim);
    } else {
        throw std::runtime_error("Embedding backward only supports Float32 and Float64");
    }

    HIP_CHECK(hipGetLastError());
    return grad_weight;
}

// ============================================================================
// Linear Kernels (using rocBLAS)
// ============================================================================

auto linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                   hipStream_t stream) -> Tensor {
    // input: [*, in_features] or [batch, in_features]
    // weight: [out_features, in_features]
    // output: [*, out_features]

    auto in_shape = input.shape();
    auto w_shape = weight.shape();
    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    // Handle batched input
    int64_t batch_size = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }

    // Build output shape
    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end() - 1);
    out_shape.push_back(out_features);

    Tensor output(out_shape, input.dtype(), input.device());

    // Use rocBLAS for matrix multiply: output = input @ weight.T
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    rocblas_set_stream(handle, stream);

    if (input.dtype() == DType::Float32) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // output (batch x out_features) = input (batch x in_features) @ weight.T (in_features x out_features)
        // rocBLAS uses column-major, so we compute: output.T = weight @ input.T
        rocblas_sgemm(handle,
            rocblas_operation_transpose, rocblas_operation_none,
            out_features, batch_size, in_features,
            &alpha,
            weight.data<float>(), in_features,
            input.data<float>(), in_features,
            &beta,
            output.data<float>(), out_features);

        // Add bias if present
        if (bias != nullptr) {
            int64_t total = batch_size * out_features;
            int num_blocks = get_num_blocks(total);
            hipLaunchKernelGGL(add_bias_to_output_kernel<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                bias->data<float>(), output.data<float>(), batch_size, out_features);
        }
    } else if (input.dtype() == DType::Float64) {
        const double alpha = 1.0;
        const double beta = 0.0;

        rocblas_dgemm(handle,
            rocblas_operation_transpose, rocblas_operation_none,
            out_features, batch_size, in_features,
            &alpha,
            weight.data<double>(), in_features,
            input.data<double>(), in_features,
            &beta,
            output.data<double>(), out_features);

        // Add bias if present
        if (bias != nullptr) {
            int64_t total = batch_size * out_features;
            int num_blocks = get_num_blocks(total);
            hipLaunchKernelGGL(add_bias_to_output_kernel<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                bias->data<double>(), output.data<double>(), batch_size, out_features);
        }
    } else {
        rocblas_destroy_handle(handle);
        throw std::runtime_error("Linear only supports Float32 and Float64");
    }

    rocblas_destroy_handle(handle);
    HIP_CHECK(hipGetLastError());
    return output;
}

auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                            hipStream_t stream) -> std::vector<Tensor> {
    // grad_output: [batch, out_features]
    // input: [batch, in_features]
    // weight: [out_features, in_features]

    auto grad_shape = grad_output.shape();
    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    int64_t batch_size = 1;
    for (size_t i = 0; i < grad_shape.size() - 1; ++i) {
        batch_size *= grad_shape[i];
    }
    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    // grad_input = grad_output @ weight
    // grad_weight = grad_output.T @ input
    // grad_bias = grad_output.sum(dim=0)

    Tensor grad_input(in_shape, input.dtype(), input.device());
    Tensor grad_weight(w_shape, weight.dtype(), weight.device());
    Tensor grad_bias({out_features}, input.dtype(), input.device());

    rocblas_handle handle;
    rocblas_create_handle(&handle);
    rocblas_set_stream(handle, stream);

    if (input.dtype() == DType::Float32) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // grad_input = grad_output @ weight
        rocblas_sgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            in_features, batch_size, out_features,
            &alpha,
            weight.data<float>(), in_features,
            grad_output.data<float>(), out_features,
            &beta,
            grad_input.data<float>(), in_features);

        // grad_weight = grad_output.T @ input
        rocblas_sgemm(handle,
            rocblas_operation_none, rocblas_operation_transpose,
            in_features, out_features, batch_size,
            &alpha,
            input.data<float>(), in_features,
            grad_output.data<float>(), out_features,
            &beta,
            grad_weight.data<float>(), in_features);

        // grad_bias = sum over batch dimension
        HIP_CHECK(hipMemsetAsync(grad_bias.data<float>(), 0, out_features * sizeof(float), stream));
        int num_blocks = get_num_blocks(out_features);
        // Sum over batch dimension for each output feature
        for (int64_t b = 0; b < batch_size; ++b) {
            hipLaunchKernelGGL(sum_over_batch_kernel<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                grad_output.data<float>(), grad_bias.data<float>(), b * out_features, out_features);
        }
    } else if (input.dtype() == DType::Float64) {
        const double alpha = 1.0;
        const double beta = 0.0;

        rocblas_dgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            in_features, batch_size, out_features,
            &alpha,
            weight.data<double>(), in_features,
            grad_output.data<double>(), out_features,
            &beta,
            grad_input.data<double>(), in_features);

        rocblas_dgemm(handle,
            rocblas_operation_none, rocblas_operation_transpose,
            in_features, out_features, batch_size,
            &alpha,
            input.data<double>(), in_features,
            grad_output.data<double>(), out_features,
            &beta,
            grad_weight.data<double>(), in_features);

        HIP_CHECK(hipMemsetAsync(grad_bias.data<double>(), 0, out_features * sizeof(double), stream));
        int num_blocks = get_num_blocks(out_features);
        for (int64_t b = 0; b < batch_size; ++b) {
            hipLaunchKernelGGL(sum_over_batch_kernel<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                grad_output.data<double>(), grad_bias.data<double>(), b * out_features, out_features);
        }
    } else {
        rocblas_destroy_handle(handle);
        throw std::runtime_error("Linear backward only supports Float32 and Float64");
    }

    rocblas_destroy_handle(handle);
    HIP_CHECK(hipGetLastError());
    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// Dropout Kernels
// ============================================================================

template<typename T>
__global__ void dropout_forward_kernel(
    const T* __restrict__ input,
    const float* __restrict__ random_values,
    T* __restrict__ output,
    float* __restrict__ mask,
    int64_t n,
    float p,
    float scale) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = random_values[idx];
        if (r < p) {
            mask[idx] = 0.0f;
            output[idx] = T(0);
        } else {
            mask[idx] = 1.0f;
            output[idx] = input[idx] * T(scale);
        }
    }
}

template<typename T>
__global__ void dropout_backward_kernel_hip(
    const T* __restrict__ grad_output,
    const float* __restrict__ mask,
    T* __restrict__ grad_input,
    int64_t n,
    float scale) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * T(mask[idx]) * T(scale);
    }
}

auto dropout_kernel(const Tensor& input, float p, bool training, hipStream_t stream)
    -> std::pair<Tensor, Tensor> {

    if (!training || p == 0.0f) {
        // During inference or p=0, just return input and empty mask
        return {input, Tensor()};
    }

    int64_t n = input.numel();
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    Tensor mask(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                DType::Float32, input.device());

    // Generate random values on device
    Tensor random_values({n}, DType::Float32, input.device());

    hiprandGenerator_t gen;
    hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandSetStream(gen, stream);
    hiprandGenerateUniform(gen, random_values.data<float>(), n);

    float scale = 1.0f / (1.0f - p);
    int num_blocks = get_num_blocks(n);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(dropout_forward_kernel<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(),
            random_values.data<float>(),
            output.data<float>(),
            mask.data<float>(),
            n, p, scale);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(dropout_forward_kernel<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(),
            random_values.data<float>(),
            output.data<double>(),
            mask.data<float>(),
            n, p, scale);
    } else {
        hiprandDestroyGenerator(gen);
        throw std::runtime_error("Dropout only supports Float32 and Float64");
    }

    hiprandDestroyGenerator(gen);
    HIP_CHECK(hipGetLastError());
    return {output, mask};
}

auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p,
                             hipStream_t stream) -> Tensor {

    int64_t n = grad_output.numel();
    Tensor grad_input(std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end()),
                      grad_output.dtype(), grad_output.device());

    float scale = 1.0f / (1.0f - p);
    int num_blocks = get_num_blocks(n);

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(dropout_backward_kernel_hip<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            grad_output.data<float>(),
            mask.data<float>(),
            grad_input.data<float>(),
            n, scale);
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(dropout_backward_kernel_hip<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            grad_output.data<double>(),
            mask.data<float>(),
            grad_input.data<double>(),
            n, scale);
    } else {
        throw std::runtime_error("Dropout backward only supports Float32 and Float64");
    }

    HIP_CHECK(hipGetLastError());
    return grad_input;
}

} // namespace rocm
} // namespace tenzor
