#include "rocm_nan_helpers.hip.h"  // E.2: safe_f2h / safe_h2f / safe_f2bf / safe_bf2f
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <hiprand/hiprand.h>
#include <cmath>
#include <cstdio>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "fp16_saturate.h"
#include "../rocm_error.hpp"
#include <stdexcept>

namespace tenzor {
namespace rocm {

constexpr int BLOCK_SIZE = 256;

// Throw on any non-success hiprand status (matches the HIP_CHECK/ROCBLAS_CHECK
// discipline used throughout the backend).
#define HIPRAND_CHECK(call)                                                       \
    do {                                                                          \
        hiprandStatus_t _hiprand_status = (call);                                 \
        if (_hiprand_status != HIPRAND_STATUS_SUCCESS) {                          \
            throw std::runtime_error("hiprand error (" +                          \
                std::to_string(static_cast<int>(_hiprand_status)) + ") at " +     \
                __FILE__ ":" + std::to_string(__LINE__));                         \
        }                                                                         \
    } while (0)

// RAII wrapper for a hiprand generator so it is destroyed on every exit path,
// including exceptions thrown by an intervening HIP/HIPRAND check.
class HiprandGeneratorGuard {
public:
    explicit HiprandGeneratorGuard(hiprandRngType_t rng_type) {
        HIPRAND_CHECK(hiprandCreateGenerator(&gen_, rng_type));
    }
    ~HiprandGeneratorGuard() {
        if (gen_) hiprandDestroyGenerator(gen_);
    }
    HiprandGeneratorGuard(const HiprandGeneratorGuard&) = delete;
    HiprandGeneratorGuard& operator=(const HiprandGeneratorGuard&) = delete;
    hiprandGenerator_t get() const { return gen_; }
private:
    hiprandGenerator_t gen_ = nullptr;
};

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

// Return a thread-local rocBLAS handle, lazily constructed on first use.
// rocblas_create_handle() is expensive (pinned host alloc, device probe), so
// the hot linear forward/backward paths reuse one handle per thread and only
// swap the stream (a cheap pointer store), mirroring matmul.hip.cpp.
static RocBLASHandleGuard& cached_rocblas_handle(hipStream_t stream) {
    thread_local RocBLASHandleGuard h;
    h.set_stream(stream);
    return h;
}

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return std::min(static_cast<int64_t>((n + block_size - 1) / block_size), static_cast<int64_t>(65535));
}

// ============================================================================
// Helper Kernels
// ============================================================================

template<typename T>
__global__ void add_bias_to_output_kernel(const T* __restrict__ bias, T* __restrict__ output,
                                          int64_t batch, int64_t features) {
    int64_t total = batch * features;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t f = idx % features;
        output[idx] += bias[f];
    }
}

// Reduce grad_output (batch x features) over the batch dimension into
// grad_bias[features] in a single grid-strided launch (one atomicAdd per
// element) instead of launching one tiny kernel per batch row.
template<typename T>
__global__ void sum_over_batch_kernel(const T* __restrict__ grad, T* __restrict__ grad_bias,
                                       int64_t batch, int64_t features) {
    int64_t total = batch * features;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        int64_t f = idx % features;
        atomicAdd(&grad_bias[f], grad[idx]);
    }
}

// Float16 bias kernel
__global__ void add_bias_to_output_kernel_fp16(const __half* __restrict__ bias, __half* __restrict__ output,
                                                int64_t batch, int64_t features) {
    int64_t total = batch * features;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t f = idx % features;
        output[idx] = tenzor::rocm::safe_f2h(tenzor::rocm::safe_h2f(output[idx]) + tenzor::rocm::safe_h2f(bias[f]));
    }
}

// Float16 sum over batch kernel (accumulate in float)
__global__ void sum_over_batch_kernel_fp16(const __half* __restrict__ grad, float* __restrict__ grad_bias_f32,
                                            int64_t batch, int64_t features) {
    int64_t total = batch * features;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        int64_t f = idx % features;
        atomicAdd(&grad_bias_f32[f], tenzor::rocm::safe_h2f(grad[idx]));
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
    int64_t embedding_dim,
    int64_t num_embeddings,
    int* error_flag) {

    int64_t total_elements = num_indices * embedding_dim;
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total_elements; tid += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t idx = tid / embedding_dim;
        int64_t dim = tid % embedding_dim;
        int64_t embedding_idx = indices[idx];
        // Bounds check: an out-of-range index writes 0 (memory-safe) and flags
        // the error; the host throws std::out_of_range after sync, matching the
        // CPU reference and the CUDA backend.
        if (embedding_idx < 0 || embedding_idx >= num_embeddings) {
            output[tid] = T(0);
            atomicOr(error_flag, 1);
            continue;
        }
        output[tid] = weight[embedding_idx * embedding_dim + dim];
    }
}

template<typename T>
__global__ void embedding_backward_kernel_hip(
    const T* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    T* __restrict__ grad_weight,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings) {

    int64_t total_elements = num_indices * embedding_dim;
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total_elements; tid += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t idx = tid / embedding_dim;
        int64_t dim = tid % embedding_dim;
        int64_t embedding_idx = indices[idx];
        // Skip out-of-range indices to avoid OOB atomic writes.
        if (embedding_idx < 0 || embedding_idx >= num_embeddings) {
            continue;
        }
        // Atomic add for accumulating gradients
        atomicAdd(&grad_weight[embedding_idx * embedding_dim + dim], grad_output[tid]);
    }
}

// Float16 embedding forward kernel
__global__ void embedding_kernel_hip_fp16(
    const __half* __restrict__ weight,
    const int64_t* __restrict__ indices,
    __half* __restrict__ output,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings,
    int* error_flag) {

    int64_t total_elements = num_indices * embedding_dim;
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total_elements; tid += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t idx = tid / embedding_dim;
        int64_t dim = tid % embedding_dim;
        int64_t embedding_idx = indices[idx];
        if (embedding_idx < 0 || embedding_idx >= num_embeddings) {
            output[tid] = __float2half(0.0f);
            atomicOr(error_flag, 1);
            continue;
        }
        output[tid] = weight[embedding_idx * embedding_dim + dim];
    }
}

// Float16 embedding backward kernel (uses float accumulation for atomicAdd)
__global__ void embedding_backward_kernel_hip_fp16(
    const __half* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_weight_f32,  // accumulate in float
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings) {

    int64_t total_elements = num_indices * embedding_dim;
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total_elements; tid += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t idx = tid / embedding_dim;
        int64_t dim = tid % embedding_dim;
        int64_t embedding_idx = indices[idx];
        if (embedding_idx < 0 || embedding_idx >= num_embeddings) {
            continue;
        }
        atomicAdd(&grad_weight_f32[embedding_idx * embedding_dim + dim], tenzor::rocm::safe_h2f(grad_output[tid]));
    }
}

// Convert float gradients to Float16
__global__ void convert_f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        dst[idx] = tenzor::rocm::safe_f2h(src[idx]);
    }
}

auto embedding_kernel(const Tensor& weight, const Tensor& indices, hipStream_t stream) -> Tensor {
    // weight: [num_embeddings, embedding_dim]
    // indices: [*] (any shape of int64 indices)
    // output: [*, embedding_dim]

    auto w_shape = weight.shape();
    auto idx_shape = indices.shape();

    int64_t embedding_dim = w_shape[1];
    int64_t num_embeddings = w_shape[0];
    int64_t num_indices = indices.numel();

    std::vector<int64_t> out_shape(idx_shape.begin(), idx_shape.end());
    out_shape.push_back(embedding_dim);

    Tensor output(out_shape, weight.dtype(), weight.device());

    if (num_indices == 0) {
        return output;
    }

    int64_t total_elements = num_indices * embedding_dim;
    int num_blocks = get_num_blocks(total_elements);

    // Device flag for out-of-range indices; the kernels write a zero row and set
    // this flag, and the host throws std::out_of_range after sync (matching the
    // CPU reference and the CUDA backend). The BFloat16 path recurses through the
    // Float32 branch, so it inherits the same check.
    Tensor err_tensor(std::vector<int64_t>{1}, DType::Int32, weight.device());
    HIP_CHECK(hipMemsetAsync(err_tensor.data_ptr(), 0, sizeof(int32_t), stream));
    int* err_ptr = err_tensor.data<int32_t>();

    if (weight.dtype() == DType::Float32) {
        hipLaunchKernelGGL(embedding_kernel_hip<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            weight.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            err_ptr);
    } else if (weight.dtype() == DType::Float64) {
        hipLaunchKernelGGL(embedding_kernel_hip<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            weight.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            err_ptr);
    } else if (weight.dtype() == DType::Float16) {
        hipLaunchKernelGGL(embedding_kernel_hip_fp16,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(weight.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            num_indices,
            embedding_dim,
            num_embeddings,
            err_ptr);
    } else if (weight.dtype() == DType::BFloat16) {
        auto weight_f32 = weight.to(DType::Float32);
        auto result_f32 = embedding_kernel(weight_f32, indices, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Embedding only supports Float32, Float64, Float16, and BFloat16");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    if (err_tensor.to(Device::cpu()).data<int32_t>()[0] != 0) {
        throw std::out_of_range("Embedding ROCm: index out of range");
    }
    return output;
}

auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                               int64_t num_embeddings, hipStream_t stream) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t embedding_dim = grad_shape[grad_shape.size() - 1];
    int64_t num_indices = indices.numel();

    // Initialize grad_weight to zeros
    Tensor grad_weight({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());
    HIP_CHECK(hipMemsetAsync(grad_weight.data_ptr(), 0,
        num_embeddings * embedding_dim * dtype_size(grad_output.dtype()), stream));

    // Empty index batch (e.g. fully-masked/padded sequences): grad_weight is
    // already correctly zeroed; launching with a zero-block grid would be an
    // invalid HIP configuration. Mirror the forward kernel's empty-input guard.
    if (num_indices == 0) {
        return grad_weight;
    }

    int64_t total_elements = num_indices * embedding_dim;
    int num_blocks = get_num_blocks(total_elements);

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(embedding_backward_kernel_hip<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            grad_output.data<float>(),
            indices.data<int64_t>(),
            grad_weight.data<float>(),
            num_indices,
            embedding_dim,
            num_embeddings);
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(embedding_backward_kernel_hip<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            grad_output.data<double>(),
            indices.data<int64_t>(),
            grad_weight.data<double>(),
            num_indices,
            embedding_dim,
            num_embeddings);
    } else if (grad_output.dtype() == DType::Float16) {
        // For Float16, accumulate gradients in float, then convert back
        int64_t grad_weight_size = num_embeddings * embedding_dim;
        Tensor grad_weight_f32({num_embeddings, embedding_dim}, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemsetAsync(grad_weight_f32.data<float>(), 0, grad_weight_size * sizeof(float), stream));

        hipLaunchKernelGGL(embedding_backward_kernel_hip_fp16,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            indices.data<int64_t>(),
            grad_weight_f32.data<float>(),
            num_indices,
            embedding_dim,
            num_embeddings);

        // Convert float gradients to Float16
        int convert_blocks = get_num_blocks(grad_weight_size);
        hipLaunchKernelGGL(convert_f32_to_f16_kernel,
            dim3(convert_blocks), dim3(BLOCK_SIZE), 0, stream,
            grad_weight_f32.data<float>(),
            reinterpret_cast<__half*>(grad_weight.data<Float16>()),
            grad_weight_size);
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = embedding_backward_kernel(grad_output_f32, indices, num_embeddings, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Embedding backward only supports Float32, Float64, Float16, and BFloat16");
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
    rocblas_handle handle = cached_rocblas_handle(stream).get();

    if (input.dtype() == DType::Float32) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // output (batch x out_features) = input (batch x in_features) @ weight.T (in_features x out_features)
        // rocBLAS uses column-major, so we compute: output.T = weight @ input.T
        ROCBLAS_CHECK(rocblas_sgemm(handle,
            rocblas_operation_transpose, rocblas_operation_none,
            out_features, batch_size, in_features,
            &alpha,
            weight.data<float>(), in_features,
            input.data<float>(), in_features,
            &beta,
            output.data<float>(), out_features));

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

        ROCBLAS_CHECK(rocblas_dgemm(handle,
            rocblas_operation_transpose, rocblas_operation_none,
            out_features, batch_size, in_features,
            &alpha,
            weight.data<double>(), in_features,
            input.data<double>(), in_features,
            &beta,
            output.data<double>(), out_features));

        // Add bias if present
        if (bias != nullptr) {
            int64_t total = batch_size * out_features;
            int num_blocks = get_num_blocks(total);
            hipLaunchKernelGGL(add_bias_to_output_kernel<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                bias->data<double>(), output.data<double>(), batch_size, out_features);
        }
    } else if (input.dtype() == DType::Float16) {
        // Float16: upcast to Float32 to prevent FP16 accumulation overflow in hgemm
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        const Tensor* bias_f32_ptr = nullptr;
        Tensor bias_f32;
        if (bias) { bias_f32 = bias->to(DType::Float32); bias_f32_ptr = &bias_f32; }
        auto result_f32 = linear_kernel(input_f32, weight_f32, bias_f32_ptr, stream);
        auto result_f16 = result_f32.to(DType::Float16);
        fp16_saturate(result_f16.data_ptr(), result_f16.numel(), stream);
        return result_f16;
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        const Tensor* bias_f32_ptr = nullptr;
        Tensor bias_f32;
        if (bias) { bias_f32 = bias->to(DType::Float32); bias_f32_ptr = &bias_f32; }
        auto result_f32 = linear_kernel(input_f32, weight_f32, bias_f32_ptr, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Linear only supports Float32, Float64, Float16, and BFloat16");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                            hipStream_t stream) -> std::vector<Tensor> {
    // grad_output: [batch, out_features]
    // input: [batch, in_features]
    // weight: [out_features, in_features]

    // Float16: upcast to Float32 to prevent FP16 accumulation overflow in hgemm
    if (input.dtype() == DType::Float16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto results_f32 = linear_backward_kernel(grad_output_f32, input_f32, weight_f32, stream);
        auto gi = results_f32[0].to(DType::Float16);
        auto gw = results_f32[1].to(DType::Float16);
        auto gb = results_f32[2].to(DType::Float16);
        fp16_saturate(gi.data_ptr(), gi.numel(), stream);
        fp16_saturate(gw.data_ptr(), gw.numel(), stream);
        fp16_saturate(gb.data_ptr(), gb.numel(), stream);
        return {gi, gw, gb};
    }

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

    Tensor grad_input(std::vector<int64_t>(in_shape.begin(), in_shape.end()), input.dtype(), input.device());
    Tensor grad_weight(std::vector<int64_t>(w_shape.begin(), w_shape.end()), weight.dtype(), weight.device());
    Tensor grad_bias({out_features}, input.dtype(), input.device());

    rocblas_handle handle = cached_rocblas_handle(stream).get();

    if (input.dtype() == DType::Float32) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // grad_input = grad_output @ weight
        ROCBLAS_CHECK(rocblas_sgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            in_features, batch_size, out_features,
            &alpha,
            weight.data<float>(), in_features,
            grad_output.data<float>(), out_features,
            &beta,
            grad_input.data<float>(), in_features));

        // grad_weight = grad_output.T @ input
        ROCBLAS_CHECK(rocblas_sgemm(handle,
            rocblas_operation_none, rocblas_operation_transpose,
            in_features, out_features, batch_size,
            &alpha,
            input.data<float>(), in_features,
            grad_output.data<float>(), out_features,
            &beta,
            grad_weight.data<float>(), in_features));

        // grad_bias = sum over batch dimension (single reduction launch)
        HIP_CHECK(hipMemsetAsync(grad_bias.data<float>(), 0, out_features * sizeof(float), stream));
        int num_blocks = get_num_blocks(batch_size * out_features);
        hipLaunchKernelGGL(sum_over_batch_kernel<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            grad_output.data<float>(), grad_bias.data<float>(), batch_size, out_features);
    } else if (input.dtype() == DType::Float64) {
        const double alpha = 1.0;
        const double beta = 0.0;

        ROCBLAS_CHECK(rocblas_dgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            in_features, batch_size, out_features,
            &alpha,
            weight.data<double>(), in_features,
            grad_output.data<double>(), out_features,
            &beta,
            grad_input.data<double>(), in_features));

        ROCBLAS_CHECK(rocblas_dgemm(handle,
            rocblas_operation_none, rocblas_operation_transpose,
            in_features, out_features, batch_size,
            &alpha,
            input.data<double>(), in_features,
            grad_output.data<double>(), out_features,
            &beta,
            grad_weight.data<double>(), in_features));

        HIP_CHECK(hipMemsetAsync(grad_bias.data<double>(), 0, out_features * sizeof(double), stream));
        int num_blocks = get_num_blocks(batch_size * out_features);
        hipLaunchKernelGGL(sum_over_batch_kernel<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            grad_output.data<double>(), grad_bias.data<double>(), batch_size, out_features);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto results_f32 = linear_backward_kernel(grad_output_f32, input_f32, weight_f32, stream);
        return {results_f32[0].to(DType::BFloat16), results_f32[1].to(DType::BFloat16), results_f32[2].to(DType::BFloat16)};
    } else {
        throw std::runtime_error("Linear backward only supports Float32, Float64, Float16, and BFloat16");
    }

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

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
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

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        grad_input[idx] = grad_output[idx] * T(mask[idx]) * T(scale);
    }
}

// Float16 dropout forward kernel
__global__ void dropout_forward_kernel_fp16(
    const __half* __restrict__ input,
    const float* __restrict__ random_values,
    __half* __restrict__ output,
    float* __restrict__ mask,
    int64_t n,
    float p,
    float scale) {

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float r = random_values[idx];
        if (r < p) {
            mask[idx] = 0.0f;
            output[idx] = tenzor::rocm::safe_f2h(0.0f);
        } else {
            mask[idx] = 1.0f;
            output[idx] = tenzor::rocm::safe_f2h(tenzor::rocm::safe_h2f(input[idx]) * scale);
        }
    }
}

// Float16 dropout backward kernel
__global__ void dropout_backward_kernel_hip_fp16(
    const __half* __restrict__ grad_output,
    const float* __restrict__ mask,
    __half* __restrict__ grad_input,
    int64_t n,
    float scale) {

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        grad_input[idx] = tenzor::rocm::safe_f2h(tenzor::rocm::safe_h2f(grad_output[idx]) * mask[idx] * scale);
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

    // Empty tensor: nothing to do. hiprandGenerateUniform(n=0) and a zero-block
    // grid would make HIP reject the launch ("invalid configuration argument").
    if (n == 0) {
        return {output, mask};
    }

    // Generate random values on device
    Tensor random_values({n}, DType::Float32, input.device());

    HiprandGeneratorGuard gen_guard(HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandGenerator_t gen = gen_guard.get();
    HIPRAND_CHECK(hiprandSetStream(gen, stream));
    HIPRAND_CHECK(hiprandGenerateUniform(gen, random_values.data<float>(), n));

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
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(dropout_forward_kernel_fp16,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            random_values.data<float>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            mask.data<float>(),
            n, p, scale);
    } else if (input.dtype() == DType::BFloat16) {
        // gen_guard frees the generator when it goes out of scope on return.
        auto input_f32 = input.to(DType::Float32);
        auto [output_f32, mask_out] = dropout_kernel(input_f32, p, training, stream);
        return {output_f32.to(DType::BFloat16), mask_out};
    } else {
        throw std::runtime_error("Dropout only supports Float32, Float64, Float16, and BFloat16");
    }

    HIP_CHECK(hipGetLastError());
    return {output, mask};
}

auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p,
                             hipStream_t stream) -> Tensor {

    int64_t n = grad_output.numel();
    Tensor grad_input(std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end()),
                      grad_output.dtype(), grad_output.device());

    // Empty tensor: get_num_blocks(0) == 0 and HIP rejects a zero-block grid
    // with "invalid configuration argument" (forward dropout_kernel guards the
    // same way). Return the empty grad_input instead of crashing.
    if (n == 0) return grad_input;

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
    } else if (grad_output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(dropout_backward_kernel_hip_fp16,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            mask.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            n, scale);
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = dropout_backward_kernel(grad_output_f32, mask, p, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("Dropout backward only supports Float32, Float64, Float16, and BFloat16");
    }

    HIP_CHECK(hipGetLastError());
    return grad_input;
}

} // namespace rocm
} // namespace tenzor
