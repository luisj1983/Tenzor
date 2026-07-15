#ifdef TENZOR_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "cuda_common.cuh"
#include "cuda_launch_utils.cuh"  // CudaBuffer (device-side error flag)
#include "../cublas_handle_pool.hpp"
#include "../cuda_stream.hpp"  // cuda::cuda_current_stream() -- JIT-R115
#include "../cuda_stream_pool.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <vector>
#include <type_traits>

namespace tenzor {
namespace cuda {

// Helper to clamp blocks to max grid size
inline int clamp_blocks(int64_t blocks) {
    // Ensure at least 1 block to avoid CUDA invalid argument error
    // Grid-stride loop will naturally handle n=0 by not executing any iterations
    int64_t clamped = std::min(blocks, static_cast<int64_t>(2147483647));  // 2^31-1
    return static_cast<int>(clamped > 0 ? clamped : 1);
}

// One-block-per-row grid (used by the fused layernorm/softmax kernels, mirroring
// the softmax_grid_blocks helper in activations.cu). `batch_size` is int64_t; a
// plain `int blocks = batch_size` truncates to a wrong/negative grid extent once
// the row count exceeds INT_MAX, silently leaving most rows uncomputed. CUDA caps
// gridDim.x at 2^31-1, so validate (throw, not silently clamp) before launch.
inline unsigned int softmax_grid_blocks(int64_t batch_size) {
    if (batch_size < 0) {
        throw std::runtime_error("fused kernel: negative batch size");
    }
    constexpr int64_t kMaxGridDimX = 2147483647LL;  // CUDA gridDim.x limit
    if (batch_size > kMaxGridDimX) {
        throw std::runtime_error(
            "fused kernel: number of rows " + std::to_string(batch_size) +
            " exceeds maximum CUDA grid dimension " + std::to_string(kMaxGridDimX));
    }
    return static_cast<unsigned int>(batch_size);
}

// Helper to create a zero-initialized CUDA tensor
inline Tensor create_cuda_zeros(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream = nullptr) {
    Tensor t(shape, dtype, device);
    size_t bytes = t.numel() * dtype_size(dtype);
    // Use data_ptr() which returns void* for any dtype
    TENZOR_CUDA_CHECK(cudaMemsetAsync(t.data_ptr(), 0, bytes, stream));
    return t;
}

// Helper to convert span to vector
inline std::vector<int64_t> to_vector(const std::span<const int64_t>& s) {
    return std::vector<int64_t>(s.begin(), s.end());
}

// Scale a scalar value in device memory entirely on-device (no D2H sync)
__global__ void scale_scalar_kernel(float* val, float scale) {
    *val *= scale;
}

inline void scale_scalar_device(float* d_val, float scale, cudaStream_t stream = nullptr) {
    scale_scalar_kernel<<<1, 1, 0, stream>>>(d_val, scale);
}

// Set a scalar value in device memory using cudaMemcpy (eliminates 1-thread kernel launch)
inline void set_scalar_host(float* d_dst, float value, cudaStream_t stream = nullptr) {
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_dst, &value, sizeof(float), cudaMemcpyHostToDevice, stream));
}

// Helper for computing (optionally scaled) sum of a 1D tensor on GPU
// When scale != 1.0f, computes sum * scale (e.g. mean = sum * (1/n))
// No D2H synchronization needed
inline Tensor cuda_sum_device(const Tensor& t, float scale = 1.0f) {
    int64_t n = t.numel();
    const float* data = t.data<float>();

    Tensor result({1}, DType::Float32, t.device());
    float* d_out = result.data<float>();

    // Allocate temp storage for CUB reduction
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temp storage requirements
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, data, d_out, n);
    backend::CachedMemoryGuard temp_guard(temp_storage_bytes);
    d_temp_storage = temp_guard.get();

    // Run sum
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, data, d_out, n);

    // Apply scale if needed
    if (scale != 1.0f) {
        scale_scalar_device(d_out, scale);
    }

    return result;
}

// ==============================================================================
// Fused Linear + ReLU: cuBLAS matmul + fused bias+ReLU kernel
// ==============================================================================

// F-090: accumulator type for the fused bias+ReLU kernel. Float16/BFloat16
// widen to float for the bias-add and ReLU-threshold compare — mirroring
// CPU's widen_narrow_compute, which wraps the ENTIRE matmul+bias-add+ReLU
// pipeline in Float32 for F16/BF16 input and narrows only the final output —
// instead of computing the bias-add directly in half precision (an extra
// half-precision rounding step the CPU reference doesn't have). Float32/
// Float64 are unaffected (accumulator == T, same as before).
template<typename T> struct BiasReluAcc { using type = T; };
template<> struct BiasReluAcc<__half> { using type = float; };
template<> struct BiasReluAcc<__nv_bfloat16> { using type = float; };

/**
 * @brief Fused bias + ReLU kernel: out[i] = max(0, out[i] + bias[i % out_features])
 */
template<typename T>
__global__ void bias_relu_kernel(
    T* output,
    const T* bias,
    int64_t total_elements,
    int64_t out_features
) {
    using Acc = typename BiasReluAcc<T>::type;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        Acc val = static_cast<Acc>(output[i]) + static_cast<Acc>(bias[i % out_features]);
        output[i] = static_cast<T>((val > Acc(0)) ? val : Acc(0));
    }
}

/**
 * @brief ReLU-only kernel (no bias): out[i] = max(0, out[i])
 */
template<typename T>
__global__ void relu_inplace_kernel(
    T* output,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        T val = output[i];
        output[i] = (val > T(0)) ? val : T(0);
    }
}

/**
 * @brief Fused linear + ReLU host function
 *
 * Uses cuBLAS GemmEx for the matmul (uses Tensor Cores when available),
 * followed by a lightweight fused bias+ReLU kernel.
 */
auto fused_linear_relu_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias
) -> Tensor {
    // Flatten input to 2D: [batch_size, in_features]
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
    Tensor output(output_shape, input.dtype(), input.device());

    // Get device and acquire a stream from the pool
    int32_t device_id = input.device().index;
    auto stream_guard = cuda::CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();
    auto handle = CuBLASHandlePool::get(stream);

    // cuBLAS uses column-major, so we compute: output^T = weight * input^T
    // which in row-major is: output = input @ weight^T
    int M = static_cast<int>(out_features);   // rows of weight
    int N = static_cast<int>(batch_size);      // cols of input^T
    int K = static_cast<int>(in_features);     // shared dim

    if (input.dtype() == DType::Float32) {
        float alpha = 1.0f, beta_val = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    M, N, K,
                    &alpha,
                    weight.data<float>(), K,      // weight: [out_features, in_features] row-major = [K, M] col-major
                    input.data<float>(), K,       // input: [batch_size, in_features] row-major = [K, N] col-major
                    &beta_val,
                    output.data<float>(), M);     // output: [batch_size, out_features] row-major = [M, N] col-major
    } else if (input.dtype() == DType::Float64) {
        double alpha = 1.0, beta_val = 0.0;
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    M, N, K,
                    &alpha,
                    weight.data<double>(), K,
                    input.data<double>(), K,
                    &beta_val,
                    output.data<double>(), M);
    } else if (input.dtype() == DType::Float16) {
        __half alpha = __float2half(1.0f), beta_val = __float2half(0.0f);
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    M, N, K,
                    &alpha,
                    reinterpret_cast<const __half*>(weight.data_ptr()), K,
                    reinterpret_cast<const __half*>(input.data_ptr()), K,
                    &beta_val,
                    reinterpret_cast<__half*>(output.data_ptr()), M);
    } else if (input.dtype() == DType::BFloat16) {
        // Use GemmEx with BFloat16 input and Float32 compute
        float alpha = 1.0f, beta_val = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     M, N, K,
                     &alpha,
                     weight.data_ptr(), CUDA_R_16BF, K,
                     input.data_ptr(), CUDA_R_16BF, K,
                     &beta_val,
                     output.data_ptr(), CUDA_R_16BF, M,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else {
        throw std::runtime_error("fused_linear_relu_cuda: Unsupported dtype");
    }

    // Launch fused bias+ReLU or ReLU-only kernel
    int64_t total_elements = batch_size * out_features;
    int block_size = 256;
    int blocks = clamp_blocks((total_elements + block_size - 1) / block_size);

    if (input.dtype() == DType::Float32) {
        if (bias) {
            bias_relu_kernel<<<blocks, block_size, 0, stream>>>(
                output.data<float>(), bias->data<float>(), total_elements, out_features);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else {
            relu_inplace_kernel<<<blocks, block_size, 0, stream>>>(
                output.data<float>(), total_elements);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }
    } else if (input.dtype() == DType::Float64) {
        if (bias) {
            bias_relu_kernel<<<blocks, block_size, 0, stream>>>(
                output.data<double>(), bias->data<double>(), total_elements, out_features);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else {
            relu_inplace_kernel<<<blocks, block_size, 0, stream>>>(
                output.data<double>(), total_elements);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }
    } else if (input.dtype() == DType::Float16) {
        if (bias) {
            bias_relu_kernel<<<blocks, block_size, 0, stream>>>(
                reinterpret_cast<__half*>(output.data_ptr()),
                reinterpret_cast<const __half*>(bias->data_ptr()),
                total_elements, out_features);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else {
            relu_inplace_kernel<<<blocks, block_size, 0, stream>>>(
                reinterpret_cast<__half*>(output.data_ptr()), total_elements);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }
    } else if (input.dtype() == DType::BFloat16) {
        if (bias) {
            bias_relu_kernel<<<blocks, block_size, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()),
                total_elements, out_features);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else {
            relu_inplace_kernel<<<blocks, block_size, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), total_elements);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return output;
}

// ==============================================================================
// Fused BatchNorm + ReLU CUDA Kernel
// ==============================================================================

// Device helper for the inverse standard deviation. Uses a correctly-rounded
// 1/sqrt (reciprocal OF a division of sqrt) rather than the rsqrt/hrsqrt
// approximation intrinsics, so the eager fused-norm result bit-matches the CPU
// and JIT reference (both use 1/sqrt); the rsqrt intrinsic diverges by ~1-2 ULP
// (JIT-F051). Half/bf16 widen to float, compute in float, then narrow.
template<typename T>
__device__ __forceinline__ T device_rsqrt(T x) {
    return static_cast<T>(1.0f / sqrtf(static_cast<float>(x)));
}

template<>
__device__ __forceinline__ float device_rsqrt<float>(float x) {
    return 1.0f / sqrtf(x);
}

template<>
__device__ __forceinline__ double device_rsqrt<double>(double x) {
    return 1.0 / sqrt(x);
}

template<>
__device__ __forceinline__ __half device_rsqrt<__half>(__half x) {
    return __float2half(1.0f / sqrtf(__half2float(x)));
}

template<>
__device__ __forceinline__ __nv_bfloat16 device_rsqrt<__nv_bfloat16>(__nv_bfloat16 x) {
    return __float2bfloat16(1.0f / sqrtf(__bfloat162float(x)));
}

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
        int64_t c = (idx / spatial_size) % num_features;

        T normalized = (input[idx] - mean[c]) * device_rsqrt(var[c] + eps);
        T scaled = normalized * gamma[c] + beta[c];

        // ReLU
        output[idx] = (scaled > T(0)) ? scaled : T(0);
    }
}

auto fused_batchnorm_relu_cuda(
    const Tensor& input_orig,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    // Contiguify: the kernel indexes input flat as NCHW, so a non-contiguous
    // (channels-last / permuted) view would map each element to the wrong
    // channel. Mirrors the CPU kernel's guard; the F16/BF16 path contiguifies
    // via .to(Float32).
    Tensor input_contig;
    const Tensor& input = input_orig.is_contiguous()
        ? input_orig : (input_contig = input_orig.contiguous());
    int64_t batch_size = input.shape()[0];
    int64_t num_features = input.shape()[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < input.shape().size(); ++i) {
        spatial_size *= input.shape()[i];
    }

    // F16/BF16 host-level widen-narrow: normalization stats (var+eps, rsqrt)
    // must be computed in Float32 to match the CPU reference, which computes
    // inv_std and the normalization in float and narrows only on the final
    // store. Computing (var+eps) and rsqrt in half (eps=1e-5) loses precision.
    // Mirrors fused_layer_norm_cuda's widen-narrow for F16/BF16.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto in_f32 = input.to(DType::Float32);
        auto mean_f32 = running_mean.to(DType::Float32);
        auto var_f32 = running_var.to(DType::Float32);
        auto wt_f32 = weight.to(DType::Float32);
        auto bs_f32 = bias.to(DType::Float32);
        auto out_f32 = fused_batchnorm_relu_cuda(
            in_f32, mean_f32, var_f32, wt_f32, bs_f32, eps);
        return out_f32.to(orig);
    }

    int32_t device_id = input.device().index;
    auto stream_guard = cuda::CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

    Tensor output = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device(), stream);

    int64_t total_elements = input.numel();
    int min_grid_size, block_size;

    if (input.dtype() == DType::Float32) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           fused_batchnorm_relu_kernel<float>, 0, 0);
        int blocks = clamp_blocks((total_elements + block_size - 1) / block_size);
        fused_batchnorm_relu_kernel<<<blocks, block_size, 0, stream>>>(
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
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           fused_batchnorm_relu_kernel<double>, 0, 0);
        int blocks = clamp_blocks((total_elements + block_size - 1) / block_size);
        fused_batchnorm_relu_kernel<<<blocks, block_size, 0, stream>>>(
            input.data<double>(),
            running_mean.data<double>(),
            running_var.data<double>(),
            weight.data<double>(),
            bias.data<double>(),
            output.data<double>(),
            batch_size,
            num_features,
            spatial_size,
            static_cast<double>(eps)
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        // F16/BF16 are handled by the host widen-narrow path above.
        throw std::runtime_error("fused_batchnorm_relu_cuda: Unsupported dtype");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return output;
}

// ==============================================================================
// Fused Softmax + CrossEntropy CUDA Kernel
// ==============================================================================

// Wave E1: Acc = F32 for F16/BF16, T otherwise. Per-element in-register
// promotion avoids overflow in sum_exp accumulation; no tensor-wide widen.
template<typename T> struct fsce_acc_type { using type = T; };
template<> struct fsce_acc_type<__half>          { using type = float; };
template<> struct fsce_acc_type<__nv_bfloat16>   { using type = float; };

template<typename T, int BLOCK_SIZE>
__global__ void fused_softmax_cross_entropy_kernel(
    const T* logits,
    const int64_t* targets,
    T* losses,
    int64_t batch_size,
    int64_t num_classes,
    int* error_flag
) {
    using Acc = typename fsce_acc_type<T>::type;
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* row = logits + b * num_classes;
    int64_t target = targets[b];

    // Validate target range. The CPU reference throws on any out-of-range
    // target (no ignore_index semantics), so we must NOT silently emit a zero
    // loss here — that would diverge cross-backend and skew the mean. Signal
    // the violation via the device error flag (checked on the host) and skip
    // the OOB device read of row[target].
    if (target < 0 || target >= num_classes) {
        if (threadIdx.x == 0) {
            atomicExch(error_flag, 1);
            losses[b] = static_cast<T>(0);
        }
        return;
    }

    // Shared memory accumulator in Acc (F32 for F16/BF16) so block-wide
    // reductions don't lose precision or overflow.
    __shared__ Acc shared_data[BLOCK_SIZE];

    Acc max_val = -INFINITY;
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        max_val = fmax(max_val, static_cast<Acc>(row[i]));
    }

    shared_data[threadIdx.x] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = fmax(shared_data[threadIdx.x], shared_data[threadIdx.x + s]);
        }
        __syncthreads();
    }

    Acc global_max = shared_data[0];
    __syncthreads();

    Acc sum_exp = 0;
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        sum_exp += exp(static_cast<Acc>(row[i]) - global_max);
    }

    shared_data[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        Acc log_sum_exp = log(shared_data[0]) + global_max;
        Acc loss = log_sum_exp - static_cast<Acc>(row[target]);
        losses[b] = static_cast<T>(loss);
    }
}

auto fused_softmax_cross_entropy_cuda(
    const Tensor& logits,
    const Tensor& targets,
    const std::string& reduction
) -> Tensor {
    // Per docs/internals/attention-contract.md (FusedSoftmaxCrossEntropy):
    // CPU supports F32/F64; CUDA gains F64 here via widen-narrow at host level
    // for the F16/BF16 paths and direct F64 dispatch via template instantiation.
    // Audit L3 — was previously F32-only, silently downcast caller F64 inputs.
    int32_t device_id = logits.device().index;
    auto stream_guard = cuda::CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();
    if (logits.dtype() == DType::Float64) {
        // F64 dispatch: instantiate the kernel template with double. The
        // reduction below uses tenzor::dispatch(OpId::Sum) instead of
        // cuda_sum_device since the latter is float-only (CUB Sum<float>).
        int64_t batch_size = logits.shape()[0];
        int64_t num_classes = logits.shape()[1];
        Tensor losses = create_cuda_zeros({batch_size}, DType::Float64, logits.device());
        constexpr int BLOCK_SIZE = 256;
        int blocks = batch_size;
        // Device-side out-of-range target flag (CPU reference throws on OOB).
        CudaBuffer error_buf(sizeof(int));
        TENZOR_CUDA_CHECK(cudaMemsetAsync(error_buf.as<int>(), 0, sizeof(int), stream));
        fused_softmax_cross_entropy_kernel<double, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            logits.data<double>(),
            targets.data<int64_t>(),
            losses.data<double>(),
            batch_size,
            num_classes,
            error_buf.as<int>()
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        {
            int host_error = 0;
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(&host_error, error_buf.as<int>(), sizeof(int),
                                         cudaMemcpyDeviceToHost, stream));
            TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));
            if (host_error) {
                throw std::runtime_error(
                    "fused_softmax_cross_entropy: target index out of range [0, " +
                    std::to_string(num_classes) + ")");
            }
        }
        if (reduction == "none") return losses;
        // Reduction via dispatch — works for any dtype the Sum kernel supports.
        // Avoid tenzor::mul/add (math.hpp) here because including math.hpp into
        // this CUDA TU collides with kernel-local `eps`/`sqrt` symbols in the
        // optimizer-fused code below — use OpId-based dispatch instead.
        NewOpAttributes sum_attrs;
        std::vector<Tensor> sum_inputs = {losses};
        Tensor total = tenzor::dispatch(OpId::Sum, sum_inputs, sum_attrs)[0];
        if (reduction == "mean") {
            Tensor scale_t = tenzor::full({1}, 1.0 / static_cast<double>(batch_size),
                                           DType::Float64, losses.device());
            std::vector<Tensor> mul_inputs = {total, scale_t};
            return tenzor::dispatch(OpId::Mul, mul_inputs)[0];
        }
        return total;  // "sum"
    }
    // Wave E1: native F16/BF16 dispatch — kernel template uses F32 accumulator
    // for these element types so no tensor-wide widen is needed.
    int64_t batch_size = logits.shape()[0];
    int64_t num_classes = logits.shape()[1];

    Tensor losses = create_cuda_zeros({batch_size}, logits.dtype(), logits.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    // Device-side out-of-range target flag (CPU reference throws on OOB; we
    // must match that contract rather than silently zeroing the loss).
    CudaBuffer error_buf(sizeof(int));
    TENZOR_CUDA_CHECK(cudaMemsetAsync(error_buf.as<int>(), 0, sizeof(int), stream));

    if (logits.dtype() == DType::Float32) {
        fused_softmax_cross_entropy_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            logits.data<float>(),
            targets.data<int64_t>(),
            losses.data<float>(),
            batch_size,
            num_classes,
            error_buf.as<int>()
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (logits.dtype() == DType::Float16) {
        fused_softmax_cross_entropy_kernel<__half, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(logits.data<Float16>()),
            targets.data<int64_t>(),
            reinterpret_cast<__half*>(losses.data<Float16>()),
            batch_size,
            num_classes,
            error_buf.as<int>()
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (logits.dtype() == DType::BFloat16) {
        fused_softmax_cross_entropy_kernel<__nv_bfloat16, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data<BFloat16>()),
            targets.data<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(losses.data<BFloat16>()),
            batch_size,
            num_classes,
            error_buf.as<int>()
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_softmax_cross_entropy_cuda: Unsupported dtype "
                                 "(F32/F64/F16/BF16 only)");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    {
        int host_error = 0;
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(&host_error, error_buf.as<int>(), sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));
        TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (host_error) {
            throw std::runtime_error(
                "fused_softmax_cross_entropy: target index out of range [0, " +
                std::to_string(num_classes) + ")");
        }
    }

    // Apply reduction — F32 uses cuda_sum_device fast path; F16/BF16 take
    // the dispatch-based reduction (cuda_sum_device internally requires F32).
    if (reduction == "none") {
        return losses;
    }
    if (logits.dtype() == DType::Float32) {
        if (reduction == "mean") {
            return cuda_sum_device(losses, 1.0f / batch_size);
        }
        return cuda_sum_device(losses);  // "sum"
    }
    // F16/BF16 dispatch-based reduction.
    NewOpAttributes sum_attrs;
    std::vector<Tensor> sum_inputs = {losses};
    Tensor total = tenzor::dispatch(OpId::Sum, sum_inputs, sum_attrs)[0];
    if (reduction == "mean") {
        Tensor scale_t = tenzor::full({1},
                                       1.0 / static_cast<double>(batch_size),
                                       losses.dtype(), losses.device());
        std::vector<Tensor> mul_inputs = {total, scale_t};
        return tenzor::dispatch(OpId::Mul, mul_inputs)[0];
    }
    return total;  // "sum"
}

// ==============================================================================
// Fused Add + ReLU CUDA Kernel
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

auto fused_add_relu_cuda(const Tensor& a_orig, const Tensor& b_orig) -> Tensor {
    // Contiguify both operands: the kernel reads a[i]+b[i] flat, so two views
    // with differing physical layouts (e.g. one transposed) would be paired
    // element-for-element incorrectly. Mirrors the CPU kernel's guard.
    Tensor a_contig, b_contig;
    const Tensor& a = a_orig.is_contiguous() ? a_orig : (a_contig = a_orig.contiguous());
    const Tensor& b = b_orig.is_contiguous() ? b_orig : (b_contig = b_orig.contiguous());
    int32_t device_id = a.device().index;
    auto stream_guard = cuda::CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

    Tensor result = create_cuda_zeros(to_vector(a.shape()), a.dtype(), a.device(), stream);

    int64_t n = a.numel();
    int min_grid_size, block_size;

    if (a.dtype() == DType::Float32) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           fused_add_relu_kernel<float>, 0, 0);
        int blocks = clamp_blocks((n + block_size - 1) / block_size);
        fused_add_relu_kernel<<<blocks, block_size, 0, stream>>>(
            a.data<float>(),
            b.data<float>(),
            result.data<float>(),
            n
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (a.dtype() == DType::Float64) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           fused_add_relu_kernel<double>, 0, 0);
        int blocks = clamp_blocks((n + block_size - 1) / block_size);
        fused_add_relu_kernel<<<blocks, block_size, 0, stream>>>(
            a.data<double>(),
            b.data<double>(),
            result.data<double>(),
            n
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (a.dtype() == DType::Float16) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           fused_add_relu_kernel<__half>, 0, 0);
        int blocks = clamp_blocks((n + block_size - 1) / block_size);
        fused_add_relu_kernel<<<blocks, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data_ptr()),
            reinterpret_cast<const __half*>(b.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            n
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (a.dtype() == DType::BFloat16) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           fused_add_relu_kernel<__nv_bfloat16>, 0, 0);
        int blocks = clamp_blocks((n + block_size - 1) / block_size);
        fused_add_relu_kernel<<<blocks, block_size, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()),
            n
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_add_relu_cuda: Unsupported dtype");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return result;
}

// ==============================================================================
// Fused GELU CUDA Kernel
// ==============================================================================

// Device helper for tanh that works with different types
template<typename T>
__device__ __forceinline__ T device_tanh(T x) {
    return tanhf(static_cast<float>(x));
}

template<>
__device__ __forceinline__ float device_tanh<float>(float x) {
    return tanhf(x);
}

template<>
__device__ __forceinline__ double device_tanh<double>(double x) {
    return tanh(x);
}

template<>
__device__ __forceinline__ __half device_tanh<__half>(__half x) {
    // Convert to float, compute tanh, convert back
    return __float2half(tanhf(__half2float(x)));
}

template<>
__device__ __forceinline__ __nv_bfloat16 device_tanh<__nv_bfloat16>(__nv_bfloat16 x) {
    // Convert to float, compute tanh, convert back
    return __float2bfloat16(tanhf(__bfloat162float(x)));
}

template<typename T>
__global__ void fused_gelu_kernel(
    const T* input,
    T* output,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        // Exact GELU (erf form) to match the CPU / oneAPI / Vulkan kernels and
        // PyTorch's default. The tanh approximation used previously differed by
        // ~5e-4, breaking cross-backend parity.
        float x_f = static_cast<float>(input[i]);
        constexpr float inv_sqrt2 = 0.70710678118654752440f;
        float result = 0.5f * x_f * (1.0f + erff(x_f * inv_sqrt2));
        output[i] = static_cast<T>(result);
    }
}

// Specialized GELU kernel for float (optimized)
template<>
__global__ void fused_gelu_kernel<float>(
    const float* input,
    float* output,
    int64_t n
) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;  // exact GELU (erf form)

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * inv_sqrt2));
    }
}

// Specialized GELU kernel for double (optimized)
template<>
__global__ void fused_gelu_kernel<double>(
    const double* input,
    double* output,
    int64_t n
) {
    constexpr double inv_sqrt2 = 0.70710678118654752440;  // exact GELU (erf form)

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        double x = input[i];
        output[i] = 0.5 * x * (1.0 + erf(x * inv_sqrt2));
    }
}

auto fused_gelu_cuda(const Tensor& input_orig) -> Tensor {
    // Contiguify: the kernel reads/writes input flat, so a non-contiguous view
    // would read scrambled positions (matches the CPU kernel's guard).
    Tensor input_contig;
    const Tensor& input = input_orig.is_contiguous()
        ? input_orig : (input_contig = input_orig.contiguous());
    Tensor output = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());

    int64_t n = input.numel();
    int min_grid_size, block_size;
    int32_t device_id = input.device().index;
    auto stream_guard = cuda::CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

    if (input.dtype() == DType::Float32) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           fused_gelu_kernel<float>, 0, 0);
        int blocks = clamp_blocks((n + block_size - 1) / block_size);
        fused_gelu_kernel<<<blocks, block_size, 0, stream>>>(
            input.data<float>(),
            output.data<float>(),
            n
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           fused_gelu_kernel<double>, 0, 0);
        int blocks = clamp_blocks((n + block_size - 1) / block_size);
        fused_gelu_kernel<<<blocks, block_size, 0, stream>>>(
            input.data<double>(),
            output.data<double>(),
            n
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           fused_gelu_kernel<__half>, 0, 0);
        int blocks = clamp_blocks((n + block_size - 1) / block_size);
        fused_gelu_kernel<<<blocks, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            n
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           fused_gelu_kernel<__nv_bfloat16>, 0, 0);
        int blocks = clamp_blocks((n + block_size - 1) / block_size);
        fused_gelu_kernel<<<blocks, block_size, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            n
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_gelu_cuda: Unsupported dtype");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return output;
}

// ==============================================================================
// Fused Layer Norm CUDA Kernel
// ==============================================================================

template<typename T, int BLOCK_SIZE>
__global__ void fused_layer_norm_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    T* mean_out,        // Output: saved mean for backward
    T* inv_std_out,     // Output: saved inv_std for backward
    int64_t batch_size,
    int64_t norm_size,
    T eps
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_in = input + b * norm_size;
    T* batch_out = output + b * norm_size;

    // Accumulate mean/variance stats in double to avoid catastrophic
    // cancellation, matching the CPU reference (fused_ln_sum_f64 /
    // fused_ln_sumsq_f64 always accumulate F32 LayerNorm stats in double).
    // The F16/BF16 paths widen to Float32 at host level and thus instantiate
    // this kernel with T=float, so a double accumulator here covers them too;
    // the T=double instantiation is also exact. Only the final mean/inv_std
    // and the normalization are narrowed back to T.
    using Acc = double;
    __shared__ Acc shared_data[BLOCK_SIZE];

    // Compute mean
    Acc sum = 0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        sum += static_cast<Acc>(batch_in[i]);
    }

    shared_data[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    Acc mean = shared_data[0] / static_cast<Acc>(norm_size);
    __syncthreads();

    // Compute variance
    Acc var_sum = 0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        Acc diff = static_cast<Acc>(batch_in[i]) - mean;
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

    Acc variance = shared_data[0] / static_cast<Acc>(norm_size);
    // 1/sqrt (not the rsqrt approximation) to match the CPU/JIT reference and
    // the sibling kernels (JIT-F060).
    Acc inv_std = static_cast<Acc>(1) / sqrt(variance + static_cast<Acc>(eps));

    // Save mean and inv_std for backward pass (narrow to T)
    if (threadIdx.x == 0) {
        mean_out[b] = static_cast<T>(mean);
        inv_std_out[b] = static_cast<T>(inv_std);
    }

    // Normalize and scale
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        Acc normalized = (static_cast<Acc>(batch_in[i]) - mean) * inv_std;
        batch_out[i] = static_cast<T>(normalized * static_cast<Acc>(weight[i])
                                      + static_cast<Acc>(bias[i]));
    }
}

/**
 * @brief Host wrapper for LayerNorm forward CUDA kernel
 *
 * @param input Input tensor
 * @param normalized_shape Shape of normalized dimensions
 * @param weight Weight (gamma) parameter
 * @param bias Bias (beta) parameter
 * @param eps Epsilon for numerical stability
 * @return Tuple of (output, mean, inv_std) where mean and inv_std are saved for backward
 */
auto fused_layer_norm_cuda(
    const Tensor& input_orig,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Contiguify: the kernel indexes input flat (input + b*norm_size), so a
    // non-contiguous view (e.g. LayerNorm over the last dim of a transposed
    // activation) would read the wrong storage and corrupt the saved mean/rstd.
    // Mirrors the CPU kernel; the F16/BF16 path contiguifies via .to(Float32).
    Tensor input_contig;
    const Tensor& input = input_orig.is_contiguous()
        ? input_orig : (input_contig = input_orig.contiguous());
    // Per docs/internals/attention-contract.md: mean/inv_std must be Float32
    // for FP16/BF16 inputs (rstd dynamic range exceeds FP16 max=65504 when
    // var ~ 1e-11). Mirrors fused_rms_norm_cuda's widen-narrow at line 1241
    // (audit H1 for CUDA LayerNorm).
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto in_f32 = input.to(DType::Float32);
        auto wt_f32 = weight.to(DType::Float32);
        auto bs_f32 = bias.to(DType::Float32);
        auto [out_f32, mean_f32, inv_std_f32] = fused_layer_norm_cuda(
            in_f32, normalized_shape, wt_f32, bs_f32, eps, stream);
        return std::make_tuple(out_f32.to(orig), mean_f32, inv_std_f32);
    }

    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    Tensor output = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());
    Tensor mean = create_cuda_zeros({batch_size}, input.dtype(), input.device());
    Tensor inv_std = create_cuda_zeros({batch_size}, input.dtype(), input.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        fused_layer_norm_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(),
            weight.data<float>(),
            bias.data<float>(),
            output.data<float>(),
            mean.data<float>(),
            inv_std.data<float>(),
            batch_size,
            norm_size,
            eps
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        fused_layer_norm_kernel<double, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(),
            weight.data<double>(),
            bias.data<double>(),
            output.data<double>(),
            mean.data<double>(),
            inv_std.data<double>(),
            batch_size,
            norm_size,
            static_cast<double>(eps)
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        fused_layer_norm_kernel<__half, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            reinterpret_cast<const __half*>(bias.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<__half*>(mean.data_ptr()),
            reinterpret_cast<__half*>(inv_std.data_ptr()),
            batch_size,
            norm_size,
            __float2half(eps)
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        fused_layer_norm_kernel<__nv_bfloat16, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(mean.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(inv_std.data_ptr()),
            batch_size,
            norm_size,
            __float2bfloat16(eps)
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_layer_norm_cuda: Unsupported dtype");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
    // NOTE: Removed cudaDeviceSynchronize() - async execution is critical for performance

    return std::make_tuple(output, mean, inv_std);
}

// ==============================================================================
// Fused Layer Norm Backward CUDA Kernel
// ==============================================================================

/**
 * @brief CUDA kernel for LayerNorm backward pass
 *
 * Computes gradients for input, weight, and bias given output gradients.
 * Uses efficient parallel reduction for batch-wise operations.
 */
template<typename T, int BLOCK_SIZE>
__global__ void fused_layer_norm_backward_kernel(
    const T* grad_output,    // Gradient from next layer
    const T* input,          // Original input from forward pass
    const T* weight,         // Weight (gamma) from forward pass
    const T* mean,           // Saved mean from forward pass
    const T* inv_std,        // Saved 1/sqrt(var + eps) from forward pass
    T* grad_input,           // Output: gradient w.r.t. input
    double* grad_weight_accum, // Output: double-precision scratch accumulator for
                                // grad_weight (F-085); narrowed to T by
                                // narrow_layer_norm_grad_accum_kernel once every
                                // block has contributed.
    double* grad_bias_accum,   // Output: double-precision scratch accumulator for
                                // grad_bias (F-085); see grad_weight_accum above.
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

    __shared__ double shared_sum1[BLOCK_SIZE];  // For sum(grad_out * weight)
    __shared__ double shared_sum2[BLOCK_SIZE];  // For sum(grad_out * weight * normalized)

    // Compute sums needed for input gradient
    double sum_grad_out = 0.0;  // accumulate in double (forward uses double)
    double sum_grad_out_normalized = 0.0;

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        T grad_out_weighted = batch_grad_out[i] * weight[i];

        sum_grad_out += static_cast<double>(grad_out_weighted);
        sum_grad_out_normalized += static_cast<double>(grad_out_weighted) * static_cast<double>(normalized);

        // F-085: accumulate weight/bias gradients atomically in double
        // precision across every block (batch element), matching CPU's
        // double accumulation in fused_layer_norm_backward_impl. Narrowing to
        // T happens exactly once, in narrow_layer_norm_grad_accum_kernel,
        // after all blocks have contributed — this avoids the float32
        // accumulator precision loss that atomicAdd'ing directly into a
        // T-typed buffer would incur for large batch sizes. Native double
        // atomicAdd requires only compute capability >= 6.0, well below this
        // build's minimum target architecture (see CMAKE_CUDA_ARCHITECTURES
        // in src/backends/cuda/CMakeLists.txt, minimum 70).
        atomicAdd(&grad_weight_accum[i],
                  static_cast<double>(batch_grad_out[i]) * static_cast<double>(normalized));
        atomicAdd(&grad_bias_accum[i], static_cast<double>(batch_grad_out[i]));
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

    double mean_grad_out = shared_sum1[0] / static_cast<double>(norm_size);
    double mean_grad_out_normalized = shared_sum2[0] / static_cast<double>(norm_size);

    // Compute input gradients
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        T grad_out_weighted = batch_grad_out[i] * weight[i];

        batch_grad_in[i] = static_cast<T>((static_cast<double>(grad_out_weighted) - mean_grad_out -
                           static_cast<double>(normalized) * mean_grad_out_normalized) * static_cast<double>(batch_inv_std));
    }
}

// F-085: narrows the double-precision grad_weight_accum/grad_bias_accum
// scratch buffers (populated via atomicAdd across every block in the kernel
// above) down to T exactly once, after all blocks have finished contributing
// their per-feature partial sums — mirroring CPU's accumulate-in-double,
// narrow-to-T-once approach in fused_layer_norm_backward_impl.
template<typename T>
__global__ void narrow_layer_norm_grad_accum_kernel(
    const double* grad_weight_accum,
    const double* grad_bias_accum,
    T* grad_weight,
    T* grad_bias,
    int64_t norm_size
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < norm_size) {
        grad_weight[i] = static_cast<T>(grad_weight_accum[i]);
        grad_bias[i] = static_cast<T>(grad_bias_accum[i]);
    }
}

/**
 * @brief Host wrapper for LayerNorm backward CUDA kernel
 *
 * @param grad_output Gradient from next layer
 * @param input Original input from forward pass
 * @param weight Weight (gamma) parameter
 * @param mean Saved mean from forward pass
 * @param inv_std Saved inverse standard deviation from forward pass
 * @param normalized_shape Shape of normalized dimensions
 * @return Tuple of (grad_input, grad_weight, grad_bias)
 */
auto fused_layer_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& mean,
    const Tensor& inv_std,
    const std::vector<int64_t>& normalized_shape,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    // F16/BF16: accumulate in Float32. A half-precision accumulator corrupts
    // grad_weight/grad_bias and diverges from the CPU (double/Float32) reference,
    // so widen, run the Float32 path, then narrow back — mirroring the forward
    // (fused_layer_norm_cuda) and the RMSNorm backward. Because of this
    // unconditional early return, dtype is guaranteed to be Float32 or
    // Float64 for the rest of this function (F-087: this used to be followed
    // by unreachable __half/__nv_bfloat16 kernel-launch branches, which have
    // been removed).
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto [gi, gw, gb] = fused_layer_norm_backward_cuda(
            grad_output.to(DType::Float32), input.to(DType::Float32),
            weight.to(DType::Float32), mean.to(DType::Float32),
            inv_std.to(DType::Float32), normalized_shape, stream);
        return {gi.to(orig), gw.to(orig), gb.to(orig)};
    }

    // Allocate output tensors
    Tensor grad_input = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());
    Tensor grad_weight = create_cuda_zeros({norm_size}, input.dtype(), input.device());
    Tensor grad_bias = create_cuda_zeros({norm_size}, input.dtype(), input.device());
    // F-085: double-precision scratch accumulators, atomicAdd'd into by every
    // block/batch element in fused_layer_norm_backward_kernel, then narrowed
    // to input.dtype() in a single pass (narrow_layer_norm_grad_accum_kernel)
    // once the main kernel completes.
    Tensor grad_weight_accum = create_cuda_zeros({norm_size}, DType::Float64, input.device());
    Tensor grad_bias_accum = create_cuda_zeros({norm_size}, DType::Float64, input.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;
    int narrow_blocks = static_cast<int>((norm_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (input.dtype() == DType::Float32) {
        fused_layer_norm_backward_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<float>(),
            input.data<float>(),
            weight.data<float>(),
            mean.data<float>(),
            inv_std.data<float>(),
            grad_input.data<float>(),
            grad_weight_accum.data<double>(),
            grad_bias_accum.data<double>(),
            batch_size,
            norm_size
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        narrow_layer_norm_grad_accum_kernel<float><<<narrow_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_weight_accum.data<double>(), grad_bias_accum.data<double>(),
            grad_weight.data<float>(), grad_bias.data<float>(), norm_size
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        fused_layer_norm_backward_kernel<double, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(),
            input.data<double>(),
            weight.data<double>(),
            mean.data<double>(),
            inv_std.data<double>(),
            grad_input.data<double>(),
            grad_weight_accum.data<double>(),
            grad_bias_accum.data<double>(),
            batch_size,
            norm_size
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        narrow_layer_norm_grad_accum_kernel<double><<<narrow_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_weight_accum.data<double>(), grad_bias_accum.data<double>(),
            grad_weight.data<double>(), grad_bias.data<double>(), norm_size
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_layer_norm_backward_cuda: Unsupported dtype");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
    // NOTE: Removed cudaDeviceSynchronize() - async execution is critical for performance

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ==============================================================================
// Fused RMSNorm CUDA Kernel
// ==============================================================================

/**
 * @brief Optimized RMSNorm forward kernel using warp-level primitives
 *
 * RMSNorm: output = x * weight / sqrt(mean(x^2) + eps)
 * Uses warp shuffle for fast reduction and vectorized memory access.
 */
// Wave E2: Acc = F32 for F16/BF16, T otherwise. RRMS dynamic range can
// exceed F16's max (65504) when var ~ 1e-11, so rrms_out is stored as Acc
// (always F32 for half-types) — same pattern as LayerNorm's mean/inv_std.
template<typename T> struct rms_acc_type { using type = T; };
template<> struct rms_acc_type<__half>        { using type = float; };
template<> struct rms_acc_type<__nv_bfloat16> { using type = float; };

template<typename T, int BLOCK_SIZE>
__global__ void fused_rms_norm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    typename rms_acc_type<T>::type* __restrict__ rrms_out,
    int64_t batch_size,
    int64_t norm_size,
    typename rms_acc_type<T>::type eps
) {
    using Acc = typename rms_acc_type<T>::type;
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_in = input + b * norm_size;
    T* batch_out = output + b * norm_size;

    // Phase 22-followup #37 fix preserved: float4 vectorization is gated on
    // T=float; F16/BF16 take the scalar widen-on-load path so sum_sq stays
    // in Acc (F32) — sums of squares of half-precision values can otherwise
    // overflow F16's max (65504) for moderate norm_size.
    // Accumulate sum-of-squares in double (matching LayerNorm); rrms is still
    // stored as Acc so the forward/backward storage contract is unchanged.
    double sum_sq = 0.0;
    if constexpr (std::is_same_v<T, float>) {
        // float4 requires every row offset (b*norm_size) to be 16-byte aligned,
        // which holds only when norm_size % 4 == 0; otherwise a misaligned float4
        // load is UB. Fall back to the scalar path for unaligned norm_size.
        if (norm_size % 4 == 0) {
            const int vec_size = 4;
            int64_t vec_norm_size = norm_size / vec_size;
            const float4* batch_in_vec = reinterpret_cast<const float4*>(batch_in);
            for (int64_t i = threadIdx.x; i < vec_norm_size; i += blockDim.x) {
                float4 v = batch_in_vec[i];
                sum_sq += static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y + static_cast<double>(v.z) * v.z + static_cast<double>(v.w) * v.w;
            }
        } else {
            for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
                double val = static_cast<double>(batch_in[i]);
                sum_sq += val * val;
            }
        }
    } else {
        for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
            double val = static_cast<double>(batch_in[i]);
            sum_sq += val * val;
        }
    }

    // Warp-level reduction using shuffle (Acc-typed)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ double warp_sums[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < (BLOCK_SIZE / 32)) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ Acc shared_rrms;
    if (threadIdx.x == 0) {
        double mean_sq = sum_sq / static_cast<double>(norm_size);
        // 1/sqrt in double (not the rsqrt approximation) — matches CPU/JIT (JIT-F060).
        shared_rrms = static_cast<Acc>(1.0 / sqrt(mean_sq + static_cast<double>(eps)));
        rrms_out[b] = shared_rrms;
    }
    __syncthreads();
    Acc rrms = shared_rrms;

    if constexpr (std::is_same_v<T, float>) {
        // float4 only when norm_size % 4 == 0 (every row offset 16-byte aligned);
        // otherwise scalar to avoid a misaligned float4 store/load.
        if (norm_size % 4 == 0) {
            const int vec_size = 4;
            int64_t vec_norm_size = norm_size / vec_size;
            float4* batch_out_vec = reinterpret_cast<float4*>(batch_out);
            const float4* batch_in_vec = reinterpret_cast<const float4*>(batch_in);
            const float4* weight_vec = reinterpret_cast<const float4*>(weight);
            for (int64_t i = threadIdx.x; i < vec_norm_size; i += blockDim.x) {
                float4 v = batch_in_vec[i];
                float4 w = weight_vec[i];
                float4 out;
                out.x = v.x * rrms * w.x;
                out.y = v.y * rrms * w.y;
                out.z = v.z * rrms * w.z;
                out.w = v.w * rrms * w.w;
                batch_out_vec[i] = out;
            }
        } else {
            for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
                batch_out[i] = batch_in[i] * static_cast<T>(rrms) * weight[i];
            }
        }
    } else {
        for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
            Acc x = static_cast<Acc>(batch_in[i]);
            Acc w = static_cast<Acc>(weight[i]);
            batch_out[i] = static_cast<T>(x * rrms * w);
        }
    }
}

/**
 * @brief Fused RMSNorm forward pass
 *
 * @param input Input tensor [batch_size, ..., norm_size]
 * @param weight Weight tensor [norm_size]
 * @param eps Epsilon for numerical stability
 * @return Tuple of (output, rrms) where rrms is saved for backward
 */
auto fused_rms_norm_cuda(
    const Tensor& input_orig,
    const Tensor& weight,
    float eps,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor> {
    // Contiguify: the kernel indexes input flat (input + b*norm_size, and
    // float4-vectorizes the row), so a non-contiguous residual view would read
    // the wrong storage and corrupt the saved rrms. Mirrors the CPU kernel.
    Tensor input_contig;
    const Tensor& input = input_orig.is_contiguous()
        ? input_orig : (input_contig = input_orig.contiguous());
    // Wave E2: native F16/BF16 dispatch via Acc=F32 inside the kernel template.
    // RRMS tensor is allocated as F32 for half-precision inputs (rstd dynamic
    // range exceeds F16 max=65504 when var ~ 1e-11) — same pattern as
    // fused_layer_norm_cuda's mean/inv_std.
    auto shape = input.shape();
    int64_t norm_size = shape.back();

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }

    Tensor output = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());
    // rrms stays F32 for F16/BF16 inputs (per LayerNorm precedent).
    DType rrms_dtype = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16)
                            ? DType::Float32
                            : input.dtype();
    Tensor rrms = create_cuda_zeros({batch_size}, rrms_dtype, input.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        fused_rms_norm_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(),
            weight.data<float>(),
            output.data<float>(),
            rrms.data<float>(),
            batch_size,
            norm_size,
            eps
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        fused_rms_norm_kernel<double, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(),
            weight.data<double>(),
            output.data<double>(),
            rrms.data<double>(),
            batch_size,
            norm_size,
            static_cast<double>(eps)
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        fused_rms_norm_kernel<__half, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<const __half*>(weight.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            rrms.data<float>(),  // F32 storage for half-type
            batch_size,
            norm_size,
            eps
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        fused_rms_norm_kernel<__nv_bfloat16, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            rrms.data<float>(),  // F32 storage for half-type
            batch_size,
            norm_size,
            eps
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_rms_norm_cuda: dtype not supported (F32/F64/F16/BF16 only)");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return std::make_tuple(output, rrms);
}

/**
 * @brief Fused RMSNorm backward kernel
 *
 * Computes gradients for input and weight.
 * grad_input = weight * rrms * (grad_out - x * rrms^2 * mean(grad_out * x * weight))
 */
template<typename T, int BLOCK_SIZE>
__global__ void fused_rms_norm_backward_kernel(
    const T* grad_output,    // Gradient from next layer
    const T* input,          // Original input from forward pass
    const T* weight,         // Weight from forward pass
    const typename rms_acc_type<T>::type* rrms,  // F32 for half-types (per forward contract)
    T* grad_input,           // Output: gradient w.r.t. input (T storage)
    typename rms_acc_type<T>::type* grad_weight,  // F32 for half-types (atomicAdd-safe)
    int64_t batch_size,
    int64_t norm_size
) {
    // H1 fix: accumulator type is Acc=float for F16/BF16 (per
    // rms_acc_type<T> traits, same as forward). All intermediate reductions
    // and shared-mem use Acc; loads widen-on-load, stores narrow-on-store.
    using Acc = typename rms_acc_type<T>::type;
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_grad_out = grad_output + b * norm_size;
    const T* batch_in = input + b * norm_size;
    T* batch_grad_in = grad_input + b * norm_size;

    Acc batch_rrms = static_cast<Acc>(rrms[b]);

    __shared__ Acc shared_sum[BLOCK_SIZE];

    // Compute sum(grad_out * x * weight) / norm_size for input gradient
    Acc sum_grad_x_w = Acc{0};
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        sum_grad_x_w += static_cast<Acc>(batch_grad_out[i])
                      * static_cast<Acc>(batch_in[i])
                      * static_cast<Acc>(weight[i]);
    }

    shared_sum[threadIdx.x] = sum_grad_x_w;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    Acc mean_grad_x_w = shared_sum[0] / static_cast<Acc>(norm_size);

    // Compute input gradient and accumulate weight gradient
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        Acc x_i = static_cast<Acc>(batch_in[i]);
        Acc w_i = static_cast<Acc>(weight[i]);
        Acc grad_out_i = static_cast<Acc>(batch_grad_out[i]);

        // grad_input = rrms * (grad_out * weight - x * rrms^2 * mean_grad_x_w)
        Acc gi = batch_rrms * (grad_out_i * w_i - x_i * batch_rrms * batch_rrms * mean_grad_x_w);
        batch_grad_in[i] = static_cast<T>(gi);

        // grad_weight accumulation (atomic on Acc storage — atomicAdd<float>
        // is well-defined on all SMs; atomicAdd<__half> requires SM 70+
        // and has different precision semantics).
        atomicAdd(&grad_weight[i], grad_out_i * x_i * batch_rrms);
    }
}

/**
 * @brief Fused RMSNorm backward pass
 */
auto fused_rms_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& rrms,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t norm_size = shape.back();

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }

    Tensor grad_input = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());
    // H1 fix: grad_weight follows the same dtype convention as forward's
    // rrms — F32 storage for half-precision inputs (atomicAdd<float> is
    // safe on all SMs; atomicAdd<__half> is SM-70+ and has different
    // precision). For F32/F64, grad_weight matches input dtype.
    DType gw_dtype = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16)
                     ? DType::Float32 : input.dtype();
    Tensor grad_weight = create_cuda_zeros({norm_size}, gw_dtype, input.device());

    // The half-precision kernels keep rrms (the per-row reciprocal-RMS scalar)
    // in F32 for accuracy. The autograd layer, however, narrows every saved
    // tensor — rrms included — to the input dtype before dispatch, so for
    // F16/BF16 inputs rrms arrives as half here. Widen it back to F32 so the
    // F16/BF16 branches' `const float* rrms` reads the right storage instead of
    // throwing a Float16 type mismatch (and silently zeroing the gradient).
    Tensor rrms_f32 = (rrms.dtype() == DType::Float32) ? rrms
                      : rrms.to(DType::Float32).contiguous();

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        fused_rms_norm_backward_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<float>(),
            input.data<float>(),
            weight.data<float>(),
            rrms.data<float>(),
            grad_input.data<float>(),
            grad_weight.data<float>(),
            batch_size,
            norm_size
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        fused_rms_norm_backward_kernel<double, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            grad_output.data<double>(),
            input.data<double>(),
            weight.data<double>(),
            rrms.data<double>(),
            grad_input.data<double>(),
            grad_weight.data<double>(),
            batch_size,
            norm_size
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        // H1 fix: native F16 dispatch. Acc=float per rms_acc_type<__half>.
        // Per forward contract, rrms is F32 and grad_weight is F32.
        // grad_output / input / weight / grad_input are __half.
        fused_rms_norm_backward_kernel<__half, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            rrms_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data_ptr()),
            grad_weight.data<float>(),
            batch_size,
            norm_size
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        // H1 fix: native BF16 dispatch, mirrors F16 path.
        fused_rms_norm_backward_kernel<__nv_bfloat16, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            rrms_f32.data<float>(),
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()),
            grad_weight.data<float>(),
            batch_size,
            norm_size
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_rms_norm_backward_cuda: dtype not supported (need Float32/Float64/Float16/BFloat16)");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return std::make_tuple(grad_input, grad_weight);
}

// ==============================================================================
// Fused Conv2D + BatchNorm + ReLU CUDA Kernel
// ==============================================================================

/**
 * @brief Fused Conv2D + BatchNorm + ReLU kernel
 *
 * Combines convolution, batch normalization, and ReLU into single kernel.
 * Highly optimized for training and inference pipelines.
 */
template<typename T>
__global__ void fused_conv2d_bn_relu_kernel(
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
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    T eps,
    bool has_bias,
    int64_t groups
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * out_channels * out_h * out_w;
    int64_t stride_loop = blockDim.x * gridDim.x;

    // F061: groups>1 restricts each output channel to its own input-channel
    // window; `weight` is laid out [C_out, C_in/groups, kH, kW] (standard
    // grouped-conv layout — same as plain Conv2d), so the weight index uses
    // in_channels_per_group as the per-output-channel stride, not the full
    // in_channels.
    int64_t out_channels_per_group = out_channels / groups;
    int64_t in_channels_per_group = in_channels / groups;

    for (int64_t idx = tid; idx < total_elements; idx += stride_loop) {
        // Decode output position
        int64_t w_out = idx % out_w;
        int64_t h_out = (idx / out_w) % out_h;
        int64_t c_out = (idx / (out_w * out_h)) % out_channels;
        int64_t n = idx / (out_w * out_h * out_channels);
        int64_t g = c_out / out_channels_per_group;
        int64_t in_start = g * in_channels_per_group;

        // Compute convolution
        T conv_sum = 0;
        for (int64_t c_in_local = 0; c_in_local < in_channels_per_group; ++c_in_local) {
            int64_t c_in = in_start + c_in_local;
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int64_t w_in = w_out * stride_w - padding_w + kw * dilation_w;

                    if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        int64_t input_idx = ((n * in_channels + c_in) * in_h + h_in) * in_w + w_in;
                        int64_t weight_idx = ((c_out * in_channels_per_group + c_in_local) * kernel_h + kh) * kernel_w + kw;
                        conv_sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias if present
        if (has_bias) {
            conv_sum += bias[c_out];
        }

        // Apply batch normalization (type-generic rsqrt so the Float64 instantiation
        // keeps double precision; sqrt has float/double device overloads).
        T normalized = (conv_sum - bn_mean[c_out]) / sqrt(bn_var[c_out] + eps);
        T bn_out = normalized * bn_gamma[c_out] + bn_beta[c_out];

        // Apply ReLU
        output[idx] = (bn_out > T(0)) ? bn_out : T(0);
    }
}

template <typename T>
static void launch_fused_conv2d_bn_relu(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    const Tensor& bn_mean, const Tensor& bn_var, const Tensor& bn_gamma, const Tensor& bn_beta,
    Tensor& output,
    int64_t batch_size, int64_t in_channels, int64_t out_channels,
    int64_t in_h, int64_t in_w, int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w,
    float eps, cudaStream_t stream, int64_t groups)
{
    int64_t total_elements = batch_size * out_channels * out_h * out_w;
    int min_grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       fused_conv2d_bn_relu_kernel<T>, 0, 0);
    int blocks = clamp_blocks((total_elements + block_size - 1) / block_size);
    const T* bias_ptr = bias ? bias->data<T>() : nullptr;
    fused_conv2d_bn_relu_kernel<T><<<blocks, block_size, 0, stream>>>(
        input.data<T>(), weight.data<T>(), bias_ptr,
        bn_mean.data<T>(), bn_var.data<T>(), bn_gamma.data<T>(), bn_beta.data<T>(),
        output.data<T>(),
        batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, static_cast<T>(eps), bias != nullptr, groups);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

// --- Training path (findings F050): compute per-channel batch statistics, update
// running_mean/running_var in place (momentum), normalize with the BATCH stats,
// then scale/shift + ReLU — matching the CPU conv_bn_relu_training semantics
// exactly (biased var for normalization, unbiased var for the running update,
// double-precision reductions).

// Raw conv (+bias), dilation-aware, into conv_out [N, C_out, out_h, out_w].
template <typename T>
__global__ void fused_conv2d_raw_kernel(
    const T* input, const T* weight, const T* bias, T* conv_out,
    int64_t batch_size, int64_t in_channels, int64_t out_channels,
    int64_t in_h, int64_t in_w, int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w, bool has_bias, int64_t groups)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * out_channels * out_h * out_w;
    int64_t step = blockDim.x * gridDim.x;
    // F061: same grouped-conv channel-offset scheme as fused_conv2d_bn_relu_kernel.
    int64_t out_channels_per_group = out_channels / groups;
    int64_t in_channels_per_group = in_channels / groups;
    for (int64_t idx = tid; idx < total; idx += step) {
        int64_t w_out = idx % out_w;
        int64_t h_out = (idx / out_w) % out_h;
        int64_t c_out = (idx / (out_w * out_h)) % out_channels;
        int64_t n = idx / (out_w * out_h * out_channels);
        int64_t g = c_out / out_channels_per_group;
        int64_t in_start = g * in_channels_per_group;
        T s = 0;
        for (int64_t c_in_local = 0; c_in_local < in_channels_per_group; ++c_in_local) {
            int64_t c_in = in_start + c_in_local;
            for (int64_t kh = 0; kh < kernel_h; ++kh)
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride_h - padding_h + kh * dilation_h;
                    int64_t w_in = w_out * stride_w - padding_w + kw * dilation_w;
                    if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        int64_t ii = ((n * in_channels + c_in) * in_h + h_in) * in_w + w_in;
                        int64_t wi = ((c_out * in_channels_per_group + c_in_local) * kernel_h + kh) * kernel_w + kw;
                        s += input[ii] * weight[wi];
                    }
                }
        }
        if (has_bias) s += bias[c_out];
        conv_out[idx] = s;
    }
}

// One block per output channel: reduce mean and biased variance in double,
// write batch_mean/batch_var (biased) and update running stats in place.
template <typename T>
__global__ void fused_bn_channel_stats_kernel(
    const T* conv_out, T* batch_mean, T* batch_var,
    T* running_mean, T* running_var,
    int64_t batch_size, int64_t out_channels, int64_t spatial, float momentum)
{
    int64_t oc = blockIdx.x;
    if (oc >= out_channels) return;
    int64_t samples = batch_size * spatial;
    extern __shared__ double sdata[];
    double* ssum = sdata;
    double* ssq = sdata + blockDim.x;

    double local_sum = 0.0, local_sq = 0.0;
    for (int64_t b = 0; b < batch_size; ++b) {
        const T* base = conv_out + (b * out_channels + oc) * spatial;
        for (int64_t s = threadIdx.x; s < spatial; s += blockDim.x) {
            double v = static_cast<double>(base[s]);
            local_sum += v;
            local_sq += v * v;
        }
    }
    ssum[threadIdx.x] = local_sum;
    ssq[threadIdx.x] = local_sq;
    __syncthreads();
    for (int64_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
            ssq[threadIdx.x] += ssq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double n_d = static_cast<double>(samples);
        double mean_d = ssum[0] / n_d;
        double var_biased = ssq[0] / n_d - mean_d * mean_d;
        if (var_biased < 0.0) var_biased = 0.0;  // guard fp rounding
        batch_mean[oc] = static_cast<T>(mean_d);
        batch_var[oc] = static_cast<T>(var_biased);
        double var_unbiased = (samples > 1) ? var_biased * n_d / (n_d - 1.0) : var_biased;
        running_mean[oc] = static_cast<T>((1.0 - momentum) * static_cast<double>(running_mean[oc])
                                          + momentum * mean_d);
        running_var[oc] = static_cast<T>((1.0 - momentum) * static_cast<double>(running_var[oc])
                                         + momentum * var_unbiased);
    }
}

// Normalize conv_out with batch stats, scale/shift, ReLU into output.
template <typename T>
__global__ void fused_bn_relu_apply_kernel(
    const T* conv_out, const T* batch_mean, const T* batch_var,
    const T* bn_gamma, const T* bn_beta, T* output,
    int64_t total, int64_t out_channels, int64_t spatial, T eps)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = blockDim.x * gridDim.x;
    for (int64_t idx = tid; idx < total; idx += step) {
        int64_t oc = (idx / spatial) % out_channels;
        T inv_std = T(1) / sqrt(batch_var[oc] + eps);
        T norm = (conv_out[idx] - batch_mean[oc]) * inv_std;
        T bn = norm * bn_gamma[oc] + bn_beta[oc];
        output[idx] = (bn > T(0)) ? bn : T(0);
    }
}

template <typename T>
static void launch_fused_conv2d_bn_relu_training(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    Tensor& running_mean, Tensor& running_var, const Tensor& bn_gamma, const Tensor& bn_beta,
    Tensor& output,
    int64_t batch_size, int64_t in_channels, int64_t out_channels,
    int64_t in_h, int64_t in_w, int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w,
    float momentum, float eps, cudaStream_t stream, int64_t groups)
{
    int64_t spatial = out_h * out_w;
    int64_t total = batch_size * out_channels * spatial;

    Tensor conv_out = create_cuda_zeros({batch_size, out_channels, out_h, out_w},
                                        input.dtype(), input.device());
    Tensor batch_mean = create_cuda_zeros({out_channels}, input.dtype(), input.device());
    Tensor batch_var = create_cuda_zeros({out_channels}, input.dtype(), input.device());

    int block = 256;
    int blocks = clamp_blocks((total + block - 1) / block);
    const T* bias_ptr = bias ? bias->data<T>() : nullptr;
    fused_conv2d_raw_kernel<T><<<blocks, block, 0, stream>>>(
        input.data<T>(), weight.data<T>(), bias_ptr, conv_out.data<T>(),
        batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, bias != nullptr, groups);
    TENZOR_CUDA_POST_LAUNCH_CHECK();

    int stat_block = 256;
    size_t shmem = 2 * static_cast<size_t>(stat_block) * sizeof(double);
    fused_bn_channel_stats_kernel<T><<<static_cast<unsigned>(out_channels), stat_block, shmem, stream>>>(
        conv_out.data<T>(), batch_mean.data<T>(), batch_var.data<T>(),
        running_mean.data<T>(), running_var.data<T>(),
        batch_size, out_channels, spatial, momentum);
    TENZOR_CUDA_POST_LAUNCH_CHECK();

    fused_bn_relu_apply_kernel<T><<<blocks, block, 0, stream>>>(
        conv_out.data<T>(), batch_mean.data<T>(), batch_var.data<T>(),
        bn_gamma.data<T>(), bn_beta.data<T>(), output.data<T>(),
        total, out_channels, spatial, static_cast<T>(eps));
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

auto fused_conv2d_bn_relu_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    const Tensor& bn_mean,
    const Tensor& bn_var,
    const Tensor& bn_gamma,
    const Tensor& bn_beta,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    float momentum,
    float eps,
    bool training,
    int64_t groups
) -> Tensor {
    // F061: groups previously had no parameter at all on this kernel — it
    // silently behaved as groups=1 regardless of the source conv's config.
    if (groups <= 0) {
        throw std::invalid_argument(
            "fused_conv2d_bn_relu_cuda: groups must be positive (got " +
            std::to_string(groups) + ")");
    }

    // Float16/BFloat16: widen to Float32 on device, compute, narrow back (on-GPU
    // casts). Mirrors the CPU kernel: the running-stat update in training mode
    // lands on the Float32 copies (as on CPU), so neither backend writes the
    // half-precision running tensors — parity preserved.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        Tensor bias_f32;
        const Tensor* bias_ptr = nullptr;
        if (bias) { bias_f32 = bias->to(DType::Float32); bias_ptr = &bias_f32; }
        Tensor out = fused_conv2d_bn_relu_cuda(
            input.to(DType::Float32), weight.to(DType::Float32), bias_ptr,
            bn_mean.to(DType::Float32), bn_var.to(DType::Float32),
            bn_gamma.to(DType::Float32), bn_beta.to(DType::Float32),
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
            momentum, eps, training, groups);
        return out.to(orig);
    }

    int64_t batch_size = input.shape()[0];
    int64_t in_channels = input.shape()[1];
    int64_t in_h = input.shape()[2];
    int64_t in_w = input.shape()[3];

    int64_t out_channels = weight.shape()[0];
    int64_t kernel_h = weight.shape()[2];
    int64_t kernel_w = weight.shape()[3];

    int64_t out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    Tensor output = create_cuda_zeros({batch_size, out_channels, out_h, out_w}, input.dtype(), input.device());
    int32_t device_id = input.device().index;
    auto stream_guard = cuda::CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

    if (training) {
        // running_mean/running_var are updated in place; take non-const handles
        // that share storage with the passed tensors (mirrors the CPU kernel).
        Tensor rm = bn_mean;
        Tensor rv = bn_var;
        if (input.dtype() == DType::Float32) {
            launch_fused_conv2d_bn_relu_training<float>(input, weight, bias, rm, rv, bn_gamma, bn_beta,
                output, batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w,
                kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w, momentum, eps, stream, groups);
        } else if (input.dtype() == DType::Float64) {
            launch_fused_conv2d_bn_relu_training<double>(input, weight, bias, rm, rv, bn_gamma, bn_beta,
                output, batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w,
                kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w, momentum, eps, stream, groups);
        } else {
            throw std::runtime_error("fused_conv2d_bn_relu_cuda: unsupported dtype");
        }
    } else if (input.dtype() == DType::Float32) {
        launch_fused_conv2d_bn_relu<float>(input, weight, bias, bn_mean, bn_var, bn_gamma, bn_beta,
            output, batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w,
            kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, eps, stream, groups);
    } else if (input.dtype() == DType::Float64) {
        launch_fused_conv2d_bn_relu<double>(input, weight, bias, bn_mean, bn_var, bn_gamma, bn_beta,
            output, batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w,
            kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, eps, stream, groups);
    } else {
        throw std::runtime_error("fused_conv2d_bn_relu_cuda: unsupported dtype");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return output;
}

// ==============================================================================
// F-088: fused_matmul_add_cuda / fused_elementwise_chain_cuda removed.
//
// These were fully-implemented, Float32-only kernels with no OpId in
// include/tenzor/ops/op_id.hpp and no call site in cuda_kernel_registry.cpp —
// dead code. Verified this is not a CUDA-only gap: OneAPI only forward-
// declares an equivalent fused_matmul_add_kernel in oneapi_kernel_registry.cpp
// (src/backends/oneapi/oneapi_kernel_registry.cpp:688) but never registers it
// under any OpId either, and has no fused_elementwise_chain at all. ROCm
// implements both fused_matmul_add_hip and fused_elementwise_chain_hip
// (src/backends/rocm/kernels/fused_ops.hip.cpp) but rocm_kernel_registry.cpp
// never references either symbol — equally dead there. The only other
// reference in the whole tree is tests/benchmarks/bench_fusion.cpp, which is
// not wired into any CMakeLists.txt (not built) and calls a free function
// `fused_elementwise_chain` that isn't declared in any public header —
// itself dead/orphaned. Since neither op is part of any documented public
// API or reachable on any backend, removal (rather than inventing a new
// OpId nothing else uses) is the root-cause fix; this comment intentionally
// stays put as a signpost.
// ==============================================================================

// ==============================================================================
// Philox4x32-10 Counter-Based PRNG (CUDA device port of CPU Philox4x32 in
// src/backends/cpu/kernels/flash_attention.cpp). Same algorithm and constants
// so dropout results within a single backend's forward/backward pair match
// bit-exactly. Cross-backend reproducibility is not part of the contract.
// ==============================================================================

__device__ __forceinline__ void philox_round(uint32_t ctr[4], const uint32_t key[2]) {
    constexpr uint64_t M0 = 0xD2511F53ULL;
    constexpr uint64_t M1 = 0xCD9E8D57ULL;
    uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
    uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);
    uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
    uint32_t lo0 = static_cast<uint32_t>(prod0);
    uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
    uint32_t lo1 = static_cast<uint32_t>(prod1);
    uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t new1 = lo1;
    uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t new3 = lo0;
    ctr[0] = new0; ctr[1] = new1; ctr[2] = new2; ctr[3] = new3;
}

__device__ __forceinline__ float philox_uniform(uint32_t batch_head, uint32_t query_idx,
                                                 uint32_t kv_pos, uint32_t rng_seed) {
    // 10-round Philox4x32 with counter (batch_head, query_idx, kv_pos, 0) and
    // key (rng_seed, rng_seed^0x1BD11BDA). Returns a uniform float in [0, 1).
    uint32_t ctr[4] = {batch_head, query_idx, kv_pos, 0};
    uint32_t k[2] = {rng_seed, rng_seed ^ 0x1BD11BDAU};
    constexpr uint32_t W0 = 0x9E3779B9U;
    constexpr uint32_t W1 = 0xBB67AE85U;
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        philox_round(ctr, k);
        if (r < 9) { k[0] += W0; k[1] += W1; }
    }
    // Convert to [0, 1) via the 24-bit mantissa trick (same as CPU).
    return (static_cast<float>(ctr[0] >> 8)) * (1.0f / 16777216.0f);
}

// ==============================================================================
// Flash Attention CUDA Kernel - Optimized with Warp-Level Parallelism
// ==============================================================================

/**
 * @brief Warp-level max reduction using shuffle
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * @brief Warp-level sum reduction using shuffle
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Optimized Flash Attention V2 kernel
 *
 * This kernel implements the Flash Attention algorithm with online softmax,
 * achieving O(N) memory complexity instead of O(N^2).
 *
 * Key optimizations:
 * 1. Each block processes one query row with 256 threads
 * 2. KV tiles loaded cooperatively into shared memory
 * 3. Online softmax with running max/sum for numerical stability
 * 4. Warp shuffle for fast reductions
 * 5. All threads participate in output accumulation
 *
 * Supports head_dim: 32, 64, 80, 96, 128
 */
template<int HEAD_DIM, int BLOCK_SIZE = 256>
__global__ void flash_attention_v2_kernel(
    const float* __restrict__ Q,     // [batch_heads, seq_len_q, head_dim]
    const float* __restrict__ K,     // [batch_heads, seq_len_k, head_dim]
    const float* __restrict__ V,     // [batch_heads, seq_len_k, head_dim]
    float* __restrict__ O,           // [batch_heads, seq_len_q, head_dim]
    float* __restrict__ L,           // [batch_heads, seq_len_q] logsumexp (may be nullptr)
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const bool causal,               // applies upper-triangular mask before softmax
    const float dropout_p,           // dropout probability (0 disables); applied
                                     // post-softmax with inverted scaling 1/(1-p).
    const uint32_t rng_seed          // Philox seed for dropout reproducibility
) {
    // Configuration
    constexpr int Bc = 32;  // Keys per tile - small for register pressure

    // query_idx uses grid.x (limit 2^31-1) and batch_head grid.y (limit 65535)
    // so sequences longer than 65535 launch and compute correctly.
    const int query_idx = blockIdx.x;
    const int batch_head = blockIdx.y;

    if (query_idx >= seq_len_q) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BLOCK_SIZE / 32;

    // Base pointers
    const float* Q_row = Q + batch_head * seq_len_q * HEAD_DIM + query_idx * HEAD_DIM;
    const float* K_base = K + batch_head * seq_len_k * HEAD_DIM;
    const float* V_base = V + batch_head * seq_len_k * HEAD_DIM;
    float* O_row = O + batch_head * seq_len_q * HEAD_DIM + query_idx * HEAD_DIM;

    // Shared memory layout
    constexpr int K_STRIDE = HEAD_DIM + 4;  // Padding for bank conflicts
    extern __shared__ float smem[];
    float* K_tile = smem;                          // [Bc][K_STRIDE]
    float* V_tile = smem + Bc * K_STRIDE;          // [Bc][K_STRIDE]
    float* Q_shared = smem + 2 * Bc * K_STRIDE;    // [HEAD_DIM]
    float* scores_shared = Q_shared + HEAD_DIM;    // [Bc]
    float* reduce_buf = scores_shared + Bc;        // [num_warps]

    // Load Q row into shared memory
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        Q_shared[d] = Q_row[d] * scale;
    }
    __syncthreads();

    // Output accumulator - each thread handles a subset of output dimensions
    float o_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Up to 4 elements per thread

    // Online softmax state (shared across block via reduction)
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    const int num_kv_blocks = (seq_len_k + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int k_start = kv_block * Bc;
        const int actual_Bc = min(Bc, seq_len_k - k_start);

        // Load K/V tile cooperatively
        for (int idx = tid; idx < actual_Bc * HEAD_DIM; idx += BLOCK_SIZE) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            K_tile[row * K_STRIDE + col] = K_base[(k_start + row) * HEAD_DIM + col];
            V_tile[row * K_STRIDE + col] = V_base[(k_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        // Step 1: Compute all Q·K scores and find max. Causal masking applied
        // here so masked positions get score = -INFINITY before the per-tile
        // softmax max-subtract — produces clean exp(-INF) = 0 in step 2 below.
        // Per docs/internals/attention-contract.md sentinel rule.
        float local_max = -INFINITY;

        for (int j = tid; j < actual_Bc; j += BLOCK_SIZE) {
            int kv_pos = k_start + j;
            float score;
            // F021: bottom-right causal alignment (matches the MHA/GQA manual
            // BMM path and PyTorch). A query at absolute position query_idx
            // attends to keys kv_pos <= query_idx + (seq_len_k - seq_len_q).
            // For self-attention the offset is 0 (kv_pos <= query_idx); for
            // KV-cache cross-attention (seq_len_q < seq_len_k) the single query
            // correctly sees all preceding keys instead of only key 0.
            if (causal && kv_pos > query_idx + (seq_len_k - seq_len_q)) {
                score = -INFINITY;
            } else {
                score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_shared[d] * K_tile[j * K_STRIDE + d];
                }
            }
            scores_shared[j] = score;
            local_max = fmaxf(local_max, score);
        }

        // Reduce max across block using warp shuffle + shared memory
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
        }
        if (lane_id == 0) {
            reduce_buf[warp_id] = local_max;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? reduce_buf[lane_id] : -INFINITY;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
            }
            if (lane_id == 0) {
                reduce_buf[0] = val;
            }
        }
        __syncthreads();
        float block_max = reduce_buf[0];

        // All-tile-masked early-out: when every score in this tile is -INF
        // (causal mask kills the whole tile, e.g. q=4 with kv_block at [32,64)),
        // exp(scores - block_max) = exp(-INF - (-INF)) = NaN. Skip the tile —
        // the online-softmax math correctly leaves running state alone in this
        // case (l_prev unchanged, o_local unchanged, m_prev unchanged).
        if (block_max == -INFINITY) {
            __syncthreads();  // pair the next iteration's K/V tile load
            continue;
        }

        // Step 2: Compute exp(score - max) and sum
        float local_sum = 0.0f;

        // Apply Philox dropout post-softmax with inverted scaling 1/(1-p)
        // when dropout_p > 0 and rng_seed != 0. Counter is (batch_head,
        // query_idx, kv_pos, 0); each thread independently draws its own
        // uniform sample so the mask is bit-reproducible from (seed,
        // batch_head, query_idx, kv_pos) for the backward replay.
        const bool apply_dropout = (dropout_p > 0.0f) && (rng_seed != 0u);
        const float dropout_scale = apply_dropout ? (1.0f / (1.0f - dropout_p)) : 1.0f;

        for (int j = tid; j < actual_Bc; j += BLOCK_SIZE) {
            float exp_score = expf(scores_shared[j] - block_max);
            if (apply_dropout) {
                int kv_pos = k_start + j;
                float u = philox_uniform(static_cast<uint32_t>(batch_head),
                                          static_cast<uint32_t>(query_idx),
                                          static_cast<uint32_t>(kv_pos),
                                          rng_seed);
                if (u < dropout_p) {
                    exp_score = 0.0f;
                } else {
                    exp_score *= dropout_scale;
                }
            }
            scores_shared[j] = exp_score;  // Store exp (post-dropout) for P @ V
            local_sum += exp_score;
        }
        __syncthreads();

        // Reduce sum
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        }
        if (lane_id == 0) {
            reduce_buf[warp_id] = local_sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? reduce_buf[lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_xor_sync(0xffffffff, val, offset);
            }
            if (lane_id == 0) {
                reduce_buf[0] = val;
            }
        }
        __syncthreads();
        float block_sum = reduce_buf[0];

        // Step 3: Online softmax rescaling
        float m_new = fmaxf(m_prev, block_max);
        float exp_prev = expf(m_prev - m_new);
        float exp_curr = expf(block_max - m_new);
        float l_new = exp_prev * l_prev + exp_curr * block_sum;

        // Step 4: Rescale previous output and accumulate P @ V
        // Each thread handles HEAD_DIM / BLOCK_SIZE output elements
        for (int i = 0; i < 4; ++i) {
            int d = tid + i * BLOCK_SIZE;
            if (d < HEAD_DIM) {
                // Rescale previous accumulator
                o_local[i] *= exp_prev;

                // Add new contribution: sum over j of P[j] * V[j, d]
                float pv_sum = 0.0f;
                for (int j = 0; j < actual_Bc; ++j) {
                    pv_sum += scores_shared[j] * V_tile[j * K_STRIDE + d];
                }
                o_local[i] += exp_curr * pv_sum;
            }
        }

        m_prev = m_new;
        l_prev = l_new;

        __syncthreads();
    }

    // Final normalization and write output
    float l_inv = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;

    for (int i = 0; i < 4; ++i) {
        int d = tid + i * BLOCK_SIZE;
        if (d < HEAD_DIM) {
            O_row[d] = o_local[i] * l_inv;
        }
    }

    // Save logsumexp for backward pass: LSE = m + log(l)
    if (L != nullptr && tid == 0) {
        float lse = m_prev + logf(fmaxf(l_prev, 1e-10f));
        L[batch_head * seq_len_q + query_idx] = lse;
    }
}

auto fused_attention_cuda(
    const Tensor& Q,     // (batch_heads, seq_len_q, head_dim)
    const Tensor& K,     // (batch_heads, seq_len_k, head_dim)
    const Tensor& V,     // (batch_heads, seq_len_k, head_dim)
    float scale,
    bool causal,         // Honored — flash kernel applies mask inline
    float dropout_p,     // Dropout probability; 0 disables
    uint32_t rng_seed    // Philox seed; 0 disables dropout regardless of dropout_p
) -> std::pair<Tensor, Tensor> {
    int64_t batch_heads = Q.shape()[0];
    int64_t seq_len_q = Q.shape()[1];
    int64_t head_dim = Q.shape()[2];
    int64_t seq_len_k = K.shape()[1];

    // Support FP16 and BF16 by upcasting to FP32 for computation
    if (Q.dtype() == DType::Float16 || Q.dtype() == DType::BFloat16) {
        auto Q_f32 = Q.to(DType::Float32);
        auto K_f32 = K.to(DType::Float32);
        auto V_f32 = V.to(DType::Float32);
        auto [output_f32, lse_f32] = fused_attention_cuda(Q_f32, K_f32, V_f32, scale, causal, dropout_p, rng_seed);
        return {output_f32.to(Q.dtype()), lse_f32};
    }

    if (Q.dtype() != DType::Float32) {
        throw std::runtime_error("fused_attention_cuda: Only Float32, Float16, BFloat16 supported");
    }

    Tensor output = create_cuda_zeros({batch_heads, seq_len_q, head_dim}, Q.dtype(), Q.device());
    Tensor lse = create_cuda_zeros({batch_heads, seq_len_q}, Q.dtype(), Q.device());

    // Optimized Flash Attention V2 - supports multiple head dimensions
    constexpr int BLOCK_SIZE = 256;
    constexpr int Bc = 32;  // Keys per tile

    // Grid: one block per (query_row, batch_head) pair. query_row is the x-dim
    // (grid limit 2^31-1) so seq_len_q > 65535 is supported; batch_head is the
    // y-dim (limit 65535, always ample for batch*heads).
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(seq_len_q, batch_heads);
    // JIT-R115: launching on the implicit legacy/default stream instead of
    // the current stream is illegal ("operation would make the legacy
    // stream depend on a capturing blocking stream") the moment this runs
    // during active CUDA-graph capture, and silently excludes this kernel
    // from the captured graph otherwise -- the same bug class already fixed
    // for activations.cu/reduction.cu's dispatch wrappers. FusedAttention
    // was made tracer-visible specifically so it participates in JIT graph/
    // CUDA-graph capture, so this path is now reachable in exactly that
    // scenario.
    cudaStream_t stream = cuda::cuda_current_stream();

    // Compute shared memory size based on head_dim
    // Layout: K_tile[Bc][HEAD_DIM+4] + V_tile[Bc][HEAD_DIM+4] + Q_shared[HEAD_DIM] + scores[Bc] + reduce[8]
    auto compute_smem_size = [](int hd) {
        int k_stride = hd + 4;
        return (2 * Bc * k_stride + hd + Bc + 8) * sizeof(float);
    };

    float* lse_ptr = lse.data<float>();

    // Dispatch based on head_dim for optimal unrolling
    if (head_dim == 64) {
        size_t smem_size = compute_smem_size(64);
        flash_attention_v2_kernel<64, BLOCK_SIZE><<<blocks, threads, smem_size, stream>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(), output.data<float>(), lse_ptr,
            static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale, causal, dropout_p, rng_seed);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (head_dim == 128) {
        size_t smem_size = compute_smem_size(128);
        flash_attention_v2_kernel<128, BLOCK_SIZE><<<blocks, threads, smem_size, stream>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(), output.data<float>(), lse_ptr,
            static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale, causal, dropout_p, rng_seed);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (head_dim == 32) {
        size_t smem_size = compute_smem_size(32);
        flash_attention_v2_kernel<32, BLOCK_SIZE><<<blocks, threads, smem_size, stream>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(), output.data<float>(), lse_ptr,
            static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale, causal, dropout_p, rng_seed);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (head_dim == 80) {
        size_t smem_size = compute_smem_size(80);
        flash_attention_v2_kernel<80, BLOCK_SIZE><<<blocks, threads, smem_size, stream>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(), output.data<float>(), lse_ptr,
            static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale, causal, dropout_p, rng_seed);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (head_dim == 96) {
        size_t smem_size = compute_smem_size(96);
        flash_attention_v2_kernel<96, BLOCK_SIZE><<<blocks, threads, smem_size, stream>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(), output.data<float>(), lse_ptr,
            static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale, causal, dropout_p, rng_seed);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        // Non-standard head_dim: pad Q/K/V up to a supported specialized HEAD_DIM
        // (zeros in the extra columns are numerically identity for dot products),
        // run the specialized kernel, then slice the output back down.
        //
        // The legacy tiled kernel template-parameterizes HEAD_DIM while smem is
        // sized from runtime head_dim, causing out-of-bounds reads when the two
        // disagree (observed NaNs for head_dim=4). Padding to a specialized kernel
        // avoids that template/runtime mismatch entirely.
        auto pick_padded = [&](int64_t hd) -> int {
            if (hd <= 32)  return 32;
            if (hd <= 64)  return 64;
            if (hd <= 80)  return 80;
            if (hd <= 96)  return 96;
            return 128;
        };
        int padded_hd = pick_padded(head_dim);

        auto pad_last = [&](const Tensor& t) -> Tensor {
            // Pad the last dimension with zeros from head_dim to padded_hd.
            auto padded = create_cuda_zeros(
                {t.shape()[0], t.shape()[1], padded_hd}, t.dtype(), t.device());
            // slice_scatter copies t into padded[..., 0:head_dim] along dim 2.
            return slice_scatter(padded, t, /*dim=*/2, /*start=*/0,
                                 /*end=*/static_cast<int64_t>(head_dim));
        };
        Tensor Qp = pad_last(Q);
        Tensor Kp = pad_last(K);
        Tensor Vp = pad_last(V);

        Tensor padded_output = create_cuda_zeros(
            {batch_heads, seq_len_q, padded_hd}, Q.dtype(), Q.device());

        auto compute_smem_size_gen = [](int hd) {
            int k_stride = hd + 4;
            return (2 * Bc * k_stride + hd + Bc + 8) * sizeof(float);
        };

        if (padded_hd == 32) {
            flash_attention_v2_kernel<32, BLOCK_SIZE><<<blocks, threads, compute_smem_size_gen(32), stream>>>(
                Qp.data<float>(), Kp.data<float>(), Vp.data<float>(), padded_output.data<float>(),
                lse_ptr, static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale, causal, dropout_p, rng_seed);
        } else if (padded_hd == 64) {
            flash_attention_v2_kernel<64, BLOCK_SIZE><<<blocks, threads, compute_smem_size_gen(64), stream>>>(
                Qp.data<float>(), Kp.data<float>(), Vp.data<float>(), padded_output.data<float>(),
                lse_ptr, static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale, causal, dropout_p, rng_seed);
        } else if (padded_hd == 80) {
            flash_attention_v2_kernel<80, BLOCK_SIZE><<<blocks, threads, compute_smem_size_gen(80), stream>>>(
                Qp.data<float>(), Kp.data<float>(), Vp.data<float>(), padded_output.data<float>(),
                lse_ptr, static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale, causal, dropout_p, rng_seed);
        } else if (padded_hd == 96) {
            flash_attention_v2_kernel<96, BLOCK_SIZE><<<blocks, threads, compute_smem_size_gen(96), stream>>>(
                Qp.data<float>(), Kp.data<float>(), Vp.data<float>(), padded_output.data<float>(),
                lse_ptr, static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale, causal, dropout_p, rng_seed);
        } else {
            flash_attention_v2_kernel<128, BLOCK_SIZE><<<blocks, threads, compute_smem_size_gen(128), stream>>>(
                Qp.data<float>(), Kp.data<float>(), Vp.data<float>(), padded_output.data<float>(),
                lse_ptr, static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale, causal, dropout_p, rng_seed);
        }
        TENZOR_CUDA_POST_LAUNCH_CHECK();

        // Slice the padded output back down to head_dim along the last axis.
        output = slice(padded_output, /*dim=*/2, /*start=*/0,
                       /*end=*/static_cast<int64_t>(head_dim)).contiguous();
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    return {output, lse};
}

// ==============================================================================
// Fused Flash Attention Backward CUDA Kernel (Tiled, Memory-Efficient)
// ==============================================================================

/**
 * @brief Tiled Flash Attention backward kernel
 *
 * Recomputes attention scores in tiles using saved logsumexp from the forward pass,
 * avoiding materialization of the full NxN attention matrix.
 *
 * Each thread block processes one KV tile (column block of size Bc) across all Q tiles.
 * dK and dV are accumulated directly in global memory (one block per KV tile, no race).
 * dQ is accumulated via atomicAdd since multiple KV tiles contribute to each Q row.
 *
 * Grid: (num_kv_tiles, batch_heads)
 * Block: (BLOCK_SIZE) threads
 *
 * Shared memory layout (fits in 48KB for HEAD_DIM <= 64; uses extended smem for 128):
 *   K_tile[Bc][HEAD_DIM], V_tile[Bc][HEAD_DIM],
 *   Q_tile[Br][HEAD_DIM], dO_tile[Br][HEAD_DIM],
 *   S_tile[Br][Bc], l_tile[Br], D_tile[Br]
 * dK/dV accumulated in per-thread register arrays, written to global at end.
 */
template<int HEAD_DIM, int Br, int Bc, int BLOCK_SIZE>
__global__ void flash_attention_backward_kernel(
    const float* __restrict__ Q,     // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ K,     // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ V,     // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ O,     // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ dO,    // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ L,     // [batch_heads, seq_len] logsumexp
    float* __restrict__ dQ,          // [batch_heads, seq_len, HEAD_DIM] (atomicAdd)
    float* __restrict__ dK,          // [batch_heads, seq_len, HEAD_DIM] (one block per KV tile)
    float* __restrict__ dV,          // [batch_heads, seq_len, HEAD_DIM] (one block per KV tile)
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const bool causal,
    // Phase P0 / Fix 6: dropout reproduction. When apply_dropout is true,
    // the backward kernel regenerates the same Philox mask the forward
    // used (deterministic given the (batch_head, query_idx, kv_pos)
    // counter triple and the rng_seed). Without this, gradients leak
    // from dropped attention positions.
    const float dropout_p,
    const uint32_t rng_seed
) {
    const int kv_tile_idx = blockIdx.x;  // which KV tile (column block)
    const int batch_head = blockIdx.y;
    const int tid = threadIdx.x;

    // Bottom-right causal alignment: a query at absolute row r attends to
    // keys up to r + (seq_len_k - seq_len_q) (matches the forward kernel).
    const int causal_offset = seq_len_k - seq_len_q;

    const int kv_start = kv_tile_idx * Bc;
    if (kv_start >= seq_len_k) return;
    const int actual_Bc = min(Bc, seq_len_k - kv_start);

    // Base pointers for this batch-head. Q/O/dO/L/dQ stride by seq_len_q;
    // K/V/dK/dV stride by seq_len_k.
    const float* Q_base  = Q  + batch_head * seq_len_q * HEAD_DIM;
    const float* K_base  = K  + batch_head * seq_len_k * HEAD_DIM;
    const float* V_base  = V  + batch_head * seq_len_k * HEAD_DIM;
    const float* O_base  = O  + batch_head * seq_len_q * HEAD_DIM;
    const float* dO_base = dO + batch_head * seq_len_q * HEAD_DIM;
    const float* L_base  = L  + batch_head * seq_len_q;
    float* dQ_base = dQ + batch_head * seq_len_q * HEAD_DIM;
    float* dK_base = dK + batch_head * seq_len_k * HEAD_DIM;
    float* dV_base = dV + batch_head * seq_len_k * HEAD_DIM;

    // Shared memory layout (no dK/dV tiles — those go directly to global)
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

    // Per-thread accumulators for dK and dV for the elements this thread is responsible for.
    // Each thread handles ceil(actual_Bc * HEAD_DIM / BLOCK_SIZE) elements.
    // We use register arrays for accumulation then write once at the end.
    // Max elements per thread: ceil(Bc * HEAD_DIM / BLOCK_SIZE)
    // For Bc=32, HEAD_DIM=128, BLOCK_SIZE=256: 32*128/256 = 16 elements per thread
    constexpr int MAX_ELEMS_PER_THREAD = (Bc * HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float dk_acc[MAX_ELEMS_PER_THREAD];
    float dv_acc[MAX_ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
        dk_acc[e] = 0.0f;
        dv_acc[e] = 0.0f;
    }

    // Iterate over Q tiles (row blocks)
    const int num_q_tiles = (seq_len_q + Br - 1) / Br;

    for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
        const int q_start = q_tile_idx * Br;
        if (q_start >= seq_len_q) break;
        const int actual_Br = min(Br, seq_len_q - q_start);

        // For causal masking: skip if all Q rows come before all K cols
        // (bottom-right aligned: query row r attends keys <= r + causal_offset)
        if (causal && (q_start + actual_Br - 1) + causal_offset < kv_start) {
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
        // Phase P0 / Fix 6: also replay the same dropout mask the forward
        // kernel applied at lines 2027-2040. Counter is (batch_head,
        // query_idx, kv_pos, 0) — identical to forward — so the same
        // (seed, counter) triple deterministically reproduces the mask.
        const bool apply_dropout = (dropout_p > 0.0f) && (rng_seed != 0u);
        const float dropout_scale = apply_dropout ? (1.0f / (1.0f - dropout_p)) : 1.0f;
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            float p = 0.0f;
            if (i < actual_Br && j < actual_Bc) {
                if (causal && (q_start + i) + causal_offset < (kv_start + j)) {
                    p = 0.0f;
                } else {
                    p = expf(S_tile[i * Bc + j] - l_tile[i]);
                    if (apply_dropout) {
                        const uint32_t query_idx = static_cast<uint32_t>(q_start + i);
                        const uint32_t kv_pos    = static_cast<uint32_t>(kv_start + j);
                        const float u = philox_uniform(
                            static_cast<uint32_t>(batch_head),
                            query_idx, kv_pos, rng_seed);
                        if (u < dropout_p) {
                            p = 0.0f;
                        } else {
                            p *= dropout_scale;
                        }
                    }
                }
            }
            S_tile[i * Bc + j] = p;  // Reuse S_tile for (dropout-masked) P_ij
        }
        __syncthreads();

        // Accumulate dV_j += P_ij^T @ dO_i  [Bc x HEAD_DIM]
        // Each thread accumulates for its assigned (j, d) elements
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
        // dK = scale * sum_i dS^T @ Q  (since S = Q @ K^T * scale)
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

// ==============================================================================
// Mixed-Precision Flash Attention Backward CUDA Kernel (FP16 / BF16)
// ==============================================================================

/**
 * @brief Converts a low-precision value to float for accumulation.
 */
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_float(float x) { return x; }

__device__ __forceinline__ __half from_float_to(__half /*tag*/, float x) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float_to(__nv_bfloat16 /*tag*/, float x) { return __float2bfloat16(x); }

/**
 * @brief Mixed-precision flash attention backward kernel.
 *
 * Reads Q, K, V, O, dO in type T (FP16 or BF16), performs all accumulation
 * in FP32 shared memory / registers, writes dQ/dK/dV back in type T.
 * L (logsumexp) is always FP32 (saved from forward pass).
 *
 * Same tiling strategy as the FP32 kernel: each block owns one KV tile,
 * iterates over Q tiles.  dQ via atomicAdd (FP32 scratch), dK/dV in registers.
 */
template<typename T, int HEAD_DIM, int Br, int Bc, int BLOCK_SIZE>
__global__ void flash_attention_backward_kernel_mp(
    const T* __restrict__ Q,     // [batch_heads, seq_len, HEAD_DIM]
    const T* __restrict__ K,     // [batch_heads, seq_len, HEAD_DIM]
    const T* __restrict__ V,     // [batch_heads, seq_len, HEAD_DIM]
    const T* __restrict__ O,     // [batch_heads, seq_len, HEAD_DIM]
    const T* __restrict__ dO,    // [batch_heads, seq_len, HEAD_DIM]
    const float* __restrict__ L, // [batch_heads, seq_len] logsumexp (always FP32)
    float* __restrict__ dQ_f32,  // [batch_heads, seq_len, HEAD_DIM] FP32 scratch for atomicAdd
    T* __restrict__ dK,          // [batch_heads, seq_len, HEAD_DIM]
    T* __restrict__ dV,          // [batch_heads, seq_len, HEAD_DIM]
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const bool causal,
    // Phase P0 / Fix 6: dropout reproduction. See FP32 backward kernel
    // for the rationale; mask application is identical (post-softmax,
    // pre-dV/dS).
    const float dropout_p,
    const uint32_t rng_seed
) {
    const int kv_tile_idx = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int tid = threadIdx.x;

    // Bottom-right causal alignment: query row r attends keys up to
    // r + (seq_len_k - seq_len_q) (matches the forward kernel).
    const int causal_offset = seq_len_k - seq_len_q;

    const int kv_start = kv_tile_idx * Bc;
    if (kv_start >= seq_len_k) return;
    const int actual_Bc = min(Bc, seq_len_k - kv_start);

    // Base pointers. Q/O/dO/L/dQ stride by seq_len_q; K/V/dK/dV by seq_len_k.
    const T* Q_base      = Q  + batch_head * seq_len_q * HEAD_DIM;
    const T* K_base      = K  + batch_head * seq_len_k * HEAD_DIM;
    const T* V_base      = V  + batch_head * seq_len_k * HEAD_DIM;
    const T* O_base      = O  + batch_head * seq_len_q * HEAD_DIM;
    const T* dO_base     = dO + batch_head * seq_len_q * HEAD_DIM;
    const float* L_base  = L  + batch_head * seq_len_q;
    float* dQ_base       = dQ_f32 + batch_head * seq_len_q * HEAD_DIM;
    T* dK_base           = dK + batch_head * seq_len_k * HEAD_DIM;
    T* dV_base           = dV + batch_head * seq_len_k * HEAD_DIM;

    // All shared memory tiles in FP32 for numerical stability
    extern __shared__ float smem[];
    float* K_tile  = smem;
    float* V_tile  = K_tile  + Bc * HEAD_DIM;
    float* Q_tile  = V_tile  + Bc * HEAD_DIM;
    float* dO_tile = Q_tile  + Br * HEAD_DIM;
    float* S_tile  = dO_tile + Br * HEAD_DIM;
    float* l_tile  = S_tile  + Br * Bc;
    float* D_tile  = l_tile  + Br;

    // Load K_j and V_j tiles: convert T -> float
    for (int i = tid; i < actual_Bc * HEAD_DIM; i += BLOCK_SIZE) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        K_tile[row * HEAD_DIM + col] = to_float(K_base[(kv_start + row) * HEAD_DIM + col]);
        V_tile[row * HEAD_DIM + col] = to_float(V_base[(kv_start + row) * HEAD_DIM + col]);
    }
    for (int i = tid + actual_Bc * HEAD_DIM; i < Bc * HEAD_DIM; i += BLOCK_SIZE) {
        K_tile[i] = 0.0f;
        V_tile[i] = 0.0f;
    }
    __syncthreads();

    // Per-thread register accumulators for dK and dV (FP32)
    constexpr int MAX_ELEMS_PER_THREAD = (Bc * HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float dk_acc[MAX_ELEMS_PER_THREAD];
    float dv_acc[MAX_ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
        dk_acc[e] = 0.0f;
        dv_acc[e] = 0.0f;
    }

    const int num_q_tiles = (seq_len_q + Br - 1) / Br;

    for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
        const int q_start = q_tile_idx * Br;
        if (q_start >= seq_len_q) break;
        const int actual_Br = min(Br, seq_len_q - q_start);

        if (causal && (q_start + actual_Br - 1) + causal_offset < kv_start) {
            continue;
        }

        // Load Q_i and dO_i: convert T -> float
        for (int i = tid; i < actual_Br * HEAD_DIM; i += BLOCK_SIZE) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            Q_tile[row * HEAD_DIM + col]  = to_float(Q_base[(q_start + row) * HEAD_DIM + col]);
            dO_tile[row * HEAD_DIM + col] = to_float(dO_base[(q_start + row) * HEAD_DIM + col]);
        }
        for (int i = tid + actual_Br * HEAD_DIM; i < Br * HEAD_DIM; i += BLOCK_SIZE) {
            Q_tile[i] = 0.0f;
            dO_tile[i] = 0.0f;
        }

        // Load l_i (already FP32) and compute D_i = rowsum(dO_i * O_i) in FP32
        for (int row = tid; row < actual_Br; row += BLOCK_SIZE) {
            l_tile[row] = L_base[q_start + row];

            float d_sum = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                d_sum += to_float(dO_base[(q_start + row) * HEAD_DIM + d])
                       * to_float(O_base[(q_start + row) * HEAD_DIM + d]);
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
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            if (i >= actual_Br || j >= actual_Bc) {
                S_tile[idx] = -INFINITY;
            }
        }
        __syncthreads();

        // Compute P_ij = exp(S_ij - l_i), causal mask, dropout-mask replay.
        // Phase P0 / Fix 6 — same logic as the FP32 backward.
        const bool apply_dropout = (dropout_p > 0.0f) && (rng_seed != 0u);
        const float dropout_scale = apply_dropout ? (1.0f / (1.0f - dropout_p)) : 1.0f;
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            float p = 0.0f;
            if (i < actual_Br && j < actual_Bc) {
                if (causal && (q_start + i) + causal_offset < (kv_start + j)) {
                    p = 0.0f;
                } else {
                    p = expf(S_tile[i * Bc + j] - l_tile[i]);
                    if (apply_dropout) {
                        const uint32_t query_idx = static_cast<uint32_t>(q_start + i);
                        const uint32_t kv_pos    = static_cast<uint32_t>(kv_start + j);
                        const float u = philox_uniform(
                            static_cast<uint32_t>(batch_head),
                            query_idx, kv_pos, rng_seed);
                        if (u < dropout_p) {
                            p = 0.0f;
                        } else {
                            p *= dropout_scale;
                        }
                    }
                }
            }
            S_tile[i * Bc + j] = p;
        }
        __syncthreads();

        // Accumulate dV_j += P_ij^T @ dO_i
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

        // Compute dS_ij = P_ij * (dP_ij - D_i)
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

        // Accumulate dK_j += dS_ij^T @ Q_i * scale
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

        // dQ_i += dS_ij @ K_j * scale — atomicAdd into FP32 scratch
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

    // Write dK and dV: convert float -> T
    {
        int e = 0;
        for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE, ++e) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            if (row < actual_Bc) {
                dK_base[(kv_start + row) * HEAD_DIM + col] = from_float_to(T{}, dk_acc[e]);
                dV_base[(kv_start + row) * HEAD_DIM + col] = from_float_to(T{}, dv_acc[e]);
            }
        }
    }
}

// Host wrapper for fused flash attention backward
auto flash_attention_backward_cuda(
    const Tensor& dO,    // [batch_heads, seq_len, head_dim]
    const Tensor& Q,     // [batch_heads, seq_len, head_dim]
    const Tensor& K,     // [batch_heads, seq_len, head_dim]
    const Tensor& V,     // [batch_heads, seq_len, head_dim]
    const Tensor& O,     // [batch_heads, seq_len, head_dim]
    const Tensor& L,     // [batch_heads, seq_len] logsumexp
    float scale,
    bool causal,
    float dropout_p,
    const Tensor& philox_seed,
    const Tensor& philox_offset
) -> std::vector<Tensor> {
    // Phase P0 / Fix 6: extract the rng seed from the saved Philox state
    // tensor (a single uint32) so the backward kernel can regenerate the
    // exact same dropout mask the forward used. philox_offset is the
    // counter[3] slot — fixed at 0 in our forward kernel, so unused.
    //
    // Audit K.4: avoid unconditional `philox_seed.to(Device::cpu()).contiguous()`
    // — that path serialises the stream and triggers a full allocator + dispatch
    // round-trip for what is at most an 8-byte scalar. `dtype()` is a host-side
    // metadata query (no transfer); the actual value is fetched with a single
    // `cudaMemcpy` D2H of the scalar, only when dropout is active.
    uint32_t rng_seed = 0;
    if (dropout_p > 0.0f && philox_seed.numel() > 0) {
        const DType seed_dtype = philox_seed.dtype();
        if (seed_dtype == DType::Int64) {
            int64_t seed_host = 0;
            // audit V.18: surface cudaMemcpy errors instead of silently dropping them.
            TENZOR_CUDA_CHECK(cudaMemcpy(&seed_host, philox_seed.data_ptr(),
                       sizeof(int64_t), cudaMemcpyDeviceToHost));
            rng_seed = static_cast<uint32_t>(seed_host);
        } else if (seed_dtype == DType::Int32) {
            int32_t seed_host = 0;
            // audit V.18.
            TENZOR_CUDA_CHECK(cudaMemcpy(&seed_host, philox_seed.data_ptr(),
                       sizeof(int32_t), cudaMemcpyDeviceToHost));
            rng_seed = static_cast<uint32_t>(seed_host);
        } else {
            throw std::runtime_error(
                "flash_attention_backward_cuda: philox_seed must be Int32 or "
                "Int64 (got dtype id " +
                std::to_string(static_cast<int>(seed_dtype)) + ").");
        }
    }
    (void)philox_offset;  // counter[3] hardcoded to 0 in philox_uniform
    int64_t batch_heads = Q.shape()[0];
    int64_t seq_len_q = Q.shape()[1];
    int64_t seq_len_k = K.shape()[1];
    int64_t head_dim = Q.shape()[2];

    const auto dtype = Q.dtype();

    if (dtype != DType::Float32 && dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error(
            "flash_attention_backward_cuda: Unsupported dtype. "
            "Supported: Float32, Float16, BFloat16");
    }

    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int BLOCK_SIZE = 256;

    int num_kv_tiles = (seq_len_k + Bc - 1) / Bc;
    dim3 grid(num_kv_tiles, batch_heads);
    dim3 threads(BLOCK_SIZE);
    // JIT-R115: see fused_attention_cuda's identical fix/rationale above.
    cudaStream_t stream = cuda::cuda_current_stream();

    // Shared memory is always FP32 regardless of input dtype
    auto compute_bwd_smem = [&](int hd) -> size_t {
        return (2 * Bc * hd + 2 * Br * hd + Br * Bc + Br + Br) * sizeof(float);
    };

    // Helper to optionally request extended shared memory for large tile sizes
    auto maybe_set_max_smem = [](const void* func, size_t smem_bytes) {
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem_bytes));
        }
    };

    if (dtype == DType::Float32) {
        // Float32 path — use original kernel (no conversion overhead)
        Tensor dQ = create_cuda_zeros({batch_heads, seq_len_q, head_dim}, DType::Float32, Q.device());
        Tensor dK = create_cuda_zeros({batch_heads, seq_len_k, head_dim}, DType::Float32, K.device());
        Tensor dV = create_cuda_zeros({batch_heads, seq_len_k, head_dim}, DType::Float32, V.device());

        const float* q_ptr  = Q.data<float>();
        const float* k_ptr  = K.data<float>();
        const float* v_ptr  = V.data<float>();
        const float* o_ptr  = O.data<float>();
        const float* do_ptr = dO.data<float>();
        const float* l_ptr  = L.data<float>();
        float* dq_ptr = dQ.data<float>();
        float* dk_ptr = dK.data<float>();
        float* dv_ptr = dV.data<float>();
        int seq_len_q_int = static_cast<int>(seq_len_q);
        int seq_len_k_int = static_cast<int>(seq_len_k);

        // Wave E3 (deferred → landed): extended head_dim support. Added
        // arms for {16, 48, 80, 96, 160} — covers ViT (head_dim=64), DeiT
        // (64), Mistral (128), Llama-7B (128), Llama-30B (128 still),
        // smaller-model variants (32-80), and Whisper (160).
        auto launch_f32 = [&](auto kernel_fn, int hd) {
            size_t smem = compute_bwd_smem(hd);
            maybe_set_max_smem(reinterpret_cast<const void*>(kernel_fn), smem);
            kernel_fn<<<grid, threads, smem, stream>>>(
                q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_ptr, dk_ptr, dv_ptr,
                seq_len_q_int, seq_len_k_int, scale, causal, dropout_p, rng_seed);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        };
        switch (head_dim) {
            case 16:  launch_f32(flash_attention_backward_kernel<16,  Br, Bc, BLOCK_SIZE>, 16);  break;
            case 32:  launch_f32(flash_attention_backward_kernel<32,  Br, Bc, BLOCK_SIZE>, 32);  break;
            case 48:  launch_f32(flash_attention_backward_kernel<48,  Br, Bc, BLOCK_SIZE>, 48);  break;
            case 64:  launch_f32(flash_attention_backward_kernel<64,  Br, Bc, BLOCK_SIZE>, 64);  break;
            case 80:  launch_f32(flash_attention_backward_kernel<80,  Br, Bc, BLOCK_SIZE>, 80);  break;
            case 96:  launch_f32(flash_attention_backward_kernel<96,  Br, Bc, BLOCK_SIZE>, 96);  break;
            case 128: launch_f32(flash_attention_backward_kernel<128, Br, Bc, BLOCK_SIZE>, 128); break;
            case 160: launch_f32(flash_attention_backward_kernel<160, Br, Bc, BLOCK_SIZE>, 160); break;
            default:
                throw std::runtime_error(
                    "flash_attention_backward_cuda: Unsupported head_dim " +
                    std::to_string(head_dim) +
                    ". Fused backward supports {16, 32, 48, 64, 80, 96, 128, 160}; "
                    "head_dim 192/256 exceed shared-memory limits on common arches "
                    "and are deferred until a multi-block-tiled backward lands.");
        }

        TENZOR_CUDA_POST_LAUNCH_CHECK();
        return {dQ, dK, dV};
    }

    // FP16 / BF16 path — mixed-precision kernel with FP32 accumulation
    // dQ uses FP32 scratch for atomicAdd, then convert to output dtype
    Tensor dQ_f32 = create_cuda_zeros({batch_heads, seq_len_q, head_dim}, DType::Float32, Q.device());
    Tensor dK = create_cuda_zeros({batch_heads, seq_len_k, head_dim}, dtype, K.device());
    Tensor dV = create_cuda_zeros({batch_heads, seq_len_k, head_dim}, dtype, V.device());

    const float* l_ptr = L.data<float>();
    float* dq_f32_ptr = dQ_f32.data<float>();
    int seq_len_q_int = static_cast<int>(seq_len_q);
    int seq_len_k_int = static_cast<int>(seq_len_k);

    // Dispatch FP16 / BF16 kernel launches
    auto launch_mp_kernels = [&](auto q_ptr, auto k_ptr, auto v_ptr, auto o_ptr, auto do_ptr,
                                  auto dk_ptr, auto dv_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(q_ptr)>>;
        auto launch_mp = [&](auto kernel_fn, int hd) {
            size_t smem = compute_bwd_smem(hd);
            maybe_set_max_smem(reinterpret_cast<const void*>(kernel_fn), smem);
            kernel_fn<<<grid, threads, smem, stream>>>(
                q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_f32_ptr, dk_ptr, dv_ptr,
                seq_len_q_int, seq_len_k_int, scale, causal, dropout_p, rng_seed);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        };
        // Wave E3 (deferred → landed): extended head_dim arms, same set
        // as the F32 path above.
        switch (head_dim) {
            case 16:  launch_mp(flash_attention_backward_kernel_mp<T, 16,  Br, Bc, BLOCK_SIZE>, 16);  break;
            case 32:  launch_mp(flash_attention_backward_kernel_mp<T, 32,  Br, Bc, BLOCK_SIZE>, 32);  break;
            case 48:  launch_mp(flash_attention_backward_kernel_mp<T, 48,  Br, Bc, BLOCK_SIZE>, 48);  break;
            case 64:  launch_mp(flash_attention_backward_kernel_mp<T, 64,  Br, Bc, BLOCK_SIZE>, 64);  break;
            case 80:  launch_mp(flash_attention_backward_kernel_mp<T, 80,  Br, Bc, BLOCK_SIZE>, 80);  break;
            case 96:  launch_mp(flash_attention_backward_kernel_mp<T, 96,  Br, Bc, BLOCK_SIZE>, 96);  break;
            case 128: launch_mp(flash_attention_backward_kernel_mp<T, 128, Br, Bc, BLOCK_SIZE>, 128); break;
            case 160: launch_mp(flash_attention_backward_kernel_mp<T, 160, Br, Bc, BLOCK_SIZE>, 160); break;
            default:
                throw std::runtime_error(
                    "flash_attention_backward_cuda: Unsupported head_dim " +
                    std::to_string(head_dim) +
                    ". FP16/BF16 fused backward supports {16, 32, 48, 64, 80, 96, 128, 160}.");
        }
    };

    if (dtype == DType::Float16) {
        launch_mp_kernels(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<const __half*>(O.data_ptr()),
            reinterpret_cast<const __half*>(dO.data_ptr()),
            reinterpret_cast<__half*>(dK.data_ptr()),
            reinterpret_cast<__half*>(dV.data_ptr()));
    } else {
        launch_mp_kernels(
            reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(O.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(dO.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dK.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dV.data_ptr()));
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    // Convert dQ from FP32 scratch to output dtype
    Tensor dQ = dQ_f32.to(dtype);

    return {dQ, dK, dV};
}

// ==============================================================================
// Fused Adam Optimizer CUDA Kernel
// ==============================================================================

/**
 * @brief Fused Adam/AdamW optimizer kernel
 *
 * Performs the complete Adam update in a single kernel:
 * - First moment update: m = beta1 * m + (1-beta1) * grad
 * - Second moment update: v = beta2 * v + (1-beta2) * grad^2
 * - Bias correction applied through step_size
 * - Parameter update: param = param - step_size * m / (sqrt(v_hat) + eps)
 *
 * For AdamW (decoupled weight decay): param = param * (1 - lr * weight_decay) first
 *
 * This eliminates ~15-20 kernel launches per parameter in the naive implementation.
 */
template<typename T>
__global__ void fused_adam_kernel(
    T* __restrict__ param,             // Parameter tensor (modified in-place)
    const T* __restrict__ grad,        // Gradient tensor
    T* __restrict__ exp_avg,           // First moment (m) - modified in-place
    T* __restrict__ exp_avg_sq,        // Second moment (v) - modified in-place
    T* __restrict__ max_exp_avg_sq,    // Max second moment (AMSGrad) - nullptr if disabled
    int64_t numel,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double bias_correction1,    // 1 - beta1^step
    double bias_correction2,    // 1 - beta2^step
    bool amsgrad,
    bool decoupled_weight_decay  // True for AdamW, false for L2 regularization
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    T g = grad[idx];
    T p = param[idx];
    T m = exp_avg[idx];
    T v = exp_avg_sq[idx];

    // Apply L2 regularization to gradient (Adam style) if not decoupled
    if (weight_decay > 0.0 && !decoupled_weight_decay) {
        g = g + T(weight_decay) * p;
    }

    // Update biased first moment estimate
    m = T(beta1) * m + T(1.0 - beta1) * g;

    // Update biased second raw moment estimate
    v = T(beta2) * v + T(1.0 - beta2) * g * g;

    // Compute step size with bias correction
    T step_size = T(lr) / T(bias_correction1);

    // Bias-corrected second moment. AMSGrad must track the running maximum over
    // the RAW (un-bias-corrected) second moment and apply the bias correction
    // AFTER, matching the CPU reference and PyTorch — maxing the already
    // bias-corrected v_hat is wrong because bias_correction2 grows toward 1.
    T v_hat;
    if (amsgrad && max_exp_avg_sq != nullptr) {
        T max_v = max_exp_avg_sq[idx];
        if (v > max_v) max_v = v;
        max_exp_avg_sq[idx] = max_v;
        v_hat = max_v / T(bias_correction2);
    } else {
        v_hat = v / T(bias_correction2);
    }

    // denom = sqrt(v_hat) + eps
    T denom = sqrt(v_hat) + T(eps);

    // Apply decoupled weight decay (AdamW style) before update
    if (weight_decay > 0.0 && decoupled_weight_decay) {
        p = p * (T(1) - T(lr) * T(weight_decay));
    }

    // Update parameter
    p = p - step_size * m / denom;

    // Store updated values
    param[idx] = p;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
}

/**
 * @brief Vectorized fused Adam kernel using float4
 *
 * Processes 4 elements per thread for better memory bandwidth utilization.
 */
__global__ void fused_adam_kernel_vec4(
    float4* __restrict__ param,
    const float4* __restrict__ grad,
    float4* __restrict__ exp_avg,
    float4* __restrict__ exp_avg_sq,
    float4* __restrict__ max_exp_avg_sq,
    int64_t numel_vec4,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    bool amsgrad,
    bool decoupled_weight_decay
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_vec4) return;

    float4 g = grad[idx];
    float4 p = param[idx];
    float4 m = exp_avg[idx];
    float4 v = exp_avg_sq[idx];
    float4 mv;
    if (amsgrad && max_exp_avg_sq) mv = max_exp_avg_sq[idx];

    float step_size = lr / bias_correction1;
    float bc2_inv = 1.0f / bias_correction2;

    // Process all 4 elements
    #define ADAM_UPDATE(comp) \
        if (weight_decay > 0.0f && !decoupled_weight_decay) { \
            g.comp = g.comp + weight_decay * p.comp; \
        } \
        m.comp = beta1 * m.comp + (1.0f - beta1) * g.comp; \
        v.comp = beta2 * v.comp + (1.0f - beta2) * g.comp * g.comp; \
        { float v_hat; \
          if (amsgrad && max_exp_avg_sq) { \
              if (v.comp > mv.comp) mv.comp = v.comp; /* max over RAW v */ \
              v_hat = mv.comp * bc2_inv;              /* bias-correct after */ \
          } else { \
              v_hat = v.comp * bc2_inv; \
          } \
          if (weight_decay > 0.0f && decoupled_weight_decay) { \
              p.comp = p.comp * (1.0f - lr * weight_decay); \
          } \
          p.comp = p.comp - step_size * m.comp / (sqrtf(v_hat) + eps); \
        }

    ADAM_UPDATE(x)
    ADAM_UPDATE(y)
    ADAM_UPDATE(z)
    ADAM_UPDATE(w)

    #undef ADAM_UPDATE

    param[idx] = p;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
    if (amsgrad && max_exp_avg_sq) max_exp_avg_sq[idx] = mv;
}

/**
 * @brief Vectorized fused Adam kernel using double2
 *
 * Processes 2 elements per thread for Float64 memory bandwidth utilization.
 */
__global__ void fused_adam_kernel_vec2(
    double2* __restrict__ param,
    const double2* __restrict__ grad,
    double2* __restrict__ exp_avg,
    double2* __restrict__ exp_avg_sq,
    double2* __restrict__ max_exp_avg_sq,
    int64_t numel_vec2,
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
    if (idx >= numel_vec2) return;

    double2 g = grad[idx];
    double2 p = param[idx];
    double2 m = exp_avg[idx];
    double2 v = exp_avg_sq[idx];
    double2 mv;
    if (amsgrad && max_exp_avg_sq) mv = max_exp_avg_sq[idx];

    double step_size = lr / bias_correction1;
    double bc2_inv = 1.0 / bias_correction2;

    #define ADAM_UPDATE_D(comp) \
        if (weight_decay > 0.0 && !decoupled_weight_decay) { \
            g.comp = g.comp + weight_decay * p.comp; \
        } \
        m.comp = beta1 * m.comp + (1.0 - beta1) * g.comp; \
        v.comp = beta2 * v.comp + (1.0 - beta2) * g.comp * g.comp; \
        { double v_hat; \
          if (amsgrad && max_exp_avg_sq) { \
              if (v.comp > mv.comp) mv.comp = v.comp; /* max over RAW v */ \
              v_hat = mv.comp * bc2_inv;              /* bias-correct after */ \
          } else { \
              v_hat = v.comp * bc2_inv; \
          } \
          if (weight_decay > 0.0 && decoupled_weight_decay) { \
              p.comp = p.comp * (1.0 - lr * weight_decay); \
          } \
          p.comp = p.comp - step_size * m.comp / (sqrt(v_hat) + eps); \
        }

    ADAM_UPDATE_D(x)
    ADAM_UPDATE_D(y)

    #undef ADAM_UPDATE_D

    param[idx] = p;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
    if (amsgrad && max_exp_avg_sq) max_exp_avg_sq[idx] = mv;
}

/**
 * @brief Host function for fused Adam optimizer step
 *
 * @param param Parameter tensor (modified in-place)
 * @param grad Gradient tensor
 * @param exp_avg First moment buffer (modified in-place)
 * @param exp_avg_sq Second moment buffer (modified in-place)
 * @param lr Learning rate
 * @param beta1 First moment decay rate
 * @param beta2 Second moment decay rate
 * @param eps Epsilon for numerical stability
 * @param weight_decay Weight decay coefficient
 * @param step Current step count (for bias correction)
 * @param decoupled_weight_decay If true, use AdamW style decoupled weight decay
 */
auto fused_adam_step_cuda(
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
    cudaStream_t stream,
    Tensor* max_exp_avg_sq,
    bool amsgrad
) -> void {
    int64_t numel = param.numel();

    // Compute bias corrections in double precision for accuracy
    double bias_correction1 = 1.0 - std::pow(static_cast<double>(beta1), static_cast<double>(step));
    double bias_correction2 = 1.0 - std::pow(static_cast<double>(beta2), static_cast<double>(step));

    constexpr int BLOCK_SIZE = 256;

    if (param.dtype() == DType::Float32) {
        float* max_sq_ptr = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<float>() : nullptr;

        // Check if we can use vectorized kernel
        bool can_vectorize = (numel % 4 == 0) &&
                             (reinterpret_cast<uintptr_t>(param.data<float>()) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(grad.data<float>()) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(exp_avg.data<float>()) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(exp_avg_sq.data<float>()) % 16 == 0) &&
                             (!amsgrad || !max_sq_ptr || (reinterpret_cast<uintptr_t>(max_sq_ptr) % 16 == 0));

        if (can_vectorize && numel >= 1024) {
            int64_t numel_vec4 = numel / 4;
            int blocks = (numel_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            blocks = clamp_blocks(blocks);

            fused_adam_kernel_vec4<<<blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<float4*>(param.data<float>()),
                reinterpret_cast<const float4*>(grad.data<float>()),
                reinterpret_cast<float4*>(exp_avg.data<float>()),
                reinterpret_cast<float4*>(exp_avg_sq.data<float>()),
                max_sq_ptr ? reinterpret_cast<float4*>(max_sq_ptr) : nullptr,
                numel_vec4,
                lr, beta1, beta2, eps, weight_decay,
                bias_correction1, bias_correction2,
                amsgrad, decoupled_weight_decay
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else {
            int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
            blocks = clamp_blocks(blocks);

            fused_adam_kernel<float><<<blocks, BLOCK_SIZE, 0, stream>>>(
                param.data<float>(),
                grad.data<float>(),
                exp_avg.data<float>(),
                exp_avg_sq.data<float>(),
                max_sq_ptr,
                numel, lr, beta1, beta2, eps, weight_decay,
                bias_correction1, bias_correction2,
                amsgrad, decoupled_weight_decay
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }
    } else if (param.dtype() == DType::Float64) {
        double* max_sq_ptr = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<double>() : nullptr;

        // Check if we can use vectorized kernel (double2 = 16 bytes alignment)
        bool can_vectorize = (numel % 2 == 0) &&
                             (reinterpret_cast<uintptr_t>(param.data<double>()) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(grad.data<double>()) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(exp_avg.data<double>()) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(exp_avg_sq.data<double>()) % 16 == 0) &&
                             (!amsgrad || !max_sq_ptr || (reinterpret_cast<uintptr_t>(max_sq_ptr) % 16 == 0));

        if (can_vectorize && numel >= 512) {
            int64_t numel_vec2 = numel / 2;
            int blocks = (numel_vec2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            blocks = clamp_blocks(blocks);

            fused_adam_kernel_vec2<<<blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<double2*>(param.data<double>()),
                reinterpret_cast<const double2*>(grad.data<double>()),
                reinterpret_cast<double2*>(exp_avg.data<double>()),
                reinterpret_cast<double2*>(exp_avg_sq.data<double>()),
                max_sq_ptr ? reinterpret_cast<double2*>(max_sq_ptr) : nullptr,
                numel_vec2,
                lr, beta1, beta2, eps, weight_decay,
                bias_correction1, bias_correction2,
                amsgrad, decoupled_weight_decay
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        } else {
            int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
            blocks = clamp_blocks(blocks);

            fused_adam_kernel<double><<<blocks, BLOCK_SIZE, 0, stream>>>(
                param.data<double>(),
                grad.data<double>(),
                exp_avg.data<double>(),
                exp_avg_sq.data<double>(),
                max_sq_ptr,
                numel, lr, beta1, beta2, eps, weight_decay,
                bias_correction1, bias_correction2,
                amsgrad, decoupled_weight_decay
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }
    } else {
        throw std::runtime_error("fused_adam_step_cuda: Only Float32 and Float64 supported");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

// ==============================================================================
// Fused SGD with Momentum CUDA Kernel
// ==============================================================================

/**
 * @brief Fused SGD with momentum and weight decay
 *
 * Nesterov momentum: v = momentum * v + grad; param = param - lr * (grad + momentum * v)
 * Standard momentum: v = momentum * v + grad; param = param - lr * v
 */
template<typename T>
__global__ void fused_sgd_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ momentum_buffer,
    int64_t numel,
    double lr,
    double momentum,
    double weight_decay,
    double dampening,
    bool nesterov,
    bool has_momentum_buffer,
    bool first_step
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    T g = grad[idx];
    T p = param[idx];

    // Apply weight decay
    if (weight_decay > 0.0) {
        g = g + T(weight_decay) * p;
    }

    if (has_momentum_buffer && momentum > 0.0) {
        T v;

        // PyTorch SGD: on the very first momentum step the buffer is
        // initialised to the (weight-decayed) gradient with NO dampening;
        // dampening is only applied on subsequent steps. Applying
        // (1 - dampening) on step 1 (buffer == 0) is the latent bug this
        // fixes — it must match the CPU reference / torch.optim.SGD.
        if (first_step) {
            v = g;
        } else {
            v = T(momentum) * momentum_buffer[idx] + T(1.0 - dampening) * g;
        }
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

auto fused_sgd_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor* momentum_buffer,
    double lr,
    double momentum,
    double weight_decay,
    double dampening,
    bool nesterov,
    bool first_step,
    cudaStream_t stream
) -> void {
    int64_t numel = param.numel();
    constexpr int BLOCK_SIZE = 256;
    int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = clamp_blocks(blocks);

    if (param.dtype() == DType::Float32) {
        float* momentum_ptr = momentum_buffer ? momentum_buffer->data<float>() : nullptr;

        fused_sgd_kernel<float><<<blocks, BLOCK_SIZE, 0, stream>>>(
            param.data<float>(),
            grad.data<float>(),
            momentum_ptr,
            numel, lr, momentum, weight_decay, dampening,
            nesterov, momentum_buffer != nullptr, first_step
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (param.dtype() == DType::Float64) {
        double* momentum_ptr = momentum_buffer ? momentum_buffer->data<double>() : nullptr;

        fused_sgd_kernel<double><<<blocks, BLOCK_SIZE, 0, stream>>>(
            param.data<double>(),
            grad.data<double>(),
            momentum_ptr,
            numel, lr, momentum, weight_decay, dampening,
            nesterov, momentum_buffer != nullptr, first_step
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_sgd_step_cuda: Only Float32 and Float64 supported");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
}


// ============================================================================
// Fused Softmax Cross Entropy
// ============================================================================

template<typename T>
__global__ void fused_softmax_cross_entropy_kernel(
    const T* __restrict__ logits,    // (batch_size, num_classes)
    const int64_t* __restrict__ targets,  // (batch_size)
    T* __restrict__ loss,            // (batch_size) or scalar
    T* __restrict__ grad_logits,     // (batch_size, num_classes) - optional
    int64_t batch_size,
    int64_t num_classes,
    bool compute_grad,
    int* __restrict__ error_flag) {

    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    // Validate target range. The CPU reference throws on any out-of-range
    // target (no ignore_index semantics), so signal the violation via the
    // device error flag (checked on the host) rather than silently emitting a
    // zero loss/grad, which would diverge cross-backend. Avoid the OOB device
    // read of logits_row[target] by returning here.
    {
        int64_t target_check = targets[b];
        if (target_check < 0 || target_check >= num_classes) {
            if (threadIdx.x == 0) {
                atomicExch(error_flag, 1);
                loss[b] = static_cast<T>(0);
            }
            if (compute_grad && grad_logits) {
                T* grad_row = grad_logits + b * num_classes;
                for (int64_t c = threadIdx.x; c < num_classes; c += blockDim.x) {
                    grad_row[c] = T(0);
                }
            }
            return;
        }
    }

    const T* logits_row = logits + b * num_classes;

    // Step 1: Find max for numerical stability (parallel reduction in block)
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    T thread_max = -1e30f;
    for (int64_t c = threadIdx.x; c < num_classes; c += blockDim.x) {
        thread_max = max(thread_max, static_cast<T>(logits_row[c]));
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // Block reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    T max_val = sdata[0];
    __syncthreads();

    // Step 2: Compute sum of exp(logits - max)
    T thread_sum = T(0);
    for (int64_t c = threadIdx.x; c < num_classes; c += blockDim.x) {
        thread_sum += exp(static_cast<T>(logits_row[c]) - max_val);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    T log_sum_exp = log(sdata[0]) + max_val;

    // Step 3: Compute loss = log_sum_exp - logits[target]
    int64_t target = targets[b];
    if (threadIdx.x == 0) {
        loss[b] = log_sum_exp - static_cast<T>(logits_row[target]);
    }

    // Step 4: Compute gradient if requested.
    // F130: return the RAW per-sample gradient (softmax - onehot), UNSCALED.
    // The registry (cuda_kernel_registry.cpp) applies the reduction scale
    // (1/num_rows for "mean") exactly once, matching CPU (cpu/kernels/
    // fused_ops.cpp) and ROCm (fused_ops.hip.cpp), which also return an unscaled
    // grad. Baking 1/batch_size in here double-scaled the "mean" grad to 1/N^2
    // and added a spurious 1/N to "sum"/"none". (void)batch_size keeps the
    // signature stable for the loss-only computation above.
    (void)batch_size;
    if (compute_grad && grad_logits) {
        T* grad_row = grad_logits + b * num_classes;
        for (int64_t c = threadIdx.x; c < num_classes; c += blockDim.x) {
            T softmax_val = exp(static_cast<T>(logits_row[c]) - log_sum_exp);
            T grad = softmax_val;
            if (c == target) grad -= T(1);
            grad_row[c] = grad;
        }
    }
}

auto fused_softmax_cross_entropy_cuda(
    const Tensor& logits,
    const Tensor& targets,
    bool compute_grad
) -> std::tuple<Tensor, Tensor> {
    auto shape_span = logits.shape();
    int64_t batch_size = shape_span[0];
    int64_t num_classes = shape_span[1];

    Tensor loss({batch_size}, logits.dtype(), logits.device());
    Tensor grad_logits;
    if (compute_grad) {
        grad_logits = Tensor(to_vector(shape_span), logits.dtype(), logits.device());
    }

    int block_size = std::min<int>(256, static_cast<int>(num_classes));
    // Round up to next power of 2 for reduction
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;
    if (block_size > 1024) block_size = 1024;

    size_t shared_mem = block_size * dtype_size(logits.dtype());

    int32_t device_id = logits.device().index;
    auto stream_guard = cuda::CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

    // Device-side out-of-range target flag (CPU reference throws on OOB; match
    // that contract rather than silently zeroing loss/grad).
    CudaBuffer error_buf(sizeof(int));
    TENZOR_CUDA_CHECK(cudaMemsetAsync(error_buf.as<int>(), 0, sizeof(int), stream));

    if (logits.dtype() == DType::Float32) {
        fused_softmax_cross_entropy_kernel<float><<<batch_size, block_size, shared_mem, stream>>>(
            logits.data<float>(), targets.data<int64_t>(),
            loss.data<float>(),
            compute_grad ? grad_logits.data<float>() : nullptr,
            batch_size, num_classes, compute_grad, error_buf.as<int>());
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (logits.dtype() == DType::Float64) {
        fused_softmax_cross_entropy_kernel<double><<<batch_size, block_size, shared_mem, stream>>>(
            logits.data<double>(), targets.data<int64_t>(),
            loss.data<double>(),
            compute_grad ? grad_logits.data<double>() : nullptr,
            batch_size, num_classes, compute_grad, error_buf.as<int>());
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_softmax_cross_entropy: unsupported dtype");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    {
        int host_error = 0;
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(&host_error, error_buf.as<int>(), sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));
        TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (host_error) {
            throw std::runtime_error(
                "fused_softmax_cross_entropy: target index out of range [0, " +
                std::to_string(num_classes) + ")");
        }
    }

    return {loss, grad_logits};
}

// ============================================================================
// Fused RMSProp Optimizer Step
// ============================================================================

template<typename T>
__global__ void fused_rmsprop_step_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ square_avg,
    T* __restrict__ grad_avg,       // For centered RMSProp, nullptr otherwise
    T* __restrict__ momentum_buffer, // For momentum, nullptr otherwise
    double lr, double alpha, double eps,
    double weight_decay, double momentum,
    bool centered,
    int64_t n) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T g = grad[idx];

    // Weight decay
    if (weight_decay != 0.0) {
        g = g + T(weight_decay) * param[idx];
    }

    // Update square average: v = alpha * v + (1 - alpha) * g^2
    T sq = square_avg[idx];
    sq = T(alpha) * sq + T(1.0 - alpha) * g * g;
    square_avg[idx] = sq;

    T avg;
    if (centered && grad_avg) {
        // Update grad average: g_avg = alpha * g_avg + (1 - alpha) * g
        T ga = grad_avg[idx];
        ga = T(alpha) * ga + T(1.0 - alpha) * g;
        grad_avg[idx] = ga;
        // eps OUTSIDE sqrt (PyTorch-correct): sqrt(v - m^2) + eps. The eager
        // path (rmsprop.hpp) and CPU/ROCm/oneAPI/MPS all add eps after the sqrt;
        // the old sqrt(v - m^2 + eps) diverged on CUDA (F092).
        avg = sqrt(sq - ga * ga) + T(eps);
    } else {
        // eps OUTSIDE sqrt (PyTorch-correct): sqrt(v) + eps. See note above (F092).
        avg = sqrt(sq) + T(eps);
    }

    if (momentum > 0.0 && momentum_buffer) {
        T buf = momentum_buffer[idx];
        buf = T(momentum) * buf + g / avg;
        momentum_buffer[idx] = buf;
        param[idx] = param[idx] - T(lr) * buf;
    } else {
        param[idx] = param[idx] - T(lr) * g / avg;
    }
}

auto fused_rmsprop_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor& square_avg,
    Tensor* grad_avg,
    Tensor* momentum_buffer,
    double lr, double alpha, double eps,
    double weight_decay, double momentum,
    bool centered,
    cudaStream_t stream
) -> void {
    int64_t n = param.numel();
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    if (param.dtype() == DType::Float32) {
        fused_rmsprop_step_kernel<float><<<num_blocks, block_size, 0, stream>>>(
            param.data<float>(), grad.data<float>(), square_avg.data<float>(),
            (centered && grad_avg) ? grad_avg->data<float>() : nullptr,
            (momentum > 0.0 && momentum_buffer) ? momentum_buffer->data<float>() : nullptr,
            lr, alpha, eps, weight_decay, momentum, centered, n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (param.dtype() == DType::Float64) {
        fused_rmsprop_step_kernel<double><<<num_blocks, block_size, 0, stream>>>(
            param.data<double>(), grad.data<double>(), square_avg.data<double>(),
            (centered && grad_avg) ? grad_avg->data<double>() : nullptr,
            (momentum > 0.0 && momentum_buffer) ? momentum_buffer->data<double>() : nullptr,
            lr, alpha, eps, weight_decay, momentum, centered, n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_rmsprop_step: unsupported dtype");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

// ============================================================================
// Fused Adadelta Optimizer Step
// ============================================================================

template<typename T>
__global__ void fused_adadelta_step_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ square_avg,
    T* __restrict__ acc_delta,
    double rho, double eps, double lr, double weight_decay,
    int64_t n) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T g = grad[idx];
    if (weight_decay != 0.0) {
        g = g + T(weight_decay) * param[idx];
    }

    // v = rho * v + (1 - rho) * g^2
    T sq = square_avg[idx];
    sq = T(rho) * sq + T(1.0 - rho) * g * g;
    square_avg[idx] = sq;

    // delta = sqrt(acc_delta + eps) / sqrt(sq + eps) * g
    T std_val = sqrt(sq + T(eps));
    T delta = sqrt(acc_delta[idx] + T(eps)) / std_val * g;

    // acc_delta = rho * acc_delta + (1 - rho) * delta^2
    acc_delta[idx] = T(rho) * acc_delta[idx] + T(1.0 - rho) * delta * delta;

    param[idx] = param[idx] - T(lr) * delta;
}

// Float16 / BFloat16 kernel: read half → compute in float → write back half.
// Runs directly on the original tensor storage so the optimizer sees the
// update without needing any host-side rebind.
template <typename HalfT>
__global__ void fused_adadelta_step_half_kernel(
    HalfT* __restrict__ param,
    const HalfT* __restrict__ grad,
    HalfT* __restrict__ square_avg,
    HalfT* __restrict__ acc_delta,
    float rho, float eps, float lr, float weight_decay,
    int64_t n) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = static_cast<float>(grad[idx]);
    float p = static_cast<float>(param[idx]);
    if (weight_decay != 0.0f) {
        g = g + weight_decay * p;
    }

    float sq = static_cast<float>(square_avg[idx]);
    sq = rho * sq + (1.0f - rho) * g * g;
    square_avg[idx] = static_cast<HalfT>(sq);

    float ad = static_cast<float>(acc_delta[idx]);
    float std_val = sqrtf(sq + eps);
    float delta = sqrtf(ad + eps) / std_val * g;

    acc_delta[idx] = static_cast<HalfT>(rho * ad + (1.0f - rho) * delta * delta);
    param[idx] = static_cast<HalfT>(p - lr * delta);
}

auto fused_adadelta_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor& square_avg,
    Tensor& acc_delta,
    double rho, double eps, double lr, double weight_decay,
    cudaStream_t stream
) -> void {
    int64_t n = param.numel();
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    if (param.dtype() == DType::Float32) {
        fused_adadelta_step_kernel<float><<<num_blocks, block_size, 0, stream>>>(
            param.data<float>(), grad.data<float>(), square_avg.data<float>(), acc_delta.data<float>(),
            rho, eps, lr, weight_decay, n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (param.dtype() == DType::Float64) {
        fused_adadelta_step_kernel<double><<<num_blocks, block_size, 0, stream>>>(
            param.data<double>(), grad.data<double>(), square_avg.data<double>(), acc_delta.data<double>(),
            rho, eps, lr, weight_decay, n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (param.dtype() == DType::Float16) {
        // grad may be passed in Float32 (optimizer often does not cast grads
        // to match param dtype); accept either.
        Tensor grad_h = (grad.dtype() == DType::Float16) ? grad : grad.to(DType::Float16);
        fused_adadelta_step_half_kernel<__half><<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<__half*>(param.data<Float16>()),
            reinterpret_cast<const __half*>(grad_h.data<Float16>()),
            reinterpret_cast<__half*>(square_avg.data<Float16>()),
            reinterpret_cast<__half*>(acc_delta.data<Float16>()),
            static_cast<float>(rho), static_cast<float>(eps),
            static_cast<float>(lr), static_cast<float>(weight_decay), n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (param.dtype() == DType::BFloat16) {
        Tensor grad_h = (grad.dtype() == DType::BFloat16) ? grad : grad.to(DType::BFloat16);
        fused_adadelta_step_half_kernel<__nv_bfloat16><<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(param.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad_h.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(square_avg.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(acc_delta.data<BFloat16>()),
            static_cast<float>(rho), static_cast<float>(eps),
            static_cast<float>(lr), static_cast<float>(weight_decay), n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_adadelta_step: unsupported dtype");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

// ============================================================================
// Fused Adagrad Optimizer Step
// ============================================================================

template<typename T>
__global__ void fused_adagrad_step_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ sum_sq,
    double lr, double lr_decay, double eps, double weight_decay,
    int64_t step, int64_t n) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T g = grad[idx];
    if (weight_decay != 0.0) {
        g = g + T(weight_decay) * param[idx];
    }

    // Effective learning rate, computed in the parameter's compute type so that
    // Float64 params keep full double precision (matches CPU/Vulkan contract).
    T clr = T(lr) / (T(1) + T(step - 1) * T(lr_decay));

    // sum_sq += g^2
    T sq = sum_sq[idx] + g * g;
    sum_sq[idx] = sq;

    // param -= clr * g / (sqrt(sum_sq) + eps)
    param[idx] = param[idx] - clr * g / (sqrt(sq) + T(eps));
}

auto fused_adagrad_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor& sum_sq,
    double lr, double lr_decay, double eps, double weight_decay,
    int64_t step,
    cudaStream_t stream
) -> void {
    int64_t n = param.numel();
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    if (param.dtype() == DType::Float32) {
        fused_adagrad_step_kernel<float><<<num_blocks, block_size, 0, stream>>>(
            param.data<float>(), grad.data<float>(), sum_sq.data<float>(),
            lr, lr_decay, eps, weight_decay, step, n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (param.dtype() == DType::Float64) {
        fused_adagrad_step_kernel<double><<<num_blocks, block_size, 0, stream>>>(
            param.data<double>(), grad.data<double>(), sum_sq.data<double>(),
            lr, lr_decay, eps, weight_decay, step, n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_adagrad_step: unsupported dtype");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}
// ============================================================================
// Fused Adam-Atan2 Optimizer Step
// ============================================================================

template<typename T>
__global__ void fused_adam_atan2_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ exp_avg,
    T* __restrict__ exp_avg_sq,
    T* __restrict__ max_exp_avg_sq,  // nullptr if !amsgrad
    int64_t numel,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double bias_correction1,
    double bias_correction2,
    bool amsgrad
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    T g = grad[idx];
    T p = param[idx];
    T m = exp_avg[idx];
    T v = exp_avg_sq[idx];

    // Update biased first moment estimate
    m = T(beta1) * m + T(1.0 - beta1) * g;

    // Update biased second raw moment estimate
    v = T(beta2) * v + T(1.0 - beta2) * g * g;

    // Bias-corrected first moment estimate
    T m_hat = m / T(bias_correction1);

    // Second moment estimate. AMSGrad must track the running maximum over the
    // RAW (un-bias-corrected) second moment and apply the bias correction
    // AFTER, matching the CPU reference and PyTorch. Maxing the already
    // bias-corrected v_hat is incorrect because bias_correction2 grows toward 1
    // across steps, so it does not preserve a consistent comparison basis.
    T v_hat;
    if (amsgrad && max_exp_avg_sq != nullptr) {
        T max_v = max_exp_avg_sq[idx];
        if (v > max_v) max_v = v;
        max_exp_avg_sq[idx] = max_v;
        v_hat = max_v / T(bias_correction2);
    } else {
        v_hat = v / T(bias_correction2);
    }

    // Decoupled weight decay (like AdamW) before update
    if (weight_decay > 0.0) {
        p = p * (T(1) - T(lr) * T(weight_decay));
    }

    // Adam-atan2 update: atan2(m_hat, sqrt(v_hat) + eps)
    T denom = sqrt(v_hat) + T(eps);
    T update = atan2(m_hat, denom);

    // Apply update
    p = p - T(lr) * update;

    // Store updated values
    param[idx] = p;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
}

auto fused_adam_atan2_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    Tensor* max_exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step,
    bool amsgrad,
    cudaStream_t stream
) -> void {
    int64_t numel = param.numel();
    constexpr int BLOCK_SIZE = 256;
    int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = clamp_blocks(blocks);

    double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));

    if (param.dtype() == DType::Float32) {
        float* max_sq_ptr = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<float>() : nullptr;

        fused_adam_atan2_kernel<float><<<blocks, BLOCK_SIZE, 0, stream>>>(
            param.data<float>(), grad.data<float>(),
            exp_avg.data<float>(), exp_avg_sq.data<float>(),
            max_sq_ptr, numel,
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2, amsgrad
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (param.dtype() == DType::Float64) {
        double* max_sq_ptr = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<double>() : nullptr;

        fused_adam_atan2_kernel<double><<<blocks, BLOCK_SIZE, 0, stream>>>(
            param.data<double>(), grad.data<double>(),
            exp_avg.data<double>(), exp_avg_sq.data<double>(),
            max_sq_ptr, numel,
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2, amsgrad
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("fused_adam_atan2_step_cuda: Only Float32 and Float64 supported");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_CUDA_AVAILABLE
