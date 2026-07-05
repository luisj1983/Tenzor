#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "../hip_buffer.hpp"
// Note: NOT including tenzor/ops/math.hpp here — its tenzor::sqrt/exp/etc.
// declarations collide with HIP device sqrt/exp inside the __global__ kernels
// in this TU. Use OpId-based dispatch (tenzor::dispatch + AttrKey) for the
// host-side composed-ops fallback below.
#include "reduction_utils.hip.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>
#include <span>

namespace tenzor {
namespace rocm {

// Defined in flash_attention_f64.hip.cpp — native double-precision attention.
auto fused_attention_hip_f64(const Tensor& Q, const Tensor& K, const Tensor& V,
                             double scale, bool causal,
                             hipStream_t stream) -> std::pair<Tensor, Tensor>;

// Helper to create zero-initialized tensor on HIP device
inline Tensor create_hip_zeros(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream = nullptr) {
    Tensor t(shape, dtype, device);
    size_t bytes = t.numel() * dtype_size(dtype);
    if (bytes > 0) {
        hipError_t err = hipMemset(t.data_ptr(), 0, bytes);
        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipMemset failed: ") + hipGetErrorString(err));
        }
    }
    return t;
}

// Helper to convert span to vector
inline std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return std::vector<int64_t>(s.begin(), s.end());
}

// GPU reduction: sum all elements to a scalar tensor using two-phase parallel reduction
inline Tensor reduce_sum_hip(const Tensor& t) {
    int64_t n = t.numel();
    Tensor result({}, t.dtype(), t.device());
    if (t.dtype() == DType::Float32) {
        launch_full_reduction_sum<float>(t.data<float>(), result.data<float>(), n, nullptr);
    } else if (t.dtype() == DType::Float64) {
        launch_full_reduction_sum<double>(t.data<double>(), result.data<double>(), n, nullptr);
    }
    return result;
}

// GPU reduction: mean all elements to a scalar tensor (sum / n)
inline Tensor reduce_mean_hip(const Tensor& t) {
    int64_t n = t.numel();
    Tensor result({}, t.dtype(), t.device());
    if (t.dtype() == DType::Float32) {
        launch_full_reduction_mean<float>(t.data<float>(), result.data<float>(), n, nullptr);
    } else if (t.dtype() == DType::Float64) {
        launch_full_reduction_mean<double>(t.data<double>(), result.data<double>(), n, nullptr);
    }
    return result;
}

// HIP_CHECK is provided by rocm_error.hpp (pulled in via ../hip_buffer.hpp).
// The shared definition is functionally equivalent, so no local redefinition
// is needed here.

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
    // Float16/BFloat16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
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
    // Empty input/output: skip the launch. A zero-element grid makes HIP reject
    // the launch with "invalid configuration argument".
    if (total_elements == 0) {
        return output;
    }
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
    } else if (input.dtype() == DType::Float64) {
        const double* bias_ptr = bias ? bias->data<double>() : nullptr;
        hipLaunchKernelGGL(fused_linear_relu_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            weight.data<double>(),
            bias_ptr,
            output.data<double>(),
            batch_size,
            in_features,
            out_features,
            bias != nullptr
        );
    } else {
        throw std::runtime_error("fused_linear_relu_hip: unsupported dtype");
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

        T normalized = (input[idx] - mean[c]) * (T(1) / sqrt(var[c] + eps));
        T scaled = normalized * gamma[c] + beta[c];

        // ReLU
        output[idx] = (scaled > T(0)) ? scaled : T(0);
    }
}

auto fused_batchnorm_relu_hip(
    const Tensor& input_orig,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    // Contiguify: the kernel indexes input flat as NCHW, so a channels-last /
    // permuted view would map elements to the wrong channel (matches CPU).
    const Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
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

    // Find max (for numerical stability). Use the type-generic fmax/exp/log so
    // a Float64 instantiation computes in double instead of silently truncating
    // through the float-only fmaxf/expf/logf.
    T max_val = std::numeric_limits<T>::lowest();
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        max_val = fmax(max_val, row[i]);
    }

    // Block-wide max reduction
    shared_data[threadIdx.x] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = fmax(shared_data[threadIdx.x], shared_data[threadIdx.x + s]);
        }
        __syncthreads();
    }

    T global_max = shared_data[0];
    __syncthreads();

    // Compute sum(exp(x - max))
    T sum_exp = 0;
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        sum_exp += exp(row[i] - global_max);
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
        T log_sum_exp = log(shared_data[0]) + global_max;
        losses[b] = log_sum_exp - row[target];
    }
}

auto fused_softmax_cross_entropy_hip(
    const Tensor& logits,
    const Tensor& targets,
    const std::string& reduction
) -> Tensor {
    // Float16/BFloat16: widen to Float32, compute, return loss in the input
    // dtype. Float64 is computed NATIVELY (the kernel is now type-generic) so a
    // Float64 cross-entropy keeps full double precision instead of silently
    // collapsing to a Float32 loss, matching the LayerNorm Float64 fix.
    if (logits.dtype() == DType::Float16 || logits.dtype() == DType::BFloat16) {
        const DType orig = logits.dtype();
        auto logits_f32 = logits.to(DType::Float32);
        return fused_softmax_cross_entropy_hip(logits_f32, targets, reduction).to(orig);
    }

    int64_t batch_size = logits.shape()[0];
    int64_t num_classes = logits.shape()[1];

    // Validate target labels host-side: the kernel reads row[target] with no
    // bounds check, so an out-of-range label would read logits out of bounds.
    if (batch_size > 0) {
        Tensor t_host = targets.is_contiguous() ? targets : targets.contiguous();
        t_host = t_host.to(Device::cpu());
        const int64_t* tp = t_host.data<int64_t>();
        for (int64_t i = 0; i < batch_size; ++i) {
            if (tp[i] < 0 || tp[i] >= num_classes) {
                throw std::out_of_range("cross_entropy: target " + std::to_string(tp[i]) +
                    " out of range [0, " + std::to_string(num_classes) + ")");
            }
        }
    }

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
    } else if (logits.dtype() == DType::Float64) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_softmax_cross_entropy_kernel<double, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
            logits.data<double>(),
            targets.data<int64_t>(),
            losses.data<double>(),
            batch_size,
            num_classes
        );
    } else {
        throw std::runtime_error("fused_softmax_cross_entropy_hip: only Float32/Float64/Float16/BFloat16 supported");
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

// Grad-returning variant: writes per-row loss AND grad_logits = softmax - onehot.
template<typename T, int BLOCK_SIZE>
__global__ void fused_softmax_ce_grad_kernel(
    const T* logits,
    const int64_t* targets,
    T* losses,
    T* grad_logits,
    int64_t batch_size,
    int64_t num_classes
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* row = logits + b * num_classes;
    T* grad_row = grad_logits + b * num_classes;
    int64_t target = targets[b];

    __shared__ T shared_data[BLOCK_SIZE];

    // Block-wide max for numerical stability.
    T max_val = std::numeric_limits<T>::lowest();
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }
    shared_data[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + s]);
        __syncthreads();
    }
    T global_max = shared_data[0];
    __syncthreads();

    // Block-wide sum(exp(x - max)).
    T sum_exp = 0;
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        sum_exp += expf(row[i] - global_max);
    }
    shared_data[threadIdx.x] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        __syncthreads();
    }
    T total = shared_data[0];
    __syncthreads();

    // grad_i = softmax_i - [i == target]; per-sample loss = logsumexp - row[target].
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        T p = expf(row[i] - global_max) / total;
        grad_row[i] = p - (i == target ? T(1) : T(0));
    }
    if (threadIdx.x == 0) {
        losses[b] = (logf(total) + global_max) - row[target];
    }
}

// Returns {per-sample loss (batch,), grad_logits (batch, C)} for rank-2 inputs.
auto fused_softmax_cross_entropy_grad_hip(
    const Tensor& logits,
    const Tensor& targets
) -> std::pair<Tensor, Tensor> {
    if (logits.dtype() != DType::Float32) {
        throw std::runtime_error("fused_softmax_cross_entropy_grad_hip: Only Float32 supported");
    }
    int64_t batch_size = logits.shape()[0];
    int64_t num_classes = logits.shape()[1];

    Tensor losses = create_hip_zeros({batch_size}, DType::Float32, logits.device());
    Tensor grad = create_hip_zeros({batch_size, num_classes}, DType::Float32, logits.device());

    constexpr int BLOCK_SIZE = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(fused_softmax_ce_grad_kernel<float, BLOCK_SIZE>),
        dim3(batch_size), dim3(BLOCK_SIZE), 0, 0,
        logits.data<float>(),
        targets.data<int64_t>(),
        losses.data<float>(),
        grad.data<float>(),
        batch_size,
        num_classes
    );
    HIP_CHECK(hipGetLastError());
    return {losses, grad};
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

template<typename T>
__global__ void fused_add_relu_broadcast_kernel(
    const T* a,
    const T* b,
    T* output,
    const int64_t* strides_a,
    const int64_t* strides_b,
    const int64_t* output_shape,
    int64_t ndim,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t out_idx = tid; out_idx < n; out_idx += stride) {
        int64_t idx_a = 0;
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % output_shape[i];
            tmp /= output_shape[i];
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
        }

        T sum = a[idx_a] + b[idx_b];
        output[out_idx] = (sum > T(0)) ? sum : T(0);
    }
}

namespace detail_fused {

inline std::vector<int64_t> compute_broadcast_shape(
    const std::vector<int64_t>& shape_a,
    const std::vector<int64_t>& shape_b) {
    size_t max_ndim = std::max(shape_a.size(), shape_b.size());
    std::vector<int64_t> result(max_ndim);
    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = i < shape_a.size() ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = i < shape_b.size() ? shape_b[shape_b.size() - 1 - i] : 1;
        if (dim_a == dim_b || dim_a == 1 || dim_b == 1) {
            result[max_ndim - 1 - i] = std::max(dim_a, dim_b);
        } else {
            throw std::runtime_error("fused_add_relu: shapes are not broadcastable");
        }
    }
    return result;
}

inline std::vector<int64_t> compute_broadcast_strides(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& broadcast_shape) {
    std::vector<int64_t> strides(broadcast_shape.size(), 0);
    std::vector<int64_t> original_strides(shape.size());
    if (!shape.empty()) {
        original_strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            original_strides[i] = original_strides[i + 1] * shape[i + 1];
        }
    }
    int64_t offset = static_cast<int64_t>(broadcast_shape.size()) - static_cast<int64_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 1) {
            strides[offset + i] = 0;
        } else {
            strides[offset + i] = original_strides[i];
        }
    }
    return strides;
}

inline bool have_same_shape(const Tensor& a, const Tensor& b) {
    if (a.ndim() != b.ndim()) return false;
    auto sa = a.shape();
    auto sb = b.shape();
    for (size_t i = 0; i < sa.size(); ++i) {
        if (sa[i] != sb[i]) return false;
    }
    return true;
}

} // namespace detail_fused

auto fused_add_relu_hip(const Tensor& a_orig, const Tensor& b_orig) -> Tensor {
    // Contiguify both operands: the kernel reads a[i]+b[i] flat, so views with
    // differing physical layouts would be paired incorrectly (matches CPU).
    const Tensor a = a_orig.is_contiguous() ? a_orig : a_orig.contiguous();
    const Tensor b = b_orig.is_contiguous() ? b_orig : b_orig.contiguous();
    // Float16/BFloat16: upcast to Float32, compute, convert back
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        DType orig_dtype = a.dtype();
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto result = fused_add_relu_hip(a_f32, b_f32);
        return result.to(orig_dtype);
    }

    auto a_shape = to_vec(a.shape());
    auto b_shape = to_vec(b.shape());

    // Fast path: same shape, no broadcasting needed
    if (detail_fused::have_same_shape(a, b)) {
        Tensor result = create_hip_zeros(a_shape, a.dtype(), a.device());
        int64_t n = a.numel();
        int threads = 256;
        int blocks = std::min((int)((n + threads - 1) / threads), 65535);

        if (a.dtype() == DType::Float64) {
            hipLaunchKernelGGL(fused_add_relu_kernel<double>,
                dim3(blocks), dim3(threads), 0, 0,
                a.data<double>(), b.data<double>(), result.data<double>(), n);
        } else {
            hipLaunchKernelGGL(fused_add_relu_kernel<float>,
                dim3(blocks), dim3(threads), 0, 0,
                a.data<float>(), b.data<float>(), result.data<float>(), n);
        }
        HIP_CHECK(hipGetLastError());
        return result;
    }

    // Broadcasting path
    auto output_shape = detail_fused::compute_broadcast_shape(a_shape, b_shape);
    auto strides_a = detail_fused::compute_broadcast_strides(a_shape, output_shape);
    auto strides_b = detail_fused::compute_broadcast_strides(b_shape, output_shape);
    int64_t ndim = static_cast<int64_t>(output_shape.size());
    int64_t n = 1;
    for (auto d : output_shape) n *= d;

    Tensor result = create_hip_zeros(output_shape, a.dtype(), a.device());

    // Copy strides and shape to device. RAII buffers are freed on any exception
    // path (e.g. a throwing HIP_CHECK below) instead of leaking.
    size_t meta_bytes = ndim * sizeof(int64_t);
    HipBuffer strides_a_buf(meta_bytes);
    HipBuffer strides_b_buf(meta_bytes);
    HipBuffer output_shape_buf(meta_bytes);
    int64_t* d_strides_a = strides_a_buf.as<int64_t>();
    int64_t* d_strides_b = strides_b_buf.as<int64_t>();
    int64_t* d_output_shape = output_shape_buf.as<int64_t>();
    HIP_CHECK(hipMemcpy(d_strides_a, strides_a.data(), meta_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides_b, strides_b.data(), meta_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output_shape, output_shape.data(), meta_bytes, hipMemcpyHostToDevice));

    int threads = 256;
    int blocks = std::min((int)((n + threads - 1) / threads), 65535);

    if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(fused_add_relu_broadcast_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            a.data<double>(), b.data<double>(), result.data<double>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else {
        hipLaunchKernelGGL(fused_add_relu_broadcast_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            a.data<float>(), b.data<float>(), result.data<float>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    }
    HIP_CHECK(hipGetLastError());
    // Synchronize before the RAII metadata buffers are destroyed at scope exit,
    // since the kernel reads from them. Sync only the launch stream (0) rather
    // than hipDeviceSynchronize(), which would block the host on ALL device work
    // across every stream and serialize otherwise-independent GPU activity.
    HIP_CHECK(hipStreamSynchronize(0));

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
    // Exact erf GELU: 0.5 * x * (1 + erf(x / sqrt(2))) — matches the canonical
    // GELU and PyTorch default (approximate='none').
    constexpr T inv_sqrt2 = T(0.70710678118654752);

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        T x = input[i];
        output[i] = T(0.5) * x * (T(1.0) + erf(x * inv_sqrt2));
    }
}

auto fused_gelu_hip(const Tensor& input_orig) -> Tensor {
    // Contiguify: the kernel reads/writes input flat (matches the CPU kernel).
    const Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
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

    // Accumulate mean/variance stats in double to avoid catastrophic
    // cancellation, matching the CPU reference and the CUDA sibling
    // (fused_ops.cu forces Acc=double). The F16/BF16 paths widen to Float32 at
    // host level and instantiate this kernel with T=float, so a double
    // accumulator here covers them too; the T=double instantiation is exact.
    // Only the final mean/inv_std and the normalization are narrowed back to T.
    using Acc = double;
    __shared__ Acc shared_data[BLOCK_SIZE];

    // Compute mean (in double)
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

    // Compute variance (in double)
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
    Acc inv_std = rsqrt(variance + static_cast<Acc>(eps));

    // Normalize and scale (compute in double, narrow to T on store)
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        Acc normalized = (static_cast<Acc>(batch_in[i]) - mean) * inv_std;
        batch_out[i] = static_cast<T>(normalized * static_cast<Acc>(weight[i])
                                      + static_cast<Acc>(bias[i]));
    }
}

auto fused_layer_norm_hip(
    const Tensor& input_orig,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps,
    hipStream_t stream
) -> Tensor {
    // Contiguify: the kernel indexes input flat (input + b*norm_size), so a
    // non-contiguous view would read the wrong storage and corrupt the saved
    // mean/inv_std. Mirrors the CPU kernel; F16/BF16 contiguifies via .to().
    const Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    // Float16/BFloat16: upcast to Float32, compute, convert back. Float64
    // computes natively in double precision below — previously Float64 was
    // also routed through Float32, which silently dropped the input to
    // single precision and produced ~1e-7 absolute error vs CPU/CUDA's
    // native-FP64 LayerNorm.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto b_f32 = bias.to(DType::Float32);
        auto result = fused_layer_norm_hip(input_f32, normalized_shape, w_f32, b_f32, eps, stream);
        return result.to(orig_dtype);
    }

    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    Tensor output = create_hip_zeros(to_vec(input.shape()), input.dtype(), input.device(), stream);

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_layer_norm_kernel<float, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(),
            weight.data<float>(),
            bias.data<float>(),
            output.data<float>(),
            batch_size,
            norm_size,
            static_cast<float>(eps)
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_layer_norm_kernel<double, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(),
            weight.data<double>(),
            bias.data<double>(),
            output.data<double>(),
            batch_size,
            norm_size,
            static_cast<double>(eps)
        );
    } else {
        throw std::runtime_error("fused_layer_norm_hip: unsupported dtype " +
                                 std::string(dtype_name(input.dtype())));
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
        T normalized = (conv_output[idx] - mean[c]) * (T(1) / sqrt(var[c] + eps));
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
// Philox4x32-10 Counter-Based PRNG (HIP device port — same algorithm as CPU
// and CUDA implementations so within a single backend's forward/backward pair
// the dropout mask is bit-reproducible).
// ==============================================================================

__device__ __forceinline__ void philox_round_hip(uint32_t ctr[4], const uint32_t key[2]) {
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

__device__ __forceinline__ float philox_uniform_hip(uint32_t batch_head, uint32_t query_idx,
                                                     uint32_t kv_pos, uint32_t rng_seed) {
    uint32_t ctr[4] = {batch_head, query_idx, kv_pos, 0};
    uint32_t k[2] = {rng_seed, rng_seed ^ 0x1BD11BDAU};
    constexpr uint32_t W0 = 0x9E3779B9U;
    constexpr uint32_t W1 = 0xBB67AE85U;
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        philox_round_hip(ctr, k);
        if (r < 9) { k[0] += W0; k[1] += W1; }
    }
    return (static_cast<float>(ctr[0] >> 8)) * (1.0f / 16777216.0f);
}

// ==============================================================================
// Flash Attention v2 Forward HIP Kernel (Tiled, Memory-Efficient)
// ==============================================================================

/**
 * @brief Tiled Flash Attention v2 forward kernel for AMD GPUs
 *
 * Processes one query row per block, iterating over KV tiles.
 * Uses online softmax to avoid materializing the full NxN attention matrix.
 *
 * Grid: (batch_heads, seq_len_q)
 * Block: (BLOCK_SIZE) threads
 *
 * Shared memory layout:
 *   K_tile[Bc][K_STRIDE], V_tile[Bc][K_STRIDE],
 *   Q_shared[HEAD_DIM], scores_shared[Bc], reduce_buf[num_warps]
 *
 * Adapted from CUDA for AMD wavefront-64 architecture.
 * Uses shared-memory reductions (no warp shuffles) for portability across
 * GCN/CDNA/RDNA which have different wavefront sizes.
 */
template<int HEAD_DIM, int BLOCK_SIZE = 256>
__global__ void flash_attention_v2_kernel_hip(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    const int seq_len_q,
    const int seq_len_k,
    const float scale,
    const bool causal,        // applied before per-tile softmax max-subtract
    const float dropout_p,    // dropout probability; 0 disables
    const uint32_t rng_seed   // Philox seed; 0 disables dropout
) {
    const int batch_head = blockIdx.x;
    const int q_row = blockIdx.y;
    const int tid = threadIdx.x;

    if (q_row >= seq_len_q) return;

    constexpr int Bc = 32;  // KV tile size
    constexpr int K_STRIDE = HEAD_DIM + 4;  // Padding for bank conflict avoidance
    const int num_kv_blocks = (seq_len_k + Bc - 1) / Bc;
    const int num_warps = BLOCK_SIZE / 64;  // AMD wavefront = 64

    // Shared memory layout
    extern __shared__ float smem[];
    float* K_tile = smem;                              // [Bc][K_STRIDE]
    float* V_tile = smem + Bc * K_STRIDE;              // [Bc][K_STRIDE]
    float* Q_shared = smem + 2 * Bc * K_STRIDE;        // [HEAD_DIM]
    float* scores_shared = Q_shared + HEAD_DIM;        // [Bc]
    float* reduce_buf = scores_shared + Bc;            // [num_warps + 1]

    // Pointers for this batch/head
    const float* Q_row = Q + (batch_head * seq_len_q + q_row) * HEAD_DIM;
    const float* K_base = K + batch_head * seq_len_k * HEAD_DIM;
    const float* V_base = V + batch_head * seq_len_k * HEAD_DIM;
    float* O_row = O + (batch_head * seq_len_q + q_row) * HEAD_DIM;

    // Load query row into shared memory cooperatively
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        Q_shared[d] = Q_row[d];
    }
    __syncthreads();

    // Online softmax accumulators (per-thread output elements)
    constexpr int ELEMS_PER_THREAD = (HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float o_acc[ELEMS_PER_THREAD];
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) o_acc[e] = 0.0f;
    float m_prev = -1e30f;  // Running max
    float l_prev = 0.0f;    // Running sum of exp

    // Iterate over KV tiles
    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int kv_start = kv_block * Bc;
        const int kv_end_actual = (kv_start + Bc < seq_len_k) ? Bc : (seq_len_k - kv_start);

        // Skip tiles that are fully causally masked. The smallest key position in
        // this tile is kv_start; if even that exceeds q_row then every score in the
        // tile would be -INFINITY, yielding tile_max = -INF and expf(-INF-(-INF)) =
        // expf(NaN) = NaN. Since m_prev is finite (-1e30f) the online-softmax merge
        // produces rescale_tile = 0 and l_new = l_prev + NaN*0 = NaN (IEEE-754
        // NaN*0 = NaN), corrupting the whole output row and logsumexp. A fully
        // masked tile contributes nothing to the softmax, so skip it entirely.
        // All threads in the block evaluate the same condition, so this branch is
        // uniform and does not desync the __syncthreads() calls below.
        if (causal && kv_start > q_row) {
            continue;
        }

        // Cooperatively load K tile: K[kv_start:kv_start+Bc, :]
        for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            if (row < kv_end_actual) {
                K_tile[row * K_STRIDE + col] = K_base[(kv_start + row) * HEAD_DIM + col];
            } else {
                K_tile[row * K_STRIDE + col] = 0.0f;
            }
        }
        // Cooperatively load V tile
        for (int idx = tid; idx < Bc * HEAD_DIM; idx += BLOCK_SIZE) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            if (row < kv_end_actual) {
                V_tile[row * K_STRIDE + col] = V_base[(kv_start + row) * HEAD_DIM + col];
            } else {
                V_tile[row * K_STRIDE + col] = 0.0f;
            }
        }
        __syncthreads();

        // Compute attention scores: S[j] = Q @ K[j]^T * scale for j in [0, Bc).
        // Causal masking applied here so masked positions get a -INF score
        // that cleanly produces exp(-INF) = 0 in the softmax below — matches
        // the contract sentinel rule (audit C1 ROCm fix).
        for (int j = tid; j < Bc; j += BLOCK_SIZE) {
            float score;
            int kv_pos = kv_start + j;
            // F021: bottom-right causal alignment (matches the MHA/GQA manual
            // BMM path and PyTorch). Query at absolute position q_row attends to
            // keys kv_pos <= q_row + (seq_len_k - seq_len_q). Self-attention =>
            // offset 0; KV-cache cross-attention (seq_len_q < seq_len_k) lets the
            // query see all preceding keys instead of only key 0.
            if (j < kv_end_actual && !(causal && kv_pos > q_row + (seq_len_k - seq_len_q))) {
                score = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_shared[d] * K_tile[j * K_STRIDE + d];
                }
                score *= scale;
            } else {
                score = -INFINITY;  // Beyond seq_len_k OR causally masked
            }
            scores_shared[j] = score;
            // Stash the score in reduce_buf (Bc slots) so the exp pass can read
            // it back after the in-place max reduction clobbers scores_shared —
            // avoids recomputing the full Q·K dot product a second time.
            reduce_buf[j] = score;
        }
        __syncthreads();

        // Find tile_max via proper shared-memory tree reduction over scores_shared.
        // Per audit C3 ROCm: replaces the dead atomicMax block (was doing
        // atomicMax<int>(reinterpret_cast<int*>(&reduce_buf[0]), __float_as_int(local_max))
        // which gives wrong order for negative floats — and was followed by a
        // serial-in-tid==0 fallback that ignored the atomic result anyway).
        // The new tree reduction is wavefront-agnostic so it works on both
        // wave32 (RDNA) and wave64 (CDNA/MI200/MI300).
        //
        // scores_shared has Bc=32 entries — we use the first Bc threads to do
        // an in-place pairwise reduction. For Bc <= BLOCK_SIZE this is
        // single-pass; we don't need the reduce_buf for max anymore.
        for (int stride = Bc / 2; stride > 0; stride >>= 1) {
            if (tid < stride && tid + stride < Bc) {
                scores_shared[tid] = fmaxf(scores_shared[tid], scores_shared[tid + stride]);
            }
            __syncthreads();
        }
        float tile_max = scores_shared[0];
        __syncthreads();

        // Restore scores into scores_shared for the exp pass. The max reduction
        // destroyed the original values, but reduce_buf still holds the saved
        // scores from the score-compute loop above — a plain copy instead of a
        // second Q·K recompute (which would double the dominant score FLOP).
        for (int j = tid; j < Bc; j += BLOCK_SIZE) {
            scores_shared[j] = reduce_buf[j];
        }
        __syncthreads();

        // Compute exp(score - tile_max), apply Philox dropout (post-softmax,
        // inverted-scaled), and store P. Counter is (batch_head, q_row,
        // kv_pos, 0) so backward replays bit-exactly given the same seed.
        const bool apply_dropout = (dropout_p > 0.0f) && (rng_seed != 0u);
        const float dropout_scale = apply_dropout ? (1.0f / (1.0f - dropout_p)) : 1.0f;
        for (int j = tid; j < Bc; j += BLOCK_SIZE) {
            float exp_val = expf(scores_shared[j] - tile_max);
            if (apply_dropout) {
                int kv_pos = kv_start + j;
                float u = philox_uniform_hip(static_cast<uint32_t>(batch_head),
                                              static_cast<uint32_t>(q_row),
                                              static_cast<uint32_t>(kv_pos),
                                              rng_seed);
                if (u < dropout_p) {
                    exp_val = 0.0f;
                } else {
                    exp_val *= dropout_scale;
                }
            }
            scores_shared[j] = exp_val;
        }
        __syncthreads();

        // Sum reduction over scores_shared via in-place pairwise (wavefront-agnostic).
        // We use reduce_buf[0..Bc-1] as a scratch copy so scores_shared keeps the P values.
        for (int j = tid; j < Bc; j += BLOCK_SIZE) {
            reduce_buf[j] = scores_shared[j];
        }
        __syncthreads();
        for (int stride = Bc / 2; stride > 0; stride >>= 1) {
            if (tid < stride && tid + stride < Bc) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            __syncthreads();
        }
        float tile_sum = reduce_buf[0];
        __syncthreads();

        // Online softmax rescaling
        float m_new = fmaxf(m_prev, tile_max);
        float rescale_prev = expf(m_prev - m_new);
        float rescale_tile = expf(tile_max - m_new);
        float l_new = l_prev * rescale_prev + tile_sum * rescale_tile;

        // Rescale previous output accumulator and add P @ V contribution
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            int d = tid + e * BLOCK_SIZE;
            if (d < HEAD_DIM) {
                // Rescale previous accumulation
                o_acc[e] *= rescale_prev;
                // Accumulate P @ V for this dimension
                float pv = 0.0f;
                for (int j = 0; j < Bc; ++j) {
                    pv += scores_shared[j] * V_tile[j * K_STRIDE + d];
                }
                o_acc[e] += pv * rescale_tile;
            }
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    // Final normalization: O = o_acc / l_prev
    float l_inv = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        int d = tid + e * BLOCK_SIZE;
        if (d < HEAD_DIM) {
            O_row[d] = o_acc[e] * l_inv;
        }
    }

    // Write logsumexp if requested
    if (L != nullptr && tid == 0) {
        L[batch_head * seq_len_q + q_row] = m_prev + logf(l_prev + 1e-30f);
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
    const bool causal,
    const float dropout_p,    // dropout probability used in the forward; 0 disables
    const uint32_t rng_seed   // Philox seed used in the forward; 0 disables
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
            l_tile[row] = -__builtin_huge_valf();
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
                S_tile[idx] = -__builtin_huge_valf();
            }
        }
        __syncthreads();

        // Compute P_ij = exp(S_ij - l_i)  [Br x Bc]
        // Apply causal mask, then RE-APPLY the exact Philox dropout mask the
        // forward used (counter (batch_head, q_start+i, kv_start+j, seed) with
        // the same inverted 1/(1-p) scale) so dQ/dK/dV differentiate the DROPPED
        // attention weights. Without this the backward used undropped weights and
        // produced wrong gradients whenever dropout was active.
        const bool bwd_apply_dropout = (dropout_p > 0.0f) && (rng_seed != 0u);
        const float bwd_dropout_scale = bwd_apply_dropout ? (1.0f / (1.0f - dropout_p)) : 1.0f;
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            float p = 0.0f;
            if (i < actual_Br && j < actual_Bc) {
                if (causal && (q_start + i) < (kv_start + j)) {
                    p = 0.0f;
                } else {
                    p = expf(S_tile[i * Bc + j] - l_tile[i]);
                    if (bwd_apply_dropout) {
                        float u = philox_uniform_hip(static_cast<uint32_t>(batch_head),
                                                     static_cast<uint32_t>(q_start + i),
                                                     static_cast<uint32_t>(kv_start + j),
                                                     rng_seed);
                        if (u < dropout_p) p = 0.0f;
                        else p *= bwd_dropout_scale;
                    }
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
    bool causal,
    float dropout_p = 0.0f,      // forward dropout probability; 0 disables
    uint32_t rng_seed = 0u       // forward Philox seed; 0 disables
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
            auto result = flash_attention_backward_hip(dO_f32, Q_f32, K_f32, V_f32, O_f32, L, scale, causal, dropout_p, rng_seed);
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
            auto result = flash_attention_backward_hip(dO_f32, Q_f32, K_f32, V_f32, O_f32, L, scale, causal, dropout_p, rng_seed);
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
            seq_len_int, scale, causal, dropout_p, rng_seed);
        HIP_CHECK(hipGetLastError());
    } else if (head_dim == 64) {
        size_t smem = compute_bwd_smem(64);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(flash_attention_backward_kernel_hip<64, Br, Bc, BLOCK_SIZE>),
            grid, threads, smem, 0,
            q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_ptr, dk_ptr, dv_ptr,
            seq_len_int, scale, causal, dropout_p, rng_seed);
        HIP_CHECK(hipGetLastError());
    } else if (head_dim == 128) {
        size_t smem = compute_bwd_smem(128);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(flash_attention_backward_kernel_hip<128, Br, Bc, BLOCK_SIZE>),
            grid, threads, smem, 0,
            q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_ptr, dk_ptr, dv_ptr,
            seq_len_int, scale, causal, dropout_p, rng_seed);
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
    float scale,
    bool causal,         // M5: plumbed to flash_attention_v2_kernel_hip
    float dropout_p,     // M5-rem: dropout probability
    uint32_t rng_seed    // M5-rem: Philox seed; 0 disables dropout
) -> std::pair<Tensor, Tensor> {
    // Float64: use the native double-precision kernel for supported head dims so
    // the forward keeps full precision instead of downcasting to Float32 (audit
    // rocm-attn-01). Unsupported head dims / dropout fall through to the widen
    // path below. fused_attention_hip_f64 lives in flash_attention_f64.hip.cpp.
    if (Q.dtype() == DType::Float64 && dropout_p == 0.0f) {
        const int64_t head_dim = Q.shape()[Q.ndim() - 1];
        switch (head_dim) {
            case 16: case 32: case 48: case 64: case 80: case 96: case 128:
                return fused_attention_hip_f64(Q, K, V, static_cast<double>(scale),
                                               causal, /*stream=*/nullptr);
            default: break;
        }
    }

    // Non-Float32: upcast to Float32, compute, convert back
    if (Q.dtype() != DType::Float32) {
        DType orig_dtype = Q.dtype();
        auto q_f32 = Q.to(DType::Float32);
        auto k_f32 = K.to(DType::Float32);
        auto v_f32 = V.to(DType::Float32);
        auto [result, lse] = fused_attention_hip(q_f32, k_f32, v_f32, scale, causal, dropout_p, rng_seed);
        return {result.to(orig_dtype), lse};
    }

    int64_t batch_size = Q.shape()[0];
    int64_t seq_len = Q.shape()[1];
    int64_t d_k = Q.shape()[2];
    int64_t d_v = V.shape()[2];

    Tensor output = create_hip_zeros({batch_size, seq_len, d_v}, Q.dtype(), Q.device());
    Tensor lse = create_hip_zeros({batch_size, seq_len}, Q.dtype(), Q.device());

    constexpr int BLOCK_SIZE = 256;

    if (Q.dtype() != DType::Float32) {
        throw std::runtime_error("fused_attention_hip: Only Float32 supported");
    }

    // Use tiled Flash Attention v2 for supported head dimensions
    bool use_tiled = (d_k == 32 || d_k == 64 || d_k == 128) && d_k == d_v;

    if (use_tiled) {
        // Flash Attention v2: tiled, O(1) memory per query row
        constexpr int Bc = 32;
        int seq_len_int = static_cast<int>(seq_len);

        dim3 grid(static_cast<int>(batch_size), seq_len_int);
        dim3 threads(BLOCK_SIZE);

        // Shared memory layout (post-M5-rem reduction rewrite):
        //   K_tile[Bc * K_STRIDE] + V_tile[Bc * K_STRIDE] + Q_shared[HD]
        //   + scores_shared[Bc] + reduce_buf[Bc]
        // The new reduce_buf needs Bc entries for the in-place pairwise sum
        // reduction (was previously num_warps+1 — too small once we stopped
        // using the broken atomicMax-based reduction at line 1325).
        auto compute_fwd_smem = [&](int hd) -> size_t {
            int k_stride = hd + 4;
            return (2 * Bc * k_stride + hd + Bc + Bc) * sizeof(float);
        };

        const float* q_ptr = Q.data<float>();
        const float* k_ptr = K.data<float>();
        const float* v_ptr = V.data<float>();
        float* o_ptr = output.data<float>();
        float* l_ptr = lse.data<float>();

        if (d_k == 32) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_v2_kernel_hip<32, BLOCK_SIZE>),
                grid, threads, compute_fwd_smem(32), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, seq_len_int, seq_len_int, scale, causal, dropout_p, rng_seed);
        } else if (d_k == 64) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_v2_kernel_hip<64, BLOCK_SIZE>),
                grid, threads, compute_fwd_smem(64), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, seq_len_int, seq_len_int, scale, causal, dropout_p, rng_seed);
        } else if (d_k == 128) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(flash_attention_v2_kernel_hip<128, BLOCK_SIZE>),
                grid, threads, compute_fwd_smem(128), 0,
                q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, seq_len_int, seq_len_int, scale, causal, dropout_p, rng_seed);
        }
        HIP_CHECK(hipGetLastError());
    } else {
        // Composed-ops fallback for unsupported head_dim (d_k != {32,64,128} or
        // d_k != d_v). Replaces the previous naive fused_attention_kernel which
        // had the audit H6 bug class: shared_scores stale-write inside the
        // strided col loop, recomputed Q·K three times, and didn't bound the
        // sum reduction on seq_len. Uses OpId-based dispatch only (math.hpp
        // ops would collide with HIP device sqrt/exp in this TU).
        // Path: BMM(Q, K^T) * scale → softmax → BMM(P, V).
        NewOpAttributes empty;
        // K^T via Transpose on last two dims. ROCm Transpose dispatch reads
        // AttrKey::Dim0/Dim1 (not Dim/Dim2) — using the wrong key silently
        // defaulted to (0,1) and produced an incorrectly-permuted K_T,
        // breaking subsequent bmm with "Inner dimensions 8 != 2".
        NewOpAttributes tr_attrs;
        int64_t k_ndim = static_cast<int64_t>(K.shape().size());
        tr_attrs.set(AttrKey::Dim0, k_ndim - 2);
        tr_attrs.set(AttrKey::Dim1, k_ndim - 1);
        std::vector<Tensor> tr_in = {K};
        Tensor Kt = tenzor::dispatch(OpId::Transpose, tr_in, tr_attrs)[0];
        std::vector<Tensor> bmm_in = {Q, Kt};
        Tensor scores = tenzor::dispatch(OpId::Bmm, bmm_in, empty)[0];
        std::vector<int64_t> sshape(scores.shape().begin(), scores.shape().end());
        Tensor scale_t = tenzor::full(sshape, static_cast<double>(scale),
                                       scores.dtype(), scores.device());
        std::vector<Tensor> mul_in = {scores, scale_t};
        scores = tenzor::dispatch(OpId::Mul, mul_in, empty)[0];
        if (causal) {
            int64_t sl = sshape[sshape.size() - 1];
            Tensor rows = tenzor::arange(0, sl, 1, DType::Int64, scores.device());
            Tensor cols = tenzor::arange(0, sl, 1, DType::Int64, scores.device());
            std::vector<int64_t> rshape{sl, 1};
            std::vector<int64_t> cshape{1, sl};
            Tensor rows_2d = tenzor::reshape(rows, rshape);
            Tensor cols_2d = tenzor::reshape(cols, cshape);
            Tensor rows_f = rows_2d.to(DType::Float32);
            Tensor cols_f = cols_2d.to(DType::Float32);
            std::vector<Tensor> gt_in = {cols_f, rows_f};
            Tensor cmask = tenzor::dispatch(OpId::Gt, gt_in, empty)[0];
            // Use a FINITE large-negative sentinel (-1e30) rather than -inf: the
            // additive mask is built as cmask * sentinel, so at KEPT positions
            // (cmask==0) the product is 0*(-1e30)=0 — a plain -inf would give
            // 0*-inf=NaN and poison the whole softmax row. -1e30 still drives
            // softmax to ~0 at masked positions. Mirrors the ROCm Flash/Flex
            // registry paths and the CPU/OneAPI where()-based masking.
            Tensor neg_big = tenzor::full(sshape,
                -1e30,
                scores.dtype(), scores.device());
            Tensor cmask_t = cmask.to(scores.dtype());
            std::vector<Tensor> mul2_in = {cmask_t, neg_big};
            Tensor mask_v = tenzor::dispatch(OpId::Mul, mul2_in, empty)[0];
            std::vector<Tensor> add_in = {scores, mask_v};
            scores = tenzor::dispatch(OpId::Add, add_in, empty)[0];
        }
        NewOpAttributes sm_attrs;
        sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
        std::vector<Tensor> sm_in = {scores};
        Tensor probs = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];
        std::vector<Tensor> bmm2_in = {probs, V};
        output = tenzor::dispatch(OpId::Bmm, bmm2_in, empty)[0];
        // LSE not surfaced by the composed path; backward uses
        // function_attention.cpp's composed_attention_backward fallback when
        // L is missing.
        lse = Tensor{};
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

    // Accumulate sum-of-squares in double for T=float (matching the CUDA
    // sibling, which uses `double sum_sq`); a float accumulator over norm_size
    // drifts. For T=double this is exact anyway.
    __shared__ double shared_data[BLOCK_SIZE];

    // Compute sum of squares (in double)
    double sum_sq = 0.0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        double val = static_cast<double>(batch_in[i]);
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

    // Compute reciprocal RMS. Use the double-precision reciprocal-sqrt for both
    // T=float and T=double (rsqrtf would truncate the reciprocal-RMS to single
    // precision and diverge from the CPU/CUDA reference), then narrow to T.
    __shared__ T shared_rrms;
    if (threadIdx.x == 0) {
        double mean_sq = shared_data[0] / static_cast<double>(norm_size);
        shared_rrms = static_cast<T>(rsqrt(mean_sq + static_cast<double>(eps)));
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
    const Tensor& input_orig,
    const Tensor& weight,
    float eps,
    hipStream_t stream
) -> std::pair<Tensor, Tensor> {
    // Contiguify: the kernel indexes input flat (input + b*norm_size), so a
    // non-contiguous residual view would read the wrong storage and corrupt the
    // saved rrms. Mirrors the CPU kernel; the F16/BF16 path contiguifies via .to().
    const Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    // Float16/BFloat16: upcast to Float32, compute, convert back. Float64 is
    // computed natively below — downcasting it to Float32 would lose precision
    // and diverge from the CPU/CUDA native-FP64 path.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto [result, rrms] = fused_rms_norm_hip(input_f32, weight_f32, eps, stream);
        return {result.to(orig_dtype), rrms};
    }

    auto shape = input.shape();
    int64_t norm_size = shape[shape.size() - 1];

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }

    Tensor output = create_hip_zeros(to_vec(input.shape()), input.dtype(), input.device(), stream);
    Tensor rrms = create_hip_zeros({batch_size}, input.dtype(), input.device(), stream);

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_rms_norm_kernel<float, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(),
            weight.data<float>(),
            output.data<float>(),
            rrms.data<float>(),
            batch_size,
            norm_size,
            eps
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_rms_norm_kernel<double, BLOCK_SIZE>),
            dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(),
            weight.data<double>(),
            output.data<double>(),
            rrms.data<double>(),
            batch_size,
            norm_size,
            eps
        );
    } else {
        throw std::runtime_error("fused_rms_norm_hip: Only Float32/Float64 supported (Float16/BFloat16 widen above)");
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
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel; idx += stride) {
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
    // Float16/BFloat16: upcast to Float32, compute, convert back
    if (param.dtype() == DType::Float16 || param.dtype() == DType::BFloat16) {
        DType orig_dtype = param.dtype();
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
        param = param_f32.to(orig_dtype);
        if (momentum_buffer) *momentum_buffer = mom_f32.to(orig_dtype);
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
    double step_size = lr / bias_correction1;
    double bc2_inv = 1.0 / bias_correction2;

    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel; idx += stride) {
        T g = grad[idx];
        T p = param[idx];
        T m = exp_avg[idx];
        T v = exp_avg_sq[idx];

        // L2 regularization (added to grad)
        if (weight_decay > 0.0 && !decoupled_weight_decay) {
            g = g + T(weight_decay) * p;
        }

        // Update biased first moment estimate
        m = T(beta1) * m + T(1.0 - beta1) * g;

        // Update biased second raw moment estimate
        v = T(beta2) * v + T(1.0 - beta2) * g * g;

        // Bias-corrected second moment. AMSGrad tracks the running maximum over
        // the RAW (un-bias-corrected) second moment and applies bias correction
        // AFTER, matching the CPU reference and PyTorch (maxing the already
        // bias-corrected v_hat is wrong because bias_correction2 grows toward 1).
        T v_hat;
        if (amsgrad && max_exp_avg_sq) {
            T max_v = max_exp_avg_sq[idx];
            if (v > max_v) max_v = v;
            max_exp_avg_sq[idx] = max_v;
            v_hat = max_v * T(bc2_inv);
        } else {
            v_hat = v * T(bc2_inv);
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
    // Float16/BFloat16: upcast to Float32, compute, convert back
    if (param.dtype() == DType::Float16 || param.dtype() == DType::BFloat16) {
        DType orig_dtype = param.dtype();
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
        param = param_f32.to(orig_dtype);
        exp_avg = ea_f32.to(orig_dtype);
        exp_avg_sq = eas_f32.to(orig_dtype);
        if (max_exp_avg_sq) *max_exp_avg_sq = meas_f32.to(orig_dtype);
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

    // Compute the current learning rate in T so the Float64 instantiation keeps
    // full precision (a float local would truncate the FP64 lr path back to single
    // precision, diverging from native-FP64 Adagrad).
    T clr = T(lr) / (T(1) + T(step - 1) * T(lr_decay));

    T sq = sum_sq[idx] + g * g;
    sum_sq[idx] = sq;

    param[idx] = param[idx] - clr * g / (sqrt(sq) + T(eps));
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
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel; idx += stride) {
        T g = grad[idx];
        T p = param[idx];
        T m = exp_avg[idx];
        T v = exp_avg_sq[idx];

        m = T(beta1) * m + T(1.0f - beta1) * g;
        v = T(beta2) * v + T(1.0f - beta2) * g * g;

        T m_hat = m / T(bias_correction1);
        // AMSGrad tracks the running maximum over the RAW second moment and
        // applies bias correction AFTER, matching the CPU reference / PyTorch.
        T v_hat;
        if (amsgrad && max_exp_avg_sq != nullptr) {
            T max_v = max_exp_avg_sq[idx];
            if (v > max_v) max_v = v;
            max_exp_avg_sq[idx] = max_v;
            v_hat = max_v / T(bias_correction2);
        } else {
            v_hat = v / T(bias_correction2);
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
    // Float16/BFloat16: upcast to Float32, compute, convert back
    if (param.dtype() == DType::Float16 || param.dtype() == DType::BFloat16) {
        DType orig_dtype = param.dtype();
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
        param = param_f32.to(orig_dtype);
        exp_avg = ea_f32.to(orig_dtype);
        exp_avg_sq = eas_f32.to(orig_dtype);
        if (max_exp_avg_sq) *max_exp_avg_sq = meas_f32.to(orig_dtype);
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
    // Half precision (BFloat16/Float16): upcast to Float32, compute, convert
    // back. The native kernel only has Float32/Float64 instantiations, and the
    // autograd layer narrows every saved tensor — rrms included — to the input
    // dtype before dispatch, so a half rrms would otherwise hit the
    // unsupported-dtype throw and silently drop the RMSNorm gradient.
    if (input.dtype() == DType::BFloat16 || input.dtype() == DType::Float16) {
        DType orig = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto rrms_f32 = rrms.to(DType::Float32);
        auto [gi, gw] = fused_rms_norm_backward_hip(go_f32, input_f32, w_f32, rrms_f32);
        return {gi.to(orig), gw.to(orig)};
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

    // Accumulate the two grad dot-products in double (the forward stats are in
    // double), matching the CUDA sibling — shared buffers and locals are double.
    __shared__ double shared_sum1[BLOCK_SZ];  // For sum(grad_out * weight)
    __shared__ double shared_sum2[BLOCK_SZ];  // For sum(grad_out * weight * normalized)

    // Compute sums needed for input gradient (accumulate in double)
    double sum_grad_out = 0.0;
    double sum_grad_out_normalized = 0.0;

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        T grad_out_weighted = batch_grad_out[i] * weight[i];

        sum_grad_out += static_cast<double>(grad_out_weighted);
        sum_grad_out_normalized += static_cast<double>(grad_out_weighted) * static_cast<double>(normalized);

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

    double mean_grad_out = shared_sum1[0] / static_cast<double>(norm_size);
    double mean_grad_out_normalized = shared_sum2[0] / static_cast<double>(norm_size);

    // Compute input gradients (in double, narrow to T on store)
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        T grad_out_weighted = batch_grad_out[i] * weight[i];

        batch_grad_in[i] = static_cast<T>((static_cast<double>(grad_out_weighted) - mean_grad_out -
                           static_cast<double>(normalized) * mean_grad_out_normalized) * static_cast<double>(batch_inv_std));
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
