/**
 * @file comparison.hip.cpp
 * @brief HIP comparison kernels for AMD GPUs
 *
 * Implements element-wise comparison operations (Eq, Ne, Lt, Le, Gt, Ge) using HIP.
 * All operations return Boolean tensors.
 */

#include "tenzor/core/tensor.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdexcept>
#include <algorithm>

namespace tenzor {
namespace rocm {

// Helper to compare tensor shapes (spans)
inline bool shapes_equal(std::span<const int64_t> a, std::span<const int64_t> b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
}

// Broadcast two shapes to a common shape per numpy rules, or return empty if
// incompatible.
inline std::vector<int64_t> broadcast_shape(
    std::span<const int64_t> a, std::span<const int64_t> b) {
    std::vector<int64_t> out;
    auto ai = a.rbegin(); auto bi = b.rbegin();
    while (ai != a.rend() || bi != b.rend()) {
        int64_t da = (ai != a.rend()) ? *ai : 1;
        int64_t db = (bi != b.rend()) ? *bi : 1;
        if (da != db && da != 1 && db != 1) return {};
        out.push_back(std::max(da, db));
        if (ai != a.rend()) ++ai;
        if (bi != b.rend()) ++bi;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// Materialize a broadcast of `t` to `target_shape` as a contiguous Tensor.
// Uses Tensor::expand + .contiguous() which routes through the regular
// contiguous op (already fixed for UAF on ROCm).
inline Tensor broadcast_to_shape(const Tensor& t,
                                  const std::vector<int64_t>& target_shape) {
    if (shapes_equal(t.shape(), std::span<const int64_t>(target_shape))) {
        return t.is_contiguous() ? t : t.contiguous();
    }
    return t.expand(target_shape).contiguous();
}

// HIP Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err) \
            ); \
        } \
    } while(0)

// Grid-stride loop for HIP kernels
#define HIP_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// ==============================================================================
// Comparison Kernels
// ==============================================================================

template<typename T>
__global__ void eq_kernel(const T* a, const T* b, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] == b[idx]);
    }
}

template<typename T>
__global__ void ne_kernel(const T* a, const T* b, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] != b[idx]);
    }
}

template<typename T>
__global__ void lt_kernel(const T* a, const T* b, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] < b[idx]);
    }
}

template<typename T>
__global__ void le_kernel(const T* a, const T* b, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] <= b[idx]);
    }
}

template<typename T>
__global__ void gt_kernel(const T* a, const T* b, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] > b[idx]);
    }
}

template<typename T>
__global__ void ge_kernel(const T* a, const T* b, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] >= b[idx]);
    }
}

// Scalar comparison kernels
template<typename T>
__global__ void eq_scalar_kernel(const T* a, T scalar, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] == scalar);
    }
}

template<typename T>
__global__ void ne_scalar_kernel(const T* a, T scalar, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] != scalar);
    }
}

template<typename T>
__global__ void lt_scalar_kernel(const T* a, T scalar, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] < scalar);
    }
}

template<typename T>
__global__ void le_scalar_kernel(const T* a, T scalar, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] <= scalar);
    }
}

template<typename T>
__global__ void gt_scalar_kernel(const T* a, T scalar, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] > scalar);
    }
}

template<typename T>
__global__ void ge_scalar_kernel(const T* a, T scalar, bool* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] >= scalar);
    }
}

// ==============================================================================
// Broadcast comparison kernels
// ==============================================================================

template<typename T>
__global__ void eq_broadcast_kernel(
    const T* a, const T* b, bool* output,
    const int64_t* a_strides, const int64_t* b_strides,
    const int64_t* out_shape, int64_t ndim, int64_t n
) {
    HIP_KERNEL_LOOP(idx, n) {
        // Convert linear index to multi-dimensional indices
        int64_t tmp = idx;
        int64_t a_offset = 0;
        int64_t b_offset = 0;

        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t coord = tmp % out_shape[d];
            tmp /= out_shape[d];
            a_offset += coord * a_strides[d];
            b_offset += coord * b_strides[d];
        }

        output[idx] = (a[a_offset] == b[b_offset]);
    }
}

// ==============================================================================
// Public API - Tensor vs Tensor comparisons
// ==============================================================================

auto eq_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    if (a_in.dtype() != b_in.dtype()) {
        throw std::runtime_error("eq_kernel: tensor dtypes must match");
    }
    // Broadcast to common shape so callers can pass e.g. (N,) == (1,)
    // without special-casing at every call site. Previously this threw for
    // any shape mismatch, which broke MoE's eq(idx_col, expert_scalar).
    Tensor a = a_in;
    Tensor b = b_in;
    if (!shapes_equal(a.shape(), b.shape())) {
        auto shape = broadcast_shape(a.shape(), b.shape());
        if (shape.empty()) {
            throw std::runtime_error("eq_kernel: tensor shapes not broadcastable");
        }
        a = broadcast_to_shape(a_in, shape);
        b = broadcast_to_shape(b_in, shape);
    } else {
        if (!a.is_contiguous()) a = a.contiguous();
        if (!b.is_contiguous()) b = b.contiguous();
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()), DType::Bool, a.device());
    int64_t n = a.numel();

    if (n == 0) return output;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    switch (a.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(eq_kernel<float>, dim3(blocks), dim3(threads), 0, stream,
                a.data<float>(), b.data<float>(), output.data<bool>(), n);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(eq_kernel<double>, dim3(blocks), dim3(threads), 0, stream,
                a.data<double>(), b.data<double>(), output.data<bool>(), n);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(eq_kernel<int32_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), output.data<bool>(), n);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(eq_kernel<int64_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), output.data<bool>(), n);
            break;
        case DType::Bool:
            hipLaunchKernelGGL(eq_kernel<bool>, dim3(blocks), dim3(threads), 0, stream,
                a.data<bool>(), b.data<bool>(), output.data<bool>(), n);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(eq_kernel<__half>, dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<const __half*>(a.data<Float16>()), reinterpret_cast<const __half*>(b.data<Float16>()), output.data<bool>(), n);
            break;
        case DType::BFloat16: {
            auto a_f32 = a.to(DType::Float32);
            auto b_f32 = b.to(DType::Float32);
            return eq_kernel(a_f32, b_f32, stream);
        }
        default:
            throw std::runtime_error("eq_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto ne_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    if (a_in.dtype() != b_in.dtype()) {
        throw std::runtime_error("ne_kernel: tensor dtypes must match");
    }
    Tensor a = a_in; Tensor b = b_in;
    if (!shapes_equal(a.shape(), b.shape())) {
        auto shape = broadcast_shape(a.shape(), b.shape());
        if (shape.empty()) throw std::runtime_error("ne_kernel: tensor shapes not broadcastable");
        a = broadcast_to_shape(a_in, shape);
        b = broadcast_to_shape(b_in, shape);
    } else {
        if (!a.is_contiguous()) a = a.contiguous();
        if (!b.is_contiguous()) b = b.contiguous();
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()), DType::Bool, a.device());
    int64_t n = a.numel();

    if (n == 0) return output;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    switch (a.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(ne_kernel<float>, dim3(blocks), dim3(threads), 0, stream,
                a.data<float>(), b.data<float>(), output.data<bool>(), n);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(ne_kernel<double>, dim3(blocks), dim3(threads), 0, stream,
                a.data<double>(), b.data<double>(), output.data<bool>(), n);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(ne_kernel<int32_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), output.data<bool>(), n);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(ne_kernel<int64_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), output.data<bool>(), n);
            break;
        case DType::Bool:
            hipLaunchKernelGGL(ne_kernel<bool>, dim3(blocks), dim3(threads), 0, stream,
                a.data<bool>(), b.data<bool>(), output.data<bool>(), n);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(ne_kernel<__half>, dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<const __half*>(a.data<Float16>()), reinterpret_cast<const __half*>(b.data<Float16>()), output.data<bool>(), n);
            break;
        case DType::BFloat16: {
            auto a_f32 = a.to(DType::Float32);
            auto b_f32 = b.to(DType::Float32);
            return ne_kernel(a_f32, b_f32, stream);
        }
        default:
            throw std::runtime_error("ne_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto lt_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    Tensor a = a_in; Tensor b = b_in;
    if (!shapes_equal(a.shape(), b.shape())) {
        auto shape = broadcast_shape(a.shape(), b.shape());
        if (shape.empty()) throw std::runtime_error("lt_kernel: tensor shapes not broadcastable");
        a = broadcast_to_shape(a_in, shape);
        b = broadcast_to_shape(b_in, shape);
    } else {
        if (!a.is_contiguous()) a = a.contiguous();
        if (!b.is_contiguous()) b = b.contiguous();
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("lt_kernel: tensor dtypes must match");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()), DType::Bool, a.device());
    int64_t n = a.numel();

    if (n == 0) return output;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    switch (a.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(lt_kernel<float>, dim3(blocks), dim3(threads), 0, stream,
                a.data<float>(), b.data<float>(), output.data<bool>(), n);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(lt_kernel<double>, dim3(blocks), dim3(threads), 0, stream,
                a.data<double>(), b.data<double>(), output.data<bool>(), n);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(lt_kernel<int32_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), output.data<bool>(), n);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(lt_kernel<int64_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), output.data<bool>(), n);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(lt_kernel<__half>, dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<const __half*>(a.data<Float16>()), reinterpret_cast<const __half*>(b.data<Float16>()), output.data<bool>(), n);
            break;
        case DType::BFloat16: {
            auto a_f32 = a.to(DType::Float32);
            auto b_f32 = b.to(DType::Float32);
            return lt_kernel(a_f32, b_f32, stream);
        }
        default:
            throw std::runtime_error("lt_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto le_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    Tensor a = a_in; Tensor b = b_in;
    if (!shapes_equal(a.shape(), b.shape())) {
        auto shape = broadcast_shape(a.shape(), b.shape());
        if (shape.empty()) throw std::runtime_error("le_kernel: tensor shapes not broadcastable");
        a = broadcast_to_shape(a_in, shape);
        b = broadcast_to_shape(b_in, shape);
    } else {
        if (!a.is_contiguous()) a = a.contiguous();
        if (!b.is_contiguous()) b = b.contiguous();
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("le_kernel: tensor dtypes must match");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()), DType::Bool, a.device());
    int64_t n = a.numel();

    if (n == 0) return output;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    switch (a.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(le_kernel<float>, dim3(blocks), dim3(threads), 0, stream,
                a.data<float>(), b.data<float>(), output.data<bool>(), n);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(le_kernel<double>, dim3(blocks), dim3(threads), 0, stream,
                a.data<double>(), b.data<double>(), output.data<bool>(), n);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(le_kernel<int32_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), output.data<bool>(), n);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(le_kernel<int64_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), output.data<bool>(), n);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(le_kernel<__half>, dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<const __half*>(a.data<Float16>()), reinterpret_cast<const __half*>(b.data<Float16>()), output.data<bool>(), n);
            break;
        case DType::BFloat16: {
            auto a_f32 = a.to(DType::Float32);
            auto b_f32 = b.to(DType::Float32);
            return le_kernel(a_f32, b_f32, stream);
        }
        default:
            throw std::runtime_error("le_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto gt_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    Tensor a = a_in; Tensor b = b_in;
    if (!shapes_equal(a.shape(), b.shape())) {
        auto shape = broadcast_shape(a.shape(), b.shape());
        if (shape.empty()) throw std::runtime_error("gt_kernel: tensor shapes not broadcastable");
        a = broadcast_to_shape(a_in, shape);
        b = broadcast_to_shape(b_in, shape);
    } else {
        if (!a.is_contiguous()) a = a.contiguous();
        if (!b.is_contiguous()) b = b.contiguous();
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("gt_kernel: tensor dtypes must match");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()), DType::Bool, a.device());
    int64_t n = a.numel();

    if (n == 0) return output;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    switch (a.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(gt_kernel<float>, dim3(blocks), dim3(threads), 0, stream,
                a.data<float>(), b.data<float>(), output.data<bool>(), n);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(gt_kernel<double>, dim3(blocks), dim3(threads), 0, stream,
                a.data<double>(), b.data<double>(), output.data<bool>(), n);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(gt_kernel<int32_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), output.data<bool>(), n);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(gt_kernel<int64_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), output.data<bool>(), n);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(gt_kernel<__half>, dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<const __half*>(a.data<Float16>()), reinterpret_cast<const __half*>(b.data<Float16>()), output.data<bool>(), n);
            break;
        case DType::BFloat16: {
            auto a_f32 = a.to(DType::Float32);
            auto b_f32 = b.to(DType::Float32);
            return gt_kernel(a_f32, b_f32, stream);
        }
        default:
            throw std::runtime_error("gt_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto ge_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    Tensor a = a_in; Tensor b = b_in;
    if (!shapes_equal(a.shape(), b.shape())) {
        auto shape = broadcast_shape(a.shape(), b.shape());
        if (shape.empty()) throw std::runtime_error("ge_kernel: tensor shapes not broadcastable");
        a = broadcast_to_shape(a_in, shape);
        b = broadcast_to_shape(b_in, shape);
    } else {
        if (!a.is_contiguous()) a = a.contiguous();
        if (!b.is_contiguous()) b = b.contiguous();
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("ge_kernel: tensor dtypes must match");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()), DType::Bool, a.device());
    int64_t n = a.numel();

    if (n == 0) return output;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    switch (a.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(ge_kernel<float>, dim3(blocks), dim3(threads), 0, stream,
                a.data<float>(), b.data<float>(), output.data<bool>(), n);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(ge_kernel<double>, dim3(blocks), dim3(threads), 0, stream,
                a.data<double>(), b.data<double>(), output.data<bool>(), n);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(ge_kernel<int32_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), output.data<bool>(), n);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(ge_kernel<int64_t>, dim3(blocks), dim3(threads), 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), output.data<bool>(), n);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(ge_kernel<__half>, dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<const __half*>(a.data<Float16>()), reinterpret_cast<const __half*>(b.data<Float16>()), output.data<bool>(), n);
            break;
        case DType::BFloat16: {
            auto a_f32 = a.to(DType::Float32);
            auto b_f32 = b.to(DType::Float32);
            return ge_kernel(a_f32, b_f32, stream);
        }
        default:
            throw std::runtime_error("ge_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

} // namespace rocm
} // namespace tenzor
