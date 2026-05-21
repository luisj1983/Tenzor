#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "tenzor/backend/backend.hpp"  // For OpAttributes (dispatch wrappers)
#include "tenzor/ops/creation.hpp"     // For tenzor::get_global_seed
#include "cuda_common.cuh"
#include "cuda_launch_utils.cuh"
#include "launch_config.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cuComplex.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <charconv>  // For std::from_chars (dispatch wrappers)
#include <span>      // For std::span (dispatch wrappers)
#include "../cuda_stream_pool.hpp"

namespace tenzor {
namespace cuda {

// ============================================================================
// FP16/BF16 Conversion Functions
// ============================================================================

// Convert Tenzor Float16 to CUDA __half
__device__ __host__ inline __half to_cuda_half(const Float16& x) {
    __half_raw raw;
    raw.x = x.bits;
    return __half(raw);
}

// Convert CUDA __half to Tenzor Float16
__device__ __host__ inline Float16 from_cuda_half(const __half& x) {
    return Float16(__half_as_ushort(x));
}

// Convert Tenzor BFloat16 to CUDA __nv_bfloat16
__device__ __host__ inline __nv_bfloat16 to_cuda_bfloat16(const BFloat16& x) {
    __nv_bfloat16_raw raw;
    raw.x = x.bits;
    return __nv_bfloat16(raw);
}

// Convert CUDA __nv_bfloat16 to Tenzor BFloat16
__device__ __host__ inline BFloat16 from_cuda_bfloat16(const __nv_bfloat16& x) {
    return BFloat16(__bfloat16_as_ushort(x));
}

// CUDA error checking — use the canonical macro from cuda_error.hpp
#include "../cuda_error.hpp"

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

// Uses cudaOccupancyMaxPotentialBlockSize via a representative kernel
// (fill_kernel_device<float>) to determine an architecture-optimal block
// size instead of the previous hardcoded 256.  All 1-D grid-stride loop
// kernels in this file have similar register usage, so using a single
// representative kernel for the occupancy query is safe and avoids per-
// call-site changes.
//
// The forward declaration of fill_kernel_device<float> lives below (~line
// 2350); because this is an inline function used only in host code, the
// linker resolves the symbol just fine.
template<typename T>
__global__ void fill_kernel_device(T* output, T value, int64_t n);

// compute_launch_config_1d() and OCCUPANCY_CONFIG are now in
// cuda_launch_utils.cuh (shared across all CUDA kernel files).

// ============================================================================
// Division by Zero Check (for integer types)
// ============================================================================

// Kernel to check if any element is zero (for integer division check)
template<typename T>
__global__ void check_for_zeros_kernel(const T* data, int64_t n, int* has_zero) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        if (data[idx] == T(0)) {
            atomicExch(has_zero, 1);
        }
    }
}

// RAII wrapper for the persistent thread-local device flag used by
// check_integer_divisor_for_zeros.  Ensures cudaFree is called when the
// thread exits, preventing a device-memory leak.
struct DeviceFlagHolder {
    int* ptr = nullptr;

    DeviceFlagHolder() {
        // Allocation deferred to first use (see get())
    }

    int* get() {
        if (!ptr) {
            CUDA_CHECK(cudaMalloc(&ptr, sizeof(int)));
        }
        return ptr;
    }

    ~DeviceFlagHolder() {
        if (ptr) {
            // Best-effort free — if the CUDA context is already torn down
            // (e.g. during process exit), cudaFree may return an error;
            // we silently ignore it to avoid throwing from a destructor.
            (void)cudaFree(ptr);
            ptr = nullptr;
        }
    }

    DeviceFlagHolder(const DeviceFlagHolder&) = delete;
    DeviceFlagHolder& operator=(const DeviceFlagHolder&) = delete;
};

// Host function to check for zeros in an integer tensor
// Only enabled in debug builds to avoid the stream sync penalty in release.
// Uses a persistent thread-local device flag to avoid per-call cudaMalloc
// and cudaFree overhead.  The flag is reset via cudaMemsetAsync before each
// check and only copied back (D2H) when we actually need to inspect the result.
// The DeviceFlagHolder RAII wrapper ensures the allocation is freed on thread exit.
template<typename T>
inline void check_integer_divisor_for_zeros(const T* data, int64_t n, cudaStream_t stream) {
#ifndef TENZOR_SKIP_INTEGER_DIV_CHECK
    // Persistent thread-local device flag — allocated once per thread, freed on thread exit.
    static thread_local DeviceFlagHolder flag_holder;
    int* d_flag = flag_holder.get();

    // Reset flag to 0 asynchronously on the current stream
    CUDA_CHECK(cudaMemsetAsync(d_flag, 0, sizeof(int), stream));

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    check_for_zeros_kernel<<<grid, block, 0, stream>>>(data, n, d_flag);
    CUDA_CHECK(cudaGetLastError());

    // Synchronize and read back the flag
    int h_flag = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (h_flag != 0) {
        throw std::runtime_error("Integer division by zero");
    }
#else
    (void)data; (void)n; (void)stream;
#endif
}

// ============================================================================
// Broadcasting Helpers (Device-side)
// ============================================================================

// Device function to check if shapes are broadcastable
__device__ inline bool are_broadcastable_device(const int64_t* shape_a, int64_t ndim_a,
                                                 const int64_t* shape_b, int64_t ndim_b) {
    int64_t max_ndim = max(ndim_a, ndim_b);

    for (int64_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = i < ndim_a ? shape_a[ndim_a - 1 - i] : 1;
        int64_t dim_b = i < ndim_b ? shape_b[ndim_b - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }

    return true;
}

// Host-side broadcasting helpers
namespace detail {

// Check if two shapes are broadcastable
inline bool are_broadcastable(const std::vector<int64_t>& shape_a,
                               const std::vector<int64_t>& shape_b) {
    size_t max_ndim = std::max(shape_a.size(), shape_b.size());

    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = i < shape_a.size() ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = i < shape_b.size() ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }

    return true;
}

// Compute the broadcasted output shape
inline std::vector<int64_t> compute_broadcast_shape(const std::vector<int64_t>& shape_a,
                                                     const std::vector<int64_t>& shape_b) {
    size_t max_ndim = std::max(shape_a.size(), shape_b.size());
    std::vector<int64_t> result(max_ndim);

    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = i < shape_a.size() ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = i < shape_b.size() ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a == dim_b || dim_a == 1 || dim_b == 1) {
            result[max_ndim - 1 - i] = std::max(dim_a, dim_b);
        } else {
            throw std::runtime_error("Shapes are not broadcastable");
        }
    }

    return result;
}

// Compute strides for broadcasting
inline std::vector<int64_t> compute_broadcast_strides(const std::vector<int64_t>& shape,
                                                       const std::vector<int64_t>& broadcast_shape) {
    std::vector<int64_t> strides(broadcast_shape.size(), 0);

    // Compute normal strides for the original shape
    std::vector<int64_t> original_strides(shape.size());
    if (!shape.empty()) {
        original_strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            original_strides[i] = original_strides[i + 1] * shape[i + 1];
        }
    }

    // Map to broadcast strides
    int64_t offset = static_cast<int64_t>(broadcast_shape.size()) - static_cast<int64_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 1) {
            strides[offset + i] = 0;  // Broadcasting dimension
        } else {
            strides[offset + i] = original_strides[i];
        }
    }

    return strides;
}

// Check if tensors have identical shapes (for optimized path)
inline bool have_same_shape(const Tensor& a, const Tensor& b) {
    if (a.ndim() != b.ndim()) {
        return false;
    }

    auto shape_a = a.shape();
    auto shape_b = b.shape();

    for (size_t i = 0; i < shape_a.size(); ++i) {
        if (shape_a[i] != shape_b[i]) {
            return false;
        }
    }

    return true;
}

// Check if shape_b can be broadcast to shape_a (for in-place operations)
// Returns true if shape_b can be broadcast to match shape_a
inline bool can_broadcast_to(const std::vector<int64_t>& shape_a,
                              const std::vector<int64_t>& shape_b) {
    size_t ndim_a = shape_a.size();
    size_t ndim_b = shape_b.size();

    // Check from the rightmost (trailing) dimension
    for (size_t i = 0; i < std::max(ndim_a, ndim_b); ++i) {
        int64_t dim_a = i < ndim_a ? shape_a[ndim_a - 1 - i] : 1;
        int64_t dim_b = i < ndim_b ? shape_b[ndim_b - 1 - i] : 1;

        // For in-place broadcast: dim_b must be 1 or equal to dim_a
        if (dim_b != 1 && dim_b != dim_a) {
            return false;
        }
    }

    return true;
}

} // namespace detail

// ============================================================================
// Element-wise Binary Operations (with Broadcasting Support)
// ============================================================================

// Fast path: element-wise addition (same shape)
template<typename T>
__global__ void add_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Metadata struct passed by value to broadcast kernels (avoids cudaMalloc for small arrays)
// Audit F.12: rank cap lifted from 8 → 16 across CUDA broadcast/expand kernels.
// Matches DIM_META_MAX_RANK in reduction.cu/activations.cu and the maximum
// rank supported by CPU/ROCm. The runtime throw is preserved so any tensor
// exceeding the cap fails loudly rather than silently truncating.
constexpr int MATH_DIM_META_MAX_RANK = 16;

struct BroadcastMeta {
    int64_t strides_a[MATH_DIM_META_MAX_RANK];
    int64_t strides_b[MATH_DIM_META_MAX_RANK];
    int64_t output_shape[MATH_DIM_META_MAX_RANK];
};

static BroadcastMeta make_broadcast_meta(
    const std::vector<int64_t>& strides_a,
    const std::vector<int64_t>& strides_b,
    const std::vector<int64_t>& output_shape) {
    if (output_shape.size() > MATH_DIM_META_MAX_RANK) {
        throw std::runtime_error(
            "CUDA broadcast: tensor rank " + std::to_string(output_shape.size()) +
            " exceeds maximum " + std::to_string(MATH_DIM_META_MAX_RANK) +
            " (raise MATH_DIM_META_MAX_RANK if needed).");
    }
    BroadcastMeta meta{};
    for (size_t i = 0; i < output_shape.size(); ++i) {
        meta.strides_a[i] = strides_a[i];
        meta.strides_b[i] = strides_b[i];
        meta.output_shape[i] = output_shape[i];
    }
    return meta;
}

// Generic broadcast kernel - works for all binary operations
template<typename T, typename Op>
__global__ void broadcast_kernel(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
    BroadcastMeta meta, int64_t ndim, int64_t n, Op op) {

    TENZOR_CUDA_KERNEL_LOOP(out_idx, n) {
        int64_t idx_a = 0;
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        // Convert flat index to multi-dimensional indices
        // Working from rightmost (fastest-varying) dimension
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % meta.output_shape[i];
            tmp /= meta.output_shape[i];
            idx_a += coord * meta.strides_a[i];
            idx_b += coord * meta.strides_b[i];
        }

        c[out_idx] = op(a[idx_a], b[idx_b]);
    }
}

// float2half_sat is defined in cuda_common.cuh
// Forward declaration for float2bfloat16_sat - defined below
__device__ __forceinline__ __nv_bfloat16 float2bfloat16_sat(float x);

// Device-side operation functors with FP16/BF16 saturating specializations
// FP16/BF16 ops promote to float32 and saturate back, matching the fast-path kernels
struct AddOp {
    template<typename T>
    __device__ T operator()(T a, T b) const { return a + b; }
};
template<> __device__ inline __half AddOp::operator()(const __half a, const __half b) const {
    return float2half_sat(__half2float(a) + __half2float(b));
}
template<> __device__ inline __nv_bfloat16 AddOp::operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return float2bfloat16_sat(__bfloat162float(a) + __bfloat162float(b));
}

struct SubOp {
    template<typename T>
    __device__ T operator()(T a, T b) const { return a - b; }
};
template<> __device__ inline __half SubOp::operator()(const __half a, const __half b) const {
    return float2half_sat(__half2float(a) - __half2float(b));
}
template<> __device__ inline __nv_bfloat16 SubOp::operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return float2bfloat16_sat(__bfloat162float(a) - __bfloat162float(b));
}

struct MulOp {
    template<typename T>
    __device__ T operator()(T a, T b) const { return a * b; }
};
template<> __device__ inline __half MulOp::operator()(const __half a, const __half b) const {
    return float2half_sat(__half2float(a) * __half2float(b));
}
template<> __device__ inline __nv_bfloat16 MulOp::operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return float2bfloat16_sat(__bfloat162float(a) * __bfloat162float(b));
}

struct DivOp {
    // Floating-point div delegates to hardware so IEEE 754 semantics hold:
    // x/0 → ±Inf for x≠0, 0/0 → NaN. The earlier `if (b == 0) return INF;`
    // shortcut silently turned 0/0 into +Inf and broke NaN_Propagation on
    // CUDA (test_numerical_stability). Integer specializations below still
    // need an explicit check because integer div-by-zero is UB in C++.
    template<typename T>
    __device__ T operator()(T a, T b) const {
        return a / b;
    }
};
template<> __device__ inline int32_t DivOp::operator()(int32_t a, int32_t b) const {
    if (b == 0) return 0;
    return a / b;
}
template<> __device__ inline int64_t DivOp::operator()(int64_t a, int64_t b) const {
    if (b == 0) return 0;
    return a / b;
}
template<> __device__ inline __half DivOp::operator()(const __half a, const __half b) const {
    // FP16 path computes in Float32; hardware Float32 divide produces
    // NaN for 0/0 and ±Inf for x/0. float2half_sat passes NaN/Inf
    // through unchanged.
    return float2half_sat(__half2float(a) / __half2float(b));
}
template<> __device__ inline __nv_bfloat16 DivOp::operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return float2bfloat16_sat(__bfloat162float(a) / __bfloat162float(b));
}

// Generic in-place broadcast kernel - works for all in-place binary operations
// Reads from a (target) and b (other), writes result back to a
template<typename T, typename Op>
__global__ void broadcast_inplace_kernel(
    T* a, const T* b,
    BroadcastMeta meta, int64_t ndim, int64_t n, Op op) {

    TENZOR_CUDA_KERNEL_LOOP(out_idx, n) {
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        // Convert flat index to multi-dimensional indices
        // Working from rightmost (fastest-varying) dimension
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % meta.output_shape[i];
            tmp /= meta.output_shape[i];
            idx_b += coord * meta.strides_b[i];
        }

        a[out_idx] = op(a[out_idx], b[idx_b]);
    }
}

// Subtract kernel - element-wise subtraction
template<typename T>
__global__ void sub_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = a[idx] - b[idx];
    }
}

// Multiply kernel - element-wise multiplication
template<typename T>
__global__ void mul_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Divide kernel - element-wise division
// Uses DivOp for consistent div-by-zero handling across all code paths
// (floating-point returns INFINITY, integer returns 0 — no UB).
template<typename T>
__global__ void div_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    DivOp op;
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = op(a[idx], b[idx]);
    }
}

// ============================================================================
// FP16 Saturating Conversion
// ============================================================================

// float2half_sat is provided by cuda_common.cuh

// Saturating Float32 -> BFloat16 conversion: clamps to max finite BFloat16
// value (~3.39e38) instead of producing Inf.  BFloat16 has the same exponent
// range as Float32 so overflow is rare, but accumulation across deep nets or
// large reductions can exceed the finite range.
__device__ __forceinline__ __nv_bfloat16 float2bfloat16_sat(float x) {
    // Preserve NaN and Inf through conversion
    if (::isnan(x) || ::isinf(x)) {
        return __float2bfloat16(x);
    }
    constexpr float kBF16Max = 3.3895313892515355e+38f;  // 0x7F7F in BF16
    x = fminf(fmaxf(x, -kBF16Max), kBF16Max);
    return __float2bfloat16(x);
}

// ============================================================================
// FP16 Binary Operations
// ============================================================================

// FP16 addition kernel (compute in Float32, saturating conversion)
__global__ void add_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = float2half_sat(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

// FP16 subtraction kernel (compute in Float32, saturating conversion)
__global__ void sub_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = float2half_sat(__half2float(a[idx]) - __half2float(b[idx]));
    }
}

// FP16 multiplication kernel (compute in Float32, saturating conversion)
__global__ void mul_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = float2half_sat(__half2float(a[idx]) * __half2float(b[idx]));
    }
}

// FP16 division kernel (compute in Float32, saturating conversion)
__global__ void div_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = float2half_sat(__half2float(a[idx]) / __half2float(b[idx]));
    }
}

// ============================================================================
// BFloat16 Binary Operations (compute in Float32, saturating conversion)
// ============================================================================

// BFloat16 addition kernel
__global__ void add_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = float2bfloat16_sat(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

// BFloat16 subtraction kernel
__global__ void sub_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = float2bfloat16_sat(__bfloat162float(a[idx]) - __bfloat162float(b[idx]));
    }
}

// BFloat16 multiplication kernel
__global__ void mul_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = float2bfloat16_sat(__bfloat162float(a[idx]) * __bfloat162float(b[idx]));
    }
}

// BFloat16 division kernel (with div-by-zero protection, matching FP16 pattern)
__global__ void div_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float fb = __bfloat162float(b[idx]);
        if (fb == 0.0f) {
            c[idx] = __float2bfloat16(INFINITY);
        } else {
            c[idx] = float2bfloat16_sat(__bfloat162float(a[idx]) / fb);
        }
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

// Negate kernel
template<typename T>
__global__ void neg_kernel_device(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = -input[idx];
    }
}

// Absolute value kernel
template<typename T>
__global__ void abs_kernel_device(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        T val = input[idx];
        output[idx] = val >= T(0) ? val : -val;
    }
}

// Absolute value kernel (specialized for float)
__global__ void abs_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = fabsf(input[idx]);
    }
}

// Absolute value kernel (specialized for double)
__global__ void abs_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = fabs(input[idx]);
    }
}

// Negate for Complex64: -(a + bi) = -a - bi.
__global__ void neg_kernel_complex64(const cuFloatComplex* input, cuFloatComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = make_cuFloatComplex(-cuCrealf(input[idx]), -cuCimagf(input[idx]));
    }
}

// Negate for Complex128.
__global__ void neg_kernel_complex128(const cuDoubleComplex* input, cuDoubleComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = make_cuDoubleComplex(-cuCreal(input[idx]), -cuCimag(input[idx]));
    }
}

// Absolute value for Complex64: |a + bi| = hypot(a, b). Output is Float32.
__global__ void abs_kernel_complex64(const cuFloatComplex* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hypotf(cuCrealf(input[idx]), cuCimagf(input[idx]));
    }
}

// Absolute value for Complex128: output is Float64.
__global__ void abs_kernel_complex128(const cuDoubleComplex* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hypot(cuCreal(input[idx]), cuCimag(input[idx]));
    }
}

// Transcendentals on complex numbers.
// exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
__global__ void exp_kernel_complex64(const cuFloatComplex* input, cuFloatComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float a = cuCrealf(input[idx]);
        float b = cuCimagf(input[idx]);
        float ea = expf(a);
        output[idx] = make_cuFloatComplex(ea * cosf(b), ea * sinf(b));
    }
}
__global__ void exp_kernel_complex128(const cuDoubleComplex* input, cuDoubleComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double a = cuCreal(input[idx]);
        double b = cuCimag(input[idx]);
        double ea = exp(a);
        output[idx] = make_cuDoubleComplex(ea * cos(b), ea * sin(b));
    }
}

// log(a+bi) = log(hypot(a,b)) + i*atan2(b,a)
__global__ void log_kernel_complex64(const cuFloatComplex* input, cuFloatComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float a = cuCrealf(input[idx]);
        float b = cuCimagf(input[idx]);
        output[idx] = make_cuFloatComplex(logf(hypotf(a, b)), atan2f(b, a));
    }
}
__global__ void log_kernel_complex128(const cuDoubleComplex* input, cuDoubleComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double a = cuCreal(input[idx]);
        double b = cuCimag(input[idx]);
        output[idx] = make_cuDoubleComplex(log(hypot(a, b)), atan2(b, a));
    }
}

// sqrt(a+bi) principal branch — Kahan/Hull 1994 cancellation-free formulation.
// Let s = sqrt((|a| + hypot(a,b)) / 2).
//   a >= 0:  sqrt(z) = s + i*b/(2s)
//   a <  0:  sqrt(z) = |b|/(2s) + i*copysign(s, b)
__global__ void sqrt_kernel_complex64(const cuFloatComplex* input, cuFloatComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float a = cuCrealf(input[idx]);
        float b = cuCimagf(input[idx]);
        if (a == 0.0f && b == 0.0f) {
            output[idx] = make_cuFloatComplex(0.0f, 0.0f);
            continue;
        }
        float s = sqrtf(0.5f * (fabsf(a) + hypotf(a, b)));
        float re, im;
        if (a >= 0.0f) { re = s;                 im = b / (2.0f * s); }
        else            { re = fabsf(b) / (2.0f * s); im = copysignf(s, b); }
        output[idx] = make_cuFloatComplex(re, im);
    }
}
__global__ void sqrt_kernel_complex128(const cuDoubleComplex* input, cuDoubleComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double a = cuCreal(input[idx]);
        double b = cuCimag(input[idx]);
        if (a == 0.0 && b == 0.0) {
            output[idx] = make_cuDoubleComplex(0.0, 0.0);
            continue;
        }
        double s = sqrt(0.5 * (fabs(a) + hypot(a, b)));
        double re, im;
        if (a >= 0.0) { re = s;              im = b / (2.0 * s); }
        else           { re = fabs(b) / (2.0 * s); im = copysign(s, b); }
        output[idx] = make_cuDoubleComplex(re, im);
    }
}

// sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
__global__ void sin_kernel_complex64(const cuFloatComplex* input, cuFloatComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float a = cuCrealf(input[idx]);
        float b = cuCimagf(input[idx]);
        output[idx] = make_cuFloatComplex(sinf(a) * coshf(b), cosf(a) * sinhf(b));
    }
}
__global__ void sin_kernel_complex128(const cuDoubleComplex* input, cuDoubleComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double a = cuCreal(input[idx]);
        double b = cuCimag(input[idx]);
        output[idx] = make_cuDoubleComplex(sin(a) * cosh(b), cos(a) * sinh(b));
    }
}

// cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
__global__ void cos_kernel_complex64(const cuFloatComplex* input, cuFloatComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float a = cuCrealf(input[idx]);
        float b = cuCimagf(input[idx]);
        output[idx] = make_cuFloatComplex(cosf(a) * coshf(b), -sinf(a) * sinhf(b));
    }
}
__global__ void cos_kernel_complex128(const cuDoubleComplex* input, cuDoubleComplex* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double a = cuCreal(input[idx]);
        double b = cuCimag(input[idx]);
        output[idx] = make_cuDoubleComplex(cos(a) * cosh(b), -sin(a) * sinh(b));
    }
}

// ============================================================================
// FP16 Unary Operations
// ============================================================================

// FP16 negate kernel
__global__ void neg_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __hneg(input[idx]);
    }
}

// FP16 absolute value kernel
__global__ void abs_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __habs(input[idx]);
    }
}

// ============================================================================
// BFloat16 Unary Operations
// ============================================================================

// BFloat16 negate kernel
__global__ void neg_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __hneg(input[idx]);
    }
}

// BFloat16 absolute value kernel
__global__ void abs_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __habs(input[idx]);
    }
}

// ============================================================================
// Mathematical Functions
// ============================================================================

// Square root kernel (float)
__global__ void sqrt_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = sqrtf(input[idx]);
    }
}

// Square root kernel (double)
__global__ void sqrt_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = sqrt(input[idx]);
    }
}

// Exponential kernel (float)
__global__ void exp_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = expf(input[idx]);
    }
}

// Exponential kernel (double)
__global__ void exp_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = exp(input[idx]);
    }
}

// Natural logarithm kernel (float)
__global__ void log_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = logf(input[idx]);
    }
}

// Natural logarithm kernel (double)
__global__ void log_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = log(input[idx]);
    }
}

// Power kernel (float)
__global__ void pow_kernel_f32(const float* input, float* output, float exponent, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = powf(input[idx], exponent);
    }
}

// Power kernel (double)
__global__ void pow_kernel_f64(const double* input, double* output, double exponent, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = pow(input[idx], exponent);
    }
}

// Clamp kernel (float)
__global__ void clamp_kernel_f32(const float* input, float* output, float min_val, float max_val, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = input[idx];
        output[idx] = fminf(fmaxf(val, min_val), max_val);
    }
}

// Clamp kernel (double)
__global__ void clamp_kernel_f64(const double* input, double* output, double min_val, double max_val, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double val = input[idx];
        output[idx] = fmin(fmax(val, min_val), max_val);
    }
}

// Sign kernel (float)
__global__ void sign_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = input[idx];
        // Sign function: -1 if x < 0, 0 if x == 0, +1 if x > 0
        output[idx] = (val > 0.0f) - (val < 0.0f);
    }
}

// Sign kernel (double)
__global__ void sign_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double val = input[idx];
        output[idx] = (val > 0.0) - (val < 0.0);
    }
}

// ============================================================================
// FP16 Mathematical Functions
// ============================================================================

// FP16 square root kernel
__global__ void sqrt_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hsqrt(input[idx]);
    }
}

// FP16 exponential kernel
__global__ void exp_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hexp(input[idx]);
    }
}

// FP16 natural logarithm kernel
__global__ void log_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hlog(input[idx]);
    }
}

// FP16 power kernel
__global__ void pow_kernel_f16(const __half* input, __half* output, __half exponent, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        float exp_val = __half2float(exponent);
        output[idx] = __float2half(powf(val, exp_val));
    }
}

// FP16 clamp kernel
__global__ void clamp_kernel_f16(const __half* input, __half* output, __half min_val, __half max_val, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        __half val = input[idx];
        output[idx] = __hmax(__hmin(val, max_val), min_val);
    }
}

// FP16 sign kernel
__global__ void sign_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        __half val = input[idx];
        __half zero = __float2half(0.0f);
        __half one = __float2half(1.0f);
        __half neg_one = __float2half(-1.0f);

        bool is_pos = __hgt(val, zero);
        bool is_neg = __hlt(val, zero);
        output[idx] = is_pos ? one : (is_neg ? neg_one : zero);
    }
}

// ============================================================================
// BFloat16 Mathematical Functions
// ============================================================================

// BFloat16 square root kernel
__global__ void sqrt_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hsqrt(input[idx]);
    }
}

// BFloat16 exponential kernel
__global__ void exp_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hexp(input[idx]);
    }
}

// BFloat16 natural logarithm kernel
__global__ void log_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hlog(input[idx]);
    }
}

// BFloat16 power kernel
__global__ void pow_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, __nv_bfloat16 exponent, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = __bfloat162float(input[idx]);
        float exp_val = __bfloat162float(exponent);
        output[idx] = __float2bfloat16(powf(val, exp_val));
    }
}

// BFloat16 clamp kernel
__global__ void clamp_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, __nv_bfloat16 min_val, __nv_bfloat16 max_val, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        __nv_bfloat16 val = input[idx];
        output[idx] = __hmax(__hmin(val, max_val), min_val);
    }
}

// BFloat16 sign kernel
__global__ void sign_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        __nv_bfloat16 val = input[idx];
        __nv_bfloat16 zero = __float2bfloat16(0.0f);
        __nv_bfloat16 one = __float2bfloat16(1.0f);
        __nv_bfloat16 neg_one = __float2bfloat16(-1.0f);

        bool is_pos = __hgt(val, zero);
        bool is_neg = __hlt(val, zero);
        output[idx] = is_pos ? one : (is_neg ? neg_one : zero);
    }
}

// ============================================================================
// Complex Elementwise Arithmetic Kernels
// ============================================================================

// Complex add: (ar+ai*i) + (br+bi*i) = (ar+br) + (ai+bi)*i
template<typename T>
__global__ void complex_add_kernel(const T* a, const T* b, T* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int64_t base = idx * 2;
        c[base]     = a[base]     + b[base];
        c[base + 1] = a[base + 1] + b[base + 1];
    }
}

// Complex sub: (ar+ai*i) - (br+bi*i) = (ar-br) + (ai-bi)*i
template<typename T>
__global__ void complex_sub_kernel(const T* a, const T* b, T* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int64_t base = idx * 2;
        c[base]     = a[base]     - b[base];
        c[base + 1] = a[base + 1] - b[base + 1];
    }
}

// Complex mul: (ar+ai*i)*(br+bi*i) = (ar*br - ai*bi) + (ar*bi + ai*br)*i
template<typename T>
__global__ void complex_mul_kernel(const T* a, const T* b, T* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int64_t base = idx * 2;
        T ar = a[base], ai = a[base + 1];
        T br = b[base], bi = b[base + 1];
        c[base]     = ar * br - ai * bi;
        c[base + 1] = ar * bi + ai * br;
    }
}

// Complex div: (ar+ai*i)/(br+bi*i) = ((ar*br+ai*bi) + (ai*br-ar*bi)*i) / (br*br+bi*bi)
template<typename T>
__global__ void complex_div_kernel(const T* a, const T* b, T* c, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int64_t base = idx * 2;
        T ar = a[base], ai = a[base + 1];
        T br = b[base], bi = b[base + 1];
        T denom = br * br + bi * bi;
        c[base]     = (ar * br + ai * bi) / denom;
        c[base + 1] = (ai * br - ar * bi) / denom;
    }
}

// Complex broadcast binary ops
struct ComplexAddOp {
    template<typename T>
    __device__ void operator()(const T* a, const T* b, T* c, int64_t a_base, int64_t b_base, int64_t c_base) const {
        c[c_base]     = a[a_base]     + b[b_base];
        c[c_base + 1] = a[a_base + 1] + b[b_base + 1];
    }
};
struct ComplexSubOp {
    template<typename T>
    __device__ void operator()(const T* a, const T* b, T* c, int64_t a_base, int64_t b_base, int64_t c_base) const {
        c[c_base]     = a[a_base]     - b[b_base];
        c[c_base + 1] = a[a_base + 1] - b[b_base + 1];
    }
};
struct ComplexMulOp {
    template<typename T>
    __device__ void operator()(const T* a, const T* b, T* c, int64_t a_base, int64_t b_base, int64_t c_base) const {
        T ar = a[a_base], ai = a[a_base + 1];
        T br = b[b_base], bi = b[b_base + 1];
        c[c_base]     = ar * br - ai * bi;
        c[c_base + 1] = ar * bi + ai * br;
    }
};
struct ComplexDivOp {
    template<typename T>
    __device__ void operator()(const T* a, const T* b, T* c, int64_t a_base, int64_t b_base, int64_t c_base) const {
        T ar = a[a_base], ai = a[a_base + 1];
        T br = b[b_base], bi = b[b_base + 1];
        T denom = br * br + bi * bi;
        c[c_base]     = (ar * br + ai * bi) / denom;
        c[c_base + 1] = (ai * br - ar * bi) / denom;
    }
};

// Broadcast kernel for complex types - operates on interleaved real/imag pairs
// Strides are in units of complex elements; actual memory offsets are 2x for the float/double array
template<typename T, typename Op>
__global__ void broadcast_complex_kernel(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
    BroadcastMeta meta, int64_t ndim, int64_t n, Op op) {

    TENZOR_CUDA_KERNEL_LOOP(out_idx, n) {
        int64_t idx_a = 0;
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % meta.output_shape[i];
            tmp /= meta.output_shape[i];
            idx_a += coord * meta.strides_a[i];
            idx_b += coord * meta.strides_b[i];
        }
        op(a, b, c, idx_a * 2, idx_b * 2, out_idx * 2);
    }
}

// ============================================================================
// Optimized Kernels with Shared Memory (for reduction-like operations)
// ============================================================================

// Optimized add with shared memory for small tensors
template<typename T>
__global__ void add_kernel_shared(const T* a, const T* b, T* c, int64_t n) {
    __shared__ T s_a[256];
    __shared__ T s_b[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    if (idx < n) {
        s_a[tid] = a[idx];
        s_b[tid] = b[idx];
    }
    __syncthreads();

    // Compute and write result
    if (idx < n) {
        c[idx] = s_a[tid] + s_b[tid];
    }
}

// ============================================================================
// Host Launch Functions
// ============================================================================

// Add kernel launcher
auto add_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype");
    }

    auto a_shape_span = a.shape();
    auto b_shape_span = b.shape();
    std::vector<int64_t> a_shape_vec(a_shape_span.begin(), a_shape_span.end());
    std::vector<int64_t> b_shape_vec(b_shape_span.begin(), b_shape_span.end());

    // Check if broadcastable
    if (!detail::are_broadcastable(a_shape_vec, b_shape_vec)) {
        throw std::runtime_error("Shapes are not broadcastable");
    }

    // Check for fast path (same shape, no broadcasting needed)
    if (detail::have_same_shape(a, b)) {
        int64_t n = a.numel();
        Tensor result(a_shape_vec, a.dtype(), a.device());

        if (n == 0) {
            return result;
        }

        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);

        if (a.dtype() == DType::Float32) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<float>(), b.data<float>(), result.data<float>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Float64) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<double>(), b.data<double>(), result.data<double>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int32) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int64) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Float16) {
            add_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::BFloat16) {
            add_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int8) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::UInt8) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int16) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Bool) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<bool>(), b.data<bool>(), result.data<bool>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Complex64) {
            complex_add_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(a.data_ptr()),
                reinterpret_cast<const float*>(b.data_ptr()),
                reinterpret_cast<float*>(result.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Complex128) {
            complex_add_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const double*>(a.data_ptr()),
                reinterpret_cast<const double*>(b.data_ptr()),
                reinterpret_cast<double*>(result.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("Unsupported dtype for add operation");
        }

        CUDA_CHECK(cudaGetLastError());
        // NOTE: Removed cudaStreamSynchronize - async execution for performance
        return result;
    }

    // Broadcasting path
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(a_shape_vec, b_shape_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    int64_t n = result.numel();
    if (n == 0) {
        return result;
    }

    // Compute strides
    std::vector<int64_t> strides_a = detail::compute_broadcast_strides(a_shape_vec, output_shape);
    std::vector<int64_t> strides_b = detail::compute_broadcast_strides(b_shape_vec, output_shape);

    BroadcastMeta meta = make_broadcast_meta(strides_a, strides_b, output_shape);

    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(),
            meta, ndim, n, AddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<double>(),
            meta, ndim, n, AddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            meta, ndim, n, AddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            meta, ndim, n, AddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            meta, ndim, n, AddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            meta, ndim, n, AddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int8) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(),
            meta, ndim, n, AddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::UInt8) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(),
            meta, ndim, n, AddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(),
            meta, ndim, n, AddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Bool) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<bool>(), b.data<bool>(), result.data<bool>(),
            meta, ndim, n, AddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Complex64) {
        broadcast_complex_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(a.data_ptr()),
            reinterpret_cast<const float*>(b.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()),
            meta, ndim, n, ComplexAddOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Complex128) {
        broadcast_complex_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double*>(a.data_ptr()),
            reinterpret_cast<const double*>(b.data_ptr()),
            reinterpret_cast<double*>(result.data_ptr()),
            meta, ndim, n, ComplexAddOp());
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for add operation");
    }
    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - cudaFree already synchronizes
    return result;
}

// Subtract kernel launcher
auto sub_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype");
    }

    auto a_shape_span = a.shape();
    auto b_shape_span = b.shape();
    std::vector<int64_t> a_shape_vec(a_shape_span.begin(), a_shape_span.end());
    std::vector<int64_t> b_shape_vec(b_shape_span.begin(), b_shape_span.end());

    if (!detail::are_broadcastable(a_shape_vec, b_shape_vec)) {
        throw std::runtime_error("Shapes are not broadcastable");
    }

    // Fast path: same shape
    if (detail::have_same_shape(a, b)) {
        int64_t n = a.numel();
        Tensor result(a_shape_vec, a.dtype(), a.device());

        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);

        if (a.dtype() == DType::Float32) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Float64) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int32) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int64) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Float16) {
            sub_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::BFloat16) {
            sub_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int8) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::UInt8) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int16) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::UInt16) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<uint16_t>(), b.data<uint16_t>(), result.data<uint16_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::UInt32) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<uint32_t>(), b.data<uint32_t>(), result.data<uint32_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::UInt64) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<uint64_t>(), b.data<uint64_t>(), result.data<uint64_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Complex64) {
            complex_sub_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(a.data_ptr()),
                reinterpret_cast<const float*>(b.data_ptr()),
                reinterpret_cast<float*>(result.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Complex128) {
            complex_sub_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const double*>(a.data_ptr()),
                reinterpret_cast<const double*>(b.data_ptr()),
                reinterpret_cast<double*>(result.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("Unsupported dtype for sub operation");
        }

        CUDA_CHECK(cudaGetLastError());
        // NOTE: Removed cudaStreamSynchronize - async execution for performance
        return result;
    }

    // Broadcasting path
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(a_shape_vec, b_shape_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    int64_t n = result.numel();
    if (n == 0) {
        return result;
    }

    std::vector<int64_t> strides_a = detail::compute_broadcast_strides(a_shape_vec, output_shape);
    std::vector<int64_t> strides_b = detail::compute_broadcast_strides(b_shape_vec, output_shape);

    BroadcastMeta meta = make_broadcast_meta(strides_a, strides_b, output_shape);

    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<double>(),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int8) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::UInt8) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::UInt16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<uint16_t>(), b.data<uint16_t>(), result.data<uint16_t>(),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::UInt32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<uint32_t>(), b.data<uint32_t>(), result.data<uint32_t>(),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::UInt64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<uint64_t>(), b.data<uint64_t>(), result.data<uint64_t>(),
            meta, ndim, n, SubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Complex64) {
        broadcast_complex_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(a.data_ptr()),
            reinterpret_cast<const float*>(b.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()),
            meta, ndim, n, ComplexSubOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Complex128) {
        broadcast_complex_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double*>(a.data_ptr()),
            reinterpret_cast<const double*>(b.data_ptr()),
            reinterpret_cast<double*>(result.data_ptr()),
            meta, ndim, n, ComplexSubOp());
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for sub operation");
    }
    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Multiply kernel launcher
auto mul_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype");
    }

    auto a_shape_span = a.shape();
    auto b_shape_span = b.shape();
    std::vector<int64_t> a_shape_vec(a_shape_span.begin(), a_shape_span.end());
    std::vector<int64_t> b_shape_vec(b_shape_span.begin(), b_shape_span.end());

    if (!detail::are_broadcastable(a_shape_vec, b_shape_vec)) {
        throw std::runtime_error("Shapes are not broadcastable");
    }

    // Fast path: same shape
    if (detail::have_same_shape(a, b)) {
        int64_t n = a.numel();
        Tensor result(a_shape_vec, a.dtype(), a.device());

        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);

        if (a.dtype() == DType::Float32) {
            mul_kernel_device<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Float64) {
            mul_kernel_device<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int32) {
            mul_kernel_device<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int64) {
            mul_kernel_device<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Float16) {
            mul_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::BFloat16) {
            mul_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Bool) {
            mul_kernel_device<<<grid, block, 0, stream>>>(a.data<bool>(), b.data<bool>(), result.data<bool>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Complex64) {
            complex_mul_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(a.data_ptr()),
                reinterpret_cast<const float*>(b.data_ptr()),
                reinterpret_cast<float*>(result.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Complex128) {
            complex_mul_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const double*>(a.data_ptr()),
                reinterpret_cast<const double*>(b.data_ptr()),
                reinterpret_cast<double*>(result.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("Unsupported dtype for mul operation");
        }

        CUDA_CHECK(cudaGetLastError());
        // NOTE: Removed cudaStreamSynchronize - async execution for performance
        return result;
    }

    // Broadcasting path
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(a_shape_vec, b_shape_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    int64_t n = result.numel();
    if (n == 0) {
        return result;
    }

    std::vector<int64_t> strides_a = detail::compute_broadcast_strides(a_shape_vec, output_shape);
    std::vector<int64_t> strides_b = detail::compute_broadcast_strides(b_shape_vec, output_shape);

    BroadcastMeta meta = make_broadcast_meta(strides_a, strides_b, output_shape);

    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(),
            meta, ndim, n, MulOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<double>(),
            meta, ndim, n, MulOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            meta, ndim, n, MulOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            meta, ndim, n, MulOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            meta, ndim, n, MulOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            meta, ndim, n, MulOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Bool) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<bool>(), b.data<bool>(), result.data<bool>(),
            meta, ndim, n, MulOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Complex64) {
        broadcast_complex_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(a.data_ptr()),
            reinterpret_cast<const float*>(b.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()),
            meta, ndim, n, ComplexMulOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Complex128) {
        broadcast_complex_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double*>(a.data_ptr()),
            reinterpret_cast<const double*>(b.data_ptr()),
            reinterpret_cast<double*>(result.data_ptr()),
            meta, ndim, n, ComplexMulOp());
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for mul operation");
    }
    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Divide kernel launcher
auto div_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype");
    }

    auto a_shape_span = a.shape();
    auto b_shape_span = b.shape();
    std::vector<int64_t> a_shape_vec(a_shape_span.begin(), a_shape_span.end());
    std::vector<int64_t> b_shape_vec(b_shape_span.begin(), b_shape_span.end());

    if (!detail::are_broadcastable(a_shape_vec, b_shape_vec)) {
        throw std::runtime_error("Shapes are not broadcastable");
    }

    // Fast path: same shape
    if (detail::have_same_shape(a, b)) {
        int64_t n = a.numel();
        Tensor result(a_shape_vec, a.dtype(), a.device());

        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);

        if (a.dtype() == DType::Float32) {
            div_kernel_device<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Float64) {
            div_kernel_device<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int32) {
            check_integer_divisor_for_zeros(b.data<int32_t>(), n, stream);
            div_kernel_device<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int64) {
            check_integer_divisor_for_zeros(b.data<int64_t>(), n, stream);
            div_kernel_device<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Float16) {
            div_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::BFloat16) {
            div_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Complex64) {
            complex_div_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const float*>(a.data_ptr()),
                reinterpret_cast<const float*>(b.data_ptr()),
                reinterpret_cast<float*>(result.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Complex128) {
            complex_div_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const double*>(a.data_ptr()),
                reinterpret_cast<const double*>(b.data_ptr()),
                reinterpret_cast<double*>(result.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("Unsupported dtype for div operation");
        }

        CUDA_CHECK(cudaGetLastError());
        // NOTE: Removed cudaStreamSynchronize - async execution for performance
        return result;
    }

    // Broadcasting path
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(a_shape_vec, b_shape_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    int64_t n = result.numel();
    if (n == 0) {
        return result;
    }

    std::vector<int64_t> strides_a = detail::compute_broadcast_strides(a_shape_vec, output_shape);
    std::vector<int64_t> strides_b = detail::compute_broadcast_strides(b_shape_vec, output_shape);

    BroadcastMeta meta = make_broadcast_meta(strides_a, strides_b, output_shape);

    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(),
            meta, ndim, n, DivOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<double>(),
            meta, ndim, n, DivOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int32) {
        check_integer_divisor_for_zeros(b.data<int32_t>(), b.numel(), stream);
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            meta, ndim, n, DivOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int64) {
        check_integer_divisor_for_zeros(b.data<int64_t>(), b.numel(), stream);
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            meta, ndim, n, DivOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            meta, ndim, n, DivOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            meta, ndim, n, DivOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Complex64) {
        broadcast_complex_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(a.data_ptr()),
            reinterpret_cast<const float*>(b.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()),
            meta, ndim, n, ComplexDivOp());
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Complex128) {
        broadcast_complex_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double*>(a.data_ptr()),
            reinterpret_cast<const double*>(b.data_ptr()),
            reinterpret_cast<double*>(result.data_ptr()),
            meta, ndim, n, ComplexDivOp());
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for div operation");
    }
    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Negate kernel launcher
auto neg_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        neg_kernel_device<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        neg_kernel_device<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int32) {
        neg_kernel_device<<<grid, block, 0, stream>>>(input.data<int32_t>(), result.data<int32_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int64) {
        neg_kernel_device<<<grid, block, 0, stream>>>(input.data<int64_t>(), result.data<int64_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        neg_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        neg_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex64) {
        neg_kernel_complex64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuFloatComplex*>(result.data<uint8_t>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex128) {
        neg_kernel_complex128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuDoubleComplex*>(result.data<uint8_t>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for neg operation");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Absolute value kernel launcher
auto abs_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    // Complex inputs: |z| = hypot(re, im), output is real-valued.
    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        abs_kernel_complex64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuFloatComplex*>(input.data<uint8_t>()),
            result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        abs_kernel_complex128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuDoubleComplex*>(input.data<uint8_t>()),
            result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        abs_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        abs_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int32) {
        abs_kernel_device<<<grid, block, 0, stream>>>(input.data<int32_t>(), result.data<int32_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int64) {
        abs_kernel_device<<<grid, block, 0, stream>>>(input.data<int64_t>(), result.data<int64_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        abs_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        abs_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for abs operation");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Square root kernel launcher
auto sqrt_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        sqrt_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        sqrt_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        sqrt_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        sqrt_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex64) {
        sqrt_kernel_complex64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuFloatComplex*>(result.data<uint8_t>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex128) {
        sqrt_kernel_complex128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuDoubleComplex*>(result.data<uint8_t>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("sqrt operation only supports Float32, Float64, Float16, BFloat16, Complex64, Complex128 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Exponential kernel launcher
auto exp_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        exp_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        exp_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        exp_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        exp_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex64) {
        exp_kernel_complex64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuFloatComplex*>(result.data<uint8_t>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex128) {
        exp_kernel_complex128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuDoubleComplex*>(result.data<uint8_t>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("exp operation only supports Float32, Float64, Float16, BFloat16, Complex64, Complex128 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Natural logarithm kernel launcher
auto log_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        log_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        log_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        log_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        log_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex64) {
        log_kernel_complex64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuFloatComplex*>(result.data<uint8_t>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex128) {
        log_kernel_complex128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuDoubleComplex*>(result.data<uint8_t>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("log operation only supports Float32, Float64, Float16, BFloat16, Complex64, Complex128 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Power kernel launcher
auto pow_kernel(const Tensor& input, float exponent, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        pow_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), exponent, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        double exp_d = static_cast<double>(exponent);
        pow_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), exp_d, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        __half exp_h = __float2half(exponent);
        pow_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), exp_h, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        __nv_bfloat16 exp_bf = __float2bfloat16(exponent);
        pow_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), exp_bf, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("pow operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Clamp kernel launcher
auto clamp_kernel(const Tensor& input, float min_val, float max_val, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        clamp_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), min_val, max_val, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        double min_d = static_cast<double>(min_val);
        double max_d = static_cast<double>(max_val);
        clamp_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), min_d, max_d, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        __half min_h = __float2half(min_val);
        __half max_h = __float2half(max_val);
        clamp_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), min_h, max_h, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        __nv_bfloat16 min_bf = __float2bfloat16(min_val);
        __nv_bfloat16 max_bf = __float2bfloat16(max_val);
        clamp_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), min_bf, max_bf, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("clamp operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Sign kernel launcher
auto sign_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        sign_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        sign_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        sign_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        sign_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("sign operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Trigonometric functions
template<typename T>
__global__ void sin_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = sin(input[idx]);
    }
}

template<typename T>
__global__ void cos_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = cos(input[idx]);
    }
}

template<typename T>
__global__ void tan_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = tan(input[idx]);
    }
}

template<typename T>
__global__ void asin_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = asin(input[idx]);
    }
}

template<typename T>
__global__ void acos_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = acos(input[idx]);
    }
}

template<typename T>
__global__ void atan_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = atan(input[idx]);
    }
}

template<typename T>
__global__ void sinh_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = sinh(input[idx]);
    }
}

template<typename T>
__global__ void cosh_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = cosh(input[idx]);
    }
}

// Rounding functions
template<typename T>
__global__ void ceil_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = ceil(input[idx]);
    }
}

template<typename T>
__global__ void floor_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = floor(input[idx]);
    }
}

template<typename T>
__global__ void round_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = round(input[idx]);
    }
}

template<typename T>
__global__ void trunc_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = trunc(input[idx]);
    }
}

template<typename T>
__global__ void reciprocal_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = T(1.0) / input[idx];
    }
}

// Launcher functions for trigonometric operations
#define DEFINE_TRIG_KERNEL(name) \
auto name##_kernel(const Tensor& input, cudaStream_t stream) -> Tensor { \
    /* Float16/BFloat16: upcast to Float32, compute, downcast */ \
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) { \
        DType orig = input.dtype(); \
        Tensor f32_input = input.to(DType::Float32); \
        Tensor f32_result = name##_kernel(f32_input, stream); \
        return f32_result.to(orig); \
    } \
    int64_t n = input.numel(); \
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end()); \
    Tensor result(shape, input.dtype(), input.device()); \
    dim3 grid, block; \
    compute_launch_config_1d(n, grid, block); \
    if (input.dtype() == DType::Float32) { \
        name##_kernel_impl<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n); \
    } else if (input.dtype() == DType::Float64) { \
        name##_kernel_impl<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n); \
    } else { \
        throw std::runtime_error(#name " operation only supports floating point dtypes"); \
    } \
    CUDA_CHECK(cudaGetLastError()); \
    return result; \
}

// sin/cos: explicit definitions so Complex64/Complex128 can be added
// alongside the standard float paths. Other trig ops (tan/asin/acos/atan/
// sinh/cosh) stick with the DEFINE_TRIG_KERNEL macro because their complex
// branches aren't implemented here.
auto sin_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor f32_input = input.to(DType::Float32);
        Tensor f32_result = sin_kernel(f32_input, stream);
        return f32_result.to(orig);
    }
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        sin_kernel_impl<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        sin_kernel_impl<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Complex64) {
        sin_kernel_complex64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuFloatComplex*>(result.data<uint8_t>()), n);
    } else if (input.dtype() == DType::Complex128) {
        sin_kernel_complex128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuDoubleComplex*>(result.data<uint8_t>()), n);
    } else {
        throw std::runtime_error("sin operation only supports floating point and complex dtypes");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

auto cos_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor f32_input = input.to(DType::Float32);
        Tensor f32_result = cos_kernel(f32_input, stream);
        return f32_result.to(orig);
    }
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        cos_kernel_impl<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        cos_kernel_impl<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Complex64) {
        cos_kernel_complex64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuFloatComplex*>(result.data<uint8_t>()), n);
    } else if (input.dtype() == DType::Complex128) {
        cos_kernel_complex128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<cuDoubleComplex*>(result.data<uint8_t>()), n);
    } else {
        throw std::runtime_error("cos operation only supports floating point and complex dtypes");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

DEFINE_TRIG_KERNEL(tan)
DEFINE_TRIG_KERNEL(asin)
DEFINE_TRIG_KERNEL(acos)
DEFINE_TRIG_KERNEL(atan)
DEFINE_TRIG_KERNEL(sinh)
DEFINE_TRIG_KERNEL(cosh)
DEFINE_TRIG_KERNEL(ceil)
DEFINE_TRIG_KERNEL(floor)
DEFINE_TRIG_KERNEL(round)
DEFINE_TRIG_KERNEL(trunc)
DEFINE_TRIG_KERNEL(reciprocal)

// Clamp min/max functions
template<typename T>
__global__ void clamp_min_kernel_impl(const T* input, T* output, T min_val, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx] < min_val ? min_val : input[idx];
    }
}

template<typename T>
__global__ void clamp_max_kernel_impl(const T* input, T* output, T max_val, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx] > max_val ? max_val : input[idx];
    }
}

// Float16 / BFloat16 clamp kernels — compute through Float32 for stability
// across architectures.
__global__ void clamp_min_kernel_f16(const __half* input, __half* output, float min_val, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = __half2float(input[idx]);
        output[idx] = __float2half(v < min_val ? min_val : v);
    }
}

__global__ void clamp_max_kernel_f16(const __half* input, __half* output, float max_val, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = __half2float(input[idx]);
        output[idx] = __float2half(v > max_val ? max_val : v);
    }
}

__global__ void clamp_min_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, float min_val, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(v < min_val ? min_val : v);
    }
}

__global__ void clamp_max_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, float max_val, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16(v > max_val ? max_val : v);
    }
}

auto clamp_min_kernel(const Tensor& input, float min_val, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        clamp_min_kernel_impl<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), min_val, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        clamp_min_kernel_impl<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), static_cast<double>(min_val), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        clamp_min_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), min_val, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        clamp_min_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), min_val, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("clamp_min operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

auto clamp_max_kernel(const Tensor& input, float max_val, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        clamp_max_kernel_impl<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), max_val, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        clamp_max_kernel_impl<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), static_cast<double>(max_val), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        clamp_max_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), max_val, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        clamp_max_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), max_val, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("clamp_max operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// In-place operations
template<typename T>
__global__ void add_inplace_kernel_impl(T* data, const T* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] += other[idx];
    }
}

template<typename T>
__global__ void sub_inplace_kernel_impl(T* data, const T* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] -= other[idx];
    }
}

template<typename T>
__global__ void mul_inplace_kernel_impl(T* data, const T* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] *= other[idx];
    }
}

template<typename T>
__global__ void div_inplace_kernel_impl(T* data, const T* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] /= other[idx];
    }
}

// Float16 in-place kernels
__global__ void add_inplace_kernel_f16(__half* data, const __half* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = __hadd(data[idx], other[idx]);
    }
}

__global__ void sub_inplace_kernel_f16(__half* data, const __half* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = __hsub(data[idx], other[idx]);
    }
}

__global__ void mul_inplace_kernel_f16(__half* data, const __half* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = __hmul(data[idx], other[idx]);
    }
}

__global__ void div_inplace_kernel_f16(__half* data, const __half* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = __hdiv(data[idx], other[idx]);
    }
}

// BFloat16 in-place kernels (compute in Float32 with saturating conversion, matching out-of-place pattern)
__global__ void add_inplace_kernel_bf16(__nv_bfloat16* data, const __nv_bfloat16* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = float2bfloat16_sat(__bfloat162float(data[idx]) + __bfloat162float(other[idx]));
    }
}

__global__ void sub_inplace_kernel_bf16(__nv_bfloat16* data, const __nv_bfloat16* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = float2bfloat16_sat(__bfloat162float(data[idx]) - __bfloat162float(other[idx]));
    }
}

__global__ void mul_inplace_kernel_bf16(__nv_bfloat16* data, const __nv_bfloat16* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = float2bfloat16_sat(__bfloat162float(data[idx]) * __bfloat162float(other[idx]));
    }
}

__global__ void div_inplace_kernel_bf16(__nv_bfloat16* data, const __nv_bfloat16* other, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float fb = __bfloat162float(other[idx]);
        if (fb == 0.0f) {
            data[idx] = __float2bfloat16(INFINITY);
        } else {
            data[idx] = float2bfloat16_sat(__bfloat162float(data[idx]) / fb);
        }
    }
}

auto add_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor {
    int64_t n = inout.numel();
    bool same_shape = detail::have_same_shape(inout, other);

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (same_shape) {
        // Fast path: same shape, element-wise operation
        if (inout.dtype() == DType::Float32) {
            add_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<float>(), other.data<float>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float64) {
            add_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<double>(), other.data<double>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float16) {
            add_inplace_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::BFloat16) {
            add_inplace_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(inout.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(other.data<BFloat16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("add_inplace operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
        }
    } else {
        // Broadcast path: compute strides for the 'other' tensor
        auto inout_shape = inout.shape();
        auto other_shape = other.shape();
        std::vector<int64_t> inout_shape_vec(inout_shape.begin(), inout_shape.end());
        std::vector<int64_t> other_shape_vec(other_shape.begin(), other_shape.end());

        // Validate that other can be broadcast to inout's shape
        if (!detail::can_broadcast_to(inout_shape_vec, other_shape_vec)) {
            throw std::runtime_error("In-place add: shapes are not compatible for broadcasting");
        }

        // For in-place, output shape is always inout's shape
        std::vector<int64_t> strides_b = detail::compute_broadcast_strides(other_shape_vec, inout_shape_vec);

        int64_t ndim = inout_shape_vec.size();
        std::vector<int64_t> strides_a_zeros(ndim, 0);
        BroadcastMeta meta = make_broadcast_meta(strides_a_zeros, strides_b, inout_shape_vec);

        if (inout.dtype() == DType::Float32) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<float>(), other.data<float>(),
                meta, ndim, n, AddOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float64) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<double>(), other.data<double>(),
                meta, ndim, n, AddOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()),
                meta, ndim, n, AddOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::BFloat16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(inout.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(other.data<BFloat16>()),
                meta, ndim, n, AddOp{});
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("add_inplace operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
        }
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return inout;
}

auto sub_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor {
    int64_t n = inout.numel();
    bool same_shape = detail::have_same_shape(inout, other);

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (same_shape) {
        if (inout.dtype() == DType::Float32) {
            sub_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<float>(), other.data<float>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float64) {
            sub_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<double>(), other.data<double>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float16) {
            sub_inplace_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::BFloat16) {
            sub_inplace_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(inout.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(other.data<BFloat16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("sub_inplace operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
        }
    } else {
        auto inout_shape = inout.shape();
        auto other_shape = other.shape();
        std::vector<int64_t> inout_shape_vec(inout_shape.begin(), inout_shape.end());
        std::vector<int64_t> other_shape_vec(other_shape.begin(), other_shape.end());

        // Validate that other can be broadcast to inout's shape
        if (!detail::can_broadcast_to(inout_shape_vec, other_shape_vec)) {
            throw std::runtime_error("In-place sub: shapes are not compatible for broadcasting");
        }

        std::vector<int64_t> strides_b = detail::compute_broadcast_strides(other_shape_vec, inout_shape_vec);

        int64_t ndim = inout_shape_vec.size();
        std::vector<int64_t> strides_a_zeros(ndim, 0);
        BroadcastMeta meta = make_broadcast_meta(strides_a_zeros, strides_b, inout_shape_vec);

        if (inout.dtype() == DType::Float32) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<float>(), other.data<float>(),
                meta, ndim, n, SubOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float64) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<double>(), other.data<double>(),
                meta, ndim, n, SubOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()),
                meta, ndim, n, SubOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::BFloat16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(inout.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(other.data<BFloat16>()),
                meta, ndim, n, SubOp{});
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("sub_inplace operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
        }
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return inout;
}

auto mul_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor {
    int64_t n = inout.numel();
    bool same_shape = detail::have_same_shape(inout, other);

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (same_shape) {
        if (inout.dtype() == DType::Float32) {
            mul_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<float>(), other.data<float>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float64) {
            mul_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<double>(), other.data<double>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float16) {
            mul_inplace_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::BFloat16) {
            mul_inplace_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(inout.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(other.data<BFloat16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("mul_inplace operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
        }
    } else {
        auto inout_shape = inout.shape();
        auto other_shape = other.shape();
        std::vector<int64_t> inout_shape_vec(inout_shape.begin(), inout_shape.end());
        std::vector<int64_t> other_shape_vec(other_shape.begin(), other_shape.end());

        // Validate that other can be broadcast to inout's shape
        if (!detail::can_broadcast_to(inout_shape_vec, other_shape_vec)) {
            throw std::runtime_error("In-place mul: shapes are not compatible for broadcasting");
        }

        std::vector<int64_t> strides_b = detail::compute_broadcast_strides(other_shape_vec, inout_shape_vec);

        int64_t ndim = inout_shape_vec.size();
        std::vector<int64_t> strides_a_zeros(ndim, 0);
        BroadcastMeta meta = make_broadcast_meta(strides_a_zeros, strides_b, inout_shape_vec);

        if (inout.dtype() == DType::Float32) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<float>(), other.data<float>(),
                meta, ndim, n, MulOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float64) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<double>(), other.data<double>(),
                meta, ndim, n, MulOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()),
                meta, ndim, n, MulOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::BFloat16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(inout.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(other.data<BFloat16>()),
                meta, ndim, n, MulOp{});
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("mul_inplace operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
        }
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return inout;
}

auto div_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor {
    int64_t n = inout.numel();
    bool same_shape = detail::have_same_shape(inout, other);

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (same_shape) {
        if (inout.dtype() == DType::Float32) {
            div_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<float>(), other.data<float>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float64) {
            div_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<double>(), other.data<double>(), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float16) {
            div_inplace_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::BFloat16) {
            div_inplace_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(inout.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(other.data<BFloat16>()), n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("div_inplace operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
        }
    } else {
        auto inout_shape = inout.shape();
        auto other_shape = other.shape();
        std::vector<int64_t> inout_shape_vec(inout_shape.begin(), inout_shape.end());
        std::vector<int64_t> other_shape_vec(other_shape.begin(), other_shape.end());

        // Validate that other can be broadcast to inout's shape
        if (!detail::can_broadcast_to(inout_shape_vec, other_shape_vec)) {
            throw std::runtime_error("In-place div: shapes are not compatible for broadcasting");
        }

        std::vector<int64_t> strides_b = detail::compute_broadcast_strides(other_shape_vec, inout_shape_vec);

        int64_t ndim = inout_shape_vec.size();
        std::vector<int64_t> strides_a_zeros(ndim, 0);
        BroadcastMeta meta = make_broadcast_meta(strides_a_zeros, strides_b, inout_shape_vec);

        if (inout.dtype() == DType::Float32) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<float>(), other.data<float>(),
                meta, ndim, n, DivOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float64) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<double>(), other.data<double>(),
                meta, ndim, n, DivOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::Float16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()),
                meta, ndim, n, DivOp{});
            CUDA_CHECK(cudaGetLastError());
        } else if (inout.dtype() == DType::BFloat16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(inout.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(other.data<BFloat16>()),
                meta, ndim, n, DivOp{});
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("div_inplace operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
        }
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return inout;
}

// Metadata struct passed by value to expand kernel (avoids cudaMalloc for shape/stride arrays).
// Audit F.12: rank cap lifted from 8 → MATH_DIM_META_MAX_RANK (16). Previously
// the make-time loop guarded `i < 8` and silently dropped higher dimensions.
struct ExpandMeta {
    int64_t input_shape[MATH_DIM_META_MAX_RANK];
    int64_t input_strides[MATH_DIM_META_MAX_RANK];
    int64_t output_shape[MATH_DIM_META_MAX_RANK];
};

// Expand kernel - replicate tensor along specified dimensions
template<typename T>
__global__ void expand_kernel_device(
    const T* input, T* output,
    ExpandMeta meta, int64_t input_ndim, int64_t output_ndim, int64_t n) {

    TENZOR_CUDA_KERNEL_LOOP(out_idx, n) {
        int64_t temp = out_idx;
        int64_t in_idx = 0;

        // Dimension offset (output can have more dims than input due to leading 1s)
        int64_t input_dim_offset = output_ndim - input_ndim;

        // Convert output flat index to multi-dimensional coordinates
        for (int64_t i = output_ndim - 1; i >= 0; --i) {
            int64_t coord = temp % meta.output_shape[i];
            temp /= meta.output_shape[i];

            int64_t input_dim = i - input_dim_offset;
            if (input_dim >= 0 && input_dim < input_ndim) {
                // For dimensions of size 1, we don't advance the index (broadcast)
                if (meta.input_shape[input_dim] != 1) {
                    in_idx += coord * meta.input_strides[input_dim];
                }
                // If input_shape[input_dim] == 1, stride is effectively 0 (broadcast)
            }
        }

        output[out_idx] = input[in_idx];
    }
}

// Expand kernel launcher
auto expand_kernel(const Tensor& input_in, const std::vector<int64_t>& shape, void* stream_ptr) -> Tensor {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    // The kernel below computes input strides from the input shape assuming a
    // contiguous layout. If the caller passed a non-contiguous view (e.g. the
    // result of permute or unsqueeze-on-permuted), the computed strides would
    // be wrong and the kernel would read garbage. Materialize to contiguous
    // first to match the CPU semantics. (CPU does the same in src/ops/transform.cpp.)
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape_vec(input_shape_span.begin(), input_shape_span.end());

    // Validate expansion is possible
    if (shape.size() < input_shape_vec.size()) {
        throw std::invalid_argument("Expanded shape must have at least as many dimensions as input");
    }

    // Check if already the right shape
    bool same_shape = (shape.size() == input_shape_vec.size());
    if (same_shape) {
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != input_shape_vec[i]) {
                same_shape = false;
                break;
            }
        }
    }
    if (same_shape) {
        return input;
    }

    // Validate each dimension can be expanded
    int64_t input_dim_offset = shape.size() - input_shape_vec.size();
    for (size_t i = 0; i < input_shape_vec.size(); ++i) {
        int64_t output_dim = i + input_dim_offset;
        if (input_shape_vec[i] != 1 && input_shape_vec[i] != shape[output_dim]) {
            throw std::runtime_error(
                "Cannot expand dimension from size " + std::to_string(input_shape_vec[i]) +
                " to " + std::to_string(shape[output_dim]));
        }
    }

    // Create output tensor
    Tensor result(shape, input.dtype(), input.device());

    // Calculate input strides
    std::vector<int64_t> input_strides(input_shape_vec.size());
    int64_t input_stride = 1;
    for (int i = input_shape_vec.size() - 1; i >= 0; --i) {
        input_strides[i] = input_stride;
        input_stride *= input_shape_vec[i];
    }

    // Build metadata struct passed by value (avoids cudaMalloc).
    // Audit F.12: enforce the rank cap explicitly instead of silently truncating.
    if (input_shape_vec.size() > MATH_DIM_META_MAX_RANK ||
        shape.size() > MATH_DIM_META_MAX_RANK) {
        throw std::runtime_error(
            "CUDA expand_kernel: tensor rank exceeds maximum " +
            std::to_string(MATH_DIM_META_MAX_RANK) +
            " (input ndim=" + std::to_string(input_shape_vec.size()) +
            ", output ndim=" + std::to_string(shape.size()) +
            "; raise MATH_DIM_META_MAX_RANK if needed).");
    }
    ExpandMeta meta{};
    for (size_t i = 0; i < input_shape_vec.size(); ++i) {
        meta.input_shape[i] = input_shape_vec[i];
        meta.input_strides[i] = input_strides[i];
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        meta.output_shape[i] = shape[i];
    }

    int64_t n = result.numel();
    int64_t input_ndim = input_shape_vec.size();
    int64_t output_ndim = shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<float>(), result.data<float>(),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<double>(), result.data<double>(),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int32) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<int32_t>(), result.data<int32_t>(),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int64) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<int64_t>(), result.data<int64_t>(),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Bool) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<bool>(), result.data<bool>(),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int8) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<int8_t>(), result.data<int8_t>(),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::UInt8) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<uint8_t>(), result.data<uint8_t>(),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int16) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<int16_t>(), result.data<int16_t>(),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex64) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuFloatComplex*>(input.data_ptr()),
            reinterpret_cast<cuFloatComplex*>(result.data_ptr()),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex128) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const cuDoubleComplex*>(input.data_ptr()),
            reinterpret_cast<cuDoubleComplex*>(result.data_ptr()),
            meta, input_ndim, output_ndim, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for expand operation");
    }

    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ============================================================================
// Fill Operations (zeros, ones, full)
// ============================================================================

// Fill kernel - set all elements to a constant value.
// Note: This operation is asynchronous on the given stream. Callers must
// synchronize the stream before reading the tensor on the host.
template<typename T>
__global__ void fill_kernel_device(T* output, T value, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = value;
    }
}

// Fill kernel launcher - fills tensor with constant value
auto fill_kernel(const Tensor& tensor, double value, cudaStream_t stream) -> Tensor {
    int64_t n = tensor.numel();

    if (n == 0) {
        return tensor;
    }

    // Create a copy to modify
    auto result = tensor;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (tensor.dtype() == DType::Float32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<float>(), static_cast<float>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (tensor.dtype() == DType::Float64) {
        // Take the double value directly — narrowing to float here would
        // collapse Float64 subnormals (~5e-324) to zero because the
        // smallest Float32 subnormal is ~1.4e-45.
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<double>(), value, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (tensor.dtype() == DType::Int32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int32_t>(), static_cast<int32_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (tensor.dtype() == DType::Int64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int64_t>(), static_cast<int64_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (tensor.dtype() == DType::Float16) {
        __half h_value = __float2half(static_cast<float>(value));
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), h_value, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (tensor.dtype() == DType::BFloat16) {
        __nv_bfloat16 bf_value = __float2bfloat16(static_cast<float>(value));
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), bf_value, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (tensor.dtype() == DType::Int8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int8_t>(), static_cast<int8_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (tensor.dtype() == DType::UInt8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint8_t>(), static_cast<uint8_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (tensor.dtype() == DType::Bool) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<bool>(), static_cast<bool>(value != 0.0), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (tensor.dtype() == DType::Complex64) {
        cuFloatComplex c = make_cuFloatComplex(static_cast<float>(value), 0.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<cuFloatComplex*>(result.data_ptr()), c, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (tensor.dtype() == DType::Complex128) {
        cuDoubleComplex z = make_cuDoubleComplex(static_cast<double>(value), 0.0);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<cuDoubleComplex*>(result.data_ptr()), z, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for fill operation");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Strided fill kernel — fills non-contiguous tensor elements in-place on GPU
// Each thread computes its own multi-dimensional index from a flat index,
// then writes the fill value at the strided offset.
template<typename T>
__global__ void strided_fill_kernel_device(
    T* base, T value, int64_t n,
    const int64_t* shape, const int64_t* strides, int32_t ndims) {
    TENZOR_CUDA_KERNEL_LOOP(flat_idx, n) {
        int64_t remaining = flat_idx;
        int64_t offset = 0;
        for (int32_t d = ndims - 1; d >= 0; --d) {
            int64_t coord = remaining % shape[d];
            remaining /= shape[d];
            offset += coord * strides[d];
        }
        base[offset] = value;
    }
}

// Strided fill for non-contiguous tensors. Asynchronous on the given stream.
auto strided_fill_kernel(Tensor& self, double value, cudaStream_t stream) -> void {
    int64_t n = self.numel();
    if (n == 0) return;

    auto ndims = self.ndim();
    auto shp = self.shape();
    auto str = self.strides();

    // Copy shape and strides to device
    std::vector<int64_t> meta(ndims * 2);
    for (int64_t d = 0; d < ndims; ++d) {
        meta[d] = shp[d];
        meta[ndims + d] = str[d];
    }
    int64_t* d_meta = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_meta, meta.size() * sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_meta, meta.data(), meta.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    auto launch = [&](auto* ptr, auto typed_value) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        strided_fill_kernel_device<<<grid, block, 0, stream>>>(
            ptr, typed_value, n, d_meta, d_meta + ndims, static_cast<int32_t>(ndims));
        CUDA_CHECK(cudaGetLastError());
    };

    if (self.dtype() == DType::Float32) {
        launch(self.data<float>(), static_cast<float>(value));
    } else if (self.dtype() == DType::Float64) {
        launch(self.data<double>(), value);
    } else if (self.dtype() == DType::Int32) {
        launch(self.data<int32_t>(), static_cast<int32_t>(value));
    } else if (self.dtype() == DType::Int64) {
        launch(self.data<int64_t>(), static_cast<int64_t>(value));
    } else if (self.dtype() == DType::Float16) {
        __half h_value = __float2half(static_cast<float>(value));
        launch(reinterpret_cast<__half*>(self.data<Float16>()), h_value);
    } else if (self.dtype() == DType::BFloat16) {
        __nv_bfloat16 bf_value = __float2bfloat16(static_cast<float>(value));
        launch(reinterpret_cast<__nv_bfloat16*>(self.data<BFloat16>()), bf_value);
    } else if (self.dtype() == DType::Int8) {
        launch(self.data<int8_t>(), static_cast<int8_t>(value));
    } else if (self.dtype() == DType::UInt8) {
        launch(self.data<uint8_t>(), static_cast<uint8_t>(value));
    } else if (self.dtype() == DType::Bool) {
        launch(self.data<bool>(), value != 0.0f);
    } else {
        CUDA_CHECK(cudaFreeAsync(d_meta, stream));
        throw std::runtime_error("strided_fill: unsupported dtype");
    }

    CUDA_CHECK(cudaFreeAsync(d_meta, stream));
}

// Zeros kernel launcher - creates tensor filled with zeros
auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    // CUDA tensors are zero-initialized by default, but let's be explicit
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (dtype == DType::Float32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<float>(), 0.0f, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<double>(), 0.0, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int32_t>(), static_cast<int32_t>(0), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int64_t>(), static_cast<int64_t>(0), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float16) {
        __half zero_h = __float2half(0.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), zero_h, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::BFloat16) {
        __nv_bfloat16 zero_bf = __float2bfloat16(0.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), zero_bf, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Bool) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<bool>(), false, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int8_t>(), static_cast<int8_t>(0), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint8_t>(), static_cast<uint8_t>(0), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int16_t>(), static_cast<int16_t>(0), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint16_t>(), static_cast<uint16_t>(0), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint32_t>(), static_cast<uint32_t>(0), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint64_t>(), static_cast<uint64_t>(0), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Complex64) {
        cuFloatComplex zero_c = make_cuFloatComplex(0.0f, 0.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<cuFloatComplex*>(result.data_ptr()), zero_c, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Complex128) {
        cuDoubleComplex zero_z = make_cuDoubleComplex(0.0, 0.0);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<cuDoubleComplex*>(result.data_ptr()), zero_z, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for zeros operation");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Ones kernel launcher - creates tensor filled with ones
auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (dtype == DType::Float32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<float>(), 1.0f, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<double>(), 1.0, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int32_t>(), static_cast<int32_t>(1), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int64_t>(), static_cast<int64_t>(1), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float16) {
        __half one_h = __float2half(1.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), one_h, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::BFloat16) {
        __nv_bfloat16 one_bf = __float2bfloat16(1.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), one_bf, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Bool) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<bool>(), true, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int8_t>(), static_cast<int8_t>(1), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint8_t>(), static_cast<uint8_t>(1), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int16_t>(), static_cast<int16_t>(1), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint16_t>(), static_cast<uint16_t>(1), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint32_t>(), static_cast<uint32_t>(1), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint64_t>(), static_cast<uint64_t>(1), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Complex64) {
        cuFloatComplex one_c = make_cuFloatComplex(1.0f, 0.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<cuFloatComplex*>(result.data_ptr()), one_c, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Complex128) {
        cuDoubleComplex one_z = make_cuDoubleComplex(1.0, 0.0);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<cuDoubleComplex*>(result.data_ptr()), one_z, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for ones operation");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// Full kernel launcher - creates tensor filled with specified value
auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (dtype == DType::Float32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<float>(), static_cast<float>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<double>(), static_cast<double>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int32_t>(), static_cast<int32_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int64_t>(), static_cast<int64_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float16) {
        __half h_value = __float2half(value);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), h_value, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::BFloat16) {
        __nv_bfloat16 bf_value = __float2bfloat16(value);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), bf_value, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int8_t>(), static_cast<int8_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint8_t>(), static_cast<uint8_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int16_t>(), static_cast<int16_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint16_t>(), static_cast<uint16_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint32_t>(), static_cast<uint32_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::UInt64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint64_t>(), static_cast<uint64_t>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Bool) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<bool>(), static_cast<bool>(value), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for full operation");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// ============================================================================
// Random Number Generation (cuRAND)
// ============================================================================

// Kernel to initialize cuRAND states
__global__ void init_curand_states(curandState* states, unsigned long long seed, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        // Each thread gets different seed, a different sequence number, no offset
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Kernel for uniform random [0, 1) generation
__global__ void rand_kernel_device(float* output, curandState* states, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = curand_uniform(&states[idx]);
    }
}

// Kernel for normal distribution N(0,1) generation
__global__ void randn_kernel_device(float* output, curandState* states, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = curand_normal(&states[idx]);
    }
}

// FP16 uniform random kernel
__global__ void rand_kernel_f16(__half* output, curandState* states, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = curand_uniform(&states[idx]);
        output[idx] = __float2half(val);
    }
}

// FP16 normal distribution kernel
__global__ void randn_kernel_f16(__half* output, curandState* states, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = curand_normal(&states[idx]);
        output[idx] = __float2half(val);
    }
}

// BFloat16 uniform random kernel
__global__ void rand_kernel_bf16(__nv_bfloat16* output, curandState* states, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = curand_uniform(&states[idx]);
        output[idx] = __float2bfloat16(val);
    }
}

// BFloat16 normal distribution kernel
__global__ void randn_kernel_bf16(__nv_bfloat16* output, curandState* states, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = curand_normal(&states[idx]);
        output[idx] = __float2bfloat16(val);
    }
}

// Float-to-double conversion kernel for proper type conversion
__global__ void convert_float_to_double_kernel(const float* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<double>(input[idx]);
    }
}

// Rand kernel launcher - uniform random [0, 1)
auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("rand operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    // Allocate cuRAND states
    backend::CachedMemoryGuard d_states_guard(n * sizeof(curandState));
    auto* d_states = static_cast<curandState*>(d_states_guard.get());

    // Honor `tenzor::manual_seed`. Time-based seed in the unsetted case.
    uint64_t seed = ::tenzor::get_global_seed();

    init_curand_states<<<grid, block, 0, stream>>>(d_states, seed, n);
    CUDA_CHECK(cudaGetLastError());

    if (dtype == DType::Float32) {
        // Generate uniform random numbers
        rand_kernel_device<<<grid, block, 0, stream>>>(result.data<float>(), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        // For Float64, generate as float then convert properly
        backend::CachedMemoryGuard temp_float_guard(n * sizeof(float));
        auto* temp_float = static_cast<float*>(temp_float_guard.get());
        rand_kernel_device<<<grid, block, 0, stream>>>(temp_float, d_states, n);
        CUDA_CHECK(cudaGetLastError());

        // Convert float to double using proper conversion kernel
        double* output_double = result.data<double>();
        convert_float_to_double_kernel<<<grid, block, 0, stream>>>(temp_float, output_double, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float16) {
        rand_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::BFloat16) {
        rand_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    }

    return result;
}

// Randn kernel launcher - normal distribution N(0,1)
auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("randn operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    // Allocate cuRAND states
    backend::CachedMemoryGuard d_states_guard(n * sizeof(curandState));
    auto* d_states = static_cast<curandState*>(d_states_guard.get());

    // Honor `tenzor::manual_seed` for reproducibility. When the user has not
    // set a manual seed, `get_global_seed` returns a time-based value, so
    // we keep the existing entropy mix on top of that to preserve the
    // previous "random per call" behavior in the unsetted case.
    uint64_t seed = ::tenzor::get_global_seed();

    init_curand_states<<<grid, block, 0, stream>>>(d_states, seed, n);
    CUDA_CHECK(cudaGetLastError());

    if (dtype == DType::Float32) {
        randn_kernel_device<<<grid, block, 0, stream>>>(result.data<float>(), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        // For Float64, generate as float then convert properly
        backend::CachedMemoryGuard temp_float_guard(n * sizeof(float));
        auto* temp_float = static_cast<float*>(temp_float_guard.get());
        randn_kernel_device<<<grid, block, 0, stream>>>(temp_float, d_states, n);
        CUDA_CHECK(cudaGetLastError());

        // Convert float to double using proper conversion kernel
        double* output_double = result.data<double>();
        convert_float_to_double_kernel<<<grid, block, 0, stream>>>(temp_float, output_double, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float16) {
        randn_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::BFloat16) {
        randn_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    }

    return result;
}

// ============================================================================
// Randint - Random Integer Generation
// ============================================================================

// CUDA kernel: generate uniform float [0,1), scale to [low, high), cast to int
template<typename T>
__global__ void randint_kernel_device(T* output, curandState* states, int64_t n, int64_t low, int64_t high) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        // Generate uniform float in [0, 1)
        float r = curand_uniform(&states[idx]);
        // Scale to [low, high) and cast to integer type
        // curand_uniform returns (0, 1], so clamp to avoid hitting 'high'
        int64_t range = high - low;
        int64_t val = low + static_cast<int64_t>(r * static_cast<float>(range));
        // Clamp in case r == 1.0
        if (val >= high) val = high - 1;
        output[idx] = static_cast<T>(val);
    }
}

auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape,
                    DType dtype, Device device, cudaStream_t stream) -> Tensor {
    if (dtype != DType::Int32 && dtype != DType::Int64) {
        throw std::runtime_error("randint operation only supports Int32 and Int64 dtypes");
    }

    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    // Allocate cuRAND states
    backend::CachedMemoryGuard d_states_guard(n * sizeof(curandState));
    auto* d_states = static_cast<curandState*>(d_states_guard.get());

    // Honor `tenzor::manual_seed`. Time-based seed in the unsetted case.
    uint64_t seed = ::tenzor::get_global_seed();

    init_curand_states<<<grid, block, 0, stream>>>(d_states, seed, n);
    CUDA_CHECK(cudaGetLastError());

    if (dtype == DType::Int32) {
        randint_kernel_device<int32_t><<<grid, block, 0, stream>>>(
            result.data<int32_t>(), d_states, n, low, high);
        CUDA_CHECK(cudaGetLastError());
    } else {  // Int64
        randint_kernel_device<int64_t><<<grid, block, 0, stream>>>(
            result.data<int64_t>(), d_states, n, low, high);
        CUDA_CHECK(cudaGetLastError());
    }

    return result;
}

// ============================================================================
// Comparison Operations
// ============================================================================

// Comparison operation functors
struct EqOp {
    template<typename T>
    __device__ bool operator()(T a, T b) const { return a == b; }
};

// Specialization for __half
template<>
__device__ inline bool EqOp::operator()<__half>(__half a, __half b) const {
    return __heq(a, b);
}

struct NeOp {
    template<typename T>
    __device__ bool operator()(T a, T b) const { return a != b; }
};

template<>
__device__ inline bool NeOp::operator()<__half>(__half a, __half b) const {
    return __hne(a, b);
}

struct LtOp {
    template<typename T>
    __device__ bool operator()(T a, T b) const { return a < b; }
};

template<>
__device__ inline bool LtOp::operator()<__half>(__half a, __half b) const {
    return __hlt(a, b);
}

struct LeOp {
    template<typename T>
    __device__ bool operator()(T a, T b) const { return a <= b; }
};

template<>
__device__ inline bool LeOp::operator()<__half>(__half a, __half b) const {
    return __hle(a, b);
}

struct GtOp {
    template<typename T>
    __device__ bool operator()(T a, T b) const { return a > b; }
};

template<>
__device__ inline bool GtOp::operator()<__half>(__half a, __half b) const {
    return __hgt(a, b);
}

struct GeOp {
    template<typename T>
    __device__ bool operator()(T a, T b) const { return a >= b; }
};

template<>
__device__ inline bool GeOp::operator()<__half>(__half a, __half b) const {
    return __hge(a, b);
}

// Fast path: element-wise comparison (same shape)
template<typename T, typename Op>
__global__ void compare_kernel_device(const T* a, const T* b, bool* c, int64_t n, Op op) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = op(a[idx], b[idx]);
    }
}

// Generic broadcast comparison kernel
template<typename T, typename Op>
__global__ void broadcast_compare_kernel(
    const T* a, const T* b, bool* c,
    BroadcastMeta meta, int64_t ndim, int64_t n, Op op) {

    TENZOR_CUDA_KERNEL_LOOP(out_idx, n) {
        int64_t idx_a = 0;
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        // Convert flat index to multi-dimensional indices
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % meta.output_shape[i];
            tmp /= meta.output_shape[i];
            idx_a += coord * meta.strides_a[i];
            idx_b += coord * meta.strides_b[i];
        }

        c[out_idx] = op(a[idx_a], b[idx_b]);
    }
}

// Generic comparison launcher
template<typename Op>
auto compare_kernel_launcher(const Tensor& a, const Tensor& b, cudaStream_t stream, Op op) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype for comparison");
    }

    auto a_shape_span = a.shape();
    auto b_shape_span = b.shape();
    std::vector<int64_t> a_shape_vec(a_shape_span.begin(), a_shape_span.end());
    std::vector<int64_t> b_shape_vec(b_shape_span.begin(), b_shape_span.end());

    // Check if broadcastable
    if (!detail::are_broadcastable(a_shape_vec, b_shape_vec)) {
        throw std::runtime_error("Shapes are not broadcastable");
    }

    // Check for fast path (same shape, no broadcasting needed)
    if (detail::have_same_shape(a, b)) {
        int64_t n = a.numel();
        std::vector<int64_t> output_shape(a_shape_vec);
        Tensor result(output_shape, DType::Bool, a.device());  // Result is Bool type

        if (n == 0) {
            return result;
        }

        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);

        if (a.dtype() == DType::Float32) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<float>(), b.data<float>(), result.data<bool>(), n, op);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Float64) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<double>(), b.data<double>(), result.data<bool>(), n, op);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int32) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int32_t>(), b.data<int32_t>(), result.data<bool>(), n, op);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int64) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int64_t>(), b.data<int64_t>(), result.data<bool>(), n, op);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Float16) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a.data_ptr()),
                reinterpret_cast<const __half*>(b.data_ptr()),
                result.data<bool>(), n, op);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::BFloat16) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
                result.data<bool>(), n, op);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Int8) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int8_t>(), b.data<int8_t>(), result.data<bool>(), n, op);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::UInt8) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<uint8_t>(), b.data<uint8_t>(), result.data<bool>(), n, op);
            CUDA_CHECK(cudaGetLastError());
        } else if (a.dtype() == DType::Bool) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<bool>(), b.data<bool>(), result.data<bool>(), n, op);
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("Unsupported dtype for comparison operation");
        }

        CUDA_CHECK(cudaGetLastError());
        // NOTE: Removed cudaStreamSynchronize - async execution for performance
        return result;
    }

    // Broadcasting path
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(a_shape_vec, b_shape_vec);
    Tensor result(output_shape, DType::Bool, a.device());

    int64_t n = result.numel();
    if (n == 0) {
        return result;
    }

    // Compute strides
    std::vector<int64_t> strides_a = detail::compute_broadcast_strides(a_shape_vec, output_shape);
    std::vector<int64_t> strides_b = detail::compute_broadcast_strides(b_shape_vec, output_shape);

    BroadcastMeta meta = make_broadcast_meta(strides_a, strides_b, output_shape);

    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<bool>(),
            meta, ndim, n, op);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<bool>(),
            meta, ndim, n, op);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int32) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<bool>(),
            meta, ndim, n, op);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int64) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<bool>(),
            meta, ndim, n, op);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data_ptr()),
            reinterpret_cast<const __half*>(b.data_ptr()),
            result.data<bool>(),
            meta, ndim, n, op);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
            result.data<bool>(),
            meta, ndim, n, op);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int8) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<int8_t>(), b.data<int8_t>(), result.data<bool>(),
            meta, ndim, n, op);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::UInt8) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<uint8_t>(), b.data<uint8_t>(), result.data<bool>(),
            meta, ndim, n, op);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Bool) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<bool>(), b.data<bool>(), result.data<bool>(),
            meta, ndim, n, op);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Unsupported dtype for comparison operation");
    }
    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - cudaFree already synchronizes
    return result;
}

// Equal kernel launcher
auto eq_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    return compare_kernel_launcher(a, b, stream, EqOp());
}

// Not equal kernel launcher
auto ne_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    return compare_kernel_launcher(a, b, stream, NeOp());
}

// Less than kernel launcher
auto lt_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    return compare_kernel_launcher(a, b, stream, LtOp());
}

// Less than or equal kernel launcher
auto le_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    return compare_kernel_launcher(a, b, stream, LeOp());
}

// Greater than kernel launcher
auto gt_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    return compare_kernel_launcher(a, b, stream, GtOp());
}

// Greater than or equal kernel launcher
auto ge_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    return compare_kernel_launcher(a, b, stream, GeOp());
}

// Dot product kernel - element-wise multiply then sum
template<typename T>
__global__ void dot_product_kernel(const T* a, const T* b, T* output, int64_t n) {
    __shared__ T shared[256];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Grid-stride loop for element-wise multiplication
    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum += a[i] * b[i];
    }

    shared[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction
    if (tid < 32) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Dot product launcher
auto dot_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    // Verify both tensors are 1D
    if (a.ndim() != 1 || b.ndim() != 1) {
        throw std::invalid_argument("dot: inputs must be 1D tensors");
    }

    // Verify same shape
    if (a.shape()[0] != b.shape()[0]) {
        throw std::invalid_argument("dot: inputs must have the same length");
    }

    // Verify same dtype
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("dot: inputs must have the same dtype");
    }

    int64_t n = a.shape()[0];

    // Create scalar output tensor
    Tensor output({1}, a.dtype(), a.device());

    constexpr int block_size = 256;
    int num_blocks = std::min<int>((n + block_size - 1) / block_size, 1024);

    switch (a.dtype()) {
        case DType::Float32: {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* output_data = output.data<float>();

            if (num_blocks == 1) {
                dot_product_kernel<<<1, block_size, 0, stream>>>(a_data, b_data, output_data, n);
                CUDA_CHECK(cudaGetLastError());
            } else {
                // Two-phase reduction for large arrays
                backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(float));
                auto* d_temp = static_cast<float*>(d_temp_guard.get());
                dot_product_kernel<<<num_blocks, block_size, 0, stream>>>(a_data, b_data, d_temp, n);
                CUDA_CHECK(cudaGetLastError());
                dot_product_kernel<<<1, block_size, 0, stream>>>(d_temp, d_temp, output_data, num_blocks);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }
        case DType::Float64: {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* output_data = output.data<double>();

            if (num_blocks == 1) {
                dot_product_kernel<<<1, block_size, 0, stream>>>(a_data, b_data, output_data, n);
                CUDA_CHECK(cudaGetLastError());
            } else {
                // Two-phase reduction for large arrays
                backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(double));
                auto* d_temp = static_cast<double*>(d_temp_guard.get());
                dot_product_kernel<<<num_blocks, block_size, 0, stream>>>(a_data, b_data, d_temp, n);
                CUDA_CHECK(cudaGetLastError());
                dot_product_kernel<<<1, block_size, 0, stream>>>(d_temp, d_temp, output_data, num_blocks);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }
        default:
            throw std::runtime_error("dot: only Float32 and Float64 are supported");
    }

    // NOTE: CachedMemoryGuard handles cleanup via RAII

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in dot_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

// ============================================================================
// Adaptive Average Pooling 2D
// ============================================================================

// Forward kernel for adaptive average pooling
template<typename T>
__global__ void adaptive_avg_pool2d_forward_kernel(
    const T* input, T* output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;

    if (idx >= total) return;

    // Decode output index
    int64_t w_out = idx % W_out;
    int64_t h_out = (idx / W_out) % H_out;
    int64_t c = (idx / (W_out * H_out)) % C;
    int64_t n = idx / (W_out * H_out * C);

    // Calculate adaptive pooling window
    int64_t h_start = (h_out * H_in) / H_out;
    int64_t h_end = ((h_out + 1) * H_in) / H_out;
    int64_t w_start = (w_out * W_in) / W_out;
    int64_t w_end = ((w_out + 1) * W_in) / W_out;

    // Compute average
    T sum = T(0);
    int64_t count = 0;

    for (int64_t h = h_start; h < h_end; ++h) {
        for (int64_t w = w_start; w < w_end; ++w) {
            int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
            sum += input[input_idx];
            count++;
        }
    }

    output[idx] = sum / T(count);
}

// Backward kernel for adaptive average pooling
template<typename T>
__global__ void adaptive_avg_pool2d_backward_kernel(
    const T* grad_output, T* grad_input,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;

    if (idx >= total) return;

    // Decode output index
    int64_t w_out = idx % W_out;
    int64_t h_out = (idx / W_out) % H_out;
    int64_t c = (idx / (W_out * H_out)) % C;
    int64_t n = idx / (W_out * H_out * C);

    // Calculate adaptive pooling window
    int64_t h_start = (h_out * H_in) / H_out;
    int64_t h_end = ((h_out + 1) * H_in) / H_out;
    int64_t w_start = (w_out * W_in) / W_out;
    int64_t w_end = ((w_out + 1) * W_in) / W_out;

    int64_t count = (h_end - h_start) * (w_end - w_start);
    T grad_val = grad_output[idx] / T(count);

    // Distribute gradient to input positions
    for (int64_t h = h_start; h < h_end; ++h) {
        for (int64_t w = w_start; w < w_end; ++w) {
            int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
            atomicAdd(&grad_input[input_idx], grad_val);
        }
    }
}

// Launcher for adaptive avg pool 2d forward
auto adaptive_avg_pool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w, cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_in = shape[2];
    int64_t W_in = shape[3];

    Tensor output({N, C, output_h, output_w}, input.dtype(), input.device());

    int64_t total = N * C * output_h * output_w;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    if (input.dtype() == DType::Float32) {
        adaptive_avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            N, C, H_in, W_in, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        adaptive_avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            N, C, H_in, W_in, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        adaptive_avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            N, C, H_in, W_in, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        adaptive_avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            N, C, H_in, W_in, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("adaptive_avg_pool2d_forward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// Launcher for adaptive avg pool 2d backward
auto adaptive_avg_pool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in, cudaStream_t stream) -> Tensor {
    auto shape = grad_output.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_out = shape[2];
    int64_t W_out = shape[3];

    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // Initialize to zeros
    cudaMemsetAsync(grad_input.data_ptr(), 0, grad_input.numel() * dtype_size(grad_input.dtype()), stream);

    int64_t total = N * C * H_out * W_out;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    if (grad_output.dtype() == DType::Float32) {
        adaptive_avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, H_in, W_in, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::Float64) {
        adaptive_avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, H_in, W_in, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::Float16) {
        adaptive_avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<__half*>(grad_input.data_ptr()),
            N, C, H_in, W_in, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::BFloat16) {
        adaptive_avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()),
            N, C, H_in, W_in, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("adaptive_avg_pool2d_backward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

// ============================================================================
// Adaptive Max Pooling 2D
// ============================================================================

template<typename T>
__global__ void adaptive_max_pool2d_forward_kernel(
    const T* input, T* output, int64_t* indices,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (idx >= total) return;

    int64_t w_out = idx % W_out;
    int64_t h_out = (idx / W_out) % H_out;
    int64_t c = (idx / (W_out * H_out)) % C;
    int64_t n = idx / (W_out * H_out * C);

    int64_t h_start = (h_out * H_in) / H_out;
    int64_t h_end = ((h_out + 1) * H_in) / H_out;
    int64_t w_start = (w_out * W_in) / W_out;
    int64_t w_end = ((w_out + 1) * W_in) / W_out;

    T max_val = input[((n * C + c) * H_in + h_start) * W_in + w_start];
    int64_t max_idx = ((n * C + c) * H_in + h_start) * W_in + w_start;

    for (int64_t h = h_start; h < h_end; ++h) {
        for (int64_t w = w_start; w < w_end; ++w) {
            int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
            if (input[input_idx] > max_val) {
                max_val = input[input_idx];
                max_idx = input_idx;
            }
        }
    }

    output[idx] = max_val;
    indices[idx] = max_idx;
}

auto adaptive_max_pool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w, cudaStream_t stream) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], H_in = shape[2], W_in = shape[3];

    Tensor output({N, C, output_h, output_w}, input.dtype(), input.device());
    Tensor indices({N, C, output_h, output_w}, DType::Int64, input.device());

    int64_t total = N * C * output_h * output_w;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    if (input.dtype() == DType::Float32) {
        adaptive_max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, H_in, W_in, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        adaptive_max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, H_in, W_in, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        adaptive_max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            indices.data<int64_t>(),
            N, C, H_in, W_in, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        adaptive_max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            indices.data<int64_t>(),
            N, C, H_in, W_in, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("adaptive_max_pool2d_forward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return {output, indices};
}

template<typename T>
__global__ void adaptive_max_pool2d_backward_kernel(
    const T* grad_output, const int64_t* indices,
    T* grad_input, int64_t total_output) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output) return;
    atomicAdd(&grad_input[indices[idx]], grad_output[idx]);
}

auto adaptive_max_pool2d_backward(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor {
    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
    cudaMemsetAsync(grad_input.data_ptr(), 0, grad_input.numel() * dtype_size(grad_input.dtype()), stream);

    int64_t total = grad_output.numel();
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    if (grad_output.dtype() == DType::Float32) {
        adaptive_max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(), grad_input.data<float>(), total);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::Float64) {
        adaptive_max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(), grad_input.data<double>(), total);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::Float16) {
        adaptive_max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(grad_input.data_ptr()), total);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::BFloat16) {
        adaptive_max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            indices.data<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()), total);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("adaptive_max_pool2d_backward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}



// ============================================================================
// Max Pooling 2D
// ============================================================================

// Forward kernel for max pooling - returns output and indices
template<typename T>
__global__ void max_pool2d_forward_kernel(
    const T* input, T* output, int64_t* indices,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t kernel_size, int64_t stride, int64_t padding) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;

    if (idx >= total) return;

    // Decode output index
    int64_t w_out = idx % W_out;
    int64_t h_out = (idx / W_out) % H_out;
    int64_t c = (idx / (W_out * H_out)) % C;
    int64_t n = idx / (W_out * H_out * C);

    // Calculate pooling window bounds
    int64_t h_start = h_out * stride - padding;
    int64_t w_start = w_out * stride - padding;
    int64_t h_end = h_start + kernel_size;
    int64_t w_end = w_start + kernel_size;

    // Find max value and its index
    T max_val = T(-1e38);  // Use large negative number for initialization
    int64_t max_idx = 0;

    for (int64_t h = h_start; h < h_end; ++h) {
        for (int64_t w = w_start; w < w_end; ++w) {
            if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                T val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }
    }

    output[idx] = max_val;
    indices[idx] = max_idx;
}

// Backward kernel for max pooling
template<typename T>
__global__ void max_pool2d_backward_kernel(
    const T* grad_output, const int64_t* indices, T* grad_input,
    int64_t total_output, int64_t total_input) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_output) return;

    int64_t max_idx = indices[idx];
    if (max_idx >= 0 && max_idx < total_input) {
        atomicAdd(&grad_input[max_idx], grad_output[idx]);
    }
}

// Launcher for max pool 2d forward
auto max_pool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_in = shape[2];
    int64_t W_in = shape[3];

    // Calculate output dimensions
    int64_t H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    Tensor indices({N, C, H_out, W_out}, DType::Int64, input.device());

    int64_t total = N * C * H_out * W_out;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    if (input.dtype() == DType::Float32) {
        max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            indices.data<int64_t>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            indices.data<int64_t>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("max_pool2d_forward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return {output, indices};
}

// Launcher for max pool 2d backward
auto max_pool2d_backward(const Tensor& grad_output, const Tensor& indices,
                         int64_t H_in, int64_t W_in, cudaStream_t stream) -> Tensor {
    auto shape = grad_output.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];

    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // Initialize to zeros
    cudaMemsetAsync(grad_input.data_ptr(), 0, grad_input.numel() * dtype_size(grad_input.dtype()), stream);

    int64_t total_output = grad_output.numel();
    int64_t total_input = grad_input.numel();
    dim3 grid, block;
    compute_launch_config_1d(total_output, grid, block);

    if (grad_output.dtype() == DType::Float32) {
        max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(), grad_input.data<float>(),
            total_output, total_input);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::Float64) {
        max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(), grad_input.data<double>(),
            total_output, total_input);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::Float16) {
        max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(grad_input.data_ptr()),
            total_output, total_input);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::BFloat16) {
        max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            indices.data<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()),
            total_output, total_input);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("max_pool2d_backward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

// Public API wrappers without stream
auto max_pool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) -> std::pair<Tensor, Tensor> {
    return max_pool2d_forward(input, kernel_size, stride, padding, nullptr);
}

auto max_pool2d_backward(const Tensor& grad_output, const Tensor& indices, int64_t H_in, int64_t W_in) -> Tensor {
    return max_pool2d_backward(grad_output, indices, H_in, W_in, nullptr);
}

// ============================================================================
// Average Pooling 2D
// ============================================================================

// Forward kernel for average pooling
template<typename T>
__global__ void avg_pool2d_forward_kernel(
    const T* input, T* output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t kernel_size, int64_t stride, int64_t padding) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;

    if (idx >= total) return;

    // Decode output index
    int64_t w_out = idx % W_out;
    int64_t h_out = (idx / W_out) % H_out;
    int64_t c = (idx / (W_out * H_out)) % C;
    int64_t n = idx / (W_out * H_out * C);

    // Calculate pooling window bounds
    int64_t h_start = h_out * stride - padding;
    int64_t w_start = w_out * stride - padding;
    int64_t h_end = h_start + kernel_size;
    int64_t w_end = w_start + kernel_size;

    // Compute average value
    T sum = T(0);
    int64_t count = 0;

    for (int64_t h = h_start; h < h_end; ++h) {
        for (int64_t w = w_start; w < w_end; ++w) {
            if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                sum += input[input_idx];
                count++;
            }
        }
    }

    output[idx] = count > 0 ? sum / T(count) : T(0);
}

// Backward kernel for average pooling
template<typename T>
__global__ void avg_pool2d_backward_kernel(
    const T* grad_output, T* grad_input,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t kernel_size, int64_t stride, int64_t padding) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;

    if (idx >= total) return;

    // Decode output index
    int64_t w_out = idx % W_out;
    int64_t h_out = (idx / W_out) % H_out;
    int64_t c = (idx / (W_out * H_out)) % C;
    int64_t n = idx / (W_out * H_out * C);

    // Calculate pooling window bounds
    int64_t h_start = h_out * stride - padding;
    int64_t w_start = w_out * stride - padding;
    int64_t h_end = h_start + kernel_size;
    int64_t w_end = w_start + kernel_size;

    // Count valid elements in pooling window
    int64_t count = 0;
    for (int64_t h = h_start; h < h_end; ++h) {
        for (int64_t w = w_start; w < w_end; ++w) {
            if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                count++;
            }
        }
    }

    if (count == 0) return;

    T grad = grad_output[idx] / T(count);

    // Distribute gradient to input elements
    for (int64_t h = h_start; h < h_end; ++h) {
        for (int64_t w = w_start; w < w_end; ++w) {
            if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                atomicAdd(&grad_input[input_idx], grad);
            }
        }
    }
}

// Launcher for avg pool 2d forward
auto avg_pool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_in = shape[2];
    int64_t W_in = shape[3];

    // Calculate output dimensions
    int64_t H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());

    int64_t total = N * C * H_out * W_out;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    if (input.dtype() == DType::Float32) {
        avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("avg_pool2d_forward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// Launcher for avg pool 2d backward
auto avg_pool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in,
                         int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor {
    auto shape = grad_output.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_out = shape[2];
    int64_t W_out = shape[3];

    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // Initialize to zeros
    cudaMemsetAsync(grad_input.data_ptr(), 0, grad_input.numel() * dtype_size(grad_input.dtype()), stream);

    int64_t total_output = grad_output.numel();
    dim3 grid, block;
    compute_launch_config_1d(total_output, grid, block);

    if (grad_output.dtype() == DType::Float32) {
        avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::Float64) {
        avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::Float16) {
        avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<__half*>(grad_input.data_ptr()),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (grad_output.dtype() == DType::BFloat16) {
        avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("avg_pool2d_backward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

// Public API wrappers without stream
auto avg_pool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor {
    return avg_pool2d_forward(input, kernel_size, stride, padding, nullptr);
}

auto avg_pool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in,
                         int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor {
    return avg_pool2d_backward(grad_output, H_in, W_in, kernel_size, stride, padding, nullptr);
}

// ============================================================================
// Gather operation for relative position bias
// ============================================================================

template<typename T>
__global__ void gather_2d_kernel(
    const T* table, const int64_t* indices, T* output,
    int64_t num_positions, int64_t num_heads, int64_t table_stride) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_positions * num_positions * num_heads;

    if (idx >= total) return;

    int64_t h = idx % num_heads;
    int64_t j = (idx / num_heads) % num_positions;
    int64_t i = idx / (num_heads * num_positions);

    int64_t table_idx = indices[i * num_positions + j];
    output[idx] = table[table_idx * num_heads + h];
}

auto gather_relative_position_bias(const Tensor& table, const Tensor& indices,
                                   int64_t num_positions, int64_t num_heads,
                                   cudaStream_t stream) -> Tensor {
    // table: [table_size*table_size, num_heads]
    // indices: [num_positions, num_positions]
    // output: [num_positions, num_positions, num_heads]

    Tensor output({num_positions, num_positions, num_heads}, table.dtype(), table.device());

    int64_t total = num_positions * num_positions * num_heads;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    // Ensure indices are on the same device
    Tensor indices_device = indices.device() == table.device() ? indices : indices.to(table.device());

    if (table.dtype() == DType::Float32) {
        gather_2d_kernel<<<grid, block, 0, stream>>>(
            table.data<float>(), indices_device.data<int64_t>(), output.data<float>(),
            num_positions, num_heads, num_heads);
        CUDA_CHECK(cudaGetLastError());
    } else if (table.dtype() == DType::Float64) {
        gather_2d_kernel<<<grid, block, 0, stream>>>(
            table.data<double>(), indices_device.data<int64_t>(), output.data<double>(),
            num_positions, num_heads, num_heads);
        CUDA_CHECK(cudaGetLastError());
    } else if (table.dtype() == DType::Float16) {
        gather_2d_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(table.data_ptr()),
            indices_device.data<int64_t>(),
            reinterpret_cast<__half*>(output.data_ptr()),
            num_positions, num_heads, num_heads);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("gather_relative_position_bias: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// Shifted window mask creation
// ============================================================================

__global__ void create_window_mask_kernel(
    float* mask, int64_t H, int64_t W, int64_t window_size, int64_t shift_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;

    int64_t h = idx / W;
    int64_t w = idx % W;

    // Determine region based on position
    int64_t h_region = 0;
    int64_t w_region = 0;

    if (h >= H - shift_size) h_region = 2;
    else if (h >= H - window_size) h_region = 1;

    if (w >= W - shift_size) w_region = 2;
    else if (w >= W - window_size) w_region = 1;

    mask[idx] = static_cast<float>(h_region * 3 + w_region);
}

__global__ void create_attention_mask_kernel(
    const float* window_mask, float* attn_mask,
    int64_t num_windows, int64_t M) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_windows * M * M;

    if (idx >= total) return;

    int64_t j = idx % M;
    int64_t i = (idx / M) % M;
    int64_t w = idx / (M * M);

    float val_i = window_mask[w * M + i];
    float val_j = window_mask[w * M + j];

    attn_mask[idx] = (val_i != val_j) ? -100.0f : 0.0f;
}

__global__ void window_partition_kernel(
    const float* img_mask, float* window_mask,
    int64_t H, int64_t W, int64_t window_size,
    int64_t nH, int64_t nW, int64_t M) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = nH * nW * M;
    if (idx >= total) return;

    int64_t pos_in_window = idx % M;
    int64_t window_idx = idx / M;
    int64_t ww = window_idx % nW;
    int64_t wh = window_idx / nW;

    int64_t h_local = pos_in_window / window_size;
    int64_t w_local = pos_in_window % window_size;

    int64_t h_global = wh * window_size + h_local;
    int64_t w_global = ww * window_size + w_local;

    window_mask[idx] = img_mask[h_global * W + w_global];
}



// ============================================================================
// Public API wrappers (without explicit stream parameter)
// ============================================================================

auto adaptive_avg_pool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w) -> Tensor {
    return adaptive_avg_pool2d_forward(input, output_h, output_w, nullptr);
}

auto adaptive_avg_pool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in) -> Tensor {
    return adaptive_avg_pool2d_backward(grad_output, H_in, W_in, nullptr);
}

auto gather_relative_position_bias(const Tensor& table, const Tensor& indices,
                                   int64_t num_positions, int64_t num_heads) -> Tensor {
    return gather_relative_position_bias(table, indices, num_positions, num_heads, nullptr);
}

auto create_shifted_window_mask_cuda(int64_t H, int64_t W,
                                      int64_t window_size,
                                      int64_t shift_size,
                                      DType dtype) -> Tensor {
    // Create tensors on CUDA device
    Device cuda_device(Device::Type::CUDA, 0);

    // Step 1: Create window region mask
    Tensor img_mask({H * W}, DType::Float32, cuda_device);

    dim3 grid1, block1;
    compute_launch_config_1d(H * W, grid1, block1);
    create_window_mask_kernel<<<grid1, block1>>>(
        img_mask.data<float>(), H, W, window_size, shift_size);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Partition into windows
    int64_t num_windows = (H / window_size) * (W / window_size);
    int64_t M = window_size * window_size;

    // Reshape img_mask to window format. The actual window partitioning is
    // performed below by `window_partition_kernel`, which writes the per-window
    // [num_windows, M] view into this buffer.
    Tensor window_mask({num_windows, M}, DType::Float32, cuda_device);

    // Copy with window partitioning logic
    // For each window (h_w, w_w), copy the corresponding M elements
    float* window_data = window_mask.data<float>();
    const float* img_data = img_mask.data<float>();

    // Window partition: copy from img_mask to windows
    int64_t nH = H / window_size;
    int64_t nW = W / window_size;

    // Launch kernel to partition windows and compute attention mask
    int partition_threads = 256;
    int partition_blocks = (num_windows * M + partition_threads - 1) / partition_threads;
    window_partition_kernel<<<partition_blocks, partition_threads>>>(
        img_data, window_data, H, W, window_size, nH, nW, M);
    CUDA_CHECK(cudaGetLastError());

    // Create attention mask: mask[i, j] = -100 if window_mask[i] != window_mask[j]
    Tensor attn_mask({num_windows, M, M}, DType::Float32, cuda_device);
    float* attn_data = attn_mask.data<float>();

    int mask_threads = 256;
    int mask_blocks = (num_windows * M * M + mask_threads - 1) / mask_threads;
    create_attention_mask_kernel<<<mask_blocks, mask_threads>>>(
        window_data, attn_data, num_windows, M);
        CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cudaGetLastError());

    // Convert to target dtype
    if (dtype != DType::Float32) {
        return attn_mask.to(dtype);
    }
    return attn_mask;
}


// ============================================================================
// Creation Operations
// ============================================================================

template<typename T>
__global__ void arange_kernel_impl(T* output, T start, T step, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = start + static_cast<T>(idx) * step;
}

// Specialized arange kernel for Float16 - compute in float, store as half
__global__ void arange_kernel_f16(__half* output, float start, float step, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __float2half(start + static_cast<float>(idx) * step);
}

// Specialized arange kernel for BFloat16 - compute in float, store as bfloat16
__global__ void arange_kernel_bf16(__nv_bfloat16* output, float start, float step, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __float2bfloat16(start + static_cast<float>(idx) * step);
}

auto arange_kernel(float start, float end, float step, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    int64_t n = static_cast<int64_t>(std::ceil((end - start) / step));
    if (n <= 0) n = 0;
    Tensor output({n}, dtype, device);
    if (n == 0) return output;

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    if (dtype == DType::Float32) {
        arange_kernel_impl<float><<<num_blocks, block_size, 0, stream>>>(output.data<float>(), start, step, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        arange_kernel_impl<double><<<num_blocks, block_size, 0, stream>>>(output.data<double>(), static_cast<double>(start), static_cast<double>(step), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float16) {
        arange_kernel_f16<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<__half*>(output.data_ptr()), start, step, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::BFloat16) {
        arange_kernel_bf16<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), start, step, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int32) {
        arange_kernel_impl<int32_t><<<num_blocks, block_size, 0, stream>>>(output.data<int32_t>(), static_cast<int32_t>(start), static_cast<int32_t>(step), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int64) {
        arange_kernel_impl<int64_t><<<num_blocks, block_size, 0, stream>>>(output.data<int64_t>(), static_cast<int64_t>(start), static_cast<int64_t>(step), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("arange: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

template<typename T>
__global__ void linspace_kernel_impl(T* output, T start, T step, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = start + static_cast<T>(idx) * step;
}

// Specialized linspace kernel for Float16 - compute in float, store as half
__global__ void linspace_kernel_f16(__half* output, float start, float step, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __float2half(start + static_cast<float>(idx) * step);
}

// Specialized linspace kernel for BFloat16 - compute in float, store as bfloat16
__global__ void linspace_kernel_bf16(__nv_bfloat16* output, float start, float step, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __float2bfloat16(start + static_cast<float>(idx) * step);
}

auto linspace_kernel(float start, float end, int64_t steps, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    if (steps <= 0) return Tensor({0}, dtype, device);
    Tensor output({steps}, dtype, device);

    int block_size = 256;
    int num_blocks = (steps + block_size - 1) / block_size;

    if (dtype == DType::Float32) {
        float step_val = (steps > 1) ? (end - start) / static_cast<float>(steps - 1) : 0.0f;
        linspace_kernel_impl<float><<<num_blocks, block_size, 0, stream>>>(output.data<float>(), start, step_val, steps);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        double step_val = (steps > 1) ? (static_cast<double>(end) - static_cast<double>(start)) / static_cast<double>(steps - 1) : 0.0;
        linspace_kernel_impl<double><<<num_blocks, block_size, 0, stream>>>(output.data<double>(), static_cast<double>(start), step_val, steps);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float16) {
        float step_val = (steps > 1) ? (end - start) / static_cast<float>(steps - 1) : 0.0f;
        linspace_kernel_f16<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<__half*>(output.data_ptr()), start, step_val, steps);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::BFloat16) {
        float step_val = (steps > 1) ? (end - start) / static_cast<float>(steps - 1) : 0.0f;
        linspace_kernel_bf16<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), start, step_val, steps);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("linspace: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

template<typename T>
__global__ void eye_kernel_impl(T* output, int64_t rows, int64_t cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;
    int64_t r = idx / cols;
    int64_t c = idx % cols;
    output[idx] = (r == c) ? T(1) : T(0);
}

// Specialized eye kernel for Float16
__global__ void eye_kernel_f16(__half* output, int64_t rows, int64_t cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;
    int64_t r = idx / cols;
    int64_t c = idx % cols;
    output[idx] = (r == c) ? __float2half(1.0f) : __float2half(0.0f);
}

// Specialized eye kernel for BFloat16
__global__ void eye_kernel_bf16(__nv_bfloat16* output, int64_t rows, int64_t cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;
    int64_t r = idx / cols;
    int64_t c = idx % cols;
    output[idx] = (r == c) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
}

auto eye_kernel(int64_t n, int64_t m, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    if (m <= 0) m = n;
    Tensor output({n, m}, dtype, device);
    int64_t total = n * m;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    if (dtype == DType::Float32) {
        eye_kernel_impl<float><<<num_blocks, block_size, 0, stream>>>(output.data<float>(), n, m);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        eye_kernel_impl<double><<<num_blocks, block_size, 0, stream>>>(output.data<double>(), n, m);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float16) {
        eye_kernel_f16<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<__half*>(output.data_ptr()), n, m);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::BFloat16) {
        eye_kernel_bf16<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), n, m);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int32) {
        eye_kernel_impl<int32_t><<<num_blocks, block_size, 0, stream>>>(output.data<int32_t>(), n, m);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Int64) {
        eye_kernel_impl<int64_t><<<num_blocks, block_size, 0, stream>>>(output.data<int64_t>(), n, m);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("eye: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}


// ============================================================================
// Dispatch-Conformant Wrappers (SingleOutputKernelFn signature)
// ============================================================================
// These wrappers match Tensor(*)(std::span<const Tensor>, const OpAttributes&)
// for direct registration with register_single_output_kernel()

// Helper to extract stream from attrs or acquire from pool
static std::pair<cudaStream_t, cuda::StreamGuard> get_dispatch_stream(
    const OpAttributes& attrs, const Tensor& ref_tensor) {
    if (!attrs.empty() && attrs.has(AttrKey::Stream)) {
        auto stream = reinterpret_cast<cudaStream_t>(
            static_cast<uint64_t>(attrs.get_int(AttrKey::Stream, 0)));
        return {stream, cuda::StreamGuard{}};
    }
    int device_id = ref_tensor.device().index;
    auto guard = cuda::CUDAStreamPool::instance().acquire_guard(device_id);
    return {guard.get(), std::move(guard)};
}

// Helper: upcast FP8 inputs to Float32, perform op, downcast result
static bool is_fp8(DType dt) {
    return dt == DType::FP8_E4M3 || dt == DType::FP8_E5M2;
}

Tensor add_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    if (is_fp8(inputs[0].dtype())) {
        DType orig = inputs[0].dtype();
        return add_kernel(inputs[0].to(DType::Float32), inputs[1].to(DType::Float32), stream).to(orig);
    }
    return add_kernel(inputs[0], inputs[1], stream);
}

Tensor sub_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    if (is_fp8(inputs[0].dtype())) {
        DType orig = inputs[0].dtype();
        return sub_kernel(inputs[0].to(DType::Float32), inputs[1].to(DType::Float32), stream).to(orig);
    }
    return sub_kernel(inputs[0], inputs[1], stream);
}

Tensor mul_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    if (is_fp8(inputs[0].dtype())) {
        DType orig = inputs[0].dtype();
        return mul_kernel(inputs[0].to(DType::Float32), inputs[1].to(DType::Float32), stream).to(orig);
    }
    return mul_kernel(inputs[0], inputs[1], stream);
}

Tensor div_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    if (is_fp8(inputs[0].dtype())) {
        DType orig = inputs[0].dtype();
        return div_kernel(inputs[0].to(DType::Float32), inputs[1].to(DType::Float32), stream).to(orig);
    }
    return div_kernel(inputs[0], inputs[1], stream);
}

// Note: matmul_dispatch and dot_dispatch are defined in cublas_ops.cu
// since matmul_kernel and dot_kernel are implemented there

// Inplace dispatch wrappers (InplaceKernelFn signature)
Tensor& add_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, target);
    add_inplace_kernel(target, others[0], stream);
    return target;
}

Tensor& sub_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, target);
    sub_inplace_kernel(target, others[0], stream);
    return target;
}

Tensor& mul_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, target);
    mul_inplace_kernel(target, others[0], stream);
    return target;
}

Tensor& div_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, target);
    div_inplace_kernel(target, others[0], stream);
    return target;
}

// Unary operation dispatch wrappers
Tensor sqrt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return sqrt_kernel(inputs[0], stream);
}

Tensor neg_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return neg_kernel(inputs[0], stream);
}

Tensor abs_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return abs_kernel(inputs[0], stream);
}

Tensor sign_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return sign_kernel(inputs[0], stream);
}

Tensor log_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return log_kernel(inputs[0], stream);
}

Tensor exp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return exp_kernel(inputs[0], stream);
}

Tensor reciprocal_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return reciprocal_kernel(inputs[0], stream);
}

Tensor floor_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return floor_kernel(inputs[0], stream);
}

Tensor ceil_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return ceil_kernel(inputs[0], stream);
}

Tensor round_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return round_kernel(inputs[0], stream);
}

Tensor trunc_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return trunc_kernel(inputs[0], stream);
}

// Trigonometric dispatch wrappers
Tensor sin_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return sin_kernel(inputs[0], stream);
}

Tensor cos_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return cos_kernel(inputs[0], stream);
}

Tensor tan_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return tan_kernel(inputs[0], stream);
}

Tensor asin_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return asin_kernel(inputs[0], stream);
}

Tensor acos_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return acos_kernel(inputs[0], stream);
}

Tensor atan_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return atan_kernel(inputs[0], stream);
}

Tensor sinh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return sinh_kernel(inputs[0], stream);
}

Tensor cosh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return cosh_kernel(inputs[0], stream);
}

// Comparison dispatch wrappers
Tensor eq_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return eq_kernel(inputs[0], inputs[1], stream);
}

Tensor ne_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return ne_kernel(inputs[0], inputs[1], stream);
}

Tensor lt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return lt_kernel(inputs[0], inputs[1], stream);
}

Tensor le_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return le_kernel(inputs[0], inputs[1], stream);
}

Tensor gt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return gt_kernel(inputs[0], inputs[1], stream);
}

Tensor ge_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return ge_kernel(inputs[0], inputs[1], stream);
}

// ============================================================================
// FP8 device conversion functions
// ============================================================================

/// Convert a float32 value to FP8 E4M3 (1 sign, 4 exponent, 3 mantissa, bias=7, no inf)
__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float f) {
    uint32_t f_bits = __float_as_uint(f);
    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;

    uint8_t h_exp, h_mantissa;

    if (exp == 0xFF) {
        // Inf/NaN -> E4M3 NaN (no infinity in E4M3)
        h_exp = 0xF;
        h_mantissa = 0x7;
    } else if (exp == 0) {
        h_exp = 0;
        h_mantissa = 0;
    } else {
        int32_t new_exp = static_cast<int32_t>(exp) - 127 + 7;
        if (new_exp >= 0xF) {
            // Clamp to max finite (exp=0xE, mantissa=0x7 = 448)
            h_exp = 0xE;
            h_mantissa = 0x7;
        } else if (new_exp <= 0) {
            if (new_exp >= -3) {
                uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                h_mantissa = static_cast<uint8_t>((m >> 20) & 0x7);
                h_exp = 0;
            } else {
                h_exp = 0;
                h_mantissa = 0;
            }
        } else {
            h_exp = static_cast<uint8_t>(new_exp);
            h_mantissa = static_cast<uint8_t>((mantissa >> 20) & 0x7);
        }
    }
    return static_cast<uint8_t>((sign << 7) | (h_exp << 3) | h_mantissa);
}

/// Convert an FP8 E4M3 byte to float32
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mantissa = bits & 0x7;

    uint32_t f_exp, f_mantissa;

    if (exp == 0) {
        if (mantissa == 0) {
            f_exp = 0;
            f_mantissa = 0;
        } else {
            // Denormalized: normalize
            int e = -1;
            uint32_t m = mantissa;
            do { e++; m <<= 1; } while ((m & 0x8) == 0);
            f_exp = 127 - 7 - e;
            f_mantissa = (m & 0x7) << 20;
        }
    } else if (exp == 0xF && mantissa != 0) {
        // NaN
        f_exp = 0xFF;
        f_mantissa = mantissa << 20;
    } else {
        // Normalized (exp=0xF with mantissa=0 is max finite, not inf)
        f_exp = exp - 7 + 127;
        f_mantissa = mantissa << 20;
    }

    uint32_t f_bits = (sign << 31) | (f_exp << 23) | f_mantissa;
    return __uint_as_float(f_bits);
}

/// Convert a float32 value to FP8 E5M2 (1 sign, 5 exponent, 2 mantissa, bias=15, has inf)
__device__ __forceinline__ uint8_t float_to_fp8_e5m2(float f) {
    uint32_t f_bits = __float_as_uint(f);
    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;

    uint8_t h_exp, h_mantissa;

    if (exp == 0xFF) {
        h_exp = 0x1F;
        h_mantissa = mantissa ? 0x3 : 0;  // NaN vs Inf
    } else if (exp == 0) {
        h_exp = 0;
        h_mantissa = 0;
    } else {
        int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;
        if (new_exp >= 0x1F) {
            // Overflow to infinity
            h_exp = 0x1F;
            h_mantissa = 0;
        } else if (new_exp <= 0) {
            if (new_exp >= -2) {
                uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                h_mantissa = static_cast<uint8_t>((m >> 21) & 0x3);
                h_exp = 0;
            } else {
                h_exp = 0;
                h_mantissa = 0;
            }
        } else {
            h_exp = static_cast<uint8_t>(new_exp);
            h_mantissa = static_cast<uint8_t>((mantissa >> 21) & 0x3);
        }
    }
    return static_cast<uint8_t>((sign << 7) | (h_exp << 2) | h_mantissa);
}

/// Convert an FP8 E5M2 byte to float32
__device__ __forceinline__ float fp8_e5m2_to_float(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 2) & 0x1F;
    uint32_t mantissa = bits & 0x3;

    uint32_t f_exp, f_mantissa;

    if (exp == 0) {
        if (mantissa == 0) {
            f_exp = 0;
            f_mantissa = 0;
        } else {
            int e = -1;
            uint32_t m = mantissa;
            do { e++; m <<= 1; } while ((m & 0x4) == 0);
            f_exp = 127 - 15 - e;
            f_mantissa = (m & 0x3) << 21;
        }
    } else if (exp == 0x1F) {
        // Inf or NaN
        f_exp = 0xFF;
        f_mantissa = mantissa << 21;
    } else {
        f_exp = exp - 15 + 127;
        f_mantissa = mantissa << 21;
    }

    uint32_t f_bits = (sign << 31) | (f_exp << 23) | f_mantissa;
    return __uint_as_float(f_bits);
}

// ---- FP8 cast kernels ----

// Float32 -> FP8
__global__ void cast_f32_to_fp8_e4m3_kernel(const float* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e4m3(input[idx]);
    }
}
__global__ void cast_f32_to_fp8_e5m2_kernel(const float* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e5m2(input[idx]);
    }
}

// FP8 -> Float32
__global__ void cast_fp8_e4m3_to_f32_kernel(const uint8_t* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = fp8_e4m3_to_float(input[idx]);
    }
}
__global__ void cast_fp8_e5m2_to_f32_kernel(const uint8_t* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = fp8_e5m2_to_float(input[idx]);
    }
}

// Float16 -> FP8 (via Float32 intermediate)
__global__ void cast_f16_to_fp8_e4m3_kernel(const __half* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e4m3(__half2float(input[idx]));
    }
}
__global__ void cast_f16_to_fp8_e5m2_kernel(const __half* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e5m2(__half2float(input[idx]));
    }
}

// BFloat16 -> FP8 (via Float32 intermediate)
__global__ void cast_bf16_to_fp8_e4m3_kernel(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e4m3(__bfloat162float(input[idx]));
    }
}
__global__ void cast_bf16_to_fp8_e5m2_kernel(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e5m2(__bfloat162float(input[idx]));
    }
}

// FP8 -> Float16
__global__ void cast_fp8_e4m3_to_f16_kernel(const uint8_t* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(fp8_e4m3_to_float(input[idx]));
    }
}
__global__ void cast_fp8_e5m2_to_f16_kernel(const uint8_t* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(fp8_e5m2_to_float(input[idx]));
    }
}

// FP8 -> BFloat16
__global__ void cast_fp8_e4m3_to_bf16_kernel(const uint8_t* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2bfloat16(fp8_e4m3_to_float(input[idx]));
    }
}
__global__ void cast_fp8_e5m2_to_bf16_kernel(const uint8_t* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2bfloat16(fp8_e5m2_to_float(input[idx]));
    }
}

// FP8 -> integer/bool types (via Float32 intermediate)
template<typename To>
__global__ void cast_fp8_e4m3_to_kernel(const uint8_t* input, To* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<To>(fp8_e4m3_to_float(input[idx]));
    }
}
template<typename To>
__global__ void cast_fp8_e5m2_to_kernel(const uint8_t* input, To* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<To>(fp8_e5m2_to_float(input[idx]));
    }
}

// Integer/bool types -> FP8 (via Float32 intermediate)
template<typename From>
__global__ void cast_to_fp8_e4m3_kernel(const From* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e4m3(static_cast<float>(input[idx]));
    }
}
template<typename From>
__global__ void cast_to_fp8_e5m2_kernel(const From* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e5m2(static_cast<float>(input[idx]));
    }
}

// FP8 <-> FP8 cross-format
__global__ void cast_fp8_e4m3_to_e5m2_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e5m2(fp8_e4m3_to_float(input[idx]));
    }
}
__global__ void cast_fp8_e5m2_to_e4m3_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e4m3(fp8_e5m2_to_float(input[idx]));
    }
}

// FP8 -> Float64 (via Float32 intermediate)
__global__ void cast_fp8_e4m3_to_f64_kernel(const uint8_t* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<double>(fp8_e4m3_to_float(input[idx]));
    }
}
__global__ void cast_fp8_e5m2_to_f64_kernel(const uint8_t* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<double>(fp8_e5m2_to_float(input[idx]));
    }
}

// Float64 -> FP8 (via Float32 intermediate)
__global__ void cast_f64_to_fp8_e4m3_kernel(const double* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e4m3(static_cast<float>(input[idx]));
    }
}
__global__ void cast_f64_to_fp8_e5m2_kernel(const double* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float_to_fp8_e5m2(static_cast<float>(input[idx]));
    }
}

// ============================================================================
// Cast (dtype conversion) Kernels
// ============================================================================

template<typename From, typename To>
__global__ void cast_element_kernel(const From* input, To* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<To>(input[idx]);
    }
}

// Specializations for Float16 source (convert via float)
template<typename To>
__global__ void cast_from_f16_kernel(const __half* input, To* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<To>(__half2float(input[idx]));
    }
}

// Specialization for Float16 destination (convert via float with saturation)
template<typename From>
__global__ void cast_to_f16_kernel(const From* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float2half_sat(static_cast<float>(input[idx]));
    }
}

// Float16 -> Float16 (no-op copy)
__global__ void cast_f16_to_f16_kernel(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx];
    }
}

// Specializations for BFloat16 source (convert via float)
template<typename To>
__global__ void cast_from_bf16_kernel(const __nv_bfloat16* input, To* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<To>(__bfloat162float(input[idx]));
    }
}

// Specialization for BFloat16 destination (convert via float)
template<typename From>
__global__ void cast_to_bf16_kernel(const From* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2bfloat16(static_cast<float>(input[idx]));
    }
}

// BFloat16 -> BFloat16 (no-op copy)
__global__ void cast_bf16_to_bf16_kernel(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx];
    }
}

// Float16 -> BFloat16
__global__ void cast_f16_to_bf16_kernel(const __half* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2bfloat16(__half2float(input[idx]));
    }
}

// BFloat16 -> Float16
__global__ void cast_bf16_to_f16_kernel(const __nv_bfloat16* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = float2half_sat(__bfloat162float(input[idx]));
    }
}

/// Helper: launch a cast kernel from a standard (non-half) source type to the target dtype.
/// Returns the result tensor allocated with target dtype on the same device.
template<typename SrcT>
Tensor cast_from_standard(const Tensor& input, DType target_dtype, int64_t n,
                          dim3 grid, dim3 block, cudaStream_t stream) {
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, target_dtype, input.device());
    const SrcT* src = input.data<SrcT>();

    switch (target_dtype) {
        case DType::Float32:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<float>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::Float64:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<double>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::Float16:
            cast_to_f16_kernel<<<grid, block, 0, stream>>>(src,
                reinterpret_cast<__half*>(result.data<Float16>()), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::BFloat16:
            cast_to_bf16_kernel<<<grid, block, 0, stream>>>(src,
                reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::Int8:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<int8_t>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::Int16:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<int16_t>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::Int32:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<int32_t>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::Int64:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<int64_t>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::UInt8:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<uint8_t>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::UInt16:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<uint16_t>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::UInt32:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<uint32_t>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::UInt64:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<uint64_t>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        case DType::Bool:
            cast_element_kernel<<<grid, block, 0, stream>>>(src, result.data<bool>(), n);
            CUDA_CHECK(cudaGetLastError()); break;
        default:
            throw std::runtime_error("cast: unsupported target dtype");
    }
    return result;
}

// Complex64/Complex128 cast helpers. Storage is interleaved (re, im)
// pairs of float/double, so real→complex zero-fills the imag channel
// and complex→real drops it. Needed by any op that calls .to(Complex*)
// (the FFT parity tests hit this when promoting Float32 inputs before
// calling fft()).
__global__ void cast_f32_to_c64_kernel(const float* in, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(i, n) {
        out[2 * i]     = in[i];
        out[2 * i + 1] = 0.0f;
    }
}
__global__ void cast_f64_to_c128_kernel(const double* in, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(i, n) {
        out[2 * i]     = in[i];
        out[2 * i + 1] = 0.0;
    }
}
__global__ void cast_c64_to_f32_kernel(const float* in, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(i, n) { out[i] = in[2 * i]; }
}
__global__ void cast_c128_to_f64_kernel(const double* in, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(i, n) { out[i] = in[2 * i]; }
}
__global__ void cast_c64_to_c128_kernel(const float* in, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(i, n) {
        out[2 * i]     = static_cast<double>(in[2 * i]);
        out[2 * i + 1] = static_cast<double>(in[2 * i + 1]);
    }
}
__global__ void cast_c128_to_c64_kernel(const double* in, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(i, n) {
        out[2 * i]     = static_cast<float>(in[2 * i]);
        out[2 * i + 1] = static_cast<float>(in[2 * i + 1]);
    }
}

auto cuda_cast_kernel(const Tensor& input, DType target_dtype, cudaStream_t stream) -> Tensor {
    if (input.dtype() == target_dtype) {
        return input;  // No conversion needed
    }

    int64_t n = input.numel();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    Tensor result(shape, target_dtype, input.device());
    DType src_dtype = input.dtype();

    // Complex conversions: short-circuit before the main per-type dispatch
    // because Complex64/128 storage is interleaved pairs of real/imag, not
    // a single element the generic cast_from_standard template understands.
    if (src_dtype == DType::Float32 && target_dtype == DType::Complex64) {
        cast_f32_to_c64_kernel<<<grid, block, 0, stream>>>(
            input.data<float>(),
            reinterpret_cast<float*>(const_cast<void*>(result.data_ptr())), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (src_dtype == DType::Float64 && target_dtype == DType::Complex128) {
        cast_f64_to_c128_kernel<<<grid, block, 0, stream>>>(
            input.data<double>(),
            reinterpret_cast<double*>(const_cast<void*>(result.data_ptr())), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (src_dtype == DType::Complex64 && target_dtype == DType::Float32) {
        cast_c64_to_f32_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (src_dtype == DType::Complex128 && target_dtype == DType::Float64) {
        cast_c128_to_f64_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double*>(input.data_ptr()),
            result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (src_dtype == DType::Complex64 && target_dtype == DType::Complex128) {
        cast_c64_to_c128_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<double*>(const_cast<void*>(result.data_ptr())), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (src_dtype == DType::Complex128 && target_dtype == DType::Complex64) {
        cast_c128_to_c64_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double*>(input.data_ptr()),
            reinterpret_cast<float*>(const_cast<void*>(result.data_ptr())), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    // ---- Float16 source ----
    if (src_dtype == DType::Float16) {
        const __half* src = reinterpret_cast<const __half*>(input.data<Float16>());
        switch (target_dtype) {
            case DType::Float32:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<float>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Float64:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<double>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Float16:
                cast_f16_to_f16_kernel<<<grid, block, 0, stream>>>(src,
                    reinterpret_cast<__half*>(result.data<Float16>()), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::BFloat16:
                cast_f16_to_bf16_kernel<<<grid, block, 0, stream>>>(src,
                    reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int8:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<int8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int16:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<int16_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int32:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<int32_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int64:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<int64_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt8:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<uint8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt16:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<uint16_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt32:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<uint32_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt64:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<uint64_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Bool:
                cast_from_f16_kernel<<<grid, block, 0, stream>>>(src, result.data<bool>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::FP8_E4M3:
                cast_f16_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(src,
                    result.data<uint8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::FP8_E5M2:
                cast_f16_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(src,
                    result.data<uint8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            default:
                throw std::runtime_error("cast: unsupported target dtype for Float16 source");
        }
    }
    // ---- BFloat16 source ----
    else if (src_dtype == DType::BFloat16) {
        const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>());
        switch (target_dtype) {
            case DType::Float32:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<float>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Float64:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<double>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Float16:
                cast_bf16_to_f16_kernel<<<grid, block, 0, stream>>>(src,
                    reinterpret_cast<__half*>(result.data<Float16>()), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::BFloat16:
                cast_bf16_to_bf16_kernel<<<grid, block, 0, stream>>>(src,
                    reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int8:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<int8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int16:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<int16_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int32:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<int32_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int64:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<int64_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt8:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<uint8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt16:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<uint16_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt32:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<uint32_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt64:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<uint64_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Bool:
                cast_from_bf16_kernel<<<grid, block, 0, stream>>>(src, result.data<bool>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::FP8_E4M3:
                cast_bf16_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(src,
                    result.data<uint8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::FP8_E5M2:
                cast_bf16_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(src,
                    result.data<uint8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            default:
                throw std::runtime_error("cast: unsupported target dtype for BFloat16 source");
        }
    }
    // ---- FP8 source ----
    else if (src_dtype == DType::FP8_E4M3) {
        const uint8_t* src = input.data<uint8_t>();
        switch (target_dtype) {
            case DType::Float32:
                cast_fp8_e4m3_to_f32_kernel<<<grid, block, 0, stream>>>(src, result.data<float>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Float64:
                cast_fp8_e4m3_to_f64_kernel<<<grid, block, 0, stream>>>(src, result.data<double>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Float16:
                cast_fp8_e4m3_to_f16_kernel<<<grid, block, 0, stream>>>(src,
                    reinterpret_cast<__half*>(result.data<Float16>()), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::BFloat16:
                cast_fp8_e4m3_to_bf16_kernel<<<grid, block, 0, stream>>>(src,
                    reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::FP8_E4M3:
                // Same type, already handled above
                break;
            case DType::FP8_E5M2:
                cast_fp8_e4m3_to_e5m2_kernel<<<grid, block, 0, stream>>>(src,
                    result.data<uint8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int8:
                cast_fp8_e4m3_to_kernel<<<grid, block, 0, stream>>>(src, result.data<int8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int16:
                cast_fp8_e4m3_to_kernel<<<grid, block, 0, stream>>>(src, result.data<int16_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int32:
                cast_fp8_e4m3_to_kernel<<<grid, block, 0, stream>>>(src, result.data<int32_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int64:
                cast_fp8_e4m3_to_kernel<<<grid, block, 0, stream>>>(src, result.data<int64_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt8:
                cast_fp8_e4m3_to_kernel<<<grid, block, 0, stream>>>(src, result.data<uint8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt16:
                cast_fp8_e4m3_to_kernel<<<grid, block, 0, stream>>>(src, result.data<uint16_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt32:
                cast_fp8_e4m3_to_kernel<<<grid, block, 0, stream>>>(src, result.data<uint32_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt64:
                cast_fp8_e4m3_to_kernel<<<grid, block, 0, stream>>>(src, result.data<uint64_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Bool:
                cast_fp8_e4m3_to_kernel<<<grid, block, 0, stream>>>(src, result.data<bool>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            default:
                throw std::runtime_error("cast: unsupported target dtype for FP8_E4M3 source");
        }
    }
    else if (src_dtype == DType::FP8_E5M2) {
        const uint8_t* src = input.data<uint8_t>();
        switch (target_dtype) {
            case DType::Float32:
                cast_fp8_e5m2_to_f32_kernel<<<grid, block, 0, stream>>>(src, result.data<float>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Float64:
                cast_fp8_e5m2_to_f64_kernel<<<grid, block, 0, stream>>>(src, result.data<double>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Float16:
                cast_fp8_e5m2_to_f16_kernel<<<grid, block, 0, stream>>>(src,
                    reinterpret_cast<__half*>(result.data<Float16>()), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::BFloat16:
                cast_fp8_e5m2_to_bf16_kernel<<<grid, block, 0, stream>>>(src,
                    reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::FP8_E4M3:
                cast_fp8_e5m2_to_e4m3_kernel<<<grid, block, 0, stream>>>(src,
                    result.data<uint8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::FP8_E5M2:
                // Same type, already handled above
                break;
            case DType::Int8:
                cast_fp8_e5m2_to_kernel<<<grid, block, 0, stream>>>(src, result.data<int8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int16:
                cast_fp8_e5m2_to_kernel<<<grid, block, 0, stream>>>(src, result.data<int16_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int32:
                cast_fp8_e5m2_to_kernel<<<grid, block, 0, stream>>>(src, result.data<int32_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Int64:
                cast_fp8_e5m2_to_kernel<<<grid, block, 0, stream>>>(src, result.data<int64_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt8:
                cast_fp8_e5m2_to_kernel<<<grid, block, 0, stream>>>(src, result.data<uint8_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt16:
                cast_fp8_e5m2_to_kernel<<<grid, block, 0, stream>>>(src, result.data<uint16_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt32:
                cast_fp8_e5m2_to_kernel<<<grid, block, 0, stream>>>(src, result.data<uint32_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::UInt64:
                cast_fp8_e5m2_to_kernel<<<grid, block, 0, stream>>>(src, result.data<uint64_t>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            case DType::Bool:
                cast_fp8_e5m2_to_kernel<<<grid, block, 0, stream>>>(src, result.data<bool>(), n);
                CUDA_CHECK(cudaGetLastError()); break;
            default:
                throw std::runtime_error("cast: unsupported target dtype for FP8_E5M2 source");
        }
    }
    // ---- Standard source types (non-half, non-FP8) ----
    else {
        if (target_dtype == DType::FP8_E4M3) {
            uint8_t* dst = result.data<uint8_t>();
            switch (src_dtype) {
                case DType::Float32:
                    cast_f32_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<float>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Float64:
                    cast_f64_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<double>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Int8:
                    cast_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<int8_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Int16:
                    cast_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<int16_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Int32:
                    cast_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<int32_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Int64:
                    cast_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<int64_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::UInt8:
                    cast_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<uint8_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::UInt16:
                    cast_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<uint16_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::UInt32:
                    cast_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<uint32_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::UInt64:
                    cast_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<uint64_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Bool:
                    cast_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(input.data<bool>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                default:
                    throw std::runtime_error("cast: unsupported source dtype for FP8_E4M3 target");
            }
        } else if (target_dtype == DType::FP8_E5M2) {
            uint8_t* dst = result.data<uint8_t>();
            switch (src_dtype) {
                case DType::Float32:
                    cast_f32_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<float>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Float64:
                    cast_f64_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<double>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Int8:
                    cast_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<int8_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Int16:
                    cast_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<int16_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Int32:
                    cast_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<int32_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Int64:
                    cast_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<int64_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::UInt8:
                    cast_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<uint8_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::UInt16:
                    cast_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<uint16_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::UInt32:
                    cast_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<uint32_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::UInt64:
                    cast_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<uint64_t>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                case DType::Bool:
                    cast_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(input.data<bool>(), dst, n);
                    CUDA_CHECK(cudaGetLastError()); break;
                default:
                    throw std::runtime_error("cast: unsupported source dtype for FP8_E5M2 target");
            }
        } else {
        switch (src_dtype) {
            case DType::Float32:
                result = cast_from_standard<float>(input, target_dtype, n, grid, block, stream); break;
            case DType::Float64:
                result = cast_from_standard<double>(input, target_dtype, n, grid, block, stream); break;
            case DType::Int8:
                result = cast_from_standard<int8_t>(input, target_dtype, n, grid, block, stream); break;
            case DType::Int16:
                result = cast_from_standard<int16_t>(input, target_dtype, n, grid, block, stream); break;
            case DType::Int32:
                result = cast_from_standard<int32_t>(input, target_dtype, n, grid, block, stream); break;
            case DType::Int64:
                result = cast_from_standard<int64_t>(input, target_dtype, n, grid, block, stream); break;
            case DType::UInt8:
                result = cast_from_standard<uint8_t>(input, target_dtype, n, grid, block, stream); break;
            case DType::UInt16:
                result = cast_from_standard<uint16_t>(input, target_dtype, n, grid, block, stream); break;
            case DType::UInt32:
                result = cast_from_standard<uint32_t>(input, target_dtype, n, grid, block, stream); break;
            case DType::UInt64:
                result = cast_from_standard<uint64_t>(input, target_dtype, n, grid, block, stream); break;
            case DType::Bool:
                result = cast_from_standard<bool>(input, target_dtype, n, grid, block, stream); break;
            default:
                throw std::runtime_error("cast: unsupported source dtype");
        }
        }
    }

    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor cast_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    // Target dtype is passed via attributes
    if (!attrs.has(AttrKey::TargetDtype)) {
        throw std::runtime_error("cast: missing 'target_dtype' attribute");
    }
    DType target_dtype = static_cast<DType>(attrs.get_int(AttrKey::TargetDtype));
    return cuda_cast_kernel(inputs[0], target_dtype, stream);
}

// ============================================================================
// Extended Unary Math Kernels: log2, log10, log1p, exp2, expm1, erf, erfc
// ============================================================================

// --- log2 ---
__global__ void log2_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = log2f(input[idx]); }
}
__global__ void log2_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = log2(input[idx]); }
}
__global__ void log2_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(log2f(__half2float(input[idx]))); }
}
__global__ void log2_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(log2f(__bfloat162float(input[idx]))); }
}

auto log2_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        log2_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        log2_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        log2_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        log2_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("log2 operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// --- log10 ---
__global__ void log10_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = log10f(input[idx]); }
}
__global__ void log10_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = log10(input[idx]); }
}
__global__ void log10_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(log10f(__half2float(input[idx]))); }
}
__global__ void log10_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(log10f(__bfloat162float(input[idx]))); }
}

auto log10_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        log10_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        log10_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        log10_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        log10_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("log10 operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// --- log1p ---
__global__ void log1p_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = log1pf(input[idx]); }
}
__global__ void log1p_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = log1p(input[idx]); }
}
__global__ void log1p_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(log1pf(__half2float(input[idx]))); }
}
__global__ void log1p_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(log1pf(__bfloat162float(input[idx]))); }
}

auto log1p_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        log1p_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        log1p_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        log1p_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        log1p_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("log1p operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// --- exp2 ---
__global__ void exp2_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = exp2f(input[idx]); }
}
__global__ void exp2_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = exp2(input[idx]); }
}
__global__ void exp2_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(exp2f(__half2float(input[idx]))); }
}
__global__ void exp2_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(exp2f(__bfloat162float(input[idx]))); }
}

auto exp2_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        exp2_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        exp2_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        exp2_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        exp2_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("exp2 operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// --- expm1 ---
__global__ void expm1_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = expm1f(input[idx]); }
}
__global__ void expm1_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = expm1(input[idx]); }
}
__global__ void expm1_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(expm1f(__half2float(input[idx]))); }
}
__global__ void expm1_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(expm1f(__bfloat162float(input[idx]))); }
}

auto expm1_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        expm1_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        expm1_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        expm1_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        expm1_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("expm1 operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// --- erf ---
__global__ void erf_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = erff(input[idx]); }
}
__global__ void erf_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = erf(input[idx]); }
}
__global__ void erf_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(erff(__half2float(input[idx]))); }
}
__global__ void erf_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(erff(__bfloat162float(input[idx]))); }
}

auto erf_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        erf_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        erf_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        erf_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        erf_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("erf operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// --- erfc ---
__global__ void erfc_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = erfcf(input[idx]); }
}
__global__ void erfc_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = erfc(input[idx]); }
}
__global__ void erfc_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(erfcf(__half2float(input[idx]))); }
}
__global__ void erfc_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(erfcf(__bfloat162float(input[idx]))); }
}

auto erfc_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        erfc_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        erfc_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        erfc_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        erfc_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("erfc operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// ============================================================================
// Bool Predicate Kernels: isnan, isinf, isfinite
// ============================================================================

// --- isnan ---
__global__ void isnan_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(isnan(input[idx]) ? 1 : 0); }
}
__global__ void isnan_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(isnan(input[idx]) ? 1 : 0); }
}
__global__ void isnan_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(__hisnan(input[idx]) ? 1 : 0); }
}
__global__ void isnan_kernel_bf16(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(isnan(__bfloat162float(input[idx])) ? 1 : 0); }
}

auto isnan_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        isnan_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        isnan_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        isnan_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        isnan_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("isnan operation only supports floating point dtypes");
    }
    return result;
}

// --- isinf ---
__global__ void isinf_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(isinf(input[idx]) ? 1 : 0); }
}
__global__ void isinf_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(isinf(input[idx]) ? 1 : 0); }
}
__global__ void isinf_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(__hisinf(input[idx]) ? 1 : 0); }
}
__global__ void isinf_kernel_bf16(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(isinf(__bfloat162float(input[idx])) ? 1 : 0); }
}

auto isinf_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        isinf_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        isinf_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        isinf_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        isinf_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("isinf operation only supports floating point dtypes");
    }
    return result;
}

// --- isfinite ---
__global__ void isfinite_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(isfinite(input[idx]) ? 1 : 0); }
}
__global__ void isfinite_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(isfinite(input[idx]) ? 1 : 0); }
}
__global__ void isfinite_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        output[idx] = static_cast<uint8_t>(isfinite(val) ? 1 : 0);
    }
}
__global__ void isfinite_kernel_bf16(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = __bfloat162float(input[idx]);
        output[idx] = static_cast<uint8_t>(isfinite(val) ? 1 : 0);
    }
}

auto isfinite_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        isfinite_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        isfinite_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        isfinite_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        isfinite_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("isfinite operation only supports floating point dtypes");
    }
    return result;
}

// ============================================================================
// Binary Math Kernels: atan2, fmod, remainder
// ============================================================================

// --- atan2 ---
__global__ void atan2_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = atan2f(a[idx], b[idx]); }
}
__global__ void atan2_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = atan2(a[idx], b[idx]); }
}
__global__ void atan2_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(atan2f(__half2float(a[idx]), __half2float(b[idx]))); }
}
__global__ void atan2_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(atan2f(__bfloat162float(a[idx]), __bfloat162float(b[idx]))); }
}

auto atan2_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("atan2: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        atan2_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        atan2_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        atan2_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        atan2_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("atan2 operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// --- fmod ---
__global__ void fmod_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = fmodf(a[idx], b[idx]); }
}
__global__ void fmod_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = fmod(a[idx], b[idx]); }
}
__global__ void fmod_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(fmodf(__half2float(a[idx]), __half2float(b[idx]))); }
}
__global__ void fmod_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(fmodf(__bfloat162float(a[idx]), __bfloat162float(b[idx]))); }
}

auto fmod_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("fmod: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        fmod_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        fmod_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        fmod_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        fmod_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("fmod operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// --- remainder ---
__global__ void remainder_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = remainderf(a[idx], b[idx]); }
}
__global__ void remainder_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = remainder(a[idx], b[idx]); }
}
__global__ void remainder_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(remainderf(__half2float(a[idx]), __half2float(b[idx]))); }
}
__global__ void remainder_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(remainderf(__bfloat162float(a[idx]), __bfloat162float(b[idx]))); }
}

auto remainder_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("remainder: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        remainder_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        remainder_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        remainder_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        remainder_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("remainder operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// ============================================================================
// Ternary Kernel: lerp
// ============================================================================

__global__ void lerp_kernel_f32(const float* start, const float* end, const float* weight, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = start[idx] + weight[idx] * (end[idx] - start[idx]); }
}
__global__ void lerp_kernel_f64(const double* start, const double* end, const double* weight, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = start[idx] + weight[idx] * (end[idx] - start[idx]); }
}
__global__ void lerp_kernel_f16(const __half* start, const __half* end, const __half* weight, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float s = __half2float(start[idx]);
        float e = __half2float(end[idx]);
        float w = __half2float(weight[idx]);
        output[idx] = __float2half(s + w * (e - s));
    }
}
__global__ void lerp_kernel_bf16(const __nv_bfloat16* start, const __nv_bfloat16* end, const __nv_bfloat16* weight, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float s = __bfloat162float(start[idx]);
        float e = __bfloat162float(end[idx]);
        float w = __bfloat162float(weight[idx]);
        output[idx] = __float2bfloat16(s + w * (e - s));
    }
}

auto lerp_kernel(const Tensor& start, const Tensor& end, const Tensor& weight, cudaStream_t stream) -> Tensor {
    if (start.dtype() != end.dtype() || start.dtype() != weight.dtype())
        throw std::runtime_error("lerp: all tensors must have the same dtype");
    int64_t n = start.numel();
    // Broadcast weight to match start/end shape if needed
    Tensor w_bcast = weight;
    if (weight.numel() != n) {
        std::vector<int64_t> target_shape(start.shape().begin(), start.shape().end());
        w_bcast = weight.expand(target_shape).contiguous();
    }
    std::vector<int64_t> shape(start.shape().begin(), start.shape().end());
    Tensor result(shape, start.dtype(), start.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (start.dtype() == DType::Float32) {
        lerp_kernel_f32<<<grid, block, 0, stream>>>(start.data<float>(), end.data<float>(), w_bcast.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (start.dtype() == DType::Float64) {
        lerp_kernel_f64<<<grid, block, 0, stream>>>(start.data<double>(), end.data<double>(), w_bcast.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (start.dtype() == DType::Float16) {
        lerp_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(start.data<Float16>()),
            reinterpret_cast<const __half*>(end.data<Float16>()),
            reinterpret_cast<const __half*>(w_bcast.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (start.dtype() == DType::BFloat16) {
        lerp_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(start.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(end.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(w_bcast.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("lerp operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
    return result;
}

// ============================================================================
// Logical Kernels: logical_and, logical_or, logical_not, logical_xor
// ============================================================================

// Generic templated device kernels for logical ops (works for any numeric type)
template<typename T>
__global__ void logical_and_kernel_device(const T* a, const T* b, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((a[idx] != T(0)) && (b[idx] != T(0)) ? 1 : 0);
    }
}
template<typename T>
__global__ void logical_or_kernel_device(const T* a, const T* b, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((a[idx] != T(0)) || (b[idx] != T(0)) ? 1 : 0);
    }
}
template<typename T>
__global__ void logical_not_kernel_device(const T* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(input[idx] == T(0) ? 1 : 0);
    }
}
template<typename T>
__global__ void logical_xor_kernel_device(const T* a, const T* b, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        bool va = (a[idx] != T(0));
        bool vb = (b[idx] != T(0));
        output[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
    }
}

// FP16 specializations for logical ops
__global__ void logical_and_kernel_f16(const __half* a, const __half* b, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((__half2float(a[idx]) != 0.0f) && (__half2float(b[idx]) != 0.0f) ? 1 : 0);
    }
}
__global__ void logical_or_kernel_f16(const __half* a, const __half* b, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((__half2float(a[idx]) != 0.0f) || (__half2float(b[idx]) != 0.0f) ? 1 : 0);
    }
}
__global__ void logical_not_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(__half2float(input[idx]) == 0.0f ? 1 : 0);
    }
}
__global__ void logical_xor_kernel_f16(const __half* a, const __half* b, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        bool va = (__half2float(a[idx]) != 0.0f);
        bool vb = (__half2float(b[idx]) != 0.0f);
        output[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
    }
}

// BF16 specializations for logical ops
__global__ void logical_and_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((__bfloat162float(a[idx]) != 0.0f) && (__bfloat162float(b[idx]) != 0.0f) ? 1 : 0);
    }
}
__global__ void logical_or_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((__bfloat162float(a[idx]) != 0.0f) || (__bfloat162float(b[idx]) != 0.0f) ? 1 : 0);
    }
}
__global__ void logical_not_kernel_bf16(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(__bfloat162float(input[idx]) == 0.0f ? 1 : 0);
    }
}
__global__ void logical_xor_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        bool va = (__bfloat162float(a[idx]) != 0.0f);
        bool vb = (__bfloat162float(b[idx]) != 0.0f);
        output[idx] = static_cast<uint8_t>((va != vb) ? 1 : 0);
    }
}

// Helper macro for binary logical host wrappers to avoid code duplication
#define DEFINE_BINARY_LOGICAL_KERNEL(name) \
auto name##_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor { \
    if (a.dtype() != b.dtype()) throw std::runtime_error(#name ": tensors must have the same dtype"); \
    int64_t n = a.numel(); \
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end()); \
    Tensor result(shape, DType::Bool, a.device()); \
    dim3 grid, block; \
    compute_launch_config_1d(n, grid, block); \
    if (a.dtype() == DType::Float32) { \
        name##_kernel_device<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<uint8_t>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } else if (a.dtype() == DType::Float64) { \
        name##_kernel_device<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<uint8_t>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } else if (a.dtype() == DType::Int32) { \
        name##_kernel_device<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<uint8_t>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } else if (a.dtype() == DType::Int64) { \
        name##_kernel_device<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<uint8_t>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } else if (a.dtype() == DType::Int8) { \
        name##_kernel_device<<<grid, block, 0, stream>>>(a.data<int8_t>(), b.data<int8_t>(), result.data<uint8_t>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } else if (a.dtype() == DType::UInt8) { \
        name##_kernel_device<<<grid, block, 0, stream>>>(a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } else if (a.dtype() == DType::Bool) { \
        name##_kernel_device<<<grid, block, 0, stream>>>(a.data<bool>(), b.data<bool>(), result.data<uint8_t>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } else if (a.dtype() == DType::Float16) { \
        name##_kernel_f16<<<grid, block, 0, stream>>>( \
            reinterpret_cast<const __half*>(a.data<Float16>()), \
            reinterpret_cast<const __half*>(b.data<Float16>()), result.data<uint8_t>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } else if (a.dtype() == DType::BFloat16) { \
        name##_kernel_bf16<<<grid, block, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()), \
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()), result.data<uint8_t>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } else { \
        throw std::runtime_error(#name " operation: unsupported dtype"); \
    } \
    return result; \
}

DEFINE_BINARY_LOGICAL_KERNEL(logical_and)
DEFINE_BINARY_LOGICAL_KERNEL(logical_or)
DEFINE_BINARY_LOGICAL_KERNEL(logical_xor)

#undef DEFINE_BINARY_LOGICAL_KERNEL

auto logical_not_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        logical_not_kernel_device<<<grid, block, 0, stream>>>(input.data<float>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        logical_not_kernel_device<<<grid, block, 0, stream>>>(input.data<double>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int32) {
        logical_not_kernel_device<<<grid, block, 0, stream>>>(input.data<int32_t>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int64) {
        logical_not_kernel_device<<<grid, block, 0, stream>>>(input.data<int64_t>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int8) {
        logical_not_kernel_device<<<grid, block, 0, stream>>>(input.data<int8_t>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::UInt8) {
        logical_not_kernel_device<<<grid, block, 0, stream>>>(input.data<uint8_t>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Bool) {
        logical_not_kernel_device<<<grid, block, 0, stream>>>(input.data<bool>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        logical_not_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        logical_not_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("logical_not operation: unsupported dtype");
    }
    return result;
}

// ============================================================================
// Element-wise Min/Max Kernels: minimum, maximum
// ============================================================================

// --- minimum ---
__global__ void minimum_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = fminf(a[idx], b[idx]); }
}
__global__ void minimum_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = fmin(a[idx], b[idx]); }
}
__global__ void minimum_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(fminf(__half2float(a[idx]), __half2float(b[idx]))); }
}
__global__ void minimum_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(fminf(__bfloat162float(a[idx]), __bfloat162float(b[idx]))); }
}
template<typename T>
__global__ void minimum_kernel_int(const T* a, const T* b, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx]; }
}

auto minimum_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("minimum: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        minimum_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        minimum_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        minimum_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        minimum_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int32) {
        minimum_kernel_int<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int64) {
        minimum_kernel_int<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("minimum operation: unsupported dtype");
    }
    return result;
}

// --- maximum ---
__global__ void maximum_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = fmaxf(a[idx], b[idx]); }
}
__global__ void maximum_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = fmax(a[idx], b[idx]); }
}
__global__ void maximum_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2half(fmaxf(__half2float(a[idx]), __half2float(b[idx]))); }
}
__global__ void maximum_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = __float2bfloat16(fmaxf(__bfloat162float(a[idx]), __bfloat162float(b[idx]))); }
}
template<typename T>
__global__ void maximum_kernel_int(const T* a, const T* b, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx]; }
}

auto maximum_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("maximum: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        maximum_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        maximum_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        maximum_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        maximum_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int32) {
        maximum_kernel_int<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Int64) {
        maximum_kernel_int<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("maximum operation: unsupported dtype");
    }
    return result;
}

// ============================================================================
// Dispatch Wrappers for New Kernels
// ============================================================================

// Extended unary math dispatch wrappers
Tensor log2_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return log2_kernel(inputs[0], stream);
}
Tensor log10_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return log10_kernel(inputs[0], stream);
}
Tensor log1p_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return log1p_kernel(inputs[0], stream);
}
Tensor exp2_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return exp2_kernel(inputs[0], stream);
}
Tensor expm1_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return expm1_kernel(inputs[0], stream);
}
Tensor erf_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return erf_kernel(inputs[0], stream);
}
Tensor erfc_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return erfc_kernel(inputs[0], stream);
}

// Bool predicate dispatch wrappers
Tensor isnan_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return isnan_kernel(inputs[0], stream);
}
Tensor isinf_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return isinf_kernel(inputs[0], stream);
}
Tensor isfinite_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return isfinite_kernel(inputs[0], stream);
}

// Binary math dispatch wrappers
Tensor atan2_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return atan2_kernel(inputs[0], inputs[1], stream);
}
Tensor fmod_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return fmod_kernel(inputs[0], inputs[1], stream);
}
Tensor remainder_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return remainder_kernel(inputs[0], inputs[1], stream);
}

// Lerp dispatch wrapper
Tensor lerp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return lerp_kernel(inputs[0], inputs[1], inputs[2], stream);
}

// Logical dispatch wrappers
Tensor logical_and_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return logical_and_kernel(inputs[0], inputs[1], stream);
}
Tensor logical_or_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return logical_or_kernel(inputs[0], inputs[1], stream);
}
Tensor logical_not_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return logical_not_kernel(inputs[0], stream);
}
Tensor logical_xor_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return logical_xor_kernel(inputs[0], inputs[1], stream);
}

// Element-wise min/max dispatch wrappers
Tensor minimum_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return minimum_kernel(inputs[0], inputs[1], stream);
}
Tensor maximum_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return maximum_kernel(inputs[0], inputs[1], stream);
}

// =========================================================================
// Complex Number Operations
// =========================================================================

// --- Conj ---
__global__ void conj_kernel_c64(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[2 * idx]     =  input[2 * idx];      // real
        output[2 * idx + 1] = -input[2 * idx + 1];  // -imag
    }
}
__global__ void conj_kernel_c128(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[2 * idx]     =  input[2 * idx];
        output[2 * idx + 1] = -input[2 * idx + 1];
    }
}

auto conj_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Complex64, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        conj_kernel_c64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Complex128, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        conj_kernel_c128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double*>(input.data_ptr()),
            reinterpret_cast<double*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    // For real dtypes, conjugate is identity
    // Clone via a simple copy
    Tensor result(shape, input.dtype(), input.device());
    cudaMemcpyAsync(result.data_ptr(), input.data_ptr(),
                    n * dtype_size(input.dtype()), cudaMemcpyDeviceToDevice, stream);
    return result;
}

// --- Real ---
__global__ void real_kernel_c64(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[2 * idx];
    }
}
__global__ void real_kernel_c128(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[2 * idx];
    }
}

auto real_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        real_kernel_c64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        real_kernel_c128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double*>(input.data_ptr()),
            result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    // For real dtypes, real() is identity
    Tensor result(shape, input.dtype(), input.device());
    cudaMemcpyAsync(result.data_ptr(), input.data_ptr(),
                    n * dtype_size(input.dtype()), cudaMemcpyDeviceToDevice, stream);
    return result;
}

// --- Imag ---
__global__ void imag_kernel_c64(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[2 * idx + 1];
    }
}
__global__ void imag_kernel_c128(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[2 * idx + 1];
    }
}

auto imag_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        imag_kernel_c64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        imag_kernel_c128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double*>(input.data_ptr()),
            result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    // For real dtypes, imaginary part is zero
    Tensor result(shape, input.dtype(), input.device());
    cudaMemsetAsync(result.data_ptr(), 0, n * dtype_size(input.dtype()), stream);
    return result;
}

// --- Angle ---
__global__ void angle_kernel_c64(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = atan2f(input[2 * idx + 1], input[2 * idx]);
    }
}
__global__ void angle_kernel_c128(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = atan2(input[2 * idx + 1], input[2 * idx]);
    }
}
__global__ void angle_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = atan2f(0.0f, input[idx]);
    }
}
__global__ void angle_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = atan2(0.0, input[idx]);
    }
}

auto angle_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        angle_kernel_c64<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        angle_kernel_c128<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double*>(input.data_ptr()),
            result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (input.dtype() == DType::Float32) {
        Tensor result(shape, DType::Float32, input.device());
        angle_kernel_f32<<<grid, block, 0, stream>>>(
            input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (input.dtype() == DType::Float64) {
        Tensor result(shape, DType::Float64, input.device());
        angle_kernel_f64<<<grid, block, 0, stream>>>(
            input.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Upcast to Float32, compute angle, downcast
        DType orig = input.dtype();
        Tensor f32_input = input.to(DType::Float32);
        Tensor result(shape, DType::Float32, input.device());
        angle_kernel_f32<<<grid, block, 0, stream>>>(
            f32_input.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
        return result.to(orig);
    }
    throw std::runtime_error("angle: unsupported dtype");
}

// --- Polar ---
__global__ void polar_kernel_f32(const float* abs_in, const float* angle_in,
                                  float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float r = abs_in[idx];
        float theta = angle_in[idx];
        output[2 * idx]     = r * cosf(theta);
        output[2 * idx + 1] = r * sinf(theta);
    }
}
__global__ void polar_kernel_f64(const double* abs_in, const double* angle_in,
                                  double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double r = abs_in[idx];
        double theta = angle_in[idx];
        output[2 * idx]     = r * cos(theta);
        output[2 * idx + 1] = r * sin(theta);
    }
}

auto polar_kernel(const Tensor& abs_t, const Tensor& angle_t, cudaStream_t stream) -> Tensor {
    if (abs_t.dtype() != angle_t.dtype()) {
        throw std::runtime_error("polar: abs and angle must have the same dtype");
    }
    auto shape_a = abs_t.shape();
    auto shape_b = angle_t.shape();
    if (!std::equal(shape_a.begin(), shape_a.end(), shape_b.begin(), shape_b.end())) {
        throw std::runtime_error("polar: abs and angle must have the same shape");
    }

    int64_t n = abs_t.numel();
    std::vector<int64_t> shape(shape_a.begin(), shape_a.end());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (abs_t.dtype() == DType::Float32) {
        Tensor result(shape, DType::Complex64, abs_t.device());
        polar_kernel_f32<<<grid, block, 0, stream>>>(
            abs_t.data<float>(), angle_t.data<float>(),
            reinterpret_cast<float*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (abs_t.dtype() == DType::Float64) {
        Tensor result(shape, DType::Complex128, abs_t.device());
        polar_kernel_f64<<<grid, block, 0, stream>>>(
            abs_t.data<double>(), angle_t.data<double>(),
            reinterpret_cast<double*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (abs_t.dtype() == DType::Float16 || abs_t.dtype() == DType::BFloat16) {
        // Upcast to Float32, compute polar, return Complex64
        Tensor abs_f32 = abs_t.to(DType::Float32);
        Tensor angle_f32 = angle_t.to(DType::Float32);
        Tensor result(shape, DType::Complex64, abs_t.device());
        polar_kernel_f32<<<grid, block, 0, stream>>>(
            abs_f32.data<float>(), angle_f32.data<float>(),
            reinterpret_cast<float*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    throw std::runtime_error("polar: only Float32, Float64, Float16, and BFloat16 inputs are supported");
}

// Complex number dispatch wrappers
Tensor conj_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return conj_kernel(inputs[0], stream);
}
Tensor real_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return real_kernel(inputs[0], stream);
}
Tensor imag_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return imag_kernel(inputs[0], stream);
}
Tensor angle_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return angle_kernel(inputs[0], stream);
}
Tensor polar_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return polar_kernel(inputs[0], inputs[1], stream);
}

// ============================================================================
// ComplexTensor Kernel — interleave real + imag into Complex64/Complex128
// ============================================================================

__global__ void complex_tensor_kernel_f32(const float* real, const float* imag,
                                           float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[2 * idx]     = real[idx];
        output[2 * idx + 1] = imag[idx];
    }
}
__global__ void complex_tensor_kernel_f64(const double* real, const double* imag,
                                           double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[2 * idx]     = real[idx];
        output[2 * idx + 1] = imag[idx];
    }
}

auto complex_tensor_kernel(const Tensor& real_t, const Tensor& imag_t, cudaStream_t stream) -> Tensor {
    if (real_t.dtype() != imag_t.dtype()) {
        throw std::runtime_error("complex: real and imag must have the same dtype");
    }
    auto shape_r = real_t.shape();
    auto shape_i = imag_t.shape();
    if (!std::equal(shape_r.begin(), shape_r.end(), shape_i.begin(), shape_i.end())) {
        throw std::runtime_error("complex: real and imag must have the same shape");
    }

    int64_t n = real_t.numel();
    std::vector<int64_t> shape(shape_r.begin(), shape_r.end());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (real_t.dtype() == DType::Float32) {
        Tensor result(shape, DType::Complex64, real_t.device());
        complex_tensor_kernel_f32<<<grid, block, 0, stream>>>(
            real_t.data<float>(), imag_t.data<float>(),
            reinterpret_cast<float*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (real_t.dtype() == DType::Float64) {
        Tensor result(shape, DType::Complex128, real_t.device());
        complex_tensor_kernel_f64<<<grid, block, 0, stream>>>(
            real_t.data<double>(), imag_t.data<double>(),
            reinterpret_cast<double*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (real_t.dtype() == DType::Float16 || real_t.dtype() == DType::BFloat16) {
        // Upcast to Float32, interleave, return Complex64
        Tensor real_f32 = real_t.to(DType::Float32);
        Tensor imag_f32 = imag_t.to(DType::Float32);
        Tensor result(shape, DType::Complex64, real_t.device());
        complex_tensor_kernel_f32<<<grid, block, 0, stream>>>(
            real_f32.data<float>(), imag_f32.data<float>(),
            reinterpret_cast<float*>(result.data_ptr()), n);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    throw std::runtime_error("complex: only Float32, Float64, Float16, and BFloat16 inputs are supported");
}

Tensor complex_tensor_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return complex_tensor_kernel(inputs[0], inputs[1], stream);
}

// ============================================================================
// Cross Product Kernel
// ============================================================================

template<typename T>
__global__ void cross_kernel_device(const T* a, const T* b, T* c,
                                     int64_t num_pairs, int64_t dim_stride) {
    TENZOR_CUDA_KERNEL_LOOP(idx, num_pairs) {
        int64_t o = idx / dim_stride;
        int64_t i = idx % dim_stride;
        int64_t base = o * 3 * dim_stride + i;
        T a0 = a[base], a1 = a[base + dim_stride], a2 = a[base + 2*dim_stride];
        T b0 = b[base], b1 = b[base + dim_stride], b2 = b[base + 2*dim_stride];
        c[base]                  = a1*b2 - a2*b1;
        c[base + dim_stride]     = a2*b0 - a0*b2;
        c[base + 2*dim_stride]   = a0*b1 - a1*b0;
    }
}

__global__ void cross_kernel_f16(const __half* a, const __half* b, __half* c,
                                  int64_t num_pairs, int64_t dim_stride) {
    TENZOR_CUDA_KERNEL_LOOP(idx, num_pairs) {
        int64_t o = idx / dim_stride;
        int64_t i = idx % dim_stride;
        int64_t base = o * 3 * dim_stride + i;
        float a0 = __half2float(a[base]), a1 = __half2float(a[base + dim_stride]), a2 = __half2float(a[base + 2*dim_stride]);
        float b0 = __half2float(b[base]), b1 = __half2float(b[base + dim_stride]), b2 = __half2float(b[base + 2*dim_stride]);
        c[base]                  = __float2half(a1*b2 - a2*b1);
        c[base + dim_stride]     = __float2half(a2*b0 - a0*b2);
        c[base + 2*dim_stride]   = __float2half(a0*b1 - a1*b0);
    }
}

__global__ void cross_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c,
                                   int64_t num_pairs, int64_t dim_stride) {
    TENZOR_CUDA_KERNEL_LOOP(idx, num_pairs) {
        int64_t o = idx / dim_stride;
        int64_t i = idx % dim_stride;
        int64_t base = o * 3 * dim_stride + i;
        float a0 = __bfloat162float(a[base]), a1 = __bfloat162float(a[base + dim_stride]), a2 = __bfloat162float(a[base + 2*dim_stride]);
        float b0 = __bfloat162float(b[base]), b1 = __bfloat162float(b[base + dim_stride]), b2 = __bfloat162float(b[base + 2*dim_stride]);
        c[base]                  = __float2bfloat16(a1*b2 - a2*b1);
        c[base + dim_stride]     = __float2bfloat16(a2*b0 - a0*b2);
        c[base + 2*dim_stride]   = __float2bfloat16(a0*b1 - a1*b0);
    }
}

auto cross_kernel(const Tensor& a, const Tensor& b, int64_t dim, cudaStream_t stream) -> Tensor {
    auto shape = a.shape();
    int64_t ndim = shape.size();
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor result(out_shape, a.dtype(), a.device());

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape[d];
    int64_t num_pairs = outer * inner;

    if (num_pairs == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(num_pairs, grid, block);

    if (a.dtype() == DType::Float32) {
        cross_kernel_device<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(), num_pairs, inner);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        cross_kernel_device<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<double>(), num_pairs, inner);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        cross_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), num_pairs, inner);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        cross_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), num_pairs, inner);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("cross: unsupported dtype");
    }
    return result;
}

Tensor cross_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    return cross_kernel(inputs[0], inputs[1], dim, stream);
}

// =========================================================================
// Special Math Functions — native CUDA device implementations
// (replaces the previous CPU-roundtrip fallbacks in cuda_kernel_registry.cpp)
// =========================================================================
//
// CUDA's libdevice provides native implementations of most special functions:
//   tgammaf, lgammaf       — gamma, lgamma
//   j0f, j1f, y0f, y1f     — cylindrical Bessel J/Y
//   cyl_bessel_i0f, cyl_bessel_i1f — modified Bessel I (CUDA Math API extensions)
//   erfinvf                — inverse error function
//
// For functions without native libdevice equivalents (digamma, polygamma,
// betainc, sinc, zeta) we implement them inline using polynomial expansions
// matching the CPU backend's algorithms.

// --- Digamma (psi function) — Cephes-style asymptotic expansion ---
__device__ inline float digamma_dev_f32(float x) {
    float result = 0.0f;
    if (x < 0.5f) {
        // Reflection: ψ(x) = ψ(1-x) - π·cot(πx)
        float y = 1.0f - x;
        float r = 0.0f;
        while (y < 7.0f) { r -= 1.0f / y; y += 1.0f; }
        float y2 = 1.0f / (y * y);
        r += logf(y) - 0.5f / y
            - y2 * (0.0833333333f - y2 * (0.00833333333f - y2 * (0.00396825397f
            - y2 * (0.00416666667f - y2 * 0.00757575758f))));
        return r - 3.14159265358979f / tanf(3.14159265358979f * x);
    }
    while (x < 7.0f) { result -= 1.0f / x; x += 1.0f; }
    float x2 = 1.0f / (x * x);
    result += logf(x) - 0.5f / x
            - x2 * (0.0833333333f - x2 * (0.00833333333f - x2 * (0.00396825397f
            - x2 * (0.00416666667f - x2 * 0.00757575758f))));
    return result;
}
__device__ inline double digamma_dev_f64(double x) {
    double result = 0.0;
    if (x < 0.5) {
        double y = 1.0 - x;
        double r = 0.0;
        while (y < 7.0) { r -= 1.0 / y; y += 1.0; }
        double y2 = 1.0 / (y * y);
        r += log(y) - 0.5 / y
            - y2 * (1.0/12.0 - y2 * (1.0/120.0 - y2 * (1.0/252.0
            - y2 * (1.0/240.0 - y2 * (1.0/132.0)))));
        return r - 3.14159265358979323846 / tan(3.14159265358979323846 * x);
    }
    while (x < 7.0) { result -= 1.0 / x; x += 1.0; }
    double x2 = 1.0 / (x * x);
    result += log(x) - 0.5 / x
            - x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0
            - x2 * (1.0/240.0 - x2 * (1.0/132.0)))));
    return result;
}

// --- Sinc (normalized: sin(πx)/(πx), sinc(0)=1) ---
__device__ inline float sinc_dev_f32(float x) {
    if (x == 0.0f) return 1.0f;
    float px = 3.14159265358979f * x;
    // Use sinf (1-ULP accurate) rather than __sinf (fast intrinsic).
    // sinc is numerically delicate near x = ±1, ±2, ... where π·x is a
    // multiple of π and sin saturates to zero — __sinf's error grows
    // sharply at those points, producing ~1e-5 cross-backend diffs vs
    // CPU std::sin. The accuracy/perf trade-off favours accuracy for
    // special-math ops, which are rarely a hot path.
    return sinf(px) / px;
}
__device__ inline double sinc_dev_f64(double x) {
    if (x == 0.0) return 1.0;
    double px = 3.14159265358979323846 * x;
    return sin(px) / px;
}

// --- Hurwitz zeta ζ(s,q) — Euler-Maclaurin partial sum ---
__device__ inline float zeta_dev_f32(float s, float a) {
    float result = 0.0f;
    #pragma unroll
    for (int n = 0; n < 12; ++n) {
        result += powf(a + static_cast<float>(n), -s);
    }
    float aN = a + 12.0f;
    if (s != 1.0f) result += powf(aN, 1.0f - s) / (s - 1.0f);
    result += 0.5f * powf(aN, -s);
    return result;
}
__device__ inline double zeta_dev_f64(double s, double a) {
    double result = 0.0;
    #pragma unroll
    for (int n = 0; n < 12; ++n) {
        result += pow(a + static_cast<double>(n), -s);
    }
    double aN = a + 12.0;
    if (s != 1.0) result += pow(aN, 1.0 - s) / (s - 1.0);
    result += 0.5 * pow(aN, -s);
    return result;
}

// --- Polygamma ψ^(n)(x) for n ≥ 1 ---
// Direct series Σ 1/(x+k)^(n+1) converges only as O(1/k^n), so a bounded
// truncation leaves a large tail at Float64 precision. Match the CPU path:
// shift x via the closed-form recurrence until x ≥ 7, then evaluate the
// asymptotic Bernoulli expansion (residual O(1/x^15) is below double epsilon).
__device__ inline double polygamma_dev_f64(int n, double x) {
    if (n == 0) return digamma_dev_f64(x);
    double fact_n = 1.0;
    for (int k = 1; k <= n; ++k) fact_n *= static_cast<double>(k);
    const double sign = ((n + 1) % 2 == 0) ? 1.0 : -1.0;

    double recurrence_sum = 0.0;
    while (x < 7.0) {
        recurrence_sum += 1.0 / pow(x, static_cast<double>(n + 1));
        x += 1.0;
    }

    double fact_nm1 = (n >= 1) ? fact_n / static_cast<double>(n) : 1.0;
    double asym = fact_nm1 / pow(x, static_cast<double>(n))
                + fact_n / (2.0 * pow(x, static_cast<double>(n + 1)));

    // First six Bernoulli coefficients B_2 .. B_12.
    const double B2[6] = {
         1.0 / 6.0,        // B_2
        -1.0 / 30.0,       // B_4
         1.0 / 42.0,       // B_6
        -1.0 / 30.0,       // B_8
         5.0 / 66.0,       // B_10
      -691.0 / 2730.0      // B_12
    };
    double fact_2k = 2.0;   // (2k)! starting at k=1: 2! = 2
    for (int k = 1; k <= 6; ++k) {
        double two_k = 2.0 * k;
        // (2k + n - 1)!
        double num = 1.0;
        int upper = static_cast<int>(two_k + n - 1);
        for (int i = 2; i <= upper; ++i) num *= i;
        if (k > 1) fact_2k *= (two_k - 1.0) * two_k;
        asym += B2[k - 1] * num / fact_2k / pow(x, two_k + n);
    }

    return sign * fact_n * (recurrence_sum + asym / fact_n);
}
__device__ inline float polygamma_dev_f32(int n, float x) {
    return static_cast<float>(polygamma_dev_f64(n, static_cast<double>(x)));
}

// --- Regularized incomplete beta function I_x(a,b) — Lentz continued fraction ---
__device__ inline double betainc_dev_f64(double a, double b, double x) {
    if (x < 0.0 || x > 1.0) return nan("");
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    // Symmetry: I_x(a,b) = 1 - I_{1-x}(b,a) when x past inflection
    bool flipped = false;
    if (x > (a + 1.0) / (a + b + 2.0)) {
        double tmp_a = a; a = b; b = tmp_a;
        x = 1.0 - x;
        flipped = true;
    }

    double lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    double front = exp(log(x) * a + log(1.0 - x) * b - lbeta) / a;

    double f = 1.0, c = 1.0, d = 1.0 - (a + b) * x / (a + 1.0);
    if (fabs(d) < 1e-30) d = 1e-30;
    d = 1.0 / d;
    f = d;

    for (int m = 1; m <= 200; ++m) {
        double num = static_cast<double>(m) * (b - m) * x
                   / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + num * d; if (fabs(d) < 1e-30) d = 1e-30; d = 1.0 / d;
        c = 1.0 + num / c; if (fabs(c) < 1e-30) c = 1e-30;
        f *= d * c;

        num = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + num * d; if (fabs(d) < 1e-30) d = 1e-30; d = 1.0 / d;
        c = 1.0 + num / c; if (fabs(c) < 1e-30) c = 1e-30;
        double delta = d * c;
        f *= delta;
        if (fabs(delta - 1.0) < 1e-12) break;
    }
    double val = front * f;
    return flipped ? (1.0 - val) : val;
}

// =========================================================================
// Generic unary special-math kernel template
// =========================================================================
template<typename FnFloat, typename FnDouble>
__global__ void special_unary_kernel_f32(const float* in, float* out, int64_t n, FnFloat fn) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = fn(in[idx]); }
}
template<typename FnDouble>
__global__ void special_unary_kernel_f64(const double* in, double* out, int64_t n, FnDouble fn) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = fn(in[idx]); }
}

// Macro to declare kernel + dispatch for a unary special function.
// All dtypes accumulate in float (or double for f64); FP16/BF16 storage variants
// load, cast to float, compute via the f32 expression, then cast back.
#define DEFINE_CUDA_SPECIAL_UNARY(NAME, FN_F32_EXPR, FN_F64_EXPR)                            \
    __global__ void NAME##_kernel_f32(const float* in, float* out, int64_t n) {              \
        TENZOR_CUDA_KERNEL_LOOP(idx, n) { float x = in[idx]; out[idx] = (FN_F32_EXPR); }    \
    }                                                                                         \
    __global__ void NAME##_kernel_f64(const double* in, double* out, int64_t n) {            \
        TENZOR_CUDA_KERNEL_LOOP(idx, n) { double x = in[idx]; out[idx] = (FN_F64_EXPR); }   \
    }                                                                                         \
    __global__ void NAME##_kernel_f16(const __half* in, __half* out, int64_t n) {            \
        TENZOR_CUDA_KERNEL_LOOP(idx, n) {                                                     \
            float x = __half2float(in[idx]);                                                  \
            out[idx] = __float2half(FN_F32_EXPR);                                             \
        }                                                                                     \
    }                                                                                         \
    __global__ void NAME##_kernel_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int64_t n) { \
        TENZOR_CUDA_KERNEL_LOOP(idx, n) {                                                     \
            float x = __bfloat162float(in[idx]);                                              \
            out[idx] = __float2bfloat16(FN_F32_EXPR);                                         \
        }                                                                                     \
    }                                                                                         \
    auto NAME##_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {                 \
        int64_t n = input.numel();                                                            \
        std::vector<int64_t> shape(input.shape().begin(), input.shape().end());              \
        Tensor result(shape, input.dtype(), input.device());                                 \
        if (n == 0) return result;                                                            \
        dim3 grid, block; compute_launch_config_1d(n, grid, block);                          \
        if (input.dtype() == DType::Float32) {                                                \
            NAME##_kernel_f32<<<grid, block, 0, stream>>>(                                    \
                input.data<float>(), result.data<float>(), n);                               \
            CUDA_CHECK(cudaGetLastError());                                                   \
        } else if (input.dtype() == DType::Float64) {                                         \
            NAME##_kernel_f64<<<grid, block, 0, stream>>>(                                    \
                input.data<double>(), result.data<double>(), n);                             \
            CUDA_CHECK(cudaGetLastError());                                                   \
        } else if (input.dtype() == DType::Float16) {                                         \
            NAME##_kernel_f16<<<grid, block, 0, stream>>>(                                    \
                reinterpret_cast<const __half*>(input.data<Float16>()),                       \
                reinterpret_cast<__half*>(result.data<Float16>()), n);                        \
            CUDA_CHECK(cudaGetLastError());                                                   \
        } else if (input.dtype() == DType::BFloat16) {                                        \
            NAME##_kernel_bf16<<<grid, block, 0, stream>>>(                                   \
                reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),               \
                reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);                \
            CUDA_CHECK(cudaGetLastError());                                                   \
        } else {                                                                              \
            throw std::runtime_error(#NAME " only supports Float32, Float64, Float16, BFloat16"); \
        }                                                                                     \
        return result;                                                                        \
    }                                                                                         \
    Tensor NAME##_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {      \
        auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);                        \
        return NAME##_kernel(inputs[0], stream);                                              \
    }

DEFINE_CUDA_SPECIAL_UNARY(gamma,     tgammaf(x),               tgamma(x))
DEFINE_CUDA_SPECIAL_UNARY(lgamma,    lgammaf(x),               lgamma(x))
DEFINE_CUDA_SPECIAL_UNARY(digamma,   digamma_dev_f32(x),       digamma_dev_f64(x))
DEFINE_CUDA_SPECIAL_UNARY(bessel_j0, j0f(x),                   j0(x))
DEFINE_CUDA_SPECIAL_UNARY(bessel_j1, j1f(x),                   j1(x))
DEFINE_CUDA_SPECIAL_UNARY(bessel_y0, y0f(x),                   y0(x))
DEFINE_CUDA_SPECIAL_UNARY(bessel_y1, y1f(x),                   y1(x))
DEFINE_CUDA_SPECIAL_UNARY(bessel_i0, cyl_bessel_i0f(x),        cyl_bessel_i0(x))
DEFINE_CUDA_SPECIAL_UNARY(bessel_i1, cyl_bessel_i1f(x),        cyl_bessel_i1(x))
DEFINE_CUDA_SPECIAL_UNARY(erfinv,    erfinvf(x),               erfinv(x))
DEFINE_CUDA_SPECIAL_UNARY(sinc,      sinc_dev_f32(x),          sinc_dev_f64(x))

// --- Deg2Rad / Rad2Deg ---
DEFINE_CUDA_SPECIAL_UNARY(deg2rad,   x * (3.14159265358979f / 180.0f),   x * (3.14159265358979323846 / 180.0))
DEFINE_CUDA_SPECIAL_UNARY(rad2deg,   x * (180.0f / 3.14159265358979f),   x * (180.0 / 3.14159265358979323846))

// --- Logit: log(clamp(x) / (1 - clamp(x))) ---
__device__ inline float logit_dev_f32(float x) {
    float cx = fmaxf(fminf(x, 1.0f - 1e-6f), 1e-6f);
    return logf(cx / (1.0f - cx));
}
__device__ inline double logit_dev_f64(double x) {
    double cx = fmax(fmin(x, 1.0 - 1e-15), 1e-15);
    return log(cx / (1.0 - cx));
}
DEFINE_CUDA_SPECIAL_UNARY(logit,     logit_dev_f32(x),         logit_dev_f64(x))

// --- Ndtr: Normal CDF Φ(x) = 0.5 * erfc(-x * M_SQRT1_2) ---
DEFINE_CUDA_SPECIAL_UNARY(ndtr,
    0.5f * erfcf(-x * 0.7071067811865476f),
    0.5  * erfc(-x * 0.7071067811865476))

// --- LogNdtr: log(Φ(x)) with stable tail ---
__device__ inline float log_ndtr_dev_f32(float x) {
    if (x >= -5.0f) {
        return logf(0.5f * erfcf(-x * 0.7071067811865476f));
    }
    float x2 = x * x;
    float inv_x2 = 1.0f / x2;
    float series = 1.0f - inv_x2 * (1.0f - inv_x2 * (3.0f - inv_x2 * 15.0f));
    return -0.5f * x2 - logf(-x) - 0.9189385332046727f + logf(series);
}
__device__ inline double log_ndtr_dev_f64(double x) {
    if (x >= -5.0) {
        return log(0.5 * erfc(-x * 0.7071067811865476));
    }
    double x2 = x * x;
    double inv_x2 = 1.0 / x2;
    double series = 1.0 - inv_x2 * (1.0 - inv_x2 * (3.0 - inv_x2 * 15.0));
    return -0.5 * x2 - log(-x) - 0.9189385332046727 + log(series);
}
DEFINE_CUDA_SPECIAL_UNARY(log_ndtr,  log_ndtr_dev_f32(x),      log_ndtr_dev_f64(x))

#undef DEFINE_CUDA_SPECIAL_UNARY

// --- Multigammaln: multivariate log-gamma with dimension parameter d ---
__global__ void multigammaln_kernel_f32(const float* in, float* out, int64_t n, int d, float log_pi_coeff) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float val = log_pi_coeff;
        for (int j = 0; j < d; ++j)
            val += lgammaf(in[idx] - static_cast<float>(j) * 0.5f);
        out[idx] = val;
    }
}
__global__ void multigammaln_kernel_f64(const double* in, double* out, int64_t n, int d, double log_pi_coeff) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double val = log_pi_coeff;
        for (int j = 0; j < d; ++j)
            val += lgamma(in[idx] - static_cast<double>(j) * 0.5);
        out[idx] = val;
    }
}
__global__ void multigammaln_kernel_f16(const __half* in, __half* out, int64_t n, int d, float log_pi_coeff) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __half2float(in[idx]);
        float val = log_pi_coeff;
        for (int j = 0; j < d; ++j)
            val += lgammaf(x - static_cast<float>(j) * 0.5f);
        out[idx] = __float2half(val);
    }
}
__global__ void multigammaln_kernel_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int64_t n, int d, float log_pi_coeff) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __bfloat162float(in[idx]);
        float val = log_pi_coeff;
        for (int j = 0; j < d; ++j)
            val += lgammaf(x - static_cast<float>(j) * 0.5f);
        out[idx] = __float2bfloat16(val);
    }
}
auto multigammaln_kernel_impl(const Tensor& input, int d, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    double log_pi_coeff = static_cast<double>(d) * static_cast<double>(d - 1) / 4.0 * log(M_PI);
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        multigammaln_kernel_f32<<<grid, block, 0, stream>>>(
            input.data<float>(), result.data<float>(), n, d, static_cast<float>(log_pi_coeff));
    } else if (input.dtype() == DType::Float64) {
        multigammaln_kernel_f64<<<grid, block, 0, stream>>>(
            input.data<double>(), result.data<double>(), n, d, log_pi_coeff);
    } else if (input.dtype() == DType::Float16) {
        multigammaln_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n, d, static_cast<float>(log_pi_coeff));
    } else if (input.dtype() == DType::BFloat16) {
        multigammaln_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n, d, static_cast<float>(log_pi_coeff));
    } else {
        throw std::runtime_error("multigammaln only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor multigammaln_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    int d = static_cast<int>(attrs.get_int(AttrKey::Dim, 1));
    return multigammaln_kernel_impl(inputs[0], d, stream);
}

// --- Beta(a, b) = exp(lgamma(a) + lgamma(b) - lgamma(a + b)) ---
__device__ inline float beta_dev_f32(float a, float b) {
    return expf(lgammaf(a) + lgammaf(b) - lgammaf(a + b));
}
__device__ inline double beta_dev_f64(double a, double b) {
    return exp(lgamma(a) + lgamma(b) - lgamma(a + b));
}
__global__ void beta_kernel_f32(const float* a, const float* b, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = beta_dev_f32(a[idx], b[idx]); }
}
__global__ void beta_kernel_f64(const double* a, const double* b, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = beta_dev_f64(a[idx], b[idx]); }
}
__global__ void beta_kernel_f16(const __half* a, const __half* b, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = __float2half(beta_dev_f32(__half2float(a[idx]), __half2float(b[idx])));
    }
}
__global__ void beta_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = __float2bfloat16(beta_dev_f32(__bfloat162float(a[idx]), __bfloat162float(b[idx])));
    }
}
auto beta_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        beta_kernel_f32<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        beta_kernel_f64<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        beta_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        beta_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("beta only supports Float32, Float64, Float16, BFloat16");
    }
    return result;
}
Tensor beta_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return beta_kernel(inputs[0], inputs[1], stream);
}

// --- Hurwitz zeta ζ(s, q) ---
__global__ void zeta_kernel_f32(const float* s, const float* q, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = zeta_dev_f32(s[idx], q[idx]); }
}
__global__ void zeta_kernel_f64(const double* s, const double* q, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = zeta_dev_f64(s[idx], q[idx]); }
}
__global__ void zeta_kernel_f16(const __half* s, const __half* q, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = __float2half(zeta_dev_f32(__half2float(s[idx]), __half2float(q[idx])));
    }
}
__global__ void zeta_kernel_bf16(const __nv_bfloat16* s, const __nv_bfloat16* q, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = __float2bfloat16(zeta_dev_f32(__bfloat162float(s[idx]), __bfloat162float(q[idx])));
    }
}
auto zeta_kernel(const Tensor& s, const Tensor& q, cudaStream_t stream) -> Tensor {
    int64_t n = s.numel();
    std::vector<int64_t> shape(s.shape().begin(), s.shape().end());
    Tensor result(shape, s.dtype(), s.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (s.dtype() == DType::Float32) {
        zeta_kernel_f32<<<grid, block, 0, stream>>>(
            s.data<float>(), q.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (s.dtype() == DType::Float64) {
        zeta_kernel_f64<<<grid, block, 0, stream>>>(
            s.data<double>(), q.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (s.dtype() == DType::Float16) {
        zeta_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(s.data<Float16>()),
            reinterpret_cast<const __half*>(q.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (s.dtype() == DType::BFloat16) {
        zeta_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(s.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(q.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("zeta only supports Float32, Float64, Float16, BFloat16");
    }
    return result;
}
Tensor zeta_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return zeta_kernel(inputs[0], inputs[1], stream);
}

// --- Polygamma ψ^(n)(x) — n is a scalar order from attrs ---
__global__ void polygamma_kernel_f32(int n, const float* in, float* out, int64_t numel) {
    TENZOR_CUDA_KERNEL_LOOP(idx, numel) { out[idx] = polygamma_dev_f32(n, in[idx]); }
}
__global__ void polygamma_kernel_f64(int n, const double* in, double* out, int64_t numel) {
    TENZOR_CUDA_KERNEL_LOOP(idx, numel) { out[idx] = polygamma_dev_f64(n, in[idx]); }
}
__global__ void polygamma_kernel_f16(int n, const __half* in, __half* out, int64_t numel) {
    TENZOR_CUDA_KERNEL_LOOP(idx, numel) {
        out[idx] = __float2half(polygamma_dev_f32(n, __half2float(in[idx])));
    }
}
__global__ void polygamma_kernel_bf16(int n, const __nv_bfloat16* in, __nv_bfloat16* out, int64_t numel) {
    TENZOR_CUDA_KERNEL_LOOP(idx, numel) {
        out[idx] = __float2bfloat16(polygamma_dev_f32(n, __bfloat162float(in[idx])));
    }
}
auto polygamma_kernel(int64_t n, const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t numel = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (numel == 0) return result;
    dim3 grid, block; compute_launch_config_1d(numel, grid, block);
    if (input.dtype() == DType::Float32) {
        polygamma_kernel_f32<<<grid, block, 0, stream>>>(
            static_cast<int>(n), input.data<float>(), result.data<float>(), numel);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        polygamma_kernel_f64<<<grid, block, 0, stream>>>(
            static_cast<int>(n), input.data<double>(), result.data<double>(), numel);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        polygamma_kernel_f16<<<grid, block, 0, stream>>>(
            static_cast<int>(n),
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), numel);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        polygamma_kernel_bf16<<<grid, block, 0, stream>>>(
            static_cast<int>(n),
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), numel);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("polygamma only supports Float32, Float64, Float16, BFloat16");
    }
    return result;
}
Tensor polygamma_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    int64_t order = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
    return polygamma_kernel(order, inputs[0], stream);
}

// --- Regularized incomplete beta I_x(a, b) ---
__global__ void betainc_kernel_f32(const float* a, const float* b, const float* x, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = static_cast<float>(betainc_dev_f64(
            static_cast<double>(a[idx]), static_cast<double>(b[idx]), static_cast<double>(x[idx])));
    }
}
__global__ void betainc_kernel_f64(const double* a, const double* b, const double* x, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = betainc_dev_f64(a[idx], b[idx], x[idx]);
    }
}
__global__ void betainc_kernel_f16(const __half* a, const __half* b, const __half* x, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double r = betainc_dev_f64(
            static_cast<double>(__half2float(a[idx])),
            static_cast<double>(__half2float(b[idx])),
            static_cast<double>(__half2float(x[idx])));
        out[idx] = __float2half(static_cast<float>(r));
    }
}
__global__ void betainc_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double r = betainc_dev_f64(
            static_cast<double>(__bfloat162float(a[idx])),
            static_cast<double>(__bfloat162float(b[idx])),
            static_cast<double>(__bfloat162float(x[idx])));
        out[idx] = __float2bfloat16(static_cast<float>(r));
    }
}
auto betainc_kernel(const Tensor& a, const Tensor& b, const Tensor& x, cudaStream_t stream) -> Tensor {
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        betainc_kernel_f32<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), x.data<float>(), result.data<float>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float64) {
        betainc_kernel_f64<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), x.data<double>(), result.data<double>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::Float16) {
        betainc_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (a.dtype() == DType::BFloat16) {
        betainc_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(x.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("betainc only supports Float32, Float64, Float16, BFloat16");
    }
    return result;
}
Tensor betainc_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return betainc_kernel(inputs[0], inputs[1], inputs[2], stream);
}

// ============================================================================
// New element-wise ops: Frac, Heaviside, NanToNum
// ============================================================================

// frac is sign-preserving: frac(x) = x - trunc(x). The previous
// implementation used floor(x), which always returns a non-negative
// fractional part (frac(-2.75) → 0.25 instead of -0.75) and diverged from
// CPU + PyTorch semantics. Use truncf/trunc so negative inputs round toward
// zero, matching the CPU reference and ROCm/OneAPI kernels.
__global__ void frac_kernel_f32(const float* in, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] - truncf(in[idx]); }
}
__global__ void frac_kernel_f64(const double* in, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] - trunc(in[idx]); }
}
auto frac_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        frac_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        frac_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else {
        auto f32 = input.to(DType::Float32);
        return frac_kernel(f32, stream).to(input.dtype());
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor frac_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return frac_kernel(inputs[0], stream);
}

__global__ void heaviside_kernel_f32(const float* input, const float* values, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = input[idx];
        out[idx] = (x < 0.0f) ? 0.0f : (x == 0.0f ? values[idx] : 1.0f);
    }
}
__global__ void heaviside_kernel_f64(const double* input, const double* values, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double x = input[idx];
        out[idx] = (x < 0.0) ? 0.0 : (x == 0.0 ? values[idx] : 1.0);
    }
}
auto heaviside_kernel(const Tensor& input, const Tensor& values, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        heaviside_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), values.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        heaviside_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), values.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        auto f32_in = input.to(DType::Float32); auto f32_val = values.to(DType::Float32);
        return heaviside_kernel(f32_in, f32_val, stream).to(DType::Float16);
    } else if (input.dtype() == DType::BFloat16) {
        auto f32_in = input.to(DType::Float32); auto f32_val = values.to(DType::Float32);
        return heaviside_kernel(f32_in, f32_val, stream).to(DType::BFloat16);
    } else { throw std::runtime_error("heaviside: unsupported dtype"); }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor heaviside_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return heaviside_kernel(inputs[0], inputs[1], stream);
}

__global__ void nan_to_num_kernel_f32(const float* input, float* out, int64_t n,
                                       float nan_val, float posinf_val, float neginf_val) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = input[idx];
        if (isnan(x)) out[idx] = nan_val;
        else if (isinf(x) && x > 0) out[idx] = posinf_val;
        else if (isinf(x) && x < 0) out[idx] = neginf_val;
        else out[idx] = x;
    }
}
__global__ void nan_to_num_kernel_f64(const double* input, double* out, int64_t n,
                                       double nan_val, double posinf_val, double neginf_val) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double x = input[idx];
        if (isnan(x)) out[idx] = nan_val;
        else if (isinf(x) && x > 0) out[idx] = posinf_val;
        else if (isinf(x) && x < 0) out[idx] = neginf_val;
        else out[idx] = x;
    }
}
auto nan_to_num_kernel(const Tensor& input, double nan_v, double posinf_v, double neginf_v, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        float pf = (posinf_v >= static_cast<double>(std::numeric_limits<float>::max())) ? std::numeric_limits<float>::max() : static_cast<float>(posinf_v);
        float nf = (neginf_v <= static_cast<double>(std::numeric_limits<float>::lowest())) ? std::numeric_limits<float>::lowest() : static_cast<float>(neginf_v);
        nan_to_num_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n, static_cast<float>(nan_v), pf, nf);
    } else if (input.dtype() == DType::Float64) {
        nan_to_num_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n, nan_v, posinf_v, neginf_v);
    } else if (input.dtype() == DType::Float16) {
        auto f32 = input.to(DType::Float32);
        return nan_to_num_kernel(f32, nan_v, posinf_v, neginf_v, stream).to(DType::Float16);
    } else if (input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        return nan_to_num_kernel(f32, nan_v, posinf_v, neginf_v, stream).to(DType::BFloat16);
    } else { throw std::runtime_error("nan_to_num: unsupported dtype"); }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor nan_to_num_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    double nan_v = attrs.get_float(AttrKey::NanValue, 0.0);
    double posinf = attrs.get_float(AttrKey::PosInfValue, std::numeric_limits<double>::max());
    double neginf = attrs.get_float(AttrKey::NegInfValue, std::numeric_limits<double>::lowest());
    return nan_to_num_kernel(inputs[0], nan_v, posinf, neginf, stream);
}

// LogSigmoid: log(sigmoid(x)) = -softplus(-x)
__global__ void log_sigmoid_kernel_f32(const float* in, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = in[idx];
        out[idx] = (x >= 0.0f) ? -log1pf(expf(-x)) : x - log1pf(expf(x));
    }
}
__global__ void log_sigmoid_kernel_f64(const double* in, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double x = in[idx];
        out[idx] = (x >= 0.0) ? -log1p(exp(-x)) : x - log1p(exp(x));
    }
}
auto log_sigmoid_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        log_sigmoid_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        log_sigmoid_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else {
        auto f32 = input.to(DType::Float32);
        return log_sigmoid_kernel(f32, stream).to(input.dtype());
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor log_sigmoid_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return log_sigmoid_kernel(inputs[0], stream);
}

// Bitwise ops
__global__ void bitwise_and_i8(const int8_t* a, const int8_t* b, int8_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] & b[idx]; }
}
__global__ void bitwise_and_i16(const int16_t* a, const int16_t* b, int16_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] & b[idx]; }
}
__global__ void bitwise_and_i32(const int32_t* a, const int32_t* b, int32_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] & b[idx]; }
}
__global__ void bitwise_and_i64(const int64_t* a, const int64_t* b, int64_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] & b[idx]; }
}
auto bitwise_and_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Int8) {
        bitwise_and_i8<<<grid, block, 0, stream>>>(a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n);
    } else if (a.dtype() == DType::Int16) {
        bitwise_and_i16<<<grid, block, 0, stream>>>(a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(), n);
    } else if (a.dtype() == DType::Int32) {
        bitwise_and_i32<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        bitwise_and_i64<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
    } else { throw std::runtime_error("bitwise_and: unsupported dtype"); }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor bitwise_and_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return bitwise_and_kernel(inputs[0], inputs[1], stream);
}

__global__ void bitwise_or_i8(const int8_t* a, const int8_t* b, int8_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] | b[idx]; }
}
__global__ void bitwise_or_i16(const int16_t* a, const int16_t* b, int16_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] | b[idx]; }
}
__global__ void bitwise_or_i32(const int32_t* a, const int32_t* b, int32_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] | b[idx]; }
}
__global__ void bitwise_or_i64(const int64_t* a, const int64_t* b, int64_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] | b[idx]; }
}
auto bitwise_or_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Int8) { bitwise_or_i8<<<grid, block, 0, stream>>>(a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n); }
    else if (a.dtype() == DType::Int16) { bitwise_or_i16<<<grid, block, 0, stream>>>(a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(), n); }
    else if (a.dtype() == DType::Int32) { bitwise_or_i32<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n); }
    else if (a.dtype() == DType::Int64) { bitwise_or_i64<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_or: unsupported dtype"); }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor bitwise_or_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return bitwise_or_kernel(inputs[0], inputs[1], stream);
}

__global__ void bitwise_xor_i8(const int8_t* a, const int8_t* b, int8_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] ^ b[idx]; }
}
__global__ void bitwise_xor_i16(const int16_t* a, const int16_t* b, int16_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] ^ b[idx]; }
}
__global__ void bitwise_xor_i32(const int32_t* a, const int32_t* b, int32_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] ^ b[idx]; }
}
__global__ void bitwise_xor_i64(const int64_t* a, const int64_t* b, int64_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = a[idx] ^ b[idx]; }
}
auto bitwise_xor_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Int8) { bitwise_xor_i8<<<grid, block, 0, stream>>>(a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n); }
    else if (a.dtype() == DType::Int16) { bitwise_xor_i16<<<grid, block, 0, stream>>>(a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(), n); }
    else if (a.dtype() == DType::Int32) { bitwise_xor_i32<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n); }
    else if (a.dtype() == DType::Int64) { bitwise_xor_i64<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_xor: unsupported dtype"); }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor bitwise_xor_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return bitwise_xor_kernel(inputs[0], inputs[1], stream);
}

__global__ void bitwise_not_i8(const int8_t* in, int8_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = ~in[idx]; }
}
__global__ void bitwise_not_i16(const int16_t* in, int16_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = ~in[idx]; }
}
__global__ void bitwise_not_i32(const int32_t* in, int32_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = ~in[idx]; }
}
__global__ void bitwise_not_i64(const int64_t* in, int64_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = ~in[idx]; }
}
auto bitwise_not_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Int8) { bitwise_not_i8<<<grid, block, 0, stream>>>(input.data<int8_t>(), result.data<int8_t>(), n); }
    else if (input.dtype() == DType::Int16) { bitwise_not_i16<<<grid, block, 0, stream>>>(input.data<int16_t>(), result.data<int16_t>(), n); }
    else if (input.dtype() == DType::Int32) { bitwise_not_i32<<<grid, block, 0, stream>>>(input.data<int32_t>(), result.data<int32_t>(), n); }
    else if (input.dtype() == DType::Int64) { bitwise_not_i64<<<grid, block, 0, stream>>>(input.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_not: unsupported dtype"); }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor bitwise_not_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return bitwise_not_kernel(inputs[0], stream);
}

__global__ void bitwise_lshift_i8(const int8_t* in, const int8_t* sh, int8_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] << sh[idx]; }
}
__global__ void bitwise_lshift_i16(const int16_t* in, const int16_t* sh, int16_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] << sh[idx]; }
}
__global__ void bitwise_lshift_i32(const int32_t* in, const int32_t* sh, int32_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] << sh[idx]; }
}
__global__ void bitwise_lshift_i64(const int64_t* in, const int64_t* sh, int64_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] << sh[idx]; }
}
auto bitwise_left_shift_kernel(const Tensor& input, const Tensor& shift, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Int8) { bitwise_lshift_i8<<<grid, block, 0, stream>>>(input.data<int8_t>(), shift.data<int8_t>(), result.data<int8_t>(), n); }
    else if (input.dtype() == DType::Int16) { bitwise_lshift_i16<<<grid, block, 0, stream>>>(input.data<int16_t>(), shift.data<int16_t>(), result.data<int16_t>(), n); }
    else if (input.dtype() == DType::Int32) { bitwise_lshift_i32<<<grid, block, 0, stream>>>(input.data<int32_t>(), shift.data<int32_t>(), result.data<int32_t>(), n); }
    else if (input.dtype() == DType::Int64) { bitwise_lshift_i64<<<grid, block, 0, stream>>>(input.data<int64_t>(), shift.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_left_shift: unsupported dtype"); }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor bitwise_left_shift_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return bitwise_left_shift_kernel(inputs[0], inputs[1], stream);
}

__global__ void bitwise_rshift_i8(const int8_t* in, const int8_t* sh, int8_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] >> sh[idx]; }
}
__global__ void bitwise_rshift_i16(const int16_t* in, const int16_t* sh, int16_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] >> sh[idx]; }
}
__global__ void bitwise_rshift_i32(const int32_t* in, const int32_t* sh, int32_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] >> sh[idx]; }
}
__global__ void bitwise_rshift_i64(const int64_t* in, const int64_t* sh, int64_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] >> sh[idx]; }
}
auto bitwise_right_shift_kernel(const Tensor& input, const Tensor& shift, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Int8) { bitwise_rshift_i8<<<grid, block, 0, stream>>>(input.data<int8_t>(), shift.data<int8_t>(), result.data<int8_t>(), n); }
    else if (input.dtype() == DType::Int16) { bitwise_rshift_i16<<<grid, block, 0, stream>>>(input.data<int16_t>(), shift.data<int16_t>(), result.data<int16_t>(), n); }
    else if (input.dtype() == DType::Int32) { bitwise_rshift_i32<<<grid, block, 0, stream>>>(input.data<int32_t>(), shift.data<int32_t>(), result.data<int32_t>(), n); }
    else if (input.dtype() == DType::Int64) { bitwise_rshift_i64<<<grid, block, 0, stream>>>(input.data<int64_t>(), shift.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_right_shift: unsupported dtype"); }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor bitwise_right_shift_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return bitwise_right_shift_kernel(inputs[0], inputs[1], stream);
}

// ============================================================================
// Rsqrt kernel: 1/sqrt(x)
// ============================================================================
__global__ void rsqrt_kernel_f32(const float* in, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = rsqrtf(in[idx]); }
}
__global__ void rsqrt_kernel_f64(const double* in, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = rsqrt(in[idx]); }
}
__global__ void rsqrt_kernel_f16(const __half* in, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2half(rsqrtf(__half2float(in[idx]))); }
}
__global__ void rsqrt_kernel_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2bfloat16(rsqrtf(__bfloat162float(in[idx]))); }
}
auto rsqrt_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        rsqrt_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        rsqrt_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        rsqrt_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        rsqrt_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("rsqrt only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor rsqrt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return rsqrt_kernel(inputs[0], stream);
}

// ============================================================================
// Square kernel: x*x
// ============================================================================
__global__ void square_kernel_f32(const float* in, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] * in[idx]; }
}
__global__ void square_kernel_f64(const double* in, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = in[idx] * in[idx]; }
}
__global__ void square_kernel_f16(const __half* in, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __half2float(in[idx]);
        out[idx] = __float2half(x * x);
    }
}
__global__ void square_kernel_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __bfloat162float(in[idx]);
        out[idx] = __float2bfloat16(x * x);
    }
}
auto square_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        square_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        square_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        square_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        square_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("square only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor square_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return square_kernel(inputs[0], stream);
}

// ============================================================================
// Asinh kernel: inverse hyperbolic sine
// ============================================================================
__global__ void asinh_kernel_f32(const float* in, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = asinhf(in[idx]); }
}
__global__ void asinh_kernel_f64(const double* in, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = asinh(in[idx]); }
}
__global__ void asinh_kernel_f16(const __half* in, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2half(asinhf(__half2float(in[idx]))); }
}
__global__ void asinh_kernel_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2bfloat16(asinhf(__bfloat162float(in[idx]))); }
}
auto asinh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        asinh_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        asinh_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        asinh_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        asinh_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("asinh only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor asinh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return asinh_kernel(inputs[0], stream);
}

// ============================================================================
// Acosh kernel: inverse hyperbolic cosine
// ============================================================================
__global__ void acosh_kernel_f32(const float* in, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = acoshf(in[idx]); }
}
__global__ void acosh_kernel_f64(const double* in, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = acosh(in[idx]); }
}
__global__ void acosh_kernel_f16(const __half* in, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2half(acoshf(__half2float(in[idx]))); }
}
__global__ void acosh_kernel_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2bfloat16(acoshf(__bfloat162float(in[idx]))); }
}
auto acosh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        acosh_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        acosh_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        acosh_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        acosh_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("acosh only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor acosh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return acosh_kernel(inputs[0], stream);
}

// ============================================================================
// Atanh kernel: inverse hyperbolic tangent
// ============================================================================
__global__ void atanh_kernel_f32(const float* in, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = atanhf(in[idx]); }
}
__global__ void atanh_kernel_f64(const double* in, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = atanh(in[idx]); }
}
__global__ void atanh_kernel_f16(const __half* in, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2half(atanhf(__half2float(in[idx]))); }
}
__global__ void atanh_kernel_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2bfloat16(atanhf(__bfloat162float(in[idx]))); }
}
auto atanh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        atanh_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        atanh_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        atanh_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        atanh_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("atanh only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor atanh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return atanh_kernel(inputs[0], stream);
}

// ============================================================================
// Hypot kernel: sqrt(x*x + y*y) overflow-safe
// ============================================================================
__global__ void hypot_kernel_f32(const float* a, const float* b, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = hypotf(a[idx], b[idx]); }
}
__global__ void hypot_kernel_f64(const double* a, const double* b, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = hypot(a[idx], b[idx]); }
}
__global__ void hypot_kernel_f16(const __half* a, const __half* b, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2half(hypotf(__half2float(a[idx]), __half2float(b[idx]))); }
}
__global__ void hypot_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2bfloat16(hypotf(__bfloat162float(a[idx]), __bfloat162float(b[idx]))); }
}
auto hypot_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("hypot: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        hypot_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hypot_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hypot_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        hypot_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("hypot only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor hypot_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return hypot_kernel(inputs[0], inputs[1], stream);
}

// ============================================================================
// Copysign kernel: copysign(magnitude, sign)
// ============================================================================
__global__ void copysign_kernel_f32(const float* a, const float* b, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = copysignf(a[idx], b[idx]); }
}
__global__ void copysign_kernel_f64(const double* a, const double* b, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = copysign(a[idx], b[idx]); }
}
__global__ void copysign_kernel_f16(const __half* a, const __half* b, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2half(copysignf(__half2float(a[idx]), __half2float(b[idx]))); }
}
__global__ void copysign_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = __float2bfloat16(copysignf(__bfloat162float(a[idx]), __bfloat162float(b[idx]))); }
}
auto copysign_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("copysign: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        copysign_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        copysign_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        copysign_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        copysign_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("copysign only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor copysign_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return copysign_kernel(inputs[0], inputs[1], stream);
}

// ============================================================================
// Nextafter kernel: next representable float
// ============================================================================
__global__ void nextafter_kernel_f32(const float* a, const float* b, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = nextafterf(a[idx], b[idx]); }
}
__global__ void nextafter_kernel_f64(const double* a, const double* b, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = nextafter(a[idx], b[idx]); }
}
auto nextafter_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("nextafter: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        nextafter_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        nextafter_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        auto f32_a = a.to(DType::Float32);
        auto f32_b = b.to(DType::Float32);
        auto f32_result = nextafter_kernel(f32_a, f32_b, stream);
        return f32_result.to(DType::Float16);
    } else if (a.dtype() == DType::BFloat16) {
        auto f32_a = a.to(DType::Float32);
        auto f32_b = b.to(DType::Float32);
        auto f32_result = nextafter_kernel(f32_a, f32_b, stream);
        return f32_result.to(DType::BFloat16);
    } else {
        throw std::runtime_error("nextafter only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor nextafter_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return nextafter_kernel(inputs[0], inputs[1], stream);
}

// ============================================================================
// Gcd kernel: greatest common divisor (integer types)
// ============================================================================
__device__ inline int32_t gcd_dev_i32(int32_t a, int32_t b) {
    a = abs(a); b = abs(b);
    while (b != 0) { int32_t t = b; b = a % b; a = t; }
    return a;
}
__device__ inline int64_t gcd_dev_i64(int64_t a, int64_t b) {
    a = (a < 0) ? -a : a; b = (b < 0) ? -b : b;
    while (b != 0) { int64_t t = b; b = a % b; a = t; }
    return a;
}
__global__ void gcd_kernel_i32(const int32_t* a, const int32_t* b, int32_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = gcd_dev_i32(a[idx], b[idx]); }
}
__global__ void gcd_kernel_i64(const int64_t* a, const int64_t* b, int64_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = gcd_dev_i64(a[idx], b[idx]); }
}
auto gcd_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("gcd: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Int32) {
        gcd_kernel_i32<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        gcd_kernel_i64<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
    } else {
        throw std::runtime_error("gcd only supports Int32 and Int64");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor gcd_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return gcd_kernel(inputs[0], inputs[1], stream);
}

// ============================================================================
// Lcm kernel: least common multiple (integer types)
// ============================================================================
__global__ void lcm_kernel_i32(const int32_t* a, const int32_t* b, int32_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int32_t g = gcd_dev_i32(a[idx], b[idx]);
        out[idx] = (g == 0) ? 0 : abs(a[idx] / g * b[idx]);
    }
}
__global__ void lcm_kernel_i64(const int64_t* a, const int64_t* b, int64_t* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int64_t g = gcd_dev_i64(a[idx], b[idx]);
        int64_t aa = (a[idx] < 0) ? -a[idx] : a[idx];
        int64_t bb = (b[idx] < 0) ? -b[idx] : b[idx];
        out[idx] = (g == 0) ? 0 : (aa / g * bb);
    }
}
auto lcm_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("lcm: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Int32) {
        lcm_kernel_i32<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        lcm_kernel_i64<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
    } else {
        throw std::runtime_error("lcm only supports Int32 and Int64");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor lcm_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return lcm_kernel(inputs[0], inputs[1], stream);
}

// ============================================================================
// Addcmul kernel: input + alpha * tensor1 * tensor2 (ternary)
// ============================================================================
__global__ void addcmul_kernel_f32(const float* input, const float* t1, const float* t2, float* out, float alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = input[idx] + alpha * t1[idx] * t2[idx]; }
}
__global__ void addcmul_kernel_f64(const double* input, const double* t1, const double* t2, double* out, double alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = input[idx] + alpha * t1[idx] * t2[idx]; }
}
__global__ void addcmul_kernel_f16(const __half* input, const __half* t1, const __half* t2, __half* out, float alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float i = __half2float(input[idx]);
        float a = __half2float(t1[idx]);
        float b = __half2float(t2[idx]);
        out[idx] = __float2half(i + alpha * a * b);
    }
}
__global__ void addcmul_kernel_bf16(const __nv_bfloat16* input, const __nv_bfloat16* t1, const __nv_bfloat16* t2, __nv_bfloat16* out, float alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float i = __bfloat162float(input[idx]);
        float a = __bfloat162float(t1[idx]);
        float b = __bfloat162float(t2[idx]);
        out[idx] = __float2bfloat16(i + alpha * a * b);
    }
}
auto addcmul_kernel(const Tensor& input, const Tensor& t1, const Tensor& t2, double alpha, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        addcmul_kernel_f32<<<grid, block, 0, stream>>>(
            input.data<float>(), t1.data<float>(), t2.data<float>(),
            result.data<float>(), static_cast<float>(alpha), n);
    } else if (input.dtype() == DType::Float64) {
        addcmul_kernel_f64<<<grid, block, 0, stream>>>(
            input.data<double>(), t1.data<double>(), t2.data<double>(),
            result.data<double>(), alpha, n);
    } else if (input.dtype() == DType::Float16) {
        addcmul_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<const __half*>(t1.data<Float16>()),
            reinterpret_cast<const __half*>(t2.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            static_cast<float>(alpha), n);
    } else if (input.dtype() == DType::BFloat16) {
        addcmul_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(t1.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(t2.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            static_cast<float>(alpha), n);
    } else {
        throw std::runtime_error("addcmul only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ============================================================================
// Addcdiv kernel: input + alpha * tensor1 / tensor2 (ternary)
// ============================================================================
__global__ void addcdiv_kernel_f32(const float* input, const float* t1, const float* t2, float* out, float alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = input[idx] + alpha * t1[idx] / t2[idx]; }
}
__global__ void addcdiv_kernel_f64(const double* input, const double* t1, const double* t2, double* out, double alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = input[idx] + alpha * t1[idx] / t2[idx]; }
}
__global__ void addcdiv_kernel_f16(const __half* input, const __half* t1, const __half* t2, __half* out, float alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float i = __half2float(input[idx]);
        float a = __half2float(t1[idx]);
        float b = __half2float(t2[idx]);
        out[idx] = __float2half(i + alpha * a / b);
    }
}
__global__ void addcdiv_kernel_bf16(const __nv_bfloat16* input, const __nv_bfloat16* t1, const __nv_bfloat16* t2, __nv_bfloat16* out, float alpha, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float i = __bfloat162float(input[idx]);
        float a = __bfloat162float(t1[idx]);
        float b = __bfloat162float(t2[idx]);
        out[idx] = __float2bfloat16(i + alpha * a / b);
    }
}
auto addcdiv_kernel(const Tensor& input, const Tensor& t1, const Tensor& t2, double alpha, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        addcdiv_kernel_f32<<<grid, block, 0, stream>>>(
            input.data<float>(), t1.data<float>(), t2.data<float>(),
            result.data<float>(), static_cast<float>(alpha), n);
    } else if (input.dtype() == DType::Float64) {
        addcdiv_kernel_f64<<<grid, block, 0, stream>>>(
            input.data<double>(), t1.data<double>(), t2.data<double>(),
            result.data<double>(), alpha, n);
    } else if (input.dtype() == DType::Float16) {
        addcdiv_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<const __half*>(t1.data<Float16>()),
            reinterpret_cast<const __half*>(t2.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            static_cast<float>(alpha), n);
    } else if (input.dtype() == DType::BFloat16) {
        addcdiv_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(t1.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(t2.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            static_cast<float>(alpha), n);
    } else {
        throw std::runtime_error("addcdiv only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ============================================================================
// Igamma / Igammac: regularized incomplete gamma functions
//
// Uses the series expansion for the lower regularized incomplete gamma:
//   P(a, x) = (e^{-x} * x^a / Gamma(a)) * sum_{k=0..inf} x^k / (a*(a+1)*...*(a+k))
// Igammac = 1 - Igamma (upper regularized)
// ============================================================================
__device__ inline float igamma_series_f32(float a, float x) {
    if (x <= 0.0f) return 0.0f;
    if (a <= 0.0f) return 1.0f;

    float term = 1.0f / a;
    float sum = term;
    for (int k = 1; k < 200; ++k) {
        term *= x / (a + k);
        sum += term;
        if (fabsf(term) < 1e-7f * fabsf(sum)) break;
    }
    // P(a, x) = e^{-x} * x^a * sum / Gamma(a)
    // = exp(-x + a*log(x) - lgamma(a)) * sum
    float log_prefix = -x + a * logf(x) - lgammaf(a);
    return expf(log_prefix) * sum;
}

__device__ inline double igamma_series_f64(double a, double x) {
    if (x <= 0.0) return 0.0;
    if (a <= 0.0) return 1.0;

    double term = 1.0 / a;
    double sum = term;
    for (int k = 1; k < 500; ++k) {
        term *= x / (a + k);
        sum += term;
        if (fabs(term) < 1e-15 * fabs(sum)) break;
    }
    double log_prefix = -x + a * log(x) - lgamma(a);
    return exp(log_prefix) * sum;
}

// --- Igamma ---
__global__ void igamma_kernel_f32(const float* a, const float* x, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = igamma_series_f32(a[idx], x[idx]); }
}
__global__ void igamma_kernel_f64(const double* a, const double* x, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = igamma_series_f64(a[idx], x[idx]); }
}
__global__ void igamma_kernel_f16(const __half* a, const __half* x, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = __float2half(igamma_series_f32(__half2float(a[idx]), __half2float(x[idx])));
    }
}
__global__ void igamma_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = __float2bfloat16(igamma_series_f32(__bfloat162float(a[idx]), __bfloat162float(x[idx])));
    }
}
auto igamma_kernel(const Tensor& a, const Tensor& x, cudaStream_t stream) -> Tensor {
    if (a.dtype() != x.dtype()) throw std::runtime_error("igamma: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        igamma_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), x.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        igamma_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), x.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        igamma_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        igamma_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(x.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("igamma only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor igamma_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return igamma_kernel(inputs[0], inputs[1], stream);
}

// --- Igammac (upper regularized: 1 - igamma) ---
__global__ void igammac_kernel_f32(const float* a, const float* x, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = 1.0f - igamma_series_f32(a[idx], x[idx]); }
}
__global__ void igammac_kernel_f64(const double* a, const double* x, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { out[idx] = 1.0 - igamma_series_f64(a[idx], x[idx]); }
}
__global__ void igammac_kernel_f16(const __half* a, const __half* x, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = __float2half(1.0f - igamma_series_f32(__half2float(a[idx]), __half2float(x[idx])));
    }
}
__global__ void igammac_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = __float2bfloat16(1.0f - igamma_series_f32(__bfloat162float(a[idx]), __bfloat162float(x[idx])));
    }
}
auto igammac_kernel(const Tensor& a, const Tensor& x, cudaStream_t stream) -> Tensor {
    if (a.dtype() != x.dtype()) throw std::runtime_error("igammac: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        igammac_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), x.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        igammac_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), x.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        igammac_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        igammac_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(x.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("igammac only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
Tensor igammac_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return igammac_kernel(inputs[0], inputs[1], stream);
}

// ============================================================================
// Bool Predicate Kernels: signbit, isposinf, isneginf
// ============================================================================

// --- signbit ---
__global__ void signbit_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(signbit(input[idx]) ? 1 : 0); }
}
__global__ void signbit_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(signbit(input[idx]) ? 1 : 0); }
}
__global__ void signbit_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(signbit(__half2float(input[idx])) ? 1 : 0); }
}
__global__ void signbit_kernel_bf16(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>(signbit(__bfloat162float(input[idx])) ? 1 : 0); }
}

auto signbit_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        signbit_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        signbit_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        signbit_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        signbit_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("signbit operation only supports floating point dtypes");
    }
    return result;
}

Tensor signbit_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return signbit_kernel(inputs[0], stream);
}

// --- isposinf ---
__global__ void isposinf_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>((isinf(input[idx]) && input[idx] > 0) ? 1 : 0); }
}
__global__ void isposinf_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>((isinf(input[idx]) && input[idx] > 0) ? 1 : 0); }
}
__global__ void isposinf_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = __half2float(input[idx]);
        output[idx] = static_cast<uint8_t>((isinf(v) && v > 0) ? 1 : 0);
    }
}
__global__ void isposinf_kernel_bf16(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = __bfloat162float(input[idx]);
        output[idx] = static_cast<uint8_t>((isinf(v) && v > 0) ? 1 : 0);
    }
}

auto isposinf_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        isposinf_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        isposinf_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        isposinf_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        isposinf_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("isposinf operation only supports floating point dtypes");
    }
    return result;
}

Tensor isposinf_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return isposinf_kernel(inputs[0], stream);
}

// --- isneginf ---
__global__ void isneginf_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>((isinf(input[idx]) && input[idx] < 0) ? 1 : 0); }
}
__global__ void isneginf_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = static_cast<uint8_t>((isinf(input[idx]) && input[idx] < 0) ? 1 : 0); }
}
__global__ void isneginf_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = __half2float(input[idx]);
        output[idx] = static_cast<uint8_t>((isinf(v) && v < 0) ? 1 : 0);
    }
}
__global__ void isneginf_kernel_bf16(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = __bfloat162float(input[idx]);
        output[idx] = static_cast<uint8_t>((isinf(v) && v < 0) ? 1 : 0);
    }
}

auto isneginf_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        isneginf_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        isneginf_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        isneginf_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        isneginf_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), result.data<uint8_t>(), n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("isneginf operation only supports floating point dtypes");
    }
    return result;
}

Tensor isneginf_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return isneginf_kernel(inputs[0], stream);
}

// ============================================================================
// IsReal — trivial dtype check, no kernel needed
// ============================================================================

Tensor isreal_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    // All real dtypes return true (scalar bool tensor). Complex would return false.
    // Currently Tenzor has no complex dtype, so always true.
    (void)attrs;
    const auto& input = inputs[0];
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    // Fill with 1 (true) — use cudaMemset since Bool is uint8_t
    if (input.numel() > 0) {
        cudaStream_t stream = nullptr;
        if (!attrs.empty() && attrs.has(AttrKey::Stream)) {
            stream = reinterpret_cast<cudaStream_t>(
                static_cast<uint64_t>(attrs.get_int(AttrKey::Stream, 0)));
        }
        CUDA_CHECK(cudaMemsetAsync(result.data<uint8_t>(), 1, input.numel(), stream));
    }
    return result;
}

// ============================================================================
// Binary math ops: float_power, xlog1py, ldexp
// ============================================================================

// --- float_power: promotes to Float64, computes pow ---
__global__ void float_power_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = pow(a[idx], b[idx]); }
}

auto float_power_kernel(const Tensor& base, const Tensor& exp_tensor, cudaStream_t stream) -> Tensor {
    // Always promote to Float64
    Tensor a = (base.dtype() != DType::Float64) ? base.to(DType::Float64) : base;
    Tensor b = (exp_tensor.dtype() != DType::Float64) ? exp_tensor.to(DType::Float64) : exp_tensor;
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, DType::Float64, a.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    float_power_kernel_f64<<<grid, block, 0, stream>>>(
        a.data<double>(), b.data<double>(), result.data<double>(), n);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor float_power_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return float_power_kernel(inputs[0], inputs[1], stream);
}

// --- xlog1py: x == 0 ? 0 : x * log1p(y) ---
__global__ void xlog1py_kernel_f32(const float* x, const float* y, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = (x[idx] == 0.0f) ? 0.0f : x[idx] * log1pf(y[idx]); }
}
__global__ void xlog1py_kernel_f64(const double* x, const double* y, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = (x[idx] == 0.0) ? 0.0 : x[idx] * log1p(y[idx]); }
}
__global__ void xlog1py_kernel_f16(const __half* x, const __half* y, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float xf = __half2float(x[idx]);
        output[idx] = __float2half((xf == 0.0f) ? 0.0f : xf * log1pf(__half2float(y[idx])));
    }
}
__global__ void xlog1py_kernel_bf16(const __nv_bfloat16* x, const __nv_bfloat16* y, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float xf = __bfloat162float(x[idx]);
        output[idx] = __float2bfloat16((xf == 0.0f) ? 0.0f : xf * log1pf(__bfloat162float(y[idx])));
    }
}

auto xlog1py_kernel(const Tensor& x, const Tensor& y, cudaStream_t stream) -> Tensor {
    if (x.dtype() != y.dtype()) throw std::runtime_error("xlog1py: tensors must have the same dtype");
    int64_t n = x.numel();
    std::vector<int64_t> shape(x.shape().begin(), x.shape().end());
    Tensor result(shape, x.dtype(), x.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (x.dtype() == DType::Float32) {
        xlog1py_kernel_f32<<<grid, block, 0, stream>>>(x.data<float>(), y.data<float>(), result.data<float>(), n);
    } else if (x.dtype() == DType::Float64) {
        xlog1py_kernel_f64<<<grid, block, 0, stream>>>(x.data<double>(), y.data<double>(), result.data<double>(), n);
    } else if (x.dtype() == DType::Float16) {
        xlog1py_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<const __half*>(y.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (x.dtype() == DType::BFloat16) {
        xlog1py_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(y.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("xlog1py only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor xlog1py_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return xlog1py_kernel(inputs[0], inputs[1], stream);
}

// --- ldexp: x * 2^n ---
__global__ void ldexp_kernel_f32(const float* x, const float* n_in, float* output, int64_t total) {
    TENZOR_CUDA_KERNEL_LOOP(idx, total) { output[idx] = ldexpf(x[idx], static_cast<int>(n_in[idx])); }
}
__global__ void ldexp_kernel_f64(const double* x, const double* n_in, double* output, int64_t total) {
    TENZOR_CUDA_KERNEL_LOOP(idx, total) { output[idx] = ldexp(x[idx], static_cast<int>(n_in[idx])); }
}
__global__ void ldexp_kernel_f16(const __half* x, const __half* n_in, __half* output, int64_t total) {
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        output[idx] = __float2half(ldexpf(__half2float(x[idx]), static_cast<int>(__half2float(n_in[idx]))));
    }
}
__global__ void ldexp_kernel_bf16(const __nv_bfloat16* x, const __nv_bfloat16* n_in, __nv_bfloat16* output, int64_t total) {
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        output[idx] = __float2bfloat16(ldexpf(__bfloat162float(x[idx]), static_cast<int>(__bfloat162float(n_in[idx]))));
    }
}

auto ldexp_kernel_impl(const Tensor& x, const Tensor& n_tensor, cudaStream_t stream) -> Tensor {
    if (x.dtype() != n_tensor.dtype()) throw std::runtime_error("ldexp: tensors must have the same dtype");
    int64_t n = x.numel();
    std::vector<int64_t> shape(x.shape().begin(), x.shape().end());
    Tensor result(shape, x.dtype(), x.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (x.dtype() == DType::Float32) {
        ldexp_kernel_f32<<<grid, block, 0, stream>>>(x.data<float>(), n_tensor.data<float>(), result.data<float>(), n);
    } else if (x.dtype() == DType::Float64) {
        ldexp_kernel_f64<<<grid, block, 0, stream>>>(x.data<double>(), n_tensor.data<double>(), result.data<double>(), n);
    } else if (x.dtype() == DType::Float16) {
        ldexp_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<const __half*>(n_tensor.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (x.dtype() == DType::BFloat16) {
        ldexp_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(n_tensor.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("ldexp only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor ldexp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return ldexp_kernel_impl(inputs[0], inputs[1], stream);
}

// ============================================================================
// Frexp: decompose into mantissa and exponent (two outputs)
// ============================================================================

__global__ void frexp_kernel_f32(const float* in, float* mantissa, int32_t* exponent, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int e;
        mantissa[idx] = frexpf(in[idx], &e);
        exponent[idx] = e;
    }
}
__global__ void frexp_kernel_f64(const double* in, double* mantissa, int32_t* exponent, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int e;
        mantissa[idx] = frexp(in[idx], &e);
        exponent[idx] = e;
    }
}
__global__ void frexp_kernel_f16(const __half* in, __half* mantissa, int32_t* exponent, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int e;
        float m = frexpf(__half2float(in[idx]), &e);
        mantissa[idx] = __float2half(m);
        exponent[idx] = e;
    }
}
__global__ void frexp_kernel_bf16(const __nv_bfloat16* in, __nv_bfloat16* mantissa, int32_t* exponent, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int e;
        float m = frexpf(__bfloat162float(in[idx]), &e);
        mantissa[idx] = __float2bfloat16(m);
        exponent[idx] = e;
    }
}

// F1: frexp_dispatch now honestly returns both (mantissa, exponent) so the
// OpId::Frexp dispatch contract matches CPU. Previously the function returned
// only the mantissa as a `Tensor` via `register_single_output_kernel`, and the
// exponent — already computed inside the CUDA kernel — was simply discarded
// at the dispatch boundary. Anyone calling `dispatch(OpId::Frexp, …)` on CUDA
// got back a 1-element output vector and silently lost half the result.
std::vector<Tensor> frexp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    const auto& input = inputs[0];
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor mantissa(shape, input.dtype(), input.device());
    Tensor exponent(shape, DType::Int32, input.device());
    if (n == 0) return {mantissa, exponent};
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        frexp_kernel_f32<<<grid, block, 0, stream>>>(
            input.data<float>(), mantissa.data<float>(), exponent.data<int32_t>(), n);
    } else if (input.dtype() == DType::Float64) {
        frexp_kernel_f64<<<grid, block, 0, stream>>>(
            input.data<double>(), mantissa.data<double>(), exponent.data<int32_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        frexp_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(mantissa.data<Float16>()),
            exponent.data<int32_t>(), n);
    } else if (input.dtype() == DType::BFloat16) {
        frexp_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(mantissa.data<BFloat16>()),
            exponent.data<int32_t>(), n);
    } else {
        throw std::runtime_error("frexp only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return {mantissa, exponent};
}

// ============================================================================
// DiagEmbed: embed vector(s) as batch diagonal matrices
// ============================================================================

template<typename T>
__global__ void diag_embed_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch_size,
    int64_t diag_len,
    int64_t n,
    int64_t offset
) {
    int64_t total = batch_size * n * n;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t b = idx / (n * n);
        int64_t rem = idx % (n * n);
        int64_t i = rem / n;
        int64_t j = rem % n;

        int64_t diag_idx;
        if (offset >= 0) {
            diag_idx = (i == j - offset && j >= offset && i < diag_len) ? i : -1;
        } else {
            diag_idx = (j == i + offset && i >= -offset && j < diag_len) ? j : -1;
        }

        if (diag_idx >= 0 && diag_idx < diag_len) {
            output[idx] = input[b * diag_len + diag_idx];
        } else {
            output[idx] = T(0);
        }
    }
}

template<>
__global__ void diag_embed_kernel_impl<__half>(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t batch_size,
    int64_t diag_len,
    int64_t n,
    int64_t offset
) {
    int64_t total = batch_size * n * n;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t b = idx / (n * n);
        int64_t rem = idx % (n * n);
        int64_t i = rem / n;
        int64_t j = rem % n;

        int64_t diag_idx;
        if (offset >= 0) {
            diag_idx = (i == j - offset && j >= offset && i < diag_len) ? i : -1;
        } else {
            diag_idx = (j == i + offset && i >= -offset && j < diag_len) ? j : -1;
        }

        output[idx] = (diag_idx >= 0 && diag_idx < diag_len)
            ? input[b * diag_len + diag_idx] : __float2half(0.0f);
    }
}

template<>
__global__ void diag_embed_kernel_impl<__nv_bfloat16>(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int64_t batch_size,
    int64_t diag_len,
    int64_t n,
    int64_t offset
) {
    int64_t total = batch_size * n * n;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t b = idx / (n * n);
        int64_t rem = idx % (n * n);
        int64_t i = rem / n;
        int64_t j = rem % n;

        int64_t diag_idx;
        if (offset >= 0) {
            diag_idx = (i == j - offset && j >= offset && i < diag_len) ? i : -1;
        } else {
            diag_idx = (j == i + offset && i >= -offset && j < diag_len) ? j : -1;
        }

        output[idx] = (diag_idx >= 0 && diag_idx < diag_len)
            ? input[b * diag_len + diag_idx] : __float2bfloat16(0.0f);
    }
}

Tensor diag_embed_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    const auto& input = inputs[0];
    int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);

    // Last dimension is the diagonal length
    int64_t diag_len = input.shape().back();
    int64_t n = diag_len + std::abs(offset);

    // Compute batch size (all dims except last)
    int64_t batch_size = 1;
    for (int64_t i = 0; i + 1 < input.ndim(); ++i) {
        batch_size *= input.shape()[i];
    }

    // Build output shape: [batch_dims..., n, n]
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i + 1 < input.ndim(); ++i) {
        out_shape.push_back(input.shape()[i]);
    }
    out_shape.push_back(n);
    out_shape.push_back(n);

    Tensor result(out_shape, input.dtype(), input.device());
    int64_t total = batch_size * n * n;
    if (total == 0) return result;

    Tensor cont = input.is_contiguous() ? input : input.contiguous();
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    switch (input.dtype()) {
        case DType::Float32:
            diag_embed_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<float>(), result.data<float>(), batch_size, diag_len, n, offset);
            break;
        case DType::Float64:
            diag_embed_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<double>(), result.data<double>(), batch_size, diag_len, n, offset);
            break;
        case DType::Float16:
            diag_embed_kernel_impl<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(cont.data_ptr()),
                reinterpret_cast<__half*>(result.data_ptr()),
                batch_size, diag_len, n, offset);
            break;
        case DType::BFloat16:
            diag_embed_kernel_impl<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(cont.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(result.data_ptr()),
                batch_size, diag_len, n, offset);
            break;
        case DType::Int32:
            diag_embed_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<int32_t>(), result.data<int32_t>(), batch_size, diag_len, n, offset);
            break;
        case DType::Int64:
            diag_embed_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<int64_t>(), result.data<int64_t>(), batch_size, diag_len, n, offset);
            break;
        default:
            throw std::runtime_error("diag_embed: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// Diagflat is registered directly in cuda_kernel_registry.cpp since it
// delegates to diag_kernel (defined in transform.cu).

// ============================================================================
// NanVar and NanStd: NaN-ignoring variance and standard deviation
// ============================================================================

// Full-reduction NaN-aware variance kernel
__global__ void nanvar_all_f32(const float* input, float* output, int64_t n, float correction) {
    // Two-pass: first compute nanmean, then sum of squared deviations
    __shared__ float ssum;
    __shared__ int64_t scount;
    if (threadIdx.x == 0) { ssum = 0.0f; scount = 0; }
    __syncthreads();

    // Pass 1: sum + count
    float local_sum = 0.0f;
    int64_t local_count = 0;
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = input[idx];
        if (!isnan(v)) { local_sum += v; local_count++; }
    }
    atomicAdd(&ssum, local_sum);
    atomicAdd(reinterpret_cast<unsigned long long*>(&scount), static_cast<unsigned long long>(local_count));
    __syncthreads();

    float mean = (scount > 0) ? ssum / static_cast<float>(scount) : 0.0f;
    int64_t count = scount;

    // Reset shared mem for pass 2
    __shared__ float svar;
    if (threadIdx.x == 0) svar = 0.0f;
    __syncthreads();

    // Pass 2: sum of squared deviations
    float local_var = 0.0f;
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float v = input[idx];
        if (!isnan(v)) {
            float diff = v - mean;
            local_var += diff * diff;
        }
    }
    atomicAdd(&svar, local_var);
    __syncthreads();

    if (threadIdx.x == 0) {
        float denom = static_cast<float>(count) - correction;
        output[0] = (denom > 0.0f) ? svar / denom : 0.0f;
    }
}

Tensor nanvar_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    const auto& input_orig = inputs[0];
    float correction = static_cast<float>(attrs.get_int(AttrKey::Correction, 1));

    // Upcast to Float32 for computation
    Tensor input = (input_orig.dtype() != DType::Float32) ? input_orig.to(DType::Float32) : input_orig;
    int64_t n = input.numel();

    Tensor result({1}, DType::Float32, input.device());
    if (n == 0) return result;

    nanvar_all_f32<<<1, 256, 0, stream>>>(input.data<float>(), result.data<float>(), n, correction);
    CUDA_CHECK(cudaGetLastError());
    return (input_orig.dtype() != DType::Float32) ? result.to(input_orig.dtype()) : result;
}

// --- nanstd = sqrt(nanvar) ---
__global__ void sqrt_scalar_f32(const float* input, float* output) {
    if (threadIdx.x == 0) output[0] = sqrtf(input[0]);
}

Tensor nanstd_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    // Compute nanvar first
    Tensor var = nanvar_dispatch(inputs, attrs);
    // Then sqrt
    Tensor result({1}, DType::Float32, var.device());
    Tensor var_f32 = (var.dtype() != DType::Float32) ? var.to(DType::Float32) : var;
    sqrt_scalar_f32<<<1, 1, 0, stream>>>(var_f32.data<float>(), result.data<float>());
    CUDA_CHECK(cudaGetLastError());
    DType orig_dtype = inputs[0].dtype();
    return (orig_dtype != DType::Float32) ? result.to(orig_dtype) : result;
}

// ============================================================================
// LogAddExp: max(a,b) + log1p(exp(-|a-b|))
// ============================================================================

__global__ void logaddexp_kernel_f32(const float* a, const float* b, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float ai = a[idx], bi = b[idx];
        float m = fmaxf(ai, bi);
        out[idx] = m + log1pf(expf(-fabsf(ai - bi)));
    }
}
__global__ void logaddexp_kernel_f64(const double* a, const double* b, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double ai = a[idx], bi = b[idx];
        double m = fmax(ai, bi);
        out[idx] = m + log1p(exp(-fabs(ai - bi)));
    }
}
__global__ void logaddexp_kernel_f16(const __half* a, const __half* b, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float ai = __half2float(a[idx]), bi = __half2float(b[idx]);
        float m = fmaxf(ai, bi);
        out[idx] = __float2half(m + log1pf(expf(-fabsf(ai - bi))));
    }
}
__global__ void logaddexp_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float ai = __bfloat162float(a[idx]), bi = __bfloat162float(b[idx]);
        float m = fmaxf(ai, bi);
        out[idx] = __float2bfloat16(m + log1pf(expf(-fabsf(ai - bi))));
    }
}

auto logaddexp_kernel_impl(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("logaddexp: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        logaddexp_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        logaddexp_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        logaddexp_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        logaddexp_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("logaddexp only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor logaddexp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return logaddexp_kernel_impl(inputs[0], inputs[1], stream);
}

// ============================================================================
// LogAddExp2: max(a,b) + log2(1 + exp2(-|a-b|))
// ============================================================================

__global__ void logaddexp2_kernel_f32(const float* a, const float* b, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float ai = a[idx], bi = b[idx];
        float m = fmaxf(ai, bi);
        out[idx] = m + log2f(1.0f + exp2f(-fabsf(ai - bi)));
    }
}
__global__ void logaddexp2_kernel_f64(const double* a, const double* b, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double ai = a[idx], bi = b[idx];
        double m = fmax(ai, bi);
        out[idx] = m + log2(1.0 + exp2(-fabs(ai - bi)));
    }
}
__global__ void logaddexp2_kernel_f16(const __half* a, const __half* b, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float ai = __half2float(a[idx]), bi = __half2float(b[idx]);
        float m = fmaxf(ai, bi);
        out[idx] = __float2half(m + log2f(1.0f + exp2f(-fabsf(ai - bi))));
    }
}
__global__ void logaddexp2_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float ai = __bfloat162float(a[idx]), bi = __bfloat162float(b[idx]);
        float m = fmaxf(ai, bi);
        out[idx] = __float2bfloat16(m + log2f(1.0f + exp2f(-fabsf(ai - bi))));
    }
}

auto logaddexp2_kernel_impl(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) throw std::runtime_error("logaddexp2: tensors must have the same dtype");
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        logaddexp2_kernel_f32<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        logaddexp2_kernel_f64<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        logaddexp2_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        logaddexp2_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("logaddexp2 only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor logaddexp2_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return logaddexp2_kernel_impl(inputs[0], inputs[1], stream);
}

// ============================================================================
// XLogY: x == 0 ? 0 : x * log(y)
// ============================================================================

__global__ void xlogy_kernel_f32(const float* x, const float* y, float* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = (x[idx] == 0.0f) ? 0.0f : x[idx] * logf(y[idx]);
    }
}
__global__ void xlogy_kernel_f64(const double* x, const double* y, double* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        out[idx] = (x[idx] == 0.0) ? 0.0 : x[idx] * log(y[idx]);
    }
}
__global__ void xlogy_kernel_f16(const __half* x, const __half* y, __half* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float xf = __half2float(x[idx]);
        out[idx] = __float2half((xf == 0.0f) ? 0.0f : xf * logf(__half2float(y[idx])));
    }
}
__global__ void xlogy_kernel_bf16(const __nv_bfloat16* x, const __nv_bfloat16* y, __nv_bfloat16* out, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float xf = __bfloat162float(x[idx]);
        out[idx] = __float2bfloat16((xf == 0.0f) ? 0.0f : xf * logf(__bfloat162float(y[idx])));
    }
}

auto xlogy_kernel_impl(const Tensor& x, const Tensor& y, cudaStream_t stream) -> Tensor {
    if (x.dtype() != y.dtype()) throw std::runtime_error("xlogy: tensors must have the same dtype");
    int64_t n = x.numel();
    std::vector<int64_t> shape(x.shape().begin(), x.shape().end());
    Tensor result(shape, x.dtype(), x.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (x.dtype() == DType::Float32) {
        xlogy_kernel_f32<<<grid, block, 0, stream>>>(x.data<float>(), y.data<float>(), result.data<float>(), n);
    } else if (x.dtype() == DType::Float64) {
        xlogy_kernel_f64<<<grid, block, 0, stream>>>(x.data<double>(), y.data<double>(), result.data<double>(), n);
    } else if (x.dtype() == DType::Float16) {
        xlogy_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<const __half*>(y.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (x.dtype() == DType::BFloat16) {
        xlogy_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(y.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("xlogy only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor xlogy_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return xlogy_kernel_impl(inputs[0], inputs[1], stream);
}

// ============================================================================
// I0e: exp(-|x|) * BesselI0(x) — Abramowitz & Stegun polynomial approx
// ============================================================================

// Chebyshev polynomial coefficients for I0e
// |x| <= 3.75: I0(x) = sum(A[k] * t^(2k)), t = x/3.75, then multiply by exp(-|x|)
// |x| > 3.75: I0e(x) = (1/sqrt(|x|)) * sum(B[k] * (3.75/|x|)^k)
__device__ __forceinline__ float i0e_f32(float x) {
    float ax = fabsf(x);
    if (ax < 3.75f) {
        float t = x / 3.75f;
        float t2 = t * t;
        float val = 1.0f + t2 * (3.5156229f + t2 * (3.0899424f + t2 * (1.2067492f
                  + t2 * (0.2659732f + t2 * (0.0360768f + t2 * 0.0045813f)))));
        return val * expf(-ax);
    } else {
        float t = 3.75f / ax;
        float val = (0.39894228f + t * (0.01328592f + t * (0.00225319f
                  + t * (-0.00157565f + t * (0.00916281f + t * (-0.02057706f
                  + t * (0.02635537f + t * (-0.01647633f + t * 0.00392377f))))))));
        return val / sqrtf(ax);
    }
}

__device__ __forceinline__ double i0e_f64(double x) {
    double ax = fabs(x);
    if (ax < 3.75) {
        double t = x / 3.75;
        double t2 = t * t;
        double val = 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492
                   + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
        return val * exp(-ax);
    } else {
        double t = 3.75 / ax;
        double val = (0.39894228 + t * (0.01328592 + t * (0.00225319
                   + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706
                   + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))));
        return val / sqrt(ax);
    }
}

__global__ void i0e_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = i0e_f32(input[idx]); }
}
__global__ void i0e_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = i0e_f64(input[idx]); }
}
__global__ void i0e_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(i0e_f32(__half2float(input[idx])));
    }
}
__global__ void i0e_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2bfloat16(i0e_f32(__bfloat162float(input[idx])));
    }
}

auto i0e_kernel_impl(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        i0e_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        i0e_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        i0e_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        i0e_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("i0e only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor i0e_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return i0e_kernel_impl(inputs[0], stream);
}

// ============================================================================
// I1e: exp(-|x|) * BesselI1(x) — Abramowitz & Stegun polynomial approx
// ============================================================================

__device__ __forceinline__ float i1e_f32(float x) {
    float ax = fabsf(x);
    float val;
    if (ax < 3.75f) {
        float t = x / 3.75f;
        float t2 = t * t;
        val = ax * (0.5f + t2 * (0.87890594f + t2 * (0.51498869f + t2 * (0.15084934f
            + t2 * (0.02658733f + t2 * (0.00301532f + t2 * 0.00032411f))))));
        val *= expf(-ax);
    } else {
        float t = 3.75f / ax;
        val = (0.39894228f + t * (-0.03988024f + t * (-0.00362018f
            + t * (0.00163801f + t * (-0.01031555f + t * (0.02282967f
            + t * (-0.02895312f + t * (0.01787654f + t * (-0.00420059f)))))))));
        val /= sqrtf(ax);
    }
    return (x < 0.0f) ? -val : val;
}

__device__ __forceinline__ double i1e_f64(double x) {
    double ax = fabs(x);
    double val;
    if (ax < 3.75) {
        double t = x / 3.75;
        double t2 = t * t;
        val = ax * (0.5 + t2 * (0.87890594 + t2 * (0.51498869 + t2 * (0.15084934
            + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411))))));
        val *= exp(-ax);
    } else {
        double t = 3.75 / ax;
        val = (0.39894228 + t * (-0.03988024 + t * (-0.00362018
            + t * (0.00163801 + t * (-0.01031555 + t * (0.02282967
            + t * (-0.02895312 + t * (0.01787654 + t * (-0.00420059)))))))));
        val /= sqrt(ax);
    }
    return (x < 0.0) ? -val : val;
}

__global__ void i1e_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = i1e_f32(input[idx]); }
}
__global__ void i1e_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) { output[idx] = i1e_f64(input[idx]); }
}
__global__ void i1e_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(i1e_f32(__half2float(input[idx])));
    }
}
__global__ void i1e_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __float2bfloat16(i1e_f32(__bfloat162float(input[idx])));
    }
}

auto i1e_kernel_impl(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        i1e_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        i1e_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        i1e_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        i1e_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("i1e only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor i1e_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return i1e_kernel_impl(inputs[0], stream);
}

// ============================================================================
// Entr: -x * log(x), with 0 -> 0, negative -> -inf
// ============================================================================

__global__ void entr_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = input[idx];
        if (x == 0.0f) output[idx] = 0.0f;
        else if (x < 0.0f) output[idx] = -__int_as_float(0x7f800000);
        else output[idx] = -x * logf(x);
    }
}
__global__ void entr_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double x = input[idx];
        if (x == 0.0) output[idx] = 0.0;
        else if (x < 0.0) output[idx] = -__longlong_as_double(0x7ff0000000000000LL);
        else output[idx] = -x * log(x);
    }
}
__global__ void entr_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __half2float(input[idx]);
        float r;
        if (x == 0.0f) r = 0.0f;
        else if (x < 0.0f) r = -__int_as_float(0x7f800000);
        else r = -x * logf(x);
        output[idx] = __float2half(r);
    }
}
__global__ void entr_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __bfloat162float(input[idx]);
        float r;
        if (x == 0.0f) r = 0.0f;
        else if (x < 0.0f) r = -__int_as_float(0x7f800000);
        else r = -x * logf(x);
        output[idx] = __float2bfloat16(r);
    }
}

auto entr_kernel_impl(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        entr_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        entr_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        entr_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        entr_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("entr only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor entr_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return entr_kernel_impl(inputs[0], stream);
}

// ============================================================================
// SphericalBesselJ0: x == 0 ? 1 : sin(x)/x
// ============================================================================

__global__ void spherical_bessel_j0_kernel_f32(const float* input, float* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = input[idx];
        output[idx] = (x == 0.0f) ? 1.0f : sinf(x) / x;
    }
}
__global__ void spherical_bessel_j0_kernel_f64(const double* input, double* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double x = input[idx];
        output[idx] = (x == 0.0) ? 1.0 : sin(x) / x;
    }
}
__global__ void spherical_bessel_j0_kernel_f16(const __half* input, __half* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half((x == 0.0f) ? 1.0f : sinf(x) / x);
    }
}
__global__ void spherical_bessel_j0_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        float x = __bfloat162float(input[idx]);
        output[idx] = __float2bfloat16((x == 0.0f) ? 1.0f : sinf(x) / x);
    }
}

auto spherical_bessel_j0_kernel_impl(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        spherical_bessel_j0_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        spherical_bessel_j0_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        spherical_bessel_j0_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        spherical_bessel_j0_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("spherical_bessel_j0 only supports Float32, Float64, Float16, BFloat16");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor spherical_bessel_j0_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = get_dispatch_stream(attrs, inputs[0]);
    return spherical_bessel_j0_kernel_impl(inputs[0], stream);
}

} // namespace cuda
} // namespace tenzor
