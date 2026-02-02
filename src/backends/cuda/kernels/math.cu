#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/backend.hpp"  // For OpAttributes (dispatch wrappers)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <charconv>  // For std::from_chars (dispatch wrappers)
#include <span>      // For std::span (dispatch wrappers)

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

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

// Compute optimal grid/block dimensions for 1D kernels
inline void compute_launch_config_1d(int64_t n, dim3& grid, dim3& block) {
    const int block_size = 256;  // Optimal for most GPUs
    block = dim3(block_size, 1, 1);
    // Ensure at least 1 block to avoid CUDA invalid argument error
    // Grid-stride loop will naturally handle n=0 by not executing any iterations
    int64_t num_blocks = (n + block_size - 1) / block_size;
    grid = dim3(num_blocks > 0 ? static_cast<unsigned int>(num_blocks) : 1, 1, 1);
}

// Grid-stride loop pattern for better scalability
#define CUDA_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// ============================================================================
// Division by Zero Check (for integer types)
// ============================================================================

// Kernel to check if any element is zero (for integer division check)
template<typename T>
__global__ void check_for_zeros_kernel(const T* data, int64_t n, int* has_zero) {
    CUDA_KERNEL_LOOP(idx, n) {
        if (data[idx] == T(0)) {
            atomicExch(has_zero, 1);
        }
    }
}

// Host function to check for zeros in an integer tensor
template<typename T>
inline void check_integer_divisor_for_zeros(const T* data, int64_t n, cudaStream_t stream) {
    int* d_has_zero;
    int h_has_zero = 0;

    CUDA_CHECK(cudaMalloc(&d_has_zero, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_has_zero, 0, sizeof(int), stream));

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    check_for_zeros_kernel<<<grid, block, 0, stream>>>(data, n, d_has_zero);

    CUDA_CHECK(cudaMemcpyAsync(&h_has_zero, d_has_zero, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // Need to sync to check the flag
    CUDA_CHECK(cudaFree(d_has_zero));

    if (h_has_zero) {
        throw std::runtime_error("Integer division by zero");
    }
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
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Generic broadcast kernel - works for all binary operations
template<typename T, typename Op>
__global__ void broadcast_kernel(
    const T* a, const T* b, T* c,
    const int64_t* strides_a, const int64_t* strides_b,
    const int64_t* output_shape, int64_t ndim, int64_t n, Op op) {

    CUDA_KERNEL_LOOP(out_idx, n) {
        int64_t idx_a = 0;
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        // Convert flat index to multi-dimensional indices
        // Working from rightmost (fastest-varying) dimension
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % output_shape[i];
            tmp /= output_shape[i];
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
        }

        c[out_idx] = op(a[idx_a], b[idx_b]);
    }
}

// Device-side operation functors
struct AddOp {
    template<typename T>
    __device__ T operator()(T a, T b) const { return a + b; }
};

struct SubOp {
    template<typename T>
    __device__ T operator()(T a, T b) const { return a - b; }
};

struct MulOp {
    template<typename T>
    __device__ T operator()(T a, T b) const { return a * b; }
};

struct DivOp {
    template<typename T>
    __device__ T operator()(T a, T b) const {
        if (b == T(0)) {
            return T(INFINITY);
        }
        return a / b;
    }
};

// Generic in-place broadcast kernel - works for all in-place binary operations
// Reads from a (target) and b (other), writes result back to a
template<typename T, typename Op>
__global__ void broadcast_inplace_kernel(
    T* a, const T* b,
    const int64_t* strides_b,
    const int64_t* output_shape, int64_t ndim, int64_t n, Op op) {

    CUDA_KERNEL_LOOP(out_idx, n) {
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        // Convert flat index to multi-dimensional indices
        // Working from rightmost (fastest-varying) dimension
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % output_shape[i];
            tmp /= output_shape[i];
            idx_b += coord * strides_b[i];
        }

        a[out_idx] = op(a[out_idx], b[idx_b]);
    }
}

// Subtract kernel - element-wise subtraction
template<typename T>
__global__ void sub_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = a[idx] - b[idx];
    }
}

// Multiply kernel - element-wise multiplication
template<typename T>
__global__ void mul_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Divide kernel - element-wise division
template<typename T>
__global__ void div_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        T divisor = b[idx];
        if (divisor == T(0)) {
            c[idx] = INFINITY;  // Handle division by zero
        } else {
            c[idx] = a[idx] / divisor;
        }
    }
}

// ============================================================================
// FP16 Saturating Conversion
// ============================================================================

// Saturating Float32 → Float16 conversion: clamps to max finite Float16 value
// instead of producing Inf. This matches CPU Float16 operator behavior where
// per-element clamping naturally limits value growth through deep networks.
__device__ __forceinline__ __half __float2half_sat(float x) {
    constexpr float kHalfMax = 65504.0f;
    x = fminf(fmaxf(x, -kHalfMax), kHalfMax);
    return __float2half(x);
}

// ============================================================================
// FP16 Binary Operations
// ============================================================================

// FP16 addition kernel (compute in Float32, saturating conversion)
__global__ void add_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = __float2half_sat(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

// FP16 subtraction kernel (compute in Float32, saturating conversion)
__global__ void sub_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = __float2half_sat(__half2float(a[idx]) - __half2float(b[idx]));
    }
}

// FP16 multiplication kernel (compute in Float32, saturating conversion)
__global__ void mul_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = __float2half_sat(__half2float(a[idx]) * __half2float(b[idx]));
    }
}

// FP16 division kernel (compute in Float32, saturating conversion)
__global__ void div_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = __float2half_sat(__half2float(a[idx]) / __half2float(b[idx]));
    }
}

// ============================================================================
// BFloat16 Binary Operations
// ============================================================================

// BFloat16 addition kernel
__global__ void add_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// BFloat16 subtraction kernel
__global__ void sub_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = __hsub(a[idx], b[idx]);
    }
}

// BFloat16 multiplication kernel
__global__ void mul_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = __hmul(a[idx], b[idx]);
    }
}

// BFloat16 division kernel
__global__ void div_kernel_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = __hdiv(a[idx], b[idx]);
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

// Negate kernel
template<typename T>
__global__ void neg_kernel_device(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = -input[idx];
    }
}

// Absolute value kernel
template<typename T>
__global__ void abs_kernel_device(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        T val = input[idx];
        output[idx] = val >= T(0) ? val : -val;
    }
}

// Absolute value kernel (specialized for float)
__global__ void abs_kernel_f32(const float* input, float* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = fabsf(input[idx]);
    }
}

// Absolute value kernel (specialized for double)
__global__ void abs_kernel_f64(const double* input, double* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = fabs(input[idx]);
    }
}

// ============================================================================
// FP16 Unary Operations
// ============================================================================

// FP16 negate kernel
__global__ void neg_kernel_f16(const __half* input, __half* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __hneg(input[idx]);
    }
}

// FP16 absolute value kernel
__global__ void abs_kernel_f16(const __half* input, __half* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __habs(input[idx]);
    }
}

// ============================================================================
// BFloat16 Unary Operations
// ============================================================================

// BFloat16 negate kernel
__global__ void neg_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __hneg(input[idx]);
    }
}

// BFloat16 absolute value kernel
__global__ void abs_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = __habs(input[idx]);
    }
}

// ============================================================================
// Mathematical Functions
// ============================================================================

// Square root kernel (float)
__global__ void sqrt_kernel_f32(const float* input, float* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = sqrtf(input[idx]);
    }
}

// Square root kernel (double)
__global__ void sqrt_kernel_f64(const double* input, double* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = sqrt(input[idx]);
    }
}

// Exponential kernel (float)
__global__ void exp_kernel_f32(const float* input, float* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = expf(input[idx]);
    }
}

// Exponential kernel (double)
__global__ void exp_kernel_f64(const double* input, double* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = exp(input[idx]);
    }
}

// Natural logarithm kernel (float)
__global__ void log_kernel_f32(const float* input, float* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = logf(input[idx]);
    }
}

// Natural logarithm kernel (double)
__global__ void log_kernel_f64(const double* input, double* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = log(input[idx]);
    }
}

// Power kernel (float)
__global__ void pow_kernel_f32(const float* input, float* output, float exponent, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = powf(input[idx], exponent);
    }
}

// Power kernel (double)
__global__ void pow_kernel_f64(const double* input, double* output, double exponent, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = pow(input[idx], exponent);
    }
}

// Clamp kernel (float)
__global__ void clamp_kernel_f32(const float* input, float* output, float min_val, float max_val, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        float val = input[idx];
        output[idx] = fminf(fmaxf(val, min_val), max_val);
    }
}

// Clamp kernel (double)
__global__ void clamp_kernel_f64(const double* input, double* output, double min_val, double max_val, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        double val = input[idx];
        output[idx] = fmin(fmax(val, min_val), max_val);
    }
}

// Sign kernel (float)
__global__ void sign_kernel_f32(const float* input, float* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        float val = input[idx];
        // Sign function: -1 if x < 0, 0 if x == 0, +1 if x > 0
        output[idx] = (val > 0.0f) - (val < 0.0f);
    }
}

// Sign kernel (double)
__global__ void sign_kernel_f64(const double* input, double* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        double val = input[idx];
        output[idx] = (val > 0.0) - (val < 0.0);
    }
}

// ============================================================================
// FP16 Mathematical Functions
// ============================================================================

// FP16 square root kernel
__global__ void sqrt_kernel_f16(const __half* input, __half* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hsqrt(input[idx]);
    }
}

// FP16 exponential kernel
__global__ void exp_kernel_f16(const __half* input, __half* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hexp(input[idx]);
    }
}

// FP16 natural logarithm kernel
__global__ void log_kernel_f16(const __half* input, __half* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hlog(input[idx]);
    }
}

// FP16 power kernel
__global__ void pow_kernel_f16(const __half* input, __half* output, __half exponent, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        float exp_val = __half2float(exponent);
        output[idx] = __float2half(powf(val, exp_val));
    }
}

// FP16 clamp kernel
__global__ void clamp_kernel_f16(const __half* input, __half* output, __half min_val, __half max_val, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        __half val = input[idx];
        output[idx] = __hmax(__hmin(val, max_val), min_val);
    }
}

// FP16 sign kernel
__global__ void sign_kernel_f16(const __half* input, __half* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
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
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hsqrt(input[idx]);
    }
}

// BFloat16 exponential kernel
__global__ void exp_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hexp(input[idx]);
    }
}

// BFloat16 natural logarithm kernel
__global__ void log_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hlog(input[idx]);
    }
}

// BFloat16 power kernel
__global__ void pow_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, __nv_bfloat16 exponent, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        float val = __bfloat162float(input[idx]);
        float exp_val = __bfloat162float(exponent);
        output[idx] = __float2bfloat16(powf(val, exp_val));
    }
}

// BFloat16 clamp kernel
__global__ void clamp_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, __nv_bfloat16 min_val, __nv_bfloat16 max_val, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        __nv_bfloat16 val = input[idx];
        output[idx] = __hmax(__hmin(val, max_val), min_val);
    }
}

// BFloat16 sign kernel
__global__ void sign_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
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
        } else if (a.dtype() == DType::Float64) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<double>(), b.data<double>(), result.data<double>(), n);
        } else if (a.dtype() == DType::Int32) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else if (a.dtype() == DType::Float16) {
            add_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
        } else if (a.dtype() == DType::BFloat16) {
            add_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        } else if (a.dtype() == DType::Int8) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n);
        } else if (a.dtype() == DType::UInt8) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(), n);
        } else if (a.dtype() == DType::Int16) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(), n);
        } else if (a.dtype() == DType::Bool) {
            add_kernel_device<<<grid, block, 0, stream>>>(
                a.data<bool>(), b.data<bool>(), result.data<bool>(), n);
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

    // Copy strides to device
    int64_t* d_strides_a;
    int64_t* d_strides_b;
    int64_t* d_output_shape;
    CUDA_CHECK(cudaMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Float64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<double>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Int32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Int64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Float16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::BFloat16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Int8) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::UInt8) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Int16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Bool) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<bool>(), b.data<bool>(), result.data<bool>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else {
        throw std::runtime_error("Unsupported dtype for add operation");
    }

    // Cleanup - note: cudaFree is synchronous, so no explicit sync needed after it
    CUDA_CHECK(cudaFree(d_strides_a));
    CUDA_CHECK(cudaFree(d_strides_b));
    CUDA_CHECK(cudaFree(d_output_shape));
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
        } else if (a.dtype() == DType::Float64) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
        } else if (a.dtype() == DType::Int32) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else if (a.dtype() == DType::Float16) {
            sub_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
        } else if (a.dtype() == DType::BFloat16) {
            sub_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        } else if (a.dtype() == DType::Int8) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n);
        } else if (a.dtype() == DType::UInt8) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(), n);
        } else if (a.dtype() == DType::Int16) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(), n);
        } else if (a.dtype() == DType::UInt16) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<uint16_t>(), b.data<uint16_t>(), result.data<uint16_t>(), n);
        } else if (a.dtype() == DType::UInt32) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<uint32_t>(), b.data<uint32_t>(), result.data<uint32_t>(), n);
        } else if (a.dtype() == DType::UInt64) {
            sub_kernel_device<<<grid, block, 0, stream>>>(a.data<uint64_t>(), b.data<uint64_t>(), result.data<uint64_t>(), n);
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

    int64_t* d_strides_a;
    int64_t* d_strides_b;
    int64_t* d_output_shape;
    CUDA_CHECK(cudaMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Float64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<double>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Int32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Int64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Float16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::BFloat16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Int8) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::UInt8) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Int16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::UInt16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<uint16_t>(), b.data<uint16_t>(), result.data<uint16_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::UInt32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<uint32_t>(), b.data<uint32_t>(), result.data<uint32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::UInt64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<uint64_t>(), b.data<uint64_t>(), result.data<uint64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else {
        throw std::runtime_error("Unsupported dtype for sub operation");
    }

    CUDA_CHECK(cudaFree(d_strides_a));
    CUDA_CHECK(cudaFree(d_strides_b));
    CUDA_CHECK(cudaFree(d_output_shape));
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
        } else if (a.dtype() == DType::Float64) {
            mul_kernel_device<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
        } else if (a.dtype() == DType::Int32) {
            mul_kernel_device<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            mul_kernel_device<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else if (a.dtype() == DType::Float16) {
            mul_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
        } else if (a.dtype() == DType::BFloat16) {
            mul_kernel_device<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
        } else if (a.dtype() == DType::Bool) {
            mul_kernel_device<<<grid, block, 0, stream>>>(a.data<bool>(), b.data<bool>(), result.data<bool>(), n);
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

    int64_t* d_strides_a;
    int64_t* d_strides_b;
    int64_t* d_output_shape;
    CUDA_CHECK(cudaMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Float64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<double>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Int32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Int64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Float16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::BFloat16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Bool) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<bool>(), b.data<bool>(), result.data<bool>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else {
        throw std::runtime_error("Unsupported dtype for mul operation");
    }

    CUDA_CHECK(cudaFree(d_strides_a));
    CUDA_CHECK(cudaFree(d_strides_b));
    CUDA_CHECK(cudaFree(d_output_shape));
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
        } else if (a.dtype() == DType::Float64) {
            div_kernel_device<<<grid, block, 0, stream>>>(a.data<double>(), b.data<double>(), result.data<double>(), n);
        } else if (a.dtype() == DType::Int32) {
            check_integer_divisor_for_zeros(b.data<int32_t>(), n, stream);
            div_kernel_device<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            check_integer_divisor_for_zeros(b.data<int64_t>(), n, stream);
            div_kernel_device<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else if (a.dtype() == DType::Float16) {
            div_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
        } else if (a.dtype() == DType::BFloat16) {
            div_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
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

    int64_t* d_strides_a;
    int64_t* d_strides_b;
    int64_t* d_output_shape;
    CUDA_CHECK(cudaMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else if (a.dtype() == DType::Float64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<double>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else if (a.dtype() == DType::Int32) {
        check_integer_divisor_for_zeros(b.data<int32_t>(), b.numel(), stream);
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else if (a.dtype() == DType::Int64) {
        check_integer_divisor_for_zeros(b.data<int64_t>(), b.numel(), stream);
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else if (a.dtype() == DType::Float16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else if (a.dtype() == DType::BFloat16) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else {
        throw std::runtime_error("Unsupported dtype for div operation");
    }

    CUDA_CHECK(cudaFree(d_strides_a));
    CUDA_CHECK(cudaFree(d_strides_b));
    CUDA_CHECK(cudaFree(d_output_shape));
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
    } else if (input.dtype() == DType::Float64) {
        neg_kernel_device<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Int32) {
        neg_kernel_device<<<grid, block, 0, stream>>>(input.data<int32_t>(), result.data<int32_t>(), n);
    } else if (input.dtype() == DType::Int64) {
        neg_kernel_device<<<grid, block, 0, stream>>>(input.data<int64_t>(), result.data<int64_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        neg_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        neg_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
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
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        abs_kernel_f32<<<grid, block, 0, stream>>>(input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        abs_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Int32) {
        abs_kernel_device<<<grid, block, 0, stream>>>(input.data<int32_t>(), result.data<int32_t>(), n);
    } else if (input.dtype() == DType::Int64) {
        abs_kernel_device<<<grid, block, 0, stream>>>(input.data<int64_t>(), result.data<int64_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        abs_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        abs_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
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
    } else if (input.dtype() == DType::Float64) {
        sqrt_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        sqrt_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        sqrt_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("sqrt operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::Float64) {
        exp_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        exp_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        exp_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("exp operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::Float64) {
        log_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        log_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        log_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("log operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::Float64) {
        double exp_d = static_cast<double>(exponent);
        pow_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), exp_d, n);
    } else if (input.dtype() == DType::Float16) {
        __half exp_h = __float2half(exponent);
        pow_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), exp_h, n);
    } else if (input.dtype() == DType::BFloat16) {
        __nv_bfloat16 exp_bf = __float2bfloat16(exponent);
        pow_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), exp_bf, n);
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
    } else if (input.dtype() == DType::Float64) {
        double min_d = static_cast<double>(min_val);
        double max_d = static_cast<double>(max_val);
        clamp_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), min_d, max_d, n);
    } else if (input.dtype() == DType::Float16) {
        __half min_h = __float2half(min_val);
        __half max_h = __float2half(max_val);
        clamp_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), min_h, max_h, n);
    } else if (input.dtype() == DType::BFloat16) {
        __nv_bfloat16 min_bf = __float2bfloat16(min_val);
        __nv_bfloat16 max_bf = __float2bfloat16(max_val);
        clamp_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), min_bf, max_bf, n);
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
    } else if (input.dtype() == DType::Float64) {
        sign_kernel_f64<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        sign_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        sign_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), n);
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
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = sin(input[idx]);
    }
}

template<typename T>
__global__ void cos_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = cos(input[idx]);
    }
}

template<typename T>
__global__ void tan_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = tan(input[idx]);
    }
}

template<typename T>
__global__ void asin_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = asin(input[idx]);
    }
}

template<typename T>
__global__ void acos_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = acos(input[idx]);
    }
}

template<typename T>
__global__ void atan_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = atan(input[idx]);
    }
}

template<typename T>
__global__ void sinh_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = sinh(input[idx]);
    }
}

template<typename T>
__global__ void cosh_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = cosh(input[idx]);
    }
}

// Rounding functions
template<typename T>
__global__ void ceil_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = ceil(input[idx]);
    }
}

template<typename T>
__global__ void floor_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = floor(input[idx]);
    }
}

template<typename T>
__global__ void round_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = round(input[idx]);
    }
}

template<typename T>
__global__ void trunc_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = trunc(input[idx]);
    }
}

template<typename T>
__global__ void reciprocal_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = T(1.0) / input[idx];
    }
}

// Launcher functions for trigonometric operations
#define DEFINE_TRIG_KERNEL(name) \
auto name##_kernel(const Tensor& input, cudaStream_t stream) -> Tensor { \
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
        throw std::runtime_error(#name " operation only supports Float32 and Float64 dtypes"); \
    } \
    CUDA_CHECK(cudaGetLastError()); \
    return result; \
}

DEFINE_TRIG_KERNEL(sin)
DEFINE_TRIG_KERNEL(cos)
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
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx] < min_val ? min_val : input[idx];
    }
}

template<typename T>
__global__ void clamp_max_kernel_impl(const T* input, T* output, T max_val, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx] > max_val ? max_val : input[idx];
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
    } else if (input.dtype() == DType::Float64) {
        clamp_min_kernel_impl<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), static_cast<double>(min_val), n);
    } else {
        throw std::runtime_error("clamp_min operation only supports Float32 and Float64 dtypes");
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
    } else if (input.dtype() == DType::Float64) {
        clamp_max_kernel_impl<<<grid, block, 0, stream>>>(input.data<double>(), result.data<double>(), static_cast<double>(max_val), n);
    } else {
        throw std::runtime_error("clamp_max operation only supports Float32 and Float64 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// In-place operations
template<typename T>
__global__ void add_inplace_kernel_impl(T* data, const T* other, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        data[idx] += other[idx];
    }
}

template<typename T>
__global__ void sub_inplace_kernel_impl(T* data, const T* other, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        data[idx] -= other[idx];
    }
}

template<typename T>
__global__ void mul_inplace_kernel_impl(T* data, const T* other, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        data[idx] *= other[idx];
    }
}

template<typename T>
__global__ void div_inplace_kernel_impl(T* data, const T* other, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        data[idx] /= other[idx];
    }
}

// Float16 in-place kernels
__global__ void add_inplace_kernel_f16(__half* data, const __half* other, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = __hadd(data[idx], other[idx]);
    }
}

__global__ void sub_inplace_kernel_f16(__half* data, const __half* other, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = __hsub(data[idx], other[idx]);
    }
}

__global__ void mul_inplace_kernel_f16(__half* data, const __half* other, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = __hmul(data[idx], other[idx]);
    }
}

__global__ void div_inplace_kernel_f16(__half* data, const __half* other, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        data[idx] = __hdiv(data[idx], other[idx]);
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
        } else if (inout.dtype() == DType::Float64) {
            add_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<double>(), other.data<double>(), n);
        } else if (inout.dtype() == DType::Float16) {
            add_inplace_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()), n);
        } else {
            throw std::runtime_error("add_inplace operation only supports Float32, Float64, and Float16 dtypes");
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

        // Copy shapes/strides to device
        int64_t ndim = inout_shape_vec.size();
        int64_t* d_strides_b = nullptr;
        int64_t* d_output_shape = nullptr;
        CUDA_CHECK(cudaMalloc(&d_strides_b, ndim * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_output_shape, ndim * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_strides_b, strides_b.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_output_shape, inout_shape_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        if (inout.dtype() == DType::Float32) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<float>(), other.data<float>(),
                d_strides_b, d_output_shape, ndim, n, AddOp{});
        } else if (inout.dtype() == DType::Float64) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<double>(), other.data<double>(),
                d_strides_b, d_output_shape, ndim, n, AddOp{});
        } else if (inout.dtype() == DType::Float16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()),
                d_strides_b, d_output_shape, ndim, n, AddOp{});
        } else {
            CUDA_CHECK(cudaFree(d_strides_b));
            CUDA_CHECK(cudaFree(d_output_shape));
            throw std::runtime_error("add_inplace operation only supports Float32, Float64, and Float16 dtypes");
        }

        CUDA_CHECK(cudaFree(d_strides_b));
        CUDA_CHECK(cudaFree(d_output_shape));
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
        } else if (inout.dtype() == DType::Float64) {
            sub_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<double>(), other.data<double>(), n);
        } else if (inout.dtype() == DType::Float16) {
            sub_inplace_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()), n);
        } else {
            throw std::runtime_error("sub_inplace operation only supports Float32, Float64, and Float16 dtypes");
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
        int64_t* d_strides_b = nullptr;
        int64_t* d_output_shape = nullptr;
        CUDA_CHECK(cudaMalloc(&d_strides_b, ndim * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_output_shape, ndim * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_strides_b, strides_b.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_output_shape, inout_shape_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        if (inout.dtype() == DType::Float32) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<float>(), other.data<float>(),
                d_strides_b, d_output_shape, ndim, n, SubOp{});
        } else if (inout.dtype() == DType::Float64) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<double>(), other.data<double>(),
                d_strides_b, d_output_shape, ndim, n, SubOp{});
        } else if (inout.dtype() == DType::Float16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()),
                d_strides_b, d_output_shape, ndim, n, SubOp{});
        } else {
            CUDA_CHECK(cudaFree(d_strides_b));
            CUDA_CHECK(cudaFree(d_output_shape));
            throw std::runtime_error("sub_inplace operation only supports Float32, Float64, and Float16 dtypes");
        }

        CUDA_CHECK(cudaFree(d_strides_b));
        CUDA_CHECK(cudaFree(d_output_shape));
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
        } else if (inout.dtype() == DType::Float64) {
            mul_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<double>(), other.data<double>(), n);
        } else if (inout.dtype() == DType::Float16) {
            mul_inplace_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()), n);
        } else {
            throw std::runtime_error("mul_inplace operation only supports Float32, Float64, and Float16 dtypes");
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
        int64_t* d_strides_b = nullptr;
        int64_t* d_output_shape = nullptr;
        CUDA_CHECK(cudaMalloc(&d_strides_b, ndim * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_output_shape, ndim * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_strides_b, strides_b.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_output_shape, inout_shape_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        if (inout.dtype() == DType::Float32) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<float>(), other.data<float>(),
                d_strides_b, d_output_shape, ndim, n, MulOp{});
        } else if (inout.dtype() == DType::Float64) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<double>(), other.data<double>(),
                d_strides_b, d_output_shape, ndim, n, MulOp{});
        } else if (inout.dtype() == DType::Float16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()),
                d_strides_b, d_output_shape, ndim, n, MulOp{});
        } else {
            CUDA_CHECK(cudaFree(d_strides_b));
            CUDA_CHECK(cudaFree(d_output_shape));
            throw std::runtime_error("mul_inplace operation only supports Float32, Float64, and Float16 dtypes");
        }

        CUDA_CHECK(cudaFree(d_strides_b));
        CUDA_CHECK(cudaFree(d_output_shape));
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
        } else if (inout.dtype() == DType::Float64) {
            div_inplace_kernel_impl<<<grid, block, 0, stream>>>(inout.data<double>(), other.data<double>(), n);
        } else if (inout.dtype() == DType::Float16) {
            div_inplace_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()), n);
        } else {
            throw std::runtime_error("div_inplace operation only supports Float32, Float64, and Float16 dtypes");
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
        int64_t* d_strides_b = nullptr;
        int64_t* d_output_shape = nullptr;
        CUDA_CHECK(cudaMalloc(&d_strides_b, ndim * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_output_shape, ndim * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_strides_b, strides_b.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_output_shape, inout_shape_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        if (inout.dtype() == DType::Float32) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<float>(), other.data<float>(),
                d_strides_b, d_output_shape, ndim, n, DivOp{});
        } else if (inout.dtype() == DType::Float64) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                inout.data<double>(), other.data<double>(),
                d_strides_b, d_output_shape, ndim, n, DivOp{});
        } else if (inout.dtype() == DType::Float16) {
            broadcast_inplace_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(inout.data<Float16>()),
                reinterpret_cast<const __half*>(other.data<Float16>()),
                d_strides_b, d_output_shape, ndim, n, DivOp{});
        } else {
            CUDA_CHECK(cudaFree(d_strides_b));
            CUDA_CHECK(cudaFree(d_output_shape));
            throw std::runtime_error("div_inplace operation only supports Float32, Float64, and Float16 dtypes");
        }

        CUDA_CHECK(cudaFree(d_strides_b));
        CUDA_CHECK(cudaFree(d_output_shape));
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return inout;
}

// Expand kernel - replicate tensor along specified dimensions
template<typename T>
__global__ void expand_kernel_device(
    const T* input, T* output,
    const int64_t* input_shape, const int64_t* input_strides,
    const int64_t* output_shape, int64_t input_ndim, int64_t output_ndim, int64_t n) {

    CUDA_KERNEL_LOOP(out_idx, n) {
        int64_t temp = out_idx;
        int64_t in_idx = 0;

        // Dimension offset (output can have more dims than input due to leading 1s)
        int64_t input_dim_offset = output_ndim - input_ndim;

        // Convert output flat index to multi-dimensional coordinates
        for (int64_t i = output_ndim - 1; i >= 0; --i) {
            int64_t coord = temp % output_shape[i];
            temp /= output_shape[i];

            int64_t input_dim = i - input_dim_offset;
            if (input_dim >= 0 && input_dim < input_ndim) {
                // For dimensions of size 1, we don't advance the index (broadcast)
                if (input_shape[input_dim] != 1) {
                    in_idx += coord * input_strides[input_dim];
                }
                // If input_shape[input_dim] == 1, stride is effectively 0 (broadcast)
            }
        }

        output[out_idx] = input[in_idx];
    }
}

// Expand kernel launcher
auto expand_kernel(const Tensor& input, const std::vector<int64_t>& shape, void* stream_ptr) -> Tensor {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
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
            throw std::invalid_argument(
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

    // Copy metadata to device
    int64_t* d_input_shape;
    int64_t* d_input_strides;
    int64_t* d_output_shape;
    CUDA_CHECK(cudaMalloc(&d_input_shape, input_shape_vec.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_input_strides, input_strides.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_shape, shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_input_shape, input_shape_vec.data(), input_shape_vec.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_strides, input_strides.data(), input_strides.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_shape, shape.data(), shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t n = result.numel();
    int64_t input_ndim = input_shape_vec.size();
    int64_t output_ndim = shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<float>(), result.data<float>(),
            d_input_shape, d_input_strides, d_output_shape,
            input_ndim, output_ndim, n);
    } else if (input.dtype() == DType::Float64) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<double>(), result.data<double>(),
            d_input_shape, d_input_strides, d_output_shape,
            input_ndim, output_ndim, n);
    } else if (input.dtype() == DType::Int32) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<int32_t>(), result.data<int32_t>(),
            d_input_shape, d_input_strides, d_output_shape,
            input_ndim, output_ndim, n);
    } else if (input.dtype() == DType::Int64) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            input.data<int64_t>(), result.data<int64_t>(),
            d_input_shape, d_input_strides, d_output_shape,
            input_ndim, output_ndim, n);
    } else if (input.dtype() == DType::Float16) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_input_shape, d_input_strides, d_output_shape,
            input_ndim, output_ndim, n);
    } else if (input.dtype() == DType::BFloat16) {
        expand_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            d_input_shape, d_input_strides, d_output_shape,
            input_ndim, output_ndim, n);
    } else {
        throw std::runtime_error("Unsupported dtype for expand operation");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input_shape));
    CUDA_CHECK(cudaFree(d_input_strides));
    CUDA_CHECK(cudaFree(d_output_shape));
    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
}

// ============================================================================
// Fill Operations (zeros, ones, full)
// ============================================================================

// Fill kernel - set all elements to a constant value
template<typename T>
__global__ void fill_kernel_device(T* output, T value, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = value;
    }
}

// Fill kernel launcher - fills tensor with constant value
auto fill_kernel(const Tensor& tensor, float value, cudaStream_t stream) -> Tensor {
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
    } else if (tensor.dtype() == DType::Float64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<double>(), static_cast<double>(value), n);
    } else if (tensor.dtype() == DType::Int32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int32_t>(), static_cast<int32_t>(value), n);
    } else if (tensor.dtype() == DType::Int64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int64_t>(), static_cast<int64_t>(value), n);
    } else if (tensor.dtype() == DType::Float16) {
        __half h_value = __float2half(value);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), h_value, n);
    } else if (tensor.dtype() == DType::BFloat16) {
        __nv_bfloat16 bf_value = __float2bfloat16(value);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), bf_value, n);
    } else {
        throw std::runtime_error("Unsupported dtype for fill operation");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaStreamSynchronize - async execution for performance
    return result;
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
    } else if (dtype == DType::Float64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<double>(), 0.0, n);
    } else if (dtype == DType::Int32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int32_t>(), static_cast<int32_t>(0), n);
    } else if (dtype == DType::Int64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int64_t>(), static_cast<int64_t>(0), n);
    } else if (dtype == DType::Float16) {
        __half zero_h = __float2half(0.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), zero_h, n);
    } else if (dtype == DType::BFloat16) {
        __nv_bfloat16 zero_bf = __float2bfloat16(0.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), zero_bf, n);
    } else if (dtype == DType::Bool) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<bool>(), false, n);
    } else if (dtype == DType::Int8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int8_t>(), static_cast<int8_t>(0), n);
    } else if (dtype == DType::UInt8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint8_t>(), static_cast<uint8_t>(0), n);
    } else if (dtype == DType::Int16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int16_t>(), static_cast<int16_t>(0), n);
    } else if (dtype == DType::UInt16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint16_t>(), static_cast<uint16_t>(0), n);
    } else if (dtype == DType::UInt32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint32_t>(), static_cast<uint32_t>(0), n);
    } else if (dtype == DType::UInt64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint64_t>(), static_cast<uint64_t>(0), n);
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
    } else if (dtype == DType::Float64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<double>(), 1.0, n);
    } else if (dtype == DType::Int32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int32_t>(), static_cast<int32_t>(1), n);
    } else if (dtype == DType::Int64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int64_t>(), static_cast<int64_t>(1), n);
    } else if (dtype == DType::Float16) {
        __half one_h = __float2half(1.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), one_h, n);
    } else if (dtype == DType::BFloat16) {
        __nv_bfloat16 one_bf = __float2bfloat16(1.0f);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), one_bf, n);
    } else if (dtype == DType::Bool) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<bool>(), true, n);
    } else if (dtype == DType::Int8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int8_t>(), static_cast<int8_t>(1), n);
    } else if (dtype == DType::UInt8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint8_t>(), static_cast<uint8_t>(1), n);
    } else if (dtype == DType::Int16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int16_t>(), static_cast<int16_t>(1), n);
    } else if (dtype == DType::UInt16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint16_t>(), static_cast<uint16_t>(1), n);
    } else if (dtype == DType::UInt32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint32_t>(), static_cast<uint32_t>(1), n);
    } else if (dtype == DType::UInt64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint64_t>(), static_cast<uint64_t>(1), n);
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
    } else if (dtype == DType::Float64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<double>(), static_cast<double>(value), n);
    } else if (dtype == DType::Int32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int32_t>(), static_cast<int32_t>(value), n);
    } else if (dtype == DType::Int64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int64_t>(), static_cast<int64_t>(value), n);
    } else if (dtype == DType::Float16) {
        __half h_value = __float2half(value);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), h_value, n);
    } else if (dtype == DType::BFloat16) {
        __nv_bfloat16 bf_value = __float2bfloat16(value);
        fill_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), bf_value, n);
    } else if (dtype == DType::Int8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int8_t>(), static_cast<int8_t>(value), n);
    } else if (dtype == DType::UInt8) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint8_t>(), static_cast<uint8_t>(value), n);
    } else if (dtype == DType::Int16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<int16_t>(), static_cast<int16_t>(value), n);
    } else if (dtype == DType::UInt16) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint16_t>(), static_cast<uint16_t>(value), n);
    } else if (dtype == DType::UInt32) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint32_t>(), static_cast<uint32_t>(value), n);
    } else if (dtype == DType::UInt64) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<uint64_t>(), static_cast<uint64_t>(value), n);
    } else if (dtype == DType::Bool) {
        fill_kernel_device<<<grid, block, 0, stream>>>(
            result.data<bool>(), static_cast<bool>(value), n);
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
    CUDA_KERNEL_LOOP(idx, n) {
        // Each thread gets different seed, a different sequence number, no offset
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Kernel for uniform random [0, 1) generation
__global__ void rand_kernel_device(float* output, curandState* states, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = curand_uniform(&states[idx]);
    }
}

// Kernel for normal distribution N(0,1) generation
__global__ void randn_kernel_device(float* output, curandState* states, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = curand_normal(&states[idx]);
    }
}

// FP16 uniform random kernel
__global__ void rand_kernel_f16(__half* output, curandState* states, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        float val = curand_uniform(&states[idx]);
        output[idx] = __float2half(val);
    }
}

// FP16 normal distribution kernel
__global__ void randn_kernel_f16(__half* output, curandState* states, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        float val = curand_normal(&states[idx]);
        output[idx] = __float2half(val);
    }
}

// BFloat16 uniform random kernel
__global__ void rand_kernel_bf16(__nv_bfloat16* output, curandState* states, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        float val = curand_uniform(&states[idx]);
        output[idx] = __float2bfloat16(val);
    }
}

// BFloat16 normal distribution kernel
__global__ void randn_kernel_bf16(__nv_bfloat16* output, curandState* states, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        float val = curand_normal(&states[idx]);
        output[idx] = __float2bfloat16(val);
    }
}

// Float-to-double conversion kernel for proper type conversion
__global__ void convert_float_to_double_kernel(const float* input, double* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
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
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));

    // Thread-safe seed generation with better entropy
    // Mix: high-res time + thread ID + random_device + atomicincrement counter
    static std::atomic<uint64_t> seed_counter{0};
    static std::random_device rd;
    auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto random_bits = rd();
    auto counter = seed_counter.fetch_add(1, std::memory_order_relaxed);

    // Mix all entropy sources with XOR and rotation
    uint64_t seed = time_seed ^ (thread_id << 32) ^ (random_bits << 16) ^ counter;

    init_curand_states<<<grid, block, 0, stream>>>(d_states, seed, n);
    CUDA_CHECK(cudaGetLastError());

    if (dtype == DType::Float32) {
        // Generate uniform random numbers
        rand_kernel_device<<<grid, block, 0, stream>>>(result.data<float>(), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        // For Float64, generate as float then convert properly
        float* temp_float;
        CUDA_CHECK(cudaMalloc(&temp_float, n * sizeof(float)));
        rand_kernel_device<<<grid, block, 0, stream>>>(temp_float, d_states, n);
        CUDA_CHECK(cudaGetLastError());

        // Convert float to double using proper conversion kernel
        double* output_double = result.data<double>();
        convert_float_to_double_kernel<<<grid, block, 0, stream>>>(temp_float, output_double, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(temp_float));
    } else if (dtype == DType::Float16) {
        rand_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::BFloat16) {
        rand_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_states));

    return result;
}

// Randn kernel launcher - normal distribution N(0,1)
auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    //printf("[DEBUG randn_kernel] Entry - dtype=%d, device type=%d\n", static_cast<int>(dtype), static_cast<int>(device.type));

    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("randn operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    //printf("[DEBUG randn_kernel] Creating tensor...\n");
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();
    //printf("[DEBUG randn_kernel] Tensor created, n=%lld\n", (long long)n);

    if (n == 0) {
        return result;
    }

    //printf("[DEBUG randn_kernel] Computing launch config...\n");
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    //printf("[DEBUG randn_kernel] Launch config done: grid=(%u,%u,%u), block=(%u,%u,%u)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    // Allocate cuRAND states
    //printf("[DEBUG randn_kernel] Allocating cuRAND states...\n");
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));
    //printf("[DEBUG randn_kernel] cuRAND states allocated\n");

    // Thread-safe seed generation with better entropy
    // Mix: high-res time + thread ID + random_device + atomic counter
    //printf("[DEBUG randn_kernel] Generating seed...\n");
    static std::atomic<uint64_t> seed_counter{0};
    static std::random_device rd;
    //printf("[DEBUG randn_kernel] Getting time seed...\n");
    auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    //printf("[DEBUG randn_kernel] Getting thread ID...\n");
    auto thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    //printf("[DEBUG randn_kernel] Calling random_device...\n");
    auto random_bits = rd();
    //printf("[DEBUG randn_kernel] Getting counter...\n");
    auto counter = seed_counter.fetch_add(1, std::memory_order_relaxed);
    //printf("[DEBUG randn_kernel] Seed generation complete\n");

    // Mix all entropy sources with XOR and rotation
    uint64_t seed = time_seed ^ (thread_id << 32) ^ (random_bits << 16) ^ counter;
    //printf("[DEBUG randn_kernel] Final seed=%llu\n", (unsigned long long)seed);

    //printf("[DEBUG randn_kernel] Initializing cuRAND states...\n");
    init_curand_states<<<grid, block, 0, stream>>>(d_states, seed, n);
    CUDA_CHECK(cudaGetLastError());
    //printf("[DEBUG randn_kernel] cuRAND states initialized\n");

    if (dtype == DType::Float32) {
        //printf("[DEBUG randn_kernel] Generating Float32 random numbers...\n");
        // Generate normal random numbers
        randn_kernel_device<<<grid, block, 0, stream>>>(result.data<float>(), d_states, n);
        CUDA_CHECK(cudaGetLastError());
        //printf("[DEBUG randn_kernel] Float32 generation complete\n");
    } else if (dtype == DType::Float64) {
        //printf("[DEBUG randn_kernel] Generating Float64 random numbers...\n");
        // For Float64, generate as float then convert properly
        float* temp_float;
        CUDA_CHECK(cudaMalloc(&temp_float, n * sizeof(float)));
        randn_kernel_device<<<grid, block, 0, stream>>>(temp_float, d_states, n);
        CUDA_CHECK(cudaGetLastError());

        // Convert float to double using proper conversion kernel
        double* output_double = result.data<double>();
        convert_float_to_double_kernel<<<grid, block, 0, stream>>>(temp_float, output_double, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(temp_float));
        //printf("[DEBUG randn_kernel] Float64 generation complete\n");
    } else if (dtype == DType::Float16) {
        //printf("[DEBUG randn_kernel] Generating Float16 random numbers...\n");
        randn_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(result.data<Float16>()), d_states, n);
        CUDA_CHECK(cudaGetLastError());
        //printf("[DEBUG randn_kernel] Float16 generation complete\n");
    } else if (dtype == DType::BFloat16) {
        //printf("[DEBUG randn_kernel] Generating BFloat16 random numbers...\n");
        randn_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), d_states, n);
        CUDA_CHECK(cudaGetLastError());
        //printf("[DEBUG randn_kernel] BFloat16 generation complete\n");
    }

    // Cleanup
    //printf("[DEBUG randn_kernel] Cleaning up...\n");
    CUDA_CHECK(cudaFree(d_states));
    //printf("[DEBUG randn_kernel] Cleanup complete, returning result\n");

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
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = op(a[idx], b[idx]);
    }
}

// Generic broadcast comparison kernel
template<typename T, typename Op>
__global__ void broadcast_compare_kernel(
    const T* a, const T* b, bool* c,
    const int64_t* strides_a, const int64_t* strides_b,
    const int64_t* output_shape, int64_t ndim, int64_t n, Op op) {

    CUDA_KERNEL_LOOP(out_idx, n) {
        int64_t idx_a = 0;
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        // Convert flat index to multi-dimensional indices
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % output_shape[i];
            tmp /= output_shape[i];
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
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
        } else if (a.dtype() == DType::Float64) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<double>(), b.data<double>(), result.data<bool>(), n, op);
        } else if (a.dtype() == DType::Int32) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int32_t>(), b.data<int32_t>(), result.data<bool>(), n, op);
        } else if (a.dtype() == DType::Int64) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<int64_t>(), b.data<int64_t>(), result.data<bool>(), n, op);
        } else if (a.dtype() == DType::Float16) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a.data_ptr()),
                reinterpret_cast<const __half*>(b.data_ptr()),
                result.data<bool>(), n, op);
        } else if (a.dtype() == DType::Bool) {
            compare_kernel_device<<<grid, block, 0, stream>>>(
                a.data<bool>(), b.data<bool>(), result.data<bool>(), n, op);
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

    // Copy strides to device
    int64_t* d_strides_a;
    int64_t* d_strides_b;
    int64_t* d_output_shape;
    CUDA_CHECK(cudaMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<bool>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, op);
    } else if (a.dtype() == DType::Float64) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<double>(), b.data<double>(), result.data<bool>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, op);
    } else if (a.dtype() == DType::Int32) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<bool>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, op);
    } else if (a.dtype() == DType::Int64) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<bool>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, op);
    } else if (a.dtype() == DType::Float16) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data_ptr()),
            reinterpret_cast<const __half*>(b.data_ptr()),
            result.data<bool>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, op);
    } else if (a.dtype() == DType::Bool) {
        broadcast_compare_kernel<<<grid, block, 0, stream>>>(
            a.data<bool>(), b.data<bool>(), result.data<bool>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, op);
    } else {
        throw std::runtime_error("Unsupported dtype for comparison operation");
    }

    // Cleanup - note: cudaFree is synchronous, so no explicit sync needed after it
    CUDA_CHECK(cudaFree(d_strides_a));
    CUDA_CHECK(cudaFree(d_strides_b));
    CUDA_CHECK(cudaFree(d_output_shape));
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
            } else {
                // Two-phase reduction for large arrays
                float* d_temp;
                cudaMalloc(&d_temp, num_blocks * sizeof(float));
                dot_product_kernel<<<num_blocks, block_size, 0, stream>>>(a_data, b_data, d_temp, n);
                dot_product_kernel<<<1, block_size, 0, stream>>>(d_temp, d_temp, output_data, num_blocks);
                cudaFree(d_temp);
            }
            break;
        }
        case DType::Float64: {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* output_data = output.data<double>();

            if (num_blocks == 1) {
                dot_product_kernel<<<1, block_size, 0, stream>>>(a_data, b_data, output_data, n);
            } else {
                // Two-phase reduction for large arrays
                double* d_temp;
                cudaMalloc(&d_temp, num_blocks * sizeof(double));
                dot_product_kernel<<<num_blocks, block_size, 0, stream>>>(a_data, b_data, d_temp, n);
                dot_product_kernel<<<1, block_size, 0, stream>>>(d_temp, d_temp, output_data, num_blocks);
                cudaFree(d_temp);
            }
            break;
        }
        default:
            throw std::runtime_error("dot: only Float32 and Float64 are supported");
    }

    // NOTE: Removed cudaStreamSynchronize - cudaFree in the switch cases is already synchronous

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
    } else if (input.dtype() == DType::Float64) {
        adaptive_avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            N, C, H_in, W_in, output_h, output_w);
    } else if (input.dtype() == DType::Float16) {
        adaptive_avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            N, C, H_in, W_in, output_h, output_w);
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
    } else if (grad_output.dtype() == DType::Float64) {
        adaptive_avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, H_in, W_in, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float16) {
        adaptive_avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<__half*>(grad_input.data_ptr()),
            N, C, H_in, W_in, H_out, W_out);
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
    } else if (input.dtype() == DType::Float64) {
        adaptive_max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, H_in, W_in, output_h, output_w);
    } else if (input.dtype() == DType::Float16) {
        adaptive_max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            indices.data<int64_t>(),
            N, C, H_in, W_in, output_h, output_w);
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
    } else if (grad_output.dtype() == DType::Float64) {
        adaptive_max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(), grad_input.data<double>(), total);
    } else if (grad_output.dtype() == DType::Float16) {
        adaptive_max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(grad_input.data_ptr()), total);
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
    } else if (input.dtype() == DType::Float64) {
        max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float16) {
        max_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            indices.data<int64_t>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
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
    } else if (grad_output.dtype() == DType::Float64) {
        max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(), grad_input.data<double>(),
            total_output, total_input);
    } else if (grad_output.dtype() == DType::Float16) {
        max_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(grad_input.data_ptr()),
            total_output, total_input);
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
    } else if (input.dtype() == DType::Float64) {
        avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float16) {
        avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
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
    } else if (grad_output.dtype() == DType::Float64) {
        avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
    } else if (grad_output.dtype() == DType::Float16) {
        avg_pool2d_backward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data_ptr()),
            reinterpret_cast<__half*>(grad_input.data_ptr()),
            N, C, H_in, W_in, H_out, W_out, kernel_size, stride, padding);
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
    } else if (table.dtype() == DType::Float64) {
        gather_2d_kernel<<<grid, block, 0, stream>>>(
            table.data<double>(), indices_device.data<int64_t>(), output.data<double>(),
            num_positions, num_heads, num_heads);
    } else if (table.dtype() == DType::Float16) {
        gather_2d_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(table.data_ptr()),
            indices_device.data<int64_t>(),
            reinterpret_cast<__half*>(output.data_ptr()),
            num_positions, num_heads, num_heads);
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

    // Reshape img_mask to window format
    // This is a simplified version - we need to properly partition windows
    // For now, create a simple partitioned version
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

    // Create attention mask: mask[i, j] = -100 if window_mask[i] != window_mask[j]
    Tensor attn_mask({num_windows, M, M}, DType::Float32, cuda_device);
    float* attn_data = attn_mask.data<float>();

    int mask_threads = 256;
    int mask_blocks = (num_windows * M * M + mask_threads - 1) / mask_threads;
    create_attention_mask_kernel<<<mask_blocks, mask_threads>>>(
        window_data, attn_data, num_windows, M);


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

auto arange_kernel(float start, float end, float step, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    int64_t n = static_cast<int64_t>(std::ceil((end - start) / step));
    if (n <= 0) n = 0;
    Tensor output({n}, dtype, device);
    if (n == 0) return output;

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    if (dtype == DType::Float32) {
        arange_kernel_impl<float><<<num_blocks, block_size, 0, stream>>>(output.data<float>(), start, step, n);
    } else if (dtype == DType::Float64) {
        arange_kernel_impl<double><<<num_blocks, block_size, 0, stream>>>(output.data<double>(), static_cast<double>(start), static_cast<double>(step), n);
    } else if (dtype == DType::Int32) {
        arange_kernel_impl<int32_t><<<num_blocks, block_size, 0, stream>>>(output.data<int32_t>(), static_cast<int32_t>(start), static_cast<int32_t>(step), n);
    } else if (dtype == DType::Int64) {
        arange_kernel_impl<int64_t><<<num_blocks, block_size, 0, stream>>>(output.data<int64_t>(), static_cast<int64_t>(start), static_cast<int64_t>(step), n);
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

auto linspace_kernel(float start, float end, int64_t steps, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    if (steps <= 0) return Tensor({0}, dtype, device);
    Tensor output({steps}, dtype, device);

    if (steps == 1) {
        // Single element: just the start value
        if (dtype == DType::Float32) {
            cudaMemcpyAsync(output.data<float>(), &start, sizeof(float), cudaMemcpyHostToDevice, stream);
        } else if (dtype == DType::Float64) {
            double start_d = static_cast<double>(start);
            cudaMemcpyAsync(output.data<double>(), &start_d, sizeof(double), cudaMemcpyHostToDevice, stream);
        }
        return output;
    }

    int block_size = 256;
    int num_blocks = (steps + block_size - 1) / block_size;

    if (dtype == DType::Float32) {
        float step_val = (end - start) / static_cast<float>(steps - 1);
        linspace_kernel_impl<float><<<num_blocks, block_size, 0, stream>>>(output.data<float>(), start, step_val, steps);
    } else if (dtype == DType::Float64) {
        double step_val = (static_cast<double>(end) - static_cast<double>(start)) / static_cast<double>(steps - 1);
        linspace_kernel_impl<double><<<num_blocks, block_size, 0, stream>>>(output.data<double>(), static_cast<double>(start), step_val, steps);
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

auto eye_kernel(int64_t n, int64_t m, DType dtype, Device device, cudaStream_t stream) -> Tensor {
    if (m <= 0) m = n;
    Tensor output({n, m}, dtype, device);
    int64_t total = n * m;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    if (dtype == DType::Float32) {
        eye_kernel_impl<float><<<num_blocks, block_size, 0, stream>>>(output.data<float>(), n, m);
    } else if (dtype == DType::Float64) {
        eye_kernel_impl<double><<<num_blocks, block_size, 0, stream>>>(output.data<double>(), n, m);
    } else if (dtype == DType::Int32) {
        eye_kernel_impl<int32_t><<<num_blocks, block_size, 0, stream>>>(output.data<int32_t>(), n, m);
    } else if (dtype == DType::Int64) {
        eye_kernel_impl<int64_t><<<num_blocks, block_size, 0, stream>>>(output.data<int64_t>(), n, m);
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

Tensor add_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return add_kernel(inputs[0], inputs[1], stream);
}

Tensor sub_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return sub_kernel(inputs[0], inputs[1], stream);
}

Tensor mul_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return mul_kernel(inputs[0], inputs[1], stream);
}

Tensor div_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return div_kernel(inputs[0], inputs[1], stream);
}

// Note: matmul_dispatch and dot_dispatch are defined in cublas_ops.cu
// since matmul_kernel and dot_kernel are implemented there

// Inplace dispatch wrappers (InplaceKernelFn signature)
Tensor& add_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    add_inplace_kernel(target, others[0], stream);
    return target;
}

Tensor& sub_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    sub_inplace_kernel(target, others[0], stream);
    return target;
}

Tensor& mul_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    mul_inplace_kernel(target, others[0], stream);
    return target;
}

Tensor& div_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    div_inplace_kernel(target, others[0], stream);
    return target;
}

// Unary operation dispatch wrappers
Tensor sqrt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return sqrt_kernel(inputs[0], stream);
}

Tensor neg_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return neg_kernel(inputs[0], stream);
}

Tensor abs_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return abs_kernel(inputs[0], stream);
}

Tensor sign_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return sign_kernel(inputs[0], stream);
}

Tensor log_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return log_kernel(inputs[0], stream);
}

Tensor exp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return exp_kernel(inputs[0], stream);
}

Tensor reciprocal_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return reciprocal_kernel(inputs[0], stream);
}

Tensor floor_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return floor_kernel(inputs[0], stream);
}

Tensor ceil_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return ceil_kernel(inputs[0], stream);
}

Tensor round_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return round_kernel(inputs[0], stream);
}

// Trigonometric dispatch wrappers
Tensor sin_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return sin_kernel(inputs[0], stream);
}

Tensor cos_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return cos_kernel(inputs[0], stream);
}

Tensor tan_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return tan_kernel(inputs[0], stream);
}

Tensor asin_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return asin_kernel(inputs[0], stream);
}

Tensor acos_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return acos_kernel(inputs[0], stream);
}

Tensor atan_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return atan_kernel(inputs[0], stream);
}

Tensor sinh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return sinh_kernel(inputs[0], stream);
}

Tensor cosh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return cosh_kernel(inputs[0], stream);
}

// Comparison dispatch wrappers
Tensor eq_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return eq_kernel(inputs[0], inputs[1], stream);
}

Tensor ne_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return ne_kernel(inputs[0], inputs[1], stream);
}

Tensor lt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return lt_kernel(inputs[0], inputs[1], stream);
}

Tensor le_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return le_kernel(inputs[0], inputs[1], stream);
}

Tensor gt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return gt_kernel(inputs[0], inputs[1], stream);
}

Tensor ge_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    cudaStream_t stream = nullptr;
    if (!attrs.empty()) {
        auto it = attrs.find("stream");
        if (it != attrs.end()) {
            uint64_t val = 0;
            std::from_chars(it->second.data(), it->second.data() + it->second.size(), val);
            stream = reinterpret_cast<cudaStream_t>(val);
        }
    }
    return ge_kernel(inputs[0], inputs[1], stream);
}

} // namespace cuda
} // namespace tenzor
