#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#ifdef TENZOR_HAS_HIPRAND
#include <hiprand_kernel.h>
#endif
#include <hipcub/hipcub.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>
#include "fp16_saturate.h"

namespace tenzor {
namespace rocm {

// ============================================================================
// HIP Error Checking
// ============================================================================

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

// Compute optimal grid/block dimensions for 1D kernels
// Optimized for AMD GPU wavefront size of 64
inline void compute_launch_config_1d(int64_t n, dim3& grid, dim3& block) {
    const int block_size = 256;  // Multiple of wavefront size (64) for optimal performance
    block = dim3(block_size, 1, 1);
    grid = dim3((n + block_size - 1) / block_size, 1, 1);
}

// Grid-stride loop pattern for better scalability across AMD GPU architectures
#define HIP_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// ============================================================================
// Broadcasting Helpers (Device-side)
// ============================================================================

/**
 * @brief Device function to check if shapes are broadcastable
 * @param shape_a First shape array
 * @param ndim_a Number of dimensions in first shape
 * @param shape_b Second shape array
 * @param ndim_b Number of dimensions in second shape
 * @return true if shapes are broadcastable, false otherwise
 */
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

/**
 * @brief Check if two shapes are broadcastable (NumPy-style broadcasting)
 * @param shape_a First tensor shape
 * @param shape_b Second tensor shape
 * @return true if shapes can be broadcast together
 */
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

/**
 * @brief Compute the broadcasted output shape from two input shapes
 * @param shape_a First tensor shape
 * @param shape_b Second tensor shape
 * @return Broadcasted output shape
 * @throws std::runtime_error if shapes are not broadcastable
 */
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

/**
 * @brief Compute strides for broadcasting a tensor to a larger shape
 * @param shape Original tensor shape
 * @param broadcast_shape Target broadcast shape
 * @return Strides array (0 for broadcast dimensions, normal stride otherwise)
 */
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

    // Map to broadcast strides (stride 0 means broadcast dimension)
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

/**
 * @brief Check if tensors have identical shapes (enables fast path optimization)
 * @param a First tensor
 * @param b Second tensor
 * @return true if shapes are identical
 */
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

} // namespace detail

// ============================================================================
// Element-wise Binary Operations (with Broadcasting Support)
// ============================================================================

/**
 * @brief Fast path: element-wise addition kernel (same shape tensors)
 * @tparam T Data type (float, double, int32_t, int64_t)
 * @param a First input tensor data
 * @param b Second input tensor data
 * @param c Output tensor data
 * @param n Number of elements
 *
 * Optimized for AMD GPUs with wavefront-aware access patterns
 */
template<typename T>
__global__ void add_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief Generic broadcast kernel - works for all binary operations
 * @tparam T Data type
 * @tparam Op Operation functor (AddOp, SubOp, MulOp, DivOp)
 * @param a First input tensor data
 * @param b Second input tensor data
 * @param c Output tensor data
 * @param strides_a Broadcast strides for tensor a
 * @param strides_b Broadcast strides for tensor b
 * @param output_shape Shape of output tensor
 * @param ndim Number of dimensions
 * @param n Total number of output elements
 * @param op Binary operation to perform
 *
 * Handles NumPy-style broadcasting by converting flat output index
 * to multi-dimensional coordinates and mapping to input indices
 */
template<typename T, typename Op>
__global__ void broadcast_kernel(
    const T* a, const T* b, T* c,
    const int64_t* strides_a, const int64_t* strides_b,
    const int64_t* output_shape, int64_t ndim, int64_t n, Op op) {

    HIP_KERNEL_LOOP(out_idx, n) {
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

/**
 * @brief Addition operation functor
 */
struct AddOp {
    template<typename T>
    __device__ T operator()(T a, T b) const { return a + b; }
};

/**
 * @brief Subtraction operation functor
 */
struct SubOp {
    template<typename T>
    __device__ T operator()(T a, T b) const { return a - b; }
};

/**
 * @brief Multiplication operation functor
 */
struct MulOp {
    template<typename T>
    __device__ T operator()(T a, T b) const { return a * b; }
};

/**
 * @brief Division operation functor with zero-division handling
 */
struct DivOp {
    template<typename T>
    __device__ T operator()(T a, T b) const {
        if (b == T(0)) {
            return T(INFINITY);
        }
        return a / b;
    }
};

/**
 * @brief Element-wise subtraction kernel (same shape tensors)
 * @tparam T Data type
 */
template<typename T>
__global__ void sub_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        c[idx] = a[idx] - b[idx];
    }
}

/**
 * @brief Element-wise multiplication kernel (same shape tensors)
 * @tparam T Data type
 */
template<typename T>
__global__ void mul_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        c[idx] = a[idx] * b[idx];
    }
}

/**
 * @brief Element-wise division kernel with zero-division handling
 * @tparam T Data type
 */
template<typename T>
__global__ void div_kernel_device(const T* a, const T* b, T* c, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        T divisor = b[idx];
        if (divisor == T(0)) {
            c[idx] = INFINITY;  // Handle division by zero
        } else {
            c[idx] = a[idx] / divisor;
        }
    }
}

/**
 * @brief Element-wise division kernel for half precision
 * Uses float conversion for correct division and infinity handling
 */
__global__ void div_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float divisor = __half2float(b[idx]);
        if (divisor == 0.0f) {
            c[idx] = __float2half(INFINITY);
        } else {
            float result = __half2float(a[idx]) / divisor;
            c[idx] = __float2half(result);
        }
    }
}

/**
 * @brief Broadcasting division kernel specialized for Float16
 * Uses float conversion for correct division and infinity handling with broadcasting
 */
__global__ void broadcast_div_kernel_f16(
    const __half* a, const __half* b, __half* c,
    const int64_t* strides_a, const int64_t* strides_b,
    const int64_t* output_shape, int64_t ndim, int64_t n) {

    HIP_KERNEL_LOOP(out_idx, n) {
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

        float divisor = __half2float(b[idx_b]);
        if (divisor == 0.0f) {
            c[out_idx] = __float2half(INFINITY);
        } else {
            float result = __half2float(a[idx_a]) / divisor;
            c[out_idx] = __float2half(result);
        }
    }
}

/**
 * @brief Division kernel specialized for BFloat16
 * Uses float conversion for correct division and infinity handling
 */
__global__ void div_kernel_bf16(const hip_bfloat16* a, const hip_bfloat16* b, hip_bfloat16* c, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float divisor = static_cast<float>(b[idx]);
        if (divisor == 0.0f) {
            c[idx] = hip_bfloat16(INFINITY);
        } else {
            float result = static_cast<float>(a[idx]) / divisor;
            c[idx] = hip_bfloat16(result);
        }
    }
}

/**
 * @brief Broadcasting division kernel specialized for BFloat16
 */
__global__ void broadcast_div_kernel_bf16(
    const hip_bfloat16* a, const hip_bfloat16* b, hip_bfloat16* c,
    const int64_t* strides_a, const int64_t* strides_b,
    const int64_t* output_shape, int64_t ndim, int64_t n) {

    HIP_KERNEL_LOOP(out_idx, n) {
        int64_t idx_a = 0;
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % output_shape[i];
            tmp /= output_shape[i];
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
        }

        float divisor = static_cast<float>(b[idx_b]);
        if (divisor == 0.0f) {
            c[out_idx] = hip_bfloat16(INFINITY);
        } else {
            float result = static_cast<float>(a[idx_a]) / divisor;
            c[out_idx] = hip_bfloat16(result);
        }
    }
}

// ============================================================================
// Complex Elementwise Arithmetic Kernels
// ============================================================================

// Complex add: (ar+ai*i) + (br+bi*i) = (ar+br) + (ai+bi)*i
template<typename T>
__global__ void complex_add_kernel(const T* a, const T* b, T* c, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        int64_t base = idx * 2;
        c[base]     = a[base]     + b[base];
        c[base + 1] = a[base + 1] + b[base + 1];
    }
}

// Complex sub: (ar+ai*i) - (br+bi*i) = (ar-br) + (ai-bi)*i
template<typename T>
__global__ void complex_sub_kernel(const T* a, const T* b, T* c, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        int64_t base = idx * 2;
        c[base]     = a[base]     - b[base];
        c[base + 1] = a[base + 1] - b[base + 1];
    }
}

// Complex mul: (ar+ai*i)*(br+bi*i) = (ar*br - ai*bi) + (ar*bi + ai*br)*i
template<typename T>
__global__ void complex_mul_kernel(const T* a, const T* b, T* c, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
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
    HIP_KERNEL_LOOP(idx, n) {
        int64_t base = idx * 2;
        T ar = a[base], ai = a[base + 1];
        T br = b[base], bi = b[base + 1];
        T denom = br * br + bi * bi;
        c[base]     = (ar * br + ai * bi) / denom;
        c[base + 1] = (ai * br - ar * bi) / denom;
    }
}

// Broadcast kernel for complex types - strides are in complex element units
template<typename T>
__global__ void broadcast_complex_add_kernel(
    const T* a, const T* b, T* c,
    const int64_t* strides_a, const int64_t* strides_b,
    const int64_t* output_shape, int64_t ndim, int64_t n) {
    HIP_KERNEL_LOOP(out_idx, n) {
        int64_t idx_a = 0, idx_b = 0, tmp = out_idx;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % output_shape[i];
            tmp /= output_shape[i];
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
        }
        int64_t a_base = idx_a * 2, b_base = idx_b * 2, c_base = out_idx * 2;
        c[c_base]     = a[a_base]     + b[b_base];
        c[c_base + 1] = a[a_base + 1] + b[b_base + 1];
    }
}

template<typename T>
__global__ void broadcast_complex_sub_kernel(
    const T* a, const T* b, T* c,
    const int64_t* strides_a, const int64_t* strides_b,
    const int64_t* output_shape, int64_t ndim, int64_t n) {
    HIP_KERNEL_LOOP(out_idx, n) {
        int64_t idx_a = 0, idx_b = 0, tmp = out_idx;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % output_shape[i];
            tmp /= output_shape[i];
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
        }
        int64_t a_base = idx_a * 2, b_base = idx_b * 2, c_base = out_idx * 2;
        c[c_base]     = a[a_base]     - b[b_base];
        c[c_base + 1] = a[a_base + 1] - b[b_base + 1];
    }
}

template<typename T>
__global__ void broadcast_complex_mul_kernel(
    const T* a, const T* b, T* c,
    const int64_t* strides_a, const int64_t* strides_b,
    const int64_t* output_shape, int64_t ndim, int64_t n) {
    HIP_KERNEL_LOOP(out_idx, n) {
        int64_t idx_a = 0, idx_b = 0, tmp = out_idx;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % output_shape[i];
            tmp /= output_shape[i];
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
        }
        int64_t a_base = idx_a * 2, b_base = idx_b * 2, c_base = out_idx * 2;
        T ar = a[a_base], ai = a[a_base + 1];
        T br = b[b_base], bi = b[b_base + 1];
        c[c_base]     = ar * br - ai * bi;
        c[c_base + 1] = ar * bi + ai * br;
    }
}

template<typename T>
__global__ void broadcast_complex_div_kernel(
    const T* a, const T* b, T* c,
    const int64_t* strides_a, const int64_t* strides_b,
    const int64_t* output_shape, int64_t ndim, int64_t n) {
    HIP_KERNEL_LOOP(out_idx, n) {
        int64_t idx_a = 0, idx_b = 0, tmp = out_idx;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % output_shape[i];
            tmp /= output_shape[i];
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
        }
        int64_t a_base = idx_a * 2, b_base = idx_b * 2, c_base = out_idx * 2;
        T ar = a[a_base], ai = a[a_base + 1];
        T br = b[b_base], bi = b[b_base + 1];
        T denom = br * br + bi * bi;
        c[c_base]     = (ar * br + ai * bi) / denom;
        c[c_base + 1] = (ai * br - ar * bi) / denom;
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

/**
 * @brief Negation kernel: output = -input
 * @tparam T Data type
 */
template<typename T>
__global__ void neg_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = -input[idx];
    }
}

/**
 * @brief Absolute value kernel (generic template)
 * @tparam T Data type
 */
template<typename T>
__global__ void abs_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        T val = input[idx];
        output[idx] = val >= T(0) ? val : -val;
    }
}

/**
 * @brief Absolute value kernel (specialized for float, uses fabsf)
 */
__global__ void abs_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = fabsf(input[idx]);
    }
}

/**
 * @brief Absolute value kernel (specialized for double, uses fabs)
 */
__global__ void abs_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = fabs(input[idx]);
    }
}

// ============================================================================
// Mathematical Functions
// ============================================================================

/**
 * @brief Square root kernel (float precision)
 * Uses sqrtf for optimal AMD GPU performance
 */
__global__ void sqrt_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = sqrtf(input[idx]);
    }
}

/**
 * @brief Square root kernel (double precision)
 */
__global__ void sqrt_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = sqrt(input[idx]);
    }
}

/**
 * @brief Exponential kernel (float precision): output = e^input
 * Uses expf for optimal AMD GPU performance
 */
__global__ void exp_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = expf(input[idx]);
    }
}

/**
 * @brief Exponential kernel (double precision)
 */
__global__ void exp_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = exp(input[idx]);
    }
}

/**
 * @brief Natural logarithm kernel (float precision): output = ln(input)
 * Uses logf for optimal AMD GPU performance
 */
__global__ void log_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = logf(input[idx]);
    }
}

/**
 * @brief Natural logarithm kernel (double precision)
 */
__global__ void log_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = log(input[idx]);
    }
}

/**
 * @brief Power kernel (float precision): output = input^exponent
 * Uses powf for optimal AMD GPU performance
 */
__global__ void pow_kernel_f32(const float* input, float* output, float exponent, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = powf(input[idx], exponent);
    }
}

/**
 * @brief Power kernel (double precision)
 */
__global__ void pow_kernel_f64(const double* input, double* output, double exponent, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = pow(input[idx], exponent);
    }
}

/**
 * @brief Power kernel (half precision) - compute via float conversion
 */
__global__ void pow_kernel_f16(const __half* input, __half* output, float exponent, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        float result = powf(val, exponent);
        output[idx] = __float2half(result);
    }
}

/**
 * @brief Clamp kernel (float): clamp values to [min_val, max_val]
 * Uses fminf/fmaxf for optimal AMD GPU performance
 */
__global__ void clamp_kernel_f32(const float* input, float* output, float min_val, float max_val, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = input[idx];
        output[idx] = fminf(fmaxf(val, min_val), max_val);
    }
}

/**
 * @brief Clamp kernel (double precision)
 */
__global__ void clamp_kernel_f64(const double* input, double* output, double min_val, double max_val, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double val = input[idx];
        output[idx] = fmin(fmax(val, min_val), max_val);
    }
}

/**
 * @brief Clamp kernel (half precision) - compute via float conversion
 */
__global__ void clamp_kernel_f16(const __half* input, __half* output, float min_val, float max_val, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        float result = fminf(fmaxf(val, min_val), max_val);
        output[idx] = __float2half(result);
    }
}

/**
 * @brief Sign kernel (float): returns -1, 0, or +1
 * Sign function: -1 if x < 0, 0 if x == 0, +1 if x > 0
 */
__global__ void sign_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = input[idx];
        output[idx] = (val > 0.0f) - (val < 0.0f);
    }
}

/**
 * @brief Sign kernel (double precision)
 */
__global__ void sign_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double val = input[idx];
        output[idx] = (val > 0.0) - (val < 0.0);
    }
}

// ============================================================================
// Float16 Kernels (compute via float conversion for accuracy)
// ============================================================================

__global__ void sqrt_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(sqrtf(__half2float(input[idx])));
    }
}

__global__ void exp_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(expf(__half2float(input[idx])));
    }
}

__global__ void log_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(logf(__half2float(input[idx])));
    }
}

__global__ void sign_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        output[idx] = __float2half((val > 0.0f) - (val < 0.0f));
    }
}

__global__ void sin_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(sinf(__half2float(input[idx])));
    }
}

__global__ void cos_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(cosf(__half2float(input[idx])));
    }
}

__global__ void tan_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(tanf(__half2float(input[idx])));
    }
}

__global__ void asin_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(asinf(__half2float(input[idx])));
    }
}

__global__ void acos_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(acosf(__half2float(input[idx])));
    }
}

__global__ void atan_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(atanf(__half2float(input[idx])));
    }
}

__global__ void sinh_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(sinhf(__half2float(input[idx])));
    }
}

__global__ void cosh_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(coshf(__half2float(input[idx])));
    }
}

__global__ void reciprocal_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        output[idx] = __float2half(1.0f / val);
    }
}

__global__ void floor_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(floorf(__half2float(input[idx])));
    }
}

__global__ void ceil_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(ceilf(__half2float(input[idx])));
    }
}

__global__ void round_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(roundf(__half2float(input[idx])));
    }
}

__global__ void div_inplace_kernel_f16(__half* a, const __half* b, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float divisor = __half2float(b[idx]);
        float result = __half2float(a[idx]) / divisor;
        a[idx] = __float2half(result);
    }
}

// Dot product kernel for Float16 - computes in float for accuracy
__global__ void dot_kernel_f16(const __half* a, const __half* b, float* partial_sums, int64_t n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    while (idx < n) {
        sum += __half2float(a[idx]) * __half2float(b[idx]);
        idx += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Optimized Kernels with Shared Memory (for reduction-like operations)
// ============================================================================

/**
 * @brief Optimized add with LDS (Local Data Share) for small tensors
 * @tparam T Data type
 *
 * Uses AMD GPU's LDS (equivalent to CUDA shared memory) for better
 * memory access patterns and reduced global memory traffic
 */
template<typename T>
__global__ void add_kernel_shared(const T* a, const T* b, T* c, int64_t n) {
    __shared__ T s_a[256];
    __shared__ T s_b[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory (LDS)
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

/**
 * @brief Add kernel launcher with broadcasting support
 * @param a First input tensor
 * @param b Second input tensor
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (a + b)
 *
 * Supports both fast path (same shape) and broadcast path
 */
auto add_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
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
            hipLaunchKernelGGL(add_kernel_device<float>, grid, block, 0, stream,
                a.data<float>(), b.data<float>(), result.data<float>(), n);
        } else if (a.dtype() == DType::Float64) {
            hipLaunchKernelGGL(add_kernel_device<double>, grid, block, 0, stream,
                a.data<double>(), b.data<double>(), result.data<double>(), n);
        } else if (a.dtype() == DType::Int32) {
            hipLaunchKernelGGL(add_kernel_device<int32_t>, grid, block, 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            hipLaunchKernelGGL(add_kernel_device<int64_t>, grid, block, 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else if (a.dtype() == DType::Float16) {
            hipLaunchKernelGGL(add_kernel_device<__half>, grid, block, 0, stream,
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
        } else if (a.dtype() == DType::BFloat16) {
            hipLaunchKernelGGL(add_kernel_device<hip_bfloat16>, grid, block, 0, stream,
                reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
        } else if (a.dtype() == DType::Int8) {
            hipLaunchKernelGGL(add_kernel_device<int8_t>, grid, block, 0, stream,
                a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n);
        } else if (a.dtype() == DType::UInt8) {
            hipLaunchKernelGGL(add_kernel_device<uint8_t>, grid, block, 0, stream,
                a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(), n);
        } else if (a.dtype() == DType::Bool) {
            // For Bool, add acts as logical OR
            hipLaunchKernelGGL(add_kernel_device<bool>, grid, block, 0, stream,
                a.data<bool>(), b.data<bool>(), result.data<bool>(), n);
        } else if (a.dtype() == DType::Complex64) {
            hipLaunchKernelGGL(complex_add_kernel<float>, grid, block, 0, stream,
                reinterpret_cast<const float*>(a.data_ptr()),
                reinterpret_cast<const float*>(b.data_ptr()),
                reinterpret_cast<float*>(result.data_ptr()), n);
        } else if (a.dtype() == DType::Complex128) {
            hipLaunchKernelGGL(complex_add_kernel<double>, grid, block, 0, stream,
                reinterpret_cast<const double*>(a.data_ptr()),
                reinterpret_cast<const double*>(b.data_ptr()),
                reinterpret_cast<double*>(result.data_ptr()), n);
        } else {
            throw std::runtime_error("Unsupported dtype for add operation");
        }

        HIP_CHECK(hipGetLastError());
        if (a.dtype() == DType::Float16) {
            fp16_saturate(result.data_ptr(), result.numel(), stream);
        }
        return result;
    }

    // Broadcasting path
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(a_shape_vec, b_shape_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    // Compute strides
    std::vector<int64_t> strides_a = detail::compute_broadcast_strides(a_shape_vec, output_shape);
    std::vector<int64_t> strides_b = detail::compute_broadcast_strides(b_shape_vec, output_shape);

    // Copy strides to device
    int64_t* d_strides_a;
    int64_t* d_strides_b;
    int64_t* d_output_shape;
    HIP_CHECK(hipMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));

    int64_t n = result.numel();
    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<float, AddOp>), grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<double, AddOp>), grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int32_t, AddOp>), grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int64_t, AddOp>), grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<__half, AddOp>), grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<hip_bfloat16, AddOp>), grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Int8) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int8_t, AddOp>), grid, block, 0, stream,
            a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::UInt8) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<uint8_t, AddOp>), grid, block, 0, stream,
            a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Bool) {
        // For Bool, add acts as logical OR
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<bool, AddOp>), grid, block, 0, stream,
            a.data<bool>(), b.data<bool>(), result.data<bool>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, AddOp());
    } else if (a.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(broadcast_complex_add_kernel<float>, grid, block, 0, stream,
            reinterpret_cast<const float*>(a.data_ptr()),
            reinterpret_cast<const float*>(b.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else if (a.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(broadcast_complex_add_kernel<double>, grid, block, 0, stream,
            reinterpret_cast<const double*>(a.data_ptr()),
            reinterpret_cast<const double*>(b.data_ptr()),
            reinterpret_cast<double*>(result.data_ptr()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else {
        throw std::runtime_error("Unsupported dtype for add operation");
    }

    // Cleanup
    HIP_CHECK(hipFree(d_strides_a));
    HIP_CHECK(hipFree(d_strides_b));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());
    if (a.dtype() == DType::Float16) {
        fp16_saturate(result.data_ptr(), result.numel(), stream);
    }

    return result;
}

/**
 * @brief Subtract kernel launcher with broadcasting support
 * @param a First input tensor (minuend)
 * @param b Second input tensor (subtrahend)
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (a - b)
 */
auto sub_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
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
            hipLaunchKernelGGL(sub_kernel_device<float>, grid, block, 0, stream,
                a.data<float>(), b.data<float>(), result.data<float>(), n);
        } else if (a.dtype() == DType::Float64) {
            hipLaunchKernelGGL(sub_kernel_device<double>, grid, block, 0, stream,
                a.data<double>(), b.data<double>(), result.data<double>(), n);
        } else if (a.dtype() == DType::Float16) {
            hipLaunchKernelGGL(sub_kernel_device<__half>, grid, block, 0, stream,
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
        } else if (a.dtype() == DType::BFloat16) {
            hipLaunchKernelGGL(sub_kernel_device<hip_bfloat16>, grid, block, 0, stream,
                reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
        } else if (a.dtype() == DType::Int32) {
            hipLaunchKernelGGL(sub_kernel_device<int32_t>, grid, block, 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            hipLaunchKernelGGL(sub_kernel_device<int64_t>, grid, block, 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else if (a.dtype() == DType::Int8) {
            hipLaunchKernelGGL(sub_kernel_device<int8_t>, grid, block, 0, stream,
                a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n);
        } else if (a.dtype() == DType::UInt8) {
            hipLaunchKernelGGL(sub_kernel_device<uint8_t>, grid, block, 0, stream,
                a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(), n);
        } else if (a.dtype() == DType::Complex64) {
            hipLaunchKernelGGL(complex_sub_kernel<float>, grid, block, 0, stream,
                reinterpret_cast<const float*>(a.data_ptr()),
                reinterpret_cast<const float*>(b.data_ptr()),
                reinterpret_cast<float*>(result.data_ptr()), n);
        } else if (a.dtype() == DType::Complex128) {
            hipLaunchKernelGGL(complex_sub_kernel<double>, grid, block, 0, stream,
                reinterpret_cast<const double*>(a.data_ptr()),
                reinterpret_cast<const double*>(b.data_ptr()),
                reinterpret_cast<double*>(result.data_ptr()), n);
        } else {
            throw std::runtime_error("Unsupported dtype for sub operation");
        }

        HIP_CHECK(hipGetLastError());
        if (a.dtype() == DType::Float16) {
            fp16_saturate(result.data_ptr(), result.numel(), stream);
        }
        return result;
    }

    // Broadcasting path
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(a_shape_vec, b_shape_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    std::vector<int64_t> strides_a = detail::compute_broadcast_strides(a_shape_vec, output_shape);
    std::vector<int64_t> strides_b = detail::compute_broadcast_strides(b_shape_vec, output_shape);

    int64_t* d_strides_a;
    int64_t* d_strides_b;
    int64_t* d_output_shape;
    HIP_CHECK(hipMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));

    int64_t n = result.numel();
    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<float, SubOp>), grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<double, SubOp>), grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<__half, SubOp>), grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<hip_bfloat16, SubOp>), grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int32_t, SubOp>), grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int64_t, SubOp>), grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Int8) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int8_t, SubOp>), grid, block, 0, stream,
            a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::UInt8) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<uint8_t, SubOp>), grid, block, 0, stream,
            a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(broadcast_complex_sub_kernel<float>, grid, block, 0, stream,
            reinterpret_cast<const float*>(a.data_ptr()),
            reinterpret_cast<const float*>(b.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else if (a.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(broadcast_complex_sub_kernel<double>, grid, block, 0, stream,
            reinterpret_cast<const double*>(a.data_ptr()),
            reinterpret_cast<const double*>(b.data_ptr()),
            reinterpret_cast<double*>(result.data_ptr()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else {
        throw std::runtime_error("Unsupported dtype for sub operation");
    }

    HIP_CHECK(hipFree(d_strides_a));
    HIP_CHECK(hipFree(d_strides_b));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());
    if (a.dtype() == DType::Float16) {
        fp16_saturate(result.data_ptr(), result.numel(), stream);
    }

    return result;
}

/**
 * @brief Multiply kernel launcher with broadcasting support
 * @param a First input tensor
 * @param b Second input tensor
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (a * b)
 */
auto mul_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
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
            hipLaunchKernelGGL(mul_kernel_device<float>, grid, block, 0, stream,
                a.data<float>(), b.data<float>(), result.data<float>(), n);
        } else if (a.dtype() == DType::Float64) {
            hipLaunchKernelGGL(mul_kernel_device<double>, grid, block, 0, stream,
                a.data<double>(), b.data<double>(), result.data<double>(), n);
        } else if (a.dtype() == DType::Int32) {
            hipLaunchKernelGGL(mul_kernel_device<int32_t>, grid, block, 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            hipLaunchKernelGGL(mul_kernel_device<int64_t>, grid, block, 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else if (a.dtype() == DType::Float16) {
            hipLaunchKernelGGL(mul_kernel_device<__half>, grid, block, 0, stream,
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
        } else if (a.dtype() == DType::BFloat16) {
            hipLaunchKernelGGL(mul_kernel_device<hip_bfloat16>, grid, block, 0, stream,
                reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
        } else if (a.dtype() == DType::Bool) {
            // For Bool, mul acts as logical AND
            hipLaunchKernelGGL(mul_kernel_device<bool>, grid, block, 0, stream,
                a.data<bool>(), b.data<bool>(), result.data<bool>(), n);
        } else if (a.dtype() == DType::Complex64) {
            hipLaunchKernelGGL(complex_mul_kernel<float>, grid, block, 0, stream,
                reinterpret_cast<const float*>(a.data_ptr()),
                reinterpret_cast<const float*>(b.data_ptr()),
                reinterpret_cast<float*>(result.data_ptr()), n);
        } else if (a.dtype() == DType::Complex128) {
            hipLaunchKernelGGL(complex_mul_kernel<double>, grid, block, 0, stream,
                reinterpret_cast<const double*>(a.data_ptr()),
                reinterpret_cast<const double*>(b.data_ptr()),
                reinterpret_cast<double*>(result.data_ptr()), n);
        } else {
            throw std::runtime_error("Unsupported dtype for mul operation");
        }

        HIP_CHECK(hipGetLastError());
        if (a.dtype() == DType::Float16) {
            fp16_saturate(result.data_ptr(), result.numel(), stream);
        }
        return result;
    }

    // Broadcasting path
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(a_shape_vec, b_shape_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    std::vector<int64_t> strides_a = detail::compute_broadcast_strides(a_shape_vec, output_shape);
    std::vector<int64_t> strides_b = detail::compute_broadcast_strides(b_shape_vec, output_shape);

    int64_t* d_strides_a;
    int64_t* d_strides_b;
    int64_t* d_output_shape;
    HIP_CHECK(hipMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));

    int64_t n = result.numel();
    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<float, MulOp>), grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<double, MulOp>), grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int32_t, MulOp>), grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int64_t, MulOp>), grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<__half, MulOp>), grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<hip_bfloat16, MulOp>), grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Bool) {
        // For Bool, mul acts as logical AND
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<bool, MulOp>), grid, block, 0, stream,
            a.data<bool>(), b.data<bool>(), result.data<bool>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(broadcast_complex_mul_kernel<float>, grid, block, 0, stream,
            reinterpret_cast<const float*>(a.data_ptr()),
            reinterpret_cast<const float*>(b.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else if (a.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(broadcast_complex_mul_kernel<double>, grid, block, 0, stream,
            reinterpret_cast<const double*>(a.data_ptr()),
            reinterpret_cast<const double*>(b.data_ptr()),
            reinterpret_cast<double*>(result.data_ptr()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else {
        throw std::runtime_error("Unsupported dtype for mul operation");
    }

    HIP_CHECK(hipFree(d_strides_a));
    HIP_CHECK(hipFree(d_strides_b));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());
    if (a.dtype() == DType::Float16) {
        fp16_saturate(result.data_ptr(), result.numel(), stream);
    }

    return result;
}

/**
 * @brief Divide kernel launcher with broadcasting support
 * @param a First input tensor (dividend)
 * @param b Second input tensor (divisor)
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (a / b)
 *
 * Division by zero returns INFINITY
 */
auto div_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
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
            hipLaunchKernelGGL(div_kernel_device<float>, grid, block, 0, stream,
                a.data<float>(), b.data<float>(), result.data<float>(), n);
        } else if (a.dtype() == DType::Float64) {
            hipLaunchKernelGGL(div_kernel_device<double>, grid, block, 0, stream,
                a.data<double>(), b.data<double>(), result.data<double>(), n);
        } else if (a.dtype() == DType::Float16) {
            hipLaunchKernelGGL(div_kernel_f16, grid, block, 0, stream,
                reinterpret_cast<const __half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()), n);
        } else if (a.dtype() == DType::BFloat16) {
            hipLaunchKernelGGL(div_kernel_bf16, grid, block, 0, stream,
                reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
                reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
        } else if (a.dtype() == DType::Int32) {
            hipLaunchKernelGGL(div_kernel_device<int32_t>, grid, block, 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            hipLaunchKernelGGL(div_kernel_device<int64_t>, grid, block, 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else if (a.dtype() == DType::Complex64) {
            hipLaunchKernelGGL(complex_div_kernel<float>, grid, block, 0, stream,
                reinterpret_cast<const float*>(a.data_ptr()),
                reinterpret_cast<const float*>(b.data_ptr()),
                reinterpret_cast<float*>(result.data_ptr()), n);
        } else if (a.dtype() == DType::Complex128) {
            hipLaunchKernelGGL(complex_div_kernel<double>, grid, block, 0, stream,
                reinterpret_cast<const double*>(a.data_ptr()),
                reinterpret_cast<const double*>(b.data_ptr()),
                reinterpret_cast<double*>(result.data_ptr()), n);
        } else {
            throw std::runtime_error("Unsupported dtype for div operation");
        }

        HIP_CHECK(hipGetLastError());
        if (a.dtype() == DType::Float16) {
            fp16_saturate(result.data_ptr(), result.numel(), stream);
        }
        return result;
    }

    // Broadcasting path
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(a_shape_vec, b_shape_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    std::vector<int64_t> strides_a = detail::compute_broadcast_strides(a_shape_vec, output_shape);
    std::vector<int64_t> strides_b = detail::compute_broadcast_strides(b_shape_vec, output_shape);

    int64_t* d_strides_a;
    int64_t* d_strides_b;
    int64_t* d_output_shape;
    HIP_CHECK(hipMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));

    int64_t n = result.numel();
    int64_t ndim = output_shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<float, DivOp>), grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<double, DivOp>), grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int32_t, DivOp>), grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int64_t, DivOp>), grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(broadcast_div_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(broadcast_div_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else if (a.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(broadcast_complex_div_kernel<float>, grid, block, 0, stream,
            reinterpret_cast<const float*>(a.data_ptr()),
            reinterpret_cast<const float*>(b.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else if (a.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(broadcast_complex_div_kernel<double>, grid, block, 0, stream,
            reinterpret_cast<const double*>(a.data_ptr()),
            reinterpret_cast<const double*>(b.data_ptr()),
            reinterpret_cast<double*>(result.data_ptr()),
            d_strides_a, d_strides_b, d_output_shape, ndim, n);
    } else {
        throw std::runtime_error("Unsupported dtype for div operation");
    }

    HIP_CHECK(hipFree(d_strides_a));
    HIP_CHECK(hipFree(d_strides_b));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());
    if (a.dtype() == DType::Float16) {
        fp16_saturate(result.data_ptr(), result.numel(), stream);
    }

    return result;
}

/**
 * @brief Negate kernel launcher: output = -input
 * @param input Input tensor
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (-input)
 */
auto neg_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(neg_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(neg_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(neg_kernel_device<int32_t>, grid, block, 0, stream,
            input.data<int32_t>(), result.data<int32_t>(), n);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(neg_kernel_device<int64_t>, grid, block, 0, stream,
            input.data<int64_t>(), result.data<int64_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(neg_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(neg_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("Unsupported dtype for neg operation");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Absolute value kernel launcher: output = |input|
 * @param input Input tensor
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (|input|)
 */
auto abs_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(abs_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(abs_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(abs_kernel_device<int32_t>, grid, block, 0, stream,
            input.data<int32_t>(), result.data<int32_t>(), n);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(abs_kernel_device<int64_t>, grid, block, 0, stream,
            input.data<int64_t>(), result.data<int64_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(abs_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(abs_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("Unsupported dtype for abs operation");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Square root kernel launcher: output = sqrt(input)
 * @param input Input tensor
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (sqrt(input))
 */
auto sqrt_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(sqrt_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(sqrt_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(sqrt_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("sqrt operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Exponential kernel launcher: output = e^input
 * @param input Input tensor
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (e^input)
 */
auto exp_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(exp_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(exp_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(exp_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("exp operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Natural logarithm kernel launcher: output = ln(input)
 * @param input Input tensor
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (ln(input))
 */
auto log_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(log_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(log_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(log_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("log operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Power kernel launcher: output = input^exponent
 * @param input Input tensor (base)
 * @param exponent Power exponent
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (input^exponent)
 */
auto pow_kernel(const Tensor& input, float exponent, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(pow_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), exponent, n);
    } else if (input.dtype() == DType::Float64) {
        double exp_d = static_cast<double>(exponent);
        hipLaunchKernelGGL(pow_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), exp_d, n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(pow_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), exponent, n);
    } else {
        throw std::runtime_error("pow operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Clamp kernel launcher: output = clamp(input, min_val, max_val)
 * @param input Input tensor
 * @param min_val Minimum value
 * @param max_val Maximum value
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (clamped values)
 */
auto clamp_kernel(const Tensor& input, float min_val, float max_val, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(clamp_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), min_val, max_val, n);
    } else if (input.dtype() == DType::Float64) {
        double min_d = static_cast<double>(min_val);
        double max_d = static_cast<double>(max_val);
        hipLaunchKernelGGL(clamp_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), min_d, max_d, n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(clamp_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), min_val, max_val, n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = clamp_kernel(input_f32, min_val, max_val, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("clamp operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Sign kernel launcher: output = sign(input) ∈ {-1, 0, +1}
 * @param input Input tensor
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (sign values)
 */
auto sign_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(sign_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(sign_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(sign_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("sign operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Trigonometric Operations
// ============================================================================

template<typename T>
__global__ void sin_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = sin(input[idx]);
    }
}

template<typename T>
__global__ void cos_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = cos(input[idx]);
    }
}

template<typename T>
__global__ void tan_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tan(input[idx]);
    }
}

template<typename T>
__global__ void asin_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = asin(input[idx]);
    }
}

template<typename T>
__global__ void acos_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = acos(input[idx]);
    }
}

template<typename T>
__global__ void atan_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = atan(input[idx]);
    }
}

template<typename T>
__global__ void sinh_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = sinh(input[idx]);
    }
}

template<typename T>
__global__ void cosh_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = cosh(input[idx]);
    }
}

// ============================================================================
// Additional Math Operations (Reciprocal, Floor, Ceil, Round)
// ============================================================================

template<typename T>
__global__ void reciprocal_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = T(1) / input[idx];
    }
}

template<typename T>
__global__ void floor_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = floor(input[idx]);
    }
}

template<typename T>
__global__ void ceil_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = ceil(input[idx]);
    }
}

template<typename T>
__global__ void round_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = round(input[idx]);
    }
}

template<typename T>
__global__ void trunc_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = trunc(input[idx]);
    }
}

__global__ void trunc_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(truncf(__half2float(input[idx])));
    }
}

// ============================================================================
// In-place Binary Operations
// ============================================================================

template<typename T>
__global__ void add_inplace_kernel_device(T* a, const T* b, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        a[idx] += b[idx];
    }
}

template<typename T>
__global__ void sub_inplace_kernel_device(T* a, const T* b, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        a[idx] -= b[idx];
    }
}

template<typename T>
__global__ void mul_inplace_kernel_device(T* a, const T* b, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        a[idx] *= b[idx];
    }
}

template<typename T>
__global__ void div_inplace_kernel_device(T* a, const T* b, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        a[idx] /= b[idx];
    }
}

// Broadcast in-place kernels: a[i] op= b[broadcast_index(i)]
template<typename T>
__global__ void add_inplace_broadcast_kernel(T* a, const T* b, const int64_t* strides_b,
    const int64_t* a_shape, int64_t ndim, int64_t n) {
    HIP_KERNEL_LOOP(out_idx, n) {
        int64_t idx_b = 0, tmp = out_idx;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % a_shape[i];
            tmp /= a_shape[i];
            idx_b += coord * strides_b[i];
        }
        a[out_idx] += b[idx_b];
    }
}

template<typename T>
__global__ void sub_inplace_broadcast_kernel(T* a, const T* b, const int64_t* strides_b,
    const int64_t* a_shape, int64_t ndim, int64_t n) {
    HIP_KERNEL_LOOP(out_idx, n) {
        int64_t idx_b = 0, tmp = out_idx;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % a_shape[i];
            tmp /= a_shape[i];
            idx_b += coord * strides_b[i];
        }
        a[out_idx] -= b[idx_b];
    }
}

template<typename T>
__global__ void mul_inplace_broadcast_kernel(T* a, const T* b, const int64_t* strides_b,
    const int64_t* a_shape, int64_t ndim, int64_t n) {
    HIP_KERNEL_LOOP(out_idx, n) {
        int64_t idx_b = 0, tmp = out_idx;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % a_shape[i];
            tmp /= a_shape[i];
            idx_b += coord * strides_b[i];
        }
        a[out_idx] *= b[idx_b];
    }
}

template<typename T>
__global__ void div_inplace_broadcast_kernel(T* a, const T* b, const int64_t* strides_b,
    const int64_t* a_shape, int64_t ndim, int64_t n) {
    HIP_KERNEL_LOOP(out_idx, n) {
        int64_t idx_b = 0, tmp = out_idx;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % a_shape[i];
            tmp /= a_shape[i];
            idx_b += coord * strides_b[i];
        }
        a[out_idx] /= b[idx_b];
    }
}

// ============================================================================
// Dot Product Kernel
// ============================================================================

template<typename T>
__global__ void dot_kernel_device(const T* a, const T* b, T* partial_sums, int64_t n) {
    __shared__ T sdata[256];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = T(0);
    while (idx < n) {
        sum += a[idx] * b[idx];
        idx += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Host Wrapper Functions for Trigonometric Operations
// ============================================================================

auto sin_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(sin_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(sin_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(sin_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("sin operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto cos_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(cos_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(cos_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(cos_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("cos operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto tan_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(tan_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(tan_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(tan_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("tan operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto asin_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(asin_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(asin_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(asin_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("asin operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto acos_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(acos_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(acos_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(acos_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("acos operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto atan_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(atan_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(atan_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(atan_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("atan operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto sinh_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(sinh_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(sinh_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(sinh_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("sinh operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto cosh_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(cosh_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(cosh_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(cosh_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("cosh operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Host Wrapper Functions for Additional Math Operations
// ============================================================================

auto reciprocal_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(reciprocal_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(reciprocal_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(reciprocal_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("reciprocal operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto floor_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(floor_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(floor_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(floor_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("floor operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto ceil_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(ceil_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(ceil_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(ceil_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("ceil operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto round_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(round_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(round_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(round_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("round operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto trunc_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(trunc_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(trunc_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(trunc_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("trunc operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Host Wrapper Functions for In-place Binary Operations
// ============================================================================

namespace detail {

// Check if b is broadcastable to a's shape (for in-place ops, result shape must be a's shape)
inline bool needs_broadcast_inplace(const Tensor& a, const Tensor& b) {
    if (a.ndim() == b.ndim()) {
        auto sa = a.shape();
        auto sb = b.shape();
        for (size_t i = 0; i < sa.size(); ++i) {
            if (sa[i] != sb[i]) return true;
        }
        return false;
    }
    return true;
}

// Compute broadcast strides for b relative to a's shape
// Returns empty vector if not broadcastable
inline std::vector<int64_t> compute_inplace_broadcast_strides(const Tensor& a, const Tensor& b) {
    auto a_shape_span = a.shape();
    auto b_shape_span = b.shape();
    std::vector<int64_t> a_shape(a_shape_span.begin(), a_shape_span.end());
    std::vector<int64_t> b_shape(b_shape_span.begin(), b_shape_span.end());
    return compute_broadcast_strides(b_shape, a_shape);
}

struct InplaceBroadcastMeta {
    int64_t* d_strides_b;
    int64_t* d_a_shape;
    int64_t ndim;

    static InplaceBroadcastMeta create(const Tensor& a, const Tensor& b) {
        auto a_shape_span = a.shape();
        std::vector<int64_t> a_shape(a_shape_span.begin(), a_shape_span.end());
        auto b_shape_span = b.shape();
        std::vector<int64_t> b_shape(b_shape_span.begin(), b_shape_span.end());
        auto strides_b = compute_broadcast_strides(b_shape, a_shape);

        InplaceBroadcastMeta meta;
        meta.ndim = static_cast<int64_t>(a_shape.size());
        size_t bytes = meta.ndim * sizeof(int64_t);
        HIP_CHECK(hipMalloc(&meta.d_strides_b, bytes));
        HIP_CHECK(hipMalloc(&meta.d_a_shape, bytes));
        HIP_CHECK(hipMemcpy(meta.d_strides_b, strides_b.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(meta.d_a_shape, a_shape.data(), bytes, hipMemcpyHostToDevice));
        return meta;
    }

    void free() {
        HIP_CHECK(hipFree(d_strides_b));
        HIP_CHECK(hipFree(d_a_shape));
    }
};

} // namespace detail

void add_inplace_kernel(Tensor& a, const Tensor& b, hipStream_t stream) {
    int64_t n = a.numel();
    if (n == 0) return;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (detail::needs_broadcast_inplace(a, b)) {
        auto meta = detail::InplaceBroadcastMeta::create(a, b);
        if (a.dtype() == DType::Float32) {
            hipLaunchKernelGGL(add_inplace_broadcast_kernel<float>, grid, block, 0, stream,
                a.data<float>(), b.data<float>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Float64) {
            hipLaunchKernelGGL(add_inplace_broadcast_kernel<double>, grid, block, 0, stream,
                a.data<double>(), b.data<double>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Int32) {
            hipLaunchKernelGGL(add_inplace_broadcast_kernel<int32_t>, grid, block, 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Int64) {
            hipLaunchKernelGGL(add_inplace_broadcast_kernel<int64_t>, grid, block, 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Float16) {
            hipLaunchKernelGGL(add_inplace_broadcast_kernel<__half>, grid, block, 0, stream,
                reinterpret_cast<__half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::BFloat16) {
            hipLaunchKernelGGL(add_inplace_broadcast_kernel<hip_bfloat16>, grid, block, 0, stream,
                reinterpret_cast<hip_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
                meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else {
            meta.free();
            throw std::runtime_error("add_inplace operation unsupported dtype");
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        meta.free();
        return;
    }

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(add_inplace_kernel_device<float>, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(add_inplace_kernel_device<double>, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), n);
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(add_inplace_kernel_device<int32_t>, grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(add_inplace_kernel_device<int64_t>, grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(add_inplace_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(add_inplace_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("add_inplace operation unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
}

void sub_inplace_kernel(Tensor& a, const Tensor& b, hipStream_t stream) {
    int64_t n = a.numel();
    if (n == 0) return;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (detail::needs_broadcast_inplace(a, b)) {
        auto meta = detail::InplaceBroadcastMeta::create(a, b);
        if (a.dtype() == DType::Float32) {
            hipLaunchKernelGGL(sub_inplace_broadcast_kernel<float>, grid, block, 0, stream,
                a.data<float>(), b.data<float>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Float64) {
            hipLaunchKernelGGL(sub_inplace_broadcast_kernel<double>, grid, block, 0, stream,
                a.data<double>(), b.data<double>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Int32) {
            hipLaunchKernelGGL(sub_inplace_broadcast_kernel<int32_t>, grid, block, 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Int64) {
            hipLaunchKernelGGL(sub_inplace_broadcast_kernel<int64_t>, grid, block, 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Float16) {
            hipLaunchKernelGGL(sub_inplace_broadcast_kernel<__half>, grid, block, 0, stream,
                reinterpret_cast<__half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::BFloat16) {
            hipLaunchKernelGGL(sub_inplace_broadcast_kernel<hip_bfloat16>, grid, block, 0, stream,
                reinterpret_cast<hip_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
                meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else {
            meta.free();
            throw std::runtime_error("sub_inplace operation unsupported dtype");
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        meta.free();
        return;
    }

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(sub_inplace_kernel_device<float>, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(sub_inplace_kernel_device<double>, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), n);
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(sub_inplace_kernel_device<int32_t>, grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(sub_inplace_kernel_device<int64_t>, grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(sub_inplace_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(sub_inplace_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("sub_inplace operation unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
}

void mul_inplace_kernel(Tensor& a, const Tensor& b, hipStream_t stream) {
    int64_t n = a.numel();
    if (n == 0) return;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (detail::needs_broadcast_inplace(a, b)) {
        auto meta = detail::InplaceBroadcastMeta::create(a, b);
        if (a.dtype() == DType::Float32) {
            hipLaunchKernelGGL(mul_inplace_broadcast_kernel<float>, grid, block, 0, stream,
                a.data<float>(), b.data<float>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Float64) {
            hipLaunchKernelGGL(mul_inplace_broadcast_kernel<double>, grid, block, 0, stream,
                a.data<double>(), b.data<double>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Int32) {
            hipLaunchKernelGGL(mul_inplace_broadcast_kernel<int32_t>, grid, block, 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Int64) {
            hipLaunchKernelGGL(mul_inplace_broadcast_kernel<int64_t>, grid, block, 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Float16) {
            hipLaunchKernelGGL(mul_inplace_broadcast_kernel<__half>, grid, block, 0, stream,
                reinterpret_cast<__half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::BFloat16) {
            hipLaunchKernelGGL(mul_inplace_broadcast_kernel<hip_bfloat16>, grid, block, 0, stream,
                reinterpret_cast<hip_bfloat16*>(a.data<BFloat16>()),
                reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
                meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else {
            meta.free();
            throw std::runtime_error("mul_inplace operation unsupported dtype");
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        meta.free();
        return;
    }

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(mul_inplace_kernel_device<float>, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(mul_inplace_kernel_device<double>, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), n);
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(mul_inplace_kernel_device<int32_t>, grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(mul_inplace_kernel_device<int64_t>, grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(mul_inplace_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(mul_inplace_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("mul_inplace operation unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
}

void div_inplace_kernel(Tensor& a, const Tensor& b, hipStream_t stream) {
    int64_t n = a.numel();
    if (n == 0) return;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (detail::needs_broadcast_inplace(a, b)) {
        auto meta = detail::InplaceBroadcastMeta::create(a, b);
        if (a.dtype() == DType::Float32) {
            hipLaunchKernelGGL(div_inplace_broadcast_kernel<float>, grid, block, 0, stream,
                a.data<float>(), b.data<float>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Float64) {
            hipLaunchKernelGGL(div_inplace_broadcast_kernel<double>, grid, block, 0, stream,
                a.data<double>(), b.data<double>(), meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else if (a.dtype() == DType::Float16) {
            hipLaunchKernelGGL(div_inplace_broadcast_kernel<__half>, grid, block, 0, stream,
                reinterpret_cast<__half*>(a.data<Float16>()),
                reinterpret_cast<const __half*>(b.data<Float16>()),
                meta.d_strides_b, meta.d_a_shape, meta.ndim, n);
        } else {
            meta.free();
            throw std::runtime_error("div_inplace operation unsupported dtype");
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        meta.free();
        return;
    }

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(div_inplace_kernel_device<float>, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(div_inplace_kernel_device<double>, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(div_inplace_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<__half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()), n);
    } else {
        throw std::runtime_error("div_inplace operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
}

// ============================================================================
// Dot Product Kernel Host Wrapper
// ============================================================================

/**
 * @brief Final reduction kernel — reduces partial sums from dot product blocks
 * @tparam T Data type
 * @param partial Partial sums array (one per block from dot_kernel_device)
 * @param output Single output scalar
 * @param num_blocks Number of partial sums to reduce
 *
 * Launched with 1 block of 256 threads. Uses shared memory reduction
 * following the same pattern as reduction.hip.cpp sum_reduce_kernel.
 */
template<typename T>
__global__ void final_reduce_kernel(const T* partial, T* output, int num_blocks) {
    __shared__ T sdata[256];
    int tid = threadIdx.x;

    // Grid-stride accumulation of partial sums
    T sum = T(0);
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block-level reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
    }
}

auto dot_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.numel() != b.numel()) {
        throw std::invalid_argument("Tensor sizes must match for dot product");
    }

    int64_t n = a.numel();
    Tensor result({}, a.dtype(), a.device());  // Scalar output

    if (n == 0) {
        // Return zero scalar
        if (a.dtype() == DType::Float32) {
            float zero = 0.0f;
            HIP_CHECK(hipMemcpy(result.data<float>(), &zero, sizeof(float), hipMemcpyHostToDevice));
        } else if (a.dtype() == DType::Float64) {
            double zero = 0.0;
            HIP_CHECK(hipMemcpy(result.data<double>(), &zero, sizeof(double), hipMemcpyHostToDevice));
        } else if (a.dtype() == DType::Float16) {
            Float16 zero = Float16(0.0f);
            HIP_CHECK(hipMemcpy(result.data<Float16>(), &zero, sizeof(Float16), hipMemcpyHostToDevice));
        }
        return result;
    }

    const int block_size = 256;
    int num_blocks = std::min((n + block_size - 1) / block_size, static_cast<int64_t>(1024));

    if (a.dtype() == DType::Float32) {
        // Allocate partial sums
        float* d_partial;
        HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(float)));

        hipLaunchKernelGGL(dot_kernel_device<float>, dim3(num_blocks), dim3(block_size), 0, stream,
            a.data<float>(), b.data<float>(), d_partial, n);
        HIP_CHECK(hipGetLastError());

        // Final reduction on GPU — single block reduces partial sums
        hipLaunchKernelGGL(final_reduce_kernel<float>, dim3(1), dim3(block_size), 0, stream,
            d_partial, result.data<float>(), num_blocks);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipFree(d_partial));
    } else if (a.dtype() == DType::Float64) {
        double* d_partial;
        HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(double)));

        hipLaunchKernelGGL(dot_kernel_device<double>, dim3(num_blocks), dim3(block_size), 0, stream,
            a.data<double>(), b.data<double>(), d_partial, n);
        HIP_CHECK(hipGetLastError());

        // Final reduction on GPU — single block reduces partial sums
        hipLaunchKernelGGL(final_reduce_kernel<double>, dim3(1), dim3(block_size), 0, stream,
            d_partial, result.data<double>(), num_blocks);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipFree(d_partial));
    } else if (a.dtype() == DType::Float16) {
        // Compute dot product in float for accuracy, then store as Float16
        float* d_partial;
        HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(float)));

        hipLaunchKernelGGL(dot_kernel_f16, dim3(num_blocks), dim3(block_size), 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            d_partial, n);
        HIP_CHECK(hipGetLastError());

        // Final reduction on GPU in float, then convert to Float16
        // Allocate a single float on device for the reduced result
        float* d_sum;
        HIP_CHECK(hipMalloc(&d_sum, sizeof(float)));
        hipLaunchKernelGGL(final_reduce_kernel<float>, dim3(1), dim3(block_size), 0, stream,
            d_partial, d_sum, num_blocks);
        HIP_CHECK(hipGetLastError());

        // Copy reduced float sum to host, convert to Float16, write back
        float h_sum;
        HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost));
        Float16 sum_f16 = Float16(h_sum);
        HIP_CHECK(hipMemcpy(result.data<Float16>(), &sum_f16, sizeof(Float16), hipMemcpyHostToDevice));
        HIP_CHECK(hipFree(d_sum));
        HIP_CHECK(hipFree(d_partial));
    } else {
        throw std::runtime_error("dot operation only supports Float32, Float64, and Float16 dtypes");
    }

    return result;
}

// ============================================================================
// Fill Operations (zeros, ones, full)
// ============================================================================

/**
 * @brief Fill kernel - set all elements to a constant value
 * @tparam T Data type
 */
template<typename T>
__global__ void fill_kernel_device(T* output, T value, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = value;
    }
}

/**
 * @brief Fill kernel launcher - fills tensor with constant value
 * @param tensor Tensor to fill
 * @param value Fill value
 * @param stream HIP stream for asynchronous execution
 * @return Filled tensor
 */
auto fill_kernel(const Tensor& tensor, float value, hipStream_t stream) -> Tensor {
    int64_t n = tensor.numel();

    if (n == 0) {
        return tensor;
    }

    // Create a copy to modify
    auto result = tensor;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (tensor.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fill_kernel_device<float>, grid, block, 0, stream,
            result.data<float>(), static_cast<float>(value), n);
    } else if (tensor.dtype() == DType::Float64) {
        hipLaunchKernelGGL(fill_kernel_device<double>, grid, block, 0, stream,
            result.data<double>(), static_cast<double>(value), n);
    } else if (tensor.dtype() == DType::Int32) {
        hipLaunchKernelGGL(fill_kernel_device<int32_t>, grid, block, 0, stream,
            result.data<int32_t>(), static_cast<int32_t>(value), n);
    } else if (tensor.dtype() == DType::Int64) {
        hipLaunchKernelGGL(fill_kernel_device<int64_t>, grid, block, 0, stream,
            result.data<int64_t>(), static_cast<int64_t>(value), n);
    } else if (tensor.dtype() == DType::Float16) {
        hipLaunchKernelGGL(fill_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(result.data<Float16>()), __float2half(static_cast<float>(value)), n);
    } else if (tensor.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(fill_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), hip_bfloat16(static_cast<float>(value)), n);
    } else {
        throw std::runtime_error("Unsupported dtype for fill operation");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Zeros kernel launcher - creates tensor filled with zeros
 * @param shape Tensor shape
 * @param dtype Data type
 * @param device Device (ROCm)
 * @param stream HIP stream for asynchronous execution
 * @return Zero tensor
 */
auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (dtype == DType::Float32) {
        hipLaunchKernelGGL(fill_kernel_device<float>, grid, block, 0, stream,
            result.data<float>(), 0.0f, n);
    } else if (dtype == DType::Float64) {
        hipLaunchKernelGGL(fill_kernel_device<double>, grid, block, 0, stream,
            result.data<double>(), 0.0, n);
    } else if (dtype == DType::Int32) {
        hipLaunchKernelGGL(fill_kernel_device<int32_t>, grid, block, 0, stream,
            result.data<int32_t>(), static_cast<int32_t>(0), n);
    } else if (dtype == DType::Int64) {
        hipLaunchKernelGGL(fill_kernel_device<int64_t>, grid, block, 0, stream,
            result.data<int64_t>(), static_cast<int64_t>(0), n);
    } else if (dtype == DType::Bool) {
        hipLaunchKernelGGL(fill_kernel_device<bool>, grid, block, 0, stream,
            result.data<bool>(), false, n);
    } else if (dtype == DType::UInt8) {
        hipLaunchKernelGGL(fill_kernel_device<uint8_t>, grid, block, 0, stream,
            result.data<uint8_t>(), static_cast<uint8_t>(0), n);
    } else if (dtype == DType::Int8) {
        hipLaunchKernelGGL(fill_kernel_device<int8_t>, grid, block, 0, stream,
            result.data<int8_t>(), static_cast<int8_t>(0), n);
    } else if (dtype == DType::Float16) {
        hipLaunchKernelGGL(fill_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(result.data<Float16>()), __float2half(0.0f), n);
    } else if (dtype == DType::BFloat16) {
        hipLaunchKernelGGL(fill_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), hip_bfloat16(0.0f), n);
    } else {
        throw std::runtime_error("Unsupported dtype for zeros operation");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Ones kernel launcher - creates tensor filled with ones
 * @param shape Tensor shape
 * @param dtype Data type
 * @param device Device (ROCm)
 * @param stream HIP stream for asynchronous execution
 * @return Ones tensor
 */
auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (dtype == DType::Float32) {
        hipLaunchKernelGGL(fill_kernel_device<float>, grid, block, 0, stream,
            result.data<float>(), 1.0f, n);
    } else if (dtype == DType::Float64) {
        hipLaunchKernelGGL(fill_kernel_device<double>, grid, block, 0, stream,
            result.data<double>(), 1.0, n);
    } else if (dtype == DType::Int32) {
        hipLaunchKernelGGL(fill_kernel_device<int32_t>, grid, block, 0, stream,
            result.data<int32_t>(), static_cast<int32_t>(1), n);
    } else if (dtype == DType::Int64) {
        hipLaunchKernelGGL(fill_kernel_device<int64_t>, grid, block, 0, stream,
            result.data<int64_t>(), static_cast<int64_t>(1), n);
    } else if (dtype == DType::Bool) {
        hipLaunchKernelGGL(fill_kernel_device<bool>, grid, block, 0, stream,
            result.data<bool>(), true, n);
    } else if (dtype == DType::UInt8) {
        hipLaunchKernelGGL(fill_kernel_device<uint8_t>, grid, block, 0, stream,
            result.data<uint8_t>(), static_cast<uint8_t>(1), n);
    } else if (dtype == DType::Int8) {
        hipLaunchKernelGGL(fill_kernel_device<int8_t>, grid, block, 0, stream,
            result.data<int8_t>(), static_cast<int8_t>(1), n);
    } else if (dtype == DType::Float16) {
        hipLaunchKernelGGL(fill_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(result.data<Float16>()), __float2half(1.0f), n);
    } else if (dtype == DType::BFloat16) {
        hipLaunchKernelGGL(fill_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), hip_bfloat16(1.0f), n);
    } else {
        throw std::runtime_error("Unsupported dtype for ones operation");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Full kernel launcher - creates tensor filled with specified value
 * @param shape Tensor shape
 * @param value Fill value
 * @param dtype Data type
 * @param device Device (ROCm)
 * @param stream HIP stream for asynchronous execution
 * @return Filled tensor
 */
auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, hipStream_t stream) -> Tensor {
    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (dtype == DType::Float32) {
        hipLaunchKernelGGL(fill_kernel_device<float>, grid, block, 0, stream,
            result.data<float>(), static_cast<float>(value), n);
    } else if (dtype == DType::Float64) {
        hipLaunchKernelGGL(fill_kernel_device<double>, grid, block, 0, stream,
            result.data<double>(), static_cast<double>(value), n);
    } else if (dtype == DType::Int32) {
        hipLaunchKernelGGL(fill_kernel_device<int32_t>, grid, block, 0, stream,
            result.data<int32_t>(), static_cast<int32_t>(value), n);
    } else if (dtype == DType::Int64) {
        hipLaunchKernelGGL(fill_kernel_device<int64_t>, grid, block, 0, stream,
            result.data<int64_t>(), static_cast<int64_t>(value), n);
    } else if (dtype == DType::Bool) {
        hipLaunchKernelGGL(fill_kernel_device<bool>, grid, block, 0, stream,
            result.data<bool>(), static_cast<bool>(value), n);
    } else if (dtype == DType::UInt8) {
        hipLaunchKernelGGL(fill_kernel_device<uint8_t>, grid, block, 0, stream,
            result.data<uint8_t>(), static_cast<uint8_t>(value), n);
    } else if (dtype == DType::Int8) {
        hipLaunchKernelGGL(fill_kernel_device<int8_t>, grid, block, 0, stream,
            result.data<int8_t>(), static_cast<int8_t>(value), n);
    } else if (dtype == DType::Float16) {
        hipLaunchKernelGGL(fill_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(result.data<Float16>()), __float2half(value), n);
    } else if (dtype == DType::BFloat16) {
        hipLaunchKernelGGL(fill_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), hip_bfloat16(value), n);
    } else {
        throw std::runtime_error("Unsupported dtype for full operation");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Random Number Generation (hipRAND)
// ============================================================================

#ifdef TENZOR_HAS_HIPRAND
/**
 * @brief Kernel to initialize hipRAND states
 * @param states Array of random states (one per thread)
 * @param seed Random seed
 * @param n Number of states to initialize
 *
 * Each thread gets a unique state initialized with different sequence number
 */
__global__ void init_hiprand_states(hiprandState* states, unsigned long long seed, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        // Each thread gets different seed, a different sequence number, no offset
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

/**
 * @brief Kernel for uniform random [0, 1) generation
 * @param output Output tensor data
 * @param states Random states
 * @param n Number of elements to generate
 */
__global__ void rand_kernel_device(float* output, hiprandState* states, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = hiprand_uniform(&states[idx]);
    }
}

/**
 * @brief Kernel for normal distribution N(0,1) generation
 * @param output Output tensor data
 * @param states Random states
 * @param n Number of elements to generate
 */
__global__ void randn_kernel_device(float* output, hiprandState* states, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = hiprand_normal(&states[idx]);
    }
}

/**
 * @brief Kernel for uniform random [0, 1) generation - Float64 version
 * @param output Output tensor data (double precision)
 * @param states Random states
 * @param n Number of elements to generate
 */
__global__ void rand_kernel_device_f64(double* output, hiprandState* states, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = hiprand_uniform_double(&states[idx]);
    }
}

/**
 * @brief Kernel for normal distribution N(0,1) generation - Float64 version
 * @param output Output tensor data (double precision)
 * @param states Random states
 * @param n Number of elements to generate
 */
__global__ void randn_kernel_device_f64(double* output, hiprandState* states, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = hiprand_normal_double(&states[idx]);
    }
}

/**
 * @brief Kernel for uniform random [0, 1) generation - Float16 version
 * @param output Output tensor data (half precision)
 * @param states Random states
 * @param n Number of elements to generate
 */
__global__ void rand_kernel_device_f16(__half* output, hiprandState* states, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(hiprand_uniform(&states[idx]));
    }
}

/**
 * @brief Kernel for normal distribution N(0,1) generation - Float16 version
 * @param output Output tensor data (half precision)
 * @param states Random states
 * @param n Number of elements to generate
 */
__global__ void randn_kernel_device_f16(__half* output, hiprandState* states, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(hiprand_normal(&states[idx]));
    }
}

/**
 * @brief Rand kernel launcher - uniform random [0, 1)
 * @param shape Tensor shape
 * @param dtype Data type (Float32, Float64, or Float16)
 * @param device Device (ROCm)
 * @param stream HIP stream for asynchronous execution
 * @return Random tensor with uniform distribution
 */
auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor {
#ifdef TENZOR_HAS_HIPRAND
    // BFloat16: generate as Float32 and convert
    if (dtype == DType::BFloat16) {
        auto result_f32 = rand_kernel(shape, DType::Float32, device, stream);
        return result_f32.to(DType::BFloat16);
    }

    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16) {
        throw std::runtime_error("rand operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    // Allocate hipRAND states
    hiprandState* d_states;
    HIP_CHECK(hipMalloc(&d_states, n * sizeof(hiprandState)));

    // Initialize states with timestamp-based seed for randomness
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    hipLaunchKernelGGL(init_hiprand_states, grid, block, 0, stream, d_states, seed, n);
    HIP_CHECK(hipGetLastError());

    if (dtype == DType::Float32) {
        // Generate uniform random numbers
        hipLaunchKernelGGL(rand_kernel_device, grid, block, 0, stream,
            result.data<float>(), d_states, n);
        HIP_CHECK(hipGetLastError());
    } else if (dtype == DType::Float64) {
        // Generate double-precision uniform random numbers directly
        hipLaunchKernelGGL(rand_kernel_device_f64, grid, block, 0, stream,
            result.data<double>(), d_states, n);
        HIP_CHECK(hipGetLastError());
    } else if (dtype == DType::Float16) {
        // Generate half-precision uniform random numbers (generate as float, convert to half)
        hipLaunchKernelGGL(rand_kernel_device_f16, grid, block, 0, stream,
            reinterpret_cast<__half*>(result.data<Float16>()), d_states, n);
        HIP_CHECK(hipGetLastError());
    }

    // Cleanup
    HIP_CHECK(hipFree(d_states));

    return result;
#else
    throw std::runtime_error("rand operation requires hipRAND library. Please install ROCm hipRAND.");
#endif
}

/**
 * @brief Randn kernel launcher - normal distribution N(0,1)
 * @param shape Tensor shape
 * @param dtype Data type (Float32, Float64, or Float16)
 * @param device Device (ROCm)
 * @param stream HIP stream for asynchronous execution
 * @return Random tensor with normal distribution
 */
auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor {
#ifdef TENZOR_HAS_HIPRAND
    // BFloat16: generate as Float32 and convert
    if (dtype == DType::BFloat16) {
        auto result_f32 = randn_kernel(shape, DType::Float32, device, stream);
        return result_f32.to(DType::BFloat16);
    }

    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16) {
        throw std::runtime_error("randn operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    Tensor result(shape, dtype, device);
    int64_t n = result.numel();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    // Allocate hipRAND states
    hiprandState* d_states;
    HIP_CHECK(hipMalloc(&d_states, n * sizeof(hiprandState)));

    // Initialize states with timestamp-based seed
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    hipLaunchKernelGGL(init_hiprand_states, grid, block, 0, stream, d_states, seed, n);
    HIP_CHECK(hipGetLastError());

    if (dtype == DType::Float32) {
        // Generate normal random numbers
        hipLaunchKernelGGL(randn_kernel_device, grid, block, 0, stream,
            result.data<float>(), d_states, n);
        HIP_CHECK(hipGetLastError());
    } else if (dtype == DType::Float64) {
        // Generate double-precision normal random numbers directly
        hipLaunchKernelGGL(randn_kernel_device_f64, grid, block, 0, stream,
            result.data<double>(), d_states, n);
        HIP_CHECK(hipGetLastError());
    } else if (dtype == DType::Float16) {
        // Generate half-precision normal random numbers (generate as float, convert to half)
        hipLaunchKernelGGL(randn_kernel_device_f16, grid, block, 0, stream,
            reinterpret_cast<__half*>(result.data<Float16>()), d_states, n);
        HIP_CHECK(hipGetLastError());
    }

    // Cleanup
    HIP_CHECK(hipFree(d_states));

    return result;
#else
    throw std::runtime_error("randn operation requires hipRAND library. Please install ROCm hipRAND.");
#endif
}

// Device kernel for randint generation
template<typename T>
__global__ void randint_kernel_device(T* output, hiprandState* states, int64_t n, int64_t low, int64_t high) {
    HIP_KERNEL_LOOP(idx, n) {
        float r = hiprand_uniform(&states[idx]);
        int64_t range = high - low;
        int64_t val = low + static_cast<int64_t>(r * static_cast<float>(range));
        if (val >= high) val = high - 1;
        output[idx] = static_cast<T>(val);
    }
}

auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape,
                    DType dtype, Device device, hipStream_t stream) -> Tensor {
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

    // Allocate hipRAND states
    hiprandState* d_states;
    HIP_CHECK(hipMalloc(&d_states, n * sizeof(hiprandState)));

    // Thread-safe seed generation
    static std::atomic<uint64_t> seed_counter{0};
    auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto counter = seed_counter.fetch_add(1, std::memory_order_relaxed);
    uint64_t seed = time_seed ^ (thread_id << 32) ^ counter;

    hipLaunchKernelGGL(init_hiprand_states, grid, block, 0, stream, d_states, seed, n);
    HIP_CHECK(hipGetLastError());

    if (dtype == DType::Int32) {
        hipLaunchKernelGGL(randint_kernel_device<int32_t>, grid, block, 0, stream,
            result.data<int32_t>(), d_states, n, low, high);
        HIP_CHECK(hipGetLastError());
    } else {
        hipLaunchKernelGGL(randint_kernel_device<int64_t>, grid, block, 0, stream,
            result.data<int64_t>(), d_states, n, low, high);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipFree(d_states));
    return result;
}

#endif // TENZOR_HAS_HIPRAND (end of all hipRAND code)

// ============================================================================
// Creation Operations
// ============================================================================

// Arange kernel - generates values from start to end with given step
template<typename T>
__global__ void arange_kernel_impl(T* output, T start, T step, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = start + static_cast<T>(idx) * step;
    }
}

auto arange_kernel(double start, double end, double step, DType dtype, Device device, hipStream_t stream) -> Tensor {
    // Calculate number of elements
    int64_t n = static_cast<int64_t>(std::ceil((end - start) / step));
    if (n < 0) n = 0;

    Tensor result({n}, dtype, device);
    if (n == 0) return result;

    dim3 block, grid;
    compute_launch_config_1d(n, grid, block);

    switch (dtype) {
        case DType::Float32:
            hipLaunchKernelGGL(arange_kernel_impl<float>, grid, block, 0, stream,
                result.data<float>(), static_cast<float>(start), static_cast<float>(step), n);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(arange_kernel_impl<double>, grid, block, 0, stream,
                result.data<double>(), start, step, n);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(arange_kernel_impl<int32_t>, grid, block, 0, stream,
                result.data<int32_t>(), static_cast<int32_t>(start), static_cast<int32_t>(step), n);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(arange_kernel_impl<int64_t>, grid, block, 0, stream,
                result.data<int64_t>(), static_cast<int64_t>(start), static_cast<int64_t>(step), n);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(arange_kernel_impl<__half>, grid, block, 0, stream,
                reinterpret_cast<__half*>(result.data<Float16>()), __float2half(static_cast<float>(start)), __float2half(static_cast<float>(step)), n);
            break;
        case DType::Int8:
            hipLaunchKernelGGL(arange_kernel_impl<int8_t>, grid, block, 0, stream,
                result.data<int8_t>(), static_cast<int8_t>(start), static_cast<int8_t>(step), n);
            break;
        default:
            throw std::runtime_error("arange_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// Linspace kernel - generates n evenly spaced values from start to end
template<typename T>
__global__ void linspace_kernel_impl(T* output, T start, T step, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = start + static_cast<T>(idx) * step;
    }
}

auto linspace_kernel(double start, double end, int64_t steps, DType dtype, Device device, hipStream_t stream) -> Tensor {
    if (steps < 0) {
        throw std::runtime_error("linspace_kernel: steps must be non-negative");
    }

    Tensor result({steps}, dtype, device);
    if (steps == 0) return result;

    double step = (steps > 1) ? (end - start) / (steps - 1) : 0.0;

    dim3 block, grid;
    compute_launch_config_1d(steps, grid, block);

    switch (dtype) {
        case DType::Float32:
            hipLaunchKernelGGL(linspace_kernel_impl<float>, grid, block, 0, stream,
                result.data<float>(), static_cast<float>(start), static_cast<float>(step), steps);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(linspace_kernel_impl<double>, grid, block, 0, stream,
                result.data<double>(), start, step, steps);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(linspace_kernel_impl<__half>, grid, block, 0, stream,
                reinterpret_cast<__half*>(result.data<Float16>()),
                __float2half(static_cast<float>(start)),
                __float2half(static_cast<float>(step)), steps);
            break;
        case DType::BFloat16:
            hipLaunchKernelGGL(linspace_kernel_impl<hip_bfloat16>, grid, block, 0, stream,
                reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()),
                hip_bfloat16(static_cast<float>(start)),
                hip_bfloat16(static_cast<float>(step)), steps);
            break;
        default:
            throw std::runtime_error("linspace_kernel: only Float32, Float64, Float16, BFloat16 supported");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// Eye kernel - generates identity matrix
template<typename T>
__global__ void eye_kernel_impl(T* output, int64_t n, int64_t m, int64_t k) {
    int64_t total = n * m;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t row = idx / m;
        int64_t col = idx % m;
        // Set 1 on the k-th diagonal (k=0 is main diagonal)
        output[idx] = (col - row == k) ? static_cast<T>(1) : static_cast<T>(0);
    }
}

auto eye_kernel(int64_t n, int64_t m, int64_t k, DType dtype, Device device, hipStream_t stream) -> Tensor {
    if (m <= 0) m = n;  // Default to square matrix

    Tensor result({n, m}, dtype, device);
    int64_t total = n * m;

    if (total == 0) return result;

    dim3 block, grid;
    compute_launch_config_1d(total, grid, block);

    switch (dtype) {
        case DType::Float32:
            hipLaunchKernelGGL(eye_kernel_impl<float>, grid, block, 0, stream,
                result.data<float>(), n, m, k);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(eye_kernel_impl<double>, grid, block, 0, stream,
                result.data<double>(), n, m, k);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(eye_kernel_impl<int32_t>, grid, block, 0, stream,
                result.data<int32_t>(), n, m, k);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(eye_kernel_impl<int64_t>, grid, block, 0, stream,
                result.data<int64_t>(), n, m, k);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(eye_kernel_impl<__half>, grid, block, 0, stream,
                reinterpret_cast<__half*>(result.data<Float16>()), n, m, k);
            break;
        case DType::Int8:
            hipLaunchKernelGGL(eye_kernel_impl<int8_t>, grid, block, 0, stream,
                result.data<int8_t>(), n, m, k);
            break;
        default:
            throw std::runtime_error("eye_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// CumSum kernel — inclusive prefix sum along a dimension
// ============================================================================

template<typename T>
__global__ void extract_strided_slice_kernel(const T* __restrict__ input, T* __restrict__ output,
                                              int64_t dim_size, int64_t inner_size,
                                              int64_t outer, int64_t inner)
{
    HIP_KERNEL_LOOP(i, dim_size) {
        output[i] = input[outer * dim_size * inner_size + i * inner_size + inner];
    }
}

template<typename T>
__global__ void scatter_strided_slice_kernel(const T* __restrict__ input, T* __restrict__ output,
                                              int64_t dim_size, int64_t inner_size,
                                              int64_t outer, int64_t inner)
{
    HIP_KERNEL_LOOP(i, dim_size) {
        output[outer * dim_size * inner_size + i * inner_size + inner] = input[i];
    }
}

template<typename T>
static void cumsum_slice_hipcub(const T* d_in, T* d_out, int64_t n, hipStream_t stream)
{
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    hipcub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out,
                                     static_cast<int>(n), stream);
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    hipcub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out,
                                     static_cast<int>(n), stream);
    HIP_CHECK(hipFree(d_temp));
}

auto cumsum_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input.dtype();
    const auto device = input.device();

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    auto launch = [&]<typename T>() {
        if (inner_size == 1) {
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                const T* d_in = input_cont.data<T>() + outer * dim_size;
                T* d_out = output.data<T>() + outer * dim_size;
                cumsum_slice_hipcub<T>(d_in, d_out, dim_size, stream);
            }
        } else {
            T* d_slice_in = nullptr;
            T* d_slice_out = nullptr;
            HIP_CHECK(hipMalloc(&d_slice_in, dim_size * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_slice_out, dim_size * sizeof(T)));

            dim3 grid, block;
            compute_launch_config_1d(dim_size, grid, block);

            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    hipLaunchKernelGGL(extract_strided_slice_kernel<T>,
                        grid, block, 0, stream,
                        input_cont.data<T>(), d_slice_in, dim_size, inner_size, outer, inner);
                    HIP_CHECK(hipGetLastError());
                    cumsum_slice_hipcub<T>(d_slice_in, d_slice_out, dim_size, stream);
                    hipLaunchKernelGGL(scatter_strided_slice_kernel<T>,
                        grid, block, 0, stream,
                        d_slice_out, output.data<T>(), dim_size, inner_size, outer, inner);
                    HIP_CHECK(hipGetLastError());
                }
            }

            HIP_CHECK(hipFree(d_slice_in));
            HIP_CHECK(hipFree(d_slice_out));
        }
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        case DType::Int32:   launch.template operator()<int32_t>(); break;
        case DType::Int64:   launch.template operator()<int64_t>(); break;
        default: throw std::runtime_error("cumsum ROCm: unsupported dtype");
    }

    return output;
}

// ============================================================================
// CumProd kernel — inclusive prefix product along a dimension
// ============================================================================

struct HipMultOp {
    template<typename T>
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a * b; }
};

template<typename T>
static void cumprod_slice_hipcub(const T* d_in, T* d_out, int64_t n, hipStream_t stream)
{
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    hipcub::DeviceScan::InclusiveScan(d_temp, temp_bytes, d_in, d_out,
                                      HipMultOp(), static_cast<int>(n), stream);
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    hipcub::DeviceScan::InclusiveScan(d_temp, temp_bytes, d_in, d_out,
                                      HipMultOp(), static_cast<int>(n), stream);
    HIP_CHECK(hipFree(d_temp));
}

auto cumprod_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input.dtype();
    const auto device = input.device();

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    auto launch = [&]<typename T>() {
        if (inner_size == 1) {
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                const T* d_in = input_cont.data<T>() + outer * dim_size;
                T* d_out = output.data<T>() + outer * dim_size;
                cumprod_slice_hipcub<T>(d_in, d_out, dim_size, stream);
            }
        } else {
            T* d_slice_in = nullptr;
            T* d_slice_out = nullptr;
            HIP_CHECK(hipMalloc(&d_slice_in, dim_size * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_slice_out, dim_size * sizeof(T)));

            dim3 grid, block;
            compute_launch_config_1d(dim_size, grid, block);

            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    hipLaunchKernelGGL(extract_strided_slice_kernel<T>,
                        grid, block, 0, stream,
                        input_cont.data<T>(), d_slice_in, dim_size, inner_size, outer, inner);
                    HIP_CHECK(hipGetLastError());
                    cumprod_slice_hipcub<T>(d_slice_in, d_slice_out, dim_size, stream);
                    hipLaunchKernelGGL(scatter_strided_slice_kernel<T>,
                        grid, block, 0, stream,
                        d_slice_out, output.data<T>(), dim_size, inner_size, outer, inner);
                    HIP_CHECK(hipGetLastError());
                }
            }

            HIP_CHECK(hipFree(d_slice_in));
            HIP_CHECK(hipFree(d_slice_out));
        }
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        case DType::Int32:   launch.template operator()<int32_t>(); break;
        case DType::Int64:   launch.template operator()<int64_t>(); break;
        default: throw std::runtime_error("cumprod ROCm: unsupported dtype");
    }

    return output;
}

// ============================================================================
// HasInfNan kernel — check if tensor contains inf or nan
// ============================================================================

template<typename T>
__global__ void check_inf_nan_kernel(const T* data, int64_t n, int* result) {
    HIP_KERNEL_LOOP(idx, n) {
        T val = data[idx];
        if (isinf(static_cast<float>(val)) || isnan(static_cast<float>(val))) {
            atomicExch(result, 1);
        }
    }
}

// Float64 specialization — use double-precision isinf/isnan
template<>
__global__ void check_inf_nan_kernel<double>(const double* data, int64_t n, int* result) {
    HIP_KERNEL_LOOP(idx, n) {
        double val = data[idx];
        if (isinf(val) || isnan(val)) {
            atomicExch(result, 1);
        }
    }
}

auto has_inf_nan_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    const int64_t numel = input.numel();

    // Helper to create a Bool scalar tensor on the target device
    auto make_bool_scalar = [](bool value, Device device) -> Tensor {
        Tensor result({}, DType::Bool, Device::cpu());
        result.data<bool>()[0] = value;
        return result.to(device);
    };

    if (numel == 0) {
        return make_bool_scalar(false, input.device());
    }

    // Allocate device flag
    int* d_flag = nullptr;
    HIP_CHECK(hipMalloc(&d_flag, sizeof(int)));
    HIP_CHECK(hipMemsetAsync(d_flag, 0, sizeof(int), stream));

    // Handle BFloat16/Float16 by casting to Float32
    Tensor scan = input;
    if (scan.dtype() == DType::BFloat16 || scan.dtype() == DType::Float16) {
        scan = scan.to(DType::Float32);
    }

    dim3 grid, block;
    compute_launch_config_1d(numel, grid, block);

    switch (scan.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(check_inf_nan_kernel<float>,
                grid, block, 0, stream,
                scan.data<float>(), numel, d_flag);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(check_inf_nan_kernel<double>,
                grid, block, 0, stream,
                scan.data<double>(), numel, d_flag);
            break;
        default:
            // Integer types can't have inf/nan
            HIP_CHECK(hipFree(d_flag));
            return make_bool_scalar(false, input.device());
    }
    HIP_CHECK(hipGetLastError());

    int h_flag = 0;
    HIP_CHECK(hipMemcpyAsync(&h_flag, d_flag, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_flag));

    return make_bool_scalar(h_flag != 0, input.device());
}

// ============================================================================
// Extended Math Kernels (log2, log10, log1p, exp2, expm1, erf, erfc)
// ============================================================================

__global__ void log2_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = log2f(input[idx]);
    }
}

__global__ void log2_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = log2(input[idx]);
    }
}

__global__ void log2_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(log2f(__half2float(input[idx])));
    }
}

__global__ void log10_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = log10f(input[idx]);
    }
}

__global__ void log10_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = log10(input[idx]);
    }
}

__global__ void log10_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(log10f(__half2float(input[idx])));
    }
}

__global__ void log1p_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = log1pf(input[idx]);
    }
}

__global__ void log1p_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = log1p(input[idx]);
    }
}

__global__ void log1p_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(log1pf(__half2float(input[idx])));
    }
}

__global__ void exp2_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = exp2f(input[idx]);
    }
}

__global__ void exp2_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = exp2(input[idx]);
    }
}

__global__ void exp2_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(exp2f(__half2float(input[idx])));
    }
}

__global__ void expm1_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = expm1f(input[idx]);
    }
}

__global__ void expm1_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = expm1(input[idx]);
    }
}

__global__ void expm1_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(expm1f(__half2float(input[idx])));
    }
}

__global__ void erf_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = erff(input[idx]);
    }
}

__global__ void erf_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = erf(input[idx]);
    }
}

__global__ void erf_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(erff(__half2float(input[idx])));
    }
}

__global__ void erfc_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = erfcf(input[idx]);
    }
}

__global__ void erfc_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = erfc(input[idx]);
    }
}

__global__ void erfc_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = __float2half(erfcf(__half2float(input[idx])));
    }
}

// ============================================================================
// Bool Predicate Kernels (isnan, isinf, isfinite)
// ============================================================================

__global__ void isnan_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(isnan(input[idx]) ? 1 : 0);
    }
}

__global__ void isnan_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(isnan(input[idx]) ? 1 : 0);
    }
}

__global__ void isnan_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        output[idx] = static_cast<uint8_t>(isnan(val) ? 1 : 0);
    }
}

__global__ void isinf_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(isinf(input[idx]) ? 1 : 0);
    }
}

__global__ void isinf_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(isinf(input[idx]) ? 1 : 0);
    }
}

__global__ void isinf_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        output[idx] = static_cast<uint8_t>(isinf(val) ? 1 : 0);
    }
}

__global__ void isfinite_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(isfinite(input[idx]) ? 1 : 0);
    }
}

__global__ void isfinite_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(isfinite(input[idx]) ? 1 : 0);
    }
}

__global__ void isfinite_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = __half2float(input[idx]);
        output[idx] = static_cast<uint8_t>(isfinite(val) ? 1 : 0);
    }
}

// ============================================================================
// Binary Math Kernels (atan2, fmod, remainder)
// ============================================================================

__global__ void atan2_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = atan2f(a[idx], b[idx]);
    }
}

__global__ void atan2_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = atan2(a[idx], b[idx]);
    }
}

__global__ void atan2_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(atan2f(va, vb));
    }
}

__global__ void fmod_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = fmodf(a[idx], b[idx]);
    }
}

__global__ void fmod_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = fmod(a[idx], b[idx]);
    }
}

__global__ void fmod_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(fmodf(va, vb));
    }
}

__global__ void remainder_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = remainderf(a[idx], b[idx]);
    }
}

__global__ void remainder_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = remainder(a[idx], b[idx]);
    }
}

__global__ void remainder_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(remainderf(va, vb));
    }
}

// ============================================================================
// Lerp Kernel (ternary: a + weight * (b - a))
// ============================================================================

__global__ void lerp_kernel_f32(const float* a, const float* b, const float* weight, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = a[idx] + weight[idx] * (b[idx] - a[idx]);
    }
}

__global__ void lerp_kernel_f64(const double* a, const double* b, const double* weight, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = a[idx] + weight[idx] * (b[idx] - a[idx]);
    }
}

__global__ void lerp_kernel_f16(const __half* a, const __half* b, const __half* weight, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        float vw = __half2float(weight[idx]);
        output[idx] = __float2half(va + vw * (vb - va));
    }
}

// ============================================================================
// Logical Kernels (logical_and, logical_or, logical_not, logical_xor)
// ============================================================================

template<typename T>
__global__ void logical_and_kernel_device(const T* a, const T* b, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((a[idx] != T(0)) && (b[idx] != T(0)) ? 1 : 0);
    }
}

template<typename T>
__global__ void logical_or_kernel_device(const T* a, const T* b, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((a[idx] != T(0)) || (b[idx] != T(0)) ? 1 : 0);
    }
}

template<typename T>
__global__ void logical_not_kernel_device(const T* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(input[idx] == T(0) ? 1 : 0);
    }
}

template<typename T>
__global__ void logical_xor_kernel_device(const T* a, const T* b, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        bool ba = (a[idx] != T(0));
        bool bb = (b[idx] != T(0));
        output[idx] = static_cast<uint8_t>(ba != bb ? 1 : 0);
    }
}

__global__ void logical_and_kernel_f16(const __half* a, const __half* b, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((__half2float(a[idx]) != 0.0f) && (__half2float(b[idx]) != 0.0f) ? 1 : 0);
    }
}

__global__ void logical_or_kernel_f16(const __half* a, const __half* b, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((__half2float(a[idx]) != 0.0f) || (__half2float(b[idx]) != 0.0f) ? 1 : 0);
    }
}

__global__ void logical_not_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(__half2float(input[idx]) == 0.0f ? 1 : 0);
    }
}

__global__ void logical_xor_kernel_f16(const __half* a, const __half* b, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        bool ba = (__half2float(a[idx]) != 0.0f);
        bool bb = (__half2float(b[idx]) != 0.0f);
        output[idx] = static_cast<uint8_t>(ba != bb ? 1 : 0);
    }
}

// ============================================================================
// Element-wise Min/Max Kernels
// ============================================================================

__global__ void minimum_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = fminf(a[idx], b[idx]);
    }
}

__global__ void minimum_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = fmin(a[idx], b[idx]);
    }
}

__global__ void minimum_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(fminf(va, vb));
    }
}

__global__ void maximum_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = fmaxf(a[idx], b[idx]);
    }
}

__global__ void maximum_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = fmax(a[idx], b[idx]);
    }
}

__global__ void maximum_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(fmaxf(va, vb));
    }
}

// ============================================================================
// Host Wrappers: Extended Math Unary Operations
// ============================================================================

auto log2_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(log2_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(log2_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(log2_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("log2 operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto log10_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(log10_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(log10_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(log10_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("log10 operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto log1p_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(log1p_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(log1p_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(log1p_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("log1p operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto exp2_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(exp2_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(exp2_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(exp2_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("exp2 operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto expm1_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(expm1_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(expm1_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(expm1_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("expm1 operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto erf_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(erf_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(erf_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(erf_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("erf operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto erfc_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(erfc_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(erfc_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(erfc_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("erfc operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Host Wrappers: Bool Predicate Operations
// ============================================================================

auto isnan_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(isnan_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(isnan_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(isnan_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            result.data<uint8_t>(), n);
    } else {
        // Integer types cannot have NaN - return all false
        HIP_CHECK(hipMemsetAsync(result.data<uint8_t>(), 0, n * sizeof(uint8_t), stream));
        return result;
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto isinf_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(isinf_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(isinf_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(isinf_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            result.data<uint8_t>(), n);
    } else {
        // Integer types cannot have Inf - return all false
        HIP_CHECK(hipMemsetAsync(result.data<uint8_t>(), 0, n * sizeof(uint8_t), stream));
        return result;
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto isfinite_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(isfinite_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(isfinite_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(isfinite_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            result.data<uint8_t>(), n);
    } else {
        // Integer types are always finite - return all true
        HIP_CHECK(hipMemsetAsync(result.data<uint8_t>(), 1, n * sizeof(uint8_t), stream));
        return result;
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Host Wrappers: Binary Math Operations (atan2, fmod, remainder)
// ============================================================================

auto atan2_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("atan2: tensors must have the same dtype");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("atan2: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(atan2_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(atan2_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(atan2_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("atan2 operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto fmod_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("fmod: tensors must have the same dtype");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("fmod: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fmod_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(fmod_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(fmod_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("fmod operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto remainder_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("remainder: tensors must have the same dtype");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("remainder: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(remainder_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(remainder_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(remainder_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("remainder operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Host Wrapper: Lerp (ternary)
// ============================================================================

auto lerp_kernel(const Tensor& a, const Tensor& b, const Tensor& weight, hipStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype() || a.dtype() != weight.dtype()) {
        throw std::runtime_error("lerp: all tensors must have the same dtype");
    }
    if (a.numel() != b.numel() || a.numel() != weight.numel()) {
        throw std::runtime_error("lerp: all tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(lerp_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), weight.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(lerp_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), weight.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(lerp_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<const __half*>(weight.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("lerp operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Host Wrappers: Logical Operations
// ============================================================================

auto logical_and_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.numel() != b.numel()) {
        throw std::runtime_error("logical_and: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, DType::Bool, a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(logical_and_kernel_device<float>, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(logical_and_kernel_device<double>, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(logical_and_kernel_device<int32_t>, grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(logical_and_kernel_device<int64_t>, grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Bool) {
        hipLaunchKernelGGL(logical_and_kernel_device<bool>, grid, block, 0, stream,
            a.data<bool>(), b.data<bool>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(logical_and_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            result.data<uint8_t>(), n);
    } else {
        throw std::runtime_error("logical_and: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto logical_or_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.numel() != b.numel()) {
        throw std::runtime_error("logical_or: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, DType::Bool, a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(logical_or_kernel_device<float>, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(logical_or_kernel_device<double>, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(logical_or_kernel_device<int32_t>, grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(logical_or_kernel_device<int64_t>, grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Bool) {
        hipLaunchKernelGGL(logical_or_kernel_device<bool>, grid, block, 0, stream,
            a.data<bool>(), b.data<bool>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(logical_or_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            result.data<uint8_t>(), n);
    } else {
        throw std::runtime_error("logical_or: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto logical_not_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(logical_not_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(logical_not_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(logical_not_kernel_device<int32_t>, grid, block, 0, stream,
            input.data<int32_t>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(logical_not_kernel_device<int64_t>, grid, block, 0, stream,
            input.data<int64_t>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Bool) {
        hipLaunchKernelGGL(logical_not_kernel_device<bool>, grid, block, 0, stream,
            input.data<bool>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(logical_not_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            result.data<uint8_t>(), n);
    } else {
        throw std::runtime_error("logical_not: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto logical_xor_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.numel() != b.numel()) {
        throw std::runtime_error("logical_xor: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, DType::Bool, a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(logical_xor_kernel_device<float>, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(logical_xor_kernel_device<double>, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(logical_xor_kernel_device<int32_t>, grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(logical_xor_kernel_device<int64_t>, grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Bool) {
        hipLaunchKernelGGL(logical_xor_kernel_device<bool>, grid, block, 0, stream,
            a.data<bool>(), b.data<bool>(), result.data<uint8_t>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(logical_xor_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            result.data<uint8_t>(), n);
    } else {
        throw std::runtime_error("logical_xor: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Host Wrappers: Element-wise Min/Max
// ============================================================================

auto minimum_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("minimum: tensors must have the same dtype");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("minimum: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(minimum_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(minimum_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(minimum_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("minimum operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto maximum_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("maximum: tensors must have the same dtype");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("maximum: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(maximum_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(maximum_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(maximum_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("maximum operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// =========================================================================
// Complex Number Operations
// =========================================================================

// --- Conj ---
__global__ void conj_kernel_c64(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[2 * idx]     =  input[2 * idx];
        output[2 * idx + 1] = -input[2 * idx + 1];
    }
}
__global__ void conj_kernel_c128(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[2 * idx]     =  input[2 * idx];
        output[2 * idx + 1] = -input[2 * idx + 1];
    }
}

auto conj_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Complex64, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(conj_kernel_c64, grid, block, 0, stream,
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Complex128, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(conj_kernel_c128, grid, block, 0, stream,
            reinterpret_cast<const double*>(input.data_ptr()),
            reinterpret_cast<double*>(result.data_ptr()), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    // For real dtypes, conjugate is identity
    Tensor result(shape, input.dtype(), input.device());
    hipMemcpyAsync(result.data_ptr(), input.data_ptr(),
                   n * dtype_size(input.dtype()), hipMemcpyDeviceToDevice, stream);
    return result;
}

// --- Real ---
__global__ void real_kernel_c64(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = input[2 * idx];
    }
}
__global__ void real_kernel_c128(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = input[2 * idx];
    }
}

auto real_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(real_kernel_c64, grid, block, 0, stream,
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(real_kernel_c128, grid, block, 0, stream,
            reinterpret_cast<const double*>(input.data_ptr()),
            result.data<double>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    // For real dtypes, real() is identity
    Tensor result(shape, input.dtype(), input.device());
    hipMemcpyAsync(result.data_ptr(), input.data_ptr(),
                   n * dtype_size(input.dtype()), hipMemcpyDeviceToDevice, stream);
    return result;
}

// --- Imag ---
__global__ void imag_kernel_c64(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = input[2 * idx + 1];
    }
}
__global__ void imag_kernel_c128(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = input[2 * idx + 1];
    }
}

auto imag_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(imag_kernel_c64, grid, block, 0, stream,
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(imag_kernel_c128, grid, block, 0, stream,
            reinterpret_cast<const double*>(input.data_ptr()),
            result.data<double>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    // For real dtypes, imaginary part is zero
    Tensor result(shape, input.dtype(), input.device());
    hipMemsetAsync(result.data_ptr(), 0, n * dtype_size(input.dtype()), stream);
    return result;
}

// --- Angle ---
__global__ void angle_kernel_c64(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = atan2f(input[2 * idx + 1], input[2 * idx]);
    }
}
__global__ void angle_kernel_c128(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = atan2(input[2 * idx + 1], input[2 * idx]);
    }
}
__global__ void angle_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = atan2f(0.0f, input[idx]);
    }
}
__global__ void angle_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = atan2(0.0, input[idx]);
    }
}

auto angle_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        hipLaunchKernelGGL(angle_kernel_c64, grid, block, 0, stream,
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        hipLaunchKernelGGL(angle_kernel_c128, grid, block, 0, stream,
            reinterpret_cast<const double*>(input.data_ptr()),
            result.data<double>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Float32) {
        Tensor result(shape, DType::Float32, input.device());
        hipLaunchKernelGGL(angle_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Float64) {
        Tensor result(shape, DType::Float64, input.device());
        hipLaunchKernelGGL(angle_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    throw std::runtime_error("angle: unsupported dtype");
}

// --- Polar ---
__global__ void polar_kernel_f32(const float* abs_in, const float* angle_in,
                                  float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float r = abs_in[idx];
        float theta = angle_in[idx];
        output[2 * idx]     = r * cosf(theta);
        output[2 * idx + 1] = r * sinf(theta);
    }
}
__global__ void polar_kernel_f64(const double* abs_in, const double* angle_in,
                                  double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double r = abs_in[idx];
        double theta = angle_in[idx];
        output[2 * idx]     = r * cos(theta);
        output[2 * idx + 1] = r * sin(theta);
    }
}

auto polar_kernel(const Tensor& abs_t, const Tensor& angle_t, hipStream_t stream) -> Tensor {
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
        hipLaunchKernelGGL(polar_kernel_f32, grid, block, 0, stream,
            abs_t.data<float>(), angle_t.data<float>(),
            reinterpret_cast<float*>(result.data_ptr()), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (abs_t.dtype() == DType::Float64) {
        Tensor result(shape, DType::Complex128, abs_t.device());
        hipLaunchKernelGGL(polar_kernel_f64, grid, block, 0, stream,
            abs_t.data<double>(), angle_t.data<double>(),
            reinterpret_cast<double*>(result.data_ptr()), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    throw std::runtime_error("polar: only Float32 and Float64 inputs are supported");
}

} // namespace rocm
} // namespace tenzor
