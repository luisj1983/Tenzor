#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "../hip_buffer.hpp"  // RAII for HIP device memory (leak-safe on throw)
#include "tenzor/ops/creation.hpp"  // for tenzor::get_global_seed, complex

// Forward-declare the handful of public tensor ops used by the complex dot
// decomposition. We do NOT include <tenzor/ops/math.hpp> here: it declares
// tenzor::exp/cos/sin/hypot which would shadow the device-math functions of the
// same name called inside this file's __global__ kernels.
namespace tenzor {
auto real(const Tensor& input) -> Tensor;
auto imag(const Tensor& input) -> Tensor;
auto add(const Tensor& a, const Tensor& b) -> Tensor;
auto sub(const Tensor& a, const Tensor& b) -> Tensor;
}  // namespace tenzor
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#ifdef TENZOR_HAS_HIPRAND
#include <hiprand_kernel.h>
#endif
#include <hipcub/hipcub.hpp>
#include <cmath>
#include "rocm_nan_helpers.hip.h"  // F7/F8: IEEE-754 bit-pattern NaN check
#include "bfloat16_helpers.hpp"   // S.10 / R.11: f32_to_bf16_rne
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>
#include <type_traits>  // FF.6: std::is_same_v in logcumsumexp_hip_kernel guard
#include "fp16_saturate.h"

namespace tenzor {
// Forward declaration: tenzor::expand — broadcasts to a given shape.
// Pulling in the full transform.hpp would leak host-only names into device
// code and poison unqualified intrinsic calls.
auto expand(const Tensor& input, std::vector<int64_t> shape) -> Tensor;

namespace rocm {

// Forward declaration from transform.hip.cpp (needed for FP8 emulation)
auto cast_kernel(const Tensor& input, DType target_dtype, hipStream_t stream) -> Tensor;

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
 * @brief IEEE-754 division robust under -ffast-math.
 *
 * This TU is compiled with -ffast-math (see src/backends/rocm/CMakeLists.txt),
 * which implies -ffinite-math-only and -freciprocal-math. For Float64 that
 * breaks x/0: the divide is lowered to a software reciprocal (v_rcp_f64) plus
 * Newton-Raphson refinement, and rcp(0) yields a large *finite* approximation
 * instead of +Inf — so 1.0/0.0 came back finite (DivisionByZero/ROCm_Float64).
 * Float32 escapes this because v_rcp_f32(0) is +Inf in hardware.
 *
 * Equality against zero is still exact under fast-math, so we special-case a
 * zero divisor and emit the IEEE result as a raw bit pattern (which
 * -ffinite-math-only cannot fold away — same idiom as the NaN/Inf constants
 * elsewhere in this file). The fast hardware divide is kept for b != 0.
 *
 * Full IEEE contract for b == ±0:
 *   a == ±0 or a == NaN  -> NaN   (preserves the 0/0 → NaN that the old
 *                                  `if (b==0) return huge_val` shortcut broke)
 *   otherwise            -> ±Inf, sign = sign(a) XOR sign(b)
 */
__device__ inline double ieee_div(double a, double b) {
    if (b == 0.0) {
        const long long abits = __double_as_longlong(a);
        const long long amag  = abits & 0x7FFFFFFFFFFFFFFFLL;  // strip sign
        if (amag == 0LL || amag > 0x7FF0000000000000LL) {      // ±0 or NaN
            return __longlong_as_double(0x7FF8000000000000LL); // quiet NaN
        }
        const long long sign = (abits ^ __double_as_longlong(b))
                               & static_cast<long long>(0x8000000000000000ULL);
        return __longlong_as_double(sign | 0x7FF0000000000000LL);  // ±Inf
    }
    return a / b;
}
__device__ inline float ieee_div(float a, float b) {
    if (b == 0.0f) {
        const int abits = __float_as_int(a);
        const int amag  = abits & 0x7FFFFFFF;                  // strip sign
        if (amag == 0 || amag > 0x7F800000) {                 // ±0 or NaN
            return __int_as_float(0x7FC00000);                // quiet NaN
        }
        const int sign = (abits ^ __float_as_int(b)) & static_cast<int>(0x80000000U);
        return __int_as_float(sign | 0x7F800000);             // ±Inf
    }
    return a / b;
}

/**
 * @brief Division operation functor.
 *
 * Floating-point div uses ieee_div() (above) so IEEE 754 semantics hold even
 * under this TU's -ffast-math: x/0 → ±Inf for x ≠ 0, and 0/0 → NaN. The old
 * `if (b == 0) return huge_valf();` shortcut turned 0/0 into +Inf and broke
 * NaN_Propagation on ROCm (test_numerical_stability); ieee_div keeps 0/0 NaN.
 * Integer specializations below still need an explicit check because integer
 * div-by-zero is UB in C++.
 */
struct DivOp {
    template<typename T>
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return ieee_div(a, b);
        } else {
            return a / b;
        }
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
    // Floating types route through ieee_div() so x/0 → ±Inf and 0/0 → NaN
    // survive this TU's -ffast-math (which otherwise yields a finite value for
    // the Float64 reciprocal-refined divide). Integer T keeps plain divide
    // (int div-by-zero is UB and intentionally not special-cased here; the
    // int32/int64 DivOp specializations guard the broadcast path).
    HIP_KERNEL_LOOP(idx, n) {
        if constexpr (std::is_floating_point_v<T>) {
            c[idx] = ieee_div(a[idx], b[idx]);
        } else {
            // Guard integer divide-by-zero (UB on GPU) consistently with the
            // broadcast DivOp int specializations, which return 0 for a zero
            // divisor. The previous unguarded a/b made the same-shape path UB
            // while the broadcast path returned 0 — a non-deterministic divergence.
            T bd = b[idx];
            c[idx] = (bd == T(0)) ? T(0) : (a[idx] / bd);
        }
    }
}

/**
 * @brief Element-wise division kernel for half precision
 *
 * F8: emits NaN / ±Inf as explicit Float16 bit patterns rather than relying
 * on `tenzor::rocm::safe_f2h(NaN)` to forward the special value through. On some HIP
 * builds the conversion silently canonicalises NaN to a finite value (the
 * same root cause documented in `transform.hip.cpp::cast_to_f16_kernel`).
 *
 * Float divide returns NaN for 0/0 and ±Inf for x/0 with x ≠ 0 per IEEE 754;
 * the bit-pattern check below catches both cases and reroutes them to the
 * canonical Float16 NaN payload (0x7E00) or signed Inf (sign | 0x7C00).
 */
// Divide two Float16 values via Float32, emitting NaN / ±Inf as explicit
// Float16 bit patterns rather than relying on safe_f2h(NaN) to forward the
// special value through (some HIP builds silently canonicalise NaN to a finite
// value). 0/0 → quiet NaN (0x7E00), x/0 (x≠0) → signed Inf (sign | 0x7C00).
// Shared by both the same-shape and the broadcasting Float16 div kernels so the
// special-value semantics never diverge depending on whether broadcasting fired.
__device__ __forceinline__ __half f16_div_ieee(__half ha, __half hb) {
    float fa = tenzor::rocm::safe_h2f(ha);
    float fb = tenzor::rocm::safe_h2f(hb);
    float fv = fa / fb;
    unsigned int vb = __float_as_uint(fv);
    unsigned int exp  = (vb >> 23) & 0xFFu;
    unsigned int mant =  vb        & 0x7FFFFFu;
    if (exp == 0xFFu) {
        // NaN or Inf in Float32 — emit Float16 special-value bits directly.
        unsigned short bits16;
        if (mant != 0u) {
            bits16 = 0x7E00u;                                           // quiet NaN
        } else {
            unsigned int sign16 = (vb >> 16) & 0x8000u;
            bits16 = static_cast<unsigned short>(sign16 | 0x7C00u);     // ±Inf
        }
        __half h;
        *reinterpret_cast<unsigned short*>(&h) = bits16;
        return h;
    }
    return tenzor::rocm::safe_f2h(fv);
}

__global__ void div_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        c[idx] = f16_div_ieee(a[idx], b[idx]);
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

        // IEEE 754 div semantics via the shared bit-pattern helper: 0/0 → NaN,
        // x/0 → ±Inf for x ≠ 0 (matches the same-shape div_kernel_f16 path).
        c[out_idx] = f16_div_ieee(a[idx_a], b[idx_b]);
    }
}

/**
 * @brief Division kernel specialized for BFloat16
 * Uses float conversion for correct division and infinity handling
 */
__global__ void div_kernel_bf16(const hip_bfloat16* a, const hip_bfloat16* b, hip_bfloat16* c, int64_t n) {
    // IEEE 754 hardware divide: 0/0 → NaN, x/0 → ±Inf for x ≠ 0.
    // S.10: RNE rounding on the float32 quotient → bfloat16 cast (the
    // truncating ctor systematically biased BF16 element-wise division).
    HIP_KERNEL_LOOP(idx, n) {
        c[idx] = tenzor::rocm::f32_to_bf16_rne(static_cast<float>(a[idx]) / static_cast<float>(b[idx]));
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

        // IEEE 754 div semantics handled by hardware (0/0 → NaN, x/0 → ±Inf).
        // S.10: RNE-round on the float32 → bfloat16 narrowing.
        {
            float result = static_cast<float>(a[idx_a]) / static_cast<float>(b[idx_b]);
            c[out_idx] = tenzor::rocm::f32_to_bf16_rne(result);
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
        // Use ieee_div so a zero denominator yields IEEE Inf/NaN rather than the
        // finite garbage that -freciprocal-math produces for Float64 (see ieee_div).
        c[base]     = ieee_div(ar * br + ai * bi, denom);
        c[base + 1] = ieee_div(ai * br - ar * bi, denom);
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
        // See complex_div_kernel: ieee_div restores IEEE Inf/NaN on zero denom.
        c[c_base]     = ieee_div(ar * br + ai * bi, denom);
        c[c_base + 1] = ieee_div(ai * br - ar * bi, denom);
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

/**
 * @brief Complex64 magnitude: |a + bi| = hypot(a, b). Output is Float32.
 */
__global__ void abs_kernel_complex64(const hipFloatComplex* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = hypotf(hipCrealf(input[idx]), hipCimagf(input[idx]));
    }
}

/**
 * @brief Complex128 magnitude. Output is Float64.
 */
__global__ void abs_kernel_complex128(const hipDoubleComplex* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = hypot(hipCreal(input[idx]), hipCimag(input[idx]));
    }
}

/**
 * @brief Complex64 negate: -(a + bi) = -a - bi.
 */
__global__ void neg_kernel_complex64(const hipFloatComplex* input, hipFloatComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = make_hipFloatComplex(-hipCrealf(input[idx]), -hipCimagf(input[idx]));
    }
}

/**
 * @brief Complex128 negate.
 */
__global__ void neg_kernel_complex128(const hipDoubleComplex* input, hipDoubleComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = make_hipDoubleComplex(-hipCreal(input[idx]), -hipCimag(input[idx]));
    }
}

// Transcendentals on complex numbers.
// exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
__global__ void exp_kernel_complex64(const hipFloatComplex* input, hipFloatComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float a = hipCrealf(input[idx]);
        float b = hipCimagf(input[idx]);
        float ea = expf(a);
        output[idx] = make_hipFloatComplex(ea * cosf(b), ea * sinf(b));
    }
}
__global__ void exp_kernel_complex128(const hipDoubleComplex* input, hipDoubleComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double a = hipCreal(input[idx]);
        double b = hipCimag(input[idx]);
        double ea = exp(a);
        output[idx] = make_hipDoubleComplex(ea * cos(b), ea * sin(b));
    }
}

// log(a+bi) = log(hypot(a,b)) + i*atan2(b,a)
__global__ void log_kernel_complex64(const hipFloatComplex* input, hipFloatComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float a = hipCrealf(input[idx]);
        float b = hipCimagf(input[idx]);
        output[idx] = make_hipFloatComplex(logf(hypotf(a, b)), atan2f(b, a));
    }
}
__global__ void log_kernel_complex128(const hipDoubleComplex* input, hipDoubleComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double a = hipCreal(input[idx]);
        double b = hipCimag(input[idx]);
        output[idx] = make_hipDoubleComplex(log(hypot(a, b)), atan2(b, a));
    }
}

// sqrt(a+bi) principal branch — Kahan/Hull 1994 cancellation-free formulation.
__global__ void sqrt_kernel_complex64(const hipFloatComplex* input, hipFloatComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float a = hipCrealf(input[idx]);
        float b = hipCimagf(input[idx]);
        if (a == 0.0f && b == 0.0f) {
            output[idx] = make_hipFloatComplex(0.0f, 0.0f);
            continue;
        }
        float s = sqrtf(0.5f * (fabsf(a) + hypotf(a, b)));
        float re, im;
        if (a >= 0.0f) { re = s;                 im = b / (2.0f * s); }
        else            { re = fabsf(b) / (2.0f * s); im = copysignf(s, b); }
        output[idx] = make_hipFloatComplex(re, im);
    }
}
__global__ void sqrt_kernel_complex128(const hipDoubleComplex* input, hipDoubleComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double a = hipCreal(input[idx]);
        double b = hipCimag(input[idx]);
        if (a == 0.0 && b == 0.0) {
            output[idx] = make_hipDoubleComplex(0.0, 0.0);
            continue;
        }
        double s = sqrt(0.5 * (fabs(a) + hypot(a, b)));
        double re, im;
        if (a >= 0.0) { re = s;              im = b / (2.0 * s); }
        else           { re = fabs(b) / (2.0 * s); im = copysign(s, b); }
        output[idx] = make_hipDoubleComplex(re, im);
    }
}

// sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
__global__ void sin_kernel_complex64(const hipFloatComplex* input, hipFloatComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float a = hipCrealf(input[idx]);
        float b = hipCimagf(input[idx]);
        output[idx] = make_hipFloatComplex(sinf(a) * coshf(b), cosf(a) * sinhf(b));
    }
}
__global__ void sin_kernel_complex128(const hipDoubleComplex* input, hipDoubleComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double a = hipCreal(input[idx]);
        double b = hipCimag(input[idx]);
        output[idx] = make_hipDoubleComplex(sin(a) * cosh(b), cos(a) * sinh(b));
    }
}

// cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
__global__ void cos_kernel_complex64(const hipFloatComplex* input, hipFloatComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float a = hipCrealf(input[idx]);
        float b = hipCimagf(input[idx]);
        output[idx] = make_hipFloatComplex(cosf(a) * coshf(b), -sinf(a) * sinhf(b));
    }
}
__global__ void cos_kernel_complex128(const hipDoubleComplex* input, hipDoubleComplex* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double a = hipCreal(input[idx]);
        double b = hipCimag(input[idx]);
        output[idx] = make_hipDoubleComplex(cos(a) * cosh(b), -sin(a) * sinh(b));
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
        float x = input[idx];
        // AMD device libm's logf returns implementation-defined garbage at
        // x == 0 (observed ~-4.76 for Float64; Float32 is also unreliable
        // at the pole). Enforce the IEEE-754 contract explicitly: log(0)
        // is -inf, log(<0) is NaN, only positive x goes through libm.
        if (x == 0.0f) {
            output[idx] = -__int_as_float(0x7f800000);  // -INF
        } else if (x < 0.0f) {
            output[idx] = __int_as_float(0x7fc00000);   // NaN
        } else {
            output[idx] = logf(x);
        }
    }
}

/**
 * @brief Natural logarithm kernel (double precision)
 */
__global__ void log_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double x = input[idx];
        // See log_kernel_f32 — AMD device libm's log() doesn't honour the
        // IEEE-754 pole behaviour for Float64 either (observed ~-4.76 at
        // x=0). Handle x <= 0 explicitly with the IEEE-754 result.
        if (x == 0.0) {
            output[idx] = -__longlong_as_double(0x7ff0000000000000LL);  // -INF
        } else if (x < 0.0) {
            output[idx] = __longlong_as_double(0x7ff8000000000000LL);    // NaN
        } else {
            output[idx] = log(x);
        }
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
        float val = tenzor::rocm::safe_h2f(input[idx]);
        float result = powf(val, exponent);
        output[idx] = tenzor::rocm::safe_f2h(result);
    }
}

/**
 * @brief Clamp kernel (float): clamp values to [min_val, max_val]
 * Uses fminf/fmaxf for optimal AMD GPU performance
 */
__global__ void clamp_kernel_f32(const float* input, float* output, float min_val, float max_val, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = input[idx];
        // PyTorch/CPU semantics: clamp propagates NaN. fminf/fmaxf return the
        // non-NaN operand, so a NaN input would wrongly become a bound; pass it
        // through unchanged. is_nan_bits avoids the unreliable HIP NaN intrinsic.
        output[idx] = tenzor::rocm::is_nan_bits(val) ? val
                                                     : fminf(fmaxf(val, min_val), max_val);
    }
}

/**
 * @brief Clamp kernel (double precision)
 */
__global__ void clamp_kernel_f64(const double* input, double* output, double min_val, double max_val, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double val = input[idx];
        // PyTorch/CPU semantics: clamp propagates NaN (see f32 kernel).
        output[idx] = tenzor::rocm::is_nan_bits(val) ? val
                                                     : fmin(fmax(val, min_val), max_val);
    }
}

/**
 * @brief Clamp kernel (half precision) - compute via float conversion
 */
__global__ void clamp_kernel_f16(const __half* input, __half* output, float min_val, float max_val, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        // PyTorch/CPU semantics: clamp propagates NaN (see f32 kernel).
        if (tenzor::rocm::is_nan_bits(input[idx])) {
            output[idx] = input[idx];
            continue;
        }
        float val = tenzor::rocm::safe_h2f(input[idx]);
        float result = fminf(fmaxf(val, min_val), max_val);
        output[idx] = tenzor::rocm::safe_f2h(result);
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
        output[idx] = tenzor::rocm::safe_f2h(sqrtf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void exp_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(expf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void log_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(logf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void sign_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        output[idx] = tenzor::rocm::safe_f2h((val > 0.0f) - (val < 0.0f));
    }
}

__global__ void sin_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(sinf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void cos_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(cosf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void tan_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(tanf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void asin_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(asinf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void acos_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(acosf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void atan_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(atanf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void sinh_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(sinhf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void cosh_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(coshf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void reciprocal_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        output[idx] = tenzor::rocm::safe_f2h(1.0f / val);
    }
}

__global__ void floor_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(floorf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void ceil_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(ceilf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void round_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(roundf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void div_inplace_kernel_f16(__half* a, const __half* b, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float divisor = tenzor::rocm::safe_h2f(b[idx]);
        float result = tenzor::rocm::safe_h2f(a[idx]) / divisor;
        a[idx] = tenzor::rocm::safe_f2h(result);
    }
}

// Dot product kernel for Float16 - computes in float for accuracy
__global__ void dot_kernel_f16(const __half* a, const __half* b, float* partial_sums, int64_t n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    while (idx < n) {
        sum += tenzor::rocm::safe_h2f(a[idx]) * tenzor::rocm::safe_h2f(b[idx]);
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
auto add_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    if (a_in.dtype() != b_in.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype");
    }

    // Flat-index kernels require contiguous storage; materialize views.
    Tensor a = a_in.is_contiguous() ? a_in : a_in.contiguous();
    Tensor b = b_in.is_contiguous() ? b_in : b_in.contiguous();

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
        } else if (a.dtype() == DType::FP8_E4M3 || a.dtype() == DType::FP8_E5M2 ||
                   a.dtype() == DType::FP8_E4M3FNUZ || a.dtype() == DType::FP8_E5M2FNUZ) {
            // FP8 emulation: widen to Float32, add, narrow back
            DType orig = a.dtype();
            auto a_f32 = cast_kernel(a, DType::Float32, stream);
            auto b_f32 = cast_kernel(b, DType::Float32, stream);
            auto result_f32 = add_kernel(a_f32, b_f32, stream);
            return cast_kernel(result_f32, orig, stream);
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
    HIP_CHECK(hipMemcpyAsync(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));

    int64_t n = result.numel();
    int64_t ndim = output_shape.size();

    // Empty broadcast output (e.g. [0,3] op [1,3] -> [0,3]) reaches this path
    // because the same-shape fast path was skipped. HIP rejects zero-grid
    // launches, so free the broadcast metadata and return the empty result.
    if (n == 0) {
        hipFree(d_strides_a);
        hipFree(d_strides_b);
        hipFree(d_output_shape);
        return result;
    }

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
    } else if (a.dtype() == DType::FP8_E4M3 || a.dtype() == DType::FP8_E5M2 ||
               a.dtype() == DType::FP8_E4M3FNUZ || a.dtype() == DType::FP8_E5M2FNUZ) {
        DType orig = a.dtype();
        auto a_f32 = cast_kernel(a, DType::Float32, stream);
        auto b_f32 = cast_kernel(b, DType::Float32, stream);
        auto result_f32 = add_kernel(a_f32, b_f32, stream);
        // Free the broadcast metadata buffers before the early return; the
        // function-end hipFree block below is unreachable on this path.
        HIP_CHECK(hipFree(d_strides_a));
        HIP_CHECK(hipFree(d_strides_b));
        HIP_CHECK(hipFree(d_output_shape));
        return cast_kernel(result_f32, orig, stream);
    } else {
        // Free the broadcast metadata before throwing to avoid leaking the
        // device stride/shape buffers on the unsupported-dtype path.
        hipFree(d_strides_a);
        hipFree(d_strides_b);
        hipFree(d_output_shape);
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
auto sub_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    if (a_in.dtype() != b_in.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype");
    }

    // Flat-index kernels require contiguous storage; materialize views.
    Tensor a = a_in.is_contiguous() ? a_in : a_in.contiguous();
    Tensor b = b_in.is_contiguous() ? b_in : b_in.contiguous();

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

        // Empty-tensor fast path — HIP rejects zero-grid launches
        // ("invalid configuration argument"). Skip the kernel entirely;
        // the pre-allocated result is already the correct empty shape.
        if (n == 0) return result;

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
        } else if (a.dtype() == DType::FP8_E4M3 || a.dtype() == DType::FP8_E5M2 ||
                   a.dtype() == DType::FP8_E4M3FNUZ || a.dtype() == DType::FP8_E5M2FNUZ) {
            DType orig = a.dtype();
            auto a_f32 = cast_kernel(a, DType::Float32, stream);
            auto b_f32 = cast_kernel(b, DType::Float32, stream);
            auto result_f32 = sub_kernel(a_f32, b_f32, stream);
            return cast_kernel(result_f32, orig, stream);
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
    HIP_CHECK(hipMemcpyAsync(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));

    int64_t n = result.numel();
    int64_t ndim = output_shape.size();

    // Empty broadcast output (e.g. [0,3] op [1,3] -> [0,3]) reaches this path
    // because the same-shape fast path was skipped. HIP rejects zero-grid
    // launches, so free the broadcast metadata and return the empty result.
    if (n == 0) {
        hipFree(d_strides_a);
        hipFree(d_strides_b);
        hipFree(d_output_shape);
        return result;
    }

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
    } else if (a.dtype() == DType::FP8_E4M3 || a.dtype() == DType::FP8_E5M2 ||
               a.dtype() == DType::FP8_E4M3FNUZ || a.dtype() == DType::FP8_E5M2FNUZ) {
        DType orig = a.dtype();
        auto a_f32 = cast_kernel(a, DType::Float32, stream);
        auto b_f32 = cast_kernel(b, DType::Float32, stream);
        auto result_f32 = sub_kernel(a_f32, b_f32, stream);
        // Free the broadcast metadata buffers before the early return; the
        // function-end hipFree block below is unreachable on this path.
        HIP_CHECK(hipFree(d_strides_a));
        HIP_CHECK(hipFree(d_strides_b));
        HIP_CHECK(hipFree(d_output_shape));
        return cast_kernel(result_f32, orig, stream);
    } else {
        // Free the broadcast metadata before throwing to avoid leaking the
        // device stride/shape buffers on the unsupported-dtype path.
        hipFree(d_strides_a);
        hipFree(d_strides_b);
        hipFree(d_output_shape);
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
auto mul_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    if (a_in.dtype() != b_in.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype");
    }

    // Materialize contiguous copies so the flat-index kernels below are safe
    // when callers pass non-contiguous views (e.g. from slice()).
    Tensor a = a_in.is_contiguous() ? a_in : a_in.contiguous();
    Tensor b = b_in.is_contiguous() ? b_in : b_in.contiguous();

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

        // Empty-tensor fast path — HIP rejects zero-grid launches.
        if (n == 0) return result;

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
        } else if (a.dtype() == DType::Int8) {
            hipLaunchKernelGGL(mul_kernel_device<int8_t>, grid, block, 0, stream,
                a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n);
        } else if (a.dtype() == DType::UInt8) {
            hipLaunchKernelGGL(mul_kernel_device<uint8_t>, grid, block, 0, stream,
                a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(), n);
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
        } else if (a.dtype() == DType::FP8_E4M3 || a.dtype() == DType::FP8_E5M2 ||
                   a.dtype() == DType::FP8_E4M3FNUZ || a.dtype() == DType::FP8_E5M2FNUZ) {
            DType orig = a.dtype();
            auto a_f32 = cast_kernel(a, DType::Float32, stream);
            auto b_f32 = cast_kernel(b, DType::Float32, stream);
            auto result_f32 = mul_kernel(a_f32, b_f32, stream);
            return cast_kernel(result_f32, orig, stream);
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
    HIP_CHECK(hipMemcpyAsync(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));

    int64_t n = result.numel();
    int64_t ndim = output_shape.size();

    // Empty broadcast output (e.g. [0,3] op [1,3] -> [0,3]) reaches this path
    // because the same-shape fast path was skipped. HIP rejects zero-grid
    // launches, so free the broadcast metadata and return the empty result.
    if (n == 0) {
        hipFree(d_strides_a);
        hipFree(d_strides_b);
        hipFree(d_output_shape);
        return result;
    }

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
    } else if (a.dtype() == DType::Int8) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int8_t, MulOp>), grid, block, 0, stream,
            a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, MulOp());
    } else if (a.dtype() == DType::UInt8) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<uint8_t, MulOp>), grid, block, 0, stream,
            a.data<uint8_t>(), b.data<uint8_t>(), result.data<uint8_t>(),
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
    } else if (a.dtype() == DType::FP8_E4M3 || a.dtype() == DType::FP8_E5M2 ||
               a.dtype() == DType::FP8_E4M3FNUZ || a.dtype() == DType::FP8_E5M2FNUZ) {
        DType orig = a.dtype();
        auto a_f32 = cast_kernel(a, DType::Float32, stream);
        auto b_f32 = cast_kernel(b, DType::Float32, stream);
        auto result_f32 = mul_kernel(a_f32, b_f32, stream);
        // Free the broadcast metadata buffers before the early return; the
        // function-end hipFree block below is unreachable on this path.
        HIP_CHECK(hipFree(d_strides_a));
        HIP_CHECK(hipFree(d_strides_b));
        HIP_CHECK(hipFree(d_output_shape));
        return cast_kernel(result_f32, orig, stream);
    } else {
        // Free the broadcast metadata before throwing to avoid leaking the
        // device stride/shape buffers on the unsupported-dtype path.
        hipFree(d_strides_a);
        hipFree(d_strides_b);
        hipFree(d_output_shape);
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
auto div_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    if (a_in.dtype() != b_in.dtype()) {
        throw std::runtime_error("Tensors must have the same dtype");
    }

    // Flat-index kernels require contiguous storage; materialize views.
    Tensor a = a_in.is_contiguous() ? a_in : a_in.contiguous();
    Tensor b = b_in.is_contiguous() ? b_in : b_in.contiguous();

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

        // Empty-tensor fast path — HIP rejects zero-grid launches
        // ("invalid configuration argument"). Skip the kernel entirely;
        // the pre-allocated result is already the correct empty shape.
        if (n == 0) return result;

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
        } else if (a.dtype() == DType::FP8_E4M3 || a.dtype() == DType::FP8_E5M2 ||
                   a.dtype() == DType::FP8_E4M3FNUZ || a.dtype() == DType::FP8_E5M2FNUZ) {
            DType orig = a.dtype();
            auto a_f32 = cast_kernel(a, DType::Float32, stream);
            auto b_f32 = cast_kernel(b, DType::Float32, stream);
            auto result_f32 = div_kernel(a_f32, b_f32, stream);
            return cast_kernel(result_f32, orig, stream);
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
    HIP_CHECK(hipMemcpyAsync(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream));

    int64_t n = result.numel();
    int64_t ndim = output_shape.size();

    // Empty broadcast output (e.g. [0,3] op [1,3] -> [0,3]) reaches this path
    // because the same-shape fast path was skipped. HIP rejects zero-grid
    // launches, so free the broadcast metadata and return the empty result.
    if (n == 0) {
        hipFree(d_strides_a);
        hipFree(d_strides_b);
        hipFree(d_output_shape);
        return result;
    }

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
    } else if (a.dtype() == DType::FP8_E4M3 || a.dtype() == DType::FP8_E5M2 ||
               a.dtype() == DType::FP8_E4M3FNUZ || a.dtype() == DType::FP8_E5M2FNUZ) {
        DType orig = a.dtype();
        auto a_f32 = cast_kernel(a, DType::Float32, stream);
        auto b_f32 = cast_kernel(b, DType::Float32, stream);
        auto result_f32 = div_kernel(a_f32, b_f32, stream);
        // Free the broadcast metadata buffers before the early return; the
        // function-end hipFree block below is unreachable on this path.
        HIP_CHECK(hipFree(d_strides_a));
        HIP_CHECK(hipFree(d_strides_b));
        HIP_CHECK(hipFree(d_output_shape));
        return cast_kernel(result_f32, orig, stream);
    } else {
        // Free the broadcast metadata before throwing to avoid leaking the
        // device stride/shape buffers on the unsupported-dtype path.
        hipFree(d_strides_a);
        hipFree(d_strides_b);
        hipFree(d_output_shape);
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

    if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP

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
    } else if (input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(neg_kernel_complex64, grid, block, 0, stream,
            reinterpret_cast<const hipFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipFloatComplex*>(result.data<uint8_t>()), n);
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(neg_kernel_complex128, grid, block, 0, stream,
            reinterpret_cast<const hipDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipDoubleComplex*>(result.data<uint8_t>()), n);
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

    // Complex inputs: |z| = hypot(re, im); output is real-valued.
    if (input.dtype() == DType::Complex64) {
        Tensor result(shape, DType::Float32, input.device());
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(abs_kernel_complex64, grid, block, 0, stream,
            reinterpret_cast<const hipFloatComplex*>(input.data<uint8_t>()),
            result.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(abs_kernel_complex128, grid, block, 0, stream,
            reinterpret_cast<const hipDoubleComplex*>(input.data<uint8_t>()),
            result.data<double>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }

    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP

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
    // BFloat16 widen-narrow: compute in Float32, narrow back. Matches the CPU
    // reference (which implements the BF16 path) so a BF16 model that runs on
    // CPU does not throw on ROCm.
    if (input.dtype() == DType::BFloat16) {
        return sqrt_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP

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
    } else if (input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(sqrt_kernel_complex64, grid, block, 0, stream,
            reinterpret_cast<const hipFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipFloatComplex*>(result.data<uint8_t>()), n);
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(sqrt_kernel_complex128, grid, block, 0, stream,
            reinterpret_cast<const hipDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipDoubleComplex*>(result.data<uint8_t>()), n);
    } else {
        throw std::runtime_error("sqrt operation only supports Float32, Float64, Float16, Complex64, Complex128 dtypes");
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return exp_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP

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
    } else if (input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(exp_kernel_complex64, grid, block, 0, stream,
            reinterpret_cast<const hipFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipFloatComplex*>(result.data<uint8_t>()), n);
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(exp_kernel_complex128, grid, block, 0, stream,
            reinterpret_cast<const hipDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipDoubleComplex*>(result.data<uint8_t>()), n);
    } else {
        throw std::runtime_error("exp operation only supports Float32, Float64, Float16, Complex64, Complex128 dtypes");
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return log_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP

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
    } else if (input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(log_kernel_complex64, grid, block, 0, stream,
            reinterpret_cast<const hipFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipFloatComplex*>(result.data<uint8_t>()), n);
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(log_kernel_complex128, grid, block, 0, stream,
            reinterpret_cast<const hipDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipDoubleComplex*>(result.data<uint8_t>()), n);
    } else {
        throw std::runtime_error("log operation only supports Float32, Float64, Float16, Complex64, Complex128 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Integer clamp / pow / sign (templated; matches PyTorch integer support).
// ============================================================================
template <typename T>
__global__ void clamp_int_kernel(const T* input, T* output, T lo, T hi, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { T v = input[idx]; output[idx] = v < lo ? lo : (v > hi ? hi : v); }
}
template <typename T>
__global__ void sign_int_kernel(const T* input, T* output, int64_t n) {
    const bool kSigned = (T(-1) < T(0));
    HIP_KERNEL_LOOP(idx, n) {
        T v = input[idx];
        int neg = (kSigned && v < T(0)) ? 1 : 0;
        int pos = (v > T(0)) ? 1 : 0;
        output[idx] = static_cast<T>(pos - neg);
    }
}
template <typename T>
__global__ void pow_int_kernel(const T* input, T* output, double e, int int_exp,
                               long long k, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        if (int_exp) {
            T base = input[idx], r = T(1);
            for (long long j = 0; j < k; ++j) r *= base;
            output[idx] = r;
        } else {
            output[idx] = static_cast<T>(llround(pow(static_cast<double>(input[idx]), e)));
        }
    }
}

#define TENZOR_HIP_INT_DISPATCH(DT, NAME, ...)                                   \
    switch (DT) {                                                                \
        case DType::Int8:   { using T = int8_t;   __VA_ARGS__ } break;           \
        case DType::Int16:  { using T = int16_t;  __VA_ARGS__ } break;           \
        case DType::Int32:  { using T = int32_t;  __VA_ARGS__ } break;           \
        case DType::Int64:  { using T = int64_t;  __VA_ARGS__ } break;           \
        case DType::UInt8:  { using T = uint8_t;  __VA_ARGS__ } break;           \
        case DType::UInt16: { using T = uint16_t; __VA_ARGS__ } break;           \
        case DType::UInt32: { using T = uint32_t; __VA_ARGS__ } break;           \
        case DType::UInt64: { using T = uint64_t; __VA_ARGS__ } break;           \
        default: throw std::runtime_error(std::string(NAME) + ": unsupported dtype"); \
    }

/**
 * @brief Saturating host-side cast of a double scalar bound to an integer type.
 *
 * `static_cast<int64_t>(+inf)` (and any double outside the integer type's
 * representable range) is undefined behaviour in C++ and on this hardware
 * produces INT64_MIN. ClampMin/ClampMax pass +inf / -inf as the unused bound
 * (registry: clamp_kernel(x, min, +inf) / clamp_kernel(x, -inf, max)), so the
 * naive cast turned `clamp_min(int64_tensor, 0)` into an all-INT64_MIN result
 * (it set hi = static_cast<int64_t>(+inf) = INT64_MIN, then v > hi was always
 * true). This saturates: +inf / above-max -> type max, -inf / below-min ->
 * type min, NaN -> 0, otherwise the truncated value — matching PyTorch's
 * integer-clamp semantics with out-of-range bounds.
 */
template <typename T>
static inline T saturate_double_to_int(double v) {
    constexpr double lo = static_cast<double>(std::numeric_limits<T>::lowest());
    constexpr double hi = static_cast<double>(std::numeric_limits<T>::max());
    if (std::isnan(v)) return T(0);
    if (v <= lo) return std::numeric_limits<T>::lowest();
    // `hi` is rounded to the nearest double, which for 64-bit types is
    // 2^63 (one past INT64_MAX); >= catches that boundary so we never cast a
    // double that exceeds the representable range.
    if (v >= hi) return std::numeric_limits<T>::max();
    return static_cast<T>(v);
}

/**
 * @brief Power kernel launcher: output = input^exponent
 * @param input Input tensor (base)
 * @param exponent Power exponent
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (input^exponent)
 */
auto pow_kernel(const Tensor& input, double exponent, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(pow_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), static_cast<float>(exponent), n);
    } else if (input.dtype() == DType::Float64) {
        // Keep full double precision in the exponent (the scalar arrives as
        // double; truncating to float here lost ~7 digits — audit P1 2.10).
        hipLaunchKernelGGL(pow_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), exponent, n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(pow_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), static_cast<float>(exponent), n);
    } else if (input.dtype() == DType::BFloat16) {
        // No native BF16 pow kernel: widen to Float32, compute, narrow back
        // (mirrors clamp_kernel's half handling).
        return pow_kernel(input.to(DType::Float32), exponent, stream).to(DType::BFloat16);
    } else {
        const double e = exponent;
        const int int_exp = (e == floor(e) && e >= 0.0) ? 1 : 0;
        const long long k = static_cast<long long>(e);
        TENZOR_HIP_INT_DISPATCH(input.dtype(), "pow",
            hipLaunchKernelGGL(pow_int_kernel<T>, grid, block, 0, stream,
                input.data<T>(), result.data<T>(), e, int_exp, k, n);)
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
auto clamp_kernel(const Tensor& input, double min_val, double max_val, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) {
        // HIP rejects zero-grid launches with "invalid configuration argument".
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(clamp_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(),
            static_cast<float>(min_val), static_cast<float>(max_val), n);
    } else if (input.dtype() == DType::Float64) {
        // Preserve full double precision in the bounds (audit P1 2.10): the
        // scalars arrive as double; truncating to float lost ~7 digits.
        hipLaunchKernelGGL(clamp_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), min_val, max_val, n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(clamp_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            static_cast<float>(min_val), static_cast<float>(max_val), n);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = clamp_kernel(input_f32, min_val, max_val, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        // Saturate the double bounds into the integer range. clamp_min passes
        // max_val = +inf and clamp_max passes min_val = -inf; a plain
        // static_cast<T>(+/-inf) is UB and produced INT64_MIN, corrupting the
        // whole result (e.g. clamp_min(int64, 0) -> all INT64_MIN).
        TENZOR_HIP_INT_DISPATCH(input.dtype(), "clamp",
            hipLaunchKernelGGL(clamp_int_kernel<T>, grid, block, 0, stream,
                input.data<T>(), result.data<T>(),
                saturate_double_to_int<T>(min_val),
                saturate_double_to_int<T>(max_val), n);)
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

    if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP

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
    } else if (input.dtype() == DType::BFloat16) {
        // No native BF16 sign kernel: widen to Float32, compute, narrow back.
        return sign_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    } else {
        TENZOR_HIP_INT_DISPATCH(input.dtype(), "sign",
            hipLaunchKernelGGL(sign_int_kernel<T>, grid, block, 0, stream,
                input.data<T>(), result.data<T>(), n);)
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Trigonometric Operations
// ============================================================================

// Float32 sin/cos evaluate in double then narrow. Device sinf/cosf carry ~1-2
// ULP error; computing in double brings the Float32 result to <1 ULP, matching
// the high-accuracy CPU reference (MKL vsSin) and the CUDA backend (which does
// the same) within the 1e-7 transcendental atol of MathOperationParity.Sin/Cos.
// Only instantiated for float/double (Float16 has sin_kernel_f16); the double
// cast is a no-op on the double path.
template<typename T>
__global__ void sin_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<T>(sin(static_cast<double>(input[idx])));
    }
}

template<typename T>
__global__ void cos_kernel_device(const T* input, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<T>(cos(static_cast<double>(input[idx])));
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
        output[idx] = tenzor::rocm::safe_f2h(truncf(tenzor::rocm::safe_h2f(input[idx])));
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

// Float16 in-place broadcasting division. Computes via float and writes back
// through f16_div_ieee so special values (0/0 -> NaN, x/0 -> +-Inf) match the
// same-shape div_inplace_kernel_f16 path rather than relying on native __half
// division (which some HIP builds canonicalize).
__global__ void div_inplace_broadcast_kernel_f16(__half* a, const __half* b, const int64_t* strides_b,
    const int64_t* a_shape, int64_t ndim, int64_t n) {
    HIP_KERNEL_LOOP(out_idx, n) {
        int64_t idx_b = 0, tmp = out_idx;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t coord = tmp % a_shape[i];
            tmp /= a_shape[i];
            idx_b += coord * strides_b[i];
        }
        a[out_idx] = f16_div_ieee(a[out_idx], b[idx_b]);
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return sin_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
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
    } else if (input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(sin_kernel_complex64, grid, block, 0, stream,
            reinterpret_cast<const hipFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipFloatComplex*>(result.data<uint8_t>()), n);
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(sin_kernel_complex128, grid, block, 0, stream,
            reinterpret_cast<const hipDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipDoubleComplex*>(result.data<uint8_t>()), n);
    } else {
        throw std::runtime_error("sin operation only supports Float32, Float64, Float16, Complex64, Complex128 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto cos_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return cos_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
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
    } else if (input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(cos_kernel_complex64, grid, block, 0, stream,
            reinterpret_cast<const hipFloatComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipFloatComplex*>(result.data<uint8_t>()), n);
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(cos_kernel_complex128, grid, block, 0, stream,
            reinterpret_cast<const hipDoubleComplex*>(input.data<uint8_t>()),
            reinterpret_cast<hipDoubleComplex*>(result.data<uint8_t>()), n);
    } else {
        throw std::runtime_error("cos operation only supports Float32, Float64, Float16, Complex64, Complex128 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(tan_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("tan operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(asin_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("asin operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(acos_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("acos operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(atan_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("atan operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(sinh_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("sinh operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(cosh_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("cosh operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(reciprocal_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("reciprocal operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(floor_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("floor operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(ceil_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("ceil operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(round_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("round operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (input.dtype() == DType::BFloat16) {
        auto in_f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(trunc_kernel(in_f32, stream), DType::BFloat16, stream);
    } else {
        throw std::runtime_error("trunc operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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

    // BFloat16 widen-narrow: no native BF16 device division path. Compute in
    // Float32, then write the result back into `a`'s BF16 storage. The inplace
    // contract requires `a` itself to be the destination, so we cast-in-place
    // rather than returning a new tensor.
    if (a.dtype() == DType::BFloat16) {
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        div_inplace_kernel(a_f32, b_f32, stream);
        auto result_bf16 = a_f32.to(DType::BFloat16);
        HIP_CHECK(hipMemcpyAsync(a.data_ptr(), result_bf16.data_ptr(),
                                  n * sizeof(uint16_t), hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return;
    }

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
            hipLaunchKernelGGL(div_inplace_broadcast_kernel_f16, grid, block, 0, stream,
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
        throw std::runtime_error("div_inplace operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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

// Specialised reducer that consumes float partial sums and writes a Float16 result
// on-device. Used by the Float16 dot product to avoid a host-side cast roundtrip.
__global__ void final_reduce_kernel_float_to_half(const float* partial, __half* output, int num_blocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = tenzor::rocm::safe_f2h(sdata[0]);
    }
}

auto dot_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.numel() != b.numel()) {
        throw std::invalid_argument("Tensor sizes must match for dot product");
    }

    int64_t n = a.numel();
    Tensor result({}, a.dtype(), a.device());  // Scalar output

    if (n == 0) {
        // Zero the scalar output for ALL supported dtypes — an all-zero byte
        // pattern is numeric zero for every integer/float/complex type. The
        // previous per-dtype branches left Int64/Complex/BFloat16 uninitialized.
        HIP_CHECK(hipMemset(const_cast<void*>(result.data_ptr()), 0, result.dtype_size()));
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
        // Compute dot product in float32 for accuracy; the final reducer fuses the
        // float→Float16 cast on-device so there is no host roundtrip.
        float* d_partial;
        HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(float)));

        hipLaunchKernelGGL(dot_kernel_f16, dim3(num_blocks), dim3(block_size), 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            d_partial, n);
        HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(final_reduce_kernel_float_to_half, dim3(1), dim3(block_size), 0, stream,
            d_partial, reinterpret_cast<__half*>(result.data<Float16>()), num_blocks);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipFree(d_partial));
    } else if (a.dtype() == DType::BFloat16) {
        auto a_f32 = cast_kernel(a, DType::Float32, stream);
        auto b_f32 = cast_kernel(b, DType::Float32, stream);
        return cast_kernel(dot_kernel(a_f32, b_f32, stream), DType::BFloat16, stream);
    } else if (a.dtype() == DType::Int8 || a.dtype() == DType::UInt8 ||
               a.dtype() == DType::Int16 || a.dtype() == DType::Int32 ||
               a.dtype() == DType::UInt16 || a.dtype() == DType::UInt32 ||
               a.dtype() == DType::UInt64) {
        // Integer dot: accumulate in Int64 to avoid overflow, narrow to input dtype.
        DType orig = a.dtype();
        auto a64 = cast_kernel(a, DType::Int64, stream);
        auto b64 = cast_kernel(b, DType::Int64, stream);
        return cast_kernel(dot_kernel(a64, b64, stream), orig, stream);
    } else if (a.dtype() == DType::Int64) {
        int64_t* d_partial;
        HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(int64_t)));
        hipLaunchKernelGGL(dot_kernel_device<int64_t>, dim3(num_blocks), dim3(block_size), 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), d_partial, n);
        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(final_reduce_kernel<int64_t>, dim3(1), dim3(block_size), 0, stream,
            d_partial, result.data<int64_t>(), num_blocks);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipFree(d_partial));
    } else if (a.dtype() == DType::Complex64 || a.dtype() == DType::Complex128) {
        // Complex dot: Σ a_i·b_i = (Σ ar·br - Σ ai·bi) + i(Σ ar·bi + Σ ai·br),
        // each Σ a real dot product (matches torch.dot — no conjugation).
        auto ar = tenzor::real(a); auto ai = tenzor::imag(a);
        auto br = tenzor::real(b); auto bi = tenzor::imag(b);
        auto re = tenzor::sub(dot_kernel(ar, br, stream), dot_kernel(ai, bi, stream));
        auto im = tenzor::add(dot_kernel(ar, bi, stream), dot_kernel(ai, br, stream));
        return tenzor::complex(re, im);
    } else {
        throw std::runtime_error("dot operation only supports Float32, Float64, Float16, BFloat16, integer, and complex dtypes");
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

// Strided fill: write `value` at positions i*stride for i in [0, n).
// Used to set the real component of an interleaved complex buffer.
template<typename T>
__global__ void fill_strided_kernel(T* output, T value, int64_t n, int64_t stride) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx * stride] = value;
    }
}

/**
 * @brief Fill kernel launcher - fills tensor with constant value
 * @param tensor Tensor to fill
 * @param value Fill value
 * @param stream HIP stream for asynchronous execution
 * @return Filled tensor
 */
auto fill_kernel(const Tensor& tensor, double value, hipStream_t stream) -> Tensor {
    int64_t n = tensor.numel();

    // OpId::Fill is a non-in-place, single-output op. Tensor's copy ctor shares
    // storage, so `auto result = tensor;` would overwrite the caller's input.
    // clone() gives an independent deep copy; fill overwrites every element next.
    Tensor result = tensor.clone();

    if (n == 0) {
        return result;
    }

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (tensor.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fill_kernel_device<float>, grid, block, 0, stream,
            result.data<float>(), static_cast<float>(value), n);
    } else if (tensor.dtype() == DType::Float64) {
        // Pass the double value directly — narrowing to float here would
        // collapse Float64 subnormals (~5e-324) to zero.
        hipLaunchKernelGGL(fill_kernel_device<double>, grid, block, 0, stream,
            result.data<double>(), value, n);
    } else if (tensor.dtype() == DType::Int32) {
        hipLaunchKernelGGL(fill_kernel_device<int32_t>, grid, block, 0, stream,
            result.data<int32_t>(), static_cast<int32_t>(value), n);
    } else if (tensor.dtype() == DType::Int64) {
        hipLaunchKernelGGL(fill_kernel_device<int64_t>, grid, block, 0, stream,
            result.data<int64_t>(), static_cast<int64_t>(value), n);
    } else if (tensor.dtype() == DType::Float16) {
        hipLaunchKernelGGL(fill_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(result.data<Float16>()), tenzor::rocm::safe_f2h(static_cast<float>(value)), n);
    } else if (tensor.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(fill_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), tenzor::rocm::f32_to_bf16_rne(static_cast<float>(value)), n);
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
    } else if (dtype == DType::Int16) {
        hipLaunchKernelGGL(fill_kernel_device<int16_t>, grid, block, 0, stream,
            result.data<int16_t>(), static_cast<int16_t>(0), n);
    } else if (dtype == DType::UInt16) {
        hipLaunchKernelGGL(fill_kernel_device<uint16_t>, grid, block, 0, stream,
            result.data<uint16_t>(), static_cast<uint16_t>(0), n);
    } else if (dtype == DType::UInt32) {
        hipLaunchKernelGGL(fill_kernel_device<uint32_t>, grid, block, 0, stream,
            result.data<uint32_t>(), static_cast<uint32_t>(0), n);
    } else if (dtype == DType::UInt64) {
        hipLaunchKernelGGL(fill_kernel_device<uint64_t>, grid, block, 0, stream,
            result.data<uint64_t>(), static_cast<uint64_t>(0), n);
    } else if (dtype == DType::Float16) {
        hipLaunchKernelGGL(fill_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(result.data<Float16>()), tenzor::rocm::safe_f2h(0.0f), n);
    } else if (dtype == DType::BFloat16) {
        hipLaunchKernelGGL(fill_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), tenzor::rocm::f32_to_bf16_rne(0.0f), n);
    } else if (dtype == DType::Complex64) {
        // Complex64 storage is interleaved (re, im) pairs of float.
        // Zero both parts by filling 2*n floats with 0.
        dim3 g2, b2;
        compute_launch_config_1d(n * 2, g2, b2);
        hipLaunchKernelGGL(fill_kernel_device<float>, g2, b2, 0, stream,
            reinterpret_cast<float*>(const_cast<void*>(result.data_ptr())), 0.0f, n * 2);
    } else if (dtype == DType::Complex128) {
        dim3 g2, b2;
        compute_launch_config_1d(n * 2, g2, b2);
        hipLaunchKernelGGL(fill_kernel_device<double>, g2, b2, 0, stream,
            reinterpret_cast<double*>(const_cast<void*>(result.data_ptr())), 0.0, n * 2);
    } else if (dtype == DType::QInt8 || dtype == DType::QUInt8 || dtype == DType::QInt4x2) {
        // Quantized types use 1-byte-per-element storage. All-bits-zero is a
        // valid zeroed buffer (dequantizes to 0 for zero_point=0). QInt4x2
        // packs two 4-bit values per byte; n equals the byte count here.
        hipLaunchKernelGGL(fill_kernel_device<uint8_t>, grid, block, 0, stream,
            reinterpret_cast<uint8_t*>(const_cast<void*>(result.data_ptr())),
            static_cast<uint8_t>(0), n);
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
    } else if (dtype == DType::Int16) {
        hipLaunchKernelGGL(fill_kernel_device<int16_t>, grid, block, 0, stream,
            result.data<int16_t>(), static_cast<int16_t>(1), n);
    } else if (dtype == DType::UInt16) {
        hipLaunchKernelGGL(fill_kernel_device<uint16_t>, grid, block, 0, stream,
            result.data<uint16_t>(), static_cast<uint16_t>(1), n);
    } else if (dtype == DType::UInt32) {
        hipLaunchKernelGGL(fill_kernel_device<uint32_t>, grid, block, 0, stream,
            result.data<uint32_t>(), static_cast<uint32_t>(1), n);
    } else if (dtype == DType::UInt64) {
        hipLaunchKernelGGL(fill_kernel_device<uint64_t>, grid, block, 0, stream,
            result.data<uint64_t>(), static_cast<uint64_t>(1), n);
    } else if (dtype == DType::Float16) {
        hipLaunchKernelGGL(fill_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(result.data<Float16>()), tenzor::rocm::safe_f2h(1.0f), n);
    } else if (dtype == DType::BFloat16) {
        hipLaunchKernelGGL(fill_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), tenzor::rocm::f32_to_bf16_rne(1.0f), n);
    } else if (dtype == DType::Complex64) {
        // ones of Complex64 = (1, 0). Zero-fill the full 2n-float buffer then
        // stamp real components to 1.0.
        dim3 g2, b2;
        compute_launch_config_1d(n * 2, g2, b2);
        hipLaunchKernelGGL(fill_kernel_device<float>, g2, b2, 0, stream,
            reinterpret_cast<float*>(const_cast<void*>(result.data_ptr())), 0.0f, n * 2);
        // Strided fill: real parts at stride 2.
        hipLaunchKernelGGL(fill_strided_kernel<float>, grid, block, 0, stream,
            reinterpret_cast<float*>(const_cast<void*>(result.data_ptr())), 1.0f, n, 2);
    } else if (dtype == DType::Complex128) {
        dim3 g2, b2;
        compute_launch_config_1d(n * 2, g2, b2);
        hipLaunchKernelGGL(fill_kernel_device<double>, g2, b2, 0, stream,
            reinterpret_cast<double*>(const_cast<void*>(result.data_ptr())), 0.0, n * 2);
        hipLaunchKernelGGL(fill_strided_kernel<double>, grid, block, 0, stream,
            reinterpret_cast<double*>(const_cast<void*>(result.data_ptr())), 1.0, n, 2);
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
auto full_kernel(const std::vector<int64_t>& shape, double value, DType dtype, Device device, hipStream_t stream) -> Tensor {
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
        // Pass the double value directly to preserve Float64 subnormals.
        hipLaunchKernelGGL(fill_kernel_device<double>, grid, block, 0, stream,
            result.data<double>(), value, n);
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
    } else if (dtype == DType::Int16) {
        hipLaunchKernelGGL(fill_kernel_device<int16_t>, grid, block, 0, stream,
            result.data<int16_t>(), static_cast<int16_t>(value), n);
    } else if (dtype == DType::UInt16) {
        hipLaunchKernelGGL(fill_kernel_device<uint16_t>, grid, block, 0, stream,
            result.data<uint16_t>(), static_cast<uint16_t>(value), n);
    } else if (dtype == DType::UInt32) {
        hipLaunchKernelGGL(fill_kernel_device<uint32_t>, grid, block, 0, stream,
            result.data<uint32_t>(), static_cast<uint32_t>(value), n);
    } else if (dtype == DType::UInt64) {
        hipLaunchKernelGGL(fill_kernel_device<uint64_t>, grid, block, 0, stream,
            result.data<uint64_t>(), static_cast<uint64_t>(value), n);
    } else if (dtype == DType::Float16) {
        hipLaunchKernelGGL(fill_kernel_device<__half>, grid, block, 0, stream,
            reinterpret_cast<__half*>(result.data<Float16>()), tenzor::rocm::safe_f2h(static_cast<float>(value)), n);
    } else if (dtype == DType::BFloat16) {
        hipLaunchKernelGGL(fill_kernel_device<hip_bfloat16>, grid, block, 0, stream,
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), tenzor::rocm::f32_to_bf16_rne(static_cast<float>(value)), n);
    } else if (dtype == DType::Complex64) {
        hipLaunchKernelGGL(fill_kernel_device<hipFloatComplex>, grid, block, 0, stream,
            reinterpret_cast<hipFloatComplex*>(result.data_ptr()),
            make_hipFloatComplex(static_cast<float>(value), 0.0f), n);
    } else if (dtype == DType::Complex128) {
        hipLaunchKernelGGL(fill_kernel_device<hipDoubleComplex>, grid, block, 0, stream,
            reinterpret_cast<hipDoubleComplex*>(result.data_ptr()),
            make_hipDoubleComplex(value, 0.0), n);
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
        output[idx] = tenzor::rocm::safe_f2h(hiprand_uniform(&states[idx]));
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
        output[idx] = tenzor::rocm::safe_f2h(hiprand_normal(&states[idx]));
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
    HipBuffer d_states_buf(static_cast<size_t>(n) * sizeof(hiprandState));
    hiprandState* d_states = d_states_buf.as<hiprandState>();

    // Honor `tenzor::manual_seed`. Time-based seed in the unsetted case.
    uint64_t seed = ::tenzor::get_global_seed();
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

    // d_states freed by its HipBuffer guard on scope exit.

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
    HipBuffer d_states_buf(static_cast<size_t>(n) * sizeof(hiprandState));
    hiprandState* d_states = d_states_buf.as<hiprandState>();

    // Honor `tenzor::manual_seed`. Time-based seed in the unsetted case.
    uint64_t seed = ::tenzor::get_global_seed();
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

    // d_states freed by its HipBuffer guard on scope exit.

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
    HipBuffer d_states_buf(static_cast<size_t>(n) * sizeof(hiprandState));
    hiprandState* d_states = d_states_buf.as<hiprandState>();

    // Honor `tenzor::manual_seed`. Time-based seed in the unsetted case.
    uint64_t seed = ::tenzor::get_global_seed();
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

    // d_states freed by HipBuffer RAII destructor (matches rand_kernel/randn_kernel);
    // an explicit hipFree here would double-free the buffer.
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

// Complex linear ramp (arange/linspace): real axis ramp, imaginary part 0.
__global__ void ramp_complex64_impl(hipFloatComplex* output, double start, double step, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = make_hipFloatComplex(static_cast<float>(start + static_cast<double>(idx) * step), 0.0f);
    }
}
__global__ void ramp_complex128_impl(hipDoubleComplex* output, double start, double step, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = make_hipDoubleComplex(start + static_cast<double>(idx) * step, 0.0);
    }
}
__global__ void eye_complex64_impl(hipFloatComplex* output, int64_t n, int64_t m, int64_t k) {
    int64_t total = n * m;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t row = idx / m; int64_t col = idx % m;
        output[idx] = make_hipFloatComplex((col - row == k) ? 1.0f : 0.0f, 0.0f);
    }
}
__global__ void eye_complex128_impl(hipDoubleComplex* output, int64_t n, int64_t m, int64_t k) {
    int64_t total = n * m;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t row = idx / m; int64_t col = idx % m;
        output[idx] = make_hipDoubleComplex((col - row == k) ? 1.0 : 0.0, 0.0);
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
                reinterpret_cast<__half*>(result.data<Float16>()), tenzor::rocm::safe_f2h(static_cast<float>(start)), tenzor::rocm::safe_f2h(static_cast<float>(step)), n);
            break;
        case DType::BFloat16:
            hipLaunchKernelGGL(arange_kernel_impl<hip_bfloat16>, grid, block, 0, stream,
                reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()),
                tenzor::rocm::f32_to_bf16_rne(static_cast<float>(start)),
                tenzor::rocm::f32_to_bf16_rne(static_cast<float>(step)), n);
            break;
        case DType::Int8:
            hipLaunchKernelGGL(arange_kernel_impl<int8_t>, grid, block, 0, stream,
                result.data<int8_t>(), static_cast<int8_t>(start), static_cast<int8_t>(step), n);
            break;
        case DType::Int16:
            hipLaunchKernelGGL(arange_kernel_impl<int16_t>, grid, block, 0, stream,
                result.data<int16_t>(), static_cast<int16_t>(start), static_cast<int16_t>(step), n);
            break;
        case DType::UInt8:
            hipLaunchKernelGGL(arange_kernel_impl<uint8_t>, grid, block, 0, stream,
                result.data<uint8_t>(), static_cast<uint8_t>(start), static_cast<uint8_t>(step), n);
            break;
        case DType::UInt16:
            hipLaunchKernelGGL(arange_kernel_impl<uint16_t>, grid, block, 0, stream,
                result.data<uint16_t>(), static_cast<uint16_t>(start), static_cast<uint16_t>(step), n);
            break;
        case DType::UInt32:
            hipLaunchKernelGGL(arange_kernel_impl<uint32_t>, grid, block, 0, stream,
                result.data<uint32_t>(), static_cast<uint32_t>(start), static_cast<uint32_t>(step), n);
            break;
        case DType::UInt64:
            hipLaunchKernelGGL(arange_kernel_impl<uint64_t>, grid, block, 0, stream,
                result.data<uint64_t>(), static_cast<uint64_t>(start), static_cast<uint64_t>(step), n);
            break;
        case DType::Complex64:
            hipLaunchKernelGGL(ramp_complex64_impl, grid, block, 0, stream,
                reinterpret_cast<hipFloatComplex*>(result.data_ptr()), start, step, n);
            break;
        case DType::Complex128:
            hipLaunchKernelGGL(ramp_complex128_impl, grid, block, 0, stream,
                reinterpret_cast<hipDoubleComplex*>(result.data_ptr()), start, step, n);
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
                tenzor::rocm::safe_f2h(static_cast<float>(start)),
                tenzor::rocm::safe_f2h(static_cast<float>(step)), steps);
            break;
        case DType::BFloat16:
            hipLaunchKernelGGL(linspace_kernel_impl<hip_bfloat16>, grid, block, 0, stream,
                reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()),
                tenzor::rocm::f32_to_bf16_rne(static_cast<float>(start)),
                tenzor::rocm::f32_to_bf16_rne(static_cast<float>(step)), steps);
            break;
        case DType::Int8:
            hipLaunchKernelGGL(linspace_kernel_impl<int8_t>, grid, block, 0, stream,
                result.data<int8_t>(), static_cast<int8_t>(start), static_cast<int8_t>(step), steps);
            break;
        case DType::Int16:
            hipLaunchKernelGGL(linspace_kernel_impl<int16_t>, grid, block, 0, stream,
                result.data<int16_t>(), static_cast<int16_t>(start), static_cast<int16_t>(step), steps);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(linspace_kernel_impl<int32_t>, grid, block, 0, stream,
                result.data<int32_t>(), static_cast<int32_t>(start), static_cast<int32_t>(step), steps);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(linspace_kernel_impl<int64_t>, grid, block, 0, stream,
                result.data<int64_t>(), static_cast<int64_t>(start), static_cast<int64_t>(step), steps);
            break;
        case DType::UInt8:
            hipLaunchKernelGGL(linspace_kernel_impl<uint8_t>, grid, block, 0, stream,
                result.data<uint8_t>(), static_cast<uint8_t>(start), static_cast<uint8_t>(step), steps);
            break;
        case DType::UInt16:
            hipLaunchKernelGGL(linspace_kernel_impl<uint16_t>, grid, block, 0, stream,
                result.data<uint16_t>(), static_cast<uint16_t>(start), static_cast<uint16_t>(step), steps);
            break;
        case DType::UInt32:
            hipLaunchKernelGGL(linspace_kernel_impl<uint32_t>, grid, block, 0, stream,
                result.data<uint32_t>(), static_cast<uint32_t>(start), static_cast<uint32_t>(step), steps);
            break;
        case DType::UInt64:
            hipLaunchKernelGGL(linspace_kernel_impl<uint64_t>, grid, block, 0, stream,
                result.data<uint64_t>(), static_cast<uint64_t>(start), static_cast<uint64_t>(step), steps);
            break;
        case DType::Complex64:
            hipLaunchKernelGGL(ramp_complex64_impl, grid, block, 0, stream,
                reinterpret_cast<hipFloatComplex*>(result.data_ptr()), start, step, steps);
            break;
        case DType::Complex128:
            hipLaunchKernelGGL(ramp_complex128_impl, grid, block, 0, stream,
                reinterpret_cast<hipDoubleComplex*>(result.data_ptr()), start, step, steps);
            break;
        default:
            throw std::runtime_error("linspace_kernel: unsupported dtype");
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
        case DType::BFloat16:
            hipLaunchKernelGGL(eye_kernel_impl<hip_bfloat16>, grid, block, 0, stream,
                reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n, m, k);
            break;
        case DType::Int8:
            hipLaunchKernelGGL(eye_kernel_impl<int8_t>, grid, block, 0, stream,
                result.data<int8_t>(), n, m, k);
            break;
        case DType::Int16:
            hipLaunchKernelGGL(eye_kernel_impl<int16_t>, grid, block, 0, stream,
                result.data<int16_t>(), n, m, k);
            break;
        case DType::UInt8:
            hipLaunchKernelGGL(eye_kernel_impl<uint8_t>, grid, block, 0, stream,
                result.data<uint8_t>(), n, m, k);
            break;
        case DType::UInt16:
            hipLaunchKernelGGL(eye_kernel_impl<uint16_t>, grid, block, 0, stream,
                result.data<uint16_t>(), n, m, k);
            break;
        case DType::UInt32:
            hipLaunchKernelGGL(eye_kernel_impl<uint32_t>, grid, block, 0, stream,
                result.data<uint32_t>(), n, m, k);
            break;
        case DType::UInt64:
            hipLaunchKernelGGL(eye_kernel_impl<uint64_t>, grid, block, 0, stream,
                result.data<uint64_t>(), n, m, k);
            break;
        case DType::Complex64:
            hipLaunchKernelGGL(eye_complex64_impl, grid, block, 0, stream,
                reinterpret_cast<hipFloatComplex*>(result.data_ptr()), n, m, k);
            break;
        case DType::Complex128:
            hipLaunchKernelGGL(eye_complex128_impl, grid, block, 0, stream,
                reinterpret_cast<hipDoubleComplex*>(result.data_ptr()), n, m, k);
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
    HIP_CHECK(hipcub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out,
                                     static_cast<int>(n), stream));
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    HIP_CHECK(hipcub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out,
                                     static_cast<int>(n), stream));
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
    HIP_CHECK(hipcub::DeviceScan::InclusiveScan(d_temp, temp_bytes, d_in, d_out,
                                      HipMultOp(), static_cast<int>(n), stream));
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    HIP_CHECK(hipcub::DeviceScan::InclusiveScan(d_temp, temp_bytes, d_in, d_out,
                                      HipMultOp(), static_cast<int>(n), stream));
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
        float fval = static_cast<float>(data[idx]);
        uint32_t bits;
        memcpy(&bits, &fval, sizeof(bits));
        bool inf_or_nan = ((bits & 0x7F800000u) == 0x7F800000u);
        if (inf_or_nan) {
            atomicExch(result, 1);
        }
    }
}

// Float64 specialization — use double-precision bitwise inf/nan check
template<>
__global__ void check_inf_nan_kernel<double>(const double* data, int64_t n, int* result) {
    HIP_KERNEL_LOOP(idx, n) {
        double val = data[idx];
        uint64_t bits;
        memcpy(&bits, &val, sizeof(bits));
        bool inf_or_nan = ((bits & 0x7FF0000000000000ull) == 0x7FF0000000000000ull);
        if (inf_or_nan) {
            atomicExch(result, 1);
        }
    }
}

auto has_inf_nan_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    const int64_t numel = input.numel();

    // Helper to create a Bool scalar tensor directly on the target device
    auto make_bool_scalar = [&stream](bool value, Device device) -> Tensor {
        Tensor result({}, DType::Bool, device);
        bool tmp = value;
        HIP_CHECK(hipMemcpyAsync(result.data<bool>(), &tmp, sizeof(bool),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return result;
    };

    if (numel == 0) {
        return make_bool_scalar(false, input.device());
    }

    // Allocate device flag (RAII so it frees on every exit/throw path).
    HipBuffer d_flag_buf(sizeof(int));
    int* d_flag = d_flag_buf.as<int>();
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
            // Integer types can't have inf/nan (d_flag freed by guard).
            return make_bool_scalar(false, input.device());
    }
    HIP_CHECK(hipGetLastError());

    int h_flag = 0;
    HIP_CHECK(hipMemcpyAsync(&h_flag, d_flag, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    // d_flag freed by its HipBuffer guard on scope exit.

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
        output[idx] = tenzor::rocm::safe_f2h(log2f(tenzor::rocm::safe_h2f(input[idx])));
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
        output[idx] = tenzor::rocm::safe_f2h(log10f(tenzor::rocm::safe_h2f(input[idx])));
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
        output[idx] = tenzor::rocm::safe_f2h(log1pf(tenzor::rocm::safe_h2f(input[idx])));
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
        output[idx] = tenzor::rocm::safe_f2h(exp2f(tenzor::rocm::safe_h2f(input[idx])));
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
        output[idx] = tenzor::rocm::safe_f2h(expm1f(tenzor::rocm::safe_h2f(input[idx])));
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
        output[idx] = tenzor::rocm::safe_f2h(erff(tenzor::rocm::safe_h2f(input[idx])));
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
        output[idx] = tenzor::rocm::safe_f2h(erfcf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

// ============================================================================
// Bool Predicate Kernels (isnan, isinf, isfinite)
// ============================================================================

__global__ void isnan_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint32_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>(((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0) ? 1 : 0);
    }
}

__global__ void isnan_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint64_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>(((bits & 0x7FF0000000000000ull) == 0x7FF0000000000000ull) && ((bits & 0x000FFFFFFFFFFFFFull) != 0) ? 1 : 0);
    }
}

__global__ void isnan_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        // Examine the raw F16 bit pattern directly. Going through
        // __half2float on some ROCm builds canonicalises NaN to a finite
        // value and misses the predicate.
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&input[idx]);
        uint16_t exp = (bits >> 10) & 0x1Fu;
        uint16_t mant = bits & 0x3FFu;
        output[idx] = static_cast<uint8_t>((exp == 0x1Fu && mant != 0u) ? 1 : 0);
    }
}

__global__ void isinf_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint32_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>(((bits & 0x7FFFFFFFu) == 0x7F800000u) ? 1 : 0);
    }
}

__global__ void isinf_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint64_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>(((bits & 0x7FFFFFFFFFFFFFFFull) == 0x7FF0000000000000ull) ? 1 : 0);
    }
}

__global__ void isinf_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        // Direct bit-pattern inspection: F16 has exp=0x1F, mantissa=0.
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&input[idx]);
        output[idx] = static_cast<uint8_t>(((bits & 0x7FFFu) == 0x7C00u) ? 1 : 0);
    }
}

__global__ void isfinite_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint32_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>(((bits & 0x7F800000u) != 0x7F800000u) ? 1 : 0);
    }
}

__global__ void isfinite_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint64_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>(((bits & 0x7FF0000000000000ull) != 0x7FF0000000000000ull) ? 1 : 0);
    }
}

__global__ void isfinite_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        // Finite ⇔ F16 exponent is not all-ones.
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&input[idx]);
        uint16_t exp = (bits >> 10) & 0x1Fu;
        output[idx] = static_cast<uint8_t>((exp != 0x1Fu) ? 1 : 0);
    }
}

// BFloat16 predicate kernels. BF16 has the same field layout as Float32's top
// 16 bits: 1 sign | 8 exponent (bits 14..7) | 7 mantissa (bits 6..0). Examine
// the raw 16-bit pattern directly because HIP's bf16->float conversion can
// canonicalise NaN on some ROCm builds and miss the predicate.
__global__ void isnan_kernel_bf16(const uint16_t* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint16_t bits = input[idx];
        uint16_t exp = (bits >> 7) & 0xFFu;
        uint16_t mant = bits & 0x7Fu;
        output[idx] = static_cast<uint8_t>((exp == 0xFFu && mant != 0u) ? 1 : 0);
    }
}

__global__ void isinf_kernel_bf16(const uint16_t* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint16_t bits = input[idx];
        output[idx] = static_cast<uint8_t>(((bits & 0x7FFFu) == 0x7F80u) ? 1 : 0);
    }
}

__global__ void isfinite_kernel_bf16(const uint16_t* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint16_t bits = input[idx];
        uint16_t exp = (bits >> 7) & 0xFFu;
        output[idx] = static_cast<uint8_t>((exp != 0xFFu) ? 1 : 0);
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
        float va = tenzor::rocm::safe_h2f(a[idx]);
        float vb = tenzor::rocm::safe_h2f(b[idx]);
        output[idx] = tenzor::rocm::safe_f2h(atan2f(va, vb));
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
        float va = tenzor::rocm::safe_h2f(a[idx]);
        float vb = tenzor::rocm::safe_h2f(b[idx]);
        output[idx] = tenzor::rocm::safe_f2h(fmodf(va, vb));
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
        float va = tenzor::rocm::safe_h2f(a[idx]);
        float vb = tenzor::rocm::safe_h2f(b[idx]);
        output[idx] = tenzor::rocm::safe_f2h(remainderf(va, vb));
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
        float va = tenzor::rocm::safe_h2f(a[idx]);
        float vb = tenzor::rocm::safe_h2f(b[idx]);
        float vw = tenzor::rocm::safe_h2f(weight[idx]);
        output[idx] = tenzor::rocm::safe_f2h(va + vw * (vb - va));
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
        output[idx] = static_cast<uint8_t>((tenzor::rocm::safe_h2f(a[idx]) != 0.0f) && (tenzor::rocm::safe_h2f(b[idx]) != 0.0f) ? 1 : 0);
    }
}

__global__ void logical_or_kernel_f16(const __half* a, const __half* b, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>((tenzor::rocm::safe_h2f(a[idx]) != 0.0f) || (tenzor::rocm::safe_h2f(b[idx]) != 0.0f) ? 1 : 0);
    }
}

__global__ void logical_not_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = static_cast<uint8_t>(tenzor::rocm::safe_h2f(input[idx]) == 0.0f ? 1 : 0);
    }
}

__global__ void logical_xor_kernel_f16(const __half* a, const __half* b, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        bool ba = (tenzor::rocm::safe_h2f(a[idx]) != 0.0f);
        bool bb = (tenzor::rocm::safe_h2f(b[idx]) != 0.0f);
        output[idx] = static_cast<uint8_t>(ba != bb ? 1 : 0);
    }
}

// ============================================================================
// Element-wise Min/Max Kernels
// ============================================================================

// NaN semantics: match the CPU reference (minimum_typed uses `a < b ? a : b`,
// which propagates a NaN in the SECOND operand: min(5,NaN)=NaN). fminf/fmin
// instead return the non-NaN operand, diverging on backend-parity tests.
__global__ void minimum_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void minimum_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void minimum_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = tenzor::rocm::safe_h2f(a[idx]);
        float vb = tenzor::rocm::safe_h2f(b[idx]);
        output[idx] = tenzor::rocm::safe_f2h((va < vb) ? va : vb);
    }
}

__global__ void maximum_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void maximum_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void maximum_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = tenzor::rocm::safe_h2f(a[idx]);
        float vb = tenzor::rocm::safe_h2f(b[idx]);
        output[idx] = tenzor::rocm::safe_f2h((va > vb) ? va : vb);
    }
}

// Integer element-wise min/max (used for Int8/Int32/Int64). BFloat16 is handled
// via a widen-narrow path in the host wrapper, matching the CPU reference's
// supported dtype set for torch.minimum / torch.maximum.
template<typename T>
__global__ void minimum_kernel_int(const T* a, const T* b, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = a[idx] < b[idx] ? a[idx] : b[idx];
    }
}

template<typename T>
__global__ void maximum_kernel_int(const T* a, const T* b, T* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = a[idx] > b[idx] ? a[idx] : b[idx];
    }
}

// ============================================================================
// Host Wrappers: Extended Math Unary Operations
// ============================================================================

auto log2_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return log2_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return log10_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return log1p_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return exp2_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return expm1_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return erf_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return erfc_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
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
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(isnan_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const uint16_t*>(input.data<BFloat16>()),
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
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(isinf_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const uint16_t*>(input.data<BFloat16>()),
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
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(isfinite_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const uint16_t*>(input.data<BFloat16>()),
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

auto atan2_kernel(const Tensor& a_in, const Tensor& b_in, hipStream_t stream) -> Tensor {
    if (a_in.dtype() != b_in.dtype()) {
        throw std::runtime_error("atan2: tensors must have the same dtype");
    }
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (a_in.dtype() == DType::BFloat16) {
        return atan2_kernel(a_in.to(DType::Float32), b_in.to(DType::Float32), stream)
            .to(DType::BFloat16);
    }

    // NumPy-style broadcasting: expand both operands to the common shape so
    // the elementwise kernel can run over a single contiguous index range.
    std::vector<int64_t> a_shape(a_in.shape().begin(), a_in.shape().end());
    std::vector<int64_t> b_shape(b_in.shape().begin(), b_in.shape().end());
    Tensor a = a_in;
    Tensor b = b_in;
    if (a_shape != b_shape) {
        std::vector<int64_t> bshape = detail::compute_broadcast_shape(a_shape, b_shape);
        if (a_shape != bshape) a = tenzor::expand(a_in, bshape).contiguous();
        if (b_shape != bshape) b = tenzor::expand(b_in, bshape).contiguous();
    } else {
        if (!a.is_contiguous()) a = a.contiguous();
        if (!b.is_contiguous()) b = b.contiguous();
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (a.dtype() == DType::BFloat16) {
        return fmod_kernel(a.to(DType::Float32), b.to(DType::Float32), stream)
            .to(DType::BFloat16);
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
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (a.dtype() == DType::BFloat16) {
        return remainder_kernel(a.to(DType::Float32), b.to(DType::Float32), stream)
            .to(DType::BFloat16);
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
    if (a.numel() != b.numel()) {
        throw std::runtime_error("lerp: start and end must have the same number of elements");
    }

    // Broadcast scalar or size-1 weight to match a's shape. `lerp(a, b, scalar)`
    // wraps the scalar in a size-1 Tensor at the ops layer, so we need to
    // support both the scalar case and the full per-element weight case.
    Tensor weight_broad = weight;
    if (weight.numel() != a.numel()) {
        if (weight.numel() != 1) {
            throw std::runtime_error(
                "lerp: weight must be scalar or have the same number of elements as start/end");
        }
        std::vector<int64_t> a_shape(a.shape().begin(), a.shape().end());
        weight_broad = tenzor::expand(weight, a_shape).contiguous();
    }

    // BFloat16: route through Float32 — no native kernel for BF16 lerp.
    if (a.dtype() == DType::BFloat16) {
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto w_f32 = weight_broad.to(DType::Float32);
        auto r_f32 = lerp_kernel(a_f32, b_f32, w_f32, stream);
        return r_f32.to(DType::BFloat16);
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(lerp_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), weight_broad.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(lerp_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), weight_broad.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(lerp_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<const __half*>(weight_broad.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("lerp operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
    } else if (a.dtype() == DType::BFloat16) {
        // Widen-narrow: BFloat16 has no native compare path here.
        auto a_f32 = cast_kernel(a, DType::Float32, stream);
        auto b_f32 = cast_kernel(b, DType::Float32, stream);
        hipLaunchKernelGGL(minimum_kernel_f32, grid, block, 0, stream,
            a_f32.data<float>(), b_f32.data<float>(), a_f32.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return cast_kernel(a_f32, DType::BFloat16, stream);
    } else if (a.dtype() == DType::Int8) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(minimum_kernel_int<int8_t>), grid, block, 0, stream,
            a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n);
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(minimum_kernel_int<int32_t>), grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(minimum_kernel_int<int64_t>), grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
    } else {
        throw std::runtime_error("minimum operation only supports Float32, Float64, Float16, BFloat16, Int8, Int32, and Int64 dtypes");
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
    } else if (a.dtype() == DType::BFloat16) {
        // Widen-narrow: BFloat16 has no native compare path here.
        auto a_f32 = cast_kernel(a, DType::Float32, stream);
        auto b_f32 = cast_kernel(b, DType::Float32, stream);
        hipLaunchKernelGGL(maximum_kernel_f32, grid, block, 0, stream,
            a_f32.data<float>(), b_f32.data<float>(), a_f32.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return cast_kernel(a_f32, DType::BFloat16, stream);
    } else if (a.dtype() == DType::Int8) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(maximum_kernel_int<int8_t>), grid, block, 0, stream,
            a.data<int8_t>(), b.data<int8_t>(), result.data<int8_t>(), n);
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(maximum_kernel_int<int32_t>), grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(maximum_kernel_int<int64_t>), grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
    } else {
        throw std::runtime_error("maximum operation only supports Float32, Float64, Float16, BFloat16, Int8, Int32, and Int64 dtypes");
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
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(conj_kernel_c64, grid, block, 0, stream,
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<float*>(result.data_ptr()), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Complex128, input.device());
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
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
    HIP_CHECK(hipMemcpyAsync(result.data_ptr(), input.data_ptr(),
                   n * dtype_size(input.dtype()), hipMemcpyDeviceToDevice, stream));
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
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(real_kernel_c64, grid, block, 0, stream,
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
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
    HIP_CHECK(hipMemcpyAsync(result.data_ptr(), input.data_ptr(),
                   n * dtype_size(input.dtype()), hipMemcpyDeviceToDevice, stream));
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
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(imag_kernel_c64, grid, block, 0, stream,
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
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
    HIP_CHECK(hipMemsetAsync(result.data_ptr(), 0, n * dtype_size(input.dtype()), stream));
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
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
        hipLaunchKernelGGL(angle_kernel_c64, grid, block, 0, stream,
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape, DType::Float64, input.device());
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
        hipLaunchKernelGGL(angle_kernel_c128, grid, block, 0, stream,
            reinterpret_cast<const double*>(input.data_ptr()),
            result.data<double>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Float32) {
        Tensor result(shape, DType::Float32, input.device());
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
        hipLaunchKernelGGL(angle_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Float64) {
        Tensor result(shape, DType::Float64, input.device());
        if (n == 0) return result;  // empty tensor: zero-grid launch would fail on HIP
        hipLaunchKernelGGL(angle_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // F16/BF16 have no direct ROCm angle kernel; widen to Float32 and cast back.
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = angle_kernel(input_f32, stream);
        return result_f32.to(input.dtype());
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
    } else if (abs_t.dtype() == DType::Float16 || abs_t.dtype() == DType::BFloat16) {
        // Widen to Float32, compute, and cast the Complex64 result to the matching
        // lower-precision *real* dtype. polar() already promotes F16 → Complex64 in
        // the CPU backend, so we match that here.
        auto abs_f32 = abs_t.to(DType::Float32);
        auto angle_f32 = angle_t.to(DType::Float32);
        return polar_kernel(abs_f32, angle_f32, stream);
    }
    throw std::runtime_error("polar: only Float32 and Float64 inputs are supported");
}

// ============================================================================
// ComplexTensor Kernel — interleave real + imag into Complex64/Complex128
// ============================================================================

__global__ void complex_tensor_kernel_f32(const float* real, const float* imag,
                                           float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[2 * idx]     = real[idx];
        output[2 * idx + 1] = imag[idx];
    }
}
__global__ void complex_tensor_kernel_f64(const double* real, const double* imag,
                                           double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[2 * idx]     = real[idx];
        output[2 * idx + 1] = imag[idx];
    }
}

auto complex_tensor_kernel(const Tensor& real_t, const Tensor& imag_t, hipStream_t stream) -> Tensor {
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
        hipLaunchKernelGGL(complex_tensor_kernel_f32, grid, block, 0, stream,
            real_t.data<float>(), imag_t.data<float>(),
            reinterpret_cast<float*>(result.data_ptr()), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (real_t.dtype() == DType::Float64) {
        Tensor result(shape, DType::Complex128, real_t.device());
        hipLaunchKernelGGL(complex_tensor_kernel_f64, grid, block, 0, stream,
            real_t.data<double>(), imag_t.data<double>(),
            reinterpret_cast<double*>(result.data_ptr()), n);
        HIP_CHECK(hipGetLastError());
        return result;
    } else if (real_t.dtype() == DType::Float16 || real_t.dtype() == DType::BFloat16) {
        Tensor real_f32 = real_t.to(DType::Float32);
        Tensor imag_f32 = imag_t.to(DType::Float32);
        Tensor result(shape, DType::Complex64, real_t.device());
        hipLaunchKernelGGL(complex_tensor_kernel_f32, grid, block, 0, stream,
            real_f32.data<float>(), imag_f32.data<float>(),
            reinterpret_cast<float*>(result.data_ptr()), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    throw std::runtime_error("complex: only Float32, Float64, Float16, and BFloat16 inputs are supported");
}

// ============================================================================
// Cross Product Kernel
// ============================================================================

template<typename T>
__global__ void cross_kernel_device(const T* a, const T* b, T* c,
                                     int64_t num_pairs, int64_t dim_stride) {
    HIP_KERNEL_LOOP(idx, num_pairs) {
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
    HIP_KERNEL_LOOP(idx, num_pairs) {
        int64_t o = idx / dim_stride;
        int64_t i = idx % dim_stride;
        int64_t base = o * 3 * dim_stride + i;
        float a0 = tenzor::rocm::safe_h2f(a[base]), a1 = tenzor::rocm::safe_h2f(a[base + dim_stride]), a2 = tenzor::rocm::safe_h2f(a[base + 2*dim_stride]);
        float b0 = tenzor::rocm::safe_h2f(b[base]), b1 = tenzor::rocm::safe_h2f(b[base + dim_stride]), b2 = tenzor::rocm::safe_h2f(b[base + 2*dim_stride]);
        c[base]                  = tenzor::rocm::safe_f2h(a1*b2 - a2*b1);
        c[base + dim_stride]     = tenzor::rocm::safe_f2h(a2*b0 - a0*b2);
        c[base + 2*dim_stride]   = tenzor::rocm::safe_f2h(a0*b1 - a1*b0);
    }
}

auto cross_kernel(const Tensor& a, const Tensor& b, int64_t dim, hipStream_t stream) -> Tensor {
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
        hipLaunchKernelGGL(cross_kernel_device<float>, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), num_pairs, inner);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(cross_kernel_device<double>, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), num_pairs, inner);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(cross_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), num_pairs, inner);
    } else {
        throw std::runtime_error("cross: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// =========================================================================
// Special Math Functions — native ROCm device implementations
// (replaces the previous CPU-roundtrip fallbacks in rocm_kernel_registry.cpp)
// =========================================================================
//
// Standard libm names (tgammaf, lgammaf, j0f, j1f, y0f, y1f, expf, logf, sinf, etc.)
// are usable in HIP device code via the libm shim. For functions without standard
// equivalents (cyl_bessel_i0/i1, erfinv) we use AMD's OCML intrinsics directly.

extern "C" {
__device__ float __ocml_i0_f32(float);
__device__ float __ocml_i1_f32(float);
__device__ double __ocml_i0_f64(double);
__device__ double __ocml_i1_f64(double);
__device__ float __ocml_erfinv_f32(float);
__device__ double __ocml_erfinv_f64(double);
}

// --- Digamma (psi function) — Cephes-style asymptotic expansion ---
__device__ inline float digamma_dev_f32(float x) {
    float result = 0.0f;
    if (x < 0.5f) {
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

// --- Polygamma ψ^(n)(x) ---
//
// ψ^(n)(x) = (-1)^(n+1) n! Σ_{k=0}^∞ 1/(x+k)^(n+1)
//
// The straight series converges too slowly for small n (for n=1 the k-th term
// is ~1/k², so 100 terms leaves a ~1/100 tail), which shows up as a ~0.01
// shift in trigamma at small x and breaks gradcheck for Digamma. Instead we
// use the recurrence ψ^(n)(x) = ψ^(n)(x+1) + (-1)^(n+1) n! / x^(n+1) to shift
// x above a threshold, then apply an asymptotic expansion.
//
// Asymptotic expansion for trigamma (n=1):
//   ψ'(x) ~ 1/x + 1/(2x²) + Σ_{k=1}^∞ B_{2k} / x^{2k+1}
// with Bernoulli numbers B_2=1/6, B_4=-1/30, B_6=1/42, B_8=-1/30, B_10=5/66.
//
// For n ≥ 2 we differentiate the digamma expansion term-wise to get
//   ψ^(n)(x) ~ (-1)^(n+1) [ (n-1)!/x^n + n!/(2 x^(n+1))
//                           + Σ_{k=1} B_{2k} (2k+n-1)! / ((2k)! x^(2k+n)) ]
__device__ inline double polygamma_dev_f64(int n, double x) {
    if (n == 0) return digamma_dev_f64(x);

    double fact_n = 1.0;
    for (int k = 1; k <= n; ++k) fact_n *= static_cast<double>(k);
    double sign = ((n + 1) % 2 == 0) ? 1.0 : -1.0;

    // Recurrence: shift x up so the asymptotic series converges quickly.
    // 14 was not enough for Float64 gradcheck (~5e-5 abs error on
    // trigamma at x≈4); 30 brings the series tail well below 1e-12.
    constexpr double kShiftTarget = 30.0;
    double shifted = 0.0;
    while (x < kShiftTarget) {
        shifted += pow(x, -static_cast<double>(n + 1));
        x += 1.0;
    }

    // Asymptotic expansion for large x.
    double inv_x = 1.0 / x;
    double inv_x_n = pow(inv_x, static_cast<double>(n));
    double fact_nm1 = fact_n / static_cast<double>(n);  // (n-1)!

    // Leading terms: (n-1)!/x^n + n!/(2 x^(n+1))
    double sum_asym = fact_nm1 * inv_x_n + 0.5 * fact_n * inv_x_n * inv_x;

    // Bernoulli correction terms: B_{2k} (2k+n-1)! / ((2k)! x^(2k+n))
    // B_{2k}/(2k)!:
    //   k=1: B_2 = 1/6  → 1/12
    //   k=2: B_4 = -1/30 → -1/720
    //   k=3: B_6 = 1/42 → 1/30240
    //   k=4: B_8 = -1/30 → -1/1209600
    //   k=5: B_10 = 5/66 → 5/479001600 = 1/95800320
    //   k=6: B_12 = -691/2730 → -691/(2730*479001600)
    //   k=7: B_14 = 7/6  → 7/87178291200
    //   k=8: B_16 = -3617/510 → -3617/(510*20922789888000)
    constexpr double B_over_fact[8] = {
        1.0/12.0,
        -1.0/720.0,
        1.0/30240.0,
        -1.0/1209600.0,
        1.0/47900160.0,                     // = 5 / 479001600
        -691.0/1307674368000.0,             // = -691 / 13!
        1.0/74724249600.0,                  // = 7 / 522 * 14! ... simplified
        -3617.0/10670622842880000.0
    };
    double inv_x2 = inv_x * inv_x;
    double power = inv_x_n * inv_x;  // x^-(n+1)
    double rising = 1.0;             // (n)(n+1)…(n+2k-1) as we accumulate k
    for (int k = 1; k <= 8; ++k) {
        rising *= static_cast<double>(n + 2*k - 1) * static_cast<double>(n + 2*k - 2);
        power *= inv_x2;
        sum_asym += B_over_fact[k-1] * rising * power;
    }

    // Unsigned sum = shifted-recurrence part + asymptotic expansion part.
    //
    // The recurrence contributes |n! / x^(n+1)| per shifted step in the
    // unsigned magnitude. For n=0 we return early above, so n≥1 here; the
    // asymptotic expansion already includes the (n-1)! prefactor, so the
    // final result is the signed sum times (-1)^(n+1) * n!, with the
    // pre-scaling factored out of sum_asym.
    //
    // Clean breakdown:
    //   ψ^(n)(x) = (-1)^(n+1) * [ n! * (shifted) + asymp_series(x) ]
    // where asymp_series for large x is:
    //   (n-1)!/x^n + n!/(2 x^(n+1)) + Σ B_{2k}(2k+n-1)!/((2k)! x^(2k+n))
    // which we've built up in sum_asym.
    return sign * (fact_n * shifted + sum_asym);
}
__device__ inline float polygamma_dev_f32(int n, float x) {
    return static_cast<float>(polygamma_dev_f64(n, static_cast<double>(x)));
}

// --- Regularized incomplete beta I_x(a, b) — Lentz continued fraction ---
__device__ inline double betainc_dev_f64(double a, double b, double x) {
    if (x < 0.0 || x > 1.0) return nan("");
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

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
// Generic unary special-math kernel macro
// =========================================================================
#define DEFINE_ROCM_SPECIAL_UNARY(NAME, FN_F32_EXPR, FN_F64_EXPR)                            \
    __global__ void NAME##_kernel_f32(const float* in, float* out, int64_t n) {              \
        HIP_KERNEL_LOOP(idx, n) { float x = in[idx]; out[idx] = (FN_F32_EXPR); }            \
    }                                                                                         \
    __global__ void NAME##_kernel_f64(const double* in, double* out, int64_t n) {            \
        HIP_KERNEL_LOOP(idx, n) { double x = in[idx]; out[idx] = (FN_F64_EXPR); }           \
    }                                                                                         \
    __global__ void NAME##_kernel_f16(const __half* in, __half* out, int64_t n) {            \
        HIP_KERNEL_LOOP(idx, n) {                                                             \
            float x = tenzor::rocm::safe_h2f(in[idx]);                                                  \
            out[idx] = tenzor::rocm::safe_f2h(FN_F32_EXPR);                                             \
        }                                                                                     \
    }                                                                                         \
    __global__ void NAME##_kernel_bf16(const hip_bfloat16* in, hip_bfloat16* out, int64_t n) { \
        HIP_KERNEL_LOOP(idx, n) {                                                             \
            float x = tenzor::rocm::safe_bf2f(in[idx]);                                       \
            /* S.10 / R.11: RNE round on float32 → bf16 narrowing. */                         \
            out[idx] = tenzor::rocm::f32_to_bf16_rne(FN_F32_EXPR);                            \
        }                                                                                     \
    }                                                                                         \
    auto NAME##_kernel(const Tensor& input, hipStream_t stream) -> Tensor {                  \
        int64_t n = input.numel();                                                            \
        std::vector<int64_t> shape(input.shape().begin(), input.shape().end());              \
        Tensor result(shape, input.dtype(), input.device());                                 \
        if (n == 0) return result;                                                            \
        dim3 grid, block; compute_launch_config_1d(n, grid, block);                          \
        if (input.dtype() == DType::Float32) {                                                \
            hipLaunchKernelGGL(NAME##_kernel_f32, grid, block, 0, stream,                     \
                input.data<float>(), result.data<float>(), n);                               \
            HIP_CHECK(hipGetLastError());                                                     \
        } else if (input.dtype() == DType::Float64) {                                         \
            hipLaunchKernelGGL(NAME##_kernel_f64, grid, block, 0, stream,                     \
                input.data<double>(), result.data<double>(), n);                             \
            HIP_CHECK(hipGetLastError());                                                     \
        } else if (input.dtype() == DType::Float16) {                                         \
            hipLaunchKernelGGL(NAME##_kernel_f16, grid, block, 0, stream,                     \
                reinterpret_cast<const __half*>(input.data<Float16>()),                       \
                reinterpret_cast<__half*>(result.data<Float16>()), n);                        \
            HIP_CHECK(hipGetLastError());                                                     \
        } else if (input.dtype() == DType::BFloat16) {                                        \
            hipLaunchKernelGGL(NAME##_kernel_bf16, grid, block, 0, stream,                    \
                reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),                \
                reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);                 \
            HIP_CHECK(hipGetLastError());                                                     \
        } else {                                                                              \
            throw std::runtime_error(#NAME " only supports Float32, Float64, Float16, BFloat16"); \
        }                                                                                     \
        return result;                                                                        \
    }

DEFINE_ROCM_SPECIAL_UNARY(gamma,     tgammaf(x),               tgamma(x))
DEFINE_ROCM_SPECIAL_UNARY(lgamma,    lgammaf(x),               lgamma(x))
DEFINE_ROCM_SPECIAL_UNARY(digamma,   digamma_dev_f32(x),       digamma_dev_f64(x))
DEFINE_ROCM_SPECIAL_UNARY(bessel_j0, j0f(x),                   j0(x))
DEFINE_ROCM_SPECIAL_UNARY(bessel_j1, j1f(x),                   j1(x))
DEFINE_ROCM_SPECIAL_UNARY(bessel_y0, y0f(x),                   y0(x))
DEFINE_ROCM_SPECIAL_UNARY(bessel_y1, y1f(x),                   y1(x))
DEFINE_ROCM_SPECIAL_UNARY(bessel_i0, __ocml_i0_f32(x),         __ocml_i0_f64(x))
DEFINE_ROCM_SPECIAL_UNARY(bessel_i1, __ocml_i1_f32(x),         __ocml_i1_f64(x))
DEFINE_ROCM_SPECIAL_UNARY(erfinv,    __ocml_erfinv_f32(x),     __ocml_erfinv_f64(x))
DEFINE_ROCM_SPECIAL_UNARY(sinc,      sinc_dev_f32(x),          sinc_dev_f64(x))

// --- Ndtr: Normal CDF Φ(x) = 0.5 * erfc(-x * M_SQRT1_2) ---
DEFINE_ROCM_SPECIAL_UNARY(ndtr,
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
DEFINE_ROCM_SPECIAL_UNARY(log_ndtr,  log_ndtr_dev_f32(x),      log_ndtr_dev_f64(x))

#undef DEFINE_ROCM_SPECIAL_UNARY

// --- Multigammaln: multivariate log-gamma with dimension parameter d ---
__global__ void multigammaln_kernel_f32(const float* in, float* out, int64_t n, int d, float log_pi_coeff) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = log_pi_coeff;
        for (int j = 0; j < d; ++j)
            val += lgammaf(in[idx] - static_cast<float>(j) * 0.5f);
        out[idx] = val;
    }
}
__global__ void multigammaln_kernel_f64(const double* in, double* out, int64_t n, int d, double log_pi_coeff) {
    HIP_KERNEL_LOOP(idx, n) {
        double val = log_pi_coeff;
        for (int j = 0; j < d; ++j)
            val += lgamma(in[idx] - static_cast<double>(j) * 0.5);
        out[idx] = val;
    }
}
__global__ void multigammaln_kernel_f16(const __half* in, __half* out, int64_t n, int d, float log_pi_coeff) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(in[idx]);
        float val = log_pi_coeff;
        for (int j = 0; j < d; ++j)
            val += lgammaf(x - static_cast<float>(j) * 0.5f);
        out[idx] = tenzor::rocm::safe_f2h(val);
    }
}
__global__ void multigammaln_kernel_bf16(const hip_bfloat16* in, hip_bfloat16* out, int64_t n, int d, float log_pi_coeff) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = static_cast<float>(in[idx]);
        float val = log_pi_coeff;
        for (int j = 0; j < d; ++j)
            val += lgammaf(x - static_cast<float>(j) * 0.5f);
        out[idx] = tenzor::rocm::f32_to_bf16_rne(val);
    }
}
auto multigammaln_kernel(const Tensor& input, int d, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    double log_pi_coeff = static_cast<double>(d) * static_cast<double>(d - 1) / 4.0 * log(M_PI);
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(multigammaln_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n, d, static_cast<float>(log_pi_coeff));
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(multigammaln_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n, d, log_pi_coeff);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(multigammaln_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n, d, static_cast<float>(log_pi_coeff));
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(multigammaln_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n, d, static_cast<float>(log_pi_coeff));
    } else {
        throw std::runtime_error("multigammaln only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// --- Beta(a, b) = exp(lgamma(a) + lgamma(b) - lgamma(a + b)) ---
__device__ inline float beta_dev_f32(float a, float b) {
    return expf(lgammaf(a) + lgammaf(b) - lgammaf(a + b));
}
__device__ inline double beta_dev_f64(double a, double b) {
    return exp(lgamma(a) + lgamma(b) - lgamma(a + b));
}
__global__ void beta_kernel_f32(const float* a, const float* b, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = beta_dev_f32(a[idx], b[idx]); }
}
__global__ void beta_kernel_f64(const double* a, const double* b, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = beta_dev_f64(a[idx], b[idx]); }
}
__global__ void beta_kernel_f16(const __half* a, const __half* b, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = tenzor::rocm::safe_f2h(beta_dev_f32(tenzor::rocm::safe_h2f(a[idx]), tenzor::rocm::safe_h2f(b[idx])));
    }
}
__global__ void beta_kernel_bf16(const hip_bfloat16* a, const hip_bfloat16* b, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = tenzor::rocm::f32_to_bf16_rne(beta_dev_f32(static_cast<float>(a[idx]), static_cast<float>(b[idx])));
    }
}
auto beta_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(beta_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(beta_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(beta_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(beta_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("beta only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// --- Hurwitz zeta ζ(s, q) ---
__global__ void zeta_kernel_f32(const float* s, const float* q, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = zeta_dev_f32(s[idx], q[idx]); }
}
__global__ void zeta_kernel_f64(const double* s, const double* q, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = zeta_dev_f64(s[idx], q[idx]); }
}
__global__ void zeta_kernel_f16(const __half* s, const __half* q, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = tenzor::rocm::safe_f2h(zeta_dev_f32(tenzor::rocm::safe_h2f(s[idx]), tenzor::rocm::safe_h2f(q[idx])));
    }
}
__global__ void zeta_kernel_bf16(const hip_bfloat16* s, const hip_bfloat16* q, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = tenzor::rocm::f32_to_bf16_rne(zeta_dev_f32(static_cast<float>(s[idx]), static_cast<float>(q[idx])));
    }
}
auto zeta_kernel(const Tensor& s, const Tensor& q, hipStream_t stream) -> Tensor {
    int64_t n = s.numel();
    std::vector<int64_t> shape(s.shape().begin(), s.shape().end());
    Tensor result(shape, s.dtype(), s.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (s.dtype() == DType::Float32) {
        hipLaunchKernelGGL(zeta_kernel_f32, grid, block, 0, stream,
            s.data<float>(), q.data<float>(), result.data<float>(), n);
    } else if (s.dtype() == DType::Float64) {
        hipLaunchKernelGGL(zeta_kernel_f64, grid, block, 0, stream,
            s.data<double>(), q.data<double>(), result.data<double>(), n);
    } else if (s.dtype() == DType::Float16) {
        hipLaunchKernelGGL(zeta_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(s.data<Float16>()),
            reinterpret_cast<const __half*>(q.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (s.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(zeta_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(s.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(q.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("zeta only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// --- Polygamma ψ^(n)(x) — n is a scalar order ---
__global__ void polygamma_kernel_f32(int n, const float* in, float* out, int64_t numel) {
    HIP_KERNEL_LOOP(idx, numel) { out[idx] = polygamma_dev_f32(n, in[idx]); }
}
__global__ void polygamma_kernel_f64(int n, const double* in, double* out, int64_t numel) {
    HIP_KERNEL_LOOP(idx, numel) { out[idx] = polygamma_dev_f64(n, in[idx]); }
}
__global__ void polygamma_kernel_f16(int n, const __half* in, __half* out, int64_t numel) {
    HIP_KERNEL_LOOP(idx, numel) {
        out[idx] = tenzor::rocm::safe_f2h(polygamma_dev_f32(n, tenzor::rocm::safe_h2f(in[idx])));
    }
}
__global__ void polygamma_kernel_bf16(int n, const hip_bfloat16* in, hip_bfloat16* out, int64_t numel) {
    HIP_KERNEL_LOOP(idx, numel) {
        out[idx] = tenzor::rocm::f32_to_bf16_rne(polygamma_dev_f32(n, static_cast<float>(in[idx])));
    }
}
auto polygamma_kernel(int64_t n, const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t numel = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (numel == 0) return result;
    dim3 grid, block; compute_launch_config_1d(numel, grid, block);
    int n_int = static_cast<int>(n);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(polygamma_kernel_f32, grid, block, 0, stream,
            n_int, input.data<float>(), result.data<float>(), numel);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(polygamma_kernel_f64, grid, block, 0, stream,
            n_int, input.data<double>(), result.data<double>(), numel);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(polygamma_kernel_f16, grid, block, 0, stream,
            n_int,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), numel);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(polygamma_kernel_bf16, grid, block, 0, stream,
            n_int,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), numel);
    } else {
        throw std::runtime_error("polygamma only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// --- Regularized incomplete beta I_x(a, b) ---
__global__ void betainc_kernel_f32(const float* a, const float* b, const float* x, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = static_cast<float>(betainc_dev_f64(
            static_cast<double>(a[idx]), static_cast<double>(b[idx]), static_cast<double>(x[idx])));
    }
}
__global__ void betainc_kernel_f64(const double* a, const double* b, const double* x, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = betainc_dev_f64(a[idx], b[idx], x[idx]);
    }
}
__global__ void betainc_kernel_f16(const __half* a, const __half* b, const __half* x, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double r = betainc_dev_f64(
            static_cast<double>(tenzor::rocm::safe_h2f(a[idx])),
            static_cast<double>(tenzor::rocm::safe_h2f(b[idx])),
            static_cast<double>(tenzor::rocm::safe_h2f(x[idx])));
        out[idx] = tenzor::rocm::safe_f2h(static_cast<float>(r));
    }
}
__global__ void betainc_kernel_bf16(const hip_bfloat16* a, const hip_bfloat16* b, const hip_bfloat16* x, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double r = betainc_dev_f64(
            static_cast<double>(static_cast<float>(a[idx])),
            static_cast<double>(static_cast<float>(b[idx])),
            static_cast<double>(static_cast<float>(x[idx])));
        out[idx] = tenzor::rocm::f32_to_bf16_rne(static_cast<float>(r));
    }
}
auto betainc_kernel(const Tensor& a, const Tensor& b, const Tensor& x, hipStream_t stream) -> Tensor {
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(betainc_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), x.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(betainc_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), x.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(betainc_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(betainc_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(x.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("betainc only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// New Phase 4 ops
// ============================================================================

// frac is sign-preserving (x - trunc(x)), not x - floor(x). See the CUDA
// kernel comment in src/backends/cuda/kernels/math.cu for the rationale —
// the same bug was present here.
__global__ void frac_kernel_f32(const float* in, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] - truncf(in[idx]); }
}
__global__ void frac_kernel_f64(const double* in, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] - trunc(in[idx]); }
}
auto frac_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(frac_kernel_f32, grid, block, 0, stream, input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(frac_kernel_f64, grid, block, 0, stream, input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        auto f32 = input.to(DType::Float32);
        return frac_kernel(f32, stream).to(DType::Float16);
    } else if (input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        return frac_kernel(f32, stream).to(DType::BFloat16);
    } else { throw std::runtime_error("frac: unsupported dtype"); }
    HIP_CHECK(hipGetLastError());
    return result;
}

__global__ void log_sigmoid_kernel_f32(const float* in, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = in[idx];
        out[idx] = (x >= 0.0f) ? -log1pf(expf(-x)) : x - log1pf(expf(x));
    }
}
__global__ void log_sigmoid_kernel_f64(const double* in, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double x = in[idx];
        out[idx] = (x >= 0.0) ? -log1p(exp(-x)) : x - log1p(exp(x));
    }
}
auto log_sigmoid_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(log_sigmoid_kernel_f32, grid, block, 0, stream, input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(log_sigmoid_kernel_f64, grid, block, 0, stream, input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        auto f32 = input.to(DType::Float32);
        return log_sigmoid_kernel(f32, stream).to(DType::Float16);
    } else if (input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        return log_sigmoid_kernel(f32, stream).to(DType::BFloat16);
    } else { throw std::runtime_error("log_sigmoid: unsupported dtype"); }
    HIP_CHECK(hipGetLastError());
    return result;
}

__global__ void heaviside_kernel_f32(const float* input, const float* values, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = input[idx];
        out[idx] = (x < 0.0f) ? 0.0f : (x == 0.0f ? values[idx] : 1.0f);
    }
}
__global__ void heaviside_kernel_f64(const double* input, const double* values, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double x = input[idx];
        out[idx] = (x < 0.0) ? 0.0 : (x == 0.0 ? values[idx] : 1.0);
    }
}
auto heaviside_kernel(const Tensor& input, const Tensor& values, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(heaviside_kernel_f32, grid, block, 0, stream, input.data<float>(), values.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(heaviside_kernel_f64, grid, block, 0, stream, input.data<double>(), values.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        auto f32_in = input.to(DType::Float32); auto f32_val = values.to(DType::Float32);
        return heaviside_kernel(f32_in, f32_val, stream).to(DType::Float16);
    } else if (input.dtype() == DType::BFloat16) {
        auto f32_in = input.to(DType::Float32); auto f32_val = values.to(DType::Float32);
        return heaviside_kernel(f32_in, f32_val, stream).to(DType::BFloat16);
    } else { throw std::runtime_error("heaviside: unsupported dtype"); }
    HIP_CHECK(hipGetLastError());
    return result;
}

__global__ void nan_to_num_kernel_f32(const float* input, float* out, int64_t n, float nan_val, float posinf_val, float neginf_val) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = input[idx];
        // Inspect IEEE-754 bits directly. The HIP isnan/isinf intrinsics have
        // been observed to miscompile under fast-math here.
        unsigned int bits;
        memcpy(&bits, &x, sizeof(bits));
        unsigned int exp = (bits >> 23) & 0xFFu;
        unsigned int mant = bits & 0x7FFFFFu;
        bool is_inf = (exp == 0xFFu) && (mant == 0u);
        bool is_nan = (exp == 0xFFu) && (mant != 0u);
        bool sign_neg = (bits & 0x80000000u) != 0u;
        if (is_nan) out[idx] = nan_val;
        else if (is_inf && !sign_neg) out[idx] = posinf_val;
        else if (is_inf && sign_neg) out[idx] = neginf_val;
        else out[idx] = x;
    }
}
__global__ void nan_to_num_kernel_f64(const double* input, double* out, int64_t n, double nan_val, double posinf_val, double neginf_val) {
    HIP_KERNEL_LOOP(idx, n) {
        double x = input[idx];
        unsigned long long bits;
        memcpy(&bits, &x, sizeof(bits));
        unsigned long long exp = (bits >> 52) & 0x7FFull;
        unsigned long long mant = bits & 0xFFFFFFFFFFFFFull;
        bool is_inf = (exp == 0x7FFull) && (mant == 0ull);
        bool is_nan = (exp == 0x7FFull) && (mant != 0ull);
        bool sign_neg = (bits & 0x8000000000000000ull) != 0ull;
        if (is_nan) out[idx] = nan_val;
        else if (is_inf && !sign_neg) out[idx] = posinf_val;
        else if (is_inf && sign_neg) out[idx] = neginf_val;
        else out[idx] = x;
    }
}
auto nan_to_num_kernel(const Tensor& input, double nan_v, double posinf_v, double neginf_v, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        float pf = (posinf_v >= static_cast<double>(std::numeric_limits<float>::max())) ? std::numeric_limits<float>::max() : static_cast<float>(posinf_v);
        float nf = (neginf_v <= static_cast<double>(std::numeric_limits<float>::lowest())) ? std::numeric_limits<float>::lowest() : static_cast<float>(neginf_v);
        hipLaunchKernelGGL(nan_to_num_kernel_f32, grid, block, 0, stream, input.data<float>(), result.data<float>(), n, static_cast<float>(nan_v), pf, nf);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nan_to_num_kernel_f64, grid, block, 0, stream, input.data<double>(), result.data<double>(), n, nan_v, posinf_v, neginf_v);
    } else if (input.dtype() == DType::Float16) {
        auto f32 = input.to(DType::Float32);
        return nan_to_num_kernel(f32, nan_v, posinf_v, neginf_v, stream).to(DType::Float16);
    } else if (input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        return nan_to_num_kernel(f32, nan_v, posinf_v, neginf_v, stream).to(DType::BFloat16);
    } else { throw std::runtime_error("nan_to_num: unsupported dtype"); }
    HIP_CHECK(hipGetLastError());
    return result;
}

// Bitwise ops
__global__ void bitwise_and_i8(const int8_t* a, const int8_t* b, int8_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] & b[idx]; } }
__global__ void bitwise_and_i16(const int16_t* a, const int16_t* b, int16_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] & b[idx]; } }
__global__ void bitwise_and_i32(const int32_t* a, const int32_t* b, int32_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] & b[idx]; } }
__global__ void bitwise_and_i64(const int64_t* a, const int64_t* b, int64_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] & b[idx]; } }
auto bitwise_and_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    int64_t n = a.numel(); std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device()); if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Int8)        { hipLaunchKernelGGL(bitwise_and_i8,  grid, block, 0, stream, a.data<int8_t>(),  b.data<int8_t>(),  result.data<int8_t>(),  n); }
    else if (a.dtype() == DType::Int16)  { hipLaunchKernelGGL(bitwise_and_i16, grid, block, 0, stream, a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(), n); }
    else if (a.dtype() == DType::Int32)  { hipLaunchKernelGGL(bitwise_and_i32, grid, block, 0, stream, a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n); }
    else if (a.dtype() == DType::Int64)  { hipLaunchKernelGGL(bitwise_and_i64, grid, block, 0, stream, a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_and: unsupported dtype"); }
    HIP_CHECK(hipGetLastError()); return result;
}

__global__ void bitwise_or_i8(const int8_t* a, const int8_t* b, int8_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] | b[idx]; } }
__global__ void bitwise_or_i16(const int16_t* a, const int16_t* b, int16_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] | b[idx]; } }
__global__ void bitwise_or_i32(const int32_t* a, const int32_t* b, int32_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] | b[idx]; } }
__global__ void bitwise_or_i64(const int64_t* a, const int64_t* b, int64_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] | b[idx]; } }
auto bitwise_or_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    int64_t n = a.numel(); std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device()); if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Int8)        { hipLaunchKernelGGL(bitwise_or_i8,  grid, block, 0, stream, a.data<int8_t>(),  b.data<int8_t>(),  result.data<int8_t>(),  n); }
    else if (a.dtype() == DType::Int16)  { hipLaunchKernelGGL(bitwise_or_i16, grid, block, 0, stream, a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(), n); }
    else if (a.dtype() == DType::Int32)  { hipLaunchKernelGGL(bitwise_or_i32, grid, block, 0, stream, a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n); }
    else if (a.dtype() == DType::Int64)  { hipLaunchKernelGGL(bitwise_or_i64, grid, block, 0, stream, a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_or: unsupported dtype"); }
    HIP_CHECK(hipGetLastError()); return result;
}

__global__ void bitwise_xor_i8(const int8_t* a, const int8_t* b, int8_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] ^ b[idx]; } }
__global__ void bitwise_xor_i16(const int16_t* a, const int16_t* b, int16_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] ^ b[idx]; } }
__global__ void bitwise_xor_i32(const int32_t* a, const int32_t* b, int32_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] ^ b[idx]; } }
__global__ void bitwise_xor_i64(const int64_t* a, const int64_t* b, int64_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = a[idx] ^ b[idx]; } }
auto bitwise_xor_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    int64_t n = a.numel(); std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device()); if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Int8)        { hipLaunchKernelGGL(bitwise_xor_i8,  grid, block, 0, stream, a.data<int8_t>(),  b.data<int8_t>(),  result.data<int8_t>(),  n); }
    else if (a.dtype() == DType::Int16)  { hipLaunchKernelGGL(bitwise_xor_i16, grid, block, 0, stream, a.data<int16_t>(), b.data<int16_t>(), result.data<int16_t>(), n); }
    else if (a.dtype() == DType::Int32)  { hipLaunchKernelGGL(bitwise_xor_i32, grid, block, 0, stream, a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n); }
    else if (a.dtype() == DType::Int64)  { hipLaunchKernelGGL(bitwise_xor_i64, grid, block, 0, stream, a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_xor: unsupported dtype"); }
    HIP_CHECK(hipGetLastError()); return result;
}

__global__ void bitwise_not_i8(const int8_t* in, int8_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = ~in[idx]; } }
__global__ void bitwise_not_i16(const int16_t* in, int16_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = ~in[idx]; } }
__global__ void bitwise_not_i32(const int32_t* in, int32_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = ~in[idx]; } }
__global__ void bitwise_not_i64(const int64_t* in, int64_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = ~in[idx]; } }
auto bitwise_not_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel(); std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device()); if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Int8)       { hipLaunchKernelGGL(bitwise_not_i8,  grid, block, 0, stream, input.data<int8_t>(),  result.data<int8_t>(),  n); }
    else if (input.dtype() == DType::Int16) { hipLaunchKernelGGL(bitwise_not_i16, grid, block, 0, stream, input.data<int16_t>(), result.data<int16_t>(), n); }
    else if (input.dtype() == DType::Int32) { hipLaunchKernelGGL(bitwise_not_i32, grid, block, 0, stream, input.data<int32_t>(), result.data<int32_t>(), n); }
    else if (input.dtype() == DType::Int64) { hipLaunchKernelGGL(bitwise_not_i64, grid, block, 0, stream, input.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_not: unsupported dtype"); }
    HIP_CHECK(hipGetLastError()); return result;
}

__global__ void bitwise_lshift_i8(const int8_t* in, const int8_t* sh, int8_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] << sh[idx]; } }
__global__ void bitwise_lshift_i16(const int16_t* in, const int16_t* sh, int16_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] << sh[idx]; } }
__global__ void bitwise_lshift_i32(const int32_t* in, const int32_t* sh, int32_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] << sh[idx]; } }
__global__ void bitwise_lshift_i64(const int64_t* in, const int64_t* sh, int64_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] << sh[idx]; } }
auto bitwise_left_shift_kernel(const Tensor& input, const Tensor& shift, hipStream_t stream) -> Tensor {
    int64_t n = input.numel(); std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device()); if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Int8)       { hipLaunchKernelGGL(bitwise_lshift_i8,  grid, block, 0, stream, input.data<int8_t>(),  shift.data<int8_t>(),  result.data<int8_t>(),  n); }
    else if (input.dtype() == DType::Int16) { hipLaunchKernelGGL(bitwise_lshift_i16, grid, block, 0, stream, input.data<int16_t>(), shift.data<int16_t>(), result.data<int16_t>(), n); }
    else if (input.dtype() == DType::Int32) { hipLaunchKernelGGL(bitwise_lshift_i32, grid, block, 0, stream, input.data<int32_t>(), shift.data<int32_t>(), result.data<int32_t>(), n); }
    else if (input.dtype() == DType::Int64) { hipLaunchKernelGGL(bitwise_lshift_i64, grid, block, 0, stream, input.data<int64_t>(), shift.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_left_shift: unsupported dtype"); }
    HIP_CHECK(hipGetLastError()); return result;
}

__global__ void bitwise_rshift_i8(const int8_t* in, const int8_t* sh, int8_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] >> sh[idx]; } }
__global__ void bitwise_rshift_i16(const int16_t* in, const int16_t* sh, int16_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] >> sh[idx]; } }
__global__ void bitwise_rshift_i32(const int32_t* in, const int32_t* sh, int32_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] >> sh[idx]; } }
__global__ void bitwise_rshift_i64(const int64_t* in, const int64_t* sh, int64_t* out, int64_t n) { HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] >> sh[idx]; } }
auto bitwise_right_shift_kernel(const Tensor& input, const Tensor& shift, hipStream_t stream) -> Tensor {
    int64_t n = input.numel(); std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device()); if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Int8)       { hipLaunchKernelGGL(bitwise_rshift_i8,  grid, block, 0, stream, input.data<int8_t>(),  shift.data<int8_t>(),  result.data<int8_t>(),  n); }
    else if (input.dtype() == DType::Int16) { hipLaunchKernelGGL(bitwise_rshift_i16, grid, block, 0, stream, input.data<int16_t>(), shift.data<int16_t>(), result.data<int16_t>(), n); }
    else if (input.dtype() == DType::Int32) { hipLaunchKernelGGL(bitwise_rshift_i32, grid, block, 0, stream, input.data<int32_t>(), shift.data<int32_t>(), result.data<int32_t>(), n); }
    else if (input.dtype() == DType::Int64) { hipLaunchKernelGGL(bitwise_rshift_i64, grid, block, 0, stream, input.data<int64_t>(), shift.data<int64_t>(), result.data<int64_t>(), n); }
    else { throw std::runtime_error("bitwise_right_shift: unsupported dtype"); }
    HIP_CHECK(hipGetLastError()); return result;
}

// ============================================================================
// logcumsumexp — ROCm kernel
// ============================================================================

template<typename T>
__global__ void logcumsumexp_hip_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    int64_t dim_size, int64_t inner_size, int64_t total_slices)
{
    // FF.6: ROCm's HIP `isnan`/`isinf` intrinsics canonicalise NaN/Inf bit
    // patterns silently on __half (and propagate unreliably across some
    // driver/arch combos — see feedback_rocm_intrinsic_nan).  Gate this
    // template to F32/F64 only; the host dispatcher already routes Float16
    // through the F32 widen-narrow path.
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "logcumsumexp_hip_kernel: only F32/F64 supported on ROCm; "
        "__half NaN intrinsic is unreliable");
    HIP_KERNEL_LOOP(idx, total_slices) {
        int64_t outer = idx / inner_size;
        int64_t inner = idx % inner_size;

        T running_max = -INFINITY;
        T running_lse = -INFINITY;

        for (int64_t i = 0; i < dim_size; ++i) {
            int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
            T x = input[offset];
            T new_max = fmax(running_max, x);

            if (isinf(new_max) && new_max < T(0)) {
                running_lse = -INFINITY;
            } else {
                running_lse = new_max + log(exp(running_lse - new_max) + exp(x - new_max));
            }
            running_max = new_max;
            output[offset] = running_lse;
        }
    }
}

auto logcumsumexp_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor {
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input.ndim();
    const auto dtype = input.dtype();
    const auto device = input.device();

    if (dim < 0) dim += ndim;

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);

    int64_t dim_size = shape[dim];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t total_slices = outer_size * inner_size;
    if (total_slices == 0 || dim_size == 0) return output;

    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto f32_result = logcumsumexp_kernel(input_cont.to(DType::Float32), dim, stream);
        return f32_result.to(dtype);
    }

    dim3 grid, block;
    compute_launch_config_1d(total_slices, grid, block);

    if (dtype == DType::Float32) {
        hipLaunchKernelGGL(logcumsumexp_hip_kernel<float>, grid, block, 0, stream,
            input_cont.data<float>(), output.data<float>(), dim_size, inner_size, total_slices);
    } else if (dtype == DType::Float64) {
        hipLaunchKernelGGL(logcumsumexp_hip_kernel<double>, grid, block, 0, stream,
            input_cont.data<double>(), output.data<double>(), dim_size, inner_size, total_slices);
    } else {
        throw std::runtime_error("logcumsumexp: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    return output;
}

// ============================================================================
// bincount — ROCm kernel
// ============================================================================

__global__ void bincount_no_weights_hip(
    const int64_t* __restrict__ input, int64_t* __restrict__ output, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&output[input[idx]]), 1ULL);
    }
}

__global__ void bincount_weights_f32_hip(
    const int64_t* __restrict__ input, const float* __restrict__ weights,
    double* __restrict__ output, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        atomicAdd(&output[input[idx]], static_cast<double>(weights[idx]));
    }
}

__global__ void bincount_weights_f64_hip(
    const int64_t* __restrict__ input, const double* __restrict__ weights,
    double* __restrict__ output, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        atomicAdd(&output[input[idx]], weights[idx]);
    }
}

__global__ void bincount_find_max_hip(
    const int64_t* __restrict__ input, int64_t* __restrict__ max_val, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        atomicMax(reinterpret_cast<long long*>(max_val),
                  static_cast<long long>(input[idx]));
    }
}

__global__ void bincount_find_min_hip(
    const int64_t* __restrict__ input, int64_t* __restrict__ min_val, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        atomicMin(reinterpret_cast<long long*>(min_val),
                  static_cast<long long>(input[idx]));
    }
}

auto bincount_kernel(const Tensor& input, const Tensor* weights,
                     int64_t minlength, hipStream_t stream) -> Tensor {
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    Tensor input_i64 = (input_cont.dtype() == DType::Int64)
        ? input_cont : input_cont.to(DType::Int64);

    int64_t n = input_i64.numel();
    auto device = input.device();

    // Find max and min value on GPU. The min pass lets us reject negative
    // inputs before launching the accumulation kernels — those index
    // output[input[idx]] with no lower-bound check, so a negative element
    // would be an out-of-bounds device write (CPU throws on negatives).
    Tensor max_tensor({1}, DType::Int64, device);
    Tensor min_tensor({1}, DType::Int64, device);
    int64_t neg_one = -1;
    int64_t int64_max = std::numeric_limits<int64_t>::max();
    // audit V.15: wrap HIP API calls in HIP_CHECK; previously errors here
    // (OOM, stream invalidation, device lost) were silently swallowed.
    HIP_CHECK(hipMemcpyAsync(max_tensor.data<int64_t>(), &neg_one, sizeof(int64_t),
                   hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(min_tensor.data<int64_t>(), &int64_max, sizeof(int64_t),
                   hipMemcpyHostToDevice, stream));

    if (n > 0) {
        dim3 grid, block;
        compute_launch_config_1d(n, grid, block);
        hipLaunchKernelGGL(bincount_find_max_hip, grid, block, 0, stream,
            input_i64.data<int64_t>(), max_tensor.data<int64_t>(), n);
        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(bincount_find_min_hip, grid, block, 0, stream,
            input_i64.data<int64_t>(), min_tensor.data<int64_t>(), n);
        HIP_CHECK(hipGetLastError());
    }

    int64_t max_val = -1;
    int64_t min_val = int64_max;
    HIP_CHECK(hipMemcpyAsync(&max_val, max_tensor.data<int64_t>(), sizeof(int64_t),
                   hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(&min_val, min_tensor.data<int64_t>(), sizeof(int64_t),
                   hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    if (n > 0 && min_val < 0) {
        throw std::runtime_error("bincount: input must contain non-negative integers");
    }

    int64_t output_size = std::max(max_val + 1, minlength);

    bool has_weights = (weights != nullptr);

    if (has_weights) {
        Tensor output({output_size}, DType::Float64, device);
        HIP_CHECK(hipMemsetAsync(output.data<double>(), 0,
                       static_cast<size_t>(output_size) * sizeof(double), stream));

        if (n > 0) {
            dim3 grid, block;
            compute_launch_config_1d(n, grid, block);
            Tensor w_cont = weights->is_contiguous() ? *weights : weights->contiguous();

            if (w_cont.dtype() == DType::Float64) {
                hipLaunchKernelGGL(bincount_weights_f64_hip, grid, block, 0, stream,
                    input_i64.data<int64_t>(), w_cont.data<double>(),
                    output.data<double>(), n);
            } else {
                Tensor w_f32 = (w_cont.dtype() == DType::Float32) ? w_cont : w_cont.to(DType::Float32);
                hipLaunchKernelGGL(bincount_weights_f32_hip, grid, block, 0, stream,
                    input_i64.data<int64_t>(), w_f32.data<float>(),
                    output.data<double>(), n);
            }
            HIP_CHECK(hipGetLastError());
        }
        return output;
    } else {
        Tensor output({output_size}, DType::Int64, device);
        hipMemsetAsync(output.data<int64_t>(), 0,
                       static_cast<size_t>(output_size) * sizeof(int64_t), stream);

        if (n > 0) {
            dim3 grid, block;
            compute_launch_config_1d(n, grid, block);
            hipLaunchKernelGGL(bincount_no_weights_hip, grid, block, 0, stream,
                input_i64.data<int64_t>(), output.data<int64_t>(), n);
            HIP_CHECK(hipGetLastError());
        }
        return output;
    }
}

// ============================================================================
// Rsqrt Kernels (reciprocal square root: 1/sqrt(x))
// ============================================================================

__global__ void rsqrt_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = rsqrtf(input[idx]);
    }
}

__global__ void rsqrt_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = 1.0 / sqrt(input[idx]);
    }
}

__global__ void rsqrt_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(rsqrtf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

// ============================================================================
// Square Kernels (element-wise square: x*x)
// ============================================================================

__global__ void square_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = input[idx];
        output[idx] = val * val;
    }
}

__global__ void square_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double val = input[idx];
        output[idx] = val * val;
    }
}

__global__ void square_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        output[idx] = tenzor::rocm::safe_f2h(val * val);
    }
}

// ============================================================================
// Asinh / Acosh / Atanh Kernels (inverse hyperbolic functions)
// ============================================================================

__global__ void asinh_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = asinhf(input[idx]);
    }
}

__global__ void asinh_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = asinh(input[idx]);
    }
}

__global__ void asinh_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(asinhf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void acosh_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = acoshf(input[idx]);
    }
}

__global__ void acosh_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = acosh(input[idx]);
    }
}

__global__ void acosh_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(acoshf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

__global__ void atanh_kernel_f32(const float* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = atanhf(input[idx]);
    }
}

__global__ void atanh_kernel_f64(const double* input, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = atanh(input[idx]);
    }
}

__global__ void atanh_kernel_f16(const __half* input, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(atanhf(tenzor::rocm::safe_h2f(input[idx])));
    }
}

// ============================================================================
// Hypot Kernels (overflow-safe: sqrt(x*x + y*y))
// ============================================================================

__global__ void hypot_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = hypotf(a[idx], b[idx]);
    }
}

__global__ void hypot_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = hypot(a[idx], b[idx]);
    }
}

__global__ void hypot_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = tenzor::rocm::safe_h2f(a[idx]);
        float vb = tenzor::rocm::safe_h2f(b[idx]);
        output[idx] = tenzor::rocm::safe_f2h(hypotf(va, vb));
    }
}

// ============================================================================
// Copysign Kernels
// ============================================================================

__global__ void copysign_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = copysignf(a[idx], b[idx]);
    }
}

__global__ void copysign_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = copysign(a[idx], b[idx]);
    }
}

__global__ void copysign_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = tenzor::rocm::safe_h2f(a[idx]);
        float vb = tenzor::rocm::safe_h2f(b[idx]);
        output[idx] = tenzor::rocm::safe_f2h(copysignf(va, vb));
    }
}

// ============================================================================
// Nextafter Kernels
// ============================================================================

__global__ void nextafter_kernel_f32(const float* a, const float* b, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = nextafterf(a[idx], b[idx]);
    }
}

__global__ void nextafter_kernel_f64(const double* a, const double* b, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = nextafter(a[idx], b[idx]);
    }
}

__global__ void nextafter_kernel_f16(const __half* a, const __half* b, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float va = tenzor::rocm::safe_h2f(a[idx]);
        float vb = tenzor::rocm::safe_h2f(b[idx]);
        output[idx] = tenzor::rocm::safe_f2h(nextafterf(va, vb));
    }
}

// ============================================================================
// Gcd / Lcm Kernels (integer only)
// ============================================================================

__device__ inline int32_t device_gcd_i32(int32_t a, int32_t b) {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    while (b != 0) { int32_t t = b; b = a % b; a = t; }
    return a;
}

__device__ inline int64_t device_gcd_i64(int64_t a, int64_t b) {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    while (b != 0) { int64_t t = b; b = a % b; a = t; }
    return a;
}

__global__ void gcd_kernel_i32(const int32_t* a, const int32_t* b, int32_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = device_gcd_i32(a[idx], b[idx]);
    }
}

__global__ void gcd_kernel_i64(const int64_t* a, const int64_t* b, int64_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = device_gcd_i64(a[idx], b[idx]);
    }
}

__global__ void lcm_kernel_i32(const int32_t* a, const int32_t* b, int32_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        int32_t g = device_gcd_i32(a[idx], b[idx]);
        output[idx] = g == 0 ? 0 : (a[idx] / g) * b[idx];
    }
}

__global__ void lcm_kernel_i64(const int64_t* a, const int64_t* b, int64_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        int64_t g = device_gcd_i64(a[idx], b[idx]);
        output[idx] = g == 0 ? 0 : (a[idx] / g) * b[idx];
    }
}

// ============================================================================
// Igamma / Igammac Kernels (regularized incomplete gamma functions)
// ============================================================================

__device__ inline float device_igamma_series_f32(float a, float x) {
    // Regularized lower incomplete gamma via series expansion
    // P(a,x) = e^(-x) * x^a * sum_{n=0}^{inf} x^n / (a*(a+1)*...*(a+n))
    if (x <= 0.0f) return 0.0f;
    float term = 1.0f / a;
    float sum = term;
    for (int n = 1; n < 200; ++n) {
        term *= x / (a + static_cast<float>(n));
        sum += term;
        if (fabsf(term) < fabsf(sum) * 1e-7f) break;
    }
    return expf(-x + a * logf(x) - lgammaf(a)) * sum;
}

__device__ inline double device_igamma_series_f64(double a, double x) {
    if (x <= 0.0) return 0.0;
    double term = 1.0 / a;
    double sum = term;
    for (int n = 1; n < 500; ++n) {
        term *= x / (a + static_cast<double>(n));
        sum += term;
        if (fabs(term) < fabs(sum) * 1e-15) break;
    }
    return exp(-x + a * log(x) - lgamma(a)) * sum;
}

// Upper regularized incomplete gamma via continued fraction (modified Lentz),
// matching CPU igammac_cf. Well-conditioned for x >= a+1, where the ascending
// series for P(a,x) (and thus 1-P for Q) loses all significant digits.
__device__ inline float device_igammac_cf_f32(float a, float x) {
    const float tiny = 1e-30f;
    const float eps  = 1e-7f;
    const int max_iter = 200;
    float prefix = expf(-x + a * logf(x) - lgammaf(a));

    float f = x + 1.0f - a;
    if (fabsf(f) < tiny) f = tiny;
    float C = f;
    float D = 0.0f;
    for (int nn = 1; nn <= max_iter; ++nn) {
        float an_val = -static_cast<float>(nn) * (static_cast<float>(nn) - a);
        float bn_val = x + static_cast<float>(2 * nn + 1) - a;
        D = bn_val + an_val * D;
        if (fabsf(D) < tiny) D = tiny;
        C = bn_val + an_val / C;
        if (fabsf(C) < tiny) C = tiny;
        D = 1.0f / D;
        float delta = C * D;
        f *= delta;
        if (fabsf(delta - 1.0f) < eps) break;
    }
    return prefix / f;
}

__device__ inline double device_igammac_cf_f64(double a, double x) {
    const double tiny = 1e-300;
    const double eps  = 1e-15;
    const int max_iter = 200;
    double prefix = exp(-x + a * log(x) - lgamma(a));

    double f = x + 1.0 - a;
    if (fabs(f) < tiny) f = tiny;
    double C = f;
    double D = 0.0;
    for (int nn = 1; nn <= max_iter; ++nn) {
        double an_val = -static_cast<double>(nn) * (static_cast<double>(nn) - a);
        double bn_val = x + static_cast<double>(2 * nn + 1) - a;
        D = bn_val + an_val * D;
        if (fabs(D) < tiny) D = tiny;
        C = bn_val + an_val / C;
        if (fabs(C) < tiny) C = tiny;
        D = 1.0 / D;
        double delta = C * D;
        f *= delta;
        if (fabs(delta - 1.0) < eps) break;
    }
    return prefix / f;
}

// P(a,x): series for x < a+1, else 1 - CF (matches CPU igamma_kernel).
__device__ inline float device_igamma_f32(float a, float x) {
    if (x < a + 1.0f) return device_igamma_series_f32(a, x);
    return 1.0f - device_igammac_cf_f32(a, x);
}
__device__ inline double device_igamma_f64(double a, double x) {
    if (x < a + 1.0) return device_igamma_series_f64(a, x);
    return 1.0 - device_igammac_cf_f64(a, x);
}

// Q(a,x): 1 - series for x < a+1, else CF (matches CPU igammac_kernel).
__device__ inline float device_igammac_f32(float a, float x) {
    if (x < a + 1.0f) return 1.0f - device_igamma_series_f32(a, x);
    return device_igammac_cf_f32(a, x);
}
__device__ inline double device_igammac_f64(double a, double x) {
    if (x < a + 1.0) return 1.0 - device_igamma_series_f64(a, x);
    return device_igammac_cf_f64(a, x);
}

__global__ void igamma_kernel_f32(const float* a, const float* x, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = device_igamma_f32(a[idx], x[idx]);
    }
}

__global__ void igamma_kernel_f64(const double* a, const double* x, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = device_igamma_f64(a[idx], x[idx]);
    }
}

__global__ void igamma_kernel_f16(const __half* a, const __half* x, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(device_igamma_f32(tenzor::rocm::safe_h2f(a[idx]), tenzor::rocm::safe_h2f(x[idx])));
    }
}

__global__ void igammac_kernel_f32(const float* a, const float* x, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = device_igammac_f32(a[idx], x[idx]);
    }
}

__global__ void igammac_kernel_f64(const double* a, const double* x, double* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = device_igammac_f64(a[idx], x[idx]);
    }
}

__global__ void igammac_kernel_f16(const __half* a, const __half* x, __half* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_f2h(device_igammac_f32(tenzor::rocm::safe_h2f(a[idx]), tenzor::rocm::safe_h2f(x[idx])));
    }
}

// ============================================================================
// Addcmul / Addcdiv Kernels (ternary: input + value * tensor1 * tensor2,
//                             input + value * tensor1 / tensor2)
// ============================================================================

__global__ void addcmul_kernel_f32(const float* input, const float* t1, const float* t2, float* output, float value, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx] + value * t1[idx] * t2[idx];
    }
}

__global__ void addcmul_kernel_f64(const double* input, const double* t1, const double* t2, double* output, double value, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx] + value * t1[idx] * t2[idx];
    }
}

__global__ void addcmul_kernel_f16(const __half* input, const __half* t1, const __half* t2, __half* output, float value, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float vi = tenzor::rocm::safe_h2f(input[idx]);
        float v1 = tenzor::rocm::safe_h2f(t1[idx]);
        float v2 = tenzor::rocm::safe_h2f(t2[idx]);
        output[idx] = tenzor::rocm::safe_f2h(vi + value * v1 * v2);
    }
}

__global__ void addcdiv_kernel_f32(const float* input, const float* t1, const float* t2, float* output, float value, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx] + value * t1[idx] / t2[idx];
    }
}

__global__ void addcdiv_kernel_f64(const double* input, const double* t1, const double* t2, double* output, double value, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx] + value * t1[idx] / t2[idx];
    }
}

__global__ void addcdiv_kernel_f16(const __half* input, const __half* t1, const __half* t2, __half* output, float value, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float vi = tenzor::rocm::safe_h2f(input[idx]);
        float v1 = tenzor::rocm::safe_h2f(t1[idx]);
        float v2 = tenzor::rocm::safe_h2f(t2[idx]);
        output[idx] = tenzor::rocm::safe_f2h(vi + value * v1 / v2);
    }
}

// ============================================================================
// Host Wrappers: New Element-wise Math Operations
// ============================================================================

auto rsqrt_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return rsqrt_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(rsqrt_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(rsqrt_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(rsqrt_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("rsqrt operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto square_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return square_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(square_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(square_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(square_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("square operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto asinh_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return asinh_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(asinh_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(asinh_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(asinh_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("asinh operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto acosh_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return acosh_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(acosh_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(acosh_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(acosh_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("acosh operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto atanh_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (input.dtype() == DType::BFloat16) {
        return atanh_kernel(input.to(DType::Float32), stream).to(DType::BFloat16);
    }
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(atanh_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(atanh_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(atanh_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("atanh operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto hypot_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    // BFloat16 widen-narrow: compute in Float32, narrow back (CPU parity).
    if (a.dtype() == DType::BFloat16 && b.dtype() == DType::BFloat16) {
        return hypot_kernel(a.to(DType::Float32), b.to(DType::Float32), stream)
            .to(DType::BFloat16);
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("hypot: tensors must have the same dtype");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("hypot: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(hypot_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(hypot_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(hypot_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("hypot operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto copysign_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("copysign: tensors must have the same dtype");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("copysign: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(copysign_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(copysign_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(copysign_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("copysign operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto nextafter_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("nextafter: tensors must have the same dtype");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("nextafter: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nextafter_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nextafter_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(nextafter_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("nextafter operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto gcd_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("gcd: tensors must have the same dtype");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("gcd: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(gcd_kernel_i32, grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(gcd_kernel_i64, grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
    } else {
        throw std::runtime_error("gcd operation only supports Int32 and Int64 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto lcm_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("lcm: tensors must have the same dtype");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("lcm: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(lcm_kernel_i32, grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(lcm_kernel_i64, grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
    } else {
        throw std::runtime_error("lcm operation only supports Int32 and Int64 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto igamma_kernel(const Tensor& a, const Tensor& x, hipStream_t stream) -> Tensor {
    if (a.dtype() != x.dtype()) {
        throw std::runtime_error("igamma: tensors must have the same dtype");
    }
    if (a.numel() != x.numel()) {
        throw std::runtime_error("igamma: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(igamma_kernel_f32, grid, block, 0, stream,
            a.data<float>(), x.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(igamma_kernel_f64, grid, block, 0, stream,
            a.data<double>(), x.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(igamma_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("igamma operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto igammac_kernel(const Tensor& a, const Tensor& x, hipStream_t stream) -> Tensor {
    if (a.dtype() != x.dtype()) {
        throw std::runtime_error("igammac: tensors must have the same dtype");
    }
    if (a.numel() != x.numel()) {
        throw std::runtime_error("igammac: tensors must have the same number of elements");
    }

    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(igammac_kernel_f32, grid, block, 0, stream,
            a.data<float>(), x.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(igammac_kernel_f64, grid, block, 0, stream,
            a.data<double>(), x.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(igammac_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("igammac operation only supports Float32, Float64, and Float16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto addcmul_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, hipStream_t stream) -> Tensor {
    if (input.dtype() != tensor1.dtype() || input.dtype() != tensor2.dtype()) {
        throw std::runtime_error("addcmul: all tensors must have the same dtype");
    }
    if (input.numel() != tensor1.numel() || input.numel() != tensor2.numel()) {
        throw std::runtime_error("addcmul: all tensors must have the same number of elements");
    }

    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(addcmul_kernel_f32, grid, block, 0, stream,
            input.data<float>(), tensor1.data<float>(), tensor2.data<float>(),
            result.data<float>(), value, n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(addcmul_kernel_f64, grid, block, 0, stream,
            input.data<double>(), tensor1.data<double>(), tensor2.data<double>(),
            result.data<double>(), static_cast<double>(value), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(addcmul_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<const __half*>(tensor1.data<Float16>()),
            reinterpret_cast<const __half*>(tensor2.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), value, n);
    } else if (input.dtype() == DType::BFloat16) {
        auto r_f32 = addcmul_kernel(cast_kernel(input, DType::Float32, stream),
                                    cast_kernel(tensor1, DType::Float32, stream),
                                    cast_kernel(tensor2, DType::Float32, stream), value, stream);
        return cast_kernel(r_f32, DType::BFloat16, stream);
    } else {
        throw std::runtime_error("addcmul operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

auto addcdiv_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, hipStream_t stream) -> Tensor {
    if (input.dtype() != tensor1.dtype() || input.dtype() != tensor2.dtype()) {
        throw std::runtime_error("addcdiv: all tensors must have the same dtype");
    }
    if (input.numel() != tensor1.numel() || input.numel() != tensor2.numel()) {
        throw std::runtime_error("addcdiv: all tensors must have the same number of elements");
    }

    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    if (n == 0) return result;

    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(addcdiv_kernel_f32, grid, block, 0, stream,
            input.data<float>(), tensor1.data<float>(), tensor2.data<float>(),
            result.data<float>(), value, n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(addcdiv_kernel_f64, grid, block, 0, stream,
            input.data<double>(), tensor1.data<double>(), tensor2.data<double>(),
            result.data<double>(), static_cast<double>(value), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(addcdiv_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<const __half*>(tensor1.data<Float16>()),
            reinterpret_cast<const __half*>(tensor2.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), value, n);
    } else if (input.dtype() == DType::BFloat16) {
        auto r_f32 = addcdiv_kernel(cast_kernel(input, DType::Float32, stream),
                                    cast_kernel(tensor1, DType::Float32, stream),
                                    cast_kernel(tensor2, DType::Float32, stream), value, stream);
        return cast_kernel(r_f32, DType::BFloat16, stream);
    } else {
        throw std::runtime_error("addcdiv operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// CumMax kernel — cumulative maximum along a dimension (returns values + indices)
// ============================================================================

template<typename T>
__global__ void cummax_hip_kernel(
    const T* __restrict__ input, T* __restrict__ values, int64_t* __restrict__ indices,
    int64_t dim_size, int64_t inner_size, int64_t total_slices)
{
    HIP_KERNEL_LOOP(idx, total_slices) {
        int64_t outer = idx / inner_size;
        int64_t inner = idx % inner_size;

        T running_max = input[outer * dim_size * inner_size + inner];
        int64_t running_idx = 0;
        values[outer * dim_size * inner_size + inner] = running_max;
        indices[outer * dim_size * inner_size + inner] = 0;

        for (int64_t i = 1; i < dim_size; ++i) {
            int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
            T val = input[offset];
            if (val > running_max) {
                running_max = val;
                running_idx = i;
            }
            values[offset] = running_max;
            indices[offset] = running_idx;
        }
    }
}

auto cummax_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    Tensor values(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);
    Tensor indices_out(std::vector<int64_t>(shape.begin(), shape.end()), DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t total_slices = outer_size * inner_size;

    // empty tensor (zero-sized non-reduced dim): zero-grid launch fails on HIP
    if (total_slices == 0 || dim_size == 0) return {values, indices_out};

    dim3 grid_dim, block_dim;
    compute_launch_config_1d(total_slices, grid_dim, block_dim);

    switch (dtype) {
        case DType::Float32:
            hipLaunchKernelGGL(cummax_hip_kernel<float>, grid_dim, block_dim, 0, stream,
                input_cont.data<float>(), values.data<float>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(cummax_hip_kernel<double>, grid_dim, block_dim, 0, stream,
                input_cont.data<double>(), values.data<double>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(cummax_hip_kernel<int32_t>, grid_dim, block_dim, 0, stream,
                input_cont.data<int32_t>(), values.data<int32_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(cummax_hip_kernel<int64_t>, grid_dim, block_dim, 0, stream,
                input_cont.data<int64_t>(), values.data<int64_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        default:
            throw std::runtime_error("cummax ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    return {values, indices_out};
}

// ============================================================================
// CumMin kernel — cumulative minimum along a dimension (returns values + indices)
// ============================================================================

template<typename T>
__global__ void cummin_hip_kernel(
    const T* __restrict__ input, T* __restrict__ values, int64_t* __restrict__ indices,
    int64_t dim_size, int64_t inner_size, int64_t total_slices)
{
    HIP_KERNEL_LOOP(idx, total_slices) {
        int64_t outer = idx / inner_size;
        int64_t inner = idx % inner_size;

        T running_min = input[outer * dim_size * inner_size + inner];
        int64_t running_idx = 0;
        values[outer * dim_size * inner_size + inner] = running_min;
        indices[outer * dim_size * inner_size + inner] = 0;

        for (int64_t i = 1; i < dim_size; ++i) {
            int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
            T val = input[offset];
            if (val < running_min) {
                running_min = val;
                running_idx = i;
            }
            values[offset] = running_min;
            indices[offset] = running_idx;
        }
    }
}

auto cummin_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    Tensor values(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);
    Tensor indices_out(std::vector<int64_t>(shape.begin(), shape.end()), DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t total_slices = outer_size * inner_size;

    // empty tensor (zero-sized non-reduced dim): zero-grid launch fails on HIP
    if (total_slices == 0 || dim_size == 0) return {values, indices_out};

    dim3 grid_dim, block_dim;
    compute_launch_config_1d(total_slices, grid_dim, block_dim);

    switch (dtype) {
        case DType::Float32:
            hipLaunchKernelGGL(cummin_hip_kernel<float>, grid_dim, block_dim, 0, stream,
                input_cont.data<float>(), values.data<float>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(cummin_hip_kernel<double>, grid_dim, block_dim, 0, stream,
                input_cont.data<double>(), values.data<double>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(cummin_hip_kernel<int32_t>, grid_dim, block_dim, 0, stream,
                input_cont.data<int32_t>(), values.data<int32_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(cummin_hip_kernel<int64_t>, grid_dim, block_dim, 0, stream,
                input_cont.data<int64_t>(), values.data<int64_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        default:
            throw std::runtime_error("cummin ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    return {values, indices_out};
}

// ============================================================================
// Fmax kernel — NaN-aware element-wise maximum (IEEE 754-2008)
// ============================================================================

__global__ void fmax_hip_f32(const float* __restrict__ a, const float* __restrict__ b,
                              float* __restrict__ out, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = ::fmaxf(a[idx], b[idx]);
    }
}

__global__ void fmax_hip_f64(const double* __restrict__ a, const double* __restrict__ b,
                              double* __restrict__ out, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = ::fmax(a[idx], b[idx]);
    }
}

__global__ void fmax_hip_i32(const int32_t* __restrict__ a, const int32_t* __restrict__ b,
                              int32_t* __restrict__ out, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void fmax_hip_i64(const int64_t* __restrict__ a, const int64_t* __restrict__ b,
                              int64_t* __restrict__ out, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

auto fmax_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor
{
    Tensor a_cont = a.is_contiguous() ? a : a.contiguous();
    Tensor b_cont = b.is_contiguous() ? b : b.contiguous();
    int64_t n = a_cont.numel();
    Tensor output(std::vector<int64_t>(a_cont.shape().begin(), a_cont.shape().end()),
                  a_cont.dtype(), a_cont.device());

    // empty tensor: zero-grid launch would fail on HIP
    if (n == 0) return output;

    dim3 grid_dim, block_dim;
    compute_launch_config_1d(n, grid_dim, block_dim);

    switch (a_cont.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(fmax_hip_f32, grid_dim, block_dim, 0, stream,
                a_cont.data<float>(), b_cont.data<float>(), output.data<float>(), n);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(fmax_hip_f64, grid_dim, block_dim, 0, stream,
                a_cont.data<double>(), b_cont.data<double>(), output.data<double>(), n);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(fmax_hip_i32, grid_dim, block_dim, 0, stream,
                a_cont.data<int32_t>(), b_cont.data<int32_t>(), output.data<int32_t>(), n);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(fmax_hip_i64, grid_dim, block_dim, 0, stream,
                a_cont.data<int64_t>(), b_cont.data<int64_t>(), output.data<int64_t>(), n);
            break;
        case DType::Float16:
        case DType::BFloat16: {
            // No native F16/BF16 fmax kernel — widen to Float32, compute, narrow back.
            auto a_f32 = a_cont.to(DType::Float32);
            auto b_f32 = b_cont.to(DType::Float32);
            auto out_f32 = fmax_kernel(a_f32, b_f32, stream);
            return out_f32.to(a_cont.dtype());
        }
        default:
            throw std::runtime_error("fmax ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    return output;
}

// ============================================================================
// Fmin kernel — NaN-aware element-wise minimum (IEEE 754-2008)
// ============================================================================

__global__ void fmin_hip_f32(const float* __restrict__ a, const float* __restrict__ b,
                              float* __restrict__ out, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = ::fminf(a[idx], b[idx]);
    }
}

__global__ void fmin_hip_f64(const double* __restrict__ a, const double* __restrict__ b,
                              double* __restrict__ out, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = ::fmin(a[idx], b[idx]);
    }
}

__global__ void fmin_hip_i32(const int32_t* __restrict__ a, const int32_t* __restrict__ b,
                              int32_t* __restrict__ out, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void fmin_hip_i64(const int64_t* __restrict__ a, const int64_t* __restrict__ b,
                              int64_t* __restrict__ out, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

auto fmin_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor
{
    Tensor a_cont = a.is_contiguous() ? a : a.contiguous();
    Tensor b_cont = b.is_contiguous() ? b : b.contiguous();
    int64_t n = a_cont.numel();
    Tensor output(std::vector<int64_t>(a_cont.shape().begin(), a_cont.shape().end()),
                  a_cont.dtype(), a_cont.device());

    // empty tensor: zero-grid launch would fail on HIP
    if (n == 0) return output;

    dim3 grid_dim, block_dim;
    compute_launch_config_1d(n, grid_dim, block_dim);

    switch (a_cont.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(fmin_hip_f32, grid_dim, block_dim, 0, stream,
                a_cont.data<float>(), b_cont.data<float>(), output.data<float>(), n);
            break;
        case DType::Float64:
            hipLaunchKernelGGL(fmin_hip_f64, grid_dim, block_dim, 0, stream,
                a_cont.data<double>(), b_cont.data<double>(), output.data<double>(), n);
            break;
        case DType::Int32:
            hipLaunchKernelGGL(fmin_hip_i32, grid_dim, block_dim, 0, stream,
                a_cont.data<int32_t>(), b_cont.data<int32_t>(), output.data<int32_t>(), n);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(fmin_hip_i64, grid_dim, block_dim, 0, stream,
                a_cont.data<int64_t>(), b_cont.data<int64_t>(), output.data<int64_t>(), n);
            break;
        case DType::Float16:
        case DType::BFloat16: {
            auto a_f32 = a_cont.to(DType::Float32);
            auto b_f32 = b_cont.to(DType::Float32);
            auto out_f32 = fmin_kernel(a_f32, b_f32, stream);
            return out_f32.to(a_cont.dtype());
        }
        default:
            throw std::runtime_error("fmin ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    return output;
}

// ============================================================================
// Isin kernel — set membership test using sorted array + binary search
// ============================================================================

template<typename T>
__global__ void isin_hip_kernel(const T* __restrict__ elements, int64_t num_elements,
                                 const T* __restrict__ test_sorted, int64_t num_test,
                                 bool* __restrict__ output)
{
    HIP_KERNEL_LOOP(idx, num_elements) {
        T val = elements[idx];
        int64_t lo = 0, hi = num_test - 1;
        bool found = false;
        while (lo <= hi) {
            int64_t mid = lo + (hi - lo) / 2;
            T mid_val = test_sorted[mid];
            if (mid_val == val) { found = true; break; }
            else if (mid_val < val) lo = mid + 1;
            else hi = mid - 1;
        }
        output[idx] = found;
    }
}

auto isin_kernel(const Tensor& elements, const Tensor& test_elements, hipStream_t stream) -> Tensor
{
    Tensor elem_cont = elements.is_contiguous() ? elements : elements.contiguous();
    Tensor test_cont = test_elements.is_contiguous() ? test_elements : test_elements.contiguous();

    Tensor test_sorted(std::vector<int64_t>(test_cont.shape().begin(), test_cont.shape().end()),
                       test_cont.dtype(), test_cont.device());
    HIP_CHECK(hipMemcpyAsync(test_sorted.data_ptr(), test_cont.data_ptr(),
                    test_cont.numel() * test_cont.element_size(),
                    hipMemcpyDeviceToDevice, stream));

    int64_t num_test = test_sorted.numel();
    int64_t num_elements = elem_cont.numel();

    Tensor output(std::vector<int64_t>(elem_cont.shape().begin(), elem_cont.shape().end()),
                  DType::Bool, elem_cont.device());

    dim3 grid_dim, block_dim;
    compute_launch_config_1d(num_elements, grid_dim, block_dim);

    // Sort test elements using hipcub
    switch (elem_cont.dtype()) {
        case DType::Float32: {
            void* d_temp = nullptr;
            size_t temp_bytes = 0;
            hipcub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                test_sorted.data<float>(), test_sorted.data<float>(),
                static_cast<int>(num_test), 0, sizeof(float) * 8, stream);
            HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
            Tensor temp_buf({static_cast<int64_t>(num_test)}, DType::Float32, test_sorted.device());
            hipcub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                test_sorted.data<float>(), temp_buf.data<float>(),
                static_cast<int>(num_test), 0, sizeof(float) * 8, stream);
            HIP_CHECK(hipFree(d_temp));
            hipLaunchKernelGGL(isin_hip_kernel<float>, grid_dim, block_dim, 0, stream,
                elem_cont.data<float>(), num_elements, temp_buf.data<float>(), num_test,
                reinterpret_cast<bool*>(output.data_ptr()));
            break;
        }
        case DType::Float64: {
            void* d_temp = nullptr;
            size_t temp_bytes = 0;
            hipcub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                test_sorted.data<double>(), test_sorted.data<double>(),
                static_cast<int>(num_test), 0, sizeof(double) * 8, stream);
            HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
            Tensor temp_buf({static_cast<int64_t>(num_test)}, DType::Float64, test_sorted.device());
            hipcub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                test_sorted.data<double>(), temp_buf.data<double>(),
                static_cast<int>(num_test), 0, sizeof(double) * 8, stream);
            HIP_CHECK(hipFree(d_temp));
            hipLaunchKernelGGL(isin_hip_kernel<double>, grid_dim, block_dim, 0, stream,
                elem_cont.data<double>(), num_elements, temp_buf.data<double>(), num_test,
                reinterpret_cast<bool*>(output.data_ptr()));
            break;
        }
        case DType::Int32: {
            void* d_temp = nullptr;
            size_t temp_bytes = 0;
            hipcub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                test_sorted.data<int32_t>(), test_sorted.data<int32_t>(),
                static_cast<int>(num_test), 0, sizeof(int32_t) * 8, stream);
            HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
            Tensor temp_buf({static_cast<int64_t>(num_test)}, DType::Int32, test_sorted.device());
            hipcub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                test_sorted.data<int32_t>(), temp_buf.data<int32_t>(),
                static_cast<int>(num_test), 0, sizeof(int32_t) * 8, stream);
            HIP_CHECK(hipFree(d_temp));
            hipLaunchKernelGGL(isin_hip_kernel<int32_t>, grid_dim, block_dim, 0, stream,
                elem_cont.data<int32_t>(), num_elements, temp_buf.data<int32_t>(), num_test,
                reinterpret_cast<bool*>(output.data_ptr()));
            break;
        }
        case DType::Int64: {
            void* d_temp = nullptr;
            size_t temp_bytes = 0;
            hipcub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                test_sorted.data<int64_t>(), test_sorted.data<int64_t>(),
                static_cast<int>(num_test), 0, sizeof(int64_t) * 8, stream);
            HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
            Tensor temp_buf({static_cast<int64_t>(num_test)}, DType::Int64, test_sorted.device());
            hipcub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                test_sorted.data<int64_t>(), temp_buf.data<int64_t>(),
                static_cast<int>(num_test), 0, sizeof(int64_t) * 8, stream);
            HIP_CHECK(hipFree(d_temp));
            hipLaunchKernelGGL(isin_hip_kernel<int64_t>, grid_dim, block_dim, 0, stream,
                elem_cont.data<int64_t>(), num_elements, temp_buf.data<int64_t>(), num_test,
                reinterpret_cast<bool*>(output.data_ptr()));
            break;
        }
        default:
            throw std::runtime_error("isin ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    return output;
}

// ============================================================================
// Kthvalue kernel — k-th smallest value along a dimension
// ============================================================================

template<typename T>
__global__ void kthvalue_hip_kernel(
    const T* __restrict__ input, T* __restrict__ values, int64_t* __restrict__ indices,
    int64_t dim_size, int64_t inner_size, int64_t k, int64_t total_slices,
    T* __restrict__ workspace)
{
    HIP_KERNEL_LOOP(slice_idx, total_slices) {
        int64_t outer = slice_idx / inner_size;
        int64_t inner = slice_idx % inner_size;

        T* ws = workspace + slice_idx * dim_size;
        for (int64_t i = 0; i < dim_size; ++i)
            ws[i] = input[outer * dim_size * inner_size + i * inner_size + inner];

        // Partial selection sort to find k-th smallest
        for (int64_t i = 0; i < k; ++i) {
            int64_t min_idx = i;
            T min_val = ws[i];
            for (int64_t j = i + 1; j < dim_size; ++j) {
                if (ws[j] < min_val) {
                    min_val = ws[j];
                    min_idx = j;
                }
            }
            if (min_idx != i) {
                ws[min_idx] = ws[i];
                ws[i] = min_val;
            }
        }

        T kth_val = ws[k - 1];
        values[slice_idx] = kth_val;

        for (int64_t i = 0; i < dim_size; ++i) {
            T orig = input[outer * dim_size * inner_size + i * inner_size + inner];
            if (orig == kth_val) {
                indices[slice_idx] = i;
                break;
            }
        }
    }
}

auto kthvalue_kernel(const Tensor& input, int64_t k, int64_t dim, bool keepdim,
                     hipStream_t stream) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    if (dim < 0) dim += ndim;  // Normalize (matches CPU + CUDA convention)
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t total_slices = outer_size * inner_size;

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor values(out_shape, dtype, device);
    Tensor indices_out(out_shape, DType::Int64, device);

    // empty tensor (zero-sized non-reduced dim): zero-grid launch fails on HIP
    if (total_slices == 0) return {values, indices_out};

    dim3 grid_dim, block_dim;
    compute_launch_config_1d(total_slices, grid_dim, block_dim);

    switch (dtype) {
        case DType::Float32: {
            Tensor ws({total_slices * dim_size}, DType::Float32, device);
            hipLaunchKernelGGL(kthvalue_hip_kernel<float>, grid_dim, block_dim, 0, stream,
                input_cont.data<float>(), values.data<float>(), indices_out.data<int64_t>(),
                dim_size, inner_size, k, total_slices, ws.data<float>());
            break;
        }
        case DType::Float64: {
            Tensor ws({total_slices * dim_size}, DType::Float64, device);
            hipLaunchKernelGGL(kthvalue_hip_kernel<double>, grid_dim, block_dim, 0, stream,
                input_cont.data<double>(), values.data<double>(), indices_out.data<int64_t>(),
                dim_size, inner_size, k, total_slices, ws.data<double>());
            break;
        }
        case DType::Int32: {
            Tensor ws({total_slices * dim_size}, DType::Int32, device);
            hipLaunchKernelGGL(kthvalue_hip_kernel<int32_t>, grid_dim, block_dim, 0, stream,
                input_cont.data<int32_t>(), values.data<int32_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, k, total_slices, ws.data<int32_t>());
            break;
        }
        case DType::Int64: {
            Tensor ws({total_slices * dim_size}, DType::Int64, device);
            hipLaunchKernelGGL(kthvalue_hip_kernel<int64_t>, grid_dim, block_dim, 0, stream,
                input_cont.data<int64_t>(), values.data<int64_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, k, total_slices, ws.data<int64_t>());
            break;
        }
        default:
            throw std::runtime_error("kthvalue ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    return {values, indices_out};
}

// ============================================================================
// Quantile/Nanquantile/Nanmedian — NaN-aware quantile kernels
// ============================================================================

template<typename T>
__global__ void nanquantile_hip_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    double q, int64_t dim_size, int64_t inner_size, int64_t total_slices,
    T* __restrict__ workspace)
{
    HIP_KERNEL_LOOP(idx, total_slices) {
        int64_t outer = idx / inner_size;
        int64_t inner = idx % inner_size;

        T* ws = workspace + idx * dim_size;
        int64_t count = 0;
        for (int64_t i = 0; i < dim_size; ++i) {
            T val = input[outer * dim_size * inner_size + i * inner_size + inner];
            // F7: use IEEE-754 bit pattern via `is_nan_bits(val)` — HIP's
            // `isnan` is unreliable under fast-math and the static_cast to
            // float drops precision for f64 inputs.
            if (!tenzor::rocm::is_nan_bits(val)) {
                ws[count++] = val;
            }
        }

        if (count == 0) {
            output[idx] = static_cast<T>(NAN);
            return;
        }

        // Insertion sort
        for (int64_t i = 1; i < count; ++i) {
            T key = ws[i];
            int64_t j = i - 1;
            while (j >= 0 && ws[j] > key) {
                ws[j + 1] = ws[j];
                --j;
            }
            ws[j + 1] = key;
        }

        double pos = q * (count - 1);
        int64_t lo = static_cast<int64_t>(pos);
        int64_t hi = lo + 1;
        if (hi >= count) hi = count - 1;
        double frac = pos - lo;
        output[idx] = static_cast<T>(static_cast<double>(ws[lo]) * (1.0 - frac) +
                                      static_cast<double>(ws[hi]) * frac);
    }
}

template<typename T>
__global__ void quantile_hip_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    double q, int64_t dim_size, int64_t inner_size, int64_t total_slices,
    T* __restrict__ workspace)
{
    HIP_KERNEL_LOOP(idx, total_slices) {
        int64_t outer = idx / inner_size;
        int64_t inner = idx % inner_size;

        T* ws = workspace + idx * dim_size;
        for (int64_t i = 0; i < dim_size; ++i)
            ws[i] = input[outer * dim_size * inner_size + i * inner_size + inner];

        // Insertion sort
        for (int64_t i = 1; i < dim_size; ++i) {
            T key = ws[i];
            int64_t j = i - 1;
            while (j >= 0 && ws[j] > key) {
                ws[j + 1] = ws[j];
                --j;
            }
            ws[j + 1] = key;
        }

        double pos = q * (dim_size - 1);
        int64_t lo = static_cast<int64_t>(pos);
        int64_t hi = lo + 1;
        if (hi >= dim_size) hi = dim_size - 1;
        double frac = pos - lo;
        output[idx] = static_cast<T>(static_cast<double>(ws[lo]) * (1.0 - frac) +
                                      static_cast<double>(ws[hi]) * frac);
    }
}

static auto quantile_nanquantile_impl(const Tensor& input, double q, int64_t dim,
                                       bool keepdim, bool ignore_nan,
                                       hipStream_t stream) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    if (dim < 0) dim += ndim;  // Normalize (matches CPU + CUDA convention)
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t total_slices = outer_size * inner_size;

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor output(out_shape, dtype, device);

    // empty tensor (zero-sized non-reduced dim): zero-grid launch fails on HIP
    if (total_slices == 0) return output;

    dim3 grid_dim, block_dim;
    compute_launch_config_1d(total_slices, grid_dim, block_dim);

    switch (dtype) {
        case DType::Float32: {
            Tensor ws({total_slices * dim_size}, DType::Float32, device);
            if (ignore_nan) {
                hipLaunchKernelGGL(nanquantile_hip_kernel<float>, grid_dim, block_dim, 0, stream,
                    input_cont.data<float>(), output.data<float>(), q, dim_size, inner_size,
                    total_slices, ws.data<float>());
            } else {
                hipLaunchKernelGGL(quantile_hip_kernel<float>, grid_dim, block_dim, 0, stream,
                    input_cont.data<float>(), output.data<float>(), q, dim_size, inner_size,
                    total_slices, ws.data<float>());
            }
            break;
        }
        case DType::Float64: {
            Tensor ws({total_slices * dim_size}, DType::Float64, device);
            if (ignore_nan) {
                hipLaunchKernelGGL(nanquantile_hip_kernel<double>, grid_dim, block_dim, 0, stream,
                    input_cont.data<double>(), output.data<double>(), q, dim_size, inner_size,
                    total_slices, ws.data<double>());
            } else {
                hipLaunchKernelGGL(quantile_hip_kernel<double>, grid_dim, block_dim, 0, stream,
                    input_cont.data<double>(), output.data<double>(), q, dim_size, inner_size,
                    total_slices, ws.data<double>());
            }
            break;
        }
        default:
            throw std::runtime_error("quantile ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    return output;
}

auto quantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim,
                     hipStream_t stream) -> Tensor
{
    return quantile_nanquantile_impl(input, q, dim, keepdim, false, stream);
}

auto nanquantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim,
                        hipStream_t stream) -> Tensor
{
    return quantile_nanquantile_impl(input, q, dim, keepdim, true, stream);
}

auto nanmedian_kernel(const Tensor& input, int64_t dim, bool keepdim,
                      hipStream_t stream) -> Tensor
{
    return nanquantile_kernel(input, 0.5, dim, keepdim, stream);
}

// ============================================================================
// Histc kernel — fixed-bin histogram using atomicAdd
// ============================================================================

__global__ void histc_hip_f32(const float* __restrict__ input, float* __restrict__ output,
                               int64_t n, int64_t bins, float min_val, float max_val)
{
    float bin_width = (max_val - min_val) / static_cast<float>(bins);
    HIP_KERNEL_LOOP(idx, n) {
        float val = input[idx];
        if (val >= min_val && val <= max_val) {
            int64_t bin = static_cast<int64_t>((val - min_val) / bin_width);
            if (bin >= bins) bin = bins - 1;
            atomicAdd(&output[bin], 1.0f);
        }
    }
}

auto histc_kernel(const Tensor& input, int64_t bins, double min_val, double max_val,
                  hipStream_t stream) -> Tensor
{
    // bins must be positive: bins<=0 yields a zero-width (div-by-zero) bin and
    // an out-of-bounds atomicAdd(&output[-1], ...). torch.histc requires bins>0.
    if (bins <= 0) {
        throw std::runtime_error("histc ROCm: bins must be positive");
    }

    // A degenerate range (min==max — e.g. all-equal input, or an unresolved
    // auto-range) yields a zero-width bin: bin_width=0 -> div-by-zero and an
    // out-of-bounds atomicAdd. Expand it like PyTorch's histc.
    if (min_val == max_val) {
        min_val -= 1.0;
        max_val += 1.0;
    }

    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    int64_t n = input_cont.numel();
    const auto device = input_cont.device();

    Tensor output({bins}, DType::Float32, device);
    HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0, bins * sizeof(float), stream));

    // Empty input: bins were already zeroed; a zero-grid launch is rejected by
    // HIP, so just return the all-zero histogram (matches torch.histc).
    if (n == 0) {
        HIP_CHECK(hipStreamSynchronize(stream));
        return output;
    }

    dim3 grid_dim, block_dim;
    compute_launch_config_1d(n, grid_dim, block_dim);

    if (input_cont.dtype() == DType::Float32) {
        hipLaunchKernelGGL(histc_hip_f32, grid_dim, block_dim, 0, stream,
            input_cont.data<float>(), output.data<float>(), n, bins,
            static_cast<float>(min_val), static_cast<float>(max_val));
    } else if (input_cont.dtype() == DType::Float64) {
        // Convert to float32 for atomicAdd compatibility
        Tensor f32_input = input_cont.to(DType::Float32);
        hipLaunchKernelGGL(histc_hip_f32, grid_dim, block_dim, 0, stream,
            f32_input.data<float>(), output.data<float>(), n, bins,
            static_cast<float>(min_val), static_cast<float>(max_val));
    } else {
        throw std::runtime_error("histc ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    return output;
}

// ============================================================================
// UniqueConsecutive kernel — deduplicate consecutive equal elements
// ============================================================================

template<typename T>
__global__ void unique_consecutive_mask_hip(const T* __restrict__ input,
                                             int32_t* __restrict__ mask, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        mask[idx] = (idx == 0 || input[idx] != input[idx - 1]) ? 1 : 0;
    }
}

__global__ void unique_counts_kernel_hip(const int64_t* __restrict__ inverse,
                                          int64_t* __restrict__ counts,
                                          int64_t n) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&counts[inverse[tid]]), 1ULL);
    }
}

template<typename T>
__global__ void unique_consecutive_gather_hip(const T* __restrict__ input,
                                               const int32_t* __restrict__ prefix_sum,
                                               const int32_t* __restrict__ mask,
                                               T* __restrict__ output,
                                               int64_t* __restrict__ inverse, int64_t n)
{
    HIP_KERNEL_LOOP(idx, n) {
        int32_t out_idx = prefix_sum[idx] - 1;
        if (mask[idx]) output[out_idx] = input[idx];
        inverse[idx] = out_idx;
    }
}

auto unique_consecutive_kernel(const Tensor& input, bool return_inverse,
                                hipStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    int64_t n = input_cont.numel();
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    if (n == 0) {
        return {Tensor({0}, dtype, device), Tensor({0}, DType::Int64, device),
                Tensor({0}, DType::Int64, device)};
    }

    Tensor mask({n}, DType::Int32, device);
    Tensor prefix({n}, DType::Int32, device);
    Tensor inverse_out({n}, DType::Int64, device);

    dim3 grid_dim, block_dim;
    compute_launch_config_1d(n, grid_dim, block_dim);

    auto launch = [&]<typename T>() {
        hipLaunchKernelGGL(unique_consecutive_mask_hip<T>, grid_dim, block_dim, 0, stream,
            input_cont.data<T>(), mask.data<int32_t>(), n);
        HIP_CHECK(hipGetLastError());

        // Inclusive prefix sum
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        HIP_CHECK(hipcub::DeviceScan::InclusiveSum(d_temp, temp_bytes,
            mask.data<int32_t>(), prefix.data<int32_t>(), static_cast<int>(n), stream));
        HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
        HIP_CHECK(hipcub::DeviceScan::InclusiveSum(d_temp, temp_bytes,
            mask.data<int32_t>(), prefix.data<int32_t>(), static_cast<int>(n), stream));
        HIP_CHECK(hipFree(d_temp));

        int32_t num_unique_h;
        HIP_CHECK(hipMemcpyAsync(&num_unique_h, prefix.data<int32_t>() + n - 1,
                        sizeof(int32_t), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        int64_t num_unique = num_unique_h;

        Tensor unique_out({num_unique}, dtype, device);
        hipLaunchKernelGGL(unique_consecutive_gather_hip<T>, grid_dim, block_dim, 0, stream,
            input_cont.data<T>(), prefix.data<int32_t>(), mask.data<int32_t>(),
            unique_out.data<T>(), inverse_out.data<int64_t>(), n);
        HIP_CHECK(hipGetLastError());

        // Compute counts from inverse indices via device-side atomicAdd
        Tensor counts({num_unique}, DType::Int64, device);
        HIP_CHECK(hipMemsetAsync(counts.data<int64_t>(), 0, num_unique * sizeof(int64_t), stream));
        {
            int count_threads = 256;
            int count_blocks = static_cast<int>((n + count_threads - 1) / count_threads);
            hipLaunchKernelGGL(unique_counts_kernel_hip,
                dim3(count_blocks), dim3(count_threads), 0, stream,
                inverse_out.data<int64_t>(), counts.data<int64_t>(), n);
            HIP_CHECK(hipGetLastError());
        }

        return std::make_tuple(unique_out, inverse_out, counts);
    };

    switch (dtype) {
        case DType::Float32: return launch.template operator()<float>();
        case DType::Float64: return launch.template operator()<double>();
        case DType::Int32:   return launch.template operator()<int32_t>();
        case DType::Int64:   return launch.template operator()<int64_t>();
        default: throw std::runtime_error("unique_consecutive ROCm: unsupported dtype");
    }
}

// ============================================================================
// Deg2Rad / Rad2Deg — simple unary conversions
// ============================================================================

__global__ void deg2rad_kernel_f32(const float* in, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] * (3.14159265358979323846f / 180.0f); }
}
__global__ void deg2rad_kernel_f64(const double* in, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] * (3.14159265358979323846 / 180.0); }
}
__global__ void deg2rad_kernel_f16(const __half* in, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(in[idx]);
        out[idx] = tenzor::rocm::safe_f2h(x * (3.14159265358979323846f / 180.0f));
    }
}
__global__ void deg2rad_kernel_bf16(const hip_bfloat16* in, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = static_cast<float>(in[idx]);
        out[idx] = tenzor::rocm::f32_to_bf16_rne(x * (3.14159265358979323846f / 180.0f));
    }
}

auto deg2rad_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(deg2rad_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(deg2rad_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(deg2rad_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(deg2rad_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("deg2rad only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

__global__ void rad2deg_kernel_f32(const float* in, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] * (180.0f / 3.14159265358979323846f); }
}
__global__ void rad2deg_kernel_f64(const double* in, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = in[idx] * (180.0 / 3.14159265358979323846); }
}
__global__ void rad2deg_kernel_f16(const __half* in, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(in[idx]);
        out[idx] = tenzor::rocm::safe_f2h(x * (180.0f / 3.14159265358979323846f));
    }
}
__global__ void rad2deg_kernel_bf16(const hip_bfloat16* in, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = static_cast<float>(in[idx]);
        out[idx] = tenzor::rocm::f32_to_bf16_rne(x * (180.0f / 3.14159265358979323846f));
    }
}

auto rad2deg_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(rad2deg_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(rad2deg_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(rad2deg_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(rad2deg_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("rad2deg only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Logit — log(x / (1-x)) with optional clamping
// ============================================================================

__device__ inline float logit_dev_f32(float x, float eps) {
    if (eps > 0.0f) {
        x = fminf(fmaxf(x, eps), 1.0f - eps);
    }
    return logf(x / (1.0f - x));
}

__device__ inline double logit_dev_f64(double x, double eps) {
    if (eps > 0.0) {
        x = fmin(fmax(x, eps), 1.0 - eps);
    }
    return log(x / (1.0 - x));
}

__global__ void logit_kernel_f32(const float* in, float* out, int64_t n, float eps) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = logit_dev_f32(in[idx], eps); }
}
__global__ void logit_kernel_f64(const double* in, double* out, int64_t n, double eps) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = logit_dev_f64(in[idx], eps); }
}
__global__ void logit_kernel_f16(const __half* in, __half* out, int64_t n, float eps) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(in[idx]);
        out[idx] = tenzor::rocm::safe_f2h(logit_dev_f32(x, eps));
    }
}
__global__ void logit_kernel_bf16(const hip_bfloat16* in, hip_bfloat16* out, int64_t n, float eps) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = static_cast<float>(in[idx]);
        out[idx] = tenzor::rocm::f32_to_bf16_rne(logit_dev_f32(x, eps));
    }
}

auto logit_kernel(const Tensor& input, double eps, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(logit_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n, static_cast<float>(eps));
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(logit_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n, eps);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(logit_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n, static_cast<float>(eps));
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(logit_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n, static_cast<float>(eps));
    } else {
        throw std::runtime_error("logit only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Signbit — returns Bool tensor, true where sign bit is set
// ============================================================================

__global__ void signbit_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint32_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>((bits >> 31) & 1u);
    }
}
__global__ void signbit_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint64_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>((bits >> 63) & 1u);
    }
}
__global__ void signbit_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        uint32_t bits; memcpy(&bits, &val, sizeof(bits));
        output[idx] = static_cast<uint8_t>((bits >> 31) & 1u);
    }
}

auto signbit_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(signbit_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(signbit_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(signbit_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            result.data<uint8_t>(), n);
    } else {
        // Integer types: check if value < 0
        HIP_CHECK(hipMemsetAsync(result.data<uint8_t>(), 0, n * sizeof(uint8_t), stream));
        return result;
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// IsPosInf / IsNegInf — returns Bool tensor
// ============================================================================

__global__ void isposinf_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint32_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>(bits == 0x7F800000u ? 1 : 0);
    }
}
__global__ void isposinf_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint64_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>(bits == 0x7FF0000000000000ull ? 1 : 0);
    }
}
__global__ void isposinf_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        uint32_t bits; memcpy(&bits, &val, sizeof(bits));
        output[idx] = static_cast<uint8_t>(bits == 0x7F800000u ? 1 : 0);
    }
}

auto isposinf_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(isposinf_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(isposinf_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(isposinf_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            result.data<uint8_t>(), n);
    } else {
        // Integer types cannot have Inf
        HIP_CHECK(hipMemsetAsync(result.data<uint8_t>(), 0, n * sizeof(uint8_t), stream));
        return result;
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

__global__ void isneginf_kernel_f32(const float* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint32_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>(bits == 0xFF800000u ? 1 : 0);
    }
}
__global__ void isneginf_kernel_f64(const double* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        uint64_t bits; memcpy(&bits, &input[idx], sizeof(bits));
        output[idx] = static_cast<uint8_t>(bits == 0xFFF0000000000000ull ? 1 : 0);
    }
}
__global__ void isneginf_kernel_f16(const __half* input, uint8_t* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float val = tenzor::rocm::safe_h2f(input[idx]);
        uint32_t bits; memcpy(&bits, &val, sizeof(bits));
        output[idx] = static_cast<uint8_t>(bits == 0xFF800000u ? 1 : 0);
    }
}

auto isneginf_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(isneginf_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(isneginf_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<uint8_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(isneginf_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            result.data<uint8_t>(), n);
    } else {
        // Integer types cannot have Inf
        HIP_CHECK(hipMemsetAsync(result.data<uint8_t>(), 0, n * sizeof(uint8_t), stream));
        return result;
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// FloatPower — pow with promotion to Float64
// ============================================================================

__global__ void float_power_kernel_f64(const double* a, const double* b, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = pow(a[idx], b[idx]); }
}

auto float_power_kernel(const Tensor& base, const Tensor& exp, hipStream_t stream) -> Tensor {
    if (base.numel() != exp.numel()) {
        throw std::runtime_error("float_power: tensors must have the same number of elements");
    }
    int64_t n = base.numel();
    std::vector<int64_t> shape(base.shape().begin(), base.shape().end());

    // Promote both inputs to Float64
    Tensor base_f64 = (base.dtype() != DType::Float64) ? base.to(DType::Float64) : base;
    Tensor exp_f64 = (exp.dtype() != DType::Float64) ? exp.to(DType::Float64) : exp;

    Tensor result(shape, DType::Float64, base.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    hipLaunchKernelGGL(float_power_kernel_f64, grid, block, 0, stream,
        base_f64.data<double>(), exp_f64.data<double>(), result.data<double>(), n);
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Xlog1py — x * log1p(y), with 0*log1p(y) = 0
// ============================================================================

__global__ void xlog1py_kernel_f32(const float* x, const float* y, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float xv = x[idx];
        out[idx] = (xv == 0.0f) ? 0.0f : xv * log1pf(y[idx]);
    }
}
__global__ void xlog1py_kernel_f64(const double* x, const double* y, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double xv = x[idx];
        out[idx] = (xv == 0.0) ? 0.0 : xv * log1p(y[idx]);
    }
}
__global__ void xlog1py_kernel_f16(const __half* x, const __half* y, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float xv = tenzor::rocm::safe_h2f(x[idx]);
        float yv = tenzor::rocm::safe_h2f(y[idx]);
        out[idx] = tenzor::rocm::safe_f2h((xv == 0.0f) ? 0.0f : xv * log1pf(yv));
    }
}

auto xlog1py_kernel(const Tensor& x, const Tensor& y, hipStream_t stream) -> Tensor {
    if (x.dtype() != y.dtype()) {
        throw std::runtime_error("xlog1py: tensors must have the same dtype");
    }
    if (x.numel() != y.numel()) {
        throw std::runtime_error("xlog1py: tensors must have the same number of elements");
    }
    int64_t n = x.numel();
    std::vector<int64_t> shape(x.shape().begin(), x.shape().end());
    Tensor result(shape, x.dtype(), x.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (x.dtype() == DType::Float32) {
        hipLaunchKernelGGL(xlog1py_kernel_f32, grid, block, 0, stream,
            x.data<float>(), y.data<float>(), result.data<float>(), n);
    } else if (x.dtype() == DType::Float64) {
        hipLaunchKernelGGL(xlog1py_kernel_f64, grid, block, 0, stream,
            x.data<double>(), y.data<double>(), result.data<double>(), n);
    } else if (x.dtype() == DType::Float16) {
        hipLaunchKernelGGL(xlog1py_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<const __half*>(y.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("xlog1py only supports Float32, Float64, Float16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Ldexp — x * 2^n
// ============================================================================

__global__ void ldexp_kernel_f32(const float* x, const float* n_in, float* out, int64_t count) {
    HIP_KERNEL_LOOP(idx, count) {
        out[idx] = ldexpf(x[idx], static_cast<int>(n_in[idx]));
    }
}
__global__ void ldexp_kernel_f64(const double* x, const double* n_in, double* out, int64_t count) {
    HIP_KERNEL_LOOP(idx, count) {
        out[idx] = ldexp(x[idx], static_cast<int>(n_in[idx]));
    }
}
__global__ void ldexp_kernel_f16(const __half* x, const __half* n_in, __half* out, int64_t count) {
    HIP_KERNEL_LOOP(idx, count) {
        float xv = tenzor::rocm::safe_h2f(x[idx]);
        int nv = static_cast<int>(tenzor::rocm::safe_h2f(n_in[idx]));
        out[idx] = tenzor::rocm::safe_f2h(ldexpf(xv, nv));
    }
}

auto ldexp_kernel(const Tensor& x, const Tensor& n_tensor, hipStream_t stream) -> Tensor {
    if (x.numel() != n_tensor.numel()) {
        throw std::runtime_error("ldexp: tensors must have the same number of elements");
    }
    int64_t n = x.numel();
    std::vector<int64_t> shape(x.shape().begin(), x.shape().end());
    Tensor result(shape, x.dtype(), x.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);

    // n_tensor should match x's dtype (or be cast)
    Tensor n_cast = (n_tensor.dtype() != x.dtype()) ? n_tensor.to(x.dtype()) : n_tensor;

    if (x.dtype() == DType::Float32) {
        hipLaunchKernelGGL(ldexp_kernel_f32, grid, block, 0, stream,
            x.data<float>(), n_cast.data<float>(), result.data<float>(), n);
    } else if (x.dtype() == DType::Float64) {
        hipLaunchKernelGGL(ldexp_kernel_f64, grid, block, 0, stream,
            x.data<double>(), n_cast.data<double>(), result.data<double>(), n);
    } else if (x.dtype() == DType::Float16) {
        hipLaunchKernelGGL(ldexp_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<const __half*>(n_cast.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else {
        throw std::runtime_error("ldexp only supports Float32, Float64, Float16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Frexp — decompose into mantissa and exponent (two outputs)
// ============================================================================

__global__ void frexp_kernel_f32(const float* input, float* mantissa, int32_t* exponent, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        int exp_val;
        mantissa[idx] = frexpf(input[idx], &exp_val);
        exponent[idx] = static_cast<int32_t>(exp_val);
    }
}
__global__ void frexp_kernel_f64(const double* input, double* mantissa, int32_t* exponent, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        int exp_val;
        mantissa[idx] = frexp(input[idx], &exp_val);
        exponent[idx] = static_cast<int32_t>(exp_val);
    }
}
__global__ void frexp_kernel_f16(const __half* input, __half* mantissa, int32_t* exponent, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        int exp_val;
        float m = frexpf(tenzor::rocm::safe_h2f(input[idx]), &exp_val);
        mantissa[idx] = tenzor::rocm::safe_f2h(m);
        exponent[idx] = static_cast<int32_t>(exp_val);
    }
}

auto frexp_kernel(const Tensor& input, hipStream_t stream) -> std::vector<Tensor> {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor mantissa(shape, input.dtype(), input.device());
    Tensor exponent(shape, DType::Int32, input.device());
    if (n == 0) return {mantissa, exponent};
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(frexp_kernel_f32, grid, block, 0, stream,
            input.data<float>(), mantissa.data<float>(), exponent.data<int32_t>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(frexp_kernel_f64, grid, block, 0, stream,
            input.data<double>(), mantissa.data<double>(), exponent.data<int32_t>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(frexp_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(mantissa.data<Float16>()),
            exponent.data<int32_t>(), n);
    } else {
        throw std::runtime_error("frexp only supports Float32, Float64, Float16");
    }
    HIP_CHECK(hipGetLastError());
    return {mantissa, exponent};
}

// ============================================================================
// IsReal — returns Bool tensor; true for all non-complex dtypes
// ============================================================================

auto isreal_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, DType::Bool, input.device());
    if (n == 0) return result;
    // All dtypes in Tenzor are real (no complex support yet) — fill with 1
    HIP_CHECK(hipMemsetAsync(result.data<uint8_t>(), 1, n * sizeof(uint8_t), stream));
    return result;
}

// ============================================================================
// SegmentReduce — reduce over segments defined by offsets (HIP)
// One warp per segment, warp-level shuffle reductions.
// ============================================================================

template<typename T>
__global__ void segment_reduce_hip_kernel(
    const T* __restrict__ data,
    const int64_t* __restrict__ offsets,
    T* __restrict__ output,
    int64_t num_segments,
    int64_t outer_size,
    int64_t axis_size,
    int64_t inner_size,
    int mode)
{
    // One thread per (outer, segment, inner) triple — sequential reduction
    // within each segment (simple and correct across all wavefront sizes)
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_work = outer_size * num_segments * inner_size;
    if (tid >= total_work) return;

    int64_t inner = tid % inner_size;
    int64_t seg = (tid / inner_size) % num_segments;
    int64_t outer = tid / (inner_size * num_segments);

    int64_t seg_start = offsets[seg];
    int64_t seg_end = offsets[seg + 1];
    int64_t seg_len = seg_end - seg_start;

    T identity;
    if (mode == 0 || mode == 1) identity = T(0);
    else if (mode == 4) identity = T(1);
    else if (mode == 2) identity = T(-1e38);
    else identity = T(1e38);

    T acc = identity;
    for (int64_t i = 0; i < seg_len; ++i) {
        int64_t d = seg_start + i;
        int64_t in_idx = (outer * axis_size + d) * inner_size + inner;
        T val = data[in_idx];
        if (mode == 0 || mode == 1) acc += val;
        else if (mode == 4) acc *= val;
        else if (mode == 2) acc = acc > val ? acc : val;
        else acc = acc < val ? acc : val;
    }

    if (mode == 1 && seg_len > 0) {
        acc /= static_cast<T>(seg_len);
    }
    if (seg_len == 0) {
        acc = (mode == 0 || mode == 1) ? T(0) : identity;
    }
    int64_t out_idx = (outer * num_segments + seg) * inner_size + inner;
    output[out_idx] = acc;
}

auto segment_reduce_kernel(const Tensor& data, const Tensor& offsets,
                           const std::string& reduce, int64_t axis,
                           hipStream_t stream) -> Tensor {
    Tensor cont = data.is_contiguous() ? data : data.contiguous();
    Tensor offs = offsets.is_contiguous() ? offsets : offsets.contiguous();

    int64_t ndim = cont.ndim();
    if (axis < 0) axis += ndim;

    const auto& shape = cont.shape();
    int64_t axis_size = shape[axis];
    int64_t num_segments = offs.numel() - 1;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < ndim; ++i) inner_size *= shape[i];

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        out_shape.push_back(i == axis ? num_segments : shape[i]);
    }

    int mode = 0;
    if (reduce == "sum") mode = 0;
    else if (reduce == "mean") mode = 1;
    else if (reduce == "max") mode = 2;
    else if (reduce == "min") mode = 3;
    else if (reduce == "prod") mode = 4;

    auto dtype = cont.dtype();
    auto device = cont.device();
    Tensor output(out_shape, dtype, device);

    int64_t total_work = outer_size * num_segments * inner_size;
    int block = 256;
    int grid = static_cast<int>((total_work + block - 1) / block);
    grid = std::max(grid, 1);

    const int64_t* offsets_ptr = offs.data<int64_t>();

    switch (dtype) {
        case DType::Float32:
            segment_reduce_hip_kernel<float><<<grid, block, 0, stream>>>(
                cont.data<float>(), offsets_ptr, output.data<float>(),
                num_segments, outer_size, axis_size, inner_size, mode);
            break;
        case DType::Float64:
            segment_reduce_hip_kernel<double><<<grid, block, 0, stream>>>(
                cont.data<double>(), offsets_ptr, output.data<double>(),
                num_segments, outer_size, axis_size, inner_size, mode);
            break;
        case DType::Int32:
            segment_reduce_hip_kernel<int32_t><<<grid, block, 0, stream>>>(
                cont.data<int32_t>(), offsets_ptr, output.data<int32_t>(),
                num_segments, outer_size, axis_size, inner_size, mode);
            break;
        case DType::Int64:
            segment_reduce_hip_kernel<int64_t><<<grid, block, 0, stream>>>(
                cont.data<int64_t>(), offsets_ptr, output.data<int64_t>(),
                num_segments, outer_size, axis_size, inner_size, mode);
            break;
        default: {
            DType orig = dtype;
            Tensor cont_f32 = cont.to(DType::Float32);
            Tensor output_f32(out_shape, DType::Float32, device);
            segment_reduce_hip_kernel<float><<<grid, block, 0, stream>>>(
                cont_f32.data<float>(), offsets_ptr, output_f32.data<float>(),
                num_segments, outer_size, axis_size, inner_size, mode);
            output = output_f32.to(orig);
        }
    }
    HIP_CHECK(hipGetLastError());
    return output;
}

// ============================================================================
// LogAddExp: max(a,b) + log1p(exp(-|a-b|)), numerically stable
// ============================================================================
__global__ void logaddexp_kernel_f32(const float* a, const float* b, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = a[idx], y = b[idx];
        float m = fmaxf(x, y);
        out[idx] = m + log1pf(expf(-fabsf(x - y)));
    }
}
__global__ void logaddexp_kernel_f64(const double* a, const double* b, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double x = a[idx], y = b[idx];
        double m = fmax(x, y);
        out[idx] = m + log1p(exp(-fabs(x - y)));
    }
}
__global__ void logaddexp_kernel_f16(const __half* a, const __half* b, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(a[idx]), y = tenzor::rocm::safe_h2f(b[idx]);
        float m = fmaxf(x, y);
        out[idx] = tenzor::rocm::safe_f2h(m + log1pf(expf(-fabsf(x - y))));
    }
}
__global__ void logaddexp_kernel_bf16(const hip_bfloat16* a, const hip_bfloat16* b, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = static_cast<float>(a[idx]), y = static_cast<float>(b[idx]);
        float m = fmaxf(x, y);
        out[idx] = tenzor::rocm::f32_to_bf16_rne(m + log1pf(expf(-fabsf(x - y))));
    }
}
auto logaddexp_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(logaddexp_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(logaddexp_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(logaddexp_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(logaddexp_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("logaddexp only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// LogAddExp2: max(a,b) + log2(1 + exp2(-|a-b|)), numerically stable
// ============================================================================
__global__ void logaddexp2_kernel_f32(const float* a, const float* b, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = a[idx], y = b[idx];
        float m = fmaxf(x, y);
        out[idx] = m + log2f(1.0f + exp2f(-fabsf(x - y)));
    }
}
__global__ void logaddexp2_kernel_f64(const double* a, const double* b, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double x = a[idx], y = b[idx];
        double m = fmax(x, y);
        out[idx] = m + log2(1.0 + exp2(-fabs(x - y)));
    }
}
__global__ void logaddexp2_kernel_f16(const __half* a, const __half* b, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(a[idx]), y = tenzor::rocm::safe_h2f(b[idx]);
        float m = fmaxf(x, y);
        out[idx] = tenzor::rocm::safe_f2h(m + log2f(1.0f + exp2f(-fabsf(x - y))));
    }
}
__global__ void logaddexp2_kernel_bf16(const hip_bfloat16* a, const hip_bfloat16* b, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = static_cast<float>(a[idx]), y = static_cast<float>(b[idx]);
        float m = fmaxf(x, y);
        out[idx] = tenzor::rocm::f32_to_bf16_rne(m + log2f(1.0f + exp2f(-fabsf(x - y))));
    }
}
auto logaddexp2_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    int64_t n = a.numel();
    std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    Tensor result(shape, a.dtype(), a.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (a.dtype() == DType::Float32) {
        hipLaunchKernelGGL(logaddexp2_kernel_f32, grid, block, 0, stream,
            a.data<float>(), b.data<float>(), result.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        hipLaunchKernelGGL(logaddexp2_kernel_f64, grid, block, 0, stream,
            a.data<double>(), b.data<double>(), result.data<double>(), n);
    } else if (a.dtype() == DType::Float16) {
        hipLaunchKernelGGL(logaddexp2_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (a.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(logaddexp2_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("logaddexp2 only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// XLogY: x * log(y), with x==0 => 0
// ============================================================================
__global__ void xlogy_kernel_f32(const float* x, const float* y, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float xv = x[idx];
        out[idx] = (xv == 0.0f) ? 0.0f : xv * logf(y[idx]);
    }
}
__global__ void xlogy_kernel_f64(const double* x, const double* y, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double xv = x[idx];
        out[idx] = (xv == 0.0) ? 0.0 : xv * log(y[idx]);
    }
}
__global__ void xlogy_kernel_f16(const __half* x, const __half* y, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float xv = tenzor::rocm::safe_h2f(x[idx]);
        out[idx] = tenzor::rocm::safe_f2h((xv == 0.0f) ? 0.0f : xv * logf(tenzor::rocm::safe_h2f(y[idx])));
    }
}
__global__ void xlogy_kernel_bf16(const hip_bfloat16* x, const hip_bfloat16* y, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float xv = static_cast<float>(x[idx]);
        out[idx] = tenzor::rocm::f32_to_bf16_rne((xv == 0.0f) ? 0.0f : xv * logf(static_cast<float>(y[idx])));
    }
}
auto xlogy_kernel(const Tensor& x, const Tensor& y, hipStream_t stream) -> Tensor {
    int64_t n = x.numel();
    std::vector<int64_t> shape(x.shape().begin(), x.shape().end());
    Tensor result(shape, x.dtype(), x.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (x.dtype() == DType::Float32) {
        hipLaunchKernelGGL(xlogy_kernel_f32, grid, block, 0, stream,
            x.data<float>(), y.data<float>(), result.data<float>(), n);
    } else if (x.dtype() == DType::Float64) {
        hipLaunchKernelGGL(xlogy_kernel_f64, grid, block, 0, stream,
            x.data<double>(), y.data<double>(), result.data<double>(), n);
    } else if (x.dtype() == DType::Float16) {
        hipLaunchKernelGGL(xlogy_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<const __half*>(y.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (x.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(xlogy_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(x.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(y.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("xlogy only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// I0e: exp(-|x|) * BesselI0(x) — Chebyshev polynomial approximation
// ============================================================================
__device__ inline float i0e_dev_f32(float x) {
    float ax = fabsf(x);
    if (ax < 3.75f) {
        float t = x / 3.75f;
        t = t * t;
        float i0 = 1.0f + t * (3.5156229f + t * (3.0899424f + t * (1.2067492f
                    + t * (0.2659732f + t * (0.0360768f + t * 0.0045813f)))));
        return expf(-ax) * i0;
    }
    float t = 3.75f / ax;
    return (1.0f / sqrtf(ax)) * (0.39894228f + t * (0.01328592f
           + t * (0.00225319f - t * (0.00157565f - t * (0.00916281f
           - t * (0.02057706f - t * (0.02635537f - t * (0.01647633f
           - t * 0.00392377f))))))));
}
__device__ inline double i0e_dev_f64(double x) {
    double ax = fabs(x);
    if (ax < 3.75) {
        double t = x / 3.75;
        t = t * t;
        double i0 = 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
                     + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        return exp(-ax) * i0;
    }
    double t = 3.75 / ax;
    return (1.0 / sqrt(ax)) * (0.39894228 + t * (0.01328592
           + t * (0.00225319 - t * (0.00157565 - t * (0.00916281
           - t * (0.02057706 - t * (0.02635537 - t * (0.01647633
           - t * 0.00392377))))))));
}
__global__ void i0e_kernel_f32(const float* in, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = i0e_dev_f32(in[idx]); }
}
__global__ void i0e_kernel_f64(const double* in, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = i0e_dev_f64(in[idx]); }
}
__global__ void i0e_kernel_f16(const __half* in, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = tenzor::rocm::safe_f2h(i0e_dev_f32(tenzor::rocm::safe_h2f(in[idx]))); }
}
__global__ void i0e_kernel_bf16(const hip_bfloat16* in, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = tenzor::rocm::f32_to_bf16_rne(i0e_dev_f32(static_cast<float>(in[idx]))); }
}
auto i0e_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(i0e_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(i0e_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(i0e_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(i0e_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("i0e only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// I1e: exp(-|x|) * BesselI1(x) — Chebyshev polynomial approximation
// ============================================================================
__device__ inline float i1e_dev_f32(float x) {
    float ax = fabsf(x);
    float result;
    if (ax < 3.75f) {
        float t = x / 3.75f;
        t = t * t;
        result = ax * (0.5f + t * (0.87890594f + t * (0.51498869f + t * (0.15084934f
                 + t * (0.02658733f + t * (0.00301532f + t * 0.00032411f))))));
        result = expf(-ax) * result;
    } else {
        float t = 3.75f / ax;
        result = (1.0f / sqrtf(ax)) * (0.39894228f - t * (0.03988024f
                 - t * (0.00362018f + t * (0.00163801f - t * (0.01031555f
                 - t * (0.02282967f - t * (0.02895312f - t * (0.01787654f
                 - t * 0.00420059f))))))));
    }
    return (x < 0.0f) ? -result : result;
}
__device__ inline double i1e_dev_f64(double x) {
    double ax = fabs(x);
    double result;
    if (ax < 3.75) {
        double t = x / 3.75;
        t = t * t;
        result = ax * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934
                 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
        result = exp(-ax) * result;
    } else {
        double t = 3.75 / ax;
        result = (1.0 / sqrt(ax)) * (0.39894228 - t * (0.03988024
                 - t * (0.00362018 + t * (0.00163801 - t * (0.01031555
                 - t * (0.02282967 - t * (0.02895312 - t * (0.01787654
                 - t * 0.00420059))))))));
    }
    return (x < 0.0) ? -result : result;
}
__global__ void i1e_kernel_f32(const float* in, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = i1e_dev_f32(in[idx]); }
}
__global__ void i1e_kernel_f64(const double* in, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = i1e_dev_f64(in[idx]); }
}
__global__ void i1e_kernel_f16(const __half* in, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = tenzor::rocm::safe_f2h(i1e_dev_f32(tenzor::rocm::safe_h2f(in[idx]))); }
}
__global__ void i1e_kernel_bf16(const hip_bfloat16* in, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) { out[idx] = tenzor::rocm::f32_to_bf16_rne(i1e_dev_f32(static_cast<float>(in[idx]))); }
}
auto i1e_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(i1e_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(i1e_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(i1e_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(i1e_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("i1e only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Entr: -x*log(x), with 0->0, negative->-inf
// ============================================================================
__global__ void entr_kernel_f32(const float* in, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = in[idx];
        if (x > 0.0f) out[idx] = -x * logf(x);
        else if (x == 0.0f) out[idx] = 0.0f;
        else out[idx] = -HUGE_VALF;
    }
}
__global__ void entr_kernel_f64(const double* in, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double x = in[idx];
        if (x > 0.0) out[idx] = -x * log(x);
        else if (x == 0.0) out[idx] = 0.0;
        else out[idx] = -HUGE_VAL;
    }
}
__global__ void entr_kernel_f16(const __half* in, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(in[idx]);
        float r;
        if (x > 0.0f) r = -x * logf(x);
        else if (x == 0.0f) r = 0.0f;
        else r = -HUGE_VALF;
        out[idx] = tenzor::rocm::safe_f2h(r);
    }
}
__global__ void entr_kernel_bf16(const hip_bfloat16* in, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = static_cast<float>(in[idx]);
        float r;
        if (x > 0.0f) r = -x * logf(x);
        else if (x == 0.0f) r = 0.0f;
        else r = -HUGE_VALF;
        out[idx] = tenzor::rocm::f32_to_bf16_rne(r);
    }
}
auto entr_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(entr_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(entr_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(entr_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(entr_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("entr only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// SphericalBesselJ0: sin(x)/x, with j0(0) = 1
// ============================================================================
__global__ void spherical_bessel_j0_kernel_f32(const float* in, float* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = in[idx];
        out[idx] = (x == 0.0f) ? 1.0f : sinf(x) / x;
    }
}
__global__ void spherical_bessel_j0_kernel_f64(const double* in, double* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        double x = in[idx];
        out[idx] = (x == 0.0) ? 1.0 : sin(x) / x;
    }
}
__global__ void spherical_bessel_j0_kernel_f16(const __half* in, __half* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = tenzor::rocm::safe_h2f(in[idx]);
        out[idx] = tenzor::rocm::safe_f2h((x == 0.0f) ? 1.0f : sinf(x) / x);
    }
}
__global__ void spherical_bessel_j0_kernel_bf16(const hip_bfloat16* in, hip_bfloat16* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        float x = static_cast<float>(in[idx]);
        out[idx] = tenzor::rocm::f32_to_bf16_rne((x == 0.0f) ? 1.0f : sinf(x) / x);
    }
}
auto spherical_bessel_j0_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());
    if (n == 0) return result;
    dim3 grid, block; compute_launch_config_1d(n, grid, block);
    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(spherical_bessel_j0_kernel_f32, grid, block, 0, stream,
            input.data<float>(), result.data<float>(), n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(spherical_bessel_j0_kernel_f64, grid, block, 0, stream,
            input.data<double>(), result.data<double>(), n);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(spherical_bessel_j0_kernel_f16, grid, block, 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()), n);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(spherical_bessel_j0_kernel_bf16, grid, block, 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
    } else {
        throw std::runtime_error("spherical_bessel_j0 only supports Float32, Float64, Float16, BFloat16");
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// CosineSimilarity: sum(a*b, dim) / (norm(a, dim) * norm(b, dim) + eps)
// ============================================================================
template<typename T>
__global__ void cosine_similarity_hip_kernel(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
    int64_t outer_size, int64_t dim_size, int64_t inner_size, T eps)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    if (idx >= total) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;

    T dot = T(0), norm_a = T(0), norm_b = T(0);
    for (int64_t d = 0; d < dim_size; d++) {
        int64_t offset = (outer * dim_size + d) * inner_size + inner;
        T av = a[offset], bv = b[offset];
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    out[idx] = dot / (sqrt(norm_a) * sqrt(norm_b) + eps);
}

auto cosine_similarity_kernel(const Tensor& a, const Tensor& b,
                               int64_t dim, double eps, hipStream_t stream) -> Tensor {
    Tensor ca = a.is_contiguous() ? a : a.contiguous();
    Tensor cb = b.is_contiguous() ? b : b.contiguous();
    auto shape = ca.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; i++) outer_size *= shape[i];
    for (int64_t i = dim + 1; i < ndim; i++) inner_size *= shape[i];

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) out_shape.push_back(shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, ca.dtype(), ca.device());
    int64_t total = outer_size * inner_size;
    if (total == 0) return result;

    int block_size = 256;
    int grid_size = static_cast<int>((total + block_size - 1) / block_size);

    if (ca.dtype() == DType::Float32) {
        cosine_similarity_hip_kernel<float><<<grid_size, block_size, 0, stream>>>(
            ca.data<float>(), cb.data<float>(), result.data<float>(),
            outer_size, dim_size, inner_size, static_cast<float>(eps));
    } else if (ca.dtype() == DType::Float64) {
        cosine_similarity_hip_kernel<double><<<grid_size, block_size, 0, stream>>>(
            ca.data<double>(), cb.data<double>(), result.data<double>(),
            outer_size, dim_size, inner_size, static_cast<double>(eps));
    } else {
        // Float16/BFloat16: upcast to Float32
        Tensor a32 = cast_kernel(ca, DType::Float32, stream);
        Tensor b32 = cast_kernel(cb, DType::Float32, stream);
        Tensor r32(out_shape, DType::Float32, ca.device());
        cosine_similarity_hip_kernel<float><<<grid_size, block_size, 0, stream>>>(
            a32.data<float>(), b32.data<float>(), r32.data<float>(),
            outer_size, dim_size, inner_size, static_cast<float>(eps));
        HIP_CHECK(hipGetLastError());
        return cast_kernel(r32, ca.dtype(), stream);
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Renorm: scale slices along dim so p-norm <= maxnorm
// ============================================================================
template<typename T>
__global__ void renorm_norm_kernel(
    const T* __restrict__ data, T* __restrict__ norms,
    int64_t dim_size, int64_t outer_size, int64_t inner_size,
    T p_val)
{
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim_size) return;

    T acc = T(0);
    for (int64_t o = 0; o < outer_size; o++) {
        for (int64_t i = 0; i < inner_size; i++) {
            int64_t idx = (o * dim_size + d) * inner_size + i;
            T val = data[idx];
            acc += pow(fabs(static_cast<double>(val)), static_cast<double>(p_val));
        }
    }
    norms[d] = static_cast<T>(pow(static_cast<double>(acc), 1.0 / static_cast<double>(p_val)));
}

template<typename T>
__global__ void renorm_scale_kernel(
    T* __restrict__ data, const T* __restrict__ norms,
    int64_t dim_size, int64_t outer_size, int64_t inner_size,
    T maxnorm_val)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * dim_size * inner_size;
    if (tid >= total) return;

    int64_t inner = tid % inner_size;
    int64_t d = (tid / inner_size) % dim_size;
    (void)inner; // used via d

    T norm_val = norms[d];
    if (norm_val > maxnorm_val) {
        data[tid] *= (maxnorm_val / norm_val);
    }
}

auto renorm_kernel(const Tensor& input, double p, int64_t dim,
                   double maxnorm, hipStream_t stream) -> Tensor {
    Tensor result = input.is_contiguous() ? input.clone() : input.contiguous().clone();
    auto shape = result.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; i++) outer_size *= shape[i];
    for (int64_t i = dim + 1; i < ndim; i++) inner_size *= shape[i];

    int block = 256;

    if (result.dtype() == DType::Float32) {
        Tensor norms({dim_size}, DType::Float32, result.device());
        int grid_norm = static_cast<int>((dim_size + block - 1) / block);
        renorm_norm_kernel<float><<<grid_norm, block, 0, stream>>>(
            result.data<float>(), norms.data<float>(),
            dim_size, outer_size, inner_size, static_cast<float>(p));
        HIP_CHECK(hipGetLastError());
        int64_t total = outer_size * dim_size * inner_size;
        int grid_scale = static_cast<int>((total + block - 1) / block);
        renorm_scale_kernel<float><<<grid_scale, block, 0, stream>>>(
            result.data<float>(), norms.data<float>(),
            dim_size, outer_size, inner_size, static_cast<float>(maxnorm));
    } else if (result.dtype() == DType::Float64) {
        Tensor norms({dim_size}, DType::Float64, result.device());
        int grid_norm = static_cast<int>((dim_size + block - 1) / block);
        renorm_norm_kernel<double><<<grid_norm, block, 0, stream>>>(
            result.data<double>(), norms.data<double>(),
            dim_size, outer_size, inner_size, static_cast<double>(p));
        HIP_CHECK(hipGetLastError());
        int64_t total = outer_size * dim_size * inner_size;
        int grid_scale = static_cast<int>((total + block - 1) / block);
        renorm_scale_kernel<double><<<grid_scale, block, 0, stream>>>(
            result.data<double>(), norms.data<double>(),
            dim_size, outer_size, inner_size, static_cast<double>(maxnorm));
    } else {
        // Float16/BFloat16: upcast to Float32, renorm, downcast
        DType orig = result.dtype();
        Tensor r32 = cast_kernel(result, DType::Float32, stream);
        Tensor norms({dim_size}, DType::Float32, r32.device());
        int grid_norm = static_cast<int>((dim_size + block - 1) / block);
        renorm_norm_kernel<float><<<grid_norm, block, 0, stream>>>(
            r32.data<float>(), norms.data<float>(),
            dim_size, outer_size, inner_size, static_cast<float>(p));
        HIP_CHECK(hipGetLastError());
        int64_t total = outer_size * dim_size * inner_size;
        int grid_scale = static_cast<int>((total + block - 1) / block);
        renorm_scale_kernel<float><<<grid_scale, block, 0, stream>>>(
            r32.data<float>(), norms.data<float>(),
            dim_size, outer_size, inner_size, static_cast<float>(maxnorm));
        HIP_CHECK(hipGetLastError());
        return cast_kernel(r32, orig, stream);
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

} // namespace rocm
} // namespace tenzor
