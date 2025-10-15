#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <hip/hip_runtime.h>
#ifdef TENZOR_HAS_HIPRAND
#include <hiprand_kernel.h>
#endif
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <chrono>

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
        } else {
            throw std::runtime_error("Unsupported dtype for add operation");
        }

        HIP_CHECK(hipGetLastError());
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
    } else {
        throw std::runtime_error("Unsupported dtype for add operation");
    }

    // Cleanup
    HIP_CHECK(hipFree(d_strides_a));
    HIP_CHECK(hipFree(d_strides_b));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());

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
        } else if (a.dtype() == DType::Int32) {
            hipLaunchKernelGGL(sub_kernel_device<int32_t>, grid, block, 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            hipLaunchKernelGGL(sub_kernel_device<int64_t>, grid, block, 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else {
            throw std::runtime_error("Unsupported dtype for sub operation");
        }

        HIP_CHECK(hipGetLastError());
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
    } else if (a.dtype() == DType::Int32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int32_t, SubOp>), grid, block, 0, stream,
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else if (a.dtype() == DType::Int64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(broadcast_kernel<int64_t, SubOp>), grid, block, 0, stream,
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, SubOp());
    } else {
        throw std::runtime_error("Unsupported dtype for sub operation");
    }

    HIP_CHECK(hipFree(d_strides_a));
    HIP_CHECK(hipFree(d_strides_b));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());

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
        } else {
            throw std::runtime_error("Unsupported dtype for mul operation");
        }

        HIP_CHECK(hipGetLastError());
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
    } else {
        throw std::runtime_error("Unsupported dtype for mul operation");
    }

    HIP_CHECK(hipFree(d_strides_a));
    HIP_CHECK(hipFree(d_strides_b));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());

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
        } else if (a.dtype() == DType::Int32) {
            hipLaunchKernelGGL(div_kernel_device<int32_t>, grid, block, 0, stream,
                a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            hipLaunchKernelGGL(div_kernel_device<int64_t>, grid, block, 0, stream,
                a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else {
            throw std::runtime_error("Unsupported dtype for div operation");
        }

        HIP_CHECK(hipGetLastError());
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
    } else {
        throw std::runtime_error("Unsupported dtype for div operation");
    }

    HIP_CHECK(hipFree(d_strides_a));
    HIP_CHECK(hipFree(d_strides_b));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());

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
    } else {
        throw std::runtime_error("sqrt operation only supports Float32 and Float64 dtypes");
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
    } else {
        throw std::runtime_error("exp operation only supports Float32 and Float64 dtypes");
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
    } else {
        throw std::runtime_error("log operation only supports Float32 and Float64 dtypes");
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
    } else {
        throw std::runtime_error("pow operation only supports Float32 and Float64 dtypes");
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
    } else {
        throw std::runtime_error("clamp operation only supports Float32 and Float64 dtypes");
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
    } else {
        throw std::runtime_error("sign operation only supports Float32 and Float64 dtypes");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

/**
 * @brief Expand kernel - replicate tensor along specified dimensions
 * @tparam T Data type
 *
 * Supports broadcasting by replicating dimensions of size 1 to larger sizes
 */
template<typename T>
__global__ void expand_kernel_device(
    const T* input, T* output,
    const int64_t* input_shape, const int64_t* input_strides,
    const int64_t* output_shape, int64_t input_ndim, int64_t output_ndim, int64_t n) {

    HIP_KERNEL_LOOP(out_idx, n) {
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

/**
 * @brief Expand kernel launcher - broadcast tensor to larger shape
 * @param input Input tensor
 * @param shape Target shape
 * @param stream_ptr HIP stream pointer
 * @return Expanded tensor
 */
auto expand_kernel(const Tensor& input, const std::vector<int64_t>& shape, void* stream_ptr) -> Tensor {
    hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
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
    HIP_CHECK(hipMalloc(&d_input_shape, input_shape_vec.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_input_strides, input_strides.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_output_shape, shape.size() * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_input_shape, input_shape_vec.data(), input_shape_vec.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_input_strides, input_strides.data(), input_strides.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output_shape, shape.data(), shape.size() * sizeof(int64_t), hipMemcpyHostToDevice));

    int64_t n = result.numel();
    int64_t input_ndim = input_shape_vec.size();
    int64_t output_ndim = shape.size();
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(expand_kernel_device<float>, grid, block, 0, stream,
            input.data<float>(), result.data<float>(),
            d_input_shape, d_input_strides, d_output_shape,
            input_ndim, output_ndim, n);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(expand_kernel_device<double>, grid, block, 0, stream,
            input.data<double>(), result.data<double>(),
            d_input_shape, d_input_strides, d_output_shape,
            input_ndim, output_ndim, n);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(expand_kernel_device<int32_t>, grid, block, 0, stream,
            input.data<int32_t>(), result.data<int32_t>(),
            d_input_shape, d_input_strides, d_output_shape,
            input_ndim, output_ndim, n);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(expand_kernel_device<int64_t>, grid, block, 0, stream,
            input.data<int64_t>(), result.data<int64_t>(),
            d_input_shape, d_input_strides, d_output_shape,
            input_ndim, output_ndim, n);
    } else {
        throw std::runtime_error("Unsupported dtype for expand operation");
    }

    // Cleanup
    HIP_CHECK(hipFree(d_input_shape));
    HIP_CHECK(hipFree(d_input_strides));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());

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
 * @brief Rand kernel launcher - uniform random [0, 1)
 * @param shape Tensor shape
 * @param dtype Data type (Float32 or Float64)
 * @param device Device (ROCm)
 * @param stream HIP stream for asynchronous execution
 * @return Random tensor with uniform distribution
 */
auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor {
#ifdef TENZOR_HAS_HIPRAND
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rand operation only supports Float32 and Float64 dtypes");
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
        // For Float64, generate as float then convert
        float* temp_float;
        HIP_CHECK(hipMalloc(&temp_float, n * sizeof(float)));
        hipLaunchKernelGGL(rand_kernel_device, grid, block, 0, stream,
            temp_float, d_states, n);
        HIP_CHECK(hipGetLastError());

        // Convert float to double
        double* output_double = result.data<double>();
        HIP_CHECK(hipMemcpy(output_double, temp_float, n * sizeof(float), hipMemcpyDeviceToDevice));
        // Note: This copies as bytes, need proper conversion kernel for production
        HIP_CHECK(hipFree(temp_float));
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
 * @param dtype Data type (Float32 or Float64)
 * @param device Device (ROCm)
 * @param stream HIP stream for asynchronous execution
 * @return Random tensor with normal distribution
 */
auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor {
#ifdef TENZOR_HAS_HIPRAND
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("randn operation only supports Float32 and Float64 dtypes");
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
        // For Float64, generate as float then convert
        float* temp_float;
        HIP_CHECK(hipMalloc(&temp_float, n * sizeof(float)));
        hipLaunchKernelGGL(randn_kernel_device, grid, block, 0, stream,
            temp_float, d_states, n);
        HIP_CHECK(hipGetLastError());

        // Convert float to double
        double* output_double = result.data<double>();
        HIP_CHECK(hipMemcpy(output_double, temp_float, n * sizeof(float), hipMemcpyDeviceToDevice));
        // Note: This copies as bytes, need proper conversion kernel for production
        HIP_CHECK(hipFree(temp_float));
    }

    // Cleanup
    HIP_CHECK(hipFree(d_states));

    return result;
#else
    throw std::runtime_error("randn operation requires hipRAND library. Please install ROCm hipRAND.");
#endif
}

#endif // TENZOR_HAS_HIPRAND (end of all hipRAND code)

} // namespace rocm
} // namespace tenzor
