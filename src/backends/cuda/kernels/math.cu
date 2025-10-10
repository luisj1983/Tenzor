#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <chrono>

namespace tenzor {
namespace cuda {

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
    grid = dim3((n + block_size - 1) / block_size, 1, 1);
}

// Grid-stride loop pattern for better scalability
#define CUDA_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

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
        } else {
            throw std::runtime_error("Unsupported dtype for add operation");
        }

        CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t n = result.numel();
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
    } else {
        throw std::runtime_error("Unsupported dtype for add operation");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_strides_a));
    CUDA_CHECK(cudaFree(d_strides_b));
    CUDA_CHECK(cudaFree(d_output_shape));
    CUDA_CHECK(cudaGetLastError());

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
        } else {
            throw std::runtime_error("Unsupported dtype for sub operation");
        }

        CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t n = result.numel();
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
    } else {
        throw std::runtime_error("Unsupported dtype for sub operation");
    }

    CUDA_CHECK(cudaFree(d_strides_a));
    CUDA_CHECK(cudaFree(d_strides_b));
    CUDA_CHECK(cudaFree(d_output_shape));
    CUDA_CHECK(cudaGetLastError());

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
        } else {
            throw std::runtime_error("Unsupported dtype for mul operation");
        }

        CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t n = result.numel();
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
    } else {
        throw std::runtime_error("Unsupported dtype for mul operation");
    }

    CUDA_CHECK(cudaFree(d_strides_a));
    CUDA_CHECK(cudaFree(d_strides_b));
    CUDA_CHECK(cudaFree(d_output_shape));
    CUDA_CHECK(cudaGetLastError());

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
            div_kernel_device<<<grid, block, 0, stream>>>(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), n);
        } else if (a.dtype() == DType::Int64) {
            div_kernel_device<<<grid, block, 0, stream>>>(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), n);
        } else {
            throw std::runtime_error("Unsupported dtype for div operation");
        }

        CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaMalloc(&d_strides_a, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_strides_b, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_shape, output_shape.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides_a, strides_a.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_b, strides_b.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t n = result.numel();
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
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else if (a.dtype() == DType::Int64) {
        broadcast_kernel<<<grid, block, 0, stream>>>(
            a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
            d_strides_a, d_strides_b, d_output_shape, ndim, n, DivOp());
    } else {
        throw std::runtime_error("Unsupported dtype for div operation");
    }

    CUDA_CHECK(cudaFree(d_strides_a));
    CUDA_CHECK(cudaFree(d_strides_b));
    CUDA_CHECK(cudaFree(d_output_shape));
    CUDA_CHECK(cudaGetLastError());

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
    } else {
        throw std::runtime_error("Unsupported dtype for neg operation");
    }

    CUDA_CHECK(cudaGetLastError());
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
    } else {
        throw std::runtime_error("Unsupported dtype for abs operation");
    }

    CUDA_CHECK(cudaGetLastError());
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
    } else {
        throw std::runtime_error("sqrt operation only supports Float32 and Float64 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
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
    } else {
        throw std::runtime_error("exp operation only supports Float32 and Float64 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
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
    } else {
        throw std::runtime_error("log operation only supports Float32 and Float64 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
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
    } else {
        throw std::runtime_error("pow operation only supports Float32 and Float64 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
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
    } else {
        throw std::runtime_error("clamp operation only supports Float32 and Float64 dtypes");
    }

    CUDA_CHECK(cudaGetLastError());
    return result;
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
    } else {
        throw std::runtime_error("Unsupported dtype for expand operation");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input_shape));
    CUDA_CHECK(cudaFree(d_input_strides));
    CUDA_CHECK(cudaFree(d_output_shape));
    CUDA_CHECK(cudaGetLastError());

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
    } else {
        throw std::runtime_error("Unsupported dtype for fill operation");
    }

    CUDA_CHECK(cudaGetLastError());
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
    } else {
        throw std::runtime_error("Unsupported dtype for zeros operation");
    }

    CUDA_CHECK(cudaGetLastError());
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
    } else {
        throw std::runtime_error("Unsupported dtype for ones operation");
    }

    CUDA_CHECK(cudaGetLastError());
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
    } else {
        throw std::runtime_error("Unsupported dtype for full operation");
    }

    CUDA_CHECK(cudaGetLastError());
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

// Rand kernel launcher - uniform random [0, 1)
auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor {
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

    // Allocate cuRAND states
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));

    // Initialize states with timestamp-based seed for randomness
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    init_curand_states<<<grid, block, 0, stream>>>(d_states, seed, n);
    CUDA_CHECK(cudaGetLastError());

    if (dtype == DType::Float32) {
        // Generate uniform random numbers
        rand_kernel_device<<<grid, block, 0, stream>>>(result.data<float>(), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        // For Float64, generate as float then convert
        float* temp_float;
        CUDA_CHECK(cudaMalloc(&temp_float, n * sizeof(float)));
        rand_kernel_device<<<grid, block, 0, stream>>>(temp_float, d_states, n);
        CUDA_CHECK(cudaGetLastError());

        // Convert float to double
        double* output_double = result.data<double>();
        CUDA_CHECK(cudaMemcpy(output_double, temp_float, n * sizeof(float), cudaMemcpyDeviceToDevice));
        // Note: This copies as bytes, need proper conversion kernel for production
        CUDA_CHECK(cudaFree(temp_float));
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_states));

    return result;
}

// Randn kernel launcher - normal distribution N(0,1)
auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor {
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

    // Allocate cuRAND states
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));

    // Initialize states with timestamp-based seed
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    init_curand_states<<<grid, block, 0, stream>>>(d_states, seed, n);
    CUDA_CHECK(cudaGetLastError());

    if (dtype == DType::Float32) {
        // Generate normal random numbers
        randn_kernel_device<<<grid, block, 0, stream>>>(result.data<float>(), d_states, n);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        // For Float64, generate as float then convert
        float* temp_float;
        CUDA_CHECK(cudaMalloc(&temp_float, n * sizeof(float)));
        randn_kernel_device<<<grid, block, 0, stream>>>(temp_float, d_states, n);
        CUDA_CHECK(cudaGetLastError());

        // Convert float to double
        double* output_double = result.data<double>();
        CUDA_CHECK(cudaMemcpy(output_double, temp_float, n * sizeof(float), cudaMemcpyDeviceToDevice));
        // Note: This copies as bytes, need proper conversion kernel for production
        CUDA_CHECK(cudaFree(temp_float));
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_states));

    return result;
}

} // namespace cuda
} // namespace tenzor
