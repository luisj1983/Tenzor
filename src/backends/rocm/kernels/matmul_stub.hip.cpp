/**
 * @file matmul_stub.hip.cpp
 * @brief Matrix multiplication implementation using rocBLAS for ROCm backend
 *
 * Provides GEMM (General Matrix Multiply) operations using AMD's rocBLAS library.
 * Supports both 2D matrix multiplication and batched operations for float and double precision.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdexcept>
#include <vector>
#include <memory>

namespace tenzor {
namespace rocm {

// ============================================================================
// Error Checking Macros
// ============================================================================

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)

#define ROCBLAS_CHECK(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        throw std::runtime_error(std::string("rocBLAS error: ") + rocblas_status_to_string(status)); \
    } \
} while(0)

// ============================================================================
// rocBLAS Handle Management
// ============================================================================

/**
 * @brief Thread-safe rocBLAS handle manager using RAII
 *
 * Manages rocBLAS handle lifecycle with automatic cleanup.
 * Each handle is associated with a specific HIP stream for async execution.
 */
class RocblasHandle {
public:
    RocblasHandle() {
        ROCBLAS_CHECK(rocblas_create_handle(&handle_));
    }

    ~RocblasHandle() {
        if (handle_) {
            rocblas_destroy_handle(handle_);
        }
    }

    // No copy
    RocblasHandle(const RocblasHandle&) = delete;
    RocblasHandle& operator=(const RocblasHandle&) = delete;

    // Allow move
    RocblasHandle(RocblasHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    RocblasHandle& operator=(RocblasHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                rocblas_destroy_handle(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    rocblas_handle get() const { return handle_; }

    void set_stream(hipStream_t stream) {
        ROCBLAS_CHECK(rocblas_set_stream(handle_, stream));
    }

private:
    rocblas_handle handle_ = nullptr;
};

// ============================================================================
// Native HIP Matrix Multiplication Kernels (Fallback)
// ============================================================================

/**
 * @brief Tiled matrix multiplication kernel for Float32
 *
 * Uses shared memory tiling for efficient computation.
 * Computes C = A * B where:
 * - A is M x K (row-major)
 * - B is K x N (row-major)
 * - C is M x N (row-major)
 */
template<int TILE_SIZE = 16>
__global__ void matmul_tiled_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * @brief Tiled matrix multiplication kernel for Float64
 */
template<int TILE_SIZE = 16>
__global__ void matmul_tiled_f64_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int M, int N, int K
) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    double sum = 0.0;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0;
        }

        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * @brief Native HIP matrix multiplication (fallback when rocBLAS unavailable)
 */
static Tensor matmul_native_hip(const Tensor& a, const Tensor& b, hipStream_t stream) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    int64_t M = a_shape[a_shape.size() - 2];
    int64_t K_a = a_shape[a_shape.size() - 1];
    int64_t K_b = b_shape[b_shape.size() - 2];
    int64_t N = b_shape[b_shape.size() - 1];

    if (K_a != K_b) {
        throw std::runtime_error("matmul: incompatible matrix dimensions");
    }
    int64_t K = K_a;

    // Calculate batch size
    int64_t batch_size_a = 1;
    int64_t batch_size_b = 1;
    for (size_t i = 0; i < a_shape.size() - 2; ++i) {
        batch_size_a *= a_shape[i];
    }
    for (size_t i = 0; i < b_shape.size() - 2; ++i) {
        batch_size_b *= b_shape[i];
    }

    int64_t batch_size = std::max(batch_size_a, batch_size_b);
    if (batch_size_a != 1 && batch_size_b != 1 && batch_size_a != batch_size_b) {
        throw std::runtime_error("matmul: incompatible batch dimensions");
    }

    // Build output shape
    std::vector<int64_t> out_shape;
    size_t max_ndim = std::max(a_shape.size(), b_shape.size());
    for (size_t i = 0; i < max_ndim - 2; ++i) {
        int64_t dim_a = i < (a_shape.size() - 2) ? a_shape[i] : 1;
        int64_t dim_b = i < (b_shape.size() - 2) ? b_shape[i] : 1;
        out_shape.push_back(std::max(dim_a, dim_b));
    }
    out_shape.push_back(M);
    out_shape.push_back(N);

    Tensor result(out_shape, a.dtype(), a.device());

    if (M == 0 || N == 0 || K == 0) {
        return result;
    }

    // Launch kernel
    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    int64_t stride_a = M * K;
    int64_t stride_b = K * N;
    int64_t stride_c = M * N;

    if (a.dtype() == DType::Float32) {
        const float* a_data = a.data<float>();
        const float* b_data = b.data<float>();
        float* c_data = result.data<float>();

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            int64_t a_offset = (batch_size_a == 1) ? 0 : batch * stride_a;
            int64_t b_offset = (batch_size_b == 1) ? 0 : batch * stride_b;
            int64_t c_offset = batch * stride_c;

            matmul_tiled_f32_kernel<TILE_SIZE><<<grid, block, 0, stream>>>(
                a_data + a_offset,
                b_data + b_offset,
                c_data + c_offset,
                M, N, K
            );
        }
    } else if (a.dtype() == DType::Float64) {
        const double* a_data = a.data<double>();
        const double* b_data = b.data<double>();
        double* c_data = result.data<double>();

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            int64_t a_offset = (batch_size_a == 1) ? 0 : batch * stride_a;
            int64_t b_offset = (batch_size_b == 1) ? 0 : batch * stride_b;
            int64_t c_offset = batch * stride_c;

            matmul_tiled_f64_kernel<TILE_SIZE><<<grid, block, 0, stream>>>(
                a_data + a_offset,
                b_data + b_offset,
                c_data + c_offset,
                M, N, K
            );
        }
    }

    HIP_CHECK(hipGetLastError());

    return result;
}

// ============================================================================
// Matrix Multiplication Implementation (rocBLAS with Native Fallback)
// ============================================================================

/**
 * @brief Matrix multiplication using rocBLAS GEMM with native HIP fallback
 * @param a First input tensor (M x K)
 * @param b Second input tensor (K x N)
 * @param stream HIP stream for asynchronous execution
 * @return Result tensor (M x N)
 *
 * Performs C = α * A * B + β * C where α = 1.0, β = 0.0
 *
 * Supports:
 * - 2D matrix multiplication: (M x K) @ (K x N) -> (M x N)
 * - Batched matrix multiplication for higher dimensional tensors
 * - Float32 and Float64 precision
 *
 * Strategy:
 * 1. Try rocBLAS GEMM first (optimal performance on supported architectures)
 * 2. Fall back to native HIP kernel if rocBLAS fails (e.g., gfx90c APU)
 *
 * Note: rocBLAS uses column-major layout while Tenzor uses row-major.
 * We handle this by transposing the operation: C^T = B^T * A^T
 */
auto matmul_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor {
    // Validate input dtypes match
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("matmul: input tensors must have the same dtype");
    }

    // Only support float and double precision
    if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64) {
        throw std::runtime_error("matmul: only Float32 and Float64 dtypes are supported");
    }

    // Get tensor shapes
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // Validate dimensions (at least 2D)
    if (a_shape.size() < 2 || b_shape.size() < 2) {
        throw std::runtime_error("matmul: input tensors must be at least 2-dimensional");
    }

    // Extract matrix dimensions
    // For A: shape is [..., M, K]
    // For B: shape is [..., K, N]
    int64_t M = a_shape[a_shape.size() - 2];
    int64_t K_a = a_shape[a_shape.size() - 1];
    int64_t K_b = b_shape[b_shape.size() - 2];
    int64_t N = b_shape[b_shape.size() - 1];

    // Validate inner dimensions match
    if (K_a != K_b) {
        throw std::runtime_error(
            "matmul: incompatible matrix dimensions. Inner dimensions must match: " +
            std::to_string(K_a) + " != " + std::to_string(K_b)
        );
    }
    int64_t K = K_a;

    // Calculate batch size for batched operations
    int64_t batch_size_a = 1;
    int64_t batch_size_b = 1;
    for (size_t i = 0; i < a_shape.size() - 2; ++i) {
        batch_size_a *= a_shape[i];
    }
    for (size_t i = 0; i < b_shape.size() - 2; ++i) {
        batch_size_b *= b_shape[i];
    }

    // Handle broadcasting for batch dimensions
    int64_t batch_size = std::max(batch_size_a, batch_size_b);
    if (batch_size_a != 1 && batch_size_b != 1 && batch_size_a != batch_size_b) {
        throw std::runtime_error(
            "matmul: incompatible batch dimensions: " +
            std::to_string(batch_size_a) + " vs " + std::to_string(batch_size_b)
        );
    }

    // Build output shape
    std::vector<int64_t> out_shape;
    size_t max_ndim = std::max(a_shape.size(), b_shape.size());
    for (size_t i = 0; i < max_ndim - 2; ++i) {
        int64_t dim_a = i < (a_shape.size() - 2) ? a_shape[i] : 1;
        int64_t dim_b = i < (b_shape.size() - 2) ? b_shape[i] : 1;
        out_shape.push_back(std::max(dim_a, dim_b));
    }
    out_shape.push_back(M);
    out_shape.push_back(N);

    // Create output tensor
    Tensor result(out_shape, a.dtype(), a.device());

    // Handle empty matrices
    if (M == 0 || N == 0 || K == 0) {
        return result;
    }

    // Try rocBLAS first, fallback to native HIP on failure
    // rocBLAS may fail on unsupported architectures (e.g., gfx90c)
    bool use_rocblas = true;

    try {
        // Create rocBLAS handle and set stream
        RocblasHandle handle;
        handle.set_stream(stream);

        // rocBLAS GEMM parameters
        // Note: rocBLAS uses column-major, we use row-major
        // To compute C = A * B in row-major, we compute C^T = B^T * A^T in column-major
        rocblas_operation trans_a = rocblas_operation_none;
        rocblas_operation trans_b = rocblas_operation_none;

        // Leading dimensions (stride of major dimension)
        // In row-major: ld = number of columns (stride to next row)
        rocblas_int lda = static_cast<rocblas_int>(K);  // A is M x K, stride = K
        rocblas_int ldb = static_cast<rocblas_int>(N);  // B is K x N, stride = N
        rocblas_int ldc = static_cast<rocblas_int>(N);  // C is M x N, stride = N

        // Cast dimensions to rocblas_int
        rocblas_int m = static_cast<rocblas_int>(N);  // Swapped for transpose
        rocblas_int n = static_cast<rocblas_int>(M);  // Swapped for transpose
        rocblas_int k = static_cast<rocblas_int>(K);

        // Stride between matrices in batch (for batched operations)
        rocblas_stride stride_a = M * K;
        rocblas_stride stride_b = K * N;
        rocblas_stride stride_c = M * N;

        // GEMM coefficients: C = alpha * A * B + beta * C
        // We want C = A * B, so alpha = 1.0, beta = 0.0

        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();

            float alpha = 1.0f;
            float beta = 0.0f;

            if (batch_size == 1) {
                // Single matrix multiplication
                // C = B * A (transposed for row-major)
                ROCBLAS_CHECK(rocblas_sgemm(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, ldb,  // B is K x N
                    a_data, lda,  // A is M x K
                    &beta,
                    c_data, ldc   // C is M x N
                ));
            } else {
                // Batched matrix multiplication
                rocblas_int batch_count = static_cast<rocblas_int>(batch_size);

                // Handle broadcasting: if one tensor has batch_size=1, replicate it
                rocblas_stride actual_stride_a = (batch_size_a == 1) ? 0 : stride_a;
                rocblas_stride actual_stride_b = (batch_size_b == 1) ? 0 : stride_b;

                ROCBLAS_CHECK(rocblas_sgemm_strided_batched(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, ldb, actual_stride_b,
                    a_data, lda, actual_stride_a,
                    &beta,
                    c_data, ldc, stride_c,
                    batch_count
                ));
            }
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();

            double alpha = 1.0;
            double beta = 0.0;

            if (batch_size == 1) {
                // Single matrix multiplication
                ROCBLAS_CHECK(rocblas_dgemm(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, ldb,
                    a_data, lda,
                    &beta,
                    c_data, ldc
                ));
            } else {
                // Batched matrix multiplication
                rocblas_int batch_count = static_cast<rocblas_int>(batch_size);

                rocblas_stride actual_stride_a = (batch_size_a == 1) ? 0 : stride_a;
                rocblas_stride actual_stride_b = (batch_size_b == 1) ? 0 : stride_b;

                ROCBLAS_CHECK(rocblas_dgemm_strided_batched(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, ldb, actual_stride_b,
                    a_data, lda, actual_stride_a,
                    &beta,
                    c_data, ldc, stride_c,
                    batch_count
                ));
            }
        }
    } catch (const std::exception& e) {
        // rocBLAS failed (e.g., unsupported architecture like gfx90c)
        // Fall back to native HIP implementation
        use_rocblas = false;

        // Log warning once (static to avoid spam)
        static bool warning_printed = false;
        if (!warning_printed) {
            fprintf(stderr, "Warning: rocBLAS matmul failed (%s), using native HIP fallback\n", e.what());
            warning_printed = true;
        }

        return matmul_native_hip(a, b, stream);
    }

    return result;
}

// Note: Reduction operations (sum_kernel, mean_kernel, max_kernel, min_kernel)
// are now implemented in reduction.hip.cpp

// ============================================================================
// Random Operations (Conditional Compilation)
// ============================================================================

#ifndef TENZOR_HAS_HIPRAND
/**
 * @brief Uniform random generation stub
 * Requires hipRAND library
 */
auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor {
    throw std::runtime_error("rand_kernel requires hipRAND library. Please install ROCm hipRAND.");
}

/**
 * @brief Normal random generation stub
 * Requires hipRAND library
 */
auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor {
    throw std::runtime_error("randn_kernel requires hipRAND library. Please install ROCm hipRAND.");
}
#endif

} // namespace rocm
} // namespace tenzor
