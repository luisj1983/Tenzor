#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>

#ifdef TENZOR_HAS_CUBLAS
#include <cublas_v2.h>
#endif

namespace tenzor {
namespace cuda {

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            throw std::runtime_error(                                          \
                std::string("CUDA error: ") + cudaGetErrorString(error));      \
        }                                                                      \
    } while (0)

// ============================================================================
// Tiled Matrix Multiplication Kernels
// ============================================================================

// Tile size for shared memory blocking
// Optimized for modern GPUs with 48KB+ shared memory per SM
constexpr int TILE_SIZE = 32;
constexpr int TILE_SIZE_K = 16;  // K-dimension tile for better memory coalescing

// ============================================================================
// Float32 Tiled MatMul Kernel
// ============================================================================

template<int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul_tiled_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {

    // Shared memory for tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column indices
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // Accumulator for the result
    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        // Each thread loads one element if within bounds
        if (tx < TILE_K) {
            int a_col = t * TILE_K + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A[row * lda + a_col];
            } else {
                As[ty][tx] = 0.0f;
            }
        }

        // Load tile of B into shared memory
        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            if (b_row < K && col < N) {
                Bs[ty][tx] = B[b_row * ldb + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

// ============================================================================
// Float64 Tiled MatMul Kernel
// ============================================================================

template<int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul_tiled_f64_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {

    // Shared memory for tiles
    __shared__ double As[TILE_M][TILE_K];
    __shared__ double Bs[TILE_K][TILE_N];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column indices
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // Accumulator for the result
    double sum = 0.0;

    // Loop over tiles
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        if (tx < TILE_K) {
            int a_col = t * TILE_K + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A[row * lda + a_col];
            } else {
                As[ty][tx] = 0.0;
            }
        }

        // Load tile of B into shared memory
        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            if (b_row < K && col < N) {
                Bs[ty][tx] = B[b_row * ldb + col];
            } else {
                Bs[ty][tx] = 0.0;
            }
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

// ============================================================================
// Int32 Tiled MatMul Kernel
// ============================================================================

template<int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul_tiled_i32_kernel(
    const int32_t* __restrict__ A,
    const int32_t* __restrict__ B,
    int32_t* __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {

    // Shared memory for tiles
    __shared__ int32_t As[TILE_M][TILE_K];
    __shared__ int32_t Bs[TILE_K][TILE_N];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column indices
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // Accumulator for the result
    int32_t sum = 0;

    // Loop over tiles
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        if (tx < TILE_K) {
            int a_col = t * TILE_K + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A[row * lda + a_col];
            } else {
                As[ty][tx] = 0;
            }
        }

        // Load tile of B into shared memory
        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            if (b_row < K && col < N) {
                Bs[ty][tx] = B[b_row * ldb + col];
            } else {
                Bs[ty][tx] = 0;
            }
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

// ============================================================================
// Batched Matrix Multiplication Kernels
// ============================================================================

template<int TILE_M, int TILE_N, int TILE_K>
__global__ void batched_matmul_tiled_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t batch_size,
    int64_t M, int64_t N, int64_t K,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {

    // Batch index
    int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size) {
        return;
    }

    // Offset pointers for this batch
    const float* A_batch = A + batch_idx * stride_a;
    const float* B_batch = B + batch_idx * stride_b;
    float* C_batch = C + batch_idx * stride_c;

    // Shared memory for tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column indices
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // Accumulator for the result
    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        if (tx < TILE_K) {
            int a_col = t * TILE_K + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A_batch[row * K + a_col];
            } else {
                As[ty][tx] = 0.0f;
            }
        }

        // Load tile of B into shared memory
        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            if (b_row < K && col < N) {
                Bs[ty][tx] = B_batch[b_row * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}

template<int TILE_M, int TILE_N, int TILE_K>
__global__ void batched_matmul_tiled_f64_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int64_t batch_size,
    int64_t M, int64_t N, int64_t K,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {

    // Batch index
    int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size) {
        return;
    }

    // Offset pointers for this batch
    const double* A_batch = A + batch_idx * stride_a;
    const double* B_batch = B + batch_idx * stride_b;
    double* C_batch = C + batch_idx * stride_c;

    // Shared memory for tiles
    __shared__ double As[TILE_M][TILE_K];
    __shared__ double Bs[TILE_K][TILE_N];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column indices
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // Accumulator for the result
    double sum = 0.0;

    // Loop over tiles
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        if (tx < TILE_K) {
            int a_col = t * TILE_K + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A_batch[row * K + a_col];
            } else {
                As[ty][tx] = 0.0;
            }
        }

        // Load tile of B into shared memory
        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            if (b_row < K && col < N) {
                Bs[ty][tx] = B_batch[b_row * N + col];
            } else {
                Bs[ty][tx] = 0.0;
            }
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}

// ============================================================================
// cuBLAS Helper Functions
// ============================================================================

#ifdef TENZOR_HAS_CUBLAS

// Global cuBLAS handle (initialized once)
static cublasHandle_t cublas_handle = nullptr;

cublasHandle_t get_cublas_handle() {
    if (cublas_handle == nullptr) {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
    }
    return cublas_handle;
}

// Use cuBLAS for large matrices (more optimized than our custom kernels)
void matmul_cublas_f32(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K) {

    cublasHandle_t handle = get_cublas_handle();

    // cuBLAS uses column-major order, we use row-major
    // To compute C = A @ B in row-major, we compute C^T = B^T @ A^T
    // Which means: cublasSgemm(..., B, A, C)

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,    // B is not transposed
        CUBLAS_OP_N,    // A is not transposed
        N,              // Rows of B^T (cols of B)
        M,              // Cols of A^T (rows of A)
        K,              // Cols of B^T = Rows of A^T
        &alpha,
        B, N,           // B matrix with leading dimension N
        A, K,           // A matrix with leading dimension K
        &beta,
        C, N            // C matrix with leading dimension N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS SGEMM failed");
    }
}

void matmul_cublas_f64(
    const double* A, const double* B, double* C,
    int64_t M, int64_t N, int64_t K) {

    cublasHandle_t handle = get_cublas_handle();

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasStatus_t status = cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS DGEMM failed");
    }
}

// Batched cuBLAS matmul
void batched_matmul_cublas_f32(
    const float* A, const float* B, float* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {

    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N, stride_b,
        A, K, stride_a,
        &beta,
        C, N, stride_c,
        batch_size
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS batched SGEMM failed");
    }
}

void batched_matmul_cublas_f64(
    const double* A, const double* B, double* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {

    cublasHandle_t handle = get_cublas_handle();

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasStatus_t status = cublasDgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N, stride_b,
        A, K, stride_a,
        &beta,
        C, N, stride_c,
        batch_size
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS batched DGEMM failed");
    }
}

#endif // TENZOR_HAS_CUBLAS

// ============================================================================
// Host Functions - Single Matrix Multiplication
// ============================================================================

// Threshold for using cuBLAS vs custom kernels
constexpr int64_t CUBLAS_THRESHOLD = 512;

void matmul_f32(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

#ifdef TENZOR_HAS_CUBLAS
    // Use cuBLAS for large matrices
    if (M >= CUBLAS_THRESHOLD || N >= CUBLAS_THRESHOLD || K >= CUBLAS_THRESHOLD) {
        matmul_cublas_f32(A, B, C, M, N, K);
        return;
    }
#endif

    // Use custom tiled kernel for smaller matrices
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_f32_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);

    CUDA_CHECK(cudaGetLastError());
}

void matmul_f64(
    const double* A, const double* B, double* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

#ifdef TENZOR_HAS_CUBLAS
    // Use cuBLAS for large matrices
    if (M >= CUBLAS_THRESHOLD || N >= CUBLAS_THRESHOLD || K >= CUBLAS_THRESHOLD) {
        matmul_cublas_f64(A, B, C, M, N, K);
        return;
    }
#endif

    // Use custom tiled kernel for smaller matrices
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_f64_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);

    CUDA_CHECK(cudaGetLastError());
}

void matmul_i32(
    const int32_t* A, const int32_t* B, int32_t* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    // Custom kernel (no cuBLAS for int32)
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_i32_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Host Functions - Batched Matrix Multiplication
// ============================================================================

void batched_matmul_f32(
    const float* A, const float* B, float* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    int64_t stride_a = M * K;
    int64_t stride_b = K * N;
    int64_t stride_c = M * N;

#ifdef TENZOR_HAS_CUBLAS
    // Use cuBLAS for large batched matrices
    if (M >= CUBLAS_THRESHOLD || N >= CUBLAS_THRESHOLD || K >= CUBLAS_THRESHOLD) {
        batched_matmul_cublas_f32(A, B, C, batch_size, M, N, K,
                                   stride_a, stride_b, stride_c);
        return;
    }
#endif

    // Use custom batched kernel
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    batched_matmul_tiled_f32_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
        <<<grid, block, 0, stream>>>(
            A, B, C, batch_size, M, N, K,
            stride_a, stride_b, stride_c);

    CUDA_CHECK(cudaGetLastError());
}

void batched_matmul_f64(
    const double* A, const double* B, double* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    int64_t stride_a = M * K;
    int64_t stride_b = K * N;
    int64_t stride_c = M * N;

#ifdef TENZOR_HAS_CUBLAS
    // Use cuBLAS for large batched matrices
    if (M >= CUBLAS_THRESHOLD || N >= CUBLAS_THRESHOLD || K >= CUBLAS_THRESHOLD) {
        batched_matmul_cublas_f64(A, B, C, batch_size, M, N, K,
                                   stride_a, stride_b, stride_c);
        return;
    }
#endif

    // Use custom batched kernel
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    batched_matmul_tiled_f64_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
        <<<grid, block, 0, stream>>>(
            A, B, C, batch_size, M, N, K,
            stride_a, stride_b, stride_c);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Public API - Matrix Multiplication
// ============================================================================

auto matmul_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    // Validate inputs
    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error("matmul requires contiguous tensors");
    }

    if (a.device().type != Device::Type::CUDA || b.device().type != Device::Type::CUDA) {
        throw std::runtime_error("matmul_kernel requires CUDA tensors");
    }

    // Handle 2D matrices
    if (a.ndim() == 2 && b.ndim() == 2) {
        auto a_shape = a.shape();
        auto b_shape = b.shape();

        int64_t M = a_shape[0];
        int64_t K = a_shape[1];
        int64_t K2 = b_shape[0];
        int64_t N = b_shape[1];

        if (K != K2) {
            throw std::runtime_error(
                "matmul dimension mismatch: (" + std::to_string(M) + "×" +
                std::to_string(K) + ") @ (" + std::to_string(K2) + "×" +
                std::to_string(N) + ")"
            );
        }

        // Create output tensor
        Tensor result({M, N}, a.dtype(), a.device());

        // Dispatch based on dtype
        if (a.dtype() == DType::Float32 && b.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();

            matmul_f32(a_data, b_data, c_data, M, N, K, stream);

        } else if (a.dtype() == DType::Float64 && b.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();

            matmul_f64(a_data, b_data, c_data, M, N, K, stream);

        } else if (a.dtype() == DType::Int32 && b.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();

            matmul_i32(a_data, b_data, c_data, M, N, K, stream);

        } else {
            throw std::runtime_error(
                "matmul unsupported dtype combination: " +
                std::string(dtype_name(a.dtype())) + " @ " +
                std::string(dtype_name(b.dtype()))
            );
        }

        return result;
    }

    // Handle batched 3D matrices (batch_size, M, N) @ (batch_size, N, K)
    if (a.ndim() == 3 && b.ndim() == 3) {
        auto a_shape = a.shape();
        auto b_shape = b.shape();

        int64_t batch_a = a_shape[0];
        int64_t batch_b = b_shape[0];

        if (batch_a != batch_b) {
            throw std::runtime_error(
                "Batch dimensions must match for batched matmul"
            );
        }

        int64_t batch_size = batch_a;
        int64_t M = a_shape[1];
        int64_t K = a_shape[2];
        int64_t K2 = b_shape[1];
        int64_t N = b_shape[2];

        if (K != K2) {
            throw std::runtime_error(
                "matmul dimension mismatch in batched matmul"
            );
        }

        // Create output tensor
        Tensor result({batch_size, M, N}, a.dtype(), a.device());

        // Dispatch based on dtype
        if (a.dtype() == DType::Float32 && b.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();

            batched_matmul_f32(a_data, b_data, c_data, batch_size, M, N, K, stream);

        } else if (a.dtype() == DType::Float64 && b.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();

            batched_matmul_f64(a_data, b_data, c_data, batch_size, M, N, K, stream);

        } else {
            throw std::runtime_error(
                "Batched matmul unsupported dtype: " +
                std::string(dtype_name(a.dtype()))
            );
        }

        return result;
    }

    throw std::runtime_error(
        "matmul requires 2D or 3D tensors, got " +
        std::to_string(a.ndim()) + "D and " + std::to_string(b.ndim()) + "D"
    );
}

} // namespace cuda
} // namespace tenzor
