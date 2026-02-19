#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>        // For __half
#include <cuda_bf16.h>        // For __nv_bfloat16
#include <mma.h>              // For Tensor Cores (WMMA)
#include <stdexcept>
#include <string>

#ifdef TENZOR_HAS_CUBLAS
#include <cublas_v2.h>
#endif

#include <mutex>
#include <random>

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
// FP16/BF16 Conversion Utilities
// ============================================================================

// Convert Tenzor Float16 to CUDA __half
__device__ __host__ inline __half to_cuda_half(const Float16& x) {
    __half_raw raw;
    raw.x = x.bits;
    return raw;
}

// Convert CUDA __half to Tenzor Float16
__device__ __host__ inline Float16 from_cuda_half(const __half& x) {
    return Float16(__half_as_ushort(x));
}

// Convert Tenzor BFloat16 to CUDA __nv_bfloat16
__device__ __host__ inline __nv_bfloat16 to_cuda_bfloat16(const BFloat16& x) {
    __nv_bfloat16_raw raw;
    raw.x = x.bits;
    return raw;
}

// Convert CUDA __nv_bfloat16 to Tenzor BFloat16
__device__ __host__ inline BFloat16 from_cuda_bfloat16(const __nv_bfloat16& x) {
    return BFloat16(__bfloat16_as_ushort(x));
}

// ============================================================================
// Tiled Matrix Multiplication Kernels
// ============================================================================

// Tile size for shared memory blocking
// Optimized for modern GPUs with 48KB+ shared memory per SM
constexpr int TILE_SIZE = 32;
constexpr int TILE_SIZE_K = 16;  // K-dimension tile for better memory coalescing
constexpr int TILE_SIZE_F16 = 16;  // Smaller tile size for FP16 to reduce resource usage

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
// FP16 Tensor Core Matrix Multiplication (WMMA)
// ============================================================================

// WMMA dimensions for FP16 Tensor Cores
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

/**
 * FP16 Tensor Core matmul using WMMA API
 * Requires matrix dimensions to be multiples of 16
 * Uses row-major layout for both input and output matrices
 * Each warp computes one 16x16 output tile
 */
__global__ void matmul_tensor_core_f16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int64_t M, int64_t N, int64_t K) {

    using namespace nvcuda::wmma;

    // Calculate warp position in output matrix
    // With 1 warp per block, each block computes one 16x16 output tile
    // All 32 threads in the warp work together on WMMA operations
    const int warpM = blockIdx.y;
    const int warpN = blockIdx.x;

    // Check if this warp is within bounds
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) {
        return;
    }

    // Declare WMMA fragments
    // Use float accumulator for numerical stability (critical for Float16)
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator to zero
    fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension in chunks of WMMA_K
    for (int64_t k = 0; k < K; k += WMMA_K) {
        // Bounds check for K dimension
        if (k + WMMA_K <= K) {
            // Calculate starting positions in A and B
            const int64_t aRow = warpM * WMMA_M;
            const int64_t aCol = k;
            const int64_t bRow = k;
            const int64_t bCol = warpN * WMMA_N;

            // Load matrices from global memory into fragments
            load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform matrix multiply-accumulate using Tensor Cores
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        } else {
            // Handle edge case where K is not a multiple of WMMA_K
            // Load partial tiles with bounds checking
            const int64_t aRow = warpM * WMMA_M;
            const int64_t aCol = k;
            const int64_t bRow = k;
            const int64_t bCol = warpN * WMMA_N;

            // Create temporary padded tiles in shared memory
            __shared__ __half As[WMMA_M][WMMA_K];
            __shared__ __half Bs[WMMA_K][WMMA_N];

            // Load with bounds checking
            for (int i = threadIdx.y; i < WMMA_M; i += blockDim.y) {
                for (int j = threadIdx.x; j < WMMA_K; j += blockDim.x) {
                    const int64_t row = aRow + i;
                    const int64_t col = aCol + j;
                    As[i][j] = (row < M && col < K) ? A[row * K + col] : __float2half(0.0f);
                }
            }

            for (int i = threadIdx.y; i < WMMA_K; i += blockDim.y) {
                for (int j = threadIdx.x; j < WMMA_N; j += blockDim.x) {
                    const int64_t row = bRow + i;
                    const int64_t col = bCol + j;
                    Bs[i][j] = (row < K && col < N) ? B[row * N + col] : __float2half(0.0f);
                }
            }

            __syncthreads();

            // Load from shared memory
            load_matrix_sync(a_frag, &As[0][0], WMMA_K);
            load_matrix_sync(b_frag, &Bs[0][0], WMMA_N);

            // Perform matrix multiply-accumulate
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);

            __syncthreads();
        }
    }

    // Store result to global memory
    // Float accumulator needs to be converted to half when storing
    const int64_t cRow = warpM * WMMA_M;
    const int64_t cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        // Store float accumulator to shared memory first
        __shared__ float Cs_float[WMMA_M][WMMA_N];
        store_matrix_sync(&Cs_float[0][0], acc_frag, WMMA_N, mem_row_major);

        __syncthreads();

        // Convert float to half and store to global memory with bounds checking
        for (int i = threadIdx.y; i < WMMA_M; i += blockDim.y) {
            for (int j = threadIdx.x; j < WMMA_N; j += blockDim.x) {
                const int64_t row = cRow + i;
                const int64_t col = cCol + j;
                if (row < M && col < N) {
                    C[row * N + col] = __float2half(Cs_float[i][j]);
                }
            }
        }
    }
}

/**
 * FP16 fallback kernel for non-16-aligned dimensions
 * Standard tiled matmul without Tensor Cores
 */
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul_tiled_f16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {

    // Shared memory for tiles
    __shared__ __half As[TILE_M][TILE_K];
    __shared__ __half Bs[TILE_K][TILE_N];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column indices
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // Accumulator for the result (use float for numerical stability)
    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        if (tx < TILE_K) {
            int a_col = t * TILE_K + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A[row * lda + a_col];
            } else {
                As[ty][tx] = __float2half(0.0f);
            }
        }

        // Load tile of B into shared memory
        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            if (b_row < K && col < N) {
                Bs[ty][tx] = B[b_row * ldb + col];
            } else {
                Bs[ty][tx] = __float2half(0.0f);
            }
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product (accumulate in float)
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result (convert float back to half)
    if (row < M && col < N) {
        C[row * ldc + col] = __float2half(sum);
    }
}

/**
 * Batched FP16 tiled matmul (non-Tensor Core path)
 * Uses blockIdx.z for batch indexing to avoid host-side per-batch loop
 */
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void batched_matmul_tiled_f16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {

    int batch = blockIdx.z;
    const __half* A_batch = A + batch * stride_a;
    const __half* B_batch = B + batch * stride_b;
    __half* C_batch = C + batch * stride_c;

    __shared__ __half As[TILE_M][TILE_K];
    __shared__ __half Bs[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        if (tx < TILE_K) {
            int a_col = t * TILE_K + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A_batch[row * lda + a_col];
            } else {
                As[ty][tx] = __float2half(0.0f);
            }
        }

        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            if (b_row < K && col < N) {
                Bs[ty][tx] = B_batch[b_row * ldb + col];
            } else {
                Bs[ty][tx] = __float2half(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_batch[row * ldc + col] = __float2half(sum);
    }
}

/**
 * Batched FP16 Tensor Core matmul
 */
__global__ void batched_matmul_tensor_core_f16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int64_t batch_size,
    int64_t M, int64_t N, int64_t K,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {

    using namespace nvcuda::wmma;

    // Batch index
    const int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size) {
        return;
    }

    // Offset pointers for this batch
    const __half* A_batch = A + batch_idx * stride_a;
    const __half* B_batch = B + batch_idx * stride_b;
    __half* C_batch = C + batch_idx * stride_c;

    // Calculate warp position in output matrix
    // With 1 warp per block, each block computes one 16x16 output tile
    const int warpM = blockIdx.y;
    const int warpN = blockIdx.x;

    // Check if this warp is within bounds
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) {
        return;
    }

    // Declare WMMA fragments
    // Use float accumulator for numerical stability (critical for Float16)
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator to zero
    fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension in chunks of WMMA_K
    for (int64_t k = 0; k < K; k += WMMA_K) {
        if (k + WMMA_K <= K) {
            const int64_t aRow = warpM * WMMA_M;
            const int64_t aCol = k;
            const int64_t bRow = k;
            const int64_t bCol = warpN * WMMA_N;

            // Load matrices from global memory into fragments
            load_matrix_sync(a_frag, A_batch + aRow * K + aCol, K);
            load_matrix_sync(b_frag, B_batch + bRow * N + bCol, N);

            // Perform matrix multiply-accumulate using Tensor Cores
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store result to global memory
    // Float accumulator needs to be converted to half when storing
    const int64_t cRow = warpM * WMMA_M;
    const int64_t cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N && cRow + WMMA_M <= M && cCol + WMMA_N <= N) {
        // Store float accumulator to shared memory first
        __shared__ float Cs_float[WMMA_M][WMMA_N];
        store_matrix_sync(&Cs_float[0][0], acc_frag, WMMA_N, mem_row_major);

        __syncthreads();

        // Convert float to half and store to global memory
        for (int i = threadIdx.y; i < WMMA_M; i += blockDim.y) {
            for (int j = threadIdx.x; j < WMMA_N; j += blockDim.x) {
                const int64_t row = cRow + i;
                const int64_t col = cCol + j;
                if (row < M && col < N) {
                    C_batch[row * N + col] = __float2half(Cs_float[i][j]);
                }
            }
        }
    }
}

// ============================================================================
// BFloat16 Tiled Matrix Multiplication Kernels
// ============================================================================

/**
 * BF16 tiled matmul kernel
 * Uses float accumulator for numerical stability, then converts back to BF16
 */
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul_tiled_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {

    // Shared memory for tiles
    __shared__ __nv_bfloat16 As[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 Bs[TILE_K][TILE_N];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column indices
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // Accumulator for the result (use float for numerical stability)
    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        if (tx < TILE_K) {
            int a_col = t * TILE_K + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A[row * lda + a_col];
            } else {
                As[ty][tx] = __float2bfloat16(0.0f);
            }
        }

        // Load tile of B into shared memory
        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            if (b_row < K && col < N) {
                Bs[ty][tx] = B[b_row * ldb + col];
            } else {
                Bs[ty][tx] = __float2bfloat16(0.0f);
            }
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product (accumulate in float)
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += __bfloat162float(As[ty][k]) * __bfloat162float(Bs[k][tx]);
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result (convert float back to bfloat16)
    if (row < M && col < N) {
        C[row * ldc + col] = __float2bfloat16(sum);
    }
}

/**
 * Batched BF16 tiled matmul kernel
 * Uses blockIdx.z for batch indexing
 */
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void batched_matmul_tiled_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {

    int batch = blockIdx.z;
    const __nv_bfloat16* A_batch = A + batch * stride_a;
    const __nv_bfloat16* B_batch = B + batch * stride_b;
    __nv_bfloat16* C_batch = C + batch * stride_c;

    __shared__ __nv_bfloat16 As[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 Bs[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        if (tx < TILE_K) {
            int a_col = t * TILE_K + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A_batch[row * lda + a_col];
            } else {
                As[ty][tx] = __float2bfloat16(0.0f);
            }
        }

        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            if (b_row < K && col < N) {
                Bs[ty][tx] = B_batch[b_row * ldb + col];
            } else {
                Bs[ty][tx] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += __bfloat162float(As[ty][k]) * __bfloat162float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_batch[row * ldc + col] = __float2bfloat16(sum);
    }
}

// ============================================================================
// cuBLAS Helper Functions
// ============================================================================

#ifdef TENZOR_HAS_CUBLAS

// Thread-safe cuBLAS handle management
static cublasHandle_t cublas_handle = nullptr;
static std::mutex cublas_mutex;

cublasHandle_t get_cublas_handle() {
    // Double-checked locking pattern for thread-safe initialization
    if (cublas_handle == nullptr) {
        std::lock_guard<std::mutex> lock(cublas_mutex);
        // Check again after acquiring lock
        if (cublas_handle == nullptr) {
            cublasStatus_t status = cublasCreate(&cublas_handle);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cuBLAS handle");
            }
            // Enable TF32 Tensor Core acceleration for FP32 operations
            // This provides significant speedup on Ampere+ GPUs (SM >= 8.0)
            // TF32 uses 19-bit precision which is sufficient for most ML workloads
            cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
        }
    }
    return cublas_handle;
}

// Use cuBLAS for large matrices with Tensor Core acceleration (TF32)
void matmul_cublas_f32(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K) {

    cublasHandle_t handle = get_cublas_handle();

    // cuBLAS uses column-major order, we use row-major
    // To compute C = A @ B in row-major, we compute C^T = B^T @ A^T
    // Which means: cublasGemm(..., B, A, C)

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use cublasGemmEx for TF32 Tensor Core acceleration on Ampere+ GPUs
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,        // B is not transposed
        CUBLAS_OP_N,        // A is not transposed
        N,                  // Rows of B^T (cols of B)
        M,                  // Cols of A^T (rows of A)
        K,                  // Cols of B^T = Rows of A^T
        &alpha,
        B, CUDA_R_32F, N,   // B matrix with leading dimension N
        A, CUDA_R_32F, K,   // A matrix with leading dimension K
        &beta,
        C, CUDA_R_32F, N,   // C matrix with leading dimension N
        CUBLAS_COMPUTE_32F_FAST_TF32,  // Use TF32 Tensor Cores
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS GemmEx failed with status: " + std::to_string(status));
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

// Batched cuBLAS matmul with Tensor Core acceleration (TF32)
void batched_matmul_cublas_f32(
    const float* A, const float* B, float* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {

    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use cublasGemmStridedBatchedEx for Tensor Core acceleration
    // This enables TF32 on Ampere+ GPUs for ~2x speedup over cublasSgemmStridedBatched
    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N,        // B not transposed (for row-major trick)
        CUBLAS_OP_N,        // A not transposed
        N, M, K,            // Dimensions (swapped for row-major)
        &alpha,
        B, CUDA_R_32F, N, stride_b,   // B matrix
        A, CUDA_R_32F, K, stride_a,   // A matrix
        &beta,
        C, CUDA_R_32F, N, stride_c,   // C matrix
        batch_size,
        CUBLAS_COMPUTE_32F_FAST_TF32, // Use TF32 Tensor Cores
        CUBLAS_GEMM_DEFAULT           // Let cuBLAS pick best algorithm
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS batched GemmEx failed with status: " + std::to_string(status));
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

// BFloat16 cuBLAS matmul using cublasGemmEx with CUDA_R_16BF
void matmul_cublas_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int64_t M, int64_t N, int64_t K) {

    cublasHandle_t handle = get_cublas_handle();

    // cuBLAS uses column-major order, we use row-major
    // To compute C = A @ B in row-major, we compute C^T = B^T @ A^T
    // Which means: cublasGemm(..., B, A, C)

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use cublasGemmEx with BF16 input/output and FP32 compute
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,        // B is not transposed
        CUBLAS_OP_N,        // A is not transposed
        N,                  // Rows of B^T (cols of B)
        M,                  // Cols of A^T (rows of A)
        K,                  // Cols of B^T = Rows of A^T
        &alpha,
        B, CUDA_R_16BF, N,  // B matrix with leading dimension N
        A, CUDA_R_16BF, K,  // A matrix with leading dimension K
        &beta,
        C, CUDA_R_16BF, N,  // C matrix with leading dimension N
        CUBLAS_COMPUTE_32F, // Compute in FP32 for numerical stability
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS GemmEx (BF16) failed with status: " + std::to_string(status));
    }
}

// Batched BFloat16 cuBLAS matmul
void batched_matmul_cublas_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    int64_t stride_a, int64_t stride_b, int64_t stride_c) {

    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use cublasGemmStridedBatchedEx with BF16 input/output and FP32 compute
    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N,        // B not transposed (for row-major trick)
        CUBLAS_OP_N,        // A not transposed
        N, M, K,            // Dimensions (swapped for row-major)
        &alpha,
        B, CUDA_R_16BF, N, stride_b,   // B matrix
        A, CUDA_R_16BF, K, stride_a,   // A matrix
        &beta,
        C, CUDA_R_16BF, N, stride_c,   // C matrix
        batch_size,
        CUBLAS_COMPUTE_32F,           // Compute in FP32 for numerical stability
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS batched GemmEx (BF16) failed with status: " + std::to_string(status));
    }
}

#endif // TENZOR_HAS_CUBLAS

// ============================================================================
// Host Functions - Single Matrix Multiplication
// ============================================================================

// Threshold for using cuBLAS vs custom kernels
// cuBLAS is faster than custom kernels even for small matrices,
// so we use a low threshold. Set to 0 to always use cuBLAS.
constexpr int64_t CUBLAS_THRESHOLD = 32;

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

/**
 * FP16 matrix multiplication with Tensor Core acceleration
 * Automatically selects between Tensor Core and fallback kernel based on dimensions
 */
void matmul_f16(
    const __half* A, const __half* B, __half* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    // Check if dimensions are multiples of 16 (Tensor Core requirement)
    const bool use_tensor_cores = (M % WMMA_M == 0) && (N % WMMA_N == 0) && (K % WMMA_K == 0);

    if (use_tensor_cores) {
        // Use Tensor Cores for optimal performance
        // 1 warp (32 threads) per block, each warp computes one 16x16 tile
        dim3 block(32, 1);  // 1 warp per block (32 threads)
        dim3 grid((N + WMMA_N - 1) / WMMA_N,
                  (M + WMMA_M - 1) / WMMA_M);

        matmul_tensor_core_f16_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    } else {
        // Fall back to standard tiled kernel for non-aligned dimensions
        // Use smaller tile size (16x16 block) to reduce resource usage
        dim3 block(TILE_SIZE_F16, TILE_SIZE_F16);
        dim3 grid((N + TILE_SIZE_F16 - 1) / TILE_SIZE_F16,
                  (M + TILE_SIZE_F16 - 1) / TILE_SIZE_F16);

        matmul_tiled_f16_kernel<TILE_SIZE_F16, TILE_SIZE_F16, TILE_SIZE_K>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
    }

    CUDA_CHECK(cudaGetLastError());
}

/**
 * BF16 matrix multiplication with cuBLAS acceleration
 * Uses FP32 compute internally for numerical stability
 */
void matmul_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

#ifdef TENZOR_HAS_CUBLAS
    // Use cuBLAS for large matrices (BF16 benefits from Tensor Core acceleration)
    if (M >= CUBLAS_THRESHOLD || N >= CUBLAS_THRESHOLD || K >= CUBLAS_THRESHOLD) {
        matmul_cublas_bf16(A, B, C, M, N, K);
        return;
    }
#endif

    // Use custom tiled kernel for smaller matrices
    // Use smaller tile size (16x16 block) similar to FP16
    dim3 block(TILE_SIZE_F16, TILE_SIZE_F16);
    dim3 grid((N + TILE_SIZE_F16 - 1) / TILE_SIZE_F16,
              (M + TILE_SIZE_F16 - 1) / TILE_SIZE_F16);

    matmul_tiled_bf16_kernel<TILE_SIZE_F16, TILE_SIZE_F16, TILE_SIZE_K>
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
    // Always use cuBLAS for batched matmul - cublasSgemmStridedBatched is
    // highly optimized and amortizes kernel launch overhead across the batch
    batched_matmul_cublas_f32(A, B, C, batch_size, M, N, K,
                               stride_a, stride_b, stride_c);
#else
    // Fallback to custom batched kernel when cuBLAS is not available
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    batched_matmul_tiled_f32_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
        <<<grid, block, 0, stream>>>(
            A, B, C, batch_size, M, N, K,
            stride_a, stride_b, stride_c);

    CUDA_CHECK(cudaGetLastError());
#endif
}

void batched_matmul_f64(
    const double* A, const double* B, double* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    int64_t stride_a = M * K;
    int64_t stride_b = K * N;
    int64_t stride_c = M * N;

#ifdef TENZOR_HAS_CUBLAS
    // Always use cuBLAS for batched matmul - cublasDgemmStridedBatched is
    // highly optimized and amortizes kernel launch overhead across the batch
    batched_matmul_cublas_f64(A, B, C, batch_size, M, N, K,
                               stride_a, stride_b, stride_c);
#else
    // Fallback to custom batched kernel when cuBLAS is not available
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    batched_matmul_tiled_f64_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
        <<<grid, block, 0, stream>>>(
            A, B, C, batch_size, M, N, K,
            stride_a, stride_b, stride_c);

    CUDA_CHECK(cudaGetLastError());
#endif
}

/**
 * Batched FP16 matrix multiplication with Tensor Core acceleration
 */
void batched_matmul_f16(
    const __half* A, const __half* B, __half* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    int64_t stride_a = M * K;
    int64_t stride_b = K * N;
    int64_t stride_c = M * N;

    // Check if dimensions are multiples of 16 (Tensor Core requirement)
    const bool use_tensor_cores = (M % WMMA_M == 0) && (N % WMMA_N == 0) && (K % WMMA_K == 0);

    if (use_tensor_cores) {
        // Use Tensor Cores for optimal performance
        // 1 warp (32 threads) per block, each warp computes one 16x16 tile
        dim3 block(32, 1);  // 1 warp per block (32 threads)
        dim3 grid((N + WMMA_N - 1) / WMMA_N,
                  (M + WMMA_M - 1) / WMMA_M,
                  batch_size);

        batched_matmul_tensor_core_f16_kernel<<<grid, block, 0, stream>>>(
            A, B, C, batch_size, M, N, K,
            stride_a, stride_b, stride_c);
    } else {
        // Batched tiled F16 kernel using blockIdx.z for batch indexing
        dim3 block(TILE_SIZE_F16, TILE_SIZE_F16);
        dim3 grid((N + TILE_SIZE_F16 - 1) / TILE_SIZE_F16,
                  (M + TILE_SIZE_F16 - 1) / TILE_SIZE_F16,
                  batch_size);

        batched_matmul_tiled_f16_kernel<TILE_SIZE_F16, TILE_SIZE_F16, TILE_SIZE_F16>
            <<<grid, block, 0, stream>>>(
                A, B, C,
                M, N, K,
                K, N, N,  // lda, ldb, ldc for row-major
                stride_a, stride_b, stride_c);
    }

    CUDA_CHECK(cudaGetLastError());
}

/**
 * Batched BF16 matrix multiplication with cuBLAS acceleration
 * Uses FP32 compute internally for numerical stability
 */
void batched_matmul_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    int64_t stride_a = M * K;
    int64_t stride_b = K * N;
    int64_t stride_c = M * N;

#ifdef TENZOR_HAS_CUBLAS
    // Always use cuBLAS for batched matmul - BF16 benefits from Tensor Core acceleration
    batched_matmul_cublas_bf16(A, B, C, batch_size, M, N, K,
                                stride_a, stride_b, stride_c);
#else
    // Fallback to custom batched kernel when cuBLAS is not available
    dim3 block(TILE_SIZE_F16, TILE_SIZE_F16);
    dim3 grid((N + TILE_SIZE_F16 - 1) / TILE_SIZE_F16,
              (M + TILE_SIZE_F16 - 1) / TILE_SIZE_F16,
              batch_size);

    batched_matmul_tiled_bf16_kernel<TILE_SIZE_F16, TILE_SIZE_F16, TILE_SIZE_F16>
        <<<grid, block, 0, stream>>>(
            A, B, C,
            M, N, K,
            K, N, N,  // lda, ldb, ldc for row-major
            stride_a, stride_b, stride_c);

    CUDA_CHECK(cudaGetLastError());
#endif
}

// ============================================================================
// Public API - Matrix Multiplication
// ============================================================================

auto matmul_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    // Make tensors contiguous if needed (does not break autograd chain)
    Tensor a_contig = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contig = b.is_contiguous() ? b : b.contiguous();

    if (a_contig.device().type != Device::Type::CUDA || b_contig.device().type != Device::Type::CUDA) {
        throw std::runtime_error("matmul_kernel requires CUDA tensors");
    }

    // Handle 1D vector × 2D matrix (vector-matrix multiplication)
    if (a_contig.ndim() == 1 && b_contig.ndim() == 2) {
        auto a_shape = a_contig.shape();
        auto b_shape = b_contig.shape();

        int64_t N = a_shape[0];  // Vector size
        int64_t N2 = b_shape[0]; // Matrix rows
        int64_t K = b_shape[1];  // Matrix cols

        if (N != N2) {
            throw std::runtime_error(
                "matmul dimension mismatch: vector(" + std::to_string(N) +
                ") @ matrix(" + std::to_string(N2) + "×" + std::to_string(K) + ")"
            );
        }

        // Treat 1D vector as row vector (1, N) and perform matmul to get (1, K), then squeeze to (K,)
        int64_t M = 1;

        // Create output tensor with 1D shape
        Tensor result({K}, a_contig.dtype(), a_contig.device());

        // Dispatch based on dtype
        if (a_contig.dtype() == DType::Float32 && b_contig.dtype() == DType::Float32) {
            const float* a_data = a_contig.data<float>();
            const float* b_data = b_contig.data<float>();
            float* c_data = result.data<float>();

            matmul_f32(a_data, b_data, c_data, M, K, N, stream);

        } else if (a_contig.dtype() == DType::Float64 && b_contig.dtype() == DType::Float64) {
            const double* a_data = a_contig.data<double>();
            const double* b_data = b_contig.data<double>();
            double* c_data = result.data<double>();

            matmul_f64(a_data, b_data, c_data, M, K, N, stream);

        } else if (a_contig.dtype() == DType::Float16 && b_contig.dtype() == DType::Float16) {
            const Float16* a_data = a_contig.data<Float16>();
            const Float16* b_data = b_contig.data<Float16>();
            Float16* c_data = result.data<Float16>();

            const __half* a_half = reinterpret_cast<const __half*>(a_data);
            const __half* b_half = reinterpret_cast<const __half*>(b_data);
            __half* c_half = reinterpret_cast<__half*>(c_data);

            matmul_f16(a_half, b_half, c_half, M, K, N, stream);

        } else if (a_contig.dtype() == DType::Int32 && b_contig.dtype() == DType::Int32) {
            const int32_t* a_data = a_contig.data<int32_t>();
            const int32_t* b_data = b_contig.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();

            matmul_i32(a_data, b_data, c_data, M, K, N, stream);

        } else if (a_contig.dtype() == DType::BFloat16 && b_contig.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a_contig.data<BFloat16>();
            const BFloat16* b_data = b_contig.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();

            const __nv_bfloat16* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a_data);
            const __nv_bfloat16* b_bf16 = reinterpret_cast<const __nv_bfloat16*>(b_data);
            __nv_bfloat16* c_bf16 = reinterpret_cast<__nv_bfloat16*>(c_data);

            matmul_bf16(a_bf16, b_bf16, c_bf16, M, K, N, stream);

        } else {
            throw std::runtime_error(
                "matmul unsupported dtype combination: " +
                std::string(dtype_name(a_contig.dtype())) + " @ " +
                std::string(dtype_name(b_contig.dtype()))
            );
        }

        // Note: No sync here - cuBLAS calls are async and sync happens when data is accessed
        return result;
    }

    // Handle 2D matrices
    if (a_contig.ndim() == 2 && b_contig.ndim() == 2) {
        auto a_shape = a_contig.shape();
        auto b_shape = b_contig.shape();

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
        Tensor result({M, N}, a_contig.dtype(), a_contig.device());

        // Dispatch based on dtype
        if (a_contig.dtype() == DType::Float32 && b_contig.dtype() == DType::Float32) {
            const float* a_data = a_contig.data<float>();
            const float* b_data = b_contig.data<float>();
            float* c_data = result.data<float>();

            matmul_f32(a_data, b_data, c_data, M, N, K, stream);

        } else if (a_contig.dtype() == DType::Float64 && b_contig.dtype() == DType::Float64) {
            const double* a_data = a_contig.data<double>();
            const double* b_data = b_contig.data<double>();
            double* c_data = result.data<double>();

            matmul_f64(a_data, b_data, c_data, M, N, K, stream);

        } else if (a_contig.dtype() == DType::Float16 && b_contig.dtype() == DType::Float16) {
            const Float16* a_data = a_contig.data<Float16>();
            const Float16* b_data = b_contig.data<Float16>();
            Float16* c_data = result.data<Float16>();

            // Convert to CUDA __half pointers
            const __half* a_half = reinterpret_cast<const __half*>(a_data);
            const __half* b_half = reinterpret_cast<const __half*>(b_data);
            __half* c_half = reinterpret_cast<__half*>(c_data);

            matmul_f16(a_half, b_half, c_half, M, N, K, stream);

        } else if (a_contig.dtype() == DType::Int32 && b_contig.dtype() == DType::Int32) {
            const int32_t* a_data = a_contig.data<int32_t>();
            const int32_t* b_data = b_contig.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();

            matmul_i32(a_data, b_data, c_data, M, N, K, stream);

        } else if (a_contig.dtype() == DType::BFloat16 && b_contig.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a_contig.data<BFloat16>();
            const BFloat16* b_data = b_contig.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();

            // Convert to CUDA __nv_bfloat16 pointers
            const __nv_bfloat16* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a_data);
            const __nv_bfloat16* b_bf16 = reinterpret_cast<const __nv_bfloat16*>(b_data);
            __nv_bfloat16* c_bf16 = reinterpret_cast<__nv_bfloat16*>(c_data);

            matmul_bf16(a_bf16, b_bf16, c_bf16, M, N, K, stream);

        } else {
            throw std::runtime_error(
                "matmul unsupported dtype combination: " +
                std::string(dtype_name(a_contig.dtype())) + " @ " +
                std::string(dtype_name(b_contig.dtype()))
            );
        }

        // Note: No sync here - cuBLAS calls are async and sync happens when data is accessed
        return result;
    }

    // Handle batched 3D matrices (batch_size, M, N) @ (batch_size, N, K)
    if (a_contig.ndim() == 3 && b_contig.ndim() == 3) {
        auto a_shape = a_contig.shape();
        auto b_shape = b_contig.shape();

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
        Tensor result({batch_size, M, N}, a_contig.dtype(), a_contig.device());

        // Dispatch based on dtype
        if (a_contig.dtype() == DType::Float32 && b_contig.dtype() == DType::Float32) {
            const float* a_data = a_contig.data<float>();
            const float* b_data = b_contig.data<float>();
            float* c_data = result.data<float>();

            batched_matmul_f32(a_data, b_data, c_data, batch_size, M, N, K, stream);

        } else if (a_contig.dtype() == DType::Float64 && b_contig.dtype() == DType::Float64) {
            const double* a_data = a_contig.data<double>();
            const double* b_data = b_contig.data<double>();
            double* c_data = result.data<double>();

            batched_matmul_f64(a_data, b_data, c_data, batch_size, M, N, K, stream);

        } else if (a_contig.dtype() == DType::Float16 && b_contig.dtype() == DType::Float16) {
            const Float16* a_data = a_contig.data<Float16>();
            const Float16* b_data = b_contig.data<Float16>();
            Float16* c_data = result.data<Float16>();

            // Convert to CUDA __half pointers
            const __half* a_half = reinterpret_cast<const __half*>(a_data);
            const __half* b_half = reinterpret_cast<const __half*>(b_data);
            __half* c_half = reinterpret_cast<__half*>(c_data);

            batched_matmul_f16(a_half, b_half, c_half, batch_size, M, N, K, stream);

        } else if (a_contig.dtype() == DType::BFloat16 && b_contig.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a_contig.data<BFloat16>();
            const BFloat16* b_data = b_contig.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();

            // Convert to CUDA __nv_bfloat16 pointers
            const __nv_bfloat16* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a_data);
            const __nv_bfloat16* b_bf16 = reinterpret_cast<const __nv_bfloat16*>(b_data);
            __nv_bfloat16* c_bf16 = reinterpret_cast<__nv_bfloat16*>(c_data);

            batched_matmul_bf16(a_bf16, b_bf16, c_bf16, batch_size, M, N, K, stream);

        } else {
            throw std::runtime_error(
                "Batched matmul unsupported dtype: " +
                std::string(dtype_name(a_contig.dtype()))
            );
        }

        return result;
    }

    throw std::runtime_error(
        "matmul requires 2D or 3D tensors, got " +
        std::to_string(a_contig.ndim()) + "D and " + std::to_string(b_contig.ndim()) + "D"
    );
}

} // namespace cuda
} // namespace tenzor
