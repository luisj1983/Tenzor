#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "cuda_common.cuh"
#include "cuda_launch_utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>        // For __half
#include <cuda_bf16.h>        // For __nv_bfloat16
#include <mma.h>              // For Tensor Cores (WMMA)
#include <stdexcept>
#include <string>

#ifdef TENZOR_HAS_CUBLAS
#include <cublas_v2.h>
#include "../cublas_handle_pool.hpp"
#endif

#include <random>

namespace tenzor {
namespace cuda {

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
// Architecture-Aware Tile Configuration
// ============================================================================

struct TileConfig {
    int tile_m, tile_n, tile_k;
};

// Get architecture-appropriate tile config based on GPU compute capability.
// Cached per-device to avoid repeated queries.
inline TileConfig get_tile_config(int device_id = 0) {
    // Cache per device (max 16 GPUs)
    static TileConfig configs[16] = {};
    static bool initialized[16] = {};

    if (device_id < 0 || device_id >= 16) device_id = 0;
    if (initialized[device_id]) return configs[device_id];

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
    int sm = major * 10 + minor;

    if (sm < 70) {
        configs[device_id] = {16, 16, 16};       // Pascal: smaller tiles
    } else if (sm < 80) {
        configs[device_id] = {32, 32, 16};        // Volta/Turing: current default
    } else {
        configs[device_id] = {32, 32, 32};        // Ampere+: more shared mem, bigger K tile
    }
    initialized[device_id] = true;
    return configs[device_id];
}

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
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

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
    __shared__ double As[TILE_M][TILE_K + 1];
    __shared__ double Bs[TILE_K][TILE_N + 1];

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
    __shared__ int32_t As[TILE_M][TILE_K + 1];
    __shared__ int32_t Bs[TILE_K][TILE_N + 1];

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
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

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
    __shared__ double As[TILE_M][TILE_K + 1];
    __shared__ double Bs[TILE_K][TILE_N + 1];

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

    // Multi-warp kernel: 4 warps per block, each handles one 16x16 tile
    // Block covers a 2x2 grid of WMMA tiles for better occupancy
    const int warpId = threadIdx.y;  // 0..3
    const int localM = warpId / 2;   // 0 or 1 (row within block's 2x2 tile grid)
    const int localN = warpId % 2;   // 0 or 1 (col within block's 2x2 tile grid)

    // Global tile position for this warp
    const int warpM = blockIdx.y * 2 + localM;
    const int warpN = blockIdx.x * 2 + localN;

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
            // Each warp uses its own shared memory slice to avoid conflicts
            __shared__ __half As[4][WMMA_M][WMMA_K + 1];
            __shared__ __half Bs[4][WMMA_K][WMMA_N + 1];

            const int64_t aRow = warpM * WMMA_M;
            const int64_t aCol = k;
            const int64_t bRow = k;
            const int64_t bCol = warpN * WMMA_N;

            // Load with bounds checking (each thread in warp cooperates)
            for (int i = 0; i < WMMA_M; ++i) {
                for (int j = threadIdx.x; j < WMMA_K; j += 32) {
                    const int64_t row = aRow + i;
                    const int64_t col = aCol + j;
                    As[warpId][i][j] = (row < M && col < K) ? A[row * K + col] : __float2half(0.0f);
                }
            }

            for (int i = 0; i < WMMA_K; ++i) {
                for (int j = threadIdx.x; j < WMMA_N; j += 32) {
                    const int64_t row = bRow + i;
                    const int64_t col = bCol + j;
                    Bs[warpId][i][j] = (row < K && col < N) ? B[row * N + col] : __float2half(0.0f);
                }
            }

            __syncwarp();

            // Load from shared memory
            load_matrix_sync(a_frag, &As[warpId][0][0], WMMA_K + 1);
            load_matrix_sync(b_frag, &Bs[warpId][0][0], WMMA_N + 1);

            // Perform matrix multiply-accumulate
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store result to global memory
    // Each warp stores its own tile independently
    const int64_t cRow = warpM * WMMA_M;
    const int64_t cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        // Store float accumulator to per-warp shared memory, then convert to half
        __shared__ float Cs_float[4][WMMA_M][WMMA_N + 1];
        store_matrix_sync(&Cs_float[warpId][0][0], acc_frag, WMMA_N + 1, mem_row_major);

        __syncwarp();

        // Convert float to half and store to global memory with bounds checking
        for (int i = 0; i < WMMA_M; ++i) {
            for (int j = threadIdx.x; j < WMMA_N; j += 32) {
                const int64_t row = cRow + i;
                const int64_t col = cCol + j;
                if (row < M && col < N) {
                    C[row * N + col] = __float2half(Cs_float[warpId][i][j]);
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
    __shared__ __half As[TILE_M][TILE_K + 1];
    __shared__ __half Bs[TILE_K][TILE_N + 1];

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

    __shared__ __half As[TILE_M][TILE_K + 1];
    __shared__ __half Bs[TILE_K][TILE_N + 1];

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
        } else {
            // Handle edge case where K is not a multiple of WMMA_K
            __shared__ __half As_residual[WMMA_M][WMMA_K + 1];
            __shared__ __half Bs_residual[WMMA_K][WMMA_N + 1];

            const int64_t aRow = warpM * WMMA_M;
            const int64_t aCol = k;
            const int64_t bRow = k;
            const int64_t bCol = warpN * WMMA_N;

            // Load with bounds checking
            for (int i = 0; i < WMMA_M; ++i) {
                for (int j = threadIdx.x; j < WMMA_K; j += 32) {
                    const int64_t row = aRow + i;
                    const int64_t col = aCol + j;
                    As_residual[i][j] = (row < M && col < K) ? A_batch[row * K + col] : __float2half(0.0f);
                }
            }

            for (int i = 0; i < WMMA_K; ++i) {
                for (int j = threadIdx.x; j < WMMA_N; j += 32) {
                    const int64_t row = bRow + i;
                    const int64_t col = bCol + j;
                    Bs_residual[i][j] = (row < K && col < N) ? B_batch[row * N + col] : __float2half(0.0f);
                }
            }

            __syncthreads();

            // Load from shared memory
            load_matrix_sync(a_frag, &As_residual[0][0], WMMA_K + 1);
            load_matrix_sync(b_frag, &Bs_residual[0][0], WMMA_N + 1);

            // Perform matrix multiply-accumulate
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store result to global memory
    // Float accumulator needs to be converted to half when storing
    const int64_t cRow = warpM * WMMA_M;
    const int64_t cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N && cRow + WMMA_M <= M && cCol + WMMA_N <= N) {
        // Store float accumulator to shared memory first
        __shared__ float Cs_float[WMMA_M][WMMA_N + 1];
        store_matrix_sync(&Cs_float[0][0], acc_frag, WMMA_N + 1, mem_row_major);

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
    __shared__ __nv_bfloat16 As[TILE_M][TILE_K + 1];
    __shared__ __nv_bfloat16 Bs[TILE_K][TILE_N + 1];

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

    __shared__ __nv_bfloat16 As[TILE_M][TILE_K + 1];
    __shared__ __nv_bfloat16 Bs[TILE_K][TILE_N + 1];

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


// Use cuBLAS for large matrices with Tensor Core acceleration (TF32)
void matmul_cublas_f32(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    cublasHandle_t handle = CuBLASHandlePool::get(stream);

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
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    cublasHandle_t handle = CuBLASHandlePool::get(stream);

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
    int64_t stride_a, int64_t stride_b, int64_t stride_c,
    cudaStream_t stream = 0) {

    cublasHandle_t handle = CuBLASHandlePool::get(stream);

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
    int64_t stride_a, int64_t stride_b, int64_t stride_c,
    cudaStream_t stream = 0) {

    cublasHandle_t handle = CuBLASHandlePool::get(stream);

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
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    cublasHandle_t handle = CuBLASHandlePool::get(stream);

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
    int64_t stride_a, int64_t stride_b, int64_t stride_c,
    cudaStream_t stream = 0) {

    cublasHandle_t handle = CuBLASHandlePool::get(stream);

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

// Float16 cuBLAS matmul using cublasGemmEx with CUDA_R_16F
void matmul_cublas_f16(
    const __half* A, const __half* B, __half* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    cublasHandle_t handle = CuBLASHandlePool::get(stream);

    // cuBLAS uses column-major order, we use row-major
    // To compute C = A @ B in row-major, we compute C^T = B^T @ A^T
    // Which means: cublasGemm(..., B, A, C)

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use cublasGemmEx with FP16 input/output and FP32 compute for precision
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,        // B is not transposed
        CUBLAS_OP_N,        // A is not transposed
        N,                  // Rows of B^T (cols of B)
        M,                  // Cols of A^T (rows of A)
        K,                  // Cols of B^T = Rows of A^T
        &alpha,
        B, CUDA_R_16F, N,   // B matrix with leading dimension N
        A, CUDA_R_16F, K,   // A matrix with leading dimension K
        &beta,
        C, CUDA_R_16F, N,   // C matrix with leading dimension N
        CUBLAS_COMPUTE_32F, // Compute in FP32 for numerical stability
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS GemmEx (FP16) failed with status: " + std::to_string(status));
    }
}

// Batched Float16 cuBLAS matmul
void batched_matmul_cublas_f16(
    const __half* A, const __half* B, __half* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    int64_t stride_a, int64_t stride_b, int64_t stride_c,
    cudaStream_t stream = 0) {

    cublasHandle_t handle = CuBLASHandlePool::get(stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use cublasGemmStridedBatchedEx with FP16 input/output and FP32 compute
    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N,        // B not transposed (for row-major trick)
        CUBLAS_OP_N,        // A not transposed
        N, M, K,            // Dimensions (swapped for row-major)
        &alpha,
        B, CUDA_R_16F, N, stride_b,   // B matrix
        A, CUDA_R_16F, K, stride_a,   // A matrix
        &beta,
        C, CUDA_R_16F, N, stride_c,   // C matrix
        batch_size,
        CUBLAS_COMPUTE_32F,           // Compute in FP32 for numerical stability
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS batched GemmEx (FP16) failed with status: " + std::to_string(status));
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
        matmul_cublas_f32(A, B, C, M, N, K, stream);
        return;
    }
#endif

    // Use custom tiled kernel for smaller matrices
    // Select architecture-appropriate tile sizes
    auto tc = get_tile_config();
    if (tc.tile_m == 16) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        matmul_tiled_f32_kernel<16, 16, 16>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
    } else if (tc.tile_k == 32) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_tiled_f32_kernel<32, 32, 32>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
    } else {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_tiled_f32_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

void matmul_f64(
    const double* A, const double* B, double* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

#ifdef TENZOR_HAS_CUBLAS
    // Use cuBLAS for large matrices
    if (M >= CUBLAS_THRESHOLD || N >= CUBLAS_THRESHOLD || K >= CUBLAS_THRESHOLD) {
        matmul_cublas_f64(A, B, C, M, N, K, stream);
        return;
    }
#endif

    // Use custom tiled kernel for smaller matrices
    // Select architecture-appropriate tile sizes
    auto tc = get_tile_config();
    if (tc.tile_m == 16) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        matmul_tiled_f64_kernel<16, 16, 16>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
    } else if (tc.tile_k == 32) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_tiled_f64_kernel<32, 32, 32>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
    } else {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_tiled_f64_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

void matmul_i32(
    const int32_t* A, const int32_t* B, int32_t* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    // Custom kernel (no cuBLAS for int32)
    // Select architecture-appropriate tile sizes
    auto tc = get_tile_config();
    if (tc.tile_m == 16) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        matmul_tiled_i32_kernel<16, 16, 16>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
    } else if (tc.tile_k == 32) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_tiled_i32_kernel<32, 32, 32>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
    } else {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_tiled_i32_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

// Helper: round up to next multiple of 16
inline int64_t round_up_16(int64_t x) {
    return (x + 15) & ~int64_t(15);
}

// Helper: check whether padding overhead is acceptable (< 50% extra compute)
inline bool padding_overhead_ok(int64_t M, int64_t N, int64_t K) {
    int64_t Mp = round_up_16(M);
    int64_t Np = round_up_16(N);
    int64_t Kp = round_up_16(K);
    // Padded FLOPs vs original FLOPs
    double original = static_cast<double>(M) * N * K;
    double padded   = static_cast<double>(Mp) * Np * Kp;
    return (padded < original * 1.5);
}

/**
 * FP16 matrix multiplication with cuBLAS and Tensor Core acceleration
 * Uses cuBLAS with FP32 accumulation for large matrices, falls back to
 * WMMA Tensor Core or tiled kernels for smaller/unaligned dimensions.
 * When dimensions are >= 16 but not aligned, pads to multiples of 16 to
 * enable Tensor Core acceleration (if padding overhead < 50%).
 */
void matmul_f16(
    const __half* A, const __half* B, __half* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

#ifdef TENZOR_HAS_CUBLAS
    // Use cuBLAS for large matrices (FP16 benefits from Tensor Core acceleration)
    if (M >= CUBLAS_THRESHOLD || N >= CUBLAS_THRESHOLD || K >= CUBLAS_THRESHOLD) {
        matmul_cublas_f16(A, B, C, M, N, K, stream);
        return;
    }
#endif

    // Check if dimensions are multiples of 16 (Tensor Core requirement)
    const bool use_tensor_cores = (M % WMMA_M == 0) && (N % WMMA_N == 0) && (K % WMMA_K == 0);

    if (use_tensor_cores) {
        // Use Tensor Cores for optimal performance
        // 4 warps (128 threads) per block for better occupancy (~100% vs ~25%)
        // Each warp computes one 16x16 tile; block handles 2x2 = 4 tiles
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int BLOCK_TILES_M = 2;  // 2 tiles vertically
        constexpr int BLOCK_TILES_N = 2;  // 2 tiles horizontally
        dim3 block(32, WARPS_PER_BLOCK);  // 4 warps per block (128 threads)
        dim3 grid((N + WMMA_N * BLOCK_TILES_N - 1) / (WMMA_N * BLOCK_TILES_N),
                  (M + WMMA_M * BLOCK_TILES_M - 1) / (WMMA_M * BLOCK_TILES_M));

        matmul_tensor_core_f16_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (M >= WMMA_M && N >= WMMA_N && K >= WMMA_K && padding_overhead_ok(M, N, K)) {
        // Dimensions are large enough for Tensor Cores but not aligned.
        // Pad to multiples of 16 to enable TC acceleration when overhead is acceptable.
        int64_t Mp = round_up_16(M);
        int64_t Np = round_up_16(N);
        int64_t Kp = round_up_16(K);

        // Allocate zero-initialized padded buffers (RAII ensures cleanup on exception)
        CudaAsyncBuffer A_buf(Mp * Kp * sizeof(__half), stream);
        CudaAsyncBuffer B_buf(Kp * Np * sizeof(__half), stream);
        CudaAsyncBuffer C_buf(Mp * Np * sizeof(__half), stream);
        auto* A_pad = A_buf.as<__half>();
        auto* B_pad = B_buf.as<__half>();
        auto* C_pad = C_buf.as<__half>();
        TENZOR_CUDA_CHECK(cudaMemsetAsync(A_pad, 0, Mp * Kp * sizeof(__half), stream));
        TENZOR_CUDA_CHECK(cudaMemsetAsync(B_pad, 0, Kp * Np * sizeof(__half), stream));

        // Copy A (M x K) into A_pad (Mp x Kp) with proper stride
        TENZOR_CUDA_CHECK(cudaMemcpy2DAsync(
            A_pad, Kp * sizeof(__half),        // dst, dst pitch
            A,     K  * sizeof(__half),         // src, src pitch
            K  * sizeof(__half),                // width in bytes to copy per row
            M,                                  // number of rows
            cudaMemcpyDeviceToDevice, stream));

        // Copy B (K x N) into B_pad (Kp x Np) with proper stride
        TENZOR_CUDA_CHECK(cudaMemcpy2DAsync(
            B_pad, Np * sizeof(__half),         // dst, dst pitch
            B,     N  * sizeof(__half),          // src, src pitch
            N  * sizeof(__half),                 // width in bytes to copy per row
            K,                                   // number of rows
            cudaMemcpyDeviceToDevice, stream));

        // Run Tensor Core kernel on padded matrices
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int BLOCK_TILES_M = 2;
        constexpr int BLOCK_TILES_N = 2;
        dim3 block(32, WARPS_PER_BLOCK);
        dim3 grid((Np + WMMA_N * BLOCK_TILES_N - 1) / (WMMA_N * BLOCK_TILES_N),
                  (Mp + WMMA_M * BLOCK_TILES_M - 1) / (WMMA_M * BLOCK_TILES_M));

        matmul_tensor_core_f16_kernel<<<grid, block, 0, stream>>>(
            A_pad, B_pad, C_pad, Mp, Np, Kp);
        TENZOR_CUDA_POST_LAUNCH_CHECK();

        // Copy result C_pad (Mp x Np) back to C (M x N) — extract top-left M x N
        TENZOR_CUDA_CHECK(cudaMemcpy2DAsync(
            C,     N  * sizeof(__half),          // dst, dst pitch
            C_pad, Np * sizeof(__half),          // src, src pitch
            N  * sizeof(__half),                 // width in bytes to copy per row
            M,                                   // number of rows
            cudaMemcpyDeviceToDevice, stream));
        // CudaAsyncBuffer RAII handles cleanup automatically
    } else {
        // Fall back to standard tiled kernel for non-aligned dimensions
        // (either too small for TC or padding overhead too high)
        dim3 block(TILE_SIZE_F16, TILE_SIZE_F16);
        dim3 grid((N + TILE_SIZE_F16 - 1) / TILE_SIZE_F16,
                  (M + TILE_SIZE_F16 - 1) / TILE_SIZE_F16);

        matmul_tiled_f16_kernel<TILE_SIZE_F16, TILE_SIZE_F16, TILE_SIZE_K>
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K, K, N, N);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
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
        matmul_cublas_bf16(A, B, C, M, N, K, stream);
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

    TENZOR_CUDA_POST_LAUNCH_CHECK();
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
                               stride_a, stride_b, stride_c, stream);
#else
    // Fallback to custom batched kernel when cuBLAS is not available
    // Select architecture-appropriate tile sizes
    auto tc = get_tile_config();
    if (tc.tile_m == 16) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16, batch_size);
        batched_matmul_tiled_f32_kernel<16, 16, 16>
            <<<grid, block, 0, stream>>>(
                A, B, C, batch_size, M, N, K,
                stride_a, stride_b, stride_c);
    } else if (tc.tile_k == 32) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32, batch_size);
        batched_matmul_tiled_f32_kernel<32, 32, 32>
            <<<grid, block, 0, stream>>>(
                A, B, C, batch_size, M, N, K,
                stride_a, stride_b, stride_c);
    } else {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE,
                  batch_size);
        batched_matmul_tiled_f32_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
            <<<grid, block, 0, stream>>>(
                A, B, C, batch_size, M, N, K,
                stride_a, stride_b, stride_c);
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
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
                               stride_a, stride_b, stride_c, stream);
#else
    // Fallback to custom batched kernel when cuBLAS is not available
    // Select architecture-appropriate tile sizes
    auto tc = get_tile_config();
    if (tc.tile_m == 16) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16, batch_size);
        batched_matmul_tiled_f64_kernel<16, 16, 16>
            <<<grid, block, 0, stream>>>(
                A, B, C, batch_size, M, N, K,
                stride_a, stride_b, stride_c);
    } else if (tc.tile_k == 32) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32, batch_size);
        batched_matmul_tiled_f64_kernel<32, 32, 32>
            <<<grid, block, 0, stream>>>(
                A, B, C, batch_size, M, N, K,
                stride_a, stride_b, stride_c);
    } else {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE,
                  batch_size);
        batched_matmul_tiled_f64_kernel<TILE_SIZE, TILE_SIZE, TILE_SIZE_K>
            <<<grid, block, 0, stream>>>(
                A, B, C, batch_size, M, N, K,
                stride_a, stride_b, stride_c);
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
#endif
}

/**
 * Batched FP16 matrix multiplication with cuBLAS and Tensor Core acceleration
 * Uses cuBLAS with FP32 accumulation for large matrices, falls back to
 * WMMA Tensor Core or tiled kernels for smaller/unaligned dimensions.
 * When dimensions are >= 16 but not aligned, pads to multiples of 16 to
 * enable Tensor Core acceleration (if padding overhead < 50%).
 */
void batched_matmul_f16(
    const __half* A, const __half* B, __half* C,
    int64_t batch_size, int64_t M, int64_t N, int64_t K,
    cudaStream_t stream = 0) {

    int64_t stride_a = M * K;
    int64_t stride_b = K * N;
    int64_t stride_c = M * N;

#ifdef TENZOR_HAS_CUBLAS
    // Use cuBLAS for large matrices (FP16 benefits from Tensor Core acceleration)
    if (M >= CUBLAS_THRESHOLD || N >= CUBLAS_THRESHOLD || K >= CUBLAS_THRESHOLD) {
        batched_matmul_cublas_f16(A, B, C, batch_size, M, N, K,
                                   stride_a, stride_b, stride_c, stream);
        return;
    }
#endif

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
            TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (M >= WMMA_M && N >= WMMA_N && K >= WMMA_K && padding_overhead_ok(M, N, K)) {
        // Dimensions are large enough for Tensor Cores but not aligned.
        // Pad to multiples of 16 to enable TC acceleration when overhead is acceptable.
        int64_t Mp = round_up_16(M);
        int64_t Np = round_up_16(N);
        int64_t Kp = round_up_16(K);

        int64_t stride_a_pad = Mp * Kp;
        int64_t stride_b_pad = Kp * Np;
        int64_t stride_c_pad = Mp * Np;

        // Allocate zero-initialized padded buffers for the entire batch (RAII)
        CudaAsyncBuffer A_buf(batch_size * stride_a_pad * sizeof(__half), stream);
        CudaAsyncBuffer B_buf(batch_size * stride_b_pad * sizeof(__half), stream);
        CudaAsyncBuffer C_buf(batch_size * stride_c_pad * sizeof(__half), stream);
        auto* A_pad = A_buf.as<__half>();
        auto* B_pad = B_buf.as<__half>();
        auto* C_pad = C_buf.as<__half>();
        TENZOR_CUDA_CHECK(cudaMemsetAsync(A_pad, 0, batch_size * stride_a_pad * sizeof(__half), stream));
        TENZOR_CUDA_CHECK(cudaMemsetAsync(B_pad, 0, batch_size * stride_b_pad * sizeof(__half), stream));

        // Copy each batch element into the padded buffers with proper stride
        for (int64_t b = 0; b < batch_size; ++b) {
            // Copy A[b] (M x K) into A_pad[b] (Mp x Kp)
            TENZOR_CUDA_CHECK(cudaMemcpy2DAsync(
                A_pad + b * stride_a_pad, Kp * sizeof(__half),  // dst, dst pitch
                A     + b * stride_a,     K  * sizeof(__half),   // src, src pitch
                K  * sizeof(__half),                              // width in bytes
                M,                                                // number of rows
                cudaMemcpyDeviceToDevice, stream));

            // Copy B[b] (K x N) into B_pad[b] (Kp x Np)
            TENZOR_CUDA_CHECK(cudaMemcpy2DAsync(
                B_pad + b * stride_b_pad, Np * sizeof(__half),  // dst, dst pitch
                B     + b * stride_b,     N  * sizeof(__half),   // src, src pitch
                N  * sizeof(__half),                              // width in bytes
                K,                                                // number of rows
                cudaMemcpyDeviceToDevice, stream));
        }

        // Run batched Tensor Core kernel on padded matrices
        dim3 block(32, 1);
        dim3 grid((Np + WMMA_N - 1) / WMMA_N,
                  (Mp + WMMA_M - 1) / WMMA_M,
                  batch_size);

        batched_matmul_tensor_core_f16_kernel<<<grid, block, 0, stream>>>(
            A_pad, B_pad, C_pad, batch_size, Mp, Np, Kp,
            stride_a_pad, stride_b_pad, stride_c_pad);
        TENZOR_CUDA_POST_LAUNCH_CHECK();

        // Copy results back: extract top-left M x N from each batch element
        for (int64_t b = 0; b < batch_size; ++b) {
            TENZOR_CUDA_CHECK(cudaMemcpy2DAsync(
                C     + b * stride_c,     N  * sizeof(__half),   // dst, dst pitch
                C_pad + b * stride_c_pad, Np * sizeof(__half),   // src, src pitch
                N  * sizeof(__half),                              // width in bytes
                M,                                                // number of rows
                cudaMemcpyDeviceToDevice, stream));
        }
        // CudaAsyncBuffer RAII handles cleanup automatically
    } else {
        // Batched tiled F16 kernel using blockIdx.z for batch indexing
        // (either too small for TC or padding overhead too high)
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
                TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
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
                                stride_a, stride_b, stride_c, stream);
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

    TENZOR_CUDA_POST_LAUNCH_CHECK();
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
