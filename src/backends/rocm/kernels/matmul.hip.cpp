/**
 * @file matmul.hip.cpp
 * @brief Matrix multiplication implementation using rocBLAS for ROCm backend
 *
 * Provides GEMM (General Matrix Multiply) operations using AMD's rocBLAS library.
 * Supports both 2D matrix multiplication and batched operations for float and double precision.
 */

#include "rocm_nan_helpers.hip.h"  // E.2: safe_f2h / safe_h2f / safe_f2bf / safe_bf2f
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"      // real, imag, add, sub (for complex matmul)
#include "tenzor/ops/creation.hpp"  // complex (recombine re/im)
#include "tenzor/core/dtype.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// Include FP16 headers before namespace declaration to avoid conflicts
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp16.h>
#endif

#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <memory>
#include "fp16_saturate.h"
#include "../rocm_error.hpp"
#include "tenzor/utils/logging.hpp"

namespace tenzor {
namespace rocm {

// Forward declaration for FP8 emulation (defined in transform.hip.cpp)
auto cast_kernel(const Tensor& input, DType target_dtype, hipStream_t stream) -> Tensor;

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

/**
 * @brief Return a thread-local rocBLAS handle, lazily constructed on first use.
 *
 * rocblas_create_handle() is expensive (allocates pinned host memory, probes
 * device properties). The matmul path previously constructed a fresh handle
 * on every call; now we cache one per thread and only swap the stream, which
 * is a cheap pointer store inside rocBLAS.
 */
static RocblasHandle& cached_rocblas_handle(hipStream_t stream) {
    thread_local RocblasHandle h;
    h.set_stream(stream);
    return h;
}

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
 * @brief Tiled matrix multiplication kernel for Int32
 *
 * Uses shared memory tiling for efficient integer matrix multiplication.
 * Note: rocBLAS doesn't support integer GEMM, so this native kernel is the primary path.
 */
template<int TILE_SIZE = 16>
__global__ void matmul_tiled_i32_kernel(
    const int32_t* __restrict__ A,
    const int32_t* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K
) {
    __shared__ int32_t As[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int64_t sum = 0;  // Use int64 for accumulation to avoid overflow

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0;
        }

        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += static_cast<int64_t>(As[ty][k]) * static_cast<int64_t>(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        int32_t result = (sum > static_cast<int64_t>(INT32_MAX)) ? INT32_MAX
                       : (sum < static_cast<int64_t>(INT32_MIN)) ? INT32_MIN
                       : static_cast<int32_t>(sum);
        C[row * N + col] = result;
    }
}

/**
 * @brief Tiled matrix multiplication kernel for Int64
 */
template<int TILE_SIZE = 16>
__global__ void matmul_tiled_i64_kernel(
    const int64_t* __restrict__ A,
    const int64_t* __restrict__ B,
    int64_t* __restrict__ C,
    int M, int N, int K
) {
    __shared__ int64_t As[TILE_SIZE][TILE_SIZE];
    __shared__ int64_t Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int64_t sum = 0;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0;
        }

        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0;
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

// ============================================================================
// FP16/BF16 Matrix Multiplication with WMMA (AMD Matrix Cores)
// ============================================================================

#ifdef __HIP_PLATFORM_AMD__

/**
 * @brief WMMA-accelerated matrix multiplication for FP16
 *
 * Uses AMD's WMMA API (Wave Matrix Multiply-Accumulate) for FP16 tensor core acceleration.
 * Requires CDNA architecture (MI100, MI200 series) or RDNA3+ with WMMA support.
 *
 * WMMA configuration:
 * - Block size: 16x16
 * - Each wavefront (64 threads) processes a 16x16 block
 * - Uses FP16 for input/output with FP32 accumulation internally
 */
template<int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
__global__ void matmul_wmma_fp16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K
) {
    // WMMA fragment declarations would go here
    // Note: AMD's WMMA API differs from NVIDIA's, using amd_wmma intrinsics
    // For production use, proper WMMA fragment declarations and load/mma/store ops needed

    // Fallback to standard tiled implementation if WMMA not available
    __shared__ __half As[16][16];
    __shared__ __half Bs[16][16];

    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;

    float sum = 0.0f;

    int num_tiles = (K + 15) / 16;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * 16 + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = tenzor::rocm::safe_f2h(0.0f);
        }

        int b_row = t * 16 + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = tenzor::rocm::safe_f2h(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum += tenzor::rocm::safe_h2f(As[ty][k]) * tenzor::rocm::safe_h2f(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = tenzor::rocm::safe_f2h(sum);
    }
}

/**
 * @brief BF16 matrix multiplication kernel
 *
 * Note: BF16 (Brain Float16) support requires CDNA2+ (MI200 series)
 * Uses similar approach to FP16 but with bfloat16 type
 */
#ifdef __HIP_BFLOAT16__
template<int TILE_SIZE = 16>
__global__ void matmul_tiled_bf16_kernel(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    __hip_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __hip_bfloat16 As[TILE_SIZE][TILE_SIZE];
    __shared__ __hip_bfloat16 Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = tenzor::rocm::safe_f2bf(0.0f);
        }

        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = tenzor::rocm::safe_f2bf(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tenzor::rocm::safe_bf2f(As[ty][k]) * tenzor::rocm::safe_bf2f(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = tenzor::rocm::safe_f2bf(sum);
    }
}
#endif // __HIP_BFLOAT16__

#endif // __HIP_PLATFORM_AMD__

/**
 * @brief Native HIP matrix multiplication (fallback when rocBLAS unavailable)
 */
static Tensor matmul_native_hip(const Tensor& a, const Tensor& b, hipStream_t stream) {
    // The native tiled kernels below only cover Float32/Float64/Int32/Int64.
    // Half-precision inputs (reached via the rocBLAS catch fallback on e.g.
    // gfx90c) would otherwise launch no kernel and return uninitialized device
    // memory. Widen to Float32, run the f32 path, narrow back.
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        DType orig = a.dtype();
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        return matmul_native_hip(a_f32, b_f32, stream).to(orig);
    }
    if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64 &&
        a.dtype() != DType::Int32 && a.dtype() != DType::Int64) {
        throw std::runtime_error(
            "matmul_native_hip: unsupported dtype for native HIP matmul");
    }

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
    } else if (a.dtype() == DType::Int32) {
        const int32_t* a_data = a.data<int32_t>();
        const int32_t* b_data = b.data<int32_t>();
        int32_t* c_data = result.data<int32_t>();

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            int64_t a_offset = (batch_size_a == 1) ? 0 : batch * stride_a;
            int64_t b_offset = (batch_size_b == 1) ? 0 : batch * stride_b;
            int64_t c_offset = batch * stride_c;

            matmul_tiled_i32_kernel<TILE_SIZE><<<grid, block, 0, stream>>>(
                a_data + a_offset,
                b_data + b_offset,
                c_data + c_offset,
                M, N, K
            );
        }
    } else if (a.dtype() == DType::Int64) {
        const int64_t* a_data = a.data<int64_t>();
        const int64_t* b_data = b.data<int64_t>();
        int64_t* c_data = result.data<int64_t>();

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            int64_t a_offset = (batch_size_a == 1) ? 0 : batch * stride_a;
            int64_t b_offset = (batch_size_b == 1) ? 0 : batch * stride_b;
            int64_t c_offset = batch * stride_c;

            matmul_tiled_i64_kernel<TILE_SIZE><<<grid, block, 0, stream>>>(
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
    // Make tensors contiguous if needed (does not break autograd chain)
    Tensor a_contig = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contig = b.is_contiguous() ? b : b.contiguous();

    // Validate input dtypes match
    if (a_contig.dtype() != b_contig.dtype()) {
        throw std::runtime_error("matmul: input tensors must have the same dtype");
    }

    // Complex matmul via real-matmul decomposition. rocBLAS exposes cgemm/zgemm,
    // but plumbing complex through every 2D/batched/vector GEMM path is large;
    // instead use (Ar+iAi)(Br+iBi) = (Ar·Br - Ai·Bi) + i(Ar·Bi + Ai·Br), which
    // reuses the (correct) real GEMM and the real/imag/complex ops and naturally
    // covers 2D, batched (bmm) and vector·matrix shapes.
    if (a_contig.dtype() == DType::Complex64 || a_contig.dtype() == DType::Complex128) {
        auto Ar = tenzor::real(a_contig);
        auto Ai = tenzor::imag(a_contig);
        auto Br = tenzor::real(b_contig);
        auto Bi = tenzor::imag(b_contig);
        auto Re = tenzor::sub(matmul_kernel(Ar, Br, stream), matmul_kernel(Ai, Bi, stream));
        auto Im = tenzor::add(matmul_kernel(Ar, Bi, stream), matmul_kernel(Ai, Br, stream));
        return tenzor::complex(Re, Im);
    }

    // FP8 emulation: widen to Float32, matmul, narrow back
    if (a_contig.dtype() == DType::FP8_E4M3 || a_contig.dtype() == DType::FP8_E5M2 ||
        a_contig.dtype() == DType::FP8_E4M3FNUZ || a_contig.dtype() == DType::FP8_E5M2FNUZ) {
        DType orig = a_contig.dtype();
        auto a_f32 = cast_kernel(a_contig, DType::Float32, stream);
        auto b_f32 = cast_kernel(b_contig, DType::Float32, stream);
        auto result_f32 = matmul_kernel(a_f32, b_f32, stream);
        return cast_kernel(result_f32, orig, stream);
    }

    // Narrow integer types (Int16/UInt16/UInt32/UInt64): widen to Int64, matmul
    // with the native integer kernel, narrow back. rocBLAS has no integer GEMM.
    if (a_contig.dtype() == DType::Int16 || a_contig.dtype() == DType::UInt16 ||
        a_contig.dtype() == DType::UInt32 || a_contig.dtype() == DType::UInt64) {
        DType orig = a_contig.dtype();
        auto a64 = cast_kernel(a_contig, DType::Int64, stream);
        auto b64 = cast_kernel(b_contig, DType::Int64, stream);
        return cast_kernel(matmul_kernel(a64, b64, stream), orig, stream);
    }

    // Support Float32, Float64, Float16, Int32, and Int64
    // Integer types use native HIP kernels since rocBLAS doesn't support integer GEMM
    if (a_contig.dtype() != DType::Float32 && a_contig.dtype() != DType::Float64 &&
        a_contig.dtype() != DType::Float16 && a_contig.dtype() != DType::BFloat16 &&
        a_contig.dtype() != DType::Int32 && a_contig.dtype() != DType::Int64) {
        throw std::runtime_error("matmul: only Float32, Float64, Float16, BFloat16, Int32, and Int64 dtypes are supported");
    }

    // For integer types, use native HIP kernel directly (rocBLAS doesn't support integer GEMM)
    if (a_contig.dtype() == DType::Int32 || a_contig.dtype() == DType::Int64) {
        return matmul_native_hip(a_contig, b_contig, stream);
    }

    // Get tensor shapes
    auto a_shape = a_contig.shape();
    auto b_shape = b_contig.shape();

    // Handle 1D vector × 2D matrix (vector-matrix multiplication)
    if (a_shape.size() == 1 && b_shape.size() == 2) {
        int64_t vec_size = a_shape[0];  // Vector size
        int64_t mat_rows = b_shape[0];  // Matrix rows
        int64_t mat_cols = b_shape[1];  // Matrix cols

        if (vec_size != mat_rows) {
            throw std::runtime_error(
                "matmul dimension mismatch: vector(" + std::to_string(vec_size) +
                ") @ matrix(" + std::to_string(mat_rows) + "×" + std::to_string(mat_cols) + ")"
            );
        }

        // Treat 1D vector as row vector (1, vec_size) and perform matmul to get (1, mat_cols)
        // Then return result with 1D shape (mat_cols,)
        int64_t M = 1;
        int64_t K = vec_size;
        int64_t N = mat_cols;

        // Create output tensor with 1D shape
        Tensor result({N}, a_contig.dtype(), a_contig.device());

        // Handle empty matrices
        if (N == 0 || K == 0) {
            return result;
        }

        // Use rocBLAS GEMM with M=1
        try {
            auto& handle = cached_rocblas_handle(stream);

            // rocBLAS uses column-major, we use row-major
            // C = A * B in row-major => C^T = B^T * A^T in column-major
            rocblas_operation trans_a = rocblas_operation_none;
            rocblas_operation trans_b = rocblas_operation_none;

            rocblas_int lda = static_cast<rocblas_int>(K);  // A is 1 x K
            rocblas_int ldb = static_cast<rocblas_int>(N);  // B is K x N
            rocblas_int ldc = static_cast<rocblas_int>(N);  // C is 1 x N

            rocblas_int m = static_cast<rocblas_int>(N);  // Swapped for transpose
            rocblas_int n = static_cast<rocblas_int>(M);  // Swapped for transpose
            rocblas_int k = static_cast<rocblas_int>(K);

            if (a_contig.dtype() == DType::Float32) {
                const float* a_data = a_contig.data<float>();
                const float* b_data = b_contig.data<float>();
                float* c_data = result.data<float>();

                float alpha = 1.0f;
                float beta = 0.0f;

                ROCBLAS_CHECK(rocblas_sgemm(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, ldb,
                    a_data, lda,
                    &beta,
                    c_data, ldc
                ));
            } else if (a_contig.dtype() == DType::Float64) {
                const double* a_data = a_contig.data<double>();
                const double* b_data = b_contig.data<double>();
                double* c_data = result.data<double>();

                double alpha = 1.0;
                double beta = 0.0;

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
            } else if (a_contig.dtype() == DType::Float16) {
                const Float16* a_data = a_contig.data<Float16>();
                const Float16* b_data = b_contig.data<Float16>();
                Float16* c_data = result.data<Float16>();

                float alpha = 1.0f;
                float beta = 0.0f;

                ROCBLAS_CHECK(rocblas_gemm_ex(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, rocblas_datatype_f16_r, ldb,
                    a_data, rocblas_datatype_f16_r, lda,
                    &beta,
                    c_data, rocblas_datatype_f16_r, ldc,
                    c_data, rocblas_datatype_f16_r, ldc,
                    rocblas_datatype_f32_r,  // compute type
                    rocblas_gemm_algo_standard,
                    0, 0  // solution index, flags
                ));
            } else if (a_contig.dtype() == DType::BFloat16) {
                // BFloat16: convert to Float32, compute, convert back
                auto a_f32 = a_contig.to(DType::Float32);
                auto b_f32 = b_contig.to(DType::Float32);
                Tensor result_f32({N}, DType::Float32, a_contig.device());

                const float* a_data = a_f32.data<float>();
                const float* b_data = b_f32.data<float>();
                float* c_data = result_f32.data<float>();

                float alpha = 1.0f;
                float beta = 0.0f;

                ROCBLAS_CHECK(rocblas_sgemm(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, ldb,
                    a_data, lda,
                    &beta,
                    c_data, ldc
                ));
                return result_f32.to(DType::BFloat16);
            }

            if (a_contig.dtype() == DType::Float16) {
                fp16_saturate(result.data_ptr(), result.numel(), stream);
            }
            return result;
        } catch (const std::runtime_error& e) {
            // TENZOR_STRICT_BACKEND=1 re-throws so silent fallback can't hide
            // a regression. Otherwise log and use the 2D-reshape fallback.
            if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
                throw std::runtime_error(
                    std::string("ROCm matmul (1D path): rocBLAS failed "
                                "(TENZOR_STRICT_BACKEND=1): ") + e.what());
            }
            TENZOR_LOG_WARNING(std::format(
                "rocBLAS 1D matmul failed ({}), using 2D-reshape fallback", e.what()));
        }

        // Fallback: reshape to 2D, compute, and squeeze
        Tensor a_2d = a_contig.reshape({1, K});
        Tensor result_2d = matmul_kernel(a_2d, b_contig, stream);
        return result_2d.reshape({N});
    }

    // Handle 2D matrix × 1D vector (matrix-vector multiplication)
    if (a_shape.size() == 2 && b_shape.size() == 1) {
        int64_t mat_rows = a_shape[0];  // Matrix rows
        int64_t mat_cols = a_shape[1];  // Matrix cols
        int64_t vec_size = b_shape[0];  // Vector size

        if (mat_cols != vec_size) {
            throw std::runtime_error(
                "matmul dimension mismatch: matrix(" + std::to_string(mat_rows) +
                "×" + std::to_string(mat_cols) + ") @ vector(" + std::to_string(vec_size) + ")"
            );
        }

        // Treat 1D vector as column vector (vec_size, 1) and perform matmul to get (mat_rows, 1)
        // Then return result with 1D shape (mat_rows,)
        int64_t M = mat_rows;
        int64_t K = mat_cols;
        int64_t N = 1;

        // Create output tensor with 1D shape
        Tensor result({M}, a_contig.dtype(), a_contig.device());

        // Handle empty matrices
        if (M == 0 || K == 0) {
            return result;
        }

        // Use rocBLAS GEMM with N=1
        try {
            auto& handle = cached_rocblas_handle(stream);

            rocblas_operation trans_a = rocblas_operation_none;
            rocblas_operation trans_b = rocblas_operation_none;

            rocblas_int lda = static_cast<rocblas_int>(K);  // A is M x K
            rocblas_int ldb = static_cast<rocblas_int>(N);  // B is K x 1
            rocblas_int ldc = static_cast<rocblas_int>(N);  // C is M x 1

            rocblas_int m = static_cast<rocblas_int>(N);  // Swapped for transpose
            rocblas_int n = static_cast<rocblas_int>(M);  // Swapped for transpose
            rocblas_int k = static_cast<rocblas_int>(K);

            if (a_contig.dtype() == DType::Float32) {
                const float* a_data = a_contig.data<float>();
                const float* b_data = b_contig.data<float>();
                float* c_data = result.data<float>();

                float alpha = 1.0f;
                float beta = 0.0f;

                ROCBLAS_CHECK(rocblas_sgemm(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, ldb,
                    a_data, lda,
                    &beta,
                    c_data, ldc
                ));
            } else if (a_contig.dtype() == DType::Float64) {
                const double* a_data = a_contig.data<double>();
                const double* b_data = b_contig.data<double>();
                double* c_data = result.data<double>();

                double alpha = 1.0;
                double beta = 0.0;

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
            } else if (a_contig.dtype() == DType::Float16) {
                const Float16* a_data = a_contig.data<Float16>();
                const Float16* b_data = b_contig.data<Float16>();
                Float16* c_data = result.data<Float16>();

                float alpha = 1.0f;
                float beta = 0.0f;

                ROCBLAS_CHECK(rocblas_gemm_ex(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, rocblas_datatype_f16_r, ldb,
                    a_data, rocblas_datatype_f16_r, lda,
                    &beta,
                    c_data, rocblas_datatype_f16_r, ldc,
                    c_data, rocblas_datatype_f16_r, ldc,
                    rocblas_datatype_f32_r,
                    rocblas_gemm_algo_standard,
                    0, 0
                ));
            } else if (a_contig.dtype() == DType::BFloat16) {
                auto a_f32 = a_contig.to(DType::Float32);
                auto b_f32 = b_contig.to(DType::Float32);

                const float* a_data = a_f32.data<float>();
                const float* b_data = b_f32.data<float>();
                float* c_data = result.data<float>();

                float alpha = 1.0f;
                float beta = 0.0f;

                // Need Float32 result tensor for computation
                Tensor result_f32({M}, DType::Float32, a_contig.device());
                c_data = result_f32.data<float>();

                ROCBLAS_CHECK(rocblas_sgemm(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, ldb,
                    a_data, lda,
                    &beta,
                    c_data, ldc
                ));
                return result_f32.to(DType::BFloat16);
            }

            if (a_contig.dtype() == DType::Float16) {
                fp16_saturate(result.data_ptr(), result.numel(), stream);
            }
            return result;
        } catch (const std::runtime_error& e) {
            if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
                throw std::runtime_error(
                    std::string("ROCm matmul (matrix-vector): rocBLAS failed "
                                "(TENZOR_STRICT_BACKEND=1): ") + e.what());
            }
            TENZOR_LOG_WARNING(std::format(
                "rocBLAS matrix-vector matmul failed ({}), using 2D-reshape fallback",
                e.what()));
        }

        // Fallback: reshape to 2D, compute, and squeeze
        Tensor b_2d = b_contig.reshape({K, 1});
        Tensor result_2d = matmul_kernel(a_contig, b_2d, stream);
        return result_2d.reshape({M});
    }

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
    Tensor result(out_shape, a_contig.dtype(), a_contig.device());

    // Handle empty matrices
    if (M == 0 || N == 0 || K == 0) {
        return result;
    }

    // Try rocBLAS first, fallback to native HIP on failure
    // rocBLAS may fail on unsupported architectures (e.g., gfx90c)
    bool use_rocblas = true;

    try {
        // Reuse the thread-local rocBLAS handle (cheap stream swap per call)
        auto& handle = cached_rocblas_handle(stream);

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

        if (a_contig.dtype() == DType::Float32) {
            const float* a_data = a_contig.data<float>();
            const float* b_data = b_contig.data<float>();
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
        } else if (a_contig.dtype() == DType::Float64) {
            const double* a_data = a_contig.data<double>();
            const double* b_data = b_contig.data<double>();
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
        } else if (a_contig.dtype() == DType::Float16) {
            // Float16 (half precision) GEMM using rocBLAS
            // Cast tenzor::Float16* to rocblas_half* (binary compatible)
            const rocblas_half* a_data = reinterpret_cast<const rocblas_half*>(a_contig.data<Float16>());
            const rocblas_half* b_data = reinterpret_cast<const rocblas_half*>(b_contig.data<Float16>());
            rocblas_half* c_data = reinterpret_cast<rocblas_half*>(result.data<Float16>());

            // Use float for alpha/beta to maintain precision in accumulation
            float alpha = 1.0f;
            float beta = 0.0f;

            if (batch_size == 1) {
                // Single matrix multiplication using mixed-precision HGEMM
                // rocblas_hgemm computes in half but can use float for alpha/beta
                ROCBLAS_CHECK(rocblas_gemm_ex(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, rocblas_datatype_f16_r, ldb,
                    a_data, rocblas_datatype_f16_r, lda,
                    &beta,
                    c_data, rocblas_datatype_f16_r, ldc,
                    c_data, rocblas_datatype_f16_r, ldc,
                    rocblas_datatype_f32_r,  // Compute type: use FP32 for accuracy
                    rocblas_gemm_algo_standard,
                    0, 0
                ));
            } else {
                // Batched matrix multiplication
                rocblas_int batch_count = static_cast<rocblas_int>(batch_size);

                rocblas_stride actual_stride_a = (batch_size_a == 1) ? 0 : stride_a;
                rocblas_stride actual_stride_b = (batch_size_b == 1) ? 0 : stride_b;

                ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
                    handle.get(),
                    trans_a, trans_b,
                    m, n, k,
                    &alpha,
                    b_data, rocblas_datatype_f16_r, ldb, actual_stride_b,
                    a_data, rocblas_datatype_f16_r, lda, actual_stride_a,
                    &beta,
                    c_data, rocblas_datatype_f16_r, ldc, stride_c,
                    c_data, rocblas_datatype_f16_r, ldc, stride_c,
                    batch_count,
                    rocblas_datatype_f32_r,  // Compute type: use FP32 for accuracy
                    rocblas_gemm_algo_standard,
                    0, 0
                ));
            }
        } else if (a_contig.dtype() == DType::BFloat16) {
            // BFloat16: convert to Float32, compute via rocBLAS, convert back
            auto a_f32 = a_contig.to(DType::Float32);
            auto b_f32 = b_contig.to(DType::Float32);
            Tensor result_f32(out_shape, DType::Float32, a_contig.device());

            const float* a_data = a_f32.data<float>();
            const float* b_data = b_f32.data<float>();
            float* c_data = result_f32.data<float>();

            float alpha = 1.0f;
            float beta = 0.0f;

            if (batch_size == 1) {
                ROCBLAS_CHECK(rocblas_sgemm(
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
                rocblas_int batch_count = static_cast<rocblas_int>(batch_size);
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
            result = result_f32.to(DType::BFloat16);
        }
    } catch (const std::exception& e) {
        // rocBLAS failed (e.g., unsupported architecture like gfx90c).
        // TENZOR_STRICT_BACKEND=1 re-throws so silent fallback to the native
        // HIP kernel cannot mask a regression. Otherwise log every failure
        // (the previous one-shot static-bool guard hid recurring issues).
        if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
            throw std::runtime_error(
                std::string("ROCm matmul: rocBLAS failed "
                            "(TENZOR_STRICT_BACKEND=1): ") +
                e.what());
        }
        use_rocblas = false;
        TENZOR_LOG_WARNING(std::format(
            "rocBLAS matmul failed ({}), using native HIP fallback", e.what()));
        return matmul_native_hip(a_contig, b_contig, stream);
    }

    if (a_contig.dtype() == DType::Float16) {
        fp16_saturate(result.data_ptr(), result.numel(), stream);
    }
    return result;
}

// Note: Reduction operations (sum_kernel, mean_kernel, max_kernel, min_kernel)
// are now implemented in reduction.hip.cpp

// Note: Random operations (rand_kernel, randn_kernel) are now implemented in math.hip.cpp

} // namespace rocm
} // namespace tenzor
