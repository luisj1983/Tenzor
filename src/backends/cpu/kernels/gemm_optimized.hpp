/**
 * @file gemm_optimized.hpp
 * @brief High-performance GEMM implementations with register blocking and prefetching
 *
 * Features:
 * - 6x16 micro-kernel for AVX2 (uses 12 YMM accumulators)
 * - 6x32 micro-kernel for AVX-512 (uses 12 ZMM accumulators)
 * - Cache-optimized block sizes for L1/L2/L3
 * - Software prefetching for next blocks
 * - OpenMP parallelization at the right level
 *
 * Achieves 2-4x speedup over naive blocked GEMM
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>
#include "tenzor/backend/omp_thresholds.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #if defined(__AVX512F__)
        #define TENZOR_GEMM_AVX512
    #endif
    #if defined(__AVX2__)
        #define TENZOR_GEMM_AVX2
    #endif
    #if defined(__FMA__)
        #define TENZOR_GEMM_FMA
    #endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace cpu {
namespace gemm {

// ============================================================================
// Cache-optimized block sizes
// ============================================================================

// Block sizes optimized for typical CPU cache hierarchy:
// L1: 32KB, L2: 256-512KB, L3: 8-32MB
//
// Strategy:
// - MC x KC block of A fits in L2 (with some headroom)
// - KC x NC block of B fits in L3
// - MC x NC block of C stays in registers/L1

// For AVX2 (256-bit = 8 floats)
constexpr size_t MC_AVX2 = 96;   // Rows of A/C processed per iteration
constexpr size_t NC_AVX2 = 256;  // Cols of B/C processed per iteration
constexpr size_t KC_AVX2 = 256;  // Reduction dimension block

// For AVX-512 (512-bit = 16 floats)
constexpr size_t MC_AVX512 = 96;
constexpr size_t NC_AVX512 = 512;
constexpr size_t KC_AVX512 = 256;

// Micro-kernel sizes
constexpr size_t MR_AVX2 = 6;    // Micro rows
constexpr size_t NR_AVX2 = 16;   // Micro cols (2 YMM registers)

constexpr size_t MR_AVX512 = 6;
constexpr size_t NR_AVX512 = 32; // 2 ZMM registers

// Prefetch distances
constexpr size_t PREFETCH_DISTANCE_A = 64;  // Cache lines ahead
constexpr size_t PREFETCH_DISTANCE_B = 128;

// Maximum packing buffer sizes (used for heap allocation to avoid stack overflow)
constexpr size_t MAX_A_PACK_SIZE = std::max(MC_AVX2 * KC_AVX2, MC_AVX512 * KC_AVX512);
constexpr size_t MAX_B_PACK_SIZE = std::max(KC_AVX2 * NC_AVX2, KC_AVX512 * NC_AVX512);

// Packing buffers allocated per-thread on the heap inside parallel regions.
// NOT thread_local: dlopen'd shared libraries don't initialize thread_local
// storage for OMP worker threads, causing crashes.
struct GemmPackBuffers {
    alignas(64) float A_packed[MAX_A_PACK_SIZE];
    alignas(64) float B_packed[MAX_B_PACK_SIZE];
};

// ============================================================================
// AVX2 Micro-kernel (6x16)
// ============================================================================

#ifdef TENZOR_GEMM_AVX2

/**
 * @brief 6x16 micro-kernel for AVX2
 *
 * Computes C[6x16] += A[6xK] * B[Kx16]
 * Uses 12 YMM registers for C accumulation, maximizing register utilization
 *
 * @param A Pointer to 6xK micro-panel of A (row-major, packed)
 * @param B Pointer to Kx16 micro-panel of B (column-major, packed for this kernel)
 * @param C Pointer to 6x16 block of C (row-major)
 * @param K Reduction dimension
 * @param ldc Leading dimension of C
 */
__attribute__((target("avx2,fma")))
inline void microkernel_6x16_avx2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t K,
    int64_t ldc,
    float alpha = 1.0f
) {
    // 12 accumulators for 6x16 C tile (2 YMM per row)
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();

    // Process K dimension with software prefetching
    for (int64_t k = 0; k < K; ++k) {
        // Prefetch next A and B values
        if (k + static_cast<int64_t>(PREFETCH_DISTANCE_A) < K) {
            _mm_prefetch(reinterpret_cast<const char*>(A + (k + PREFETCH_DISTANCE_A) * MR_AVX2), _MM_HINT_T0);
        }
        if (k + static_cast<int64_t>(PREFETCH_DISTANCE_B) < K) {
            _mm_prefetch(reinterpret_cast<const char*>(B + (k + PREFETCH_DISTANCE_B) * NR_AVX2), _MM_HINT_T0);
        }

        // Load B row (16 elements = 2 YMM)
        __m256 b0 = _mm256_loadu_ps(B + k * NR_AVX2);
        __m256 b1 = _mm256_loadu_ps(B + k * NR_AVX2 + 8);

        // Broadcast each A element and FMA
        __m256 a0 = _mm256_broadcast_ss(A + k * MR_AVX2 + 0);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        __m256 a1 = _mm256_broadcast_ss(A + k * MR_AVX2 + 1);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 a2 = _mm256_broadcast_ss(A + k * MR_AVX2 + 2);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 a3 = _mm256_broadcast_ss(A + k * MR_AVX2 + 3);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);

        __m256 a4 = _mm256_broadcast_ss(A + k * MR_AVX2 + 4);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);

        __m256 a5 = _mm256_broadcast_ss(A + k * MR_AVX2 + 5);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    // Store results back to C (accumulate with alpha scaling fused)
    // Fusing alpha here eliminates a separate O(M*N) scaling pass over C.
    if (alpha != 1.0f) {
        __m256 valpha = _mm256_set1_ps(alpha);
        c00 = _mm256_mul_ps(c00, valpha); c01 = _mm256_mul_ps(c01, valpha);
        c10 = _mm256_mul_ps(c10, valpha); c11 = _mm256_mul_ps(c11, valpha);
        c20 = _mm256_mul_ps(c20, valpha); c21 = _mm256_mul_ps(c21, valpha);
        c30 = _mm256_mul_ps(c30, valpha); c31 = _mm256_mul_ps(c31, valpha);
        c40 = _mm256_mul_ps(c40, valpha); c41 = _mm256_mul_ps(c41, valpha);
        c50 = _mm256_mul_ps(c50, valpha); c51 = _mm256_mul_ps(c51, valpha);
    }
    _mm256_storeu_ps(C + 0 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldc + 0), c00));
    _mm256_storeu_ps(C + 0 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldc + 8), c01));
    _mm256_storeu_ps(C + 1 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldc + 0), c10));
    _mm256_storeu_ps(C + 1 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldc + 8), c11));
    _mm256_storeu_ps(C + 2 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldc + 0), c20));
    _mm256_storeu_ps(C + 2 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldc + 8), c21));
    _mm256_storeu_ps(C + 3 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldc + 0), c30));
    _mm256_storeu_ps(C + 3 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldc + 8), c31));
    _mm256_storeu_ps(C + 4 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 4 * ldc + 0), c40));
    _mm256_storeu_ps(C + 4 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 4 * ldc + 8), c41));
    _mm256_storeu_ps(C + 5 * ldc + 0, _mm256_add_ps(_mm256_loadu_ps(C + 5 * ldc + 0), c50));
    _mm256_storeu_ps(C + 5 * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 5 * ldc + 8), c51));
}

/**
 * @brief Pack A matrix into column-major micro-panels
 */
inline void pack_a_avx2(
    const float* __restrict__ A,
    float* __restrict__ A_packed,
    int64_t M, int64_t K, int64_t lda
) {
    for (int64_t i = 0; i < M; i += MR_AVX2) {
        int64_t mb = std::min(static_cast<int64_t>(MR_AVX2), M - i);

        for (int64_t k = 0; k < K; ++k) {
            for (int64_t ii = 0; ii < mb; ++ii) {
                A_packed[(i / MR_AVX2) * K * MR_AVX2 + k * MR_AVX2 + ii] = A[(i + ii) * lda + k];
            }
            // Pad with zeros if mb < MR
            for (int64_t ii = mb; ii < static_cast<int64_t>(MR_AVX2); ++ii) {
                A_packed[(i / MR_AVX2) * K * MR_AVX2 + k * MR_AVX2 + ii] = 0.0f;
            }
        }
    }
}

/**
 * @brief Pack B matrix into row-major micro-panels
 */
inline void pack_b_avx2(
    const float* __restrict__ B,
    float* __restrict__ B_packed,
    int64_t K, int64_t N, int64_t ldb
) {
    for (int64_t j = 0; j < N; j += NR_AVX2) {
        int64_t nb = std::min(static_cast<int64_t>(NR_AVX2), N - j);

        for (int64_t k = 0; k < K; ++k) {
            for (int64_t jj = 0; jj < nb; ++jj) {
                B_packed[(j / NR_AVX2) * K * NR_AVX2 + k * NR_AVX2 + jj] = B[k * ldb + j + jj];
            }
            // Pad with zeros if nb < NR
            for (int64_t jj = nb; jj < static_cast<int64_t>(NR_AVX2); ++jj) {
                B_packed[(j / NR_AVX2) * K * NR_AVX2 + k * NR_AVX2 + jj] = 0.0f;
            }
        }
    }
}

#endif // TENZOR_GEMM_AVX2

// ============================================================================
// AVX-512 Micro-kernel (6x32)
// ============================================================================

#ifdef TENZOR_GEMM_AVX512

/**
 * @brief 6x32 micro-kernel for AVX-512
 */
__attribute__((target("avx512f,avx512vl")))
inline void microkernel_6x32_avx512(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t K,
    int64_t ldc,
    float alpha = 1.0f
) {
    // 12 accumulators for 6x32 C tile (2 ZMM per row)
    __m512 c00 = _mm512_setzero_ps();
    __m512 c01 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c11 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c21 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c31 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps();
    __m512 c41 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps();
    __m512 c51 = _mm512_setzero_ps();

    for (int64_t k = 0; k < K; ++k) {
        // Prefetch
        if (k + static_cast<int64_t>(PREFETCH_DISTANCE_A) < K) {
            _mm_prefetch(reinterpret_cast<const char*>(A + (k + PREFETCH_DISTANCE_A) * MR_AVX512), _MM_HINT_T0);
        }
        if (k + static_cast<int64_t>(PREFETCH_DISTANCE_B) < K) {
            _mm_prefetch(reinterpret_cast<const char*>(B + (k + PREFETCH_DISTANCE_B) * NR_AVX512), _MM_HINT_T0);
        }

        // Load B row (32 elements = 2 ZMM)
        __m512 b0 = _mm512_loadu_ps(B + k * NR_AVX512);
        __m512 b1 = _mm512_loadu_ps(B + k * NR_AVX512 + 16);

        // Broadcast each A element and FMA
        __m512 a0 = _mm512_set1_ps(A[k * MR_AVX512 + 0]);
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);

        __m512 a1 = _mm512_set1_ps(A[k * MR_AVX512 + 1]);
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);

        __m512 a2 = _mm512_set1_ps(A[k * MR_AVX512 + 2]);
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);

        __m512 a3 = _mm512_set1_ps(A[k * MR_AVX512 + 3]);
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);

        __m512 a4 = _mm512_set1_ps(A[k * MR_AVX512 + 4]);
        c40 = _mm512_fmadd_ps(a4, b0, c40);
        c41 = _mm512_fmadd_ps(a4, b1, c41);

        __m512 a5 = _mm512_set1_ps(A[k * MR_AVX512 + 5]);
        c50 = _mm512_fmadd_ps(a5, b0, c50);
        c51 = _mm512_fmadd_ps(a5, b1, c51);
    }

    // Store results back to C (accumulate with alpha scaling fused)
    if (alpha != 1.0f) {
        __m512 valpha = _mm512_set1_ps(alpha);
        c00 = _mm512_mul_ps(c00, valpha); c01 = _mm512_mul_ps(c01, valpha);
        c10 = _mm512_mul_ps(c10, valpha); c11 = _mm512_mul_ps(c11, valpha);
        c20 = _mm512_mul_ps(c20, valpha); c21 = _mm512_mul_ps(c21, valpha);
        c30 = _mm512_mul_ps(c30, valpha); c31 = _mm512_mul_ps(c31, valpha);
        c40 = _mm512_mul_ps(c40, valpha); c41 = _mm512_mul_ps(c41, valpha);
        c50 = _mm512_mul_ps(c50, valpha); c51 = _mm512_mul_ps(c51, valpha);
    }
    _mm512_storeu_ps(C + 0 * ldc + 0,  _mm512_add_ps(_mm512_loadu_ps(C + 0 * ldc + 0),  c00));
    _mm512_storeu_ps(C + 0 * ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 0 * ldc + 16), c01));
    _mm512_storeu_ps(C + 1 * ldc + 0,  _mm512_add_ps(_mm512_loadu_ps(C + 1 * ldc + 0),  c10));
    _mm512_storeu_ps(C + 1 * ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 1 * ldc + 16), c11));
    _mm512_storeu_ps(C + 2 * ldc + 0,  _mm512_add_ps(_mm512_loadu_ps(C + 2 * ldc + 0),  c20));
    _mm512_storeu_ps(C + 2 * ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 2 * ldc + 16), c21));
    _mm512_storeu_ps(C + 3 * ldc + 0,  _mm512_add_ps(_mm512_loadu_ps(C + 3 * ldc + 0),  c30));
    _mm512_storeu_ps(C + 3 * ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 3 * ldc + 16), c31));
    _mm512_storeu_ps(C + 4 * ldc + 0,  _mm512_add_ps(_mm512_loadu_ps(C + 4 * ldc + 0),  c40));
    _mm512_storeu_ps(C + 4 * ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 4 * ldc + 16), c41));
    _mm512_storeu_ps(C + 5 * ldc + 0,  _mm512_add_ps(_mm512_loadu_ps(C + 5 * ldc + 0),  c50));
    _mm512_storeu_ps(C + 5 * ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 5 * ldc + 16), c51));
}

/**
 * @brief Pack A matrix for AVX-512
 */
inline void pack_a_avx512(
    const float* __restrict__ A,
    float* __restrict__ A_packed,
    int64_t M, int64_t K, int64_t lda
) {
    for (int64_t i = 0; i < M; i += MR_AVX512) {
        int64_t mb = std::min(static_cast<int64_t>(MR_AVX512), M - i);

        for (int64_t k = 0; k < K; ++k) {
            for (int64_t ii = 0; ii < mb; ++ii) {
                A_packed[(i / MR_AVX512) * K * MR_AVX512 + k * MR_AVX512 + ii] = A[(i + ii) * lda + k];
            }
            for (int64_t ii = mb; ii < static_cast<int64_t>(MR_AVX512); ++ii) {
                A_packed[(i / MR_AVX512) * K * MR_AVX512 + k * MR_AVX512 + ii] = 0.0f;
            }
        }
    }
}

/**
 * @brief Pack B matrix for AVX-512
 */
inline void pack_b_avx512(
    const float* __restrict__ B,
    float* __restrict__ B_packed,
    int64_t K, int64_t N, int64_t ldb
) {
    for (int64_t j = 0; j < N; j += NR_AVX512) {
        int64_t nb = std::min(static_cast<int64_t>(NR_AVX512), N - j);

        for (int64_t k = 0; k < K; ++k) {
            for (int64_t jj = 0; jj < nb; ++jj) {
                B_packed[(j / NR_AVX512) * K * NR_AVX512 + k * NR_AVX512 + jj] = B[k * ldb + j + jj];
            }
            for (int64_t jj = nb; jj < static_cast<int64_t>(NR_AVX512); ++jj) {
                B_packed[(j / NR_AVX512) * K * NR_AVX512 + k * NR_AVX512 + jj] = 0.0f;
            }
        }
    }
}

#endif // TENZOR_GEMM_AVX512

// ============================================================================
// Scalar Micro-kernel (fallback)
// ============================================================================

/**
 * @brief Scalar micro-kernel for non-SIMD fallback
 */
// Scalar edge micro-kernel. The tile is (M rows × N cols × K reduction);
// the surrounding A/B/C buffers are slices of larger matrices so the
// caller must pass the FULL row strides (lda for A, ldb for B, ldc for
// C). The earlier version implicitly assumed lda==K and ldb==N — which
// made the (27, 32, 4) edge case (mr=3, nr=16) read the wrong columns
// of B for k>0, silently corrupting the last 3 rows of the result.
inline void microkernel_scalar(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc,
    float alpha = 1.0f
) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

// ============================================================================
// High-level GEMM Interface
// ============================================================================

// Adaptive OpenMP thresholds based on operation type.
// Driven by ::tenzor::OmpThresholds::matmul() (see tenzor/backend/omp_thresholds.hpp).
#define OMP_THRESHOLD_GEMM (::tenzor::OmpThresholds::matmul())

/**
 * @brief Optimized GEMM: C = alpha * A * B + beta * C
 *
 * @param A Input matrix A (M x K), row-major
 * @param B Input matrix B (K x N), row-major
 * @param C Output matrix C (M x N), row-major
 * @param M Number of rows in A/C
 * @param N Number of columns in B/C
 * @param K Shared dimension
 * @param alpha Scalar multiplier for A*B (default 1.0)
 * @param beta Scalar multiplier for C (default 0.0)
 */
inline void gemm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    float alpha = 1.0f,
    float beta = 0.0f
) {
    // Handle beta scaling
    if (beta == 0.0f) {
        std::memset(C, 0, M * N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int64_t i = 0; i < M * N; ++i) {
            C[i] *= beta;
        }
    }

    // Small matrix: use simple loop
    if (M * N < 256) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] += alpha * sum;
            }
        }
        return;
    }

#ifdef TENZOR_GEMM_AVX512
    // AVX-512 optimized path
    const size_t MC = MC_AVX512;
    const size_t NC = NC_AVX512;
    const size_t KC = KC_AVX512;
    const size_t MR = MR_AVX512;
    const size_t NR = NR_AVX512;

    // Per-thread heap packing buffers (NOT thread_local: dlopen'd libs
    // don't initialize thread_local for OMP worker threads, causing crashes)
    #pragma omp parallel if(M * N > OMP_THRESHOLD_GEMM)
    {
        auto bufs = std::make_unique<GemmPackBuffers>();
        float* A_packed = bufs->A_packed;
        float* B_packed = bufs->B_packed;

        #pragma omp for collapse(2) schedule(dynamic)
        for (int64_t jc = 0; jc < N; jc += NC) {
            for (int64_t ic = 0; ic < M; ic += MC) {
                int64_t nc = std::min(static_cast<int64_t>(NC), N - jc);
                int64_t mc = std::min(static_cast<int64_t>(MC), M - ic);

                for (int64_t pc = 0; pc < K; pc += KC) {
                    int64_t kc = std::min(static_cast<int64_t>(KC), K - pc);

                    // Pack A block
                    pack_a_avx512(A + ic * K + pc, A_packed, mc, kc, K);

                    // Pack B block
                    pack_b_avx512(B + pc * N + jc, B_packed, kc, nc, N);

                    // Compute micro-tiles
                    for (int64_t ir = 0; ir < mc; ir += MR) {
                        for (int64_t jr = 0; jr < nc; jr += NR) {
                            int64_t mr = std::min(static_cast<int64_t>(MR), mc - ir);
                            int64_t nr = std::min(static_cast<int64_t>(NR), nc - jr);

                            if (mr == MR && nr == NR) {
                                microkernel_6x32_avx512(
                                    A_packed + (ir / MR) * kc * MR,
                                    B_packed + (jr / NR) * kc * NR,
                                    C + (ic + ir) * N + jc + jr,
                                    kc, N, alpha
                                );
                            } else {
                                // Edge case: use scalar with full row strides.
                                microkernel_scalar(
                                    A + (ic + ir) * K + pc,
                                    B + pc * N + jc + jr,
                                    C + (ic + ir) * N + jc + jr,
                                    mr, nr, kc,
                                    /*lda=*/K, /*ldb=*/N, /*ldc=*/N,
                                    alpha
                                );
                            }
                        }
                    }
                }
            }
        }
    }

#elif defined(TENZOR_GEMM_AVX2)
    // AVX2 optimized path
    const size_t MC = MC_AVX2;
    const size_t NC = NC_AVX2;
    const size_t KC = KC_AVX2;
    const size_t MR = MR_AVX2;
    const size_t NR = NR_AVX2;

    // Per-thread heap packing buffers (NOT thread_local: dlopen'd libs
    // don't initialize thread_local for OMP worker threads, causing crashes)
    #pragma omp parallel if(M * N > OMP_THRESHOLD_GEMM)
    {
        auto bufs = std::make_unique<GemmPackBuffers>();
        float* A_packed = bufs->A_packed;
        float* B_packed = bufs->B_packed;

        #pragma omp for collapse(2) schedule(dynamic)
        for (int64_t jc = 0; jc < N; jc += NC) {
            for (int64_t ic = 0; ic < M; ic += MC) {
                int64_t nc = std::min(static_cast<int64_t>(NC), N - jc);
                int64_t mc = std::min(static_cast<int64_t>(MC), M - ic);

                for (int64_t pc = 0; pc < K; pc += KC) {
                    int64_t kc = std::min(static_cast<int64_t>(KC), K - pc);

                    // Pack A block
                    pack_a_avx2(A + ic * K + pc, A_packed, mc, kc, K);

                    // Pack B block
                    pack_b_avx2(B + pc * N + jc, B_packed, kc, nc, N);

                    // Compute micro-tiles
                    for (int64_t ir = 0; ir < mc; ir += MR) {
                        for (int64_t jr = 0; jr < nc; jr += NR) {
                            int64_t mr = std::min(static_cast<int64_t>(MR), mc - ir);
                            int64_t nr = std::min(static_cast<int64_t>(NR), nc - jr);

                            if (mr == MR && nr == NR) {
                                microkernel_6x16_avx2(
                                    A_packed + (ir / MR) * kc * MR,
                                    B_packed + (jr / NR) * kc * NR,
                                    C + (ic + ir) * N + jc + jr,
                                    kc, N, alpha
                                );
                            } else {
                                // Edge case: use scalar with full row strides.
                                microkernel_scalar(
                                    A + (ic + ir) * K + pc,
                                    B + pc * N + jc + jr,
                                    C + (ic + ir) * N + jc + jr,
                                    mr, nr, kc,
                                    /*lda=*/K, /*ldb=*/N, /*ldc=*/N,
                                    alpha
                                );
                            }
                        }
                    }
                }
            }
        }
    }

#else
    // Scalar fallback with cache blocking
    constexpr size_t BLOCK = 64;

    #pragma omp parallel for collapse(2) if(M * N > OMP_THRESHOLD_GEMM)
    for (int64_t ii = 0; ii < M; ii += BLOCK) {
        for (int64_t jj = 0; jj < N; jj += BLOCK) {
            for (int64_t kk = 0; kk < K; kk += BLOCK) {
                int64_t i_end = std::min(ii + static_cast<int64_t>(BLOCK), M);
                int64_t j_end = std::min(jj + static_cast<int64_t>(BLOCK), N);
                int64_t k_end = std::min(kk + static_cast<int64_t>(BLOCK), K);

                for (int64_t i = ii; i < i_end; ++i) {
                    for (int64_t j = jj; j < j_end; ++j) {
                        float sum = 0.0f;
                        for (int64_t k = kk; k < k_end; ++k) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] += alpha * sum;
                    }
                }
            }
        }
    }
#endif
}

/**
 * @brief Optimized GEMM for transposed B: C = A * B^T
 *
 * When B is transposed, it's already in row-major which is good for our access pattern.
 */
inline void gemm_transB_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,  // B is (N x K), we compute A @ B^T
    float* __restrict__ C,
    int64_t M, int64_t N, int64_t K
) {
    // For non-trivial sizes, transpose B and use the optimized blocked GEMM.
    // The O(K*N) transpose cost is dominated by the O(M*N*K) compute.
    constexpr int64_t TRANSPOSE_THRESHOLD = 64;
    if (M > TRANSPOSE_THRESHOLD && N > TRANSPOSE_THRESHOLD && K > TRANSPOSE_THRESHOLD) {
        // Transpose B (N x K) -> B_T (K x N)
        std::vector<float> B_T(static_cast<size_t>(K * N));
        #pragma omp parallel for collapse(2) if(K * N > OMP_THRESHOLD_GEMM)
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t n = 0; n < N; ++n) {
                B_T[k * N + n] = B[n * K + k];
            }
        }
        gemm_optimized(A, B_T.data(), C, M, N, K, 1.0f, 0.0f);
        return;
    }

    // For small sizes, use direct dot-product approach
    std::memset(C, 0, M * N * sizeof(float));

#ifdef TENZOR_GEMM_AVX2
    constexpr int64_t VEC_SIZE = 8;

    #pragma omp parallel for if(M * N > OMP_THRESHOLD_GEMM)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            __m256 sum = _mm256_setzero_ps();
            int64_t k = 0;

            // Vectorized dot product
            for (; k + VEC_SIZE <= K; k += VEC_SIZE) {
                __m256 a = _mm256_loadu_ps(A + i * K + k);
                __m256 b = _mm256_loadu_ps(B + j * K + k);
                sum = _mm256_fmadd_ps(a, b, sum);
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum, 1);
            __m128 lo = _mm256_castps256_ps128(sum);
            __m128 sum128 = _mm_add_ps(hi, lo);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float result = _mm_cvtss_f32(sum128);

            // Scalar remainder
            for (; k < K; ++k) {
                result += A[i * K + k] * B[j * K + k];
            }

            C[i * N + j] = result;
        }
    }

#else
    // Scalar fallback
    #pragma omp parallel for collapse(2) if(M * N > OMP_THRESHOLD_GEMM)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
#endif
}

/**
 * @brief Optimized GEMM for transposed A: C = A^T * B
 */
inline void gemm_transA_optimized(
    const float* __restrict__ A,  // A is (K x M), we compute A^T @ B
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t M, int64_t N, int64_t K
) {
    std::memset(C, 0, M * N * sizeof(float));

#ifdef TENZOR_GEMM_AVX2
    // Computes C[i,j] = sum_k A[k,i] * B[k,j]. Both A and B are accessed with
    // stride across k (rows), so every lane of the 8-wide AVX load needs to
    // gather values spaced M (for A) or N (for B) apart. The previous version
    // loaded B as 8 *consecutive* elements at fixed k (B[k,j..j+7]) which is
    // along the wrong axis — the FMA then multiplied A[k..k+7, i] by
    // B[k, j..j+7], producing silently-wrong sums. Conv2d weight gradients
    // on CPU were ~20% off as a result; fixed by gathering B the same way.
    #pragma omp parallel for collapse(2) if(M * N > OMP_THRESHOLD_GEMM)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            __m256 sum = _mm256_setzero_ps();
            int64_t k = 0;

            for (; k + 8 <= K; k += 8) {
                __m256 a = _mm256_set_ps(
                    A[(k+7) * M + i], A[(k+6) * M + i],
                    A[(k+5) * M + i], A[(k+4) * M + i],
                    A[(k+3) * M + i], A[(k+2) * M + i],
                    A[(k+1) * M + i], A[(k+0) * M + i]
                );
                __m256 b = _mm256_set_ps(
                    B[(k+7) * N + j], B[(k+6) * N + j],
                    B[(k+5) * N + j], B[(k+4) * N + j],
                    B[(k+3) * N + j], B[(k+2) * N + j],
                    B[(k+1) * N + j], B[(k+0) * N + j]
                );
                sum = _mm256_fmadd_ps(a, b, sum);
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum, 1);
            __m128 lo = _mm256_castps256_ps128(sum);
            __m128 sum128 = _mm_add_ps(hi, lo);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float result = _mm_cvtss_f32(sum128);

            for (; k < K; ++k) {
                result += A[k * M + i] * B[k * N + j];
            }

            C[i * N + j] = result;
        }
    }

#else
    #pragma omp parallel for collapse(2) if(M * N > OMP_THRESHOLD_GEMM)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#endif
}

} // namespace gemm
} // namespace cpu
} // namespace tenzor
