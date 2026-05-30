#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/backend/dtype_dispatch.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/widen_narrow.hpp"  // S2: Float16/BFloat16 widen-narrow helper
#include "tenzor/utils/log.hpp"   // TENZOR_LOG_WARN (F.4)
#include "broadcast.hpp"
#include <cstdlib>  // std::getenv for TENZOR_STRICT_BACKEND
#include "gemm_optimized.hpp"
#include "simd_fast_math.hpp"
#include "float16_simd.hpp"
#include "bfloat16_simd.hpp"
#include "int_simd.hpp"
#include "fp8_emulation.hpp"
#include "pointwise_kernel.hpp"
#include "simd_elementwise.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <limits>
#include <complex>
#include <functional>
#include <numeric>
#include <random>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define TENZOR_HAS_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define TENZOR_HAS_AVX2 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define TENZOR_HAS_SSE2 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// Intel MKL for optimized BLAS (5-10x faster GEMM) and VML transcendentals
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#include <mkl_service.h>
#include <mkl_vml.h>
#endif

// Intel oneDNN for optimized matrix operations (alternative to MKL)
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#include "onednn_cache.hpp"
#include <list>
#include <unordered_map>
#endif

// OpenMP thresholds — uses shared definition with env var override support.
// See omp_thresholds.hpp for TENZOR_OMP_THRESHOLD_* environment variables.
#include "tenzor/backend/omp_thresholds.hpp"  // unified (F.5)

// Convenience macros — delegate to shared lazy-init function
// Audit item F.5: prefer the project-wide OmpThresholds (in
// tenzor::, not tenzor::cpu::) so there is a single source of truth.
#define OMP_THRESHOLD_SIMPLE  (::tenzor::OmpThresholds::simple())
#define OMP_THRESHOLD_MEDIUM  (::tenzor::OmpThresholds::medium())
#define OMP_THRESHOLD_COMPLEX (::tenzor::OmpThresholds::complex())
#define OMP_THRESHOLD_MATMUL  (::tenzor::OmpThresholds::matmul())

namespace tenzor {

// Free MKL internal buffers to prevent conflicts with other MKL users (e.g., PyTorch)
// Call this after a batch of Tenzor operations before using another library that uses MKL
void mkl_cleanup() {
#ifdef TENZOR_USE_MKL
    // Free all thread-local memory allocations
    mkl_thread_free_buffers();
    // Free all MKL internal memory
    mkl_free_buffers();
#endif
}

namespace cpu {

// Optimized cache block sizes for L1/L2/L3 hierarchy
// L1: 32KB, L2: 256-512KB, L3: 8-32MB (typical modern CPU)
//
// Strategy:
// - BLOCK_SIZE_K: Controls how much of A and B is accessed per iteration
//   Larger K blocks improve temporal locality for the reduction
// - BLOCK_SIZE_M/N: Control output tile size
//   Should fit in L2 along with corresponding A/B panels
constexpr size_t BLOCK_SIZE_M = 64;   // Output rows per block
constexpr size_t BLOCK_SIZE_N = 256;  // Output cols per block (wider for better B reuse)
constexpr size_t BLOCK_SIZE_K = 256;  // Reduction block (fits A panel in L2)

// Prefetch distances (in cache lines)
constexpr size_t PREFETCH_A = 64;
constexpr size_t PREFETCH_B = 128;

// Micro-kernel for small block multiplication (Float64)
#ifdef TENZOR_HAS_AVX512
__attribute__((target("avx512f")))
#elif defined(TENZOR_HAS_AVX2)
__attribute__((target("avx2,fma")))
#endif
static void matmul_microkernel_float64(
    const double* A, const double* B, double* C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {

#ifdef TENZOR_HAS_AVX512
    // AVX-512: Process 8 doubles at a time
    constexpr int64_t simd_width = 8;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; j += simd_width) {
            int64_t width = std::min(simd_width, N - j);

            if (width == simd_width) {
                __m512d c_vec = _mm512_loadu_pd(&C[i * ldc + j]);

                for (int64_t k = 0; k < K; ++k) {
                    // Prefetch next A and B values
                    if (k + PREFETCH_A < static_cast<size_t>(K)) {
                        _mm_prefetch(reinterpret_cast<const char*>(&A[i * lda + k + PREFETCH_A]), _MM_HINT_T0);
                    }
                    if (k + PREFETCH_B < static_cast<size_t>(K)) {
                        _mm_prefetch(reinterpret_cast<const char*>(&B[(k + PREFETCH_B) * ldb + j]), _MM_HINT_T0);
                    }

                    __m512d a_vec = _mm512_set1_pd(A[i * lda + k]);
                    __m512d b_vec = _mm512_loadu_pd(&B[k * ldb + j]);
                    c_vec = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
                }

                _mm512_storeu_pd(&C[i * ldc + j], c_vec);
            } else {
                for (int64_t jj = j; jj < j + width; ++jj) {
                    double sum = C[i * ldc + jj];
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A[i * lda + k] * B[k * ldb + jj];
                    }
                    C[i * ldc + jj] = sum;
                }
            }
        }
    }
#elif defined(TENZOR_HAS_AVX2)
    // AVX2: Process 4 doubles at a time
    constexpr int64_t simd_width = 4;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; j += simd_width) {
            int64_t width = std::min(simd_width, N - j);

            if (width == simd_width) {
                __m256d c_vec = _mm256_loadu_pd(&C[i * ldc + j]);

                for (int64_t k = 0; k < K; ++k) {
                    __m256d a_vec = _mm256_set1_pd(A[i * lda + k]);
                    __m256d b_vec = _mm256_loadu_pd(&B[k * ldb + j]);
                    c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                }

                _mm256_storeu_pd(&C[i * ldc + j], c_vec);
            } else {
                for (int64_t jj = j; jj < j + width; ++jj) {
                    double sum = C[i * ldc + jj];
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A[i * lda + k] * B[k * ldb + jj];
                    }
                    C[i * ldc + jj] = sum;
                }
            }
        }
    }
#else
    // Scalar fallback
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            double sum = C[i * ldc + j];
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
#endif
}

// ============================================================================
// oneDNN MatMul helper (provides optimized GEMM with better memory handling)
// ============================================================================
#ifdef TENZOR_USE_ONEDNN
// Use shared lazy-init accessors from onednn_cache.hpp to avoid static
// thread_local initialization issues in dlopen'd libraries.

// --------------------------------------------------------------------------
// MatMul Primitive Caching (eliminates ~1-5ms primitive creation overhead)
// --------------------------------------------------------------------------
struct MatMulCacheKey {
    int64_t M, N, K;

    bool operator==(const MatMulCacheKey& other) const {
        return M == other.M && N == other.N && K == other.K;
    }
};

struct MatMulCacheKeyHash {
    size_t operator()(const MatMulCacheKey& k) const {
        size_t h = std::hash<int64_t>{}(k.M);
        h ^= std::hash<int64_t>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct MatMulCachedPrimitive {
    dnnl::matmul prim;
    dnnl::memory::desc a_md, b_md, c_md;
};

static constexpr size_t MATMUL_CACHE_SIZE = 48;

// Use shared OneDNNPrimitiveCache template (O(1) LRU via splice, replaces O(n) list::remove)
using MatMulPrimitiveCache = OneDNNPrimitiveCache<MatMulCacheKey, MatMulCachedPrimitive,
                                                   MatMulCacheKeyHash, MATMUL_CACHE_SIZE>;

static thread_local MatMulPrimitiveCache g_matmul_cache;

#ifndef TENZOR_USE_MKL
// oneDNN matmul for Float32 with primitive caching (only used when MKL is unavailable)
static bool onednn_matmul_f32(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K) {

    // oneDNN matmul benefits over MKL SGEMM for medium-to-large matrices.
    // Threshold lowered from 4096x4096x2048 to 512x512x512 — benchmark validated
    // that primitive caching amortizes oneDNN's setup cost for this size range.
    if (M < 512 || N < 512 || K < 512) {
        return false;  // Fall back to MKL/custom GEMM
    }

    try {
        auto& engine = get_onednn_engine();
        auto& stream = get_onednn_stream();

        // Create cache key
        MatMulCacheKey cache_key{M, N, K};

        // Try to get cached primitive
        auto cached = g_matmul_cache.get(cache_key);

        if (!cached) {
            // Cache miss - create new primitive and cache it
            cached = std::make_shared<MatMulCachedPrimitive>();

            // Create memory descriptors for row-major matrices
            dnnl::memory::dims a_dims = {M, K};
            dnnl::memory::dims b_dims = {K, N};
            dnnl::memory::dims c_dims = {M, N};

            cached->a_md = dnnl::memory::desc(a_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
            cached->b_md = dnnl::memory::desc(b_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
            cached->c_md = dnnl::memory::desc(c_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

            // Create matmul primitive descriptor and primitive
            auto matmul_pd = dnnl::matmul::primitive_desc(engine, cached->a_md, cached->b_md, cached->c_md);
            cached->prim = dnnl::matmul(matmul_pd);

            // Store in cache
            g_matmul_cache.put(cache_key, cached);
        }

        // Create memory objects with user data (this is fast - just wraps pointers)
        auto a_mem = dnnl::memory(cached->a_md, engine, const_cast<float*>(A));
        auto b_mem = dnnl::memory(cached->b_md, engine, const_cast<float*>(B));
        auto c_mem = dnnl::memory(cached->c_md, engine, C);

        // Execute cached primitive
        cached->prim.execute(stream, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        });
        stream.wait();

        return true;
    } catch (const dnnl::error& e) {
        // Audit item F.4: log + honour TENZOR_STRICT_BACKEND.
        if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
            throw std::runtime_error(
                std::string("[GEMM] oneDNN matmul failed "
                            "(TENZOR_STRICT_BACKEND=1): ") + e.what());
        }
        TENZOR_LOG_WARN("[GEMM] oneDNN matmul failed ({}); using MKL/custom GEMM fallback",
                        e.what());
        return false;
    }
}
#endif // !TENZOR_USE_MKL
#endif // TENZOR_USE_ONEDNN

// High-performance matrix multiplication (Float32)
// Uses oneDNN or MKL SGEMM when available (5-10x faster), falls back to optimized GEMM
static void matmul_blocked_float32(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K) {

#ifdef TENZOR_USE_MKL
    // Always use MKL SGEMM - it's the fastest option for CPU matmul
    // MKL is highly optimized for all matrix sizes including small ones
    // Previous threshold (M*N*K > 1M) caused 7x slowdown for small matrices
    // like 128x128 because it fell back to slower custom GEMM
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<MKL_INT>(M),
        static_cast<MKL_INT>(N),
        static_cast<MKL_INT>(K),
        1.0f,
        A, static_cast<MKL_INT>(K),
        B, static_cast<MKL_INT>(N),
        0.0f,
        C, static_cast<MKL_INT>(N)
    );
    return;
#endif

#ifdef TENZOR_USE_ONEDNN
#ifndef TENZOR_USE_MKL
    // If MKL not available, try oneDNN as fallback
    if (onednn_matmul_f32(A, B, C, M, N, K)) {
        return;
    }
#endif
#endif

    // Fall back to custom optimized GEMM for very small matrices or when no BLAS available
    gemm::gemm_optimized(A, B, C, M, N, K, 1.0f, 0.0f);
}

// Cache-blocked matrix multiplication (Float64)
// Uses MKL DGEMM when available, falls back to blocked SIMD implementation
static void matmul_blocked_float64(
    const double* A, const double* B, double* C,
    int64_t M, int64_t N, int64_t K) {

#ifdef TENZOR_USE_MKL
    // Always use MKL DGEMM - optimized for all matrix sizes
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<MKL_INT>(M),
        static_cast<MKL_INT>(N),
        static_cast<MKL_INT>(K),
        1.0,
        A, static_cast<MKL_INT>(K),
        B, static_cast<MKL_INT>(N),
        0.0,
        C, static_cast<MKL_INT>(N)
    );
    return;
#endif

    // Zero-initialize output
    std::fill_n(C, M * N, 0.0);

    // Cache-friendly blocked algorithm
    for (int64_t ii = 0; ii < M; ii += BLOCK_SIZE_M) {
        int64_t i_end = std::min(ii + static_cast<int64_t>(BLOCK_SIZE_M), M);

        for (int64_t jj = 0; jj < N; jj += BLOCK_SIZE_N) {
            int64_t j_end = std::min(jj + static_cast<int64_t>(BLOCK_SIZE_N), N);

            for (int64_t kk = 0; kk < K; kk += BLOCK_SIZE_K) {
                int64_t k_end = std::min(kk + static_cast<int64_t>(BLOCK_SIZE_K), K);

                // Process block
                int64_t block_m = i_end - ii;
                int64_t block_n = j_end - jj;
                int64_t block_k = k_end - kk;

                matmul_microkernel_float64(
                    A + ii * K + kk,
                    B + kk * N + jj,
                    C + ii * N + jj,
                    block_m, block_n, block_k,
                    K, N, N
                );
            }
        }
    }
}

// Micro-kernel for small block multiplication (Int32)
// Scalar implementation for Phase 1
static void matmul_microkernel_int32(
    const int32_t* A, const int32_t* B, int32_t* C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {

    // Scalar implementation for integer matrix multiplication
    // Accumulate in int64 to avoid overflow from int32 * int32
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int64_t sum = static_cast<int64_t>(C[i * ldc + j]);
            for (int64_t k = 0; k < K; ++k) {
                sum += static_cast<int64_t>(A[i * lda + k]) * static_cast<int64_t>(B[k * ldb + j]);
            }
            // Saturate to int32 range
            if (sum > std::numeric_limits<int32_t>::max()) {
                sum = std::numeric_limits<int32_t>::max();
            } else if (sum < std::numeric_limits<int32_t>::min()) {
                sum = std::numeric_limits<int32_t>::min();
            }
            C[i * ldc + j] = static_cast<int32_t>(sum);
        }
    }
}

// Cache-blocked matrix multiplication (Int32) with OpenMP parallelization
static void matmul_blocked_int32(
    const int32_t* A, const int32_t* B, int32_t* C,
    int64_t M, int64_t N, int64_t K) {

    // Zero-initialize output
    std::fill_n(C, M * N, 0);

    // Cache-friendly blocked algorithm with OpenMP parallelization over row blocks
    #pragma omp parallel for collapse(2) if(M * N > ::tenzor::OmpThresholds::medium())
    for (int64_t ii = 0; ii < M; ii += BLOCK_SIZE_M) {
        for (int64_t jj = 0; jj < N; jj += BLOCK_SIZE_N) {
            int64_t i_end = std::min(ii + static_cast<int64_t>(BLOCK_SIZE_M), M);
            int64_t j_end = std::min(jj + static_cast<int64_t>(BLOCK_SIZE_N), N);

            for (int64_t kk = 0; kk < K; kk += BLOCK_SIZE_K) {
                int64_t k_end = std::min(kk + static_cast<int64_t>(BLOCK_SIZE_K), K);

                // Process block
                int64_t block_m = i_end - ii;
                int64_t block_n = j_end - jj;
                int64_t block_k = k_end - kk;

                matmul_microkernel_int32(
                    A + ii * K + kk,
                    B + kk * N + jj,
                    C + ii * N + jj,
                    block_m, block_n, block_k,
                    K, N, N
                );
            }
        }
    }
}

// ============================================================================
// Int8 SIMD matrix multiplication
// ============================================================================
// AVX-512 VNNI: _mm512_dpbusd_epi32 computes dot products of unsigned×signed
// int8 pairs with int32 accumulation. Processes 64 int8 elements per instruction.
// AVX2: _mm256_maddubs_epi16 + _mm256_madd_epi16 for 32 int8 elements.
// Accumulates in int32 to avoid overflow, saturates to int8 on output.

static void matmul_microkernel_int8(
    const int8_t* A, const int8_t* B, int32_t* C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {

#if defined(__AVX512VNNI__) || defined(__AVX2__)
    // Pack the (K, N) tile of B into a thread-local (N, K_padded) contiguous
    // buffer once per microkernel call. Inner K-loop then reads B_packed
    // sequentially instead of issuing a per-iteration scatter/gather. This
    // removes the dominant per-element cost of the previous SIMD path
    // (the per-j 64- or 32-byte `for kk { b_buf[kk] = B[(k+kk)*ldb+j] }`
    // gather that ran inside the hot K-loop). K is padded up to the SIMD
    // width so the inner loop can read past the real K with zeros, avoiding
    // a fast/slow tail split.
#if defined(__AVX512VNNI__)
    constexpr int64_t SIMD_K = 64;
#else
    constexpr int64_t SIMD_K = 32;
#endif
    const int64_t K_padded = (K + SIMD_K - 1) / SIMD_K * SIMD_K;
    thread_local std::vector<int8_t> b_packed_storage;
    if (static_cast<int64_t>(b_packed_storage.size()) < N * K_padded) {
        b_packed_storage.assign(static_cast<size_t>(N * K_padded), 0);
    } else {
        // Zero only the bytes we are about to write; the tail past K must
        // already be zero (vector grows but never shrinks; previously zeroed
        // regions stay zero, and we re-zero the live region per call).
        std::memset(b_packed_storage.data(), 0,
                    static_cast<size_t>(N * K_padded));
    }
    int8_t* B_packed = b_packed_storage.data();
    for (int64_t j = 0; j < N; ++j) {
        int8_t* dst_col = B_packed + j * K_padded;
        for (int64_t k = 0; k < K; ++k) {
            dst_col[k] = B[k * ldb + j];
        }
        // dst_col[K..K_padded) is already 0 (memset above), which is a
        // mathematical identity for the dot product.
    }

    // Per-column bias correction (sum of B column entries) for the
    // unsigned-A offset trick used by VNNI / maddubs. Computed once,
    // reused for every i in the outer loop.
    thread_local std::vector<int32_t> b_col_sum_storage;
    if (static_cast<int64_t>(b_col_sum_storage.size()) < N) {
        b_col_sum_storage.assign(static_cast<size_t>(N), 0);
    }
    int32_t* b_col_sum = b_col_sum_storage.data();
    for (int64_t j = 0; j < N; ++j) {
        int32_t s = 0;
        const int8_t* col = B_packed + j * K_padded;
        for (int64_t k = 0; k < K; ++k) s += static_cast<int32_t>(col[k]);
        b_col_sum[j] = s;
    }
#endif

#ifdef __AVX512VNNI__
    // AVX-512 VNNI path: _mm512_dpbusd_epi32 takes unsigned×signed int8 pairs.
    // A is treated as unsigned via A_u = A_s + 128; the bias is corrected at
    // the end as -128 * sum(B_col[0..K)). B is read from the packed
    // contiguous panel so the inner K-loop is a stream of aligned vector
    // loads. The SIMD pass covers the floor-aligned K region; a scalar tail
    // handles the [K_simd, K) bytes — A is never read past K (the caller
    // may not have allocated more than K-wide rows).
    const int64_t K_simd = (K / 64) * 64;
    for (int64_t i = 0; i < M; ++i) {
        const int8_t* a_row = A + i * lda;
        for (int64_t j = 0; j < N; ++j) {
            __m512i acc = _mm512_setzero_si512();
            const int8_t* b_col = B_packed + j * K_padded;
            int64_t k = 0;
            for (; k < K_simd; k += 64) {
                __m512i a_s = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_row + k));
                __m512i offset = _mm512_set1_epi8(static_cast<char>(-128));
                __m512i a_u = _mm512_sub_epi8(a_s, offset);
                __m512i b_val = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_col + k));
                acc = _mm512_dpbusd_epi32(acc, a_u, b_val);
            }
            int32_t sum = _mm512_reduce_add_epi32(acc);
            // Scalar tail [K_simd, K). Uses packed-B for cache locality.
            for (; k < K; ++k) {
                sum += static_cast<int32_t>(static_cast<uint8_t>(a_row[k] ^ 0x80))
                       * static_cast<int32_t>(b_col[k]);
            }
            // Bias correction: (a_u - 128) * b summed = a_u*b - 128*b; the
            // unsigned a_u was applied to the full [0, K) region, so the
            // correction is -128 * sum(B_col[0..K)).
            sum -= 128 * b_col_sum[j];
            C[i * ldc + j] += sum;
        }
    }

#elif defined(__AVX2__)
    // AVX2 path: same packed-B layout, same offset trick. maddubs produces
    // saturating int16, then madd horizontally pairs into int32. Scalar
    // tail again handles bytes past the SIMD-aligned floor of K so we never
    // read A past its real K width.
    const int64_t K_simd = (K / 32) * 32;
    for (int64_t i = 0; i < M; ++i) {
        const int8_t* a_row = A + i * lda;
        for (int64_t j = 0; j < N; ++j) {
            __m256i acc = _mm256_setzero_si256();
            const int8_t* b_col = B_packed + j * K_padded;
            int64_t k = 0;
            for (; k < K_simd; k += 32) {
                __m256i a_s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_row + k));
                __m256i offset = _mm256_set1_epi8(static_cast<char>(-128));
                __m256i a_u = _mm256_sub_epi8(a_s, offset);
                __m256i b_val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_col + k));
                __m256i prod16 = _mm256_maddubs_epi16(a_u, b_val);
                __m256i ones = _mm256_set1_epi16(1);
                __m256i prod32 = _mm256_madd_epi16(prod16, ones);
                acc = _mm256_add_epi32(acc, prod32);
            }
            __m128i lo128 = _mm256_castsi256_si128(acc);
            __m128i hi128 = _mm256_extracti128_si256(acc, 1);
            __m128i sum128 = _mm_add_epi32(lo128, hi128);
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 1, 0, 1)));
            int32_t sum = _mm_cvtsi128_si32(sum128);
            for (; k < K; ++k) {
                sum += static_cast<int32_t>(static_cast<uint8_t>(a_row[k] ^ 0x80))
                       * static_cast<int32_t>(b_col[k]);
            }
            sum -= 128 * b_col_sum[j];
            C[i * ldc + j] += sum;
        }
    }

#else
    // Scalar fallback
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int32_t sum = C[i * ldc + j];
            for (int64_t k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A[i * lda + k]) * static_cast<int32_t>(B[k * ldb + j]);
            }
            C[i * ldc + j] = sum;
        }
    }
#endif
}

// MKL's `cblas_gemm_s8u8s32` cannot be used for general s8×s8 GEMM in the
// MKL 2025.3 build (it interprets A as u8 contrary to the docs). We avoid
// that path entirely and use oneDNN's s8×s8→s32 matmul when oneDNN is
// available (oneDNN handles signs correctly via its own packing). The
// blocked scalar / SIMD path remains the always-available fallback for
// small problems and for builds without oneDNN.

#ifdef TENZOR_USE_ONEDNN
// oneDNN int8 matmul: src and weights as s8, dst as s32. Returns true on
// success, false on any oneDNN error so the caller can fall back to the
// blocked SIMD path. Caching follows the same one-thread-local LRU model
// as the Float32 path above.
struct Int8MatMulCacheKey {
    int64_t M, N, K;
    bool operator==(const Int8MatMulCacheKey& o) const {
        return M == o.M && N == o.N && K == o.K;
    }
};
struct Int8MatMulCacheKeyHash {
    size_t operator()(const Int8MatMulCacheKey& k) const {
        size_t h = std::hash<int64_t>{}(k.M);
        h ^= std::hash<int64_t>{}(k.N) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.K) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};
struct Int8MatMulCachedPrimitive {
    dnnl::memory::desc a_md;
    dnnl::memory::desc b_md;
    dnnl::memory::desc c_md;
    dnnl::matmul prim;
};
using Int8MatMulPrimitiveCache =
    OneDNNPrimitiveCache<Int8MatMulCacheKey, Int8MatMulCachedPrimitive,
                          Int8MatMulCacheKeyHash, MATMUL_CACHE_SIZE>;
static thread_local Int8MatMulPrimitiveCache g_matmul_int8_cache;

// audit C1: register thread-local clear-callbacks for clear_dnnl_cache() so the
// matmul primitive caches are reclaimable (g_matmul_cache is declared above).
namespace {
void clear_local_matmul_cache() { g_matmul_cache.clear(); }
void clear_local_matmul_int8_cache() { g_matmul_int8_cache.clear(); }
struct MatMulCacheClearRegistrar {
    MatMulCacheClearRegistrar() {
        ::tenzor::cpu::register_dnnl_cache_clear_callback(&clear_local_matmul_cache);
        ::tenzor::cpu::register_dnnl_cache_clear_callback(&clear_local_matmul_int8_cache);
    }
};
static MatMulCacheClearRegistrar g_matmul_cache_clear_registrar;
}

static bool onednn_matmul_int8(
    const int8_t* A, const int8_t* B, int32_t* C,
    int64_t M, int64_t N, int64_t K) {
    // Smaller threshold than F32 path — Int8 GEMM has lower compute density
    // so the primitive setup overhead amortises at a smaller size.
    if (M < 256 || N < 256 || K < 256) {
        return false;
    }
    try {
        auto& engine = get_onednn_engine();
        auto& stream = get_onednn_stream();

        Int8MatMulCacheKey cache_key{M, N, K};
        auto cached = g_matmul_int8_cache.get(cache_key);
        if (!cached) {
            cached = std::make_shared<Int8MatMulCachedPrimitive>();
            dnnl::memory::dims a_dims = {M, K};
            dnnl::memory::dims b_dims = {K, N};
            dnnl::memory::dims c_dims = {M, N};
            cached->a_md = dnnl::memory::desc(a_dims,
                dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
            cached->b_md = dnnl::memory::desc(b_dims,
                dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
            cached->c_md = dnnl::memory::desc(c_dims,
                dnnl::memory::data_type::s32, dnnl::memory::format_tag::ab);
            auto matmul_pd = dnnl::matmul::primitive_desc(
                engine, cached->a_md, cached->b_md, cached->c_md);
            cached->prim = dnnl::matmul(matmul_pd);
            g_matmul_int8_cache.put(cache_key, cached);
        }
        auto a_mem = dnnl::memory(cached->a_md, engine, const_cast<int8_t*>(A));
        auto b_mem = dnnl::memory(cached->b_md, engine, const_cast<int8_t*>(B));
        auto c_mem = dnnl::memory(cached->c_md, engine, C);
        cached->prim.execute(stream, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        });
        stream.wait();
        return true;
    } catch (const dnnl::error& e) {
        if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
            throw std::runtime_error(
                std::string("[Int8 GEMM] oneDNN matmul failed "
                            "(TENZOR_STRICT_BACKEND=1): ") + e.what());
        }
        TENZOR_LOG_WARN("[Int8 GEMM] oneDNN matmul failed ({}); using scalar / SIMD fallback",
                        e.what());
        return false;
    }
}
#endif // TENZOR_USE_ONEDNN

// Cache-blocked matrix multiplication (Int8) with OpenMP parallelization
// Accumulates in int32 to avoid overflow, then saturates back to int8
static void matmul_blocked_int8(
    const int8_t* A, const int8_t* B, int8_t* C,
    int64_t M, int64_t N, int64_t K) {

    // Allocate int32 accumulator
    std::vector<int32_t> C_i32(M * N, 0);

#ifdef TENZOR_USE_ONEDNN
    // Try the oneDNN s8×s8→s32 fast path first. It returns false (no work
    // done, no state mutated) when below threshold or when oneDNN raises
    // — in either case we fall through to the blocked SIMD/scalar path.
    if (onednn_matmul_int8(A, B, C_i32.data(), M, N, K)) {
        for (int64_t i = 0; i < M * N; ++i) {
            int32_t val = C_i32[i];
            if (val > 127) val = 127;
            else if (val < -128) val = -128;
            C[i] = static_cast<int8_t>(val);
        }
        return;
    }
#endif

    // Cache-friendly blocked algorithm with OpenMP parallelization
    #pragma omp parallel for collapse(2) if(M * N > ::tenzor::OmpThresholds::medium())
    for (int64_t ii = 0; ii < M; ii += BLOCK_SIZE_M) {
        for (int64_t jj = 0; jj < N; jj += BLOCK_SIZE_N) {
            int64_t i_end = std::min(ii + static_cast<int64_t>(BLOCK_SIZE_M), M);
            int64_t j_end = std::min(jj + static_cast<int64_t>(BLOCK_SIZE_N), N);

            for (int64_t kk = 0; kk < K; kk += BLOCK_SIZE_K) {
                int64_t k_end = std::min(kk + static_cast<int64_t>(BLOCK_SIZE_K), K);

                int64_t block_m = i_end - ii;
                int64_t block_n = j_end - jj;
                int64_t block_k = k_end - kk;

                matmul_microkernel_int8(
                    A + ii * K + kk,
                    B + kk * N + jj,
                    C_i32.data() + ii * N + jj,
                    block_m, block_n, block_k,
                    K, N, N
                );
            }
        }
    }

    // Saturate int32 results to int8
    for (int64_t i = 0; i < M * N; ++i) {
        int32_t val = C_i32[i];
        if (val > 127) val = 127;
        else if (val < -128) val = -128;
        C[i] = static_cast<int8_t>(val);
    }
}

// High-performance Float16 matrix multiplication
// Uses F16C SIMD for conversion and FP32 GEMM for computation
static void matmul_blocked_float16(
    const Float16* A, const Float16* B, Float16* C,
    int64_t M, int64_t N, int64_t K
) {
    // Strategy: Convert blocks to FP32, use optimized GEMM, convert back
    // F16C provides fast SIMD conversion, FP32 GEMM provides high throughput

    // Block sizes for FP32 workspace (fit in L2 cache)
    constexpr int64_t TILE_M = 128;
    constexpr int64_t TILE_N = 128;
    constexpr int64_t TILE_K = 256;

    // Zero-initialize output
    std::fill_n(C, M * N, Float16(0.0f));

    // Use omp parallel to allocate per-thread FP32 workspace buffers on the heap.
    // 'static thread_local' vectors are NOT safe here because this code runs inside
    // a dlopen'd shared library, and OpenMP worker threads may bypass the C++ TLS
    // initialization machinery, leaving the vectors empty (data() == nullptr).
    #pragma omp parallel if(M * N > ::tenzor::OmpThresholds::simple())
    {
    std::vector<float> A_fp32_buf(TILE_M * TILE_K);
    std::vector<float> B_fp32_buf(TILE_K * TILE_N);
    std::vector<float> C_fp32_buf(TILE_M * TILE_N);

    #pragma omp for collapse(2)
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
            int64_t tile_m = std::min(TILE_M, M - i0);
            int64_t tile_n = std::min(TILE_N, N - j0);

            // Per-thread buffer pointers
            float* A_fp32 = A_fp32_buf.data();
            float* B_fp32 = B_fp32_buf.data();
            float* C_fp32 = C_fp32_buf.data();

            // Zero C tile accumulator
            std::fill_n(C_fp32, tile_m * tile_n, 0.0f);

            for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
                int64_t tile_k = std::min(TILE_K, K - k0);

                // Convert A tile to FP32 using F16C SIMD
                #ifdef __F16C__
                for (int64_t i = 0; i < tile_m; ++i) {
                    const Float16* a_row = A + (i0 + i) * K + k0;
                    float* a32_row = A_fp32 + i * tile_k;
                    int64_t k = 0;
                    for (; k + 8 <= tile_k; k += 8) {
                        __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_row + k));
                        __m256 unpacked = _mm256_cvtph_ps(packed);
                        _mm256_storeu_ps(a32_row + k, unpacked);
                    }
                    for (; k < tile_k; ++k) {
                        a32_row[k] = static_cast<float>(a_row[k]);
                    }
                }

                // Convert B tile to FP32 using F16C SIMD
                for (int64_t k = 0; k < tile_k; ++k) {
                    const Float16* b_row = B + (k0 + k) * N + j0;
                    float* b32_row = B_fp32 + k * tile_n;
                    int64_t j = 0;
                    for (; j + 8 <= tile_n; j += 8) {
                        __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_row + j));
                        __m256 unpacked = _mm256_cvtph_ps(packed);
                        _mm256_storeu_ps(b32_row + j, unpacked);
                    }
                    for (; j < tile_n; ++j) {
                        b32_row[j] = static_cast<float>(b_row[j]);
                    }
                }
                #else
                // Scalar fallback
                for (int64_t i = 0; i < tile_m; ++i) {
                    for (int64_t k = 0; k < tile_k; ++k) {
                        A_fp32[i * tile_k + k] = static_cast<float>(A[(i0 + i) * K + k0 + k]);
                    }
                }
                for (int64_t k = 0; k < tile_k; ++k) {
                    for (int64_t j = 0; j < tile_n; ++j) {
                        B_fp32[k * tile_n + j] = static_cast<float>(B[(k0 + k) * N + j0 + j]);
                    }
                }
                #endif

                // Compute FP32 GEMM (accumulate into C_fp32)
                // MKL > FMA micro-kernel > gemm_optimized > scalar fallback
#ifdef TENZOR_USE_MKL
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<MKL_INT>(tile_m), static_cast<MKL_INT>(tile_n), static_cast<MKL_INT>(tile_k),
                            1.0f, A_fp32, static_cast<MKL_INT>(tile_k),
                            B_fp32, static_cast<MKL_INT>(tile_n),
                            1.0f, C_fp32, static_cast<MKL_INT>(tile_n));
#elif defined(__FMA__) && defined(__F16C__)
                // FMA micro-kernel: process 8 floats per iteration using fused multiply-add
                // Tuned for F16 tile sizes — avoids overhead of full gemm_optimized dispatch
                for (int64_t i = 0; i < tile_m; ++i) {
                    for (int64_t k = 0; k < tile_k; ++k) {
                        __m256 a_broadcast = _mm256_set1_ps(A_fp32[i * tile_k + k]);
                        const float* b_row = B_fp32 + k * tile_n;
                        float* c_row = C_fp32 + i * tile_n;
                        int64_t j = 0;
                        for (; j + 8 <= tile_n; j += 8) {
                            __m256 c_vec = _mm256_loadu_ps(c_row + j);
                            __m256 b_vec = _mm256_loadu_ps(b_row + j);
                            c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                            _mm256_storeu_ps(c_row + j, c_vec);
                        }
                        // Scalar tail
                        float a_val = A_fp32[i * tile_k + k];
                        for (; j < tile_n; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
#else
                gemm::gemm_optimized(A_fp32, B_fp32, C_fp32, tile_m, tile_n, tile_k, 1.0f, 1.0f);
#endif
            }

            // Convert C tile back to FP16 using F16C SIMD
            #ifdef __F16C__
            for (int64_t i = 0; i < tile_m; ++i) {
                Float16* c_row = C + (i0 + i) * N + j0;
                const float* c32_row = C_fp32 + i * tile_n;
                int64_t j = 0;
                for (; j + 8 <= tile_n; j += 8) {
                    __m256 fp32_vals = _mm256_loadu_ps(c32_row + j);
                    __m128i packed = _mm256_cvtps_ph(fp32_vals, _MM_FROUND_TO_NEAREST_INT);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(c_row + j), packed);
                }
                for (; j < tile_n; ++j) {
                    c_row[j] = Float16(c32_row[j]);
                }
            }
            #else
            // Scalar fallback
            for (int64_t i = 0; i < tile_m; ++i) {
                for (int64_t j = 0; j < tile_n; ++j) {
                    C[(i0 + i) * N + j0 + j] = Float16(C_fp32[i * tile_n + j]);
                }
            }
            #endif
        }
    }
    } // omp parallel
}

// High-performance BFloat16 matrix multiplication
// Uses scalar conversion BF16→FP32, optimized GEMM, FP32→BF16 conversion back
// On Sapphire Rapids+, uses native _mm512_dpbf16_ps for direct BF16 dot products

#if defined(__AVX512BF16__) && defined(__AVX512F__)

// ============================================================================
// AVX-512 BF16 native dot-product matmul (Sapphire Rapids+)
// ============================================================================
// Uses _mm512_dpbf16_ps which computes dot products of BF16 pairs with FP32
// accumulation in a single instruction. Each _mm512_dpbf16_ps processes 32
// BF16 elements (16 pairs from A and B), accumulating into 16 FP32 lanes.
//
// The instruction treats its __m512bh operands as vectors of 16 pairs of BF16
// values packed into 32-bit words. For each 32-bit lane i:
//   acc[i] += a_pair[i].lo * b_pair[i].lo + a_pair[i].hi * b_pair[i].hi
//
// To use this for matmul, we broadcast pairs of A elements and multiply against
// contiguous B elements, accumulating into the C row.

static void matmul_blocked_bfloat16(
    const BFloat16* A, const BFloat16* B, BFloat16* C,
    int64_t M, int64_t N, int64_t K
) {
    // Tile sizes tuned for L2 cache residency with FP32 accumulators
    // C tile is TILE_M * TILE_N * 4 bytes = 128 * 128 * 4 = 64KB (fits in L2)
    constexpr int64_t TILE_M = 128;
    constexpr int64_t TILE_N = 128;
    constexpr int64_t TILE_K = 256;

    // Zero-initialize output
    std::fill_n(C, M * N, BFloat16(0.0f));

    // Use omp parallel to allocate per-thread FP32 workspace buffers on the heap.
    // 'static thread_local' vectors are NOT safe here because this code runs inside
    // a dlopen'd shared library, and OpenMP worker threads may bypass the C++ TLS
    // initialization machinery, leaving the vectors empty (data() == nullptr).
    #pragma omp parallel if(M * N > ::tenzor::OmpThresholds::simple())
    {
    std::vector<float> C_fp32_buf_bf16(TILE_M * TILE_N);

    #pragma omp for collapse(2)
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
            int64_t tile_m = std::min(TILE_M, M - i0);
            int64_t tile_n = std::min(TILE_N, N - j0);

            float* C_fp32 = C_fp32_buf_bf16.data();

            // Zero C tile accumulator
            std::fill_n(C_fp32, tile_m * tile_n, 0.0f);

            for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
                int64_t tile_k = std::min(TILE_K, K - k0);

                // Native BF16 dot product: process K dimension in steps of 2
                // _mm512_dpbf16_ps operates on pairs of BF16 values packed
                // into 32-bit words, so we step through K by 2.
                for (int64_t i = 0; i < tile_m; ++i) {
                    const uint16_t* a_row = reinterpret_cast<const uint16_t*>(
                        A + (i0 + i) * K + k0);
                    float* c_row = C_fp32 + i * tile_n;

                    // Process N dimension in chunks of 16 (512-bit / 32-bit FP32)
                    int64_t j = 0;
                    for (; j + 16 <= tile_n; j += 16) {
                        // Load current FP32 accumulator for this C[i, j:j+16]
                        __m512 acc = _mm512_loadu_ps(c_row + j);

                        // Process K pairs: each dpbf16_ps handles 2 K-elements
                        int64_t k = 0;
                        for (; k + 2 <= tile_k; k += 2) {
                            // Broadcast a pair of A[i, k] and A[i, k+1] as a
                            // 32-bit word (two packed BF16 values) to all lanes
                            uint32_t a_pair;
                            std::memcpy(&a_pair, a_row + k, sizeof(uint32_t));
                            __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                                static_cast<int>(a_pair));

                            // Load 16 pairs of B[k, j:j+16] and B[k+1, j:j+16]
                            // interleaved as 32-bit words:
                            // We need {B[k,j], B[k+1,j]}, {B[k,j+1], B[k+1,j+1]}, ...
                            // But B is stored row-major, so B[k] and B[k+1] rows
                            // are in separate memory locations. We load both rows
                            // and interleave them.
                            const uint16_t* b_row0 = reinterpret_cast<const uint16_t*>(
                                B + (k0 + k) * N + j0 + j);
                            const uint16_t* b_row1 = reinterpret_cast<const uint16_t*>(
                                B + (k0 + k + 1) * N + j0 + j);

                            // Load 16 BF16 values from each B row
                            __m256i b0_raw = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i*>(b_row0));
                            __m256i b1_raw = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i*>(b_row1));

                            // Interleave: pack {b0[i], b1[i]} into 32-bit words
                            // unpacklo/hi on 16-bit elements gives us the
                            // interleaved pairs we need for dpbf16_ps
                            __m512i b0_ext = _mm512_cvtepu16_epi32(b0_raw);
                            __m512i b1_ext = _mm512_cvtepu16_epi32(b1_raw);
                            __m512i b_interleaved = _mm512_or_si512(
                                b0_ext, _mm512_slli_epi32(b1_ext, 16));
                            __m512bh b_bf16 = (__m512bh)b_interleaved;

                            // Native BF16 dot product with FP32 accumulation
                            acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
                        }

                        // Handle odd remaining K element (if tile_k is odd)
                        if (k < tile_k) {
                            // Single remaining K element: pair it with zero
                            uint16_t a_val = a_row[k];
                            uint32_t a_pair_last = static_cast<uint32_t>(a_val);
                            __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                                static_cast<int>(a_pair_last));

                            const uint16_t* b_row_last = reinterpret_cast<const uint16_t*>(
                                B + (k0 + k) * N + j0 + j);
                            __m256i b_raw = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i*>(b_row_last));
                            // Zero-extend B values into low 16 bits of 32-bit words
                            // (high 16 bits are zero, matching the zero in A's pair)
                            __m512i b_ext = _mm512_cvtepu16_epi32(b_raw);
                            __m512bh b_bf16 = (__m512bh)b_ext;

                            acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
                        }

                        _mm512_storeu_ps(c_row + j, acc);
                    }

                    // Scalar tail for remaining N columns (< 16)
                    for (; j < tile_n; ++j) {
                        float acc_scalar = c_row[j];
                        for (int64_t k = 0; k < tile_k; ++k) {
                            float a_val = bfloat16_simd::bf16_to_f32_scalar(a_row[k]);
                            float b_val = bfloat16_simd::bf16_to_f32_scalar(
                                reinterpret_cast<const uint16_t*>(
                                    B + (k0 + k) * N + j0)[j]);
                            acc_scalar += a_val * b_val;
                        }
                        c_row[j] = acc_scalar;
                    }
                }
            }

            // Convert C tile: FP32 → BFloat16 using SIMD batch conversion
            for (int64_t i = 0; i < tile_m; ++i) {
                BFloat16* c_row = C + (i0 + i) * N + j0;
                const float* c32_row = C_fp32 + i * tile_n;
                bfloat16_simd::convert_f32_to_bf16_batch(c32_row, c_row,
                    static_cast<size_t>(tile_n));
            }
        }
    }
    } // omp parallel
}

#else // !(__AVX512BF16__ && __AVX512F__)

// ============================================================================
// Conversion-based BFloat16 matmul fallback
// ============================================================================
// Converts BF16 tiles to FP32 using SIMD (AVX-512/AVX2) or scalar,
// performs FP32 GEMM via MKL/optimized kernel, converts result back.

static void matmul_blocked_bfloat16(
    const BFloat16* A, const BFloat16* B, BFloat16* C,
    int64_t M, int64_t N, int64_t K
) {
    // Block sizes for FP32 workspace (fit in L2 cache)
    constexpr int64_t TILE_M = 128;
    constexpr int64_t TILE_N = 128;
    constexpr int64_t TILE_K = 256;

    // Zero-initialize output
    std::fill_n(C, M * N, BFloat16(0.0f));

    // Use omp parallel to allocate per-thread FP32 workspace buffers on the heap.
    // 'static thread_local' vectors are NOT safe here because this code runs inside
    // a dlopen'd shared library, and OpenMP worker threads may bypass the C++ TLS
    // initialization machinery, leaving the vectors empty (data() == nullptr).
    #pragma omp parallel if(M * N > ::tenzor::OmpThresholds::simple())
    {
    std::vector<float> A_fp32_buf_bf16(TILE_M * TILE_K);
    std::vector<float> B_fp32_buf_bf16(TILE_K * TILE_N);
    std::vector<float> C_fp32_buf_bf16(TILE_M * TILE_N);

    #pragma omp for collapse(2)
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
            int64_t tile_m = std::min(TILE_M, M - i0);
            int64_t tile_n = std::min(TILE_N, N - j0);

            // Per-thread buffer pointers
            float* A_fp32 = A_fp32_buf_bf16.data();
            float* B_fp32 = B_fp32_buf_bf16.data();
            float* C_fp32 = C_fp32_buf_bf16.data();

            // Zero C tile accumulator
            std::fill_n(C_fp32, tile_m * tile_n, 0.0f);

            for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
                int64_t tile_k = std::min(TILE_K, K - k0);

                // Convert A tile: BFloat16 → Float32 using SIMD batch conversion
                for (int64_t i = 0; i < tile_m; ++i) {
                    const BFloat16* a_row = A + (i0 + i) * K + k0;
                    float* a32_row = A_fp32 + i * tile_k;
                    bfloat16_simd::convert_bf16_to_f32_batch(a_row, a32_row,
                        static_cast<size_t>(tile_k));
                }

                // Convert B tile: BFloat16 → Float32 using SIMD batch conversion
                for (int64_t k = 0; k < tile_k; ++k) {
                    const BFloat16* b_row = B + (k0 + k) * N + j0;
                    float* b32_row = B_fp32 + k * tile_n;
                    bfloat16_simd::convert_bf16_to_f32_batch(b_row, b32_row,
                        static_cast<size_t>(tile_n));
                }

                // Compute FP32 GEMM (accumulate into C_fp32)
#ifdef TENZOR_USE_MKL
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<MKL_INT>(tile_m), static_cast<MKL_INT>(tile_n), static_cast<MKL_INT>(tile_k),
                            1.0f, A_fp32, static_cast<MKL_INT>(tile_k),
                            B_fp32, static_cast<MKL_INT>(tile_n),
                            1.0f, C_fp32, static_cast<MKL_INT>(tile_n));
#else
                gemm::gemm_optimized(A_fp32, B_fp32, C_fp32, tile_m, tile_n, tile_k, 1.0f, 1.0f);
#endif
            }

            // Convert C tile back: Float32 → BFloat16 using SIMD batch conversion
            for (int64_t i = 0; i < tile_m; ++i) {
                BFloat16* c_row = C + (i0 + i) * N + j0;
                const float* c32_row = C_fp32 + i * tile_n;
                bfloat16_simd::convert_f32_to_bf16_batch(c32_row, c_row,
                    static_cast<size_t>(tile_n));
            }
        }
    }
    } // end #pragma omp parallel
}

#endif // __AVX512BF16__ && __AVX512F__

// ============================================================================
// Element-wise Operations Helpers
// ============================================================================

namespace detail {

// Validate tensors for element-wise operations (with broadcasting support)
inline void validate_elementwise(const Tensor& a, const Tensor& b) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("Tensors must have same dtype");
    }

    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error("Element-wise operations require contiguous tensors");
    }

    // Check if shapes are broadcastable
    auto shape_a = a.shape();
    auto shape_b = b.shape();

    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    if (!are_broadcastable(shape_a_vec, shape_b_vec)) {
        std::string msg = "Tensors shapes are not broadcastable: [";
        for (size_t i = 0; i < shape_a_vec.size(); ++i) {
            msg += std::to_string(shape_a_vec[i]);
            if (i < shape_a_vec.size() - 1) msg += ", ";
        }
        msg += "] vs [";
        for (size_t i = 0; i < shape_b_vec.size(); ++i) {
            msg += std::to_string(shape_b_vec[i]);
            if (i < shape_b_vec.size() - 1) msg += ", ";
        }
        msg += "]";
        throw std::runtime_error(msg);
    }
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

// Scalar implementations
template<typename T>
inline void add_scalar(const T* a, const T* b, T* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

template<typename T>
inline void sub_scalar(const T* a, const T* b, T* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

// Specialization for Float16
template<>
inline void sub_scalar<Float16>(const Float16* a, const Float16* b, Float16* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = Float16(static_cast<float>(a[i]) - static_cast<float>(b[i]));
    }
}

template<typename T>
inline void mul_scalar(const T* a, const T* b, T* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

template<typename T>
inline void div_scalar(const T* a, const T* b, T* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_integral_v<T>) {
            if (b[i] == T(0)) {
                throw std::runtime_error("Integer division by zero");
            }
        }
        // IEEE 754: let hardware produce correct results for FP division
        // 0/0 = NaN, x/0 = +/-Inf (sign matches x)
        c[i] = a[i] / b[i];
    }
}

// SIMD implementations for Float32
#ifdef TENZOR_HAS_AVX512

__attribute__((target("avx512f")))
inline void add_avx512_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    const size_t simd_width = 16;
    const size_t simd_end = (n / simd_width) * simd_width;

    // OpenMP parallel for large tensors
    #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }

    // Handle remaining elements (sequential, small)
    for (size_t i = simd_end; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__attribute__((target("avx512f")))
inline void sub_avx512_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    const size_t simd_width = 16;
    const size_t simd_end = (n / simd_width) * simd_width;

    #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_sub_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }

    for (size_t i = simd_end; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

__attribute__((target("avx512f")))
inline void mul_avx512_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    const size_t simd_width = 16;
    const size_t simd_end = (n / simd_width) * simd_width;

    #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }

    for (size_t i = simd_end; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

__attribute__((target("avx512f")))
inline void div_avx512_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    const size_t simd_width = 16;
    const size_t simd_end = (n / simd_width) * simd_width;

    #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_div_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }

    // Tail loop: IEEE 754 handles 0/0=NaN, x/0=Inf naturally
    for (size_t i = simd_end; i < n; ++i) {
        c[i] = a[i] / b[i];
    }
}

// SIMD implementations for Float64
__attribute__((target("avx512f")))
inline void add_avx512_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d vc = _mm512_add_pd(va, vb);
        _mm512_storeu_pd(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__attribute__((target("avx512f")))
inline void sub_avx512_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d vc = _mm512_sub_pd(va, vb);
        _mm512_storeu_pd(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

__attribute__((target("avx512f")))
inline void mul_avx512_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d vc = _mm512_mul_pd(va, vb);
        _mm512_storeu_pd(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

__attribute__((target("avx512f")))
inline void div_avx512_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d vc = _mm512_div_pd(va, vb);
        _mm512_storeu_pd(c + i, vc);
    }

    // Tail loop: IEEE 754 handles 0/0=NaN, x/0=Inf naturally
    for (; i < n; ++i) {
        c[i] = a[i] / b[i];
    }
}

// SIMD implementations for Int32
__attribute__((target("avx512f")))
inline void add_avx512_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 16;

    for (; i + simd_width <= n; i += simd_width) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        __m512i vc = _mm512_add_epi32(va, vb);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(c + i), vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__attribute__((target("avx512f")))
inline void sub_avx512_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 16;

    for (; i + simd_width <= n; i += simd_width) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        __m512i vc = _mm512_sub_epi32(va, vb);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(c + i), vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

__attribute__((target("avx512f")))
inline void mul_avx512_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 16;

    for (; i + simd_width <= n; i += simd_width) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        __m512i vc = _mm512_mullo_epi32(va, vb);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(c + i), vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

// SIMD implementations for Int64
__attribute__((target("avx512f")))
inline void add_avx512_i64(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        __m512i vc = _mm512_add_epi64(va, vb);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(c + i), vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__attribute__((target("avx512f")))
inline void sub_avx512_i64(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        __m512i vc = _mm512_sub_epi64(va, vb);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(c + i), vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

#elif defined(TENZOR_HAS_AVX2)

// AVX2 fallback implementations
__attribute__((target("avx2,fma")))
inline void add_avx2_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    const size_t simd_width = 8;
    const size_t simd_end = (n / simd_width) * simd_width;

    // OpenMP parallel for large tensors
    #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }

    // Handle remainder
    for (size_t i = simd_end; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__attribute__((target("avx2,fma")))
inline void sub_avx2_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    const size_t simd_width = 8;
    const size_t simd_end = (n / simd_width) * simd_width;

    #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }

    for (size_t i = simd_end; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

__attribute__((target("avx2,fma")))
inline void mul_avx2_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    const size_t simd_width = 8;
    const size_t simd_end = (n / simd_width) * simd_width;

    #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }

    for (size_t i = simd_end; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

__attribute__((target("avx2,fma")))
inline void div_avx2_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    const size_t simd_width = 8;
    const size_t simd_end = (n / simd_width) * simd_width;

    #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }

    // Tail loop: IEEE 754 handles 0/0=NaN, x/0=Inf naturally
    for (size_t i = simd_end; i < n; ++i) {
        c[i] = a[i] / b[i];
    }
}

__attribute__((target("avx2,fma")))
inline void add_avx2_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 4;

    for (; i + simd_width <= n; i += simd_width) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__attribute__((target("avx2,fma")))
inline void sub_avx2_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 4;

    for (; i + simd_width <= n; i += simd_width) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

__attribute__((target("avx2,fma")))
inline void mul_avx2_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 4;

    for (; i + simd_width <= n; i += simd_width) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

__attribute__((target("avx2,fma")))
inline void div_avx2_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 4;

    for (; i + simd_width <= n; i += simd_width) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_div_pd(va, vb);
        _mm256_storeu_pd(c + i, vc);
    }

    // Tail loop: IEEE 754 handles 0/0=NaN, x/0=Inf naturally
    for (; i < n; ++i) {
        c[i] = a[i] / b[i];
    }
}

#endif // TENZOR_HAS_AVX512 / TENZOR_HAS_AVX2

} // namespace detail

// ============================================================================
// CPU math kernels
// ============================================================================

auto add_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);
    return cpu::binary_pointwise_kernel<cpu::AddOp>(a, b, cpu::fp8_add_emulated);
}

auto sub_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);
    return cpu::binary_pointwise_kernel<cpu::SubOp>(a, b, cpu::fp8_sub_emulated);
}

auto mul_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);
    return cpu::binary_pointwise_kernel<cpu::MulOp>(a, b, cpu::fp8_mul_emulated);
}


auto div_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);
    return cpu::binary_pointwise_kernel<cpu::DivOp>(a, b, cpu::fp8_div_emulated);
}


auto matmul_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    // FP8 emulation: widen to Float32, GEMM, narrow back
    if (cpu::is_fp8(a.dtype()) && cpu::is_fp8(b.dtype())) {
        return cpu::fp8_gemm_emulated(a, b);
    }

    // Make tensors contiguous if needed (does not break autograd chain)
    Tensor a_contig = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contig = b.is_contiguous() ? b : b.contiguous();

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

        // Treat 1D vector as row vector (1, N) and perform matmul to get (1, K), then return as (K,)
        int64_t M = 1;

        // Create output tensor with 1D shape
        Tensor result({K}, a_contig.dtype(), a_contig.device());

        // Dispatch based on dtype
        // For 1D×2D: A is (1, N), B is (N, K), C is (1, K)
        // Function signature expects (M, N_param, K_param) where:
        //   M = rows of A/C (=1), N_param = cols of B/C (=K), K_param = shared dim (=N)
        if (a_contig.dtype() == DType::Float32 && b_contig.dtype() == DType::Float32) {
            const float* a_data = a_contig.data<float>();
            const float* b_data = b_contig.data<float>();
            float* c_data = result.data<float>();

            matmul_blocked_float32(a_data, b_data, c_data, M, K, N);

        } else if (a_contig.dtype() == DType::Float64 && b_contig.dtype() == DType::Float64) {
            const double* a_data = a_contig.data<double>();
            const double* b_data = b_contig.data<double>();
            double* c_data = result.data<double>();

            matmul_blocked_float64(a_data, b_data, c_data, M, K, N);

        } else if (a_contig.dtype() == DType::Int32 && b_contig.dtype() == DType::Int32) {
            const int32_t* a_data = a_contig.data<int32_t>();
            const int32_t* b_data = b_contig.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();

            matmul_blocked_int32(a_data, b_data, c_data, M, K, N);

        } else if (a_contig.dtype() == DType::Int8 && b_contig.dtype() == DType::Int8) {
            const int8_t* a_data = a_contig.data<int8_t>();
            const int8_t* b_data = b_contig.data<int8_t>();
            int8_t* c_data = result.data<int8_t>();

            matmul_blocked_int8(a_data, b_data, c_data, M, K, N);

        } else if (a_contig.dtype() == DType::Float16 && b_contig.dtype() == DType::Float16) {
            const Float16* a_data = a_contig.data<Float16>();
            const Float16* b_data = b_contig.data<Float16>();
            Float16* c_data = result.data<Float16>();

            matmul_blocked_float16(a_data, b_data, c_data, M, K, N);

        } else if (a_contig.dtype() == DType::BFloat16 && b_contig.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a_contig.data<BFloat16>();
            const BFloat16* b_data = b_contig.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();

#if defined(TENZOR_USE_MKL)
            // Task 6.4: use MKL gemm_bf16bf16f32 (BF16 inputs, FP32 accumulation).
            // MKL_BF16 is typedef'd as unsigned short, bit-compatible with BFloat16.
            //
            // Data is row-major. BLAS is column-major. The standard trick:
            //   row-major C = A * B   ↔   col-major C^T = B^T * A^T
            // So pass B as the first matrix and A as the second, both 'N' (no-transpose),
            // with dimensions m=N, n=M, k=K and leading dims lda=N, ldb=K, ldc=N.
            {
                std::vector<float> c_fp32(static_cast<size_t>(M) * N);
                const char transa = 'N', transb = 'N';
                MKL_INT im = static_cast<MKL_INT>(M);
                MKL_INT ik = static_cast<MKL_INT>(K);
                MKL_INT in = static_cast<MKL_INT>(N);
                float alpha = 1.0f, beta = 0.0f;
                // col-major: C^T(N×M) = B^T(N×K) * A^T(K×M)
                // m_arg=N, n_arg=M, k_arg=K, lda=N (leading dim of B), ldb=K, ldc=N
                gemm_bf16bf16f32(&transa, &transb,
                                 &in, &im, &ik,
                                 &alpha,
                                 reinterpret_cast<const MKL_BF16*>(b_data), &in,
                                 reinterpret_cast<const MKL_BF16*>(a_data), &ik,
                                 &beta, c_fp32.data(), &in);
                // Narrow F32 → BF16
                bfloat16_simd::convert_f32_to_bf16_batch(c_fp32.data(), c_data,
                                                          static_cast<size_t>(M) * N);
            }
#else
            matmul_blocked_bfloat16(a_data, b_data, c_data, M, K, N);
#endif

        } else if (a_contig.dtype() == DType::Int16 && b_contig.dtype() == DType::Int16) {
            const int16_t* a_data = a_contig.data<int16_t>();
            const int16_t* b_data = b_contig.data<int16_t>();
            int16_t* c_data = result.data<int16_t>();
            for (int64_t j = 0; j < K; ++j) {
                int32_t acc = 0;
                for (int64_t kk = 0; kk < N; ++kk)
                    acc += static_cast<int32_t>(a_data[kk]) * static_cast<int32_t>(b_data[kk * K + j]);
                c_data[j] = static_cast<int16_t>(acc);
            }
        } else if (a_contig.dtype() == DType::UInt16 && b_contig.dtype() == DType::UInt16) {
            const uint16_t* a_data = a_contig.data<uint16_t>();
            const uint16_t* b_data = b_contig.data<uint16_t>();
            uint16_t* c_data = result.data<uint16_t>();
            for (int64_t j = 0; j < K; ++j) {
                uint32_t acc = 0;
                for (int64_t kk = 0; kk < N; ++kk)
                    acc += static_cast<uint32_t>(a_data[kk]) * static_cast<uint32_t>(b_data[kk * K + j]);
                c_data[j] = static_cast<uint16_t>(acc);
            }
        } else if (a_contig.dtype() == DType::Int64 && b_contig.dtype() == DType::Int64) {
            const int64_t* a_data = a_contig.data<int64_t>();
            const int64_t* b_data = b_contig.data<int64_t>();
            int64_t* c_data = result.data<int64_t>();
            for (int64_t j = 0; j < K; ++j) {
                int64_t acc = 0;
                for (int64_t kk = 0; kk < N; ++kk)
                    acc += a_data[kk] * b_data[kk * K + j];
                c_data[j] = acc;
            }
        } else if (a_contig.dtype() == DType::UInt32 && b_contig.dtype() == DType::UInt32) {
            const uint32_t* a_data = a_contig.data<uint32_t>();
            const uint32_t* b_data = b_contig.data<uint32_t>();
            uint32_t* c_data = result.data<uint32_t>();
            for (int64_t j = 0; j < K; ++j) {
                uint64_t acc = 0;
                for (int64_t kk = 0; kk < N; ++kk)
                    acc += static_cast<uint64_t>(a_data[kk]) * static_cast<uint64_t>(b_data[kk * K + j]);
                c_data[j] = static_cast<uint32_t>(acc);
            }
        } else if (a_contig.dtype() == DType::UInt64 && b_contig.dtype() == DType::UInt64) {
            const uint64_t* a_data = a_contig.data<uint64_t>();
            const uint64_t* b_data = b_contig.data<uint64_t>();
            uint64_t* c_data = result.data<uint64_t>();
            for (int64_t j = 0; j < K; ++j) {
                uint64_t acc = 0;
                for (int64_t kk = 0; kk < N; ++kk)
                    acc += a_data[kk] * b_data[kk * K + j];
                c_data[j] = acc;
            }
        } else {
            throw std::runtime_error(
                "matmul unsupported dtype combination: " +
                std::string(dtype_name(a_contig.dtype())) + " @ " +
                std::string(dtype_name(b_contig.dtype()))
            );
        }

        return result;
    }

    if (a_contig.ndim() != 2 || b_contig.ndim() != 2) {
        throw std::runtime_error("matmul requires 2D tensors (matrices)");
    }

    auto a_shape = a_contig.shape();
    auto b_shape = b_contig.shape();

    int64_t M = a_shape[0];  // Rows of A
    int64_t K = a_shape[1];  // Cols of A
    int64_t K2 = b_shape[0]; // Rows of B
    int64_t N = b_shape[1];  // Cols of B

    // Validate dimensions: (M×K) @ (K×N) → (M×N)
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

        matmul_blocked_float32(a_data, b_data, c_data, M, N, K);

    } else if (a_contig.dtype() == DType::Float64 && b_contig.dtype() == DType::Float64) {
        const double* a_data = a_contig.data<double>();
        const double* b_data = b_contig.data<double>();
        double* c_data = result.data<double>();

        matmul_blocked_float64(a_data, b_data, c_data, M, N, K);

    } else if (a_contig.dtype() == DType::Int32 && b_contig.dtype() == DType::Int32) {
        const int32_t* a_data = a_contig.data<int32_t>();
        const int32_t* b_data = b_contig.data<int32_t>();
        int32_t* c_data = result.data<int32_t>();

        matmul_blocked_int32(a_data, b_data, c_data, M, N, K);

    } else if (a_contig.dtype() == DType::Int8 && b_contig.dtype() == DType::Int8) {
        const int8_t* a_data = a_contig.data<int8_t>();
        const int8_t* b_data = b_contig.data<int8_t>();
        int8_t* c_data = result.data<int8_t>();

        matmul_blocked_int8(a_data, b_data, c_data, M, N, K);

    } else if (a_contig.dtype() == DType::Float16 && b_contig.dtype() == DType::Float16) {
        const Float16* a_data = a_contig.data<Float16>();
        const Float16* b_data = b_contig.data<Float16>();
        Float16* c_data = result.data<Float16>();

        matmul_blocked_float16(a_data, b_data, c_data, M, N, K);

    } else if (a_contig.dtype() == DType::BFloat16 && b_contig.dtype() == DType::BFloat16) {
        const BFloat16* a_data = a_contig.data<BFloat16>();
        const BFloat16* b_data = b_contig.data<BFloat16>();
        BFloat16* c_data = result.data<BFloat16>();

#if defined(TENZOR_USE_MKL)
        {
            std::vector<float> c_fp32(static_cast<size_t>(M) * N);
            const char transa = 'N', transb = 'N';
            MKL_INT im = static_cast<MKL_INT>(M);
            MKL_INT ik = static_cast<MKL_INT>(K);
            MKL_INT in = static_cast<MKL_INT>(N);
            float alpha = 1.0f, beta = 0.0f;
            // col-major trick: C^T = B^T * A^T
            gemm_bf16bf16f32(&transa, &transb,
                             &in, &im, &ik,
                             &alpha,
                             reinterpret_cast<const MKL_BF16*>(b_data), &in,
                             reinterpret_cast<const MKL_BF16*>(a_data), &ik,
                             &beta, c_fp32.data(), &in);
            bfloat16_simd::convert_f32_to_bf16_batch(c_fp32.data(), c_data,
                                                      static_cast<size_t>(M) * N);
        }
#else
        matmul_blocked_bfloat16(a_data, b_data, c_data, M, N, K);
#endif

    } else if (a_contig.dtype() == DType::Complex64 && b_contig.dtype() == DType::Complex64) {
#ifdef TENZOR_USE_MKL
        MKL_Complex8 alpha = {1.0f, 0.0f};
        MKL_Complex8 beta = {0.0f, 0.0f};
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<MKL_INT>(M), static_cast<MKL_INT>(N), static_cast<MKL_INT>(K),
                    &alpha,
                    reinterpret_cast<const void*>(a_contig.data<std::complex<float>>()),
                    static_cast<MKL_INT>(K),
                    reinterpret_cast<const void*>(b_contig.data<std::complex<float>>()),
                    static_cast<MKL_INT>(N),
                    &beta,
                    reinterpret_cast<void*>(result.data<std::complex<float>>()),
                    static_cast<MKL_INT>(N));
#else
        // Fallback: naive complex matmul
        const auto* a_data = a_contig.data<std::complex<float>>();
        const auto* b_data = b_contig.data<std::complex<float>>();
        auto* c_data = result.data<std::complex<float>>();
        std::fill_n(c_data, M * N, std::complex<float>(0.0f, 0.0f));
        for (int64_t i = 0; i < M; ++i)
            for (int64_t k = 0; k < K; ++k)
                for (int64_t j = 0; j < N; ++j)
                    c_data[i * N + j] += a_data[i * K + k] * b_data[k * N + j];
#endif

    } else if (a_contig.dtype() == DType::Complex128 && b_contig.dtype() == DType::Complex128) {
#ifdef TENZOR_USE_MKL
        MKL_Complex16 alpha = {1.0, 0.0};
        MKL_Complex16 beta = {0.0, 0.0};
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<MKL_INT>(M), static_cast<MKL_INT>(N), static_cast<MKL_INT>(K),
                    &alpha,
                    reinterpret_cast<const void*>(a_contig.data<std::complex<double>>()),
                    static_cast<MKL_INT>(K),
                    reinterpret_cast<const void*>(b_contig.data<std::complex<double>>()),
                    static_cast<MKL_INT>(N),
                    &beta,
                    reinterpret_cast<void*>(result.data<std::complex<double>>()),
                    static_cast<MKL_INT>(N));
#else
        // Fallback: naive complex matmul
        const auto* a_data = a_contig.data<std::complex<double>>();
        const auto* b_data = b_contig.data<std::complex<double>>();
        auto* c_data = result.data<std::complex<double>>();
        std::fill_n(c_data, M * N, std::complex<double>(0.0, 0.0));
        for (int64_t i = 0; i < M; ++i)
            for (int64_t k = 0; k < K; ++k)
                for (int64_t j = 0; j < N; ++j)
                    c_data[i * N + j] += a_data[i * K + k] * b_data[k * N + j];
#endif

    } else if (a_contig.dtype() == DType::Int16 && b_contig.dtype() == DType::Int16) {
        const int16_t* a_data = a_contig.data<int16_t>();
        const int16_t* b_data = b_contig.data<int16_t>();
        int16_t* c_data = result.data<int16_t>();
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) {
                int32_t acc = 0;
                for (int64_t k = 0; k < K; ++k)
                    acc += static_cast<int32_t>(a_data[i * K + k]) * static_cast<int32_t>(b_data[k * N + j]);
                c_data[i * N + j] = static_cast<int16_t>(acc);
            }

    } else if (a_contig.dtype() == DType::UInt16 && b_contig.dtype() == DType::UInt16) {
        const uint16_t* a_data = a_contig.data<uint16_t>();
        const uint16_t* b_data = b_contig.data<uint16_t>();
        uint16_t* c_data = result.data<uint16_t>();
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) {
                uint32_t acc = 0;
                for (int64_t k = 0; k < K; ++k)
                    acc += static_cast<uint32_t>(a_data[i * K + k]) * static_cast<uint32_t>(b_data[k * N + j]);
                c_data[i * N + j] = static_cast<uint16_t>(acc);
            }

    } else if (a_contig.dtype() == DType::Int64 && b_contig.dtype() == DType::Int64) {
        const int64_t* a_data = a_contig.data<int64_t>();
        const int64_t* b_data = b_contig.data<int64_t>();
        int64_t* c_data = result.data<int64_t>();
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) {
                int64_t acc = 0;
                for (int64_t k = 0; k < K; ++k)
                    acc += a_data[i * K + k] * b_data[k * N + j];
                c_data[i * N + j] = acc;
            }

    } else if (a_contig.dtype() == DType::UInt32 && b_contig.dtype() == DType::UInt32) {
        const uint32_t* a_data = a_contig.data<uint32_t>();
        const uint32_t* b_data = b_contig.data<uint32_t>();
        uint32_t* c_data = result.data<uint32_t>();
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) {
                uint64_t acc = 0;
                for (int64_t k = 0; k < K; ++k)
                    acc += static_cast<uint64_t>(a_data[i * K + k]) * static_cast<uint64_t>(b_data[k * N + j]);
                c_data[i * N + j] = static_cast<uint32_t>(acc);
            }

    } else if (a_contig.dtype() == DType::UInt64 && b_contig.dtype() == DType::UInt64) {
        const uint64_t* a_data = a_contig.data<uint64_t>();
        const uint64_t* b_data = b_contig.data<uint64_t>();
        uint64_t* c_data = result.data<uint64_t>();
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) {
                uint64_t acc = 0;
                for (int64_t k = 0; k < K; ++k)
                    acc += a_data[i * K + k] * b_data[k * N + j];
                c_data[i * N + j] = acc;
            }

    } else {
        throw std::runtime_error(
            "matmul unsupported dtype combination: " +
            std::string(dtype_name(a_contig.dtype())) + " @ " +
            std::string(dtype_name(b_contig.dtype()))
        );
    }

    return result;
}

// Batched matrix multiplication kernel (uses MKL batch SGEMM/DGEMM)
auto bmm_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    // Validate inputs are 3D
    if (a.shape().size() != 3 || b.shape().size() != 3) {
        throw std::runtime_error(
            "bmm_kernel requires 3D tensors, got shapes: [" +
            std::to_string(a.shape().size()) + "D] and [" +
            std::to_string(b.shape().size()) + "D]");
    }

    int64_t batch_size = a.shape()[0];
    int64_t M = a.shape()[1];  // rows of A
    int64_t K = a.shape()[2];  // cols of A = rows of B
    int64_t N = b.shape()[2];  // cols of B

    if (b.shape()[0] != batch_size || b.shape()[1] != K) {
        throw std::runtime_error(
            "bmm_kernel dimension mismatch: expected b.shape=[" +
            std::to_string(batch_size) + ", " + std::to_string(K) + ", *], got [" +
            std::to_string(b.shape()[0]) + ", " + std::to_string(b.shape()[1]) + ", " +
            std::to_string(b.shape()[2]) + "]");
    }

    // Make inputs contiguous
    Tensor a_cont = a.is_contiguous() ? a : a.contiguous();
    Tensor b_cont = b.is_contiguous() ? b : b.contiguous();

    // Create output tensor
    Tensor output = Tensor::empty_uninitialized({batch_size, M, N}, a.dtype(), Device::cpu());

    // Batch strides
    int64_t a_batch_stride = M * K;
    int64_t b_batch_stride = K * N;
    int64_t c_batch_stride = M * N;

    if (a.dtype() == DType::Float32) {
        const float* a_data = a_cont.data<float>();
        const float* b_data = b_cont.data<float>();
        float* c_data = output.data<float>();

#ifdef TENZOR_USE_MKL
        // Use batch_strided for efficient batched GEMM
        cblas_sgemm_batch_strided(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            static_cast<MKL_INT>(M),
            static_cast<MKL_INT>(N),
            static_cast<MKL_INT>(K),
            1.0f,
            a_data, static_cast<MKL_INT>(K), a_batch_stride,
            b_data, static_cast<MKL_INT>(N), b_batch_stride,
            0.0f,
            c_data, static_cast<MKL_INT>(N), c_batch_stride,
            static_cast<MKL_INT>(batch_size)
        );
#else
        // Fallback: Parallelize across batches with optimized GEMM
        #pragma omp parallel for if(batch_size > 1)
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* a_batch = a_data + batch * a_batch_stride;
            const float* b_batch = b_data + batch * b_batch_stride;
            float* c_batch = c_data + batch * c_batch_stride;

            gemm::gemm_optimized(a_batch, b_batch, c_batch, M, N, K, 1.0f, 0.0f);
        }
#endif
    } else if (a.dtype() == DType::Float64) {
        const double* a_data = a_cont.data<double>();
        const double* b_data = b_cont.data<double>();
        double* c_data = output.data<double>();

#ifdef TENZOR_USE_MKL
        cblas_dgemm_batch_strided(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            static_cast<MKL_INT>(M),
            static_cast<MKL_INT>(N),
            static_cast<MKL_INT>(K),
            1.0,
            a_data, static_cast<MKL_INT>(K), a_batch_stride,
            b_data, static_cast<MKL_INT>(N), b_batch_stride,
            0.0,
            c_data, static_cast<MKL_INT>(N), c_batch_stride,
            static_cast<MKL_INT>(batch_size)
        );
#else
        #pragma omp parallel for if(batch_size > 1)
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const double* a_batch = a_data + batch * a_batch_stride;
            const double* b_batch = b_data + batch * b_batch_stride;
            double* c_batch = c_data + batch * c_batch_stride;

            // Use tiled approach for better cache behavior
            constexpr int64_t TILE = 64;
            std::fill_n(c_batch, M * N, 0.0);
            for (int64_t ii = 0; ii < M; ii += TILE) {
                for (int64_t kk = 0; kk < K; kk += TILE) {
                    for (int64_t jj = 0; jj < N; jj += TILE) {
                        int64_t i_end = std::min(ii + TILE, M);
                        int64_t k_end = std::min(kk + TILE, K);
                        int64_t j_end = std::min(jj + TILE, N);
                        for (int64_t i = ii; i < i_end; ++i) {
                            for (int64_t k = kk; k < k_end; ++k) {
                                double a_val = a_batch[i * K + k];
                                for (int64_t j = jj; j < j_end; ++j) {
                                    c_batch[i * N + j] += a_val * b_batch[k * N + j];
                                }
                            }
                        }
                    }
                }
            }
        }
#endif
    } else if (a.dtype() == DType::Float16) {
        // Float16: Convert to Float32, compute, convert back
        // (MKL doesn't support Float16 BLAS, and CPU Float16 is slow anyway)
        const Float16* a_data = a_cont.data<Float16>();
        const Float16* b_data = b_cont.data<Float16>();
        Float16* c_data = output.data<Float16>();

        // Allocate temporary Float32 buffers
        size_t a_size = static_cast<size_t>(batch_size * M * K);
        size_t b_size = static_cast<size_t>(batch_size * K * N);
        size_t c_size = static_cast<size_t>(batch_size * M * N);

        std::vector<float> a_f32(a_size);
        std::vector<float> b_f32(b_size);
        std::vector<float> c_f32(c_size);

        // Convert Float16 to Float32
        for (size_t i = 0; i < a_size; ++i) {
            a_f32[i] = static_cast<float>(a_data[i]);
        }
        for (size_t i = 0; i < b_size; ++i) {
            b_f32[i] = static_cast<float>(b_data[i]);
        }

#ifdef TENZOR_USE_MKL
        cblas_sgemm_batch_strided(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            static_cast<MKL_INT>(M),
            static_cast<MKL_INT>(N),
            static_cast<MKL_INT>(K),
            1.0f,
            a_f32.data(), static_cast<MKL_INT>(K), a_batch_stride,
            b_f32.data(), static_cast<MKL_INT>(N), b_batch_stride,
            0.0f,
            c_f32.data(), static_cast<MKL_INT>(N), c_batch_stride,
            static_cast<MKL_INT>(batch_size)
        );
#else
        #pragma omp parallel for if(batch_size > 1)
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* a_batch = a_f32.data() + batch * a_batch_stride;
            const float* b_batch = b_f32.data() + batch * b_batch_stride;
            float* c_batch = c_f32.data() + batch * c_batch_stride;

            gemm::gemm_optimized(a_batch, b_batch, c_batch, M, N, K, 1.0f, 0.0f);
        }
#endif

        // Convert Float32 back to Float16
        for (size_t i = 0; i < c_size; ++i) {
            c_data[i] = Float16(c_f32[i]);
        }
    } else if (a.dtype() == DType::BFloat16) {
        // BFloat16: Convert to Float32, compute, convert back
        const BFloat16* a_data = a_cont.data<BFloat16>();
        const BFloat16* b_data = b_cont.data<BFloat16>();
        BFloat16* c_data = output.data<BFloat16>();

        size_t a_size = static_cast<size_t>(batch_size * M * K);
        size_t b_size = static_cast<size_t>(batch_size * K * N);
        size_t c_size = static_cast<size_t>(batch_size * M * N);

        std::vector<float> a_f32(a_size);
        std::vector<float> b_f32(b_size);
        std::vector<float> c_f32(c_size);

        // Convert BFloat16 to Float32
        for (size_t i = 0; i < a_size; ++i) {
            a_f32[i] = static_cast<float>(a_data[i]);
        }
        for (size_t i = 0; i < b_size; ++i) {
            b_f32[i] = static_cast<float>(b_data[i]);
        }

#ifdef TENZOR_USE_MKL
        cblas_sgemm_batch_strided(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            static_cast<MKL_INT>(M),
            static_cast<MKL_INT>(N),
            static_cast<MKL_INT>(K),
            1.0f,
            a_f32.data(), static_cast<MKL_INT>(K), a_batch_stride,
            b_f32.data(), static_cast<MKL_INT>(N), b_batch_stride,
            0.0f,
            c_f32.data(), static_cast<MKL_INT>(N), c_batch_stride,
            static_cast<MKL_INT>(batch_size)
        );
#else
        #pragma omp parallel for if(batch_size > 1)
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* a_batch = a_f32.data() + batch * a_batch_stride;
            const float* b_batch = b_f32.data() + batch * b_batch_stride;
            float* c_batch = c_f32.data() + batch * c_batch_stride;

            gemm::gemm_optimized(a_batch, b_batch, c_batch, M, N, K, 1.0f, 0.0f);
        }
#endif

        // Convert Float32 back to BFloat16
        for (size_t i = 0; i < c_size; ++i) {
            c_data[i] = BFloat16(c_f32[i]);
        }
    } else {
        // Generic fallback: loop matmul_kernel over batch dimension.
        // Handles Complex64, Complex128, Int16, UInt16, Int64, UInt32, UInt64, etc.
        size_t elem_bytes = output.element_size();
        size_t slice_bytes = static_cast<size_t>(M * N) * elem_bytes;
        char* out_base = static_cast<char*>(output.storage()->data());
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            auto a_slice = a_cont.narrow(0, batch, 1).reshape({M, K});
            auto b_slice = b_cont.narrow(0, batch, 1).reshape({K, N});
            auto c_slice = matmul_kernel(a_slice, b_slice);  // (M, N) contiguous result
            // c_slice is a freshly-created contiguous tensor; copy into output batch slot.
            std::memcpy(
                out_base + static_cast<ptrdiff_t>(batch) * static_cast<ptrdiff_t>(slice_bytes),
                c_slice.storage()->data(),
                slice_bytes
            );
        }
    }

    return output;
}

// Square root kernel (OpenMP + SIMD optimized with chunking)
auto sqrt_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

        // For small arrays, use single-threaded SIMD
        if (n < OMP_THRESHOLD_SIMPLE) {
#ifdef TENZOR_HAS_AVX512
            const size_t simd_width = 16;
            size_t i = 0;
            for (; i + simd_width <= n; i += simd_width) {
                __m512 v = _mm512_loadu_ps(&in_data[i]);
                _mm512_storeu_ps(&out_data[i], _mm512_sqrt_ps(v));
            }
            for (; i < n; ++i) out_data[i] = std::sqrt(in_data[i]);
#elif defined(TENZOR_HAS_AVX2)
            const size_t simd_width = 8;
            size_t i = 0;
            for (; i + simd_width <= n; i += simd_width) {
                __m256 v = _mm256_loadu_ps(&in_data[i]);
                _mm256_storeu_ps(&out_data[i], _mm256_sqrt_ps(v));
            }
            for (; i < n; ++i) out_data[i] = std::sqrt(in_data[i]);
#else
            for (size_t i = 0; i < n; ++i) out_data[i] = std::sqrt(in_data[i]);
#endif
        } else {
            // For large arrays, use OpenMP with thread-local SIMD chunks
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
                size_t chunk_size = (n + nthreads - 1) / nthreads;
                size_t start = tid * chunk_size;
                size_t end = std::min(start + chunk_size, n);

#ifdef TENZOR_HAS_AVX512
                const size_t simd_width = 16;
                size_t i = start;
                for (; i + simd_width <= end; i += simd_width) {
                    __m512 v = _mm512_loadu_ps(&in_data[i]);
                    _mm512_storeu_ps(&out_data[i], _mm512_sqrt_ps(v));
                }
                for (; i < end; ++i) out_data[i] = std::sqrt(in_data[i]);
#elif defined(TENZOR_HAS_AVX2)
                const size_t simd_width = 8;
                size_t i = start;
                for (; i + simd_width <= end; i += simd_width) {
                    __m256 v = _mm256_loadu_ps(&in_data[i]);
                    _mm256_storeu_ps(&out_data[i], _mm256_sqrt_ps(v));
                }
                for (; i < end; ++i) out_data[i] = std::sqrt(in_data[i]);
#else
                for (size_t i = start; i < end; ++i) out_data[i] = std::sqrt(in_data[i]);
#endif
            }
        }

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

        if (n < OMP_THRESHOLD_SIMPLE) {
#ifdef TENZOR_HAS_AVX512
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m512d v = _mm512_loadu_pd(&in_data[i]);
                _mm512_storeu_pd(&out_data[i], _mm512_sqrt_pd(v));
            }
            for (; i < n; ++i) out_data[i] = std::sqrt(in_data[i]);
#elif defined(TENZOR_HAS_AVX2)
            size_t i = 0;
            for (; i + 4 <= n; i += 4) {
                __m256d v = _mm256_loadu_pd(&in_data[i]);
                _mm256_storeu_pd(&out_data[i], _mm256_sqrt_pd(v));
            }
            for (; i < n; ++i) out_data[i] = std::sqrt(in_data[i]);
#else
            for (size_t i = 0; i < n; ++i) out_data[i] = std::sqrt(in_data[i]);
#endif
        } else {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
                size_t chunk_size = (n + nthreads - 1) / nthreads;
                size_t start = tid * chunk_size;
                size_t end = std::min(start + chunk_size, n);

#ifdef TENZOR_HAS_AVX512
                size_t i = start;
                for (; i + 8 <= end; i += 8) {
                    __m512d v = _mm512_loadu_pd(&in_data[i]);
                    _mm512_storeu_pd(&out_data[i], _mm512_sqrt_pd(v));
                }
                for (; i < end; ++i) out_data[i] = std::sqrt(in_data[i]);
#elif defined(TENZOR_HAS_AVX2)
                size_t i = start;
                for (; i + 4 <= end; i += 4) {
                    __m256d v = _mm256_loadu_pd(&in_data[i]);
                    _mm256_storeu_pd(&out_data[i], _mm256_sqrt_pd(v));
                }
                for (; i < end; ++i) out_data[i] = std::sqrt(in_data[i]);
#else
                for (size_t i = start; i < end; ++i) out_data[i] = std::sqrt(in_data[i]);
#endif
            }
        }

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();

        if (n < OMP_THRESHOLD_SIMPLE) {
#ifdef __F16C__
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_data + i));
                __m256 fp32 = _mm256_cvtph_ps(packed);
                __m256 result_v = _mm256_sqrt_ps(fp32);
                __m128i out_packed = _mm256_cvtps_ph(result_v, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out_data + i), out_packed);
            }
            for (; i < n; ++i) out_data[i] = Float16(std::sqrt(static_cast<float>(in_data[i])));
#else
            for (size_t i = 0; i < n; ++i) out_data[i] = Float16(std::sqrt(static_cast<float>(in_data[i])));
#endif
        } else {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
                size_t chunk_size = (n + nthreads - 1) / nthreads;
                size_t start = tid * chunk_size;
                size_t end = std::min(start + chunk_size, n);

#ifdef __F16C__
                size_t i = start;
                for (; i + 8 <= end; i += 8) {
                    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_data + i));
                    __m256 fp32 = _mm256_cvtph_ps(packed);
                    __m256 result_v = _mm256_sqrt_ps(fp32);
                    __m128i out_packed = _mm256_cvtps_ph(result_v, _MM_FROUND_TO_NEAREST_INT);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_data + i), out_packed);
                }
                for (; i < end; ++i) out_data[i] = Float16(std::sqrt(static_cast<float>(in_data[i])));
#else
                for (size_t i = start; i < end; ++i) out_data[i] = Float16(std::sqrt(static_cast<float>(in_data[i])));
#endif
            }
        }

    } else if (input.dtype() == DType::Complex64) {
        const auto* in_data = input.data<std::complex<float>>();
        auto* out_data = result.data<std::complex<float>>();
        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::sqrt(in_data[i]);
        }
    } else if (input.dtype() == DType::Complex128) {
        const auto* in_data = input.data<std::complex<double>>();
        auto* out_data = result.data<std::complex<double>>();
        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::sqrt(in_data[i]);
        }
    } else if (input.dtype() == DType::BFloat16) {
        // BFloat16: widen to Float32, compute, narrow back (matches pow and the
        // other unary ops; BF16 has no native std::sqrt path).
        return tenzor::utils::widen_narrow_compute(input,
            [](const Tensor& x) { return sqrt_kernel(x); });
    } else {
        throw std::runtime_error("sqrt operation only supports Float32, Float64, Float16, BFloat16, Complex64, Complex128 dtypes");
    }

    return result;
}

auto neg_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

#ifdef TENZOR_HAS_AVX2
        // SIMD + OpenMP for large arrays
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; i += 8) {
            size_t remaining = std::min(size_t(8), n - i);
            if (remaining == 8) {
                __m256 v = _mm256_loadu_ps(&in_data[i]);
                __m256 neg_v = _mm256_sub_ps(_mm256_setzero_ps(), v);
                _mm256_storeu_ps(&out_data[i], neg_v);
            } else {
                for (size_t j = 0; j < remaining; ++j) {
                    out_data[i + j] = -in_data[i + j];
                }
            }
        }
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = -in_data[i];
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

#ifdef TENZOR_HAS_AVX2
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; i += 4) {
            size_t remaining = std::min(size_t(4), n - i);
            if (remaining == 4) {
                __m256d v = _mm256_loadu_pd(&in_data[i]);
                __m256d neg_v = _mm256_sub_pd(_mm256_setzero_pd(), v);
                _mm256_storeu_pd(&out_data[i], neg_v);
            } else {
                for (size_t j = 0; j < remaining; ++j) {
                    out_data[i + j] = -in_data[i + j];
                }
            }
        }
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = -in_data[i];
        }
#endif

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();

        // Use F16C SIMD for negation (flip sign bit)
#ifdef __F16C__
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; i += 8) {
            size_t remaining = std::min(size_t(8), n - i);
            if (remaining == 8) {
                __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_data + i));
                __m256 fp32 = _mm256_cvtph_ps(packed);
                __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), fp32);
                __m128i out_packed = _mm256_cvtps_ph(neg, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out_data + i), out_packed);
            } else {
                for (size_t j = 0; j < remaining; ++j) {
                    out_data[i + j] = Float16(-static_cast<float>(in_data[i + j]));
                }
            }
        }
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = Float16(-static_cast<float>(in_data[i]));
        }
#endif

    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();

        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = BFloat16(-static_cast<float>(in_data[i]));
        }

    } else if (input.dtype() == DType::Int32) {
        const int32_t* in_data = input.data<int32_t>();
        int32_t* out_data = result.data<int32_t>();

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = -in_data[i];
        }

    } else if (input.dtype() == DType::Complex64) {
        const auto* in_data  = reinterpret_cast<const std::complex<float>*>(input.data<uint8_t>());
        auto*       out_data = reinterpret_cast<std::complex<float>*>(result.data<uint8_t>());
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = -in_data[i];
        }

    } else if (input.dtype() == DType::Complex128) {
        const auto* in_data  = reinterpret_cast<const std::complex<double>*>(input.data<uint8_t>());
        auto*       out_data = reinterpret_cast<std::complex<double>*>(result.data<uint8_t>());
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = -in_data[i];
        }

    } else {
        throw std::runtime_error("neg operation only supports Float32, Float64, Float16, BFloat16, Int32, Complex64, and Complex128 dtypes");
    }

    return result;
}

auto abs_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    size_t n = static_cast<size_t>(input.numel());

    // Complex inputs produce a real-valued magnitude tensor (|z| = sqrt(re² + im²)),
    // so the output dtype differs from the input. Handle before allocating the
    // same-dtype result used by the scalar branches below.
    if (input.dtype() == DType::Complex64) {
        Tensor result(shape_vec, DType::Float32, input.device());
        const auto* in_data  = reinterpret_cast<const std::complex<float>*>(input.data<uint8_t>());
        float* out_data = result.data<float>();
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) out_data[i] = std::abs(in_data[i]);
        return result;
    }
    if (input.dtype() == DType::Complex128) {
        Tensor result(shape_vec, DType::Float64, input.device());
        const auto* in_data  = reinterpret_cast<const std::complex<double>*>(input.data<uint8_t>());
        double* out_data = result.data<double>();
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) out_data[i] = std::abs(in_data[i]);
        return result;
    }

    Tensor result(shape_vec, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

#ifdef TENZOR_HAS_AVX2
        // SIMD + OpenMP for large arrays
        const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; i += 8) {
            size_t remaining = std::min(size_t(8), n - i);
            if (remaining == 8) {
                __m256 v = _mm256_loadu_ps(&in_data[i]);
                __m256 abs_v = _mm256_and_ps(v, sign_mask);
                _mm256_storeu_ps(&out_data[i], abs_v);
            } else {
                for (size_t j = 0; j < remaining; ++j) {
                    out_data[i + j] = std::abs(in_data[i + j]);
                }
            }
        }
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::abs(in_data[i]);
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

#ifdef TENZOR_HAS_AVX2
        const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; i += 4) {
            size_t remaining = std::min(size_t(4), n - i);
            if (remaining == 4) {
                __m256d v = _mm256_loadu_pd(&in_data[i]);
                __m256d abs_v = _mm256_and_pd(v, sign_mask);
                _mm256_storeu_pd(&out_data[i], abs_v);
            } else {
                for (size_t j = 0; j < remaining; ++j) {
                    out_data[i + j] = std::abs(in_data[i + j]);
                }
            }
        }
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::abs(in_data[i]);
        }
#endif

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();

        // Use F16C SIMD for abs (clear sign bit in FP16: mask 0x7FFF)
#ifdef __F16C__
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; i += 8) {
            size_t remaining = std::min(size_t(8), n - i);
            if (remaining == 8) {
                __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_data + i));
                // Clear sign bit directly in FP16 format (more efficient)
                __m128i abs_mask = _mm_set1_epi16(0x7FFF);
                __m128i abs_packed = _mm_and_si128(packed, abs_mask);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out_data + i), abs_packed);
            } else {
                for (size_t j = 0; j < remaining; ++j) {
                    out_data[i + j] = Float16(std::abs(static_cast<float>(in_data[i + j])));
                }
            }
        }
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = Float16(std::abs(static_cast<float>(in_data[i])));
        }
#endif

    } else if (input.dtype() == DType::Int32) {
        const int32_t* in_data = input.data<int32_t>();
        int32_t* out_data = result.data<int32_t>();

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::abs(in_data[i]);
        }

    } else if (input.dtype() == DType::Int8) {
        const int8_t* in_data = input.data<int8_t>();
        int8_t* out_data = result.data<int8_t>();
        for (size_t i = 0; i < n; ++i)
            out_data[i] = static_cast<int8_t>(std::abs(static_cast<int>(in_data[i])));
    } else if (input.dtype() == DType::Int16) {
        const int16_t* in_data = input.data<int16_t>();
        int16_t* out_data = result.data<int16_t>();
        for (size_t i = 0; i < n; ++i)
            out_data[i] = static_cast<int16_t>(std::abs(static_cast<int>(in_data[i])));
    } else if (input.dtype() == DType::Int64) {
        const int64_t* in_data = input.data<int64_t>();
        int64_t* out_data = result.data<int64_t>();
        for (size_t i = 0; i < n; ++i)
            out_data[i] = std::abs(in_data[i]);
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();
        for (size_t i = 0; i < n; ++i)
            out_data[i] = BFloat16(std::abs(static_cast<float>(in_data[i])));
    } else {
        throw std::runtime_error("abs operation only supports Float32, Float64, Float16, BFloat16, Int8, Int16, Int32, Int64, Complex64, and Complex128 dtypes");
    }

    return result;
}

// Clamp kernel - clamps tensor values to [min_val, max_val]
auto clamp_kernel(const Tensor& input, double min_val, double max_val) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();
        float min_f = static_cast<float>(min_val);
        float max_f = static_cast<float>(max_val);

#ifdef TENZOR_HAS_AVX2
        // SIMD: Process 8 floats at a time with OpenMP
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        const __m256 min_vec = _mm256_set1_ps(min_f);
        const __m256 max_vec = _mm256_set1_ps(max_f);

        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 v = _mm256_loadu_ps(&in_data[i]);
            // Clamp: max(min(v, max), min)
            __m256 clamped = _mm256_max_ps(_mm256_min_ps(v, max_vec), min_vec);
            // NaN parity: Intel min/max return the non-NaN operand, so a NaN
            // input would become max_f. The scalar tail and PyTorch propagate
            // NaN, so blend the original value back wherever v is NaN.
            __m256 nan_mask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            clamped = _mm256_blendv_ps(clamped, v, nan_mask);
            _mm256_storeu_ps(&out_data[i], clamped);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::max(std::min(in_data[i], max_f), min_f);
        }
#else
        // Scalar fallback with OpenMP
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::max(std::min(in_data[i], max_f), min_f);
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

        double min_val_d = min_val;
        double max_val_d = max_val;

#ifdef TENZOR_HAS_AVX2
        // SIMD: Process 4 doubles at a time
        size_t simd_end = (n / 4) * 4;
        __m256d min_vec = _mm256_set1_pd(min_val_d);
        __m256d max_vec = _mm256_set1_pd(max_val_d);

        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d v = _mm256_loadu_pd(&in_data[i]);
            // Clamp: max(min(v, max), min)
            __m256d clamped = _mm256_max_pd(_mm256_min_pd(v, max_vec), min_vec);
            // NaN parity with the scalar path / PyTorch (propagate NaN).
            __m256d nan_mask = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
            clamped = _mm256_blendv_pd(clamped, v, nan_mask);
            _mm256_storeu_pd(&out_data[i], clamped);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::max(std::min(in_data[i], max_val_d), min_val_d);
        }
#else
        // Scalar fallback
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::max(std::min(in_data[i], max_val_d), min_val_d);
        }
#endif

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();
        float min_f = static_cast<float>(min_val);
        float max_f = static_cast<float>(max_val);

        // Convert to float, clamp, then convert back
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            val = std::max(std::min(val, max_f), min_f);
            out_data[i] = Float16(val);
        }

    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();
        float min_f = static_cast<float>(min_val);
        float max_f = static_cast<float>(max_val);

        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            val = std::max(std::min(val, max_f), min_f);
            out_data[i] = BFloat16(val);
        }

    } else {
        throw std::runtime_error("clamp operation only supports Float16, BFloat16, Float32 and Float64 dtypes");
    }

    return result;
}

// Clamp min kernel - clamps tensor values to min_val (x >= min_val)
auto clamp_min_kernel(const Tensor& input, double min_val) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();
        float min_f = static_cast<float>(min_val);

#ifdef TENZOR_HAS_AVX2
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        const __m256 min_vec = _mm256_set1_ps(min_f);

        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 v = _mm256_loadu_ps(&in_data[i]);
            __m256 clamped = _mm256_max_ps(v, min_vec);
            _mm256_storeu_ps(&out_data[i], clamped);
        }
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::max(in_data[i], min_f);
        }
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::max(in_data[i], min_f);
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        double min_val_d = min_val;

#ifdef TENZOR_HAS_AVX2
        size_t simd_end = (n / 4) * 4;
        __m256d min_vec = _mm256_set1_pd(min_val_d);

        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d v = _mm256_loadu_pd(&in_data[i]);
            __m256d clamped = _mm256_max_pd(v, min_vec);
            _mm256_storeu_pd(&out_data[i], clamped);
        }
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::max(in_data[i], min_val_d);
        }
#else
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::max(in_data[i], min_val_d);
        }
#endif

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();
        float min_f = static_cast<float>(min_val);

        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            val = std::max(val, min_f);
            out_data[i] = Float16(val);
        }

    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();
        float min_f = static_cast<float>(min_val);

        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            val = std::max(val, min_f);
            out_data[i] = BFloat16(val);
        }

    } else {
        throw std::runtime_error("clamp_min operation only supports Float16, BFloat16, Float32 and Float64 dtypes");
    }

    return result;
}

// Clamp max kernel - clamps tensor values to max_val (x <= max_val)
auto clamp_max_kernel(const Tensor& input, double max_val) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();
        float max_f = static_cast<float>(max_val);

#ifdef TENZOR_HAS_AVX2
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        const __m256 max_vec = _mm256_set1_ps(max_f);

        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 v = _mm256_loadu_ps(&in_data[i]);
            __m256 clamped = _mm256_min_ps(v, max_vec);
            _mm256_storeu_ps(&out_data[i], clamped);
        }
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::min(in_data[i], max_f);
        }
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::min(in_data[i], max_f);
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        double max_val_d = max_val;

#ifdef TENZOR_HAS_AVX2
        size_t simd_end = (n / 4) * 4;
        __m256d max_vec = _mm256_set1_pd(max_val_d);

        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d v = _mm256_loadu_pd(&in_data[i]);
            __m256d clamped = _mm256_min_pd(v, max_vec);
            _mm256_storeu_pd(&out_data[i], clamped);
        }
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::min(in_data[i], max_val_d);
        }
#else
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::min(in_data[i], max_val_d);
        }
#endif

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();
        float max_f = static_cast<float>(max_val);

        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            val = std::min(val, max_f);
            out_data[i] = Float16(val);
        }

    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();
        float max_f = static_cast<float>(max_val);

        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            val = std::min(val, max_f);
            out_data[i] = BFloat16(val);
        }

    } else {
        throw std::runtime_error("clamp_max operation only supports Float16, BFloat16, Float32 and Float64 dtypes");
    }

    return result;
}

// Log kernel - natural logarithm (OpenMP + SIMD optimized)
auto log_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

#if defined(TENZOR_USE_MKL)
        // MKL VML vsLn: IEEE log, internally vectorized + threaded
        vsLn(static_cast<int>(n), in_data, out_data);
#else
        // Full-exact path (audit C1): route Float32 through libm. The SIMD
        // polynomial log was ~10-20 ULP and returned finite garbage for +Inf /
        // NaN inputs; libm matches the Float64 path. MKL uses VML above.
        #pragma omp parallel for if(n > OMP_THRESHOLD_MEDIUM)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::log(in_data[i]);
        }
#endif // TENZOR_USE_MKL

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

#if defined(TENZOR_USE_MKL)
        vdLn(static_cast<int>(n), in_data, out_data);
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_MEDIUM)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::log(in_data[i]);
        }
#endif

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();

        if (n < OMP_THRESHOLD_MEDIUM) {
#ifdef __F16C__
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_data + i));
                __m256 fp32 = _mm256_cvtph_ps(packed);
                __m256 result_v = fast_math::log_avx2(fp32);
                __m128i out_packed = _mm256_cvtps_ph(result_v, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out_data + i), out_packed);
            }
            for (; i < n; ++i) {
                out_data[i] = Float16(std::log(static_cast<float>(in_data[i])));
            }
#else
            for (size_t i = 0; i < n; ++i) {
                out_data[i] = Float16(std::log(static_cast<float>(in_data[i])));
            }
#endif
        } else {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();

                size_t chunk_size = (n + nthreads - 1) / nthreads;
                size_t start = tid * chunk_size;
                size_t end = std::min(start + chunk_size, n);

#ifdef __F16C__
                size_t i = start;
                for (; i + 8 <= end; i += 8) {
                    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_data + i));
                    __m256 fp32 = _mm256_cvtph_ps(packed);
                    __m256 result_v = fast_math::log_avx2(fp32);
                    __m128i out_packed = _mm256_cvtps_ph(result_v, _MM_FROUND_TO_NEAREST_INT);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_data + i), out_packed);
                }
                for (; i < end; ++i) {
                    out_data[i] = Float16(std::log(static_cast<float>(in_data[i])));
                }
#else
                for (size_t i = start; i < end; ++i) {
                    out_data[i] = Float16(std::log(static_cast<float>(in_data[i])));
                }
#endif
            }
        }

    } else if (input.dtype() == DType::Complex64) {
        const auto* in_data = input.data<std::complex<float>>();
        auto* out_data = result.data<std::complex<float>>();
        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::log(in_data[i]);
        }
    } else if (input.dtype() == DType::Complex128) {
        const auto* in_data = input.data<std::complex<double>>();
        auto* out_data = result.data<std::complex<double>>();
        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::log(in_data[i]);
        }
    } else if (input.dtype() == DType::BFloat16) {
        // BFloat16: widen to Float32, compute, narrow back.
        return tenzor::utils::widen_narrow_compute(input,
            [](const Tensor& x) { return log_kernel(x); });
    } else {
        throw std::runtime_error("log operation only supports Float32, Float64, Float16, BFloat16, Complex64, Complex128 dtypes");
    }

    return result;
}

// Exp kernel - exponential (OpenMP + SIMD optimized)
auto exp_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

#if defined(TENZOR_USE_MKL)
        // MKL VML vsExp: internally vectorized + threaded, beats AVX2 polynomial
        vsExp(static_cast<int>(n), in_data, out_data);
#else
        // Full-exact path (audit C1): route Float32 through libm so accuracy
        // and Inf/NaN handling match the Float64 path. The SIMD polynomial in
        // simd_fast_math.hpp was ~2 ULP and mishandled Inf/NaN (e.g. exp(NaN)
        // returned a finite value); MKL builds use the exact VML path above.
        #pragma omp parallel for if(n > OMP_THRESHOLD_MEDIUM)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::exp(in_data[i]);
        }
#endif // TENZOR_USE_MKL

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

#if defined(TENZOR_USE_MKL)
        vdExp(static_cast<int>(n), in_data, out_data);
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_MEDIUM)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::exp(in_data[i]);
        }
#endif

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();

        // Use F16C + fast SIMD exp with OpenMP for large arrays
        // Clamp input to prevent Float16 overflow: exp(11) ≈ 60000 < 65504 (Float16 max)
        constexpr float fp16_exp_max = 11.0f;
        constexpr float fp16_exp_min = -88.0f;  // exp(-88) ≈ 0, underflow to 0 is acceptable
        if (n < OMP_THRESHOLD_MEDIUM) {
#ifdef __F16C__
            size_t i = 0;
            __m256 clamp_max = _mm256_set1_ps(fp16_exp_max);
            __m256 clamp_min = _mm256_set1_ps(fp16_exp_min);
            for (; i + 8 <= n; i += 8) {
                __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_data + i));
                __m256 fp32 = _mm256_cvtph_ps(packed);
                // Clamp to safe range for Float16
                fp32 = _mm256_min_ps(_mm256_max_ps(fp32, clamp_min), clamp_max);
                __m256 result_v = fast_math::exp_avx2(fp32);
                __m128i out_packed = _mm256_cvtps_ph(result_v, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out_data + i), out_packed);
            }
            for (; i < n; ++i) {
                float val = static_cast<float>(in_data[i]);
                val = std::max(fp16_exp_min, std::min(val, fp16_exp_max));
                out_data[i] = Float16(std::exp(val));
            }
#else
            for (size_t i = 0; i < n; ++i) {
                float val = static_cast<float>(in_data[i]);
                val = std::max(fp16_exp_min, std::min(val, fp16_exp_max));
                out_data[i] = Float16(std::exp(val));
            }
#endif
        } else {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();

                size_t chunk_size = (n + nthreads - 1) / nthreads;
                size_t start = tid * chunk_size;
                size_t end = std::min(start + chunk_size, n);

#ifdef __F16C__
                __m256 clamp_max = _mm256_set1_ps(fp16_exp_max);
                __m256 clamp_min = _mm256_set1_ps(fp16_exp_min);
                size_t i = start;
                for (; i + 8 <= end; i += 8) {
                    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_data + i));
                    __m256 fp32 = _mm256_cvtph_ps(packed);
                    // Clamp to safe range for Float16
                    fp32 = _mm256_min_ps(_mm256_max_ps(fp32, clamp_min), clamp_max);
                    __m256 result_v = fast_math::exp_avx2(fp32);
                    __m128i out_packed = _mm256_cvtps_ph(result_v, _MM_FROUND_TO_NEAREST_INT);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_data + i), out_packed);
                }
                for (; i < end; ++i) {
                    float val = static_cast<float>(in_data[i]);
                    val = std::max(fp16_exp_min, std::min(val, fp16_exp_max));
                    out_data[i] = Float16(std::exp(val));
                }
#else
                for (size_t i = start; i < end; ++i) {
                    float val = static_cast<float>(in_data[i]);
                    val = std::max(fp16_exp_min, std::min(val, fp16_exp_max));
                    out_data[i] = Float16(std::exp(val));
                }
#endif
            }
        }

    } else if (input.dtype() == DType::Complex64) {
        const auto* in_data = input.data<std::complex<float>>();
        auto* out_data = result.data<std::complex<float>>();
        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::exp(in_data[i]);
        }
    } else if (input.dtype() == DType::Complex128) {
        const auto* in_data = input.data<std::complex<double>>();
        auto* out_data = result.data<std::complex<double>>();
        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::exp(in_data[i]);
        }
    } else if (input.dtype() == DType::BFloat16) {
        // BFloat16: widen to Float32, compute, narrow back.
        return tenzor::utils::widen_narrow_compute(input,
            [](const Tensor& x) { return exp_kernel(x); });
    } else {
        throw std::runtime_error("exp operation only supports Float32, Float64, Float16, BFloat16, Complex64, Complex128 dtypes");
    }

    return result;
}

// Pow kernel - power function (SIMD + OpenMP optimized)
auto pow_kernel(const Tensor& input, double exponent) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();
        float exponent_f = static_cast<float>(exponent);

        // Full-exact path (audit C1): libm pow. The SIMD `pow_batch` computed
        // exp(y·log(max(x,1e-30))), silently clamping negative bases to ~0
        // (e.g. pow(-2,3) returned ~0 instead of -8). std::pow is correct.
        #pragma omp parallel for if(n > OMP_THRESHOLD_MEDIUM)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::pow(in_data[i], exponent_f);
        }

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        double exp_d = exponent;

        #pragma omp parallel for if(n > OMP_THRESHOLD_MEDIUM)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::pow(in_data[i], exp_d);
        }

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();
        float exponent_f = static_cast<float>(exponent);

        // Full-exact libm path (audit C1): no negative-base clamping.
        std::vector<float> in_f32(n), out_f32(n);
        for (size_t i = 0; i < n; ++i) {
            in_f32[i] = static_cast<float>(in_data[i]);
        }
        for (size_t i = 0; i < n; ++i) {
            out_f32[i] = std::pow(in_f32[i], exponent_f);
        }
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = Float16(out_f32[i]);
        }

    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();
        float exponent_f = static_cast<float>(exponent);
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = BFloat16(std::pow(static_cast<float>(in_data[i]), exponent_f));
        }

    } else {
        throw std::runtime_error("pow operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return result;
}

// Sign kernel - sign function
auto sign_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

        // Sign function: -1 if x < 0, 0 if x == 0, +1 if x > 0
        for (size_t i = 0; i < n; ++i) {
            float val = in_data[i];
            out_data[i] = (val > 0.0f) - (val < 0.0f);
        }

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

        for (size_t i = 0; i < n; ++i) {
            double val = in_data[i];
            out_data[i] = (val > 0.0) - (val < 0.0);
        }

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();

        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            float sign_val = static_cast<float>((val > 0.0f) - (val < 0.0f));
            out_data[i] = Float16(sign_val);
        }

    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();

        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            float sign_val = static_cast<float>((val > 0.0f) - (val < 0.0f));
            out_data[i] = BFloat16(sign_val);
        }

    } else {
        throw std::runtime_error("sign operation only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return result;
}


// ============================================================================
// Comparison Operations
// ============================================================================

// Equal kernel - element-wise equality comparison
auto eq_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        return eq_kernel(a.to(DType::Float32), b.to(DType::Float32));
    }

    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, DType::Bool, a.device());

    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        bool* c_data = result.data<bool>();

        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();

#ifdef TENZOR_HAS_AVX2
            // SIMD + OpenMP: _CMP_EQ_OQ handles NaN correctly (returns false)
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; i += 8) {
                size_t remaining = std::min(size_t(8), n - i);
                if (remaining == 8) {
                    __m256 va = _mm256_loadu_ps(&a_data[i]);
                    __m256 vb = _mm256_loadu_ps(&b_data[i]);
                    // _CMP_EQ_OQ: equal, ordered (NaN returns false)
                    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_EQ_OQ);
                    int mask = _mm256_movemask_ps(cmp);
                    for (int j = 0; j < 8; ++j) {
                        c_data[i + j] = (mask >> j) & 1;
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        c_data[i + j] = (a_data[i + j] == b_data[i + j]) &&
                                        !std::isnan(a_data[i + j]) && !std::isnan(b_data[i + j]);
                    }
                }
            }
#else
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] == b_data[i]) &&
                            !std::isnan(a_data[i]) && !std::isnan(b_data[i]);
            }
#endif
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();

#ifdef TENZOR_HAS_AVX2
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; i += 4) {
                size_t remaining = std::min(size_t(4), n - i);
                if (remaining == 4) {
                    __m256d va = _mm256_loadu_pd(&a_data[i]);
                    __m256d vb = _mm256_loadu_pd(&b_data[i]);
                    __m256d cmp = _mm256_cmp_pd(va, vb, _CMP_EQ_OQ);
                    int mask = _mm256_movemask_pd(cmp);
                    for (int j = 0; j < 4; ++j) {
                        c_data[i + j] = (mask >> j) & 1;
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        c_data[i + j] = (a_data[i + j] == b_data[i + j]) &&
                                        !std::isnan(a_data[i + j]) && !std::isnan(b_data[i + j]);
                    }
                }
            }
#else
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] == b_data[i]) &&
                            !std::isnan(a_data[i]) && !std::isnan(b_data[i]);
            }
#endif
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();

#ifdef TENZOR_HAS_AVX2
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; i += 8) {
                size_t remaining = std::min(size_t(8), n - i);
                if (remaining == 8) {
                    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a_data[i]));
                    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b_data[i]));
                    __m256i cmp = _mm256_cmpeq_epi32(va, vb);
                    // Extract mask from comparison result
                    __m256 cmp_ps = _mm256_castsi256_ps(cmp);
                    int mask = _mm256_movemask_ps(cmp_ps);
                    for (int j = 0; j < 8; ++j) {
                        c_data[i + j] = (mask >> j) & 1;
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        c_data[i + j] = (a_data[i + j] == b_data[i + j]);
                    }
                }
            }
#else
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] == b_data[i]); }
#endif
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] == b_data[i]); }
        } else if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                float af = static_cast<float>(a_data[i]);
                float bf = static_cast<float>(b_data[i]);
                c_data[i] = (af == bf) && !std::isnan(af) && !std::isnan(bf);
            }
        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                float af = static_cast<float>(a_data[i]);
                float bf = static_cast<float>(b_data[i]);
                c_data[i] = (af == bf) && !std::isnan(af) && !std::isnan(bf);
            }
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] == b_data[i]); }
        } else {
            throw std::runtime_error("Unsupported dtype for eq operation");
        }
    } else {
        bool* c_data = result.data<bool>();
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            detail::broadcast_op<float, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) {
                                    // IEEE 754: NaN != NaN
                                    if (std::isnan(x) || std::isnan(y)) return false;
                                    return x == y;
                                });
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            detail::broadcast_op<double, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) {
                                    // IEEE 754: NaN != NaN
                                    if (std::isnan(x) || std::isnan(y)) return false;
                                    return x == y;
                                });
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            detail::broadcast_op<int32_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int32_t x, int32_t y) { return x == y; });
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            detail::broadcast_op<int64_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) { return x == y; });
        } else if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            detail::broadcast_op<Float16, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](Float16 x, Float16 y) {
                                    float xf = static_cast<float>(x), yf = static_cast<float>(y);
                                    if (std::isnan(xf) || std::isnan(yf)) return false;
                                    return xf == yf;
                                });
        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            detail::broadcast_op<BFloat16, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](BFloat16 x, BFloat16 y) {
                                    float xf = static_cast<float>(x), yf = static_cast<float>(y);
                                    if (std::isnan(xf) || std::isnan(yf)) return false;
                                    return xf == yf;
                                });
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            detail::broadcast_op<bool, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](bool x, bool y) { return x == y; });
        } else {
            throw std::runtime_error("Unsupported dtype for eq operation");
        }
    }

    return result;
}

// Not equal kernel
auto ne_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    // Upcast Float16/BFloat16 to Float32 — result is Bool so no downcast needed.
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        return ne_kernel(a.to(DType::Float32), b.to(DType::Float32));
    }

    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, DType::Bool, a.device());

    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        bool* c_data = result.data<bool>();

        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] != b_data[i]); }
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] != b_data[i]); }
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] != b_data[i]); }
        } else {
            TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "ne", [&]() {
                const scalar_t* a_data = a.data<scalar_t>();
                const scalar_t* b_data = b.data<scalar_t>();
                for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] != b_data[i]); }
            });
        }
    } else {
        bool* c_data = result.data<bool>();
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            detail::broadcast_op<float, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) { return x != y; });
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            detail::broadcast_op<double, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) { return x != y; });
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            detail::broadcast_op<bool, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](bool x, bool y) { return x != y; });
        } else {
            TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "ne_broadcast", [&]() {
                const scalar_t* a_data = a.data<scalar_t>();
                const scalar_t* b_data = b.data<scalar_t>();
                detail::broadcast_op<scalar_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                    [](scalar_t x, scalar_t y) { return x != y; });
            });
        }
    }

    return result;
}

// Less than kernel
auto lt_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        return lt_kernel(a.to(DType::Float32), b.to(DType::Float32));
    }

    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, DType::Bool, a.device());

    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        bool* c_data = result.data<bool>();

        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
#ifdef TENZOR_HAS_AVX2
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; i += 8) {
                size_t remaining = std::min(size_t(8), n - i);
                if (remaining == 8) {
                    __m256 va = _mm256_loadu_ps(&a_data[i]);
                    __m256 vb = _mm256_loadu_ps(&b_data[i]);
                    // _CMP_LT_OQ: less than, ordered
                    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_LT_OQ);
                    int mask = _mm256_movemask_ps(cmp);
                    for (int j = 0; j < 8; ++j) {
                        c_data[i + j] = (mask >> j) & 1;
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        c_data[i + j] = (a_data[i + j] < b_data[i + j]);
                    }
                }
            }
#else
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] < b_data[i]);
            }
#endif
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
#ifdef TENZOR_HAS_AVX2
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; i += 4) {
                size_t remaining = std::min(size_t(4), n - i);
                if (remaining == 4) {
                    __m256d va = _mm256_loadu_pd(&a_data[i]);
                    __m256d vb = _mm256_loadu_pd(&b_data[i]);
                    __m256d cmp = _mm256_cmp_pd(va, vb, _CMP_LT_OQ);
                    int mask = _mm256_movemask_pd(cmp);
                    for (int j = 0; j < 4; ++j) {
                        c_data[i + j] = (mask >> j) & 1;
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        c_data[i + j] = (a_data[i + j] < b_data[i + j]);
                    }
                }
            }
#else
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] < b_data[i]);
            }
#endif
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
#ifdef TENZOR_HAS_AVX2
            // For less-than on integers: a < b is same as b > a
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; i += 8) {
                size_t remaining = std::min(size_t(8), n - i);
                if (remaining == 8) {
                    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a_data[i]));
                    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b_data[i]));
                    // a < b == b > a
                    __m256i cmp = _mm256_cmpgt_epi32(vb, va);
                    __m256 cmp_ps = _mm256_castsi256_ps(cmp);
                    int mask = _mm256_movemask_ps(cmp_ps);
                    for (int j = 0; j < 8; ++j) {
                        c_data[i + j] = (mask >> j) & 1;
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        c_data[i + j] = (a_data[i + j] < b_data[i + j]);
                    }
                }
            }
#else
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] < b_data[i]);
            }
#endif
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] < b_data[i]);
            }
        } else if (a.dtype() == DType::Bool) {
            // PyTorch parity: false < true is true; otherwise false.
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] < b_data[i]);
            }
        } else {
            throw std::runtime_error("Unsupported dtype for lt operation");
        }
    } else {
        bool* c_data = result.data<bool>();
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            detail::broadcast_op<float, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) { return x < y; });
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            detail::broadcast_op<double, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) { return x < y; });
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            detail::broadcast_op<bool, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](bool x, bool y) { return x < y; });
        } else {
            TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "lt_broadcast", [&]() {
                const scalar_t* a_data = a.data<scalar_t>();
                const scalar_t* b_data = b.data<scalar_t>();
                detail::broadcast_op<scalar_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                    [](scalar_t x, scalar_t y) { return x < y; });
            });
        }
    }

    return result;
}

// Less than or equal kernel
auto le_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        return le_kernel(a.to(DType::Float32), b.to(DType::Float32));
    }

    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, DType::Bool, a.device());

    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        bool* c_data = result.data<bool>();

        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] <= b_data[i]); }
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] <= b_data[i]); }
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] <= b_data[i]); }
        } else {
            TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "le", [&]() {
                const scalar_t* a_data = a.data<scalar_t>();
                const scalar_t* b_data = b.data<scalar_t>();
                for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] <= b_data[i]); }
            });
        }
    } else {
        bool* c_data = result.data<bool>();
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            detail::broadcast_op<float, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) { return x <= y; });
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            detail::broadcast_op<double, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) { return x <= y; });
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            detail::broadcast_op<bool, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](bool x, bool y) { return x <= y; });
        } else {
            TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "le_broadcast", [&]() {
                const scalar_t* a_data = a.data<scalar_t>();
                const scalar_t* b_data = b.data<scalar_t>();
                detail::broadcast_op<scalar_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                    [](scalar_t x, scalar_t y) { return x <= y; });
            });
        }
    }

    return result;
}

// Greater than kernel
auto gt_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, DType::Bool, a.device());

    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        bool* c_data = result.data<bool>();

        if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
#if defined(TENZOR_HAS_AVX2) && defined(TENZOR_HAS_F16C)
            // F16C + AVX2: Convert to float, then compare
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; i += 8) {
                size_t remaining = std::min(size_t(8), n - i);
                if (remaining == 8) {
                    // Load 8 Float16 values
                    __m128i a_f16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a_data[i]));
                    __m128i b_f16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b_data[i]));
                    // Convert to float
                    __m256 va = _mm256_cvtph_ps(a_f16);
                    __m256 vb = _mm256_cvtph_ps(b_f16);
                    // Compare: _CMP_GT_OQ = greater than, ordered
                    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_GT_OQ);
                    int mask = _mm256_movemask_ps(cmp);
                    for (int j = 0; j < 8; ++j) {
                        c_data[i + j] = (mask >> j) & 1;
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        float af = static_cast<float>(a_data[i + j]);
                        float bf = static_cast<float>(b_data[i + j]);
                        c_data[i + j] = (af > bf);
                    }
                }
            }
#else
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (static_cast<float>(a_data[i]) > static_cast<float>(b_data[i]));
            }
#endif
        } else if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
#ifdef TENZOR_HAS_AVX2
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; i += 8) {
                size_t remaining = std::min(size_t(8), n - i);
                if (remaining == 8) {
                    __m256 va = _mm256_loadu_ps(&a_data[i]);
                    __m256 vb = _mm256_loadu_ps(&b_data[i]);
                    // _CMP_GT_OQ: greater than, ordered
                    __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_GT_OQ);
                    int mask = _mm256_movemask_ps(cmp);
                    for (int j = 0; j < 8; ++j) {
                        c_data[i + j] = (mask >> j) & 1;
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        c_data[i + j] = (a_data[i + j] > b_data[i + j]);
                    }
                }
            }
#else
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] > b_data[i]);
            }
#endif
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
#ifdef TENZOR_HAS_AVX2
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; i += 4) {
                size_t remaining = std::min(size_t(4), n - i);
                if (remaining == 4) {
                    __m256d va = _mm256_loadu_pd(&a_data[i]);
                    __m256d vb = _mm256_loadu_pd(&b_data[i]);
                    __m256d cmp = _mm256_cmp_pd(va, vb, _CMP_GT_OQ);
                    int mask = _mm256_movemask_pd(cmp);
                    for (int j = 0; j < 4; ++j) {
                        c_data[i + j] = (mask >> j) & 1;
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        c_data[i + j] = (a_data[i + j] > b_data[i + j]);
                    }
                }
            }
#else
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] > b_data[i]);
            }
#endif
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
#ifdef TENZOR_HAS_AVX2
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; i += 8) {
                size_t remaining = std::min(size_t(8), n - i);
                if (remaining == 8) {
                    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a_data[i]));
                    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b_data[i]));
                    __m256i cmp = _mm256_cmpgt_epi32(va, vb);
                    __m256 cmp_ps = _mm256_castsi256_ps(cmp);
                    int mask = _mm256_movemask_ps(cmp_ps);
                    for (int j = 0; j < 8; ++j) {
                        c_data[i + j] = (mask >> j) & 1;
                    }
                } else {
                    for (size_t j = 0; j < remaining; ++j) {
                        c_data[i + j] = (a_data[i + j] > b_data[i + j]);
                    }
                }
            }
#else
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] > b_data[i]);
            }
#endif
        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (static_cast<float>(a_data[i]) > static_cast<float>(b_data[i]));
            }
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] > b_data[i]);
            }
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (a_data[i] > b_data[i]);
            }
        } else {
            throw std::runtime_error("Unsupported dtype for gt operation");
        }
    } else {
        bool* c_data = result.data<bool>();
        if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            detail::broadcast_op<Float16, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](Float16 x, Float16 y) { return static_cast<float>(x) > static_cast<float>(y); });
        } else if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            detail::broadcast_op<float, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) { return x > y; });
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            detail::broadcast_op<double, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) { return x > y; });
        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            detail::broadcast_op<BFloat16, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](BFloat16 x, BFloat16 y) { return static_cast<float>(x) > static_cast<float>(y); });
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            detail::broadcast_op<int32_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int32_t x, int32_t y) { return x > y; });
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            detail::broadcast_op<int64_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) { return x > y; });
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            detail::broadcast_op<bool, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](bool x, bool y) { return x > y; });
        } else {
            throw std::runtime_error("Unsupported dtype for gt operation");
        }
    }

    return result;
}

// Greater than or equal kernel
auto ge_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, DType::Bool, a.device());

    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        bool* c_data = result.data<bool>();

        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] >= b_data[i]); }
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] >= b_data[i]); }
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] >= b_data[i]); }
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] >= b_data[i]); }
        } else if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (static_cast<float>(a_data[i]) >= static_cast<float>(b_data[i])); }
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] >= b_data[i]); }
        } else {
            throw std::runtime_error("Unsupported dtype for ge operation");
        }
    } else {
        bool* c_data = result.data<bool>();
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            detail::broadcast_op<float, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) { return x >= y; });
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            detail::broadcast_op<double, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) { return x >= y; });
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            detail::broadcast_op<int32_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int32_t x, int32_t y) { return x >= y; });
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            detail::broadcast_op<int64_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) { return x >= y; });
        } else if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            detail::broadcast_op<Float16, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](Float16 x, Float16 y) { return static_cast<float>(x) >= static_cast<float>(y); });
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            detail::broadcast_op<bool, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](bool x, bool y) { return x >= y; });
        } else {
            throw std::runtime_error("Unsupported dtype for ge operation");
        }
    }

    return result;
}

// Dot product - sum of element-wise multiplication
auto dot_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    // Verify both tensors are 1D
    if (a.ndim() != 1 || b.ndim() != 1) {
        throw std::invalid_argument("dot: inputs must be 1D tensors");
    }

    // Verify same shape
    if (a.shape()[0] != b.shape()[0]) {
        throw std::invalid_argument("dot: inputs must have the same length");
    }

    // Verify same dtype
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("dot: inputs must have the same dtype");
    }

    int64_t n = a.shape()[0];

    // Create scalar output tensor (0-D)
    Tensor output({}, a.dtype(), a.device());

    switch (a.dtype()) {
        case DType::Float32: {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float sum = 0.0f;
            #pragma omp parallel for reduction(+:sum) if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) sum += a_data[i] * b_data[i];
            output.data<float>()[0] = sum;
            break;
        }
        case DType::Float64: {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum) if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) sum += a_data[i] * b_data[i];
            output.data<double>()[0] = sum;
            break;
        }
        case DType::Float16: {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            float sum = 0.0f;
            for (int64_t i = 0; i < n; i++)
                sum += static_cast<float>(a_data[i]) * static_cast<float>(b_data[i]);
            output.data<Float16>()[0] = Float16(sum);
            break;
        }
        case DType::BFloat16: {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            float sum = 0.0f;
            for (int64_t i = 0; i < n; i++)
                sum += static_cast<float>(a_data[i]) * static_cast<float>(b_data[i]);
            output.data<BFloat16>()[0] = BFloat16(sum);
            break;
        }
        case DType::Complex64: {
            const std::complex<float>* a_data = a.data<std::complex<float>>();
            const std::complex<float>* b_data = b.data<std::complex<float>>();
            std::complex<float> sum{0.0f, 0.0f};
            for (int64_t i = 0; i < n; i++) sum += a_data[i] * b_data[i];
            output.data<std::complex<float>>()[0] = sum;
            break;
        }
        case DType::Complex128: {
            const std::complex<double>* a_data = a.data<std::complex<double>>();
            const std::complex<double>* b_data = b.data<std::complex<double>>();
            std::complex<double> sum{0.0, 0.0};
            for (int64_t i = 0; i < n; i++) sum += a_data[i] * b_data[i];
            output.data<std::complex<double>>()[0] = sum;
            break;
        }
        case DType::UInt16: {
            const uint16_t* a_data = a.data<uint16_t>();
            const uint16_t* b_data = b.data<uint16_t>();
            uint32_t sum = 0;
            for (int64_t i = 0; i < n; i++)
                sum += static_cast<uint32_t>(a_data[i]) * static_cast<uint32_t>(b_data[i]);
            output.data<uint16_t>()[0] = static_cast<uint16_t>(sum);
            break;
        }
        case DType::UInt32: {
            const uint32_t* a_data = a.data<uint32_t>();
            const uint32_t* b_data = b.data<uint32_t>();
            uint64_t sum = 0;
            for (int64_t i = 0; i < n; i++)
                sum += static_cast<uint64_t>(a_data[i]) * static_cast<uint64_t>(b_data[i]);
            output.data<uint32_t>()[0] = static_cast<uint32_t>(sum);
            break;
        }
        case DType::UInt64: {
            const uint64_t* a_data = a.data<uint64_t>();
            const uint64_t* b_data = b.data<uint64_t>();
            uint64_t sum = 0;
            for (int64_t i = 0; i < n; i++) sum += a_data[i] * b_data[i];
            output.data<uint64_t>()[0] = sum;
            break;
        }
        default:
            TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "dot", [&]() {
                using acc_t = std::conditional_t<
                    std::is_same_v<scalar_t, int64_t>, int64_t,
                    std::conditional_t<std::is_unsigned_v<scalar_t>, uint64_t, int64_t>>;
                const scalar_t* a_data = a.data<scalar_t>();
                const scalar_t* b_data = b.data<scalar_t>();
                acc_t sum = 0;
                for (int64_t i = 0; i < n; i++)
                    sum += static_cast<acc_t>(a_data[i]) * static_cast<acc_t>(b_data[i]);
                output.data<scalar_t>()[0] = static_cast<scalar_t>(sum);
            });
            break;
    }

    return output;
}

// Trigonometric functions (SIMD + OpenMP optimized)
auto sin_kernel(const Tensor& input) -> Tensor {
    // Float16/BFloat16: upcast to Float32, compute, cast back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto orig_dtype = input.dtype();
        return sin_kernel(input.to(DType::Float32)).to(orig_dtype);
    }

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
#if defined(TENZOR_USE_MKL)
            vsSin(static_cast<int>(n), in_data, out_data);
#else
            // Full-exact path (audit C1): the SIMD polynomial used single-
            // precision range reduction that lost all significance for large
            // |x| (e.g. sin(1e6)); libm matches the Float64 path.
            #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::sin(in_data[i]);
            }
#endif
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
#if defined(TENZOR_USE_MKL)
            vdSin(static_cast<int>(n), in_data, out_data);
#else
            #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::sin(in_data[i]);
            }
#endif
            break;
        }
        case DType::Complex64: {
            const auto* in_data = input.data<std::complex<float>>();
            auto* out_data = output.data<std::complex<float>>();
            #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) out_data[i] = std::sin(in_data[i]);
            break;
        }
        case DType::Complex128: {
            const auto* in_data = input.data<std::complex<double>>();
            auto* out_data = output.data<std::complex<double>>();
            #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) out_data[i] = std::sin(in_data[i]);
            break;
        }
        default:
            throw std::runtime_error("sin: unsupported dtype");
    }
    return output;
}

auto cos_kernel(const Tensor& input) -> Tensor {
    // Float16/BFloat16: upcast to Float32, compute, cast back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto orig_dtype = input.dtype();
        return cos_kernel(input.to(DType::Float32)).to(orig_dtype);
    }

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
#if defined(TENZOR_USE_MKL)
            vsCos(static_cast<int>(n), in_data, out_data);
#else
            // Full-exact path (audit C1): libm cos (the SIMD polynomial lost
            // significance for large |x|). Matches the Float64 path.
            #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::cos(in_data[i]);
            }
#endif
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
#if defined(TENZOR_USE_MKL)
            vdCos(static_cast<int>(n), in_data, out_data);
#else
            #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::cos(in_data[i]);
            }
#endif
            break;
        }
        case DType::Complex64: {
            const auto* in_data = input.data<std::complex<float>>();
            auto* out_data = output.data<std::complex<float>>();
            #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) out_data[i] = std::cos(in_data[i]);
            break;
        }
        case DType::Complex128: {
            const auto* in_data = input.data<std::complex<double>>();
            auto* out_data = output.data<std::complex<double>>();
            #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) out_data[i] = std::cos(in_data[i]);
            break;
        }
        default:
            throw std::runtime_error("cos: unsupported dtype");
    }
    return output;
}

auto tan_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    // Native F32/F64/F16/BF16 dispatch. For F16/BF16, elementwise_unary
    // promotes each element to F32 in a register, applies std::tan, and
    // narrows the result back. No tensor-wide widen-narrow is performed.
    TENZOR_DISPATCH_FLOAT_AND_HALF(input.dtype(), "tan", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = output.data<scalar_t>();
        cpu::elementwise_unary<scalar_t>(in_data, out_data, n,
            [](auto x) { return std::tan(x); });
    });
    return output;
}

auto asin_kernel(const Tensor& input) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        std::vector<int64_t> shape_vec(x_wide.shape().begin(), x_wide.shape().end());
        auto output = Tensor::empty_uninitialized(shape_vec, x_wide.dtype(), x_wide.device());
        int64_t n = x_wide.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "asin", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = output.data<scalar_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::asin(in_data[i]);
            }
        });
        return output;
    });
}

auto acos_kernel(const Tensor& input) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        std::vector<int64_t> shape_vec(x_wide.shape().begin(), x_wide.shape().end());
        auto output = Tensor::empty_uninitialized(shape_vec, x_wide.dtype(), x_wide.device());
        int64_t n = x_wide.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "acos", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = output.data<scalar_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::acos(in_data[i]);
            }
        });
        return output;
    });
}

auto atan_kernel(const Tensor& input) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        std::vector<int64_t> shape_vec(x_wide.shape().begin(), x_wide.shape().end());
        auto output = Tensor::empty_uninitialized(shape_vec, x_wide.dtype(), x_wide.device());
        int64_t n = x_wide.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "atan", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = output.data<scalar_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::atan(in_data[i]);
            }
        });
        return output;
    });
}

// Hyperbolic functions
auto sinh_kernel(const Tensor& input) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        std::vector<int64_t> shape_vec(x_wide.shape().begin(), x_wide.shape().end());
        auto output = Tensor::empty_uninitialized(shape_vec, x_wide.dtype(), x_wide.device());
        int64_t n = x_wide.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "sinh", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = output.data<scalar_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::sinh(in_data[i]);
            }
        });
        return output;
    });
}

auto cosh_kernel(const Tensor& input) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        std::vector<int64_t> shape_vec(x_wide.shape().begin(), x_wide.shape().end());
        auto output = Tensor::empty_uninitialized(shape_vec, x_wide.dtype(), x_wide.device());
        int64_t n = x_wide.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "cosh", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = output.data<scalar_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::cosh(in_data[i]);
            }
        });
        return output;
    });
}

// Rounding functions
auto round_kernel(const Tensor& input) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        std::vector<int64_t> shape_vec(x_wide.shape().begin(), x_wide.shape().end());
        auto output = Tensor::empty_uninitialized(shape_vec, x_wide.dtype(), x_wide.device());
        int64_t n = x_wide.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "round", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = output.data<scalar_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::round(in_data[i]);
            }
        });
        return output;
    });
}

auto floor_kernel(const Tensor& input) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        std::vector<int64_t> shape_vec(x_wide.shape().begin(), x_wide.shape().end());
        auto output = Tensor::empty_uninitialized(shape_vec, x_wide.dtype(), x_wide.device());
        int64_t n = x_wide.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "floor", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = output.data<scalar_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::floor(in_data[i]);
            }
        });
        return output;
    });
}

auto ceil_kernel(const Tensor& input) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        std::vector<int64_t> shape_vec(x_wide.shape().begin(), x_wide.shape().end());
        auto output = Tensor::empty_uninitialized(shape_vec, x_wide.dtype(), x_wide.device());
        int64_t n = x_wide.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "ceil", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = output.data<scalar_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::ceil(in_data[i]);
            }
        });
        return output;
    });
}

auto trunc_kernel(const Tensor& input) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        std::vector<int64_t> shape_vec(x_wide.shape().begin(), x_wide.shape().end());
        auto output = Tensor::empty_uninitialized(shape_vec, x_wide.dtype(), x_wide.device());
        int64_t n = x_wide.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "trunc", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = output.data<scalar_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::trunc(in_data[i]);
            }
        });
        return output;
    });
}

// Reciprocal
auto reciprocal_kernel(const Tensor& input) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        std::vector<int64_t> shape_vec(x_wide.shape().begin(), x_wide.shape().end());
        auto output = Tensor::empty_uninitialized(shape_vec, x_wide.dtype(), x_wide.device());
        int64_t n = x_wide.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "reciprocal", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = output.data<scalar_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = scalar_t(1) / in_data[i];
            }
        });
        return output;
    });
}

// In-place operations
auto add_inplace_kernel(Tensor& a, const Tensor& b) -> Tensor& {
    int64_t n = a.numel();
    bool same_shape = detail::have_same_shape(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    // Validate that b can be broadcast to a's shape
    if (!same_shape && !detail::can_broadcast_to(shape_a, shape_b)) {
        throw std::runtime_error("In-place add: shapes are not compatible for broadcasting");
    }

    switch (a.dtype()) {
        case DType::Float32: {
            float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] += b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](float x, float y) { return x + y; });
            }
            break;
        }
        case DType::Float64: {
            double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] += b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](double x, double y) { return x + y; });
            }
            break;
        }
        case DType::Int32: {
            int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] += b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](int32_t x, int32_t y) { return x + y; });
            }
            break;
        }
        case DType::Int64: {
            int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] += b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](int64_t x, int64_t y) { return x + y; });
            }
            break;
        }
        case DType::Float16: {
            Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            if (same_shape) {
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] = Float16(static_cast<float>(a_data[i]) + static_cast<float>(b_data[i]));
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](Float16 x, Float16 y) {
                        return Float16(static_cast<float>(x) + static_cast<float>(y));
                    });
            }
            break;
        }
        case DType::BFloat16: {
            BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            if (same_shape) {
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] = BFloat16(static_cast<float>(a_data[i]) + static_cast<float>(b_data[i]));
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](BFloat16 x, BFloat16 y) {
                        return BFloat16(static_cast<float>(x) + static_cast<float>(y));
                    });
            }
            break;
        }
        default:
            throw std::runtime_error("add_inplace: unsupported dtype");
    }
    return a;
}

auto sub_inplace_kernel(Tensor& a, const Tensor& b) -> Tensor& {
    int64_t n = a.numel();
    bool same_shape = detail::have_same_shape(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    // Validate that b can be broadcast to a's shape
    if (!same_shape && !detail::can_broadcast_to(shape_a, shape_b)) {
        throw std::runtime_error("In-place sub: shapes are not compatible for broadcasting");
    }

    switch (a.dtype()) {
        case DType::Float32: {
            float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] -= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](float x, float y) { return x - y; });
            }
            break;
        }
        case DType::Float64: {
            double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] -= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](double x, double y) { return x - y; });
            }
            break;
        }
        case DType::Int32: {
            int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] -= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](int32_t x, int32_t y) { return x - y; });
            }
            break;
        }
        case DType::Int64: {
            int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] -= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](int64_t x, int64_t y) { return x - y; });
            }
            break;
        }
        case DType::Float16: {
            Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            if (same_shape) {
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] = Float16(static_cast<float>(a_data[i]) - static_cast<float>(b_data[i]));
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](Float16 x, Float16 y) {
                        return Float16(static_cast<float>(x) - static_cast<float>(y));
                    });
            }
            break;
        }
        case DType::BFloat16: {
            BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            if (same_shape) {
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] = BFloat16(static_cast<float>(a_data[i]) - static_cast<float>(b_data[i]));
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](BFloat16 x, BFloat16 y) {
                        return BFloat16(static_cast<float>(x) - static_cast<float>(y));
                    });
            }
            break;
        }
        default:
            throw std::runtime_error("sub_inplace: unsupported dtype");
    }
    return a;
}

auto mul_inplace_kernel(Tensor& a, const Tensor& b) -> Tensor& {
    int64_t n = a.numel();
    bool same_shape = detail::have_same_shape(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    // Validate that b can be broadcast to a's shape
    if (!same_shape && !detail::can_broadcast_to(shape_a, shape_b)) {
        throw std::runtime_error("In-place mul: shapes are not compatible for broadcasting");
    }

    switch (a.dtype()) {
        case DType::Float32: {
            float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] *= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](float x, float y) { return x * y; });
            }
            break;
        }
        case DType::Float64: {
            double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] *= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](double x, double y) { return x * y; });
            }
            break;
        }
        case DType::Int32: {
            int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] *= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](int32_t x, int32_t y) { return x * y; });
            }
            break;
        }
        case DType::Int64: {
            int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] *= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](int64_t x, int64_t y) { return x * y; });
            }
            break;
        }
        case DType::Float16: {
            Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            if (same_shape) {
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] = Float16(static_cast<float>(a_data[i]) * static_cast<float>(b_data[i]));
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](Float16 x, Float16 y) {
                        return Float16(static_cast<float>(x) * static_cast<float>(y));
                    });
            }
            break;
        }
        case DType::BFloat16: {
            BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            if (same_shape) {
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] = BFloat16(static_cast<float>(a_data[i]) * static_cast<float>(b_data[i]));
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](BFloat16 x, BFloat16 y) {
                        return BFloat16(static_cast<float>(x) * static_cast<float>(y));
                    });
            }
            break;
        }
        default:
            throw std::runtime_error("mul_inplace: unsupported dtype");
    }
    return a;
}

auto div_inplace_kernel(Tensor& a, const Tensor& b) -> Tensor& {
    int64_t n = a.numel();
    bool same_shape = detail::have_same_shape(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    // Validate that b can be broadcast to a's shape
    if (!same_shape && !detail::can_broadcast_to(shape_a, shape_b)) {
        throw std::runtime_error("In-place div: shapes are not compatible for broadcasting");
    }

    switch (a.dtype()) {
        case DType::Float32: {
            float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] /= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](float x, float y) { return x / y; });
            }
            break;
        }
        case DType::Float64: {
            double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] /= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](double x, double y) { return x / y; });
            }
            break;
        }
        case DType::Int32: {
            int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] /= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](int32_t x, int32_t y) { return x / y; });
            }
            break;
        }
        case DType::Int64: {
            int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            if (same_shape) {
                #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] /= b_data[i];
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](int64_t x, int64_t y) { return x / y; });
            }
            break;
        }
        case DType::Float16: {
            Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            if (same_shape) {
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] = Float16(static_cast<float>(a_data[i]) / static_cast<float>(b_data[i]));
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](Float16 x, Float16 y) {
                        return Float16(static_cast<float>(x) / static_cast<float>(y));
                    });
            }
            break;
        }
        case DType::BFloat16: {
            BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            if (same_shape) {
                for (int64_t i = 0; i < n; i++) {
                    a_data[i] = BFloat16(static_cast<float>(a_data[i]) / static_cast<float>(b_data[i]));
                }
            } else {
                detail::broadcast_op_inplace(a_data, b_data, shape_a_vec, shape_b_vec,
                    [](BFloat16 x, BFloat16 y) {
                        return BFloat16(static_cast<float>(x) / static_cast<float>(y));
                    });
            }
            break;
        }
        default:
            throw std::runtime_error("div_inplace: unsupported dtype");
    }
    return a;
}

// =========================================================================
// Extended Math Kernels
// =========================================================================

namespace {

// Helper: apply a unary float function element-wise with multi-dtype support
template<typename F32Fn, typename F64Fn>
auto unary_math_kernel(const Tensor& input, F32Fn f32_fn, F64Fn f64_fn,
                       const char* op_name) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();
        for (size_t i = 0; i < n; ++i) out_data[i] = f32_fn(in_data[i]);
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        for (size_t i = 0; i < n; ++i) out_data[i] = f64_fn(in_data[i]);
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();
        for (size_t i = 0; i < n; ++i)
            out_data[i] = Float16(f32_fn(static_cast<float>(in_data[i])));
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();
        for (size_t i = 0; i < n; ++i)
            out_data[i] = BFloat16(f32_fn(static_cast<float>(in_data[i])));
    } else {
        throw std::runtime_error(std::string(op_name) + ": unsupported dtype");
    }
    return result;
}

// Helper: apply a unary function via MKL VML (Float32/Float64) with scalar
// fallback for Float16/BFloat16 and non-MKL builds.
// VmlF32Fn: void(*)(int, const float*, float*)  — e.g. vsExp
// VmlF64Fn: void(*)(int, const double*, double*) — e.g. vdExp
// ScalarF32/ScalarF64: scalar fallbacks for tail/half types
template<typename VmlF32Fn, typename VmlF64Fn, typename ScalarF32, typename ScalarF64>
auto unary_vml_kernel(const Tensor& input,
                      VmlF32Fn vml_f32, VmlF64Fn vml_f64,
                      [[maybe_unused]] ScalarF32 scalar_f32, [[maybe_unused]] ScalarF64 scalar_f64,
                      const char* op_name) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();
#if defined(TENZOR_USE_MKL)
        vml_f32(static_cast<int>(n), in_data, out_data);
#else
        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < n; ++i) out_data[i] = scalar_f32(in_data[i]);
#endif
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
#if defined(TENZOR_USE_MKL)
        vml_f64(static_cast<int>(n), in_data, out_data);
#else
        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < n; ++i) out_data[i] = scalar_f64(in_data[i]);
#endif
    } else if (input.dtype() == DType::Float16) {
        // Widen to Float32, apply scalar op, narrow back
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();
#if defined(TENZOR_USE_MKL)
        std::vector<float> tmp_in(n), tmp_out(n);
        for (size_t i = 0; i < n; ++i) tmp_in[i] = static_cast<float>(in_data[i]);
        vml_f32(static_cast<int>(n), tmp_in.data(), tmp_out.data());
        for (size_t i = 0; i < n; ++i) out_data[i] = Float16(tmp_out[i]);
#else
        for (size_t i = 0; i < n; ++i)
            out_data[i] = Float16(scalar_f32(static_cast<float>(in_data[i])));
#endif
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();
#if defined(TENZOR_USE_MKL)
        std::vector<float> tmp_in(n), tmp_out(n);
        for (size_t i = 0; i < n; ++i) tmp_in[i] = static_cast<float>(in_data[i]);
        vml_f32(static_cast<int>(n), tmp_in.data(), tmp_out.data());
        for (size_t i = 0; i < n; ++i) out_data[i] = BFloat16(tmp_out[i]);
#else
        for (size_t i = 0; i < n; ++i)
            out_data[i] = BFloat16(scalar_f32(static_cast<float>(in_data[i])));
#endif
    } else {
        throw std::runtime_error(std::string(op_name) + ": unsupported dtype");
    }
    return result;
}

// Helper: apply a unary function returning Bool tensor
template<typename F32Fn, typename F64Fn>
auto unary_bool_kernel(const Tensor& input, F32Fn f32_fn, F64Fn f64_fn,
                       const char* op_name) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, DType::Bool, input.device());
    size_t n = static_cast<size_t>(input.numel());
    bool* out_data = result.data<bool>();

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        for (size_t i = 0; i < n; ++i) out_data[i] = f32_fn(in_data[i]);
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        for (size_t i = 0; i < n; ++i) out_data[i] = f64_fn(in_data[i]);
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        for (size_t i = 0; i < n; ++i) out_data[i] = f32_fn(static_cast<float>(in_data[i]));
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        for (size_t i = 0; i < n; ++i) out_data[i] = f32_fn(static_cast<float>(in_data[i]));
    } else if (input.dtype() == DType::Int32 || input.dtype() == DType::Int64) {
        // Integers are always finite, never NaN/Inf
        bool val_isnan = false;
        bool val_isinf = false;
        bool val_isfinite = true;
        // Determine which function this is by testing a known value
        bool is_nan_fn = f32_fn(std::numeric_limits<float>::quiet_NaN());
        bool is_inf_fn = f32_fn(std::numeric_limits<float>::infinity());
        bool fill_val = is_nan_fn ? val_isnan : (is_inf_fn ? val_isinf : val_isfinite);
        for (size_t i = 0; i < n; ++i) out_data[i] = fill_val;
    } else {
        throw std::runtime_error(std::string(op_name) + ": unsupported dtype");
    }
    return result;
}

// Helper: binary element-wise op with broadcast support.
//
// Float16/BFloat16 inputs are transparently upcast to Float32, computed, and
// the result cast back to the original dtype. This keeps the math kernels
// using std::atan2/std::fmod/... which are only defined for native float and
// double — and matches the accuracy strategy used by the normalization layers
// (compute in FP32 for numerical stability).
template<typename F32Fn, typename F64Fn>
auto binary_math_kernel(const Tensor& a, const Tensor& b, F32Fn f32_fn, F64Fn f64_fn,
                        const char* op_name) -> Tensor {
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        auto orig_dtype = a.dtype();
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto result_f32 = binary_math_kernel(a_f32, b_f32, f32_fn, f64_fn, op_name);
        return result_f32.to(orig_dtype);
    }

    detail::validate_elementwise(a, b);
    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    if (detail::have_same_shape(a, b)) {
        // Fast path: no broadcasting needed
        size_t n = static_cast<size_t>(a.numel());
        TENZOR_DISPATCH_FLOATING_TYPES(a.dtype(), op_name, [&]() {
            const scalar_t* ad = a.data<scalar_t>();
            const scalar_t* bd = b.data<scalar_t>();
            scalar_t* od = result.data<scalar_t>();
            for (size_t i = 0; i < n; ++i) {
                if constexpr (std::is_same_v<scalar_t, float>) {
                    od[i] = f32_fn(ad[i], bd[i]);
                } else {
                    od[i] = f64_fn(ad[i], bd[i]);
                }
            }
        });
    } else {
        // Broadcast path using stride-based indexing
        auto a_strides = detail::compute_broadcast_strides(shape_a, output_shape);
        auto b_strides = detail::compute_broadcast_strides(shape_b, output_shape);
        int64_t ndim = static_cast<int64_t>(output_shape.size());
        int64_t n = result.numel();

        TENZOR_DISPATCH_FLOATING_TYPES(a.dtype(), op_name, [&]() {
            const scalar_t* ad = a.data<scalar_t>();
            const scalar_t* bd = b.data<scalar_t>();
            scalar_t* od = result.data<scalar_t>();
            _Pragma("omp parallel for if(n > static_cast<int64_t>(OMP_THRESHOLD_SIMPLE))")
            for (int64_t i = 0; i < n; ++i) {
                int64_t a_idx = 0, b_idx = 0, idx = i;
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    int64_t coord = idx % output_shape[d];
                    idx /= output_shape[d];
                    a_idx += coord * a_strides[d];
                    b_idx += coord * b_strides[d];
                }
                if constexpr (std::is_same_v<scalar_t, float>) {
                    od[i] = f32_fn(ad[a_idx], bd[b_idx]);
                } else {
                    od[i] = f64_fn(ad[a_idx], bd[b_idx]);
                }
            }
        });
    }
    return result;
}

} // anonymous namespace

auto log2_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsLog2, vdLog2,
        [](float x) { return std::log2(x); },
        [](double x) { return std::log2(x); }, "log2");
#else
    return unary_math_kernel(input,
        [](float x) { return std::log2(x); },
        [](double x) { return std::log2(x); }, "log2");
#endif
}

auto log10_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsLog10, vdLog10,
        [](float x) { return std::log10(x); },
        [](double x) { return std::log10(x); }, "log10");
#else
    return unary_math_kernel(input,
        [](float x) { return std::log10(x); },
        [](double x) { return std::log10(x); }, "log10");
#endif
}

auto log1p_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsLog1p, vdLog1p,
        [](float x) { return std::log1p(x); },
        [](double x) { return std::log1p(x); }, "log1p");
#else
    return unary_math_kernel(input,
        [](float x) { return std::log1p(x); },
        [](double x) { return std::log1p(x); }, "log1p");
#endif
}

auto exp2_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsExp2, vdExp2,
        [](float x) { return std::exp2(x); },
        [](double x) { return std::exp2(x); }, "exp2");
#else
    return unary_math_kernel(input,
        [](float x) { return std::exp2(x); },
        [](double x) { return std::exp2(x); }, "exp2");
#endif
}

auto expm1_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsExpm1, vdExpm1,
        [](float x) { return std::expm1(x); },
        [](double x) { return std::expm1(x); }, "expm1");
#else
    return unary_math_kernel(input,
        [](float x) { return std::expm1(x); },
        [](double x) { return std::expm1(x); }, "expm1");
#endif
}

auto erf_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsErf, vdErf,
        [](float x) { return std::erf(x); },
        [](double x) { return std::erf(x); }, "erf");
#else
    return unary_math_kernel(input,
        [](float x) { return std::erf(x); },
        [](double x) { return std::erf(x); }, "erf");
#endif
}

auto erfc_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsErfc, vdErfc,
        [](float x) { return std::erfc(x); },
        [](double x) { return std::erfc(x); }, "erfc");
#else
    return unary_math_kernel(input,
        [](float x) { return std::erfc(x); },
        [](double x) { return std::erfc(x); }, "erfc");
#endif
}

auto isnan_kernel(const Tensor& input) -> Tensor {
    return unary_bool_kernel(input,
        [](float x) { return std::isnan(x); },
        [](double x) { return std::isnan(x); }, "isnan");
}

auto isinf_kernel(const Tensor& input) -> Tensor {
    return unary_bool_kernel(input,
        [](float x) { return std::isinf(x); },
        [](double x) { return std::isinf(x); }, "isinf");
}

auto isfinite_kernel(const Tensor& input) -> Tensor {
    return unary_bool_kernel(input,
        [](float x) { return std::isfinite(x); },
        [](double x) { return std::isfinite(x); }, "isfinite");
}

auto atan2_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_math_kernel(a, b,
        [](float y, float x) { return std::atan2(y, x); },
        [](double y, double x) { return std::atan2(y, x); }, "atan2");
}

auto fmod_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_math_kernel(a, b,
        [](float x, float y) { return std::fmod(x, y); },
        [](double x, double y) { return std::fmod(x, y); }, "fmod");
}

auto remainder_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_math_kernel(a, b,
        [](float x, float y) { return std::remainder(x, y); },
        [](double x, double y) { return std::remainder(x, y); }, "remainder");
}

auto lerp_kernel(std::span<const Tensor> inputs) -> Tensor {
    if (inputs.size() != 3) {
        throw std::runtime_error("lerp requires 3 inputs (start, end, weight)");
    }
    const auto& start = inputs[0];
    const auto& end = inputs[1];
    const auto& weight = inputs[2];

    // Upcast Float16/BFloat16 to Float32 for numerical stability.
    if (start.dtype() == DType::Float16 || start.dtype() == DType::BFloat16) {
        auto orig_dtype = start.dtype();
        Tensor start_f32 = start.to(DType::Float32);
        Tensor end_f32 = end.to(DType::Float32);
        Tensor weight_f32 = weight.to(DType::Float32);
        std::vector<Tensor> upcast_inputs = {start_f32, end_f32, weight_f32};
        Tensor result_f32 = lerp_kernel(upcast_inputs);
        return result_f32.to(orig_dtype);
    }

    auto shape_vec = std::vector<int64_t>(start.shape().begin(), start.shape().end());
    Tensor result(shape_vec, start.dtype(), start.device());
    size_t n = static_cast<size_t>(result.numel());

    TENZOR_DISPATCH_FLOATING_TYPES(start.dtype(), "lerp", [&]() {
        const scalar_t* s = start.data<scalar_t>();
        const scalar_t* e = end.data<scalar_t>();
        scalar_t* o = result.data<scalar_t>();
        // Scalar weight (numel==1) or element-wise
        if (weight.numel() == 1) {
            scalar_t w = weight.data<scalar_t>()[0];
            for (size_t i = 0; i < n; ++i) o[i] = s[i] + w * (e[i] - s[i]);
        } else {
            const scalar_t* w = weight.data<scalar_t>();
            for (size_t i = 0; i < n; ++i) o[i] = s[i] + w[i] * (e[i] - s[i]);
        }
    });
    return result;
}

// Helper: convert any dtype element to bool (non-zero = true)
namespace {
auto to_bool_value(const Tensor& t, size_t idx) -> bool {
    switch (t.dtype()) {
        case DType::Bool: return t.data<bool>()[idx];
        case DType::Float32: return t.data<float>()[idx] != 0.0f;
        case DType::Float64: return t.data<double>()[idx] != 0.0;
        case DType::Int32: return t.data<int32_t>()[idx] != 0;
        case DType::Int64: return t.data<int64_t>()[idx] != 0;
        case DType::Int8: return t.data<int8_t>()[idx] != 0;
        case DType::UInt8: return t.data<uint8_t>()[idx] != 0;
        case DType::Float16: return static_cast<float>(t.data<Float16>()[idx]) != 0.0f;
        case DType::BFloat16: return static_cast<float>(t.data<BFloat16>()[idx]) != 0.0f;
        default: return t.data<float>()[idx] != 0.0f;
    }
}
} // anonymous namespace

auto logical_and_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    auto shape_vec = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    Tensor result(shape_vec, DType::Bool, a.device());
    size_t n = static_cast<size_t>(a.numel());
    bool* out = result.data<bool>();
    for (size_t i = 0; i < n; ++i) {
        out[i] = to_bool_value(a, i) && to_bool_value(b, i);
    }
    return result;
}

auto logical_or_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    auto shape_vec = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    Tensor result(shape_vec, DType::Bool, a.device());
    size_t n = static_cast<size_t>(a.numel());
    bool* out = result.data<bool>();
    for (size_t i = 0; i < n; ++i) {
        out[i] = to_bool_value(a, i) || to_bool_value(b, i);
    }
    return result;
}

auto logical_not_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, DType::Bool, input.device());
    size_t n = static_cast<size_t>(input.numel());
    bool* out = result.data<bool>();
    for (size_t i = 0; i < n; ++i) {
        out[i] = !to_bool_value(input, i);
    }
    return result;
}

auto logical_xor_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    auto shape_vec = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    Tensor result(shape_vec, DType::Bool, a.device());
    size_t n = static_cast<size_t>(a.numel());
    bool* out = result.data<bool>();
    for (size_t i = 0; i < n; ++i) {
        out[i] = to_bool_value(a, i) != to_bool_value(b, i);
    }
    return result;
}

template<typename T>
static void minimum_typed(const T* a_data, const T* b_data, T* c_data,
                          const Tensor& a, const Tensor& b,
                          std::vector<int64_t>& shape_a_vec,
                          std::vector<int64_t>& shape_b_vec,
                          std::vector<int64_t>& output_shape) {
    // Compare via float promotion so the template works for Float16/BFloat16
    // (which expose explicit `operator float()` but not `operator<`). For
    // float/double/integer types this cast is a compile-time no-op.
    auto less = [](T x, T y) -> bool {
        return static_cast<float>(x) < static_cast<float>(y);
    };
    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        for (size_t i = 0; i < n; ++i)
            c_data[i] = less(a_data[i], b_data[i]) ? a_data[i] : b_data[i];
    } else {
        detail::broadcast_op<T, T>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
            [less](T x, T y) -> T { return less(x, y) ? x : y; });
    }
}

auto minimum_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    if (a.dtype() == DType::Float32) {
        minimum_typed(a.data<float>(), b.data<float>(), result.data<float>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else if (a.dtype() == DType::Float64) {
        minimum_typed(a.data<double>(), b.data<double>(), result.data<double>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else if (a.dtype() == DType::Float16) {
        minimum_typed(a.data<Float16>(), b.data<Float16>(), result.data<Float16>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else if (a.dtype() == DType::BFloat16) {
        minimum_typed(a.data<BFloat16>(), b.data<BFloat16>(), result.data<BFloat16>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else {
        TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "minimum", [&]() {
            minimum_typed(a.data<scalar_t>(), b.data<scalar_t>(), result.data<scalar_t>(),
                          a, b, shape_a_vec, shape_b_vec, output_shape);
        });
    }
    return result;
}

template<typename T>
static void maximum_typed(const T* a_data, const T* b_data, T* c_data,
                          const Tensor& a, const Tensor& b,
                          std::vector<int64_t>& shape_a_vec,
                          std::vector<int64_t>& shape_b_vec,
                          std::vector<int64_t>& output_shape) {
    auto greater = [](T x, T y) -> bool {
        return static_cast<float>(x) > static_cast<float>(y);
    };
    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        for (size_t i = 0; i < n; ++i)
            c_data[i] = greater(a_data[i], b_data[i]) ? a_data[i] : b_data[i];
    } else {
        detail::broadcast_op<T, T>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
            [greater](T x, T y) -> T { return greater(x, y) ? x : y; });
    }
}

auto maximum_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    if (a.dtype() == DType::Float32) {
        maximum_typed(a.data<float>(), b.data<float>(), result.data<float>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else if (a.dtype() == DType::Float64) {
        maximum_typed(a.data<double>(), b.data<double>(), result.data<double>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else if (a.dtype() == DType::Float16) {
        maximum_typed(a.data<Float16>(), b.data<Float16>(), result.data<Float16>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else if (a.dtype() == DType::BFloat16) {
        maximum_typed(a.data<BFloat16>(), b.data<BFloat16>(), result.data<BFloat16>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else {
        TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "maximum", [&]() {
            maximum_typed(a.data<scalar_t>(), b.data<scalar_t>(), result.data<scalar_t>(),
                          a, b, shape_a_vec, shape_b_vec, output_shape);
        });
    }
    return result;
}

// =========================================================================
// Complex Number Operations
// =========================================================================

auto conj_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t n = input.numel();

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape_vec, DType::Complex64, input.device());
        const auto* in = input.data<std::complex<float>>();
        auto* out = result.data<std::complex<float>>();
        for (int64_t i = 0; i < n; ++i) out[i] = std::conj(in[i]);
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape_vec, DType::Complex128, input.device());
        const auto* in = input.data<std::complex<double>>();
        auto* out = result.data<std::complex<double>>();
        for (int64_t i = 0; i < n; ++i) out[i] = std::conj(in[i]);
        return result;
    }
    // For real dtypes, conjugate is identity
    return input.clone();
}

auto real_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t n = input.numel();

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape_vec, DType::Float32, input.device());
        const auto* in = input.data<std::complex<float>>();
        auto* out = result.data<float>();
        for (int64_t i = 0; i < n; ++i) out[i] = in[i].real();
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape_vec, DType::Float64, input.device());
        const auto* in = input.data<std::complex<double>>();
        auto* out = result.data<double>();
        for (int64_t i = 0; i < n; ++i) out[i] = in[i].real();
        return result;
    }
    // For real dtypes, real() is identity (but possibly with dtype change)
    return input.clone();
}

auto imag_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t n = input.numel();

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape_vec, DType::Float32, input.device());
        const auto* in = input.data<std::complex<float>>();
        auto* out = result.data<float>();
        for (int64_t i = 0; i < n; ++i) out[i] = in[i].imag();
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape_vec, DType::Float64, input.device());
        const auto* in = input.data<std::complex<double>>();
        auto* out = result.data<double>();
        for (int64_t i = 0; i < n; ++i) out[i] = in[i].imag();
        return result;
    }
    // For real dtypes, imaginary part is zero
    Tensor result(shape_vec, input.dtype(), input.device());
    result.fill_(0.0f);
    return result;
}

auto angle_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t n = input.numel();

    if (input.dtype() == DType::Complex64) {
        Tensor result(shape_vec, DType::Float32, input.device());
        const auto* in = input.data<std::complex<float>>();
        auto* out = result.data<float>();
        for (int64_t i = 0; i < n; ++i) out[i] = std::arg(in[i]);
        return result;
    } else if (input.dtype() == DType::Complex128) {
        Tensor result(shape_vec, DType::Float64, input.device());
        const auto* in = input.data<std::complex<double>>();
        auto* out = result.data<double>();
        for (int64_t i = 0; i < n; ++i) out[i] = std::arg(in[i]);
        return result;
    }
    // Float16/BFloat16: compute in Float32 and cast back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto orig_dtype = input.dtype();
        Tensor result_f32 = angle_kernel(input.to(DType::Float32));
        return result_f32.to(orig_dtype);
    }
    // For real dtypes, angle is 0 for positive, pi for negative
    return TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "angle", [&]() -> Tensor {
        Tensor result(shape_vec, input.dtype(), input.device());
        const auto* in = input.data<scalar_t>();
        auto* out = result.data<scalar_t>();
        for (int64_t i = 0; i < n; ++i) out[i] = std::atan2(scalar_t(0), in[i]);
        return result;
    });
}

auto polar_kernel(const Tensor& abs_t, const Tensor& angle_t) -> Tensor {
    auto shape_a = abs_t.shape();
    auto shape_b = angle_t.shape();
    if (!std::equal(shape_a.begin(), shape_a.end(), shape_b.begin(), shape_b.end())) {
        throw std::runtime_error("polar: abs and angle must have the same shape");
    }
    auto shape_vec = std::vector<int64_t>(abs_t.shape().begin(), abs_t.shape().end());
    int64_t n = abs_t.numel();

    // Float16/BFloat16: upcast to Float32, compute (yielding Complex64), return.
    // (polar() has no "complex_half" output dtype in the system.)
    if (abs_t.dtype() == DType::Float16 || abs_t.dtype() == DType::BFloat16) {
        return polar_kernel(abs_t.to(DType::Float32), angle_t.to(DType::Float32));
    }

    return TENZOR_DISPATCH_FLOATING_TYPES(abs_t.dtype(), "polar", [&]() -> Tensor {
        using complex_t = std::complex<scalar_t>;
        auto out_dtype = std::is_same_v<scalar_t, float> ? DType::Complex64 : DType::Complex128;
        Tensor result(shape_vec, out_dtype, abs_t.device());
        const auto* r = abs_t.data<scalar_t>();
        const auto* theta = angle_t.data<scalar_t>();
        auto* out = result.data<complex_t>();
        for (int64_t i = 0; i < n; ++i) out[i] = std::polar(r[i], theta[i]);
        return result;
    });
}

auto cross_kernel(const Tensor& a, const Tensor& b, int64_t dim) -> Tensor {
    auto shape_a = a.shape();
    int64_t ndim = shape_a.size();

    auto a_cont = a.contiguous();
    auto b_cont = b.contiguous();
    auto result = Tensor(std::vector<int64_t>(shape_a.begin(), shape_a.end()),
                         a.dtype(), a.device());

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= shape_a[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape_a[d];
    int64_t dim_stride = inner;

    auto cross_typed = [&](auto* a_ptr, auto* b_ptr, auto* c_ptr) {
        #pragma omp parallel for if(outer * inner > ::tenzor::OmpThresholds::complex())
        for (int64_t idx = 0; idx < outer * inner; ++idx) {
            int64_t o = idx / inner, i = idx % inner;
            int64_t base = o * 3 * inner + i;
            auto a0 = a_ptr[base], a1 = a_ptr[base + dim_stride], a2 = a_ptr[base + 2*dim_stride];
            auto b0 = b_ptr[base], b1 = b_ptr[base + dim_stride], b2 = b_ptr[base + 2*dim_stride];
            c_ptr[base]                  = a1*b2 - a2*b1;
            c_ptr[base + dim_stride]     = a2*b0 - a0*b2;
            c_ptr[base + 2*dim_stride]   = a0*b1 - a1*b0;
        }
    };

    switch (a.dtype()) {
        case DType::Float32:
            cross_typed(a_cont.data<float>(), b_cont.data<float>(), result.data<float>());
            break;
        case DType::Float64:
            cross_typed(a_cont.data<double>(), b_cont.data<double>(), result.data<double>());
            break;
        case DType::Float16:
        case DType::BFloat16: {
            // Upcast to Float32, compute, downcast back
            auto a_f32 = a_cont.to(DType::Float32);
            auto b_f32 = b_cont.to(DType::Float32);
            auto r_f32 = Tensor(std::vector<int64_t>(shape_a.begin(), shape_a.end()),
                                DType::Float32, a.device());
            cross_typed(a_f32.data<float>(), b_f32.data<float>(), r_f32.data<float>());
            return r_f32.to(a.dtype());
        }
        default:
            throw std::runtime_error("cross: unsupported dtype");
    }
    return result;
}

// ============================================================================
// CDist Kernel - Pairwise distance computation
// ============================================================================

auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p) -> Tensor {
    // x1: (..., P, M), x2: (..., R, M) -> output: (..., P, R)
    // For simplicity, handle 2D and 3D inputs
    auto s1 = x1.shape();
    auto s2 = x2.shape();

    int64_t ndim1 = static_cast<int64_t>(s1.size());
    int64_t ndim2 = static_cast<int64_t>(s2.size());

    if (ndim1 < 2 || ndim2 < 2) {
        throw std::runtime_error("cdist: inputs must have at least 2 dimensions");
    }

    int64_t M = s1[ndim1 - 1];
    if (s2[ndim2 - 1] != M) {
        throw std::runtime_error("cdist: last dimension of x1 and x2 must match");
    }

    int64_t P = s1[ndim1 - 2];
    int64_t R = s2[ndim2 - 2];

    // Compute batch size
    int64_t batch1 = 1, batch2 = 1;
    for (int64_t d = 0; d < ndim1 - 2; ++d) batch1 *= s1[d];
    for (int64_t d = 0; d < ndim2 - 2; ++d) batch2 *= s2[d];

    if (batch1 != batch2 && batch1 != 1 && batch2 != 1) {
        throw std::runtime_error("cdist: batch dimensions must be broadcastable");
    }
    int64_t batch_size = std::max(batch1, batch2);

    // Compute in Float64 for Float64 inputs, Float32 for everything
    // else (Float32 / Float16 / BFloat16 / integer). The result tensor
    // is built in the compute dtype; reduced-precision floats are cast
    // back below so the caller sees the input dtype preserved.
    const DType orig_dtype = x1.dtype();
    const DType compute_dtype =
        (orig_dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor a = (x1.dtype() != compute_dtype) ? x1.to(compute_dtype) : x1;
    Tensor b = (x2.dtype() != compute_dtype) ? x2.to(compute_dtype) : x2;

    // Output shape
    std::vector<int64_t> out_shape;
    // Use the larger batch dims
    auto& ref_shape = (batch1 >= batch2) ? s1 : s2;
    int64_t ref_ndim = static_cast<int64_t>(ref_shape.size());
    for (int64_t d = 0; d < ref_ndim - 2; ++d) {
        out_shape.push_back(ref_shape[d]);
    }
    out_shape.push_back(P);
    out_shape.push_back(R);

    Tensor result(out_shape, compute_dtype, x1.device());

    auto compute = [&]<typename T>(T*) {
        T* out_data = result.data<T>();
        const T* a_data = a.data<T>();
        const T* b_data = b.data<T>();

        if (p == 2.0) {
            // Euclidean: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b^T
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                int64_t b1 = (batch1 == 1) ? 0 : batch;
                int64_t b2 = (batch2 == 1) ? 0 : batch;

                const T* a_batch = a_data + b1 * P * M;
                const T* b_batch = b_data + b2 * R * M;
                T* o_batch = out_data + batch * P * R;

                std::vector<T> a_sq(static_cast<size_t>(P));
                for (int64_t i = 0; i < P; ++i) {
                    T sum = T(0);
                    for (int64_t k = 0; k < M; ++k) {
                        T v = a_batch[i * M + k];
                        sum += v * v;
                    }
                    a_sq[static_cast<size_t>(i)] = sum;
                }
                std::vector<T> b_sq(static_cast<size_t>(R));
                for (int64_t j = 0; j < R; ++j) {
                    T sum = T(0);
                    for (int64_t k = 0; k < M; ++k) {
                        T v = b_batch[j * M + k];
                        sum += v * v;
                    }
                    b_sq[static_cast<size_t>(j)] = sum;
                }
                #pragma omp parallel for collapse(2) if(P * R > ::tenzor::OmpThresholds::complex())
                for (int64_t i = 0; i < P; ++i) {
                    for (int64_t j = 0; j < R; ++j) {
                        T dot = T(0);
                        for (int64_t k = 0; k < M; ++k) {
                            dot += a_batch[i * M + k] * b_batch[j * M + k];
                        }
                        T dist_sq = a_sq[static_cast<size_t>(i)] + b_sq[static_cast<size_t>(j)]
                                    - T(2) * dot;
                        o_batch[i * R + j] = std::sqrt(std::max(T(0), dist_sq));
                    }
                }
            }
        } else {
            const T inv_p = (p != 0.0) ? static_cast<T>(1.0 / p) : T(0);
            const T pf = static_cast<T>(p);
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                int64_t b1 = (batch1 == 1) ? 0 : batch;
                int64_t b2 = (batch2 == 1) ? 0 : batch;

                const T* a_batch = a_data + b1 * P * M;
                const T* b_batch = b_data + b2 * R * M;
                T* o_batch = out_data + batch * P * R;

                #pragma omp parallel for collapse(2) if(P * R > ::tenzor::OmpThresholds::complex())
                for (int64_t i = 0; i < P; ++i) {
                    for (int64_t j = 0; j < R; ++j) {
                        if (p == std::numeric_limits<double>::infinity()) {
                            T max_val = T(0);
                            for (int64_t k = 0; k < M; ++k) {
                                max_val = std::max(max_val,
                                    static_cast<T>(std::abs(a_batch[i * M + k] - b_batch[j * M + k])));
                            }
                            o_batch[i * R + j] = max_val;
                        } else if (p == 0.0) {
                            T count = T(0);
                            for (int64_t k = 0; k < M; ++k) {
                                if (a_batch[i * M + k] != b_batch[j * M + k]) count += T(1);
                            }
                            o_batch[i * R + j] = count;
                        } else if (p == 1.0) {
                            T sum = T(0);
                            for (int64_t k = 0; k < M; ++k) {
                                sum += static_cast<T>(std::abs(a_batch[i * M + k] - b_batch[j * M + k]));
                            }
                            o_batch[i * R + j] = sum;
                        } else {
                            T sum = T(0);
                            for (int64_t k = 0; k < M; ++k) {
                                sum += static_cast<T>(std::pow(
                                    std::abs(a_batch[i * M + k] - b_batch[j * M + k]), pf));
                            }
                            o_batch[i * R + j] = static_cast<T>(std::pow(sum, inv_p));
                        }
                    }
                }
            }
        }
    };

    if (compute_dtype == DType::Float64) {
        compute(static_cast<double*>(nullptr));
    } else {
        compute(static_cast<float*>(nullptr));
    }

    if (result.dtype() != orig_dtype) {
        return result.to(orig_dtype);
    }
    return result;
}

// =========================================================================
// Special Math Functions
// =========================================================================

auto gamma_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) { return std::tgamma(x); },
        [](double x) { return std::tgamma(x); }, "gamma");
}

auto lgamma_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) { return std::lgamma(x); },
        [](double x) { return std::lgamma(x); }, "lgamma");
}

auto digamma_kernel(const Tensor& input) -> Tensor {
    // Digamma via recurrence + asymptotic expansion
    auto digamma_fn = [](double x) -> double {
        double result = 0.0;
        if (x < 0.5) {
            // Reflection: ψ(x) = ψ(1-x) - π*cot(πx)
            double y = 1.0 - x;
            double r = 0.0;
            while (y < 7.0) {
                r -= 1.0 / y;
                y += 1.0;
            }
            double y2 = 1.0 / (y * y);
            r += std::log(y) - 0.5 / y
                - y2 * (1.0/12.0 - y2 * (1.0/120.0 - y2 * (1.0/252.0
                - y2 * (1.0/240.0 - y2 * (1.0/132.0)))));
            return r - M_PI / std::tan(M_PI * x);
        }
        while (x < 7.0) {
            result -= 1.0 / x;
            x += 1.0;
        }
        double x2 = 1.0 / (x * x);
        result += std::log(x) - 0.5 / x
                - x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0
                - x2 * (1.0/240.0 - x2 * (1.0/132.0)))));
        return result;
    };

    return unary_math_kernel(input,
        [&digamma_fn](float x) { return static_cast<float>(digamma_fn(static_cast<double>(x))); },
        digamma_fn, "digamma");
}

auto polygamma_kernel(const Tensor& input, int64_t n) -> Tensor {
    // Polygamma ψ^(n)(x) via asymptotic expansion
    // For n=0, this is digamma (handled separately for efficiency).
    // For n≥1: ψ^(n)(x) = (-1)^(n+1) * n! * Σ_{k=0}^∞ 1/(x+k)^(n+1)
    // We use recurrence to shift to large x, then asymptotic series.
    if (n < 0) {
        throw std::runtime_error("polygamma: order n must be non-negative");
    }

    auto polygamma_fn = [n](double x) -> double {
        if (n == 0) {
            // Digamma
            double result = 0.0;
            if (x < 0.5) {
                double y = 1.0 - x;
                double r = 0.0;
                while (y < 7.0) { r -= 1.0 / y; y += 1.0; }
                double y2 = 1.0 / (y * y);
                r += std::log(y) - 0.5 / y
                    - y2 * (1.0/12.0 - y2 * (1.0/120.0 - y2 * (1.0/252.0
                    - y2 * (1.0/240.0 - y2 * (1.0/132.0)))));
                return r - M_PI / std::tan(M_PI * x);
            }
            while (x < 7.0) { result -= 1.0 / x; x += 1.0; }
            double x2 = 1.0 / (x * x);
            result += std::log(x) - 0.5 / x
                    - x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0
                    - x2 * (1.0/240.0 - x2 * (1.0/132.0)))));
            return result;
        }
        // For n >= 1: direct summation ψ^(n)(x) = (-1)^(n+1) * n! * Σ 1/(x+k)^(n+1)
        // converges only as O(1/k^n), so a 100-term truncation leaves a tail of
        // order 1/100 ≈ 0.01 for n=1 — a huge bias that made the digamma
        // analytical gradient disagree with finite-difference at Float64. Fix:
        // shift x by recurrence until x >= 7, accumulating the closed-form
        // contributions, then evaluate the asymptotic Bernoulli expansion
        // (which converges super-polynomially for x ≥ 7).
        double fact_n = 1.0;
        for (int64_t k = 1; k <= n; ++k) fact_n *= k;
        const double sign = ((n + 1) % 2 == 0) ? 1.0 : -1.0;

        // Recurrence: ψ^(n)(x) = ψ^(n)(x+1) + sign * n! / x^(n+1)
        double recurrence_sum = 0.0;
        while (x < 7.0) {
            recurrence_sum += 1.0 / std::pow(x, n + 1);
            x += 1.0;
        }

        // Asymptotic expansion for large x:
        //   ψ^(n)(x) ≈ sign * [ (n-1)!/x^n + n!/(2 x^(n+1))
        //                       + Σ_{k=1..∞} B_{2k} (2k+n-1)! / (2k)! / x^(2k+n) ]
        // where the leading (n-1)!/x^n is present for n ≥ 1 with a positive
        // sign multiplier absorbed into `sign` below. We use the first six
        // Bernoulli terms (B2..B12); at x ≥ 7 the residual is O(1/x^15) which
        // is well below Float64 precision.
        double fact_nm1 = (n >= 1) ? fact_n / static_cast<double>(n) : 1.0;
        double asym = fact_nm1 / std::pow(x, n)
                    + fact_n / (2.0 * std::pow(x, n + 1));

        static constexpr double B2[6] = {
             1.0 / 6.0,        // B_2
            -1.0 / 30.0,       // B_4
             1.0 / 42.0,       // B_6
            -1.0 / 30.0,       // B_8
             5.0 / 66.0,       // B_10
          -691.0 / 2730.0      // B_12
        };
        double fact_2k = 2.0;   // (2k)! starting at k=1: 2! = 2
        // (2k + n - 1)! = Γ(2k+n)
        auto factd = [](double m) {
            double f = 1.0;
            for (int i = 2; i <= static_cast<int>(m); ++i) f *= i;
            return f;
        };
        for (int k = 1; k <= 6; ++k) {
            double two_k = 2.0 * k;
            double num = factd(two_k + n - 1.0);
            if (k > 1) fact_2k *= (two_k - 1.0) * two_k;
            asym += B2[k - 1] * num / fact_2k / std::pow(x, two_k + n);
        }

        return sign * fact_n * (recurrence_sum + asym / fact_n);
    };

    return unary_math_kernel(input,
        [&polygamma_fn](float x) { return static_cast<float>(polygamma_fn(static_cast<double>(x))); },
        polygamma_fn, "polygamma");
}

auto beta_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    // B(a,b) = Γ(a)*Γ(b) / Γ(a+b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))
    return binary_math_kernel(a, b,
        [](float x, float y) { return std::exp(std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y)); },
        [](double x, double y) { return std::exp(std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y)); },
        "beta");
}

auto betainc_kernel(std::span<const Tensor> inputs) -> Tensor {
    // Regularized incomplete beta function I_x(a,b) using continued fraction (Lentz's method)
    if (inputs.size() != 3) {
        throw std::runtime_error("betainc requires 3 inputs (a, b, x)");
    }
    const auto& a_t = inputs[0];
    const auto& b_t = inputs[1];
    const auto& x_t = inputs[2];

    std::function<double(double, double, double)> betainc_impl;
    betainc_impl = [&betainc_impl](double a, double b, double x) -> double {
        if (x < 0.0 || x > 1.0) return std::numeric_limits<double>::quiet_NaN();
        if (x == 0.0) return 0.0;
        if (x == 1.0) return 1.0;

        // Use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
        if (x > (a + 1.0) / (a + b + 2.0)) {
            return 1.0 - betainc_impl(b, a, 1.0 - x);
        }

        // Continued fraction (Lentz's algorithm)
        double lbeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
        double front = std::exp(std::log(x) * a + std::log(1.0 - x) * b - lbeta) / a;

        double f = 1.0, c = 1.0, d = 1.0 - (a + b) * x / (a + 1.0);
        if (std::abs(d) < 1e-30) d = 1e-30;
        d = 1.0 / d;
        f = d;

        for (int m = 1; m <= 200; ++m) {
            // Even step
            double num = m * (b - m) * x / ((a + 2.0*m - 1.0) * (a + 2.0*m));
            d = 1.0 + num * d; if (std::abs(d) < 1e-30) d = 1e-30; d = 1.0 / d;
            c = 1.0 + num / c; if (std::abs(c) < 1e-30) c = 1e-30;
            f *= d * c;

            // Odd step
            num = -((a + m) * (a + b + m) * x) / ((a + 2.0*m) * (a + 2.0*m + 1.0));
            d = 1.0 + num * d; if (std::abs(d) < 1e-30) d = 1e-30; d = 1.0 / d;
            c = 1.0 + num / c; if (std::abs(c) < 1e-30) c = 1e-30;
            double delta = d * c;
            f *= delta;
            if (std::abs(delta - 1.0) < 1e-12) break;
        }
        return front * f;
    };

    // Float16/BFloat16: upcast inputs to Float32, compute, cast result back.
    if (a_t.dtype() == DType::Float16 || a_t.dtype() == DType::BFloat16) {
        auto orig_dtype = a_t.dtype();
        std::vector<Tensor> upcast = {
            a_t.to(DType::Float32),
            b_t.to(DType::Float32),
            x_t.to(DType::Float32),
        };
        return betainc_kernel(upcast).to(orig_dtype);
    }

    // Element-wise ternary operation
    auto out_shape = a_t.shape();
    Tensor result(std::vector<int64_t>(out_shape.begin(), out_shape.end()), a_t.dtype(), a_t.device());
    size_t n = static_cast<size_t>(a_t.numel());

    if (a_t.dtype() == DType::Float32) {
        const float* ad = a_t.data<float>();
        const float* bd = b_t.data<float>();
        const float* xd = x_t.data<float>();
        float* od = result.data<float>();
        for (size_t i = 0; i < n; ++i) {
            od[i] = static_cast<float>(betainc_impl(ad[i], bd[i], xd[i]));
        }
    } else if (a_t.dtype() == DType::Float64) {
        const double* ad = a_t.data<double>();
        const double* bd = b_t.data<double>();
        const double* xd = x_t.data<double>();
        double* od = result.data<double>();
        for (size_t i = 0; i < n; ++i) {
            od[i] = betainc_impl(ad[i], bd[i], xd[i]);
        }
    } else {
        throw std::runtime_error("betainc: only Float32 and Float64 supported");
    }
    return result;
}

auto bessel_j0_kernel(const Tensor& input) -> Tensor {
#if __cplusplus >= 201703L && defined(__cpp_lib_math_special_functions)
    return unary_math_kernel(input,
        [](float x) { return static_cast<float>(std::cyl_bessel_j(0, static_cast<double>(x))); },
        [](double x) { return std::cyl_bessel_j(0, x); }, "bessel_j0");
#else
    // Fallback: polynomial approximation (Abramowitz & Stegun)
    auto j0_approx = [](double x) -> double {
        x = std::abs(x);
        if (x <= 3.0) {
            double y = x * x / 9.0;
            return 1.0 - y * (2.2499997 - y * (1.2656208 - y * (0.3163866
                   - y * (0.0444479 - y * (0.0039444 - y * 0.0002100)))));
        }
        double ax = 3.0 / x;
        double p = 0.79788456 - ax * (0.00000077 + ax * (0.00552740 + ax * (0.00009512
                 - ax * (0.00137237 - ax * (0.00072805 - ax * 0.00014476)))));
        double q = -0.04166397 - ax * (0.00003954 - ax * (0.00262573 - ax * (0.00054125
                 + ax * (0.00029333 - ax * (0.00013558 + ax * 0.00000000)))));
        double z = x - 0.785398164;
        return std::sqrt(2.0 / (M_PI * x)) * (p * std::cos(z) - q * std::sin(z));
    };
    return unary_math_kernel(input,
        [&j0_approx](float x) { return static_cast<float>(j0_approx(x)); },
        j0_approx, "bessel_j0");
#endif
}

auto bessel_j1_kernel(const Tensor& input) -> Tensor {
#if __cplusplus >= 201703L && defined(__cpp_lib_math_special_functions)
    return unary_math_kernel(input,
        [](float x) { return static_cast<float>(std::cyl_bessel_j(1, static_cast<double>(x))); },
        [](double x) { return std::cyl_bessel_j(1, x); }, "bessel_j1");
#else
    auto j1_approx = [](double x) -> double {
        double sign = (x < 0) ? -1.0 : 1.0;
        x = std::abs(x);
        if (x <= 3.0) {
            double y = x * x / 9.0;
            return sign * x * (0.5 - y * (0.56249985 - y * (0.21093573 - y * (0.03954289
                   - y * (0.00443319 - y * (0.00031761 - y * 0.00001109))))));
        }
        double ax = 3.0 / x;
        double p = 0.79788456 + ax * (0.00000156 + ax * (0.01659667 + ax * (0.00017105
                 - ax * (0.00249511 + ax * (0.00113653 - ax * 0.00020033)))));
        double q = 0.12499612 + ax * (0.00005650 - ax * (0.00637879 + ax * (0.00074348
                 + ax * (0.00079824 - ax * (0.00029166 + ax * 0.00000000)))));
        double z = x - 2.356194491;
        return sign * std::sqrt(2.0 / (M_PI * x)) * (p * std::cos(z) - q * std::sin(z));
    };
    return unary_math_kernel(input,
        [&j1_approx](float x) { return static_cast<float>(j1_approx(x)); },
        j1_approx, "bessel_j1");
#endif
}

auto bessel_y0_kernel(const Tensor& input) -> Tensor {
#if __cplusplus >= 201703L && defined(__cpp_lib_math_special_functions)
    return unary_math_kernel(input,
        [](float x) { return static_cast<float>(std::cyl_neumann(0, static_cast<double>(x))); },
        [](double x) { return std::cyl_neumann(0, x); }, "bessel_y0");
#else
    auto y0_approx = [](double x) -> double {
        if (x <= 0.0) return -std::numeric_limits<double>::infinity();
        if (x <= 3.0) {
            double y = x * x / 9.0;
            return (2.0 / M_PI) * std::log(x / 2.0) * std::cyl_bessel_j(0, x)
                   + 0.36746691 + y * (0.60559366 - y * (0.74350384 - y * (0.25300117
                   - y * (0.04261214 - y * (0.00427916 - y * 0.00024846)))));
        }
        double ax = 3.0 / x;
        double p = 0.79788456 - ax * (0.00000077 + ax * (0.00552740 + ax * (0.00009512
                 - ax * (0.00137237 - ax * (0.00072805 - ax * 0.00014476)))));
        double q = -0.04166397 - ax * (0.00003954 - ax * (0.00262573 - ax * (0.00054125
                 + ax * (0.00029333 - ax * (0.00013558)))));
        double z = x - 0.785398164;
        return std::sqrt(2.0 / (M_PI * x)) * (p * std::sin(z) + q * std::cos(z));
    };
    return unary_math_kernel(input,
        [&y0_approx](float x) { return static_cast<float>(y0_approx(x)); },
        y0_approx, "bessel_y0");
#endif
}

auto bessel_y1_kernel(const Tensor& input) -> Tensor {
#if __cplusplus >= 201703L && defined(__cpp_lib_math_special_functions)
    return unary_math_kernel(input,
        [](float x) { return static_cast<float>(std::cyl_neumann(1, static_cast<double>(x))); },
        [](double x) { return std::cyl_neumann(1, x); }, "bessel_y1");
#else
    auto y1_approx = [](double x) -> double {
        if (x <= 0.0) return -std::numeric_limits<double>::infinity();
        if (x <= 3.0) {
            double y = x * x / 9.0;
            return (2.0 / M_PI) * (std::log(x / 2.0) * std::cyl_bessel_j(1, x) - 1.0 / x)
                   + x * (0.02635537 + y * (-0.04985710 + y * (-0.00121547 + y * (0.00127120
                   - y * (0.00023895 + y * (0.00002535))))));
        }
        double ax = 3.0 / x;
        double p = 0.79788456 + ax * (0.00000156 + ax * (0.01659667 + ax * (0.00017105
                 - ax * (0.00249511 + ax * (0.00113653 - ax * 0.00020033)))));
        double q = 0.12499612 + ax * (0.00005650 - ax * (0.00637879 + ax * (0.00074348
                 + ax * (0.00079824 - ax * (0.00029166)))));
        double z = x - 2.356194491;
        return std::sqrt(2.0 / (M_PI * x)) * (p * std::sin(z) + q * std::cos(z));
    };
    return unary_math_kernel(input,
        [&y1_approx](float x) { return static_cast<float>(y1_approx(x)); },
        y1_approx, "bessel_y1");
#endif
}

auto bessel_i0_kernel(const Tensor& input) -> Tensor {
    // Modified Bessel function I_0(x), polynomial approximation (Abramowitz & Stegun 9.8.1/9.8.2)
    auto i0_approx = [](double x) -> double {
        double ax = std::abs(x);
        if (ax < 3.75) {
            double t = x / 3.75;
            t = t * t;
            return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
                   + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        }
        double t = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) * (0.39894228 + t * (0.01328592
               + t * (0.00225319 - t * (0.00157565 - t * (0.00916281
               - t * (0.02057706 - t * (0.02635537 - t * (0.01647633
               - t * 0.00392377))))))));
    };
    return unary_math_kernel(input,
        [&i0_approx](float x) { return static_cast<float>(i0_approx(x)); },
        i0_approx, "bessel_i0");
}

auto bessel_i1_kernel(const Tensor& input) -> Tensor {
    // Modified Bessel function I_1(x), polynomial approximation (Abramowitz & Stegun 9.8.3/9.8.4)
    auto i1_approx = [](double x) -> double {
        double ax = std::abs(x);
        double result;
        if (ax < 3.75) {
            double t = x / 3.75;
            t = t * t;
            result = ax * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934
                     + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
        } else {
            double t = 3.75 / ax;
            result = (std::exp(ax) / std::sqrt(ax)) * (0.39894228 - t * (0.03988024
                     - t * (0.00362018 + t * (0.00163801 - t * (0.01031555
                     - t * (0.02282967 - t * (0.02895312 - t * (0.01787654
                     - t * 0.00420059))))))));
        }
        return (x < 0.0) ? -result : result;
    };
    return unary_math_kernel(input,
        [&i1_approx](float x) { return static_cast<float>(i1_approx(x)); },
        i1_approx, "bessel_i1");
}

auto erfinv_kernel(const Tensor& input) -> Tensor {
    // Inverse error function using rational approximation (Winitzki 2008)
    auto erfinv_impl = [](double x) -> double {
        if (x <= -1.0) return -std::numeric_limits<double>::infinity();
        if (x >= 1.0) return std::numeric_limits<double>::infinity();
        if (x == 0.0) return 0.0;

        double a = 0.147;  // Winitzki constant
        double ln1mx2 = std::log(1.0 - x * x);
        double t1 = 2.0 / (M_PI * a) + 0.5 * ln1mx2;
        double t2 = ln1mx2 / a;
        double sign = (x > 0.0) ? 1.0 : -1.0;
        double result = sign * std::sqrt(std::sqrt(t1 * t1 - t2) - t1);

        // Newton refinement (2 iterations for double precision)
        for (int i = 0; i < 2; ++i) {
            double err = std::erf(result) - x;
            double deriv = 2.0 / std::sqrt(M_PI) * std::exp(-result * result);
            result -= err / deriv;
        }
        return result;
    };
    return unary_math_kernel(input,
        [&erfinv_impl](float x) { return static_cast<float>(erfinv_impl(x)); },
        erfinv_impl, "erfinv");
}

auto sinc_kernel(const Tensor& input) -> Tensor {
    // Normalized sinc: sin(πx)/(πx), with sinc(0) = 1
    return unary_math_kernel(input,
        [](float x) {
            if (x == 0.0f) return 1.0f;
            float px = static_cast<float>(M_PI) * x;
            return std::sin(px) / px;
        },
        [](double x) {
            if (x == 0.0) return 1.0;
            double px = M_PI * x;
            return std::sin(px) / px;
        }, "sinc");
}

auto zeta_kernel(const Tensor& x, const Tensor& q) -> Tensor {
    // Hurwitz zeta: ζ(s,q) = Σ_{n=0}^∞ 1/(q+n)^s
    // Uses Euler-Maclaurin summation for convergence
    // Euler-Maclaurin with the leading Bernoulli correction terms (B2,B4,B6).
    // Without them the truncated sum carries an O(s·(a+N)^(-s-1)) error
    // (~7.6e-5 for ζ(2,1)); the correction block drops it below ~1e-9.
    auto hurwitz_zeta = [](double s, double a) -> double {
        double result = 0.0;
        for (int n = 0; n < 12; ++n) {
            result += std::pow(a + n, -s);
        }
        const double aN = a + 12.0;
        if (s != 1.0) result += std::pow(aN, 1.0 - s) / (s - 1.0);
        result += 0.5 * std::pow(aN, -s);
        // Σ_k B_{2k}/(2k)! · (s)_{2k-1} · (a+N)^(-s-2k+1):
        //   B2/2! = 1/12, B4/4! = -1/720, B6/6! = 1/30240.
        result += (s) * std::pow(aN, -s - 1.0) / 12.0;
        result -= (s * (s + 1.0) * (s + 2.0)) * std::pow(aN, -s - 3.0) / 720.0;
        result += (s * (s + 1.0) * (s + 2.0) * (s + 3.0) * (s + 4.0))
                  * std::pow(aN, -s - 5.0) / 30240.0;
        return result;
    };
    return binary_math_kernel(x, q,
        [hurwitz_zeta](float s, float a) -> float {
            return static_cast<float>(hurwitz_zeta(static_cast<double>(s),
                                                   static_cast<double>(a)));
        },
        [hurwitz_zeta](double s, double a) -> double {
            return hurwitz_zeta(s, a);
        }, "zeta");
}

// ============================================================================
// New element-wise ops: Frac, Heaviside, NanToNum
// ============================================================================

auto frac_kernel(const Tensor& input) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        auto result = frac_kernel(f32);
        return result.to(input.dtype());
    }

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    // frac is sign-preserving: frac(x) = x - trunc(x). Matches PyTorch's
    // convention and the fixed CUDA / ROCm / Vulkan / OneAPI kernels. Using
    // floor here was a long-standing API divergence (frac(-2.75) returned
    // 0.25 instead of -0.75) that this project is fixing across backends.
    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "frac", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = output.data<scalar_t>();
        _Pragma("omp parallel for if(n > 10000)")
        for (int64_t i = 0; i < n; i++) {
            out_data[i] = in_data[i] - std::trunc(in_data[i]);
        }
    });
    return output;
}

auto heaviside_kernel(const Tensor& input, const Tensor& values) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32_in = input.to(DType::Float32);
        auto f32_val = values.to(DType::Float32);
        auto result = heaviside_kernel(f32_in, f32_val);
        return result.to(input.dtype());
    }

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "heaviside", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        const scalar_t* val_data = values.data<scalar_t>();
        scalar_t* out_data = output.data<scalar_t>();
        _Pragma("omp parallel for if(n > 10000)")
        for (int64_t i = 0; i < n; i++) {
            scalar_t x = in_data[i];
            if (x < scalar_t(0)) out_data[i] = scalar_t(0);
            else if (x == scalar_t(0)) out_data[i] = val_data[i];
            else out_data[i] = scalar_t(1);
        }
    });
    return output;
}

auto nan_to_num_kernel(const Tensor& input, double nan_val, double posinf_val, double neginf_val) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        auto result = nan_to_num_kernel(f32, nan_val, posinf_val, neginf_val);
        return result.to(input.dtype());
    }

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "nan_to_num", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = output.data<scalar_t>();
        scalar_t nan_rep = static_cast<scalar_t>(nan_val);
        scalar_t posinf_rep = (posinf_val == std::numeric_limits<double>::max())
            ? std::numeric_limits<scalar_t>::max()
            : static_cast<scalar_t>(posinf_val);
        scalar_t neginf_rep = (neginf_val == std::numeric_limits<double>::lowest())
            ? std::numeric_limits<scalar_t>::lowest()
            : static_cast<scalar_t>(neginf_val);
        _Pragma("omp parallel for if(n > 10000)")
        for (int64_t i = 0; i < n; i++) {
            scalar_t x = in_data[i];
            if (std::isnan(x)) out_data[i] = nan_rep;
            else if (std::isinf(x) && x > 0) out_data[i] = posinf_rep;
            else if (std::isinf(x) && x < 0) out_data[i] = neginf_rep;
            else out_data[i] = x;
        }
    });
    return output;
}

// ============================================================================
// New activations: LogSigmoid, RReLU
// ============================================================================

auto log_sigmoid_kernel(const Tensor& input) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        auto result = log_sigmoid_kernel(f32);
        return result.to(input.dtype());
    }

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "log_sigmoid", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = output.data<scalar_t>();
        _Pragma("omp parallel for if(n > 10000)")
        for (int64_t i = 0; i < n; i++) {
            scalar_t x = in_data[i];
            // log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x))
            // Numerically stable: use log(sigmoid(x)) = x - softplus(x)
            //   for x < 0: log(sigmoid(x)) = x - log(1 + exp(x))
            //   for x >= 0: log(sigmoid(x)) = -log(1 + exp(-x))
            if (x >= scalar_t(0)) {
                out_data[i] = -std::log1p(std::exp(-x));
            } else {
                out_data[i] = x - std::log1p(std::exp(x));
            }
        }
    });
    return output;
}

auto log_sigmoid_backward_kernel(const Tensor& grad, const Tensor& input) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32_g = grad.to(DType::Float32);
        auto f32_in = input.to(DType::Float32);
        auto result = log_sigmoid_backward_kernel(f32_g, f32_in);
        return result.to(input.dtype());
    }

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "log_sigmoid_backward", [&]() {
        const scalar_t* g_data = grad.data<scalar_t>();
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = output.data<scalar_t>();
        _Pragma("omp parallel for if(n > 10000)")
        for (int64_t i = 0; i < n; i++) {
            scalar_t x = in_data[i];
            // d/dx log(sigmoid(x)) = 1 - sigmoid(x) = sigmoid(-x)
            scalar_t sig_neg_x;
            if (x >= scalar_t(0)) {
                sig_neg_x = std::exp(-x) / (scalar_t(1) + std::exp(-x));
            } else {
                sig_neg_x = scalar_t(1) / (scalar_t(1) + std::exp(x));
            }
            out_data[i] = g_data[i] * sig_neg_x;
        }
    });
    return output;
}

auto rrelu_kernel(const Tensor& input, float lower, float upper, bool training) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        auto result = rrelu_kernel(f32, lower, upper, training);
        return result.to(input.dtype());
    }

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "rrelu", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = output.data<scalar_t>();
        if (training) {
            // Each element gets a random slope in [lower, upper]
            std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<float> dist(lower, upper);
            for (int64_t i = 0; i < n; i++) {
                scalar_t x = in_data[i];
                if (x >= scalar_t(0)) {
                    out_data[i] = x;
                } else {
                    out_data[i] = static_cast<scalar_t>(dist(gen)) * x;
                }
            }
        } else {
            // Use midpoint of range in eval mode
            scalar_t slope = static_cast<scalar_t>((lower + upper) / 2.0f);
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                scalar_t x = in_data[i];
                out_data[i] = x >= scalar_t(0) ? x : slope * x;
            }
        }
    });
    return output;
}

auto rrelu_backward_kernel(const Tensor& grad, const Tensor& input, float lower, float upper) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32_g = grad.to(DType::Float32);
        auto f32_in = input.to(DType::Float32);
        auto result = rrelu_backward_kernel(f32_g, f32_in, lower, upper);
        return result.to(input.dtype());
    }

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "rrelu_backward", [&]() {
        const scalar_t* g_data = grad.data<scalar_t>();
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = output.data<scalar_t>();
        scalar_t slope = static_cast<scalar_t>((lower + upper) / 2.0f);
        _Pragma("omp parallel for if(n > 10000)")
        for (int64_t i = 0; i < n; i++) {
            out_data[i] = in_data[i] >= scalar_t(0) ? g_data[i] : g_data[i] * slope;
        }
    });
    return output;
}

// ============================================================================
// Bitwise operations (integer types)
// ============================================================================

auto bitwise_and_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<int64_t> shape_vec(a.shape().begin(), a.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, a.dtype(), a.device());
    int64_t n = a.numel();

    switch (a.dtype()) {
        case DType::Int32: {
            const int32_t* a_d = a.data<int32_t>();
            const int32_t* b_d = b.data<int32_t>();
            int32_t* o_d = output.data<int32_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] & b_d[i];
            break;
        }
        case DType::Int64: {
            const int64_t* a_d = a.data<int64_t>();
            const int64_t* b_d = b.data<int64_t>();
            int64_t* o_d = output.data<int64_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] & b_d[i];
            break;
        }
        case DType::Int8: {
            const int8_t* a_d = a.data<int8_t>();
            const int8_t* b_d = b.data<int8_t>();
            int8_t* o_d = output.data<int8_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] & b_d[i];
            break;
        }
        case DType::Int16: {
            const int16_t* a_d = a.data<int16_t>();
            const int16_t* b_d = b.data<int16_t>();
            int16_t* o_d = output.data<int16_t>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] & b_d[i];
            break;
        }
        case DType::Bool: {
            const bool* a_d = a.data<bool>();
            const bool* b_d = b.data<bool>();
            bool* o_d = output.data<bool>();
            _Pragma("omp parallel for if(n > 10000)")
            for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] && b_d[i];
            break;
        }
        default:
            throw std::runtime_error("bitwise_and: unsupported dtype (expected integer or bool)");
    }
    return output;
}

auto bitwise_or_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<int64_t> shape_vec(a.shape().begin(), a.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, a.dtype(), a.device());
    int64_t n = a.numel();

    switch (a.dtype()) {
        case DType::Int32: {
            const int32_t* a_d = a.data<int32_t>(); const int32_t* b_d = b.data<int32_t>(); int32_t* o_d = output.data<int32_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] | b_d[i]; break;
        }
        case DType::Int64: {
            const int64_t* a_d = a.data<int64_t>(); const int64_t* b_d = b.data<int64_t>(); int64_t* o_d = output.data<int64_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] | b_d[i]; break;
        }
        case DType::Int8: {
            const int8_t* a_d = a.data<int8_t>(); const int8_t* b_d = b.data<int8_t>(); int8_t* o_d = output.data<int8_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] | b_d[i]; break;
        }
        case DType::Int16: {
            const int16_t* a_d = a.data<int16_t>(); const int16_t* b_d = b.data<int16_t>(); int16_t* o_d = output.data<int16_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] | b_d[i]; break;
        }
        case DType::Bool: {
            const bool* a_d = a.data<bool>(); const bool* b_d = b.data<bool>(); bool* o_d = output.data<bool>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] || b_d[i]; break;
        }
        default: throw std::runtime_error("bitwise_or: unsupported dtype");
    }
    return output;
}

auto bitwise_xor_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<int64_t> shape_vec(a.shape().begin(), a.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, a.dtype(), a.device());
    int64_t n = a.numel();

    switch (a.dtype()) {
        case DType::Int32: {
            const int32_t* a_d = a.data<int32_t>(); const int32_t* b_d = b.data<int32_t>(); int32_t* o_d = output.data<int32_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] ^ b_d[i]; break;
        }
        case DType::Int64: {
            const int64_t* a_d = a.data<int64_t>(); const int64_t* b_d = b.data<int64_t>(); int64_t* o_d = output.data<int64_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] ^ b_d[i]; break;
        }
        case DType::Int8: {
            const int8_t* a_d = a.data<int8_t>(); const int8_t* b_d = b.data<int8_t>(); int8_t* o_d = output.data<int8_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] ^ b_d[i]; break;
        }
        case DType::Int16: {
            const int16_t* a_d = a.data<int16_t>(); const int16_t* b_d = b.data<int16_t>(); int16_t* o_d = output.data<int16_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] ^ b_d[i]; break;
        }
        case DType::Bool: {
            const bool* a_d = a.data<bool>(); const bool* b_d = b.data<bool>(); bool* o_d = output.data<bool>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = a_d[i] != b_d[i]; break;
        }
        default: throw std::runtime_error("bitwise_xor: unsupported dtype");
    }
    return output;
}

auto bitwise_not_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Int32: {
            const int32_t* i_d = input.data<int32_t>(); int32_t* o_d = output.data<int32_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = ~i_d[i]; break;
        }
        case DType::Int64: {
            const int64_t* i_d = input.data<int64_t>(); int64_t* o_d = output.data<int64_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = ~i_d[i]; break;
        }
        case DType::Int8: {
            const int8_t* i_d = input.data<int8_t>(); int8_t* o_d = output.data<int8_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = ~i_d[i]; break;
        }
        case DType::Int16: {
            const int16_t* i_d = input.data<int16_t>(); int16_t* o_d = output.data<int16_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = ~i_d[i]; break;
        }
        case DType::Bool: {
            const bool* i_d = input.data<bool>(); bool* o_d = output.data<bool>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = !i_d[i]; break;
        }
        default: throw std::runtime_error("bitwise_not: unsupported dtype");
    }
    return output;
}

auto bitwise_left_shift_kernel(const Tensor& input, const Tensor& shift) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Int32: {
            const int32_t* i_d = input.data<int32_t>(); const int32_t* s_d = shift.data<int32_t>(); int32_t* o_d = output.data<int32_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = i_d[i] << s_d[i]; break;
        }
        case DType::Int64: {
            const int64_t* i_d = input.data<int64_t>(); const int64_t* s_d = shift.data<int64_t>(); int64_t* o_d = output.data<int64_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = i_d[i] << s_d[i]; break;
        }
        case DType::Int8: {
            const int8_t* i_d = input.data<int8_t>(); const int8_t* s_d = shift.data<int8_t>(); int8_t* o_d = output.data<int8_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = i_d[i] << s_d[i]; break;
        }
        case DType::Int16: {
            const int16_t* i_d = input.data<int16_t>(); const int16_t* s_d = shift.data<int16_t>(); int16_t* o_d = output.data<int16_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = i_d[i] << s_d[i]; break;
        }
        default: throw std::runtime_error("bitwise_left_shift: unsupported dtype (expected integer)");
    }
    return output;
}

auto bitwise_right_shift_kernel(const Tensor& input, const Tensor& shift) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Int32: {
            const int32_t* i_d = input.data<int32_t>(); const int32_t* s_d = shift.data<int32_t>(); int32_t* o_d = output.data<int32_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = i_d[i] >> s_d[i]; break;
        }
        case DType::Int64: {
            const int64_t* i_d = input.data<int64_t>(); const int64_t* s_d = shift.data<int64_t>(); int64_t* o_d = output.data<int64_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = i_d[i] >> s_d[i]; break;
        }
        case DType::Int8: {
            const int8_t* i_d = input.data<int8_t>(); const int8_t* s_d = shift.data<int8_t>(); int8_t* o_d = output.data<int8_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = i_d[i] >> s_d[i]; break;
        }
        case DType::Int16: {
            const int16_t* i_d = input.data<int16_t>(); const int16_t* s_d = shift.data<int16_t>(); int16_t* o_d = output.data<int16_t>();
            _Pragma("omp parallel for if(n > 10000)") for (int64_t i = 0; i < n; i++) o_d[i] = i_d[i] >> s_d[i]; break;
        }
        default: throw std::runtime_error("bitwise_right_shift: unsupported dtype (expected integer)");
    }
    return output;
}

// ============================================================================
// NaN-aware reductions: CountNonzero, Nansum, Nanmean, Aminmax
// ============================================================================

auto count_nonzero_kernel(const Tensor& input, int64_t dim) -> Tensor {
    // Full reduction (no dim)
    if (dim < 0) {
        int64_t n = input.numel();
        int64_t count = 0;
        TENZOR_DISPATCH_ALL_TYPES(input.dtype(), "count_nonzero", [&]() {
            const scalar_t* data = input.data<scalar_t>();
            _Pragma("omp parallel for reduction(+:count) if(n > 10000)")
            for (int64_t i = 0; i < n; i++) {
                if (data[i] != static_cast<scalar_t>(0.0f)) count++;
            }
        });
        auto result = Tensor({1}, DType::Int64, input.device());
        *result.data<int64_t>() = count;
        return result;
    }

    // Dimensional reduction
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t reduce_size = shape[dim];

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != dim) out_shape.push_back(shape[d]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    auto result = Tensor(out_shape, DType::Int64, input.device());
    int64_t* out_data = result.data<int64_t>();
    int64_t out_n = outer * inner;

    TENZOR_DISPATCH_ALL_TYPES(input.dtype(), "count_nonzero_dim", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        _Pragma("omp parallel for if(out_n > 1000)")
        for (int64_t idx = 0; idx < out_n; idx++) {
            int64_t o = idx / inner;
            int64_t i_inner = idx % inner;
            int64_t count = 0;
            for (int64_t r = 0; r < reduce_size; r++) {
                int64_t src_idx = (o * reduce_size + r) * inner + i_inner;
                if (in_data[src_idx] != static_cast<scalar_t>(0.0f)) count++;
            }
            out_data[idx] = count;
        }
    });
    return result;
}

auto nansum_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        auto result = nansum_kernel(f32, dim, keepdim);
        return result.to(input.dtype());
    }

    // Full reduction
    if (dim < 0) {
        int64_t n = input.numel();
        Tensor result({1}, input.dtype(), input.device());
        TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "nansum", [&]() {
            const scalar_t* data = input.data<scalar_t>();
            scalar_t acc = 0;
            for (int64_t i = 0; i < n; i++) {
                scalar_t v = data[i];
                if (!std::isnan(v)) acc += v;
            }
            *result.data<scalar_t>() = acc;
        });
        return result;
    }

    // Dim reduction: use the where+sum pattern
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t reduce_size = shape[dim];

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape[d]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    auto result = Tensor(out_shape, input.dtype(), input.device());
    int64_t out_n = outer * inner;

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "nansum_dim", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = result.data<scalar_t>();
        _Pragma("omp parallel for if(out_n > 1000)")
        for (int64_t idx = 0; idx < out_n; idx++) {
            int64_t o = idx / inner;
            int64_t i_inner = idx % inner;
            scalar_t acc = 0;
            for (int64_t r = 0; r < reduce_size; r++) {
                int64_t src_idx = (o * reduce_size + r) * inner + i_inner;
                scalar_t v = in_data[src_idx];
                if (!std::isnan(v)) acc += v;
            }
            out_data[idx] = acc;
        }
    });
    return result;
}

auto nanmean_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        auto result = nanmean_kernel(f32, dim, keepdim);
        return result.to(input.dtype());
    }

    if (dim < 0) {
        int64_t n = input.numel();
        Tensor result({1}, input.dtype(), input.device());
        TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "nanmean", [&]() {
            const scalar_t* data = input.data<scalar_t>();
            scalar_t acc = 0;
            int64_t count = 0;
            for (int64_t i = 0; i < n; i++) {
                scalar_t v = data[i];
                if (!std::isnan(v)) { acc += v; count++; }
            }
            *result.data<scalar_t>() = count > 0 ? acc / static_cast<scalar_t>(count) : scalar_t(0);
        });
        return result;
    }

    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t reduce_size = shape[dim];

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape[d]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    auto result = Tensor(out_shape, input.dtype(), input.device());
    int64_t out_n = outer * inner;

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "nanmean_dim", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = result.data<scalar_t>();
        _Pragma("omp parallel for if(out_n > 1000)")
        for (int64_t idx = 0; idx < out_n; idx++) {
            int64_t o = idx / inner;
            int64_t i_inner = idx % inner;
            scalar_t acc = 0;
            int64_t count = 0;
            for (int64_t r = 0; r < reduce_size; r++) {
                int64_t src_idx = (o * reduce_size + r) * inner + i_inner;
                scalar_t v = in_data[src_idx];
                if (!std::isnan(v)) { acc += v; count++; }
            }
            out_data[idx] = count > 0 ? acc / static_cast<scalar_t>(count) : scalar_t(0);
        }
    });
    return result;
}

auto aminmax_kernel(const Tensor& input, int64_t dim, bool keepdim) -> std::pair<Tensor, Tensor> {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        auto [mn, mx] = aminmax_kernel(f32, dim, keepdim);
        return {mn.to(input.dtype()), mx.to(input.dtype())};
    }

    if (dim < 0) {
        int64_t n = input.numel();
        Tensor min_out({1}, input.dtype(), input.device());
        Tensor max_out({1}, input.dtype(), input.device());
        TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "aminmax", [&]() {
            const scalar_t* data = input.data<scalar_t>();
            scalar_t mn = data[0], mx = data[0];
            for (int64_t i = 1; i < n; i++) {
                if (data[i] < mn) mn = data[i];
                if (data[i] > mx) mx = data[i];
            }
            *min_out.data<scalar_t>() = mn;
            *max_out.data<scalar_t>() = mx;
        });
        return {min_out, max_out};
    }

    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t reduce_size = shape[dim];

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape[d]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    auto min_out = Tensor(out_shape, input.dtype(), input.device());
    auto max_out = Tensor(out_shape, input.dtype(), input.device());
    int64_t out_n = outer * inner;

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "aminmax_dim", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* min_data = min_out.data<scalar_t>();
        scalar_t* max_data = max_out.data<scalar_t>();
        _Pragma("omp parallel for if(out_n > 1000)")
        for (int64_t idx = 0; idx < out_n; idx++) {
            int64_t o = idx / inner;
            int64_t i_inner = idx % inner;
            int64_t base = (o * reduce_size) * inner + i_inner;
            scalar_t mn = in_data[base], mx = in_data[base];
            for (int64_t r = 1; r < reduce_size; r++) {
                scalar_t v = in_data[base + r * inner];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            min_data[idx] = mn;
            max_data[idx] = mx;
        }
    });
    return {min_out, max_out};
}

// ============================================================================
// Scatter variants: IndexAdd, IndexCopy, IndexFill
// ============================================================================

auto index_add_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source) -> Tensor {
    auto output = input.clone();
    int64_t ndim = output.shape().size();
    if (dim < 0) dim += ndim;
    auto shape = output.shape();
    int64_t dim_size = shape[dim];
    int64_t idx_n = index.numel();

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    const int64_t* idx_data = index.data<int64_t>();

    if (output.dtype() == DType::Float16 || output.dtype() == DType::BFloat16) {
        auto f32_out = output.to(DType::Float32);
        auto f32_src = source.to(DType::Float32);
        auto r = index_add_kernel(f32_out, dim, index, f32_src);
        return r.to(output.dtype());
    }
    TENZOR_DISPATCH_FLOATING_TYPES(output.dtype(), "index_add", [&]() {
        scalar_t* out_data = output.data<scalar_t>();
        const scalar_t* src_data = source.data<scalar_t>();
        for (int64_t o = 0; o < outer; o++) {
            for (int64_t k = 0; k < idx_n; k++) {
                int64_t dst_idx = idx_data[k];
                for (int64_t j = 0; j < inner; j++) {
                    out_data[(o * dim_size + dst_idx) * inner + j] +=
                        src_data[(o * idx_n + k) * inner + j];
                }
            }
        }
    });
    return output;
}

auto index_copy_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source) -> Tensor {
    auto output = input.clone();
    int64_t ndim = output.shape().size();
    if (dim < 0) dim += ndim;
    auto shape = output.shape();
    int64_t dim_size = shape[dim];
    int64_t idx_n = index.numel();

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    const int64_t* idx_data = index.data<int64_t>();

    if (output.dtype() == DType::Float16 || output.dtype() == DType::BFloat16) {
        auto f32_out = output.to(DType::Float32);
        auto f32_src = source.to(DType::Float32);
        auto r = index_copy_kernel(f32_out, dim, index, f32_src);
        return r.to(output.dtype());
    }
    TENZOR_DISPATCH_FLOATING_TYPES(output.dtype(), "index_copy", [&]() {
        scalar_t* out_data = output.data<scalar_t>();
        const scalar_t* src_data = source.data<scalar_t>();
        for (int64_t o = 0; o < outer; o++) {
            for (int64_t k = 0; k < idx_n; k++) {
                int64_t dst_idx = idx_data[k];
                for (int64_t j = 0; j < inner; j++) {
                    out_data[(o * dim_size + dst_idx) * inner + j] =
                        src_data[(o * idx_n + k) * inner + j];
                }
            }
        }
    });
    return output;
}

auto index_fill_kernel(const Tensor& input, int64_t dim, const Tensor& index, double value) -> Tensor {
    auto output = input.clone();
    int64_t ndim = output.shape().size();
    if (dim < 0) dim += ndim;
    auto shape = output.shape();
    int64_t dim_size = shape[dim];
    int64_t idx_n = index.numel();

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    const int64_t* idx_data = index.data<int64_t>();

    if (output.dtype() == DType::Float16 || output.dtype() == DType::BFloat16) {
        auto f32_out = output.to(DType::Float32);
        auto r = index_fill_kernel(f32_out, dim, index, value);
        return r.to(output.dtype());
    }
    TENZOR_DISPATCH_FLOATING_TYPES(output.dtype(), "index_fill", [&]() {
        scalar_t* out_data = output.data<scalar_t>();
        scalar_t fill_val = static_cast<scalar_t>(static_cast<float>(value));
        for (int64_t o = 0; o < outer; o++) {
            for (int64_t k = 0; k < idx_n; k++) {
                int64_t dst_idx = idx_data[k];
                for (int64_t j = 0; j < inner; j++) {
                    out_data[(o * dim_size + dst_idx) * inner + j] = fill_val;
                }
            }
        }
    });
    return output;
}

// =========================================================================
// Fused GEMM Operations: addmm, addmv, baddbmm
// =========================================================================

auto addmm_kernel(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
                  double alpha, double beta) -> Tensor {
    // input: (M, N) or broadcastable, mat1: (M, K), mat2: (K, N)
    int64_t M = mat1.shape()[0];
    int64_t K = mat1.shape()[1];
    int64_t N = mat2.shape()[1];

    Tensor a_cont = mat1.is_contiguous() ? mat1 : mat1.contiguous();
    Tensor b_cont = mat2.is_contiguous() ? mat2 : mat2.contiguous();

    // Create output as a copy of input (broadcast-expanded to (M, N))
    // For beta=0, we skip the copy and just zero-init
    Tensor output = Tensor::empty_uninitialized({M, N}, mat1.dtype(), Device::cpu());

    if (mat1.dtype() == DType::Float32) {
        float* out_data = output.data<float>();

        // Initialize output with input data (broadcast if needed)
        if (beta != 0.0) {
            const Tensor inp_cont = input.is_contiguous() ? input : input.contiguous();
            const float* inp_data = inp_cont.data<float>();
            auto inp_shape = inp_cont.shape();

            if (inp_cont.ndim() == 2 && inp_shape[0] == M && inp_shape[1] == N) {
                // Direct copy
                std::memcpy(out_data, inp_data, M * N * sizeof(float));
            } else if (inp_cont.ndim() == 1 && inp_shape[0] == N) {
                // Broadcast (N,) -> (M, N): replicate row
                for (int64_t i = 0; i < M; ++i) {
                    std::memcpy(out_data + i * N, inp_data, N * sizeof(float));
                }
            } else if (inp_cont.numel() == 1) {
                // Scalar broadcast
                float val = inp_data[0];
                std::fill_n(out_data, M * N, val);
            } else {
                // General broadcast: expand input to (M, N) first
                auto expanded = input.expand({M, N}).contiguous();
                std::memcpy(out_data, expanded.data<float>(), M * N * sizeof(float));
            }
        } else {
            std::memset(out_data, 0, M * N * sizeof(float));
        }

        float alpha_f = static_cast<float>(alpha);
        float beta_f = static_cast<float>(beta);

#ifdef TENZOR_USE_MKL
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<MKL_INT>(M), static_cast<MKL_INT>(N), static_cast<MKL_INT>(K),
            alpha_f,
            a_cont.data<float>(), static_cast<MKL_INT>(K),
            b_cont.data<float>(), static_cast<MKL_INT>(N),
            beta_f,
            out_data, static_cast<MKL_INT>(N)
        );
#else
        gemm::gemm_optimized(a_cont.data<float>(), b_cont.data<float>(),
                             out_data, M, N, K, alpha_f, beta_f);
#endif
    } else if (mat1.dtype() == DType::Float64) {
        double* out_data = output.data<double>();

        if (beta != 0.0) {
            const Tensor inp_cont = input.is_contiguous() ? input : input.contiguous();
            const double* inp_data = inp_cont.data<double>();
            auto inp_shape = inp_cont.shape();

            if (inp_cont.ndim() == 2 && inp_shape[0] == M && inp_shape[1] == N) {
                std::memcpy(out_data, inp_data, M * N * sizeof(double));
            } else if (inp_cont.ndim() == 1 && inp_shape[0] == N) {
                for (int64_t i = 0; i < M; ++i) {
                    std::memcpy(out_data + i * N, inp_data, N * sizeof(double));
                }
            } else if (inp_cont.numel() == 1) {
                double val = inp_data[0];
                std::fill_n(out_data, M * N, val);
            } else {
                auto expanded = input.expand({M, N}).contiguous();
                std::memcpy(out_data, expanded.data<double>(), M * N * sizeof(double));
            }
        } else {
            std::memset(out_data, 0, M * N * sizeof(double));
        }

#ifdef TENZOR_USE_MKL
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<MKL_INT>(M), static_cast<MKL_INT>(N), static_cast<MKL_INT>(K),
            alpha,
            a_cont.data<double>(), static_cast<MKL_INT>(K),
            b_cont.data<double>(), static_cast<MKL_INT>(N),
            beta,
            out_data, static_cast<MKL_INT>(N)
        );
#else
        // Fallback: manual GEMM for Float64
        // C = beta * C + alpha * A @ B
        // beta scaling already handled if we're here; gemm_optimized only supports float
        if (beta != 1.0 && beta != 0.0) {
            for (int64_t i = 0; i < M * N; ++i) out_data[i] *= beta;
        }
        // alpha * A @ B + (already scaled) C
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t k = 0; k < K; ++k) {
                double a_val = alpha * a_cont.data<double>()[i * K + k];
                for (int64_t j = 0; j < N; ++j) {
                    out_data[i * N + j] += a_val * b_cont.data<double>()[k * N + j];
                }
            }
        }
#endif
    } else if (mat1.dtype() == DType::Float16 || mat1.dtype() == DType::BFloat16) {
        // Upcast to Float32, compute, downcast
        DType orig = mat1.dtype();
        auto result = addmm_kernel(input.to(DType::Float32),
                                   mat1.to(DType::Float32),
                                   mat2.to(DType::Float32),
                                   alpha, beta);
        return result.to(orig);
    } else {
        throw std::runtime_error(
            "addmm: unsupported dtype: " + std::string(dtype_name(mat1.dtype())));
    }

    return output;
}

auto addmv_kernel(const Tensor& input, const Tensor& mat, const Tensor& vec,
                  double alpha, double beta) -> Tensor {
    // input: (M,) or broadcastable, mat: (M, K), vec: (K,)
    int64_t M = mat.shape()[0];
    int64_t K = mat.shape()[1];

    Tensor m_cont = mat.is_contiguous() ? mat : mat.contiguous();
    Tensor v_cont = vec.is_contiguous() ? vec : vec.contiguous();

    Tensor output = Tensor::empty_uninitialized({M}, mat.dtype(), Device::cpu());

    if (mat.dtype() == DType::Float32) {
        float* out_data = output.data<float>();

        if (beta != 0.0) {
            const Tensor inp_cont = input.is_contiguous() ? input : input.contiguous();
            const float* inp_data = inp_cont.data<float>();
            if (inp_cont.ndim() == 1 && inp_cont.shape()[0] == M) {
                std::memcpy(out_data, inp_data, M * sizeof(float));
            } else if (inp_cont.numel() == 1) {
                std::fill_n(out_data, M, inp_data[0]);
            } else {
                auto expanded = input.expand({M}).contiguous();
                std::memcpy(out_data, expanded.data<float>(), M * sizeof(float));
            }
        } else {
            std::memset(out_data, 0, M * sizeof(float));
        }

        float alpha_f = static_cast<float>(alpha);
        float beta_f = static_cast<float>(beta);

#ifdef TENZOR_USE_MKL
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans,
            static_cast<MKL_INT>(M), static_cast<MKL_INT>(K),
            alpha_f,
            m_cont.data<float>(), static_cast<MKL_INT>(K),
            v_cont.data<float>(), 1,
            beta_f,
            out_data, 1
        );
#else
        // Fallback: manual GEMV
        if (beta_f != 1.0f && beta_f != 0.0f) {
            for (int64_t i = 0; i < M; ++i) out_data[i] *= beta_f;
        }
        const float* m_data = m_cont.data<float>();
        const float* v_data = v_cont.data<float>();
        for (int64_t i = 0; i < M; ++i) {
            float sum = 0.0f;
            for (int64_t j = 0; j < K; ++j) {
                sum += m_data[i * K + j] * v_data[j];
            }
            out_data[i] += alpha_f * sum;
        }
#endif
    } else if (mat.dtype() == DType::Float64) {
        double* out_data = output.data<double>();

        if (beta != 0.0) {
            const Tensor inp_cont = input.is_contiguous() ? input : input.contiguous();
            const double* inp_data = inp_cont.data<double>();
            if (inp_cont.ndim() == 1 && inp_cont.shape()[0] == M) {
                std::memcpy(out_data, inp_data, M * sizeof(double));
            } else if (inp_cont.numel() == 1) {
                std::fill_n(out_data, M, inp_data[0]);
            } else {
                auto expanded = input.expand({M}).contiguous();
                std::memcpy(out_data, expanded.data<double>(), M * sizeof(double));
            }
        } else {
            std::memset(out_data, 0, M * sizeof(double));
        }

#ifdef TENZOR_USE_MKL
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans,
            static_cast<MKL_INT>(M), static_cast<MKL_INT>(K),
            alpha,
            m_cont.data<double>(), static_cast<MKL_INT>(K),
            v_cont.data<double>(), 1,
            beta,
            out_data, 1
        );
#else
        if (beta != 1.0 && beta != 0.0) {
            for (int64_t i = 0; i < M; ++i) out_data[i] *= beta;
        }
        const double* m_data = m_cont.data<double>();
        const double* v_data = v_cont.data<double>();
        for (int64_t i = 0; i < M; ++i) {
            double sum = 0.0;
            for (int64_t j = 0; j < K; ++j) {
                sum += m_data[i * K + j] * v_data[j];
            }
            out_data[i] += alpha * sum;
        }
#endif
    } else if (mat.dtype() == DType::Float16 || mat.dtype() == DType::BFloat16) {
        DType orig = mat.dtype();
        auto result = addmv_kernel(input.to(DType::Float32),
                                   mat.to(DType::Float32),
                                   vec.to(DType::Float32),
                                   alpha, beta);
        return result.to(orig);
    } else {
        throw std::runtime_error(
            "addmv: unsupported dtype: " + std::string(dtype_name(mat.dtype())));
    }

    return output;
}

auto baddbmm_kernel(const Tensor& input, const Tensor& batch1, const Tensor& batch2,
                    double alpha, double beta) -> Tensor {
    // input: (B, M, N) or broadcastable, batch1: (B, M, K), batch2: (B, K, N)
    int64_t B = batch1.shape()[0];
    int64_t M = batch1.shape()[1];
    int64_t K = batch1.shape()[2];
    int64_t N = batch2.shape()[2];

    Tensor b1_cont = batch1.is_contiguous() ? batch1 : batch1.contiguous();
    Tensor b2_cont = batch2.is_contiguous() ? batch2 : batch2.contiguous();

    Tensor output = Tensor::empty_uninitialized({B, M, N}, batch1.dtype(), Device::cpu());

    int64_t a_stride = M * K;
    int64_t b_stride = K * N;
    int64_t c_stride = M * N;

    if (batch1.dtype() == DType::Float32) {
        float* out_data = output.data<float>();

        // Initialize output with input data (broadcast if needed)
        if (beta != 0.0) {
            const Tensor inp_cont = input.is_contiguous() ? input : input.contiguous();
            const float* inp_data = inp_cont.data<float>();
            auto inp_shape = inp_cont.shape();

            if (inp_cont.ndim() == 3 && inp_shape[0] == B && inp_shape[1] == M && inp_shape[2] == N) {
                std::memcpy(out_data, inp_data, B * M * N * sizeof(float));
            } else if (inp_cont.ndim() == 2 && inp_shape[0] == M && inp_shape[1] == N) {
                // Broadcast (M, N) -> (B, M, N)
                for (int64_t b = 0; b < B; ++b) {
                    std::memcpy(out_data + b * c_stride, inp_data, M * N * sizeof(float));
                }
            } else if (inp_cont.numel() == 1) {
                std::fill_n(out_data, B * M * N, inp_data[0]);
            } else {
                auto expanded = input.expand({B, M, N}).contiguous();
                std::memcpy(out_data, expanded.data<float>(), B * M * N * sizeof(float));
            }
        } else {
            std::memset(out_data, 0, B * M * N * sizeof(float));
        }

        float alpha_f = static_cast<float>(alpha);
        float beta_f = static_cast<float>(beta);

#ifdef TENZOR_USE_MKL
        cblas_sgemm_batch_strided(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<MKL_INT>(M), static_cast<MKL_INT>(N), static_cast<MKL_INT>(K),
            alpha_f,
            b1_cont.data<float>(), static_cast<MKL_INT>(K), a_stride,
            b2_cont.data<float>(), static_cast<MKL_INT>(N), b_stride,
            beta_f,
            out_data, static_cast<MKL_INT>(N), c_stride,
            static_cast<MKL_INT>(B)
        );
#else
        #pragma omp parallel for if(B > 1)
        for (int64_t batch = 0; batch < B; ++batch) {
            gemm::gemm_optimized(
                b1_cont.data<float>() + batch * a_stride,
                b2_cont.data<float>() + batch * b_stride,
                out_data + batch * c_stride,
                M, N, K, alpha_f, beta_f);
        }
#endif
    } else if (batch1.dtype() == DType::Float64) {
        double* out_data = output.data<double>();

        if (beta != 0.0) {
            const Tensor inp_cont = input.is_contiguous() ? input : input.contiguous();
            const double* inp_data = inp_cont.data<double>();
            auto inp_shape = inp_cont.shape();

            if (inp_cont.ndim() == 3 && inp_shape[0] == B && inp_shape[1] == M && inp_shape[2] == N) {
                std::memcpy(out_data, inp_data, B * M * N * sizeof(double));
            } else if (inp_cont.ndim() == 2 && inp_shape[0] == M && inp_shape[1] == N) {
                for (int64_t b = 0; b < B; ++b) {
                    std::memcpy(out_data + b * c_stride, inp_data, M * N * sizeof(double));
                }
            } else if (inp_cont.numel() == 1) {
                std::fill_n(out_data, B * M * N, inp_data[0]);
            } else {
                auto expanded = input.expand({B, M, N}).contiguous();
                std::memcpy(out_data, expanded.data<double>(), B * M * N * sizeof(double));
            }
        } else {
            std::memset(out_data, 0, B * M * N * sizeof(double));
        }

#ifdef TENZOR_USE_MKL
        cblas_dgemm_batch_strided(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<MKL_INT>(M), static_cast<MKL_INT>(N), static_cast<MKL_INT>(K),
            alpha,
            b1_cont.data<double>(), static_cast<MKL_INT>(K), a_stride,
            b2_cont.data<double>(), static_cast<MKL_INT>(N), b_stride,
            beta,
            out_data, static_cast<MKL_INT>(N), c_stride,
            static_cast<MKL_INT>(B)
        );
#else
        #pragma omp parallel for if(B > 1)
        for (int64_t batch = 0; batch < B; ++batch) {
            // Manual DGEMM for each batch element
            double* c_batch = out_data + batch * c_stride;
            const double* a_batch = b1_cont.data<double>() + batch * a_stride;
            const double* b_batch = b2_cont.data<double>() + batch * b_stride;

            if (beta != 1.0 && beta != 0.0) {
                for (int64_t i = 0; i < M * N; ++i) c_batch[i] *= beta;
            }
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t k = 0; k < K; ++k) {
                    double a_val = alpha * a_batch[i * K + k];
                    for (int64_t j = 0; j < N; ++j) {
                        c_batch[i * N + j] += a_val * b_batch[k * N + j];
                    }
                }
            }
        }
#endif
    } else if (batch1.dtype() == DType::Float16 || batch1.dtype() == DType::BFloat16) {
        DType orig = batch1.dtype();
        auto result = baddbmm_kernel(input.to(DType::Float32),
                                     batch1.to(DType::Float32),
                                     batch2.to(DType::Float32),
                                     alpha, beta);
        return result.to(orig);
    } else {
        throw std::runtime_error(
            "baddbmm: unsupported dtype: " + std::string(dtype_name(batch1.dtype())));
    }

    return output;
}

// =========================================================================
// New element-wise math operations
// =========================================================================

// --- Unary ops ---

auto rsqrt_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsInvSqrt, vdInvSqrt,
        [](float x) { return 1.0f / std::sqrt(x); },
        [](double x) { return 1.0 / std::sqrt(x); }, "rsqrt");
#else
    return unary_math_kernel(input,
        [](float x) { return 1.0f / std::sqrt(x); },
        [](double x) { return 1.0 / std::sqrt(x); }, "rsqrt");
#endif
}

auto square_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) { return x * x; },
        [](double x) { return x * x; }, "square");
}

auto asinh_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsAsinh, vdAsinh,
        [](float x) { return std::asinh(x); },
        [](double x) { return std::asinh(x); }, "asinh");
#else
    return unary_math_kernel(input,
        [](float x) { return std::asinh(x); },
        [](double x) { return std::asinh(x); }, "asinh");
#endif
}

auto acosh_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsAcosh, vdAcosh,
        [](float x) { return std::acosh(x); },
        [](double x) { return std::acosh(x); }, "acosh");
#else
    return unary_math_kernel(input,
        [](float x) { return std::acosh(x); },
        [](double x) { return std::acosh(x); }, "acosh");
#endif
}

auto atanh_kernel(const Tensor& input) -> Tensor {
#if defined(TENZOR_USE_MKL)
    return unary_vml_kernel(input, vsAtanh, vdAtanh,
        [](float x) { return std::atanh(x); },
        [](double x) { return std::atanh(x); }, "atanh");
#else
    return unary_math_kernel(input,
        [](float x) { return std::atanh(x); },
        [](double x) { return std::atanh(x); }, "atanh");
#endif
}

// --- Binary floating-point ops ---

auto hypot_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_math_kernel(a, b,
        [](float x, float y) { return std::hypot(x, y); },
        [](double x, double y) { return std::hypot(x, y); }, "hypot");
}

auto copysign_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_math_kernel(a, b,
        [](float x, float y) { return std::copysign(x, y); },
        [](double x, double y) { return std::copysign(x, y); }, "copysign");
}

auto nextafter_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_math_kernel(a, b,
        [](float x, float y) { return std::nextafter(x, y); },
        [](double x, double y) { return std::nextafter(x, y); }, "nextafter");
}

// --- Binary integer ops ---

auto gcd_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);
    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "gcd", [&]() {
            const scalar_t* ad = a.data<scalar_t>();
            const scalar_t* bd = b.data<scalar_t>();
            scalar_t* od = result.data<scalar_t>();
            for (size_t i = 0; i < n; ++i) {
                od[i] = static_cast<scalar_t>(std::gcd(
                    static_cast<int64_t>(ad[i]), static_cast<int64_t>(bd[i])));
            }
        });
    } else {
        auto a_strides = detail::compute_broadcast_strides(shape_a, output_shape);
        auto b_strides = detail::compute_broadcast_strides(shape_b, output_shape);
        int64_t ndim = static_cast<int64_t>(output_shape.size());
        int64_t n = result.numel();

        TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "gcd", [&]() {
            const scalar_t* ad = a.data<scalar_t>();
            const scalar_t* bd = b.data<scalar_t>();
            scalar_t* od = result.data<scalar_t>();
            _Pragma("omp parallel for if(n > static_cast<int64_t>(OMP_THRESHOLD_SIMPLE))")
            for (int64_t i = 0; i < n; ++i) {
                int64_t a_idx = 0, b_idx = 0, idx = i;
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    int64_t coord = idx % output_shape[d];
                    idx /= output_shape[d];
                    a_idx += coord * a_strides[d];
                    b_idx += coord * b_strides[d];
                }
                od[i] = static_cast<scalar_t>(std::gcd(
                    static_cast<int64_t>(ad[a_idx]), static_cast<int64_t>(bd[b_idx])));
            }
        });
    }
    return result;
}

auto lcm_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);
    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "lcm", [&]() {
            const scalar_t* ad = a.data<scalar_t>();
            const scalar_t* bd = b.data<scalar_t>();
            scalar_t* od = result.data<scalar_t>();
            for (size_t i = 0; i < n; ++i) {
                od[i] = static_cast<scalar_t>(std::lcm(
                    static_cast<int64_t>(ad[i]), static_cast<int64_t>(bd[i])));
            }
        });
    } else {
        auto a_strides = detail::compute_broadcast_strides(shape_a, output_shape);
        auto b_strides = detail::compute_broadcast_strides(shape_b, output_shape);
        int64_t ndim = static_cast<int64_t>(output_shape.size());
        int64_t n = result.numel();

        TENZOR_DISPATCH_INTEGER_TYPES(a.dtype(), "lcm", [&]() {
            const scalar_t* ad = a.data<scalar_t>();
            const scalar_t* bd = b.data<scalar_t>();
            scalar_t* od = result.data<scalar_t>();
            _Pragma("omp parallel for if(n > static_cast<int64_t>(OMP_THRESHOLD_SIMPLE))")
            for (int64_t i = 0; i < n; ++i) {
                int64_t a_idx = 0, b_idx = 0, idx = i;
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    int64_t coord = idx % output_shape[d];
                    idx /= output_shape[d];
                    a_idx += coord * a_strides[d];
                    b_idx += coord * b_strides[d];
                }
                od[i] = static_cast<scalar_t>(std::lcm(
                    static_cast<int64_t>(ad[a_idx]), static_cast<int64_t>(bd[b_idx])));
            }
        });
    }
    return result;
}

// --- Igamma / Igammac (regularized incomplete gamma functions) ---

namespace {

// Lower regularized incomplete gamma function via series expansion:
// P(a, x) = (e^{-x} * x^a / Gamma(a)) * sum_{k=0}^{inf} x^k / (a*(a+1)*...*(a+k))
template<typename T>
T igamma_series(T a, T x) {
    if (x < T(0)) return T(0);
    if (x == T(0)) return T(0);

    // Use series expansion: P(a,x) = e^{-x} x^a / Gamma(a) * Sum_{n=0}^inf x^n / (a*(a+1)*...*(a+n))
    T lgamma_a = std::lgamma(a);
    T prefix = std::exp(-x + a * std::log(x) - lgamma_a);

    T sum = T(0);
    T term = T(1) / a;
    sum += term;

    constexpr int max_iter = 200;
    constexpr T eps = std::is_same_v<T, float> ? T(1e-7f) : T(1e-15);

    for (int n = 1; n < max_iter; ++n) {
        term *= x / (a + T(n));
        sum += term;
        if (std::abs(term) < eps * std::abs(sum)) break;
    }
    return prefix * sum;
}

// Upper regularized incomplete gamma via continued fraction (Legendre):
// Q(a, x) = e^{-x} x^a / Gamma(a) * CF
// Used when x >= a + 1 for better convergence.
template<typename T>
T igammac_cf(T a, T x) {
    T lgamma_a = std::lgamma(a);
    T prefix = std::exp(-x + a * std::log(x) - lgamma_a);

    // Modified Lentz's method for continued fraction
    constexpr T tiny = std::is_same_v<T, float> ? T(1e-30f) : T(1e-300);
    constexpr T eps = std::is_same_v<T, float> ? T(1e-7f) : T(1e-15);
    constexpr int max_iter = 200;

    T f = tiny;
    T C = tiny;
    T D = T(0);

    // Use the standard Gauss continued fraction for Q(a, x):
    //   1/(x+1-a + 1*(1-a)/(x+3-a + 2*(2-a)/(x+5-a + ...)))
    // (audit item F.2 — removed an earlier dead-aborted experimental
    // loop that was left in source.)
    f = x + T(1) - a;
    if (std::abs(f) < tiny) f = tiny;
    C = f;
    D = T(0);

    for (int n = 1; n <= max_iter; ++n) {
        T an_val = -T(n) * (T(n) - a);
        T bn_val = x + T(2 * n + 1) - a;
        D = bn_val + an_val * D;
        if (std::abs(D) < tiny) D = tiny;
        C = bn_val + an_val / C;
        if (std::abs(C) < tiny) C = tiny;
        D = T(1) / D;
        T delta = C * D;
        f *= delta;
        if (std::abs(delta - T(1)) < eps) break;
    }

    return prefix / f;
}

} // anonymous namespace

auto igamma_kernel(const Tensor& a, const Tensor& x) -> Tensor {
    return binary_math_kernel(a, x,
        [](float a_val, float x_val) -> float {
            if (x_val < a_val + 1.0f) {
                return igamma_series(a_val, x_val);
            } else {
                return 1.0f - igammac_cf(a_val, x_val);
            }
        },
        [](double a_val, double x_val) -> double {
            if (x_val < a_val + 1.0) {
                return igamma_series(a_val, x_val);
            } else {
                return 1.0 - igammac_cf(a_val, x_val);
            }
        }, "igamma");
}

auto igammac_kernel(const Tensor& a, const Tensor& x) -> Tensor {
    return binary_math_kernel(a, x,
        [](float a_val, float x_val) -> float {
            if (x_val < a_val + 1.0f) {
                return 1.0f - igamma_series(a_val, x_val);
            } else {
                return igammac_cf(a_val, x_val);
            }
        },
        [](double a_val, double x_val) -> double {
            if (x_val < a_val + 1.0) {
                return 1.0 - igamma_series(a_val, x_val);
            } else {
                return igammac_cf(a_val, x_val);
            }
        }, "igammac");
}

// --- Ternary ops: addcmul / addcdiv ---

auto addcmul_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2,
                    double alpha) -> Tensor {
    // Upcast Float16/BFloat16 to Float32 for numerical stability
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto orig_dtype = input.dtype();
        auto result_f32 = addcmul_kernel(input.to(DType::Float32),
                                         tensor1.to(DType::Float32),
                                         tensor2.to(DType::Float32), alpha);
        return result_f32.to(orig_dtype);
    }

    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(result.numel());

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "addcmul", [&]() {
        const scalar_t* in = input.data<scalar_t>();
        const scalar_t* t1 = tensor1.data<scalar_t>();
        const scalar_t* t2 = tensor2.data<scalar_t>();
        scalar_t* out = result.data<scalar_t>();
        scalar_t alpha_val = static_cast<scalar_t>(alpha);
        _Pragma("omp parallel for if(n > OMP_THRESHOLD_SIMPLE)")
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i] + alpha_val * t1[i] * t2[i];
        }
    });
    return result;
}

auto addcdiv_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2,
                    double alpha) -> Tensor {
    // Upcast Float16/BFloat16 to Float32 for numerical stability
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto orig_dtype = input.dtype();
        auto result_f32 = addcdiv_kernel(input.to(DType::Float32),
                                         tensor1.to(DType::Float32),
                                         tensor2.to(DType::Float32), alpha);
        return result_f32.to(orig_dtype);
    }

    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(result.numel());

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "addcdiv", [&]() {
        const scalar_t* in = input.data<scalar_t>();
        const scalar_t* t1 = tensor1.data<scalar_t>();
        const scalar_t* t2 = tensor2.data<scalar_t>();
        scalar_t* out = result.data<scalar_t>();
        scalar_t alpha_val = static_cast<scalar_t>(alpha);
        _Pragma("omp parallel for if(n > OMP_THRESHOLD_SIMPLE)")
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i] + alpha_val * t1[i] / t2[i];
        }
    });
    return result;
}

// =========================================================================
// Phase 5 Extended Math Operations
// =========================================================================

auto deg2rad_kernel(const Tensor& input) -> Tensor {
    constexpr float pi_f = static_cast<float>(M_PI);
    return unary_math_kernel(input,
        [pi_f](float x) { return x * (pi_f / 180.0f); },
        [](double x) { return x * (M_PI / 180.0); }, "deg2rad");
}

auto rad2deg_kernel(const Tensor& input) -> Tensor {
    constexpr float pi_f = static_cast<float>(M_PI);
    return unary_math_kernel(input,
        [pi_f](float x) { return x * (180.0f / pi_f); },
        [](double x) { return x * (180.0 / M_PI); }, "rad2deg");
}

auto logit_kernel(const Tensor& input) -> Tensor {
    // logit(x) = log(x / (1-x)). This raw kernel takes no eps and therefore
    // must match the canonical eps=None contract of tenzor::logit: do NOT
    // silently clamp -- let inputs outside (0,1) propagate NaN/Inf so callers
    // see the invalid region. Eps-clamping is applied by the composed op
    // (src/ops/math.cpp) before it ever reaches a kernel.
    return unary_math_kernel(input,
        [](float x) {
            return std::log(x / (1.0f - x));
        },
        [](double x) {
            return std::log(x / (1.0 - x));
        }, "logit");
}

auto signbit_kernel(const Tensor& input) -> Tensor {
    return unary_bool_kernel(input,
        [](float x) { return std::signbit(x); },
        [](double x) { return std::signbit(x); }, "signbit");
}

auto isposinf_kernel(const Tensor& input) -> Tensor {
    return unary_bool_kernel(input,
        [](float x) { return std::isinf(x) && x > 0; },
        [](double x) { return std::isinf(x) && x > 0; }, "isposinf");
}

auto isneginf_kernel(const Tensor& input) -> Tensor {
    return unary_bool_kernel(input,
        [](float x) { return std::isinf(x) && x < 0; },
        [](double x) { return std::isinf(x) && x < 0; }, "isneginf");
}

auto isreal_kernel(const Tensor& input) -> Tensor {
    bool is_real = (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128);
    Tensor result({}, DType::Bool, input.device());
    result.data<bool>()[0] = is_real;
    return result;
}

auto float_power_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    auto a_f64 = a.to(DType::Float64);
    auto b_f64 = b.to(DType::Float64);
    return binary_math_kernel(a_f64, b_f64,
        [](float x, float y) { return std::pow(x, y); },
        [](double x, double y) { return std::pow(x, y); }, "float_power");
}

auto xlog1py_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_math_kernel(a, b,
        [](float x, float y) { return x == 0.0f ? 0.0f : x * std::log1p(y); },
        [](double x, double y) { return x == 0.0 ? 0.0 : x * std::log1p(y); }, "xlog1py");
}

auto ldexp_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_math_kernel(a, b,
        [](float x, float n) { return std::ldexp(x, static_cast<int>(n)); },
        [](double x, double n) { return std::ldexp(x, static_cast<int>(n)); }, "ldexp");
}

auto frexp_kernel(const Tensor& input) -> std::vector<Tensor> {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor mantissa(shape_vec, input.dtype(), input.device());
    Tensor exponent(shape_vec, DType::Int32, input.device());
    size_t n = static_cast<size_t>(input.numel());
    int32_t* exp_data = exponent.data<int32_t>();

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* m_data = mantissa.data<float>();
        for (size_t i = 0; i < n; ++i) {
            int exp_val;
            m_data[i] = std::frexp(in_data[i], &exp_val);
            exp_data[i] = exp_val;
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* m_data = mantissa.data<double>();
        for (size_t i = 0; i < n; ++i) {
            int exp_val;
            m_data[i] = std::frexp(in_data[i], &exp_val);
            exp_data[i] = exp_val;
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* m_data = mantissa.data<Float16>();
        for (size_t i = 0; i < n; ++i) {
            int exp_val;
            float mf = std::frexp(static_cast<float>(in_data[i]), &exp_val);
            m_data[i] = Float16(mf);
            exp_data[i] = exp_val;
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* m_data = mantissa.data<BFloat16>();
        for (size_t i = 0; i < n; ++i) {
            int exp_val;
            float mf = std::frexp(static_cast<float>(in_data[i]), &exp_val);
            m_data[i] = BFloat16(mf);
            exp_data[i] = exp_val;
        }
    } else {
        throw std::runtime_error("frexp: unsupported dtype");
    }
    return {mantissa, exponent};
}

auto diag_embed_kernel(const Tensor& input, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor {
    auto in_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t diag_size = in_shape.back();
    int64_t ndim = static_cast<int64_t>(in_shape.size()) + 1;

    if (dim1 < 0) dim1 += ndim;
    if (dim2 < 0) dim2 += ndim;

    int64_t mat_size = diag_size + std::abs(offset);

    // Build output shape: batch dims (all input dims except last) + two matrix dims inserted at dim1, dim2
    std::vector<int64_t> out_shape;
    int in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim1 || d == dim2) {
            out_shape.push_back(mat_size);
        } else {
            if (in_idx < static_cast<int>(in_shape.size())) {
                out_shape.push_back(in_shape[in_idx++]);
            }
        }
    }

    Tensor result(out_shape, input.dtype(), input.device());
    size_t total_bytes = static_cast<size_t>(result.numel()) * result.dtype_size();
    std::memset(result.data_ptr(), 0, total_bytes);

    auto elem_size = result.dtype_size();

    // For 1D input: simple diagonal fill
    if (input.ndim() == 1) {
        const auto* src = static_cast<const uint8_t*>(input.data_ptr());
        auto* dst = static_cast<uint8_t*>(result.data_ptr());

        for (int64_t i = 0; i < diag_size; ++i) {
            int64_t row = offset >= 0 ? i : i - offset;
            int64_t col = offset >= 0 ? i + offset : i;
            int64_t dst_off = (row * mat_size + col) * static_cast<int64_t>(elem_size);
            std::memcpy(dst + dst_off, src + i * static_cast<int64_t>(elem_size), elem_size);
        }
    } else {
        // Batched case: iterate over batch dimensions
        int64_t batch_size = 1;
        for (size_t d = 0; d + 1 < in_shape.size(); d++) {
            batch_size *= in_shape[d];
        }
        const auto* src = static_cast<const uint8_t*>(input.data_ptr());
        auto* dst = static_cast<uint8_t*>(result.data_ptr());
        int64_t mat_elems = mat_size * mat_size;

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < diag_size; ++i) {
                int64_t row = offset >= 0 ? i : i - offset;
                int64_t col = offset >= 0 ? i + offset : i;
                int64_t src_off = (b * diag_size + i) * static_cast<int64_t>(elem_size);
                int64_t dst_off = (b * mat_elems + row * mat_size + col) * static_cast<int64_t>(elem_size);
                std::memcpy(dst + dst_off, src + src_off, elem_size);
            }
        }
    }

    return result;
}

auto diagflat_kernel(const Tensor& input, int64_t offset) -> Tensor {
    auto flat = input.contiguous().reshape({-1});
    int64_t n = flat.numel();
    int64_t mat_size = n + std::abs(offset);
    Tensor result({mat_size, mat_size}, input.dtype(), input.device());
    size_t total_bytes = static_cast<size_t>(result.numel()) * result.dtype_size();
    std::memset(result.data_ptr(), 0, total_bytes);

    auto elem_size = result.dtype_size();
    const auto* src = static_cast<const uint8_t*>(flat.data_ptr());
    auto* dst = static_cast<uint8_t*>(result.data_ptr());

    for (int64_t i = 0; i < n; ++i) {
        int64_t row = offset >= 0 ? i : i - offset;
        int64_t col = offset >= 0 ? i + offset : i;
        int64_t dst_off = (row * mat_size + col) * static_cast<int64_t>(elem_size);
        std::memcpy(dst + dst_off, src + i * static_cast<int64_t>(elem_size), elem_size);
    }

    return result;
}

auto nanvar_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        auto result = nanvar_kernel(f32, dim, keepdim, correction);
        return result.to(input.dtype());
    }

    // Full reduction
    if (dim < 0 && dim != -1) {
        // Treat as full reduction when dim is unset (LLONG_MIN) or explicitly negative
    }
    bool full_reduce = (dim == std::numeric_limits<int64_t>::min()) || (dim < 0 && input.ndim() <= 1);
    if (full_reduce) {
        int64_t n = input.numel();
        Tensor result({1}, input.dtype(), input.device());
        TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "nanvar", [&]() {
            const scalar_t* data = input.data<scalar_t>();
            // First pass: compute nanmean
            scalar_t acc = 0;
            int64_t count = 0;
            for (int64_t i = 0; i < n; i++) {
                scalar_t v = data[i];
                if (!std::isnan(v)) { acc += v; count++; }
            }
            scalar_t mean_val = count > 0 ? acc / static_cast<scalar_t>(count) : scalar_t(0);
            // Second pass: compute sum of squared deviations
            scalar_t sum_sq = 0;
            for (int64_t i = 0; i < n; i++) {
                scalar_t v = data[i];
                if (!std::isnan(v)) {
                    scalar_t diff = v - mean_val;
                    sum_sq += diff * diff;
                }
            }
            int64_t denom = count - correction;
            *result.data<scalar_t>() = denom > 0 ? sum_sq / static_cast<scalar_t>(denom) : scalar_t(0);
        });
        return result;
    }

    // Dim reduction
    if (dim < 0) dim += input.ndim();
    auto shape = input.shape();
    int64_t ndim = shape.size();
    int64_t reduce_size = shape[dim];

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(shape[d]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    auto result = Tensor(out_shape, input.dtype(), input.device());
    int64_t out_n = outer * inner;

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "nanvar_dim", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = result.data<scalar_t>();
        _Pragma("omp parallel for if(out_n > 1000)")
        for (int64_t idx = 0; idx < out_n; idx++) {
            int64_t o = idx / inner;
            int64_t i_inner = idx % inner;
            // First pass: nanmean
            scalar_t acc = 0;
            int64_t count = 0;
            for (int64_t r = 0; r < reduce_size; r++) {
                int64_t src_idx = (o * reduce_size + r) * inner + i_inner;
                scalar_t v = in_data[src_idx];
                if (!std::isnan(v)) { acc += v; count++; }
            }
            scalar_t mean_val = count > 0 ? acc / static_cast<scalar_t>(count) : scalar_t(0);
            // Second pass: sum of squared deviations
            scalar_t sum_sq = 0;
            for (int64_t r = 0; r < reduce_size; r++) {
                int64_t src_idx = (o * reduce_size + r) * inner + i_inner;
                scalar_t v = in_data[src_idx];
                if (!std::isnan(v)) {
                    scalar_t diff = v - mean_val;
                    sum_sq += diff * diff;
                }
            }
            int64_t denom = count - correction;
            out_data[idx] = denom > 0 ? sum_sq / static_cast<scalar_t>(denom) : scalar_t(0);
        }
    });
    return result;
}

auto nanstd_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor {
    auto var = nanvar_kernel(input, dim, keepdim, correction);
    return unary_math_kernel(var,
        [](float x) { return std::sqrt(x); },
        [](double x) { return std::sqrt(x); }, "nanstd_sqrt");
}

// ============================================================================
// Trapezoid integration kernel
// ============================================================================

auto trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                      const Tensor* x_ptr) -> Tensor {
    // Half-precision: widen y (and x if provided) to Float32, recurse, narrow back.
    if (tenzor::utils::is_half_precision(y.dtype())) {
        DType orig = y.dtype();
        Tensor y32 = y.to(DType::Float32);
        if (x_ptr) {
            Tensor x32 = x_ptr->to(DType::Float32);
            return trapezoid_kernel(y32, dim, dx, &x32).to(orig);
        }
        return trapezoid_kernel(y32, dim, dx, nullptr).to(orig);
    }

    auto shape = y.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("trapezoid: dim out of range");
    }
    int64_t n = shape[dim];
    if (n < 2) {
        // Result is zero with dim removed
        std::vector<int64_t> out_shape;
        for (int64_t d = 0; d < ndim; d++) {
            if (d != dim) out_shape.push_back(shape[d]);
        }
        if (out_shape.empty()) out_shape.push_back(1);
        return Tensor(out_shape, y.dtype(), y.device());
    }

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != dim) out_shape.push_back(shape[d]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, y.dtype(), y.device());

    TENZOR_DISPATCH_FLOATING_TYPES(y.dtype(), "trapezoid", [&]() {
        const scalar_t* y_data = y.data<scalar_t>();
        scalar_t* out_data = result.data<scalar_t>();
        const scalar_t* x_data = x_ptr ? x_ptr->data<scalar_t>() : nullptr;

        int64_t out_n = outer * inner;
        _Pragma("omp parallel for if(out_n > 1000)")
        for (int64_t idx = 0; idx < out_n; idx++) {
            int64_t o = idx / inner;
            int64_t i_inner = idx % inner;

            scalar_t sum = 0;
            for (int64_t k = 0; k < n - 1; k++) {
                int64_t idx_k   = (o * n + k) * inner + i_inner;
                int64_t idx_k1  = (o * n + k + 1) * inner + i_inner;
                scalar_t y_k  = y_data[idx_k];
                scalar_t y_k1 = y_data[idx_k1];
                scalar_t h;
                if (x_data) {
                    h = x_data[idx_k1] - x_data[idx_k];
                } else {
                    h = static_cast<scalar_t>(dx);
                }
                sum += static_cast<scalar_t>(0.5) * (y_k + y_k1) * h;
            }
            out_data[idx] = sum;
        }
    });
    return result;
}

// ============================================================================
// Cumulative trapezoid integration kernel
// ============================================================================

auto cumulative_trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                                  const Tensor* x_ptr) -> Tensor {
    // Half-precision: widen y (and x if provided) to Float32, recurse, narrow back.
    if (tenzor::utils::is_half_precision(y.dtype())) {
        DType orig = y.dtype();
        Tensor y32 = y.to(DType::Float32);
        if (x_ptr) {
            Tensor x32 = x_ptr->to(DType::Float32);
            return cumulative_trapezoid_kernel(y32, dim, dx, &x32).to(orig);
        }
        return cumulative_trapezoid_kernel(y32, dim, dx, nullptr).to(orig);
    }

    auto shape = y.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("cumulative_trapezoid: dim out of range");
    }
    int64_t n = shape[dim];
    if (n < 2) {
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape[dim] = 0;
        return Tensor(out_shape, y.dtype(), y.device());
    }

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = n - 1;

    Tensor result(out_shape, y.dtype(), y.device());

    TENZOR_DISPATCH_FLOATING_TYPES(y.dtype(), "cumulative_trapezoid", [&]() {
        const scalar_t* y_data = y.data<scalar_t>();
        scalar_t* out_data = result.data<scalar_t>();
        const scalar_t* x_data = x_ptr ? x_ptr->data<scalar_t>() : nullptr;

        int64_t out_n = outer * inner;
        _Pragma("omp parallel for if(out_n > 1000)")
        for (int64_t idx = 0; idx < out_n; idx++) {
            int64_t o = idx / inner;
            int64_t i_inner = idx % inner;

            scalar_t cumsum = 0;
            for (int64_t k = 0; k < n - 1; k++) {
                int64_t idx_k   = (o * n + k) * inner + i_inner;
                int64_t idx_k1  = (o * n + k + 1) * inner + i_inner;
                scalar_t y_k  = y_data[idx_k];
                scalar_t y_k1 = y_data[idx_k1];
                scalar_t h;
                if (x_data) {
                    h = x_data[idx_k1] - x_data[idx_k];
                } else {
                    h = static_cast<scalar_t>(dx);
                }
                cumsum += static_cast<scalar_t>(0.5) * (y_k + y_k1) * h;
                int64_t out_idx = (o * (n - 1) + k) * inner + i_inner;
                out_data[out_idx] = cumsum;
            }
        }
    });
    return result;
}

// ============================================================================
// Numerical gradient kernel (NumPy-style finite differences)
// ============================================================================

auto gradient_kernel(const Tensor& input, int64_t dim, double spacing) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        auto shape = x_wide.shape();
        int64_t ndim = static_cast<int64_t>(shape.size());
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::runtime_error("gradient: dim out of range");
        }
        int64_t n = shape[dim];
        if (n < 2) {
            // For size 1, gradient is zero
            return Tensor(std::vector<int64_t>(shape.begin(), shape.end()), x_wide.dtype(), x_wide.device());
        }

        int64_t outer = 1, inner = 1;
        for (int64_t d = 0; d < dim; d++) outer *= shape[d];
        for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

        Tensor result(std::vector<int64_t>(shape.begin(), shape.end()), x_wide.dtype(), x_wide.device());

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "gradient", [&]() {
            const scalar_t* in_data = x_wide.data<scalar_t>();
            scalar_t* out_data = result.data<scalar_t>();
            scalar_t h = static_cast<scalar_t>(spacing);

            int64_t out_n = outer * inner;
            _Pragma("omp parallel for if(out_n > 1000)")
            for (int64_t idx = 0; idx < out_n; idx++) {
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;

                auto at = [&](int64_t k) -> scalar_t {
                    return in_data[(o * n + k) * inner + i_inner];
                };
                auto out_at = [&](int64_t k) -> scalar_t& {
                    return out_data[(o * n + k) * inner + i_inner];
                };

                // Forward difference at left boundary
                out_at(0) = (at(1) - at(0)) / h;

                // Central differences for interior
                for (int64_t k = 1; k < n - 1; k++) {
                    out_at(k) = (at(k + 1) - at(k - 1)) / (scalar_t(2) * h);
                }

                // Backward difference at right boundary
                out_at(n - 1) = (at(n - 1) - at(n - 2)) / h;
            }
        });
        return result;
    });
}

// ============================================================================
// Pairwise distance kernel
// ============================================================================

auto pairwise_distance_kernel(const Tensor& x1, const Tensor& x2, double p) -> Tensor {
    return tenzor::utils::widen_narrow_compute(x1, x2, [&](const Tensor& a_wide, const Tensor& b_wide) -> Tensor {
        int64_t N = a_wide.shape()[0];
        int64_t D = a_wide.shape()[1];

        Tensor result({N}, a_wide.dtype(), a_wide.device());

        TENZOR_DISPATCH_FLOATING_TYPES(a_wide.dtype(), "pairwise_distance", [&]() {
            const scalar_t* a = a_wide.data<scalar_t>();
            const scalar_t* b = b_wide.data<scalar_t>();
            scalar_t* out = result.data<scalar_t>();

            if (p == 2.0) {
                _Pragma("omp parallel for if(N > 1000)")
                for (int64_t i = 0; i < N; i++) {
                    scalar_t sum_sq = 0;
                    for (int64_t j = 0; j < D; j++) {
                        scalar_t diff = a[i * D + j] - b[i * D + j];
                        sum_sq += diff * diff;
                    }
                    out[i] = std::sqrt(sum_sq);
                }
            } else if (p == 1.0) {
                _Pragma("omp parallel for if(N > 1000)")
                for (int64_t i = 0; i < N; i++) {
                    scalar_t sum_abs = 0;
                    for (int64_t j = 0; j < D; j++) {
                        sum_abs += std::abs(a[i * D + j] - b[i * D + j]);
                    }
                    out[i] = sum_abs;
                }
            } else if (std::isinf(p)) {
                _Pragma("omp parallel for if(N > 1000)")
                for (int64_t i = 0; i < N; i++) {
                    scalar_t max_abs = 0;
                    for (int64_t j = 0; j < D; j++) {
                        max_abs = std::max(max_abs, std::abs(a[i * D + j] - b[i * D + j]));
                    }
                    out[i] = max_abs;
                }
            } else {
                scalar_t p_val = static_cast<scalar_t>(p);
                scalar_t inv_p = scalar_t(1) / p_val;
                _Pragma("omp parallel for if(N > 1000)")
                for (int64_t i = 0; i < N; i++) {
                    scalar_t sum_pow = 0;
                    for (int64_t j = 0; j < D; j++) {
                        sum_pow += std::pow(std::abs(a[i * D + j] - b[i * D + j]), p_val);
                    }
                    out[i] = std::pow(sum_pow, inv_p);
                }
            }
        });
        return result;
    });
}

// ============================================================================
// Pdist kernel (all-pairs pairwise distances)
// ============================================================================

auto pdist_kernel(const Tensor& input, double p) -> Tensor {
    return tenzor::utils::widen_narrow_compute(input, [&](const Tensor& x_wide) -> Tensor {
        int64_t N = x_wide.shape()[0];
        int64_t D = x_wide.shape()[1];
        int64_t num_pairs = N * (N - 1) / 2;

        Tensor result({num_pairs}, x_wide.dtype(), x_wide.device());

        TENZOR_DISPATCH_FLOATING_TYPES(x_wide.dtype(), "pdist", [&]() {
            const scalar_t* data = x_wide.data<scalar_t>();
            scalar_t* out = result.data<scalar_t>();

            if (p == 2.0) {
                _Pragma("omp parallel for if(num_pairs > 1000)")
                for (int64_t idx = 0; idx < num_pairs; idx++) {
                    // Map flat index to (i, j) pair where i < j
                    // Using the formula: idx = i * N - i*(i+1)/2 + j - i - 1
                    // We solve for i iteratively for correctness
                    int64_t i = 0, offset = 0;
                    while (offset + (N - 1 - i) <= idx) {
                        offset += (N - 1 - i);
                        i++;
                    }
                    int64_t j = idx - offset + i + 1;

                    scalar_t sum_sq = 0;
                    for (int64_t d = 0; d < D; d++) {
                        scalar_t diff = data[i * D + d] - data[j * D + d];
                        sum_sq += diff * diff;
                    }
                    out[idx] = std::sqrt(sum_sq);
                }
            } else if (p == 1.0) {
                _Pragma("omp parallel for if(num_pairs > 1000)")
                for (int64_t idx = 0; idx < num_pairs; idx++) {
                    int64_t i = 0, offset = 0;
                    while (offset + (N - 1 - i) <= idx) {
                        offset += (N - 1 - i);
                        i++;
                    }
                    int64_t j = idx - offset + i + 1;

                    scalar_t sum_abs = 0;
                    for (int64_t d = 0; d < D; d++) {
                        sum_abs += std::abs(data[i * D + d] - data[j * D + d]);
                    }
                    out[idx] = sum_abs;
                }
            } else if (std::isinf(p)) {
                _Pragma("omp parallel for if(num_pairs > 1000)")
                for (int64_t idx = 0; idx < num_pairs; idx++) {
                    int64_t i = 0, offset = 0;
                    while (offset + (N - 1 - i) <= idx) {
                        offset += (N - 1 - i);
                        i++;
                    }
                    int64_t j = idx - offset + i + 1;

                    scalar_t max_abs = 0;
                    for (int64_t d = 0; d < D; d++) {
                        max_abs = std::max(max_abs, std::abs(data[i * D + d] - data[j * D + d]));
                    }
                    out[idx] = max_abs;
                }
            } else {
                scalar_t p_val = static_cast<scalar_t>(p);
                scalar_t inv_p = scalar_t(1) / p_val;
                _Pragma("omp parallel for if(num_pairs > 1000)")
                for (int64_t idx = 0; idx < num_pairs; idx++) {
                    int64_t i = 0, offset = 0;
                    while (offset + (N - 1 - i) <= idx) {
                        offset += (N - 1 - i);
                        i++;
                    }
                    int64_t j = idx - offset + i + 1;

                    scalar_t sum_pow = 0;
                    for (int64_t d = 0; d < D; d++) {
                        sum_pow += std::pow(std::abs(data[i * D + d] - data[j * D + d]), p_val);
                    }
                    out[idx] = std::pow(sum_pow, inv_p);
                }
            }
        });
        return result;
    });
}

// ============================================================================
// LogAddExp: log(exp(a) + exp(b)), numerically stable
// ============================================================================
auto logaddexp_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_math_kernel(a, b,
        [](float x, float y) { float m = std::max(x,y); return m + std::log1pf(std::expf(-(std::abs(x-y)))); },
        [](double x, double y) { double m = std::max(x,y); return m + std::log1p(std::exp(-(std::abs(x-y)))); },
        "logaddexp");
}

// ============================================================================
// LogAddExp2: log2(2^a + 2^b), numerically stable
// ============================================================================
auto logaddexp2_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_math_kernel(a, b,
        [](float x, float y) { float m = std::max(x,y); return m + std::log2f(1.0f + std::exp2f(-(std::abs(x-y)))); },
        [](double x, double y) { double m = std::max(x,y); return m + std::log2(1.0 + std::exp2(-(std::abs(x-y)))); },
        "logaddexp2");
}

// ============================================================================
// XLogY: x * log(y), with 0 * log(y) = 0
// ============================================================================
auto xlogy_kernel(const Tensor& x, const Tensor& y) -> Tensor {
    return binary_math_kernel(x, y,
        [](float a, float b) { return a == 0.0f ? 0.0f : a * std::logf(b); },
        [](double a, double b) { return a == 0.0 ? 0.0 : a * std::log(b); },
        "xlogy");
}

// ============================================================================
// I0e: exp(-|x|) * BesselI0(x), exponentially scaled
// ============================================================================
auto i0e_kernel(const Tensor& input) -> Tensor {
    auto i0e_approx = [](double x) -> double {
        double ax = std::abs(x);
        if (ax < 3.75) {
            double t = x / 3.75;
            t = t * t;
            double i0 = 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
                        + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
            return std::exp(-ax) * i0;
        }
        double t = 3.75 / ax;
        return (1.0 / std::sqrt(ax)) * (0.39894228 + t * (0.01328592
               + t * (0.00225319 - t * (0.00157565 - t * (0.00916281
               - t * (0.02057706 - t * (0.02635537 - t * (0.01647633
               - t * 0.00392377))))))));
    };
    return unary_math_kernel(input,
        [&i0e_approx](float x) { return static_cast<float>(i0e_approx(x)); },
        i0e_approx, "i0e");
}

// ============================================================================
// I1e: exp(-|x|) * BesselI1(x), exponentially scaled
// ============================================================================
auto i1e_kernel(const Tensor& input) -> Tensor {
    auto i1e_approx = [](double x) -> double {
        double ax = std::abs(x);
        double result;
        if (ax < 3.75) {
            double t = x / 3.75;
            t = t * t;
            result = ax * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934
                     + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
            result = std::exp(-ax) * result;
        } else {
            double t = 3.75 / ax;
            result = (1.0 / std::sqrt(ax)) * (0.39894228 - t * (0.03988024
                     - t * (0.00362018 + t * (0.00163801 - t * (0.01031555
                     - t * (0.02282967 - t * (0.02895312 - t * (0.01787654
                     - t * 0.00420059))))))));
        }
        return (x < 0.0) ? -result : result;
    };
    return unary_math_kernel(input,
        [&i1e_approx](float x) { return static_cast<float>(i1e_approx(x)); },
        i1e_approx, "i1e");
}

// ============================================================================
// Entr: -x * log(x), with 0*log(0) = 0
// ============================================================================
auto entr_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) -> float {
            if (x > 0.0f) return -x * std::logf(x);
            if (x == 0.0f) return 0.0f;
            return -std::numeric_limits<float>::infinity();
        },
        [](double x) -> double {
            if (x > 0.0) return -x * std::log(x);
            if (x == 0.0) return 0.0;
            return -std::numeric_limits<double>::infinity();
        },
        "entr");
}

// ============================================================================
// SphericalBesselJ0: sin(x)/x, with j0(0) = 1
// ============================================================================
auto spherical_bessel_j0_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) -> float { return x == 0.0f ? 1.0f : std::sinf(x) / x; },
        [](double x) -> double { return x == 0.0 ? 1.0 : std::sin(x) / x; },
        "spherical_bessel_j0");
}

// ============================================================================
// Renorm: normalize slices along dim so p-norm <= maxnorm
// ============================================================================
auto renorm_kernel(const Tensor& input, double p, int64_t dim, double maxnorm) -> Tensor {
    auto result = input.contiguous().clone();
    auto shape = result.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; i++) outer *= shape[i];
    for (int64_t i = dim + 1; i < ndim; i++) inner *= shape[i];

    auto dispatch_typed = [&]<typename T>(T*) {
        T* data = result.data<T>();
        for (int64_t d = 0; d < dim_size; d++) {
            // Compute p-norm of slice
            T norm_val = 0;
            for (int64_t o = 0; o < outer; o++) {
                for (int64_t i = 0; i < inner; i++) {
                    int64_t idx = (o * dim_size + d) * inner + i;
                    norm_val += std::pow(std::abs(static_cast<double>(data[idx])), p);
                }
            }
            norm_val = static_cast<T>(std::pow(static_cast<double>(norm_val), 1.0 / p));

            // Scale if norm exceeds maxnorm
            if (static_cast<double>(norm_val) > maxnorm) {
                T scale = static_cast<T>(maxnorm / static_cast<double>(norm_val));
                for (int64_t o = 0; o < outer; o++) {
                    for (int64_t i = 0; i < inner; i++) {
                        int64_t idx = (o * dim_size + d) * inner + i;
                        data[idx] *= scale;
                    }
                }
            }
        }
    };

    if (result.dtype() == DType::Float32) {
        dispatch_typed(static_cast<float*>(nullptr));
    } else if (result.dtype() == DType::Float64) {
        dispatch_typed(static_cast<double*>(nullptr));
    } else {
        throw std::invalid_argument("renorm: unsupported dtype");
    }
    return result;
}

// ============================================================================
// CosineSimilarity: sum(a*b, dim) / (norm(a, dim) * norm(b, dim) + eps)
// ============================================================================
auto cosine_similarity_kernel(const Tensor& a, const Tensor& b,
                               int64_t dim, double eps) -> Tensor {
    // Float16/BFloat16: upcast to Float32 for sqrt+div chain, cast back.
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        auto orig_dtype = a.dtype();
        return cosine_similarity_kernel(
            a.to(DType::Float32), b.to(DType::Float32), dim, eps)
            .to(orig_dtype);
    }

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto shape = a_c.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;

    // Compute output shape (remove dim)
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) out_shape.push_back(shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int64_t dim_size = shape[dim];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; i++) outer *= shape[i];
    for (int64_t i = dim + 1; i < ndim; i++) inner *= shape[i];

    auto result = Tensor(out_shape, a.dtype(), a.device());

    auto dispatch_typed = [&]<typename T>(T*) {
        const T* a_data = a_c.data<T>();
        const T* b_data = b_c.data<T>();
        T* out = result.data<T>();

        for (int64_t o = 0; o < outer; o++) {
            for (int64_t i = 0; i < inner; i++) {
                T dot_val = 0, norm_a = 0, norm_b = 0;
                for (int64_t d = 0; d < dim_size; d++) {
                    int64_t idx = (o * dim_size + d) * inner + i;
                    T av = a_data[idx], bv = b_data[idx];
                    dot_val += av * bv;
                    norm_a += av * av;
                    norm_b += bv * bv;
                }
                T denom = std::sqrt(norm_a) * std::sqrt(norm_b) + static_cast<T>(eps);
                out[o * inner + i] = dot_val / denom;
            }
        }
    };

    if (a.dtype() == DType::Float32) dispatch_typed(static_cast<float*>(nullptr));
    else if (a.dtype() == DType::Float64) dispatch_typed(static_cast<double*>(nullptr));
    else throw std::invalid_argument("cosine_similarity: unsupported dtype");

    return result;
}

// =========================================================================
// Ndtr: Normal CDF  Φ(x) = 0.5 * erfc(-x * M_SQRT1_2)
// =========================================================================
auto ndtr_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) -> float { return 0.5f * std::erfc(-x * 0.7071067811865476f); },
        [](double x) -> double { return 0.5 * std::erfc(-x * 0.7071067811865476); }, "ndtr");
}

// =========================================================================
// LogNdtr: Log of Normal CDF  log Φ(x)
// For x >= -5: log(ndtr(x))
// For x < -5: asymptotic expansion for numerical stability
// =========================================================================
auto log_ndtr_kernel(const Tensor& input) -> Tensor {
    auto log_ndtr_f32 = [](float x) -> float {
        if (x >= -5.0f) {
            return std::log(0.5f * std::erfc(-x * 0.7071067811865476f));
        }
        // Asymptotic expansion: log(ndtr(x)) ≈ -x²/2 - log(-x) - 0.5*log(2π) + log(1 - 1/x² + 3/x⁴ - ...)
        float x2 = x * x;
        float inv_x2 = 1.0f / x2;
        float series = 1.0f - inv_x2 * (1.0f - inv_x2 * (3.0f - inv_x2 * 15.0f));
        return -0.5f * x2 - std::log(-x) - 0.9189385332046727f + std::log(series);
    };
    auto log_ndtr_f64 = [](double x) -> double {
        if (x >= -5.0) {
            return std::log(0.5 * std::erfc(-x * 0.7071067811865476));
        }
        double x2 = x * x;
        double inv_x2 = 1.0 / x2;
        double series = 1.0 - inv_x2 * (1.0 - inv_x2 * (3.0 - inv_x2 * 15.0));
        return -0.5 * x2 - std::log(-x) - 0.9189385332046727 + std::log(series);
    };
    return unary_math_kernel(input, log_ndtr_f32, log_ndtr_f64, "log_ndtr");
}

// =========================================================================
// Multigammaln: Multivariate log-gamma
// sum_{j=0}^{d-1} lgamma(x - j/2.0) + d*(d-1)/4 * log(π)
// =========================================================================
auto multigammaln_kernel(const Tensor& input, int64_t d) -> Tensor {
    double log_pi_coeff = static_cast<double>(d) * static_cast<double>(d - 1) / 4.0 * std::log(M_PI);
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();
        float coeff_f = static_cast<float>(log_pi_coeff);
        for (size_t i = 0; i < n; ++i) {
            float val = coeff_f;
            for (int64_t j = 0; j < d; ++j)
                val += std::lgamma(in_data[i] - static_cast<float>(j) * 0.5f);
            out_data[i] = val;
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        for (size_t i = 0; i < n; ++i) {
            double val = log_pi_coeff;
            for (int64_t j = 0; j < d; ++j)
                val += std::lgamma(in_data[i] - static_cast<double>(j) * 0.5);
            out_data[i] = val;
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();
        float coeff_f = static_cast<float>(log_pi_coeff);
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float val = coeff_f;
            for (int64_t j = 0; j < d; ++j)
                val += std::lgamma(x - static_cast<float>(j) * 0.5f);
            out_data[i] = Float16(val);
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();
        float coeff_f = static_cast<float>(log_pi_coeff);
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float val = coeff_f;
            for (int64_t j = 0; j < d; ++j)
                val += std::lgamma(x - static_cast<float>(j) * 0.5f);
            out_data[i] = BFloat16(val);
        }
    } else {
        throw std::runtime_error("multigammaln: unsupported dtype");
    }
    return result;
}

} // namespace cpu
} // namespace tenzor
