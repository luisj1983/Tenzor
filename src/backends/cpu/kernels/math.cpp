#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/utils/error.hpp"
#include "broadcast.hpp"
#include "gemm_optimized.hpp"
#include "simd_fast_math.hpp"
#include "float16_simd.hpp"
#include "bfloat16_simd.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <limits>

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

// Intel MKL for optimized BLAS (5-10x faster GEMM)
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#include <mkl_service.h>
#endif

// Intel oneDNN for optimized matrix operations (alternative to MKL)
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#include <list>
#include <unordered_map>
#endif

// Adaptive OpenMP thresholds based on operation complexity
// NOTE: Thresholds tuned to avoid OpenMP overhead for medium-sized tensors.
// For simple ops, the per-element work is tiny, so we need large tensors
// to amortize thread creation/join overhead (~50-100μs per parallel region).
// With AVX-512 processing 16 floats/cycle, we need ~100K+ elements to benefit.
struct OmpThresholds {
    size_t simple;
    size_t medium;
    size_t complex;
    size_t matmul;
};

// Function-local static guarantees thread-safe lazy initialization (C++11+),
// avoiding potential issues with file-scope static init ordering.
static auto get_omp_thresholds() -> const OmpThresholds& {
    static const OmpThresholds t = [] {
        int n = 1;
#ifdef _OPENMP
        n = omp_get_max_threads();
#endif
        return OmpThresholds{
            std::max(size_t(65536), size_t(16384) * n),
            std::max(size_t(32768), size_t(8192) * n),
            std::max(size_t(8192), size_t(2048) * n),
            1024
        };
    }();
    return t;
}

// Convenience macros — delegate to lazy-init function
#define OMP_THRESHOLD_SIMPLE  (get_omp_thresholds().simple)
#define OMP_THRESHOLD_MEDIUM  (get_omp_thresholds().medium)
#define OMP_THRESHOLD_COMPLEX (get_omp_thresholds().complex)
#define OMP_THRESHOLD_MATMUL  (get_omp_thresholds().matmul)

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

// Micro-kernel for small block multiplication (Float32)
// Uses AVX-512 if available for maximum performance
// Now with software prefetching for better cache utilization
#ifdef TENZOR_HAS_AVX512
__attribute__((target("avx512f")))
#endif
static void matmul_microkernel_float32(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {

#ifdef TENZOR_HAS_AVX512
    // AVX-512: Process 16 floats at a time with prefetching
    constexpr int64_t simd_width = 16;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; j += simd_width) {
            int64_t width = std::min(simd_width, N - j);

            if (width == simd_width) {
                __m512 c_vec = _mm512_loadu_ps(&C[i * ldc + j]);

                for (int64_t k = 0; k < K; ++k) {
                    // Prefetch next A and B values
                    if (k + PREFETCH_A < K) {
                        _mm_prefetch(reinterpret_cast<const char*>(&A[i * lda + k + PREFETCH_A]), _MM_HINT_T0);
                    }
                    if (k + PREFETCH_B < K) {
                        _mm_prefetch(reinterpret_cast<const char*>(&B[(k + PREFETCH_B) * ldb + j]), _MM_HINT_T0);
                    }

                    __m512 a_vec = _mm512_set1_ps(A[i * lda + k]);
                    __m512 b_vec = _mm512_loadu_ps(&B[k * ldb + j]);
                    c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                }

                _mm512_storeu_ps(&C[i * ldc + j], c_vec);
            } else {
                for (int64_t jj = j; jj < j + width; ++jj) {
                    float sum = C[i * ldc + jj];
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A[i * lda + k] * B[k * ldb + jj];
                    }
                    C[i * ldc + jj] = sum;
                }
            }
        }
    }
#elif defined(TENZOR_HAS_AVX2)
    // AVX2: Process 8 floats at a time with prefetching
    constexpr int64_t simd_width = 8;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; j += simd_width) {
            int64_t width = std::min(simd_width, N - j);

            if (width == simd_width) {
                __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);

                for (int64_t k = 0; k < K; ++k) {
                    // Prefetch for cache optimization
                    if (k + PREFETCH_A < K) {
                        _mm_prefetch(reinterpret_cast<const char*>(&A[i * lda + k + PREFETCH_A]), _MM_HINT_T0);
                    }
                    if (k + PREFETCH_B < K) {
                        _mm_prefetch(reinterpret_cast<const char*>(&B[(k + PREFETCH_B) * ldb + j]), _MM_HINT_T0);
                    }

                    __m256 a_vec = _mm256_set1_ps(A[i * lda + k]);
                    __m256 b_vec = _mm256_loadu_ps(&B[k * ldb + j]);
                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                }

                _mm256_storeu_ps(&C[i * ldc + j], c_vec);
            } else {
                for (int64_t jj = j; jj < j + width; ++jj) {
                    float sum = C[i * ldc + jj];
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
            float sum = C[i * ldc + j];
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
#endif
}

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
static thread_local dnnl::engine g_matmul_engine(dnnl::engine::kind::cpu, 0);
static thread_local dnnl::stream g_matmul_stream(g_matmul_engine);

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

class MatMulPrimitiveCache {
public:
    std::shared_ptr<MatMulCachedPrimitive> get(const MatMulCacheKey& key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Move to front of LRU list
            lru_list_.remove(key);
            lru_list_.push_front(key);
            return it->second;
        }
        return nullptr;
    }

    void put(const MatMulCacheKey& key, std::shared_ptr<MatMulCachedPrimitive> value) {
        // Evict if cache is full
        if (cache_.size() >= MATMUL_CACHE_SIZE) {
            auto evict_key = lru_list_.back();
            lru_list_.pop_back();
            cache_.erase(evict_key);
        }
        cache_[key] = value;
        lru_list_.push_front(key);
    }

private:
    std::unordered_map<MatMulCacheKey, std::shared_ptr<MatMulCachedPrimitive>, MatMulCacheKeyHash> cache_;
    std::list<MatMulCacheKey> lru_list_;
};

static thread_local MatMulPrimitiveCache g_matmul_cache;

// oneDNN matmul for Float32 with primitive caching
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
        auto& engine = g_matmul_engine;
        auto& stream = g_matmul_stream;

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
    } catch (const dnnl::error&) {
        return false;  // Fall back to MKL/custom GEMM
    }
}
#endif

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
    #pragma omp parallel for collapse(2) if(M * N > 10000)
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
    #pragma omp parallel if(M * N > 65536)
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

    // Thread-local FP32 accumulator for C tile
    static thread_local std::vector<float> C_fp32_buf_bf16(TILE_M * TILE_N);

    // Zero-initialize output
    std::fill_n(C, M * N, BFloat16(0.0f));

    #pragma omp parallel for collapse(2) if(M * N > 65536)
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

    // Use heap-allocated thread-local buffers to avoid stack overflow
    static thread_local std::vector<float> A_fp32_buf_bf16(TILE_M * TILE_K);
    static thread_local std::vector<float> B_fp32_buf_bf16(TILE_K * TILE_N);
    static thread_local std::vector<float> C_fp32_buf_bf16(TILE_M * TILE_N);

    // Zero-initialize output
    std::fill_n(C, M * N, BFloat16(0.0f));

    #pragma omp parallel for collapse(2) if(M * N > 65536)
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
            int64_t tile_m = std::min(TILE_M, M - i0);
            int64_t tile_n = std::min(TILE_N, N - j0);

            // Thread-local buffer pointers
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
        if (b[i] == T(0)) {
            if constexpr (std::is_integral_v<T>) {
                throw std::runtime_error("Integer division by zero");
            } else {
                c[i] = std::numeric_limits<T>::infinity();
            }
        } else {
            c[i] = a[i] / b[i];
        }
    }
}

// SIMD implementations for Float32
#ifdef TENZOR_HAS_AVX512

__attribute__((target("avx512f")))
inline void add_avx512_f32(const float* a, const float* b, float* c, size_t n) {
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
inline void sub_avx512_f32(const float* a, const float* b, float* c, size_t n) {
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
inline void mul_avx512_f32(const float* a, const float* b, float* c, size_t n) {
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
inline void div_avx512_f32(const float* a, const float* b, float* c, size_t n) {
    const size_t simd_width = 16;
    const size_t simd_end = (n / simd_width) * simd_width;

    #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_div_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }

    for (size_t i = simd_end; i < n; ++i) {
        if (b[i] == 0.0f) {
            c[i] = std::numeric_limits<float>::infinity();
        } else {
            c[i] = a[i] / b[i];
        }
    }
}

// SIMD implementations for Float64
__attribute__((target("avx512f")))
inline void add_avx512_f64(const double* a, const double* b, double* c, size_t n) {
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
inline void sub_avx512_f64(const double* a, const double* b, double* c, size_t n) {
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
inline void mul_avx512_f64(const double* a, const double* b, double* c, size_t n) {
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
inline void div_avx512_f64(const double* a, const double* b, double* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d vc = _mm512_div_pd(va, vb);
        _mm512_storeu_pd(c + i, vc);
    }

    for (; i < n; ++i) {
        if (b[i] == 0.0) {
            c[i] = std::numeric_limits<double>::infinity();
        } else {
            c[i] = a[i] / b[i];
        }
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
inline void add_avx2_f32(const float* a, const float* b, float* c, size_t n) {
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
inline void sub_avx2_f32(const float* a, const float* b, float* c, size_t n) {
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
inline void mul_avx2_f32(const float* a, const float* b, float* c, size_t n) {
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
inline void div_avx2_f32(const float* a, const float* b, float* c, size_t n) {
    const size_t simd_width = 8;
    const size_t simd_end = (n / simd_width) * simd_width;

    #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }

    for (size_t i = simd_end; i < n; ++i) {
        if (b[i] == 0.0f) {
            c[i] = std::numeric_limits<float>::infinity();
        } else {
            c[i] = a[i] / b[i];
        }
    }
}

__attribute__((target("avx2,fma")))
inline void add_avx2_f64(const double* a, const double* b, double* c, size_t n) {
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
inline void sub_avx2_f64(const double* a, const double* b, double* c, size_t n) {
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
inline void mul_avx2_f64(const double* a, const double* b, double* c, size_t n) {
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
inline void div_avx2_f64(const double* a, const double* b, double* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 4;

    for (; i + simd_width <= n; i += simd_width) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_div_pd(va, vb);
        _mm256_storeu_pd(c + i, vc);
    }

    for (; i < n; ++i) {
        if (b[i] == 0.0) {
            c[i] = std::numeric_limits<double>::infinity();
        } else {
            c[i] = a[i] / b[i];
        }
    }
}

#endif // TENZOR_HAS_AVX512 / TENZOR_HAS_AVX2

} // namespace detail

// ============================================================================
// CPU math kernels
// ============================================================================

auto add_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    // Compute output shape
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    // Check if we can use the fast path (same shape, no broadcasting)
    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());

        // Dispatch based on dtype and SIMD availability
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();

#ifdef TENZOR_HAS_AVX512
            detail::add_avx512_f32(a_data, b_data, c_data, n);
#elif defined(TENZOR_HAS_AVX2)
            detail::add_avx2_f32(a_data, b_data, c_data, n);
#else
            detail::add_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();

#ifdef TENZOR_HAS_AVX512
            detail::add_avx512_f64(a_data, b_data, c_data, n);
#elif defined(TENZOR_HAS_AVX2)
            detail::add_avx2_f64(a_data, b_data, c_data, n);
#else
            detail::add_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();

#ifdef TENZOR_HAS_AVX512
            detail::add_avx512_i32(a_data, b_data, c_data, n);
#else
            detail::add_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            int64_t* c_data = result.data<int64_t>();

#ifdef TENZOR_HAS_AVX512
            detail::add_avx512_i64(a_data, b_data, c_data, n);
#else
            detail::add_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            Float16* c_data = result.data<Float16>();

            // Use F16C SIMD for Float16 add
            float16_simd::add_f16(
                reinterpret_cast<const uint16_t*>(a_data),
                reinterpret_cast<const uint16_t*>(b_data),
                reinterpret_cast<uint16_t*>(c_data),
                n
            );

        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();
            #pragma omp parallel for schedule(static) if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = BFloat16(static_cast<float>(a_data[i]) + static_cast<float>(b_data[i]));
            }

        } else if (a.dtype() == DType::Int8) {
            const int8_t* a_data = a.data<int8_t>();
            const int8_t* b_data = b.data<int8_t>();
            int8_t* c_data = result.data<int8_t>();
            detail::add_scalar(a_data, b_data, c_data, n);

        } else if (a.dtype() == DType::UInt8) {
            const uint8_t* a_data = a.data<uint8_t>();
            const uint8_t* b_data = b.data<uint8_t>();
            uint8_t* c_data = result.data<uint8_t>();
            detail::add_scalar(a_data, b_data, c_data, n);

        } else if (a.dtype() == DType::Bool) {
            // Bool add uses numeric semantics (int(a) + int(b) -> bool)
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            bool* c_data = result.data<bool>();
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (static_cast<int>(a_data[i]) + static_cast<int>(b_data[i])) != 0;
            }

        } else {
            throw std::runtime_error("Unsupported dtype for add operation");
        }
    } else {
        // Broadcasting path
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) { return x + y; });

        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) { return x + y; });

        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int32_t x, int32_t y) { return x + y; });

        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            int64_t* c_data = result.data<int64_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) { return x + y; });

        } else if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            Float16* c_data = result.data<Float16>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](Float16 x, Float16 y) {
                                    return Float16(static_cast<float>(x) + static_cast<float>(y));
                                });

        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](BFloat16 x, BFloat16 y) {
                                    return BFloat16(static_cast<float>(x) + static_cast<float>(y));
                                });

        } else if (a.dtype() == DType::Int8) {
            const int8_t* a_data = a.data<int8_t>();
            const int8_t* b_data = b.data<int8_t>();
            int8_t* c_data = result.data<int8_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int8_t x, int8_t y) { return x + y; });

        } else if (a.dtype() == DType::UInt8) {
            const uint8_t* a_data = a.data<uint8_t>();
            const uint8_t* b_data = b.data<uint8_t>();
            uint8_t* c_data = result.data<uint8_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](uint8_t x, uint8_t y) { return x + y; });

        } else if (a.dtype() == DType::Bool) {
            // Bool add uses numeric semantics (int(a) + int(b) -> bool)
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            bool* c_data = result.data<bool>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](bool x, bool y) { return (static_cast<int>(x) + static_cast<int>(y)) != 0; });

        } else {
            throw std::runtime_error("Unsupported dtype for add operation");
        }
    }

    return result;
}

auto sub_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    // Compute output shape
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    // Check if we can use the fast path (same shape, no broadcasting)
    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());

        // Dispatch based on dtype and SIMD availability
        if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            Float16* c_data = result.data<Float16>();
            detail::sub_scalar(a_data, b_data, c_data, n);

        } else if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();

#ifdef TENZOR_HAS_AVX512
            detail::sub_avx512_f32(a_data, b_data, c_data, n);
#elif defined(TENZOR_HAS_AVX2)
            detail::sub_avx2_f32(a_data, b_data, c_data, n);
#else
            detail::sub_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();

#ifdef TENZOR_HAS_AVX512
            detail::sub_avx512_f64(a_data, b_data, c_data, n);
#elif defined(TENZOR_HAS_AVX2)
            detail::sub_avx2_f64(a_data, b_data, c_data, n);
#else
            detail::sub_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();

#ifdef TENZOR_HAS_AVX512
            detail::sub_avx512_i32(a_data, b_data, c_data, n);
#else
            detail::sub_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            int64_t* c_data = result.data<int64_t>();

#ifdef TENZOR_HAS_AVX512
            detail::sub_avx512_i64(a_data, b_data, c_data, n);
#else
            detail::sub_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();
            #pragma omp parallel for schedule(static) if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = BFloat16(static_cast<float>(a_data[i]) - static_cast<float>(b_data[i]));
            }

        } else if (a.dtype() == DType::Int8) {
            const int8_t* a_data = a.data<int8_t>();
            const int8_t* b_data = b.data<int8_t>();
            int8_t* c_data = result.data<int8_t>();
            detail::sub_scalar(a_data, b_data, c_data, n);

        } else if (a.dtype() == DType::Bool) {
            // Bool sub uses numeric semantics (int(a) - int(b) -> bool)
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            bool* c_data = result.data<bool>();
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = (static_cast<int>(a_data[i]) - static_cast<int>(b_data[i])) != 0;
            }

        } else {
            throw std::runtime_error("Unsupported dtype for sub operation");
        }
    } else {
        // Broadcasting path
        if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            Float16* c_data = result.data<Float16>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](Float16 x, Float16 y) { return Float16(static_cast<float>(x) - static_cast<float>(y)); });

        } else if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) { return x - y; });

        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) { return x - y; });

        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int32_t x, int32_t y) { return x - y; });

        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            int64_t* c_data = result.data<int64_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) { return x - y; });

        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](BFloat16 x, BFloat16 y) {
                                    return BFloat16(static_cast<float>(x) - static_cast<float>(y));
                                });

        } else if (a.dtype() == DType::Int8) {
            const int8_t* a_data = a.data<int8_t>();
            const int8_t* b_data = b.data<int8_t>();
            int8_t* c_data = result.data<int8_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int8_t x, int8_t y) { return x - y; });

        } else if (a.dtype() == DType::Bool) {
            // Bool sub uses numeric semantics (int(a) - int(b) -> bool)
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            bool* c_data = result.data<bool>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](bool x, bool y) { return (static_cast<int>(x) - static_cast<int>(y)) != 0; });

        } else {
            throw std::runtime_error("Unsupported dtype for sub operation");
        }
    }

    return result;
}

auto mul_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    // Compute output shape
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    // Check if we can use the fast path (same shape, no broadcasting)
    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());

        // Dispatch based on dtype and SIMD availability
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();

#ifdef TENZOR_HAS_AVX512
            detail::mul_avx512_f32(a_data, b_data, c_data, n);
#elif defined(TENZOR_HAS_AVX2)
            detail::mul_avx2_f32(a_data, b_data, c_data, n);
#else
            detail::mul_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();

#ifdef TENZOR_HAS_AVX512
            detail::mul_avx512_f64(a_data, b_data, c_data, n);
#elif defined(TENZOR_HAS_AVX2)
            detail::mul_avx2_f64(a_data, b_data, c_data, n);
#else
            detail::mul_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();

#ifdef TENZOR_HAS_AVX512
            detail::mul_avx512_i32(a_data, b_data, c_data, n);
#else
            detail::mul_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            int64_t* c_data = result.data<int64_t>();

            // No SIMD for int64 multiply
            detail::mul_scalar(a_data, b_data, c_data, n);

        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            bool* c_data = result.data<bool>();

            // Bool multiply is logical AND
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = a_data[i] && b_data[i];
            }

        } else if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            Float16* c_data = result.data<Float16>();

            // Use F16C SIMD for Float16 mul
            float16_simd::mul_f16(
                reinterpret_cast<const uint16_t*>(a_data),
                reinterpret_cast<const uint16_t*>(b_data),
                reinterpret_cast<uint16_t*>(c_data),
                n
            );

        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();
            #pragma omp parallel for schedule(static) if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                c_data[i] = BFloat16(static_cast<float>(a_data[i]) * static_cast<float>(b_data[i]));
            }

        } else {
            throw std::runtime_error("Unsupported dtype for mul operation");
        }
    } else {
        // Broadcasting path
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) { return x * y; });

        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) { return x * y; });

        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int32_t x, int32_t y) { return x * y; });

        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            int64_t* c_data = result.data<int64_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) { return x * y; });

        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            bool* c_data = result.data<bool>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](bool x, bool y) { return x && y; });

        } else if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            Float16* c_data = result.data<Float16>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](Float16 x, Float16 y) {
                                    return Float16(static_cast<float>(x) * static_cast<float>(y));
                                });

        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](BFloat16 x, BFloat16 y) {
                                    return BFloat16(static_cast<float>(x) * static_cast<float>(y));
                                });

        } else {
            throw std::runtime_error("Unsupported dtype for mul operation");
        }
    }

    return result;
}

auto div_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    detail::validate_elementwise(a, b);

    auto shape_a = a.shape();
    auto shape_b = b.shape();
    std::vector<int64_t> shape_a_vec(shape_a.begin(), shape_a.end());
    std::vector<int64_t> shape_b_vec(shape_b.begin(), shape_b.end());

    // Compute output shape
    std::vector<int64_t> output_shape = detail::compute_broadcast_shape(shape_a_vec, shape_b_vec);
    Tensor result(output_shape, a.dtype(), a.device());

    // Check if we can use the fast path (same shape, no broadcasting)
    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());

        // Dispatch based on dtype and SIMD availability
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();

#ifdef TENZOR_HAS_AVX512
            detail::div_avx512_f32(a_data, b_data, c_data, n);
#elif defined(TENZOR_HAS_AVX2)
            detail::div_avx2_f32(a_data, b_data, c_data, n);
#else
            detail::div_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();

#ifdef TENZOR_HAS_AVX512
            detail::div_avx512_f64(a_data, b_data, c_data, n);
#elif defined(TENZOR_HAS_AVX2)
            detail::div_avx2_f64(a_data, b_data, c_data, n);
#else
            detail::div_scalar(a_data, b_data, c_data, n);
#endif

        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();

            // Integer division uses scalar (complex to vectorize)
            detail::div_scalar(a_data, b_data, c_data, n);

        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            int64_t* c_data = result.data<int64_t>();

            detail::div_scalar(a_data, b_data, c_data, n);

        } else if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            Float16* c_data = result.data<Float16>();

            // Float16 division using scalar path (convert to float32)
            for (size_t i = 0; i < n; ++i) {
                float a_f32 = static_cast<float>(a_data[i]);
                float b_f32 = static_cast<float>(b_data[i]);
                float result_f32 = (b_f32 == 0.0f) ? std::numeric_limits<float>::infinity() : a_f32 / b_f32;
                c_data[i] = Float16(result_f32);
            }

        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();
            #pragma omp parallel for schedule(static) if(n > OMP_THRESHOLD_SIMPLE)
            for (size_t i = 0; i < n; ++i) {
                float a_f32 = static_cast<float>(a_data[i]);
                float b_f32 = static_cast<float>(b_data[i]);
                float result_f32 = (b_f32 == 0.0f) ? std::numeric_limits<float>::infinity() : a_f32 / b_f32;
                c_data[i] = BFloat16(result_f32);
            }

        } else {
            throw std::runtime_error("Unsupported dtype for div operation");
        }
    } else {
        // Broadcasting path
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* c_data = result.data<float>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) {
                                    if (y == 0.0f) return std::numeric_limits<float>::infinity();
                                    return x / y;
                                });

        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* c_data = result.data<double>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) {
                                    if (y == 0.0) return std::numeric_limits<double>::infinity();
                                    return x / y;
                                });

        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            int32_t* c_data = result.data<int32_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int32_t x, int32_t y) -> int32_t {
                                    if (y == 0) throw std::runtime_error("Integer division by zero");
                                    return x / y;
                                });

        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            int64_t* c_data = result.data<int64_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) -> int64_t {
                                    if (y == 0) throw std::runtime_error("Integer division by zero");
                                    return x / y;
                                });

        } else if (a.dtype() == DType::Float16) {
            const Float16* a_data = a.data<Float16>();
            const Float16* b_data = b.data<Float16>();
            Float16* c_data = result.data<Float16>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](Float16 x, Float16 y) {
                                    float x_f32 = static_cast<float>(x);
                                    float y_f32 = static_cast<float>(y);
                                    if (y_f32 == 0.0f) return Float16(std::numeric_limits<float>::infinity());
                                    return Float16(x_f32 / y_f32);
                                });

        } else if (a.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a.data<BFloat16>();
            const BFloat16* b_data = b.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](BFloat16 x, BFloat16 y) {
                                    float x_f32 = static_cast<float>(x);
                                    float y_f32 = static_cast<float>(y);
                                    if (y_f32 == 0.0f) return BFloat16(std::numeric_limits<float>::infinity());
                                    return BFloat16(x_f32 / y_f32);
                                });

        } else {
            throw std::runtime_error("Unsupported dtype for div operation");
        }
    }

    return result;
}

auto matmul_kernel(const Tensor& a, const Tensor& b) -> Tensor {
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

        } else if (a_contig.dtype() == DType::Float16 && b_contig.dtype() == DType::Float16) {
            const Float16* a_data = a_contig.data<Float16>();
            const Float16* b_data = b_contig.data<Float16>();
            Float16* c_data = result.data<Float16>();

            matmul_blocked_float16(a_data, b_data, c_data, M, K, N);

        } else if (a_contig.dtype() == DType::BFloat16 && b_contig.dtype() == DType::BFloat16) {
            const BFloat16* a_data = a_contig.data<BFloat16>();
            const BFloat16* b_data = b_contig.data<BFloat16>();
            BFloat16* c_data = result.data<BFloat16>();

            matmul_blocked_bfloat16(a_data, b_data, c_data, M, K, N);

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

    } else if (a_contig.dtype() == DType::Float16 && b_contig.dtype() == DType::Float16) {
        const Float16* a_data = a_contig.data<Float16>();
        const Float16* b_data = b_contig.data<Float16>();
        Float16* c_data = result.data<Float16>();

        matmul_blocked_float16(a_data, b_data, c_data, M, N, K);

    } else if (a_contig.dtype() == DType::BFloat16 && b_contig.dtype() == DType::BFloat16) {
        const BFloat16* a_data = a_contig.data<BFloat16>();
        const BFloat16* b_data = b_contig.data<BFloat16>();
        BFloat16* c_data = result.data<BFloat16>();

        matmul_blocked_bfloat16(a_data, b_data, c_data, M, N, K);

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
        throw std::runtime_error(
            "bmm_kernel unsupported dtype: " +
            std::string(dtype_name(a.dtype()))
        );
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

    } else {
        throw std::runtime_error("sqrt operation only supports Float32, Float64, and Float16 dtypes");
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

    } else {
        throw std::runtime_error("neg operation only supports Float32, Float64, Float16, BFloat16, and Int32 dtypes");
    }

    return result;
}

auto abs_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

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

    } else {
        throw std::runtime_error("abs operation only supports Float32, Float64, Float16, and Int32 dtypes");
    }

    return result;
}

// Clamp kernel - clamps tensor values to [min_val, max_val]
auto clamp_kernel(const Tensor& input, float min_val, float max_val) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

#ifdef TENZOR_HAS_AVX2
        // SIMD: Process 8 floats at a time with OpenMP
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        const __m256 min_vec = _mm256_set1_ps(min_val);
        const __m256 max_vec = _mm256_set1_ps(max_val);

        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 v = _mm256_loadu_ps(&in_data[i]);
            // Clamp: max(min(v, max), min)
            __m256 clamped = _mm256_max_ps(_mm256_min_ps(v, max_vec), min_vec);
            _mm256_storeu_ps(&out_data[i], clamped);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::max(std::min(in_data[i], max_val), min_val);
        }
#else
        // Scalar fallback with OpenMP
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::max(std::min(in_data[i], max_val), min_val);
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

        double min_val_d = static_cast<double>(min_val);
        double max_val_d = static_cast<double>(max_val);

#ifdef TENZOR_HAS_AVX2
        // SIMD: Process 4 doubles at a time
        size_t simd_end = (n / 4) * 4;
        __m256d min_vec = _mm256_set1_pd(min_val_d);
        __m256d max_vec = _mm256_set1_pd(max_val_d);

        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d v = _mm256_loadu_pd(&in_data[i]);
            // Clamp: max(min(v, max), min)
            __m256d clamped = _mm256_max_pd(_mm256_min_pd(v, max_vec), min_vec);
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

        // Convert to float, clamp, then convert back
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            val = std::max(std::min(val, max_val), min_val);
            out_data[i] = Float16(val);
        }

    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();

        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            val = std::max(std::min(val, max_val), min_val);
            out_data[i] = BFloat16(val);
        }

    } else {
        throw std::runtime_error("clamp operation only supports Float16, BFloat16, Float32 and Float64 dtypes");
    }

    return result;
}

// Clamp min kernel - clamps tensor values to min_val (x >= min_val)
auto clamp_min_kernel(const Tensor& input, float min_val) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

#ifdef TENZOR_HAS_AVX2
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        const __m256 min_vec = _mm256_set1_ps(min_val);

        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 v = _mm256_loadu_ps(&in_data[i]);
            __m256 clamped = _mm256_max_ps(v, min_vec);
            _mm256_storeu_ps(&out_data[i], clamped);
        }
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::max(in_data[i], min_val);
        }
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::max(in_data[i], min_val);
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        double min_val_d = static_cast<double>(min_val);

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

        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            val = std::max(val, min_val);
            out_data[i] = Float16(val);
        }

    } else {
        throw std::runtime_error("clamp_min operation only supports Float16, Float32 and Float64 dtypes");
    }

    return result;
}

// Clamp max kernel - clamps tensor values to max_val (x <= max_val)
auto clamp_max_kernel(const Tensor& input, float max_val) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

#ifdef TENZOR_HAS_AVX2
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        const __m256 max_vec = _mm256_set1_ps(max_val);

        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 v = _mm256_loadu_ps(&in_data[i]);
            __m256 clamped = _mm256_min_ps(v, max_vec);
            _mm256_storeu_ps(&out_data[i], clamped);
        }
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::min(in_data[i], max_val);
        }
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_SIMPLE)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::min(in_data[i], max_val);
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        double max_val_d = static_cast<double>(max_val);

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

        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            val = std::min(val, max_val);
            out_data[i] = Float16(val);
        }

    } else {
        throw std::runtime_error("clamp_max operation only supports Float16, Float32 and Float64 dtypes");
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

        // For small arrays, use single-threaded SIMD
        if (n < OMP_THRESHOLD_MEDIUM) {
#ifdef TENZOR_HAS_AVX512
            fast_math::log_batch_avx512(in_data, out_data, n);
#elif defined(TENZOR_HAS_AVX2)
            fast_math::log_batch_avx2(in_data, out_data, n);
#else
            for (size_t i = 0; i < n; ++i) {
                out_data[i] = std::log(in_data[i]);
            }
#endif
        } else {
            // For large arrays, use OpenMP with thread-local SIMD
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();

                size_t chunk_size = (n + nthreads - 1) / nthreads;
                size_t start = tid * chunk_size;
                size_t end = std::min(start + chunk_size, n);

                if (start < end) {
#ifdef TENZOR_HAS_AVX512
                    fast_math::log_batch_avx512(in_data + start, out_data + start, end - start);
#elif defined(TENZOR_HAS_AVX2)
                    fast_math::log_batch_avx2(in_data + start, out_data + start, end - start);
#else
                    for (size_t i = start; i < end; ++i) {
                        out_data[i] = std::log(in_data[i]);
                    }
#endif
                }
            }
        }

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

        #pragma omp parallel for if(n > OMP_THRESHOLD_MEDIUM)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::log(in_data[i]);
        }

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

    } else {
        throw std::runtime_error("log operation only supports Float32, Float64, and Float16 dtypes");
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

        // For small arrays, use single-threaded SIMD
        if (n < OMP_THRESHOLD_MEDIUM) {
#ifdef TENZOR_HAS_AVX512
            fast_math::exp_batch_avx512(in_data, out_data, n);
#elif defined(TENZOR_HAS_AVX2)
            fast_math::exp_batch_avx2(in_data, out_data, n);
#else
            for (size_t i = 0; i < n; ++i) {
                out_data[i] = std::exp(in_data[i]);
            }
#endif
        } else {
            // For large arrays, use OpenMP with thread-local SIMD
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();

                size_t chunk_size = (n + nthreads - 1) / nthreads;
                size_t start = tid * chunk_size;
                size_t end = std::min(start + chunk_size, n);

                if (start < end) {
#ifdef TENZOR_HAS_AVX512
                    fast_math::exp_batch_avx512(in_data + start, out_data + start, end - start);
#elif defined(TENZOR_HAS_AVX2)
                    fast_math::exp_batch_avx2(in_data + start, out_data + start, end - start);
#else
                    for (size_t i = start; i < end; ++i) {
                        out_data[i] = std::exp(in_data[i]);
                    }
#endif
                }
            }
        }

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

        #pragma omp parallel for if(n > OMP_THRESHOLD_MEDIUM)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::exp(in_data[i]);
        }

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

    } else {
        throw std::runtime_error("exp operation only supports Float32, Float64, and Float16 dtypes");
    }

    return result;
}

// Pow kernel - power function (SIMD + OpenMP optimized)
auto pow_kernel(const Tensor& input, float exponent) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

#if defined(TENZOR_HAS_AVX512) || defined(__AVX512F__)
        fast_math::pow_batch_avx512(in_data, out_data, n, exponent);
#elif defined(TENZOR_HAS_AVX2) || defined(__AVX2__)
        fast_math::pow_batch_avx2(in_data, out_data, n, exponent);
#else
        #pragma omp parallel for if(n > OMP_THRESHOLD_MEDIUM)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::pow(in_data[i], exponent);
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        double exp_d = static_cast<double>(exponent);

        #pragma omp parallel for if(n > OMP_THRESHOLD_MEDIUM)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::pow(in_data[i], exp_d);
        }

    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();

        // Use temporary float buffer for SIMD
        std::vector<float> in_f32(n), out_f32(n);
        for (size_t i = 0; i < n; ++i) {
            in_f32[i] = static_cast<float>(in_data[i]);
        }

#if defined(TENZOR_HAS_AVX512) || defined(__AVX512F__)
        fast_math::pow_batch_avx512(in_f32.data(), out_f32.data(), n, exponent);
#elif defined(TENZOR_HAS_AVX2) || defined(__AVX2__)
        fast_math::pow_batch_avx2(in_f32.data(), out_f32.data(), n, exponent);
#else
        for (size_t i = 0; i < n; ++i) {
            out_f32[i] = std::pow(in_f32[i], exponent);
        }
#endif
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = Float16(out_f32[i]);
        }

    } else {
        throw std::runtime_error("pow operation only supports Float32, Float64, and Float16 dtypes");
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

    } else {
        throw std::runtime_error("sign operation only supports Float32, Float64, and Float16 dtypes");
    }

    return result;
}


// ============================================================================
// Comparison Operations
// ============================================================================

// Equal kernel - element-wise equality comparison
auto eq_kernel(const Tensor& a, const Tensor& b) -> Tensor {
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
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] != b_data[i]); }
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] != b_data[i]); }
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] != b_data[i]); }
        } else {
            throw std::runtime_error("Unsupported dtype for ne operation");
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
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            detail::broadcast_op<int32_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int32_t x, int32_t y) { return x != y; });
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            detail::broadcast_op<int64_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) { return x != y; });
        } else if (a.dtype() == DType::Bool) {
            const bool* a_data = a.data<bool>();
            const bool* b_data = b.data<bool>();
            detail::broadcast_op<bool, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](bool x, bool y) { return x != y; });
        } else {
            throw std::runtime_error("Unsupported dtype for ne operation");
        }
    }

    return result;
}

// Less than kernel
auto lt_kernel(const Tensor& a, const Tensor& b) -> Tensor {
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
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            detail::broadcast_op<int32_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int32_t x, int32_t y) { return x < y; });
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            detail::broadcast_op<int64_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) { return x < y; });
        } else {
            throw std::runtime_error("Unsupported dtype for lt operation");
        }
    }

    return result;
}

// Less than or equal kernel
auto le_kernel(const Tensor& a, const Tensor& b) -> Tensor {
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
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] <= b_data[i]); }
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] <= b_data[i]); }
        } else {
            throw std::runtime_error("Unsupported dtype for le operation");
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
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            detail::broadcast_op<int32_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int32_t x, int32_t y) { return x <= y; });
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            detail::broadcast_op<int64_t, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) { return x <= y; });
        } else {
            throw std::runtime_error("Unsupported dtype for le operation");
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

    // Create scalar output tensor
    Tensor output({1}, a.dtype(), a.device());

    switch (a.dtype()) {
        case DType::Float32: {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* output_data = output.data<float>();

            float sum = 0.0f;
            #pragma omp parallel for reduction(+:sum) if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                sum += a_data[i] * b_data[i];
            }
            output_data[0] = sum;
            break;
        }
        case DType::Float64: {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* output_data = output.data<double>();

            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum) if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                sum += a_data[i] * b_data[i];
            }
            output_data[0] = sum;
            break;
        }
        case DType::Int32: {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            int32_t* output_data = output.data<int32_t>();

            int32_t sum = 0;
            #pragma omp parallel for reduction(+:sum) if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                sum += a_data[i] * b_data[i];
            }
            output_data[0] = sum;
            break;
        }
        case DType::Int64: {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            int64_t* output_data = output.data<int64_t>();

            int64_t sum = 0;
            #pragma omp parallel for reduction(+:sum) if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                sum += a_data[i] * b_data[i];
            }
            output_data[0] = sum;
            break;
        }
        default:
            throw std::runtime_error("dot: unsupported dtype");
    }

    return output;
}

// Trigonometric functions (SIMD + OpenMP optimized)
auto sin_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
#if defined(TENZOR_HAS_AVX512) || defined(__AVX512F__)
            fast_math::sin_batch_avx512(in_data, out_data, static_cast<size_t>(n));
#elif defined(TENZOR_HAS_AVX2) || defined(__AVX2__)
            fast_math::sin_batch_avx2(in_data, out_data, static_cast<size_t>(n));
#else
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::sin(in_data[i]);
            }
#endif
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::sin(in_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("sin: unsupported dtype");
    }
    return output;
}

auto cos_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
#if defined(TENZOR_HAS_AVX512) || defined(__AVX512F__)
            fast_math::cos_batch_avx512(in_data, out_data, static_cast<size_t>(n));
#elif defined(TENZOR_HAS_AVX2) || defined(__AVX2__)
            fast_math::cos_batch_avx2(in_data, out_data, static_cast<size_t>(n));
#else
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::cos(in_data[i]);
            }
#endif
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::cos(in_data[i]);
            }
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
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::tan(in_data[i]);
            }
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::tan(in_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("tan: unsupported dtype");
    }
    return output;
}

auto asin_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::asin(in_data[i]);
            }
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::asin(in_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("asin: unsupported dtype");
    }
    return output;
}

auto acos_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::acos(in_data[i]);
            }
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::acos(in_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("acos: unsupported dtype");
    }
    return output;
}

auto atan_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::atan(in_data[i]);
            }
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::atan(in_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("atan: unsupported dtype");
    }
    return output;
}

// Hyperbolic functions
auto sinh_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::sinh(in_data[i]);
            }
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::sinh(in_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("sinh: unsupported dtype");
    }
    return output;
}

auto cosh_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::cosh(in_data[i]);
            }
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::cosh(in_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("cosh: unsupported dtype");
    }
    return output;
}

// Rounding functions
auto round_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::round(in_data[i]);
            }
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::round(in_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("round: unsupported dtype");
    }
    return output;
}

auto floor_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::floor(in_data[i]);
            }
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::floor(in_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("floor: unsupported dtype");
    }
    return output;
}

auto ceil_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::ceil(in_data[i]);
            }
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = std::ceil(in_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("ceil: unsupported dtype");
    }
    return output;
}

// Reciprocal
auto reciprocal_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto output = Tensor::empty_uninitialized(shape_vec, input.dtype(), input.device());
    int64_t n = input.numel();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in_data = input.data<float>();
            float* out_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = 1.0f / in_data[i];
            }
            break;
        }
        case DType::Float64: {
            const double* in_data = input.data<double>();
            double* out_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                out_data[i] = 1.0 / in_data[i];
            }
            break;
        }
        default:
            throw std::runtime_error("reciprocal: unsupported dtype");
    }
    return output;
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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
                #pragma omp parallel for if(n > 10000)
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

// Helper: binary element-wise op with broadcast support
template<typename F32Fn, typename F64Fn>
auto binary_math_kernel(const Tensor& a, const Tensor& b, F32Fn f32_fn, F64Fn f64_fn,
                        const char* op_name) -> Tensor {
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
        if (a.dtype() == DType::Float32) {
            const float* ad = a.data<float>();
            const float* bd = b.data<float>();
            float* od = result.data<float>();
            for (size_t i = 0; i < n; ++i) od[i] = f32_fn(ad[i], bd[i]);
        } else if (a.dtype() == DType::Float64) {
            const double* ad = a.data<double>();
            const double* bd = b.data<double>();
            double* od = result.data<double>();
            for (size_t i = 0; i < n; ++i) od[i] = f64_fn(ad[i], bd[i]);
        } else {
            throw std::runtime_error(std::string(op_name) + ": unsupported dtype (requires float)");
        }
    } else {
        // Broadcast path using stride-based indexing
        auto a_strides = detail::compute_broadcast_strides(shape_a, output_shape);
        auto b_strides = detail::compute_broadcast_strides(shape_b, output_shape);
        int64_t ndim = static_cast<int64_t>(output_shape.size());
        int64_t n = result.numel();

        if (a.dtype() == DType::Float32) {
            const float* ad = a.data<float>();
            const float* bd = b.data<float>();
            float* od = result.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                int64_t a_idx = 0, b_idx = 0, idx = i;
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    int64_t coord = idx % output_shape[d];
                    idx /= output_shape[d];
                    a_idx += coord * a_strides[d];
                    b_idx += coord * b_strides[d];
                }
                od[i] = f32_fn(ad[a_idx], bd[b_idx]);
            }
        } else if (a.dtype() == DType::Float64) {
            const double* ad = a.data<double>();
            const double* bd = b.data<double>();
            double* od = result.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                int64_t a_idx = 0, b_idx = 0, idx = i;
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    int64_t coord = idx % output_shape[d];
                    idx /= output_shape[d];
                    a_idx += coord * a_strides[d];
                    b_idx += coord * b_strides[d];
                }
                od[i] = f64_fn(ad[a_idx], bd[b_idx]);
            }
        } else {
            throw std::runtime_error(std::string(op_name) + ": unsupported dtype (requires float)");
        }
    }
    return result;
}

} // anonymous namespace

auto log2_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) { return std::log2(x); },
        [](double x) { return std::log2(x); }, "log2");
}

auto log10_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) { return std::log10(x); },
        [](double x) { return std::log10(x); }, "log10");
}

auto log1p_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) { return std::log1p(x); },
        [](double x) { return std::log1p(x); }, "log1p");
}

auto exp2_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) { return std::exp2(x); },
        [](double x) { return std::exp2(x); }, "exp2");
}

auto expm1_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) { return std::expm1(x); },
        [](double x) { return std::expm1(x); }, "expm1");
}

auto erf_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) { return std::erf(x); },
        [](double x) { return std::erf(x); }, "erf");
}

auto erfc_kernel(const Tensor& input) -> Tensor {
    return unary_math_kernel(input,
        [](float x) { return std::erfc(x); },
        [](double x) { return std::erfc(x); }, "erfc");
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

    auto shape_vec = std::vector<int64_t>(start.shape().begin(), start.shape().end());
    Tensor result(shape_vec, start.dtype(), start.device());
    size_t n = static_cast<size_t>(result.numel());

    if (start.dtype() == DType::Float32) {
        const float* s = start.data<float>();
        const float* e = end.data<float>();
        float* o = result.data<float>();
        // Scalar weight (numel==1) or element-wise
        if (weight.numel() == 1) {
            float w = weight.data<float>()[0];
            for (size_t i = 0; i < n; ++i) o[i] = s[i] + w * (e[i] - s[i]);
        } else {
            const float* w = weight.data<float>();
            for (size_t i = 0; i < n; ++i) o[i] = s[i] + w[i] * (e[i] - s[i]);
        }
    } else if (start.dtype() == DType::Float64) {
        const double* s = start.data<double>();
        const double* e = end.data<double>();
        double* o = result.data<double>();
        if (weight.numel() == 1) {
            double w = weight.data<double>()[0];
            for (size_t i = 0; i < n; ++i) o[i] = s[i] + w * (e[i] - s[i]);
        } else {
            const double* w = weight.data<double>();
            for (size_t i = 0; i < n; ++i) o[i] = s[i] + w[i] * (e[i] - s[i]);
        }
    } else {
        throw std::runtime_error("lerp: unsupported dtype");
    }
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
    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        for (size_t i = 0; i < n; ++i)
            c_data[i] = (a_data[i] < b_data[i]) ? a_data[i] : b_data[i];
    } else {
        detail::broadcast_op<T, T>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
            [](T x, T y) -> T { return (x < y) ? x : y; });
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
    } else if (a.dtype() == DType::Int32) {
        minimum_typed(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else if (a.dtype() == DType::Int64) {
        minimum_typed(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else {
        throw std::runtime_error("Unsupported dtype for minimum operation");
    }
    return result;
}

template<typename T>
static void maximum_typed(const T* a_data, const T* b_data, T* c_data,
                          const Tensor& a, const Tensor& b,
                          std::vector<int64_t>& shape_a_vec,
                          std::vector<int64_t>& shape_b_vec,
                          std::vector<int64_t>& output_shape) {
    if (detail::have_same_shape(a, b)) {
        size_t n = static_cast<size_t>(a.numel());
        for (size_t i = 0; i < n; ++i)
            c_data[i] = (a_data[i] > b_data[i]) ? a_data[i] : b_data[i];
    } else {
        detail::broadcast_op<T, T>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
            [](T x, T y) -> T { return (x > y) ? x : y; });
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
    } else if (a.dtype() == DType::Int32) {
        maximum_typed(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else if (a.dtype() == DType::Int64) {
        maximum_typed(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
                      a, b, shape_a_vec, shape_b_vec, output_shape);
    } else {
        throw std::runtime_error("Unsupported dtype for maximum operation");
    }
    return result;
}

} // namespace cpu
} // namespace tenzor
