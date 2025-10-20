#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/utils/error.hpp"
#include "broadcast.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

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

namespace tenzor {
namespace cpu {

// Cache-friendly block sizes
constexpr size_t BLOCK_SIZE_M = 64;
constexpr size_t BLOCK_SIZE_N = 64;
constexpr size_t BLOCK_SIZE_K = 64;

// Micro-kernel for small block multiplication (Float32)
// Uses AVX-512 if available for maximum performance
#ifdef TENZOR_HAS_AVX512
__attribute__((target("avx512f")))
#endif
static void matmul_microkernel_float32(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {

#ifdef TENZOR_HAS_AVX512
    // AVX-512: Process 16 floats at a time
    constexpr int64_t simd_width = 16;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; j += simd_width) {
            // Handle remainder
            int64_t width = std::min(simd_width, N - j);

            if (width == simd_width) {
                // Full SIMD width
                __m512 c_vec = _mm512_loadu_ps(&C[i * ldc + j]);

                for (int64_t k = 0; k < K; ++k) {
                    __m512 a_vec = _mm512_set1_ps(A[i * lda + k]);
                    __m512 b_vec = _mm512_loadu_ps(&B[k * ldb + j]);
                    c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                }

                _mm512_storeu_ps(&C[i * ldc + j], c_vec);
            } else {
                // Scalar fallback for remainder
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
    // AVX2: Process 8 floats at a time
    constexpr int64_t simd_width = 8;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; j += simd_width) {
            int64_t width = std::min(simd_width, N - j);

            if (width == simd_width) {
                __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);

                for (int64_t k = 0; k < K; ++k) {
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

// Cache-blocked matrix multiplication (Float32)
static void matmul_blocked_float32(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K) {

    // Zero-initialize output
    std::fill_n(C, M * N, 0.0f);

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

                matmul_microkernel_float32(
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

// Cache-blocked matrix multiplication (Float64)
static void matmul_blocked_float64(
    const double* A, const double* B, double* C,
    int64_t M, int64_t N, int64_t K) {

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
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int32_t sum = C[i * ldc + j];
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// Cache-blocked matrix multiplication (Int32)
static void matmul_blocked_int32(
    const int32_t* A, const int32_t* B, int32_t* C,
    int64_t M, int64_t N, int64_t K) {

    // Zero-initialize output
    std::fill_n(C, M * N, 0);

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
            c[i] = std::numeric_limits<T>::infinity();
        } else {
            c[i] = a[i] / b[i];
        }
    }
}

// SIMD implementations for Float32
#ifdef TENZOR_HAS_AVX512

__attribute__((target("avx512f")))
inline void add_avx512_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 16;

    for (; i + simd_width <= n; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__attribute__((target("avx512f")))
inline void sub_avx512_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 16;

    for (; i + simd_width <= n; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_sub_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

__attribute__((target("avx512f")))
inline void mul_avx512_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 16;

    for (; i + simd_width <= n; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

__attribute__((target("avx512f")))
inline void div_avx512_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 16;

    for (; i + simd_width <= n; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_div_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }

    for (; i < n; ++i) {
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
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__attribute__((target("avx2,fma")))
inline void sub_avx2_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

__attribute__((target("avx2,fma")))
inline void mul_avx2_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }

    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

__attribute__((target("avx2,fma")))
inline void div_avx2_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    const size_t simd_width = 8;

    for (; i + simd_width <= n; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }

    for (; i < n; ++i) {
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
        if (a.dtype() == DType::Float32) {
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

        } else {
            throw std::runtime_error("Unsupported dtype for sub operation");
        }
    } else {
        // Broadcasting path
        if (a.dtype() == DType::Float32) {
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
                                [](int32_t x, int32_t y) { return (y == 0) ? 0 : x / y; });

        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            int64_t* c_data = result.data<int64_t>();
            detail::broadcast_op(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](int64_t x, int64_t y) { return (y == 0) ? 0 : x / y; });

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

    } else {
        throw std::runtime_error(
            "matmul unsupported dtype combination: " +
            std::string(dtype_name(a_contig.dtype())) + " @ " +
            std::string(dtype_name(b_contig.dtype()))
        );
    }

    return result;
}

// Square root kernel
auto sqrt_kernel(const Tensor& input) -> Tensor {
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

#ifdef TENZOR_HAS_AVX512
        // Process 16 floats at a time with AVX-512
        size_t simd_end = (n / 16) * 16;
        for (size_t i = 0; i < simd_end; i += 16) {
            __m512 v = _mm512_loadu_ps(&in_data[i]);
            __m512 sqrt_v = _mm512_sqrt_ps(v);
            _mm512_storeu_ps(&out_data[i], sqrt_v);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::sqrt(in_data[i]);
        }
#elif defined(TENZOR_HAS_AVX2)
        // Process 8 floats at a time with AVX2
        size_t simd_end = (n / 8) * 8;
        for (size_t i = 0; i < simd_end; i += 8) {
            __m256 v = _mm256_loadu_ps(&in_data[i]);
            __m256 sqrt_v = _mm256_sqrt_ps(v);
            _mm256_storeu_ps(&out_data[i], sqrt_v);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::sqrt(in_data[i]);
        }
#else
        // Scalar fallback
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::sqrt(in_data[i]);
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

#ifdef TENZOR_HAS_AVX512
        // Process 8 doubles at a time with AVX-512
        size_t simd_end = (n / 8) * 8;
        for (size_t i = 0; i < simd_end; i += 8) {
            __m512d v = _mm512_loadu_pd(&in_data[i]);
            __m512d sqrt_v = _mm512_sqrt_pd(v);
            _mm512_storeu_pd(&out_data[i], sqrt_v);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::sqrt(in_data[i]);
        }
#elif defined(TENZOR_HAS_AVX2)
        // Process 4 doubles at a time with AVX2
        size_t simd_end = (n / 4) * 4;
        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d v = _mm256_loadu_pd(&in_data[i]);
            __m256d sqrt_v = _mm256_sqrt_pd(v);
            _mm256_storeu_pd(&out_data[i], sqrt_v);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::sqrt(in_data[i]);
        }
#else
        // Scalar fallback
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::sqrt(in_data[i]);
        }
#endif

    } else {
        throw std::runtime_error("sqrt operation only supports Float32 and Float64 dtypes");
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
        // SIMD: Process 8 floats at a time
        size_t simd_end = (n / 8) * 8;
        __m256 zero = _mm256_setzero_ps();
        for (size_t i = 0; i < simd_end; i += 8) {
            __m256 v = _mm256_loadu_ps(&in_data[i]);
            __m256 neg_v = _mm256_sub_ps(zero, v);
            _mm256_storeu_ps(&out_data[i], neg_v);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = -in_data[i];
        }
#else
        // Scalar fallback
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = -in_data[i];
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

#ifdef TENZOR_HAS_AVX2
        // SIMD: Process 4 doubles at a time
        size_t simd_end = (n / 4) * 4;
        __m256d zero = _mm256_setzero_pd();
        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d v = _mm256_loadu_pd(&in_data[i]);
            __m256d neg_v = _mm256_sub_pd(zero, v);
            _mm256_storeu_pd(&out_data[i], neg_v);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = -in_data[i];
        }
#else
        // Scalar fallback
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = -in_data[i];
        }
#endif

    } else {
        throw std::runtime_error("neg operation only supports Float32 and Float64 dtypes");
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
        // SIMD: Process 8 floats at a time
        // Use sign bit mask (0x7FFFFFFF) to clear the sign bit
        size_t simd_end = (n / 8) * 8;
        __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        for (size_t i = 0; i < simd_end; i += 8) {
            __m256 v = _mm256_loadu_ps(&in_data[i]);
            __m256 abs_v = _mm256_and_ps(v, sign_mask);
            _mm256_storeu_ps(&out_data[i], abs_v);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::abs(in_data[i]);
        }
#else
        // Scalar fallback
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::abs(in_data[i]);
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

#ifdef TENZOR_HAS_AVX2
        // SIMD: Process 4 doubles at a time
        // Use sign bit mask (0x7FFFFFFFFFFFFFFF) to clear the sign bit
        size_t simd_end = (n / 4) * 4;
        __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d v = _mm256_loadu_pd(&in_data[i]);
            __m256d abs_v = _mm256_and_pd(v, sign_mask);
            _mm256_storeu_pd(&out_data[i], abs_v);
        }
        // Handle remainder
        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = std::abs(in_data[i]);
        }
#else
        // Scalar fallback
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::abs(in_data[i]);
        }
#endif

    } else {
        throw std::runtime_error("abs operation only supports Float32 and Float64 dtypes");
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
        // SIMD: Process 8 floats at a time
        size_t simd_end = (n / 8) * 8;
        __m256 min_vec = _mm256_set1_ps(min_val);
        __m256 max_vec = _mm256_set1_ps(max_val);

        for (size_t i = 0; i < simd_end; i += 8) {
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
        // Scalar fallback
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

    } else {
        throw std::runtime_error("clamp operation only supports Float32 and Float64 dtypes");
    }

    return result;
}

// Log kernel - natural logarithm
auto log_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

        // Scalar implementation (SIMD log is complex and typically uses SVML)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::log(in_data[i]);
        }

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::log(in_data[i]);
        }

    } else {
        throw std::runtime_error("log operation only supports Float32 and Float64 dtypes");
    }

    return result;
}

// Exp kernel - exponential
auto exp_kernel(const Tensor& input) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

        // Scalar implementation (SIMD exp is complex and typically uses SVML)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::exp(in_data[i]);
        }

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::exp(in_data[i]);
        }

    } else {
        throw std::runtime_error("exp operation only supports Float32 and Float64 dtypes");
    }

    return result;
}

// Pow kernel - power function
auto pow_kernel(const Tensor& input, float exponent) -> Tensor {
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor result(shape_vec, input.dtype(), input.device());
    size_t n = static_cast<size_t>(input.numel());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::pow(in_data[i], exponent);
        }

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        double exp_d = static_cast<double>(exponent);

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::pow(in_data[i], exp_d);
        }

    } else {
        throw std::runtime_error("pow operation only supports Float32 and Float64 dtypes");
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

    } else {
        throw std::runtime_error("sign operation only supports Float32 and Float64 dtypes");
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
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] == b_data[i]); }
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] == b_data[i]); }
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] == b_data[i]); }
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
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
                                [](float x, float y) { return x == y; });
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            detail::broadcast_op<double, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) { return x == y; });
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
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] < b_data[i]); }
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] < b_data[i]); }
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] < b_data[i]); }
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] < b_data[i]); }
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

        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] > b_data[i]); }
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] > b_data[i]); }
        } else if (a.dtype() == DType::Int32) {
            const int32_t* a_data = a.data<int32_t>();
            const int32_t* b_data = b.data<int32_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] > b_data[i]); }
        } else if (a.dtype() == DType::Int64) {
            const int64_t* a_data = a.data<int64_t>();
            const int64_t* b_data = b.data<int64_t>();
            for (size_t i = 0; i < n; ++i) { c_data[i] = (a_data[i] > b_data[i]); }
        } else {
            throw std::runtime_error("Unsupported dtype for gt operation");
        }
    } else {
        bool* c_data = result.data<bool>();
        if (a.dtype() == DType::Float32) {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            detail::broadcast_op<float, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](float x, float y) { return x > y; });
        } else if (a.dtype() == DType::Float64) {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            detail::broadcast_op<double, bool>(a_data, b_data, c_data, shape_a_vec, shape_b_vec, output_shape,
                                [](double x, double y) { return x > y; });
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
        } else {
            throw std::runtime_error("Unsupported dtype for ge operation");
        }
    }

    return result;
}

} // namespace cpu
} // namespace tenzor
