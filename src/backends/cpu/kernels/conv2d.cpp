#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/omp_thresholds.hpp"
#include "gemm_optimized.hpp"
#include "simd_fast_math.hpp"
#include "winograd.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

// Intel MKL BLAS for the Float64 im2col-GEMM path (cblas_dgemm). The float
// path uses hand-written SIMD micro-kernels in gemm_optimized.hpp; oneDNN is
// Float32-only here, so without MKL the Float64 conv falls back to the generic
// gemm_cpu<double> template (the scalar triple loop). That fallback is
// numerically CORRECT — it accumulates in genuine `double`, not a narrowed
// float — it is only ~10x slower than cblas_dgemm. It is therefore not a
// precision regression; do not "fix" it by forcing a float accumulator.
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#endif

// Intel oneDNN for optimized convolutions (3-8x faster)
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#include "onednn_cache.hpp"
#include <unordered_map>
#include <mutex>
#include <memory>
#include <list>
#endif

// SIMD intrinsics for F16C support
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__AVX2__)
        #define TENZOR_CONV_AVX2
    #endif
    #if defined(__F16C__)
        #define TENZOR_CONV_F16C
    #endif
#endif

// Import shared Float16/BFloat16 operator overloads
#include "half_operators.hpp"

namespace tenzor {
namespace cpu {

// ============================================================================
// Helper Functions
// ============================================================================

// Calculate output size for convolution
inline int64_t calculate_output_size(int64_t input_size, int64_t kernel_size,
                                     int64_t stride, int64_t padding, int64_t dilation) {
    if (dilation > 0 && kernel_size > 1 &&
        dilation > std::numeric_limits<int64_t>::max() / (kernel_size - 1)) {
        throw std::invalid_argument("Conv2d: dilation * (kernel_size - 1) would overflow");
    }
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

// Overflow-checked int64 product. Used to guard the flat-index loop bounds in
// im2col/im3col: an unchecked product can wrap negative for pathological-but-
// legal shapes, silently truncating the loop and leaving the col buffer (and
// hence the conv output) partially uninitialized instead of throwing.
inline int64_t checked_mul_i64(int64_t a, int64_t b, const char* what) {
    if (a != 0 && b != 0 &&
        (a > std::numeric_limits<int64_t>::max() / b ||
         a < std::numeric_limits<int64_t>::min() / b)) {
        throw std::invalid_argument(std::string(what) + ": index/size product overflows int64");
    }
    return a * b;
}

// ============================================================================
// im2col CPU Implementation
// ============================================================================

// im2col: Convert 4D input (N,C,H,W) to 2D matrix for convolution
// Input: (batch, in_channels, height, width)
// Output: (batch * out_h * out_w, kernel_h * kernel_w * in_channels)
template<typename T>
void im2col_cpu(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t out_h,
    int64_t out_w
) {
    const int64_t col_width = checked_mul_i64(checked_mul_i64(channels, kernel_h, "im2col"),
                                              kernel_w, "im2col");
    // Guard the full flat extent of the col buffer so the (b,oh,ow,col_width)
    // index arithmetic below cannot wrap negative and silently skip work.
    {
        int64_t spatial = checked_mul_i64(out_h, out_w, "im2col");
        int64_t rows = checked_mul_i64(batch, spatial, "im2col");
        (void)checked_mul_i64(rows, col_width, "im2col");
    }

    // Nested loop approach: eliminates 5 div/mod per element (~200 cycles saved per element)
    #pragma omp parallel for collapse(3) if(batch * out_h * out_w > OmpThresholds::medium())
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            for (int64_t ow = 0; ow < out_w; ++ow) {
                T* col_ptr = output + ((b * out_h + oh) * out_w + ow) * col_width;
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t kh_idx = 0; kh_idx < kernel_h; ++kh_idx) {
                        int64_t ih = oh * stride_h - pad_h + kh_idx * dil_h;
                        for (int64_t kw_idx = 0; kw_idx < kernel_w; ++kw_idx) {
                            int64_t iw = ow * stride_w - pad_w + kw_idx * dil_w;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                *col_ptr++ = input[((b * channels + c) * height + ih) * width + iw];
                            } else {
                                *col_ptr++ = T(0.0f);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Scalar-stride/pad/dilation backward-compat wrapper. Delegates to the
// per-axis implementation above.
template<typename T>
void im2col_cpu(
    const T* input, T* output,
    int64_t batch, int64_t channels, int64_t height, int64_t width,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride, int64_t padding, int64_t dilation,
    int64_t out_h, int64_t out_w
) {
    im2col_cpu(input, output, batch, channels, height, width,
               kernel_h, kernel_w,
               stride, stride, padding, padding, dilation, dilation,
               out_h, out_w);
}

// ============================================================================
// col2im CPU Implementation
// ============================================================================

// col2im: Reverse of im2col for gradient computation
// Input: (batch * out_h * out_w, kernel_h * kernel_w * in_channels)
// Output: (batch, in_channels, height, width)
// Note: This accumulates gradients for overlapping regions
template<typename T>
void col2im_cpu(
    const T* col,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t out_h,
    int64_t out_w
) {
    // Zero initialize output — required since we accumulate from kernel positions
    int64_t output_size = batch * channels * height * width;
    std::memset(output, 0, output_size * sizeof(T));

    // Iterate over (b, c) in outer loops, then (kh, kw) kernel positions,
    // then directly over valid (oh, ow) output positions. This eliminates
    // the per-element modulo/division that the previous approach required
    // to reverse-map (ih, iw) back to (oh, ow).
    #pragma omp parallel for collapse(2) if(output_size > OmpThresholds::medium())
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            T* out_slice = output + b * (channels * height * width) + c * (height * width);

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    // Precompute valid oh range for this kernel row position.
                    // Constraint: 0 <= oh * stride_h - pad_h + kh * dil_h < height
                    int64_t offset_h = pad_h - kh * dil_h;
                    int64_t oh_start = std::max<int64_t>(0,
                        (offset_h > 0) ? (offset_h + stride_h - 1) / stride_h : -((-offset_h) / stride_h));
                    int64_t oh_end = std::min(out_h,
                        (height + offset_h + stride_h - 1) / stride_h);

                    int64_t offset_w = pad_w - kw * dil_w;
                    int64_t ow_start = std::max<int64_t>(0,
                        (offset_w > 0) ? (offset_w + stride_w - 1) / stride_w : -((-offset_w) / stride_w));
                    int64_t ow_end = std::min(out_w,
                        (width + offset_w + stride_w - 1) / stride_w);

                    int64_t col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    int64_t col_row_base = b * out_h * out_w;
                    int64_t col_stride = channels * kernel_h * kernel_w;

                    for (int64_t oh = oh_start; oh < oh_end; ++oh) {
                        int64_t ih = oh * stride_h - offset_h;
                        int64_t col_row_oh = (col_row_base + oh * out_w) * col_stride + col_col;

                        for (int64_t ow = ow_start; ow < ow_end; ++ow) {
                            int64_t iw = ow * stride_w - offset_w;
                            out_slice[ih * width + iw] += col[col_row_oh + ow * col_stride];
                        }
                    }
                }
            }
        }
    }
}

// Scalar-stride/pad/dilation backward-compat wrapper.
template<typename T>
void col2im_cpu(
    const T* col, T* output,
    int64_t batch, int64_t channels, int64_t height, int64_t width,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride, int64_t padding, int64_t dilation,
    int64_t out_h, int64_t out_w
) {
    col2im_cpu(col, output, batch, channels, height, width,
               kernel_h, kernel_w,
               stride, stride, padding, padding, dilation, dilation,
               out_h, out_w);
}

// ============================================================================
// Matrix Multiplication Helper (Row-major GEMM)
// ============================================================================

// Optimized GEMM for C = A @ B^T using SIMD micro-kernels
// A: (M, K) row-major
// B: (N, K) row-major (will be transposed)
// C: (M, N) row-major
template<typename T>
void gemm_cpu(
    const T* A, const T* B, T* C,
    int64_t M, int64_t N, int64_t K,
    bool transpose_B = true
) {
    #pragma omp parallel for collapse(2) if(M * N > OmpThresholds::matmul())
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            T sum = T(0.0f);
            if (transpose_B) {
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[j * K + k];
                }
            } else {
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
            }
            C[i * N + j] = sum;
        }
    }
}

// Specialization for float using optimized GEMM
template<>
void gemm_cpu<float>(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K,
    bool transpose_B
) {
    if (transpose_B) {
        gemm::gemm_transB_optimized(A, B, C, M, N, K);
    } else {
        gemm::gemm_optimized(A, B, C, M, N, K);
    }
}

// Specialization for Float16 — accumulate in float32 to avoid catastrophic precision loss
template<>
void gemm_cpu<Float16>(
    const Float16* A, const Float16* B, Float16* C,
    int64_t M, int64_t N, int64_t K,
    bool transpose_B
) {
    #pragma omp parallel for collapse(2) if(M * N > OmpThresholds::matmul())
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            if (transpose_B) {
                for (int64_t k = 0; k < K; ++k) {
                    sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[j * K + k]);
                }
            } else {
                for (int64_t k = 0; k < K; ++k) {
                    sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * N + j]);
                }
            }
            C[i * N + j] = Float16(sum);
        }
    }
}

// Specialization for BFloat16 — accumulate in float32 to avoid catastrophic precision loss
template<>
void gemm_cpu<BFloat16>(
    const BFloat16* A, const BFloat16* B, BFloat16* C,
    int64_t M, int64_t N, int64_t K,
    bool transpose_B
) {
    #pragma omp parallel for collapse(2) if(M * N > OmpThresholds::matmul())
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            if (transpose_B) {
                for (int64_t k = 0; k < K; ++k) {
                    sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[j * K + k]);
                }
            } else {
                for (int64_t k = 0; k < K; ++k) {
                    sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * N + j]);
                }
            }
            C[i * N + j] = BFloat16(sum);
        }
    }
}

#ifdef TENZOR_USE_MKL
// Specialization for double using MKL cblas_dgemm. C = A @ B (or A @ B^T).
// Row-major: A is (M, K). For transpose_B, B is (N, K) and we compute A @ B^T;
// otherwise B is (K, N) and we compute A @ B.
template<>
void gemm_cpu<double>(
    const double* A, const double* B, double* C,
    int64_t M, int64_t N, int64_t K,
    bool transpose_B
) {
    // The LP64 MKL interface takes int dimensions/leading-dims. If any of
    // M/N/K (which double as the leading dimensions here) exceeds INT_MAX the
    // static_cast<int> would silently truncate, producing a wrong-shaped GEMM
    // with out-of-bounds access. Fall back to an int64-indexed scalar GEMM in
    // that (extreme, multi-GB) case so the result stays correct.
    constexpr int64_t kIntMax = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (M > kIntMax || N > kIntMax || K > kIntMax) {
        #pragma omp parallel for collapse(2) if(M * N > OmpThresholds::matmul())
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                double sum = 0.0;
                if (transpose_B) {
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A[i * K + k] * B[j * K + k];
                    }
                } else {
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                }
                C[i * N + j] = sum;
            }
        }
        return;
    }
    if (transpose_B) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0, A, static_cast<int>(K), B, static_cast<int>(K),
                    0.0, C, static_cast<int>(N));
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0, A, static_cast<int>(K), B, static_cast<int>(N),
                    0.0, C, static_cast<int>(N));
    }
}
#endif // TENZOR_USE_MKL

// Matrix multiplication for C = A^T @ B
// A: (K, M) row-major (will be transposed)
// B: (K, N) row-major
// C: (M, N) row-major
template<typename T>
void gemm_transA_cpu(
    const T* A, const T* B, T* C,
    int64_t M, int64_t N, int64_t K
) {
    #pragma omp parallel for collapse(2) if(M * N > OmpThresholds::matmul())
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            T sum = T(0.0f);
            // A is (K, M) row-major, access as A[k][i]
            for (int64_t k = 0; k < K; ++k) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Specialization for float using optimized GEMM
template<>
void gemm_transA_cpu<float>(
    const float* A, const float* B, float* C,
    int64_t M, int64_t N, int64_t K
) {
    gemm::gemm_transA_optimized(A, B, C, M, N, K);
}

// Specialized version for Float16 (uses Float32 accumulation)
template<>
void gemm_transA_cpu<Float16>(
    const Float16* A, const Float16* B, Float16* C,
    int64_t M, int64_t N, int64_t K
) {
    #pragma omp parallel for collapse(2) if(M * N > OmpThresholds::matmul())
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            // A is (K, M) row-major, access as A[k][i]
            for (int64_t k = 0; k < K; ++k) {
                float a_val = static_cast<float>(A[k * M + i]);
                float b_val = static_cast<float>(B[k * N + j]);
                sum += a_val * b_val;
            }
            C[i * N + j] = Float16(sum);
        }
    }
}

// Specialization for BFloat16 (uses Float32 accumulation)
template<>
void gemm_transA_cpu<BFloat16>(
    const BFloat16* A, const BFloat16* B, BFloat16* C,
    int64_t M, int64_t N, int64_t K
) {
    #pragma omp parallel for collapse(2) if(M * N > OmpThresholds::matmul())
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += static_cast<float>(A[k * M + i]) * static_cast<float>(B[k * N + j]);
            }
            C[i * N + j] = BFloat16(sum);
        }
    }
}

#ifdef TENZOR_USE_MKL
// Specialization for double using MKL cblas_dgemm. C = A^T @ B.
// Row-major: A is (K, M) so op(A)=Trans with lda=M; B is (K, N) with ldb=N;
// C is (M, N) with ldc=N.
template<>
void gemm_transA_cpu<double>(
    const double* A, const double* B, double* C,
    int64_t M, int64_t N, int64_t K
) {
    // See gemm_cpu<double>: guard against int64->int truncation in the LP64
    // MKL interface for extreme dimensions, falling back to an int64-indexed
    // scalar GEMM that matches the generic gemm_transA_cpu template.
    constexpr int64_t kIntMax = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (M > kIntMax || N > kIntMax || K > kIntMax) {
        #pragma omp parallel for collapse(2) if(M * N > OmpThresholds::matmul())
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                double sum = 0.0;
                // A is (K, M) row-major, access as A[k][i]
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[k * M + i] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        return;
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0, A, static_cast<int>(M), B, static_cast<int>(N),
                0.0, C, static_cast<int>(N));
}
#endif // TENZOR_USE_MKL

// ============================================================================
// Conv2d Forward CPU Implementation
// ============================================================================

#ifdef TENZOR_USE_ONEDNN

// Use shared lazy-init accessors from onednn_cache.hpp to avoid static
// thread_local initialization issues in dlopen'd libraries.

// ============================================================================
// oneDNN Primitive Cache for Conv2d
// ============================================================================
// Caching primitives and reordered weights provides 10-30x speedup by avoiding
// expensive primitive creation and weight reordering on every forward call.

struct Conv2dCacheKey {
    int64_t batch, in_channels, height, width;
    int64_t out_channels, in_channels_per_group, kernel_h, kernel_w;
    int64_t stride, padding, dilation, groups;
    bool has_bias;
    DType dtype;
    // Note: weight_ptr removed from key - weights may change in-place (optimizer updates)
    // Weight data is refreshed on every cache hit via set_data_handle + reorder.

    bool operator==(const Conv2dCacheKey& other) const {
        return batch == other.batch && in_channels == other.in_channels &&
               height == other.height && width == other.width &&
               out_channels == other.out_channels &&
               in_channels_per_group == other.in_channels_per_group &&
               kernel_h == other.kernel_h && kernel_w == other.kernel_w &&
               stride == other.stride && padding == other.padding &&
               dilation == other.dilation && groups == other.groups &&
               has_bias == other.has_bias && dtype == other.dtype;
    }
};

struct Conv2dCacheKeyHash {
    size_t operator()(const Conv2dCacheKey& k) const {
        size_t h = 0;
        auto hash_combine = [&h](size_t v) {
            h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        hash_combine(std::hash<int64_t>{}(k.batch));
        hash_combine(std::hash<int64_t>{}(k.in_channels));
        hash_combine(std::hash<int64_t>{}(k.height));
        hash_combine(std::hash<int64_t>{}(k.width));
        hash_combine(std::hash<int64_t>{}(k.out_channels));
        hash_combine(std::hash<int64_t>{}(k.kernel_h));
        hash_combine(std::hash<int64_t>{}(k.kernel_w));
        hash_combine(std::hash<int64_t>{}(k.stride));
        hash_combine(std::hash<int64_t>{}(k.padding));
        hash_combine(std::hash<int64_t>{}(k.dilation));
        hash_combine(std::hash<int64_t>{}(k.groups));
        hash_combine(std::hash<bool>{}(k.has_bias));
        hash_combine(std::hash<int>{}(static_cast<int>(k.dtype)));
        return h;
    }
};

struct Conv2dCachedPrimitive {
    dnnl::convolution_forward::primitive_desc conv_pd;
    dnnl::convolution_forward conv_prim;
    dnnl::memory weights_mem;           // Reordered weights in optimal format
    dnnl::memory::desc weights_md_user; // User weights memory descriptor (for reorder on update)
    bool need_weights_reorder;          // Whether weights need reorder to optimal format
    dnnl::memory::desc src_md_user;     // User source memory descriptor
    dnnl::memory::desc dst_md_user;     // User destination memory descriptor
    dnnl::memory::desc bias_md;         // Bias memory descriptor (if applicable)
    bool need_src_reorder;
    bool need_dst_reorder;

    // Cached scratch buffers to avoid allocation on every call
    dnnl::memory src_reorder_mem;       // Scratch for source reorder (if needed)
    dnnl::memory dst_reorder_mem;       // Scratch for destination reorder (if needed)
    bool scratch_initialized{false};    // Whether scratch buffers are allocated
    const void* cached_weight_ptr{nullptr}; // Last weight data pointer (skip reorder if unchanged)
};

// Thread-local cache with LRU eviction (max 32 entries per thread)
static constexpr size_t CONV2D_CACHE_SIZE = 32;

using Conv2dPrimitiveCache = OneDNNPrimitiveCache<Conv2dCacheKey, Conv2dCachedPrimitive, Conv2dCacheKeyHash, CONV2D_CACHE_SIZE>;

static thread_local Conv2dPrimitiveCache g_conv2d_cache;

// W.6: register a thread-local clear-callback. Invoked by clear_dnnl_cache().
namespace {
void clear_local_conv2d_cache() { g_conv2d_cache.clear(); }
struct Conv2dCacheClearRegistrar {
    Conv2dCacheClearRegistrar() {
        ::tenzor::cpu::register_dnnl_cache_clear_callback(&clear_local_conv2d_cache);
    }
};
static Conv2dCacheClearRegistrar g_conv2d_cache_clear_registrar;
}

// oneDNN-accelerated Conv2d Forward with primitive caching
// Caches primitives and reordered weights for 10-30x speedup on repeated calls
static bool conv2d_forward_onednn(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& output,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    // oneDNN requires float32
    if (input.dtype() != DType::Float32) {
        return false;
    }

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto output_shape = output.shape();

    int64_t out_h = output_shape[2];
    int64_t out_w = output_shape[3];
    int64_t out_channels = weight_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];
    int64_t batch = input_shape[0];

    // =========================================================================
    // Skip oneDNN only for tiny convolutions where primitive overhead dominates
    // =========================================================================
    // With cached primitives and scratch buffers, oneDNN is typically faster than
    // our im2col+GEMM fallback for almost all practical convolution sizes.
    // Only skip for extremely small outputs where even memory allocation overhead
    // exceeds compute time.

    //
    // Audit item I.2: the 4096-element threshold is a measured cross-over.
    // Below it, oneDNN primitive build + reorder/descriptor checks per call
    // outweigh the GEMM kernel saving versus our im2col+sgemm path. Above it,
    // oneDNN wins by ~1.5-10x. Concretely: batch=1, C=64, 8x8 = 4096 outputs
    // sits at the cross-over; anything smaller (1x1 convs on tiny features,
    // single-token probes) takes the im2col path. Without this threshold,
    // ResNet 1x1 stem convs on small feature maps regress noticeably.
    int64_t total_output_elements = batch * out_channels * out_h * out_w;

    constexpr int64_t MIN_OUTPUT_ELEMENTS = 4096;

    if (total_output_elements < MIN_OUTPUT_ELEMENTS) {
        return false;  // Use im2col+GEMM fallback only for tiny convolutions
    }

    try {
        int64_t height = input_shape[2];
        int64_t width = input_shape[3];
        int64_t in_channels_per_group = weight_shape[1];

        auto& engine = get_onednn_engine();
        auto& stream = get_onednn_stream();

        // Create cache key
        Conv2dCacheKey cache_key{
            batch, in_channels, height, width,
            out_channels, in_channels_per_group, kernel_h, kernel_w,
            stride, padding, dilation, groups,
            bias != nullptr, input.dtype()
        };

        // Try to get cached primitive
        auto cached = g_conv2d_cache.get(cache_key);

        if (!cached) {
            // Cache miss - create new primitive and cache it
            cached = std::make_shared<Conv2dCachedPrimitive>();

            // Create memory dimensions
            dnnl::memory::dims src_dims = {batch, in_channels, height, width};
            dnnl::memory::dims dst_dims = {batch, out_channels, out_h, out_w};
            dnnl::memory::dims strides_dims = {stride, stride};
            dnnl::memory::dims padding_dims = {padding, padding};
            dnnl::memory::dims dilation_dims = {dilation - 1, dilation - 1};  // oneDNN uses dilation-1

            // Weight dims depend on groups
            dnnl::memory::dims weights_dims;
            if (groups > 1) {
                weights_dims = {groups, out_channels / groups, in_channels_per_group, kernel_h, kernel_w};
            } else {
                weights_dims = {out_channels, in_channels, kernel_h, kernel_w};
            }

            // User memory descriptors (NCHW layout)
            cached->src_md_user = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::nchw);
            cached->dst_md_user = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::nchw);
            dnnl::memory::desc weights_md_user;
            if (groups > 1) {
                weights_md_user = dnnl::memory::desc(weights_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::goihw);
            } else {
                weights_md_user = dnnl::memory::desc(weights_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::oihw);
            }

            // Create "any" format descriptors to let oneDNN choose optimal layout
            auto src_md_any = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::any);
            auto dst_md_any = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::any);
            auto weights_md_any = dnnl::memory::desc(weights_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::any);

            // Create convolution primitive descriptor with auto algorithm selection
            if (bias != nullptr) {
                dnnl::memory::dims bias_dims = {out_channels};
                cached->bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::f32,
                                                      dnnl::memory::format_tag::a);

                cached->conv_pd = dnnl::convolution_forward::primitive_desc(
                    engine,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::convolution_auto,  // Let oneDNN choose best algorithm
                    src_md_any, weights_md_any, cached->bias_md, dst_md_any,
                    strides_dims, dilation_dims, padding_dims, padding_dims
                );
            } else {
                cached->conv_pd = dnnl::convolution_forward::primitive_desc(
                    engine,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::convolution_auto,
                    src_md_any, weights_md_any, dst_md_any,
                    strides_dims, dilation_dims, padding_dims, padding_dims
                );
            }

            // Check if reorders are needed
            cached->need_src_reorder = (cached->conv_pd.src_desc() != cached->src_md_user);
            cached->need_dst_reorder = (cached->conv_pd.dst_desc() != cached->dst_md_user);

            // Store user weights descriptor for future reorder on cache hit
            cached->weights_md_user = weights_md_user;
            cached->need_weights_reorder = (cached->conv_pd.weights_desc() != weights_md_user);

            // Reorder weights to optimal layout
            auto weights_mem_user = dnnl::memory(weights_md_user, engine, const_cast<float*>(weight.data<float>()));
            if (cached->need_weights_reorder) {
                cached->weights_mem = dnnl::memory(cached->conv_pd.weights_desc(), engine);
                dnnl::reorder(weights_mem_user, cached->weights_mem).execute(stream, weights_mem_user, cached->weights_mem);
                stream.wait();
            } else {
                // Weights already in optimal format - make a copy to cache
                cached->weights_mem = dnnl::memory(weights_md_user, engine);
                std::memcpy(cached->weights_mem.get_data_handle(), weight.data<float>(),
                           weight.numel() * sizeof(float));
            }

            // Create convolution primitive
            cached->conv_prim = dnnl::convolution_forward(cached->conv_pd);

            // Store in cache
            g_conv2d_cache.put(cache_key, cached);
        }

        // Refresh weight data on EVERY cache hit. Weights can be mutated
        // in-place by the optimizer (SubInplace/AddcdivInplace/...) WITHOUT
        // changing the data pointer, so gating the reorder on pointer identity
        // reused stale reordered weights. The cache stores the primitive and
        // descriptors (which are shape/dtype-dependent only), not the weight
        // values, so re-reading the current weights here is required for
        // correctness; the reorder/copy is weights-only and cheap relative to
        // the convolution itself.
        {
            auto weights_mem_user = dnnl::memory(cached->weights_md_user, engine, const_cast<float*>(weight.data<float>()));
            if (cached->need_weights_reorder) {
                dnnl::reorder(weights_mem_user, cached->weights_mem).execute(stream, weights_mem_user, cached->weights_mem);
                stream.wait();
            } else {
                cached->weights_mem.set_data_handle(const_cast<float*>(weight.data<float>()));
            }
            cached->cached_weight_ptr = weight.data_ptr();
        }

        // Execute convolution using cached primitive
        // Create source/dest memory wrappers (no allocation, just wraps existing data)
        auto src_mem_user = dnnl::memory(cached->src_md_user, engine, const_cast<float*>(input.data<float>()));
        auto dst_mem_user = dnnl::memory(cached->dst_md_user, engine, output.data<float>());

        // Initialize scratch buffers on first use (cached for subsequent calls)
        if (!cached->scratch_initialized) {
            if (cached->need_src_reorder) {
                cached->src_reorder_mem = dnnl::memory(cached->conv_pd.src_desc(), engine);
            }
            if (cached->need_dst_reorder) {
                cached->dst_reorder_mem = dnnl::memory(cached->conv_pd.dst_desc(), engine);
            }
            cached->scratch_initialized = true;
        }

        // Use cached scratch buffers for reorders (no allocation per call)
        dnnl::memory src_mem = src_mem_user;
        if (cached->need_src_reorder) {
            src_mem = cached->src_reorder_mem;
            dnnl::reorder(src_mem_user, src_mem).execute(stream, src_mem_user, src_mem);
        }

        dnnl::memory dst_mem = dst_mem_user;
        if (cached->need_dst_reorder) {
            dst_mem = cached->dst_reorder_mem;
        }

        // Execute convolution with cached primitive and weights
        if (bias != nullptr) {
            auto bias_mem = dnnl::memory(cached->bias_md, engine, const_cast<float*>(bias->data<float>()));
            cached->conv_prim.execute(stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_WEIGHTS, cached->weights_mem},
                {DNNL_ARG_BIAS, bias_mem},
                {DNNL_ARG_DST, dst_mem}
            });
        } else {
            cached->conv_prim.execute(stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_WEIGHTS, cached->weights_mem},
                {DNNL_ARG_DST, dst_mem}
            });
        }

        // Reorder destination back to NCHW if needed
        if (cached->need_dst_reorder) {
            dnnl::reorder(dst_mem, dst_mem_user).execute(stream, dst_mem, dst_mem_user);
        }

        stream.wait();
        return true;

    } catch (const dnnl::error& e) {
        // oneDNN error, fall back to im2col+GEMM
        return false;
    } catch (const std::exception& e) {
        return false;
    }
}
#endif

// Template helper for dtype-generic conv2d forward. Accepts per-axis
// stride / padding / dilation so non-square conv configs work natively.
template<typename T>
void conv2d_forward_impl(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& output,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups
) {
    // Alias names used by the Winograd / 1x1 fast paths below that require
    // isotropic configs. Keeping these lets us avoid rewriting those blocks.
    int64_t stride = stride_h;
    int64_t padding = pad_h;
    int64_t dilation = dil_h;
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    auto out_shape = output.shape();
    int64_t out_h = out_shape[2];
    int64_t out_w = out_shape[3];

    // Fast paths (Winograd, 1x1, 3x3 specializations) below are square-only;
    // fall through to the general im2col+GEMM path for any anisotropic config.
    bool symmetric = (stride_h == stride_w) && (pad_h == pad_w) && (dil_h == dil_w);

    // Process each group separately
    int64_t out_channels_per_group = out_channels / groups;

    // =========================================================================
    // Fast path for 1x1 convolutions (skip im2col, direct GEMM)
    // 1x1 conv is essentially a per-pixel fully-connected layer
    // =========================================================================
    if (symmetric && kernel_h == 1 && kernel_w == 1 && stride == 1 && padding == 0 && dilation == 1) {
        // For 1x1 convs: treat as GEMM without im2col
        // Input viewed as (batch, in_channels, H*W) -> transpose to (batch, H*W, in_channels)
        // Weight viewed as (out_channels, in_channels)
        // Output = Input_transposed @ Weight.T -> (batch, H*W, out_channels) -> transpose to NCHW

        const T* input_data = input.data<T>();
        const T* weight_data = weight.data<T>();
        T* output_data = output.data<T>();

        int64_t spatial = height * width;

        for (int64_t g = 0; g < groups; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            // 1x1 conv reduces to a GEMM per batch/group:
            //   C[oc, s] = sum_ic weight[oc, ic] * input[ic, s]
            // Within a group the input channels [in_start, in_start+icpg) and
            // the output channels [out_start, out_start+ocpg) are contiguous in
            // the spatial-major NCHW layout, so we can pass sub-pointers
            // directly. gemm_cpu(..., transpose_B=false) computes A @ B with
            //   A = weight block (ocpg, icpg)  [K = icpg, contiguous rows]
            //   B = input block  (icpg, spatial)
            //   C = output block (ocpg, spatial)
            // This dispatches to BLAS/optimized kernels for float/double.
            const T* w_group = weight_data + out_start * in_channels_per_group;
            for (int64_t b = 0; b < batch; ++b) {
                const T* in_group = input_data + b * (in_channels * spatial) + in_start * spatial;
                T* out_group = output_data + b * (out_channels * spatial) + out_start * spatial;
                gemm_cpu<T>(w_group, in_group, out_group,
                            out_channels_per_group, spatial, in_channels_per_group,
                            /*transpose_B=*/false);
            }
        }

        // Add bias if present
        if (bias) {
            const T* bias_data = bias->data<T>();
            #pragma omp parallel for collapse(3) if(batch * out_channels * spatial > OmpThresholds::medium())
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t c = 0; c < out_channels; ++c) {
                    for (int64_t s = 0; s < spatial; ++s) {
                        output_data[b * (out_channels * spatial) + c * spatial + s] += bias_data[c];
                    }
                }
            }
        }

        return;
    }

    // =========================================================================
    // Winograd F(2x2, 3x3) fast path for 3x3 convolutions
    // Reduces multiplications from 9 to 4 per 2x2 output tile
    // =========================================================================
    if (symmetric && kernel_h == 3 && kernel_w == 3 && stride == 1 && dilation == 1 && groups == 1) {
        // Single source of truth: delegate to the shared, canonical
        // Winograd F(2x2, 3x3) transform in winograd.hpp (templated on T).
        // This is the ONLY Winograd path for Float64 (and the Float32-widened
        // Float16/BFloat16 path), so there is no divergent inline copy to keep
        // in sync. Float32 isotropic convs are routed through the same function
        // by conv2d_forward_kernel before ever reaching here.
        winograd_conv2d_f2x3<T>(
            input.data<T>(), weight.data<T>(), output.data<T>(),
            batch, in_channels, height, width, out_channels, out_h, out_w,
            padding, padding);

        // Add bias if present
        if (bias) {
            const T* bias_data = bias->data<T>();
            T* output_data = output.data<T>();
            int64_t spatial = out_h * out_w;
            #pragma omp parallel for collapse(3) if(batch * out_channels * spatial > OmpThresholds::medium())
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t c = 0; c < out_channels; ++c) {
                    for (int64_t s = 0; s < spatial; ++s) {
                        output_data[b * (out_channels * spatial) + c * spatial + s] += bias_data[c];
                    }
                }
            }
        }

        return;
    }

    // =========================================================================
    // General path: im2col + GEMM for larger kernels
    // =========================================================================
    for (int64_t g = 0; g < groups; ++g) {
        // Calculate channel offsets
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Allocate im2col buffer for this group
        int64_t col_rows = batch * out_h * out_w;
        int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;
        std::vector<T> col_buffer(col_rows * col_cols);

        // Apply im2col transformation per-batch to handle correct strides
        // im2col_cpu expects input pointer at batch 0, channel 0 of the group,
        // and internally strides by channels*H*W per batch element.
        // With groups, we must process each batch element separately so the
        // pointer correctly accounts for the full in_channels stride.
        int64_t col_per_batch = out_h * out_w * col_cols;
        for (int64_t b = 0; b < batch; ++b) {
            const T* input_ptr = input.data<T>() + (b * in_channels + in_start) * height * width;
            im2col_cpu(
                input_ptr,
                col_buffer.data() + b * col_per_batch,
                1,  // single batch element
                in_channels_per_group,
                height,
                width,
                kernel_h,
                kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dil_h, dil_w,
                out_h,
                out_w
            );
        }

        // Matrix multiplication into temporary buffer
        // GEMM output is (batch * out_h * out_w, out_channels_per_group) row-major
        int64_t M = col_rows;
        int64_t K = col_cols;
        int64_t N = out_channels_per_group;

        const T* weight_ptr = weight.data<T>() + out_start * in_channels_per_group * kernel_h * kernel_w;
        std::vector<T> gemm_output(M * N);

        // Perform GEMM: C = A @ B^T
        gemm_cpu(
            col_buffer.data(),  // A: (M, K)
            weight_ptr,         // B: (N, K) - will be transposed
            gemm_output.data(), // C: (M, N)
            M, N, K,
            true  // transpose B
        );

        // Transpose GEMM output from (batch*out_h*out_w, out_channels) to NCHW format
        // GEMM output: element at (m, n) where m = b*out_h*out_w + oh*out_w + ow, n = c
        // NCHW output: element at (b, c + out_start, oh, ow)
        T* output_data = output.data<T>();
        #pragma omp parallel for collapse(4) if(batch * out_channels_per_group * out_h * out_w > OmpThresholds::medium())
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < out_channels_per_group; ++c) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        // Source index in GEMM output (row-major)
                        int64_t gemm_row = b * out_h * out_w + oh * out_w + ow;
                        int64_t gemm_idx = gemm_row * out_channels_per_group + c;

                        // Destination index in NCHW output
                        int64_t nchw_idx = b * (out_channels * out_h * out_w) +
                                          (c + out_start) * (out_h * out_w) +
                                          oh * out_w + ow;

                        output_data[nchw_idx] = gemm_output[gemm_idx];
                    }
                }
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        const T* bias_data = bias->data<T>();
        T* output_data = output.data<T>();
        int64_t bias_numel = batch * out_channels * out_h * out_w;

        #pragma omp parallel for collapse(4) if(bias_numel > OmpThresholds::medium())
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < out_channels; ++c) {
                for (int64_t h = 0; h < out_h; ++h) {
                    for (int64_t w = 0; w < out_w; ++w) {
                        int64_t idx = b * (out_channels * out_h * out_w) +
                                     c * (out_h * out_w) +
                                     h * out_w + w;
                        output_data[idx] += bias_data[c];
                    }
                }
            }
        }
    }
}

// Scalar-stride/pad/dilation wrapper for conv2d_forward_impl
template<typename T>
void conv2d_forward_impl(
    const Tensor& input, const Tensor& weight, const Tensor* bias, Tensor& output,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups
) {
    conv2d_forward_impl<T>(input, weight, bias, output,
                           stride, stride, padding, padding, dilation, dilation,
                           groups);
}

auto conv2d_forward_kernel(
    const Tensor& input_orig,    // (batch, in_channels, height, width)
    const Tensor& weight_orig,   // (out_channels, in_channels, kernel_h, kernel_w)
    const Tensor* bias,          // (out_channels) or nullptr
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups
) -> Tensor {
    // The impl divides out_channels/groups; a non-positive groups would SIGFPE.
    if (groups <= 0) {
        throw std::invalid_argument(
            "conv2d: groups must be positive (got " + std::to_string(groups) + ")");
    }
    // When stride/pad/dilation are isotropic we can hit the oneDNN/Winograd
    // fast paths below; otherwise fall through to the general im2col+GEMM
    // which handles per-axis values.
    bool symmetric = (stride_h == stride_w) && (pad_h == pad_w) && (dil_h == dil_w);
    int64_t stride   = stride_h;
    int64_t padding  = pad_h;
    int64_t dilation = dil_h;
    // im2col assumes contiguous row-major layout — make contiguous if needed
    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    Tensor weight = weight_orig.is_contiguous() ? weight_orig : weight_orig.contiguous();

    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t out_channels = weight_shape[0];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions — per-axis so rectangular configs
    // produce the right shape.
    int64_t out_h = calculate_output_size(height, kernel_h, stride_h, pad_h, dil_h);
    int64_t out_w = calculate_output_size(width,  kernel_w, stride_w, pad_w, dil_w);

    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument(
            "Invalid Conv2d configuration: output dimensions are non-positive (out_h=" +
            std::to_string(out_h) + ", out_w=" + std::to_string(out_w) +
            "); kernel/dilation too large for the padded input");
    }

    // Create output tensor with correct dtype
    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output = Tensor::empty_uninitialized(output_shape, input.dtype(), input.device());

#ifdef TENZOR_USE_ONEDNN
    // oneDNN supports per-axis stride/pad/dilation; the wrapper still takes
    // scalars so only the isotropic case can use it today. Route rectangular
    // configs through the im2col+GEMM path below.
    if (symmetric && conv2d_forward_onednn(input, weight, bias, output, stride, padding, dilation, groups)) {
        return output;
    }
    // Fall through to im2col+GEMM if oneDNN not applicable
#endif

    // Try Winograd for eligible 3x3 stride-1 convolutions (Float32, isotropic only)
    if (symmetric && input.dtype() == DType::Float32) {
        bool used_winograd = false;
        int64_t pad_val = static_cast<int64_t>(padding);

        if (can_use_winograd_f4x3(kernel_h, kernel_w, stride, dilation, groups, out_h, out_w)) {
            winograd_conv2d_f4x3(
                input.data<float>(), weight.data<float>(), output.data<float>(),
                batch, input_shape[1], height, width, out_channels, out_h, out_w,
                pad_val, pad_val);
            used_winograd = true;
        } else if (can_use_winograd_f2x3(kernel_h, kernel_w, stride, dilation, groups)) {
            winograd_conv2d_f2x3(
                input.data<float>(), weight.data<float>(), output.data<float>(),
                batch, input_shape[1], height, width, out_channels, out_h, out_w,
                pad_val, pad_val);
            used_winograd = true;
        }

        if (used_winograd) {
            // Add bias if present
            if (bias) {
                const float* bias_data = bias->data<float>();
                float* out_data = output.data<float>();
                int64_t spatial = out_h * out_w;
                #pragma omp parallel for collapse(3) if(batch * out_channels * spatial > OmpThresholds::medium())
                for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
                    for (int64_t oc = 0; oc < out_channels; ++oc) {
                        for (int64_t i = 0; i < spatial; ++i) {
                            out_data[(b_idx * out_channels + oc) * spatial + i] += bias_data[oc];
                        }
                    }
                }
            }
            return output;
        }
    }

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        conv2d_forward_impl<float>(input, weight, bias, output,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
    } else if (input.dtype() == DType::Float64) {
        conv2d_forward_impl<double>(input, weight, bias, output,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Half-precision on CPU: compute in Float32 for performance.
        DType original_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);

        Tensor bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &bias_f32;
        }

        Tensor output_f32(output_shape, DType::Float32, input.device());

#ifdef TENZOR_USE_ONEDNN
        if (!(symmetric && conv2d_forward_onednn(input_f32, weight_f32, bias_f32_ptr, output_f32, stride, padding, dilation, groups))) {
#endif
            conv2d_forward_impl<float>(input_f32, weight_f32, bias_f32_ptr, output_f32,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
#ifdef TENZOR_USE_ONEDNN
        }
#endif

        // Convert Float32 result back to original half-precision dtype
        const float* src = output_f32.data<float>();
        int64_t n = output.numel();
        if (original_dtype == DType::Float16) {
            Float16* dst = output.data<Float16>();
            #pragma omp parallel for if(n > OmpThresholds::simple())
            for (int64_t i = 0; i < n; ++i) {
                dst[i] = Float16(src[i]);
            }
        } else {
            BFloat16* dst = output.data<BFloat16>();
            #pragma omp parallel for if(n > OmpThresholds::simple())
            for (int64_t i = 0; i < n; ++i) {
                dst[i] = BFloat16(src[i]);
            }
        }
    } else {
        throw std::runtime_error("Unsupported dtype for conv2d_forward");
    }

    return output;
}

// Backward-compat scalar wrapper for conv2d_forward_kernel.
auto conv2d_forward_kernel(
    const Tensor& input_orig, const Tensor& weight_orig, const Tensor* bias,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups
) -> Tensor {
    return conv2d_forward_kernel(input_orig, weight_orig, bias,
                                 stride, stride, padding, padding, dilation, dilation,
                                 groups);
}

// ============================================================================
// Conv2d Backward Input CPU Implementation
// ============================================================================

// Template helper for dtype-generic conv2d backward input
template<typename T>
void conv2d_backward_input_impl(
    const Tensor& grad_output,
    const Tensor& weight,
    Tensor& grad_input,
    const std::vector<int64_t>& input_shape,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups
) {
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    int64_t out_channels_per_group = out_channels / groups;
    int64_t col_rows = batch * out_h * out_w;
    int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;

    // No explicit zero-init here: col2im_cpu is called once per (g, b) pair
    // below, and its internal memset (see col2im_cpu) exhaustively and
    // disjointly re-zeros every element of grad_input before any
    // accumulation, since groups and batches partition grad_input fully.

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Process per-batch to handle correct strides with groups
        int64_t col_per_batch = out_h * out_w * col_cols;

        // Allocate col buffer for all batches
        std::vector<T> grad_col(col_rows * col_cols);

        // Pack grad_output into GEMM A-matrix shape (M, K) = (batch*out_h*out_w,
        // out_channels_per_group). grad_output is NCHW, so its natural per-batch
        // layout is (out_channels_per_group, out_h*out_w) — the TRANSPOSE of the
        // position-major (row = b*spatial + s, col = oc) layout gemm_cpu reads.
        // A direct memcpy of the NCHW slice is only correct when
        // out_channels_per_group == 1 or out_h*out_w == 1; otherwise it produces
        // silently-wrong input gradients. Do the transpose explicitly, mirroring
        // conv2d_backward_weight_impl.
        std::vector<T> grad_out_packed(col_rows * out_channels_per_group);
        int64_t spatial = out_h * out_w;
        #pragma omp parallel for collapse(2) if(batch * out_channels_per_group > 64)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                const T* src = grad_output.data<T>() +
                               (b * out_channels + out_start + oc) * spatial;
                for (int64_t s = 0; s < spatial; ++s) {
                    grad_out_packed[(b * spatial + s) * out_channels_per_group + oc] = src[s];
                }
            }
        }

        int64_t M = col_rows;
        int64_t K = out_channels_per_group;
        int64_t N = col_cols;

        const T* weight_ptr = weight.data<T>() + out_start * in_channels_per_group * kernel_h * kernel_w;

        // Perform GEMM: C = A @ B (no transpose)
        gemm_cpu<T>(
            grad_out_packed.data(),  // A: (M, K)
            weight_ptr,              // B: (K, N) - already in correct orientation
            grad_col.data(),         // C: (M, N)
            M, N, K,
            false  // don't transpose B
        );

        // Apply col2im per-batch to accumulate gradients with correct strides
        for (int64_t b = 0; b < batch; ++b) {
            T* grad_input_ptr = grad_input.data<T>() + (b * in_channels + in_start) * height * width;
            col2im_cpu<T>(
                grad_col.data() + b * col_per_batch,
                grad_input_ptr,
                1,  // single batch element
                in_channels_per_group,
                height,
                width,
                kernel_h,
                kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dil_h, dil_w,
                out_h,
                out_w
            );
        }
    }
}

auto conv2d_backward_input_kernel(
    const Tensor& grad_output_orig,   // (batch, out_channels, out_h, out_w)
    const Tensor& weight_orig,        // (out_channels, in_channels, kernel_h, kernel_w)
    const std::vector<int64_t>& input_shape,  // (batch, in_channels, height, width)
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups
) -> Tensor {
    // The impl divides out_channels/groups; a non-positive groups would SIGFPE.
    if (groups <= 0) {
        throw std::invalid_argument(
            "conv2d_backward_input: groups must be positive (got " +
            std::to_string(groups) + ")");
    }
    // Ensure contiguous layout for pointer-arithmetic kernels
    Tensor grad_output = grad_output_orig.is_contiguous() ? grad_output_orig : grad_output_orig.contiguous();
    Tensor weight = weight_orig.is_contiguous() ? weight_orig : weight_orig.contiguous();

    // Initialize gradient w.r.t input with correct dtype. Left uninitialized
    // here because conv2d_backward_input_impl calls col2im_cpu once per
    // (g, b) pair, and col2im_cpu's own internal memset exhaustively and
    // disjointly re-zeros every element of grad_input before it scatter-adds
    // into it — that is the sole zero-init; no other memset remains.
    Tensor grad_input = Tensor::empty_uninitialized(input_shape, grad_output.dtype(), grad_output.device());

    // Dispatch based on dtype
    if (grad_output.dtype() == DType::Float32) {
        conv2d_backward_input_impl<float>(grad_output, weight, grad_input, input_shape,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        conv2d_backward_input_impl<double>(grad_output, weight, grad_input, input_shape,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Half-precision on CPU: compute in Float32 for performance
        DType original_dtype = grad_output.dtype();
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        Tensor grad_input_f32(input_shape, DType::Float32, grad_output.device());

        conv2d_backward_input_impl<float>(grad_output_f32, weight_f32, grad_input_f32, input_shape,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);

        // Convert Float32 result back to original dtype
        const float* src = grad_input_f32.data<float>();
        int64_t n = grad_input.numel();
        if (original_dtype == DType::Float16) {
            Float16* dst = grad_input.data<Float16>();
            #pragma omp parallel for if(n > OmpThresholds::simple())
            for (int64_t i = 0; i < n; ++i) dst[i] = Float16(src[i]);
        } else {
            BFloat16* dst = grad_input.data<BFloat16>();
            #pragma omp parallel for if(n > OmpThresholds::simple())
            for (int64_t i = 0; i < n; ++i) dst[i] = BFloat16(src[i]);
        }
    } else {
        throw std::runtime_error("Unsupported dtype for conv2d_backward_input");
    }

    return grad_input;
}

// Scalar backward-compat wrapper.
auto conv2d_backward_input_kernel(
    const Tensor& grad_output_orig, const Tensor& weight_orig,
    const std::vector<int64_t>& input_shape,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups
) -> Tensor {
    return conv2d_backward_input_kernel(grad_output_orig, weight_orig, input_shape,
        stride, stride, padding, padding, dilation, dilation, groups);
}

// ============================================================================
// Conv2d Backward Weight CPU Implementation
// ============================================================================

// Template helper for dtype-generic conv2d backward weight
template<typename T>
void conv2d_backward_weight_impl(
    const Tensor& grad_output,
    const Tensor& input,
    Tensor& grad_weight,
    const std::vector<int64_t>& weight_shape,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups
) {
    auto input_shape = input.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    int64_t out_channels_per_group = out_channels / groups;
    int64_t col_rows = batch * out_h * out_w;
    int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Apply im2col per-batch to handle correct strides with groups
        std::vector<T> input_col(col_rows * col_cols);
        int64_t col_per_batch = out_h * out_w * col_cols;

        for (int64_t b = 0; b < batch; ++b) {
            const T* input_ptr = input.data<T>() + (b * in_channels + in_start) * height * width;
            im2col_cpu<T>(
                input_ptr,
                input_col.data() + b * col_per_batch,
                1,  // single batch element
                in_channels_per_group,
                height,
                width,
                kernel_h,
                kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dil_h, dil_w,
                out_h,
                out_w
            );
        }

        int64_t M = out_channels_per_group;
        int64_t K = col_rows;
        int64_t N = col_cols;

        // Pack grad_output into GEMM A-matrix shape (K, M) = (out_h*out_w per
        // batch concatenated, out_channels_per_group). grad_output is NCHW —
        // its natural layout per-batch is (out_channels_per_group, out_h*out_w),
        // which is the TRANSPOSE of what gemm_transA_cpu expects. The prior
        // implementation memcpy'd the NCHW slice directly into grad_out_packed,
        // producing silently-wrong weight gradients (~20% off on non-trivial
        // inputs). Do the transpose explicitly.
        std::vector<T> grad_out_packed(col_rows * out_channels_per_group);
        int64_t spatial = out_h * out_w;
        #pragma omp parallel for collapse(2) if(batch * out_channels_per_group > 64)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                const T* src = grad_output.data<T>() +
                               (b * out_channels + out_start + oc) * spatial;
                // Destination: row index k = b*spatial + s, column = oc
                //   grad_out_packed[k * M + oc]
                for (int64_t s = 0; s < spatial; ++s) {
                    grad_out_packed[(b * spatial + s) * out_channels_per_group + oc] = src[s];
                }
            }
        }

        const T* grad_out_ptr = grad_out_packed.data();
        T* grad_weight_ptr = grad_weight.data<T>() + out_start * in_channels_per_group * kernel_h * kernel_w;

        // Perform GEMM: C = A^T @ B
        gemm_transA_cpu<T>(
            grad_out_ptr,       // A: (K, M) - will be transposed
            input_col.data(),   // B: (K, N)
            grad_weight_ptr,    // C: (M, N)
            M, N, K
        );
    }
}

auto conv2d_backward_weight_kernel(
    const Tensor& grad_output_orig,   // (batch, out_channels, out_h, out_w)
    const Tensor& input_orig,         // (batch, in_channels, height, width)
    const std::vector<int64_t>& weight_shape,  // (out_channels, in_channels, kernel_h, kernel_w)
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups
) -> Tensor {
    // The impl divides out_channels/groups; a non-positive groups would SIGFPE.
    if (groups <= 0) {
        throw std::invalid_argument(
            "conv2d_backward_weight: groups must be positive (got " +
            std::to_string(groups) + ")");
    }
    // Ensure contiguous layout for pointer-arithmetic kernels
    Tensor grad_output = grad_output_orig.is_contiguous() ? grad_output_orig : grad_output_orig.contiguous();
    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();

    // Initialize gradient w.r.t weight
    Tensor grad_weight = Tensor::empty_uninitialized(weight_shape, grad_output.dtype(), grad_output.device());

    // Dispatch based on dtype
    if (grad_output.dtype() == DType::Float32) {
        conv2d_backward_weight_impl<float>(grad_output, input, grad_weight, weight_shape,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        conv2d_backward_weight_impl<double>(grad_output, input, grad_weight, weight_shape,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Half-precision on CPU: compute in Float32 for performance
        DType original_dtype = grad_output.dtype();
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        Tensor grad_weight_f32 = Tensor::empty_uninitialized(weight_shape, DType::Float32, grad_output.device());

        conv2d_backward_weight_impl<float>(grad_output_f32, input_f32, grad_weight_f32, weight_shape,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);

        // Convert Float32 result back to original dtype
        const float* src = grad_weight_f32.data<float>();
        int64_t n = grad_weight.numel();
        if (original_dtype == DType::Float16) {
            Float16* dst = grad_weight.data<Float16>();
            #pragma omp parallel for if(n > OmpThresholds::simple())
            for (int64_t i = 0; i < n; ++i) dst[i] = Float16(src[i]);
        } else {
            BFloat16* dst = grad_weight.data<BFloat16>();
            #pragma omp parallel for if(n > OmpThresholds::simple())
            for (int64_t i = 0; i < n; ++i) dst[i] = BFloat16(src[i]);
        }
    } else {
        throw std::runtime_error("Unsupported dtype for conv2d_backward_weight");
    }

    return grad_weight;
}

// Scalar backward-compat wrapper.
auto conv2d_backward_weight_kernel(
    const Tensor& grad_output_orig, const Tensor& input_orig,
    const std::vector<int64_t>& weight_shape,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups
) -> Tensor {
    return conv2d_backward_weight_kernel(grad_output_orig, input_orig, weight_shape,
        stride, stride, padding, padding, dilation, dilation, groups);
}

// ============================================================================
// Conv2d Backward Bias CPU Implementation
// ============================================================================

// Template helper for dtype-generic conv2d backward bias
template<typename T>
void conv2d_backward_bias_impl(
    const Tensor& grad_output,
    Tensor& grad_bias
) {
    auto grad_shape = grad_output.shape();
    int64_t batch = grad_shape[0];
    int64_t out_channels = grad_shape[1];
    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    T* grad_bias_data = grad_bias.data<T>();

    const T* grad_out_data = grad_output.data<T>();

    // Sum over batch, height, width dimensions
    #pragma omp parallel for if(out_channels * batch * out_h * out_w > OmpThresholds::complex())
    for (int64_t c = 0; c < out_channels; ++c) {
        T sum = T(0.0f);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < out_h; ++h) {
                for (int64_t w = 0; w < out_w; ++w) {
                    int64_t idx = b * (out_channels * out_h * out_w) +
                                 c * (out_h * out_w) +
                                 h * out_w + w;
                    sum += grad_out_data[idx];
                }
            }
        }
        grad_bias_data[c] = sum;
    }
}

auto conv2d_backward_bias_kernel(
    const Tensor& grad_output    // (batch, out_channels, out_h, out_w)
) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t out_channels = grad_shape[1];

    // Initialize gradient w.r.t bias
    Tensor grad_bias = Tensor::empty_uninitialized({out_channels}, grad_output.dtype(), grad_output.device());

    // Dispatch based on dtype
    if (grad_output.dtype() == DType::Float32) {
        conv2d_backward_bias_impl<float>(grad_output, grad_bias);
    } else if (grad_output.dtype() == DType::Float64) {
        conv2d_backward_bias_impl<double>(grad_output, grad_bias);
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Half-precision on CPU: compute in Float32 for performance
        DType original_dtype = grad_output.dtype();
        auto grad_output_f32 = grad_output.to(DType::Float32);
        int64_t out_channels = grad_output.shape()[1];
        Tensor grad_bias_f32 = Tensor::empty_uninitialized({out_channels}, DType::Float32, grad_output.device());

        conv2d_backward_bias_impl<float>(grad_output_f32, grad_bias_f32);

        // Convert Float32 result back to original dtype
        const float* src = grad_bias_f32.data<float>();
        if (original_dtype == DType::Float16) {
            Float16* dst = grad_bias.data<Float16>();
            for (int64_t i = 0; i < out_channels; ++i) dst[i] = Float16(src[i]);
        } else {
            BFloat16* dst = grad_bias.data<BFloat16>();
            for (int64_t i = 0; i < out_channels; ++i) dst[i] = BFloat16(src[i]);
        }
    } else {
        throw std::runtime_error("Unsupported dtype for conv2d_backward_bias");
    }

    return grad_bias;
}

// ============================================================================
// ConvTranspose2d Forward CPU Implementation
// ============================================================================

// Calculate output size for transposed convolution
inline int64_t calculate_transpose_output_size(
    int64_t input_size, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation
) {
    return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
}

// Template helper for dtype-generic conv_transpose2d forward using gather approach
// Gather approach iterates over output positions and gathers contributions from input
// This is parallelizable since each output position is independent
template<typename T>
void conv_transpose2d_forward_impl(
    const Tensor& input,          // (batch, in_channels, in_h, in_w)
    const Tensor& weight,         // (in_channels, out_channels/groups, kernel_h, kernel_w)
    const Tensor* bias,           // (out_channels) or nullptr
    Tensor& output,               // (batch, out_channels, out_h, out_w)
    // Audit I5: per-axis stride/padding/output_padding/dilation.
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    [[maybe_unused]] int64_t opH, [[maybe_unused]] int64_t opW,
    int64_t dH, int64_t dW,
    int64_t groups
) {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto output_shape = output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int64_t in_channels_per_group = weight_shape[0] / groups;
    int64_t out_channels_per_group = weight_shape[1];
    int64_t out_channels = out_channels_per_group * groups;
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = output_shape[2];
    int64_t out_w = output_shape[3];

    const T* input_data = input.data<T>();
    const T* weight_data = weight.data<T>();
    T* output_data = output.data<T>();

    // Use gather approach: iterate over output positions and gather from input
    // This is parallelizable since each output position is independent
    int64_t total_output = batch * out_channels * out_h * out_w;

    #pragma omp parallel for if(total_output > OmpThresholds::complex())
    for (int64_t idx = 0; idx < total_output; ++idx) {
        // Decode output position
        int64_t w = idx % out_w;
        int64_t h = (idx / out_w) % out_h;
        int64_t c = (idx / (out_w * out_h)) % out_channels;
        int64_t b = idx / (out_w * out_h * out_channels);

        // Determine group
        int64_t g = c / out_channels_per_group;
        int64_t oc = c % out_channels_per_group;  // Output channel within group
        int64_t in_start = g * in_channels_per_group;

        // Accumulator type: double for T=double (preserve Float64 precision),
        // float otherwise (Float32 path + Float16/BFloat16 with float accum).
        using AccumT = std::conditional_t<std::is_same_v<T, double>, double, float>;
        AccumT sum = AccumT(0);

        // Gather from all input positions that contribute to this output position
        for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    // For transposed conv: out = (in - 1) * stride - 2*padding + dilation * (kernel - 1) + out_padding + 1
                    // Inverse: which input position (ih, iw) with kernel (kh, kw) contributes to (h, w)?
                    // oh = ih * stride - padding + kh * dilation
                    // ih * stride = oh + padding - kh * dilation
                    // ih = (oh + padding - kh * dilation) / stride (must be integer and in bounds)

                    int64_t h_shifted = h + pH - kh * dH;
                    int64_t w_shifted = w + pW - kw * dW;

                    // Check if this maps to a valid input position
                    if (h_shifted >= 0 && h_shifted % sH == 0 &&
                        w_shifted >= 0 && w_shifted % sW == 0) {

                        int64_t ih = h_shifted / sH;
                        int64_t iw = w_shifted / sW;

                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            // Get input value
                            int64_t input_idx = b * (in_channels * in_h * in_w) +
                                               (in_start + ic) * (in_h * in_w) +
                                               ih * in_w + iw;
                            AccumT input_val = static_cast<AccumT>(input_data[input_idx]);

                            // Get weight value
                            // Weight shape: (in_channels, out_channels/groups, kernel_h, kernel_w)
                            int64_t weight_idx = (in_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                                oc * (kernel_h * kernel_w) +
                                                kh * kernel_w + kw;
                            AccumT weight_val = static_cast<AccumT>(weight_data[weight_idx]);

                            sum += input_val * weight_val;
                        }
                    }
                }
            }
        }

        // Add bias if present
        if (bias != nullptr) {
            sum += static_cast<AccumT>(bias->data<T>()[c]);
        }

        output_data[idx] = T(sum);
    }
}

// Audit I5: per-axis public kernel (primary). Scalar overload (below) delegates.
auto conv_transpose2d_forward_kernel(
    const Tensor& input_orig,
    const Tensor& weight_orig,
    const Tensor* bias,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t opH, int64_t opW,
    int64_t dH, int64_t dW,
    int64_t groups
) -> Tensor {
    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    Tensor weight = weight_orig.is_contiguous() ? weight_orig : weight_orig.contiguous();

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int64_t out_channels_per_group = weight_shape[1];
    int64_t out_channels = out_channels_per_group * groups;
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = calculate_transpose_output_size(in_h, kernel_h, sH, pH, opH, dH);
    int64_t out_w = calculate_transpose_output_size(in_w, kernel_w, sW, pW, opW, dW);

    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument(
            "Invalid ConvTranspose2d configuration: output dimensions are non-positive (out_h=" +
            std::to_string(out_h) + ", out_w=" + std::to_string(out_w) + ")"
        );
    }

    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        conv_transpose2d_forward_impl<float>(input, weight, bias, output, sH, sW, pH, pW, opH, opW, dH, dW, groups);
    } else if (input.dtype() == DType::Float64) {
        conv_transpose2d_forward_impl<double>(input, weight, bias, output, sH, sW, pH, pW, opH, opW, dH, dW, groups);
    } else if (input.dtype() == DType::Float16) {
        conv_transpose2d_forward_impl<Float16>(input, weight, bias, output, sH, sW, pH, pW, opH, opW, dH, dW, groups);
    } else if (input.dtype() == DType::BFloat16) {
        conv_transpose2d_forward_impl<BFloat16>(input, weight, bias, output, sH, sW, pH, pW, opH, opW, dH, dW, groups);
    } else {
        throw std::runtime_error("Unsupported dtype for conv_transpose2d_forward");
    }

    return output;
}

// Scalar overload — delegates to per-axis with identical H/W values.
auto conv_transpose2d_forward_kernel(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    int64_t stride, int64_t padding, int64_t output_padding,
    int64_t dilation, int64_t groups
) -> Tensor {
    return conv_transpose2d_forward_kernel(input, weight, bias,
        stride, stride, padding, padding, output_padding, output_padding,
        dilation, dilation, groups);
}

// ============================================================================
// Depthwise Conv2d Forward CPU Implementation
// ============================================================================

template<typename T>
void depthwise_conv2d_impl(const T* in_data, const T* w_data, const T* b_data, T* out_data,
                            int64_t N, int64_t C, int64_t H, int64_t W,
                            int64_t kH, int64_t kW, int64_t H_out, int64_t W_out,
                            int64_t stride_h, int64_t stride_w,
                            int64_t padding_h, int64_t padding_w,
                            int64_t dilation_h, int64_t dilation_w) {
    // Accumulate in double for Float64 inputs to preserve full precision;
    // otherwise accumulate in float (fine for Float32 and promotes half types).
    using AccumT = std::conditional_t<std::is_same_v<T, double>, double, float>;
    int64_t depthwise_numel = N * C * H_out * W_out;
    #pragma omp parallel for collapse(4) if(depthwise_numel > OmpThresholds::medium())
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    AccumT sum = AccumT(0);
                    AccumT compensation = AccumT(0);  // Kahan compensation

                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t h = oh * stride_h - padding_h + kh * dilation_h;
                            int64_t w = ow * stride_w - padding_w + kw * dilation_w;

                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                AccumT product = static_cast<AccumT>(in_data[((n * C + c) * H + h) * W + w]) *
                                       static_cast<AccumT>(w_data[(c * 1 + 0) * kH * kW + kh * kW + kw]);
                                AccumT y = product - compensation;
                                AccumT t = sum + y;
                                compensation = (t - sum) - y;
                                sum = t;
                            }
                        }
                    }

                    if (b_data) {
                        AccumT y = static_cast<AccumT>(b_data[c]) - compensation;
                        sum = sum + y;
                    }

                    out_data[((n * C + c) * H_out + oh) * W_out + ow] = static_cast<T>(sum);
                }
            }
        }
    }
}

// ============================================================================
// AVX2-optimized Depthwise Conv2d for Float32 (stride=1, dilation=1)
// ============================================================================
// Vectorizes across the output width dimension using 8-wide SIMD.
// For stride=1 and dilation=1, consecutive output positions read from
// consecutive input positions, enabling efficient vectorized loads.

#ifdef TENZOR_CONV_AVX2
void depthwise_conv2d_avx2_f32(
    const float* __restrict in_data,
    const float* __restrict w_data,
    const float* __restrict b_data,
    float* __restrict out_data,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kH, int64_t kW, int64_t H_out, int64_t W_out,
    int64_t padding_h, int64_t padding_w) {
    // stride=1, dilation=1 (both axes) is assumed by the caller — the per-axis
    // padding terms below are independent.

    #pragma omp parallel for collapse(3) if(N * C * H_out > OmpThresholds::medium())
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                const float* filter = w_data + c * kH * kW;
                float* out_row = out_data + ((n * C + c) * H_out + oh) * W_out;

                // Vectorized path: process 8 output columns at a time
                int64_t ow = 0;
                for (; ow + 8 <= W_out; ow += 8) {
                    __m256 v_sum = _mm256_setzero_ps();

                    for (int64_t kh = 0; kh < kH; ++kh) {
                        int64_t ih = oh - padding_h + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* in_row = in_data + ((n * C + c) * H + ih) * W;

                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t iw_start = ow - padding_w + kw;

                            // Load filter weight and broadcast to all 8 lanes
                            __m256 v_w = _mm256_set1_ps(filter[kh * kW + kw]);

                            // Check if all 8 input positions are in bounds
                            if (iw_start >= 0 && iw_start + 8 <= W) {
                                // Fast path: all 8 positions in bounds
                                __m256 v_in = _mm256_loadu_ps(in_row + iw_start);
                                v_sum = _mm256_fmadd_ps(v_in, v_w, v_sum);
                            } else {
                                // Slow path: handle boundaries element-by-element
                                alignas(32) float tmp[8];
                                for (int i = 0; i < 8; ++i) {
                                    int64_t iw = iw_start + i;
                                    tmp[i] = (iw >= 0 && iw < W) ? in_row[iw] : 0.0f;
                                }
                                __m256 v_in = _mm256_load_ps(tmp);
                                v_sum = _mm256_fmadd_ps(v_in, v_w, v_sum);
                            }
                        }
                    }

                    // Add bias if present
                    if (b_data) {
                        __m256 v_bias = _mm256_set1_ps(b_data[c]);
                        v_sum = _mm256_add_ps(v_sum, v_bias);
                    }

                    _mm256_storeu_ps(out_row + ow, v_sum);
                }

                // Scalar tail for remaining columns
                for (; ow < W_out; ++ow) {
                    float sum = 0.0f;

                    for (int64_t kh = 0; kh < kH; ++kh) {
                        int64_t ih = oh - padding_h + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* in_row = in_data + ((n * C + c) * H + ih) * W;

                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t iw = ow - padding_w + kw;
                            if (iw >= 0 && iw < W) {
                                sum += in_row[iw] * filter[kh * kW + kw];
                            }
                        }
                    }

                    if (b_data) sum += b_data[c];
                    out_row[ow] = sum;
                }
            }
        }
    }
}
#endif // TENZOR_CONV_AVX2

auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight,
                              const Tensor* bias,
                              int64_t stride_h, int64_t stride_w,
                              int64_t padding_h, int64_t padding_w,
                              int64_t dilation_h, int64_t dilation_w) -> Tensor {
    // Depthwise convolution: groups = in_channels = out_channels
    // input: [N, C, H, W]
    // weight: [C, 1, kH, kW]

    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    int64_t N = in_shape[0];
    int64_t C = in_shape[1];
    int64_t H = in_shape[2];
    int64_t W = in_shape[3];
    int64_t kH = w_shape[2];
    int64_t kW = w_shape[3];

    int64_t H_out = (H + 2 * padding_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int64_t W_out = (W + 2 * padding_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    // Reject a kernel/dilation that does not fit the padded input before
    // allocating (a non-positive extent would build a negative-dim tensor).
    // Mirrors conv2d_forward / depthwise_conv1d / depthwise_conv3d.
    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument(
            "depthwise_conv2d: non-positive output dimensions (H_out=" +
            std::to_string(H_out) + ", W_out=" + std::to_string(W_out) +
            "); kernel/dilation too large for the padded input");
    }

    // data<T>() ignores strides; the impl/AVX2 paths walk input/weight as flat
    // packed NCHW / [C,1,kH,kW] buffers. A non-contiguous view (channels-last,
    // permuted, sliced-channel subview) would be read in the wrong order and
    // silently produce wrong output. Materialize contiguous copies once
    // (no-op when already packed), mirroring depthwise_conv1d/3d.
    const Tensor in_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor w_c  = weight.is_contiguous() ? weight : weight.contiguous();
    Tensor bias_c_storage;
    const Tensor* bias_c = bias;
    if (bias && !bias->is_contiguous()) {
        bias_c_storage = bias->contiguous();
        bias_c = &bias_c_storage;
    }

    auto output = Tensor::empty_uninitialized({N, C, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
#ifdef TENZOR_CONV_AVX2
        // AVX2 fast path requires stride==1 and dilation==1 on BOTH axes
        // (it walks across the W axis with vectorised loads). Padding may
        // differ per axis without changing the access pattern.
        if (stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
            depthwise_conv2d_avx2_f32(in_c.data<float>(), w_c.data<float>(),
                bias_c ? bias_c->data<float>() : nullptr, output.data<float>(),
                N, C, H, W, kH, kW, H_out, W_out, padding_h, padding_w);
        } else
#endif
        {
            depthwise_conv2d_impl<float>(in_c.data<float>(), w_c.data<float>(),
                bias_c ? bias_c->data<float>() : nullptr, output.data<float>(),
                N, C, H, W, kH, kW, H_out, W_out,
                stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
        }
    } else if (input.dtype() == DType::Float64) {
        depthwise_conv2d_impl<double>(in_c.data<double>(), w_c.data<double>(),
            bias_c ? bias_c->data<double>() : nullptr, output.data<double>(),
            N, C, H, W, kH, kW, H_out, W_out,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
    } else if (input.dtype() == DType::Float16) {
        depthwise_conv2d_impl<Float16>(in_c.data<Float16>(), w_c.data<Float16>(),
            bias_c ? bias_c->data<Float16>() : nullptr, output.data<Float16>(),
            N, C, H, W, kH, kW, H_out, W_out,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
    } else if (input.dtype() == DType::BFloat16) {
        depthwise_conv2d_impl<BFloat16>(in_c.data<BFloat16>(), w_c.data<BFloat16>(),
            bias_c ? bias_c->data<BFloat16>() : nullptr, output.data<BFloat16>(),
            N, C, H, W, kH, kW, H_out, W_out,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
    } else {
        throw std::runtime_error("Unsupported dtype for depthwise_conv2d");
    }

    return output;
}

// Backward-compatible scalar overload — replicates scalars onto both axes.
auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight,
                              const Tensor* bias, int64_t stride,
                              int64_t padding, int64_t dilation) -> Tensor {
    return depthwise_conv2d_kernel(input, weight, bias,
                                   stride, stride, padding, padding, dilation, dilation);
}

// ============================================================================
// Deformable Conv2d (DCNv2) kernels
// ============================================================================

namespace {

/// Validate group parameters before any C_in/groups, C_out/groups or
/// C_in/offset_groups division (which would SIGFPE on a zero divisor). The
/// deformable-conv op-layer only checks ndim==4, so this is the sole guard.
inline void validate_deformable_groups(int64_t C_in, int64_t C_out,
                                       int64_t groups, int64_t offset_groups) {
    if (groups <= 0 || offset_groups <= 0) {
        throw std::invalid_argument(
            "deformable_conv2d: groups and offset_groups must be positive (groups=" +
            std::to_string(groups) + ", offset_groups=" +
            std::to_string(offset_groups) + ")");
    }
    if (C_in % groups != 0 || C_out % groups != 0 || C_in % offset_groups != 0) {
        throw std::invalid_argument(
            "deformable_conv2d: in_channels and out_channels must be divisible by "
            "groups, and in_channels by offset_groups (C_in=" +
            std::to_string(C_in) + ", C_out=" + std::to_string(C_out) +
            ", groups=" + std::to_string(groups) + ", offset_groups=" +
            std::to_string(offset_groups) + ")");
    }
}

/// Accumulator type for deformable-conv interpolation: double for the Float64
/// path so finite-difference gradcheck precision is preserved, float otherwise.
/// Mirrors the AccumT used by the forward/weight-gradient impls.
template <typename T>
using DeformAccT = std::conditional_t<std::is_same_v<T, double>, double, float>;

/// Bilinear interpolation at fractional position (h, w) in a single channel plane.
/// Returns 0 for out-of-bounds positions. Interpolation arithmetic is performed
/// in AccT (double for the Float64 path) to avoid narrowing the result to float.
template <typename T>
inline DeformAccT<T> bilinear_interpolate(const T* data, int64_t H, int64_t W,
                                          float h, float w) {
    using AccT = DeformAccT<T>;
    if (h <= -1 || h >= H || w <= -1 || w >= W) return AccT(0);

    int64_t h_low = static_cast<int64_t>(std::floor(h));
    int64_t w_low = static_cast<int64_t>(std::floor(w));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    AccT lh = static_cast<AccT>(h) - static_cast<AccT>(h_low);
    AccT lw = static_cast<AccT>(w) - static_cast<AccT>(w_low);
    AccT hh = AccT(1) - lh;
    AccT hw = AccT(1) - lw;

    auto val = [&](int64_t r, int64_t c) -> AccT {
        if (r < 0 || r >= H || c < 0 || c >= W) return AccT(0);
        return static_cast<AccT>(data[r * W + c]);
    };

    AccT v = hh * hw * val(h_low, w_low) +
             hh * lw * val(h_low, w_high) +
             lh * hw * val(h_high, w_low) +
             lh * lw * val(h_high, w_high);
    return v;
}

/// Scatter gradient back through bilinear interpolation (used in backward).
template <typename T>
inline void bilinear_interpolate_gradient(T* grad_data, int64_t H, int64_t W,
                                           float h, float w, DeformAccT<T> top_grad) {
    using AccT = DeformAccT<T>;
    if (h <= -1 || h >= H || w <= -1 || w >= W) return;

    int64_t h_low = static_cast<int64_t>(std::floor(h));
    int64_t w_low = static_cast<int64_t>(std::floor(w));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    AccT lh = static_cast<AccT>(h) - static_cast<AccT>(h_low);
    AccT lw = static_cast<AccT>(w) - static_cast<AccT>(w_low);
    AccT hh = AccT(1) - lh;
    AccT hw = AccT(1) - lw;

    auto add = [&](int64_t r, int64_t c, AccT weight) {
        if (r >= 0 && r < H && c >= 0 && c < W) {
            // Use atomic for thread safety under OpenMP
            #pragma omp atomic
            grad_data[r * W + c] += static_cast<T>(weight * top_grad);
        }
    };

    add(h_low, w_low, hh * hw);
    add(h_low, w_high, hh * lw);
    add(h_high, w_low, lh * hw);
    add(h_high, w_high, lh * lw);
}

/// Compute dval/dh and dval/dw for bilinear interpolation — needed for offset gradients.
template <typename T>
inline void bilinear_interpolate_offset_gradient(
    const T* data, int64_t H, int64_t W, float h, float w,
    DeformAccT<T>& grad_h, DeformAccT<T>& grad_w) {
    using AccT = DeformAccT<T>;
    grad_h = AccT(0);
    grad_w = AccT(0);
    if (h <= -1 || h >= H || w <= -1 || w >= W) return;

    int64_t h_low = static_cast<int64_t>(std::floor(h));
    int64_t w_low = static_cast<int64_t>(std::floor(w));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    AccT lh = static_cast<AccT>(h) - static_cast<AccT>(h_low);
    AccT lw = static_cast<AccT>(w) - static_cast<AccT>(w_low);
    AccT hh = AccT(1) - lh;
    AccT hw = AccT(1) - lw;

    auto val = [&](int64_t r, int64_t c) -> AccT {
        if (r < 0 || r >= H || c < 0 || c >= W) return AccT(0);
        return static_cast<AccT>(data[r * W + c]);
    };

    AccT v00 = val(h_low, w_low);
    AccT v01 = val(h_low, w_high);
    AccT v10 = val(h_high, w_low);
    AccT v11 = val(h_high, w_high);

    // d(bilinear)/dh = (-hw * v00) + (-lw * v01) + (hw * v10) + (lw * v11)
    grad_h = -hw * v00 - lw * v01 + hw * v10 + lw * v11;
    // d(bilinear)/dw = (-hh * v00) + (hh * v01) + (-lh * v10) + (lh * v11)
    grad_w = -hh * v00 + hh * v01 - lh * v10 + lh * v11;
}

} // anonymous namespace

template <typename T>
static void deformable_conv2d_forward_impl(
    const T* input, const T* offset, const T* weight,
    const T* bias, const T* mask,
    T* output,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t kH, int64_t kW,
    int64_t H_out, int64_t W_out,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool use_mask) {

    int64_t channels_per_group = C_in / groups;
    int64_t out_channels_per_group = C_out / groups;
    int64_t channels_per_offset_group = C_in / offset_groups;

    #pragma omp parallel for collapse(3) schedule(static) if(N * C_out * H_out * W_out > ::tenzor::OmpThresholds::simple())
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < C_out; ++oc) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                int64_t g = oc / out_channels_per_group;

                for (int64_t ow = 0; ow < W_out; ++ow) {
                    // Accumulate in double for Float64 inputs (preserves
                    // gradient precision under finite-difference gradcheck);
                    // float otherwise. Position arithmetic stays float —
                    // subpixel offsets don't need double.
                    using AccumT = std::conditional_t<std::is_same_v<T, double>, double, float>;
                    AccumT sum = AccumT(0);

                    for (int64_t ic_local = 0; ic_local < channels_per_group; ++ic_local) {
                        int64_t ic = g * channels_per_group + ic_local;
                        int64_t og = ic / channels_per_offset_group; // offset group for this channel

                        const T* input_plane = input + (n * C_in + ic) * H * W;
                        const T* weight_plane = weight + (oc * channels_per_group + ic_local) * kH * kW;

                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t offset_idx = og * 2 * kH * kW;
                                int64_t mask_idx = og * kH * kW;
                                int64_t k_linear = kh * kW + kw;

                                // Deformed position
                                float h_base = oh * stride_h - pad_h + kh * dil_h;
                                float w_base = ow * stride_w - pad_w + kw * dil_w;

                                const T* off_h_plane = offset + (n * offset_groups * 2 * kH * kW +
                                    (offset_idx + 2 * k_linear)) * H_out * W_out;
                                const T* off_w_plane = offset + (n * offset_groups * 2 * kH * kW +
                                    (offset_idx + 2 * k_linear + 1)) * H_out * W_out;

                                float h_off = static_cast<float>(off_h_plane[oh * W_out + ow]);
                                float w_off = static_cast<float>(off_w_plane[oh * W_out + ow]);

                                float h_loc = h_base + h_off;
                                float w_loc = w_base + w_off;

                                AccumT val = static_cast<AccumT>(
                                    bilinear_interpolate(input_plane, H, W, h_loc, w_loc));

                                if (use_mask) {
                                    const T* mask_plane = mask + (n * offset_groups * kH * kW +
                                        (mask_idx + k_linear)) * H_out * W_out;
                                    val *= static_cast<AccumT>(mask_plane[oh * W_out + ow]);
                                }

                                sum += val * static_cast<AccumT>(weight_plane[k_linear]);
                            }
                        }
                    }

                    if (bias) {
                        sum += static_cast<AccumT>(bias[oc]);
                    }

                    output[(n * C_out + oc) * H_out * W_out + oh * W_out + ow] = static_cast<T>(sum);
                }
            }
        }
    }
}

template <typename T>
static void deformable_conv2d_backward_input_impl(
    const T* grad_output, const T* input, const T* offset,
    const T* weight, const T* mask,
    T* grad_input, T* grad_offset, T* grad_mask,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t kH, int64_t kW,
    int64_t H_out, int64_t W_out,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool use_mask) {

    int64_t channels_per_group = C_in / groups;
    int64_t out_channels_per_group = C_out / groups;
    int64_t channels_per_offset_group = C_in / offset_groups;

    #pragma omp parallel for collapse(2) schedule(static) if(N * C_out > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < C_out; ++oc) {
            int64_t g = oc / out_channels_per_group;

            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    // Accumulate gradients in double for the Float64 path so a
                    // tight finite-difference gradcheck on offset/mask passes;
                    // float otherwise. Matches the forward/weight-grad impls.
                    using AccumT = std::conditional_t<std::is_same_v<T, double>, double, float>;
                    AccumT grad_out_val = static_cast<AccumT>(
                        grad_output[(n * C_out + oc) * H_out * W_out + oh * W_out + ow]);

                    for (int64_t ic_local = 0; ic_local < channels_per_group; ++ic_local) {
                        int64_t ic = g * channels_per_group + ic_local;
                        int64_t og = ic / channels_per_offset_group;

                        const T* input_plane = input + (n * C_in + ic) * H * W;
                        T* grad_input_plane = grad_input + (n * C_in + ic) * H * W;
                        const T* weight_plane = weight + (oc * channels_per_group + ic_local) * kH * kW;

                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t offset_idx = og * 2 * kH * kW;
                                int64_t mask_idx = og * kH * kW;
                                int64_t k_linear = kh * kW + kw;

                                float h_base = oh * stride_h - pad_h + kh * dil_h;
                                float w_base = ow * stride_w - pad_w + kw * dil_w;

                                int64_t off_h_idx = (offset_idx + 2 * k_linear);
                                int64_t off_w_idx = (offset_idx + 2 * k_linear + 1);

                                const T* off_h_plane = offset + (n * offset_groups * 2 * kH * kW + off_h_idx) * H_out * W_out;
                                const T* off_w_plane = offset + (n * offset_groups * 2 * kH * kW + off_w_idx) * H_out * W_out;

                                float h_off = static_cast<float>(off_h_plane[oh * W_out + ow]);
                                float w_off = static_cast<float>(off_w_plane[oh * W_out + ow]);

                                float h_loc = h_base + h_off;
                                float w_loc = w_base + w_off;

                                AccumT w_val = static_cast<AccumT>(weight_plane[k_linear]);
                                AccumT m_val = AccumT(1);
                                if (use_mask) {
                                    const T* mask_plane = mask + (n * offset_groups * kH * kW + (mask_idx + k_linear)) * H_out * W_out;
                                    m_val = static_cast<AccumT>(mask_plane[oh * W_out + ow]);
                                }

                                AccumT top_grad = grad_out_val * w_val * m_val;

                                // grad_input
                                bilinear_interpolate_gradient(grad_input_plane, H, W, h_loc, w_loc, top_grad);

                                // grad_offset (d/dh, d/dw of bilinear)
                                AccumT grad_h, grad_w;
                                bilinear_interpolate_offset_gradient(input_plane, H, W, h_loc, w_loc, grad_h, grad_w);

                                T* grad_off_h = grad_offset + (n * offset_groups * 2 * kH * kW + off_h_idx) * H_out * W_out;
                                T* grad_off_w = grad_offset + (n * offset_groups * 2 * kH * kW + off_w_idx) * H_out * W_out;

                                AccumT off_grad_h = grad_out_val * w_val * m_val * grad_h;
                                AccumT off_grad_w = grad_out_val * w_val * m_val * grad_w;

                                #pragma omp atomic
                                grad_off_h[oh * W_out + ow] += static_cast<T>(off_grad_h);
                                #pragma omp atomic
                                grad_off_w[oh * W_out + ow] += static_cast<T>(off_grad_w);

                                // grad_mask
                                if (use_mask && grad_mask) {
                                    AccumT interp_val = static_cast<AccumT>(
                                        bilinear_interpolate(input_plane, H, W, h_loc, w_loc));
                                    T* grad_mask_plane = grad_mask + (n * offset_groups * kH * kW + (mask_idx + k_linear)) * H_out * W_out;
                                    AccumT mask_grad = grad_out_val * w_val * interp_val;
                                    #pragma omp atomic
                                    grad_mask_plane[oh * W_out + ow] += static_cast<T>(mask_grad);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
static void deformable_conv2d_backward_weight_impl(
    const T* grad_output, const T* input, const T* offset, const T* mask,
    T* grad_weight,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t kH, int64_t kW,
    int64_t H_out, int64_t W_out,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool use_mask) {

    int64_t channels_per_group = C_in / groups;
    int64_t out_channels_per_group = C_out / groups;
    int64_t channels_per_offset_group = C_in / offset_groups;

    #pragma omp parallel for collapse(2) schedule(static) if(C_out * channels_per_group > 16)
    for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t ic_local = 0; ic_local < channels_per_group; ++ic_local) {
            int64_t g = oc / out_channels_per_group;
            int64_t ic = g * channels_per_group + ic_local;
            int64_t og = ic / channels_per_offset_group;

            T* gw = grad_weight + (oc * channels_per_group + ic_local) * kH * kW;

            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t k_linear = kh * kW + kw;
                    int64_t offset_idx = og * 2 * kH * kW;
                    int64_t mask_idx = og * kH * kW;
                    using AccumT = std::conditional_t<std::is_same_v<T, double>, double, float>;
                    AccumT sum = AccumT(0);

                    for (int64_t n = 0; n < N; ++n) {
                        const T* input_plane = input + (n * C_in + ic) * H * W;

                        for (int64_t oh = 0; oh < H_out; ++oh) {
                            for (int64_t ow = 0; ow < W_out; ++ow) {
                                float h_base = oh * stride_h - pad_h + kh * dil_h;
                                float w_base = ow * stride_w - pad_w + kw * dil_w;

                                const T* off_h_plane = offset + (n * offset_groups * 2 * kH * kW +
                                    (offset_idx + 2 * k_linear)) * H_out * W_out;
                                const T* off_w_plane = offset + (n * offset_groups * 2 * kH * kW +
                                    (offset_idx + 2 * k_linear + 1)) * H_out * W_out;

                                float h_loc = h_base + static_cast<float>(off_h_plane[oh * W_out + ow]);
                                float w_loc = w_base + static_cast<float>(off_w_plane[oh * W_out + ow]);

                                AccumT val = static_cast<AccumT>(
                                    bilinear_interpolate(input_plane, H, W, h_loc, w_loc));

                                if (use_mask) {
                                    const T* mask_plane = mask + (n * offset_groups * kH * kW +
                                        (mask_idx + k_linear)) * H_out * W_out;
                                    val *= static_cast<AccumT>(mask_plane[oh * W_out + ow]);
                                }

                                AccumT go = static_cast<AccumT>(
                                    grad_output[(n * C_out + oc) * H_out * W_out + oh * W_out + ow]);
                                sum += go * val;
                            }
                        }
                    }

                    gw[k_linear] = static_cast<T>(sum);
                }
            }
        }
    }
}

auto deformable_conv2d_forward_kernel(
    const Tensor& input, const Tensor& offset, const Tensor& weight,
    const Tensor& bias, const Tensor& mask,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups) -> Tensor {

    // Float16 / BFloat16 don't have a raw-pointer template specialization
    // here — promote to Float32, compute, then narrow back to the caller's
    // dtype. Same pattern as src/nn/layers/flex_attention.cpp.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto out = deformable_conv2d_forward_kernel(
            input.to(DType::Float32),
            offset.to(DType::Float32),
            weight.to(DType::Float32),
            bias.numel() > 0 ? bias.to(DType::Float32) : bias,
            mask.numel() > 0 ? mask.to(DType::Float32) : mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, offset_groups);
        return out.to(orig);
    }

    auto ishape = input.shape();
    auto wshape = weight.shape();
    int64_t N = ishape[0], C_in = ishape[1], H = ishape[2], W = ishape[3];
    int64_t C_out = wshape[0], kH = wshape[2], kW = wshape[3];
    int64_t H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    int64_t W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

    // The impls divide C_in/groups, C_out/groups, C_in/offset_groups; a zero
    // divisor would SIGFPE. The op-layer only checks ndim==4, so validate here.
    validate_deformable_groups(C_in, C_out, groups, offset_groups);

    // data<T>() ignores strides — contiguify all raw-pointer operands so a
    // non-contiguous (channels-last/permuted/sliced) view is read correctly.
    const Tensor input_c  = input.is_contiguous()  ? input  : input.contiguous();
    const Tensor offset_c = offset.is_contiguous() ? offset : offset.contiguous();
    const Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    const Tensor bias_c   = bias.is_contiguous()   ? bias   : bias.contiguous();
    const Tensor mask_c   = mask.is_contiguous()   ? mask   : mask.contiguous();

    bool use_mask = mask.numel() > 0;
    auto output = zeros({N, C_out, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        deformable_conv2d_forward_impl<float>(
            input_c.data<float>(), offset_c.data<float>(), weight_c.data<float>(),
            bias.numel() > 0 ? bias_c.data<float>() : nullptr,
            use_mask ? mask_c.data<float>() : nullptr,
            output.data<float>(),
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else if (input.dtype() == DType::Float64) {
        deformable_conv2d_forward_impl<double>(
            input_c.data<double>(), offset_c.data<double>(), weight_c.data<double>(),
            bias.numel() > 0 ? bias_c.data<double>() : nullptr,
            use_mask ? mask_c.data<double>() : nullptr,
            output.data<double>(),
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else {
        throw std::runtime_error("deformable_conv2d: unsupported dtype (requires Float32 or Float64)");
    }

    return output;
}

auto deformable_conv2d_backward_input_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& offset,
    const Tensor& weight, const Tensor& mask,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups) -> std::vector<Tensor> {

    // Promote half precision to Float32 for the computation, narrow back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto outs = deformable_conv2d_backward_input_kernel(
            grad_output.to(DType::Float32),
            input.to(DType::Float32),
            offset.to(DType::Float32),
            weight.to(DType::Float32),
            mask.numel() > 0 ? mask.to(DType::Float32) : mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, offset_groups);
        std::vector<Tensor> narrowed;
        narrowed.reserve(outs.size());
        for (auto& t : outs) narrowed.push_back(t.numel() > 0 ? t.to(orig) : t);
        return narrowed;
    }

    auto ishape = input.shape();
    auto wshape = weight.shape();
    auto oshape = offset.shape();
    int64_t N = ishape[0], C_in = ishape[1], H = ishape[2], W = ishape[3];
    int64_t C_out = wshape[0], kH = wshape[2], kW = wshape[3];
    int64_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];

    // Guard the group divisions inside the impl (SIGFPE on a zero divisor).
    validate_deformable_groups(C_in, C_out, groups, offset_groups);

    // data<T>() ignores strides — contiguify all raw-pointer operands.
    const Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    const Tensor input_c  = input.is_contiguous()  ? input  : input.contiguous();
    const Tensor offset_c = offset.is_contiguous() ? offset : offset.contiguous();
    const Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    const Tensor mask_c   = mask.is_contiguous()   ? mask   : mask.contiguous();

    bool use_mask = mask.numel() > 0;
    auto grad_input = zeros(std::vector<int64_t>(ishape.begin(), ishape.end()), input.dtype(), input.device());
    auto grad_offset = zeros(std::vector<int64_t>(oshape.begin(), oshape.end()), input.dtype(), input.device());
    Tensor grad_mask;
    if (use_mask) {
        auto ms = mask.shape();
        grad_mask = zeros(std::vector<int64_t>(ms.begin(), ms.end()), input.dtype(), input.device());
    }

    if (input.dtype() == DType::Float32) {
        deformable_conv2d_backward_input_impl<float>(
            grad_output_c.data<float>(), input_c.data<float>(), offset_c.data<float>(),
            weight_c.data<float>(), use_mask ? mask_c.data<float>() : nullptr,
            grad_input.data<float>(), grad_offset.data<float>(),
            use_mask ? grad_mask.data<float>() : nullptr,
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else if (input.dtype() == DType::Float64) {
        deformable_conv2d_backward_input_impl<double>(
            grad_output_c.data<double>(), input_c.data<double>(), offset_c.data<double>(),
            weight_c.data<double>(), use_mask ? mask_c.data<double>() : nullptr,
            grad_input.data<double>(), grad_offset.data<double>(),
            use_mask ? grad_mask.data<double>() : nullptr,
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else {
        throw std::runtime_error("deformable_conv2d_backward_input: unsupported dtype");
    }

    if (use_mask) {
        return {grad_input, grad_offset, grad_mask};
    }
    return {grad_input, grad_offset};
}

auto deformable_conv2d_backward_weight_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& offset,
    const Tensor& mask,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    const std::vector<int64_t>& weight_shape) -> Tensor {

    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto out = deformable_conv2d_backward_weight_kernel(
            grad_output.to(DType::Float32),
            input.to(DType::Float32),
            offset.to(DType::Float32),
            mask.numel() > 0 ? mask.to(DType::Float32) : mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, offset_groups,
            weight_shape);
        return out.to(orig);
    }

    auto ishape = input.shape();
    int64_t N = ishape[0], C_in = ishape[1], H = ishape[2], W = ishape[3];
    int64_t C_out = weight_shape[0], kH = weight_shape[2], kW = weight_shape[3];
    int64_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];

    // Guard the group divisions inside the impl (SIGFPE on a zero divisor).
    validate_deformable_groups(C_in, C_out, groups, offset_groups);

    // data<T>() ignores strides — contiguify all raw-pointer operands.
    const Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    const Tensor input_c  = input.is_contiguous()  ? input  : input.contiguous();
    const Tensor offset_c = offset.is_contiguous() ? offset : offset.contiguous();
    const Tensor mask_c   = mask.is_contiguous()   ? mask   : mask.contiguous();

    bool use_mask = mask.numel() > 0;
    auto grad_weight = zeros(weight_shape, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        deformable_conv2d_backward_weight_impl<float>(
            grad_output_c.data<float>(), input_c.data<float>(), offset_c.data<float>(),
            use_mask ? mask_c.data<float>() : nullptr,
            grad_weight.data<float>(),
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else if (input.dtype() == DType::Float64) {
        deformable_conv2d_backward_weight_impl<double>(
            grad_output_c.data<double>(), input_c.data<double>(), offset_c.data<double>(),
            use_mask ? mask_c.data<double>() : nullptr,
            grad_weight.data<double>(),
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else {
        throw std::runtime_error("deformable_conv2d_backward_weight: unsupported dtype");
    }

    return grad_weight;
}

} // namespace cpu
} // namespace tenzor
