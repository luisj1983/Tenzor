#include "tenzor/core/tensor.hpp"
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

#ifdef _OPENMP
#include <omp.h>
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
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    const int64_t col_width = channels * kernel_h * kernel_w;

    // Nested loop approach: eliminates 5 div/mod per element (~200 cycles saved per element)
    #pragma omp parallel for collapse(3) if(batch * out_h * out_w > OmpThresholds::medium())
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            for (int64_t ow = 0; ow < out_w; ++ow) {
                T* col_ptr = output + ((b * out_h + oh) * out_w + ow) * col_width;
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t kh_idx = 0; kh_idx < kernel_h; ++kh_idx) {
                        int64_t ih = oh * stride - padding + kh_idx * dilation;
                        for (int64_t kw_idx = 0; kw_idx < kernel_w; ++kw_idx) {
                            int64_t iw = ow * stride - padding + kw_idx * dilation;
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
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    // Zero initialize output
    int64_t output_size = batch * channels * height * width;
    std::memset(output, 0, output_size * sizeof(T));

    // Process each output element and accumulate from all contributing col positions
    // This avoids race conditions compared to the col-centric approach
    #pragma omp parallel for collapse(4) if(output_size > OmpThresholds::medium())
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t ih = 0; ih < height; ++ih) {
                for (int64_t iw = 0; iw < width; ++iw) {
                    // Accumulate from all kernel positions that contribute to this output
                    T sum = T(0.0f);

                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            // Reverse the mapping: given (ih, iw) and (kh, kw), find (oh, ow)
                            int64_t ih_shifted = ih + padding - kh * dilation;
                            int64_t iw_shifted = iw + padding - kw * dilation;

                            // Check if this maps to a valid output position
                            if (ih_shifted % stride == 0 && iw_shifted % stride == 0) {
                                int64_t oh = ih_shifted / stride;
                                int64_t ow = iw_shifted / stride;

                                if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                    // This kernel position contributes to our output
                                    int64_t col_row = b * out_h * out_w + oh * out_w + ow;
                                    int64_t col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
                                    int64_t col_idx = col_row * (channels * kernel_h * kernel_w) + col_col;

                                    sum += col[col_idx];
                                }
                            }
                        }
                    }

                    // Write accumulated value
                    int64_t output_idx = b * (channels * height * width) +
                                        c * (height * width) +
                                        ih * width + iw;
                    output[output_idx] = sum;
                }
            }
        }
    }
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

    int64_t total_output_elements = batch * out_channels * out_h * out_w;

    // Skip only for tiny outputs (< 4K elements, e.g., batch=1, channels=64, 8x8)
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

        // Refresh weight data only if the data pointer changed (optimizer updated weights)
        if (weight.data_ptr() != cached->cached_weight_ptr) {
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

// Template helper for dtype-generic conv2d forward
template<typename T>
void conv2d_forward_impl(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& output,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
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

    // Initialize output to zeros
    std::memset(output.data<T>(), 0, output.numel() * sizeof(T));

    // Process each group separately
    int64_t out_channels_per_group = out_channels / groups;

    // =========================================================================
    // Fast path for 1x1 convolutions (skip im2col, direct GEMM)
    // 1x1 conv is essentially a per-pixel fully-connected layer
    // =========================================================================
    if (kernel_h == 1 && kernel_w == 1 && stride == 1 && padding == 0 && dilation == 1) {
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

            // Process each batch
            #pragma omp parallel for if(batch * spatial > OmpThresholds::medium())
            for (int64_t b = 0; b < batch; ++b) {
                // For each spatial position
                for (int64_t s = 0; s < spatial; ++s) {
                    // Compute dot product: out[c] = sum_k(in[k] * weight[c,k])
                    for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                        T sum{};  // Value-initialize to zero
                        for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                            int64_t in_idx = b * (in_channels * spatial) + (in_start + ic) * spatial + s;
                            int64_t w_idx = (out_start + oc) * in_channels_per_group + ic;
                            sum += input_data[in_idx] * weight_data[w_idx];
                        }
                        int64_t out_idx = b * (out_channels * spatial) + (out_start + oc) * spatial + s;
                        output_data[out_idx] = sum;
                    }
                }
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
    if (kernel_h == 3 && kernel_w == 3 && stride == 1 && dilation == 1 && groups == 1) {
        const T* input_data = input.data<T>();
        const T* weight_data = weight.data<T>();
        T* output_data = output.data<T>();

        // Winograd F(2,3): transforms 4x4 input tile -> 4x4, 3x3 filter -> 4x4
        // Element-wise multiply in transform domain, then inverse transform to 2x2 output

        // Number of 2x2 output tiles
        int64_t tile_h = (out_h + 1) / 2;
        int64_t tile_w = (out_w + 1) / 2;
        int64_t num_tiles = tile_h * tile_w;

        // Pre-transform all filters: G * g * G^T for each (oc, ic) pair
        // G is the 4x3 filter transform matrix for F(2,3):
        // G = [[1,    0,    0   ],
        //      [0.5,  0.5,  0.5 ],
        //      [0.5, -0.5,  0.5 ],
        //      [0,    0,    1   ]]
        int64_t C_in = in_channels;
        int64_t C_out = out_channels;
        std::vector<T> U(C_out * C_in * 16);  // 4x4 per filter

        #pragma omp parallel for collapse(2) if(C_out * C_in > 64)
        for (int64_t oc = 0; oc < C_out; ++oc) {
            for (int64_t ic = 0; ic < C_in; ++ic) {
                const T* g_ptr = weight_data + (oc * C_in + ic) * 9;
                T* u_ptr = U.data() + (oc * C_in + ic) * 16;

                // g is 3x3 filter: g[r][s]
                // Compute temp = G * g  (4x3 * 3x3 = 4x3)
                T temp[4][3];
                for (int s = 0; s < 3; ++s) {
                    temp[0][s] = g_ptr[0 * 3 + s];
                    temp[1][s] = static_cast<T>(0.5) * (g_ptr[0 * 3 + s] + g_ptr[1 * 3 + s] + g_ptr[2 * 3 + s]);
                    temp[2][s] = static_cast<T>(0.5) * (g_ptr[0 * 3 + s] - g_ptr[1 * 3 + s] + g_ptr[2 * 3 + s]);
                    temp[3][s] = g_ptr[2 * 3 + s];
                }
                // Compute U = temp * G^T  (4x3 * 3x4 = 4x4)
                for (int r = 0; r < 4; ++r) {
                    u_ptr[r * 4 + 0] = temp[r][0];
                    u_ptr[r * 4 + 1] = static_cast<T>(0.5) * (temp[r][0] + temp[r][1] + temp[r][2]);
                    u_ptr[r * 4 + 2] = static_cast<T>(0.5) * (temp[r][0] - temp[r][1] + temp[r][2]);
                    u_ptr[r * 4 + 3] = temp[r][2];
                }
            }
        }

        // Padded input height/width for tile extraction
        int64_t padded_h = height + 2 * padding;
        int64_t padded_w = width + 2 * padding;

        // Process each batch
        #pragma omp parallel if(batch * num_tiles > 64)
        {
            // Per-thread buffers
            std::vector<T> V(C_in * 16);   // Transformed input tiles for one tile position
            std::vector<T> M(C_out * 16);  // Element-wise product result

            #pragma omp for
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t th = 0; th < tile_h; ++th) {
                    for (int64_t tw = 0; tw < tile_w; ++tw) {
                        // Extract and transform 4x4 input tile for each input channel
                        // B^T * d * B where B^T is the 4x4 input transform:
                        // B^T = [[1,  0, -1,  0],
                        //        [0,  1,  1,  0],
                        //        [0, -1,  1,  0],
                        //        [0,  1,  0, -1]]
                        int64_t tile_start_h = th * 2 - padding;
                        int64_t tile_start_w = tw * 2 - padding;

                        for (int64_t ic = 0; ic < C_in; ++ic) {
                            // Load 4x4 tile from input (with zero-padding for out-of-bounds)
                            T d[4][4];
                            for (int r = 0; r < 4; ++r) {
                                for (int s = 0; s < 4; ++s) {
                                    int64_t ih = tile_start_h + r;
                                    int64_t iw = tile_start_w + s;
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        d[r][s] = input_data[b * (C_in * height * width) +
                                                            ic * (height * width) +
                                                            ih * width + iw];
                                    } else {
                                        d[r][s] = T{};
                                    }
                                }
                            }

                            // Compute B^T * d (4x4 * 4x4 = 4x4)
                            T temp[4][4];
                            for (int s = 0; s < 4; ++s) {
                                temp[0][s] = d[0][s] - d[2][s];
                                temp[1][s] = d[1][s] + d[2][s];
                                temp[2][s] = -d[1][s] + d[2][s];
                                temp[3][s] = d[1][s] - d[3][s];
                            }
                            // Compute V = temp * B (4x4 * 4x4 = 4x4)
                            T* v_ptr = V.data() + ic * 16;
                            for (int r = 0; r < 4; ++r) {
                                v_ptr[r * 4 + 0] = temp[r][0] - temp[r][2];
                                v_ptr[r * 4 + 1] = temp[r][1] + temp[r][2];
                                v_ptr[r * 4 + 2] = -temp[r][1] + temp[r][2];
                                v_ptr[r * 4 + 3] = temp[r][1] - temp[r][3];
                            }
                        }

                        // Element-wise multiply and accumulate: M[oc][i] = sum_ic U[oc][ic][i] * V[ic][i]
                        for (int64_t oc = 0; oc < C_out; ++oc) {
                            T* m_ptr = M.data() + oc * 16;
                            for (int i = 0; i < 16; ++i) {
                                m_ptr[i] = T{};
                            }
                            for (int64_t ic = 0; ic < C_in; ++ic) {
                                const T* u_ptr = U.data() + (oc * C_in + ic) * 16;
                                const T* v_ptr = V.data() + ic * 16;
                                for (int i = 0; i < 16; ++i) {
                                    m_ptr[i] += u_ptr[i] * v_ptr[i];
                                }
                            }
                        }

                        // Inverse transform: A^T * M * A  -> 2x2 output tile
                        // A^T = [[1, 1,  1, 0],
                        //        [0, 1, -1, -1]]
                        for (int64_t oc = 0; oc < C_out; ++oc) {
                            T* m_ptr = M.data() + oc * 16;

                            // Compute A^T * M (2x4 * 4x4 = 2x4)
                            T temp2[2][4];
                            for (int s = 0; s < 4; ++s) {
                                temp2[0][s] = m_ptr[0 * 4 + s] + m_ptr[1 * 4 + s] + m_ptr[2 * 4 + s];
                                temp2[1][s] = m_ptr[1 * 4 + s] - m_ptr[2 * 4 + s] - m_ptr[3 * 4 + s];
                            }
                            // Compute output = temp2 * A (2x4 * 4x2 = 2x2)
                            T out[2][2];
                            out[0][0] = temp2[0][0] + temp2[0][1] + temp2[0][2];
                            out[0][1] = temp2[0][1] - temp2[0][2] - temp2[0][3];
                            out[1][0] = temp2[1][0] + temp2[1][1] + temp2[1][2];
                            out[1][1] = temp2[1][1] - temp2[1][2] - temp2[1][3];

                            // Write output tile (handle boundary for odd output sizes)
                            for (int r = 0; r < 2; ++r) {
                                for (int s = 0; s < 2; ++s) {
                                    int64_t oh = th * 2 + r;
                                    int64_t ow = tw * 2 + s;
                                    if (oh < out_h && ow < out_w) {
                                        output_data[b * (C_out * out_h * out_w) +
                                                   oc * (out_h * out_w) +
                                                   oh * out_w + ow] = out[r][s];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add bias if present
        if (bias) {
            const T* bias_data = bias->data<T>();
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
                stride,
                padding,
                dilation,
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

auto conv2d_forward_kernel(
    const Tensor& input_orig,    // (batch, in_channels, height, width)
    const Tensor& weight_orig,   // (out_channels, in_channels, kernel_h, kernel_w)
    const Tensor* bias,          // (out_channels) or nullptr
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) -> Tensor {
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

    // Calculate output dimensions
    int64_t out_h = calculate_output_size(height, kernel_h, stride, padding, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride, padding, dilation);

    // Create output tensor with correct dtype
    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

#ifdef TENZOR_USE_ONEDNN
    // Try oneDNN-accelerated convolution first (3-8x faster for Float32)
    if (conv2d_forward_onednn(input, weight, bias, output, stride, padding, dilation, groups)) {
        return output;
    }
    // Fall through to im2col+GEMM if oneDNN not applicable
#endif

    // Try Winograd for eligible 3x3 stride-1 convolutions (Float32 only)
    if (input.dtype() == DType::Float32) {
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
                for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
                    for (int64_t oc = 0; oc < out_channels; ++oc) {
                        float bv = bias_data[oc];
                        float* ch = out_data + (b_idx * out_channels + oc) * spatial;
                        for (int64_t i = 0; i < spatial; ++i) ch[i] += bv;
                    }
                }
            }
            return output;
        }
    }

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        conv2d_forward_impl<float>(input, weight, bias, output, stride, padding, dilation, groups);
    } else if (input.dtype() == DType::Float64) {
        conv2d_forward_impl<double>(input, weight, bias, output, stride, padding, dilation, groups);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Half-precision on CPU: compute in Float32 for performance.
        // Scalar half-precision GEMM is 10-100x slower than optimized Float32 GEMM/oneDNN.
        // Convert inputs to Float32, compute, then convert output back.
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
        if (!conv2d_forward_onednn(input_f32, weight_f32, bias_f32_ptr, output_f32, stride, padding, dilation, groups)) {
#endif
            conv2d_forward_impl<float>(input_f32, weight_f32, bias_f32_ptr, output_f32, stride, padding, dilation, groups);
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
    int64_t stride,
    int64_t padding,
    int64_t dilation,
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

    // Zero-initialize gradient input
    std::memset(grad_input.data<T>(), 0, grad_input.numel() * sizeof(T));

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Process per-batch to handle correct strides with groups
        int64_t col_per_batch = out_h * out_w * col_cols;
        int64_t grad_out_per_batch = out_h * out_w * out_channels_per_group;

        // Allocate col buffer for all batches
        std::vector<T> grad_col(col_rows * col_cols);

        // Reshape grad_output for this group: extract per-batch, per-group data
        // and pack into contiguous GEMM input
        std::vector<T> grad_out_packed(col_rows * out_channels_per_group);
        for (int64_t b = 0; b < batch; ++b) {
            const T* src = grad_output.data<T>() + (b * out_channels + out_start) * out_h * out_w;
            T* dst = grad_out_packed.data() + b * grad_out_per_batch;
            std::memcpy(dst, src, grad_out_per_batch * sizeof(T));
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
                stride,
                padding,
                dilation,
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
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) -> Tensor {
    // Ensure contiguous layout for pointer-arithmetic kernels
    Tensor grad_output = grad_output_orig.is_contiguous() ? grad_output_orig : grad_output_orig.contiguous();
    Tensor weight = weight_orig.is_contiguous() ? weight_orig : weight_orig.contiguous();

    // Initialize gradient w.r.t input with correct dtype
    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    // Dispatch based on dtype
    if (grad_output.dtype() == DType::Float32) {
        conv2d_backward_input_impl<float>(grad_output, weight, grad_input, input_shape, stride, padding, dilation, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        conv2d_backward_input_impl<double>(grad_output, weight, grad_input, input_shape, stride, padding, dilation, groups);
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Half-precision on CPU: compute in Float32 for performance
        DType original_dtype = grad_output.dtype();
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        Tensor grad_input_f32(input_shape, DType::Float32, grad_output.device());

        conv2d_backward_input_impl<float>(grad_output_f32, weight_f32, grad_input_f32, input_shape, stride, padding, dilation, groups);

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
    int64_t stride,
    int64_t padding,
    int64_t dilation,
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

    // Zero-initialize gradient weight
    std::memset(grad_weight.data<T>(), 0, grad_weight.numel() * sizeof(T));

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
                stride,
                padding,
                dilation,
                out_h,
                out_w
            );
        }

        int64_t M = out_channels_per_group;
        int64_t K = col_rows;
        int64_t N = col_cols;

        // Pack grad_output for this group across batches
        int64_t grad_out_per_batch = out_h * out_w * out_channels_per_group;
        std::vector<T> grad_out_packed(col_rows * out_channels_per_group);
        for (int64_t b = 0; b < batch; ++b) {
            const T* src = grad_output.data<T>() + (b * out_channels + out_start) * out_h * out_w;
            T* dst = grad_out_packed.data() + b * grad_out_per_batch;
            std::memcpy(dst, src, grad_out_per_batch * sizeof(T));
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
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) -> Tensor {
    // Ensure contiguous layout for pointer-arithmetic kernels
    Tensor grad_output = grad_output_orig.is_contiguous() ? grad_output_orig : grad_output_orig.contiguous();
    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();

    // Initialize gradient w.r.t weight
    Tensor grad_weight(weight_shape, grad_output.dtype(), grad_output.device());

    // Dispatch based on dtype
    if (grad_output.dtype() == DType::Float32) {
        conv2d_backward_weight_impl<float>(grad_output, input, grad_weight, weight_shape, stride, padding, dilation, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        conv2d_backward_weight_impl<double>(grad_output, input, grad_weight, weight_shape, stride, padding, dilation, groups);
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Half-precision on CPU: compute in Float32 for performance
        DType original_dtype = grad_output.dtype();
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        Tensor grad_weight_f32(weight_shape, DType::Float32, grad_output.device());

        conv2d_backward_weight_impl<float>(grad_output_f32, input_f32, grad_weight_f32, weight_shape, stride, padding, dilation, groups);

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
    std::memset(grad_bias_data, 0, out_channels * sizeof(T));

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
    Tensor grad_bias({out_channels}, grad_output.dtype(), grad_output.device());

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
        Tensor grad_bias_f32({out_channels}, DType::Float32, grad_output.device());

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
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t dilation,
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

        // Initialize accumulator
        float sum = 0.0f;

        // Gather from all input positions that contribute to this output position
        for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    // For transposed conv: out = (in - 1) * stride - 2*padding + dilation * (kernel - 1) + out_padding + 1
                    // Inverse: which input position (ih, iw) with kernel (kh, kw) contributes to (h, w)?
                    // oh = ih * stride - padding + kh * dilation
                    // ih * stride = oh + padding - kh * dilation
                    // ih = (oh + padding - kh * dilation) / stride (must be integer and in bounds)

                    int64_t h_shifted = h + padding - kh * dilation;
                    int64_t w_shifted = w + padding - kw * dilation;

                    // Check if this maps to a valid input position
                    if (h_shifted >= 0 && h_shifted % stride == 0 &&
                        w_shifted >= 0 && w_shifted % stride == 0) {

                        int64_t ih = h_shifted / stride;
                        int64_t iw = w_shifted / stride;

                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            // Get input value
                            int64_t input_idx = b * (in_channels * in_h * in_w) +
                                               (in_start + ic) * (in_h * in_w) +
                                               ih * in_w + iw;
                            float input_val = static_cast<float>(input_data[input_idx]);

                            // Get weight value
                            // Weight shape: (in_channels, out_channels/groups, kernel_h, kernel_w)
                            int64_t weight_idx = (in_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                                oc * (kernel_h * kernel_w) +
                                                kh * kernel_w + kw;
                            float weight_val = static_cast<float>(weight_data[weight_idx]);

                            sum += input_val * weight_val;
                        }
                    }
                }
            }
        }

        // Add bias if present
        if (bias != nullptr) {
            sum += static_cast<float>(bias->data<T>()[c]);
        }

        output_data[idx] = T(sum);
    }
}

auto conv_transpose2d_forward_kernel(
    const Tensor& input_orig,     // (batch, in_channels, in_h, in_w)
    const Tensor& weight_orig,    // (in_channels, out_channels/groups, kernel_h, kernel_w)
    const Tensor* bias,           // (out_channels) or nullptr
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t dilation,
    int64_t groups
) -> Tensor {
    // Ensure contiguous layout for pointer-arithmetic kernels
    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    Tensor weight = weight_orig.is_contiguous() ? weight_orig : weight_orig.contiguous();

    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int64_t out_channels_per_group = weight_shape[1];
    int64_t out_channels = out_channels_per_group * groups;
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions for transposed convolution
    int64_t out_h = calculate_transpose_output_size(in_h, kernel_h, stride, padding, output_padding, dilation);
    int64_t out_w = calculate_transpose_output_size(in_w, kernel_w, stride, padding, output_padding, dilation);

    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument(
            "Invalid ConvTranspose2d configuration: output dimensions are non-positive (out_h=" +
            std::to_string(out_h) + ", out_w=" + std::to_string(out_w) + ")"
        );
    }

    // Create output tensor with correct dtype
    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        conv_transpose2d_forward_impl<float>(input, weight, bias, output, stride, padding, output_padding, dilation, groups);
    } else if (input.dtype() == DType::Float64) {
        conv_transpose2d_forward_impl<double>(input, weight, bias, output, stride, padding, output_padding, dilation, groups);
    } else if (input.dtype() == DType::Float16) {
        conv_transpose2d_forward_impl<Float16>(input, weight, bias, output, stride, padding, output_padding, dilation, groups);
    } else if (input.dtype() == DType::BFloat16) {
        conv_transpose2d_forward_impl<BFloat16>(input, weight, bias, output, stride, padding, output_padding, dilation, groups);
    } else {
        throw std::runtime_error("Unsupported dtype for conv_transpose2d_forward");
    }

    return output;
}

// ============================================================================
// Depthwise Conv2d Forward CPU Implementation
// ============================================================================

template<typename T>
void depthwise_conv2d_impl(const T* in_data, const T* w_data, const T* b_data, T* out_data,
                            int64_t N, int64_t C, int64_t H, int64_t W,
                            int64_t kH, int64_t kW, int64_t H_out, int64_t W_out,
                            int64_t stride, int64_t padding, int64_t dilation) {
    int64_t depthwise_numel = N * C * H_out * W_out;
    #pragma omp parallel for collapse(4) if(depthwise_numel > OmpThresholds::medium())
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    float sum = 0.0f;
                    float compensation = 0.0f;  // Kahan compensation

                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t h = oh * stride - padding + kh * dilation;
                            int64_t w = ow * stride - padding + kw * dilation;

                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                float product = static_cast<float>(in_data[((n * C + c) * H + h) * W + w]) *
                                       static_cast<float>(w_data[(c * 1 + 0) * kH * kW + kh * kW + kw]);
                                float y = product - compensation;
                                float t = sum + y;
                                compensation = (t - sum) - y;
                                sum = t;
                            }
                        }
                    }

                    if (b_data) {
                        float y = static_cast<float>(b_data[c]) - compensation;
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
    const float* __restrict__ in_data,
    const float* __restrict__ w_data,
    const float* __restrict__ b_data,
    float* __restrict__ out_data,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kH, int64_t kW, int64_t H_out, int64_t W_out,
    int64_t padding) {
    // stride=1, dilation=1 is assumed by the caller

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
                        int64_t ih = oh - padding + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* in_row = in_data + ((n * C + c) * H + ih) * W;

                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t iw_start = ow - padding + kw;

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
                        int64_t ih = oh - padding + kh;
                        if (ih < 0 || ih >= H) continue;

                        const float* in_row = in_data + ((n * C + c) * H + ih) * W;

                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t iw = ow - padding + kw;
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
                              const Tensor* bias, int64_t stride,
                              int64_t padding, int64_t dilation) -> Tensor {
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

    int64_t H_out = (H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int64_t W_out = (W + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

    auto output = Tensor::empty_uninitialized({N, C, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
#ifdef TENZOR_CONV_AVX2
        // Use AVX2 SIMD path for stride=1, dilation=1 (contiguous spatial access pattern)
        if (stride == 1 && dilation == 1) {
            depthwise_conv2d_avx2_f32(input.data<float>(), weight.data<float>(),
                bias ? bias->data<float>() : nullptr, output.data<float>(),
                N, C, H, W, kH, kW, H_out, W_out, padding);
        } else
#endif
        {
            depthwise_conv2d_impl<float>(input.data<float>(), weight.data<float>(),
                bias ? bias->data<float>() : nullptr, output.data<float>(),
                N, C, H, W, kH, kW, H_out, W_out, stride, padding, dilation);
        }
    } else if (input.dtype() == DType::Float64) {
        depthwise_conv2d_impl<double>(input.data<double>(), weight.data<double>(),
            bias ? bias->data<double>() : nullptr, output.data<double>(),
            N, C, H, W, kH, kW, H_out, W_out, stride, padding, dilation);
    } else if (input.dtype() == DType::Float16) {
        depthwise_conv2d_impl<Float16>(input.data<Float16>(), weight.data<Float16>(),
            bias ? bias->data<Float16>() : nullptr, output.data<Float16>(),
            N, C, H, W, kH, kW, H_out, W_out, stride, padding, dilation);
    } else if (input.dtype() == DType::BFloat16) {
        depthwise_conv2d_impl<BFloat16>(input.data<BFloat16>(), weight.data<BFloat16>(),
            bias ? bias->data<BFloat16>() : nullptr, output.data<BFloat16>(),
            N, C, H, W, kH, kW, H_out, W_out, stride, padding, dilation);
    } else {
        throw std::runtime_error("Unsupported dtype for depthwise_conv2d");
    }

    return output;
}

} // namespace cpu
} // namespace tenzor
