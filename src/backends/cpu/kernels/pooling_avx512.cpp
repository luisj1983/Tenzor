/**
 * @file pooling_avx512.cpp
 * @brief AVX-512 implementations of pooling operations
 *
 * Compiled separately with -mavx512f to enable AVX-512 in portable builds.
 * Provides 16-wide vectorized average pooling forward pass.
 * MaxPool is not vectorized here because it requires tracking per-element indices.
 */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #if defined(__AVX512F__)
        #include <immintrin.h>
        #define TENZOR_POOL_HAS_AVX512
    #endif
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace cpu {
namespace avx512 {

#ifdef TENZOR_POOL_HAS_AVX512

void avgpool2d_forward_f32(
    const float* __restrict in_data,
    float* __restrict out_data,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int64_t kernel_size, int64_t stride, int64_t padding)
{
    // For cases where the pooling window is fully inside the input (no padding effects),
    // we can vectorize across the W_out dimension using contiguous row reads.
    // For border cases with padding, fall back to scalar.

    #pragma omp parallel for collapse(3)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                int64_t h_start = oh * stride - padding;
                int64_t h_end = h_start + kernel_size;
                // Clamp to valid range
                int64_t h_valid_start = std::max(h_start, int64_t(0));
                int64_t h_valid_end = std::min(h_end, H);

                const float* base_ptr = in_data + (n * C + c) * H * W;
                float* out_ptr = out_data + ((n * C + c) * H_out + oh) * W_out;

                // Check if all rows in the kernel window are fully valid
                bool full_height = (h_valid_start == h_start && h_valid_end == h_end);

                int64_t ow = 0;

                if (full_height && padding == 0) {
                    // Fast path: no padding at all, kernel fully inside in both dims
                    // We can process multiple output columns using AVX-512
                    float inv_count = 1.0f / static_cast<float>(kernel_size * kernel_size);
                    __m512 vinv = _mm512_set1_ps(inv_count);

                    for (; ow + 16 <= W_out; ow += 16) {
                        __m512 vsum = _mm512_setzero_ps();

                        for (int64_t kh = 0; kh < kernel_size; ++kh) {
                            const float* row_ptr = base_ptr + (h_start + kh) * W;
                            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                                // Load 16 contiguous values starting at w_start + kw
                                // w_start for each ow is ow * stride + kw
                                // For stride=1: these are contiguous
                                // For stride>1: need gather
                                if (stride == 1) {
                                    __m512 v = _mm512_loadu_ps(row_ptr + ow + kw);
                                    vsum = _mm512_add_ps(vsum, v);
                                } else {
                                    // Use gather for non-unit stride
                                    __m512i vidx = _mm512_setr_epi32(
                                        (ow + 0) * stride + kw,
                                        (ow + 1) * stride + kw,
                                        (ow + 2) * stride + kw,
                                        (ow + 3) * stride + kw,
                                        (ow + 4) * stride + kw,
                                        (ow + 5) * stride + kw,
                                        (ow + 6) * stride + kw,
                                        (ow + 7) * stride + kw,
                                        (ow + 8) * stride + kw,
                                        (ow + 9) * stride + kw,
                                        (ow + 10) * stride + kw,
                                        (ow + 11) * stride + kw,
                                        (ow + 12) * stride + kw,
                                        (ow + 13) * stride + kw,
                                        (ow + 14) * stride + kw,
                                        (ow + 15) * stride + kw);
                                    __m512 v = _mm512_i32gather_ps(vidx, row_ptr, 4);
                                    vsum = _mm512_add_ps(vsum, v);
                                }
                            }
                        }

                        __m512 vavg = _mm512_mul_ps(vsum, vinv);
                        _mm512_storeu_ps(out_ptr + ow, vavg);
                    }
                }

                // Scalar fallback for remainder and border cases
                for (; ow < W_out; ++ow) {
                    int64_t w_start = ow * stride - padding;
                    float sum = 0.0f;
                    int64_t count = 0;

                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        int64_t h = h_start + kh;
                        if (h < 0 || h >= H) continue;
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t w = w_start + kw;
                            if (w >= 0 && w < W) {
                                sum += base_ptr[h * W + w];
                                count++;
                            }
                        }
                    }

                    out_ptr[ow] = (count > 0) ? sum / static_cast<float>(count) : 0.0f;
                }
            }
        }
    }
}

void adaptive_avgpool2d_forward_f32(
    const float* __restrict in_data,
    float* __restrict out_data,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out)
{
    #pragma omp parallel for collapse(3)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                int64_t h_start = (oh * H) / H_out;
                int64_t h_end = ((oh + 1) * H) / H_out;
                int64_t kh_size = h_end - h_start;

                const float* base_ptr = in_data + (n * C + c) * H * W;
                float* out_ptr = out_data + ((n * C + c) * H_out + oh) * W_out;

                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t w_start = (ow * W) / W_out;
                    int64_t w_end = ((ow + 1) * W) / W_out;
                    int64_t kw_size = w_end - w_start;
                    int64_t count = kh_size * kw_size;

                    // If the window row is long enough, vectorize the inner sum
                    __m512 vsum = _mm512_setzero_ps();
                    float scalar_sum = 0.0f;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        const float* row_ptr = base_ptr + h * W + w_start;
                        int64_t w = 0;

                        for (; w + 16 <= kw_size; w += 16) {
                            __m512 v = _mm512_loadu_ps(row_ptr + w);
                            vsum = _mm512_add_ps(vsum, v);
                        }
                        for (; w < kw_size; w++) {
                            scalar_sum += row_ptr[w];
                        }
                    }

                    float total = _mm512_reduce_add_ps(vsum) + scalar_sum;
                    out_ptr[ow] = count > 0 ? total / static_cast<float>(count) : 0.0f;
                }
            }
        }
    }
}

#else

// Audit item C.1: non-AVX-512 builds must NOT silently provide empty
// AVX-512 stubs that write nothing to the output buffer.  Any code path
// that compiles these symbols on a non-AVX-512 build and calls them
// would have silently produced uninitialised output.  The CMake gate
// (TENZOR_HAVE_AVX512) ensures this translation unit is not compiled
// outside AVX-512 builds; if it ever is, the abort below is the loud
// failure mode we want.
[[noreturn]] static void avx512_unavailable(const char* func) {
    std::fprintf(stderr,
        "[Tenzor] %s called on a build without AVX-512 support — "
        "rebuild with -DTENZOR_HAVE_AVX512=ON or use the scalar / AVX2 "
        "fallback path.\n", func);
    std::abort();
}

void avgpool2d_forward_f32(const float*, float*, int64_t, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, int64_t, int64_t) {
    avx512_unavailable("avgpool2d_forward_f32");
}
void adaptive_avgpool2d_forward_f32(const float*, float*, int64_t, int64_t, int64_t, int64_t,
                                     int64_t, int64_t) {
    avx512_unavailable("adaptive_avgpool2d_forward_f32");
}

#endif

} // namespace avx512
} // namespace cpu
} // namespace tenzor
