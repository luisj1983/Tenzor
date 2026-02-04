/**
 * @file quantized_conv2d.cpp
 * @brief CPU kernels for quantized convolution operations
 */

#include <cstdint>
#include <algorithm>
#include <vector>
#include <cstring>

#if defined(__AVX512VNNI__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {

// INT8 dot product helper - uses best available SIMD
static inline int32_t dot_int8(const int8_t* a, const int8_t* b, int64_t len) {
    int32_t acc = 0;
    int64_t i = 0;

#if defined(__AVX512VNNI__)
    __m512i acc_vec = _mm512_setzero_si512();
    for (; i + 64 <= len; i += 64) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        acc_vec = _mm512_dpbusd_epi32(acc_vec, va, vb);
    }
    acc = _mm512_reduce_add_epi32(acc_vec);
#elif defined(__AVX2__)
    __m256i acc_vec = _mm256_setzero_si256();
    for (; i + 32 <= len; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i prod = _mm256_maddubs_epi16(va, vb);
        __m256i prod32 = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
        acc_vec = _mm256_add_epi32(acc_vec, prod32);
    }
    __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(acc_vec),
        _mm256_extracti128_si256(acc_vec, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    acc = _mm_cvtsi128_si32(sum128);
#endif

    for (; i < len; ++i) {
        acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return acc;
}

/**
 * @brief im2col for INT8 input: extracts patches into a column buffer
 *
 * Transforms input from [in_channels, h_in, w_in] into a matrix
 * of shape [h_out * w_out, in_channels * kH * kW] for GEMM.
 */
static void im2col_int8(
    const int8_t* input,  // [in_channels, h_in, w_in]
    int8_t* col_buffer,   // [h_out * w_out, in_channels * kH * kW]
    int64_t in_channels,
    int64_t h_in, int64_t w_in,
    int64_t h_out, int64_t w_out,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding
) {
    const int64_t col_width = in_channels * kernel_size * kernel_size;

    #pragma omp parallel for if(h_out * w_out > 256)
    for (int64_t out_idx = 0; out_idx < h_out * w_out; ++out_idx) {
        int64_t oh = out_idx / w_out;
        int64_t ow = out_idx % w_out;
        int8_t* col_row = col_buffer + out_idx * col_width;

        int64_t col_pos = 0;
        for (int64_t ic = 0; ic < in_channels; ++ic) {
            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t ih = oh * stride + kh - padding;
                    int64_t iw = ow * stride + kw - padding;

                    if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                        col_row[col_pos] = input[(ic * h_in + ih) * w_in + iw];
                    } else {
                        col_row[col_pos] = 0;  // Zero padding
                    }
                    ++col_pos;
                }
            }
        }
    }
}

/**
 * @brief Quantized 2D convolution (CPU) using im2col + GEMM.
 *
 * Performs INT8 convolution with dequantization.
 * Uses im2col to transform the convolution into a matrix multiplication,
 * then uses SIMD-optimized INT8 GEMM (with VNNI/AVX2).
 */
auto quantized_conv2d_kernel(
    const int8_t* input,
    const int8_t* weight,
    const float* bias,
    float* output,
    int64_t batch,
    int64_t in_channels,
    int64_t out_channels,
    int64_t h_in,
    int64_t w_in,
    int64_t h_out,
    int64_t w_out,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    float input_scale,
    float weight_scale,
    int32_t input_zp,
    int32_t weight_zp
) -> void {
    float combined_scale = input_scale * weight_scale;
    const int64_t col_width = in_channels * kernel_size * kernel_size;
    const int64_t spatial_out = h_out * w_out;

    // im2col + GEMM approach
    #pragma omp parallel
    {
        // Per-thread column buffer to avoid allocation contention
        std::vector<int8_t> col_buffer(spatial_out * col_width);

        #pragma omp for
        for (int64_t b = 0; b < batch; ++b) {
            const int8_t* batch_input = input + b * in_channels * h_in * w_in;

            // Step 1: im2col - extract patches into column buffer
            im2col_int8(batch_input, col_buffer.data(),
                        in_channels, h_in, w_in, h_out, w_out,
                        kernel_size, stride, padding);

            // Step 2: GEMM - weight[oc, col_width] @ col_buffer^T[col_width, spatial_out]
            // For each output channel, dot product with each column
            for (int64_t oc = 0; oc < out_channels; ++oc) {
                const int8_t* weight_row = weight + oc * col_width;

                for (int64_t s = 0; s < spatial_out; ++s) {
                    const int8_t* col_row = col_buffer.data() + s * col_width;

                    // SIMD-accelerated INT8 dot product
                    int32_t acc = dot_int8(weight_row, col_row, col_width);

                    // Zero point correction
                    acc -= input_zp * weight_zp * col_width;

                    // Dequantize and add bias
                    float result = static_cast<float>(acc) * combined_scale;
                    if (bias != nullptr) {
                        result += bias[oc];
                    }

                    int64_t output_idx = ((b * out_channels + oc) * h_out + s / w_out) * w_out + s % w_out;
                    output[output_idx] = result;
                }
            }
        }
    }
}

} // namespace kernels
} // namespace quantization
} // namespace nn
} // namespace tenzor
