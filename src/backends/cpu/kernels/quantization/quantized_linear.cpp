/**
 * @file quantized_linear.cpp
 * @brief CPU kernels for quantized linear operations
 */

#include <cstdint>
#include <algorithm>
#include <immintrin.h>  // For SIMD operations

namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {

/**
 * @brief Quantized matrix multiplication for linear layer (CPU).
 *
 * Performs INT8 matrix multiplication with dequantization:
 * output = (input @ weight^T - zero_point_correction) * scale + bias
 */
auto quantized_linear_kernel(
    const int8_t* input,
    const int8_t* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    float input_scale,
    float weight_scale,
    float output_scale,
    int32_t input_zp,
    int32_t weight_zp
) -> void {
    float combined_scale = input_scale * weight_scale / output_scale;

    // Parallel over batch and output features
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            int32_t acc = 0;

            // Inner product with INT8 accumulation to INT32
            const int8_t* input_row = input + b * in_features;
            const int8_t* weight_row = weight + o * in_features;

#ifdef __AVX2__
            // SIMD-optimized path for x86 with AVX2
            __m256i acc_vec = _mm256_setzero_si256();
            int64_t i = 0;

            // Process 32 elements at a time
            for (; i + 32 <= in_features; i += 32) {
                // Load 32 INT8 values
                __m256i input_vec = _mm256_loadu_si256((__m256i*)(input_row + i));
                __m256i weight_vec = _mm256_loadu_si256((__m256i*)(weight_row + i));

                // Multiply and accumulate (INT8 -> INT16 -> INT32)
                __m256i prod_lo = _mm256_maddubs_epi16(input_vec, weight_vec);
                __m256i prod_hi = _mm256_madd_epi16(prod_lo, _mm256_set1_epi16(1));

                acc_vec = _mm256_add_epi32(acc_vec, prod_hi);
            }

            // Horizontal sum of accumulator
            __m128i sum128 = _mm_add_epi32(
                _mm256_castsi256_si128(acc_vec),
                _mm256_extracti128_si256(acc_vec, 1)
            );
            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            acc = _mm_cvtsi128_si32(sum128);

            // Process remaining elements
            for (; i < in_features; ++i) {
                acc += static_cast<int32_t>(input_row[i]) * static_cast<int32_t>(weight_row[i]);
            }
#else
            // Scalar fallback
            for (int64_t i = 0; i < in_features; ++i) {
                acc += static_cast<int32_t>(input_row[i]) * static_cast<int32_t>(weight_row[i]);
            }
#endif

            // Zero point correction
            acc -= input_zp * weight_zp * in_features;

            // Dequantize and add bias
            float result = static_cast<float>(acc) * combined_scale;
            if (bias != nullptr) {
                result += bias[o];
            }

            output[b * out_features + o] = result;
        }
    }
}

} // namespace kernels
} // namespace quantization
} // namespace nn
} // namespace tenzor
