/**
 * @file quantized_linear_int4.cpp
 * @brief CPU kernels for INT4 quantized linear operations
 *
 * INT4 packs two values per byte. Unpacking extracts low/high nibbles
 * to INT8 before inner product computation, enabling 2x memory savings
 * over INT8 quantization.
 *
 * Storage format: two INT4 values packed per uint8_t
 *   byte = (high_nibble << 4) | (low_nibble & 0x0F)
 *   low_nibble  = byte & 0x0F  (values 0-15, offset by 8 for signed: -8 to 7)
 *   high_nibble = byte >> 4     (values 0-15, offset by 8 for signed: -8 to 7)
 */

#include <cstdint>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {

/// Unpack INT4 weight byte to two signed int8 values.
/// INT4 range is [0, 15]; subtract 8 for signed range [-8, 7].
static inline void unpack_int4(uint8_t packed, int8_t& low, int8_t& high) {
    low  = static_cast<int8_t>((packed & 0x0F)) - 8;
    high = static_cast<int8_t>((packed >> 4)) - 8;
}

/**
 * @brief INT4 quantized matrix multiplication for linear layer (CPU).
 *
 * Weights are stored in packed INT4 format (2 values per byte).
 * Input is INT8. Computation unpacks INT4→INT8 then performs INT8 dot product.
 *
 * output[b][o] = sum_k(input[b][k] * unpack(weight[o][k/2])) * scale + bias[o]
 *
 * @param input INT8 input data [batch_size, in_features]
 * @param weight_packed Packed INT4 weights [out_features, in_features/2]
 * @param bias Optional FP32 bias [out_features], may be nullptr
 * @param output FP32 output [batch_size, out_features]
 * @param batch_size Number of samples
 * @param in_features Input feature dimension (must be even)
 * @param out_features Output feature dimension
 * @param input_scale Input quantization scale
 * @param weight_scale Weight quantization scale
 * @param output_scale Output quantization scale
 */
auto quantized_linear_int4_kernel(
    const int8_t* input,
    const uint8_t* weight_packed,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    float input_scale,
    float weight_scale,
    float output_scale
) -> void {
    float combined_scale = input_scale * weight_scale / output_scale;
    int64_t packed_features = in_features / 2;

    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            int32_t acc = 0;
            const int8_t* input_row = input + b * in_features;
            const uint8_t* weight_row = weight_packed + o * packed_features;

#if defined(__AVX2__)
            // AVX2 path: unpack INT4 to INT16 and use _mm256_madd_epi16
            __m256i acc_vec = _mm256_setzero_si256();
            int64_t p = 0;

            // Process 16 packed bytes (32 INT4 values) at a time
            const __m256i mask_low = _mm256_set1_epi8(0x0F);
            const __m256i offset = _mm256_set1_epi8(8);

            for (; p + 16 <= packed_features; p += 16) {
                // Load 16 packed bytes = 32 INT4 weights
                __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight_row + p));
                __m256i packed256 = _mm256_cvtepu8_epi16(packed);

                // Extract low and high nibbles
                __m256i w_low = _mm256_and_si256(packed256, _mm256_set1_epi16(0x0F));
                __m256i w_high = _mm256_srli_epi16(packed256, 4);
                w_high = _mm256_and_si256(w_high, _mm256_set1_epi16(0x0F));

                // Subtract offset (signed: -8 to 7)
                w_low = _mm256_sub_epi16(w_low, _mm256_set1_epi16(8));
                w_high = _mm256_sub_epi16(w_high, _mm256_set1_epi16(8));

                // Load corresponding input INT8 values (32 values: interleaved low, high)
                // Input indices: for packed byte p, input[2p] pairs with low, input[2p+1] with high
                __m256i in_low = _mm256_cvtepi8_epi16(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_row + p * 2)));
                __m256i in_high = _mm256_cvtepi8_epi16(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_row + p * 2 + 16)));

                // Multiply and accumulate (INT16 * INT16 -> INT32)
                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(in_low, w_low));
                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(in_high, w_high));
            }

            // Horizontal sum of acc_vec
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc_vec),
                                           _mm256_extracti128_si256(acc_vec, 1));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 1, 0, 1)));
            acc = _mm_cvtsi128_si32(sum128);

            // Scalar tail
            for (; p < packed_features; ++p) {
                int8_t w_lo, w_hi;
                unpack_int4(weight_row[p], w_lo, w_hi);
                acc += static_cast<int32_t>(input_row[p * 2]) * static_cast<int32_t>(w_lo);
                acc += static_cast<int32_t>(input_row[p * 2 + 1]) * static_cast<int32_t>(w_hi);
            }
#else
            // Scalar fallback
            for (int64_t p = 0; p < packed_features; ++p) {
                int8_t w_lo, w_hi;
                unpack_int4(weight_row[p], w_lo, w_hi);
                acc += static_cast<int32_t>(input_row[p * 2]) * static_cast<int32_t>(w_lo);
                acc += static_cast<int32_t>(input_row[p * 2 + 1]) * static_cast<int32_t>(w_hi);
            }
#endif

            float result = static_cast<float>(acc) * combined_scale;
            if (bias) {
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
