/**
 * @file quantized_linear_int4.cpp
 * @brief CPU kernels for INT4 quantized linear operations
 *
 * INT4 packs two values per byte. Unpacking extracts low/high nibbles
 * to INT8 before inner product computation, enabling 2x memory savings
 * over INT8 quantization.
 *
 * Storage format: two signed INT4 values (two's-complement, range [-8, 7])
 * packed per uint8_t, each masked to a nibble:
 *   byte = ((high & 0x0F) << 4) | (low & 0x0F)
 * Unpacking sign-extends bit 3 of each nibble (see unpack_int4), matching the
 * packers in tensor.cpp / gptq.cpp / awq.cpp and int4_utils.hpp.
 */

#include <cstdint>
#include <algorithm>
#include <vector>

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

/// Unpack an INT4 weight byte into two signed int8 values.
/// Tenzor stores signed INT4 in two's-complement form masked to a nibble
/// (range [-8, 7], `value & 0xF`), matching every INT4 packer in the codebase
/// (tensor.cpp, gptq.cpp, awq.cpp) and int4_utils.hpp.  Recover the sign by
/// extending bit 3 into the high nibble — NOT an offset-8 (zero-point) decode.
static inline void unpack_int4(uint8_t packed, int8_t& low, int8_t& high) {
    int8_t lo = static_cast<int8_t>(packed & 0x0F);
    if (lo & 0x08) lo |= static_cast<int8_t>(0xF0);  // sign-extend
    int8_t hi = static_cast<int8_t>((packed >> 4) & 0x0F);
    if (hi & 0x08) hi |= static_cast<int8_t>(0xF0);  // sign-extend
    low  = lo;
    high = hi;
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

    // Pre-unpack each weight row to INT8 once, then compute the INT8 dot
    // product against every batch element. Pre-unpacking produces contiguous
    // weights aligned 1:1 with the input layout (unpacked_weights[k] pairs with
    // input_row[k]), so the SIMD inner loop is a straightforward widen+madd over
    // matching lanes.
    //
    // A previous batch_size==1 special case operated on packed nibbles directly
    // and mis-paired de-interleaved weight nibbles (w_low lane i = byte p+i) with
    // contiguous inputs (input_row[p*2 + i]); for in_features >= 16 packed
    // features this multiplied mismatched operands and returned silently wrong
    // results. It has been removed in favour of this single correct path, which
    // is SIMD-accelerated for all batch sizes.
    #pragma omp parallel for
    for (int64_t o = 0; o < out_features; ++o) {
        const uint8_t* weight_row = weight_packed + o * packed_features;

        // Per-(thread, o) heap buffer; ~in_features bytes (4KB for 4096), L1-resident.
        std::vector<int8_t> unpacked_weights(static_cast<size_t>(in_features));
        for (int64_t p = 0; p < packed_features; ++p) {
            unpack_int4(weight_row[p], unpacked_weights[p * 2], unpacked_weights[p * 2 + 1]);
        }

        for (int64_t b = 0; b < batch_size; ++b) {
            int32_t acc = 0;
            const int8_t* input_row = input + b * in_features;

#if defined(__AVX2__)
            __m256i acc_vec = _mm256_setzero_si256();
            int64_t k = 0;

            // Process 32 INT8 values at a time. Both input and the pre-unpacked
            // weights are contiguous and lane-aligned, so widen each 16-byte half
            // to int16 and accumulate with madd_epi16 (signed*signed).
            for (; k + 32 <= in_features; k += 32) {
                __m256i vi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_row + k));
                __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(unpacked_weights.data() + k));
                __m128i vi_lo = _mm256_castsi256_si128(vi);
                __m128i vi_hi = _mm256_extracti128_si256(vi, 1);
                __m128i vw_lo = _mm256_castsi256_si128(vw);
                __m128i vw_hi = _mm256_extracti128_si256(vw, 1);

                __m256i vi16_lo = _mm256_cvtepi8_epi16(vi_lo);
                __m256i vw16_lo = _mm256_cvtepi8_epi16(vw_lo);
                __m256i vi16_hi = _mm256_cvtepi8_epi16(vi_hi);
                __m256i vw16_hi = _mm256_cvtepi8_epi16(vw_hi);

                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(vi16_lo, vw16_lo));
                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(vi16_hi, vw16_hi));
            }

            // Horizontal sum
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc_vec),
                                           _mm256_extracti128_si256(acc_vec, 1));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 1, 0, 1)));
            acc = _mm_cvtsi128_si32(sum128);

            // Scalar tail
            for (; k < in_features; ++k) {
                acc += static_cast<int32_t>(input_row[k]) * static_cast<int32_t>(unpacked_weights[k]);
            }
#else
            for (int64_t k = 0; k < in_features; ++k) {
                acc += static_cast<int32_t>(input_row[k]) * static_cast<int32_t>(unpacked_weights[k]);
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
