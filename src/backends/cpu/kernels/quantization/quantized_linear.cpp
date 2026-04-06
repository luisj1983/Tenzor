/**
 * @file quantized_linear.cpp
 * @brief CPU kernels for quantized linear operations
 */

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <immintrin.h>  // For SIMD operations

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {

// AVX512-VNNI inner product: uses _mm512_dpbusd_epi32 for 4x throughput
#if defined(__AVX512VNNI__)
static inline int32_t dot_int8_vnni(const int8_t* a, const int8_t* b, int64_t len) {
    __m512i acc_vec = _mm512_setzero_si512();
    int64_t i = 0;

    // Process 64 elements at a time with VNNI
    for (; i + 64 <= len; i += 64) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        acc_vec = _mm512_dpbusd_epi32(acc_vec, va, vb);
    }

    int32_t acc = _mm512_reduce_add_epi32(acc_vec);

    // Handle remainder
    for (; i < len; ++i) {
        acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return acc;
}
#endif

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

    // Verify alignment expectations for SIMD loads. On Haswell+ CPUs, unaligned
    // loads (_mm256_loadu_si256 / _mm512_loadu_si512) have the same throughput as
    // aligned loads when data IS aligned. The allocator provides 256-byte aligned
    // buffers, so this should always pass. If it fails, the kernel still works
    // correctly via unaligned loads, just potentially slower on pre-Haswell.
    {
        static std::once_flag align_warn_flag;
        bool input_misaligned = (reinterpret_cast<uintptr_t>(input) % 32 != 0);
        bool weight_misaligned = (reinterpret_cast<uintptr_t>(weight) % 32 != 0);
        if (input_misaligned || weight_misaligned) [[unlikely]] {
            std::call_once(align_warn_flag, [input_misaligned, weight_misaligned]() {
                if (input_misaligned) {
                    fprintf(stderr, "[tenzor] warning: quantized_linear_kernel: "
                            "input pointer not 32-byte aligned; "
                            "performance may be degraded on pre-Haswell CPUs\n");
                }
                if (weight_misaligned) {
                    fprintf(stderr, "[tenzor] warning: quantized_linear_kernel: "
                            "weight pointer not 32-byte aligned; "
                            "performance may be degraded on pre-Haswell CPUs\n");
                }
            });
        }
    }

    // Parallel over batch and output features
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            int32_t acc = 0;

            // Inner product with INT8 accumulation to INT32
            const int8_t* input_row = input + b * in_features;
            const int8_t* weight_row = weight + o * in_features;

#if defined(__AVX512VNNI__)
            // AVX512-VNNI path: ~4x throughput over AVX2
            acc = dot_int8_vnni(input_row, weight_row, in_features);
#elif defined(__AVX2__)
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

/**
 * @brief Per-channel quantized matrix multiplication for linear layer (CPU).
 *
 * Each output channel has its own weight_scale and weight_zp, enabling
 * higher accuracy quantization for weights with varying magnitude ranges.
 *
 * output[b][o] = (dot(input[b], weight[o]) - zp_correction[o]) * (input_scale * weight_scales[o] / output_scale) + bias[o]
 */
auto quantized_linear_per_channel_kernel(
    const int8_t* input,
    const int8_t* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    float input_scale,
    const float* weight_scales,     // [out_features] per-channel scales
    float output_scale,
    int32_t input_zp,
    const int32_t* weight_zps       // [out_features] per-channel zero points
) -> void {

    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            int32_t acc = 0;

            const int8_t* input_row = input + b * in_features;
            const int8_t* weight_row = weight + o * in_features;

#if defined(__AVX512VNNI__)
            acc = dot_int8_vnni(input_row, weight_row, in_features);
#elif defined(__AVX2__)
            __m256i acc_vec = _mm256_setzero_si256();
            int64_t i = 0;

            for (; i + 32 <= in_features; i += 32) {
                __m256i input_vec = _mm256_loadu_si256((__m256i*)(input_row + i));
                __m256i weight_vec = _mm256_loadu_si256((__m256i*)(weight_row + i));
                __m256i prod_lo = _mm256_maddubs_epi16(input_vec, weight_vec);
                __m256i prod_hi = _mm256_madd_epi16(prod_lo, _mm256_set1_epi16(1));
                acc_vec = _mm256_add_epi32(acc_vec, prod_hi);
            }

            __m128i sum128 = _mm_add_epi32(
                _mm256_castsi256_si128(acc_vec),
                _mm256_extracti128_si256(acc_vec, 1)
            );
            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            acc = _mm_cvtsi128_si32(sum128);

            for (; i < in_features; ++i) {
                acc += static_cast<int32_t>(input_row[i]) * static_cast<int32_t>(weight_row[i]);
            }
#else
            for (int64_t i = 0; i < in_features; ++i) {
                acc += static_cast<int32_t>(input_row[i]) * static_cast<int32_t>(weight_row[i]);
            }
#endif

            // Per-channel zero point correction and dequantization
            int32_t w_zp = weight_zps ? weight_zps[o] : 0;
            acc -= input_zp * w_zp * in_features;

            float combined_scale = input_scale * weight_scales[o] / output_scale;
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
