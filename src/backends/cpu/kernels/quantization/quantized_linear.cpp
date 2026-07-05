/**
 * @file quantized_linear.cpp
 * @brief CPU kernels for quantized linear operations
 */

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <vector>
#include <immintrin.h>  // For SIMD operations

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {

// AVX-512 signed-INT8 inner product.
//
// NOTE: We deliberately do NOT use _mm512_dpbusd_epi32 here. That VNNI
// intrinsic treats its first operand as *unsigned* 8-bit, which silently
// reinterprets any negative input byte (e.g. -1 -> 255) and produces wrong
// dot products for signed activations. The signed×signed VNNI variant
// (_mm512_dpbssd_epi32) only exists on AVX10 / AVX-VNNI-INT8, which we do
// not require. The portable signed-safe path widens int8 -> int16 via
// _mm512_cvtepi8_epi16 and uses _mm512_madd_epi16, mirroring the pattern
// in quantized_linear_int4.cpp. Throughput is lower than VNNI but
// correctness on negative activations is non-negotiable.
#if defined(__AVX512BW__)
// Returns the full int8xint8 dot product in int64 to match the CUDA kernel
// (quantized_linear.cu), which accumulates in int64 to avoid silent int32
// wraparound for very wide layers (in_features >~ 131072). Per-lane int32
// madd accumulation is safe (the row sum is spread across 16 lanes); the
// final horizontal reduction is widened to int64.
static inline int64_t dot_int8_signed_avx512(const int8_t* a, const int8_t* b, int64_t len) {
    __m512i acc_vec = _mm512_setzero_si512();
    int64_t i = 0;

    // Process 32 INT8 elements at a time: widen to INT16 (512-bit), then
    // _mm512_madd_epi16 (signed×signed -> int32 with horizontal pair-add).
    for (; i + 32 <= len; i += 32) {
        __m256i va8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m512i va16 = _mm512_cvtepi8_epi16(va8);
        __m512i vb16 = _mm512_cvtepi8_epi16(vb8);
        acc_vec = _mm512_add_epi32(acc_vec, _mm512_madd_epi16(va16, vb16));
    }

    // Widen the 16 int32 lanes to int64 before summing so the horizontal
    // reduction cannot overflow int32. Use AVX512F-only extracts (castsi512_si256
    // + extracti64x4) rather than extracti32x8 (which needs AVX512DQ) so this
    // stays within the __AVX512BW__ guard's feature set.
    __m256i lo32 = _mm512_castsi512_si256(acc_vec);
    __m256i hi32 = _mm512_extracti64x4_epi64(acc_vec, 1);
    __m512i lo64 = _mm512_cvtepi32_epi64(lo32);
    __m512i hi64 = _mm512_cvtepi32_epi64(hi32);
    int64_t acc = _mm512_reduce_add_epi64(_mm512_add_epi64(lo64, hi64));

    // Handle remainder
    for (; i < len; ++i) {
        acc += static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i]);
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

    // Precompute per-row sums for the asymmetric zero-point correction.
    // The dequantized dot product is sum_k (q_x - input_zp)(q_w - weight_zp)
    //   = raw_dot - input_zp*sum_w[o] - weight_zp*sum_x[b] + input_zp*weight_zp*K
    // so we need the per-output-row weight sum and per-batch-row input sum.
    std::vector<int64_t> sum_w(out_features, 0);
    std::vector<int64_t> sum_x(batch_size, 0);
    #pragma omp parallel for
    for (int64_t o = 0; o < out_features; ++o) {
        int64_t s = 0;
        const int8_t* wrow = weight + o * in_features;
        for (int64_t k = 0; k < in_features; ++k) s += static_cast<int64_t>(wrow[k]);
        sum_w[o] = s;
    }
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t s = 0;
        const int8_t* xrow = input + b * in_features;
        for (int64_t k = 0; k < in_features; ++k) s += static_cast<int64_t>(xrow[k]);
        sum_x[b] = s;
    }

    // Parallel over batch and output features
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            // int64 accumulator matches the CUDA kernel: each int8*int8 product
            // fits in int32 but the row sum can exceed INT32_MAX for very wide
            // layers, so accumulate in int64 to avoid silent wraparound.
            int64_t acc = 0;

            // Inner product with INT8 accumulation to INT32
            const int8_t* input_row = input + b * in_features;
            const int8_t* weight_row = weight + o * in_features;

#if defined(__AVX512BW__)
            // AVX-512 signed-safe path. Uses int8->int16 widening +
            // _mm512_madd_epi16 instead of VNNI _mm512_dpbusd_epi32
            // (the latter treats input as unsigned and corrupts negative
            // activations).
            acc = dot_int8_signed_avx512(input_row, weight_row, in_features);
#elif defined(__AVX2__)
            // SIMD-optimized path for x86 with AVX2.
            //
            // NOTE: We previously used _mm256_maddubs_epi16, which treats its
            // first operand as *unsigned* INT8. Since `input` is signed int8_t,
            // any negative byte (e.g. -1) aliased to 255 and silently produced
            // wrong accumulations. We now widen signed INT8 -> INT16 via
            // _mm256_cvtepi8_epi16 and use _mm256_madd_epi16 (signed*signed).
            // This mirrors quantized_linear_int4.cpp's signed-safe path.
            __m256i acc_vec = _mm256_setzero_si256();
            int64_t i = 0;

            // Process 32 elements at a time
            for (; i + 32 <= in_features; i += 32) {
                // Load 32 signed INT8 values
                __m256i input_vec = _mm256_loadu_si256((__m256i*)(input_row + i));
                __m256i weight_vec = _mm256_loadu_si256((__m256i*)(weight_row + i));

                // Split into low/high 128-bit halves and sign-extend to INT16
                __m128i in_lo = _mm256_castsi256_si128(input_vec);
                __m128i in_hi = _mm256_extracti128_si256(input_vec, 1);
                __m128i wt_lo = _mm256_castsi256_si128(weight_vec);
                __m128i wt_hi = _mm256_extracti128_si256(weight_vec, 1);

                __m256i in16_lo = _mm256_cvtepi8_epi16(in_lo);
                __m256i wt16_lo = _mm256_cvtepi8_epi16(wt_lo);
                __m256i in16_hi = _mm256_cvtepi8_epi16(in_hi);
                __m256i wt16_hi = _mm256_cvtepi8_epi16(wt_hi);

                // signed*signed -> int32 with horizontal pair-add
                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(in16_lo, wt16_lo));
                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(in16_hi, wt16_hi));
            }

            // Horizontal sum of accumulator
            __m128i sum128 = _mm_add_epi32(
                _mm256_castsi256_si128(acc_vec),
                _mm256_extracti128_si256(acc_vec, 1)
            );
            // Widen the 4 int32 lanes to int64 before summing so the horizontal
            // reduction cannot overflow int32 (matches the CUDA int64 accumulator).
            __m256i sum64 = _mm256_cvtepi32_epi64(sum128);
            __m128i s64 = _mm_add_epi64(
                _mm256_castsi256_si128(sum64),
                _mm256_extracti128_si256(sum64, 1)
            );
            acc = static_cast<int64_t>(_mm_extract_epi64(s64, 0)) +
                  static_cast<int64_t>(_mm_extract_epi64(s64, 1));

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

            // Asymmetric zero-point correction (full expansion, int64 to match CUDA).
            acc -= static_cast<int64_t>(input_zp) * sum_w[o];
            acc -= static_cast<int64_t>(weight_zp) * sum_x[b];
            acc += static_cast<int64_t>(input_zp) * static_cast<int64_t>(weight_zp) * in_features;

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

    // Per-row sums for the asymmetric zero-point correction (see notes in
    // quantized_linear_kernel above).
    std::vector<int64_t> sum_w(out_features, 0);
    std::vector<int64_t> sum_x(batch_size, 0);
    #pragma omp parallel for
    for (int64_t o = 0; o < out_features; ++o) {
        int64_t s = 0;
        const int8_t* wrow = weight + o * in_features;
        for (int64_t k = 0; k < in_features; ++k) s += static_cast<int64_t>(wrow[k]);
        sum_w[o] = s;
    }
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t s = 0;
        const int8_t* xrow = input + b * in_features;
        for (int64_t k = 0; k < in_features; ++k) s += static_cast<int64_t>(xrow[k]);
        sum_x[b] = s;
    }

    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            // int64 accumulator matches the CUDA kernel: each int8*int8 product
            // fits in int32 but the row sum can exceed INT32_MAX for very wide
            // layers, so accumulate in int64 to avoid silent wraparound.
            int64_t acc = 0;

            const int8_t* input_row = input + b * in_features;
            const int8_t* weight_row = weight + o * in_features;

#if defined(__AVX512BW__)
            // Signed-safe AVX-512 path (see comment on dot_int8_signed_avx512
            // above re: why we avoid VNNI _mm512_dpbusd_epi32).
            acc = dot_int8_signed_avx512(input_row, weight_row, in_features);
#elif defined(__AVX2__)
            // Signed-safe AVX2 path: widen int8 -> int16, then madd_epi16
            // (signed*signed). _mm256_maddubs_epi16 is *not* safe here because
            // its first operand is treated as unsigned.
            __m256i acc_vec = _mm256_setzero_si256();
            int64_t i = 0;

            for (; i + 32 <= in_features; i += 32) {
                __m256i input_vec = _mm256_loadu_si256((__m256i*)(input_row + i));
                __m256i weight_vec = _mm256_loadu_si256((__m256i*)(weight_row + i));

                __m128i in_lo = _mm256_castsi256_si128(input_vec);
                __m128i in_hi = _mm256_extracti128_si256(input_vec, 1);
                __m128i wt_lo = _mm256_castsi256_si128(weight_vec);
                __m128i wt_hi = _mm256_extracti128_si256(weight_vec, 1);

                __m256i in16_lo = _mm256_cvtepi8_epi16(in_lo);
                __m256i wt16_lo = _mm256_cvtepi8_epi16(wt_lo);
                __m256i in16_hi = _mm256_cvtepi8_epi16(in_hi);
                __m256i wt16_hi = _mm256_cvtepi8_epi16(wt_hi);

                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(in16_lo, wt16_lo));
                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(in16_hi, wt16_hi));
            }

            __m128i sum128 = _mm_add_epi32(
                _mm256_castsi256_si128(acc_vec),
                _mm256_extracti128_si256(acc_vec, 1)
            );
            // Widen the 4 int32 lanes to int64 before summing so the horizontal
            // reduction cannot overflow int32 (matches the CUDA int64 accumulator).
            __m256i sum64 = _mm256_cvtepi32_epi64(sum128);
            __m128i s64 = _mm_add_epi64(
                _mm256_castsi256_si128(sum64),
                _mm256_extracti128_si256(sum64, 1)
            );
            acc = static_cast<int64_t>(_mm_extract_epi64(s64, 0)) +
                  static_cast<int64_t>(_mm_extract_epi64(s64, 1));

            for (; i < in_features; ++i) {
                acc += static_cast<int32_t>(input_row[i]) * static_cast<int32_t>(weight_row[i]);
            }
#else
            for (int64_t i = 0; i < in_features; ++i) {
                acc += static_cast<int32_t>(input_row[i]) * static_cast<int32_t>(weight_row[i]);
            }
#endif

            // Per-channel asymmetric zero-point correction (full expansion, int64 to match CUDA).
            int32_t w_zp = weight_zps ? weight_zps[o] : 0;
            acc -= static_cast<int64_t>(input_zp) * sum_w[o];
            acc -= static_cast<int64_t>(w_zp) * sum_x[b];
            acc += static_cast<int64_t>(input_zp) * static_cast<int64_t>(w_zp) * in_features;

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
