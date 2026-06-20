/**
 * @file simd_elementwise.hpp
 * @brief Templated SIMD-accelerated elementwise operation helpers
 *
 * Provides a unified interface for applying scalar operations with
 * SIMD acceleration for float/double and vectorized widen-convert-narrow
 * for Float16 (F16C+AVX2) and BFloat16 (AVX2 bit-shifts).
 *
 * Usage:
 * @code
 * TENZOR_DISPATCH_FLOAT_AND_HALF(tensor.dtype(), "relu", [&]() {
 *     elementwise_unary<scalar_t>(
 *         input.data<scalar_t>(), output.data<scalar_t>(), n,
 *         [](scalar_t x) { return std::max(scalar_t(0), x); });
 * });
 * @endcode
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <omp.h>
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/omp_thresholds.hpp"
#include "bfloat16_simd.hpp"  // NaN-safe F32<->BF16 SIMD conversion helpers

#ifdef __AVX512F__
#define TENZOR_HAS_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define TENZOR_HAS_AVX2 1
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

// F16C scalar intrinsics header (needed for _cvtsh_ss / _cvtss_sh)
#if defined(__F16C__) && !defined(TENZOR_HAS_AVX512) && !defined(TENZOR_HAS_AVX2)
#include <immintrin.h>
#endif

namespace tenzor::cpu {

/// Threshold for OpenMP parallelization of elementwise ops.
/// Driven by ::tenzor::OmpThresholds::simple() (see tenzor/backend/omp_thresholds.hpp).
#define ELEMENTWISE_OMP_THRESHOLD (::tenzor::OmpThresholds::simple())

/**
 * @brief Apply a unary scalar operation elementwise with SIMD acceleration.
 *
 * For float/double: OpenMP parallelization for large tensors.
 * For Float16: vectorized widen (_mm256_cvtph_ps F16C) → op → narrow
 *   (_mm256_cvtps_ph), guarded by __F16C__ && __AVX2__.
 * For BFloat16: vectorized widen (AVX2 zero-extend + shift-left-16) → op
 *   → narrow (round-to-nearest-even + shift-right-16 + pack), guarded by __AVX2__.
 *
 * @tparam T Element type (float, double, Float16, BFloat16)
 * @tparam Op Unary operation callable: float(float)
 */
template<typename T, typename Op>
void elementwise_unary(const T* input, T* output, size_t n, Op op) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        if (n >= ELEMENTWISE_OMP_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                output[i] = op(input[i]);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                output[i] = op(input[i]);
            }
        }
    } else if constexpr (std::is_same_v<T, tenzor::Float16>) {
        const auto* in_u16 = reinterpret_cast<const uint16_t*>(input);
        auto* out_u16 = reinterpret_cast<uint16_t*>(output);
#if defined(__F16C__) && defined(__AVX2__)
        // Vectorized: 8 elements per iteration via F16C + AVX2
        auto process_chunk = [&](size_t start, size_t end) {
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __m128i packed = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(in_u16 + i));
                __m256 fp32 = _mm256_cvtph_ps(packed);
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, fp32);
                for (int j = 0; j < 8; ++j) tmp[j] = op(tmp[j]);
                __m256 out_v = _mm256_load_ps(tmp);
                __m128i out_packed = _mm256_cvtps_ph(
                    out_v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm_storeu_si128(
                    reinterpret_cast<__m128i*>(out_u16 + i), out_packed);
            }
            // Scalar tail using F16C scalar intrinsics
            for (; i < end; ++i) {
                float val = _cvtsh_ss(static_cast<unsigned short>(in_u16[i]));
                out_u16[i] = _cvtss_sh(op(val),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            }
        };
        if (n >= ELEMENTWISE_OMP_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t chunk = 0; chunk < n; chunk += 512) {
                process_chunk(chunk, std::min(chunk + 512, n));
            }
        } else {
            process_chunk(0, n);
        }
#else
        // Scalar fallback without F16C
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(input[i]);
            output[i] = static_cast<T>(op(val));
        }
#endif
    } else if constexpr (std::is_same_v<T, tenzor::BFloat16>) {
        const auto* in_u16 = reinterpret_cast<const uint16_t*>(input);
        auto* out_u16 = reinterpret_cast<uint16_t*>(output);
#if defined(__AVX2__)
        // Vectorized: 8 elements per iteration via AVX2 bit-shifts
        auto process_chunk = [&](size_t start, size_t end) {
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                // BF16 -> F32: zero-extend u16 to u32, shift left 16
                __m128i bf16_vec = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(in_u16 + i));
                __m256i extended = _mm256_cvtepu16_epi32(bf16_vec);
                __m256 fp32 = _mm256_castsi256_ps(
                    _mm256_slli_epi32(extended, 16));
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, fp32);
                for (int j = 0; j < 8; ++j) tmp[j] = op(tmp[j]);
                fp32 = _mm256_load_ps(tmp);
                // F32 -> BF16: use the shared NaN-safe converter (the inline
                // round-bias+truncate could turn some NaNs into Inf).
                bfloat16_simd::cvt_f32_to_bf16_avx2(fp32, out_u16 + i);
            }
            // Scalar tail
            for (; i < end; ++i) {
                float val = static_cast<float>(input[i]);
                output[i] = static_cast<T>(op(val));
            }
        };
        if (n >= ELEMENTWISE_OMP_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t chunk = 0; chunk < n; chunk += 512) {
                process_chunk(chunk, std::min(chunk + 512, n));
            }
        } else {
            process_chunk(0, n);
        }
#else
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(input[i]);
            output[i] = static_cast<T>(op(val));
        }
#endif
    } else {
        // Other types: scalar fallback
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(input[i]);
            output[i] = static_cast<T>(op(val));
        }
    }
}

/**
 * @brief Apply a binary scalar operation elementwise with SIMD acceleration.
 *
 * Same widen-operate-narrow strategy as elementwise_unary.
 *
 * @tparam T Element type
 * @tparam Op Binary operation callable: float(float, float)
 */
template<typename T, typename Op>
void elementwise_binary(const T* a, const T* b, T* output, size_t n, Op op) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        if (n >= ELEMENTWISE_OMP_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                output[i] = op(a[i], b[i]);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                output[i] = op(a[i], b[i]);
            }
        }
    } else if constexpr (std::is_same_v<T, tenzor::Float16>) {
        const auto* a_u16 = reinterpret_cast<const uint16_t*>(a);
        const auto* b_u16 = reinterpret_cast<const uint16_t*>(b);
        auto* out_u16 = reinterpret_cast<uint16_t*>(output);
#if defined(__F16C__) && defined(__AVX2__)
        auto process_chunk = [&](size_t start, size_t end) {
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __m128i pa = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(a_u16 + i));
                __m128i pb = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(b_u16 + i));
                __m256 fa = _mm256_cvtph_ps(pa);
                __m256 fb = _mm256_cvtph_ps(pb);
                alignas(32) float ta[8], tb[8], tc[8];
                _mm256_store_ps(ta, fa);
                _mm256_store_ps(tb, fb);
                for (int j = 0; j < 8; ++j) tc[j] = op(ta[j], tb[j]);
                __m256 fc = _mm256_load_ps(tc);
                __m128i out_packed = _mm256_cvtps_ph(
                    fc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm_storeu_si128(
                    reinterpret_cast<__m128i*>(out_u16 + i), out_packed);
            }
            for (; i < end; ++i) {
                float va = _cvtsh_ss(static_cast<unsigned short>(a_u16[i]));
                float vb = _cvtsh_ss(static_cast<unsigned short>(b_u16[i]));
                out_u16[i] = _cvtss_sh(op(va, vb),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            }
        };
        if (n >= ELEMENTWISE_OMP_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t chunk = 0; chunk < n; chunk += 512) {
                process_chunk(chunk, std::min(chunk + 512, n));
            }
        } else {
            process_chunk(0, n);
        }
#else
        for (size_t i = 0; i < n; ++i) {
            float va = static_cast<float>(a[i]);
            float vb = static_cast<float>(b[i]);
            output[i] = static_cast<T>(op(va, vb));
        }
#endif
    } else if constexpr (std::is_same_v<T, tenzor::BFloat16>) {
        const auto* a_u16 = reinterpret_cast<const uint16_t*>(a);
        const auto* b_u16 = reinterpret_cast<const uint16_t*>(b);
        auto* out_u16 = reinterpret_cast<uint16_t*>(output);
#if defined(__AVX2__)
        auto process_chunk = [&](size_t start, size_t end) {
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                auto bf16_to_f32 = [](const uint16_t* src) -> __m256 {
                    __m128i v = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(src));
                    __m256i ext = _mm256_cvtepu16_epi32(v);
                    return _mm256_castsi256_ps(_mm256_slli_epi32(ext, 16));
                };
                __m256 fa = bf16_to_f32(a_u16 + i);
                __m256 fb = bf16_to_f32(b_u16 + i);
                alignas(32) float ta[8], tb[8], tc[8];
                _mm256_store_ps(ta, fa);
                _mm256_store_ps(tb, fb);
                for (int j = 0; j < 8; ++j) tc[j] = op(ta[j], tb[j]);
                __m256 fc = _mm256_load_ps(tc);
                // F32 -> BF16: use the shared NaN-safe converter (the inline
                // round-bias+truncate could turn some NaNs into Inf).
                bfloat16_simd::cvt_f32_to_bf16_avx2(fc, out_u16 + i);
            }
            for (; i < end; ++i) {
                float va = static_cast<float>(a[i]);
                float vb = static_cast<float>(b[i]);
                output[i] = static_cast<T>(op(va, vb));
            }
        };
        if (n >= ELEMENTWISE_OMP_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t chunk = 0; chunk < n; chunk += 512) {
                process_chunk(chunk, std::min(chunk + 512, n));
            }
        } else {
            process_chunk(0, n);
        }
#else
        for (size_t i = 0; i < n; ++i) {
            float va = static_cast<float>(a[i]);
            float vb = static_cast<float>(b[i]);
            output[i] = static_cast<T>(op(va, vb));
        }
#endif
    } else {
        for (size_t i = 0; i < n; ++i) {
            float va = static_cast<float>(a[i]);
            float vb = static_cast<float>(b[i]);
            output[i] = static_cast<T>(op(va, vb));
        }
    }
}

} // namespace tenzor::cpu
