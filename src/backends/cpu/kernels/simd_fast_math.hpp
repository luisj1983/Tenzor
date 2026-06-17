/**
 * @file simd_fast_math.hpp
 * @brief High-performance SIMD implementations of transcendental functions
 *
 * Provides vectorized exp, log, tanh, sigmoid using polynomial approximations
 * with full AVX2 and AVX-512 support. Achieves 10-20x speedup over std::exp/log.
 *
 * Accuracy: < 2 ULP error for exp, ~10-20 ULP error for log in normal range
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include "tenzor/backend/omp_thresholds.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #if defined(__AVX512F__)
        #define TENZOR_FAST_MATH_AVX512
    #endif
    #if defined(__AVX2__)
        #define TENZOR_FAST_MATH_AVX2
    #endif
    #if defined(__FMA__)
        #define TENZOR_FAST_MATH_FMA
    #endif
    #if defined(__F16C__)
        #define TENZOR_FAST_MATH_F16C
    #endif
#endif

namespace tenzor {
namespace cpu {
namespace fast_math {

// ============================================================================
// Constants for exp approximation (Cody-Waite range reduction)
// ============================================================================

// log2(e) for range reduction
constexpr float LOG2E = 1.44269504088896341f;
// ln(2) split for precision (hi + lo = ln(2))
constexpr float LN2_HI = 0.693145751953125f;
constexpr float LN2_LO = 1.42860682030941723212e-6f;

// Polynomial coefficients for exp(r) where r ∈ [-ln(2)/2, ln(2)/2]
// Minimax polynomial degree 6, error < 2 ULP
constexpr float EXP_C0 = 1.0f;
constexpr float EXP_C1 = 1.0f;
constexpr float EXP_C2 = 0.5f;
constexpr float EXP_C3 = 0.166666666666666666667f;
constexpr float EXP_C4 = 0.0416666666666666666667f;
constexpr float EXP_C5 = 0.00833333333333333333333f;
constexpr float EXP_C6 = 0.00138888888888888888889f;

// ============================================================================
// Constants for log approximation
// ============================================================================

constexpr float LOG_C1 = -0.5f;
constexpr float LOG_C2 = 0.333333333333333333333f;
constexpr float LOG_C3 = -0.25f;
constexpr float LOG_C4 = 0.2f;
constexpr float LOG_C5 = -0.166666666666666666667f;

// ============================================================================
// Constants for tanh approximation
// ============================================================================

// Tanh polynomial coefficients (rational approximation)
constexpr float TANH_CLAMP = 9.0f;  // |x| > 9 → tanh(x) ≈ ±1

// ============================================================================
// AVX2 Implementations (256-bit, 8 floats)
// ============================================================================

#ifdef TENZOR_FAST_MATH_AVX2

/**
 * @brief AVX2 vectorized exp using Cody-Waite range reduction
 *
 * Algorithm:
 * 1. Range reduce: x = k*ln(2) + r, where k = round(x*log2(e))
 * 2. Compute exp(r) using polynomial (r is small)
 * 3. Reconstruct: exp(x) = 2^k * exp(r)
 */
inline __m256 exp_avx2(__m256 x) {
    // Constants
    __m256 log2e = _mm256_set1_ps(LOG2E);
    __m256 ln2_hi = _mm256_set1_ps(LN2_HI);
    __m256 ln2_lo = _mm256_set1_ps(LN2_LO);

    __m256 c1 = _mm256_set1_ps(EXP_C1);
    __m256 c2 = _mm256_set1_ps(EXP_C2);
    __m256 c3 = _mm256_set1_ps(EXP_C3);
    __m256 c4 = _mm256_set1_ps(EXP_C4);
    __m256 c5 = _mm256_set1_ps(EXP_C5);
    __m256 c6 = _mm256_set1_ps(EXP_C6);

    // Clamp to avoid overflow/underflow
    __m256 max_val = _mm256_set1_ps(88.3762626647949f);
    __m256 min_val = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(_mm256_max_ps(x, min_val), max_val);

    // Range reduction: k = round(x * log2(e))
    __m256 k = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // r = x - k * ln(2) using Cody-Waite for precision
#ifdef TENZOR_FAST_MATH_FMA
    __m256 r = _mm256_fnmadd_ps(k, ln2_hi, x);
    r = _mm256_fnmadd_ps(k, ln2_lo, r);
#else
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(k, ln2_hi));
    r = _mm256_sub_ps(r, _mm256_mul_ps(k, ln2_lo));
#endif

    // Polynomial approximation: exp(r) ≈ 1 + r + r²/2 + r³/6 + ...
    // Using Horner's method for efficiency
#ifdef TENZOR_FAST_MATH_FMA
    __m256 p = _mm256_fmadd_ps(c6, r, c5);
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
#else
    __m256 p = _mm256_add_ps(_mm256_mul_ps(c6, r), c5);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c4);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c3);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c2);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), c1);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.0f));
#endif

    // Reconstruct: exp(x) = 2^k * exp(r)
    // Convert k to integer and add to exponent field
    __m256i ki = _mm256_cvtps_epi32(k);
    ki = _mm256_slli_epi32(ki, 23);  // Shift to exponent position
    __m256 scale = _mm256_castsi256_ps(_mm256_add_epi32(ki, _mm256_set1_epi32(0x3f800000)));

    return _mm256_mul_ps(p, scale);
}

/**
 * @brief AVX2 vectorized natural log using polynomial approximation
 * Handles edge cases: log(x<0) = NaN, log(0) = -inf
 */
inline __m256 log_avx2(__m256 x) {
    // Handle edge cases: negative -> NaN, zero -> -inf
    __m256 zero = _mm256_setzero_ps();
    __m256 neg_mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);  // x < 0
    __m256 zero_mask = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ); // x == 0

    // Create NaN and -inf constants
    __m256 nan_val = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
    __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

    // Extract exponent and mantissa
    __m256i xi = _mm256_castps_si256(x);
    __m256i exponent = _mm256_srli_epi32(xi, 23);
    exponent = _mm256_sub_epi32(exponent, _mm256_set1_epi32(127));
    __m256 e = _mm256_cvtepi32_ps(exponent);

    // Normalize mantissa to [1, 2)
    __m256i mantissa = _mm256_and_si256(xi, _mm256_set1_epi32(0x007fffff));
    mantissa = _mm256_or_si256(mantissa, _mm256_set1_epi32(0x3f800000));
    __m256 m = _mm256_castsi256_ps(mantissa);

    // If m > sqrt(2), adjust
    __m256 sqrt2 = _mm256_set1_ps(1.41421356237f);
    __m256 mask = _mm256_cmp_ps(m, sqrt2, _CMP_GT_OQ);
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), mask);
    e = _mm256_blendv_ps(e, _mm256_add_ps(e, _mm256_set1_ps(1.0f)), mask);

    // log(m) where m ∈ [1, sqrt(2)] using polynomial
    // Let u = (m - 1) / (m + 1), then log(m) = 2 * (u + u³/3 + u⁵/5 + ...)
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 u = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
    __m256 u2 = _mm256_mul_ps(u, u);

    // Polynomial in u²: log(m) = 2u (1 + u²/3 + u⁴/5 + u⁶/7 + u⁸/9 + ...).
    // Carry the series out to the u⁸/9 term so the vectorized log matches
    // std::log to within ~1 ULP over m ∈ [1/sqrt2, sqrt2] (|u| < ~0.172),
    // preventing false gradcheck failures on log / log_softmax / pow.
    __m256 p = _mm256_set1_ps(0.111111111111f);  // 1/9
#ifdef TENZOR_FAST_MATH_FMA
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(0.142857142857f));   // 1/7
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(0.2f));              // 1/5
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(0.333333333333f));   // 1/3
    p = _mm256_fmadd_ps(p, u2, one);
#else
    p = _mm256_add_ps(_mm256_mul_ps(p, u2), _mm256_set1_ps(0.142857142857f));
    p = _mm256_add_ps(_mm256_mul_ps(p, u2), _mm256_set1_ps(0.2f));
    p = _mm256_add_ps(_mm256_mul_ps(p, u2), _mm256_set1_ps(0.333333333333f));
    p = _mm256_add_ps(_mm256_mul_ps(p, u2), one);
#endif

    __m256 log_m = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), u), p);

    // log(x) = e * ln(2) + log(m)
    __m256 ln2 = _mm256_set1_ps(0.693147180559945f);
    __m256 result;
#ifdef TENZOR_FAST_MATH_FMA
    result = _mm256_fmadd_ps(e, ln2, log_m);
#else
    result = _mm256_add_ps(_mm256_mul_ps(e, ln2), log_m);
#endif

    // Apply edge case handling: negative -> NaN, zero -> -inf
    result = _mm256_blendv_ps(result, neg_inf, zero_mask);
    result = _mm256_blendv_ps(result, nan_val, neg_mask);

    return result;
}

/**
 * @brief AVX2 vectorized tanh using exp
 * tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
 */
inline __m256 tanh_avx2(__m256 x) {
    // Clamp for numerical stability
    __m256 clamp = _mm256_set1_ps(TANH_CLAMP);
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_sub_ps(_mm256_setzero_ps(), clamp)), clamp);

    // Compute exp(2x)
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 exp2x = exp_avx2(_mm256_mul_ps(two, x));

    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 num = _mm256_sub_ps(exp2x, one);
    __m256 den = _mm256_add_ps(exp2x, one);

    return _mm256_div_ps(num, den);
}

/**
 * @brief AVX2 vectorized sigmoid
 * sigmoid(x) = 1 / (1 + exp(-x))
 */
inline __m256 sigmoid_avx2(__m256 x) {
    // Numerically stable: for x < 0, use exp(x)/(1+exp(x))
    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);

    // Compute exp(-|x|)
    __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    __m256 neg_abs_x = _mm256_sub_ps(zero, abs_x);
    __m256 exp_neg_abs = exp_avx2(neg_abs_x);

    // sigmoid(-|x|) = exp(-|x|) / (1 + exp(-|x|))
    __m256 sig_neg = _mm256_div_ps(exp_neg_abs, _mm256_add_ps(one, exp_neg_abs));

    // sigmoid(|x|) = 1 - sigmoid(-|x|)
    __m256 sig_pos = _mm256_sub_ps(one, sig_neg);

    // Select based on sign of x
    __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);
    return _mm256_blendv_ps(sig_neg, sig_pos, mask);
}

/**
/**
 * @brief AVX2 vectorized pow for positive bases: x^y = exp(y * log(x))
 *
 * WARNING: This function only produces correct results for x > 0.
 * Negative bases are clamped to 1e-30 (treated as ~0), which silently
 * returns ~0 instead of the mathematically correct result.  For integer
 * exponents with negative bases (e.g. (-2)^3 = -8), use std::pow or a
 * dedicated signed-pow implementation instead.
 *
 * Accuracy: ~1-2 ULP for positive x in normal float range.
 */
inline __m256 pow_avx2(__m256 x, __m256 y) {
    __m256 min_val = _mm256_set1_ps(1e-30f);
    x = _mm256_max_ps(x, min_val);
    __m256 log_x = log_avx2(x);
    __m256 y_log_x = _mm256_mul_ps(y, log_x);
    return exp_avx2(y_log_x);
}

inline __m256 pow_avx2(__m256 x, float y) {
    return pow_avx2(x, _mm256_set1_ps(y));
}

/**
 * @brief AVX2 vectorized sin using polynomial approximation.
 *
 * Range-reduces to [-π/2, π/2] using k = round(x/π), x' = x - k·π,
 * then applies a degree-9 Taylor series for sin centered at 0. On the
 * reduced interval the truncation error is bounded by (π/2)^11/11! ≈
 * 9e-6, well within Float32 epsilon. Using [-π, π] reduction instead
 * (as an earlier version did) produced ~2e-3 error near ±π because the
 * Taylor polynomial diverges that far out.
 *
 * sin(x + k·π) = (-1)^k · sin(x), so we flip the sign of the result
 * when k is odd.
 */
inline __m256 sin_avx2(__m256 x) {
    const __m256 x_orig = x;  // retained for the large-argument scalar fallback
    __m256 inv_pi = _mm256_set1_ps(0.318309886183791f);   // 1/π
    __m256 pi     = _mm256_set1_ps(3.14159265358979f);    // π

    // k = round(x / π) — nearest-even rounding to an integer.
    __m256 kf = _mm256_round_ps(_mm256_mul_ps(x, inv_pi),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // x <- x - k·π, now in [-π/2, π/2].
#ifdef TENZOR_FAST_MATH_FMA
    x = _mm256_fnmadd_ps(kf, pi, x);
#else
    x = _mm256_sub_ps(x, _mm256_mul_ps(kf, pi));
#endif

    // Degree-11 Taylor series for sin about 0. On [-π/2, π/2] the
    // truncation error is bounded by (π/2)^13/13! ≈ 5.7e-8, well
    // within Float32 epsilon. Degree-9 alone was ~3e-6 at the edge,
    // which matters for precision-sensitive comparisons.
    __m256 x2 = _mm256_mul_ps(x, x);

    __m256 c3  = _mm256_set1_ps(-0.16666666666f);    // -1/6
    __m256 c5  = _mm256_set1_ps(0.00833333333f);     // 1/120
    __m256 c7  = _mm256_set1_ps(-0.00019841270f);    // -1/5040
    __m256 c9  = _mm256_set1_ps(0.0000027557319f);   // 1/362880
    __m256 c11 = _mm256_set1_ps(-2.5052108e-8f);     // -1/39916800

#ifdef TENZOR_FAST_MATH_FMA
    __m256 p = _mm256_fmadd_ps(c11, x2, c9);
    p = _mm256_fmadd_ps(p, x2, c7);
    p = _mm256_fmadd_ps(p, x2, c5);
    p = _mm256_fmadd_ps(p, x2, c3);
    p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(1.0f));
    __m256 result = _mm256_mul_ps(p, x);
#else
    __m256 p = _mm256_add_ps(_mm256_mul_ps(c11, x2), c9);
    p = _mm256_add_ps(_mm256_mul_ps(p, x2), c7);
    p = _mm256_add_ps(_mm256_mul_ps(p, x2), c5);
    p = _mm256_add_ps(_mm256_mul_ps(p, x2), c3);
    p = _mm256_add_ps(_mm256_mul_ps(p, x2), _mm256_set1_ps(1.0f));
    __m256 result = _mm256_mul_ps(p, x);
#endif

    // Flip the sign of lanes where k is odd: sin(x + k·π) = (-1)^k · sin(x').
    // Convert kf to integer, mask the low bit, and produce a sign mask.
    __m256i ki = _mm256_cvtps_epi32(kf);
    __m256i one = _mm256_set1_epi32(1);
    __m256i odd = _mm256_and_si256(ki, one);                // 0 or 1
    // Build a lane mask that's all-ones when odd, all-zero when even.
    __m256i odd_mask = _mm256_cmpeq_epi32(odd, one);
    // sign_bit = 0x80000000 where odd, 0 elsewhere — XOR-in the sign.
    __m256i sign_bit = _mm256_and_si256(odd_mask, _mm256_set1_epi32(0x80000000));
    result = _mm256_xor_ps(result, _mm256_castsi256_ps(sign_bit));

    // Large-argument safety: for |x| above ~2^22·π the k = round(x/π) value no
    // longer fits reliably in int32 (_mm256_cvtps_epi32 returns 0x80000000 for
    // out-of-range inputs, which would corrupt the odd/even sign), and the
    // single-step Cody-Waite reduction loses too much precision. Fall back to
    // scalar std::sin for exactly those lanes; the fast path is unchanged for
    // normal magnitudes.
    constexpr float kSinCosSafe = 13176794.0f;  // ~2^22 · π
    __m256 abs_x_orig = _mm256_and_ps(x_orig, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    __m256 big_mask = _mm256_cmp_ps(abs_x_orig, _mm256_set1_ps(kSinCosSafe), _CMP_GT_OQ);
    if (_mm256_movemask_ps(big_mask) != 0) {
        alignas(32) float in_lanes[8];
        alignas(32) float out_lanes[8];
        _mm256_store_ps(in_lanes, x_orig);
        _mm256_store_ps(out_lanes, result);
        for (int lane = 0; lane < 8; ++lane) {
            if (std::abs(in_lanes[lane]) > kSinCosSafe) {
                out_lanes[lane] = std::sin(in_lanes[lane]);
            }
        }
        result = _mm256_load_ps(out_lanes);
    }
    return result;
}

/**
 * @brief AVX2 vectorized cos using polynomial approximation
 * cos(x) = sin(x + π/2)
 */
inline __m256 cos_avx2(__m256 x) {
    __m256 pi_2 = _mm256_set1_ps(1.57079632679f);  // π/2
    return sin_avx2(_mm256_add_ps(x, pi_2));
}

/**
 * @brief AVX2 vectorized leaky ReLU
 * leaky_relu(x, alpha) = x if x > 0 else alpha * x
 */
inline __m256 leaky_relu_avx2(__m256 x, float alpha = 0.01f) {
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 zero = _mm256_setzero_ps();
    __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
    __m256 scaled = _mm256_mul_ps(x, alpha_vec);
    return _mm256_blendv_ps(scaled, x, mask);
}

// NOTE: a tanh-approximation GELU helper (gelu_avx2) was removed here: it was
// dead code (no callers) and described the tanh approximation, whereas every
// live GELU path in the codebase computes the EXACT erf GELU
// 0.5*x*(1+erf(x/sqrt2)). Keeping it risked confusion about which formula the
// backend uses. Re-add a dedicated approximate='tanh' kernel only when wired up.

#endif // TENZOR_FAST_MATH_AVX2

// ============================================================================
// AVX-512 Implementations (512-bit, 16 floats)
// ============================================================================

#ifdef TENZOR_FAST_MATH_AVX512

/**
 * @brief AVX-512 vectorized exp
 */
inline __m512 exp_avx512(__m512 x) {
    __m512 log2e = _mm512_set1_ps(LOG2E);
    __m512 ln2_hi = _mm512_set1_ps(LN2_HI);
    __m512 ln2_lo = _mm512_set1_ps(LN2_LO);

    __m512 c1 = _mm512_set1_ps(EXP_C1);
    __m512 c2 = _mm512_set1_ps(EXP_C2);
    __m512 c3 = _mm512_set1_ps(EXP_C3);
    __m512 c4 = _mm512_set1_ps(EXP_C4);
    __m512 c5 = _mm512_set1_ps(EXP_C5);
    __m512 c6 = _mm512_set1_ps(EXP_C6);

    // Clamp
    __m512 max_val = _mm512_set1_ps(88.3762626647949f);
    __m512 min_val = _mm512_set1_ps(-88.3762626647949f);
    x = _mm512_min_ps(_mm512_max_ps(x, min_val), max_val);

    // Range reduction
    __m512 k = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT);

    __m512 r = _mm512_fnmadd_ps(k, ln2_hi, x);
    r = _mm512_fnmadd_ps(k, ln2_lo, r);

    // Polynomial
    __m512 p = _mm512_fmadd_ps(c6, r, c5);
    p = _mm512_fmadd_ps(p, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, c1);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));

    // Scale by 2^k
    __m512i ki = _mm512_cvtps_epi32(k);
    ki = _mm512_slli_epi32(ki, 23);
    __m512 scale = _mm512_castsi512_ps(_mm512_add_epi32(ki, _mm512_set1_epi32(0x3f800000)));

    return _mm512_mul_ps(p, scale);
}

/**
 * @brief AVX-512 vectorized log
 * Handles edge cases: log(x<0) = NaN, log(0) = -inf
 */
inline __m512 log_avx512(__m512 x) {
    // Handle edge cases: negative -> NaN, zero -> -inf
    __m512 zero = _mm512_setzero_ps();
    __mmask16 neg_mask = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);  // x < 0
    __mmask16 zero_mask = _mm512_cmp_ps_mask(x, zero, _CMP_EQ_OQ); // x == 0

    // Create NaN and -inf constants
    __m512 nan_val = _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN());
    __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());

    __m512i xi = _mm512_castps_si512(x);
    __m512i exponent = _mm512_srli_epi32(xi, 23);
    exponent = _mm512_sub_epi32(exponent, _mm512_set1_epi32(127));
    __m512 e = _mm512_cvtepi32_ps(exponent);

    __m512i mantissa = _mm512_and_epi32(xi, _mm512_set1_epi32(0x007fffff));
    mantissa = _mm512_or_epi32(mantissa, _mm512_set1_epi32(0x3f800000));
    __m512 m = _mm512_castsi512_ps(mantissa);

    __m512 sqrt2 = _mm512_set1_ps(1.41421356237f);
    __mmask16 mask = _mm512_cmp_ps_mask(m, sqrt2, _CMP_GT_OQ);
    m = _mm512_mask_mul_ps(m, mask, m, _mm512_set1_ps(0.5f));
    e = _mm512_mask_add_ps(e, mask, e, _mm512_set1_ps(1.0f));

    __m512 one = _mm512_set1_ps(1.0f);
    __m512 u = _mm512_div_ps(_mm512_sub_ps(m, one), _mm512_add_ps(m, one));
    __m512 u2 = _mm512_mul_ps(u, u);

    // log(m) = 2u (1 + u²/3 + u⁴/5 + u⁶/7 + u⁸/9 + ...). Carry the series out
    // to the u⁸/9 term so the vectorized log matches std::log to within ~1 ULP
    // over the reduced range, preventing false gradcheck failures.
    __m512 p = _mm512_set1_ps(0.111111111111f);  // 1/9
    p = _mm512_fmadd_ps(p, u2, _mm512_set1_ps(0.142857142857f));   // 1/7
    p = _mm512_fmadd_ps(p, u2, _mm512_set1_ps(0.2f));              // 1/5
    p = _mm512_fmadd_ps(p, u2, _mm512_set1_ps(0.333333333333f));   // 1/3
    p = _mm512_fmadd_ps(p, u2, one);

    __m512 log_m = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(2.0f), u), p);

    __m512 ln2 = _mm512_set1_ps(0.693147180559945f);
    __m512 result = _mm512_fmadd_ps(e, ln2, log_m);

    // Apply edge case handling: negative -> NaN, zero -> -inf
    result = _mm512_mask_blend_ps(zero_mask, result, neg_inf);
    result = _mm512_mask_blend_ps(neg_mask, result, nan_val);

    return result;
}

/**
 * @brief AVX-512 vectorized tanh
 */
inline __m512 tanh_avx512(__m512 x) {
    __m512 clamp = _mm512_set1_ps(TANH_CLAMP);
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_sub_ps(_mm512_setzero_ps(), clamp)), clamp);

    __m512 two = _mm512_set1_ps(2.0f);
    __m512 exp2x = exp_avx512(_mm512_mul_ps(two, x));

    __m512 one = _mm512_set1_ps(1.0f);
    __m512 num = _mm512_sub_ps(exp2x, one);
    __m512 den = _mm512_add_ps(exp2x, one);

    return _mm512_div_ps(num, den);
}

/**
 * @brief AVX-512 vectorized sigmoid
 */
inline __m512 sigmoid_avx512(__m512 x) {
    __m512 zero = _mm512_setzero_ps();
    __m512 one = _mm512_set1_ps(1.0f);

    __m512 abs_x = _mm512_abs_ps(x);
    __m512 neg_abs_x = _mm512_sub_ps(zero, abs_x);
    __m512 exp_neg_abs = exp_avx512(neg_abs_x);

    __m512 sig_neg = _mm512_div_ps(exp_neg_abs, _mm512_add_ps(one, exp_neg_abs));
    __m512 sig_pos = _mm512_sub_ps(one, sig_neg);

    __mmask16 mask = _mm512_cmp_ps_mask(x, zero, _CMP_GE_OQ);
    return _mm512_mask_blend_ps(mask, sig_neg, sig_pos);
}

// NOTE: a tanh-approximation GELU helper (gelu_avx512) was removed here for the
// same reason as gelu_avx2 above: dead code describing the tanh approximation
// while the live GELU paths all compute the exact erf GELU.

/**
 * @brief AVX-512 vectorized pow: x^y = exp(y * log(x))
 */
inline __m512 pow_avx512(__m512 x, __m512 y) {
    __m512 min_val = _mm512_set1_ps(1e-30f);
    x = _mm512_max_ps(x, min_val);

    __m512 log_x = log_avx512(x);
    __m512 y_log_x = _mm512_mul_ps(y, log_x);
    return exp_avx512(y_log_x);
}

/**
 * @brief AVX-512 vectorized pow with scalar exponent
 */
inline __m512 pow_avx512(__m512 x, float y) {
    return pow_avx512(x, _mm512_set1_ps(y));
}

/**
 * @brief AVX-512 vectorized sin.
 *
 * Range-reduces to [-π/2, π/2] via k = round(x/π), then applies a
 * degree-9 Taylor series centered at 0 and flips sign on odd k. The
 * earlier [-π, π] reduction left ~2e-3 error near ±π (the Taylor
 * series diverges that far out); this reduction bounds the error at
 * ~9e-6 on the whole domain.
 */
inline __m512 sin_avx512(__m512 x) {
    const __m512 x_orig = x;  // retained for the large-argument scalar fallback
    __m512 inv_pi = _mm512_set1_ps(0.318309886183791f);  // 1/π
    __m512 pi     = _mm512_set1_ps(3.14159265358979f);   // π

    __m512 kf = _mm512_roundscale_ps(_mm512_mul_ps(x, inv_pi),
                                     _MM_FROUND_TO_NEAREST_INT);
    x = _mm512_fnmadd_ps(kf, pi, x);

    __m512 x2 = _mm512_mul_ps(x, x);

    __m512 c3  = _mm512_set1_ps(-0.16666666666f);
    __m512 c5  = _mm512_set1_ps(0.00833333333f);
    __m512 c7  = _mm512_set1_ps(-0.00019841270f);
    __m512 c9  = _mm512_set1_ps(0.0000027557319f);
    __m512 c11 = _mm512_set1_ps(-2.5052108e-8f);

    __m512 p = _mm512_fmadd_ps(c11, x2, c9);
    p = _mm512_fmadd_ps(p, x2, c7);
    p = _mm512_fmadd_ps(p, x2, c5);
    p = _mm512_fmadd_ps(p, x2, c3);
    p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(1.0f));
    __m512 result = _mm512_mul_ps(p, x);

    // Flip the sign of lanes where k is odd: sin(x + k·π) = (-1)^k · sin(x').
    __m512i ki = _mm512_cvtps_epi32(kf);
    __m512i odd_mask = _mm512_and_si512(ki, _mm512_set1_epi32(1));
    __mmask16 odd_lanes =
        _mm512_cmpeq_epi32_mask(odd_mask, _mm512_set1_epi32(1));
    // XOR in the sign bit for odd lanes.
    __m512 sign_flip = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));
    result = _mm512_mask_xor_ps(result, odd_lanes, result, sign_flip);

    // Large-argument safety: for |x| above ~2^22·π the k = round(x/π) value no
    // longer fits reliably in int32 (_mm512_cvtps_epi32 saturates/returns the
    // integer indefinite, corrupting the odd/even sign) and the single-step
    // reduction loses precision. Fall back to scalar std::sin for those lanes.
    constexpr float kSinCosSafe = 13176794.0f;  // ~2^22 · π
    __m512 abs_x_orig = _mm512_abs_ps(x_orig);
    __mmask16 big_lanes = _mm512_cmp_ps_mask(abs_x_orig, _mm512_set1_ps(kSinCosSafe), _CMP_GT_OQ);
    if (big_lanes != 0) {
        alignas(64) float in_lanes[16];
        alignas(64) float out_lanes[16];
        _mm512_store_ps(in_lanes, x_orig);
        _mm512_store_ps(out_lanes, result);
        for (int lane = 0; lane < 16; ++lane) {
            if (std::abs(in_lanes[lane]) > kSinCosSafe) {
                out_lanes[lane] = std::sin(in_lanes[lane]);
            }
        }
        result = _mm512_load_ps(out_lanes);
    }
    return result;
}

/**
 * @brief AVX-512 vectorized cos
 */
inline __m512 cos_avx512(__m512 x) {
    __m512 pi_2 = _mm512_set1_ps(1.57079632679f);
    return sin_avx512(_mm512_add_ps(x, pi_2));
}

/**
 * @brief AVX-512 vectorized leaky ReLU
 */
inline __m512 leaky_relu_avx512(__m512 x, float alpha = 0.01f) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 zero = _mm512_setzero_ps();
    __mmask16 mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
    __m512 scaled = _mm512_mul_ps(x, alpha_vec);
    return _mm512_mask_blend_ps(mask, scaled, x);
}

#endif // TENZOR_FAST_MATH_AVX512

// ============================================================================
// Float16 F16C Support
// ============================================================================

#ifdef TENZOR_FAST_MATH_F16C

/**
 * @brief Convert 8 Float16 values to 8 Float32 values
 */
inline __m256 f16_to_f32_avx2(const void* fp16_ptr) {
    __m128i fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(fp16_ptr));
    return _mm256_cvtph_ps(fp16);
}

/**
 * @brief Convert 8 Float32 values to 8 Float16 values
 */
inline __m128i f32_to_f16_avx2(__m256 fp32) {
    return _mm256_cvtps_ph(fp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

#ifdef TENZOR_FAST_MATH_AVX512

/**
 * @brief Convert 16 Float16 values to 16 Float32 values (AVX-512)
 */
inline __m512 f16_to_f32_avx512(const void* fp16_ptr) {
    __m256i fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fp16_ptr));
    return _mm512_cvtph_ps(fp16);
}

/**
 * @brief Convert 16 Float32 values to 16 Float16 values (AVX-512)
 */
inline __m256i f32_to_f16_avx512(__m512 fp32) {
    return _mm512_cvtps_ph(fp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

#endif // TENZOR_FAST_MATH_AVX512

#endif // TENZOR_FAST_MATH_F16C

// ============================================================================
// Batch Processing Functions with Loop Unrolling
// ============================================================================

#ifdef TENZOR_FAST_MATH_AVX2

/**
 * @brief Process array with exp using loop unrolling (2x unroll)
 */
inline void exp_batch_avx2(const float* input, float* output, size_t n) {
    size_t i = 0;

    // Process 16 elements per iteration (2x unroll)
    for (; i + 16 <= n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(input + i);
        __m256 v1 = _mm256_loadu_ps(input + i + 8);

        __m256 r0 = exp_avx2(v0);
        __m256 r1 = exp_avx2(v1);

        _mm256_storeu_ps(output + i, r0);
        _mm256_storeu_ps(output + i + 8, r1);
    }

    // Process remaining 8 elements
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256 r = exp_avx2(v);
        _mm256_storeu_ps(output + i, r);
    }

    // Scalar remainder
    for (; i < n; ++i) {
        output[i] = std::exp(input[i]);
    }
}

/**
 * @brief Process array with log using loop unrolling
 */
inline void log_batch_avx2(const float* input, float* output, size_t n) {
    size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(input + i);
        __m256 v1 = _mm256_loadu_ps(input + i + 8);

        __m256 r0 = log_avx2(v0);
        __m256 r1 = log_avx2(v1);

        _mm256_storeu_ps(output + i, r0);
        _mm256_storeu_ps(output + i + 8, r1);
    }

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256 r = log_avx2(v);
        _mm256_storeu_ps(output + i, r);
    }

    for (; i < n; ++i) {
        output[i] = std::log(input[i]);
    }
}

/**
 * @brief Process array with tanh using loop unrolling
 */
inline void tanh_batch_avx2(const float* input, float* output, size_t n) {
    size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(input + i);
        __m256 v1 = _mm256_loadu_ps(input + i + 8);

        __m256 r0 = tanh_avx2(v0);
        __m256 r1 = tanh_avx2(v1);

        _mm256_storeu_ps(output + i, r0);
        _mm256_storeu_ps(output + i + 8, r1);
    }

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256 r = tanh_avx2(v);
        _mm256_storeu_ps(output + i, r);
    }

    for (; i < n; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

/**
 * @brief Process array with sigmoid using loop unrolling
 */
inline void sigmoid_batch_avx2(const float* input, float* output, size_t n) {
    size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(input + i);
        __m256 v1 = _mm256_loadu_ps(input + i + 8);

        __m256 r0 = sigmoid_avx2(v0);
        __m256 r1 = sigmoid_avx2(v1);

        _mm256_storeu_ps(output + i, r0);
        _mm256_storeu_ps(output + i + 8, r1);
    }

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256 r = sigmoid_avx2(v);
        _mm256_storeu_ps(output + i, r);
    }

    for (; i < n; ++i) {
        float x = input[i];
        output[i] = 1.0f / (1.0f + std::exp(-x));
    }
}

/**
 * @brief Process array with GELU using loop unrolling
 */
inline void gelu_batch_avx2(const float* input, float* output, size_t n) {
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) — matches the canonical
    // gelu_kernel and PyTorch's default (approximate='none').  erf has no AVX2
    // intrinsic, so this is a scalar loop (the prior tanh approximation was
    // ~1e-3 off and inconsistent with the op-level GELU).
    constexpr float inv_sqrt2 = 0.70710678f;
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::erf(x * inv_sqrt2));
    }
}

/**
 * @brief Process where operation with SIMD + OpenMP
 */
inline void where_batch_avx2(const float* cond, const float* x, const float* y,
                             float* output, size_t n) {
    const size_t OMP_THRESHOLD = static_cast<size_t>(::tenzor::OmpThresholds::medium());
    const size_t simd_end = (n / 8) * 8;
    __m256 zero = _mm256_setzero_ps();

    #pragma omp parallel for schedule(static) if(n > OMP_THRESHOLD)
    for (size_t i = 0; i < simd_end; i += 8) {
        __m256 c = _mm256_loadu_ps(cond + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);

        // cond != 0 means true
        __m256 mask = _mm256_cmp_ps(c, zero, _CMP_NEQ_UQ);
        __m256 r = _mm256_blendv_ps(vy, vx, mask);

        _mm256_storeu_ps(output + i, r);
    }

    for (size_t i = simd_end; i < n; ++i) {
        output[i] = (cond[i] != 0.0f) ? x[i] : y[i];
    }
}

#endif // TENZOR_FAST_MATH_AVX2

#ifdef TENZOR_FAST_MATH_AVX512

/**
 * @brief Process where operation using AVX-512 + OpenMP
 */
inline void where_batch_avx512(const float* cond, const float* x, const float* y,
                               float* output, size_t n) {
    const size_t OMP_THRESHOLD = static_cast<size_t>(::tenzor::OmpThresholds::medium());
    const size_t simd_end = (n / 16) * 16;
    __m512 zero = _mm512_setzero_ps();

    #pragma omp parallel for schedule(static) if(n > OMP_THRESHOLD)
    for (size_t i = 0; i < simd_end; i += 16) {
        __m512 c = _mm512_loadu_ps(cond + i);
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);

        __mmask16 mask = _mm512_cmp_ps_mask(c, zero, _CMP_NEQ_UQ);
        __m512 r = _mm512_mask_blend_ps(mask, vy, vx);

        _mm512_storeu_ps(output + i, r);
    }

#ifdef TENZOR_FAST_MATH_AVX2
    where_batch_avx2(cond + simd_end, x + simd_end, y + simd_end, output + simd_end, n - simd_end);
#else
    for (size_t i = simd_end; i < n; ++i) {
        output[i] = (cond[i] != 0.0f) ? x[i] : y[i];
    }
#endif
}

/**
 * @brief Process array with exp using AVX-512 with loop unrolling
 */
inline void exp_batch_avx512(const float* input, float* output, size_t n) {
    size_t i = 0;

    // Process 32 elements per iteration (2x unroll)
    for (; i + 32 <= n; i += 32) {
        __m512 v0 = _mm512_loadu_ps(input + i);
        __m512 v1 = _mm512_loadu_ps(input + i + 16);

        __m512 r0 = exp_avx512(v0);
        __m512 r1 = exp_avx512(v1);

        _mm512_storeu_ps(output + i, r0);
        _mm512_storeu_ps(output + i + 16, r1);
    }

    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512 r = exp_avx512(v);
        _mm512_storeu_ps(output + i, r);
    }

    // Handle remainder with AVX2
#ifdef TENZOR_FAST_MATH_AVX2
    exp_batch_avx2(input + i, output + i, n - i);
#else
    for (; i < n; ++i) {
        output[i] = std::exp(input[i]);
    }
#endif
}

/**
 * @brief Process array with log using AVX-512
 */
inline void log_batch_avx512(const float* input, float* output, size_t n) {
    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m512 v0 = _mm512_loadu_ps(input + i);
        __m512 v1 = _mm512_loadu_ps(input + i + 16);

        __m512 r0 = log_avx512(v0);
        __m512 r1 = log_avx512(v1);

        _mm512_storeu_ps(output + i, r0);
        _mm512_storeu_ps(output + i + 16, r1);
    }

    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512 r = log_avx512(v);
        _mm512_storeu_ps(output + i, r);
    }

#ifdef TENZOR_FAST_MATH_AVX2
    log_batch_avx2(input + i, output + i, n - i);
#else
    for (; i < n; ++i) {
        output[i] = std::log(input[i]);
    }
#endif
}

/**
 * @brief Process array with tanh using AVX-512
 */
inline void tanh_batch_avx512(const float* input, float* output, size_t n) {
    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m512 v0 = _mm512_loadu_ps(input + i);
        __m512 v1 = _mm512_loadu_ps(input + i + 16);

        __m512 r0 = tanh_avx512(v0);
        __m512 r1 = tanh_avx512(v1);

        _mm512_storeu_ps(output + i, r0);
        _mm512_storeu_ps(output + i + 16, r1);
    }

    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512 r = tanh_avx512(v);
        _mm512_storeu_ps(output + i, r);
    }

#ifdef TENZOR_FAST_MATH_AVX2
    tanh_batch_avx2(input + i, output + i, n - i);
#else
    for (; i < n; ++i) {
        output[i] = std::tanh(input[i]);
    }
#endif
}

/**
 * @brief Process array with sigmoid using AVX-512
 */
inline void sigmoid_batch_avx512(const float* input, float* output, size_t n) {
    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m512 v0 = _mm512_loadu_ps(input + i);
        __m512 v1 = _mm512_loadu_ps(input + i + 16);

        __m512 r0 = sigmoid_avx512(v0);
        __m512 r1 = sigmoid_avx512(v1);

        _mm512_storeu_ps(output + i, r0);
        _mm512_storeu_ps(output + i + 16, r1);
    }

    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512 r = sigmoid_avx512(v);
        _mm512_storeu_ps(output + i, r);
    }

#ifdef TENZOR_FAST_MATH_AVX2
    sigmoid_batch_avx2(input + i, output + i, n - i);
#else
    for (; i < n; ++i) {
        float x = input[i];
        output[i] = 1.0f / (1.0f + std::exp(-x));
    }
#endif
}

/**
 * @brief Process array with GELU using AVX-512
 */
inline void gelu_batch_avx512(const float* input, float* output, size_t n) {
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) — matches the canonical
    // gelu_kernel and PyTorch's default (approximate='none').  erf has no
    // AVX-512 intrinsic, so this is a scalar loop.
    constexpr float inv_sqrt2 = 0.70710678f;
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::erf(x * inv_sqrt2));
    }
}

#endif // TENZOR_FAST_MATH_AVX512

} // namespace fast_math
} // namespace cpu
} // namespace tenzor
