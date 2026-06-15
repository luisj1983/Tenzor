/**
 * @file dtype.cpp
 * @brief Implementation of dtype conversions for half-precision types
 */

#include "tenzor/core/dtype.hpp"
#include <cstring>
#include <cmath>

namespace tenzor {

// ============================================================================
// Float16 Conversions (IEEE 754 half-precision)
// ============================================================================

/**
 * @brief Convert Float32 to Float16 using IEEE 754 half-precision format
 *
 * Implementation based on IEEE 754-2008 standard
 * 1 sign bit, 5 exponent bits, 10 mantissa bits
 */
Float16::Float16(float f) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(float));

    // Half sign bit (bit 31 of float -> bit 15 of half)
    uint16_t sign = static_cast<uint16_t>((f_bits >> 16) & 0x8000);
    int32_t  exp  = static_cast<int32_t>((f_bits >> 23) & 0xFF);
    uint32_t mant = f_bits & 0x7FFFFF;

    if (exp == 0xFF) {
        // Inf (mant==0) or NaN (mant!=0); keep a non-zero mantissa for NaN.
        bits = static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x0200 : 0));
        return;
    }

    // Rebias exponent from float (127) to half (15).
    int32_t e = exp - 127 + 15;

    if (e >= 0x1F) {
        // Overflow -> signed infinity.
        bits = static_cast<uint16_t>(sign | 0x7C00);
        return;
    }

    if (e <= 0) {
        // Subnormal half or underflow to (signed) zero, round-to-nearest-even.
        if (e < -10) {
            bits = sign;  // magnitude too small to represent
            return;
        }
        mant |= 0x800000;            // restore implicit leading 1
        int32_t shift = 14 - e;      // discards `shift` low bits -> 10-bit field
        uint32_t half_mant = mant >> shift;
        uint32_t remainder = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mant & 1))) {
            half_mant++;  // may carry subnormal -> smallest normal (exp field 1)
        }
        bits = static_cast<uint16_t>(sign | half_mant);
        return;
    }

    // Normal case, round-to-nearest-even on the 13 discarded mantissa bits.
    uint32_t half_mant = mant >> 13;
    uint32_t remainder = mant & 0x1FFF;
    uint16_t h = static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(e) << 10) | half_mant);
    if (remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1))) {
        // Increment propagates a mantissa carry into the exponent; if the
        // exponent was 0x1E this correctly overflows to 0x7C00 (infinity).
        h++;
    }
    bits = h;
}

/**
 * @brief Convert Float16 to Float32
 */
Float16::operator float() const {
    // Extract components
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exp = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;

    uint32_t f_sign = sign;
    uint32_t f_exp;
    uint32_t f_mantissa;

    if (exp == 0) {
        if (mantissa == 0) {
            // Zero
            f_exp = 0;
            f_mantissa = 0;
        } else {
            // Denormalized number
            // Normalize it
            int e = -1;
            uint32_t m = mantissa;
            do {
                e++;
                m <<= 1;
            } while ((m & 0x400) == 0);
            f_exp = 127 - 15 - e;
            f_mantissa = (m & 0x3FF) << 13;
        }
    } else if (exp == 0x1F) {
        // Infinity or NaN
        f_exp = 0xFF;
        f_mantissa = mantissa << 13;
    } else {
        // Normalized number
        f_exp = exp - 15 + 127;
        f_mantissa = mantissa << 13;
    }

    uint32_t f_bits = (f_sign << 31) | (f_exp << 23) | f_mantissa;
    float result;
    std::memcpy(&result, &f_bits, sizeof(float));
    return result;
}

// ============================================================================
// BFloat16 Conversions (Brain Float)
// ============================================================================

/**
 * @brief Convert Float32 to BFloat16
 *
 * BFloat16 uses the same exponent bits as Float32, making conversion simple.
 * We can just truncate the lower 16 bits of mantissa.
 * With rounding: we add 0x7FFF before truncation for round-to-nearest-even.
 */
BFloat16::BFloat16(float f) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(float));

    // Check for NaN
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;

    if (exp == 0xFF && mantissa != 0) {
        // Preserve NaN
        bits = static_cast<uint16_t>((f_bits >> 16) | 0x0001);
    } else {
        // Round-to-nearest-even: add 0x7FFF (half of least significant bit in bf16)
        // This implements banker's rounding
        uint32_t rounding_bias = 0x7FFF + ((f_bits >> 16) & 1);
        bits = static_cast<uint16_t>((f_bits + rounding_bias) >> 16);
    }
}

/**
 * @brief Convert BFloat16 to Float32
 *
 * Simple zero-extension of the lower 16 bits
 */
BFloat16::operator float() const {
    uint32_t f_bits = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &f_bits, sizeof(float));
    return result;
}

// ============================================================================
// FP8 E4M3 Conversions (1 sign, 4 exponent, 3 mantissa)
// ============================================================================

/**
 * @brief Convert Float32 to FP8 E4M3
 *
 * E4M3 format: 1 sign + 4 exponent + 3 mantissa bits
 * Bias: 7, Range: ~[-448, 448], no infinities (NaN uses max exponent + non-zero mantissa)
 * This format matches NVIDIA's __nv_fp8_e4m3 used by Hopper Tensor Cores.
 */
FP8_E4M3::FP8_E4M3(float f) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(float));

    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;

    uint8_t h_sign = static_cast<uint8_t>(sign);
    uint8_t h_exp;
    uint8_t h_mantissa;

    if (exp == 0xFF) {
        // Float32 Inf/NaN -> E4M3 NaN (0x7F without sign, or 0xFF with sign)
        // E4M3 has no infinity representation; max exponent + non-zero mantissa = NaN
        h_exp = 0xF;
        h_mantissa = 0x7;  // NaN
    } else if (exp == 0) {
        // Zero or float32 denorm -> zero in E4M3
        h_exp = 0;
        h_mantissa = 0;
    } else {
        // Normalized: float32 bias=127, e4m3 bias=7
        int32_t new_exp = static_cast<int32_t>(exp) - 127 + 7;

        if (new_exp >= 0xF) {
            // Overflow: clamp to E4M3's true max finite value, which is
            // exp=0xF, mantissa=0x6 = 2^(15-7) * (1 + 6/8) = 448. Only
            // mantissa=0x7 at exp=0xF is reserved for NaN — mantissa=0x0..0x6
            // are valid finite numbers (256, 288, 320, 352, 384, 416, 448).
            // The previous "mantissa=0x7 at exp=0xE" (= 240) was a PyTorch
            // saturation convention that diverges from NVIDIA's native E4M3,
            // and "mantissa=0x0 at exp=0xF" (= 256) was an intermediate
            // off-by-6 fix. This uses the correct bit pattern.
            h_exp = 0xF;
            h_mantissa = 0x6;
        } else if (new_exp <= 0) {
            // Denormalized E4M3 or underflow, round-to-nearest-even.
            // Shift the implicit-1 mantissa down into the subnormal 3-bit field.
            // At new_exp == -3 the magnitude sits at half the smallest subnormal,
            // so RNE may round it up (ties-to-even keeps zero); below that it
            // underflows to (signed) zero.
            if (new_exp >= -3) {
                uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                // Keep the top 3 bits, round-to-nearest-even on the discarded 20.
                uint32_t kept = (m >> 20) & 0x7;
                uint32_t remainder = m & 0xFFFFF;          // low 20 bits
                uint32_t halfway = 0x80000;                // 1 << 19
                if (remainder > halfway || (remainder == halfway && (kept & 1))) {
                    kept++;  // may carry subnormal -> smallest normal (exp field 1)
                }
                // kept can be 0x8 after carry: that promotes to exp=1, mant=0,
                // which the packed (h_exp<<3)|h_mantissa layout encodes naturally.
                h_exp = static_cast<uint8_t>(kept >> 3);
                h_mantissa = static_cast<uint8_t>(kept & 0x7);
            } else {
                h_exp = 0;
                h_mantissa = 0;
            }
        } else {
            // Normalized, round-to-nearest-even on the 20 discarded mantissa bits.
            uint32_t kept = (mantissa >> 20) & 0x7;
            uint32_t remainder = mantissa & 0xFFFFF;       // low 20 bits
            uint32_t halfway = 0x80000;                    // 1 << 19
            // Pack so a mantissa carry propagates into the exponent.
            uint32_t packed = (static_cast<uint32_t>(new_exp) << 3) | kept;
            if (remainder > halfway || (remainder == halfway && (kept & 1))) {
                packed++;
            }
            uint32_t r_exp = packed >> 3;
            uint32_t r_mant = packed & 0x7;
            // Clamp to the max finite E4M3 value (exp=0xF, mant=0x6 = 448).
            // exp==0xF with mant>=0x7 would be NaN; saturate per NVIDIA's RN.
            if (r_exp > 0xF || (r_exp == 0xF && r_mant >= 0x7)) {
                r_exp = 0xF;
                r_mant = 0x6;
            }
            h_exp = static_cast<uint8_t>(r_exp);
            h_mantissa = static_cast<uint8_t>(r_mant);
        }
    }

    bits = (h_sign << 7) | (h_exp << 3) | h_mantissa;
}

/**
 * @brief Convert FP8 E4M3 to Float32
 */
FP8_E4M3::operator float() const {
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mantissa = bits & 0x7;

    uint32_t f_sign = sign;
    uint32_t f_exp;
    uint32_t f_mantissa;

    if (exp == 0) {
        if (mantissa == 0) {
            f_exp = 0;
            f_mantissa = 0;
        } else {
            // Denormalized: normalize
            int e = -1;
            uint32_t m = mantissa;
            do {
                e++;
                m <<= 1;
            } while ((m & 0x8) == 0);
            f_exp = 127 - 7 - e;
            f_mantissa = (m & 0x7) << 20;
        }
    } else if (exp == 0xF && mantissa == 0x7) {
        // NaN — only this exact bit pattern (exp=0xF AND mantissa=0x7) is
        // reserved for NaN in NVIDIA's E4M3. All other mantissa values at
        // exp=0xF are valid finite numbers (the max finite is 0xF / 0x6 = 448).
        f_exp = 0xFF;
        f_mantissa = 0x700000;
    } else {
        // Normalized. exp=0xF with mantissa < 0x7 is legal finite — no
        // special case needed because the general exp-rebias formula
        // handles it (f_exp = 15-7+127 = 135, 2^8 * (1 + mantissa/8) covers
        // 256, 288, ..., 448 as mantissa walks 0..6).
        f_exp = exp - 7 + 127;
        f_mantissa = mantissa << 20;
    }

    uint32_t f_bits = (f_sign << 31) | (f_exp << 23) | f_mantissa;
    float result;
    std::memcpy(&result, &f_bits, sizeof(float));
    return result;
}

// ============================================================================
// FP8 E5M2 Conversions (1 sign, 5 exponent, 2 mantissa)
// ============================================================================

/**
 * @brief Convert Float32 to FP8 E5M2
 *
 * E5M2 format: 1 sign + 5 exponent + 2 mantissa bits
 * Bias: 15, Range: ~[-57344, 57344], supports infinity and NaN
 * Same exponent width as Float16, matching its range.
 * This format matches NVIDIA's __nv_fp8_e5m2 used by Hopper Tensor Cores.
 */
FP8_E5M2::FP8_E5M2(float f) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(float));

    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;

    uint8_t h_sign = static_cast<uint8_t>(sign);
    uint8_t h_exp;
    uint8_t h_mantissa;

    if (exp == 0xFF) {
        // Inf or NaN
        h_exp = 0x1F;
        h_mantissa = mantissa ? 0x3 : 0;  // Preserve NaN vs Inf
    } else if (exp == 0) {
        h_exp = 0;
        h_mantissa = 0;
    } else {
        // float32 bias=127, e5m2 bias=15
        int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;

        if (new_exp >= 0x1F) {
            // Overflow to infinity
            h_exp = 0x1F;
            h_mantissa = 0;
        } else if (new_exp <= 0) {
            // Denormalized E5M2 or underflow, round-to-nearest-even.
            if (new_exp >= -2) {
                uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                // Keep the top 2 bits, round-to-nearest-even on the discarded 21.
                uint32_t kept = (m >> 21) & 0x3;
                uint32_t remainder = m & 0x1FFFFF;         // low 21 bits
                uint32_t halfway = 0x100000;               // 1 << 20
                if (remainder > halfway || (remainder == halfway && (kept & 1))) {
                    kept++;  // may carry subnormal -> smallest normal (exp field 1)
                }
                h_exp = static_cast<uint8_t>(kept >> 2);
                h_mantissa = static_cast<uint8_t>(kept & 0x3);
            } else {
                h_exp = 0;
                h_mantissa = 0;
            }
        } else {
            // Normalized, round-to-nearest-even on the 21 discarded mantissa bits.
            uint32_t kept = (mantissa >> 21) & 0x3;
            uint32_t remainder = mantissa & 0x1FFFFF;      // low 21 bits
            uint32_t halfway = 0x100000;                   // 1 << 20
            // Pack so a mantissa carry propagates into the exponent; if it pushes
            // the exponent to 0x1F (mantissa 0) the result is correctly infinity.
            uint32_t packed = (static_cast<uint32_t>(new_exp) << 2) | kept;
            if (remainder > halfway || (remainder == halfway && (kept & 1))) {
                packed++;
            }
            h_exp = static_cast<uint8_t>(packed >> 2);
            h_mantissa = static_cast<uint8_t>(packed & 0x3);
        }
    }

    bits = (h_sign << 7) | (h_exp << 2) | h_mantissa;
}

/**
 * @brief Convert FP8 E5M2 to Float32
 */
FP8_E5M2::operator float() const {
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 2) & 0x1F;
    uint32_t mantissa = bits & 0x3;

    uint32_t f_sign = sign;
    uint32_t f_exp;
    uint32_t f_mantissa;

    if (exp == 0) {
        if (mantissa == 0) {
            f_exp = 0;
            f_mantissa = 0;
        } else {
            int e = -1;
            uint32_t m = mantissa;
            do {
                e++;
                m <<= 1;
            } while ((m & 0x4) == 0);
            f_exp = 127 - 15 - e;
            f_mantissa = (m & 0x3) << 21;
        }
    } else if (exp == 0x1F) {
        f_exp = 0xFF;
        f_mantissa = mantissa << 21;
    } else {
        f_exp = exp - 15 + 127;
        f_mantissa = mantissa << 21;
    }

    uint32_t f_bits = (f_sign << 31) | (f_exp << 23) | f_mantissa;
    float result;
    std::memcpy(&result, &f_bits, sizeof(float));
    return result;
}

// ============================================================================
// Helper conversion functions
// ============================================================================

/**
 * @brief Convert between any supported data types
 */
template<typename To, typename From>
auto convert_dtype(From value) -> To {
    if constexpr (std::is_same_v<To, From>) {
        return value;
    } else if constexpr (std::is_same_v<To, Float16>) {
        return Float16(static_cast<float>(value));
    } else if constexpr (std::is_same_v<To, BFloat16>) {
        return BFloat16(static_cast<float>(value));
    } else if constexpr (std::is_same_v<To, FP8_E4M3>) {
        return FP8_E4M3(static_cast<float>(value));
    } else if constexpr (std::is_same_v<To, FP8_E5M2>) {
        return FP8_E5M2(static_cast<float>(value));
    } else if constexpr (std::is_same_v<From, Float16>) {
        return static_cast<To>(static_cast<float>(value));
    } else if constexpr (std::is_same_v<From, BFloat16>) {
        return static_cast<To>(static_cast<float>(value));
    } else if constexpr (std::is_same_v<From, FP8_E4M3>) {
        return static_cast<To>(static_cast<float>(value));
    } else if constexpr (std::is_same_v<From, FP8_E5M2>) {
        return static_cast<To>(static_cast<float>(value));
    } else {
        return static_cast<To>(value);
    }
}

// Explicit instantiations for common conversions
template auto convert_dtype<float, Float16>(Float16) -> float;
template auto convert_dtype<float, BFloat16>(BFloat16) -> float;
template auto convert_dtype<Float16, float>(float) -> Float16;
template auto convert_dtype<BFloat16, float>(float) -> BFloat16;
template auto convert_dtype<double, Float16>(Float16) -> double;
template auto convert_dtype<double, BFloat16>(BFloat16) -> double;
template auto convert_dtype<Float16, double>(double) -> Float16;
template auto convert_dtype<BFloat16, double>(double) -> BFloat16;
template auto convert_dtype<float, FP8_E4M3>(FP8_E4M3) -> float;
template auto convert_dtype<float, FP8_E5M2>(FP8_E5M2) -> float;
template auto convert_dtype<FP8_E4M3, float>(float) -> FP8_E4M3;
template auto convert_dtype<FP8_E5M2, float>(float) -> FP8_E5M2;
template auto convert_dtype<double, FP8_E4M3>(FP8_E4M3) -> double;
template auto convert_dtype<double, FP8_E5M2>(FP8_E5M2) -> double;
template auto convert_dtype<FP8_E4M3, double>(double) -> FP8_E4M3;
template auto convert_dtype<FP8_E5M2, double>(double) -> FP8_E5M2;

} // namespace tenzor
