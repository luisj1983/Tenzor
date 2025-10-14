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

    // Extract sign, exponent, and mantissa from float32
    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;

    uint16_t h_sign = static_cast<uint16_t>(sign);
    uint16_t h_exp;
    uint16_t h_mantissa;

    // Handle special cases
    if (exp == 0xFF) {
        // Infinity or NaN
        h_exp = 0x1F;
        h_mantissa = mantissa ? 0x3FF : 0;  // Preserve NaN, zero for infinity
    } else if (exp == 0) {
        // Zero or denormalized number
        h_exp = 0;
        h_mantissa = 0;
    } else {
        // Normalized number
        int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;

        if (new_exp >= 0x1F) {
            // Overflow to infinity
            h_exp = 0x1F;
            h_mantissa = 0;
        } else if (new_exp <= 0) {
            // Underflow to zero or denormalized
            if (new_exp >= -10) {
                // Denormalized number
                uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                h_mantissa = static_cast<uint16_t>((m >> 13) & 0x3FF);
                h_exp = 0;
            } else {
                // Complete underflow to zero
                h_exp = 0;
                h_mantissa = 0;
            }
        } else {
            // Normal case
            h_exp = static_cast<uint16_t>(new_exp);
            h_mantissa = static_cast<uint16_t>((mantissa >> 13) & 0x3FF);
        }
    }

    bits = (h_sign << 15) | (h_exp << 10) | h_mantissa;
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
    } else if constexpr (std::is_same_v<From, Float16>) {
        return static_cast<To>(static_cast<float>(value));
    } else if constexpr (std::is_same_v<From, BFloat16>) {
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

} // namespace tenzor
