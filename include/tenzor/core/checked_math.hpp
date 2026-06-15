#pragma once

/// @file checked_math.hpp
/// @brief Overflow-safe integer arithmetic for tensor offset/stride calculations.

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tenzor {

/// Multiply two int64_t values, throwing on overflow.
inline int64_t checked_mul(int64_t a, int64_t b) {
#if defined(__GNUC__) || defined(__clang__)
    int64_t result;
    if (__builtin_mul_overflow(a, b, &result)) {
        throw std::overflow_error("Integer overflow in multiplication");
    }
    return result;
#else
    // MSVC fallback
    if (a == 0 || b == 0) return 0;
    int64_t result = a * b;
    if (a != 0 && result / a != b) {
        throw std::overflow_error("Integer overflow in multiplication");
    }
    return result;
#endif
}

/// Add two int64_t values, throwing on overflow.
inline int64_t checked_add(int64_t a, int64_t b) {
#if defined(__GNUC__) || defined(__clang__)
    int64_t result;
    if (__builtin_add_overflow(a, b, &result)) {
        throw std::overflow_error("Integer overflow in addition");
    }
    return result;
#else
    if ((b > 0 && a > std::numeric_limits<int64_t>::max() - b) ||
        (b < 0 && a < std::numeric_limits<int64_t>::min() - b)) {
        throw std::overflow_error("Integer overflow in addition");
    }
    return a + b;
#endif
}

/// Narrow a double to an integer type T, throwing if out of range.
template<typename T>
T checked_narrow(double value) {
    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_same_v<T, bool>) {
            return value != 0.0;
        } else if constexpr (sizeof(T) >= 8) {
            // 64-bit integer maxima are not exactly representable as double:
            // (double)INT64_MAX rounds up to 2^63 and (double)UINT64_MAX to 2^64.
            // Comparing against those rounded maxima lets value == 2^63 / 2^64 slip
            // through, then static_cast<T>(value) is out-of-range UB. Use strict
            // power-of-two bounds that are exactly representable as double instead.
            constexpr int bits = std::numeric_limits<T>::digits + (std::is_signed_v<T> ? 1 : 0);
            if constexpr (std::is_signed_v<T>) {
                const double lo = -std::ldexp(1.0, bits - 1);   // -2^(bits-1)
                const double hi = std::ldexp(1.0, bits - 1);    //  2^(bits-1) (exclusive)
                if (!(value >= lo && value < hi)) {
                    throw std::overflow_error("Value out of range for target integer type");
                }
            } else {
                const double hi = std::ldexp(1.0, bits);        //  2^bits (exclusive)
                if (!(value >= 0.0 && value < hi)) {
                    throw std::overflow_error("Value out of range for target integer type");
                }
            }
            return static_cast<T>(value);
        } else {
            // Sub-64-bit integer maxima are exactly representable as double.
            if (value < static_cast<double>(std::numeric_limits<T>::min()) ||
                value > static_cast<double>(std::numeric_limits<T>::max())) {
                throw std::overflow_error("Value out of range for target integer type");
            }
            return static_cast<T>(value);
        }
    } else {
        return static_cast<T>(value);
    }
}

}  // namespace tenzor
