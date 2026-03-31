#pragma once

/// @file checked_math.hpp
/// @brief Overflow-safe integer arithmetic for tensor offset/stride calculations.

#include <cstdint>
#include <limits>
#include <stdexcept>

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
        } else {
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
