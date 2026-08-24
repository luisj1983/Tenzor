/**
 * @file safe_math.hpp
 * @brief Safe math utilities that avoid undefined behavior
 */

#pragma once

#include <cstdint>
#include <limits>

#include "tenzor/core/dtype.hpp"

namespace tenzor::detail {

/// Returns a small epsilon suitable for zero-safety clamping in backward passes.
/// FP16/BF16 use a larger epsilon due to limited precision.
inline double dtype_epsilon(DType dtype) {
    switch (dtype) {
        case DType::Float16:
        case DType::BFloat16: return 1e-4;
        default: return 1e-30;
    }
}

/// Epsilon for clamping a uniform sample away from the exact 0/1 boundary
/// before a log()/log1p() inverse-CDF transform (Gumbel, Weibull, etc.).
///
/// Must exceed half the dtype's ULP at 1.0, or the clamped value rounds
/// straight back to the boundary when narrowed to the sample's storage
/// dtype -- producing log(0) = -inf downstream. BFloat16's ULP(1.0) is
/// ~7.8e-3 (7 mantissa bits) and Float16's is ~4.9e-4 (10 mantissa bits),
/// so the Float32/Float64 epsilon (~1e-7) is far too small for either.
inline float boundary_clamp_epsilon(DType dtype) {
    switch (dtype) {
        case DType::BFloat16: return 1e-2f;
        case DType::Float16: return 1e-3f;
        default: return 1e-7f;
    }
}

/// Safe absolute value for int64_t that avoids UB for INT64_MIN.
/// Returns uint64_t since |INT64_MIN| > INT64_MAX.
inline constexpr auto safe_abs(int64_t v) -> uint64_t {
    if (v == std::numeric_limits<int64_t>::min())
        return static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1;
    return static_cast<uint64_t>(v < 0 ? -v : v);
}

/// Portable checked multiply mirroring GCC/Clang's __builtin_mul_overflow,
/// which has no MSVC equivalent. Only valid for non-negative operands — every
/// call site computes a tensor element/byte count (dims/sizes are validated
/// non-negative before multiplying), so a division-based check is correct
/// for both the signed (int64_t) and unsigned (size_t/uint64_t)
/// instantiations used across the codebase.
template <typename T>
inline bool checked_mul_overflow(T a, T b, T* result) {
    if (a != 0 && b > (std::numeric_limits<T>::max)() / a) {
        return true;
    }
    *result = a * b;
    return false;
}

/// Portable checked add mirroring GCC/Clang's __builtin_add_overflow.
/// Same non-negative-operand caveat as checked_mul_overflow above.
template <typename T>
inline bool checked_add_overflow(T a, T b, T* result) {
    if (a > (std::numeric_limits<T>::max)() - b) {
        return true;
    }
    *result = a + b;
    return false;
}

} // namespace tenzor::detail
