/**
 * @file safe_math.hpp
 * @brief Safe math utilities that avoid undefined behavior
 */

#pragma once

#include <cstdint>
#include <limits>

namespace tenzor::detail {

/// Safe absolute value for int64_t that avoids UB for INT64_MIN.
/// Returns uint64_t since |INT64_MIN| > INT64_MAX.
inline constexpr auto safe_abs(int64_t v) -> uint64_t {
    if (v == std::numeric_limits<int64_t>::min())
        return static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1;
    return static_cast<uint64_t>(v < 0 ? -v : v);
}

} // namespace tenzor::detail
