/**
 * @file int4_utils.hpp
 * @brief INT4 packing and unpacking utilities for quantized inference.
 *
 * Provides routines for packing int8 values in [-8, 7] range into a packed
 * INT4 representation (two values per byte) and unpacking them back with
 * correct sign extension.  The packing convention is little-nibble-first:
 * low nibble holds the first element, high nibble holds the second.
 */
#pragma once

#include <cstdint>
#include <cstddef>

namespace tenzor {
namespace cpu {

/**
 * @brief Unpack a single INT4 packed byte into two sign-extended int8 values.
 *
 * Low nibble -> @p low, high nibble -> @p high; each 4-bit value is
 * sign-extended back to 8 bits (range [-8, 7]).
 */
inline void unpack_int4(uint8_t packed, int8_t& low, int8_t& high) {
    int8_t lo = static_cast<int8_t>(packed & 0x0F);
    if (lo & 0x08) lo |= static_cast<int8_t>(0xF0);  // sign-extend
    int8_t hi = static_cast<int8_t>((packed >> 4) & 0x0F);
    if (hi & 0x08) hi |= static_cast<int8_t>(0xF0);  // sign-extend
    low = lo;
    high = hi;
}

} // namespace cpu
} // namespace tenzor
