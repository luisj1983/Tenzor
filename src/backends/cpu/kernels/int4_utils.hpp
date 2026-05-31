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
 * @brief Pack int8 values into INT4 packed bytes.
 *
 * Each pair of consecutive int8 values (which must lie in [-8, 7]) is
 * packed into a single byte: low nibble = first value, high nibble = second.
 * If @p n is odd the final nibble is stored in the low nibble of the last
 * byte with the high nibble zeroed.
 *
 * @param src  Source buffer of int8 values (length @p n)
 * @param dst  Destination buffer of packed bytes (length ceil(n/2))
 * @param n    Number of int8 elements to pack
 */
inline void pack_int4(const int8_t* src, uint8_t* dst, int64_t n) {
    int64_t pairs = n / 2;
    for (int64_t i = 0; i < pairs; ++i) {
        uint8_t lo = static_cast<uint8_t>(src[2 * i] & 0x0F);
        uint8_t hi = static_cast<uint8_t>(src[2 * i + 1] & 0x0F);
        dst[i] = lo | (hi << 4);
    }
    if (n % 2 != 0) {
        dst[pairs] = static_cast<uint8_t>(src[n - 1] & 0x0F);
    }
}

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

/**
 * @brief Unpack INT4 packed bytes into sign-extended int8 values.
 *
 * Reverses the transformation performed by @ref pack_int4.  Each packed
 * byte yields two int8 values whose 4-bit representations are sign-extended
 * back to 8 bits (range [-8, 7]).
 *
 * @param src  Source buffer of packed bytes (length ceil(n/2))
 * @param dst  Destination buffer of int8 values (length @p n)
 * @param n    Number of int8 elements to unpack
 */
inline void unpack_int4(const uint8_t* src, int8_t* dst, int64_t n) {
    int64_t pairs = n / 2;
    for (int64_t i = 0; i < pairs; ++i) {
        unpack_int4(src[i], dst[2 * i], dst[2 * i + 1]);
    }
    if (n % 2 != 0) {
        int8_t discard;
        unpack_int4(src[pairs], dst[n - 1], discard);
    }
}

} // namespace cpu
} // namespace tenzor
