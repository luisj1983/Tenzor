/**
 * @file int4_utils.hpp
 * @brief INT4 unpacking utility for quantized inference.
 *
 * Provides `unpack_int4`, which decodes a packed INT4 byte (two 4-bit values,
 * low-nibble-first: low nibble holds the first element, high nibble holds the
 * second) back into two sign-extended int8 values in [-8, 7].
 *
 * The corresponding pack_int4 routines are NOT in this file — they live as
 * member functions on the quantizers that produce packed weights:
 * `AWQQuantizer::pack_int4` (src/nn/quantization/awq.cpp) and
 * `GPTQQuantizer::pack_int4` (src/nn/quantization/gptq.cpp). Both use the
 * same low-nibble-first convention and round-trip through this file's
 * `unpack_int4`, so the two halves of the format should be read together.
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
