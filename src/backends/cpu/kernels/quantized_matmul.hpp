/**
 * @file quantized_matmul.hpp
 * @brief INT8 activations x INT4 weights matrix multiplication kernel.
 *
 * Provides a scalar (non-SIMD) tiled implementation of mixed-precision
 * quantized matmul where activations are INT8 and weights are packed INT4
 * (two values per byte, low nibble first).  Accumulation is performed in
 * int32 to avoid overflow.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>

namespace tenzor {
namespace cpu {

namespace detail {

/// Sign-extend a 4-bit value stored in the low nibble of a byte to int8.
inline int8_t sign_extend_int4(uint8_t nibble) {
    int8_t val = static_cast<int8_t>(nibble & 0x0F);
    if (val & 0x08) val |= static_cast<int8_t>(0xF0);
    return val;
}

} // namespace detail

/**
 * @brief Mixed-precision quantized matmul: INT8 activations x INT4 weights.
 *
 * Computes  C[M, N] = A[M, K] @ B[K, N]  where A is int8 and B is
 * packed INT4 (two K-values per byte, low nibble first).
 *
 * Weights layout (packed):
 *   For a logical weight matrix of shape [K, N], each column of K int4
 *   values is packed into ceil(K/2) bytes.  The packed buffer has shape
 *   [ceil(K/2), N] stored in row-major order (i.e., for byte row @p r
 *   and column @p n, the address is  weights_packed[r * N + n]).
 *
 * A simple tiled loop structure is used for reasonable cache behaviour
 * without SIMD.
 *
 * @param activations   Row-major int8 buffer  [M x K]
 * @param weights_packed  Row-major packed int4 buffer  [ceil(K/2) x N]
 * @param output        Row-major int32 accumulator buffer  [M x N]
 * @param M  Number of rows (batch / tokens)
 * @param N  Number of columns (output features)
 * @param K  Inner dimension (input features, must be even for full packing)
 */
inline void quantized_matmul_w4a8(const int8_t* activations,
                                  const uint8_t* weights_packed,
                                  int32_t* output,
                                  int64_t M, int64_t N, int64_t K) {
    // Zero-initialize output accumulator
    std::memset(output, 0, static_cast<size_t>(M * N) * sizeof(int32_t));

    // Tile sizes chosen for L1 residency on typical CPUs
    constexpr int64_t TILE_M = 4;
    constexpr int64_t TILE_N = 16;
    constexpr int64_t TILE_K = 32;  // in logical (unpacked) elements

    int64_t packed_stride_n = N;  // packed row length
    int64_t packed_K = (K + 1) / 2;  // number of packed byte rows

    for (int64_t m0 = 0; m0 < M; m0 += TILE_M) {
        int64_t m_end = (m0 + TILE_M < M) ? m0 + TILE_M : M;

        for (int64_t n0 = 0; n0 < N; n0 += TILE_N) {
            int64_t n_end = (n0 + TILE_N < N) ? n0 + TILE_N : N;

            for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
                int64_t k_end = (k0 + TILE_K < K) ? k0 + TILE_K : K;

                for (int64_t m = m0; m < m_end; ++m) {
                    for (int64_t n = n0; n < n_end; ++n) {
                        int32_t acc = 0;

                        for (int64_t k = k0; k < k_end; ++k) {
                            // Fetch activation (int8)
                            int8_t a_val = activations[m * K + k];

                            // Fetch and unpack weight (int4)
                            int64_t packed_row = k / 2;
                            uint8_t packed_byte = weights_packed[packed_row * packed_stride_n + n];
                            uint8_t nibble = (k % 2 == 0)
                                ? (packed_byte & 0x0F)
                                : ((packed_byte >> 4) & 0x0F);
                            int8_t w_val = detail::sign_extend_int4(nibble);

                            acc += static_cast<int32_t>(a_val) * static_cast<int32_t>(w_val);
                        }

                        output[m * N + n] += acc;
                    }
                }
            }
        }
    }
}

} // namespace cpu
} // namespace tenzor
