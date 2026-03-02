#pragma once
#include <cstdint>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace tenzor::cpu {

/// Transpose an 8x8 block of floats using AVX2 intrinsics.
/// Falls back to scalar for non-AVX2 platforms.
inline void transpose_8x8_avx2(const float* src, int64_t src_stride,
                                float* dst, int64_t dst_stride) {
#ifdef __AVX2__
    // Load 8 rows of 8 floats each
    __m256 r0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 r1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);

    // Phase 1: Interleave 32-bit floats within 128-bit lanes
    // unpacklo/hi interleave adjacent pairs:
    //   unpacklo(r0, r1) = [r0[0], r1[0], r0[1], r1[1], r0[4], r1[4], r0[5], r1[5]]
    //   unpackhi(r0, r1) = [r0[2], r1[2], r0[3], r1[3], r0[6], r1[6], r0[7], r1[7]]
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);  // a0b0 a1b1 | a4b4 a5b5
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);  // a2b2 a3b3 | a6b6 a7b7
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);  // c0d0 c1d1 | c4d4 c5d5
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);  // c2d2 c3d3 | c6d6 c7d7
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);  // e0f0 e1f1 | e4f4 e5f5
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);  // e2f2 e3f3 | e6f6 e7f7
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);  // g0h0 g1h1 | g4h4 g5h5
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);  // g2h2 g3h3 | g6h6 g7h7

    // Phase 2: Shuffle 64-bit pairs to group columns 0-1 and 2-3
    // shuffle_ps with mask 0x44 = 01 00 01 00 takes [lo0,lo1] from both operands
    // shuffle_ps with mask 0xEE = 11 10 11 10 takes [hi0,hi1] from both operands
    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);  // a0b0c0d0 | a4b4c4d4
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);  // a1b1c1d1 | a5b5c5d5
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);  // a2b2c2d2 | a6b6c6d6
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);  // a3b3c3d3 | a7b7c7d7
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);  // e0f0g0h0 | e4f4g4h4
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);  // e1f1g1h1 | e5f5g5h5
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);  // e2f2g2h2 | e6f6g6h6
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);  // e3f3g3h3 | e7f7g7h7

    // Phase 3: Permute 128-bit lanes to complete the transpose
    // permute2f128 with 0x20 = [lo_lane(a), lo_lane(b)]
    // permute2f128 with 0x31 = [hi_lane(a), hi_lane(b)]
    __m256 d0 = _mm256_permute2f128_ps(s0, s4, 0x20);  // col 0: a0b0c0d0e0f0g0h0
    __m256 d1 = _mm256_permute2f128_ps(s1, s5, 0x20);  // col 1: a1b1c1d1e1f1g1h1
    __m256 d2 = _mm256_permute2f128_ps(s2, s6, 0x20);  // col 2: a2b2c2d2e2f2g2h2
    __m256 d3 = _mm256_permute2f128_ps(s3, s7, 0x20);  // col 3: a3b3c3d3e3f3g3h3
    __m256 d4 = _mm256_permute2f128_ps(s0, s4, 0x31);  // col 4: a4b4c4d4e4f4g4h4
    __m256 d5 = _mm256_permute2f128_ps(s1, s5, 0x31);  // col 5: a5b5c5d5e5f5g5h5
    __m256 d6 = _mm256_permute2f128_ps(s2, s6, 0x31);  // col 6: a6b6c6d6e6f6g6h6
    __m256 d7 = _mm256_permute2f128_ps(s3, s7, 0x31);  // col 7: a7b7c7d7e7f7g7h7

    // Store 8 transposed rows
    _mm256_storeu_ps(dst + 0 * dst_stride, d0);
    _mm256_storeu_ps(dst + 1 * dst_stride, d1);
    _mm256_storeu_ps(dst + 2 * dst_stride, d2);
    _mm256_storeu_ps(dst + 3 * dst_stride, d3);
    _mm256_storeu_ps(dst + 4 * dst_stride, d4);
    _mm256_storeu_ps(dst + 5 * dst_stride, d5);
    _mm256_storeu_ps(dst + 6 * dst_stride, d6);
    _mm256_storeu_ps(dst + 7 * dst_stride, d7);
#else
    // Scalar fallback
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            dst[j * dst_stride + i] = src[i * src_stride + j];
#endif
}

/// Block-wise transpose for large matrices.
/// Uses 8x8 SIMD blocks with scalar cleanup for edges.
inline void transpose_blocked(const float* src, float* dst,
                               int64_t rows, int64_t cols,
                               int64_t src_stride, int64_t dst_stride) {
    // Process full 8x8 blocks
    int64_t row_blocks = rows / 8;
    int64_t col_blocks = cols / 8;

    for (int64_t rb = 0; rb < row_blocks; ++rb) {
        for (int64_t cb = 0; cb < col_blocks; ++cb) {
            transpose_8x8_avx2(
                src + rb * 8 * src_stride + cb * 8,
                src_stride,
                dst + cb * 8 * dst_stride + rb * 8,
                dst_stride
            );
        }
    }

    // Handle remaining columns (right edge, width < 8)
    int64_t col_rem = cols - col_blocks * 8;
    if (col_rem > 0) {
        for (int64_t rb = 0; rb < row_blocks; ++rb) {
            int64_t r_start = rb * 8;
            int64_t c_start = col_blocks * 8;
            for (int64_t i = 0; i < 8; ++i) {
                for (int64_t j = 0; j < col_rem; ++j) {
                    dst[(c_start + j) * dst_stride + (r_start + i)] =
                        src[(r_start + i) * src_stride + (c_start + j)];
                }
            }
        }
    }

    // Handle remaining rows (bottom edge, height < 8)
    int64_t row_rem = rows - row_blocks * 8;
    if (row_rem > 0) {
        int64_t r_start = row_blocks * 8;
        for (int64_t i = 0; i < row_rem; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                dst[j * dst_stride + (r_start + i)] =
                    src[(r_start + i) * src_stride + j];
            }
        }
    }
}

} // namespace tenzor::cpu
