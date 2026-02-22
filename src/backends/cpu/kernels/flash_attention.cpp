#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/dtype.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD intrinsics
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__AVX512F__)
        #define TENZOR_FLASH_AVX512
    #endif
    #if defined(__AVX2__) && defined(__FMA__)
        #define TENZOR_FLASH_AVX2
    #endif
#endif

namespace tenzor::cpu {

// ============================================================================
// Philox 4x32-10 Counter-Based PRNG for Dropout
// ============================================================================
// Philox is a counter-based RNG ideal for parallel dropout:
// - No shared mutable state between threads (each position has a unique counter)
// - Deterministic: same (seed, counter) always produces the same output
// - Statistically excellent (passes BigCrush)
// - Very fast: just a few multiplies and XORs per sample

struct Philox4x32 {
    uint32_t counter[4];
    uint32_t key[2];

    /// Single Philox round: multiply-and-xor mixing
    static void philox_round(uint32_t* ctr, const uint32_t* key) {
        // Philox constants (from the original paper)
        constexpr uint64_t M0 = 0xD2511F53ULL;
        constexpr uint64_t M1 = 0xCD9E8D57ULL;

        uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
        uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);

        uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
        uint32_t lo0 = static_cast<uint32_t>(prod0);
        uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
        uint32_t lo1 = static_cast<uint32_t>(prod1);

        uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
        uint32_t new1 = lo1;
        uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
        uint32_t new3 = lo0;

        ctr[0] = new0;
        ctr[1] = new1;
        ctr[2] = new2;
        ctr[3] = new3;
    }

    /// Bump the key (Weyl sequence) between rounds
    static void bump_key(uint32_t* key) {
        constexpr uint32_t W0 = 0x9E3779B9U;  // golden ratio
        constexpr uint32_t W1 = 0xBB67AE85U;  // sqrt(3) - 1
        key[0] += W0;
        key[1] += W1;
    }

    /// Generate 4 uniform uint32 values from (seed, counter position)
    /// Uses 10 rounds (Philox4x32-10) for full statistical quality
    void generate(uint32_t output[4]) const {
        uint32_t ctr[4] = {counter[0], counter[1], counter[2], counter[3]};
        uint32_t k[2] = {key[0], key[1]};

        // 10 rounds of Philox
        for (int r = 0; r < 10; ++r) {
            philox_round(ctr, k);
            bump_key(k);
        }

        output[0] = ctr[0];
        output[1] = ctr[1];
        output[2] = ctr[2];
        output[3] = ctr[3];
    }

    /// Convert a uint32 to a uniform float in [0, 1)
    static float uint32_to_uniform(uint32_t x) {
        // Use top 24 bits for mantissa (ensures uniform distribution)
        return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);  // 1 / 2^24
    }
};

// ============================================================================
// SIMD Horizontal Reduction Helpers
// ============================================================================

#ifdef TENZOR_FLASH_AVX2
/// Horizontal sum of 8 floats in an __m256 register
inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

#ifdef TENZOR_FLASH_AVX512
/// Horizontal sum of 16 floats in an __m512 register
inline float hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif

// ============================================================================
// Vectorized micro-kernels for flash attention hot loops
// ============================================================================

/// Dot product of two float arrays of length `len`
inline float dot_product(const float* a, const float* b, int64_t len) {
    float result = 0.0f;
    int64_t d = 0;

#if defined(TENZOR_FLASH_AVX512)
    {
        __m512 acc = _mm512_setzero_ps();
        for (; d + 16 <= len; d += 16) {
            __m512 va = _mm512_loadu_ps(a + d);
            __m512 vb = _mm512_loadu_ps(b + d);
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        result += hsum_avx512(acc);
    }
#elif defined(TENZOR_FLASH_AVX2)
    {
        __m256 acc = _mm256_setzero_ps();
        for (; d + 8 <= len; d += 8) {
            __m256 va = _mm256_loadu_ps(a + d);
            __m256 vb = _mm256_loadu_ps(b + d);
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        result += hsum_avx2(acc);
    }
#endif

    // Scalar tail
    for (; d < len; ++d) {
        result += a[d] * b[d];
    }
    return result;
}

/// In-place scale: out[d] *= scale, for d in [0, len)
inline void scale_vector(float* out, float scale, int64_t len) {
    int64_t d = 0;

#if defined(TENZOR_FLASH_AVX512)
    {
        __m512 vs = _mm512_set1_ps(scale);
        for (; d + 16 <= len; d += 16) {
            __m512 v = _mm512_loadu_ps(out + d);
            _mm512_storeu_ps(out + d, _mm512_mul_ps(v, vs));
        }
    }
#elif defined(TENZOR_FLASH_AVX2)
    {
        __m256 vs = _mm256_set1_ps(scale);
        for (; d + 8 <= len; d += 8) {
            __m256 v = _mm256_loadu_ps(out + d);
            _mm256_storeu_ps(out + d, _mm256_mul_ps(v, vs));
        }
    }
#endif

    // Scalar tail
    for (; d < len; ++d) {
        out[d] *= scale;
    }
}

/// Fused multiply-add: out[d] += weight * src[d], for d in [0, len)
inline void fma_vector(float* out, float weight, const float* src, int64_t len) {
    int64_t d = 0;

#if defined(TENZOR_FLASH_AVX512)
    {
        __m512 vw = _mm512_set1_ps(weight);
        for (; d + 16 <= len; d += 16) {
            __m512 vo = _mm512_loadu_ps(out + d);
            __m512 vs = _mm512_loadu_ps(src + d);
            _mm512_storeu_ps(out + d, _mm512_fmadd_ps(vw, vs, vo));
        }
    }
#elif defined(TENZOR_FLASH_AVX2)
    {
        __m256 vw = _mm256_set1_ps(weight);
        for (; d + 8 <= len; d += 8) {
            __m256 vo = _mm256_loadu_ps(out + d);
            __m256 vs = _mm256_loadu_ps(src + d);
            _mm256_storeu_ps(out + d, _mm256_fmadd_ps(vw, vs, vo));
        }
    }
#endif

    // Scalar tail
    for (; d < len; ++d) {
        out[d] += weight * src[d];
    }
}

// Tiled Flash Attention forward with O(N) memory and fused dropout
// Algorithm: For each block of Q, iterate over blocks of K,V:
//   S_block = Q_block @ K_block^T / sqrt(d)
//   Apply causal mask if needed
//   Track running max for numerical stability
//   P_block = exp(S_block - running_max)
//   If training with dropout: apply Philox-based dropout mask to P_block
//   Update running sum for softmax denominator
//   O_block += P_block @ V_block
// Finally rescale O_block by softmax denominator
//
// Dropout is applied to each attention weight (post-softmax) using a
// Philox 4x32-10 counter-based PRNG. The counter is derived from
// (batch, head, query_pos, key_pos) so every element has a unique,
// deterministic random value. Inverted dropout scales surviving weights
// by 1/(1-p) so the expected value is preserved without rescaling at
// inference time.

auto flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                              float scale, bool causal,
                              float dropout_p, bool is_training) -> Tensor {
    // Q, K, V: [batch, num_heads, seq_len, head_dim]
    auto shape = Q.shape();
    int64_t batch = shape[0];
    int64_t num_heads = shape[1];
    int64_t seq_len = shape[2];
    int64_t head_dim = shape[3];

    // Block sizes tuned for L2 cache (~256KB)
    constexpr int64_t BLOCK_Q = 64;
    constexpr int64_t BLOCK_KV = 64;

    // Output tensor
    auto O = zeros({batch, num_heads, seq_len, head_dim}, Q.dtype(), Q.device());

    if (Q.dtype() != DType::Float32) {
        throw std::runtime_error("Flash attention currently only supports Float32");
    }

    const float* q_data = Q.data<float>();
    const float* k_data = K.data<float>();
    const float* v_data = V.data<float>();
    float* o_data = O.data<float>();

    // Dropout configuration
    const bool apply_dropout = is_training && dropout_p > 0.0f;
    const float dropout_scale = apply_dropout ? 1.0f / (1.0f - dropout_p) : 1.0f;

    // Seed for Philox RNG. Use a fixed seed derived from the tensor data pointer
    // for reproducibility within a single forward pass. For true randomness across
    // calls, the caller should vary the seed (e.g., via an attribute).
    // We mix in the data pointer to get a different stream per call.
    const uint32_t rng_seed = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(q_data) * 2654435761U);

    // Parallelize over batch and heads
    #pragma omp parallel for collapse(2) if(batch * num_heads > 1)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            int64_t bh_offset = (b * num_heads + h) * seq_len * head_dim;
            const float* q_bh = q_data + bh_offset;
            const float* k_bh = k_data + bh_offset;
            const float* v_bh = v_data + bh_offset;
            float* o_bh = o_data + bh_offset;

            // Per-row running max and sum for online softmax
            std::vector<float> row_max(seq_len, -std::numeric_limits<float>::infinity());
            std::vector<float> row_sum(seq_len, 0.0f);

            // Process Q blocks
            for (int64_t q_start = 0; q_start < seq_len; q_start += BLOCK_Q) {
                int64_t q_end = std::min(q_start + BLOCK_Q, seq_len);

                // Process K/V blocks
                for (int64_t kv_start = 0; kv_start < seq_len; kv_start += BLOCK_KV) {
                    int64_t kv_end = std::min(kv_start + BLOCK_KV, seq_len);

                    // Compute S_block = Q_block @ K_block^T * scale
                    // S_block: [block_q, block_kv]
                    for (int64_t qi = q_start; qi < q_end; ++qi) {
                        // Causal: skip this qi's contribution from this KV block
                        // if all K positions in this block are after qi
                        if (causal && kv_start > qi) continue;

                        const float* q_row = q_bh + qi * head_dim;

                        for (int64_t ki = kv_start; ki < kv_end; ++ki) {
                            // Causal mask
                            if (causal && ki > qi) {
                                continue;
                            }

                            const float* k_row = k_bh + ki * head_dim;

                            // Dot product (SIMD-vectorized)
                            float dot = dot_product(q_row, k_row, head_dim) * scale;

                            // Online softmax update
                            float prev_max = row_max[qi];
                            float new_max = std::max(prev_max, dot);

                            // Rescale previous accumulator
                            float rescale = std::exp(prev_max - new_max);
                            row_sum[qi] = row_sum[qi] * rescale;

                            // Rescale previous output (SIMD-vectorized)
                            float* o_row = o_bh + qi * head_dim;
                            scale_vector(o_row, rescale, head_dim);

                            // Add current contribution
                            float weight = std::exp(dot - new_max);

                            // Apply dropout to the attention weight (post-softmax)
                            // Philox counter: (batch, head, qi, ki) uniquely identifies
                            // each attention weight element.
                            if (apply_dropout) {
                                Philox4x32 philox;
                                philox.counter[0] = static_cast<uint32_t>(b);
                                philox.counter[1] = static_cast<uint32_t>(h);
                                philox.counter[2] = static_cast<uint32_t>(qi);
                                philox.counter[3] = static_cast<uint32_t>(ki);
                                philox.key[0] = rng_seed;
                                philox.key[1] = rng_seed ^ 0x1BD11BDAU;  // secondary key

                                uint32_t rng_out[4];
                                philox.generate(rng_out);

                                float rand_val = Philox4x32::uint32_to_uniform(rng_out[0]);
                                if (rand_val < dropout_p) {
                                    weight = 0.0f;  // Drop this attention connection
                                } else {
                                    weight *= dropout_scale;  // Inverted dropout scaling
                                }
                            }

                            row_sum[qi] += weight;

                            // Weighted V accumulation (SIMD-vectorized FMA)
                            const float* v_row = v_bh + ki * head_dim;
                            fma_vector(o_row, weight, v_row, head_dim);

                            row_max[qi] = new_max;
                        }
                    }
                }
            }

            // Final normalization by softmax denominator (SIMD-vectorized)
            for (int64_t qi = 0; qi < seq_len; ++qi) {
                if (row_sum[qi] > 0.0f) {
                    float inv_sum = 1.0f / row_sum[qi];
                    float* o_row = o_bh + qi * head_dim;
                    scale_vector(o_row, inv_sum, head_dim);
                }
            }
        }
    }

    return O;
}

} // namespace tenzor::cpu
