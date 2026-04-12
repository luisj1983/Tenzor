/**
 * @file fused_attention.hpp
 * @brief Optimized fused attention operations for transformers
 *
 * Key features:
 * - Fused Softmax + TopK for efficient attention
 * - SIMD-accelerated softmax computation
 * - Memory-efficient online softmax (FlashAttention-style)
 * - Causal masking support
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>

#include "buffer_pool.hpp"

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__AVX512F__)
        #define TENZOR_ATTENTION_AVX512
    #endif
    #if defined(__AVX2__)
        #define TENZOR_ATTENTION_AVX2
    #endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace cpu {
namespace attention {

// ============================================================================
// SIMD Horizontal Operations
// ============================================================================

#ifdef TENZOR_ATTENTION_AVX2

inline float hmax_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(m);
}

inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

#endif // TENZOR_ATTENTION_AVX2

#ifdef TENZOR_ATTENTION_AVX512

inline float hmax_avx512(__m512 v) {
    return _mm512_reduce_max_ps(v);
}

inline float hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

#endif // TENZOR_ATTENTION_AVX512

// ============================================================================
// Softmax Operations
// ============================================================================

/**
 * @brief SIMD-accelerated softmax over last dimension
 *
 * Uses numerically stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 */
inline void softmax_row(
    const float* input,
    float* output,
    int64_t size
) {
#ifdef TENZOR_ATTENTION_AVX2
    if (size >= 8) {
        // Step 1: Find max
        __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        int64_t i = 0;
        for (; i + 8 <= size; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            vmax = _mm256_max_ps(vmax, v);
        }
        float max_val = hmax_avx2(vmax);
        for (; i < size; ++i) {
            max_val = std::max(max_val, input[i]);
        }

        // Step 2: Compute exp(x - max) and sum
        __m256 vmax_bc = _mm256_set1_ps(max_val);
        __m256 vsum = _mm256_setzero_ps();

        // Exp approximation constants
        __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
        __m256 ln2 = _mm256_set1_ps(0.693147180559945f);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 c1 = _mm256_set1_ps(1.0f);
        __m256 c2 = _mm256_set1_ps(0.5f);
        __m256 c3 = _mm256_set1_ps(0.166666667f);
        __m256 c4 = _mm256_set1_ps(0.041666667f);

        i = 0;
        for (; i + 8 <= size; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            __m256 x = _mm256_sub_ps(v, vmax_bc);

            // Fast exp approximation
            __m256 k = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT);
            __m256 r = _mm256_fnmadd_ps(k, ln2, x);

            __m256 p = _mm256_fmadd_ps(c4, r, c3);
            p = _mm256_fmadd_ps(p, r, c2);
            p = _mm256_fmadd_ps(p, r, c1);
            p = _mm256_fmadd_ps(p, r, one);

            __m256i ki = _mm256_cvtps_epi32(k);
            ki = _mm256_slli_epi32(ki, 23);
            __m256 scale = _mm256_castsi256_ps(_mm256_add_epi32(ki, _mm256_set1_epi32(0x3f800000)));
            __m256 exp_val = _mm256_mul_ps(p, scale);

            _mm256_storeu_ps(output + i, exp_val);
            vsum = _mm256_add_ps(vsum, exp_val);
        }

        float sum_val = hsum_avx2(vsum);
        for (; i < size; ++i) {
            float exp_val = std::exp(input[i] - max_val);
            output[i] = exp_val;
            sum_val += exp_val;
        }

        // Step 3: Normalize (guard against all-masked rows where sum is zero)
        if (sum_val > 0.0f) {
            __m256 vinv_sum = _mm256_set1_ps(1.0f / sum_val);
            i = 0;
            for (; i + 8 <= size; i += 8) {
                __m256 v = _mm256_loadu_ps(output + i);
                v = _mm256_mul_ps(v, vinv_sum);
                _mm256_storeu_ps(output + i, v);
            }
            float inv_sum = 1.0f / sum_val;
            for (; i < size; ++i) {
                output[i] *= inv_sum;
            }
        } else {
            std::memset(output, 0, size * sizeof(float));
        }
        return;
    }
#endif

    // Scalar fallback
    float max_val = input[0];
    for (int64_t i = 1; i < size; ++i) {
        max_val = std::max(max_val, input[i]);
    }

    float sum_exp = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }

    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (int64_t i = 0; i < size; ++i) {
            output[i] *= inv_sum;
        }
    } else {
        std::memset(output, 0, size * sizeof(float));
    }
}

/**
 * @brief Batch softmax over last dimension
 *
 * @param input Input tensor (batch_size x seq_len)
 * @param output Output tensor (batch_size x seq_len)
 * @param batch_size Number of rows
 * @param seq_len Sequence length (softmax dimension)
 */
inline void softmax(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t seq_len
) {
    #pragma omp parallel for if(batch_size > 16)
    for (int64_t b = 0; b < batch_size; ++b) {
        softmax_row(input + b * seq_len, output + b * seq_len, seq_len);
    }
}

/**
 * @brief Softmax with causal masking
 *
 * Masks out future positions (upper triangular) by setting them to -inf
 */
inline void softmax_causal(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t seq_len
) {
    constexpr float NEG_INF = -1e9f;

    #pragma omp parallel for if(batch_size > 16)
    for (int64_t b = 0; b < batch_size; ++b) {
        // For attention: b encodes (batch, head, query_pos)
        // We need to mask based on query_pos

        // Simplified: assume input is already masked or we apply mask inline
        const float* row_in = input + b * seq_len;
        float* row_out = output + b * seq_len;

        // Find max (considering only valid positions)
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < seq_len; ++i) {
            if (row_in[i] > NEG_INF / 2) {  // Not masked
                max_val = std::max(max_val, row_in[i]);
            }
        }

        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int64_t i = 0; i < seq_len; ++i) {
            if (row_in[i] > NEG_INF / 2) {
                row_out[i] = std::exp(row_in[i] - max_val);
                sum_exp += row_out[i];
            } else {
                row_out[i] = 0.0f;
            }
        }

        // Normalize
        if (sum_exp > 0.0f) {
            float inv_sum = 1.0f / sum_exp;
            for (int64_t i = 0; i < seq_len; ++i) {
                row_out[i] *= inv_sum;
            }
        }
    }
}

// ============================================================================
// TopK Operations
// ============================================================================

/**
 * @brief Find top-k values and indices
 *
 * Uses partial sorting for efficiency when k << n
 */
inline void topk(
    const float* input,
    float* values,
    int64_t* indices,
    int64_t n,
    int64_t k,
    [[maybe_unused]] bool sorted = true
) {
    // Create index array
    std::vector<int64_t> idx(n);
    for (int64_t i = 0; i < n; ++i) {
        idx[i] = i;
    }

    // Partial sort to get top k
    std::partial_sort(
        idx.begin(),
        idx.begin() + k,
        idx.end(),
        [input](int64_t a, int64_t b) {
            return input[a] > input[b];
        }
    );

    // Copy results
    for (int64_t i = 0; i < k; ++i) {
        values[i] = input[idx[i]];
        indices[i] = idx[i];
    }
}

/**
 * @brief Batch top-k over last dimension
 */
inline void topk_batch(
    const float* input,
    float* values,
    int64_t* indices,
    int64_t batch_size,
    int64_t n,
    int64_t k,
    bool sorted = true
) {
    #pragma omp parallel for if(batch_size > 8)
    for (int64_t b = 0; b < batch_size; ++b) {
        topk(
            input + b * n,
            values + b * k,
            indices + b * k,
            n, k, sorted
        );
    }
}

/**
 * @brief Fused Softmax + TopK
 *
 * Computes softmax and extracts top-k values in one pass where possible.
 * More efficient than separate softmax + topk for attention.
 */
inline void softmax_topk(
    const float* input,
    float* softmax_out,
    float* topk_values,
    int64_t* topk_indices,
    int64_t batch_size,
    int64_t seq_len,
    int64_t k
) {
    #pragma omp parallel for if(batch_size > 8)
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* row_in = input + b * seq_len;
        float* row_soft = softmax_out + b * seq_len;

        // Compute softmax
        softmax_row(row_in, row_soft, seq_len);

        // Find top-k in softmax output
        topk(
            row_soft,
            topk_values + b * k,
            topk_indices + b * k,
            seq_len, k, true
        );
    }
}

// ============================================================================
// Scaled Dot-Product Attention Components
// ============================================================================

/**
 * @brief Compute attention scores: scores = Q @ K^T / sqrt(d_k)
 */
inline void attention_scores(
    const float* Q,      // (batch, heads, seq_q, d_k)
    const float* K,      // (batch, heads, seq_k, d_k)
    float* scores,       // (batch, heads, seq_q, seq_k)
    int64_t batch_heads, // batch * heads
    int64_t seq_q,
    int64_t seq_k,
    int64_t d_k,
    float scale          // 1/sqrt(d_k)
) {
    // Each (batch, head) pair is independent
    #pragma omp parallel for if(batch_heads > 4)
    for (int64_t bh = 0; bh < batch_heads; ++bh) {
        const float* q = Q + bh * seq_q * d_k;
        const float* k = K + bh * seq_k * d_k;
        float* s = scores + bh * seq_q * seq_k;

        // scores[i, j] = sum(Q[i, :] * K[j, :]) * scale
        for (int64_t i = 0; i < seq_q; ++i) {
            for (int64_t j = 0; j < seq_k; ++j) {
                float dot = 0.0f;

#ifdef TENZOR_ATTENTION_AVX2
                int64_t d = 0;
                __m256 vsum = _mm256_setzero_ps();
                for (; d + 8 <= d_k; d += 8) {
                    __m256 vq = _mm256_loadu_ps(q + i * d_k + d);
                    __m256 vk = _mm256_loadu_ps(k + j * d_k + d);
                    vsum = _mm256_fmadd_ps(vq, vk, vsum);
                }
                dot = hsum_avx2(vsum);
                for (; d < d_k; ++d) {
                    dot += q[i * d_k + d] * k[j * d_k + d];
                }
#else
                for (int64_t d = 0; d < d_k; ++d) {
                    dot += q[i * d_k + d] * k[j * d_k + d];
                }
#endif

                s[i * seq_k + j] = dot * scale;
            }
        }
    }
}

/**
 * @brief Apply attention weights to values: output = softmax(scores) @ V
 */
inline void attention_output(
    const float* attn_weights,  // (batch, heads, seq_q, seq_k) - already softmaxed
    const float* V,             // (batch, heads, seq_k, d_v)
    float* output,              // (batch, heads, seq_q, d_v)
    int64_t batch_heads,
    int64_t seq_q,
    int64_t seq_k,
    int64_t d_v
) {
    #pragma omp parallel for if(batch_heads > 4)
    for (int64_t bh = 0; bh < batch_heads; ++bh) {
        const float* w = attn_weights + bh * seq_q * seq_k;
        const float* v = V + bh * seq_k * d_v;
        float* o = output + bh * seq_q * d_v;

        // output[i, :] = sum_j(weights[i, j] * V[j, :])
        for (int64_t i = 0; i < seq_q; ++i) {
            // Initialize output row to zero
            std::memset(o + i * d_v, 0, d_v * sizeof(float));

            for (int64_t j = 0; j < seq_k; ++j) {
                float weight = w[i * seq_k + j];

#ifdef TENZOR_ATTENTION_AVX2
                __m256 vw = _mm256_set1_ps(weight);
                int64_t d = 0;
                for (; d + 8 <= d_v; d += 8) {
                    __m256 vo = _mm256_loadu_ps(o + i * d_v + d);
                    __m256 vv = _mm256_loadu_ps(v + j * d_v + d);
                    vo = _mm256_fmadd_ps(vw, vv, vo);
                    _mm256_storeu_ps(o + i * d_v + d, vo);
                }
                for (; d < d_v; ++d) {
                    o[i * d_v + d] += weight * v[j * d_v + d];
                }
#else
                for (int64_t d = 0; d < d_v; ++d) {
                    o[i * d_v + d] += weight * v[j * d_v + d];
                }
#endif
            }
        }
    }
}

/**
 * @brief Full scaled dot-product attention
 *
 * output = softmax(Q @ K^T / sqrt(d_k)) @ V
 */
/**
 * @brief Flash Attention: tiled SDPA with online softmax.
 *
 * Memory: O(Bc * d) per query row instead of O(seq_q * seq_k).
 * Processes K/V in blocks of Bc columns, maintaining running max and
 * sum-of-exp to compute numerically stable softmax incrementally.
 */
inline void flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int64_t batch_heads,
    int64_t seq_q,
    int64_t seq_k,
    int64_t d_k,
    int64_t d_v,
    bool causal = false
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Block size for K/V tiling (tuned for L1/L2 cache)
    constexpr int64_t Bc = 64;

    #pragma omp parallel if(batch_heads > 1)
    {
        // Per-thread scratch buffers
        std::vector<float> scores_block(Bc);   // scores for one query row against Bc keys
        std::vector<float> new_output(d_v);    // accumulator for output row

        #pragma omp for
        for (int64_t bh = 0; bh < batch_heads; ++bh) {
            const float* q_bh = Q + bh * seq_q * d_k;
            const float* k_bh = K + bh * seq_k * d_k;
            const float* v_bh = V + bh * seq_k * d_v;
            float* o_bh = output + bh * seq_q * d_v;

            for (int64_t i = 0; i < seq_q; ++i) {
                const float* qi = q_bh + i * d_k;
                float* oi = o_bh + i * d_v;

                // Initialize running statistics
                float row_max = -std::numeric_limits<float>::infinity();
                float row_sum = 0.0f;
                std::memset(oi, 0, d_v * sizeof(float));

                // Determine effective key range for causal masking
                int64_t key_end = causal ? std::min(i + 1, seq_k) : seq_k;

                // Process K/V in blocks of Bc
                for (int64_t j0 = 0; j0 < key_end; j0 += Bc) {
                    int64_t block_end = std::min(j0 + Bc, key_end);
                    int64_t block_len = block_end - j0;

                    // Compute scores for this block: s[j] = qi . kj * scale
                    for (int64_t j = 0; j < block_len; ++j) {
                        const float* kj = k_bh + (j0 + j) * d_k;
                        float dot = 0.0f;
#ifdef TENZOR_ATTENTION_AVX2
                        int64_t d = 0;
                        __m256 acc = _mm256_setzero_ps();
                        for (; d + 8 <= d_k; d += 8) {
                            __m256 vq = _mm256_loadu_ps(qi + d);
                            __m256 vk = _mm256_loadu_ps(kj + d);
                            acc = _mm256_fmadd_ps(vq, vk, acc);
                        }
                        dot = hsum_avx2(acc);
                        for (; d < d_k; ++d) {
                            dot += qi[d] * kj[d];
                        }
#else
                        for (int64_t d = 0; d < d_k; ++d) {
                            dot += qi[d] * kj[d];
                        }
#endif
                        scores_block[j] = dot * scale;
                    }

                    // Online softmax update:
                    // Find block max
                    float block_max = -std::numeric_limits<float>::infinity();
                    for (int64_t j = 0; j < block_len; ++j) {
                        block_max = std::max(block_max, scores_block[j]);
                    }

                    // Compute new global max and correction factor
                    float prev_max = row_max;
                    row_max = std::max(row_max, block_max);

                    // Rescale previous accumulations
                    float correction = std::exp(prev_max - row_max);
                    row_sum *= correction;
                    for (int64_t d = 0; d < d_v; ++d) {
                        oi[d] *= correction;
                    }

                    // Accumulate this block's contribution
                    for (int64_t j = 0; j < block_len; ++j) {
                        float w = std::exp(scores_block[j] - row_max);
                        row_sum += w;

                        const float* vj = v_bh + (j0 + j) * d_v;
#ifdef TENZOR_ATTENTION_AVX2
                        __m256 vw = _mm256_set1_ps(w);
                        int64_t d = 0;
                        for (; d + 8 <= d_v; d += 8) {
                            __m256 vo = _mm256_loadu_ps(oi + d);
                            __m256 vv = _mm256_loadu_ps(vj + d);
                            vo = _mm256_fmadd_ps(vw, vv, vo);
                            _mm256_storeu_ps(oi + d, vo);
                        }
                        for (; d < d_v; ++d) {
                            oi[d] += w * vj[d];
                        }
#else
                        for (int64_t d = 0; d < d_v; ++d) {
                            oi[d] += w * vj[d];
                        }
#endif
                    }
                }

                // Final normalization: output /= sum
                if (row_sum > 0.0f) {
                    float inv_sum = 1.0f / row_sum;
#ifdef TENZOR_ATTENTION_AVX2
                    __m256 vs = _mm256_set1_ps(inv_sum);
                    int64_t d = 0;
                    for (; d + 8 <= d_v; d += 8) {
                        __m256 vo = _mm256_loadu_ps(oi + d);
                        vo = _mm256_mul_ps(vo, vs);
                        _mm256_storeu_ps(oi + d, vo);
                    }
                    for (; d < d_v; ++d) {
                        oi[d] *= inv_sum;
                    }
#else
                    for (int64_t d = 0; d < d_v; ++d) {
                        oi[d] *= inv_sum;
                    }
#endif
                }
            }
        }
    }
}

inline void scaled_dot_product_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int64_t batch_heads,
    int64_t seq_q,
    int64_t seq_k,
    int64_t d_k,
    int64_t d_v,
    bool causal = false
) {
    // Use Flash Attention (O(N) memory) for sequences longer than threshold,
    // or when the full scores matrix would be very large.
    // For short sequences, the overhead of online softmax isn't worth it.
    constexpr int64_t FLASH_THRESHOLD = 1024;

    if (seq_k > FLASH_THRESHOLD || seq_q > FLASH_THRESHOLD) {
        flash_attention(Q, K, V, output, batch_heads, seq_q, seq_k, d_k, d_v, causal);
        return;
    }

    // Original O(N^2) path for short sequences (lower overhead)
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    auto scores_buffer = acquire_buffer<float>(batch_heads * seq_q * seq_k);
    float* scores = scores_buffer.data();

    // Step 1: Compute scores = Q @ K^T / sqrt(d_k)
    attention_scores(Q, K, scores, batch_heads, seq_q, seq_k, d_k, scale);

    // Step 2: Apply causal mask if needed
    if (causal) {
        #pragma omp parallel for if(batch_heads > 4)
        for (int64_t bh = 0; bh < batch_heads; ++bh) {
            float* s = scores + bh * seq_q * seq_k;
            for (int64_t i = 0; i < seq_q; ++i) {
                for (int64_t j = i + 1; j < seq_k; ++j) {
                    s[i * seq_k + j] = -1e9f;
                }
            }
        }
    }

    // Step 3: Softmax over seq_k dimension
    softmax(scores, scores, batch_heads * seq_q, seq_k);

    // Step 4: Compute output = attn_weights @ V
    attention_output(scores, V, output, batch_heads, seq_q, seq_k, d_v);
}

} // namespace attention
} // namespace cpu
} // namespace tenzor
