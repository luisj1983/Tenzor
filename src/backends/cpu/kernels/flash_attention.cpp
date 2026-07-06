#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/dtype.hpp"
#include "philox.hpp"
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
// Philox 4x32-10 Counter-Based PRNG for Dropout — shared from philox.hpp
// ============================================================================
// Bring the shared implementation into this scope for existing usages below.
using Philox4x32 = tenzor::cpu::philox::Philox4x32;

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
/// E.4: Horizontal sum of 8 doubles in an __m512d register (Float64 path)
inline double hsum_avx512(__m512d v) {
    return _mm512_reduce_add_pd(v);
}
#endif

#ifdef TENZOR_FLASH_AVX2
/// E.4: Horizontal sum of 4 doubles in an __m256d register (Float64 path)
inline double hsum_avx2(__m256d v) {
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d sum = _mm_add_pd(hi, lo);
    sum = _mm_hadd_pd(sum, sum);
    return _mm_cvtsd_f64(sum);
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

// ============================================================================
// E.4: Native Float64 SIMD micro-kernels (mirrors the float versions above).
// Allows flash_attention_backward to compute natively in `double` rather
// than widening to float and narrowing back, preserving full FP64 mantissa
// precision through the GEMM + softmax + dropout chain.
// ============================================================================

inline double dot_product(const double* a, const double* b, int64_t len) {
    double result = 0.0;
    int64_t d = 0;

#if defined(TENZOR_FLASH_AVX512)
    {
        __m512d acc = _mm512_setzero_pd();
        for (; d + 8 <= len; d += 8) {
            __m512d va = _mm512_loadu_pd(a + d);
            __m512d vb = _mm512_loadu_pd(b + d);
            acc = _mm512_fmadd_pd(va, vb, acc);
        }
        result += hsum_avx512(acc);
    }
#elif defined(TENZOR_FLASH_AVX2)
    {
        __m256d acc = _mm256_setzero_pd();
        for (; d + 4 <= len; d += 4) {
            __m256d va = _mm256_loadu_pd(a + d);
            __m256d vb = _mm256_loadu_pd(b + d);
            acc = _mm256_fmadd_pd(va, vb, acc);
        }
        result += hsum_avx2(acc);
    }
#endif

    for (; d < len; ++d) {
        result += a[d] * b[d];
    }
    return result;
}

inline void scale_vector(double* out, double scale, int64_t len) {
    int64_t d = 0;

#if defined(TENZOR_FLASH_AVX512)
    {
        __m512d vs = _mm512_set1_pd(scale);
        for (; d + 8 <= len; d += 8) {
            __m512d v = _mm512_loadu_pd(out + d);
            _mm512_storeu_pd(out + d, _mm512_mul_pd(v, vs));
        }
    }
#elif defined(TENZOR_FLASH_AVX2)
    {
        __m256d vs = _mm256_set1_pd(scale);
        for (; d + 4 <= len; d += 4) {
            __m256d v = _mm256_loadu_pd(out + d);
            _mm256_storeu_pd(out + d, _mm256_mul_pd(v, vs));
        }
    }
#endif

    for (; d < len; ++d) {
        out[d] *= scale;
    }
}

inline void fma_vector(double* out, double weight, const double* src, int64_t len) {
    int64_t d = 0;

#if defined(TENZOR_FLASH_AVX512)
    {
        __m512d vw = _mm512_set1_pd(weight);
        for (; d + 8 <= len; d += 8) {
            __m512d vo = _mm512_loadu_pd(out + d);
            __m512d vs = _mm512_loadu_pd(src + d);
            _mm512_storeu_pd(out + d, _mm512_fmadd_pd(vw, vs, vo));
        }
    }
#elif defined(TENZOR_FLASH_AVX2)
    {
        __m256d vw = _mm256_set1_pd(weight);
        for (; d + 4 <= len; d += 4) {
            __m256d vo = _mm256_loadu_pd(out + d);
            __m256d vs = _mm256_loadu_pd(src + d);
            _mm256_storeu_pd(out + d, _mm256_fmadd_pd(vw, vs, vo));
        }
    }
#endif

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
                              float dropout_p, bool is_training,
                              uint64_t seed_in) -> std::vector<Tensor> {
    // Per docs/internals/attention-contract.md, FlashAttention forward returns
    // (output, logsumexp, philox_seed, philox_offset). LSE is always Float32;
    // seed/offset are Int64 scalar tensors and are empty when dropout_p == 0.
    //
    // Q, K, V: [batch, num_heads, seq_len, head_dim]
    //
    // Callers commonly pass Q/K/V as the output of `permute({0,2,1,3})` on a
    // freshly reshaped [B, Sq, H, head_dim] projection — that's a strided view,
    // not a contiguous tensor. The kernel below uses raw pointer arithmetic
    // (bh_offset = (b*H + h)*L*D) which assumes a contiguous [B,H,L,D] layout,
    // so non-contiguous inputs silently produce wrong outputs (matched the
    // GPU FlashAttention path which goes through `reshape(...)` and implicitly
    // contiguises). Materialise contiguous copies up front to keep the
    // CPU/GPU SDPA paths in agreement.
    if (!Q.is_contiguous() || !K.is_contiguous() || !V.is_contiguous()) {
        return flash_attention_forward(
            Q.is_contiguous() ? Q : Q.contiguous(),
            K.is_contiguous() ? K : K.contiguous(),
            V.is_contiguous() ? V : V.contiguous(),
            scale, causal, dropout_p, is_training, seed_in);
    }

    auto shape = Q.shape();
    int64_t batch = shape[0];
    int64_t num_heads = shape[1];
    int64_t seq_len = shape[2];
    int64_t head_dim = shape[3];
    // F081: K/V carry their own sequence length (kv_len), which differs from Q's
    // seq_len for cross-attention / KV-cache (seq_q != seq_k). Q and O are indexed
    // with seq_len; K and V must be indexed and bounded with kv_len, matching the
    // backward (which uses M = k_shape[2]). Using seq_len for K/V strides/bounds
    // reads past the K/V buffers when seq_q > seq_k and mis-strides heads when
    // seq_q < seq_k.
    const int64_t kv_len = K.shape()[2];

    // F021: bottom-right causal alignment (matches the MHA/GQA manual BMM path
    // and PyTorch, per docs/internals/attention-contract.md). A query at row
    // qi attends to keys ki <= qi + (seq_k - seq_q). For self-attention
    // (seq_q == seq_k) the offset is 0 and this reduces to ki <= qi; for
    // KV-cache cross-attention (seq_q < seq_k) it lets the query see the keys
    // that precede it in the full sequence instead of only key 0.
    const int64_t causal_offset = K.shape()[2] - seq_len;

    // Multi-dtype dispatch (audit A.11):
    //   Float32 → native float typed kernel.
    //   Float64 → native double typed kernel (no Float32 round-trip — keeps
    //             full FP64 mantissa precision through GEMM, softmax, dropout
    //             so autograd gradcheck against the composed-ops backward
    //             matches to FP64 tolerance).
    //   Float16 / BFloat16 → widen to Float32, compute, narrow output. These
    //             dtypes have less mantissa than Float32, so widen-narrow is
    //             mathematically lossless and matches PyTorch SDPA.
    //
    // LSE stays Float32 in every case per the contract (the dynamic range of
    // max + log(sum) exceeds FP16/BF16, and Float32 is sufficient for FP64
    // backward — which in any case recomputes softmax rather than reading LSE).
    if (Q.dtype() == DType::Float16 || Q.dtype() == DType::BFloat16) {
        auto orig_dtype = Q.dtype();
        auto outs_f32 = flash_attention_forward(
            Q.to(DType::Float32), K.to(DType::Float32), V.to(DType::Float32),
            scale, causal, dropout_p, is_training, seed_in);
        // outs_f32: [O_f32, L_f32, seed?, offset?]
        outs_f32[0] = outs_f32[0].to(orig_dtype);
        // L stays Float32. seed/offset are int64 — leave them as-is.
        return outs_f32;
    } else if (Q.dtype() != DType::Float32 && Q.dtype() != DType::Float64) {
        throw std::runtime_error("Flash attention: unsupported dtype " +
                                 std::string(dtype_name(Q.dtype())));
    }

    const bool is_f64 = (Q.dtype() == DType::Float64);

    // Block sizes tuned for L2 cache (~256KB)
    constexpr int64_t BLOCK_Q = 64;
    constexpr int64_t BLOCK_KV = 64;

    // Output tensor (matches input dtype) and LSE buffer (always Float32 per
    // attention contract).
    auto O = zeros({batch, num_heads, seq_len, head_dim}, Q.dtype(), Q.device());
    auto L = zeros({batch, num_heads, seq_len}, DType::Float32, Q.device());

    float* lse_data = L.data<float>();

    // Dropout configuration
    const bool apply_dropout = is_training && dropout_p > 0.0f;

    // Seed for Philox RNG. If the caller passed seed_in == 0 (the default),
    // derive a per-call seed from the data pointer for backward reproducibility
    // within a single forward; the saved seed below is then echoed back so the
    // backward can replay the exact same mask. Use the output buffer's pointer
    // (typed below per dtype) — read the raw storage address generically.
    const void* q_addr = (Q.dtype() == DType::Float64)
        ? static_cast<const void*>(Q.data<double>())
        : static_cast<const void*>(Q.data<float>());
    const uint64_t actual_seed = seed_in != 0
        ? seed_in
        : (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(q_addr)) * 2654435761ULL);
    const uint32_t rng_seed = static_cast<uint32_t>(actual_seed);

    // Templated hot-loop body. T = float (Float32) or double (Float64). All
    // arithmetic — running max, running sum, exp, log, Philox compare —
    // happens in T; the only Float32 cross-over is the LSE write (cast on
    // store) per the contract.
    auto run_typed = [&](auto tag) {
        using T = decltype(tag);
        const T* q_data = Q.data<T>();
        const T* k_data = K.data<T>();
        const T* v_data = V.data<T>();
        T* o_data = O.data<T>();
        const T dropout_p_T = static_cast<T>(dropout_p);
        const T scale_T = static_cast<T>(scale);
        const T dropout_scale_T = apply_dropout ? T(1) / (T(1) - dropout_p_T) : T(1);

        // Parallelize over batch and heads
        #pragma omp parallel for collapse(2) if(batch * num_heads > 1)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < num_heads; ++h) {
                int64_t bh_offset = (b * num_heads + h) * seq_len * head_dim;
                int64_t kv_bh_offset = (b * num_heads + h) * kv_len * head_dim;
                const T* q_bh = q_data + bh_offset;
                const T* k_bh = k_data + kv_bh_offset;
                const T* v_bh = v_data + kv_bh_offset;
                T* o_bh = o_data + bh_offset;

                // Per-row running max and sum for online softmax
                std::vector<T> row_max(seq_len, -std::numeric_limits<T>::infinity());
                std::vector<T> row_sum(seq_len, T(0));

                // Process Q blocks
                for (int64_t q_start = 0; q_start < seq_len; q_start += BLOCK_Q) {
                    int64_t q_end = std::min(q_start + BLOCK_Q, seq_len);

                    // Process K/V blocks
                    for (int64_t kv_start = 0; kv_start < kv_len; kv_start += BLOCK_KV) {
                        int64_t kv_end = std::min(kv_start + BLOCK_KV, kv_len);

                        // Compute S_block = Q_block @ K_block^T * scale
                        // S_block: [block_q, block_kv]
                        for (int64_t qi = q_start; qi < q_end; ++qi) {
                            // Causal: skip this qi's contribution from this KV block
                            // if all K positions in this block are after qi
                            // (bottom-right aligned — see causal_offset, F021).
                            if (causal && kv_start > qi + causal_offset) continue;

                            const T* q_row = q_bh + qi * head_dim;

                            for (int64_t ki = kv_start; ki < kv_end; ++ki) {
                                // Causal mask (bottom-right aligned, F021).
                                if (causal && ki > qi + causal_offset) {
                                    continue;
                                }

                                const T* k_row = k_bh + ki * head_dim;

                                // Dot product (SIMD-vectorized; overloads for float/double)
                                T dot = dot_product(q_row, k_row, head_dim) * scale_T;

                                // Online softmax update
                                T prev_max = row_max[qi];
                                T new_max = std::max(prev_max, dot);

                                // Rescale previous accumulator
                                T rescale = std::exp(prev_max - new_max);
                                row_sum[qi] = row_sum[qi] * rescale;

                                // Rescale previous output (SIMD-vectorized)
                                T* o_row = o_bh + qi * head_dim;
                                scale_vector(o_row, rescale, head_dim);

                                // Add current contribution. `weight` is the
                                // *pre-dropout* softmax numerator exp(dot-new_max);
                                // it always feeds the softmax denominator (row_sum)
                                // so the denominator is the full softmax sum over
                                // ALL keys. Dropout is applied only to the V
                                // accumulation term (`o_weight`), exactly matching
                                // the backward which builds P = exp/sum_ALL first
                                // (Step 3) then masks/scales P (Step 3b). Folding
                                // dropout_scale into row_sum (the old behaviour)
                                // made it cancel against the denominator and turned
                                // the forward into a renormalized-over-survivors
                                // softmax that the backward never differentiates.
                                T weight = std::exp(dot - new_max);
                                T o_weight = weight;

                                // Apply dropout to the V-accumulation weight only
                                // (post-softmax masking). Philox counter:
                                // (batch, head, qi, ki) uniquely identifies each
                                // attention weight element. Philox emits Float32
                                // uniform — promote to T for the compare so the
                                // dropout mask is bit-identical across float and
                                // double paths.
                                if (apply_dropout) {
                                    // Counter words hold (bh=b*num_heads+h, qi,
                                    // ki, 0) truncated to uint32 — the SAME
                                    // bh-combined convention used by the
                                    // CUDA/ROCm/OneAPI/Vulkan FA kernels and by
                                    // the host replay helper
                                    // tenzor::philox_dropout_mask. Folding batch
                                    // and head into one counter word (rather than
                                    // two separate b/h words) is required so the
                                    // dropout backward — which replays the mask
                                    // via philox_dropout_mask's (bh,qi,ki,0)
                                    // counter — enumerates the identical Philox
                                    // stream for B>1 / H>1. Each index is < 2^32
                                    // (a single dim at that size is physically
                                    // unreachable), so the truncation is lossless.
                                    Philox4x32 philox;
                                    philox.counter[0] = static_cast<uint32_t>(b * num_heads + h);
                                    philox.counter[1] = static_cast<uint32_t>(qi);
                                    philox.counter[2] = static_cast<uint32_t>(ki);
                                    philox.counter[3] = 0u;
                                    philox.key[0] = rng_seed;
                                    philox.key[1] = rng_seed ^ 0x1BD11BDAU;  // secondary key

                                    uint32_t rng_out[4];
                                    philox.generate(rng_out);

                                    T rand_val = static_cast<T>(
                                        Philox4x32::uint32_to_uniform(rng_out[0]));
                                    if (rand_val < dropout_p_T) {
                                        o_weight = T(0);  // Drop this attention connection
                                    } else {
                                        o_weight *= dropout_scale_T;  // Inverted dropout scaling
                                    }
                                }

                                // Denominator uses the undropped weight so the
                                // softmax normalization (and LSE) is over the full
                                // key set, not just survivors.
                                row_sum[qi] += weight;

                                // Weighted V accumulation (SIMD-vectorized FMA; overloads for float/double)
                                const T* v_row = v_bh + ki * head_dim;
                                fma_vector(o_row, o_weight, v_row, head_dim);

                                row_max[qi] = new_max;
                            }
                        }
                    }
                }

                // Final normalization by softmax denominator (SIMD-vectorized)
                // and LSE write per-row. LSE = row_max + log(row_sum), with
                // -INFINITY for fully-masked rows (row_sum == 0). The contract
                // requires LSE to be a sentinel that produces zero P in backward
                // (`P = exp(S - L)` with `L=-INFINITY` evaluates to 0).
                float* lse_bh = lse_data + (b * num_heads + h) * seq_len;
                for (int64_t qi = 0; qi < seq_len; ++qi) {
                    if (row_sum[qi] > T(0)) {
                        T inv_sum = T(1) / row_sum[qi];
                        T* o_row = o_bh + qi * head_dim;
                        scale_vector(o_row, inv_sum, head_dim);
                        // LSE remains Float32 per the contract. Compute in T,
                        // then narrow on store.
                        lse_bh[qi] = static_cast<float>(
                            row_max[qi] + std::log(row_sum[qi]));
                    } else {
                        lse_bh[qi] = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
    };

    if (is_f64) {
        run_typed(double{});
    } else {
        run_typed(float{});
    }

    // Allocate seed/offset return tensors only when dropout actually fired —
    // empty Tensors when dropout_p == 0 keeps the contract's "seed/offset are
    // empty if dropout disabled" rule machine-checkable downstream.
    Tensor seed_t;
    Tensor offset_t;
    if (apply_dropout) {
        seed_t = zeros({1}, DType::Int64, Q.device());
        offset_t = zeros({1}, DType::Int64, Q.device());
        seed_t.data<int64_t>()[0] = static_cast<int64_t>(actual_seed);
        // Offset is currently 0: the per-element Philox counter uses the
        // (b,h,qi,ki) tuple directly as the position, so no global offset
        // is needed. Backward reconstructs the same counter from the same
        // tuple and gets the same mask.
        offset_t.data<int64_t>()[0] = 0;
    }
    return {O, L, seed_t, offset_t};
}

// ============================================================================
// Flash Attention Backward (materialized attention matrix; dropout supported
// via dropout_p / philox_seed, reconstructing the same mask as the forward)
// ============================================================================
// Inputs:
//   dO [B, H, N, D]  - gradient of output
//   Q  [B, H, N, D]  - queries
//   K  [B, H, M, D]  - keys
//   V  [B, H, M, D]  - values
//   O  [B, H, N, D]  - forward output (used for D = rowsum(dO * O))
//   scale             - attention scale (typically 1/sqrt(d))
//   causal            - whether to apply causal mask
//
// Algorithm per batch/head:
//   S = Q @ K^T * scale                              [N x M]
//   If causal: S[i,j] = -inf for j > i
//   P = softmax(S, dim=-1)                            [N x M]
//   dV = P^T @ dO                                     [M x D]
//   dP = dO @ V^T                                     [N x M]
//   D_i = rowsum(dO * O, dim=-1)                      [N]  (elementwise then reduce)
//   dS = P * (dP - D_i)                               [N x M]
//   dQ = dS @ K * scale                               [N x D]
//   dK = dS^T @ Q * scale                             [M x D]
//
// Returns {dQ, dK, dV}

// E.4: native Float32/Float64 kernel. Templated on T (float or double) so
// FP64 inputs use cblas_dgemm + `double` arithmetic throughout — no
// widen-narrow precision loss. FP16/BF16 still widen to Float32 (the
// canonical mixed-precision pattern matching PyTorch / FlashAttention-2).
template <typename T>
static auto flash_attention_backward_typed(
    const Tensor& dO, const Tensor& Q, const Tensor& K,
    const Tensor& V, const Tensor& O,
    T scale, bool causal,
    T dropout_p,
    uint64_t philox_seed) -> std::vector<Tensor>
{
    auto q_shape = Q.shape();
    int64_t batch     = q_shape[0];
    int64_t num_heads = q_shape[1];
    int64_t N = q_shape[2];      // query sequence length
    int64_t D = q_shape[3];      // head dimension

    auto k_shape = K.shape();
    int64_t M = k_shape[2];      // key/value sequence length

    // F025: bottom-right causal alignment — MUST match the forward
    // (flash_attention_forward_typed uses causal_offset = M - N). The backward
    // previously masked with j > i (top-left, offset 0), so for seq_q != seq_k
    // (cached-KV / cross-attention) it differentiated a differently-masked
    // softmax than the forward, giving wrong dQ/dK/dV. Mask/zero j > i + offset.
    const int64_t causal_offset = M - N;

    const bool apply_dropout    = dropout_p > T(0) && philox_seed != 0;
    const T    dropout_scale    = apply_dropout ? T(1) / (T(1) - dropout_p) : T(1);
    const uint32_t rng_seed     = static_cast<uint32_t>(philox_seed);

    std::vector<int64_t> q_shape_vec(q_shape.begin(), q_shape.end());
    std::vector<int64_t> k_shape_vec(k_shape.begin(), k_shape.end());
    Tensor dQ = zeros(q_shape_vec, Q.dtype(), Q.device());
    Tensor dK = zeros(k_shape_vec, K.dtype(), K.device());
    Tensor dV = zeros(k_shape_vec, V.dtype(), V.device());

    const T* dO_data = dO.data<T>();
    const T* q_data  = Q.data<T>();
    const T* k_data  = K.data<T>();
    const T* v_data  = V.data<T>();
    const T* o_data  = O.data<T>();
    T* dq_data = dQ.data<T>();
    T* dk_data = dK.data<T>();
    T* dv_data = dV.data<T>();

    #pragma omp parallel for collapse(2) if(batch * num_heads > 1)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            int64_t q_offset  = ((b * num_heads + h) * N) * D;
            int64_t kv_offset = ((b * num_heads + h) * M) * D;

            const T* q_bh  = q_data  + q_offset;
            const T* k_bh  = k_data  + kv_offset;
            const T* v_bh  = v_data  + kv_offset;
            const T* dO_bh = dO_data + q_offset;
            const T* o_bh  = o_data  + q_offset;
            T* dq_bh = dq_data + q_offset;
            T* dk_bh = dk_data + kv_offset;
            T* dv_bh = dv_data + kv_offset;

            std::vector<T> S(static_cast<size_t>(N * M));
            std::vector<T> P(static_cast<size_t>(N * M));
            std::vector<T> D_vec(static_cast<size_t>(N));

            // Step 1: S = Q @ K^T * scale
#ifdef TENZOR_USE_MKL
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            static_cast<MKL_INT>(N), static_cast<MKL_INT>(M), static_cast<MKL_INT>(D),
                            scale,
                            q_bh, static_cast<MKL_INT>(D),
                            k_bh, static_cast<MKL_INT>(D),
                            0.0f,
                            S.data(), static_cast<MKL_INT>(M));
            } else {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            static_cast<MKL_INT>(N), static_cast<MKL_INT>(M), static_cast<MKL_INT>(D),
                            scale,
                            q_bh, static_cast<MKL_INT>(D),
                            k_bh, static_cast<MKL_INT>(D),
                            0.0,
                            S.data(), static_cast<MKL_INT>(M));
            }
#else
            for (int64_t i = 0; i < N; ++i) {
                for (int64_t j = 0; j < M; ++j) {
                    S[i * M + j] = dot_product(q_bh + i * D, k_bh + j * D, D) * scale;
                }
            }
#endif

            // Step 2: causal mask (bottom-right aligned — F025)
            if (causal) {
                for (int64_t i = 0; i < N; ++i) {
                    for (int64_t j = std::max<int64_t>(0, i + causal_offset + 1); j < M; ++j) {
                        S[i * M + j] = -std::numeric_limits<T>::infinity();
                    }
                }
            }

            // Step 3: P = softmax(S, dim=-1) row-wise
            for (int64_t i = 0; i < N; ++i) {
                T max_val = -std::numeric_limits<T>::infinity();
                for (int64_t j = 0; j < M; ++j) {
                    max_val = std::max(max_val, S[i * M + j]);
                }
                T sum_exp = T(0);
                for (int64_t j = 0; j < M; ++j) {
                    T val = std::exp(S[i * M + j] - max_val);
                    P[i * M + j] = val;
                    sum_exp += val;
                }
                T inv_sum = (sum_exp > T(0)) ? T(1) / sum_exp : T(0);
                for (int64_t j = 0; j < M; ++j) {
                    P[i * M + j] *= inv_sum;
                }
            }

            // F051: the softmax-normalization term of dS (the `- D_i` factor)
            // must be multiplied by the UNDROPPED softmax P, not the
            // dropped/scaled P̃. Step 3b mutates P into P̃ in place, so snapshot
            // the true softmax here (only needed when dropout actually fires).
            std::vector<T> P_true;
            if (apply_dropout) {
                P_true.assign(P.begin(), P.end());
            }

            // Step 3b: Replay forward dropout via Philox(seed; counter=
            // bh=b*num_heads+h, i, j, 0) — the bh-combined convention shared by
            // the forward kernel and philox_dropout_mask. Must match the
            // forward exactly or the replayed mask diverges for B>1 / H>1.
            if (apply_dropout) {
                for (int64_t i = 0; i < N; ++i) {
                    for (int64_t j = 0; j < M; ++j) {
                        Philox4x32 philox;
                        philox.counter[0] = static_cast<uint32_t>(b * num_heads + h);
                        philox.counter[1] = static_cast<uint32_t>(i);
                        philox.counter[2] = static_cast<uint32_t>(j);
                        philox.counter[3] = 0u;
                        philox.key[0] = rng_seed;
                        philox.key[1] = rng_seed ^ 0x1BD11BDAU;
                        uint32_t rng_out[4];
                        philox.generate(rng_out);
                        // Philox emits Float32 uniform — promote to T for the
                        // comparison so the dropout mask is bit-identical
                        // across float and double paths.
                        T rand_val = static_cast<T>(
                            Philox4x32::uint32_to_uniform(rng_out[0]));
                        T& p_ref = P[i * M + j];
                        if (rand_val < dropout_p) {
                            p_ref = T(0);
                        } else {
                            p_ref *= dropout_scale;
                        }
                    }
                }
            }

            // Step 4: D_i = rowsum(dO * O, dim=-1)
            for (int64_t i = 0; i < N; ++i) {
                D_vec[i] = dot_product(dO_bh + i * D, o_bh + i * D, D);
            }

            // Step 5: dV = P^T @ dO  [M x D]
#ifdef TENZOR_USE_MKL
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            static_cast<MKL_INT>(M), static_cast<MKL_INT>(D), static_cast<MKL_INT>(N),
                            1.0f,
                            P.data(), static_cast<MKL_INT>(M),
                            dO_bh, static_cast<MKL_INT>(D),
                            0.0f,
                            dv_bh, static_cast<MKL_INT>(D));
            } else {
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            static_cast<MKL_INT>(M), static_cast<MKL_INT>(D), static_cast<MKL_INT>(N),
                            1.0,
                            P.data(), static_cast<MKL_INT>(M),
                            dO_bh, static_cast<MKL_INT>(D),
                            0.0,
                            dv_bh, static_cast<MKL_INT>(D));
            }
#else
            for (int64_t j = 0; j < M; ++j) {
                for (int64_t i = 0; i < N; ++i) {
                    T p_val = P[i * M + j];
                    if (p_val != T(0)) {
                        fma_vector(dv_bh + j * D, p_val, dO_bh + i * D, D);
                    }
                }
            }
#endif

            // Step 6: dP = dO @ V^T, then dS = P̃ * dP - P * D_i [reuse S storage].
            // First factor is the dropped/scaled weight (P, which now holds P̃);
            // the normalization term uses the undropped softmax (P_norm). Without
            // dropout the two coincide, reducing to the classic P * (dP - D_i).
            std::vector<T>& dS = S;
            const T* P_norm = apply_dropout ? P_true.data() : P.data();

#ifdef TENZOR_USE_MKL
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            static_cast<MKL_INT>(N), static_cast<MKL_INT>(M), static_cast<MKL_INT>(D),
                            1.0f,
                            dO_bh, static_cast<MKL_INT>(D),
                            v_bh, static_cast<MKL_INT>(D),
                            0.0f,
                            dS.data(), static_cast<MKL_INT>(M));
            } else {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            static_cast<MKL_INT>(N), static_cast<MKL_INT>(M), static_cast<MKL_INT>(D),
                            1.0,
                            dO_bh, static_cast<MKL_INT>(D),
                            v_bh, static_cast<MKL_INT>(D),
                            0.0,
                            dS.data(), static_cast<MKL_INT>(M));
            }

            for (int64_t i = 0; i < N; ++i) {
                T d_i = D_vec[i];
                for (int64_t j = 0; j < M; ++j) {
                    int64_t idx = i * M + j;
                    dS[idx] = P[idx] * dS[idx] - P_norm[idx] * d_i;
                }
            }
#else
            for (int64_t i = 0; i < N; ++i) {
                T d_i = D_vec[i];
                for (int64_t j = 0; j < M; ++j) {
                    T dp = dot_product(dO_bh + i * D, v_bh + j * D, D);
                    dS[i * M + j] = P[i * M + j] * dp - P_norm[i * M + j] * d_i;
                }
            }
#endif

            if (causal) {
                for (int64_t i = 0; i < N; ++i) {
                    for (int64_t j = std::max<int64_t>(0, i + causal_offset + 1); j < M; ++j) {
                        dS[i * M + j] = T(0);
                    }
                }
            }

            // Step 7: dQ = dS @ K * scale  [N x D]
#ifdef TENZOR_USE_MKL
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<MKL_INT>(N), static_cast<MKL_INT>(D), static_cast<MKL_INT>(M),
                            scale,
                            dS.data(), static_cast<MKL_INT>(M),
                            k_bh, static_cast<MKL_INT>(D),
                            0.0f,
                            dq_bh, static_cast<MKL_INT>(D));
            } else {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<MKL_INT>(N), static_cast<MKL_INT>(D), static_cast<MKL_INT>(M),
                            scale,
                            dS.data(), static_cast<MKL_INT>(M),
                            k_bh, static_cast<MKL_INT>(D),
                            0.0,
                            dq_bh, static_cast<MKL_INT>(D));
            }
#else
            for (int64_t i = 0; i < N; ++i) {
                for (int64_t j = 0; j < M; ++j) {
                    T ds_val = dS[i * M + j] * scale;
                    if (ds_val != T(0)) {
                        fma_vector(dq_bh + i * D, ds_val, k_bh + j * D, D);
                    }
                }
            }
#endif

            // Step 8: dK = dS^T @ Q * scale  [M x D]
#ifdef TENZOR_USE_MKL
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            static_cast<MKL_INT>(M), static_cast<MKL_INT>(D), static_cast<MKL_INT>(N),
                            scale,
                            dS.data(), static_cast<MKL_INT>(M),
                            q_bh, static_cast<MKL_INT>(D),
                            0.0f,
                            dk_bh, static_cast<MKL_INT>(D));
            } else {
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            static_cast<MKL_INT>(M), static_cast<MKL_INT>(D), static_cast<MKL_INT>(N),
                            scale,
                            dS.data(), static_cast<MKL_INT>(M),
                            q_bh, static_cast<MKL_INT>(D),
                            0.0,
                            dk_bh, static_cast<MKL_INT>(D));
            }
#else
            for (int64_t i = 0; i < N; ++i) {
                for (int64_t j = 0; j < M; ++j) {
                    T ds_val = dS[i * M + j] * scale;
                    if (ds_val != T(0)) {
                        fma_vector(dk_bh + j * D, ds_val, q_bh + i * D, D);
                    }
                }
            }
#endif
        }
    }

    return std::vector<Tensor>{dQ, dK, dV};
}

// E.4: native half-precision (Float16 / BFloat16) FlashAttention backward.
// Storage stays Float16 throughout; arithmetic uses FP32 accumulators at
// the per-element level (load Float16 → cast to float → math → cast back
// to Float16 on store). The GEMMs are hand-rolled triple loops — MKL's
// cblas_hgemm exists only on Sapphire Rapids AVX-512-FP16 hardware, so
// the truly portable native FP16 path is this explicit micro-kernel.
//
// Intermediate buffers (S, P, dS, D_vec) stay FP32 because FP16 lacks
// the dynamic range for softmax exp() and accumulated row sums — that's
// not widening, it's standard mixed-precision accumulation (the same
// pattern NVIDIA's tensor cores use on GPU).
template <typename HalfT>
static auto flash_attention_backward_half(
    const Tensor& dO, const Tensor& Q, const Tensor& K,
    const Tensor& V, const Tensor& O,
    float scale, bool causal,
    float dropout_p,
    uint64_t philox_seed) -> std::vector<Tensor>
{
    auto q_shape = Q.shape();
    int64_t batch     = q_shape[0];
    int64_t num_heads = q_shape[1];
    int64_t N = q_shape[2];
    int64_t D = q_shape[3];
    auto k_shape = K.shape();
    int64_t M = k_shape[2];
    const int64_t causal_offset = M - N;  // F025: bottom-right causal alignment

    const bool apply_dropout = dropout_p > 0.0f && philox_seed != 0;
    const float dropout_scale = apply_dropout ? 1.0f / (1.0f - dropout_p) : 1.0f;
    const uint32_t rng_seed = static_cast<uint32_t>(philox_seed);

    std::vector<int64_t> q_shape_vec(q_shape.begin(), q_shape.end());
    std::vector<int64_t> k_shape_vec(k_shape.begin(), k_shape.end());
    Tensor dQ = zeros(q_shape_vec, Q.dtype(), Q.device());
    Tensor dK = zeros(k_shape_vec, K.dtype(), K.device());
    Tensor dV = zeros(k_shape_vec, V.dtype(), V.device());

    const HalfT* dO_data = dO.data<HalfT>();
    const HalfT* q_data  = Q.data<HalfT>();
    const HalfT* k_data  = K.data<HalfT>();
    const HalfT* v_data  = V.data<HalfT>();
    const HalfT* o_data  = O.data<HalfT>();
    HalfT* dq_data = dQ.data<HalfT>();
    HalfT* dk_data = dK.data<HalfT>();
    HalfT* dv_data = dV.data<HalfT>();

    auto h2f = [](HalfT v) -> float { return static_cast<float>(v); };
    auto f2h = [](float v) -> HalfT { return HalfT(v); };

    #pragma omp parallel for collapse(2) if(batch * num_heads > 1)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            int64_t q_offset  = ((b * num_heads + h) * N) * D;
            int64_t kv_offset = ((b * num_heads + h) * M) * D;
            const HalfT* q_bh  = q_data  + q_offset;
            const HalfT* k_bh  = k_data  + kv_offset;
            const HalfT* v_bh  = v_data  + kv_offset;
            const HalfT* dO_bh = dO_data + q_offset;
            const HalfT* o_bh  = o_data  + q_offset;
            HalfT* dq_bh = dq_data + q_offset;
            HalfT* dk_bh = dk_data + kv_offset;
            HalfT* dv_bh = dv_data + kv_offset;

            std::vector<float> S(static_cast<size_t>(N * M));
            std::vector<float> P(static_cast<size_t>(N * M));
            std::vector<float> D_vec(static_cast<size_t>(N));

            // Step 1: S = Q @ K^T * scale  (FP16 load, FP32 accumulate)
            for (int64_t i = 0; i < N; ++i) {
                for (int64_t j = 0; j < M; ++j) {
                    float acc = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        acc += h2f(q_bh[i * D + d]) * h2f(k_bh[j * D + d]);
                    }
                    S[i * M + j] = scale * acc;
                }
            }

            // Step 2: causal mask
            if (causal) {
                for (int64_t i = 0; i < N; ++i) {
                    for (int64_t j = std::max<int64_t>(0, i + causal_offset + 1); j < M; ++j) {
                        S[i * M + j] = -std::numeric_limits<float>::infinity();
                    }
                }
            }

            // Step 3: P = softmax(S, dim=-1) row-wise (FP32 throughout)
            for (int64_t i = 0; i < N; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t j = 0; j < M; ++j) {
                    max_val = std::max(max_val, S[i * M + j]);
                }
                float sum_exp = 0.0f;
                for (int64_t j = 0; j < M; ++j) {
                    float val = std::exp(S[i * M + j] - max_val);
                    P[i * M + j] = val;
                    sum_exp += val;
                }
                float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
                for (int64_t j = 0; j < M; ++j) {
                    P[i * M + j] *= inv_sum;
                }
            }

            // F051: snapshot the undropped softmax before Step 3b mutates P into
            // P̃; the dS normalization term below must use the true softmax.
            std::vector<float> P_true;
            if (apply_dropout) {
                P_true.assign(P.begin(), P.end());
            }

            // Step 3b: dropout replay — bh-combined counter
            // (bh=b*num_heads+h, i, j, 0) matching the forward kernel and
            // philox_dropout_mask.
            if (apply_dropout) {
                for (int64_t i = 0; i < N; ++i) {
                    for (int64_t j = 0; j < M; ++j) {
                        Philox4x32 philox;
                        philox.counter[0] = static_cast<uint32_t>(b * num_heads + h);
                        philox.counter[1] = static_cast<uint32_t>(i);
                        philox.counter[2] = static_cast<uint32_t>(j);
                        philox.counter[3] = 0u;
                        philox.key[0] = rng_seed;
                        philox.key[1] = rng_seed ^ 0x1BD11BDAU;
                        uint32_t rng_out[4];
                        philox.generate(rng_out);
                        float rand_val = Philox4x32::uint32_to_uniform(rng_out[0]);
                        float& p_ref = P[i * M + j];
                        if (rand_val < dropout_p) p_ref = 0.0f;
                        else                      p_ref *= dropout_scale;
                    }
                }
            }

            // Step 4: D_i = rowsum(dO * O)   (FP16 load, FP32 accumulate)
            for (int64_t i = 0; i < N; ++i) {
                float acc = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    acc += h2f(dO_bh[i * D + d]) * h2f(o_bh[i * D + d]);
                }
                D_vec[i] = acc;
            }

            // Step 5: dV = P^T @ dO   (FP16 store, FP32 inner)
            for (int64_t j = 0; j < M; ++j) {
                for (int64_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int64_t i = 0; i < N; ++i) {
                        acc += P[i * M + j] * h2f(dO_bh[i * D + d]);
                    }
                    dv_bh[j * D + d] = f2h(acc);
                }
            }

            // Step 6: dS = P̃ * (dO @ V^T) - P * D_i   (reuse S as dS). First
            // factor is the dropped/scaled P; normalization uses the undropped
            // softmax P_norm (== P without dropout).
            std::vector<float>& dS = S;
            const float* P_norm = apply_dropout ? P_true.data() : P.data();
            for (int64_t i = 0; i < N; ++i) {
                float d_i = D_vec[i];
                for (int64_t j = 0; j < M; ++j) {
                    float dp = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        dp += h2f(dO_bh[i * D + d]) * h2f(v_bh[j * D + d]);
                    }
                    dS[i * M + j] = P[i * M + j] * dp - P_norm[i * M + j] * d_i;
                }
            }
            if (causal) {
                for (int64_t i = 0; i < N; ++i) {
                    for (int64_t j = std::max<int64_t>(0, i + causal_offset + 1); j < M; ++j) {
                        dS[i * M + j] = 0.0f;
                    }
                }
            }

            // Step 7: dQ = dS @ K * scale
            for (int64_t i = 0; i < N; ++i) {
                for (int64_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j < M; ++j) {
                        acc += dS[i * M + j] * h2f(k_bh[j * D + d]);
                    }
                    dq_bh[i * D + d] = f2h(scale * acc);
                }
            }

            // Step 8: dK = dS^T @ Q * scale
            for (int64_t j = 0; j < M; ++j) {
                for (int64_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int64_t i = 0; i < N; ++i) {
                        acc += dS[i * M + j] * h2f(q_bh[i * D + d]);
                    }
                    dk_bh[j * D + d] = f2h(scale * acc);
                }
            }
        }
    }

    return std::vector<Tensor>{dQ, dK, dV};
}

auto flash_attention_backward(const Tensor& dO, const Tensor& Q, const Tensor& K,
                               const Tensor& V, const Tensor& O,
                               float scale, bool causal,
                               float dropout_p,
                               uint64_t philox_seed,
                               uint64_t philox_offset) -> std::vector<Tensor> {
    // Contiguity guard (mirrors flash_attention_forward ~:298-304). The typed
    // and half backward kernels below use raw pointer arithmetic with
    // contiguous [B,H,L,D] offsets (q_offset = ((b*H + h)*N)*D). Callers
    // commonly pass dO/Q/K/V/O as `permute({0,2,1,3})` strided views of a
    // [B,Sq,H,head_dim] projection, so a non-contiguous input silently
    // produces wrong dQ/dK/dV on the standard attention layout. Materialise
    // contiguous copies up front, exactly as the forward does.
    if (!dO.is_contiguous() || !Q.is_contiguous() || !K.is_contiguous() ||
        !V.is_contiguous() || !O.is_contiguous()) {
        return flash_attention_backward(
            dO.is_contiguous() ? dO : dO.contiguous(),
            Q.is_contiguous()  ? Q  : Q.contiguous(),
            K.is_contiguous()  ? K  : K.contiguous(),
            V.is_contiguous()  ? V  : V.contiguous(),
            O.is_contiguous()  ? O  : O.contiguous(),
            scale, causal, dropout_p, philox_seed, philox_offset);
    }
    // E.4: dtype dispatch.
    // - Float32 → native float kernel.
    // - Float64 → native double kernel (cblas_dgemm + double SIMD helpers).
    //   No widen-narrow; full mantissa preserved.
    // - Float16 / BFloat16 → widen to Float32 (these dtypes have less
    //   mantissa than FP32, so this is mathematically lossless and matches
    //   PyTorch / FlashAttention-2 reference behaviour).
    if (Q.dtype() == DType::Float32) {
        return flash_attention_backward_typed<float>(
            dO, Q, K, V, O, scale, causal, dropout_p, philox_seed);
    }
    if (Q.dtype() == DType::Float64) {
        return flash_attention_backward_typed<double>(
            dO, Q, K, V, O,
            static_cast<double>(scale), causal,
            static_cast<double>(dropout_p), philox_seed);
    }
    if (Q.dtype() == DType::Float16) {
        // E.4: native half-precision path. Float16 storage end-to-end;
        // FP32 used only for per-element accumulators (the GPU tensor-core
        // pattern). No tensor-level widen-narrow.
        return flash_attention_backward_half<tenzor::Float16>(
            dO, Q, K, V, O, scale, causal, dropout_p, philox_seed);
    }
    if (Q.dtype() == DType::BFloat16) {
        return flash_attention_backward_half<tenzor::BFloat16>(
            dO, Q, K, V, O, scale, causal, dropout_p, philox_seed);
    }
    throw std::runtime_error("Flash attention backward: unsupported dtype " +
                             std::string(dtype_name(Q.dtype())));
}

} // namespace tenzor::cpu
