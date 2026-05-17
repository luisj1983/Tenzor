/// \file attention_contract.hpp
/// \brief C++ companion to `docs/internals/attention-contract.md`.
///
/// Every backend's FlashAttention / FusedAttention / FlexAttention /
/// NestedAttention kernel must conform to the cross-backend contract
/// codified in `docs/internals/attention-contract.md`. This header is the
/// source of truth for the *machine-readable* parts of that contract:
/// dtypes, sentinels, output-tuple sizes, ScoreModId values, and tile
/// recommendations. Backends should `#include` this header rather than
/// re-deriving any constant locally.
///
/// If you find yourself writing `-1e9f` or `-65504.0f` for a causal mask
/// sentinel in a backend kernel: stop. Use the constants below.

#pragma once

#include "tenzor/core/dtype.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tenzor::ops::attention {

// =====================================================================
// Output tuple sizes
// =====================================================================
//
// FlashAttention forward returns FOUR tensors: `(output, lse,
// philox_seed, philox_offset)`. The Philox outputs are empty Tensor{}
// when DropoutP == 0; LSE is always present and always Float32.
//
// FusedAttention forward returns TWO: `(output, lse)`.
//
// FlexAttention forward returns TWO: `(output, lse)`.
//
// NestedAttention forward returns TWO: `(output, lse)`.

inline constexpr std::size_t kFlashAttentionForwardOutputs   = 4;
inline constexpr std::size_t kFlashAttentionBackwardOutputs  = 3;  // (dQ, dK, dV)
inline constexpr std::size_t kFusedAttentionForwardOutputs   = 2;
inline constexpr std::size_t kFlexAttentionForwardOutputs    = 2;
inline constexpr std::size_t kFlexAttentionBackwardOutputs   = 3;
inline constexpr std::size_t kNestedAttentionForwardOutputs  = 2;

// =====================================================================
// Sentinels
// =====================================================================

/// Causal-mask sentinel: the value added to masked-out attention scores.
/// MUST be `-INFINITY`, never `-1e9f` or `-1e30f` — those saturate to
/// `-65504` in FP16 and leak gradient mass through softmax.
template <typename T>
[[nodiscard]] inline constexpr auto causal_mask_sentinel() noexcept -> T {
    static_assert(std::numeric_limits<T>::has_infinity,
                  "Causal mask sentinel requires IEEE-754 floating-point infinity.");
    return -std::numeric_limits<T>::infinity();
}

/// LSE sentinel for fully-masked rows. Backward computes `P = exp(S - L)`
/// which evaluates to 0 for `L = -inf`, correctly producing zero grads.
inline constexpr auto kLseSentinelF32 = -std::numeric_limits<float>::infinity();

// =====================================================================
// Required dtype for LSE
// =====================================================================
//
// LSE stores `row_max + log(row_sum_exp(scores - row_max))`. Its dynamic
// range exceeds FP16/BF16 even when Q/K/V are FP16. Therefore LSE is
// always Float32 regardless of input dtype, per the contract.

inline constexpr DType kLseDType = DType::Float32;

/// Same applies to FusedLayerNorm / FusedRMSNorm saved stats.
inline constexpr DType kLayerNormStatsDType = DType::Float32;

// =====================================================================
// Philox state dtypes
// =====================================================================

inline constexpr DType kPhiloxSeedDType   = DType::Int64;
inline constexpr DType kPhiloxOffsetDType = DType::Int64;

// =====================================================================
// ScoreModId registry — mirrors the table in attention-contract.md
// =====================================================================

namespace score_mod_id {
inline constexpr int64_t Identity       = 0;  // FusedAttention-equivalent
inline constexpr int64_t Causal         = 1;  // upper-triangular mask
inline constexpr int64_t SlidingWindow  = 2;  // |i-j| > WindowSize/2

// Built-ins added in 2.21 (CPU full set; GPU backends still defer to the
// host-side reference at src/nn/layers/flex_attention.cpp for now):
inline constexpr int64_t RelPosBias        = 3;  // additive bias tensor, last input
inline constexpr int64_t Alibi             = 4;  // ALiBi: -slope_h * (q_idx - k_idx)
inline constexpr int64_t PrefixLM          = 5;  // bidi for first PrefixLength, causal after
inline constexpr int64_t SlidingWindowSym  = 6;  // alias of SlidingWindow for symmetry
inline constexpr int64_t UserLambda        = 7;  // arbitrary user functor via find_registered_score_mod
}  // namespace score_mod_id

// =====================================================================
// FlashAttention tile/work-group recommendations
// =====================================================================
//
// Backends choose tile sizes within these recommendations. The defaults
// match the CPU reference at `src/backends/cpu/kernels/flash_attention.cpp`
// and have been validated on H100, MI300, Intel Arc, and Apple M-series
// gradients. Backends may override per architecture; this is a starting
// point, not a mandate.

struct TileShape {
    int q_tile;   // queries per work-group  / threadblock
    int k_tile;   // keys    per inner loop iteration
};

[[nodiscard]] inline constexpr auto recommended_tile(int head_dim) noexcept
        -> TileShape {
    if (head_dim <= 32)  return TileShape{ 128, 128 };
    if (head_dim <= 64)  return TileShape{ 128,  64 };
    if (head_dim <= 128) return TileShape{ 64,   64 };
    // head_dim > 128 — typical for vision transformers / wide MLPs
    return TileShape{ 32, 32 };
}

// =====================================================================
// LSE update formula — every backend must agree on this exact expression
// =====================================================================
//
// Streaming softmax (Online softmax / FlashAttention's row-update rule):
//   m_new       = max(m_old, max_over_k(scores))
//   row_sum_new = row_sum_old * exp(m_old - m_new)
//                 + sum_over_k(exp(scores - m_new))
//   o_new       = o_old       * exp(m_old - m_new) * row_sum_old / row_sum_new
//                 + sum_over_k(exp(scores - m_new) * V) / row_sum_new
//
// Final per-row LSE = m + log(row_sum). For zero-sum rows (e.g. fully
// causal-masked first row), emit -INFINITY.
//
// CPU reference is at src/backends/cpu/kernels/flash_attention.cpp:518-528.

}  // namespace tenzor::ops::attention
