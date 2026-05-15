/**
 * @file philox_dropout.hpp
 * @brief Deterministic Philox4x32-10-based dropout mask generator.
 *
 * Audit F13/F22-followup: provides a Tensor-level Philox-deterministic
 * Bernoulli mask so the FlashAttention composed-ops dropout fallback can
 * (a) return real `seed`/`offset` Tensors that the contract specifies, and
 * (b) be replayed bit-exactly by the corresponding backward path.
 *
 * Philox4x32-10 is the same algorithm used by:
 *   - PyTorch's `torch.cuda.dropout` (and CUDA `flash_attn` references)
 *   - tenzor's CUDA/ROCm FlashAttention kernels
 *   - tenzor's host Philox replay in `src/autograd/function_attention.cpp`
 *
 * Counter convention used here MATCHES the CUDA/ROCm FA kernels:
 *
 *     ctr = (batch_head, query_idx, kv_pos, 0)
 *     key = (seed_low_32, seed_low_32 ^ 0x1BD11BDA)
 *
 * So a (seed, offset) pair saved by an OneAPI / Vulkan forward dropout
 * can be replayed by either an OneAPI / Vulkan backward composed-ops path,
 * by the autograd-level host helper in `function_attention.cpp`, or by
 * any of the existing GPU FA backward kernels.
 *
 * The mask returned is shape-equivalent to the attention matrix that
 * dropout is applied to:
 *
 *     shape = [batch_heads, seq_q, seq_kv]  (3-D), or
 *     shape = [batch, heads, seq_q, seq_kv]  (4-D — flattened internally to 3-D)
 *
 * For non-attention dropout shapes the helper falls back to the standard
 * Philox counter `(linear_index, 0, 0, 0)`.
 *
 * Convention for dropout mask semantics:
 *
 *     uniform = philox(seed, ctr) ∈ [0, 1)
 *     keep    = (uniform >= dropout_p)
 *     mask    = keep ? (1 / (1 - dropout_p)) : 0    // pre-scaled
 *
 * The returned mask is therefore ready to multiply into `attn` directly —
 * no separate scale step needed. Use `dropout_p == 0` to get an all-keep
 * mask (no-op contract).
 */

#pragma once

#include "tenzor/core/tensor.hpp"

#include <cstdint>
#include <vector>

namespace tenzor {

/**
 * @brief Build a deterministic Philox-keyed Bernoulli dropout mask on CPU.
 *
 * @param shape   Output shape. Must be 3-D `[BH, Sq, Sk]` or 4-D
 *                `[B, H, Sq, Sk]` for attention-style dropout; any other
 *                rank falls back to flat per-element Philox addressing.
 * @param p       Dropout probability ∈ [0, 1). `p == 0` returns an all-ones
 *                tensor scaled by `1` (no-op mask).
 * @param seed    Philox seed (low 32 bits used; PyTorch convention).
 * @param offset  Philox counter offset (added to the leading counter dim).
 * @param dtype   Output dtype — typically `Float32`. Float64/Float16/BF16
 *                also supported.
 *
 * @return CPU tensor of shape `shape` with the pre-scaled Bernoulli mask.
 *         Caller moves to target device as needed (`.to(device)`).
 */
auto philox_dropout_mask(const std::vector<int64_t>& shape,
                          double p,
                          uint64_t seed,
                          uint64_t offset,
                          DType dtype = DType::Float32)
    -> Tensor;

/**
 * @brief Generate a fresh (seed, offset) pair for a Philox dropout stream.
 *
 * Returns two int64 1-element CPU tensors. The seed comes from
 * `std::random_device` (so distinct training steps get distinct masks);
 * the offset is 0 (the FA convention is to use per-element counters that
 * already disambiguate within a step).
 */
struct PhiloxStream { Tensor seed; Tensor offset; };
auto new_philox_stream() -> PhiloxStream;

}  // namespace tenzor
