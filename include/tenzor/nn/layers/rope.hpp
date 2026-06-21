/**
 * @file rope.hpp
 * @brief Rotary Position Embedding (RoPE)
 *
 * Implements rotary position embeddings as described in
 * "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021).
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Rotary Position Embedding (RoPE).
 *
 * Encodes position information by rotating pairs of dimensions in the
 * query and key vectors using sinusoidal frequencies. The rotation ensures
 * that the dot product between query and key depends on their relative
 * position, providing translation-invariant attention.
 *
 * This implementation uses the split-half (GPT-NeoX / Llama) pairing, NOT the
 * interleaved (RoFormer) (2i, 2i+1) layout. The head dimension is split into
 * two halves x1 = x[..., :dim/2] and x2 = x[..., dim/2:], and for each
 * i in [0, dim/2) the pair (x1[i], x2[i]) is rotated:
 *   x_rot1[i] = x1[i] * cos(θ_i * pos) - x2[i] * sin(θ_i * pos)
 *   x_rot2[i] = x2[i] * cos(θ_i * pos) + x1[i] * sin(θ_i * pos)
 *
 * Where θ_i = 1 / (base^(2i/dim))
 *
 * Q and K use the same scheme, so attention scores are self-consistent.
 * Checkpoints trained with the interleaved (2i, 2i+1) layout are NOT directly
 * compatible without permuting the head dimension.
 *
 * Shape:
 * - Input: (..., seq_len, head_dim) where head_dim is even
 * - Output: Same as input
 *
 * @code
 * RoPE rope(64, 2048);  // head_dim=64, max_seq_len=2048
 *
 * auto q_rot = rope.forward(q, position_offset);
 * auto k_rot = rope.forward(k, position_offset);
 * @endcode
 */
class RoPE : public Module {
public:
    /**
     * @brief Construct RoPE.
     *
     * @param dim Head dimension (must be even)
     * @param max_seq_len Maximum sequence length for precomputed frequencies
     * @param base Base for frequency computation (default: 10000.0)
     */
    RoPE(int64_t dim, int64_t max_seq_len = 2048, double base = 10000.0);

    /**
     * @brief Apply rotary position embedding.
     *
     * @param input Input variable of shape (..., seq_len, head_dim)
     * @return Rotated output (same shape)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Apply RoPE with position offset.
     *
     * Useful for autoregressive generation where tokens are processed
     * one at a time with increasing position.
     *
     * @param input Input variable
     * @param offset Starting position index
     * @return Rotated output
     */
    auto forward(const Variable& input, int64_t offset) -> Variable;

private:
    int64_t dim_;
    int64_t max_seq_len_;
    double base_;

    // Precomputed cos/sin tables: shape (max_seq_len, dim/2)
    Tensor cos_cached_;
    Tensor sin_cached_;

    auto precompute_freqs() -> void;
};

} // namespace nn
} // namespace tenzor
