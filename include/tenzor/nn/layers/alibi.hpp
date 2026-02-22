/**
 * @file alibi.hpp
 * @brief Attention with Linear Biases (ALiBi)
 *
 * Implements ALiBi positional encoding as described in
 * "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
 * (Press et al., ICLR 2022).
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Attention with Linear Biases (ALiBi).
 *
 * Adds a linear position-dependent bias to attention scores instead of
 * using positional embeddings. Each attention head gets a geometric slope m_i
 * and the bias for position pair (i,j) is -m * |i - j|.
 *
 * Benefits over sinusoidal/learned positional embeddings:
 * - No learned parameters for position encoding
 * - Better length extrapolation (train short, test long)
 * - Simpler implementation
 *
 * The slopes are set as geometric sequence: m_i = 2^(-8*i/num_heads)
 * for i = 1, ..., num_heads.
 *
 * @code
 * ALiBi alibi(8);  // 8 attention heads
 *
 * // Add to attention scores before softmax:
 * scores = scores + alibi.get_bias(seq_len_q, seq_len_k, device);
 * @endcode
 */
class ALiBi : public Module {
public:
    /**
     * @brief Construct ALiBi.
     *
     * @param num_heads Number of attention heads
     */
    explicit ALiBi(int64_t num_heads);

    /**
     * @brief Get ALiBi bias tensor.
     *
     * Returns a bias tensor of shape (1, num_heads, seq_q, seq_k) that can
     * be directly added to attention scores.
     *
     * @param seq_q Query sequence length
     * @param seq_k Key sequence length
     * @param device Target device
     * @param dtype Target dtype (default: Float32)
     * @return Bias tensor (1, num_heads, seq_q, seq_k)
     */
    auto get_bias(int64_t seq_q, int64_t seq_k,
                  Device device = Device::cpu(),
                  DType dtype = DType::Float32) -> Tensor;

    /**
     * @brief Forward pass — returns input unchanged.
     *
     * ALiBi doesn't modify the input directly; use get_bias() to obtain
     * the bias tensor and add it to attention scores.
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t num_heads_;
    std::vector<float> slopes_;  ///< Per-head slopes

    auto compute_slopes() -> void;
};

} // namespace nn
} // namespace tenzor
