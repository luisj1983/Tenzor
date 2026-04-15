/**
 * @file flex_attention.hpp
 * @brief FlexAttention: block-sparse attention with custom score modification
 *
 * Implements FlexAttention, a generalization of Flash Attention that supports
 * arbitrary attention patterns via BlockMask and element-wise score modifications
 * via ScoreModFn. Masked-out blocks are never computed, giving O(active_blocks)
 * rather than O(S^2) work for sparse patterns.
 *
 * Based on the FlexAttention approach from PyTorch 2.5+.
 */

#pragma once

#include <cstdint>
#include <functional>
#include "../../core/tensor.hpp"
#include "../../core/device.hpp"
#include "../../core/dtype.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Block-level mask for FlexAttention.
 *
 * BlockMask describes which blocks of the attention matrix to compute.
 * Each block covers a (block_size x block_size) region of the full (S, S)
 * attention matrix. Blocks marked as inactive are skipped entirely,
 * avoiding both computation and memory traffic.
 *
 * The mask is stored as a 2D boolean tensor of shape
 * (num_q_blocks, num_kv_blocks), where true indicates that the block
 * should be computed.
 *
 * @code
 * // Causal mask: lower-triangular block pattern
 * auto mask = BlockMask::causal(1024, 128);
 *
 * // Sliding window: only nearby blocks
 * auto mask = BlockMask::sliding_window(4096, 512, 128);
 *
 * // Custom: provide your own block-level mask
 * Tensor custom_mask({8, 8}, DType::Bool, Device::cpu());
 * // ... fill custom_mask ...
 * auto mask = BlockMask(custom_mask, 128);
 * @endcode
 */
class BlockMask {
public:
    BlockMask() = default;

    /**
     * @brief Create from a 2D boolean mask tensor.
     *
     * @param block_mask Bool tensor of shape (num_q_blocks, num_kv_blocks).
     *        True means the corresponding block is active (should be computed).
     * @param block_size Side length of each square block (default: 128).
     *
     * @throws std::invalid_argument if block_mask is not 2D
     * @throws std::invalid_argument if block_size <= 0
     */
    explicit BlockMask(Tensor block_mask, int64_t block_size = 128);

    /**
     * @brief Create a causal (lower-triangular) block mask.
     *
     * A block (q_blk, kv_blk) is active iff kv_blk <= q_blk, which is the
     * block-level analog of the standard causal mask.
     *
     * @param seq_len Sequence length
     * @param block_size Side length of each block
     * @return Causal BlockMask
     */
    static auto causal(int64_t seq_len, int64_t block_size = 128) -> BlockMask;

    /**
     * @brief Create a sliding-window block mask.
     *
     * A block (q_blk, kv_blk) is active iff the key block falls within
     * window_size positions of the query block.
     *
     * @param seq_len Sequence length
     * @param window_size Number of positions in the sliding window
     * @param block_size Side length of each block
     * @return Sliding-window BlockMask
     */
    static auto sliding_window(int64_t seq_len, int64_t window_size,
                               int64_t block_size = 128) -> BlockMask;

    /**
     * @brief Create a prefix-LM block mask.
     *
     * The first prefix_len positions attend to each other fully (bidirectional),
     * while remaining positions use causal masking.
     *
     * @param seq_len Sequence length
     * @param prefix_len Length of the bidirectional prefix
     * @param block_size Side length of each block
     * @return Prefix-LM BlockMask
     */
    static auto prefix_lm(int64_t seq_len, int64_t prefix_len,
                           int64_t block_size = 128) -> BlockMask;

    /**
     * @brief Create a fully dense block mask (all blocks active).
     *
     * @param seq_len Sequence length
     * @param block_size Side length of each block
     * @return Full BlockMask
     */
    static auto full(int64_t seq_len, int64_t block_size = 128) -> BlockMask;

    /// @brief Get the underlying bool mask tensor.
    auto mask() const -> const Tensor& { return mask_; }

    /// @brief Get the block size.
    auto block_size() const -> int64_t { return block_size_; }

    /// @brief Number of query blocks.
    auto num_q_blocks() const -> int64_t;

    /// @brief Number of key/value blocks.
    auto num_kv_blocks() const -> int64_t;

    /**
     * @brief Check whether a specific block should be computed.
     *
     * @param q_block Query block index
     * @param kv_block Key/value block index
     * @return true if the block is active
     */
    auto is_active(int64_t q_block, int64_t kv_block) const -> bool;

private:
    Tensor mask_;             ///< Bool tensor of shape (num_q_blocks, num_kv_blocks)
    int64_t block_size_ = 128;
};

// =============================================================================
// Score modification
// =============================================================================

/**
 * @brief Score modification function type.
 *
 * A ScoreModFn receives the raw attention scores for one (batch, head)
 * combination covering a single query-block x kv-block tile, and returns
 * modified scores of the same shape. The function also receives positional
 * context (batch index, head index, starting query position, starting kv
 * position) so it can apply position-dependent modifications.
 *
 * @param score Attention score tensor for this block tile (block_size x block_size)
 * @param b Batch index
 * @param h Head index
 * @param q_start Starting query position of this block
 * @param kv_start Starting key/value position of this block
 * @return Modified score tensor (same shape as input)
 */
using ScoreModFn = std::function<Tensor(const Tensor& score, int64_t b, int64_t h,
                                        int64_t q_start, int64_t kv_start)>;

/**
 * @brief Create a causal score modification.
 *
 * Within each active block, positions where kv_idx > q_idx receive -infinity,
 * enforcing exact causal masking at the element level (complementing the
 * block-level causal mask which operates at coarser granularity).
 *
 * @return ScoreModFn that applies causal masking
 */
auto causal_score_mod() -> ScoreModFn;

/**
 * @brief Create an ALiBi (Attention with Linear Biases) score modification.
 *
 * Adds a position-dependent linear bias: score[q][kv] += slope * (kv - q).
 * Slopes should be a 1D tensor of shape (num_heads,).
 *
 * @param slopes Per-head slope values, shape (num_heads,)
 * @return ScoreModFn that applies ALiBi biases
 *
 * @see "Train Short, Test Long" (Press et al., 2022)
 */
auto alibi_score_mod(const Tensor& slopes) -> ScoreModFn;

// =============================================================================
// FlexAttention function
// =============================================================================

/**
 * @brief Compute FlexAttention with block-sparse masking and score modification.
 *
 * FlexAttention generalizes scaled dot-product attention by:
 * 1. Only computing attention for blocks marked as active in the BlockMask,
 *    skipping masked-out regions entirely.
 * 2. Applying an optional per-element score modification function after
 *    computing raw attention logits but before softmax.
 *
 * Uses the online softmax algorithm (Milakov & Gimelshein, 2018) to compute
 * numerically stable softmax across all active KV blocks without materializing
 * the full attention matrix.
 *
 * Input shapes follow the (B, H, S, D) convention:
 * - B = batch size
 * - H = number of attention heads
 * - S = sequence length
 * - D = head dimension
 *
 * @param query  Query tensor of shape (B, H, S, D)
 * @param key    Key tensor of shape (B, H, S, D)
 * @param value  Value tensor of shape (B, H, S, D)
 * @param block_mask Block-level sparsity mask
 * @param score_mod Optional element-wise score modification (applied after
 *        Q*K^T*scale but before softmax). nullptr means no modification.
 * @param scale  Attention scale factor. Negative value triggers auto-scaling
 *        with 1/sqrt(D). Default: -1.0f (auto).
 * @return Output tensor of shape (B, H, S, D)
 *
 * @throws std::invalid_argument if Q/K/V shapes are incompatible
 * @throws std::invalid_argument if block_mask dimensions don't match sequence length
 *
 * @code
 * // Causal FlexAttention with ALiBi
 * auto mask = BlockMask::causal(seq_len);
 * Tensor slopes = compute_alibi_slopes(num_heads);
 * auto out = flex_attention(q, k, v, mask,
 *     causal_score_mod(),  // fine-grained causal within blocks
 *     -1.0f);
 *
 * // Sliding window attention
 * auto mask = BlockMask::sliding_window(seq_len, 512);
 * auto out = flex_attention(q, k, v, mask);
 * @endcode
 */
auto flex_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const BlockMask& block_mask,
    ScoreModFn score_mod = nullptr,
    float scale = -1.0f
) -> Tensor;

} // namespace nn
} // namespace tenzor
