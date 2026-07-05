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
#include <span>
#include <vector>
#include "../../core/tensor.hpp"
#include "../../core/device.hpp"
#include "../../core/dtype.hpp"
#include "../../backend/op_attributes.hpp"  // OpAttributes for shared score-mod fwd

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
     * A block (q_blk, kv_blk) is active iff kv_blk <= q_blk. In addition the
     * returned mask sets requires_element_causal(), so the attention kernel
     * enforces an exact position-level causal constraint (kv_pos <= q_pos)
     * inside the diagonal blocks. The result is therefore strictly causal for
     * any block_size, with no need to also pass causal_score_mod().
     *
     * @param seq_len Sequence length
     * @param block_size Side length of each block
     * @return Causal BlockMask (element-level causal within blocks)
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
     * while remaining positions use causal masking. The returned mask sets
     * requires_element_causal() so that the causal region is enforced exactly
     * at the element level (kv_pos <= q_pos) within diagonal blocks; keys in
     * the prefix region remain fully visible. (The element-level constraint is
     * applied only on diagonal blocks, which is precisely where causal leakage
     * could occur; off-diagonal active blocks are unaffected.)
     *
     * @param seq_len Sequence length
     * @param prefix_len Length of the bidirectional prefix
     * @param block_size Side length of each block
     * @return Prefix-LM BlockMask (element-level causal within blocks)
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

    /**
     * @brief Whether element-level causal masking must be applied within blocks.
     *
     * Block-level activation alone is too coarse: an active diagonal block
     * (kv_block == q_block) covers a (block_size x block_size) region in which
     * a query may attend to keys at strictly greater positions. For causal
     * masks (BlockMask::causal, the causal region of prefix_lm) this would leak
     * future tokens whenever block_size > 1. When this flag is set, the
     * attention kernel applies an exact position-level causal mask
     * (kv_pos <= q_pos) inside every active block, so the result is correctly
     * causal regardless of block_size without requiring an extra
     * causal_score_mod().
     *
     * @return true if the kernel must enforce kv_pos <= q_pos within blocks.
     */
    auto requires_element_causal() const -> bool { return requires_element_causal_; }

    /**
     * @brief Prefix length for element-level causal masking.
     *
     * Only meaningful when requires_element_causal() is true. Within active
     * blocks the kernel keeps a score iff (kv_pos < causal_prefix_len() ||
     * kv_pos <= q_pos). For a pure causal mask this is 0, reducing to the
     * standard kv_pos <= q_pos constraint. For a prefix-LM mask it equals the
     * bidirectional prefix length so prefix keys remain fully visible.
     *
     * @return The bidirectional prefix length (0 for pure causal).
     */
    auto causal_prefix_len() const -> int64_t { return causal_prefix_len_; }

private:
    Tensor mask_;             ///< Bool tensor of shape (num_q_blocks, num_kv_blocks)
    int64_t block_size_ = 128;
    bool requires_element_causal_ = false;  ///< Enforce kv_pos <= q_pos within active blocks.
    int64_t causal_prefix_len_ = 0;         ///< Prefix keys (pos < this) stay visible under causal masking.
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
 * @brief Register a user-defined score-mod functor for FlexAttention's
 * ScoreModId path (audit J12).
 *
 * The backend dispatch uses `AttrKey::ScoreModId` to select a score
 * modification. IDs 0-7 are fixed built-ins on EVERY backend (0=identity,
 * 1=causal, 2=sliding_window, 3=relpos_bias, 4=alibi, 5=prefix_lm,
 * 6=sliding_window_sym, 7=user_lambda). IDs >= 8 are reserved for
 * user-registered functors. Register a functor here so the backend's
 * `OpId::FlexAttention` lambda can locate it by ID.
 *
 * Functor signature: takes a 2D scores block (shape (S_q, S_kv)) and the
 * (b, h, q_start, kv_start) indices, returns the modified block. The dispatch
 * path invokes the functor once per (batch, head) slice so batch/head/position
 * indexing is correct.
 *
 * @param id  Score-mod ID to register under (≥ 8).
 * @param fn  Score-mod functor.
 *
 * @throws std::invalid_argument if `id < 8`.
 */
auto register_score_mod(int64_t id, ScoreModFn fn) -> void;

/**
 * @brief Look up a previously registered score-mod functor by ID
 * (audit J12). Returns nullptr if no such functor is registered.
 */
auto find_registered_score_mod(int64_t id) -> ScoreModFn;

/**
 * @brief Shared, device-agnostic forward for FlexAttention built-in score
 * modifications (ScoreModIds 2-7) and user-registered functors (IDs >= 8).
 *
 * Composed entirely from `tenzor::` tensor ops (Q@K^T → mask/bias → softmax →
 * @V), which dispatch to whatever backend Q/K/V live on, so the CPU reference
 * and every GPU backend produce identical results. Callers handle the fused
 * fast path for ScoreModId 0 (identity) / 1 (causal) and delegate all other
 * IDs here. `inputs` is [Q, K, V, (optional block_mask / bias)]; `attrs`
 * carries Scale, ScoreModId, WindowSize, PrefixLength.
 *
 * @return {output, lse} where lse = logsumexp(scores, dim=-1).
 */
auto flex_attention_score_mod_forward(std::span<const Tensor> inputs,
                                      const OpAttributes& attrs)
    -> std::vector<Tensor>;

/**
 * @brief Shared, device-agnostic backward for FlexAttention built-in score
 * modifications (ScoreModIds 2-7) and user-registered functors (IDs >= 8).
 *
 * Replays Q@K^T*scale, reapplies the same score modification as the forward
 * (so forward/backward stay mutually consistent, incl. canonical ALiBi slopes),
 * then runs the softmax-attention chain rule. Callers handle ScoreModId 0/1 via
 * the fused FlashAttention backward and delegate all other IDs here. `inputs`
 * is [dO, Q, K, V, O, (LSE optional), (relpos bias for id 3 as last input)].
 *
 * @return {dQ, dK, dV}.
 */
auto flex_attention_score_mod_backward(std::span<const Tensor> inputs,
                                       const OpAttributes& attrs)
    -> std::vector<Tensor>;

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
