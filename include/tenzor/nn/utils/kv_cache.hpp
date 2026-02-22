/**
 * @file kv_cache.hpp
 * @brief KV Cache utilities for efficient autoregressive inference
 *
 * Provides pre-allocated per-layer key/value caches for transformer models.
 * The cache uses pre-allocated memory and returns sliced views (zero-copy)
 * for maximum performance during autoregressive generation.
 *
 * @code
 * // Create cache for 12-layer model
 * KVCacheConfig config{
 *     .num_layers = 12,
 *     .max_seq_len = 2048,
 *     .num_kv_heads = 8,
 *     .head_dim = 64,
 *     .batch_size = 1,
 *     .dtype = DType::Float32,
 *     .device = Device::cpu()
 * };
 * KVCache cache(config);
 *
 * // During generation (layer 0, position 0)
 * auto [cached_k, cached_v] = cache.update(0, new_keys, new_values, 0);
 * // cached_k/cached_v are views into the pre-allocated buffer
 * @endcode
 */

#pragma once

#include <vector>
#include <cstdint>
#include "../../core/tensor.hpp"
#include "../../core/dtype.hpp"
#include "../../core/device.hpp"

namespace tenzor {
namespace nn {
namespace utils {

/**
 * @brief Configuration for KV cache allocation.
 */
struct KVCacheConfig {
    int64_t num_layers;      ///< Number of transformer layers
    int64_t max_seq_len;     ///< Maximum sequence length the cache can hold
    int64_t num_kv_heads;    ///< Number of key/value heads
    int64_t head_dim;        ///< Dimension per attention head
    int64_t batch_size{1};   ///< Batch size
    DType dtype{DType::Float32};  ///< Data type for cache tensors
    Device device{Device::cpu()}; ///< Device for cache tensors
};

/**
 * @brief Pre-allocated per-layer KV cache for autoregressive inference.
 *
 * Manages pre-allocated key and value tensors for each transformer layer.
 * During generation, new key/value pairs are written into the cache at
 * the current position, and sliced views are returned for the attention
 * computation. This avoids repeated concatenation and allocation.
 *
 * Cache layout per layer:
 * - k_cache: (batch_size, num_kv_heads, max_seq_len, head_dim)
 * - v_cache: (batch_size, num_kv_heads, max_seq_len, head_dim)
 *
 * @par Thread Safety
 * Not thread-safe. Use separate cache instances for parallel sequences.
 *
 * @see GroupedQueryAttention for attention layer using this cache
 */
class KVCache {
public:
    /**
     * @brief Construct pre-allocated KV cache.
     *
     * Allocates all cache memory upfront for the specified configuration.
     *
     * @param config Cache configuration
     * @throws std::invalid_argument if any config parameter is non-positive
     */
    explicit KVCache(const KVCacheConfig& config);

    /**
     * @brief Update cache for a layer and return sliced views.
     *
     * Writes new_k and new_v into the cache at the given position,
     * then returns views covering positions [0, pos + new_seq_len).
     *
     * @param layer Layer index (0-based)
     * @param new_k New key tensor of shape (batch, num_kv_heads, new_seq_len, head_dim)
     * @param new_v New value tensor of shape (batch, num_kv_heads, new_seq_len, head_dim)
     * @param pos Starting position in the sequence (where to write new tokens)
     *
     * @return Pair of (cached_keys, cached_values), each of shape
     *         (batch, num_kv_heads, pos + new_seq_len, head_dim).
     *         These are sliced views into the pre-allocated cache (zero-copy).
     *
     * @throws std::out_of_range if layer >= num_layers
     * @throws std::runtime_error if pos + new_seq_len > max_seq_len
     */
    auto update(int64_t layer, const Tensor& new_k, const Tensor& new_v, int64_t pos)
        -> std::pair<Tensor, Tensor>;

    /**
     * @brief Get cached keys for a layer up to the given length.
     *
     * @param layer Layer index
     * @param seq_len Number of cached positions to return
     * @return Sliced view of shape (batch, num_kv_heads, seq_len, head_dim)
     */
    auto get_keys(int64_t layer, int64_t seq_len) const -> Tensor;

    /**
     * @brief Get cached values for a layer up to the given length.
     *
     * @param layer Layer index
     * @param seq_len Number of cached positions to return
     * @return Sliced view of shape (batch, num_kv_heads, seq_len, head_dim)
     */
    auto get_values(int64_t layer, int64_t seq_len) const -> Tensor;

    /**
     * @brief Reset all caches to zero.
     *
     * Clears the cache contents without deallocating memory.
     */
    auto reset() -> void;

    /**
     * @brief Reset cache for a specific layer.
     *
     * @param layer Layer index
     */
    auto reset_layer(int64_t layer) -> void;

    /** @brief Get the maximum sequence length. */
    auto max_seq_len() const -> int64_t { return config_.max_seq_len; }

    /** @brief Get the number of layers. */
    auto num_layers() const -> int64_t { return config_.num_layers; }

    /** @brief Get the cache configuration. */
    auto config() const -> const KVCacheConfig& { return config_; }

private:
    KVCacheConfig config_;
    std::vector<Tensor> k_caches_;  ///< Per-layer key caches
    std::vector<Tensor> v_caches_;  ///< Per-layer value caches
};

} // namespace utils
} // namespace nn
} // namespace tenzor
