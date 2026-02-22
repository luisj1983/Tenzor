/**
 * @file kv_cache.cpp
 * @brief Implementation of KV cache utilities for autoregressive inference
 */

#include "tenzor/nn/utils/kv_cache.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include <stdexcept>
#include <cstring>

namespace tenzor::nn::utils {

KVCache::KVCache(const KVCacheConfig& config)
    : config_(config) {

    if (config_.num_layers <= 0) {
        throw std::invalid_argument("KVCache: num_layers must be positive");
    }
    if (config_.max_seq_len <= 0) {
        throw std::invalid_argument("KVCache: max_seq_len must be positive");
    }
    if (config_.num_kv_heads <= 0) {
        throw std::invalid_argument("KVCache: num_kv_heads must be positive");
    }
    if (config_.head_dim <= 0) {
        throw std::invalid_argument("KVCache: head_dim must be positive");
    }
    if (config_.batch_size <= 0) {
        throw std::invalid_argument("KVCache: batch_size must be positive");
    }

    // Pre-allocate caches for all layers
    std::vector<int64_t> cache_shape = {
        config_.batch_size,
        config_.num_kv_heads,
        config_.max_seq_len,
        config_.head_dim
    };

    k_caches_.reserve(config_.num_layers);
    v_caches_.reserve(config_.num_layers);

    for (int64_t i = 0; i < config_.num_layers; ++i) {
        k_caches_.push_back(zeros(cache_shape, config_.dtype, config_.device));
        v_caches_.push_back(zeros(cache_shape, config_.dtype, config_.device));
    }
}

auto KVCache::update(int64_t layer, const Tensor& new_k, const Tensor& new_v, int64_t pos)
    -> std::pair<Tensor, Tensor> {

    if (layer < 0 || layer >= config_.num_layers) {
        throw std::out_of_range(
            "KVCache::update: layer index " + std::to_string(layer) +
            " out of range [0, " + std::to_string(config_.num_layers) + ")");
    }

    auto new_shape = new_k.shape();
    int64_t new_seq_len = new_shape[2];  // (batch, num_kv_heads, new_seq_len, head_dim)

    if (pos + new_seq_len > config_.max_seq_len) {
        throw std::runtime_error(
            "KVCache::update: pos (" + std::to_string(pos) + ") + new_seq_len (" +
            std::to_string(new_seq_len) + ") exceeds max_seq_len (" +
            std::to_string(config_.max_seq_len) + ")");
    }

    // Copy new key/value data into the cache at position [pos, pos + new_seq_len)
    // We use slice to get a view of the target region and then copy data into it
    Tensor& k_cache = k_caches_[layer];
    Tensor& v_cache = v_caches_[layer];

    // Get the destination slice
    Tensor k_dst = tenzor::slice(k_cache, 2, pos, pos + new_seq_len);
    Tensor v_dst = tenzor::slice(v_cache, 2, pos, pos + new_seq_len);

    // Copy data: use element-wise copy from new_k/new_v into the cache slice
    // Since slice returns a view sharing the same storage, we copy the source data
    // directly into the cache storage at the correct offset
    int64_t batch_size = config_.batch_size;
    int64_t num_kv_heads = config_.num_kv_heads;
    int64_t head_dim = config_.head_dim;
    int64_t elem_size = dtype_size(config_.dtype);

    // Ensure new tensors are contiguous for memcpy
    Tensor new_k_contig = new_k.is_contiguous() ? new_k : new_k.contiguous();
    Tensor new_v_contig = new_v.is_contiguous() ? new_v : new_v.contiguous();

    // Copy row by row along the sequence dimension
    // Cache layout: (batch, num_kv_heads, max_seq_len, head_dim)
    // Row stride in bytes = head_dim * elem_size
    // Seq stride in cache = max_seq_len * head_dim * elem_size
    int64_t row_bytes = head_dim * elem_size;
    int64_t cache_seq_stride = config_.max_seq_len * head_dim * elem_size;
    int64_t new_seq_stride = new_seq_len * head_dim * elem_size;

    auto* k_cache_ptr = static_cast<uint8_t*>(k_cache.data_ptr());
    auto* v_cache_ptr = static_cast<uint8_t*>(v_cache.data_ptr());
    auto* new_k_ptr = static_cast<const uint8_t*>(new_k_contig.data_ptr());
    auto* new_v_ptr = static_cast<const uint8_t*>(new_v_contig.data_ptr());

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_kv_heads; ++h) {
            int64_t bh = b * num_kv_heads + h;
            int64_t cache_offset = bh * cache_seq_stride + pos * row_bytes;
            int64_t new_offset = bh * new_seq_stride;

            std::memcpy(k_cache_ptr + cache_offset,
                       new_k_ptr + new_offset,
                       new_seq_len * row_bytes);
            std::memcpy(v_cache_ptr + cache_offset,
                       new_v_ptr + new_offset,
                       new_seq_len * row_bytes);
        }
    }

    // Return sliced views covering [0, pos + new_seq_len)
    int64_t total_len = pos + new_seq_len;
    Tensor cached_k = tenzor::slice(k_cache, 2, 0, total_len);
    Tensor cached_v = tenzor::slice(v_cache, 2, 0, total_len);

    return {cached_k, cached_v};
}

auto KVCache::get_keys(int64_t layer, int64_t seq_len) const -> Tensor {
    if (layer < 0 || layer >= config_.num_layers) {
        throw std::out_of_range(
            "KVCache::get_keys: layer index " + std::to_string(layer) +
            " out of range [0, " + std::to_string(config_.num_layers) + ")");
    }
    return tenzor::slice(k_caches_[layer], 2, 0, seq_len);
}

auto KVCache::get_values(int64_t layer, int64_t seq_len) const -> Tensor {
    if (layer < 0 || layer >= config_.num_layers) {
        throw std::out_of_range(
            "KVCache::get_values: layer index " + std::to_string(layer) +
            " out of range [0, " + std::to_string(config_.num_layers) + ")");
    }
    return tenzor::slice(v_caches_[layer], 2, 0, seq_len);
}

auto KVCache::reset() -> void {
    for (int64_t i = 0; i < config_.num_layers; ++i) {
        reset_layer(i);
    }
}

auto KVCache::reset_layer(int64_t layer) -> void {
    if (layer < 0 || layer >= config_.num_layers) {
        throw std::out_of_range(
            "KVCache::reset_layer: layer index " + std::to_string(layer) +
            " out of range [0, " + std::to_string(config_.num_layers) + ")");
    }

    // Zero out the cache memory
    int64_t total_bytes = config_.batch_size * config_.num_kv_heads *
                          config_.max_seq_len * config_.head_dim *
                          dtype_size(config_.dtype);

    if (k_caches_[layer].device().type == Device::Type::CPU) {
        std::memset(k_caches_[layer].data_ptr(), 0, total_bytes);
        std::memset(v_caches_[layer].data_ptr(), 0, total_bytes);
    } else {
        // For GPU tensors, replace with fresh zero tensors on the same device
        std::vector<int64_t> cache_shape = {
            config_.batch_size, config_.num_kv_heads,
            config_.max_seq_len, config_.head_dim
        };
        k_caches_[layer] = zeros(cache_shape, config_.dtype, config_.device);
        v_caches_[layer] = zeros(cache_shape, config_.dtype, config_.device);
    }
}

} // namespace tenzor::nn::utils
