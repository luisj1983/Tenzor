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
#include <algorithm>

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

    // Validate input ranks/shapes against the cache configuration before
    // deriving any offsets. update() copies new_k/new_v block-by-block using
    // src/dst offsets computed purely from config_ (B, H, D); if the supplied
    // tensors do not match {batch_size, num_kv_heads, *, head_dim} the memcpy
    // would over-read the source and write the wrong cache region (OOB).
    if (new_k.ndim() != 4 || new_v.ndim() != 4) {
        throw std::invalid_argument(
            "KVCache::update: new_k/new_v must be 4-D [batch, num_kv_heads, "
            "new_seq_len, head_dim]; got new_k.ndim()=" +
            std::to_string(new_k.ndim()) + ", new_v.ndim()=" +
            std::to_string(new_v.ndim()));
    }

    auto new_shape = new_k.shape();
    if (new_shape[0] != config_.batch_size || new_shape[1] != config_.num_kv_heads ||
        new_shape[3] != config_.head_dim) {
        throw std::invalid_argument(
            "KVCache::update: new_k shape [" + std::to_string(new_shape[0]) + ", " +
            std::to_string(new_shape[1]) + ", " + std::to_string(new_shape[2]) + ", " +
            std::to_string(new_shape[3]) + "] does not match cache config {batch_size=" +
            std::to_string(config_.batch_size) + ", num_kv_heads=" +
            std::to_string(config_.num_kv_heads) + ", *, head_dim=" +
            std::to_string(config_.head_dim) + "}");
    }
    auto v_shape = new_v.shape();
    if (!std::ranges::equal(v_shape, new_shape)) {
        throw std::invalid_argument(
            "KVCache::update: new_v shape must equal new_k shape");
    }

    int64_t new_seq_len = new_shape[2];  // (batch, num_kv_heads, new_seq_len, head_dim)

    // Reject negative positions/lengths: dst offsets promote to size_t and a
    // negative pos becomes a huge index, writing far out of bounds.
    if (pos < 0 || new_seq_len < 0) {
        throw std::out_of_range(
            "KVCache::update: pos (" + std::to_string(pos) + ") and new_seq_len (" +
            std::to_string(new_seq_len) + ") must be non-negative");
    }

    if (pos + new_seq_len > config_.max_seq_len) {
        throw std::runtime_error(
            "KVCache::update: pos (" + std::to_string(pos) + ") + new_seq_len (" +
            std::to_string(new_seq_len) + ") exceeds max_seq_len (" +
            std::to_string(config_.max_seq_len) + ")");
    }

    // Write new key/value data into the cache at [pos, pos + new_seq_len) along
    // dim 2. The previous implementation used slice_scatter, which forces a
    // contiguous copy and returns a brand-new full-size tensor reassigned back
    // into k_caches_/v_caches_ — O(max_seq_len) per token, O(max_seq_len^2)
    // over a generation, defeating the whole point of pre-allocating the cache.
    //
    // The cache is pre-allocated contiguous [B, H, max_seq_len, D]. For each
    // (b, h) the destination block [pos, pos+new_seq_len) x D is contiguous, as
    // is the matching source block in new_k/new_v [B, H, new_seq_len, D], so the
    // whole update is a set of plain memcpys into the existing buffer (CPU) with
    // no allocation. Non-CPU or non-contiguous caches fall back to slice_scatter.
    Tensor new_k_contig = new_k.is_contiguous() ? new_k : new_k.contiguous();
    Tensor new_v_contig = new_v.is_contiguous() ? new_v : new_v.contiguous();

    const bool can_inplace =
        config_.device.type == Device::Type::CPU &&
        k_caches_[layer].is_contiguous() && v_caches_[layer].is_contiguous();

    if (can_inplace) {
        const int64_t B = config_.batch_size;
        const int64_t H = config_.num_kv_heads;
        const int64_t S = config_.max_seq_len;
        const int64_t D = config_.head_dim;
        const size_t elem = dtype_size(config_.dtype);
        const int64_t rows = B * H;                  // (b,h) blocks
        const size_t block = static_cast<size_t>(new_seq_len) * D * elem;

        auto write_inplace = [&](Tensor& dst, const Tensor& src) {
            char* dst_base = static_cast<char*>(dst.data_ptr());
            const char* src_base = static_cast<const char*>(src.data_ptr());
            for (int64_t r = 0; r < rows; ++r) {
                size_t dst_off = (static_cast<size_t>(r) * S + pos) * D * elem;
                size_t src_off = static_cast<size_t>(r) * new_seq_len * D * elem;
                std::memcpy(dst_base + dst_off, src_base + src_off, block);
            }
        };
        write_inplace(k_caches_[layer], new_k_contig);
        write_inplace(v_caches_[layer], new_v_contig);
    } else {
        k_caches_[layer] = tenzor::slice_scatter(
            k_caches_[layer], new_k_contig, /*dim=*/2, /*start=*/pos,
            /*end=*/pos + new_seq_len, /*step=*/1);
        v_caches_[layer] = tenzor::slice_scatter(
            v_caches_[layer], new_v_contig, /*dim=*/2, /*start=*/pos,
            /*end=*/pos + new_seq_len, /*step=*/1);
    }

    // Return sliced views covering [0, pos + new_seq_len)
    int64_t total_len = pos + new_seq_len;
    Tensor cached_k = tenzor::slice(k_caches_[layer], 2, 0, total_len);
    Tensor cached_v = tenzor::slice(v_caches_[layer], 2, 0, total_len);

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
