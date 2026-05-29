#pragma once

/// @file onednn_cache.hpp
/// @brief Shared thread-safe LRU cache for oneDNN primitives.
///
/// Provides a generic `OneDNNPrimitiveCache<Key, Value, Hash>` template that
/// encapsulates the LRU eviction logic duplicated across math.cpp,
/// activations.cpp, pooling.cpp, nn_kernels.cpp, conv2d.cpp, batchnorm.cpp,
/// and rnn_onednn.hpp.
///
/// Usage:
///   1. Define a CacheKey struct with `operator==`
///   2. Define a CacheKeyHash functor (use `hash_combine` helper)
///   3. Define a CachedPrimitive struct holding your dnnl objects
///   4. Create a `thread_local OneDNNPrimitiveCache<Key, Value, Hash>` instance
///
/// Each file's existing per-op caches can be migrated to use this template
/// one at a time. This header is designed for incremental adoption — existing
/// caches continue to work alongside new ones using this template.
///
/// @note All caches are thread_local so no mutex is needed.
/// @note Full migration of existing caches is deferred to a follow-up PR.

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <unordered_map>

#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// Hash combiner utility
// ============================================================================

/// Combines a seed hash with a new value using the boost hash_combine pattern.
/// Use this to build composite hash functions for cache keys.
///
/// Example:
///   size_t h = 0;
///   hash_combine(h, algo);
///   hash_combine(h, n);
///   hash_combine(h, alpha);
///   return h;
inline void hash_combine(size_t& seed, size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/// Convenience overload that hashes the value before combining.
template <typename T>
inline void hash_combine(size_t& seed, const T& value) {
    hash_combine(seed, std::hash<T>{}(value));
}

// ============================================================================
// Generic LRU Primitive Cache
// ============================================================================

/// Thread-local LRU cache for oneDNN primitives.
///
/// @tparam Key     Cache key type. Must support `operator==`.
/// @tparam Value   Cached primitive type (typically a struct holding dnnl objects).
/// @tparam Hash    Hash functor for Key.
/// @tparam MaxSize Maximum number of cached entries before LRU eviction.
///
/// All operations are O(1) amortized (hash map + doubly-linked list).
/// Not thread-safe — intended for use as `static thread_local`.
template <typename Key, typename Value, typename Hash, size_t MaxSize = 64>
class OneDNNPrimitiveCache {
public:
    /// Look up a cached primitive. Returns nullptr on miss.
    /// On hit, promotes the entry to most-recently-used.
    std::shared_ptr<Value> get(const Key& key) {
        auto it = map_.find(key);
        if (it == map_.end()) {
            return nullptr;
        }
        // Move to front (most recently used)
        lru_.splice(lru_.begin(), lru_, it->second.lru_iter);
        return it->second.value;
    }

    /// Insert or update a cached primitive.
    /// If the cache is full, evicts the least-recently-used entry.
    void put(const Key& key, std::shared_ptr<Value> value) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            // Update existing entry and promote to front
            it->second.value = std::move(value);
            lru_.splice(lru_.begin(), lru_, it->second.lru_iter);
            return;
        }

        // Evict LRU if at capacity
        if (map_.size() >= MaxSize) {
            auto& evict_key = lru_.back();
            map_.erase(evict_key);
            lru_.pop_back();
        }

        // Insert new entry at front
        lru_.push_front(key);
        map_[key] = {std::move(value), lru_.begin()};
    }

    /// Returns current number of cached entries.
    size_t size() const { return map_.size(); }

    /// Clear all cached entries.
    void clear() {
        map_.clear();
        lru_.clear();
    }

private:
    struct Entry {
        std::shared_ptr<Value> value;
        typename std::list<Key>::iterator lru_iter;
    };

    std::unordered_map<Key, Entry, Hash> map_;
    std::list<Key> lru_;
};

// ============================================================================
// Shared oneDNN Engine and Stream accessor
// ============================================================================

#ifdef TENZOR_USE_ONEDNN

/// Returns a thread-local oneDNN CPU engine.
/// Shared across all operations to avoid creating multiple engines per thread.
inline dnnl::engine& get_onednn_engine() {
    static thread_local dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    return engine;
}

/// Returns a thread-local oneDNN stream bound to the shared engine.
inline dnnl::stream& get_onednn_stream() {
    static thread_local dnnl::stream stream(get_onednn_engine());
    return stream;
}

#endif // TENZOR_USE_ONEDNN

// ============================================================================
// W.6: explicit cache-clear API
// ============================================================================
//
// Each per-op .cpp owns a `thread_local` cache (g_conv2d_cache,
// g_batchnorm_cache, …). Long-running training jobs that recycle stream
// configurations would otherwise let these grow to MaxSize entries and
// hold onto the underlying dnnl primitives + reordered weight buffers for
// the process lifetime. `clear_dnnl_cache()` walks every registered cache
// on the calling thread and clears it.
//
// THREADING CONTRACT (important — read before relying on this for memory
// reclamation across a thread pool):
//   The per-op caches are `thread_local`. They are deliberately lock-free:
//   each thread reads/writes only its own instance with no synchronisation.
//   Consequently `clear_dnnl_cache()` clears ONLY the calling thread's
//   caches. It does NOT (and must not) reach into other threads' instances —
//   mutating a `thread_local` cache that another thread may be concurrently
//   reading is a data race / undefined behaviour, and a "registry of live
//   caches" that did so would be incorrect, not merely slow.
//
//   Other threads' caches are still reclaimed deterministically: each
//   thread_local instance is destroyed (freeing its dnnl primitives and
//   reordered-weight buffers) when that thread exits. For an OpenMP / fixed
//   worker pool the recommended pattern under memory pressure is to call
//   `clear_dnnl_cache()` from inside a parallel region (e.g.
//   `#pragma omp parallel { clear_dnnl_cache(); }`) so every worker clears
//   its own cache on its own thread — the only data-race-free way to achieve
//   a synchronous global clear.

/// Register a thread-local cache clear callback. Each per-op cpp registers
/// its cache via a static initializer; clear_dnnl_cache() invokes them all
/// on whatever thread calls clear_dnnl_cache().
void register_dnnl_cache_clear_callback(void (*cb)());

/// Clear the calling thread's oneDNN primitive caches. Idle entries are
/// destroyed; subsequent lookups lazily rebuild. Per the threading contract
/// above this affects only the current thread; other threads' caches are
/// freed on their own thread exit. To clear a worker pool synchronously,
/// invoke this once per worker from within a parallel region.
void clear_dnnl_cache();

} // namespace cpu
} // namespace tenzor
