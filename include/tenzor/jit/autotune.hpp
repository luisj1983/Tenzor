/**
 * @file autotune.hpp
 * @brief Kernel autotuning cache for JIT compilation
 *
 * Provides a thread-safe cache for kernel autotuning results. During
 * execution, multiple kernel algorithms may be benchmarked for a given
 * operation configuration (op + dtype + shapes). The fastest algorithm
 * is recorded and reused in subsequent runs.
 *
 * The cache can be persisted to disk in JSON format so tuning results
 * survive across program restarts.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tenzor {
namespace jit {

/**
 * @brief Describes a candidate kernel algorithm for benchmarking.
 *
 * Each candidate represents a specific algorithm implementation for
 * an operation, identified by an algorithm ID and workspace requirement.
 */
struct KernelCandidate {
    std::string name;          ///< Human-readable algorithm name
    int algorithm_id;          ///< Backend-specific algorithm identifier
    size_t workspace_bytes;    ///< Workspace memory required by this algorithm
};

/**
 * @brief Thread-safe cache for kernel autotuning results.
 *
 * Stores the best algorithm ID for each unique operation configuration.
 * Keys encode the operation name, data type, and tensor shapes:
 *   `op_name:dtype:shape1xshape2x...`
 *
 * The cache supports concurrent reads (shared lock) and exclusive writes.
 * Results can be saved to and loaded from a JSON file at
 * `~/.tenzor/autotune_cache.json`.
 *
 * @code
 * auto& cache = AutotuneCache::instance();
 *
 * std::string key = "MatMul:Float32:512x512x512";
 * auto best = cache.lookup(key);
 * if (!best) {
 *     // Benchmark candidates
 *     for (auto& candidate : candidates) {
 *         double time = benchmark(candidate);
 *         cache.record(key, candidate.algorithm_id, time);
 *     }
 *     best = cache.lookup(key);
 * }
 * @endcode
 */
class AutotuneCache {
public:
    /**
     * @brief Get the global singleton instance.
     *
     * @return Reference to the global cache
     */
    static auto instance() -> AutotuneCache&;

    /**
     * @brief Look up the best algorithm for a given key.
     *
     * Thread-safe (shared lock). Returns the algorithm ID with the
     * lowest recorded benchmark time for this key.
     *
     * @param key Operation configuration key
     * @return Best algorithm ID, or std::nullopt if no entry exists
     */
    auto lookup(const std::string& key) const -> std::optional<int>;

    /**
     * @brief Record a benchmark result for a key.
     *
     * Thread-safe (exclusive lock). If the recorded time is better than
     * the current best for this key, the entry is updated.
     *
     * @param key Operation configuration key
     * @param algorithm_id Algorithm identifier
     * @param time_ms Benchmark time in milliseconds
     */
    auto record(const std::string& key, int algorithm_id, double time_ms) -> void;

    /**
     * @brief Save the cache to a JSON file.
     *
     * Thread-safe (shared lock). The file is written atomically
     * (write to temp, then rename).
     *
     * @param path Output file path
     */
    auto save(const std::string& path) const -> void;

    /**
     * @brief Load the cache from a JSON file.
     *
     * Thread-safe (exclusive lock). Existing entries are replaced.
     *
     * @param path Input file path
     */
    auto load(const std::string& path) -> void;

    /**
     * @brief Save to the default cache location (~/.tenzor/autotune_cache.json).
     */
    auto save_default() const -> void;

    /**
     * @brief Load from the default cache location (~/.tenzor/autotune_cache.json).
     */
    auto load_default() -> void;

    /**
     * @brief Get the default cache file path.
     *
     * @return Path to ~/.tenzor/autotune_cache.json
     */
    static auto default_path() -> std::string;

    /**
     * @brief Clear all cached entries.
     *
     * Thread-safe (exclusive lock).
     */
    auto clear() -> void;

    /**
     * @brief Get number of cached entries.
     *
     * Thread-safe (shared lock).
     *
     * @return Number of entries
     */
    auto size() const -> size_t;

    /**
     * @brief Build a cache key from operation parameters.
     *
     * Convenience method for constructing keys in the standard format.
     *
     * @param op_name Operation name (e.g., "MatMul", "Conv2d")
     * @param dtype Data type string (e.g., "Float32")
     * @param shapes Tensor shapes
     * @return Formatted key string
     */
    static auto make_key(const std::string& op_name,
                         const std::string& dtype,
                         const std::vector<std::vector<int64_t>>& shapes) -> std::string;

private:
    AutotuneCache() = default;
    ~AutotuneCache() = default;
    AutotuneCache(const AutotuneCache&) = delete;
    AutotuneCache& operator=(const AutotuneCache&) = delete;

    /**
     * @brief Entry storing the best algorithm and its time.
     */
    struct CacheEntry {
        int algorithm_id;    ///< Best algorithm ID
        double time_ms;      ///< Best recorded time
    };

    mutable std::shared_mutex mutex_;                         ///< Reader-writer lock
    std::unordered_map<std::string, CacheEntry> cache_;       ///< Key -> best entry
};

} // namespace jit
} // namespace tenzor
