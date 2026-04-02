/**
 * @file memory_profiler.hpp
 * @brief Global memory profiler with atomic counters for allocation tracking.
 *
 * Provides a thread-safe singleton that records every allocation and
 * deallocation, tracks current/peak/total bytes, and exposes a
 * human-readable summary.  Intended to be called from allocators
 * (CachingAllocator, TensorImpl constructor, etc.).
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace tenzor {

/// Snapshot of global memory profiler counters.
struct MemoryStats {
    int64_t current_allocated_bytes;
    int64_t peak_allocated_bytes;
    int64_t total_allocated_bytes;
    int64_t allocation_count;
    int64_t deallocation_count;
};

/**
 * @brief Process-wide memory profiler singleton.
 *
 * All counters use relaxed atomics — they are not sequentially consistent
 * with each other, but each individual counter is correct.  This is the
 * right trade-off for a profiler (minimal overhead, approximate snapshots).
 */
class MemoryProfiler {
public:
    /// Access the singleton instance.
    static auto instance() -> MemoryProfiler&;

    /// Record an allocation of @p bytes.
    auto on_allocate(size_t bytes) -> void;

    /// Record a deallocation of @p bytes.
    auto on_deallocate(size_t bytes) -> void;

    /// Return a snapshot of all counters.
    auto memory_stats() const -> MemoryStats;

    /// Reset peak to current (does not touch other counters).
    auto reset_peak_memory_stats() -> void;

    /// Human-readable multi-line summary string.
    auto memory_summary() const -> std::string;

    // Non-copyable / non-movable (singleton)
    MemoryProfiler(const MemoryProfiler&) = delete;
    MemoryProfiler& operator=(const MemoryProfiler&) = delete;

private:
    MemoryProfiler() = default;

    std::atomic<int64_t> current_allocated_bytes_{0};
    std::atomic<int64_t> peak_allocated_bytes_{0};
    std::atomic<int64_t> total_allocated_bytes_{0};
    std::atomic<int64_t> allocation_count_{0};
    std::atomic<int64_t> deallocation_count_{0};
};

} // namespace tenzor
