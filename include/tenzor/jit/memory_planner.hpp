/**
 * @file memory_planner.hpp
 * @brief Memory planning for JIT compiled modules
 *
 * Performs live range analysis and greedy buffer assignment to enable
 * memory reuse between non-overlapping intermediate values. This reduces
 * allocation overhead during graph execution by pre-allocating memory
 * pools and assigning buffer slots to values at compile time.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#include "graph.hpp"

namespace tenzor {
namespace jit {

/**
 * @brief Describes the allocation of a single value within a memory pool.
 *
 * Each value that participates in memory planning receives a BufferAllocation
 * specifying which pool buffer it belongs to and its byte offset within that buffer.
 */
struct BufferAllocation {
    size_t buffer_id;    ///< Index of the pool buffer
    size_t offset;       ///< Byte offset within the pool buffer
    size_t size;         ///< Size in bytes of this allocation
};

/**
 * @brief Describes the live range of a value in the graph execution order.
 *
 * The live range spans from the node index where the value is first produced
 * (born) to the node index where it is last consumed (dies). Two values with
 * non-overlapping live ranges can share the same memory.
 */
struct LiveRange {
    size_t begin;        ///< Node index where value is produced
    size_t end;          ///< Node index where value is last consumed
    std::string value_id; ///< ID of the corresponding Value
    size_t size;         ///< Size in bytes
};

/**
 * @brief Complete memory plan for a JIT graph.
 *
 * Contains the pool buffer sizes and the mapping from value IDs to
 * their buffer assignments. The executor uses this plan to pre-allocate
 * memory pools and place tensors at the correct offsets.
 *
 * @code
 * MemoryPlanner planner;
 * auto plan = planner.plan(graph);
 *
 * // Pre-allocate pools
 * std::vector<void*> pools;
 * for (size_t pool_size : plan.pool_sizes) {
 *     pools.push_back(allocator.allocate(pool_size));
 * }
 *
 * // Look up where a value should be placed
 * auto it = plan.value_allocations.find(value_id);
 * if (it != plan.value_allocations.end()) {
 *     void* ptr = static_cast<char*>(pools[it->second.buffer_id]) + it->second.offset;
 * }
 * @endcode
 */
struct MemoryPlan {
    std::vector<size_t> pool_sizes;                                ///< Size of each pool buffer in bytes
    std::unordered_map<std::string, BufferAllocation> value_allocations;  ///< Value ID -> allocation
    size_t total_memory{0};                                        ///< Sum of all pool sizes
    size_t num_values_planned{0};                                  ///< Number of values with assignments
    size_t num_values_reused{0};                                   ///< Number of values sharing buffers
};

/**
 * @brief Memory planner that analyzes a JIT graph and produces a memory plan.
 *
 * The planner performs three phases:
 *   1. **Live range analysis**: Iterates nodes in topological order to determine
 *      when each intermediate value is first produced and last consumed.
 *   2. **Size computation**: Computes the byte size of each value from its
 *      shape and dtype.
 *   3. **Greedy buffer assignment**: Assigns values to buffer slots, reusing
 *      slots when live ranges do not overlap. Values are processed largest-first
 *      to maximize reuse opportunities.
 *
 * Graph inputs and outputs are excluded from memory planning since their
 * lifetimes extend beyond the graph execution. Constants are also excluded
 * since they hold persistent weight data.
 *
 * After planning, each eligible Value in the graph is annotated with its
 * buffer_id and buffer_offset for use during execution.
 *
 * @code
 * MemoryPlanner planner;
 * MemoryPlan plan = planner.plan(graph);
 *
 * // plan.total_memory gives the total pre-allocation needed
 * // plan.num_values_reused shows how many values share memory
 * @endcode
 */
class MemoryPlanner {
public:
    /**
     * @brief Analyze graph and produce a memory plan.
     *
     * Computes live ranges, value sizes, and performs greedy buffer
     * assignment. Annotates Value objects with buffer_id and buffer_offset.
     *
     * @param graph Graph to plan (values are annotated in-place)
     * @return Complete memory plan
     */
    auto plan(Graph& graph) -> MemoryPlan;

    /**
     * @brief Set minimum alignment for buffer allocations.
     *
     * All offsets within a buffer will be aligned to this boundary.
     * Default is 64 bytes (cache line aligned).
     *
     * @param alignment Alignment in bytes (must be power of 2)
     */
    auto set_alignment(size_t alignment) -> void { alignment_ = alignment; }

    /**
     * @brief Get the alignment setting.
     *
     * @return Alignment in bytes
     */
    auto alignment() const -> size_t { return alignment_; }

private:
    /**
     * @brief Compute live ranges for all intermediate values.
     *
     * Iterates nodes in topological order. A value is "born" at the index
     * of the node that produces it, and "dies" at the index of the last
     * node that consumes it.
     *
     * @param graph Graph with nodes in topological order
     * @return Vector of live ranges for plannable values
     */
    auto compute_live_ranges(const Graph& graph) -> std::vector<LiveRange>;

    /**
     * @brief Compute byte size for a value.
     *
     * Size = dtype_size(dtype) * product(shape dimensions).
     * Returns 0 for scalar (empty shape) values.
     *
     * @param value Value to compute size for
     * @return Size in bytes
     */
    auto compute_value_size(const Value& value) -> size_t;

    /**
     * @brief Perform greedy buffer assignment.
     *
     * Sorts values by size (largest first) and assigns each to the smallest
     * existing buffer that fits and has no live range overlap. If no suitable
     * buffer exists, a new buffer is created.
     *
     * @param live_ranges Live ranges with sizes filled in
     * @return Pair of (pool_sizes, value_id -> BufferAllocation map)
     */
    auto greedy_assign(std::vector<LiveRange>& live_ranges)
        -> std::pair<std::vector<size_t>,
                     std::unordered_map<std::string, BufferAllocation>>;

    /**
     * @brief Check if a value should be excluded from memory planning.
     *
     * Graph inputs, graph outputs, and constants are excluded because
     * their lifetimes extend beyond a single graph execution.
     *
     * @param value Value to check
     * @param graph Graph containing the value
     * @return true if the value should be skipped
     */
    auto is_excluded(const Value& value, const Graph& graph) -> bool;

    /**
     * @brief Round up a value to the next multiple of alignment.
     *
     * @param size Value to align
     * @return Aligned value
     */
    auto align_up(size_t size) const -> size_t {
        return (size + alignment_ - 1) & ~(alignment_ - 1);
    }

    size_t alignment_{64};  ///< Buffer offset alignment (bytes)
};

} // namespace jit
} // namespace tenzor
