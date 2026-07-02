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
    auto set_alignment(size_t alignment) -> void {
        // Alignment must be a power of 2 (and non-zero): the planner rounds sizes
        // with `(x + a - 1) & ~(a - 1)`, which corrupts offsets for a==0 or a
        // non-power-of-2.
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("MemoryPlanner::set_alignment: alignment must be a non-zero power of 2");
        }
        alignment_ = alignment;
    }

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
        // Adding (alignment_ - 1) can wrap around for a pathologically large
        // size, yielding a tiny aligned value and a buffer far too small.
        // Signal the overflow with 0, which callers treat as "don't plan".
        if (size > std::numeric_limits<size_t>::max() - (alignment_ - 1)) return 0;
        return (size + alignment_ - 1) & ~(alignment_ - 1);
    }

    size_t alignment_{64};  ///< Buffer offset alignment (bytes)
};

// ============================================================================
// Rematerialization Support
// ============================================================================

/**
 * @brief Cost model for deciding whether to rematerialize a value.
 *
 * Compares the FLOPS cost of recomputing a value against the memory
 * saved by not keeping it alive. Operations that are cheap to recompute
 * (elementwise, activations) are good candidates for rematerialization.
 */
struct RematerializationCandidate {
    std::string value_id;           ///< ID of the value to potentially recompute
    std::string producer_node_name; ///< Name of the node that produces this value
    OpType producer_op;             ///< Operation type of the producer
    size_t memory_saved;            ///< Bytes freed by dropping this value
    double recompute_flops;         ///< Estimated FLOPS to recompute
    double cost_ratio;              ///< memory_saved / recompute_flops (higher = better candidate)
};

/**
 * @brief Rematerialization planner that identifies values to recompute.
 *
 * Analyzes the graph to find intermediate values with long live ranges
 * whose producers are cheap to recompute (activations, elementwise ops).
 * By dropping these values and recomputing them when needed, peak memory
 * usage can be reduced at the cost of extra compute.
 *
 * The planner uses a cost model:
 *   score = memory_saved_bytes / recompute_flops
 * Values above a threshold score are selected for rematerialization.
 *
 * @code
 * RematerializationPlanner planner;
 * planner.set_memory_budget(1024 * 1024 * 512);  // 512 MB
 * auto candidates = planner.find_candidates(graph);
 * planner.apply(graph, candidates);
 * @endcode
 */
class RematerializationPlanner {
public:
    /**
     * @brief Find rematerialization candidates in the graph.
     *
     * Scans intermediate values for those whose producers are cheap
     * to recompute and whose live ranges consume significant memory.
     *
     * @param graph Graph to analyze
     * @return Sorted candidates (best ratio first)
     */
    auto find_candidates(const Graph& graph) -> std::vector<RematerializationCandidate>;

    /**
     * @brief Apply rematerialization by inserting recompute nodes.
     *
     * For each selected candidate, inserts a duplicate of the producer
     * node at the point of last use, allowing the original value's
     * memory to be freed earlier.
     *
     * @param graph Graph to modify
     * @param candidates Candidates to rematerialize
     * @return Number of values rematerialized
     */
    auto apply(Graph& graph,
               const std::vector<RematerializationCandidate>& candidates) -> size_t;

    /**
     * @brief Set the memory budget. Only rematerialize if peak usage
     *        exceeds this budget.
     *
     * @param bytes Memory budget in bytes (0 = always rematerialize good candidates)
     */
    auto set_memory_budget(size_t bytes) -> void { memory_budget_ = bytes; }

    /**
     * @brief Set the minimum cost ratio for a candidate to be selected.
     *
     * @param ratio Minimum memory_saved / recompute_flops ratio
     */
    auto set_min_cost_ratio(double ratio) -> void { min_cost_ratio_ = ratio; }

private:
    /**
     * @brief Check if an operation is cheap to recompute.
     *
     * Activations (ReLU, GELU, Sigmoid, Tanh) and elementwise ops
     * (Add, Mul, Exp, Log, Sqrt, etc.) are considered cheap.
     *
     * @param op Operation type
     * @return true if operation is cheap
     */
    auto is_cheap_to_recompute(OpType op) -> bool;

    /**
     * @brief Estimate FLOPS for recomputing a value.
     *
     * For elementwise ops: FLOPS = numel
     * For activations: FLOPS = numel * activation_cost_factor
     *
     * @param op Operation type
     * @param shape Output shape
     * @return Estimated FLOPS
     */
    auto estimate_flops(OpType op, const std::vector<int64_t>& shape) -> double;

    size_t memory_budget_{0};        ///< Memory budget (0 = unlimited)
    double min_cost_ratio_{1.0};     ///< Minimum cost ratio for selection
};

// ============================================================================
// Memory Swapping Support
// ============================================================================

/**
 * @brief Describes a swap-out/swap-in pair for a value.
 *
 * When a large activation cannot be rematerialized, it can be swapped
 * out to CPU memory and prefetched back before it is needed.
 */
struct SwapSchedule {
    std::string value_id;     ///< Value to swap
    size_t swap_out_after;    ///< Node index after which to schedule SwapOut
    size_t swap_in_before;    ///< Node index before which to schedule SwapIn
    size_t size_bytes;        ///< Size of the value in bytes
};

/**
 * @brief Memory swap planner that inserts SwapOut/SwapIn pseudo-nodes.
 *
 * For large activations that cannot be cheaply rematerialized, schedules
 * asynchronous GPU->CPU transfers (SwapOut) after the value is produced
 * and CPU->GPU prefetches (SwapIn) before the value is next consumed.
 *
 * The planner inserts pseudo-nodes into the graph:
 * - SwapOut: Placed immediately after the producer, triggers async D2H copy
 * - SwapIn: Placed before the last consumer, triggers async H2D prefetch
 *
 * @code
 * MemorySwapPlanner planner;
 * planner.set_swap_threshold(64 * 1024 * 1024);  // 64 MB
 * auto schedule = planner.plan(graph);
 * planner.apply(graph, schedule);
 * @endcode
 */
class MemorySwapPlanner {
public:
    /**
     * @brief Plan swap schedules for large activations.
     *
     * Identifies values whose live ranges are long and sizes exceed
     * the swap threshold. Computes optimal swap-out and swap-in points.
     *
     * @param graph Graph to analyze
     * @return Vector of swap schedules
     */
    auto plan(const Graph& graph) -> std::vector<SwapSchedule>;

    /**
     * @brief Apply swap schedules by inserting SwapOut/SwapIn nodes.
     *
     * @param graph Graph to modify
     * @param schedules Swap schedules to apply
     * @return Number of swap pairs inserted
     */
    auto apply(Graph& graph, const std::vector<SwapSchedule>& schedules) -> size_t;

    /**
     * @brief Set the minimum value size for swapping consideration.
     *
     * Values smaller than this threshold will not be swapped.
     *
     * @param bytes Minimum size in bytes (default: 64 MB)
     */
    auto set_swap_threshold(size_t bytes) -> void { swap_threshold_ = bytes; }

    /**
     * @brief Set the minimum live range gap for swapping.
     *
     * Only swap values whose live range spans at least this many nodes,
     * to ensure there is enough time for async transfer.
     *
     * @param gap Minimum node gap (default: 5)
     */
    auto set_min_gap(size_t gap) -> void { min_gap_ = gap; }

private:
    size_t swap_threshold_{64 * 1024 * 1024};  ///< Min value size for swap (bytes)
    size_t min_gap_{5};                         ///< Min live range gap (nodes)
};

} // namespace jit
} // namespace tenzor
