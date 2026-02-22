/**
 * @file memory_planner.cpp
 * @brief Implementation of memory planning for JIT compiled modules
 *
 * Performs live range analysis and greedy buffer assignment to enable
 * memory reuse between non-overlapping intermediate values in a JIT graph.
 */

#include "../../include/tenzor/jit/memory_planner.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/core/dtype.hpp"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <unordered_set>

namespace tenzor {
namespace jit {

// ============================================================================
// MemoryPlanner - Main entry point
// ============================================================================

auto MemoryPlanner::plan(Graph& graph) -> MemoryPlan {
    MemoryPlan result;

    // Ensure topological order before analysis
    graph.topological_sort();

    // Phase 1: Compute live ranges for intermediate values
    auto live_ranges = compute_live_ranges(graph);

    if (live_ranges.empty()) {
        return result;
    }

    // Phase 2: Greedy buffer assignment
    auto [pool_sizes, allocations] = greedy_assign(live_ranges);

    // Phase 3: Annotate graph values with buffer assignments
    size_t num_reused = 0;
    for (const auto& [value_id, alloc] : allocations) {
        auto value = graph.get_value(value_id);
        if (value) {
            value->set_buffer_id(alloc.buffer_id);
            value->set_buffer_offset(alloc.offset);
        }
    }

    // Count reuse: if a buffer is used by more than one value, those are reused
    std::unordered_map<size_t, size_t> buffer_use_counts;
    for (const auto& [value_id, alloc] : allocations) {
        buffer_use_counts[alloc.buffer_id]++;
    }
    for (const auto& [buffer_id, count] : buffer_use_counts) {
        if (count > 1) {
            num_reused += count;
        }
    }

    // Fill result
    result.pool_sizes = std::move(pool_sizes);
    result.value_allocations = std::move(allocations);
    result.total_memory = 0;
    for (size_t ps : result.pool_sizes) {
        result.total_memory += ps;
    }
    result.num_values_planned = live_ranges.size();
    result.num_values_reused = num_reused;

    return result;
}

// ============================================================================
// Live Range Analysis
// ============================================================================

auto MemoryPlanner::compute_live_ranges(const Graph& graph) -> std::vector<LiveRange> {
    std::vector<LiveRange> ranges;

    // Build node index map: node pointer -> topological index
    std::unordered_map<const Node*, size_t> node_index;
    const auto& nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        node_index[nodes[i].get()] = i;
    }

    // Track first-use (birth) and last-use (death) for each value
    // Birth = index of the node that produces the value
    // Death = index of the last node that consumes the value
    struct RangeInfo {
        size_t begin{SIZE_MAX};
        size_t end{0};
        std::string value_id;
    };

    std::unordered_map<std::string, RangeInfo> range_map;

    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];

        // Output values are born at this node index
        for (const auto& output : node->outputs()) {
            if (is_excluded(*output, graph)) {
                continue;
            }

            auto& info = range_map[output->id()];
            info.value_id = output->id();
            info.begin = i;
            info.end = i;  // At minimum, lives through its producing node
        }

        // Input values: update their last-use to this node index
        for (const auto& input : node->inputs()) {
            auto it = range_map.find(input->id());
            if (it != range_map.end()) {
                it->second.end = std::max(it->second.end, i);
            }
        }
    }

    // Convert to LiveRange structs with sizes computed
    ranges.reserve(range_map.size());
    for (auto& [id, info] : range_map) {
        auto value = graph.get_value(id);
        if (!value) continue;

        size_t size = compute_value_size(*value);
        if (size == 0) continue;  // Skip zero-size values

        LiveRange lr;
        lr.begin = info.begin;
        lr.end = info.end;
        lr.value_id = info.value_id;
        lr.size = size;
        ranges.push_back(std::move(lr));
    }

    return ranges;
}

// ============================================================================
// Value Size Computation
// ============================================================================

auto MemoryPlanner::compute_value_size(const Value& value) -> size_t {
    const auto& shape = value.shape();
    if (shape.empty()) {
        // Scalar value: single element
        return dtype_size(value.dtype());
    }

    size_t numel = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            // Dynamic or invalid dimension: cannot plan statically
            return 0;
        }
        numel *= static_cast<size_t>(dim);
    }

    return numel * dtype_size(value.dtype());
}

// ============================================================================
// Greedy Buffer Assignment
// ============================================================================

auto MemoryPlanner::greedy_assign(std::vector<LiveRange>& live_ranges)
    -> std::pair<std::vector<size_t>,
                 std::unordered_map<std::string, BufferAllocation>> {

    std::vector<size_t> pool_sizes;
    std::unordered_map<std::string, BufferAllocation> allocations;

    if (live_ranges.empty()) {
        return {pool_sizes, allocations};
    }

    // Sort by size descending (largest first) for better packing.
    // Ties broken by earlier birth time to encourage temporal locality.
    std::sort(live_ranges.begin(), live_ranges.end(),
        [](const LiveRange& a, const LiveRange& b) {
            if (a.size != b.size) return a.size > b.size;
            return a.begin < b.begin;
        });

    // Each buffer slot tracks: current high-water-mark size and occupied intervals.
    // An interval is [begin, end] inclusive.
    struct BufferSlot {
        size_t high_water_mark{0};  ///< Maximum allocation size in this buffer
        std::vector<std::pair<size_t, size_t>> intervals;  ///< Occupied time intervals
    };

    std::vector<BufferSlot> slots;

    for (const auto& lr : live_ranges) {
        size_t aligned_size = align_up(lr.size);

        // Try to find an existing slot that:
        //   1. Has no overlapping live range
        //   2. Has sufficient size (or can be grown)
        // Among candidates, prefer the one with the smallest high-water mark
        // that is >= aligned_size (best-fit), to minimize wasted space.
        size_t best_slot = SIZE_MAX;
        size_t best_hwm = SIZE_MAX;

        for (size_t s = 0; s < slots.size(); ++s) {
            // Check for time overlap with any interval in this slot
            bool overlaps = false;
            for (const auto& [ib, ie] : slots[s].intervals) {
                if (lr.begin <= ie && lr.end >= ib) {
                    overlaps = true;
                    break;
                }
            }
            if (overlaps) continue;

            // This slot is available. Prefer the smallest slot that already
            // fits (best-fit), which avoids growing small slots unnecessarily.
            size_t slot_hwm = slots[s].high_water_mark;
            if (slot_hwm >= aligned_size) {
                // Fits without growth - prefer smallest such slot
                if (best_slot == SIZE_MAX || slot_hwm < best_hwm) {
                    best_slot = s;
                    best_hwm = slot_hwm;
                }
            } else if (best_slot == SIZE_MAX) {
                // Doesn't fit but no fitting slot found yet - track as fallback
                // We'll grow this slot. Prefer the one closest to our size
                // (to minimize growth).
                best_slot = s;
                best_hwm = slot_hwm;
            } else if (best_hwm < aligned_size && slot_hwm > best_hwm) {
                // Both don't fit, prefer the larger one (less growth needed)
                best_slot = s;
                best_hwm = slot_hwm;
            }
        }

        if (best_slot != SIZE_MAX) {
            // Reuse existing slot
            auto& slot = slots[best_slot];
            slot.intervals.push_back({lr.begin, lr.end});
            slot.high_water_mark = std::max(slot.high_water_mark, aligned_size);

            BufferAllocation alloc;
            alloc.buffer_id = best_slot;
            alloc.offset = 0;
            alloc.size = lr.size;
            allocations[lr.value_id] = alloc;
        } else {
            // Create new slot
            size_t new_slot_id = slots.size();
            BufferSlot new_slot;
            new_slot.high_water_mark = aligned_size;
            new_slot.intervals.push_back({lr.begin, lr.end});
            slots.push_back(std::move(new_slot));

            BufferAllocation alloc;
            alloc.buffer_id = new_slot_id;
            alloc.offset = 0;
            alloc.size = lr.size;
            allocations[lr.value_id] = alloc;
        }
    }

    // Compute final pool sizes from high-water marks
    pool_sizes.reserve(slots.size());
    for (const auto& slot : slots) {
        pool_sizes.push_back(slot.high_water_mark);
    }

    return {pool_sizes, allocations};
}

// ============================================================================
// Exclusion Check
// ============================================================================

auto MemoryPlanner::is_excluded(const Value& value, const Graph& graph) -> bool {
    // Exclude graph inputs
    for (const auto& input : graph.inputs()) {
        if (input->id() == value.id()) {
            return true;
        }
    }

    // Exclude graph outputs (their memory must persist after execution)
    for (const auto& output : graph.outputs()) {
        if (output->id() == value.id()) {
            return true;
        }
    }

    // Exclude constants (they hold persistent weight data)
    auto producer = value.node();
    if (producer && producer->op_type() == OpType::Constant) {
        return true;
    }

    return false;
}

} // namespace jit
} // namespace tenzor
