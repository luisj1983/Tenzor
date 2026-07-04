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
#include <functional>
#include <numeric>
#include <unordered_set>

namespace tenzor {
namespace jit {

namespace {
// Compute the byte size of a tensor given its shape and per-element size,
// guarding every multiplication against size_t overflow. Returns 0 (which all
// callers treat as "cannot plan this value statically") on a non-positive dim
// or on overflow. Shapes loaded from an untrusted .graph file are only
// validated non-negative in serialization, not bounded for the planner, so a
// large-but-positive shape product could otherwise silently wrap to a tiny
// size and under-allocate a buffer slot.
inline auto checked_byte_size(const std::vector<int64_t>& shape,
                              size_t elem_size) -> size_t {
    size_t numel = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) return 0;  // dynamic or invalid dimension
        if (__builtin_mul_overflow(numel, static_cast<size_t>(dim), &numel)) {
            return 0;  // overflow: skip planning this value
        }
    }
    size_t bytes = 0;
    if (__builtin_mul_overflow(numel, elem_size, &bytes)) return 0;
    return bytes;
}
}  // namespace

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
            // The first occupant of a buffer did not reuse previously-owned
            // memory; only the remaining (count - 1) values actually reuse it.
            num_reused += count - 1;
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

        // Control-flow nodes (If/Loop) execute their subgraph bodies at this
        // node's position. Any OUTER value referenced inside a body must stay
        // live through this node; the top-level input scan above misses those
        // (they are not threaded as this node's direct inputs), which would let
        // the planner free their buffer while the body still reads it.
        std::function<void(const std::shared_ptr<Graph>&)> extend_for_subgraph =
            [&](const std::shared_ptr<Graph>& sub) {
                if (!sub) return;
                for (const auto& sn : sub->nodes()) {
                    for (const auto& sin : sn->inputs()) {
                        auto it = range_map.find(sin->id());
                        if (it != range_map.end()) {
                            it->second.end = std::max(it->second.end, i);
                        }
                    }
                    extend_for_subgraph(sn->then_branch());
                    extend_for_subgraph(sn->else_branch());
                    extend_for_subgraph(sn->body());
                }
            };
        extend_for_subgraph(node->then_branch());
        extend_for_subgraph(node->else_branch());
        extend_for_subgraph(node->body());
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
        lr.device = value->device();
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

    return checked_byte_size(shape, dtype_size(value.dtype()));
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
        Device device{};            ///< Device this pool buffer lives on
    };

    std::vector<BufferSlot> slots;

    for (const auto& lr : live_ranges) {
        size_t aligned_size = align_up(lr.size);
        // align_up returns 0 on overflow (or for a genuinely zero-size value);
        // in either case there is nothing to place, so skip planning it.
        if (aligned_size == 0) continue;

        // Try to find an existing slot that:
        //   1. Has no overlapping live range
        //   2. Has sufficient size (or can be grown)
        // Among candidates, prefer the smallest slot whose high-water mark
        // already fits aligned_size (best-fit, no growth). Only if no fitting
        // slot exists do we grow the largest non-fitting slot (least growth).
        //
        // Tracking the two candidate classes separately is required for
        // correctness of the heuristic: a single best_slot/best_hwm pair (as a
        // prior version used) lets a non-fitting slot encountered first lock in
        // best_hwm < aligned_size, after which a later fitting slot — which by
        // definition has slot_hwm >= aligned_size > best_hwm — can never win the
        // `slot_hwm < best_hwm` test, so the planner would grow the smaller
        // fallback instead of reusing an already-large-enough slot, inflating
        // the total pool order-dependently.
        size_t best_fit_slot = SIZE_MAX;     // smallest slot with hwm >= aligned_size
        size_t best_fit_hwm = SIZE_MAX;
        size_t best_grow_slot = SIZE_MAX;    // largest slot with hwm < aligned_size
        size_t best_grow_hwm = 0;

        for (size_t s = 0; s < slots.size(); ++s) {
            // A pool buffer is a single device allocation; never let a value on
            // one device reuse a slot backing another device (that would alias a
            // CPU buffer with a GPU buffer -> data corruption / invalid pointer).
            if (slots[s].device != lr.device) continue;

            // Check for time overlap with any interval in this slot
            bool overlaps = false;
            for (const auto& [ib, ie] : slots[s].intervals) {
                if (lr.begin <= ie && lr.end >= ib) {
                    overlaps = true;
                    break;
                }
            }
            if (overlaps) continue;

            size_t slot_hwm = slots[s].high_water_mark;
            if (slot_hwm >= aligned_size) {
                // Fits without growth - keep the smallest such slot.
                if (best_fit_slot == SIZE_MAX || slot_hwm < best_fit_hwm) {
                    best_fit_slot = s;
                    best_fit_hwm = slot_hwm;
                }
            } else {
                // Doesn't fit - track the largest as the growth fallback
                // (minimizes the growth amount).
                if (best_grow_slot == SIZE_MAX || slot_hwm > best_grow_hwm) {
                    best_grow_slot = s;
                    best_grow_hwm = slot_hwm;
                }
            }
        }

        // Always prefer a fitting slot; fall back to growth only if none fits.
        size_t best_slot = (best_fit_slot != SIZE_MAX) ? best_fit_slot
                                                        : best_grow_slot;

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
            new_slot.device = lr.device;
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

// ============================================================================
// Rematerialization Planner
// ============================================================================

auto RematerializationPlanner::is_cheap_to_recompute(OpType op) -> bool {
    switch (op) {
        // Activations
        case OpType::ReLU:
        case OpType::Sigmoid:
        case OpType::Tanh:
        case OpType::GELU:
        // Elementwise
        case OpType::Add:
        case OpType::Sub:
        case OpType::Mul:
        case OpType::Div:
        case OpType::Exp:
        case OpType::Log:
        case OpType::Sqrt:
        case OpType::Pow:
        case OpType::Abs:
        case OpType::Neg:
        case OpType::Clamp:
        // Shape ops (zero FLOPS, just view changes)
        case OpType::Reshape:
        case OpType::Transpose:
        case OpType::Permute:
        case OpType::Squeeze:
        case OpType::Unsqueeze:
            return true;
        default:
            return false;
    }
}

auto RematerializationPlanner::estimate_flops(OpType op,
                                               const std::vector<int64_t>& shape) -> double {
    if (shape.empty()) return 0.0;

    double numel = 1.0;
    for (int64_t dim : shape) {
        if (dim <= 0) return 0.0;
        numel *= static_cast<double>(dim);
    }

    switch (op) {
        // Shape ops: essentially free
        case OpType::Reshape:
        case OpType::Transpose:
        case OpType::Permute:
        case OpType::Squeeze:
        case OpType::Unsqueeze:
            return 1.0;  // Negligible

        // Simple elementwise: 1 FLOP per element
        case OpType::Add:
        case OpType::Sub:
        case OpType::Mul:
        case OpType::Div:
        case OpType::Abs:
        case OpType::Neg:
        case OpType::ReLU:
        case OpType::Clamp:
            return numel;

        // Transcendentals: ~10 FLOPS per element
        case OpType::Exp:
        case OpType::Log:
        case OpType::Sqrt:
        case OpType::Pow:
            return numel * 10.0;

        // Complex activations: ~20 FLOPS per element
        case OpType::Sigmoid:
        case OpType::Tanh:
        case OpType::GELU:
            return numel * 20.0;

        default:
            return numel * 100.0;  // Conservatively expensive
    }
}

auto RematerializationPlanner::find_candidates(
    const Graph& graph) -> std::vector<RematerializationCandidate> {

    std::vector<RematerializationCandidate> candidates;

    // Build node index for live range computation
    const auto& nodes = graph.nodes();
    std::unordered_map<const Node*, size_t> node_index;
    for (size_t i = 0; i < nodes.size(); ++i) {
        node_index[nodes[i].get()] = i;
    }

    // Collect excluded value IDs (inputs, outputs, constants)
    std::unordered_set<std::string> excluded;
    for (const auto& inp : graph.inputs()) excluded.insert(inp->id());
    for (const auto& out : graph.outputs()) excluded.insert(out->id());
    for (const auto& node : nodes) {
        if (node->op_type() == OpType::Constant) {
            for (const auto& out : node->outputs()) {
                excluded.insert(out->id());
            }
        }
    }

    for (const auto& node : nodes) {
        if (!is_cheap_to_recompute(node->op_type())) continue;

        for (const auto& output : node->outputs()) {
            if (excluded.count(output->id())) continue;

            // Compute live range
            size_t birth = node_index.count(node.get()) ? node_index[node.get()] : 0;
            size_t death = birth;
            for (const auto& use : output->uses()) {
                auto user = use.lock();
                if (user && node_index.count(user.get())) {
                    death = std::max(death, node_index[user.get()]);
                }
            }

            // Only consider values with meaningful live ranges
            size_t range_length = death - birth;
            if (range_length < 3) continue;

            // Compute memory cost
            const auto& shape = output->shape();
            if (shape.empty()) continue;
            size_t mem_bytes = checked_byte_size(shape, dtype_size(output->dtype()));
            if (mem_bytes == 0) continue;

            double flops = estimate_flops(node->op_type(), shape);
            if (flops <= 0.0) continue;

            double ratio = static_cast<double>(mem_bytes) / flops;
            if (ratio < min_cost_ratio_) continue;

            RematerializationCandidate candidate;
            candidate.value_id = output->id();
            candidate.producer_node_name = node->name();
            candidate.producer_op = node->op_type();
            candidate.memory_saved = mem_bytes;
            candidate.recompute_flops = flops;
            candidate.cost_ratio = ratio;
            candidates.push_back(std::move(candidate));
        }
    }

    // Sort by cost ratio descending (best candidates first)
    std::sort(candidates.begin(), candidates.end(),
        [](const RematerializationCandidate& a, const RematerializationCandidate& b) {
            return a.cost_ratio > b.cost_ratio;
        });

    return candidates;
}

auto RematerializationPlanner::apply(
    Graph& graph,
    const std::vector<RematerializationCandidate>& candidates) -> size_t {

    size_t count = 0;
    size_t memory_freed = 0;

    // Build the node->index map once. It is used only to locate the last
    // pre-existing CONSUMER of a value (drawn from value->uses()). The
    // recompute nodes we append below via graph.add_node are never looked up
    // as consumers, so the original (pre-loop) index map remains correct for
    // every candidate and rebuilding it per iteration would be wasted work.
    const auto& nodes = graph.nodes();
    std::unordered_map<const Node*, size_t> node_index;
    node_index.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        node_index[nodes[i].get()] = i;
    }

    for (const auto& candidate : candidates) {
        // If we have a memory budget, stop once we're under it
        if (memory_budget_ > 0 && memory_freed >= memory_budget_) break;

        auto value = graph.get_value(candidate.value_id);
        if (!value) continue;

        auto producer = value->node();
        if (!producer) continue;

        // Find the last consumer of this value
        size_t last_use_idx = 0;
        std::shared_ptr<Node> last_consumer;
        size_t live_consumer_count = 0;
        for (const auto& use : value->uses()) {
            auto user = use.lock();
            if (user && node_index.count(user.get())) {
                ++live_consumer_count;
                size_t idx = node_index[user.get()];
                if (idx >= last_use_idx) {
                    last_use_idx = idx;
                    last_consumer = user;
                }
            }
        }

        if (!last_consumer) continue;

        // Only rematerialize values with a single live consumer. We redirect
        // exactly ONE consumer (the last) to the recomputed output; for a
        // multi-consumer value the original still feeds the earlier consumers
        // and therefore stays live until its penultimate use — it is NOT freed,
        // so crediting candidate.memory_saved (the full value byte size) would
        // over-report the saving and let the budget loop stop early having
        // reclaimed far less than reported, while still paying the duplicate
        // recompute FLOPs/buffer. With a single consumer, redirecting it makes
        // the original immediately dead, so the full credit is accurate.
        if (live_consumer_count != 1) continue;

        // Insert a recompute node (duplicate of the producer) just before the
        // last consumer. The recompute node takes the same inputs and produces
        // a new value that replaces the original at the last use point.
        auto recompute_node = graph.create_node(producer->op_type(),
                                                  producer->name() + "_remat");

        // Copy inputs from the original producer
        for (const auto& inp : producer->inputs()) {
            recompute_node->add_input(inp);
        }

        // Copy attributes
        auto [attrs, int_attrs, vec_attrs, bool_attrs, tensor_attrs] = producer->get_all_attrs();
        for (const auto& [k, v] : attrs) recompute_node->set_attr(k, v);
        for (const auto& [k, v] : int_attrs) recompute_node->set_int_attr(k, v);
        for (const auto& [k, v] : vec_attrs) recompute_node->set_vec_attr(k, v);
        for (const auto& [k, v] : bool_attrs) recompute_node->set_bool_attr(k, v);
        for (const auto& [k, v] : tensor_attrs) recompute_node->set_tensor_attr(k, v);

        // Mark as rematerialized
        recompute_node->set_bool_attr("rematerialized", true);

        // Create output
        std::string remat_out_id = recompute_node->name() + "_out";
        auto remat_output = graph.create_value(
            remat_out_id, value->shape(), value->dtype(), value->device());
        remat_output->set_node(recompute_node);
        recompute_node->add_output(remat_output);

        graph.add_node(recompute_node);

        // Replace the original value at the last consumer only
        // (other consumers still use the original, which allows earlier freeing)
        for (size_t inp_idx = 0; inp_idx < last_consumer->inputs().size(); ++inp_idx) {
            if (last_consumer->inputs()[inp_idx]->id() == candidate.value_id) {
                last_consumer->replace_input(inp_idx, remat_output);
                break;
            }
        }

        memory_freed += candidate.memory_saved;
        ++count;
    }

    if (count > 0) {
        graph.topological_sort();
    }

    return count;
}

// ============================================================================
// Memory Swap Planner
// ============================================================================

auto MemorySwapPlanner::plan(const Graph& graph) -> std::vector<SwapSchedule> {
    std::vector<SwapSchedule> schedules;

    const auto& nodes = graph.nodes();
    std::unordered_map<const Node*, size_t> node_index;
    for (size_t i = 0; i < nodes.size(); ++i) {
        node_index[nodes[i].get()] = i;
    }

    // Collect excluded value IDs
    std::unordered_set<std::string> excluded;
    for (const auto& inp : graph.inputs()) excluded.insert(inp->id());
    for (const auto& out : graph.outputs()) excluded.insert(out->id());
    for (const auto& node : nodes) {
        if (node->op_type() == OpType::Constant) {
            for (const auto& out : node->outputs()) {
                excluded.insert(out->id());
            }
        }
    }

    for (const auto& node : nodes) {
        for (const auto& output : node->outputs()) {
            if (excluded.count(output->id())) continue;

            // Compute value size
            const auto& shape = output->shape();
            if (shape.empty()) continue;
            size_t size_bytes = checked_byte_size(shape, dtype_size(output->dtype()));
            if (size_bytes == 0) continue;  // invalid dim or overflow: skip
            if (size_bytes < swap_threshold_) continue;

            // Compute live range
            size_t birth = node_index.count(node.get()) ? node_index[node.get()] : 0;
            size_t death = birth;
            for (const auto& use : output->uses()) {
                auto user = use.lock();
                if (user && node_index.count(user.get())) {
                    death = std::max(death, node_index[user.get()]);
                }
            }

            size_t gap = death - birth;
            if (gap < min_gap_) continue;

            SwapSchedule sched;
            sched.value_id = output->id();
            sched.swap_out_after = birth;
            sched.swap_in_before = death;
            sched.size_bytes = size_bytes;
            schedules.push_back(std::move(sched));
        }
    }

    // Sort by size descending (swap largest first)
    std::sort(schedules.begin(), schedules.end(),
        [](const SwapSchedule& a, const SwapSchedule& b) {
            return a.size_bytes > b.size_bytes;
        });

    return schedules;
}

auto MemorySwapPlanner::apply(Graph& graph,
                               const std::vector<SwapSchedule>& schedules) -> size_t {
    size_t count = 0;

    // Build the node->index map once. It is used only to locate the last
    // pre-existing CONSUMER of a value (drawn from value->uses()). The
    // swap_out/swap_in nodes we append below via graph.add_node are never
    // looked up as consumers (and SwapOut uses are explicitly skipped), so the
    // original (pre-loop) index map stays correct for every schedule and
    // rebuilding it per iteration would be wasted work.
    const auto& nodes = graph.nodes();
    std::unordered_map<const Node*, size_t> node_index;
    node_index.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        node_index[nodes[i].get()] = i;
    }

    for (const auto& sched : schedules) {
        auto value = graph.get_value(sched.value_id);
        if (!value) continue;

        // Only swap values with exactly ONE live consumer (JIT-009). This planner
        // restores the value before its LAST consumer and redirects only that
        // consumer, so any intermediate consumer scheduled between SwapOut and
        // SwapIn would read the evicted (freed/stale) buffer. Mirror
        // RematerializationPlanner's single-consumer guard rather than corrupt a
        // multi-consumer value.
        size_t live_consumers = 0;
        for (const auto& use : value->uses()) {
            auto user = use.lock();
            if (user && user->op_type() != OpType::SwapOut) ++live_consumers;
        }
        if (live_consumers != 1) continue;

        // Create SwapOut node: placed after the producer
        auto swap_out = graph.create_node(OpType::SwapOut,
                                           "swap_out_" + sched.value_id);
        swap_out->add_input(value);
        swap_out->set_int_attr("size_bytes", static_cast<int64_t>(sched.size_bytes));

        // SwapOut produces a CPU-side handle
        std::string cpu_handle_id = swap_out->name() + "_cpu";
        auto cpu_handle = graph.create_value(
            cpu_handle_id, value->shape(), value->dtype(),
            Device::cpu());
        cpu_handle->set_node(swap_out);
        swap_out->add_output(cpu_handle);
        graph.add_node(swap_out);

        // Create SwapIn node: placed before the last consumer
        auto swap_in = graph.create_node(OpType::SwapIn,
                                          "swap_in_" + sched.value_id);
        swap_in->add_input(cpu_handle);
        swap_in->set_int_attr("size_bytes", static_cast<int64_t>(sched.size_bytes));

        // SwapIn produces the GPU-side tensor again
        std::string gpu_restored_id = swap_in->name() + "_gpu";
        auto gpu_restored = graph.create_value(
            gpu_restored_id, value->shape(), value->dtype(), value->device());
        gpu_restored->set_node(swap_in);
        swap_in->add_output(gpu_restored);
        graph.add_node(swap_in);

        // Redirect the last consumer to use the swapped-in value
        // Find the last consumer node (node_index built once above)
        size_t last_idx = 0;
        std::shared_ptr<Node> last_consumer;
        for (const auto& use : value->uses()) {
            auto user = use.lock();
            if (!user) continue;
            // Skip our own swap_out node
            if (user->op_type() == OpType::SwapOut) continue;
            if (node_index.count(user.get())) {
                size_t idx = node_index[user.get()];
                if (idx >= last_idx) {
                    last_idx = idx;
                    last_consumer = user;
                }
            }
        }

        if (last_consumer) {
            for (size_t inp_idx = 0; inp_idx < last_consumer->inputs().size(); ++inp_idx) {
                if (last_consumer->inputs()[inp_idx]->id() == sched.value_id) {
                    last_consumer->replace_input(inp_idx, gpu_restored);
                    break;
                }
            }
        }

        ++count;
    }

    if (count > 0) {
        graph.topological_sort();
    }

    return count;
}

} // namespace jit
} // namespace tenzor
