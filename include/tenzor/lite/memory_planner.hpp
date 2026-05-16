/**
 * @file memory_planner.hpp
 * @brief Phase 5 memory-pool planner for the Lite runtime.
 *
 * Given a topologically-ordered LiteGraph + TVAL table (TensorValue per
 * tensor_id), the planner computes:
 *   - one or more byte-pool sizes,
 *   - per-tensor (pool_index, byte_offset) placements,
 * such that all live tensors fit within their pool and no two live
 * tensors overlap. Weight tensors and graph inputs/outputs are EXCLUDED
 * — those are caller-owned (graph I/O) or WGTS-backed views (weights).
 *
 * Algorithm: greedy-by-size offset allocation (TFLite MicroAllocator
 * pattern). For each intermediate, sorted descending by byte size:
 *   1. Determine live range [first_use, last_use] in node-order.
 *   2. Find the lowest pool offset where placing this tensor wouldn't
 *      overlap any previously-placed tensor whose live range intersects.
 *   3. Place; extend the pool's size if needed (rounded up to alignment).
 *
 * Single pool by default — multi-pool is supported by the wire format
 * (TZLITE_TAG_MMPL) but the v2 planner emits a single pool. Multi-pool
 * lands later when a heterogeneous-device backend needs distinct arenas.
 */

#pragma once

#include "lite_graph.hpp"
#include "model_format.hpp"

#include <cstdint>
#include <vector>

namespace tenzor::lite {

/** Per-tensor placement: (pool_index, byte_offset_within_pool). */
struct MmplPlacement {
    int16_t tensor_id{-1};
    uint8_t pool_index{0};
    uint64_t offset{0};
};

/** Output of `compute_memory_plan`. */
struct MmplPlan {
    /** Total byte size of each pool. Pool i is `pool_sizes[i]` bytes. */
    std::vector<uint64_t> pool_sizes;
    /** Alignment in bytes, same for all pools (64 by default). */
    uint64_t alignment{64};
    /** Placement for every Intermediate tensor_id; ordered by tensor_id. */
    std::vector<MmplPlacement> placements;
};

/** Compute a memory plan from a finalised LiteGraph + TVAL table.
 *
 * @param graph     Graph with input_ids() and output_ids() already set.
 * @param tvs       TensorValue table — typically the same one written
 *                  to TZLITE_TAG_TVAL by the writer.
 * @param alignment Byte alignment for each placement (default 64).
 *
 * @return MmplPlan with a single pool sized to the high-water mark.
 *
 * Tensors NOT included in `placements`:
 *   - Weight: stored in WGTS, not the runtime arena.
 *   - Input/output: caller-owned buffers.
 *
 * The planner is deterministic: same graph → same plan, byte-exactly.
 */
auto compute_memory_plan(const LiteGraph& graph,
                         const std::vector<TensorValue>& tvs,
                         uint64_t alignment = 64) -> MmplPlan;

}  // namespace tenzor::lite
