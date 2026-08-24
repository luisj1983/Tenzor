/**
 * @file memory_planner.cpp
 * @brief Greedy-by-size offset allocator for Lite runtime tensor arenas.
 */

#include "tenzor/lite/memory_planner.hpp"
#include "tenzor/utils/safe_math.hpp"

#include "tenzor/core/dtype.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>

namespace tenzor::lite {

namespace {

// Round `n` up to the next multiple of `align`. Align must be > 0.
inline auto round_up(uint64_t n, uint64_t align) -> uint64_t {
    return ((n + align - 1) / align) * align;
}

// Per-tensor byte size from its declared shape × dtype size. Returns 0 for
// scalar/unset tensors (those don't need a pool slot). Returns UINT64_MAX as a
// saturating sentinel when the element count or byte size overflows uint64, so
// an overflowing tensor is never assigned a bogus small slot (mirrors the
// overflow-checked sizing in runtime.cpp's MMPL load validator).
auto bytes_of(const TensorValue& tv) -> uint64_t {
    if (tv.shape.empty()) return 0;
    uint64_t n = 1;
    for (int64_t d : tv.shape) {
        if (d <= 0) return 0;  // dynamic dim — defer to runtime alloc
        uint64_t prod = 0;
        if (tenzor::detail::checked_mul_overflow(n, static_cast<uint64_t>(d), &prod)) {
            return std::numeric_limits<uint64_t>::max();
        }
        n = prod;
    }
    uint64_t bytes = 0;
    if (tenzor::detail::checked_mul_overflow(n, static_cast<uint64_t>(dtype_size(tv.dtype)),
                               &bytes)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return bytes;
}

// A working record for the planner.
struct LiveTensor {
    int16_t tensor_id;
    uint64_t bytes;
    int first_use;       // node index of first appearance (as input or output)
    int last_use;        // node index of last appearance (as input)
    uint64_t offset{0};  // assigned offset in pool
};

// Build the liveness table for every Intermediate tensor_id referenced in
// the graph. Weight / Input / Output tensors get NULL liveness (excluded).
auto compute_liveness(const LiteGraph& graph,
                      const std::vector<TensorValue>& tvs)
    -> std::vector<LiveTensor> {
    // Quick lookup of source by tensor_id.
    auto source_of = [&](int16_t id) -> TensorSource {
        if (id < 0 || static_cast<size_t>(id) >= tvs.size()) {
            return TensorSource::Intermediate;
        }
        return tvs[static_cast<size_t>(id)].source;
    };
    std::unordered_set<int16_t> graph_io(graph.input_ids().begin(),
                                         graph.input_ids().end());
    for (int16_t id : graph.output_ids()) graph_io.insert(id);

    std::vector<LiveTensor> out;
    std::unordered_set<int16_t> seen;

    const auto& nodes = graph.nodes();
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        const auto& node = nodes[i];
        // Outputs come first chronologically (writes), then inputs (reads).
        auto observe = [&](int16_t id) {
            if (graph_io.count(id)) return;            // caller-owned
            if (source_of(id) == TensorSource::Weight) return;  // WGTS-backed
            auto it = std::find_if(out.begin(), out.end(),
                                   [&](const LiveTensor& lt) {
                                       return lt.tensor_id == id;
                                   });
            if (it == out.end()) {
                LiveTensor lt;
                lt.tensor_id = id;
                lt.bytes = (static_cast<size_t>(id) < tvs.size())
                               ? bytes_of(tvs[static_cast<size_t>(id)])
                               : 0;
                lt.first_use = i;
                lt.last_use = i;
                out.push_back(lt);
                seen.insert(id);
            } else {
                it->last_use = i;
            }
        };
        for (int16_t id : node.output_ids) observe(id);
        for (int16_t id : node.input_ids)  observe(id);
    }
    return out;
}

}  // namespace

auto compute_memory_plan(const LiteGraph& graph,
                         const std::vector<TensorValue>& tvs,
                         uint64_t alignment) -> MmplPlan {
    if (alignment == 0) alignment = 64;

    auto live = compute_liveness(graph, tvs);

    // Strip zero-byte tensors (dynamic-dim or scalar) — those allocate at
    // runtime via the kernel, not from the pool.
    live.erase(std::remove_if(live.begin(), live.end(),
                              [](const LiveTensor& lt) { return lt.bytes == 0; }),
               live.end());

    // Sort descending by byte size; ties broken by tensor_id for determinism.
    std::sort(live.begin(), live.end(),
              [](const LiveTensor& a, const LiveTensor& b) {
                  if (a.bytes != b.bytes) return a.bytes > b.bytes;
                  return a.tensor_id < b.tensor_id;
              });

    // Greedy placement: for each tensor in size order, find the lowest
    // offset where its live range doesn't overlap any already-placed
    // tensor's live range.
    struct Placed {
        uint64_t offset;
        uint64_t end;       // offset + aligned-size
        int first_use;
        int last_use;
    };
    std::vector<Placed> placed;
    placed.reserve(live.size());

    for (auto& lt : live) {
        uint64_t need = round_up(lt.bytes, alignment);
        // Candidate offsets to try, sorted ascending. Start at 0; whenever
        // we hit a conflict at offset `o`, try o = round_up(conflict.end, align).
        uint64_t candidate = 0;
        bool placed_ok = false;
        while (!placed_ok) {
            placed_ok = true;
            for (const auto& p : placed) {
                bool live_overlaps =
                    !(lt.last_use < p.first_use || p.last_use < lt.first_use);
                if (!live_overlaps) continue;
                bool offset_overlaps =
                    !(candidate + need <= p.offset || p.end <= candidate);
                if (offset_overlaps) {
                    candidate = round_up(p.end, alignment);
                    placed_ok = false;
                    break;
                }
            }
        }
        lt.offset = candidate;
        placed.push_back({candidate, candidate + need, lt.first_use, lt.last_use});
    }

    // High-water mark is the pool size.
    uint64_t pool_size = 0;
    for (const auto& p : placed) pool_size = std::max(pool_size, p.end);

    MmplPlan plan;
    plan.alignment = alignment;
    if (pool_size > 0) plan.pool_sizes.push_back(pool_size);

    // Emit placements ordered by tensor_id for determinism.
    plan.placements.reserve(live.size());
    for (const auto& lt : live) {
        MmplPlacement mp;
        mp.tensor_id = lt.tensor_id;
        mp.pool_index = 0;
        mp.offset = lt.offset;
        plan.placements.push_back(mp);
    }
    std::sort(plan.placements.begin(), plan.placements.end(),
              [](const MmplPlacement& a, const MmplPlacement& b) {
                  return a.tensor_id < b.tensor_id;
              });

    return plan;
}

}  // namespace tenzor::lite
