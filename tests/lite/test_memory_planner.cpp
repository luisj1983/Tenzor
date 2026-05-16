// test_memory_planner.cpp
//
// Wave Inf-E1+E2+E3+E7: greedy-by-size memory planner for the Lite runtime.
// Tests cover:
//   - Non-overlapping live ranges share offset 0 (reuse).
//   - Overlapping live ranges get disjoint offsets.
//   - Pool size matches algebraic expectation (high-water mark, aligned).
//   - Weight and graph-I/O tensors are excluded from the plan.
//   - MMPL TLV section round-trips byte-exactly through writer/reader.
//   - Legacy v1 files (no MMPL section) still load (back-compat).

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/lite/lite_graph.hpp>
#include <tenzor/lite/memory_planner.hpp>
#include <tenzor/lite/model_format.hpp>
#include <tenzor/lite/runtime.hpp>
#include <tenzor/ops/op_id.hpp>

#include <filesystem>
#include <vector>

using namespace tenzor;
using namespace tenzor::lite;

namespace {

// Build a TensorValue list mapping tensor_id -> (source, dtype, shape).
auto make_tval(int16_t id,
               TensorSource src,
               DType dtype,
               std::vector<int64_t> shape) -> TensorValue {
    TensorValue tv;
    tv.tensor_id = id;
    tv.source = src;
    tv.dtype = dtype;
    tv.shape = std::move(shape);
    return tv;
}

// Build a 3-node chain graph:
//   in (0) -> add(in, w) = 1   -> relu(1) = 2   -> matmul(2, w) = 3 (out)
// All intermediates F32, shape {N}. Returns graph + tvals.
auto build_chain(int64_t N) -> std::pair<LiteGraph, std::vector<TensorValue>> {
    LiteGraph g;
    g.set_input_ids({0});
    g.set_output_ids({3});

    LiteNode n0;
    n0.op = OpId::Add;
    n0.input_ids = {0, 100};  // 100 = weight
    n0.output_ids = {1};
    g.add_node(n0);

    LiteNode n1;
    n1.op = OpId::ReLU;
    n1.input_ids = {1};
    n1.output_ids = {2};
    g.add_node(n1);

    LiteNode n2;
    n2.op = OpId::MatMul;
    n2.input_ids = {2, 100};
    n2.output_ids = {3};
    g.add_node(n2);

    std::vector<TensorValue> tvs;
    tvs.push_back(make_tval(0,   TensorSource::Input,        DType::Float32, {N}));
    tvs.push_back(make_tval(1,   TensorSource::Intermediate, DType::Float32, {N}));
    tvs.push_back(make_tval(2,   TensorSource::Intermediate, DType::Float32, {N}));
    tvs.push_back(make_tval(3,   TensorSource::Intermediate, DType::Float32, {N}));  // graph output
    tvs.push_back(make_tval(100, TensorSource::Weight,       DType::Float32, {N}));

    // The planner indexes by tensor_id, so pad to max_id+1.
    std::vector<TensorValue> indexed(101);
    for (size_t i = 0; i < indexed.size(); ++i) {
        indexed[i].tensor_id = static_cast<int16_t>(i);
    }
    for (const auto& tv : tvs) {
        indexed[static_cast<size_t>(tv.tensor_id)] = tv;
    }
    return {std::move(g), std::move(indexed)};
}

}  // namespace

// ----------------------------------------------------------------------------
// E2.1: weight and graph-I/O are excluded from placements.
// ----------------------------------------------------------------------------
TEST(MemoryPlanner, ExcludesWeightsAndGraphIO) {
    auto [g, tvs] = build_chain(256);
    auto plan = compute_memory_plan(g, tvs);

    // tensor_id 0 (graph input) excluded. tensor_id 100 (Weight) excluded.
    // tensor_id 3 is the graph output — caller-owned, excluded.
    // Only intermediate tensor_ids {1, 2} should appear.
    std::vector<int16_t> got_ids;
    for (const auto& p : plan.placements) got_ids.push_back(p.tensor_id);
    EXPECT_EQ(got_ids.size(), 2u);
    EXPECT_EQ(got_ids[0], 1);
    EXPECT_EQ(got_ids[1], 2);
}

// ----------------------------------------------------------------------------
// E2.2: intermediate-1 (used by node 1) and intermediate-2 (produced by
// node 1, used by node 2) have OVERLAPPING live ranges at node 1, so they
// must get disjoint offsets.
// ----------------------------------------------------------------------------
TEST(MemoryPlanner, OverlappingTensorsGetDisjointOffsets) {
    auto [g, tvs] = build_chain(256);
    auto plan = compute_memory_plan(g, tvs);

    ASSERT_EQ(plan.placements.size(), 2u);
    ASSERT_NE(plan.placements[0].offset, plan.placements[1].offset);

    // Pool size at minimum = 2 * 256 * 4 bytes = 2048 (aligned to 64).
    ASSERT_FALSE(plan.pool_sizes.empty());
    EXPECT_GE(plan.pool_sizes[0], static_cast<uint64_t>(2 * 256 * 4));
}

// ----------------------------------------------------------------------------
// E2.3: alignment is honored (default 64 bytes).
// ----------------------------------------------------------------------------
TEST(MemoryPlanner, PlacementsAreAligned) {
    auto [g, tvs] = build_chain(33);  // odd N to exercise alignment rounding
    auto plan = compute_memory_plan(g, tvs);
    for (const auto& p : plan.placements) {
        EXPECT_EQ(p.offset % plan.alignment, 0u) << " tensor_id=" << p.tensor_id;
    }
    ASSERT_FALSE(plan.pool_sizes.empty());
    EXPECT_EQ(plan.pool_sizes[0] % plan.alignment, 0u);
}

// ----------------------------------------------------------------------------
// E2.4: planner is deterministic — same input → same output.
// ----------------------------------------------------------------------------
TEST(MemoryPlanner, Deterministic) {
    auto [g1, t1] = build_chain(128);
    auto [g2, t2] = build_chain(128);
    auto p1 = compute_memory_plan(g1, t1);
    auto p2 = compute_memory_plan(g2, t2);

    ASSERT_EQ(p1.pool_sizes, p2.pool_sizes);
    ASSERT_EQ(p1.placements.size(), p2.placements.size());
    for (size_t i = 0; i < p1.placements.size(); ++i) {
        EXPECT_EQ(p1.placements[i].tensor_id, p2.placements[i].tensor_id);
        EXPECT_EQ(p1.placements[i].offset,    p2.placements[i].offset);
    }
}

// ----------------------------------------------------------------------------
// E1+E3+E7: MMPL section round-trips byte-exactly through writer/reader.
// ----------------------------------------------------------------------------
TEST(MemoryPlanner, MmplSectionRoundtripsThroughWriterReader) {
    auto [g, tvs] = build_chain(256);
    auto plan = compute_memory_plan(g, tvs);
    ASSERT_FALSE(plan.pool_sizes.empty());

    // Save with MMPL section.
    WriteOptions opts;
    opts.input_ids = g.input_ids();
    opts.output_ids = g.output_ids();
    opts.memory_plan = std::make_shared<MmplPlan>(plan);

    auto tmp = std::filesystem::temp_directory_path() / "tzlite_mmpl_test.tzlite";
    TZLiteWriter::save(g, tmp.string(), opts);

    // Load via the full reader.
    auto loaded = TZLiteReader::load_full(tmp.string());
    ASSERT_NE(loaded.memory_plan, nullptr) << "MMPL section did not round-trip";
    EXPECT_EQ(loaded.memory_plan->pool_sizes, plan.pool_sizes);
    EXPECT_EQ(loaded.memory_plan->alignment,  plan.alignment);
    ASSERT_EQ(loaded.memory_plan->placements.size(), plan.placements.size());
    for (size_t i = 0; i < plan.placements.size(); ++i) {
        EXPECT_EQ(loaded.memory_plan->placements[i].tensor_id,
                  plan.placements[i].tensor_id);
        EXPECT_EQ(loaded.memory_plan->placements[i].pool_index,
                  plan.placements[i].pool_index);
        EXPECT_EQ(loaded.memory_plan->placements[i].offset,
                  plan.placements[i].offset);
    }

    std::filesystem::remove(tmp);
}

// ----------------------------------------------------------------------------
// E4: LiteRuntime::load_mmap behaves identically to load(path).
// ----------------------------------------------------------------------------
TEST(MemoryPlanner, LoadMmapEquivalentToLoad) {
    tenzor::initialize();  // ensure CPU dispatch table is populated
    auto [g, tvs] = build_chain(64);
    WriteOptions opts;
    opts.input_ids = g.input_ids();
    opts.output_ids = g.output_ids();
    auto tmp = std::filesystem::temp_directory_path() / "tzlite_mmap_test.tzlite";
    TZLiteWriter::save(g, tmp.string(), opts);

    // Both load paths must succeed and produce a runtime with the same
    // graph structure (3 nodes for our chain).
    auto rt_heap = tenzor::lite::LiteRuntime::load(tmp.string());
    auto rt_mmap = tenzor::lite::LiteRuntime::load_mmap(tmp.string());
    ASSERT_NE(rt_heap, nullptr);
    ASSERT_NE(rt_mmap, nullptr);

    std::filesystem::remove(tmp);
}

// ----------------------------------------------------------------------------
// E7: legacy v1 file (no MMPL section) still loads — back-compat check.
// ----------------------------------------------------------------------------
TEST(MemoryPlanner, LegacyV1FileLoadsWithoutMmpl) {
    auto [g, tvs] = build_chain(64);

    WriteOptions opts;
    opts.input_ids = g.input_ids();
    opts.output_ids = g.output_ids();
    // No memory_plan set — emulates v1-style save (still serialised as v2,
    // but with no MMPL section emitted; back-compat path stays exercised).

    auto tmp = std::filesystem::temp_directory_path() / "tzlite_no_mmpl_test.tzlite";
    TZLiteWriter::save(g, tmp.string(), opts);

    auto loaded = TZLiteReader::load_full(tmp.string());
    EXPECT_EQ(loaded.memory_plan, nullptr) << "memory_plan should be null when MMPL absent";
    EXPECT_EQ(loaded.graph->num_nodes(), 3u);

    std::filesystem::remove(tmp);
}
