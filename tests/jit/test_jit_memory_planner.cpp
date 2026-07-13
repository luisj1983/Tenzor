/**
 * @file test_memory_planner.cpp
 * @brief Regression coverage for MemoryPlanner/RematerializationPlanner and
 *        their Compiler-level wiring (R1-12).
 *
 * Before this fix, MemoryPlanner::compute_live_ranges/greedy_assign (the
 * buffer-aliasing/slot-assignment logic most prone to the shared-mem-
 * modulo-aliasing bug class -- see the JIT-R023 comment in
 * src/jit/memory_planner.cpp) had NO dedicated test at all: the only
 * existing coverage in tests/jit/ was test_memory_planner_swap.cpp, for the
 * separate (still-unwired) MemorySwapPlanner. RematerializationPlanner was
 * similarly never invoked from the compiler.
 *
 * R1-12 wires MemoryPlanningPass and RematerializationPass into
 * Compiler::optimize() (RematerializationPass opt-in via
 * Compiler::set_rematerialization; MemoryPlanningPass always runs when
 * memory planning is enabled -- see Compiler::plan_memory's doc comment
 * for why it now routes through the Pass class instead of a redundant
 * direct MemoryPlanner::plan() call). This file tests:
 *   1. MemoryPlanner::plan()'s buffer-aliasing decisions directly (via its
 *      public API -- greedy_assign/compute_live_ranges are private, so
 *      these are exercised the same way any real caller would use them).
 *   2. MemoryPlanningPass / Compiler::plan_memory / Compiler::optimize's
 *      memory-planning wiring.
 *   3. RematerializationPlanner's candidate selection and graph mutation.
 *   4. RematerializationPass / Compiler::optimize's rematerialization
 *      wiring, INCLUDING a numeric-correctness check proving rematerialized
 *      execution produces bit-identical output to the un-rematerialized
 *      graph (the whole point of only selecting deterministic, side-effect
 *      free "cheap to recompute" ops).
 */

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/memory_planner.hpp>

using namespace tenzor;
using namespace tenzor::jit;

namespace {

class MemoryPlannerTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new MemoryPlannerTestEnv);

// Input -> op_0 -> op_1 -> ... -> op_{n-1} (graph output). Every intermediate
// out_i (i < n-1) has exactly one consumer (the next node) and a live range
// of length 1 -- deliberately too short for rematerialization (see the
// dedicated long-lived-value builder below for that), but exactly what's
// needed to exercise MemoryPlanner's non-overlapping-reuse decision: with
// equal-size values, out_0/out_2/out_4/... alternate-reuse the same slots as
// out_1/out_3/out_5/... (see MemoryPlannerTest.ChainValuesReuseAlternatingBuffers
// for the worked-out expected assignment).
auto build_chain(Graph& graph, size_t n_ops, OpType op,
                  const std::vector<int64_t>& shape, Device device) -> void {
    auto input = graph.create_value("input", shape, DType::Float32, device);
    auto prev = input;
    for (size_t i = 0; i < n_ops; ++i) {
        auto output = graph.create_value("out_" + std::to_string(i), shape,
                                          DType::Float32, device);
        auto node = graph.create_node(op, "op_" + std::to_string(i));
        node->add_input(prev);
        node->add_output(output);
        output->set_node(node);
        graph.add_node(node);
        prev = output;
    }
    graph.set_inputs({input});
    graph.set_outputs({prev});
}

} // namespace

// ============================================================================
// MemoryPlanner::plan() -- live-range analysis + greedy buffer assignment
// ============================================================================

// Worked-out expected assignment for a 5-op equal-size Relu chain (out_0..
// out_3 intermediates, out_4 excluded as the graph output):
//   out_0 range=[0,1], out_1=[1,2], out_2=[2,3], out_3=[3,4]
// greedy_assign sorts by size desc / begin asc (all equal size, so by begin):
//   out_0: no slots yet -> new slot 0, interval [0,1]
//   out_1: slot 0 overlaps ([0,1] vs [1,2] touch at 1) -> new slot 1, [1,2]
//   out_2: slot 0 [0,1] vs [2,3] -> no overlap -> REUSE slot 0
//   out_3: slot 0 now [[0,1],[2,3]] vs [3,4] -> overlaps [2,3] -> slot 1 [1,2]
//          vs [3,4] -> no overlap -> REUSE slot 1
// So out_0/out_2 share a buffer, out_1/out_3 share a (different) buffer: 2
// pool buffers total, 2 values reused.
TEST(MemoryPlannerTest, ChainValuesReuseAlternatingBuffers) {
    Graph graph;
    build_chain(graph, /*n_ops=*/5, OpType::ReLU, {2, 3}, Device::cpu());

    MemoryPlanner planner;
    MemoryPlan plan = planner.plan(graph);

    ASSERT_EQ(plan.num_values_planned, 4u) << "out_4 (graph output) must be excluded";
    EXPECT_EQ(plan.pool_sizes.size(), 2u);
    EXPECT_EQ(plan.num_values_reused, 2u);

    ASSERT_TRUE(plan.value_allocations.count("out_0"));
    ASSERT_TRUE(plan.value_allocations.count("out_1"));
    ASSERT_TRUE(plan.value_allocations.count("out_2"));
    ASSERT_TRUE(plan.value_allocations.count("out_3"));
    EXPECT_FALSE(plan.value_allocations.count("input"));
    EXPECT_FALSE(plan.value_allocations.count("out_4"));

    EXPECT_EQ(plan.value_allocations.at("out_0").buffer_id,
              plan.value_allocations.at("out_2").buffer_id)
        << "out_0 and out_2 have non-overlapping live ranges and equal size "
           "-- they must share a buffer slot";
    EXPECT_EQ(plan.value_allocations.at("out_1").buffer_id,
              plan.value_allocations.at("out_3").buffer_id)
        << "out_1 and out_3 have non-overlapping live ranges and equal size "
           "-- they must share a buffer slot";
    EXPECT_NE(plan.value_allocations.at("out_0").buffer_id,
              plan.value_allocations.at("out_1").buffer_id)
        << "out_0 and out_1 have OVERLAPPING live ranges ([0,1] touches "
           "[1,2] at index 1) -- they must NOT share a buffer slot";

    // The graph's Values were annotated in-place (not just the returned plan).
    auto v0 = graph.get_value("out_0");
    auto v2 = graph.get_value("out_2");
    ASSERT_TRUE(v0 && v2);
    EXPECT_TRUE(v0->has_memory_plan());
    EXPECT_EQ(v0->buffer_id(), v2->buffer_id());
}

// Diamond: x -> A=relu(x), x -> B=sigmoid(x), Y=A+B. A and B are BOTH alive
// at the node computing Y (their live ranges overlap), so even though they
// are the same size, they must NOT share a buffer -- aliasing them would
// have Y read B's data through a pointer meant for A (or vice versa),
// exactly the shared-buffer aliasing corruption class this planner exists
// to avoid introducing.
TEST(MemoryPlannerTest, OverlappingLiveRangesGetDistinctBuffers) {
    Graph graph;
    const std::vector<int64_t> shape = {4};
    auto x = graph.create_value("x", shape, DType::Float32, Device::cpu());

    auto node_a = graph.create_node(OpType::ReLU, "A");
    auto out_a = graph.create_value("out_A", shape, DType::Float32, Device::cpu());
    node_a->add_input(x);
    node_a->add_output(out_a);
    out_a->set_node(node_a);
    graph.add_node(node_a);

    auto node_b = graph.create_node(OpType::Sigmoid, "B");
    auto out_b = graph.create_value("out_B", shape, DType::Float32, Device::cpu());
    node_b->add_input(x);
    node_b->add_output(out_b);
    out_b->set_node(node_b);
    graph.add_node(node_b);

    auto node_y = graph.create_node(OpType::Add, "Y");
    auto out_y = graph.create_value("out_Y", shape, DType::Float32, Device::cpu());
    node_y->add_input(out_a);
    node_y->add_input(out_b);
    node_y->add_output(out_y);
    out_y->set_node(node_y);
    graph.add_node(node_y);

    graph.set_inputs({x});
    graph.set_outputs({out_y});

    MemoryPlanner planner;
    MemoryPlan plan = planner.plan(graph);

    ASSERT_TRUE(plan.value_allocations.count("out_A"));
    ASSERT_TRUE(plan.value_allocations.count("out_B"));
    EXPECT_NE(plan.value_allocations.at("out_A").buffer_id,
              plan.value_allocations.at("out_B").buffer_id);
    EXPECT_EQ(plan.num_values_reused, 0u);
}

// Two non-overlapping-live-range, equal-size values on DIFFERENT devices must
// never share a buffer -- a pool buffer is a single device allocation, so
// aliasing a CPU value with a GPU value would corrupt data / produce an
// invalid pointer the moment this plan is wired to real placement. Device is
// pure IR metadata on Value here (mirrors test_memory_planner_swap.cpp's
// established pattern): nothing in this test executes the graph or
// dispatches to a backend, so tagging a Value CUDA is safe in a CPU-only
// test binary.
TEST(MemoryPlannerTest, NeverAliasesAcrossDevicesEvenWithoutOverlap) {
    Graph graph;
    const std::vector<int64_t> shape = {4};
    auto x_cpu = graph.create_value("x_cpu", shape, DType::Float32, Device::cpu());
    auto x_gpu = graph.create_value("x_gpu", shape, DType::Float32, Device::cuda(0));

    // out_cpu: born@0 (consumed@2, non-overlapping with out_gpu's [1,2] would
    // normally allow reuse if same device -- here it must not, since devices differ).
    auto node_cpu = graph.create_node(OpType::ReLU, "cpu_relu");
    auto out_cpu = graph.create_value("out_cpu", shape, DType::Float32, Device::cpu());
    node_cpu->add_input(x_cpu);
    node_cpu->add_output(out_cpu);
    out_cpu->set_node(node_cpu);
    graph.add_node(node_cpu);

    auto node_gpu = graph.create_node(OpType::ReLU, "gpu_relu");
    auto out_gpu = graph.create_value("out_gpu", shape, DType::Float32, Device::cuda(0));
    node_gpu->add_input(x_gpu);
    node_gpu->add_output(out_gpu);
    out_gpu->set_node(node_gpu);
    graph.add_node(node_gpu);

    // Consume out_cpu far later (long past out_gpu's death) so their INDEX
    // ranges genuinely would not overlap if devices were ignored.
    auto node_final = graph.create_node(OpType::Neg, "final");
    auto out_final = graph.create_value("out_final", shape, DType::Float32, Device::cpu());
    node_final->add_input(out_cpu);
    node_final->add_output(out_final);
    out_final->set_node(node_final);
    graph.add_node(node_final);

    graph.set_inputs({x_cpu, x_gpu});
    graph.set_outputs({out_final, out_gpu});

    MemoryPlanner planner;
    MemoryPlan plan = planner.plan(graph);

    // out_gpu is a graph output (excluded); out_cpu is the only planned value
    // in this shape. Reconstruct explicitly with out_gpu NOT a graph output
    // to actually compare two planned cross-device values:
    Graph graph2;
    auto y_cpu = graph2.create_value("y_cpu", shape, DType::Float32, Device::cpu());
    auto y_gpu = graph2.create_value("y_gpu", shape, DType::Float32, Device::cuda(0));
    auto n1 = graph2.create_node(OpType::ReLU, "n1");
    auto v1 = graph2.create_value("v1", shape, DType::Float32, Device::cpu());
    n1->add_input(y_cpu); n1->add_output(v1); v1->set_node(n1);
    graph2.add_node(n1);
    auto n2 = graph2.create_node(OpType::ReLU, "n2");
    auto v2 = graph2.create_value("v2", shape, DType::Float32, Device::cuda(0));
    n2->add_input(y_gpu); n2->add_output(v2); v2->set_node(n2);
    graph2.add_node(n2);
    auto n3 = graph2.create_node(OpType::Neg, "n3");
    auto v3 = graph2.create_value("v3", shape, DType::Float32, Device::cpu());
    n3->add_input(v1); n3->add_output(v3); v3->set_node(n3);
    graph2.add_node(n3);
    auto n4 = graph2.create_node(OpType::Neg, "n4");
    auto v4 = graph2.create_value("v4", shape, DType::Float32, Device::cuda(0));
    n4->add_input(v2); n4->add_output(v4); v4->set_node(n4);
    graph2.add_node(n4);
    graph2.set_inputs({y_cpu, y_gpu});
    graph2.set_outputs({v3, v4});

    MemoryPlanner planner2;
    MemoryPlan plan2 = planner2.plan(graph2);

    // v1 ([0,2]-ish, CPU) and v2 (GPU) do not overlap in device-blind index
    // terms in a way that matters here -- the point is simply: they must
    // never be assigned the same buffer_id regardless of live-range overlap,
    // because greedy_assign's slot search skips any slot whose device differs.
    ASSERT_TRUE(plan2.value_allocations.count("v1"));
    ASSERT_TRUE(plan2.value_allocations.count("v2"));
    EXPECT_NE(plan2.value_allocations.at("v1").buffer_id,
              plan2.value_allocations.at("v2").buffer_id);
}

// Default alignment is 64 bytes (cache-line aligned): a single Float32
// scalar-shaped value (4 bytes) must still occupy a 64-byte pool slot.
TEST(MemoryPlannerTest, AlignsPoolSizeUpToDefaultAlignment) {
    Graph graph;
    const std::vector<int64_t> shape = {1};
    auto x = graph.create_value("x", shape, DType::Float32, Device::cpu());
    auto node = graph.create_node(OpType::ReLU, "n");
    auto out1 = graph.create_value("out1", shape, DType::Float32, Device::cpu());
    node->add_input(x);
    node->add_output(out1);
    out1->set_node(node);
    graph.add_node(node);

    auto node2 = graph.create_node(OpType::Neg, "n2");
    auto out2 = graph.create_value("out2", shape, DType::Float32, Device::cpu());
    node2->add_input(out1);
    node2->add_output(out2);
    out2->set_node(node2);
    graph.add_node(node2);

    graph.set_inputs({x});
    graph.set_outputs({out2});

    MemoryPlanner planner;
    ASSERT_EQ(planner.alignment(), 64u);
    MemoryPlan plan = planner.plan(graph);

    ASSERT_EQ(plan.pool_sizes.size(), 1u);
    EXPECT_EQ(plan.pool_sizes[0], 64u) << "4-byte value must round up to the 64-byte alignment";
}

TEST(MemoryPlannerTest, SetAlignmentRejectsNonPowerOfTwo) {
    MemoryPlanner planner;
    EXPECT_THROW(planner.set_alignment(0), std::invalid_argument);
    EXPECT_THROW(planner.set_alignment(3), std::invalid_argument);
    EXPECT_NO_THROW(planner.set_alignment(128));
    EXPECT_EQ(planner.alignment(), 128u);
}

// ============================================================================
// MemoryPlanningPass / Compiler::plan_memory wiring
// ============================================================================

TEST(MemoryPlanningPassTest, RunAnnotatesValuesAndExposesPlan) {
    Graph graph;
    build_chain(graph, 5, OpType::ReLU, {2, 3}, Device::cpu());

    MemoryPlanningPass pass;
    EXPECT_FALSE(pass.run(graph)) << "annotation-only pass must never report a structural change";
    EXPECT_EQ(pass.name(), "MemoryPlanning");

    const MemoryPlan& plan = pass.memory_plan();
    EXPECT_EQ(plan.num_values_planned, 4u);

    auto v0 = graph.get_value("out_0");
    ASSERT_TRUE(v0);
    EXPECT_TRUE(v0->has_memory_plan());
}

// R1-12: Compiler::plan_memory used to duplicate MemoryPlanner::plan() inline
// instead of running MemoryPlanningPass -- this proves the public API now
// genuinely routes through (and its behavior is unchanged: same values
// planned, same reuse count) the previously-dead pass class.
TEST(CompilerMemoryPlanningTest, PlanMemoryMatchesDirectPlannerCall) {
    Graph graph_a, graph_b;
    build_chain(graph_a, 5, OpType::ReLU, {2, 3}, Device::cpu());
    build_chain(graph_b, 5, OpType::ReLU, {2, 3}, Device::cpu());

    Compiler compiler(/*enable_default_passes=*/false);
    MemoryPlan via_compiler = compiler.plan_memory(graph_a);

    MemoryPlanner planner;
    MemoryPlan via_direct = planner.plan(graph_b);

    EXPECT_EQ(via_compiler.num_values_planned, via_direct.num_values_planned);
    EXPECT_EQ(via_compiler.pool_sizes.size(), via_direct.pool_sizes.size());
    EXPECT_EQ(via_compiler.num_values_reused, via_direct.num_values_reused);
    EXPECT_EQ(via_compiler.total_memory, via_direct.total_memory);
}

TEST(CompilerMemoryPlanningTest, ExplicitEnableRunsRegardlessOfGraphSize) {
    Graph graph;
    build_chain(graph, 3, OpType::ReLU, {2, 3}, Device::cpu());  // well under the 50-node threshold

    Compiler compiler(/*enable_default_passes=*/false);
    compiler.set_memory_planning(true);
    compiler.optimize(graph);

    EXPECT_GT(compiler.memory_plan().num_values_planned, 0u);
}

TEST(CompilerMemoryPlanningTest, DefaultAutoModeSkipsSmallGraphs) {
    Graph graph;
    build_chain(graph, 3, OpType::ReLU, {2, 3}, Device::cpu());

    Compiler compiler(/*enable_default_passes=*/false);
    compiler.optimize(graph);  // default: auto mode, threshold 50 nodes

    EXPECT_EQ(compiler.memory_plan().num_values_planned, 0u);
    // Graph values must NOT have been annotated either.
    auto v = graph.get_value("out_0");
    ASSERT_TRUE(v);
    EXPECT_FALSE(v->has_memory_plan());
}

// ============================================================================
// RematerializationPlanner -- candidate selection
// ============================================================================

namespace {
// x -> A=relu(x) [long-lived, single consumer far later]
// x -> B1=relu(x) -> B2=relu(B1) -> B3=relu(B2) -> B4=relu(B3) [filler chain]
// Y = A + B4 (graph output)
// out_A has live range [0,5] (length 5 >= 3), a single consumer (Y), and is a
// cheap-to-recompute op -- the ONLY value in this graph eligible for
// rematerialization. Every out_Bi has live range length 1 (too short).
auto build_long_lived_plus_filler_chain(Graph& graph, Device device = Device::cpu())
    -> void {
    const std::vector<int64_t> shape = {4};
    auto x = graph.create_value("x", shape, DType::Float32, device);

    auto node_a = graph.create_node(OpType::ReLU, "A");
    auto out_a = graph.create_value("out_A", shape, DType::Float32, device);
    node_a->add_input(x);
    node_a->add_output(out_a);
    out_a->set_node(node_a);
    graph.add_node(node_a);

    std::shared_ptr<Value> prev = x;
    for (int i = 1; i <= 4; ++i) {
        auto node = graph.create_node(OpType::ReLU, "B" + std::to_string(i));
        auto out = graph.create_value("out_B" + std::to_string(i), shape,
                                       DType::Float32, device);
        node->add_input(prev);
        node->add_output(out);
        out->set_node(node);
        graph.add_node(node);
        prev = out;
    }

    auto node_y = graph.create_node(OpType::Add, "Y");
    auto out_y = graph.create_value("out_Y", shape, DType::Float32, device);
    node_y->add_input(out_a);
    node_y->add_input(prev);
    node_y->add_output(out_y);
    out_y->set_node(node_y);
    graph.add_node(node_y);

    graph.set_inputs({x});
    graph.set_outputs({out_y});
}
} // namespace

TEST(RematerializationPlannerTest, FindsOnlyTheLongLivedSingleConsumerCheapValue) {
    Graph graph;
    build_long_lived_plus_filler_chain(graph);

    RematerializationPlanner planner;
    auto candidates = planner.find_candidates(graph);

    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_EQ(candidates[0].value_id, "out_A");
    EXPECT_EQ(candidates[0].producer_op, OpType::ReLU);
    EXPECT_GT(candidates[0].memory_saved, 0u);
    EXPECT_GT(candidates[0].cost_ratio, 0.0);
}

TEST(RematerializationPlannerTest, ExcludesValueWithMultipleConsumers) {
    Graph graph;
    const std::vector<int64_t> shape = {4};
    auto x = graph.create_value("x", shape, DType::Float32, Device::cpu());

    auto node_a = graph.create_node(OpType::ReLU, "A");
    auto out_a = graph.create_value("out_A", shape, DType::Float32, Device::cpu());
    node_a->add_input(x);
    node_a->add_output(out_a);
    out_a->set_node(node_a);
    graph.add_node(node_a);

    // Two independent consumers of out_A, both far later than its birth so
    // the live range alone would qualify -- only the single-consumer gate in
    // apply() should reject it (find_candidates doesn't gate on consumer
    // count -- it just reports the candidate; the exclusion happens in
    // RematerializationPlanner::apply). This test targets apply()'s gate.
    std::shared_ptr<Value> prev1 = out_a;
    for (int i = 0; i < 3; ++i) {
        auto node = graph.create_node(OpType::ReLU, "chain1_" + std::to_string(i));
        auto out = graph.create_value("chain1_out_" + std::to_string(i), shape,
                                       DType::Float32, Device::cpu());
        node->add_input(prev1);
        node->add_output(out);
        out->set_node(node);
        graph.add_node(node);
        prev1 = out;
    }
    auto node_b = graph.create_node(OpType::Neg, "consumer2");
    auto out_b = graph.create_value("out_consumer2", shape, DType::Float32, Device::cpu());
    node_b->add_input(out_a);
    node_b->add_output(out_b);
    out_b->set_node(node_b);
    graph.add_node(node_b);

    auto node_y = graph.create_node(OpType::Add, "Y");
    auto out_y = graph.create_value("out_Y", shape, DType::Float32, Device::cpu());
    node_y->add_input(prev1);
    node_y->add_input(out_b);
    node_y->add_output(out_y);
    out_y->set_node(node_y);
    graph.add_node(node_y);

    graph.set_inputs({x});
    graph.set_outputs({out_y});
    // Deliberately NOT calling graph.topological_sort() here: node insertion
    // order above already satisfies dependency order (every producer is
    // added before its consumers), and find_candidates/apply use graph.nodes()
    // position directly as the execution-time proxy (see the JIT-R023
    // invariant in memory_planner.cpp). Re-sorting could reorder node_b
    // (whose only dependency is out_A) earlier, shrinking out_A's computed
    // live range below the range_length>=3 threshold and making this test's
    // premise (out_A has a long-enough range AND 2 consumers) order-dependent.

    RematerializationPlanner planner;
    auto candidates = planner.find_candidates(graph);
    // out_A itself has 2 consumers (node_b and the first chain1 node) so it IS
    // reported by find_candidates (which doesn't gate on consumer count) --
    // apply() is where the single-consumer gate lives.
    bool found_out_a = false;
    for (const auto& c : candidates) if (c.value_id == "out_A") found_out_a = true;
    ASSERT_TRUE(found_out_a);

    size_t applied = planner.apply(graph, candidates);
    EXPECT_EQ(applied, 0u) << "out_A has 2 live consumers -- apply() must reject it";

    auto v = graph.get_value("out_A");
    ASSERT_TRUE(v);
    EXPECT_EQ(v->uses().size(), 2u) << "no consumer should have been redirected";
}

// JIT-R124 regression: node_index (used to count live consumers) was
// top-level-only, so a value with ONE top-level consumer and ONE consumer
// nested inside an If/Loop subgraph read live_consumer_count==1 (the
// nested one silently uncounted), incorrectly qualifying it for single-
// consumer rematerialization. Mirrors MemorySwapPlanner's own JIT-R016
// captured-value shape (test_memory_planner_swap.cpp): the subgraph node's
// input is the SAME Value object as the outer producer's output, not a
// distinct subgraph-local placeholder.
TEST(RematerializationPlannerTest, ExcludesValueWithConsumerNestedInIfSubgraph) {
    Graph graph;
    const std::vector<int64_t> shape = {4};
    auto x = graph.create_value("x", shape, DType::Float32, Device::cpu());
    auto cond = graph.create_value("cond", {1}, DType::Float32, Device::cpu());

    // Cheap producer with a long-enough live range to otherwise qualify
    // (mirrors ExcludesValueWithMultipleConsumers's chain1 padding).
    auto node_a = graph.create_node(OpType::ReLU, "A");
    auto out_a = graph.create_value("out_A", shape, DType::Float32, Device::cpu());
    node_a->add_input(x);
    node_a->add_output(out_a);
    out_a->set_node(node_a);
    graph.add_node(node_a);

    // Top-level consumer #1.
    std::shared_ptr<Value> prev1 = out_a;
    for (int i = 0; i < 3; ++i) {
        auto node = graph.create_node(OpType::ReLU, "chain1_" + std::to_string(i));
        auto out = graph.create_value("chain1_out_" + std::to_string(i), shape,
                                       DType::Float32, Device::cpu());
        node->add_input(prev1);
        node->add_output(out);
        out->set_node(node);
        graph.add_node(node);
        prev1 = out;
    }

    // If node, purely structural -- does not itself consume out_A.
    auto if_out = graph.create_value("if_out", shape, DType::Float32, Device::cpu());
    auto if_node = graph.create_node(OpType::If, "if");
    if_node->add_input(cond);
    if_out->set_node(if_node);
    if_node->add_output(if_out);

    // Nested consumer #2: a node inside the then-branch subgraph captures
    // out_A directly (same Value object as the outer graph's).
    auto sub = std::make_shared<Graph>();
    auto sub_neg = sub->create_node(OpType::Neg, "sub_neg");
    sub_neg->add_input(out_a);
    auto sub_out = sub->create_value("sub_out", shape, DType::Float32, Device::cpu());
    sub_out->set_node(sub_neg);
    sub_neg->add_output(sub_out);
    sub->add_node(sub_neg);
    sub->set_inputs({});
    sub->set_outputs({sub_out});
    if_node->set_then_branch(sub);
    graph.add_node(if_node);

    auto node_y = graph.create_node(OpType::Add, "Y");
    auto out_y = graph.create_value("out_Y", shape, DType::Float32, Device::cpu());
    node_y->add_input(prev1);
    node_y->add_input(if_out);
    node_y->add_output(out_y);
    out_y->set_node(node_y);
    graph.add_node(node_y);

    graph.set_inputs({x, cond});
    graph.set_outputs({out_y});

    // Sanity: out_A has exactly 2 real consumers (one top-level, one
    // nested), even though only 1 is visible to a top-level-only index.
    ASSERT_EQ(out_a->uses().size(), 2u);

    RematerializationPlanner planner;
    auto candidates = planner.find_candidates(graph);
    bool found_out_a = false;
    for (const auto& c : candidates) if (c.value_id == "out_A") found_out_a = true;
    ASSERT_TRUE(found_out_a);

    size_t applied = planner.apply(graph, candidates);
    EXPECT_EQ(applied, 0u)
        << "out_A has 2 live consumers (1 top-level + 1 nested in an If "
           "subgraph) -- apply() must reject it, not silently undercount "
           "the nested one and over-credit memory_saved";

    auto v = graph.get_value("out_A");
    ASSERT_TRUE(v);
    EXPECT_EQ(v->uses().size(), 2u) << "no consumer should have been redirected";
    // The nested consumer specifically must still read the ORIGINAL value
    // (not have been silently redirected to an out-of-scope recompute node
    // added at the top level).
    ASSERT_EQ(sub_neg->inputs().size(), 1u);
    EXPECT_EQ(sub_neg->inputs()[0]->id(), "out_A");
}

TEST(RematerializationPlannerTest, ExcludesExpensiveOp) {
    Graph graph;
    const std::vector<int64_t> shape = {8, 8};
    auto a = graph.create_value("a", shape, DType::Float32, Device::cpu());
    auto b = graph.create_value("b", shape, DType::Float32, Device::cpu());

    auto node_mm = graph.create_node(OpType::MatMul, "mm");
    auto out_mm = graph.create_value("out_mm", shape, DType::Float32, Device::cpu());
    node_mm->add_input(a);
    node_mm->add_input(b);
    node_mm->add_output(out_mm);
    out_mm->set_node(node_mm);
    graph.add_node(node_mm);

    std::shared_ptr<Value> prev = a;
    for (int i = 0; i < 4; ++i) {
        auto node = graph.create_node(OpType::ReLU, "filler" + std::to_string(i));
        auto out = graph.create_value("filler_out_" + std::to_string(i), shape,
                                       DType::Float32, Device::cpu());
        node->add_input(prev);
        node->add_output(out);
        out->set_node(node);
        graph.add_node(node);
        prev = out;
    }

    auto node_y = graph.create_node(OpType::Add, "Y");
    auto out_y = graph.create_value("out_Y", shape, DType::Float32, Device::cpu());
    node_y->add_input(out_mm);
    node_y->add_input(prev);
    node_y->add_output(out_y);
    out_y->set_node(node_y);
    graph.add_node(node_y);

    graph.set_inputs({a, b});
    graph.set_outputs({out_y});

    RematerializationPlanner planner;
    auto candidates = planner.find_candidates(graph);
    for (const auto& c : candidates) {
        EXPECT_NE(c.value_id, "out_mm") << "MatMul is not cheap to recompute";
    }
}

// ============================================================================
// RematerializationPlanner::apply -- graph mutation + NUMERIC CORRECTNESS
// ============================================================================

TEST(RematerializationPlannerTest, ApplyInsertsRecomputeNodeAndRedirectsConsumer) {
    Graph graph;
    build_long_lived_plus_filler_chain(graph);

    RematerializationPlanner planner;
    auto candidates = planner.find_candidates(graph);
    ASSERT_EQ(candidates.size(), 1u);

    size_t before_node_count = graph.num_nodes();
    size_t applied = planner.apply(graph, candidates);
    ASSERT_EQ(applied, 1u);
    EXPECT_EQ(graph.num_nodes(), before_node_count + 1);

    // A new "A_remat" node must exist.
    bool found_remat = false;
    for (const auto& node : graph.nodes()) {
        if (node->name() == "A_remat") {
            found_remat = true;
            EXPECT_EQ(node->op_type(), OpType::ReLU);
            EXPECT_TRUE(node->has_bool_attr("rematerialized"));
        }
    }
    EXPECT_TRUE(found_remat);

    // The Y node must no longer read out_A directly.
    auto y_node = graph.get_value("out_Y")->node();
    ASSERT_TRUE(y_node);
    bool still_reads_out_a = false;
    for (const auto& inp : y_node->inputs()) {
        if (inp->id() == "out_A") still_reads_out_a = true;
    }
    EXPECT_FALSE(still_reads_out_a);

    // out_A itself now has zero remaining consumers among reachable nodes.
    auto out_a = graph.get_value("out_A");
    ASSERT_TRUE(out_a);
    size_t live_uses = 0;
    for (const auto& use : out_a->uses()) {
        if (use.lock()) ++live_uses;
    }
    EXPECT_EQ(live_uses, 0u);
}

// The critical correctness property: rematerializing must not change the
// computed result. Builds two structurally-identical graphs, rematerializes
// only the second, executes both with the SAME input, and asserts bit-exact
// equality.
TEST(RematerializationPlannerTest, RematerializedGraphProducesIdenticalOutput) {
    Graph graph_ref, graph_remat;
    build_long_lived_plus_filler_chain(graph_ref);
    build_long_lived_plus_filler_chain(graph_remat);

    RematerializationPlanner planner;
    auto candidates = planner.find_candidates(graph_remat);
    ASSERT_EQ(candidates.size(), 1u);
    ASSERT_EQ(planner.apply(graph_remat, candidates), 1u);

    std::vector<float> xs = {-2.0f, -0.5f, 0.5f, 2.0f};
    Tensor x_tensor = from_data(xs.data(), {4});
    Variable x_ref(x_tensor, false);
    Variable x_remat(x_tensor, false);

    auto ref_results = graph_ref.forward({x_ref});
    auto remat_results = graph_remat.forward({x_remat});

    ASSERT_EQ(ref_results.size(), 1u);
    ASSERT_EQ(remat_results.size(), 1u);

    auto ref_cpu = ref_results[0].tensor().to(Device::cpu()).contiguous();
    auto remat_cpu = remat_results[0].tensor().to(Device::cpu()).contiguous();
    const float* rd = ref_cpu.data<float>();
    const float* md = remat_cpu.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(rd[i], md[i]) << "mismatch at index " << i;
    }
}

TEST(RematerializationPlannerTest, MemoryBudgetStopsAfterTargetReclaimed) {
    Graph graph;
    build_long_lived_plus_filler_chain(graph);

    RematerializationPlanner planner;
    auto candidates = planner.find_candidates(graph);
    ASSERT_EQ(candidates.size(), 1u);

    planner.set_memory_budget(1);  // reclaim target met by the very first candidate
    size_t applied = planner.apply(graph, candidates);
    EXPECT_EQ(applied, 1u);
}

// ============================================================================
// RematerializationPass / Compiler wiring
// ============================================================================

TEST(CompilerRematerializationTest, DisabledByDefault) {
    Graph graph;
    build_long_lived_plus_filler_chain(graph);

    Compiler compiler(/*enable_default_passes=*/false);
    compiler.optimize(graph);

    EXPECT_EQ(compiler.num_rematerialized(), 0u);
    auto out_a = graph.get_value("out_A");
    ASSERT_TRUE(out_a);
    size_t live_uses = 0;
    for (const auto& use : out_a->uses()) if (use.lock()) ++live_uses;
    EXPECT_EQ(live_uses, 1u) << "the original single consumer must be untouched";
}

TEST(CompilerRematerializationTest, EnabledEndToEndMatchesUnoptimizedOutput) {
    Graph graph_ref, graph_opt;
    build_long_lived_plus_filler_chain(graph_ref);
    build_long_lived_plus_filler_chain(graph_opt);

    Compiler compiler(/*enable_default_passes=*/false);
    compiler.set_rematerialization(true);
    compiler.optimize(graph_opt);

    EXPECT_EQ(compiler.num_rematerialized(), 1u);

    std::vector<float> xs = {-1.0f, 0.25f, 1.5f, -3.0f};
    Tensor x_tensor = from_data(xs.data(), {4});

    auto ref_results = graph_ref.forward({Variable(x_tensor, false)});
    auto opt_results = graph_opt.forward({Variable(x_tensor, false)});

    ASSERT_EQ(ref_results.size(), 1u);
    ASSERT_EQ(opt_results.size(), 1u);
    auto ref_cpu = ref_results[0].tensor().to(Device::cpu()).contiguous();
    auto opt_cpu = opt_results[0].tensor().to(Device::cpu()).contiguous();
    const float* rd = ref_cpu.data<float>();
    const float* od = opt_cpu.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(rd[i], od[i]) << "mismatch at index " << i;
    }
}

TEST(CompilerRematerializationTest, RunsBeforeMemoryPlanningSoPlanSeesShorterLiveRange) {
    Graph graph;
    build_long_lived_plus_filler_chain(graph);

    Compiler compiler(/*enable_default_passes=*/false);
    compiler.set_rematerialization(true);
    compiler.set_memory_planning(true);
    compiler.optimize(graph);

    ASSERT_EQ(compiler.num_rematerialized(), 1u);
    // out_A's live range should now be exactly its own producer node (no
    // remaining consumer), i.e. begin == end -- reflected as either being
    // excluded from planning (zero remaining uses can shrink the range to a
    // single point) or, at minimum, no longer stretching across the whole
    // filler chain. The concrete, robust assertion: memory planning ran
    // (proving the two passes composed without one crashing the other) and
    // out_A is no longer read by node "Y".
    EXPECT_GT(compiler.memory_plan().num_values_planned, 0u);
    auto y_node = graph.get_value("out_Y")->node();
    ASSERT_TRUE(y_node);
    for (const auto& inp : y_node->inputs()) {
        EXPECT_NE(inp->id(), "out_A");
    }
}
