/**
 * @file test_memory_planner_swap.cpp
 * @brief Regression coverage for MemorySwapPlanner (JIT-R016, JIT-R017).
 *
 * MemorySwapPlanner is currently unwired (no Compiler/codegen path invokes
 * it), so it has no other test coverage. It is still worth being correct,
 * since the class's own doc comment describes real GPU-buffer-eviction
 * semantics as the intended future behavior once some backend's codegen
 * lowers SwapOut/SwapIn.
 *
 * JIT-R016: MemorySwapPlanner::apply's single-consumer gate counted ALL of
 *   value->uses() unfiltered, while its redirect search only considered
 *   nodes findable in a node_index built solely from the top-level graph's
 *   nodes(). Node::add_input (src/jit/graph.cpp) wires up a value's uses()
 *   list purely based on which Node consumes it, with no regard for which
 *   Graph (top-level or an If/Loop subgraph body) owns that node. So a
 *   value whose single real consumer lives inside a subgraph body passed
 *   the gate (live_consumers == 1) but the redirect search found nothing,
 *   leaving an orphaned SwapOut/SwapIn pair spliced into the graph while
 *   the real (subgraph) consumer kept reading the original value directly.
 *   The fix makes the reachable-node set used by both checks recursively
 *   subgraph-aware, so the gate and the redirect search always agree: the
 *   consumer is found and correctly redirected.
 *
 * JIT-R017: MemorySwapPlanner::plan never gated swap candidates by device,
 *   despite the class's doc comment describing GPU-activation
 *   swap-to-CPU semantics. A CPU-resident intermediate could be scheduled
 *   for a pointless CPU->CPU "swap". The fix skips any candidate whose
 *   output device is CPU.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/tracer.hpp>  // OpType
#include <tenzor/jit/memory_planner.hpp>

using namespace tenzor;
using namespace tenzor::jit;

namespace {

// JIT-R016 repro: build an If graph on CPU where the value `y` (produced by
// a top-level node) is captured DIRECTLY by a node inside the then-branch
// subgraph -- i.e. the subgraph node's input is the very same Value object
// as the outer `y`, not a distinct subgraph-local placeholder routed through
// the If node's own inputs. This is the "captured value" shape JIT-R016 is
// about: Node::add_input doesn't care which Graph owns the consuming node,
// so y's uses() legitimately contains a node that lives in the subgraph.
TEST(MemorySwapPlanner, RedirectsSoleConsumerInsideIfSubgraph) {
    auto g = std::make_shared<Graph>();
    const std::vector<int64_t> xs = {2, 3};

    auto x = g->create_value("x", xs, DType::Float32, Device::cpu());
    auto cond = g->create_value("cond", {1}, DType::Float32, Device::cpu());

    // Producer of the swap candidate `y`. Tagged non-CPU (CUDA) so it passes
    // JIT-R017's device gate in plan() below -- this test targets JIT-R016
    // (apply()'s consumer redirect), not JIT-R017 (covered separately by
    // NeverSelectsCpuResidentCandidate). This test binary is CPU-only, but
    // Device is pure IR metadata on Value here: nothing in this test
    // executes the graph or dispatches to a backend, so tagging a Value
    // CUDA is safe without a live CUDA runtime.
    auto make_y = g->create_node(OpType::Neg, "make_y");
    make_y->add_input(x);
    auto y = g->create_value("y", xs, DType::Float32, Device::cuda(0));
    y->set_node(make_y);
    make_y->add_output(y);
    g->add_node(make_y);

    // If node: purely structural here (condition/branches aren't executed by
    // this test), routes control flow but does NOT itself consume `y`.
    auto if_out = g->create_value("if_out", xs, DType::Float32, Device::cpu());
    auto if_node = g->create_node(OpType::If, "if");
    if_node->add_input(cond);
    if_out->set_node(if_node);
    if_node->add_output(if_out);

    // then-branch subgraph: its only node consumes `y` directly (captured
    // from the outer graph, not passed through a subgraph-local input).
    auto sub = std::make_shared<Graph>();
    auto sub_neg = sub->create_node(OpType::Neg, "sub_neg");
    sub_neg->add_input(y);
    auto t_out = sub->create_value("t_out", xs, DType::Float32, Device::cpu());
    t_out->set_node(sub_neg);
    sub_neg->add_output(t_out);
    sub->add_node(sub_neg);
    sub->set_inputs({});
    sub->set_outputs({t_out});
    if_node->set_then_branch(sub);

    g->add_node(if_node);
    g->set_inputs({x, cond});
    g->set_outputs({if_out});

    // Sanity: y has exactly one real consumer, and it lives in the subgraph.
    ASSERT_EQ(y->uses().size(), 1u);

    // Run the full plan()+apply() pipeline. Thresholds are permissive (0) so
    // `y` qualifies purely on the strength of being GPU-resident with a
    // single (subgraph) consumer -- note plan()'s own death/gap computation
    // is top-level-only (out of scope for JIT-R016, which is confined to
    // apply()), so with min_gap_ == 0 the zero live-range gap it computes
    // for a subgraph-only consumer still clears the (permissive) bar and a
    // schedule is produced.
    MemorySwapPlanner planner;
    planner.set_swap_threshold(0);
    planner.set_min_gap(0);

    auto schedules = planner.plan(*g);
    ASSERT_EQ(schedules.size(), 1u);
    EXPECT_EQ(schedules[0].value_id, "y");

    size_t count = planner.apply(*g, schedules);

    // The redirect must have actually happened: exactly one swap pair
    // spliced in, and it must be wired to the real (subgraph) consumer --
    // not an orphaned pair with the subgraph node still reading `y`.
    EXPECT_EQ(count, 1u);

    ASSERT_EQ(sub_neg->inputs().size(), 1u);
    EXPECT_NE(sub_neg->inputs()[0]->id(), "y");
    EXPECT_EQ(sub_neg->inputs()[0]->id(), "swap_in_y_gpu");

    // `y` itself must now only be referenced by the SwapOut node -- the
    // subgraph consumer was redirected away, so there is no dangling
    // reference to the (to-be-evicted) original value.
    size_t y_live_uses = 0;
    for (const auto& use : y->uses()) {
        if (use.lock()) ++y_live_uses;
    }
    EXPECT_EQ(y_live_uses, 1u);

    auto swap_out_node = y->uses().empty() ? nullptr : y->uses()[0].lock();
    ASSERT_TRUE(swap_out_node);
    EXPECT_EQ(swap_out_node->op_type(), OpType::SwapOut);

    // The new swapped-in value must have exactly one use: the redirected
    // subgraph consumer (not orphaned, not double-wired).
    auto gpu_restored = g->get_value("swap_in_y_gpu");
    ASSERT_TRUE(gpu_restored);
    size_t gpu_restored_uses = 0;
    for (const auto& use : gpu_restored->uses()) {
        if (use.lock()) ++gpu_restored_uses;
    }
    EXPECT_EQ(gpu_restored_uses, 1u);
}

// JIT-R016, fail-closed side: if a value's gated single consumer cannot
// actually be found/redirected (here: the value has no consumer at all,
// live_consumers == 0), apply() must not splice an orphaned SwapOut/SwapIn
// pair -- it must leave the value alone.
TEST(MemorySwapPlanner, DoesNotSpliceWhenNoConsumerFound) {
    auto g = std::make_shared<Graph>();
    const std::vector<int64_t> xs = {2, 3};

    auto x = g->create_value("x", xs, DType::Float32, Device::cpu());
    auto make_y = g->create_node(OpType::Neg, "make_y");
    make_y->add_input(x);
    auto y = g->create_value("y", xs, DType::Float32, Device::cpu());
    y->set_node(make_y);
    make_y->add_output(y);
    g->add_node(make_y);

    g->set_inputs({x});
    g->set_outputs({y});

    ASSERT_EQ(y->uses().size(), 0u);

    MemorySwapPlanner planner;
    SwapSchedule sched;
    sched.value_id = "y";
    sched.swap_out_after = 0;
    sched.swap_in_before = 0;
    sched.size_bytes = 24;

    size_t count = planner.apply(*g, {sched});

    EXPECT_EQ(count, 0u);
    // No SwapOut/SwapIn nodes should have been added to the graph.
    for (const auto& node : g->nodes()) {
        EXPECT_NE(node->op_type(), OpType::SwapOut);
        EXPECT_NE(node->op_type(), OpType::SwapIn);
    }
}

// JIT-R017 repro: a CPU-resident intermediate, sized and live-ranged well
// past the swap thresholds, must never be selected as a swap candidate by
// plan(). Thresholds are set permissively (0) so the ONLY thing that can be
// excluding it is the device gate.
TEST(MemorySwapPlanner, NeverSelectsCpuResidentCandidate) {
    auto g = std::make_shared<Graph>();
    const std::vector<int64_t> xs = {4, 4};

    auto x = g->create_value("x", xs, DType::Float32, Device::cpu());

    auto n1 = g->create_node(OpType::Neg, "n1");
    n1->add_input(x);
    auto y = g->create_value("y", xs, DType::Float32, Device::cpu());
    y->set_node(n1);
    n1->add_output(y);
    g->add_node(n1);

    // A few more nodes so y's live range comfortably exceeds any min_gap_.
    auto n2 = g->create_node(OpType::Neg, "n2");
    n2->add_input(y);
    auto z1 = g->create_value("z1", xs, DType::Float32, Device::cpu());
    z1->set_node(n2);
    n2->add_output(z1);
    g->add_node(n2);

    auto n3 = g->create_node(OpType::Neg, "n3");
    n3->add_input(z1);
    auto z2 = g->create_value("z2", xs, DType::Float32, Device::cpu());
    z2->set_node(n3);
    n3->add_output(z2);
    g->add_node(n3);

    auto n4 = g->create_node(OpType::Neg, "n4");
    n4->add_input(z2);
    // Second (later) consumer of y to widen its live range further.
    n4->add_input(y);
    auto out = g->create_value("out", xs, DType::Float32, Device::cpu());
    out->set_node(n4);
    n4->add_output(out);
    g->add_node(n4);

    g->set_inputs({x});
    g->set_outputs({out});

    MemorySwapPlanner planner;
    planner.set_swap_threshold(0);  // permissive: any nonzero size qualifies
    planner.set_min_gap(0);         // permissive: any live range qualifies

    auto schedules = planner.plan(*g);

    // Every intermediate here (x excluded as a graph input, out excluded as
    // a graph output) is CPU-resident, so none should be selected.
    for (const auto& sched : schedules) {
        EXPECT_NE(sched.value_id, "y");
        EXPECT_NE(sched.value_id, "z1");
        EXPECT_NE(sched.value_id, "z2");
    }
    EXPECT_TRUE(schedules.empty());
}

}  // namespace
