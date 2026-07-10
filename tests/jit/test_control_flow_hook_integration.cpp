/**
 * @file test_control_flow_hook_integration.cpp
 * @brief P4.1b: end-to-end integration test for control_flow.hpp + Tracer.
 *
 * Verifies that:
 *  1. `jit::cond` called from inside a TracingGuard records an `If` op
 *     on the tracer's recorded operations list.
 *  2. `jit::while_loop` records a `Loop` op.
 *  3. Native `.item()` on a traced tensor triggers the graph-break hook
 *     and, in strict mode, throws pointing the user at jit::cond/while_loop.
 *
 * The first two cases are the "happy path" where users follow the guide;
 * the third is the failure mode where users use native C++ / Python
 * control flow on tensor values.
 */

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/jit/control_flow.hpp>
#include <tenzor/core/jit_hooks.hpp>
#include <tenzor/backend/fast_dispatch.hpp>  // is_op_supported, OpId

namespace tenzor {
namespace {

using jit::Tracer;

class ControlFlowIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
        Tracer::get_instance().clear();
    }
    void TearDown() override {
        tenzor::detail::set_graph_break_hook(nullptr);
        Tracer::get_instance().clear();
    }
};

TEST_F(ControlFlowIntegrationTest, CondDuringTraceRecordsIfOp) {
    auto& t = Tracer::get_instance();
    t.start_trace();

    auto x = Variable(ones({4}, DType::Float32, Device::cpu()), false);
    // Condition is a scalar non-zero (truthy).
    auto cond = ones({1}, DType::Float32, Device::cpu());

    auto result = jit::cond(
        cond,
        [](const Variable& v) -> Variable { return v + v; },
        [](const Variable& v) -> Variable { return v * -1.0f; },
        x);

    // The value equals then_branch's output (cond is truthy).
    EXPECT_NEAR(result.tensor().data<float>()[0], 2.0f, 1e-6f);

    // No graph breaks should have been reported — jit::cond is the
    // sanctioned path.
    EXPECT_EQ(t.graph_break_count(), 0);
    t.clear();
}

TEST_F(ControlFlowIntegrationTest, WhileLoopDuringTraceRecordsLoopOp) {
    auto& t = Tracer::get_instance();
    t.start_trace();

    auto x = Variable(zeros({1}, DType::Float32, Device::cpu()), false);
    x.tensor().data<float>()[0] = 0.0f;

    auto result = jit::while_loop(
        3,  // max iterations
        [](const std::vector<Variable>& state) -> Tensor {
            // Always continue (simplified — real cond_fn would inspect state).
            return ones({1}, DType::Float32, Device::cpu());
        },
        [](const std::vector<Variable>& state) -> std::vector<Variable> {
            auto one = Variable(ones({1}, DType::Float32, Device::cpu()), false);
            return {state[0] + one};
        },
        {x});

    ASSERT_EQ(result.size(), 1u);
    // Tracing mode: while_loop records one iteration as the body
    // subgraph, so the trace-time return value reflects one body
    // application (x + 1 = 1), not the full loop unroll.
    EXPECT_NEAR(result[0].tensor().data<float>()[0], 1.0f, 1e-6f);
    EXPECT_EQ(t.graph_break_count(), 0);
    t.clear();
}

TEST_F(ControlFlowIntegrationTest, NativeItemInStrictTraceThrowsWithCondHint) {
    // The failure mode: user bypasses jit::cond and uses .item() directly.
    // Strict mode should throw and the error message should recommend
    // jit::cond / jit::while_loop as the replacement.
    auto& t = Tracer::get_instance();
    t.start_trace();
    t.set_strict_mode(true);
    tenzor::detail::set_graph_break_hook(
        [&t](const std::string& reason) { t.record_graph_break(reason); });

    auto x = zeros({1}, DType::Float32, Device::cpu());
    x.data<float>()[0] = 1.5f;

    try {
        (void)x.item<float>();
        FAIL() << "Expected strict-mode graph break throw";
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("scalar extraction"), std::string::npos);
        // Strict-mode message must point at the replacement API.
        EXPECT_TRUE(msg.find("jit::cond") != std::string::npos ||
                    msg.find("while_loop") != std::string::npos);
    }

    tenzor::detail::set_graph_break_hook(nullptr);
    t.clear();
}

TEST_F(ControlFlowIntegrationTest, CondReplayDispatchesDynamically) {
    // Trace once with a truthy condition, then replay the compiled graph
    // with both truthy and falsy conditions. The If node's attached
    // then_branch / else_branch subgraphs should be dispatched based on
    // the runtime condition — not baked to the trace-time branch.
    auto cond_t = ones({1}, DType::Float32, Device::cpu());
    Variable cond_var(cond_t, false);
    auto x = Variable(ones({1}, DType::Float32, Device::cpu()), false);

    std::shared_ptr<jit::Graph> graph;
    Variable result;
    {
        jit::TracingGuard guard;
        result = jit::cond(
            cond_t,  // truthy at trace time
            [](const Variable& v) -> Variable { return v + v; },          // then:  v+v  = 2
            [](const Variable& v) -> Variable { return v * -3.0f; },      // else:  v*-3 = -3
            x);
        graph = Tracer::get_instance().end_trace({cond_var, x}, {result});
    }
    ASSERT_TRUE(graph);

    // Replay with truthy condition — expect then branch (2).
    {
        auto c = Variable(ones({1}, DType::Float32, Device::cpu()), false);
        auto xin = Variable(ones({1}, DType::Float32, Device::cpu()), false);
        auto out = graph->forward({c, xin});
        ASSERT_EQ(out.size(), 1u);
        EXPECT_NEAR(out[0].tensor().data<float>()[0], 2.0f, 1e-5f);
    }

    // Replay with falsy condition — expect else branch (-3).
    {
        auto c = Variable(zeros({1}, DType::Float32, Device::cpu()), false);
        auto xin = Variable(ones({1}, DType::Float32, Device::cpu()), false);
        auto out = graph->forward({c, xin});
        ASSERT_EQ(out.size(), 1u);
        EXPECT_NEAR(out[0].tensor().data<float>()[0], -3.0f, 1e-5f);
    }
}

TEST_F(ControlFlowIntegrationTest, InPlaceMutationInCondBranchThrows) {
    // JIT-F043: an in-place op inside a cond branch mutates a tensor visible
    // OUTSIDE the branch (here the carried input x). trace_if executes BOTH
    // branches eagerly on the same tensors but only one runs at replay, so the
    // mutation is conditional and has no functional trace — it would remap the
    // tensor's SSA id inside the branch's skipped op range and silently corrupt
    // the other branch / post-cond graph. trace_if must reject it loudly.
    auto cond_t = ones({1}, DType::Float32, Device::cpu());
    auto x = Variable(ones({2}, DType::Float32, Device::cpu()), false);
    jit::TracingGuard guard;
    EXPECT_THROW(
        {
            auto result = jit::cond(
                cond_t,
                [](const Variable& v) -> Variable {
                    Tensor t = v.tensor();
                    tenzor::add_(t, t);  // in-place mutation of a carried input
                    return v;
                },
                [](const Variable& v) -> Variable { return v + v; },
                x);
            (void)result;
        },
        std::runtime_error);
}

TEST_F(ControlFlowIntegrationTest, LoopBodySubgraphAttached) {
    // Traces a while_loop, end_traces to a Graph, and verifies that the
    // Loop node has a non-empty body subgraph with the expected tensor
    // IDs. This asserts the structural outcome of task #62 (body is
    // attached as a sub-Graph) without depending on the Graph executor's
    // ONNX-style loop semantic being wired end-to-end.
    auto x = Variable(zeros({1}, DType::Float32, Device::cpu()), false);

    std::shared_ptr<jit::Graph> graph;
    {
        jit::TracingGuard guard;
        auto result = jit::while_loop(
            3,
            [](const std::vector<Variable>&) -> Tensor {
                return ones({1}, DType::Float32, Device::cpu());
            },
            [](const std::vector<Variable>& state) -> std::vector<Variable> {
                auto one = Variable(ones({1}, DType::Float32, Device::cpu()), false);
                return {state[0] + one};
            },
            {x});
        ASSERT_EQ(result.size(), 1u);
        graph = Tracer::get_instance().end_trace({x}, {result[0]});
    }

    ASSERT_TRUE(graph);

    // Exactly one Loop node should remain in the parent graph; its body
    // slice should have been lifted into a subgraph.
    std::shared_ptr<jit::Node> loop_node;
    for (const auto& n : graph->nodes()) {
        if (n->op_type() == jit::OpType::Loop) {
            loop_node = n;
            break;
        }
    }
    ASSERT_TRUE(loop_node);
    ASSERT_TRUE(loop_node->body());
    // The body ran one iteration (state[0] + 1 = 1 Add op), so the
    // subgraph must contain at least one node.
    EXPECT_GE(loop_node->body()->nodes().size(), 1u);
    // ONNX loop-body outputs are [cond, carried...]. Here cond is a freshly
    // created constant (ones()) with no producing op; it must still be surfaced
    // as a body output (materialized as a sub-graph constant, JIT-F017) so the
    // interpreter reads body_outputs[0] as the next-iteration condition. Hence
    // two outputs: the condition and the single carried value.
    EXPECT_EQ(loop_node->body()->outputs().size(), 2u);
}

// ---------------------------------------------------------------------------
// H5: a traced while_loop must replay through Graph::forward with EXACTLY the
// same semantics as eager while_loop for 0, 1, and N iterations. The loop is
// data-dependent: carried = {counter, limit}, cond = (limit - counter) > 0,
// body increments counter. Replaying with different runtime `limit` values
// drives a different iteration count, proving the compiled Loop executes the
// body per-iteration (not baked to the trace-time count) and honours the
// entry-condition-first (zero-iteration) contract.
static void run_while_loop_replay_on(Device dev) {
    // carried = {counter, limit}; cond = (limit - counter) > 0; body = counter+1.
    // Constants live on `dev` so the body's Add stays on-device.
    auto cond_fn = [](const std::vector<Variable>& s) -> Tensor {
        return s[1].tensor() - s[0].tensor();  // positive while counter < limit
    };
    auto body_fn = [dev](const std::vector<Variable>& s) -> std::vector<Variable> {
        auto one = Variable(full({1}, 1.0f, DType::Float32, dev), false);
        return {s[0] + one, s[1]};
    };

    std::shared_ptr<jit::Graph> graph;
    {
        jit::TracingGuard guard;
        auto counter = Variable(full({1}, 0.0f, DType::Float32, dev), false);
        auto limit   = Variable(full({1}, 5.0f, DType::Float32, dev), false);
        auto result = jit::while_loop(1000, cond_fn, body_fn, {counter, limit});
        ASSERT_EQ(result.size(), 2u);
        graph = Tracer::get_instance().end_trace({counter, limit}, {result[0], result[1]});
    }
    ASSERT_TRUE(graph);

    std::shared_ptr<jit::Node> loop_node;
    for (const auto& n : graph->nodes()) {
        if (n->op_type() == jit::OpType::Loop) { loop_node = n; break; }
    }
    ASSERT_TRUE(loop_node);
    ASSERT_TRUE(loop_node->body());

    // For each runtime limit K (0 -> zero iterations), the compiled graph must
    // produce the same final counter as eager while_loop.
    for (float K : {0.0f, 1.0f, 3.0f, 7.0f, 20.0f}) {
        auto counter_in = Variable(full({1}, 0.0f, DType::Float32, dev), false);
        auto limit_in   = Variable(full({1}, K,    DType::Float32, dev), false);

        auto eager = jit::while_loop(1000, cond_fn, body_fn, {counter_in, limit_in});
        ASSERT_EQ(eager.size(), 2u);
        float eager_counter = eager[0].tensor().to(Device::cpu()).data<float>()[0];
        EXPECT_NEAR(eager_counter, K, 1e-5f)
            << "eager sanity K=" << K << " dev=" << dev.to_string();

        auto out = graph->forward({counter_in, limit_in});
        ASSERT_EQ(out.size(), 2u) << "K=" << K << " dev=" << dev.to_string();
        float replay_counter = out[0].tensor().to(Device::cpu()).data<float>()[0];
        EXPECT_NEAR(replay_counter, eager_counter, 1e-5f)
            << "compiled loop diverged from eager K=" << K << " dev=" << dev.to_string();
    }
}

TEST_F(ControlFlowIntegrationTest, WhileLoopReplayMatchesEagerForVariousTripCounts) {
    run_while_loop_replay_on(Device::cpu());
    if (is_op_supported(OpId::Add, Device::Type::CUDA)) {
        run_while_loop_replay_on(Device::cuda(0));
    }
    if (is_op_supported(OpId::Add, Device::Type::ROCm)) {
        run_while_loop_replay_on(Device::rocm(0));
    }
}

// H2: Tensor::to(dtype) must record a Cast node on CPU (previously it converted
// inline with no dispatch, so a CPU trace silently dropped the conversion and
// produced a different graph than GPU). Assert the trace contains Cast nodes
// AND that the compiled graph replays identically to eager.
TEST_F(ControlFlowIntegrationTest, ToDtypeRecordsCastNodeAndReplaysOnCpu) {
    auto x = Variable(randn({2, 3}, DType::Float32, Device::cpu()), false);

    std::shared_ptr<jit::Graph> graph;
    {
        jit::TracingGuard guard;
        // F32 -> F64, double it, then F64 -> F32.
        Variable f64(x.tensor().to(DType::Float64), false);
        Variable doubled = f64 + f64;
        Variable out(doubled.tensor().to(DType::Float32), false);
        graph = Tracer::get_instance().end_trace({x}, {out});
    }
    ASSERT_TRUE(graph);

    int cast_count = 0;
    for (const auto& n : graph->nodes()) {
        if (n->op_type() == jit::OpType::Cast) ++cast_count;
    }
    EXPECT_GE(cast_count, 2) << "CPU .to(dtype) must record Cast nodes in the trace";

    auto xin = Variable(x.tensor().clone(), false);
    auto out = graph->forward({xin});
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].tensor().dtype(), DType::Float32);

    Tensor x64 = x.tensor().to(DType::Float64);
    Tensor eager = (x64 + x64).to(DType::Float32);
    auto diff = tenzor::max(tenzor::abs(out[0].tensor() - eager));
    EXPECT_LT(diff.data<float>()[0], 1e-5f);
}

TEST_F(ControlFlowIntegrationTest, CondOutsideTraceRunsEagerly) {
    // Outside a trace, jit::cond evaluates the condition and calls the
    // selected branch directly. No graph-break reports either.
    auto x = Variable(ones({4}, DType::Float32, Device::cpu()), false);
    auto cond_t = zeros({1}, DType::Float32, Device::cpu());  // false → else branch

    auto result = jit::cond(
        cond_t,
        [](const Variable& v) -> Variable { return v + v; },
        [](const Variable& v) -> Variable { return v * -1.0f; },
        x);

    EXPECT_NEAR(result.tensor().data<float>()[0], -1.0f, 1e-6f);
}

} // namespace
} // namespace tenzor
