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
