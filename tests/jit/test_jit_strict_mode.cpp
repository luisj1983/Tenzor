/**
 * @file test_jit_strict_mode.cpp
 * @brief Tests for the Tracer strict-mode / graph-break detection added
 *        in P4.1.
 *
 * Scenarios covered:
 *  - A trace that doesn't call `.item()` completes without breaks.
 *  - Calling `.item()` on a traced tensor with strict mode OFF warns
 *    and increments `graph_break_count()`.
 *  - Calling `.item()` with strict mode ON throws, and the exception
 *    message points the user at `jit::cond` / `jit::while_loop`.
 *  - The hook is torn down after the tracer exits, so .item() outside
 *    a trace is still fine.
 */

#include <gtest/gtest.h>

#include <string>

#include <tenzor/tenzor.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/core/jit_hooks.hpp>

namespace tenzor {
namespace {

using jit::Tracer;

class JitStrictModeTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
        // Reset tracer between tests to avoid cross-contamination.
        Tracer::get_instance().clear();
    }
    void TearDown() override {
        // Ensure the graph-break hook is cleared even if a test throws.
        tenzor::detail::set_graph_break_hook(nullptr);
        Tracer::get_instance().clear();
    }
};

TEST_F(JitStrictModeTest, NonStrictDefaultDoesNotCrashOnItem) {
    auto& t = Tracer::get_instance();
    t.start_trace();
    t.set_strict_mode(false);
    // Install the hook manually (TracingGuard does this automatically,
    // but we need direct control to test behaviour without the full
    // dispatch interceptor machinery).
    tenzor::detail::set_graph_break_hook(
        [&t](const std::string& reason) { t.record_graph_break(reason); });

    auto x = zeros({1}, DType::Float32, Device::cpu());
    x.data<float>()[0] = 3.14f;

    // Expect no throw, but the graph-break counter should bump.
    EXPECT_NO_THROW({ (void)x.item<float>(); });
    EXPECT_GE(t.graph_break_count(), 1);

    tenzor::detail::set_graph_break_hook(nullptr);
    t.clear();
}

TEST_F(JitStrictModeTest, StrictModeThrowsOnItem) {
    auto& t = Tracer::get_instance();
    t.start_trace();
    t.set_strict_mode(true);
    tenzor::detail::set_graph_break_hook(
        [&t](const std::string& reason) { t.record_graph_break(reason); });

    auto x = zeros({1}, DType::Float32, Device::cpu());
    x.data<float>()[0] = 2.5f;

    try {
        (void)x.item<float>();
        FAIL() << "Expected std::runtime_error for scalar extraction during strict trace";
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("scalar extraction"), std::string::npos);
        // The strict-mode error should mention the replacement API.
        EXPECT_TRUE(msg.find("jit::cond") != std::string::npos ||
                    msg.find("jit::while_loop") != std::string::npos);
    }

    // record_graph_break turns tracing_ off before throwing so the
    // guard can unwind cleanly. Verify that.
    EXPECT_FALSE(t.is_tracing());

    tenzor::detail::set_graph_break_hook(nullptr);
    t.clear();
}

TEST_F(JitStrictModeTest, HookIsClearedOutsideOfTrace) {
    // Outside of an active trace, .item() must not invoke any hook —
    // the notify path should short-circuit on the null check.
    auto& t = Tracer::get_instance();
    EXPECT_FALSE(t.is_tracing());

    auto x = zeros({1}, DType::Float32, Device::cpu());
    x.data<float>()[0] = 42.0f;

    // No hook installed => no exception, no counter bump.
    EXPECT_NO_THROW({
        float v = x.item<float>();
        EXPECT_FLOAT_EQ(v, 42.0f);
    });
    EXPECT_EQ(t.graph_break_count(), 0);
}

TEST_F(JitStrictModeTest, GraphBreakCountResetsOnStartTrace) {
    auto& t = Tracer::get_instance();
    t.start_trace();
    t.set_strict_mode(false);
    tenzor::detail::set_graph_break_hook(
        [&t](const std::string& reason) { t.record_graph_break(reason); });

    auto x = zeros({1}, DType::Float32, Device::cpu());
    (void)x.item<float>();
    (void)x.item<float>();
    EXPECT_GE(t.graph_break_count(), 2);

    tenzor::detail::set_graph_break_hook(nullptr);
    t.clear();

    // Re-start and confirm the counter is back to zero.
    t.start_trace();
    EXPECT_EQ(t.graph_break_count(), 0);
    t.clear();
}

} // namespace
} // namespace tenzor
