/**
 * @file test_jit_control_flow_inplace_fixes.cpp
 * @brief Regression tests for two JIT tracer correctness fixes:
 *
 *  M7 — a scripted `if` with a data-dependent condition must NOT silently bake
 *       one branch. Reading the condition scalar now goes through Tensor::item(),
 *       which fires the tracer graph-break hook: strict mode throws; non-strict
 *       mode warns and the taken branch is baked (matching eager for the traced
 *       configuration) rather than reading a raw data<float>()[0] that bypassed
 *       the hook.
 *
 *  M8 — in-place ops (add_/mul_/…) dispatch through dispatch_inplace, which
 *       bypasses the DispatchInterceptorStack. They were therefore invisible to
 *       the trace, and later reads of the mutated tensor resolved to its PRE-op
 *       graph value. The tracer now records a value-versioned functional node
 *       (new = op(old, …)) so replay applies the mutation.
 *
 * Every numeric assertion runs on all available backends (CPU, CUDA, ROCm, …).
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include <tenzor/tenzor.hpp>
#include <tenzor/jit/script.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/jit/control_flow.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/ops/math.hpp>

#include "../backend_parity/parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {
class JitCfInplaceEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new JitCfInplaceEnv);

// A scripted function with a data-dependent `if`. The condition `x.sum()` is a
// representable scalar, but a scripted if still bakes the trace-time branch —
// so reading it is a graph break.
constexpr const char* kScriptWithIf = R"(
    def forward(x):
        y = x
        if x.sum():
            y = x * 2.0
        else:
            y = x.neg()
        return y + 1.0
)";
}  // namespace

// ---------------------------------------------------------------------------
// M7: strict mode raises on a data-dependent scripted `if`.
// ---------------------------------------------------------------------------
TEST(JitScriptedIf, StrictModeRaisesOnDataDependentIf) {
    // TENZOR_JIT_STRICT is read at start_trace(); set it before compiling so the
    // trace's graph-break hook throws instead of silently baking one branch.
    ::setenv("TENZOR_JIT_STRICT", "1", /*overwrite=*/1);
    Tensor dummy = ones({2}, DType::Float32, Device::cpu());
    EXPECT_THROW(
        {
            auto compiled = jit::compile_script(kScriptWithIf, dummy);
            (void)compiled;
        },
        std::runtime_error)
        << "A scripted if with a data-dependent condition must raise in strict "
           "mode instead of silently recording only the taken branch.";
    ::unsetenv("TENZOR_JIT_STRICT");
}

// ---------------------------------------------------------------------------
// M7: non-strict mode compiles and replays == eager for the traced branch,
// on every available backend.
// ---------------------------------------------------------------------------
TEST(JitScriptedIf, NonStrictReplayMatchesEagerAllBackends) {
    ::unsetenv("TENZOR_JIT_STRICT");  // default: warn, don't throw

    for (const auto& dev : get_available_backends()) {
        SCOPED_TRACE(backend_name(dev));
        // Dummy has sum = 2 > 0 -> the `then` branch is baked at trace time.
        Tensor dummy = ones({2}, DType::Float32, dev);
        auto compiled = jit::compile_script(kScriptWithIf, dummy);
        ASSERT_NE(compiled, nullptr) << backend_name(dev);

        // Input also takes the `then` branch (sum = 6 > 0).
        Tensor x = full({2}, 3.0f, DType::Float32, dev);
        auto replay = compiled->forward(Variable(x, false)).tensor();

        // Eager `then` branch: y = x * 2; return y + 1  ->  3*2 + 1 = 7.
        Tensor eager = x * 2.0f + 1.0f;
        dev.synchronize();
        EXPECT_TENSORS_CLOSE(eager, replay, 1e-5f, 1e-5f);
    }
}

// ---------------------------------------------------------------------------
// M8: an in-place op whose result is later read is applied on replay,
// on every available backend.
// ---------------------------------------------------------------------------
TEST(JitInplaceTrace, InplaceMutationAppliedOnReplayAllBackends) {
    for (const auto& dev : get_available_backends()) {
        SCOPED_TRACE(backend_name(dev));
        Tensor one = ones({4}, DType::Float32, dev);

        // fn(x):  y = x + 1        (traced Add, produces an intermediate value)
        //         y.add_(1)        (in-place: y = x + 2, mutates in place)
        //         z = y + 1        (reads the MUTATED y)   -> z = x + 3
        auto fn = [&one](const std::vector<Variable>& ins) -> std::vector<Variable> {
            Tensor y = ins[0].tensor() + one;
            tenzor::add_(y, one);        // in-place mutation
            Tensor z = y + one;          // reads post-mutation value
            return {Variable(z, false)};
        };

        Tensor x = full({4}, 2.0f, DType::Float32, dev);
        Variable xv(x, false);

        auto graph = jit::trace(fn, {xv});
        ASSERT_NE(graph, nullptr) << backend_name(dev);

        auto outs = graph->forward({xv});
        ASSERT_EQ(outs.size(), 1u) << backend_name(dev);
        Tensor replay = outs[0].tensor();

        // Eager reference computed the same way.
        Tensor ey = x + one;
        tenzor::add_(ey, one);
        Tensor eager = ey + one;         // = x + 3

        dev.synchronize();

        // Guard against the pre-fix behavior: dropping the in-place add_ would
        // have made replay == x + 2, not x + 3.
        Tensor x_plus_2 = x + 2.0f;
        EXPECT_FALSE(tensors_close(x_plus_2, replay, 1e-5f, 1e-5f))
            << "in-place add_ was dropped from the trace (replay == x+2, the "
               "pre-op value)";
        EXPECT_TENSORS_CLOSE(eager, replay, 1e-5f, 1e-5f);
    }
}

// JIT-R175: Graph::has_side_effecting_node() must detect a genuine traced
// in-place mutation (used by CompiledFunction::operator()'s cache-hit catch
// block to decide whether a mid-replay failure can safely fall back to
// eager re-execution, or must be escalated to avoid double-applying a side
// effect that already fired for real before the failure).
TEST(JitInplaceTrace, GraphHasSideEffectingNodeDetectsInplaceMutation_JIT175) {
    Tensor one = ones({4}, DType::Float32, Device::cpu());

    // A graph with NO in-place mutation must report false.
    {
        auto fn = [&one](const std::vector<Variable>& ins) -> std::vector<Variable> {
            Tensor z = ins[0].tensor() + one;
            return {Variable(z, false)};
        };
        Variable xv(full({4}, 2.0f, DType::Float32, Device::cpu()), false);
        auto graph = jit::trace(fn, {xv});
        ASSERT_NE(graph, nullptr);
        EXPECT_FALSE(graph->has_side_effecting_node())
            << "a graph with no in-place mutation must not be flagged as "
               "side-effecting";
    }

    // A graph WITH a genuine in-place mutation must report true.
    {
        auto fn = [&one](const std::vector<Variable>& ins) -> std::vector<Variable> {
            Tensor y = ins[0].tensor() + one;
            tenzor::add_(y, one);  // in-place mutation
            Tensor z = y + one;
            return {Variable(z, false)};
        };
        Variable xv(full({4}, 2.0f, DType::Float32, Device::cpu()), false);
        auto graph = jit::trace(fn, {xv});
        ASSERT_NE(graph, nullptr);
        EXPECT_TRUE(graph->has_side_effecting_node())
            << "a graph containing a traced in-place mutation must be "
               "flagged as side-effecting (JIT-R175)";
    }
}

// ---------------------------------------------------------------------------
// M8b: an in-place op that mutates a CAPTURED CONSTANT LEAF must bake that
// leaf's PRE-mutation value. The tracer reuses the leaf's pre-op value id as
// the node input and, having no producing node, end_trace() bakes it as a
// constant. Before the fix the retained storage was a shallow copy sharing the
// (now-mutated) buffer, so the baked constant captured the POST-op value and
// replay double-counted the mutation. Replaying with a DIFFERENT input than was
// used at trace time exposes the stale constant.
// ---------------------------------------------------------------------------
TEST(JitInplaceTrace, InplaceMutationOfCapturedConstantBakesPreOpValue) {
    for (const auto& dev : get_available_backends()) {
        SCOPED_TRACE(backend_name(dev));

        // c is a captured constant leaf (created outside fn, not a trace input).
        Tensor c = full({4}, 5.0f, DType::Float32, dev);

        // fn(x): c.add_(x); return c   => 5 + x
        auto fn = [&c](const std::vector<Variable>& ins) -> std::vector<Variable> {
            tenzor::add_(c, ins[0].tensor());  // in-place mutate captured leaf
            return {Variable(c, false)};
        };

        Tensor x_trace = full({4}, 2.0f, DType::Float32, dev);
        auto graph = jit::trace(fn, {Variable(x_trace, false)});
        ASSERT_NE(graph, nullptr) << backend_name(dev);

        // Replay with a DIFFERENT input. Correct baked constant is 5, so
        // replay == 5 + 10 == 15. The pre-fix bug baked c's post-mutation value
        // (5 + 2 == 7), yielding replay == 7 + 10 == 17.
        Tensor x_new = full({4}, 10.0f, DType::Float32, dev);
        auto outs = graph->forward({Variable(x_new, false)});
        ASSERT_EQ(outs.size(), 1u) << backend_name(dev);
        Tensor replay = outs[0].tensor();

        dev.synchronize();

        Tensor stale = full({4}, 17.0f, DType::Float32, dev);
        EXPECT_FALSE(tensors_close(stale, replay, 1e-5f, 1e-5f))
            << "captured constant baked its POST-mutation value (replay == 17)";
        Tensor expected = full({4}, 15.0f, DType::Float32, dev);
        EXPECT_TENSORS_CLOSE(expected, replay, 1e-5f, 1e-5f);
    }
}

// JIT-F031: control flow (cond / while_loop) must execute correctly on EVERY
// available backend, not just CPU — exercising the host-side widened predicate
// readback (JIT-F008) and the loop trip count on backend-resident tensors.
TEST(JitControlFlowBackends, CondAndWhileLoopAllBackends) {
    using namespace tenzor::jit;
    for (const auto& dev : get_available_backends()) {
        SCOPED_TRACE(backend_name(dev));
        auto x = Variable(full({4}, 3.0f, DType::Float32, dev), false);

        auto out_t = cond(full({1}, 1.0f, DType::Float32, dev),
            [](const Variable& in) { return in + in; },          // then -> 6
            [](const Variable& in) { return tenzor::neg(in); },  // else -> -3
            x);
        auto rt = out_t.tensor().to(Device::cpu());
        for (int i = 0; i < 4; ++i) EXPECT_NEAR(rt.data<float>()[i], 6.0f, 1e-5f);

        auto out_f = cond(full({1}, 0.0f, DType::Float32, dev),
            [](const Variable& in) { return in + in; },
            [](const Variable& in) { return tenzor::neg(in); },
            x);
        auto rf = out_f.tensor().to(Device::cpu());
        for (int i = 0; i < 4; ++i) EXPECT_NEAR(rf.data<float>()[i], -3.0f, 1e-5f);

        auto counter = Variable(full({1}, 0.0f, DType::Float32, dev), false);
        auto res = while_loop(3,
            [dev](const std::vector<Variable>&) -> Tensor {
                return ones({1}, DType::Float32, dev);  // always-true cond
            },
            [dev](const std::vector<Variable>& st) -> std::vector<Variable> {
                return {st[0] + Variable(ones({1}, DType::Float32, dev), false)};
            },
            {counter});
        ASSERT_EQ(res.size(), 1u);
        auto rc = res[0].tensor().to(Device::cpu());
        EXPECT_NEAR(rc.data<float>()[0], 3.0f, 1e-5f);  // three +1 increments
    }
}

// JIT-R165: a NaN condition tensor must take the SAME branch on every
// backend as it does on CPU -- NaN != 0.0 is true under IEEE-754, so a NaN
// condition is "truthy" (then-branch / loop-continue). This guards the
// CPU-first, Float64-widening cast order (control_flow.hpp's
// tensor_condition_to_bool, shared by cond()/while_loop()/the scripted `if`/
// the compiled-graph interpreter) against a regression where a device-side
// cast canonicalizes NaN to 0 on some backend, silently flipping the branch
// taken vs CPU. The existing CondAndWhileLoopAllBackends test above only
// ever used clean 0.0f/1.0f conditions, leaving this exact class of bug
// undetectable on GPU.
TEST(JitControlFlowBackends, CondAndWhileLoopNanConditionAllBackends) {
    using namespace tenzor::jit;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const auto& dev : get_available_backends()) {
        SCOPED_TRACE(backend_name(dev));
        auto x = Variable(full({4}, 3.0f, DType::Float32, dev), false);

        // Float64 (not Float32): tensor_condition_to_bool's CPU-first cast
        // order matters only when the condition is a wider dtype that would
        // otherwise be NARROWED on the device before transfer -- an
        // already-Float32 NaN survives a same-dtype no-op cast regardless of
        // order, so it wouldn't exercise the risk this test targets.
        // NaN condition must take the THEN branch, matching CPU (NaN != 0).
        auto out = cond(full({1}, nan, DType::Float64, dev),
            [](const Variable& in) { return in + in; },          // then -> 6
            [](const Variable& in) { return tenzor::neg(in); },  // else -> -3
            x);
        auto rt = out.tensor().to(Device::cpu());
        for (int i = 0; i < 4; ++i) EXPECT_NEAR(rt.data<float>()[i], 6.0f, 1e-5f);

        // A while_loop whose predicate is always NaN must run all max_iter
        // iterations (never exit early as if NaN were falsy).
        auto counter = Variable(full({1}, 0.0f, DType::Float32, dev), false);
        auto res = while_loop(3,
            [dev, nan](const std::vector<Variable>&) -> Tensor {
                return full({1}, nan, DType::Float64, dev);
            },
            [dev](const std::vector<Variable>& st) -> std::vector<Variable> {
                return {st[0] + Variable(ones({1}, DType::Float32, dev), false)};
            },
            {counter});
        ASSERT_EQ(res.size(), 1u);
        auto rc = res[0].tensor().to(Device::cpu());
        EXPECT_NEAR(rc.data<float>()[0], 3.0f, 1e-5f);  // ran all 3 iterations
    }
}
