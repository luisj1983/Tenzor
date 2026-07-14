/**
 * @file test_control_flow.cpp
 * @brief Tests for JIT control flow (If/Loop) tracing
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/jit/control_flow.hpp>

using namespace tenzor;
using namespace tenzor::jit;

class JITControlFlowTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new JITControlFlowTestEnv);

TEST(JITControlFlow, EagerCondTinyFloat64PredicateIsTruthy) {
    // A nonzero Float64 condition below the Float32 denormal floor must NOT be
    // flushed to "false" (JIT-F008): 1e-40 underflows Float32 to 0.0f but is a
    // genuine nonzero predicate, so the then-branch must be taken.
    auto tiny = full({1}, 1e-40, DType::Float64, Device::cpu());
    auto x = Variable(ones({2}, DType::Float32, Device::cpu()), false);
    auto out = cond(
        tiny,
        [](const Variable& in) -> Variable { return in + in; },          // then -> 2
        [](const Variable& in) -> Variable { return tenzor::neg(in); },  // else -> -1
        x);
    auto r = out.tensor().to(Device::cpu());
    EXPECT_NEAR(r.data<float>()[0], 2.0f, 1e-5);
}

TEST(JITControlFlow, TraceIfBasic) {
    Tracer tracer;
    tracer.start_trace();

    auto x = Variable(ones({2, 3}, DType::Float32, Device::cpu()), false);
    auto cond = ones({1}, DType::Float32, Device::cpu());  // true condition

    auto results = tracer.trace_if(
        cond,
        // then branch: double the input
        [](const std::vector<Variable>& inputs) -> std::vector<Variable> {
            return {inputs[0] + inputs[0]};
        },
        // else branch: negate the input
        [](const std::vector<Variable>& inputs) -> std::vector<Variable> {
            return {tenzor::neg(inputs[0])};
        },
        {x}
    );

    ASSERT_EQ(results.size(), 1u);
    // At trace time, returns then-branch output (x + x = 2)
    auto result = results[0].tensor().to(Device::cpu());
    EXPECT_NEAR(result.data<float>()[0], 2.0f, 1e-5);

    tracer.clear();
}

// JIT-R148: assert_no_inplace_on_shared must still catch a GENUINE in-place
// mutation of a tensor shared/carried across a cond() branch (no prior test
// exercised this positive case at all -- only the false-positive fix below
// had regression risk without this).
TEST(JITControlFlow, TraceIfDetectsGenuineInplaceMutationOfSharedTensor) {
    Tracer tracer;
    tracer.start_trace();

    auto x = Variable(ones({2, 3}, DType::Float32, Device::cpu()), false);
    auto other = ones({2, 3}, DType::Float32, Device::cpu());
    auto cond_t = ones({1}, DType::Float32, Device::cpu());

    EXPECT_THROW(
        {
            tracer.trace_if(
                cond_t,
                // then branch: genuinely mutate the shared/carried input `x`
                // in place, via the same hook dispatch_inplace uses.
                [&](const std::vector<Variable>&) -> std::vector<Variable> {
                    Tensor& target = x.tensor();
                    tracer.record_inplace(OpId::AddInplace, target,
                                          std::span<const Tensor>(&other, 1),
                                          OpAttributes{}, nullptr);
                    return {x};
                },
                [](const std::vector<Variable>& inputs) -> std::vector<Variable> {
                    return {inputs[0]};
                },
                {x});
        },
        std::runtime_error);

    tracer.clear();
}

// JIT-R174: Tracer::clear() previously left inplace_remapped_fingerprints_
// unreset, so a fingerprint left over from an EARLIER, unrelated trace on
// the same thread could mask a genuine in-place mutation in a LATER trace
// that happens to touch a same-fingerprint (ptr#dtype#shape#strides#device)
// tensor. This is thread_local production state
// (Tracer::get_instance()) -- unlike every other test in this file, which
// uses a fresh stack-local `Tracer` per test and so never exercises
// cross-trace persistence at all. Reproduced deterministically (no reliance
// on the allocator coincidentally reusing a freed address) by reusing the
// exact same tensor object across two separate, sequential traces.
TEST(JITControlFlow, ClearResetsInplaceFingerprintsAcrossSeparateTraces_JIT174) {
    auto& tracer = Tracer::get_instance();
    tracer.clear();  // clean slate regardless of test execution order

    auto shared = Variable(ones({2, 3}, DType::Float32, Device::cpu()), false);
    auto other = ones({2, 3}, DType::Float32, Device::cpu());

    // Trace #1: a completely unrelated trace that happens to in-place-
    // mutate `shared`, populating inplace_remapped_fingerprints_.
    tracer.start_trace();
    tracer.record_inplace(OpId::AddInplace, shared.tensor(),
                          std::span<const Tensor>(&other, 1), OpAttributes{}, nullptr);
    tracer.clear();

    // Trace #2: a brand-new, separate trace. `shared`'s fingerprint must NOT
    // still be considered "already remapped before this branch" just
    // because trace #1 (now over) remapped it.
    tracer.start_trace();
    auto cond_t2 = ones({1}, DType::Float32, Device::cpu());
    EXPECT_THROW(
        {
            tracer.trace_if(
                cond_t2,
                [&](const std::vector<Variable>&) -> std::vector<Variable> {
                    tracer.record_inplace(OpId::AddInplace, shared.tensor(),
                                          std::span<const Tensor>(&other, 1),
                                          OpAttributes{}, nullptr);
                    return {shared};
                },
                [](const std::vector<Variable>& inputs) -> std::vector<Variable> {
                    return {inputs[0]};
                },
                {shared});
        },
        std::runtime_error)
        << "a fresh trace must detect a genuine in-place mutation even "
           "though an EARLIER, unrelated trace on this thread already "
           "remapped a same-fingerprint tensor (JIT-R174)";

    tracer.clear();
}

TEST(JITControlFlow, TraceLoopBasic) {
    Tracer tracer;
    tracer.start_trace();

    auto x = Variable(ones({2, 3}, DType::Float32, Device::cpu()), false);

    auto results = tracer.trace_loop(
        5,  // max 5 iterations
        // condition: always true (simplified)
        [](const std::vector<Variable>& carried) -> Tensor {
            return ones({1}, DType::Float32, Device::cpu());
        },
        // body: add 1 to carried state
        [](const std::vector<Variable>& carried) -> std::vector<Variable> {
            auto one = Variable(ones_like(carried[0].tensor()), false);
            return {carried[0] + one};
        },
        {x}
    );

    ASSERT_EQ(results.size(), 1u);
    // After trace (single iteration), x + 1 = 2
    auto result = results[0].tensor().to(Device::cpu());
    EXPECT_NEAR(result.data<float>()[0], 2.0f, 1e-5);

    tracer.clear();
}

TEST(JITControlFlow, TraceIfMultipleOutputs) {
    Tracer tracer;
    tracer.start_trace();

    auto x = Variable(ones({3}, DType::Float32, Device::cpu()), false);
    auto y = Variable(tenzor::mul(ones({3}, DType::Float32, Device::cpu()), 2.0), false);
    auto cond = ones({1}, DType::Float32, Device::cpu());

    auto results = tracer.trace_if(
        cond,
        [](const std::vector<Variable>& inputs) -> std::vector<Variable> {
            return {inputs[0] + inputs[1], inputs[0] * inputs[1]};
        },
        [](const std::vector<Variable>& inputs) -> std::vector<Variable> {
            return {inputs[0] - inputs[1], inputs[0] / inputs[1]};
        },
        {x, y}
    );

    ASSERT_EQ(results.size(), 2u);
    tracer.clear();
}

TEST(JITControlFlow, TraceLoopMultipleCarried) {
    Tracer tracer;
    tracer.start_trace();

    auto a = Variable(zeros({2}, DType::Float32, Device::cpu()), false);
    auto b = Variable(ones({2}, DType::Float32, Device::cpu()), false);

    auto results = tracer.trace_loop(
        10,
        [](const std::vector<Variable>&) -> Tensor {
            return ones({1}, DType::Float32, Device::cpu());
        },
        // Fibonacci-like: (a, b) -> (b, a+b)
        [](const std::vector<Variable>& carried) -> std::vector<Variable> {
            return {carried[1], carried[0] + carried[1]};
        },
        {a, b}
    );

    ASSERT_EQ(results.size(), 2u);
    tracer.clear();
}
