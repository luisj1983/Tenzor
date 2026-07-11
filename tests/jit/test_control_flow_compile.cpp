/**
 * @file test_control_flow_compile.cpp
 * @brief Tests for JIT-compatible control flow (cond, while_loop)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/control_flow.hpp>
#include <tenzor/jit/compile.hpp>

using namespace tenzor;
using namespace tenzor::jit;

class ControlFlowTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
    void TearDown() override { tenzor::finalize(); }
};
static auto* _env = ::testing::AddGlobalTestEnvironment(new ControlFlowTestEnv());

TEST(ControlFlowTest, CondEagerTrueBranch) {
    auto cond_tensor = full({1}, 1.0f, DType::Float32, Device::cpu());
    auto input = Variable(randn({2, 3}, DType::Float32, Device::cpu()), false);

    auto result = cond(cond_tensor,
        [](const Variable& x) -> Variable { return x * Variable(full({1}, 2.0f, DType::Float32), false); },
        [](const Variable& x) -> Variable { return x * Variable(full({1}, 0.5f, DType::Float32), false); },
        input);

    // True branch: x * 2
    auto expected = input.tensor() * 2.0f;
    auto diff = tenzor::max(tenzor::abs(result.tensor() - expected));
    EXPECT_LT(diff.data<float>()[0], 1e-6f);
}

TEST(ControlFlowTest, CondEagerFalseBranch) {
    auto cond_tensor = full({1}, 0.0f, DType::Float32, Device::cpu());
    auto input = Variable(randn({2, 3}, DType::Float32, Device::cpu()), false);

    auto result = cond(cond_tensor,
        [](const Variable& x) -> Variable { return x * Variable(full({1}, 2.0f, DType::Float32), false); },
        [](const Variable& x) -> Variable { return x * Variable(full({1}, 0.5f, DType::Float32), false); },
        input);

    // False branch: x * 0.5
    auto expected = input.tensor() * 0.5f;
    auto diff = tenzor::max(tenzor::abs(result.tensor() - expected));
    EXPECT_LT(diff.data<float>()[0], 1e-6f);
}

TEST(ControlFlowTest, CondMultiOutput) {
    auto cond_tensor = full({1}, 1.0f, DType::Float32, Device::cpu());
    auto input = Variable(randn({2, 3}, DType::Float32, Device::cpu()), false);

    auto results = cond(cond_tensor,
        [](const std::vector<Variable>& args) -> std::vector<Variable> {
            return {args[0], args[0]};
        },
        [](const std::vector<Variable>& args) -> std::vector<Variable> {
            return {args[0]};
        },
        {input});

    EXPECT_EQ(results.size(), 2);
}

TEST(ControlFlowTest, WhileLoopEagerExecution) {
    auto counter = Variable(full({1}, 0.0f, DType::Float32, Device::cpu()), false);
    auto limit = Variable(full({1}, 5.0f, DType::Float32, Device::cpu()), false);

    auto results = while_loop(100,
        [](const std::vector<Variable>& state) -> Tensor {
            // Continue while counter < limit
            auto diff = state[1].tensor() - state[0].tensor();
            return diff;  // positive = continue
        },
        [](const std::vector<Variable>& state) -> std::vector<Variable> {
            // Increment counter
            auto new_counter = state[0] + Variable(full({1}, 1.0f, DType::Float32), false);
            return {new_counter, state[1]};
        },
        {counter, limit});

    EXPECT_EQ(results.size(), 2);
    // Counter should reach 5
    auto final_count = results[0].tensor().data<float>()[0];
    EXPECT_NEAR(final_count, 5.0f, 1e-6f);
}

TEST(ControlFlowTest, WhileLoopMaxIterations) {
    auto state = Variable(full({1}, 0.0f, DType::Float32, Device::cpu()), false);

    auto results = while_loop(3,  // Only 3 iterations max
        [](const std::vector<Variable>& s) -> Tensor {
            return full({1}, 1.0f, DType::Float32, Device::cpu());  // Always true
        },
        [](const std::vector<Variable>& s) -> std::vector<Variable> {
            return {s[0] + Variable(full({1}, 1.0f, DType::Float32), false)};
        },
        {state});

    // Should stop at 3 due to max_iter
    auto final_val = results[0].tensor().data<float>()[0];
    EXPECT_NEAR(final_val, 3.0f, 1e-6f);
}

// JIT-R003 regression: LoopUnrollingPass used to unroll a while_loop's body
// unconditionally trip_count times, completely ignoring the runtime cond —
// silently running body iterations PAST the point where eager/interpreted
// execution would have stopped early (the documented while_loop early-exit
// contract, exercised by WhileLoopEagerExecution above). Use max_iter=8
// (within LoopUnrollingPass's default max_unroll_ so it actually fires) with
// a cond that goes false after 3 iterations; the compiled result must match
// eager exactly, not run all 8 iterations.
TEST(ControlFlowTest, WhileLoopUnrollRespectsEarlyExit) {
    auto fn = [](const Variable& start) -> Variable {
        auto limit = Variable(full({1}, 3.0f, DType::Float32, Device::cpu()), false);
        auto results = while_loop(8,
            [](const std::vector<Variable>& state) -> Tensor {
                // Continue while counter < limit (matches
                // WhileLoopEagerExecution's convention above).
                return (state[1].tensor() - state[0].tensor());
            },
            [](const std::vector<Variable>& state) -> std::vector<Variable> {
                auto one = Variable(full({1}, 1.0f, DType::Float32), false);
                return {state[0] + one, state[1]};
            },
            {start, limit});
        return results[0];
    };

    auto compiled = jit::compile(fn);
    auto input = Variable(full({1}, 0.0f, DType::Float32, Device::cpu()), false);

    auto eager_result = fn(input).tensor();
    ASSERT_NEAR(eager_result.data<float>()[0], 3.0f, 1e-6f)
        << "test setup: eager while_loop should stop at counter==limit==3";

    // First call is a cache MISS: CompiledFunction::operator() returns the
    // raw trace-time result from trace_loop's single (subgraph-building)
    // body run, not a replay of the compiled/optimized graph — that's an
    // unrelated, pre-existing property of tracing a data-dependent loop, not
    // what R003 is about. Discard it and call again so the SECOND call is a
    // cache HIT, which routes through CompiledModule::forward -> the actual
    // LoopUnrollingPass-optimized graph — the thing under test here.
    (void)compiled(input);
    ASSERT_GE(compiled.num_cached(), 1u)
        << "while_loop did not compile/cache a graph — likely graph-broke "
           "and silently fell back to eager for every call";

    auto compiled_result = compiled(input).tensor();
    EXPECT_NEAR(compiled_result.data<float>()[0], 3.0f, 1e-6f)
        << "LoopUnrollingPass ignored the runtime cond and ran the body past "
           "the early-exit point (compiled result should match eager: 3, "
           "not max_iter: 8)";
}

TEST(ControlFlowTest, GuardNodeOpTypeExists) {
    // Verify GuardNode is in the OpType enum
    auto guard = OpType::GuardNode;
    EXPECT_NE(static_cast<int>(guard), static_cast<int>(OpType::ShapeGuard));
}
