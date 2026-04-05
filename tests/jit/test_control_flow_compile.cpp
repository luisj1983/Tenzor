/**
 * @file test_control_flow_compile.cpp
 * @brief Tests for JIT-compatible control flow (cond, while_loop)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/control_flow.hpp>

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

TEST(ControlFlowTest, GuardNodeOpTypeExists) {
    // Verify GuardNode is in the OpType enum
    auto guard = OpType::GuardNode;
    EXPECT_NE(static_cast<int>(guard), static_cast<int>(OpType::ShapeGuard));
}
