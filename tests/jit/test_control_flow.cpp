/**
 * @file test_control_flow.cpp
 * @brief Tests for JIT control flow (If/Loop) tracing
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/tracer.hpp>

using namespace tenzor;
using namespace tenzor::jit;

class JITControlFlowTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new JITControlFlowTestEnv);

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
