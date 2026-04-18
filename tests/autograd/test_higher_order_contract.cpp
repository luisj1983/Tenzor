/**
 * @file test_higher_order_contract.cpp
 * @brief Locks in the HigherOrderGradMode contract for stub backwards.
 *
 * The existing test suites (test_higher_order_gradients.cpp,
 * test_higher_order_nn.cpp, test_higher_order_activations.cpp) verify the
 * "happy path" — ops that have a real `backward_with_variables` produce
 * correct second-order gradients. This file does the opposite: it verifies
 * that ops flagged as `is_higher_order_stub()` honor the
 * HigherOrderGradMode contract — Error throws, Warn logs + increments the
 * disconnection counter. Without this, a regression that accidentally
 * re-flags a real implementation as a stub would silently corrupt
 * create_graph=true for that op.
 *
 * Single-process, CPU-only by design: the contract is engine-level and does
 * not vary across backends.
 */

#include <gtest/gtest.h>

#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

namespace {

using namespace tenzor;

class HigherOrderContractTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override {
        saved_mode_ = get_higher_order_grad_mode();
        reset_higher_order_disconnection_count();
    }
    void TearDown() override {
        set_higher_order_grad_mode(saved_mode_);
    }
    HigherOrderGradMode saved_mode_ = HigherOrderGradMode::Error;
};

// MaxPool2d is flagged STRUCTURAL_ZERO_STUB (argmax selection has a
// structurally-zero second derivative). It's a stable, cheap stub op to
// anchor the contract test against.

TEST_F(HigherOrderContractTest, WarnMode_LogsAndDisconnects) {
    set_higher_order_grad_mode(HigherOrderGradMode::Warn);
    reset_higher_order_disconnection_count();

    auto x = Variable(randn({1, 1, 4, 4}, DType::Float32, Device::cpu()), true);
    nn::MaxPool2d pool(2, 2);
    auto y = pool.forward(x);
    auto loss = tenzor::sum(y);

    EXPECT_NO_THROW(
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true));
    EXPECT_GE(higher_order_disconnection_count(), 1u)
        << "Warn mode should bump the disconnection counter";
}

TEST_F(HigherOrderContractTest, ErrorMode_ThrowsForStubOp) {
    set_higher_order_grad_mode(HigherOrderGradMode::Error);

    auto x = Variable(randn({1, 1, 4, 4}, DType::Float32, Device::cpu()), true);
    nn::MaxPool2d pool(2, 2);
    auto y = pool.forward(x);
    auto loss = tenzor::sum(y);

    EXPECT_THROW(
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true),
        std::runtime_error)
        << "Error mode must throw when a stub op is reached under create_graph=true";
}

// A real (non-stub) op must NOT trip the contract path — verifies we're
// detecting stubs specifically, not firing on every create_graph=true call.

TEST_F(HigherOrderContractTest, ErrorMode_AllowsRealBackwardWithVariables) {
    set_higher_order_grad_mode(HigherOrderGradMode::Error);
    reset_higher_order_disconnection_count();

    auto a = Variable(randn({4}, DType::Float32, Device::cpu()), true);
    auto b = Variable(randn({4}, DType::Float32, Device::cpu()), true);
    auto y = a * b;                 // MulBackward has a real backward_with_variables
    auto loss = tenzor::sum(y);

    EXPECT_NO_THROW(
        loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true));
    EXPECT_EQ(higher_order_disconnection_count(), 0u)
        << "Real bwv ops must not bump the disconnection counter";
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try { tenzor::initialize(); } catch (...) {}
    return RUN_ALL_TESTS();
}
