/**
 * @file test_optimizer_step_hooks.cpp
 * @brief Audit G.10: Optimizer post-step hooks fire after every step()
 *        call and are usable for pruning auto-reapply.
 */

#include <gtest/gtest.h>

#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"

#include <atomic>

using namespace tenzor;

namespace {

class OptimizerStepHookTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(OptimizerStepHookTest, HookFiresAfterStep) {
    // Tiny SGD with a single parameter that has a gradient.
    auto p_tensor = tenzor::zeros({3}, DType::Float32, Device::cpu());
    auto param = std::make_shared<Variable>(p_tensor, /*requires_grad=*/true);
    auto grad = tenzor::ones({3}, DType::Float32, Device::cpu());
    param->set_grad(grad);

    optim::SGD sgd({param}, /*lr=*/0.1);

    std::atomic<int> hook_calls{0};
    auto id = sgd.register_post_step_hook([&]() { ++hook_calls; });
    EXPECT_GT(id, 0u);

    sgd.step();
    EXPECT_EQ(hook_calls.load(), 1);
    sgd.step();
    EXPECT_EQ(hook_calls.load(), 2);

    // Remove and confirm no further firings.
    EXPECT_TRUE(sgd.remove_post_step_hook(id));
    sgd.step();
    EXPECT_EQ(hook_calls.load(), 2) << "hook fired after removal";
}

TEST_F(OptimizerStepHookTest, RemoveUnknownIdReturnsFalse) {
    auto p_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto param = std::make_shared<Variable>(p_tensor, /*requires_grad=*/true);
    optim::SGD sgd({param}, 0.1);
    EXPECT_FALSE(sgd.remove_post_step_hook(/*hook_id=*/9999));
}

TEST_F(OptimizerStepHookTest, MultipleHooksFireInRegistrationOrder) {
    auto p_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto param = std::make_shared<Variable>(p_tensor, /*requires_grad=*/true);
    param->set_grad(tenzor::ones({1}, DType::Float32, Device::cpu()));

    optim::SGD sgd({param}, 0.1);

    std::vector<int> order;
    sgd.register_post_step_hook([&]() { order.push_back(1); });
    sgd.register_post_step_hook([&]() { order.push_back(2); });
    sgd.register_post_step_hook([&]() { order.push_back(3); });

    sgd.step();
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], 1);
    EXPECT_EQ(order[1], 2);
    EXPECT_EQ(order[2], 3);
}

TEST_F(OptimizerStepHookTest, PruningMaskRemainsZeroAfterStep) {
    // Audit G.10: the key motivating use case — a pruning mask that
    // would otherwise drift back to non-zero after each gradient step.
    //
    // Setup: a parameter where indices [1, 3] should stay zero. Without
    // any auto-reapply, an SGD step with grad=ones moves every value by
    // -lr (so indices [1, 3] become -0.1, not zero). With the post-step
    // hook re-masking them, they stay at zero.
    auto p_tensor = tenzor::zeros({5}, DType::Float32, Device::cpu());
    auto param = std::make_shared<Variable>(p_tensor, /*requires_grad=*/true);
    param->set_grad(tenzor::ones({5}, DType::Float32, Device::cpu()));

    optim::SGD sgd({param}, /*lr=*/0.1);

    // Manual minimal hook (we exercise the optimizer hook contract here;
    // the full register_pruning_auto_reapply path is exercised via the
    // pruning module's integration tests).
    sgd.register_post_step_hook([&]() {
        auto* data = param->tensor().data<float>();
        data[1] = 0.0f;
        data[3] = 0.0f;
    });

    sgd.step();

    const auto* d = param->tensor().data<float>();
    EXPECT_NEAR(d[0], -0.1f, 1e-6);
    EXPECT_FLOAT_EQ(d[1], 0.0f) << "mask must keep index 1 at zero";
    EXPECT_NEAR(d[2], -0.1f, 1e-6);
    EXPECT_FLOAT_EQ(d[3], 0.0f) << "mask must keep index 3 at zero";
    EXPECT_NEAR(d[4], -0.1f, 1e-6);
}

}  // namespace
