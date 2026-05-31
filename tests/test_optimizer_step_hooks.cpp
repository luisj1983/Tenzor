/**
 * @file test_optimizer_step_hooks.cpp
 * @brief Audit G.10: Optimizer post-step hooks fire after every step()
 *        call and are usable for pruning auto-reapply.
 */

#include <gtest/gtest.h>

#include "backend_test_fixture.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/tenzor.hpp"

#include <atomic>
#include <vector>

using namespace tenzor;

namespace {

class OptimizerStepHookTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(OptimizerStepHookTest, HookFiresAfterStep) {
    // Tiny SGD with a single parameter that has a gradient.
    auto p_tensor = tenzor::zeros({3}, DType::Float32, device);
    auto param = std::make_shared<Variable>(p_tensor, /*requires_grad=*/true);
    auto grad = tenzor::ones({3}, DType::Float32, device);
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

TEST_P(OptimizerStepHookTest, RemoveUnknownIdReturnsFalse) {
    auto p_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto param = std::make_shared<Variable>(p_tensor, /*requires_grad=*/true);
    optim::SGD sgd({param}, 0.1);
    EXPECT_FALSE(sgd.remove_post_step_hook(/*hook_id=*/9999));
}

TEST_P(OptimizerStepHookTest, MultipleHooksFireInRegistrationOrder) {
    auto p_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto param = std::make_shared<Variable>(p_tensor, /*requires_grad=*/true);
    param->set_grad(tenzor::ones({1}, DType::Float32, device));

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

TEST_P(OptimizerStepHookTest, PruningMaskRemainsZeroAfterStep) {
    // Audit G.10: the key motivating use case — a pruning mask that
    // would otherwise drift back to non-zero after each gradient step.
    //
    // Setup: a parameter where indices [1, 3] should stay zero. Without
    // any auto-reapply, an SGD step with grad=ones moves every value by
    // -lr (so indices [1, 3] become -0.1, not zero). With the post-step
    // hook re-masking them, they stay at zero.
    auto p_tensor = tenzor::zeros({5}, DType::Float32, device);
    auto param = std::make_shared<Variable>(p_tensor, /*requires_grad=*/true);
    param->set_grad(tenzor::ones({5}, DType::Float32, device));

    optim::SGD sgd({param}, /*lr=*/0.1);

    // Manual minimal hook (we exercise the optimizer hook contract here;
    // the full register_pruning_auto_reapply path is exercised via the
    // pruning module's integration tests). Re-mask indices [1, 3] to zero
    // by multiplying the parameter tensor by a device-resident mask, so the
    // hook stays correct on every backend (no raw host pointer access).
    float mask_host[5] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    auto mask = tenzor::from_data(mask_host, {5}, device);
    sgd.register_post_step_hook([&]() {
        param->tensor() = mul(param->tensor(), mask);
    });

    sgd.step();

    // Host reads: bring the parameter to CPU before touching raw data.
    auto result_cpu = param->tensor().cpu();
    const auto* d = result_cpu.data<float>();
    EXPECT_NEAR(d[0], -0.1f, 1e-6);
    EXPECT_FLOAT_EQ(d[1], 0.0f) << "mask must keep index 1 at zero";
    EXPECT_NEAR(d[2], -0.1f, 1e-6);
    EXPECT_FLOAT_EQ(d[3], 0.0f) << "mask must keep index 3 at zero";
    EXPECT_NEAR(d[4], -0.1f, 1e-6);
}

INSTANTIATE_BACKEND_TESTS(OptimizerStepHookTest);

}  // namespace
