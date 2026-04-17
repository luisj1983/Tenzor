/**
 * @file test_hooks_multidtype.cpp
 * @brief Multi-backend, multi-dtype tests for autograd hooks
 *
 * Converted from test_hooks.cpp to exercise all backend + dtype combinations.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class HooksMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Variable backward hooks
// ============================================================================

TEST_P(HooksMultiDTypeTest, RegisterAndExecuteBackwardHook) {
    auto a = Variable(ones({2, 2}, dtype(), device()), true);
    auto b = Variable(ones({2, 2}, dtype(), device()), true);

    bool hook_called = false;
    a.register_hook([&hook_called](const Tensor& grad) {
        hook_called = true;
        return grad;
    });

    auto c = a + b;
    c.backward(ones({2, 2}, dtype(), device()));

    EXPECT_TRUE(hook_called) << "Backward hook was not called on " << device().to_string();
}

TEST_P(HooksMultiDTypeTest, HookModifiesGradient) {
    auto a = Variable(ones({2, 2}, dtype(), device()), true);
    auto b = Variable(ones({2, 2}, dtype(), device()), true);

    // Hook that doubles the gradient
    a.register_hook([](const Tensor& grad) {
        auto two = full({2, 2}, 2.0f, grad.dtype(), grad.device());
        return mul(grad, two);
    });

    auto c = a + b;
    c.backward(ones({2, 2}, dtype(), device()));

    ASSERT_TRUE(a.has_grad());
    auto grad = a.grad().value().to(Device::cpu()).to(DType::Float32);
    auto* gp = grad.data<float>();
    // Original gradient is 1.0, hook doubles it to 2.0
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(gp[i], 2.0f, atol());
    }
}

TEST_P(HooksMultiDTypeTest, MultipleHooksExecuteInOrder) {
    auto a = Variable(ones({2, 2}, dtype(), device()), true);
    auto b = Variable(ones({2, 2}, dtype(), device()), true);

    std::vector<int> execution_order;

    a.register_hook([&execution_order](const Tensor& grad) {
        execution_order.push_back(1);
        return grad;
    });
    a.register_hook([&execution_order](const Tensor& grad) {
        execution_order.push_back(2);
        return grad;
    });
    a.register_hook([&execution_order](const Tensor& grad) {
        execution_order.push_back(3);
        return grad;
    });

    auto c = a + b;
    c.backward(ones({2, 2}, dtype(), device()));

    ASSERT_EQ(execution_order.size(), 3u);
    EXPECT_EQ(execution_order[0], 1);
    EXPECT_EQ(execution_order[1], 2);
    EXPECT_EQ(execution_order[2], 3);
}

TEST_P(HooksMultiDTypeTest, UnregisterHook) {
    auto a = Variable(ones({2, 2}, dtype(), device()), true);
    auto b = Variable(ones({2, 2}, dtype(), device()), true);

    int call_count = 0;
    size_t hook_id = a.register_hook([&call_count](const Tensor& grad) {
        call_count++;
        return grad;
    });

    auto c = a + b;
    c.backward(ones({2, 2}, dtype(), device()));
    EXPECT_EQ(call_count, 1);

    // Unregister hook
    bool removed = a.unregister_hook(hook_id);
    EXPECT_TRUE(removed);

    // Reset grad and run backward again
    a.zero_grad();
    b.zero_grad();
    auto c2 = a + b;
    c2.backward(ones({2, 2}, dtype(), device()));
    EXPECT_EQ(call_count, 1);  // Hook should not fire again
}

TEST_P(HooksMultiDTypeTest, HookOnNonLeaf) {
    auto a = Variable(ones({2, 2}, dtype(), device()), true);
    auto b = Variable(ones({2, 2}, dtype(), device()), true);

    auto c = a + b;  // c is non-leaf

    bool hook_called = false;
    c.register_hook([&hook_called](const Tensor& grad) {
        hook_called = true;
        return grad;
    });

    auto d = c + a;
    d.backward(ones({2, 2}, dtype(), device()));

    // Hook on intermediate variable should be called
    EXPECT_TRUE(hook_called);
}

// ============================================================================
// Module forward hooks
// ============================================================================

TEST_P(HooksMultiDTypeTest, ForwardPreHook) {
    auto linear = std::make_shared<nn::Linear>(4, 2);
    convert_model(linear);

    bool pre_hook_called = false;
    linear->register_forward_pre_hook([&pre_hook_called](nn::Module*, const Variable&) {
        pre_hook_called = true;
    });

    auto input = Variable(ones({1, 4}, dtype(), device()), false);
    auto output = linear->forward(input);

    EXPECT_TRUE(pre_hook_called);
}

TEST_P(HooksMultiDTypeTest, ForwardPostHook) {
    auto linear = std::make_shared<nn::Linear>(4, 2);
    convert_model(linear);

    bool post_hook_called = false;
    linear->register_forward_post_hook(
        [&post_hook_called](nn::Module*, const Variable& input, const Variable& output) {
            post_hook_called = true;
        });

    auto input = Variable(ones({1, 4}, dtype(), device()), false);
    auto output = linear->forward(input);

    EXPECT_TRUE(post_hook_called);
}

TEST_P(HooksMultiDTypeTest, RemoveModuleHook) {
    auto linear = std::make_shared<nn::Linear>(4, 2);
    convert_model(linear);

    int call_count = 0;
    size_t hook_id = linear->register_forward_post_hook(
        [&call_count](nn::Module*, const Variable&, const Variable&) {
            call_count++;
        });

    auto input = Variable(ones({1, 4}, dtype(), device()), false);
    linear->forward(input);
    EXPECT_EQ(call_count, 1);

    linear->remove_hook(hook_id);
    linear->forward(input);
    EXPECT_EQ(call_count, 1);  // Should not increment after removal
}

// ============================================================================
// Instantiate for all available backends and dtypes
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HooksMultiDTypeTest);
