#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class HooksTest : public BackendTest {};

// ============================================================================
// Variable backward hooks
// ============================================================================

TEST_P(HooksTest, RegisterAndExecuteBackwardHook) {
    auto a = Variable(ones({2, 2}, DType::Float32, device), true);
    auto b = Variable(ones({2, 2}, DType::Float32, device), true);

    bool hook_called = false;
    a.register_hook([&hook_called](const Tensor& grad) {
        hook_called = true;
        return grad;
    });

    auto c = a + b;
    c.backward(ones({2, 2}, DType::Float32, device));

    EXPECT_TRUE(hook_called) << "Backward hook was not called on " << device.to_string();
}

TEST_P(HooksTest, HookModifiesGradient) {
    auto a = Variable(ones({2, 2}, DType::Float32, device), true);
    auto b = Variable(ones({2, 2}, DType::Float32, device), true);

    // Hook that doubles the gradient
    a.register_hook([](const Tensor& grad) {
        auto two = full({2, 2}, 2.0f, grad.dtype(), grad.device());
        return mul(grad, two);
    });

    auto c = a + b;
    c.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(a.has_grad());
    auto grad = a.grad().value().to(Device::cpu());
    auto* gp = grad.data<float>();
    // Original gradient is 1.0, hook doubles it to 2.0
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(gp[i], 2.0f);
    }
}

TEST_P(HooksTest, MultipleHooksExecuteInOrder) {
    auto a = Variable(ones({2, 2}, DType::Float32, device), true);
    auto b = Variable(ones({2, 2}, DType::Float32, device), true);

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
    c.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_EQ(execution_order.size(), 3u);
    EXPECT_EQ(execution_order[0], 1);
    EXPECT_EQ(execution_order[1], 2);
    EXPECT_EQ(execution_order[2], 3);
}

TEST_P(HooksTest, UnregisterHook) {
    auto a = Variable(ones({2, 2}, DType::Float32, device), true);
    auto b = Variable(ones({2, 2}, DType::Float32, device), true);

    int call_count = 0;
    size_t hook_id = a.register_hook([&call_count](const Tensor& grad) {
        call_count++;
        return grad;
    });

    auto c = a + b;
    c.backward(ones({2, 2}, DType::Float32, device));
    EXPECT_EQ(call_count, 1);

    // Unregister hook
    bool removed = a.unregister_hook(hook_id);
    EXPECT_TRUE(removed);

    // Reset grad and run backward again
    a.zero_grad();
    b.zero_grad();
    auto c2 = a + b;
    c2.backward(ones({2, 2}, DType::Float32, device));
    EXPECT_EQ(call_count, 1);  // Hook should not fire again
}

TEST_P(HooksTest, HookOnNonLeaf) {
    auto a = Variable(ones({2, 2}, DType::Float32, device), true);
    auto b = Variable(ones({2, 2}, DType::Float32, device), true);

    auto c = a + b;  // c is non-leaf

    bool hook_called = false;
    c.register_hook([&hook_called](const Tensor& grad) {
        hook_called = true;
        return grad;
    });

    auto d = c + a;
    d.backward(ones({2, 2}, DType::Float32, device));

    // Hook on intermediate variable should be called
    EXPECT_TRUE(hook_called);
}

// ============================================================================
// Module forward hooks
// ============================================================================

TEST_P(HooksTest, ForwardPreHook) {
    auto linear = std::make_shared<nn::Linear>(4, 2);

    bool pre_hook_called = false;
    linear->register_forward_pre_hook([&pre_hook_called](nn::Module*, const Variable&) {
        pre_hook_called = true;
    });

    auto input = Variable(ones({1, 4}, DType::Float32, device), false);
    auto output = linear->forward(input);

    EXPECT_TRUE(pre_hook_called);
}

TEST_P(HooksTest, ForwardPostHook) {
    auto linear = std::make_shared<nn::Linear>(4, 2);

    bool post_hook_called = false;
    linear->register_forward_post_hook(
        [&post_hook_called](nn::Module*, const Variable& input, const Variable& output) {
            post_hook_called = true;
        });

    auto input = Variable(ones({1, 4}, DType::Float32, device), false);
    auto output = linear->forward(input);

    EXPECT_TRUE(post_hook_called);
}

TEST_P(HooksTest, RemoveModuleHook) {
    auto linear = std::make_shared<nn::Linear>(4, 2);

    int call_count = 0;
    size_t hook_id = linear->register_forward_post_hook(
        [&call_count](nn::Module*, const Variable&, const Variable&) {
            call_count++;
        });

    auto input = Variable(ones({1, 4}, DType::Float32, device), false);
    linear->forward(input);
    EXPECT_EQ(call_count, 1);

    linear->remove_hook(hook_id);
    linear->forward(input);
    EXPECT_EQ(call_count, 1);  // Should not increment after removal
}

// ============================================================================
// Instantiate for CPU backend
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    CPU, HooksTest,
    ::testing::Values("cpu"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);
