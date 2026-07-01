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

TEST_P(HooksTest, NonLeafHookPropagatesDownstream) {
    // A hook on a non-leaf must transform the gradient that flows THROUGH it to
    // upstream leaves (PyTorch semantics) — not merely fire. Here c = a * 2, so
    // d c/d a = 2. A hook on c that scales the incoming gradient by 10 must make
    // a.grad = 2 * (10 * 1) = 20. The old engine applied the hook only to c's
    // own (discarded) accumulation and propagated the un-hooked gradient,
    // leaving a.grad = 2.
    auto a = Variable(ones({2, 2}, DType::Float32, device), true);
    auto two = Variable(full({2, 2}, 2.0f, DType::Float32, device), false);
    auto c = a * two;  // non-leaf intermediate, dc/da = 2

    c.register_hook([&](const Tensor& grad) {
        auto ten = full({2, 2}, 10.0f, grad.dtype(), grad.device());
        return mul(grad, ten);
    });

    // c must be an INPUT to a downstream op so the hook fires on the gradient
    // flowing through it (dd/dc = 1), which then reaches the leaf a.
    auto zero = Variable(zeros({2, 2}, DType::Float32, device), false);
    auto d = c + zero;
    d.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(a.has_grad());
    auto grad = a.grad().value().to(Device::cpu());
    auto* gp = grad.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(gp[i], 20.0f);
    }
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
// Phase 7 expansion: exception safety + leak + stress tests
// ============================================================================

// Hook that throws must not leave the autograd graph in a corrupt state.
// After catching the exception we should still be able to run another
// forward+backward with no residual effect from the aborted hook.
TEST_P(HooksTest, HookThrowDoesNotCorruptGraph) {
    auto a = Variable(ones({2, 2}, DType::Float32, device), true);

    a.register_hook([](const Tensor&) -> Tensor {
        throw std::runtime_error("hook-intentional");
    });

    auto b = a * 2.0f;
    bool caught = false;
    try {
        b.backward(ones({2, 2}, DType::Float32, device));
    } catch (const std::exception&) {
        caught = true;
    }
    EXPECT_TRUE(caught) << "throwing hook must propagate";

    // Register a NEW variable with a well-behaved hook. If the throwing
    // hook left anything registered or broke the tape, this will crash
    // or see the stale hook's effects.
    auto c = Variable(ones({2, 2}, DType::Float32, device), true);
    int counter = 0;
    c.register_hook([&counter](const Tensor& g) {
        counter++;
        return g;
    });
    auto d = c * 3.0f;
    d.backward(ones({2, 2}, DType::Float32, device));
    EXPECT_EQ(counter, 1) << "recovery hook must fire exactly once";
}

// Registering and immediately removing many hooks should not leak memory
// or leave stale hooks attached. This is a cheap stress check (1000
// hooks — 10k would slow the suite) that would catch an obvious leak.
TEST_P(HooksTest, RepeatedRegisterRemoveDoesNotAccumulate) {
    auto a = Variable(ones({4}, DType::Float32, device), true);

    // Baseline: register 1 hook, backward once, observe 1 call.
    int calls = 0;
    auto hook_id = a.register_hook([&calls](const Tensor& g) {
        calls++;
        return g;
    });
    // Register+unregister 1000 additional no-op hooks.
    for (int i = 0; i < 1000; ++i) {
        auto id = a.register_hook([](const Tensor& g) { return g; });
        a.unregister_hook(id);
    }

    auto b = a * 2.0f;  // `a` appears once in the forward graph
    b.backward(ones({4}, DType::Float32, device));
    EXPECT_EQ(calls, 1) << "after 1000 register/removes only the surviving hook should fire";

    // Clean up.
    a.unregister_hook(hook_id);
}

// Multiple hooks must fire in registration order — guarded against a
// data-structure-reordering regression.
TEST_P(HooksTest, HooksFireInRegistrationOrder) {
    auto a = Variable(ones({2}, DType::Float32, device), true);

    std::vector<int> sequence;
    a.register_hook([&sequence](const Tensor& g) { sequence.push_back(1); return g; });
    a.register_hook([&sequence](const Tensor& g) { sequence.push_back(2); return g; });
    a.register_hook([&sequence](const Tensor& g) { sequence.push_back(3); return g; });

    auto b = a * 2.0f;
    b.backward(ones({2}, DType::Float32, device));
    ASSERT_EQ(sequence.size(), 3u);
    EXPECT_EQ(sequence[0], 1);
    EXPECT_EQ(sequence[1], 2);
    EXPECT_EQ(sequence[2], 3);
}

// ============================================================================
// Instantiate for all available backends
// ============================================================================

INSTANTIATE_BACKEND_TESTS(HooksTest);
