/**
 * @file test_custom_op_autograd.cpp
 * @brief Tests for custom op autograd integration
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/custom_op.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class CustomOpAutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Test 1: Custom op without backward (no grad tracking)
TEST_F(CustomOpAutogradTest, ForwardOnlyCustomOp) {
    auto op_id = register_custom_op("test::double_it",
        Device::Type::CPU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0] * 2.0f;
        });

    Variable x(Tensor({2, 3}, DType::Float32, Device::cpu()), true);
    x.tensor().fill_(3.0f);

    // Dispatch through autograd — no backward registered, so no grad tracking
    auto result = dispatch_custom_op(op_id, {x});
    EXPECT_FALSE(result.requires_grad());
    EXPECT_NEAR(result.tensor().data<float>()[0], 6.0f, 1e-5);
}

// Test 2: Custom op with backward — gradient computation
TEST_F(CustomOpAutogradTest, CustomOpWithBackward) {
    // f(x) = x^2, f'(x) = 2x
    auto op_id = register_custom_op_with_backward(
        "test::square",
        Device::Type::CPU,
        // forward: x^2
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0] * inputs[0];
        },
        // backward: d(x^2)/dx = 2x * grad_out
        [](std::span<const Tensor> saved, std::span<const Tensor> grads) -> std::vector<Tensor> {
            return {saved[0] * grads[0] * 2.0f};
        }
    );

    Variable x(Tensor({2, 2}, DType::Float32, Device::cpu()), true);
    x.tensor().fill_(3.0f);

    auto y = dispatch_custom_op(op_id, {x});
    EXPECT_TRUE(y.requires_grad());

    // y = x^2 = 9 per element
    EXPECT_NEAR(y.tensor().data<float>()[0], 9.0f, 1e-5);

    // Backward: sum first to get scalar
    y = sum(y);
    y.backward();

    // dy/dx = 2x = 6 per element
    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu());
    EXPECT_NEAR(grad.data<float>()[0], 6.0f, 1e-5);
}

// Test 3: Custom op with custom save function
TEST_F(CustomOpAutogradTest, CustomSaveForBackward) {
    // f(x) = x^3, save x^2 for backward to compute 3*x^2
    auto op_id = register_custom_op_with_backward(
        "test::cube",
        Device::Type::CPU,
        // forward
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0] * inputs[0] * inputs[0];
        },
        // backward: d(x^3)/dx = 3*x^2 * grad_out (x^2 is saved)
        [](std::span<const Tensor> saved, std::span<const Tensor> grads) -> std::vector<Tensor> {
            return {saved[0] * grads[0] * 3.0f};
        },
        // save: save x^2 instead of x (more memory efficient)
        [](std::span<const Tensor> inputs, const Tensor&) -> std::vector<Tensor> {
            return {inputs[0] * inputs[0]};
        }
    );

    Variable x(Tensor({2}, DType::Float32, Device::cpu()), true);
    x.tensor().fill_(2.0f);

    auto y = dispatch_custom_op(op_id, {x});
    EXPECT_TRUE(y.requires_grad());

    y = sum(y);
    y.backward();

    // dy/dx = 3 * x^2 = 3 * 4 = 12
    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu());
    EXPECT_NEAR(grad.data<float>()[0], 12.0f, 1e-5);
}

// Test 4: No-grad inputs skip backward
TEST_F(CustomOpAutogradTest, NoGradInputsSkipBackward) {
    auto op_id = register_custom_op_with_backward(
        "test::add_one",
        Device::Type::CPU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0] + 1.0f;
        },
        [](std::span<const Tensor> saved, std::span<const Tensor> grads) -> std::vector<Tensor> {
            return {grads[0]};
        }
    );

    // Input without requires_grad
    Variable x(Tensor({3}, DType::Float32, Device::cpu()), false);
    x.tensor().fill_(5.0f);

    auto y = dispatch_custom_op(op_id, {x});
    EXPECT_FALSE(y.requires_grad());
}

// Test 5: Chained custom ops
TEST_F(CustomOpAutogradTest, ChainedCustomOps) {
    // op1: f(x) = 2x, f'(x) = 2
    auto op1 = register_custom_op_with_backward(
        "test::double_v2",
        Device::Type::CPU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0] * 2.0f;
        },
        [](std::span<const Tensor>, std::span<const Tensor> grads) -> std::vector<Tensor> {
            return {grads[0] * 2.0f};
        }
    );

    // op2: g(x) = x + 3, g'(x) = 1
    auto op2 = register_custom_op_with_backward(
        "test::add_three",
        Device::Type::CPU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0] + 3.0f;
        },
        [](std::span<const Tensor>, std::span<const Tensor> grads) -> std::vector<Tensor> {
            return {grads[0]};
        }
    );

    Variable x(Tensor({1}, DType::Float32, Device::cpu()), true);
    x.tensor().fill_(5.0f);

    // y = g(f(x)) = 2x + 3 = 13
    auto y = dispatch_custom_op(op1, {x});
    y = dispatch_custom_op(op2, {y});

    EXPECT_NEAR(y.tensor().data<float>()[0], 13.0f, 1e-5);

    y.backward();

    // dy/dx = g'(f(x)) * f'(x) = 1 * 2 = 2
    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad()->to(Device::cpu());
    EXPECT_NEAR(grad.data<float>()[0], 2.0f, 1e-5);
}

// Test 6: Binary custom op (two inputs)
TEST_F(CustomOpAutogradTest, BinaryCustomOp) {
    // f(a, b) = a * b
    auto op_id = register_custom_op_with_backward(
        "test::custom_mul",
        Device::Type::CPU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0] * inputs[1];
        },
        // backward: d(a*b)/da = b*grad, d(a*b)/db = a*grad
        [](std::span<const Tensor> saved, std::span<const Tensor> grads) -> std::vector<Tensor> {
            return {saved[1] * grads[0], saved[0] * grads[0]};
        }
    );

    Variable a(Tensor({2}, DType::Float32, Device::cpu()), true);
    Variable b(Tensor({2}, DType::Float32, Device::cpu()), true);
    a.tensor().fill_(3.0f);
    b.tensor().fill_(4.0f);

    auto y = dispatch_custom_op(op_id, {a, b});
    EXPECT_TRUE(y.requires_grad());

    y = sum(y);
    y.backward();

    // da = b = 4, db = a = 3
    ASSERT_TRUE(a.has_grad());
    ASSERT_TRUE(b.has_grad());
    EXPECT_NEAR(a.grad()->to(Device::cpu()).data<float>()[0], 4.0f, 1e-5);
    EXPECT_NEAR(b.grad()->to(Device::cpu()).data<float>()[0], 3.0f, 1e-5);
}

// Test 7: Registering same name twice returns same ID
TEST_F(CustomOpAutogradTest, IdempotentRegistration) {
    auto id1 = register_custom_op("test::idempotent",
        Device::Type::CPU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0];
        });

    auto id2 = register_custom_op("test::idempotent",
        Device::Type::CPU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0];
        });

    EXPECT_EQ(id1, id2);
}

// Test 8: Mixed custom op with built-in ops
TEST_F(CustomOpAutogradTest, MixedWithBuiltinOps) {
    // Custom op: f(x) = x^2
    auto square_op = register_custom_op_with_backward(
        "test::square_v2",
        Device::Type::CPU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0] * inputs[0];
        },
        [](std::span<const Tensor> saved, std::span<const Tensor> grads) -> std::vector<Tensor> {
            return {saved[0] * grads[0] * 2.0f};
        }
    );

    Variable x(Tensor({2}, DType::Float32, Device::cpu()), true);
    x.tensor().fill_(2.0f);

    // Chain: custom_square -> builtin_sum
    // y = sum(x^2) = 2*(2^2) = 8
    auto y = dispatch_custom_op(square_op, {x});
    y = sum(y);

    EXPECT_NEAR(y.tensor().data<float>()[0], 8.0f, 1e-5);

    y.backward();

    // dy/dx = 2x = 4
    ASSERT_TRUE(x.has_grad());
    EXPECT_NEAR(x.grad()->to(Device::cpu()).data<float>()[0], 4.0f, 1e-5);
}

// ============================================================================
// Backend-parameterized custom-op autograd (plan 4.3)
//
// register_custom_op takes a Device::Type, so we register the same kernel for
// each available backend and verify the autograd path on each.
// ============================================================================

class CustomOpAutogradBackendTest : public tenzor::testing::BackendTest {};

TEST_P(CustomOpAutogradBackendTest, CustomOp_Square_Backward_CrossBackend) {
    // Audit: previously try/catch -> GTEST_SKIP("custom op unsupported on this
    // backend"), which buried a real custom-op registration/dispatch/backward
    // break as a clean skip. Let exceptions propagate so the gap surfaces.
    auto op_id = register_custom_op_with_backward(
        std::string("test::square_") + GetParam(),
        device.type,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return inputs[0] * inputs[0];
        },
        [](std::span<const Tensor> saved, std::span<const Tensor> grads)
            -> std::vector<Tensor> {
            return {saved[0] * grads[0] * 2.0f};
        });

    Variable x(Tensor({4}, DType::Float32, device), true);
    x.tensor().fill_(3.0f);
    auto y = dispatch_custom_op(op_id, {x});
    sum(y).backward();
    device.synchronize();
    ASSERT_TRUE(x.has_grad());
    auto grad_cpu = x.grad().value().to(Device::cpu());
    // d(x^2)/dx = 2x, x=3 => grad=6 per element
    EXPECT_NEAR(grad_cpu.data<float>()[0], 6.0f, 1e-4f);
}

INSTANTIATE_BACKEND_TESTS(CustomOpAutogradBackendTest);
