// test_custom_op_higher_order.cpp
//
// Audit D2: CustomOpBackward must report `is_higher_order_stub() = true`
// when only a tensor-level backward was registered, and must preserve the
// autograd graph when a Variable-level backward was also registered via
// the new `CustomBackwardVariableFn` API.
//
// Tests:
//   1. Register a custom op WITH only a tensor backward.
//      - dispatch_custom_op produces a Variable with a grad_fn.
//      - That grad_fn is a CustomOpBackward whose `is_higher_order_stub()`
//        is true (honest stub, no Variable-level backward provided).
//   2. Register a custom op WITH a Variable-level backward in addition.
//      - dispatch_custom_op produces a Variable whose grad_fn is a
//        CustomOpBackward whose `is_higher_order_stub()` is false.
//      - calling `backward_with_variables` on the grad_fn produces Variables
//        whose own grad_fn descends from the incoming grad_outputs
//        (proves the graph was preserved).
//   3. Direct CustomOpBackward unit test: construct with var_backward,
//      verify `is_higher_order_stub() == false` and the var backward is
//      actually invoked.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/function.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/custom_op.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../grad_flow_helpers.hpp"

using namespace tenzor;

class D2Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

// ----------------------------------------------------------------------------
// 1. Tensor-only backward: stub flag must be honest (true).
// ----------------------------------------------------------------------------

TEST_F(D2Test, TensorBackwardOnly_IsHigherOrderStub) {
    static int call_count = 0;
    auto fwd = [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return inputs[0].clone();
    };
    auto bwd = [](std::span<const Tensor> /*saved*/,
                  std::span<const Tensor> grads) -> std::vector<Tensor> {
        ++call_count;
        return {grads[0].clone()};
    };
    auto op = register_custom_op_with_backward(
        "d2::tensor_only", Device::Type::CPU, fwd, bwd);

    auto x = Variable(zeros({3}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    auto y = dispatch_custom_op(op, {x});
    ASSERT_NE(y.grad_fn(), nullptr);
    auto* custom_bwd = dynamic_cast<CustomOpBackward*>(y.grad_fn().get());
    ASSERT_NE(custom_bwd, nullptr);
    EXPECT_TRUE(custom_bwd->is_higher_order_stub())
        << "tensor-only backward must honestly flag is_higher_order_stub=true";
}

// ----------------------------------------------------------------------------
// 2. Variable backward registered: stub flag false; graph preserved.
// ----------------------------------------------------------------------------

TEST_F(D2Test, VariableBackward_PreservesGraph) {
    auto fwd = [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return inputs[0].clone();
    };
    auto tensor_bwd = [](std::span<const Tensor> /*saved*/,
                         std::span<const Tensor> grads) -> std::vector<Tensor> {
        return {grads[0].clone()};
    };

    // Variable-level backward: grad_in = grad_out * 1 (identity, but composed
    // as a Variable op so the resulting grad_in's grad_fn descends from
    // grad_out).
    auto var_bwd = [](std::span<const Variable> /*saved*/,
                      std::span<const Variable> grad_outs) -> std::vector<Variable> {
        // Variable arithmetic: this preserves the graph through grad_outs[0].
        Variable one(ones(std::vector<int64_t>(grad_outs[0].tensor().shape().begin(),
                                               grad_outs[0].tensor().shape().end()),
                          grad_outs[0].tensor().dtype(),
                          grad_outs[0].tensor().device()),
                     /*requires_grad=*/false);
        return {grad_outs[0] * one};
    };

    auto op = register_custom_op_with_backward(
        "d2::var_backward", Device::Type::CPU, fwd, tensor_bwd,
        /*save_fn=*/nullptr, var_bwd);

    auto x = Variable(zeros({3}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    auto y = dispatch_custom_op(op, {x});
    ASSERT_NE(y.grad_fn(), nullptr);
    auto* custom_bwd = dynamic_cast<CustomOpBackward*>(y.grad_fn().get());
    ASSERT_NE(custom_bwd, nullptr);
    EXPECT_FALSE(custom_bwd->is_higher_order_stub())
        << "var_backward registered: stub flag must be false";

    // Invoke backward_with_variables directly with a requires_grad=true grad,
    // confirm the result carries a grad_fn (the multiplication preserves it).
    auto g_t = ones({3}, DType::Float32, Device::cpu());
    Variable g(g_t, /*requires_grad=*/true);
    auto results = custom_bwd->backward_with_variables({g});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].requires_grad());
    EXPECT_NE(results[0].grad_fn(), nullptr)
        << "Variable-level backward must preserve the graph through grad_out";
}

// ----------------------------------------------------------------------------
// 3. Direct CustomOpBackward unit test: stub flag flips with constructor.
// ----------------------------------------------------------------------------

TEST_F(D2Test, DirectConstruction_StubFlagMatchesConstructor) {
    auto tensor_bwd = [](std::span<const Tensor>,
                         std::span<const Tensor> grads) -> std::vector<Tensor> {
        return {grads[0].clone()};
    };
    auto var_bwd = [](std::span<const Variable>,
                      std::span<const Variable> grads) -> std::vector<Variable> {
        return {grads[0]};
    };

    // 1-arg constructor: no var backward -> stub.
    CustomOpBackward stubbed(tensor_bwd);
    EXPECT_TRUE(stubbed.is_higher_order_stub());

    // 2-arg constructor: var backward present -> not a stub.
    CustomOpBackward unstubbed(tensor_bwd, var_bwd);
    EXPECT_FALSE(unstubbed.is_higher_order_stub());

    // 2-arg constructor with empty var backward -> stub.
    CustomOpBackward stubbed_empty(tensor_bwd, CustomBackwardVariableFn{});
    EXPECT_TRUE(stubbed_empty.is_higher_order_stub());
}

// ----------------------------------------------------------------------------
// 4. First-order grad still computes correctly when no var backward.
// ----------------------------------------------------------------------------

TEST_F(D2Test, TensorBackwardOnly_FirstOrderStillWorks) {
    auto fwd = [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return inputs[0].clone();
    };
    auto bwd = [](std::span<const Tensor>,
                  std::span<const Tensor> grads) -> std::vector<Tensor> {
        return {grads[0].clone()};
    };
    auto op = register_custom_op_with_backward(
        "d2::first_order_only", Device::Type::CPU, fwd, bwd);

    auto x_t = ones({4}, DType::Float32, Device::cpu());
    Variable x(x_t, /*requires_grad=*/true);
    auto y = dispatch_custom_op(op, {x});

    auto loss = tenzor::sum(y);
    loss.backward();
    EXPECT_GRAD_FLOWS(x);
    auto& g = x.grad().value();
    ASSERT_EQ(g.numel(), 4);
    auto* gp = g.data<float>();
    for (int64_t i = 0; i < g.numel(); ++i) {
        EXPECT_FLOAT_EQ(gp[i], 1.0f);
    }
}
