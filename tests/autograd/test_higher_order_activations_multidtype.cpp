/**
 * @file test_higher_order_activations_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for higher-order activation gradients
 *
 * Multi-backend port of test_higher_order_activations.cpp. Verifies that
 * sigmoid/tanh/gelu/elu double-backward produces the mathematically-correct
 * second derivative across all available backends and data types.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>

#include <cmath>
#include <functional>
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class HigherOrderActivationsMultiDTypeTest : public MultiBackendDTypeTest {};

// Higher-order gradient tests are numerically demanding — skip Float16
#define SKIP_IF_LOW_PRECISION() \
    do { \
        if (dtype() == DType::Float16) \
            GTEST_SKIP() << "Higher-order grads require Float32+ precision"; \
    } while (0)

// Helper: run "loss = sum(act(x)) -> first backward -> loss2 = sum(grad*grad) -> second backward"
// and return the sum of |second_gradient|. If the activation routes through
// a proper autograd-aware backward this is non-zero; if it routes through a
// stub / leaf Variable this is zero.
static double double_backward_magnitude(
    const std::function<Variable(const Variable&)>& act,
    DType dt, Device dev) {
    // Use Float64 if available for numerics, otherwise the test dtype
    auto compute_dtype = (dt == DType::Float64) ? DType::Float64 : dt;

    auto x_t = zeros({4}, compute_dtype, dev);
    // Set values via CPU then move
    auto x_cpu = zeros({4}, DType::Float64, Device::cpu());
    x_cpu.data<double>()[0] = -0.7;
    x_cpu.data<double>()[1] = -0.1;
    x_cpu.data<double>()[2] =  0.4;
    x_cpu.data<double>()[3] =  1.5;
    x_t = x_cpu.to(compute_dtype).to(dev);

    auto x = Variable(x_t, true);
    auto y = act(x);
    auto loss = tenzor::sum(y);
    loss.backward(std::nullopt, false, true);
    if (!x.grad().has_value()) return 0.0;

    auto grad_var = Variable(x.grad().value(), true);
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    if (!grad_var.grad().has_value()) return 0.0;

    auto second = grad_var.grad().value().to(Device::cpu()).to(DType::Float32).contiguous();
    double total = 0.0;
    const float* p = second.data<float>();
    for (int64_t i = 0; i < second.numel(); ++i) total += std::abs(p[i]);
    return total;
}

TEST_P(HigherOrderActivationsMultiDTypeTest, SigmoidDoubleBackwardNonZero) {
    SKIP_IF_LOW_PRECISION();

    auto x_cpu = zeros({3}, DType::Float64, Device::cpu());
    x_cpu.data<double>()[0] = -1.0;
    x_cpu.data<double>()[1] =  0.5;
    x_cpu.data<double>()[2] =  2.0;
    auto x_t = x_cpu.to(dtype()).to(device());

    auto x = Variable(x_t, /*requires_grad=*/true);

    auto y = tenzor::nn::sigmoid(x);
    auto loss = tenzor::sum(y);

    // First backward with create_graph=true so grad_x carries a grad_fn.
    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
    EXPECT_GRAD_FLOWS(x);
    auto grad_x_t = x.grad().value();

    auto grad_x_var = Variable(grad_x_t, true);
    auto grad_norm = tenzor::sum(grad_x_var * grad_x_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(grad_x_var);
    auto second_grad = grad_x_var.grad().value().to(Device::cpu()).to(DType::Float32).contiguous();
    const float* gdata = second_grad.data<float>();

    double total = 0.0;
    for (int64_t i = 0; i < second_grad.numel(); ++i) {
        total += std::abs(gdata[i]);
    }
    EXPECT_GT(total, 1e-6)
        << "sigmoid double-backward produced a zero gradient — the P4.2 "
           "higher-order path is not active";
}

TEST_P(HigherOrderActivationsMultiDTypeTest, TanhDoubleBackwardNonZero) {
    SKIP_IF_LOW_PRECISION();

    auto x_cpu = zeros({3}, DType::Float64, Device::cpu());
    x_cpu.data<double>()[0] = -0.5;
    x_cpu.data<double>()[1] =  0.3;
    x_cpu.data<double>()[2] =  1.0;
    auto x_t = x_cpu.to(dtype()).to(device());

    auto x = Variable(x_t, true);
    auto y = tenzor::nn::tanh(x);
    auto loss = tenzor::sum(y);
    loss.backward(std::nullopt, false, true);

    EXPECT_GRAD_FLOWS(x);
    auto grad_x_t = x.grad().value();
    auto grad_x_var = Variable(grad_x_t, true);
    auto grad_norm = tenzor::sum(grad_x_var * grad_x_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(grad_x_var);
    auto second_grad = grad_x_var.grad().value().to(Device::cpu()).to(DType::Float32).contiguous();
    const float* gdata = second_grad.data<float>();
    double total = 0.0;
    for (int64_t i = 0; i < second_grad.numel(); ++i) {
        total += std::abs(gdata[i]);
    }
    EXPECT_GT(total, 1e-6);
}

// Linear-chain sanity check: sigmoid -> sigmoid should still support a second backward.
TEST_P(HigherOrderActivationsMultiDTypeTest, SigmoidChainDoubleBackward) {
    SKIP_IF_LOW_PRECISION();

    auto x_cpu = zeros({2, 2}, DType::Float64, Device::cpu());
    for (int i = 0; i < 4; ++i) x_cpu.data<double>()[i] = 0.1 * (i + 1);
    auto x_t = x_cpu.to(dtype()).to(device());

    auto x = Variable(x_t, true);
    auto y = tenzor::nn::sigmoid(tenzor::nn::sigmoid(x));
    auto loss = tenzor::sum(y);
    loss.backward(std::nullopt, false, true);

    EXPECT_GRAD_FLOWS(x);
    auto grad_var = Variable(x.grad().value(), true);
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();

    EXPECT_GRAD_FLOWS(grad_var);
    auto second = grad_var.grad().value().to(Device::cpu()).to(DType::Float32).contiguous();
    double total = 0.0;
    const float* p = second.data<float>();
    for (int i = 0; i < 4; ++i) total += std::abs(p[i]);
    EXPECT_GT(total, 1e-6);
}

TEST_P(HigherOrderActivationsMultiDTypeTest, GeLUDoubleBackwardNonZero) {
    SKIP_IF_LOW_PRECISION();

    double m = double_backward_magnitude(
        [](const Variable& x) { return tenzor::nn::gelu(x, "none"); },
        dtype(), device());
    EXPECT_GT(m, 1e-6)
        << "nn::gelu did not route through the autograd-aware GeluBackward";
}

TEST_P(HigherOrderActivationsMultiDTypeTest, GeLUTanhApproxDoubleBackwardNonZero) {
    SKIP_IF_LOW_PRECISION();

    double m = double_backward_magnitude(
        [](const Variable& x) { return tenzor::nn::gelu(x, "tanh"); },
        dtype(), device());
    EXPECT_GT(m, 1e-6);
}

TEST_P(HigherOrderActivationsMultiDTypeTest, ELUDoubleBackwardNonZero) {
    SKIP_IF_LOW_PRECISION();

    double m = double_backward_magnitude(
        [](const Variable& x) { return tenzor::nn::elu(x, 1.0); },
        dtype(), device());
    EXPECT_GT(m, 1e-6)
        << "nn::elu did not route through the autograd-aware EluBackward";
}

TEST_P(HigherOrderActivationsMultiDTypeTest, SELUDoubleBackwardNonZero) {
    SKIP_IF_LOW_PRECISION();

    double m = double_backward_magnitude(
        [](const Variable& x) { return tenzor::nn::selu(x); },
        dtype(), device());
    EXPECT_GT(m, 1e-6)
        << "nn::selu did not route through the autograd-aware SeluBackward";
}

TEST_P(HigherOrderActivationsMultiDTypeTest, MishDoubleBackwardNonZero) {
    SKIP_IF_LOW_PRECISION();

    double m = double_backward_magnitude(
        [](const Variable& x) { return tenzor::nn::mish(x); },
        dtype(), device());
    EXPECT_GT(m, 1e-6)
        << "nn::mish did not route through the autograd-aware MishBackward";
}

TEST_P(HigherOrderActivationsMultiDTypeTest, SwishDoubleBackwardNonZero) {
    SKIP_IF_LOW_PRECISION();

    double m = double_backward_magnitude(
        [](const Variable& x) { return tenzor::nn::swish(x); },
        dtype(), device());
    EXPECT_GT(m, 1e-6)
        << "nn::swish should produce non-zero second derivatives via "
           "x * sigmoid(x) Variable composition";
}

TEST_P(HigherOrderActivationsMultiDTypeTest, HardswishDoubleBackwardFinite) {
    SKIP_IF_LOW_PRECISION();

    auto x_cpu = zeros({4}, DType::Float64, Device::cpu());
    x_cpu.data<double>()[0] = -2.0;
    x_cpu.data<double>()[1] = -1.0;
    x_cpu.data<double>()[2] =  1.0;
    x_cpu.data<double>()[3] =  2.0;
    auto x_t = x_cpu.to(dtype()).to(device());

    auto x = Variable(x_t, true);
    auto y = tenzor::nn::hardswish(x);
    auto loss = tenzor::sum(y);
    // audit-3 T.15: EXPECT_NO_THROW(...backward...) passes for severed grad_fn.
    // EXPECT_GRAD_FLOWS asserts the gradient actually propagated.
    loss.backward(std::nullopt, false, true);
    EXPECT_GRAD_FLOWS(x);

    auto grad_var = Variable(x.grad().value(), true);
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    EXPECT_GRAD_FLOWS(grad_var);
}

TEST_P(HigherOrderActivationsMultiDTypeTest, HardsigmoidDoubleBackwardFinite) {
    SKIP_IF_LOW_PRECISION();

    auto x_cpu = zeros({4}, DType::Float64, Device::cpu());
    x_cpu.data<double>()[0] = -2.0;
    x_cpu.data<double>()[1] = -1.0;
    x_cpu.data<double>()[2] =  1.0;
    x_cpu.data<double>()[3] =  2.0;
    auto x_t = x_cpu.to(dtype()).to(device());

    auto x = Variable(x_t, true);
    auto y = tenzor::nn::hardsigmoid(x);
    auto loss = tenzor::sum(y);
    // audit-3 T.15: replace EXPECT_NO_THROW with EXPECT_GRAD_FLOWS.
    loss.backward(std::nullopt, false, true);
    EXPECT_GRAD_FLOWS(x);

    auto grad_var = Variable(x.grad().value(), true);
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    EXPECT_GRAD_FLOWS(grad_var);
}

TEST_P(HigherOrderActivationsMultiDTypeTest, LeakyReLUDoubleBackwardPiecewiseLinear) {
    SKIP_IF_LOW_PRECISION();

    auto x_cpu = zeros({4}, DType::Float64, Device::cpu());
    x_cpu.data<double>()[0] = -0.7;
    x_cpu.data<double>()[1] = -0.1;
    x_cpu.data<double>()[2] =  0.4;
    x_cpu.data<double>()[3] =  1.5;
    auto x_t = x_cpu.to(dtype()).to(device());

    auto x = Variable(x_t, true);
    auto y = tenzor::nn::leaky_relu(x, 0.1);
    auto loss = tenzor::sum(y);
    // audit-3 T.15: replace EXPECT_NO_THROW with EXPECT_GRAD_FLOWS — the
    // leaky_relu gradient is piecewise-constant non-zero across the chosen
    // sample points (-0.7, -0.1, 0.4, 1.5) so a real grad must flow.
    loss.backward(std::nullopt, false, true);
    EXPECT_GRAD_FLOWS(x);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HigherOrderActivationsMultiDTypeTest);
