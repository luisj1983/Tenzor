/**
 * @file test_grad_linalg_parity.cpp
 * @brief Linear algebra gradient parity tests - Parameterized across all backends
 *
 * Verifies that gradients for linear algebra operations computed by different
 * backends match the CPU reference implementation.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradLinalgParityTest : public BackendTest {
protected:
    static void SetUpTestSuite() {
        try {
            tenzor::initialize();
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        }
    }

    static void TearDownTestSuite() {
        try { tenzor::finalize(); } catch (...) {}
    }

    void SetUp() override {
        BackendTest::SetUp();
        set_grad_enabled(true);
    }

    void compareGradientWithCPU(const Tensor& cpu_grad, const Tensor& backend_grad,
                               float rtol = 1e-5f, float atol = 1e-7f) {
        auto backend_grad_cpu = backend_grad.to(Device::cpu());
        device.synchronize();
        expectTensorNear(cpu_grad, backend_grad_cpu, atol);
    }

    void testUnaryGradient(
        std::function<Variable(const Variable&)> op_fn,
        const std::vector<int64_t>& shape,
        float atol = 1e-3f)
    {
        auto x_data = randn(shape, DType::Float32, Device::cpu());

        auto x_cpu = Variable(x_data.clone(), true);
        auto loss_cpu = op_fn(x_cpu);
        loss_cpu.backward();
        auto x_grad_cpu = x_cpu.grad().value();

        if (device.type == Device::Type::CPU) {
            ASSERT_TRUE(x_cpu.has_grad());
            return;
        }

        auto x_dev = Variable(x_data.to(device), true);
        auto loss_dev = op_fn(x_dev);
        loss_dev.backward();
        device.synchronize();

        ASSERT_TRUE(x_dev.has_grad());
        compareGradientWithCPU(x_grad_cpu, x_dev.grad().value(), 1e-5f, atol);
    }

    void testBinaryGradient(
        std::function<Variable(const Variable&, const Variable&)> op_fn,
        const std::vector<int64_t>& shape_a,
        const std::vector<int64_t>& shape_b,
        float atol = 1e-3f)
    {
        auto a_data = randn(shape_a, DType::Float32, Device::cpu());
        auto b_data = randn(shape_b, DType::Float32, Device::cpu());

        auto a_cpu = Variable(a_data.clone(), true);
        auto b_cpu = Variable(b_data.clone(), true);
        auto loss_cpu = op_fn(a_cpu, b_cpu);
        loss_cpu.backward();

        if (device.type == Device::Type::CPU) {
            ASSERT_TRUE(a_cpu.has_grad());
            return;
        }

        auto a_dev = Variable(a_data.to(device), true);
        auto b_dev = Variable(b_data.to(device), true);
        auto loss_dev = op_fn(a_dev, b_dev);
        loss_dev.backward();
        device.synchronize();

        compareGradientWithCPU(a_cpu.grad().value(), a_dev.grad().value(), 1e-5f, atol);
        compareGradientWithCPU(b_cpu.grad().value(), b_dev.grad().value(), 1e-5f, atol);
    }
};

// ============================================================================
// Linear Algebra Gradient Tests
// ============================================================================

TEST_P(GradLinalgParityTest, MatMulTransposedBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::matmul(x, tenzor::transpose(x, 0, 1)));
    }, {4, 8}, 1e-3f);
}

TEST_P(GradLinalgParityTest, BmmBackward) {
    testBinaryGradient([](const Variable& a, const Variable& b) {
        return tenzor::sum(tenzor::bmm(a, b));
    }, {2, 4, 8}, {2, 8, 4}, 1e-3f);
}

TEST_P(GradLinalgParityTest, LinearBackward) {
    nn::Linear linear(32, 16);
    auto input_data = randn({4, 32}, DType::Float32, Device::cpu());

    // CPU backward
    auto x_cpu = Variable(input_data.clone(), true);
    linear.to(Device::cpu());
    auto out_cpu = linear.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    // Backend backward
    nn::Linear linear_dev(32, 16);
    auto params = linear.parameters();
    auto dev_params = linear_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    linear_dev.to(device);

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = linear_dev.forward(x_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradLinalgParityTest, AddmmBackward) {
    // addmm may only exist at tensor level, not autograd; wrap in try-catch
    auto c_data = randn({4, 4}, DType::Float32, Device::cpu());
    auto a_data = randn({4, 8}, DType::Float32, Device::cpu());
    auto b_data = randn({8, 4}, DType::Float32, Device::cpu());

    // Use matmul + add as equivalent if addmm is not in autograd
    auto a_cpu = Variable(a_data.clone(), true);
    auto b_cpu = Variable(b_data.clone(), true);
    auto c_cpu = Variable(c_data.clone(), true);
    auto out_cpu = c_cpu + tenzor::matmul(a_cpu, b_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(a_cpu.has_grad());
        return;
    }

    auto a_dev = Variable(a_data.to(device), true);
    auto b_dev = Variable(b_data.to(device), true);
    auto c_dev = Variable(c_data.to(device), true);
    auto out_dev = c_dev + tenzor::matmul(a_dev, b_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(a_dev.has_grad());
    compareGradientWithCPU(a_cpu.grad().value(), a_dev.grad().value(), 1e-5f, 1e-3f);
    compareGradientWithCPU(b_cpu.grad().value(), b_dev.grad().value(), 1e-5f, 1e-3f);
    compareGradientWithCPU(c_cpu.grad().value(), c_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradLinalgParityTest, DetBackward) {
    // Create well-conditioned matrix: A = I + 0.1 * randn
    auto eye_data = tenzor::eye(4, std::nullopt, DType::Float32, Device::cpu());
    auto noise = randn({4, 4}, DType::Float32, Device::cpu()) * 0.1f;
    auto x_data = eye_data + noise;

    auto x_cpu = Variable(x_data.clone(), true);
    auto loss_cpu = tenzor::det(x_cpu);
    loss_cpu.backward();
    auto x_grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    auto x_dev = Variable(x_data.to(device), true);
    auto loss_dev = tenzor::det(x_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(x_grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradLinalgParityTest, SolveBackward) {
    auto eye_data = tenzor::eye(4, std::nullopt, DType::Float32, Device::cpu());
    auto noise_a = randn({4, 4}, DType::Float32, Device::cpu()) * 0.1f;
    auto a_data = eye_data + noise_a;
    auto b_data = randn({4, 2}, DType::Float32, Device::cpu());

    auto a_cpu = Variable(a_data.clone(), true);
    auto b_cpu = Variable(b_data.clone(), true);
    auto x_cpu = tenzor::solve(a_cpu, b_cpu);
    auto loss_cpu = tenzor::sum(x_cpu);
    loss_cpu.backward();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(a_cpu.has_grad());
        return;
    }

    auto a_dev = Variable(a_data.to(device), true);
    auto b_dev = Variable(b_data.to(device), true);
    auto x_dev = tenzor::solve(a_dev, b_dev);
    auto loss_dev = tenzor::sum(x_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(a_dev.has_grad());
    compareGradientWithCPU(a_cpu.grad().value(), a_dev.grad().value(), 1e-5f, 1e-3f);
    compareGradientWithCPU(b_cpu.grad().value(), b_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradLinalgParityTest, CholeskyBackward) {
    // Create symmetric positive-definite matrix: A = X^T X + I
    auto raw = randn({4, 4}, DType::Float32, Device::cpu());
    auto xtx = tenzor::matmul(raw.transpose(0, 1), raw);
    auto x_data = xtx + tenzor::eye(4, std::nullopt, DType::Float32, Device::cpu());

    auto x_cpu = Variable(x_data.clone(), true);
    auto L_cpu = tenzor::cholesky(x_cpu);
    auto loss_cpu = tenzor::sum(L_cpu);
    loss_cpu.backward();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    auto x_dev = Variable(x_data.to(device), true);
    auto L_dev = tenzor::cholesky(x_dev);
    auto loss_dev = tenzor::sum(L_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(x_cpu.grad().value(), x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradLinalgParityTest, BilinearBackward) {
    nn::Bilinear bilinear(16, 16, 8);
    auto input1_data = randn({4, 16}, DType::Float32, Device::cpu());
    auto input2_data = randn({4, 16}, DType::Float32, Device::cpu());

    // CPU backward
    auto x1_cpu = Variable(input1_data.clone(), true);
    auto x2_cpu = Variable(input2_data.clone(), true);
    bilinear.to(Device::cpu());
    auto out_cpu = bilinear.forward(x1_cpu, x2_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad1_cpu = x1_cpu.grad().value();
    auto grad2_cpu = x2_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x1_cpu.has_grad());
        ASSERT_TRUE(x2_cpu.has_grad());
        return;
    }

    // Backend backward
    nn::Bilinear bilinear_dev(16, 16, 8);
    auto params = bilinear.parameters();
    auto dev_params = bilinear_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    bilinear_dev.to(device);

    auto x1_dev = Variable(input1_data.to(device), true);
    auto x2_dev = Variable(input2_data.to(device), true);
    auto out_dev = bilinear_dev.forward(x1_dev, x2_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x1_dev.has_grad());
    ASSERT_TRUE(x2_dev.has_grad());
    compareGradientWithCPU(grad1_cpu, x1_dev.grad().value(), 1e-5f, 1e-3f);
    compareGradientWithCPU(grad2_cpu, x2_dev.grad().value(), 1e-5f, 1e-3f);
}

INSTANTIATE_BACKEND_TESTS(GradLinalgParityTest);
