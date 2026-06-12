/**
 * @file test_grad_shape_parity.cpp
 * @brief Shape/indexing operation gradient parity tests - Parameterized across all backends
 *
 * Verifies that gradients for shape manipulation and indexing operations
 * computed by different backends match the CPU reference implementation.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradShapeParityTest : public BackendTest {
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
        float atol = 1e-5f)
    {
        auto x_data = randn(shape, DType::Float32, Device::cpu());

        auto x_cpu = Variable(x_data.clone(), true);
        auto loss_cpu = op_fn(x_cpu);
        loss_cpu.backward();
        auto x_grad_cpu = x_cpu.grad().value();

        if (device.type == Device::Type::CPU) {
            EXPECT_GRAD_FLOWS(x_cpu);
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
        const std::vector<int64_t>& shape,
        float atol = 1e-5f)
    {
        auto a_data = randn(shape, DType::Float32, Device::cpu());
        auto b_data = randn(shape, DType::Float32, Device::cpu());

        auto a_cpu = Variable(a_data.clone(), true);
        auto b_cpu = Variable(b_data.clone(), true);
        auto loss_cpu = op_fn(a_cpu, b_cpu);
        loss_cpu.backward();

        if (device.type == Device::Type::CPU) {
            EXPECT_GRAD_FLOWS(a_cpu);
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
// Shape/Indexing Gradient Tests
// ============================================================================

TEST_P(GradShapeParityTest, ReshapeBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::reshape(x, {2, 8}));
    }, {4, 4}, 1e-5f);
}

TEST_P(GradShapeParityTest, TransposeBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::transpose(x, 0, 1));
    }, {4, 8}, 1e-5f);
}

TEST_P(GradShapeParityTest, PermuteBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::permute(x, {2, 0, 1}));
    }, {2, 4, 8}, 1e-5f);
}

TEST_P(GradShapeParityTest, GatherBackward) {
    // Create deterministic index tensor for gather along dim 1
    auto run_test = [&](Device test_device) {
        auto x_data = randn({4, 8}, DType::Float32, Device::cpu());
        // Indices: select 4 elements from dim 1 for each row
        auto idx_data = tenzor::randint(0, 8, {4, 4}, DType::Int64, Device::cpu());

        auto x_var = Variable(x_data.to(test_device), true);
        auto idx = idx_data.to(test_device);
        auto gathered = tenzor::gather(x_var, 1, idx);
        auto loss = tenzor::sum(gathered);
        loss.backward();
        test_device.synchronize();
        return std::make_pair(x_var.grad().value(), idx_data);
    };

    auto [grad_cpu, idx_data] = run_test(Device::cpu());

    if (device.type == Device::Type::CPU) {
        ASSERT_EQ(grad_cpu.dim(), 2);
        ASSERT_EQ(grad_cpu.size(0), 4);
        ASSERT_EQ(grad_cpu.size(1), 8);
        Tensor grad_abs_sum = tenzor::sum(tenzor::abs(grad_cpu));
        const float* grad_ptr = grad_abs_sum.data<float>();
        ASSERT_NE(grad_ptr, nullptr);
        float grad_sum = grad_ptr[0];
        ASSERT_GT(grad_sum, 0.0f);
        return;
    }

    // Re-run on device with same index
    auto x_data = randn({4, 8}, DType::Float32, Device::cpu());
    auto x_cpu = Variable(x_data.clone(), true);
    auto gathered_cpu = tenzor::gather(x_cpu, 1, idx_data);
    auto loss_cpu = tenzor::sum(gathered_cpu);
    loss_cpu.backward();

    auto x_dev = Variable(x_data.to(device), true);
    auto gathered_dev = tenzor::gather(x_dev, 1, idx_data.to(device));
    auto loss_dev = tenzor::sum(gathered_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(x_cpu.grad().value(), x_dev.grad().value(), 1e-5f, 1e-5f);
}

TEST_P(GradShapeParityTest, ScatterAddBackward) {
    auto x_data = randn({4, 4}, DType::Float32, Device::cpu());
    auto idx_data = tenzor::randint(0, 8, {4, 4}, DType::Int64, Device::cpu());

    auto run = [&](Device d) {
        auto zeros_var = Variable(tenzor::zeros({4, 8}, DType::Float32, d), true);
        auto x_var = Variable(x_data.to(d), true);
        auto out = tenzor::scatter_add(zeros_var, 1, idx_data.to(d), x_var);
        auto loss = tenzor::sum(out);
        loss.backward();
        d.synchronize();
        return x_var.grad().value();
    };

    auto grad_cpu = run(Device::cpu());

    if (device.type == Device::Type::CPU) {
        ASSERT_EQ(grad_cpu.dim(), 2);
        ASSERT_EQ(grad_cpu.size(0), 4);
        ASSERT_EQ(grad_cpu.size(1), 4);
        Tensor grad_abs_sum = tenzor::sum(tenzor::abs(grad_cpu));
        const float* grad_ptr = grad_abs_sum.data<float>();
        ASSERT_NE(grad_ptr, nullptr);
        float grad_sum = grad_ptr[0];
        ASSERT_GT(grad_sum, 0.0f);
        return;
    }

    auto grad_dev = run(device);
    compareGradientWithCPU(grad_cpu, grad_dev, 1e-5f, 1e-5f);
}

TEST_P(GradShapeParityTest, WhereBackward) {
    auto x_data = randn({4, 4}, DType::Float32, Device::cpu());
    auto y_data = randn({4, 4}, DType::Float32, Device::cpu());
    auto cond_data = tenzor::gt(randn({4, 4}, DType::Float32, Device::cpu()), tenzor::zeros({4, 4}, DType::Float32, Device::cpu()));

    auto x_cpu = Variable(x_data.clone(), true);
    auto y_cpu = Variable(y_data.clone(), true);
    auto cond_cpu = Variable(cond_data.clone(), false);
    auto out_cpu = tenzor::where(cond_cpu, x_cpu, y_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(x_cpu);
        EXPECT_GRAD_FLOWS(y_cpu);
        return;
    }

    auto x_dev = Variable(x_data.to(device), true);
    auto y_dev = Variable(y_data.to(device), true);
    auto cond_dev = Variable(cond_data.to(device), false);
    auto out_dev = tenzor::where(cond_dev, x_dev, y_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    ASSERT_TRUE(y_dev.has_grad());
    compareGradientWithCPU(x_cpu.grad().value(), x_dev.grad().value(), 1e-5f, 1e-5f);
    compareGradientWithCPU(y_cpu.grad().value(), y_dev.grad().value(), 1e-5f, 1e-5f);
}

TEST_P(GradShapeParityTest, CatBackward) {
    auto x_data = randn({4, 4}, DType::Float32, Device::cpu());
    auto y_data = randn({4, 4}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(x_data.clone(), true);
    auto y_cpu = Variable(y_data.clone(), true);
    auto out_cpu = tenzor::cat({x_cpu, y_cpu}, 0);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(x_cpu);
        EXPECT_GRAD_FLOWS(y_cpu);
        return;
    }

    auto x_dev = Variable(x_data.to(device), true);
    auto y_dev = Variable(y_data.to(device), true);
    auto out_dev = tenzor::cat({x_dev, y_dev}, 0);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    ASSERT_TRUE(y_dev.has_grad());
    compareGradientWithCPU(x_cpu.grad().value(), x_dev.grad().value(), 1e-5f, 1e-5f);
    compareGradientWithCPU(y_cpu.grad().value(), y_dev.grad().value(), 1e-5f, 1e-5f);
}

TEST_P(GradShapeParityTest, SplitBackward) {
    // Use slice (narrow) as autograd split equivalent
    testUnaryGradient([](const Variable& x) {
        auto part0 = tenzor::slice(x, 0, 0, 2);  // rows 0-1
        auto part1 = tenzor::slice(x, 0, 2, 4);  // rows 2-3
        return tenzor::sum(part0) + tenzor::sum(part1);
    }, {4, 4}, 1e-5f);
}

INSTANTIATE_BACKEND_TESTS(GradShapeParityTest);
