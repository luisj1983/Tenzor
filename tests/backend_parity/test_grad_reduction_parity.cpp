/**
 * @file test_grad_reduction_parity.cpp
 * @brief Reduction operation gradient parity tests - Parameterized across all backends
 *
 * Verifies that gradients for reduction operations computed by different
 * backends match the CPU reference implementation.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradReductionParityTest : public BackendTest {
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
        float atol = 1e-4f)
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
        const std::vector<int64_t>& shape,
        float atol = 1e-4f)
    {
        auto a_data = randn(shape, DType::Float32, Device::cpu());
        auto b_data = randn(shape, DType::Float32, Device::cpu());

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
// Reduction Gradient Tests
// ============================================================================

TEST_P(GradReductionParityTest, SumDimBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::sum(x, 1));
    }, {4, 8}, 1e-4f);
}

TEST_P(GradReductionParityTest, MeanDimBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::mean(x, 1));
    }, {4, 8}, 1e-4f);
}

TEST_P(GradReductionParityTest, VarBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::var(x, 1));
    }, {4, 8}, 1e-4f);
}

TEST_P(GradReductionParityTest, StdBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::std(x, 1));
    }, {4, 8}, 1e-4f);
}

TEST_P(GradReductionParityTest, NormBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::vector_norm(x, 2.0, {1}));
    }, {4, 8}, 1e-4f);
}

TEST_P(GradReductionParityTest, LogSumExpBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::logsumexp(x, 1));
    }, {4, 8}, 1e-4f);
}

TEST_P(GradReductionParityTest, MaxDimBackward) {
    testUnaryGradient([](const Variable& x) {
        auto vals = tenzor::max(x, 1);
        return tenzor::sum(vals);
    }, {4, 8}, 1e-4f);
}

TEST_P(GradReductionParityTest, MinDimBackward) {
    testUnaryGradient([](const Variable& x) {
        auto vals = tenzor::min(x, 1);
        return tenzor::sum(vals);
    }, {4, 8}, 1e-4f);
}

TEST_P(GradReductionParityTest, ProdBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::prod(tenzor::abs(x) + 0.5f, 1));
    }, {4, 4}, 1e-4f);
}

TEST_P(GradReductionParityTest, CumSumBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::cumsum(x, 1));
    }, {4, 8}, 1e-4f);
}

INSTANTIATE_BACKEND_TESTS(GradReductionParityTest);
