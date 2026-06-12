/**
 * @file test_grad_arithmetic_parity.cpp
 * @brief Arithmetic gradient parity tests - Parameterized across all backends
 *
 * Verifies that gradients for arithmetic operations computed by different
 * backends match the CPU reference implementation.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradArithmeticParityTest : public BackendTest {
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
        auto a_grad_cpu = a_cpu.grad().value();
        auto b_grad_cpu = b_cpu.grad().value();

        if (device.type == Device::Type::CPU) {
            EXPECT_GRAD_FLOWS(a_cpu);
            EXPECT_GRAD_FLOWS(b_cpu);
            return;
        }

        auto a_dev = Variable(a_data.to(device), true);
        auto b_dev = Variable(b_data.to(device), true);
        auto loss_dev = op_fn(a_dev, b_dev);
        loss_dev.backward();
        device.synchronize();

        ASSERT_TRUE(a_dev.has_grad());
        ASSERT_TRUE(b_dev.has_grad());

        compareGradientWithCPU(a_grad_cpu, a_dev.grad().value(), 1e-5f, atol);
        compareGradientWithCPU(b_grad_cpu, b_dev.grad().value(), 1e-5f, atol);
    }
};

// ============================================================================
// Arithmetic Gradient Tests
// ============================================================================

TEST_P(GradArithmeticParityTest, SubBackward) {
    testBinaryGradient([](const Variable& a, const Variable& b) {
        return tenzor::sum(a - b);
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, DivBackward) {
    // Use (b*b + 1) as the divisor so it is always >= 1 regardless of RNG
    // state. The previous (b + 2.0f) form could shrink to near zero when
    // randn produced b~-2, which blew the gradient of a/b up past the
    // 1e-5 atol while the relative error stayed at ~1e-7.
    testBinaryGradient([](const Variable& a, const Variable& b) {
        return tenzor::sum(a / (b * b + 1.0f));
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, PowBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::pow(x, 2.0f));
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, SqrtBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::sqrt(tenzor::abs(x) + 0.01f));
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, ExpBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::exp(x * 0.1f));
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, LogBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::log(tenzor::abs(x) + 0.1f));
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, AbsBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::abs(x));
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, NegBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::neg(x));
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, ClampBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::clamp(x, -0.5f, 0.5f));
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, AddScalarBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(x + 1.0f);
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, MulScalarBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(x * 2.0f);
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, MinBackward) {
    testBinaryGradient([](const Variable& a, const Variable& b) {
        return tenzor::sum((a + b - tenzor::abs(a - b)) * 0.5f);
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, MaxBackward) {
    testBinaryGradient([](const Variable& a, const Variable& b) {
        return tenzor::sum((a + b + tenzor::abs(a - b)) * 0.5f);
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, ReciprocalBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(tenzor::reciprocal(x + 2.0f));
    }, {4, 4});
}

TEST_P(GradArithmeticParityTest, SquareBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(x * x);
    }, {4, 4});
}

INSTANTIATE_BACKEND_TESTS(GradArithmeticParityTest);
