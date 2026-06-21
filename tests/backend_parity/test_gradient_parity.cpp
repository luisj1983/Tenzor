/**
 * @file test_gradient_parity.cpp
 * @brief Gradient computation parity tests - Parameterized across all backends
 *
 * Verifies that gradients computed by different backends match,
 * using numerical gradient checking as reference.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradientParityBackendTest : public BackendTest {
protected:
    static void SetUpTestSuite() {
        try {
            tenzor::initialize();
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        }
    }

    static void TearDownTestSuite() {
        try {
            tenzor::finalize();
        } catch (...) {}
    }

    void SetUp() override {
        BackendTest::SetUp();
        set_grad_enabled(true);
    }

    // Helper to compare gradients between CPU and the test device
    void compareGradientWithCPU(const Tensor& cpu_grad, const Tensor& backend_grad,
                               float rtol = 1e-5f, float atol = 1e-7f) {
        auto backend_grad_cpu = backend_grad.to(Device::cpu());
        device.synchronize();

        expectTensorNear(cpu_grad, backend_grad_cpu, rtol, atol);
    }

    // Run a gradient test on the current device and compare to CPU reference
    // op_fn: takes two Variables (a, b) on a given device and returns a scalar loss Variable
    void testBinaryGradient(
        std::function<Variable(const Variable&, const Variable&)> op_fn,
        const std::vector<int64_t>& shape,
        float atol = 1e-5f)
    {
        // Create CPU reference data
        auto a_data = randn(shape, DType::Float32, Device::cpu());
        auto b_data = randn(shape, DType::Float32, Device::cpu());

        // CPU reference gradients
        auto a_cpu = Variable(a_data.clone(), true);
        auto b_cpu = Variable(b_data.clone(), true);
        auto loss_cpu = op_fn(a_cpu, b_cpu);
        loss_cpu.backward();
        auto a_grad_cpu = a_cpu.grad().value();
        auto b_grad_cpu = b_cpu.grad().value();

        if (device.type == Device::Type::CPU) {
            // We already have the CPU result; just verify grad flowed
            EXPECT_GRAD_FLOWS(a_cpu);
            EXPECT_GRAD_FLOWS(b_cpu);
            return;
        }

        // Backend gradients
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

    // Run a unary gradient test
    void testUnaryGradient(
        std::function<Variable(const Variable&)> op_fn,
        const std::vector<int64_t>& shape,
        float atol = 1e-5f)
    {
        auto x_data = randn(shape, DType::Float32, Device::cpu());

        // CPU reference
        auto x_cpu = Variable(x_data.clone(), true);
        auto loss_cpu = op_fn(x_cpu);
        loss_cpu.backward();
        auto x_grad_cpu = x_cpu.grad().value();

        if (device.type == Device::Type::CPU) {
            ASSERT_TRUE(x_cpu.has_grad());
            return;
        }

        // Backend
        auto x_dev = Variable(x_data.to(device), true);
        auto loss_dev = op_fn(x_dev);
        loss_dev.backward();
        device.synchronize();

        ASSERT_TRUE(x_dev.has_grad());
        compareGradientWithCPU(x_grad_cpu, x_dev.grad().value(), 1e-5f, atol);
    }
};

// ============================================================================
// Basic Operation Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, AddBackward) {
    testBinaryGradient([](const Variable& a, const Variable& b) {
        auto c = a + b;
        return tenzor::sum(c);
    }, {4, 4});
}

TEST_P(GradientParityBackendTest, MulBackward) {
    testBinaryGradient([](const Variable& a, const Variable& b) {
        auto c = a * b;
        return tenzor::sum(c);
    }, {4, 4});
}

TEST_P(GradientParityBackendTest, MatMulBackward) {
    // MatMul: (4x8) @ (8x6) -> (4x6)
    auto a_data = randn({4, 8}, DType::Float32, Device::cpu());
    auto b_data = randn({8, 6}, DType::Float32, Device::cpu());

    // CPU reference
    auto a_cpu = Variable(a_data.clone(), true);
    auto b_cpu = Variable(b_data.clone(), true);
    auto c_cpu = tenzor::matmul(a_cpu, b_cpu);
    auto loss_cpu = tenzor::sum(c_cpu);
    loss_cpu.backward();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(a_cpu);
        EXPECT_GRAD_FLOWS(b_cpu);
        return;
    }

    // Backend
    auto a_dev = Variable(a_data.to(device), true);
    auto b_dev = Variable(b_data.to(device), true);
    auto c_dev = tenzor::matmul(a_dev, b_dev);
    auto loss_dev = tenzor::sum(c_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(a_dev.has_grad());
    ASSERT_TRUE(b_dev.has_grad());
    compareGradientWithCPU(a_cpu.grad().value(), a_dev.grad().value(), 1e-5f, 1e-4f);
    compareGradientWithCPU(b_cpu.grad().value(), b_dev.grad().value(), 1e-5f, 1e-4f);
}

// ============================================================================
// Activation Function Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, ReLUBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(nn::relu(x));
    }, {8, 8});
}

TEST_P(GradientParityBackendTest, SigmoidBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(nn::sigmoid(x));
    }, {8, 8});
}

TEST_P(GradientParityBackendTest, TanhBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(nn::tanh(x));
    }, {8, 8});
}

TEST_P(GradientParityBackendTest, GELUBackward) {
    testUnaryGradient([](const Variable& x) {
        return tenzor::sum(nn::gelu(x));
    }, {8, 8}, 1e-4f);
}

TEST_P(GradientParityBackendTest, SoftmaxBackward) {
    // sum(softmax(x, dim=1)) is identically the batch size (each row sums to
    // 1), so its gradient w.r.t. x is exactly zero — that exercises the
    // softmax-backward kernel with a zero upstream grad, which a backend that
    // returns zeros without running the kernel would pass. Use sum(sm*sm) so
    // the upstream gradient is the non-constant 2*sm, forcing the real
    // softmax-backward Jacobian (diag(sm) - sm sm^T) to be exercised.
    testUnaryGradient([](const Variable& x) {
        auto sm = tenzor::softmax(x, 1);
        return tenzor::sum(sm * sm);
    }, {4, 8}, 1e-4f);
}

// ============================================================================
// Complex Operation Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, Conv2dBackward) {
    // Previously this test called tenzor::linear() as a "proxy" with a
    // 4D input — that doesn't even make sense shape-wise and every
    // non-CPU backend returned zeros while CPU silently produced
    // garbage. Replace with a real conv2d backward: input (1,3,8,8),
    // weight (4,3,3,3), bias (4), stride=1, padding=1.
    auto input_data = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    auto weight_data = randn({4, 3, 3, 3}, DType::Float32, Device::cpu());
    auto bias_data = randn({4}, DType::Float32, Device::cpu());

    auto run = [&](const Device& dev) {
        auto input = Variable(input_data.to(dev), true);
        auto weight = Variable(weight_data.to(dev), true);
        auto bias = Variable(bias_data.to(dev), true);
        auto out = tenzor::nn::functional::conv2d(
            input, weight, bias, /*stride=*/{1, 1},
            /*padding=*/{1, 1}, /*dilation=*/{1, 1}, /*groups=*/1);
        auto loss = tenzor::sum(out);
        loss.backward();
        if (dev.type != Device::Type::CPU) dev.synchronize();
        return input;
    };

    auto input_cpu = run(Device::cpu());

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(input_cpu);
        return;
    }

    auto input_dev = run(device);
    ASSERT_TRUE(input_dev.has_grad());
    compareGradientWithCPU(input_cpu.grad().value(), input_dev.grad().value(),
                           1e-4f, 1e-3f);
}

TEST_P(GradientParityBackendTest, BatchNormBackward) {
    // Real BatchNorm2d backward parity. The previous version substituted
    // tenzor::mean(x) ("batch norm needs module state"), so a BatchNorm
    // backward bug on any backend was invisible and mean was already covered
    // by MeanReduction. Exercise the actual BN2d backward in train mode.
    nn::BatchNorm2d bn(8);
    auto input_data = randn({2, 8, 4, 4}, DType::Float32, Device::cpu());

    bn.train();
    bn.to(Device::cpu());
    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = bn.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(x_cpu);
        return;
    }

    nn::BatchNorm2d bn_dev(8);
    bn_dev.train();
    auto params = bn.parameters();
    auto dev_params = bn_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    bn_dev.to(device);

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = bn_dev.forward(x_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(x_cpu.grad().value(), x_dev.grad().value(),
                           1e-5f, 1e-3f);
}

// ============================================================================
// Loss Function Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, MSELossBackward) {
    auto pred_data = randn({4, 4}, DType::Float32, Device::cpu());
    auto target_data = randn({4, 4}, DType::Float32, Device::cpu());

    // CPU reference
    auto pred_cpu = Variable(pred_data.clone(), true);
    auto target_cpu = Variable(target_data.clone(), false);
    auto diff_cpu = pred_cpu - target_cpu;
    auto sq_cpu = diff_cpu * diff_cpu;
    auto loss_cpu = tenzor::mean(sq_cpu);
    loss_cpu.backward();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(pred_cpu);
        return;
    }

    // Backend
    auto pred_dev = Variable(pred_data.to(device), true);
    auto target_dev = Variable(target_data.to(device), false);
    auto diff_dev = pred_dev - target_dev;
    auto sq_dev = diff_dev * diff_dev;
    auto loss_dev = tenzor::mean(sq_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(pred_dev.has_grad());
    compareGradientWithCPU(pred_cpu.grad().value(), pred_dev.grad().value(), 1e-5f, 1e-5f);
}

TEST_P(GradientParityBackendTest, CrossEntropyLossBackward) {
    auto logits_data = randn({4, 10}, DType::Float32, Device::cpu());

    // CPU reference - softmax + log + neg (manual cross-entropy-like gradient)
    auto logits_cpu = Variable(logits_data.clone(), true);
    auto sm_cpu = tenzor::softmax(logits_cpu, 1);
    auto log_cpu = tenzor::log(sm_cpu);
    auto loss_cpu = tenzor::neg(tenzor::mean(log_cpu));
    loss_cpu.backward();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(logits_cpu);
        return;
    }

    // Backend
    auto logits_dev = Variable(logits_data.to(device), true);
    auto sm_dev = tenzor::softmax(logits_dev, 1);
    auto log_dev = tenzor::log(sm_dev);
    auto loss_dev = tenzor::neg(tenzor::mean(log_dev));
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(logits_dev.has_grad());
    compareGradientWithCPU(logits_cpu.grad().value(), logits_dev.grad().value(), 1e-4f, 1e-3f);
}

// ============================================================================
// Multi-step Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, ChainedOperations) {
    // x -> relu -> *2 -> sum -> backward
    auto x_data = randn({4, 8}, DType::Float32, Device::cpu());

    // CPU reference
    auto x_cpu = Variable(x_data.clone(), true);
    auto h_cpu = nn::relu(x_cpu);
    auto s_cpu = h_cpu * 2.0f;
    auto loss_cpu = tenzor::sum(s_cpu);
    loss_cpu.backward();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(x_cpu);
        return;
    }

    // Backend
    auto x_dev = Variable(x_data.to(device), true);
    auto h_dev = nn::relu(x_dev);
    auto s_dev = h_dev * 2.0f;
    auto loss_dev = tenzor::sum(s_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(x_cpu.grad().value(), x_dev.grad().value());
}

INSTANTIATE_BACKEND_TESTS(GradientParityBackendTest);
