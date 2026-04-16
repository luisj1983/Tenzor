/**
 * @file test_grad_nn_parity.cpp
 * @brief Neural network layer gradient parity tests - Parameterized across all backends
 *
 * Verifies that gradients for neural network layers computed by different
 * backends match the CPU reference implementation.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradNNParityTest : public BackendTest {
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
        float atol = 1e-5f)
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
// Neural Network Layer Gradient Tests
// ============================================================================

TEST_P(GradNNParityTest, Conv2dBackward) {
    nn::Conv2d conv(3, 8, 3, 1, 1);
    auto input_data = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());

    // CPU backward
    auto x_cpu = Variable(input_data.clone(), true);
    conv.to(Device::cpu());
    auto out_cpu = conv.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    // Backend backward
    nn::Conv2d conv_dev(3, 8, 3, 1, 1);
    auto params = conv.parameters();
    auto dev_params = conv_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    conv_dev.to(device);

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = conv_dev.forward(x_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradNNParityTest, Conv1dBackward) {
    nn::Conv1d conv(3, 8, 3, 1, 1);
    auto input_data = randn({1, 3, 16}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    conv.to(Device::cpu());
    auto out_cpu = conv.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    nn::Conv1d conv_dev(3, 8, 3, 1, 1);
    auto params = conv.parameters();
    auto dev_params = conv_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    conv_dev.to(device);

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = conv_dev.forward(x_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradNNParityTest, Conv3dBackward) {
    nn::Conv3d conv(2, 4, 3, 1, 1);
    auto input_data = randn({1, 2, 4, 4, 4}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    conv.to(Device::cpu());
    auto out_cpu = conv.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    nn::Conv3d conv_dev(2, 4, 3, 1, 1);
    auto params = conv.parameters();
    auto dev_params = conv_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    conv_dev.to(device);

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = conv_dev.forward(x_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-2f);
}

TEST_P(GradNNParityTest, MaxPool2dBackward) {
    nn::MaxPool2d pool(2);
    auto input_data = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = pool.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = pool.forward(x_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-4f);
}

TEST_P(GradNNParityTest, AvgPool2dBackward) {
    nn::AvgPool2d pool(2);
    auto input_data = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = pool.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = pool.forward(x_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-4f);
}

TEST_P(GradNNParityTest, BatchNorm2dBackward) {
    nn::BatchNorm2d bn(8);
    auto input_data = randn({2, 8, 4, 4}, DType::Float32, Device::cpu());

    // CPU backward (train mode)
    bn.train();
    bn.to(Device::cpu());
    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = bn.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    // Backend backward
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
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradNNParityTest, LayerNormBackward) {
    nn::LayerNorm ln({16});
    auto input_data = randn({4, 16}, DType::Float32, Device::cpu());

    ln.to(Device::cpu());
    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = ln.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    nn::LayerNorm ln_dev({16});
    auto params = ln.parameters();
    auto dev_params = ln_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    ln_dev.to(device);

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = ln_dev.forward(x_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradNNParityTest, GroupNormBackward) {
    nn::GroupNorm gn(4, 8);
    auto input_data = randn({2, 8, 4, 4}, DType::Float32, Device::cpu());

    gn.to(Device::cpu());
    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = gn.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    nn::GroupNorm gn_dev(4, 8);
    auto params = gn.parameters();
    auto dev_params = gn_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    gn_dev.to(device);

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = gn_dev.forward(x_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradNNParityTest, EmbeddingBackward) {
    nn::Embedding emb(100, 32);
    // Int64 input indices
    auto idx_data = tenzor::randint(0, 100, {4, 10}, DType::Int64, Device::cpu());

    emb.to(Device::cpu());
    auto idx_cpu = Variable(idx_data.clone(), false);  // indices not differentiable
    auto out_cpu = emb.forward(idx_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();

    // For Embedding, gradient is on the weight, not the input
    auto weight_params = emb.parameters();
    ASSERT_FALSE(weight_params.empty());
    auto weight_grad_cpu = weight_params[0]->grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(weight_params[0]->has_grad());
        return;
    }

    nn::Embedding emb_dev(100, 32);
    auto params = emb.parameters();
    auto dev_params = emb_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    emb_dev.to(device);

    auto idx_dev = Variable(idx_data.to(device), false);
    auto out_dev = emb_dev.forward(idx_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    auto weight_params_dev = emb_dev.parameters();
    ASSERT_TRUE(weight_params_dev[0]->has_grad());
    compareGradientWithCPU(weight_grad_cpu, weight_params_dev[0]->grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradNNParityTest, LSTMCellBackward) {
    nn::LSTMCell lstm(32, 64);
    auto input_data = randn({4, 32}, DType::Float32, Device::cpu());
    auto h_data = randn({4, 64}, DType::Float32, Device::cpu());
    auto c_data = randn({4, 64}, DType::Float32, Device::cpu());

    lstm.to(Device::cpu());
    auto x_cpu = Variable(input_data.clone(), true);
    auto h_cpu = Variable(h_data.clone(), true);
    auto c_cpu = Variable(c_data.clone(), true);
    auto [h_next_cpu, c_next_cpu] = lstm.forward(x_cpu, h_cpu, c_cpu);
    auto loss_cpu = tenzor::sum(h_next_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    nn::LSTMCell lstm_dev(32, 64);
    auto params = lstm.parameters();
    auto dev_params = lstm_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    lstm_dev.to(device);

    auto x_dev = Variable(input_data.to(device), true);
    auto h_dev = Variable(h_data.to(device), true);
    auto c_dev = Variable(c_data.to(device), true);
    auto [h_next_dev, c_next_dev] = lstm_dev.forward(x_dev, h_dev, c_dev);
    auto loss_dev = tenzor::sum(h_next_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradNNParityTest, GRUCellBackward) {
    nn::GRUCell gru(32, 64);
    auto input_data = randn({4, 32}, DType::Float32, Device::cpu());
    auto h_data = randn({4, 64}, DType::Float32, Device::cpu());

    gru.to(Device::cpu());
    auto x_cpu = Variable(input_data.clone(), true);
    auto h_cpu = Variable(h_data.clone(), true);
    auto h_next_cpu = gru.forward(x_cpu, h_cpu);
    auto loss_cpu = tenzor::sum(h_next_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    nn::GRUCell gru_dev(32, 64);
    auto params = gru.parameters();
    auto dev_params = gru_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    gru_dev.to(device);

    auto x_dev = Variable(input_data.to(device), true);
    auto h_dev = Variable(h_data.to(device), true);
    auto h_next_dev = gru_dev.forward(x_dev, h_dev);
    auto loss_dev = tenzor::sum(h_next_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradNNParityTest, InstanceNormBackward) {
    nn::InstanceNorm2d in(8);
    auto input_data = randn({2, 8, 4, 4}, DType::Float32, Device::cpu());

    in.to(Device::cpu());
    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = in.forward(x_cpu);
    auto loss_cpu = tenzor::sum(out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        ASSERT_TRUE(x_cpu.has_grad());
        return;
    }

    nn::InstanceNorm2d in_dev(8);
    auto params = in.parameters();
    auto dev_params = in_dev.parameters();
    for (size_t p = 0; p < params.size(); ++p)
        dev_params[p]->tensor() = params[p]->tensor().clone();
    in_dev.to(device);

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = in_dev.forward(x_dev);
    auto loss_dev = tenzor::sum(out_dev);
    loss_dev.backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

INSTANTIATE_BACKEND_TESTS(GradNNParityTest);
