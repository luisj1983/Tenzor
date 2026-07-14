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
#include <tenzor/nn/layers/dropout.hpp>
#include <tenzor/nn/layers/pooling.hpp>
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"

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
        expectTensorNear(cpu_grad, backend_grad_cpu, rtol, atol);
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
        EXPECT_GRAD_FLOWS(x_cpu);
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
        EXPECT_GRAD_FLOWS(x_cpu);
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
        EXPECT_GRAD_FLOWS(x_cpu);
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
        EXPECT_GRAD_FLOWS(x_cpu);
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

// Regression for the Vulkan MaxPool2d Float16 backward bug: the f16 backward
// shaders wrote F32 atomics into an F16-sized (2-byte/elem) grad_input buffer
// (OOB + corrupted adjacent halves), so the half-precision input gradient came
// back as garbage. The fix accumulates into a packed-F16 buffer via CAS,
// mirroring avg_pool2d_backward_f16. Exercises that path on every backend.
TEST_P(GradNNParityTest, MaxPool2dBackward_Half) {
    for (DType dt : {DType::Float16, DType::BFloat16}) {
        nn::MaxPool2d pool(2);
        // Use distinct, exactly-representable integer values (0..71) so the F16
        // argmax matches the F32 reference (no half-precision ties that would
        // route the unit gradient to a different in-window element). With the
        // OOB/garbage bug present the grad came back as garbage (large/NaN
        // values, lost mass); with the fix it is a clean 0/1 indicator matching
        // CPU exactly.
        auto input_data = arange(0.0, 72.0, 1.0, DType::Float32, Device::cpu())
                              .reshape({1, 2, 6, 6});

        auto x_cpu = Variable(input_data.clone(), true);
        auto out_cpu = pool.forward(x_cpu);
        auto loss_cpu = tenzor::sum(out_cpu);
        loss_cpu.backward();
        auto grad_cpu = x_cpu.grad().value();

        if (device.type == Device::Type::CPU) {
            EXPECT_GRAD_FLOWS(x_cpu);
            continue;
        }

        auto x_dev = Variable(input_data.to(dt).to(device), true);
        auto out_dev = pool.forward(x_dev);
        auto loss_dev = tenzor::sum(out_dev);
        loss_dev.backward();
        device.synchronize();

        ASSERT_TRUE(x_dev.has_grad()) << dtype_name(dt);
        auto grad_dev_f32 = x_dev.grad().value().to(DType::Float32).to(Device::cpu());
        compareGradientWithCPU(grad_cpu, grad_dev_f32, 5e-2f, 5e-2f);
    }
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
        EXPECT_GRAD_FLOWS(x_cpu);
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
    // sum(out) is degenerate for BatchNorm in train mode: affine=true with
    // weight=ones/bias=zeros normalizes to exactly zero mean, so the true
    // gradient is 0 identically — a broken backward kernel would pass
    // identically to a correct one. sum(out*out) exercises the real
    // gradient (mirrors test_gradient_parity.cpp::SoftmaxBackward's fix).
    auto loss_cpu = tenzor::sum(out_cpu * out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(x_cpu);
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
    auto loss_dev = tenzor::sum(out_dev * out_dev);
    loss_dev.backward();
    device.synchronize();

    EXPECT_GRAD_FLOWS(x_dev);
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

TEST_P(GradNNParityTest, LayerNormBackward) {
    nn::LayerNorm ln({16});
    auto input_data = randn({4, 16}, DType::Float32, Device::cpu());

    ln.to(Device::cpu());
    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = ln.forward(x_cpu);
    // sum(out) is degenerate: affine=true with weight=ones/bias=zeros
    // normalizes to exactly zero mean, so the true gradient is 0
    // identically — sum(out*out) exercises the real gradient.
    auto loss_cpu = tenzor::sum(out_cpu * out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(x_cpu);
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
    auto loss_dev = tenzor::sum(out_dev * out_dev);
    loss_dev.backward();
    device.synchronize();

    EXPECT_GRAD_FLOWS(x_dev);
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

// Regression for the FP16/BF16/FP64 LayerNorm backward grad_weight/grad_bias
// bug: the half/double cudnn_layer_norm_backward kernels reduced grad_weight /
// grad_bias with a cross-lane __shfl_down that summed DIFFERENT feature indices
// and only lane 0 stored, so every feature handled by a non-zero lane received
// ZERO grad_weight/grad_bias. The FP32 kernel was fixed to per-thread atomicAdd;
// the half/double kernels were not. This compares grad_weight AND grad_bias of a
// non-trivial-norm-size LayerNorm against the CPU Float32 reference for each of
// Float16, BFloat16, Float64. With the bug present, most grad_weight/grad_bias
// entries are 0 -> large divergence even at loose half tolerances.
TEST_P(GradNNParityTest, LayerNormBackward_WeightBias_HalfDouble) {
    if (device.type == Device::Type::CPU) GTEST_SKIP() << "device == CPU";
    // norm_size = 40 > warpSize so the broken shfl reduction drops features
    // 1..31 within each warp; pick a non-multiple-of-32 to also catch the
    // partial-warp undefined-shfl case.
    const int64_t F = 40;
    auto input_data  = randn({6, F}, DType::Float32, Device::cpu());
    auto weight_data = randn({F},    DType::Float32, Device::cpu());
    auto bias_data   = randn({F},    DType::Float32, Device::cpu());
    auto gout_data   = randn({6, F}, DType::Float32, Device::cpu());

    // CPU Float32 ground truth for grad_weight / grad_bias.
    nn::LayerNorm ln_cpu({F});
    auto cpu_params = ln_cpu.parameters();
    cpu_params[0]->tensor() = weight_data.clone();  // weight
    cpu_params[1]->tensor() = bias_data.clone();    // bias
    ln_cpu.to(Device::cpu());
    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = ln_cpu.forward(x_cpu);
    // Weight the output by a fixed grad so grad_weight/grad_bias are non-trivial.
    auto loss_cpu = tenzor::sum(out_cpu * Variable(gout_data.clone(), false));
    loss_cpu.backward();
    auto gw_cpu = cpu_params[0]->grad().value();
    auto gb_cpu = cpu_params[1]->grad().value();

    for (DType dt : {DType::Float16, DType::BFloat16, DType::Float64}) {
        SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int>(dt));
        nn::LayerNorm ln_dev({F});
        auto dev_params = ln_dev.parameters();
        dev_params[0]->tensor() = weight_data.clone();
        dev_params[1]->tensor() = bias_data.clone();
        ln_dev.to(device);
        ln_dev.to(dt);

        auto x_dev = Variable(input_data.to(device).to(dt), true);
        auto out_dev = ln_dev.forward(x_dev);
        auto loss_dev = tenzor::sum(out_dev * Variable(gout_data.to(device).to(dt), false));
        loss_dev.backward();
        device.synchronize();

        ASSERT_TRUE(dev_params[0]->has_grad());
        ASSERT_TRUE(dev_params[1]->has_grad());
        // grad_weight / grad_bias back to Float32 for comparison.
        auto gw_dev = dev_params[0]->grad().value().to(Device::cpu()).to(DType::Float32);
        auto gb_dev = dev_params[1]->grad().value().to(Device::cpu()).to(DType::Float32);
        // Half precision: loose tol; the bug produces ~0 so any non-tiny tol catches it.
        float rtol = (dt == DType::Float64) ? 1e-5f : 2e-2f;
        float atol = (dt == DType::Float64) ? 1e-6f : 3e-2f;
        expectTensorNear(gw_cpu, gw_dev, rtol, atol);
        expectTensorNear(gb_cpu, gb_dev, rtol, atol);
    }
}

TEST_P(GradNNParityTest, GroupNormBackward) {
    nn::GroupNorm gn(4, 8);
    auto input_data = randn({2, 8, 4, 4}, DType::Float32, Device::cpu());

    gn.to(Device::cpu());
    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = gn.forward(x_cpu);
    // sum(out) is degenerate: affine=true with weight=ones/bias=zeros
    // normalizes to exactly zero mean, so the true gradient is 0
    // identically — sum(out*out) exercises the real gradient.
    auto loss_cpu = tenzor::sum(out_cpu * out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(x_cpu);
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
    auto loss_dev = tenzor::sum(out_dev * out_dev);
    loss_dev.backward();
    device.synchronize();

    EXPECT_GRAD_FLOWS(x_dev);
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
        EXPECT_GRAD_FLOWS(*weight_params[0]);
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
        EXPECT_GRAD_FLOWS(x_cpu);
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
        EXPECT_GRAD_FLOWS(x_cpu);
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
    // sum(out) is degenerate: affine=true with weight=ones/bias=zeros
    // normalizes to exactly zero mean, so the true gradient is 0
    // identically — sum(out*out) exercises the real gradient.
    auto loss_cpu = tenzor::sum(out_cpu * out_cpu);
    loss_cpu.backward();
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) {
        EXPECT_GRAD_FLOWS(x_cpu);
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
    auto loss_dev = tenzor::sum(out_dev * out_dev);
    loss_dev.backward();
    device.synchronize();

    EXPECT_GRAD_FLOWS(x_dev);
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-3f);
}

// ============================================================================
// D2 expansion — additional layer backward parity tests.
// ============================================================================

TEST_P(GradNNParityTest, ConvTranspose2dBackward) {
    auto input_data = randn({2, 4, 6, 6}, DType::Float32, Device::cpu());
    auto weight_data = randn({4, 3, 3, 3}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    auto w_cpu = Variable(weight_data.clone(), false);
    auto out_cpu = tenzor::nn::functional::conv_transpose2d(x_cpu, w_cpu);
    tenzor::sum(out_cpu).backward();
    EXPECT_GRAD_FLOWS(x_cpu);
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) return;

    auto x_dev = Variable(input_data.to(device), true);
    auto w_dev = Variable(weight_data.to(device), false);
    auto out_dev = tenzor::nn::functional::conv_transpose2d(x_dev, w_dev);
    tenzor::sum(out_dev).backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-4f, 1e-3f);
}

TEST_P(GradNNParityTest, AdaptiveAvgPool2dBackward) {
    auto input_data = randn({2, 4, 8, 8}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = tenzor::nn::functional::adaptive_avg_pool2d(x_cpu, {4, 4});
    tenzor::sum(out_cpu).backward();
    EXPECT_GRAD_FLOWS(x_cpu);
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) return;

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = tenzor::nn::functional::adaptive_avg_pool2d(x_dev, {4, 4});
    tenzor::sum(out_dev).backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-6f);
}

TEST_P(GradNNParityTest, DropoutEvalBackward) {
    // Dropout in eval mode is identity; its backward must simply pass through
    // the incoming gradient. This catches regressions where a backend emits
    // a mask even in eval.
    auto input_data = randn({4, 8}, DType::Float32, Device::cpu());

    auto make_dropout_eval = []() {
        nn::Dropout d(0.5);
        d.eval();
        return d;
    };

    auto x_cpu = Variable(input_data.clone(), true);
    auto dc_cpu = make_dropout_eval();
    auto out_cpu = dc_cpu.forward(x_cpu);
    tenzor::sum(out_cpu).backward();
    EXPECT_GRAD_FLOWS(x_cpu);
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) return;

    auto dc_dev = make_dropout_eval();
    dc_dev.to(device);
    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = dc_dev.forward(x_dev);
    tenzor::sum(out_dev).backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 0.0f, 0.0f);
}

// ============================================================================
// D3 expansion — backward parity for *Backward OpIds that weren't in the
// original GradNNParityTest set: pool 1D, softmax, log_softmax, gelu.
// These are common training-path ops where a backend-specific backward
// divergence would silently produce wrong gradients at scale.
// ============================================================================

TEST_P(GradNNParityTest, MaxPool1dBackward) {
    auto input_data = randn({2, 4, 16}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    ::tenzor::nn::MaxPool1d pool(/*kernel=*/2, /*stride=*/2, /*padding=*/0);
    auto out_cpu = pool.forward(x_cpu);
    tenzor::sum(out_cpu).backward();
    EXPECT_GRAD_FLOWS(x_cpu);
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) return;

    ::tenzor::nn::MaxPool1d pool_dev(/*kernel=*/2, /*stride=*/2, /*padding=*/0);
    pool_dev.to(device);
    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = pool_dev.forward(x_dev);
    tenzor::sum(out_dev).backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-6f);
}

TEST_P(GradNNParityTest, AvgPool1dBackward) {
    auto input_data = randn({2, 4, 16}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    ::tenzor::nn::AvgPool1d pool(/*kernel=*/2, /*stride=*/2, /*padding=*/0);
    auto out_cpu = pool.forward(x_cpu);
    tenzor::sum(out_cpu).backward();
    EXPECT_GRAD_FLOWS(x_cpu);
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) return;

    ::tenzor::nn::AvgPool1d pool_dev(/*kernel=*/2, /*stride=*/2, /*padding=*/0);
    pool_dev.to(device);
    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = pool_dev.forward(x_dev);
    tenzor::sum(out_dev).backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-6f);
}

TEST_P(GradNNParityTest, SoftmaxBackward) {
    auto input_data = randn({4, 8}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = tenzor::softmax(x_cpu, /*dim=*/-1);
    // softmax sums to 1 along dim; compose with a non-uniform weight so
    // sum(loss).backward() isn't identically zero.
    auto w = randn({4, 8}, DType::Float32, Device::cpu());
    tenzor::sum(out_cpu * Variable(w, false)).backward();
    EXPECT_GRAD_FLOWS(x_cpu);
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) return;

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = tenzor::softmax(x_dev, /*dim=*/-1);
    tenzor::sum(out_dev * Variable(w.to(device), false)).backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-5f);
}

TEST_P(GradNNParityTest, LogSoftmaxBackward) {
    auto input_data = randn({4, 8}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = tenzor::log_softmax(x_cpu, /*dim=*/-1);
    tenzor::sum(out_cpu).backward();
    EXPECT_GRAD_FLOWS(x_cpu);
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) return;

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = tenzor::log_softmax(x_dev, /*dim=*/-1);
    tenzor::sum(out_dev).backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-5f, 1e-6f);
}

TEST_P(GradNNParityTest, GeluBackward) {
    auto input_data = randn({6, 10}, DType::Float32, Device::cpu());

    auto x_cpu = Variable(input_data.clone(), true);
    auto out_cpu = tenzor::gelu(x_cpu);
    tenzor::sum(out_cpu).backward();
    EXPECT_GRAD_FLOWS(x_cpu);
    auto grad_cpu = x_cpu.grad().value();

    if (device.type == Device::Type::CPU) return;

    auto x_dev = Variable(input_data.to(device), true);
    auto out_dev = tenzor::gelu(x_dev);
    tenzor::sum(out_dev).backward();
    device.synchronize();

    ASSERT_TRUE(x_dev.has_grad());
    compareGradientWithCPU(grad_cpu, x_dev.grad().value(), 1e-4f, 1e-5f);
}

INSTANTIATE_BACKEND_TESTS(GradNNParityTest);
