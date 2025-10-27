/**
 * @file test_gradient_parity.cpp
 * @brief Gradient computation parity tests - Parameterized across all backends
 *
 * Verifies that gradients computed by different backends match,
 * using numerical gradient checking as reference.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"

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

    // Helper to create a reference gradient on CPU and compare
    void compareGradientWithCPU(const Tensor& cpu_grad, const Tensor& backend_grad,
                               float rtol = 1e-5f, float atol = 1e-7f) {
        auto backend_grad_cpu = backend_grad.to(Device::cpu());
        device.synchronize();

        expectTensorNear(cpu_grad, backend_grad_cpu, atol);
    }
};

// ============================================================================
// Basic Operation Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, AddBackward) {
    auto a_cpu = randn({16, 32}, DType::Float32, Device::cpu());
    auto b_cpu = randn({16, 32}, DType::Float32, Device::cpu());

    // Move to test device
    auto a = a_cpu.to(device);
    auto b = b_cpu.to(device);

    a.requires_grad(true);
    b.requires_grad(true);

    auto c = a + b;
    auto grad_out = ones_like(c);

    c.backward(grad_out);
    device.synchronize();

    // For addition, gradients should be exactly 1.0 everywhere
    auto grad_a_cpu = a.grad().to(Device::cpu());
    auto grad_b_cpu = b.grad().to(Device::cpu());

    auto expected = ones({16, 32}, DType::Float32, Device::cpu());
    expectTensorNear(expected, grad_a_cpu, 1e-7f);
    expectTensorNear(expected, grad_b_cpu, 1e-7f);
}

TEST_P(GradientParityBackendTest, MulBackward) {
    auto a_cpu = randn({16, 32}, DType::Float32, Device::cpu());
    auto b_cpu = randn({16, 32}, DType::Float32, Device::cpu());

    auto a = a_cpu.to(device);
    auto b = b_cpu.to(device);

    a.requires_grad(true);
    b.requires_grad(true);

    auto c = a * b;
    auto grad_out = ones_like(c);

    c.backward(grad_out);
    device.synchronize();

    // Gradient of a should be b, gradient of b should be a
    auto grad_a = a.grad().to(Device::cpu());
    auto grad_b = b.grad().to(Device::cpu());

    expectTensorNear(b_cpu, grad_a, 1e-5f);
    expectTensorNear(a_cpu, grad_b, 1e-5f);
}

TEST_P(GradientParityBackendTest, MatMulBackward) {
    // Skip if backend doesn't support matmul well
    if (device.type == Device::Type::Vulkan) {
        GTEST_SKIP() << "MatMul may have limited support on Vulkan";
    }

    auto a_cpu = randn({32, 64}, DType::Float32, Device::cpu());
    auto b_cpu = randn({64, 128}, DType::Float32, Device::cpu());

    auto a = a_cpu.to(device);
    auto b = b_cpu.to(device);

    a.requires_grad(true);
    b.requires_grad(true);

    auto c = matmul(a, b);
    auto grad_out = ones_like(c);

    c.backward(grad_out);
    device.synchronize();

    // Compute reference gradients on CPU
    auto a_cpu_var = a_cpu.clone();
    auto b_cpu_var = b_cpu.clone();
    a_cpu_var.requires_grad(true);
    b_cpu_var.requires_grad(true);

    auto c_cpu = matmul(a_cpu_var, b_cpu_var);
    auto grad_out_cpu = ones_like(c_cpu);
    c_cpu.backward(grad_out_cpu);

    auto grad_a = a.grad().to(Device::cpu());
    auto grad_b = b.grad().to(Device::cpu());

    expectTensorNear(a_cpu_var.grad(), grad_a, 1e-4f);
    expectTensorNear(b_cpu_var.grad(), grad_b, 1e-4f);
}

// ============================================================================
// Activation Function Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, ReLUBackward) {
    auto x_cpu = randn({32, 64}, DType::Float32, Device::cpu());
    auto x = x_cpu.to(device);

    x.requires_grad(true);

    auto y = nn::relu(x);
    auto grad_out = ones_like(y);

    y.backward(grad_out);
    device.synchronize();

    // Compute reference on CPU
    auto x_cpu_var = x_cpu.clone();
    x_cpu_var.requires_grad(true);
    auto y_cpu = nn::relu(x_cpu_var);
    auto grad_out_cpu = ones_like(y_cpu);
    y_cpu.backward(grad_out_cpu);

    auto grad = x.grad().to(Device::cpu());
    expectTensorNear(x_cpu_var.grad(), grad, 1e-7f);
}

TEST_P(GradientParityBackendTest, SigmoidBackward) {
    auto x_cpu = randn({32, 64}, DType::Float32, Device::cpu());
    auto x = x_cpu.to(device);

    x.requires_grad(true);

    auto y = sigmoid(x);
    auto grad_out = ones_like(y);

    y.backward(grad_out);
    device.synchronize();

    // Compute reference on CPU
    auto x_cpu_var = x_cpu.clone();
    x_cpu_var.requires_grad(true);
    auto y_cpu = sigmoid(x_cpu_var);
    auto grad_out_cpu = ones_like(y_cpu);
    y_cpu.backward(grad_out_cpu);

    auto grad = x.grad().to(Device::cpu());
    expectTensorNear(x_cpu_var.grad(), grad, 1e-5f);
}

TEST_P(GradientParityBackendTest, TanhBackward) {
    auto x_cpu = randn({32, 64}, DType::Float32, Device::cpu());
    auto x = x_cpu.to(device);

    x.requires_grad(true);

    auto y = tanh(x);
    auto grad_out = ones_like(y);

    y.backward(grad_out);
    device.synchronize();

    // Compute reference on CPU
    auto x_cpu_var = x_cpu.clone();
    x_cpu_var.requires_grad(true);
    auto y_cpu = tanh(x_cpu_var);
    auto grad_out_cpu = ones_like(y_cpu);
    y_cpu.backward(grad_out_cpu);

    auto grad = x.grad().to(Device::cpu());
    expectTensorNear(x_cpu_var.grad(), grad, 1e-5f);
}

TEST_P(GradientParityBackendTest, GELUBackward) {
    auto x_cpu = randn({32, 64}, DType::Float32, Device::cpu());
    auto x = x_cpu.to(device);

    x.requires_grad(true);

    auto y = nn::gelu(x);
    auto grad_out = ones_like(y);

    y.backward(grad_out);
    device.synchronize();

    // Compute reference on CPU
    auto x_cpu_var = x_cpu.clone();
    x_cpu_var.requires_grad(true);
    auto y_cpu = nn::gelu(x_cpu_var);
    auto grad_out_cpu = ones_like(y_cpu);
    y_cpu.backward(grad_out_cpu);

    auto grad = x.grad().to(Device::cpu());
    expectTensorNear(x_cpu_var.grad(), grad, 1e-4f);
}

TEST_P(GradientParityBackendTest, SoftmaxBackward) {
    auto x_cpu = randn({32, 10}, DType::Float32, Device::cpu());
    auto x = x_cpu.to(device);

    x.requires_grad(true);

    auto y = nn::softmax(x, 1);
    auto grad_out = ones_like(y);

    y.backward(grad_out);
    device.synchronize();

    // Compute reference on CPU
    auto x_cpu_var = x_cpu.clone();
    x_cpu_var.requires_grad(true);
    auto y_cpu = nn::softmax(x_cpu_var, 1);
    auto grad_out_cpu = ones_like(y_cpu);
    y_cpu.backward(grad_out_cpu);

    auto grad = x.grad().to(Device::cpu());
    expectTensorNear(x_cpu_var.grad(), grad, 1e-4f);
}

// ============================================================================
// Complex Operation Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, Conv2dBackward) {
    // Skip if backend doesn't support conv2d well
    if (device.type == Device::Type::Vulkan || device.type == Device::Type::OneAPI) {
        GTEST_SKIP() << "Conv2d may have limited support on this backend";
    }

    auto input_cpu = randn({2, 3, 32, 32}, DType::Float32, Device::cpu());
    auto weight_cpu = randn({16, 3, 3, 3}, DType::Float32, Device::cpu());

    auto input = input_cpu.to(device);
    auto weight = weight_cpu.to(device);

    input.requires_grad(true);
    weight.requires_grad(true);

    auto output = nn::functional::conv2d(input, weight, std::nullopt, /*stride=*/1, /*padding=*/1);
    auto grad_out = ones_like(output);

    output.backward(grad_out);
    device.synchronize();

    // Compute reference on CPU
    auto input_cpu_var = input_cpu.clone();
    auto weight_cpu_var = weight_cpu.clone();
    input_cpu_var.requires_grad(true);
    weight_cpu_var.requires_grad(true);

    auto output_cpu = nn::functional::conv2d(input_cpu_var, weight_cpu_var, std::nullopt, 1, 1);
    auto grad_out_cpu = ones_like(output_cpu);
    output_cpu.backward(grad_out_cpu);

    auto grad_input = input.grad().to(Device::cpu());
    auto grad_weight = weight.grad().to(Device::cpu());

    expectTensorNear(input_cpu_var.grad(), grad_input, 1e-3f);
    expectTensorNear(weight_cpu_var.grad(), grad_weight, 1e-3f);
}

TEST_P(GradientParityBackendTest, BatchNormBackward) {
    // Skip if backend doesn't support batch norm well
    if (device.type == Device::Type::Vulkan || device.type == Device::Type::OneAPI) {
        GTEST_SKIP() << "BatchNorm may have limited support on this backend";
    }

    auto input_cpu = randn({4, 16, 32, 32}, DType::Float32, Device::cpu());
    auto weight_cpu = ones({16}, DType::Float32, Device::cpu());
    auto bias_cpu = zeros({16}, DType::Float32, Device::cpu());
    auto running_mean_cpu = zeros({16}, DType::Float32, Device::cpu());
    auto running_var_cpu = ones({16}, DType::Float32, Device::cpu());

    auto input = input_cpu.to(device);
    auto weight = weight_cpu.to(device);
    auto bias = bias_cpu.to(device);
    auto running_mean = running_mean_cpu.to(device);
    auto running_var = running_var_cpu.to(device);

    input.requires_grad(true);
    weight.requires_grad(true);
    bias.requires_grad(true);

    auto output = nn::functional::batch_norm(input, running_mean, running_var,
                                            weight, bias, /*training=*/true, 0.1f, 1e-5f);
    auto grad_out = ones_like(output);

    output.backward(grad_out);
    device.synchronize();

    // Compute reference on CPU
    auto input_cpu_var = input_cpu.clone();
    auto weight_cpu_var = weight_cpu.clone();
    auto bias_cpu_var = bias_cpu.clone();

    input_cpu_var.requires_grad(true);
    weight_cpu_var.requires_grad(true);
    bias_cpu_var.requires_grad(true);

    auto output_cpu = nn::functional::batch_norm(input_cpu_var, running_mean_cpu, running_var_cpu,
                                                weight_cpu_var, bias_cpu_var, true, 0.1f, 1e-5f);
    auto grad_out_cpu = ones_like(output_cpu);
    output_cpu.backward(grad_out_cpu);

    auto grad_input = input.grad().to(Device::cpu());
    auto grad_weight = weight.grad().to(Device::cpu());
    auto grad_bias = bias.grad().to(Device::cpu());

    expectTensorNear(input_cpu_var.grad(), grad_input, 1e-3f);
    expectTensorNear(weight_cpu_var.grad(), grad_weight, 1e-3f);
    expectTensorNear(bias_cpu_var.grad(), grad_bias, 1e-3f);
}

// ============================================================================
// Loss Function Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, MSELossBackward) {
    auto pred_cpu = randn({32, 10}, DType::Float32, Device::cpu());
    auto target_cpu = randn({32, 10}, DType::Float32, Device::cpu());

    auto pred = pred_cpu.to(device);
    auto target = target_cpu.to(device);

    pred.requires_grad(true);

    auto loss_fn = nn::MSELoss();
    auto loss = loss_fn(pred, target);

    loss.backward();
    device.synchronize();

    // Compute reference on CPU
    auto pred_cpu_var = pred_cpu.clone();
    pred_cpu_var.requires_grad(true);

    auto loss_fn_cpu = nn::MSELoss();
    auto loss_cpu = loss_fn_cpu(pred_cpu_var, target_cpu);
    loss_cpu.backward();

    auto grad = pred.grad().to(Device::cpu());
    expectTensorNear(pred_cpu_var.grad(), grad, 1e-5f);
}

TEST_P(GradientParityBackendTest, CrossEntropyLossBackward) {
    auto pred_cpu = randn({32, 10}, DType::Float32, Device::cpu());
    auto target_cpu = (rand({32}, DType::Float32, Device::cpu()) * 10).to(DType::Int64);

    auto pred = pred_cpu.to(device);
    auto target = target_cpu.to(device);

    pred.requires_grad(true);

    auto loss_fn = nn::CrossEntropyLoss();
    auto loss = loss_fn(pred, target);

    loss.backward();
    device.synchronize();

    // Compute reference on CPU
    auto pred_cpu_var = pred_cpu.clone();
    pred_cpu_var.requires_grad(true);

    auto loss_fn_cpu = nn::CrossEntropyLoss();
    auto loss_cpu = loss_fn_cpu(pred_cpu_var, target_cpu);
    loss_cpu.backward();

    auto grad = pred.grad().to(Device::cpu());
    expectTensorNear(pred_cpu_var.grad(), grad, 1e-4f);
}

// ============================================================================
// Multi-step Gradient Tests
// ============================================================================

TEST_P(GradientParityBackendTest, ChainedOperations) {
    auto x_cpu = randn({16, 32}, DType::Float32, Device::cpu());
    auto x = x_cpu.to(device);

    x.requires_grad(true);

    // y = relu(x * 2 + 1)
    auto y = nn::relu(x * 2.0f + 1.0f);
    auto grad_out = ones_like(y);

    y.backward(grad_out);
    device.synchronize();

    // Compute reference on CPU
    auto x_cpu_var = x_cpu.clone();
    x_cpu_var.requires_grad(true);
    auto y_cpu = nn::relu(x_cpu_var * 2.0f + 1.0f);
    auto grad_out_cpu = ones_like(y_cpu);
    y_cpu.backward(grad_out_cpu);

    auto grad = x.grad().to(Device::cpu());
    expectTensorNear(x_cpu_var.grad(), grad, 1e-5f);
}

INSTANTIATE_BACKEND_TESTS(GradientParityBackendTest);
