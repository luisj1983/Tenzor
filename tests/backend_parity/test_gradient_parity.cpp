/**
 * @file test_gradient_parity.cpp
 * @brief Gradient computation parity tests
 *
 * Verifies that gradients computed by different backends match,
 * using numerical gradient checking as reference.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Basic Operation Gradient Tests
// ============================================================================

TEST(GradientParity, AddBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({16, 32}, DType::Float32, Device::cpu());
    auto b = randn({16, 32}, DType::Float32, Device::cpu());

    a.requires_grad(true);
    b.requires_grad(true);

    std::vector<Tensor> cpu_results;
    std::vector<Tensor> cpu_grads_a;
    std::vector<Tensor> cpu_grads_b;

    for (const auto& backend : backends) {
        auto a_dev = a.to(backend);
        auto b_dev = b.to(backend);
        a_dev.requires_grad(true);
        b_dev.requires_grad(true);

        auto c = a_dev + b_dev;
        auto grad_out = ones_like(c);

        c.backward(grad_out);
        backend.synchronize();

        cpu_results.push_back(c.to(Device::cpu()));
        cpu_grads_a.push_back(a_dev.grad().to(Device::cpu()));
        cpu_grads_b.push_back(b_dev.grad().to(Device::cpu()));
    }

    // Compare gradients across backends
    for (size_t i = 1; i < cpu_grads_a.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads_a[0], cpu_grads_a[i], 1e-6f, 1e-8f);
        EXPECT_TENSORS_CLOSE(cpu_grads_b[0], cpu_grads_b[i], 1e-6f, 1e-8f);
    }
}

TEST(GradientParity, MulBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({16, 32}, DType::Float32, Device::cpu());
    auto b = randn({16, 32}, DType::Float32, Device::cpu());

    a.requires_grad(true);
    b.requires_grad(true);

    std::vector<Tensor> cpu_grads_a;
    std::vector<Tensor> cpu_grads_b;

    for (const auto& backend : backends) {
        auto a_dev = a.to(backend);
        auto b_dev = b.to(backend);
        a_dev.requires_grad(true);
        b_dev.requires_grad(true);

        auto c = a_dev * b_dev;
        auto grad_out = ones_like(c);

        c.backward(grad_out);
        backend.synchronize();

        cpu_grads_a.push_back(a_dev.grad().to(Device::cpu()));
        cpu_grads_b.push_back(b_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads_a.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads_a[0], cpu_grads_a[i], 1e-5f, 1e-7f);
        EXPECT_TENSORS_CLOSE(cpu_grads_b[0], cpu_grads_b[i], 1e-5f, 1e-7f);
    }
}

TEST(GradientParity, MatMulBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu());
    auto b = randn({64, 128}, DType::Float32, Device::cpu());

    a.requires_grad(true);
    b.requires_grad(true);

    std::vector<Tensor> cpu_grads_a;
    std::vector<Tensor> cpu_grads_b;

    for (const auto& backend : backends) {
        auto a_dev = a.to(backend);
        auto b_dev = b.to(backend);
        a_dev.requires_grad(true);
        b_dev.requires_grad(true);

        auto c = matmul(a_dev, b_dev);
        auto grad_out = ones_like(c);

        c.backward(grad_out);
        backend.synchronize();

        cpu_grads_a.push_back(a_dev.grad().to(Device::cpu()));
        cpu_grads_b.push_back(b_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads_a.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads_a[0], cpu_grads_a[i], 1e-4f, 1e-6f);
        EXPECT_TENSORS_CLOSE(cpu_grads_b[0], cpu_grads_b[i], 1e-4f, 1e-6f);
    }
}

// ============================================================================
// Activation Function Gradient Tests
// ============================================================================

TEST(GradientParity, ReLUBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({32, 64}, DType::Float32, Device::cpu());
    x.requires_grad(true);

    std::vector<Tensor> cpu_grads;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);
        x_dev.requires_grad(true);

        auto y = nn::relu(x_dev);
        auto grad_out = ones_like(y);

        y.backward(grad_out);
        backend.synchronize();

        cpu_grads.push_back(x_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads[0], cpu_grads[i], 1e-7f, 1e-9f);
    }
}

TEST(GradientParity, SigmoidBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({32, 64}, DType::Float32, Device::cpu());
    x.requires_grad(true);

    std::vector<Tensor> cpu_grads;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);
        x_dev.requires_grad(true);

        auto y = sigmoid(x_dev);
        auto grad_out = ones_like(y);

        y.backward(grad_out);
        backend.synchronize();

        cpu_grads.push_back(x_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads[0], cpu_grads[i], 1e-6f, 1e-8f);
    }
}

TEST(GradientParity, TanhBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({32, 64}, DType::Float32, Device::cpu());
    x.requires_grad(true);

    std::vector<Tensor> cpu_grads;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);
        x_dev.requires_grad(true);

        auto y = tanh(x_dev);
        auto grad_out = ones_like(y);

        y.backward(grad_out);
        backend.synchronize();

        cpu_grads.push_back(x_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads[0], cpu_grads[i], 1e-6f, 1e-8f);
    }
}

TEST(GradientParity, GELUBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({32, 64}, DType::Float32, Device::cpu());
    x.requires_grad(true);

    std::vector<Tensor> cpu_grads;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);
        x_dev.requires_grad(true);

        auto y = nn::gelu(x_dev);
        auto grad_out = ones_like(y);

        y.backward(grad_out);
        backend.synchronize();

        cpu_grads.push_back(x_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads[0], cpu_grads[i], 1e-5f, 1e-7f);
    }
}

TEST(GradientParity, SoftmaxBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({32, 10}, DType::Float32, Device::cpu());
    x.requires_grad(true);

    std::vector<Tensor> cpu_grads;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);
        x_dev.requires_grad(true);

        auto y = nn::softmax(x_dev, 1);
        auto grad_out = ones_like(y);

        y.backward(grad_out);
        backend.synchronize();

        cpu_grads.push_back(x_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads[0], cpu_grads[i], 1e-5f, 1e-7f);
    }
}

// ============================================================================
// Complex Operation Gradient Tests
// ============================================================================

TEST(GradientParity, Conv2dBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 3, 32, 32}, DType::Float32, Device::cpu());
    auto weight = randn({16, 3, 3, 3}, DType::Float32, Device::cpu());

    input.requires_grad(true);
    weight.requires_grad(true);

    std::vector<Tensor> cpu_grads_input;
    std::vector<Tensor> cpu_grads_weight;

    for (const auto& backend : backends) {
        auto input_dev = input.to(backend);
        auto weight_dev = weight.to(backend);
        input_dev.requires_grad(true);
        weight_dev.requires_grad(true);

        auto output = nn::functional::conv2d(input_dev, weight_dev, std::nullopt,
                                            /*stride=*/1, /*padding=*/1);
        auto grad_out = ones_like(output);

        output.backward(grad_out);
        backend.synchronize();

        cpu_grads_input.push_back(input_dev.grad().to(Device::cpu()));
        cpu_grads_weight.push_back(weight_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads_input.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads_input[0], cpu_grads_input[i], 1e-4f, 1e-6f);
        EXPECT_TENSORS_CLOSE(cpu_grads_weight[0], cpu_grads_weight[i], 1e-4f, 1e-6f);
    }
}

TEST(GradientParity, BatchNormBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 16, 32, 32}, DType::Float32, Device::cpu());
    auto weight = ones({16}, DType::Float32, Device::cpu());
    auto bias = zeros({16}, DType::Float32, Device::cpu());
    auto running_mean = zeros({16}, DType::Float32, Device::cpu());
    auto running_var = ones({16}, DType::Float32, Device::cpu());

    input.requires_grad(true);
    weight.requires_grad(true);
    bias.requires_grad(true);

    std::vector<Tensor> cpu_grads_input;
    std::vector<Tensor> cpu_grads_weight;
    std::vector<Tensor> cpu_grads_bias;

    for (const auto& backend : backends) {
        auto input_dev = input.to(backend);
        auto weight_dev = weight.to(backend);
        auto bias_dev = bias.to(backend);
        auto rm_dev = running_mean.to(backend);
        auto rv_dev = running_var.to(backend);

        input_dev.requires_grad(true);
        weight_dev.requires_grad(true);
        bias_dev.requires_grad(true);

        auto output = nn::functional::batch_norm(input_dev, rm_dev, rv_dev,
                                                weight_dev, bias_dev,
                                                /*training=*/true, 0.1f, 1e-5f);
        auto grad_out = ones_like(output);

        output.backward(grad_out);
        backend.synchronize();

        cpu_grads_input.push_back(input_dev.grad().to(Device::cpu()));
        cpu_grads_weight.push_back(weight_dev.grad().to(Device::cpu()));
        cpu_grads_bias.push_back(bias_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads_input.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads_input[0], cpu_grads_input[i], 1e-4f, 1e-6f);
        EXPECT_TENSORS_CLOSE(cpu_grads_weight[0], cpu_grads_weight[i], 1e-4f, 1e-6f);
        EXPECT_TENSORS_CLOSE(cpu_grads_bias[0], cpu_grads_bias[i], 1e-4f, 1e-6f);
    }
}

// ============================================================================
// Loss Function Gradient Tests
// ============================================================================

TEST(GradientParity, MSELossBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto pred = randn({32, 10}, DType::Float32, Device::cpu());
    auto target = randn({32, 10}, DType::Float32, Device::cpu());

    pred.requires_grad(true);

    std::vector<Tensor> cpu_grads;

    for (const auto& backend : backends) {
        auto pred_dev = pred.to(backend);
        auto target_dev = target.to(backend);
        pred_dev.requires_grad(true);

        auto loss_fn = nn::MSELoss();
        auto loss = loss_fn(pred_dev, target_dev);

        loss.backward();
        backend.synchronize();

        cpu_grads.push_back(pred_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads[0], cpu_grads[i], 1e-6f, 1e-8f);
    }
}

TEST(GradientParity, CrossEntropyLossBackward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto pred = randn({32, 10}, DType::Float32, Device::cpu());
    auto target = (rand({32}, DType::Float32, Device::cpu()) * 10).to(DType::Int64);

    pred.requires_grad(true);

    std::vector<Tensor> cpu_grads;

    for (const auto& backend : backends) {
        auto pred_dev = pred.to(backend);
        auto target_dev = target.to(backend);
        pred_dev.requires_grad(true);

        auto loss_fn = nn::CrossEntropyLoss();
        auto loss = loss_fn(pred_dev, target_dev);

        loss.backward();
        backend.synchronize();

        cpu_grads.push_back(pred_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads[0], cpu_grads[i], 1e-5f, 1e-7f);
    }
}

// ============================================================================
// Multi-step Gradient Tests
// ============================================================================

TEST(GradientParity, ChainedOperations) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({16, 32}, DType::Float32, Device::cpu());
    x.requires_grad(true);

    std::vector<Tensor> cpu_grads;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);
        x_dev.requires_grad(true);

        // y = relu(x * 2 + 1)
        auto y = nn::relu(x_dev * 2.0f + 1.0f);
        auto grad_out = ones_like(y);

        y.backward(grad_out);
        backend.synchronize();

        cpu_grads.push_back(x_dev.grad().to(Device::cpu()));
    }

    for (size_t i = 1; i < cpu_grads.size(); ++i) {
        EXPECT_TENSORS_CLOSE(cpu_grads[0], cpu_grads[i], 1e-6f, 1e-8f);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
