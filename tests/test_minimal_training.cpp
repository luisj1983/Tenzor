/**
 * @file test_minimal_training.cpp
 * @brief Minimal test to debug NaN issue in training
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;

class MinimalTraining : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        // Fixed seed to keep std::rand()-driven test inputs reproducible.
        std::srand(42);
    }
};

TEST_F(MinimalTraining, SimpleLinearRegression) {
    // Create very simple model: just one linear layer
    Linear model(2, 1, true);  // 2 inputs -> 1 output

    auto params = model.parameters();
    std::cout << "Number of parameters: " << params.size() << "\n";

    // Print initial weights
    auto& weight = *params[0];
    auto& bias = *params[1];
    std::cout << "Initial weight[0,0]: " << weight.tensor().data<float>()[0] << "\n";
    std::cout << "Initial bias[0]: " << bias.tensor().data<float>()[0] << "\n";

    // Create optimizer with small learning rate
    Adam optimizer(params, 0.01);

    // Simple training data: y = 2*x1 + 3*x2 + 1
    auto x = Tensor({4, 2}, DType::Float32, Device::cpu());
    auto* x_data = x.data<float>();
    x_data[0] = 1.0f; x_data[1] = 2.0f;  // [1, 2]
    x_data[2] = 3.0f; x_data[3] = 4.0f;  // [3, 4]
    x_data[4] = 5.0f; x_data[5] = 6.0f;  // [5, 6]
    x_data[6] = 7.0f; x_data[7] = 8.0f;  // [7, 8]

    auto y = Tensor({4, 1}, DType::Float32, Device::cpu());
    auto* y_data = y.data<float>();
    y_data[0] = 2*1.0f + 3*2.0f + 1.0f;  // 9
    y_data[1] = 2*3.0f + 3*4.0f + 1.0f;  // 19
    y_data[2] = 2*5.0f + 3*6.0f + 1.0f;  // 29
    y_data[3] = 2*7.0f + 3*8.0f + 1.0f;  // 39

    std::cout << "Target y[0]: " << y_data[0] << "\n";

    // Training loop
    for (int step = 0; step < 10; ++step) {
        optimizer.zero_grad();

        // Forward pass
        auto x_var = Variable(x, false);
        auto output = model.forward(x_var);

        // Compute MSE loss manually
        auto diff = output.tensor() - y;
        auto squared = diff * diff;

        // Sum and mean
        float loss_value = 0.0f;
        auto* sq_data = squared.data<float>();
        for (int i = 0; i < 4; ++i) {
            loss_value += sq_data[i];
        }
        loss_value /= 4.0f;

        std::cout << "Step " << step << ": loss = " << loss_value;

        // Check for NaN
        if (std::isnan(loss_value)) {
            std::cout << " <- NaN detected!\n";

            // Print outputs
            auto* out_data = output.tensor().data<float>();
            std::cout << "  Outputs: [" << out_data[0] << ", " << out_data[1]
                      << ", " << out_data[2] << ", " << out_data[3] << "]\n";

            // Print weights
            std::cout << "  Weight[0,0]: " << weight.tensor().data<float>()[0] << "\n";
            std::cout << "  Bias[0]: " << bias.tensor().data<float>()[0] << "\n";

            FAIL() << "NaN detected at step " << step;
        }
        std::cout << "\n";

        // Backward pass (create scalar variable for loss)
        auto loss_tensor = Tensor({1}, DType::Float32, Device::cpu());
        loss_tensor.data<float>()[0] = loss_value;
        auto loss_var = Variable(loss_tensor, true);

        // We need to compute gradients through the computation graph
        // For now, let's use MSE from nn
        auto y_var = Variable(y, false);
        auto mse_loss_var = mse_loss(output, y_var);

        mse_loss_var.backward();

        // Check gradients
        if (weight.has_grad()) {
            auto& grad = weight.grad().value();
            float grad_val = grad.data<float>()[0];
            if (std::isnan(grad_val)) {
                std::cout << "  Gradient is NaN!\n";
                FAIL() << "Gradient NaN at step " << step;
            }
            std::cout << "  Gradient[0,0]: " << grad_val << "\n";
        }

        // Optimizer step
        optimizer.step();

        // Print updated weight
        std::cout << "  Updated weight[0,0]: " << weight.tensor().data<float>()[0] << "\n";
    }

    std::cout << "Training completed successfully!\n";
}

TEST_F(MinimalTraining, StandardAdamOnly) {
    // Test standard Adam without ZeRO to isolate the issue
    Linear model(10, 5, true);
    auto params = model.parameters();

    Adam optimizer(params, 0.001);

    for (int step = 0; step < 20; ++step) {
        optimizer.zero_grad();

        // Random input
        auto x = randn({8, 10}, DType::Float32, Device::cpu());
        auto y = randn({8, 5}, DType::Float32, Device::cpu());

        auto x_var = Variable(x, false);
        auto y_var = Variable(y, false);

        auto output = model.forward(x_var);
        auto loss = mse_loss(output, y_var);

        float loss_val = loss.tensor().template data<float>()[0];

        if (step % 5 == 0) {
            std::cout << "Step " << step << ": loss = " << loss_val << "\n";
        }

        if (std::isnan(loss_val)) {
            FAIL() << "NaN at step " << step;
        }

        loss.backward();
        optimizer.step();
    }

    std::cout << "Standard Adam test passed!\n";
}

TEST_F(MinimalTraining, IntegrationTestSetup) {
    // Replicate the exact setup from integration tests
    // Model: 784 -> 256 -> 128 -> 10 (MLP)
    auto seq = Sequential();
    seq.add_module(std::make_shared<Linear>(784, 256))
       .add_module(std::make_shared<ReLU>())
       .add_module(std::make_shared<Linear>(256, 128))
       .add_module(std::make_shared<ReLU>())
       .add_module(std::make_shared<Linear>(128, 10));

    auto params = seq.parameters();
    std::cout << "Model has " << params.size() << " parameters\n";

    Adam optimizer(params, 0.001);

    for (int step = 0; step < 20; ++step) {
        optimizer.zero_grad();

        // Random input like integration tests
        auto inputs = randn({32, 784}, DType::Float32, Device::cpu());

        // Random integer targets manually
        auto targets = empty({32}, DType::Int64, Device::cpu());
        auto* target_data = targets.template data<int64_t>();
        for (int i = 0; i < 32; ++i) {
            target_data[i] = std::rand() % 10;
        }

        // Forward pass
        auto outputs = seq.forward(Variable(inputs, false));

        // Check outputs for NaN before loss
        auto* out_data = outputs.tensor().template data<float>();
        bool outputs_have_nan = false;
        for (int i = 0; i < 32 * 10; ++i) {
            if (std::isnan(out_data[i])) {
                outputs_have_nan = true;
                break;
            }
        }

        if (outputs_have_nan) {
            std::cout << "Step " << step << ": Outputs contain NaN before loss!\n";
            std::cout << "  First few outputs: [";
            for (int i = 0; i < 10; ++i) {
                std::cout << out_data[i];
                if (i < 9) std::cout << ", ";
            }
            std::cout << "]\n";

            // Check parameter values
            std::cout << "  First parameter value: " << params[0]->tensor().template data<float>()[0] << "\n";

            FAIL() << "NaN in outputs at step " << step;
        }

        // Cross entropy loss
        auto loss = cross_entropy(outputs, targets);

        float loss_val = loss.tensor().template data<float>()[0];

        std::cout << "Step " << step << ": loss = " << loss_val << "\n";

        if (std::isnan(loss_val)) {
            FAIL() << "NaN in loss at step " << step;
        }

        loss.backward();

        // Check gradients before optimizer step
        float max_grad = 0.0f;
        for (auto& param : params) {
            if (param->has_grad()) {
                auto& grad = param->grad().value();
                auto* grad_data = grad.template data<float>();
                int64_t numel = grad.numel();
                for (int64_t i = 0; i < numel; ++i) {
                    float abs_val = std::abs(grad_data[i]);
                    if (abs_val > max_grad) {
                        max_grad = abs_val;
                    }
                    if (std::isnan(grad_data[i]) || std::isinf(grad_data[i])) {
                        std::cout << "Step " << step << ": Gradient has NaN/Inf before update!\n";
                        std::cout << "  Gradient value: " << grad_data[i] << "\n";
                        FAIL() << "NaN/Inf in gradient before step " << step;
                    }
                }
            }
        }
        std::cout << "  Max gradient magnitude: " << max_grad << "\n";

        // Test without clipping first - if fix worked, no clipping needed
        // float clip_value = 1.0f;
        // ...clipping code commented out...

        optimizer.step();

        // Check parameters after step for inf/nan
        bool params_have_nan = false;
        for (auto& param : params) {
            auto* p_data = param->tensor().template data<float>();
            int64_t numel = param->tensor().numel();
            for (int64_t i = 0; i < numel; ++i) {
                if (std::isnan(p_data[i]) || std::isinf(p_data[i])) {
                    params_have_nan = true;
                    std::cout << "Step " << step << ": Parameter has NaN/Inf after update!\n";
                    std::cout << "  Value: " << p_data[i] << "\n";
                    break;
                }
            }
            if (params_have_nan) break;
        }

        if (params_have_nan) {
            FAIL() << "NaN/Inf in parameters after step " << step;
        }
    }

    std::cout << "Integration test setup passed!\n";
}
