#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::optim;

//==============================================================================
// SGD Optimizer Tests
//==============================================================================

class OptimizerTestSGDBasicStep : public BackendTest {};

TEST_P(OptimizerTestSGDBasicStep, BasicStep) {
    // Create a parameter with gradient
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    param->set_grad(ones({2, 2}, DType::Float32, device));

    // Create SGD optimizer with lr=0.1
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    // Take one step
    optimizer.step();

    // After step: param = 1 - 0.1 * 1 = 0.9
    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i], 0.9f) << "Mismatch at index " << i << " - Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestSGDBasicStep);

class OptimizerTestSGDMultipleSteps : public BackendTest {};

TEST_P(OptimizerTestSGDMultipleSteps, MultipleSteps) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    // Take multiple steps with constant gradient
    for (int step = 0; step < 5; step++) {
        param->set_grad(ones({2, 2}, DType::Float32, device));
        optimizer.step();
    }

    // After 5 steps: param = 1 - 5 * 0.1 * 1 = 0.5
    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.5f, 1e-6f) << "Mismatch at index " << i << " - Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestSGDMultipleSteps);

class OptimizerTestSGDWithMomentum : public BackendTest {};

TEST_P(OptimizerTestSGDWithMomentum, WithMomentum) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    param->set_grad(ones({2, 2}, DType::Float32, device));

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1, 0.9);  // lr=0.1, momentum=0.9

    // First step: velocity = 0 * 0.9 + 1 = 1, param = 1 - 0.1 * 1 = 0.9
    optimizer.step();

    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.9f, 1e-6f) << "First step mismatch at index " << i << " - Failed on " << device.to_string();
    }

    // Second step: velocity = 1 * 0.9 + 1 = 1.9, param = 0.9 - 0.1 * 1.9 = 0.71
    param->set_grad(ones({2, 2}, DType::Float32, device));
    optimizer.step();

    cpu_tensor = param->tensor().to(Device::cpu());
    data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.71f, 1e-5f) << "Second step mismatch at index " << i << " - Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestSGDWithMomentum);

class OptimizerTestSGDWithWeightDecay : public BackendTest {};

TEST_P(OptimizerTestSGDWithWeightDecay, WithWeightDecay) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    param->set_grad(zeros({2, 2}, DType::Float32, device));  // Zero gradient

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1, 0.0, 0.0, 0.01);  // weight_decay=0.01

    // Step: effective_grad = 0 + 1 * 0.01 = 0.01, param = 1 - 0.1 * 0.01 = 0.999
    optimizer.step();

    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.999f, 1e-6f) << "Mismatch at index " << i << " - Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestSGDWithWeightDecay);

class OptimizerTestSGDMultipleParameters : public BackendTest {};

TEST_P(OptimizerTestSGDMultipleParameters, MultipleParameters) {
    auto param1 = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    auto param2 = std::make_shared<Variable>(full({2, 2}, 2.0f, DType::Float32, device), true);

    param1->set_grad(ones({2, 2}, DType::Float32, device));
    param2->set_grad(full({2, 2}, 0.5f, DType::Float32, device));

    auto params = std::vector<std::shared_ptr<Variable>>{param1, param2};
    auto optimizer = SGD(params, 0.1);

    optimizer.step();

    // param1 = 1 - 0.1 * 1 = 0.9
    auto cpu_tensor1 = param1->tensor().to(Device::cpu());
    auto data1 = cpu_tensor1.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data1[i], 0.9f) << "Failed on " << device.to_string();
    }

    // param2 = 2 - 0.1 * 0.5 = 1.95
    auto cpu_tensor2 = param2->tensor().to(Device::cpu());
    auto data2 = cpu_tensor2.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data2[i], 1.95f) << "Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestSGDMultipleParameters);

class OptimizerTestSGDZeroGrad : public BackendTest {};

TEST_P(OptimizerTestSGDZeroGrad, ZeroGrad) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    param->set_grad(ones({2, 2}, DType::Float32, device));

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    // Zero gradients
    optimizer.zero_grad();

    // Check gradient is zero
    ASSERT_TRUE(param->has_grad()) << "Failed on " << device.to_string();
    auto grad_cpu = param->grad().value().to(Device::cpu());
    auto grad_data = grad_cpu.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(grad_data[i], 0.0f) << "Gradient not zeroed at index " << i << " - Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestSGDZeroGrad);

class OptimizerTestSGDNoGradient : public BackendTest {};

TEST_P(OptimizerTestSGDNoGradient, NoGradient) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    // Don't set gradient

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    optimizer.step();

    // Parameter should not change
    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f) << "Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestSGDNoGradient);

//==============================================================================
// Adam Optimizer Tests
//==============================================================================

class OptimizerTestAdamBasicStep : public BackendTest {};

TEST_P(OptimizerTestAdamBasicStep, BasicStep) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    param->set_grad(ones({2, 2}, DType::Float32, device));

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    optimizer.step();

    // After first step, parameter should have decreased
    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_LT(data[i], 1.0f) << "Parameter not updated at index " << i << " - Failed on " << device.to_string();
        EXPECT_GT(data[i], 0.99f) << "Parameter update too large at index " << i << " - Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestAdamBasicStep);

class OptimizerTestAdamBiasCorrection : public BackendTest {};

TEST_P(OptimizerTestAdamBiasCorrection, BiasCorrection) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    param->set_grad(ones({2, 2}, DType::Float32, device));

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001, 0.9, 0.999, 1e-8);

    optimizer.step();

    // First step with bias correction
    // m = 0.9 * 0 + 0.1 * 1 = 0.1
    // v = 0.999 * 0 + 0.001 * 1 = 0.001
    // m_hat = 0.1 / (1 - 0.9) = 1.0
    // v_hat = 0.001 / (1 - 0.999) = 1.0
    // update = 0.001 * 1.0 / (sqrt(1.0) + 1e-8) ≈ 0.001
    // param = 1 - 0.001 ≈ 0.999

    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.999f, 1e-4f) << "Mismatch at index " << i << " - Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestAdamBiasCorrection);

class OptimizerTestAdamMultipleSteps : public BackendTest {};

TEST_P(OptimizerTestAdamMultipleSteps, MultipleSteps) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01);  // Higher learning rate for visible effect

    float prev_value = 1.0f;
    for (int step = 0; step < 10; step++) {
        param->set_grad(ones({2, 2}, DType::Float32, device));
        optimizer.step();

        auto cpu_tensor = param->tensor().to(Device::cpu());
        auto data = cpu_tensor.data<float>();
        EXPECT_LT(data[0], prev_value) << "Parameter not decreasing at step " << step << " - Failed on " << device.to_string();
        prev_value = data[0];
    }

    // After 10 steps, parameter should be noticeably smaller
    EXPECT_LT(prev_value, 0.95f) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestAdamMultipleSteps);

class OptimizerTestAdamWithWeightDecay : public BackendTest {};

TEST_P(OptimizerTestAdamWithWeightDecay, WithWeightDecay) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    param->set_grad(zeros({2, 2}, DType::Float32, device));

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01, 0.9, 0.999, 1e-8, 0.01);  // weight_decay=0.01

    optimizer.step();

    // Even with zero gradient, weight decay should reduce parameters
    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_LT(data[i], 1.0f) << "Weight decay not applied at index " << i << " - Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestAdamWithWeightDecay);

class OptimizerTestAdamConvergenceTest : public BackendTest {};

TEST_P(OptimizerTestAdamConvergenceTest, ConvergenceTest) {
    // Test that Adam can optimize towards a target
    // Target: make parameter close to zero
    auto param = std::make_shared<Variable>(ones({4}, DType::Float32, device), true);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.1);

    // Simulate gradient descent towards zero
    for (int step = 0; step < 100; step++) {
        // Gradient points in direction of current value
        param->set_grad(param->tensor());
        optimizer.step();
    }

    // Parameter should be close to zero
    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.0f, 0.01f) << "Failed to converge at index " << i << " - Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestAdamConvergenceTest);

class OptimizerTestAdamMultipleParameters : public BackendTest {};

TEST_P(OptimizerTestAdamMultipleParameters, MultipleParameters) {
    auto param1 = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    auto param2 = std::make_shared<Variable>(full({2, 2}, 2.0f, DType::Float32, device), true);

    param1->set_grad(ones({2, 2}, DType::Float32, device));
    param2->set_grad(full({2, 2}, 2.0f, DType::Float32, device));

    auto params = std::vector<std::shared_ptr<Variable>>{param1, param2};
    auto optimizer = Adam(params, 0.01);

    optimizer.step();

    // Both parameters should decrease
    auto cpu_tensor1 = param1->tensor().to(Device::cpu());
    auto cpu_tensor2 = param2->tensor().to(Device::cpu());
    auto data1 = cpu_tensor1.data<float>();
    auto data2 = cpu_tensor2.data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_LT(data1[i], 1.0f) << "Failed on " << device.to_string();
        EXPECT_LT(data2[i], 2.0f) << "Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestAdamMultipleParameters);

class OptimizerTestAdamZeroGrad : public BackendTest {};

TEST_P(OptimizerTestAdamZeroGrad, ZeroGrad) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    param->set_grad(ones({2, 2}, DType::Float32, device));

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    optimizer.zero_grad();

    ASSERT_TRUE(param->has_grad()) << "Failed on " << device.to_string();
    auto grad_cpu = param->grad().value().to(Device::cpu());
    auto grad_data = grad_cpu.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(grad_data[i], 0.0f) << "Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestAdamZeroGrad);

class OptimizerTestAdamNoGradient : public BackendTest {};

TEST_P(OptimizerTestAdamNoGradient, NoGradient) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    optimizer.step();

    // Parameter should not change without gradient
    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f) << "Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestAdamNoGradient);

//==============================================================================
// Learning Rate Management Tests
//==============================================================================

class OptimizerTestSGDLearningRateChange : public BackendTest {};

TEST_P(OptimizerTestSGDLearningRateChange, LearningRateChange) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    param->set_grad(ones({2, 2}, DType::Float32, device));

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.1) << "Failed on " << device.to_string();

    // Change learning rate
    optimizer.set_lr(0.01);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01) << "Failed on " << device.to_string();

    optimizer.step();

    // param = 1 - 0.01 * 1 = 0.99
    auto cpu_tensor = param->tensor().to(Device::cpu());
    auto data = cpu_tensor.data<float>();
    EXPECT_NEAR(data[0], 0.99f, 1e-6f) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestSGDLearningRateChange);

class OptimizerTestAdamLearningRateChange : public BackendTest {};

TEST_P(OptimizerTestAdamLearningRateChange, LearningRateChange) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32, device), true);
    param->set_grad(ones({2, 2}, DType::Float32, device));

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.001) << "Failed on " << device.to_string();

    optimizer.set_lr(0.01);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestAdamLearningRateChange);

//==============================================================================
// Integration Tests with Real Gradients
//==============================================================================

class OptimizerTestSGDSimpleLinearRegression : public BackendTest {};

TEST_P(OptimizerTestSGDSimpleLinearRegression, SimpleLinearRegression) {
    // y = 2x + 3, learn to fit this
    auto weight = std::make_shared<Variable>(zeros({1}, DType::Float32, device), true);
    auto bias = std::make_shared<Variable>(zeros({1}, DType::Float32, device), true);

    auto params = std::vector<std::shared_ptr<Variable>>{weight, bias};
    auto optimizer = SGD(params, 0.01);

    // Training data
    float x_train[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y_train[] = {5.0f, 7.0f, 9.0f, 11.0f};  // 2x + 3

    // Train for several iterations
    for (int epoch = 0; epoch < 50; epoch++) {
        optimizer.zero_grad();

        float total_loss = 0.0f;
        for (int i = 0; i < 4; i++) {
            // Forward pass: y_pred = w * x + b
            auto weight_cpu = weight->tensor().to(Device::cpu());
            auto bias_cpu = bias->tensor().to(Device::cpu());
            float y_pred = weight_cpu.data<float>()[0] * x_train[i] +
                          bias_cpu.data<float>()[0];

            // Loss: MSE
            float loss = (y_pred - y_train[i]) * (y_pred - y_train[i]);
            total_loss += loss;

            // Manual gradient computation (for simplicity)
            float grad_w = 2 * (y_pred - y_train[i]) * x_train[i];
            float grad_b = 2 * (y_pred - y_train[i]);

            if (i == 0) {
                weight->set_grad(full({1}, 0.0f, DType::Float32, device));
                bias->set_grad(full({1}, 0.0f, DType::Float32, device));
            }

            auto weight_grad_cpu = weight->grad().value().to(Device::cpu());
            auto bias_grad_cpu = bias->grad().value().to(Device::cpu());
            weight_grad_cpu.data<float>()[0] += grad_w;
            bias_grad_cpu.data<float>()[0] += grad_b;

            // Copy back to device
            weight->set_grad(weight_grad_cpu.to(device));
            bias->set_grad(bias_grad_cpu.to(device));
        }

        optimizer.step();
    }

    // Check learned parameters are close to target (w=2, b=3)
    // Note: With 50 epochs and lr=0.01, convergence may not be perfect
    auto weight_cpu = weight->tensor().to(Device::cpu());
    auto bias_cpu = bias->tensor().to(Device::cpu());
    float learned_w = weight_cpu.data<float>()[0];
    float learned_b = bias_cpu.data<float>()[0];

    EXPECT_NEAR(learned_w, 2.0f, 0.5f) << "Failed on " << device.to_string();
    EXPECT_NEAR(learned_b, 3.0f, 1.5f) << "Failed on " << device.to_string();  // Relaxed tolerance for bias convergence
}

INSTANTIATE_BACKEND_TESTS(OptimizerTestSGDSimpleLinearRegression);
