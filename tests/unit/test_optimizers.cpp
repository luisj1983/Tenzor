#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::optim;

// Global test environment
class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

//==============================================================================
// SGD Optimizer Tests
//==============================================================================

TEST(OptimizerTest, SGD_BasicStep) {
    // Create a parameter with gradient
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    param->grad() = ones({2, 2}, DType::Float32);

    // Create SGD optimizer with lr=0.1
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    // Take one step
    optimizer.step();

    // After step: param = 1 - 0.1 * 1 = 0.9
    auto data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i], 0.9f) << "Mismatch at index " << i;
    }
}

TEST(OptimizerTest, SGD_MultipleSteps) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    // Take multiple steps with constant gradient
    for (int step = 0; step < 5; step++) {
        param->grad() = ones({2, 2}, DType::Float32);
        optimizer.step();
    }

    // After 5 steps: param = 1 - 5 * 0.1 * 1 = 0.5
    auto data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.5f, 1e-6f) << "Mismatch at index " << i;
    }
}

TEST(OptimizerTest, SGD_WithMomentum) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    param->grad() = ones({2, 2}, DType::Float32);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1, 0.9);  // lr=0.1, momentum=0.9

    // First step: velocity = 0 * 0.9 + 1 = 1, param = 1 - 0.1 * 1 = 0.9
    optimizer.step();

    auto data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.9f, 1e-6f) << "First step mismatch at index " << i;
    }

    // Second step: velocity = 1 * 0.9 + 1 = 1.9, param = 0.9 - 0.1 * 1.9 = 0.71
    param->grad() = ones({2, 2}, DType::Float32);
    optimizer.step();

    data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.71f, 1e-5f) << "Second step mismatch at index " << i;
    }
}

TEST(OptimizerTest, SGD_WithWeightDecay) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    param->grad() = zeros({2, 2}, DType::Float32);  // Zero gradient

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1, 0.0, 0.0, 0.01);  // weight_decay=0.01

    // Step: effective_grad = 0 + 1 * 0.01 = 0.01, param = 1 - 0.1 * 0.01 = 0.999
    optimizer.step();

    auto data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.999f, 1e-6f) << "Mismatch at index " << i;
    }
}

TEST(OptimizerTest, SGD_MultipleParameters) {
    auto param1 = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto param2 = std::make_shared<Variable>(full({2, 2}, 2.0f, DType::Float32), true);

    param1->grad() = ones({2, 2}, DType::Float32);
    param2->grad() = full({2, 2}, 0.5f, DType::Float32);

    auto params = std::vector<std::shared_ptr<Variable>>{param1, param2};
    auto optimizer = SGD(params, 0.1);

    optimizer.step();

    // param1 = 1 - 0.1 * 1 = 0.9
    auto data1 = param1->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data1[i], 0.9f);
    }

    // param2 = 2 - 0.1 * 0.5 = 1.95
    auto data2 = param2->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data2[i], 1.95f);
    }
}

TEST(OptimizerTest, SGD_ZeroGrad) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    param->grad() = ones({2, 2}, DType::Float32);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    // Zero gradients
    optimizer.zero_grad();

    // Check gradient is zero
    ASSERT_TRUE(param->has_grad());
    auto grad_data = param->grad().value().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(grad_data[i], 0.0f) << "Gradient not zeroed at index " << i;
    }
}

TEST(OptimizerTest, SGD_NoGradient) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    // Don't set gradient

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    optimizer.step();

    // Parameter should not change
    auto data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

//==============================================================================
// Adam Optimizer Tests
//==============================================================================

TEST(OptimizerTest, Adam_BasicStep) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    param->grad() = ones({2, 2}, DType::Float32);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    optimizer.step();

    // After first step, parameter should have decreased
    auto data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_LT(data[i], 1.0f) << "Parameter not updated at index " << i;
        EXPECT_GT(data[i], 0.99f) << "Parameter update too large at index " << i;
    }
}

TEST(OptimizerTest, Adam_BiasCorrection) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    param->grad() = ones({2, 2}, DType::Float32);

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

    auto data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.999f, 1e-4f) << "Mismatch at index " << i;
    }
}

TEST(OptimizerTest, Adam_MultipleSteps) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01);  // Higher learning rate for visible effect

    float prev_value = 1.0f;
    for (int step = 0; step < 10; step++) {
        param->grad() = ones({2, 2}, DType::Float32);
        optimizer.step();

        auto data = param->tensor().data<float>();
        EXPECT_LT(data[0], prev_value) << "Parameter not decreasing at step " << step;
        prev_value = data[0];
    }

    // After 10 steps, parameter should be noticeably smaller
    EXPECT_LT(prev_value, 0.95f);
}

TEST(OptimizerTest, Adam_WithWeightDecay) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    param->grad() = zeros({2, 2}, DType::Float32);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.01, 0.9, 0.999, 1e-8, 0.01);  // weight_decay=0.01

    optimizer.step();

    // Even with zero gradient, weight decay should reduce parameters
    auto data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_LT(data[i], 1.0f) << "Weight decay not applied at index " << i;
    }
}

TEST(OptimizerTest, Adam_ConvergenceTest) {
    // Test that Adam can optimize towards a target
    // Target: make parameter close to zero
    auto param = std::make_shared<Variable>(ones({4}, DType::Float32), true);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.1);

    // Simulate gradient descent towards zero
    for (int step = 0; step < 100; step++) {
        // Gradient points in direction of current value
        param->grad() = param->tensor();
        optimizer.step();
    }

    // Parameter should be close to zero
    auto data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.0f, 0.01f) << "Failed to converge at index " << i;
    }
}

TEST(OptimizerTest, Adam_MultipleParameters) {
    auto param1 = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    auto param2 = std::make_shared<Variable>(full({2, 2}, 2.0f, DType::Float32), true);

    param1->grad() = ones({2, 2}, DType::Float32);
    param2->grad() = full({2, 2}, 2.0f, DType::Float32);

    auto params = std::vector<std::shared_ptr<Variable>>{param1, param2};
    auto optimizer = Adam(params, 0.01);

    optimizer.step();

    // Both parameters should decrease
    auto data1 = param1->tensor().data<float>();
    auto data2 = param2->tensor().data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_LT(data1[i], 1.0f);
        EXPECT_LT(data2[i], 2.0f);
    }
}

TEST(OptimizerTest, Adam_ZeroGrad) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    param->grad() = ones({2, 2}, DType::Float32);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    optimizer.zero_grad();

    ASSERT_TRUE(param->has_grad());
    auto grad_data = param->grad().value().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(grad_data[i], 0.0f);
    }
}

TEST(OptimizerTest, Adam_NoGradient) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    optimizer.step();

    // Parameter should not change without gradient
    auto data = param->tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

//==============================================================================
// Learning Rate Management Tests
//==============================================================================

TEST(OptimizerTest, SGD_LearningRateChange) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    param->grad() = ones({2, 2}, DType::Float32);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = SGD(params, 0.1);

    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.1);

    // Change learning rate
    optimizer.set_lr(0.01);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01);

    optimizer.step();

    // param = 1 - 0.01 * 1 = 0.99
    auto data = param->tensor().data<float>();
    EXPECT_NEAR(data[0], 0.99f, 1e-6f);
}

TEST(OptimizerTest, Adam_LearningRateChange) {
    auto param = std::make_shared<Variable>(ones({2, 2}, DType::Float32), true);
    param->grad() = ones({2, 2}, DType::Float32);

    auto params = std::vector<std::shared_ptr<Variable>>{param};
    auto optimizer = Adam(params, 0.001);

    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.001);

    optimizer.set_lr(0.01);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01);
}

//==============================================================================
// Integration Tests with Real Gradients
//==============================================================================

TEST(OptimizerTest, SGD_SimpleLinearRegression) {
    // y = 2x + 3, learn to fit this
    auto weight = std::make_shared<Variable>(zeros({1}, DType::Float32), true);
    auto bias = std::make_shared<Variable>(zeros({1}, DType::Float32), true);

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
            float y_pred = weight->tensor().data<float>()[0] * x_train[i] +
                          bias->tensor().data<float>()[0];

            // Loss: MSE
            float loss = (y_pred - y_train[i]) * (y_pred - y_train[i]);
            total_loss += loss;

            // Manual gradient computation (for simplicity)
            float grad_w = 2 * (y_pred - y_train[i]) * x_train[i];
            float grad_b = 2 * (y_pred - y_train[i]);

            if (i == 0) {
                weight->grad() = full({1}, 0.0f, DType::Float32);
                bias->grad() = full({1}, 0.0f, DType::Float32);
            }

            weight->grad().value().data<float>()[0] += grad_w;
            bias->grad().value().data<float>()[0] += grad_b;
        }

        optimizer.step();
    }

    // Check learned parameters are close to target (w=2, b=3)
    // Note: With 50 epochs and lr=0.01, convergence may not be perfect
    float learned_w = weight->tensor().data<float>()[0];
    float learned_b = bias->tensor().data<float>()[0];

    EXPECT_NEAR(learned_w, 2.0f, 0.5f);
    EXPECT_NEAR(learned_b, 3.0f, 1.5f);  // Relaxed tolerance for bias convergence
}
