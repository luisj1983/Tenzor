/**
 * @file test_grad_scaler.cpp
 * @brief Unit tests for GradScaler (AMP gradient scaling)
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/amp/grad_scaler.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn::amp;
using namespace tenzor::optim;

class GradScalerTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
    }

    Device device_;
};

// Test 1: Constructor with default parameters
TEST_F(GradScalerTest, DefaultConstructor) {
    GradScaler scaler;

    EXPECT_FLOAT_EQ(scaler.get_scale(), 65536.0f);
    EXPECT_EQ(scaler.get_growth_tracker(), 0);
    EXPECT_FALSE(scaler.found_inf_nan());
}

// Test 2: Constructor with custom parameters
TEST_F(GradScalerTest, CustomConstructor) {
    GradScaler scaler(1024.0f, 1.5f, 0.75f, 1000);

    EXPECT_FLOAT_EQ(scaler.get_scale(), 1024.0f);
    EXPECT_EQ(scaler.get_growth_tracker(), 0);
}

// Test 3: Constructor parameter validation
TEST_F(GradScalerTest, ConstructorValidation) {
    // Invalid init_scale
    EXPECT_THROW(GradScaler(-1.0f), std::invalid_argument);
    EXPECT_THROW(GradScaler(0.0f), std::invalid_argument);

    // Invalid growth_factor
    EXPECT_THROW(GradScaler(1024.0f, 0.5f), std::invalid_argument);
    EXPECT_THROW(GradScaler(1024.0f, 1.0f), std::invalid_argument);

    // Invalid backoff_factor
    EXPECT_THROW(GradScaler(1024.0f, 2.0f, 0.0f), std::invalid_argument);
    EXPECT_THROW(GradScaler(1024.0f, 2.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(GradScaler(1024.0f, 2.0f, 1.5f), std::invalid_argument);

    // Invalid growth_interval
    EXPECT_THROW(GradScaler(1024.0f, 2.0f, 0.5f, 0), std::invalid_argument);
    EXPECT_THROW(GradScaler(1024.0f, 2.0f, 0.5f, -100), std::invalid_argument);
}

// Test 4: Loss scaling
TEST_F(GradScalerTest, LossScaling) {
    GradScaler scaler(1000.0f);

    // Create a simple loss value
    auto loss_tensor = full({}, 0.5f, DType::Float32, device_);
    auto loss = Variable(loss_tensor, true);

    // Scale the loss
    auto scaled_loss = scaler.scale(loss);

    // Verify scaling
    EXPECT_FLOAT_EQ(scaled_loss.tensor().item<float>(), 500.0f);
}

// Test 5: Gradient unscaling
TEST_F(GradScalerTest, GradientUnscaling) {
    GradScaler scaler(100.0f);

    // Create a parameter with gradient
    auto param_tensor = ones({2, 3}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);

    // Create fake scaled gradient
    auto grad_tensor = full({2, 3}, 200.0f, DType::Float32, device_);
    param.grad() = grad_tensor;

    // Create optimizer with this parameter
    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    // Unscale gradients
    scaler.unscale_(optimizer);

    // Verify gradient was unscaled: 200.0 / 100.0 = 2.0
    const float* grad_data = param.grad()->data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 2.0f);
    }
}

// Test 6: Inf/NaN detection - normal gradients
TEST_F(GradScalerTest, NoInfNanDetection) {
    GradScaler scaler;

    // Create parameter with normal gradient
    auto param_tensor = ones({3, 3}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);

    auto grad_tensor = full({3, 3}, 1.5f, DType::Float32, device_);
    param.grad() = grad_tensor;

    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    // Step should succeed
    bool success = scaler.step(optimizer);

    EXPECT_TRUE(success);
    EXPECT_FALSE(scaler.found_inf_nan());
}

// Test 7: Inf detection
TEST_F(GradScalerTest, InfDetection) {
    GradScaler scaler;

    // Create parameter with inf in gradient
    auto param_tensor = ones({2, 2}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);

    auto grad_tensor = full({2, 2}, 1.0f, DType::Float32, device_);
    float* grad_data = grad_tensor.data<float>();
    grad_data[1] = std::numeric_limits<float>::infinity();
    param.grad() = grad_tensor;

    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    // Step should fail due to inf
    bool success = scaler.step(optimizer);

    EXPECT_FALSE(success);
    EXPECT_TRUE(scaler.found_inf_nan());
}

// Test 8: NaN detection
TEST_F(GradScalerTest, NanDetection) {
    GradScaler scaler;

    // Create parameter with nan in gradient
    auto param_tensor = ones({2, 2}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);

    auto grad_tensor = full({2, 2}, 1.0f, DType::Float32, device_);
    float* grad_data = grad_tensor.data<float>();
    grad_data[2] = std::numeric_limits<float>::quiet_NaN();
    param.grad() = grad_tensor;

    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    // Step should fail due to nan
    bool success = scaler.step(optimizer);

    EXPECT_FALSE(success);
    EXPECT_TRUE(scaler.found_inf_nan());
}

// Test 9: Scale backoff on overflow
TEST_F(GradScalerTest, ScaleBackoff) {
    GradScaler scaler(1000.0f, 2.0f, 0.5f, 10);

    float initial_scale = scaler.get_scale();

    // Create parameter with inf gradient
    auto param_tensor = ones({2, 2}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);

    auto grad_tensor = full({2, 2}, std::numeric_limits<float>::infinity(),
                                   DType::Float32, device_);
    param.grad() = grad_tensor;

    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    // Step will fail
    scaler.step(optimizer);

    // Update should decrease scale
    scaler.update();

    float new_scale = scaler.get_scale();
    EXPECT_FLOAT_EQ(new_scale, initial_scale * 0.5f);
    EXPECT_EQ(scaler.get_growth_tracker(), 0);
}

// Test 10: Scale growth after successful iterations
TEST_F(GradScalerTest, ScaleGrowth) {
    GradScaler scaler(1000.0f, 2.0f, 0.5f, 3);

    // Create parameter with normal gradient
    auto param_tensor = ones({2, 2}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);

    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    // Perform 3 successful iterations
    for (int i = 0; i < 3; ++i) {
        auto grad_tensor = full({2, 2}, 1.0f, DType::Float32, device_);
        param.grad() = grad_tensor;

        bool success = scaler.step(optimizer);
        EXPECT_TRUE(success);
        scaler.update();
    }

    // Scale should have grown
    EXPECT_FLOAT_EQ(scaler.get_scale(), 2000.0f);
    EXPECT_EQ(scaler.get_growth_tracker(), 0);
}

// Test 11: Integration with SGD optimizer
TEST_F(GradScalerTest, SGDIntegration) {
    GradScaler scaler(100.0f);

    // Create simple parameter
    auto param_tensor = full({3, 3}, 1.0f, DType::Float32, device_);
    auto param = Variable(param_tensor, true);

    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.1);

    // Create loss and compute scaled gradients
    auto loss_tensor = full({}, 0.5f, DType::Float32, device_);
    auto loss = Variable(loss_tensor, true);

    // Manually set gradient (simulating backward)
    auto grad_tensor = full({3, 3}, 10.0f, DType::Float32, device_);
    param.grad() = grad_tensor;

    // Step with scaler
    bool success = scaler.step(optimizer);
    scaler.update();

    EXPECT_TRUE(success);

    // Verify parameter was updated (unscaled gradient = 10/100 = 0.1)
    // Update: param = param - lr * grad = 1.0 - 0.1 * 0.1 = 0.99
    const float* param_data = param.tensor().data<float>();
    EXPECT_NEAR(param_data[0], 0.99f, 1e-5f);
}

// Test 12: Integration with Adam optimizer
TEST_F(GradScalerTest, AdamIntegration) {
    GradScaler scaler(100.0f);

    // Create parameter
    auto param_tensor = full({2, 2}, 1.0f, DType::Float32, device_);
    auto param = Variable(param_tensor, true);

    std::vector<Variable*> params = {&param};
    Adam optimizer(params, 0.01);

    // Set gradient
    auto grad_tensor = full({2, 2}, 50.0f, DType::Float32, device_);
    param.grad() = grad_tensor;

    // Step with scaler
    bool success = scaler.step(optimizer);
    scaler.update();

    EXPECT_TRUE(success);

    // Verify parameter was updated (unscaled gradient = 50/100 = 0.5)
    const float* param_data = param.tensor().data<float>();
    // Adam update is more complex, just verify it changed
    EXPECT_NE(param_data[0], 1.0f);
}

// Test 13: Reset functionality
TEST_F(GradScalerTest, Reset) {
    GradScaler scaler(2048.0f, 2.0f, 0.5f, 100);

    // Modify state
    auto param_tensor = ones({2, 2}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);
    auto grad_tensor = full({2, 2}, 1.0f, DType::Float32, device_);
    param.grad() = grad_tensor;

    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    // Perform some steps
    for (int i = 0; i < 5; ++i) {
        scaler.step(optimizer);
        scaler.update();
    }

    // Reset
    scaler.reset();

    // Verify reset to default initial state
    EXPECT_FLOAT_EQ(scaler.get_scale(), 65536.0f);
    EXPECT_EQ(scaler.get_growth_tracker(), 0);
    EXPECT_FALSE(scaler.found_inf_nan());
}

// Test 14: State dict save/load
TEST_F(GradScalerTest, StateDictSaveLoad) {
    GradScaler scaler1(2048.0f, 3.0f, 0.25f, 500);

    // Modify state
    auto param_tensor = ones({2, 2}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);
    auto grad_tensor = full({2, 2}, 1.0f, DType::Float32, device_);
    param.grad() = grad_tensor;

    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    // Perform steps
    for (int i = 0; i < 10; ++i) {
        scaler1.step(optimizer);
        scaler1.update();
    }

    // Save state
    auto state = scaler1.state_dict();

    // Create new scaler and load state
    GradScaler scaler2;
    scaler2.load_state_dict(state);

    // Verify state matches
    EXPECT_FLOAT_EQ(scaler2.get_scale(), scaler1.get_scale());
    EXPECT_EQ(scaler2.get_growth_tracker(), scaler1.get_growth_tracker());
}

// Test 15: Multiple parameters
TEST_F(GradScalerTest, MultipleParameters) {
    GradScaler scaler(100.0f);

    // Create multiple parameters
    auto param1_tensor = ones({2, 2}, DType::Float32, device_);
    auto param1 = Variable(param1_tensor, true);

    auto param2_tensor = ones({3, 3}, DType::Float32, device_);
    auto param2 = Variable(param2_tensor, true);

    // Set gradients
    auto grad1_tensor = full({2, 2}, 100.0f, DType::Float32, device_);
    param1.grad() = grad1_tensor;

    auto grad2_tensor = full({3, 3}, 200.0f, DType::Float32, device_);
    param2.grad() = grad2_tensor;

    std::vector<Variable*> params = {&param1, &param2};
    SGD optimizer(params, 0.1);

    // Step
    bool success = scaler.step(optimizer);

    EXPECT_TRUE(success);

    // Verify both gradients were unscaled
    const float* grad1_data = param1.grad()->data<float>();
    const float* grad2_data = param2.grad()->data<float>();

    EXPECT_FLOAT_EQ(grad1_data[0], 1.0f);  // 100 / 100 = 1.0
    EXPECT_FLOAT_EQ(grad2_data[0], 2.0f);  // 200 / 100 = 2.0
}

// Test 16: Training loop simulation
TEST_F(GradScalerTest, TrainingLoopSimulation) {
    GradScaler scaler(1024.0f, 2.0f, 0.5f, 5);

    auto param_tensor = ones({10, 10}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);

    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    int successful_steps = 0;
    int skipped_steps = 0;

    // Simulate 20 iterations
    for (int iter = 0; iter < 20; ++iter) {
        optimizer.zero_grad();

        // Simulate loss computation
        auto loss_tensor = full({}, 0.1f, DType::Float32, device_);
        auto loss = Variable(loss_tensor, true);
        auto scaled_loss = scaler.scale(loss);

        // Simulate gradients (with occasional overflow)
        float grad_value = (iter % 7 == 0) ?
            std::numeric_limits<float>::infinity() : 1.0f;

        auto grad_tensor = full({10, 10}, grad_value, DType::Float32, device_);
        param.grad() = grad_tensor;

        // Step and update
        bool success = scaler.step(optimizer);
        scaler.update();

        if (success) {
            successful_steps++;
        } else {
            skipped_steps++;
        }
    }

    // Verify some steps succeeded and some were skipped
    EXPECT_GT(successful_steps, 0);
    EXPECT_GT(skipped_steps, 0);
    EXPECT_EQ(successful_steps + skipped_steps, 20);
}

// Test 17: Scale limits
TEST_F(GradScalerTest, ScaleLimits) {
    // Test minimum scale limit
    GradScaler scaler_min(10.0f, 2.0f, 0.1f, 100);

    auto param_tensor = ones({2, 2}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);
    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    // Cause multiple overflows to reduce scale
    for (int i = 0; i < 10; ++i) {
        auto grad_tensor = full({2, 2}, std::numeric_limits<float>::infinity(),
                                       DType::Float32, device_);
        param.grad() = grad_tensor;
        scaler_min.step(optimizer);
        scaler_min.update();
    }

    // Scale should not go below 1.0
    EXPECT_GE(scaler_min.get_scale(), 1.0f);
}

// Test 18: Double unscale protection
TEST_F(GradScalerTest, DoubleUnscaleProtection) {
    GradScaler scaler(100.0f);

    auto param_tensor = ones({2, 2}, DType::Float32, device_);
    auto param = Variable(param_tensor, true);

    auto grad_tensor = full({2, 2}, 200.0f, DType::Float32, device_);
    param.grad() = grad_tensor;

    std::vector<Variable*> params = {&param};
    SGD optimizer(params, 0.01);

    // Unscale once
    scaler.unscale_(optimizer);
    float first_grad = param.grad()->data<float>()[0];

    // Unscale again (should be no-op)
    scaler.unscale_(optimizer);
    float second_grad = param.grad()->data<float>()[0];

    // Gradient should be same after second unscale
    EXPECT_FLOAT_EQ(first_grad, second_grad);
    EXPECT_FLOAT_EQ(first_grad, 2.0f);  // 200 / 100 = 2.0
}

int main(int argc, char** argv) {
    // Initialize Tenzor library
    tenzor::initialize();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
