/**
 * @file test_linear_reshape_integration.cpp
 * @brief Integration test for Linear layer with autograd reshape
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;

// Initialize library before all tests
class GlobalEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Register global environment
static ::testing::Environment* const global_env =
    ::testing::AddGlobalTestEnvironment(new GlobalEnvironment);

class LinearReshapeIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        set_grad_enabled(true);
    }
};

TEST_F(LinearReshapeIntegrationTest, LinearWithReshapeInput) {
    // Create a Linear layer
    auto linear = std::make_shared<Linear>(4, 3);

    // Create input with shape {2, 4} and reshape to {8}
    auto data = ones({2, 4}, DType::Float32, Device::cpu());
    Variable x(data, true);

    // Reshape before passing to linear
    auto x_flat = reshape(x, {8});

    // Reshape back to {2, 4} for linear layer
    auto x_reshaped = reshape(x_flat, {2, 4});

    // Pass through linear layer
    auto output = linear->forward(x_reshaped);

    // Check output shape
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);

    // Compute loss and backward
    auto loss = sum(output);
    loss.backward();

    // Check that input gradient is accumulated correctly
    ASSERT_TRUE(x.grad().has_value());
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 4);

    // Check that layer parameters have gradients
    auto params = linear->parameters();
    ASSERT_GE(params.size(), 2);  // At least weight and bias
    for (auto* param : params) {
        ASSERT_TRUE(param->grad().has_value());
    }
}

TEST_F(LinearReshapeIntegrationTest, LinearWithPermuteInput) {
    // Create a Linear layer
    auto linear = std::make_shared<Linear>(4, 3);

    // Create input with shape {2, 3, 4}
    auto data = ones({2, 3, 4}, DType::Float32, Device::cpu());
    Variable x(data, true);

    // Permute dimensions
    auto x_permuted = permute(x, {0, 2, 1});  // {2, 4, 3}

    // Reshape for linear layer (batch=2*4=8, features=3)
    auto x_reshaped = reshape(x_permuted, {8, 3});

    // Note: Linear expects last dimension to be in_features, so we need 3 features
    auto linear2 = std::make_shared<Linear>(3, 5);
    auto output = linear2->forward(x_reshaped);

    // Check output shape
    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 5);

    // Backward pass
    auto loss = sum(output);
    loss.backward();

    // Check gradients flow back to original input
    ASSERT_TRUE(x.grad().has_value());
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 4);
}

TEST_F(LinearReshapeIntegrationTest, MultipleReshapeOps) {
    // Test multiple reshape operations in the graph
    auto data = ones({6}, DType::Float32, Device::cpu());
    Variable x(data, true);

    // Chain of reshape operations
    auto y1 = reshape(x, {2, 3});
    auto y2 = reshape(y1, {3, 2});
    auto y3 = reshape(y2, {6});

    // Pass through linear layer
    auto linear = std::make_shared<Linear>(6, 4);
    auto output = linear->forward(y3);

    // Backward pass
    auto loss = sum(output);
    loss.backward();

    // Check gradient propagates back through all reshapes
    ASSERT_TRUE(x.grad().has_value());
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 6);

    // Gradient should be non-zero
    auto grad_data = grad.data<float>();
    bool has_nonzero = false;
    for (int64_t i = 0; i < grad.numel(); ++i) {
        if (grad_data[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}
