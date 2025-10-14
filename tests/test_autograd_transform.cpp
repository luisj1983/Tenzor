/**
 * @file test_autograd_transform.cpp
 * @brief Test autograd-aware reshape and permute operations
 */

#include <gtest/gtest.h>
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <vector>

using namespace tenzor;

class AutogradTransformTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Enable gradient computation
        set_grad_enabled(true);
    }
};

TEST_F(AutogradTransformTest, ReshapeForwardPass) {
    // Create input variable with requires_grad=true
    auto data = ones({2, 3}, DType::Float32, Device::cpu());
    Variable x(data, true);

    // Reshape to different shape
    auto y = reshape(x, {3, 2});

    EXPECT_EQ(y.shape().size(), 2);
    EXPECT_EQ(y.shape()[0], 3);
    EXPECT_EQ(y.shape()[1], 2);
    EXPECT_TRUE(y.requires_grad());
    EXPECT_TRUE(y.grad_fn() != nullptr);
}

TEST_F(AutogradTransformTest, ReshapeBackwardPass) {
    // Create input variable
    auto data = ones({2, 3}, DType::Float32, Device::cpu());
    Variable x(data, true);

    // Reshape and compute sum
    auto y = reshape(x, {6});
    auto loss = sum(y);

    // Backward pass
    loss.backward();

    // Check gradient has correct shape
    ASSERT_TRUE(x.grad().has_value());
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape().size(), 2);
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);

    // All gradients should be 1.0 (sum backward)
    auto grad_data = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 1.0f);
    }
}

TEST_F(AutogradTransformTest, ReshapeChainedOperations) {
    // Create input
    auto data = full({2, 3}, 2.0f, DType::Float32, Device::cpu());
    Variable x(data, true);

    // Chain reshape with multiplication
    auto y = reshape(x, {3, 2});
    auto scalar = Variable(full({1}, 3.0f, DType::Float32, Device::cpu()), false);
    auto z = y * scalar;
    auto loss = sum(z);

    // Backward pass
    loss.backward();

    // Check gradient (d(sum(3*x))/dx = 3)
    ASSERT_TRUE(x.grad().has_value());
    auto grad = x.grad().value();
    auto grad_data = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 3.0f);
    }
}

TEST_F(AutogradTransformTest, PermuteForwardPass) {
    // Create input variable
    auto data = ones({2, 3, 4}, DType::Float32, Device::cpu());
    Variable x(data, true);

    // Permute dimensions
    auto y = permute(x, {2, 0, 1});

    EXPECT_EQ(y.shape().size(), 3);
    EXPECT_EQ(y.shape()[0], 4);
    EXPECT_EQ(y.shape()[1], 2);
    EXPECT_EQ(y.shape()[2], 3);
    EXPECT_TRUE(y.requires_grad());
    EXPECT_TRUE(y.grad_fn() != nullptr);
}

TEST_F(AutogradTransformTest, PermuteBackwardPass) {
    // Create input variable
    auto data = ones({2, 3, 4}, DType::Float32, Device::cpu());
    Variable x(data, true);

    // Permute and compute sum
    auto y = permute(x, {2, 0, 1});
    auto loss = sum(y);

    // Backward pass
    loss.backward();

    // Check gradient has correct shape (same as input)
    ASSERT_TRUE(x.grad().has_value());
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape().size(), 3);
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 4);

    // All gradients should be 1.0
    auto grad_data = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 1.0f);
    }
}

TEST_F(AutogradTransformTest, PermuteTranspose) {
    // Test that permute works like transpose
    auto data = full({3, 4}, 2.0f, DType::Float32, Device::cpu());
    Variable x(data, true);

    // Permute to transpose
    auto y = permute(x, {1, 0});
    auto loss = sum(y);

    // Backward pass
    loss.backward();

    // Check gradient shape matches input
    ASSERT_TRUE(x.grad().has_value());
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 3);
    EXPECT_EQ(grad.shape()[1], 4);
}

TEST_F(AutogradTransformTest, PermuteWithSum) {
    // Create input
    auto data = full({2, 3, 4}, 1.5f, DType::Float32, Device::cpu());
    Variable x(data, true);

    // Chain permute with sum (sum works on non-contiguous tensors)
    auto y = permute(x, {1, 2, 0});  // {3, 4, 2}
    auto loss = sum(y);

    // Backward pass
    loss.backward();

    // Check gradient (d(sum(x))/dx = 1)
    ASSERT_TRUE(x.grad().has_value());
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 4);

    auto grad_data = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 1.0f);
    }
}

TEST_F(AutogradTransformTest, ReshapeAndPermuteCombined) {
    // Test combining reshape and permute
    auto data = full({2, 3}, 1.0f, DType::Float32, Device::cpu());
    Variable x(data, true);

    // First reshape, then create a 3D tensor and permute
    auto y = reshape(x, {2, 3, 1});
    auto z = permute(y, {2, 0, 1});  // {1, 2, 3}
    auto loss = sum(z);

    // Backward pass
    loss.backward();

    // Check gradient shape matches original input
    ASSERT_TRUE(x.grad().has_value());
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);

    // All gradients should be 1.0
    auto grad_data = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 1.0f);
    }
}

TEST_F(AutogradTransformTest, ReshapeNoGrad) {
    // Test that reshape works without gradients
    auto data = ones({2, 3}, DType::Float32, Device::cpu());
    Variable x(data, false);  // requires_grad=false

    auto y = reshape(x, {6});

    EXPECT_EQ(y.shape()[0], 6);
    EXPECT_FALSE(y.requires_grad());
    EXPECT_TRUE(y.grad_fn() == nullptr);
}

TEST_F(AutogradTransformTest, PermuteNoGrad) {
    // Test that permute works without gradients
    auto data = ones({2, 3, 4}, DType::Float32, Device::cpu());
    Variable x(data, false);  // requires_grad=false

    auto y = permute(x, {2, 0, 1});

    EXPECT_EQ(y.shape()[0], 4);
    EXPECT_FALSE(y.requires_grad());
    EXPECT_TRUE(y.grad_fn() == nullptr);
}
