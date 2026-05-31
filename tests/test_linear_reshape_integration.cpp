/**
 * @file test_linear_reshape_integration.cpp
 * @brief Integration test for Linear layer with autograd reshape
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <memory>
#include "grad_flow_helpers.hpp"
#include "backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;

class LinearReshapeIntegrationTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        set_grad_enabled(true);
    }
};

TEST_P(LinearReshapeIntegrationTest, LinearWithReshapeInput) {
    // Create a Linear layer
    auto linear = std::make_shared<Linear>(4, 3);
    linear->to(device);

    // Create input with shape {2, 4} and reshape to {8}
    auto data = ones({2, 4}, DType::Float32, device);
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
    EXPECT_GRAD_FLOWS(x);
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 4);

    // Check that layer parameters have gradients
    auto params = linear->parameters();
    ASSERT_GE(params.size(), 2);  // At least weight and bias
    for (auto& param : params) {
        ASSERT_TRUE(param->grad().has_value());
    }
}

TEST_P(LinearReshapeIntegrationTest, LinearWithPermuteInput) {
    // Create a Linear layer
    auto linear = std::make_shared<Linear>(4, 3);
    linear->to(device);

    // Create input with shape {2, 3, 4}
    auto data = ones({2, 3, 4}, DType::Float32, device);
    Variable x(data, true);

    // Permute dimensions
    auto x_permuted = permute(x, {0, 2, 1});  // {2, 4, 3}

    // Reshape for linear layer (batch=2*4=8, features=3)
    auto x_reshaped = reshape(x_permuted, {8, 3});

    // Note: Linear expects last dimension to be in_features, so we need 3 features
    auto linear2 = std::make_shared<Linear>(3, 5);
    linear2->to(device);
    auto output = linear2->forward(x_reshaped);

    // Check output shape
    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 5);

    // Backward pass
    auto loss = sum(output);
    loss.backward();

    // Check gradients flow back to original input
    EXPECT_GRAD_FLOWS(x);
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 4);
}

TEST_P(LinearReshapeIntegrationTest, MultipleReshapeOps) {
    // Test multiple reshape operations in the graph
    auto data = ones({6}, DType::Float32, device);
    Variable x(data, true);

    // Chain of reshape operations
    auto y1 = reshape(x, {2, 3});
    auto y2 = reshape(y1, {3, 2});
    auto y3 = reshape(y2, {6});

    // Pass through linear layer
    auto linear = std::make_shared<Linear>(6, 4);
    linear->to(device);
    auto output = linear->forward(y3);

    // Backward pass
    auto loss = sum(output);
    loss.backward();

    // Check gradient propagates back through all reshapes
    EXPECT_GRAD_FLOWS(x);
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 6);

    // Gradient should be non-zero
    auto grad_cpu = grad.cpu();
    auto grad_data = grad_cpu.data<float>();
    bool has_nonzero = false;
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        if (grad_data[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

INSTANTIATE_BACKEND_TESTS(LinearReshapeIntegrationTest);
