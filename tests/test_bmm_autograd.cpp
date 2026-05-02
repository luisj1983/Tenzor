/**
 * @file test_bmm_autograd.cpp
 * @brief Test autograd-aware batch matrix multiplication
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "backend_parity/parity_test_utils.hpp"
#include "grad_flow_helpers.hpp"
#include <cmath>

using namespace tenzor;

// Namespace alias for autograd operations
namespace autograd = tenzor;

// Global test environment to initialize Tenzor once
class TenzorEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        initialize();
    }
};

class BmmAutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Enable gradient computation
        set_grad_enabled(true);
    }
};

TEST_F(BmmAutogradTest, ForwardPass) {
    // Create two 3D tensors for batch matrix multiplication
    // a: (2, 3, 4), b: (2, 4, 5) -> result: (2, 3, 5)
    auto a_tensor = ones({2, 3, 4}, DType::Float32, Device::cpu());
    auto b_tensor = ones({2, 4, 5}, DType::Float32, Device::cpu());

    Variable a(a_tensor, true);
    Variable b(b_tensor, true);

    // Perform bmm
    auto c = autograd::bmm(a, b);

    // Check shape
    ASSERT_EQ(c.shape().size(), 3);
    EXPECT_EQ(c.shape()[0], 2);  // batch
    EXPECT_EQ(c.shape()[1], 3);  // n
    EXPECT_EQ(c.shape()[2], 5);  // p

    // Check that gradient function is set
    EXPECT_TRUE(c.grad_fn() != nullptr);

    // Check values (ones @ ones with middle dim=4 should give all 4s)
    auto* c_data = c.tensor().data<float>();
    for (int64_t i = 0; i < c.tensor().numel(); ++i) {
        EXPECT_NEAR(c_data[i], 4.0f, 1e-5f);
    }
}

TEST_F(BmmAutogradTest, BackwardGradientA) {
    // Create simple test case
    // a: (1, 2, 3), b: (1, 3, 2) -> c: (1, 2, 2)
    auto a_tensor = ones({1, 2, 3}, DType::Float32, Device::cpu());
    auto b_tensor = ones({1, 3, 2}, DType::Float32, Device::cpu());

    // Fill with specific values
    auto* a_data = a_tensor.data<float>();
    for (int i = 0; i < 6; ++i) {
        a_data[i] = static_cast<float>(i + 1);  // [1, 2, 3, 4, 5, 6]
    }

    Variable a(a_tensor, true);
    Variable b(b_tensor, true);

    // Forward pass
    auto c = autograd::bmm(a, b);

    // Sum for scalar loss
    auto loss = autograd::sum(c);

    // Backward pass
    loss.backward();

    // Check that gradients are computed
    EXPECT_TRUE(a.has_grad());
    EXPECT_TRUE(b.has_grad());

    // Verify gradient shapes
    EXPECT_EQ(a.grad()->shape().size(), 3);
    EXPECT_EQ(a.grad()->shape()[0], 1);
    EXPECT_EQ(a.grad()->shape()[1], 2);
    EXPECT_EQ(a.grad()->shape()[2], 3);

    // grad_a = grad_c @ b^T
    // grad_c is all ones after sum backward
    // b is all ones with shape (1, 3, 2)
    // b^T has shape (1, 2, 3)
    // grad_a = (1, 2, 2) @ (1, 2, 3) = (1, 2, 3)
    // Each element should be sum along rows of b^T = 2 (since b has 2 columns)
    auto* grad_a_data = a.grad()->data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(grad_a_data[i], 2.0f, 1e-5f);
    }
}

TEST_F(BmmAutogradTest, BackwardGradientB) {
    // Test gradient computation for b
    auto a_tensor = ones({1, 2, 3}, DType::Float32, Device::cpu());
    auto b_tensor = ones({1, 3, 2}, DType::Float32, Device::cpu());

    Variable a(a_tensor, true);
    Variable b(b_tensor, true);

    // Forward pass
    auto c = autograd::bmm(a, b);

    // Sum for scalar loss
    auto loss = autograd::sum(c);

    // Backward pass
    loss.backward();

    // Verify gradient shape for b
    EXPECT_EQ(b.grad()->shape().size(), 3);
    EXPECT_EQ(b.grad()->shape()[0], 1);
    EXPECT_EQ(b.grad()->shape()[1], 3);
    EXPECT_EQ(b.grad()->shape()[2], 2);

    // grad_b = a^T @ grad_c
    // a has shape (1, 2, 3), a^T has shape (1, 3, 2)
    // grad_c is all ones with shape (1, 2, 2)
    // grad_b = (1, 3, 2) @ (1, 2, 2) = (1, 3, 2)
    // Each element should be sum along rows of grad_c = 2
    auto* grad_b_data = b.grad()->data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(grad_b_data[i], 2.0f, 1e-5f);
    }
}

TEST_F(BmmAutogradTest, LargerBatchSize) {
    // Test with larger batch size
    // a: (4, 5, 6), b: (4, 6, 7) -> c: (4, 5, 7)
    auto a_tensor = ones({4, 5, 6}, DType::Float32, Device::cpu());
    auto b_tensor = ones({4, 6, 7}, DType::Float32, Device::cpu());

    Variable a(a_tensor, true);
    Variable b(b_tensor, true);

    auto c = autograd::bmm(a, b);

    // Check shape
    EXPECT_EQ(c.shape()[0], 4);
    EXPECT_EQ(c.shape()[1], 5);
    EXPECT_EQ(c.shape()[2], 7);

    // Sum and backward
    auto loss = autograd::sum(c);
    loss.backward();

    // Verify gradients exist and have correct shapes
    EXPECT_TRUE(a.has_grad());
    EXPECT_TRUE(b.has_grad());
    EXPECT_EQ(a.grad()->shape()[0], 4);
    EXPECT_EQ(a.grad()->shape()[1], 5);
    EXPECT_EQ(a.grad()->shape()[2], 6);
    EXPECT_EQ(b.grad()->shape()[0], 4);
    EXPECT_EQ(b.grad()->shape()[1], 6);
    EXPECT_EQ(b.grad()->shape()[2], 7);
}

TEST_F(BmmAutogradTest, NoGradWhenDisabled) {
    // Test that gradients are not computed when grad is disabled
    auto a_tensor = ones({2, 3, 4}, DType::Float32, Device::cpu());
    auto b_tensor = ones({2, 4, 5}, DType::Float32, Device::cpu());

    Variable a(a_tensor, false);  // No grad
    Variable b(b_tensor, false);  // No grad

    auto c = autograd::bmm(a, b);

    // grad_fn should be null since no gradients required
    EXPECT_FALSE(c.requires_grad());
    EXPECT_TRUE(c.grad_fn() == nullptr);
}

TEST_F(BmmAutogradTest, OneInputRequiresGrad) {
    // Test when only one input requires gradient
    auto a_tensor = ones({1, 2, 3}, DType::Float32, Device::cpu());
    auto b_tensor = ones({1, 3, 2}, DType::Float32, Device::cpu());

    Variable a(a_tensor, true);   // Requires grad
    Variable b(b_tensor, false);  // No grad

    auto c = autograd::bmm(a, b);

    // Output should require grad
    EXPECT_TRUE(c.requires_grad());
    EXPECT_TRUE(c.grad_fn() != nullptr);

    // Backward pass
    auto loss = autograd::sum(c);
    loss.backward();

    // Only a should have gradient
    EXPECT_TRUE(a.has_grad());
    EXPECT_FALSE(b.has_grad());
}

#ifdef TENZOR_USE_CUDA
TEST_F(BmmAutogradTest, CUDABackward) {
    // Skip when CUDA is not available on this host. SKIP_IF_NO_CUDA goes
    // through the canonical has_cuda() probe, which honors
    // TENZOR_SKIP_BACKENDS — TESTING.md bans inline cuda_available() checks
    // because they bypass that env var.
    SKIP_IF_NO_CUDA;

    auto cuda_device = Device::cuda(0);

    auto a_tensor = ones({2, 3, 4}, DType::Float32, cuda_device);
    auto b_tensor = ones({2, 4, 5}, DType::Float32, cuda_device);

    Variable a(a_tensor, true);
    Variable b(b_tensor, true);

    auto c = autograd::bmm(a, b);
    auto loss = autograd::sum(c);

    loss.backward();

    EXPECT_GRAD_FLOWS(a);
    EXPECT_GRAD_FLOWS(b);
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new TenzorEnvironment);
    return RUN_ALL_TESTS();
}
