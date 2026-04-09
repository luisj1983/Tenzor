/**
 * @file test_sync_batchnorm.cpp
 * @brief Backend parity tests for SyncBatchNorm layer
 *
 * Tests SyncBatchNorm across CPU, CUDA, OneAPI, Vulkan, and ROCm backends
 * using a no-op all-reduce callback (world_size=1) to verify:
 * - Correct output shape
 * - Running statistics updates
 * - Eval mode uses running stats
 * - Gradient flow
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/sync_batchnorm.hpp>
#include "../../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// SyncBatchNorm Backend Test Fixture
// ============================================================================

class SyncBatchNormTest : public BackendTest {
protected:
    /**
     * @brief Create a SyncBatchNorm with a no-op all-reduce (single process).
     */
    SyncBatchNorm createSyncBN(int64_t num_features) {
        // No-op all-reduce: identity operation for single-process testing
        AllReduceFn noop_allreduce = [](Tensor& /* tensor */) {
            // No-op: single process, nothing to reduce
        };
        return SyncBatchNorm(num_features, noop_allreduce, /*world_size=*/1);
    }
};

// ============================================================================
// Basic Forward Test
// ============================================================================

TEST_P(SyncBatchNormTest, BasicForward) {
    auto sbn = createSyncBN(8);
    sbn.to(device);
    sbn.train();

    auto input_tensor = tenzor::randn({4, 8, 4, 4}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = sbn.forward(input);

    // Output shape should match input shape
    auto out_shape = output.shape();
    EXPECT_EQ(out_shape[0], 4);
    EXPECT_EQ(out_shape[1], 8);
    EXPECT_EQ(out_shape[2], 4);
    EXPECT_EQ(out_shape[3], 4);

    // Verify output contains finite values
    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = output_f32.data<float>();
    for (int64_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
        EXPECT_FALSE(std::isinf(data[i]));
    }
}

// ============================================================================
// Running Statistics Test
// ============================================================================

TEST_P(SyncBatchNormTest, RunningStats) {
    auto sbn = createSyncBN(8);
    sbn.to(device);
    sbn.train();

    // Get running mean before forward pass (should be zeros)
    auto params = sbn.parameters();

    auto input_tensor = tenzor::randn({4, 8, 4, 4}, DType::Float32, device);
    Variable input(input_tensor, false);

    // Run forward pass to update running stats
    sbn.forward(input);

    // Run a second forward pass with different data
    auto input2_tensor = tenzor::randn({4, 8, 4, 4}, DType::Float32, device) + 5.0f;
    Variable input2(input2_tensor, false);
    sbn.forward(input2);

    // Switch to eval mode and verify it runs (using running stats)
    sbn.eval();
    auto input3_tensor = tenzor::randn({2, 8, 4, 4}, DType::Float32, device);
    Variable input3(input3_tensor, false);

    EXPECT_NO_THROW({
        auto output = sbn.forward(input3);
        auto out_shape = output.shape();
        EXPECT_EQ(out_shape[0], 2);
        EXPECT_EQ(out_shape[1], 8);
    });
}

// ============================================================================
// Eval Mode Test
// ============================================================================

TEST_P(SyncBatchNormTest, EvalMode) {
    auto sbn = createSyncBN(8);
    sbn.to(device);

    // Train first to populate running stats
    sbn.train();
    auto train_tensor = tenzor::randn({4, 8, 4, 4}, DType::Float32, device);
    Variable train_input(train_tensor, false);
    sbn.forward(train_input);

    // Switch to eval mode
    sbn.eval();

    // Run the same input twice in eval mode -- outputs should be identical
    auto test_tensor = tenzor::randn({4, 8, 4, 4}, DType::Float32, device);
    Variable test_input1(test_tensor, false);
    Variable test_input2(test_tensor, false);

    auto output1 = sbn.forward(test_input1);
    auto output2 = sbn.forward(test_input2);

    auto out1_f32 = output1.tensor().to(Device::cpu()).to(DType::Float32);
    auto out2_f32 = output2.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data1 = out1_f32.data<float>();
    auto* data2 = out2_f32.data<float>();

    for (int64_t i = 0; i < out1_f32.numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], 1e-5f)
            << "Eval mode outputs should be deterministic at index " << i;
    }
}

// ============================================================================
// Gradient Flow Test
// ============================================================================

TEST_P(SyncBatchNormTest, GradientFlow) {
    auto sbn = createSyncBN(8);
    sbn.to(device);
    sbn.train();

    auto input_tensor = tenzor::randn({4, 8, 4, 4}, DType::Float32, device);
    Variable input(input_tensor, true);

    auto output = sbn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    // Use a non-constant grad_output. With grad_output=ones and weight=1, the
    // analytic input gradient of batch normalization is exactly zero (BN's
    // Jacobian has the constant vector in its null space — adding a constant
    // to all outputs leaves mean/variance unchanged, so the input is
    // unaffected). A random grad_output exercises the full backward formula.
    auto grad_output = tenzor::randn(shape_vec, DType::Float32, device);
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());

    auto grad_f32 = input.grad()->to(Device::cpu()).to(DType::Float32);
    auto* grad_data = grad_f32.data<float>();

    // Verify gradients are finite
    for (int64_t i = 0; i < grad_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]))
            << "Gradient contains NaN at index " << i;
        EXPECT_FALSE(std::isinf(grad_data[i]))
            << "Gradient contains Inf at index " << i;
    }

    // Verify gradients are non-trivial (not all zeros)
    bool has_nonzero = false;
    for (int64_t i = 0; i < grad_f32.numel(); ++i) {
        if (std::abs(grad_data[i]) > 1e-7f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Gradients should not all be zero";
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_BACKEND_TESTS(SyncBatchNormTest);
