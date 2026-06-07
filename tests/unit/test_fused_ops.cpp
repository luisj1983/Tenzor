#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/fused_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::ops;

class FusedOpsTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // Helper to compare tensor shapes
    void assertShapesEqual(const Tensor& a, const Tensor& b, const std::string& msg = "Shape mismatch") {
        auto a_shape = a.shape();
        auto b_shape = b.shape();
        ASSERT_EQ(a_shape.size(), b_shape.size()) << msg << ": dimensionality mismatch";
        for (size_t i = 0; i < a_shape.size(); ++i) {
            ASSERT_EQ(a_shape[i], b_shape[i]) << msg << " at dimension " << i;
        }
    }

    // Helper to compare tensors within tolerance
    void assertTensorsClose(const Tensor& a, const Tensor& b, float rtol = 1e-4f, float atol = 1e-6f) {
        // Compare shapes element by element
        auto a_shape = a.shape();
        auto b_shape = b.shape();
        ASSERT_EQ(a_shape.size(), b_shape.size()) << "Shape dimensionality mismatch";
        for (size_t i = 0; i < a_shape.size(); ++i) {
            ASSERT_EQ(a_shape[i], b_shape[i]) << "Shape mismatch at dimension " << i;
        }
        ASSERT_EQ(a.dtype(), b.dtype()) << "DType mismatch";

        if (a.dtype() == DType::Float32) {
            auto a_cpu = a.cpu();
            auto b_cpu = b.cpu();
            const float* a_data = a_cpu.data<float>();
            const float* b_data = b_cpu.data<float>();
            for (int64_t i = 0; i < a_cpu.numel(); ++i) {
                float diff = std::abs(a_data[i] - b_data[i]);
                float threshold = atol + rtol * std::abs(b_data[i]);
                ASSERT_LE(diff, threshold)
                    << "Mismatch at index " << i
                    << ": got " << a_data[i]
                    << ", expected " << b_data[i];
            }
        }
    }
};

// ==============================================================================
// Fused Linear + ReLU Tests
// ==============================================================================

TEST_P(FusedOpsTest, FusedLinearReLU_ForwardCorrectness_2D) {
    auto input = randn({32, 128}, DType::Float32, device);
    auto weight = randn({64, 128}, DType::Float32, device);
    auto bias = randn({64}, DType::Float32, device);

    // Fused operation
    auto fused_output = fused_linear_relu(input, weight, &bias);

    // Unfused operations
    auto linear_out = add(matmul(input, weight.transpose(0, 1)), bias);
    auto linear_var = Variable(linear_out, false);  // Wrap in Variable for nn::relu
    auto unfused_var = nn::relu(linear_var);
    auto unfused_output = unfused_var.tensor();

    assertTensorsClose(fused_output, unfused_output);
}

TEST_P(FusedOpsTest, FusedLinearReLU_ForwardCorrectness_3D) {
    auto input = randn({8, 16, 256}, DType::Float32, device);
    auto weight = randn({128, 256}, DType::Float32, device);
    auto bias = randn({128}, DType::Float32, device);

    auto fused_output = fused_linear_relu(input, weight, &bias);

    // Verify shape
    ASSERT_EQ(fused_output.shape()[0], 8);
    ASSERT_EQ(fused_output.shape()[1], 16);
    ASSERT_EQ(fused_output.shape()[2], 128);

    // Verify no negative values (ReLU applied)
    auto out_cpu = fused_output.cpu();
    const float* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        ASSERT_GE(data[i], 0.0f) << "ReLU not applied at index " << i;
    }
}

TEST_P(FusedOpsTest, FusedLinearReLU_NoBias) {
    auto input = randn({16, 64}, DType::Float32, device);
    auto weight = randn({32, 64}, DType::Float32, device);

    auto fused_output = fused_linear_relu(input, weight, nullptr);

    // Unfused equivalent
    auto linear_out = matmul(input, weight.transpose(0, 1));
    auto linear_var = Variable(linear_out, false);
    auto unfused_var = nn::relu(linear_var);
    auto unfused_output = unfused_var.tensor();

    assertTensorsClose(fused_output, unfused_output);
}

TEST_P(FusedOpsTest, FusedLinearReLU_SingleBatch) {
    auto input = randn({1, 512}, DType::Float32, device);
    auto weight = randn({256, 512}, DType::Float32, device);
    auto bias = randn({256}, DType::Float32, device);

    auto output = fused_linear_relu(input, weight, &bias);

    ASSERT_EQ(output.shape()[0], 1);
    ASSERT_EQ(output.shape()[1], 256);
}

TEST_P(FusedOpsTest, FusedLinearReLU_LargeBatch) {
    auto input = randn({256, 1024}, DType::Float32, device);
    auto weight = randn({512, 1024}, DType::Float32, device);
    auto bias = randn({512}, DType::Float32, device);

    auto output = fused_linear_relu(input, weight, &bias);

    ASSERT_EQ(output.shape()[0], 256);
    ASSERT_EQ(output.shape()[1], 512);

    // Verify all non-negative
    auto out_cpu = output.cpu();
    const float* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        ASSERT_GE(data[i], 0.0f);
    }
}

// ==============================================================================
// Fused BatchNorm + ReLU Tests
// ==============================================================================

TEST_P(FusedOpsTest, FusedBatchNormReLU_ForwardCorrectness_2D) {
    auto input = randn({32, 64}, DType::Float32, device);
    auto mean = zeros({64}, DType::Float32, device);
    auto var = ones({64}, DType::Float32, device);
    auto gamma = ones({64}, DType::Float32, device);
    auto beta = zeros({64}, DType::Float32, device);

    auto fused_output = fused_batchnorm_relu(input, mean, var, gamma, beta);

    // Verify shape
    auto fused_shape = fused_output.shape();
    auto input_shape = input.shape();
    ASSERT_EQ(fused_shape.size(), input_shape.size());
    for (size_t i = 0; i < fused_shape.size(); ++i) {
        ASSERT_EQ(fused_shape[i], input_shape[i]);
    }

    // Verify all non-negative
    auto out_cpu = fused_output.cpu();
    const float* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        ASSERT_GE(data[i], 0.0f);
    }
}

TEST_P(FusedOpsTest, FusedBatchNormReLU_ForwardCorrectness_4D) {
    auto input = randn({8, 32, 16, 16}, DType::Float32, device);
    auto mean = zeros({32}, DType::Float32, device);
    auto var = ones({32}, DType::Float32, device);
    auto gamma = ones({32}, DType::Float32, device);
    auto beta = zeros({32}, DType::Float32, device);

    auto output = fused_batchnorm_relu(input, mean, var, gamma, beta);

    ASSERT_EQ(output.shape()[0], 8);
    ASSERT_EQ(output.shape()[1], 32);
    ASSERT_EQ(output.shape()[2], 16);
    ASSERT_EQ(output.shape()[3], 16);
}

TEST_P(FusedOpsTest, FusedBatchNormReLU_CustomEpsilon) {
    auto input = randn({16, 128}, DType::Float32, device);
    auto mean = zeros({128}, DType::Float32, device);
    auto var = ones({128}, DType::Float32, device);
    auto gamma = ones({128}, DType::Float32, device);
    auto beta = zeros({128}, DType::Float32, device);

    auto output = fused_batchnorm_relu(input, mean, var, gamma, beta, 1e-3f);

    assertShapesEqual(output, input);
}

// ==============================================================================
// Fused Softmax + CrossEntropy Tests
// ==============================================================================

// Root cause was CPU kernel ignoring reduction attribute (always mean).
// Fixed in fused_ops.cpp to respect AttrKey::Reduction for all modes.
TEST_P(FusedOpsTest, FusedSoftmaxCrossEntropy_MeanReduction) {
    auto logits = randn({32, 10}, DType::Float32, device);
    auto targets = randint(0, 10, {32}, DType::Int64, device);

    auto loss = fused_softmax_cross_entropy(logits, targets, "mean");

    // Loss should be a scalar
    ASSERT_EQ(loss.numel(), 1);

    // Loss should be non-negative
    auto loss_cpu = loss.cpu();
    float loss_val = loss_cpu.data<float>()[0];
    ASSERT_GE(loss_val, 0.0f);
}

TEST_P(FusedOpsTest, FusedSoftmaxCrossEntropy_SumReduction) {
    auto logits = randn({16, 5}, DType::Float32, device);
    auto targets = randint(0, 5, {16}, DType::Int64, device);

    auto loss = fused_softmax_cross_entropy(logits, targets, "sum");

    ASSERT_EQ(loss.numel(), 1);
    auto loss_cpu = loss.cpu();
    ASSERT_GE(loss_cpu.data<float>()[0], 0.0f);
}

TEST_P(FusedOpsTest, FusedSoftmaxCrossEntropy_NoReduction) {
    auto logits = randn({8, 100}, DType::Float32, device);
    auto targets = randint(0, 100, {8}, DType::Int64, device);

    auto losses = fused_softmax_cross_entropy(logits, targets, "none");

    // Should return per-sample losses
    ASSERT_EQ(losses.shape()[0], 8);

    // All losses should be non-negative
    auto losses_cpu = losses.cpu();
    const float* data = losses_cpu.data<float>();
    for (int64_t i = 0; i < losses_cpu.numel(); ++i) {
        ASSERT_GE(data[i], 0.0f);
    }
}

TEST_P(FusedOpsTest, FusedSoftmaxCrossEntropy_NumericalStability) {
    // Large logits to test numerical stability
    auto logits = randn({4, 3}, DType::Float32, device) * 100.0f;
    auto targets = randint(0, 3, {4}, DType::Int64, device);

    auto loss = fused_softmax_cross_entropy(logits, targets, "mean");

    // Should not produce NaN or Inf
    auto loss_cpu = loss.cpu();
    float loss_val = loss_cpu.data<float>()[0];
    ASSERT_FALSE(std::isnan(loss_val));
    ASSERT_FALSE(std::isinf(loss_val));
}

// ==============================================================================
// Fused Add + ReLU Tests
// ==============================================================================

TEST_P(FusedOpsTest, FusedAddReLU_ForwardCorrectness) {
    auto a = randn({32, 64}, DType::Float32, device);
    auto b = randn({32, 64}, DType::Float32, device);

    auto fused_output = fused_add_relu(a, b);
    auto added = add(a, b);
    auto added_var = Variable(added, false);
    auto unfused_var = nn::relu(added_var);
    auto unfused_output = unfused_var.tensor();

    assertTensorsClose(fused_output, unfused_output);
}

TEST_P(FusedOpsTest, FusedAddReLU_Broadcasting) {
    auto a = randn({32, 64, 16}, DType::Float32, device);
    auto b = randn({64, 1}, DType::Float32, device);

    auto output = fused_add_relu(a, b);

    assertShapesEqual(output, a);

    // Verify non-negative
    auto out_cpu = output.cpu();
    const float* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        ASSERT_GE(data[i], 0.0f);
    }
}

TEST_P(FusedOpsTest, FusedAddReLU_ResidualConnection) {
    // Simulates residual connection: x + f(x)
    auto x = randn({16, 128}, DType::Float32, device);
    auto residual = randn({16, 128}, DType::Float32, device);

    auto output = fused_add_relu(x, residual);

    assertShapesEqual(output, x);
}

// ==============================================================================
// Fused GELU Tests
// ==============================================================================

TEST_P(FusedOpsTest, FusedGELU_ForwardCorrectness) {
    auto input = randn({32, 512}, DType::Float32, device);
    auto output = fused_gelu(input);

    assertShapesEqual(output, input);

    // GELU should produce smooth activations (not just 0/positive)
    auto in_cpu = input.cpu();
    auto out_cpu = output.cpu();
    const float* in_data = in_cpu.data<float>();
    const float* out_data = out_cpu.data<float>();

    // Check that positive inputs produce positive outputs
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        if (in_data[i] > 2.0f) {
            ASSERT_GT(out_data[i], 0.0f);
        }
    }
}

TEST_P(FusedOpsTest, FusedGELU_ZeroInput) {
    auto input = zeros({16, 64}, DType::Float32, device);
    auto output = fused_gelu(input);

    // GELU(0) = 0
    auto out_cpu = output.cpu();
    const float* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        ASSERT_NEAR(data[i], 0.0f, 1e-6f);
    }
}

TEST_P(FusedOpsTest, FusedGELU_LargeInputs) {
    auto input = randn({8, 256}, DType::Float32, device) * 10.0f;
    auto output = fused_gelu(input);

    // Should not produce NaN or Inf
    auto out_cpu = output.cpu();
    const float* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        ASSERT_FALSE(std::isnan(data[i]));
        ASSERT_FALSE(std::isinf(data[i]));
    }
}

// ==============================================================================
// Fused Layer Norm Tests
// ==============================================================================

TEST_P(FusedOpsTest, FusedLayerNorm_ForwardCorrectness_2D) {
    auto input = randn({32, 512}, DType::Float32, device);
    auto weight = ones({512}, DType::Float32, device);
    auto bias = zeros({512}, DType::Float32, device);

    auto output = fused_layer_norm(input, {512}, weight, bias);

    assertShapesEqual(output, input);

    // Verify normalization: mean ~= 0, std ~= 1 for each sample
    auto out_cpu = output.cpu();
    const float* out_data = out_cpu.data<float>();
    for (int64_t i = 0; i < 32; ++i) {
        const float* row = out_data + i * 512;

        // Compute mean
        float mean = 0.0f;
        for (int64_t j = 0; j < 512; ++j) {
            mean += row[j];
        }
        mean /= 512;

        ASSERT_NEAR(mean, 0.0f, 1e-4f) << "Mean should be near 0 for sample " << i;

        // Compute variance
        float var = 0.0f;
        for (int64_t j = 0; j < 512; ++j) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= 512;

        ASSERT_NEAR(var, 1.0f, 1e-3f) << "Variance should be near 1 for sample " << i;
    }
}

TEST_P(FusedOpsTest, FusedLayerNorm_MultiDimensional) {
    auto input = randn({8, 16, 256}, DType::Float32, device);
    auto weight = ones({256}, DType::Float32, device);
    auto bias = zeros({256}, DType::Float32, device);

    auto output = fused_layer_norm(input, {256}, weight, bias);

    assertShapesEqual(output, input);
}

TEST_P(FusedOpsTest, FusedLayerNorm_CustomWeightBias) {
    auto input = randn({16, 128}, DType::Float32, device);
    auto weight = randn({128}, DType::Float32, device);
    auto bias = randn({128}, DType::Float32, device);

    auto output = fused_layer_norm(input, {128}, weight, bias);

    assertShapesEqual(output, input);
}

TEST_P(FusedOpsTest, FusedLayerNorm_SmallEpsilon) {
    auto input = randn({4, 64}, DType::Float32, device);
    auto weight = ones({64}, DType::Float32, device);
    auto bias = zeros({64}, DType::Float32, device);

    auto output = fused_layer_norm(input, {64}, weight, bias, 1e-8f);

    assertShapesEqual(output, input);
}

// ==============================================================================
// Edge Case Tests
// ==============================================================================

TEST_P(FusedOpsTest, EdgeCase_EmptyTensor) {
    // Matches PyTorch semantics: empty batch is a valid input for Linear and
    // produces an empty output of shape (0, out_features) rather than raising.
    // The previous assertion expected a throw — that was aspirational, not how
    // the library actually behaves, and trains that use dynamic-batch padding
    // depend on this being a no-op.
    auto input = zeros({0, 64}, DType::Float32, device);
    auto weight = randn({32, 64}, DType::Float32, device);

    auto output = fused_linear_relu(input, weight, nullptr);

    ASSERT_EQ(output.shape().size(), 2u);
    EXPECT_EQ(output.shape()[0], 0);
    EXPECT_EQ(output.shape()[1], 32);
    EXPECT_EQ(output.numel(), 0);
}

TEST_P(FusedOpsTest, EdgeCase_SingleElement) {
    auto input = randn({1, 1}, DType::Float32, device);
    auto weight = randn({1, 1}, DType::Float32, device);
    auto bias = randn({1}, DType::Float32, device);

    auto output = fused_linear_relu(input, weight, &bias);

    ASSERT_EQ(output.numel(), 1);
    auto out_cpu = output.cpu();
    ASSERT_GE(out_cpu.data<float>()[0], 0.0f);
}

TEST_P(FusedOpsTest, EdgeCase_LargeFeatureDimension) {
    auto input = randn({2, 4096}, DType::Float32, device);
    auto weight = randn({2048, 4096}, DType::Float32, device);
    auto bias = randn({2048}, DType::Float32, device);

    auto output = fused_linear_relu(input, weight, &bias);

    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 2048);
}

TEST_P(FusedOpsTest, EdgeCase_InvalidTargetIndex) {
    auto logits = randn({4, 10}, DType::Float32, device);
    auto targets_cpu = zeros({4}, DType::Int64);
    targets_cpu.data<int64_t>()[0] = 15;  // Out of range
    auto targets = targets_cpu.to(device);

    EXPECT_THROW({
        fused_softmax_cross_entropy(logits, targets, "mean");
    }, std::runtime_error);
}

// ==============================================================================
// Performance Baseline Tests (compare fused vs unfused)
// ==============================================================================

TEST_P(FusedOpsTest, Performance_LinearReLU) {
    // This test documents expected behavior; actual performance testing
    // would require timing infrastructure
    auto input = randn({1024, 2048}, DType::Float32, device);
    auto weight = randn({1024, 2048}, DType::Float32, device);
    auto bias = randn({1024}, DType::Float32, device);

    // Fused version
    auto fused_output = fused_linear_relu(input, weight, &bias);

    // Unfused version
    auto linear_out = add(matmul(input, weight.transpose(0, 1)), bias);
    auto linear_var = Variable(linear_out, false);
    auto unfused_var = nn::relu(linear_var);
    auto unfused_output = unfused_var.tensor();

    // Fused (single GEMM with fused bias+ReLU) and unfused (separate matmul +
    // add + ReLU) accumulate the 2048-deep dot products in a different order, so
    // outputs differ by Float32 non-associativity — most visible on near-zero
    // ReLU-boundary values produced by catastrophic cancellation of ~45-magnitude
    // partial sums. atol=1e-5 sits below that noise floor (oneAPI's GEMM diverges
    // from its unfused path by ~1.5e-5 there). Use a realistic tolerance, matching
    // the rationale in test_fused_ops_multidtype.cpp::getFusedOpTolerances; a real
    // correctness bug would diverge by O(0.1)+, far above this.
    assertTensorsClose(fused_output, unfused_output, 1e-2f, 1e-3f);
}

TEST_P(FusedOpsTest, Performance_SoftmaxCrossEntropy) {
    auto logits = randn({512, 1000}, DType::Float32, device);
    auto targets = randint(0, 1000, {512}, DType::Int64, device);

    auto fused_loss = fused_softmax_cross_entropy(logits, targets, "mean");

    // Verify result is valid
    ASSERT_EQ(fused_loss.numel(), 1);
    auto loss_cpu = fused_loss.cpu();
    ASSERT_GE(loss_cpu.data<float>()[0], 0.0f);
}

INSTANTIATE_BACKEND_TESTS(FusedOpsTest);
