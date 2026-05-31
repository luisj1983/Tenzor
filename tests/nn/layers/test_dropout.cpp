#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../grad_flow_helpers.hpp"
#include "../../backend_test_fixture.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::nn;

class DropoutTest : public ::tenzor::testing::BackendTest {};
class Dropout2dTest : public ::tenzor::testing::BackendTest {};

// ============================================================================
// Test 1: Training Mode vs Inference Mode
// ============================================================================

TEST_P(DropoutTest, InferenceModeNoModification) {
    Dropout dropout(0.5);
    dropout.eval();  // Set to inference mode
    dropout.to(device);

    auto input_tensor = ones({2, 3, 4}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // In inference mode, output should equal input exactly
    ASSERT_EQ(output.tensor().shape().size(), 3);
    ASSERT_EQ(output.tensor().shape()[0], 2);
    ASSERT_EQ(output.tensor().shape()[1], 3);
    ASSERT_EQ(output.tensor().shape()[2], 4);

    auto out_cpu = output.tensor().cpu();
    auto* out_data = out_cpu.data<float>();
    for (size_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(out_data[i], 1.0f);
    }
}

TEST_P(DropoutTest, TrainingModeModifiesOutput) {
    Dropout dropout(0.5);
    dropout.train();  // Set to training mode
    dropout.to(device);

    auto input_tensor = ones({100, 100}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // In training mode, some values should be zeroed
    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();
    size_t zero_count = 0;
    size_t non_zero_count = 0;

    for (size_t i = 0; i < out_cpu.numel(); ++i) {
        if (data[i] == 0.0f) {
            zero_count++;
        } else {
            non_zero_count++;
        }
    }

    // Should have both zeros and non-zeros
    EXPECT_GT(zero_count, 0);
    EXPECT_GT(non_zero_count, 0);
}

// ============================================================================
// Test 2: Different Dropout Probabilities
// ============================================================================

TEST_P(DropoutTest, ProbabilityZeroNoDropout) {
    Dropout dropout(0.0);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({50, 50}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // With p=0.0, all values should remain 1.0
    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();
    for (size_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_P(DropoutTest, ProbabilityHalf) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({200, 200}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Count non-zero values
    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();
    size_t non_zero_count = 0;

    for (size_t i = 0; i < out_cpu.numel(); ++i) {
        if (data[i] != 0.0f) {
            non_zero_count++;
        }
    }

    // With p=0.5, expect approximately 50% dropout (50% kept)
    double keep_rate = static_cast<double>(non_zero_count) / out_cpu.numel();
    EXPECT_NEAR(keep_rate, 0.5, 0.05);  // 5% tolerance for large sample
}

TEST_P(DropoutTest, ProbabilityNinetyPercent) {
    Dropout dropout(0.9);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({200, 200}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Count non-zero values
    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();
    size_t non_zero_count = 0;

    for (size_t i = 0; i < out_cpu.numel(); ++i) {
        if (data[i] != 0.0f) {
            non_zero_count++;
        }
    }

    // With p=0.9, expect approximately 10% kept
    double keep_rate = static_cast<double>(non_zero_count) / out_cpu.numel();
    EXPECT_NEAR(keep_rate, 0.1, 0.05);  // 5% tolerance
}

TEST_P(DropoutTest, InvalidProbabilityNegative) {
    EXPECT_THROW(Dropout(-0.1), std::invalid_argument);
}

TEST_P(DropoutTest, InvalidProbabilityOne) {
    EXPECT_THROW(Dropout(1.0), std::invalid_argument);
}

TEST_P(DropoutTest, InvalidProbabilityGreaterThanOne) {
    EXPECT_THROW(Dropout(1.5), std::invalid_argument);
}

// ============================================================================
// Test 3: Inverted Dropout Scaling Verification
// ============================================================================

TEST_P(DropoutTest, InvertedDropoutScaling) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({100, 100}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Check that non-zero values are scaled by 1/(1-p) = 2.0
    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();
    double sum_non_zero = 0.0;
    size_t non_zero_count = 0;

    for (size_t i = 0; i < out_cpu.numel(); ++i) {
        if (data[i] != 0.0f) {
            sum_non_zero += data[i];
            non_zero_count++;
        }
    }

    double avg_non_zero = sum_non_zero / non_zero_count;
    double expected_scale = 1.0 / (1.0 - 0.5);  // 2.0
    EXPECT_NEAR(avg_non_zero, expected_scale, 0.1);
}

TEST_P(DropoutTest, ScalingWithDifferentProbability) {
    Dropout dropout(0.8);
    dropout.train();
    dropout.to(device);

    auto input_tensor = full({200, 200}, 3.0f, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Non-zero values should be scaled by 1/(1-0.8) = 5.0
    // Original value 3.0 * 5.0 = 15.0
    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();
    double sum_non_zero = 0.0;
    size_t non_zero_count = 0;

    for (size_t i = 0; i < out_cpu.numel(); ++i) {
        if (data[i] != 0.0f) {
            sum_non_zero += data[i];
            non_zero_count++;
        }
    }

    if (non_zero_count > 0) {
        double avg_non_zero = sum_non_zero / non_zero_count;
        EXPECT_NEAR(avg_non_zero, 15.0, 1.0);
    }
}

// ============================================================================
// Test 4: Statistical Distribution (Bernoulli)
// ============================================================================

TEST_P(DropoutTest, BernoulliDistribution) {
    Dropout dropout(0.3);
    dropout.train();
    dropout.to(device);

    // Large sample for statistical test
    auto input_tensor = ones({500, 500}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();
    size_t kept_count = 0;
    size_t total = out_cpu.numel();

    for (size_t i = 0; i < total; ++i) {
        if (data[i] != 0.0f) {
            kept_count++;
        }
    }

    // With p=0.3, we keep 70% of values
    double keep_rate = static_cast<double>(kept_count) / total;
    double expected_keep_rate = 1.0 - 0.3;

    // Chi-square test would be more rigorous, but simple proportion test suffices
    EXPECT_NEAR(keep_rate, expected_keep_rate, 0.03);
}

// ============================================================================
// Test 5: Device placement
// ============================================================================

TEST_P(DropoutTest, DevicePlacement) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({50, 50}, DType::Float32, device);
    Variable input(input_tensor, false);

    EXPECT_NO_THROW({
        auto output = dropout.forward(input);
        EXPECT_EQ(output.tensor().device().type, device.type);
    });
}

// ============================================================================
// Test 6: Gradient Checking for Backward Pass
// ============================================================================

TEST_P(DropoutTest, BackwardPassGradientShape) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = randn({10, 20}, DType::Float32, device);
    Variable input(input_tensor, true);  // requires_grad = true

    auto output = dropout.forward(input);

    // Check gradient function is set
    EXPECT_TRUE(output.grad_fn() != nullptr);

    // Backward pass
    auto grad_output = ones({10, 20}, DType::Float32, device);
    output.backward(grad_output);
    EXPECT_GRAD_FLOWS(input);
}

TEST_P(DropoutTest, BackwardPassGradientValues) {
    Dropout dropout(0.0);  // No dropout for deterministic test
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({5, 5}, DType::Float32, device);
    Variable input(input_tensor, true);

    auto output = dropout.forward(input);

    // With p=0, gradient should pass through unchanged
    auto grad_output = full({5, 5}, 2.0f, DType::Float32, device);
    output.backward(grad_output);

    auto input_grad = input.grad();
    ASSERT_TRUE(input_grad.has_value());

    auto grad_cpu = input_grad->cpu();
    auto* grad_data = grad_cpu.data<float>();
    for (size_t i = 0; i < grad_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 2.0f);
    }
}

// ============================================================================
// Test 7: Different Tensor Shapes
// ============================================================================

TEST_P(DropoutTest, OneDimensionalTensor) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({1000}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    EXPECT_EQ(output.tensor().shape().size(), 1);
    EXPECT_EQ(output.tensor().shape()[0], 1000);
}

TEST_P(DropoutTest, TwoDimensionalTensor) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({32, 64}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    EXPECT_EQ(output.tensor().shape().size(), 2);
    EXPECT_EQ(output.tensor().shape()[0], 32);
    EXPECT_EQ(output.tensor().shape()[1], 64);
}

TEST_P(DropoutTest, FourDimensionalTensor) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({8, 16, 32, 32}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    EXPECT_EQ(output.tensor().shape().size(), 4);
    EXPECT_EQ(output.tensor().shape()[0], 8);
    EXPECT_EQ(output.tensor().shape()[1], 16);
    EXPECT_EQ(output.tensor().shape()[2], 32);
    EXPECT_EQ(output.tensor().shape()[3], 32);
}

// ============================================================================
// Test 8: Edge Cases
// ============================================================================

TEST_P(DropoutTest, EdgeCaseEmptyTensor) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({0, 10}, DType::Float32, device);
    Variable input(input_tensor, false);

    EXPECT_NO_THROW({
        auto output = dropout.forward(input);
        EXPECT_EQ(output.tensor().numel(), 0);
    });
}

TEST_P(DropoutTest, EdgeCaseSingleElement) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({1}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();
    // Should be either 0 or 2.0 (scaled)
    EXPECT_TRUE(data[0] == 0.0f || std::abs(data[0] - 2.0f) < 0.01f);
}

// ============================================================================
// Test 9: Reproducibility with Fixed Random Seed
// ============================================================================

TEST_P(DropoutTest, ReproducibilityInInferenceMode) {
    Dropout dropout(0.5);
    dropout.eval();
    dropout.to(device);

    auto input_tensor = randn({20, 20}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output1 = dropout.forward(input);
    auto output2 = dropout.forward(input);

    // In eval mode, should be identical
    auto out1_cpu = output1.tensor().cpu();
    auto out2_cpu = output2.tensor().cpu();
    auto* data1 = out1_cpu.data<float>();
    auto* data2 = out2_cpu.data<float>();

    for (size_t i = 0; i < out1_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(data1[i], data2[i]);
    }
}

// ============================================================================
// Test 10: Dropout2d Tests
// ============================================================================

TEST_P(Dropout2dTest, InferenceMode) {
    Dropout2d dropout(0.5);
    dropout.eval();
    dropout.to(device);

    auto input_tensor = ones({2, 4, 8, 8}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    ASSERT_EQ(output.tensor().shape().size(), 4);
    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();
    for (size_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_P(Dropout2dTest, ChannelWiseDropout) {
    Dropout2d dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({2, 10, 8, 8}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Verify entire channels are uniformly dropped or kept
    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();

    for (size_t n = 0; n < 2; ++n) {
        for (size_t c = 0; c < 10; ++c) {
            size_t channel_offset = (n * 10 + c) * 8 * 8;
            float first_value = data[channel_offset];

            // All pixels in channel should have same value
            for (size_t i = 0; i < 8 * 8; ++i) {
                EXPECT_FLOAT_EQ(data[channel_offset + i], first_value);
            }
        }
    }
}

TEST_P(Dropout2dTest, InvalidDimensions) {
    Dropout2d dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_1d = ones({10}, DType::Float32, device);
    Variable var_1d(input_1d, false);

    EXPECT_THROW(dropout.forward(var_1d), std::invalid_argument);
}

TEST_P(Dropout2dTest, ThreeDimensionalInput) {
    Dropout2d dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({4, 8, 8}, DType::Float32, device);
    Variable input(input_tensor, false);

    EXPECT_NO_THROW({
        auto output = dropout.forward(input);
        EXPECT_EQ(output.tensor().shape().size(), 3);
    });
}

// ============================================================================
// Additional Quality Tests
// ============================================================================

TEST_P(DropoutTest, PreservesExpectedValueInTraining) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    // Test that expected value is preserved due to scaling
    auto input_tensor = full({1000, 1000}, 10.0f, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    auto out_cpu = output.tensor().cpu();
    auto* data = out_cpu.data<float>();
    double sum = 0.0;
    for (size_t i = 0; i < out_cpu.numel(); ++i) {
        sum += data[i];
    }

    double mean = sum / out_cpu.numel();
    EXPECT_NEAR(mean, 10.0, 0.5);  // Expected value should be preserved
}

TEST_P(DropoutTest, DifferentInputValues) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = randn({50, 50}, DType::Float32, device);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Just verify it runs without errors
    EXPECT_EQ(output.tensor().shape().size(), 2);
    EXPECT_EQ(output.tensor().shape()[0], 50);
    EXPECT_EQ(output.tensor().shape()[1], 50);
}

TEST_P(DropoutTest, ConsecutiveForwardPasses) {
    Dropout dropout(0.5);
    dropout.train();
    dropout.to(device);

    auto input_tensor = ones({10, 10}, DType::Float32, device);
    Variable input(input_tensor, false);

    // Multiple forward passes should produce different masks
    auto output1 = dropout.forward(input);
    auto output2 = dropout.forward(input);

    auto out1_cpu = output1.tensor().cpu();
    auto out2_cpu = output2.tensor().cpu();
    auto* data1 = out1_cpu.data<float>();
    auto* data2 = out2_cpu.data<float>();

    // At least some values should differ (with high probability)
    size_t diff_count = 0;
    for (size_t i = 0; i < out1_cpu.numel(); ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-6) {
            diff_count++;
        }
    }

    // With p=0.5, expect many differences
    EXPECT_GT(diff_count, 0);
}

INSTANTIATE_BACKEND_TESTS(DropoutTest);
INSTANTIATE_BACKEND_TESTS(Dropout2dTest);
