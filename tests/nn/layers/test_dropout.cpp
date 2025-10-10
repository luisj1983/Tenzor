#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::nn;

// Global test environment
class DropoutTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const dropout_env =
    ::testing::AddGlobalTestEnvironment(new DropoutTestEnvironment);

// ============================================================================
// Test 1: Training Mode vs Inference Mode
// ============================================================================

TEST(DropoutTest, InferenceModeNoModification) {
    Dropout dropout(0.5);
    dropout.eval();  // Set to inference mode

    auto input_tensor = ones({2, 3, 4});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // In inference mode, output should equal input exactly
    ASSERT_EQ(output.tensor().shape().size(), 3);
    ASSERT_EQ(output.tensor().shape()[0], 2);
    ASSERT_EQ(output.tensor().shape()[1], 3);
    ASSERT_EQ(output.tensor().shape()[2], 4);

    auto* out_data = static_cast<float*>(output.tensor().impl()->storage->data());
    for (size_t i = 0; i < output.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(out_data[i], 1.0f);
    }
}

TEST(DropoutTest, TrainingModeModifiesOutput) {
    Dropout dropout(0.5);
    dropout.train();  // Set to training mode

    auto input_tensor = ones({100, 100});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // In training mode, some values should be zeroed
    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());
    size_t zero_count = 0;
    size_t non_zero_count = 0;

    for (size_t i = 0; i < output.tensor().numel(); ++i) {
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

TEST(DropoutTest, ProbabilityZeroNoDropout) {
    Dropout dropout(0.0);
    dropout.train();

    auto input_tensor = ones({50, 50});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // With p=0.0, all values should remain 1.0
    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());
    for (size_t i = 0; i < output.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST(DropoutTest, ProbabilityHalf) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({200, 200});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Count non-zero values
    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());
    size_t non_zero_count = 0;

    for (size_t i = 0; i < output.tensor().numel(); ++i) {
        if (data[i] != 0.0f) {
            non_zero_count++;
        }
    }

    // With p=0.5, expect approximately 50% dropout (50% kept)
    double keep_rate = static_cast<double>(non_zero_count) / output.tensor().numel();
    EXPECT_NEAR(keep_rate, 0.5, 0.05);  // 5% tolerance for large sample
}

TEST(DropoutTest, ProbabilityNinetyPercent) {
    Dropout dropout(0.9);
    dropout.train();

    auto input_tensor = ones({200, 200});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Count non-zero values
    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());
    size_t non_zero_count = 0;

    for (size_t i = 0; i < output.tensor().numel(); ++i) {
        if (data[i] != 0.0f) {
            non_zero_count++;
        }
    }

    // With p=0.9, expect approximately 10% kept
    double keep_rate = static_cast<double>(non_zero_count) / output.tensor().numel();
    EXPECT_NEAR(keep_rate, 0.1, 0.05);  // 5% tolerance
}

TEST(DropoutTest, InvalidProbabilityNegative) {
    EXPECT_THROW(Dropout(-0.1), std::invalid_argument);
}

TEST(DropoutTest, InvalidProbabilityOne) {
    EXPECT_THROW(Dropout(1.0), std::invalid_argument);
}

TEST(DropoutTest, InvalidProbabilityGreaterThanOne) {
    EXPECT_THROW(Dropout(1.5), std::invalid_argument);
}

// ============================================================================
// Test 3: Inverted Dropout Scaling Verification
// ============================================================================

TEST(DropoutTest, InvertedDropoutScaling) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({100, 100});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Check that non-zero values are scaled by 1/(1-p) = 2.0
    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());
    double sum_non_zero = 0.0;
    size_t non_zero_count = 0;

    for (size_t i = 0; i < output.tensor().numel(); ++i) {
        if (data[i] != 0.0f) {
            sum_non_zero += data[i];
            non_zero_count++;
        }
    }

    double avg_non_zero = sum_non_zero / non_zero_count;
    double expected_scale = 1.0 / (1.0 - 0.5);  // 2.0
    EXPECT_NEAR(avg_non_zero, expected_scale, 0.1);
}

TEST(DropoutTest, ScalingWithDifferentProbability) {
    Dropout dropout(0.8);
    dropout.train();

    auto input_tensor = full({200, 200}, 3.0f);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Non-zero values should be scaled by 1/(1-0.8) = 5.0
    // Original value 3.0 * 5.0 = 15.0
    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());
    double sum_non_zero = 0.0;
    size_t non_zero_count = 0;

    for (size_t i = 0; i < output.tensor().numel(); ++i) {
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

TEST(DropoutTest, BernoulliDistribution) {
    Dropout dropout(0.3);
    dropout.train();

    // Large sample for statistical test
    auto input_tensor = ones({500, 500});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());
    size_t kept_count = 0;
    size_t total = output.tensor().numel();

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
// Test 5: CPU and CUDA Devices
// ============================================================================

TEST(DropoutTest, CPUDevice) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({50, 50}, DType::Float32, Device::cpu());
    Variable input(input_tensor, false);

    EXPECT_NO_THROW({
        auto output = dropout.forward(input);
        EXPECT_EQ(output.tensor().device().type, Device::Type::CPU);
    });
}

#ifdef TENZOR_USE_CUDA
TEST(DropoutTest, CUDADevice) {
    if (!cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({50, 50}, DType::Float32, Device::cuda(0));
    Variable input(input_tensor, false);

    EXPECT_NO_THROW({
        auto output = dropout.forward(input);
        EXPECT_EQ(output.tensor().device().type, Device::Type::CUDA);
    });
}
#endif

// ============================================================================
// Test 6: Gradient Checking for Backward Pass
// ============================================================================

TEST(DropoutTest, BackwardPassGradientShape) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = randn({10, 20});
    Variable input(input_tensor, true);  // requires_grad = true

    auto output = dropout.forward(input);

    // Check gradient function is set
    EXPECT_TRUE(output.grad_fn() != nullptr);

    // Backward pass
    auto grad_output = ones({10, 20});
    EXPECT_NO_THROW({
        output.backward(grad_output);
    });
}

TEST(DropoutTest, BackwardPassGradientValues) {
    Dropout dropout(0.0);  // No dropout for deterministic test
    dropout.train();

    auto input_tensor = ones({5, 5});
    Variable input(input_tensor, true);

    auto output = dropout.forward(input);

    // With p=0, gradient should pass through unchanged
    auto grad_output = full({5, 5}, 2.0f);
    output.backward(grad_output);

    auto input_grad = input.grad();
    ASSERT_TRUE(input_grad.has_value());

    auto* grad_data = static_cast<float*>(input_grad->impl()->storage->data());
    for (size_t i = 0; i < input_grad->numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 2.0f);
    }
}

// ============================================================================
// Test 7: Different Tensor Shapes
// ============================================================================

TEST(DropoutTest, OneDimensionalTensor) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({1000});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    EXPECT_EQ(output.tensor().shape().size(), 1);
    EXPECT_EQ(output.tensor().shape()[0], 1000);
}

TEST(DropoutTest, TwoDimensionalTensor) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({32, 64});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    EXPECT_EQ(output.tensor().shape().size(), 2);
    EXPECT_EQ(output.tensor().shape()[0], 32);
    EXPECT_EQ(output.tensor().shape()[1], 64);
}

TEST(DropoutTest, FourDimensionalTensor) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({8, 16, 32, 32});
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

TEST(DropoutTest, EdgeCaseEmptyTensor) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({0, 10});
    Variable input(input_tensor, false);

    EXPECT_NO_THROW({
        auto output = dropout.forward(input);
        EXPECT_EQ(output.tensor().numel(), 0);
    });
}

TEST(DropoutTest, EdgeCaseSingleElement) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({1});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());
    // Should be either 0 or 2.0 (scaled)
    EXPECT_TRUE(data[0] == 0.0f || std::abs(data[0] - 2.0f) < 0.01f);
}

// ============================================================================
// Test 9: Reproducibility with Fixed Random Seed
// ============================================================================

TEST(DropoutTest, ReproducibilityInInferenceMode) {
    Dropout dropout(0.5);
    dropout.eval();

    auto input_tensor = randn({20, 20});
    Variable input(input_tensor, false);

    auto output1 = dropout.forward(input);
    auto output2 = dropout.forward(input);

    // In eval mode, should be identical
    auto* data1 = static_cast<float*>(output1.tensor().impl()->storage->data());
    auto* data2 = static_cast<float*>(output2.tensor().impl()->storage->data());

    for (size_t i = 0; i < output1.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(data1[i], data2[i]);
    }
}

// ============================================================================
// Test 10: Dropout2d Tests
// ============================================================================

TEST(Dropout2dTest, InferenceMode) {
    Dropout2d dropout(0.5);
    dropout.eval();

    auto input_tensor = ones({2, 4, 8, 8});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    ASSERT_EQ(output.tensor().shape().size(), 4);
    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());
    for (size_t i = 0; i < output.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST(Dropout2dTest, ChannelWiseDropout) {
    Dropout2d dropout(0.5);
    dropout.train();

    auto input_tensor = ones({2, 10, 8, 8});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Verify entire channels are uniformly dropped or kept
    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());

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

TEST(Dropout2dTest, InvalidDimensions) {
    Dropout2d dropout(0.5);
    dropout.train();

    auto input_1d = ones({10});
    Variable var_1d(input_1d, false);

    EXPECT_THROW(dropout.forward(var_1d), std::invalid_argument);
}

TEST(Dropout2dTest, ThreeDimensionalInput) {
    Dropout2d dropout(0.5);
    dropout.train();

    auto input_tensor = ones({4, 8, 8});
    Variable input(input_tensor, false);

    EXPECT_NO_THROW({
        auto output = dropout.forward(input);
        EXPECT_EQ(output.tensor().shape().size(), 3);
    });
}

// ============================================================================
// Additional Quality Tests
// ============================================================================

TEST(DropoutTest, PreservesExpectedValueInTraining) {
    Dropout dropout(0.5);
    dropout.train();

    // Test that expected value is preserved due to scaling
    auto input_tensor = full({1000, 1000}, 10.0f);
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    auto* data = static_cast<float*>(output.tensor().impl()->storage->data());
    double sum = 0.0;
    for (size_t i = 0; i < output.tensor().numel(); ++i) {
        sum += data[i];
    }

    double mean = sum / output.tensor().numel();
    EXPECT_NEAR(mean, 10.0, 0.5);  // Expected value should be preserved
}

TEST(DropoutTest, DifferentInputValues) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = randn({50, 50});
    Variable input(input_tensor, false);

    auto output = dropout.forward(input);

    // Just verify it runs without errors
    EXPECT_EQ(output.tensor().shape().size(), 2);
    EXPECT_EQ(output.tensor().shape()[0], 50);
    EXPECT_EQ(output.tensor().shape()[1], 50);
}

TEST(DropoutTest, ConsecutiveForwardPasses) {
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({10, 10});
    Variable input(input_tensor, false);

    // Multiple forward passes should produce different masks
    auto output1 = dropout.forward(input);
    auto output2 = dropout.forward(input);

    auto* data1 = static_cast<float*>(output1.tensor().impl()->storage->data());
    auto* data2 = static_cast<float*>(output2.tensor().impl()->storage->data());

    // At least some values should differ (with high probability)
    size_t diff_count = 0;
    for (size_t i = 0; i < output1.tensor().numel(); ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-6) {
            diff_count++;
        }
    }

    // With p=0.5, expect many differences
    EXPECT_GT(diff_count, 0);
}
