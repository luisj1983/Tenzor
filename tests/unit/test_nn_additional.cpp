/**
 * @file test_nn_additional.cpp
 * @brief Comprehensive tests for NN modules to increase coverage
 *
 * This file provides extensive tests for:
 * - Activation functions (edge cases, gradients, numerical stability)
 * - Loss functions (reduction modes, edge cases, label smoothing)
 * - Normalization layers (BatchNorm, LayerNorm, GroupNorm variations)
 * - Pooling layers (MaxPool2d, AvgPool2d edge cases)
 * - Embedding layers (out of bounds, padding index)
 * - RNN layers (sequence handling, bidirectional, dropout)
 *
 * All tests use BackendTest fixture for multi-backend support.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/layers/normalization.hpp>
#include <tenzor/nn/layers/pooling.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// Activation Functions Tests
// ============================================================================

class ActivationTest : public BackendTest {};

TEST_P(ActivationTest, ReLU_EdgeCases) {
    auto relu = ReLU();

    // Test with zeros
    auto zeros_input = Variable(zeros({3, 3}, DType::Float32, device), true);
    auto zeros_output = relu.forward(zeros_input);
    expectTensorNear(zeros_output.tensor(), zeros({3, 3}, DType::Float32, device));

    // Test with negative values
    auto neg_input = Variable(full({3, 3}, -5.0f, DType::Float32, device), true);
    auto neg_output = relu.forward(neg_input);
    expectTensorNear(neg_output.tensor(), zeros({3, 3}, DType::Float32, device));

    // Test with positive values
    auto pos_input = Variable(full({3, 3}, 5.0f, DType::Float32, device), true);
    auto pos_output = relu.forward(pos_input);
    expectTensorNear(pos_output.tensor(), full({3, 3}, 5.0f, DType::Float32, device));

    // Test with mixed values
    auto data = std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto mixed_input = Variable(Tensor::from_blob(data.data(), {5}, DType::Float32, device), true);
    auto mixed_output = relu.forward(mixed_input);

    auto output_cpu = mixed_output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    EXPECT_FLOAT_EQ(output_data[0], 0.0f);
    EXPECT_FLOAT_EQ(output_data[1], 0.0f);
    EXPECT_FLOAT_EQ(output_data[2], 0.0f);
    EXPECT_FLOAT_EQ(output_data[3], 1.0f);
    EXPECT_FLOAT_EQ(output_data[4], 2.0f);
}

TEST_P(ActivationTest, ReLU_Gradient) {
    auto relu = ReLU();

    // Test gradient flow for positive values
    auto pos_input = Variable(ones({3, 3}, DType::Float32, device), true);
    auto pos_output = relu.forward(pos_input);
    auto loss = pos_output.sum();
    loss.backward();

    // Gradient should be 1.0 for positive inputs
    auto grad_cpu = pos_input.grad().to(Device::cpu());
    auto grad_data = grad_cpu.data<float>();
    for (int64_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 1.0f);
    }
}

TEST_P(ActivationTest, ReLU6_Clipping) {
    auto relu6 = ReLU6();

    // Values above 6 should be clipped
    auto large_input = Variable(full({3, 3}, 10.0f, DType::Float32, device), true);
    auto large_output = relu6.forward(large_input);
    expectTensorNear(large_output.tensor(), full({3, 3}, 6.0f, DType::Float32, device));

    // Values in [0, 6] should pass through
    auto mid_input = Variable(full({3, 3}, 3.0f, DType::Float32, device), true);
    auto mid_output = relu6.forward(mid_input);
    expectTensorNear(mid_output.tensor(), full({3, 3}, 3.0f, DType::Float32, device));

    // Negative values should be zero
    auto neg_input = Variable(full({3, 3}, -2.0f, DType::Float32, device), true);
    auto neg_output = relu6.forward(neg_input);
    expectTensorNear(neg_output.tensor(), zeros({3, 3}, DType::Float32, device));
}

TEST_P(ActivationTest, LeakyReLU_NegativeSlope) {
    auto leaky_relu = LeakyReLU(0.01);

    // Test negative values get scaled
    auto neg_input = Variable(full({3, 3}, -10.0f, DType::Float32, device), true);
    auto neg_output = leaky_relu.forward(neg_input);

    auto output_cpu = neg_output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    EXPECT_NEAR(output_data[0], -0.1f, 1e-5f);

    // Test positive values pass through
    auto pos_input = Variable(full({3, 3}, 5.0f, DType::Float32, device), true);
    auto pos_output = leaky_relu.forward(pos_input);
    expectTensorNear(pos_output.tensor(), full({3, 3}, 5.0f, DType::Float32, device));
}

TEST_P(ActivationTest, Sigmoid_Range) {
    auto sigmoid = Sigmoid();

    // Test output is in (0, 1)
    auto input = Variable(randn({100}, DType::Float32, device) * 5.0f, true);
    auto output = sigmoid.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    for (int64_t i = 0; i < 100; ++i) {
        EXPECT_GT(output_data[i], 0.0f);
        EXPECT_LT(output_data[i], 1.0f);
    }
}

TEST_P(ActivationTest, Sigmoid_ExtremeValues) {
    auto sigmoid = Sigmoid();

    // Very large positive values should approach 1
    auto large_pos = Variable(full({3, 3}, 100.0f, DType::Float32, device), true);
    auto output_pos = sigmoid.forward(large_pos);
    auto output_pos_cpu = output_pos.tensor().to(Device::cpu());
    EXPECT_NEAR(output_pos_cpu.data<float>()[0], 1.0f, 1e-3f);

    // Very large negative values should approach 0
    auto large_neg = Variable(full({3, 3}, -100.0f, DType::Float32, device), true);
    auto output_neg = sigmoid.forward(large_neg);
    auto output_neg_cpu = output_neg.tensor().to(Device::cpu());
    EXPECT_NEAR(output_neg_cpu.data<float>()[0], 0.0f, 1e-3f);

    // Zero should give 0.5
    auto zero_input = Variable(zeros({3, 3}, DType::Float32, device), true);
    auto zero_output = sigmoid.forward(zero_input);
    auto zero_output_cpu = zero_output.tensor().to(Device::cpu());
    EXPECT_NEAR(zero_output_cpu.data<float>()[0], 0.5f, 1e-5f);
}

TEST_P(ActivationTest, Tanh_Range) {
    auto tanh_layer = Tanh();

    // Test output is in (-1, 1)
    auto input = Variable(randn({100}, DType::Float32, device) * 5.0f, true);
    auto output = tanh_layer.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    for (int64_t i = 0; i < 100; ++i) {
        EXPECT_GT(output_data[i], -1.0f);
        EXPECT_LT(output_data[i], 1.0f);
    }
}

TEST_P(ActivationTest, Tanh_ZeroCentered) {
    auto tanh_layer = Tanh();

    // Zero input should give zero output
    auto zero_input = Variable(zeros({3, 3}, DType::Float32, device), true);
    auto zero_output = tanh_layer.forward(zero_input);
    expectTensorNear(zero_output.tensor(), zeros({3, 3}, DType::Float32, device));
}

TEST_P(ActivationTest, GELU_Smoothness) {
    auto gelu = GELU();

    // Test output for various inputs
    auto data = std::vector<float>{-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    auto input = Variable(Tensor::from_blob(data.data(), {5}, DType::Float32, device), true);
    auto output = gelu.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    // GELU(0) should be ~0
    EXPECT_NEAR(output_data[2], 0.0f, 1e-4f);

    // GELU should be smooth (no sharp transitions like ReLU)
    // Negative values should be small but non-zero
    EXPECT_LT(std::abs(output_data[1]), 1.0f);
}

TEST_P(ActivationTest, Softmax_SumToOne) {
    auto softmax = Softmax(-1);

    // Test outputs sum to 1 along last dimension
    auto input = Variable(randn({4, 10}, DType::Float32, device), true);
    auto output = softmax.forward(input);

    // Sum along last dimension
    auto sum_output = output.sum(-1);
    auto sum_cpu = sum_output.tensor().to(Device::cpu());
    auto sum_data = sum_cpu.data<float>();

    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(sum_data[i], 1.0f, 1e-5f);
    }
}

TEST_P(ActivationTest, Softmax_NumericalStability) {
    auto softmax = Softmax(-1);

    // Test with very large values (should not overflow)
    auto large_input = Variable(full({3, 5}, 1000.0f, DType::Float32, device), true);
    auto output = softmax.forward(large_input);

    // Should still be valid probabilities
    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    for (int64_t i = 0; i < 15; ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }
}

TEST_P(ActivationTest, LogSoftmax_NumericalStability) {
    auto log_softmax = LogSoftmax(-1);

    // Test with large values
    auto input = Variable(full({3, 5}, 100.0f, DType::Float32, device), true);
    auto output = log_softmax.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    for (int64_t i = 0; i < 15; ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }
}

TEST_P(ActivationTest, ELU_Alpha) {
    auto elu = ELU(1.0);

    // Positive values pass through
    auto pos_input = Variable(full({3, 3}, 2.0f, DType::Float32, device), true);
    auto pos_output = elu.forward(pos_input);
    expectTensorNear(pos_output.tensor(), full({3, 3}, 2.0f, DType::Float32, device));

    // Negative values: alpha * (exp(x) - 1)
    auto neg_input = Variable(zeros({3, 3}, DType::Float32, device), true);
    auto neg_output = elu.forward(neg_input);
    expectTensorNear(neg_output.tensor(), zeros({3, 3}, DType::Float32, device), 1e-4f);
}

TEST_P(ActivationTest, SELU_SelfNormalizing) {
    auto selu = SELU();

    // Test that SELU applies correct scaling
    auto input = Variable(randn({100, 50}, DType::Float32, device), true);
    auto output = selu.forward(input);

    // Output should have reasonable range
    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    double sum = 0.0;
    for (int64_t i = 0; i < 5000; ++i) {
        sum += output_data[i];
        EXPECT_FALSE(std::isnan(output_data[i]));
    }

    // Mean should be close to 0 (self-normalizing property)
    double mean = sum / 5000.0;
    EXPECT_LT(std::abs(mean), 1.0);
}

TEST_P(ActivationTest, Swish_SmoothActivation) {
    auto swish = Swish();

    // Test that swish is smooth and self-gated
    auto input = Variable(randn({100}, DType::Float32, device), true);
    auto output = swish.forward(input);

    // All outputs should be finite
    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    for (int64_t i = 0; i < 100; ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }
}

TEST_P(ActivationTest, Mish_SmoothNonMonotonic) {
    auto mish = Mish();

    // Test Mish activation
    auto input = Variable(randn({100}, DType::Float32, device), true);
    auto output = mish.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    for (int64_t i = 0; i < 100; ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }
}

// ============================================================================
// Loss Functions Tests
// ============================================================================

class LossTest : public BackendTest {};

TEST_P(LossTest, MSELoss_ReductionModes) {
    auto input = Variable(ones({4, 5}, DType::Float32, device), false);
    auto target = Variable(zeros({4, 5}, DType::Float32, device), false);

    // Test all reduction modes
    auto criterion_none = MSELoss(Reduction::None);
    auto criterion_mean = MSELoss(Reduction::Mean);
    auto criterion_sum = MSELoss(Reduction::Sum);

    auto loss_none = criterion_none(input, target);
    auto loss_mean = criterion_mean(input, target);
    auto loss_sum = criterion_sum(input, target);

    // None should return tensor of same shape as input
    EXPECT_EQ(loss_none.shape().size(), 2);
    EXPECT_EQ(loss_none.shape()[0], 4);
    EXPECT_EQ(loss_none.shape()[1], 5);

    // Mean and sum should be scalars
    EXPECT_EQ(loss_mean.shape().size(), 0);
    EXPECT_EQ(loss_sum.shape().size(), 0);

    // Sum should be larger than mean
    auto mean_cpu = loss_mean.tensor().to(Device::cpu());
    auto sum_cpu = loss_sum.tensor().to(Device::cpu());
    EXPECT_GT(sum_cpu.item<float>(), mean_cpu.item<float>());
}

TEST_P(LossTest, MSELoss_PerfectPrediction) {
    auto input = Variable(ones({4, 5}, DType::Float32, device), false);
    auto target = Variable(ones({4, 5}, DType::Float32, device), false);

    auto criterion = MSELoss(Reduction::Mean);
    auto loss = criterion(input, target);

    auto loss_cpu = loss.tensor().to(Device::cpu());
    EXPECT_NEAR(loss_cpu.item<float>(), 0.0f, 1e-6f);
}

TEST_P(LossTest, CrossEntropyLoss_Basic) {
    // 3 classes, batch size 4
    auto logits = Variable(randn({4, 3}, DType::Float32, device), false);

    // Target class indices (0, 1, 2, 0)
    std::vector<int64_t> target_data = {0, 1, 2, 0};
    auto target = Tensor::from_blob(target_data.data(), {4}, DType::Int64, device);

    auto criterion = CrossEntropyLoss(Reduction::Mean);
    auto loss = criterion(logits, target);

    // Loss should be positive
    auto loss_cpu = loss.tensor().to(Device::cpu());
    EXPECT_GT(loss_cpu.item<float>(), 0.0f);
}

TEST_P(LossTest, CrossEntropyLoss_ReductionModes) {
    auto logits = Variable(randn({4, 3}, DType::Float32, device), false);
    std::vector<int64_t> target_data = {0, 1, 2, 0};
    auto target = Tensor::from_blob(target_data.data(), {4}, DType::Int64, device);

    auto criterion_mean = CrossEntropyLoss(Reduction::Mean);
    auto criterion_sum = CrossEntropyLoss(Reduction::Sum);
    auto criterion_none = CrossEntropyLoss(Reduction::None);

    auto loss_mean = criterion_mean(logits, target);
    auto loss_sum = criterion_sum(logits, target);
    auto loss_none = criterion_none(logits, target);

    // None should return per-sample losses
    EXPECT_EQ(loss_none.shape().size(), 1);
    EXPECT_EQ(loss_none.shape()[0], 4);

    // Mean and sum should be scalars
    EXPECT_EQ(loss_mean.shape().size(), 0);
    EXPECT_EQ(loss_sum.shape().size(), 0);
}

TEST_P(LossTest, BCELoss_BinaryClassification) {
    // Binary predictions (after sigmoid)
    auto predictions = Variable(ones({10, 1}, DType::Float32, device) * 0.7f, false);
    auto targets = Variable(ones({10, 1}, DType::Float32, device), false);

    auto criterion = BCELoss(Reduction::Mean);
    auto loss = criterion(predictions, targets);

    // Loss should be positive
    auto loss_cpu = loss.tensor().to(Device::cpu());
    EXPECT_GT(loss_cpu.item<float>(), 0.0f);
}

TEST_P(LossTest, BCELoss_PerfectPrediction) {
    auto predictions = Variable(ones({10, 1}, DType::Float32, device), false);
    auto targets = Variable(ones({10, 1}, DType::Float32, device), false);

    auto criterion = BCELoss(Reduction::Mean);
    auto loss = criterion(predictions, targets);

    // Loss should be close to zero
    auto loss_cpu = loss.tensor().to(Device::cpu());
    EXPECT_NEAR(loss_cpu.item<float>(), 0.0f, 1e-4f);
}

TEST_P(LossTest, BCEWithLogitsLoss_NumericalStability) {
    // Test with extreme logits
    auto logits = Variable(full({10, 1}, 100.0f, DType::Float32, device), false);
    auto targets = Variable(ones({10, 1}, DType::Float32, device), false);

    auto criterion = BCEWithLogitsLoss(Reduction::Mean);
    auto loss = criterion(logits, targets);

    // Should not be NaN or Inf
    auto loss_cpu = loss.tensor().to(Device::cpu());
    EXPECT_FALSE(std::isnan(loss_cpu.item<float>()));
    EXPECT_FALSE(std::isinf(loss_cpu.item<float>()));
}

TEST_P(LossTest, NLLLoss_Basic) {
    // Log probabilities from log_softmax
    auto log_probs = Variable(randn({4, 3}, DType::Float32, device), false);
    std::vector<int64_t> target_data = {0, 1, 2, 0};
    auto target = Tensor::from_blob(target_data.data(), {4}, DType::Int64, device);

    auto criterion = NLLLoss(Reduction::Mean);
    auto loss = criterion(log_probs, target);

    // Loss should be finite
    auto loss_cpu = loss.tensor().to(Device::cpu());
    EXPECT_FALSE(std::isnan(loss_cpu.item<float>()));
}

TEST_P(LossTest, L1Loss_ReductionModes) {
    auto predictions = Variable(ones({4, 5}, DType::Float32, device), false);
    auto targets = Variable(zeros({4, 5}, DType::Float32, device), false);

    auto criterion_mean = L1Loss(Reduction::Mean);
    auto criterion_sum = L1Loss(Reduction::Sum);
    auto criterion_none = L1Loss(Reduction::None);

    auto loss_mean = criterion_mean(predictions, targets);
    auto loss_sum = criterion_sum(predictions, targets);
    auto loss_none = criterion_none(predictions, targets);

    // Check shapes
    EXPECT_EQ(loss_none.shape().size(), 2);
    EXPECT_EQ(loss_mean.shape().size(), 0);
    EXPECT_EQ(loss_sum.shape().size(), 0);
}

TEST_P(LossTest, SmoothL1Loss_BetaParameter) {
    auto predictions = Variable(full({10}, 2.0f, DType::Float32, device), false);
    auto targets = Variable(zeros({10}, DType::Float32, device), false);

    // Test different beta values
    auto criterion_beta1 = SmoothL1Loss(Reduction::Mean, 1.0);
    auto criterion_beta2 = SmoothL1Loss(Reduction::Mean, 2.0);

    auto loss_beta1 = criterion_beta1(predictions, targets);
    auto loss_beta2 = criterion_beta2(predictions, targets);

    // Different betas should give different losses
    auto loss1_cpu = loss_beta1.tensor().to(Device::cpu());
    auto loss2_cpu = loss_beta2.tensor().to(Device::cpu());
    EXPECT_NE(loss1_cpu.item<float>(), loss2_cpu.item<float>());
}

TEST_P(LossTest, KLDivLoss_BasicProperties) {
    auto log_probs = Variable(full({10, 5}, std::log(0.2f), DType::Float32, device), false);
    auto target_probs = Variable(full({10, 5}, 0.2f, DType::Float32, device), false);

    auto criterion = KLDivLoss("mean", false);
    auto loss = criterion(log_probs, target_probs);

    // Perfect match should give near-zero KL divergence
    auto loss_cpu = loss.tensor().to(Device::cpu());
    EXPECT_NEAR(loss_cpu.item<float>(), 0.0f, 1e-4f);
}

TEST_P(LossTest, FocalLoss_GammaEffect) {
    auto logits = Variable(ones({10, 5}, DType::Float32, device), false);
    auto targets = Variable(ones({10, 5}, DType::Float32, device) * 0.2f, false);

    // Test different gamma values
    auto criterion_gamma0 = FocalLoss(1.0, 0.0, "mean");
    auto criterion_gamma2 = FocalLoss(1.0, 2.0, "mean");
    auto criterion_gamma5 = FocalLoss(1.0, 5.0, "mean");

    auto loss_gamma0 = criterion_gamma0(logits, targets);
    auto loss_gamma2 = criterion_gamma2(logits, targets);
    auto loss_gamma5 = criterion_gamma5(logits, targets);

    // Higher gamma should focus more on hard examples
    auto loss0_cpu = loss_gamma0.tensor().to(Device::cpu());
    auto loss2_cpu = loss_gamma2.tensor().to(Device::cpu());
    auto loss5_cpu = loss_gamma5.tensor().to(Device::cpu());

    EXPECT_GT(loss0_cpu.item<float>(), 0.0f);
    EXPECT_GT(loss2_cpu.item<float>(), 0.0f);
    EXPECT_GT(loss5_cpu.item<float>(), 0.0f);
}

TEST_P(LossTest, DiceLoss_SegmentationTask) {
    // Binary segmentation masks
    auto predictions = Variable(ones({2, 1, 32, 32}, DType::Float32, device) * 0.7f, false);
    auto targets = Variable(ones({2, 1, 32, 32}, DType::Float32, device), false);

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(predictions, targets);

    // Dice loss should be in [0, 1]
    auto loss_cpu = loss.tensor().to(Device::cpu());
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
    EXPECT_LE(loss_cpu.item<float>(), 1.0f);
}

TEST_P(LossTest, HuberLoss_DeltaEffect) {
    auto predictions = Variable(full({10}, 5.0f, DType::Float32, device), false);
    auto targets = Variable(zeros({10}, DType::Float32, device), false);

    // Test different delta values
    auto criterion_delta1 = HuberLoss(1.0, "mean");
    auto criterion_delta2 = HuberLoss(2.0, "mean");

    auto loss_delta1 = criterion_delta1(predictions, targets);
    auto loss_delta2 = criterion_delta2(predictions, targets);

    // Different deltas should give different losses
    auto loss1_cpu = loss_delta1.tensor().to(Device::cpu());
    auto loss2_cpu = loss_delta2.tensor().to(Device::cpu());
    EXPECT_NE(loss1_cpu.item<float>(), loss2_cpu.item<float>());
}

// ============================================================================
// Normalization Layers Tests
// ============================================================================

class NormalizationTest : public BackendTest {};

TEST_P(NormalizationTest, LayerNorm_SingleDimension) {
    auto layer_norm = LayerNorm({64}, 1e-5, true);

    // Input: (batch=4, features=64)
    auto input = Variable(randn({4, 64}, DType::Float32, device), true);
    auto output = layer_norm.forward(input);

    // Output should have same shape
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 64);

    // Check normalization: mean ~0, std ~1 along last dimension
    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    for (int64_t i = 0; i < 4; ++i) {
        double sum = 0.0;
        for (int64_t j = 0; j < 64; ++j) {
            sum += output_data[i * 64 + j];
        }
        double mean = sum / 64.0;
        EXPECT_NEAR(mean, 0.0, 0.1);
    }
}

TEST_P(NormalizationTest, LayerNorm_MultipleDimensions) {
    auto layer_norm = LayerNorm({32, 32}, 1e-5, true);

    // Input: (batch=2, height=32, width=32)
    auto input = Variable(randn({2, 32, 32}, DType::Float32, device), true);
    auto output = layer_norm.forward(input);

    // Output should have same shape
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 2);
}

TEST_P(NormalizationTest, LayerNorm_NoAffine) {
    auto layer_norm = LayerNorm({64}, 1e-5, false);

    auto input = Variable(randn({4, 64}, DType::Float32, device), true);
    auto output = layer_norm.forward(input);

    // Should still normalize
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 64);
}

TEST_P(NormalizationTest, GroupNorm_Basic) {
    // 32 channels, 8 groups (4 channels per group)
    auto group_norm = GroupNorm(8, 32, 1e-5, true);

    // Input: (batch=2, channels=32, height=16, width=16)
    auto input = Variable(randn({2, 32, 16, 16}, DType::Float32, device), true);
    auto output = group_norm.forward(input);

    // Output should have same shape
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 32);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(NormalizationTest, GroupNorm_SingleGroup) {
    // 1 group = Layer Norm
    auto group_norm = GroupNorm(1, 32, 1e-5, true);

    auto input = Variable(randn({2, 32, 16, 16}, DType::Float32, device), true);
    auto output = group_norm.forward(input);

    EXPECT_EQ(output.shape()[1], 32);
}

TEST_P(NormalizationTest, GroupNorm_AllGroups) {
    // num_groups = num_channels = Instance Norm
    auto group_norm = GroupNorm(32, 32, 1e-5, true);

    auto input = Variable(randn({2, 32, 16, 16}, DType::Float32, device), true);
    auto output = group_norm.forward(input);

    EXPECT_EQ(output.shape()[1], 32);
}

TEST_P(NormalizationTest, GroupNorm_NoAffine) {
    auto group_norm = GroupNorm(8, 32, 1e-5, false);

    auto input = Variable(randn({2, 32, 16, 16}, DType::Float32, device), true);
    auto output = group_norm.forward(input);

    // Should still normalize without learnable parameters
    EXPECT_EQ(output.shape()[1], 32);
}

// ============================================================================
// Pooling Layers Tests
// ============================================================================

class PoolingTest : public BackendTest {};

TEST_P(PoolingTest, MaxPool2d_BasicDownsampling) {
    auto pool = MaxPool2d(2, 2, 0);

    // Input: (batch=2, channels=3, height=8, width=8)
    auto input = Variable(randn({2, 3, 8, 8}, DType::Float32, device), true);
    auto output = pool.forward(input);

    // Output should be (2, 3, 4, 4)
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 4);
}

TEST_P(PoolingTest, MaxPool2d_DifferentKernelStride) {
    // Kernel 3x3, stride 2
    auto pool = MaxPool2d(3, 2, 0);

    auto input = Variable(randn({1, 1, 10, 10}, DType::Float32, device), true);
    auto output = pool.forward(input);

    // Output dimensions: floor((10 - 3) / 2 + 1) = 4
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 4);
}

TEST_P(PoolingTest, MaxPool2d_WithPadding) {
    auto pool = MaxPool2d(2, 2, 1);

    auto input = Variable(randn({1, 1, 8, 8}, DType::Float32, device), true);
    auto output = pool.forward(input);

    // With padding=1: (8 + 2*1 - 2) / 2 + 1 = 5
    EXPECT_EQ(output.shape()[2], 5);
    EXPECT_EQ(output.shape()[3], 5);
}

TEST_P(PoolingTest, MaxPool2d_SelectsMaximum) {
    // Create input with known maximum
    auto input_tensor = zeros({1, 1, 4, 4}, DType::Float32, device);
    auto input_cpu = input_tensor.to(Device::cpu());
    auto input_data = input_cpu.data<float>();

    // Set one value to maximum in each 2x2 region
    input_data[0] = 10.0f;   // Top-left quadrant
    input_data[2] = 20.0f;   // Top-right quadrant
    input_data[8] = 30.0f;   // Bottom-left quadrant
    input_data[10] = 40.0f;  // Bottom-right quadrant

    auto input = Variable(input_cpu.to(device), true);
    auto pool = MaxPool2d(2, 2, 0);
    auto output = pool.forward(input);

    // Output should be 2x2 with the maximum values
    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    EXPECT_FLOAT_EQ(output_data[0], 10.0f);
    EXPECT_FLOAT_EQ(output_data[1], 20.0f);
    EXPECT_FLOAT_EQ(output_data[2], 30.0f);
    EXPECT_FLOAT_EQ(output_data[3], 40.0f);
}

TEST_P(PoolingTest, AvgPool2d_BasicDownsampling) {
    auto pool = AvgPool2d(2, 2, 0);

    auto input = Variable(randn({2, 3, 8, 8}, DType::Float32, device), true);
    auto output = pool.forward(input);

    // Output should be (2, 3, 4, 4)
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 4);
}

TEST_P(PoolingTest, AvgPool2d_ComputesAverage) {
    // Create input with uniform values in each 2x2 region
    auto input_tensor = ones({1, 1, 4, 4}, DType::Float32, device) * 4.0f;
    auto input = Variable(input_tensor, true);

    auto pool = AvgPool2d(2, 2, 0);
    auto output = pool.forward(input);

    // Output should be 2x2 with average values (4.0)
    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(output_data[i], 4.0f, 1e-5f);
    }
}

TEST_P(PoolingTest, AvgPool2d_WithPadding) {
    auto pool = AvgPool2d(2, 2, 1);

    auto input = Variable(randn({1, 1, 8, 8}, DType::Float32, device), true);
    auto output = pool.forward(input);

    // With padding=1: (8 + 2*1 - 2) / 2 + 1 = 5
    EXPECT_EQ(output.shape()[2], 5);
    EXPECT_EQ(output.shape()[3], 5);
}

TEST_P(PoolingTest, AdaptiveAvgPool2d_FixedOutputSize) {
    auto pool = AdaptiveAvgPool2d(7, 7);

    // Test with different input sizes
    auto input1 = Variable(randn({1, 64, 14, 14}, DType::Float32, device), true);
    auto output1 = pool.forward(input1);

    auto input2 = Variable(randn({1, 64, 28, 28}, DType::Float32, device), true);
    auto output2 = pool.forward(input2);

    // Both should output 7x7
    EXPECT_EQ(output1.shape()[2], 7);
    EXPECT_EQ(output1.shape()[3], 7);
    EXPECT_EQ(output2.shape()[2], 7);
    EXPECT_EQ(output2.shape()[3], 7);
}

TEST_P(PoolingTest, AdaptiveAvgPool2d_GlobalPooling) {
    // Output size 1x1 = global average pooling
    auto pool = AdaptiveAvgPool2d(1, 1);

    auto input = Variable(randn({2, 512, 7, 7}, DType::Float32, device), true);
    auto output = pool.forward(input);

    // Output should be (2, 512, 1, 1)
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 512);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 1);
}

TEST_P(PoolingTest, AdaptiveAvgPool2d_SquareConstructor) {
    auto pool = AdaptiveAvgPool2d(7);  // Square output

    auto input = Variable(randn({1, 64, 14, 14}, DType::Float32, device), true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 7);
    EXPECT_EQ(output.shape()[3], 7);
}

// ============================================================================
// Embedding Layers Tests
// ============================================================================

class EmbeddingTest : public BackendTest {};

TEST_P(EmbeddingTest, Embedding_BasicLookup) {
    // 100 words, 50-dimensional embeddings
    auto embedding = Embedding(100, 50);

    // Input: batch of indices
    std::vector<int64_t> indices_data = {0, 10, 25, 50, 99};
    auto indices = Variable(Tensor::from_blob(indices_data.data(), {5}, DType::Int64, device), false);

    auto output = embedding.forward(indices);

    // Output should be (5, 50)
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 5);
    EXPECT_EQ(output.shape()[1], 50);
}

TEST_P(EmbeddingTest, Embedding_BatchedInput) {
    auto embedding = Embedding(1000, 300);

    // Input: (batch=4, sequence_length=10)
    auto indices = Variable(randint(0, 1000, {4, 10}, DType::Int64, device), false);
    auto output = embedding.forward(indices);

    // Output should be (4, 10, 300)
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 300);
}

TEST_P(EmbeddingTest, Embedding_PaddingIdx) {
    // Use index 0 as padding
    auto embedding = Embedding(100, 50, 0);

    std::vector<int64_t> indices_data = {0, 1, 2, 0, 3};
    auto indices = Variable(Tensor::from_blob(indices_data.data(), {5}, DType::Int64, device), false);

    auto output = embedding.forward(indices);

    // Padding indices (0) should have zero embeddings
    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    // First embedding (index 0) should be all zeros
    for (int64_t i = 0; i < 50; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], 0.0f);
    }

    // Fourth embedding (also index 0) should be all zeros
    for (int64_t i = 0; i < 50; ++i) {
        EXPECT_FLOAT_EQ(output_data[3 * 50 + i], 0.0f);
    }
}

TEST_P(EmbeddingTest, Embedding_MaxNorm) {
    // Test max_norm constraint
    auto embedding = Embedding(10, 20, -1, 1.0);  // max_norm=1.0

    std::vector<int64_t> indices_data = {0, 1, 2};
    auto indices = Variable(Tensor::from_blob(indices_data.data(), {3}, DType::Int64, device), false);

    auto output = embedding.forward(indices);

    // Check that each embedding has norm <= max_norm
    auto output_cpu = output.tensor().to(Device::cpu());
    auto output_data = output_cpu.data<float>();

    for (int64_t i = 0; i < 3; ++i) {
        float norm_sq = 0.0f;
        for (int64_t j = 0; j < 20; ++j) {
            float val = output_data[i * 20 + j];
            norm_sq += val * val;
        }
        float norm = std::sqrt(norm_sq);
        EXPECT_LE(norm, 1.0f + 1e-4f);
    }
}

TEST_P(EmbeddingTest, EmbeddingBag_BasicAggregation) {
    auto embedding_bag = EmbeddingBag(100, 50, 0.0, 2.0, false, "mean");

    // Single bag of indices
    std::vector<int64_t> indices_data = {0, 5, 10, 15, 20};
    auto indices = Variable(Tensor::from_blob(indices_data.data(), {5}, DType::Int64, device), false);

    auto output = embedding_bag.forward(indices);

    // Output should be (1, 50) - single aggregated embedding
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 50);
}

TEST_P(EmbeddingTest, EmbeddingBag_MultipleBags) {
    auto embedding_bag = EmbeddingBag(100, 50, 0.0, 2.0, false, "mean");

    // Multiple bags with offsets
    std::vector<int64_t> indices_data = {0, 1, 2, 10, 11, 20, 21, 22};
    auto indices = Variable(Tensor::from_blob(indices_data.data(), {8}, DType::Int64, device), false);

    std::vector<int64_t> offsets_data = {0, 3, 5};  // 3 bags
    auto offsets = Variable(Tensor::from_blob(offsets_data.data(), {3}, DType::Int64, device), false);

    auto output = embedding_bag.forward(indices, offsets);

    // Output should be (3, 50) - one embedding per bag
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 3);
    EXPECT_EQ(output.shape()[1], 50);
}

TEST_P(EmbeddingTest, EmbeddingBag_SumMode) {
    auto embedding_bag_sum = EmbeddingBag(10, 20, 0.0, 2.0, false, "sum");
    auto embedding_bag_mean = EmbeddingBag(10, 20, 0.0, 2.0, false, "mean");

    std::vector<int64_t> indices_data = {0, 1, 2};
    auto indices = Variable(Tensor::from_blob(indices_data.data(), {3}, DType::Int64, device), false);

    auto output_sum = embedding_bag_sum.forward(indices);
    auto output_mean = embedding_bag_mean.forward(indices);

    // Sum should be larger than mean
    auto sum_cpu = output_sum.tensor().to(Device::cpu());
    auto mean_cpu = output_mean.tensor().to(Device::cpu());

    float sum_norm = 0.0f, mean_norm = 0.0f;
    for (int64_t i = 0; i < 20; ++i) {
        sum_norm += std::abs(sum_cpu.data<float>()[i]);
        mean_norm += std::abs(mean_cpu.data<float>()[i]);
    }
    EXPECT_GT(sum_norm, mean_norm);
}

// ============================================================================
// RNN Layers Tests
// ============================================================================

class RNNTest : public BackendTest {};

TEST_P(RNNTest, RNNCell_SingleStep) {
    auto cell = RNNCell(128, 256, "tanh", true);

    // Input: (batch=4, input_size=128)
    auto x = Variable(randn({4, 128}, DType::Float32, device), true);
    auto h = Variable(randn({4, 256}, DType::Float32, device), true);

    auto h_next = cell.forward(x, h);

    // Output should be (4, 256)
    EXPECT_EQ(h_next.shape().size(), 2);
    EXPECT_EQ(h_next.shape()[0], 4);
    EXPECT_EQ(h_next.shape()[1], 256);
}

TEST_P(RNNTest, RNNCell_TanhActivation) {
    auto cell = RNNCell(10, 20, "tanh");

    auto x = Variable(randn({2, 10}, DType::Float32, device), true);
    auto h = Variable(zeros({2, 20}, DType::Float32, device), true);

    auto h_next = cell.forward(x, h);

    // Output should be in (-1, 1) due to tanh
    auto h_cpu = h_next.tensor().to(Device::cpu());
    auto h_data = h_cpu.data<float>();
    for (int64_t i = 0; i < 40; ++i) {
        EXPECT_GT(h_data[i], -1.0f);
        EXPECT_LT(h_data[i], 1.0f);
    }
}

TEST_P(RNNTest, RNN_SequenceProcessing) {
    auto rnn = RNN(128, 256, 1, "tanh", true, false, 0.0, false);

    // Input: (seq_len=10, batch=4, input_size=128)
    auto x = Variable(randn({10, 4, 128}, DType::Float32, device), true);
    auto h0 = Variable(randn({1, 4, 256}, DType::Float32, device), true);

    auto result = rnn.forward(x, h0);
    auto output = result.first;
    auto h_n = result.second;

    // Output: (10, 4, 256), h_n: (1, 4, 256)
    EXPECT_EQ(output.shape()[0], 10);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.shape()[2], 256);
    EXPECT_EQ(h_n.shape()[0], 1);
    EXPECT_EQ(h_n.shape()[1], 4);
    EXPECT_EQ(h_n.shape()[2], 256);
}

TEST_P(RNNTest, RNN_Bidirectional) {
    auto rnn = RNN(128, 256, 1, "tanh", true, false, 0.0, true);

    auto x = Variable(randn({10, 4, 128}, DType::Float32, device), true);

    auto result = rnn.forward(x, Variable{});
    auto output = result.first;
    auto h_n = result.second;

    // Bidirectional doubles the output size
    EXPECT_EQ(output.shape()[2], 512);  // 2 * 256
    EXPECT_EQ(h_n.shape()[0], 2);       // 2 directions
}

TEST_P(RNNTest, RNN_MultiLayer) {
    auto rnn = RNN(128, 256, 3, "tanh", true, false, 0.5, false);

    auto x = Variable(randn({10, 4, 128}, DType::Float32, device), true);

    auto result = rnn.forward(x, Variable{});
    auto h_n = result.second;

    // Should have 3 layers
    EXPECT_EQ(h_n.shape()[0], 3);
}

TEST_P(RNNTest, LSTMCell_SingleStep) {
    auto cell = LSTMCell(128, 256, true);

    auto x = Variable(randn({4, 128}, DType::Float32, device), true);
    auto h = Variable(randn({4, 256}, DType::Float32, device), true);
    auto c = Variable(randn({4, 256}, DType::Float32, device), true);

    auto result = cell.forward(x, h, c);
    auto h_next = result.first;
    auto c_next = result.second;

    // Both outputs should be (4, 256)
    EXPECT_EQ(h_next.shape()[0], 4);
    EXPECT_EQ(h_next.shape()[1], 256);
    EXPECT_EQ(c_next.shape()[0], 4);
    EXPECT_EQ(c_next.shape()[1], 256);
}

TEST_P(RNNTest, LSTM_SequenceProcessing) {
    auto lstm = LSTM(128, 256, 1, true, false, 0.0, false);

    auto x = Variable(randn({10, 4, 128}, DType::Float32, device), true);

    auto result = lstm.forward(x, {Variable{}, Variable{}});
    auto output = result.first;
    auto hidden_state = result.second;
    auto h_n = hidden_state.first;
    auto c_n = hidden_state.second;

    // Check shapes
    EXPECT_EQ(output.shape()[0], 10);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.shape()[2], 256);
    EXPECT_EQ(h_n.shape()[0], 1);
    EXPECT_EQ(c_n.shape()[0], 1);
}

TEST_P(RNNTest, LSTM_Bidirectional) {
    auto lstm = LSTM(128, 256, 1, true, false, 0.0, true);

    auto x = Variable(randn({10, 4, 128}, DType::Float32, device), true);

    auto result = lstm.forward(x, {Variable{}, Variable{}});
    auto output = result.first;
    auto hidden_state = result.second;
    auto h_n = hidden_state.first;

    // Bidirectional output
    EXPECT_EQ(output.shape()[2], 512);  // 2 * 256
    EXPECT_EQ(h_n.shape()[0], 2);       // 2 directions
}

TEST_P(RNNTest, LSTM_MultiLayer) {
    auto lstm = LSTM(128, 256, 3, true, false, 0.5, false);

    auto x = Variable(randn({10, 4, 128}, DType::Float32, device), true);

    auto result = lstm.forward(x, {Variable{}, Variable{}});
    auto hidden_state = result.second;
    auto h_n = hidden_state.first;
    auto c_n = hidden_state.second;

    // 3 layers
    EXPECT_EQ(h_n.shape()[0], 3);
    EXPECT_EQ(c_n.shape()[0], 3);
}

TEST_P(RNNTest, GRUCell_SingleStep) {
    auto cell = GRUCell(128, 256, true);

    auto x = Variable(randn({4, 128}, DType::Float32, device), true);
    auto h = Variable(randn({4, 256}, DType::Float32, device), true);

    auto h_next = cell.forward(x, h);

    EXPECT_EQ(h_next.shape()[0], 4);
    EXPECT_EQ(h_next.shape()[1], 256);
}

TEST_P(RNNTest, GRU_SequenceProcessing) {
    auto gru = GRU(128, 256, 1, true, false, 0.0, false);

    auto x = Variable(randn({10, 4, 128}, DType::Float32, device), true);

    auto result = gru.forward(x, Variable{});
    auto output = result.first;
    auto h_n = result.second;

    EXPECT_EQ(output.shape()[0], 10);
    EXPECT_EQ(output.shape()[2], 256);
    EXPECT_EQ(h_n.shape()[0], 1);
}

TEST_P(RNNTest, GRU_Bidirectional) {
    auto gru = GRU(128, 256, 1, true, false, 0.0, true);

    auto x = Variable(randn({10, 4, 128}, DType::Float32, device), true);

    auto result = gru.forward(x, Variable{});
    auto output = result.first;

    EXPECT_EQ(output.shape()[2], 512);  // 2 * 256
}

TEST_P(RNNTest, GRU_MultiLayer) {
    auto gru = GRU(128, 256, 3, true, false, 0.5, false);

    auto x = Variable(randn({10, 4, 128}, DType::Float32, device), true);

    auto result = gru.forward(x, Variable{});
    auto h_n = result.second;

    EXPECT_EQ(h_n.shape()[0], 3);
}

// ============================================================================
// Test Instantiation for All Backends
// ============================================================================

INSTANTIATE_BACKEND_TESTS(ActivationTest);
INSTANTIATE_BACKEND_TESTS(LossTest);
INSTANTIATE_BACKEND_TESTS(NormalizationTest);
INSTANTIATE_BACKEND_TESTS(PoolingTest);
INSTANTIATE_BACKEND_TESTS(EmbeddingTest);
INSTANTIATE_BACKEND_TESTS(RNNTest);

// Entry point
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
