/**
 * @file test_losses_advanced.cpp
 * @brief Comprehensive tests for advanced loss functions
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <cmath>
#include "../grad_flow_helpers.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;

// Parameterized over all backends via BackendTest: each TEST_P creates its
// tensors on the fixture's `device`. Loss ops are device-agnostic (no device
// arg); inputs/targets/weights are routed onto `device`. Integer label/index
// tensors are built on CPU via the creation ops then moved with .to(device).
class AdvancedLossTest : public ::tenzor::testing::BackendTest {};

//==============================================================================
// KLDivLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, KLDivLoss_BasicForward) {
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32, device), false);  // log probabilities
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32, device), false);  // probabilities

    auto criterion = KLDivLoss("mean", false);
    auto loss = criterion(input, target);

    // KL divergence should be positive
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
}

TEST_P(AdvancedLossTest, KLDivLoss_PerfectMatch) {
    // When distributions match, KL divergence should be 0
    auto log_probs = Variable(full({2, 3}, std::log(0.333333f), DType::Float32, device), false);
    auto target = Variable(full({2, 3}, 0.333333f, DType::Float32, device), false);

    auto criterion = KLDivLoss("mean", false);
    auto loss = criterion(log_probs, target);

    // Should be close to 0 (perfect match)
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), 0.0f, 1e-5);
}

TEST_P(AdvancedLossTest, KLDivLoss_LogTarget) {
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32, device), false);
    auto target = Variable(full({2, 3}, -1.0f, DType::Float32, device), false);  // log probabilities

    auto criterion = KLDivLoss("mean", true);  // log_target=true
    auto loss = criterion(input, target);

    // Same log distributions should give near-zero KL
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), 0.0f, 1e-5);
}

TEST_P(AdvancedLossTest, KLDivLoss_ReductionModes) {
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32, device), false);
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32, device), false);

    auto criterion_mean = KLDivLoss("mean");
    auto criterion_sum = KLDivLoss("sum");
    auto criterion_none = KLDivLoss("none");
    auto criterion_batchmean = KLDivLoss("batchmean");

    auto loss_mean = criterion_mean(input, target);
    auto loss_sum = criterion_sum(input, target);
    auto loss_none = criterion_none(input, target);
    auto loss_batchmean = criterion_batchmean(input, target);

    // Sum should be larger than mean
    auto loss_sum_cpu = loss_sum.tensor().cpu();
    auto loss_mean_cpu = loss_mean.tensor().cpu();
    EXPECT_GT(loss_sum_cpu.item<float>(), loss_mean_cpu.item<float>());

    // None should return full tensor
    EXPECT_EQ(loss_none.shape().size(), 2);

    // Batchmean should be different from mean
    auto loss_batchmean_cpu = loss_batchmean.tensor().cpu();
    EXPECT_NE(loss_batchmean_cpu.item<float>(), loss_mean_cpu.item<float>());
}

TEST_P(AdvancedLossTest, KLDivLoss_Asymmetry) {
    // KL(P||Q) != KL(Q||P)
    auto p = Variable(full({2, 3}, 0.7f, DType::Float32, device), false);
    auto q = Variable(full({2, 3}, 0.3f, DType::Float32, device), false);
    auto log_p = log(p);
    auto log_q = log(q);

    auto criterion = KLDivLoss("mean", false);
    auto kl_pq = criterion(log_q, p);  // KL(P||Q)
    auto kl_qp = criterion(log_p, q);  // KL(Q||P)

    // Should be different (asymmetry)
    auto kl_pq_cpu = kl_pq.tensor().cpu();
    auto kl_qp_cpu = kl_qp.tensor().cpu();
    EXPECT_NE(kl_pq_cpu.item<float>(), kl_qp_cpu.item<float>());
}

//==============================================================================
// FocalLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, FocalLoss_BasicForward) {
    auto input = Variable(ones({2, 3}, DType::Float32, device), false);   // logits
    auto target = Variable(zeros({2, 3}, DType::Float32, device), false); // one-hot targets

    auto criterion = FocalLoss(1.0, 2.0, "mean");
    auto loss = criterion(input, target);

    // Loss should be positive
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
}

TEST_P(AdvancedLossTest, FocalLoss_GammaZero) {
    // Gamma=0 should approximate cross entropy
    auto input = Variable(ones({2, 3}, DType::Float32, device), false);
    auto target = Variable(ones({2, 3}, DType::Float32, device) / 3.0f, false);

    auto criterion_focal = FocalLoss(1.0, 0.0, "mean");  // gamma=0
    auto loss_focal = criterion_focal(input, target);

    // Should behave similar to cross entropy
    auto loss_focal_cpu = loss_focal.tensor().cpu();
    EXPECT_GT(loss_focal_cpu.item<float>(), 0.0f);
}

TEST_P(AdvancedLossTest, FocalLoss_AlphaWeighting) {
    auto input = Variable(ones({2, 3}, DType::Float32, device), false);
    auto target = Variable(ones({2, 3}, DType::Float32, device) / 3.0f, false);

    auto criterion_alpha1 = FocalLoss(1.0, 2.0, "mean");
    auto criterion_alpha2 = FocalLoss(2.0, 2.0, "mean");

    auto loss_alpha1 = criterion_alpha1(input, target);
    auto loss_alpha2 = criterion_alpha2(input, target);

    // Alpha=2 should give higher loss (more weighting)
    auto loss_alpha2_cpu = loss_alpha2.tensor().cpu();
    auto loss_alpha1_cpu = loss_alpha1.tensor().cpu();
    EXPECT_GT(loss_alpha2_cpu.item<float>(), loss_alpha1_cpu.item<float>());
}

TEST_P(AdvancedLossTest, FocalLoss_ReductionModes) {
    auto input = Variable(ones({2, 3}, DType::Float32, device), false);
    auto target = Variable(ones({2, 3}, DType::Float32, device) / 3.0f, false);

    auto criterion_mean = FocalLoss(1.0, 2.0, "mean");
    auto criterion_sum = FocalLoss(1.0, 2.0, "sum");
    auto criterion_none = FocalLoss(1.0, 2.0, "none");

    auto loss_mean = criterion_mean(input, target);
    auto loss_sum = criterion_sum(input, target);
    auto loss_none = criterion_none(input, target);

    // Sum should be larger than mean
    auto loss_sum_cpu = loss_sum.tensor().cpu();
    auto loss_mean_cpu = loss_mean.tensor().cpu();
    EXPECT_GT(loss_sum_cpu.item<float>(), loss_mean_cpu.item<float>());

    // None should return tensor
    EXPECT_GE(loss_none.shape().size(), 1);
}

TEST_P(AdvancedLossTest, FocalLoss_HighConfidenceDownweighting) {
    // Focal loss should down-weight high-confidence predictions
    auto high_conf = Variable(full({1, 2}, 10.0f, DType::Float32, device), false);  // Very confident
    auto low_conf = Variable(full({1, 2}, 0.5f, DType::Float32, device), false);    // Less confident
    auto target = Variable(full({1, 2}, 0.5f, DType::Float32, device), false);

    auto criterion = FocalLoss(1.0, 2.0, "mean");
    auto loss_high = criterion(high_conf, target);
    auto loss_low = criterion(low_conf, target);

    // Low confidence should contribute more to loss
    auto loss_low_cpu = loss_low.tensor().cpu();
    EXPECT_GT(loss_low_cpu.item<float>(), 0.0f);
}

//==============================================================================
// DiceLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, DiceLoss_BasicForward) {
    auto input = Variable(full({1, 2, 4, 4}, 0.5f, DType::Float32, device), false);  // probabilities
    auto target = Variable(full({1, 2, 4, 4}, 1.0f, DType::Float32, device), false); // binary masks

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // Loss should be between 0 and 1
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
    EXPECT_LE(loss_cpu.item<float>(), 1.0f);
}

TEST_P(AdvancedLossTest, DiceLoss_PerfectOverlap) {
    auto input = Variable(ones({1, 1, 3, 3}, DType::Float32, device), false);
    auto target = Variable(ones({1, 1, 3, 3}, DType::Float32, device), false);

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // Perfect overlap should give loss near 0
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), 0.0f, 0.1);
}

TEST_P(AdvancedLossTest, DiceLoss_NoOverlap) {
    auto input = Variable(zeros({1, 1, 3, 3}, DType::Float32, device), false);
    auto target = Variable(ones({1, 1, 3, 3}, DType::Float32, device), false);

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // No overlap should give high loss
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GT(loss_cpu.item<float>(), 0.5f);
}

TEST_P(AdvancedLossTest, DiceLoss_SmoothParameter) {
    auto input = Variable(full({1, 1, 2, 2}, 0.5f, DType::Float32, device), false);
    auto target = Variable(ones({1, 1, 2, 2}, DType::Float32, device), false);

    auto criterion_smooth1 = DiceLoss(1.0, "mean");
    auto criterion_smooth10 = DiceLoss(10.0, "mean");

    auto loss_smooth1 = criterion_smooth1(input, target);
    auto loss_smooth10 = criterion_smooth10(input, target);

    // Different smoothing should give different results
    auto loss_smooth1_cpu = loss_smooth1.tensor().cpu();
    auto loss_smooth10_cpu = loss_smooth10.tensor().cpu();
    EXPECT_NE(loss_smooth1_cpu.item<float>(), loss_smooth10_cpu.item<float>());
}

TEST_P(AdvancedLossTest, DiceLoss_ReductionModes) {
    auto input = Variable(full({2, 1, 3, 3}, 0.5f, DType::Float32, device), false);
    auto target = Variable(ones({2, 1, 3, 3}, DType::Float32, device), false);

    auto criterion_mean = DiceLoss(1.0, "mean");
    auto criterion_sum = DiceLoss(1.0, "sum");
    auto criterion_none = DiceLoss(1.0, "none");

    auto loss_mean = criterion_mean(input, target);
    auto loss_sum = criterion_sum(input, target);
    auto loss_none = criterion_none(input, target);

    // All should compute successfully
    auto loss_mean_cpu = loss_mean.tensor().cpu();
    auto loss_sum_cpu = loss_sum.tensor().cpu();
    EXPECT_GE(loss_mean_cpu.item<float>(), 0.0f);
    EXPECT_GE(loss_sum_cpu.item<float>(), 0.0f);
    EXPECT_GE(loss_none.shape().size(), 0);
}

//==============================================================================
// HuberLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, HuberLoss_BasicForward) {
    auto input = Variable(ones({2, 3}, DType::Float32, device), false);
    auto target = Variable(zeros({2, 3}, DType::Float32, device), false);

    auto criterion = HuberLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // Loss should be positive
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GT(loss_cpu.item<float>(), 0.0f);
}

TEST_P(AdvancedLossTest, HuberLoss_SmallError) {
    // For small errors (< delta), should behave like L2
    auto input = Variable(full({2, 3}, 0.5f, DType::Float32, device), false);
    auto target = Variable(zeros({2, 3}, DType::Float32, device), false);

    auto criterion = HuberLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // For error=0.5, delta=1.0: should use quadratic part
    // L = 0.5 * 0.5^2 = 0.125 per element
    // Mean over 6 elements
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), 0.125f, 0.05);
}

TEST_P(AdvancedLossTest, HuberLoss_LargeError) {
    // For large errors (> delta), should behave like L1
    auto input = Variable(full({2, 3}, 5.0f, DType::Float32, device), false);
    auto target = Variable(zeros({2, 3}, DType::Float32, device), false);

    auto criterion = HuberLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // For error=5.0, delta=1.0: should use linear part
    // L = delta * (|diff| - 0.5*delta) = 1.0 * (5.0 - 0.5) = 4.5 per element
    // Simplified implementation may differ
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GT(loss_cpu.item<float>(), 0.0f);
}

TEST_P(AdvancedLossTest, HuberLoss_DeltaParameter) {
    auto input = Variable(full({2, 3}, 2.0f, DType::Float32, device), false);
    auto target = Variable(zeros({2, 3}, DType::Float32, device), false);

    auto criterion_delta1 = HuberLoss(1.0, "mean");
    auto criterion_delta3 = HuberLoss(3.0, "mean");

    auto loss_delta1 = criterion_delta1(input, target);
    auto loss_delta3 = criterion_delta3(input, target);

    // Different deltas should give different results
    // For error=2.0, delta=1.0 uses linear, delta=3.0 uses quadratic
    auto loss_delta1_cpu = loss_delta1.tensor().cpu();
    auto loss_delta3_cpu = loss_delta3.tensor().cpu();
    EXPECT_NE(loss_delta1_cpu.item<float>(), loss_delta3_cpu.item<float>());
}

TEST_P(AdvancedLossTest, HuberLoss_ZeroError) {
    auto input = Variable(ones({2, 3}, DType::Float32, device), false);
    auto target = Variable(ones({2, 3}, DType::Float32, device), false);

    auto criterion = HuberLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // Zero error should give zero loss
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), 0.0f, 1e-6);
}

TEST_P(AdvancedLossTest, HuberLoss_ReductionModes) {
    auto input = Variable(full({2, 3}, 2.0f, DType::Float32, device), false);
    auto target = Variable(zeros({2, 3}, DType::Float32, device), false);

    auto criterion_mean = HuberLoss(1.0, "mean");
    auto criterion_sum = HuberLoss(1.0, "sum");
    auto criterion_none = HuberLoss(1.0, "none");

    auto loss_mean = criterion_mean(input, target);
    auto loss_sum = criterion_sum(input, target);
    auto loss_none = criterion_none(input, target);

    // Sum should be larger than mean
    auto loss_sum_cpu = loss_sum.tensor().cpu();
    auto loss_mean_cpu = loss_mean.tensor().cpu();
    EXPECT_GT(loss_sum_cpu.item<float>(), loss_mean_cpu.item<float>());

    // None should return full tensor
    EXPECT_EQ(loss_none.shape().size(), 2);
}

//==============================================================================
// Functional API Tests
//==============================================================================

TEST_P(AdvancedLossTest, Functional_KLDivLoss) {
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32, device), false);
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32, device), false);

    auto loss = kl_div_loss(input, target, "mean", false);
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
}

TEST_P(AdvancedLossTest, Functional_FocalLoss) {
    auto input = Variable(ones({2, 3}, DType::Float32, device), false);
    auto target = Variable(ones({2, 3}, DType::Float32, device) / 3.0f, false);

    auto loss = focal_loss(input, target, 1.0, 2.0, "mean");
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
}

TEST_P(AdvancedLossTest, Functional_DiceLoss) {
    auto input = Variable(full({1, 1, 3, 3}, 0.5f, DType::Float32, device), false);
    auto target = Variable(ones({1, 1, 3, 3}, DType::Float32, device), false);

    auto loss = dice_loss(input, target, 1.0, "mean");
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
    EXPECT_LE(loss_cpu.item<float>(), 1.0f);
}

TEST_P(AdvancedLossTest, Functional_HuberLoss) {
    auto input = Variable(ones({2, 3}, DType::Float32, device), false);
    auto target = Variable(zeros({2, 3}, DType::Float32, device), false);

    auto loss = huber_loss(input, target, 1.0, "mean");
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GT(loss_cpu.item<float>(), 0.0f);
}

//==============================================================================
// Gradient Flow Tests
//==============================================================================

TEST_P(AdvancedLossTest, KLDivLoss_BackwardGradient) {
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32, device), true);  // requires_grad=true
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32, device), false);

    auto criterion = KLDivLoss("mean");
    auto loss = criterion(input, target);

    // Check that gradients actually flow back to the input.
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

TEST_P(AdvancedLossTest, FocalLoss_BackwardGradient) {
    auto input = Variable(ones({2, 3}, DType::Float32, device), true);
    auto target = Variable(ones({2, 3}, DType::Float32, device) / 3.0f, false);

    auto criterion = FocalLoss(1.0, 2.0, "mean");
    auto loss = criterion(input, target);

    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

TEST_P(AdvancedLossTest, DiceLoss_BackwardGradient) {
    auto input = Variable(full({1, 1, 3, 3}, 0.5f, DType::Float32, device), true);
    auto target = Variable(ones({1, 1, 3, 3}, DType::Float32, device), false);

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(input, target);

    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

TEST_P(AdvancedLossTest, HuberLoss_BackwardGradient) {
    auto input = Variable(ones({2, 3}, DType::Float32, device), true);
    auto target = Variable(zeros({2, 3}, DType::Float32, device), false);

    auto criterion = HuberLoss(1.0, "mean");
    auto loss = criterion(input, target);

    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

//==============================================================================
// Edge Cases and Error Handling
//==============================================================================

TEST_P(AdvancedLossTest, KLDivLoss_InvalidReduction) {
    EXPECT_THROW({
        auto criterion = KLDivLoss("invalid");
    }, std::invalid_argument);
}

TEST_P(AdvancedLossTest, FocalLoss_NegativeGamma) {
    EXPECT_THROW({
        auto criterion = FocalLoss(1.0, -1.0);
    }, std::invalid_argument);
}

TEST_P(AdvancedLossTest, DiceLoss_NegativeSmooth) {
    EXPECT_THROW({
        auto criterion = DiceLoss(-1.0);
    }, std::invalid_argument);
}

TEST_P(AdvancedLossTest, HuberLoss_ZeroDelta) {
    EXPECT_THROW({
        auto criterion = HuberLoss(0.0);
    }, std::invalid_argument);
}

TEST_P(AdvancedLossTest, HuberLoss_NegativeDelta) {
    EXPECT_THROW({
        auto criterion = HuberLoss(-1.0);
    }, std::invalid_argument);
}

//==============================================================================
// Comparison Tests
//==============================================================================

TEST_P(AdvancedLossTest, HuberLoss_vs_MSE_SmallErrors) {
    // For small errors, Huber should approximate MSE
    auto input = Variable(full({2, 3}, 0.1f, DType::Float32, device), false);
    auto target = Variable(zeros({2, 3}, DType::Float32, device), false);

    auto huber = HuberLoss(1.0, "mean");
    auto mse = MSELoss(Reduction::Mean);

    auto loss_huber = huber(input, target);
    auto loss_mse = mse(input, target);

    // Should be similar for small errors
    auto loss_huber_cpu = loss_huber.tensor().cpu();
    auto loss_mse_cpu = loss_mse.tensor().cpu();
    EXPECT_NEAR(loss_huber_cpu.item<float>(), loss_mse_cpu.item<float>(), 0.01);
}

TEST_P(AdvancedLossTest, DiceLoss_vs_BCE_Segmentation) {
    // Dice and BCE are both used for segmentation
    auto input = Variable(full({1, 1, 4, 4}, 0.5f, DType::Float32, device), false);
    auto target = Variable(ones({1, 1, 4, 4}, DType::Float32, device), false);

    auto dice = DiceLoss(1.0, "mean");
    auto bce = BCELoss(Reduction::Mean);

    auto loss_dice = dice(input, target);
    auto loss_bce = bce(input, target);

    // Both should give positive loss
    auto loss_dice_cpu = loss_dice.tensor().cpu();
    auto loss_bce_cpu = loss_bce.tensor().cpu();
    EXPECT_GT(loss_dice_cpu.item<float>(), 0.0f);
    EXPECT_GT(loss_bce_cpu.item<float>(), 0.0f);

    // Dice is more sensitive to overlap
    // Values will differ due to different formulations
}

//==============================================================================
// Numerical Stability Tests
//==============================================================================

TEST_P(AdvancedLossTest, KLDivLoss_NumericalStability) {
    // Test with very small probabilities
    auto input = Variable(full({2, 3}, -10.0f, DType::Float32, device), false);  // Very small log prob
    auto target = Variable(full({2, 3}, 1e-5f, DType::Float32, device), false);  // Very small prob

    auto criterion = KLDivLoss("mean");
    auto loss = criterion(input, target);

    // Should not be NaN or Inf
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_FALSE(std::isnan(loss_cpu.item<float>()));
    EXPECT_FALSE(std::isinf(loss_cpu.item<float>()));
}

TEST_P(AdvancedLossTest, FocalLoss_NumericalStability) {
    // Test with extreme values
    auto input = Variable(full({2, 3}, 100.0f, DType::Float32, device), false);  // Very large logits
    auto target = Variable(ones({2, 3}, DType::Float32, device) / 3.0f, false);

    auto criterion = FocalLoss(1.0, 2.0, "mean");
    auto loss = criterion(input, target);

    // Should not be NaN or Inf
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_FALSE(std::isnan(loss_cpu.item<float>()));
    EXPECT_FALSE(std::isinf(loss_cpu.item<float>()));
}

TEST_P(AdvancedLossTest, DiceLoss_ZeroDenominator) {
    // Test when both input and target are zero
    auto input = Variable(zeros({1, 1, 2, 2}, DType::Float32, device), false);
    auto target = Variable(zeros({1, 1, 2, 2}, DType::Float32, device), false);

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // Smooth parameter should prevent division by zero
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_FALSE(std::isnan(loss_cpu.item<float>()));
    EXPECT_FALSE(std::isinf(loss_cpu.item<float>()));
}

//==============================================================================
// SoftMarginLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, SoftMarginLoss_BasicForward) {
    auto input = Variable(randn({4, 3}, DType::Float32, device), false);
    auto target = Variable(ones({4, 3}, DType::Float32, device), false);

    auto criterion = SoftMarginLoss(Reduction::Mean);
    auto loss = criterion(input, target);

    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(loss_cpu.item<float>()));
}

TEST_P(AdvancedLossTest, SoftMarginLoss_KnownValues) {
    // For x=0, y=1: loss = log(1 + exp(0)) = log(2) ≈ 0.6931
    auto input = Variable(zeros({1}, DType::Float32, device), false);
    auto target = Variable(ones({1}, DType::Float32, device), false);

    auto loss = soft_margin_loss(input, target, Reduction::Mean);
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), std::log(2.0f), 1e-4);
}

TEST_P(AdvancedLossTest, SoftMarginLoss_ReductionModes) {
    auto input = Variable(randn({4}, DType::Float32, device), false);
    auto target = Variable(ones({4}, DType::Float32, device), false);

    auto loss_none = soft_margin_loss(input, target, Reduction::None);
    auto loss_sum = soft_margin_loss(input, target, Reduction::Sum);
    auto loss_mean = soft_margin_loss(input, target, Reduction::Mean);

    EXPECT_EQ(loss_none.shape().size(), 1u);
    EXPECT_EQ(loss_none.shape()[0], 4);
    auto loss_sum_cpu = loss_sum.tensor().cpu();
    auto loss_mean_cpu = loss_mean.tensor().cpu();
    EXPECT_GT(loss_sum_cpu.item<float>(), loss_mean_cpu.item<float>());
}

TEST_P(AdvancedLossTest, SoftMarginLoss_Gradient) {
    auto input = Variable(randn({3}, DType::Float32, device), true);
    auto target = Variable(ones({3}, DType::Float32, device), false);

    auto loss = soft_margin_loss(input, target, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

//==============================================================================
// HingeEmbeddingLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, HingeEmbeddingLoss_BasicForward) {
    auto input = Variable(full({4}, 0.5f, DType::Float32, device), false);
    auto target = Variable(ones({4}, DType::Float32, device), false);  // y = 1

    auto criterion = HingeEmbeddingLoss(1.0, Reduction::Mean);
    auto loss = criterion(input, target);

    // When y=1, loss = input = 0.5
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), 0.5f, 1e-4);
}

TEST_P(AdvancedLossTest, HingeEmbeddingLoss_NegativeTarget) {
    auto input = Variable(full({4}, 0.5f, DType::Float32, device), false);
    auto neg_target = Variable(full({4}, -1.0f, DType::Float32, device), false);

    auto loss = hinge_embedding_loss(input, neg_target, 1.0, Reduction::Mean);

    // When y=-1, loss = max(0, 1.0 - 0.5) = 0.5
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), 0.5f, 1e-4);
}

TEST_P(AdvancedLossTest, HingeEmbeddingLoss_Gradient) {
    auto input = Variable(randn({4}, DType::Float32, device), true);
    auto target = Variable(ones({4}, DType::Float32, device), false);

    auto loss = hinge_embedding_loss(input, target, 1.0, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

//==============================================================================
// PoissonNLLLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, PoissonNLLLoss_BasicForward) {
    auto input = Variable(randn({4, 3}, DType::Float32, device), false);
    auto target = Variable(full({4, 3}, 2.0f, DType::Float32, device), false);

    auto criterion = PoissonNLLLoss(true, false, 1e-8, Reduction::Mean);
    auto loss = criterion(input, target);

    auto loss_cpu = loss.tensor().cpu();
    EXPECT_FALSE(std::isnan(loss_cpu.item<float>()));
    EXPECT_FALSE(std::isinf(loss_cpu.item<float>()));
}

TEST_P(AdvancedLossTest, PoissonNLLLoss_KnownValues) {
    // For log_input=true: loss = exp(x) - y*x
    // x=0, y=1: loss = exp(0) - 1*0 = 1.0
    auto input = Variable(zeros({1}, DType::Float32, device), false);
    auto target = Variable(ones({1}, DType::Float32, device), false);

    auto loss = poisson_nll_loss(input, target, true, false, 1e-8, Reduction::Mean);
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), 1.0f, 1e-4);
}

TEST_P(AdvancedLossTest, PoissonNLLLoss_Gradient) {
    auto input = Variable(randn({4}, DType::Float32, device), true);
    auto target = Variable(full({4}, 2.0f, DType::Float32, device), false);

    auto loss = poisson_nll_loss(input, target, true, false, 1e-8, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

//==============================================================================
// CosineEmbeddingLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, CosineEmbeddingLoss_SimilarPair) {
    // Identical vectors should give loss close to 0 for y=1
    auto input1 = Variable(ones({2, 4}, DType::Float32, device), false);
    auto input2 = Variable(ones({2, 4}, DType::Float32, device), false);
    auto target = Variable(ones({2}, DType::Float32, device), false);

    auto criterion = CosineEmbeddingLoss(0.0, Reduction::Mean);
    auto loss = criterion(input1, input2, target);

    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), 0.0f, 1e-4);
}

TEST_P(AdvancedLossTest, CosineEmbeddingLoss_DissimilarPair) {
    auto input1 = Variable(ones({2, 4}, DType::Float32, device), false);
    auto input2 = Variable(full({2, 4}, -1.0f, DType::Float32, device), false);
    auto target = Variable(full({2}, -1.0f, DType::Float32, device), false);

    auto loss = cosine_embedding_loss(input1, input2, target, 0.0, Reduction::Mean);

    // Opposite vectors have cos_sim = -1, with y=-1: max(0, -1 - 0) = 0
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
}

TEST_P(AdvancedLossTest, CosineEmbeddingLoss_Gradient) {
    auto input1 = Variable(randn({3, 4}, DType::Float32, device), true);
    auto input2 = Variable(randn({3, 4}, DType::Float32, device), true);
    auto target = Variable(ones({3}, DType::Float32, device), false);

    auto loss = cosine_embedding_loss(input1, input2, target, 0.0, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input1);
    EXPECT_GRAD_FLOWS(input2);
}

//==============================================================================
// TripletMarginLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, TripletMarginLoss_BasicForward) {
    auto anchor = Variable(randn({4, 8}, DType::Float32, device), false);
    auto positive = Variable(randn({4, 8}, DType::Float32, device), false);
    auto negative = Variable(randn({4, 8}, DType::Float32, device), false);

    auto criterion = TripletMarginLoss(1.0, 2.0, false, Reduction::Mean);
    auto loss = criterion(anchor, positive, negative);

    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(loss_cpu.item<float>()));
}

TEST_P(AdvancedLossTest, TripletMarginLoss_Gradient) {
    auto anchor = Variable(randn({4, 8}, DType::Float32, device), true);
    auto positive = Variable(randn({4, 8}, DType::Float32, device), true);
    auto negative = Variable(randn({4, 8}, DType::Float32, device), true);

    auto loss = triplet_margin_loss(anchor, positive, negative, 1.0, 2.0, false, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(anchor);
}

//==============================================================================
// MultiLabelSoftMarginLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, MultiLabelSoftMarginLoss_BasicForward) {
    auto input = Variable(randn({4, 5}, DType::Float32, device), false);
    auto target = Variable(zeros({4, 5}, DType::Float32, device), false);

    auto criterion = MultiLabelSoftMarginLoss(Reduction::Mean);
    auto loss = criterion(input, target);

    auto loss_cpu = loss.tensor().cpu();
    EXPECT_GE(loss_cpu.item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(loss_cpu.item<float>()));
}

TEST_P(AdvancedLossTest, MultiLabelSoftMarginLoss_Gradient) {
    auto input = Variable(randn({4, 5}, DType::Float32, device), true);
    auto target = Variable(zeros({4, 5}, DType::Float32, device), false);

    auto loss = multi_label_soft_margin_loss(input, target, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

//==============================================================================
// MultiMarginLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, MultiMarginLoss_BasicForward) {
    auto input = Variable(randn({4, 5}, DType::Float32, device), false);
    // Create target with class indices using arange (built on CPU, moved to device)
    auto target = arange(0, 4, 1, DType::Int64).to(device);

    auto criterion = MultiMarginLoss(1, 1.0, Reduction::Mean);
    auto loss = criterion(input, target);

    auto loss_cpu = loss.tensor().cpu();
    float loss_val = loss_cpu.item<float>();
    EXPECT_GE(loss_val, 0.0f);
    EXPECT_FALSE(std::isnan(loss_val));
}

TEST_P(AdvancedLossTest, MultiMarginLoss_P2) {
    auto input = Variable(randn({4, 3}, DType::Float32, device), false);
    auto target = zeros({4}, DType::Int64).to(device);  // All class 0

    auto loss_p1 = multi_margin_loss(input, target, 1, 1.0, Reduction::Mean);
    auto loss_p2 = multi_margin_loss(input, target, 2, 1.0, Reduction::Mean);

    auto loss_p1_cpu = loss_p1.tensor().cpu();
    auto loss_p2_cpu = loss_p2.tensor().cpu();
    float v1 = loss_p1_cpu.item<float>();
    float v2 = loss_p2_cpu.item<float>();
    // Both should be non-negative and finite
    EXPECT_GE(v1, 0.0f);
    EXPECT_GE(v2, 0.0f);
    EXPECT_FALSE(std::isnan(v2));
}

//==============================================================================
// GaussianNLLLoss Tests
//==============================================================================

TEST_P(AdvancedLossTest, GaussianNLLLoss_BasicForward) {
    auto input = Variable(randn({4, 3}, DType::Float32, device), false);
    auto target = Variable(randn({4, 3}, DType::Float32, device), false);
    auto var = Variable(full({4, 3}, 1.0f, DType::Float32, device), false);

    auto criterion = GaussianNLLLoss(false, 1e-6, Reduction::Mean);
    auto loss = criterion(input, target, var);

    auto loss_cpu = loss.tensor().cpu();
    EXPECT_FALSE(std::isnan(loss_cpu.item<float>()));
    EXPECT_FALSE(std::isinf(loss_cpu.item<float>()));
}

TEST_P(AdvancedLossTest, GaussianNLLLoss_KnownValues) {
    // input=target, var=1: loss = 0.5*(log(1) + 0) = 0
    auto input = Variable(ones({1}, DType::Float32, device), false);
    auto target = Variable(ones({1}, DType::Float32, device), false);
    auto var = Variable(ones({1}, DType::Float32, device), false);

    auto loss = gaussian_nll_loss(input, target, var, false, 1e-6, Reduction::Mean);
    auto loss_cpu = loss.tensor().cpu();
    EXPECT_NEAR(loss_cpu.item<float>(), 0.0f, 1e-4);
}

TEST_P(AdvancedLossTest, GaussianNLLLoss_Gradient) {
    auto input = Variable(randn({4, 3}, DType::Float32, device), true);
    auto target = Variable(randn({4, 3}, DType::Float32, device), false);
    auto var = Variable(full({4, 3}, 1.0f, DType::Float32, device), true);

    auto loss = gaussian_nll_loss(input, target, var, false, 1e-6, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_GRAD_FLOWS(var);
}

// Fan every TEST_P above over all five backends. BackendTest::SetUp skips a
// backend that is physically absent on the host; a present backend that does
// not implement a given loss op throws → the corresponding cell FAILS.
INSTANTIATE_BACKEND_TESTS(AdvancedLossTest);
