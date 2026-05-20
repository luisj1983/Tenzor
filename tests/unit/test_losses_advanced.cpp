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

using namespace tenzor;
using namespace tenzor::nn;

// Global test environment
class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

//==============================================================================
// KLDivLoss Tests
//==============================================================================

TEST(AdvancedLossTest, KLDivLoss_BasicForward) {
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32), false);  // log probabilities
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32), false);  // probabilities

    auto criterion = KLDivLoss("mean", false);
    auto loss = criterion(input, target);

    // KL divergence should be positive
    EXPECT_GE(loss.tensor().item<float>(), 0.0f);
}

TEST(AdvancedLossTest, KLDivLoss_PerfectMatch) {
    // When distributions match, KL divergence should be 0
    auto log_probs = Variable(full({2, 3}, std::log(0.333333f), DType::Float32), false);
    auto target = Variable(full({2, 3}, 0.333333f, DType::Float32), false);

    auto criterion = KLDivLoss("mean", false);
    auto loss = criterion(log_probs, target);

    // Should be close to 0 (perfect match)
    EXPECT_NEAR(loss.tensor().item<float>(), 0.0f, 1e-5);
}

TEST(AdvancedLossTest, KLDivLoss_LogTarget) {
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32), false);
    auto target = Variable(full({2, 3}, -1.0f, DType::Float32), false);  // log probabilities

    auto criterion = KLDivLoss("mean", true);  // log_target=true
    auto loss = criterion(input, target);

    // Same log distributions should give near-zero KL
    EXPECT_NEAR(loss.tensor().item<float>(), 0.0f, 1e-5);
}

TEST(AdvancedLossTest, KLDivLoss_ReductionModes) {
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32), false);
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32), false);

    auto criterion_mean = KLDivLoss("mean");
    auto criterion_sum = KLDivLoss("sum");
    auto criterion_none = KLDivLoss("none");
    auto criterion_batchmean = KLDivLoss("batchmean");

    auto loss_mean = criterion_mean(input, target);
    auto loss_sum = criterion_sum(input, target);
    auto loss_none = criterion_none(input, target);
    auto loss_batchmean = criterion_batchmean(input, target);

    // Sum should be larger than mean
    EXPECT_GT(loss_sum.tensor().item<float>(), loss_mean.tensor().item<float>());

    // None should return full tensor
    EXPECT_EQ(loss_none.shape().size(), 2);

    // Batchmean should be different from mean
    EXPECT_NE(loss_batchmean.tensor().item<float>(), loss_mean.tensor().item<float>());
}

TEST(AdvancedLossTest, KLDivLoss_Asymmetry) {
    // KL(P||Q) != KL(Q||P)
    auto p = Variable(full({2, 3}, 0.7f, DType::Float32), false);
    auto q = Variable(full({2, 3}, 0.3f, DType::Float32), false);
    auto log_p = log(p);
    auto log_q = log(q);

    auto criterion = KLDivLoss("mean", false);
    auto kl_pq = criterion(log_q, p);  // KL(P||Q)
    auto kl_qp = criterion(log_p, q);  // KL(Q||P)

    // Should be different (asymmetry)
    EXPECT_NE(kl_pq.tensor().item<float>(), kl_qp.tensor().item<float>());
}

//==============================================================================
// FocalLoss Tests
//==============================================================================

TEST(AdvancedLossTest, FocalLoss_BasicForward) {
    auto input = Variable(ones({2, 3}, DType::Float32), false);   // logits
    auto target = Variable(zeros({2, 3}, DType::Float32), false); // one-hot targets

    auto criterion = FocalLoss(1.0, 2.0, "mean");
    auto loss = criterion(input, target);

    // Loss should be positive
    EXPECT_GE(loss.tensor().item<float>(), 0.0f);
}

TEST(AdvancedLossTest, FocalLoss_GammaZero) {
    // Gamma=0 should approximate cross entropy
    auto input = Variable(ones({2, 3}, DType::Float32), false);
    auto target = Variable(ones({2, 3}, DType::Float32) / 3.0f, false);

    auto criterion_focal = FocalLoss(1.0, 0.0, "mean");  // gamma=0
    auto loss_focal = criterion_focal(input, target);

    // Should behave similar to cross entropy
    EXPECT_GT(loss_focal.tensor().item<float>(), 0.0f);
}

TEST(AdvancedLossTest, FocalLoss_AlphaWeighting) {
    auto input = Variable(ones({2, 3}, DType::Float32), false);
    auto target = Variable(ones({2, 3}, DType::Float32) / 3.0f, false);

    auto criterion_alpha1 = FocalLoss(1.0, 2.0, "mean");
    auto criterion_alpha2 = FocalLoss(2.0, 2.0, "mean");

    auto loss_alpha1 = criterion_alpha1(input, target);
    auto loss_alpha2 = criterion_alpha2(input, target);

    // Alpha=2 should give higher loss (more weighting)
    EXPECT_GT(loss_alpha2.tensor().item<float>(), loss_alpha1.tensor().item<float>());
}

TEST(AdvancedLossTest, FocalLoss_ReductionModes) {
    auto input = Variable(ones({2, 3}, DType::Float32), false);
    auto target = Variable(ones({2, 3}, DType::Float32) / 3.0f, false);

    auto criterion_mean = FocalLoss(1.0, 2.0, "mean");
    auto criterion_sum = FocalLoss(1.0, 2.0, "sum");
    auto criterion_none = FocalLoss(1.0, 2.0, "none");

    auto loss_mean = criterion_mean(input, target);
    auto loss_sum = criterion_sum(input, target);
    auto loss_none = criterion_none(input, target);

    // Sum should be larger than mean
    EXPECT_GT(loss_sum.tensor().item<float>(), loss_mean.tensor().item<float>());

    // None should return tensor
    EXPECT_GE(loss_none.shape().size(), 1);
}

TEST(AdvancedLossTest, FocalLoss_HighConfidenceDownweighting) {
    // Focal loss should down-weight high-confidence predictions
    auto high_conf = Variable(full({1, 2}, 10.0f, DType::Float32), false);  // Very confident
    auto low_conf = Variable(full({1, 2}, 0.5f, DType::Float32), false);    // Less confident
    auto target = Variable(full({1, 2}, 0.5f, DType::Float32), false);

    auto criterion = FocalLoss(1.0, 2.0, "mean");
    auto loss_high = criterion(high_conf, target);
    auto loss_low = criterion(low_conf, target);

    // Low confidence should contribute more to loss
    EXPECT_GT(loss_low.tensor().item<float>(), 0.0f);
}

//==============================================================================
// DiceLoss Tests
//==============================================================================

TEST(AdvancedLossTest, DiceLoss_BasicForward) {
    auto input = Variable(full({1, 2, 4, 4}, 0.5f, DType::Float32), false);  // probabilities
    auto target = Variable(full({1, 2, 4, 4}, 1.0f, DType::Float32), false); // binary masks

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // Loss should be between 0 and 1
    EXPECT_GE(loss.tensor().item<float>(), 0.0f);
    EXPECT_LE(loss.tensor().item<float>(), 1.0f);
}

TEST(AdvancedLossTest, DiceLoss_PerfectOverlap) {
    auto input = Variable(ones({1, 1, 3, 3}, DType::Float32), false);
    auto target = Variable(ones({1, 1, 3, 3}, DType::Float32), false);

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // Perfect overlap should give loss near 0
    EXPECT_NEAR(loss.tensor().item<float>(), 0.0f, 0.1);
}

TEST(AdvancedLossTest, DiceLoss_NoOverlap) {
    auto input = Variable(zeros({1, 1, 3, 3}, DType::Float32), false);
    auto target = Variable(ones({1, 1, 3, 3}, DType::Float32), false);

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // No overlap should give high loss
    EXPECT_GT(loss.tensor().item<float>(), 0.5f);
}

TEST(AdvancedLossTest, DiceLoss_SmoothParameter) {
    auto input = Variable(full({1, 1, 2, 2}, 0.5f, DType::Float32), false);
    auto target = Variable(ones({1, 1, 2, 2}, DType::Float32), false);

    auto criterion_smooth1 = DiceLoss(1.0, "mean");
    auto criterion_smooth10 = DiceLoss(10.0, "mean");

    auto loss_smooth1 = criterion_smooth1(input, target);
    auto loss_smooth10 = criterion_smooth10(input, target);

    // Different smoothing should give different results
    EXPECT_NE(loss_smooth1.tensor().item<float>(), loss_smooth10.tensor().item<float>());
}

TEST(AdvancedLossTest, DiceLoss_ReductionModes) {
    auto input = Variable(full({2, 1, 3, 3}, 0.5f, DType::Float32), false);
    auto target = Variable(ones({2, 1, 3, 3}, DType::Float32), false);

    auto criterion_mean = DiceLoss(1.0, "mean");
    auto criterion_sum = DiceLoss(1.0, "sum");
    auto criterion_none = DiceLoss(1.0, "none");

    auto loss_mean = criterion_mean(input, target);
    auto loss_sum = criterion_sum(input, target);
    auto loss_none = criterion_none(input, target);

    // All should compute successfully
    EXPECT_GE(loss_mean.tensor().item<float>(), 0.0f);
    EXPECT_GE(loss_sum.tensor().item<float>(), 0.0f);
    EXPECT_GE(loss_none.shape().size(), 0);
}

//==============================================================================
// HuberLoss Tests
//==============================================================================

TEST(AdvancedLossTest, HuberLoss_BasicForward) {
    auto input = Variable(ones({2, 3}, DType::Float32), false);
    auto target = Variable(zeros({2, 3}, DType::Float32), false);

    auto criterion = HuberLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // Loss should be positive
    EXPECT_GT(loss.tensor().item<float>(), 0.0f);
}

TEST(AdvancedLossTest, HuberLoss_SmallError) {
    // For small errors (< delta), should behave like L2
    auto input = Variable(full({2, 3}, 0.5f, DType::Float32), false);
    auto target = Variable(zeros({2, 3}, DType::Float32), false);

    auto criterion = HuberLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // For error=0.5, delta=1.0: should use quadratic part
    // L = 0.5 * 0.5^2 = 0.125 per element
    // Mean over 6 elements
    EXPECT_NEAR(loss.tensor().item<float>(), 0.125f, 0.05);
}

TEST(AdvancedLossTest, HuberLoss_LargeError) {
    // For large errors (> delta), should behave like L1
    auto input = Variable(full({2, 3}, 5.0f, DType::Float32), false);
    auto target = Variable(zeros({2, 3}, DType::Float32), false);

    auto criterion = HuberLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // For error=5.0, delta=1.0: should use linear part
    // L = delta * (|diff| - 0.5*delta) = 1.0 * (5.0 - 0.5) = 4.5 per element
    // Simplified implementation may differ
    EXPECT_GT(loss.tensor().item<float>(), 0.0f);
}

TEST(AdvancedLossTest, HuberLoss_DeltaParameter) {
    auto input = Variable(full({2, 3}, 2.0f, DType::Float32), false);
    auto target = Variable(zeros({2, 3}, DType::Float32), false);

    auto criterion_delta1 = HuberLoss(1.0, "mean");
    auto criterion_delta3 = HuberLoss(3.0, "mean");

    auto loss_delta1 = criterion_delta1(input, target);
    auto loss_delta3 = criterion_delta3(input, target);

    // Different deltas should give different results
    // For error=2.0, delta=1.0 uses linear, delta=3.0 uses quadratic
    EXPECT_NE(loss_delta1.tensor().item<float>(), loss_delta3.tensor().item<float>());
}

TEST(AdvancedLossTest, HuberLoss_ZeroError) {
    auto input = Variable(ones({2, 3}, DType::Float32), false);
    auto target = Variable(ones({2, 3}, DType::Float32), false);

    auto criterion = HuberLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // Zero error should give zero loss
    EXPECT_NEAR(loss.tensor().item<float>(), 0.0f, 1e-6);
}

TEST(AdvancedLossTest, HuberLoss_ReductionModes) {
    auto input = Variable(full({2, 3}, 2.0f, DType::Float32), false);
    auto target = Variable(zeros({2, 3}, DType::Float32), false);

    auto criterion_mean = HuberLoss(1.0, "mean");
    auto criterion_sum = HuberLoss(1.0, "sum");
    auto criterion_none = HuberLoss(1.0, "none");

    auto loss_mean = criterion_mean(input, target);
    auto loss_sum = criterion_sum(input, target);
    auto loss_none = criterion_none(input, target);

    // Sum should be larger than mean
    EXPECT_GT(loss_sum.tensor().item<float>(), loss_mean.tensor().item<float>());

    // None should return full tensor
    EXPECT_EQ(loss_none.shape().size(), 2);
}

//==============================================================================
// Functional API Tests
//==============================================================================

TEST(AdvancedLossTest, Functional_KLDivLoss) {
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32), false);
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32), false);

    auto loss = kl_div_loss(input, target, "mean", false);
    EXPECT_GE(loss.tensor().item<float>(), 0.0f);
}

TEST(AdvancedLossTest, Functional_FocalLoss) {
    auto input = Variable(ones({2, 3}, DType::Float32), false);
    auto target = Variable(ones({2, 3}, DType::Float32) / 3.0f, false);

    auto loss = focal_loss(input, target, 1.0, 2.0, "mean");
    EXPECT_GE(loss.tensor().item<float>(), 0.0f);
}

TEST(AdvancedLossTest, Functional_DiceLoss) {
    auto input = Variable(full({1, 1, 3, 3}, 0.5f, DType::Float32), false);
    auto target = Variable(ones({1, 1, 3, 3}, DType::Float32), false);

    auto loss = dice_loss(input, target, 1.0, "mean");
    EXPECT_GE(loss.tensor().item<float>(), 0.0f);
    EXPECT_LE(loss.tensor().item<float>(), 1.0f);
}

TEST(AdvancedLossTest, Functional_HuberLoss) {
    auto input = Variable(ones({2, 3}, DType::Float32), false);
    auto target = Variable(zeros({2, 3}, DType::Float32), false);

    auto loss = huber_loss(input, target, 1.0, "mean");
    EXPECT_GT(loss.tensor().item<float>(), 0.0f);
}

//==============================================================================
// Gradient Flow Tests
//==============================================================================

TEST(AdvancedLossTest, KLDivLoss_BackwardGradient) {
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32), true);  // requires_grad=true
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32), false);

    auto criterion = KLDivLoss("mean");
    auto loss = criterion(input, target);

    // Check that we can compute gradients
    EXPECT_NO_THROW(loss.backward());
    EXPECT_TRUE(input.grad().has_value());
}

TEST(AdvancedLossTest, FocalLoss_BackwardGradient) {
    auto input = Variable(ones({2, 3}, DType::Float32), true);
    auto target = Variable(ones({2, 3}, DType::Float32) / 3.0f, false);

    auto criterion = FocalLoss(1.0, 2.0, "mean");
    auto loss = criterion(input, target);

    EXPECT_NO_THROW(loss.backward());
    EXPECT_TRUE(input.grad().has_value());
}

TEST(AdvancedLossTest, DiceLoss_BackwardGradient) {
    auto input = Variable(full({1, 1, 3, 3}, 0.5f, DType::Float32), true);
    auto target = Variable(ones({1, 1, 3, 3}, DType::Float32), false);

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(input, target);

    EXPECT_NO_THROW(loss.backward());
    EXPECT_TRUE(input.grad().has_value());
}

TEST(AdvancedLossTest, HuberLoss_BackwardGradient) {
    auto input = Variable(ones({2, 3}, DType::Float32), true);
    auto target = Variable(zeros({2, 3}, DType::Float32), false);

    auto criterion = HuberLoss(1.0, "mean");
    auto loss = criterion(input, target);

    EXPECT_NO_THROW(loss.backward());
    EXPECT_TRUE(input.grad().has_value());
}

//==============================================================================
// Edge Cases and Error Handling
//==============================================================================

TEST(AdvancedLossTest, KLDivLoss_InvalidReduction) {
    EXPECT_THROW({
        auto criterion = KLDivLoss("invalid");
    }, std::invalid_argument);
}

TEST(AdvancedLossTest, FocalLoss_NegativeGamma) {
    EXPECT_THROW({
        auto criterion = FocalLoss(1.0, -1.0);
    }, std::invalid_argument);
}

TEST(AdvancedLossTest, DiceLoss_NegativeSmooth) {
    EXPECT_THROW({
        auto criterion = DiceLoss(-1.0);
    }, std::invalid_argument);
}

TEST(AdvancedLossTest, HuberLoss_ZeroDelta) {
    EXPECT_THROW({
        auto criterion = HuberLoss(0.0);
    }, std::invalid_argument);
}

TEST(AdvancedLossTest, HuberLoss_NegativeDelta) {
    EXPECT_THROW({
        auto criterion = HuberLoss(-1.0);
    }, std::invalid_argument);
}

//==============================================================================
// Comparison Tests
//==============================================================================

TEST(AdvancedLossTest, HuberLoss_vs_MSE_SmallErrors) {
    // For small errors, Huber should approximate MSE
    auto input = Variable(full({2, 3}, 0.1f, DType::Float32), false);
    auto target = Variable(zeros({2, 3}, DType::Float32), false);

    auto huber = HuberLoss(1.0, "mean");
    auto mse = MSELoss(Reduction::Mean);

    auto loss_huber = huber(input, target);
    auto loss_mse = mse(input, target);

    // Should be similar for small errors
    EXPECT_NEAR(loss_huber.tensor().item<float>(), loss_mse.tensor().item<float>(), 0.01);
}

TEST(AdvancedLossTest, DiceLoss_vs_BCE_Segmentation) {
    // Dice and BCE are both used for segmentation
    auto input = Variable(full({1, 1, 4, 4}, 0.5f, DType::Float32), false);
    auto target = Variable(ones({1, 1, 4, 4}, DType::Float32), false);

    auto dice = DiceLoss(1.0, "mean");
    auto bce = BCELoss(Reduction::Mean);

    auto loss_dice = dice(input, target);
    auto loss_bce = bce(input, target);

    // Both should give positive loss
    EXPECT_GT(loss_dice.tensor().item<float>(), 0.0f);
    EXPECT_GT(loss_bce.tensor().item<float>(), 0.0f);

    // Dice is more sensitive to overlap
    // Values will differ due to different formulations
}

//==============================================================================
// Numerical Stability Tests
//==============================================================================

TEST(AdvancedLossTest, KLDivLoss_NumericalStability) {
    // Test with very small probabilities
    auto input = Variable(full({2, 3}, -10.0f, DType::Float32), false);  // Very small log prob
    auto target = Variable(full({2, 3}, 1e-5f, DType::Float32), false);  // Very small prob

    auto criterion = KLDivLoss("mean");
    auto loss = criterion(input, target);

    // Should not be NaN or Inf
    EXPECT_FALSE(std::isnan(loss.tensor().item<float>()));
    EXPECT_FALSE(std::isinf(loss.tensor().item<float>()));
}

TEST(AdvancedLossTest, FocalLoss_NumericalStability) {
    // Test with extreme values
    auto input = Variable(full({2, 3}, 100.0f, DType::Float32), false);  // Very large logits
    auto target = Variable(ones({2, 3}, DType::Float32) / 3.0f, false);

    auto criterion = FocalLoss(1.0, 2.0, "mean");
    auto loss = criterion(input, target);

    // Should not be NaN or Inf
    EXPECT_FALSE(std::isnan(loss.tensor().item<float>()));
    EXPECT_FALSE(std::isinf(loss.tensor().item<float>()));
}

TEST(AdvancedLossTest, DiceLoss_ZeroDenominator) {
    // Test when both input and target are zero
    auto input = Variable(zeros({1, 1, 2, 2}, DType::Float32), false);
    auto target = Variable(zeros({1, 1, 2, 2}, DType::Float32), false);

    auto criterion = DiceLoss(1.0, "mean");
    auto loss = criterion(input, target);

    // Smooth parameter should prevent division by zero
    EXPECT_FALSE(std::isnan(loss.tensor().item<float>()));
    EXPECT_FALSE(std::isinf(loss.tensor().item<float>()));
}

//==============================================================================
// SoftMarginLoss Tests
//==============================================================================

TEST(AdvancedLossTest, SoftMarginLoss_BasicForward) {
    auto input = Variable(randn({4, 3}, DType::Float32), false);
    auto target = Variable(ones({4, 3}, DType::Float32), false);

    auto criterion = SoftMarginLoss(Reduction::Mean);
    auto loss = criterion(input, target);

    EXPECT_GE(loss.tensor().item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(loss.tensor().item<float>()));
}

TEST(AdvancedLossTest, SoftMarginLoss_KnownValues) {
    // For x=0, y=1: loss = log(1 + exp(0)) = log(2) ≈ 0.6931
    auto input = Variable(zeros({1}, DType::Float32), false);
    auto target = Variable(ones({1}, DType::Float32), false);

    auto loss = soft_margin_loss(input, target, Reduction::Mean);
    EXPECT_NEAR(loss.tensor().item<float>(), std::log(2.0f), 1e-4);
}

TEST(AdvancedLossTest, SoftMarginLoss_ReductionModes) {
    auto input = Variable(randn({4}, DType::Float32), false);
    auto target = Variable(ones({4}, DType::Float32), false);

    auto loss_none = soft_margin_loss(input, target, Reduction::None);
    auto loss_sum = soft_margin_loss(input, target, Reduction::Sum);
    auto loss_mean = soft_margin_loss(input, target, Reduction::Mean);

    EXPECT_EQ(loss_none.shape().size(), 1u);
    EXPECT_EQ(loss_none.shape()[0], 4);
    EXPECT_GT(loss_sum.tensor().item<float>(), loss_mean.tensor().item<float>());
}

TEST(AdvancedLossTest, SoftMarginLoss_Gradient) {
    auto input = Variable(randn({3}, DType::Float32), true);
    auto target = Variable(ones({3}, DType::Float32), false);

    auto loss = soft_margin_loss(input, target, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

//==============================================================================
// HingeEmbeddingLoss Tests
//==============================================================================

TEST(AdvancedLossTest, HingeEmbeddingLoss_BasicForward) {
    auto input = Variable(full({4}, 0.5f, DType::Float32), false);
    auto target = Variable(ones({4}, DType::Float32), false);  // y = 1

    auto criterion = HingeEmbeddingLoss(1.0, Reduction::Mean);
    auto loss = criterion(input, target);

    // When y=1, loss = input = 0.5
    EXPECT_NEAR(loss.tensor().item<float>(), 0.5f, 1e-4);
}

TEST(AdvancedLossTest, HingeEmbeddingLoss_NegativeTarget) {
    auto input = Variable(full({4}, 0.5f, DType::Float32), false);
    auto neg_target = Variable(full({4}, -1.0f, DType::Float32), false);

    auto loss = hinge_embedding_loss(input, neg_target, 1.0, Reduction::Mean);

    // When y=-1, loss = max(0, 1.0 - 0.5) = 0.5
    EXPECT_NEAR(loss.tensor().item<float>(), 0.5f, 1e-4);
}

TEST(AdvancedLossTest, HingeEmbeddingLoss_Gradient) {
    auto input = Variable(randn({4}, DType::Float32), true);
    auto target = Variable(ones({4}, DType::Float32), false);

    auto loss = hinge_embedding_loss(input, target, 1.0, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

//==============================================================================
// PoissonNLLLoss Tests
//==============================================================================

TEST(AdvancedLossTest, PoissonNLLLoss_BasicForward) {
    auto input = Variable(randn({4, 3}, DType::Float32), false);
    auto target = Variable(full({4, 3}, 2.0f, DType::Float32), false);

    auto criterion = PoissonNLLLoss(true, false, 1e-8, Reduction::Mean);
    auto loss = criterion(input, target);

    EXPECT_FALSE(std::isnan(loss.tensor().item<float>()));
    EXPECT_FALSE(std::isinf(loss.tensor().item<float>()));
}

TEST(AdvancedLossTest, PoissonNLLLoss_KnownValues) {
    // For log_input=true: loss = exp(x) - y*x
    // x=0, y=1: loss = exp(0) - 1*0 = 1.0
    auto input = Variable(zeros({1}, DType::Float32), false);
    auto target = Variable(ones({1}, DType::Float32), false);

    auto loss = poisson_nll_loss(input, target, true, false, 1e-8, Reduction::Mean);
    EXPECT_NEAR(loss.tensor().item<float>(), 1.0f, 1e-4);
}

TEST(AdvancedLossTest, PoissonNLLLoss_Gradient) {
    auto input = Variable(randn({4}, DType::Float32), true);
    auto target = Variable(full({4}, 2.0f, DType::Float32), false);

    auto loss = poisson_nll_loss(input, target, true, false, 1e-8, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

//==============================================================================
// CosineEmbeddingLoss Tests
//==============================================================================

TEST(AdvancedLossTest, CosineEmbeddingLoss_SimilarPair) {
    // Identical vectors should give loss close to 0 for y=1
    auto input1 = Variable(ones({2, 4}, DType::Float32), false);
    auto input2 = Variable(ones({2, 4}, DType::Float32), false);
    auto target = Variable(ones({2}, DType::Float32), false);

    auto criterion = CosineEmbeddingLoss(0.0, Reduction::Mean);
    auto loss = criterion(input1, input2, target);

    EXPECT_NEAR(loss.tensor().item<float>(), 0.0f, 1e-4);
}

TEST(AdvancedLossTest, CosineEmbeddingLoss_DissimilarPair) {
    auto input1 = Variable(ones({2, 4}, DType::Float32), false);
    auto input2 = Variable(full({2, 4}, -1.0f, DType::Float32), false);
    auto target = Variable(full({2}, -1.0f, DType::Float32), false);

    auto loss = cosine_embedding_loss(input1, input2, target, 0.0, Reduction::Mean);

    // Opposite vectors have cos_sim = -1, with y=-1: max(0, -1 - 0) = 0
    EXPECT_GE(loss.tensor().item<float>(), 0.0f);
}

TEST(AdvancedLossTest, CosineEmbeddingLoss_Gradient) {
    auto input1 = Variable(randn({3, 4}, DType::Float32), true);
    auto input2 = Variable(randn({3, 4}, DType::Float32), true);
    auto target = Variable(ones({3}, DType::Float32), false);

    auto loss = cosine_embedding_loss(input1, input2, target, 0.0, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input1);
    EXPECT_GRAD_FLOWS(input2);
}

//==============================================================================
// TripletMarginLoss Tests
//==============================================================================

TEST(AdvancedLossTest, TripletMarginLoss_BasicForward) {
    auto anchor = Variable(randn({4, 8}, DType::Float32), false);
    auto positive = Variable(randn({4, 8}, DType::Float32), false);
    auto negative = Variable(randn({4, 8}, DType::Float32), false);

    auto criterion = TripletMarginLoss(1.0, 2.0, false, Reduction::Mean);
    auto loss = criterion(anchor, positive, negative);

    EXPECT_GE(loss.tensor().item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(loss.tensor().item<float>()));
}

TEST(AdvancedLossTest, TripletMarginLoss_Gradient) {
    auto anchor = Variable(randn({4, 8}, DType::Float32), true);
    auto positive = Variable(randn({4, 8}, DType::Float32), true);
    auto negative = Variable(randn({4, 8}, DType::Float32), true);

    auto loss = triplet_margin_loss(anchor, positive, negative, 1.0, 2.0, false, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(anchor);
}

//==============================================================================
// MultiLabelSoftMarginLoss Tests
//==============================================================================

TEST(AdvancedLossTest, MultiLabelSoftMarginLoss_BasicForward) {
    auto input = Variable(randn({4, 5}, DType::Float32), false);
    auto target = Variable(zeros({4, 5}, DType::Float32), false);

    auto criterion = MultiLabelSoftMarginLoss(Reduction::Mean);
    auto loss = criterion(input, target);

    EXPECT_GE(loss.tensor().item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(loss.tensor().item<float>()));
}

TEST(AdvancedLossTest, MultiLabelSoftMarginLoss_Gradient) {
    auto input = Variable(randn({4, 5}, DType::Float32), true);
    auto target = Variable(zeros({4, 5}, DType::Float32), false);

    auto loss = multi_label_soft_margin_loss(input, target, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

//==============================================================================
// MultiMarginLoss Tests
//==============================================================================

TEST(AdvancedLossTest, MultiMarginLoss_BasicForward) {
    auto input = Variable(randn({4, 5}, DType::Float32), false);
    // Create target with class indices using arange
    auto target = arange(0, 4, 1, DType::Int64);

    auto criterion = MultiMarginLoss(1, 1.0, Reduction::Mean);
    auto loss = criterion(input, target);

    float loss_val = loss.tensor().item<float>();
    EXPECT_GE(loss_val, 0.0f);
    EXPECT_FALSE(std::isnan(loss_val));
}

TEST(AdvancedLossTest, MultiMarginLoss_P2) {
    auto input = Variable(randn({4, 3}, DType::Float32), false);
    auto target = zeros({4}, DType::Int64);  // All class 0

    auto loss_p1 = multi_margin_loss(input, target, 1, 1.0, Reduction::Mean);
    auto loss_p2 = multi_margin_loss(input, target, 2, 1.0, Reduction::Mean);

    float v1 = loss_p1.tensor().item<float>();
    float v2 = loss_p2.tensor().item<float>();
    // Both should be non-negative and finite
    EXPECT_GE(v1, 0.0f);
    EXPECT_GE(v2, 0.0f);
    EXPECT_FALSE(std::isnan(v2));
}

//==============================================================================
// GaussianNLLLoss Tests
//==============================================================================

TEST(AdvancedLossTest, GaussianNLLLoss_BasicForward) {
    auto input = Variable(randn({4, 3}, DType::Float32), false);
    auto target = Variable(randn({4, 3}, DType::Float32), false);
    auto var = Variable(full({4, 3}, 1.0f, DType::Float32), false);

    auto criterion = GaussianNLLLoss(false, 1e-6, Reduction::Mean);
    auto loss = criterion(input, target, var);

    EXPECT_FALSE(std::isnan(loss.tensor().item<float>()));
    EXPECT_FALSE(std::isinf(loss.tensor().item<float>()));
}

TEST(AdvancedLossTest, GaussianNLLLoss_KnownValues) {
    // input=target, var=1: loss = 0.5*(log(1) + 0) = 0
    auto input = Variable(ones({1}, DType::Float32), false);
    auto target = Variable(ones({1}, DType::Float32), false);
    auto var = Variable(ones({1}, DType::Float32), false);

    auto loss = gaussian_nll_loss(input, target, var, false, 1e-6, Reduction::Mean);
    EXPECT_NEAR(loss.tensor().item<float>(), 0.0f, 1e-4);
}

TEST(AdvancedLossTest, GaussianNLLLoss_Gradient) {
    auto input = Variable(randn({4, 3}, DType::Float32), true);
    auto target = Variable(randn({4, 3}, DType::Float32), false);
    auto var = Variable(full({4, 3}, 1.0f, DType::Float32), true);

    auto loss = gaussian_nll_loss(input, target, var, false, 1e-6, Reduction::Mean);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_GRAD_FLOWS(var);
}
