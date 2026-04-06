/**
 * @file test_losses_missing_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for 9 previously untested loss functions
 *
 * Covers: MarginRankingLoss, PoissonNLLLoss, CosineEmbeddingLoss,
 *         TripletMarginLoss, GaussianNLLLoss, SoftMarginLoss,
 *         HingeEmbeddingLoss, MultiLabelSoftMarginLoss, MultiMarginLoss
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class MissingLossesMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// MarginRankingLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, MarginRankingLoss_ForwardShape) {
    nn::MarginRankingLoss loss(0.5, nn::Reduction::Mean);

    auto input1 = createInput({4, 1}, false);
    auto input2 = createInput({4, 1}, false);
    // Target: +1 or -1
    auto target = Variable(tenzor::ones({4, 1}, dtype(), device()), false);

    auto result = loss.forward(
        Variable(input1.tensor(), true), input2, target);
    // Mean reduction => scalar
    EXPECT_LE(result.tensor().numel(), 1);
    expectDevice(result.tensor());
}

TEST_P(MissingLossesMultiDTypeTest, MarginRankingLoss_ReductionModes) {
    auto input1 = Variable(createRandn({8}), false);
    auto input2 = Variable(createRandn({8}), false);
    auto target = Variable(tenzor::ones({8}, dtype(), device()), false);

    {
        nn::MarginRankingLoss loss_mean(0.0, nn::Reduction::Mean);
        auto r = loss_mean.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::MarginRankingLoss loss_sum(0.0, nn::Reduction::Sum);
        auto r = loss_sum.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::MarginRankingLoss loss_none(0.0, nn::Reduction::None);
        auto r = loss_none.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_EQ(r.tensor().numel(), 8);
    }
}

TEST_P(MissingLossesMultiDTypeTest, MarginRankingLoss_GradientFlow) {
    nn::MarginRankingLoss loss(0.5, nn::Reduction::Mean);

    auto input1 = createInput({4}, true);
    auto input2 = createInput({4}, false);
    auto target = Variable(tenzor::ones({4}, dtype(), device()), false);

    auto result = loss.forward(input1, input2, target);
    result.backward();

    EXPECT_TRUE(input1.grad().has_value());
    expectShape(input1.grad().value(), {4});
}

// ============================================================================
// PoissonNLLLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, PoissonNLLLoss_ForwardShape) {
    nn::PoissonNLLLoss loss(true, false, 1e-8, nn::Reduction::Mean);

    auto input = createInput({4, 5}, false);
    // Target should be non-negative counts
    auto target = Variable(tenzor::abs(createRandn({4, 5})), false);

    auto result = loss.forward(Variable(input.tensor(), true), target);
    EXPECT_LE(result.tensor().numel(), 1);
    expectDevice(result.tensor());
}

TEST_P(MissingLossesMultiDTypeTest, PoissonNLLLoss_ReductionModes) {
    auto input = Variable(createRandn({6, 3}), false);
    auto target = Variable(tenzor::abs(createRandn({6, 3})), false);

    {
        nn::PoissonNLLLoss loss_mean(true, false, 1e-8, nn::Reduction::Mean);
        auto r = loss_mean.forward(Variable(input.tensor(), true), target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::PoissonNLLLoss loss_sum(true, false, 1e-8, nn::Reduction::Sum);
        auto r = loss_sum.forward(Variable(input.tensor(), true), target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::PoissonNLLLoss loss_none(true, false, 1e-8, nn::Reduction::None);
        auto r = loss_none.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 18);
    }
}

TEST_P(MissingLossesMultiDTypeTest, PoissonNLLLoss_GradientFlow) {
    nn::PoissonNLLLoss loss(true, false, 1e-8, nn::Reduction::Mean);

    auto input = createInput({4, 5}, true);
    auto target = Variable(tenzor::abs(createRandn({4, 5})), false);

    auto result = loss.forward(input, target);
    result.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectShape(input.grad().value(), {4, 5});
}

// ============================================================================
// CosineEmbeddingLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, CosineEmbeddingLoss_ForwardShape) {
    nn::CosineEmbeddingLoss loss(0.0, nn::Reduction::Mean);

    auto input1 = createInput({4, 8}, false);
    auto input2 = createInput({4, 8}, false);
    // Target: +1 (similar) or -1 (dissimilar)
    auto target = Variable(tenzor::ones({4}, dtype(), device()), false);

    auto result = loss.forward(
        Variable(input1.tensor(), true), input2, target);
    EXPECT_LE(result.tensor().numel(), 1);
    expectDevice(result.tensor());
}

TEST_P(MissingLossesMultiDTypeTest, CosineEmbeddingLoss_ReductionModes) {
    auto input1 = Variable(createRandn({6, 4}), false);
    auto input2 = Variable(createRandn({6, 4}), false);
    auto target = Variable(tenzor::ones({6}, dtype(), device()), false);

    {
        nn::CosineEmbeddingLoss loss_mean(0.0, nn::Reduction::Mean);
        auto r = loss_mean.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::CosineEmbeddingLoss loss_sum(0.0, nn::Reduction::Sum);
        auto r = loss_sum.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::CosineEmbeddingLoss loss_none(0.0, nn::Reduction::None);
        auto r = loss_none.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_EQ(r.tensor().numel(), 6);
    }
}

TEST_P(MissingLossesMultiDTypeTest, CosineEmbeddingLoss_GradientFlow) {
    nn::CosineEmbeddingLoss loss(0.0, nn::Reduction::Mean);

    auto input1 = createInput({4, 8}, true);
    auto input2 = createInput({4, 8}, false);
    auto target = Variable(tenzor::ones({4}, dtype(), device()), false);

    auto result = loss.forward(input1, input2, target);
    result.backward();

    EXPECT_TRUE(input1.grad().has_value());
    expectShape(input1.grad().value(), {4, 8});
}

// ============================================================================
// TripletMarginLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, TripletMarginLoss_ForwardShape) {
    nn::TripletMarginLoss loss(1.0, 2.0, false, nn::Reduction::Mean);

    auto anchor   = createInput({4, 16}, false);
    auto positive = createInput({4, 16}, false);
    auto negative = createInput({4, 16}, false);

    auto result = loss.forward(
        Variable(anchor.tensor(), true), positive, negative);
    EXPECT_LE(result.tensor().numel(), 1);
    expectDevice(result.tensor());
}

TEST_P(MissingLossesMultiDTypeTest, TripletMarginLoss_ReductionModes) {
    auto anchor   = Variable(createRandn({6, 8}), false);
    auto positive = Variable(createRandn({6, 8}), false);
    auto negative = Variable(createRandn({6, 8}), false);

    {
        nn::TripletMarginLoss loss_mean(1.0, 2.0, false, nn::Reduction::Mean);
        auto r = loss_mean.forward(
            Variable(anchor.tensor(), true), positive, negative);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::TripletMarginLoss loss_sum(1.0, 2.0, false, nn::Reduction::Sum);
        auto r = loss_sum.forward(
            Variable(anchor.tensor(), true), positive, negative);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::TripletMarginLoss loss_none(1.0, 2.0, false, nn::Reduction::None);
        auto r = loss_none.forward(
            Variable(anchor.tensor(), true), positive, negative);
        EXPECT_EQ(r.tensor().numel(), 6);
    }
}

TEST_P(MissingLossesMultiDTypeTest, TripletMarginLoss_GradientFlow) {
    nn::TripletMarginLoss loss(1.0, 2.0, false, nn::Reduction::Mean);

    auto anchor   = createInput({4, 16}, true);
    auto positive = createInput({4, 16}, false);
    auto negative = createInput({4, 16}, false);

    auto result = loss.forward(anchor, positive, negative);
    result.backward();

    EXPECT_TRUE(anchor.grad().has_value());
    expectShape(anchor.grad().value(), {4, 16});
}

// ============================================================================
// GaussianNLLLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, GaussianNLLLoss_ForwardShape) {
    nn::GaussianNLLLoss loss(false, 1e-6, nn::Reduction::Mean);

    auto input  = createInput({4, 3}, false);
    auto target = Variable(createRandn({4, 3}), false);
    // Variance must be positive
    auto var    = Variable(tenzor::abs(createRandn({4, 3})) +
                  tenzor::full({4, 3}, 0.1f, dtype(), device()), false);

    auto result = loss.forward(Variable(input.tensor(), true), target, var);
    EXPECT_LE(result.tensor().numel(), 1);
    expectDevice(result.tensor());
}

TEST_P(MissingLossesMultiDTypeTest, GaussianNLLLoss_ReductionModes) {
    auto input  = Variable(createRandn({6, 2}), false);
    auto target = Variable(createRandn({6, 2}), false);
    auto var    = Variable(tenzor::abs(createRandn({6, 2})) +
                  tenzor::full({6, 2}, 0.1f, dtype(), device()), false);

    {
        nn::GaussianNLLLoss loss_mean(false, 1e-6, nn::Reduction::Mean);
        auto r = loss_mean.forward(
            Variable(input.tensor(), true), target, var);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::GaussianNLLLoss loss_sum(false, 1e-6, nn::Reduction::Sum);
        auto r = loss_sum.forward(
            Variable(input.tensor(), true), target, var);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::GaussianNLLLoss loss_none(false, 1e-6, nn::Reduction::None);
        auto r = loss_none.forward(
            Variable(input.tensor(), true), target, var);
        EXPECT_EQ(r.tensor().numel(), 12);
    }
}

TEST_P(MissingLossesMultiDTypeTest, GaussianNLLLoss_GradientFlow) {
    nn::GaussianNLLLoss loss(false, 1e-6, nn::Reduction::Mean);

    auto input  = createInput({4, 3}, true);
    auto target = Variable(createRandn({4, 3}), false);
    auto var    = Variable(tenzor::abs(createRandn({4, 3})) +
                  tenzor::full({4, 3}, 0.1f, dtype(), device()), false);

    auto result = loss.forward(input, target, var);
    result.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectShape(input.grad().value(), {4, 3});
}

// ============================================================================
// SoftMarginLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, SoftMarginLoss_ForwardShape) {
    nn::SoftMarginLoss loss(nn::Reduction::Mean);

    auto input = createInput({4, 5}, false);
    // Target: +1 or -1
    auto target = Variable(tenzor::ones({4, 5}, dtype(), device()), false);

    auto result = loss.forward(Variable(input.tensor(), true), target);
    EXPECT_LE(result.tensor().numel(), 1);
    expectDevice(result.tensor());
}

TEST_P(MissingLossesMultiDTypeTest, SoftMarginLoss_ReductionModes) {
    auto input  = Variable(createRandn({6, 4}), false);
    auto target = Variable(tenzor::ones({6, 4}, dtype(), device()), false);

    {
        nn::SoftMarginLoss loss_mean(nn::Reduction::Mean);
        auto r = loss_mean.forward(Variable(input.tensor(), true), target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::SoftMarginLoss loss_sum(nn::Reduction::Sum);
        auto r = loss_sum.forward(Variable(input.tensor(), true), target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::SoftMarginLoss loss_none(nn::Reduction::None);
        auto r = loss_none.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 24);
    }
}

TEST_P(MissingLossesMultiDTypeTest, SoftMarginLoss_GradientFlow) {
    nn::SoftMarginLoss loss(nn::Reduction::Mean);

    auto input  = createInput({4, 5}, true);
    auto target = Variable(tenzor::ones({4, 5}, dtype(), device()), false);

    auto result = loss.forward(input, target);
    result.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectShape(input.grad().value(), {4, 5});
}

// ============================================================================
// HingeEmbeddingLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, HingeEmbeddingLoss_ForwardShape) {
    nn::HingeEmbeddingLoss loss(1.0, nn::Reduction::Mean);

    auto input = createInput({4, 5}, false);
    // Target: +1 or -1
    auto target = Variable(tenzor::ones({4, 5}, dtype(), device()), false);

    auto result = loss.forward(Variable(input.tensor(), true), target);
    EXPECT_LE(result.tensor().numel(), 1);
    expectDevice(result.tensor());
}

TEST_P(MissingLossesMultiDTypeTest, HingeEmbeddingLoss_ReductionModes) {
    auto input  = Variable(createRandn({8}), false);
    auto target = Variable(tenzor::ones({8}, dtype(), device()), false);

    {
        nn::HingeEmbeddingLoss loss_mean(1.0, nn::Reduction::Mean);
        auto r = loss_mean.forward(Variable(input.tensor(), true), target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::HingeEmbeddingLoss loss_sum(1.0, nn::Reduction::Sum);
        auto r = loss_sum.forward(Variable(input.tensor(), true), target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::HingeEmbeddingLoss loss_none(1.0, nn::Reduction::None);
        auto r = loss_none.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 8);
    }
}

TEST_P(MissingLossesMultiDTypeTest, HingeEmbeddingLoss_GradientFlow) {
    nn::HingeEmbeddingLoss loss(1.0, nn::Reduction::Mean);

    auto input  = createInput({4, 5}, true);
    auto target = Variable(tenzor::ones({4, 5}, dtype(), device()), false);

    auto result = loss.forward(input, target);
    result.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectShape(input.grad().value(), {4, 5});
}

// ============================================================================
// MultiLabelSoftMarginLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, MultiLabelSoftMarginLoss_ForwardShape) {
    nn::MultiLabelSoftMarginLoss loss(nn::Reduction::Mean);

    auto input = createInput({4, 6}, false);
    // Target: 0 or 1 multi-label indicators
    auto target = Variable(tenzor::zeros({4, 6}, dtype(), device()), false);

    auto result = loss.forward(Variable(input.tensor(), true), target);
    EXPECT_LE(result.tensor().numel(), 1);
    expectDevice(result.tensor());
}

TEST_P(MissingLossesMultiDTypeTest, MultiLabelSoftMarginLoss_ReductionModes) {
    auto input  = Variable(createRandn({6, 4}), false);
    auto target = Variable(tenzor::ones({6, 4}, dtype(), device()), false);

    {
        nn::MultiLabelSoftMarginLoss loss_mean(nn::Reduction::Mean);
        auto r = loss_mean.forward(Variable(input.tensor(), true), target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::MultiLabelSoftMarginLoss loss_sum(nn::Reduction::Sum);
        auto r = loss_sum.forward(Variable(input.tensor(), true), target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::MultiLabelSoftMarginLoss loss_none(nn::Reduction::None);
        auto r = loss_none.forward(Variable(input.tensor(), true), target);
        // None reduction: one loss per sample in batch
        EXPECT_EQ(r.tensor().numel(), 6);
    }
}

TEST_P(MissingLossesMultiDTypeTest, MultiLabelSoftMarginLoss_GradientFlow) {
    nn::MultiLabelSoftMarginLoss loss(nn::Reduction::Mean);

    auto input  = createInput({4, 6}, true);
    auto target = Variable(tenzor::ones({4, 6}, dtype(), device()), false);

    auto result = loss.forward(input, target);
    result.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectShape(input.grad().value(), {4, 6});
}

// ============================================================================
// MultiMarginLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, MultiMarginLoss_ForwardShape) {
    nn::MultiMarginLoss loss(1, 1.0, nn::Reduction::Mean);

    auto input = createInput({4, 5}, false);
    // Target: class indices as Int64 tensor
    auto target = tenzor::zeros({4}, DType::Int64, device());

    auto result = loss.forward(Variable(input.tensor(), true), target);
    EXPECT_LE(result.tensor().numel(), 1);
    expectDevice(result.tensor());
}

TEST_P(MissingLossesMultiDTypeTest, MultiMarginLoss_ReductionModes) {
    auto input  = Variable(createRandn({8, 5}), false);
    auto target = tenzor::zeros({8}, DType::Int64, device());

    {
        nn::MultiMarginLoss loss_mean(1, 1.0, nn::Reduction::Mean);
        auto r = loss_mean.forward(Variable(input.tensor(), true), target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::MultiMarginLoss loss_sum(1, 1.0, nn::Reduction::Sum);
        auto r = loss_sum.forward(Variable(input.tensor(), true), target);
        EXPECT_LE(r.tensor().numel(), 1);
    }
    {
        nn::MultiMarginLoss loss_none(1, 1.0, nn::Reduction::None);
        auto r = loss_none.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 8);
    }
}

TEST_P(MissingLossesMultiDTypeTest, MultiMarginLoss_GradientFlow) {
    nn::MultiMarginLoss loss(1, 1.0, nn::Reduction::Mean);

    auto input  = createInput({4, 5}, true);
    auto target = tenzor::zeros({4}, DType::Int64, device());

    auto result = loss.forward(input, target);
    result.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectShape(input.grad().value(), {4, 5});
}

// ============================================================================
// Instantiate
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MissingLossesMultiDTypeTest);
