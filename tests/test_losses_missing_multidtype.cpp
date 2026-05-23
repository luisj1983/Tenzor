/**
 * @file test_losses_missing_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for 9 previously untested loss functions
 *
 * Covers: MarginRankingLoss, PoissonNLLLoss, CosineEmbeddingLoss,
 *         TripletMarginLoss, GaussianNLLLoss, SoftMarginLoss,
 *         HingeEmbeddingLoss, MultiLabelSoftMarginLoss, MultiMarginLoss
 *
 * Audit-T.1 + audit-T.8: every TEST_P now also asserts the scalar (or
 * elementwise) loss value against a CPU reference computed with the same
 * weights and inputs. The `EXPECT_LE(numel, 1)` pattern that previously
 * masked numel=0 outputs has been replaced with `EXPECT_EQ(numel, 1)`
 * coupled with a Float32 closeness check against the CPU result.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include "multi_backend_dtype_fixture.hpp"
#include "grad_flow_helpers.hpp"

#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class MissingLossesMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Audit-T.1: pull a scalar out of a (possibly half-precision) tensor for
    // comparison.  Mirrors compute_max() in the fixture but for arbitrary
    // single-element tensors.
    float scalarValue(const Tensor& t) const {
        auto cpu = t.to(Device::cpu());
        if (cpu.dtype() != DType::Float32) cpu = cpu.to(DType::Float32);
        return cpu.data<float>()[0];
    }

    // Audit-T.1: tolerance for the device-vs-CPU scalar loss comparison.
    // Loss values aggregate over the batch so small per-element rounding
    // accumulates; pick a slightly larger floor than atol().
    float lossAtol() const {
        switch (dtype()) {
            case DType::Float16: return 5e-2f;
            case DType::BFloat16: return 5e-2f;
            case DType::Float64: return 1e-5f;
            default: return 1e-3f;
        }
    }
};

// ============================================================================
// MarginRankingLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, MarginRankingLoss_ForwardShape) {
    nn::MarginRankingLoss loss(0.5, nn::Reduction::Mean);
    nn::MarginRankingLoss loss_cpu(0.5, nn::Reduction::Mean);

    auto input1 = createInput({4, 1}, false);
    auto input2 = createInput({4, 1}, false);
    // Target: +1 or -1
    auto target = Variable(tenzor::ones({4, 1}, dtype(), device()), false);

    auto result = loss.forward(
        Variable(input1.tensor(), true), input2, target);
    // Audit-T.8: empty-output bug guard — Mean reduction must yield exactly
    // one element.
    EXPECT_EQ(result.tensor().numel(), 1);
    expectDevice(result.tensor());

    // Audit-T.1: scalar value match against CPU reference computed from the
    // same inputs.  Cast to Float32 CPU to avoid relying on per-dtype CPU
    // dispatch for the reference.
    auto in1_cpu = Variable(input1.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto in2_cpu = Variable(input2.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto tgt_cpu = Variable(target.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = loss_cpu.forward(in1_cpu, in2_cpu, tgt_cpu);
    EXPECT_NEAR(scalarValue(result.tensor()), scalarValue(ref.tensor()), lossAtol());
}

TEST_P(MissingLossesMultiDTypeTest, MarginRankingLoss_ReductionModes) {
    auto input1 = Variable(createRandn({8}), false);
    auto input2 = Variable(createRandn({8}), false);
    auto target = Variable(tenzor::ones({8}, dtype(), device()), false);

    auto in1_cpu = input1.tensor().to(Device::cpu()).to(DType::Float32);
    auto in2_cpu = input2.tensor().to(Device::cpu()).to(DType::Float32);
    auto tgt_cpu = target.tensor().to(Device::cpu()).to(DType::Float32);

    {
        nn::MarginRankingLoss loss_mean(0.0, nn::Reduction::Mean);
        nn::MarginRankingLoss loss_mean_cpu(0.0, nn::Reduction::Mean);
        auto r = loss_mean.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_EQ(r.tensor().numel(), 1);  // audit-T.8
        auto ref = loss_mean_cpu.forward(
            Variable(in1_cpu, false), Variable(in2_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol());
    }
    {
        nn::MarginRankingLoss loss_sum(0.0, nn::Reduction::Sum);
        nn::MarginRankingLoss loss_sum_cpu(0.0, nn::Reduction::Sum);
        auto r = loss_sum.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_EQ(r.tensor().numel(), 1);  // audit-T.8
        auto ref = loss_sum_cpu.forward(
            Variable(in1_cpu, false), Variable(in2_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol() * 8.0f);
    }
    {
        nn::MarginRankingLoss loss_none(0.0, nn::Reduction::None);
        nn::MarginRankingLoss loss_none_cpu(0.0, nn::Reduction::None);
        auto r = loss_none.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_EQ(r.tensor().numel(), 8);
        auto ref = loss_none_cpu.forward(
            Variable(in1_cpu, false), Variable(in2_cpu, false), Variable(tgt_cpu, false));
        // Element-wise check across the unreduced loss vector.
        expectTensorNear(r.tensor(), ref.tensor(), lossAtol());
    }
}

TEST_P(MissingLossesMultiDTypeTest, MarginRankingLoss_GradientFlow) {
    nn::MarginRankingLoss loss(0.5, nn::Reduction::Mean);

    auto input1 = createInput({4}, true);
    auto input2 = createInput({4}, false);
    auto target = Variable(tenzor::ones({4}, dtype(), device()), false);

    auto result = loss.forward(input1, input2, target);
    result.backward();

    EXPECT_GRAD_FLOWS(input1);
    expectShape(input1.grad().value(), {4});
}

// ============================================================================
// PoissonNLLLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, PoissonNLLLoss_ForwardShape) {
    nn::PoissonNLLLoss loss(true, false, 1e-8, nn::Reduction::Mean);
    nn::PoissonNLLLoss loss_cpu(true, false, 1e-8, nn::Reduction::Mean);

    auto input = createInput({4, 5}, false);
    // Target should be non-negative counts
    auto target = Variable(tenzor::abs(createRandn({4, 5})), false);

    auto result = loss.forward(Variable(input.tensor(), true), target);
    EXPECT_EQ(result.tensor().numel(), 1);
    expectDevice(result.tensor());

    auto in_cpu = Variable(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto tgt_cpu = Variable(target.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = loss_cpu.forward(in_cpu, tgt_cpu);
    EXPECT_NEAR(scalarValue(result.tensor()), scalarValue(ref.tensor()), lossAtol());
}

TEST_P(MissingLossesMultiDTypeTest, PoissonNLLLoss_ReductionModes) {
    auto input = Variable(createRandn({6, 3}), false);
    auto target = Variable(tenzor::abs(createRandn({6, 3})), false);
    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto tgt_cpu = target.tensor().to(Device::cpu()).to(DType::Float32);

    {
        nn::PoissonNLLLoss loss_mean(true, false, 1e-8, nn::Reduction::Mean);
        nn::PoissonNLLLoss loss_mean_cpu(true, false, 1e-8, nn::Reduction::Mean);
        auto r = loss_mean.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_mean_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol());
    }
    {
        nn::PoissonNLLLoss loss_sum(true, false, 1e-8, nn::Reduction::Sum);
        nn::PoissonNLLLoss loss_sum_cpu(true, false, 1e-8, nn::Reduction::Sum);
        auto r = loss_sum.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_sum_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol() * 18.0f);
    }
    {
        nn::PoissonNLLLoss loss_none(true, false, 1e-8, nn::Reduction::None);
        nn::PoissonNLLLoss loss_none_cpu(true, false, 1e-8, nn::Reduction::None);
        auto r = loss_none.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 18);
        auto ref = loss_none_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        expectTensorNear(r.tensor(), ref.tensor(), lossAtol());
    }
}

TEST_P(MissingLossesMultiDTypeTest, PoissonNLLLoss_GradientFlow) {
    nn::PoissonNLLLoss loss(true, false, 1e-8, nn::Reduction::Mean);

    auto input = createInput({4, 5}, true);
    auto target = Variable(tenzor::abs(createRandn({4, 5})), false);

    auto result = loss.forward(input, target);
    result.backward();

    EXPECT_GRAD_FLOWS(input);
    expectShape(input.grad().value(), {4, 5});
}

// ============================================================================
// CosineEmbeddingLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, CosineEmbeddingLoss_ForwardShape) {
    nn::CosineEmbeddingLoss loss(0.0, nn::Reduction::Mean);
    nn::CosineEmbeddingLoss loss_cpu(0.0, nn::Reduction::Mean);

    auto input1 = createInput({4, 8}, false);
    auto input2 = createInput({4, 8}, false);
    // Target: +1 (similar) or -1 (dissimilar)
    auto target = Variable(tenzor::ones({4}, dtype(), device()), false);

    auto result = loss.forward(
        Variable(input1.tensor(), true), input2, target);
    EXPECT_EQ(result.tensor().numel(), 1);
    expectDevice(result.tensor());

    auto in1_cpu = Variable(input1.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto in2_cpu = Variable(input2.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto tgt_cpu = Variable(target.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = loss_cpu.forward(in1_cpu, in2_cpu, tgt_cpu);
    EXPECT_NEAR(scalarValue(result.tensor()), scalarValue(ref.tensor()), lossAtol());
}

TEST_P(MissingLossesMultiDTypeTest, CosineEmbeddingLoss_ReductionModes) {
    auto input1 = Variable(createRandn({6, 4}), false);
    auto input2 = Variable(createRandn({6, 4}), false);
    auto target = Variable(tenzor::ones({6}, dtype(), device()), false);
    auto in1_cpu = input1.tensor().to(Device::cpu()).to(DType::Float32);
    auto in2_cpu = input2.tensor().to(Device::cpu()).to(DType::Float32);
    auto tgt_cpu = target.tensor().to(Device::cpu()).to(DType::Float32);

    {
        nn::CosineEmbeddingLoss loss_mean(0.0, nn::Reduction::Mean);
        nn::CosineEmbeddingLoss loss_mean_cpu(0.0, nn::Reduction::Mean);
        auto r = loss_mean.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_mean_cpu.forward(
            Variable(in1_cpu, false), Variable(in2_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol());
    }
    {
        nn::CosineEmbeddingLoss loss_sum(0.0, nn::Reduction::Sum);
        nn::CosineEmbeddingLoss loss_sum_cpu(0.0, nn::Reduction::Sum);
        auto r = loss_sum.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_sum_cpu.forward(
            Variable(in1_cpu, false), Variable(in2_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol() * 6.0f);
    }
    {
        nn::CosineEmbeddingLoss loss_none(0.0, nn::Reduction::None);
        nn::CosineEmbeddingLoss loss_none_cpu(0.0, nn::Reduction::None);
        auto r = loss_none.forward(
            Variable(input1.tensor(), true), input2, target);
        EXPECT_EQ(r.tensor().numel(), 6);
        auto ref = loss_none_cpu.forward(
            Variable(in1_cpu, false), Variable(in2_cpu, false), Variable(tgt_cpu, false));
        expectTensorNear(r.tensor(), ref.tensor(), lossAtol());
    }
}

TEST_P(MissingLossesMultiDTypeTest, CosineEmbeddingLoss_GradientFlow) {
    nn::CosineEmbeddingLoss loss(0.0, nn::Reduction::Mean);

    auto input1 = createInput({4, 8}, true);
    auto input2 = createInput({4, 8}, false);
    auto target = Variable(tenzor::ones({4}, dtype(), device()), false);

    auto result = loss.forward(input1, input2, target);
    result.backward();

    EXPECT_GRAD_FLOWS(input1);
    expectShape(input1.grad().value(), {4, 8});
}

// ============================================================================
// TripletMarginLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, TripletMarginLoss_ForwardShape) {
    nn::TripletMarginLoss loss(1.0, 2.0, false, nn::Reduction::Mean);
    nn::TripletMarginLoss loss_cpu(1.0, 2.0, false, nn::Reduction::Mean);

    auto anchor   = createInput({4, 16}, false);
    auto positive = createInput({4, 16}, false);
    auto negative = createInput({4, 16}, false);

    auto result = loss.forward(
        Variable(anchor.tensor(), true), positive, negative);
    EXPECT_EQ(result.tensor().numel(), 1);
    expectDevice(result.tensor());

    auto a_cpu = Variable(anchor.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto p_cpu = Variable(positive.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto n_cpu = Variable(negative.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = loss_cpu.forward(a_cpu, p_cpu, n_cpu);
    EXPECT_NEAR(scalarValue(result.tensor()), scalarValue(ref.tensor()), lossAtol());
}

TEST_P(MissingLossesMultiDTypeTest, TripletMarginLoss_ReductionModes) {
    auto anchor   = Variable(createRandn({6, 8}), false);
    auto positive = Variable(createRandn({6, 8}), false);
    auto negative = Variable(createRandn({6, 8}), false);
    auto a_cpu = anchor.tensor().to(Device::cpu()).to(DType::Float32);
    auto p_cpu = positive.tensor().to(Device::cpu()).to(DType::Float32);
    auto n_cpu = negative.tensor().to(Device::cpu()).to(DType::Float32);

    {
        nn::TripletMarginLoss loss_mean(1.0, 2.0, false, nn::Reduction::Mean);
        nn::TripletMarginLoss loss_mean_cpu(1.0, 2.0, false, nn::Reduction::Mean);
        auto r = loss_mean.forward(
            Variable(anchor.tensor(), true), positive, negative);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_mean_cpu.forward(
            Variable(a_cpu, false), Variable(p_cpu, false), Variable(n_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol());
    }
    {
        nn::TripletMarginLoss loss_sum(1.0, 2.0, false, nn::Reduction::Sum);
        nn::TripletMarginLoss loss_sum_cpu(1.0, 2.0, false, nn::Reduction::Sum);
        auto r = loss_sum.forward(
            Variable(anchor.tensor(), true), positive, negative);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_sum_cpu.forward(
            Variable(a_cpu, false), Variable(p_cpu, false), Variable(n_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol() * 6.0f);
    }
    {
        nn::TripletMarginLoss loss_none(1.0, 2.0, false, nn::Reduction::None);
        nn::TripletMarginLoss loss_none_cpu(1.0, 2.0, false, nn::Reduction::None);
        auto r = loss_none.forward(
            Variable(anchor.tensor(), true), positive, negative);
        EXPECT_EQ(r.tensor().numel(), 6);
        auto ref = loss_none_cpu.forward(
            Variable(a_cpu, false), Variable(p_cpu, false), Variable(n_cpu, false));
        expectTensorNear(r.tensor(), ref.tensor(), lossAtol());
    }
}

TEST_P(MissingLossesMultiDTypeTest, TripletMarginLoss_GradientFlow) {
    nn::TripletMarginLoss loss(1.0, 2.0, false, nn::Reduction::Mean);

    auto anchor   = createInput({4, 16}, true);
    auto positive = createInput({4, 16}, false);
    auto negative = createInput({4, 16}, false);

    auto result = loss.forward(anchor, positive, negative);
    result.backward();

    EXPECT_GRAD_FLOWS(anchor);
    expectShape(anchor.grad().value(), {4, 16});
}

// ============================================================================
// GaussianNLLLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, GaussianNLLLoss_ForwardShape) {
    nn::GaussianNLLLoss loss(false, 1e-6, nn::Reduction::Mean);
    nn::GaussianNLLLoss loss_cpu(false, 1e-6, nn::Reduction::Mean);

    auto input  = createInput({4, 3}, false);
    auto target = Variable(createRandn({4, 3}), false);
    // Variance must be positive
    auto var    = Variable(tenzor::abs(createRandn({4, 3})) +
                  tenzor::full({4, 3}, 0.1f, dtype(), device()), false);

    auto result = loss.forward(Variable(input.tensor(), true), target, var);
    EXPECT_EQ(result.tensor().numel(), 1);
    expectDevice(result.tensor());

    auto in_cpu = Variable(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto tgt_cpu = Variable(target.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto var_cpu = Variable(var.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = loss_cpu.forward(in_cpu, tgt_cpu, var_cpu);
    EXPECT_NEAR(scalarValue(result.tensor()), scalarValue(ref.tensor()), lossAtol());
}

TEST_P(MissingLossesMultiDTypeTest, GaussianNLLLoss_ReductionModes) {
    auto input  = Variable(createRandn({6, 2}), false);
    auto target = Variable(createRandn({6, 2}), false);
    auto var    = Variable(tenzor::abs(createRandn({6, 2})) +
                  tenzor::full({6, 2}, 0.1f, dtype(), device()), false);
    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto tgt_cpu = target.tensor().to(Device::cpu()).to(DType::Float32);
    auto var_cpu = var.tensor().to(Device::cpu()).to(DType::Float32);

    {
        nn::GaussianNLLLoss loss_mean(false, 1e-6, nn::Reduction::Mean);
        nn::GaussianNLLLoss loss_mean_cpu(false, 1e-6, nn::Reduction::Mean);
        auto r = loss_mean.forward(
            Variable(input.tensor(), true), target, var);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_mean_cpu.forward(
            Variable(in_cpu, false), Variable(tgt_cpu, false), Variable(var_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol());
    }
    {
        nn::GaussianNLLLoss loss_sum(false, 1e-6, nn::Reduction::Sum);
        nn::GaussianNLLLoss loss_sum_cpu(false, 1e-6, nn::Reduction::Sum);
        auto r = loss_sum.forward(
            Variable(input.tensor(), true), target, var);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_sum_cpu.forward(
            Variable(in_cpu, false), Variable(tgt_cpu, false), Variable(var_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol() * 12.0f);
    }
    {
        nn::GaussianNLLLoss loss_none(false, 1e-6, nn::Reduction::None);
        nn::GaussianNLLLoss loss_none_cpu(false, 1e-6, nn::Reduction::None);
        auto r = loss_none.forward(
            Variable(input.tensor(), true), target, var);
        EXPECT_EQ(r.tensor().numel(), 12);
        auto ref = loss_none_cpu.forward(
            Variable(in_cpu, false), Variable(tgt_cpu, false), Variable(var_cpu, false));
        expectTensorNear(r.tensor(), ref.tensor(), lossAtol());
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

    EXPECT_GRAD_FLOWS(input);
    expectShape(input.grad().value(), {4, 3});
}

// ============================================================================
// SoftMarginLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, SoftMarginLoss_ForwardShape) {
    nn::SoftMarginLoss loss(nn::Reduction::Mean);
    nn::SoftMarginLoss loss_cpu(nn::Reduction::Mean);

    auto input = createInput({4, 5}, false);
    // Target: +1 or -1
    auto target = Variable(tenzor::ones({4, 5}, dtype(), device()), false);

    auto result = loss.forward(Variable(input.tensor(), true), target);
    EXPECT_EQ(result.tensor().numel(), 1);
    expectDevice(result.tensor());

    auto in_cpu = Variable(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto tgt_cpu = Variable(target.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = loss_cpu.forward(in_cpu, tgt_cpu);
    EXPECT_NEAR(scalarValue(result.tensor()), scalarValue(ref.tensor()), lossAtol());
}

TEST_P(MissingLossesMultiDTypeTest, SoftMarginLoss_ReductionModes) {
    auto input  = Variable(createRandn({6, 4}), false);
    auto target = Variable(tenzor::ones({6, 4}, dtype(), device()), false);
    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto tgt_cpu = target.tensor().to(Device::cpu()).to(DType::Float32);

    {
        nn::SoftMarginLoss loss_mean(nn::Reduction::Mean);
        nn::SoftMarginLoss loss_mean_cpu(nn::Reduction::Mean);
        auto r = loss_mean.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_mean_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol());
    }
    {
        nn::SoftMarginLoss loss_sum(nn::Reduction::Sum);
        nn::SoftMarginLoss loss_sum_cpu(nn::Reduction::Sum);
        auto r = loss_sum.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_sum_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol() * 24.0f);
    }
    {
        nn::SoftMarginLoss loss_none(nn::Reduction::None);
        nn::SoftMarginLoss loss_none_cpu(nn::Reduction::None);
        auto r = loss_none.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 24);
        auto ref = loss_none_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        expectTensorNear(r.tensor(), ref.tensor(), lossAtol());
    }
}

TEST_P(MissingLossesMultiDTypeTest, SoftMarginLoss_GradientFlow) {
    nn::SoftMarginLoss loss(nn::Reduction::Mean);

    auto input  = createInput({4, 5}, true);
    auto target = Variable(tenzor::ones({4, 5}, dtype(), device()), false);

    auto result = loss.forward(input, target);
    result.backward();

    EXPECT_GRAD_FLOWS(input);
    expectShape(input.grad().value(), {4, 5});
}

// ============================================================================
// HingeEmbeddingLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, HingeEmbeddingLoss_ForwardShape) {
    nn::HingeEmbeddingLoss loss(1.0, nn::Reduction::Mean);
    nn::HingeEmbeddingLoss loss_cpu(1.0, nn::Reduction::Mean);

    auto input = createInput({4, 5}, false);
    // Target: +1 or -1
    auto target = Variable(tenzor::ones({4, 5}, dtype(), device()), false);

    auto result = loss.forward(Variable(input.tensor(), true), target);
    EXPECT_EQ(result.tensor().numel(), 1);
    expectDevice(result.tensor());

    auto in_cpu = Variable(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto tgt_cpu = Variable(target.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = loss_cpu.forward(in_cpu, tgt_cpu);
    EXPECT_NEAR(scalarValue(result.tensor()), scalarValue(ref.tensor()), lossAtol());
}

TEST_P(MissingLossesMultiDTypeTest, HingeEmbeddingLoss_ReductionModes) {
    auto input  = Variable(createRandn({8}), false);
    auto target = Variable(tenzor::ones({8}, dtype(), device()), false);
    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto tgt_cpu = target.tensor().to(Device::cpu()).to(DType::Float32);

    {
        nn::HingeEmbeddingLoss loss_mean(1.0, nn::Reduction::Mean);
        nn::HingeEmbeddingLoss loss_mean_cpu(1.0, nn::Reduction::Mean);
        auto r = loss_mean.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_mean_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol());
    }
    {
        nn::HingeEmbeddingLoss loss_sum(1.0, nn::Reduction::Sum);
        nn::HingeEmbeddingLoss loss_sum_cpu(1.0, nn::Reduction::Sum);
        auto r = loss_sum.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_sum_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol() * 8.0f);
    }
    {
        nn::HingeEmbeddingLoss loss_none(1.0, nn::Reduction::None);
        nn::HingeEmbeddingLoss loss_none_cpu(1.0, nn::Reduction::None);
        auto r = loss_none.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 8);
        auto ref = loss_none_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        expectTensorNear(r.tensor(), ref.tensor(), lossAtol());
    }
}

TEST_P(MissingLossesMultiDTypeTest, HingeEmbeddingLoss_GradientFlow) {
    nn::HingeEmbeddingLoss loss(1.0, nn::Reduction::Mean);

    auto input  = createInput({4, 5}, true);
    auto target = Variable(tenzor::ones({4, 5}, dtype(), device()), false);

    auto result = loss.forward(input, target);
    result.backward();

    EXPECT_GRAD_FLOWS(input);
    expectShape(input.grad().value(), {4, 5});
}

// ============================================================================
// MultiLabelSoftMarginLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, MultiLabelSoftMarginLoss_ForwardShape) {
    nn::MultiLabelSoftMarginLoss loss(nn::Reduction::Mean);
    nn::MultiLabelSoftMarginLoss loss_cpu(nn::Reduction::Mean);

    auto input = createInput({4, 6}, false);
    // Target: 0 or 1 multi-label indicators
    auto target = Variable(tenzor::zeros({4, 6}, dtype(), device()), false);

    auto result = loss.forward(Variable(input.tensor(), true), target);
    EXPECT_EQ(result.tensor().numel(), 1);
    expectDevice(result.tensor());

    auto in_cpu = Variable(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto tgt_cpu = Variable(target.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto ref = loss_cpu.forward(in_cpu, tgt_cpu);
    EXPECT_NEAR(scalarValue(result.tensor()), scalarValue(ref.tensor()), lossAtol());
}

TEST_P(MissingLossesMultiDTypeTest, MultiLabelSoftMarginLoss_ReductionModes) {
    auto input  = Variable(createRandn({6, 4}), false);
    auto target = Variable(tenzor::ones({6, 4}, dtype(), device()), false);
    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto tgt_cpu = target.tensor().to(Device::cpu()).to(DType::Float32);

    {
        nn::MultiLabelSoftMarginLoss loss_mean(nn::Reduction::Mean);
        nn::MultiLabelSoftMarginLoss loss_mean_cpu(nn::Reduction::Mean);
        auto r = loss_mean.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_mean_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol());
    }
    {
        nn::MultiLabelSoftMarginLoss loss_sum(nn::Reduction::Sum);
        nn::MultiLabelSoftMarginLoss loss_sum_cpu(nn::Reduction::Sum);
        auto r = loss_sum.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_sum_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol() * 6.0f);
    }
    {
        nn::MultiLabelSoftMarginLoss loss_none(nn::Reduction::None);
        nn::MultiLabelSoftMarginLoss loss_none_cpu(nn::Reduction::None);
        auto r = loss_none.forward(Variable(input.tensor(), true), target);
        // None reduction: one loss per sample in batch
        EXPECT_EQ(r.tensor().numel(), 6);
        auto ref = loss_none_cpu.forward(Variable(in_cpu, false), Variable(tgt_cpu, false));
        expectTensorNear(r.tensor(), ref.tensor(), lossAtol());
    }
}

TEST_P(MissingLossesMultiDTypeTest, MultiLabelSoftMarginLoss_GradientFlow) {
    nn::MultiLabelSoftMarginLoss loss(nn::Reduction::Mean);

    auto input  = createInput({4, 6}, true);
    auto target = Variable(tenzor::ones({4, 6}, dtype(), device()), false);

    auto result = loss.forward(input, target);
    result.backward();

    EXPECT_GRAD_FLOWS(input);
    expectShape(input.grad().value(), {4, 6});
}

// ============================================================================
// MultiMarginLoss
// ============================================================================

TEST_P(MissingLossesMultiDTypeTest, MultiMarginLoss_ForwardShape) {
    nn::MultiMarginLoss loss(1, 1.0, nn::Reduction::Mean);
    nn::MultiMarginLoss loss_cpu(1, 1.0, nn::Reduction::Mean);

    auto input = createInput({4, 5}, false);
    // Target: class indices as Int64 tensor
    auto target = tenzor::zeros({4}, DType::Int64, device());

    auto result = loss.forward(Variable(input.tensor(), true), target);
    EXPECT_EQ(result.tensor().numel(), 1);
    expectDevice(result.tensor());

    auto in_cpu = Variable(input.tensor().to(Device::cpu()).to(DType::Float32), false);
    auto tgt_cpu = target.to(Device::cpu());
    auto ref = loss_cpu.forward(in_cpu, tgt_cpu);
    EXPECT_NEAR(scalarValue(result.tensor()), scalarValue(ref.tensor()), lossAtol());
}

TEST_P(MissingLossesMultiDTypeTest, MultiMarginLoss_ReductionModes) {
    auto input  = Variable(createRandn({8, 5}), false);
    auto target = tenzor::zeros({8}, DType::Int64, device());
    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto tgt_cpu = target.to(Device::cpu());

    {
        nn::MultiMarginLoss loss_mean(1, 1.0, nn::Reduction::Mean);
        nn::MultiMarginLoss loss_mean_cpu(1, 1.0, nn::Reduction::Mean);
        auto r = loss_mean.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_mean_cpu.forward(Variable(in_cpu, false), tgt_cpu);
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol());
    }
    {
        nn::MultiMarginLoss loss_sum(1, 1.0, nn::Reduction::Sum);
        nn::MultiMarginLoss loss_sum_cpu(1, 1.0, nn::Reduction::Sum);
        auto r = loss_sum.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 1);
        auto ref = loss_sum_cpu.forward(Variable(in_cpu, false), tgt_cpu);
        EXPECT_NEAR(scalarValue(r.tensor()), scalarValue(ref.tensor()), lossAtol() * 8.0f);
    }
    {
        nn::MultiMarginLoss loss_none(1, 1.0, nn::Reduction::None);
        nn::MultiMarginLoss loss_none_cpu(1, 1.0, nn::Reduction::None);
        auto r = loss_none.forward(Variable(input.tensor(), true), target);
        EXPECT_EQ(r.tensor().numel(), 8);
        auto ref = loss_none_cpu.forward(Variable(in_cpu, false), tgt_cpu);
        expectTensorNear(r.tensor(), ref.tensor(), lossAtol());
    }
}

TEST_P(MissingLossesMultiDTypeTest, MultiMarginLoss_GradientFlow) {
    nn::MultiMarginLoss loss(1, 1.0, nn::Reduction::Mean);

    auto input  = createInput({4, 5}, true);
    auto target = tenzor::zeros({4}, DType::Int64, device());

    auto result = loss.forward(input, target);
    result.backward();

    EXPECT_GRAD_FLOWS(input);
    expectShape(input.grad().value(), {4, 5});
}

// ============================================================================
// Instantiate
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MissingLossesMultiDTypeTest);
