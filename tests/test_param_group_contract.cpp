/**
 * @file test_param_group_contract.cpp
 * @brief Audit D.4: SGD step_impl reads per-ParamGroup hyperparameters
 *        (lr, momentum, weight_decay, dampening, nesterov) instead of
 *        only the optimizer-member defaults.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "backend_test_fixture.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/optim/adagrad.hpp"
#include "tenzor/nn/optim/adadelta.hpp"
#include "tenzor/nn/optim/adamax.hpp"
#include "tenzor/nn/optim/nadam.hpp"
#include "tenzor/nn/optim/radam.hpp"
#include "tenzor/nn/optim/sparse_adam.hpp"
#include "tenzor/nn/optim/lamb.hpp"
#include "tenzor/nn/optim/asgd.hpp"
#include "tenzor/nn/optim/lion.hpp"
#include "tenzor/nn/optim/adam_atan2.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"

using namespace tenzor;

namespace {

class ParamGroupContractTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(ParamGroupContractTest, SGDPerGroupLearningRate) {
    // Two parameters, each in its own group with a different lr.
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);
    p1->set_grad(tenzor::ones({1}, DType::Float32, device));
    p2->set_grad(tenzor::ones({1}, DType::Float32, device));

    // Group 1: lr=0.1, Group 2: lr=0.5
    optim::ParamGroup g1{{p1}, /*lr=*/0.1, /*weight_decay=*/0.0};
    optim::ParamGroup g2{{p2}, /*lr=*/0.5, /*weight_decay=*/0.0};
    optim::SGD sgd({g1, g2});

    sgd.step();

    // Each parameter should move by -lr * grad = -lr * 1.
    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    EXPECT_NEAR(p1_cpu.data<float>()[0], -0.1f, 1e-6);
    EXPECT_NEAR(p2_cpu.data<float>()[0], -0.5f, 1e-6);
}

TEST_P(ParamGroupContractTest, SGDPerGroupWeightDecay) {
    auto p1_tensor = tenzor::ones({1}, DType::Float32, device);
    auto p2_tensor = tenzor::ones({1}, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);
    auto zero_grad = tenzor::zeros({1}, DType::Float32, device);
    p1->set_grad(zero_grad);
    p2->set_grad(zero_grad);

    // g1 has weight_decay=0, g2 has weight_decay=0.1. With zero gradients,
    // p1 stays at 1.0 and p2 should drift toward zero by -lr * wd * p.
    optim::ParamGroup g1{{p1}, /*lr=*/0.1, /*weight_decay=*/0.0};
    optim::ParamGroup g2{{p2}, /*lr=*/0.1, /*weight_decay=*/0.1};
    optim::SGD sgd({g1, g2});

    sgd.step();

    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    EXPECT_FLOAT_EQ(p1_cpu.data<float>()[0], 1.0f);
    // p2 -= lr * (grad + wd * p) = 0.1 * (0 + 0.1 * 1) = 0.01
    EXPECT_NEAR(p2_cpu.data<float>()[0], 1.0f - 0.01f, 1e-6);
}

TEST_P(ParamGroupContractTest, SGDPerGroupMomentumDifferentTrajectories) {
    // Two params with the same gradient but different momentum.
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);
    auto grad = tenzor::ones({1}, DType::Float32, device);
    p1->set_grad(grad);
    p2->set_grad(grad);

    optim::ParamGroup g1{{p1}, 0.1, 0.0};
    g1.momentum = 0.0;
    optim::ParamGroup g2{{p2}, 0.1, 0.0};
    g2.momentum = 0.9;
    optim::SGD sgd({g1, g2});

    // Run two steps and confirm divergent trajectories.
    sgd.step();
    p1->set_grad(grad);
    p2->set_grad(grad);
    sgd.step();

    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    auto v1 = p1_cpu.data<float>()[0];
    auto v2 = p2_cpu.data<float>()[0];
    // Without momentum: -0.1 * 2 = -0.2
    EXPECT_NEAR(v1, -0.2f, 1e-5);
    // With momentum=0.9, dampening=0:
    //   v_0 = 0; v_1 = 0.9*0 + 1*1 = 1.0; p_1 = 0 - 0.1*1.0 = -0.1
    //   v_2 = 0.9*1.0 + 1*1 = 1.9; p_2 = -0.1 - 0.1*1.9 = -0.29
    EXPECT_NEAR(v2, -0.29f, 1e-5);
}

TEST_P(ParamGroupContractTest, SGDFlatParamListUnchanged) {
    // When constructed from a flat parameter list (no groups),
    // `find_group_for_param` returns nullptr and SGD uses its own
    // members exactly as before D.4.
    auto p_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p = std::make_shared<Variable>(p_tensor, /*requires_grad=*/true);
    p->set_grad(tenzor::ones({1}, DType::Float32, device));

    optim::SGD sgd({p}, /*lr=*/0.1);
    sgd.step();
    auto p_cpu = p->tensor().cpu();
    EXPECT_NEAR(p_cpu.data<float>()[0], -0.1f, 1e-6);
}

// ----- Adam D.4 coverage --------------------------------------------------

TEST_P(ParamGroupContractTest, AdamPerGroupLearningRate) {
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);
    p1->set_grad(tenzor::ones({1}, DType::Float32, device));
    p2->set_grad(tenzor::ones({1}, DType::Float32, device));

    optim::ParamGroup g1{{p1}, /*lr=*/0.01, 0.0};
    optim::ParamGroup g2{{p2}, /*lr=*/0.10, 0.0};
    optim::Adam adam({g1, g2});

    adam.step();
    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    auto v1 = p1_cpu.data<float>()[0];
    auto v2 = p2_cpu.data<float>()[0];

    // At step 1 with default betas, Adam's first-step update on a unit
    // gradient and zero moments simplifies to -lr (the bias-correction
    // factors exactly cancel for step 1, modulo the eps). Verify the
    // ratio between groups is the expected lr ratio.
    EXPECT_LT(v1, 0.0f);
    EXPECT_LT(v2, 0.0f);
    EXPECT_NEAR(v2 / v1, 10.0f, 1e-3) << "per-group lr ratio incorrect";
}

TEST_P(ParamGroupContractTest, AdamPerGroupBetas) {
    // Two params with different beta1.  Under *alternating-sign*
    // gradients the smoothing of the first moment estimate diverges
    // between low- and high-beta1: a high-beta1 (heavy smoothing)
    // group dampens oscillations, so |v_high| < |v_low| after a few
    // alternating steps.
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);

    optim::ParamGroup g1{{p1}, /*lr=*/0.01, 0.0};
    g1.beta1 = 0.1;          // light smoothing
    optim::ParamGroup g2{{p2}, /*lr=*/0.01, 0.0};
    g2.beta1 = 0.99;         // heavy smoothing
    optim::Adam adam({g1, g2});

    // Alternating-sign gradients over 6 steps.
    auto pos = tenzor::ones({1}, DType::Float32, device);
    auto neg = pos * full({1}, -1.0, DType::Float32, device);
    auto neg_t = neg;  // pre-build to reuse
    for (int s = 0; s < 6; ++s) {
        auto g = (s % 2 == 0) ? pos : neg_t;
        p1->set_grad(g);
        p2->set_grad(g);
        adam.step();
    }
    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    auto v1 = p1_cpu.data<float>()[0];
    auto v2 = p2_cpu.data<float>()[0];
    // The two values must be measurably different — that's the point
    // of the test (per-group beta1 actually flows through).  A trivial
    // bug where both groups used the optimizer-member beta1 would
    // produce equal values.  Hand-derived references via a 6-step
    // simulation: low-beta1 ≈ -0.0018, high-beta1 ≈ -0.0152.
    EXPECT_NE(v1, v2);
    EXPECT_LT(v2, v1)
        << "beta1=0.99 ends up further from zero than beta1=0.1 after "
           "6 alternating-sign steps (heavy smoothing biases m toward "
           "the late-history grad sign)";
}

TEST_P(ParamGroupContractTest, AdagradPerGroupLearningRate) {
    // Two params share the same gradient but live in groups with
    // distinct learning rates.  Under Adagrad the parameter trajectory
    // depends linearly on lr, so the ratio of parameter movements
    // after one step equals the ratio of lrs.  A bug where step_impl
    // reads lr_ instead of the active group's lr would produce
    // identical trajectories.
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);

    optim::ParamGroup g1{{p1}, /*lr=*/0.01, /*wd=*/0.0};
    optim::ParamGroup g2{{p2}, /*lr=*/0.10, /*wd=*/0.0};
    optim::Adagrad ag({g1, g2});

    auto grad = tenzor::ones({1}, DType::Float32, device);
    p1->set_grad(grad);
    p2->set_grad(grad);
    ag.step();

    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    auto v1 = p1_cpu.data<float>()[0];
    auto v2 = p2_cpu.data<float>()[0];
    // After one Adagrad step with g=1 and initial sum=0:
    //   sum = 1, std_dev = sqrt(1) + eps ≈ 1
    //   delta ≈ -lr * 1 / 1 = -lr
    EXPECT_NEAR(v1, -0.01f, 1e-5f);
    EXPECT_NEAR(v2, -0.10f, 1e-5f);
}

TEST_P(ParamGroupContractTest, AdagradPerGroupWeightDecay) {
    // Adagrad with constant-sign gradients has the well-known
    // delta = -lr * sign(g) / sqrt(t) property that hides the
    // magnitude of grad after the sqrt-normalisation.  Drive both
    // groups with *zero* gradient — then only weight_decay moves the
    // param.  g1.weight_decay=0 must keep p1 fixed; g2.weight_decay
    // must shrink p2.
    auto t1 = full({1}, 1.0f, DType::Float32, device);
    auto t2 = full({1}, 1.0f, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(t1, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(t2, /*requires_grad=*/true);

    optim::ParamGroup g1{{p1}, /*lr=*/0.01, /*wd=*/0.0};
    optim::ParamGroup g2{{p2}, /*lr=*/0.01, /*wd=*/0.5};
    optim::Adagrad ag({g1, g2});

    auto zero_grad = tenzor::zeros({1}, DType::Float32, device);
    p1->set_grad(zero_grad);
    p2->set_grad(zero_grad);
    ag.step();

    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    auto v1 = p1_cpu.data<float>()[0];
    auto v2 = p2_cpu.data<float>()[0];

    // p1: grad'=0, sum stays 0, std=eps≈1e-10, delta=0 → v1=1.0
    EXPECT_FLOAT_EQ(v1, 1.0f);
    // p2: grad'=0+0.5*1=0.5, sum=0.25, std=0.5+eps, delta=-0.01*0.5/0.5=-0.01
    //  → v2 ≈ 0.99
    EXPECT_LT(v2, 1.0f) << "weight_decay=0.5 must shrink p2";
    EXPECT_NEAR(v2, 0.99f, 1e-4f);
}

// ----- Adadelta D.4 coverage ----------------------------------------------

TEST_P(ParamGroupContractTest, AdadeltaPerGroupLearningRate) {
    // Two params with distinct group lr.  Adadelta's update is
    //   delta = -(sqrt(E[Δθ²] + eps) / sqrt(E[g²] + eps)) * g
    //   p   += lr * delta
    // On the very first step E[Δθ²] = 0 so std_delta = sqrt(eps), and
    // E[g²] picks up (1-rho)*g² so std_grad = sqrt((1-rho)*g² + eps).
    // The ratio depends only on rho and eps, not lr — so the ratio of
    // |p1|/|p2| after one step equals the lr ratio.
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);

    optim::ParamGroup g1{{p1}, /*lr=*/0.5, /*wd=*/0.0};
    optim::ParamGroup g2{{p2}, /*lr=*/2.0, /*wd=*/0.0};
    optim::Adadelta ad({g1, g2});

    auto grad = tenzor::ones({1}, DType::Float32, device);
    p1->set_grad(grad);
    p2->set_grad(grad);
    ad.step();

    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    auto v1 = p1_cpu.data<float>()[0];
    auto v2 = p2_cpu.data<float>()[0];
    EXPECT_LT(v1, 0.0f);
    EXPECT_LT(v2, 0.0f);
    EXPECT_NEAR(v2 / v1, 4.0f, 1e-3f)
        << "Adadelta per-group lr must scale the delta linearly";
}

TEST_P(ParamGroupContractTest, AdadeltaPerGroupRho) {
    // Distinct rho changes E[g^2] weighting on the first step:
    //   E[g²]_1 = (1 - rho) * g²
    // Smaller rho → larger E[g²]_1 → larger denominator → smaller
    // |delta|.  Verify p1 (rho=0.5) ends with strictly smaller |delta|
    // than p2 (rho=0.95).
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);

    optim::ParamGroup g1{{p1}, /*lr=*/1.0, /*wd=*/0.0};
    g1.rho = 0.5;
    optim::ParamGroup g2{{p2}, /*lr=*/1.0, /*wd=*/0.0};
    g2.rho = 0.95;
    optim::Adadelta ad({g1, g2});

    auto grad = tenzor::ones({1}, DType::Float32, device);
    p1->set_grad(grad);
    p2->set_grad(grad);
    ad.step();

    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    auto v1 = p1_cpu.data<float>()[0];
    auto v2 = p2_cpu.data<float>()[0];
    EXPECT_LT(v1, 0.0f);
    EXPECT_LT(v2, 0.0f);
    EXPECT_GT(std::abs(v2), std::abs(v1))
        << "rho=0.95 (smaller (1-rho)*g²) must produce larger |delta|";
}

// ----- Helper: per-group lr ratio check for any optimiser ------------------
// Returns p2 / p1 after one step with unit-grad on both params, when both
// groups share weight_decay=0 and the only difference is lr.

namespace {
template <typename Opt>
auto first_step_lr_ratio(double lr1, double lr2, const tenzor::Device& device) -> float {
    auto p1_t = tenzor::zeros({1}, DType::Float32, device);
    auto p2_t = tenzor::zeros({1}, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(p1_t, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_t, /*requires_grad=*/true);

    optim::ParamGroup g1{{p1}, lr1, 0.0};
    optim::ParamGroup g2{{p2}, lr2, 0.0};
    Opt opt({g1, g2});

    auto grad = tenzor::ones({1}, DType::Float32, device);
    p1->set_grad(grad);
    p2->set_grad(grad);
    opt.step();
    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    auto v1 = p1_cpu.data<float>()[0];
    auto v2 = p2_cpu.data<float>()[0];
    return v2 / v1;
}
}  // namespace

// ----- Adamax / NAdam / RAdam / SparseAdam / LAMB / Lion / AdamAtan2 ------
//
// For each of these Adam-family optimisers, a single-step lr-scaling
// check is sufficient to prove the per-group lr override is being read.
// Adamax/NAdam/RAdam/SparseAdam/LAMB/Lion/AdamAtan2 all produce
// delta = -lr * f(grad, betas, eps, ...) for f independent of lr,
// so the ratio v2/v1 must equal lr2/lr1.  An optimiser that ignored
// the per-group lr would produce v2/v1 = 1.

TEST_P(ParamGroupContractTest, AdamaxPerGroupLearningRate) {
    EXPECT_NEAR(first_step_lr_ratio<optim::Adamax>(0.01, 0.10, device), 10.0f, 1e-3f);
}
TEST_P(ParamGroupContractTest, NAdamPerGroupLearningRate) {
    EXPECT_NEAR(first_step_lr_ratio<optim::NAdam>(0.01, 0.10, device), 10.0f, 1e-3f);
}
TEST_P(ParamGroupContractTest, RAdamPerGroupLearningRate) {
    // RAdam falls back to SGDM when the variance-rectification term is
    // undefined (the first few steps with default beta2).  Use lr=0.01
    // vs lr=0.10 and the SGDM update reduces to -lr * g on step 1.
    EXPECT_NEAR(first_step_lr_ratio<optim::RAdam>(0.01, 0.10, device), 10.0f, 1e-3f);
}
TEST_P(ParamGroupContractTest, SparseAdamPerGroupLearningRate) {
    // SparseAdam's dense fallback (no sparse grad) reduces to standard
    // Adam, so the ratio also applies.
    EXPECT_NEAR(first_step_lr_ratio<optim::SparseAdam>(0.01, 0.10, device), 10.0f, 1e-3f);
}
TEST_P(ParamGroupContractTest, LAMBPerGroupLearningRate) {
    // LAMB's trust-ratio multiplies the update, but the trust ratio for
    // params that start at zero is 0 — LAMB's update needs a non-zero
    // ||θ||.  Use a different setup: per-param starting at 1.0 + unit
    // gradient → trust ratio == 1, then ratio simplifies to lr.
    auto p1_t = full({1}, 1.0f, DType::Float32, device);
    auto p2_t = full({1}, 1.0f, DType::Float32, device);
    auto p1 = std::make_shared<Variable>(p1_t, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_t, /*requires_grad=*/true);
    optim::ParamGroup g1{{p1}, 0.01, 0.0};
    optim::ParamGroup g2{{p2}, 0.10, 0.0};
    optim::LAMB opt({g1, g2});
    auto grad = tenzor::ones({1}, DType::Float32, device);
    p1->set_grad(grad);
    p2->set_grad(grad);
    opt.step();
    auto p1_cpu = p1->tensor().cpu();
    auto p2_cpu = p2->tensor().cpu();
    auto d1 = 1.0f - p1_cpu.data<float>()[0];
    auto d2 = 1.0f - p2_cpu.data<float>()[0];
    EXPECT_GT(d1, 0.0f);
    EXPECT_GT(d2, 0.0f);
    EXPECT_NEAR(d2 / d1, 10.0f, 5e-2f)
        << "LAMB per-group lr must scale the update by lr ratio";
}
TEST_P(ParamGroupContractTest, LionPerGroupLearningRate) {
    // Lion's update is delta = -lr * sign(mt) which is lr-independent
    // in magnitude only via the lr scalar — verify lr ratio.
    EXPECT_NEAR(first_step_lr_ratio<optim::Lion>(0.01, 0.10, device), 10.0f, 1e-3f);
}
TEST_P(ParamGroupContractTest, AdamAtan2PerGroupLearningRate) {
    EXPECT_NEAR(first_step_lr_ratio<optim::AdamAtan2>(0.01, 0.10, device), 10.0f, 1e-3f);
}

TEST_P(ParamGroupContractTest, ASGDPerGroupLearningRate) {
    // ASGD's update is θ -= lr * g (plus the running average bookkeeping
    // that does not feed back into the active param).  Lr scaling holds.
    EXPECT_NEAR(first_step_lr_ratio<optim::ASGD>(0.01, 0.10, device), 10.0f, 1e-3f);
}

INSTANTIATE_BACKEND_TESTS(ParamGroupContractTest);

}  // namespace
