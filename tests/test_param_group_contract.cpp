/**
 * @file test_param_group_contract.cpp
 * @brief Audit D.4: SGD step_impl reads per-ParamGroup hyperparameters
 *        (lr, momentum, weight_decay, dampening, nesterov) instead of
 *        only the optimizer-member defaults.
 */

#include <gtest/gtest.h>

#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"

using namespace tenzor;

namespace {

class ParamGroupContractTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(ParamGroupContractTest, SGDPerGroupLearningRate) {
    // Two parameters, each in its own group with a different lr.
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);
    p1->set_grad(tenzor::ones({1}, DType::Float32, Device::cpu()));
    p2->set_grad(tenzor::ones({1}, DType::Float32, Device::cpu()));

    // Group 1: lr=0.1, Group 2: lr=0.5
    optim::ParamGroup g1{{p1}, /*lr=*/0.1, /*weight_decay=*/0.0};
    optim::ParamGroup g2{{p2}, /*lr=*/0.5, /*weight_decay=*/0.0};
    optim::SGD sgd({g1, g2});

    sgd.step();

    // Each parameter should move by -lr * grad = -lr * 1.
    EXPECT_NEAR(p1->tensor().data<float>()[0], -0.1f, 1e-6);
    EXPECT_NEAR(p2->tensor().data<float>()[0], -0.5f, 1e-6);
}

TEST_F(ParamGroupContractTest, SGDPerGroupWeightDecay) {
    auto p1_tensor = tenzor::ones({1}, DType::Float32, Device::cpu());
    auto p2_tensor = tenzor::ones({1}, DType::Float32, Device::cpu());
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);
    auto zero_grad = tenzor::zeros({1}, DType::Float32, Device::cpu());
    p1->set_grad(zero_grad);
    p2->set_grad(zero_grad);

    // g1 has weight_decay=0, g2 has weight_decay=0.1. With zero gradients,
    // p1 stays at 1.0 and p2 should drift toward zero by -lr * wd * p.
    optim::ParamGroup g1{{p1}, /*lr=*/0.1, /*weight_decay=*/0.0};
    optim::ParamGroup g2{{p2}, /*lr=*/0.1, /*weight_decay=*/0.1};
    optim::SGD sgd({g1, g2});

    sgd.step();

    EXPECT_FLOAT_EQ(p1->tensor().data<float>()[0], 1.0f);
    // p2 -= lr * (grad + wd * p) = 0.1 * (0 + 0.1 * 1) = 0.01
    EXPECT_NEAR(p2->tensor().data<float>()[0], 1.0f - 0.01f, 1e-6);
}

TEST_F(ParamGroupContractTest, SGDPerGroupMomentumDifferentTrajectories) {
    // Two params with the same gradient but different momentum.
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);
    auto grad = tenzor::ones({1}, DType::Float32, Device::cpu());
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

    auto v1 = p1->tensor().data<float>()[0];
    auto v2 = p2->tensor().data<float>()[0];
    // Without momentum: -0.1 * 2 = -0.2
    EXPECT_NEAR(v1, -0.2f, 1e-5);
    // With momentum=0.9, dampening=0:
    //   v_0 = 0; v_1 = 0.9*0 + 1*1 = 1.0; p_1 = 0 - 0.1*1.0 = -0.1
    //   v_2 = 0.9*1.0 + 1*1 = 1.9; p_2 = -0.1 - 0.1*1.9 = -0.29
    EXPECT_NEAR(v2, -0.29f, 1e-5);
}

TEST_F(ParamGroupContractTest, SGDFlatParamListUnchanged) {
    // When constructed from a flat parameter list (no groups),
    // `find_group_for_param` returns nullptr and SGD uses its own
    // members exactly as before D.4.
    auto p_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto p = std::make_shared<Variable>(p_tensor, /*requires_grad=*/true);
    p->set_grad(tenzor::ones({1}, DType::Float32, Device::cpu()));

    optim::SGD sgd({p}, /*lr=*/0.1);
    sgd.step();
    EXPECT_NEAR(p->tensor().data<float>()[0], -0.1f, 1e-6);
}

// ----- Adam D.4 coverage --------------------------------------------------

TEST_F(ParamGroupContractTest, AdamPerGroupLearningRate) {
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);
    p1->set_grad(tenzor::ones({1}, DType::Float32, Device::cpu()));
    p2->set_grad(tenzor::ones({1}, DType::Float32, Device::cpu()));

    optim::ParamGroup g1{{p1}, /*lr=*/0.01, 0.0};
    optim::ParamGroup g2{{p2}, /*lr=*/0.10, 0.0};
    optim::Adam adam({g1, g2});

    adam.step();
    auto v1 = p1->tensor().data<float>()[0];
    auto v2 = p2->tensor().data<float>()[0];

    // At step 1 with default betas, Adam's first-step update on a unit
    // gradient and zero moments simplifies to -lr (the bias-correction
    // factors exactly cancel for step 1, modulo the eps). Verify the
    // ratio between groups is the expected lr ratio.
    EXPECT_LT(v1, 0.0f);
    EXPECT_LT(v2, 0.0f);
    EXPECT_NEAR(v2 / v1, 10.0f, 1e-3) << "per-group lr ratio incorrect";
}

TEST_F(ParamGroupContractTest, AdamPerGroupBetas) {
    // Two params with different beta1.  Under *alternating-sign*
    // gradients the smoothing of the first moment estimate diverges
    // between low- and high-beta1: a high-beta1 (heavy smoothing)
    // group dampens oscillations, so |v_high| < |v_low| after a few
    // alternating steps.
    auto p1_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto p2_tensor = tenzor::zeros({1}, DType::Float32, Device::cpu());
    auto p1 = std::make_shared<Variable>(p1_tensor, /*requires_grad=*/true);
    auto p2 = std::make_shared<Variable>(p2_tensor, /*requires_grad=*/true);

    optim::ParamGroup g1{{p1}, /*lr=*/0.01, 0.0};
    g1.beta1 = 0.1;          // light smoothing
    optim::ParamGroup g2{{p2}, /*lr=*/0.01, 0.0};
    g2.beta1 = 0.99;         // heavy smoothing
    optim::Adam adam({g1, g2});

    // Alternating-sign gradients over 6 steps.
    auto pos = tenzor::ones({1}, DType::Float32, Device::cpu());
    auto neg = pos * full({1}, -1.0, DType::Float32, Device::cpu());
    auto neg_t = neg;  // pre-build to reuse
    for (int s = 0; s < 6; ++s) {
        auto g = (s % 2 == 0) ? pos : neg_t;
        p1->set_grad(g);
        p2->set_grad(g);
        adam.step();
    }
    auto v1 = p1->tensor().data<float>()[0];
    auto v2 = p2->tensor().data<float>()[0];
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

}  // namespace
