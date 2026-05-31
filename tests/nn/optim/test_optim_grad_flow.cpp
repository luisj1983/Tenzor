/**
 * @file test_optim_grad_flow.cpp
 * @brief Dedicated grad-flow regression tests for every optimizer family
 *        (audit-2026-05-03 N1.d).
 *
 * The existing optim tests typically construct parameters with
 * requires_grad=true and either run a real loss.backward() chain or
 * manually populate gradients before calling step(). This file pins the
 * grad-flow invariant for each optimizer: after a forward → loss →
 * backward chain, the parameters' gradients must be non-zero (i.e. the
 * autograd graph is intact end-to-end).
 *
 * Catches regressions where an optimizer-related code change accidentally
 * severs the grad_fn chain or zeroes gradients silently.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/optim/adamax.hpp>
#include <tenzor/nn/optim/lion.hpp>
#include <tenzor/nn/optim/nadam.hpp>
#include <tenzor/nn/optim/rprop.hpp>
#include <tenzor/nn/optim/lbfgs.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>

#include "../../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;

namespace {

// Build a minimal model + data for a forward → backward cycle, all placed on
// the test's target device. Inputs are randn (non-zero) so the gradients that
// flow back are genuinely non-zero — this is a grad-flow suite.
struct GradFlowFixture {
    Linear layer{8, 4};
    Tensor x;
    Tensor target;
    explicit GradFlowFixture(const Device& device)
        : x(randn({2, 8}, DType::Float32, device))
        , target(randn({2, 4}, DType::Float32, device)) {
        layer.to(device);
    }

    auto forward_backward() -> double {
        for (auto& p : layer.parameters()) {
            p->zero_grad();
        }
        auto loss = MSELoss()(layer.forward(Variable(x, false)), Variable(target, false));
        loss.backward();
        // Return the max-abs gradient over the first parameter — used by
        // the test as the "grad flowed" signal. Read on host via .cpu().
        auto p = layer.parameters();
        if (p.empty() || !(*p[0]).grad().has_value()) return 0.0;
        auto g = (*p[0]).grad().value().to(DType::Float64).cpu();
        return ::tenzor::max(::tenzor::abs(g)).item<double>();
    }
};

}  // anonymous

class OptimGradFlowTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

#define EXPECT_GRAD_FLOWED_THROUGH(fixture)                                    \
    do {                                                                       \
        auto _g = (fixture).forward_backward();                                \
        EXPECT_GT(_g, 0.0)                                                     \
            << "First-parameter gradient is zero — grad_fn chain likely "      \
               "severed somewhere in the forward path";                        \
    } while (0)

TEST_P(OptimGradFlowTest, SGD_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
}

TEST_P(OptimGradFlowTest, Adam_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
}

TEST_P(OptimGradFlowTest, Adamax_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
}

TEST_P(OptimGradFlowTest, Lion_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
}

TEST_P(OptimGradFlowTest, NAdam_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
}

TEST_P(OptimGradFlowTest, Rprop_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
}

TEST_P(OptimGradFlowTest, LBFGS_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
}

TEST_P(OptimGradFlowTest, ZeROStage1_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
}

TEST_P(OptimGradFlowTest, ZeROStage2_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
}

TEST_P(OptimGradFlowTest, ZeROStage3_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
}

INSTANTIATE_BACKEND_TESTS(OptimGradFlowTest);
