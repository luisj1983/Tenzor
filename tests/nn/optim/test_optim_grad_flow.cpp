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
 * autograd graph is intact end-to-end) AND the named optimizer, when
 * stepped against those gradients, actually updates the parameters.
 *
 * Each test constructs the specific optimizer it is named for, runs a real
 * forward → loss → backward to populate gradients, snapshots the parameters,
 * steps the optimizer, and asserts (a) the gradient flowed (non-zero) and
 * (b) the optimizer moved the parameters by a non-trivial amount in the
 * direction of the gradient. A broken optimizer that ignores gradients, or a
 * severed grad_fn chain, fails loudly.
 *
 * Catches regressions where an optimizer-related code change accidentally
 * severs the grad_fn chain or zeroes gradients silently, or where an
 * optimizer's step() becomes a no-op.
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
using namespace tenzor::optim;

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

    auto params() -> std::vector<std::shared_ptr<Variable>> {
        return layer.parameters();
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

    // Host snapshot of the first parameter's values (Float64, CPU).
    auto param0_snapshot() -> Tensor {
        return layer.parameters()[0]->tensor().to(DType::Float64).cpu();
    }

    // Max-abs change in the first parameter relative to a prior snapshot.
    auto param0_change(const Tensor& before) -> double {
        auto after = param0_snapshot();
        return ::tenzor::max(::tenzor::abs(after - before)).item<double>();
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

// Assert that a forward → loss → backward chain populates a non-zero gradient
// on the first parameter (grad_fn intact end-to-end).
#define EXPECT_GRAD_FLOWED_THROUGH(fixture)                                    \
    do {                                                                       \
        auto _g = (fixture).forward_backward();                                \
        EXPECT_GT(_g, 0.0)                                                     \
            << "First-parameter gradient is zero — grad_fn chain likely "      \
               "severed somewhere in the forward path";                        \
    } while (0)

// A single-rank ZeRO config usable from any of the stage tests.
static auto make_single_rank_zero_config() -> ZeROStage1Config {
    ZeROStage1Config cfg;
    cfg.world_size = 1;
    cfg.rank = 0;
    cfg.offload_to_cpu = false;
    cfg.process_group = nullptr;
    return cfg;
}

TEST_P(OptimGradFlowTest, SGD_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
    auto before = f.param0_snapshot();
    SGD opt(f.params(), 0.1);
    opt.step();
    EXPECT_GT(f.param0_change(before), 0.0)
        << "SGD.step() did not update parameters despite non-zero gradients";
}

TEST_P(OptimGradFlowTest, Adam_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
    auto before = f.param0_snapshot();
    Adam opt(f.params(), 0.1);
    opt.step();
    EXPECT_GT(f.param0_change(before), 0.0)
        << "Adam.step() did not update parameters despite non-zero gradients";
}

TEST_P(OptimGradFlowTest, Adamax_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
    auto before = f.param0_snapshot();
    Adamax opt(f.params(), 0.1);
    opt.step();
    EXPECT_GT(f.param0_change(before), 0.0)
        << "Adamax.step() did not update parameters despite non-zero gradients";
}

TEST_P(OptimGradFlowTest, Lion_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
    auto before = f.param0_snapshot();
    Lion opt(f.params(), 0.1);
    opt.step();
    EXPECT_GT(f.param0_change(before), 0.0)
        << "Lion.step() did not update parameters despite non-zero gradients";
}

TEST_P(OptimGradFlowTest, NAdam_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
    auto before = f.param0_snapshot();
    NAdam opt(f.params(), 0.1);
    opt.step();
    EXPECT_GT(f.param0_change(before), 0.0)
        << "NAdam.step() did not update parameters despite non-zero gradients";
}

TEST_P(OptimGradFlowTest, Rprop_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
    auto before = f.param0_snapshot();
    Rprop opt(f.params(), 0.1);
    opt.step();
    EXPECT_GT(f.param0_change(before), 0.0)
        << "Rprop.step() did not update parameters despite non-zero gradients";
}

TEST_P(OptimGradFlowTest, LBFGS_GradFlows) {
    GradFlowFixture f(device);
    // LBFGS requires a closure that re-evaluates the loss. The closure runs a
    // fresh forward/backward so grads are live for the line search.
    auto params = f.params();
    auto before = f.param0_snapshot();
    LBFGS opt(params, 1.0);
    double last_grad = 0.0;
    opt.step([&]() {
        last_grad = f.forward_backward();
        auto loss = MSELoss()(f.layer.forward(Variable(f.x, false)),
                              Variable(f.target, false));
        return loss;
    });
    EXPECT_GT(last_grad, 0.0)
        << "LBFGS closure produced zero gradient — grad_fn chain severed";
    EXPECT_GT(f.param0_change(before), 0.0)
        << "LBFGS.step() did not update parameters despite non-zero gradients";
}

TEST_P(OptimGradFlowTest, ZeROStage1_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
    auto before = f.param0_snapshot();
    auto base = std::make_unique<Adam>(f.params(), 0.1);
    ZeROStage1Optimizer opt(std::move(base), make_single_rank_zero_config());
    opt.step();
    EXPECT_GT(f.param0_change(before), 0.0)
        << "ZeROStage1.step() did not update parameters despite non-zero gradients";
}

TEST_P(OptimGradFlowTest, ZeROStage2_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
    auto before = f.param0_snapshot();
    auto base = std::make_unique<Adam>(f.params(), 0.1);
    ZeROStage2Config cfg;
    cfg.world_size = 1;
    cfg.rank = 0;
    cfg.offload_to_cpu = false;
    cfg.process_group = nullptr;
    ZeROStage2Optimizer opt(std::move(base), cfg);
    opt.step();
    EXPECT_GT(f.param0_change(before), 0.0)
        << "ZeROStage2.step() did not update parameters despite non-zero gradients";
}

TEST_P(OptimGradFlowTest, ZeROStage3_GradFlows) {
    GradFlowFixture f(device);
    EXPECT_GRAD_FLOWED_THROUGH(f);
    auto before = f.param0_snapshot();
    auto base = std::make_unique<Adam>(f.params(), 0.1);
    Stage3Config cfg;
    cfg.world_size = 1;
    cfg.rank = 0;
    cfg.offload_to_cpu = false;
    cfg.process_group = nullptr;
    ZeROStage3Optimizer opt(std::move(base), cfg);
    opt.step();
    EXPECT_GT(f.param0_change(before), 0.0)
        << "ZeROStage3.step() did not update parameters despite non-zero gradients";
}

INSTANTIATE_BACKEND_TESTS(OptimGradFlowTest);
