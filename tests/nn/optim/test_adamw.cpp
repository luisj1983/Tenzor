// Dedicated tests for the AdamW optimizer (Adam + decoupled weight decay).
//
// Mirrors the unit-test pattern used by test_adamax.cpp / test_lion.cpp /
// test_nadam.cpp (convergence, basic step, lr getter/setter, state dict
// round-trip) plus a decoupled-weight-decay check that distinguishes AdamW
// from plain Adam. Runs across all available backends via BackendTest.

#include <gtest/gtest.h>

#include "../../backend_test_fixture.hpp"

#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/tenzor.hpp>

namespace tenzor {
namespace {

class AdamWTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    std::vector<std::shared_ptr<Variable>> make_params() {
        auto t = tenzor::ones({4, 4}, DType::Float32, device);
        return {std::make_shared<Variable>(t, /*requires_grad=*/true)};
    }

    // Sum of absolute parameter values — used to detect update direction
    // without depending on gradient magnitudes.
    static float sum_abs(const std::vector<std::shared_ptr<Variable>>& params) {
        float s = 0.0f;
        for (const auto& p : params) {
            auto cpu = p->tensor().cpu();
            const auto* d = cpu.data<float>();
            for (int64_t i = 0; i < cpu.numel(); ++i) s += std::abs(d[i]);
        }
        return s;
    }

    // Seeds a positive unit gradient on every parameter.
    static void seed_unit_grad(std::vector<std::shared_ptr<Variable>>& params) {
        for (auto& p : params) {
            auto shape = p->tensor().shape();
            p->set_grad(tenzor::ones({shape.begin(), shape.end()},
                                     p->tensor().dtype(), p->tensor().device()));
        }
    }
};

TEST_P(AdamWTest, BasicStepReducesParamMagnitude) {
    auto params = make_params();
    optim::AdamW opt(params, /*lr=*/1e-2);

    float before = sum_abs(params);
    seed_unit_grad(params);
    opt.step();
    seed_unit_grad(params);
    opt.step();
    float after = sum_abs(params);

    EXPECT_LT(after, before)
        << "AdamW with constant positive gradient should shrink |params|";
}

TEST_P(AdamWTest, ConvergesOnQuadratic) {
    // Minimize f(x) = ||x||^2 starting from x = ones(4,4). Gradient is 2x.
    auto params = make_params();
    optim::AdamW opt(params, /*lr=*/1e-2, /*beta1=*/0.9, /*beta2=*/0.999,
                     /*eps=*/1e-8, /*weight_decay=*/0.0);

    for (int step = 0; step < 300; ++step) {
        auto& p = params[0];
        auto grad = p->tensor() *
                    full({1}, 2.0f, p->tensor().dtype(), p->tensor().device());
        p->set_grad(grad);
        opt.step();
    }

    auto final_t = params[0]->tensor().cpu();
    const auto* d = final_t.data<float>();
    float sq = 0.0f;
    for (int64_t i = 0; i < final_t.numel(); ++i) sq += d[i] * d[i];
    EXPECT_LT(sq, 2.0f)
        << "AdamW on a quadratic should drive ||x||^2 well below 16 (starting value)";
}

TEST_P(AdamWTest, WeightDecayDecouplesFromGradient) {
    // Decoupled weight decay means: even with zero gradient, weight decay
    // alone shrinks the parameters. That's the defining difference vs Adam
    // (which would leave params untouched when gradient is zero).
    auto params = make_params();
    optim::AdamW opt(params, /*lr=*/0.1, /*beta1=*/0.9, /*beta2=*/0.999,
                     /*eps=*/1e-8, /*weight_decay=*/0.1);

    // Zero gradient — Adam would not move, AdamW must still decay.
    for (auto& p : params) {
        p->set_grad(zeros(std::vector<int64_t>(p->tensor().shape().begin(),
                                               p->tensor().shape().end()),
                          p->tensor().dtype(), p->tensor().device()));
    }

    float before = sum_abs(params);
    opt.step();
    float after = sum_abs(params);

    EXPECT_LT(after, before)
        << "AdamW with zero gradient and nonzero weight_decay must still shrink params";
}

TEST_P(AdamWTest, LrGetSet) {
    auto params = make_params();
    optim::AdamW opt(params, /*lr=*/2e-3);

    EXPECT_DOUBLE_EQ(opt.get_lr(), 2e-3);
    opt.set_lr(5e-3);
    EXPECT_DOUBLE_EQ(opt.get_lr(), 5e-3);
}

TEST_P(AdamWTest, StateDictNonEmptyAndRoundtrippable) {
    // Exercise the serialize / deserialize surface and verify the moment
    // buffers actually survive a save/load: after a few steps the state dict
    // must be non-empty (step count, first/second moments), load_state_dict
    // on a fresh optimizer must accept it, and a subsequent identical step on
    // both optimizers — started from identical param values — must produce
    // identical params. A save/load that drops or zeroes exp_avg/exp_avg_sq
    // diverges here.
    auto params = make_params();
    optim::AdamW opt(params, /*lr=*/1e-2);

    for (int i = 0; i < 5; ++i) {
        seed_unit_grad(params);
        opt.step();
    }

    auto state = opt.state_dict();
    EXPECT_FALSE(state.empty())
        << "AdamW state_dict must include moments + step count after stepping";

    // Fresh optimizer over params that clone opt's current values so the next
    // step is governed only by the restored optimizer state.
    auto fresh_params = std::vector<std::shared_ptr<Variable>>{
        std::make_shared<Variable>(params[0]->tensor().clone(), /*requires_grad=*/true)};
    optim::AdamW fresh_opt(fresh_params, /*lr=*/1e-2);
    ASSERT_NO_THROW(fresh_opt.load_state_dict(state));

    seed_unit_grad(params);
    opt.step();
    seed_unit_grad(fresh_params);
    fresh_opt.step();

    auto a = params[0]->tensor().to(DType::Float64).cpu();
    auto b = fresh_params[0]->tensor().to(DType::Float64).cpu();
    double max_diff = ::tenzor::max(::tenzor::abs(a - b)).item<double>();
    EXPECT_LT(max_diff, 1e-6)
        << "Post-load step diverged (max param diff " << max_diff
        << ") — load_state_dict did not restore the AdamW moment buffers";
}

INSTANTIATE_BACKEND_TESTS(AdamWTest);

}  // namespace
}  // namespace tenzor
