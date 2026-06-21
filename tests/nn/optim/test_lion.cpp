// Tests for the Lion (EvoLved Sign Momentum) optimizer.
// Cross-backend. Verifies convergence on a small quadratic, lr get/set,
// and state_dict round-trip.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/lion.hpp>
#include <tenzor/ops/creation.hpp>

#include "../../backend_test_fixture.hpp"

namespace tenzor {
namespace {

class LionTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // One 4x4 parameter initialized to ones, on the test device, Float32.
    std::vector<std::shared_ptr<Variable>> make_params() {
        auto t = tenzor::ones({4, 4}, DType::Float32, device);
        auto param = std::make_shared<Variable>(t, /*requires_grad=*/true);
        return {param};
    }

    // Set a constant gradient of ones on every parameter, then step once.
    // Returns the sum of |param| *before* the step so the caller can
    // observe a decrease across calls.
    float step_with_unit_grad(std::vector<std::shared_ptr<Variable>>& params,
                              optim::Optimizer& opt) {
        float sum = 0.0f;
        for (auto& p : params) {
            auto cpu_t = p->tensor().to(Device::cpu());
            const auto* d = cpu_t.data<float>();
            for (int64_t i = 0; i < cpu_t.numel(); ++i) sum += std::abs(d[i]);
        }
        for (auto& p : params) {
            auto shape = p->tensor().shape();
            p->set_grad(tenzor::ones({shape.begin(), shape.end()},
                                     p->tensor().dtype(), device));
        }
        opt.step();
        return sum;
    }
};

TEST_P(LionTest, BasicStepReducesLoss) {
    // Params start at ones, gradient is constantly +1, so Lion should
    // monotonically push every element toward zero. The |sum| must shrink
    // at every step.
    auto params = make_params();
    optim::Lion opt(params, /*lr=*/0.01);

    float before = step_with_unit_grad(params, opt);
    float after  = step_with_unit_grad(params, opt);

    EXPECT_LT(after, before)
        << "Lion should reduce |param| sum when gradient is constant positive";
}

TEST_P(LionTest, LrGetSet) {
    auto params = make_params();
    optim::Lion opt(params, /*lr=*/1e-4);

    EXPECT_DOUBLE_EQ(opt.get_lr(), 1e-4);
    opt.set_lr(5e-4);
    EXPECT_DOUBLE_EQ(opt.get_lr(), 5e-4);
}

TEST_P(LionTest, StateDictRoundtrip) {
    auto params = make_params();
    optim::Lion opt(params, /*lr=*/0.01, /*beta1=*/0.9, /*beta2=*/0.99,
                    /*weight_decay=*/0.1);

    for (int i = 0; i < 3; ++i) step_with_unit_grad(params, opt);

    auto state = opt.state_dict();
    EXPECT_FALSE(state.empty());
    EXPECT_EQ(state.count("step_count"), 1u);
    EXPECT_EQ(state.count("lr"), 1u);
    EXPECT_EQ(state.count("beta1"), 1u);
    EXPECT_EQ(state.count("beta2"), 1u);
    EXPECT_EQ(state.count("weight_decay"), 1u);
    EXPECT_EQ(state.count("momentum_0"), 1u);

    // Round-trip: load into a fresh optimizer whose params clone opt's current
    // values, so a subsequent identical step is governed only by the restored
    // optimizer state (momentum_0 / step_count), then confirm both step the
    // same.
    auto params2 = std::vector<std::shared_ptr<Variable>>{
        std::make_shared<Variable>(params[0]->tensor().clone(), /*requires_grad=*/true)};
    optim::Lion opt2(params2, /*lr=*/1e-5);
    opt2.load_state_dict(state);

    EXPECT_DOUBLE_EQ(opt2.get_lr(), 0.01) << "lr should come from loaded state";

    auto state2 = opt2.state_dict();
    EXPECT_EQ(state.size(), state2.size());

    // Strong round-trip check: one identical unit-gradient step on both. If
    // momentum_0 and step_count were restored, the sign-based updates match.
    step_with_unit_grad(params, opt);
    step_with_unit_grad(params2, opt2);
    auto a = params[0]->tensor().to(DType::Float64).cpu();
    auto b = params2[0]->tensor().to(DType::Float64).cpu();
    double max_diff = ::tenzor::max(::tenzor::abs(a - b)).item<double>();
    EXPECT_LT(max_diff, 1e-6)
        << "Post-load step diverged (max param diff " << max_diff
        << ") — load_state_dict did not restore the momentum buffer";
}

TEST_P(LionTest, SignBasedUpdateMagnitudeIsLr) {
    // Lion's update is lr * sign(c_t). With weight_decay=0, after one step
    // from ones with grad=+1, every element must equal exactly 1 - lr.
    // This is the distinguishing property of a sign-based optimizer.
    auto params = make_params();
    const double lr = 0.005;
    optim::Lion opt(params, lr, /*beta1=*/0.9, /*beta2=*/0.99, /*weight_decay=*/0.0);

    for (auto& p : params) {
        auto shape = p->tensor().shape();
        p->set_grad(tenzor::ones({shape.begin(), shape.end()},
                                 p->tensor().dtype(), device));
    }
    opt.step();

    auto cpu = params[0]->tensor().to(Device::cpu());
    const auto* d = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_NEAR(d[i], 1.0f - static_cast<float>(lr), 1e-6)
            << "Lion update magnitude should be exactly lr (sign-based), "
               "element " << i << " was " << d[i];
    }
}

INSTANTIATE_BACKEND_TESTS(LionTest);

} // namespace
} // namespace tenzor
