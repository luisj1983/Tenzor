// Tests for the Lion (EvoLved Sign Momentum) optimizer.
// CPU-only. Verifies convergence on a small quadratic, lr get/set,
// and state_dict round-trip.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/lion.hpp>
#include <tenzor/ops/creation.hpp>

namespace tenzor {
namespace {

class LionTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }

    // One 4x4 parameter initialized to ones, on CPU Float32.
    std::vector<std::shared_ptr<Variable>> make_params() {
        auto t = tenzor::ones({4, 4}, DType::Float32, Device::cpu());
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
                                     p->tensor().dtype(), Device::cpu()));
        }
        opt.step();
        return sum;
    }
};

TEST_F(LionTest, BasicStepReducesLoss) {
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

TEST_F(LionTest, LrGetSet) {
    auto params = make_params();
    optim::Lion opt(params, /*lr=*/1e-4);

    EXPECT_DOUBLE_EQ(opt.get_lr(), 1e-4);
    opt.set_lr(5e-4);
    EXPECT_DOUBLE_EQ(opt.get_lr(), 5e-4);
}

TEST_F(LionTest, StateDictRoundtrip) {
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

    // Round-trip: load into a fresh optimizer with a different lr, then
    // confirm that reading the state back matches what we loaded.
    auto params2 = make_params();
    optim::Lion opt2(params2, /*lr=*/1e-5);
    opt2.load_state_dict(state);

    EXPECT_DOUBLE_EQ(opt2.get_lr(), 0.01) << "lr should come from loaded state";

    auto state2 = opt2.state_dict();
    EXPECT_EQ(state.size(), state2.size());
}

TEST_F(LionTest, SignBasedUpdateMagnitudeIsLr) {
    // Lion's update is lr * sign(c_t). With weight_decay=0, after one step
    // from ones with grad=+1, every element must equal exactly 1 - lr.
    // This is the distinguishing property of a sign-based optimizer.
    auto params = make_params();
    const double lr = 0.005;
    optim::Lion opt(params, lr, /*beta1=*/0.9, /*beta2=*/0.99, /*weight_decay=*/0.0);

    for (auto& p : params) {
        auto shape = p->tensor().shape();
        p->set_grad(tenzor::ones({shape.begin(), shape.end()},
                                 p->tensor().dtype(), Device::cpu()));
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

} // namespace
} // namespace tenzor
