// Tests for the Adamax optimizer (Adam variant with infinity-norm denominator).
// Cross-backend. Follows the pattern of test_lion.cpp / test_nadam.cpp.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/adamax.hpp>
#include <tenzor/ops/creation.hpp>

#include "../../backend_test_fixture.hpp"

namespace tenzor {
namespace {

class AdamaxTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    std::vector<std::shared_ptr<Variable>> make_params() {
        auto t = tenzor::ones({4, 4}, DType::Float32, device);
        auto param = std::make_shared<Variable>(t, /*requires_grad=*/true);
        return {param};
    }

    float step_with_unit_grad(std::vector<std::shared_ptr<Variable>>& params,
                              optim::Optimizer& opt) {
        float sum = 0.0f;
        for (auto& p : params) {
            auto cpu_t = p->tensor().cpu();
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

TEST_P(AdamaxTest, BasicStepReducesLoss) {
    auto params = make_params();
    optim::Adamax opt(params, /*lr=*/1e-2);

    float before = step_with_unit_grad(params, opt);
    float after  = step_with_unit_grad(params, opt);

    EXPECT_LT(after, before)
        << "Adamax should reduce |param| sum when gradient is constant positive";
}

TEST_P(AdamaxTest, ConvergesOnQuadratic) {
    auto params = make_params();
    optim::Adamax opt(params, /*lr=*/1e-2);

    for (int step = 0; step < 200; ++step) {
        auto& p = params[0];
        auto grad = p->tensor() * full({1}, 2.0, p->tensor().dtype(), p->tensor().device());
        p->set_grad(grad);
        opt.step();
    }

    auto final = params[0]->tensor().cpu();
    const auto* d = final.data<float>();
    float norm = 0.0f;
    for (int64_t i = 0; i < final.numel(); ++i) norm += d[i] * d[i];
    EXPECT_LT(norm, 2.0f)
        << "Adamax on a quadratic should drive ||x||^2 well below the initial 16";
}

TEST_P(AdamaxTest, LrGetSet) {
    auto params = make_params();
    optim::Adamax opt(params, /*lr=*/2e-3);

    EXPECT_DOUBLE_EQ(opt.get_lr(), 2e-3);
    opt.set_lr(5e-3);
    EXPECT_DOUBLE_EQ(opt.get_lr(), 5e-3);
}

TEST_P(AdamaxTest, StateDictRoundtrip) {
    auto params = make_params();
    optim::Adamax opt(params, /*lr=*/2e-3);

    for (int i = 0; i < 3; ++i) step_with_unit_grad(params, opt);

    auto state = opt.state_dict();
    EXPECT_EQ(state.count("step_count"), 1u);
    EXPECT_EQ(state.count("lr"), 1u);
    EXPECT_EQ(state.count("beta1"), 1u);
    EXPECT_EQ(state.count("beta2"), 1u);
    EXPECT_EQ(state.count("exp_avg_0"), 1u);
    EXPECT_EQ(state.count("exp_inf_0"), 1u);

    auto params2 = make_params();
    optim::Adamax opt2(params2, /*lr=*/1e-5);
    opt2.load_state_dict(state);
    EXPECT_DOUBLE_EQ(opt2.get_lr(), 2e-3) << "lr should come from loaded state";
}

INSTANTIATE_BACKEND_TESTS(AdamaxTest);

} // namespace
} // namespace tenzor
