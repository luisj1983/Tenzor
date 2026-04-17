/**
 * @file test_adamax_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Adamax optimizer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/adamax.hpp>

using namespace tenzor;
using namespace tenzor::testing;

class AdamaxMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::vector<std::shared_ptr<Variable>> make_params() {
        auto t = createOnes({4, 4});
        auto param = std::make_shared<Variable>(t, true);
        return {param};
    }

    float step_with_unit_grad(std::vector<std::shared_ptr<Variable>>& params,
                              optim::Optimizer& opt) {
        float sum_val = 0.0f;
        for (auto& p : params) {
            auto cpu_t = p->tensor().to(Device::cpu()).to(DType::Float32);
            const auto* d = cpu_t.data<float>();
            for (int64_t i = 0; i < cpu_t.numel(); ++i) sum_val += std::abs(d[i]);
        }
        for (auto& p : params) {
            auto shape = p->tensor().shape();
            p->set_grad(tenzor::ones({shape.begin(), shape.end()},
                                     p->tensor().dtype(), p->tensor().device()));
        }
        opt.step();
        return sum_val;
    }
};

TEST_P(AdamaxMultiDTypeTest, BasicStepReducesLoss) {
    auto params = make_params();
    optim::Adamax opt(params, 1e-2);

    float before = step_with_unit_grad(params, opt);
    float after  = step_with_unit_grad(params, opt);

    EXPECT_LT(after, before);
}

TEST_P(AdamaxMultiDTypeTest, LrGetSet) {
    auto params = make_params();
    optim::Adamax opt(params, 2e-3);

    EXPECT_DOUBLE_EQ(opt.get_lr(), 2e-3);
    opt.set_lr(5e-3);
    EXPECT_DOUBLE_EQ(opt.get_lr(), 5e-3);
}

TEST_P(AdamaxMultiDTypeTest, StateDictRoundtrip) {
    auto params = make_params();
    optim::Adamax opt(params, 2e-3);

    for (int i = 0; i < 3; ++i) step_with_unit_grad(params, opt);

    auto state = opt.state_dict();
    EXPECT_EQ(state.count("step_count"), 1u);
    EXPECT_EQ(state.count("lr"), 1u);
    EXPECT_EQ(state.count("exp_avg_0"), 1u);
    EXPECT_EQ(state.count("exp_inf_0"), 1u);

    auto params2 = make_params();
    optim::Adamax opt2(params2, 1e-5);
    opt2.load_state_dict(state);
    EXPECT_DOUBLE_EQ(opt2.get_lr(), 2e-3);
}

TEST_P(AdamaxMultiDTypeTest, ConvergesOnQuadratic) {
    auto params = make_params();
    optim::Adamax opt(params, 1e-2);

    for (int step = 0; step < 200; ++step) {
        auto& p = params[0];
        auto grad = p->tensor() * full({1}, 2.0, p->tensor().dtype(), p->tensor().device());
        p->set_grad(grad);
        opt.step();
    }

    auto final_t = params[0]->tensor().to(Device::cpu()).to(DType::Float32);
    const auto* d = final_t.data<float>();
    float norm = 0.0f;
    for (int64_t i = 0; i < final_t.numel(); ++i) norm += d[i] * d[i];
    EXPECT_LT(norm, 2.0f);
}

TEST_P(AdamaxMultiDTypeTest, ParamsOnCorrectDevice) {
    auto params = make_params();
    optim::Adamax opt(params, 1e-2);
    step_with_unit_grad(params, opt);
    expectDevice(params[0]->tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(AdamaxMultiDTypeTest);
