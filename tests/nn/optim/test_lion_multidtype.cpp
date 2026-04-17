/**
 * @file test_lion_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Lion optimizer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/lion.hpp>

using namespace tenzor;
using namespace tenzor::testing;

class LionMultiDTypeTest : public MultiBackendDTypeTest {
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

TEST_P(LionMultiDTypeTest, BasicStepReducesLoss) {
    auto params = make_params();
    optim::Lion opt(params, 0.01);

    float before = step_with_unit_grad(params, opt);
    float after  = step_with_unit_grad(params, opt);

    EXPECT_LT(after, before);
}

TEST_P(LionMultiDTypeTest, LrGetSet) {
    auto params = make_params();
    optim::Lion opt(params, 1e-4);

    EXPECT_DOUBLE_EQ(opt.get_lr(), 1e-4);
    opt.set_lr(5e-4);
    EXPECT_DOUBLE_EQ(opt.get_lr(), 5e-4);
}

TEST_P(LionMultiDTypeTest, StateDictRoundtrip) {
    auto params = make_params();
    optim::Lion opt(params, 0.01, 0.9, 0.99, 0.1);

    for (int i = 0; i < 3; ++i) step_with_unit_grad(params, opt);

    auto state = opt.state_dict();
    EXPECT_FALSE(state.empty());
    EXPECT_EQ(state.count("step_count"), 1u);
    EXPECT_EQ(state.count("lr"), 1u);
    EXPECT_EQ(state.count("momentum_0"), 1u);

    auto params2 = make_params();
    optim::Lion opt2(params2, 1e-5);
    opt2.load_state_dict(state);
    EXPECT_DOUBLE_EQ(opt2.get_lr(), 0.01);
}

TEST_P(LionMultiDTypeTest, SignBasedUpdateMagnitude) {
    auto params = make_params();
    const double lr = 0.005;
    optim::Lion opt(params, lr, 0.9, 0.99, 0.0);

    for (auto& p : params) {
        auto shape = p->tensor().shape();
        p->set_grad(tenzor::ones({shape.begin(), shape.end()},
                                 p->tensor().dtype(), p->tensor().device()));
    }
    opt.step();

    auto cpu = params[0]->tensor().to(Device::cpu()).to(DType::Float32);
    const auto* d = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_NEAR(d[i], 1.0f - static_cast<float>(lr), atol());
    }
}

TEST_P(LionMultiDTypeTest, ParamsOnCorrectDevice) {
    auto params = make_params();
    optim::Lion opt(params, 0.01);
    step_with_unit_grad(params, opt);
    expectDevice(params[0]->tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LionMultiDTypeTest);
