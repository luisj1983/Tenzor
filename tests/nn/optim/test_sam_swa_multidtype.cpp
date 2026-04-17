/**
 * @file test_sam_swa_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for SAM optimizer, AveragedModel, SWALR
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/sam.hpp>
#include <tenzor/nn/optim/swa.hpp>
#include <tenzor/nn/optim/sgd.hpp>

using namespace tenzor;
using namespace tenzor::testing;

class SAMSWAMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::vector<std::shared_ptr<Variable>> make_params() {
        auto t = createOnes({4, 4});
        auto param = std::make_shared<Variable>(t, true);
        return {param};
    }

    void set_unit_grad(std::vector<std::shared_ptr<Variable>>& params) {
        for (auto& p : params) {
            auto shape = p->tensor().shape();
            p->set_grad(tenzor::ones({shape.begin(), shape.end()},
                                     p->tensor().dtype(), p->tensor().device()));
        }
    }

    float param_sum(const std::vector<std::shared_ptr<Variable>>& params) {
        float sum_val = 0.0f;
        for (auto& p : params) {
            auto cpu_t = p->tensor().to(Device::cpu()).to(DType::Float32);
            const auto* d = cpu_t.data<float>();
            for (int64_t i = 0; i < cpu_t.numel(); ++i) sum_val += d[i];
        }
        return sum_val;
    }
};

TEST_P(SAMSWAMultiDTypeTest, SAMFirstStepPerturbsWeights) {
    auto params = make_params();
    auto base_opt = std::make_shared<optim::SGD>(params, 0.1);
    optim::SAM sam(base_opt, 0.05);

    float before = param_sum(params);
    set_unit_grad(params);
    sam.first_step();
    float after = param_sum(params);

    EXPECT_NE(before, after);
}

TEST_P(SAMSWAMultiDTypeTest, SAMSecondStepRestoresAndSteps) {
    auto params = make_params();
    auto base_opt = std::make_shared<optim::SGD>(params, 0.1);
    optim::SAM sam(base_opt, 0.05);

    float original = param_sum(params);
    set_unit_grad(params);
    sam.first_step();
    float perturbed = param_sum(params);
    EXPECT_NE(original, perturbed);

    set_unit_grad(params);
    sam.second_step();
    float final_val = param_sum(params);
    EXPECT_NE(final_val, perturbed);
}

TEST_P(SAMSWAMultiDTypeTest, SAMLrGetSet) {
    auto params = make_params();
    auto base_opt = std::make_shared<optim::SGD>(params, 0.01);
    optim::SAM sam(base_opt, 0.05);

    EXPECT_DOUBLE_EQ(sam.get_lr(), 0.01);
    sam.set_lr(0.05);
    EXPECT_DOUBLE_EQ(sam.get_lr(), 0.05);
}

TEST_P(SAMSWAMultiDTypeTest, AveragedModelInitialState) {
    auto params = make_params();
    optim::AveragedModel avg(params);
    EXPECT_EQ(avg.n_averaged(), 1);
    EXPECT_EQ(avg.averaged_params().size(), params.size());
}

TEST_P(SAMSWAMultiDTypeTest, SWALRReturnsCorrectLR) {
    // SWALR uses CPU params for simplicity
    auto param = std::make_shared<Variable>(
        tenzor::ones({4, 4}, DType::Float32, Device::cpu()), true);
    auto params = std::vector<std::shared_ptr<Variable>>{param};
    optim::SGD opt(params, 0.1);
    optim::SWALR swalr(opt, 0.05, 5);

    for (int i = 0; i < 10; ++i) {
        swalr.step();
    }

    EXPECT_NEAR(swalr.get_last_lr(), 0.05, 1e-6);
}

TEST_P(SAMSWAMultiDTypeTest, ParamsOnCorrectDevice) {
    auto params = make_params();
    auto base_opt = std::make_shared<optim::SGD>(params, 0.1);
    optim::SAM sam(base_opt, 0.05);
    set_unit_grad(params);
    sam.first_step();
    expectDevice(params[0]->tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SAMSWAMultiDTypeTest);
