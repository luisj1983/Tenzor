/**
 * @file test_zero_stage2_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for ZeRO Stage 2 Optimizer
 *
 * Distributed/multi-device tests. CPU and single-device backends are skipped.
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::testing;

class ZeROStage2MultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "ZeRO tests require non-CPU multi-device backend";
        }

        default_config.world_size = 1;
        default_config.rank = 0;
        default_config.offload_to_cpu = false;
        default_config.overlap_comm = true;
        default_config.process_group = nullptr;
    }

    auto create_test_params(size_t count, const std::vector<int64_t>& shape = {64, 64})
        -> std::vector<std::shared_ptr<Variable>> {
        std::vector<std::shared_ptr<Variable>> params;
        for (size_t i = 0; i < count; ++i) {
            auto t = tenzor::ones(shape, dtype(), device());
            params.push_back(std::make_shared<Variable>(t, true));
        }
        return params;
    }

    ZeROStage2Config default_config;
};

TEST_P(ZeROStage2MultiDTypeTest, Construction) {
    auto params = create_test_params(2);
    auto adam = std::make_unique<Adam>(params, 1e-3);

    EXPECT_NO_THROW({
        ZeROStage2Optimizer optimizer(std::move(adam), default_config);
    });
}

TEST_P(ZeROStage2MultiDTypeTest, Step) {
    // audit T.1: assert parameter trajectory matches a CPU+Float32 plain-Adam
    // reference (world_size=1 ZeRO Stage2 collapses to a single-process Adam
    // step on the gradient-partitioned local copy).
    auto params = create_test_params(2, {16, 16});
    for (auto& p : params) {
        p->set_grad(tenzor::ones({16, 16}, dtype(), device()) * 0.1f);
    }

    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage2Optimizer optimizer(std::move(adam), default_config);

    EXPECT_NO_THROW(optimizer.step());

    std::vector<std::shared_ptr<Variable>> ref_params;
    for (size_t i = 0; i < 2; ++i) {
        ref_params.push_back(std::make_shared<Variable>(
            tenzor::ones({16, 16}, DType::Float32, Device::cpu()), true));
    }
    for (auto& p : ref_params) {
        p->set_grad(tenzor::ones({16, 16}, DType::Float32, Device::cpu()) * 0.1f);
    }
    Adam ref_adam(ref_params, 1e-3);
    ref_adam.step();

    for (size_t i = 0; i < params.size(); ++i) {
        expectTensorNear(params[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
}

TEST_P(ZeROStage2MultiDTypeTest, ZeroGrad) {
    auto params = create_test_params(2, {16, 16});
    for (auto& p : params) {
        p->set_grad(tenzor::ones({16, 16}, dtype(), device()));
    }

    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage2Optimizer optimizer(std::move(adam), default_config);

    EXPECT_NO_THROW(optimizer.zero_grad());

    // audit T.1: assert grads were actually zeroed.
    for (auto& p : params) {
        ASSERT_TRUE(p->grad().has_value());
        auto zeros_ref = tenzor::zeros({16, 16}, DType::Float32, Device::cpu());
        expectTensorNear(*p->grad(), zeros_ref, atol_);
    }
}

TEST_P(ZeROStage2MultiDTypeTest, LocalParamCount) {
    auto params = create_test_params(4);
    auto adam = std::make_unique<Adam>(params, 1e-3);

    ZeROStage2Optimizer optimizer(std::move(adam), default_config);
    EXPECT_GT(optimizer.local_param_count(), 0u);
}

TEST_P(ZeROStage2MultiDTypeTest, GradientBucketing) {
    auto params = create_test_params(4, {32, 32});
    for (auto& p : params) {
        p->set_grad(tenzor::ones({32, 32}, dtype(), device()) * 0.01f);
    }

    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage2Optimizer optimizer(std::move(adam), default_config);

    EXPECT_NO_THROW({
        optimizer.step();
        optimizer.zero_grad();
    });

    // audit T.1: after step+zero_grad, params must match CPU+F32 Adam step
    // result, and grads must be zero. Bucketing is an internal optimization;
    // observable behaviour matches plain Adam.
    std::vector<std::shared_ptr<Variable>> ref_params;
    for (size_t i = 0; i < 4; ++i) {
        ref_params.push_back(std::make_shared<Variable>(
            tenzor::ones({32, 32}, DType::Float32, Device::cpu()), true));
    }
    for (auto& p : ref_params) {
        p->set_grad(tenzor::ones({32, 32}, DType::Float32, Device::cpu()) * 0.01f);
    }
    Adam ref_adam(ref_params, 1e-3);
    ref_adam.step();

    auto zero_ref = tenzor::zeros({32, 32}, DType::Float32, Device::cpu());
    for (size_t i = 0; i < params.size(); ++i) {
        expectTensorNear(params[i]->tensor(), ref_params[i]->tensor(), atol_);
        ASSERT_TRUE(params[i]->grad().has_value());
        expectTensorNear(*params[i]->grad(), zero_ref, atol_);
    }
}

TEST_P(ZeROStage2MultiDTypeTest, MultipleStepsTrajectory) {
    // audit T.1: 3-step trajectory parity with CPU+F32 plain Adam.
    auto params = create_test_params(1, {8, 8});
    auto adam = std::make_unique<Adam>(params, 1e-2);
    ZeROStage2Optimizer optimizer(std::move(adam), default_config);

    std::vector<std::shared_ptr<Variable>> ref_params;
    ref_params.push_back(std::make_shared<Variable>(
        tenzor::ones({8, 8}, DType::Float32, Device::cpu()), true));
    Adam ref_adam(ref_params, 1e-2);

    for (int s = 0; s < 3; ++s) {
        params[0]->set_grad(tenzor::ones({8, 8}, dtype(), device()) * 0.05f);
        ref_params[0]->set_grad(tenzor::ones({8, 8}, DType::Float32, Device::cpu()) * 0.05f);
        optimizer.step();
        ref_adam.step();
        expectTensorNear(params[0]->tensor(), ref_params[0]->tensor(), atol_);
    }
}

TEST_P(ZeROStage2MultiDTypeTest, StateDictRoundtrip) {
    // audit T.1: save state, reload into fresh optimizer, take one more step,
    // assert outputs match the continued original.
    auto params = create_test_params(1, {8, 8});
    auto adam = std::make_unique<Adam>(params, 1e-2);
    ZeROStage2Optimizer optimizer(std::move(adam), default_config);

    for (int s = 0; s < 2; ++s) {
        params[0]->set_grad(tenzor::ones({8, 8}, dtype(), device()) * 0.05f);
        optimizer.step();
    }
    auto state = optimizer.state_dict();

    auto pre_extra = params[0]->tensor().to(Device::cpu()).to(DType::Float32);
    params[0]->set_grad(tenzor::ones({8, 8}, dtype(), device()) * 0.05f);
    optimizer.step();
    auto after_extra = params[0]->tensor().to(Device::cpu()).to(DType::Float32);

    auto fresh_params = create_test_params(1, {8, 8});
    fresh_params[0]->tensor() = pre_extra.to(dtype()).to(device());
    auto fresh_adam = std::make_unique<Adam>(fresh_params, 1e-2);
    ZeROStage2Optimizer fresh_opt(std::move(fresh_adam), default_config);
    fresh_opt.load_state_dict(state);
    fresh_params[0]->set_grad(tenzor::ones({8, 8}, dtype(), device()) * 0.05f);
    fresh_opt.step();

    expectTensorNear(fresh_params[0]->tensor(), after_extra, atol_);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ZeROStage2MultiDTypeTest);
