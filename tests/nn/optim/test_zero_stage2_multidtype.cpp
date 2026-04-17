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
    auto params = create_test_params(2, {16, 16});
    for (auto& p : params) {
        p->set_grad(tenzor::ones({16, 16}, dtype(), device()) * 0.1f);
    }

    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage2Optimizer optimizer(std::move(adam), default_config);

    EXPECT_NO_THROW(optimizer.step());
}

TEST_P(ZeROStage2MultiDTypeTest, ZeroGrad) {
    auto params = create_test_params(2, {16, 16});
    for (auto& p : params) {
        p->set_grad(tenzor::ones({16, 16}, dtype(), device()));
    }

    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage2Optimizer optimizer(std::move(adam), default_config);

    EXPECT_NO_THROW(optimizer.zero_grad());
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
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ZeROStage2MultiDTypeTest);
