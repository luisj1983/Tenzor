/**
 * @file test_zero_stage3_integration_multidtype.cpp
 * @brief Multi-backend multi-dtype integration tests for ZeRO Stage 3 Optimizer
 *
 * Distributed/multi-device tests. CPU and single-device backends are skipped.
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::nn;
using namespace tenzor::testing;

class ZeROStage3IntegrationMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "ZeRO Stage 3 integration tests require non-CPU multi-device backend";
        }

        default_config.world_size = 1;
        default_config.rank = 0;
        default_config.offload_to_cpu = false;
        default_config.overlap_comm = true;
        default_config.process_group = nullptr;
    }

    Stage3Config default_config;
};

TEST_P(ZeROStage3IntegrationMultiDTypeTest, TrainingLoopSingleStep) {
    auto linear = std::make_shared<Linear>(16, 8);
    convert_model(linear);

    auto params = linear->parameters();
    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage3Optimizer optimizer(std::move(adam), default_config);

    auto input = createInput({4, 16}, true);
    auto output = linear->forward(input);
    auto loss = sum(output);
    loss.backward();

    EXPECT_NO_THROW({
        optimizer.step();
        optimizer.zero_grad();
    });
}

TEST_P(ZeROStage3IntegrationMultiDTypeTest, MultipleSteps) {
    auto linear = std::make_shared<Linear>(8, 4);
    convert_model(linear);

    auto params = linear->parameters();
    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage3Optimizer optimizer(std::move(adam), default_config);

    for (int step = 0; step < 3; ++step) {
        auto input = createInput({2, 8}, true);
        auto output = linear->forward(input);
        auto loss = sum(output);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
}

TEST_P(ZeROStage3IntegrationMultiDTypeTest, WithSGDOptimizer) {
    auto linear = std::make_shared<Linear>(8, 4);
    convert_model(linear);

    auto params = linear->parameters();
    auto sgd = std::make_unique<SGD>(params, 0.01);
    ZeROStage3Optimizer optimizer(std::move(sgd), default_config);

    auto input = createInput({2, 8}, true);
    auto output = linear->forward(input);
    auto loss = sum(output);
    loss.backward();

    EXPECT_NO_THROW(optimizer.step());
}

TEST_P(ZeROStage3IntegrationMultiDTypeTest, CPUOffloadConfig) {
    auto linear = std::make_shared<Linear>(8, 4);
    convert_model(linear);

    auto params = linear->parameters();
    auto adam = std::make_unique<Adam>(params, 1e-3);

    auto config = default_config;
    config.offload_to_cpu = true;

    EXPECT_NO_THROW({
        ZeROStage3Optimizer optimizer(std::move(adam), config);
    });
}

TEST_P(ZeROStage3IntegrationMultiDTypeTest, GatherAndFreeLifecycle) {
    auto linear = std::make_shared<Linear>(8, 4);
    convert_model(linear);

    auto params = linear->parameters();
    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage3Optimizer optimizer(std::move(adam), default_config);

    // Run forward/backward twice to exercise gather/free lifecycle
    for (int i = 0; i < 2; ++i) {
        auto input = createInput({2, 8}, true);
        auto output = linear->forward(input);
        auto loss = sum(output);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ZeROStage3IntegrationMultiDTypeTest);
