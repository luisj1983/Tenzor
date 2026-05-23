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
    // audit T.1: trajectory parity with CPU+F32 plain Adam.
    auto linear = std::make_shared<Linear>(16, 8);
    convert_model(linear);
    auto ref_linear = std::make_shared<Linear>(16, 8);
    ref_linear->weight()->tensor() =
        linear->weight()->tensor().to(Device::cpu()).to(DType::Float32);
    if (linear->has_bias()) {
        ref_linear->bias()->tensor() =
            linear->bias()->tensor().to(Device::cpu()).to(DType::Float32);
    }

    auto params = linear->parameters();
    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage3Optimizer optimizer(std::move(adam), default_config);
    auto ref_params = ref_linear->parameters();
    Adam ref_adam(ref_params, 1e-3);

    auto input = Variable(tenzor::ones({4, 16}, dtype(), device()), true);
    auto ref_input = Variable(tenzor::ones({4, 16}, DType::Float32, Device::cpu()), true);
    sum(linear->forward(input)).backward();
    sum(ref_linear->forward(ref_input)).backward();

    EXPECT_NO_THROW({
        optimizer.step();
        optimizer.zero_grad();
    });
    ref_adam.step();

    for (size_t i = 0; i < params.size(); ++i) {
        expectTensorNear(params[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
}

TEST_P(ZeROStage3IntegrationMultiDTypeTest, MultipleSteps) {
    // audit T.1: 3-step trajectory parity with CPU+F32 plain Adam.
    auto linear = std::make_shared<Linear>(8, 4);
    convert_model(linear);
    auto ref_linear = std::make_shared<Linear>(8, 4);
    ref_linear->weight()->tensor() =
        linear->weight()->tensor().to(Device::cpu()).to(DType::Float32);
    if (linear->has_bias()) {
        ref_linear->bias()->tensor() =
            linear->bias()->tensor().to(Device::cpu()).to(DType::Float32);
    }

    auto params = linear->parameters();
    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage3Optimizer optimizer(std::move(adam), default_config);
    auto ref_params = ref_linear->parameters();
    Adam ref_adam(ref_params, 1e-3);

    for (int step = 0; step < 3; ++step) {
        auto input = Variable(tenzor::ones({2, 8}, dtype(), device()), true);
        auto ref_input = Variable(tenzor::ones({2, 8}, DType::Float32, Device::cpu()), true);
        sum(linear->forward(input)).backward();
        sum(ref_linear->forward(ref_input)).backward();
        optimizer.step();
        optimizer.zero_grad();
        ref_adam.step();
        ref_adam.zero_grad();
    }

    for (size_t i = 0; i < params.size(); ++i) {
        expectTensorNear(params[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
}

TEST_P(ZeROStage3IntegrationMultiDTypeTest, WithSGDOptimizer) {
    // audit T.1: SGD trajectory parity vs CPU+F32 reference.
    auto linear = std::make_shared<Linear>(8, 4);
    convert_model(linear);
    auto ref_linear = std::make_shared<Linear>(8, 4);
    ref_linear->weight()->tensor() =
        linear->weight()->tensor().to(Device::cpu()).to(DType::Float32);
    if (linear->has_bias()) {
        ref_linear->bias()->tensor() =
            linear->bias()->tensor().to(Device::cpu()).to(DType::Float32);
    }

    auto params = linear->parameters();
    auto sgd = std::make_unique<SGD>(params, 0.01);
    ZeROStage3Optimizer optimizer(std::move(sgd), default_config);
    auto ref_params = ref_linear->parameters();
    SGD ref_sgd(ref_params, 0.01);

    auto input = Variable(tenzor::ones({2, 8}, dtype(), device()), true);
    auto ref_input = Variable(tenzor::ones({2, 8}, DType::Float32, Device::cpu()), true);
    sum(linear->forward(input)).backward();
    sum(ref_linear->forward(ref_input)).backward();

    EXPECT_NO_THROW(optimizer.step());
    ref_sgd.step();

    for (size_t i = 0; i < params.size(); ++i) {
        expectTensorNear(params[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
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
    // audit T.1: gather/free lifecycle must preserve trajectory across two
    // forward/backward/step iterations. Compare against a CPU+F32 reference
    // running the same loop.
    auto linear = std::make_shared<Linear>(8, 4);
    convert_model(linear);
    auto ref_linear = std::make_shared<Linear>(8, 4);
    ref_linear->weight()->tensor() =
        linear->weight()->tensor().to(Device::cpu()).to(DType::Float32);
    if (linear->has_bias()) {
        ref_linear->bias()->tensor() =
            linear->bias()->tensor().to(Device::cpu()).to(DType::Float32);
    }

    auto params = linear->parameters();
    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage3Optimizer optimizer(std::move(adam), default_config);
    auto ref_params = ref_linear->parameters();
    Adam ref_adam(ref_params, 1e-3);

    for (int i = 0; i < 2; ++i) {
        auto input = Variable(tenzor::ones({2, 8}, dtype(), device()), true);
        auto ref_input = Variable(tenzor::ones({2, 8}, DType::Float32, Device::cpu()), true);
        sum(linear->forward(input)).backward();
        sum(ref_linear->forward(ref_input)).backward();
        optimizer.step();
        optimizer.zero_grad();
        ref_adam.step();
        ref_adam.zero_grad();
    }

    for (size_t i = 0; i < params.size(); ++i) {
        expectTensorNear(params[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ZeROStage3IntegrationMultiDTypeTest);
