/**
 * @file test_zero_stage1_integration_multidtype.cpp
 * @brief Multi-backend multi-dtype integration tests for ZeRO Stage 1 Optimizer
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

class ZeROStage1IntegrationMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "ZeRO integration tests require non-CPU multi-device backend";
        }

        default_config.world_size = 1;
        default_config.rank = 0;
        default_config.offload_to_cpu = false;
        default_config.overlap_comm = true;
        default_config.process_group = nullptr;
    }

    ZeROStage1Config default_config;
};

TEST_P(ZeROStage1IntegrationMultiDTypeTest, TrainingLoopSingleStep) {
    // audit T.1: assert parameter trajectory after a real forward/backward/step
    // matches a CPU+Float32 reference Linear with cloned weights and the same
    // input. Mirrors the actual training loop end-to-end.
    auto linear = std::make_shared<Linear>(16, 8);
    convert_model(linear);

    // Build CPU+F32 reference with the *same* initial weights.
    auto ref_linear = std::make_shared<Linear>(16, 8);
    {
        auto src_w = linear->weight()->tensor().to(Device::cpu()).to(DType::Float32);
        ref_linear->weight()->tensor() = src_w;
        if (linear->has_bias()) {
            auto src_b = linear->bias()->tensor().to(Device::cpu()).to(DType::Float32);
            ref_linear->bias()->tensor() = src_b;
        }
    }

    auto params = linear->parameters();
    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage1Optimizer optimizer(std::move(adam), default_config);

    auto ref_params = ref_linear->parameters();
    Adam ref_adam(ref_params, 1e-3);

    // Deterministic input: ones — avoids randn divergence between the device
    // and CPU runs.
    auto input = Variable(tenzor::ones({4, 16}, dtype(), device()), true);
    auto ref_input = Variable(tenzor::ones({4, 16}, DType::Float32, Device::cpu()), true);

    auto output = linear->forward(input);
    auto loss = sum(output);
    loss.backward();

    auto ref_output = ref_linear->forward(ref_input);
    auto ref_loss = sum(ref_output);
    ref_loss.backward();

    EXPECT_NO_THROW({
        optimizer.step();
        optimizer.zero_grad();
    });
    ref_adam.step();

    for (size_t i = 0; i < params.size(); ++i) {
        expectTensorNear(params[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
}

TEST_P(ZeROStage1IntegrationMultiDTypeTest, MultipleSteps) {
    // audit T.1: 3-step trajectory parity vs CPU+F32 plain Adam on a cloned
    // Linear.
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
    ZeROStage1Optimizer optimizer(std::move(adam), default_config);
    auto ref_params = ref_linear->parameters();
    Adam ref_adam(ref_params, 1e-3);

    for (int step = 0; step < 3; ++step) {
        auto input = Variable(tenzor::ones({2, 8}, dtype(), device()), true);
        auto ref_input = Variable(tenzor::ones({2, 8}, DType::Float32, Device::cpu()), true);
        auto output = linear->forward(input);
        auto loss = sum(output);
        loss.backward();
        auto ref_output = ref_linear->forward(ref_input);
        auto ref_loss = sum(ref_output);
        ref_loss.backward();
        optimizer.step();
        optimizer.zero_grad();
        ref_adam.step();
        ref_adam.zero_grad();
    }

    for (size_t i = 0; i < params.size(); ++i) {
        expectTensorNear(params[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
}

TEST_P(ZeROStage1IntegrationMultiDTypeTest, WithSGDOptimizer) {
    // audit T.1: SGD step parity vs CPU+F32 reference Linear.
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
    ZeROStage1Optimizer optimizer(std::move(sgd), default_config);
    auto ref_params = ref_linear->parameters();
    SGD ref_sgd(ref_params, 0.01);

    auto input = Variable(tenzor::ones({2, 8}, dtype(), device()), true);
    auto ref_input = Variable(tenzor::ones({2, 8}, DType::Float32, Device::cpu()), true);
    auto output = linear->forward(input);
    auto loss = sum(output);
    loss.backward();
    auto ref_output = ref_linear->forward(ref_input);
    auto ref_loss = sum(ref_output);
    ref_loss.backward();

    EXPECT_NO_THROW(optimizer.step());
    ref_sgd.step();

    for (size_t i = 0; i < params.size(); ++i) {
        expectTensorNear(params[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
}

TEST_P(ZeROStage1IntegrationMultiDTypeTest, ParametersUpdated) {
    // audit T.1: prove params actually change (the original
    // "EXPECT_TRUE(true)" was a no-op). Also assert the *update direction*
    // matches a CPU+F32 reference, not just any change.
    auto linear = std::make_shared<Linear>(8, 4);
    convert_model(linear);

    auto params = linear->parameters();
    EXPECT_FALSE(params.empty());

    auto initial_val = params[0]->tensor().to(Device::cpu()).to(DType::Float32);

    // CPU+F32 reference with cloned initial weights.
    auto ref_linear = std::make_shared<Linear>(8, 4);
    ref_linear->weight()->tensor() =
        linear->weight()->tensor().to(Device::cpu()).to(DType::Float32);
    if (linear->has_bias()) {
        ref_linear->bias()->tensor() =
            linear->bias()->tensor().to(Device::cpu()).to(DType::Float32);
    }
    auto ref_params = ref_linear->parameters();

    auto adam = std::make_unique<Adam>(params, 1e-1);  // large lr
    ZeROStage1Optimizer optimizer(std::move(adam), default_config);
    Adam ref_adam(ref_params, 1e-1);

    auto input = Variable(tenzor::ones({2, 8}, dtype(), device()), true);
    auto ref_input = Variable(tenzor::ones({2, 8}, DType::Float32, Device::cpu()), true);
    auto output = linear->forward(input);
    auto loss = sum(output);
    loss.backward();
    auto ref_output = ref_linear->forward(ref_input);
    auto ref_loss = sum(ref_output);
    ref_loss.backward();
    optimizer.step();
    ref_adam.step();

    auto updated_val = params[0]->tensor().to(Device::cpu()).to(DType::Float32);
    // Params must have moved away from their initial value (lr=0.1 is large
    // enough that float roundoff alone could not explain identity).
    const auto* init_data = initial_val.data<float>();
    const auto* upd_data = updated_val.data<float>();
    float max_delta = 0.0f;
    for (int64_t i = 0; i < initial_val.numel(); ++i) {
        max_delta = std::max(max_delta, std::abs(upd_data[i] - init_data[i]));
    }
    EXPECT_GT(max_delta, 1e-4f) << "params did not change after a step";

    // Trajectory matches CPU+F32 reference.
    for (size_t i = 0; i < params.size(); ++i) {
        expectTensorNear(params[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
}

TEST_P(ZeROStage1IntegrationMultiDTypeTest, CPUOffloadConfig) {
    auto linear = std::make_shared<Linear>(8, 4);
    convert_model(linear);

    auto params = linear->parameters();
    auto adam = std::make_unique<Adam>(params, 1e-3);

    auto config = default_config;
    config.offload_to_cpu = true;

    EXPECT_NO_THROW({
        ZeROStage1Optimizer optimizer(std::move(adam), config);
    });
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ZeROStage1IntegrationMultiDTypeTest);
