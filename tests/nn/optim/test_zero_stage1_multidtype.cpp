/**
 * @file test_zero_stage1_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for ZeRO Stage 1 Optimizer
 *
 * Distributed/multi-device tests. CPU and single-device backends are skipped.
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::testing;

class ZeROStage1MultiDTypeTest : public MultiBackendDTypeTest {
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
        params.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            auto t = tenzor::ones(shape, dtype(), device());
            params.push_back(std::make_shared<Variable>(t, true));
        }
        return params;
    }

    ZeROStage1Config default_config;
};

TEST_P(ZeROStage1MultiDTypeTest, Construction) {
    auto params = create_test_params(2);
    auto adam = std::make_unique<Adam>(params, 1e-3);

    EXPECT_NO_THROW({
        ZeROStage1Optimizer optimizer(std::move(adam), default_config);
    });
}

TEST_P(ZeROStage1MultiDTypeTest, PartitionCount) {
    auto params = create_test_params(4);
    auto adam = std::make_unique<Adam>(params, 1e-3);

    ZeROStage1Optimizer optimizer(std::move(adam), default_config);
    // With world_size=1, all params should be local
    EXPECT_GT(optimizer.local_param_count(), 0u);
}

TEST_P(ZeROStage1MultiDTypeTest, Step) {
    auto params = create_test_params(2, {16, 16});
    // Set gradients
    for (auto& p : params) {
        p->set_grad(tenzor::ones({16, 16}, dtype(), device()) * 0.1f);
    }

    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage1Optimizer optimizer(std::move(adam), default_config);

    EXPECT_NO_THROW(optimizer.step());
}

TEST_P(ZeROStage1MultiDTypeTest, ZeroGrad) {
    auto params = create_test_params(2, {16, 16});
    for (auto& p : params) {
        p->set_grad(tenzor::ones({16, 16}, dtype(), device()));
    }

    auto adam = std::make_unique<Adam>(params, 1e-3);
    ZeROStage1Optimizer optimizer(std::move(adam), default_config);

    EXPECT_NO_THROW(optimizer.zero_grad());
}

TEST_P(ZeROStage1MultiDTypeTest, WithSGD) {
    auto params = create_test_params(2, {16, 16});
    for (auto& p : params) {
        p->set_grad(tenzor::ones({16, 16}, dtype(), device()) * 0.1f);
    }

    auto sgd = std::make_unique<SGD>(params, 0.01);
    ZeROStage1Optimizer optimizer(std::move(sgd), default_config);

    EXPECT_NO_THROW(optimizer.step());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ZeROStage1MultiDTypeTest);

// ============================================================================
// CPU-only, dtype-specific tests that don't fit the parameterized fixture
// (the parameterized fixture skips CPU, so these use a plain TEST_F fixture).
// ============================================================================

class ZeROStage1cpuDTypeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    auto create_fp16_params(size_t count, const std::vector<int64_t>& shape)
        -> std::vector<std::shared_ptr<Variable>> {
        std::vector<std::shared_ptr<Variable>> params;
        params.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            auto t = tenzor::ones(shape, DType::Float16, Device::cpu());
            params.push_back(std::make_shared<Variable>(t, true));
        }
        return params;
    }
};

TEST_F(ZeROStage1cpuDTypeTest, ElementLevel_MasterFP32_FP16Param) {
    auto params = create_fp16_params(2, {16, 16});
    auto base = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config cfg;
    cfg.world_size = 1;
    cfg.rank = 0;
    cfg.partitioning_mode = PartitioningMode::ElementLevel;
    cfg.use_master_fp32 = true;
    cfg.state_dtype = DType::Float32;

    ZeROStage1Optimizer opt(std::move(base), cfg);

    for (int step = 0; step < 5; ++step) {
        for (auto& p : params) {
            Tensor g = ones_like(p->tensor()) * 0.01f;
            p->set_grad(g);
        }
        opt.step();
    }

    // Sanity: param dtype preserved after fp16 round-trip through master-fp32.
    EXPECT_EQ(params[0]->tensor().dtype(), DType::Float16);

    // Spot-check param is finite (the fp16 round-trip + Adam math should
    // produce valid fp16 values after 5 steps of lr=0.001, grad=0.01).
    Tensor flat = params[0]->tensor().contiguous().view({-1}).to(DType::Float32).to(Device::cpu());
    const float* d = flat.data<float>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(d[i])) << "non-finite value at index " << i;
    }
}
