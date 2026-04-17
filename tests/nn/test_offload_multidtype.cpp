/**
 * @file test_offload_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for ZeRO Stage 2 Parameter Offloading
 *
 * These tests require multi-device setups (GPU offload). CPU and single-device backends are skipped.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/offload.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class OffloadMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "Offload tests require non-CPU multi-device backend";
        }
    }

    auto createTestModule(int input_size, int hidden_size, int output_size)
        -> std::unique_ptr<Sequential> {
        auto model = std::make_unique<Sequential>();
        auto linear1 = std::make_shared<Linear>(input_size, hidden_size);
        auto linear2 = std::make_shared<Linear>(hidden_size, output_size);
        model->add_module(linear1);
        model->add_module(linear2);
        return model;
    }
};

TEST_P(OffloadMultiDTypeTest, OffloadContextConstructor) {
    auto model = createTestModule(128, 256, 10);
    convert_model(model);

    OffloadContext::Config config;
    config.offload_parameters = true;
    config.offload_gradients = false;
    config.offload_threshold = 0;
    config.prefetch_depth = 1;

    EXPECT_NO_THROW({
        OffloadContext ctx(*model, config);
    });
}

TEST_P(OffloadMultiDTypeTest, OffloadContextEnableDisable) {
    auto model = createTestModule(128, 256, 10);
    convert_model(model);

    OffloadContext::Config config;
    config.offload_parameters = true;
    config.offload_threshold = 0;

    OffloadContext ctx(*model, config);

    EXPECT_FALSE(ctx.is_enabled());
    ctx.enable();
    EXPECT_TRUE(ctx.is_enabled());
    ctx.disable();
    EXPECT_FALSE(ctx.is_enabled());
}

TEST_P(OffloadMultiDTypeTest, OffloadContextGetStats) {
    auto model = createTestModule(128, 256, 10);
    convert_model(model);

    OffloadContext::Config config;
    config.offload_parameters = true;
    config.offload_threshold = 0;

    OffloadContext ctx(*model, config);
    ctx.enable();

    auto stats = ctx.get_stats();
    EXPECT_GE(stats.num_parameters_offloaded, 0);
}

TEST_P(OffloadMultiDTypeTest, ModelParametersExist) {
    auto model = createTestModule(64, 128, 10);
    convert_model(model);

    auto params = model->parameters();
    EXPECT_FALSE(params.empty());
}

TEST_P(OffloadMultiDTypeTest, OffloadContextMultipleLayers) {
    auto model = createTestModule(32, 64, 16);
    convert_model(model);

    OffloadContext::Config config;
    config.offload_parameters = true;
    config.offload_threshold = 0;

    EXPECT_NO_THROW({
        OffloadContext ctx(*model, config);
        ctx.enable();
        ctx.disable();
    });
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(OffloadMultiDTypeTest);
