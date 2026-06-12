/**
 * @file test_pipeline_parallel_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for PipelineStage and pipeline schedules
 *
 * These tests require multi-device setups. CPU and single-device backends are skipped.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "../grad_flow_helpers.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/pipeline_parallel.hpp>
#include <tenzor/nn/layers/linear.hpp>

using namespace tenzor;
using namespace tenzor::distributed;
using namespace tenzor::testing;

class PipelineParallelMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "Pipeline parallel tests require non-CPU multi-device backend";
        }
    }

    std::shared_ptr<nn::Linear> make_linear(int in, int out) {
        auto layer = std::make_shared<nn::Linear>(in, out);
        convert_model(layer);
        return layer;
    }
};

TEST_P(PipelineParallelMultiDTypeTest, ConstructFirstStage) {
    auto module = make_linear(8, 16);
    PipelineStage stage(module, 0, 4);
    EXPECT_EQ(stage.stage_id(), 0);
    EXPECT_TRUE(stage.is_first());
    EXPECT_FALSE(stage.is_last());
}

TEST_P(PipelineParallelMultiDTypeTest, ConstructLastStage) {
    auto module = make_linear(16, 8);
    PipelineStage stage(module, 3, 4);
    EXPECT_TRUE(stage.is_last());
    EXPECT_FALSE(stage.is_first());
}

TEST_P(PipelineParallelMultiDTypeTest, ForwardProducesExpectedShape) {
    auto module = make_linear(8, 16);
    PipelineStage stage(module, 0, 2);

    auto input = createInput({4, 8}, false);
    auto output = stage.forward(input);

    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 16);
}

TEST_P(PipelineParallelMultiDTypeTest, ForwardPreservesBatchDimension) {
    auto module = make_linear(10, 5);
    PipelineStage stage(module, 1, 3);

    auto input = createInput({7, 10}, false);
    auto output = stage.forward(input);

    EXPECT_EQ(output.tensor().shape()[0], 7);
    EXPECT_EQ(output.tensor().shape()[1], 5);
}

TEST_P(PipelineParallelMultiDTypeTest, ForwardSupportsAutograd) {
    auto module = make_linear(4, 4);
    PipelineStage stage(module, 0, 2);

    auto input = createInput({2, 4}, true);
    auto output = stage.forward(input);

    auto grad = createOnes(std::vector<int64_t>(output.shape().begin(), output.shape().end()));
    EXPECT_NO_THROW(output.backward(grad));
    EXPECT_GRAD_FLOWS(input);
}

TEST_P(PipelineParallelMultiDTypeTest, MultipleStagesComposable) {
    auto m1 = make_linear(4, 4);
    auto m2 = make_linear(4, 4);
    PipelineStage s1(m1, 0, 2);
    PipelineStage s2(m2, 1, 2);

    auto input = createInput({3, 4}, false);
    auto out1 = s1.forward(input);
    auto out2 = s2.forward(out1);

    EXPECT_EQ(out2.tensor().shape()[0], 3);
    EXPECT_EQ(out2.tensor().shape()[1], 4);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(PipelineParallelMultiDTypeTest);
