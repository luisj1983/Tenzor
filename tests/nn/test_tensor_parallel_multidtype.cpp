/**
 * @file test_tensor_parallel_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for tensor parallelism layers
 *
 * These tests require multi-device setups. CPU and single-device backends are skipped.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/tensor_parallel.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/ops/creation.hpp>
#include <memory>

using namespace tenzor;
namespace dist = tenzor::distributed;
using namespace tenzor::testing;

class TensorParallelMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "Tensor parallel tests require non-CPU multi-device backend";
        }
    }
};

TEST_P(TensorParallelMultiDTypeTest, ColumnParallelConstruction) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29750);

    EXPECT_NO_THROW({
        dist::ColumnParallelLinear layer(16, 32, *pg, true, true);
    });
}

TEST_P(TensorParallelMultiDTypeTest, ColumnParallelOutputShape) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29751);

    dist::ColumnParallelLinear layer(16, 32, *pg, true, true);

    auto input = createInput({4, 16}, false);
    Variable output = layer.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 32);
}

TEST_P(TensorParallelMultiDTypeTest, RowParallelConstruction) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29752);

    EXPECT_NO_THROW({
        dist::RowParallelLinear layer(32, 16, *pg, true, true);
    });
}

TEST_P(TensorParallelMultiDTypeTest, RowParallelOutputShape) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29753);

    dist::RowParallelLinear layer(32, 16, *pg, true, true);

    auto input = createInput({4, 32}, false);
    Variable output = layer.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 16);
}

TEST_P(TensorParallelMultiDTypeTest, ParallelAttentionConstruction) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29754);

    EXPECT_NO_THROW({
        dist::ParallelAttention attn(64, 4, *pg);
    });
}

TEST_P(TensorParallelMultiDTypeTest, ColumnParallelLocalFeatures) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29755);

    dist::ColumnParallelLinear layer(16, 32, *pg);

    EXPECT_EQ(layer.in_features(), 16);
    EXPECT_EQ(layer.out_features(), 32);
    EXPECT_EQ(layer.local_out_features(), 32);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(TensorParallelMultiDTypeTest);
