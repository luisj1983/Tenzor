/**
 * @file test_tensor_parallel.cpp
 * @brief Tests for tensor parallelism layers (ColumnParallel, RowParallel, ParallelAttention)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/tensor_parallel.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/core/dtype.hpp>
#include <memory>
#include <cmath>

using namespace tenzor;
namespace dist = tenzor::distributed;

// ============================================================================
// Test Environment
// ============================================================================

class TensorParallelTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tp_env =
    ::testing::AddGlobalTestEnvironment(new TensorParallelTestEnvironment);

// ============================================================================
// ColumnParallelLinear Tests
// ============================================================================

TEST(ColumnParallelLinearTest, Construction) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29700);

    EXPECT_NO_THROW({
        dist::ColumnParallelLinear layer(16, 32, *pg, true, true);
    });

    EXPECT_NO_THROW({
        dist::ColumnParallelLinear layer(16, 32, *pg, false, true);
    });

    EXPECT_NO_THROW({
        dist::ColumnParallelLinear layer(16, 32, *pg, true, false);
    });
}

TEST(ColumnParallelLinearTest, OutputShape) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29701);

    dist::ColumnParallelLinear layer(16, 32, *pg, true, true);

    Variable input(rand({4, 16}), false);
    Variable output = layer.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 32);
}

TEST(ColumnParallelLinearTest, LocalOutputShape) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29702);

    dist::ColumnParallelLinear layer(16, 32, *pg, true, false);

    Variable input(rand({4, 16}), false);
    Variable output = layer.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 32);  // world_size=1, local = full
}

TEST(ColumnParallelLinearTest, LocalFeatures) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29703);

    dist::ColumnParallelLinear layer(16, 32, *pg);

    EXPECT_EQ(layer.in_features(), 16);
    EXPECT_EQ(layer.out_features(), 32);
    EXPECT_EQ(layer.local_out_features(), 32);
}

TEST(ColumnParallelLinearTest, ExtraRepr) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29704);

    dist::ColumnParallelLinear layer(16, 32, *pg);
    std::string repr = layer.extra_repr();

    EXPECT_NE(repr.find("in_features=16"), std::string::npos);
    EXPECT_NE(repr.find("out_features=32"), std::string::npos);
}

// ============================================================================
// RowParallelLinear Tests
// ============================================================================

TEST(RowParallelLinearTest, Construction) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29710);

    EXPECT_NO_THROW({
        dist::RowParallelLinear layer(32, 16, *pg, true, true);
    });

    EXPECT_NO_THROW({
        dist::RowParallelLinear layer(32, 16, *pg, false, true);
    });

    EXPECT_NO_THROW({
        dist::RowParallelLinear layer(32, 16, *pg, true, false);
    });
}

TEST(RowParallelLinearTest, OutputShape) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29711);

    dist::RowParallelLinear layer(32, 16, *pg, true, true);

    Variable input(rand({4, 32}), false);
    Variable output = layer.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 16);
}

TEST(RowParallelLinearTest, NonParallelInput) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29712);

    dist::RowParallelLinear layer(32, 16, *pg, true, false);

    Variable input(rand({4, 32}), false);
    Variable output = layer.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 16);
}

TEST(RowParallelLinearTest, LocalFeatures) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29713);

    dist::RowParallelLinear layer(32, 16, *pg);

    EXPECT_EQ(layer.in_features(), 32);
    EXPECT_EQ(layer.out_features(), 16);
    EXPECT_EQ(layer.local_in_features(), 32);
}

// ============================================================================
// ParallelAttention Tests
// ============================================================================

TEST(ParallelAttentionTest, Construction) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29720);

    EXPECT_NO_THROW({
        dist::ParallelAttention attn(64, 4, *pg);
    });

    EXPECT_NO_THROW({
        dist::ParallelAttention attn(64, 4, *pg, 0.1f);
    });
}

TEST(ParallelAttentionTest, InvalidConfig) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29721);

    EXPECT_THROW({
        dist::ParallelAttention attn(63, 4, *pg);
    }, std::invalid_argument);
}

TEST(ParallelAttentionTest, OutputShape) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29722);

    dist::ParallelAttention attn(64, 4, *pg);

    Variable input(rand({2, 8, 64}), false);
    Variable output = attn.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 8);
    EXPECT_EQ(shape[2], 64);
}

TEST(ParallelAttentionTest, HeadDistribution) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29723);

    dist::ParallelAttention attn(64, 8, *pg);

    EXPECT_EQ(attn.local_num_heads(), 8);
    EXPECT_EQ(attn.head_dim(), 8);
}

TEST(ParallelAttentionTest, ExtraRepr) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29724);

    dist::ParallelAttention attn(64, 8, *pg, 0.1f);
    std::string repr = attn.extra_repr();

    EXPECT_NE(repr.find("embed_dim=64"), std::string::npos);
    EXPECT_NE(repr.find("num_heads=8"), std::string::npos);
}

// ============================================================================
// Column + Row Pipeline Tests
// ============================================================================

TEST(TensorParallelPipelineTest, ColumnToRow) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29730);

    dist::ColumnParallelLinear col_layer(16, 32, *pg, true, false);
    dist::RowParallelLinear row_layer(32, 8, *pg, true, true);

    Variable input(rand({4, 16}), false);

    Variable col_output = col_layer.forward(input);
    auto col_shape = col_output.tensor().shape();
    EXPECT_EQ(col_shape[1], 32);

    Variable row_output = row_layer.forward(col_output);
    auto row_shape = row_output.tensor().shape();
    EXPECT_EQ(row_shape[0], 4);
    EXPECT_EQ(row_shape[1], 8);
}

// ============================================================================
// Single-rank numerical correctness
// ============================================================================

TEST(ColumnParallelLinearTest, SingleRankEquivalence) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29740);

    dist::ColumnParallelLinear col_layer(8, 16, *pg, true, true);

    Variable input(rand({2, 8}), false);
    Variable output = col_layer.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 16);

    Tensor out_t = output.tensor();
    float sum = 0.0f;
    const float* data = static_cast<const float*>(out_t.data_ptr());
    for (int64_t i = 0; i < out_t.numel(); ++i) {
        sum += std::abs(data[i]);
        EXPECT_FALSE(std::isnan(data[i])) << "NaN at index " << i;
    }
    EXPECT_GT(sum, 0.0f);
}

TEST(RowParallelLinearTest, SingleRankEquivalence) {
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29741);

    dist::RowParallelLinear row_layer(16, 8, *pg, true, true);

    Variable input(rand({2, 16}), false);
    Variable output = row_layer.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 8);

    Tensor out_t = output.tensor();
    float sum = 0.0f;
    const float* data = static_cast<const float*>(out_t.data_ptr());
    for (int64_t i = 0; i < out_t.numel(); ++i) {
        sum += std::abs(data[i]);
        EXPECT_FALSE(std::isnan(data[i])) << "NaN at index " << i;
    }
    EXPECT_GT(sum, 0.0f);
}
