/**
 * @file test_scatter_reduce_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for scatter_reduce operation
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/indexing.hpp>

using namespace tenzor;
using namespace tenzor::testing;

class ScatterReduceMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ScatterReduceMultiDTypeTest, SumBasic) {
    auto input = Tensor({5}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = 1; in_data[1] = 2; in_data[2] = 3; in_data[3] = 4; in_data[4] = 5;

    auto index_t = Tensor({5}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 1; idx[2] = 0; idx[3] = 1; idx[4] = 0;

    auto src = Tensor({5}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 10; src_data[1] = 20; src_data[2] = 30; src_data[3] = 40; src_data[4] = 50;

    auto result = scatter_reduce(input, 0, index_t, src, "sum");
    auto* out = result.data<float>();

    EXPECT_FLOAT_EQ(out[0], 91.0f);
    EXPECT_FLOAT_EQ(out[1], 62.0f);
}

TEST_P(ScatterReduceMultiDTypeTest, ProdBasic) {
    auto input = tenzor::ones({4}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = 2; in_data[1] = 3;

    auto index_t = Tensor({3}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 0; idx[2] = 1;

    auto src = Tensor({3}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 5; src_data[1] = 4; src_data[2] = 6;

    auto result = scatter_reduce(input, 0, index_t, src, "prod");
    auto* out = result.data<float>();

    EXPECT_FLOAT_EQ(out[0], 40.0f);
    EXPECT_FLOAT_EQ(out[1], 18.0f);
}

TEST_P(ScatterReduceMultiDTypeTest, AmaxBasic) {
    auto input = Tensor({4}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = -10; in_data[1] = -10; in_data[2] = -10; in_data[3] = -10;

    auto index_t = Tensor({4}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 0; idx[2] = 1; idx[3] = 1;

    auto src = Tensor({4}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 3; src_data[1] = 7; src_data[2] = 2; src_data[3] = 5;

    auto result = scatter_reduce(input, 0, index_t, src, "amax");
    auto* out = result.data<float>();

    EXPECT_FLOAT_EQ(out[0], 7.0f);
    EXPECT_FLOAT_EQ(out[1], 5.0f);
}

TEST_P(ScatterReduceMultiDTypeTest, EmptyIndex) {
    auto input = tenzor::ones({5}, DType::Float32, Device::cpu());
    auto index_t = Tensor({0}, DType::Int64, Device::cpu());
    auto src = Tensor({0}, DType::Float32, Device::cpu());

    auto result = scatter_reduce(input, 0, index_t, src, "sum");
    EXPECT_EQ(result.numel(), 5);
}

TEST_P(ScatterReduceMultiDTypeTest, InvalidReduceMode) {
    auto input = tenzor::ones({5}, DType::Float32, Device::cpu());
    auto index_t = Tensor({3}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 1; idx[2] = 2;
    auto src = tenzor::ones({3}, DType::Float32, Device::cpu());

    EXPECT_THROW(scatter_reduce(input, 0, index_t, src, "invalid_mode"), std::invalid_argument);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ScatterReduceMultiDTypeTest);
