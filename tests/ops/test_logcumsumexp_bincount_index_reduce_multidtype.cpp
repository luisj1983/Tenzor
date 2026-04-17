/**
 * @file test_logcumsumexp_bincount_index_reduce_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for logcumsumexp, bincount, index_reduce
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/indexing.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class LogcumsumexpBincountMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(LogcumsumexpBincountMultiDTypeTest, LogcumsumexpBasic1D) {
    auto input = Tensor({4}, DType::Float32, Device::cpu());
    auto* data = input.data<float>();
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;

    auto result = logcumsumexp(input, 0);
    auto* out = result.data<float>();

    EXPECT_NEAR(out[0], 1.0f, 1e-5f);
    float expected1 = std::log(std::exp(1.0f) + std::exp(2.0f));
    EXPECT_NEAR(out[1], expected1, 1e-5f);
}

TEST_P(LogcumsumexpBincountMultiDTypeTest, LogcumsumexpNumericalStability) {
    auto input = Tensor({3}, DType::Float32, Device::cpu());
    auto* data = input.data<float>();
    data[0] = 100.0f; data[1] = 101.0f; data[2] = 102.0f;

    auto result = logcumsumexp(input, 0);
    auto* out = result.data<float>();

    EXPECT_NEAR(out[0], 100.0f, 1e-4f);
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_TRUE(std::isfinite(out[1]));
    EXPECT_TRUE(std::isfinite(out[2]));
}

TEST_P(LogcumsumexpBincountMultiDTypeTest, BincountBasic) {
    auto input = Tensor({6}, DType::Int64, Device::cpu());
    auto* data = input.data<int64_t>();
    data[0] = 0; data[1] = 1; data[2] = 1; data[3] = 3; data[4] = 0; data[5] = 3;

    auto result = bincount(input);
    EXPECT_EQ(result.numel(), 4);
    auto* out = result.data<int64_t>();
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 2);
    EXPECT_EQ(out[2], 0);
    EXPECT_EQ(out[3], 2);
}

TEST_P(LogcumsumexpBincountMultiDTypeTest, BincountMinlength) {
    auto input = Tensor({3}, DType::Int64, Device::cpu());
    auto* data = input.data<int64_t>();
    data[0] = 0; data[1] = 1; data[2] = 0;

    auto result = bincount(input, std::nullopt, 5);
    EXPECT_EQ(result.numel(), 5);
}

TEST_P(LogcumsumexpBincountMultiDTypeTest, IndexReduceSum) {
    auto input = Tensor({5}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = 1; in_data[1] = 2; in_data[2] = 3; in_data[3] = 4; in_data[4] = 5;

    auto index_t = Tensor({5}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 1; idx[2] = 0; idx[3] = 1; idx[4] = 0;

    auto src = Tensor({5}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 10; src_data[1] = 20; src_data[2] = 30; src_data[3] = 40; src_data[4] = 50;

    auto result = index_reduce(input, 0, index_t, src, "sum");
    auto* out = result.data<float>();

    EXPECT_FLOAT_EQ(out[0], 91.0f);
    EXPECT_FLOAT_EQ(out[1], 62.0f);
}

TEST_P(LogcumsumexpBincountMultiDTypeTest, Logcumsumexp2D) {
    auto input = Tensor({2, 3}, DType::Float32, Device::cpu());
    auto* data = input.data<float>();
    data[0] = 0.0f; data[1] = 1.0f; data[2] = 2.0f;
    data[3] = 3.0f; data[4] = 4.0f; data[5] = 5.0f;

    auto result = logcumsumexp(input, 1);
    auto* out = result.data<float>();

    EXPECT_NEAR(out[0], 0.0f, 1e-5f);
    EXPECT_NEAR(out[3], 3.0f, 1e-5f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LogcumsumexpBincountMultiDTypeTest);
