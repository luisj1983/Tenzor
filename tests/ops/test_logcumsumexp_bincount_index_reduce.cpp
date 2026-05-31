/**
 * @file test_logcumsumexp_bincount_index_reduce.cpp
 * @brief Tests for logcumsumexp, bincount, and index_reduce operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/indexing.hpp>
#include "../backend_test_fixture.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;

class NewOpsTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// logcumsumexp tests
// ============================================================================

TEST_P(NewOpsTest, LogcumsumexpBasic1D) {
    auto input = Tensor({4}, DType::Float32, Device::cpu());
    auto* data = input.data<float>();
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;
    input = input.to(device);

    auto result = logcumsumexp(input, 0);
    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<float>();

    // output[0] = log(exp(1)) = 1
    EXPECT_NEAR(out[0], 1.0f, 1e-5f);
    // output[1] = log(exp(1) + exp(2))
    float expected1 = std::log(std::exp(1.0f) + std::exp(2.0f));
    EXPECT_NEAR(out[1], expected1, 1e-5f);
    // output[2] = log(exp(1) + exp(2) + exp(3))
    float expected2 = std::log(std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f));
    EXPECT_NEAR(out[2], expected2, 1e-5f);
    // output[3] = log(exp(1) + exp(2) + exp(3) + exp(4))
    float expected3 = std::log(std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f) + std::exp(4.0f));
    EXPECT_NEAR(out[3], expected3, 1e-5f);
}

TEST_P(NewOpsTest, LogcumsumexpNumericalStability) {
    // Large values that would overflow with naive exp
    auto input = Tensor({3}, DType::Float32, Device::cpu());
    auto* data = input.data<float>();
    data[0] = 100.0f; data[1] = 101.0f; data[2] = 102.0f;
    input = input.to(device);

    auto result = logcumsumexp(input, 0);
    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<float>();

    // output[0] = 100
    EXPECT_NEAR(out[0], 100.0f, 1e-4f);
    // output[1] = log(exp(100) + exp(101)) = 101 + log(exp(-1) + 1)
    float expected1 = 101.0f + std::log(std::exp(-1.0f) + 1.0f);
    EXPECT_NEAR(out[1], expected1, 1e-4f);
    // All results should be finite
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_TRUE(std::isfinite(out[1]));
    EXPECT_TRUE(std::isfinite(out[2]));
}

TEST_P(NewOpsTest, Logcumsumexp2D) {
    // 2x3 tensor, cumsum-exp along dim=1
    auto input = Tensor({2, 3}, DType::Float32, Device::cpu());
    auto* data = input.data<float>();
    // Row 0: [0, 1, 2]
    data[0] = 0.0f; data[1] = 1.0f; data[2] = 2.0f;
    // Row 1: [3, 4, 5]
    data[3] = 3.0f; data[4] = 4.0f; data[5] = 5.0f;
    input = input.to(device);

    auto result = logcumsumexp(input, 1);
    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<float>();

    // Row 0, position 0: log(exp(0)) = 0
    EXPECT_NEAR(out[0], 0.0f, 1e-5f);
    // Row 0, position 1: log(exp(0) + exp(1))
    float expected_01 = std::log(std::exp(0.0f) + std::exp(1.0f));
    EXPECT_NEAR(out[1], expected_01, 1e-5f);
    // Row 1, position 0: log(exp(3)) = 3
    EXPECT_NEAR(out[3], 3.0f, 1e-5f);
}

TEST_P(NewOpsTest, LogcumsumexpFloat64) {
    auto input = Tensor({3}, DType::Float64, Device::cpu());
    auto* data = input.data<double>();
    data[0] = 1.0; data[1] = 2.0; data[2] = 3.0;
    input = input.to(device);

    auto result = logcumsumexp(input, 0);
    EXPECT_EQ(result.dtype(), DType::Float64);
    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<double>();

    EXPECT_NEAR(out[0], 1.0, 1e-10);
    double expected1 = std::log(std::exp(1.0) + std::exp(2.0));
    EXPECT_NEAR(out[1], expected1, 1e-10);
}

// ============================================================================
// bincount tests
// ============================================================================

TEST_P(NewOpsTest, BincountBasic) {
    auto input = Tensor({6}, DType::Int64, Device::cpu());
    auto* data = input.data<int64_t>();
    data[0] = 0; data[1] = 1; data[2] = 1; data[3] = 3; data[4] = 0; data[5] = 3;
    input = input.to(device);

    auto result = bincount(input);
    EXPECT_EQ(result.dtype(), DType::Int64);
    EXPECT_EQ(result.numel(), 4); // max=3, so size=4

    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<int64_t>();
    EXPECT_EQ(out[0], 2);  // 0 appears 2 times
    EXPECT_EQ(out[1], 2);  // 1 appears 2 times
    EXPECT_EQ(out[2], 0);  // 2 appears 0 times
    EXPECT_EQ(out[3], 2);  // 3 appears 2 times
}

TEST_P(NewOpsTest, BincountMinlength) {
    auto input = Tensor({3}, DType::Int64, Device::cpu());
    auto* data = input.data<int64_t>();
    data[0] = 0; data[1] = 1; data[2] = 0;
    input = input.to(device);

    auto result = bincount(input, std::nullopt, 5);
    EXPECT_EQ(result.numel(), 5); // minlength=5 > max+1=2

    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<int64_t>();
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 1);
    EXPECT_EQ(out[2], 0);
    EXPECT_EQ(out[3], 0);
    EXPECT_EQ(out[4], 0);
}

TEST_P(NewOpsTest, BincountWithWeights) {
    auto input = Tensor({4}, DType::Int64, Device::cpu());
    auto* data = input.data<int64_t>();
    data[0] = 0; data[1] = 1; data[2] = 0; data[3] = 2;
    input = input.to(device);

    auto weights = Tensor({4}, DType::Float32, Device::cpu());
    auto* w = weights.data<float>();
    w[0] = 0.5f; w[1] = 1.0f; w[2] = 1.5f; w[3] = 2.0f;
    weights = weights.to(device);

    auto result = bincount(input, weights);
    EXPECT_EQ(result.dtype(), DType::Float64);
    EXPECT_EQ(result.numel(), 3);

    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<double>();
    EXPECT_NEAR(out[0], 2.0, 1e-10);   // 0.5 + 1.5
    EXPECT_NEAR(out[1], 1.0, 1e-10);   // 1.0
    EXPECT_NEAR(out[2], 2.0, 1e-10);   // 2.0
}

TEST_P(NewOpsTest, BincountInt32) {
    auto input = Tensor({3}, DType::Int32, Device::cpu());
    auto* data = input.data<int32_t>();
    data[0] = 2; data[1] = 2; data[2] = 0;
    input = input.to(device);

    auto result = bincount(input);
    EXPECT_EQ(result.numel(), 3);

    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<int64_t>();
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 0);
    EXPECT_EQ(out[2], 2);
}

// ============================================================================
// index_reduce tests
// ============================================================================

TEST_P(NewOpsTest, IndexReduceSum) {
    // index_reduce is a wrapper around scatter_reduce
    auto input = Tensor({5}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = 1; in_data[1] = 2; in_data[2] = 3; in_data[3] = 4; in_data[4] = 5;
    input = input.to(device);

    auto index_t = Tensor({5}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 1; idx[2] = 0; idx[3] = 1; idx[4] = 0;
    index_t = index_t.to(device);

    auto src = Tensor({5}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 10; src_data[1] = 20; src_data[2] = 30; src_data[3] = 40; src_data[4] = 50;
    src = src.to(device);

    auto result = index_reduce(input, 0, index_t, src, "sum");
    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<float>();

    EXPECT_FLOAT_EQ(out[0], 91.0f);  // 1 + 10 + 30 + 50
    EXPECT_FLOAT_EQ(out[1], 62.0f);  // 2 + 20 + 40
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);
    EXPECT_FLOAT_EQ(out[4], 5.0f);
}

TEST_P(NewOpsTest, IndexReduceProd) {
    auto input = Tensor({3}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = 2.0f; in_data[1] = 3.0f; in_data[2] = 1.0f;
    input = input.to(device);

    auto index_t = Tensor({2}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 0;
    index_t = index_t.to(device);

    auto src = Tensor({2}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 5.0f; src_data[1] = 3.0f;
    src = src.to(device);

    auto result = index_reduce(input, 0, index_t, src, "prod");
    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<float>();

    // output[0] = 2.0 * 5.0 * 3.0 = 30.0
    EXPECT_FLOAT_EQ(out[0], 30.0f);
    EXPECT_FLOAT_EQ(out[1], 3.0f);   // unchanged
    EXPECT_FLOAT_EQ(out[2], 1.0f);   // unchanged
}

TEST_P(NewOpsTest, IndexReduceNoIncludeSelf) {
    auto input = Tensor({3}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = 100.0f; in_data[1] = 200.0f; in_data[2] = 300.0f;
    input = input.to(device);

    auto index_t = Tensor({2}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 1;
    index_t = index_t.to(device);

    auto src = Tensor({2}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 10.0f; src_data[1] = 20.0f;
    src = src.to(device);

    auto result = index_reduce(input, 0, index_t, src, "sum", false);
    auto result_cpu = result.cpu();
    auto* out = result_cpu.data<float>();

    // With include_self=false, positions receiving values are initialized to
    // the identity (0 for sum), so output[0] = 10, output[1] = 20
    EXPECT_FLOAT_EQ(out[0], 10.0f);
    EXPECT_FLOAT_EQ(out[1], 20.0f);
    EXPECT_FLOAT_EQ(out[2], 300.0f); // unchanged (no scatter to position 2)
}

INSTANTIATE_BACKEND_TESTS(NewOpsTest);
