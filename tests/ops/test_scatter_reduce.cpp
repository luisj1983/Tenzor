/**
 * @file test_scatter_reduce.cpp
 * @brief Tests for scatter_reduce operation (sum, prod, mean, amax, amin)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/indexing.hpp>
#include <cmath>

using namespace tenzor;

class ScatterReduceTest : public ::testing::Test {
protected:
    static bool initialized;
    void SetUp() override {
        if (!initialized) {
            tenzor::initialize();
            initialized = true;
        }
    }
};

bool ScatterReduceTest::initialized = false;

// ============================================================================
// scatter_reduce with "sum" mode
// ============================================================================

TEST_F(ScatterReduceTest, SumBasic) {
    // input: [1, 2, 3, 4, 5] (1D, 5 elements)
    // index: [0, 1, 0, 1, 0] -> scatter src into positions 0 and 1
    // src:   [10, 20, 30, 40, 50]
    // Expected: output[0] = 1 + 10 + 30 + 50 = 91, output[1] = 2 + 20 + 40 = 62, rest unchanged
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

    EXPECT_FLOAT_EQ(out[0], 91.0f);  // 1 + 10 + 30 + 50
    EXPECT_FLOAT_EQ(out[1], 62.0f);  // 2 + 20 + 40
    EXPECT_FLOAT_EQ(out[2], 3.0f);   // unchanged
    EXPECT_FLOAT_EQ(out[3], 4.0f);   // unchanged
    EXPECT_FLOAT_EQ(out[4], 5.0f);   // unchanged
}

TEST_F(ScatterReduceTest, SumNoIncludeSelf) {
    auto input = Tensor({5}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = 1; in_data[1] = 2; in_data[2] = 3; in_data[3] = 4; in_data[4] = 5;

    auto index_t = Tensor({5}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 1; idx[2] = 0; idx[3] = 1; idx[4] = 0;

    auto src = Tensor({5}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 10; src_data[1] = 20; src_data[2] = 30; src_data[3] = 40; src_data[4] = 50;

    auto result = scatter_reduce(input, 0, index_t, src, "sum", /*include_self=*/false);
    auto* out = result.data<float>();

    // include_self=false: touched positions start at 0 (sum identity)
    EXPECT_FLOAT_EQ(out[0], 90.0f);  // 0 + 10 + 30 + 50
    EXPECT_FLOAT_EQ(out[1], 60.0f);  // 0 + 20 + 40
    EXPECT_FLOAT_EQ(out[2], 3.0f);   // untouched, keeps original
    EXPECT_FLOAT_EQ(out[3], 4.0f);   // untouched
    EXPECT_FLOAT_EQ(out[4], 5.0f);   // untouched
}

// ============================================================================
// scatter_reduce with "prod" mode
// ============================================================================

TEST_F(ScatterReduceTest, ProdBasic) {
    auto input = ones({4}, DType::Float32, Device::cpu());
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

    // output[0] = 2 * 5 * 4 = 40, output[1] = 3 * 6 = 18
    EXPECT_FLOAT_EQ(out[0], 40.0f);
    EXPECT_FLOAT_EQ(out[1], 18.0f);
    EXPECT_FLOAT_EQ(out[2], 1.0f);  // unchanged
    EXPECT_FLOAT_EQ(out[3], 1.0f);  // unchanged
}

// ============================================================================
// scatter_reduce with "amax" mode
// ============================================================================

TEST_F(ScatterReduceTest, AmaxBasic) {
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

    // output[0] = max(-10, 3, 7) = 7
    // output[1] = max(-10, 2, 5) = 5
    EXPECT_FLOAT_EQ(out[0], 7.0f);
    EXPECT_FLOAT_EQ(out[1], 5.0f);
    EXPECT_FLOAT_EQ(out[2], -10.0f);
    EXPECT_FLOAT_EQ(out[3], -10.0f);
}

// ============================================================================
// scatter_reduce with "amin" mode
// ============================================================================

TEST_F(ScatterReduceTest, AminBasic) {
    auto input = Tensor({4}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = 100; in_data[1] = 100; in_data[2] = 100; in_data[3] = 100;

    auto index_t = Tensor({4}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 0; idx[2] = 1; idx[3] = 1;

    auto src = Tensor({4}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 3; src_data[1] = 7; src_data[2] = 2; src_data[3] = 5;

    auto result = scatter_reduce(input, 0, index_t, src, "amin");
    auto* out = result.data<float>();

    // output[0] = min(100, 3, 7) = 3
    // output[1] = min(100, 2, 5) = 2
    EXPECT_FLOAT_EQ(out[0], 3.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 100.0f);
    EXPECT_FLOAT_EQ(out[3], 100.0f);
}

// ============================================================================
// scatter_reduce with "mean" mode
// ============================================================================

TEST_F(ScatterReduceTest, MeanBasic) {
    auto input = Tensor({3}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = 10; in_data[1] = 20; in_data[2] = 30;

    auto index_t = Tensor({4}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 0; idx[2] = 1; idx[3] = 1;

    auto src = Tensor({4}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 2; src_data[1] = 4; src_data[2] = 6; src_data[3] = 12;

    auto result = scatter_reduce(input, 0, index_t, src, "mean");
    auto* out = result.data<float>();

    // output[0] = mean(10, 2, 4) = 16/3 = 5.333...
    // output[1] = mean(20, 6, 12) = 38/3 = 12.666...
    // output[2] = 30 (unchanged, count=1 so no division)
    EXPECT_NEAR(out[0], 16.0f / 3.0f, 1e-5);
    EXPECT_NEAR(out[1], 38.0f / 3.0f, 1e-5);
    EXPECT_FLOAT_EQ(out[2], 30.0f);
}

// ============================================================================
// 2D scatter_reduce along dim=1
// ============================================================================

TEST_F(ScatterReduceTest, Sum2D) {
    // input: [[1, 2, 3], [4, 5, 6]]  (2x3)
    // index: [[0, 0], [1, 2]]  (2x2) - scatter along dim=1
    // src:   [[10, 20], [30, 40]]

    auto input = Tensor({2, 3}, DType::Float32, Device::cpu());
    auto* in_data = input.data<float>();
    in_data[0] = 1; in_data[1] = 2; in_data[2] = 3;
    in_data[3] = 4; in_data[4] = 5; in_data[5] = 6;

    auto index_t = Tensor({2, 2}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 0;  // row 0: both go to col 0
    idx[2] = 1; idx[3] = 2;  // row 1: go to col 1 and 2

    auto src = Tensor({2, 2}, DType::Float32, Device::cpu());
    auto* src_data = src.data<float>();
    src_data[0] = 10; src_data[1] = 20;
    src_data[2] = 30; src_data[3] = 40;

    auto result = scatter_reduce(input, 1, index_t, src, "sum");
    auto* out = result.data<float>();

    // Row 0: [1+10+20, 2, 3] = [31, 2, 3]
    // Row 1: [4, 5+30, 6+40] = [4, 35, 46]
    EXPECT_FLOAT_EQ(out[0], 31.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);
    EXPECT_FLOAT_EQ(out[4], 35.0f);
    EXPECT_FLOAT_EQ(out[5], 46.0f);
}

// ============================================================================
// Float64 dtype
// ============================================================================

TEST_F(ScatterReduceTest, SumFloat64) {
    auto input = zeros({3}, DType::Float64, Device::cpu());
    auto index_t = Tensor({3}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 0; idx[2] = 1;

    auto src = ones({3}, DType::Float64, Device::cpu());
    auto result = scatter_reduce(input, 0, index_t, src, "sum");
    auto* out = result.data<double>();

    EXPECT_DOUBLE_EQ(out[0], 2.0);
    EXPECT_DOUBLE_EQ(out[1], 1.0);
    EXPECT_DOUBLE_EQ(out[2], 0.0);
}

// ============================================================================
// Edge case: empty index
// ============================================================================

TEST_F(ScatterReduceTest, EmptyIndex) {
    auto input = ones({5}, DType::Float32, Device::cpu());
    auto index_t = Tensor({0}, DType::Int64, Device::cpu());
    auto src = Tensor({0}, DType::Float32, Device::cpu());

    auto result = scatter_reduce(input, 0, index_t, src, "sum");
    EXPECT_EQ(result.numel(), 5);
    auto* out = result.data<float>();
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(out[i], 1.0f);
    }
}

// ============================================================================
// Invalid reduce mode
// ============================================================================

TEST_F(ScatterReduceTest, InvalidReduceMode) {
    auto input = ones({5}, DType::Float32, Device::cpu());
    auto index_t = Tensor({3}, DType::Int64, Device::cpu());
    auto* idx = index_t.data<int64_t>();
    idx[0] = 0; idx[1] = 1; idx[2] = 2;
    auto src = ones({3}, DType::Float32, Device::cpu());

    EXPECT_THROW(scatter_reduce(input, 0, index_t, src, "invalid_mode"), std::invalid_argument);
}
