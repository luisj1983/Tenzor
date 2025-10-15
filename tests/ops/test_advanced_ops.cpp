/**
 * @file test_advanced_ops.cpp
 * @brief Tests for Phase 6 advanced tensor operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <cmath>

using namespace tenzor;

class AdvancedOpsTest : public ::testing::Test {
protected:
    Device cpu = Device::cpu();
};

// ============================================================================
// Expand Tests
// ============================================================================

TEST_F(AdvancedOpsTest, ExpandBroadcast) {
    auto t = ones({1, 3}, DType::Float32, cpu);
    auto expanded = expand(t, {4, 3});

    EXPECT_EQ(expanded.shape()[0], 4);
    EXPECT_EQ(expanded.shape()[1], 3);

    // Verify all values are 1
    auto data = expanded.contiguous().data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_F(AdvancedOpsTest, ExpandInvalidShape) {
    auto t = ones({2, 3}, DType::Float32, cpu);

    // Cannot expand dimension 2 to 4 (not singleton)
    EXPECT_THROW(expand(t, {4, 3}), std::runtime_error);
}

// ============================================================================
// TopK Tests
// ============================================================================

TEST_F(AdvancedOpsTest, TopKLargest) {
    auto t = Tensor({5}, DType::Float32, cpu);
    auto data = t.data<float>();
    data[0] = 3.0f;
    data[1] = 1.0f;
    data[2] = 4.0f;
    data[3] = 2.0f;
    data[4] = 5.0f;

    auto [values, indices] = topk(t, 3, 0, true, true);

    EXPECT_EQ(values.shape()[0], 3);
    EXPECT_EQ(indices.shape()[0], 3);

    auto vals = values.data<float>();
    auto idxs = indices.data<int64_t>();

    EXPECT_FLOAT_EQ(vals[0], 5.0f);
    EXPECT_FLOAT_EQ(vals[1], 4.0f);
    EXPECT_FLOAT_EQ(vals[2], 3.0f);

    EXPECT_EQ(idxs[0], 4);
    EXPECT_EQ(idxs[1], 2);
    EXPECT_EQ(idxs[2], 0);
}

TEST_F(AdvancedOpsTest, TopKSmallest) {
    auto t = Tensor({5}, DType::Float32, cpu);
    auto data = t.data<float>();
    data[0] = 3.0f;
    data[1] = 1.0f;
    data[2] = 4.0f;
    data[3] = 2.0f;
    data[4] = 5.0f;

    auto [values, indices] = topk(t, 2, 0, false, true);

    auto vals = values.data<float>();
    EXPECT_FLOAT_EQ(vals[0], 1.0f);
    EXPECT_FLOAT_EQ(vals[1], 2.0f);
}

// ============================================================================
// Sort Tests
// ============================================================================

TEST_F(AdvancedOpsTest, SortAscending) {
    auto t = Tensor({4}, DType::Float32, cpu);
    auto data = t.data<float>();
    data[0] = 3.0f;
    data[1] = 1.0f;
    data[2] = 4.0f;
    data[3] = 2.0f;

    auto [sorted, indices] = sort(t, 0, false);

    auto vals = sorted.data<float>();
    EXPECT_FLOAT_EQ(vals[0], 1.0f);
    EXPECT_FLOAT_EQ(vals[1], 2.0f);
    EXPECT_FLOAT_EQ(vals[2], 3.0f);
    EXPECT_FLOAT_EQ(vals[3], 4.0f);

    auto idxs = indices.data<int64_t>();
    EXPECT_EQ(idxs[0], 1);
    EXPECT_EQ(idxs[1], 3);
    EXPECT_EQ(idxs[2], 0);
    EXPECT_EQ(idxs[3], 2);
}

TEST_F(AdvancedOpsTest, SortDescending) {
    auto t = Tensor({4}, DType::Float32, cpu);
    auto data = t.data<float>();
    data[0] = 1.0f;
    data[1] = 4.0f;
    data[2] = 2.0f;
    data[3] = 3.0f;

    auto [sorted, indices] = sort(t, 0, true);

    auto vals = sorted.data<float>();
    EXPECT_FLOAT_EQ(vals[0], 4.0f);
    EXPECT_FLOAT_EQ(vals[1], 3.0f);
    EXPECT_FLOAT_EQ(vals[2], 2.0f);
    EXPECT_FLOAT_EQ(vals[3], 1.0f);
}

// ============================================================================
// Unique Tests
// ============================================================================

TEST_F(AdvancedOpsTest, UniqueBasic) {
    auto t = Tensor({6}, DType::Float32, cpu);
    auto data = t.data<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 1.0f;
    data[3] = 3.0f;
    data[4] = 2.0f;
    data[5] = 3.0f;

    auto [unique_vals, inverse, counts] = unique(t, true, false, false);

    EXPECT_EQ(unique_vals.numel(), 3);

    auto vals = unique_vals.data<float>();
    EXPECT_FLOAT_EQ(vals[0], 1.0f);
    EXPECT_FLOAT_EQ(vals[1], 2.0f);
    EXPECT_FLOAT_EQ(vals[2], 3.0f);
}

TEST_F(AdvancedOpsTest, UniqueWithCounts) {
    auto t = Tensor({5}, DType::Float32, cpu);
    auto data = t.data<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 1.0f;
    data[3] = 1.0f;
    data[4] = 2.0f;

    auto [unique_vals, inverse, counts_tensor] = unique(t, true, false, true);

    EXPECT_EQ(unique_vals.numel(), 2);

    auto counts_data = counts_tensor.data<int64_t>();
    EXPECT_EQ(counts_data[0], 3);  // 1.0 appears 3 times
    EXPECT_EQ(counts_data[1], 2);  // 2.0 appears 2 times
}

// ============================================================================
// Cumsum Tests
// ============================================================================

TEST_F(AdvancedOpsTest, CumsumBasic) {
    auto t = Tensor({4}, DType::Float32, cpu);
    auto data = t.data<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    auto result = cumsum(t, 0);

    auto res_data = result.data<float>();
    EXPECT_FLOAT_EQ(res_data[0], 1.0f);
    EXPECT_FLOAT_EQ(res_data[1], 3.0f);
    EXPECT_FLOAT_EQ(res_data[2], 6.0f);
    EXPECT_FLOAT_EQ(res_data[3], 10.0f);
}

TEST_F(AdvancedOpsTest, Cumsum2D) {
    auto t = Tensor({2, 3}, DType::Float32, cpu);
    auto data = t.data<float>();
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    auto result = cumsum(t, 1);

    auto res_data = result.data<float>();
    EXPECT_FLOAT_EQ(res_data[0], 1.0f);
    EXPECT_FLOAT_EQ(res_data[1], 3.0f);
    EXPECT_FLOAT_EQ(res_data[2], 6.0f);
    EXPECT_FLOAT_EQ(res_data[3], 4.0f);
    EXPECT_FLOAT_EQ(res_data[4], 9.0f);
    EXPECT_FLOAT_EQ(res_data[5], 15.0f);
}

// ============================================================================
// Cumprod Tests
// ============================================================================

TEST_F(AdvancedOpsTest, CumprodBasic) {
    auto t = Tensor({4}, DType::Float32, cpu);
    auto data = t.data<float>();
    data[0] = 2.0f;
    data[1] = 3.0f;
    data[2] = 4.0f;
    data[3] = 5.0f;

    auto result = cumprod(t, 0);

    auto res_data = result.data<float>();
    EXPECT_FLOAT_EQ(res_data[0], 2.0f);
    EXPECT_FLOAT_EQ(res_data[1], 6.0f);
    EXPECT_FLOAT_EQ(res_data[2], 24.0f);
    EXPECT_FLOAT_EQ(res_data[3], 120.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
