/**
 * @file test_scatter_variants.cpp
 * @brief Unit tests for select_scatter, slice_scatter, diagonal_scatter
 */

#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "../backend_test_fixture.hpp"
#include <vector>

using namespace tenzor;

class ScatterVariantsTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// select_scatter tests
// ============================================================================

TEST_P(ScatterVariantsTest, SelectScatterBasic) {
    // Create a 3x4 zeros tensor, scatter a row of ones at index 1 along dim 0
    auto input = zeros({3, 4}, DType::Float32, device);
    auto src = ones({4}, DType::Float32, device);

    auto result = select_scatter(input, src, /*dim=*/0, /*index=*/1);

    // Shape should be preserved
    ASSERT_EQ(result.shape().size(), 2u);
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 4);

    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<float>();
    // Row 0: all zeros
    for (int j = 0; j < 4; ++j) {
        EXPECT_FLOAT_EQ(data[0 * 4 + j], 0.0f) << "row 0, col " << j;
    }
    // Row 1: all ones (scattered)
    for (int j = 0; j < 4; ++j) {
        EXPECT_FLOAT_EQ(data[1 * 4 + j], 1.0f) << "row 1, col " << j;
    }
    // Row 2: all zeros
    for (int j = 0; j < 4; ++j) {
        EXPECT_FLOAT_EQ(data[2 * 4 + j], 0.0f) << "row 2, col " << j;
    }
}

TEST_P(ScatterVariantsTest, SelectScatterDim1) {
    // Scatter a column at index 2 along dim 1
    auto input = zeros({3, 4}, DType::Float32, device);
    float src_data[] = {10.0f, 20.0f, 30.0f};
    auto src = from_data(src_data, {3}, device);

    auto result = select_scatter(input, src, /*dim=*/1, /*index=*/2);

    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            float expected = (j == 2) ? src_data[i] : 0.0f;
            EXPECT_FLOAT_EQ(data[i * 4 + j], expected)
                << "at (" << i << ", " << j << ")";
        }
    }
}

TEST_P(ScatterVariantsTest, SelectScatterNonMutating) {
    auto input = ones({3, 3}, DType::Float32, device);
    auto src = zeros({3}, DType::Float32, device);

    auto result = select_scatter(input, src, 0, 0);

    // Original input should be unchanged
    auto input_cpu = input.cpu();
    auto* orig = input_cpu.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(orig[i], 1.0f) << "input mutated at index " << i;
    }

    // Result row 0 should be zeros
    auto result_cpu = result.cpu();
    auto* res = result_cpu.data<float>();
    for (int j = 0; j < 3; ++j) {
        EXPECT_FLOAT_EQ(res[j], 0.0f);
    }
}

// ============================================================================
// slice_scatter tests
// ============================================================================

TEST_P(ScatterVariantsTest, SliceScatterBasic) {
    // 4x4 zeros, scatter 2x4 ones at rows [1, 3) (i.e. rows 1-2)
    auto input = zeros({4, 4}, DType::Float32, device);
    auto src = ones({2, 4}, DType::Float32, device);

    auto result = slice_scatter(input, src, /*dim=*/0, /*start=*/1, /*end=*/3);

    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<float>();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float expected = (i == 1 || i == 2) ? 1.0f : 0.0f;
            EXPECT_FLOAT_EQ(data[i * 4 + j], expected)
                << "at (" << i << ", " << j << ")";
        }
    }
}

TEST_P(ScatterVariantsTest, SliceScatterWithStep) {
    // 6-element zeros, scatter at indices 0, 2, 4 (step=2)
    auto input = zeros({6}, DType::Float32, device);
    float src_data[] = {10.0f, 20.0f, 30.0f};
    auto src = from_data(src_data, {3}, device);

    auto result = slice_scatter(input, src, /*dim=*/0, /*start=*/0, /*end=*/6, /*step=*/2);

    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 10.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 20.0f);
    EXPECT_FLOAT_EQ(data[3], 0.0f);
    EXPECT_FLOAT_EQ(data[4], 30.0f);
    EXPECT_FLOAT_EQ(data[5], 0.0f);
}

TEST_P(ScatterVariantsTest, SliceScatterNonMutating) {
    auto input = ones({4}, DType::Float32, device);
    auto src = zeros({2}, DType::Float32, device);

    auto result = slice_scatter(input, src, 0, 0, 2);

    // Original unchanged
    auto input_cpu = input.cpu();
    auto* orig = input_cpu.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(orig[i], 1.0f);
    }
}

// ============================================================================
// diagonal_scatter tests
// ============================================================================

TEST_P(ScatterVariantsTest, DiagonalScatterBasic) {
    // 3x3 zeros, scatter [1,2,3] on main diagonal
    auto input = zeros({3, 3}, DType::Float32, device);
    float src_data[] = {1.0f, 2.0f, 3.0f};
    auto src = from_data(src_data, {3}, device);

    auto result = diagonal_scatter(input, src, /*offset=*/0);

    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? src_data[i] : 0.0f;
            EXPECT_FLOAT_EQ(data[i * 3 + j], expected)
                << "at (" << i << ", " << j << ")";
        }
    }
}

TEST_P(ScatterVariantsTest, DiagonalScatterPositiveOffset) {
    // 3x3 zeros, scatter [5,6] on superdiagonal (offset=1)
    auto input = zeros({3, 3}, DType::Float32, device);
    float src_data[] = {5.0f, 6.0f};
    auto src = from_data(src_data, {2}, device);

    auto result = diagonal_scatter(input, src, /*offset=*/1);

    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<float>();
    // Superdiagonal: (0,1)=5, (1,2)=6
    EXPECT_FLOAT_EQ(data[0 * 3 + 1], 5.0f);
    EXPECT_FLOAT_EQ(data[1 * 3 + 2], 6.0f);

    // Everything else should be zero
    EXPECT_FLOAT_EQ(data[0 * 3 + 0], 0.0f);
    EXPECT_FLOAT_EQ(data[0 * 3 + 2], 0.0f);
    EXPECT_FLOAT_EQ(data[1 * 3 + 0], 0.0f);
    EXPECT_FLOAT_EQ(data[1 * 3 + 1], 0.0f);
    EXPECT_FLOAT_EQ(data[2 * 3 + 0], 0.0f);
    EXPECT_FLOAT_EQ(data[2 * 3 + 1], 0.0f);
    EXPECT_FLOAT_EQ(data[2 * 3 + 2], 0.0f);
}

TEST_P(ScatterVariantsTest, DiagonalScatterNegativeOffset) {
    // 3x3 zeros, scatter [7,8] on subdiagonal (offset=-1)
    auto input = zeros({3, 3}, DType::Float32, device);
    float src_data[] = {7.0f, 8.0f};
    auto src = from_data(src_data, {2}, device);

    auto result = diagonal_scatter(input, src, /*offset=*/-1);

    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<float>();
    // Subdiagonal: (1,0)=7, (2,1)=8
    EXPECT_FLOAT_EQ(data[1 * 3 + 0], 7.0f);
    EXPECT_FLOAT_EQ(data[2 * 3 + 1], 8.0f);
}

TEST_P(ScatterVariantsTest, DiagonalScatterNonMutating) {
    auto input = ones({3, 3}, DType::Float32, device);
    auto src = zeros({3}, DType::Float32, device);

    auto result = diagonal_scatter(input, src, 0);

    // Original unchanged
    auto input_cpu = input.cpu();
    auto* orig = input_cpu.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(orig[i], 1.0f);
    }
}

INSTANTIATE_BACKEND_TESTS(ScatterVariantsTest);
