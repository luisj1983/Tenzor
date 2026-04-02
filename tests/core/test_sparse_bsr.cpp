/**
 * @file test_sparse_bsr.cpp
 * @brief Tests for BSR (Block Sparse Row) tensor format
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"

using namespace tenzor;

class SparseBSRTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(SparseBSRTest, Construction) {
    // Create a simple BSR tensor: 4x4 matrix with 2x2 blocks
    // Block (0,0) and Block (1,1)
    auto bsr_row_ptr = tenzor::zeros({3}, DType::Int64);
    bsr_row_ptr.data<int64_t>()[0] = 0;
    bsr_row_ptr.data<int64_t>()[1] = 1;
    bsr_row_ptr.data<int64_t>()[2] = 2;

    auto bsr_col_ind = tenzor::zeros({2}, DType::Int64);
    bsr_col_ind.data<int64_t>()[0] = 0;
    bsr_col_ind.data<int64_t>()[1] = 1;

    // Values: 2 blocks of 2x2
    auto values = tenzor::zeros({2, 2, 2});
    auto* v = values.data<float>();
    v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;  // Block 0
    v[4] = 5; v[5] = 6; v[6] = 7; v[7] = 8;  // Block 1

    auto bsr = SparseTensor::sparse_bsr(bsr_row_ptr, bsr_col_ind, values,
                                         {4, 4}, {2, 2});

    EXPECT_EQ(bsr.layout(), SparseLayout::BSR);
    EXPECT_EQ(bsr.shape()[0], 4);
    EXPECT_EQ(bsr.shape()[1], 4);
    EXPECT_EQ(bsr.block_size().first, 2);
    EXPECT_EQ(bsr.block_size().second, 2);
}

TEST_F(SparseBSRTest, ToDenseConversion) {
    auto bsr_row_ptr = tenzor::zeros({3}, DType::Int64);
    bsr_row_ptr.data<int64_t>()[0] = 0;
    bsr_row_ptr.data<int64_t>()[1] = 1;
    bsr_row_ptr.data<int64_t>()[2] = 2;

    auto bsr_col_ind = tenzor::zeros({2}, DType::Int64);
    bsr_col_ind.data<int64_t>()[0] = 0;
    bsr_col_ind.data<int64_t>()[1] = 1;

    auto values = tenzor::zeros({2, 2, 2});
    auto* v = values.data<float>();
    v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;
    v[4] = 5; v[5] = 6; v[6] = 7; v[7] = 8;

    auto bsr = SparseTensor::sparse_bsr(bsr_row_ptr, bsr_col_ind, values,
                                         {4, 4}, {2, 2});

    auto dense = bsr.to_dense();
    EXPECT_EQ(dense.shape()[0], 4);
    EXPECT_EQ(dense.shape()[1], 4);

    auto* d = dense.data<float>();
    // Block (0,0): rows 0-1, cols 0-1
    EXPECT_EQ(d[0*4+0], 1); EXPECT_EQ(d[0*4+1], 2);
    EXPECT_EQ(d[1*4+0], 3); EXPECT_EQ(d[1*4+1], 4);
    // Block (1,1): rows 2-3, cols 2-3
    EXPECT_EQ(d[2*4+2], 5); EXPECT_EQ(d[2*4+3], 6);
    EXPECT_EQ(d[3*4+2], 7); EXPECT_EQ(d[3*4+3], 8);
    // Zeros elsewhere
    EXPECT_EQ(d[0*4+2], 0); EXPECT_EQ(d[0*4+3], 0);
    EXPECT_EQ(d[2*4+0], 0); EXPECT_EQ(d[2*4+1], 0);
}
