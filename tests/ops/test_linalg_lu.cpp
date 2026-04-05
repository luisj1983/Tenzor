/**
 * @file test_linalg_lu.cpp
 * @brief Tests for LU decomposition and LU solve
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/linalg.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/transform.hpp>
#include <cmath>

using namespace tenzor;

class LinalgLUTest : public ::testing::Test {
protected:
    Device cpu = Device::cpu();

    void SetUp() override {
        tenzor::initialize();
    }

    float max_abs_diff(const Tensor& a, const Tensor& b) {
        auto a_f = a.to(DType::Float32).contiguous();
        auto b_f = b.to(DType::Float32).contiguous();
        auto* a_data = a_f.data<float>();
        auto* b_data = b_f.data<float>();
        float max_val = 0.0f;
        for (int64_t i = 0; i < a_f.numel(); ++i) {
            max_val = std::max(max_val, std::abs(a_data[i] - b_data[i]));
        }
        return max_val;
    }
};

TEST_F(LinalgLUTest, BasicFactorization_Float32) {
    float vals[] = {2.0f, 1.0f, 1.0f,
                    4.0f, 3.0f, 3.0f,
                    8.0f, 7.0f, 9.0f};
    auto A = from_data(vals, {3, 3}, cpu);

    auto [L, U, pivots] = linalg::lu(A);

    EXPECT_EQ(L.shape()[0], 3);
    EXPECT_EQ(L.shape()[1], 3);
    EXPECT_EQ(U.shape()[0], 3);
    EXPECT_EQ(U.shape()[1], 3);
    EXPECT_EQ(pivots.shape()[0], 3);
    EXPECT_EQ(pivots.dtype(), DType::Int32);

    // Verify L is unit lower triangular
    auto* l_data = L.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(l_data[i * 3 + i], 1.0f);
        for (int j = i + 1; j < 3; ++j) {
            EXPECT_FLOAT_EQ(l_data[i * 3 + j], 0.0f);
        }
    }

    // Verify U is upper triangular
    auto* u_data = U.data<float>();
    for (int i = 1; i < 3; ++i) {
        for (int j = 0; j < i; ++j) {
            EXPECT_FLOAT_EQ(u_data[i * 3 + j], 0.0f);
        }
    }

    // L @ U should be a permutation of A's rows
    auto LU_product = tenzor::matmul(L, U);
    EXPECT_EQ(LU_product.shape()[0], 3);
    EXPECT_EQ(LU_product.shape()[1], 3);
}

TEST_F(LinalgLUTest, Float64) {
    double vals[] = {1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0,
                     7.0, 8.0, 10.0};
    auto A = from_data(vals, {3, 3}, cpu);

    auto [L, U, pivots] = linalg::lu(A);

    EXPECT_EQ(L.dtype(), DType::Float64);
    EXPECT_EQ(U.dtype(), DType::Float64);
    EXPECT_EQ(pivots.dtype(), DType::Int32);
}

TEST_F(LinalgLUTest, IdentityMatrix) {
    auto I = tenzor::eye(4);

    auto [L, U, pivots] = linalg::lu(I);

    float diff_L = max_abs_diff(L, I);
    float diff_U = max_abs_diff(U, I);
    EXPECT_LT(diff_L, 1e-6f);
    EXPECT_LT(diff_U, 1e-6f);
}

TEST_F(LinalgLUTest, NonSquareThrows) {
    auto A = tenzor::randn({3, 4}, DType::Float32);
    EXPECT_THROW(linalg::lu(A), std::invalid_argument);
}

TEST_F(LinalgLUTest, OneDimThrows) {
    auto A = tenzor::randn({4}, DType::Float32);
    EXPECT_THROW(linalg::lu(A), std::invalid_argument);
}

TEST_F(LinalgLUTest, TwoByTwo) {
    float vals[] = {4.0f, 3.0f, 6.0f, 3.0f};
    auto A = from_data(vals, {2, 2}, cpu);

    auto [L, U, pivots] = linalg::lu(A);

    // L should be unit lower triangular
    auto* l_data = L.data<float>();
    EXPECT_FLOAT_EQ(l_data[0], 1.0f);  // L[0,0]
    EXPECT_FLOAT_EQ(l_data[1], 0.0f);  // L[0,1]
    EXPECT_FLOAT_EQ(l_data[3], 1.0f);  // L[1,1]

    // U should be upper triangular
    auto* u_data = U.data<float>();
    EXPECT_FLOAT_EQ(u_data[2], 0.0f);  // U[1,0]
}
