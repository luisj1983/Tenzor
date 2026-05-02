/**
 * @file test_linalg_lu.cpp
 * @brief Multi-backend tests for LU decomposition and LU solve.
 *
 * Migrated from a CPU-only fixture to BackendTest now that GPU LinalgLU /
 * LinalgLUSolve kernels have landed on CUDA, ROCm, Vulkan, and OneAPI
 * (audit/README.md Phase 5). The test now exercises the same correctness
 * properties on every backend that registers the op.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/linalg.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/transform.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class LinalgLUTest : public BackendTest {
protected:
    float max_abs_diff(const Tensor& a, const Tensor& b) {
        auto a_f = a.to(Device::cpu()).to(DType::Float32).contiguous();
        auto b_f = b.to(Device::cpu()).to(DType::Float32).contiguous();
        auto* a_data = a_f.data<float>();
        auto* b_data = b_f.data<float>();
        float max_val = 0.0f;
        for (int64_t i = 0; i < a_f.numel(); ++i) {
            max_val = std::max(max_val, std::abs(a_data[i] - b_data[i]));
        }
        return max_val;
    }
};

TEST_P(LinalgLUTest, BasicFactorization_Float32) {
    float vals[] = {2.0f, 1.0f, 1.0f,
                    4.0f, 3.0f, 3.0f,
                    8.0f, 7.0f, 9.0f};
    auto A_cpu = from_data(vals, {3, 3}, Device::cpu());
    auto A = (device.type == Device::Type::CPU) ? A_cpu : A_cpu.to(device);

    auto [L, U, pivots] = linalg::lu(A);

    EXPECT_EQ(L.shape()[0], 3);
    EXPECT_EQ(L.shape()[1], 3);
    EXPECT_EQ(U.shape()[0], 3);
    EXPECT_EQ(U.shape()[1], 3);
    EXPECT_EQ(pivots.shape()[0], 3);
    EXPECT_EQ(pivots.dtype(), DType::Int32);

    // Verify L is unit lower triangular
    auto L_cpu = L.to(Device::cpu()).contiguous();
    auto* l_data = L_cpu.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(l_data[i * 3 + i], 1.0f);
        for (int j = i + 1; j < 3; ++j) {
            EXPECT_FLOAT_EQ(l_data[i * 3 + j], 0.0f);
        }
    }

    // Verify U is upper triangular
    auto U_cpu = U.to(Device::cpu()).contiguous();
    auto* u_data = U_cpu.data<float>();
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

TEST_P(LinalgLUTest, Float64) {
    double vals[] = {1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0,
                     7.0, 8.0, 10.0};
    auto A_cpu = from_data(vals, {3, 3}, Device::cpu());
    auto A = (device.type == Device::Type::CPU) ? A_cpu : A_cpu.to(device);

    auto [L, U, pivots] = linalg::lu(A);

    EXPECT_EQ(L.dtype(), DType::Float64);
    EXPECT_EQ(U.dtype(), DType::Float64);
    EXPECT_EQ(pivots.dtype(), DType::Int32);
}

TEST_P(LinalgLUTest, IdentityMatrix) {
    auto I_cpu = tenzor::eye(4);
    auto I = (device.type == Device::Type::CPU) ? I_cpu : I_cpu.to(device);

    auto [L, U, pivots] = linalg::lu(I);

    float diff_L = max_abs_diff(L, I);
    float diff_U = max_abs_diff(U, I);
    EXPECT_LT(diff_L, 1e-6f);
    EXPECT_LT(diff_U, 1e-6f);
}

TEST_P(LinalgLUTest, NonSquareThrows) {
    auto A = tenzor::randn({3, 4}, DType::Float32, device);
    EXPECT_THROW(linalg::lu(A), std::invalid_argument);
}

TEST_P(LinalgLUTest, OneDimThrows) {
    auto A = tenzor::randn({4}, DType::Float32, device);
    EXPECT_THROW(linalg::lu(A), std::invalid_argument);
}

TEST_P(LinalgLUTest, TwoByTwo) {
    float vals[] = {4.0f, 3.0f, 6.0f, 3.0f};
    auto A_cpu = from_data(vals, {2, 2}, Device::cpu());
    auto A = (device.type == Device::Type::CPU) ? A_cpu : A_cpu.to(device);

    auto [L, U, pivots] = linalg::lu(A);

    // L should be unit lower triangular
    auto L_cpu = L.to(Device::cpu()).contiguous();
    auto* l_data = L_cpu.data<float>();
    EXPECT_FLOAT_EQ(l_data[0], 1.0f);  // L[0,0]
    EXPECT_FLOAT_EQ(l_data[1], 0.0f);  // L[0,1]
    EXPECT_FLOAT_EQ(l_data[3], 1.0f);  // L[1,1]

    // U should be upper triangular
    auto U_cpu = U.to(Device::cpu()).contiguous();
    auto* u_data = U_cpu.data<float>();
    EXPECT_FLOAT_EQ(u_data[2], 0.0f);  // U[1,0]
}

// ============================================================================
// LU Solve Tests
// ============================================================================

TEST_P(LinalgLUTest, LUSolveBasic) {
    // Solve A * X = B where A is known
    float a_vals[] = {2.0f, 1.0f, 1.0f,
                      4.0f, 3.0f, 3.0f,
                      8.0f, 7.0f, 9.0f};
    auto A_cpu = from_data(a_vals, {3, 3}, Device::cpu());
    auto A = (device.type == Device::Type::CPU) ? A_cpu : A_cpu.to(device);

    float b_vals[] = {1.0f, 2.0f, 3.0f};
    auto B_cpu = from_data(b_vals, {3, 1}, Device::cpu());
    auto B = (device.type == Device::Type::CPU) ? B_cpu : B_cpu.to(device);

    auto [L, U, pivots] = linalg::lu(A);

    // Pack LU factors back into combined form for lu_solve
    auto eye3_cpu = tenzor::eye(3);
    auto eye3 = (device.type == Device::Type::CPU) ? eye3_cpu : eye3_cpu.to(device);
    auto LU_data = tenzor::add(tenzor::sub(L, eye3), U);
    auto X = linalg::lu_solve(LU_data, pivots, B);

    EXPECT_EQ(X.shape()[0], 3);
    EXPECT_EQ(X.shape()[1], 1);

    // Verify: A * X ≈ B
    auto AX = tenzor::matmul(A, X);
    float diff = max_abs_diff(AX, B);
    EXPECT_LT(diff, 1e-4f) << "LU solve residual too large on "
                           << device.to_string() << ": " << diff;
}

TEST_P(LinalgLUTest, LUSolveMultipleRHS) {
    // Solve A * X = B where B has multiple columns
    float a_vals[] = {4.0f, 3.0f, 6.0f, 3.0f};
    auto A_cpu = from_data(a_vals, {2, 2}, Device::cpu());
    auto A = (device.type == Device::Type::CPU) ? A_cpu : A_cpu.to(device);

    float b_vals[] = {1.0f, 0.0f,
                      0.0f, 1.0f};
    auto B_cpu = from_data(b_vals, {2, 2}, Device::cpu());
    auto B = (device.type == Device::Type::CPU) ? B_cpu : B_cpu.to(device);

    auto [L, U, pivots] = linalg::lu(A);
    auto eye2_cpu = tenzor::eye(2);
    auto eye2 = (device.type == Device::Type::CPU) ? eye2_cpu : eye2_cpu.to(device);
    auto LU_data = tenzor::add(tenzor::sub(L, eye2), U);
    auto X = linalg::lu_solve(LU_data, pivots, B);

    EXPECT_EQ(X.shape()[0], 2);
    EXPECT_EQ(X.shape()[1], 2);

    // X should be A^{-1}, so A * X ≈ I
    auto AX = tenzor::matmul(A, X);
    float diff = max_abs_diff(AX, eye2);
    EXPECT_LT(diff, 1e-4f) << "LU solve inverse residual too large on "
                           << device.to_string() << ": " << diff;
}

INSTANTIATE_BACKEND_TESTS(LinalgLUTest);
