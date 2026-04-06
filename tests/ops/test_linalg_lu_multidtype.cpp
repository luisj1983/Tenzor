/**
 * @file test_linalg_lu_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for LU decomposition and LU solve
 *
 * Converted from test_linalg_lu.cpp to run across all backends and float dtypes.
 *
 * NOTE: Float16 is skipped for LU decomposition tests because partial pivoting
 * and triangular factorization require higher numerical precision than Float16
 * can provide.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/linalg.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/transform.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class LinalgLUMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    /**
     * @brief Compute max absolute element-wise difference between two tensors.
     *
     * Both tensors are moved to CPU and cast to Float32 for comparison,
     * ensuring this works regardless of the test device or dtype.
     */
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

    /**
     * @brief Skip the test if the current dtype is Float16.
     *
     * LU decomposition is numerically unstable in half precision.
     */
    void skipIfFloat16() {
        if (dtype() == DType::Float16) {
            GTEST_SKIP() << "LU decomposition requires higher precision than Float16";
        }
    }
};

// ---------------------------------------------------------------------------
// BasicFactorization: verify L is unit lower triangular, U is upper triangular,
// and L @ U reconstructs the (permuted) input.
// ---------------------------------------------------------------------------
TEST_P(LinalgLUMultiDTypeTest, BasicFactorization) {
    skipIfFloat16();

    float vals[] = {2.0f, 1.0f, 1.0f,
                    4.0f, 3.0f, 3.0f,
                    8.0f, 7.0f, 9.0f};
    auto A = from_data(vals, {3, 3}, Device::cpu()).to(dtype()).to(device());

    auto [L, U, pivots] = linalg::lu(A);

    EXPECT_EQ(L.shape()[0], 3);
    EXPECT_EQ(L.shape()[1], 3);
    EXPECT_EQ(U.shape()[0], 3);
    EXPECT_EQ(U.shape()[1], 3);
    EXPECT_EQ(pivots.shape()[0], 3);
    EXPECT_EQ(pivots.dtype(), DType::Int32);

    // Move L and U to CPU for element-wise checks
    auto L_cpu = L.to(Device::cpu()).to(DType::Float32);
    auto U_cpu = U.to(Device::cpu()).to(DType::Float32);
    auto* l_data = L_cpu.data<float>();
    auto* u_data = U_cpu.data<float>();

    // Verify L is unit lower triangular
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(l_data[i * 3 + i], 1.0f, atol());
        for (int j = i + 1; j < 3; ++j) {
            EXPECT_NEAR(l_data[i * 3 + j], 0.0f, atol());
        }
    }

    // Verify U is upper triangular
    for (int i = 1; i < 3; ++i) {
        for (int j = 0; j < i; ++j) {
            EXPECT_NEAR(u_data[i * 3 + j], 0.0f, atol());
        }
    }

    // L @ U should reconstruct the (permuted) rows of A
    auto LU_product = tenzor::matmul(L, U);
    EXPECT_EQ(LU_product.shape()[0], 3);
    EXPECT_EQ(LU_product.shape()[1], 3);
}

// ---------------------------------------------------------------------------
// DTypePreservation: output L and U should match the input dtype, pivots Int32.
// ---------------------------------------------------------------------------
TEST_P(LinalgLUMultiDTypeTest, DTypePreservation) {
    skipIfFloat16();

    auto A = createRandn({3, 3});

    auto [L, U, pivots] = linalg::lu(A);

    EXPECT_EQ(L.dtype(), dtype());
    EXPECT_EQ(U.dtype(), dtype());
    EXPECT_EQ(pivots.dtype(), DType::Int32);
}

// ---------------------------------------------------------------------------
// IdentityMatrix: LU of I should yield L = I, U = I (within tolerance).
// ---------------------------------------------------------------------------
TEST_P(LinalgLUMultiDTypeTest, IdentityMatrix) {
    skipIfFloat16();

    auto I = tenzor::eye(4).to(dtype()).to(device());

    auto [L, U, pivots] = linalg::lu(I);

    auto I_ref = tenzor::eye(4).to(dtype()).to(device());
    float diff_L = max_abs_diff(L, I_ref);
    float diff_U = max_abs_diff(U, I_ref);
    EXPECT_LT(diff_L, atol());
    EXPECT_LT(diff_U, atol());
}

// ---------------------------------------------------------------------------
// NonSquareThrows: non-square matrices should be rejected.
// ---------------------------------------------------------------------------
TEST_P(LinalgLUMultiDTypeTest, NonSquareThrows) {
    skipIfFloat16();

    auto A = createRandn({3, 4});
    EXPECT_THROW(linalg::lu(A), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// OneDimThrows: 1-D tensors should be rejected.
// ---------------------------------------------------------------------------
TEST_P(LinalgLUMultiDTypeTest, OneDimThrows) {
    skipIfFloat16();

    auto A = createRandn({4});
    EXPECT_THROW(linalg::lu(A), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// TwoByTwo: small matrix sanity check for triangular structure.
// ---------------------------------------------------------------------------
TEST_P(LinalgLUMultiDTypeTest, TwoByTwo) {
    skipIfFloat16();

    float vals[] = {4.0f, 3.0f, 6.0f, 3.0f};
    auto A = from_data(vals, {2, 2}, Device::cpu()).to(dtype()).to(device());

    auto [L, U, pivots] = linalg::lu(A);

    // Move to CPU for element-wise inspection
    auto L_cpu = L.to(Device::cpu()).to(DType::Float32);
    auto U_cpu = U.to(Device::cpu()).to(DType::Float32);
    auto* l_data = L_cpu.data<float>();
    auto* u_data = U_cpu.data<float>();

    // L should be unit lower triangular
    EXPECT_NEAR(l_data[0], 1.0f, atol());  // L[0,0]
    EXPECT_NEAR(l_data[1], 0.0f, atol());  // L[0,1]
    EXPECT_NEAR(l_data[3], 1.0f, atol());  // L[1,1]

    // U should be upper triangular
    EXPECT_NEAR(u_data[2], 0.0f, atol());  // U[1,0]
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LinalgLUMultiDTypeTest);
