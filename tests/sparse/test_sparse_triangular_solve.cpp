/**
 * @file test_sparse_triangular_solve.cpp
 * @brief Multi-backend forward parity for sparse_triangular_solve
 *        (audit-2026-05-03 N4). The autograd-level gradcheck for this op
 *        is already in tests/autograd/test_gradcheck_missing.cpp (CPU-only);
 *        this file enforces backend coverage on the forward path.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/sparse/sparse_ops.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class SparseTriangularSolveTest : public BackendTest {};

namespace {

// Build a small lower-triangular dense matrix on CPU, then convert to a
// COO SparseTensor on the requested device. Returns the dense version too
// for cross-checking.
auto build_lower_tri(int64_t n, DType dt, Device dev)
    -> std::pair<SparseTensor, Tensor>
{
    auto eye_t = eye(n, std::nullopt, dt, Device::cpu());
    // Add a small lower-triangular perturbation to make the system non-trivial
    // but well-conditioned (diag still 1.0).
    auto noise = mul(tril(randn({n, n}, dt, Device::cpu()), -1), 0.1);
    auto L_dense_cpu = eye_t + noise;
    auto L_dense = L_dense_cpu.to(dev);
    auto L_sparse = SparseTensor::from_dense(L_dense, SparseLayout::COO);
    return {L_sparse, L_dense};
}

} // anonymous

TEST_P(SparseTriangularSolveTest, LowerTriangularSolve) {
    // BackendTest::SetUp() already handles backend availability + env vars.
    int64_t n = 4;
    auto [L_sparse, L_dense] = build_lower_tri(n, DType::Float64, device);
    auto b = randn({n}, DType::Float64, device);

    auto x = sparse::sparse_triangular_solve(L_sparse, b, /*upper=*/false);

    // Verify: L @ x ≈ b
    auto reconstructed = matmul(L_dense, x.unsqueeze(-1)).squeeze(-1);
    auto diff = (reconstructed - b).to(Device::cpu()).contiguous();
    auto* p = diff.data<double>();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(p[i], 0.0, 1e-6) << "row " << i;
    }
}

TEST_P(SparseTriangularSolveTest, BatchedRHS) {
    // BackendTest::SetUp() already handles backend availability + env vars.
    int64_t n = 4, k = 2;
    auto [L_sparse, L_dense] = build_lower_tri(n, DType::Float64, device);
    auto B = randn({n, k}, DType::Float64, device);

    auto X = sparse::sparse_triangular_solve(L_sparse, B, /*upper=*/false);
    EXPECT_EQ(X.shape().size(), 2u);
    EXPECT_EQ(X.shape()[0], n);
    EXPECT_EQ(X.shape()[1], k);

    auto reconstructed = matmul(L_dense, X);
    auto diff = (reconstructed - B).to(Device::cpu()).contiguous();
    auto* p = diff.data<double>();
    for (int64_t i = 0; i < n * k; ++i) {
        EXPECT_NEAR(p[i], 0.0, 1e-6);
    }
}

INSTANTIATE_BACKEND_TESTS(SparseTriangularSolveTest);
