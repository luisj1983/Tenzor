/**
 * @file test_cholesky_solve_multidtype.cpp
 * @brief Multi-backend tests for linalg::cholesky_solve
 */

#include "backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <gtest/gtest.h>

using namespace tenzor;
using namespace tenzor::testing;

class CholeskySolveTest : public BackendTest {};

TEST_P(CholeskySolveTest, SolvesCorrectly) {
    // Create a positive-definite matrix A = X^T @ X + I
    int64_t N = 4;
    auto X = randn({N, N}, DType::Float32, device);
    auto A = tenzor::matmul(tenzor::transpose(X, 0, 1), X);
    A = tenzor::add(A, eye(N, std::nullopt, DType::Float32, device));

    // Create RHS
    auto B = randn({N, 2}, DType::Float32, device);

    // Cholesky factorize
    auto L = linalg::cholesky(A, false);

    // Solve via cholesky_solve
    auto X_sol = linalg::cholesky_solve(B, L, false);

    // Verify: A @ X_sol should equal B
    auto AX = tenzor::matmul(A, X_sol);

    expectTensorNear(AX, B, 1e-2f);
}

TEST_P(CholeskySolveTest, UpperTriangular) {
    int64_t N = 3;
    auto X = randn({N, N}, DType::Float32, device);
    auto A = tenzor::matmul(tenzor::transpose(X, 0, 1), X);
    A = tenzor::add(A, eye(N, std::nullopt, DType::Float32, device));

    auto B = randn({N, 1}, DType::Float32, device);

    // Upper Cholesky
    auto U = linalg::cholesky(A, true);
    auto X_sol = linalg::cholesky_solve(B, U, true);

    auto AX = tenzor::matmul(A, X_sol);
    expectTensorNear(AX, B, 1e-2f);
}

TEST_P(CholeskySolveTest, VectorRHS) {
    int64_t N = 5;
    auto X = randn({N, N}, DType::Float32, device);
    auto A = tenzor::matmul(tenzor::transpose(X, 0, 1), X);
    A = tenzor::add(A, eye(N, std::nullopt, DType::Float32, device));

    auto b = randn({N, 1}, DType::Float32, device);

    auto L = linalg::cholesky(A, false);
    auto x = linalg::cholesky_solve(b, L, false);

    auto Ax = tenzor::matmul(A, x);
    expectTensorNear(Ax, b, 1e-3f);
}

INSTANTIATE_BACKEND_TESTS(CholeskySolveTest);
