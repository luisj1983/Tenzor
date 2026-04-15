/**
 * @file test_lobpcg_multidtype.cpp
 * @brief Multi-backend tests for linalg::lobpcg (LOBPCG eigenvalue solver)
 */

#include "backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/indexing.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class LOBPCGTest : public BackendTest {};

TEST_P(LOBPCGTest, TridiagonalSmallestEigenvalues) {
    // Build a tridiagonal matrix: diag=2, off-diag=-1
    // Known eigenvalues: 2 - 2*cos(j*pi/(N+1)) for j=1..N
    // Smallest eigenvalue: 2 - 2*cos(pi/(N+1))
    int64_t N = 20;
    int64_t k = 3;

    // Build tridiagonal on CPU then move to device
    auto A_cpu = tenzor::zeros({N, N}, DType::Float32, Device::cpu());
    auto* A_data = A_cpu.data<float>();
    for (int64_t i = 0; i < N; ++i) {
        A_data[i * N + i] = 2.0f;
        if (i > 0) A_data[i * N + (i - 1)] = -1.0f;
        if (i < N - 1) A_data[i * N + (i + 1)] = -1.0f;
    }
    auto A = A_cpu.to(device);

    // Initial guess: random
    auto X0 = tenzor::randn({N, k}, DType::Float32, device);

    // Solve
    auto [evals, evecs] = linalg::lobpcg(A, X0, k, Tensor(), 200, 1e-4);

    // Move results to CPU for checking
    auto evals_cpu = evals.to(Device::cpu()).contiguous();
    auto* ev_data = evals_cpu.data<float>();

    // Compute expected eigenvalues analytically
    for (int64_t j = 0; j < k; ++j) {
        double expected = 2.0 - 2.0 * std::cos((j + 1) * M_PI / (N + 1));
        EXPECT_NEAR(ev_data[j], expected, 0.02f)
            << "Eigenvalue " << j << " mismatch on " << device.to_string()
            << ": got " << ev_data[j] << ", expected " << expected;
    }

    // Verify eigenvector orthogonality: evecs^T @ evecs should be close to I
    auto evecs_cpu = evecs.to(Device::cpu()).contiguous();
    auto VtV = tenzor::matmul(
        tenzor::transpose(evecs_cpu, 0, 1).contiguous(),
        evecs_cpu);
    auto I_k = tenzor::eye(k, std::nullopt, DType::Float32, Device::cpu());
    expectTensorNear(VtV, I_k, 0.05f);
}

TEST_P(LOBPCGTest, DiagonalMatrix) {
    // Simple diagonal matrix: eigenvalues are the diagonal entries
    int64_t N = 10;
    int64_t k = 2;

    auto diag_vals_cpu = tenzor::zeros({N}, DType::Float32, Device::cpu());
    auto* d_data = diag_vals_cpu.data<float>();
    for (int64_t i = 0; i < N; ++i) {
        d_data[i] = static_cast<float>(i + 1);  // eigenvalues: 1, 2, 3, ...
    }

    // Build diagonal matrix
    auto A_cpu = tenzor::zeros({N, N}, DType::Float32, Device::cpu());
    auto* A_data = A_cpu.data<float>();
    for (int64_t i = 0; i < N; ++i) {
        A_data[i * N + i] = d_data[i];
    }
    auto A = A_cpu.to(device);

    auto X0 = tenzor::randn({N, k}, DType::Float32, device);
    auto [evals, evecs] = linalg::lobpcg(A, X0, k, Tensor(), 200, 1e-5);

    auto evals_cpu = evals.to(Device::cpu()).contiguous();
    auto* ev_data = evals_cpu.data<float>();

    // Smallest eigenvalues should be 1.0 and 2.0
    EXPECT_NEAR(ev_data[0], 1.0f, 0.05f)
        << "Smallest eigenvalue mismatch on " << device.to_string();
    EXPECT_NEAR(ev_data[1], 2.0f, 0.05f)
        << "Second eigenvalue mismatch on " << device.to_string();
}

TEST_P(LOBPCGTest, SingleEigenvalue) {
    // Find just 1 eigenvalue of a small symmetric matrix
    int64_t N = 8;
    int64_t k = 1;

    // Build SPD matrix: A = X^T X + I
    auto X_cpu = tenzor::randn({N, N}, DType::Float32, Device::cpu());
    auto A_cpu = tenzor::matmul(
        tenzor::transpose(X_cpu, 0, 1).contiguous(), X_cpu);
    A_cpu = tenzor::add(A_cpu, eye(N, std::nullopt, DType::Float32, Device::cpu()));
    auto A = A_cpu.to(device);

    // Get ground truth via eigh
    auto [ref_evals, ref_evecs] = linalg::eigh(A_cpu);
    auto* ref_data = ref_evals.data<float>();
    float expected_min = ref_data[0];

    auto X0 = tenzor::randn({N, k}, DType::Float32, device);
    auto [evals, evecs] = linalg::lobpcg(A, X0, k, Tensor(), 200, 1e-5);

    auto evals_cpu = evals.to(Device::cpu()).contiguous();
    EXPECT_NEAR(evals_cpu.data<float>()[0], expected_min, 0.1f)
        << "Smallest eigenvalue mismatch on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(LOBPCGTest);
