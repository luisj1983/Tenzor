/**
 * @file test_gradcheck_missing.cpp
 * @brief Gradient checks for autograd functions that previously had no
 *        gradient verification tests.
 *
 * Each test creates appropriately conditioned inputs and uses gradcheck()
 * to verify numerical vs analytical gradients match. All tests use Float64
 * for numerical stability.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradCheckMissingTest : public BackendTest {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        BackendTest::SetUp();
        set_grad_enabled(true);
    }

    // Helper: create an SPD matrix A = X^T X + eps*I on the current device
    Tensor make_spd(int64_t n, double eps = 0.1) {
        auto x = randn({n, n}, DType::Float64, Device::cpu());
        auto xt = tenzor::transpose(x, 0, 1);
        auto ata = tenzor::matmul(xt, x);
        auto eye_t = tenzor::eye(n, std::nullopt, DType::Float64, Device::cpu());
        auto result = tenzor::add(ata, tenzor::mul(eye_t, eps));
        return result.to(device);
    }
};

// ============================================================================
// HIGH RISK — numerically sensitive
// ============================================================================

TEST_P(GradCheckMissingTest, CholeskyInverse) {
    auto spd = make_spd(4);
    Variable x(spd, true);

    auto f = [](const Variable& v) -> Variable {
        return cholesky_inverse(v);
    };

    bool passed = gradcheck(f, x, 1e-5, 1e-3, 1e-3);
    EXPECT_TRUE(passed) << "cholesky_inverse gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Median) {
    // Use distinct values to avoid ties at the median boundary
    auto data = tenzor::arange(0, 15, 1.0, DType::Float64, Device::cpu()).to(device);
    data = tenzor::reshape(data, {3, 5});
    // Add small noise to break ties
    data = tenzor::add(data, tenzor::mul(randn({3, 5}, DType::Float64, device), 0.01));
    Variable x(data, true);

    auto f = [](const Variable& v) -> Variable {
        return median(v, /*dim=*/1, /*keepdim=*/true);
    };

    bool passed = gradcheck(f, x, 1e-5, 1e-3, 1e-3);
    EXPECT_TRUE(passed) << "median gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, TensorInv) {
    // tensorinv with ind=2: input shape (a, b, a*b) reshaped to (a*b, a*b) and inverted
    // Use a well-conditioned 2x3x6 tensor
    auto n = 2;
    auto m = 3;
    auto nm = n * m;  // 6
    // Start from an invertible matrix and reshape
    auto mat = randn({nm, nm}, DType::Float64, Device::cpu());
    // Add diagonal to make well-conditioned
    auto eye_t = tenzor::eye(nm, std::nullopt, DType::Float64, Device::cpu());
    mat = tenzor::add(mat, tenzor::mul(eye_t, 2.0));
    auto t = tenzor::reshape(mat, {n, m, nm}).to(device);
    Variable x(t, true);

    auto f = [](const Variable& v) -> Variable {
        return tensorinv(v, /*ind=*/2);
    };

    bool passed = gradcheck(f, x, 1e-5, 1e-3, 1e-3);
    EXPECT_TRUE(passed) << "tensorinv gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, TensorSolve) {
    // tensorsolve(A, B): solve for X such that A @ X = B
    // A: (n, n, n) resolvable to (n^2, n) system; B: (n, n)
    // Simpler: use (n, n) A and (n,) B
    int64_t n = 4;
    auto a_data = randn({n, n}, DType::Float64, Device::cpu());
    auto eye_t = tenzor::eye(n, std::nullopt, DType::Float64, Device::cpu());
    a_data = tenzor::add(a_data, tenzor::mul(eye_t, 3.0));  // well-conditioned
    auto b_data = randn({n}, DType::Float64, Device::cpu());

    Variable a(a_data.to(device), true);
    Variable b(b_data.to(device), true);

    // gradcheck for A input
    auto f_a = [&b](const Variable& v) -> Variable {
        return tensorsolve(v, b);
    };
    bool passed_a = gradcheck(f_a, a, 1e-5, 1e-3, 1e-3);
    EXPECT_TRUE(passed_a) << "tensorsolve gradcheck (wrt A) failed on " << device.to_string();
}

// ============================================================================
// MEDIUM RISK
// ============================================================================

TEST_P(GradCheckMissingTest, Entr) {
    // entr(x) = -x*log(x), domain: x > 0
    auto data = tenzor::add(
        tenzor::abs(randn({3, 4}, DType::Float64, device)),
        0.1  // avoid zero
    );
    Variable x(data, true);

    auto f = [](const Variable& v) -> Variable {
        return entr(v);
    };

    bool passed = gradcheck(f, x, 1e-5, 1e-4, 1e-4);
    EXPECT_TRUE(passed) << "entr gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, HouseholderProduct) {
    // Construct valid Householder reflectors manually.
    // input: (m, n) lower-triangular packed reflectors, tau: (n,) scalar factors.
    // Use small dimensions for numerical stability.
    int64_t m = 4, n = 3;
    auto input_data = randn({m, n}, DType::Float64, Device::cpu()).to(device);
    auto tau_data = randn({n}, DType::Float64, Device::cpu()).to(device);

    Variable input_var(input_data, true);
    Variable tau_var(tau_data.clone(), false);  // gradcheck wrt input only

    auto f = [&tau_var](const Variable& v) -> Variable {
        return householder_product(v, tau_var);
    };

    bool passed = gradcheck(f, input_var, 1e-5, 1e-3, 1e-3);
    EXPECT_TRUE(passed) << "householder_product gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, LinalgNorm) {
    // Frobenius norm of a matrix
    auto data = randn({4, 4}, DType::Float64, device);
    // Ensure non-zero to avoid grad issues at zero
    data = tenzor::add(data, 0.1);
    Variable x(data, true);

    auto f = [](const Variable& v) -> Variable {
        return linalg_norm(v, "fro");
    };

    bool passed = gradcheck(f, x, 1e-5, 1e-4, 1e-4);
    EXPECT_TRUE(passed) << "linalg_norm gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Renorm) {
    // renorm(input, p, dim, maxnorm): renormalize slices
    auto data = randn({3, 5}, DType::Float64, device);
    // Scale up so renorm actually clips some slices
    data = tenzor::mul(data, 10.0);
    Variable x(data, true);

    auto f = [](const Variable& v) -> Variable {
        return renorm(v, 2.0, /*dim=*/0, /*maxnorm=*/1.0);
    };

    bool passed = gradcheck(f, x, 1e-5, 1e-3, 1e-3);
    EXPECT_TRUE(passed) << "renorm gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Xlogy) {
    // xlogy(x, y) = x * log(y), with 0*log(0) = 0
    // Domain: x >= 0, y > 0 for smooth gradients
    auto x_data = tenzor::add(tenzor::abs(randn({3, 4}, DType::Float64, device)), 0.1);
    auto y_data = tenzor::add(tenzor::abs(randn({3, 4}, DType::Float64, device)), 0.1);
    Variable x(x_data, true);
    Variable y(y_data, true);

    // gradcheck wrt x
    auto f_x = [&y](const Variable& v) -> Variable {
        return xlogy(v, y);
    };
    bool passed_x = gradcheck(f_x, x, 1e-5, 1e-4, 1e-4);
    EXPECT_TRUE(passed_x) << "xlogy gradcheck (wrt x) failed on " << device.to_string();

    // gradcheck wrt y
    auto f_y = [&x](const Variable& v) -> Variable {
        return xlogy(x, v);
    };
    bool passed_y = gradcheck(f_y, y, 1e-5, 1e-4, 1e-4);
    EXPECT_TRUE(passed_y) << "xlogy gradcheck (wrt y) failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(GradCheckMissingTest);
