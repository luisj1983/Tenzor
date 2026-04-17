/**
 * @file test_gradcheck_missing_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for previously missing gradient checks
 *
 * Converted from test_gradcheck_missing.cpp. Uses Float64 where possible for
 * numerical stability, skips Float16 entirely since gradcheck relies on
 * finite-difference numerical gradient computation.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradCheckMissingMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        set_grad_enabled(true);
    }

    // Helper: create an SPD matrix A = X^T X + eps*I on the current device/dtype
    Tensor make_spd(int64_t n, double eps = 0.1) {
        // Build in Float32 on CPU, then convert
        auto x = randn({n, n}, DType::Float32, Device::cpu());
        auto xt = tenzor::transpose(x, 0, 1);
        auto ata = tenzor::matmul(xt, x);
        auto eye_t = tenzor::eye(n, std::nullopt, DType::Float32, Device::cpu());
        auto result = tenzor::add(ata, tenzor::mul(eye_t, static_cast<float>(eps)));
        if (dtype() != DType::Float32) result = result.to(dtype());
        return result.to(device());
    }

    // Gradcheck tolerances appropriate for the dtype
    double gc_eps() const {
        return (dtype() == DType::Float64) ? 1e-5 : 1e-4;
    }
    double gc_atol() const {
        return (dtype() == DType::Float64) ? 1e-3 : 1e-2;
    }
    double gc_rtol() const {
        return (dtype() == DType::Float64) ? 1e-3 : 1e-2;
    }
};

// ============================================================================
// HIGH RISK -- numerically sensitive
// ============================================================================

TEST_P(GradCheckMissingMultiDTypeTest, CholeskyInverse) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";

    auto spd = make_spd(4);
    Variable x(spd, true);

    auto f = [](const Variable& v) -> Variable {
        return cholesky_inverse(v);
    };

    bool passed = gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol());
    EXPECT_TRUE(passed) << "cholesky_inverse gradcheck failed on "
                        << device().to_string() << " with dtype " << static_cast<int>(dtype());
}

TEST_P(GradCheckMissingMultiDTypeTest, Median) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";

    // Create in Float32 on CPU, convert to target dtype/device
    auto data = tenzor::arange(0, 15, 1.0, DType::Float32, Device::cpu());
    data = tenzor::reshape(data, {3, 5});
    data = tenzor::add(data, tenzor::mul(randn({3, 5}, DType::Float32, Device::cpu()), 0.01f));
    if (dtype() != DType::Float32) data = data.to(dtype());
    data = data.to(device());
    Variable x(data, true);

    auto f = [](const Variable& v) -> Variable {
        return median(v, /*dim=*/1, /*keepdim=*/true);
    };

    bool passed = gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol());
    EXPECT_TRUE(passed) << "median gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMissingMultiDTypeTest, TensorInv) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";

    auto n = 2;
    auto m = 3;
    auto nm = n * m;
    auto mat = randn({nm, nm}, DType::Float32, Device::cpu());
    auto eye_t = tenzor::eye(nm, std::nullopt, DType::Float32, Device::cpu());
    mat = tenzor::add(mat, tenzor::mul(eye_t, 2.0f));
    auto t = tenzor::reshape(mat, {n, m, nm});
    if (dtype() != DType::Float32) t = t.to(dtype());
    t = t.to(device());
    Variable x(t, true);

    auto f = [](const Variable& v) -> Variable {
        return tensorinv(v, /*ind=*/2);
    };

    bool passed = gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol());
    EXPECT_TRUE(passed) << "tensorinv gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMissingMultiDTypeTest, TensorSolve) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";

    int64_t n = 4;
    auto a_data = randn({n, n}, DType::Float32, Device::cpu());
    auto eye_t = tenzor::eye(n, std::nullopt, DType::Float32, Device::cpu());
    a_data = tenzor::add(a_data, tenzor::mul(eye_t, 3.0f));
    auto b_data = randn({n}, DType::Float32, Device::cpu());

    if (dtype() != DType::Float32) {
        a_data = a_data.to(dtype());
        b_data = b_data.to(dtype());
    }

    Variable a(a_data.to(device()), true);
    Variable b(b_data.to(device()), true);

    auto f_a = [&b](const Variable& v) -> Variable {
        return tensorsolve(v, b);
    };
    bool passed_a = gradcheck(f_a, a, gc_eps(), gc_atol(), gc_rtol());
    EXPECT_TRUE(passed_a) << "tensorsolve gradcheck (wrt A) failed on " << device().to_string();
}

// ============================================================================
// MEDIUM RISK
// ============================================================================

TEST_P(GradCheckMissingMultiDTypeTest, Entr) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";

    auto data = tenzor::add(
        tenzor::abs(randn({3, 4}, DType::Float32, Device::cpu())),
        0.1f
    );
    if (dtype() != DType::Float32) data = data.to(dtype());
    data = data.to(device());
    Variable x(data, true);

    auto f = [](const Variable& v) -> Variable {
        return entr(v);
    };

    bool passed = gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol());
    EXPECT_TRUE(passed) << "entr gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMissingMultiDTypeTest, HouseholderProduct) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";

    int64_t m = 4, n = 3;
    auto input_data = randn({m, n}, DType::Float32, Device::cpu());
    auto tau_data = randn({n}, DType::Float32, Device::cpu());

    if (dtype() != DType::Float32) {
        input_data = input_data.to(dtype());
        tau_data = tau_data.to(dtype());
    }

    Variable input_var(input_data.to(device()), true);
    Variable tau_var(tau_data.to(device()).clone(), false);

    auto f = [&tau_var](const Variable& v) -> Variable {
        return householder_product(v, tau_var);
    };

    bool passed = gradcheck(f, input_var, gc_eps(), gc_atol(), gc_rtol());
    EXPECT_TRUE(passed) << "householder_product gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMissingMultiDTypeTest, LinalgNorm) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";

    auto data = randn({4, 4}, DType::Float32, Device::cpu());
    data = tenzor::add(data, 0.1f);
    if (dtype() != DType::Float32) data = data.to(dtype());
    data = data.to(device());
    Variable x(data, true);

    auto f = [](const Variable& v) -> Variable {
        return linalg_norm(v, "fro");
    };

    bool passed = gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol());
    EXPECT_TRUE(passed) << "linalg_norm gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMissingMultiDTypeTest, Renorm) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";

    auto data = randn({3, 5}, DType::Float32, Device::cpu());
    data = tenzor::mul(data, 10.0f);
    if (dtype() != DType::Float32) data = data.to(dtype());
    data = data.to(device());
    Variable x(data, true);

    auto f = [](const Variable& v) -> Variable {
        return renorm(v, 2.0, /*dim=*/0, /*maxnorm=*/1.0);
    };

    bool passed = gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol());
    EXPECT_TRUE(passed) << "renorm gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMissingMultiDTypeTest, Xlogy) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";

    auto x_data = tenzor::add(tenzor::abs(randn({3, 4}, DType::Float32, Device::cpu())), 0.1f);
    auto y_data = tenzor::add(tenzor::abs(randn({3, 4}, DType::Float32, Device::cpu())), 0.1f);
    if (dtype() != DType::Float32) {
        x_data = x_data.to(dtype());
        y_data = y_data.to(dtype());
    }
    x_data = x_data.to(device());
    y_data = y_data.to(device());
    Variable x(x_data, true);
    Variable y(y_data, true);

    // gradcheck wrt x
    auto f_x = [&y](const Variable& v) -> Variable {
        return xlogy(v, y);
    };
    bool passed_x = gradcheck(f_x, x, gc_eps(), gc_atol(), gc_rtol());
    EXPECT_TRUE(passed_x) << "xlogy gradcheck (wrt x) failed on " << device().to_string();

    // gradcheck wrt y
    auto f_y = [&x](const Variable& v) -> Variable {
        return xlogy(x, v);
    };
    bool passed_y = gradcheck(f_y, y, gc_eps(), gc_atol(), gc_rtol());
    EXPECT_TRUE(passed_y) << "xlogy gradcheck (wrt y) failed on " << device().to_string();
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GradCheckMissingMultiDTypeTest);
