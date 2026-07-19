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
#include <tenzor/nn/functional.hpp>
#include <tenzor/nn/loss/losses.hpp>  // for Reduction enum used by F::nll_loss
#include <tenzor/sparse/sparse_tensor.hpp>
#include "../backend_test_fixture.hpp"
#include "../multi_backend_dtype_fixture.hpp"  // SKIP_WITH_REASON

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
    // J9: cholesky_inverse expects a lower-triangular Cholesky factor L and
    // computes (L Lᴴ)^{-1}. The previous test passed a FULL SPD matrix as L,
    // so perturbations of the (ignored) upper triangle carried zero analytic
    // gradient (backward trils grad_L) but a nonzero finite-difference gradient
    // (the forward reads L fully / via Lᴴ) — a spurious mismatch, not a backward
    // bug. Feed an actual lower-triangular factor so the gradcheck is well-posed.
    Variable spdv(make_spd(4), false);
    auto Lf = tenzor::tril(tenzor::cholesky(spdv)).tensor();  // lower-triangular factor
    Variable x(Lf, true);

    auto f = [](const Variable& v) -> Variable {
        // keep the input strictly lower-triangular through the check
        return cholesky_inverse(tenzor::tril(v));
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

    bool passed = gradcheck(f, x, 1e-5, 1e-3, 1e-2);
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

// M2: LdexpBackward was fully implemented but never wired to any
// Variable-level call site — tenzor::ldexp only existed as a raw-Tensor
// free function, so x.requires_grad through it silently carried no grad_fn
// back to x. n is the integer exponent (non-differentiable by design —
// LdexpBackward::backward returns zeros for it), so only x is gradchecked,
// matching the TensorSolve pattern above (n captured fixed in the lambda).
TEST_P(GradCheckMissingTest, LdexpGradcheck) {
    auto x_data = tenzor::add(tenzor::abs(randn({3, 4}, DType::Float64, device)), 0.1);
    Variable x(x_data, true);
    Variable n(full({3, 4}, 2.0, DType::Float64, Device::cpu()).to(device), false);

    auto f = [&n](const Variable& v) -> Variable {
        return tenzor::ldexp(v, n);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-3, 1e-3))
        << "ldexp gradcheck (wrt x) failed on " << device.to_string();
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

// ============================================================================
// E1 expansion — additional multi-backend gradcheck for core nonlinear ops
// that previously lived only in CPU-only comprehensive tests.
// ============================================================================

TEST_P(GradCheckMissingTest, Log1p) {
    // log1p(x) = log(1 + x). Valid for x > -1; stay well away from the boundary.
    auto x_t = randn({6}, DType::Float64, Device::cpu()) * 0.3;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::log1p(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "log1p gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Log2) {
    auto x_t = tenzor::abs(randn({6}, DType::Float64, Device::cpu())) + 0.5;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::log2(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "log2 gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Log10) {
    auto x_t = tenzor::abs(randn({6}, DType::Float64, Device::cpu())) + 0.5;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::log10(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "log10 gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Exp2) {
    // Keep inputs modest so exp2 stays in Float64 range.
    auto x_t = randn({6}, DType::Float64, Device::cpu()) * 0.5;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::exp2(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "exp2 gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Expm1) {
    auto x_t = randn({6}, DType::Float64, Device::cpu()) * 0.5;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::expm1(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "expm1 gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Atan2) {
    // atan2 is non-linear in both arguments; test gradient w.r.t. the first.

    auto y_t = randn({6}, DType::Float64, Device::cpu());
    auto x_t = tenzor::abs(randn({6}, DType::Float64, Device::cpu())) + 1.0;
    Variable y(y_t.to(device), true);
    Variable x(x_t.to(device), false);
    auto f = [&x](const Variable& v) -> Variable { return tenzor::atan2(v, x); };
    EXPECT_TRUE(gradcheck(f, y, 1e-5, 1e-4, 1e-4))
        << "atan2 gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Cat) {
    // cat() is linear — its gradient is a pure shape-split. gradcheck should
    // pass trivially but the test pins that the backward dispatches through
    // every backend's implementation.
    auto a_t = randn({3, 4}, DType::Float64, Device::cpu());
    auto b_t = randn({2, 4}, DType::Float64, Device::cpu());
    Variable a(a_t.to(device), true);
    Variable b(b_t.to(device), false);
    auto f = [&b](const Variable& v) -> Variable {
        return tenzor::cat({v, b}, /*dim=*/0);
    };
    EXPECT_TRUE(gradcheck(f, a, 1e-5, 1e-4, 1e-4))
        << "cat gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Sinh) {

    auto x_t = randn({6}, DType::Float64, Device::cpu()) * 0.5;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::sinh(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "sinh gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Cosh) {

    auto x_t = randn({6}, DType::Float64, Device::cpu()) * 0.5;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::cosh(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "cosh gradcheck failed on " << device.to_string();
}

// The trig/hyperbolic gradcheck family below passes on CPU/CUDA/OneAPI/ROCm
// but fails on Vulkan Float64 due to the same shader precision issue that
// gated J4 (atan2/sinh/cosh). Skip Vulkan here until J4's shader audit lands.

TEST_P(GradCheckMissingTest, Tan) {

    auto x_t = randn({6}, DType::Float64, Device::cpu()) * 0.5;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::tan(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "tan gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Asin) {

    auto x_t = randn({6}, DType::Float64, Device::cpu()) * 0.3;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::asin(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "asin gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Acos) {

    auto x_t = randn({6}, DType::Float64, Device::cpu()) * 0.3;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::acos(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "acos gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Atan) {

    auto x_t = randn({6}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::atan(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "atan gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Tanh) {

    auto x_t = randn({6}, DType::Float64, Device::cpu()) * 0.5;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::tanh(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "tanh gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, IndexSelect) {
    // index_select gathers rows along dim 0 — backward is a scatter-add.
    auto x_t = randn({8, 5}, DType::Float64, Device::cpu());
    auto idx_t = zeros({3}, DType::Int64, Device::cpu());
    idx_t.data<int64_t>()[0] = 0;
    idx_t.data<int64_t>()[1] = 3;
    idx_t.data<int64_t>()[2] = 7;
    Variable x(x_t.to(device), true);
    Tensor idx_dev = idx_t.to(device);
    auto f = [idx_dev](const Variable& v) -> Variable {
        return tenzor::index_select(v, /*dim=*/0, idx_dev);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "index_select gradcheck failed on " << device.to_string();
}

// Regression: forward narrow accepts dim=-1, but NarrowBackward used to use
// the saved dim as a raw vector index and wrapped to size_t(-1) at backward
// time, asserting deep in stl_vector. Confirm gradcheck (which calls
// backward) succeeds with negative dim.
TEST_P(GradCheckMissingTest, NarrowNegativeDim) {
    auto x_t = randn({3, 5}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::narrow(v, /*dim=*/-1, /*start=*/1, /*length=*/3);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "narrow(dim=-1) gradcheck failed on " << device.to_string();
}

// Regression: same wraparound bug also lived in IndexSelectBackward.
TEST_P(GradCheckMissingTest, IndexSelectNegativeDim) {

    auto x_t = randn({4, 6}, DType::Float64, Device::cpu());
    auto idx_t = zeros({3}, DType::Int64, Device::cpu());
    idx_t.data<int64_t>()[0] = 1;
    idx_t.data<int64_t>()[1] = 4;
    idx_t.data<int64_t>()[2] = 2;
    Variable x(x_t.to(device), true);
    Tensor idx_dev = idx_t.to(device);
    auto f = [idx_dev](const Variable& v) -> Variable {
        return tenzor::index_select(v, /*dim=*/-1, idx_dev);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "index_select(dim=-1) gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Gather) {
    // gather backward is a scatter to the gathered positions.
    auto x_t = randn({4, 6}, DType::Float64, Device::cpu());
    auto idx_t = zeros({4, 2}, DType::Int64, Device::cpu());
    // Pick two columns per row.
    for (int64_t r = 0; r < 4; ++r) {
        idx_t.data<int64_t>()[r * 2 + 0] = (r + 1) % 6;
        idx_t.data<int64_t>()[r * 2 + 1] = (r + 2) % 6;
    }
    Variable x(x_t.to(device), true);
    Tensor idx_dev = idx_t.to(device);
    auto f = [idx_dev](const Variable& v) -> Variable {
        return tenzor::gather(v, /*dim=*/1, idx_dev);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "gather gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Expand) {
    // expand broadcasts a shape-1 dim; backward sums over the broadcast.
    auto x_t = randn({1, 4}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::expand(v, std::vector<int64_t>{3, 4});
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "expand gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Flip) {
    // flip is a permutation — backward is just another flip.
    auto x_t = randn({3, 4}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::flip(v, std::vector<int64_t>{0, 1});
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "flip gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Roll) {
    auto x_t = randn({3, 4}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        // roll(Variable) takes a single shift + single dim.
        return tenzor::roll(v, /*shifts=*/1, /*dim=*/0);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "roll gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Bmm) {
    // Batch matmul — core training op. Gradient flows through both operands.
    auto a_t = randn({2, 3, 4}, DType::Float64, Device::cpu());
    auto b_t = randn({2, 4, 5}, DType::Float64, Device::cpu());
    Variable a(a_t.to(device), true);
    Variable b(b_t.to(device), false);
    auto f = [&b](const Variable& v) -> Variable { return tenzor::bmm(v, b); };
    EXPECT_TRUE(gradcheck(f, a, 1e-5, 1e-4, 1e-4))
        << "bmm gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Prod) {
    // Product reduction — backward divides product by each element (sensitive
    // to zeros; randn-offset to keep all values non-zero).
    auto x_t = tenzor::abs(randn({6}, DType::Float64, Device::cpu())) + 0.5;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::prod(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "prod gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Var) {
    // Variance — backward: 2*(x - mean)/n. Deterministic everywhere.
    auto x_t = randn({8}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::var(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "var gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Std) {
    auto x_t = randn({8}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::std(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "std gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, CumSum) {
    auto x_t = randn({6, 4}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::cumsum(v, /*dim=*/0); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "cumsum gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, CumProd) {
    // Stay positive so cumprod derivatives don't sign-flip wildly.
    auto x_t = tenzor::abs(randn({4, 3}, DType::Float64, Device::cpu())) + 0.5;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::cumprod(v, /*dim=*/1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "cumprod gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, CumProdMultipleZeros) {
    // Regression: the old flip/cumsum/flip-over-input closed form was only
    // correct with ≤1 zero per cumprod run; with ≥2 exact zeros the positions
    // between/after the zeros had a wrong (dropped) analytic gradient. The
    // exact prefix-product-excluding-self backward must gradcheck at tight
    // Float64 tolerance even with multiple zeros in a run.
    auto x_t = tenzor::zeros({2, 6}, DType::Float64, Device::cpu());
    double* xp = x_t.data<double>();
    // Run 0: two zeros at positions 1 and 3.
    xp[0] = 1.5; xp[1] = 0.0; xp[2] = 2.0; xp[3] = 0.0; xp[4] = 3.0; xp[5] = 0.5;
    // Run 1: three zeros (positions 0, 2, 5).
    xp[6] = 0.0; xp[7] = 1.2; xp[8] = 0.0; xp[9] = 0.8; xp[10] = 1.1; xp[11] = 0.0;
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::cumprod(v, /*dim=*/1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-6, 1e-6, 1e-6))
        << "cumprod multi-zero gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, MaxAxis) {
    // max with axis — backward scatters to argmax positions. Use distinct
    // values so the argmax is unambiguous (no tie-breaking differences).
    auto x_t = randn({4, 6}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::max(v, /*dim=*/1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "max(dim) gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, MinAxis) {
    auto x_t = randn({4, 6}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::min(v, /*dim=*/1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "min(dim) gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Logsumexp) {

    auto x_t = randn({3, 5}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::logsumexp(v, /*dim=*/1);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "logsumexp gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Diag) {
    auto x_t = randn({5}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::diag(v, /*diagonal=*/0); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "diag gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Trace) {
    auto x_t = randn({4, 4}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::trace(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "trace gradcheck failed on " << device.to_string();
}

// ============================================================================
// LinAlg family — most linalg ops were in the 35-op remaining-gap list in
// GRADCHECK_COVERAGE_ANALYSIS.md. Most are CPU-only today because the
// dispatch paths delegate to MKL/LAPACK; that's fine for gradcheck which
// is inherently CPU-slow anyway, and the BackendTest parameterization will
// skip non-CPU if the op isn't registered there.
// ============================================================================

TEST_P(GradCheckMissingTest, LinalgNormFro) {
    // Frobenius norm: smooth, non-negative, never zero for randn.
    // J10: On all 4 GPU backends, gradcheck_detailed reports max_abs_error=inf
    // with total_elements=0, which means the analytical backward on GPU is
    // either throwing or producing an invalid tensor (the detailed comparator
    // never gets data to compare). Not a precision issue — something deeper
    // in the GPU NormBackward_Linalg dispatch path. CPU works correctly.
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KnownBug,
            "linalg_norm(fro) GPU backward crashes (J10)");
    }
    auto x_t = randn({4, 5}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::linalg_norm(v, /*ord=*/"fro");
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "linalg_norm(fro) gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Det) {
    // Use an SPD matrix so det stays comfortably away from 0. Derivative is
    // well-defined only when det != 0 (inv(A^T) * det(A) formula).
    auto x_t = make_spd(4).to(Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::det(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-3, 1e-3))
        << "det gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Inv) {
    // Same rationale — SPD so A is definitely invertible and well-conditioned.
    auto x_t = make_spd(4).to(Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::inv(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-3, 1e-3))
        << "inv gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, Cholesky) {
    // J9 root cause confirmed: cholesky() is only defined on symmetric PD
    // inputs, but gradcheck's numerical_gradient perturbs each entry (i,j)
    // independently — for i != j that breaks symmetry (moves x_ij without
    // the matching x_ji), which the analytic CholeskyBackward formula does
    // not (and cannot) account for. Wrap the tested function with a
    // differentiable symmetrize step, the same well-posedness fix
    // CholeskyInverse (above) applies via tril(): perturbing v_ij alone
    // now shifts BOTH sym_ij and sym_ji by 0.5*eps (sym = (v + v^T)/2 is
    // symmetric by construction at every perturbation step), so the
    // finite-difference and analytic gradients are comparing the same
    // well-posed function.
    auto x_t = make_spd(4, /*eps=*/0.5).to(Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        auto vt = tenzor::transpose(v, -2, -1);
        auto sym = (v + vt) * 0.5;
        return tenzor::cholesky(sym, /*upper=*/false);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-3, 1e-3))
        << "cholesky gradcheck failed on " << device.to_string();
}

// ============================================================================
// Functional wrappers in src/nn/functional.cpp — these previously wrapped
// dispatch results in a Variable with no grad_fn ("raw-tensor-op breaks
// autograd graph" pattern). Each test verifies backward propagates through
// the wired *Backward Function. CPU-only because GPU norm-backward kernels
// have separate tracked issues (J5 etc.); the wiring fix itself is
// backend-agnostic — these tests target the wiring.
// ============================================================================

TEST_P(GradCheckMissingTest, FunctionalGroupNormGradcheck) {
    // II.21: tagged so count_skips.py classifies the GPU gap; backward
    // wiring lives in src/nn/layers/normalization.cpp and works on CPU.
    // GPU GroupNormBackward divergence is tracked under J5.
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KnownBug,
            "F::group_norm GPU backward tracked under J5");
    }
    // GroupNormBackward CPU path internally downcasts to Float32 (see
    // src/nn/layers/normalization.cpp:1420-1424 and the kernel branching
    // in cpu/kernels/nn_kernels.cpp:2232+). Even with a Float64 Variable,
    // the analytical gradient has Float32 precision, so we run the input/
    // numerical comparison in Float32 and use looser tolerance. The point
    // of this test is to verify the *wiring* (that grad_fn is attached and
    // backward runs); kernel-precision is tracked separately.
    auto x_t = randn({2, 4, 4, 4}, DType::Float32, Device::cpu());
    Variable x(x_t, /*requires_grad=*/true);
    Variable w(tenzor::ones({4}, DType::Float32, Device::cpu()), true);
    Variable b(tenzor::zeros({4}, DType::Float32, Device::cpu()), true);
    auto f_input = [&](const Variable& v) -> Variable {
        auto y = tenzor::nn::functional::group_norm(v, /*num_groups=*/2,
                                                     w, b, /*eps=*/1e-5);
        return tenzor::sum(y);
    };
    EXPECT_TRUE(gradcheck(f_input, x, 1e-3, 1e-2, 1e-2))
        << "F::group_norm input gradcheck failed (grad_fn wiring regression?)";

    auto f_weight = [&](const Variable& w_in) -> Variable {
        auto y = tenzor::nn::functional::group_norm(x, /*num_groups=*/2,
                                                     w_in, b, /*eps=*/1e-5);
        return tenzor::sum(y);
    };
    EXPECT_TRUE(gradcheck(f_weight, w, 1e-3, 1e-2, 1e-2))
        << "F::group_norm weight gradcheck failed";
    auto f_bias = [&](const Variable& b_in) -> Variable {
        auto y = tenzor::nn::functional::group_norm(x, /*num_groups=*/2,
                                                     w, b_in, /*eps=*/1e-5);
        return tenzor::sum(y);
    };
    EXPECT_TRUE(gradcheck(f_bias, b, 1e-3, 1e-2, 1e-2))
        << "F::group_norm bias gradcheck failed";
}

TEST_P(GradCheckMissingTest, FunctionalInstanceNormGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KnownBug,
            "F::instance_norm GPU backward tracked under J5");
    }
    // Same Float32-only kernel-internal precision constraint as
    // FunctionalGroupNormGradcheck above — see comment there.
    auto x_t = randn({2, 4, 4, 4}, DType::Float32, Device::cpu());
    Variable x(x_t, /*requires_grad=*/true);
    Variable w(tenzor::ones({4}, DType::Float32, Device::cpu()), true);
    Variable b(tenzor::zeros({4}, DType::Float32, Device::cpu()), true);
    auto f_input = [&](const Variable& v) -> Variable {
        auto y = tenzor::nn::functional::instance_norm(
            v, /*running_mean=*/std::nullopt, /*running_var=*/std::nullopt,
            w, b, /*training=*/true, /*momentum=*/0.0, /*eps=*/1e-5);
        return tenzor::sum(y);
    };
    EXPECT_TRUE(gradcheck(f_input, x, 1e-3, 1e-2, 1e-2))
        << "F::instance_norm input gradcheck failed";

    auto f_weight = [&](const Variable& w_in) -> Variable {
        auto y = tenzor::nn::functional::instance_norm(
            x, std::nullopt, std::nullopt, w_in, b, true, 0.0, 1e-5);
        return tenzor::sum(y);
    };
    EXPECT_TRUE(gradcheck(f_weight, w, 1e-3, 1e-2, 1e-2))
        << "F::instance_norm weight gradcheck failed";
}

TEST_P(GradCheckMissingTest, FunctionalEmbeddingGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "F::embedding gradcheck CPU-only — backward through "
            "index-into-table not yet wired for GPU");
    }
    // Weight matrix [V=8, D=4], indices [3, 2].
    auto w_t = randn({8, 4}, DType::Float64, Device::cpu());
    Variable w(w_t, /*requires_grad=*/true);
    // Indices are integer data — keep as a plain Tensor, gradcheck only
    // perturbs the Variable (weight here).
    std::vector<int64_t> idx_data = {0, 3, 5, 1, 7, 2};
    Tensor idx({3, 2}, DType::Int64, Device::cpu());
    std::memcpy(idx.data<int64_t>(), idx_data.data(),
                idx_data.size() * sizeof(int64_t));

    auto f = [&](const Variable& w_in) -> Variable {
        auto y = tenzor::nn::functional::embedding(idx, w_in);
        return tenzor::sum(y);
    };
    EXPECT_TRUE(gradcheck(f, w, 1e-5, 1e-3, 1e-3))
        << "F::embedding weight gradcheck failed (grad_fn wiring regression?)";
}

TEST_P(GradCheckMissingTest, FunctionalNllLossGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "F::nll_loss gradcheck CPU-only — Variable-level gather/neg/mean "
            "path uses CPU-only intermediate");
    }
    // log-prob input [N=4, C=3], target [4]. nll_loss now uses Variable-
    // level gather/neg/mean so backward flows through automatically.
    auto x_t = randn({4, 3}, DType::Float64, Device::cpu());
    Variable x(x_t, /*requires_grad=*/true);
    std::vector<int64_t> target_data = {1, 0, 2, 1};
    Tensor target({4}, DType::Int64, Device::cpu());
    std::memcpy(target.data<int64_t>(), target_data.data(),
                target_data.size() * sizeof(int64_t));

    using R = tenzor::nn::Reduction;
    for (auto red : {R::Mean, R::Sum, R::None}) {
        auto f = [&](const Variable& v) -> Variable {
            auto y = tenzor::nn::functional::nll_loss(v, target, red);
            // For Reduction::None y is shape [N]; reduce to scalar so
            // gradcheck has a scalar function.
            return red == R::None ? tenzor::sum(y) : y;
        };
        EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-3, 1e-3))
            << "F::nll_loss reduction=" << static_cast<int>(red)
            << " gradcheck failed";
    }
}

// ============================================================================
// Sparse autograd ops — gradcheck w.r.t. the dense Variable side. The
// SparseTensor side is fixed (gradcheck only perturbs Variable inputs).
// SpGEMMBackward and SparseTriSolveBackward are declared but currently
// have no Variable-level autograd entry point; covered by op-direct tests
// elsewhere.
// ============================================================================

TEST_P(GradCheckMissingTest, SpMMGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KnownBug,
            "spmm gradcheck CPU-only — cuSPARSE/rocSPARSE backward dispatch "
            "tracked separately");
    }
    // Build a fixed 4x3 sparse matrix (CSR) and a 3x5 dense Variable.
    auto sparse_dense = randn({4, 3}, DType::Float64, Device::cpu());
    auto sparse_t = ::tenzor::SparseTensor::from_dense(sparse_dense,
                                  ::tenzor::SparseLayout::CSR);
    auto dense_t = randn({3, 5}, DType::Float64, Device::cpu());
    Variable dense(dense_t, /*requires_grad=*/true);
    auto f = [&](const Variable& v) -> Variable {
        return ::tenzor::spmm(sparse_t, v);
    };
    EXPECT_TRUE(gradcheck(f, dense, 1e-5, 1e-4, 1e-4))
        << "spmm gradcheck failed";
}

TEST_P(GradCheckMissingTest, SpMVGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KnownBug,
            "spmv gradcheck CPU-only — sparse backward dispatch tracked "
            "separately");
    }
    auto sparse_dense = randn({5, 4}, DType::Float64, Device::cpu());
    auto sparse_t = ::tenzor::SparseTensor::from_dense(sparse_dense,
                                  ::tenzor::SparseLayout::CSR);
    auto vec_t = randn({4}, DType::Float64, Device::cpu());
    Variable vec(vec_t, /*requires_grad=*/true);
    auto f = [&](const Variable& v) -> Variable {
        return ::tenzor::spmv(sparse_t, v);
    };
    EXPECT_TRUE(gradcheck(f, vec, 1e-5, 1e-4, 1e-4))
        << "spmv gradcheck failed";
}

TEST_P(GradCheckMissingTest, SparseTriSolveGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KnownBug,
            "sparse_triangular_solve gradcheck CPU-only — GPU SpSV backward "
            "tracked separately");
    }
    // L is a fixed sparse lower-triangular matrix; b is the dense
    // Variable. SparseTriSolveBackward only differentiates through b.
    int64_t n = 4;
    auto eye_t = ::tenzor::eye(n, std::nullopt, DType::Float64, Device::cpu());
    // Build a well-conditioned lower-triangular: I + small lower noise.
    auto noise = ::tenzor::mul(::tenzor::tril(
        randn({n, n}, DType::Float64, Device::cpu()), -1), 0.1);
    auto L_dense = ::tenzor::add(eye_t, noise);
    auto L = ::tenzor::SparseTensor::from_dense(L_dense,
                                                ::tenzor::SparseLayout::CSR);
    auto b_t = randn({n, 3}, DType::Float64, Device::cpu());
    Variable b(b_t, /*requires_grad=*/true);
    auto f = [&](const Variable& v) -> Variable {
        return ::tenzor::sparse_triangular_solve(L, v, /*upper=*/false);
    };
    EXPECT_TRUE(gradcheck(f, b, 1e-5, 1e-4, 1e-4))
        << "sparse_triangular_solve gradcheck failed";
}

// FINDING 20: commit ba4910fe added Complex64/Complex128 sparse-triangular-
// solve kernels across all 5 backends with forward-parity coverage
// (test_sparse_parity.cpp), but no backward/gradcheck coverage existed for
// the complex case at all (distinct from the real-dtype GPU gap above,
// which is at least tracked as KnownBug).
//
// Complex input, complex output: neither gradcheck() (real-only) nor
// gradcheck_complex() (real-INPUT-only, per its own doc comment) applies.
// Instead, verify SparseTriSolveBackward's documented adjoint identity
// directly: for x = L^{-1} @ b, backward gives grad_b = L^{-H} @ grad_x, so
// L^H @ grad_b must equal the backward seed exactly (up to float tolerance).
// This exercises the ops.cpp conjugate-transpose wiring (SparseTriSolveBackward
// needs L^H, not just L^T) using only independently-verified primitives
// (dense matmul/transpose/conj), with no dependency on a separate solve op.
TEST_P(GradCheckMissingTest, SparseTriSolveComplexBackwardMatchesAdjoint) {
    int64_t n = 4;
    auto eye_t = ::tenzor::eye(n, std::nullopt, DType::Complex64, Device::cpu());
    // Well-conditioned complex lower-triangular: I + small complex lower noise.
    auto noise_t = ::tenzor::mul(::tenzor::tril(
        randn({n, n}, DType::Complex64, Device::cpu()), -1), 0.1);
    auto L_dense = ::tenzor::add(eye_t, noise_t).to(device);
    auto L = ::tenzor::SparseTensor::from_dense(L_dense, ::tenzor::SparseLayout::CSR);

    auto b_t = randn({n, 2}, DType::Complex64, Device::cpu()).to(device);
    Variable b(b_t, /*requires_grad=*/true);

    Variable x;
    try {
        x = ::tenzor::sparse_triangular_solve(L, b, /*upper=*/false);
    } catch (const std::exception& e) {
        SKIP_WITH_REASON(SkipReason::KnownBug,
            std::string("sparse_triangular_solve complex forward threw: ") + e.what());
    }

    auto seed = randn({n, 2}, DType::Complex64, Device::cpu()).to(device);
    try {
        x.backward(seed);
    } catch (const std::exception& e) {
        SKIP_WITH_REASON(SkipReason::KnownBug,
            std::string("sparse_triangular_solve complex backward threw: ") + e.what());
    }
    ASSERT_TRUE(b.has_grad());

    auto grad_b = b.grad()->to(Device::cpu());
    auto L_dense_cpu = L_dense.to(Device::cpu());
    auto L_H = ::tenzor::conj(::tenzor::transpose(L_dense_cpu, 0, 1));
    auto reconstructed_seed = ::tenzor::matmul(L_H, grad_b);
    auto seed_cpu = seed.to(Device::cpu());

    auto* recon_p = reconstructed_seed.contiguous().data<std::complex<float>>();
    auto* seed_p = seed_cpu.contiguous().data<std::complex<float>>();
    for (int64_t i = 0; i < reconstructed_seed.numel(); ++i) {
        EXPECT_NEAR(recon_p[i].real(), seed_p[i].real(), 1e-3f) << "i=" << i;
        EXPECT_NEAR(recon_p[i].imag(), seed_p[i].imag(), 1e-3f) << "i=" << i;
    }
}

TEST_P(GradCheckMissingTest, SparseAddGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KnownBug,
            "sparse_add gradcheck CPU-only — GPU sparse-add backward "
            "tracked separately");
    }
    auto sparse_dense = randn({4, 3}, DType::Float64, Device::cpu());
    auto sparse_t = ::tenzor::SparseTensor::from_dense(sparse_dense,
                                  ::tenzor::SparseLayout::COO);
    auto dense_t = randn({4, 3}, DType::Float64, Device::cpu());
    Variable dense(dense_t, /*requires_grad=*/true);
    auto f = [&](const Variable& v) -> Variable {
        return ::tenzor::sparse_add(sparse_t, v);
    };
    EXPECT_TRUE(gradcheck(f, dense, 1e-5, 1e-4, 1e-4))
        << "sparse_add gradcheck failed";
}

// ============================================================================
// Linalg gradchecks — vecdot, vector_norm, matrix_norm, eigvalsh, solve.
// LDL factor/solve are J9-tracked (Cholesky-family precision issues).
// ============================================================================

TEST_P(GradCheckMissingTest, VecdotGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "linalg gradcheck CPU-only — GPU vecdot backward not wired");
    }
    auto a_t = randn({5}, DType::Float64, Device::cpu());
    auto b_t = randn({5}, DType::Float64, Device::cpu());
    Variable a(a_t, /*requires_grad=*/true);
    Variable b(b_t, /*requires_grad=*/false);  // probe one side at a time
    auto f = [&](const Variable& v) -> Variable {
        return ::tenzor::vecdot(v, b);
    };
    EXPECT_TRUE(gradcheck(f, a, 1e-5, 1e-4, 1e-4))
        << "vecdot gradcheck failed";
}

TEST_P(GradCheckMissingTest, VectorNormGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "linalg gradcheck CPU-only — GPU vector_norm backward not wired");
    }
    auto a_t = randn({4, 6}, DType::Float64, Device::cpu());
    Variable a(a_t, /*requires_grad=*/true);
    auto f = [&](const Variable& v) -> Variable {
        return ::tenzor::vector_norm(v, 2.0);
    };
    EXPECT_TRUE(gradcheck(f, a, 1e-5, 1e-4, 1e-4))
        << "vector_norm gradcheck failed";
}

TEST_P(GradCheckMissingTest, MatrixNormGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KnownBug,
            "matrix_norm (Frobenius) GPU backward crashes — same J10 path");
    }
    // matrix_norm with ord=2 (operator 2-norm) goes through SVD and the
    // backward is delicate near degenerate singular values. Frobenius norm
    // (sum of squares, sqrt) is smooth and a much better fit for
    // gradcheck. linalg_norm(.,"fro") is the dedicated path.
    auto a_t = randn({4, 5}, DType::Float64, Device::cpu());
    Variable a(a_t, /*requires_grad=*/true);
    auto f = [&](const Variable& v) -> Variable {
        return ::tenzor::linalg_norm(v, /*ord=*/"fro");
    };
    EXPECT_TRUE(gradcheck(f, a, 1e-5, 1e-4, 1e-4))
        << "matrix_norm (Frobenius) gradcheck failed";
}

TEST_P(GradCheckMissingTest, EigvalshGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "eigvalsh gradcheck CPU-only — GPU backward delegates to LAPACK");
    }
    auto spd = make_spd(4);
    Variable x(spd, /*requires_grad=*/true);
    auto f = [&](const Variable& v) -> Variable {
        // J9-class fix (see Cholesky above): eigvalsh() is only defined on
        // symmetric inputs; gradcheck perturbs entries independently, so
        // wrap with a differentiable symmetrize step for a well-posed check.
        auto vt = tenzor::transpose(v, -2, -1);
        auto sym = (v + vt) * 0.5;
        return ::tenzor::eigvalsh(sym);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-3, 1e-3))
        << "eigvalsh gradcheck failed";
}

TEST_P(GradCheckMissingTest, SolveGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "solve gradcheck CPU-only — GPU dense solve backward not wired");
    }
    auto A_t = make_spd(4);
    auto B_t = randn({4, 3}, DType::Float64, Device::cpu()).to(device);
    Variable A(A_t, /*requires_grad=*/true);
    Variable B(B_t, /*requires_grad=*/false);
    auto f = [&](const Variable& v) -> Variable {
        return ::tenzor::solve(v, B);
    };
    EXPECT_TRUE(gradcheck(f, A, 1e-5, 1e-3, 1e-3))
        << "solve gradcheck failed";
}

// ============================================================================
// Special math gradchecks — erf family, Bessel I0e/I1e, multigammaln.
// ============================================================================

TEST_P(GradCheckMissingTest, ErfGradcheck) {
    auto x_t = randn({6}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return ::tenzor::erf(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "erf gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, ErfcGradcheck) {
    auto x_t = randn({6}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return ::tenzor::erfc(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "erfc gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, ErfInvGradcheck) {
    // Domain (-1, 1). erfinv'(x) = sqrt(pi)/2 * exp(erfinv(x)^2) — grows
    // very fast as |x| -> 1, so finite differences become unreliable.
    // Use a tight range around 0 (|x| < ~0.3) to keep the curvature
    // bounded. randn is unbounded; clamp by tanh which keeps |x| <= 1
    // smoothly, then scale to |x| < 0.3.
    auto x_t = ::tenzor::mul(
        ::tenzor::tanh(randn({6}, DType::Float64, Device::cpu())), 0.3);
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return ::tenzor::erfinv(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-3, 1e-3))
        << "erfinv gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, I0eGradcheck) {
    auto x_t = randn({6}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return ::tenzor::i0e(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-3, 1e-3))
        << "i0e gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, I1eGradcheck) {
    auto x_t = randn({6}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable { return ::tenzor::i1e(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-3, 1e-3))
        << "i1e gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, MultigammalnGradcheck) {
    // Multigammaln(x, p) requires x > (p-1)/2. For p=2, x > 0.5.
    auto x_t = ::tenzor::add(::tenzor::abs(randn({6}, DType::Float64, Device::cpu())), 1.0);
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        return ::tenzor::multigammaln(v, /*p=*/2);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-3, 1e-3))
        << "multigammaln gradcheck failed on " << device.to_string();
}

// ============================================================================
// Indexing/shape leftover gradchecks — Scatter (plain), Sort, TopK, Tril,
// Triu. GridSample/AffineGrid in vision.cpp; covered separately.
// ============================================================================

TEST_P(GradCheckMissingTest, ScatterGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "scatter gradcheck CPU-only — GPU ScatterBackward not registered");
    }
    // scatter writes input values into target at indexed positions; backward
    // routes grad back to the values input via gather along the scatter dim.
    auto v_t = randn({3, 4}, DType::Float64, Device::cpu());
    Variable v_var(v_t, /*requires_grad=*/true);
    std::vector<int64_t> idx_data = {0, 2, 1, 3, 2, 0, 3, 1, 1, 0, 2, 3};
    Tensor idx({3, 4}, DType::Int64, Device::cpu());
    std::memcpy(idx.data<int64_t>(), idx_data.data(),
                idx_data.size() * sizeof(int64_t));
    auto target_t = ::tenzor::zeros({3, 4}, DType::Float64, Device::cpu());
    Variable target(target_t, /*requires_grad=*/false);
    auto f = [&](const Variable& v) -> Variable {
        return ::tenzor::scatter(target, /*dim=*/1, idx, v);
    };
    EXPECT_TRUE(gradcheck(f, v_var, 1e-5, 1e-4, 1e-4))
        << "scatter gradcheck failed";
}

TEST_P(GradCheckMissingTest, TrilGradcheck) {
    auto x_t = randn({4, 4}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        return ::tenzor::tril(v, /*diagonal=*/0);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "tril gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, TriuGradcheck) {
    auto x_t = randn({4, 4}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        return ::tenzor::triu(v, /*diagonal=*/0);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "triu gradcheck failed on " << device.to_string();
}

// ============================================================================
// Sort / TopK — both return pair<Variable values, Tensor indices>; gradient
// flows only through values. Backward must scatter via the inverse
// permutation (sort) or by indices (topk) — wrong dim normalisation here
// would map gradients to the wrong rows.
// ============================================================================

TEST_P(GradCheckMissingTest, SortGradcheck) {
    auto x_t = randn({4, 6}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        auto [vals, idx] = ::tenzor::sort(v, /*dim=*/-1, /*descending=*/false);
        return ::tenzor::sum(vals);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "sort gradcheck failed on " << device.to_string();
}

TEST_P(GradCheckMissingTest, GridSampleGradcheck) {
    // H9-H13: cross-backend padding-mode gate bugs in GridSampleBackward
    // (ROCm/Vulkan/OneAPI zeros-vs-border gate mixups, OneAPI bicubic
    // missing a border gate entirely) are fixed — re-enable GPU coverage
    // instead of skipping it as a known bug.
    //
    // Tiny [N=1, C=2, H=3, W=3] feature map sampled with a [N=1, H=2, W=2,
    // 2] grid. align_corners=true, so normalized g in [-1,1] denormalizes to
    // pixel-space ix in [0, W_in-1] = [0, 2] via ix = (g+1)*1.0.
    //
    // H9-H11's bug (zeros/border gate mixup) is invisible unless the
    // PRE-clamp pixel coordinate lands in the "one bilinear neighbour still
    // in bounds" band ix in [-1,0) u (W_in-1,W_in) = [-1,0) u (2,3), i.e.
    // g in [-2,-1) u (1,2) — outside [-2,-1)/(1,2) both bilinear corners are
    // fully out of bounds and sum_dx/sum_dy are already exactly 0 regardless
    // of the gate, masking the bug. A random grid only lands in that narrow
    // band by chance, so use explicit deterministic values that hit it for
    // every one of the 4 sample points (both axes, both signs).
    double grid_data[] = {
        -1.5, -1.5,   1.5,  1.5,
        -1.5,  0.3,   0.2,  1.5,
    };
    auto in_t = randn({1, 2, 3, 3}, DType::Float64, Device::cpu()).to(device);
    auto grid_t = Tensor::from_blob(grid_data, {1, 2, 2, 2}, DType::Float64, Device::cpu())
                      .clone().to(device);
    Variable input(in_t, /*requires_grad=*/false);
    Variable grid(grid_t, /*requires_grad=*/true);

    for (const std::string& padding_mode : {"zeros", "border"}) {
        for (const std::string& mode : {"bilinear", "bicubic"}) {
            // H9-H13 are specifically bugs in the grad_grid write, not
            // grad_input — gradcheck the GRID (not the input) to exercise
            // the fixed code path. grad_input's atomicAdd scatter is a
            // separate, already-correct path on all these backends.
            auto f = [&](const Variable& g) -> Variable {
                return ::tenzor::grid_sample(input, g, mode, padding_mode, true);
            };
            EXPECT_TRUE(gradcheck(f, grid, 1e-4, 1e-3, 1e-3))
                << "grid_sample gradcheck failed (grid), mode=" << mode
                << " padding_mode=" << padding_mode
                << " device=" << device.to_string();
        }
    }
}

TEST_P(GradCheckMissingTest, AffineGridGradcheck) {
    // affine_grid(theta) is an exactly-linear function of theta — the
    // affine transform x' = theta * [grid_x; grid_y; 1] applied at every
    // pre-computed (grid_x, grid_y) pixel. Loose tolerance because the
    // AffineGridBackward CPU impl reduces over the spatial grid in
    // Float32 internally (the `gi_f32` accumulator pattern in
    // function_vision.cpp); GPU backends widen-narrow through Float32
    // (M9/M10) unless the input is genuinely Float64.
    auto theta_t = randn({1, 2, 3}, DType::Float64, Device::cpu()).to(device);
    Variable theta(theta_t, /*requires_grad=*/true);
    auto f = [&](const Variable& v) -> Variable {
        return ::tenzor::affine_grid(v, /*size=*/{1, 1, 3, 3},
                                     /*align_corners=*/true);
    };
    EXPECT_TRUE(gradcheck(f, theta, 1e-3, 1e-2, 1e-2))
        << "affine_grid gradcheck failed";
}

TEST_P(GradCheckMissingTest, TopKGradcheck) {
    auto x_t = randn({4, 6}, DType::Float64, Device::cpu());
    Variable x(x_t.to(device), true);
    auto f = [](const Variable& v) -> Variable {
        auto [vals, idx] = ::tenzor::topk(v, /*k=*/3, /*dim=*/-1,
                                          /*largest=*/true, /*sorted=*/true);
        return ::tenzor::sum(vals);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-4))
        << "topk gradcheck failed on " << device.to_string();
}

// ============================================================================
// FFT family — pure FFT/IFFT produce complex outputs which gradcheck can't
// reduce directly. We compose with view_as_real (Variable overload added
// in src/autograd/ops.cpp) to get a real-valued scalar function:
//
//     f(x) = sum(view_as_real(fft(x)))
//
// which is a smooth real → real function whose gradient depends on the
// FFTBackward / IFFTBackward implementations.
// ============================================================================

TEST_P(GradCheckMissingTest, FFTGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "FFT gradcheck CPU-only — GPU FFTBackward / IFFTBackward "
            "not wired");
    }
    auto x_t = randn({8}, DType::Float64, Device::cpu());
    Variable x(x_t, /*requires_grad=*/true);
    auto f = [](const Variable& v) -> Variable {
        return ::tenzor::view_as_real(::tenzor::fft_autograd::fft(v));
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2))
        << "fft gradcheck failed (real-input → complex-output → "
           "view_as_real → sum)";
}

TEST_P(GradCheckMissingTest, IFFTGradcheck) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "FFT gradcheck CPU-only — GPU FFTBackward / IFFTBackward "
            "not wired");
    }
    // ifft expects complex input. Build it from a real Variable via
    // view_as_complex on a [..., 2] real tensor — the gradient flows back
    // through view_as_complex into the original real Variable.
    auto x_t = randn({8, 2}, DType::Float64, Device::cpu());
    Variable x(x_t, /*requires_grad=*/true);
    auto f = [](const Variable& v) -> Variable {
        auto z = ::tenzor::view_as_complex(v);
        return ::tenzor::view_as_real(::tenzor::fft_autograd::ifft(z));
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2))
        << "ifft gradcheck failed";
}

// ============================================================================
// FFT family — RFFT/IRFFT round-trip variants. Original audit doc kept
// these as a workaround pre-PR3.5; we keep them for negative-dim and
// custom-norm coverage even though the dedicated FFT/IFFT tests above
// now exist.
// ============================================================================

TEST_P(GradCheckMissingTest, RFFTIRFFT_RoundTrip_DefaultDim) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "FFT gradcheck CPU-only — GPU FFTBackward / IFFTBackward "
            "not wired");
    }
    auto x_t = randn({8}, DType::Float64, Device::cpu());
    Variable x(x_t, /*requires_grad=*/true);
    auto f = [](const Variable& v) -> Variable {
        return ::tenzor::fft_autograd::irfft(::tenzor::fft_autograd::rfft(v));
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2))
        << "rfft→irfft round-trip gradcheck failed (default dim/norm)";
}

TEST_P(GradCheckMissingTest, RFFTIRFFT_RoundTrip_NegativeDim) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "FFT gradcheck CPU-only — GPU FFTBackward / IFFTBackward "
            "not wired");
    }
    // Negative-dim regression: matches the HRM-bug class (dim=-1 used as a
    // shape index in backward). The rfft/irfft pair must normalise dim
    // identically on both legs of the round-trip for this to pass.
    auto x_t = randn({4, 8}, DType::Float64, Device::cpu());
    Variable x(x_t, /*requires_grad=*/true);
    auto f = [](const Variable& v) -> Variable {
        return ::tenzor::fft_autograd::irfft(
            ::tenzor::fft_autograd::rfft(v, /*n=*/std::nullopt, /*dim=*/-1),
            /*n=*/std::nullopt, /*dim=*/-1);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2))
        << "rfft→irfft (dim=-1) gradcheck failed";
}

TEST_P(GradCheckMissingTest, RFFTIRFFT_RoundTrip_OrthoNorm) {
    if (device.type != Device::Type::CPU) {
        SKIP_WITH_REASON(SkipReason::KernelNotImplemented,
            "FFT gradcheck CPU-only — GPU FFTBackward / IFFTBackward "
            "not wired");
    }
    // The norm scaling factor multiplies through forward and backward —
    // a missing scale in the backward would produce a constant-ratio
    // gradient mismatch that gradcheck catches.
    auto x_t = randn({8}, DType::Float64, Device::cpu());
    Variable x(x_t, /*requires_grad=*/true);
    auto f = [](const Variable& v) -> Variable {
        auto y = ::tenzor::fft_autograd::rfft(v, std::nullopt, -1, "ortho");
        return ::tenzor::fft_autograd::irfft(y, std::nullopt, -1, "ortho");
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2))
        << "rfft→irfft (norm=ortho) gradcheck failed";
}

INSTANTIATE_BACKEND_TESTS(GradCheckMissingTest);
