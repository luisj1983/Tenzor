/**
 * @file test_gradcheck_comprehensive.cpp
 * @brief Comprehensive numerical gradient checking for all autograd operations.
 *
 * Verifies backward pass correctness via finite-difference gradient checking
 * for operations in include/tenzor/autograd/ops.hpp. Cross-backend: each test
 * runs on every available backend. gradcheck() is device-aware — it perturbs
 * on a CPU copy and runs the forward pass on the input's device — so inputs are
 * simply created on `device` and passed in as usual.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fft.hpp>
#include <cmath>

#include "gradcheck_complex.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class GradCheckComprehensiveTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        set_grad_enabled(true);
    }

    // Create a small Variable with positive values in roughly [0.5, 3.2].
    // abs() + offset guarantees strictly positive inputs so ops like log /
    // sqrt / sinc never hit a singularity regardless of random seed.
    static auto make_var(std::vector<int64_t> shape, const tenzor::Device& device) -> Variable {
        auto t = randn(shape, DType::Float32, device);
        auto abs_t = tenzor::abs(t);
        auto scale = full(shape, 0.3f, DType::Float32, device);
        auto offset = full(shape, 0.5f, DType::Float32, device);
        auto shifted = tenzor::add(tenzor::mul(abs_t, scale), offset);
        return Variable(shifted, true);
    }

    // Variable with values in [-0.5, 0.5] for ops that need centered data
    static auto make_centered_var(std::vector<int64_t> shape, const tenzor::Device& device) -> Variable {
        auto t = randn(shape, DType::Float32, device);
        auto scale = full(shape, 0.3f, DType::Float32, device);
        auto scaled = tenzor::mul(t, scale);
        return Variable(scaled, true);
    }

    // Variable with values in (0.05, 0.95) for ops like erfinv
    static auto make_unit_var(std::vector<int64_t> shape, const tenzor::Device& device) -> Variable {
        auto t = randn(shape, DType::Float32, device);
        // Manual sigmoid to stay in Tensor domain
        auto neg_t = tenzor::neg(t);
        auto exp_neg = tenzor::exp(neg_t);
        auto one = full(shape, 1.0f, DType::Float32, device);
        auto sig_t = tenzor::div(one, tenzor::add(one, exp_neg));
        auto clamped = tenzor::clamp(sig_t, 0.05f, 0.95f);
        return Variable(clamped, true);
    }

    // Square matrix variable
    static auto make_square_var(int64_t n, const tenzor::Device& device) -> Variable {
        // Host write of the diagonal: build on CPU, then move to device.
        auto t = randn({n, n}, DType::Float32, Device::cpu());
        // Make diagonally dominant for invertibility
        for (int64_t i = 0; i < n; ++i) {
            t.data<float>()[i * n + i] += 3.0f;
        }
        return Variable(t.to(device), true);
    }

    // Positive-definite matrix for cholesky etc.
    static auto make_posdef_var(int64_t n, const tenzor::Device& device) -> Variable {
        auto t = randn({n, n}, DType::Float32, device);
        // A^T A + I ensures positive definite (use tensor-level ops)
        auto t_T = tenzor::transpose(t, 0, 1);
        auto ata = tenzor::matmul(t_T, t);
        auto eye_t = eye(n, std::nullopt, DType::Float32, device);
        auto result = tenzor::add(ata, eye_t);
        return Variable(result, true);
    }
};

// ============================================================================
// Reduction Operations
// ============================================================================

TEST_P(GradCheckComprehensiveTest, Min) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return min(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Max) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return max(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Std) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return tenzor::std(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Var) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return tenzor::var(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Prod) {
    auto x = make_var({3, 4}, device);
    auto f = [](const Variable& v) { return prod(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Logsumexp) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return logsumexp(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-3));
}

TEST_P(GradCheckComprehensiveTest, Cumsum) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return cumsum(v, 1); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Cumprod) {
    auto x = make_var({3, 4}, device);
    auto f = [](const Variable& v) { return cumprod(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

// ============================================================================
// Activation Functions
// ============================================================================

TEST_P(GradCheckComprehensiveTest, Softmax) {
    auto x = make_centered_var({4, 8}, device);
    auto f = [](const Variable& v) { return softmax(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-3));
}

TEST_P(GradCheckComprehensiveTest, LogSoftmax) {
    // Deterministic, well-conditioned input that is identical on every backend
    // (randn is drawn per-device, so each backend otherwise sees different data
    // — defeating the parity check and occasionally drawing a row that pushes
    // the Float32 central difference past atol on a single element). log_softmax
    // is smooth, so a fixed moderate-range input is a clean, reproducible check.
    std::vector<float> vals(32);
    for (int i = 0; i < 32; ++i) vals[i] = 0.4f * std::sin(0.7f * static_cast<float>(i));
    auto x = Variable(from_data(vals.data(), {4, 8}, device), true);
    auto f = [](const Variable& v) { return log_softmax(v, 1); };
    // log_softmax backward involves exp(y) * sum(grad_y), which loses a
    // couple of decimal digits in Float32 for batched data. The default
    // Float32-adjusted tolerances (atol=5e-4, rtol=1e-2) are tight enough
    // to catch real regressions here.
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Elu) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return elu(v, 1.0f); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Selu) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return selu(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Mish) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return mish(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, LeakyRelu) {
    // leaky_relu has a non-differentiable kink at x==0; central-difference
    // gradcheck is mathematically invalid for any sample within eps of it.
    // Use a deterministic input bounded away from zero (|x| >= 0.15, identical
    // on every backend) that still exercises both the positive (slope 1) and
    // negative (slope alpha) branches. The previous per-device randn could
    // draw a value at the kink and fail despite a correct kernel.
    std::vector<float> vals(24);
    for (int i = 0; i < 24; ++i) {
        float mag = 0.15f + 0.07f * static_cast<float>(i % 5);
        vals[i] = ((i % 2 == 0) ? 1.0f : -1.0f) * mag;
    }
    auto x = Variable(from_data(vals.data(), {4, 6}, device), true);
    auto f = [](const Variable& v) { return leaky_relu(v, 0.01f); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Softplus) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return softplus(v, 1.0f); };
    EXPECT_TRUE(gradcheck(f, x));
}

// ============================================================================
// Tensor Manipulation Operations
// ============================================================================

TEST_P(GradCheckComprehensiveTest, Reshape) {
    auto x = make_var({3, 4}, device);
    auto f = [](const Variable& v) { return reshape(v, {2, 6}); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Transpose) {
    auto x = make_var({3, 5}, device);
    auto f = [](const Variable& v) { return transpose(v, 0, 1); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Permute) {
    auto x = make_var({2, 3, 4}, device);
    auto f = [](const Variable& v) { return permute(v, {2, 0, 1}); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Squeeze) {
    auto x = make_var({3, 1, 4}, device);
    auto f = [](const Variable& v) { return squeeze(v, 1); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Unsqueeze) {
    auto x = make_var({3, 4}, device);
    auto f = [](const Variable& v) { return unsqueeze(v, 1); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Flatten) {
    auto x = make_var({2, 3, 4}, device);
    auto f = [](const Variable& v) { return flatten(v, 0, -1); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Cat) {
    auto a = make_var({3, 4}, device);
    auto b = make_var({2, 4}, device);
    auto f = [&b](const Variable& v) {
        return cat({v, b}, 0);
    };
    EXPECT_TRUE(gradcheck(f, a));
}

TEST_P(GradCheckComprehensiveTest, Slice) {
    auto x = make_var({6, 4}, device);
    auto f = [](const Variable& v) { return slice(v, 0, 1, 4); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Narrow) {
    auto x = make_var({6, 4}, device);
    auto f = [](const Variable& v) { return narrow(v, 0, 1, 3); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Flip) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return flip(v, {0, 1}); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Roll) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return roll(v, 2, 0); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Repeat) {
    auto x = make_var({2, 3}, device);
    auto f = [](const Variable& v) { return repeat(v, {2, 3}); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Expand) {
    auto x = make_var({1, 4}, device);
    auto f = [](const Variable& v) { return expand(v, {3, 4}); };
    EXPECT_TRUE(gradcheck(f, x));
}

// ============================================================================
// Indexing Operations
// ============================================================================

TEST_P(GradCheckComprehensiveTest, Gather) {
    auto x = make_var({4, 6}, device);
    auto idx = randint(0, 6, {4, 3}, DType::Int64, device);
    auto f = [&idx](const Variable& v) { return gather(v, 1, idx); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, IndexSelect) {
    auto x = make_var({6, 4}, device);
    // Host write of index values: build on CPU, then move to device.
    auto idx_cpu = zeros({3}, DType::Int64, Device::cpu());
    idx_cpu.data<int64_t>()[0] = 0;
    idx_cpu.data<int64_t>()[1] = 2;
    idx_cpu.data<int64_t>()[2] = 4;
    auto idx_data = idx_cpu.to(device);
    auto f = [&idx_data](const Variable& v) { return index_select(v, 0, idx_data); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Where) {
    auto x = make_var({4, 6}, device);
    auto y = make_var({4, 6}, device);
    // Host write of the boolean condition: build on CPU, then move to device.
    auto cond_cpu = zeros({4, 6}, DType::Bool, Device::cpu());
    auto* cond_data = cond_cpu.data<bool>();
    for (int64_t i = 0; i < 24; ++i) cond_data[i] = (i % 2 == 0);
    auto cond_var = Variable(cond_cpu.to(device), false);
    auto f = [&y, &cond_var](const Variable& v) { return where(cond_var, v, y); };
    EXPECT_TRUE(gradcheck(f, x));
}

// ============================================================================
// Linear Algebra Operations
// ============================================================================

TEST_P(GradCheckComprehensiveTest, Linear) {
    auto x = make_var({3, 4}, device);
    auto w = make_var({5, 4}, device);
    auto b_var = make_var({5}, device);
    auto f = [&w, &b_var](const Variable& v) { return linear(v, w, b_var); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Bmm) {
    auto a = make_var({2, 3, 4}, device);
    auto b = make_var({2, 4, 5}, device);
    auto f = [&b](const Variable& v) { return bmm(v, b); };
    EXPECT_TRUE(gradcheck(f, a));
}

TEST_P(GradCheckComprehensiveTest, Det) {
    auto x = make_square_var(3, device);
    auto f = [](const Variable& v) { return det(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Inv) {
    auto x = make_square_var(3, device);
    auto f = [](const Variable& v) { return inv(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Solve) {
    auto A = make_square_var(3, device);
    auto B = make_var({3, 2}, device);
    auto f = [&B](const Variable& v) { return solve(v, B); };
    EXPECT_TRUE(gradcheck(f, A, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Cholesky) {
    // Cholesky is only defined for symmetric positive-definite inputs, so the
    // straight f(A) = cholesky(A) isn't valid under gradcheck's asymmetric
    // finite-difference perturbations. Wrap the input in A @ A^T + I so the
    // actual cholesky argument is symmetric-PD for any perturbation of A, and
    // the composite gradient stays well-defined.
    auto x = make_var({3, 3}, device);
    auto dev = device;
    auto f = [dev](const Variable& v) {
        auto vt = tenzor::transpose(v, 0, 1);
        auto sym = tenzor::matmul(v, vt);
        auto n = v.shape()[0];
        auto i = eye(n, std::nullopt, DType::Float32, dev);
        return cholesky(sym + Variable(i, false));
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Diag) {
    auto x = make_var({4}, device);
    auto f = [](const Variable& v) { return diag(v, 0); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Trace) {
    auto x = make_var({4, 4}, device);
    auto f = [](const Variable& v) { return trace(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Triu) {
    auto x = make_var({4, 4}, device);
    auto f = [](const Variable& v) { return triu(v, 0); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Tril) {
    auto x = make_var({4, 4}, device);
    auto f = [](const Variable& v) { return tril(v, 0); };
    EXPECT_TRUE(gradcheck(f, x));
}

// ============================================================================
// Math Operations
// ============================================================================

TEST_P(GradCheckComprehensiveTest, Clamp) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return clamp(v, -0.2f, 0.2f); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Pow) {
    auto x = make_var({4, 6}, device);  // positive values
    auto f = [](const Variable& v) { return pow(v, 2.5f); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Reciprocal) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return reciprocal(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Sqrt) {
    auto x = make_var({4, 6}, device);  // positive values
    auto f = [](const Variable& v) { return sqrt(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

// ============================================================================
// Trigonometric Operations
// ============================================================================

TEST_P(GradCheckComprehensiveTest, Tan) {
    auto x = make_centered_var({4, 6}, device);  // small values to avoid tan singularities
    auto f = [](const Variable& v) { return tan(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Asin) {
    auto x = make_unit_var({4, 6}, device);  // values in (0, 1) ⊂ (-1, 1)
    auto f = [](const Variable& v) { return asin(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Acos) {
    auto x = make_unit_var({4, 6}, device);
    auto f = [](const Variable& v) { return acos(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Atan) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return atan(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Sinh) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return sinh(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Cosh) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return cosh(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

// ============================================================================
// Special Math Functions
// ============================================================================

TEST_P(GradCheckComprehensiveTest, Erf) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return erf(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Erfc) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return erfc(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Erfinv) {
    auto x = make_unit_var({4, 6}, device);  // values in (0.05, 0.95)
    // Shift to (-0.9, 0.9) for erfinv domain
    auto scale = full({4, 6}, 1.8f, DType::Float32, device);
    auto offset = full({4, 6}, 0.9f, DType::Float32, device);
    auto shifted = tenzor::sub(tenzor::mul(x.tensor(), scale), offset);
    auto x_erfinv = Variable(shifted, true);
    auto f = [](const Variable& v) { return erfinv(v); };
    EXPECT_TRUE(gradcheck(f, x_erfinv, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Lgamma) {
    auto x = make_var({4, 6}, device);  // positive values
    auto f = [](const Variable& v) { return lgamma(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Digamma) {
    auto x = make_var({4, 6}, device);  // positive values
    auto f = [](const Variable& v) { return digamma(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Sinc) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return sinc(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Log2) {
    auto x = make_var({4, 6}, device);  // positive
    auto f = [](const Variable& v) { return log2(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Log10) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return log10(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Log1p) {
    auto x = make_var({4, 6}, device);  // > -1
    auto f = [](const Variable& v) { return log1p(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Exp2) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return exp2(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, Expm1) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return expm1(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, BesselI0) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return bessel_i0(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, BesselI1) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return bessel_i1(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, I0e) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return i0e(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, I1e) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return i1e(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Ndtr) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return ndtr(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(GradCheckComprehensiveTest, LogNdtr) {
    auto x = make_centered_var({4, 6}, device);
    auto f = [](const Variable& v) { return log_ndtr(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, SphericalBesselJ0) {
    auto x = make_var({4, 6}, device);
    auto f = [](const Variable& v) { return spherical_bessel_j0(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

// ============================================================================
// Binary Operations
// ============================================================================

TEST_P(GradCheckComprehensiveTest, Atan2) {
    auto y_var = make_var({4, 6}, device);
    auto x_var = make_var({4, 6}, device);
    auto f = [&x_var](const Variable& v) { return atan2(v, x_var); };
    EXPECT_TRUE(gradcheck(f, y_var, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Logaddexp) {
    auto a = make_centered_var({4, 6}, device);
    auto b = make_centered_var({4, 6}, device);
    auto f = [&b](const Variable& v) { return logaddexp(v, b); };
    EXPECT_TRUE(gradcheck(f, a));
}

TEST_P(GradCheckComprehensiveTest, Logaddexp2) {
    auto a = make_centered_var({4, 6}, device);
    auto b = make_centered_var({4, 6}, device);
    auto f = [&b](const Variable& v) { return logaddexp2(v, b); };
    EXPECT_TRUE(gradcheck(f, a));
}

TEST_P(GradCheckComprehensiveTest, CosineSimilarity) {
    auto a = make_var({4, 8}, device);
    auto b = make_var({4, 8}, device);
    auto f = [&b](const Variable& v) { return cosine_similarity(v, b, 1); };
    EXPECT_TRUE(gradcheck(f, a, 1e-4, 1e-3, 1e-2));
}

// ============================================================================
// Missing Priority 1-3 Operations
// ============================================================================

TEST_P(GradCheckComprehensiveTest, Scatter) {
    auto x = make_var({4, 8}, device);
    auto src = make_var({4, 3}, device);
    auto idx = randint(0, 8, {4, 3}, DType::Int64, device);
    auto f = [&src, &idx](const Variable& v) { return scatter(v, 1, idx, src); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, ScatterAdd) {
    auto x = make_var({4, 8}, device);
    auto src = make_var({4, 3}, device);
    auto idx = randint(0, 8, {4, 3}, DType::Int64, device);
    auto f = [&src, &idx](const Variable& v) { return scatter_add(v, 1, idx, src); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, AsStrided) {
    auto x = make_var({6, 8}, device);
    auto f = [](const Variable& v) {
        std::vector<int64_t> size = {3, 4};
        std::vector<int64_t> stride = {8, 1};
        return as_strided(v, size, stride);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-5, 1e-4, 1e-3));
}

TEST_P(GradCheckComprehensiveTest, TopK) {
    // Gradcheck the REAL autograd topk values output (not a slice proxy).
    // make_var produces strictly-positive, distinct random values, so the
    // top-k boundary is well away from ties and the scatter-to-original-
    // position backward is differentiable under finite differences.
    // A position-dependent weight (squared values) makes the reduction
    // sensitive to WHICH input positions the gradient is scattered back to,
    // so a wrong scatter would fail the check (a plain sum would not).
    auto x = make_var({4, 16}, device);
    auto f = [](const Variable& v) {
        auto vals = tenzor::topk(v, 5, 1, /*largest=*/true,
                                 /*sorted=*/true).first;
        return sum(vals * vals);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Sort) {
    // Gradcheck the REAL autograd sort values output. The backward scatters
    // grad through the inverse permutation; weighting the sorted values by a
    // per-position coefficient (arange) makes the reduction order-sensitive,
    // so an incorrect inverse permutation in the backward fails the check.
    auto x = make_var({4, 8}, device);
    // Deterministic, distinct per-position weights [1..8] (no grad).
    auto w_t = arange(1.0, 9.0, 1.0, DType::Float32, device).reshape({1, 8});
    auto w = Variable(w_t, /*requires_grad=*/false);
    auto f = [&w](const Variable& v) {
        auto sorted = tenzor::sort(v, 1, /*descending=*/false).first;
        return sum(sorted * w);
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, Gamma) {
    auto x = make_var({4, 6}, device);  // positive values
    auto f = [](const Variable& v) { return gamma(v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-3, 1e-2, 1e-1));
}

TEST_P(GradCheckComprehensiveTest, Polygamma) {
    auto x = make_var({4, 6}, device);  // positive values
    auto f = [](const Variable& v) { return polygamma(1, v); };
    EXPECT_TRUE(gradcheck(f, x, 1e-3, 1e-2, 1e-1));
}

// ============================================================================
// FFT Operations
// ============================================================================

// FFT / IFFT produce complex outputs from real inputs. The library-level
// gradcheck() handles this via a scalar contraction L = sum(Re(y) + Im(y));
// the stronger `gradcheck_complex` helper (see gradcheck_complex.hpp) runs
// TWO backward passes — one seeded `(1+0i)` and one `(0+1i)` — so the real
// and imaginary parts of the Jacobian are exercised independently.
TEST_P(GradCheckComprehensiveTest, FFT) {
    auto x = make_centered_var({8}, device);
    auto f = [](const Variable& v) {
        return fft_autograd::fft(v);
    };
    EXPECT_TRUE(tenzor_test::gradcheck_complex(f, x, 1e-3, 5e-2, 5e-2));
}

TEST_P(GradCheckComprehensiveTest, IFFT) {
    auto x = make_centered_var({8}, device);
    auto f = [](const Variable& v) {
        return fft_autograd::ifft(v);
    };
    EXPECT_TRUE(tenzor_test::gradcheck_complex(f, x, 1e-3, 5e-2, 5e-2));
}

TEST_P(GradCheckComprehensiveTest, RFFT) {
    auto x = make_centered_var({8}, device);
    auto f = [](const Variable& v) {
        return fft_autograd::irfft(fft_autograd::rfft(v));
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

TEST_P(GradCheckComprehensiveTest, IRFFT) {
    auto x = make_centered_var({8}, device);
    auto f = [](const Variable& v) {
        return fft_autograd::irfft(fft_autograd::rfft(v));
    };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

// ============================================================================
// Finding-fix regression coverage (2026-07 autograd review)
// ============================================================================

// Helper: build a gradient-tracked DCT node (no fft_autograd::dct wrapper
// exists). For the square (unset-n) case DCTBackward reconstructs the adjoint
// length from grad_output, so no saved input is required here.
static auto dct_autograd(const Variable& v, const std::string& norm) -> Variable {
    auto out_t = ::tenzor::fft::dct(v.tensor(), 2, std::nullopt, -1, norm);
    auto out = Variable(out_t, true);
    auto gf = std::make_shared<DCTBackward>(2, static_cast<int64_t>(-1),
                                            static_cast<int64_t>(-1), norm);
    gf->set_next_functions({v.grad_fn()});
    gf->set_input_variables({v});
    out.set_grad_fn(gf);
    return out;
}

// N-01: DCT backward for the default "backward" norm must be the true adjoint
// (transpose C^T), not the inverse idct. This is the case the old code got
// wrong (mis-scaled by ~O(1/N)); gradcheck it on every backend.
TEST_P(GradCheckComprehensiveTest, DCTBackwardDefaultNorm) {
    auto x = make_centered_var({8}, device);
    auto f = [](const Variable& v) { return dct_autograd(v, "backward"); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

// N-01: the orthonormal path (already correct) must still gradcheck.
TEST_P(GradCheckComprehensiveTest, DCTBackwardOrtho) {
    auto x = make_centered_var({8}, device);
    auto f = [](const Variable& v) { return dct_autograd(v, "ortho"); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

// N-01: forward normalisation should not matter for correctness — "forward"
// norm must also produce the true adjoint.
TEST_P(GradCheckComprehensiveTest, DCTBackwardForwardNorm) {
    auto x = make_centered_var({8}, device);
    auto f = [](const Variable& v) { return dct_autograd(v, "forward"); };
    EXPECT_TRUE(gradcheck(f, x, 1e-4, 1e-3, 1e-2));
}

// J-02 companion: LogitBackward must honour eps — zero gradient outside
// [eps, 1-eps] (the forward clamps there, so the derivative is 0), and
// 1/(x(1-x)) strictly inside. The pre-fix reverse-mode ignored eps.
TEST_P(GradCheckComprehensiveTest, LogitEpsClampsGradient) {
    const double eps = 0.1;
    auto xc = zeros({4}, DType::Float32, Device::cpu());
    xc.data<float>()[0] = 0.02f;  // below eps      -> grad 0
    xc.data<float>()[1] = 0.50f;  // inside window  -> 1/(0.5*0.5) = 4
    xc.data<float>()[2] = 0.97f;  // above 1-eps    -> grad 0
    xc.data<float>()[3] = 0.30f;  // inside window  -> 1/(0.3*0.7)
    Variable x(xc.to(device), true);
    auto y = logit(x, eps);
    y.backward(ones_like(y.tensor()));
    auto g = x.grad().value().to(Device::cpu());
    EXPECT_NEAR(g.data<float>()[0], 0.0f, 1e-4);
    EXPECT_NEAR(g.data<float>()[1], 4.0f, 2e-3);
    EXPECT_NEAR(g.data<float>()[2], 0.0f, 1e-4);
    EXPECT_NEAR(g.data<float>()[3], 1.0f / (0.3f * 0.7f), 2e-3);
}

// N-02: MFCC backward now uses the true STFT adjoint (stft_adjoint = STFT^H
// without window-sum normalisation) instead of istft (the inverse). STFTBackward
// routes through the same stft_adjoint_impl, so gradchecking STFT validates the
// adjoint MFCC reuses. Real input, complex output → gradcheck_complex.
TEST_P(GradCheckComprehensiveTest, STFTAdjoint) {
    auto x = make_centered_var({16}, device);
    auto f = [](const Variable& v) {
        return fft_autograd::stft(v, /*n_fft=*/8, /*hop=*/4);
    };
    EXPECT_TRUE(tenzor_test::gradcheck_complex(f, x, 1e-3, 5e-2, 5e-2));
}

INSTANTIATE_BACKEND_TESTS(GradCheckComprehensiveTest);

// M-01: complex sqrt backward must use the Wirtinger conjugate:
// grad_input = grad_output / (2 * conj(sqrt(z))). Complex sqrt is a CPU forward
// path, so this is a CPU-targeted check. The second assertion proves the test is
// discriminating (the pre-fix formula g/(2*sqrt(z)) genuinely differs here).
class ComplexCpuAutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        ::tenzor::testing::EnsureInitialized();
        set_grad_enabled(true);
    }
};

TEST_F(ComplexCpuAutogradTest, SqrtComplexConjugation) {
    const auto dev = Device::cpu();
    auto re = zeros({3}, DType::Float32, dev);
    auto im = zeros({3}, DType::Float32, dev);
    re.data<float>()[0] = 1.5f; re.data<float>()[1] = 2.0f; re.data<float>()[2] = 0.7f;
    im.data<float>()[0] = 0.4f; im.data<float>()[1] = -0.3f; im.data<float>()[2] = 0.9f;
    Variable z(tenzor::complex(re, im), true);
    auto y = sqrt(z);
    auto g = tenzor::complex(full({3}, 1.0f, DType::Float32, dev),
                             full({3}, 0.5f, DType::Float32, dev));
    y.backward(g);
    auto grad = z.grad().value();
    auto expected = tenzor::div(g, tenzor::mul(tenzor::conj(y.tensor()), 2.0));
    float max_err = tenzor::max(tenzor::abs(tenzor::sub(grad, expected))).item<float>();
    EXPECT_LT(max_err, 1e-5f);
    // Sanity: conjugation must matter for these inputs, else the test is vacuous.
    auto unfixed = tenzor::div(g, tenzor::mul(y.tensor(), 2.0));
    float conj_gap = tenzor::max(tenzor::abs(tenzor::sub(expected, unfixed))).item<float>();
    EXPECT_GT(conj_gap, 1e-3f);
}

// M-02/M-03/M-04: the higher-order (create_graph=true) path routes through
// backward_with_variables, which previously dropped the complex Wirtinger conj
// that the first-order backward() applies. These tests exercise create_graph=true
// and assert the higher-order first derivative equals the first-order path (and
// the analytic conj convention), with a discriminating check that conj matters.
TEST_F(ComplexCpuAutogradTest, LogComplexHigherOrderConjugation) {
    const auto dev = Device::cpu();
    auto re = zeros({3}, DType::Float32, dev);
    auto im = zeros({3}, DType::Float32, dev);
    re.data<float>()[0] = 1.5f; re.data<float>()[1] = 2.0f; re.data<float>()[2] = 0.7f;
    im.data<float>()[0] = 0.4f; im.data<float>()[1] = -0.3f; im.data<float>()[2] = 0.9f;
    auto g = tenzor::complex(full({3}, 1.0f, DType::Float32, dev),
                             full({3}, 0.5f, DType::Float32, dev));

    Variable z1(tenzor::complex(re, im), true);
    auto y1 = log(z1);
    y1.backward(g, /*retain_graph=*/false, /*create_graph=*/false);
    auto grad_first = z1.grad().value();

    Variable z2(tenzor::complex(re, im), true);
    auto y2 = log(z2);
    y2.backward(g, /*retain_graph=*/false, /*create_graph=*/true);
    auto grad_ho = z2.grad().value();

    // higher-order first derivative must match the first-order path
    EXPECT_LT(tenzor::max(tenzor::abs(tenzor::sub(grad_ho, grad_first))).item<float>(), 1e-5f);
    // and the analytic conj convention: g / conj(z)
    auto expected = tenzor::div(g, tenzor::conj(z1.tensor()));
    EXPECT_LT(tenzor::max(tenzor::abs(tenzor::sub(grad_ho, expected))).item<float>(), 1e-5f);
    // discriminating: the pre-fix (no-conj) formula genuinely differs
    auto unfixed = tenzor::div(g, z1.tensor());
    EXPECT_GT(tenzor::max(tenzor::abs(tenzor::sub(expected, unfixed))).item<float>(), 1e-3f);
}

TEST_F(ComplexCpuAutogradTest, ExpComplexHigherOrderConjugation) {
    const auto dev = Device::cpu();
    auto re = zeros({3}, DType::Float32, dev);
    auto im = zeros({3}, DType::Float32, dev);
    re.data<float>()[0] = 0.5f; re.data<float>()[1] = -0.2f; re.data<float>()[2] = 0.3f;
    im.data<float>()[0] = 0.4f; im.data<float>()[1] = 0.7f; im.data<float>()[2] = -0.6f;
    auto g = tenzor::complex(full({3}, 1.0f, DType::Float32, dev),
                             full({3}, 0.5f, DType::Float32, dev));

    Variable z1(tenzor::complex(re, im), true);
    auto y1 = exp(z1);
    y1.backward(g, false, false);
    auto grad_first = z1.grad().value();

    Variable z2(tenzor::complex(re, im), true);
    auto y2 = exp(z2);
    y2.backward(g, false, true);
    auto grad_ho = z2.grad().value();

    EXPECT_LT(tenzor::max(tenzor::abs(tenzor::sub(grad_ho, grad_first))).item<float>(), 1e-5f);
    // analytic: g * conj(exp(z))
    auto expected = tenzor::mul(g, tenzor::conj(y1.tensor()));
    EXPECT_LT(tenzor::max(tenzor::abs(tenzor::sub(grad_ho, expected))).item<float>(), 1e-5f);
    auto unfixed = tenzor::mul(g, y1.tensor());
    EXPECT_GT(tenzor::max(tenzor::abs(tenzor::sub(expected, unfixed))).item<float>(), 1e-3f);
}

TEST_F(ComplexCpuAutogradTest, AbsComplexHigherOrderConjugation) {
    const auto dev = Device::cpu();
    auto re = zeros({3}, DType::Float32, dev);
    auto im = zeros({3}, DType::Float32, dev);
    re.data<float>()[0] = 1.5f; re.data<float>()[1] = 2.0f; re.data<float>()[2] = 0.7f;
    im.data<float>()[0] = 0.4f; im.data<float>()[1] = -0.3f; im.data<float>()[2] = 0.9f;
    // abs of complex is real, so the incoming grad is real
    auto g = full({3}, 1.25f, DType::Float32, dev);

    Variable z1(tenzor::complex(re, im), true);
    auto y1 = abs(z1);
    y1.backward(g, false, false);
    auto grad_first = z1.grad().value();

    Variable z2(tenzor::complex(re, im), true);
    auto y2 = abs(z2);
    y2.backward(g, false, true);
    auto grad_ho = z2.grad().value();

    // higher-order path must match the first-order complex sgn(z) result
    EXPECT_LT(tenzor::max(tenzor::abs(tenzor::sub(grad_ho, grad_first))).item<float>(), 1e-5f);
    // analytic: g * z/|z|  (grad promoted to complex)
    auto scale = tenzor::div(z1.tensor(), tenzor::abs(z1.tensor()));
    auto expected = tenzor::mul(g.to(DType::Complex64), scale);
    EXPECT_LT(tenzor::max(tenzor::abs(tenzor::sub(grad_ho, expected))).item<float>(), 1e-5f);
    // discriminating: gradient must be genuinely complex (imag part non-trivial)
    EXPECT_GT(tenzor::max(tenzor::abs(tenzor::imag(grad_ho))).item<float>(), 1e-3f);
}

// L-01: EigvalshBackward for a complex Hermitian input must use the conjugate
// transpose V^H (like EighBackward), not V^T. Verify by consistency: the
// gradient of sum(eigenvalues) via eigvalsh must equal the eigh path (whose
// backward already used adjoint). A V^T bug would make the two disagree.
TEST_F(ComplexCpuAutogradTest, EigvalshComplexMatchesEigh) {
    const auto dev = Device::cpu();
    const int64_t n = 3;
    auto B = tenzor::complex(randn({n, n}, DType::Float32, dev),
                             randn({n, n}, DType::Float32, dev));
    auto Bh = tenzor::conj(tenzor::transpose(B, 0, 1));
    auto A = tenzor::mul(tenzor::add(B, Bh), 0.5);  // Hermitian

    Variable a1(A, true);
    auto w1 = eigvalsh(a1);
    sum(w1).backward();
    auto g1 = a1.grad().value();

    Variable a2(A, true);
    auto eh = eigh(a2);
    sum(std::get<0>(eh)).backward();
    auto g2 = a2.grad().value();

    float err = tenzor::max(tenzor::abs(tenzor::sub(g1, g2))).item<float>();
    EXPECT_LT(err, 1e-4f);
}
