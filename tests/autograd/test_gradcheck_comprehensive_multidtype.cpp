/**
 * @file test_gradcheck_comprehensive_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for comprehensive gradient checking
 *
 * Converted from test_gradcheck_comprehensive.cpp. Gradcheck tests skip Float16
 * due to insufficient precision for finite-difference numerical gradient computation.
 * Helper functions create tensors on the parameterized device/dtype instead of
 * hardcoded CPU/Float32.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class GradCheckComprehensiveMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        set_grad_enabled(true);
    }

    // Create a small Variable with random values in [0.1, 1.1] to avoid singularities
    auto make_var(std::vector<int64_t> shape) -> Variable {
        auto t = randn(shape, DType::Float32, Device::cpu());
        auto scale = full(shape, 0.3f, DType::Float32, Device::cpu());
        auto offset = full(shape, 0.5f, DType::Float32, Device::cpu());
        auto shifted = tenzor::add(tenzor::mul(t, scale), offset);
        if (dtype() != DType::Float32) shifted = shifted.to(dtype());
        shifted = shifted.to(device());
        return Variable(shifted, true);
    }

    // Variable with values in [-0.5, 0.5] for ops that need centered data
    auto make_centered_var(std::vector<int64_t> shape) -> Variable {
        auto t = randn(shape, DType::Float32, Device::cpu());
        auto scale = full(shape, 0.3f, DType::Float32, Device::cpu());
        auto scaled = tenzor::mul(t, scale);
        if (dtype() != DType::Float32) scaled = scaled.to(dtype());
        scaled = scaled.to(device());
        return Variable(scaled, true);
    }

    // Variable with values in (0.05, 0.95) for ops like erfinv
    auto make_unit_var(std::vector<int64_t> shape) -> Variable {
        auto t = randn(shape, DType::Float32, Device::cpu());
        auto neg_t = tenzor::neg(t);
        auto exp_neg = tenzor::exp(neg_t);
        auto one = full(shape, 1.0f, DType::Float32, Device::cpu());
        auto sig_t = tenzor::div(one, tenzor::add(one, exp_neg));
        auto clamped = tenzor::clamp(sig_t, 0.05f, 0.95f);
        if (dtype() != DType::Float32) clamped = clamped.to(dtype());
        clamped = clamped.to(device());
        return Variable(clamped, true);
    }

    // Square matrix variable
    auto make_square_var(int64_t n) -> Variable {
        auto t = randn({n, n}, DType::Float32, Device::cpu());
        for (int64_t i = 0; i < n; ++i) {
            t.data<float>()[i * n + i] += 3.0f;
        }
        if (dtype() != DType::Float32) t = t.to(dtype());
        t = t.to(device());
        return Variable(t, true);
    }

    // Positive-definite matrix for cholesky etc.
    auto make_posdef_var(int64_t n) -> Variable {
        auto t = randn({n, n}, DType::Float32, Device::cpu());
        auto t_T = tenzor::transpose(t, 0, 1);
        auto ata = tenzor::matmul(t_T, t);
        auto eye_t = eye(n, std::nullopt, DType::Float32, Device::cpu());
        auto result = tenzor::add(ata, eye_t);
        if (dtype() != DType::Float32) result = result.to(dtype());
        result = result.to(device());
        return Variable(result, true);
    }

    // Gradcheck tolerances appropriate for the dtype
    double gc_atol() const {
        return (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    }
    double gc_rtol() const {
        return (dtype() == DType::Float64) ? 1e-4 : 1e-2;
    }
    double gc_eps() const {
        return (dtype() == DType::Float64) ? 1e-6 : 1e-4;
    }
};

// ============================================================================
// Reduction Operations
// ============================================================================

TEST_P(GradCheckComprehensiveMultiDTypeTest, Min) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return min(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Max) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return max(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Std) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return tenzor::std(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Var) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return tenzor::var(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Prod) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({3, 4});
    auto f = [](const Variable& v) { return prod(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Logsumexp) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return logsumexp(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Cumsum) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return cumsum(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Cumprod) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({3, 4});
    auto f = [](const Variable& v) { return cumprod(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

// ============================================================================
// Activation Functions
// ============================================================================

TEST_P(GradCheckComprehensiveMultiDTypeTest, Softmax) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 8});
    auto f = [](const Variable& v) { return softmax(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, LogSoftmax) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 8});
    auto f = [](const Variable& v) { return log_softmax(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Elu) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return elu(v, 1.0f); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Selu) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return selu(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Mish) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return mish(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, LeakyRelu) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return leaky_relu(v, 0.01f); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Softplus) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return softplus(v, 1.0f); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

// ============================================================================
// Tensor Manipulation Operations
// ============================================================================

TEST_P(GradCheckComprehensiveMultiDTypeTest, Reshape) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({3, 4});
    auto f = [](const Variable& v) { return reshape(v, {2, 6}); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Transpose) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({3, 5});
    auto f = [](const Variable& v) { return transpose(v, 0, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Permute) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({2, 3, 4});
    auto f = [](const Variable& v) { return permute(v, {2, 0, 1}); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Squeeze) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({3, 1, 4});
    auto f = [](const Variable& v) { return squeeze(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Unsqueeze) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({3, 4});
    auto f = [](const Variable& v) { return unsqueeze(v, 1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Flatten) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({2, 3, 4});
    auto f = [](const Variable& v) { return flatten(v, 0, -1); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Cat) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto a = make_var({3, 4});
    auto b = make_var({2, 4});
    auto f = [&b](const Variable& v) {
        return cat({v, b}, 0);
    };
    EXPECT_TRUE(gradcheck(f, a, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Slice) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({6, 4});
    auto f = [](const Variable& v) { return slice(v, 0, 1, 4); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Narrow) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({6, 4});
    auto f = [](const Variable& v) { return narrow(v, 0, 1, 3); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Flip) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return flip(v, {0, 1}); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Roll) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return roll(v, 2, 0); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Repeat) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({2, 3});
    auto f = [](const Variable& v) { return repeat(v, {2, 3}); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Expand) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({1, 4});
    auto f = [](const Variable& v) { return expand(v, {3, 4}); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

// ============================================================================
// Indexing Operations
// ============================================================================

TEST_P(GradCheckComprehensiveMultiDTypeTest, Gather) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto idx = randint(0, 6, {4, 3}, DType::Int64, device());
    auto f = [&idx](const Variable& v) { return gather(v, 1, idx); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, IndexSelect) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({6, 4});
    auto idx_data = zeros({3}, DType::Int64, Device::cpu());
    idx_data.data<int64_t>()[0] = 0;
    idx_data.data<int64_t>()[1] = 2;
    idx_data.data<int64_t>()[2] = 4;
    idx_data = idx_data.to(device());
    auto f = [&idx_data](const Variable& v) { return index_select(v, 0, idx_data); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Where) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto y = make_var({4, 6});
    auto cond_t = zeros({4, 6}, DType::Bool, device());
    auto cond_cpu = cond_t.to(Device::cpu());
    auto* cond_data = cond_cpu.data<bool>();
    for (int64_t i = 0; i < 24; ++i) cond_data[i] = (i % 2 == 0);
    cond_t = cond_cpu.to(device());
    auto cond_var = Variable(cond_t, false);
    auto f = [&y, &cond_var](const Variable& v) { return where(cond_var, v, y); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

// ============================================================================
// Linear Algebra Operations
// ============================================================================

TEST_P(GradCheckComprehensiveMultiDTypeTest, Linear) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({3, 4});
    auto w = make_var({5, 4});
    auto b_var = make_var({5});
    auto f = [&w, &b_var](const Variable& v) { return linear(v, w, b_var); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Bmm) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto a = make_var({2, 3, 4});
    auto b = make_var({2, 4, 5});
    auto f = [&b](const Variable& v) { return bmm(v, b); };
    EXPECT_TRUE(gradcheck(f, a, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Det) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_square_var(3);
    auto f = [](const Variable& v) { return det(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Inv) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_square_var(3);
    auto f = [](const Variable& v) { return inv(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Solve) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto A = make_square_var(3);
    auto B = make_var({3, 2});
    auto f = [&B](const Variable& v) { return solve(v, B); };
    EXPECT_TRUE(gradcheck(f, A, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Cholesky) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_posdef_var(3);
    auto f = [](const Variable& v) { return cholesky(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Diag) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4});
    auto f = [](const Variable& v) { return diag(v, 0); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Trace) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 4});
    auto f = [](const Variable& v) { return trace(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Triu) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 4});
    auto f = [](const Variable& v) { return triu(v, 0); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Tril) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 4});
    auto f = [](const Variable& v) { return tril(v, 0); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

// ============================================================================
// Math Operations
// ============================================================================

TEST_P(GradCheckComprehensiveMultiDTypeTest, Clamp) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return clamp(v, -0.2f, 0.2f); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Pow) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return pow(v, 2.5f); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Reciprocal) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return reciprocal(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Sqrt) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return sqrt(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

// ============================================================================
// Trigonometric Operations
// ============================================================================

TEST_P(GradCheckComprehensiveMultiDTypeTest, Tan) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return tan(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Asin) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_unit_var({4, 6});
    auto f = [](const Variable& v) { return asin(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Acos) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_unit_var({4, 6});
    auto f = [](const Variable& v) { return acos(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Atan) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return atan(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Sinh) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return sinh(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Cosh) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return cosh(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

// ============================================================================
// Special Math Functions
// ============================================================================

TEST_P(GradCheckComprehensiveMultiDTypeTest, Erf) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return erf(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Erfc) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return erfc(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Lgamma) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return lgamma(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Digamma) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return digamma(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Log2) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return log2(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Log10) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return log10(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Log1p) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_var({4, 6});
    auto f = [](const Variable& v) { return log1p(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Exp2) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return exp2(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Expm1) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({4, 6});
    auto f = [](const Variable& v) { return expm1(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

// ============================================================================
// Binary Operations
// ============================================================================

TEST_P(GradCheckComprehensiveMultiDTypeTest, Atan2) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto y_var = make_var({4, 6});
    auto x_var = make_var({4, 6});
    auto f = [&x_var](const Variable& v) { return atan2(v, x_var); };
    EXPECT_TRUE(gradcheck(f, y_var, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, Logaddexp) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto a = make_centered_var({4, 6});
    auto b = make_centered_var({4, 6});
    auto f = [&b](const Variable& v) { return logaddexp(v, b); };
    EXPECT_TRUE(gradcheck(f, a, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, CosineSimilarity) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto a = make_var({4, 8});
    auto b = make_var({4, 8});
    auto f = [&b](const Variable& v) { return cosine_similarity(v, b, 1); };
    EXPECT_TRUE(gradcheck(f, a, gc_eps(), gc_atol(), gc_rtol()));
}

// ============================================================================
// FFT Operations
// ============================================================================

TEST_P(GradCheckComprehensiveMultiDTypeTest, FFT) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({8});
    auto f = [](const Variable& v) { return fft_autograd::fft(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, IFFT) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({8});
    auto f = [](const Variable& v) { return fft_autograd::ifft(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, RFFT) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({8});
    auto f = [](const Variable& v) { return fft_autograd::rfft(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

TEST_P(GradCheckComprehensiveMultiDTypeTest, IRFFT) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Gradcheck requires Float32+ precision";
    auto x = make_centered_var({8});
    auto f = [](const Variable& v) { return fft_autograd::irfft(v); };
    EXPECT_TRUE(gradcheck(f, x, gc_eps(), gc_atol(), gc_rtol()));
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GradCheckComprehensiveMultiDTypeTest);
