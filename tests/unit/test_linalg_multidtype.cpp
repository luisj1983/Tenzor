/**
 * @file test_linalg_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for core linalg operations:
 *        det, inv, solve, cholesky, qr, svd, eigh
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/ops/linalg.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/transform.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::testing;

class LinalgMultiDTypeTest : public MultiBackendDTypeTest {};

// Helper: create a matrix from float data, cast to test dtype/device
static Tensor make_matrix(std::vector<int64_t> shape, const float* data,
                          Device dev, DType dt) {
    auto t = zeros(shape, DType::Float32, Device::cpu());
    int64_t n = 1;
    for (auto s : shape) n *= s;
    std::memcpy(t.data<float>(), data, n * sizeof(float));
    return t.to(dt).to(dev);
}

// All linalg tests skip Float16 — precision too low for decompositions
#define SKIP_IF_FLOAT16() \
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 too imprecise for linalg"

TEST_P(LinalgMultiDTypeTest, Det2x2) {
    SKIP_IF_FLOAT16();
    float data[] = {1, 2, 3, 4};
    auto A = make_matrix({2, 2}, data, device(), dtype());

    auto d = linalg::det(A);
    auto d_cpu = d.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(d_cpu.data<float>()[0], -2.0f, atol());
}

TEST_P(LinalgMultiDTypeTest, DetIdentity) {
    SKIP_IF_FLOAT16();
    auto I = eye(4, std::nullopt, dtype(), device());
    auto d = linalg::det(I);
    auto d_cpu = d.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(d_cpu.data<float>()[0], 1.0f, atol());
}

TEST_P(LinalgMultiDTypeTest, Inv3x3) {
    SKIP_IF_FLOAT16();
    float data[] = {2, 1, 0, 0, 3, 1, 1, 0, 2};
    auto A = make_matrix({3, 3}, data, device(), dtype());

    auto Ainv = linalg::inv(A);
    auto I_approx = matmul(A, Ainv);

    auto I_cpu = I_approx.to(Device::cpu()).to(DType::Float32);
    auto* p = I_cpu.data<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(p[i * 3 + j], expected, atol())
                << "i=" << i << " j=" << j;
        }
    }
}

TEST_P(LinalgMultiDTypeTest, Solve2x2) {
    SKIP_IF_FLOAT16();
    float a[] = {2, 1, 1, 3};
    float b[] = {5, 10};
    auto A = make_matrix({2, 2}, a, device(), dtype());
    auto B = make_matrix({2, 1}, b, device(), dtype());
    auto x = linalg::solve(A, B);

    auto x_cpu = x.to(Device::cpu()).to(DType::Float32);
    auto* xp = x_cpu.data<float>();
    EXPECT_NEAR(xp[0], 1.0f, atol());
    EXPECT_NEAR(xp[1], 3.0f, atol());
}

TEST_P(LinalgMultiDTypeTest, Cholesky3x3) {
    SKIP_IF_FLOAT16();
    float data[] = {4, 2, 0, 2, 5, 1, 0, 1, 3};
    auto A = make_matrix({3, 3}, data, device(), dtype());

    auto L = linalg::cholesky(A, false);
    auto Lt = transpose(L, 0, 1);
    auto A_rec = matmul(L, Lt);

    auto A_rec_cpu = A_rec.to(Device::cpu()).to(DType::Float32);
    auto A_orig_cpu = A.to(Device::cpu()).to(DType::Float32);
    auto* rp = A_rec_cpu.data<float>();
    auto* op = A_orig_cpu.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(rp[i], op[i], atol()) << "index=" << i;
    }
}

TEST_P(LinalgMultiDTypeTest, QR3x3) {
    SKIP_IF_FLOAT16();
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 10};
    auto A = make_matrix({3, 3}, data, device(), dtype());

    auto [Q, R] = linalg::qr(A);
    auto A_rec = matmul(Q, R);
    auto A_rec_cpu = A_rec.to(Device::cpu()).to(DType::Float32);
    auto A_orig_cpu = A.to(Device::cpu()).to(DType::Float32);
    auto* rp = A_rec_cpu.data<float>();
    auto* op = A_orig_cpu.data<float>();
    float tol = (dtype() == DType::Float64) ? 1e-6f : 1e-3f;
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(rp[i], op[i], tol) << "index=" << i;
    }
}

TEST_P(LinalgMultiDTypeTest, SVD3x3) {
    SKIP_IF_FLOAT16();
    float data[] = {1, 0, 0, 0, 2, 0, 0, 0, 3};
    auto A = make_matrix({3, 3}, data, device(), dtype());

    auto [U, S, Vt] = linalg::svd(A, true);
    auto S_cpu = S.to(Device::cpu()).to(DType::Float32);
    auto* sp = S_cpu.data<float>();
    EXPECT_NEAR(sp[0], 3.0f, atol());
    EXPECT_NEAR(sp[1], 2.0f, atol());
    EXPECT_NEAR(sp[2], 1.0f, atol());
}

TEST_P(LinalgMultiDTypeTest, Eigh2x2) {
    SKIP_IF_FLOAT16();
    float data[] = {2, 1, 1, 2};
    auto A = make_matrix({2, 2}, data, device(), dtype());

    auto [W, V] = linalg::eigh(A);
    auto W_cpu = W.to(Device::cpu()).to(DType::Float32);
    auto* wp = W_cpu.data<float>();
    EXPECT_NEAR(wp[0], 1.0f, atol());
    EXPECT_NEAR(wp[1], 3.0f, atol());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LinalgMultiDTypeTest);
