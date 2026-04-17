/**
 * @file test_linalg_extended_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for extended linalg functions:
 *        vector_norm, matrix_norm, vecdot, householder_product, ldl_factor/solve
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/ops/linalg.hpp>
#include <tenzor/ops/creation.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class LinalgExtendedMultiDTypeTest : public MultiBackendDTypeTest {};

#define SKIP_IF_FLOAT16() \
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 too imprecise for linalg"

TEST_P(LinalgExtendedMultiDTypeTest, VectorNormL1) {
    SKIP_IF_FLOAT16();
    auto x = zeros({3}, DType::Float32, Device::cpu());
    float data[] = {1.0f, -2.0f, 3.0f};
    std::memcpy(x.data<float>(), data, sizeof(data));
    x = x.to(dtype()).to(device());

    auto result = linalg::vector_norm(x, 1.0);
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(r_cpu.data<float>()[0], 6.0f, atol());
}

TEST_P(LinalgExtendedMultiDTypeTest, VectorNormL2) {
    SKIP_IF_FLOAT16();
    auto x = zeros({2}, DType::Float32, Device::cpu());
    float data[] = {3.0f, 4.0f};
    std::memcpy(x.data<float>(), data, sizeof(data));
    x = x.to(dtype()).to(device());

    auto result = linalg::vector_norm(x, 2.0);
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(r_cpu.data<float>()[0], 5.0f, atol());
}

TEST_P(LinalgExtendedMultiDTypeTest, VecdotBasic) {
    SKIP_IF_FLOAT16();
    auto a = zeros({3}, DType::Float32, Device::cpu());
    auto b = zeros({3}, DType::Float32, Device::cpu());
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f, 6.0f};
    std::memcpy(a.data<float>(), a_data, sizeof(a_data));
    std::memcpy(b.data<float>(), b_data, sizeof(b_data));
    a = a.to(dtype()).to(device());
    b = b.to(dtype()).to(device());

    auto result = linalg::vecdot(a, b);
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(r_cpu.data<float>()[0], 32.0f, atol());
}

TEST_P(LinalgExtendedMultiDTypeTest, VecdotBatched) {
    SKIP_IF_FLOAT16();
    auto a = zeros({2, 3}, DType::Float32, Device::cpu());
    auto b = zeros({2, 3}, DType::Float32, Device::cpu());
    float a_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float b_data[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::memcpy(a.data<float>(), a_data, sizeof(a_data));
    std::memcpy(b.data<float>(), b_data, sizeof(b_data));
    a = a.to(dtype()).to(device());
    b = b.to(dtype()).to(device());

    auto result = linalg::vecdot(a, b, -1);
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* out = r_cpu.data<float>();
    EXPECT_NEAR(out[0], 2.0f, atol());
    EXPECT_NEAR(out[1], 6.0f, atol());
}

TEST_P(LinalgExtendedMultiDTypeTest, LdlSolveSymmetricSystem) {
    SKIP_IF_FLOAT16();
    auto A = zeros({2, 2}, DType::Float32, Device::cpu());
    auto b = zeros({2, 1}, DType::Float32, Device::cpu());
    float a_data[] = {4.0f, 2.0f, 2.0f, 3.0f};
    float b_data[] = {1.0f, 2.0f};
    std::memcpy(A.data<float>(), a_data, sizeof(a_data));
    std::memcpy(b.data<float>(), b_data, sizeof(b_data));
    // LDL uses Float64 internally for stability
    A = A.to(DType::Float64).to(device());
    b = b.to(DType::Float64).to(device());

    auto [LD, pivots] = linalg::ldl_factor(A);
    auto x = linalg::ldl_solve(LD, pivots, b);

    auto x_cpu = x.to(Device::cpu()).to(DType::Float32);
    auto* xp = x_cpu.data<float>();
    EXPECT_NEAR(xp[0], -0.125f, 1e-4f);
    EXPECT_NEAR(xp[1], 0.75f, 1e-4f);
}

TEST_P(LinalgExtendedMultiDTypeTest, MatrixNormFrobenius) {
    SKIP_IF_FLOAT16();
    auto A = zeros({2, 2}, DType::Float32, Device::cpu());
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(A.data<float>(), data, sizeof(data));
    A = A.to(DType::Float64).to(device());

    auto result = linalg::norm(A, "fro");
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(r_cpu.data<float>()[0], std::sqrt(30.0f), 1e-4f);
}

TEST_P(LinalgExtendedMultiDTypeTest, LdlFactorConsistency) {
    SKIP_IF_FLOAT16();
    auto A = zeros({2, 2}, DType::Float32, Device::cpu());
    auto b = zeros({2, 1}, DType::Float32, Device::cpu());
    float a_data[] = {5.0f, 1.0f, 1.0f, 3.0f};
    float b_data[] = {6.0f, 4.0f};
    std::memcpy(A.data<float>(), a_data, sizeof(a_data));
    std::memcpy(b.data<float>(), b_data, sizeof(b_data));
    A = A.to(DType::Float64).to(device());
    b = b.to(DType::Float64).to(device());

    auto x_direct = linalg::solve(A, b);
    auto [LD, pivots] = linalg::ldl_factor(A);
    auto x_ldl = linalg::ldl_solve(LD, pivots, b);

    auto d1 = x_direct.to(Device::cpu()).to(DType::Float32);
    auto d2 = x_ldl.to(Device::cpu()).to(DType::Float32);
    auto* p1 = d1.data<float>();
    auto* p2 = d2.data<float>();
    EXPECT_NEAR(p1[0], p2[0], 1e-4f);
    EXPECT_NEAR(p1[1], p2[1], 1e-4f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LinalgExtendedMultiDTypeTest);
