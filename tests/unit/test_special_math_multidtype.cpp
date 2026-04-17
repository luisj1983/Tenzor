/**
 * @file test_special_math_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for special math functions:
 *        gamma, lgamma, digamma, beta, bessel_j0, erfinv, sinc
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include <cmath>
#include <cstring>

using namespace tenzor;
using namespace tenzor::testing;

class SpecialMathMultiDTypeTest : public MultiBackendDTypeTest {};

// Helper: build 1-D tensor from float data, cast to test dtype/device
static Tensor make_vec(const float* data, int64_t n, Device dev, DType dt) {
    auto t = zeros({n}, DType::Float32, Device::cpu());
    std::memcpy(t.data<float>(), data, n * sizeof(float));
    return t.to(dt).to(dev);
}

TEST_P(SpecialMathMultiDTypeTest, Gamma) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for gamma";
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x = make_vec(input, 5, device(), dtype());
    auto result = tenzor::gamma(x);
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* d = r_cpu.data<float>();
    EXPECT_NEAR(d[0], 1.0f, atol());
    EXPECT_NEAR(d[1], 1.0f, atol());
    EXPECT_NEAR(d[2], 2.0f, atol());
    EXPECT_NEAR(d[3], 6.0f, 1e-3f);
    EXPECT_NEAR(d[4], 24.0f, 1e-2f);
}

TEST_P(SpecialMathMultiDTypeTest, Lgamma) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for lgamma";
    float input[] = {1.0f, 2.0f, 5.0f, 10.0f};
    auto x = make_vec(input, 4, device(), dtype());
    auto result = tenzor::lgamma(x);
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* d = r_cpu.data<float>();
    EXPECT_NEAR(d[0], 0.0f, atol());
    EXPECT_NEAR(d[1], 0.0f, atol());
    EXPECT_NEAR(d[2], std::lgamma(5.0f), 1e-3f);
    EXPECT_NEAR(d[3], std::lgamma(10.0f), 1e-2f);
}

TEST_P(SpecialMathMultiDTypeTest, Beta) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for beta";
    float a_data[] = {1.0f, 2.0f, 1.0f};
    float b_data[] = {1.0f, 2.0f, 2.0f};
    auto a = make_vec(a_data, 3, device(), dtype());
    auto b = make_vec(b_data, 3, device(), dtype());
    auto result = tenzor::beta(a, b);
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* d = r_cpu.data<float>();
    EXPECT_NEAR(d[0], 1.0f, atol());
    EXPECT_NEAR(d[1], 1.0f / 6.0f, 1e-3f);
    EXPECT_NEAR(d[2], 0.5f, atol());
}

TEST_P(SpecialMathMultiDTypeTest, BesselJ0) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for bessel";
    float input[] = {0.0f, 1.0f, 5.0f};
    auto x = make_vec(input, 3, device(), dtype());
    auto result = tenzor::bessel_j0(x);
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* d = r_cpu.data<float>();
    EXPECT_NEAR(d[0], 1.0f, atol());
    EXPECT_NEAR(d[1], 0.7652f, 0.01f);
}

TEST_P(SpecialMathMultiDTypeTest, ErfInvRoundtrip) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for erfinv";
    float input[] = {0.0f, 0.5f, -0.5f, 0.9f};
    auto x = make_vec(input, 4, device(), dtype());
    auto roundtrip = tenzor::erf(tenzor::erfinv(x));
    auto rt_cpu = roundtrip.to(Device::cpu()).to(DType::Float32);
    auto x_cpu = x.to(Device::cpu()).to(DType::Float32);
    auto* rt = rt_cpu.data<float>();
    auto* inp = x_cpu.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(rt[i], inp[i], 1e-3f) << "Round-trip failed at index " << i;
    }
}

TEST_P(SpecialMathMultiDTypeTest, Sinc) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for sinc";
    float input[] = {0.0f, 1.0f, 0.5f};
    auto x = make_vec(input, 3, device(), dtype());
    auto result = tenzor::sinc(x);
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* d = r_cpu.data<float>();
    EXPECT_NEAR(d[0], 1.0f, atol());
    EXPECT_NEAR(d[1], 0.0f, atol());
    EXPECT_NEAR(d[2], static_cast<float>(2.0 / M_PI), 1e-3f);
}

TEST_P(SpecialMathMultiDTypeTest, Digamma) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for digamma";
    float input[] = {1.0f, 2.0f, 5.0f};
    auto x = make_vec(input, 3, device(), dtype());
    auto result = tenzor::digamma(x);
    auto r_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto* d = r_cpu.data<float>();
    EXPECT_NEAR(d[0], -0.5772f, 0.01f);
    EXPECT_NEAR(d[1], 0.4228f, 0.01f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SpecialMathMultiDTypeTest);
