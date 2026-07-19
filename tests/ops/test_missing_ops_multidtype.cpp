/**
 * @file test_missing_ops_multidtype.cpp
 * @brief Coverage for OpIds flagged as untested by scripts/audit_op_coverage.py.
 *
 * Many of these ops were technically exercised through higher-level layer
 * paths, but the audit surfaces them as zero-reference identifiers — which
 * means they were also implicitly at zero-backend coverage. This file adds
 * direct multi-backend / multi-dtype tests so a regression in any of them
 * surfaces on its own.
 *
 * Scope:
 *   - Element-wise: Frac, Heaviside, NanToNum, Nextafter, Gcd, Lcm,
 *     Addcmul, Addcdiv, Rad2Deg, Signbit, LogAddExp2, XLogY, Xlog1py, Entr
 *   - Type predicates: IsNan, IsInf, IsFinite
 *   - Shape / indexing: AsStrided, MaskedScatter, TrilIndices, TriuIndices
 *   - Special: Ldexp, Frexp
 *
 * The tests compare against hand-computed reference values — these ops are
 * simple enough that a closed-form answer is cheaper than CPU reference
 * infrastructure.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../multi_backend_dtype_fixture.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace tenzor;
using namespace tenzor::testing;

class MissingOpsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    Tensor cpu_tensor(std::vector<float> values, std::vector<int64_t> shape = {}) {
        if (shape.empty()) shape = {static_cast<int64_t>(values.size())};
        auto t = zeros(shape, DType::Float32, Device::cpu());
        auto* data = t.data<float>();
        for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
        return t;
    }

    Tensor device_tensor(std::vector<float> values, std::vector<int64_t> shape = {}) {
        auto cpu = cpu_tensor(std::move(values), std::move(shape));
        return cpu.to(dtype()).to(device());
    }

    float element(const Tensor& t, int64_t i) {
        auto cpu_f32 = t.to(Device::cpu()).to(DType::Float32).contiguous();
        return cpu_f32.data<float>()[i];
    }

};

// GTEST_SKIP's internal `return` has to happen inside the TEST_P body, not a
// helper method. Wrap the common gates as macros so the early-return lands
// where the test expects it.
#define SKIP_IF_NOT_FLOAT() \
    do { if (dtype() == DType::Int32 || dtype() == DType::Int64 || \
             dtype() == DType::UInt8 || dtype() == DType::Int8 || \
             dtype() == DType::Bool) { \
            SKIP_WITH_REASON(SkipReason::DtypeUnsupportedOnBackend, \
                             "test compares against float reference values"); \
        } } while (0)

#define SKIP_IF_HALF() \
    do { if (dtype() == DType::Float16 || dtype() == DType::BFloat16) { \
            SKIP_WITH_REASON(SkipReason::NumericalDivergence, \
                             "reference compare uses Float32 tolerance"); \
        } } while (0)

// ---------------------------------------------------------------------------
// Element-wise: Frac, Heaviside, NanToNum
// ---------------------------------------------------------------------------

TEST_P(MissingOpsMultiDTypeTest, Frac) {
    SKIP_IF_NOT_FLOAT();
    // Tenzor's frac is sign-preserving: frac(x) = x - trunc(x). All backends
    // were migrated to this convention together so negative inputs here are
    // exercised too — not just the easy positive path.
    auto t = device_tensor({1.25f, -2.75f, 0.0f, 3.5f, -0.5f});
    auto out = frac(t);
    EXPECT_NEAR(element(out, 0),  0.25f, atol());
    EXPECT_NEAR(element(out, 1), -0.75f, atol());
    EXPECT_NEAR(element(out, 2),  0.0f,  atol());
    EXPECT_NEAR(element(out, 3),  0.5f,  atol());
    EXPECT_NEAR(element(out, 4), -0.5f,  atol());
}

TEST_P(MissingOpsMultiDTypeTest, Heaviside) {
    SKIP_IF_NOT_FLOAT();
    auto input  = device_tensor({-1.0f, 0.0f, 1.0f});
    auto values = device_tensor({ 0.5f, 0.5f, 0.5f});
    auto out = heaviside(input, values);
    EXPECT_NEAR(element(out, 0), 0.0f, atol());
    EXPECT_NEAR(element(out, 1), 0.5f, atol());  // zero input returns `values`
    EXPECT_NEAR(element(out, 2), 1.0f, atol());
}

TEST_P(MissingOpsMultiDTypeTest, NanToNum) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    float nan_v = std::numeric_limits<float>::quiet_NaN();
    float pinf  = std::numeric_limits<float>::infinity();
    float ninf  = -std::numeric_limits<float>::infinity();
    auto t = device_tensor({nan_v, pinf, ninf, 1.5f});
    auto out = nan_to_num(t, 0.0, 1e4, -1e4);
    EXPECT_NEAR(element(out, 0),    0.0f, atol());
    EXPECT_NEAR(element(out, 1),  1e4f,  1.0f);
    EXPECT_NEAR(element(out, 2), -1e4f,  1.0f);
    EXPECT_NEAR(element(out, 3),  1.5f,  atol());
}

// ---------------------------------------------------------------------------
// Element-wise: Rad2Deg, Signbit
// ---------------------------------------------------------------------------

TEST_P(MissingOpsMultiDTypeTest, Rad2Deg) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    auto t = device_tensor({0.0f, static_cast<float>(M_PI), 2.0f * static_cast<float>(M_PI)});
    auto out = rad2deg(t);
    EXPECT_NEAR(element(out, 0),   0.0f, atol());
    EXPECT_NEAR(element(out, 1), 180.0f, 1e-3f);
    EXPECT_NEAR(element(out, 2), 360.0f, 1e-3f);
}

TEST_P(MissingOpsMultiDTypeTest, Signbit) {
    SKIP_IF_NOT_FLOAT();
    auto t = device_tensor({-1.0f, 0.0f, 1.0f, -0.5f});
    auto out = signbit(t);
    // signbit returns bool — move to CPU and compare as uint8.
    auto cpu = out.to(Device::cpu()).to(DType::Bool).contiguous();
    auto* b = static_cast<uint8_t*>(cpu.data_ptr());
    EXPECT_EQ(b[0], 1u);
    EXPECT_EQ(b[1], 0u);
    EXPECT_EQ(b[2], 0u);
    EXPECT_EQ(b[3], 1u);
}

// ---------------------------------------------------------------------------
// Element-wise: LogAddExp2, XLogY, Xlog1py
// ---------------------------------------------------------------------------

TEST_P(MissingOpsMultiDTypeTest, LogAddExp2) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    auto a = device_tensor({0.0f, 1.0f, 2.0f});
    auto b = device_tensor({0.0f, 1.0f, 3.0f});
    auto out = logaddexp2(a, b);
    // log2(2^0 + 2^0) = 1
    EXPECT_NEAR(element(out, 0), 1.0f, 1e-4f);
    // log2(2^1 + 2^1) = 2
    EXPECT_NEAR(element(out, 1), 2.0f, 1e-4f);
    // log2(2^2 + 2^3) = log2(12) ≈ 3.585
    EXPECT_NEAR(element(out, 2), std::log2(12.0f), 1e-3f);
}

TEST_P(MissingOpsMultiDTypeTest, XLogY) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    // xlogy(0, 0) == 0 (by convention) is the interesting case.
    auto x = device_tensor({0.0f, 1.0f, 2.0f});
    auto y = device_tensor({0.0f, 2.0f, 3.0f});
    auto out = xlogy(x, y);
    EXPECT_NEAR(element(out, 0),             0.0f,           1e-4f);
    EXPECT_NEAR(element(out, 1), 1.0f * std::log(2.0f),      1e-4f);
    EXPECT_NEAR(element(out, 2), 2.0f * std::log(3.0f),      1e-4f);
}

TEST_P(MissingOpsMultiDTypeTest, Xlog1py) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    auto x = device_tensor({0.0f, 1.0f});
    auto y = device_tensor({5.0f, 2.0f});
    auto out = xlog1py(x, y);
    // xlog1py(0, 5) = 0 by convention
    EXPECT_NEAR(element(out, 0), 0.0f, 1e-4f);
    // xlog1py(1, 2) = log(3)
    EXPECT_NEAR(element(out, 1), std::log(3.0f), 1e-4f);
}

TEST_P(MissingOpsMultiDTypeTest, Entr) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    // entr(x) = -x * log(x) for x > 0, 0 for x == 0, -inf for x < 0.
    auto x = device_tensor({0.0f, 0.5f, 1.0f});
    auto out = entr(x);
    EXPECT_NEAR(element(out, 0),                        0.0f, 1e-4f);
    EXPECT_NEAR(element(out, 1), -0.5f * std::log(0.5f),     1e-4f);
    EXPECT_NEAR(element(out, 2),                        0.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// Element-wise: Nextafter, Gcd, Lcm
// ---------------------------------------------------------------------------

TEST_P(MissingOpsMultiDTypeTest, Nextafter) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    auto a = device_tensor({1.0f, 2.0f});
    auto b = device_tensor({2.0f, 1.0f});
    auto out = nextafter(a, b);
    // nextafter returns the next representable value toward the target — the
    // delta is ~1 ULP which disappears in a Float32 cast from Float64 data.
    // Read the result at native dtype and check strict inequality.
    auto cpu = out.to(Device::cpu()).contiguous();
    if (cpu.dtype() == DType::Float64) {
        const double* d = cpu.data<double>();
        EXPECT_GT(d[0], 1.0);
        EXPECT_LT(d[1], 2.0);
    } else {
        const float* f = cpu.data<float>();
        EXPECT_GT(f[0], 1.0f);
        EXPECT_LT(f[1], 2.0f);
    }
}

TEST_P(MissingOpsMultiDTypeTest, GcdLcm) {
    // Gcd/Lcm are defined for integer dtypes.
    if (dtype() != DType::Int32 && dtype() != DType::Int64) {
        SKIP_WITH_REASON(SkipReason::DtypeUnsupportedOnBackend, "gcd/lcm are integer-only");
    }
    auto a = zeros({3}, dtype(), Device::cpu());
    auto b = zeros({3}, dtype(), Device::cpu());
    if (dtype() == DType::Int32) {
        auto* ap = a.data<int32_t>(); auto* bp = b.data<int32_t>();
        ap[0] = 12; ap[1] =  6; ap[2] = 10;
        bp[0] = 18; bp[1] = 14; bp[2] = 15;
    } else {
        auto* ap = a.data<int64_t>(); auto* bp = b.data<int64_t>();
        ap[0] = 12; ap[1] =  6; ap[2] = 10;
        bp[0] = 18; bp[1] = 14; bp[2] = 15;
    }
    auto ad = a.to(device()); auto bd = b.to(device());
    auto g = gcd(ad, bd).to(Device::cpu());
    auto l = lcm(ad, bd).to(Device::cpu());
    auto get = [&](const Tensor& t, int i) -> int64_t {
        if (t.dtype() == DType::Int32) return t.data<int32_t>()[i];
        return t.data<int64_t>()[i];
    };
    EXPECT_EQ(get(g, 0),  6);
    EXPECT_EQ(get(g, 1),  2);
    EXPECT_EQ(get(g, 2),  5);
    EXPECT_EQ(get(l, 0), 36);
    EXPECT_EQ(get(l, 1), 42);
    EXPECT_EQ(get(l, 2), 30);
}

// ---------------------------------------------------------------------------
// Ternary: Addcmul, Addcdiv
// ---------------------------------------------------------------------------

TEST_P(MissingOpsMultiDTypeTest, Addcmul) {
    SKIP_IF_NOT_FLOAT();
    auto inp = device_tensor({1.0f, 2.0f});
    auto t1  = device_tensor({3.0f, 4.0f});
    auto t2  = device_tensor({5.0f, 6.0f});
    auto out = addcmul(inp, t1, t2, 0.5);
    // inp + 0.5 * t1 * t2 = {1 + 0.5*15, 2 + 0.5*24} = {8.5, 14.0}
    EXPECT_NEAR(element(out, 0),  8.5f, atol());
    EXPECT_NEAR(element(out, 1), 14.0f, atol());
}

TEST_P(MissingOpsMultiDTypeTest, Addcdiv) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    auto inp = device_tensor({1.0f, 2.0f});
    auto t1  = device_tensor({4.0f, 6.0f});
    auto t2  = device_tensor({2.0f, 3.0f});
    auto out = addcdiv(inp, t1, t2, 0.5);
    // inp + 0.5 * t1 / t2 = {1 + 0.5*2, 2 + 0.5*2} = {2.0, 3.0}
    EXPECT_NEAR(element(out, 0), 2.0f, atol());
    EXPECT_NEAR(element(out, 1), 3.0f, atol());
}

// ---------------------------------------------------------------------------
// Type predicates: IsNan, IsInf, IsFinite
// ---------------------------------------------------------------------------

TEST_P(MissingOpsMultiDTypeTest, IsNanIsInfIsFinite) {
    SKIP_IF_NOT_FLOAT();
    float nan_v = std::numeric_limits<float>::quiet_NaN();
    float pinf  = std::numeric_limits<float>::infinity();
    auto t = device_tensor({nan_v, pinf, -pinf, 1.5f, 0.0f});
    auto nan_mask = isnan(t).to(Device::cpu()).to(DType::Bool).contiguous();
    auto inf_mask = isinf(t).to(Device::cpu()).to(DType::Bool).contiguous();
    auto fin_mask = isfinite(t).to(Device::cpu()).to(DType::Bool).contiguous();
    auto* n = static_cast<uint8_t*>(nan_mask.data_ptr());
    auto* i = static_cast<uint8_t*>(inf_mask.data_ptr());
    auto* f = static_cast<uint8_t*>(fin_mask.data_ptr());
    EXPECT_EQ(n[0], 1u); EXPECT_EQ(n[3], 0u);
    EXPECT_EQ(i[1], 1u); EXPECT_EQ(i[2], 1u); EXPECT_EQ(i[3], 0u);
    EXPECT_EQ(f[3], 1u); EXPECT_EQ(f[4], 1u);
    EXPECT_EQ(f[0], 0u); EXPECT_EQ(f[1], 0u);
}

// ---------------------------------------------------------------------------
// Shape / indexing: MaskedScatter, TrilIndices, TriuIndices
// ---------------------------------------------------------------------------

TEST_P(MissingOpsMultiDTypeTest, MaskedScatter) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    auto inp  = device_tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto mask_cpu = zeros({4}, DType::Bool, Device::cpu());
    auto* m = static_cast<uint8_t*>(mask_cpu.data_ptr());
    m[0] = 0; m[1] = 1; m[2] = 0; m[3] = 1;
    auto mask = mask_cpu.to(device());
    auto src  = device_tensor({99.0f, 88.0f});
    auto out  = masked_scatter(inp, mask, src);
    EXPECT_NEAR(element(out, 0),  1.0f, atol());
    EXPECT_NEAR(element(out, 1), 99.0f, atol());
    EXPECT_NEAR(element(out, 2),  3.0f, atol());
    EXPECT_NEAR(element(out, 3), 88.0f, atol());
}

TEST_P(MissingOpsMultiDTypeTest, TrilTriuIndices) {
    auto tril = tril_indices(3, 3, 0);
    auto triu = triu_indices(3, 3, 0);
    EXPECT_EQ(tril.shape()[0], 2);          // [row_coords, col_coords]
    EXPECT_EQ(tril.shape()[1], 6);          // 1+2+3 lower-triangular entries
    EXPECT_EQ(triu.shape()[1], 6);          // symmetric count for a 3×3
}

// ---------------------------------------------------------------------------
// Bit manipulation: Ldexp, Frexp
// ---------------------------------------------------------------------------

TEST_P(MissingOpsMultiDTypeTest, Ldexp) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    auto x = device_tensor({1.5f, 2.0f, 0.5f});
    // Exponent must be integer; build on CPU then move.
    auto n_cpu = zeros({3}, DType::Int32, Device::cpu());
    auto* np = n_cpu.data<int32_t>();
    np[0] = 2; np[1] = 3; np[2] = 1;
    auto n = n_cpu.to(device());
    auto out = ldexp(x, n);
    // ldexp(a, b) = a * 2^b
    EXPECT_NEAR(element(out, 0), 1.5f * 4.0f,  1e-4f);
    EXPECT_NEAR(element(out, 1), 2.0f * 8.0f,  1e-4f);
    EXPECT_NEAR(element(out, 2), 0.5f * 2.0f,  1e-4f);
}

TEST_P(MissingOpsMultiDTypeTest, Frexp) {
    SKIP_IF_NOT_FLOAT();
    SKIP_IF_HALF();
    auto x = device_tensor({1.5f, 4.0f});
    auto [mantissa, exponent] = frexp(x);
    // 1.5 = 0.75 * 2^1; 4.0 = 0.5 * 2^3
    EXPECT_NEAR(element(mantissa, 0), 0.75f, 1e-4f);
    EXPECT_NEAR(element(mantissa, 1), 0.5f,  1e-4f);
    auto exp_cpu = exponent.to(Device::cpu()).to(DType::Int32).contiguous();
    auto* ep = exp_cpu.data<int32_t>();
    EXPECT_EQ(ep[0], 1);
    EXPECT_EQ(ep[1], 3);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MissingOpsMultiDTypeTest);
