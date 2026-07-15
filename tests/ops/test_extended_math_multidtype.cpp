/**
 * @file test_extended_math_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for extended math operations
 *
 * Covers OpIds 320-339: Log2, Log10, Log1p, Exp2, Expm1, Erf, Erfc,
 * Atan2, Fmod, Remainder, Lerp
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class ExtendedMathMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Log2 Tests
// ============================================================================

TEST_P(ExtendedMathMultiDTypeTest, Log2PowersOfTwo) {
    // log2(1)=0, log2(2)=1, log2(4)=2, log2(8)=3
    auto input = tenzor::full({4}, 1.0f, DType::Float32, device()).to(dtype());
    auto input_f32 = tenzor::full({4}, 1.0f, DType::Float32, Device::cpu());
    float vals[] = {1.0f, 2.0f, 4.0f, 8.0f};
    auto* d = input_f32.data<float>();
    for (int i = 0; i < 4; ++i) d[i] = vals[i];
    input = input_f32.to(device()).to(dtype());

    auto result = tenzor::log2(input);
    expectDevice(result);
    expectDType(result);

    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    EXPECT_NEAR(r[0], 0.0f, atol());
    EXPECT_NEAR(r[1], 1.0f, atol());
    EXPECT_NEAR(r[2], 2.0f, atol());
    EXPECT_NEAR(r[3], 3.0f, atol());
}

// ============================================================================
// Log10 Tests
// ============================================================================

TEST_P(ExtendedMathMultiDTypeTest, Log10KnownValues) {
    // log10(1)=0, log10(10)=1, log10(100)=2
    auto input_f32 = tenzor::full({3}, 1.0f, DType::Float32, Device::cpu());
    auto* d = input_f32.data<float>();
    d[0] = 1.0f; d[1] = 10.0f; d[2] = 100.0f;
    auto input = input_f32.to(device()).to(dtype());

    auto result = tenzor::log10(input);
    expectDevice(result);

    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    EXPECT_NEAR(r[0], 0.0f, atol());
    EXPECT_NEAR(r[1], 1.0f, atol());
    EXPECT_NEAR(r[2], 2.0f, atol());
}

// ============================================================================
// Log1p Tests
// ============================================================================

TEST_P(ExtendedMathMultiDTypeTest, Log1pSmallValues) {
    // log1p(0)=0, log1p is more accurate than log(1+x) for small x
    auto input_f32 = tenzor::full({3}, 0.0f, DType::Float32, Device::cpu());
    auto* d = input_f32.data<float>();
    d[0] = 0.0f; d[1] = 1.0f; d[2] = 0.001f;
    auto input = input_f32.to(device()).to(dtype());

    auto result = tenzor::log1p(input);
    expectDevice(result);

    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    EXPECT_NEAR(r[0], 0.0f, atol());                   // log1p(0) = 0
    EXPECT_NEAR(r[1], std::log(2.0f), atol());          // log1p(1) = ln(2)
    EXPECT_NEAR(r[2], std::log1p(0.001f), atol());      // numerical precision
}

// ============================================================================
// Exp2 Tests
// ============================================================================

TEST_P(ExtendedMathMultiDTypeTest, Exp2KnownValues) {
    // exp2(0)=1, exp2(1)=2, exp2(3)=8
    auto input_f32 = tenzor::full({3}, 0.0f, DType::Float32, Device::cpu());
    auto* d = input_f32.data<float>();
    d[0] = 0.0f; d[1] = 1.0f; d[2] = 3.0f;
    auto input = input_f32.to(device()).to(dtype());

    auto result = tenzor::exp2(input);
    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    EXPECT_NEAR(r[0], 1.0f, atol());
    EXPECT_NEAR(r[1], 2.0f, atol());
    EXPECT_NEAR(r[2], 8.0f, atol());
}

// ============================================================================
// Expm1 Tests
// ============================================================================

TEST_P(ExtendedMathMultiDTypeTest, Expm1SmallValues) {
    // expm1(0) = 0, expm1(x) ≈ x for small x
    auto input_f32 = tenzor::full({3}, 0.0f, DType::Float32, Device::cpu());
    auto* d = input_f32.data<float>();
    d[0] = 0.0f; d[1] = 1.0f; d[2] = 0.001f;
    auto input = input_f32.to(device()).to(dtype());

    auto result = tenzor::expm1(input);
    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    EXPECT_NEAR(r[0], 0.0f, atol());                                  // expm1(0) = 0
    // Use std::expm1 (the accurate implementation) rather than
    // `exp(1.0f) - 1.0f`, which loses precision in Float32 subtraction
    // and fails the tight Float64 tolerance.
    EXPECT_NEAR(r[1], static_cast<float>(std::expm1(1.0)), atol());   // e - 1
    EXPECT_NEAR(r[2], static_cast<float>(std::expm1(0.001)),
                std::max(atol(), 1e-4f));
}

// ============================================================================
// Erf / Erfc Tests
// ============================================================================

TEST_P(ExtendedMathMultiDTypeTest, ErfKnownValues) {
    // erf(0)=0, erf is odd: erf(-x) = -erf(x), erf(large) → 1
    auto input_f32 = tenzor::full({4}, 0.0f, DType::Float32, Device::cpu());
    auto* d = input_f32.data<float>();
    d[0] = 0.0f; d[1] = 1.0f; d[2] = -1.0f; d[3] = 3.0f;
    auto input = input_f32.to(device()).to(dtype());

    auto result = tenzor::erf(input);
    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    EXPECT_NEAR(r[0], 0.0f, atol());
    EXPECT_NEAR(r[1], std::erf(1.0f), atol());
    EXPECT_NEAR(r[2], -std::erf(1.0f), atol());  // odd symmetry
    EXPECT_NEAR(r[3], std::erf(3.0f), atol());    // ≈ 0.9999779
}

TEST_P(ExtendedMathMultiDTypeTest, ErfcComplement) {
    // erfc(x) + erf(x) = 1
    auto input_f32 = tenzor::full({3}, 0.0f, DType::Float32, Device::cpu());
    auto* d = input_f32.data<float>();
    d[0] = 0.0f; d[1] = 0.5f; d[2] = 2.0f;
    auto input = input_f32.to(device()).to(dtype());

    auto erf_result = tenzor::erf(input);
    auto erfc_result = tenzor::erfc(input);
    auto sum = tenzor::add(erf_result, erfc_result);

    auto sum_f32 = sum.to(Device::cpu()).to(DType::Float32);
    auto* s = sum_f32.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(s[i], 1.0f, std::max(atol(), 1e-3f)) << "Index " << i;
    }
}

// ============================================================================
// Atan2 Tests
// ============================================================================

TEST_P(ExtendedMathMultiDTypeTest, Atan2Quadrants) {
    // atan2(y, x): test all four quadrants
    auto y_f32 = tenzor::full({4}, 0.0f, DType::Float32, Device::cpu());
    auto x_f32 = tenzor::full({4}, 0.0f, DType::Float32, Device::cpu());
    auto* yd = y_f32.data<float>();
    auto* xd = x_f32.data<float>();
    // Q1: (+, +), Q2: (+, -), Q3: (-, -), Q4: (-, +)
    yd[0] = 1.0f; xd[0] = 1.0f;   // pi/4
    yd[1] = 1.0f; xd[1] = -1.0f;  // 3pi/4
    yd[2] = -1.0f; xd[2] = -1.0f; // -3pi/4
    yd[3] = -1.0f; xd[3] = 1.0f;  // -pi/4

    auto y = y_f32.to(device()).to(dtype());
    auto x = x_f32.to(device()).to(dtype());
    auto result = tenzor::atan2(y, x);

    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    float tol = std::max(atol(), 1e-3f);
    EXPECT_NEAR(r[0], static_cast<float>(M_PI / 4.0), tol);
    EXPECT_NEAR(r[1], static_cast<float>(3.0 * M_PI / 4.0), tol);
    EXPECT_NEAR(r[2], static_cast<float>(-3.0 * M_PI / 4.0), tol);
    EXPECT_NEAR(r[3], static_cast<float>(-M_PI / 4.0), tol);
}

// ============================================================================
// Fmod Tests
// ============================================================================

TEST_P(ExtendedMathMultiDTypeTest, FmodValues) {
    // fmod(7, 3) = 1, fmod(-7, 3) = -1 (sign follows dividend)
    auto a_f32 = tenzor::full({3}, 0.0f, DType::Float32, Device::cpu());
    auto b_f32 = tenzor::full({3}, 3.0f, DType::Float32, Device::cpu());
    auto* ad = a_f32.data<float>();
    ad[0] = 7.0f; ad[1] = -7.0f; ad[2] = 5.5f;

    auto a = a_f32.to(device()).to(dtype());
    auto b = b_f32.to(device()).to(dtype());
    auto result = tenzor::fmod(a, b);

    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    EXPECT_NEAR(r[0], std::fmod(7.0f, 3.0f), atol());
    EXPECT_NEAR(r[1], std::fmod(-7.0f, 3.0f), atol());
    EXPECT_NEAR(r[2], std::fmod(5.5f, 3.0f), atol());
}

// ============================================================================
// Remainder Tests
// ============================================================================

// Reference oracle for tenzor::remainder's documented contract: the
// divisor-sign convention (a - floor(a/b)*b), matching Python/NumPy/PyTorch
// `%`/`remainder` -- NOT libm's IEEE round-to-nearest-even std::remainder.
// e.g. floor_remainder(-7, 3) == 2, not std::remainder(-7,3) == -1.
static float floor_remainder(float x, float y) {
    float r = std::fmod(x, y);
    if (r != 0.0f && ((r < 0.0f) != (y < 0.0f))) r += y;
    return r;
}

TEST_P(ExtendedMathMultiDTypeTest, RemainderValues) {
    // remainder follows the divisor's sign (different from fmod)
    auto a_f32 = tenzor::full({3}, 0.0f, DType::Float32, Device::cpu());
    auto b_f32 = tenzor::full({3}, 3.0f, DType::Float32, Device::cpu());
    auto* ad = a_f32.data<float>();
    ad[0] = 7.0f; ad[1] = -7.0f; ad[2] = 5.5f;

    auto a = a_f32.to(device()).to(dtype());
    auto b = b_f32.to(device()).to(dtype());
    auto result = tenzor::remainder(a, b);

    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    // remainder(7,3)=1, remainder(-7,3)=+2 (not -1), remainder(5.5,3)=2.5
    EXPECT_NEAR(r[0], floor_remainder(7.0f, 3.0f), atol());
    EXPECT_NEAR(r[1], floor_remainder(-7.0f, 3.0f), atol());
    EXPECT_NEAR(r[2], floor_remainder(5.5f, 3.0f), atol());
}

// ============================================================================
// Lerp Tests
// ============================================================================

TEST_P(ExtendedMathMultiDTypeTest, LerpBasic) {
    // lerp(start, end, weight): start + weight * (end - start)
    auto start_f32 = tenzor::full({3}, 0.0f, DType::Float32, Device::cpu());
    auto end_f32 = tenzor::full({3}, 10.0f, DType::Float32, Device::cpu());

    auto start = start_f32.to(device()).to(dtype());
    auto end_t = end_f32.to(device()).to(dtype());

    auto result = tenzor::lerp(start, end_t, 0.5);
    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(r[i], 5.0f, atol());
    }
}

TEST_P(ExtendedMathMultiDTypeTest, LerpBoundaries) {
    // lerp at weight=0 returns start, at weight=1 returns end
    auto start_f32 = tenzor::full({2}, 3.0f, DType::Float32, Device::cpu());
    auto end_f32 = tenzor::full({2}, 7.0f, DType::Float32, Device::cpu());

    auto start = start_f32.to(device()).to(dtype());
    auto end_t = end_f32.to(device()).to(dtype());

    auto result_0 = tenzor::lerp(start, end_t, 0.0);
    auto result_1 = tenzor::lerp(start, end_t, 1.0);

    auto r0 = result_0.to(Device::cpu()).to(DType::Float32);
    auto r1 = result_1.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(r0.data<float>()[0], 3.0f, atol());
    EXPECT_NEAR(r1.data<float>()[0], 7.0f, atol());
}

TEST_P(ExtendedMathMultiDTypeTest, LerpTensorWeight) {
    // lerp with tensor weight
    auto start_f32 = tenzor::zeros({3}, DType::Float32, Device::cpu());
    auto end_f32 = tenzor::full({3}, 10.0f, DType::Float32, Device::cpu());
    auto weight_f32 = tenzor::full({3}, 0.0f, DType::Float32, Device::cpu());
    auto* wd = weight_f32.data<float>();
    wd[0] = 0.0f; wd[1] = 0.5f; wd[2] = 1.0f;

    auto start = start_f32.to(device()).to(dtype());
    auto end_t = end_f32.to(device()).to(dtype());
    auto weight = weight_f32.to(device()).to(dtype());

    auto result = tenzor::lerp(start, end_t, weight);
    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    EXPECT_NEAR(r[0], 0.0f, atol());
    EXPECT_NEAR(r[1], 5.0f, atol());
    EXPECT_NEAR(r[2], 10.0f, atol());
}

// ============================================================================
// DType and Device Preservation
// ============================================================================

TEST_P(ExtendedMathMultiDTypeTest, UnaryOpsDTypePreservation) {
    auto input = createRandn({4});
    // Clamp to positive for log operations
    input = tenzor::abs(input);
    input = tenzor::add(input, tenzor::full({4}, 0.1f, dtype(), device()));

    auto r_log2 = tenzor::log2(input);
    auto r_log10 = tenzor::log10(input);
    auto r_log1p = tenzor::log1p(input);
    auto r_exp2 = tenzor::exp2(createRandn({4}));
    auto r_expm1 = tenzor::expm1(createRandn({4}));
    auto r_erf = tenzor::erf(createRandn({4}));
    auto r_erfc = tenzor::erfc(createRandn({4}));

    expectDType(r_log2);
    expectDType(r_log10);
    expectDType(r_log1p);
    expectDType(r_exp2);
    expectDType(r_expm1);
    expectDType(r_erf);
    expectDType(r_erfc);

    expectDevice(r_log2);
    expectDevice(r_exp2);
    expectDevice(r_erf);
}

TEST_P(ExtendedMathMultiDTypeTest, BinaryOpsDTypePreservation) {
    auto a = createRandn({4});
    auto b = createRandn({4});
    // Avoid division by zero for fmod/remainder
    b = tenzor::add(tenzor::abs(b), tenzor::full({4}, 1.0f, dtype(), device()));

    auto r_atan2 = tenzor::atan2(a, b);
    auto r_fmod = tenzor::fmod(a, b);
    auto r_remainder = tenzor::remainder(a, b);

    expectDType(r_atan2);
    expectDType(r_fmod);
    expectDType(r_remainder);
    expectDevice(r_atan2);
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ExtendedMathMultiDTypeTest);
