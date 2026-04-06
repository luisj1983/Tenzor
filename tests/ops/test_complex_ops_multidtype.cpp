/**
 * @file test_complex_ops_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for complex number operations
 *
 * Covers OpIds 440-449: Conj, Real, Imag, Angle, Polar
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

class ComplexOpsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    Tensor fromValues(const std::vector<float>& vals) {
        auto t = tenzor::full({static_cast<int64_t>(vals.size())},
                              0.0f, DType::Float32, Device::cpu());
        auto* d = t.data<float>();
        for (size_t i = 0; i < vals.size(); ++i) d[i] = vals[i];
        return t.to(device()).to(dtype());
    }

    std::vector<float> toFloats(const Tensor& t) {
        auto f32 = t.to(Device::cpu()).to(DType::Float32);
        auto* d = f32.data<float>();
        return std::vector<float>(d, d + f32.numel());
    }
};

// ============================================================================
// Conj Tests
// ============================================================================

TEST_P(ComplexOpsMultiDTypeTest, ConjRealTensor) {
    // Conjugate of a real tensor should be itself
    auto input = fromValues({1.0f, -2.0f, 3.5f});
    auto result = tenzor::conj(input);

    expectDevice(result);
    expectDType(result);

    auto in_vals = toFloats(input);
    auto out_vals = toFloats(result);
    for (size_t i = 0; i < in_vals.size(); ++i) {
        EXPECT_NEAR(out_vals[i], in_vals[i], atol()) << "Index " << i;
    }
}

// ============================================================================
// Real Tests
// ============================================================================

TEST_P(ComplexOpsMultiDTypeTest, RealOfRealTensor) {
    // real() on a real tensor returns identity
    auto input = fromValues({1.0f, -3.0f, 0.0f});
    auto result = tenzor::real(input);

    auto in_vals = toFloats(input);
    auto out_vals = toFloats(result);
    for (size_t i = 0; i < in_vals.size(); ++i) {
        EXPECT_NEAR(out_vals[i], in_vals[i], atol()) << "Index " << i;
    }
    expectDevice(result);
}

// ============================================================================
// Imag Tests
// ============================================================================

TEST_P(ComplexOpsMultiDTypeTest, ImagOfRealTensor) {
    // imag() on a real tensor returns zeros
    auto input = fromValues({1.0f, -3.0f, 5.0f});
    auto result = tenzor::imag(input);

    auto out_vals = toFloats(result);
    for (size_t i = 0; i < out_vals.size(); ++i) {
        EXPECT_NEAR(out_vals[i], 0.0f, atol()) << "Index " << i;
    }
    expectDevice(result);
}

// ============================================================================
// Angle Tests
// ============================================================================

TEST_P(ComplexOpsMultiDTypeTest, AngleOfPositiveReal) {
    // angle of positive real numbers = 0
    auto input = fromValues({1.0f, 2.0f, 5.0f});
    auto result = tenzor::angle(input);

    auto out_vals = toFloats(result);
    for (size_t i = 0; i < out_vals.size(); ++i) {
        EXPECT_NEAR(out_vals[i], 0.0f, atol()) << "Index " << i;
    }
}

TEST_P(ComplexOpsMultiDTypeTest, AngleOfNegativeReal) {
    // angle of negative real numbers = pi
    auto input = fromValues({-1.0f, -2.0f, -5.0f});
    auto result = tenzor::angle(input);

    auto out_vals = toFloats(result);
    for (size_t i = 0; i < out_vals.size(); ++i) {
        EXPECT_NEAR(out_vals[i], static_cast<float>(M_PI), std::max(atol(), 1e-3f))
            << "Index " << i;
    }
}

// ============================================================================
// Polar Tests
// ============================================================================

TEST_P(ComplexOpsMultiDTypeTest, PolarKnownValues) {
    // polar(abs, angle): abs * (cos(angle) + i*sin(angle))
    // For real result: polar(r, 0) = r (real part = r, imag = 0)
    auto abs_t = fromValues({1.0f, 2.0f, 3.0f});
    auto angle_t = fromValues({0.0f, 0.0f, 0.0f});

    auto result = tenzor::polar(abs_t, angle_t);
    // Real part should be abs values
    auto real_part = tenzor::real(result);
    auto out_vals = toFloats(real_part);
    EXPECT_NEAR(out_vals[0], 1.0f, atol());
    EXPECT_NEAR(out_vals[1], 2.0f, atol());
    EXPECT_NEAR(out_vals[2], 3.0f, atol());
}

// ============================================================================
// Shape and DType Preservation
// ============================================================================

TEST_P(ComplexOpsMultiDTypeTest, ShapePreservation) {
    auto input = createRandn({3, 4});
    auto conj_r = tenzor::conj(input);
    auto real_r = tenzor::real(input);
    auto imag_r = tenzor::imag(input);
    auto angle_r = tenzor::angle(input);

    expectShape(conj_r, {3, 4});
    expectShape(real_r, {3, 4});
    expectShape(imag_r, {3, 4});
    expectShape(angle_r, {3, 4});
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ComplexOpsMultiDTypeTest);
