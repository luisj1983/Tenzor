/**
 * @file test_special_math_parity.cpp
 * @brief Special math function parity tests across backends
 *
 * Tests 16 special math operations (gamma, lgamma, digamma, erfinv, sinc,
 * bessel_j0, bessel_j1, bessel_i0, bessel_i1, i0e, ndtr, log_ndtr, igamma,
 * beta, and composition/cross-validation tests) to ensure all backends
 * (CPU, CUDA, ROCm, Vulkan, OneAPI) produce identical results.
 *
 * These operations have known kernel gaps on some backends (ROCm -13,
 * Vulkan -15, OneAPI -22), so test_operation_parity_backends gracefully
 * skips backends that throw exceptions.
 *
 * Wider tolerances (rtol=1e-3, atol=1e-4) are used throughout because
 * special functions have larger numerical variation across implementations.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class SpecialMathParity : public BackendTest {};
// ============================================================================
// Gamma Family
// ============================================================================

TEST_P(SpecialMathParity, Gamma) {

    // Positive input required for gamma function
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 100);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return gamma(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "Gamma");
}

TEST_P(SpecialMathParity, Lgamma) {

    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 101);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return lgamma(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "Lgamma");
}

TEST_P(SpecialMathParity, Digamma) {

    // Positive input, avoid 0 and negative integers where digamma has poles
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 102);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return digamma(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "Digamma");
}

// ============================================================================
// Error Function Family
// ============================================================================

TEST_P(SpecialMathParity, Erfinv) {

    // Input must be in (-1, 1); use (-0.9, 0.9) to avoid boundary instability
    auto a = generate_uniform_tensor({16, 16}, -0.9f, 0.9f, DType::Float32, Device::cpu(), 103);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return erfinv(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "Erfinv");
}

// ============================================================================
// Sinc
// ============================================================================

TEST_P(SpecialMathParity, Sinc) {

    auto a = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sinc(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "Sinc");
}

// ============================================================================
// Bessel Functions
// ============================================================================

TEST_P(SpecialMathParity, BesselJ0) {

    auto a = generate_uniform_tensor({16, 16}, 0.1f, 10.0f, DType::Float32, Device::cpu(), 104);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bessel_j0(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "BesselJ0");
}

TEST_P(SpecialMathParity, BesselJ1) {

    auto a = generate_uniform_tensor({16, 16}, 0.1f, 10.0f, DType::Float32, Device::cpu(), 105);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bessel_j1(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "BesselJ1");
}

TEST_P(SpecialMathParity, BesselI0) {

    auto a = generate_uniform_tensor({16, 16}, -3.0f, 3.0f, DType::Float32, Device::cpu(), 106);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bessel_i0(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "BesselI0");
}

TEST_P(SpecialMathParity, BesselI1) {

    auto a = generate_uniform_tensor({16, 16}, -3.0f, 3.0f, DType::Float32, Device::cpu(), 107);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return bessel_i1(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "BesselI1");
}

// ============================================================================
// Composition / Cross-Validation Tests
// ============================================================================

TEST_P(SpecialMathParity, Erf_Composition) {
    // Verify erf(erfinv(x)) ~= x for x in (-0.9, 0.9) — functional identity test

    auto x = generate_uniform_tensor({16, 16}, -0.9f, 0.9f, DType::Float32, Device::cpu(), 108);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return erf(erfinv(inputs[0]));
    }, {x}, device, 1e-3f, 1e-4f, "Erf_Composition");

    // Also verify the identity holds on CPU: erf(erfinv(x)) should be close to x
    auto roundtrip = erf(erfinv(x));
    EXPECT_TENSORS_CLOSE(roundtrip, x, 1e-3f, 1e-4f);
}

TEST_P(SpecialMathParity, Lgamma_vs_Gamma) {
    // Cross-validation: lgamma(x) ~= log(gamma(x)) for positive x

    auto x = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 109);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return lgamma(inputs[0]);
    }, {x}, device, 1e-3f, 1e-4f, "Lgamma_vs_Gamma_parity");

    // Verify identity on CPU: lgamma(x) ~= log(gamma(x))
    auto lgamma_result = lgamma(x);
    auto log_gamma_result = log(gamma(x));
    EXPECT_TENSORS_CLOSE(lgamma_result, log_gamma_result, 1e-3f, 1e-4f);
}

// ============================================================================
// Scaled Bessel & Normal CDF
// ============================================================================

TEST_P(SpecialMathParity, I0e) {

    auto a = generate_uniform_tensor({16, 16}, -5.0f, 5.0f, DType::Float32, Device::cpu(), 110);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return i0e(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "I0e");
}

TEST_P(SpecialMathParity, Ndtr) {

    auto a = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return ndtr(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "Ndtr");
}

TEST_P(SpecialMathParity, LogNdtr) {

    auto a = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return log_ndtr(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "LogNdtr");
}

// ============================================================================
// Binary Special Functions
// ============================================================================

TEST_P(SpecialMathParity, Igamma) {

    // Both inputs must be positive for igamma
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 111);
    auto x = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 112);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return igamma(inputs[0], inputs[1]);
    }, {a, x}, device, 1e-3f, 1e-4f, "Igamma");
}

TEST_P(SpecialMathParity, Beta) {

    // Both inputs must be positive for beta function
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 113);
    auto b = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 114);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return beta(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-3f, 1e-4f, "Beta");
}

// ============================================================================
// Phase 3.7 additions: BesselY0, BesselY1, Zeta, I1e, LogAddExp, XLog1Py,
// SphericalBesselJ0
// ============================================================================

TEST_P(SpecialMathParity, BesselY0) {
    // bessel_y0 is singular at x=0 — use strictly positive input.
    auto input = rand({4, 16}, DType::Float32, Device::cpu()) + 0.1f;
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::bessel_y0(ins[0]);
    }, {input}, device, 1e-3f, 1e-4f, "bessel_y0");
}

TEST_P(SpecialMathParity, BesselY1) {
    auto input = rand({4, 16}, DType::Float32, Device::cpu()) + 0.1f;
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::bessel_y1(ins[0]);
    }, {input}, device, 1e-3f, 1e-4f, "bessel_y1");
}

TEST_P(SpecialMathParity, Zeta) {
    // zeta(x, q) is real for x > 1. Use x in [2, 4], q > 0.
    auto x = rand({4, 8}, DType::Float32, Device::cpu()) * 2.0f + 2.0f;
    auto q = rand({4, 8}, DType::Float32, Device::cpu()) + 0.5f;
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::zeta(ins[0], ins[1]);
    }, {x, q}, device, 1e-3f, 1e-4f, "zeta");
}

TEST_P(SpecialMathParity, I1e) {
    auto input = randn({4, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::i1e(ins[0]);
    }, {input}, device, 1e-3f, 1e-4f, "i1e");
}

TEST_P(SpecialMathParity, LogAddExp) {
    auto a = randn({4, 16}, DType::Float32, Device::cpu());
    auto b = randn({4, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::logaddexp(ins[0], ins[1]);
    }, {a, b}, device, 1e-4f, 1e-6f, "logaddexp");
}

TEST_P(SpecialMathParity, XLog1Py) {
    auto x = randn({4, 16}, DType::Float32, Device::cpu());
    // y > -1 for log1p(y) to be finite; sample in [0, 1].
    auto y = rand({4, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::xlog1py(ins[0], ins[1]);
    }, {x, y}, device, 1e-4f, 1e-6f, "xlog1py");
}

TEST_P(SpecialMathParity, SphericalBesselJ0) {
    auto input = rand({4, 16}, DType::Float32, Device::cpu()) + 0.1f;
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::spherical_bessel_j0(ins[0]);
    }, {input}, device, 1e-3f, 1e-4f, "spherical_bessel_j0");
}

TEST_P(SpecialMathParity, BetaInc) {
    // Incomplete beta I_x(a, b). Requires a > 0, b > 0, x ∈ [0, 1].
    auto a = rand({4, 8}, DType::Float32, Device::cpu()) + 0.5f;
    auto b = rand({4, 8}, DType::Float32, Device::cpu()) + 0.5f;
    auto x = rand({4, 8}, DType::Float32, Device::cpu());  // in [0,1)
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::betainc(ins[0], ins[1], ins[2]);
    }, {a, b, x}, device, 1e-3f, 1e-4f, "betainc");
}

INSTANTIATE_BACKEND_TESTS(SpecialMathParity);


int main(int argc, char** argv) {
    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    try {
        tenzor::finalize();
    } catch (...) {}
    return result;
}
