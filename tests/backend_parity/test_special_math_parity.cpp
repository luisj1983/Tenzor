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
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Gamma Family
// ============================================================================

TEST(SpecialMathParity, Gamma) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Positive input required for gamma function
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 100);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return gamma(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "Gamma");
}

TEST(SpecialMathParity, Lgamma) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 101);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return lgamma(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "Lgamma");
}

TEST(SpecialMathParity, Digamma) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Positive input, avoid 0 and negative integers where digamma has poles
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 102);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return digamma(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "Digamma");
}

// ============================================================================
// Error Function Family
// ============================================================================

TEST(SpecialMathParity, Erfinv) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Input must be in (-1, 1); use (-0.9, 0.9) to avoid boundary instability
    auto a = generate_uniform_tensor({16, 16}, -0.9f, 0.9f, DType::Float32, Device::cpu(), 103);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return erfinv(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "Erfinv");
}

// ============================================================================
// Sinc
// ============================================================================

TEST(SpecialMathParity, Sinc) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return sinc(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "Sinc");
}

// ============================================================================
// Bessel Functions
// ============================================================================

TEST(SpecialMathParity, BesselJ0) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({16, 16}, 0.1f, 10.0f, DType::Float32, Device::cpu(), 104);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return bessel_j0(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "BesselJ0");
}

TEST(SpecialMathParity, BesselJ1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({16, 16}, 0.1f, 10.0f, DType::Float32, Device::cpu(), 105);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return bessel_j1(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "BesselJ1");
}

TEST(SpecialMathParity, BesselI0) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({16, 16}, -3.0f, 3.0f, DType::Float32, Device::cpu(), 106);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return bessel_i0(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "BesselI0");
}

TEST(SpecialMathParity, BesselI1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({16, 16}, -3.0f, 3.0f, DType::Float32, Device::cpu(), 107);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return bessel_i1(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "BesselI1");
}

// ============================================================================
// Composition / Cross-Validation Tests
// ============================================================================

TEST(SpecialMathParity, Erf_Composition) {
    // Verify erf(erfinv(x)) ~= x for x in (-0.9, 0.9) — functional identity test
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = generate_uniform_tensor({16, 16}, -0.9f, 0.9f, DType::Float32, Device::cpu(), 108);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return erf(erfinv(inputs[0]));
    }, {x}, backends, 1e-3f, 1e-4f, "Erf_Composition");

    // Also verify the identity holds on CPU: erf(erfinv(x)) should be close to x
    auto roundtrip = erf(erfinv(x));
    EXPECT_TENSORS_CLOSE(roundtrip, x, 1e-3f, 1e-4f);
}

TEST(SpecialMathParity, Lgamma_vs_Gamma) {
    // Cross-validation: lgamma(x) ~= log(gamma(x)) for positive x
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 109);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return lgamma(inputs[0]);
    }, {x}, backends, 1e-3f, 1e-4f, "Lgamma_vs_Gamma_parity");

    // Verify identity on CPU: lgamma(x) ~= log(gamma(x))
    auto lgamma_result = lgamma(x);
    auto log_gamma_result = log(gamma(x));
    EXPECT_TENSORS_CLOSE(lgamma_result, log_gamma_result, 1e-3f, 1e-4f);
}

// ============================================================================
// Scaled Bessel & Normal CDF
// ============================================================================

TEST(SpecialMathParity, I0e) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({16, 16}, -5.0f, 5.0f, DType::Float32, Device::cpu(), 110);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return i0e(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "I0e");
}

TEST(SpecialMathParity, Ndtr) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return ndtr(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "Ndtr");
}

TEST(SpecialMathParity, LogNdtr) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return log_ndtr(inputs[0]);
    }, {a}, backends, 1e-3f, 1e-4f, "LogNdtr");
}

// ============================================================================
// Binary Special Functions
// ============================================================================

TEST(SpecialMathParity, Igamma) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Both inputs must be positive for igamma
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 111);
    auto x = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 112);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return igamma(inputs[0], inputs[1]);
    }, {a, x}, backends, 1e-3f, 1e-4f, "Igamma");
}

TEST(SpecialMathParity, Beta) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Both inputs must be positive for beta function
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 113);
    auto b = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu(), 114);

    test_operation_parity_backends([](const std::vector<Tensor>& inputs) {
        return beta(inputs[0], inputs[1]);
    }, {a, b}, backends, 1e-3f, 1e-4f, "Beta");
}



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
