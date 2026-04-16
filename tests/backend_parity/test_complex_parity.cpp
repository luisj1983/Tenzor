/**
 * @file test_complex_parity.cpp
 * @brief Complex number operation parity tests across backends
 *
 * Tests conj, real, imag, angle, polar, complex arithmetic, and abs
 * operations on Complex64 tensors to ensure all backends (CPU, CUDA,
 * ROCm, Vulkan, OneAPI) produce identical results.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Complex Unary Operations
// ============================================================================

TEST(ComplexParity, Conj) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return conj(inputs[0]);
    }, {a}, 0.0f, 0.0f, "Conj");
}

TEST(ComplexParity, Real) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return real(inputs[0]);
    }, {a}, 0.0f, 0.0f, "Real");
}

TEST(ComplexParity, Imag) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return imag(inputs[0]);
    }, {a}, 0.0f, 0.0f, "Imag");
}

TEST(ComplexParity, Angle) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return angle(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Angle");
}

// ============================================================================
// Polar Construction
// ============================================================================

TEST(ComplexParity, Polar) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Magnitude in [0.1, 5.0], angle in [-pi, pi]
    auto abs_t = generate_uniform_tensor({32, 32}, 0.1f, 5.0f, DType::Float32, Device::cpu());
    auto ang_t = generate_uniform_tensor({32, 32}, -3.14f, 3.14f, DType::Float32, Device::cpu(), 99999);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return polar(inputs[0], inputs[1]);
    }, {abs_t, ang_t}, 1e-5f, 1e-7f, "Polar");
}

// ============================================================================
// Complex Arithmetic
// ============================================================================

TEST(ComplexParity, Complex_Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());
    auto b = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 0.0f, 0.0f, "Complex_Add");
}

TEST(ComplexParity, Complex_Mul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());
    auto b = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 1e-5f, 1e-7f, "Complex_Mul");
}

TEST(ComplexParity, Complex_Abs) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return abs(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Complex_Abs");
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
