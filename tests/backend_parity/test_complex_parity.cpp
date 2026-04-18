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
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class ComplexParity : public BackendTest {};
// ============================================================================
// Complex Unary Operations
// ============================================================================

TEST_P(ComplexParity, Conj) {

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return conj(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Conj");
}

TEST_P(ComplexParity, Real) {

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return real(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Real");
}

TEST_P(ComplexParity, Imag) {

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return imag(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Imag");
}

TEST_P(ComplexParity, Angle) {

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return angle(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Angle");
}

// ============================================================================
// Polar Construction
// ============================================================================

TEST_P(ComplexParity, Polar) {

    // Magnitude in [0.1, 5.0], angle in [-pi, pi]
    auto abs_t = generate_uniform_tensor({32, 32}, 0.1f, 5.0f, DType::Float32, Device::cpu());
    auto ang_t = generate_uniform_tensor({32, 32}, -3.14f, 3.14f, DType::Float32, Device::cpu(), 99999);

    // atol=1e-5 matches other transcendental parity tests — polar is a
    // cos/sin-based op, whose accuracy on GLSL/Vulkan is up to ~2 ULP
    // looser than libm's, so the default 1e-7 atol is unreachable.
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return polar(inputs[0], inputs[1]);
    }, {abs_t, ang_t}, device, 1e-5f, 1e-5f, "Polar");
}

// ============================================================================
// Complex Arithmetic
// ============================================================================

TEST_P(ComplexParity, Complex_Add) {

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());
    auto b = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, device, 0.0f, 0.0f, "Complex_Add");
}

TEST_P(ComplexParity, Complex_Mul) {

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());
    auto b = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, device, 1e-5f, 1e-7f, "Complex_Mul");
}

TEST_P(ComplexParity, Complex_Abs) {

    auto a = randn({32, 32}, DType::Complex64, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return abs(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Complex_Abs");
}

// ============================================================================
// Complex Transcendentals
// ============================================================================

TEST_P(ComplexParity, Complex_Exp) {

    // Keep real parts bounded so exp(Re) doesn't overflow Float32.
    auto a = randn({16, 16}, DType::Complex64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);
    }, {a}, device, 1e-4f, 1e-5f, "Complex_Exp");
}

TEST_P(ComplexParity, Complex_Log) {

    // Avoid log(0) — shift away from the origin.
    auto base = randn({16, 16}, DType::Complex64, Device::cpu());
    auto* d = base.data<std::complex<float>>();
    for (int64_t i = 0; i < base.numel(); ++i) {
        d[i] = std::complex<float>(d[i].real() + 2.0f, d[i].imag());
    }
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return log(inputs[0]);
    }, {base}, device, 1e-5f, 1e-6f, "Complex_Log");
}

TEST_P(ComplexParity, Complex_Sqrt) {

    auto a = randn({16, 16}, DType::Complex64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sqrt(inputs[0]);
    }, {a}, device, 1e-5f, 1e-6f, "Complex_Sqrt");
}

TEST_P(ComplexParity, Complex_Sin) {

    // Keep imag bounded so cosh(Im)/sinh(Im) stay finite in Float32.
    auto a = randn({16, 16}, DType::Complex64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]);
    }, {a}, device, 1e-4f, 1e-5f, "Complex_Sin");
}

TEST_P(ComplexParity, Complex_Cos) {

    auto a = randn({16, 16}, DType::Complex64, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return cos(inputs[0]);
    }, {a}, device, 1e-4f, 1e-5f, "Complex_Cos");
}

INSTANTIATE_BACKEND_TESTS(ComplexParity);




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
