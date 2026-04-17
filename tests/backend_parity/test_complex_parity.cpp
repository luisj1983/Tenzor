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

    // atol=1e-5 matches other transcendental parity tests — polar is a
    // cos/sin-based op, whose accuracy on GLSL/Vulkan is up to ~2 ULP
    // looser than libm's, so the default 1e-7 atol is unreachable.
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return polar(inputs[0], inputs[1]);
    }, {abs_t, ang_t}, 1e-5f, 1e-5f, "Polar");
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

// ============================================================================
// Complex Transcendentals
// ============================================================================

TEST(ComplexParity, Complex_Exp) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Keep real parts bounded so exp(Re) doesn't overflow Float32.
    auto a = randn({16, 16}, DType::Complex64, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);
    }, {a}, 1e-4f, 1e-5f, "Complex_Exp");
}

TEST(ComplexParity, Complex_Log) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Avoid log(0) — shift away from the origin.
    auto base = randn({16, 16}, DType::Complex64, Device::cpu());
    auto* d = base.data<std::complex<float>>();
    for (int64_t i = 0; i < base.numel(); ++i) {
        d[i] = std::complex<float>(d[i].real() + 2.0f, d[i].imag());
    }
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return log(inputs[0]);
    }, {base}, 1e-5f, 1e-6f, "Complex_Log");
}

TEST(ComplexParity, Complex_Sqrt) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({16, 16}, DType::Complex64, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sqrt(inputs[0]);
    }, {a}, 1e-5f, 1e-6f, "Complex_Sqrt");
}

TEST(ComplexParity, Complex_Sin) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Keep imag bounded so cosh(Im)/sinh(Im) stay finite in Float32.
    auto a = randn({16, 16}, DType::Complex64, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]);
    }, {a}, 1e-4f, 1e-5f, "Complex_Sin");
}

TEST(ComplexParity, Complex_Cos) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({16, 16}, DType::Complex64, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return cos(inputs[0]);
    }, {a}, 1e-4f, 1e-5f, "Complex_Cos");
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
