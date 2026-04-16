/**
 * @file test_trig_parity.cpp
 * @brief Backend parity tests for trigonometric operations
 *
 * Tests 12 trigonometric operations (sin, cos, tan, asin, acos, atan,
 * sinh, cosh, asinh, acosh, atanh, and the Pythagorean identity) to
 * ensure all backends (CPU, CUDA, OneAPI, Vulkan) produce identical results.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Inverse Trigonometric Operations
// ============================================================================

TEST(TrigOperationParity, Asin) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -0.99f, 0.99f, DType::Float32, Device::cpu(), 42);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return asin(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Asin");
}

TEST(TrigOperationParity, Acos) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -0.99f, 0.99f, DType::Float32, Device::cpu(), 43);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return acos(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Acos");
}

TEST(TrigOperationParity, Atan) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return atan(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Atan");
}

// ============================================================================
// Hyperbolic Operations
// ============================================================================

TEST(TrigOperationParity, Sinh) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -3.0f, 3.0f, DType::Float32, Device::cpu(), 44);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sinh(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Sinh");
}

TEST(TrigOperationParity, Cosh) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -3.0f, 3.0f, DType::Float32, Device::cpu(), 44);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return cosh(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Cosh");
}

// ============================================================================
// Inverse Hyperbolic Operations
// ============================================================================

TEST(TrigOperationParity, Asinh) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return asinh(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Asinh");
}

TEST(TrigOperationParity, Acosh) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, 1.01f, 10.0f, DType::Float32, Device::cpu(), 45);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return acosh(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Acosh");
}

TEST(TrigOperationParity, Atanh) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -0.99f, 0.99f, DType::Float32, Device::cpu(), 46);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return atanh(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Atanh");
}

// ============================================================================
// Trigonometric Operations with Large Values
// ============================================================================

TEST(TrigOperationParity, SinLargeValues) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -100.0f, 100.0f, DType::Float32, Device::cpu(), 47);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]);
    }, {a}, 1e-4f, 1e-5f, "SinLargeValues");
}

TEST(TrigOperationParity, CosLargeValues) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -100.0f, 100.0f, DType::Float32, Device::cpu(), 47);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return cos(inputs[0]);
    }, {a}, 1e-4f, 1e-5f, "CosLargeValues");
}

TEST(TrigOperationParity, Tan) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -1.5f, 1.5f, DType::Float32, Device::cpu(), 48);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tan(inputs[0]);
    }, {a}, 1e-4f, 1e-5f, "Tan");
}

// ============================================================================
// Pythagorean Identity: sin^2(x) + cos^2(x) = 1
// ============================================================================

TEST(TrigOperationParity, PythagoreanIdentity) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -100.0f, 100.0f, DType::Float32, Device::cpu(), 49);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]) * sin(inputs[0]) + cos(inputs[0]) * cos(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "PythagoreanIdentity");
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
