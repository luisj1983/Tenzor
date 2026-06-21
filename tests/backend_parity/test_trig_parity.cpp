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
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class TrigOperationParity : public BackendTest {};
// ============================================================================
// Inverse Trigonometric Operations
// ============================================================================

TEST_P(TrigOperationParity, Asin) {

    auto a = generate_uniform_tensor({32, 32}, -0.99f, 0.99f, DType::Float32, Device::cpu(), 42);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return asin(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Asin");
}

TEST_P(TrigOperationParity, Acos) {

    auto a = generate_uniform_tensor({32, 32}, -0.99f, 0.99f, DType::Float32, Device::cpu(), 43);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return acos(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Acos");
}

TEST_P(TrigOperationParity, Atan) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return atan(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Atan");
}

// ============================================================================
// Hyperbolic Operations
// ============================================================================

TEST_P(TrigOperationParity, Sinh) {

    auto a = generate_uniform_tensor({32, 32}, -3.0f, 3.0f, DType::Float32, Device::cpu(), 44);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sinh(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Sinh");
}

TEST_P(TrigOperationParity, Cosh) {

    auto a = generate_uniform_tensor({32, 32}, -3.0f, 3.0f, DType::Float32, Device::cpu(), 44);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return cosh(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Cosh");
}

// ============================================================================
// Inverse Hyperbolic Operations
// ============================================================================

TEST_P(TrigOperationParity, Asinh) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return asinh(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Asinh");
}

TEST_P(TrigOperationParity, Acosh) {

    auto a = generate_uniform_tensor({32, 32}, 1.01f, 10.0f, DType::Float32, Device::cpu(), 45);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return acosh(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Acosh");
}

TEST_P(TrigOperationParity, Atanh) {

    auto a = generate_uniform_tensor({32, 32}, -0.99f, 0.99f, DType::Float32, Device::cpu(), 46);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return atanh(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Atanh");
}

// ============================================================================
// Trigonometric Operations with Large Values
// ============================================================================

TEST_P(TrigOperationParity, SinLargeValues) {

    auto a = generate_uniform_tensor({32, 32}, -100.0f, 100.0f, DType::Float32, Device::cpu(), 47);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]);
    }, {a}, device, 1e-4f, 1e-5f, "SinLargeValues");
}

TEST_P(TrigOperationParity, CosLargeValues) {

    auto a = generate_uniform_tensor({32, 32}, -100.0f, 100.0f, DType::Float32, Device::cpu(), 47);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return cos(inputs[0]);
    }, {a}, device, 1e-4f, 1e-5f, "CosLargeValues");
}

TEST_P(TrigOperationParity, Tan) {

    auto a = generate_uniform_tensor({32, 32}, -1.5f, 1.5f, DType::Float32, Device::cpu(), 48);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tan(inputs[0]);
    }, {a}, device, 1e-4f, 1e-5f, "Tan");
}

// ============================================================================
// Pythagorean Identity: sin^2(x) + cos^2(x) = 1
// ============================================================================

TEST_P(TrigOperationParity, PythagoreanIdentity) {

    // Keep arguments in a range where Float32 sin/cos are accurate so the
    // identity sin^2+cos^2 == 1 holds tightly; the point is to compare to the
    // KNOWN CONSTANT 1.0, not to a (possibly equally-wrong) CPU self-reference.
    auto a = generate_uniform_tensor({32, 32}, -6.28f, 6.28f, DType::Float32, Device::cpu(), 49);

    auto identity = [](const std::vector<Tensor>& inputs) {
        return sin(inputs[0]) * sin(inputs[0]) + cos(inputs[0]) * cos(inputs[0]);
    };

    // Direct correctness check against the constant 1.0 on the target backend.
    // A backend consistently returning e.g. 0.97 (which would still match a CPU
    // self-reference) fails here.
    auto target_input = a.to(device);
    auto target_id = identity({target_input}).cpu();
    auto ones_ref = full({32, 32}, 1.0, DType::Float32, Device::cpu());
    EXPECT_TRUE(tensors_close(target_id, ones_ref, 1e-4f, 1e-4f))
        << "sin^2+cos^2 deviates from 1.0 on " << backend_name(device);

    test_operation_parity_single(identity, {a}, device, 1e-5f, 1e-7f, "PythagoreanIdentity");
}

// Phase 6-followup #27: gradient parity for trig/hyperbolic functions.
TEST_P(TrigOperationParity, Atan_GradientParity) {
    auto a = randn({16, 16}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return atan(in[0]); },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Atan_Grad");
}

TEST_P(TrigOperationParity, Sinh_GradientParity) {
    auto a = generate_uniform_tensor({16, 16}, -2.0f, 2.0f, DType::Float32, Device::cpu(), 44);
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return sinh(in[0]); },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Sinh_Grad");
}

TEST_P(TrigOperationParity, Cosh_GradientParity) {
    auto a = generate_uniform_tensor({16, 16}, -2.0f, 2.0f, DType::Float32, Device::cpu(), 45);
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return cosh(in[0]); },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Cosh_Grad");
}

// Atanh has no Variable overload — gradient parity not testable through
// the Variable interface today. (Tensor-only.)

INSTANTIATE_BACKEND_TESTS(TrigOperationParity);




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
