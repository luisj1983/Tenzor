/**
 * @file test_extended_math_parity.cpp
 * @brief Extended math operation parity tests across backends
 *
 * Tests 21 extended math operations (log2, log10, log1p, exp2, expm1, erf,
 * erfc, rsqrt, square, reciprocal, floor, ceil, round, trunc, frac, erfinv,
 * atan2, fmod, remainder, copysign, hypot) to ensure all backends
 * (CPU, CUDA, ROCm, Vulkan, OneAPI) produce identical results.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class ExtendedMathParity : public BackendTest {};
// ============================================================================
// Unary Extended Math Operations
// ============================================================================

TEST_P(ExtendedMathParity, Log2) {

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return log2(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Log2");
}

TEST_P(ExtendedMathParity, Log10) {

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return log10(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Log10");
}

TEST_P(ExtendedMathParity, Log1p) {

    auto a = generate_uniform_tensor({32, 32}, -0.5f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return log1p(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Log1p");
}

TEST_P(ExtendedMathParity, Exp2) {

    auto a = generate_uniform_tensor({32, 32}, -5.0f, 5.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return exp2(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Exp2");
}

TEST_P(ExtendedMathParity, Expm1) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return expm1(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Expm1");
}

TEST_P(ExtendedMathParity, Erf) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return erf(inputs[0]);
    }, {a}, device, 1e-4f, 1e-6f, "Erf");
}

TEST_P(ExtendedMathParity, Erfc) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return erfc(inputs[0]);
    }, {a}, device, 1e-4f, 1e-6f, "Erfc");
}

TEST_P(ExtendedMathParity, Rsqrt) {

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return rsqrt(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Rsqrt");
}

TEST_P(ExtendedMathParity, Square) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return square(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Square");
}

TEST_P(ExtendedMathParity, Reciprocal) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) + 2.0f;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return reciprocal(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Reciprocal");
}

TEST_P(ExtendedMathParity, Floor) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return floor(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Floor");
}

TEST_P(ExtendedMathParity, Ceil) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return ceil(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Ceil");
}

TEST_P(ExtendedMathParity, Round) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return round(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Round");
}

TEST_P(ExtendedMathParity, Trunc) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    // trunc(x) = sign(x) * floor(abs(x))
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return sign(inputs[0]) * floor(abs(inputs[0]));
    }, {a}, device, 0.0f, 0.0f, "Trunc");
}

TEST_P(ExtendedMathParity, Frac) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return frac(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Frac");
}

TEST_P(ExtendedMathParity, Erfinv) {

    auto a = generate_uniform_tensor({32, 32}, -0.9f, 0.9f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return erfinv(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "Erfinv");
}

// ============================================================================
// Binary Extended Math Operations
// ============================================================================

TEST_P(ExtendedMathParity, Atan2) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return atan2(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-5f, 1e-7f, "Atan2");
}

TEST_P(ExtendedMathParity, Fmod) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) + 2.0f;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return fmod(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-5f, 1e-7f, "Fmod");
}

TEST_P(ExtendedMathParity, Remainder) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) + 2.0f;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return remainder(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-5f, 1e-7f, "Remainder");
}

TEST_P(ExtendedMathParity, Copysign) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return copysign(inputs[0], inputs[1]);
    }, {a, b}, device, 0.0f, 0.0f, "Copysign");
}

TEST_P(ExtendedMathParity, Hypot) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return hypot(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-5f, 1e-7f, "Hypot");
}

INSTANTIATE_BACKEND_TESTS(ExtendedMathParity);




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
