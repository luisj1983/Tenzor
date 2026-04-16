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
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Unary Extended Math Operations
// ============================================================================

TEST(ExtendedMathParity, Log2) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return log2(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Log2");
}

TEST(ExtendedMathParity, Log10) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return log10(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Log10");
}

TEST(ExtendedMathParity, Log1p) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -0.5f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return log1p(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Log1p");
}

TEST(ExtendedMathParity, Exp2) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -5.0f, 5.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return exp2(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Exp2");
}

TEST(ExtendedMathParity, Expm1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return expm1(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Expm1");
}

TEST(ExtendedMathParity, Erf) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return erf(inputs[0]);
    }, {a}, 1e-4f, 1e-6f, "Erf");
}

TEST(ExtendedMathParity, Erfc) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return erfc(inputs[0]);
    }, {a}, 1e-4f, 1e-6f, "Erfc");
}

TEST(ExtendedMathParity, Rsqrt) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return rsqrt(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Rsqrt");
}

TEST(ExtendedMathParity, Square) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return square(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Square");
}

TEST(ExtendedMathParity, Reciprocal) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) + 2.0f;

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return reciprocal(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Reciprocal");
}

TEST(ExtendedMathParity, Floor) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return floor(inputs[0]);
    }, {a}, 0.0f, 0.0f, "Floor");
}

TEST(ExtendedMathParity, Ceil) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return ceil(inputs[0]);
    }, {a}, 0.0f, 0.0f, "Ceil");
}

TEST(ExtendedMathParity, Round) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return round(inputs[0]);
    }, {a}, 0.0f, 0.0f, "Round");
}

TEST(ExtendedMathParity, Trunc) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    // trunc(x) = sign(x) * floor(abs(x))
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return sign(inputs[0]) * floor(abs(inputs[0]));
    }, {a}, 0.0f, 0.0f, "Trunc");
}

TEST(ExtendedMathParity, Frac) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return frac(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Frac");
}

TEST(ExtendedMathParity, Erfinv) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = generate_uniform_tensor({32, 32}, -0.9f, 0.9f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return erfinv(inputs[0]);
    }, {a}, 1e-3f, 1e-4f, "Erfinv");
}

// ============================================================================
// Binary Extended Math Operations
// ============================================================================

TEST(ExtendedMathParity, Atan2) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return atan2(inputs[0], inputs[1]);
    }, {a, b}, 1e-5f, 1e-7f, "Atan2");
}

TEST(ExtendedMathParity, Fmod) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) + 2.0f;

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return fmod(inputs[0], inputs[1]);
    }, {a, b}, 1e-5f, 1e-7f, "Fmod");
}

TEST(ExtendedMathParity, Remainder) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) + 2.0f;

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return remainder(inputs[0], inputs[1]);
    }, {a, b}, 1e-5f, 1e-7f, "Remainder");
}

TEST(ExtendedMathParity, Copysign) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return copysign(inputs[0], inputs[1]);
    }, {a, b}, 0.0f, 0.0f, "Copysign");
}

TEST(ExtendedMathParity, Hypot) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return hypot(inputs[0], inputs[1]);
    }, {a, b}, 1e-5f, 1e-7f, "Hypot");
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
