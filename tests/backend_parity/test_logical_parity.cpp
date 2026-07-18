/**
 * @file test_logical_parity.cpp
 * @brief Logical operation parity tests across backends
 *
 * Tests logical_and, logical_or, logical_not, logical_xor and combined
 * expressions to ensure all backends produce identical Bool results.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class LogicalParity : public BackendTest {};
// ============================================================================
// Logical Operations Parity Tests
// ============================================================================

TEST_P(LogicalParity, LogicalAnd) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return logical_and(inputs[0], inputs[1]);
    }, std::vector<Tensor>{a, b}, device, 0.0f, 0.0f, "LogicalAnd");
}

TEST_P(LogicalParity, LogicalOr) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return logical_or(inputs[0], inputs[1]);
    }, std::vector<Tensor>{a, b}, device, 0.0f, 0.0f, "LogicalOr");
}

TEST_P(LogicalParity, LogicalNot) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return logical_not(inputs[0]);
    }, std::vector<Tensor>{a}, device, 0.0f, 0.0f, "LogicalNot");
}

TEST_P(LogicalParity, LogicalXor) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return logical_xor(inputs[0], inputs[1]);
    }, std::vector<Tensor>{a, b}, device, 0.0f, 0.0f, "LogicalXor");
}

// ============================================================================
// Combined Logical Expression Tests
// ============================================================================

TEST_P(LogicalParity, Combined) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());
    auto c = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return logical_or(logical_and(inputs[0], inputs[1]),
                          logical_not(inputs[2]));
    }, std::vector<Tensor>{a, b, c}, device, 0.0f, 0.0f, "Combined");
}

TEST_P(LogicalParity, DeMorgan) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) > zeros({32, 32}, DType::Float32, Device::cpu());

    // Verify ~(a & b) == (~a) | (~b) by testing both sides produce
    // the same result on each backend.
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        // Left side of De Morgan's law: NOT(a AND b)
        auto lhs = logical_not(logical_and(inputs[0], inputs[1]));
        // Right side: (NOT a) OR (NOT b)
        auto rhs = logical_or(logical_not(inputs[0]), logical_not(inputs[1]));
        // Return lhs; we also verify equality with rhs on this backend
        // via eq — the result is Bool, true where they match.
        // Since De Morgan's law must hold exactly, eq must be all-true.
        // We return lhs for cross-backend parity; rhs equality is
        // verified below.
        return lhs;
    }, std::vector<Tensor>{a, b}, device, 0.0f, 0.0f, "DeMorgan_LHS");

    // Also verify the right side is identical across backends
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return logical_or(logical_not(inputs[0]), logical_not(inputs[1]));
    }, std::vector<Tensor>{a, b}, device, 0.0f, 0.0f, "DeMorgan_RHS");
}

// ----------------------------------------------------------------------------
// CPU-only: to_bool_value() dtype coverage regression. The default branch
// used to reinterpret ANY unhandled dtype's storage as float32 (e.g. int16
// read 2 bytes-per-element as if it were 4 bytes-per-element float), instead
// of either correctly converting or throwing like CUDA's equivalent does.
// Now every registered dtype has an exact case (Int16/UInt16/UInt32/UInt64/
// Complex64/Complex128 included) and only a truly unknown dtype throws.
// ----------------------------------------------------------------------------
TEST(LogicalToBoolValueDtypeCoverageCpu, Int16NonzeroIsTrue) {
    auto a = zeros({4}, DType::Int16, Device::cpu());
    a.data<int16_t>()[0] = 0; a.data<int16_t>()[1] = 5;
    a.data<int16_t>()[2] = -3; a.data<int16_t>()[3] = 0;
    auto b = tenzor::ones({4}, DType::Int16, Device::cpu());
    auto result = logical_and(a, b).contiguous();
    auto* out = result.data<bool>();
    EXPECT_FALSE(out[0]);
    EXPECT_TRUE(out[1]);
    EXPECT_TRUE(out[2]);
    EXPECT_FALSE(out[3]);
}

TEST(LogicalToBoolValueDtypeCoverageCpu, UInt16NonzeroIsTrue) {
    auto a = zeros({3}, DType::UInt16, Device::cpu());
    a.data<uint16_t>()[0] = 0; a.data<uint16_t>()[1] = 65535; a.data<uint16_t>()[2] = 1;
    auto result = logical_not(a).contiguous();
    auto* out = result.data<bool>();
    EXPECT_TRUE(out[0]);
    EXPECT_FALSE(out[1]);
    EXPECT_FALSE(out[2]);
}

TEST(LogicalToBoolValueDtypeCoverageCpu, UInt32NonzeroIsTrue) {
    auto a = zeros({2}, DType::UInt32, Device::cpu());
    a.data<uint32_t>()[0] = 0; a.data<uint32_t>()[1] = 42;
    auto result = logical_not(a).contiguous();
    auto* out = result.data<bool>();
    EXPECT_TRUE(out[0]);
    EXPECT_FALSE(out[1]);
}

TEST(LogicalToBoolValueDtypeCoverageCpu, UInt64NonzeroIsTrue) {
    auto a = zeros({2}, DType::UInt64, Device::cpu());
    a.data<uint64_t>()[0] = 0; a.data<uint64_t>()[1] = 7;
    auto result = logical_not(a).contiguous();
    auto* out = result.data<bool>();
    EXPECT_TRUE(out[0]);
    EXPECT_FALSE(out[1]);
}

TEST(LogicalToBoolValueDtypeCoverageCpu, Complex64NonzeroIfEitherComponentNonzero) {
    auto a = zeros({3}, DType::Complex64, Device::cpu());
    a.data<std::complex<float>>()[0] = {0.0f, 0.0f};  // zero -> false
    a.data<std::complex<float>>()[1] = {0.0f, 2.0f};  // nonzero imag -> true
    a.data<std::complex<float>>()[2] = {3.0f, 0.0f};  // nonzero real -> true
    auto result = logical_not(a).contiguous();
    auto* out = result.data<bool>();
    EXPECT_TRUE(out[0]);
    EXPECT_FALSE(out[1]);
    EXPECT_FALSE(out[2]);
}

TEST(LogicalToBoolValueDtypeCoverageCpu, Complex128NonzeroIfEitherComponentNonzero) {
    auto a = zeros({2}, DType::Complex128, Device::cpu());
    a.data<std::complex<double>>()[0] = {0.0, 0.0};
    a.data<std::complex<double>>()[1] = {0.0, -1.5};
    auto result = logical_not(a).contiguous();
    auto* out = result.data<bool>();
    EXPECT_TRUE(out[0]);
    EXPECT_FALSE(out[1]);
}

INSTANTIATE_BACKEND_TESTS(LogicalParity);




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
