/**
 * @file test_bitwise_parity.cpp
 * @brief Cross-backend parity tests for bitwise integer operations.
 *
 * Covers OpIds 520-525: BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot,
 * BitwiseLeftShift, BitwiseRightShift across Int8, Int16, Int32, Int64.
 *
 * Per the audit, the bitwise OpId family had no dedicated parity test —
 * coverage was implicit through `test_operation_parity.cpp` which exercises
 * a single dtype. This file ensures every backend produces bit-identical
 * results across the integer dtypes supported by each kernel.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class BitwiseParity : public BackendTest {};

namespace {
// Bit-exact equality is required for integer ops — relax the tolerance to 0.
constexpr float kIntRtol = 0.0f;
constexpr float kIntAtol = 0.0f;
}  // namespace

// ----------------------------------------------------------------------------
// Binary bitwise ops: AND, OR, XOR
// ----------------------------------------------------------------------------

#define DEFINE_BINARY_BITWISE_TEST(NAME, OP, DTYPE)                            \
    TEST_P(BitwiseParity, NAME##_##DTYPE) {                                    \
        auto a = randint(-128, 127, {4, 8}, DType::DTYPE, Device::cpu());      \
        auto b = randint(-128, 127, {4, 8}, DType::DTYPE, Device::cpu());      \
        test_operation_parity_single([](const std::vector<Tensor>& inputs) {   \
            return OP(inputs[0], inputs[1]);                                   \
        }, {a, b}, device, kIntRtol, kIntAtol, #NAME "_" #DTYPE);              \
    }

DEFINE_BINARY_BITWISE_TEST(BitwiseAnd, bitwise_and, Int8)
DEFINE_BINARY_BITWISE_TEST(BitwiseAnd, bitwise_and, Int16)
DEFINE_BINARY_BITWISE_TEST(BitwiseAnd, bitwise_and, Int32)
DEFINE_BINARY_BITWISE_TEST(BitwiseAnd, bitwise_and, Int64)

DEFINE_BINARY_BITWISE_TEST(BitwiseOr,  bitwise_or,  Int8)
DEFINE_BINARY_BITWISE_TEST(BitwiseOr,  bitwise_or,  Int16)
DEFINE_BINARY_BITWISE_TEST(BitwiseOr,  bitwise_or,  Int32)
DEFINE_BINARY_BITWISE_TEST(BitwiseOr,  bitwise_or,  Int64)

DEFINE_BINARY_BITWISE_TEST(BitwiseXor, bitwise_xor, Int8)
DEFINE_BINARY_BITWISE_TEST(BitwiseXor, bitwise_xor, Int16)
DEFINE_BINARY_BITWISE_TEST(BitwiseXor, bitwise_xor, Int32)
DEFINE_BINARY_BITWISE_TEST(BitwiseXor, bitwise_xor, Int64)

#undef DEFINE_BINARY_BITWISE_TEST

// ----------------------------------------------------------------------------
// Unary bitwise op: NOT
// ----------------------------------------------------------------------------

#define DEFINE_UNARY_BITWISE_TEST(DTYPE)                                       \
    TEST_P(BitwiseParity, BitwiseNot_##DTYPE) {                                \
        auto a = randint(-128, 127, {4, 8}, DType::DTYPE, Device::cpu());      \
        test_operation_parity_single([](const std::vector<Tensor>& inputs) {   \
            return bitwise_not(inputs[0]);                                     \
        }, {a}, device, kIntRtol, kIntAtol, "BitwiseNot_" #DTYPE);             \
    }

DEFINE_UNARY_BITWISE_TEST(Int8)
DEFINE_UNARY_BITWISE_TEST(Int16)
DEFINE_UNARY_BITWISE_TEST(Int32)
DEFINE_UNARY_BITWISE_TEST(Int64)

#undef DEFINE_UNARY_BITWISE_TEST

// ----------------------------------------------------------------------------
// Shifts: LeftShift, RightShift. Use small non-negative shift counts so the
// test is well-defined across signed/unsigned interpretations.
// ----------------------------------------------------------------------------

#define DEFINE_SHIFT_TEST(NAME, OP, DTYPE)                                     \
    TEST_P(BitwiseParity, NAME##_##DTYPE) {                                    \
        auto a = randint(-128, 127, {4, 8}, DType::DTYPE, Device::cpu());      \
        /* Per-element shift count in [0, 5] keeps signed shifts defined. */   \
        auto s = randint(0, 5, {4, 8}, DType::DTYPE, Device::cpu());           \
        test_operation_parity_single([](const std::vector<Tensor>& inputs) {   \
            return OP(inputs[0], inputs[1]);                                   \
        }, {a, s}, device, kIntRtol, kIntAtol, #NAME "_" #DTYPE);              \
    }

DEFINE_SHIFT_TEST(BitwiseLeftShift,  bitwise_left_shift,  Int8)
DEFINE_SHIFT_TEST(BitwiseLeftShift,  bitwise_left_shift,  Int16)
DEFINE_SHIFT_TEST(BitwiseLeftShift,  bitwise_left_shift,  Int32)
DEFINE_SHIFT_TEST(BitwiseLeftShift,  bitwise_left_shift,  Int64)

DEFINE_SHIFT_TEST(BitwiseRightShift, bitwise_right_shift, Int8)
DEFINE_SHIFT_TEST(BitwiseRightShift, bitwise_right_shift, Int16)
DEFINE_SHIFT_TEST(BitwiseRightShift, bitwise_right_shift, Int32)
DEFINE_SHIFT_TEST(BitwiseRightShift, bitwise_right_shift, Int64)

#undef DEFINE_SHIFT_TEST

INSTANTIATE_BACKEND_TESTS(BitwiseParity);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
