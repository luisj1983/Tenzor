/**
 * @file test_inplace_arithmetic_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for in-place arithmetic
 *
 * Covers AddInplace/SubInplace/MulInplace/DivInplace (add_, sub_, mul_, div_)
 * across all registered backends × {Float32, Float64, Float16, BFloat16}.
 *
 * Verifies:
 *   - Storage pointer stability (mutation in-place, no reallocation)
 *   - Value correctness vs. the out-of-place equivalent
 *   - Broadcasting works
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/math.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class InplaceArithmeticMultiDTypeTest : public MultiBackendDTypeTest {};

namespace {

Tensor makeRand(const std::vector<int64_t>& shape, DType dtype, Device dev) {
    auto t = tenzor::randn(shape, DType::Float32, dev);
    if (dtype != DType::Float32) t = t.to(dtype);
    return t;
}

}  // namespace

// ============================================================================
// add_
// ============================================================================

TEST_P(InplaceArithmeticMultiDTypeTest, AddInplaceMatchesOutOfPlace) {
    auto a = makeRand({4, 4}, dtype_, device_);
    auto b = makeRand({4, 4}, dtype_, device_);
    auto expected = a + b;

    void* before = a.data_ptr();
    tenzor::add_(a, b);
    EXPECT_EQ(a.data_ptr(), before) << "add_ must mutate storage in place";

    expectTensorNear(a, expected, atol_);
}

TEST_P(InplaceArithmeticMultiDTypeTest, AddInplaceBroadcasts) {
    auto a = makeRand({3, 4}, dtype_, device_);
    auto b = makeRand({4}, dtype_, device_);
    auto expected = a + b;

    tenzor::add_(a, b);
    expectTensorNear(a, expected, atol_);
}

// ============================================================================
// sub_
// ============================================================================

TEST_P(InplaceArithmeticMultiDTypeTest, SubInplaceMatchesOutOfPlace) {
    auto a = makeRand({4, 4}, dtype_, device_);
    auto b = makeRand({4, 4}, dtype_, device_);
    auto expected = a - b;

    void* before = a.data_ptr();
    tenzor::sub_(a, b);
    EXPECT_EQ(a.data_ptr(), before) << "sub_ must mutate storage in place";

    expectTensorNear(a, expected, atol_);
}

TEST_P(InplaceArithmeticMultiDTypeTest, SubInplaceBroadcasts) {
    auto a = makeRand({3, 4}, dtype_, device_);
    auto b = makeRand({4}, dtype_, device_);
    auto expected = a - b;

    tenzor::sub_(a, b);
    expectTensorNear(a, expected, atol_);
}

// ============================================================================
// mul_
// ============================================================================

TEST_P(InplaceArithmeticMultiDTypeTest, MulInplaceMatchesOutOfPlace) {
    auto a = makeRand({4, 4}, dtype_, device_);
    auto b = makeRand({4, 4}, dtype_, device_);
    auto expected = a * b;

    void* before = a.data_ptr();
    tenzor::mul_(a, b);
    EXPECT_EQ(a.data_ptr(), before) << "mul_ must mutate storage in place";

    expectTensorNear(a, expected, atol_);
}

TEST_P(InplaceArithmeticMultiDTypeTest, MulInplaceBroadcasts) {
    auto a = makeRand({3, 4}, dtype_, device_);
    auto b = makeRand({4}, dtype_, device_);
    auto expected = a * b;

    tenzor::mul_(a, b);
    expectTensorNear(a, expected, atol_);
}

// ============================================================================
// div_
// ============================================================================

TEST_P(InplaceArithmeticMultiDTypeTest, DivInplaceMatchesOutOfPlace) {
    auto a = makeRand({4, 4}, dtype_, device_);
    // Bias the divisor away from zero to avoid precision blowups at lower dtypes.
    auto b_base = makeRand({4, 4}, dtype_, device_);
    auto two = tenzor::full({4, 4}, 2.0f, dtype_, device_);
    auto b = b_base + two;  // values in ~[1, 3]
    auto expected = a / b;

    void* before = a.data_ptr();
    tenzor::div_(a, b);
    EXPECT_EQ(a.data_ptr(), before) << "div_ must mutate storage in place";

    expectTensorNear(a, expected, atol_);
}

TEST_P(InplaceArithmeticMultiDTypeTest, DivInplaceBroadcasts) {
    auto a = makeRand({3, 4}, dtype_, device_);
    auto b_base = makeRand({4}, dtype_, device_);
    auto two = tenzor::full({4}, 2.0f, dtype_, device_);
    auto b = b_base + two;
    auto expected = a / b;

    tenzor::div_(a, b);
    expectTensorNear(a, expected, atol_);
}

// ============================================================================
// Multi-call chain (verifies repeated in-place mutation is stable)
// ============================================================================

TEST_P(InplaceArithmeticMultiDTypeTest, ChainedInplaceOps) {
    auto a = tenzor::ones({4, 4}, dtype_, device_);
    auto b = tenzor::ones({4, 4}, dtype_, device_);

    tenzor::add_(a, b);   // a = 2
    tenzor::mul_(a, b);   // a = 2
    tenzor::add_(a, b);   // a = 3
    tenzor::sub_(a, b);   // a = 2

    auto expected = tenzor::full({4, 4}, 2.0f, dtype_, device_);
    expectTensorNear(a, expected, atol_);
}

INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS(InplaceArithmeticMultiDTypeTest);
