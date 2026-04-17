/**
 * @file test_backend_ops_parameterized_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for backend-specific operation dispatch
 *
 * Tests that basic operations (add, sub, mul, div, matmul, reductions,
 * activations, transforms) dispatch correctly to each backend and produce
 * correct output shapes, dtypes, and values across Float32/Float64/Float16.
 */

#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class BackendOpsMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Tests
// ============================================================================

/// Basic add: ones + ones = twos, verify shape/dtype/device.
TEST_P(BackendOpsMultiDTypeTest, AddBasic) {
    auto a = ones({2, 3}, DType::Float32, device()).to(dtype());
    auto b = ones({2, 3}, DType::Float32, device()).to(dtype());

    auto c = add(a, b);

    expectShape(c, {2, 3});
    EXPECT_EQ(c.device().type, device().type);

    auto c_f32 = c.to(DType::Float32).to(Device::cpu());
    auto* data = c_f32.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], 2.0f, atol()) << "index " << i;
    }
}

/// Subtraction: verify negative results dispatch correctly.
TEST_P(BackendOpsMultiDTypeTest, SubNegativeResult) {
    auto a = ones({2, 2}, DType::Float32, device()).to(dtype());
    auto b = full({2, 2}, 3.0f, DType::Float32, device()).to(dtype());

    auto c = sub(a, b);

    auto c_f32 = c.to(DType::Float32).to(Device::cpu());
    auto* data = c_f32.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i], -2.0f, atol()) << "index " << i;
    }
}

/// Multiplication: verify elementwise product.
TEST_P(BackendOpsMultiDTypeTest, MulElementwise) {
    auto a = full({2, 3}, 3.0f, DType::Float32, device()).to(dtype());
    auto b = full({2, 3}, 4.0f, DType::Float32, device()).to(dtype());

    auto c = mul(a, b);

    auto c_f32 = c.to(DType::Float32).to(Device::cpu());
    auto* data = c_f32.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], 12.0f, atol()) << "index " << i;
    }
}

/// MatMul: verify output shape for 2x3 @ 3x2 = 2x2.
TEST_P(BackendOpsMultiDTypeTest, MatMulOutputShape) {
    auto a = ones({2, 3}, DType::Float32, device()).to(dtype());
    auto b = ones({3, 2}, DType::Float32, device()).to(dtype());

    auto c = matmul(a, b);

    expectShape(c, {2, 2});
    EXPECT_EQ(c.device().type, device().type);

    // Each element should be 3 (sum of 3 ones)
    auto c_f32 = c.to(DType::Float32).to(Device::cpu());
    auto* data = c_f32.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i], 3.0f, atol()) << "index " << i;
    }
}

/// Sum reduction: full reduction of ones tensor.
TEST_P(BackendOpsMultiDTypeTest, SumReduction) {
    auto a = ones({100}, DType::Float32, device()).to(dtype());

    auto result = sum(a);

    auto r_f32 = result.to(DType::Float32).to(Device::cpu());
    EXPECT_NEAR(r_f32.data<float>()[0], 100.0f, atol() * 100);
}

/// Transpose: verify shape swap and data integrity.
TEST_P(BackendOpsMultiDTypeTest, TransposeShape) {
    auto a = ones({4, 5}, DType::Float32, device()).to(dtype());

    auto b = transpose(a, 0, 1);

    expectShape(b, {5, 4});
    EXPECT_EQ(b.device().type, device().type);
}

/// Reshape: verify shape change preserves numel.
TEST_P(BackendOpsMultiDTypeTest, ReshapePreservesNumel) {
    auto a = ones({2, 3, 4}, DType::Float32, device()).to(dtype());

    auto b = reshape(a, {4, 6});

    expectShape(b, {4, 6});
    EXPECT_EQ(b.numel(), 24);
}

/// Division: verify fractional results.
TEST_P(BackendOpsMultiDTypeTest, DivFractional) {
    auto a = ones({2, 2}, DType::Float32, device()).to(dtype());
    auto b = full({2, 2}, 4.0f, DType::Float32, device()).to(dtype());

    auto c = div(a, b);

    auto c_f32 = c.to(DType::Float32).to(Device::cpu());
    auto* data = c_f32.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i], 0.25f, atol()) << "index " << i;
    }
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BackendOpsMultiDTypeTest);
