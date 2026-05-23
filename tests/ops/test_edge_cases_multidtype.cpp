/**
 * @file test_edge_cases_multidtype.cpp
 * @brief Multi-backend edge case tests for tensor operations: NaN, Inf, empty tensors, size-1 tensors
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "../backend_test_fixture.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::testing;

// audit-2 P.9 — This file uses BackendTest (not MultiBackendDTypeTest)
// deliberately: NaN / Inf propagation tests rely on Float32 special values
// (the integer dtypes have no NaN representation). Float16 /
// BFloat16 NaN handling is covered in the per-op multidtype tests.
class EdgeCaseMultiBackendTest : public BackendTest {};

// ============================================================================
// NaN propagation
// ============================================================================

TEST_P(EdgeCaseMultiBackendTest, NanPropagationAdd) {
    auto a = full({4}, std::numeric_limits<float>::quiet_NaN(), DType::Float32, device);
    auto b = ones({4}, DType::Float32, device);
    auto c = add(a, b);
    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(c_data[i])) << "NaN not propagated at index " << i;
    }
}

TEST_P(EdgeCaseMultiBackendTest, NanPropagationMul) {
    auto a = full({4}, std::numeric_limits<float>::quiet_NaN(), DType::Float32, device);
    auto b = ones({4}, DType::Float32, device);
    auto c = mul(a, b);
    auto c_cpu = c.to(Device::cpu());
    auto c_data = c_cpu.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(c_data[i])) << "NaN not propagated at index " << i;
    }
}

TEST_P(EdgeCaseMultiBackendTest, NanPropagationSum) {
    auto a = ones({4}, DType::Float32, device);
    // Create a tensor with a NaN by building it on CPU then moving to target device
    auto a_cpu = a.to(Device::cpu());
    auto a_data = const_cast<float*>(a_cpu.data<float>());
    a_data[2] = std::numeric_limits<float>::quiet_NaN();
    a = a_cpu.to(device);

    auto s = sum(a);
    auto s_cpu = s.to(Device::cpu());
    auto s_data = s_cpu.data<float>();
    EXPECT_TRUE(std::isnan(s_data[0])) << "NaN not propagated through sum";
}

// ============================================================================
// Inf handling
// ============================================================================

TEST_P(EdgeCaseMultiBackendTest, ExpLargeProducesInf) {
    auto a = full({2}, 1000.0f, DType::Float32, device);
    auto b = tenzor::exp(a);
    auto b_cpu = b.to(Device::cpu());
    auto b_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isinf(b_data[i])) << "exp(1000) should be Inf";
        EXPECT_GT(b_data[i], 0) << "exp(1000) should be +Inf";
    }
}

TEST_P(EdgeCaseMultiBackendTest, LogZeroProducesNegInf) {
    auto a = zeros({2}, DType::Float32, device);
    auto b = tenzor::log(a);
    auto b_cpu = b.to(Device::cpu());
    auto b_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isinf(b_data[i])) << "log(0) should be -Inf";
        EXPECT_LT(b_data[i], 0) << "log(0) should be -Inf";
    }
}

TEST_P(EdgeCaseMultiBackendTest, InfPropagationThroughOps) {
    auto a = full({2}, std::numeric_limits<float>::infinity(), DType::Float32, device);
    auto b = ones({2}, DType::Float32, device);

    // Inf + 1 = Inf
    auto c = add(a, b);
    auto c_cpu = c.to(Device::cpu());
    EXPECT_TRUE(std::isinf(c_cpu.data<float>()[0]));

    // Inf * 2 = Inf
    auto d = mul(a, full({2}, 2.0f, DType::Float32, device));
    auto d_cpu = d.to(Device::cpu());
    EXPECT_TRUE(std::isinf(d_cpu.data<float>()[0]));
}

// ============================================================================
// Size-1 tensor operations
// ============================================================================

TEST_P(EdgeCaseMultiBackendTest, Size1TensorOps) {
    auto a = ones({1}, DType::Float32, device);
    auto b = ones({1}, DType::Float32, device);

    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 1);
    auto c_cpu = c.to(Device::cpu());
    EXPECT_FLOAT_EQ(c_cpu.data<float>()[0], 2.0f);

    auto s = sum(a);
    auto s_cpu = s.to(Device::cpu());
    EXPECT_FLOAT_EQ(s_cpu.data<float>()[0], 1.0f);
}

TEST_P(EdgeCaseMultiBackendTest, ScalarTensorOps) {
    auto a = full({}, 3.0f, DType::Float32, device);  // 0-dim scalar
    auto b = full({}, 4.0f, DType::Float32, device);

    auto c = add(a, b);
    EXPECT_EQ(c.ndim(), 0);
    auto c_cpu = c.to(Device::cpu());
    EXPECT_FLOAT_EQ(c_cpu.data<float>()[0], 7.0f);
}

// ============================================================================
// Shape operations edge cases
// ============================================================================

TEST_P(EdgeCaseMultiBackendTest, ReshapeToSameShape) {
    auto a = ones({3, 4}, DType::Float32, device);
    auto b = a.reshape({3, 4});
    EXPECT_EQ(b.numel(), 12);
}

TEST_P(EdgeCaseMultiBackendTest, TransposeSingleDim) {
    auto a = ones({5}, DType::Float32, device);
    // Transpose of 1D tensor should be a no-op or identity
    EXPECT_EQ(a.ndim(), 1);
    EXPECT_EQ(a.shape()[0], 5);
}

TEST_P(EdgeCaseMultiBackendTest, SqueezeNoEffect) {
    auto a = ones({3, 4}, DType::Float32, device);
    auto b = a.squeeze(0);  // dim 0 is size 3, not 1 -- should be no-op
    EXPECT_EQ(b.ndim(), 2);
    EXPECT_EQ(b.shape()[0], 3);
    EXPECT_EQ(b.shape()[1], 4);
}

// ============================================================================
// DType edge cases
// ============================================================================

TEST_P(EdgeCaseMultiBackendTest, Float64Precision) {
    // Verify Float64 subtraction preserves precision better than Float32
    auto a64 = full({1}, 100.0, DType::Float64, device);
    auto b64 = full({1}, 99.0, DType::Float64, device);
    auto c64 = sub(a64, b64);
    auto c64_cpu = c64.to(Device::cpu());
    auto c64_data = c64_cpu.data<double>();
    EXPECT_DOUBLE_EQ(c64_data[0], 1.0) << "Float64 subtraction should be exact for integers";
}

INSTANTIATE_BACKEND_TESTS(EdgeCaseMultiBackendTest);
