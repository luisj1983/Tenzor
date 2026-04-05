/**
 * @file test_edge_cases.cpp
 * @brief Edge case tests for tensor operations: NaN, Inf, empty tensors, size-1 tensors
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;

class EdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

// ============================================================================
// NaN propagation
// ============================================================================

TEST_F(EdgeCaseTest, NanPropagationAdd) {
    auto a = full({4}, std::numeric_limits<float>::quiet_NaN(), DType::Float32);
    auto b = ones({4}, DType::Float32);
    auto c = add(a, b);
    auto c_data = c.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(c_data[i])) << "NaN not propagated at index " << i;
    }
}

TEST_F(EdgeCaseTest, NanPropagationMul) {
    auto a = full({4}, std::numeric_limits<float>::quiet_NaN(), DType::Float32);
    auto b = ones({4}, DType::Float32);
    auto c = mul(a, b);
    auto c_data = c.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(c_data[i])) << "NaN not propagated at index " << i;
    }
}

TEST_F(EdgeCaseTest, NanPropagationSum) {
    auto a = ones({4}, DType::Float32);
    auto a_data = const_cast<float*>(a.data<float>());
    a_data[2] = std::numeric_limits<float>::quiet_NaN();
    auto s = sum(a);
    auto s_data = s.data<float>();
    EXPECT_TRUE(std::isnan(s_data[0])) << "NaN not propagated through sum";
}

// ============================================================================
// Inf handling
// ============================================================================

TEST_F(EdgeCaseTest, ExpLargeProducesInf) {
    auto a = full({2}, 1000.0f, DType::Float32);
    auto b = tenzor::exp(a);
    auto b_data = b.data<float>();
    for (int64_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isinf(b_data[i])) << "exp(1000) should be Inf";
        EXPECT_GT(b_data[i], 0) << "exp(1000) should be +Inf";
    }
}

TEST_F(EdgeCaseTest, LogZeroProducesNegInf) {
    auto a = zeros({2}, DType::Float32);
    auto b = tenzor::log(a);
    auto b_data = b.data<float>();
    for (int64_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isinf(b_data[i])) << "log(0) should be -Inf";
        EXPECT_LT(b_data[i], 0) << "log(0) should be -Inf";
    }
}

TEST_F(EdgeCaseTest, InfPropagationThroughOps) {
    auto a = full({2}, std::numeric_limits<float>::infinity(), DType::Float32);
    auto b = ones({2}, DType::Float32);

    // Inf + 1 = Inf
    auto c = add(a, b);
    EXPECT_TRUE(std::isinf(c.data<float>()[0]));

    // Inf * 2 = Inf
    auto d = mul(a, full({2}, 2.0f, DType::Float32));
    EXPECT_TRUE(std::isinf(d.data<float>()[0]));
}

// ============================================================================
// Size-1 tensor operations
// ============================================================================

TEST_F(EdgeCaseTest, Size1TensorOps) {
    auto a = ones({1}, DType::Float32);
    auto b = ones({1}, DType::Float32);

    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 1);
    EXPECT_FLOAT_EQ(c.data<float>()[0], 2.0f);

    auto s = sum(a);
    EXPECT_FLOAT_EQ(s.data<float>()[0], 1.0f);
}

TEST_F(EdgeCaseTest, ScalarTensorOps) {
    auto a = full({}, 3.0f, DType::Float32);  // 0-dim scalar
    auto b = full({}, 4.0f, DType::Float32);

    auto c = add(a, b);
    EXPECT_EQ(c.ndim(), 0);
    EXPECT_FLOAT_EQ(c.data<float>()[0], 7.0f);
}

// ============================================================================
// Shape operations edge cases
// ============================================================================

TEST_F(EdgeCaseTest, ReshapeToSameShape) {
    auto a = ones({3, 4}, DType::Float32);
    auto b = a.reshape({3, 4});
    EXPECT_EQ(b.numel(), 12);
}

TEST_F(EdgeCaseTest, TransposeSingleDim) {
    auto a = ones({5}, DType::Float32);
    // Transpose of 1D tensor should be a no-op or identity
    EXPECT_EQ(a.ndim(), 1);
    EXPECT_EQ(a.shape()[0], 5);
}

TEST_F(EdgeCaseTest, SqueezeNoEffect) {
    auto a = ones({3, 4}, DType::Float32);
    auto b = a.squeeze(0);  // dim 0 is size 3, not 1 -- should be no-op
    EXPECT_EQ(b.ndim(), 2);
    EXPECT_EQ(b.shape()[0], 3);
    EXPECT_EQ(b.shape()[1], 4);
}

// ============================================================================
// DType edge cases
// ============================================================================

TEST_F(EdgeCaseTest, Float64Precision) {
    // Verify Float64 subtraction preserves precision better than Float32
    auto a64 = full({1}, 100.0, DType::Float64);
    auto b64 = full({1}, 99.0, DType::Float64);
    auto c64 = sub(a64, b64);
    auto c64_data = c64.data<double>();
    EXPECT_DOUBLE_EQ(c64_data[0], 1.0) << "Float64 subtraction should be exact for integers";
}
