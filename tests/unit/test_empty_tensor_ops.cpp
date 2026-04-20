/**
 * @file test_empty_tensor_ops.cpp
 * @brief Systematic tests for major operations with empty tensors
 *
 * Verifies that all major operation categories handle empty tensors
 * ({0}, {0, N}, {N, 0} shapes) without crashing and produce correct
 * output shapes. This complements test_empty_tensors.cpp which covers
 * basic creation and properties.
 *
 * Categories tested:
 * - Creation ops (zeros, ones, full, randn)
 * - Math ops (add, mul, matmul)
 * - Reduction ops (sum, mean, max, min)
 * - Shape ops (reshape, transpose, squeeze)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/transform.hpp>
#include <cmath>
#include <limits>

using namespace tenzor;

// ============================================================================
// Test Fixture
// ============================================================================

class EmptyTensorOpsTest : public ::testing::Test {
protected:
    static bool initialized;
    void SetUp() override {
        if (!initialized) {
            tenzor::initialize();
            initialized = true;
        }
    }
};

bool EmptyTensorOpsTest::initialized = false;

// ============================================================================
// 1. Creation Ops with Empty Shapes
// ============================================================================

TEST_F(EmptyTensorOpsTest, ZerosEmptyShapes) {
    // {0} - 1D empty
    auto t0 = zeros({0}, DType::Float32, Device::cpu());
    EXPECT_EQ(t0.numel(), 0);
    EXPECT_EQ(t0.ndim(), 1);
    EXPECT_EQ(t0.shape()[0], 0);

    // {0, N}
    auto t0n = zeros({0, 4}, DType::Float32, Device::cpu());
    EXPECT_EQ(t0n.numel(), 0);
    EXPECT_EQ(t0n.ndim(), 2);
    EXPECT_EQ(t0n.shape()[0], 0);
    EXPECT_EQ(t0n.shape()[1], 4);

    // {N, 0}
    auto tn0 = zeros({3, 0}, DType::Float32, Device::cpu());
    EXPECT_EQ(tn0.numel(), 0);
    EXPECT_EQ(tn0.ndim(), 2);
    EXPECT_EQ(tn0.shape()[0], 3);
    EXPECT_EQ(tn0.shape()[1], 0);
}

TEST_F(EmptyTensorOpsTest, OnesEmptyShapes) {
    auto t0 = ones({0}, DType::Float32, Device::cpu());
    EXPECT_EQ(t0.numel(), 0);
    EXPECT_EQ(t0.ndim(), 1);

    auto t0n = ones({0, 5}, DType::Float32, Device::cpu());
    EXPECT_EQ(t0n.numel(), 0);
    EXPECT_EQ(t0n.shape()[0], 0);
    EXPECT_EQ(t0n.shape()[1], 5);

    auto tn0 = ones({7, 0}, DType::Float32, Device::cpu());
    EXPECT_EQ(tn0.numel(), 0);
    EXPECT_EQ(tn0.shape()[0], 7);
    EXPECT_EQ(tn0.shape()[1], 0);
}

TEST_F(EmptyTensorOpsTest, FullEmptyShapes) {
    auto t0 = full({0}, 42.0f, DType::Float32, Device::cpu());
    EXPECT_EQ(t0.numel(), 0);

    auto t0n = full({0, 3}, 42.0f, DType::Float32, Device::cpu());
    EXPECT_EQ(t0n.numel(), 0);
    EXPECT_EQ(t0n.shape()[0], 0);
    EXPECT_EQ(t0n.shape()[1], 3);

    auto tn0 = full({5, 0}, 42.0f, DType::Float32, Device::cpu());
    EXPECT_EQ(tn0.numel(), 0);
    EXPECT_EQ(tn0.shape()[0], 5);
    EXPECT_EQ(tn0.shape()[1], 0);
}

TEST_F(EmptyTensorOpsTest, RandnEmptyShapes) {
    // randn with empty shape should produce a valid tensor with 0 elements
    auto t0 = randn({0}, DType::Float32, Device::cpu());
    EXPECT_EQ(t0.numel(), 0);
    EXPECT_EQ(t0.ndim(), 1);

    auto t0n = randn({0, 8}, DType::Float32, Device::cpu());
    EXPECT_EQ(t0n.numel(), 0);
    EXPECT_EQ(t0n.shape()[0], 0);
    EXPECT_EQ(t0n.shape()[1], 8);

    auto tn0 = randn({4, 0}, DType::Float32, Device::cpu());
    EXPECT_EQ(tn0.numel(), 0);
    EXPECT_EQ(tn0.shape()[0], 4);
    EXPECT_EQ(tn0.shape()[1], 0);
}

TEST_F(EmptyTensorOpsTest, CreationPreservesDtype) {
    auto tf32 = zeros({0, 3}, DType::Float32, Device::cpu());
    EXPECT_EQ(tf32.dtype(), DType::Float32);

    auto tf64 = zeros({0, 3}, DType::Float64, Device::cpu());
    EXPECT_EQ(tf64.dtype(), DType::Float64);

    auto ti32 = zeros({0, 3}, DType::Int32, Device::cpu());
    EXPECT_EQ(ti32.dtype(), DType::Int32);
}

// ============================================================================
// 2. Math Ops with Empty Tensors
// ============================================================================

TEST_F(EmptyTensorOpsTest, AddEmptyTensors) {
    // Same empty shape
    auto a = zeros({0, 4}, DType::Float32, Device::cpu());
    auto b = ones({0, 4}, DType::Float32, Device::cpu());
    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 0);
    EXPECT_EQ(c.shape()[1], 4);
}

TEST_F(EmptyTensorOpsTest, AddEmptyWithBroadcast) {
    auto a = zeros({0, 4}, DType::Float32, Device::cpu());
    auto b = ones({1, 4}, DType::Float32, Device::cpu());
    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 0);
    EXPECT_EQ(c.shape()[1], 4);
}

TEST_F(EmptyTensorOpsTest, MulEmptyTensors) {
    auto a = zeros({3, 0}, DType::Float32, Device::cpu());
    auto b = ones({3, 0}, DType::Float32, Device::cpu());
    auto c = mul(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 0);
}

TEST_F(EmptyTensorOpsTest, SubEmptyTensors) {
    auto a = zeros({0}, DType::Float32, Device::cpu());
    auto b = zeros({0}, DType::Float32, Device::cpu());
    auto c = sub(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.ndim(), 1);
    EXPECT_EQ(c.shape()[0], 0);
}

TEST_F(EmptyTensorOpsTest, MatmulEmptyInnerDim) {
    // (3, 0) x (0, 4) -> (3, 4) with all zeros (empty inner product)
    auto a = zeros({3, 0}, DType::Float32, Device::cpu());
    auto b = zeros({0, 4}, DType::Float32, Device::cpu());
    auto c = matmul(a, b);
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);
    // Result should be all zeros (sum over empty dimension)
    auto* data = c.data<float>();
    for (int64_t i = 0; i < c.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST_F(EmptyTensorOpsTest, MatmulEmptyOuterDim) {
    // (0, 4) x (4, 3) -> (0, 3)
    auto a = zeros({0, 4}, DType::Float32, Device::cpu());
    auto b = ones({4, 3}, DType::Float32, Device::cpu());
    auto c = matmul(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 0);
    EXPECT_EQ(c.shape()[1], 3);
}

// ============================================================================
// 3. Reduction Ops on Empty Tensors
// ============================================================================

TEST_F(EmptyTensorOpsTest, SumEmptyGlobal) {
    // Global sum of empty tensor -> scalar 0
    auto t = zeros({0, 5}, DType::Float32, Device::cpu());
    auto result = sum(t);
    EXPECT_EQ(result.numel(), 1);
    EXPECT_FLOAT_EQ(result.item<float>(), 0.0f);
}

TEST_F(EmptyTensorOpsTest, SumEmptyAlongEmptyDim) {
    // (2, 0) summed along dim=1 -> shape (2,), values = 0
    auto t = zeros({2, 0}, DType::Float32, Device::cpu());
    auto result = sum(t, /*dim=*/1);
    EXPECT_EQ(result.numel(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    auto* data = result.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
}

TEST_F(EmptyTensorOpsTest, SumEmptyAlongNonEmptyDim) {
    // (0, 5) summed along dim=1 -> shape (0,)
    auto t = zeros({0, 5}, DType::Float32, Device::cpu());
    auto result = sum(t, /*dim=*/1);
    EXPECT_EQ(result.numel(), 0);
    EXPECT_EQ(result.shape()[0], 0);
}

TEST_F(EmptyTensorOpsTest, SumEmpty1D) {
    auto t = zeros({0}, DType::Float32, Device::cpu());
    auto result = sum(t);
    EXPECT_EQ(result.numel(), 1);
    EXPECT_FLOAT_EQ(result.item<float>(), 0.0f);
}

TEST_F(EmptyTensorOpsTest, MeanEmpty1DReturnsNaN) {
    // Mean of empty tensor is undefined (0/0) - should return NaN or handle gracefully
    auto t = zeros({0}, DType::Float32, Device::cpu());
    // This may throw or return NaN depending on implementation;
    // the key requirement is that it does not crash
    EXPECT_NO_FATAL_FAILURE({
        try {
            auto result = mean(t);
            // If it doesn't throw, result should be NaN (mean of nothing)
            if (result.numel() == 1) {
                float val = result.item<float>();
                EXPECT_TRUE(std::isnan(val) || val == 0.0f)
                    << "Mean of empty tensor should be NaN or 0, got " << val;
            }
        } catch (const std::exception&) {
            // Throwing is also acceptable behavior
        }
    });
}

TEST_F(EmptyTensorOpsTest, MaxEmpty1D) {
    // Max of empty tensor is undefined; should not crash
    auto t = zeros({0}, DType::Float32, Device::cpu());
    EXPECT_NO_FATAL_FAILURE({
        try {
            auto result = max(t);
            // If returns, should be -inf or throw
        } catch (const std::exception&) {
            // Throwing is acceptable
        }
    });
}

TEST_F(EmptyTensorOpsTest, MinEmpty1D) {
    // Min of empty tensor is undefined; should not crash
    auto t = zeros({0}, DType::Float32, Device::cpu());
    EXPECT_NO_FATAL_FAILURE({
        try {
            auto result = min(t);
            // If returns, should be +inf or throw
        } catch (const std::exception&) {
            // Throwing is acceptable
        }
    });
}

TEST_F(EmptyTensorOpsTest, SumEmptyKeepDim) {
    // (0, 4) summed along dim=0, keepdim=true -> (1, 4) with zeros
    auto t = zeros({0, 4}, DType::Float32, Device::cpu());
    auto result = sum(t, /*dim=*/0, /*keepdim=*/true);
    EXPECT_EQ(result.shape()[0], 1);
    EXPECT_EQ(result.shape()[1], 4);
    EXPECT_EQ(result.numel(), 4);
    auto* data = result.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

// ============================================================================
// 4. Shape Ops on Empty Tensors
// ============================================================================

TEST_F(EmptyTensorOpsTest, ReshapeEmptyPreservesNumel) {
    auto t = zeros({0, 6}, DType::Float32, Device::cpu());
    auto r = reshape(t, {0, 2, 3});
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 3);
    EXPECT_EQ(r.shape()[0], 0);
    EXPECT_EQ(r.shape()[1], 2);
    EXPECT_EQ(r.shape()[2], 3);
}

TEST_F(EmptyTensorOpsTest, ReshapeEmptyToFlat) {
    auto t = zeros({0, 5}, DType::Float32, Device::cpu());
    auto r = reshape(t, {0});
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 1);
    EXPECT_EQ(r.shape()[0], 0);
}

TEST_F(EmptyTensorOpsTest, ReshapeEmptySwapDims) {
    auto t = zeros({0, 3}, DType::Float32, Device::cpu());
    auto r = reshape(t, {3, 0});
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 0);
}

TEST_F(EmptyTensorOpsTest, TransposeEmpty2D) {
    auto t = zeros({0, 5}, DType::Float32, Device::cpu());
    auto r = transpose(t, 0, 1);
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 2);
    EXPECT_EQ(r.shape()[0], 5);
    EXPECT_EQ(r.shape()[1], 0);
}

TEST_F(EmptyTensorOpsTest, TransposeEmptyMiddleDim) {
    auto t = zeros({2, 0, 3}, DType::Float32, Device::cpu());
    auto r = transpose(t, 0, 2);
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 0);
    EXPECT_EQ(r.shape()[2], 2);
}

TEST_F(EmptyTensorOpsTest, SqueezeEmptyNoop) {
    // Squeezing dimension that is 0 (not 1) should be a no-op
    auto t = zeros({0, 5}, DType::Float32, Device::cpu());
    auto r = squeeze(t, 0);
    // dim 0 is 0, not 1, so squeeze should not remove it
    EXPECT_EQ(r.numel(), 0);
    // Shape depends on implementation: squeeze of non-1 dim is typically a no-op
    EXPECT_GE(r.ndim(), 1);
}

TEST_F(EmptyTensorOpsTest, SqueezeEmptyRemovesUnitDim) {
    // (1, 0, 5) squeezed at dim=0 -> (0, 5)
    auto t = zeros({1, 0, 5}, DType::Float32, Device::cpu());
    auto r = squeeze(t, 0);
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 2);
    EXPECT_EQ(r.shape()[0], 0);
    EXPECT_EQ(r.shape()[1], 5);
}

TEST_F(EmptyTensorOpsTest, UnsqueezeEmpty) {
    // (0, 3) unsqueezed at dim=0 -> (1, 0, 3)
    auto t = zeros({0, 3}, DType::Float32, Device::cpu());
    auto r = unsqueeze(t, 0);
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 3);
    EXPECT_EQ(r.shape()[0], 1);
    EXPECT_EQ(r.shape()[1], 0);
    EXPECT_EQ(r.shape()[2], 3);
}

TEST_F(EmptyTensorOpsTest, PermuteEmpty3D) {
    auto t = zeros({2, 0, 4}, DType::Float32, Device::cpu());
    auto r = permute(t, {2, 0, 1});
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.shape()[0], 4);
    EXPECT_EQ(r.shape()[1], 2);
    EXPECT_EQ(r.shape()[2], 0);
}

// ============================================================================
// 5. Clone and To operations on Empty Tensors
// ============================================================================

TEST_F(EmptyTensorOpsTest, CloneEmpty) {
    auto t = zeros({0, 3}, DType::Float32, Device::cpu());
    auto c = t.clone();
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 0);
    EXPECT_EQ(c.shape()[1], 3);
    EXPECT_EQ(c.dtype(), DType::Float32);
}

TEST_F(EmptyTensorOpsTest, ToDtypeEmpty) {
    auto t = zeros({0, 3}, DType::Float32, Device::cpu());
    auto t64 = t.to(DType::Float64);
    EXPECT_EQ(t64.numel(), 0);
    EXPECT_EQ(t64.shape()[0], 0);
    EXPECT_EQ(t64.shape()[1], 3);
    EXPECT_EQ(t64.dtype(), DType::Float64);
}

// ============================================================================
// 6. Combined Empty Dimension Patterns
// ============================================================================

TEST_F(EmptyTensorOpsTest, ChainedOpsOnEmpty) {
    // Verify chaining multiple operations on empty tensors does not crash
    auto a = zeros({0, 4}, DType::Float32, Device::cpu());
    auto b = ones({0, 4}, DType::Float32, Device::cpu());

    // add -> mul -> sum
    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 0);

    auto d = mul(c, c);
    EXPECT_EQ(d.numel(), 0);

    auto s = sum(d);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_FLOAT_EQ(s.item<float>(), 0.0f);
}

TEST_F(EmptyTensorOpsTest, ReshapeTransposeChained) {
    auto t = zeros({0, 6}, DType::Float32, Device::cpu());
    auto r = reshape(t, {0, 2, 3});
    auto p = transpose(r, 1, 2);
    EXPECT_EQ(p.numel(), 0);
    EXPECT_EQ(p.shape()[0], 0);
    EXPECT_EQ(p.shape()[1], 3);
    EXPECT_EQ(p.shape()[2], 2);
}

TEST_F(EmptyTensorOpsTest, AllZeroDims) {
    // {0, 0} tensor
    auto t = zeros({0, 0}, DType::Float32, Device::cpu());
    EXPECT_EQ(t.numel(), 0);

    // Operations on {0,0} tensor
    auto s = sum(t);
    EXPECT_FLOAT_EQ(s.item<float>(), 0.0f);

    auto r = reshape(t, {0, 0});
    EXPECT_EQ(r.numel(), 0);
}

TEST_F(EmptyTensorOpsTest, HigherRankEmpty) {
    // 4D tensor with an empty dimension
    auto t = zeros({2, 0, 3, 4}, DType::Float32, Device::cpu());
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.ndim(), 4);

    auto s = sum(t);
    EXPECT_FLOAT_EQ(s.item<float>(), 0.0f);

    auto p = permute(t, {3, 2, 1, 0});
    EXPECT_EQ(p.shape()[0], 4);
    EXPECT_EQ(p.shape()[1], 3);
    EXPECT_EQ(p.shape()[2], 0);
    EXPECT_EQ(p.shape()[3], 2);
}
