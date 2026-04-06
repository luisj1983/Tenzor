/**
 * @file test_empty_tensor_ops_multidtype.cpp
 * @brief Multi-backend tests for major operations with empty tensors
 *
 * Converted from test_empty_tensor_ops.cpp to run across all backends.
 *
 * Verifies that all major operation categories handle empty tensors
 * ({0}, {0, N}, {N, 0} shapes) without crashing and produce correct
 * output shapes.
 *
 * Categories tested:
 * - Creation ops (zeros, ones, full, randn)
 * - Math ops (add, mul, matmul)
 * - Reduction ops (sum, mean, max, min)
 * - Shape ops (reshape, transpose, squeeze)
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/transform.hpp>
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::testing;

class EmptyTensorOpsMultiBackendTest : public BackendTest {};

// ============================================================================
// 1. Creation Ops with Empty Shapes
// ============================================================================

TEST_P(EmptyTensorOpsMultiBackendTest, ZerosEmptyShapes) {
    // {0} - 1D empty
    auto t0 = zeros({0}, DType::Float32, device);
    EXPECT_EQ(t0.numel(), 0);
    EXPECT_EQ(t0.ndim(), 1);
    EXPECT_EQ(t0.shape()[0], 0);

    // {0, N}
    auto t0n = zeros({0, 4}, DType::Float32, device);
    EXPECT_EQ(t0n.numel(), 0);
    EXPECT_EQ(t0n.ndim(), 2);
    EXPECT_EQ(t0n.shape()[0], 0);
    EXPECT_EQ(t0n.shape()[1], 4);

    // {N, 0}
    auto tn0 = zeros({3, 0}, DType::Float32, device);
    EXPECT_EQ(tn0.numel(), 0);
    EXPECT_EQ(tn0.ndim(), 2);
    EXPECT_EQ(tn0.shape()[0], 3);
    EXPECT_EQ(tn0.shape()[1], 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, OnesEmptyShapes) {
    auto t0 = ones({0}, DType::Float32, device);
    EXPECT_EQ(t0.numel(), 0);
    EXPECT_EQ(t0.ndim(), 1);

    auto t0n = ones({0, 5}, DType::Float32, device);
    EXPECT_EQ(t0n.numel(), 0);
    EXPECT_EQ(t0n.shape()[0], 0);
    EXPECT_EQ(t0n.shape()[1], 5);

    auto tn0 = ones({7, 0}, DType::Float32, device);
    EXPECT_EQ(tn0.numel(), 0);
    EXPECT_EQ(tn0.shape()[0], 7);
    EXPECT_EQ(tn0.shape()[1], 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, FullEmptyShapes) {
    auto t0 = full({0}, 42.0f, DType::Float32, device);
    EXPECT_EQ(t0.numel(), 0);

    auto t0n = full({0, 3}, 42.0f, DType::Float32, device);
    EXPECT_EQ(t0n.numel(), 0);
    EXPECT_EQ(t0n.shape()[0], 0);
    EXPECT_EQ(t0n.shape()[1], 3);

    auto tn0 = full({5, 0}, 42.0f, DType::Float32, device);
    EXPECT_EQ(tn0.numel(), 0);
    EXPECT_EQ(tn0.shape()[0], 5);
    EXPECT_EQ(tn0.shape()[1], 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, RandnEmptyShapes) {
    auto t0 = randn({0}, DType::Float32, device);
    EXPECT_EQ(t0.numel(), 0);
    EXPECT_EQ(t0.ndim(), 1);

    auto t0n = randn({0, 8}, DType::Float32, device);
    EXPECT_EQ(t0n.numel(), 0);
    EXPECT_EQ(t0n.shape()[0], 0);
    EXPECT_EQ(t0n.shape()[1], 8);

    auto tn0 = randn({4, 0}, DType::Float32, device);
    EXPECT_EQ(tn0.numel(), 0);
    EXPECT_EQ(tn0.shape()[0], 4);
    EXPECT_EQ(tn0.shape()[1], 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, CreationPreservesDtype) {
    auto tf32 = zeros({0, 3}, DType::Float32, device);
    EXPECT_EQ(tf32.dtype(), DType::Float32);

    auto tf64 = zeros({0, 3}, DType::Float64, device);
    EXPECT_EQ(tf64.dtype(), DType::Float64);

    auto ti32 = zeros({0, 3}, DType::Int32, device);
    EXPECT_EQ(ti32.dtype(), DType::Int32);
}

// ============================================================================
// 2. Math Ops with Empty Tensors
// ============================================================================

TEST_P(EmptyTensorOpsMultiBackendTest, AddEmptyTensors) {
    auto a = zeros({0, 4}, DType::Float32, device);
    auto b = ones({0, 4}, DType::Float32, device);
    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 0);
    EXPECT_EQ(c.shape()[1], 4);
}

TEST_P(EmptyTensorOpsMultiBackendTest, MulEmptyTensors) {
    auto a = zeros({3, 0}, DType::Float32, device);
    auto b = ones({3, 0}, DType::Float32, device);
    auto c = mul(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, SubEmptyTensors) {
    auto a = zeros({0}, DType::Float32, device);
    auto b = zeros({0}, DType::Float32, device);
    auto c = sub(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.ndim(), 1);
    EXPECT_EQ(c.shape()[0], 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, MatmulEmptyInnerDim) {
    // (3, 0) x (0, 4) -> (3, 4) with all zeros (empty inner product)
    auto a = zeros({3, 0}, DType::Float32, device);
    auto b = zeros({0, 4}, DType::Float32, device);
    auto c = matmul(a, b);
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);
    // Result should be all zeros (sum over empty dimension)
    auto* data = c.to(Device::cpu()).data<float>();
    for (int64_t i = 0; i < c.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST_P(EmptyTensorOpsMultiBackendTest, MatmulEmptyOuterDim) {
    // (0, 4) x (4, 3) -> (0, 3)
    auto a = zeros({0, 4}, DType::Float32, device);
    auto b = ones({4, 3}, DType::Float32, device);
    auto c = matmul(a, b);
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 0);
    EXPECT_EQ(c.shape()[1], 3);
}

// ============================================================================
// 3. Reduction Ops on Empty Tensors
// ============================================================================

TEST_P(EmptyTensorOpsMultiBackendTest, SumEmptyGlobal) {
    auto t = zeros({0, 5}, DType::Float32, device);
    auto result = sum(t);
    EXPECT_EQ(result.numel(), 1);
    EXPECT_FLOAT_EQ(result.to(Device::cpu()).item<float>(), 0.0f);
}

TEST_P(EmptyTensorOpsMultiBackendTest, SumEmptyAlongEmptyDim) {
    // (2, 0) summed along dim=1 -> shape (2,), values = 0
    auto t = zeros({2, 0}, DType::Float32, device);
    auto result = sum(t, /*dim=*/1);
    EXPECT_EQ(result.numel(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    auto* data = result.to(Device::cpu()).data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
}

TEST_P(EmptyTensorOpsMultiBackendTest, SumEmptyAlongNonEmptyDim) {
    // (0, 5) summed along dim=1 -> shape (0,)
    auto t = zeros({0, 5}, DType::Float32, device);
    auto result = sum(t, /*dim=*/1);
    EXPECT_EQ(result.numel(), 0);
    EXPECT_EQ(result.shape()[0], 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, SumEmpty1D) {
    auto t = zeros({0}, DType::Float32, device);
    auto result = sum(t);
    EXPECT_EQ(result.numel(), 1);
    EXPECT_FLOAT_EQ(result.to(Device::cpu()).item<float>(), 0.0f);
}

TEST_P(EmptyTensorOpsMultiBackendTest, MeanEmpty1DReturnsNaN) {
    auto t = zeros({0}, DType::Float32, device);
    EXPECT_NO_FATAL_FAILURE({
        try {
            auto result = mean(t);
            if (result.numel() == 1) {
                float val = result.to(Device::cpu()).item<float>();
                EXPECT_TRUE(std::isnan(val) || val == 0.0f)
                    << "Mean of empty tensor should be NaN or 0, got " << val;
            }
        } catch (const std::exception&) {
            // Throwing is also acceptable behavior
        }
    });
}

TEST_P(EmptyTensorOpsMultiBackendTest, MaxEmpty1D) {
    auto t = zeros({0}, DType::Float32, device);
    EXPECT_NO_FATAL_FAILURE({
        try {
            auto result = max(t);
        } catch (const std::exception&) {
            // Throwing is acceptable
        }
    });
}

TEST_P(EmptyTensorOpsMultiBackendTest, MinEmpty1D) {
    auto t = zeros({0}, DType::Float32, device);
    EXPECT_NO_FATAL_FAILURE({
        try {
            auto result = min(t);
        } catch (const std::exception&) {
            // Throwing is acceptable
        }
    });
}

TEST_P(EmptyTensorOpsMultiBackendTest, SumEmptyKeepDim) {
    // (0, 4) summed along dim=0, keepdim=true -> (1, 4) with zeros
    auto t = zeros({0, 4}, DType::Float32, device);
    auto result = sum(t, /*dim=*/0, /*keepdim=*/true);
    EXPECT_EQ(result.shape()[0], 1);
    EXPECT_EQ(result.shape()[1], 4);
    EXPECT_EQ(result.numel(), 4);
    auto* data = result.to(Device::cpu()).data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

// ============================================================================
// 4. Shape Ops on Empty Tensors
// ============================================================================

TEST_P(EmptyTensorOpsMultiBackendTest, ReshapeEmptyPreservesNumel) {
    auto t = zeros({0, 6}, DType::Float32, device);
    auto r = reshape(t, {0, 2, 3});
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 3);
    EXPECT_EQ(r.shape()[0], 0);
    EXPECT_EQ(r.shape()[1], 2);
    EXPECT_EQ(r.shape()[2], 3);
}

TEST_P(EmptyTensorOpsMultiBackendTest, ReshapeEmptyToFlat) {
    auto t = zeros({0, 5}, DType::Float32, device);
    auto r = reshape(t, {0});
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 1);
    EXPECT_EQ(r.shape()[0], 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, ReshapeEmptySwapDims) {
    auto t = zeros({0, 3}, DType::Float32, device);
    auto r = reshape(t, {3, 0});
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, TransposeEmpty2D) {
    auto t = zeros({0, 5}, DType::Float32, device);
    auto r = transpose(t, 0, 1);
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 2);
    EXPECT_EQ(r.shape()[0], 5);
    EXPECT_EQ(r.shape()[1], 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, TransposeEmptyMiddleDim) {
    auto t = zeros({2, 0, 3}, DType::Float32, device);
    auto r = transpose(t, 0, 2);
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 0);
    EXPECT_EQ(r.shape()[2], 2);
}

TEST_P(EmptyTensorOpsMultiBackendTest, SqueezeEmptyNoop) {
    auto t = zeros({0, 5}, DType::Float32, device);
    auto r = squeeze(t, 0);
    EXPECT_EQ(r.numel(), 0);
    EXPECT_GE(r.ndim(), 1);
}

TEST_P(EmptyTensorOpsMultiBackendTest, SqueezeEmptyRemovesUnitDim) {
    // (1, 0, 5) squeezed at dim=0 -> (0, 5)
    auto t = zeros({1, 0, 5}, DType::Float32, device);
    auto r = squeeze(t, 0);
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 2);
    EXPECT_EQ(r.shape()[0], 0);
    EXPECT_EQ(r.shape()[1], 5);
}

TEST_P(EmptyTensorOpsMultiBackendTest, UnsqueezeEmpty) {
    // (0, 3) unsqueezed at dim=0 -> (1, 0, 3)
    auto t = zeros({0, 3}, DType::Float32, device);
    auto r = unsqueeze(t, 0);
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.ndim(), 3);
    EXPECT_EQ(r.shape()[0], 1);
    EXPECT_EQ(r.shape()[1], 0);
    EXPECT_EQ(r.shape()[2], 3);
}

TEST_P(EmptyTensorOpsMultiBackendTest, PermuteEmpty3D) {
    auto t = zeros({2, 0, 4}, DType::Float32, device);
    auto r = permute(t, {2, 0, 1});
    EXPECT_EQ(r.numel(), 0);
    EXPECT_EQ(r.shape()[0], 4);
    EXPECT_EQ(r.shape()[1], 2);
    EXPECT_EQ(r.shape()[2], 0);
}

// ============================================================================
// 5. Clone and To operations on Empty Tensors
// ============================================================================

TEST_P(EmptyTensorOpsMultiBackendTest, CloneEmpty) {
    auto t = zeros({0, 3}, DType::Float32, device);
    auto c = t.clone();
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape()[0], 0);
    EXPECT_EQ(c.shape()[1], 3);
    EXPECT_EQ(c.dtype(), DType::Float32);
}

TEST_P(EmptyTensorOpsMultiBackendTest, ToDtypeEmpty) {
    auto t = zeros({0, 3}, DType::Float32, device);
    auto t64 = t.to(DType::Float64);
    EXPECT_EQ(t64.numel(), 0);
    EXPECT_EQ(t64.shape()[0], 0);
    EXPECT_EQ(t64.shape()[1], 3);
    EXPECT_EQ(t64.dtype(), DType::Float64);
}

// ============================================================================
// 6. Combined Empty Dimension Patterns
// ============================================================================

TEST_P(EmptyTensorOpsMultiBackendTest, ChainedOpsOnEmpty) {
    auto a = zeros({0, 4}, DType::Float32, device);
    auto b = ones({0, 4}, DType::Float32, device);

    // add -> mul -> sum
    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 0);

    auto d = mul(c, c);
    EXPECT_EQ(d.numel(), 0);

    auto s = sum(d);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_FLOAT_EQ(s.to(Device::cpu()).item<float>(), 0.0f);
}

TEST_P(EmptyTensorOpsMultiBackendTest, ReshapeTransposeChained) {
    auto t = zeros({0, 6}, DType::Float32, device);
    auto r = reshape(t, {0, 2, 3});
    auto p = transpose(r, 1, 2);
    EXPECT_EQ(p.numel(), 0);
    EXPECT_EQ(p.shape()[0], 0);
    EXPECT_EQ(p.shape()[1], 3);
    EXPECT_EQ(p.shape()[2], 2);
}

TEST_P(EmptyTensorOpsMultiBackendTest, AllZeroDims) {
    auto t = zeros({0, 0}, DType::Float32, device);
    EXPECT_EQ(t.numel(), 0);

    auto s = sum(t);
    EXPECT_FLOAT_EQ(s.to(Device::cpu()).item<float>(), 0.0f);

    auto r = reshape(t, {0, 0});
    EXPECT_EQ(r.numel(), 0);
}

TEST_P(EmptyTensorOpsMultiBackendTest, HigherRankEmpty) {
    auto t = zeros({2, 0, 3, 4}, DType::Float32, device);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.ndim(), 4);

    auto s = sum(t);
    EXPECT_FLOAT_EQ(s.to(Device::cpu()).item<float>(), 0.0f);

    auto p = permute(t, {3, 2, 1, 0});
    EXPECT_EQ(p.shape()[0], 4);
    EXPECT_EQ(p.shape()[1], 3);
    EXPECT_EQ(p.shape()[2], 0);
    EXPECT_EQ(p.shape()[3], 2);
}

INSTANTIATE_BACKEND_TESTS(EmptyTensorOpsMultiBackendTest);
