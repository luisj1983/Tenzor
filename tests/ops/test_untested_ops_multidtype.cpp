/**
 * @file test_untested_ops_multidtype.cpp
 * @brief Multi-backend tests for previously untested operations
 *
 * Covers: inplace activations (ReLU/Tanh/LeakyReLU/GELU), LogSumExp,
 * GridSample, AffineGrid — all of which had zero direct test coverage.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/vision.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class UntestedOpsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void skipIfHalf() {
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
            GTEST_SKIP() << "Test requires higher precision than Float16";
        }
    }
};

// ============================================================================
// Inplace Activation Tests
// ============================================================================

TEST_P(UntestedOpsMultiDTypeTest, ReLUInplace) {
    auto t = tenzor::randn({4, 8}, dtype(), device());
    auto orig_data_ptr = t.data_ptr();

    nn::relu_(t);

    // Verify in-place: same data pointer
    EXPECT_EQ(t.data_ptr(), orig_data_ptr) << "ReLU inplace should not allocate new buffer";

    // Verify all values are >= 0
    auto cpu_t = t.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* data = cpu_t.data<float>();
    for (int64_t i = 0; i < cpu_t.numel(); ++i) {
        EXPECT_GE(data[i], 0.0f) << "ReLU output negative at index " << i;
    }
}

TEST_P(UntestedOpsMultiDTypeTest, TanhInplace) {
    skipIfHalf();
    auto t = tenzor::randn({4, 8}, dtype(), device());
    auto orig_ptr = t.data_ptr();

    nn::tanh_(t);

    EXPECT_EQ(t.data_ptr(), orig_ptr);

    // Verify all values in [-1, 1]
    auto cpu_t = t.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* data = cpu_t.data<float>();
    for (int64_t i = 0; i < cpu_t.numel(); ++i) {
        EXPECT_GE(data[i], -1.0f - atol());
        EXPECT_LE(data[i], 1.0f + atol());
    }
}

TEST_P(UntestedOpsMultiDTypeTest, LeakyReLUInplace) {
    auto t = tenzor::randn({4, 8}, dtype(), device());
    auto orig_ptr = t.data_ptr();

    nn::leaky_relu_(t, 0.1);

    EXPECT_EQ(t.data_ptr(), orig_ptr);
    expectShape(t, {4, 8});
}

TEST_P(UntestedOpsMultiDTypeTest, GELUInplace) {
    skipIfHalf();
    auto t = tenzor::randn({4, 8}, dtype(), device());
    auto orig_ptr = t.data_ptr();

    nn::gelu_(t);

    EXPECT_EQ(t.data_ptr(), orig_ptr);
    expectShape(t, {4, 8});
}

// ============================================================================
// LogSumExp Tests
// ============================================================================

TEST_P(UntestedOpsMultiDTypeTest, LogSumExpReducesDim) {
    skipIfHalf();
    auto t = tenzor::randn({3, 4}, dtype(), device());
    auto result = tenzor::logsumexp(t, /*dim=*/1);
    expectShape(result, {3});
    expectDevice(result);
}

TEST_P(UntestedOpsMultiDTypeTest, LogSumExpKeepdim) {
    skipIfHalf();
    auto t = tenzor::randn({3, 4}, dtype(), device());
    auto result = tenzor::logsumexp(t, /*dim=*/1, /*keepdim=*/true);
    expectShape(result, {3, 1});
}

TEST_P(UntestedOpsMultiDTypeTest, LogSumExpMatchesNaiveFormula) {
    skipIfHalf();
    if (dtype() != DType::Float32 && dtype() != DType::Float64) {
        GTEST_SKIP() << "Numerical comparison only for Float32/Float64";
    }
    auto t = tenzor::randn({2, 5}, dtype(), device());

    auto lse = tenzor::logsumexp(t, /*dim=*/1);
    // Naive: log(sum(exp(t), dim=1))
    auto naive = tenzor::log(tenzor::sum(tenzor::exp(t), 1, false));

    expectTensorNear(lse, naive, std::max(atol() * 100.0f, 1e-3f));
}

// ============================================================================
// GridSample / AffineGrid Tests
// ============================================================================

TEST_P(UntestedOpsMultiDTypeTest, AffineGridShape) {
    skipIfHalf();
    // 2x3 affine matrices for batch of 2
    auto theta = tenzor::zeros({2, 2, 3}, dtype(), device());
    // Set to identity-like (but on GPU we can't set element-wise easily, so just test shape)
    auto grid = ops::affine_grid(theta, {2, 3, 4, 4}, /*align_corners=*/false);
    // Output: (N, H, W, 2)
    expectShape(grid, {2, 4, 4, 2});
    expectDevice(grid);
}

TEST_P(UntestedOpsMultiDTypeTest, GridSampleShape) {
    skipIfHalf();
    auto input = tenzor::randn({1, 3, 8, 8}, dtype(), device());
    // Grid: (N=1, H_out=4, W_out=4, 2)
    auto grid = tenzor::zeros({1, 4, 4, 2}, dtype(), device());

    auto output = ops::grid_sample(input, grid, "bilinear", "zeros", false);
    expectShape(output, {1, 3, 4, 4});
    expectDevice(output);
}

TEST_P(UntestedOpsMultiDTypeTest, GridSampleNearest) {
    skipIfHalf();
    auto input = tenzor::randn({1, 2, 4, 4}, dtype(), device());
    auto grid = tenzor::zeros({1, 2, 2, 2}, dtype(), device());

    auto output = ops::grid_sample(input, grid, "nearest", "zeros", false);
    expectShape(output, {1, 2, 2, 2});
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(UntestedOpsMultiDTypeTest);
