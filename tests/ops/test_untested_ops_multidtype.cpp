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

// Macro (not a method) so that GTEST_SKIP's internal `return`
// statement returns from the TEST_P body rather than from a helper
// method — otherwise the test continues and fails on the first op
// that doesn't support Float16.
#define skipIfHalf() \
    do { \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) { \
            SKIP_WITH_REASON(::tenzor::testing::SkipReason::NumericalDivergence, \
                             "Test reference values are tighter than FP16 can represent"); \
        } \
    } while (0)

class UntestedOpsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
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

// Regression: ROCm previously computed affine_grid forward by unconditionally
// downcasting theta to Float32, computing, then casting the OUTPUT back to
// Float64 -- restoring the dtype label but not the precision. Use theta
// values that differ only in the low bits a Float32 round-trip would erase
// (matches 1.0 + i*1e-10, the same pattern used elsewhere in this codebase
// to catch an accidental F32 round-trip) and require the CPU (native double)
// vs device() result to match far tighter than Float32's ~7 decimal digits.
TEST_P(UntestedOpsMultiDTypeTest, AffineGridFloat64PrecisionNoRoundTrip) {
    if (dtype() != DType::Float64) {
        GTEST_SKIP() << "Precision regression only meaningful for Float64";
    }
    auto theta_cpu = tenzor::zeros({1, 2, 3}, DType::Float64, Device::cpu());
    {
        double* p = theta_cpu.data<double>();
        // Identity-like affine transform, perturbed at the 1e-10 digit --
        // indistinguishable from the unperturbed identity once rounded to
        // Float32, but resolvable in true double precision.
        p[0] = 1.0 + 1e-10; p[1] = 0.0;        p[2] = 0.0;
        p[3] = 0.0;        p[4] = 1.0 + 2e-10; p[5] = 0.0;
    }

    auto grid_cpu = ops::affine_grid(theta_cpu, {1, 3, 64, 64}, /*align_corners=*/false);

    auto theta_dev = theta_cpu.to(device());
    auto grid_dev = ops::affine_grid(theta_dev, {1, 3, 64, 64}, /*align_corners=*/false).to(Device::cpu());

    double max_diff = 0.0;
    const double* a = grid_cpu.data<double>();
    const double* b = grid_dev.data<double>();
    for (int64_t i = 0; i < grid_cpu.numel(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    // A Float32 round-trip would collapse the 1e-10-scale perturbation
    // entirely (Float32 has ~7 decimal digits); require far tighter
    // agreement than that, proving the device computed in true double.
    EXPECT_LT(max_diff, 1e-12) << "affine_grid diverges from CPU by more than Float32 "
                                   "precision would allow on " << device().to_string();
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
