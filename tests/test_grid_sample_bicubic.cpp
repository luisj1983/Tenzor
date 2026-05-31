/**
 * @file test_grid_sample_bicubic.cpp
 * @brief Phase P0 / Fix 7: grid_sample(mode='bicubic') must use real bicubic
 *        interpolation, not silently fall back to bilinear, on both CPU and
 *        CUDA. Backward must produce real gradients via the analytic
 *        derivative of the cubic basis weights.
 *
 * Pre-fix:
 *  - CPU grid_sample_kernel had a comment "bicubic fallback (simplified as
 *    bilinear)" and unconditionally ran the bilinear math.
 *  - CUDA grid_sample_cuda unconditionally dispatched
 *    `grid_sample_bilinear_kernel` for bicubic mode.
 *  - GridSampleBackward only handled `mode_ == "bilinear"` and silently
 *    returned zero gradients for bicubic.
 *
 * Post-fix the test asserts: forward differs from bilinear on non-linear
 * inputs (proving the new path is actually bicubic), and backward produces
 * non-zero gradients.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/vision.hpp>

#include "backend_test_fixture.hpp"

#include <cmath>
#include <vector>

using namespace tenzor;

namespace {

// A simple non-linear input: a 4x4 image with values 0..15 (row-major).
// Bicubic should differ from bilinear at fractional grid coordinates on
// such inputs (linear interpolation matches bilinear exactly only on
// linear inputs).
auto make_nonlinear_input(Device device) -> Tensor {
    std::vector<float> data(16);
    for (int i = 0; i < 16; ++i) data[i] = static_cast<float>(i * i);  // 0, 1, 4, 9, ...
    auto t = Tensor::from_blob(data.data(), {1, 1, 4, 4}, DType::Float32, Device::cpu())
                  .clone();
    return t.to(device);
}

// 2x2 grid sampling at fractional positions away from cell centers so
// bilinear vs. bicubic disagree.
auto make_grid(Device device) -> Tensor {
    std::vector<float> g = {
        -0.5f, -0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f,  0.5f,  0.5f,
    };
    auto t = Tensor::from_blob(g.data(), {1, 2, 2, 2}, DType::Float32, Device::cpu()).clone();
    return t.to(device);
}

}  // namespace

namespace {

class GridSampleBicubicForward : public ::tenzor::testing::BackendTest {};
class GridSampleBicubicBackward : public ::tenzor::testing::BackendTest {};

TEST_P(GridSampleBicubicForward, DiffersFromBilinear) {
    auto input = make_nonlinear_input(device);
    auto grid  = make_grid(device);

    auto bil = tenzor::ops::grid_sample(input, grid, "bilinear", "zeros", false);
    auto bic = tenzor::ops::grid_sample(input, grid, "bicubic",  "zeros", false);

    auto b_cpu = bil.to(Device::cpu()).contiguous();
    auto c_cpu = bic.to(Device::cpu()).contiguous();
    ASSERT_EQ(b_cpu.numel(), c_cpu.numel());

    const float* b = b_cpu.data<float>();
    const float* c = c_cpu.data<float>();
    double sumsq = 0.0;
    for (int64_t i = 0; i < b_cpu.numel(); ++i) {
        double d = b[i] - c[i];
        sumsq += d * d;
    }
    EXPECT_GT(sumsq, 0.01)
        << "Bicubic output is identical to bilinear — indicates the silent "
           "fallback hasn't been fixed.";
}

TEST_P(GridSampleBicubicBackward, ProducesNonZeroGradients) {
    auto input_t = make_nonlinear_input(device);
    auto grid_t  = make_grid(device);
    Variable input(input_t, /*requires_grad=*/true);
    Variable grid(grid_t, /*requires_grad=*/true);

    auto out = tenzor::grid_sample(input, grid, "bicubic", "zeros", false);
    // Sum-of-Tensor not available as a Variable scalar reduce; use autograd::sum.
    auto loss = tenzor::sum(out.tensor());
    // Build a Variable scalar around the loss tensor and call backward via
    // the Variable's stored grad_fn (loss is itself a Tensor, so we need
    // to drive backward through the original `out` Variable).
    auto ones = tenzor::ones_like(out.tensor());
    out.backward(ones);

    auto gi_opt = input.grad();
    auto gg_opt = grid.grad();
    ASSERT_TRUE(gi_opt.has_value()) << "grad_input missing";
    ASSERT_TRUE(gg_opt.has_value()) << "grad_grid missing";

    auto gi_cpu = gi_opt->to(Device::cpu()).contiguous();
    auto gg_cpu = gg_opt->to(Device::cpu()).contiguous();

    const float* gi = gi_cpu.data<float>();
    const float* gg = gg_cpu.data<float>();

    double gi_sumsq = 0.0;
    for (int64_t i = 0; i < gi_cpu.numel(); ++i) {
        float v = gi[i];
        gi_sumsq += v * v;
    }
    double gg_sumsq = 0.0;
    for (int64_t i = 0; i < gg_cpu.numel(); ++i) {
        float v = gg[i];
        gg_sumsq += v * v;
    }
    // Pre-fix: GridSampleBackward only handled mode=='bilinear' and returned
    // zero gradients for bicubic. Both should be non-zero now.
    EXPECT_GT(gi_sumsq, 0.0) << "bicubic backward returned zero grad_input";
    EXPECT_GT(gg_sumsq, 0.0) << "bicubic backward returned zero grad_grid";
}

// Regression guard: bilinear mode unchanged.
TEST_P(GridSampleBicubicForward, BilinearForwardUnchanged) {
    auto input = make_nonlinear_input(device);
    auto grid  = make_grid(device);
    auto out = tenzor::ops::grid_sample(input, grid, "bilinear", "zeros", false);
    ASSERT_EQ(out.ndim(), 4);
    // The bilinear path is exercised by other tests; here we just ensure
    // it doesn't crash and produces the expected output shape.
    EXPECT_EQ(out.size(0), 1);
    EXPECT_EQ(out.size(1), 1);
    EXPECT_EQ(out.size(2), 2);
    EXPECT_EQ(out.size(3), 2);
}

INSTANTIATE_BACKEND_TESTS(GridSampleBicubicForward);
INSTANTIATE_BACKEND_TESTS(GridSampleBicubicBackward);

}  // namespace
