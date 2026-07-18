/**
 * @file test_fractional_maxpool_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for FractionalMaxPool2d/3d
 *
 * Covers forward correctness of fractional_max_pool2d / fractional_max_pool3d
 * across all registered backends × {Float32, Float64, Float16}. Value-level
 * checks: output is always within the input's range (since it's a max-pool
 * variant), output shape matches the requested output_size, and indices are
 * in [0, spatial_numel).
 *
 * Backward path: src/nn/functional.cpp wires an IndexedPoolBackward grad_fn
 * that dispatches to OpId::FractionalMaxPool{2,3}dBackward on the appropriate
 * backend. Tests below include GradientFlow assertions.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class FractionalMaxPoolMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// FractionalMaxPool2d
// ============================================================================

TEST_P(FractionalMaxPoolMultiDTypeTest, FractionalMaxPool2dForwardShape) {
    Variable input = createInput({1, 2, 8, 8}, false);
    auto [out, idx] = nn::functional::fractional_max_pool2d(
        input,
        std::make_pair<int64_t, int64_t>(2, 2),
        std::make_pair<int64_t, int64_t>(4, 4));
    expectShape(out.tensor(), {1, 2, 4, 4});
    expectDevice(out.tensor());
    expectDType(out.tensor());

    // Indices must point into the spatial extent of the input (8*8 = 64).
    auto idx_cpu = idx.to(Device::cpu());
    const auto* ip = idx_cpu.data<int64_t>();
    for (int64_t i = 0; i < idx_cpu.numel(); ++i) {
        EXPECT_GE(ip[i], 0);
        EXPECT_LT(ip[i], 64);
    }
}

TEST_P(FractionalMaxPoolMultiDTypeTest, FractionalMaxPool2dGradientFlow) {
    Variable input = createInput({1, 1, 6, 6}, true);
    auto [out, _] = nn::functional::fractional_max_pool2d(
        input,
        std::make_pair<int64_t, int64_t>(2, 2),
        std::make_pair<int64_t, int64_t>(3, 3));
    auto grad = tenzor::ones({1, 1, 3, 3}, dtype_, device_);
    out.backward(grad);
    ASSERT_TRUE(input.grad().has_value())
        << "FractionalMaxPool2d backward did not populate input.grad on "
        << device().to_string();
    expectShape(*input.grad(), {1, 1, 6, 6});
}

TEST_P(FractionalMaxPoolMultiDTypeTest, FractionalMaxPool2dOutputInInputRange) {
    auto input_cpu = tenzor::randn({1, 1, 8, 8}, DType::Float32, Device::cpu());
    float in_min, in_max;
    {
        const auto* p = input_cpu.data<float>();
        in_min = in_max = p[0];
        for (int64_t i = 1; i < input_cpu.numel(); ++i) {
            in_min = std::min(in_min, p[i]);
            in_max = std::max(in_max, p[i]);
        }
    }
    if (dtype_ != DType::Float32) input_cpu = input_cpu.to(dtype_);
    Variable input(input_cpu.to(device_), false);

    auto [out, _] = nn::functional::fractional_max_pool2d(
        input,
        std::make_pair<int64_t, int64_t>(2, 2),
        std::make_pair<int64_t, int64_t>(4, 4));

    auto out_cpu = out.tensor().to(Device::cpu());
    if (out_cpu.dtype() != DType::Float32) out_cpu = out_cpu.to(DType::Float32);
    const auto* op = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_GE(op[i], in_min - atol_);
        EXPECT_LE(op[i], in_max + atol_);
    }
}

// frac_pool_start's two clamp bounds must be applied upper-bound-first,
// lower-bound-second: when kernel_size > input spatial size, in_size - pool
// is negative, and clamping to it AFTER the >=0 clamp drives start negative
// again -- an out-of-bounds input read (and a negative index written to the
// indices output) in the CUDA/ROCm/OneAPI/Vulkan kernels that had the clamps
// reversed relative to CPU's reference order. Exercises exactly the
// finding's failure scenario: input[N,C,4,4], kernel_size=(6,6),
// output_size=(2,2).
TEST_P(FractionalMaxPoolMultiDTypeTest, FractionalMaxPool2dKernelLargerThanInput_NoOOB) {
    auto input_cpu = tenzor::randn({1, 2, 4, 4}, DType::Float32, Device::cpu());
    float in_min, in_max;
    {
        const auto* p = input_cpu.data<float>();
        in_min = in_max = p[0];
        for (int64_t i = 1; i < input_cpu.numel(); ++i) {
            in_min = std::min(in_min, p[i]);
            in_max = std::max(in_max, p[i]);
        }
    }
    if (dtype_ != DType::Float32) input_cpu = input_cpu.to(dtype_);
    Variable input(input_cpu.to(device_), false);

    auto [out, idx] = nn::functional::fractional_max_pool2d(
        input,
        std::make_pair<int64_t, int64_t>(6, 6),
        std::make_pair<int64_t, int64_t>(2, 2));
    expectShape(out.tensor(), {1, 2, 2, 2});

    auto out_cpu = out.tensor().to(Device::cpu());
    if (out_cpu.dtype() != DType::Float32) out_cpu = out_cpu.to(DType::Float32);
    const auto* op = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_GE(op[i], in_min - atol_) << "output[" << i << "] below input range -- OOB read?";
        EXPECT_LE(op[i], in_max + atol_) << "output[" << i << "] above input range -- OOB read?";
    }

    // Indices must point into the spatial extent of the input (4*4 = 16), and
    // in particular must never be negative.
    auto idx_cpu = idx.to(Device::cpu());
    const auto* ip = idx_cpu.data<int64_t>();
    for (int64_t i = 0; i < idx_cpu.numel(); ++i) {
        EXPECT_GE(ip[i], 0) << "index[" << i << "] is negative -- frac_pool_start clamp bug";
        EXPECT_LT(ip[i], 16);
    }
}

// ============================================================================
// FractionalMaxPool3d
// ============================================================================

TEST_P(FractionalMaxPoolMultiDTypeTest, FractionalMaxPool3dForwardShape) {
    Variable input = createInput({1, 2, 4, 4, 4}, false);
    auto [out, idx] = nn::functional::fractional_max_pool3d(
        input,
        std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2),
        std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2));
    expectShape(out.tensor(), {1, 2, 2, 2, 2});
    expectDevice(out.tensor());
    expectDType(out.tensor());

    auto idx_cpu = idx.to(Device::cpu());
    const auto* ip = idx_cpu.data<int64_t>();
    for (int64_t i = 0; i < idx_cpu.numel(); ++i) {
        EXPECT_GE(ip[i], 0);
        EXPECT_LT(ip[i], 64);  // 4*4*4
    }
}

TEST_P(FractionalMaxPoolMultiDTypeTest, FractionalMaxPool3dGradientFlow) {
    Variable input = createInput({1, 1, 4, 4, 4}, true);
    auto [out, _] = nn::functional::fractional_max_pool3d(
        input,
        std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2),
        std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2));
    auto grad = tenzor::ones({1, 1, 2, 2, 2}, dtype_, device_);
    out.backward(grad);
    ASSERT_TRUE(input.grad().has_value())
        << "FractionalMaxPool3d backward did not populate input.grad on "
        << device().to_string();
    expectShape(*input.grad(), {1, 1, 4, 4, 4});
}

TEST_P(FractionalMaxPoolMultiDTypeTest, FractionalMaxPool3dOutputInInputRange) {
    auto input_cpu = tenzor::randn({1, 1, 4, 4, 4}, DType::Float32, Device::cpu());
    float in_min, in_max;
    {
        const auto* p = input_cpu.data<float>();
        in_min = in_max = p[0];
        for (int64_t i = 1; i < input_cpu.numel(); ++i) {
            in_min = std::min(in_min, p[i]);
            in_max = std::max(in_max, p[i]);
        }
    }
    if (dtype_ != DType::Float32) input_cpu = input_cpu.to(dtype_);
    Variable input(input_cpu.to(device_), false);

    auto [out, _] = nn::functional::fractional_max_pool3d(
        input,
        std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2),
        std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2));

    auto out_cpu = out.tensor().to(Device::cpu());
    if (out_cpu.dtype() != DType::Float32) out_cpu = out_cpu.to(DType::Float32);
    const auto* op = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_GE(op[i], in_min - atol_);
        EXPECT_LE(op[i], in_max + atol_);
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FractionalMaxPoolMultiDTypeTest);
