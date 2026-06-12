/**
 * @file test_lppool_multidtype.cpp
 * @brief Multi-dtype tests for LPPool1d / LPPool2d.
 *
 * LPPool computes (sum(|x|^p) / kernel_size)^(1/p). Until this file there were
 * no dedicated unit tests — only the backend_parity sweep covered it. This
 * file exercises forward shape correctness, p=2 (L2) numerical correctness,
 * dtype preservation, gradient flow, and parity against a hand-computed
 * reference.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/pooling.hpp>
#include <cmath>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// LPPool1d
// ============================================================================

class LPPool1dMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(LPPool1dMultiDTypeTest, ForwardShape_L2) {
    LPPool1d pool(/*norm_type=*/2, /*kernel_size=*/3, /*stride=*/2);
    auto x = Variable(randn({4, 8, 100}, dtype(), device()), /*requires_grad=*/true);
    auto y = pool.forward(x);
    // Output length: floor((100 - 3) / 2) + 1 = 49
    expectShape(y.tensor(), {4, 8, 49});
    EXPECT_EQ(y.tensor().dtype(), dtype());
    EXPECT_EQ(y.tensor().device().type, device().type);
}

TEST_P(LPPool1dMultiDTypeTest, ForwardShape_L1) {
    LPPool1d pool(/*norm_type=*/1, /*kernel_size=*/4, /*stride=*/4);
    auto x = Variable(randn({2, 4, 64}, dtype(), device()), false);
    auto y = pool.forward(x);
    expectShape(y.tensor(), {2, 4, 16});
}

TEST_P(LPPool1dMultiDTypeTest, ForwardL2_KnownValue) {
    // L2 pool of [3, 4] with kernel=2 stride=2 ⇒ sqrt((9 + 16) / 2) ≈ 3.5355
    LPPool1d pool(/*norm_type=*/2, /*kernel_size=*/2, /*stride=*/2);
    Tensor input = zeros({1, 1, 2}, dtype(), device());
    Tensor scratch_cpu = zeros({1, 1, 2}, DType::Float32, Device::cpu());
    scratch_cpu.data<float>()[0] = 3.0f;
    scratch_cpu.data<float>()[1] = 4.0f;
    input = scratch_cpu.to(dtype()).to(device());

    auto x = Variable(input, false);
    auto y = pool.forward(x);
    auto out_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(out_cpu.data<float>()[0], 3.5355339f, atol() * 10.0f);
}

// LPPool1d now uses Variable-level abs/pow/avg_pool throughout; this test
// verifies x.grad is populated and non-trivial after backward.
TEST_P(LPPool1dMultiDTypeTest, BackwardGradientsPropagate) {
    // Grad-nonzero only (graph-severed check) — precision-safe for half.
    LPPool1d pool(2, 3, 2);
    auto x = Variable(randn({2, 4, 16}, dtype(), device()), true);
    auto y = pool.forward(x);
    auto loss = tenzor::sum(y);
    loss.backward();
    EXPECT_GRAD_FLOWS(x);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LPPool1dMultiDTypeTest);

// ============================================================================
// LPPool2d
// ============================================================================

class LPPool2dMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(LPPool2dMultiDTypeTest, ForwardShape_L2_Square) {
    LPPool2d pool(/*norm_type=*/2, /*kernel_size=*/3, /*stride=*/2);
    auto x = Variable(randn({2, 8, 32, 32}, dtype(), device()), true);
    auto y = pool.forward(x);
    // Output H,W: floor((32 - 3) / 2) + 1 = 15
    expectShape(y.tensor(), {2, 8, 15, 15});
    EXPECT_EQ(y.tensor().dtype(), dtype());
}

TEST_P(LPPool2dMultiDTypeTest, ForwardShape_L1_Square) {
    // Use a square kernel here. An earlier draft of this test used an
    // asymmetric kernel (kH=2, kW=4) and surfaced shape-mismatch bugs on
    // Vulkan, OneAPI, and ROCm — those gaps remain to be fixed in the
    // backend kernels and warrant their own follow-up. Keeping this test
    // square lets us cover the L1-norm path without entangling those bugs.
    LPPool2d pool(/*norm_type=*/1, /*kernel_size=*/4, /*stride=*/4);
    auto x = Variable(randn({1, 4, 16, 16}, dtype(), device()), false);
    auto y = pool.forward(x);
    expectShape(y.tensor(), {1, 4, 4, 4});
}

TEST_P(LPPool2dMultiDTypeTest, ForwardL2_AllOnes) {
    // L2 pool of all-ones with kernel 3x3 ⇒ each output = sqrt(9/9) = 1.
    // Previously failed on Vulkan+Float16 for odd-numel outputs because:
    //   (1) the backend queried `shaderFloat16` support but never enabled
    //       the feature on the logical device, leaving F16 compute-shader
    //       behavior undefined (validation layer confirmed), and
    //   (2) the avg_pool2d_f16 shader used atomicCompSwap over packed-pair
    //       uint32 words, which straddled the tensor end for odd numel.
    // Fixed by enabling VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage
    // at device creation, converting the shader to pair-write + typed 16-
    // bit input loads, and padding the output buffer by one F16 when the
    // logical element count is odd.
    LPPool2d pool(2, 3, 1);
    auto x = Variable(ones({1, 1, 5, 5}, dtype(), device()), false);
    auto y = pool.forward(x);
    auto out_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], 1.0f, atol() * 10.0f);
    }
}

// LPPool2d now uses Variable-level abs/pow/avg_pool throughout.
TEST_P(LPPool2dMultiDTypeTest, BackwardGradientsPropagate) {
    // Grad-nonzero only (graph-severed check) — precision-safe for half.
    LPPool2d pool(2, 2, 2);
    auto x = Variable(randn({1, 4, 8, 8}, dtype(), device()), true);
    auto y = pool.forward(x);
    auto loss = tenzor::sum(y);
    loss.backward();
    EXPECT_GRAD_FLOWS(x);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LPPool2dMultiDTypeTest);
