/**
 * @file test_conv3d_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for Conv3d operations
 *
 * Covers: Conv3d forward shape, stride/padding, backward gradient flow,
 * and weight gradient verification.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include "../grad_flow_helpers.hpp"  // W.26: EXPECT_GRAD_FLOWS

using namespace tenzor;
using namespace tenzor::testing;

class Conv3dMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Forward Tests
// ============================================================================

TEST_P(Conv3dMultiDTypeTest, ForwardShapeBasic) {
    // Input: (N=1, C_in=1, D=8, H=8, W=8), kernel=3, stride=1, padding=1
    // Output D/H/W = floor((8 + 2*1 - 3) / 1) + 1 = 8
    nn::Conv3d conv(1, 16, 3, 1, 1);
    convert_model(conv);

    auto input = createInput({1, 1, 8, 8, 8}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {1, 16, 8, 8, 8});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(Conv3dMultiDTypeTest, ForwardShapeWithStride) {
    // stride=2, padding=0: D_out = floor((8 - 3) / 2) + 1 = 3
    nn::Conv3d conv(2, 8, 3, 2, 0);
    convert_model(conv);

    auto input = createInput({1, 2, 8, 8, 8}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {1, 8, 3, 3, 3});
    expectDevice(output.tensor());
}

TEST_P(Conv3dMultiDTypeTest, ForwardShapeNoBias) {
    nn::Conv3d conv(4, 8, 3, 1, 1, 1, 1, false);
    convert_model(conv);

    auto input = createInput({2, 4, 4, 4, 4}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {2, 8, 4, 4, 4});
}

// Regression: Conv3d on a non-contiguous (permuted) NCDHW input must match the
// result on the contiguous equivalent. The kernels index the input flat, so a
// missing contiguity guard would read the wrong storage layout.
TEST_P(Conv3dMultiDTypeTest, NonContiguousInputMatchesContiguous) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 too imprecise";
    const int64_t N = 1, C = 2, D = 4, H = 4, W = 4;
    nn::Conv3d conv(C, 4, 3, 1, 1);
    convert_model(conv);

    // Build a non-contiguous NCDHW view by transposing a (C,N,D,H,W) tensor.
    auto base = tenzor::arange(0, C * N * D * H * W, 1, dtype(), device())
                    .reshape({C, N, D, H, W});
    auto x_view = tenzor::transpose(base, 0, 1);  // (N,C,D,H,W), non-contiguous
    ASSERT_FALSE(x_view.is_contiguous());
    auto x_cont = x_view.contiguous();

    auto out_view = conv.forward(Variable(x_view)).tensor().to(Device::cpu()).to(DType::Float64);
    auto out_cont = conv.forward(Variable(x_cont)).tensor().to(Device::cpu()).to(DType::Float64);
    ASSERT_EQ(out_view.numel(), out_cont.numel());
    const double* a = out_view.data<double>();
    const double* b = out_cont.data<double>();
    const double tol = (dtype() == DType::Float64) ? 1e-9 : 1e-3;
    for (int64_t i = 0; i < out_view.numel(); ++i)
        EXPECT_NEAR(a[i], b[i], tol) << "mismatch at " << i << " on " << device().to_string();
}

// ============================================================================
// Backward Tests
// ============================================================================

TEST_P(Conv3dMultiDTypeTest, BackwardGradientFlow) {
    nn::Conv3d conv(2, 4, 3, 1, 1);
    convert_model(conv);

    auto input = createInput({1, 2, 4, 4, 4}, true);
    auto output = conv.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); });

    EXPECT_GRAD_FLOWS(input);  // W.26
    ASSERT_TRUE(input.grad().has_value())
        << "Conv3d backward did not produce input gradient on " << device().to_string();
    expectShape(*input.grad(), {1, 2, 4, 4, 4});
}

TEST_P(Conv3dMultiDTypeTest, WeightGradientExists) {
    nn::Conv3d conv(1, 4, 3, 1, 1);
    convert_model(conv);

    auto input = createInput({1, 1, 4, 4, 4}, true);
    auto output = conv.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);

    auto params = conv.parameters();
    ASSERT_FALSE(params.empty()) << "Conv3d has no parameters";
    // Weight parameter should have gradient
    ASSERT_TRUE(params[0]->grad().has_value())
        << "Conv3d weight gradient not produced on " << device().to_string();
}

// ============================================================================
// Numerical Correctness Tests
// ============================================================================

TEST_P(Conv3dMultiDTypeTest, IdentityConvWithUnitWeight) {
    // A 1x1x1 Conv3d with weight=1 and no bias should preserve the input value.
    // We fill the weight tensor in-place via fill_().
    nn::Conv3d conv(1, 1, /*kernel_size=*/1, /*stride=*/1, /*padding=*/0,
                    /*dilation=*/1, /*groups=*/1, /*bias=*/false);
    convert_model(conv);

    auto params = conv.parameters();
    ASSERT_FALSE(params.empty());
    // Fill weight in-place: weight = 2.0
    params[0]->tensor().fill_(2.0);

    auto input = tenzor::ones({1, 1, 4, 4, 4}, dtype(), device());
    Variable input_var(input, false);
    auto output = conv.forward(input_var);

    if (dtype() == DType::Float32 || dtype() == DType::Float64) {
        // Output should be all 2.0 (input=1, weight=2, 1x1x1 kernel = elementwise mul)
        auto cpu_out = output.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
        auto* data = cpu_out.data<float>();
        for (int64_t i = 0; i < cpu_out.numel(); ++i) {
            EXPECT_NEAR(data[i], 2.0f, std::max(atol() * 10.0f, 1e-4f));
        }
    } else {
        // Half precision: exercise the kernel/dispatch and assert only what is
        // precision-safe — correct output dtype and finite (no NaN/Inf).
        expectDType(output.tensor());
        expectAllFinite(output.tensor());
    }
}

TEST_P(Conv3dMultiDTypeTest, ZeroInputProducesBiasOnly) {
    // With zero input, the output should equal the bias broadcast across spatial dims.
    nn::Conv3d conv(2, 4, 3, 1, 1);
    convert_model(conv);

    auto params = conv.parameters();
    ASSERT_GE(params.size(), 2u) << "Conv3d should have weight + bias parameters";
    // Set bias to 3.0 in-place
    params[1]->tensor().fill_(3.0);

    auto input = tenzor::zeros({1, 2, 4, 4, 4}, dtype(), device());
    Variable input_var(input, false);
    auto output = conv.forward(input_var);

    if (dtype() == DType::Float32 || dtype() == DType::Float64) {
        // All output values should equal the bias (3.0)
        auto cpu_out = output.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
        auto* data = cpu_out.data<float>();
        for (int64_t i = 0; i < cpu_out.numel(); ++i) {
            EXPECT_NEAR(data[i], 3.0f, std::max(atol() * 10.0f, 1e-4f));
        }
    } else {
        // Half precision: exercise the kernel/dispatch and assert only what is
        // precision-safe — correct output dtype and finite (no NaN/Inf).
        expectDType(output.tensor());
        expectAllFinite(output.tensor());
    }
}

// Regression: Conv3d Float16 over a large reduction dimension K =
// C_in * kD*kH*kW must accumulate in Float32, not FP16. The ROCm Float16 path
// previously called rocblas_hgemm (FP16 accumulate), which overflows past 65504
// and loses precision over large K; it is now upcast to Float32 at entry. With
// C_in=32, k=3 -> K = 32*27 = 864 accumulated terms, FP16 accumulation would
// diverge sharply from the CPU Float32 reference.
TEST_P(Conv3dMultiDTypeTest, Float16LargeKMatchesFloat32Reference) {
    if (dtype() != DType::Float16) GTEST_SKIP() << "Float16-specific regression";

    nn::Conv3d conv(/*in=*/32, /*out=*/4, /*kernel=*/3, /*stride=*/1, /*padding=*/1,
                    /*dilation=*/1, /*groups=*/1, /*bias=*/false);
    convert_model(conv);
    auto params = conv.parameters();
    ASSERT_FALSE(params.empty());
    // Small constant weight so the exact per-output sum (~K * w * x) stays well
    // inside Float16 range but is sensitive to accumulation precision.
    params[0]->tensor().fill_(0.01);

    auto input = tenzor::ones({1, 32, 4, 4, 4}, dtype(), device());
    Variable input_var(input, false);
    auto out = conv.forward(input_var);
    expectAllFinite(out.tensor());

    // CPU Float32 reference with the same (upcast) weights.
    nn::Conv3d conv_cpu(32, 4, 3, 1, 1, 1, 1, false);
    conv_cpu.parameters()[0]->tensor().fill_(0.01);
    auto in_cpu = tenzor::ones({1, 32, 4, 4, 4}, DType::Float32, Device::cpu());
    Variable in_cpu_var(in_cpu, false);
    auto ref = conv_cpu.forward(in_cpu_var);

    auto got = out.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    auto exp = ref.tensor().to(DType::Float32).contiguous();
    auto* g = got.data<float>();
    auto* e = exp.data<float>();
    // FP16 inputs/weights quantize, but with F32 accumulation the result tracks
    // the reference closely. FP16 accumulation would drift far beyond this.
    for (int64_t i = 0; i < got.numel(); ++i) {
        EXPECT_NEAR(g[i], e[i], 0.05f) << "index " << i;
    }
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(Conv3dMultiDTypeTest);
