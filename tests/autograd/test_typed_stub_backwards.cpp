/**
 * @file test_typed_stub_backwards.cpp
 * @brief S15: validate that the 5 typed-stub Backward classes — previously
 *        throwing NonDifferentiable — now compute correct gradients.
 *
 * Tests:
 *   - ROIAlignBackward
 *   - DeformableConv2dBackward (smoke test; full backward path heavy)
 *   - MelScaleBackward
 *   - DCTBackward (DCT-II ortho round-trip)
 *   - MFCCBackward
 *
 * For each: instantiate Function with config, run forward, capture output
 * Variable, run backward(), assert grad is finite and has reasonable
 * magnitude (not NaN, not all-zero).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/function.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/fft.hpp>
#include <tenzor/utils/error.hpp>
#include <cmath>

using namespace tenzor;

namespace {

bool tensor_is_finite_and_nonzero(const Tensor& t) {
    auto d = t.to(DType::Float64).contiguous();
    const double* p = d.data<double>();
    bool any_nonzero = false;
    for (int64_t i = 0; i < d.numel(); ++i) {
        if (!std::isfinite(p[i])) return false;
        if (std::abs(p[i]) > 1e-12) any_nonzero = true;
    }
    return any_nonzero;
}

}  // namespace

class TypedStubBackwardsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

// ---- ROIAlignBackward ----------------------------------------------------
TEST_F(TypedStubBackwardsTest, ROIAlignBackward_ComputesFiniteGrad) {
    // Tiny feature map [N=1, C=2, H=4, W=4]
    auto features = randn({1, 2, 4, 4}, DType::Float32, Device::cpu());
    // One ROI: [batch_idx, x1, y1, x2, y2]
    std::vector<float> rois_data = {0.0f, 0.0f, 0.0f, 3.0f, 3.0f};
    auto rois = Tensor::from_blob(rois_data.data(), {1, 5},
                                  DType::Float32, Device::cpu()).clone();

    auto fn = std::make_shared<ROIAlignBackward>(
        /*output_h=*/2, /*output_w=*/2,
        /*spatial_scale=*/1.0, /*sampling_ratio=*/2, /*aligned=*/true);

    Variable feat_v(features, /*requires_grad=*/true);
    Variable rois_v(rois, /*requires_grad=*/false);
    auto outs = fn->forward({feat_v, rois_v});
    ASSERT_EQ(outs.size(), 1u);
    // Build grad_outputs as ones-like the output.
    auto out_t = outs[0].tensor();
    auto grad_out = full(
        std::vector<int64_t>(out_t.shape().begin(), out_t.shape().end()),
        1.0, out_t.dtype(), out_t.device());

    auto grads = fn->backward({grad_out});
    ASSERT_GE(grads.size(), 1u);
    // grads[0] is the gradient for features.
    EXPECT_EQ(grads[0].numel(), features.numel());
    EXPECT_TRUE(tensor_is_finite_and_nonzero(grads[0]));
}

// ---- DCTBackward ---------------------------------------------------------
TEST_F(TypedStubBackwardsTest, DCTBackward_ComputesFiniteGrad) {
    auto x = randn({2, 8}, DType::Float32, Device::cpu());
    auto fn = std::make_shared<DCTBackward>(
        /*type=*/2, /*n=*/-1, /*dim=*/-1, "ortho");
    Variable x_v(x, /*requires_grad=*/true);
    auto outs = fn->forward({x_v});
    ASSERT_EQ(outs.size(), 1u);
    auto out_t = outs[0].tensor();
    auto grad_out = full(
        std::vector<int64_t>(out_t.shape().begin(), out_t.shape().end()),
        1.0, out_t.dtype(), out_t.device());
    auto grads = fn->backward({grad_out});
    ASSERT_EQ(grads.size(), 1u);
    EXPECT_EQ(grads[0].numel(), x.numel());
    EXPECT_TRUE(tensor_is_finite_and_nonzero(grads[0]));
}

// ---- MelScaleBackward ----------------------------------------------------
TEST_F(TypedStubBackwardsTest, MelScaleBackward_ComputesFiniteGrad) {
    // mel_scale expects spectrogram of shape (..., n_freqs, time)
    int64_t n_freqs = 9;  // n_fft = (n_freqs - 1) * 2 = 16
    int64_t time = 4;
    auto spec = abs(randn({n_freqs, time}, DType::Float32, Device::cpu())) + 0.1f;
    auto fn = std::make_shared<MelScaleBackward>(
        /*n_mels=*/4, /*f_min=*/0.0, /*f_max=*/0.0, /*sample_rate=*/16000);

    Variable spec_v(spec, /*requires_grad=*/true);
    auto outs = fn->forward({spec_v});
    ASSERT_EQ(outs.size(), 1u);
    auto out_t = outs[0].tensor();
    auto grad_out = full(
        std::vector<int64_t>(out_t.shape().begin(), out_t.shape().end()),
        1.0, out_t.dtype(), out_t.device());
    auto grads = fn->backward({grad_out});
    ASSERT_EQ(grads.size(), 1u);
    EXPECT_EQ(grads[0].numel(), spec.numel());
    EXPECT_TRUE(tensor_is_finite_and_nonzero(grads[0]));
}

// ---- MFCCBackward --------------------------------------------------------
TEST_F(TypedStubBackwardsTest, MFCCBackward_ComputesFiniteGrad) {
    // Use a small waveform; n_fft chosen small for fast test.
    int64_t n_fft = 16;
    int64_t hop = 4;
    int64_t length = 64;
    auto wave = randn({length}, DType::Float32, Device::cpu());
    auto fn = std::make_shared<MFCCBackward>(
        /*sample_rate=*/16000,
        /*n_mfcc=*/4,
        /*n_mels=*/8,
        n_fft,
        hop,
        /*f_min=*/0.0,
        /*f_max=*/0.0);
    Variable wave_v(wave, /*requires_grad=*/true);
    auto outs = fn->forward({wave_v});
    ASSERT_EQ(outs.size(), 1u);
    auto out_t = outs[0].tensor();
    auto grad_out = full(
        std::vector<int64_t>(out_t.shape().begin(), out_t.shape().end()),
        1.0, out_t.dtype(), out_t.device());
    auto grads = fn->backward({grad_out});
    ASSERT_EQ(grads.size(), 1u);
    EXPECT_EQ(grads[0].numel(), wave.numel());
    // Allow zero on the boundary samples (overlap-add structure); just check finiteness.
    auto d = grads[0].to(DType::Float64).contiguous();
    const double* p = d.data<double>();
    for (int64_t i = 0; i < d.numel(); ++i) {
        ASSERT_TRUE(std::isfinite(p[i])) << "MFCC backward produced NaN/Inf at " << i;
    }
}

// ---- DeformableConv2dBackward -------------------------------------------
// Exercises constructor + forward + backward end-to-end. Previously this
// SIGSEGV'd because the dispatch lambda indexed inputs[3]/[4] past the input
// span when bias/mask were omitted (std::span has no bounds checking) — NOT
// an MKL bug. The forward now normalises to the canonical 5-tensor
// {input,offset,weight,bias,mask} layout and the dispatch lambdas guard
// optional trailing inputs.

// Case 1: no bias, no mask (DCNv1 with zero offsets) — the original crash.
TEST_F(TypedStubBackwardsTest, DeformableConv2dBackward_NoBiasNoMask) {
    int64_t N = 1, Cin = 2, H = 5, W = 5;
    int64_t Cout = 2, kH = 3, kW = 3;
    auto input  = randn({N, Cin, H, W}, DType::Float32, Device::cpu());
    auto offset = randn({N, 2 * kH * kW, H, W}, DType::Float32, Device::cpu()) * 0.0f;
    auto weight = randn({Cout, Cin, kH, kW}, DType::Float32, Device::cpu());

    auto fn = std::make_shared<DeformableConv2dBackward>(
        /*stride_h=*/1, /*stride_w=*/1, /*pad_h=*/1, /*pad_w=*/1,
        /*dilation_h=*/1, /*dilation_w=*/1, /*groups=*/1, /*offset_groups=*/1,
        /*use_mask=*/false, /*has_bias=*/false);

    std::vector<Variable> ins = {
        Variable(input,  /*requires_grad=*/true),
        Variable(offset, /*requires_grad=*/true),
        Variable(weight, /*requires_grad=*/true)};

    auto outs = fn->forward(ins);
    ASSERT_EQ(outs.size(), 1u);
    auto out_t = outs[0].tensor();
    ASSERT_EQ(out_t.shape().size(), 4u);
    EXPECT_EQ(out_t.shape()[0], N);
    EXPECT_EQ(out_t.shape()[1], Cout);

    auto grad_out = full(
        std::vector<int64_t>(out_t.shape().begin(), out_t.shape().end()),
        1.0, out_t.dtype(), out_t.device());
    auto grads = fn->backward({grad_out});
    ASSERT_GE(grads.size(), 3u);  // dinput, doffset, dweight
    for (const auto& g : grads) {
        ASSERT_TRUE(g.is_valid());
        auto* p = g.data<float>();
        for (int64_t i = 0; i < g.numel(); ++i) {
            ASSERT_TRUE(std::isfinite(p[i])) << "deformable backward grad NaN/Inf";
        }
    }
}

// Case 2: with bias and with mask (full DCNv2) — verifies the canonical
// {input,offset,weight,bias,mask} ordering survives forward+backward.
TEST_F(TypedStubBackwardsTest, DeformableConv2dBackward_WithBiasAndMask) {
    int64_t N = 1, Cin = 2, H = 5, W = 5;
    int64_t Cout = 3, kH = 3, kW = 3;
    auto input  = randn({N, Cin, H, W}, DType::Float32, Device::cpu());
    auto offset = randn({N, 2 * kH * kW, H, W}, DType::Float32, Device::cpu()) * 0.0f;
    auto weight = randn({Cout, Cin, kH, kW}, DType::Float32, Device::cpu());
    auto bias   = randn({Cout}, DType::Float32, Device::cpu());
    // DCNv2 modulation mask: (N, kH*kW, H, W), values in (0,1) via 0.5 const.
    auto mask   = full({N, kH * kW, H, W}, 0.5, DType::Float32, Device::cpu());

    auto fn = std::make_shared<DeformableConv2dBackward>(
        /*stride_h=*/1, /*stride_w=*/1, /*pad_h=*/1, /*pad_w=*/1,
        /*dilation_h=*/1, /*dilation_w=*/1, /*groups=*/1, /*offset_groups=*/1,
        /*use_mask=*/true, /*has_bias=*/true);

    // Canonical input order: input, offset, weight, bias, mask.
    std::vector<Variable> ins = {
        Variable(input,  /*requires_grad=*/true),
        Variable(offset, /*requires_grad=*/true),
        Variable(weight, /*requires_grad=*/true),
        Variable(bias,   /*requires_grad=*/true),
        Variable(mask,   /*requires_grad=*/true)};

    auto outs = fn->forward(ins);
    ASSERT_EQ(outs.size(), 1u);
    auto out_t = outs[0].tensor();
    EXPECT_EQ(out_t.shape()[1], Cout);

    auto grad_out = full(
        std::vector<int64_t>(out_t.shape().begin(), out_t.shape().end()),
        1.0, out_t.dtype(), out_t.device());
    auto grads = fn->backward({grad_out});
    // dinput, doffset, dweight, dmask, dbias (mask before bias in output pack).
    ASSERT_GE(grads.size(), 5u);
    for (const auto& g : grads) {
        ASSERT_TRUE(g.is_valid());
        auto* p = g.data<float>();
        for (int64_t i = 0; i < g.numel(); ++i) {
            ASSERT_TRUE(std::isfinite(p[i])) << "deformable backward grad NaN/Inf";
        }
    }
}
