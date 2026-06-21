#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/awq.hpp>
#include "../../backend_test_fixture.hpp"
#include "awq_dequant_helper.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn::quantization;

class AWQQuantizerTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // Reconstruct the AWQ result and assert every element is within the
    // analytic half-quant-step of the original weight. Returns the max abs
    // reconstruction error so callers can add their own loose sanity bound.
    static double assert_awq_roundtrip_within_step(
        const AWQResult& result, const Tensor& weight, int group_size) {
        auto recon = ::tenzor::testing::awq_detail::reconstruct_awq(result, group_size);
        auto w_cpu = weight.to(Device::cpu()).to(DType::Float32);
        const float* w_p = w_cpu.template data<float>();
        auto scales_cpu = result.scales.to(Device::cpu()).to(DType::Float32);
        auto act_cpu    = result.act_scales.to(Device::cpu()).to(DType::Float32);
        const float* scales_p = scales_cpu.template data<float>();
        const float* act_p    = act_cpu.template data<float>();
        const int64_t out_features = result.scales.size(0);
        const int64_t num_groups   = result.scales.size(1);
        const int64_t in_features  = result.in_features;
        double max_excess = 0.0, max_abs_err = 0.0;
        for (int64_t o = 0; o < out_features; ++o) {
            for (int64_t j = 0; j < in_features; ++j) {
                int64_t g = j / group_size;
                float scale = scales_p[o * num_groups + g];
                float aj    = act_p[j];
                double bound = 0.5 * static_cast<double>(scale) /
                    std::max(static_cast<double>(std::abs(aj)), 1e-12);
                double err = std::abs(
                    static_cast<double>(recon[o * in_features + j]) -
                    static_cast<double>(w_p[o * in_features + j]));
                max_abs_err = std::max(max_abs_err, err);
                max_excess = std::max(max_excess, err - bound);
            }
        }
        EXPECT_LE(max_excess, 1e-4)
            << "AWQ reconstruction exceeds analytic half-quant-step by "
            << max_excess;
        return max_abs_err;
    }
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST_P(AWQQuantizerTest, DefaultConstruction) {
    AWQQuantizer q;
    // Should not throw; default is 4-bit, group_size=128
}

TEST_P(AWQQuantizerTest, CustomBitsAndGroupSize) {
    AWQConfig config;
    config.bits = 8;
    config.group_size = 64;
    AWQQuantizer q(config);
    // Should not throw with non-default parameters
}

// ---------------------------------------------------------------------------
// compute_act_scales
// ---------------------------------------------------------------------------

TEST_P(AWQQuantizerTest, ComputeActScalesShape) {
    // activations: [batch=8, features=256]
    auto act = tenzor::randn({8, 256}, DType::Float32, device);
    auto scales = AWQQuantizer::compute_act_scales(act);

    // Scales should be per-feature, shape [256]
    ASSERT_EQ(scales.dim(), 1);
    EXPECT_EQ(scales.size(0), 256);
}

TEST_P(AWQQuantizerTest, ComputeActScalesNonNegative) {
    auto act = tenzor::randn({16, 64}, DType::Float32, device);
    auto scales = AWQQuantizer::compute_act_scales(act);

    auto scales_cpu = scales.to(Device::cpu());
    const float* data = scales_cpu.data<float>();
    for (int64_t i = 0; i < scales_cpu.numel(); ++i) {
        EXPECT_GE(data[i], 0.0f) << "Activation scale at index " << i << " is negative";
    }
}

// ---------------------------------------------------------------------------
// quantize_layer exercises the internal find_optimal_scales path
// ---------------------------------------------------------------------------

TEST_P(AWQQuantizerTest, QuantizeLayerBasicSmall) {
    AWQConfig config;
    config.bits = 4;
    config.group_size = 64;
    AWQQuantizer q(config);
    auto weight = tenzor::randn({32, 64}, DType::Float32, device);
    auto act_scales = AWQQuantizer::compute_act_scales(tenzor::randn({8, 64}, DType::Float32, device));

    auto result = q.quantize_layer(weight, act_scales);

    EXPECT_EQ(result.quantized_weight.size(0), 32);
    // The reconstructed weight must stay within the analytic INT4 step bound;
    // a quantizer that emits a correctly-shaped but wrong buffer fails here.
    assert_awq_roundtrip_within_step(result, weight, config.group_size);
}

// ---------------------------------------------------------------------------
// quantize_layer
// ---------------------------------------------------------------------------

TEST_P(AWQQuantizerTest, QuantizeLayerOutputShapes) {
    AWQConfig config;
    config.bits = 4;
    config.group_size = 128;
    AWQQuantizer q(config);
    auto weight = tenzor::randn({64, 128}, DType::Float32, device);
    auto scales = tenzor::abs(tenzor::randn({128}, DType::Float32, device)) + 0.1f;

    auto result = q.quantize_layer(weight, scales);

    // Quantized weight should have correct first dimension
    ASSERT_EQ(result.quantized_weight.dim(), 2);
    EXPECT_EQ(result.quantized_weight.size(0), 64);
    // Scales should also be produced
    EXPECT_GT(result.scales.numel(), 0);
    assert_awq_roundtrip_within_step(result, weight, config.group_size);
}

TEST_P(AWQQuantizerTest, QuantizeLayerRoundTripError) {
    AWQConfig config;
    config.bits = 4;
    config.group_size = 128;
    AWQQuantizer q(config);
    // Use small values to keep quantization error manageable
    auto weight = tenzor::randn({32, 128}, DType::Float32, device) * 0.1f;
    auto act = tenzor::randn({16, 128}, DType::Float32, device);
    auto act_scales = AWQQuantizer::compute_act_scales(act);

    auto result = q.quantize_layer(weight, act_scales);

    ASSERT_EQ(result.quantized_weight.dim(), 2);
    ASSERT_EQ(result.quantized_weight.size(0), 32);
    ASSERT_EQ(result.in_features, 128);

    // Reconstruct the dequantized weight from the packed INT4 codes + scales,
    // then bound the residual by the *analytic* per-element half-quant-step
    // (in the un-prescaled domain: 0.5 * scale[o,g] / act_scales[j]). A correct
    // symmetric INT4 quantizer cannot exceed this; a quantizer returning zeros,
    // NaNs, or garbage of the right shape blows past it. This is the only
    // assertion that actually verifies the round-trip the test is named for.
    auto recon = ::tenzor::testing::awq_detail::reconstruct_awq(result,
                                                                config.group_size);

    auto w_cpu = weight.to(Device::cpu()).to(DType::Float32);
    const float* w_p = w_cpu.template data<float>();

    auto scales_cpu = result.scales.to(Device::cpu()).to(DType::Float32);
    auto act_cpu    = result.act_scales.to(Device::cpu()).to(DType::Float32);
    const float* scales_p = scales_cpu.template data<float>();
    const float* act_p    = act_cpu.template data<float>();
    const int64_t out_features = result.scales.size(0);
    const int64_t num_groups   = result.scales.size(1);
    const int64_t in_features  = result.in_features;

    double max_excess = 0.0;
    double max_abs_err = 0.0;
    for (int64_t o = 0; o < out_features; ++o) {
        for (int64_t j = 0; j < in_features; ++j) {
            int64_t g = j / config.group_size;
            float scale = scales_p[o * num_groups + g];
            float aj    = act_p[j];
            // Half a quant step, mapped back through the per-channel prescale.
            double bound = 0.5 * static_cast<double>(scale) /
                           std::max(static_cast<double>(std::abs(aj)), 1e-12);
            double err = std::abs(static_cast<double>(recon[o * in_features + j]) -
                                  static_cast<double>(w_p[o * in_features + j]));
            max_abs_err = std::max(max_abs_err, err);
            max_excess = std::max(max_excess, err - bound);
        }
    }
    // Allow a tiny absolute slack for Float32 rounding in the reconstruction.
    EXPECT_LE(max_excess, 1e-4)
        << "AWQ reconstruction exceeds the analytic half-quant-step bound by "
        << max_excess << " (max abs err " << max_abs_err << ")";
    // And the reconstruction must actually be close (not trivially zero scale):
    // small-magnitude weights => sub-0.5 reconstruction error overall.
    EXPECT_LT(max_abs_err, 0.5)
        << "AWQ reconstruction max abs error " << max_abs_err
        << " is implausibly large for 0.1*randn weights";
}

// ---------------------------------------------------------------------------
// quantize_layer produces packed INT4 output (pack_int4 is internal)
// ---------------------------------------------------------------------------

TEST_P(AWQQuantizerTest, QuantizeLayerInt4ProducesPacked) {
    AWQConfig config;
    config.bits = 4;
    config.group_size = 128;
    AWQQuantizer q(config);
    auto weight = tenzor::randn({64, 128}, DType::Float32, device);
    auto scales = tenzor::abs(tenzor::randn({128}, DType::Float32, device)) + 0.1f;

    auto result = q.quantize_layer(weight, scales);

    // The quantized_weight from a 4-bit quantizer should be packed: the column
    // count must be ceil(in_features/2), not in_features.
    ASSERT_EQ(result.quantized_weight.dim(), 2);
    EXPECT_EQ(result.quantized_weight.size(0), 64);
    EXPECT_EQ(result.quantized_weight.size(1), (128 + 1) / 2)
        << "INT4 output should be nibble-packed (2 cols per byte)";
    EXPECT_EQ(result.quantized_weight.dtype(), DType::UInt8);
    assert_awq_roundtrip_within_step(result, weight, config.group_size);
}

TEST_P(AWQQuantizerTest, QuantizeLayerLargerWeight) {
    AWQConfig config;
    config.bits = 4;
    config.group_size = 128;
    AWQQuantizer q(config);
    auto weight = tenzor::randn({16, 256}, DType::Float32, device);
    auto scales = tenzor::abs(tenzor::randn({256}, DType::Float32, device)) + 0.1f;

    auto result = q.quantize_layer(weight, scales);

    EXPECT_EQ(result.quantized_weight.size(0), 16);
    EXPECT_GT(result.scales.numel(), 0);
    assert_awq_roundtrip_within_step(result, weight, config.group_size);
}

INSTANTIATE_BACKEND_TESTS(AWQQuantizerTest);
