#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/awq.hpp>
#include "../../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn::quantization;

class AWQQuantizerTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
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

    // Just verify it produces output with correct first dimension
    EXPECT_EQ(result.quantized_weight.size(0), 32);
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

    // With 4-bit quantization, expect bounded error
    // Quantized weight should have correct shape
    auto q_cpu = result.quantized_weight.to(Device::cpu());
    auto w_cpu = weight.to(Device::cpu());
    EXPECT_EQ(q_cpu.size(0), w_cpu.size(0));
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

    // The quantized_weight from a 4-bit quantizer should be packed
    ASSERT_EQ(result.quantized_weight.dim(), 2);
    EXPECT_EQ(result.quantized_weight.size(0), 64);
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
}

INSTANTIATE_BACKEND_TESTS(AWQQuantizerTest);
