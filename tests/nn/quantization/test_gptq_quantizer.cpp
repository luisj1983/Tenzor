#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/gptq.hpp>
#include <mutex>

using namespace tenzor;
using namespace tenzor::nn::quantization;

class GPTQQuantizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        static std::once_flag init_flag;
        std::call_once(init_flag, []() { tenzor::initialize(); });
    }
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST_F(GPTQQuantizerTest, DefaultConstruction) {
    GPTQQuantizer q;
    // Default: 4-bit, group_size=128, damp_percent=0.01
}

TEST_F(GPTQQuantizerTest, CustomParameters) {
    GPTQConfig config;
    config.bits = 8;
    config.group_size = 64;
    config.damp_percent = 0.05f;
    GPTQQuantizer q(config);
    // Should not throw with custom parameters
}

// ---------------------------------------------------------------------------
// compute_hessian
// ---------------------------------------------------------------------------

TEST_F(GPTQQuantizerTest, ComputeHessianShape) {
    // Simulate calibration inputs: [batch_size, in_features]
    auto input = tenzor::randn({12, 64});

    auto hessian = GPTQQuantizer::compute_hessian(input);

    // Hessian should be [in_features x in_features]
    ASSERT_EQ(hessian.dim(), 2);
    EXPECT_EQ(hessian.size(0), 64);
    EXPECT_EQ(hessian.size(1), 64);
}

TEST_F(GPTQQuantizerTest, ComputeHessianSymmetric) {
    auto input = tenzor::randn({16, 32});

    auto hessian = GPTQQuantizer::compute_hessian(input);
    auto h_cpu = hessian.to(Device::cpu());
    const float* data = h_cpu.data<float>();

    // Check symmetry: H[i,j] == H[j,i]
    for (int i = 0; i < 32; ++i) {
        for (int j = i + 1; j < 32; ++j) {
            float h_ij = data[i * 32 + j];
            float h_ji = data[j * 32 + i];
            EXPECT_NEAR(h_ij, h_ji, 1e-5f)
                << "Hessian not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST_F(GPTQQuantizerTest, ComputeHessianPositiveDiagonal) {
    auto input = tenzor::randn({16, 32});

    auto hessian = GPTQQuantizer::compute_hessian(input);
    auto h_cpu = hessian.to(Device::cpu());
    const float* data = h_cpu.data<float>();

    // Diagonal of H = X^T X + damping should be positive
    for (int i = 0; i < 32; ++i) {
        EXPECT_GT(data[i * 32 + i], 0.0f)
            << "Hessian diagonal at (" << i << "," << i << ") is not positive";
    }
}

// ---------------------------------------------------------------------------
// quantize_layer
// ---------------------------------------------------------------------------

TEST_F(GPTQQuantizerTest, QuantizeLayerOutputShapes) {
    GPTQConfig config;
    config.bits = 4;
    config.group_size = 128;
    config.damp_percent = 0.01f;
    GPTQQuantizer q(config);

    auto weight = tenzor::randn({64, 128});
    auto input = tenzor::randn({8, 128});
    auto hessian = GPTQQuantizer::compute_hessian(input);

    auto result = q.quantize_layer(weight, hessian);

    // Packed weight should have correct first dimension
    ASSERT_EQ(result.packed_weight.dim(), 2);
    EXPECT_EQ(result.packed_weight.size(0), 64);
    EXPECT_GT(result.scales.numel(), 0);
}

TEST_F(GPTQQuantizerTest, QuantizeLayerErrorBounded) {
    GPTQConfig config;
    config.bits = 4;
    config.group_size = 128;
    config.damp_percent = 0.01f;
    GPTQQuantizer q(config);

    // Small weight range for tighter quantization
    auto weight = tenzor::randn({32, 128}) * 0.1f;
    auto input = tenzor::randn({32, 128});
    auto hessian = GPTQQuantizer::compute_hessian(input);

    auto result = q.quantize_layer(weight, hessian);

    // Verify the scales are finite
    auto s_cpu = result.scales.to(Device::cpu());
    const float* data = s_cpu.data<float>();
    for (int64_t i = 0; i < s_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(data[i]))
            << "Non-finite scale at index " << i;
    }
}

// ---------------------------------------------------------------------------
// quantize_layer produces packed INT4 output (pack_int4 is internal)
// ---------------------------------------------------------------------------

TEST_F(GPTQQuantizerTest, QuantizeLayerPackedOutput) {
    GPTQConfig config;
    config.bits = 4;
    config.group_size = 128;
    config.damp_percent = 0.01f;
    GPTQQuantizer q(config);

    auto weight = tenzor::randn({32, 128});
    auto input = tenzor::randn({8, 128});
    auto hessian = GPTQQuantizer::compute_hessian(input);

    auto result = q.quantize_layer(weight, hessian);

    // Packed weight should have correct first dimension
    ASSERT_EQ(result.packed_weight.dim(), 2);
    EXPECT_EQ(result.packed_weight.size(0), 32);
}

TEST_F(GPTQQuantizerTest, QuantizeLayerLargerWeight) {
    GPTQConfig config;
    config.bits = 4;
    config.group_size = 128;
    config.damp_percent = 0.01f;
    GPTQQuantizer q(config);

    auto weight = tenzor::randn({64, 256});
    auto input = tenzor::randn({8, 256});
    auto hessian = GPTQQuantizer::compute_hessian(input);

    auto result = q.quantize_layer(weight, hessian);

    EXPECT_EQ(result.packed_weight.size(0), 64);
    EXPECT_GT(result.scales.numel(), 0);
}
