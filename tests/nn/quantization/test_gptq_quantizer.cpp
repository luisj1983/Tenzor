#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/gptq.hpp>
#include "../../backend_test_fixture.hpp"
#include "gptq_dequant_helper.hpp"

#include <cmath>

using namespace tenzor;
using namespace tenzor::nn::quantization;

class GPTQQuantizerTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST_P(GPTQQuantizerTest, DefaultConstruction) {
    GPTQQuantizer q;
    // Default: 4-bit, group_size=128, damp_percent=0.01
}

TEST_P(GPTQQuantizerTest, CustomParameters) {
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

TEST_P(GPTQQuantizerTest, ComputeHessianShape) {
    // Simulate calibration inputs: [batch_size, in_features]
    auto input = tenzor::randn({12, 64}, DType::Float32, device);

    auto hessian = GPTQQuantizer::compute_hessian(input);

    // Hessian should be [in_features x in_features]
    ASSERT_EQ(hessian.dim(), 2);
    EXPECT_EQ(hessian.size(0), 64);
    EXPECT_EQ(hessian.size(1), 64);
}

TEST_P(GPTQQuantizerTest, ComputeHessianSymmetric) {
    auto input = tenzor::randn({16, 32}, DType::Float32, device);

    auto hessian = GPTQQuantizer::compute_hessian(input);
    auto h_cpu = hessian.cpu();
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

TEST_P(GPTQQuantizerTest, ComputeHessianPositiveDiagonal) {
    auto input = tenzor::randn({16, 32}, DType::Float32, device);

    auto hessian = GPTQQuantizer::compute_hessian(input);
    auto h_cpu = hessian.cpu();
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

TEST_P(GPTQQuantizerTest, QuantizeLayerOutputShapes) {
    GPTQConfig config;
    config.bits = 4;
    config.group_size = 128;
    config.damp_percent = 0.01f;
    GPTQQuantizer q(config);

    auto weight = tenzor::randn({64, 128}, DType::Float32, device);
    auto input = tenzor::randn({8, 128}, DType::Float32, device);
    auto hessian = GPTQQuantizer::compute_hessian(input);

    auto result = q.quantize_layer(weight, hessian);

    // Packed weight should have correct first dimension and be INT4-packed.
    ASSERT_EQ(result.packed_weight.dim(), 2);
    EXPECT_EQ(result.packed_weight.size(0), 64);
    EXPECT_EQ(result.packed_weight.size(1), (128 + 1) / 2);
    EXPECT_GT(result.scales.numel(), 0);

    // Round-trip: reconstruction must be far from the zero/garbage baseline.
    auto recon = ::tenzor::testing::gptq_detail::reconstruct_gptq(
        result, config.group_size, config.sym);
    double rel_err = ::tenzor::testing::gptq_detail::relative_frobenius_error(
        recon, weight);
    EXPECT_LT(rel_err, 0.40)
        << "GPTQ INT4 relative reconstruction error " << rel_err;
}

TEST_P(GPTQQuantizerTest, QuantizeLayerErrorBounded) {
    GPTQConfig config;
    config.bits = 4;
    config.group_size = 128;
    config.damp_percent = 0.01f;
    GPTQQuantizer q(config);

    // Small weight range for tighter quantization
    auto weight = tenzor::randn({32, 128}, DType::Float32, device) * 0.1f;
    auto input = tenzor::randn({32, 128}, DType::Float32, device);
    auto hessian = GPTQQuantizer::compute_hessian(input);

    auto result = q.quantize_layer(weight, hessian);
    ASSERT_EQ(result.in_features, 128);

    // Verify the scales are finite
    auto s_cpu = result.scales.cpu();
    const float* data = s_cpu.data<float>();
    for (int64_t i = 0; i < s_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(data[i]))
            << "Non-finite scale at index " << i;
    }

    // Actually measure the round-trip error the test is named for: dequantize
    // the packed INT4 weight and bound the relative reconstruction error.
    // GPTQ error-compensates, so the per-element half-step bound does not hold;
    // the meaningful, name-matching quantity is the aggregate relative error,
    // which a correct quantizer keeps well below 1 (returning zeros => 1.0).
    auto recon = ::tenzor::testing::gptq_detail::reconstruct_gptq(
        result, config.group_size, config.sym);
    double rel_err = ::tenzor::testing::gptq_detail::relative_frobenius_error(
        recon, weight);
    EXPECT_LT(rel_err, 0.30)
        << "GPTQ INT4 relative reconstruction error " << rel_err
        << " too large — quantizer likely broken (zeros/garbage gives ~1.0)";
}

// ---------------------------------------------------------------------------
// quantize_layer produces packed INT4 output (pack_int4 is internal)
// ---------------------------------------------------------------------------

TEST_P(GPTQQuantizerTest, QuantizeLayerPackedOutput) {
    GPTQConfig config;
    config.bits = 4;
    config.group_size = 128;
    config.damp_percent = 0.01f;
    GPTQQuantizer q(config);

    auto weight = tenzor::randn({32, 128}, DType::Float32, device);
    auto input = tenzor::randn({8, 128}, DType::Float32, device);
    auto hessian = GPTQQuantizer::compute_hessian(input);

    auto result = q.quantize_layer(weight, hessian);

    // Packed weight should have correct first dimension
    ASSERT_EQ(result.packed_weight.dim(), 2);
    EXPECT_EQ(result.packed_weight.size(0), 32);
    EXPECT_EQ(result.packed_weight.size(1), (128 + 1) / 2)
        << "INT4 output should be nibble-packed (2 cols per byte)";
    EXPECT_EQ(result.packed_weight.dtype(), DType::UInt8);

    auto recon = ::tenzor::testing::gptq_detail::reconstruct_gptq(
        result, config.group_size, config.sym);
    double rel_err = ::tenzor::testing::gptq_detail::relative_frobenius_error(
        recon, weight);
    EXPECT_LT(rel_err, 0.40)
        << "GPTQ INT4 relative reconstruction error " << rel_err;
}

TEST_P(GPTQQuantizerTest, QuantizeLayerLargerWeight) {
    GPTQConfig config;
    config.bits = 4;
    config.group_size = 128;
    config.damp_percent = 0.01f;
    GPTQQuantizer q(config);

    auto weight = tenzor::randn({64, 256}, DType::Float32, device);
    auto input = tenzor::randn({8, 256}, DType::Float32, device);
    auto hessian = GPTQQuantizer::compute_hessian(input);

    auto result = q.quantize_layer(weight, hessian);

    EXPECT_EQ(result.packed_weight.size(0), 64);
    EXPECT_GT(result.scales.numel(), 0);
    ASSERT_EQ(result.in_features, 256);

    auto recon = ::tenzor::testing::gptq_detail::reconstruct_gptq(
        result, config.group_size, config.sym);
    double rel_err = ::tenzor::testing::gptq_detail::relative_frobenius_error(
        recon, weight);
    EXPECT_LT(rel_err, 0.40)
        << "GPTQ INT4 relative reconstruction error " << rel_err;
}

INSTANTIATE_BACKEND_TESTS(GPTQQuantizerTest);
