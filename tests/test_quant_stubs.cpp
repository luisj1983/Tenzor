/**
 * @file test_quant_stubs.cpp
 * @brief Unit tests for QuantStub and DeQuantStub quantization layers
 *
 * Tests full INT8 quantization and dequantization functionality including:
 * - Symmetric and asymmetric quantization
 * - Per-tensor and per-channel quantization
 * - Proper rounding and clamping
 * - Forward and backward compatibility
 */

#include <gtest/gtest.h>
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/quantization/quantize.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::quantization;

class QuantStubTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test data
        test_input_ = Tensor({4, 8}, DType::Float32, Device::cpu());
        float* data = test_input_.data<float>();

        // Fill with values in range [-10, 10]
        for (int i = 0; i < 32; ++i) {
            data[i] = -10.0f + (i * 20.0f / 31.0f);
        }
    }

    Tensor test_input_;
};

// ============================================================================
// QuantStub Tests - Symmetric Quantization
// ============================================================================

TEST_F(QuantStubTest, SymmetricPerTensorQuantization) {
    // Create quantization parameters for symmetric INT8
    Tensor scale({1}, DType::Float32, Device::cpu());
    Tensor zero_point({1}, DType::Int32, Device::cpu());

    scale.fill_(10.0f / 127.0f);  // Scale for [-10, 10] range
    zero_point.fill_(0);           // Symmetric uses zero_point = 0

    QuantizationParams qparams(scale, zero_point, QuantDType::INT8,
                              QuantizationScheme::PerTensorSymmetric);

    // Create QuantStub
    auto quant_stub = QuantStub(qparams);

    // Test forward_to_quantized
    QuantizedTensor q_output = quant_stub.forward_to_quantized(test_input_);

    // Verify quantized tensor properties
    EXPECT_EQ(q_output.data().dtype(), DType::Int8);
    EXPECT_EQ(q_output.shape().size(), 2);
    EXPECT_EQ(q_output.shape()[0], 4);
    EXPECT_EQ(q_output.shape()[1], 8);

    // Verify quantization parameters
    const auto& params = q_output.params();
    EXPECT_EQ(params.dtype, QuantDType::INT8);
    EXPECT_EQ(params.scheme, QuantizationScheme::PerTensorSymmetric);
    EXPECT_EQ(params.axis, -1);  // Per-tensor

    // Check zero_point is 0 for symmetric
    EXPECT_EQ(params.zero_point.data<const int32_t>()[0], 0);

    // Verify quantized values are in valid INT8 range
    const int8_t* q_data = q_output.data().data<const int8_t>();
    for (int i = 0; i < 32; ++i) {
        EXPECT_GE(q_data[i], -127);
        EXPECT_LE(q_data[i], 127);
    }
}

TEST_F(QuantStubTest, SymmetricQuantizationAccuracy) {
    // Test that quantization error is within acceptable bounds
    auto qparams = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(-10.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(10.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    auto quant_stub = QuantStub(qparams);
    QuantizedTensor q_output = quant_stub.forward_to_quantized(test_input_);

    // Dequantize and compare
    Tensor dequantized = q_output.dequantize();
    const float* orig_data = test_input_.data<const float>();
    const float* deq_data = dequantized.data<const float>();

    float max_error = 0.0f;
    for (int i = 0; i < 32; ++i) {
        float error = std::abs(orig_data[i] - deq_data[i]);
        max_error = std::max(max_error, error);
    }

    // Error should be less than scale (quantization step size)
    float scale = qparams.scale.data<const float>()[0];
    EXPECT_LT(max_error, scale * 1.5f);  // Allow 1.5x scale for rounding
}

TEST_F(QuantStubTest, ForwardPassWithVariable) {
    // Test Variable wrapper compatibility
    auto qparams = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(-10.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(10.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    auto quant_stub = QuantStub(qparams);
    Variable input_var(test_input_, false);

    // Forward should quantize then dequantize
    Variable output_var = quant_stub.forward(input_var);

    EXPECT_EQ(output_var.tensor().dtype(), DType::Float32);
    EXPECT_FALSE(output_var.requires_grad());
}

// ============================================================================
// QuantStub Tests - Asymmetric Quantization
// ============================================================================

TEST_F(QuantStubTest, AsymmetricPerTensorQuantization) {
    // Create asymmetric quantization parameters
    auto qparams = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(-10.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(10.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorAsymmetric
    );

    auto quant_stub = QuantStub(qparams);
    QuantizedTensor q_output = quant_stub.forward_to_quantized(test_input_);

    // Verify scheme
    EXPECT_EQ(q_output.params().scheme, QuantizationScheme::PerTensorAsymmetric);

    // For asymmetric, zero_point can be non-zero
    int32_t zp = q_output.params().zero_point.data<const int32_t>()[0];
    EXPECT_GE(zp, -128);
    EXPECT_LE(zp, 127);
}

TEST_F(QuantStubTest, UINT8Quantization) {
    // Test UINT8 quantization
    Tensor positive_input({2, 4}, DType::Float32, Device::cpu());
    float* data = positive_input.data<float>();
    for (int i = 0; i < 8; ++i) {
        data[i] = i * 2.0f;  // Values [0, 14]
    }

    auto qparams = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(0.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(14.0f),
        QuantDType::UINT8,
        QuantizationScheme::PerTensorAsymmetric
    );

    auto quant_stub = QuantStub(qparams);
    QuantizedTensor q_output = quant_stub.forward_to_quantized(positive_input);

    // Verify UINT8 dtype
    EXPECT_EQ(q_output.data().dtype(), DType::UInt8);
    EXPECT_EQ(q_output.params().dtype, QuantDType::UINT8);

    // Verify values in UINT8 range
    const uint8_t* q_data = q_output.data().data<const uint8_t>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_GE(q_data[i], 0);
        EXPECT_LE(q_data[i], 255);
    }
}

// ============================================================================
// QuantStub Tests - Per-Channel Quantization
// ============================================================================

TEST_F(QuantStubTest, PerChannelSymmetricQuantization) {
    // Create per-channel quantization parameters
    Tensor weights({4, 8}, DType::Float32, Device::cpu());

    auto q_weights = quantize_per_channel_symmetric(weights, 0, QuantDType::INT8);
    const auto& qparams = q_weights.params();

    auto quant_stub = QuantStub(qparams);

    // Verify per-channel properties
    EXPECT_TRUE(quant_stub.is_per_channel());
    EXPECT_TRUE(quant_stub.is_symmetric());
    EXPECT_EQ(qparams.axis, 0);

    // Scale and zero_point should have one value per channel
    EXPECT_EQ(qparams.scale.numel(), 4);
    EXPECT_EQ(qparams.zero_point.numel(), 4);
}

// ============================================================================
// DeQuantStub Tests
// ============================================================================

TEST_F(QuantStubTest, DeQuantStubBasic) {
    // Create quantized tensor
    auto qparams = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(-10.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(10.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    auto quant_stub = QuantStub(qparams);
    QuantizedTensor q_tensor = quant_stub.forward_to_quantized(test_input_);

    // Dequantize
    auto dequant_stub = DeQuantStub();
    Tensor dequantized = dequant_stub.forward_from_quantized(q_tensor);

    // Verify output is Float32
    EXPECT_EQ(dequantized.dtype(), DType::Float32);
    EXPECT_EQ(dequantized.shape().size(), 2);
    EXPECT_EQ(dequantized.shape()[0], 4);
    EXPECT_EQ(dequantized.shape()[1], 8);

    // Check last_was_per_channel flag
    EXPECT_FALSE(dequant_stub.last_was_per_channel());
}

TEST_F(QuantStubTest, DeQuantStubPerChannel) {
    // Test per-channel dequantization tracking
    Tensor weights({4, 8}, DType::Float32, Device::cpu());
    auto q_weights = quantize_per_channel_symmetric(weights, 0, QuantDType::INT8);

    auto dequant_stub = DeQuantStub();
    Tensor dequantized = dequant_stub.forward_from_quantized(q_weights);

    // Check per-channel flag is set
    EXPECT_TRUE(dequant_stub.last_was_per_channel());
}

TEST_F(QuantStubTest, RoundTripQuantization) {
    // Test full quantize -> dequantize round trip
    auto qparams = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(-10.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(10.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    auto quant_stub = QuantStub(qparams);
    auto dequant_stub = DeQuantStub();

    // Quantize
    QuantizedTensor q_tensor = quant_stub.forward_to_quantized(test_input_);

    // Dequantize
    Tensor output = dequant_stub.forward_from_quantized(q_tensor);

    // Compare with original
    const float* orig_data = test_input_.data<const float>();
    const float* out_data = output.data<const float>();

    for (int i = 0; i < 32; ++i) {
        float error = std::abs(orig_data[i] - out_data[i]);
        // Error should be within quantization step
        EXPECT_LT(error, 0.2f);  // ~2% error for [-10, 10] range
    }
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(QuantStubTest, InvalidInputType) {
    // Test that non-float input raises error
    Tensor int_input({2, 2}, DType::Int32, Device::cpu());

    auto qparams = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(0.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(1.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    auto quant_stub = QuantStub(qparams);

    EXPECT_THROW({
        quant_stub.forward_to_quantized(int_input);
    }, std::runtime_error);
}

TEST_F(QuantStubTest, ZeroRangeInput) {
    // Test quantization with all identical values
    Tensor constant_input({2, 2}, DType::Float32, Device::cpu());
    constant_input.fill_(5.0f);

    auto qparams = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(5.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(5.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    auto quant_stub = QuantStub(qparams);

    // Should not crash
    QuantizedTensor q_output = quant_stub.forward_to_quantized(constant_input);

    // Dequantize should give back approximately same values
    Tensor dequantized = q_output.dequantize();
    const float* deq_data = dequantized.data<const float>();

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(deq_data[i], 5.0f, 0.1f);
    }
}

TEST_F(QuantStubTest, ParameterUpdate) {
    // Test that qparams can be updated
    auto qparams1 = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(-10.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(10.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    auto quant_stub = QuantStub(qparams1);

    // Update parameters
    auto qparams2 = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(-5.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(5.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorAsymmetric
    );

    quant_stub.set_qparams(qparams2);

    // Verify updated
    const auto& params = quant_stub.qparams();
    EXPECT_EQ(params.scheme, QuantizationScheme::PerTensorAsymmetric);
}

TEST_F(QuantStubTest, SchemeDetection) {
    // Test is_symmetric and is_per_channel methods
    auto sym_params = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(-1.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(1.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    auto quant_stub_sym = QuantStub(sym_params);
    EXPECT_TRUE(quant_stub_sym.is_symmetric());
    EXPECT_FALSE(quant_stub_sym.is_per_channel());

    // Test asymmetric
    auto asym_params = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(0.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(10.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorAsymmetric
    );

    auto quant_stub_asym = QuantStub(asym_params);
    EXPECT_FALSE(quant_stub_asym.is_symmetric());
    EXPECT_FALSE(quant_stub_asym.is_per_channel());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(QuantStubTest, ModelInputOutput) {
    // Simulate model with QuantStub at input and DeQuantStub at output

    // Model input layer
    auto input_qparams = compute_quantization_params(
        Tensor({1}, DType::Float32, Device::cpu()).fill_(-1.0f),
        Tensor({1}, DType::Float32, Device::cpu()).fill_(1.0f),
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );
    auto quant_stub = std::make_shared<QuantStub>(input_qparams);

    // Model output layer
    auto dequant_stub = std::make_shared<DeQuantStub>();

    // Simulate inference
    Tensor model_input({1, 10}, DType::Float32, Device::cpu());
    float* data = model_input.data<float>();
    for (int i = 0; i < 10; ++i) {
        data[i] = -1.0f + (i * 2.0f / 9.0f);
    }

    // Forward through stubs
    QuantizedTensor q_input = quant_stub->forward_to_quantized(model_input);
    // ... quantized operations ...
    Tensor model_output = dequant_stub->forward_from_quantized(q_input);

    // Verify shapes match
    EXPECT_EQ(model_output.shape()[0], model_input.shape()[0]);
    EXPECT_EQ(model_output.shape()[1], model_input.shape()[1]);

    // Verify reasonable accuracy
    const float* in_data = model_input.data<const float>();
    const float* out_data = model_output.data<const float>();

    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(in_data[i], out_data[i], 0.05f);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
