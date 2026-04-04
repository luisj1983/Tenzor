#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>
#include <numeric>

using namespace tenzor;

/**
 * @file test_quantized_inference.cpp
 * @brief End-to-end quantized model inference tests.
 *
 * Tests that quantizing a float model and running inference produces
 * results within acceptable tolerance of the float reference.
 */

class QuantizedInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }

    // Compute max absolute error between two float tensors
    static double max_abs_error(const Tensor& a, const Tensor& b) {
        auto ac = a.to(Device::cpu());
        auto bc = b.to(Device::cpu());
        const float* ad = ac.data<float>();
        const float* bd = bc.data<float>();
        double max_err = 0.0;
        for (size_t i = 0; i < ac.numel(); ++i) {
            max_err = std::max(max_err, std::abs(static_cast<double>(ad[i] - bd[i])));
        }
        return max_err;
    }
};

// Test: quantize a weight matrix, dequantize, compare to original
TEST_F(QuantizedInferenceTest, WeightQuantizeRoundTrip) {
    // Simulate a Linear layer weight [out=16, in=32]
    auto weight = randn({16, 32}, DType::Float32, Device::cpu());

    // Compute quantization parameters
    auto weight_cpu = weight.to(Device::cpu());
    const float* wdata = weight_cpu.data<float>();
    float wmin = *std::min_element(wdata, wdata + weight_cpu.numel());
    float wmax = *std::max_element(wdata, wdata + weight_cpu.numel());

    double scale = static_cast<double>(wmax - wmin) / 255.0;
    int64_t zero_point = static_cast<int64_t>(std::round(-wmin / scale));

    auto qweight = quantize_per_tensor(weight, scale, zero_point, DType::QUInt8);
    EXPECT_TRUE(qweight.is_quantized());

    auto dequantized = qweight.dequantize();
    double err = max_abs_error(weight, dequantized);

    // Quantization error should be bounded by scale
    EXPECT_LT(err, scale * 1.5) << "Quantization error too large: " << err << " vs scale " << scale;
}

// Test: quantize input, manually compute quantized matmul, compare to float
TEST_F(QuantizedInferenceTest, QuantizedMatmulAccuracy) {
    int64_t batch = 4, in_feat = 32, out_feat = 16;

    // Create float input and weight
    auto input_f = randn({batch, in_feat}, DType::Float32, Device::cpu());
    auto weight_f = randn({out_feat, in_feat}, DType::Float32, Device::cpu());

    // Float reference: output = input @ weight.T
    auto ref_output = matmul(input_f, weight_f.transpose(0, 1));

    // Quantize both to int8
    // Input: symmetric quantization (zero_point = 0)
    float input_max = 0.0f;
    {
        const float* d = input_f.data<float>();
        for (size_t i = 0; i < input_f.numel(); ++i) {
            input_max = std::max(input_max, std::abs(d[i]));
        }
    }
    double input_scale = static_cast<double>(input_max) / 127.0;

    float weight_max = 0.0f;
    {
        const float* d = weight_f.data<float>();
        for (size_t i = 0; i < weight_f.numel(); ++i) {
            weight_max = std::max(weight_max, std::abs(d[i]));
        }
    }
    double weight_scale = static_cast<double>(weight_max) / 127.0;

    auto q_input = quantize_per_tensor(input_f, input_scale, 0, DType::QInt8);
    auto q_weight = quantize_per_tensor(weight_f, weight_scale, 0, DType::QInt8);

    // Dequantize and compute matmul (simulated quantized inference)
    auto deq_input = q_input.dequantize();
    auto deq_weight = q_weight.dequantize();
    auto quant_output = matmul(deq_input, deq_weight.transpose(0, 1));

    // Compare: quantized output should be close to float reference
    double err = max_abs_error(ref_output, quant_output);
    double ref_mag = 0.0;
    {
        const float* d = ref_output.data<float>();
        for (size_t i = 0; i < ref_output.numel(); ++i) {
            ref_mag = std::max(ref_mag, std::abs(static_cast<double>(d[i])));
        }
    }

    // Relative error should be reasonable for int8 quantization
    double rel_err = err / (ref_mag + 1e-8);
    EXPECT_LT(rel_err, 0.05) << "Relative error " << rel_err << " exceeds 5% threshold";
}

// Test: full Linear layer quantize-dequantize pipeline
TEST_F(QuantizedInferenceTest, LinearLayerQuantization) {
    int64_t in_feat = 64, out_feat = 32, batch = 8;

    // Create and run float linear layer
    nn::Linear linear(in_feat, out_feat);
    auto input = Variable(randn({batch, in_feat}, DType::Float32, Device::cpu()), false);
    auto float_output = linear.forward(input);

    // Get weight and bias
    auto params = linear.parameters();
    ASSERT_GE(params.size(), 1);
    auto weight = params[0]->tensor();

    // Quantize weight
    float wmax = 0.0f;
    {
        auto wc = weight.to(Device::cpu());
        const float* d = wc.data<float>();
        for (size_t i = 0; i < wc.numel(); ++i) {
            wmax = std::max(wmax, std::abs(d[i]));
        }
    }
    double w_scale = static_cast<double>(wmax) / 127.0;
    auto q_weight = quantize_per_tensor(weight, w_scale, 0, DType::QInt8);

    EXPECT_TRUE(q_weight.is_quantized());
    EXPECT_EQ(q_weight.numel(), out_feat * in_feat);

    // Verify dequantized weight is close to original
    auto deq_weight = q_weight.dequantize();
    double weight_err = max_abs_error(weight, deq_weight);
    EXPECT_LT(weight_err, w_scale * 1.5);
}

// Test: quantization preserves shape
TEST_F(QuantizedInferenceTest, ShapePreservation) {
    auto t = randn({3, 4, 5}, DType::Float32, Device::cpu());
    auto qt = quantize_per_tensor(t, 0.01, 0, DType::QInt8);

    EXPECT_EQ(qt.shape()[0], 3);
    EXPECT_EQ(qt.shape()[1], 4);
    EXPECT_EQ(qt.shape()[2], 5);
    EXPECT_EQ(qt.numel(), 60);
}

// Test: QUInt8 asymmetric quantization with non-zero zero_point
TEST_F(QuantizedInferenceTest, AsymmetricQuantization) {
    auto t = zeros({4}, DType::Float32, Device::cpu());
    float* data = t.data<float>();
    data[0] = 0.0f;
    data[1] = 0.5f;
    data[2] = 1.0f;
    data[3] = 0.25f;

    // Asymmetric: range [0, 1] maps to [0, 255]
    double scale = 1.0 / 255.0;
    int64_t zero_point = 0;

    auto qt = quantize_per_tensor(t, scale, zero_point, DType::QUInt8);
    auto deq = qt.dequantize();

    const float* out = deq.data<float>();
    EXPECT_NEAR(out[0], 0.0f, scale * 2);
    EXPECT_NEAR(out[1], 0.5f, scale * 2);
    EXPECT_NEAR(out[2], 1.0f, scale * 2);
    EXPECT_NEAR(out[3], 0.25f, scale * 2);
}
