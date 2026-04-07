/**
 * @file test_quantization.cpp
 * @brief Comprehensive unit tests for quantization functionality
 */

#include <gtest/gtest.h>
#include "tenzor/nn/quantization/quantize.hpp"
#include "tenzor/nn/quantization/observer.hpp"
#include "tenzor/nn/quantization/fake_quantize.hpp"
#include "tenzor/nn/quantization/qconfig.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn::quantization;

// Global test environment for initialization
class QuantizationTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const quant_env =
    ::testing::AddGlobalTestEnvironment(new QuantizationTestEnvironment);

class QuantizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for tests
    }

    void TearDown() override {
        // Cleanup
    }

    // Helper to check if values are close
    bool isClose(float a, float b, float rtol = 1e-5f, float atol = 1e-8f) {
        return std::abs(a - b) <= (atol + rtol * std::abs(b));
    }
};

// ============================================================================
// Quantization Parameter Tests
// ============================================================================

TEST_F(QuantizationTest, ComputeQuantizationParams_PerTensorSymmetric) {
    Tensor min({1}, DType::Float32, Device::cpu());
    Tensor max({1}, DType::Float32, Device::cpu());
    min.fill_(-2.0f);
    max.fill_(3.0f);

    auto params = compute_quantization_params(
        min, max, QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );

    EXPECT_EQ(params.dtype, QuantDType::INT8);
    EXPECT_EQ(params.scheme, QuantizationScheme::PerTensorSymmetric);
    EXPECT_EQ(params.axis, -1);

    // Scale should be max(abs(-2), abs(3)) / 127 ≈ 3.0 / 127
    float expected_scale = 3.0f / 127.0f;
    EXPECT_TRUE(isClose(params.scale.data<float>()[0], expected_scale, 1e-4f));

    // Zero point should be 0 for symmetric
    EXPECT_EQ(params.zero_point.data<int32_t>()[0], 0);
}

TEST_F(QuantizationTest, ComputeQuantizationParams_PerTensorAsymmetric) {
    Tensor min({1}, DType::Float32, Device::cpu());
    Tensor max({1}, DType::Float32, Device::cpu());
    min.fill_(-1.0f);
    max.fill_(2.0f);

    auto params = compute_quantization_params(
        min, max, QuantDType::INT8, QuantizationScheme::PerTensorAsymmetric
    );

    EXPECT_EQ(params.dtype, QuantDType::INT8);
    EXPECT_EQ(params.scheme, QuantizationScheme::PerTensorAsymmetric);

    // Range is 3.0, quantized range is 255
    float expected_scale = 3.0f / 255.0f;
    EXPECT_TRUE(isClose(params.scale.data<float>()[0], expected_scale, 1e-4f));

    // Zero point should map 0.0 to appropriate quantized value
    int32_t zp = params.zero_point.data<int32_t>()[0];
    EXPECT_GE(zp, -128);
    EXPECT_LE(zp, 127);
}

// ============================================================================
// Quantization/Dequantization Tests
// ============================================================================

TEST_F(QuantizationTest, QuantizeAndDequantize_PerTensorSymmetric) {
    Tensor input({4}, DType::Float32, Device::cpu());
    float* data = input.data<float>();
    data[0] = -2.0f;
    data[1] = -1.0f;
    data[2] = 1.0f;
    data[3] = 2.0f;

    // Quantize
    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);

    // Check quantized values
    const int8_t* q_data = q_tensor.data().data<int8_t>();
    EXPECT_EQ(q_tensor.data().dtype(), DType::Int8);

    // Dequantize
    Tensor deq = q_tensor.dequantize();

    // Check dequantized values are close to original
    const float* deq_data = deq.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(isClose(data[i], deq_data[i], 0.1f))
            << "Mismatch at index " << i
            << ": original=" << data[i]
            << ", dequantized=" << deq_data[i];
    }
}

TEST_F(QuantizationTest, QuantizeAndDequantize_PerChannelSymmetric) {
    Tensor input({2, 4}, DType::Float32, Device::cpu());
    float* data = input.data<float>();

    // Channel 0: [-2, -1, 1, 2]
    data[0] = -2.0f; data[1] = -1.0f; data[2] = 1.0f; data[3] = 2.0f;
    // Channel 1: [-1, 0, 1, 2]
    data[4] = -1.0f; data[5] = 0.0f; data[6] = 1.0f; data[7] = 2.0f;

    auto q_tensor = quantize_per_channel_symmetric(input, 0, QuantDType::INT8);

    // Check per-channel scales
    EXPECT_EQ(q_tensor.params().axis, 0);
    EXPECT_EQ(q_tensor.params().scale.numel(), 2);

    // Dequantize
    Tensor deq = q_tensor.dequantize();
    const float* deq_data = deq.data<float>();

    for (int i = 0; i < 8; ++i) {
        EXPECT_TRUE(isClose(data[i], deq_data[i], 0.1f));
    }
}

// ============================================================================
// Observer Tests
// ============================================================================

TEST_F(QuantizationTest, MinMaxObserver_SingleTensor) {
    MinMaxObserver observer;

    Tensor input({4}, DType::Float32, Device::cpu());
    float* data = input.data<float>();
    data[0] = -2.0f;
    data[1] = 1.0f;
    data[2] = 3.0f;
    data[3] = -1.0f;

    observer.observe(input);

    EXPECT_TRUE(observer.has_data());

    auto params = observer.calculate_qparams(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    // Min should be -2.0, max should be 3.0
    EXPECT_TRUE(isClose(observer.get_min().data<float>()[0], -2.0f));
    EXPECT_TRUE(isClose(observer.get_max().data<float>()[0], 3.0f));
}

TEST_F(QuantizationTest, MinMaxObserver_MultipleTensors) {
    MinMaxObserver observer;

    // First observation
    Tensor input1({4}, DType::Float32, Device::cpu());
    float* data1 = input1.data<float>();
    data1[0] = -1.0f; data1[1] = 2.0f; data1[2] = 0.0f; data1[3] = 1.0f;
    observer.observe(input1);

    // Second observation - should update min/max
    Tensor input2({4}, DType::Float32, Device::cpu());
    float* data2 = input2.data<float>();
    data2[0] = -3.0f; data2[1] = 1.0f; data2[2] = 4.0f; data2[3] = 0.0f;
    observer.observe(input2);

    // Min should be -3.0, max should be 4.0
    EXPECT_TRUE(isClose(observer.get_min().data<float>()[0], -3.0f));
    EXPECT_TRUE(isClose(observer.get_max().data<float>()[0], 4.0f));
}

TEST_F(QuantizationTest, MovingAverageMinMaxObserver) {
    MovingAverageMinMaxObserver observer(0.9f);

    Tensor input1({4}, DType::Float32, Device::cpu());
    float* data1 = input1.data<float>();
    data1[0] = -1.0f; data1[1] = 1.0f; data1[2] = 0.0f; data1[3] = 0.5f;
    observer.observe(input1);

    Tensor input2({4}, DType::Float32, Device::cpu());
    float* data2 = input2.data<float>();
    data2[0] = -0.5f; data2[1] = 2.0f; data2[2] = 0.0f; data2[3] = 1.0f;
    observer.observe(input2);

    // Values should be moving average, not exact min/max
    auto params = observer.calculate_qparams(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    EXPECT_TRUE(observer.has_data());
}

TEST_F(QuantizationTest, HistogramObserver) {
    HistogramObserver observer(256, 0.001f, 0.999f);

    // Generate sample data
    Tensor input({100}, DType::Float32, Device::cpu());
    float* data = input.data<float>();
    for (int i = 0; i < 100; ++i) {
        data[i] = static_cast<float>(i) / 50.0f - 1.0f;  // [-1, 1]
    }

    observer.observe(input);
    EXPECT_TRUE(observer.has_data());

    auto params = observer.calculate_qparams(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    // Should compute reasonable quantization parameters
    EXPECT_GT(params.scale.data<float>()[0], 0.0f);
}

// ============================================================================
// Fake Quantization Tests
// ============================================================================

TEST_F(QuantizationTest, FakeQuantize_Forward) {
    auto fake_quant = std::make_shared<FakeQuantize>(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric,
        false,  // Not learnable
        true    // Observer enabled
    );

    Tensor input_tensor({4}, DType::Float32, Device::cpu());
    float* data = input_tensor.data<float>();
    data[0] = -2.0f; data[1] = -1.0f; data[2] = 1.0f; data[3] = 2.0f;

    Variable input(input_tensor, true);

    // Enable training mode to collect statistics
    fake_quant->train();

    // Forward pass - should update observer
    Variable output = fake_quant->forward(input);

    // Output should be fake-quantized (quantize then dequantize)
    EXPECT_EQ(output.tensor().shape()[0], 4);

    // Values should be slightly different due to quantization
    const float* out_data = output.tensor().data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i], out_data[i], 0.2f);
    }
}

TEST_F(QuantizationTest, FakeQuantize_EnableDisable) {
    auto fake_quant = std::make_shared<FakeQuantize>();

    Tensor input_tensor({4}, DType::Float32, Device::cpu());
    float* data = input_tensor.data<float>();
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;

    Variable input(input_tensor, false);

    // Disable fake quantization - should act as identity
    fake_quant->disable_fake_quant();
    fake_quant->eval();

    Variable output = fake_quant->forward(input);

    // Output should be identical to input
    const float* out_data = output.tensor().data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], out_data[i]);
    }
}

// ============================================================================
// QConfig Tests
// ============================================================================

TEST_F(QuantizationTest, DefaultQConfigs) {
    auto qconfig = DefaultQConfigs::default_qconfig();

    EXPECT_EQ(qconfig.weight_dtype(), QuantDType::INT8);
    EXPECT_EQ(qconfig.activation_dtype(), QuantDType::INT8);
    EXPECT_EQ(qconfig.weight_scheme(), QuantizationScheme::PerChannelSymmetric);
    EXPECT_EQ(qconfig.activation_scheme(), QuantizationScheme::PerTensorSymmetric);

    // Should be able to create observers
    auto weight_obs = qconfig.create_weight_observer();
    auto act_obs = qconfig.create_activation_observer();

    EXPECT_NE(weight_obs, nullptr);
    EXPECT_NE(act_obs, nullptr);
}

TEST_F(QuantizationTest, QConfigMapping) {
    QConfigMapping mapping;

    auto default_cfg = DefaultQConfigs::default_qconfig();
    auto high_acc_cfg = DefaultQConfigs::high_accuracy_qconfig();

    mapping.set_global(default_cfg);
    mapping.set_layer_qconfig("layer1", high_acc_cfg);
    mapping.disable_layer("layer2");

    // Should get layer-specific config
    auto cfg1 = mapping.get_qconfig("layer1", "Linear");
    EXPECT_NE(cfg1, nullptr);

    // Should get nullptr for disabled layer
    auto cfg2 = mapping.get_qconfig("layer2", "Linear");
    EXPECT_EQ(cfg2, nullptr);

    // Should get global config for unspecified layer
    auto cfg3 = mapping.get_qconfig("layer3", "Conv2d");
    EXPECT_NE(cfg3, nullptr);
}

// ============================================================================
// Quantization Error Tests
// ============================================================================

TEST_F(QuantizationTest, ComputeQuantizationError) {
    Tensor original({10}, DType::Float32, Device::cpu());
    float* data = original.data<float>();
    for (int i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(i) / 10.0f;
    }

    auto q_tensor = quantize_per_tensor_symmetric(original);

    auto [mae, mse, snr] = compute_quantization_error(original, q_tensor);

    // Errors should be small for this simple case
    EXPECT_GT(mae, 0.0f);  // Some error expected
    EXPECT_LT(mae, 0.1f);  // But not too large

    EXPECT_GT(mse, 0.0f);
    EXPECT_LT(mse, 0.01f);

    // SNR should be reasonable (positive dB)
    EXPECT_GT(snr, 20.0f);  // At least 20 dB SNR
}

// ============================================================================
// Calibration Tests
// ============================================================================

TEST_F(QuantizationTest, CalibrateQuantizationParams) {
    std::vector<Tensor> samples;

    // Create multiple sample tensors
    for (int i = 0; i < 5; ++i) {
        Tensor sample({20}, DType::Float32, Device::cpu());
        float* data = sample.data<float>();
        for (int j = 0; j < 20; ++j) {
            data[j] = static_cast<float>(j - 10 + i) / 10.0f;
        }
        samples.push_back(sample);
    }

    auto params = calibrate_quantization_params(
        samples,
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    // Should compute reasonable parameters
    EXPECT_GT(params.scale.data<float>()[0], 0.0f);
    EXPECT_EQ(params.zero_point.data<int32_t>()[0], 0);  // Symmetric
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(QuantizationTest, EndToEnd_PTQ_Workflow) {
    // 1. Create observer
    MinMaxObserver observer;

    // 2. Collect statistics from calibration data
    for (int i = 0; i < 3; ++i) {
        Tensor calib_data({10}, DType::Float32, Device::cpu());
        float* data = calib_data.data<float>();
        for (int j = 0; j < 10; ++j) {
            data[j] = static_cast<float>(j + i) / 5.0f - 1.0f;
        }
        observer.observe(calib_data);
    }

    // 3. Calculate quantization parameters
    auto params = observer.calculate_qparams(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    // 4. Quantize test tensor
    Tensor test_input({10}, DType::Float32, Device::cpu());
    float* test_data = test_input.data<float>();
    for (int i = 0; i < 10; ++i) {
        test_data[i] = static_cast<float>(i) / 5.0f - 1.0f;
    }

    auto q_tensor = quantize_tensor(test_input, params);

    // 5. Dequantize and verify
    Tensor deq = q_tensor.dequantize();
    const float* deq_data = deq.data<float>();

    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(test_data[i], deq_data[i], 0.2f);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
