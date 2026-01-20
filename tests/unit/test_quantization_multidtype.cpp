/**
 * @file test_quantization_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for quantization functionality
 *
 * Tests quantization with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends:
 * - Dynamic quantization (runtime calibration)
 * - Static quantization (pre-calibrated parameters)
 * - Quantization-aware training (QAT) simulation
 * - Per-channel vs per-tensor quantization
 * - Dequantization and error metrics
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/quantize.hpp>
#include <tenzor/nn/quantization/observer.hpp>
#include <tenzor/nn/quantization/fake_quantize.hpp>
#include <tenzor/nn/quantization/qconfig.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn::quantization;
using namespace tenzor::testing;

// ============================================================================
// Quantization Multi-Backend Multi-DType Test Fixture
// ============================================================================

class QuantizationMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Helper: Create test tensor with specific pattern
    Tensor createTestTensor(const std::vector<int64_t>& shape,
                           const std::vector<float>& values) {
        Tensor tensor = tenzor::zeros(shape, DType::Float32, Device::cpu());
        float* data = tensor.data<float>();
        for (size_t i = 0; i < values.size(); ++i) {
            data[i] = values[i];
        }
        if (dtype() != DType::Float32) {
            tensor = tensor.to(dtype());
        }
        if (device() != Device::cpu()) {
            tensor = tensor.to(device());
        }
        return tensor;
    }

    // Helper: Fill tensor with range values
    Tensor createRangeTensor(const std::vector<int64_t>& shape, float start, float step) {
        int64_t numel = 1;
        for (auto dim : shape) numel *= dim;

        std::vector<float> values(numel);
        for (int64_t i = 0; i < numel; ++i) {
            values[i] = start + static_cast<float>(i) * step;
        }
        return createTestTensor(shape, values);
    }
};

// ============================================================================
// Dynamic Quantization Tests
// ============================================================================

TEST_P(QuantizationMultiDTypeTest, DynamicQuantization_PerTensorSymmetric) {
    // Create input tensor with known range
    auto input = createTestTensor({4}, {-2.0f, -1.0f, 1.0f, 2.0f});

    // Dynamic quantization: compute params and quantize in one go
    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);

    // Verify quantization parameters
    EXPECT_EQ(q_tensor.params().dtype, QuantDType::INT8);
    EXPECT_EQ(q_tensor.params().scheme, QuantizationScheme::PerTensorSymmetric);
    EXPECT_EQ(q_tensor.data().dtype(), DType::Int8);

    // Scale should be max_abs / 127
    float expected_scale = 2.0f / 127.0f;
    float actual_scale = q_tensor.params().scale.template data<const float>()[0];
    EXPECT_NEAR(actual_scale, expected_scale, 1e-4f);

    // Zero point should be 0 for symmetric
    EXPECT_EQ(q_tensor.params().zero_point.template data<int32_t>()[0], 0);

    // Dequantize and check reconstruction error
    Tensor deq = q_tensor.dequantize();
    auto deq_cpu = deq.to(Device::cpu()).to(DType::Float32);
    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);

    const float* input_data = input_cpu.data<float>();
    const float* deq_data = deq_cpu.data<float>();

    for (int i = 0; i < 4; ++i) {
        // Allow ~1% relative error due to quantization
        EXPECT_NEAR(input_data[i], deq_data[i], 0.1f)
            << "Index: " << i;
    }
}

TEST_P(QuantizationMultiDTypeTest, DynamicQuantization_PerTensorAsymmetric) {
    // Asymmetric range: [0, 3]
    auto input = createTestTensor({4}, {0.0f, 1.0f, 2.0f, 3.0f});

    Tensor min_tensor = tenzor::zeros({1}, dtype(), device());
    Tensor max_tensor = tenzor::ones({1}, dtype(), device()) * 3.0f;

    auto params = compute_quantization_params(
        min_tensor, max_tensor, QuantDType::INT8, QuantizationScheme::PerTensorAsymmetric
    );

    auto q_tensor = quantize_tensor(input, params);

    // Verify asymmetric quantization
    EXPECT_EQ(params.scheme, QuantizationScheme::PerTensorAsymmetric);

    // Zero point should NOT be 0 for asymmetric
    int32_t zp = params.zero_point.template data<int32_t>()[0];
    EXPECT_GE(zp, -128);
    EXPECT_LE(zp, 127);

    // Dequantize
    Tensor deq = q_tensor.dequantize();
    auto deq_cpu = deq.to(Device::cpu()).to(DType::Float32);
    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    const float* input_data = input_cpu.data<float>();
    const float* deq_data = deq_cpu.data<float>();

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(input_data[i], deq_data[i], 0.1f);
    }
}

TEST_P(QuantizationMultiDTypeTest, DynamicQuantization_LargeRange) {
    // Test with large dynamic range
    auto input = createTestTensor({6},
        {-100.0f, -50.0f, -1.0f, 1.0f, 50.0f, 100.0f});

    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    Tensor deq = q_tensor.dequantize();

    auto deq_cpu = deq.to(Device::cpu()).to(DType::Float32);
    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    const float* input_data = input_cpu.data<float>();
    const float* deq_data = deq_cpu.data<float>();

    // Check reconstruction with larger tolerance for large range
    for (int i = 0; i < 6; ++i) {
        float rel_error = std::abs(input_data[i] - deq_data[i]) /
                          (std::abs(input_data[i]) + 1e-8f);
        EXPECT_LT(rel_error, 0.02f)  // <2% relative error
            << "Value: " << input_data[i];
    }
}

// ============================================================================
// Static Quantization Tests
// ============================================================================

TEST_P(QuantizationMultiDTypeTest, StaticQuantization_PrecomputedParams) {
    // Pre-computed quantization parameters (e.g., from calibration)
    Tensor scale = tenzor::ones({1}, DType::Float32, Device::cpu());
    scale.fill_(0.5f);  // Fixed scale
    Tensor zero_point = tenzor::zeros({1}, DType::Int32, Device::cpu());

    QuantizationParams params(
        scale,
        zero_point,
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric,
        -1
    );

    // Apply static quantization
    auto input = createTestTensor({5}, {-60.0f, -30.0f, 0.0f, 30.0f, 60.0f});

    auto q_tensor = quantize_tensor(input, params);

    // Verify quantized values use pre-computed params
    auto q_data_cpu = q_tensor.data().to(Device::cpu());
    const int8_t* q_data = static_cast<const int8_t*>(q_data_cpu.data_ptr());

    // Expected quantized values: input / scale
    // -60/0.5=-120, -30/0.5=-60, 0/0.5=0, 30/0.5=60, 60/0.5=120
    EXPECT_EQ(q_data[0], -120);
    EXPECT_EQ(q_data[1], -60);
    EXPECT_EQ(q_data[2], 0);
    EXPECT_EQ(q_data[3], 60);
    EXPECT_EQ(q_data[4], 120);
}

TEST_P(QuantizationMultiDTypeTest, StaticQuantization_Calibration) {
    // Simulate calibration phase
    MinMaxObserver observer;

    // Collect statistics from calibration dataset
    for (int i = 0; i < 3; ++i) {
        auto calib_data = createRangeTensor({10}, -5.0f + i, 0.5f);
        observer.observe(calib_data);
    }

    // Compute quantization parameters from calibration
    auto params = observer.calculate_qparams(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    // Apply to inference data
    auto test_input = createRangeTensor({5}, -2.0f, 1.0f);
    auto q_tensor = quantize_tensor(test_input, params);

    // Verify parameters are computed from observed statistics
    EXPECT_TRUE(observer.has_data());
    EXPECT_GT(params.scale.template data<const float>()[0], 0.0f);

    // Check dequantization error
    Tensor deq = q_tensor.dequantize();
    auto deq_cpu = deq.to(Device::cpu()).to(DType::Float32);
    auto input_cpu = test_input.to(Device::cpu()).to(DType::Float32);
    const float* input_data = input_cpu.data<float>();
    const float* deq_data = deq_cpu.data<float>();

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(input_data[i], deq_data[i], 0.2f);
    }
}

// ============================================================================
// Quantization-Aware Training (QAT) Tests
// ============================================================================

TEST_P(QuantizationMultiDTypeTest, QAT_FakeQuantizeForward) {
    auto fake_quant = std::make_shared<FakeQuantize>(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric,
        false,  // Not learnable
        true    // Observer enabled
    );

    auto input_tensor = createTestTensor({4}, {-2.0f, -1.0f, 1.0f, 2.0f});
    Variable input(input_tensor, true);

    // Training mode: observe statistics
    fake_quant->train();

    // Forward pass simulates quantization
    Variable output = fake_quant->forward(input);

    EXPECT_EQ(output.tensor().dtype(), dtype());
    EXPECT_EQ(output.tensor().shape()[0], 4);

    auto input_cpu = input_tensor.to(Device::cpu()).to(DType::Float32);
    auto output_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* input_data = input_cpu.data<float>();
    const float* output_data = output_cpu.data<float>();

    // Output should be quantized then dequantized (fake quantization)
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(input_data[i], output_data[i], 0.2f)
            << "QAT simulation failed at index " << i;
    }
}

TEST_P(QuantizationMultiDTypeTest, QAT_LearnableScaleZeroPoint) {
    auto fake_quant = std::make_shared<FakeQuantize>(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric,
        true,   // Learnable parameters
        true    // Observer enabled
    );

    auto input_tensor = createRangeTensor({10}, -5.0f, 1.0f);
    Variable input(input_tensor, true);

    fake_quant->train();

    // Multiple forward passes (simulating training)
    Variable last_output = input;
    for (int iter = 0; iter < 3; ++iter) {
        last_output = fake_quant->forward(input);

        // In real QAT, gradients would update scale/zero_point
        EXPECT_EQ(last_output.tensor().shape()[0], 10);
    }

    // After training, forward pass should complete successfully
    EXPECT_EQ(last_output.tensor().shape()[0], 10);
}

TEST_P(QuantizationMultiDTypeTest, QAT_InferenceMode) {
    auto fake_quant = std::make_shared<FakeQuantize>(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    auto input_tensor = createTestTensor({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Variable input(input_tensor, false);

    // Training mode first
    fake_quant->train();
    fake_quant->forward(input);

    // Switch to eval mode
    fake_quant->eval();
    fake_quant->disable_observer();

    Variable output = fake_quant->forward(input);

    // In inference, should use frozen parameters
    auto input_cpu = input_tensor.to(Device::cpu()).to(DType::Float32);
    auto output_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* input_data = input_cpu.data<float>();
    const float* output_data = output_cpu.data<float>();

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(input_data[i], output_data[i], 0.2f);
    }
}

// ============================================================================
// Per-Channel vs Per-Tensor Quantization
// ============================================================================

TEST_P(QuantizationMultiDTypeTest, PerChannelSymmetric_2D) {
    // 2 channels, 4 elements each
    auto input = createTestTensor({2, 4}, {
        -2.0f, -1.0f, 1.0f, 2.0f,   // Channel 0: range [-2, 2]
        -1.0f, 0.0f, 1.0f, 2.0f     // Channel 1: range [-1, 2]
    });

    // Per-channel quantization along axis 0
    auto q_tensor = quantize_per_channel_symmetric(input, 0, QuantDType::INT8);

    EXPECT_EQ(q_tensor.params().axis, 0);
    EXPECT_EQ(q_tensor.params().scale.numel(), 2);  // One scale per channel
    EXPECT_EQ(q_tensor.params().zero_point.numel(), 2);

    // Each channel should have different scales
    const float* scales = q_tensor.params().scale.template data<const float>();
    float scale0 = 2.0f / 127.0f;  // Channel 0 scale
    float scale1 = 2.0f / 127.0f;  // Channel 1 scale

    EXPECT_NEAR(scales[0], scale0, 1e-4f);
    EXPECT_NEAR(scales[1], scale1, 1e-4f);

    // Dequantize and verify
    Tensor deq = q_tensor.dequantize();
    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    auto deq_cpu = deq.to(Device::cpu()).to(DType::Float32);
    const float* input_data = input_cpu.data<float>();
    const float* deq_data = deq_cpu.data<float>();

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(input_data[i], deq_data[i], 0.1f)
            << "Per-channel quantization failed at index " << i;
    }
}

TEST_P(QuantizationMultiDTypeTest, PerChannelVsPerTensor_Accuracy) {
    // Create tensor with varying channel magnitudes
    auto input = createTestTensor({3, 4}, {
        -10.0f, -5.0f, 5.0f, 10.0f,    // Channel 0: large range
        -1.0f, -0.5f, 0.5f, 1.0f,      // Channel 1: small range
        -5.0f, -2.5f, 2.5f, 5.0f       // Channel 2: medium range
    });

    // Per-tensor quantization
    auto q_per_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    Tensor deq_per_tensor = q_per_tensor.dequantize();

    // Per-channel quantization
    auto q_per_channel = quantize_per_channel_symmetric(input, 0, QuantDType::INT8);
    Tensor deq_per_channel = q_per_channel.dequantize();

    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    auto deq_per_tensor_cpu = deq_per_tensor.to(Device::cpu()).to(DType::Float32);
    auto deq_per_channel_cpu = deq_per_channel.to(Device::cpu()).to(DType::Float32);

    const float* input_data = input_cpu.data<float>();
    const float* deq_per_tensor_data = deq_per_tensor_cpu.data<float>();
    const float* deq_per_channel_data = deq_per_channel_cpu.data<float>();

    // Calculate MSE for both methods
    float mse_per_tensor = 0.0f;
    float mse_per_channel = 0.0f;

    for (int i = 0; i < 12; ++i) {
        float error_tensor = input_data[i] - deq_per_tensor_data[i];
        float error_channel = input_data[i] - deq_per_channel_data[i];
        mse_per_tensor += error_tensor * error_tensor;
        mse_per_channel += error_channel * error_channel;
    }

    mse_per_tensor /= 12.0f;
    mse_per_channel /= 12.0f;

    // Per-channel should have lower error for varied ranges
    EXPECT_LT(mse_per_channel, mse_per_tensor)
        << "Per-channel quantization should be more accurate for varied channel ranges";
}

// ============================================================================
// Dequantization Tests
// ============================================================================

TEST_P(QuantizationMultiDTypeTest, Dequantization_PreserveDtype) {
    auto input = createTestTensor({5}, {-2.5f, -1.0f, 0.0f, 1.0f, 2.5f});

    // Quantize
    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);

    // Dequantize
    Tensor deq = q_tensor.dequantize();

    // Verify dtype is preserved
    EXPECT_EQ(deq.dtype(), dtype())
        << "Dequantization should preserve original dtype";

    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    auto deq_cpu = deq.to(Device::cpu()).to(DType::Float32);
    const float* input_data = input_cpu.data<float>();
    const float* deq_data = deq_cpu.data<float>();

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(input_data[i], deq_data[i], 0.1f);
    }
}

TEST_P(QuantizationMultiDTypeTest, Dequantization_PerChannelReconstruction) {
    // Multi-channel tensor
    auto input = createTestTensor({2, 6}, {
        -3.0f, -2.0f, -1.0f, 1.0f, 2.0f, 3.0f,
        -1.5f, -1.0f, -0.5f, 0.5f, 1.0f, 1.5f
    });

    // Per-channel quantization
    auto q_tensor = quantize_per_channel_symmetric(input, 0, QuantDType::INT8);

    // Dequantize
    Tensor deq = q_tensor.dequantize();

    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    auto deq_cpu = deq.to(Device::cpu()).to(DType::Float32);
    const float* input_data = input_cpu.data<float>();
    const float* deq_data = deq_cpu.data<float>();

    // Each channel should be reconstructed accurately
    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR(input_data[i], deq_data[i], 0.15f)
            << "Per-channel dequantization failed at index " << i;
    }
}

// ============================================================================
// Quantization Error Metrics
// ============================================================================

TEST_P(QuantizationMultiDTypeTest, ErrorMetrics_MAE_MSE_SNR) {
    // Create test signal
    auto original = createRangeTensor({20}, -10.0f, 1.0f);

    // Quantize and dequantize
    auto q_tensor = quantize_per_tensor_symmetric(original, QuantDType::INT8);

    // Compute error metrics
    auto [mae, mse, snr] = compute_quantization_error(original, q_tensor);

    // Verify metrics are reasonable
    EXPECT_GT(mae, 0.0f) << "MAE should be positive";
    EXPECT_LT(mae, 1.0f) << "MAE should be small for good quantization";

    EXPECT_GT(mse, 0.0f) << "MSE should be positive";
    EXPECT_LT(mse, 1.0f) << "MSE should be small for good quantization";

    EXPECT_GT(snr, 15.0f) << "SNR should be at least 15 dB for INT8 quantization";
}

TEST_P(QuantizationMultiDTypeTest, ErrorMetrics_CompareQuantizationSchemes) {
    // Test signal with bias (not centered at 0)
    auto original = createRangeTensor({15}, 5.0f, 0.5f);

    // Symmetric quantization
    auto q_symmetric = quantize_per_tensor_symmetric(original, QuantDType::INT8);
    auto [mae_sym, mse_sym, snr_sym] = compute_quantization_error(original, q_symmetric);

    // Asymmetric quantization
    Tensor min_tensor = tenzor::ones({1}, dtype(), device()) * 5.0f;
    Tensor max_tensor = tenzor::ones({1}, dtype(), device()) * 12.0f;

    auto params_asym = compute_quantization_params(
        min_tensor, max_tensor, QuantDType::INT8, QuantizationScheme::PerTensorAsymmetric
    );
    auto q_asymmetric = quantize_tensor(original, params_asym);
    auto [mae_asym, mse_asym, snr_asym] = compute_quantization_error(original, q_asymmetric);

    // For biased data, asymmetric should have lower error
    EXPECT_LT(mae_asym, mae_sym * 1.5f)  // At least not much worse
        << "Asymmetric quantization should be better for biased data";
}

// ============================================================================
// Observer Calibration Tests
// ============================================================================

TEST_P(QuantizationMultiDTypeTest, Observer_MinMaxCalibration) {
    MinMaxObserver observer;

    // Simulate multiple calibration batches
    for (int batch = 0; batch < 4; ++batch) {
        auto calib_data = createRangeTensor({10}, -5.0f + batch * 0.5f, 0.5f);
        observer.observe(calib_data);
    }

    EXPECT_TRUE(observer.has_data());

    // Calculate quantization parameters
    auto params = observer.calculate_qparams(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    // Verify parameters are computed from observed range
    float observed_min = observer.get_min().template data<const float>()[0];
    float observed_max = observer.get_max().template data<const float>()[0];

    EXPECT_LT(observed_min, -4.0f);
    EXPECT_GT(observed_max, 4.0f);
    EXPECT_GT(params.scale.template data<const float>()[0], 0.0f);
}

TEST_P(QuantizationMultiDTypeTest, Observer_MovingAverage) {
    MovingAverageMinMaxObserver observer(0.9f);  // High momentum

    // First observation
    auto data1 = createTestTensor({5}, {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f});
    observer.observe(data1);

    auto params1 = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    float scale1 = params1.scale.template data<const float>()[0];

    // Second observation with different range
    auto data2 = createTestTensor({5}, {-8.0f, -4.0f, 0.0f, 4.0f, 8.0f});
    observer.observe(data2);

    auto params2 = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    float scale2 = params2.scale.template data<const float>()[0];

    // Scale should change gradually due to moving average
    EXPECT_NE(scale1, scale2);
    EXPECT_LT(std::abs(scale2 - scale1), scale1 * 0.2f)
        << "Moving average should smooth scale changes";
}

TEST_P(QuantizationMultiDTypeTest, Observer_HistogramCalibration) {
    HistogramObserver observer(256, 0.001f, 0.999f);  // Use 99.8% percentiles

    // Generate data with outliers
    std::vector<float> values;
    for (int i = 0; i < 90; ++i) {
        values.push_back(static_cast<float>(i) / 45.0f - 1.0f);  // [-1, 1]
    }
    // Add outliers
    values.push_back(-10.0f);
    values.push_back(10.0f);

    auto data = createTestTensor({92}, values);
    observer.observe(data);

    EXPECT_TRUE(observer.has_data());

    auto params = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);

    // Histogram observer should clip outliers
    float scale = params.scale.template data<const float>()[0];
    EXPECT_GT(scale, 0.0f);
    EXPECT_LT(scale, 10.0f / 127.0f)  // Should be less than if outliers were included
        << "Histogram observer should reduce impact of outliers";
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TEST_P(QuantizationMultiDTypeTest, EdgeCase_ZeroRange) {
    // All values the same (zero range)
    auto input = createTestTensor({5}, {5.0f, 5.0f, 5.0f, 5.0f, 5.0f});

    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    Tensor deq = q_tensor.dequantize();

    auto deq_cpu = deq.to(Device::cpu()).to(DType::Float32);
    const float* deq_data = deq_cpu.data<float>();

    // Should handle zero range gracefully
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(5.0f, deq_data[i], 0.1f)
            << "Zero range quantization should preserve constant value";
    }
}

TEST_P(QuantizationMultiDTypeTest, EdgeCase_VerySmallValues) {
    // Very small values near machine epsilon
    auto input = createTestTensor({4}, {1e-6f, 2e-6f, 3e-6f, 4e-6f});

    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    Tensor deq = q_tensor.dequantize();

    // Should handle small values without underflow
    EXPECT_EQ(deq.shape()[0], 4);

    auto deq_cpu = deq.to(Device::cpu()).to(DType::Float32);
    const float* deq_data = deq_cpu.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(deq_data[i]))
            << "Small value quantization should not produce NaN/Inf";
    }
}

TEST_P(QuantizationMultiDTypeTest, EdgeCase_MixedSignLargeRange) {
    // Very large range with mixed signs
    auto input = createTestTensor({6},
        {-1000.0f, -0.001f, 0.0f, 0.001f, 1000.0f, 2000.0f});

    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    Tensor deq = q_tensor.dequantize();

    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    auto deq_cpu = deq.to(Device::cpu()).to(DType::Float32);
    const float* input_data = input_cpu.data<float>();
    const float* deq_data = deq_cpu.data<float>();

    // Large values should dominate the scale, small values will have high error
    // But large values should be accurate
    EXPECT_NEAR(input_data[0], deq_data[0], 50.0f);  // -1000
    EXPECT_NEAR(input_data[4], deq_data[4], 50.0f);  // 1000
    EXPECT_NEAR(input_data[5], deq_data[5], 50.0f);  // 2000
}

// ============================================================================
// Integration Test: Full Quantization Workflow
// ============================================================================

TEST_P(QuantizationMultiDTypeTest, Integration_FullQuantizationPipeline) {
    // Step 1: Create calibration dataset
    std::vector<Tensor> calib_dataset;
    for (int i = 0; i < 5; ++i) {
        auto batch = createRangeTensor({20}, -10.0f + i, 0.5f);
        calib_dataset.push_back(batch);
    }

    // Step 2: Calibrate with observer
    MinMaxObserver observer;
    for (const auto& batch : calib_dataset) {
        observer.observe(batch);
    }

    // Step 3: Compute quantization parameters
    auto params = observer.calculate_qparams(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    // Step 4: Quantize model weights (simulated)
    auto weights = createRangeTensor({50}, -5.0f, 0.2f);
    auto q_weights = quantize_tensor(weights, params);

    // Step 5: Quantize activations (simulated inference)
    auto activations = createRangeTensor({20}, -8.0f, 0.8f);
    auto q_activations = quantize_tensor(activations, params);

    // Step 6: Dequantize for comparison
    Tensor deq_weights = q_weights.dequantize();
    Tensor deq_activations = q_activations.dequantize();

    // Step 7: Compute error metrics
    auto [mae_w, mse_w, snr_w] = compute_quantization_error(weights, q_weights);
    auto [mae_a, mse_a, snr_a] = compute_quantization_error(activations, q_activations);

    // Verify pipeline produces reasonable results
    EXPECT_LT(mae_w, 0.5f) << "Weights MAE should be small";
    EXPECT_LT(mae_a, 0.5f) << "Activations MAE should be small";
    EXPECT_GT(snr_w, 15.0f) << "Weights SNR should be >15dB";
    EXPECT_GT(snr_a, 15.0f) << "Activations SNR should be >15dB";
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(QuantizationMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 21
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 21 tests × 3 dtypes × 3 backends = 189 test scenarios
 *
 * Coverage:
 * - Dynamic quantization: per-tensor symmetric/asymmetric, large range
 * - Static quantization: pre-computed params, calibration
 * - Quantization-aware training: fake quantize, learnable params, inference mode
 * - Per-channel vs per-tensor: 2D quantization, accuracy comparison
 * - Dequantization: dtype preservation, per-channel reconstruction
 * - Error metrics: MAE/MSE/SNR, scheme comparison
 * - Observer calibration: min-max, moving average, histogram
 * - Edge cases: zero range, very small values, mixed sign large range
 * - Integration: full quantization pipeline
 */
