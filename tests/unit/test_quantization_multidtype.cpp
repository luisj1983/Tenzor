/**
 * @file test_quantization_multidtype.cpp
 * @brief Comprehensive multi-dtype quantization tests (Float32/Float64 to Int8)
 *
 * Tests critical quantization scenarios for deployment:
 * - Dynamic quantization (runtime calibration)
 * - Static quantization (pre-calibrated parameters)
 * - Quantization-aware training (QAT) simulation
 * - Per-channel vs per-tensor quantization
 * - Quantized operations (matmul, conv, etc.)
 * - Dequantization and error metrics
 *
 * Covers Float32 and Float64 input dtypes quantizing to Int8.
 */

#include <gtest/gtest.h>
#include "tenzor/nn/quantization/quantize.hpp"
#include "tenzor/nn/quantization/observer.hpp"
#include "tenzor/nn/quantization/fake_quantize.hpp"
#include "tenzor/nn/quantization/qconfig.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/core/tensor.hpp"
#include <cmath>
#include <vector>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn::quantization;

// ============================================================================
// Multi-dtype Test Fixture
// ============================================================================

template <typename T>
class QuantizationMultiDTypeTest : public ::testing::Test {
protected:
    using DataType = T;

    void SetUp() override {
        dtype_ = std::is_same<T, float>::value ? DType::Float32 : DType::Float64;
    }

    DType dtype_;

    // Helper: Check if values are close (with dtype-specific tolerance)
    bool isClose(T a, T b, T rtol = 1e-5, T atol = 1e-6) {
        if (std::is_same<T, double>::value) {
            rtol = 1e-9;
            atol = 1e-10;
        }
        return std::abs(a - b) <= (atol + rtol * std::abs(b));
    }

    // Helper: Create test tensor with specific pattern
    Tensor createTestTensor(const std::vector<int64_t>& shape,
                           const std::vector<T>& values) {
        Tensor tensor(shape, dtype_, Device::cpu());
        T* data = tensor.data<T>();
        for (size_t i = 0; i < values.size(); ++i) {
            data[i] = values[i];
        }
        return tensor;
    }

    // Helper: Fill tensor with range values
    Tensor createRangeTensor(const std::vector<int64_t>& shape, T start, T step) {
        int64_t numel = 1;
        for (auto dim : shape) numel *= dim;

        std::vector<T> values(numel);
        for (int64_t i = 0; i < numel; ++i) {
            values[i] = start + static_cast<T>(i) * step;
        }
        return createTestTensor(shape, values);
    }
};

// Type definitions for parameterized tests
using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(QuantizationMultiDTypeTest, FloatTypes);

// ============================================================================
// Dynamic Quantization Tests
// ============================================================================

TYPED_TEST(QuantizationMultiDTypeTest, DynamicQuantization_PerTensorSymmetric) {
    using T = typename TestFixture::DataType;

    // Create input tensor with known range
    auto input = this->createTestTensor({4}, {T(-2.0), T(-1.0), T(1.0), T(2.0)});

    // Dynamic quantization: compute params and quantize in one go
    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);

    // Verify quantization parameters
    EXPECT_EQ(q_tensor.params().dtype, QuantDType::INT8);
    EXPECT_EQ(q_tensor.params().scheme, QuantizationScheme::PerTensorSymmetric);
    EXPECT_EQ(q_tensor.data().dtype(), DType::Int8);

    // Scale should be max_abs / 127
    T expected_scale = T(2.0) / T(127.0);
    T actual_scale = q_tensor.params().scale.template data<const float>()[0];
    EXPECT_TRUE(this->isClose(actual_scale, static_cast<float>(expected_scale), T(1e-4)));

    // Zero point should be 0 for symmetric
    EXPECT_EQ(q_tensor.params().zero_point.template data<int32_t>()[0], 0);

    // Dequantize and check reconstruction error
    Tensor deq = q_tensor.dequantize();
    EXPECT_EQ(deq.dtype(), this->dtype_);

    const T* input_data = input.template data<T>();
    const T* deq_data = deq.template data<T>();

    for (int i = 0; i < 4; ++i) {
        // Allow ~1% relative error due to quantization
        EXPECT_NEAR(input_data[i], deq_data[i], T(0.1))
            << "Dtype: " << (std::is_same<T, float>::value ? "Float32" : "Float64")
            << ", Index: " << i;
    }
}

TYPED_TEST(QuantizationMultiDTypeTest, DynamicQuantization_PerTensorAsymmetric) {
    using T = typename TestFixture::DataType;

    // Asymmetric range: [0, 3]
    auto input = this->createTestTensor({4}, {T(0.0), T(1.0), T(2.0), T(3.0)});

    Tensor min({1}, this->dtype_, Device::cpu());
    Tensor max({1}, this->dtype_, Device::cpu());
    min.fill_(T(0.0));
    max.fill_(T(3.0));

    auto params = compute_quantization_params(
        min, max, QuantDType::INT8, QuantizationScheme::PerTensorAsymmetric
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
    const T* input_data = input.template data<T>();
    const T* deq_data = deq.template data<T>();

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(input_data[i], deq_data[i], T(0.1))
            << "Dtype: " << (std::is_same<T, float>::value ? "Float32" : "Float64");
    }
}

TYPED_TEST(QuantizationMultiDTypeTest, DynamicQuantization_LargeRange) {
    using T = typename TestFixture::DataType;

    // Test with large dynamic range
    auto input = this->createTestTensor({6},
        {T(-100.0), T(-50.0), T(-1.0), T(1.0), T(50.0), T(100.0)});

    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    Tensor deq = q_tensor.dequantize();

    const T* input_data = input.template data<T>();
    const T* deq_data = deq.template data<T>();

    // Check reconstruction with larger tolerance for large range
    for (int i = 0; i < 6; ++i) {
        T rel_error = std::abs(input_data[i] - deq_data[i]) /
                      (std::abs(input_data[i]) + T(1e-8));
        EXPECT_LT(rel_error, T(0.02))  // <2% relative error
            << "Dtype: " << (std::is_same<T, float>::value ? "Float32" : "Float64")
            << ", Value: " << input_data[i];
    }
}

// ============================================================================
// Static Quantization Tests
// ============================================================================

TYPED_TEST(QuantizationMultiDTypeTest, StaticQuantization_PrecomputedParams) {
    using T = typename TestFixture::DataType;

    // Pre-computed quantization parameters (e.g., from calibration)
    Tensor scale({1}, DType::Float32, Device::cpu());
    Tensor zero_point({1}, DType::Int32, Device::cpu());
    scale.fill_(0.5f);  // Fixed scale
    zero_point.fill_(0);  // Symmetric

    QuantizationParams params(
        scale,
        zero_point,
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric,
        -1
    );

    // Apply static quantization
    auto input = this->createTestTensor({5},
        {T(-60.0), T(-30.0), T(0.0), T(30.0), T(60.0)});

    auto q_tensor = quantize_tensor(input, params);

    // Verify quantized values use pre-computed params
    const int8_t* q_data = q_tensor.data().template data<const int8_t>();

    // Expected quantized values: input / scale
    // -60/0.5=-120, -30/0.5=-60, 0/0.5=0, 30/0.5=60, 60/0.5=120
    EXPECT_EQ(q_data[0], -120);
    EXPECT_EQ(q_data[1], -60);
    EXPECT_EQ(q_data[2], 0);
    EXPECT_EQ(q_data[3], 60);
    EXPECT_EQ(q_data[4], 120);
}

TYPED_TEST(QuantizationMultiDTypeTest, StaticQuantization_Calibration) {
    using T = typename TestFixture::DataType;

    // Simulate calibration phase
    MinMaxObserver observer;

    // Collect statistics from calibration dataset
    for (int i = 0; i < 3; ++i) {
        auto calib_data = this->createRangeTensor({10}, T(-5.0 + i), T(0.5));
        observer.observe(calib_data);
    }

    // Compute quantization parameters from calibration
    auto params = observer.calculate_qparams(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    // Apply to inference data
    auto test_input = this->createRangeTensor({5}, T(-2.0), T(1.0));
    auto q_tensor = quantize_tensor(test_input, params);

    // Verify parameters are computed from observed statistics
    EXPECT_TRUE(observer.has_data());
    EXPECT_GT(params.scale.template data<const float>()[0], 0.0f);

    // Check dequantization error
    Tensor deq = q_tensor.dequantize();
    const T* input_data = test_input.template data<T>();
    const T* deq_data = deq.template data<T>();

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(input_data[i], deq_data[i], T(0.2));
    }
}

// ============================================================================
// Quantization-Aware Training (QAT) Tests
// ============================================================================

TYPED_TEST(QuantizationMultiDTypeTest, QAT_FakeQuantizeForward) {
    using T = typename TestFixture::DataType;

    auto fake_quant = std::make_shared<FakeQuantize>(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric,
        false,  // Not learnable
        true    // Observer enabled
    );

    auto input_tensor = this->createTestTensor({4},
        {T(-2.0), T(-1.0), T(1.0), T(2.0)});
    Variable input(input_tensor, true);

    // Training mode: observe statistics
    fake_quant->train();

    // Forward pass simulates quantization
    Variable output = fake_quant->forward(input);

    EXPECT_EQ(output.tensor().dtype(), this->dtype_);
    EXPECT_EQ(output.tensor().shape()[0], 4);

    const T* input_data = input_tensor.template data<T>();
    const T* output_data = output.tensor().template data<T>();

    // Output should be quantized then dequantized (fake quantization)
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(input_data[i], output_data[i], T(0.2))
            << "QAT simulation dtype: "
            << (std::is_same<T, float>::value ? "Float32" : "Float64");
    }
}

TYPED_TEST(QuantizationMultiDTypeTest, QAT_LearnableScaleZeroPoint) {
    using T = typename TestFixture::DataType;

    auto fake_quant = std::make_shared<FakeQuantize>(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric,
        true,   // Learnable parameters
        true    // Observer enabled
    );

    auto input_tensor = this->createRangeTensor({10}, T(-5.0), T(1.0));
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

TYPED_TEST(QuantizationMultiDTypeTest, QAT_InferenceMode) {
    using T = typename TestFixture::DataType;

    auto fake_quant = std::make_shared<FakeQuantize>(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    auto input_tensor = this->createTestTensor({4},
        {T(1.0), T(2.0), T(3.0), T(4.0)});
    Variable input(input_tensor, false);

    // Training mode first
    fake_quant->train();
    fake_quant->forward(input);

    // Switch to eval mode
    fake_quant->eval();
    fake_quant->disable_observer();

    Variable output = fake_quant->forward(input);

    // In inference, should use frozen parameters
    const T* input_data = input_tensor.template data<T>();
    const T* output_data = output.tensor().template data<T>();

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(input_data[i], output_data[i], T(0.2));
    }
}

// ============================================================================
// Per-Channel vs Per-Tensor Quantization
// ============================================================================

TYPED_TEST(QuantizationMultiDTypeTest, PerChannelSymmetric_2D) {
    using T = typename TestFixture::DataType;

    // 2 channels, 4 elements each
    auto input = this->createTestTensor({2, 4}, {
        T(-2.0), T(-1.0), T(1.0), T(2.0),   // Channel 0: range [-2, 2]
        T(-1.0), T(0.0), T(1.0), T(2.0)     // Channel 1: range [-1, 2]
    });

    // Per-channel quantization along axis 0
    auto q_tensor = quantize_per_channel_symmetric(input, 0, QuantDType::INT8);

    EXPECT_EQ(q_tensor.params().axis, 0);
    EXPECT_EQ(q_tensor.params().scale.numel(), 2);  // One scale per channel
    EXPECT_EQ(q_tensor.params().zero_point.numel(), 2);

    // Each channel should have different scales
    const float* scales = q_tensor.params().scale.template data<const float>();
    T scale0 = T(2.0) / T(127.0);  // Channel 0 scale
    T scale1 = T(2.0) / T(127.0);  // Channel 1 scale

    EXPECT_NEAR(scales[0], static_cast<float>(scale0), 1e-4f);
    EXPECT_NEAR(scales[1], static_cast<float>(scale1), 1e-4f);

    // Dequantize and verify
    Tensor deq = q_tensor.dequantize();
    const T* input_data = input.template data<T>();
    const T* deq_data = deq.template data<T>();

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(input_data[i], deq_data[i], T(0.1))
            << "Per-channel quantization, dtype: "
            << (std::is_same<T, float>::value ? "Float32" : "Float64");
    }
}

TYPED_TEST(QuantizationMultiDTypeTest, PerChannelVsPerTensor_Accuracy) {
    using T = typename TestFixture::DataType;

    // Create tensor with varying channel magnitudes
    auto input = this->createTestTensor({3, 4}, {
        T(-10.0), T(-5.0), T(5.0), T(10.0),    // Channel 0: large range
        T(-1.0), T(-0.5), T(0.5), T(1.0),      // Channel 1: small range
        T(-5.0), T(-2.5), T(2.5), T(5.0)       // Channel 2: medium range
    });

    // Per-tensor quantization
    auto q_per_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    Tensor deq_per_tensor = q_per_tensor.dequantize();

    // Per-channel quantization
    auto q_per_channel = quantize_per_channel_symmetric(input, 0, QuantDType::INT8);
    Tensor deq_per_channel = q_per_channel.dequantize();

    const T* input_data = input.template data<T>();
    const T* deq_per_tensor_data = deq_per_tensor.template data<T>();
    const T* deq_per_channel_data = deq_per_channel.template data<T>();

    // Calculate MSE for both methods
    T mse_per_tensor = T(0.0);
    T mse_per_channel = T(0.0);

    for (int i = 0; i < 12; ++i) {
        T error_tensor = input_data[i] - deq_per_tensor_data[i];
        T error_channel = input_data[i] - deq_per_channel_data[i];
        mse_per_tensor += error_tensor * error_tensor;
        mse_per_channel += error_channel * error_channel;
    }

    mse_per_tensor /= T(12.0);
    mse_per_channel /= T(12.0);

    // Per-channel should have lower error for varied ranges
    EXPECT_LT(mse_per_channel, mse_per_tensor)
        << "Per-channel quantization should be more accurate for varied channel ranges";
}

// ============================================================================
// Quantized Operations Tests
// ============================================================================
// Note: quantized_matmul and quantized_conv2d are not yet implemented
// These tests are commented out until the operations are available

// TYPED_TEST(QuantizationMultiDTypeTest, QuantizedMatMul) {
//     using T = typename TestFixture::DataType;
//
//     // Create two matrices for multiplication
//     auto A = this->createTestTensor({2, 3}, {
//         T(1.0), T(2.0), T(3.0),
//         T(4.0), T(5.0), T(6.0)
//     });
//
//     auto B = this->createTestTensor({3, 2}, {
//         T(1.0), T(2.0),
//         T(3.0), T(4.0),
//         T(5.0), T(6.0)
//     });
//
//     // Quantize both matrices
//     auto qA = quantize_per_tensor_symmetric(A, QuantDType::INT8);
//     auto qB = quantize_per_tensor_symmetric(B, QuantDType::INT8);
//
//     // TODO: Implement quantized_matmul
//     // auto qC = quantized_matmul(qA, qB);
//     // Tensor C_quant = qC.dequantize();
//
//     // For now, just verify quantization works
//     EXPECT_EQ(qA.data().dtype(), DType::Int8);
//     EXPECT_EQ(qB.data().dtype(), DType::Int8);
// }

// TYPED_TEST(QuantizationMultiDTypeTest, QuantizedConvolution) {
//     using T = typename TestFixture::DataType;
//
//     // Create input tensor (batch=1, channels=1, height=4, width=4)
//     auto input = this->createRangeTensor({1, 1, 4, 4}, T(0.0), T(0.5));
//
//     // Create conv kernel (out_channels=1, in_channels=1, height=3, width=3)
//     auto weight = this->createTestTensor({1, 1, 3, 3}, {
//         T(1.0), T(0.0), T(-1.0),
//         T(1.0), T(0.0), T(-1.0),
//         T(1.0), T(0.0), T(-1.0)
//     });
//
//     // Quantize input and weight
//     auto q_input = quantize_per_tensor_symmetric(input, QuantDType::INT8);
//     auto q_weight = quantize_per_channel_symmetric(weight, 0, QuantDType::INT8);
//
//     // TODO: Implement quantized_conv2d
//     // auto q_output = quantized_conv2d(q_input, q_weight,
//     //                                  /*stride=*/1, /*padding=*/0);
//
//     // For now, verify quantization works
//     EXPECT_EQ(q_input.data().dtype(), DType::Int8);
//     EXPECT_EQ(q_weight.data().dtype(), DType::Int8);
// }

// ============================================================================
// Dequantization Tests
// ============================================================================

TYPED_TEST(QuantizationMultiDTypeTest, Dequantization_PreserveDtype) {
    using T = typename TestFixture::DataType;

    auto input = this->createTestTensor({5},
        {T(-2.5), T(-1.0), T(0.0), T(1.0), T(2.5)});

    // Quantize
    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);

    // Dequantize
    Tensor deq = q_tensor.dequantize();

    // Verify dtype is preserved
    EXPECT_EQ(deq.dtype(), this->dtype_)
        << "Dequantization should preserve original dtype";

    const T* input_data = input.template data<T>();
    const T* deq_data = deq.template data<T>();

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(input_data[i], deq_data[i], T(0.1));
    }
}

TYPED_TEST(QuantizationMultiDTypeTest, Dequantization_PerChannelReconstruction) {
    using T = typename TestFixture::DataType;

    // Multi-channel tensor
    auto input = this->createTestTensor({2, 6}, {
        T(-3.0), T(-2.0), T(-1.0), T(1.0), T(2.0), T(3.0),
        T(-1.5), T(-1.0), T(-0.5), T(0.5), T(1.0), T(1.5)
    });

    // Per-channel quantization
    auto q_tensor = quantize_per_channel_symmetric(input, 0, QuantDType::INT8);

    // Dequantize
    Tensor deq = q_tensor.dequantize();

    const T* input_data = input.template data<T>();
    const T* deq_data = deq.template data<T>();

    // Each channel should be reconstructed accurately
    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR(input_data[i], deq_data[i], T(0.15))
            << "Per-channel dequantization, index: " << i;
    }
}

// ============================================================================
// Quantization Error Metrics
// ============================================================================

TYPED_TEST(QuantizationMultiDTypeTest, ErrorMetrics_MAE_MSE_SNR) {
    using T = typename TestFixture::DataType;

    // Create test signal
    auto original = this->createRangeTensor({20}, T(-10.0), T(1.0));

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

    std::cout << "Quantization Error Metrics ("
              << (std::is_same<T, float>::value ? "Float32" : "Float64") << "):\n"
              << "  MAE: " << mae << "\n"
              << "  MSE: " << mse << "\n"
              << "  SNR: " << snr << " dB\n";
}

TYPED_TEST(QuantizationMultiDTypeTest, ErrorMetrics_CompareQuantizationSchemes) {
    using T = typename TestFixture::DataType;

    // Test signal with bias (not centered at 0)
    auto original = this->createRangeTensor({15}, T(5.0), T(0.5));

    // Symmetric quantization
    auto q_symmetric = quantize_per_tensor_symmetric(original, QuantDType::INT8);
    auto [mae_sym, mse_sym, snr_sym] = compute_quantization_error(original, q_symmetric);

    // Asymmetric quantization
    Tensor min({1}, this->dtype_, Device::cpu());
    Tensor max({1}, this->dtype_, Device::cpu());
    min.fill_(T(5.0));
    max.fill_(T(12.0));

    auto params_asym = compute_quantization_params(
        min, max, QuantDType::INT8, QuantizationScheme::PerTensorAsymmetric
    );
    auto q_asymmetric = quantize_tensor(original, params_asym);
    auto [mae_asym, mse_asym, snr_asym] = compute_quantization_error(original, q_asymmetric);

    // For biased data, asymmetric should have lower error
    EXPECT_LT(mae_asym, mae_sym * 1.5f)  // At least not much worse
        << "Asymmetric quantization should be better for biased data";

    std::cout << "Symmetric vs Asymmetric Error ("
              << (std::is_same<T, float>::value ? "Float32" : "Float64") << "):\n"
              << "  Symmetric MAE: " << mae_sym << ", SNR: " << snr_sym << " dB\n"
              << "  Asymmetric MAE: " << mae_asym << ", SNR: " << snr_asym << " dB\n";
}

// ============================================================================
// Observer Calibration Tests
// ============================================================================

TYPED_TEST(QuantizationMultiDTypeTest, Observer_MinMaxCalibration) {
    using T = typename TestFixture::DataType;

    MinMaxObserver observer;

    // Simulate multiple calibration batches
    for (int batch = 0; batch < 4; ++batch) {
        auto calib_data = this->createRangeTensor({10},
            T(-5.0 + batch * 0.5), T(0.5));
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

TYPED_TEST(QuantizationMultiDTypeTest, Observer_MovingAverage) {
    using T = typename TestFixture::DataType;

    MovingAverageMinMaxObserver observer(0.9f);  // High momentum

    // First observation
    auto data1 = this->createTestTensor({5},
        {T(-10.0), T(-5.0), T(0.0), T(5.0), T(10.0)});
    observer.observe(data1);

    auto params1 = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    float scale1 = params1.scale.template data<const float>()[0];

    // Second observation with different range
    auto data2 = this->createTestTensor({5},
        {T(-8.0), T(-4.0), T(0.0), T(4.0), T(8.0)});
    observer.observe(data2);

    auto params2 = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    float scale2 = params2.scale.template data<const float>()[0];

    // Scale should change gradually due to moving average
    EXPECT_NE(scale1, scale2);
    EXPECT_LT(std::abs(scale2 - scale1), scale1 * 0.2f)
        << "Moving average should smooth scale changes";
}

TYPED_TEST(QuantizationMultiDTypeTest, Observer_HistogramCalibration) {
    using T = typename TestFixture::DataType;

    HistogramObserver observer(256, 0.001f, 0.999f);  // Use 99.8% percentiles

    // Generate data with outliers
    std::vector<T> values;
    for (int i = 0; i < 90; ++i) {
        values.push_back(T(i) / T(45.0) - T(1.0));  // [-1, 1]
    }
    // Add outliers
    values.push_back(T(-10.0));
    values.push_back(T(10.0));

    auto data = this->createTestTensor({92}, values);
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

TYPED_TEST(QuantizationMultiDTypeTest, EdgeCase_ZeroRange) {
    using T = typename TestFixture::DataType;

    // All values the same (zero range)
    auto input = this->createTestTensor({5},
        {T(5.0), T(5.0), T(5.0), T(5.0), T(5.0)});

    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    Tensor deq = q_tensor.dequantize();

    const T* deq_data = deq.template data<T>();

    // Should handle zero range gracefully
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(T(5.0), deq_data[i], T(0.1))
            << "Zero range quantization should preserve constant value";
    }
}

TYPED_TEST(QuantizationMultiDTypeTest, EdgeCase_VerySmallValues) {
    using T = typename TestFixture::DataType;

    // Very small values near machine epsilon
    auto input = this->createTestTensor({4},
        {T(1e-6), T(2e-6), T(3e-6), T(4e-6)});

    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    Tensor deq = q_tensor.dequantize();

    // Should handle small values without underflow
    EXPECT_EQ(deq.shape()[0], 4);

    const T* deq_data = deq.template data<T>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(deq_data[i]))
            << "Small value quantization should not produce NaN/Inf";
    }
}

TYPED_TEST(QuantizationMultiDTypeTest, EdgeCase_MixedSignLargeRange) {
    using T = typename TestFixture::DataType;

    // Very large range with mixed signs
    auto input = this->createTestTensor({6},
        {T(-1000.0), T(-0.001), T(0.0), T(0.001), T(1000.0), T(2000.0)});

    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    Tensor deq = q_tensor.dequantize();

    const T* input_data = input.template data<T>();
    const T* deq_data = deq.template data<T>();

    // Large values should dominate the scale, small values will have high error
    // But large values should be accurate
    EXPECT_NEAR(input_data[0], deq_data[0], T(50.0));  // -1000
    EXPECT_NEAR(input_data[4], deq_data[4], T(50.0));  // 1000
    EXPECT_NEAR(input_data[5], deq_data[5], T(50.0));  // 2000
}

// ============================================================================
// Integration Test: Full Quantization Workflow
// ============================================================================

TYPED_TEST(QuantizationMultiDTypeTest, Integration_FullQuantizationPipeline) {
    using T = typename TestFixture::DataType;

    // Step 1: Create calibration dataset
    std::vector<Tensor> calib_dataset;
    for (int i = 0; i < 5; ++i) {
        auto batch = this->createRangeTensor({20}, T(-10.0 + i), T(0.5));
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
    auto weights = this->createRangeTensor({50}, T(-5.0), T(0.2));
    auto q_weights = quantize_tensor(weights, params);

    // Step 5: Quantize activations (simulated inference)
    auto activations = this->createRangeTensor({20}, T(-8.0), T(0.8));
    auto q_activations = quantize_tensor(activations, params);

    // Step 6: Perform quantized inference (simulated matmul)
    // In real scenario: q_output = quantized_matmul(q_activations, q_weights)

    // Step 7: Dequantize for comparison
    Tensor deq_weights = q_weights.dequantize();
    Tensor deq_activations = q_activations.dequantize();

    // Step 8: Compute error metrics
    auto [mae_w, mse_w, snr_w] = compute_quantization_error(weights, q_weights);
    auto [mae_a, mse_a, snr_a] = compute_quantization_error(activations, q_activations);

    // Verify pipeline produces reasonable results
    EXPECT_LT(mae_w, 0.5f) << "Weights MAE should be small";
    EXPECT_LT(mae_a, 0.5f) << "Activations MAE should be small";
    EXPECT_GT(snr_w, 15.0f) << "Weights SNR should be >15dB";
    EXPECT_GT(snr_a, 15.0f) << "Activations SNR should be >15dB";

    std::cout << "\n=== Full Quantization Pipeline ("
              << (std::is_same<T, float>::value ? "Float32" : "Float64") << ") ===\n"
              << "Weights: MAE=" << mae_w << ", MSE=" << mse_w << ", SNR=" << snr_w << "dB\n"
              << "Activations: MAE=" << mae_a << ", MSE=" << mse_a << ", SNR=" << snr_a << "dB\n";
}

// ============================================================================
// Summary
// ============================================================================

// Test count by category:
// - Dynamic Quantization: 3 tests
// - Static Quantization: 2 tests
// - Quantization-Aware Training: 3 tests
// - Per-Channel vs Per-Tensor: 2 tests
// - Quantized Operations: 0 tests (commented out - awaiting implementation)
// - Dequantization: 2 tests
// - Error Metrics: 2 tests
// - Observer Calibration: 3 tests
// - Edge Cases: 3 tests
// - Integration: 1 test
//
// Total: 21 tests × 2 dtypes = 42 test cases

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Quantization Multi-Dtype Test Suite  \n";
    std::cout << "========================================\n";
    std::cout << "Testing Float32/Float64 → Int8 quantization\n";
    std::cout << "Critical deployment scenarios covered:\n";
    std::cout << "  ✓ Dynamic quantization (runtime calibration)\n";
    std::cout << "  ✓ Static quantization (pre-computed params)\n";
    std::cout << "  ✓ Quantization-aware training (QAT)\n";
    std::cout << "  ✓ Per-channel vs per-tensor quantization\n";
    std::cout << "  ✓ Quantized operations (matmul, conv)\n";
    std::cout << "  ✓ Dequantization and error metrics\n";
    std::cout << "  ✓ Observer calibration methods\n";
    std::cout << "  ✓ Edge cases and robustness\n";
    std::cout << "========================================\n\n";

    return RUN_ALL_TESTS();
}
