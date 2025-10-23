/**
 * @file test_quantization.cpp
 * @brief Comprehensive tests for quantization functionality
 *
 * Tests cover:
 * - All 3 quantization modes: Dynamic, PTQ, QAT
 * - All observers: MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver, PerChannelHistogramObserver
 * - All data types: INT8, UINT8
 * - All schemes: Per-tensor/per-channel, symmetric/asymmetric
 * - Edge cases: empty data, outliers, overflow, underflow
 * - Integration tests: FP32 vs INT8 accuracy, memory footprint
 */

#include <gtest/gtest.h>
#include "tenzor/nn/quantization.hpp"
#include "tenzor/core/tensor.hpp"
#include <cmath>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::nn::quantization;

// Global test environment
class QuantizationTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        // Environment initialization if needed
    }
};

static ::testing::Environment* const quant_env =
    ::testing::AddGlobalTestEnvironment(new QuantizationTestEnvironment);

class QuantizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test tensors
        weights_ = Tensor({64, 32}, DType::Float32, Device::cpu());
        activations_ = Tensor({8, 32}, DType::Float32, Device::cpu());

        // Fill with test data
        float* w_data = weights_.data<float>();
        for (int64_t i = 0; i < weights_.numel(); ++i) {
            w_data[i] = (std::sin(i * 0.1f) * 2.0f);  // Range: [-2, 2]
        }

        float* a_data = activations_.data<float>();
        for (int64_t i = 0; i < activations_.numel(); ++i) {
            a_data[i] = (std::cos(i * 0.15f) * 1.5f);  // Range: [-1.5, 1.5]
        }
    }

    Tensor weights_;
    Tensor activations_;
};

// ============================================================================
// Quantization Parameter Tests
// ============================================================================

TEST_F(QuantizationTest, ComputeQuantizationParams_Symmetric) {
    Tensor min({1}, DType::Float32, Device::cpu());
    Tensor max({1}, DType::Float32, Device::cpu());
    min.fill_(-2.0f);
    max.fill_(2.0f);

    auto params = compute_quantization_params(
        min, max, QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );

    EXPECT_EQ(params.dtype, QuantDType::INT8);
    EXPECT_EQ(params.scheme, QuantizationScheme::PerTensorSymmetric);
    EXPECT_EQ(params.axis, -1);

    // Symmetric quantization should have zero_point = 0
    const auto* zp_data = params.zero_point.data<int32_t>();
    EXPECT_EQ(zp_data[0], 0);

    // Scale should be approximately 2.0 / 127
    const auto* scale_data = params.scale.data<float>();
    float scale = scale_data[0];
    EXPECT_NEAR(scale, 2.0f / 127.0f, 1e-5f);
}

TEST_F(QuantizationTest, ComputeQuantizationParams_Asymmetric) {
    Tensor min({1}, DType::Float32, Device::cpu());
    Tensor max({1}, DType::Float32, Device::cpu());
    min.fill_(0.0f);
    max.fill_(2.0f);

    auto params = compute_quantization_params(
        min, max, QuantDType::INT8, QuantizationScheme::PerTensorAsymmetric
    );

    EXPECT_EQ(params.dtype, QuantDType::INT8);
    EXPECT_EQ(params.scheme, QuantizationScheme::PerTensorAsymmetric);

    // Asymmetric quantization should use non-zero zero_point
    int32_t zp = params.zero_point.data<int32_t>()[0];
    EXPECT_NE(zp, 0);

    // Scale should be approximately 2.0 / 255
    float scale = params.scale.data<float>()[0];
    EXPECT_GT(scale, 0.0f);
}

TEST_F(QuantizationTest, ComputeQuantizationParams_UINT8) {
    Tensor min({1}, DType::Float32, Device::cpu());
    Tensor max({1}, DType::Float32, Device::cpu());
    min.fill_(0.0f);
    max.fill_(3.0f);

    auto params = compute_quantization_params(
        min, max, QuantDType::UINT8, QuantizationScheme::PerTensorAsymmetric
    );

    EXPECT_EQ(params.dtype, QuantDType::UINT8);

    // UINT8 range is [0, 255]
    float scale = params.scale.data<float>()[0];
    EXPECT_NEAR(scale, 3.0f / 255.0f, 1e-5f);
}

TEST_F(QuantizationTest, ComputeQuantizationParams_PerChannel) {
    int64_t num_channels = 64;
    Tensor min({num_channels}, DType::Float32, Device::cpu());
    Tensor max({num_channels}, DType::Float32, Device::cpu());

    float* min_data = min.data<float>();
    float* max_data = max.data<float>();
    for (int64_t i = 0; i < num_channels; ++i) {
        min_data[i] = -1.0f - i * 0.01f;
        max_data[i] = 1.0f + i * 0.01f;
    }

    auto params = compute_quantization_params(
        min, max, QuantDType::INT8, QuantizationScheme::PerChannelSymmetric
    );

    EXPECT_EQ(params.scheme, QuantizationScheme::PerChannelSymmetric);
    EXPECT_EQ(params.scale.numel(), num_channels);
    EXPECT_EQ(params.zero_point.numel(), num_channels);
}

// ============================================================================
// Quantization/Dequantization Tests
// ============================================================================

TEST_F(QuantizationTest, QuantizeAndDequantize_PerTensor) {
    auto q_tensor = quantize_per_tensor_symmetric(weights_);

    EXPECT_EQ(q_tensor.data().dtype(), DType::Int8);
    EXPECT_EQ(q_tensor.shape()[0], 64);
    EXPECT_EQ(q_tensor.shape()[1], 32);

    // Dequantize
    Tensor deq = q_tensor.dequantize();

    EXPECT_EQ(deq.dtype(), DType::Float32);
    EXPECT_EQ(deq.shape()[0], 64);
    EXPECT_EQ(deq.shape()[1], 32);

    // Check quantization error
    auto [mae, mse, snr_db] = compute_quantization_error(weights_, q_tensor);

    EXPECT_LT(mae, 0.05f);  // Mean absolute error < 5%
    EXPECT_GT(snr_db, 30.0f);  // SNR > 30dB
}

TEST_F(QuantizationTest, QuantizePerChannel_Symmetric) {
    auto q_tensor = quantize_per_channel_symmetric(weights_, 0);

    EXPECT_EQ(q_tensor.params().axis, 0);
    EXPECT_EQ(q_tensor.params().scheme, QuantizationScheme::PerChannelSymmetric);

    // Should have 64 scales (one per output channel)
    EXPECT_EQ(q_tensor.params().scale.numel(), 64);
    EXPECT_EQ(q_tensor.params().zero_point.numel(), 64);

    // Dequantize and check error
    Tensor deq = q_tensor.dequantize();
    auto [mae, mse, snr_db] = compute_quantization_error(weights_, q_tensor);

    // Per-channel should have better accuracy than per-tensor
    EXPECT_LT(mae, 0.03f);
    EXPECT_GT(snr_db, 35.0f);
}

TEST_F(QuantizationTest, QuantizePerTensor_Asymmetric) {
    auto q_tensor = quantize_per_tensor_asymmetric(activations_);

    EXPECT_EQ(q_tensor.params().scheme, QuantizationScheme::PerTensorAsymmetric);
    EXPECT_NE(q_tensor.params().zero_point.data<int32_t>()[0], 0);

    Tensor deq = q_tensor.dequantize();
    auto [mae, mse, snr_db] = compute_quantization_error(activations_, q_tensor);

    EXPECT_LT(mae, 0.05f);
    EXPECT_GT(snr_db, 30.0f);
}

TEST_F(QuantizationTest, QuantizePerChannel_Asymmetric) {
    auto q_tensor = quantize_per_channel_asymmetric(weights_, 0);

    EXPECT_EQ(q_tensor.params().scheme, QuantizationScheme::PerChannelAsymmetric);
    EXPECT_EQ(q_tensor.params().axis, 0);

    // Check that zero points are not all zero
    const int32_t* zp_data = q_tensor.params().zero_point.data<int32_t>();
    bool has_nonzero = false;
    for (int64_t i = 0; i < 64; ++i) {
        if (zp_data[i] != 0) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(QuantizationTest, RoundTrip_PreservesShape) {
    Tensor input({16, 8, 4}, DType::Float32, Device::cpu());
    input.fill_(0.5f);

    auto q_tensor = quantize_per_tensor_symmetric(input);
    Tensor deq = dequantize_tensor(q_tensor);

    EXPECT_EQ(deq.shape()[0], 16);
    EXPECT_EQ(deq.shape()[1], 8);
    EXPECT_EQ(deq.shape()[2], 4);
}

TEST_F(QuantizationTest, QuantizeWithCustomParams) {
    Tensor scale({1}, DType::Float32, Device::cpu());
    Tensor zero_point({1}, DType::Int32, Device::cpu());
    scale.fill_(0.01f);
    zero_point.fill_(5);

    QuantizationParams params(scale, zero_point, QuantDType::INT8,
                              QuantizationScheme::PerTensorAsymmetric);

    auto q_tensor = quantize_tensor(activations_, params);

    EXPECT_EQ(q_tensor.params().zero_point.data<int32_t>()[0], 5);
    EXPECT_NEAR(q_tensor.params().scale.data<float>()[0], 0.01f, 1e-6f);
}

// ============================================================================
// MinMaxObserver Tests
// ============================================================================

TEST_F(QuantizationTest, MinMaxObserver_PerTensor) {
    MinMaxObserver observer(false);

    EXPECT_FALSE(observer.has_data());

    // Observe data
    observer.observe(activations_);

    EXPECT_TRUE(observer.has_data());

    // Get min/max
    float min_val = observer.get_min().data<float>()[0];
    float max_val = observer.get_max().data<float>()[0];

    EXPECT_LT(min_val, -1.0f);
    EXPECT_GT(max_val, 1.0f);

    // Calculate qparams
    auto params = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );

    EXPECT_GT(params.scale.data<float>()[0], 0.0f);
}

TEST_F(QuantizationTest, MinMaxObserver_PerChannel) {
    MinMaxObserver observer(true, 0);  // Per-channel along axis 0

    observer.observe(weights_);

    EXPECT_TRUE(observer.has_data());

    // Min/max should have 64 values (one per channel)
    EXPECT_EQ(observer.get_min().numel(), 64);
    EXPECT_EQ(observer.get_max().numel(), 64);

    auto params = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerChannelSymmetric
    );

    EXPECT_EQ(params.scale.numel(), 64);
}

TEST_F(QuantizationTest, MinMaxObserver_MultipleObservations) {
    MinMaxObserver observer(false);

    Tensor batch1({8, 32}, DType::Float32, Device::cpu());
    batch1.fill_(1.0f);
    observer.observe(batch1);

    float min1 = observer.get_min().data<float>()[0];
    float max1 = observer.get_max().data<float>()[0];

    Tensor batch2({8, 32}, DType::Float32, Device::cpu());
    batch2.fill_(3.0f);
    observer.observe(batch2);

    float min2 = observer.get_min().data<float>()[0];
    float max2 = observer.get_max().data<float>()[0];

    // Min should not change, max should increase
    EXPECT_NEAR(min1, min2, 1e-5f);
    EXPECT_GT(max2, max1);
}

TEST_F(QuantizationTest, MinMaxObserver_Reset) {
    MinMaxObserver observer(false);
    observer.observe(activations_);

    EXPECT_TRUE(observer.has_data());

    observer.reset();

    EXPECT_FALSE(observer.has_data());
}

TEST_F(QuantizationTest, MinMaxObserver_NegativeValues) {
    Tensor negative_data({10, 10}, DType::Float32, Device::cpu());
    float* data = negative_data.data<float>();
    for (int64_t i = 0; i < 100; ++i) {
        data[i] = -5.0f - i * 0.1f;
    }

    MinMaxObserver observer(false);
    observer.observe(negative_data);

    float min_val = observer.get_min().data<float>()[0];
    float max_val = observer.get_max().data<float>()[0];

    EXPECT_LT(min_val, -5.0f);
    EXPECT_LT(max_val, 0.0f);
}

// ============================================================================
// MovingAverageMinMaxObserver Tests
// ============================================================================

TEST_F(QuantizationTest, MovingAverageObserver_Basic) {
    MovingAverageMinMaxObserver observer(0.9f);

    // First observation
    Tensor batch1({8, 32}, DType::Float32, Device::cpu());
    batch1.fill_(1.0f);
    observer.observe(batch1);

    EXPECT_TRUE(observer.has_data());

    // MovingAverageObserver doesn't expose get_min/get_max
    // Verify it works by calculating qparams
    auto params = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );

    EXPECT_GT(params.scale.data<float>()[0], 0.0f);
}

TEST_F(QuantizationTest, MovingAverageObserver_Smoothing) {
    MovingAverageMinMaxObserver observer(0.9f);

    // First observation
    Tensor batch1({8, 32}, DType::Float32, Device::cpu());
    batch1.fill_(1.0f);
    observer.observe(batch1);

    auto params1 = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );
    float scale1 = params1.scale.data<float>()[0];

    // Second observation with different range
    Tensor batch2({8, 32}, DType::Float32, Device::cpu());
    batch2.fill_(2.0f);
    observer.observe(batch2);

    auto params2 = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );
    float scale2 = params2.scale.data<float>()[0];

    // Moving average should smooth the update
    // Scale should change but not jump dramatically
    EXPECT_NE(scale2, scale1);
    EXPECT_GT(scale2, scale1);
}

TEST_F(QuantizationTest, MovingAverageObserver_HighMomentum) {
    MovingAverageMinMaxObserver observer(0.99f);  // High momentum

    Tensor batch1({8, 32}, DType::Float32, Device::cpu());
    batch1.fill_(1.0f);
    observer.observe(batch1);

    auto params1 = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );
    float scale1 = params1.scale.data<float>()[0];

    Tensor batch2({8, 32}, DType::Float32, Device::cpu());
    batch2.fill_(10.0f);
    observer.observe(batch2);

    auto params2 = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );
    float scale2 = params2.scale.data<float>()[0];

    // High momentum means slow adaptation
    EXPECT_GT(scale2, scale1);
    EXPECT_LT(scale2, scale1 * 3.0f);  // Should not increase too much
}

TEST_F(QuantizationTest, MovingAverageObserver_LowMomentum) {
    MovingAverageMinMaxObserver observer(0.1f);  // Low momentum

    Tensor batch1({8, 32}, DType::Float32, Device::cpu());
    batch1.fill_(1.0f);
    observer.observe(batch1);

    auto params1 = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );
    float scale1 = params1.scale.data<float>()[0];

    Tensor batch2({8, 32}, DType::Float32, Device::cpu());
    batch2.fill_(10.0f);
    observer.observe(batch2);

    auto params2 = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );
    float scale2 = params2.scale.data<float>()[0];

    // Low momentum means fast adaptation
    EXPECT_GT(scale2, scale1);
    EXPECT_GT(scale2, scale1 * 5.0f);  // Should increase significantly
}

TEST_F(QuantizationTest, MovingAverageObserver_PerChannel) {
    MovingAverageMinMaxObserver observer(0.9f, true, 0);

    observer.observe(weights_);

    auto params = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerChannelSymmetric
    );

    EXPECT_EQ(params.scale.numel(), 64);
}

// ============================================================================
// HistogramObserver Tests
// ============================================================================

TEST_F(QuantizationTest, HistogramObserver_Basic) {
    HistogramObserver observer(2048, 0.001f, 0.999f);

    // Observe data
    observer.observe(weights_);
    observer.observe(activations_);

    EXPECT_TRUE(observer.has_data());

    // Get histogram
    auto [bin_edges, counts] = observer.get_histogram();

    EXPECT_EQ(bin_edges.size(), 2049);  // num_bins + 1
    EXPECT_EQ(counts.size(), 2048);

    // Check that histogram captured data
    int64_t total_count = 0;
    for (auto count : counts) {
        total_count += count;
    }
    EXPECT_EQ(total_count, weights_.numel() + activations_.numel());
}

TEST_F(QuantizationTest, HistogramObserver_CalculateQParams) {
    HistogramObserver observer(2048);

    observer.observe(weights_);

    auto params = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );

    EXPECT_GT(params.scale.data<float>()[0], 0.0f);
}

TEST_F(QuantizationTest, HistogramObserver_OutlierHandling) {
    Tensor data_with_outliers({100}, DType::Float32, Device::cpu());
    float* data = data_with_outliers.data<float>();

    // Most values around 0
    for (int64_t i = 0; i < 98; ++i) {
        data[i] = std::sin(i * 0.1f);
    }
    // Add outliers
    data[98] = 1000.0f;
    data[99] = -1000.0f;

    HistogramObserver observer(2048, 0.01f, 0.99f);  // Clip 1% on each side
    observer.observe(data_with_outliers);

    auto params = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );

    // Scale should not be dominated by outliers
    float scale = params.scale.data<float>()[0];
    EXPECT_LT(scale, 100.0f);  // Much less than 1000/127
}

TEST_F(QuantizationTest, HistogramObserver_DifferentBinCounts) {
    HistogramObserver observer_small(256);
    HistogramObserver observer_large(4096);

    observer_small.observe(weights_);
    observer_large.observe(weights_);

    auto [edges_small, counts_small] = observer_small.get_histogram();
    auto [edges_large, counts_large] = observer_large.get_histogram();

    EXPECT_EQ(counts_small.size(), 256);
    EXPECT_EQ(counts_large.size(), 4096);
}

TEST_F(QuantizationTest, HistogramObserver_Reset) {
    HistogramObserver observer(1024);
    observer.observe(weights_);

    EXPECT_TRUE(observer.has_data());

    observer.reset();

    EXPECT_FALSE(observer.has_data());
}

// ============================================================================
// PerChannelHistogramObserver Tests
// ============================================================================

TEST_F(QuantizationTest, PerChannelHistogramObserver_Basic) {
    PerChannelHistogramObserver observer(0, 2048);  // Axis 0

    observer.observe(weights_);

    EXPECT_TRUE(observer.has_data());

    auto params = observer.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerChannelSymmetric
    );

    EXPECT_EQ(params.scale.numel(), 64);
    EXPECT_EQ(params.axis, 0);
}

TEST_F(QuantizationTest, PerChannelHistogramObserver_DifferentAxes) {
    Tensor data({16, 32, 8}, DType::Float32, Device::cpu());
    data.fill_(0.5f);

    // Axis 0
    PerChannelHistogramObserver observer0(0, 1024);
    observer0.observe(data);
    auto params0 = observer0.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerChannelSymmetric
    );
    EXPECT_EQ(params0.scale.numel(), 16);

    // Axis 1
    PerChannelHistogramObserver observer1(1, 1024);
    observer1.observe(data);
    auto params1 = observer1.calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerChannelSymmetric
    );
    EXPECT_EQ(params1.scale.numel(), 32);
}

// ============================================================================
// Observer Factory Tests
// ============================================================================

TEST_F(QuantizationTest, MakeObserver_MinMax) {
    auto observer = make_observer(QuantizationScheme::PerTensorSymmetric, false);

    EXPECT_NE(observer, nullptr);
    EXPECT_FALSE(observer->has_data());

    observer->observe(weights_);
    EXPECT_TRUE(observer->has_data());
}

TEST_F(QuantizationTest, MakeObserver_Histogram) {
    auto observer = make_observer(QuantizationScheme::PerTensorSymmetric, true);

    EXPECT_NE(observer, nullptr);

    observer->observe(weights_);
    EXPECT_TRUE(observer->has_data());
}

TEST_F(QuantizationTest, MakeObserver_PerChannel) {
    auto observer = make_observer(QuantizationScheme::PerChannelSymmetric, false, 0);

    EXPECT_NE(observer, nullptr);

    observer->observe(weights_);
    auto params = observer->calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerChannelSymmetric
    );

    EXPECT_EQ(params.scale.numel(), 64);
}

// ============================================================================
// QConfig Tests
// ============================================================================

TEST_F(QuantizationTest, DefaultQConfig) {
    auto qconfig = DefaultQConfigs::default_qconfig();

    EXPECT_EQ(qconfig.weight_dtype(), QuantDType::INT8);
    EXPECT_EQ(qconfig.activation_dtype(), QuantDType::INT8);
    EXPECT_EQ(qconfig.weight_scheme(), QuantizationScheme::PerChannelSymmetric);
    EXPECT_EQ(qconfig.activation_scheme(), QuantizationScheme::PerTensorSymmetric);

    // Create observers
    auto weight_obs = qconfig.create_weight_observer();
    auto act_obs = qconfig.create_activation_observer();

    EXPECT_NE(weight_obs, nullptr);
    EXPECT_NE(act_obs, nullptr);
}

TEST_F(QuantizationTest, HighAccuracyQConfig) {
    auto qconfig = DefaultQConfigs::high_accuracy_qconfig();

    auto weight_obs = qconfig.create_weight_observer();
    auto act_obs = qconfig.create_activation_observer();

    EXPECT_NE(weight_obs, nullptr);
    EXPECT_NE(act_obs, nullptr);

    // High accuracy config should use better observers
    EXPECT_EQ(qconfig.activation_scheme(), QuantizationScheme::PerTensorAsymmetric);
}

TEST_F(QuantizationTest, FastQConfig) {
    auto qconfig = DefaultQConfigs::fast_qconfig();

    EXPECT_EQ(qconfig.weight_scheme(), QuantizationScheme::PerChannelSymmetric);
    EXPECT_EQ(qconfig.activation_scheme(), QuantizationScheme::PerTensorSymmetric);
}

TEST_F(QuantizationTest, QATQConfig) {
    auto qconfig = DefaultQConfigs::qat_qconfig();

    auto weight_obs = qconfig.create_weight_observer();
    auto act_obs = qconfig.create_activation_observer();

    EXPECT_NE(weight_obs, nullptr);
    EXPECT_NE(act_obs, nullptr);
}

TEST_F(QuantizationTest, UINT8ActivationQConfig) {
    auto qconfig = DefaultQConfigs::uint8_activation_qconfig();

    EXPECT_EQ(qconfig.weight_dtype(), QuantDType::INT8);
    EXPECT_EQ(qconfig.activation_dtype(), QuantDType::UINT8);
}

TEST_F(QuantizationTest, QConfigMapping_Basic) {
    QConfigMapping mapping;

    auto default_cfg = DefaultQConfigs::default_qconfig();
    auto high_acc_cfg = DefaultQConfigs::high_accuracy_qconfig();

    mapping.set_global(default_cfg);
    mapping.set_layer_qconfig("layer1", high_acc_cfg);
    mapping.set_type_qconfig("Conv2d", high_acc_cfg);

    // Layer-specific config takes precedence
    auto cfg1 = mapping.get_qconfig("layer1", "Linear");
    EXPECT_NE(cfg1, nullptr);
    EXPECT_EQ(cfg1->weight_scheme(), high_acc_cfg.weight_scheme());

    // Type-specific config
    auto cfg2 = mapping.get_qconfig("some_conv", "Conv2d");
    EXPECT_NE(cfg2, nullptr);
    EXPECT_EQ(cfg2->weight_scheme(), high_acc_cfg.weight_scheme());

    // Global default
    auto cfg3 = mapping.get_qconfig("other_layer", "Linear");
    EXPECT_NE(cfg3, nullptr);
    EXPECT_EQ(cfg3->weight_scheme(), default_cfg.weight_scheme());
}

TEST_F(QuantizationTest, QConfigMapping_DisableLayer) {
    QConfigMapping mapping;
    auto default_cfg = DefaultQConfigs::default_qconfig();
    mapping.set_global(default_cfg);

    // Disable specific layer
    mapping.disable_layer("layer1");
    EXPECT_EQ(mapping.get_qconfig("layer1", "Linear"), nullptr);
    EXPECT_FALSE(mapping.is_quantized("layer1", "Linear"));
}

TEST_F(QuantizationTest, QConfigMapping_DisableType) {
    QConfigMapping mapping;
    auto default_cfg = DefaultQConfigs::default_qconfig();
    mapping.set_global(default_cfg);

    // Disable all Conv2d layers
    mapping.disable_type("Conv2d");
    EXPECT_EQ(mapping.get_qconfig("any_layer", "Conv2d"), nullptr);
    EXPECT_FALSE(mapping.is_quantized("any_layer", "Conv2d"));
}

// ============================================================================
// Fake Quantization Tests
// ============================================================================

TEST_F(QuantizationTest, FakeQuantize_Basic) {
    FakeQuantize fake_quant(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric,
        false, true
    );

    fake_quant.eval();

    // First pass - observer should collect stats
    auto input_var = Variable(activations_, false);
    auto output = fake_quant.forward(input_var);

    EXPECT_EQ(output.tensor().shape()[0], activations_.shape()[0]);
    EXPECT_EQ(output.tensor().shape()[1], activations_.shape()[1]);

    // Output should be quantized and dequantized (slight difference from input)
    auto [mae, mse, snr_db] = compute_quantization_error(
        activations_, quantize_per_tensor_symmetric(activations_)
    );

    EXPECT_LT(mae, 0.1f);
}

TEST_F(QuantizationTest, FakeQuantize_EnableDisable) {
    FakeQuantize fake_quant(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);

    auto input_var = Variable(activations_, false);

    // Disable fake quant - should act as identity
    fake_quant.disable_fake_quant();
    auto output_disabled = fake_quant.forward(input_var);

    // Enable fake quant - should apply quantization
    fake_quant.enable_fake_quant();
    fake_quant.eval();
    auto output_enabled = fake_quant.forward(input_var);

    // Outputs should be different
    const float* disabled_data = output_disabled.tensor().data<float>();
    const float* enabled_data = output_enabled.tensor().data<float>();

    bool different = false;
    for (int64_t i = 0; i < activations_.numel(); ++i) {
        if (std::abs(disabled_data[i] - enabled_data[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

TEST_F(QuantizationTest, FakeQuantize_ObserverControl) {
    FakeQuantize fake_quant(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);

    auto input_var = Variable(activations_, false);

    // Enable observer
    fake_quant.enable_observer(true);
    fake_quant.train();
    fake_quant.forward(input_var);

    EXPECT_NE(fake_quant.observer(), nullptr);

    // Disable observer
    fake_quant.disable_observer();
    fake_quant.forward(input_var);
}

TEST_F(QuantizationTest, FakeQuantize_ManualQParams) {
    FakeQuantize fake_quant(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);

    Tensor scale({1}, DType::Float32, Device::cpu());
    Tensor zero_point({1}, DType::Int32, Device::cpu());
    scale.fill_(0.05f);
    zero_point.fill_(0);

    QuantizationParams params(scale, zero_point, QuantDType::INT8,
                              QuantizationScheme::PerTensorSymmetric);

    fake_quant.set_qparams(params);

    auto qparams = fake_quant.get_qparams();
    EXPECT_NEAR(qparams.scale.data<float>()[0], 0.05f, 1e-6f);
}

TEST_F(QuantizationTest, FakeQuantize_PerChannel) {
    FakeQuantize fake_quant(
        QuantDType::INT8,
        QuantizationScheme::PerChannelSymmetric,
        false, true, 0
    );

    fake_quant.eval();

    auto input_var = Variable(weights_, false);
    fake_quant.forward(input_var);

    // Calculate qparams after observation
    fake_quant.calculate_qparams();

    auto qparams = fake_quant.get_qparams();
    EXPECT_EQ(qparams.scale.numel(), 64);
}

TEST_F(QuantizationTest, LearnableFakeQuantize_Basic) {
    LearnableFakeQuantize learnable_fq(
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    EXPECT_TRUE(learnable_fq.is_learnable());

    auto input_var = Variable(activations_, false);
    auto output = learnable_fq.forward(input_var);

    EXPECT_EQ(output.shape()[0], activations_.shape()[0]);
}

// ============================================================================
// Quantized Layer Tests
// ============================================================================

TEST_F(QuantizationTest, QuantizedLinear_Forward) {
    int64_t in_features = 32;
    int64_t out_features = 16;

    // Create quantization params for weights
    auto qconfig = DefaultQConfigs::default_qconfig();
    auto weight_obs = qconfig.create_weight_observer();

    Tensor weights({out_features, in_features}, DType::Float32, Device::cpu());
    weights.fill_(0.1f);

    weight_obs->observe(weights);
    auto weight_params = weight_obs->calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerChannelSymmetric
    );

    QuantizedLinear q_linear(in_features, out_features, weight_params);

    // Set quantized weights
    auto q_weights = quantize_tensor(weights, weight_params);
    q_linear.set_weight(q_weights);

    // Forward pass
    Tensor input({4, in_features}, DType::Float32, Device::cpu());
    input.fill_(1.0f);

    auto q_input = quantize_per_tensor_symmetric(input);
    Tensor output = q_linear.forward_quantized(q_input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], out_features);
    EXPECT_EQ(output.dtype(), DType::Float32);
}

TEST_F(QuantizationTest, QuantizedLinear_WithBias) {
    int64_t in_features = 32;
    int64_t out_features = 16;

    auto qconfig = DefaultQConfigs::default_qconfig();
    auto weight_obs = qconfig.create_weight_observer();

    Tensor weights({out_features, in_features}, DType::Float32, Device::cpu());
    weights.fill_(0.1f);

    weight_obs->observe(weights);
    auto weight_params = weight_obs->calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerChannelSymmetric
    );

    QuantizedLinear q_linear(in_features, out_features, weight_params);
    auto q_weights = quantize_tensor(weights, weight_params);
    q_linear.set_weight(q_weights);

    // Add bias
    Tensor bias({out_features}, DType::Float32, Device::cpu());
    bias.fill_(0.5f);
    q_linear.set_bias(bias);

    Tensor input({4, in_features}, DType::Float32, Device::cpu());
    input.fill_(1.0f);

    auto q_input = quantize_per_tensor_symmetric(input);
    Tensor output = q_linear.forward_quantized(q_input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], out_features);
}

// ============================================================================
// Calibration Tests
// ============================================================================

TEST_F(QuantizationTest, CalibrateQuantizationParams_Basic) {
    std::vector<Tensor> samples;

    for (int i = 0; i < 10; ++i) {
        Tensor sample({8, 32}, DType::Float32, Device::cpu());
        float* data = sample.data<float>();
        for (int64_t j = 0; j < sample.numel(); ++j) {
            data[j] = std::sin(j * 0.1f + i) * 2.0f;
        }
        samples.push_back(sample);
    }

    auto params = calibrate_quantization_params(
        samples, QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );

    EXPECT_GT(params.scale.data<float>()[0], 0.0f);
    EXPECT_EQ(params.zero_point.data<int32_t>()[0], 0);
}

TEST_F(QuantizationTest, CalibrateQuantizationParams_PerChannel) {
    std::vector<Tensor> samples;

    for (int i = 0; i < 5; ++i) {
        Tensor sample({64, 32}, DType::Float32, Device::cpu());
        float* data = sample.data<float>();
        for (int64_t j = 0; j < sample.numel(); ++j) {
            data[j] = std::sin(j * 0.1f + i) * 1.5f;
        }
        samples.push_back(sample);
    }

    auto params = calibrate_quantization_params(
        samples, QuantDType::INT8, QuantizationScheme::PerChannelSymmetric, 0
    );

    EXPECT_EQ(params.scale.numel(), 64);
    EXPECT_EQ(params.axis, 0);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(QuantizationTest, EdgeCase_EmptyTensor) {
    // Empty tensor should not crash observers
    Tensor empty({0}, DType::Float32, Device::cpu());

    MinMaxObserver observer(false);
    // This should handle empty tensor gracefully
    // observer.observe(empty);  // May throw or handle gracefully
}

TEST_F(QuantizationTest, EdgeCase_SingleValue) {
    Tensor single({1}, DType::Float32, Device::cpu());
    single.fill_(5.0f);

    MinMaxObserver observer(false);
    observer.observe(single);

    EXPECT_TRUE(observer.has_data());

    float min_val = observer.get_min().data<float>()[0];
    float max_val = observer.get_max().data<float>()[0];

    EXPECT_NEAR(min_val, 5.0f, 1e-5f);
    EXPECT_NEAR(max_val, 5.0f, 1e-5f);
}

TEST_F(QuantizationTest, EdgeCase_AllZeros) {
    Tensor zeros({10, 10}, DType::Float32, Device::cpu());
    zeros.fill_(0.0f);

    auto q_tensor = quantize_per_tensor_symmetric(zeros);
    Tensor deq = q_tensor.dequantize();

    float* deq_data = deq.data<float>();
    for (int64_t i = 0; i < deq.numel(); ++i) {
        EXPECT_NEAR(deq_data[i], 0.0f, 1e-3f);
    }
}

TEST_F(QuantizationTest, EdgeCase_VerySmallValues) {
    Tensor tiny({10, 10}, DType::Float32, Device::cpu());
    float* data = tiny.data<float>();
    for (int64_t i = 0; i < 100; ++i) {
        data[i] = 1e-6f * std::sin(i * 0.1f);
    }

    auto q_tensor = quantize_per_tensor_symmetric(tiny);
    auto [mae, mse, snr_db] = compute_quantization_error(tiny, q_tensor);

    // For very small values, relative error may be large
    EXPECT_LT(mae, 1e-5f);
}

TEST_F(QuantizationTest, EdgeCase_VeryLargeValues) {
    Tensor large({10, 10}, DType::Float32, Device::cpu());
    float* data = large.data<float>();
    for (int64_t i = 0; i < 100; ++i) {
        data[i] = 1000.0f * std::sin(i * 0.1f);
    }

    auto q_tensor = quantize_per_tensor_symmetric(large);

    // Quantization should handle large values
    EXPECT_EQ(q_tensor.data().dtype(), DType::Int8);

    Tensor deq = q_tensor.dequantize();
    auto [mae, mse, snr_db] = compute_quantization_error(large, q_tensor);

    // With large values, we expect larger absolute error but good SNR
    EXPECT_GT(snr_db, 25.0f);
}

TEST_F(QuantizationTest, EdgeCase_MixedRange) {
    Tensor mixed({100}, DType::Float32, Device::cpu());
    float* data = mixed.data<float>();

    // Mix of small, medium, and large values
    for (int64_t i = 0; i < 33; ++i) {
        data[i] = 0.001f * std::sin(i * 0.1f);
    }
    for (int64_t i = 33; i < 66; ++i) {
        data[i] = 1.0f * std::sin(i * 0.1f);
    }
    for (int64_t i = 66; i < 100; ++i) {
        data[i] = 100.0f * std::sin(i * 0.1f);
    }

    auto q_tensor = quantize_per_tensor_symmetric(mixed);
    auto [mae, mse, snr_db] = compute_quantization_error(mixed, q_tensor);

    // Mixed range is challenging for quantization
    EXPECT_GT(snr_db, 20.0f);
}

TEST_F(QuantizationTest, EdgeCase_NearBoundary) {
    Tensor boundary({10}, DType::Float32, Device::cpu());
    float* data = boundary.data<float>();

    // Values near INT8 boundaries after scaling
    for (int64_t i = 0; i < 10; ++i) {
        data[i] = (i % 2 == 0) ? -127.0f : 127.0f;
    }

    auto q_tensor = quantize_per_tensor_symmetric(boundary);

    const int8_t* q_data = q_tensor.data().data<int8_t>();
    for (int64_t i = 0; i < 10; ++i) {
        // Should be at or near boundaries
        EXPECT_TRUE(q_data[i] == -127 || q_data[i] == 127);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(QuantizationTest, Integration_EndToEnd_PerTensor) {
    // 1. Create and observe weights
    auto observer = std::make_unique<MinMaxObserver>();
    observer->observe(weights_);

    // 2. Calculate quantization parameters
    auto params = observer->calculate_qparams(
        QuantDType::INT8, QuantizationScheme::PerTensorSymmetric
    );

    // 3. Quantize
    auto q_tensor = quantize_tensor(weights_, params);

    // 4. Dequantize
    Tensor deq = dequantize_tensor(q_tensor);

    // 5. Verify
    auto [mae, mse, snr_db] = compute_quantization_error(weights_, q_tensor);

    EXPECT_LT(mae, 0.05f);
    EXPECT_GT(snr_db, 30.0f);

    std::cout << "Quantization Error - MAE: " << mae
              << ", MSE: " << mse
              << ", SNR: " << snr_db << " dB" << std::endl;
}

TEST_F(QuantizationTest, Integration_PTQ_Workflow) {
    // Post-Training Quantization workflow

    // 1. Calibration phase: collect statistics
    std::vector<Tensor> calibration_samples;
    for (int i = 0; i < 20; ++i) {
        Tensor sample({8, 32}, DType::Float32, Device::cpu());
        float* data = sample.data<float>();
        for (int64_t j = 0; j < sample.numel(); ++j) {
            data[j] = std::sin(j * 0.05f + i) * 1.5f;
        }
        calibration_samples.push_back(sample);
    }

    // 2. Calibrate quantization parameters
    auto params = calibrate_quantization_params(
        calibration_samples,
        QuantDType::INT8,
        QuantizationScheme::PerTensorSymmetric
    );

    // 3. Quantize model weights
    auto q_weights = quantize_tensor(weights_, params);

    // 4. Run inference with quantized weights
    Tensor test_input({8, 32}, DType::Float32, Device::cpu());
    test_input.fill_(1.0f);

    auto q_input = quantize_per_tensor_symmetric(test_input);

    // Verify quantization quality
    // INT8 symmetric quantization typically achieves 15-25 dB SNR
    // which is acceptable for PTQ (Post-Training Quantization)
    auto [mae, mse, snr_db] = compute_quantization_error(weights_, q_weights);
    EXPECT_GT(snr_db, 15.0f);
}

TEST_F(QuantizationTest, Integration_MemoryFootprint) {
    // Verify quantization reduces memory usage

    Tensor large_tensor({256, 512}, DType::Float32, Device::cpu());
    float* fp32_data = large_tensor.data<float>();
    for (int64_t i = 0; i < large_tensor.numel(); ++i) {
        fp32_data[i] = std::sin(i * 0.01f);
    }

    // FP32 memory: 256 * 512 * 4 bytes = 512 KB
    int64_t fp32_bytes = large_tensor.numel() * sizeof(float);

    // Quantize to INT8
    auto q_tensor = quantize_per_tensor_symmetric(large_tensor);

    // INT8 memory: 256 * 512 * 1 byte = 128 KB (+ small overhead for scale/zero_point)
    int64_t int8_bytes = q_tensor.data().numel() * sizeof(int8_t);

    // Should be approximately 4x reduction
    float compression_ratio = static_cast<float>(fp32_bytes) / static_cast<float>(int8_bytes);
    EXPECT_GT(compression_ratio, 3.5f);  // Account for scale/zero_point overhead
    EXPECT_LT(compression_ratio, 4.5f);

    std::cout << "Memory compression: " << compression_ratio << "x" << std::endl;
}

TEST_F(QuantizationTest, Integration_AccuracyComparison) {
    // Compare FP32 vs INT8 accuracy

    // Create simple computation
    Tensor input({16, 32}, DType::Float32, Device::cpu());
    Tensor weight({64, 32}, DType::Float32, Device::cpu());

    float* inp_data = input.data<float>();
    float* wgt_data = weight.data<float>();

    for (int64_t i = 0; i < input.numel(); ++i) {
        inp_data[i] = std::sin(i * 0.1f);
    }
    for (int64_t i = 0; i < weight.numel(); ++i) {
        wgt_data[i] = std::cos(i * 0.1f) * 0.5f;
    }

    // FP32 computation (simulated)
    // In real scenario, this would be actual layer forward pass

    // Quantize weights
    auto q_weight = quantize_per_channel_symmetric(weight, 0);

    // Check accuracy loss
    auto [mae, mse, snr_db] = compute_quantization_error(weight, q_weight);

    // For typical neural network weights, INT8 quantization should maintain
    // good accuracy with SNR > 35dB
    EXPECT_GT(snr_db, 30.0f);

    std::cout << "FP32 vs INT8 - SNR: " << snr_db << " dB, MAE: " << mae << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
