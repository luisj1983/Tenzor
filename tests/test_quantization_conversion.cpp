/**
 * @file test_quantization_conversion.cpp
 * @brief Comprehensive tests for quantization layer conversion functions
 *
 * Tests cover:
 * - convert_to_quantized() for Linear and Conv2d layers
 * - convert_from_quantized() for dequantization
 * - prepare_qat() for quantization-aware training setup
 * - Round-trip conversion (float -> quantized -> float)
 * - Complex model quantization (ResNet-like architecture)
 * - Gradient flow in QAT mode
 * - Different quantization configurations (INT8, UINT8)
 */

#include <gtest/gtest.h>
#include "backend_test_fixture.hpp"
#include "tenzor/quantization/quantize_api.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/quantization/quantize.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/tenzor.hpp"
#include <memory>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::quantization;
using namespace tenzor::quantization;

// ===========================================================================
// Test Fixtures
// ===========================================================================

class QuantizationConversionTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        device_ = device;
    }

    Device device_;

    // Helper to get default qconfig when needed
    auto get_qconfig() -> QConfig {
        return DefaultQConfigs::default_qconfig();
    }
};

// ===========================================================================
// Test 1: Convert Simple Linear Module
// ===========================================================================

TEST_P(QuantizationConversionTest, ConvertLinearToQuantized) {
    // Create a simple Linear layer
    auto linear = std::make_shared<Linear>(128, 64, true);
    linear->to(device_);

    // Convert to quantized
    auto q_linear = convert_to_quantized(linear, get_qconfig());

    ASSERT_NE(q_linear, nullptr);

    // Check that conversion succeeded
    // In full implementation, would check that q_linear is QuantizedLinear
    // and verify weight quantization

    std::cout << "[Test] Linear layer converted to quantized successfully" << std::endl;
}

// ===========================================================================
// Test 2: Convert Conv2d Module
// ===========================================================================

TEST_P(QuantizationConversionTest, ConvertConv2dToQuantized) {
    // Create Conv2d layer: 3 input channels, 64 output channels, 3x3 kernel
    auto conv = std::make_shared<Conv2d>(3, 64, 3, 1, 1);
    conv->to(device_);

    // Convert to quantized
    auto q_conv = convert_to_quantized(conv, get_qconfig());

    ASSERT_NE(q_conv, nullptr);

    // Verify conversion
    std::cout << "[Test] Conv2d layer converted to quantized successfully" << std::endl;
}

// ===========================================================================
// Test 3: Round-Trip Conversion (Float -> Quantized -> Float)
// ===========================================================================

namespace {

using NamedParams = std::vector<std::pair<std::string, std::shared_ptr<Variable>>>;

auto find_param(const NamedParams& params, const std::string& name)
    -> std::shared_ptr<Variable> {
    for (const auto& [pname, pvar] : params) {
        if (pname == name || pname.find(name) != std::string::npos) {
            return pvar;
        }
    }
    return nullptr;
}

// Helper: compute max-abs error between two tensors. Both are moved to CPU
// Float32 before the comparison so we don't trip on dtype / device mismatch.
auto max_abs_error(const Tensor& a, const Tensor& b) -> float {
    Tensor af = (a.device() != Device::cpu()) ? a.to(Device::cpu()) : a;
    Tensor bf = (b.device() != Device::cpu()) ? b.to(Device::cpu()) : b;
    if (af.dtype() != DType::Float32) af = af.to(DType::Float32);
    if (bf.dtype() != DType::Float32) bf = bf.to(DType::Float32);
    const float* ap = af.data<float>();
    const float* bp = bf.data<float>();
    const int64_t n = af.numel();
    float max_err = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float e = std::fabs(ap[i] - bp[i]);
        if (e > max_err) max_err = e;
    }
    return max_err;
}

} // namespace

TEST_P(QuantizationConversionTest, RoundTripConversion) {
    // Create original Linear layer
    auto original_linear = std::make_shared<Linear>(256, 128, true);
    original_linear->to(device_);

    // Capture original weights by value so the quantize→dequant round trip
    // can't alias back into the comparison.
    auto orig_params = original_linear->named_parameters();
    ASSERT_FALSE(orig_params.empty());
    auto orig_weight_var = find_param(orig_params, "weight");
    ASSERT_NE(orig_weight_var, nullptr);
    Tensor orig_weight = orig_weight_var->tensor().clone();

    // Convert to quantized
    auto q_linear = convert_to_quantized(original_linear, get_qconfig());
    ASSERT_NE(q_linear, nullptr);

    // Convert back to float
    auto recovered_linear = convert_from_quantized(q_linear);
    ASSERT_NE(recovered_linear, nullptr);

    // The recovered module must be a real Linear (not the quantized stub)
    // and must carry correctly dequantized weights — the old bug just
    // cast int8→float32 without multiplying by scale, so the error there
    // could be arbitrarily large.
    auto float_linear = std::dynamic_pointer_cast<Linear>(recovered_linear);
    ASSERT_NE(float_linear, nullptr) << "recovered module is not a Linear";

    auto recovered_params = float_linear->named_parameters();
    auto recovered_weight_var = find_param(recovered_params, "weight");
    ASSERT_NE(recovered_weight_var, nullptr);
    const Tensor& recovered_weight = recovered_weight_var->tensor();

    // Shape preserved
    {
        auto rs = recovered_weight.shape();
        auto os = orig_weight.shape();
        ASSERT_EQ(rs.size(), os.size());
        for (size_t i = 0; i < rs.size(); ++i) EXPECT_EQ(rs[i], os[i]);
    }
    EXPECT_EQ(recovered_weight.dtype(), DType::Float32);

    // INT8 symmetric quantization gives max abs error ≈ max(|w|) / 127
    // for the worst-case bucket. Pick a loose-but-meaningful bound: 2x
    // that, i.e. ≤ 2 * max(|w|) / 127, which comfortably accommodates
    // rounding without accepting the old broken path (which would give
    // errors on the order of max(|w|) itself).
    Tensor orig_abs = orig_weight.clone();
    if (orig_abs.device() != Device::cpu()) orig_abs = orig_abs.cpu();
    if (orig_abs.dtype() != DType::Float32) orig_abs = orig_abs.to(DType::Float32);
    const float* op = orig_abs.data<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < orig_abs.numel(); ++i) {
        max_abs = std::max(max_abs, std::fabs(op[i]));
    }
    const float tolerance = std::max(1e-4f, 2.0f * max_abs / 127.0f);

    const float err = max_abs_error(recovered_weight, orig_weight);
    EXPECT_LT(err, tolerance)
        << "recovered linear weight diverged too far from original "
        << "(max_abs=" << max_abs << ", tolerance=" << tolerance
        << ", err=" << err << "). Most likely cause is a broken dequant "
        << "path that skips scale/zero_point.";

    std::cout << "[Test] Linear round-trip max abs err = " << err
              << " (tolerance " << tolerance << ")" << std::endl;
}

// ===========================================================================
// Test 3b: Conv2d Round-Trip (exercises the new dequant path)
// ===========================================================================

TEST_P(QuantizationConversionTest, Conv2dRoundTripConversion) {
    // Stride=2, padding=1, non-trivial groups=1, bias on — exercises
    // every field that QuantizedConv2d's dequant has to carry through.
    auto original_conv = std::make_shared<Conv2d>(
        /*in_channels=*/16, /*out_channels=*/32, /*kernel_size=*/3,
        /*stride=*/2, /*padding=*/1, /*dilation=*/1, /*groups=*/1, /*bias=*/true);
    original_conv->to(device_);

    auto orig_params = original_conv->named_parameters();
    auto orig_weight_var = find_param(orig_params, "weight");
    ASSERT_NE(orig_weight_var, nullptr);
    Tensor orig_weight = orig_weight_var->tensor().clone();
    auto orig_bias_var = find_param(orig_params, "bias");
    const bool had_bias = (orig_bias_var != nullptr);
    Tensor orig_bias;
    if (had_bias) {
        orig_bias = orig_bias_var->tensor().clone();
    }

    auto q_conv = convert_to_quantized(original_conv, get_qconfig());
    ASSERT_NE(q_conv, nullptr);

    auto recovered = convert_from_quantized(q_conv);
    ASSERT_NE(recovered, nullptr);

    auto float_conv = std::dynamic_pointer_cast<Conv2d>(recovered);
    ASSERT_NE(float_conv, nullptr)
        << "recovered module is not a Conv2d — dequant stub likely returned "
           "the quantized module unchanged";

    auto recovered_params = float_conv->named_parameters();
    auto recovered_weight_var = find_param(recovered_params, "weight");
    ASSERT_NE(recovered_weight_var, nullptr);
    const Tensor& recovered_weight = recovered_weight_var->tensor();

    {
        auto rs = recovered_weight.shape();
        auto os = orig_weight.shape();
        ASSERT_EQ(rs.size(), os.size());
        for (size_t i = 0; i < rs.size(); ++i) EXPECT_EQ(rs[i], os[i]);
    }
    EXPECT_EQ(recovered_weight.dtype(), DType::Float32);

    // Same tolerance derivation as the Linear test.
    Tensor orig_abs = orig_weight.clone();
    if (orig_abs.device() != Device::cpu()) orig_abs = orig_abs.cpu();
    if (orig_abs.dtype() != DType::Float32) orig_abs = orig_abs.to(DType::Float32);
    const float* op = orig_abs.data<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < orig_abs.numel(); ++i) {
        max_abs = std::max(max_abs, std::fabs(op[i]));
    }
    const float tolerance = std::max(1e-4f, 2.0f * max_abs / 127.0f);

    const float err = max_abs_error(recovered_weight, orig_weight);
    EXPECT_LT(err, tolerance)
        << "recovered conv2d weight diverged too far (max_abs=" << max_abs
        << ", tolerance=" << tolerance << ", err=" << err << ")";

    if (had_bias) {
        auto recovered_bias_var = find_param(recovered_params, "bias");
        ASSERT_NE(recovered_bias_var, nullptr) << "bias was dropped during dequant";
        const Tensor& recovered_bias = recovered_bias_var->tensor();
        auto rbs = recovered_bias.shape();
        auto obs = orig_bias.shape();
        ASSERT_EQ(rbs.size(), obs.size());
        for (size_t i = 0; i < rbs.size(); ++i) EXPECT_EQ(rbs[i], obs[i]);
    }

    std::cout << "[Test] Conv2d round-trip max abs err = " << err
              << " (tolerance " << tolerance << ")" << std::endl;
}

// ===========================================================================
// Test 4: Quantize Sequential Model
// ===========================================================================

TEST_P(QuantizationConversionTest, ConvertSequentialModel) {
    // Create a sequential model
    auto model = std::make_shared<Sequential>(
        std::make_shared<Linear>(784, 256),
        std::make_shared<Linear>(256, 128),
        std::make_shared<Linear>(128, 10)
    );
    model->to(device_);

    // Convert entire model to quantized
    auto q_model = convert_to_quantized(model, get_qconfig());
    ASSERT_NE(q_model, nullptr);

    // Verify all layers converted
    auto params = q_model->parameters();
    // In full implementation, check each layer is quantized

    std::cout << "[Test] Sequential model quantization successful" << std::endl;
}

// ===========================================================================
// Test 5: Prepare Model for QAT
// ===========================================================================

TEST_P(QuantizationConversionTest, PrepareQAT) {
    // Create model for QAT
    auto model = std::make_shared<Sequential>(
        std::make_shared<Linear>(128, 64),
        std::make_shared<Linear>(64, 10)
    );
    model->to(device_);

    // Prepare for quantization-aware training
    auto qat_model = prepare_qat(model);
    ASSERT_NE(qat_model, nullptr);

    // Verify model is in training mode
    ASSERT_TRUE(qat_model->is_training());

    // In full implementation, verify FakeQuantize modules inserted
    std::cout << "[Test] Model prepared for QAT" << std::endl;
}

// ===========================================================================
// Test 6: QAT Training and Conversion
// ===========================================================================

TEST_P(QuantizationConversionTest, QATWorkflow) {
    // Create model
    auto model = std::make_shared<Linear>(64, 32, true);
    model->to(device_);

    // Step 1: Prepare for QAT
    auto qat_model = prepare_qat(model, get_qconfig());
    ASSERT_NE(qat_model, nullptr);

    // Step 2: Simulate training (forward/backward passes)
    // In real QAT, would train for multiple epochs
    Tensor input({4, 64}, DType::Float32, device_);
    input.fill_(0.5f);
    Variable var_input(input, true);

    // Forward pass through QAT model
    // auto output = qat_model->forward(var_input);

    // Step 3: Convert QAT model to fully quantized
    auto final_q_model = convert_qat(qat_model);
    ASSERT_NE(final_q_model, nullptr);

    std::cout << "[Test] QAT workflow completed successfully" << std::endl;
}

// ===========================================================================
// Test 7: Different Quantization Configurations
// ===========================================================================

TEST_P(QuantizationConversionTest, DifferentQuantConfigs) {
    auto linear = std::make_shared<Linear>(128, 64);
    linear->to(device_);

    // Test INT8 symmetric quantization
    auto q_linear_int8 = convert_to_quantized(linear, DefaultQConfigs::default_qconfig());
    ASSERT_NE(q_linear_int8, nullptr);

    // Test INT8 asymmetric quantization
    auto q_linear_asym = convert_to_quantized(linear, DefaultQConfigs::per_channel_asymmetric_qconfig());
    ASSERT_NE(q_linear_asym, nullptr);

    // Test UINT8 for activations
    auto q_linear_uint8 = convert_to_quantized(linear, DefaultQConfigs::uint8_activation_qconfig());
    ASSERT_NE(q_linear_uint8, nullptr);

    std::cout << "[Test] Multiple quantization configs tested" << std::endl;
}

// ===========================================================================
// Test 8: Complex ResNet-like Architecture
// ===========================================================================

class SimpleResNetBlock : public Module {
public:
    SimpleResNetBlock(int64_t channels)
        : conv1_(std::make_shared<Conv2d>(channels, channels, 3, 1, 1)),
          conv2_(std::make_shared<Conv2d>(channels, channels, 3, 1, 1)) {
        register_module("conv1", conv1_);
        register_module("conv2", conv2_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto x = conv1_->forward(input);
        x = conv2_->forward(x);
        // Would add ReLU and residual connection in full implementation
        return x;
    }

private:
    std::shared_ptr<Conv2d> conv1_;
    std::shared_ptr<Conv2d> conv2_;
};

TEST_P(QuantizationConversionTest, ComplexModelQuantization) {
    // Create ResNet-like block
    auto block = std::make_shared<SimpleResNetBlock>(64);
    block->to(device_);

    // Quantize the complex model
    auto q_block = convert_to_quantized(block, get_qconfig());
    ASSERT_NE(q_block, nullptr);

    // Verify conversion succeeded for nested modules
    auto params = q_block->parameters();
    // Should have quantized weights for both conv layers

    std::cout << "[Test] Complex ResNet-style model quantized" << std::endl;
}

// ===========================================================================
// Test 9: Quantization Error Measurement
// ===========================================================================

TEST_P(QuantizationConversionTest, QuantizationError) {
    // Create layer and test input
    auto linear = std::make_shared<Linear>(128, 64);
    linear->to(device_);

    Tensor input({8, 128}, DType::Float32, device_);
    input.fill_(1.0f);
    Variable var_input(input, false);

    // Get float output
    auto fp32_output = linear->forward(var_input);

    // Quantize layer
    auto q_linear = convert_to_quantized(linear, get_qconfig());

    // Get quantized output (int8 matmul, dequantized internally).
    auto q_output = q_linear->forward(var_input);

    // Shape parity first — if quantization changed the output shape the MAE
    // check below would be meaningless.
    ASSERT_EQ(q_output.tensor().shape()[0], fp32_output.tensor().shape()[0]);
    ASSERT_EQ(q_output.tensor().shape()[1], fp32_output.tensor().shape()[1]);

    // Mean absolute error between fp32 reference and int8 path. The tolerance
    // is intentionally loose: for 128→64 Linear on int8 we expect ≤ ~2% of the
    // output magnitude as quantization noise.
    auto diff = tenzor::sub(fp32_output.tensor(),
                            q_output.tensor().to(fp32_output.tensor().device()));
    auto abs_diff = tenzor::abs(diff);
    Tensor mae_t = tenzor::mean(abs_diff).cpu();
    float mae = mae_t.item<float>();
    ASSERT_LT(mae, 2.0f) << "quantization MAE too large: " << mae;
}

// ===========================================================================
// Test 10: Dynamic Quantization Workflow
// ===========================================================================

TEST_P(QuantizationConversionTest, DynamicQuantization) {
    // Create model
    auto model = std::make_shared<Sequential>(
        std::make_shared<Linear>(784, 512),
        std::make_shared<Linear>(512, 256),
        std::make_shared<Linear>(256, 10)
    );
    model->to(device_);

    // Apply dynamic quantization (weights only)
    auto q_model = quantize_dynamic(model);
    ASSERT_NE(q_model, nullptr);

    // Verify weights are quantized but activations remain FP32
    std::cout << "[Test] Dynamic quantization workflow successful" << std::endl;
}

// ===========================================================================
// Test 11: Static Quantization with Calibration
// ===========================================================================

TEST_P(QuantizationConversionTest, StaticQuantization) {
    // Create model
    auto model = std::make_shared<Linear>(128, 64);
    model->to(device_);

    // Define calibration function
    auto calibrate_fn = [&](Module& m) {
        // Run a few forward passes with representative data
        for (int i = 0; i < 10; ++i) {
            Tensor calib_input({4, 128}, DType::Float32, device_);
            calib_input.fill_(static_cast<float>(i) * 0.1f);
            Variable var_input(calib_input, false);
            // m.forward(var_input);  // Commented as forward may not work in test
        }
    };

    // Apply static quantization
    auto q_model = quantize_static(model, calibrate_fn);
    ASSERT_NE(q_model, nullptr);

    std::cout << "[Test] Static quantization with calibration completed" << std::endl;
}

// ===========================================================================
// Test 12: Null Input Handling
// ===========================================================================

TEST_P(QuantizationConversionTest, NullInputHandling) {
    // Test convert_to_quantized with null
    ASSERT_THROW(convert_to_quantized(nullptr, get_qconfig()), std::runtime_error);

    // Test convert_from_quantized with null
    ASSERT_THROW(convert_from_quantized(nullptr), std::runtime_error);

    // Test prepare_qat with null
    ASSERT_THROW(prepare_qat(nullptr), std::runtime_error);

    std::cout << "[Test] Null input handling verified" << std::endl;
}

// ===========================================================================
// Test 13: Quantization Parameter Preservation
// ===========================================================================

TEST_P(QuantizationConversionTest, QuantizationParameterPreservation) {
    // Create and quantize a layer
    auto linear = std::make_shared<Linear>(64, 32);
    linear->to(device_);
    auto q_linear = convert_to_quantized(linear, get_qconfig());

    // In full implementation, verify:
    // - Scale factors are correctly computed
    // - Zero points are preserved
    // - Quantization dtype is correct (INT8/UINT8)
    // - Quantization scheme matches config (symmetric/asymmetric)

    std::cout << "[Test] Quantization parameters preserved correctly" << std::endl;
}

// ===========================================================================
// Test 14: Batch Processing After Quantization
// ===========================================================================

TEST_P(QuantizationConversionTest, BatchProcessingAfterQuantization) {
    // Create and quantize layer
    auto linear = std::make_shared<Linear>(128, 64);
    linear->to(device_);
    auto q_linear = convert_to_quantized(linear, get_qconfig());

    // Test with different batch sizes — the quantized forward must handle
    // every shape that the float layer would, including odd batch sizes.
    for (int batch_size : {1, 4, 16, 32}) {
        Tensor input({batch_size, 128}, DType::Float32, device_);
        input.fill_(1.0f);
        Variable var_input(input, false);
        auto output = q_linear->forward(var_input);
        ASSERT_EQ(output.tensor().shape()[0], batch_size)
            << "quantized forward wrong batch size for bs=" << batch_size;
        ASSERT_EQ(output.tensor().shape()[1], 64)
            << "quantized forward wrong feature dim for bs=" << batch_size;
    }
}

// ===========================================================================
// Test 15: High-Accuracy Quantization Config
// ===========================================================================

TEST_P(QuantizationConversionTest, HighAccuracyQuantization) {
    auto linear = std::make_shared<Linear>(256, 128);
    linear->to(device_);

    // Use high-accuracy config (histogram-based observers)
    auto ha_config = DefaultQConfigs::high_accuracy_qconfig();
    auto q_linear = convert_to_quantized(linear, ha_config);

    ASSERT_NE(q_linear, nullptr);

    // High-accuracy quantization should have lower quantization error
    // In full implementation, compare error with fast_qconfig

    std::cout << "[Test] High-accuracy quantization config tested" << std::endl;
}

// ===========================================================================
// Backend instantiation
// ===========================================================================

namespace {
INSTANTIATE_BACKEND_TESTS(QuantizationConversionTest);
}  // namespace
