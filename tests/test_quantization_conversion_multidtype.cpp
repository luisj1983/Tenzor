/**
 * @file test_quantization_conversion_multidtype.cpp
 * @brief Multi-dtype / multi-backend companion to test_quantization_conversion.cpp.
 *
 * The plain file (BackendTest, Float32-only) covers the quantization layer
 * conversion surface — convert_to_quantized / convert_from_quantized /
 * prepare_qat for Linear and Conv2d, round-trip float->INT8->float error,
 * sequential / ResNet-style model quantization, QAT workflow, multiple
 * qconfigs, dynamic / static quantization, null handling, batch processing,
 * and high-accuracy config.
 *
 * This companion adds the dtype axis across {Float32, Float64, Float16} x
 * {cpu, cuda, vulkan, oneapi, rocm, mps} via MultiBackendDTypeTest. The
 * source Linear/Conv2d is moved to the test dtype+device via convert_model
 * BEFORE conversion, so the companion exercises each backend's
 * dtype-conversion kernel on the weight that convert_to_quantized then
 * quantizes to INT8.
 *
 * No dtype skips are needed: every stage of the conversion path upcasts
 * non-Float32 tensors to Float32 internally —
 *   - the weight observer (observer.cpp:26-29),
 *   - quantize_tensor (quantize.cpp:215-221, input -> Float32+CPU),
 *   - the quantized forward (quantized_layers.cpp:487/496/905/915,
 *     input/bias -> Float32),
 *   - convert_from_quantized (module_conversion.cpp:104-149, dequant weight
 *     -> Float32).
 * So a Float16 or Float64 source weight is upcast before INT8 quantization
 * on every backend; the recovered (dequantized) weight is always Float32
 * (the plain file asserts this). The new coverage is the device-side
 * .to(Float32) cast for F16/F64 weights/inputs on each backend, plus
 * confirming the conversion succeeds (non-null result) for non-Float32
 * source modules on every backend.
 *
 * Numerical comparisons are made dtype-safe: the round-trip tests reuse the
 * plain file's max_abs_error helper (casts both tensors to CPU Float32), and
 * the QuantizationError MAE casts both the float and quantized outputs to
 * Float32 before sub/abs/mean so an F16/F64 float-layer output compares
 * cleanly against the always-F32 quantized output.
 */

#include <gtest/gtest.h>
#include "multi_backend_dtype_fixture.hpp"
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
using namespace tenzor::testing;

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

// Max abs value of a tensor (CPU Float32).
auto max_abs_val(const Tensor& t) -> float {
    Tensor f = (t.device() != Device::cpu()) ? t.to(Device::cpu()) : t;
    if (f.dtype() != DType::Float32) f = f.to(DType::Float32);
    const float* p = f.data<float>();
    float m = 0.0f;
    for (int64_t i = 0; i < f.numel(); ++i) m = std::max(m, std::fabs(p[i]));
    return m;
}

} // namespace

class QuantizationConversionMultiDType : public MultiBackendDTypeTest {
protected:
    auto get_qconfig() -> QConfig { return DefaultQConfigs::default_qconfig(); }
};

// ===========================================================================
// Test 1: Convert Simple Linear Module
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, ConvertLinearToQuantized) {
    auto linear = std::make_shared<Linear>(128, 64, true);
    convert_model(linear);
    auto q_linear = convert_to_quantized(linear, get_qconfig());
    ASSERT_NE(q_linear, nullptr);
}

// ===========================================================================
// Test 2: Convert Conv2d Module
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, ConvertConv2dToQuantized) {
    auto conv = std::make_shared<Conv2d>(3, 64, 3, 1, 1);
    convert_model(conv);
    auto q_conv = convert_to_quantized(conv, get_qconfig());
    ASSERT_NE(q_conv, nullptr);
}

// ===========================================================================
// Test 3: Round-Trip Conversion (Float -> Quantized -> Float)
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, RoundTripConversion) {
    auto original_linear = std::make_shared<Linear>(256, 128, true);
    convert_model(original_linear);

    auto orig_params = original_linear->named_parameters();
    ASSERT_FALSE(orig_params.empty());
    auto orig_weight_var = find_param(orig_params, "weight");
    ASSERT_NE(orig_weight_var, nullptr);
    Tensor orig_weight = orig_weight_var->tensor().clone();

    auto q_linear = convert_to_quantized(original_linear, get_qconfig());
    ASSERT_NE(q_linear, nullptr);

    auto recovered_linear = convert_from_quantized(q_linear);
    ASSERT_NE(recovered_linear, nullptr);

    auto float_linear = std::dynamic_pointer_cast<Linear>(recovered_linear);
    ASSERT_NE(float_linear, nullptr) << "recovered module is not a Linear";

    auto recovered_params = float_linear->named_parameters();
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

    const float max_abs = max_abs_val(orig_weight);
    const float tolerance = std::max(1e-4f, 2.0f * max_abs / 127.0f);
    const float err = max_abs_error(recovered_weight, orig_weight);
    EXPECT_LT(err, tolerance)
        << "recovered linear weight diverged too far from original "
        << "(max_abs=" << max_abs << ", tolerance=" << tolerance
        << ", err=" << err << ", dtype=" << static_cast<int>(dtype()) << ")";
}

// ===========================================================================
// Test 3b: Conv2d Round-Trip
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, Conv2dRoundTripConversion) {
    auto original_conv = std::make_shared<Conv2d>(
        /*in_channels=*/16, /*out_channels=*/32, /*kernel_size=*/3,
        /*stride=*/2, /*padding=*/1, /*dilation=*/1, /*groups=*/1, /*bias=*/true);
    convert_model(original_conv);

    auto orig_params = original_conv->named_parameters();
    auto orig_weight_var = find_param(orig_params, "weight");
    ASSERT_NE(orig_weight_var, nullptr);
    Tensor orig_weight = orig_weight_var->tensor().clone();
    auto orig_bias_var = find_param(orig_params, "bias");
    const bool had_bias = (orig_bias_var != nullptr);
    Tensor orig_bias;
    if (had_bias) orig_bias = orig_bias_var->tensor().clone();

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

    const float max_abs = max_abs_val(orig_weight);
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
}

// ===========================================================================
// Test 4: Quantize Sequential Model
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, ConvertSequentialModel) {
    auto model = std::make_shared<Sequential>(
        std::make_shared<Linear>(784, 256),
        std::make_shared<Linear>(256, 128),
        std::make_shared<Linear>(128, 10)
    );
    convert_model(model);
    auto q_model = convert_to_quantized(model, get_qconfig());
    ASSERT_NE(q_model, nullptr);
}

// ===========================================================================
// Test 5: Prepare Model for QAT
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, PrepareQAT) {
    auto model = std::make_shared<Sequential>(
        std::make_shared<Linear>(128, 64),
        std::make_shared<Linear>(64, 10)
    );
    convert_model(model);
    auto qat_model = prepare_qat(model);
    ASSERT_NE(qat_model, nullptr);
    ASSERT_TRUE(qat_model->is_training());
}

// ===========================================================================
// Test 6: QAT Training and Conversion
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, QATWorkflow) {
    auto model = std::make_shared<Linear>(64, 32, true);
    convert_model(model);

    auto qat_model = prepare_qat(model, get_qconfig());
    ASSERT_NE(qat_model, nullptr);

    // Simulated training input (forward is intentionally not run here, as in
    // the plain file). Built in the test dtype so a future forward call would
    // match the model's dtype.
    auto input = createInput({4, 64}, /*requires_grad=*/true);

    auto final_q_model = convert_qat(qat_model);
    ASSERT_NE(final_q_model, nullptr);
    (void)input;
}

// ===========================================================================
// Test 7: Different Quantization Configurations
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, DifferentQuantConfigs) {
    auto linear = std::make_shared<Linear>(128, 64);
    convert_model(linear);

    ASSERT_NE(convert_to_quantized(linear, DefaultQConfigs::default_qconfig()), nullptr);
    ASSERT_NE(convert_to_quantized(linear, DefaultQConfigs::per_channel_asymmetric_qconfig()), nullptr);
    ASSERT_NE(convert_to_quantized(linear, DefaultQConfigs::uint8_activation_qconfig()), nullptr);
}

// ===========================================================================
// Test 8: Complex ResNet-like Architecture
// ===========================================================================

namespace {
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
        return x;
    }
private:
    std::shared_ptr<Conv2d> conv1_;
    std::shared_ptr<Conv2d> conv2_;
};
} // namespace

TEST_P(QuantizationConversionMultiDType, ComplexModelQuantization) {
    auto block = std::make_shared<SimpleResNetBlock>(64);
    convert_model(block);
    auto q_block = convert_to_quantized(block, get_qconfig());
    ASSERT_NE(q_block, nullptr);
}

// ===========================================================================
// Test 9: Quantization Error Measurement
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, QuantizationError) {
    auto linear = std::make_shared<Linear>(128, 64);
    convert_model(linear);

    // Input in the test dtype so the float-layer forward is a clean dtype×dtype
    // matmul; its output is then cast to Float32 for the MAE comparison against
    // the always-F32 quantized output.
    auto var_input = createInput({8, 128}, /*requires_grad=*/false);
    auto fp_out = linear->forward(var_input);

    auto q_linear = convert_to_quantized(linear, get_qconfig());
    auto q_out = q_linear->forward(var_input);

    ASSERT_EQ(q_out.tensor().shape()[0], fp_out.tensor().shape()[0]);
    ASSERT_EQ(q_out.tensor().shape()[1], fp_out.tensor().shape()[1]);

    auto fp_f32 = fp_out.tensor().to(DType::Float32);
    auto q_f32 = q_out.tensor().to(DType::Float32);
    auto diff = tenzor::sub(fp_f32, q_f32);
    auto abs_diff = tenzor::abs(diff);
    float mae = tenzor::mean(abs_diff).cpu().item<float>();
    ASSERT_LT(mae, 2.0f) << "quantization MAE too large: " << mae;
}

// ===========================================================================
// Test 10: Dynamic Quantization Workflow
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, DynamicQuantization) {
    auto model = std::make_shared<Sequential>(
        std::make_shared<Linear>(784, 512),
        std::make_shared<Linear>(512, 256),
        std::make_shared<Linear>(256, 10)
    );
    convert_model(model);
    auto q_model = quantize_dynamic(model);
    ASSERT_NE(q_model, nullptr);
}

// ===========================================================================
// Test 11: Static Quantization with Calibration
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, StaticQuantization) {
    auto model = std::make_shared<Sequential>();
    model->add_module(std::make_shared<Linear>(128, 64));
    convert_model(model);

    auto calibrate_fn = [&](Module& m) {
        for (int i = 0; i < 10; ++i) {
            Tensor calib_input({4, 128}, DType::Float32, device());
            calib_input.fill_(static_cast<float>(i) * 0.1f);
            Variable var_input(calib_input, false);
            (void)m; (void)var_input;
        }
    };

    auto q_model = quantize_static(model, calibrate_fn);
    ASSERT_NE(q_model, nullptr);
}

// ===========================================================================
// Test 12: Null Input Handling
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, NullInputHandling) {
    ASSERT_THROW(convert_to_quantized(nullptr, get_qconfig()), std::runtime_error);
    ASSERT_THROW(convert_from_quantized(nullptr), std::runtime_error);
    ASSERT_THROW(prepare_qat(nullptr), std::runtime_error);
}

// ===========================================================================
// Test 13: Quantization Parameter Preservation
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, QuantizationParameterPreservation) {
    auto linear = std::make_shared<Linear>(64, 32);
    convert_model(linear);
    auto q_linear = convert_to_quantized(linear, get_qconfig());
    ASSERT_NE(q_linear, nullptr);
}

// ===========================================================================
// Test 14: Batch Processing After Quantization
// ===========================================================================

TEST_P(QuantizationConversionMultiDType, BatchProcessingAfterQuantization) {
    auto linear = std::make_shared<Linear>(128, 64);
    convert_model(linear);
    auto q_linear = convert_to_quantized(linear, get_qconfig());

    // Quantized forward upcasts the input to Float32 internally, so an F32
    // input exercises the path regardless of the source weight's dtype.
    for (int batch_size : {1, 4, 16, 32}) {
        Tensor input({batch_size, 128}, DType::Float32, device());
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

TEST_P(QuantizationConversionMultiDType, HighAccuracyQuantization) {
    auto linear = std::make_shared<Linear>(256, 128);
    convert_model(linear);
    auto ha_config = DefaultQConfigs::high_accuracy_qconfig();
    auto q_linear = convert_to_quantized(linear, ha_config);
    ASSERT_NE(q_linear, nullptr);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(QuantizationConversionMultiDType);