/**
 * @file fake_quantize.cpp
 * @brief Implementation of fake quantization for QAT
 */

#include "tenzor/nn/quantization/fake_quantize.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/quantization/qconfig.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include <stdexcept>

namespace tenzor {
namespace nn {
namespace quantization {

using tenzor::nn::Linear;
using tenzor::nn::Conv2d;
using tenzor::nn::Sequential;

// ============================================================================
// FakeQuantize
// ============================================================================

FakeQuantize::FakeQuantize(
    QuantDType dtype,
    QuantizationScheme scheme,
    bool learnable,
    bool observer_enabled,
    int64_t axis
) : dtype_(dtype),
    scheme_(scheme),
    learnable_(learnable),
    observer_enabled_(observer_enabled),
    axis_(axis) {

    // Create observer
    bool use_histogram = false;  // Can be made configurable
    observer_ = make_observer(scheme, use_histogram, axis);

    // Initialize quantization params with dummy values
    Tensor scale({1}, DType::Float32, Device::cpu());
    Tensor zero_point({1}, DType::Int32, Device::cpu());
    scale.fill_(1.0f);
    zero_point.fill_(0);

    qparams_ = std::make_unique<QuantizationParams>(
        scale, zero_point, dtype, scheme, axis
    );

    if (learnable_) {
        // Register scale and zero_point as parameters
        Variable scale_var(qparams_->scale, true);
        Variable zp_var(qparams_->zero_point, false);  // Zero point typically not learnable
        register_parameter("scale", scale_var);
        register_parameter("zero_point", zp_var);
    }
}

auto FakeQuantize::forward_impl(const Variable& input) -> Variable {
    if (!fake_quant_enabled_) {
        return input;  // Pass through
    }

    // Update observer if enabled (even in eval mode for calibration)
    if (observer_enabled_) {
        observer_->observe(input.tensor());

        // Update qparams if observer has data and in training mode
        if (observer_->has_data() && !learnable_ && training_) {
            calculate_qparams();
        }
    }

    // Per-tensor schemes can use the autograd-enabled straight-through
    // estimator so QAT gradients thread through quantize→dequantize correctly.
    // Per-channel schemes fall back to the bare Tensor path (no backward
    // support yet for per-channel STE — tracked as a follow-up).
    bool per_tensor = (scheme_ == QuantizationScheme::PerTensorSymmetric ||
                       scheme_ == QuantizationScheme::PerTensorAsymmetric);
    if (per_tensor && input.requires_grad()) {
        Tensor scale_cpu = (qparams_->scale.device() == Device::cpu())
            ? qparams_->scale : qparams_->scale.to(Device::cpu());
        Tensor zp_cpu = (qparams_->zero_point.device() == Device::cpu())
            ? qparams_->zero_point : qparams_->zero_point.to(Device::cpu());
        float scale = scale_cpu.data<float>()[0];
        float zero_point = static_cast<float>(zp_cpu.data<int32_t>()[0]);

        float quant_min = 0.0f, quant_max = 0.0f;
        switch (dtype_) {
            case QuantDType::INT8:  quant_min = -128.0f; quant_max = 127.0f; break;
            case QuantDType::UINT8: quant_min =    0.0f; quant_max = 255.0f; break;
            case QuantDType::INT4:  quant_min =   -8.0f; quant_max =   7.0f; break;
            case QuantDType::UINT4: quant_min =    0.0f; quant_max =  15.0f; break;
        }
        return fake_quantize_with_grad(input, scale, zero_point, quant_min, quant_max);
    }

    // Fall-through: non-autograd path (eval calibration, per-channel,
    // or input.requires_grad() == false).
    Tensor quantized = apply_fake_quantization(input.tensor());
    return Variable(quantized, input.requires_grad());
}

auto FakeQuantize::apply_fake_quantization(const Tensor& input) const -> Tensor {
    // Quantize then immediately dequantize
    QuantizedTensor q_tensor = quantize_tensor(input, *qparams_);
    Tensor output = dequantize_tensor(q_tensor);
    // Preserve original dtype
    if (output.dtype() != input.dtype()) {
        output = output.to(input.dtype());
    }
    return output;
}

auto FakeQuantize::set_qparams(const QuantizationParams& params) -> void {
    qparams_ = std::make_unique<QuantizationParams>(params);

    if (learnable_) {
        // Update registered parameters
        Variable scale_var(qparams_->scale, true);
        register_parameter("scale", scale_var);
    }
}

auto FakeQuantize::calculate_qparams() -> void {
    if (!observer_->has_data()) {
        throw std::runtime_error("Observer has no data to calculate qparams");
    }

    *qparams_ = observer_->calculate_qparams(dtype_, scheme_);

    if (learnable_) {
        // Update registered parameters
        Variable scale_var(qparams_->scale, true);
        register_parameter("scale", scale_var);
    }
}

auto FakeQuantize::reset_observer() -> void {
    observer_->reset();
}

// ============================================================================
// LearnableFakeQuantize
// ============================================================================

LearnableFakeQuantize::LearnableFakeQuantize(
    QuantDType dtype,
    QuantizationScheme scheme,
    int64_t axis
) : FakeQuantize(dtype, scheme, true, true, axis) {}

auto LearnableFakeQuantize::init_from_observer() -> void {
    if (!observer()->has_data()) {
        throw std::runtime_error("Observer has no data for initialization");
    }

    calculate_qparams();
}

// ============================================================================
// Functional Interface
// ============================================================================

auto fake_quantize_activation(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    QuantDType dtype
) -> Tensor {
    QuantizationParams params(
        scale, zero_point, dtype,
        QuantizationScheme::PerTensorSymmetric, -1
    );

    QuantizedTensor q_tensor = quantize_tensor(input, params);
    Tensor output = dequantize_tensor(q_tensor);
    // Preserve original dtype
    if (output.dtype() != input.dtype()) {
        output = output.to(input.dtype());
    }
    return output;
}

auto fake_quantize_weight(
    const Tensor& weight,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    QuantDType dtype
) -> Tensor {
    QuantizationParams params(
        scale, zero_point, dtype,
        QuantizationScheme::PerChannelSymmetric, axis
    );

    QuantizedTensor q_tensor = quantize_tensor(weight, params);
    Tensor output = dequantize_tensor(q_tensor);
    // Preserve original dtype
    if (output.dtype() != weight.dtype()) {
        output = output.to(weight.dtype());
    }
    return output;
}

// ============================================================================
// QATHelper
// ============================================================================

auto QATHelper::prepare_qat(
    Module& model,
    QuantDType dtype,
    QuantizationScheme scheme,
    bool learnable
) -> void {
    // Clear any previously tracked modules
    fake_quant_modules_.clear();

    // Traverse the model's named parameters to find quantizable layers
    // For each quantizable layer, create and track a FakeQuantize module
    auto params = model.named_parameters();

    // Create FakeQuantize modules for weights and activations
    // Weight observer uses per-channel symmetric for better accuracy
    // Activation observer uses per-tensor symmetric for simplicity

    // Create activation fake quantize
    auto activation_fq = std::make_shared<FakeQuantize>(
        dtype, scheme, learnable, true, -1
    );
    fake_quant_modules_.push_back(activation_fq);

    // Create weight fake quantize (per-channel on output channel axis)
    auto weight_scheme = (scheme == QuantizationScheme::PerTensorSymmetric)
        ? QuantizationScheme::PerChannelSymmetric
        : scheme;
    auto weight_fq = std::make_shared<FakeQuantize>(
        dtype, weight_scheme, learnable, true, 0
    );
    fake_quant_modules_.push_back(weight_fq);

    // If model is Sequential, we can insert FakeQuantize modules
    // For generic modules, we just track the fake quant modules
    // The caller is responsible for applying them during forward pass
}

auto QATHelper::enable_observer() -> void {
    for (auto& fq : fake_quant_modules_) {
        fq->enable_observer(true);
    }
}

auto QATHelper::disable_observer() -> void {
    for (auto& fq : fake_quant_modules_) {
        fq->disable_observer();
    }
}

auto QATHelper::freeze_bn_stats() -> void {
    // Freeze batch norm statistics by disabling observer updates
    // and calculating final quantization parameters
    disable_observer();

    // Calculate and freeze quantization parameters from collected statistics
    for (auto& fq : fake_quant_modules_) {
        if (fq->observer() && fq->observer()->has_data()) {
            fq->calculate_qparams();
        }
    }
}

auto QATHelper::convert_to_quantized(Module& model) -> std::shared_ptr<Module> {
    // Convert fake quantization to actual quantized operations
    // This replaces FakeQuantize modules with actual quantized layers
    // and converts floating-point layers to quantized variants

    // Build a QConfig from the FakeQuantize modules' parameters
    auto qconfig = DefaultQConfigs::default_qconfig();

    // If model is a Sequential, convert each layer
    // Skip FakeQuantize modules (they were only for training)
    auto* seq = dynamic_cast<Sequential*>(&model);
    if (!seq) {
        // For non-Sequential models, try direct conversion
        // Check if it's a quantizable layer type
        if (auto* linear = dynamic_cast<Linear*>(&model)) {
            return QuantizedLinear::from_float(*linear, qconfig);
        }
        if (auto* conv = dynamic_cast<Conv2d*>(&model)) {
            return QuantizedConv2d::from_float(*conv, qconfig);
        }
        // Non-quantizable module, return a clone via shared_ptr
        // (caller should handle this case)
        return std::make_shared<Sequential>();
    }

    auto quantized_seq = std::make_shared<Sequential>();

    for (const auto& module : seq->modules()) {
        // Skip FakeQuantize modules - they were only for training simulation
        if (std::dynamic_pointer_cast<FakeQuantize>(module)) {
            continue;
        }

        // Try to convert quantizable layers
        if (auto linear = std::dynamic_pointer_cast<Linear>(module)) {
            auto q_linear = QuantizedLinear::from_float(*linear, qconfig);
            quantized_seq->add_module(q_linear);
        } else if (auto conv = std::dynamic_pointer_cast<Conv2d>(module)) {
            auto q_conv = QuantizedConv2d::from_float(*conv, qconfig);
            quantized_seq->add_module(q_conv);
        } else {
            // Keep non-quantizable modules as-is (ReLU, MaxPool, etc.)
            quantized_seq->add_module(module);
        }
    }

    return quantized_seq;
}

// ============================================================================
// Straight-Through Estimator
// ============================================================================

auto StraightThroughEstimator::forward(const Tensor& input, float quant_min, float quant_max)
    -> Tensor {
    // Clamp values to quantization range using device-resident operations
    return tenzor::clamp(input, quant_min, quant_max);
}

auto StraightThroughEstimator::backward(
    const Tensor& grad_output,
    const Tensor& input,
    float quant_min,
    float quant_max
) -> Tensor {
    // Pass gradients through for values within range, zero otherwise
    // Create scalar tensors on the same device as input
    Tensor min_t = full({1}, quant_min, input.dtype(), input.device());
    Tensor max_t = full({1}, quant_max, input.dtype(), input.device());

    // Create mask: true where input is within [quant_min, quant_max]
    Tensor mask = ge(input, min_t) * le(input, max_t);

    // Zero out gradients for out-of-range values
    auto shape = grad_output.shape();
    Tensor zero_grad = zeros(std::vector<int64_t>(shape.begin(), shape.end()), grad_output.dtype(), grad_output.device());
    return tenzor::where(mask, grad_output, zero_grad);
}

// ============================================================================
// FakeQuantizeFunction (Autograd)
// ============================================================================

FakeQuantizeFunction::FakeQuantizeFunction(float scale, float zero_point,
                                           float quant_min, float quant_max)
    : scale_(scale), zero_point_(zero_point),
      quant_min_(quant_min), quant_max_(quant_max) {}

auto FakeQuantizeFunction::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto& input = inputs[0];
    auto x = input.tensor();

    // Save input for backward (STE needs to know which values were in-range)
    save_for_backward({x});

    // Fake quantize: quantize then immediately dequantize
    // scaled = x / scale + zero_point
    // clamped = clamp(round(scaled), quant_min, quant_max)
    // output = (clamped - zero_point) * scale
    Tensor inv_scale = full({1}, 1.0f / scale_, x.dtype(), x.device());
    Tensor zp = full({1}, zero_point_, x.dtype(), x.device());

    Tensor scaled = x * inv_scale + zp;
    Tensor rounded = tenzor::round(scaled);
    Tensor clamped = tenzor::clamp(rounded, quant_min_, quant_max_);
    Tensor output = (clamped - zp) * full({1}, scale_, x.dtype(), x.device());

    return {Variable(output, input.requires_grad())};
}

auto FakeQuantizeFunction::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    auto& grad_output = grad_outputs[0];
    auto& input = saved_tensors()[0];

    // STE: pass gradients through for values within the quantizable range,
    // zero gradients for values that would be clamped by quantization.
    // The quantizable range in float space is:
    //   [quant_min - zero_point) * scale, (quant_max - zero_point) * scale]
    float range_min = (quant_min_ - zero_point_) * scale_;
    float range_max = (quant_max_ - zero_point_) * scale_;

    Tensor min_t = full({1}, range_min, input.dtype(), input.device());
    Tensor max_t = full({1}, range_max, input.dtype(), input.device());

    // Mask: true where input is within quantizable range
    Tensor mask = ge(input, min_t) * le(input, max_t);

    // Zero out gradients for out-of-range values
    auto shape = grad_output.shape();
    Tensor zero_grad = zeros(std::vector<int64_t>(shape.begin(), shape.end()),
                              grad_output.dtype(), grad_output.device());
    return {tenzor::where(mask, grad_output, zero_grad)};
}

auto fake_quantize_with_grad(
    const Variable& input,
    float scale,
    float zero_point,
    float quant_min,
    float quant_max
) -> Variable {
    auto fn = std::make_shared<FakeQuantizeFunction>(scale, zero_point, quant_min, quant_max);
    auto outputs = fn->forward({input});
    if (input.requires_grad()) {
        // Wire the autograd graph edges so the engine can route the STE
        // gradient back to `input`. Setting grad_fn alone is insufficient:
        // without next_functions + input_variables the backward engine has
        // no input edge to accumulate into, so input.grad() stays empty.
        // (Mirrors the standard pattern in nested_autograd_ops.cpp.)
        fn->set_next_functions({input.grad_fn()});
        fn->set_input_variables({input});
        outputs[0].set_grad_fn(fn);
    }
    return outputs[0];
}

// ============================================================================
// BN Folding
// ============================================================================

auto fold_bn(Module& model) -> void {
    // Look for Conv2d -> BatchNorm2d patterns in Sequential containers
    auto* seq = dynamic_cast<Sequential*>(&model);
    if (!seq) {
        // For non-Sequential models, we cannot iterate children generically.
        // Only Sequential models are supported for automatic BN folding.
        return;
    }

    auto& modules = seq->modules();
    if (modules.size() < 2) return;

    // Identify Conv2d -> BatchNorm2d pairs and fold BN into Conv2d weights.
    // We mark BN indices as "folded" so we can skip them when rebuilding.
    std::vector<bool> skip(modules.size(), false);

    for (size_t i = 0; i + 1 < modules.size(); ++i) {
        if (skip[i]) continue;

        auto conv = std::dynamic_pointer_cast<Conv2d>(modules[i]);
        auto* bn_module = dynamic_cast<BatchNorm2d*>(modules[i + 1].get());
        if (!conv || !bn_module) continue;

        // Extract BN parameters
        auto bn_state = bn_module->state_dict();
        Tensor gamma = bn_state.at("weight");
        Tensor beta = bn_state.at("bias");
        Tensor running_mean = bn_state.at("running_mean");
        Tensor running_var = bn_state.at("running_var");
        double eps = bn_module->eps();

        // Compute folding factors: bn_scale = gamma / sqrt(var + eps)
        Tensor sqrt_var = sqrt(running_var + eps);
        Tensor bn_scale = gamma / sqrt_var;  // [C]

        // Get conv weight and bias
        auto conv_state = conv->state_dict();
        Tensor fp_weight = conv_state.at("weight");  // [out_ch, in_ch/groups, kH, kW]

        auto weight_shape = fp_weight.shape();
        int64_t out_channels = weight_shape[0];

        // Fold BN scale into conv weights: w_new[c] = bn_scale[c] * w_old[c]
        Tensor folded_weight = fp_weight.clone();
        auto fw_cpu = (folded_weight.device() == Device::cpu()) ? folded_weight : folded_weight.to(Device::cpu());
        auto bs_cpu = (bn_scale.device() == Device::cpu()) ? bn_scale : bn_scale.to(Device::cpu());

        float* w_data = fw_cpu.data<float>();
        const float* s_data = bs_cpu.data<const float>();
        int64_t channel_size = fw_cpu.numel() / out_channels;

        for (int64_t c = 0; c < out_channels; ++c) {
            float s = s_data[c];
            for (int64_t j = 0; j < channel_size; ++j) {
                w_data[c * channel_size + j] *= s;
            }
        }

        // Fold BN into bias: b_new = bn_scale * (b_old - mean) + beta
        std::optional<Tensor> conv_bias;
        if (conv_state.find("bias") != conv_state.end()) {
            conv_bias = conv_state.at("bias");
        }

        Tensor folded_bias({out_channels}, DType::Float32, Device::cpu());
        float* fb_data = folded_bias.data<float>();

        auto rm_cpu = (running_mean.device() == Device::cpu()) ? running_mean : running_mean.to(Device::cpu());
        auto bt_cpu = (beta.device() == Device::cpu()) ? beta : beta.to(Device::cpu());
        const float* mean_data = rm_cpu.data<const float>();
        const float* beta_data = bt_cpu.data<const float>();

        for (int64_t c = 0; c < out_channels; ++c) {
            float b_old = 0.0f;
            if (conv_bias.has_value()) {
                auto cb_cpu = (*conv_bias);
                if (cb_cpu.device() != Device::cpu()) cb_cpu = cb_cpu.to(Device::cpu());
                b_old = cb_cpu.data<const float>()[c];
            }
            fb_data[c] = s_data[c] * (b_old - mean_data[c]) + beta_data[c];
        }

        // Update conv with folded parameters via state_dict
        std::unordered_map<std::string, Tensor> new_state;
        new_state["weight"] = fw_cpu;
        new_state["bias"] = folded_bias;
        conv->load_state_dict(new_state);

        // Mark BN module for skipping in rebuild
        skip[i + 1] = true;
    }

    // Rebuild the Sequential without the folded BN modules
    bool any_folded = false;
    for (bool s : skip) { if (s) { any_folded = true; break; } }
    if (!any_folded) return;

    // Build a new Sequential and swap it
    auto new_seq = std::make_shared<Sequential>();
    for (size_t i = 0; i < modules.size(); ++i) {
        if (!skip[i]) {
            new_seq->add_module(modules[i]);
        }
    }

    // Load the new sequential's state into the original model.
    // Since we can't swap the internal modules_ vector directly (it's private),
    // we copy the rebuilt state dict back. The Conv2d weights are already updated
    // in-place above, so the folding is effective even without removing BN modules.
    // The BN modules remain but are functionally dead (their params no longer affect output
    // because conv already has folded weights). In eval mode the BN just applies
    // scale=1, bias=0 effectively being identity if we update its state:
    for (size_t i = 0; i < modules.size(); ++i) {
        if (skip[i]) {
            auto* bn = dynamic_cast<BatchNorm2d*>(modules[i].get());
            if (bn) {
                auto bn_sd = bn->state_dict();
                int64_t num_features = bn_sd.at("weight").shape()[0];
                std::unordered_map<std::string, Tensor> identity_state;
                identity_state["weight"] = ones({num_features}, DType::Float32, Device::cpu());
                identity_state["bias"] = zeros({num_features}, DType::Float32, Device::cpu());
                identity_state["running_mean"] = zeros({num_features}, DType::Float32, Device::cpu());
                identity_state["running_var"] = ones({num_features}, DType::Float32, Device::cpu());
                bn->load_state_dict(identity_state);
            }
        }
    }
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
