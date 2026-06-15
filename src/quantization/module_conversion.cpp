/**
 * @file module_conversion.cpp
 * @brief Implementation of module conversion functions for quantization
 */

#include "tenzor/quantization/quantize_api.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/quantization/quantize.hpp"
#include "tenzor/nn/quantization/fake_quantize.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include <iostream>
#include <typeinfo>
#include <stdexcept>
#include <cstdlib>

namespace tenzor {
namespace quantization {

using namespace nn;
using namespace nn::quantization;

namespace {

// Library code must stay silent on stdout by default (it corrupts piped /
// serialized output of embedding applications). Progress messages route through
// this sink: discarded unless TENZOR_QUANT_VERBOSE is set, then sent to stderr.
// Mirrors the qlog() sink in quantize_api.cpp.
inline auto qlog() -> std::ostream& {
    static const bool verbose = (std::getenv("TENZOR_QUANT_VERBOSE") != nullptr);
    static std::ostream null_sink(nullptr);
    return verbose ? std::cerr : null_sink;
}

/**
 * @brief Module visitor for converting float modules to quantized
 */
class ModuleQuantizer {
public:
    ModuleQuantizer(const QConfig& qconfig) : qconfig_(qconfig) {}

    auto convert_to_quantized(std::shared_ptr<Module> module) -> std::shared_ptr<Module> {
        if (!module) {
            return nullptr;
        }

        // Try to cast to Sequential first (container type)
        if (auto seq = std::dynamic_pointer_cast<Sequential>(module)) {
            return convert_sequential(seq);
        }

        // Try converting individual layer types
        // Note: C++ doesn't have great built-in reflection, so we use dynamic_cast
        // In production, would use a type registry or visitor pattern

        // Try Linear
        if (auto linear = std::dynamic_pointer_cast<Linear>(module)) {
            return QuantizedLinear::from_float(*linear, qconfig_);
        }

        // Try Conv2d
        if (auto conv2d = std::dynamic_pointer_cast<Conv2d>(module)) {
            return QuantizedConv2d::from_float(*conv2d, qconfig_);
        }

        // Nested Sequential containers are handled by the Sequential branch
        // above (each child is recursively converted). Arbitrary custom Module
        // containers cannot be converted in place: their forward() binds typed
        // member fields (e.g. `Linear fc1_;`) rather than dispatching through the
        // submodule map, so swapping a submodule entry would not change their
        // computation. Such models must expose their layers via Sequential (or a
        // model-specific quantized reimplementation) to be quantized.
        return module;
    }

    auto convert_from_quantized(std::shared_ptr<Module> module) -> std::shared_ptr<Module> {
        if (!module) {
            return nullptr;
        }

        // Sequential containers recurse first so nested modules get dequantized.
        if (auto seq = std::dynamic_pointer_cast<Sequential>(module)) {
            return dequantize_sequential(seq);
        }

        if (auto q_linear = std::dynamic_pointer_cast<QuantizedLinear>(module)) {
            // Proper dequantization: dequantize_tensor() applies scale and
            // zero_point, so the resulting float tensor is the real recovered
            // weight, not a plain int→float cast of the quantized integers.
            Tensor weight_dequant = dequantize_tensor(q_linear->weight());
            if (weight_dequant.dtype() != DType::Float32) {
                weight_dequant = weight_dequant.to(DType::Float32);
            }

            auto shape = weight_dequant.shape();
            if (shape.size() < 2) {
                return module;
            }
            const int64_t out_features = shape[0];
            const int64_t in_features = shape[1];
            const bool has_bias = q_linear->has_bias();
            auto linear = std::make_shared<Linear>(in_features, out_features, has_bias);

            std::unordered_map<std::string, Tensor> state;
            state["weight"] = weight_dequant;
            if (has_bias) {
                const Tensor& b = q_linear->bias();
                state["bias"] = (b.dtype() == DType::Float32) ? b : b.to(DType::Float32);
            }
            linear->load_state_dict(state);
            return linear;
        }

        if (auto q_conv = std::dynamic_pointer_cast<QuantizedConv2d>(module)) {
            Tensor weight_dequant = dequantize_tensor(q_conv->weight());
            if (weight_dequant.dtype() != DType::Float32) {
                weight_dequant = weight_dequant.to(DType::Float32);
            }

            const bool has_bias = q_conv->has_bias();
            auto conv = std::make_shared<Conv2d>(
                q_conv->in_channels(),
                q_conv->out_channels(),
                q_conv->kernel_size(),
                q_conv->stride(),
                q_conv->padding(),
                q_conv->dilation(),
                q_conv->groups(),
                has_bias
            );

            std::unordered_map<std::string, Tensor> state;
            state["weight"] = weight_dequant;
            if (has_bias) {
                const Tensor& b = q_conv->bias();
                state["bias"] = (b.dtype() == DType::Float32) ? b : b.to(DType::Float32);
            }
            conv->load_state_dict(state);
            return conv;
        }

        return module;
    }

    auto prepare_qat(std::shared_ptr<Module> module) -> std::shared_ptr<Module> {
        if (!module) {
            return nullptr;
        }

        // Insert FakeQuantize modules around quantizable layers
        // This enables quantization-aware training

        if (auto seq = std::dynamic_pointer_cast<Sequential>(module)) {
            auto qat_seq = std::make_shared<Sequential>();

            // Insert FakeQuantize after each quantizable layer
            for (const auto& child : seq->modules()) {
                qat_seq->add_module(child);

                // Insert FakeQuantize after quantizable layers (Linear, Conv2d)
                bool is_quantizable =
                    std::dynamic_pointer_cast<Linear>(child) != nullptr ||
                    std::dynamic_pointer_cast<Conv2d>(child) != nullptr;

                if (is_quantizable) {
                    // The post-layer FakeQuantize observes ACTIVATIONS, so it must
                    // use the activation dtype/scheme — not the weight settings.
                    // Mirror ModuleConverter::prepare_for_qat (the maintained
                    // path): activation quantization is per-tensor by convention
                    // (axis = -1) unless the config selects a per-channel
                    // activation scheme.
                    const auto ascheme = qconfig_.activation_scheme();
                    const bool per_channel =
                        ascheme == QuantizationScheme::PerChannelSymmetric ||
                        ascheme == QuantizationScheme::PerChannelAsymmetric;
                    const int64_t axis = per_channel ? 1 : -1;
                    auto fake_quant = std::make_shared<FakeQuantize>(
                        qconfig_.activation_dtype(),
                        ascheme,
                        /*learnable=*/false,
                        /*observer_enabled=*/true,
                        axis);
                    qat_seq->add_module(fake_quant);
                }
            }

            return qat_seq;
        }

        return module;
    }

private:
    auto convert_sequential(std::shared_ptr<Sequential> seq) -> std::shared_ptr<Sequential> {
        auto quantized_seq = std::make_shared<Sequential>();

        const auto& mods = seq->modules();
        for (size_t i = 0; i < mods.size(); ++i) {
            auto converted = convert_to_quantized(mods[i]);

            // Static-quant calibration: if the next module is the FakeQuantize
            // inserted after this layer, freeze its observed activation qparams
            // into the quantized layer and drop the FakeQuantize (audit
            // quant-static-02 — previously the calibrated stats were discarded).
            if (i + 1 < mods.size()) {
                if (auto fq = std::dynamic_pointer_cast<FakeQuantize>(mods[i + 1])) {
                    // Only freeze qparams if the observer actually received
                    // calibration data. calculate_qparams() throws on an
                    // uncalibrated observer, so a prepared-but-uncalibrated (or
                    // partially calibrated) model would crash here. Mirror
                    // ModuleConverter::extract_activation_qparams and skip the
                    // FakeQuantize without absorbing when there is no data.
                    if (!fq->observer() || !fq->observer()->has_data()) {
                        quantized_seq->add_module(converted);
                        ++i;  // drop the uncalibrated FakeQuantize
                        continue;
                    }
                    fq->calculate_qparams();
                    const auto& aq = fq->get_qparams();
                    bool absorbed = false;
                    if (auto ql = std::dynamic_pointer_cast<QuantizedLinear>(converted)) {
                        ql->set_activation_qparams(aq); absorbed = true;
                    } else if (auto qc = std::dynamic_pointer_cast<QuantizedConv2d>(converted)) {
                        qc->set_activation_qparams(aq); absorbed = true;
                    }
                    if (absorbed) {
                        quantized_seq->add_module(converted);
                        ++i;  // skip the absorbed FakeQuantize
                        continue;
                    }
                }
            }
            quantized_seq->add_module(converted);
        }

        return quantized_seq;
    }

    auto dequantize_sequential(std::shared_ptr<Sequential> seq) -> std::shared_ptr<Sequential> {
        auto float_seq = std::make_shared<Sequential>();

        // Iterate through the Sequential's module list and convert each back
        for (const auto& module : seq->modules()) {
            auto converted = convert_from_quantized(module);
            float_seq->add_module(converted);
        }

        return float_seq;
    }

    const QConfig& qconfig_;
};

} // anonymous namespace

/**
 * @brief Convert floating-point module to quantized module
 */
auto convert_to_quantized(
    const std::shared_ptr<Module>& module,
    const QConfig& qconfig
) -> std::shared_ptr<Module> {
    if (!module) {
        throw std::runtime_error("Cannot convert null module to quantized");
    }

    qlog() << "[Quantization] Converting module to quantized..." << std::endl;

    ModuleQuantizer quantizer(qconfig);
    auto quantized = quantizer.convert_to_quantized(module);

    qlog() << "[Quantization] Module conversion complete" << std::endl;

    return quantized;
}

/**
 * @brief Convert quantized module back to floating-point
 */
auto convert_from_quantized(
    const std::shared_ptr<Module>& quantized_module
) -> std::shared_ptr<Module> {
    if (!quantized_module) {
        throw std::runtime_error("Cannot convert null quantized module");
    }

    qlog() << "[Quantization] Converting quantized module to float..." << std::endl;

    // Use default qconfig (won't be used for dequantization)
    auto qconfig = DefaultQConfigs::default_qconfig();
    ModuleQuantizer quantizer(qconfig);

    auto float_module = quantizer.convert_from_quantized(quantized_module);

    qlog() << "[Quantization] Dequantization complete" << std::endl;

    return float_module;
}

// prepare_qat is defined in quantize_api.cpp

} // namespace quantization
} // namespace tenzor
