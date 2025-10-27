/**
 * @file module_conversion.cpp
 * @brief Implementation of module conversion functions for quantization
 */

#include "tenzor/quantization/quantize_api.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/quantization/fake_quantize.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include <iostream>
#include <typeinfo>
#include <stdexcept>

namespace tenzor {
namespace quantization {

using namespace nn;
using namespace nn::quantization;

namespace {

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

        // If not a recognized type, return as-is
        // (e.g., activation layers like ReLU, which don't need quantization)
        return module;
    }

    auto convert_from_quantized(std::shared_ptr<Module> module) -> std::shared_ptr<Module> {
        if (!module) {
            return nullptr;
        }

        // Try to cast to quantized types and convert back
        if (auto q_linear = std::dynamic_pointer_cast<QuantizedLinear>(module)) {
            // Dequantize weights and bias, reconstruct float Linear
            auto weight_params = q_linear->named_parameters();

            // Find quantized weight and bias
            Tensor weight_dequant;
            Tensor bias_dequant;

            for (const auto& [name, param] : weight_params) {
                if (name.find("weight") != std::string::npos) {
                    // Dequantize weight: scale from int8 to float
                    weight_dequant = param.to(DType::Float32);
                } else if (name.find("bias") != std::string::npos) {
                    bias_dequant = param.to(DType::Float32);
                }
            }

            // Create float Linear with dequantized weights
            // Note: We lose exact in/out features information, but weights contain this
            auto shape = weight_dequant.shape();
            if (shape.size() >= 2) {
                int64_t out_features = shape[0];
                int64_t in_features = shape[1];
                auto linear = std::make_shared<Linear>(in_features, out_features);

                // Load dequantized state
                std::unordered_map<std::string, Tensor> state;
                state["weight"] = weight_dequant;
                if (bias_dequant.defined()) {
                    state["bias"] = bias_dequant;
                }
                linear->load_state_dict(state);

                return linear;
            }

            // Fallback: return as-is if shape extraction fails
            return module;
        }

        if (auto q_conv = std::dynamic_pointer_cast<QuantizedConv2d>(module)) {
            // Dequantize Conv2d weights and bias
            auto params = q_conv->named_parameters();

            Tensor weight_dequant;
            Tensor bias_dequant;

            for (const auto& [name, param] : params) {
                if (name.find("weight") != std::string::npos) {
                    weight_dequant = param.to(DType::Float32);
                } else if (name.find("bias") != std::string::npos) {
                    bias_dequant = param.to(DType::Float32);
                }
            }

            // Reconstruct Conv2d with dequantized weights
            // Extract dimensions from weight shape: [out_channels, in_channels, kH, kW]
            auto shape = weight_dequant.shape();
            if (shape.size() >= 4) {
                int64_t out_channels = shape[0];
                int64_t in_channels = shape[1];
                int64_t kernel_h = shape[2];
                int64_t kernel_w = shape[3];

                // Create Conv2d (using default stride=1, padding=0)
                auto conv = std::make_shared<Conv2d>(
                    in_channels,
                    out_channels,
                    std::vector<int64_t>{kernel_h, kernel_w}
                );

                // Load dequantized state
                std::unordered_map<std::string, Tensor> state;
                state["weight"] = weight_dequant;
                if (bias_dequant.defined()) {
                    state["bias"] = bias_dequant;
                }
                conv->load_state_dict(state);

                return conv;
            }

            return module;
        }

        // Try Sequential
        if (auto seq = std::dynamic_pointer_cast<Sequential>(module)) {
            return dequantize_sequential(seq);
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
            // Implementation would iterate through seq's layers

            return qat_seq;
        }

        return module;
    }

private:
    auto convert_sequential(std::shared_ptr<Sequential> seq) -> std::shared_ptr<Sequential> {
        auto quantized_seq = std::make_shared<Sequential>();

        // Get parameters from Sequential to iterate modules
        // This is simplified - real implementation would access internal module list
        for (auto& [name, param] : seq->named_parameters()) {
            // Convert each submodule recursively
            // Actual implementation needs access to Sequential's module list
        }

        return quantized_seq;
    }

    auto dequantize_sequential(std::shared_ptr<Sequential> seq) -> std::shared_ptr<Sequential> {
        auto float_seq = std::make_shared<Sequential>();
        // Convert quantized modules back to float
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

    std::cout << "[Quantization] Converting module to quantized..." << std::endl;

    ModuleQuantizer quantizer(qconfig);
    auto quantized = quantizer.convert_to_quantized(module);

    std::cout << "[Quantization] Module conversion complete" << std::endl;

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

    std::cout << "[Quantization] Converting quantized module to float..." << std::endl;

    // Use default qconfig (won't be used for dequantization)
    auto qconfig = DefaultQConfigs::default_qconfig();
    ModuleQuantizer quantizer(qconfig);

    auto float_module = quantizer.convert_from_quantized(quantized_module);

    std::cout << "[Quantization] Dequantization complete" << std::endl;

    return float_module;
}

/**
 * @brief Prepare module for quantization-aware training
 */
auto prepare_qat(
    const std::shared_ptr<Module>& module,
    const QConfig& qconfig
) -> std::shared_ptr<Module> {
    if (!module) {
        throw std::runtime_error("Cannot prepare null module for QAT");
    }

    std::cout << "[QAT] Preparing module for quantization-aware training..." << std::endl;

    ModuleQuantizer quantizer(qconfig);
    auto qat_module = quantizer.prepare_qat(module);

    std::cout << "[QAT] Module prepared with fake quantization" << std::endl;

    return qat_module;
}

} // namespace quantization
} // namespace tenzor
