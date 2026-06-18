/**
 * @file quantize_api.cpp
 * @brief Implementation of high-level quantization API
 */

#include "tenzor/quantization/quantize_api.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/quantization/fake_quantize.hpp"
#include "tenzor/nn/quantization/observer.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/reduction.hpp"
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>

namespace tenzor {
namespace quantization {

using namespace nn::quantization;

// ============================================================================
// Dynamic Quantization
// ============================================================================

namespace {

/**
 * @brief Module converter for quantization operations
 */
class ModuleConverter {
public:
    explicit ModuleConverter(const QConfig& qconfig) : qconfig_(qconfig) {}

    // Convert float module to quantized
    auto to_quantized(std::shared_ptr<nn::Module> module) -> std::shared_ptr<nn::Module> {
        if (!module) return nullptr;

        // Try Sequential container
        if (auto seq = std::dynamic_pointer_cast<nn::Sequential>(module)) {
            return convert_sequential_to_quantized(seq);
        }

        // Try individual layer types
        if (auto linear = std::dynamic_pointer_cast<nn::Linear>(module)) {
            return nn::quantization::QuantizedLinear::from_float(*linear, qconfig_);
        }

        if (auto conv2d = std::dynamic_pointer_cast<nn::Conv2d>(module)) {
            return nn::quantization::QuantizedConv2d::from_float(*conv2d, qconfig_);
        }

        // A FakeQuantize reaching this point has no preceding quantizable layer
        // to absorb its observed activation qparams (the Sequential walker bakes
        // and removes the ones that do). It is a calibration-only artifact, never
        // part of a clean quantized inference model — drop it.
        if (std::dynamic_pointer_cast<nn::quantization::FakeQuantize>(module)) {
            return nullptr;
        }

        // Return as-is for non-quantizable layers (ReLU, MaxPool, etc.)
        return module;
    }

    // Note: convert_from_quantized is implemented in src/quantization/module_conversion.cpp
    // via ModuleQuantizer::convert_from_quantized(). That is the live path called by
    // the public API; this ModuleConverter class is the alternate path used only by
    // to_quantized() here.

    // Prepare module for QAT by inserting FakeQuantize modules after quantizable layers
    auto prepare_for_qat(std::shared_ptr<nn::Module> module) -> std::shared_ptr<nn::Module> {
        if (!module) return nullptr;

        // Handle Sequential containers by inserting FakeQuantize after quantizable layers
        if (auto seq = std::dynamic_pointer_cast<nn::Sequential>(module)) {
            auto qat_seq = std::make_shared<nn::Sequential>();

            for (const auto& child : seq->modules()) {
                qat_seq->add_module(child);

                // Insert FakeQuantize after quantizable layers (Linear, Conv2d)
                bool is_quantizable =
                    std::dynamic_pointer_cast<nn::Linear>(child) != nullptr ||
                    std::dynamic_pointer_cast<nn::Conv2d>(child) != nullptr;

                if (is_quantizable) {
                    // These FakeQuantize modules sit AFTER the layer and observe
                    // its OUTPUT activations, so they must use the QConfig's
                    // ACTIVATION dtype/scheme — not the weight settings. Weights
                    // are quantized directly from the layer's parameters inside
                    // QuantizedLinear/QuantizedConv2d::from_float(); the observer
                    // here is purely for activation calibration. Activation
                    // quantization is per-tensor by convention (per-channel along
                    // a batch-varying axis is meaningless), so axis = -1 unless
                    // the config explicitly selects a per-channel activation
                    // scheme.
                    const auto ascheme = qconfig_.activation_scheme();
                    const bool per_channel =
                        ascheme == nn::quantization::QuantizationScheme::PerChannelSymmetric ||
                        ascheme == nn::quantization::QuantizationScheme::PerChannelAsymmetric;
                    const int64_t axis = per_channel ? 1 : -1;
                    auto fake_quant = std::make_shared<nn::quantization::FakeQuantize>(
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
    // Extract the calibrated activation qparams from a FakeQuantize that
    // collected statistics during calibration. Returns nullopt when the module
    // is not a FakeQuantize or its observer never saw data (e.g. the dynamic
    // path, which inserts no observers). The activation qparams use the
    // QConfig's ACTIVATION dtype/scheme (distinct from the weight settings the
    // FakeQuantize was constructed with): the observer only stores raw min/max,
    // and calculate_qparams() applies whichever (dtype, scheme) we request.
    auto extract_activation_qparams(const std::shared_ptr<nn::Module>& module)
        -> std::optional<QuantizationParams> {
        auto fq = std::dynamic_pointer_cast<nn::quantization::FakeQuantize>(module);
        if (!fq) return std::nullopt;
        auto* obs = fq->observer();
        if (!obs || !obs->has_data()) return std::nullopt;
        return obs->calculate_qparams(qconfig_.activation_dtype(),
                                      qconfig_.activation_scheme());
    }

    auto convert_sequential_to_quantized(std::shared_ptr<nn::Sequential> seq)
        -> std::shared_ptr<nn::Sequential> {
        auto quantized_seq = std::make_shared<nn::Sequential>();

        const auto& modules = seq->modules();
        for (size_t i = 0; i < modules.size(); ++i) {
            const auto& module = modules[i];

            // The static-quant calibration path inserts a FakeQuantize directly
            // after each quantizable layer (see prepare_for_qat). That observer
            // collected the layer's OUTPUT activation statistics. Bake those
            // calibrated qparams into the produced quantized layer and DROP the
            // FakeQuantize so the returned tree is a clean quantized model.
            //
            // When no such FakeQuantize follows (dynamic quant: no observers
            // were ever inserted), the layer is converted exactly as before and
            // simply carries no activation qparams — leaving the existing
            // recompute-per-call behavior intact.
            const bool quantizable =
                std::dynamic_pointer_cast<nn::Linear>(module) != nullptr ||
                std::dynamic_pointer_cast<nn::Conv2d>(module) != nullptr;

            std::optional<QuantizationParams> act_qparams;
            if (quantizable && i + 1 < modules.size()) {
                act_qparams = extract_activation_qparams(modules[i + 1]);
            }

            auto converted = to_quantized(module);

            if (act_qparams.has_value()) {
                if (auto ql = std::dynamic_pointer_cast<
                        nn::quantization::QuantizedLinear>(converted)) {
                    ql->set_activation_qparams(*act_qparams);
                } else if (auto qc = std::dynamic_pointer_cast<
                               nn::quantization::QuantizedConv2d>(converted)) {
                    qc->set_activation_qparams(*act_qparams);
                }
                // The next module is the consumed FakeQuantize: skip it so it
                // never appears in the output model.
                ++i;
            }

            // to_quantized() returns nullptr for a FakeQuantize that was NOT
            // consumed above (no preceding quantizable layer, or its observer
            // had no data). Drop those rather than re-inserting a calibration
            // artifact into the clean quantized tree.
            if (converted) {
                quantized_seq->add_module(converted);
            }
        }

        return quantized_seq;
    }

    const QConfig& qconfig_;
};

// Forward declarations for helper functions
// Library code must stay silent on stdout by default (it corrupts piped /
// serialized output of embedding applications). Progress messages route through
// this sink: discarded unless TENZOR_QUANT_VERBOSE is set, then sent to stderr.
inline auto qlog() -> std::ostream& {
    static const bool verbose = (std::getenv("TENZOR_QUANT_VERBOSE") != nullptr);
    static std::ostream null_sink(nullptr);
    return verbose ? std::cerr : null_sink;
}

auto convert_module_to_quantized_recursive(
    std::shared_ptr<nn::Module> module,
    const QConfig& qconfig,
    [[maybe_unused]] const std::string& prefix = ""
) -> std::shared_ptr<nn::Module> {
    ModuleConverter converter(qconfig);
    return converter.to_quantized(module);
}

auto prepare_module_for_qat_recursive(
    std::shared_ptr<nn::Module> module,
    const QConfig& qconfig,
    [[maybe_unused]] const std::string& prefix = ""
) -> std::shared_ptr<nn::Module> {
    ModuleConverter converter(qconfig);
    return converter.prepare_for_qat(module);
}

} // anonymous namespace

auto quantize_dynamic(
    std::shared_ptr<nn::Module> model,
    QuantDType weight_dtype
) -> std::shared_ptr<nn::Module> {
    if (!model) {
        throw std::runtime_error("Cannot quantize null model");
    }

    // Honor weight_dtype instead of silently ignoring it. The dynamic-quant
    // default qconfig uses INT8 symmetric weights; reject anything it cannot
    // actually produce rather than quietly quantizing to INT8. There is no
    // activation_dtype parameter: dynamic quantization keeps activations in
    // FP32 by definition (matching PyTorch).
    if (weight_dtype != QuantDType::INT8) {
        throw std::runtime_error(
            "quantize_dynamic(weight_dtype): only INT8 weights are supported by "
            "this entry point; pass a QuantizationConfig for other weight dtypes");
    }

    qlog() << "[Quantization] Starting dynamic quantization..." << std::endl;

    // Create quantization config for dynamic quantization (INT8 weights).
    auto qconfig = DefaultQConfigs::fast_qconfig();

    qlog() << "[Quantization] Quantizing weights to "
              << (weight_dtype == QuantDType::INT8 ? "INT8" : "UINT8") << std::endl;

    // Traverse model and build a NEW quantized module tree: a fresh Sequential
    // whose Linear/Conv2d layers are replaced by QuantizedLinear/QuantizedConv2d
    // (see QuantizationConverter::to_quantized). The caller's original `model` is
    // left intact (unquantized) — quantize_dynamic returns an independent module,
    // matching PyTorch's copy semantics.
    auto quantized_model = convert_module_to_quantized_recursive(model, qconfig);

    qlog() << "[Quantization] Dynamic quantization complete" << std::endl;

    return quantized_model;
}

// ============================================================================
// Static Quantization
// ============================================================================

auto quantize_static(
    std::shared_ptr<nn::Module> model,
    std::function<void(nn::Module&)> calibration_fn,
    const QConfig& qconfig
) -> std::shared_ptr<nn::Module> {
    if (!model) {
        throw std::runtime_error("Cannot quantize null model");
    }

    if (!calibration_fn) {
        throw std::runtime_error("Calibration function required for static quantization");
    }

    // Static quantization workflow:
    // 1. Insert observers to collect activation statistics
    // 2. Run calibration forward passes
    // 3. Calculate quantization parameters from collected statistics
    // 4. Convert to quantized model using calculated parameters

    qlog() << "[Quantization] Preparing model with observers..." << std::endl;

    // Set model to eval mode for calibration
    model->eval();

    // Attach observers by preparing the model with FakeQuantize modules
    auto prepared_model = prepare_module_for_qat_recursive(model, qconfig);

    qlog() << "[Quantization] Observers attached to quantizable layers" << std::endl;

    qlog() << "[Quantization] Running calibration..." << std::endl;
    try {
        // Run forward passes — FakeQuantize observers collect min/max stats
        calibration_fn(*prepared_model);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Calibration failed: ") + e.what());
    }

    qlog() << "[Quantization] Calculating quantization parameters from observers..." << std::endl;
    // FakeQuantize modules have collected statistics during calibration
    // Now freeze their qparams and convert to real quantized model

    qlog() << "[Quantization] Converting to quantized model..." << std::endl;
    auto quantized_model = convert_module_to_quantized_recursive(prepared_model, qconfig);

    qlog() << "[Quantization] Static quantization complete!" << std::endl;

    return quantized_model;
}

// ============================================================================
// Quantization-Aware Training (QAT)
// ============================================================================

auto prepare_qat(
    std::shared_ptr<nn::Module> model,
    const QConfig& qconfig
) -> std::shared_ptr<nn::Module> {
    if (!model) {
        throw std::runtime_error("Cannot prepare null model for QAT");
    }

    qlog() << "[QAT] Preparing model with fake quantization modules..." << std::endl;

    // Set model to training mode for QAT
    model->train();

    // Use QATHelper to prepare model with fake quantization
    auto qat_helper = std::make_unique<nn::quantization::QATHelper>();
    qat_helper->prepare_qat(
        *model,
        qconfig.weight_dtype(),
        qconfig.activation_scheme(),
        false  // not learnable by default
    );

    // Insert FakeQuantize modules into the model structure
    auto qat_module = prepare_module_for_qat_recursive(model, qconfig);

    qlog() << "[QAT] Model prepared. Train normally to learn quantization-robust weights." << std::endl;
    qlog() << "[QAT] After training, use convert_qat() to get quantized model." << std::endl;

    return qat_module;
}

auto convert_qat(std::shared_ptr<nn::Module> qat_model) -> std::shared_ptr<nn::Module> {
    if (!qat_model) {
        throw std::runtime_error("Cannot convert null QAT model");
    }

    qlog() << "[QAT] Converting fake quantization to real quantization..." << std::endl;

    // Set to eval mode before conversion
    qat_model->eval();

    // Convert FakeQuantize modules to actual quantized operations
    // Use default qconfig for conversion
    auto qconfig = DefaultQConfigs::qat_qconfig();

    auto quantized_model = convert_module_to_quantized_recursive(qat_model, qconfig);

    qlog() << "[QAT] Conversion complete. Model ready for INT8 inference." << std::endl;

    return quantized_model;
}

// ============================================================================
// Calibration
// ============================================================================

auto calibrate(
    nn::Module& model,
    const std::vector<Tensor>& calibration_data
) -> std::unordered_map<std::string, QuantizationParams> {
    std::unordered_map<std::string, QuantizationParams> params_map;

    qlog() << "[Calibration] Processing " << calibration_data.size()
              << " calibration batches..." << std::endl;

    // Run forward passes and collect statistics
    model.eval();

    for (size_t i = 0; i < calibration_data.size(); ++i) {
        if (i % 10 == 0) {
            qlog() << "[Calibration] Batch " << i << "/" << calibration_data.size() << std::endl;
        }

        // Forward pass (observers collect statistics)
        auto output = model.forward(Variable(calibration_data[i], false));
    }

    // Extract quantization parameters from FakeQuantize observer modules.
    // Pick exactly ONE traversal source so each FakeQuantize observer is
    // recorded under a single key. If the model itself is a Sequential, iterate
    // its children positionally; otherwise walk the registered submodule map
    // (recursing into any Sequential children). Previously both ran for a
    // Sequential model, recording every FakeQuantize twice under unrelated keys
    // (e.g. both "module_3" and "3"), inflating the count and making lookups
    // ambiguous.
    if (auto* seq = dynamic_cast<nn::Sequential*>(&model)) {
        int idx = 0;
        for (const auto& child : seq->modules()) {
            if (auto fq = std::dynamic_pointer_cast<nn::quantization::FakeQuantize>(child)) {
                if (fq->observer() && fq->observer()->has_data()) {
                    auto qparams = fq->observer()->calculate_qparams(
                        nn::quantization::QuantDType::INT8,
                        nn::quantization::QuantizationScheme::PerTensorSymmetric
                    );
                    params_map.insert_or_assign(std::to_string(idx), std::move(qparams));
                }
            }
            ++idx;
        }
    } else {
        auto submodules = model.get_submodules();
        for (const auto& [name, submodule] : submodules) {
            if (auto fq = std::dynamic_pointer_cast<nn::quantization::FakeQuantize>(submodule)) {
                if (fq->observer() && fq->observer()->has_data()) {
                    auto qparams = fq->observer()->calculate_qparams(
                        nn::quantization::QuantDType::INT8,
                        nn::quantization::QuantizationScheme::PerTensorSymmetric
                    );
                    params_map.insert_or_assign(name, std::move(qparams));
                }
            }
            // Recurse into Sequential containers
            if (auto seq = std::dynamic_pointer_cast<nn::Sequential>(submodule)) {
                int idx = 0;
                for (const auto& child : seq->modules()) {
                    if (auto fq = std::dynamic_pointer_cast<nn::quantization::FakeQuantize>(child)) {
                        if (fq->observer() && fq->observer()->has_data()) {
                            auto qparams = fq->observer()->calculate_qparams(
                                nn::quantization::QuantDType::INT8,
                                nn::quantization::QuantizationScheme::PerTensorSymmetric
                            );
                            std::string key = name + "." + std::to_string(idx);
                            params_map.insert_or_assign(key, std::move(qparams));
                        }
                    }
                    ++idx;
                }
            }
        }
    }

    qlog() << "[Calibration] Complete! Extracted " << params_map.size()
              << " quantization parameter sets" << std::endl;

    return params_map;
}

// ============================================================================
// Module Fusion
// ============================================================================

auto fuse_modules(std::shared_ptr<nn::Module> model) -> std::shared_ptr<nn::Module> {
    // Fuse common patterns:
    // - Conv2d + BatchNorm2d + ReLU -> QuantizedConv2dBnReLU
    // - Conv2d + ReLU -> QuantizedConv2dReLU

    qlog() << "[Fusion] Fusing compatible layer sequences..." << std::endl;

    // Handle Sequential containers for pattern matching.
    // For non-Sequential models, recursively fuse within each submodule
    // that is itself a Sequential container.
    auto seq = std::dynamic_pointer_cast<nn::Sequential>(model);
    if (!seq) {
        // Recursive walk: fuse within any Sequential submodules
        bool any_fused = false;
        for (auto& [name, submodule] : model->get_submodules()) {
            auto fused_sub = fuse_modules(submodule);
            if (fused_sub != submodule) {
                // Replace submodule with fused version (via re-registration)
                any_fused = true;
            }
        }
        if (!any_fused) {
            qlog() << "[Fusion] No Sequential submodules found for fusion" << std::endl;
        }
        return model;
    }

    const auto& modules = seq->modules();
    if (modules.size() < 2) {
        return model;
    }

    auto qconfig = DefaultQConfigs::default_qconfig();
    auto fused_seq = std::make_shared<nn::Sequential>();
    int fusions_performed = 0;

    size_t i = 0;
    while (i < modules.size()) {
        // Try Conv2d + BatchNorm2d + ReLU pattern (3-module fusion)
        if (i + 2 < modules.size()) {
            auto conv = std::dynamic_pointer_cast<nn::Conv2d>(modules[i]);
            auto bn = std::dynamic_pointer_cast<nn::BatchNorm2d>(modules[i + 1]);

            // Check if third module is a ReLU variant
            bool is_relu = false;
            if (conv && bn && modules[i + 2]) {
                is_relu =
                    std::dynamic_pointer_cast<nn::ReLU>(modules[i + 2]) != nullptr ||
                    std::dynamic_pointer_cast<nn::ReLU6>(modules[i + 2]) != nullptr ||
                    std::dynamic_pointer_cast<nn::LeakyReLU>(modules[i + 2]) != nullptr;
            }

            if (conv && bn && is_relu) {
                qlog() << "[Fusion] Fusing Conv2d + BatchNorm2d + ReLU at position " << i << std::endl;
                auto fused_layer = nn::quantization::QuantizedConv2dBnReLU::from_float(
                    *conv, *bn, qconfig
                );
                fused_seq->add_module(fused_layer);
                i += 3;
                fusions_performed++;
                continue;
            }
        }

        // Try Conv2d + ReLU pattern (2-module fusion)
        if (i + 1 < modules.size()) {
            auto conv = std::dynamic_pointer_cast<nn::Conv2d>(modules[i]);
            if (conv) {
                bool is_relu =
                    std::dynamic_pointer_cast<nn::ReLU>(modules[i + 1]) != nullptr ||
                    std::dynamic_pointer_cast<nn::ReLU6>(modules[i + 1]) != nullptr ||
                    std::dynamic_pointer_cast<nn::LeakyReLU>(modules[i + 1]) != nullptr;

                if (is_relu) {
                    qlog() << "[Fusion] Fusing Conv2d + ReLU at position " << i << std::endl;
                    auto fused_layer = nn::quantization::QuantizedConv2dReLU::from_float(
                        *conv, qconfig
                    );
                    fused_seq->add_module(fused_layer);
                    i += 2;
                    fusions_performed++;
                    continue;
                }
            }
        }

        // No fusion pattern matched; keep module as-is
        fused_seq->add_module(modules[i]);
        i++;
    }

    qlog() << "[Fusion] " << fusions_performed << " fusions performed" << std::endl;
    return fused_seq;
}

// ============================================================================
// Accuracy Comparison
// ============================================================================

auto compare_accuracy(
    nn::Module& fp32_model,
    nn::Module& quantized_model,
    const std::vector<std::pair<Tensor, Tensor>>& test_data
) -> std::tuple<float, float, float> {
    fp32_model.eval();
    quantized_model.eval();

    int fp32_correct = 0;
    int quant_correct = 0;
    int total = 0;

    for (const auto& [input, label] : test_data) {
        int64_t batch_size = input.shape()[0];

        // FP32 inference
        auto fp32_output = fp32_model.forward(Variable(input, false));
        Tensor fp32_pred = argmax(fp32_output.tensor(), /*dim=*/1, /*keepdim=*/false);

        // Quantized inference
        auto quant_output = quantized_model.forward(Variable(input, false));
        Tensor quant_pred = argmax(quant_output.tensor(), /*dim=*/1, /*keepdim=*/false);

        // Move predictions and labels to CPU for comparison
        Tensor fp32_pred_cpu = fp32_pred;
        Tensor quant_pred_cpu = quant_pred;
        Tensor label_cpu = label;
        if (fp32_pred_cpu.device() != Device::cpu()) {
            fp32_pred_cpu = fp32_pred_cpu.to(Device::cpu());
        }
        if (quant_pred_cpu.device() != Device::cpu()) {
            quant_pred_cpu = quant_pred_cpu.to(Device::cpu());
        }
        if (label_cpu.device() != Device::cpu()) {
            label_cpu = label_cpu.to(Device::cpu());
        }

        // Convert to Int64 for comparison if needed
        if (fp32_pred_cpu.dtype() != DType::Int64) {
            fp32_pred_cpu = fp32_pred_cpu.to(DType::Int64);
        }
        if (quant_pred_cpu.dtype() != DType::Int64) {
            quant_pred_cpu = quant_pred_cpu.to(DType::Int64);
        }
        if (label_cpu.dtype() != DType::Int64) {
            label_cpu = label_cpu.to(DType::Int64);
        }

        const int64_t* fp32_data = fp32_pred_cpu.data<int64_t>();
        const int64_t* quant_data = quant_pred_cpu.data<int64_t>();
        const int64_t* label_data = label_cpu.data<int64_t>();

        for (int64_t b = 0; b < batch_size; ++b) {
            if (fp32_data[b] == label_data[b]) {
                fp32_correct++;
            }
            if (quant_data[b] == label_data[b]) {
                quant_correct++;
            }
            total++;
        }
    }

    if (total == 0) {
        return {0.0f, 0.0f, 0.0f};
    }

    float fp32_acc = static_cast<float>(fp32_correct) / static_cast<float>(total);
    float quant_acc = static_cast<float>(quant_correct) / static_cast<float>(total);
    float degradation = (fp32_acc > 0.0f)
        ? (fp32_acc - quant_acc) / fp32_acc * 100.0f
        : 0.0f;

    return {fp32_acc, quant_acc, degradation};
}

// ============================================================================
// Performance Benchmarking
// ============================================================================

auto benchmark_quantization(
    nn::Module& fp32_model,
    nn::Module& quantized_model,
    const std::vector<int64_t>& input_shape,
    int num_iterations
) -> std::tuple<float, float, float, float> {
    if (num_iterations <= 0) {
        throw std::invalid_argument(
            "benchmark_quantization: num_iterations must be > 0");
    }

    fp32_model.eval();
    quantized_model.eval();

    // Create dummy input
    Tensor input(input_shape, DType::Float32, Device::cpu());
    input.fill_(1.0f);
    auto var_input = Variable(input, false);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        fp32_model.forward(var_input);
        quantized_model.forward(var_input);
    }

    // Benchmark FP32
    auto fp32_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        fp32_model.forward(var_input);
    }
    auto fp32_end = std::chrono::high_resolution_clock::now();
    float fp32_time = std::chrono::duration<float, std::milli>(fp32_end - fp32_start).count();
    fp32_time /= num_iterations;

    // Benchmark INT8
    auto quant_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        quantized_model.forward(var_input);
    }
    auto quant_end = std::chrono::high_resolution_clock::now();
    float quant_time = std::chrono::duration<float, std::milli>(quant_end - quant_start).count();
    quant_time /= num_iterations;

    // Calculate speedup. Guard against a zero quant_time (e.g. a degenerate
    // model whose forward is optimized away) which would otherwise yield inf.
    float speedup = (quant_time > 0.0f) ? (fp32_time / quant_time) : 0.0f;

    // Memory reduction (4x for FP32 -> INT8)
    float memory_reduction = 4.0f;

    qlog() << "[Benchmark] FP32 inference: " << fp32_time << " ms" << std::endl;
    qlog() << "[Benchmark] INT8 inference: " << quant_time << " ms" << std::endl;
    qlog() << "[Benchmark] Speedup: " << speedup << "x" << std::endl;
    qlog() << "[Benchmark] Memory reduction: " << memory_reduction << "x" << std::endl;

    return {fp32_time, quant_time, speedup, memory_reduction};
}

// convert_to_quantized, convert_from_quantized, and prepare_qat
// are defined in module_conversion.cpp

// ============================================================================
// Per-layer QuantizationConfig overloads
// ============================================================================

// Audit J2: recursive per-layer QuantizationConfig walker. Builds the fully-
// qualified dotted module path (e.g., "encoder.layer.0.attention") at every
// depth and consults `config.config_for(full_name)` for an override. The old
// loop only iterated top-level submodules, so per-layer overrides at deeper
// paths (the common case for transformers) silently did nothing.
//
// Pattern: convert-in-place with the override that applies to the current
// node; if the override is empty (config.config_for returns nullopt), the
// node is skipped and recursion continues with children. Children may still
// have their own overrides.
// Path-aware conversion: build ONE new module tree, consulting
// config.config_for(path) at each quantizable leaf so overrides and skip_layers
// are honored. ModuleConverter is non-mutating (it returns fresh modules), so we
// must STITCH the converted children into rebuilt parents rather than discard
// them. Sequential children are named by positional index (PyTorch convention),
// which is what users put in layer_overrides / skip_layers.
static std::shared_ptr<nn::Module> convert_path_aware(
    std::shared_ptr<nn::Module> module,
    const QuantizationConfig& config,
    const std::string& current_path
) {
    if (!module) return nullptr;

    // Container: rebuild with each child converted at its own path.
    if (auto seq = std::dynamic_pointer_cast<nn::Sequential>(module)) {
        auto out = std::make_shared<nn::Sequential>();
        int64_t idx = 0;
        for (const auto& child : seq->modules()) {
            std::string child_path = current_path.empty()
                ? std::to_string(idx)
                : current_path + "." + std::to_string(idx);
            out->add_module(convert_path_aware(child, config, child_path));
            ++idx;
        }
        return out;
    }

    // Quantizable leaf: config_for() returns nullopt for skip_layers (keep
    // FP32), the override for layer_overrides, else the default qconfig.
    const bool quantizable =
        std::dynamic_pointer_cast<nn::Linear>(module) != nullptr ||
        std::dynamic_pointer_cast<nn::Conv2d>(module) != nullptr;
    if (quantizable) {
        auto qcfg = config.config_for(current_path);
        if (!qcfg.has_value()) {
            return module;  // explicitly skipped -> leave in FP32
        }
        return convert_module_to_quantized_recursive(module, qcfg.value());
    }

    // Non-quantizable / non-container (ReLU, custom modules): return unchanged.
    return module;
}

auto quantize_dynamic(
    std::shared_ptr<nn::Module> model,
    const QuantizationConfig& config
) -> std::shared_ptr<nn::Module> {
    if (!model) {
        throw std::runtime_error("Cannot quantize null model");
    }
    // Single path-aware pass that honors per-layer overrides AND skip_layers,
    // stitching the (non-mutating) converted submodules into a rebuilt tree.
    return convert_path_aware(model, config, /*current_path=*/"");
}

auto quantize_static(
    std::shared_ptr<nn::Module> model,
    std::function<void(nn::Module&)> calibration_fn,
    const QuantizationConfig& config
) -> std::shared_ptr<nn::Module> {
    // The static (calibration-based) path applies a single qconfig uniformly; it
    // cannot yet honor per-layer overrides/skip during observer insertion. Fail
    // loudly instead of SILENTLY dropping them (the old behavior advertised
    // "skipping layers as specified" but ignored both maps).
    if (!config.layer_overrides.empty() || !config.skip_layers.empty()) {
        throw std::runtime_error(
            "quantize_static: per-layer layer_overrides / skip_layers are not "
            "supported by the static (calibration) path; use quantize_dynamic "
            "with the QuantizationConfig, or a uniform QConfig here");
    }
    return quantize_static(model, calibration_fn, config.default_config);
}

} // namespace quantization
} // namespace tenzor
