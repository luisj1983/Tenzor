/**
 * @file quantize_api.hpp
 * @brief High-level quantization API for model conversion
 *
 * Provides simple interfaces for post-training quantization, dynamic quantization,
 * and quantization-aware training workflows.
 */

#pragma once

#include "../nn/quantization.hpp"
#include "../nn/module.hpp"
#include "../data/dataloader.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace tenzor {
namespace quantization {

/**
 * @brief Quantization mode for model conversion.
 */
enum class QuantizationMode {
    Dynamic,     ///< Weights quantized, activations computed in FP32
    Static,      ///< Weights and activations quantized (requires calibration)
    QAT          ///< Quantization-aware training
};

/**
 * @brief Per-layer quantization configuration.
 *
 * Allows specifying different QConfigs for different layers, and
 * excluding specific layers from quantization entirely.
 *
 * Keys are POSITIONAL dotted paths, NOT semantic module names. The path-aware
 * converter walks Sequential children by index and builds each key from the
 * zero-based child index joined by '.' (e.g. the second top-level child is
 * "1"; the first child of the third child is "2.0"). config_for() does an
 * exact lookup against these positional keys — semantic names like "fc1" will
 * never match and the layer is quantized with default_config.
 *
 * @code
 * QuantizationConfig config;
 * config.default_config = DefaultQConfigs::default_qconfig();
 * config.layer_overrides["1"] = qconfig_int4;  // INT4 for the 2nd child
 * config.skip_layers.insert("0");               // Keep the 1st child in FP32
 *
 * auto q_model = quantize_dynamic(model, config);
 * @endcode
 */
struct QuantizationConfig {
    nn::quantization::QConfig default_config = nn::quantization::DefaultQConfigs::default_qconfig();
    std::unordered_map<std::string, nn::quantization::QConfig> layer_overrides;
    std::unordered_set<std::string> skip_layers;

    /// Get the QConfig for a named layer (checks overrides, then default).
    /// Returns nullopt if the layer should be skipped.
    auto config_for(const std::string& name) const
        -> std::optional<nn::quantization::QConfig> {
        if (skip_layers.count(name)) return std::nullopt;
        auto it = layer_overrides.find(name);
        if (it != layer_overrides.end()) return it->second;
        return default_config;
    }
};

/**
 * @brief Dynamic quantization - quantize weights only.
 *
 * Converts floating-point model weights to INT8 while keeping activations
 * in FP32. No calibration required, fastest to apply but less performance
 * improvement than static quantization.
 *
 * Activations are always kept in FP32 by definition of dynamic quantization
 * (they are quantized on the fly per inference, not stored as INT8), so there
 * is deliberately no activation_dtype parameter — passing one previously had
 * no effect and only misrepresented the FP32-activation contract.
 *
 * @param model Model to quantize
 * @param weight_dtype Weight quantization data type (default: INT8; only INT8
 *        is supported by this entry point — pass a QuantizationConfig for other
 *        weight dtypes)
 * @return Quantized model
 *
 * @code
 * auto model = std::make_shared<ResNet50>();
 * // Train model...
 *
 * // Apply dynamic quantization
 * auto q_model = quantize_dynamic(model);
 * @endcode
 */
auto quantize_dynamic(
    std::shared_ptr<nn::Module> model,
    nn::quantization::QuantDType weight_dtype = nn::quantization::QuantDType::INT8
) -> std::shared_ptr<nn::Module>;

/// Overload with per-layer config — allows skipping layers or using different QConfigs.
auto quantize_dynamic(
    std::shared_ptr<nn::Module> model,
    const QuantizationConfig& config
) -> std::shared_ptr<nn::Module>;

/**
 * @brief Static quantization - quantize weights and activations.
 *
 * Converts both weights and activations to INT8. Requires calibration
 * on representative data to determine activation quantization parameters.
 * Provides best inference performance.
 *
 * @param model Model to quantize
 * @param calibration_fn Function that runs calibration forward passes
 * @param qconfig Quantization configuration
 * @return Quantized model
 *
 * @code
 * auto model = std::make_shared<ResNet50>();
 * // Train model...
 *
 * // Prepare calibration data
 * DataLoader calib_loader(calib_dataset, 32);
 *
 * // Calibration function
 * auto calibrate = [&](nn::Module& m) {
 *     for (int i = 0; i < 100; ++i) {
 *         auto batch = calib_loader.next();
 *         m.forward(batch.data);
 *     }
 * };
 *
 * // Apply static quantization
 * auto q_model = quantize_static(model, calibrate);
 * @endcode
 */
auto quantize_static(
    std::shared_ptr<nn::Module> model,
    std::function<void(nn::Module&)> calibration_fn,
    const nn::quantization::QConfig& qconfig = nn::quantization::DefaultQConfigs::default_qconfig()
) -> std::shared_ptr<nn::Module>;

/// Overload accepting a QuantizationConfig.
///
/// @note The static (calibration-based) path currently applies only
///       `config.default_config` uniformly to all quantizable layers. Per-layer
///       customization is NOT yet supported here: if `config.layer_overrides`
///       or `config.skip_layers` is non-empty, this overload throws
///       std::runtime_error rather than silently ignoring them. For per-layer
///       overrides / skips use quantize_dynamic, which honors them.
/// @throws std::runtime_error if config.layer_overrides or config.skip_layers
///         is non-empty.
auto quantize_static(
    std::shared_ptr<nn::Module> model,
    std::function<void(nn::Module&)> calibration_fn,
    const QuantizationConfig& config
) -> std::shared_ptr<nn::Module>;

/**
 * @brief Prepare model for quantization-aware training.
 *
 * Inserts fake quantization modules into the model to simulate quantization
 * during training. The model learns to be robust to quantization errors.
 *
 * @param model Model to prepare for QAT
 * @param qconfig Quantization configuration
 * @return Model with fake quantization modules
 *
 * @code
 * auto model = std::make_shared<ResNet50>();
 *
 * // Prepare for QAT
 * auto qat_model = prepare_qat(model);
 *
 * // Train with quantization simulation
 * for (int epoch = 0; epoch < epochs; ++epoch) {
 *     for (auto& batch : train_loader) {
 *         auto output = qat_model->forward(batch.data);
 *         auto loss = criterion(output, batch.labels);
 *         loss.backward();
 *         optimizer.step();
 *     }
 * }
 *
 * // Convert to quantized model
 * auto q_model = convert_qat(qat_model);
 * @endcode
 */
auto prepare_qat(
    std::shared_ptr<nn::Module> model,
    const nn::quantization::QConfig& qconfig = nn::quantization::DefaultQConfigs::qat_qconfig()
) -> std::shared_ptr<nn::Module>;

/**
 * @brief Convert QAT model to quantized model.
 *
 * Replaces fake quantization modules with actual quantized operations
 * after QAT training is complete.
 *
 * @param qat_model Model prepared with prepare_qat()
 * @return Quantized model ready for deployment
 */
auto convert_qat(std::shared_ptr<nn::Module> qat_model) -> std::shared_ptr<nn::Module>;

/**
 * @brief Calibrate model for static quantization.
 *
 * Runs forward passes on calibration data to collect activation statistics
 * for determining quantization parameters.
 *
 * @param model Model with observers attached
 * @param calibration_data Calibration dataset batches
 * @return Calibrated quantization parameters per layer
 */
auto calibrate(
    nn::Module& model,
    const std::vector<Tensor>& calibration_data
) -> std::unordered_map<std::string, nn::quantization::QuantizationParams>;

/**
 * @brief Fuse common layer patterns for optimized quantization.
 *
 * Fuses sequences like Conv2d-BatchNorm2d-ReLU into single quantized
 * operations for better performance.
 *
 * @param model Model to fuse
 * @return Model with fused operations
 *
 * @code
 * auto model = std::make_shared<ResNet50>();
 * auto fused_model = fuse_modules(model);
 * auto q_model = quantize_static(fused_model, calibrate_fn);
 * @endcode
 */
auto fuse_modules(std::shared_ptr<nn::Module> model) -> std::shared_ptr<nn::Module>;

/**
 * @brief Compare FP32 and quantized model accuracy.
 *
 * Utility function to measure accuracy degradation from quantization.
 *
 * @param fp32_model Original floating-point model
 * @param quantized_model Quantized model
 * @param test_data Test dataset
 * @return Accuracy metrics (FP32 accuracy, INT8 accuracy, degradation %)
 */
auto compare_accuracy(
    nn::Module& fp32_model,
    nn::Module& quantized_model,
    const std::vector<std::pair<Tensor, Tensor>>& test_data
) -> std::tuple<float, float, float>;

/**
 * @brief Measure quantization performance improvement.
 *
 * Benchmarks inference latency and memory usage for FP32 vs INT8 models.
 *
 * @param fp32_model Original model
 * @param quantized_model Quantized model
 * @param input_shape Input tensor shape for benchmarking
 * @param num_iterations Number of iterations for timing
 * @return Performance metrics (FP32 time, INT8 time, speedup, memory reduction)
 */
auto benchmark_quantization(
    nn::Module& fp32_model,
    nn::Module& quantized_model,
    const std::vector<int64_t>& input_shape,
    int num_iterations = 100
) -> std::tuple<float, float, float, float>;

/**
 * @brief Convert floating-point module to quantized module.
 *
 * Standalone function to convert individual modules or complete models
 * from floating-point to quantized INT8 representation.
 *
 * @param module Module to quantize
 * @param config Quantization configuration
 * @return Quantized module
 *
 * @code
 * auto linear = std::make_shared<nn::Linear>(128, 64);
 * auto q_linear = convert_to_quantized(linear, qconfig);
 * @endcode
 */
auto convert_to_quantized(
    const std::shared_ptr<nn::Module>& module,
    const nn::quantization::QConfig& config = nn::quantization::DefaultQConfigs::default_qconfig()
) -> std::shared_ptr<nn::Module>;

/**
 * @brief Convert quantized module back to floating-point.
 *
 * Reverses quantization by dequantizing INT8 weights and biases back
 * to FP32 representation. Useful for model analysis and debugging.
 *
 * @param quantized_module Quantized module to convert
 * @return Floating-point module
 *
 * @code
 * auto q_model = quantize_dynamic(model);
 * auto fp_model = convert_from_quantized(q_model);  // Back to float
 * @endcode
 */
auto convert_from_quantized(
    const std::shared_ptr<nn::Module>& quantized_module
) -> std::shared_ptr<nn::Module>;

} // namespace quantization
} // namespace tenzor
