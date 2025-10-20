/**
 * @file qconfig.hpp
 * @brief Quantization configuration for neural network layers
 *
 * Provides configuration classes for specifying quantization settings
 * for different layers in a neural network, including weight and
 * activation quantization schemes.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "quantize.hpp"
#include "observer.hpp"
#include "fake_quantize.hpp"

namespace tenzor {
namespace nn {
namespace quantization {

/**
 * @brief Quantization configuration for a layer.
 *
 * Specifies how weights and activations should be quantized for a
 * particular layer. Different layers may use different configurations
 * based on their sensitivity to quantization.
 */
class QConfig {
public:
    /**
     * @brief Construct quantization configuration.
     *
     * @param weight_observer_factory Factory for weight observer
     * @param activation_observer_factory Factory for activation observer
     * @param weight_dtype Weight quantization data type
     * @param activation_dtype Activation quantization data type
     * @param weight_scheme Weight quantization scheme
     * @param activation_scheme Activation quantization scheme
     */
    QConfig(
        std::function<std::unique_ptr<Observer>()> weight_observer_factory,
        std::function<std::unique_ptr<Observer>()> activation_observer_factory,
        QuantDType weight_dtype = QuantDType::INT8,
        QuantDType activation_dtype = QuantDType::INT8,
        QuantizationScheme weight_scheme = QuantizationScheme::PerChannelSymmetric,
        QuantizationScheme activation_scheme = QuantizationScheme::PerTensorSymmetric
    );

    /**
     * @brief Create weight observer instance.
     */
    auto create_weight_observer() const -> std::unique_ptr<Observer> {
        return weight_observer_factory_();
    }

    /**
     * @brief Create activation observer instance.
     */
    auto create_activation_observer() const -> std::unique_ptr<Observer> {
        return activation_observer_factory_();
    }

    /**
     * @brief Get weight quantization data type.
     */
    auto weight_dtype() const -> QuantDType { return weight_dtype_; }

    /**
     * @brief Get activation quantization data type.
     */
    auto activation_dtype() const -> QuantDType { return activation_dtype_; }

    /**
     * @brief Get weight quantization scheme.
     */
    auto weight_scheme() const -> QuantizationScheme { return weight_scheme_; }

    /**
     * @brief Get activation quantization scheme.
     */
    auto activation_scheme() const -> QuantizationScheme { return activation_scheme_; }

private:
    std::function<std::unique_ptr<Observer>()> weight_observer_factory_;
    std::function<std::unique_ptr<Observer>()> activation_observer_factory_;
    QuantDType weight_dtype_;
    QuantDType activation_dtype_;
    QuantizationScheme weight_scheme_;
    QuantizationScheme activation_scheme_;
};

/**
 * @brief Default quantization configurations.
 *
 * Provides pre-configured QConfig instances for common use cases.
 */
class DefaultQConfigs {
public:
    /**
     * @brief Default INT8 configuration.
     *
     * - Weights: Per-channel symmetric INT8 with min-max observer
     * - Activations: Per-tensor symmetric INT8 with moving average min-max
     *
     * Good general-purpose configuration for most models.
     */
    static auto default_qconfig() -> QConfig;

    /**
     * @brief High accuracy configuration.
     *
     * - Weights: Per-channel symmetric INT8 with histogram observer
     * - Activations: Per-tensor asymmetric INT8 with histogram observer
     *
     * More robust to outliers, better accuracy but slower calibration.
     */
    static auto high_accuracy_qconfig() -> QConfig;

    /**
     * @brief Fast calibration configuration.
     *
     * - Weights: Per-channel symmetric INT8 with min-max observer
     * - Activations: Per-tensor symmetric INT8 with min-max observer
     *
     * Fastest calibration, may be less accurate with outliers.
     */
    static auto fast_qconfig() -> QConfig;

    /**
     * @brief QAT-friendly configuration.
     *
     * - Weights: Per-channel symmetric INT8 with moving average
     * - Activations: Per-tensor symmetric INT8 with moving average
     *
     * Smooth parameter updates suitable for quantization-aware training.
     */
    static auto qat_qconfig() -> QConfig;

    /**
     * @brief Per-channel asymmetric configuration.
     *
     * - Weights: Per-channel asymmetric INT8
     * - Activations: Per-tensor asymmetric INT8
     *
     * Maximum flexibility, may be needed for models with large weight ranges.
     */
    static auto per_channel_asymmetric_qconfig() -> QConfig;

    /**
     * @brief UINT8 configuration for activations.
     *
     * - Weights: Per-channel symmetric INT8
     * - Activations: Per-tensor asymmetric UINT8
     *
     * Common for models with ReLU activations (non-negative).
     */
    static auto uint8_activation_qconfig() -> QConfig;
};

/**
 * @brief Layer-specific quantization configuration.
 *
 * Maps layer names or types to their quantization configurations,
 * allowing fine-grained control over quantization for different
 * parts of the network.
 */
class QConfigMapping {
public:
    QConfigMapping() = default;

    /**
     * @brief Set global default quantization config.
     *
     * @param qconfig Configuration to use as default
     */
    auto set_global(const QConfig& qconfig) -> void {
        global_qconfig_ = std::make_unique<QConfig>(qconfig);
    }

    /**
     * @brief Set quantization config for specific layer.
     *
     * @param layer_name Name or pattern for layer
     * @param qconfig Configuration for this layer
     */
    auto set_layer_qconfig(const std::string& layer_name, const QConfig& qconfig) -> void;

    /**
     * @brief Set quantization config for layer type.
     *
     * @param layer_type Type name (e.g., "Linear", "Conv2d")
     * @param qconfig Configuration for this type
     */
    auto set_type_qconfig(const std::string& layer_type, const QConfig& qconfig) -> void;

    /**
     * @brief Get quantization config for layer.
     *
     * Returns layer-specific config if available, type-specific if available,
     * otherwise global default.
     *
     * @param layer_name Layer name
     * @param layer_type Layer type
     * @return Quantization configuration
     */
    auto get_qconfig(const std::string& layer_name,
                    const std::string& layer_type) const -> const QConfig*;

    /**
     * @brief Check if layer should be quantized.
     *
     * @param layer_name Layer name
     * @param layer_type Layer type
     * @return True if quantization is enabled for this layer
     */
    auto is_quantized(const std::string& layer_name,
                     const std::string& layer_type) const -> bool;

    /**
     * @brief Disable quantization for specific layer.
     *
     * @param layer_name Layer name to exclude from quantization
     */
    auto disable_layer(const std::string& layer_name) -> void;

    /**
     * @brief Disable quantization for layer type.
     *
     * @param layer_type Layer type to exclude
     */
    auto disable_type(const std::string& layer_type) -> void;

private:
    std::unique_ptr<QConfig> global_qconfig_;
    std::unordered_map<std::string, std::unique_ptr<QConfig>> layer_qconfigs_;
    std::unordered_map<std::string, std::unique_ptr<QConfig>> type_qconfigs_;
    std::unordered_set<std::string> disabled_layers_;
    std::unordered_set<std::string> disabled_types_;
};

/**
 * @brief Quantization backend configuration.
 *
 * Specifies target backend for quantized model execution and
 * backend-specific optimizations.
 */
enum class QuantizationBackend {
    Default,        ///< Default backend (CPU or CUDA)
    FBGEMM,         ///< Facebook GEMM for x86 CPU
    QNNPACK,        ///< Qualcomm NN Pack for ARM CPU
    OneDNN,         ///< Intel OneDNN for x86 CPU
    CUDAInt8,       ///< NVIDIA TensorCore INT8
    Custom          ///< Custom backend
};

/**
 * @brief Backend-specific quantization settings.
 */
struct BackendConfig {
    QuantizationBackend backend{QuantizationBackend::Default};
    bool use_int8_ops{true};              ///< Use int8 operations
    bool fuse_operations{true};           ///< Fuse quantized ops
    bool optimize_for_inference{true};    ///< Apply inference optimizations
    int num_calibration_batches{100};     ///< Number of calibration batches
};

/**
 * @brief Quantization strategy configuration.
 *
 * High-level configuration for quantization workflow, combining
 * QConfig, backend settings, and calibration parameters.
 */
class QuantizationStrategy {
public:
    /**
     * @brief Construct quantization strategy.
     *
     * @param qconfig_mapping Layer-specific quantization configs
     * @param backend_config Backend-specific settings
     */
    QuantizationStrategy(
        QConfigMapping qconfig_mapping,
        BackendConfig backend_config
    );

    /**
     * @brief Get QConfig mapping.
     */
    auto qconfig_mapping() const -> const QConfigMapping& { return qconfig_mapping_; }

    /**
     * @brief Get backend configuration.
     */
    auto backend_config() const -> const BackendConfig& { return backend_config_; }

    /**
     * @brief Set calibration data size.
     */
    auto set_calibration_batches(int num_batches) -> void {
        backend_config_.num_calibration_batches = num_batches;
    }

    /**
     * @brief Enable or disable operation fusion.
     */
    auto set_operation_fusion(bool enable) -> void {
        backend_config_.fuse_operations = enable;
    }

private:
    QConfigMapping qconfig_mapping_;
    BackendConfig backend_config_;
};

/**
 * @brief Quantization workflow builder.
 *
 * Fluent interface for configuring quantization strategies.
 *
 * @code
 * auto strategy = QuantizationStrategyBuilder()
 *     .set_global_qconfig(DefaultQConfigs::default_qconfig())
 *     .set_layer_qconfig("fc1", DefaultQConfigs::high_accuracy_qconfig())
 *     .set_backend(QuantizationBackend::FBGEMM)
 *     .set_calibration_batches(200)
 *     .build();
 * @endcode
 */
class QuantizationStrategyBuilder {
public:
    QuantizationStrategyBuilder() = default;

    auto set_global_qconfig(const QConfig& qconfig) -> QuantizationStrategyBuilder&;
    auto set_layer_qconfig(const std::string& layer_name, const QConfig& qconfig)
        -> QuantizationStrategyBuilder&;
    auto set_type_qconfig(const std::string& layer_type, const QConfig& qconfig)
        -> QuantizationStrategyBuilder&;
    auto disable_layer(const std::string& layer_name) -> QuantizationStrategyBuilder&;
    auto disable_type(const std::string& layer_type) -> QuantizationStrategyBuilder&;
    auto set_backend(QuantizationBackend backend) -> QuantizationStrategyBuilder&;
    auto set_calibration_batches(int num_batches) -> QuantizationStrategyBuilder&;
    auto enable_operation_fusion(bool enable = true) -> QuantizationStrategyBuilder&;

    auto build() -> QuantizationStrategy;

private:
    QConfigMapping qconfig_mapping_;
    BackendConfig backend_config_;
};

} // namespace quantization
} // namespace nn
} // namespace tenzor
