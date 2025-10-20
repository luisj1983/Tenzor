/**
 * @file qconfig.cpp
 * @brief Implementation of quantization configuration classes
 */

#include "tenzor/nn/quantization/qconfig.hpp"

namespace tenzor {
namespace nn {
namespace quantization {

// ============================================================================
// QConfig
// ============================================================================

QConfig::QConfig(
    std::function<std::unique_ptr<Observer>()> weight_observer_factory,
    std::function<std::unique_ptr<Observer>()> activation_observer_factory,
    QuantDType weight_dtype,
    QuantDType activation_dtype,
    QuantizationScheme weight_scheme,
    QuantizationScheme activation_scheme
) : weight_observer_factory_(std::move(weight_observer_factory)),
    activation_observer_factory_(std::move(activation_observer_factory)),
    weight_dtype_(weight_dtype),
    activation_dtype_(activation_dtype),
    weight_scheme_(weight_scheme),
    activation_scheme_(activation_scheme) {}

// ============================================================================
// DefaultQConfigs
// ============================================================================

auto DefaultQConfigs::default_qconfig() -> QConfig {
    return QConfig(
        // Weight observer: per-channel symmetric with min-max
        []() { return make_observer(QuantizationScheme::PerChannelSymmetric, false, 0); },
        // Activation observer: per-tensor symmetric with moving average
        []() { return std::make_unique<MovingAverageMinMaxObserver>(0.9f, false, 0); },
        QuantDType::INT8,
        QuantDType::INT8,
        QuantizationScheme::PerChannelSymmetric,
        QuantizationScheme::PerTensorSymmetric
    );
}

auto DefaultQConfigs::high_accuracy_qconfig() -> QConfig {
    return QConfig(
        // Weight observer: per-channel symmetric with histogram
        []() { return make_observer(QuantizationScheme::PerChannelSymmetric, true, 0); },
        // Activation observer: per-tensor asymmetric with histogram
        []() { return make_observer(QuantizationScheme::PerTensorAsymmetric, true, 0); },
        QuantDType::INT8,
        QuantDType::INT8,
        QuantizationScheme::PerChannelSymmetric,
        QuantizationScheme::PerTensorAsymmetric
    );
}

auto DefaultQConfigs::fast_qconfig() -> QConfig {
    return QConfig(
        // Weight observer: per-channel symmetric with min-max
        []() { return make_observer(QuantizationScheme::PerChannelSymmetric, false, 0); },
        // Activation observer: per-tensor symmetric with min-max
        []() { return make_observer(QuantizationScheme::PerTensorSymmetric, false, 0); },
        QuantDType::INT8,
        QuantDType::INT8,
        QuantizationScheme::PerChannelSymmetric,
        QuantizationScheme::PerTensorSymmetric
    );
}

auto DefaultQConfigs::qat_qconfig() -> QConfig {
    return QConfig(
        // Weight observer: per-channel symmetric with moving average
        []() { return std::make_unique<MovingAverageMinMaxObserver>(0.95f, true, 0); },
        // Activation observer: per-tensor symmetric with moving average
        []() { return std::make_unique<MovingAverageMinMaxObserver>(0.9f, false, 0); },
        QuantDType::INT8,
        QuantDType::INT8,
        QuantizationScheme::PerChannelSymmetric,
        QuantizationScheme::PerTensorSymmetric
    );
}

auto DefaultQConfigs::per_channel_asymmetric_qconfig() -> QConfig {
    return QConfig(
        // Weight observer: per-channel asymmetric
        []() { return make_observer(QuantizationScheme::PerChannelAsymmetric, false, 0); },
        // Activation observer: per-tensor asymmetric
        []() { return make_observer(QuantizationScheme::PerTensorAsymmetric, false, 0); },
        QuantDType::INT8,
        QuantDType::INT8,
        QuantizationScheme::PerChannelAsymmetric,
        QuantizationScheme::PerTensorAsymmetric
    );
}

auto DefaultQConfigs::uint8_activation_qconfig() -> QConfig {
    return QConfig(
        // Weight observer: per-channel symmetric INT8
        []() { return make_observer(QuantizationScheme::PerChannelSymmetric, false, 0); },
        // Activation observer: per-tensor asymmetric UINT8
        []() { return make_observer(QuantizationScheme::PerTensorAsymmetric, false, 0); },
        QuantDType::INT8,
        QuantDType::UINT8,
        QuantizationScheme::PerChannelSymmetric,
        QuantizationScheme::PerTensorAsymmetric
    );
}

// ============================================================================
// QConfigMapping
// ============================================================================

auto QConfigMapping::set_layer_qconfig(const std::string& layer_name, const QConfig& qconfig)
    -> void {
    layer_qconfigs_[layer_name] = std::make_unique<QConfig>(qconfig);
}

auto QConfigMapping::set_type_qconfig(const std::string& layer_type, const QConfig& qconfig)
    -> void {
    type_qconfigs_[layer_type] = std::make_unique<QConfig>(qconfig);
}

auto QConfigMapping::get_qconfig(const std::string& layer_name,
                                const std::string& layer_type) const
    -> const QConfig* {
    // Check if layer is disabled
    if (disabled_layers_.count(layer_name) || disabled_types_.count(layer_type)) {
        return nullptr;
    }

    // Layer-specific config takes precedence
    auto layer_it = layer_qconfigs_.find(layer_name);
    if (layer_it != layer_qconfigs_.end()) {
        return layer_it->second.get();
    }

    // Type-specific config
    auto type_it = type_qconfigs_.find(layer_type);
    if (type_it != type_qconfigs_.end()) {
        return type_it->second.get();
    }

    // Global default
    return global_qconfig_.get();
}

auto QConfigMapping::is_quantized(const std::string& layer_name,
                                  const std::string& layer_type) const -> bool {
    return get_qconfig(layer_name, layer_type) != nullptr;
}

auto QConfigMapping::disable_layer(const std::string& layer_name) -> void {
    disabled_layers_.insert(layer_name);
}

auto QConfigMapping::disable_type(const std::string& layer_type) -> void {
    disabled_types_.insert(layer_type);
}

// ============================================================================
// QuantizationStrategy
// ============================================================================

QuantizationStrategy::QuantizationStrategy(
    QConfigMapping qconfig_mapping,
    BackendConfig backend_config
) : qconfig_mapping_(std::move(qconfig_mapping)),
    backend_config_(std::move(backend_config)) {}

// ============================================================================
// QuantizationStrategyBuilder
// ============================================================================

auto QuantizationStrategyBuilder::set_global_qconfig(const QConfig& qconfig)
    -> QuantizationStrategyBuilder& {
    qconfig_mapping_.set_global(qconfig);
    return *this;
}

auto QuantizationStrategyBuilder::set_layer_qconfig(
    const std::string& layer_name,
    const QConfig& qconfig
) -> QuantizationStrategyBuilder& {
    qconfig_mapping_.set_layer_qconfig(layer_name, qconfig);
    return *this;
}

auto QuantizationStrategyBuilder::set_type_qconfig(
    const std::string& layer_type,
    const QConfig& qconfig
) -> QuantizationStrategyBuilder& {
    qconfig_mapping_.set_type_qconfig(layer_type, qconfig);
    return *this;
}

auto QuantizationStrategyBuilder::disable_layer(const std::string& layer_name)
    -> QuantizationStrategyBuilder& {
    qconfig_mapping_.disable_layer(layer_name);
    return *this;
}

auto QuantizationStrategyBuilder::disable_type(const std::string& layer_type)
    -> QuantizationStrategyBuilder& {
    qconfig_mapping_.disable_type(layer_type);
    return *this;
}

auto QuantizationStrategyBuilder::set_backend(QuantizationBackend backend)
    -> QuantizationStrategyBuilder& {
    backend_config_.backend = backend;
    return *this;
}

auto QuantizationStrategyBuilder::set_calibration_batches(int num_batches)
    -> QuantizationStrategyBuilder& {
    backend_config_.num_calibration_batches = num_batches;
    return *this;
}

auto QuantizationStrategyBuilder::enable_operation_fusion(bool enable)
    -> QuantizationStrategyBuilder& {
    backend_config_.fuse_operations = enable;
    return *this;
}

auto QuantizationStrategyBuilder::build() -> QuantizationStrategy {
    return QuantizationStrategy(std::move(qconfig_mapping_), std::move(backend_config_));
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
