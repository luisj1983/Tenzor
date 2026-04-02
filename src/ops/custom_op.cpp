/**
 * @file custom_op.cpp
 * @brief Implementation of runtime custom operation registration
 */

#include "tenzor/ops/custom_op.hpp"
#include "tenzor/backend/dispatch_table.hpp"

#include <stdexcept>

namespace tenzor {

// ============================================================================
// CustomOpRegistry singleton
// ============================================================================

auto CustomOpRegistry::instance() -> CustomOpRegistry& {
    static CustomOpRegistry registry;
    return registry;
}

auto CustomOpRegistry::register_op(const std::string& name) -> CustomOpId {
    std::unique_lock lock(registry_mutex_);

    auto it = name_to_id_.find(name);
    if (it != name_to_id_.end()) {
        return CustomOpId(it->second);
    }

    uint32_t id = next_id_++;
    name_to_id_[name] = id;
    id_to_name_.push_back(name);
    return CustomOpId(id);
}

auto CustomOpRegistry::find_op(std::string_view name) const -> std::optional<CustomOpId> {
    std::shared_lock lock(registry_mutex_);

    auto it = name_to_id_.find(std::string(name));
    if (it != name_to_id_.end()) {
        return CustomOpId(it->second);
    }
    return std::nullopt;
}

auto CustomOpRegistry::op_name(CustomOpId id) const -> std::string_view {
    std::shared_lock lock(registry_mutex_);

    if (id.value < OP_COUNT) {
        return op_id_to_name(static_cast<OpId>(id.value));
    }

    uint32_t idx = id.value - OP_COUNT;
    if (idx < id_to_name_.size()) {
        return id_to_name_[idx];
    }
    return "";
}

void CustomOpRegistry::register_kernel(CustomOpId id, Device::Type device_type,
                                       CustomKernelFn kernel) {
    auto& dk = device_kernels_[static_cast<size_t>(device_type)];
    std::unique_lock lock(dk.mutex);
    dk.kernels[id.value] = std::move(kernel);
}

auto CustomOpRegistry::dispatch(CustomOpId id, Device::Type device_type,
                                std::span<const Tensor> inputs,
                                const OpAttributes& attrs) const -> Tensor {
    auto& dk = device_kernels_[static_cast<size_t>(device_type)];
    std::shared_lock lock(dk.mutex);

    auto it = dk.kernels.find(id.value);
    if (it == dk.kernels.end()) {
        std::string name(op_name(id));
        throw std::runtime_error(
            "No custom kernel registered for op '" + name +
            "' on device type " + std::to_string(static_cast<int>(device_type)));
    }

    return it->second(inputs, attrs);
}

bool CustomOpRegistry::has_kernel(CustomOpId id, Device::Type device_type) const {
    auto& dk = device_kernels_[static_cast<size_t>(device_type)];
    std::shared_lock lock(dk.mutex);
    return dk.kernels.contains(id.value);
}

// ============================================================================
// Public API
// ============================================================================

auto register_custom_op(const std::string& name,
                        Device::Type device_type,
                        CustomKernelFn kernel) -> CustomOpId {
    auto& registry = CustomOpRegistry::instance();
    auto id = registry.register_op(name);
    registry.register_kernel(id, device_type, std::move(kernel));
    return id;
}

auto register_custom_op(const std::string& name,
                        std::initializer_list<std::pair<Device::Type, CustomKernelFn>> kernels)
    -> CustomOpId {
    auto& registry = CustomOpRegistry::instance();
    auto id = registry.register_op(name);
    for (auto& [device_type, kernel] : kernels) {
        registry.register_kernel(id, device_type, std::move(kernel));
    }
    return id;
}

auto dispatch_custom_op(CustomOpId id,
                        std::span<const Tensor> inputs,
                        const OpAttributes& attrs) -> Tensor {
    if (inputs.empty()) {
        throw std::runtime_error("dispatch_custom_op: inputs must not be empty");
    }

    Device::Type device_type = inputs[0].device().type;
    return CustomOpRegistry::instance().dispatch(id, device_type, inputs, attrs);
}

} // namespace tenzor
