#include "tenzor/backend/registry.hpp"
#include "tenzor/utils/error.hpp"
#include <mutex>
#include <shared_mutex>

namespace tenzor {

auto OperationRegistry::register_kernel(std::string_view op_name,
                                       Device::Type device_type,
                                       KernelFunction kernel) -> void {
    std::unique_lock lock(mutex_);
    kernels_[std::string(op_name)][device_type] = std::move(kernel);
}

auto OperationRegistry::dispatch(const std::string& op_name,
                                std::span<const Tensor> inputs,
                                const OpAttributes& attrs) -> std::vector<Tensor> {
    std::shared_lock lock(mutex_);

    // Get device type from first input
    if (inputs.empty()) {
        throw TenzorException("Cannot dispatch operation with no inputs");
    }

    auto device_type = inputs[0].device().type;

    // Find kernel
    auto op_it = kernels_.find(op_name);
    if (op_it == kernels_.end()) {
        throw TenzorException("Operation not registered: " + op_name);
    }

    auto kernel_it = op_it->second.find(device_type);
    if (kernel_it == op_it->second.end()) {
        throw TenzorException("Operation not implemented for device: " + op_name);
    }

    // Execute kernel
    return kernel_it->second(inputs, attrs);
}

auto OperationRegistry::has_kernel(std::string_view op_name,
                                  Device::Type device_type) const -> bool {
    std::shared_lock lock(mutex_);

    auto op_it = kernels_.find(std::string(op_name));
    if (op_it == kernels_.end()) {
        return false;
    }

    return op_it->second.contains(device_type);
}

auto OperationRegistry::registered_operations() const -> std::vector<std::string> {
    std::shared_lock lock(mutex_);

    std::vector<std::string> ops;
    ops.reserve(kernels_.size());

    for (const auto& [name, _] : kernels_) {
        ops.push_back(name);
    }

    return ops;
}

// Global registry
auto operation_registry() -> OperationRegistry& {
    static OperationRegistry registry;
    return registry;
}

} // namespace tenzor
