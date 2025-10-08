#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <span>
#include <shared_mutex>
#include "../core/tensor.hpp"
#include "../core/device.hpp"
#include "backend.hpp"

namespace tenzor {

// Kernel function type
using KernelFunction = std::function<
    std::vector<Tensor>(std::span<const Tensor>, const OpAttributes&)
>;

// Operation registry for kernel dispatch
class OperationRegistry {
public:
    OperationRegistry() = default;

    // Register kernel for specific device type
    auto register_kernel(std::string_view op_name,
                        Device::Type device_type,
                        KernelFunction kernel) -> void;

    // Dispatch to appropriate kernel
    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor>;

    // Check if operation is registered
    auto has_kernel(std::string_view op_name, Device::Type device_type) const -> bool;

    // List all registered operations
    auto registered_operations() const -> std::vector<std::string>;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<
        std::string,
        std::unordered_map<Device::Type, KernelFunction>
    > kernels_;
};

// Global operation registry
auto operation_registry() -> OperationRegistry&;

} // namespace tenzor
