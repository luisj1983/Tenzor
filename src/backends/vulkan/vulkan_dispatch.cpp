/**
 * @file vulkan_dispatch.cpp
 * @brief Vulkan backend string dispatch() method - thin wrapper over OpId dispatch table
 *
 * The VulkanBackend::dispatch(string, ...) pure virtual override now delegates
 * directly to the O(1) OpId-based dispatch table, matching all other backends.
 * All actual kernel registrations live in vulkan_kernel_registry.cpp.
 */

#include "vulkan_helpers.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tenzor {

auto VulkanBackend::dispatch(const std::string& op_name,
                            std::span<const Tensor> inputs,
                            const OpAttributes& attrs) -> std::vector<Tensor> {
    // Check for device lost state before submitting work
    auto device_id = inputs.empty() ? 0 : inputs[0].device().index;
    if (is_device_lost(device_id)) {
        if (!try_reset_device(device_id)) {
            throw std::runtime_error(
                "VulkanBackend::dispatch: device " + std::to_string(device_id) +
                " is lost and recovery failed for operation '" + op_name + "'");
        }
    }

    // Cache string→OpId lookups to avoid repeated linear scans.
    // Thread-local, bounded by OP_COUNT (~530 entries), no locking needed.
    thread_local std::unordered_map<std::string, OpId> op_cache;
    OpId op;
    auto it = op_cache.find(op_name);
    if (it != op_cache.end()) {
        op = it->second;
    } else {
        op = string_to_op_id(op_name);
        if (op == OpId::OP_COUNT) {
            throw std::runtime_error("VulkanBackend::dispatch: unknown operation '" + op_name + "'");
        }
        op_cache[op_name] = op;
    }
    auto& table = DispatchTableRegistry::get_table(Device::Type::Vulkan);
    return table.dispatch(op, inputs, attrs);
}

} // namespace tenzor
