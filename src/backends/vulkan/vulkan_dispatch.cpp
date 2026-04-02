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
#include <vector>

namespace tenzor {

auto VulkanBackend::dispatch(const std::string& op_name,
                            std::span<const Tensor> inputs,
                            const OpAttributes& attrs) -> std::vector<Tensor> {
    auto op = string_to_op_id(op_name);
    if (op == OpId::OP_COUNT) {
        throw std::runtime_error("VulkanBackend::dispatch: unknown operation '" + op_name + "'");
    }
    auto& table = DispatchTableRegistry::get_table(Device::Type::Vulkan);
    return table.dispatch(op, inputs, attrs);
}

} // namespace tenzor
