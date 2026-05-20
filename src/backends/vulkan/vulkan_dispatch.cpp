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

// Legacy string-keyed dispatch removed (audit Phase C). Production dispatch
// is OpId-based via DispatchTableRegistry::get_table(Device::Type::Vulkan).

} // namespace tenzor
