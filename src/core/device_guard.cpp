/**
 * @file device_guard.cpp
 * @brief Implementation of RAII device context management
 */

#include "tenzor/core/device_guard.hpp"
#include "tenzor/backend/loader.hpp"

#include <array>

namespace tenzor {

namespace detail {

auto current_device_index(Device::Type type) -> int32_t& {
    static thread_local std::array<int32_t, static_cast<size_t>(Device::Type::COUNT)> indices{};
    return indices[static_cast<size_t>(type)];
}

void switch_device(Device::Type type, int32_t device_id) {
    // CPU is always device 0, no switching needed
    if (type == Device::Type::CPU) {
        return;
    }

    auto& current = current_device_index(type);
    if (current == device_id) {
        return;  // Already on the target device
    }

    // Delegate to backend for the actual hardware device switch
    auto* backend = try_get_backend(type);
    if (backend) {
        backend->set_device(device_id);
    }

    current = device_id;
}

} // namespace detail

// --- DeviceGuard ---

DeviceGuard::DeviceGuard(Device device)
    : device_(device)
    , prev_index_(detail::current_device_index(device.type))
{
    detail::switch_device(device.type, device.index);
}

DeviceGuard::~DeviceGuard() {
    // Restore previous device. Swallow exceptions in destructor.
    try {
        detail::switch_device(device_.type, prev_index_);
    } catch (...) {
        // Cannot throw from destructor. The thread-local state is still
        // updated even if the hardware call fails, so subsequent guards
        // will attempt the correct restore.
    }
}

// --- OptionalDeviceGuard ---

OptionalDeviceGuard::OptionalDeviceGuard(std::optional<Device> device) {
    if (device.has_value() && device->type != Device::Type::CPU) {
        guard_.emplace(device.value());
    }
}

OptionalDeviceGuard::OptionalDeviceGuard(Device device) {
    if (device.type != Device::Type::CPU) {
        guard_.emplace(device);
    }
}

} // namespace tenzor
