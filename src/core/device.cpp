#include "tenzor/core/device.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/utils/error.hpp"
#include <stdexcept>

namespace tenzor {

// Parse device index from string suffix and validate it's non-negative
static auto parse_device_index(std::string_view str, size_t prefix_len) -> int {
    auto suffix = std::string(str.substr(prefix_len));
    int idx;
    try {
        idx = std::stoi(suffix);
    } catch (const std::invalid_argument&) {
        throw DeviceException("Invalid device index in '" + std::string(str) +
                              "' — expected integer after ':'");
    } catch (const std::out_of_range&) {
        throw DeviceException("Device index out of range in '" + std::string(str) + "'");
    }
    if (idx < 0) {
        throw DeviceException("Device index must be non-negative, got: " + std::to_string(idx));
    }
    return idx;
}

auto Device::from_string(std::string_view str) -> Device {
    if (str == "cpu") {
        return Device::cpu();
    }

    if (str.starts_with("cuda:")) {
        return Device::cuda(parse_device_index(str, 5));
    }

    if (str.starts_with("rocm:")) {
        return Device::rocm(parse_device_index(str, 5));
    }

    if (str.starts_with("oneapi:")) {
        return Device::oneapi(parse_device_index(str, 7));
    }

    if (str.starts_with("vulkan:")) {
        return Device::vulkan(parse_device_index(str, 7));
    }

    if (str.starts_with("metal:")) {
        return Device::metal(parse_device_index(str, 6));
    }

    if (str.starts_with("webgpu:")) {
        return Device::webgpu(parse_device_index(str, 7));
    }

    throw DeviceException("Invalid device string: " + std::string(str));
}

auto Device::synchronize() const -> void {
    // Get the backend for this device type
    auto* backend = backend_registry().get_backend(type);
    if (!backend) {
        throw std::runtime_error("Backend not available for device: " + to_string());
    }

    // Call the backend's synchronize method
    backend->synchronize(index);
}

} // namespace tenzor
