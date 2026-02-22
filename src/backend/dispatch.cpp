#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/registry.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/utils/error.hpp"

namespace tenzor {

auto Dispatcher::dispatch(const std::string& op_name,
                         std::span<const Tensor> inputs,
                         const OpAttributes& attrs) -> std::vector<Tensor> {
    // Check device compatibility
    if (!check_device_compatibility(inputs)) {
        throw DeviceException("All input tensors must be on the same device");
    }

    // Route to the appropriate backend's dispatch method directly
    auto device_type = inputs.empty() ? Device::Type::CPU : inputs[0].device().type;
    auto* backend = backend_registry().get_backend(device_type);
    if (!backend) {
        throw std::runtime_error("No backend registered for device type");
    }
    return backend->dispatch(op_name, inputs, attrs);
}

auto Dispatcher::get_backend(std::span<const Tensor> tensors) -> Backend* {
    if (tensors.empty()) {
        return nullptr;
    }

    auto device_type = tensors[0].device().type;
    return backend_registry().get_backend(device_type);
}

auto Dispatcher::check_device_compatibility(std::span<const Tensor> tensors) -> bool {
    if (tensors.empty()) {
        return true;
    }

    auto first_device = tensors[0].device();

    for (const auto& tensor : tensors) {
        if (tensor.device() != first_device) {
            return false;
        }
    }

    return true;
}

} // namespace tenzor
