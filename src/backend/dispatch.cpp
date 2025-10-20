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

    // Get the appropriate backend and dispatch to it
    Backend* backend = get_backend(inputs);
    if (!backend) {
        throw TenzorException("No backend available for tensors");
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
