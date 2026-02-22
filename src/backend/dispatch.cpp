#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/loader.hpp"

namespace tenzor {

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
