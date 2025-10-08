#pragma once

#include <span>
#include <vector>
#include "../core/tensor.hpp"
#include "backend.hpp"

namespace tenzor {

// Kernel dispatcher
class Dispatcher {
public:
    // Dispatch operation to appropriate backend
    static auto dispatch(const std::string& op_name,
                        std::span<const Tensor> inputs,
                        const OpAttributes& attrs = {}) -> std::vector<Tensor>;

    // Get appropriate backend for tensors
    static auto get_backend(std::span<const Tensor> tensors) -> Backend*;

    // Check device compatibility
    static auto check_device_compatibility(std::span<const Tensor> tensors) -> bool;
};

} // namespace tenzor
