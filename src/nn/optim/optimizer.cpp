#include "tenzor/nn/optim/optimizer.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/device.hpp"
#include <cstring>

namespace tenzor::optim {

Optimizer::Optimizer(std::vector<std::shared_ptr<Variable>> params)
    : parameters_(std::move(params)) {}

auto Optimizer::zero_grad() -> void {
    for (auto& param : parameters_) {
        if (param && param->has_grad()) {
            auto& grad = param->grad().value();

            // CPU path: use direct memset for maximum performance
            if (grad.device().type == Device::Type::CPU) {
                std::memset(grad.data_ptr(), 0, grad.numel() * dtype_size(grad.dtype()));
            } else {
                // GPU path: create new zero tensor
                // TODO: Add in-place zero via cudaMemsetAsync for better performance
                param->grad() = zeros_like(grad);
            }
        }
    }
}

auto Optimizer::parameters() const -> const std::vector<std::shared_ptr<Variable>>& {
    return parameters_;
}

auto Optimizer::save_state(const std::string& path) const -> void {
    auto state = state_dict();
    nn::Serializer::save(state, path);
}

auto Optimizer::load_state(const std::string& path) -> void {
    auto state = nn::Serializer::load(path);
    load_state_dict(state);
}

} // namespace tenzor::optim
