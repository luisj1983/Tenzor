#include "tenzor/nn/optim/optimizer.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor::optim {

Optimizer::Optimizer(std::vector<std::shared_ptr<Variable>> params)
    : parameters_(std::move(params)) {}

auto Optimizer::zero_grad() -> void {
    for (auto& param : parameters_) {
        if (param && param->has_grad()) {
            // Zero gradient in-place (keeps tensor allocated for performance)
            // This is different from param->zero_grad() which resets the optional
            auto& grad = param->grad().value();
            param->grad() = zeros_like(grad);
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
