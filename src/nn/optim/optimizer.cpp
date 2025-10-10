#include "tenzor/nn/optim/optimizer.hpp"
#include "tenzor/nn/serialize.hpp"

namespace tenzor::optim {

Optimizer::Optimizer(std::vector<Variable*> params)
    : parameters_(std::move(params)) {}

auto Optimizer::zero_grad() -> void {
    for (auto* param : parameters_) {
        param->zero_grad();
    }
}

auto Optimizer::parameters() const -> const std::vector<Variable*>& {
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
