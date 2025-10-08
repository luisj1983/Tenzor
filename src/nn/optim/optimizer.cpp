#include "tenzor/nn/optim/optimizer.hpp"

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

} // namespace tenzor::optim
