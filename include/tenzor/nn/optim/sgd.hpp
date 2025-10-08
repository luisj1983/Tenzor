#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

// Stochastic Gradient Descent optimizer
class SGD : public Optimizer {
public:
    SGD(std::vector<Variable*> params,
        double lr,
        double momentum = 0.0,
        double dampening = 0.0,
        double weight_decay = 0.0,
        bool nesterov = false);

    auto step() -> void override;

    // Learning rate management
    auto set_lr(double lr) -> void;
    auto get_lr() const -> double;

private:
    double lr_;
    double momentum_;
    double dampening_;
    double weight_decay_;
    bool nesterov_;

    std::vector<Tensor> velocity_buffers_;

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
