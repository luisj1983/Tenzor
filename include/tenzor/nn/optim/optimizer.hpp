#pragma once

#include <vector>
#include <memory>
#include "../../autograd/variable.hpp"

namespace tenzor {
namespace optim {

// Base optimizer class
class Optimizer {
public:
    virtual ~Optimizer() = default;

    // Update parameters
    virtual auto step() -> void = 0;

    // Zero gradients
    auto zero_grad() -> void;

    // Get parameters
    auto parameters() const -> const std::vector<Variable*>&;

protected:
    explicit Optimizer(std::vector<Variable*> params);

    std::vector<Variable*> parameters_;
};

// Parameter group for different learning rates
struct ParamGroup {
    std::vector<Variable*> params;
    double lr;
    double weight_decay{0.0};
};

} // namespace optim
} // namespace tenzor
