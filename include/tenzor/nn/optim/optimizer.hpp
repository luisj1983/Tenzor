#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
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

    // Serialization (to be implemented by derived classes)
    virtual auto state_dict() const -> std::unordered_map<std::string, Tensor> = 0;
    virtual auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void = 0;

    // File I/O convenience methods
    auto save_state(const std::string& path) const -> void;
    auto load_state(const std::string& path) -> void;

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
