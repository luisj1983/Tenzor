#pragma once

#include <memory>
#include <vector>
#include "../core/tensor.hpp"
#include "variable.hpp"

namespace tenzor {

// Base class for autograd functions
class Function : public std::enable_shared_from_this<Function> {
public:
    virtual ~Function() = default;

    // Forward pass (called automatically)
    virtual auto forward(std::vector<Variable> inputs) -> std::vector<Variable> = 0;

    // Backward pass (gradient computation)
    virtual auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> = 0;

    // Input/output tracking
    auto set_next_functions(std::vector<std::shared_ptr<Function>> funcs) -> void;
    auto next_functions() const -> const std::vector<std::shared_ptr<Function>>&;

    // Saved tensor management
    auto num_saved_tensors() const -> size_t { return saved_tensors_.size(); }

protected:
    // Save tensors for backward pass
    auto save_for_backward(std::vector<Tensor> tensors) -> void;
    auto saved_tensors() const -> const std::vector<Tensor>&;

    std::vector<Tensor> saved_tensors_;
    std::vector<std::shared_ptr<Function>> next_functions_;
};

// Common autograd functions

class AddBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};

class SubBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};

class MulBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};

class MatMulBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};

class ReLUBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};

} // namespace tenzor
