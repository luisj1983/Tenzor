#pragma once

#include <memory>
#include <vector>
#include "../core/tensor.hpp"
#include "variable.hpp"

namespace tenzor {

// Forward declaration
class Variable;

// Base class for autograd functions
class Function : public std::enable_shared_from_this<Function> {
    friend class Variable;  // Allow Variable to access saved_tensors_

public:
    virtual ~Function() = default;

    // Forward pass (called automatically)
    virtual auto forward(std::vector<Variable> inputs) -> std::vector<Variable> = 0;

    // Backward pass (gradient computation)
    virtual auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> = 0;

    // Input/output tracking
    auto set_next_functions(std::vector<std::shared_ptr<Function>> funcs) -> void;
    auto next_functions() const -> const std::vector<std::shared_ptr<Function>>&;

    // Track input variables for gradient accumulation
    auto set_input_variables(std::vector<Variable*> inputs) -> void;
    auto input_variables() const -> const std::vector<Variable*>&;

    // Saved tensor management
    auto num_saved_tensors() const -> size_t { return saved_tensors_.size(); }

    // Save tensors for backward pass (public for wrapper functions)
    auto save_for_backward(std::vector<Tensor> tensors) -> void;
    auto saved_tensors() const -> const std::vector<Tensor>&;

protected:

    std::vector<Tensor> saved_tensors_;
    std::vector<std::shared_ptr<Function>> next_functions_;
    std::vector<Variable*> input_variables_;  // Track leaf variables for gradient accumulation
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

class DivBackward : public Function {
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

class SumBackward : public Function {
public:
    SumBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

class MeanBackward : public Function {
public:
    MeanBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

class LogBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};

class ExpBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};

class NegBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};

class LogSoftmaxBackward : public Function {
public:
    LogSoftmaxBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
private:
    int64_t dim_;
};

} // namespace tenzor
