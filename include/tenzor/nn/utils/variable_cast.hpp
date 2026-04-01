#pragma once

#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/function.hpp>
#include <memory>
#include <vector>

namespace tenzor::nn {

/// Backward function for autograd-aware dtype casting.
/// Gradients are cast back to the original (input) dtype so that parameter
/// gradients match the parameter dtype (e.g., Float16 weight gets Float16 grad).
class TypeCastBackward : public Function {
public:
    DType original_dtype_ = DType::Float32;

    TypeCastBackward() = default;

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("TypeCastBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad = grad_outputs[0];
        if (grad.dtype() != original_dtype_) {
            return {grad.to(original_dtype_)};
        }
        return {grad};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        auto& grad = grad_outputs[0];
        if (grad.dtype() != original_dtype_) {
            return {Variable(grad.tensor().to(original_dtype_), grad.requires_grad())};
        }
        return {grad};
    }
};

/// Cast a Variable to a new dtype with proper autograd graph connectivity.
/// Unlike a raw `Variable(tensor.to(dtype), requires_grad)` + `set_grad_fn()`,
/// this creates a TypeCastBackward node that correctly converts gradients back
/// to the original parameter dtype during the backward pass.
inline auto variable_cast(const Variable& input, DType target_dtype) -> Variable {
    if (input.dtype() == target_dtype) {
        return input;
    }

    auto converted_tensor = input.tensor().to(target_dtype);
    Variable result(converted_tensor, input.requires_grad());

    if (input.requires_grad() && is_grad_enabled()) {
        auto grad_fn = std::make_shared<TypeCastBackward>();
        grad_fn->original_dtype_ = input.dtype();

        // Track input variable for gradient accumulation
        std::vector<Variable> input_vars = {input};
        grad_fn->set_input_variables(input_vars);

        // Connect to input's grad_fn
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        grad_fn->set_next_functions(next_funcs);

        result.set_grad_fn(grad_fn);
    }

    return result;
}

} // namespace tenzor::nn
