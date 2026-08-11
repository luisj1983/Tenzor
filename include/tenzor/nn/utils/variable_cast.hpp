#pragma once

#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/function.hpp>
#include <memory>
#include <vector>

namespace tenzor::nn {

// Forward declaration: TypeCastBackward::backward_with_variables() recurses
// into variable_cast() (defined below) to build a properly graph-connected
// second-order cast node -- see the root-cause note on that method.
inline auto variable_cast(const Variable& input, DType target_dtype) -> Variable;

/// Backward function for autograd-aware dtype casting.
/// Gradients are cast back to the original (input) dtype so that parameter
/// gradients match the parameter dtype (e.g., Float16 weight gets Float16 grad).
class TypeCastBackward : public Function {
public:
    DType original_dtype_ = DType::Float32;

    TypeCastBackward() = default;

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("TypeCastBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad = grad_outputs[0];
        if (grad.dtype() != original_dtype_) {
            return {grad.to(original_dtype_)};
        }
        return {grad};
    }

    // Root cause (found via systematic debugging of
    // test_higher_order_nn_multidtype's LSTM/GRU_InnerLoop_Gradient/
    // cpu_BFloat16 failures): a dtype cast's SECOND derivative is
    // structurally zero, but that does not mean the cast should be a
    // higher-order "stub" (disconnection point) -- the cast's FIRST
    // derivative (needed to keep differentiating whatever comes after it
    // in the graph, e.g. sigmoid/tanh/matmul inside LSTMCell) is a
    // perfectly well-defined linear map, and a create_graph=true backward
    // must keep that chain connected. The previous implementation
    // returned `Variable(grad.tensor().to(dtype), grad.requires_grad())`
    // with NO grad_fn -- a graph-severing op masquerading as one that
    // "supports higher order" -- and ALSO set is_higher_order_stub()=true,
    // which makes the engine (src/autograd/engine.cpp) hard-throw in the
    // default HigherOrderGradMode::Error before the (silently broken)
    // result above would even have been used. Recursing into
    // variable_cast() below builds a genuine autograd-tracked cast node
    // for the second-order graph, so this op is NOT a stub and should not
    // be marked as one.
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        auto& grad = grad_outputs[0];
        if (grad.dtype() != original_dtype_) {
            return {variable_cast(grad, original_dtype_)};
        }
        return {grad};
    }

    // Genuinely supports higher-order gradients (see backward_with_variables
    // above) -- do NOT also mark is_higher_order_stub()=true; that flag is
    // for ops whose backward_with_variables is the generic disconnecting
    // TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB() passthrough, which this is
    // not. Leaving is_higher_order_stub() at the Function base class's
    // default (false) is intentional and load-bearing, not an omission.
    auto supports_higher_order() const -> bool override { return true; }
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
