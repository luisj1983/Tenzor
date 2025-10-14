#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/creation.hpp"
#include <atomic>
#include <iostream>

namespace tenzor {

// Global gradient state
static std::atomic<bool> grad_enabled{true};

Variable::Variable(Tensor data, bool requires_grad)
    : data_(std::move(data)), requires_grad_(requires_grad) {}

auto Variable::tensor() const -> const Tensor& {
    return data_;
}

auto Variable::tensor() -> Tensor& {
    return data_;
}

auto Variable::grad() const -> const std::optional<Tensor>& {
    return grad_;
}

auto Variable::grad() -> std::optional<Tensor>& {
    return grad_;
}

auto Variable::has_grad() const -> bool {
    return grad_.has_value();
}

auto Variable::backward(std::optional<Tensor> gradient) -> void {
    backward_engine().execute(*this, gradient);
}

auto Variable::zero_grad() -> void {
    if (grad_.has_value()) {
        grad_ = zeros_like(grad_.value());
    }
}

auto Variable::detach() -> Variable {
    return Variable(data_.detach(), false);
}

auto Variable::requires_grad() const -> bool {
    return requires_grad_;
}

auto Variable::set_requires_grad(bool requires_grad) -> void {
    requires_grad_ = requires_grad;
}

auto Variable::is_leaf() const -> bool {
    return !grad_fn_;
}

auto Variable::set_grad_fn(std::shared_ptr<Function> fn) -> void {
    grad_fn_ = std::move(fn);
}

auto Variable::grad_fn() const -> std::shared_ptr<Function> {
    return grad_fn_;
}

auto Variable::shape() const -> std::span<const int64_t> {
    return data_.shape();
}

auto Variable::dtype() const -> DType {
    return data_.dtype();
}

auto Variable::device() const -> const Device& {
    return data_.device();
}

// NoGradGuard implementation
NoGradGuard::NoGradGuard() : prev_state_(is_grad_enabled()) {
    set_grad_enabled(false);
}

NoGradGuard::~NoGradGuard() {
    set_grad_enabled(prev_state_);
}

// Global functions
auto is_grad_enabled() -> bool {
    return grad_enabled.load();
}

auto set_grad_enabled(bool enabled) -> void {
    grad_enabled.store(enabled);
}

// Arithmetic operators
auto Variable::operator+(const Variable& other) const -> Variable {
    auto grad_fn = std::make_shared<AddBackward>();

    // Set up backward graph - MUST maintain index correspondence with input_grads!
    // Use nullptr for leaf variables to preserve indices
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(grad_fn_);        // nullptr if this is leaf
    next_funcs.push_back(other.grad_fn_);  // nullptr if other is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    grad_fn->set_input_variables({const_cast<Variable*>(this), const_cast<Variable*>(&other)});

    // Save input shapes for broadcasting-aware backward pass
    grad_fn->input_shape_a_ = std::vector<int64_t>(data_.shape().begin(), data_.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(other.data_.shape().begin(), other.data_.shape().end());

    // Compute result
    auto result = data_ + other.data_;
    Variable output(result, requires_grad_ || other.requires_grad_);

    if (is_grad_enabled() && (requires_grad_ || other.requires_grad_)) {
        output.set_grad_fn(grad_fn);
    }

    return output;
}

auto Variable::operator-(const Variable& other) const -> Variable {
    auto grad_fn = std::make_shared<SubBackward>();

    // Set up backward graph - MUST maintain index correspondence with input_grads!
    // Use nullptr for leaf variables to preserve indices
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(grad_fn_);        // nullptr if this is leaf
    next_funcs.push_back(other.grad_fn_);  // nullptr if other is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    grad_fn->set_input_variables({const_cast<Variable*>(this), const_cast<Variable*>(&other)});

    // Save input shapes for broadcasting-aware backward pass
    grad_fn->input_shape_a_ = std::vector<int64_t>(data_.shape().begin(), data_.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(other.data_.shape().begin(), other.data_.shape().end());

    // Compute result
    auto result = data_ - other.data_;
    Variable output(result, requires_grad_ || other.requires_grad_);

    if (is_grad_enabled() && (requires_grad_ || other.requires_grad_)) {
        output.set_grad_fn(grad_fn);
    }

    return output;
}

auto Variable::operator*(const Variable& other) const -> Variable {
    auto grad_fn = std::make_shared<MulBackward>();

    // Set up backward graph - MUST maintain index correspondence with input_grads!
    // Use nullptr for leaf variables to preserve indices
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(grad_fn_);        // nullptr if this is leaf
    next_funcs.push_back(other.grad_fn_);  // nullptr if other is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    grad_fn->set_input_variables({const_cast<Variable*>(this), const_cast<Variable*>(&other)});

    // Save tensors AND input shapes for backward - clone to preserve values
    grad_fn->saved_tensors_ = {data_.clone(), other.data_.clone()};
    grad_fn->input_shape_a_ = std::vector<int64_t>(data_.shape().begin(), data_.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(other.data_.shape().begin(), other.data_.shape().end());

    // Compute result
    auto result = data_ * other.data_;
    Variable output(result, requires_grad_ || other.requires_grad_);

    if (is_grad_enabled() && (requires_grad_ || other.requires_grad_)) {
        output.set_grad_fn(grad_fn);
    }

    return output;
}

auto Variable::operator/(const Variable& other) const -> Variable {
    auto grad_fn = std::make_shared<DivBackward>();

    // Set up backward graph - MUST maintain index correspondence with input_grads!
    // Use nullptr for leaf variables to preserve indices
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(grad_fn_);        // nullptr if this is leaf
    next_funcs.push_back(other.grad_fn_);  // nullptr if other is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    grad_fn->set_input_variables({const_cast<Variable*>(this), const_cast<Variable*>(&other)});

    // Save tensors for backward - clone to preserve values
    grad_fn->saved_tensors_ = {data_.clone(), other.data_.clone()};

    // Compute result
    auto result = data_ / other.data_;
    Variable output(result, requires_grad_ || other.requires_grad_);

    if (is_grad_enabled() && (requires_grad_ || other.requires_grad_)) {
        output.set_grad_fn(grad_fn);
    }

    return output;
}

} // namespace tenzor
