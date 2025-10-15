#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/creation.hpp"
#include <atomic>
#include <iostream>
#include <vector>

namespace tenzor {

// Global gradient state
static std::atomic<bool> grad_enabled{true};

Variable::Variable(Tensor data, bool requires_grad)
    : impl_(std::make_shared<VariableImpl>(std::move(data), requires_grad)) {
}

auto Variable::is_initialized() const -> bool {
    return impl_ != nullptr;
}

Variable::operator bool() const {
    return is_initialized();
}

auto Variable::tensor() const -> const Tensor& {
    if (!impl_) {
        throw std::runtime_error("Cannot access tensor of uninitialized Variable");
    }
    return impl_->data_;
}

auto Variable::tensor() -> Tensor& {
    if (!impl_) {
        throw std::runtime_error("Cannot access tensor of uninitialized Variable");
    }
    return impl_->data_;
}

auto Variable::grad() const -> const std::optional<Tensor>& {
    return impl_->grad_;
}

auto Variable::grad() -> std::optional<Tensor>& {
    return impl_->grad_;
}

auto Variable::has_grad() const -> bool {
    return impl_ && impl_->grad_.has_value();
}

auto Variable::set_grad(Tensor gradient) -> void {
    impl_->grad_ = std::move(gradient);
}

auto Variable::backward(std::optional<Tensor> gradient, bool retain_graph) -> void {
    backward_engine().execute(*this, gradient, retain_graph);
}

auto Variable::register_hook(std::function<Tensor(const Tensor&)> hook) -> size_t {
    impl_->hooks_.push_back(std::move(hook));
    return impl_->hooks_.size() - 1;
}

auto Variable::retain_grad() -> void {
    impl_->retain_grad_ = true;
}

auto Variable::retains_grad() const -> bool {
    return impl_ && impl_->retain_grad_;
}

auto Variable::zero_grad() -> void {
    if (impl_ && impl_->grad_.has_value()) {
        impl_->grad_ = zeros_like(impl_->grad_.value());
    }
}

auto Variable::detach() -> Variable {
    return Variable(impl_->data_.detach(), false);
}

auto Variable::requires_grad() const -> bool {
    return impl_ && impl_->requires_grad_;
}

auto Variable::set_requires_grad(bool requires_grad) -> void {
    impl_->requires_grad_ = requires_grad;
}

auto Variable::is_leaf() const -> bool {
    return !impl_ || !impl_->grad_fn_;
}

auto Variable::set_grad_fn(std::shared_ptr<Function> fn) -> void {
    impl_->grad_fn_ = std::move(fn);
}

auto Variable::grad_fn() const -> std::shared_ptr<Function> {
    return impl_ ? impl_->grad_fn_ : nullptr;
}

auto Variable::shape() const -> std::span<const int64_t> {
    return impl_->data_.shape();
}

auto Variable::dtype() const -> DType {
    return impl_->data_.dtype();
}

auto Variable::device() const -> const Device& {
    return impl_->data_.device();
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
    next_funcs.push_back(impl_->grad_fn_);        // nullptr if this is leaf
    next_funcs.push_back(other.impl_->grad_fn_);  // nullptr if other is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    // Store Variables by value - their impl_ shared_ptr keeps data alive
    grad_fn->set_input_variables({*this, other});

    // Save input shapes for broadcasting-aware backward pass
    grad_fn->input_shape_a_ = std::vector<int64_t>(impl_->data_.shape().begin(), impl_->data_.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(other.impl_->data_.shape().begin(), other.impl_->data_.shape().end());

    // Compute result
    auto result = impl_->data_ + other.impl_->data_;
    Variable output(result, impl_->requires_grad_ || other.impl_->requires_grad_);

    if (is_grad_enabled() && (impl_->requires_grad_ || other.impl_->requires_grad_)) {
        output.set_grad_fn(grad_fn);
    }

    return output;
}

auto Variable::operator-(const Variable& other) const -> Variable {
    auto grad_fn = std::make_shared<SubBackward>();

    // Set up backward graph - MUST maintain index correspondence with input_grads!
    // Use nullptr for leaf variables to preserve indices
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(impl_->grad_fn_);        // nullptr if this is leaf
    next_funcs.push_back(other.impl_->grad_fn_);  // nullptr if other is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    // Store Variables by value - their impl_ shared_ptr keeps data alive
    grad_fn->set_input_variables({*this, other});

    // Save input shapes for broadcasting-aware backward pass
    grad_fn->input_shape_a_ = std::vector<int64_t>(impl_->data_.shape().begin(), impl_->data_.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(other.impl_->data_.shape().begin(), other.impl_->data_.shape().end());

    // Compute result
    auto result = impl_->data_ - other.impl_->data_;
    Variable output(result, impl_->requires_grad_ || other.impl_->requires_grad_);

    if (is_grad_enabled() && (impl_->requires_grad_ || other.impl_->requires_grad_)) {
        output.set_grad_fn(grad_fn);
    }

    return output;
}

auto Variable::operator*(const Variable& other) const -> Variable {
    auto grad_fn = std::make_shared<MulBackward>();

    // Set up backward graph - MUST maintain index correspondence with input_grads!
    // Use nullptr for leaf variables to preserve indices
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(impl_->grad_fn_);        // nullptr if this is leaf
    next_funcs.push_back(other.impl_->grad_fn_);  // nullptr if other is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    // Store Variables by value - their impl_ shared_ptr keeps data alive
    grad_fn->set_input_variables({*this, other});

    // Save tensors AND input shapes for backward - clone to preserve values
    grad_fn->saved_tensors_ = {impl_->data_.clone(), other.impl_->data_.clone()};
    grad_fn->input_shape_a_ = std::vector<int64_t>(impl_->data_.shape().begin(), impl_->data_.shape().end());
    grad_fn->input_shape_b_ = std::vector<int64_t>(other.impl_->data_.shape().begin(), other.impl_->data_.shape().end());

    // Compute result
    auto result = impl_->data_ * other.impl_->data_;
    Variable output(result, impl_->requires_grad_ || other.impl_->requires_grad_);

    if (is_grad_enabled() && (impl_->requires_grad_ || other.impl_->requires_grad_)) {
        output.set_grad_fn(grad_fn);
    }

    return output;
}

auto Variable::operator/(const Variable& other) const -> Variable {
    auto grad_fn = std::make_shared<DivBackward>();

    // Set up backward graph - MUST maintain index correspondence with input_grads!
    // Use nullptr for leaf variables to preserve indices
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(impl_->grad_fn_);        // nullptr if this is leaf
    next_funcs.push_back(other.impl_->grad_fn_);  // nullptr if other is leaf
    grad_fn->set_next_functions(next_funcs);

    // Track input variables for gradient accumulation
    // Store Variables by value - their impl_ shared_ptr keeps data alive
    grad_fn->set_input_variables({*this, other});

    // Save tensors for backward - clone to preserve values
    grad_fn->saved_tensors_ = {impl_->data_.clone(), other.impl_->data_.clone()};

    // Compute result
    auto result = impl_->data_ / other.impl_->data_;
    Variable output(result, impl_->requires_grad_ || other.impl_->requires_grad_);

    if (is_grad_enabled() && (impl_->requires_grad_ || other.impl_->requires_grad_)) {
        output.set_grad_fn(grad_fn);
    }

    return output;
}

} // namespace tenzor
