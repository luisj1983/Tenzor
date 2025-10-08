#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/engine.hpp"
#include <atomic>

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
    grad_ = std::nullopt;
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
    auto result = data_ + other.data_;
    return Variable(result, requires_grad_ || other.requires_grad_);
}

auto Variable::operator-(const Variable& other) const -> Variable {
    auto result = data_ - other.data_;
    return Variable(result, requires_grad_ || other.requires_grad_);
}

auto Variable::operator*(const Variable& other) const -> Variable {
    auto result = data_ * other.data_;
    return Variable(result, requires_grad_ || other.requires_grad_);
}

auto Variable::operator/(const Variable& other) const -> Variable {
    auto result = data_ / other.data_;
    return Variable(result, requires_grad_ || other.requires_grad_);
}

} // namespace tenzor
