#pragma once

#include <memory>
#include <optional>
#include "../core/tensor.hpp"

namespace tenzor {

// Forward declarations
class Function;

// Gradient-enabled tensor wrapper
class Variable {
public:
    // Construction
    Variable() = default;
    explicit Variable(Tensor data, bool requires_grad = false);

    // Access underlying tensor
    auto tensor() const -> const Tensor&;
    auto tensor() -> Tensor&;

    // Gradient access
    auto grad() const -> const std::optional<Tensor>&;
    auto grad() -> std::optional<Tensor>&;
    auto has_grad() const -> bool;

    // Gradient computation
    auto backward(std::optional<Tensor> gradient = std::nullopt) -> void;

    // Gradient management
    auto zero_grad() -> void;
    auto detach() -> Variable;
    auto requires_grad() const -> bool;
    auto set_requires_grad(bool requires_grad) -> void;

    // Autograd context
    auto set_grad_fn(std::shared_ptr<Function> fn) -> void;
    auto grad_fn() const -> std::shared_ptr<Function>;

    // Tensor operations (forward to underlying tensor)
    auto shape() const -> std::span<const int64_t>;
    auto dtype() const -> DType;
    auto device() const -> const Device&;

    // Arithmetic operators (return Variables with grad tracking)
    auto operator+(const Variable& other) const -> Variable;
    auto operator-(const Variable& other) const -> Variable;
    auto operator*(const Variable& other) const -> Variable;
    auto operator/(const Variable& other) const -> Variable;

private:
    Tensor data_;
    std::optional<Tensor> grad_;
    std::shared_ptr<Function> grad_fn_;
    bool requires_grad_{false};

    friend class Function;
    friend class BackwardEngine;
};

// RAII guard for disabling gradient computation
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_state_;
};

// Global gradient state
auto is_grad_enabled() -> bool;
auto set_grad_enabled(bool enabled) -> void;

} // namespace tenzor
