/**
 * @file dual.hpp
 * @brief Dual numbers for forward-mode automatic differentiation
 *
 * Provides DualTensor class that holds a primal value and tangent vector,
 * enabling forward-mode AD (Jacobian-vector products).
 */

#pragma once

#include <optional>

#include "../core/tensor.hpp"
#include "variable.hpp"

namespace tenzor {

/**
 * @brief A dual number for forward-mode AD: holds a primal value and tangent vector.
 *
 * DualTensor represents the pair (primal, tangent) used in forward-mode
 * automatic differentiation. Given a function f and input x with tangent v,
 * the output DualTensor contains (f(x), J_f(x) * v) where J_f is the Jacobian.
 */
class DualTensor {
public:
    /**
     * @brief Construct DualTensor with explicit primal and tangent.
     * @param primal The primal (value) tensor
     * @param tangent The tangent (derivative direction) tensor
     */
    DualTensor(Tensor primal, Tensor tangent);

    /**
     * @brief Construct DualTensor with zero tangent.
     * @param primal The primal (value) tensor; tangent is zeros_like(primal)
     */
    explicit DualTensor(Tensor primal);

    auto primal() const -> const Tensor& { return primal_; }
    auto tangent() const -> const Tensor& { return tangent_; }

    auto primal() -> Tensor& { return primal_; }
    auto tangent() -> Tensor& { return tangent_; }

private:
    Tensor primal_;
    Tensor tangent_;
};

/// Thread-local flag indicating dual-mode computation is active
auto is_dual_mode() -> bool;

/// Set dual-mode state (prefer DualModeGuard for exception-safe usage)
void set_dual_mode(bool enabled);

/// Public alias for `set_dual_mode`. Provided so user code reads more
/// intuitively: `tenzor::enable_dual_mode(true);` ... `enable_dual_mode(false);`.
/// Prefer `DualModeGuard` over either form for exception safety.
inline void enable_dual_mode(bool enabled) { set_dual_mode(enabled); }

/**
 * @brief RAII guard for entering/exiting dual (forward-mode AD) mode.
 *
 * While active, operations that support forward-mode AD will propagate
 * tangent vectors through the computation.
 */
class DualModeGuard {
public:
    DualModeGuard();
    ~DualModeGuard();
    DualModeGuard(const DualModeGuard&) = delete;
    DualModeGuard& operator=(const DualModeGuard&) = delete;
private:
    bool prev_state_;
};

/**
 * @brief Variable-level analog of DualTensor: a (primal Variable, tangent Tensor) pair.
 *
 * Carries an autograd Variable (the primal) together with a forward-mode tangent
 * Tensor. Used by the `dual_apply` helper and by `jvp()` when invoked with
 * `JvpMode::Dual` to thread tangents through a computation without first building
 * and walking a backward graph.
 *
 * The tangent must have the same shape and dtype as `primal().tensor()`. If
 * constructed from a primal alone, the tangent is `zeros_like(primal.tensor())`.
 */
class DualVariable {
public:
    /// Construct with explicit primal Variable and tangent Tensor.
    DualVariable(Variable primal, Tensor tangent);

    /// Construct with zero tangent (`zeros_like(primal.tensor())`).
    explicit DualVariable(Variable primal);

    [[nodiscard]] auto primal() const -> const Variable& { return primal_; }
    [[nodiscard]] auto tangent() const -> const Tensor&  { return tangent_; }

    auto primal()  -> Variable& { return primal_; }
    auto tangent() -> Tensor&   { return tangent_; }

    /// Convenience: shape/dtype/device forwarders to the primal tensor.
    [[nodiscard]] auto shape()  const { return primal_.shape();  }
    [[nodiscard]] auto dtype()  const { return primal_.dtype();  }
    [[nodiscard]] auto device() const { return primal_.device(); }

private:
    Variable primal_;
    Tensor   tangent_;
};

} // namespace tenzor
