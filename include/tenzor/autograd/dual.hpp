/**
 * @file dual.hpp
 * @brief Dual numbers for forward-mode automatic differentiation
 *
 * Provides DualTensor class that holds a primal value and tangent vector,
 * enabling forward-mode AD (Jacobian-vector products).
 */

#pragma once

#include "../core/tensor.hpp"

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

} // namespace tenzor
