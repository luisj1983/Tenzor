/**
 * @file variable.hpp
 * @brief Automatic differentiation wrapper for tensors
 *
 * Provides Variable class that wraps tensors with gradient tracking
 * and computation graph building for automatic differentiation.
 */

#pragma once

#include <memory>
#include <optional>
#include "../core/tensor.hpp"

namespace tenzor {

// Forward declarations
class Function;

/**
 * @brief Gradient-enabled tensor wrapper for automatic differentiation.
 *
 * Variable wraps a Tensor and tracks gradient information for automatic
 * differentiation. It builds a computation graph by recording operations
 * and can compute gradients via backpropagation.
 *
 * Key features:
 * - Automatic gradient computation via backward()
 * - Computation graph building with Function objects
 * - Leaf/non-leaf variable distinction
 * - Gradient accumulation for leaf variables
 *
 * @code
 * // Create variables that require gradients
 * Variable x(Tensor({2, 3}, DType::Float32, Device::cpu()), true);
 * Variable y(Tensor({2, 3}, DType::Float32, Device::cpu()), true);
 *
 * // Perform operations (builds computation graph)
 * Variable z = x + y;
 * Variable loss = z.sum();
 *
 * // Compute gradients
 * loss.backward();
 *
 * // Access gradients
 * Tensor x_grad = *x.grad();  // dLoss/dx
 * @endcode
 */
class Variable {
public:
    /**
     * @brief Default constructor creating empty variable.
     */
    Variable() = default;

    /**
     * @brief Construct variable from tensor.
     *
     * @param data Underlying tensor data
     * @param requires_grad Whether to track gradients (default: false)
     *
     * @code
     * Tensor t({3, 4}, DType::Float32, Device::cpu());
     * Variable v(t, true);  // Enable gradient tracking
     * @endcode
     */
    explicit Variable(Tensor data, bool requires_grad = false);

    // ============================================================================
    // Tensor Access
    // ============================================================================

    /**
     * @brief Get const reference to underlying tensor.
     *
     * @return Const reference to tensor data
     */
    auto tensor() const -> const Tensor&;

    /**
     * @brief Get mutable reference to underlying tensor.
     *
     * @return Mutable reference to tensor data
     */
    auto tensor() -> Tensor&;

    // ============================================================================
    // Gradient Access
    // ============================================================================

    /**
     * @brief Get const reference to gradient.
     *
     * @return Optional tensor containing gradient (nullopt if no gradient)
     */
    auto grad() const -> const std::optional<Tensor>&;

    /**
     * @brief Get mutable reference to gradient.
     *
     * @return Optional tensor containing gradient
     */
    auto grad() -> std::optional<Tensor>&;

    /**
     * @brief Check if variable has gradient.
     *
     * @return true if gradient has been computed
     */
    auto has_grad() const -> bool;

    // ============================================================================
    // Gradient Computation
    // ============================================================================

    /**
     * @brief Compute gradients via backpropagation.
     *
     * Computes gradient of this variable with respect to leaf variables
     * by traversing the computation graph backwards. For non-scalar outputs,
     * a gradient tensor must be provided.
     *
     * @param gradient Optional gradient tensor (required for non-scalar outputs)
     * @throws std::runtime_error if gradient is required but not provided
     *
     * @code
     * Variable x(Tensor({3}, DType::Float32, Device::cpu()), true);
     * Variable y = x * 2.0f;
     * Variable loss = y.sum();  // Scalar output
     *
     * loss.backward();  // No gradient needed for scalar
     * // x.grad() now contains gradient
     * @endcode
     */
    auto backward(std::optional<Tensor> gradient = std::nullopt) -> void;

    // ============================================================================
    // Gradient Management
    // ============================================================================

    /**
     * @brief Zero out gradient.
     *
     * Sets gradient to zero or removes it. Call before each backward pass
     * to prevent gradient accumulation.
     *
     * @code
     * for (int epoch = 0; epoch < 10; ++epoch) {
     *     x.zero_grad();  // Clear previous gradients
     *     auto loss = compute_loss(x);
     *     loss.backward();
     * }
     * @endcode
     */
    auto zero_grad() -> void;

    /**
     * @brief Detach variable from computation graph.
     *
     * Creates a new variable with the same data but no gradient history.
     * Useful when you want to use a value without backpropagating through it.
     *
     * @return New detached variable
     *
     * @code
     * Variable x_detached = x.detach();  // No gradients flow through x_detached
     * @endcode
     */
    auto detach() -> Variable;

    /**
     * @brief Check if variable requires gradient.
     *
     * @return true if gradients should be computed for this variable
     */
    auto requires_grad() const -> bool;

    /**
     * @brief Set whether variable requires gradient.
     *
     * @param requires_grad New gradient requirement state
     */
    auto set_requires_grad(bool requires_grad) -> void;

    /**
     * @brief Check if variable is a leaf node.
     *
     * Leaf nodes are created directly by the user (not from operations).
     * Gradients accumulate in leaf nodes during backward().
     *
     * @return true if variable is a leaf node
     */
    auto is_leaf() const -> bool;

    // ============================================================================
    // Autograd Context
    // ============================================================================

    /**
     * @brief Set gradient function (internal use).
     *
     * Sets the function that created this variable. Used internally
     * to build the computation graph.
     *
     * @param fn Function that produced this variable
     */
    auto set_grad_fn(std::shared_ptr<Function> fn) -> void;

    /**
     * @brief Get gradient function.
     *
     * Returns the function that created this variable in the forward pass.
     *
     * @return Shared pointer to gradient function (nullptr for leaf variables)
     */
    auto grad_fn() const -> std::shared_ptr<Function>;

    // ============================================================================
    // Tensor Properties
    // ============================================================================

    /**
     * @brief Get shape of underlying tensor.
     *
     * @return Span of dimension sizes
     */
    auto shape() const -> std::span<const int64_t>;

    /**
     * @brief Get data type of underlying tensor.
     *
     * @return DType enumeration value
     */
    auto dtype() const -> DType;

    /**
     * @brief Get device of underlying tensor.
     *
     * @return Device reference
     */
    auto device() const -> const Device&;

    // ============================================================================
    // Arithmetic Operators
    // ============================================================================

    /**
     * @brief Add two variables with gradient tracking.
     *
     * @param other Variable to add
     * @return New variable with gradient function
     */
    auto operator+(const Variable& other) const -> Variable;

    /**
     * @brief Subtract two variables with gradient tracking.
     *
     * @param other Variable to subtract
     * @return New variable with gradient function
     */
    auto operator-(const Variable& other) const -> Variable;

    /**
     * @brief Multiply two variables with gradient tracking.
     *
     * @param other Variable to multiply
     * @return New variable with gradient function
     */
    auto operator*(const Variable& other) const -> Variable;

    /**
     * @brief Divide two variables with gradient tracking.
     *
     * @param other Variable to divide by
     * @return New variable with gradient function
     */
    auto operator/(const Variable& other) const -> Variable;

private:
    Tensor data_;
    std::optional<Tensor> grad_;
    std::shared_ptr<Function> grad_fn_;
    bool requires_grad_{false};

    friend class Function;
    friend class BackwardEngine;
};

/**
 * @brief RAII guard for temporarily disabling gradient computation.
 *
 * Creates a context where gradient computation is disabled.
 * Restores previous state when destroyed. Useful for inference
 * or when computing values that shouldn't be part of the gradient graph.
 *
 * @code
 * Variable x(tensor, true);
 *
 * {
 *     NoGradGuard guard;
 *     Variable y = x * 2.0f;  // No gradient tracking
 *     // y.backward() would fail
 * }
 *
 * Variable z = x * 3.0f;  // Gradient tracking restored
 * @endcode
 */
class NoGradGuard {
public:
    /**
     * @brief Construct guard and disable gradients.
     *
     * Saves current gradient state and disables gradient computation.
     */
    NoGradGuard();

    /**
     * @brief Restore previous gradient state.
     */
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_state_;  ///< Previous gradient enabled state
};

/**
 * @brief Check if gradient computation is globally enabled.
 *
 * @return true if gradients are being computed
 */
auto is_grad_enabled() -> bool;

/**
 * @brief Set global gradient computation state.
 *
 * @param enabled Whether to enable gradient computation
 *
 * @code
 * set_grad_enabled(false);  // Disable gradients globally
 * // ... inference code ...
 * set_grad_enabled(true);   // Re-enable gradients
 * @endcode
 */
auto set_grad_enabled(bool enabled) -> void;

} // namespace tenzor
