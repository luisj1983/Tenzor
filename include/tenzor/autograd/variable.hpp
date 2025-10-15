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
#include <functional>
#include "../core/tensor.hpp"

namespace tenzor {

// Forward declarations
class Function;

/**
 * @brief Implementation class for Variable's handle pattern.
 *
 * VariableImpl holds all state for a Variable, enabling handle semantics
 * with shallow copy behavior. Multiple Variable handles can reference the
 * same VariableImpl for zero-copy operations.
 *
 * This follows the PImpl (Pointer to Implementation) pattern, matching
 * Tensor's architecture where Tensor is a handle to TensorImpl.
 *
 * @note Thread safety: VariableImpl is NOT thread-safe by default.
 *       External synchronization required for concurrent access to mutable
 *       state (grad_, hooks_). Read-only operations on data_ and grad_fn_
 *       are safe due to shared_ptr semantics.
 */
struct VariableImpl {
    /**
     * @brief Construct VariableImpl with tensor data.
     *
     * @param data Underlying tensor (moved into impl)
     * @param requires_grad Whether to track gradients
     */
    explicit VariableImpl(Tensor data, bool requires_grad = false)
        : data_(std::move(data)),
          requires_grad_(requires_grad) {}

    // Default copy/move constructors for standard semantics
    VariableImpl(const VariableImpl&) = default;
    VariableImpl(VariableImpl&&) noexcept = default;
    VariableImpl& operator=(const VariableImpl&) = default;
    VariableImpl& operator=(VariableImpl&&) noexcept = default;
    ~VariableImpl() = default;

    // === State Members (moved from Variable) ===

    /// Underlying tensor data (handle type, already thread-safe)
    Tensor data_;

    /// Accumulated gradient tensor (requires synchronization for writes)
    std::optional<Tensor> grad_;

    /// Gradient function that created this variable (thread-safe for reads)
    std::shared_ptr<Function> grad_fn_;

    /// Whether gradient tracking is enabled (consider atomic if modified concurrently)
    bool requires_grad_{false};

    /// Whether to retain gradient for non-leaf variables (consider atomic if modified concurrently)
    bool retain_grad_{false};

    /// Backward hooks (requires synchronization for modifications)
    std::vector<std::function<Tensor(const Tensor&)>> hooks_;

    // Note: No mutex included in initial implementation (Option B from design doc).
    // Users responsible for external synchronization if accessing Variable handles
    // from multiple threads. This matches PyTorch's behavior.
};

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

    /**
     * @brief Set gradient tensor directly.
     *
     * Sets the gradient tensor for this variable. Used internally
     * for gradient checkpointing and custom backward passes.
     *
     * @param gradient Tensor to set as gradient
     */
    auto set_grad(Tensor gradient) -> void;

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
     * @param retain_graph If true, keep computation graph for multiple backward passes
     * @throws std::runtime_error if gradient is required but not provided
     *
     * @code
     * Variable x(Tensor({3}, DType::Float32, Device::cpu()), true);
     * Variable y = x * 2.0f;
     * Variable loss = y.sum();  // Scalar output
     *
     * loss.backward(std::nullopt, false);  // Normal backward, clears graph
     * // x.grad() now contains gradient
     *
     * // For multiple backward passes:
     * loss.backward(std::nullopt, true);  // First backward, keep graph
     * loss.backward(std::nullopt, false); // Second backward, clear graph
     * @endcode
     */
    auto backward(std::optional<Tensor> gradient = std::nullopt, bool retain_graph = false) -> void;

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

    /**
     * @brief Register a backward hook function.
     *
     * Registers a callable that will be called during backward pass after
     * the gradient has been computed. The hook receives the gradient as input
     * and can modify or inspect it.
     *
     * @param hook Function that takes gradient tensor and returns (optionally modified) gradient
     * @return Hook handle (currently unused, for future hook removal)
     *
     * @code
     * Variable x(tensor, true);
     * x.register_hook([](const Tensor& grad) {
     *     std::cout << "Gradient norm: " << grad.norm().item<float>() << std::endl;
     *     return grad;  // Return unmodified gradient
     * });
     * @endcode
     */
    auto register_hook(std::function<Tensor(const Tensor&)> hook) -> size_t;

    /**
     * @brief Enable gradient retention for non-leaf variables.
     *
     * By default, only leaf variables retain gradients after backward().
     * Call this to retain gradients for intermediate (non-leaf) variables.
     *
     * @code
     * Variable x(tensor, true);
     * Variable y = x * 2.0f;  // Non-leaf
     * y.retain_grad();  // Keep gradient after backward()
     * Variable loss = y.sum();
     * loss.backward();
     * // y.grad() is now available (normally would be cleared)
     * @endcode
     */
    auto retain_grad() -> void;

    /**
     * @brief Check if variable retains gradient.
     *
     * @return true if gradients should be retained (even for non-leaf variables)
     */
    auto retains_grad() const -> bool;

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
     * @brief Check if variable is initialized with valid data.
     *
     * @return true if variable has been constructed with a tensor
     */
    auto is_initialized() const -> bool;

    /**
     * @brief Boolean conversion operator for validity checking.
     *
     * @return true if variable is initialized
     */
    explicit operator bool() const;

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
    std::shared_ptr<VariableImpl> impl_;

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
