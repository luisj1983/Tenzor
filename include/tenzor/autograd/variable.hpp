/**
 * @file variable.hpp
 * @brief Automatic differentiation wrapper for tensors
 *
 * Provides Variable class that wraps tensors with gradient tracking
 * and computation graph building for automatic differentiation.
 */

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <functional>
#include <map>
#include <mutex>
#include <shared_mutex>
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

    // Custom copy/move constructors needed because std::atomic is non-copyable
    VariableImpl(const VariableImpl& other)
        : data_(other.data_),
          grad_(other.grad_),
          grad_fn_(other.grad_fn_),
          requires_grad_(other.requires_grad_.load(std::memory_order_relaxed)),
          retain_grad_(other.retain_grad_.load(std::memory_order_relaxed)),
          hooks_(other.hooks_),
          next_hook_id_(other.next_hook_id_.load(std::memory_order_relaxed)),
          grad_mutex_(other.grad_mutex_),  // Share mutex — copies share grad_ storage
          thread_safe_(other.thread_safe_.load(std::memory_order_relaxed)) {}

    VariableImpl(VariableImpl&& other) noexcept
        : data_(std::move(other.data_)),
          grad_(std::move(other.grad_)),
          grad_fn_(std::move(other.grad_fn_)),
          requires_grad_(other.requires_grad_.load(std::memory_order_relaxed)),
          retain_grad_(other.retain_grad_.load(std::memory_order_relaxed)),
          hooks_(std::move(other.hooks_)),
          next_hook_id_(other.next_hook_id_.load(std::memory_order_relaxed)),
          grad_mutex_(std::move(other.grad_mutex_)),
          thread_safe_(other.thread_safe_.load(std::memory_order_relaxed)) {}

    VariableImpl& operator=(const VariableImpl& other) {
        if (this != &other) {
            // Lock both mutexes to prevent race during grad_mutex_ reassignment.
            // Use std::lock to avoid deadlock if two assignments happen concurrently.
            auto old_mutex = grad_mutex_;
            std::unique_lock<std::mutex> lock1(*old_mutex, std::defer_lock);
            std::unique_lock<std::mutex> lock2(*other.grad_mutex_, std::defer_lock);
            std::lock(lock1, lock2);

            data_ = other.data_;
            grad_ = other.grad_;
            grad_fn_ = other.grad_fn_;
            requires_grad_.store(other.requires_grad_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            retain_grad_.store(other.retain_grad_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hooks_ = other.hooks_;
            next_hook_id_.store(other.next_hook_id_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            grad_mutex_ = other.grad_mutex_;
            thread_safe_.store(other.thread_safe_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    VariableImpl& operator=(VariableImpl&& other) noexcept {
        if (this != &other) {
            // Lock old mutex before replacing it to prevent racing with
            // concurrent gradient accumulation on this Variable.
            auto old_mutex = grad_mutex_;
            std::lock_guard<std::mutex> lock(*old_mutex);

            data_ = std::move(other.data_);
            grad_ = std::move(other.grad_);
            grad_fn_ = std::move(other.grad_fn_);
            requires_grad_.store(other.requires_grad_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            retain_grad_.store(other.retain_grad_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hooks_ = std::move(other.hooks_);
            next_hook_id_.store(other.next_hook_id_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            grad_mutex_ = std::move(other.grad_mutex_);
            thread_safe_.store(other.thread_safe_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    ~VariableImpl() = default;

    // === State Members (moved from Variable) ===

    /// Underlying tensor data (handle type, already thread-safe)
    Tensor data_;

    /// Accumulated gradient tensor (requires synchronization for writes)
    std::optional<Tensor> grad_;

    /// Gradient function that created this variable (thread-safe for reads)
    std::shared_ptr<Function> grad_fn_;

    /// Whether gradient tracking is enabled (atomic for thread-safe concurrent backward traversal)
    std::atomic<bool> requires_grad_{false};

    /// Whether to retain gradient for non-leaf variables (atomic for thread-safe concurrent access)
    std::atomic<bool> retain_grad_{false};

    /// Backward hooks keyed by ID, ordered by registration order
    std::map<size_t, std::function<Tensor(const Tensor&)>> hooks_;

    /// Mutex protecting hooks_ for concurrent register_hook + backward
    mutable std::shared_mutex hooks_mutex_;

    /// Next hook ID for register_hook (atomic for thread-safe concurrent register_hook calls)
    std::atomic<size_t> next_hook_id_{0};

    /// Mutex protecting grad_ for concurrent gradient accumulation.
    /// Shared via shared_ptr so copies (which share grad_ via Tensor handle
    /// semantics) also share the mutex for correct thread safety.
    /// Only locked when thread_safe_ is true, so zero overhead for
    /// single-threaded code.
    std::shared_ptr<std::mutex> grad_mutex_ = std::make_shared<std::mutex>();

    /// Opt-in flag for thread-safe gradient access. When true, grad_mutex_
    /// is locked around gradient reads/writes. Enable via Variable::make_thread_safe().
    std::atomic<bool> thread_safe_{false};
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
 * @par Thread safety
 * All concurrent access to a Variable requires external synchronization.
 * While shared_ptr reference counting is atomic, the Tensor data, gradient
 * state, and computation graph are NOT. In particular:
 * - Forward/backward passes sharing Variables must run on the same thread
 * - Gradient accumulation is NOT atomic unless make_thread_safe() is called
 * - Hook registration/invocation is serialized via shared_mutex
 * - NoGradGuard is thread-local (does NOT propagate to spawned threads)
 * - The backward engine uses a per-thread singleton model
 * For concurrent training, use separate Variable instances per thread,
 * or call make_thread_safe() on shared parameters before concurrent
 * gradient accumulation (e.g., data-parallel training).
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
     * @brief Get mutable reference to gradient (internal use only).
     *
     * @warning This accessor bypasses thread-safe locking. Prefer set_grad()
     * for gradient mutation. Only use mutable_grad() when you need in-place
     * access to the underlying buffer (e.g., memset for zero_grad).
     *
     * @return Optional tensor containing gradient
     */
    auto mutable_grad() -> std::optional<Tensor>&;

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
     * @param create_graph If true, the backward pass itself is differentiable,
     *        enabling higher-order gradients (e.g., for WGAN-GP, MAML, Hessian computation).
     *        When true, gradient computations use Variable operations that build a new
     *        computation graph, so you can call backward() again on the resulting gradients.
     *        Implies retain_graph=true.
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
     *
     * // For higher-order gradients (double backward):
     * loss.backward(std::nullopt, false, true);  // create_graph=true
     * // x.grad() is now a Variable-backed gradient that can be differentiated again
     * @endcode
     */
    auto backward(std::optional<Tensor> gradient = std::nullopt,
                  bool retain_graph = false,
                  bool create_graph = false) -> void;

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
     * @brief Remove a previously registered backward hook.
     *
     * @param hook_id Handle returned by register_hook()
     * @return true if the hook was found and removed
     */
    auto unregister_hook(size_t hook_id) -> bool;

    /**
     * @brief Remove all registered backward hooks.
     *
     * Useful for cleanup in test fixtures and module teardown.
     */
    auto clear_hooks() -> void;

    /**
     * @brief Enable thread-safe gradient access for this variable.
     *
     * When enabled, gradient reads/writes are protected by a mutex,
     * allowing safe concurrent gradient accumulation from multiple
     * backward passes (e.g., data-parallel training).
     *
     * Zero overhead when not enabled — the mutex exists but is never locked.
     *
     * @code
     * Variable shared_param(tensor, true);
     * shared_param.make_thread_safe();
     * // Now safe to call backward() concurrently from multiple threads
     * @endcode
     */
    auto make_thread_safe() -> void;

    /**
     * @brief Check if thread-safe gradient access is enabled.
     *
     * @return true if make_thread_safe() has been called
     */
    auto is_thread_safe() const -> bool;

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
    // Shape Transformation Methods
    // ============================================================================

    /** @brief Reshape variable. */
    auto reshape(std::vector<int64_t> shape) const -> Variable;

    /** @brief Transpose two dimensions. */
    auto transpose(int64_t dim0, int64_t dim1) const -> Variable;

    /** @brief Permute dimensions. */
    auto permute(std::vector<int64_t> dims) const -> Variable;

    /** @brief Matrix multiplication. */
    auto matmul(const Variable& other) const -> Variable;

    /** @brief Remove dimensions of size 1. */
    auto squeeze(int64_t dim) const -> Variable;

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

    /**
     * @brief Add scalar to variable with gradient tracking.
     *
     * @param scalar Scalar value to add
     * @return New variable with gradient function
     */
    auto operator+(float scalar) const -> Variable;
    auto operator+(double scalar) const -> Variable;

    /**
     * @brief Multiply variable by scalar with gradient tracking.
     *
     * @param scalar Scalar value to multiply
     * @return New variable with gradient function
     */
    auto operator*(float scalar) const -> Variable;
    auto operator*(double scalar) const -> Variable;

    /** @brief Subtract scalar from variable. */
    auto operator-(float scalar) const -> Variable;
    auto operator-(double scalar) const -> Variable;

    /** @brief Divide variable by scalar. */
    auto operator/(float scalar) const -> Variable;
    auto operator/(double scalar) const -> Variable;

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
 *
 * @warning Thread-local — does NOT propagate to spawned threads.
 * Each thread starts with gradients enabled regardless of parent thread state.
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
 * @brief RAII guard for inference mode.
 *
 * Stronger than NoGradGuard: disables gradient computation AND
 * skips version counter increments for in-place ops, enabling
 * faster inference execution.
 *
 * @code
 * {
 *     InferenceModeGuard guard;
 *     // All ops here skip grad tracking and version counting
 *     auto y = model.forward(x);
 * }
 * @endcode
 *
 * @warning Thread-local — does NOT propagate to spawned threads.
 */
class InferenceModeGuard {
public:
    InferenceModeGuard();
    ~InferenceModeGuard();

    InferenceModeGuard(const InferenceModeGuard&) = delete;
    InferenceModeGuard& operator=(const InferenceModeGuard&) = delete;

private:
    bool prev_grad_state_;
    bool prev_inference_state_;
};

/**
 * @brief Check if inference mode is currently active.
 *
 * When inference mode is active, gradient tracking and version counter
 * increments are both disabled.
 *
 * @return true if inference mode is enabled
 */
auto is_inference_mode_enabled() -> bool;

/**
 * @brief Check if gradient computation is globally enabled.
 *
 * @return true if gradients are being computed
 */
auto is_grad_enabled() -> bool;

/**
 * @brief Check if higher-order gradient graph creation is active.
 *
 * When true, backward pass operations should use Variable operations
 * instead of raw Tensor operations, so the backward computation itself
 * is tracked by autograd and can be differentiated again.
 *
 * @return true if create_graph mode is active (set during backward with create_graph=true)
 */
auto is_creating_graph() -> bool;

/**
 * @brief Set the higher-order gradient graph creation state.
 *
 * @param creating Whether to enable graph creation during backward
 */
auto set_creating_graph(bool creating) -> void;

/**
 * @brief RAII guard for create_graph mode during backward pass.
 *
 * Sets the thread-local create_graph flag on construction and restores
 * the previous state on destruction. Used internally by the backward engine
 * when create_graph=true.
 *
 * @code
 * {
 *     CreateGraphGuard guard;  // Enables create_graph mode
 *     // backward functions now use Variable ops instead of Tensor ops
 * }
 * // create_graph mode restored to previous state
 * @endcode
 *
 * @warning Thread-local — does NOT propagate to spawned threads.
 */
class CreateGraphGuard {
public:
    /**
     * @brief Construct guard and enable create_graph mode.
     */
    CreateGraphGuard();

    /**
     * @brief Restore previous create_graph state.
     */
    ~CreateGraphGuard();

    CreateGraphGuard(const CreateGraphGuard&) = delete;
    CreateGraphGuard& operator=(const CreateGraphGuard&) = delete;

private:
    bool prev_state_;  ///< Previous create_graph state
};

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
