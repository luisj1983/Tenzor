/**
 * @file function.hpp
 * @brief Autograd function interface and built-in operations
 *
 * Defines the Function base class for autograd operations and
 * provides implementations for common differentiable operations.
 */

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>
#include <atomic>
#include "../core/tensor.hpp"
#include "../sparse/sparse_tensor.hpp"
#include "variable.hpp"

namespace tenzor {

/**
 * @brief Controls behavior when create_graph=true but the operation
 * doesn't implement backward_with_variables().
 */
enum class HigherOrderGradMode : uint8_t {
    Error,   ///< Throw std::runtime_error (default, safe)
    Warn,    ///< Log warning per disconnection, fall through with disconnected graph
    Silent [[deprecated("Silent mode hides correctness bugs — use Warn or Error instead.")]]
             = 2  ///< @deprecated Behaves like Warn since v1.1. Will be removed.
};

/// Set the higher-order gradient fallback mode programmatically.
/// This takes precedence over the TENZOR_HIGHER_ORDER_GRAD env var.
void set_higher_order_grad_mode(HigherOrderGradMode mode);

/// Get the current higher-order gradient fallback mode.
auto get_higher_order_grad_mode() -> HigherOrderGradMode;

/// Returns the number of times higher-order gradient graph was disconnected
/// due to an operation not supporting backward_with_variables() (Warn/Silent mode).
/// Useful for programmatic detection in tests and debugging.
auto higher_order_disconnection_count() -> uint64_t;

/// Reset the disconnection counter to zero.
void reset_higher_order_disconnection_count();

namespace detail {
/// Internal: increment the disconnection counter (used by engine for stub ops)
void increment_higher_order_disconnection_count();
} // namespace detail

/// Enable/disable activation offloading to CPU for GPU-resident saved tensors.
/// When enabled, save_for_backward() moves GPU tensors to CPU RAM to reduce
/// GPU memory pressure, and saved_tensors() loads them back on access.
void set_activation_offload(bool enabled);
bool activation_offload_enabled();

// Forward declaration
class Variable;

/**
 * @brief Base class for autograd differentiable functions.
 *
 * Function represents a differentiable operation in the computation graph.
 * Each operation implements forward() and backward() to support automatic
 * differentiation.
 *
 * To create custom differentiable operations:
 * 1. Inherit from Function
 * 2. Implement forward() to compute output
 * 3. Implement backward() to compute gradients
 * 4. Use save_for_backward() to store tensors needed for gradient computation
 *
 * @code
 * class MyCustomOp : public Function {
 * public:
 *     auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
 *         // Save inputs if needed for backward
 *         save_for_backward({inputs[0].tensor()});
 *
 *         // Compute output
 *         Tensor result = compute(inputs[0].tensor());
 *         return {Variable(result)};
 *     }
 *
 *     auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
 *         // Retrieve saved tensors
 *         auto& input = saved_tensors()[0];
 *
 *         // Compute input gradient from output gradient
 *         Tensor grad_input = compute_gradient(input, grad_outputs[0]);
 *         return {grad_input};
 *     }
 * };
 * @endcode
 */
class Function : public std::enable_shared_from_this<Function> {
    friend class Variable;  // Allow Variable to access saved_tensors_

public:
    Function() : id_(next_id_.fetch_add(1, std::memory_order_relaxed)) {}
    virtual ~Function() = default;

    /**
     * @brief Unique identifier for this Function instance.
     *
     * Used as a stable key in gradient accumulation maps instead of raw
     * pointers, avoiding potential issues with address reuse after free.
     */
    auto id() const noexcept -> uint64_t { return id_; }

    /**
     * @brief Forward pass computation.
     *
     * Computes the function output from inputs. Should save any tensors
     * needed for the backward pass using save_for_backward().
     *
     * @param inputs Input variables
     * @return Output variables
     */
    virtual auto forward(std::vector<Variable> inputs) -> std::vector<Variable> = 0;

    /**
     * @brief Backward pass gradient computation.
     *
     * Computes gradients of inputs given gradients of outputs.
     * Uses saved tensors from forward pass.
     *
     * @param grad_outputs Gradients with respect to outputs
     * @return Gradients with respect to inputs
     */
    virtual auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> = 0;

    /**
     * @brief Higher-order backward pass using Variables for graph creation.
     *
     * When create_graph=true during backward(), this method is called instead
     * of backward(). It receives and returns Variables so that the gradient
     * computation itself is tracked by autograd, enabling higher-order gradients.
     *
     * The default implementation throws an error if any grad_output requires grad,
     * since the gradient graph cannot be built without a proper implementation.
     * Subclasses that support higher-order gradients must override this method.
     *
     * **create_graph coverage:** Currently the following ops have full
     * higher-order gradient support via backward_with_variables():
     *   - AddBackward, SubBackward, MulBackward, DivBackward, MatMulBackward
     *   - LinearBackward, SumBackward, MeanBackward, LogBackward, ExpBackward, NegBackward
     *   - SigmoidBackward_AG, TanhBackward_AG, GeluBackward, EluBackward, SeluBackward
     *   - MishBackward, LeakyReluBackward, SoftplusBackward, ReLUBackward
     *   - SinBackward, CosBackward, TanBackward, AsinBackward, AcosBackward, AtanBackward
     *   - SinhBackward, CoshBackward
     *   - ErfBackward, ErfcBackward, Log2Backward, Log10Backward, Log1pBackward
     *   - Exp2Backward, Expm1Backward, Atan2Backward
     *   - SqrtBackward, PowBackward, ReciprocalBackward, AbsBackward, ClampBackward
     *   - VarBackward, StdBackward, ProdBackward, LogSumExpBackward
     *   - ReshapeBackward, TransposeBackward, PermuteBackward, SqueezeBackward
     *   - UnsqueezeBackward, ExpandBackward, FlattenBackward, RollBackward
     *   - BmmBackward, CatBackward, SliceBackward, WhereBackward
     *   - FlipBackward, RepeatBackward, CumSumBackward, CumProdBackward
     *   - DiagBackward, TraceBackward, TriuBackward, TrilBackward
     *   - FFTBackward, IFFTBackward, RFFTBackward, IRFFTBackward
     *   - LogSoftmaxBackward, SoftmaxBackward
     *   - DetBackward, InvBackward, SolveBackward, NormBackward_Linalg, SlogdetBackward
     *   - CholeskyBackward, SvdBackward, QrBackward, EighBackward, EigvalshBackward
     *   - SpMMBackward, SpMVBackward, SparseAddBackward
     *   - MedianBackward, ModeBackward
     *
     * Ops with full backward_with_variables using Variable-level scatter_add:
     *   - GatherBackward, IndexSelectBackward, IndexBackward, ScatterAddBackward
     *
     * Ops with passthrough stubs (second derivative is zero or non-differentiable):
     *   - MaxBackward, MinBackward (non-differentiable mask)
     *   - ScatterBackward, NarrowBackward
     *   - TopKBackward, SortBackward, UpsampleBilinearBackward
     *
     * @param grad_outputs Gradient Variables with respect to outputs
     * @return Gradient Variables with respect to inputs (with grad_fn set)
     * @throws std::runtime_error if create_graph=true and op doesn't support it
     */
    virtual auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable>;

    /**
     * @brief Query whether this function supports higher-order gradients.
     *
     * Returns true if the subclass has a proper backward_with_variables()
     * implementation that builds a gradient graph. The default returns false.
     * Subclasses that override backward_with_variables() should also override
     * this to return true.
     *
     * @note In practice, if a subclass overrides backward_with_variables(),
     * the base class fallback (which checks this flag) is never reached.
     * This method is primarily useful for introspection and pre-flight checks.
     */
    virtual auto supports_higher_order() const -> bool { return false; }

    /**
     * @brief Query whether this function's backward_with_variables() is a stub.
     *
     * Returns true if the subclass has a passthrough stub that delegates to
     * backward() and wraps results without building a gradient graph. This
     * means higher-order gradients through this operation will be zero.
     *
     * Operations where this returns true include non-differentiable mask ops
     * (MaxBackward, MinBackward) and ops where the second derivative is
     * structurally zero (ScatterBackward, TopKBackward, SortBackward, etc.).
     *
     * @return true if backward_with_variables() is a passthrough stub
     */
    virtual auto is_higher_order_stub() const -> bool { return false; }

    /**
     * @brief Get readable name for this backward function.
     *
     * Used in error messages and debugging output. Default returns
     * demangled C++ type name. Subclasses should override for readability.
     *
     * @return Human-readable function name (e.g., "AddBackward", "MatMulBackward")
     */
    virtual auto name() const -> std::string;

    /**
     * @brief Set next functions in computation graph.
     *
     * Links this function to preceding functions for backpropagation.
     *
     * @param funcs Vector of gradient functions to chain to
     */
    auto set_next_functions(std::vector<std::shared_ptr<Function>> funcs) -> void;

    /**
     * @brief Get next functions in computation graph.
     *
     * @return Vector of chained gradient functions
     */
    auto next_functions() const -> const std::vector<std::shared_ptr<Function>>&;

    /**
     * @brief Set input variables for gradient accumulation.
     *
     * Stores Variables by value for gradient accumulation during backward pass.
     * The Variables' shared_ptr<VariableImpl> keeps the data alive even if the
     * Variable handle (temporary) is destroyed.
     *
     * @param inputs Vector of input Variables to track
     */
    auto set_input_variables(std::vector<Variable> inputs) -> void;

    /**
     * @brief Get input variables for gradient accumulation.
     *
     * @return Vector of tracked Variables
     */
    auto input_variables() const -> const std::vector<Variable>&;
    auto input_variables() -> std::vector<Variable>&;

    /**
     * @brief Reload offloaded saved tensors back to GPU.
     *
     * If activation offloading moved saved tensors to CPU during forward,
     * this reloads them to the original GPU device before backward.
     */
    void reload_saved_tensors() const;

    /**
     * @brief Release saved tensors to free memory.
     *
     * Called after backward() to release GPU memory held by saved tensors.
     */
    void release_saved_tensors() {
        std::lock_guard lock(offload_mutex_);
        saved_tensors_.clear();
        saved_versions_.clear();
        tensors_offloaded_.store(false, std::memory_order_relaxed);
    }

    /**
     * @brief Validate that saved tensors have not been modified in-place.
     *
     * Checks version counters recorded by save_for_backward() against
     * current tensor versions. Throws if any tensor was modified in-place
     * after the forward pass.
     */
    void validate_saved_tensors() const;

    /// Assert that at least @p count tensors were saved. Throws if not.
    void require_saved_tensors(size_t count) const;

    /// Assert that at least @p count variables were saved. Throws if not.
    void require_saved_variables(size_t count) const;

    /**
     * @brief Get number of saved tensors.
     *
     * @return Count of tensors saved for backward pass
     */
    auto num_saved_tensors() const -> size_t { return saved_tensors_.size(); }

    /**
     * @brief Save tensors for backward pass.
     *
     * Stores tensors that will be needed to compute gradients.
     * Call this during forward pass.
     *
     * @param tensors Tensors to save for gradient computation
     */
    auto save_for_backward(std::vector<Tensor> tensors) -> void;

    /**
     * @brief Get saved tensors.
     *
     * Retrieves tensors saved during forward pass for use in backward pass.
     * Validates that saved tensors have not been modified in-place since save.
     *
     * @return Reference to saved tensors vector
     */
    auto saved_tensors() const -> const std::vector<Tensor>&;

    /**
     * @brief Save Variables for backward pass (preserves graph connections).
     *
     * When create_graph=true, forward operations should save Variables instead
     * of raw Tensors so that the backward pass can use Variable operations
     * and build a higher-order gradient graph.
     *
     * @param variables Variables to save for gradient computation
     */
    auto save_variables_for_backward(std::vector<Variable> variables) -> void;

    /**
     * @brief Get saved Variables.
     *
     * Retrieves Variables saved during forward pass. If no Variables were saved
     * but Tensors were saved, wraps the Tensors as Variables without grad tracking.
     *
     * @return Vector of saved Variables
     */
    auto saved_variables() const -> const std::vector<Variable>&;

    /**
     * @brief Check if Variables were saved for backward.
     *
     * @return true if save_variables_for_backward() was called
     */
    auto has_saved_variables() const -> bool { return !saved_variables_.empty(); }

protected:
    mutable std::vector<Tensor> saved_tensors_;                     ///< Tensors saved for backward
    mutable std::vector<uint64_t> saved_versions_;                  ///< Tensor versions at save time (for in-place detection)
    mutable std::vector<Variable> saved_variables_;                 ///< Variables saved for backward (preserves graph for create_graph)
    mutable Device offloaded_device_{Device::cpu()};                ///< Original device when offloaded
    mutable std::atomic<bool> tensors_offloaded_{false};            ///< Whether saved tensors are on CPU due to offloading
    mutable std::mutex offload_mutex_;                              ///< Guards offload/reload of saved tensors
    std::vector<std::shared_ptr<Function>> next_functions_;         ///< Chained gradient functions
    std::vector<Variable> input_variables_;                          ///< Input variables for gradient accumulation (stored by value)

private:
    uint64_t id_;                                                    ///< Unique identifier (stable key for gradient accumulators)
    static std::atomic<uint64_t> next_id_;                           ///< Global counter for unique IDs
};

// ============================================================================
// Built-in Autograd Functions
// ============================================================================

/**
 * @brief Addition gradient function.
 *
 * Implements forward and backward for element-wise addition with broadcasting support.
 *
 * Forward: C = A + B (with broadcasting)
 * Backward: dL/dA = sum_reduce(dL/dC), dL/dB = sum_reduce(dL/dC)
 *
 * @note Gradients are sum-reduced along broadcasted dimensions to match input shapes.
 *
 * @code
 * Variable a(Tensor({2, 3}, DType::Float32, Device::cpu()), true);
 * Variable b(Tensor({3}, DType::Float32, Device::cpu()), true);
 * Variable c = a + b;  // Broadcasting: {2,3} + {3} -> {2,3}
 * // Backward: grad_b summed from {2,3} to {3}
 * @endcode
 */
class AddBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }

    // Public for direct access from Variable operators
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};

/**
 * @brief Subtraction gradient function.
 *
 * Implements forward and backward for element-wise subtraction with broadcasting support.
 *
 * Forward: C = A - B (with broadcasting)
 * Backward: dL/dA = sum_reduce(dL/dC), dL/dB = -sum_reduce(dL/dC)
 *
 * @note Gradients are sum-reduced along broadcasted dimensions to match input shapes.
 */
class SubBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }

    // Public for direct access from Variable operators
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};

/**
 * @brief Multiplication gradient function.
 *
 * Implements forward and backward for element-wise multiplication with broadcasting support.
 *
 * Forward: C = A * B (with broadcasting)
 * Backward: dL/dA = sum_reduce(dL/dC * B), dL/dB = sum_reduce(dL/dC * A)
 *
 * @note Uses product rule for differentiation. Input tensors and shapes are saved for backward pass.
 */
class MulBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }

    // Public for direct access from Variable operators
    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};

/**
 * @brief Division gradient function.
 *
 * Implements forward and backward for element-wise division.
 *
 * Forward: C = A / B
 * Backward: dL/dA = dL/dC / B, dL/dB = -dL/dC * A / (B^2)
 *
 * @note Uses quotient rule for differentiation. Input tensors are saved for backward pass.
 */
class DivBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }

    std::vector<int64_t> input_shape_a_;
    std::vector<int64_t> input_shape_b_;
};

/**
 * @brief Matrix multiplication gradient function.
 *
 * Implements forward and backward for matrix multiplication.
 *
 * Forward: C = A @ B
 * Backward: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
 *
 * @note Input matrices are saved for backward pass. Supports batched matrix multiplication.
 *
 * @code
 * Variable A(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable B(Tensor({4, 5}, DType::Float32, Device::cpu()), true);
 * Variable C = matmul(A, B);  // Shape: {3, 5}
 * @endcode
 */
class MatMulBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

/**
 * @brief Fused linear layer gradient function.
 *
 * Implements fused forward and backward for y = x @ W.T + b
 * More efficient than separate matmul + add operations.
 *
 * Forward: y = x @ W.T + b (via linear_kernel)
 * Backward:
 *   dL/dx = dL/dy @ W
 *   dL/dW = dL/dy.T @ x
 *   dL/db = sum(dL/dy, dim=0)
 *
 * @note All three inputs (x, W, b) are saved for backward pass.
 *       Uses optimized MKL kernels internally.
 */
class LinearBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

// Note: ReLUBackward is implemented in src/nn/activations/activations.cpp
// with full higher-order gradient support. No separate declaration needed here.

/**
 * @brief Sum reduction gradient function.
 *
 * Implements forward and backward for sum reduction operation.
 *
 * Forward: y = sum(x, dim, keepdim)
 * Backward: dL/dx = broadcast(dL/dy, original_shape)
 *
 * @note Gradient is broadcast back to original input shape. Original shape is saved.
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = x.sum(1);  // Sum along dimension 1, shape: {3}
 * // Gradient will broadcast back to {3, 4}
 * @endcode
 */
class SumBackward : public Function {
public:
    SumBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

/**
 * @brief Mean reduction gradient function.
 *
 * Implements forward and backward for mean reduction operation.
 *
 * Forward: y = mean(x, dim, keepdim)
 * Backward: dL/dx = broadcast(dL/dy / count, original_shape)
 *
 * @note Gradient is divided by element count then broadcast to original shape.
 */
class MeanBackward : public Function {
public:
    MeanBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

/**
 * @brief Natural logarithm gradient function.
 *
 * Forward: y = log(x)
 * Backward: dL/dx = dL/dy / x
 *
 * @note Input is saved for gradient computation. Undefined for x <= 0.
 */
class LogBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

/**
 * @brief Exponential gradient function.
 *
 * Forward: y = exp(x)
 * Backward: dL/dx = dL/dy * exp(x) = dL/dy * y
 *
 * @note Output is saved for efficient gradient computation.
 */
class ExpBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

/**
 * @brief Negation gradient function.
 *
 * Forward: y = -x
 * Backward: dL/dx = -dL/dy
 *
 * @note Gradient is simply negated.
 */
class NegBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

/**
 * @brief Log-softmax gradient function.
 *
 * Implements numerically stable log-softmax operation.
 *
 * Forward: y_i = x_i - log(sum(exp(x_j)))
 * Backward: dL/dx_i = dL/dy_i - exp(y_i) * sum(dL/dy_j)
 *
 * @note Output is saved for gradient computation. More numerically stable than log(softmax(x)).
 *
 * @code
 * Variable logits(Tensor({batch_size, num_classes}, DType::Float32, Device::cpu()), true);
 * Variable log_probs = log_softmax(logits, 1);  // Compute along class dimension
 * @endcode
 */
class LogSoftmaxBackward : public Function {
public:
    LogSoftmaxBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    int64_t dim_;
};

/**
 * @brief Softmax gradient function.
 *
 * Implements softmax activation with autograd support.
 *
 * Forward: y_i = exp(x_i) / sum(exp(x_j))
 * Backward: dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
 *
 * @note Output is saved for gradient computation.
 */
class SoftmaxBackward : public Function {
public:
    SoftmaxBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    int64_t dim_;
};

/**
 * @brief Absolute value gradient function.
 *
 * Forward: y = |x|
 * Backward: dL/dx = dL/dy * sign(x)
 *
 * @note Input is saved to compute sign. Gradient is undefined at x=0.
 */
class AbsBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

/**
 * @brief Clamp gradient function.
 *
 * Forward: y = clamp(x, min, max) = max(min, min(x, max))
 * Backward: dL/dx = dL/dy * (min < x < max)
 *
 * @note Gradient passes through only for elements within bounds. Input is saved.
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = clamp(x, -1.0f, 1.0f);  // Clamp to [-1, 1]
 * @endcode
 */
class ClampBackward : public Function {
public:
    ClampBackward(float min, float max) : min_(min), max_(max) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    float min_;
    float max_;
};

/**
 * @brief Max reduction gradient function.
 *
 * Forward: y = max(x, dim, keepdim)
 * Backward: dL/dx_i = dL/dy if x_i == max, else 0
 *
 * @note Gradient flows only to maximum elements. Input and indices are saved.
 * If multiple elements are tied for max, gradient is split among them.
 */
class MaxBackward : public Function {
public:
    MaxBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

/**
 * @brief Median reduction gradient function.
 *
 * Forward: y = median(x, dim, keepdim)
 * Backward: dL/dx_i = dL/dy if x_i is the median element, else 0
 */
class MedianBackward : public Function {
public:
    MedianBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

/**
 * @brief Mode reduction gradient function.
 *
 * Forward: y = mode(x, dim, keepdim)
 * Backward: dL/dx_i = dL/dy if x_i is the first occurrence of mode value, else 0
 */
class ModeBackward : public Function {
public:
    ModeBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    std::optional<int64_t> dim_;
    bool keepdim_;
};

/**
 * @brief Reshape gradient function.
 *
 * Forward: y = reshape(x, shape)
 * Backward: dL/dx = reshape(dL/dy, input_shape)
 *
 * @note Original input shape is saved for gradient reshaping.
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = reshape(x, {12});  // Forward: {3, 4} -> {12}
 * // Backward: gradient reshaped from {12} back to {3, 4}
 * @endcode
 */
class ReshapeBackward : public Function {
public:
    ReshapeBackward(std::vector<int64_t> input_shape) : input_shape_(std::move(input_shape)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    std::vector<int64_t> input_shape_;
};

/**
 * @brief Permute gradient function.
 *
 * Forward: y = permute(x, dims)
 * Backward: dL/dx = permute(dL/dy, inverse_dims)
 *
 * @note Inverse permutation is computed and saved for gradient computation.
 *
 * @code
 * Variable x(Tensor({2, 3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = permute(x, {2, 0, 1});  // Forward: {2, 3, 4} -> {4, 2, 3}
 * // Backward: gradient permuted from {4, 2, 3} back to {2, 3, 4}
 * @endcode
 */
class PermuteBackward : public Function {
public:
    PermuteBackward(std::vector<int64_t> dims) : dims_(std::move(dims)) {
        // Compute inverse permutation
        inv_dims_.resize(dims_.size());
        for (size_t i = 0; i < dims_.size(); ++i) {
            inv_dims_[dims_[i]] = static_cast<int64_t>(i);
        }
    }
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    std::vector<int64_t> dims_;
    std::vector<int64_t> inv_dims_;
};

/**
 * @brief Transpose gradient function.
 *
 * Forward: y = transpose(x, dim0, dim1)
 * Backward: dL/dx = transpose(dL/dy, dim0, dim1)
 *
 * @note Transpose is its own inverse, so backward uses same dimensions.
 */
class TransposeBackward : public Function {
public:
    TransposeBackward(int64_t dim0, int64_t dim1) : dim0_(dim0), dim1_(dim1) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    int64_t dim0_;
    int64_t dim1_;
};

/**
 * @brief Roll gradient function.
 *
 * Forward: y = roll(x, shifts, dim)
 * Backward: dL/dx = roll(dL/dy, -shifts, dim)
 *
 * @note Rolling is reversed by rolling in the opposite direction.
 */
class RollBackward : public Function {
public:
    RollBackward(int64_t shifts, int64_t dim) : shifts_(shifts), dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    int64_t shifts_;
    int64_t dim_;
};

/**
 * @brief Squeeze gradient function.
 *
 * Forward: y = squeeze(x, dim)
 * Backward: dL/dx = unsqueeze(dL/dy, dim)
 *
 * @note Squeezing dimension is saved for unsqueezing in backward pass.
 */
class SqueezeBackward : public Function {
public:
    SqueezeBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    int64_t dim_;
};

/**
 * @brief Batch matrix multiplication gradient function.
 *
 * Implements forward and backward for batched matrix multiplication.
 *
 * Forward: C = bmm(A, B) where A: (batch, n, m), B: (batch, m, p), C: (batch, n, p)
 * Backward:
 *   dL/dA = bmm(dL/dC, permute(B, {0, 2, 1}))
 *   dL/dB = bmm(permute(A, {0, 2, 1}), dL/dC)
 *
 * @note Input tensors are saved for gradient computation using transposition.
 *
 * @code
 * Variable a(Tensor({32, 10, 20}, DType::Float32, Device::cpu()), true);
 * Variable b(Tensor({32, 20, 30}, DType::Float32, Device::cpu()), true);
 * Variable c = bmm(a, b);  // Uses BmmBackward internally
 * c.backward();  // Computes gradients w.r.t. a and b
 * @endcode
 */
class BmmBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

/**
 * @brief Concatenation gradient function.
 *
 * Implements forward and backward for tensor concatenation.
 *
 * Forward: y = cat([x1, x2, ..., xn], dim)
 * Backward: Split dL/dy back to [dL/dx1, dL/dx2, ..., dL/dxn] along dim
 *
 * @note Split sizes and concatenation dimension are saved for gradient splitting.
 *
 * @code
 * Variable x1(Tensor({2, 3}, DType::Float32, Device::cpu()), true);
 * Variable x2(Tensor({2, 5}, DType::Float32, Device::cpu()), true);
 * Variable y = cat({x1, x2}, 1);  // Forward: {2,3} + {2,5} -> {2,8}
 * // Backward: gradient split from {2,8} back to {2,3} and {2,5}
 * @endcode
 */
class CatBackward : public Function {
public:
    CatBackward(std::vector<int64_t> split_sizes, int64_t dim)
        : split_sizes_(std::move(split_sizes)), dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    std::vector<int64_t> split_sizes_;  ///< Size of each input along concat dimension
    int64_t dim_;                        ///< Concatenation dimension
};

/**
 * @brief Slice gradient function.
 *
 * Implements forward and backward for tensor slicing.
 *
 * Forward: y = slice(x, dim, start, end, step)
 * Backward: Scatter dL/dy back to positions in dL/dx, zeros elsewhere
 *
 * @note Original input shape and slice parameters are saved for gradient scattering.
 *
 * @code
 * Variable x(Tensor({10, 20}, DType::Float32, Device::cpu()), true);
 * Variable y = slice(x, 1, 5, 15, 2);  // Shape: {10, 5} - every 2nd element from 5 to 15
 * // Backward: gradient scattered back to original positions in {10, 20}
 * @endcode
 */
class SliceBackward : public Function {
public:
    SliceBackward(std::vector<int64_t> input_shape, int64_t dim, int64_t start, int64_t end, int64_t step)
        : input_shape_(std::move(input_shape)), dim_(dim), start_(start), end_(end), step_(step) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
private:
    std::vector<int64_t> input_shape_;  ///< Original input shape
    int64_t dim_;                        ///< Slice dimension
    int64_t start_;                      ///< Start index
    int64_t end_;                        ///< End index (exclusive)
    int64_t step_;                       ///< Step size
};

/**
 * @brief Bilinear upsample gradient function.
 *
 * Implements forward and backward for bilinear upsampling (currently using nearest neighbor).
 *
 * Forward: y = upsample(x, target_h, target_w)
 * Backward: Distribute dL/dy back to input pixels
 *
 * For nearest neighbor upsampling:
 * - Each output pixel comes from exactly one input pixel
 * - Gradient at output pixel is added back to corresponding input pixel
 *
 * @note Current implementation uses nearest neighbor for both forward and backward.
 *       True bilinear interpolation would distribute gradients to 4 neighboring pixels.
 */
class UpsampleBilinearBackward : public Function {
public:
    UpsampleBilinearBackward(int64_t input_h, int64_t input_w, int64_t output_h, int64_t output_w)
        : input_h_(input_h), input_w_(input_w), output_h_(output_h), output_w_(output_w) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }
private:
    int64_t input_h_;   ///< Input height
    int64_t input_w_;   ///< Input width
    int64_t output_h_;  ///< Output height
    int64_t output_w_;  ///< Output width
};

// =========================================================================
// Activation Backward Functions
// =========================================================================

class SigmoidBackward_AG : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class TanhBackward_AG : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class GeluBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class EluBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class SeluBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class MishBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class LeakyReluBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class SoftplusBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

// =========================================================================
// Element-wise Math Backward Functions
// =========================================================================

class SqrtBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class PowBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class ReciprocalBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class SinBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class CosBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class TanBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class AsinBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class AcosBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class AtanBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class SinhBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class CoshBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

// =========================================================================
// Extended Math Backward Functions
// =========================================================================

class ErfBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class ErfcBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class ErfInvBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class GammaBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class LgammaBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class DigammaBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class BesselI0Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class BesselI1Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class SincBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class Log2Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class Log10Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class Log1pBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class Exp2Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class Expm1Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class Atan2Backward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

// =========================================================================
// Reduction Backward Functions
// =========================================================================

class MinBackward : public Function {
public:
    MinBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }
    std::optional<int64_t> dim_;
    bool keepdim_;
};

class StdBackward : public Function {
public:
    StdBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    std::optional<int64_t> dim_;
    bool keepdim_;
};

class VarBackward : public Function {
public:
    VarBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    std::optional<int64_t> dim_;
    bool keepdim_;
};

class ProdBackward : public Function {
public:
    ProdBackward(std::optional<int64_t> dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    std::optional<int64_t> dim_;
    bool keepdim_;
};

class LogSumExpBackward : public Function {
public:
    LogSumExpBackward(int64_t dim, bool keepdim) : dim_(dim), keepdim_(keepdim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    int64_t dim_;
    bool keepdim_;
};

// =========================================================================
// Shape/Indexing Backward Functions
// =========================================================================

class UnsqueezeBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class ExpandBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class DeviceTransferBackward : public Function {
public:
    Device source_device;  // Device to transfer gradients back to
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};

class FlattenBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class WhereBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class GatherBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class ScatterBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }
};

/**
 * @brief Backward for scatter_add: output[dim][index[i]] += src[i]
 *
 * Forward:  y = scatter_add(x, dim, index, src)
 * Backward: grad_x = grad_y (identity), grad_src = gather(grad_y, dim, index)
 *
 * Saved tensors:
 *   [0] = dim (scalar Float32, cast to int64_t)
 *   [1] = index tensor
 */
class ScatterAddBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class IndexSelectBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

/**
 * @brief Backward for advanced (fancy) indexing.
 *
 * Forward:  y = x[idx0, idx1, ...]   (gather)
 * Backward: grad_x = zeros_like(x); grad_x.index_put(indices, grad_y, accumulate=true)
 *
 * Saved tensors:
 *   [0] = input shape (1D Int64)
 *   [1..N] = index tensors (0-element sentinel for null dims)
 *   N stored in num_indices_.
 */
class IndexBackward : public Function {
public:
    void set_num_indices(int64_t n) { num_indices_ = n; }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "IndexBackward"; }

private:
    int64_t num_indices_ = 0;
};

class NarrowBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }
};

class FlipBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

class RepeatBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
};

// =========================================================================
// Cumulative, Sorting, and Triangular Backward Functions
// =========================================================================

/**
 * @brief Cumulative sum gradient function.
 *
 * Forward: y = cumsum(x, dim)
 * Backward: dL/dx = flip(cumsum(flip(dL/dy, dim), dim), dim)
 *   i.e. reverse cumulative sum of the gradient
 */
class CumSumBackward : public Function {
public:
    explicit CumSumBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CumSumBackward"; }
private:
    int64_t dim_;
};

/**
 * @brief Cumulative product gradient function.
 *
 * Forward: y = cumprod(x, dim)
 * Backward: dL/dx = flip(cumsum(flip(y * dL/dy, dim), dim), dim) / x
 *   with zero-safe division using nan_to_num
 *
 * @note Saves input and output for backward.
 */
class CumProdBackward : public Function {
public:
    explicit CumProdBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CumProdBackward"; }
private:
    int64_t dim_;
};

/**
 * @brief TopK gradient function.
 *
 * Forward: (values, indices) = topk(x, k, dim)
 * Backward: scatter grad into zeros at saved indices positions
 *
 * @note Only values receive gradients; indices are non-differentiable.
 */
class TopKBackward : public Function {
public:
    TopKBackward(int64_t k, int64_t dim) : k_(k), dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }
    auto name() const -> std::string override { return "TopKBackward"; }
private:
    int64_t k_;
    int64_t dim_;
};

/**
 * @brief Sort gradient function.
 *
 * Forward: (sorted_values, indices) = sort(x, dim)
 * Backward: scatter grad using inverse permutation of sort indices
 *
 * @note Only sorted values receive gradients; indices are non-differentiable.
 */
class SortBackward : public Function {
public:
    explicit SortBackward(int64_t dim) : dim_(dim) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }
    auto name() const -> std::string override { return "SortBackward"; }
private:
    int64_t dim_;
};

/**
 * @brief Diag gradient function.
 *
 * Forward: y = diag(x, k)
 * Backward:
 *   If input was 1D (output 2D): dL/dx = diag(dL/dy, k) — extract diagonal
 *   If input was 2D (output 1D): dL/dx = diag(dL/dy, k) — construct diagonal matrix
 *
 * @note Saves input ndim to select correct backward path.
 */
class DiagBackward : public Function {
public:
    DiagBackward(int64_t input_ndim, int64_t diagonal) : input_ndim_(input_ndim), diagonal_(diagonal) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "DiagBackward"; }
private:
    int64_t input_ndim_;
    int64_t diagonal_;
};

/**
 * @brief Trace gradient function.
 *
 * Forward: y = trace(A)  (sum of diagonal elements)
 * Backward: dL/dA = dL/dy * eye(n)
 */
class TraceBackward : public Function {
public:
    explicit TraceBackward(int64_t n) : n_(n) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "TraceBackward"; }
private:
    int64_t n_;
};

/**
 * @brief Upper triangular gradient function.
 *
 * Forward: y = triu(x, k)
 * Backward: dL/dx = triu(dL/dy, k)
 */
class TriuBackward : public Function {
public:
    explicit TriuBackward(int64_t diagonal) : diagonal_(diagonal) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "TriuBackward"; }
private:
    int64_t diagonal_;
};

/**
 * @brief Lower triangular gradient function.
 *
 * Forward: y = tril(x, k)
 * Backward: dL/dx = tril(dL/dy, k)
 */
class TrilBackward : public Function {
public:
    explicit TrilBackward(int64_t diagonal) : diagonal_(diagonal) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "TrilBackward"; }
private:
    int64_t diagonal_;
};

// =========================================================================
// FFT Backward Functions
// =========================================================================

/**
 * @brief FFT gradient function.
 *
 * Forward: y = fft(x, n, dim, norm)
 * Backward: dL/dx = ifft(dL/dy, n, dim, norm)
 */
class FFTBackward : public Function {
public:
    FFTBackward(std::optional<int64_t> n, int64_t dim, std::string norm)
        : n_(n), dim_(dim), norm_(std::move(norm)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "FFTBackward"; }
private:
    std::optional<int64_t> n_;
    int64_t dim_;
    std::string norm_;
};

/**
 * @brief Inverse FFT gradient function.
 *
 * Forward: y = ifft(x, n, dim, norm)
 * Backward: dL/dx = fft(dL/dy, n, dim, norm)
 */
class IFFTBackward : public Function {
public:
    IFFTBackward(std::optional<int64_t> n, int64_t dim, std::string norm)
        : n_(n), dim_(dim), norm_(std::move(norm)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "IFFTBackward"; }
private:
    std::optional<int64_t> n_;
    int64_t dim_;
    std::string norm_;
};

/**
 * @brief Real FFT gradient function.
 *
 * Forward: y = rfft(x, n, dim, norm)
 * Backward: dL/dx = irfft(dL/dy, n=signal_length, dim, norm)
 *
 * @note Saves original signal length for irfft reconstruction.
 */
class RFFTBackward : public Function {
public:
    RFFTBackward(int64_t signal_length, int64_t dim, std::string norm)
        : signal_length_(signal_length), dim_(dim), norm_(std::move(norm)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "RFFTBackward"; }
private:
    int64_t signal_length_;
    int64_t dim_;
    std::string norm_;
};

/**
 * @brief Inverse real FFT gradient function.
 *
 * Forward: y = irfft(x, n, dim, norm)
 * Backward: dL/dx = rfft(dL/dy, dim, norm)
 */
class IRFFTBackward : public Function {
public:
    IRFFTBackward(int64_t dim, std::string norm)
        : dim_(dim), norm_(std::move(norm)) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "IRFFTBackward"; }
private:
    int64_t dim_;
    std::string norm_;
};

// =========================================================================
// Linear Algebra Backward Functions
// =========================================================================

/**
 * @brief Determinant gradient function.
 *
 * Forward: y = det(A)
 * Backward: dL/dA = dL/dy * det(A) * A^{-T}
 *
 * @note Saves det result and inverse for backward computation.
 */
class DetBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "DetBackward"; }
};

/**
 * @brief Matrix inverse gradient function.
 *
 * Forward: Y = A^{-1}
 * Backward: dL/dA = -Y^T @ dL/dY @ Y^T
 *
 * @note Saves inverse result for backward computation.
 */
class InvBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "InvBackward"; }
};

/**
 * @brief Linear solve gradient function.
 *
 * Forward: X = solve(A, B) where AX = B
 * Backward:
 *   dL/dB = solve(A^T, dL/dX)
 *   dL/dA = -dL/dB @ X^T
 *
 * @note Saves A and solution X for backward computation.
 */
class SolveBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SolveBackward"; }
};

/**
 * @brief Cholesky decomposition gradient function.
 *
 * Forward: L = cholesky(A) where A = L @ L^T
 * Backward: dL/dA = L^{-T} @ phi(L^T @ dL/dL) @ L^{-1}
 *           where phi(X) = tril(X) with diagonal halved
 *
 * @note Saves L for backward computation.
 */
class CholeskyBackward : public Function {
public:
    CholeskyBackward(bool upper = false) : upper_(upper) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CholeskyBackward"; }
private:
    bool upper_;
};

/**
 * @brief SVD gradient function.
 *
 * Forward: (U, S, Vh) = svd(A)
 * Backward: Uses the SVD backward formula involving F matrix
 *
 * @note Saves U, S, Vh for backward computation. Full backward is complex.
 */
class SvdBackward : public Function {
public:
    SvdBackward(bool full_matrices = true) : full_matrices_(full_matrices) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SvdBackward"; }
private:
    bool full_matrices_;
};

/**
 * @brief QR decomposition gradient function.
 *
 * Forward: (Q, R) = qr(A)
 * Backward: Uses the QR backward formula
 *   dL/dA = (dL/dQ + Q @ copyltu(Q^T @ dL/dQ - dL/dR^T @ R^{-T})) @ R^{-T}
 *
 * @note Saves Q and R for backward computation.
 */
class QrBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "QrBackward"; }
};

/**
 * @brief Symmetric eigendecomposition gradient function.
 *
 * Forward: (W, V) = eigh(A) where A = V @ diag(W) @ V^T
 * Backward:
 *   F_{ij} = 1/(w_j - w_i) for i != j, 0 on diagonal
 *   dL/dA = V @ (F * (V^T @ dL/dV) + diag(dL/dW)) @ V^T
 *
 * @note Saves eigenvalues W and eigenvectors V for backward computation.
 */
class EighBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "EighBackward"; }
};

/**
 * @brief Eigenvalues-only gradient function for symmetric matrices.
 *
 * Forward: W = eigvalsh(A)
 * Backward: dL/dA = V @ diag(dL/dW) @ V^T
 *           where V is computed via eigh during forward
 *
 * @note Saves eigenvectors V for backward computation (computed during forward).
 */
class EigvalshBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "EigvalshBackward"; }
};

/**
 * @brief Matrix norm gradient function.
 *
 * Forward: y = norm(A, ord)
 * Backward (Frobenius): dL/dA = dL/dy * A / norm(A)
 *
 * @note Saves input and norm result for backward computation.
 *       Currently only supports Frobenius norm backward.
 */
class NormBackward_Linalg : public Function {
public:
    NormBackward_Linalg(const std::string& ord = "fro") : ord_(ord) {}
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "NormBackward_Linalg"; }
private:
    std::string ord_;
};

/**
 * @brief Slogdet gradient function.
 *
 * Forward: (sign, logabsdet) = slogdet(A)
 * Backward: dL/dA = dL/d(logabsdet) * A^{-T}
 *           (sign gradient is zero since sign is discrete)
 *
 * @note Saves inverse for backward computation.
 */
class SlogdetBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SlogdetBackward"; }
};

// =========================================================================
// Sparse Backward Functions
// =========================================================================

/**
 * @brief Backward function for sparse-dense matrix multiplication (spmm).
 *
 * For Y = S @ D where S is sparse (M,K) and D is dense (K,N):
 * - grad_D = S^T @ grad_Y  (uses sparse::spmm with transposed sparse matrix)
 * - The sparse matrix S is not differentiated (treated as constant).
 *
 * The transposed sparse matrix S^T is stored directly as a SparseTensor,
 * avoiding the memory cost of converting to dense.
 */
class SpMMBackward : public Function {
public:
    /// Set the transposed sparse matrix for backward computation.
    void set_sparse_transposed(SparseTensor sparse_t) {
        sparse_transposed_.emplace(std::move(sparse_t));
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SpMMBackward"; }

private:
    std::optional<SparseTensor> sparse_transposed_;  ///< S^T stored in sparse format
};

/**
 * @brief Backward function for sparse-dense matrix-vector multiplication (spmv).
 *
 * For y = S @ v where S is sparse (M,K) and v is dense (K,):
 * - grad_v = S^T @ grad_y  (uses sparse::spmv with transposed sparse matrix)
 * - The sparse matrix S is not differentiated (treated as constant).
 *
 * The transposed sparse matrix S^T is stored directly as a SparseTensor,
 * avoiding the memory cost of converting to dense.
 */
class SpMVBackward : public Function {
public:
    /// Set the transposed sparse matrix for backward computation.
    void set_sparse_transposed(SparseTensor sparse_t) {
        sparse_transposed_.emplace(std::move(sparse_t));
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SpMVBackward"; }

private:
    std::optional<SparseTensor> sparse_transposed_;  ///< S^T stored in sparse format
};

/**
 * @brief Backward function for sparse-dense addition.
 *
 * For Y = S + D (sparse + dense = dense):
 * - grad_D = grad_Y  (gradient passes through to the dense input)
 * - The sparse matrix S is not differentiated (treated as constant).
 */
class SparseAddBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SparseAddBackward"; }
};

/**
 * @brief Backward function for sparse-sparse matrix multiplication (SpGEMM).
 *
 * For C = A @ B where A and B are sparse:
 * - grad_A = grad_C @ B^T (sparse)
 * - grad_B = A^T @ grad_C (sparse)
 * Both A and B are treated as constants (sparse matrices are not differentiable
 * through the sparsity pattern), so gradients flow through dense conversions.
 */
class SpGEMMBackward : public Function {
public:
    void set_sparse_a_transposed(SparseTensor at) { sparse_a_t_.emplace(std::move(at)); }
    void set_sparse_b_transposed(SparseTensor bt) { sparse_b_t_.emplace(std::move(bt)); }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SpGEMMBackward"; }

private:
    std::optional<SparseTensor> sparse_a_t_;  ///< A^T in sparse format
    std::optional<SparseTensor> sparse_b_t_;  ///< B^T in sparse format
};

/**
 * @brief Backward function for sparse triangular solve.
 *
 * For x = L^{-1} @ b (lower triangular solve):
 * - grad_b = L^{-T} @ grad_x
 * - L is not differentiated (treated as constant sparse matrix).
 */
class SparseTriSolveBackward : public Function {
public:
    void set_sparse_l_transposed(SparseTensor lt) { sparse_l_t_.emplace(std::move(lt)); }
    void set_upper(bool u) { upper_ = u; }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SparseTriSolveBackward"; }

private:
    std::optional<SparseTensor> sparse_l_t_;  ///< L^T (or U^T) in sparse format
    bool upper_{false};
};

// ============================================================================
// Custom Op Autograd
// ============================================================================

/// Forward declaration of custom op backward function type
using CustomBackwardFn = std::function<std::vector<Tensor>(
    std::span<const Tensor> saved_tensors,
    std::span<const Tensor> grad_outputs)>;

/**
 * @brief Autograd function for user-registered custom operations.
 *
 * Created by autograd::dispatch_custom_op() when inputs require gradients.
 * Stores the user-provided backward function and calls it during backpropagation.
 *
 * @see register_custom_op_with_backward()
 */
/**
 * @brief Fused Linear+ReLU backward pass.
 *
 * Combines the backward passes of MatMul and ReLU into a single operation.
 * Given forward: z = ReLU(W @ x + b), the backward is:
 *   grad_input = W^T @ (grad_output * (z > 0))
 * where the ReLU mask is computed from the forward output.
 *
 * This saves one intermediate tensor (the pre-ReLU activation) and
 * one kernel launch compared to separate MatMul + ReLU backward.
 */
class FusedLinearReLUBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "FusedLinearReLUBackward"; }

    void set_relu_output(Tensor output) { relu_output_ = std::move(output); }

private:
    Tensor relu_output_;  // Output of ReLU (for computing mask in backward)
};

class CustomOpBackward : public Function {
public:
    explicit CustomOpBackward(CustomBackwardFn backward_fn)
        : backward_fn_(std::move(backward_fn)) {}

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;
    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "CustomOpBackward"; }

private:
    CustomBackwardFn backward_fn_;
};

} // namespace tenzor
