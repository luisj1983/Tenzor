/**
 * @file function.hpp
 * @brief Autograd function interface and built-in operations
 *
 * Defines the Function base class for autograd operations and
 * provides implementations for common differentiable operations.
 */

#pragma once

#include <memory>
#include <vector>
#include "../core/tensor.hpp"
#include "variable.hpp"

namespace tenzor {

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
    virtual ~Function() = default;

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
     *
     * @return Vector of saved tensors
     */
    auto saved_tensors() const -> const std::vector<Tensor>&;

protected:
    std::vector<Tensor> saved_tensors_;                             ///< Tensors saved for backward
    std::vector<std::shared_ptr<Function>> next_functions_;         ///< Chained gradient functions
    std::vector<Variable> input_variables_;                          ///< Input variables for gradient accumulation (stored by value)
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
};

/**
 * @brief ReLU activation gradient function.
 *
 * Implements forward and backward for ReLU (Rectified Linear Unit) activation.
 *
 * Forward: y = max(0, x)
 * Backward: dL/dx = dL/dy * (x > 0)
 *
 * @note Input is saved to determine which elements to pass gradient through.
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = relu(x);  // Applies ReLU with ReLUBackward
 * @endcode
 */
class ReLUBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};

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
private:
    int64_t input_h_;   ///< Input height
    int64_t input_w_;   ///< Input width
    int64_t output_h_;  ///< Output height
    int64_t output_w_;  ///< Output width
};

} // namespace tenzor
