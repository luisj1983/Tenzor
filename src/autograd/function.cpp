#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/error.hpp"
#include <cmath>

namespace tenzor {

// Global activation offload flag
static std::atomic<bool> g_activation_offload_enabled{false};

void set_activation_offload(bool enabled) {
    g_activation_offload_enabled.store(enabled, std::memory_order_release);
}

bool activation_offload_enabled() {
    return g_activation_offload_enabled.load(std::memory_order_acquire);
}

auto Function::set_next_functions(std::vector<std::shared_ptr<Function>> funcs) -> void {
    next_functions_ = std::move(funcs);
}

auto Function::next_functions() const -> const std::vector<std::shared_ptr<Function>>& {
    return next_functions_;
}

auto Function::set_input_variables(std::vector<Variable> inputs) -> void {
    input_variables_ = std::move(inputs);
}

auto Function::input_variables() const -> const std::vector<Variable>& {
    return input_variables_;
}

auto Function::save_for_backward(std::vector<Tensor> tensors) -> void {
    if (activation_offload_enabled() && !tensors.empty()) {
        // Check if any tensor is on a GPU device
        auto dev = tensors[0].device();
        if (dev.type != Device::Type::CPU) {
            offloaded_device_ = dev;
            tensors_offloaded_ = true;
            for (auto& t : tensors) {
                t = t.to(Device::cpu());
            }
        }
    }
    saved_tensors_ = std::move(tensors);
}

auto Function::saved_tensors() const -> const std::vector<Tensor>& {
    if (tensors_offloaded_) {
        reload_saved_tensors();
    }
    return saved_tensors_;
}

void Function::reload_saved_tensors() const {
    if (!tensors_offloaded_) return;
    for (auto& t : saved_tensors_) {
        t = t.to(offloaded_device_);
    }
    tensors_offloaded_ = false;
}

auto Function::save_variables_for_backward(std::vector<Variable> variables) -> void {
    saved_variables_ = std::move(variables);
}

auto Function::saved_variables() const -> const std::vector<Variable>& {
    return saved_variables_;
}

auto Function::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Default implementation: extract Tensors, call backward(), wrap results back
    // This provides no higher-order gradient support but ensures backward compatibility
    std::vector<Tensor> tensor_grads;
    tensor_grads.reserve(grad_outputs.size());
    for (auto& var : grad_outputs) {
        tensor_grads.push_back(var.tensor());
    }

    auto result_tensors = backward(tensor_grads);

    std::vector<Variable> result_vars;
    result_vars.reserve(result_tensors.size());
    for (auto& t : result_tensors) {
        result_vars.emplace_back(t, false);
    }
    return result_vars;
}

// Helper function to reduce gradient Variable along broadcasted dimensions (for create_graph)
static auto reduce_grad_var_for_broadcasting(const Variable& grad, const std::vector<int64_t>& target_shape) -> Variable {
    auto grad_shape_vec = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());

    // If shapes match, no reduction needed
    if (grad_shape_vec == target_shape) {
        return grad;
    }

    auto result = grad;

    // Handle size difference (prepended dimensions in grad)
    int64_t ndim_diff = static_cast<int64_t>(grad_shape_vec.size()) - static_cast<int64_t>(target_shape.size());

    if (ndim_diff > 0) {
        // grad has MORE dimensions than target - sum along prepended dimensions
        for (int64_t i = 0; i < ndim_diff; ++i) {
            result = tenzor::sum(result, 0, false);
        }
    } else if (ndim_diff < 0) {
        throw std::runtime_error(
            "Autograd bug: gradient has fewer dimensions (" +
            std::to_string(grad_shape_vec.size()) + ") than target shape (" +
            std::to_string(target_shape.size()) + ")");
    }

    // Now result and target should have same ndim
    // Sum along dimensions that were broadcasted (size 1 in target but > 1 in result)
    auto result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && result_shape_vec[i] > 1) {
            result = tenzor::sum(result, static_cast<int64_t>(i), true);
            result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
        }
    }

    // Final reshape to exact target shape
    if (result_shape_vec != target_shape) {
        result = tenzor::reshape(result, target_shape);
    }

    return result;
}

// Helper function to reduce gradient along broadcasted dimensions
static auto reduce_grad_for_broadcasting(const Tensor& grad, const std::vector<int64_t>& target_shape) -> Tensor {
    auto grad_shape_vec = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());

    // If shapes match, no reduction needed
    if (grad_shape_vec == target_shape) {
        return grad;
    }

    auto result = grad;

    // Handle size difference (prepended dimensions in grad)
    int64_t ndim_diff = static_cast<int64_t>(grad_shape_vec.size()) - static_cast<int64_t>(target_shape.size());

    if (ndim_diff > 0) {
        // grad has MORE dimensions than target - sum along prepended dimensions
        for (int64_t i = 0; i < ndim_diff; ++i) {
            result = tenzor::sum(result, 0, false);  // Sum and remove dimension
        }
    } else if (ndim_diff < 0) {
        // grad has FEWER dimensions than target — this indicates an autograd graph bug
        // (gradient should never have fewer dimensions than the variable it flows to)
        throw std::runtime_error(
            "Autograd bug: gradient has fewer dimensions (" +
            std::to_string(grad_shape_vec.size()) + ") than target shape (" +
            std::to_string(target_shape.size()) + ")");
    }

    // Now result and target should have same ndim
    // Sum along dimensions that were broadcasted (size 1 in target but > 1 in result)
    auto result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && result_shape_vec[i] > 1) {
            result = tenzor::sum(result, static_cast<int64_t>(i), true);  // Keep dim as size 1
            result_shape_vec = std::vector<int64_t>(result.shape().begin(), result.shape().end());
        }
    }

    // Final reshape to exact target shape (handle keepdim=true above)
    if (result_shape_vec != target_shape) {
        result = reshape(result, target_shape);
    }

    return result;
}

// AddBackward implementation
auto AddBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = add(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto AddBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // Fast path: same shape (common case for residual connections)
    // Avoids function call overhead and vector comparisons
    if (input_shape_a_ == input_shape_b_) {
        auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        if (grad_shape == input_shape_a_) {
            // Both inputs have same shape as gradient - just return gradient twice
            return {grad, grad};
        }
    }

    // Slow path: handle broadcasting
    auto grad_a = reduce_grad_for_broadcasting(grad, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad, input_shape_b_);
    return {grad_a, grad_b};
}

auto AddBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    const auto& grad = grad_outputs[0];

    // For add: gradients are just the incoming gradient (possibly with broadcast reduction)
    // Since these are identity operations on the gradient, the Variable grad already has
    // its grad_fn set, so higher-order gradients flow through naturally
    if (input_shape_a_ == input_shape_b_) {
        auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        if (grad_shape == input_shape_a_) {
            return {grad, grad};
        }
    }

    auto grad_a = reduce_grad_var_for_broadcasting(grad, input_shape_a_);
    auto grad_b = reduce_grad_var_for_broadcasting(grad, input_shape_b_);
    return {grad_a, grad_b};
}

// SubBackward implementation
auto SubBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = sub(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto SubBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(a-b)/da = 1, d(a-b)/db = -1
    // Handle broadcasting
    auto grad_a = reduce_grad_for_broadcasting(grad_outputs[0], input_shape_a_);
    auto grad_b_unreduced = neg(grad_outputs[0]);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

auto SubBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(a-b)/da = 1, d(a-b)/db = -1
    // Use Variable operations so negation is tracked for higher-order gradients
    auto grad_a = reduce_grad_var_for_broadcasting(grad_outputs[0], input_shape_a_);
    auto grad_b_unreduced = tenzor::neg(grad_outputs[0]);  // Variable neg - tracked by autograd
    auto grad_b = reduce_grad_var_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

// MulBackward implementation
auto MulBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor(), inputs[1].tensor()};
    // Save input shapes for broadcasting-aware backward pass
    input_shape_a_ = std::vector<int64_t>(inputs[0].shape().begin(), inputs[0].shape().end());
    input_shape_b_ = std::vector<int64_t>(inputs[1].shape().begin(), inputs[1].shape().end());

    auto result = mul(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto MulBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(a*b)/da = b, d(a*b)/db = a
    // Handle broadcasting
    auto grad_a_unreduced = mul(grad_outputs[0], saved_tensors_[1]);
    auto grad_b_unreduced = mul(grad_outputs[0], saved_tensors_[0]);

    auto grad_a = reduce_grad_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);

    return {grad_a, grad_b};
}

auto MulBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(a*b)/da = b, d(a*b)/db = a
    // Use saved Variables if available, otherwise wrap saved Tensors
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        // Wrap raw tensors - no grad tracking but still works for first-order
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }

    // Use Variable operations so multiplication is tracked for higher-order gradients
    auto grad_a_unreduced = grad_outputs[0] * saved_b;
    auto grad_b_unreduced = grad_outputs[0] * saved_a;

    auto grad_a = reduce_grad_var_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_var_for_broadcasting(grad_b_unreduced, input_shape_b_);

    return {grad_a, grad_b};
}

// DivBackward implementation
auto DivBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor(), inputs[1].tensor()};
    auto result = div(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto DivBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(a/b)/da = 1/b, d(a/b)/db = -a/(b^2)
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];

    auto grad_a = div(grad_outputs[0], b);
    // grad_b = -a / (b^2) * grad_output = -(a * grad_output) / (b * b)
    auto grad_b = neg(div(mul(a, grad_outputs[0]), mul(b, b)));
    return {grad_a, grad_b};
}

auto DivBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(a/b)/da = 1/b, d(a/b)/db = -a/(b^2)
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }

    // Use Variable operations for higher-order gradient tracking
    auto grad_a = grad_outputs[0] / saved_b;
    // grad_b = -(a * grad_output) / (b * b)
    auto grad_b = tenzor::neg((saved_a * grad_outputs[0]) / (saved_b * saved_b));
    return {grad_a, grad_b};
}

// MatMulBackward implementation
auto MatMulBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor(), inputs[1].tensor()};
    auto result = matmul(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto MatMulBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // For C = A @ B:
    // dL/dA = dL/dC @ B.T
    // dL/dB = A.T @ dL/dC
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad_out = grad_outputs[0];

    // Get the number of dimensions
    auto a_ndim = a.shape().size();
    auto b_ndim = b.shape().size();

    // For 2D matrices: grad_a = grad_out @ b.T, grad_b = a.T @ grad_out
    auto b_t = transpose(b, b_ndim - 2, b_ndim - 1);
    auto a_t = transpose(a, a_ndim - 2, a_ndim - 1);

    auto grad_a = matmul(grad_out, b_t);
    auto grad_b = matmul(a_t, grad_out);

    return {grad_a, grad_b};
}

auto MatMulBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // For C = A @ B:
    // dL/dA = dL/dC @ B.T
    // dL/dB = A.T @ dL/dC
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }

    const auto& grad_out = grad_outputs[0];

    // Use Variable operations for higher-order gradient tracking
    auto a_ndim = saved_a.shape().size();
    auto b_ndim = saved_b.shape().size();

    auto b_t = tenzor::transpose(saved_b, b_ndim - 2, b_ndim - 1);
    auto a_t = tenzor::transpose(saved_a, a_ndim - 2, a_ndim - 1);

    auto grad_a = tenzor::matmul(grad_out, b_t);
    auto grad_b = tenzor::matmul(a_t, grad_out);

    return {grad_a, grad_b};
}

// LinearBackward implementation
auto LinearBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // This is typically not called - autograd::linear() handles forward directly
    // But implement for completeness
    // inputs[0] = x (batch_size, in_features)
    // inputs[1] = W (out_features, in_features)
    // inputs[2] = b (out_features)
    saved_tensors_ = {inputs[0].tensor(), inputs[1].tensor(), inputs[2].tensor()};

    // Compute: y = x @ W.T + b
    auto x = inputs[0].tensor();
    auto w = inputs[1].tensor();
    auto b = inputs[2].tensor();

    // Transpose weight: (out_features, in_features) -> (in_features, out_features)
    auto w_t = transpose(w, 0, 1);

    // Matrix multiplication: (batch, in) @ (in, out) -> (batch, out)
    auto matmul_result = matmul(x, w_t);

    // Add bias (broadcasts automatically)
    auto result = add(matmul_result, b);

    return {Variable(result, true)};
}

auto LinearBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // For y = x @ W.T + b:
    // dL/dx = dL/dy @ W          -> (batch, out) @ (out, in) = (batch, in)
    // dL/dW = dL/dy.T @ x        -> (out, batch) @ (batch, in) = (out, in)
    // dL/db = sum(dL/dy, dim=0)  -> reduce batch dimension

    const auto& x = saved_tensors_[0];       // (batch, in)
    const auto& w = saved_tensors_[1];       // (out, in)
    const auto& grad_out = grad_outputs[0];  // (batch, out)

    // Use optimized LinearBackward kernel for Float32/Float64
    // Fall back to tensor ops for Float16 and other types
    if (grad_out.dtype() == DType::Float32 || grad_out.dtype() == DType::Float64) {
        std::vector<Tensor> inputs = {grad_out, x, w};
        return dispatch<OpId::LinearBackward>(inputs);
    }

    // Fallback for Float16 and other types using tensor operations
    // grad_input = grad_out @ W
    auto grad_x = matmul(grad_out, w);

    // grad_weight = grad_out.T @ x
    auto grad_out_t = transpose(grad_out, 0, 1);  // (out, batch)
    auto grad_w = matmul(grad_out_t, x);          // (out, in)

    // grad_bias = sum(grad_out, dim=0)
    auto grad_b = tenzor::sum(grad_out, 0, false);  // (out,)

    return {grad_x, grad_w, grad_b};
}

auto LinearBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // For y = x @ W.T + b:
    // dL/dx = dL/dy @ W
    // dL/dW = dL/dy.T @ x
    // dL/db = sum(dL/dy, dim=0)
    Variable saved_x, saved_w;
    if (has_saved_variables()) {
        saved_x = saved_variables_[0];
        saved_w = saved_variables_[1];
    } else {
        saved_x = Variable(saved_tensors_[0], false);
        saved_w = Variable(saved_tensors_[1], false);
    }
    const auto& grad_out = grad_outputs[0];

    auto grad_x = tenzor::matmul(grad_out, saved_w);
    auto grad_out_t = tenzor::transpose(grad_out, 0, 1);
    auto grad_w = tenzor::matmul(grad_out_t, saved_x);
    auto grad_b = tenzor::sum(grad_out, 0, false);
    return {grad_x, grad_w, grad_b};
}

// SumBackward implementation
auto SumBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor()};
    auto result = sum(inputs[0].tensor(), dim_, keepdim_);
    return {Variable(result, true)};
}

auto SumBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "SumBackward: cannot compute gradient of sum over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Reduced all dimensions - broadcast scalar tensor back to original shape
        // Use pure tensor operations (no CPU transfers) - backend agnostic!
        auto grad = grad_output;

        // Ensure grad is a 0-d tensor (may be 1-element tensor from some reductions)
        if (grad.ndim() > 0) {
            grad = reshape(grad, {});
        }

        // Use expand() to broadcast the scalar to input shape natively on device
        auto result = expand(grad, input_shape_vec);

        return {result};
    } else {
        // Dimension-specific reduction backward using unsqueeze + expand
        // expand() now uses native CUDA implementation - no device transfers!
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }

        return {expand(grad, input_shape_vec)};
    }
}

auto SumBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Sum backward is just expanding the gradient back to input shape.
    // This operation doesn't depend on saved inputs, so we can use Tensor-level
    // expand/reshape and wrap the result. The gradient Variable itself carries
    // its computation graph for higher-order differentiation.
    const auto& input = saved_tensors_[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "SumBackward: cannot compute gradient of sum over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto grad_tensor = grad_outputs[0].tensor();

    if (!dim_.has_value()) {
        if (grad_tensor.ndim() > 0) {
            grad_tensor = reshape(grad_tensor, {});
        }
        auto result = expand(grad_tensor, input_shape_vec);
        // Wrap as Variable preserving requires_grad from the incoming gradient
        return {Variable(result, grad_outputs[0].requires_grad())};
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_tensor;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }
        auto result = expand(grad, input_shape_vec);
        return {Variable(result, grad_outputs[0].requires_grad())};
    }
}

// MeanBackward implementation
auto MeanBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor()};
    auto result = mean(inputs[0].tensor(), dim_, keepdim_);
    return {Variable(result, true)};
}

auto MeanBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    // Calculate the number of elements that were averaged
    int64_t n_elements = 1;
    if (dim_.has_value()) {
        n_elements = input.shape()[dim_.value()];
    } else {
        n_elements = input.numel();
    }

    TENZOR_CHECK_SHAPE(n_elements > 0,
        "MeanBackward: cannot compute mean of empty tensor (0 elements)");

    // Use double for scale calculation to preserve precision for Float64 tensors
    double scale = 1.0 / static_cast<double>(n_elements);
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Reduced all dimensions - broadcast scalar tensor back to original shape
        // Use pure tensor operations (no CPU transfers) - backend agnostic!
        auto grad = grad_output;

        // Ensure grad is a 0-d tensor (may be 1-element tensor from some reductions)
        if (grad.ndim() > 0) {
            grad = reshape(grad, {});
        }

        // Expand the scalar to input shape natively on device
        auto expanded = expand(grad, input_shape_vec);

        // Scale by 1/N using backend-agnostic tensor multiplication
        // Create scalar tensor with same dtype and device as expanded gradient
        // Use double overload of full() to preserve precision for Float64
        auto scale_tensor = full({}, scale, expanded.dtype(), expanded.device());

        auto result = mul(expanded, scale_tensor);

        return {result};
    } else {
        // Dimension-specific reduction backward
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }

        // Expand to input shape - works on CPU, transfers back if needed
        auto expanded = expand(grad, input_shape_vec);

        // Scale the expanded gradient using native Tensor multiplication
        // This now uses CUDA broadcasting automatically
        // Use expanded.dtype() to ensure dtypes match for element-wise operations
        // Use double overload of full() to preserve precision for Float64
        auto scale_tensor = full(input_shape_vec, scale, expanded.dtype(), expanded.device());
        return {mul(expanded, scale_tensor)};
    }
}

auto MeanBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Mean backward: expand gradient and scale by 1/N.
    // The scaling by 1/N uses Variable::operator*(double) which IS tracked by autograd
    // for higher-order gradient support.
    const auto& input = saved_tensors_[0];

    int64_t n_elements = 1;
    if (dim_.has_value()) {
        n_elements = input.shape()[dim_.value()];
    } else {
        n_elements = input.numel();
    }

    TENZOR_CHECK_SHAPE(n_elements > 0,
        "MeanBackward: cannot compute mean of empty tensor (0 elements)");

    double scale = 1.0 / static_cast<double>(n_elements);
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Scale the gradient Variable - this uses Variable::operator*(double) which
    // builds autograd graph when create_graph is active
    auto scaled_grad = grad_outputs[0] * scale;

    // Now expand to input shape using Tensor-level operations
    auto grad_tensor = scaled_grad.tensor();

    if (!dim_.has_value()) {
        if (grad_tensor.ndim() > 0) {
            grad_tensor = reshape(grad_tensor, {});
        }
        auto result = expand(grad_tensor, input_shape_vec);
        return {Variable(result, scaled_grad.requires_grad())};
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_tensor;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }
        auto result = expand(grad, input_shape_vec);
        return {Variable(result, scaled_grad.requires_grad())};
    }
}

// LogBackward implementation
auto LogBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor()};
    auto result = log(inputs[0].tensor());
    return {Variable(result, true)};
}

auto LogBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(log(x))/dx = 1/x
    const auto& input = saved_tensors_[0];
    auto grad_input = div(grad_outputs[0], input);
    return {grad_input};
}

auto LogBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(log(x))/dx = 1/x
    // Use Variable division for higher-order gradient tracking
    Variable saved_input(saved_tensors_[0], false);
    return {grad_outputs[0] / saved_input};
}

// ExpBackward implementation
auto ExpBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = exp(inputs[0].tensor());
    saved_tensors_ = {result};  // Save output for backward
    return {Variable(result, true)};
}

auto ExpBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(exp(x))/dx = exp(x)
    const auto& output = saved_tensors_[0];
    auto grad_input = mul(grad_outputs[0], output);
    return {grad_input};
}

auto ExpBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(exp(x))/dx = exp(x) = saved output
    // Use Variable multiplication for higher-order gradient tracking
    Variable saved_output(saved_tensors_[0], false);
    return {grad_outputs[0] * saved_output};
}

// NegBackward implementation
auto NegBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = neg(inputs[0].tensor());
    return {Variable(result, true)};
}

auto NegBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(-x)/dx = -1
    return {neg(grad_outputs[0])};
}

auto NegBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(-x)/dx = -1
    // Use Variable neg for higher-order gradient tracking
    return {tenzor::neg(grad_outputs[0])};
}

// LogSoftmaxBackward implementation
auto LogSoftmaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim_);
    std::vector<Tensor> input_tensors = {inputs[0].tensor()};
    auto result = dispatch(OpId::LogSoftmax, input_tensors, attrs)[0];

    // Save output for backward
    saved_tensors_ = {result};

    return {Variable(result, true)};
}

auto LogSoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Use backend's log_softmax_backward kernel
    const auto& output = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim_);
    std::vector<Tensor> inputs = {grad_output, output};
    auto grad_input = dispatch(OpId::LogSoftmaxBackward, inputs, attrs)[0];

    return {grad_input};
}

auto LogSoftmaxBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dx_i = dL/dy_i - exp(y_i) * sum_j(dL/dy_j)
    // Use Variable operations for higher-order gradient tracking
    Variable output_var(saved_tensors_[0], false);
    auto grad_sum = tenzor::sum(grad_outputs[0], dim_, true);
    auto softmax_output = tenzor::exp(output_var);
    auto grad_input = grad_outputs[0] - softmax_output * grad_sum;
    return {grad_input};
}

// SoftmaxBackward implementation
auto SoftmaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim_);
    std::vector<Tensor> input_tensors = {inputs[0].tensor()};
    auto result = dispatch(OpId::Softmax, input_tensors, attrs)[0];

    // Save output for backward
    saved_tensors_ = {result};

    return {Variable(result, true)};
}

auto SoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Use backend-optimized softmax_backward kernel via dispatch
    const auto& output = saved_tensors_[0];  // y = softmax(x)
    const auto& grad_output = grad_outputs[0];  // dL/dy

    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim_);
    std::vector<Tensor> inputs = {grad_output, output};
    auto grad_input = dispatch(OpId::SoftmaxBackward, inputs, attrs)[0];

    return {grad_input};
}

auto SoftmaxBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
    // Use Variable operations for higher-order gradient tracking
    Variable output_var(saved_tensors_[0], false);
    auto dot_product = tenzor::sum(grad_outputs[0] * output_var, dim_, true);
    auto grad_input = output_var * (grad_outputs[0] - dot_product);
    return {grad_input};
}

// AbsBackward implementation
auto AbsBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor()};
    auto result = abs(inputs[0].tensor());
    return {Variable(result, true)};
}

auto AbsBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(abs(x))/dx = sign(x)
    const auto& input = saved_tensors_[0];
    auto grad_input = mul(grad_outputs[0], sign(input));
    return {grad_input};
}

auto AbsBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(abs(x))/dx = sign(x)
    // sign is non-differentiable, so compute it at Tensor level
    auto sign_mask = sign(saved_tensors_[0]);
    Variable sign_var(sign_mask, false);
    return {grad_outputs[0] * sign_var};
}

// ClampBackward implementation
auto ClampBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor()};
    auto result = clamp(inputs[0].tensor(), min_, max_);
    return {Variable(result, true)};
}

auto ClampBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(clamp(x, min, max))/dx = 1 if min <= x <= max else 0
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    // Create mask: 1 where min <= x <= max, 0 otherwise
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
    auto zeros_tensor = zeros(input_shape_vec, input.dtype(), input.device());

    // Check if input >= min
    auto min_tensor = full(input_shape_vec, min_, input.dtype(), input.device());
    auto max_tensor = full(input_shape_vec, max_, input.dtype(), input.device());

    // Mask = (input >= min) & (input <= max)
    // For now, use clamp and compare approach
    auto clamped = clamp(input, min_, max_);

    // grad = grad_output where input == clamped else 0
    // This is approximately: mask = 1 - abs(sign(input - clamped))
    auto diff = sub(input, clamped);
    auto diff_sign = abs(sign(diff));
    auto mask = sub(ones_tensor, diff_sign);

    return {mul(grad_output, mask)};
}

auto ClampBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(clamp(x, min, max))/dx = 1 if min <= x <= max, else 0
    // The mask is non-differentiable, compute at Tensor level
    const auto& input = saved_tensors_[0];
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
    auto clamped = clamp(input, min_, max_);
    auto diff = sub(input, clamped);
    auto diff_sign = abs(sign(diff));
    auto mask = sub(ones_tensor, diff_sign);
    Variable mask_var(mask, false);
    return {grad_outputs[0] * mask_var};
}

// MaxBackward implementation
auto MaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = max(inputs[0].tensor(), dim_, keepdim_);
    // Save both input and output for backward
    saved_tensors_ = {inputs[0].tensor(), result};
    return {Variable(result, true)};
}

auto MaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "MaxBackward: cannot compute gradient of max over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Global max: gradient flows only to the maximum element
        // Create mask where input == output (broadcasted)

        // Reshape scalar output to match input dimensions before expanding
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        // Create mask where input == output (within epsilon)
        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        // Select epsilon appropriate for the tensor's precision
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        // Convert boolean mask to float for gradient computation
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto zeros_tensor = zeros(input_shape_vec, input.dtype(), input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        // Normalize mask by tie count so gradient is split among tied elements
        auto tie_count = sum(mask);
        mask = div(mask, tie_count);

        // Broadcast grad_output to input shape
        // FIX: grad_output is also scalar, need to reshape before expanding
        auto grad_reshaped = grad_output;
        if (grad_output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_reshaped = reshape(grad_output, ones_shape);
        }
        auto grad_broadcasted = expand(grad_reshaped, input_shape_vec);

        return {mul(grad_broadcasted, mask)};
    } else {
        // Dimension-specific max
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        auto out = output;

        // Unsqueeze if needed
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
            out = unsqueeze(out, dim);
        }

        // Expand to input shape
        auto out_expanded = expand(out, input_shape_vec);
        auto grad_expanded = expand(grad, input_shape_vec);

        // Create mask where input == max_value
        auto diff = sub(input, out_expanded);
        auto abs_diff = abs(diff);

        // Select epsilon appropriate for the tensor's precision
        double eps_val2;
        switch (input.dtype()) {
            case DType::Float64:  eps_val2 = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val2 = 1e-3; break;
            default:              eps_val2 = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val2, input.dtype(), input.device());
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto scaled_diff = div(abs_diff, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        // Normalize mask by tie count along dim so gradient is split among tied elements
        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        mask = div(mask, tie_count);

        return {mul(grad_expanded, mask)};
    }
}

// ReshapeBackward implementation
auto ReshapeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("ReshapeBackward::forward should not be called");
}

auto ReshapeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Reshape gradient back to input shape and ensure contiguity
    // Reshape may create non-contiguous views, which can cause issues in element-wise operations
    auto grad_input = reshape(grad_outputs[0], input_shape_).contiguous();
    return {grad_input};
}

auto ReshapeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = reshape(grad_outputs[0].tensor(), input_shape_).contiguous();
    return {Variable(result, grad_outputs[0].requires_grad())};
}

// PermuteBackward implementation
auto PermuteBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("PermuteBackward::forward should not be called");
}

auto PermuteBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Apply inverse permutation to gradient and ensure contiguity
    // Permute creates non-contiguous views, which can cause issues in element-wise operations
    auto grad_input = permute(grad_outputs[0], inv_dims_).contiguous();
    return {grad_input};
}

auto PermuteBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = permute(grad_outputs[0].tensor(), inv_dims_).contiguous();
    return {Variable(result, grad_outputs[0].requires_grad())};
}

// TransposeBackward implementation
auto TransposeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("TransposeBackward::forward should not be called");
}

auto TransposeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Transpose is its own inverse, so apply same transpose to gradient
    auto grad_input = transpose(grad_outputs[0], dim0_, dim1_).contiguous();
    return {grad_input};
}

auto TransposeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = transpose(grad_outputs[0].tensor(), dim0_, dim1_).contiguous();
    return {Variable(result, grad_outputs[0].requires_grad())};
}

// RollBackward implementation
auto RollBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("RollBackward::forward should not be called");
}

auto RollBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Roll backward is roll with negative shift
    auto grad_input = roll(grad_outputs[0], -shifts_, dim_);
    return {grad_input};
}

auto RollBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = roll(grad_outputs[0].tensor(), -shifts_, dim_);
    return {Variable(result, grad_outputs[0].requires_grad())};
}

// SqueezeBackward implementation
auto SqueezeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("SqueezeBackward::forward should not be called");
}

auto SqueezeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Unsqueeze gradient back to original shape
    auto grad_input = unsqueeze(grad_outputs[0], dim_);
    return {grad_input};
}

auto SqueezeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = unsqueeze(grad_outputs[0].tensor(), dim_);
    return {Variable(result, grad_outputs[0].requires_grad())};
}

// BmmBackward implementation
auto BmmBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor(), inputs[1].tensor()};
    auto result = bmm(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto BmmBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // For C = bmm(A, B):
    // A: (batch, n, m), B: (batch, m, p), C: (batch, n, p)
    // grad_output: (batch, n, p)
    //
    // Backward gradients:
    // grad_a = grad_output @ B^T = (batch, n, p) @ (batch, p, m) = (batch, n, m)
    // grad_b = A^T @ grad_output = (batch, m, n) @ (batch, n, p) = (batch, m, p)

    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    // Transpose last two dimensions: (batch, m, p) -> (batch, p, m)
    auto b_transposed = permute(b, {0, 2, 1});

    // grad_a = grad_output @ b^T
    auto grad_a = bmm(grad_output, b_transposed);

    // Transpose a: (batch, n, m) -> (batch, m, n)
    auto a_transposed = permute(a, {0, 2, 1});

    // grad_b = a^T @ grad_output
    auto grad_b = bmm(a_transposed, grad_output);

    return {grad_a, grad_b};
}

auto BmmBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // For C = bmm(A, B):
    // grad_a = grad_output @ B^T
    // grad_b = A^T @ grad_output
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }
    const auto& grad_out = grad_outputs[0];
    auto b_t = tenzor::transpose(saved_b, saved_b.shape().size() - 2, saved_b.shape().size() - 1);
    auto a_t = tenzor::transpose(saved_a, saved_a.shape().size() - 2, saved_a.shape().size() - 1);
    auto grad_a = tenzor::bmm(grad_out, b_t);
    auto grad_b = tenzor::bmm(a_t, grad_out);
    return {grad_a, grad_b};
}

// CatBackward implementation
auto CatBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Convert Variables to Tensors for concatenation
    std::vector<Tensor> tensors;
    tensors.reserve(inputs.size());
    for (const auto& var : inputs) {
        tensors.push_back(var.tensor());
    }

    auto result = cat(tensors, dim_);
    return {Variable(result, true)};
}

auto CatBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Split gradient back along concatenation dimension
    // grad_output shape: [..., sum(split_sizes), ...]
    // Need to split into gradients of shape [..., split_sizes[i], ...]

    const auto& grad_output = grad_outputs[0];
    std::vector<Tensor> grad_inputs;
    grad_inputs.reserve(split_sizes_.size());

    int64_t offset = 0;
    for (int64_t split_size : split_sizes_) {
        // Slice grad_output from offset to offset+split_size along dim_
        auto grad_slice = slice(grad_output, dim_, offset, offset + split_size);
        grad_inputs.push_back(grad_slice);
        offset += split_size;
    }

    return grad_inputs;
}

auto CatBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    const auto& grad_output = grad_outputs[0];
    std::vector<Variable> grad_inputs;
    grad_inputs.reserve(split_sizes_.size());
    int64_t offset = 0;
    for (int64_t split_size : split_sizes_) {
        auto grad_slice = slice(grad_output.tensor(), dim_, offset, offset + split_size);
        grad_inputs.emplace_back(grad_slice, grad_output.requires_grad());
        offset += split_size;
    }
    return grad_inputs;
}

// SliceBackward implementation
auto SliceBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = slice(inputs[0].tensor(), dim_, start_, end_, step_);
    return {Variable(result, true)};
}

auto SliceBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_output = grad_outputs[0];

    // Create zero gradient tensor with original input shape
    auto grad_input = zeros(input_shape_, grad_output.dtype(), grad_output.device());

    // Build index tensor for scatter operation
    // Index tensor must have same shape as grad_output
    int64_t slice_size = grad_output.shape()[dim_];
    int64_t total_elements = grad_output.numel();

    // Create index tensor with same shape as grad_output
    auto index_shape = std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end());
    auto index = zeros(index_shape, DType::Int64, Device::cpu());

    // Fill index tensor on CPU
    int64_t* index_ptr = index.data<int64_t>();

    // Calculate stride for the sliced dimension
    int64_t dim_stride = 1;
    for (int64_t d = dim_ + 1; d < grad_output.ndim(); ++d) {
        dim_stride *= grad_output.shape()[d];
    }

    // Fill index tensor: each element along dim_ gets mapped to (start_ + pos * step_)
    for (int64_t i = 0; i < total_elements; ++i) {
        int64_t pos_in_dim = (i / dim_stride) % slice_size;
        index_ptr[i] = start_ + pos_in_dim * step_;
    }

    // Transfer to target device if needed
    if (grad_output.device() != Device::cpu()) {
        index = index.to(grad_output.device());
    }

    // Use scatter to place gradients - dispatches to appropriate backend
    grad_input = scatter(grad_input, dim_, index, grad_output);

    return {grad_input};
}

auto SliceBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Same as tensor-level backward but wrapping with requires_grad=true
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

// UpsampleBilinearBackward implementation
auto UpsampleBilinearBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Save input tensor for backward pass
    saved_tensors_ = {inputs[0].tensor()};

    // Forward computation is done externally in the wrapper function
    // This method is not typically called directly
    throw std::runtime_error("UpsampleBilinearBackward::forward should not be called directly");
}

auto UpsampleBilinearBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Distribute gradients from upsampled output back to input size
    // For nearest neighbor upsampling: each output pixel's gradient goes to its source input pixel

    const auto& grad_output_orig = grad_outputs[0];
    const auto& shape = grad_output_orig.shape();

    if (shape.size() != 4) {
        throw std::runtime_error("UpsampleBilinearBackward: Expected 4D gradient tensor (N, C, H, W)");
    }

    // Remember original dtype and device for output conversion
    DType original_dtype = grad_output_orig.dtype();
    Device original_device = grad_output_orig.device();

    // Convert to Float32 on CPU for computation
    Tensor grad_output = grad_output_orig.to(Device::cpu());
    if (grad_output.dtype() != DType::Float32) {
        grad_output = grad_output.to(DType::Float32);
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_out = shape[2];
    int64_t W_out = shape[3];

    // Create gradient tensor for input (all zeros initially) in Float32
    auto grad_input = zeros({N, C, input_h_, input_w_}, DType::Float32, Device::cpu());

    // Calculate scaling factors (align_corners=false convention)
    float scale_h = static_cast<float>(input_h_) / static_cast<float>(output_h_);
    float scale_w = static_cast<float>(input_w_) / static_cast<float>(output_w_);

    // Distribute gradients using bilinear interpolation weights
    auto* grad_in_ptr = grad_input.data<float>();
    const auto* grad_out_ptr = grad_output.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H_out; ++h) {
                for (int64_t w = 0; w < W_out; ++w) {
                    // Map output pixel to input coordinate (align_corners=false)
                    float src_h = (h + 0.5f) * scale_h - 0.5f;
                    float src_w = (w + 0.5f) * scale_w - 0.5f;

                    // Bounding input pixels
                    int64_t h0 = static_cast<int64_t>(std::floor(src_h));
                    int64_t w0 = static_cast<int64_t>(std::floor(src_w));
                    int64_t h1 = h0 + 1;
                    int64_t w1 = w0 + 1;

                    // Interpolation weights from fractional part
                    float fh = src_h - h0;
                    float fw = src_w - w0;

                    float grad_val = grad_out_ptr[((n * C + c) * H_out + h) * W_out + w];
                    int64_t base = (n * C + c) * input_h_;

                    // Accumulate weighted gradient to each of the 4 neighbors
                    if (h0 >= 0 && h0 < input_h_ && w0 >= 0 && w0 < input_w_)
                        grad_in_ptr[(base + h0) * input_w_ + w0] += grad_val * (1.0f - fh) * (1.0f - fw);
                    if (h0 >= 0 && h0 < input_h_ && w1 >= 0 && w1 < input_w_)
                        grad_in_ptr[(base + h0) * input_w_ + w1] += grad_val * (1.0f - fh) * fw;
                    if (h1 >= 0 && h1 < input_h_ && w0 >= 0 && w0 < input_w_)
                        grad_in_ptr[(base + h1) * input_w_ + w0] += grad_val * fh * (1.0f - fw);
                    if (h1 >= 0 && h1 < input_h_ && w1 >= 0 && w1 < input_w_)
                        grad_in_ptr[(base + h1) * input_w_ + w1] += grad_val * fh * fw;
                }
            }
        }
    }

    // Convert back to original dtype and device
    if (grad_input.dtype() != original_dtype) {
        grad_input = grad_input.to(original_dtype);
    }
    if (grad_input.device() != original_device) {
        grad_input = grad_input.to(original_device);
    }

    return {grad_input};
}

} // namespace tenzor
