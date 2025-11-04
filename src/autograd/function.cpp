#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/dispatch.hpp"
#include <iostream>
#include <cmath>

namespace tenzor {

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
    saved_tensors_ = std::move(tensors);
}

auto Function::saved_tensors() const -> const std::vector<Tensor>& {
    return saved_tensors_;
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
        // grad has FEWER dimensions than target - broadcast by adding dimensions
        // This happens when gradient was reduced to scalar but target has shape
        // We need to broadcast the scalar to target shape
        return expand(result, target_shape);
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
    // Reduce gradients to match input shapes (handle broadcasting)
    auto grad_a = reduce_grad_for_broadcasting(grad_outputs[0], input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad_outputs[0], input_shape_b_);
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

// SumBackward implementation
auto SumBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor()};
    auto result = sum(inputs[0].tensor(), dim_, keepdim_);
    return {Variable(result, true)};
}

auto SumBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

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
        return {expand(grad, input_shape_vec)};
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

    float scale = 1.0f / static_cast<float>(n_elements);
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
        auto scale_tensor = full({}, scale, expanded.dtype(), expanded.device());
        return {mul(expanded, scale_tensor)};
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
        auto scale_tensor = full(input_shape_vec, scale, expanded.dtype(), expanded.device());
        return {mul(expanded, scale_tensor)};
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

// NegBackward implementation
auto NegBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = neg(inputs[0].tensor());
    return {Variable(result, true)};
}

auto NegBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(-x)/dx = -1
    return {neg(grad_outputs[0])};
}

// LogSoftmaxBackward implementation
auto LogSoftmaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim_);
    std::vector<Tensor> input_tensors = {inputs[0].tensor()};
    auto result = Dispatcher::dispatch("log_softmax", input_tensors, attrs)[0];

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
    auto grad_input = Dispatcher::dispatch("log_softmax_backward", inputs, attrs)[0];

    return {grad_input};
}

// SoftmaxBackward implementation
auto SoftmaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim_);
    std::vector<Tensor> input_tensors = {inputs[0].tensor()};
    auto result = Dispatcher::dispatch("softmax", input_tensors, attrs)[0];

    // Save output for backward
    saved_tensors_ = {result};

    return {Variable(result, true)};
}

auto SoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Softmax backward: dL/dx = y * (dL/dy - sum(dL/dy * y))
    const auto& output = saved_tensors_[0];  // y = softmax(x)
    const auto& grad_output = grad_outputs[0];  // dL/dy

    // Compute dL/dy * y (element-wise)
    auto grad_y_prod = mul(grad_output, output);

    // Sum along the softmax dimension
    auto grad_y_sum = tenzor::sum(grad_y_prod, dim_, true);

    // Compute dL/dy - sum(dL/dy * y) (broadcast)
    auto grad_centered = sub(grad_output, grad_y_sum);

    // Multiply by y to get final gradient
    auto grad_input = mul(grad_centered, output);

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

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Global max: gradient flows only to the maximum element
        // Create mask where input == output (broadcasted)
        auto output_expanded = expand(output, input_shape_vec);

        // mask = (input == output) ? 1 : 0
        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);

        // Small epsilon for floating point comparison
        auto epsilon = full(input_shape_vec, 1e-7f, input.dtype(), input.device());
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto zeros_tensor = zeros(input_shape_vec, input.dtype(), input.device());

        // mask = abs_diff < epsilon ? 1 : 0
        // Approximate using: 1 - clamp(abs_diff / epsilon, 0, 1)
        auto scaled_diff = div(abs_diff, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        // Broadcast grad_output to input shape
        auto grad_broadcasted = expand(grad_output, input_shape_vec);

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

        auto epsilon = full(input_shape_vec, 1e-7f, input.dtype(), input.device());
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto scaled_diff = div(abs_diff, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

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

// TransposeBackward implementation
auto TransposeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("TransposeBackward::forward should not be called");
}

auto TransposeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Transpose is its own inverse, so apply same transpose to gradient
    auto grad_input = transpose(grad_outputs[0], dim0_, dim1_).contiguous();
    return {grad_input};
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

// SliceBackward implementation
auto SliceBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = slice(inputs[0].tensor(), dim_, start_, end_, step_);
    return {Variable(result, true)};
}

auto SliceBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Create zero gradient tensor with original input shape
    const auto& grad_output = grad_outputs[0];
    auto grad_input = zeros(input_shape_, grad_output.dtype(), grad_output.device());

    // Scatter grad_output values back to the sliced positions using optimized implementation
    // This uses efficient memory access patterns and vectorization where possible

    // Calculate strides for both tensors
    auto grad_out_shape = grad_output.shape();
    auto ndim = input_shape_.size();

    std::vector<int64_t> in_strides(ndim);
    std::vector<int64_t> out_strides(ndim);

    in_strides[ndim - 1] = 1;
    out_strides[ndim - 1] = 1;
    for (int64_t i = ndim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * input_shape_[i + 1];
        out_strides[i] = out_strides[i + 1] * grad_out_shape[i + 1];
    }

    // Optimized scatter implementation using native memory operations
    auto scatter_gradients = [&]<typename T>(const T* grad_out_data, T* grad_in_data) {
        // Use iterative approach with optimized indexing for better cache locality
        int64_t total_elements = 1;
        for (int64_t dim_size : grad_out_shape) {
            total_elements *= dim_size;
        }

        // Process elements with optimized scatter pattern
        for (int64_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
            // Decompose linear index into multi-dimensional indices
            std::vector<int64_t> out_indices(ndim);
            int64_t remaining = linear_idx;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                out_indices[d] = remaining % grad_out_shape[d];
                remaining /= grad_out_shape[d];
            }

            // Map output indices to input indices
            std::vector<int64_t> in_indices(ndim);
            for (int64_t d = 0; d < static_cast<int64_t>(ndim); ++d) {
                if (d == dim_) {
                    // Sliced dimension: map back using start and step
                    in_indices[d] = start_ + out_indices[d] * step_;
                } else {
                    // Other dimensions: direct mapping
                    in_indices[d] = out_indices[d];
                }
            }

            // Calculate linear input index efficiently
            int64_t in_linear = 0;
            for (int64_t d = 0; d < static_cast<int64_t>(ndim); ++d) {
                in_linear += in_indices[d] * in_strides[d];
            }

            // Scatter gradient value
            grad_in_data[in_linear] = grad_out_data[linear_idx];
        }
    };

    // Dispatch based on dtype
    switch (grad_output.dtype()) {
        case DType::Float32:
            scatter_gradients(grad_output.data<float>(), grad_input.data<float>());
            break;
        case DType::Float64:
            scatter_gradients(grad_output.data<double>(), grad_input.data<double>());
            break;
        case DType::Float16: {
            // Float16 requires special handling - cast via uint16_t
            // For now, throw an error and plan for future implementation
            throw std::runtime_error("SliceBackward: Float16 support requires specialized implementation");
        }
        default:
            throw std::runtime_error("SliceBackward: Unsupported dtype. Supported types: Float32, Float64");
    }

    return {grad_input};
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

    std::cout << "[DEBUG] UpsampleBilinearBackward::backward() CALLED" << std::endl;

    const auto& grad_output = grad_outputs[0];
    const auto& shape = grad_output.shape();

    if (shape.size() != 4) {
        throw std::runtime_error("UpsampleBilinearBackward: Expected 4D gradient tensor (N, C, H, W)");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_out = shape[2];
    int64_t W_out = shape[3];

    std::cout << "[DEBUG] grad_output shape: [" << N << ", " << C << ", " << H_out << ", " << W_out << "]" << std::endl;
    std::cout << "[DEBUG] target input shape: [" << N << ", " << C << ", " << input_h_ << ", " << input_w_ << "]" << std::endl;

    // Create gradient tensor for input (all zeros initially)
    auto grad_input = zeros({N, C, input_h_, input_w_}, grad_output.dtype(), grad_output.device());

    // Calculate scaling factors (same as forward pass)
    float scale_h = static_cast<float>(input_h_) / output_h_;
    float scale_w = static_cast<float>(input_w_) / output_w_;

    std::cout << "[DEBUG] scale_h=" << scale_h << ", scale_w=" << scale_w << std::endl;

    // Accumulate gradients using nearest neighbor logic
    auto* grad_in_ptr = grad_input.data<float>();
    const auto* grad_out_ptr = grad_output.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H_out; ++h) {
                for (int64_t w = 0; w < W_out; ++w) {
                    // Find source input pixel (nearest neighbor)
                    int64_t in_h = static_cast<int64_t>(h * scale_h);
                    int64_t in_w = static_cast<int64_t>(w * scale_w);

                    in_h = std::min(in_h, input_h_ - 1);
                    in_w = std::min(in_w, input_w_ - 1);

                    // Accumulate gradient to source pixel
                    int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                    int64_t in_idx = ((n * C + c) * input_h_ + in_h) * input_w_ + in_w;

                    grad_in_ptr[in_idx] += grad_out_ptr[out_idx];
                }
            }
        }
    }

    // Check if gradients are non-zero
    float grad_sum = 0.0f;
    for (int64_t i = 0; i < N * C * input_h_ * input_w_; ++i) {
        grad_sum += std::abs(grad_in_ptr[i]);
    }
    std::cout << "[DEBUG] grad_input sum(abs): " << grad_sum << std::endl;

    return {grad_input};
}

} // namespace tenzor
