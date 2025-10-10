#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/dispatch.hpp"

namespace tenzor {

auto Function::set_next_functions(std::vector<std::shared_ptr<Function>> funcs) -> void {
    next_functions_ = std::move(funcs);
}

auto Function::next_functions() const -> const std::vector<std::shared_ptr<Function>>& {
    return next_functions_;
}

auto Function::set_input_variables(std::vector<Variable*> inputs) -> void {
    input_variables_ = std::move(inputs);
}

auto Function::input_variables() const -> const std::vector<Variable*>& {
    return input_variables_;
}

auto Function::save_for_backward(std::vector<Tensor> tensors) -> void {
    saved_tensors_ = std::move(tensors);
}

auto Function::saved_tensors() const -> const std::vector<Tensor>& {
    return saved_tensors_;
}

// AddBackward implementation
auto AddBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = add(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto AddBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {grad_outputs[0], grad_outputs[0]};
}

// SubBackward implementation
auto SubBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = sub(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto SubBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(a-b)/da = 1, d(a-b)/db = -1
    return {grad_outputs[0], neg(grad_outputs[0])};
}

// MulBackward implementation
auto MulBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    saved_tensors_ = {inputs[0].tensor(), inputs[1].tensor()};
    auto result = mul(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto MulBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(a*b)/da = b, d(a*b)/db = a
    return {
        mul(grad_outputs[0], saved_tensors_[1]),
        mul(grad_outputs[0], saved_tensors_[0])
    };
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
        // Reduced all dimensions - create tensor filled with scalar grad value
        // Use full() which works natively on all devices
        if (grad_output.device().type == Device::Type::CUDA) {
            // Transfer scalar to CPU to extract value
            auto grad_cpu = grad_output.to(Device::cpu());
            float grad_val = grad_cpu.data<float>()[0];
            return {full(input_shape_vec, grad_val, input.dtype(), input.device())};
        } else {
            float grad_val = grad_output.data<float>()[0];
            return {full(input_shape_vec, grad_val, input.dtype(), input.device())};
        }
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
        // Reduced all dimensions - use full() which works natively on all devices
        if (grad_output.device().type == Device::Type::CUDA) {
            auto grad_cpu = grad_output.to(Device::cpu());
            float grad_val = grad_cpu.data<float>()[0] * scale;
            return {full(input_shape_vec, grad_val, input.dtype(), input.device())};
        } else {
            float grad_val = grad_output.data<float>()[0] * scale;
            return {full(input_shape_vec, grad_val, input.dtype(), input.device())};
        }
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
        auto scale_tensor = full(input_shape_vec, scale, input.dtype(), expanded.device());
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

} // namespace tenzor
