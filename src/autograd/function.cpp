#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
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

    // Debug for Float64
    if (a.dtype() == DType::Float64) {
        std::cerr << "[MATMUL_BACKWARD_F64] a.shape: [";
        for (size_t i = 0; i < a.shape().size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << a.shape()[i];
        }
        std::cerr << "], b.shape: [";
        for (size_t i = 0; i < b.shape().size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << b.shape()[i];
        }
        std::cerr << "], grad_out.shape: [";
        for (size_t i = 0; i < grad_out.shape().size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << grad_out.shape()[i];
        }
        std::cerr << "], device=" << static_cast<int>(a.device().type) << std::endl;

        // Check for inf in saved tensors
        auto a_cpu = a.to(Device::cpu());
        auto* a_data = a_cpu.data<double>();
        bool a_has_inf = false;
        for (int i = 0; i < std::min(1000, static_cast<int>(a_cpu.numel())); ++i) {
            if (std::isinf(a_data[i])) {
                a_has_inf = true;
                break;
            }
        }
        auto b_cpu = b.to(Device::cpu());
        auto* b_data = b_cpu.data<double>();
        bool b_has_inf = false;
        for (int i = 0; i < std::min(1000, static_cast<int>(b_cpu.numel())); ++i) {
            if (std::isinf(b_data[i])) {
                b_has_inf = true;
                break;
            }
        }
        auto go_cpu = grad_out.to(Device::cpu());
        auto* go_data = go_cpu.data<double>();
        bool go_has_inf = false;
        for (int i = 0; i < std::min(1000, static_cast<int>(go_cpu.numel())); ++i) {
            if (std::isinf(go_data[i])) {
                go_has_inf = true;
                break;
            }
        }
        // Also check max values of saved tensors
        double a_max = 0.0;
        for (int i = 0; i < static_cast<int>(a_cpu.numel()); ++i) {
            if (std::abs(a_data[i]) > a_max) a_max = std::abs(a_data[i]);
        }
        double b_max = 0.0;
        for (int i = 0; i < static_cast<int>(b_cpu.numel()); ++i) {
            if (std::abs(b_data[i]) > b_max) b_max = std::abs(b_data[i]);
        }
        std::cerr << "[MATMUL_BACKWARD_F64] a_has_inf=" << a_has_inf << " (max=" << a_max << ")"
                  << ", b_has_inf=" << b_has_inf << " (max=" << b_max << ")"
                  << ", grad_out_has_inf=" << go_has_inf << std::endl;
    }

    // Debug for Float16
    if (a.dtype() == DType::Float16) {
        std::cerr << "[MATMUL_BACKWARD_F16] a.shape: [";
        for (size_t i = 0; i < a.shape().size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << a.shape()[i];
        }
        std::cerr << "], b.shape: [";
        for (size_t i = 0; i < b.shape().size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << b.shape()[i];
        }
        std::cerr << "], grad_out.shape: [";
        for (size_t i = 0; i < grad_out.shape().size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << grad_out.shape()[i];
        }
        std::cerr << "]" << std::endl;

        // Check if inputs have non-zero values
        auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
        auto* a_data = a_cpu.data<float>();
        float a_sum = 0.0f;
        for (int i = 0; i < std::min(10, static_cast<int>(a_cpu.numel())); ++i) {
            a_sum += std::abs(a_data[i]);
        }
        std::cerr << "[MATMUL_BACKWARD_F16] a avg_abs_first10=" << (a_sum / std::min(10, static_cast<int>(a_cpu.numel()))) << std::endl;

        auto grad_out_cpu = grad_out.to(Device::cpu()).to(DType::Float32);
        auto* grad_out_data = grad_out_cpu.data<float>();
        float grad_sum = 0.0f;
        for (int i = 0; i < std::min(10, static_cast<int>(grad_out_cpu.numel())); ++i) {
            grad_sum += std::abs(grad_out_data[i]);
        }
        std::cerr << "[MATMUL_BACKWARD_F16] grad_out avg_abs_first10=" << (grad_sum / std::min(10, static_cast<int>(grad_out_cpu.numel()))) << std::endl;
    }

    // Get the number of dimensions
    auto a_ndim = a.shape().size();
    auto b_ndim = b.shape().size();

    // For 2D matrices: grad_a = grad_out @ b.T, grad_b = a.T @ grad_out
    auto b_t = transpose(b, b_ndim - 2, b_ndim - 1);
    auto a_t = transpose(a, a_ndim - 2, a_ndim - 1);

    // Debug: Check transposes for Float64
    if (a.dtype() == DType::Float64) {
        auto b_t_cpu = b_t.to(Device::cpu());
        auto* b_t_data = b_t_cpu.data<double>();
        bool b_t_has_inf = false;
        double b_t_max = 0.0;
        for (int i = 0; i < static_cast<int>(b_t_cpu.numel()); ++i) {
            if (std::isinf(b_t_data[i])) { b_t_has_inf = true; }
            if (std::abs(b_t_data[i]) > b_t_max) b_t_max = std::abs(b_t_data[i]);
        }
        auto a_t_cpu = a_t.to(Device::cpu());
        auto* a_t_data = a_t_cpu.data<double>();
        bool a_t_has_inf = false;
        double a_t_max = 0.0;
        for (int i = 0; i < static_cast<int>(a_t_cpu.numel()); ++i) {
            if (std::isinf(a_t_data[i])) { a_t_has_inf = true; }
            if (std::abs(a_t_data[i]) > a_t_max) a_t_max = std::abs(a_t_data[i]);
        }
        auto go_cpu = grad_out.to(Device::cpu());
        auto* go_data = go_cpu.data<double>();
        double go_max = 0.0;
        for (int i = 0; i < static_cast<int>(go_cpu.numel()); ++i) {
            if (std::abs(go_data[i]) > go_max) go_max = std::abs(go_data[i]);
        }
        std::cerr << "[MATMUL_BACKWARD_F64] After transpose: b_t_has_inf=" << b_t_has_inf << " (max=" << b_t_max << ")"
                  << ", a_t_has_inf=" << a_t_has_inf << " (max=" << a_t_max << ")"
                  << ", grad_out_max=" << go_max << std::endl;
    }

    auto grad_a = matmul(grad_out, b_t);

    // Debug: Check grad_a for Float64 - check ALL elements
    // Also check if values overflow when converted to Float32
    if (a.dtype() == DType::Float64) {
        auto grad_a_cpu = grad_a.to(Device::cpu());
        auto* grad_a_data = grad_a_cpu.data<double>();
        bool grad_a_has_inf = false;
        bool grad_a_overflow_f32 = false;
        int inf_idx = -1;
        int overflow_idx = -1;
        double max_val = 0.0;
        for (int i = 0; i < static_cast<int>(grad_a_cpu.numel()); ++i) {
            double v = grad_a_data[i];
            if (std::isinf(v)) {
                grad_a_has_inf = true;
                if (inf_idx < 0) inf_idx = i;
            }
            if (std::abs(v) > max_val) max_val = std::abs(v);
            // Check if it would overflow in Float32
            if (!std::isinf(v) && std::abs(v) > 3.4e38) {  // Float32 max ~3.4e38
                grad_a_overflow_f32 = true;
                if (overflow_idx < 0) overflow_idx = i;
            }
        }
        std::cerr << "[MATMUL_BACKWARD_F64] After grad_a = matmul(grad_out, b_t): grad_a_has_inf=" << grad_a_has_inf
                  << " (idx=" << inf_idx << "/" << grad_a_cpu.numel() << ")"
                  << ", max=" << max_val
                  << ", overflow_f32=" << grad_a_overflow_f32 << " (idx=" << overflow_idx << ")"
                  << ", grad_a.shape=[";
        for (size_t i = 0; i < grad_a.shape().size(); ++i) {
            if (i > 0) std::cerr << ",";
            std::cerr << grad_a.shape()[i];
        }
        std::cerr << "]" << std::endl;
    }

    auto grad_b = matmul(a_t, grad_out);

    // Debug: Check grad_b for Float64 - check ALL elements
    if (a.dtype() == DType::Float64) {
        auto grad_b_cpu = grad_b.to(Device::cpu());
        auto* grad_b_data = grad_b_cpu.data<double>();
        bool grad_b_has_inf = false;
        bool grad_b_overflow_f32 = false;
        int inf_idx = -1;
        int overflow_idx = -1;
        double max_val = 0.0;
        for (int i = 0; i < static_cast<int>(grad_b_cpu.numel()); ++i) {
            double v = grad_b_data[i];
            if (std::isinf(v)) {
                grad_b_has_inf = true;
                if (inf_idx < 0) inf_idx = i;
            }
            if (std::abs(v) > max_val) max_val = std::abs(v);
            if (!std::isinf(v) && std::abs(v) > 3.4e38) {
                grad_b_overflow_f32 = true;
                if (overflow_idx < 0) overflow_idx = i;
            }
        }
        std::cerr << "[MATMUL_BACKWARD_F64] After grad_b = matmul(a_t, grad_out): grad_b_has_inf=" << grad_b_has_inf
                  << " (idx=" << inf_idx << "/" << grad_b_cpu.numel() << ")"
                  << ", max=" << max_val
                  << ", overflow_f32=" << grad_b_overflow_f32 << " (idx=" << overflow_idx << ")"
                  << ", grad_b.shape=[";
        for (size_t i = 0; i < grad_b.shape().size(); ++i) {
            if (i > 0) std::cerr << ",";
            std::cerr << grad_b.shape()[i];
        }
        std::cerr << "]" << std::endl;
    }

    // Debug output gradients for Float16
    if (a.dtype() == DType::Float16) {
        auto grad_b_cpu = grad_b.to(Device::cpu()).to(DType::Float32);
        auto* grad_b_data = grad_b_cpu.data<float>();
        float grad_b_sum = 0.0f;
        for (int i = 0; i < std::min(10, static_cast<int>(grad_b_cpu.numel())); ++i) {
            grad_b_sum += std::abs(grad_b_data[i]);
        }
        std::cerr << "[MATMUL_BACKWARD_F16] grad_b avg_abs_first10=" << (grad_b_sum / std::min(10, static_cast<int>(grad_b_cpu.numel()))) << std::endl;
    }

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

    // Use double for scale calculation to preserve precision for Float64 tensors
    double scale = 1.0 / static_cast<double>(n_elements);
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Debug for Float64 and Float16
    bool debug = (input.dtype() == DType::Float64 || input.dtype() == DType::Float16) &&
                 input.device().type == Device::Type::Vulkan;

    if (!dim_.has_value()) {
        // Reduced all dimensions - broadcast scalar tensor back to original shape
        // Use pure tensor operations (no CPU transfers) - backend agnostic!
        auto grad = grad_output;

        if (debug) {
            auto grad_cpu = grad.to(Device::cpu()).to(DType::Float32);
            auto* data = grad_cpu.data<float>();
            std::cerr << "[MEAN_BACKWARD] grad_output numel=" << grad.numel()
                      << " dtype=" << static_cast<int>(grad.dtype())
                      << " value=" << (grad.numel() > 0 ? data[0] : -999.0f) << std::endl;
        }

        // Ensure grad is a 0-d tensor (may be 1-element tensor from some reductions)
        if (grad.ndim() > 0) {
            grad = reshape(grad, {});
        }

        // Expand the scalar to input shape natively on device
        auto expanded = expand(grad, input_shape_vec);

        if (debug) {
            auto exp_cpu = expanded.to(Device::cpu()).to(DType::Float32);
            auto* data = exp_cpu.data<float>();
            std::cerr << "[MEAN_BACKWARD] expanded numel=" << expanded.numel()
                      << " first5=";
            for (int i = 0; i < std::min(5, (int)exp_cpu.numel()); ++i) {
                std::cerr << data[i] << " ";
            }
            std::cerr << std::endl;
        }

        // Scale by 1/N using backend-agnostic tensor multiplication
        // Create scalar tensor with same dtype and device as expanded gradient
        // Use double overload of full() to preserve precision for Float64
        auto scale_tensor = full({}, scale, expanded.dtype(), expanded.device());

        if (debug) {
            auto scale_cpu = scale_tensor.to(Device::cpu()).to(DType::Float32);
            auto* data = scale_cpu.data<float>();
            std::cerr << "[MEAN_BACKWARD] scale_tensor numel=" << scale_tensor.numel()
                      << " value=" << data[0]
                      << " (scale=" << scale << ", n_elements=" << n_elements << ")" << std::endl;
        }

        auto result = mul(expanded, scale_tensor);

        if (debug) {
            auto res_cpu = result.to(Device::cpu()).to(DType::Float32);
            auto* data = res_cpu.data<float>();
            std::cerr << "[MEAN_BACKWARD] result numel=" << result.numel()
                      << " first5=";
            for (int i = 0; i < std::min(5, (int)res_cpu.numel()); ++i) {
                std::cerr << data[i] << " ";
            }
            std::cerr << std::endl;
        }

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
        auto epsilon = full(input_shape_vec, 1e-7f, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        // Convert boolean mask to float for gradient computation
        auto ones_tensor = ones(input_shape_vec, input.dtype(), input.device());
        auto zeros_tensor = zeros(input_shape_vec, input.dtype(), input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

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

// RollBackward implementation
auto RollBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("RollBackward::forward should not be called");
}

auto RollBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Roll backward is roll with negative shift
    auto grad_input = roll(grad_outputs[0], -shifts_, dim_);
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
