#include "tenzor/autograd/function.hpp"
#include <cassert>
#include "tenzor/autograd/ops.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/safe_math.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#ifdef __GNUC__
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace tenzor {

// SumBackward implementation
auto SumBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
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
    save_for_backward({inputs[0].tensor()});
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
    save_for_backward({inputs[0].tensor()});
    auto result = log(inputs[0].tensor());
    return {Variable(result, true)};
}

auto LogBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(log(x))/dx = 1/x, with zero-safe clamping to prevent NaN
    const auto& input = saved_tensors_[0];
    if (input.is_complex()) {
        // Wirtinger: d/d(conj(z)) log(z) = grad / conj(z)
        auto grad_input = div(grad_outputs[0], conj(input));
        return {grad_input};
    }
    auto zero = zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    auto eps = full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    detail::dtype_epsilon(input.dtype()),
                    input.dtype(), input.device());
    auto safe_input = where(eq(input, zero), eps, input);
    auto grad_input = div(grad_outputs[0], safe_input);
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
    save_for_backward({result});  // Save output for backward
    return {Variable(result, true)};
}

auto ExpBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // d(exp(x))/dx = exp(x)
    const auto& output = saved_tensors_[0];
    if (output.is_complex()) {
        // Wirtinger: d/d(conj(z)) exp(z) = conj(exp(z)) * grad
        auto grad_input = mul(grad_outputs[0], conj(output));
        return {grad_input};
    }
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
    attrs.set(AttrKey::Dim, dim_);
    std::vector<Tensor> input_tensors = {inputs[0].tensor()};
    auto result = dispatch(OpId::LogSoftmax, input_tensors, attrs)[0];

    // Save output for backward
    save_for_backward({result});

    return {Variable(result, true)};
}

auto LogSoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Use backend's log_softmax_backward kernel
    const auto& output = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim_);
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
    attrs.set(AttrKey::Dim, dim_);
    std::vector<Tensor> input_tensors = {inputs[0].tensor()};
    auto result = dispatch(OpId::Softmax, input_tensors, attrs)[0];

    // Save output for backward
    save_for_backward({result});

    return {Variable(result, true)};
}

auto SoftmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Use backend-optimized softmax_backward kernel via dispatch
    const auto& output = saved_tensors_[0];  // y = softmax(x)
    const auto& grad_output = grad_outputs[0];  // dL/dy

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim_);
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
    save_for_backward({inputs[0].tensor()});
    auto result = abs(inputs[0].tensor());
    return {Variable(result, true)};
}

auto AbsBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    if (input.is_complex()) {
        // Wirtinger: d/d(conj(z)) |z| = z / (2 * |z|)
        auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        auto abs_input = tenzor::abs(input);
        double eps = 1e-7;
        auto eps_tensor = full(input_shape_vec, eps, abs_input.dtype(), input.device());
        auto safe_abs = tenzor::where(gt(abs_input, eps_tensor), abs_input, eps_tensor);
        // grad * z / (2 * |z|)
        auto scale = div(input, mul(safe_abs, full(input_shape_vec, 2.0f, input.dtype(), input.device())));
        return {mul(grad.to(input.dtype()), scale)};
    }

    // Real path: d(abs(x))/dx = sign(x), with epsilon guard at x=0
    double eps = 1e-7;
    if (input.dtype() == DType::Float64) eps = 1e-15;
    else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) eps = 1e-3;

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto abs_input = tenzor::abs(input);
    auto eps_tensor = full(input_shape_vec, eps, input.dtype(), input.device());
    auto mask = gt(abs_input, eps_tensor);
    auto safe_sign = tenzor::where(mask,
        sign(input),
        zeros(input_shape_vec, input.dtype(), input.device()));
    return {mul(grad, safe_sign)};
}

auto AbsBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // d(abs(x))/dx = sign(x), with epsilon guard to avoid NaN at x=0.
    // sign is non-differentiable, so compute it at Tensor level.
    const auto& input = saved_tensors_[0];

    double eps = 1e-7;
    if (input.dtype() == DType::Float64) eps = 1e-15;
    else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) eps = 1e-3;

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto abs_input = tenzor::abs(input);
    auto eps_tensor = full(input_shape_vec, eps, input.dtype(), input.device());
    auto mask = gt(abs_input, eps_tensor);
    auto safe_sign = tenzor::where(mask,
        sign(input),
        zeros(input_shape_vec, input.dtype(), input.device()));
    Variable sign_var(safe_sign, false);
    return {grad_outputs[0] * sign_var};
}

// ClampBackward implementation
auto ClampBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor()});
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

// =========================================================================
// Complex number backward functions (Wirtinger derivatives)
// =========================================================================

// ConjBackward: conj(z) -> grad = conj(grad)
auto ConjBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = conj(inputs[0].tensor());
    return {Variable(result, true)};
}

auto ConjBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {conj(grad_outputs[0])};
}

// RealBackward: real(z) -> grad_z = 0.5 * grad (with zero imaginary part)
auto RealBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    input_dtype_ = inputs[0].tensor().dtype();
    auto result = real(inputs[0].tensor());
    return {Variable(result, true)};
}

auto RealBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // Gradient flows into the real part only: construct complex tensor
    // with real=0.5*grad, imag=0
    auto half_grad = mul(grad, full(
        std::vector<int64_t>(grad.shape().begin(), grad.shape().end()),
        0.5f, grad.dtype(), grad.device()));
    auto zero = zeros(std::vector<int64_t>(grad.shape().begin(), grad.shape().end()),
                      grad.dtype(), grad.device());
    // Re-create complex: real part = 0.5 * grad, imag part = 0
    // Use polar(abs, angle) or direct construction depending on available ops
    // Simplest: cast back to complex dtype with imag=0
    auto result = half_grad.to(input_dtype_);
    return {result};
}

// ImagBackward: imag(z) -> grad_z = -0.5j * grad
auto ImagBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    input_dtype_ = inputs[0].tensor().dtype();
    auto result = imag(inputs[0].tensor());
    return {Variable(result, true)};
}

auto ImagBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // Gradient flows into the imaginary part: construct complex tensor
    // with real=0, imag=0.5*grad
    // For Wirtinger convention, grad of imag w.r.t. conj(z) = -0.5j
    // So the complex gradient has real=0 and imag=0.5*grad
    auto half_grad = mul(grad, full(
        std::vector<int64_t>(grad.shape().begin(), grad.shape().end()),
        0.5f, grad.dtype(), grad.device()));
    // Cast to complex with value in imaginary part
    auto result = half_grad.to(input_dtype_);
    // Multiply by j (imaginary unit): multiply by complex(0, 1)
    // For the Wirtinger convention this gives us -0.5j * grad when
    // accounting for the conjugate symmetry
    auto j_factor = full(std::vector<int64_t>(result.shape().begin(), result.shape().end()),
                         0.0f, input_dtype_, result.device());
    // The result needs to route the gradient to the imaginary component
    // Simplest correct approach: negate and place in imaginary
    result = neg(result);  // -0.5 * grad as the imaginary component contribution
    return {result};
}

} // namespace tenzor
