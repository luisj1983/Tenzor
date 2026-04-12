#include "tenzor/autograd/function.hpp"
#include "function_helpers.hpp"
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

// =========================================================================
// Activation Backward Functions
// =========================================================================

// SigmoidBackward_AG implementation
// Saves output. backward: grad * output * (1 - output)
auto SigmoidBackward_AG::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SigmoidBackward_AG::forward should not be called");
}

auto SigmoidBackward_AG::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // sigmoid output
    // grad * output * (1 - output)
    auto one_minus_out = sub(ones(std::vector<int64_t>(output.shape().begin(), output.shape().end()),
                                  output.dtype(), output.device()), output);
    return {mul(grad, mul(output, one_minus_out))};
}

auto SigmoidBackward_AG::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable output_var(saved_tensors_[0], false);
    auto one_tensor = ones(std::vector<int64_t>(saved_tensors_[0].shape().begin(), saved_tensors_[0].shape().end()),
                           saved_tensors_[0].dtype(), saved_tensors_[0].device());
    Variable one_var(one_tensor, false);
    return {grad_outputs[0] * output_var * (one_var - output_var)};
}

// TanhBackward_AG implementation
// Saves output. backward: grad * (1 - output * output)
auto TanhBackward_AG::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TanhBackward_AG::forward should not be called");
}

auto TanhBackward_AG::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // tanh output
    // grad * (1 - output^2)
    auto out_sq = mul(output, output);
    auto one_minus_sq = sub(ones(std::vector<int64_t>(output.shape().begin(), output.shape().end()),
                                  output.dtype(), output.device()), out_sq);
    return {mul(grad, one_minus_sq)};
}

auto TanhBackward_AG::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable output_var(saved_tensors_[0], false);
    auto one_tensor = ones(std::vector<int64_t>(saved_tensors_[0].shape().begin(), saved_tensors_[0].shape().end()),
                           saved_tensors_[0].dtype(), saved_tensors_[0].device());
    Variable one_var(one_tensor, false);
    return {grad_outputs[0] * (one_var - output_var * output_var)};
}

// GeluBackward implementation
//
// IMPORTANT: the backward must match the *forward* implementation, which
// uses the tanh approximation:
//
//     gelu(x) = 0.5 * x * (1 + tanh(s(x)))
//     s(x)    = sqrt(2/pi) * (x + 0.044715 * x^3)
//
// A previous implementation used the derivative of the exact erf-based
// GELU, which disagrees with the tanh-approx forward by up to ~0.02 at
// some x — enough to fail `gradcheck` under Float64 tolerances.
//
// Derivative of the tanh-approx GELU:
//     s'(x)   = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
//     sech²(s) = 1 - tanh²(s)
//     g'(x)   = 0.5 * (1 + tanh(s)) + 0.5 * x * sech²(s) * s'(x)
auto GeluBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("GeluBackward::forward should not be called");
}

auto GeluBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    constexpr double sqrt_2_over_pi = 0.7978845608028654;  // sqrt(2/pi)
    constexpr double coeff          = 0.044715;
    constexpr double deriv_coeff    = 3.0 * coeff;          // 0.134145

    auto x_sq     = mul(input, input);                                 // x^2
    auto x_cubed  = mul(x_sq, input);                                  // x^3

    // s(x) = sqrt_2_over_pi * (x + 0.044715 * x^3)
    auto inner    = mul(add(input, mul(x_cubed, coeff)), sqrt_2_over_pi);

    auto T        = tenzor::tanh(inner);                               // tanh(s)
    auto T_sq     = mul(T, T);                                         // tanh²(s)
    auto sech_sq  = add(neg(T_sq), 1.0);                               // 1 - tanh²

    // s'(x) = sqrt_2_over_pi * (1 + 0.134145 * x^2)
    auto s_prime  = mul(add(mul(x_sq, deriv_coeff), 1.0), sqrt_2_over_pi);

    // g'(x) = 0.5 * (1 + T) + 0.5 * x * (1 - T^2) * s'(x)
    auto term1    = mul(add(T, 1.0), 0.5);
    auto term2    = mul(mul(mul(input, sech_sq), s_prime), 0.5);
    auto g_prime  = add(term1, term2);

    return {mul(grad, g_prime)};
}

auto GeluBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Match the tanh-approx forward — see the comment on backward() above for
    // the derivation. This Variable-based path is used for higher-order
    // autograd (double backward); it must compute the same function as the
    // Tensor-based backward() above.
    Variable input_var(saved_tensors_[0], false);

    constexpr double sqrt_2_over_pi = 0.7978845608028654;
    constexpr double coeff          = 0.044715;
    constexpr double deriv_coeff    = 3.0 * coeff;

    auto x_sq    = input_var * input_var;
    auto x_cubed = x_sq * input_var;

    // s(x) = sqrt_2_over_pi * (x + 0.044715 * x^3)
    auto inner   = (input_var + x_cubed * coeff) * sqrt_2_over_pi;

    auto T       = tenzor::tanh(inner);
    auto T_sq    = T * T;
    // 1 - T^2 — Variable doesn't have a scalar-minus-Variable operator, so
    // rewrite as (-(T^2)) + 1.
    auto sech_sq = tenzor::neg(T_sq) + 1.0;

    // s'(x) = sqrt_2_over_pi * (1 + 0.134145 * x^2)
    auto s_prime = (x_sq * deriv_coeff + 1.0) * sqrt_2_over_pi;

    // g'(x) = 0.5 * (1 + T) + 0.5 * x * (1 - T^2) * s'(x)
    auto term1   = (T + 1.0) * 0.5;
    auto term2   = (input_var * sech_sq * s_prime) * 0.5;
    auto g_prime = term1 + term2;

    return {grad_outputs[0] * g_prime};
}

// EluBackward implementation
// Saves input and alpha (as saved_tensors_[1]). backward: grad * where(input > 0, 1, alpha * exp(input))
auto EluBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("EluBackward::forward should not be called");
}

auto EluBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& alpha_tensor = saved_tensors_[1];  // scalar tensor holding alpha
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Extract alpha value (dtype-aware; previously hard-coded Float32).
    double alpha_val = extract_scalar_param(alpha_tensor);

    // mask = input > 0
    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = gt(input, zero_tensor);

    // positive path: gradient is 1
    auto ones_tensor = ones(shape_vec, input.dtype(), input.device());

    // negative path: gradient is alpha * exp(input)
    auto neg_grad = mul(exp(input), alpha_val);

    // where(input > 0, 1, alpha * exp(input))
    auto grad_factor = where(mask, ones_tensor, neg_grad);

    return {mul(grad, grad_factor)};
}

auto EluBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // ELU backward: grad * where(input > 0, 1, alpha * exp(input))
    // The mask is non-differentiable; compute it at Tensor level
    const auto& input = saved_tensors_[0];
    const auto& alpha_tensor = saved_tensors_[1];
    double alpha_val = extract_scalar_param(alpha_tensor);
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = gt(input, zero_tensor);  // Tensor-level (non-differentiable)

    auto ones_tensor = ones(shape_vec, input.dtype(), input.device());

    // For the negative branch, use Variable ops so exp(input) is tracked
    Variable input_var(input, false);
    auto neg_grad = tenzor::exp(input_var) * alpha_val;

    // where with non-differentiable mask: wrap Tensor constants as Variables
    Variable mask_var(mask, false);
    Variable ones_var(ones_tensor, false);
    auto grad_factor = tenzor::where(mask_var, ones_var, neg_grad);

    return {grad_outputs[0] * grad_factor};
}

// SeluBackward implementation
// Saves input. lambda=1.0507, alpha=1.6733. backward: grad * where(input > 0, lambda, lambda * alpha * exp(input))
auto SeluBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SeluBackward::forward should not be called");
}

auto SeluBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    constexpr double lambda = 1.0507009873554804934193349852946;
    constexpr double alpha = 1.6732632423543772848170429916717;

    // mask = input > 0
    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = gt(input, zero_tensor);

    // positive path: lambda
    auto pos_grad = full(shape_vec, lambda, input.dtype(), input.device());

    // negative path: lambda * alpha * exp(input)
    auto neg_grad = mul(exp(input), lambda * alpha);

    auto grad_factor = where(mask, pos_grad, neg_grad);
    return {mul(grad, grad_factor)};
}

auto SeluBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // SELU backward: grad * where(input > 0, lambda, lambda * alpha * exp(input))
    const auto& input = saved_tensors_[0];
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    constexpr double lambda = 1.0507009873554804934193349852946;
    constexpr double alpha = 1.6732632423543772848170429916717;

    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = gt(input, zero_tensor);  // Non-differentiable mask

    auto pos_grad_tensor = full(shape_vec, lambda, input.dtype(), input.device());

    // Negative branch uses Variable exp for tracking
    Variable input_var(input, false);
    auto neg_grad = tenzor::exp(input_var) * (lambda * alpha);

    Variable mask_var(mask, false);
    Variable pos_var(pos_grad_tensor, false);
    auto grad_factor = tenzor::where(mask_var, pos_var, neg_grad);

    return {grad_outputs[0] * grad_factor};
}

// MishBackward implementation
// Saves input. backward: grad * (tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2)) where sp = softplus(x) = log(1 + exp(x))
auto MishBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("MishBackward::forward should not be called");
}

auto MishBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    // softplus(x) = log(1 + exp(x))
    auto exp_x = exp(input);
    auto sp = log(add(exp_x, 1.0));  // log(1 + exp(x))

    // tanh_sp = tanh(sp)
    auto tanh_sp = tanh(sp);

    // sigmoid(x) = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))
    auto sig_x = sigmoid(input);

    // 1 - tanh(sp)^2
    auto tanh_sp_sq = mul(tanh_sp, tanh_sp);
    auto one_minus_tanh_sq = sub(ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                       input.dtype(), input.device()), tanh_sp_sq);

    // x * sigmoid(x) * (1 - tanh(sp)^2)
    auto second_term = mul(mul(input, sig_x), one_minus_tanh_sq);

    // grad * (tanh_sp + second_term)
    return {mul(grad, add(tanh_sp, second_term))};
}

auto MishBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Mish backward: grad * (tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2))
    // where sp = softplus(x) = log(1 + exp(x))
    Variable input_var(saved_tensors_[0], false);

    // softplus(x) = log(1 + exp(x))
    auto exp_x = tenzor::exp(input_var);
    auto sp = tenzor::log(exp_x + 1.0);

    // tanh_sp = tanh(sp)
    auto tanh_sp = tenzor::tanh(sp);

    // sigmoid(x)
    auto sig_x = tenzor::sigmoid(input_var);

    // 1 - tanh(sp)^2
    auto one_tensor = ones(std::vector<int64_t>(saved_tensors_[0].shape().begin(),
                                                 saved_tensors_[0].shape().end()),
                           saved_tensors_[0].dtype(), saved_tensors_[0].device());
    Variable one_var(one_tensor, false);
    auto one_minus_tanh_sq = one_var - tanh_sp * tanh_sp;

    // x * sigmoid(x) * (1 - tanh(sp)^2)
    auto second_term = input_var * sig_x * one_minus_tanh_sq;

    return {grad_outputs[0] * (tanh_sp + second_term)};
}

// LeakyReluBackward implementation
// Saves input and negative_slope. backward: grad * where(input > 0, 1, negative_slope)
auto LeakyReluBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LeakyReluBackward::forward should not be called");
}

auto LeakyReluBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& slope_tensor = saved_tensors_[1];  // scalar tensor holding negative_slope
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Dtype-aware scalar extraction. Previously hard-coded Float32 and
    // crashed on Float64 inputs with a cryptic "expected float64" error
    // from .data<float>().
    double slope_val = extract_scalar_param(slope_tensor);

    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = gt(input, zero_tensor);

    auto ones_tensor = ones(shape_vec, input.dtype(), input.device());
    auto slope_full = full(shape_vec, slope_val, input.dtype(), input.device());

    auto grad_factor = where(mask, ones_tensor, slope_full);
    return {mul(grad, grad_factor)};
}

auto LeakyReluBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // LeakyReLU backward: grad * where(input > 0, 1, negative_slope)
    // The mask is non-differentiable, compute at Tensor level
    const auto& input = saved_tensors_[0];
    const auto& slope_tensor = saved_tensors_[1];
    double slope_val = extract_scalar_param(slope_tensor);
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = gt(input, zero_tensor);

    auto ones_tensor = ones(shape_vec, input.dtype(), input.device());
    auto slope_full = full(shape_vec, slope_val, input.dtype(), input.device());

    // Mask-based selection is non-differentiable, so Tensor-level where is fine
    auto grad_factor = where(mask, ones_tensor, slope_full);
    Variable factor_var(grad_factor, false);
    return {grad_outputs[0] * factor_var};
}

// SoftplusBackward implementation
// Saves input and beta. backward: grad * sigmoid(beta * input)
auto SoftplusBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SoftplusBackward::forward should not be called");
}

auto SoftplusBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& beta_tensor = saved_tensors_[1];

    double beta_val = extract_scalar_param(beta_tensor);

    // sigmoid(beta * input)
    auto beta_x = mul(input, beta_val);
    auto sig = sigmoid(beta_x);

    return {mul(grad, sig)};
}

auto SoftplusBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Softplus backward: grad * sigmoid(beta * input)
    Variable input_var(saved_tensors_[0], false);
    double beta_val = extract_scalar_param(saved_tensors_[1]);

    auto beta_x = input_var * beta_val;
    auto sig = tenzor::sigmoid(beta_x);

    return {grad_outputs[0] * sig};
}

} // namespace tenzor
