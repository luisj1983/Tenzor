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
    // GG.1: recompute sigmoid from saved input Variable on the higher-order path.
    Variable output_var;
    if (has_saved_variables()) {
        output_var = tenzor::sigmoid(saved_variables_[0]);
    } else {
        output_var = Variable(saved_tensors_[0], false);
    }
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
    // GG.1: recompute tanh from saved input Variable on the higher-order path.
    Variable output_var;
    if (has_saved_variables()) {
        output_var = tenzor::tanh(saved_variables_[0]);
    } else {
        output_var = Variable(saved_tensors_[0], false);
    }
    auto one_tensor = ones(std::vector<int64_t>(saved_tensors_[0].shape().begin(), saved_tensors_[0].shape().end()),
                           saved_tensors_[0].dtype(), saved_tensors_[0].device());
    Variable one_var(one_tensor, false);
    return {grad_outputs[0] * (one_var - output_var * output_var)};
}

// GeluBackward implementation
//
// The default GELU forward is the *exact* erf form (matches PyTorch's
// approximate='none'); the backward must use the matching analytic derivative:
//
//     gelu(x)  = 0.5 * x * (1 + erf(x / sqrt(2)))
//     gelu'(x) = 0.5 * (1 + erf(x / sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
//
// (The explicit tanh-approximation mode, nn::gelu(x, "tanh"), is composed from
// Variable ops in nn/activations and does not use this node.)
auto GeluBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("GeluBackward::forward should not be called");
}

auto GeluBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    constexpr double INV_SQRT2 = 0.7071067811865475244;  // 1/sqrt(2)
    constexpr double PDF_COEF  = 0.3989422804014326779;  // 1/sqrt(2*pi)

    // cdf = 0.5 * (1 + erf(x / sqrt(2)))
    auto cdf     = mul(add(tenzor::erf(mul(input, INV_SQRT2)), 1.0), 0.5);
    // pdf = (1/sqrt(2*pi)) * exp(-x^2 / 2)
    auto x_sq    = mul(input, input);
    auto pdf     = mul(tenzor::exp(mul(x_sq, -0.5)), PDF_COEF);
    // g'(x) = cdf + x * pdf
    auto g_prime = add(cdf, mul(input, pdf));

    return {mul(grad, g_prime)};
}

auto GeluBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Exact erf-GELU derivative (matches backward() above). Variable ops so the
    // graph chains for higher-order autograd (erf/exp have their own backward).
    // GG.1: prefer the saved input Variable so the graph chains back.
    Variable input_var = has_saved_variables() ? saved_variables_[0]
                                                : Variable(saved_tensors_[0], false);

    constexpr double INV_SQRT2 = 0.7071067811865475244;  // 1/sqrt(2)
    constexpr double PDF_COEF  = 0.3989422804014326779;  // 1/sqrt(2*pi)

    auto cdf     = (tenzor::erf(input_var * INV_SQRT2) + 1.0) * 0.5;
    auto x_sq    = input_var * input_var;
    auto pdf     = tenzor::exp(x_sq * -0.5) * PDF_COEF;
    auto g_prime = cdf + input_var * pdf;

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

    // mask = input >= 0. Inclusive boundary: at exactly x==0 ELU takes the
    // positive-branch derivative 1 (matching PyTorch and the forward output's
    // right-derivative), so forward/backward/gradcheck agree on the boundary
    // point for non-default alpha. The ELU value is identical on both branches
    // at 0, so this does not change any output, only the x==0 subgradient.
    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = ge(input, zero_tensor);

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
    // Inclusive boundary (x==0 -> positive branch, derivative 1); see the
    // EluBackward::backward twin above.
    auto mask = ge(input, zero_tensor);  // Tensor-level (non-differentiable)

    auto ones_tensor = ones(shape_vec, input.dtype(), input.device());

    // For the negative branch, use Variable ops so exp(input) is tracked.
    // GG.1: prefer the saved input Variable so the graph chains back.
    Variable input_var = has_saved_variables() ? saved_variables_[0]
                                                : Variable(input, false);
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

    // mask = input >= 0. Inclusive boundary: at x==0 SELU takes the positive-
    // branch derivative lambda (PyTorch convention / forward right-derivative);
    // the SELU value is identical on both branches at 0, so only the x==0
    // subgradient changes — forward/backward/gradcheck now agree there.
    auto zero_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto mask = ge(input, zero_tensor);

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
    // Inclusive boundary (x==0 -> positive branch, derivative lambda); see the
    // SeluBackward::backward twin above.
    auto mask = ge(input, zero_tensor);  // Non-differentiable mask

    auto pos_grad_tensor = full(shape_vec, lambda, input.dtype(), input.device());

    // Negative branch uses Variable exp for tracking.
    // GG.1: prefer the saved input Variable so the graph chains back.
    Variable input_var = has_saved_variables() ? saved_variables_[0]
                                                : Variable(input, false);
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
    // GG.1: prefer the saved input Variable so the graph chains back.
    Variable input_var = has_saved_variables() ? saved_variables_[0]
                                                : Variable(saved_tensors_[0], false);

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
    // GG.1: prefer the saved input Variable so the graph chains back.
    Variable input_var = has_saved_variables() ? saved_variables_[0]
                                                : Variable(saved_tensors_[0], false);
    double beta_val = extract_scalar_param(saved_tensors_[1]);

    auto beta_x = input_var * beta_val;
    auto sig = tenzor::sigmoid(beta_x);

    return {grad_outputs[0] * sig};
}

// ---------------------------------------------------------------------------
// saved_attributes() — A.4 multi-op JVP traversal
//
// These activations save the scalar shape parameter as a single-element
// tensor in saved_tensors_[1] (see ops.cpp::unary_autograd_with_param).
// Re-expose it as an OpAttributes entry under the AttrKey the corresponding
// JVP rule reads, so the forward-mode dispatch table can be invoked through
// an autograd grad_fn graph.
// ---------------------------------------------------------------------------

auto EluBackward::saved_attributes() const -> OpAttributes {
    OpAttributes attrs;
    if (saved_tensors_.size() >= 2) {
        attrs.set(AttrKey::Alpha, extract_scalar_param(saved_tensors_[1]));
    }
    return attrs;
}

auto LeakyReluBackward::saved_attributes() const -> OpAttributes {
    OpAttributes attrs;
    if (saved_tensors_.size() >= 2) {
        attrs.set(AttrKey::Negative_slope, extract_scalar_param(saved_tensors_[1]));
    }
    return attrs;
}

auto SoftplusBackward::saved_attributes() const -> OpAttributes {
    OpAttributes attrs;
    if (saved_tensors_.size() >= 2) {
        attrs.set(AttrKey::Beta, extract_scalar_param(saved_tensors_[1]));
    }
    return attrs;
}

} // namespace tenzor
