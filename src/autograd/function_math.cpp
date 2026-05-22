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
#include "function_helpers.hpp"
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
// Element-wise Math Backward Functions
// =========================================================================

// SqrtBackward implementation
// Saves output. backward: grad / (2 * output)
auto SqrtBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SqrtBackward::forward should not be called");
}

auto SqrtBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // sqrt(x)
    // grad / (2 * output)
    auto two_output = mul(output, 2.0);
    return {div(grad, two_output)};
}

auto SqrtBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable output_var(saved_tensors_[0], false);
    auto two_output = output_var * 2.0;
    return {grad_outputs[0] / two_output};
}

// PowBackward implementation
// Saves input and exponent. backward: grad * exponent * pow(input, exponent - 1)
auto PowBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("PowBackward::forward should not be called");
}

auto PowBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& exp_tensor = saved_tensors_[1];  // scalar tensor holding exponent

    double exp_val = extract_scalar_param(exp_tensor);

    if (exp_val == 0.0) {
        return {zeros(std::vector<int64_t>(grad.shape().begin(), grad.shape().end()),
                       grad.dtype(), grad.device())};
    }
    if (exp_val == 1.0) {
        return {grad};
    }

    // d/dx (x^n) = n * x^(n-1)
    // For positive x the direct form is exact. For negative x with
    // fractional n the derivative is undefined; let pow produce NaN so the
    // caller sees the invalid region, rather than fabricating a wrong
    // value via |x|+eps + sign(x).
    auto pow_term = pow(input, static_cast<float>(exp_val - 1.0));
    auto scaled = mul(pow_term, exp_val);
    return {mul(grad, scaled)};
}

auto PowBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // pow backward: grad * exponent * pow(input, exponent - 1)
    const auto& input = saved_tensors_[0];
    double exp_val = extract_scalar_param(saved_tensors_[1]);

    if (exp_val == 0.0) {
        return {Variable(zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                               input.dtype(), input.device()), false)};
    }
    if (exp_val == 1.0) {
        return {grad_outputs[0]};
    }

    Variable input_var(input, false);
    auto pow_term = tenzor::pow(input_var, static_cast<float>(exp_val - 1.0));
    auto scaled = pow_term * exp_val;
    return {grad_outputs[0] * scaled};
}

// ReciprocalBackward implementation
// Saves output. backward: grad * (-output * output)
auto ReciprocalBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ReciprocalBackward::forward should not be called");
}

auto ReciprocalBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // 1/x
    // grad * (-output^2)
    auto neg_out_sq = neg(mul(output, output));
    return {mul(grad, neg_out_sq)};
}

auto ReciprocalBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable output_var(saved_tensors_[0], false);
    return {grad_outputs[0] * tenzor::neg(output_var * output_var)};
}

// SinBackward implementation
// Saves input. backward: grad * cos(input)
auto SinBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SinBackward::forward should not be called");
}

auto SinBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, cos(input))};
}

auto SinBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // sin backward: grad * cos(input)
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] * tenzor::cos(input_var)};
}

// CosBackward implementation
// Saves input. backward: grad * (-sin(input))
auto CosBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CosBackward::forward should not be called");
}

auto CosBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, neg(sin(input)))};
}

auto CosBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // cos backward: grad * (-sin(input))
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] * tenzor::neg(tenzor::sin(input_var))};
}

// TanBackward implementation
// Saves output. backward: grad * (1 + output * output)
auto TanBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TanBackward::forward should not be called");
}

auto TanBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // tan(x)
    // 1 + tan^2(x) = sec^2(x)
    auto out_sq = mul(output, output);
    auto sec_sq = add(out_sq, 1.0);
    return {mul(grad, sec_sq)};
}

auto TanBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // tan backward: grad * (1 + tan(x)^2) = grad * sec^2(x)
    // saved_tensors_[0] = tan(x) (output)
    Variable output_var(saved_tensors_[0], false);
    auto sec_sq = output_var * output_var + 1.0;
    return {grad_outputs[0] * sec_sq};
}

// AsinBackward implementation
// Saves input. backward: grad / sqrt(1 - input * input)
auto AsinBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("AsinBackward::forward should not be called");
}

auto AsinBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    // 1 / sqrt(1 - x^2), with clamping for numerical safety
    auto x_sq = mul(input, input);
    auto one_minus_sq = sub(ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                  input.dtype(), input.device()), x_sq);
    auto clamped = clamp_min(one_minus_sq, 0.0);
    auto eps = full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto denom = sqrt(add(clamped, eps));
    return {div(grad, denom)};
}

auto AsinBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // asin backward: grad / sqrt(1 - x^2)
    Variable input_var(saved_tensors_[0], false);
    auto one_tensor = ones(std::vector<int64_t>(saved_tensors_[0].shape().begin(),
                                                 saved_tensors_[0].shape().end()),
                           saved_tensors_[0].dtype(), saved_tensors_[0].device());
    Variable one_var(one_tensor, false);
    auto one_minus_sq = one_var - input_var * input_var;
    // Clamp at Tensor level for numerical safety, wrap as constant
    auto clamped = clamp_min(one_minus_sq.tensor(), 0.0);
    auto eps = full(std::vector<int64_t>(saved_tensors_[0].shape().begin(),
                                          saved_tensors_[0].shape().end()),
                    detail::dtype_epsilon(saved_tensors_[0].dtype()),
                    saved_tensors_[0].dtype(), saved_tensors_[0].device());
    auto denom = Variable(sqrt(add(clamped, eps)), false);
    return {grad_outputs[0] / denom};
}

// AcosBackward implementation
// Saves input. backward: -grad / sqrt(1 - input * input)
auto AcosBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("AcosBackward::forward should not be called");
}

auto AcosBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    // -1 / sqrt(1 - x^2), with clamping for numerical safety
    auto x_sq = mul(input, input);
    auto one_minus_sq = sub(ones(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                                  input.dtype(), input.device()), x_sq);
    auto clamped = clamp_min(one_minus_sq, 0.0);
    auto eps = full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto denom = sqrt(add(clamped, eps));
    return {neg(div(grad, denom))};
}

auto AcosBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // acos backward: -grad / sqrt(1 - x^2)
    Variable input_var(saved_tensors_[0], false);
    auto one_tensor = ones(std::vector<int64_t>(saved_tensors_[0].shape().begin(),
                                                 saved_tensors_[0].shape().end()),
                           saved_tensors_[0].dtype(), saved_tensors_[0].device());
    Variable one_var(one_tensor, false);
    auto one_minus_sq = one_var - input_var * input_var;
    // Clamp at Tensor level for numerical safety, wrap as constant
    auto clamped = clamp_min(one_minus_sq.tensor(), 0.0);
    auto eps = full(std::vector<int64_t>(saved_tensors_[0].shape().begin(),
                                          saved_tensors_[0].shape().end()),
                    detail::dtype_epsilon(saved_tensors_[0].dtype()),
                    saved_tensors_[0].dtype(), saved_tensors_[0].device());
    auto denom = Variable(sqrt(add(clamped, eps)), false);
    return {tenzor::neg(grad_outputs[0] / denom)};
}

// AtanBackward implementation
// Saves input. backward: grad / (1 + input * input)
auto AtanBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("AtanBackward::forward should not be called");
}

auto AtanBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    // 1 / (1 + x^2)
    auto x_sq = mul(input, input);
    auto denom = add(x_sq, 1.0);
    return {div(grad, denom)};
}

auto AtanBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // atan backward: grad / (1 + x^2)
    Variable input_var(saved_tensors_[0], false);
    auto denom = input_var * input_var + 1.0;
    return {grad_outputs[0] / denom};
}

// SinhBackward implementation
// Saves input. backward: grad * cosh(input)
auto SinhBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SinhBackward::forward should not be called");
}

auto SinhBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, cosh(input))};
}

auto SinhBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // sinh backward: grad * cosh(input)
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] * tenzor::cosh(input_var)};
}

// CoshBackward implementation
// Saves input. backward: grad * sinh(input)
auto CoshBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CoshBackward::forward should not be called");
}

auto CoshBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, sinh(input))};
}

auto CoshBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // cosh backward: grad * sinh(input)
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] * tenzor::sinh(input_var)};
}

// =========================================================================
// Extended Math Backward Functions
// =========================================================================

// ErfBackward implementation
// Saves input. backward: grad * (2/sqrt(pi)) * exp(-input^2)
auto ErfBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ErfBackward::forward should not be called");
}

auto ErfBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    constexpr double two_over_sqrt_pi = 1.1283791670955126;  // 2/sqrt(pi)

    auto neg_x_sq = neg(mul(input, input));
    auto exp_term = exp(neg_x_sq);
    auto factor = mul(exp_term, two_over_sqrt_pi);
    return {mul(grad, factor)};
}

auto ErfBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // erf backward: grad * (2/sqrt(pi)) * exp(-x^2)
    Variable input_var(saved_tensors_[0], false);
    constexpr double two_over_sqrt_pi = 1.1283791670955126;

    auto neg_x_sq = tenzor::neg(input_var * input_var);
    auto exp_term = tenzor::exp(neg_x_sq);
    return {grad_outputs[0] * exp_term * two_over_sqrt_pi};
}

// ErfcBackward implementation
// Saves input. backward: grad * (-2/sqrt(pi)) * exp(-input^2)
auto ErfcBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ErfcBackward::forward should not be called");
}

auto ErfcBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    constexpr double neg_two_over_sqrt_pi = -1.1283791670955126;  // -2/sqrt(pi)

    auto neg_x_sq = neg(mul(input, input));
    auto exp_term = exp(neg_x_sq);
    auto factor = mul(exp_term, neg_two_over_sqrt_pi);
    return {mul(grad, factor)};
}

auto ErfcBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // erfc backward: grad * (-2/sqrt(pi)) * exp(-x^2)
    Variable input_var(saved_tensors_[0], false);
    constexpr double neg_two_over_sqrt_pi = -1.1283791670955126;

    auto neg_x_sq = tenzor::neg(input_var * input_var);
    auto exp_term = tenzor::exp(neg_x_sq);
    return {grad_outputs[0] * exp_term * neg_two_over_sqrt_pi};
}

// ErfInvBackward implementation
// Saves output. backward: grad * sqrt(pi)/2 * exp(erfinv(x)^2)
auto ErfInvBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ErfInvBackward::forward should not be called");
}

auto ErfInvBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // erfinv(x) saved as output
    constexpr double half_sqrt_pi = 0.8862269254527580;  // sqrt(pi)/2
    auto out_sq = mul(output, output);
    auto exp_term = exp(out_sq);
    return {mul(grad, mul(exp_term, half_sqrt_pi))};
}

auto ErfInvBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable output_var(saved_tensors_[0], false);
    constexpr double half_sqrt_pi = 0.8862269254527580;
    auto out_sq = output_var * output_var;
    auto exp_term = tenzor::exp(out_sq);
    return {grad_outputs[0] * exp_term * half_sqrt_pi};
}

// GammaBackward implementation
// Saves input. backward: grad * gamma(x) * digamma(x)
auto GammaBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("GammaBackward::forward should not be called");
}

auto GammaBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    auto gamma_val = tenzor::gamma(input);
    auto digamma_val = tenzor::digamma(input);
    return {mul(grad, mul(gamma_val, digamma_val))};
}

auto GammaBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable input_var(saved_tensors_[0], false);
    auto gamma_val = tenzor::gamma(input_var);
    auto digamma_val = tenzor::digamma(input_var);
    return {grad_outputs[0] * gamma_val * digamma_val};
}

// LgammaBackward implementation
// Saves input. backward: grad * digamma(x)
auto LgammaBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LgammaBackward::forward should not be called");
}

auto LgammaBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, tenzor::digamma(input))};
}

auto LgammaBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] * tenzor::digamma(input_var)};
}

// DigammaBackward implementation
// Saves input. backward: grad * polygamma(1, x)
auto DigammaBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("DigammaBackward::forward should not be called");
}

auto DigammaBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, tenzor::polygamma(1, input))};
}

auto DigammaBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] * tenzor::polygamma(1, input_var)};
}

// BesselI0Backward implementation
// Saves input. backward: grad * bessel_i1(x)
auto BesselI0Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("BesselI0Backward::forward should not be called");
}

auto BesselI0Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, tenzor::bessel_i1(input))};
}

auto BesselI0Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] * tenzor::bessel_i1(input_var)};
}

// BesselI1Backward implementation
// Saves input. backward: grad * (bessel_i0(x) + bessel_i1(x) * (-1/x))
// I1'(x) = I0(x) - I1(x)/x
auto BesselI1Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("BesselI1Backward::forward should not be called");
}

auto BesselI1Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    auto i0 = tenzor::bessel_i0(input);
    auto i1 = tenzor::bessel_i1(input);
    auto i1_over_x = div(i1, input);
    return {mul(grad, sub(i0, i1_over_x))};
}

auto BesselI1Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable input_var(saved_tensors_[0], false);
    auto i0 = tenzor::bessel_i0(input_var);
    auto i1 = tenzor::bessel_i1(input_var);
    return {grad_outputs[0] * (i0 - i1 / input_var)};
}

// SincBackward implementation
// Saves input. backward: grad * (cos(πx)/(x) - sin(πx)/(πx²)) for x≠0, else 0
auto SincBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SincBackward::forward should not be called");
}

auto SincBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    // sinc(x) = sin(πx) / (πx).  Derivative:
    //   d/dx [sin(πx)/(πx)] = (πx·cos(πx) − sin(πx)) / (πx²)
    auto pi_x = mul(input, M_PI);
    auto cos_px = tenzor::cos(pi_x);
    auto sin_px = tenzor::sin(pi_x);
    auto x_sq = mul(input, input);
    auto numer = sub(mul(pi_x, cos_px), sin_px);
    auto denom = mul(x_sq, M_PI);
    auto deriv = div(numer, denom);
    // x = 0 is a removable singularity (derivative is 0); mask it out.
    auto zero = tenzor::zeros_like(input);
    auto mask = tenzor::ne(input, zero);
    return {mul(grad, mul(deriv, mask))};
}

auto SincBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    Variable input_var(saved_tensors_[0], false);
    auto pi_x = input_var * M_PI;
    auto cos_px = tenzor::cos(pi_x);
    auto sin_px = tenzor::sin(pi_x);
    auto numer = pi_x * cos_px - sin_px;
    auto denom = input_var * input_var * M_PI;
    auto zero_t = tenzor::zeros_like(input_var.tensor());
    auto mask_tensor = tenzor::ne(input_var.tensor(), zero_t);
    Variable mask_var(mask_tensor, false);
    auto deriv = numer / denom;
    return {grad_outputs[0] * deriv * mask_var};
}

// Log2Backward implementation
// Saves input. backward: grad / (input * log(2))
auto Log2Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Log2Backward::forward should not be called");
}

auto Log2Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    constexpr double ln2 = 0.6931471805599453;  // log(2)

    auto denom = mul(input, ln2);
    return {div(grad, denom)};
}

auto Log2Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    constexpr double ln2 = 0.6931471805599453;
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] / (input_var * ln2)};
}

// Log10Backward implementation
// Saves input. backward: grad / (input * log(10))
auto Log10Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Log10Backward::forward should not be called");
}

auto Log10Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];

    constexpr double ln10 = 2.302585092994046;  // log(10)

    auto denom = mul(input, ln10);
    return {div(grad, denom)};
}

auto Log10Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    constexpr double ln10 = 2.302585092994046;
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] / (input_var * ln10)};
}

// Log1pBackward implementation
// Saves input. backward: grad / (1 + input)
auto Log1pBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Log1pBackward::forward should not be called");
}

auto Log1pBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    auto denom = add(input, 1.0);
    return {div(grad, denom)};
}

auto Log1pBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // log1p backward: grad / (1 + x)
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] / (input_var + 1.0)};
}

// Exp2Backward implementation
// Saves output. backward: grad * output * log(2)
auto Exp2Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Exp2Backward::forward should not be called");
}

auto Exp2Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& output = saved_tensors_[0];  // 2^x

    constexpr double ln2 = 0.6931471805599453;  // log(2)

    auto factor = mul(output, ln2);
    return {mul(grad, factor)};
}

auto Exp2Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    constexpr double ln2 = 0.6931471805599453;
    Variable output_var(saved_tensors_[0], false);
    return {grad_outputs[0] * (output_var * ln2)};
}

// Expm1Backward implementation
// Saves input. backward: grad * exp(input)
auto Expm1Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Expm1Backward::forward should not be called");
}

auto Expm1Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    return {mul(grad, exp(input))};
}

auto Expm1Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // expm1 backward: grad * exp(input)
    Variable input_var(saved_tensors_[0], false);
    return {grad_outputs[0] * tenzor::exp(input_var)};
}

// Atan2Backward implementation
// Saves inputs (y, x). backward: grad_y = grad * x / (x^2 + y^2), grad_x = grad * (-y) / (x^2 + y^2)
auto Atan2Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("Atan2Backward::forward should not be called");
}

auto Atan2Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& y = saved_tensors_[0];
    const auto& x = saved_tensors_[1];

    // denom = x^2 + y^2
    auto denom = add(mul(x, x), mul(y, y));

    // grad_y = grad * x / denom, reduced to match input y shape
    auto grad_y = reduce_grad_for_broadcasting(div(mul(grad, x), denom), input_shape_y_);

    // grad_x = grad * (-y) / denom, reduced to match input x shape
    auto grad_x = reduce_grad_for_broadcasting(div(mul(grad, neg(y)), denom), input_shape_x_);

    return {grad_y, grad_x};
}

auto Atan2Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // atan2 backward: grad_y = grad * x / (x^2 + y^2), grad_x = grad * (-y) / (x^2 + y^2)
    Variable y_var(saved_tensors_[0], false);
    Variable x_var(saved_tensors_[1], false);
    auto denom = x_var * x_var + y_var * y_var;
    auto grad_y = reduce_grad_var_for_broadcasting(grad_outputs[0] * x_var / denom, input_shape_y_);
    auto grad_x = reduce_grad_var_for_broadcasting(grad_outputs[0] * tenzor::neg(y_var) / denom, input_shape_x_);
    return {grad_y, grad_x};
}

// A.4 multi-op JVP traversal: expose the scalar exponent stored in
// saved_tensors_[1] as the `Exponent` attribute consumed by
// jvp_adapter_pow.
auto PowBackward::saved_attributes() const -> OpAttributes {
    OpAttributes attrs;
    if (saved_tensors_.size() >= 2) {
        attrs.set(AttrKey::Exponent, extract_scalar_param(saved_tensors_[1]));
    }
    return attrs;
}

} // namespace tenzor
