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

    float exp_val = exp_tensor.data<float>()[0];

    // Special cases for numerical safety
    if (exp_val == 0.0f) {
        // d/dx(x^0) = 0
        return {zeros(std::vector<int64_t>(grad.shape().begin(), grad.shape().end()),
                       grad.dtype(), grad.device())};
    }
    if (exp_val == 1.0f) {
        // d/dx(x^1) = 1
        return {grad};
    }

    // General case: grad * exponent * pow(input, exponent - 1)
    // Use abs(input) + eps for negative base safety with fractional exponents
    auto eps = full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto safe_input = add(abs(input), eps);
    auto pow_term = pow(safe_input, exp_val - 1.0f);
    auto scaled = mul(pow_term, static_cast<double>(exp_val));
    // Restore sign: d/dx(|x|^n) * sign(x) for odd integer exponents
    auto result = mul(grad, mul(scaled, sign(input)));
    return {result};
}

auto PowBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // pow backward: grad * exponent * pow(input, exponent - 1)
    const auto& input = saved_tensors_[0];
    float exp_val = saved_tensors_[1].data<float>()[0];

    if (exp_val == 0.0f) {
        // d/dx(x^0) = 0
        return {Variable(zeros(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                               input.dtype(), input.device()), false)};
    }
    if (exp_val == 1.0f) {
        // d/dx(x^1) = 1
        return {grad_outputs[0]};
    }

    // General case: grad * exponent * pow(input, exponent - 1) * sign(input)
    // Use Variable ops for pow and sign tracking
    Variable input_var(input, false);
    auto eps_tensor = full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                           detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    Variable eps_var(eps_tensor, false);
    auto safe_input = tenzor::abs(input_var) + eps_var;
    auto pow_term = tenzor::pow(safe_input, exp_val - 1.0f);
    auto scaled = pow_term * static_cast<double>(exp_val);
    auto sign_tensor = sign(input);
    Variable sign_var(sign_tensor, false);
    return {grad_outputs[0] * scaled * sign_var};
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

    // grad_y = grad * x / denom
    auto grad_y = div(mul(grad, x), denom);

    // grad_x = grad * (-y) / denom
    auto grad_x = div(mul(grad, neg(y)), denom);

    return {grad_y, grad_x};
}

auto Atan2Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // atan2 backward: grad_y = grad * x / (x^2 + y^2), grad_x = grad * (-y) / (x^2 + y^2)
    Variable y_var(saved_tensors_[0], false);
    Variable x_var(saved_tensors_[1], false);
    auto denom = x_var * x_var + y_var * y_var;
    auto grad_y = grad_outputs[0] * x_var / denom;
    auto grad_x = grad_outputs[0] * tenzor::neg(y_var) / denom;
    return {grad_y, grad_x};
}

} // namespace tenzor
