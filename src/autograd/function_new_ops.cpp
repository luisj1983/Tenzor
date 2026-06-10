#include "tenzor/autograd/function.hpp"
#include "function_helpers.hpp"
#include <cassert>
#include <cstdlib>
#include <limits>
#include <string>
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/safe_math.hpp"
#include <cmath>

namespace tenzor {

// =========================================================================
// Element-wise Binary Op Backward Functions
// =========================================================================

// LogAddExpBackward: logaddexp(a, b) = log(exp(a) + exp(b))
// grad_a = grad * sigmoid(a - b) = grad * exp(a) / (exp(a) + exp(b))
// grad_b = grad * sigmoid(b - a) = grad * exp(b) / (exp(a) + exp(b))
auto LogAddExpBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LogAddExpBackward::forward should not be called directly");
}

auto LogAddExpBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    // sigmoid(a - b) = exp(a - b) / (1 + exp(a - b)) = 1 / (1 + exp(b - a))
    auto grad_a_unreduced = mul(grad, sigmoid(sub(a, b)));
    auto grad_b_unreduced = mul(grad, sigmoid(sub(b, a)));

    auto grad_a = reduce_grad_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

auto LogAddExpBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(2);
    Variable a_var(saved_tensors_[0], false);
    Variable b_var(saved_tensors_[1], false);

    auto grad_a_unreduced = grad_outputs[0] * Variable(sigmoid(sub(saved_tensors_[0], saved_tensors_[1])), false);
    auto grad_b_unreduced = grad_outputs[0] * Variable(sigmoid(sub(saved_tensors_[1], saved_tensors_[0])), false);

    auto grad_a = reduce_grad_var_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_var_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

// LogAddExp2Backward: logaddexp2(a, b) = log2(2^a + 2^b)
// grad_a = grad * 2^a / (2^a + 2^b) = grad * exp2(a - logaddexp2(a, b))
// grad_b = grad * 2^b / (2^a + 2^b) = grad * exp2(b - logaddexp2(a, b))
auto LogAddExp2Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LogAddExp2Backward::forward should not be called directly");
}

auto LogAddExp2Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(3);
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& output = saved_tensors_[2];  // logaddexp2(a, b)
    const auto& grad = grad_outputs[0];

    auto grad_a_unreduced = mul(grad, exp2(sub(a, output)));
    auto grad_b_unreduced = mul(grad, exp2(sub(b, output)));

    auto grad_a = reduce_grad_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

auto LogAddExp2Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(3);
    Variable output_var(saved_tensors_[2], false);

    auto wa = Variable(exp2(sub(saved_tensors_[0], saved_tensors_[2])), false);
    auto wb = Variable(exp2(sub(saved_tensors_[1], saved_tensors_[2])), false);

    auto grad_a_unreduced = grad_outputs[0] * wa;
    auto grad_b_unreduced = grad_outputs[0] * wb;

    auto grad_a = reduce_grad_var_for_broadcasting(grad_a_unreduced, input_shape_a_);
    auto grad_b = reduce_grad_var_for_broadcasting(grad_b_unreduced, input_shape_b_);
    return {grad_a, grad_b};
}

// XLogYBackward: xlogy(x, y) = x * log(y), with 0 * log(y) = 0
// grad_x = grad * log(y)       (with 0-safe: if x == 0, grad_x = 0)
// grad_y = grad * x / y        (with 0-safe: if x == 0, grad_y = 0)
auto XLogYBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("XLogYBackward::forward should not be called directly");
}

auto XLogYBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& x = saved_tensors_[0];
    const auto& y = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    auto x_shape = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    auto zero_x = zeros(x_shape, x.dtype(), x.device());
    auto x_is_zero = eq(x, zero_x);

    // grad_x = grad * log(y), zeroed where x == 0
    auto log_y = tenzor::log(y);
    auto grad_x_raw = mul(grad, log_y);
    auto grad_x_unreduced = where(x_is_zero, zeros_like(grad_x_raw), grad_x_raw);

    // grad_y = grad * x / y, zeroed where x == 0
    auto y_shape = std::vector<int64_t>(y.shape().begin(), y.shape().end());
    auto eps_y = full(y_shape, detail::dtype_epsilon(y.dtype()), y.dtype(), y.device());
    auto safe_y = where(eq(y, zeros(y_shape, y.dtype(), y.device())), eps_y, y);
    auto grad_y_raw = mul(grad, div(x, safe_y));
    auto grad_y_unreduced = where(x_is_zero, zeros_like(grad_y_raw), grad_y_raw);

    auto grad_x = reduce_grad_for_broadcasting(grad_x_unreduced, input_shape_x_);
    auto grad_y = reduce_grad_for_broadcasting(grad_y_unreduced, input_shape_y_);
    return {grad_x, grad_y};
}

auto XLogYBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(2);
    const auto& x = saved_tensors_[0];
    const auto& y = saved_tensors_[1];

    auto x_shape = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    auto x_is_zero = eq(x, zeros(x_shape, x.dtype(), x.device()));

    auto log_y = tenzor::log(y);
    auto grad_x_raw = mul(grad_outputs[0].tensor(), log_y);
    auto grad_x_unreduced_t = where(x_is_zero, zeros_like(grad_x_raw), grad_x_raw);
    auto grad_x_unreduced = Variable(grad_x_unreduced_t, false);

    auto y_shape = std::vector<int64_t>(y.shape().begin(), y.shape().end());
    auto eps_y = full(y_shape, detail::dtype_epsilon(y.dtype()), y.dtype(), y.device());
    auto safe_y = where(eq(y, zeros(y_shape, y.dtype(), y.device())), eps_y, y);
    auto grad_y_raw = mul(grad_outputs[0].tensor(), div(x, safe_y));
    auto grad_y_unreduced_t = where(x_is_zero, zeros_like(grad_y_raw), grad_y_raw);
    auto grad_y_unreduced = Variable(grad_y_unreduced_t, false);

    auto grad_x = reduce_grad_var_for_broadcasting(grad_x_unreduced, input_shape_x_);
    auto grad_y = reduce_grad_var_for_broadcasting(grad_y_unreduced, input_shape_y_);
    return {grad_x, grad_y};
}

// =========================================================================
// Element-wise Unary Op Backward Functions
// =========================================================================

// I0eBackward: i0e(x) = exp(-|x|) * I0(x)
// d/dx i0e(x) = i1e(x) - sign(x) * i0e(x)
auto I0eBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("I0eBackward::forward should not be called directly");
}

auto I0eBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    const auto& grad = grad_outputs[0];

    auto i1e_val = tenzor::i1e(input);
    auto i0e_val = tenzor::i0e(input);
    auto sgn = tenzor::sign(input);
    // derivative = i1e(x) - sign(x) * i0e(x)
    auto deriv = sub(i1e_val, mul(sgn, i0e_val));
    return {mul(grad, deriv)};
}

auto I0eBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    auto deriv = sub(tenzor::i1e(input), mul(tenzor::sign(input), tenzor::i0e(input)));
    Variable deriv_var(deriv, false);
    return {grad_outputs[0] * deriv_var};
}

// I1eBackward: i1e(x) = exp(-|x|) * I1(x)
// d/dx i1e(x) = i0e(x) - i1e(x) * (1/x + sign(x))   for x != 0
// d/dx i1e(0) = 0.5 (limit)
auto I1eBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("I1eBackward::forward should not be called directly");
}

auto I1eBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    const auto& grad = grad_outputs[0];

    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto i0e_val = tenzor::i0e(input);
    auto i1e_val = tenzor::i1e(input);
    auto sgn = tenzor::sign(input);

    // Safe reciprocal: avoid division by zero at x=0
    auto eps = full(input_shape, detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto zero_t = zeros(input_shape, input.dtype(), input.device());
    auto x_is_zero = eq(input, zero_t);
    auto safe_x = where(x_is_zero, eps, input);

    // 1/x + sign(x)
    auto inv_x_plus_sgn = add(reciprocal(safe_x), sgn);
    // i0e(x) - i1e(x) * (1/x + sign(x))
    auto deriv_nonzero = sub(i0e_val, mul(i1e_val, inv_x_plus_sgn));

    // At x=0, the derivative is 0.5
    auto half = full(input_shape, 0.5, input.dtype(), input.device());
    auto deriv = where(x_is_zero, half, deriv_nonzero);

    return {mul(grad, deriv)};
}

auto I1eBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    auto i0e_val = tenzor::i0e(input);
    auto i1e_val = tenzor::i1e(input);
    auto sgn = tenzor::sign(input);

    auto eps = full(input_shape, detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto zero_t = zeros(input_shape, input.dtype(), input.device());
    auto x_is_zero = eq(input, zero_t);
    auto safe_x = where(x_is_zero, eps, input);

    auto inv_x_plus_sgn = add(reciprocal(safe_x), sgn);
    auto deriv_nonzero = sub(i0e_val, mul(i1e_val, inv_x_plus_sgn));
    auto half = full(input_shape, 0.5, input.dtype(), input.device());
    auto deriv = where(x_is_zero, half, deriv_nonzero);

    Variable deriv_var(deriv, false);
    return {grad_outputs[0] * deriv_var};
}

// EntrBackward: entr(x) = -x * log(x) for x > 0, 0 for x == 0, -inf for x < 0
// d/dx entr(x) = -(1 + log(x)) for x > 0, 0 otherwise
auto EntrBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("EntrBackward::forward should not be called directly");
}

auto EntrBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    const auto& grad = grad_outputs[0];

    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto zero_t = zeros(input_shape, input.dtype(), input.device());
    auto one_t = ones(input_shape, input.dtype(), input.device());
    auto x_positive = gt(input, zero_t);

    // For x > 0: -(1 + log(x))
    auto eps = full(input_shape, detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto safe_x = where(x_positive, input, eps);  // avoid log(0)
    auto deriv_pos = neg(add(one_t, tenzor::log(safe_x)));

    auto deriv = where(x_positive, deriv_pos, zero_t);
    return {mul(grad, deriv)};
}

auto EntrBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto zero_t = zeros(input_shape, input.dtype(), input.device());
    auto one_t = ones(input_shape, input.dtype(), input.device());
    auto x_positive = gt(input, zero_t);

    auto eps = full(input_shape, detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto safe_x = where(x_positive, input, eps);
    auto deriv_pos = neg(add(one_t, tenzor::log(safe_x)));
    auto deriv = where(x_positive, deriv_pos, zero_t);

    Variable deriv_var(deriv, false);
    return {grad_outputs[0] * deriv_var};
}

// SphericalBesselJ0Backward: j0(x) = sin(x)/x
// d/dx j0(x) = cos(x)/x - sin(x)/(x*x) for x != 0, 0 for x = 0
auto SphericalBesselJ0Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SphericalBesselJ0Backward::forward should not be called directly");
}

auto SphericalBesselJ0Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    const auto& grad = grad_outputs[0];

    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto zero_t = zeros(input_shape, input.dtype(), input.device());
    auto eps = full(input_shape, detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto x_is_zero = lt(abs(input), eps);
    auto safe_x = where(x_is_zero, ones(input_shape, input.dtype(), input.device()), input);

    // cos(x)/x - sin(x)/(x*x)
    auto cos_x = tenzor::cos(safe_x);
    auto sin_x = tenzor::sin(safe_x);
    auto x_sq = mul(safe_x, safe_x);
    auto deriv_nonzero = sub(div(cos_x, safe_x), div(sin_x, x_sq));

    auto deriv = where(x_is_zero, zero_t, deriv_nonzero);
    return {mul(grad, deriv)};
}

auto SphericalBesselJ0Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto zero_t = zeros(input_shape, input.dtype(), input.device());
    auto eps = full(input_shape, detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto x_is_zero = lt(abs(input), eps);
    auto safe_x = where(x_is_zero, ones(input_shape, input.dtype(), input.device()), input);

    auto cos_x = tenzor::cos(safe_x);
    auto sin_x = tenzor::sin(safe_x);
    auto x_sq = mul(safe_x, safe_x);
    auto deriv_nonzero = sub(div(cos_x, safe_x), div(sin_x, x_sq));
    auto deriv = where(x_is_zero, zero_t, deriv_nonzero);

    Variable deriv_var(deriv, false);
    return {grad_outputs[0] * deriv_var};
}

// =========================================================================
// Statistical/Special Op Backward Functions
// =========================================================================

// NdtrBackward: ndtr(x) = 0.5 * erfc(-x / sqrt(2))
// d/dx ndtr(x) = (1/sqrt(2*pi)) * exp(-x^2/2)  (standard normal PDF)
auto NdtrBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("NdtrBackward::forward should not be called directly");
}

auto NdtrBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    const auto& grad = grad_outputs[0];

    constexpr double inv_sqrt_2pi = 0.3989422804014327;  // 1/sqrt(2*pi)
    auto x_sq = mul(input, input);
    auto neg_half_x_sq = mul(x_sq, -0.5);
    auto pdf = mul(exp(neg_half_x_sq), inv_sqrt_2pi);
    return {mul(grad, pdf)};
}

auto NdtrBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];

    constexpr double inv_sqrt_2pi = 0.3989422804014327;
    auto x_sq = mul(input, input);
    auto neg_half_x_sq = mul(x_sq, -0.5);
    auto pdf = mul(exp(neg_half_x_sq), inv_sqrt_2pi);

    Variable pdf_var(pdf, false);
    return {grad_outputs[0] * pdf_var};
}

// LogNdtrBackward: log_ndtr(x) = log(ndtr(x))
// d/dx log_ndtr(x) = ndtr'(x) / ndtr(x) = pdf(x) / ndtr(x)
// = (1/sqrt(2*pi)) * exp(-x^2/2) / ndtr(x)
// For numerical stability when ndtr(x) is very small (large negative x),
// we use: exp(-x^2/2 - log_ndtr(x)) / sqrt(2*pi)
auto LogNdtrBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LogNdtrBackward::forward should not be called directly");
}

auto LogNdtrBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& log_ndtr_val = saved_tensors_[1];  // log(ndtr(x)), saved output
    const auto& grad = grad_outputs[0];

    constexpr double log_inv_sqrt_2pi = -0.9189385332046727;  // log(1/sqrt(2*pi))
    auto x_sq = mul(input, input);
    auto log_pdf = add(mul(x_sq, -0.5), log_inv_sqrt_2pi);
    // exp(log_pdf - log_ndtr) = pdf / ndtr
    auto ratio = exp(sub(log_pdf, log_ndtr_val));
    return {mul(grad, ratio)};
}

auto LogNdtrBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& log_ndtr_val = saved_tensors_[1];

    constexpr double log_inv_sqrt_2pi = -0.9189385332046727;
    auto x_sq = mul(input, input);
    auto log_pdf = add(mul(x_sq, -0.5), log_inv_sqrt_2pi);
    auto ratio = exp(sub(log_pdf, log_ndtr_val));

    Variable ratio_var(ratio, false);
    return {grad_outputs[0] * ratio_var};
}

// MultigammalnBackward: multigammaln(x, p) = sum_{j=0}^{p-1} lgamma(x - j/2)
// d/dx multigammaln(x, p) = sum_{j=0}^{p-1} digamma(x - j/2)
auto MultigammalnBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("MultigammalnBackward::forward should not be called directly");
}

auto MultigammalnBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    const auto& grad = grad_outputs[0];

    auto result = zeros_like(input);
    for (int64_t j = 0; j < p_; ++j) {
        double offset = static_cast<double>(j) / 2.0;
        auto shifted = sub(input, offset);
        result = add(result, tenzor::digamma(shifted));
    }
    return {mul(grad, result)};
}

auto MultigammalnBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];

    auto result = zeros_like(input);
    for (int64_t j = 0; j < p_; ++j) {
        double offset = static_cast<double>(j) / 2.0;
        auto shifted = sub(input, offset);
        result = add(result, tenzor::digamma(shifted));
    }

    Variable deriv_var(result, false);
    return {grad_outputs[0] * deriv_var};
}

// =========================================================================
// Reduction Op Backward Functions
// =========================================================================

// CosineSimilarityBackward:
// cos_sim(x1, x2) = dot(x1, x2) / (norm(x1) * norm(x2))
// grad_x1 = grad * (x2 / (n1 * n2) - cos_sim * x1 / (n1 * n1)) along dim
// grad_x2 = grad * (x1 / (n1 * n2) - cos_sim * x2 / (n2 * n2)) along dim
auto CosineSimilarityBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CosineSimilarityBackward::forward should not be called directly");
}

auto CosineSimilarityBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(3);
    const auto& x1 = saved_tensors_[0];
    const auto& x2 = saved_tensors_[1];
    const auto& output = saved_tensors_[2];  // cos_sim value
    const auto& grad = grad_outputs[0];

    auto x1_shape = std::vector<int64_t>(x1.shape().begin(), x1.shape().end());

    // Compute norms along dim
    auto n1_sq = tenzor::sum(mul(x1, x1), dim_, true);
    auto n2_sq = tenzor::sum(mul(x2, x2), dim_, true);

    auto eps_t = full(std::vector<int64_t>(n1_sq.shape().begin(), n1_sq.shape().end()),
                      eps_, n1_sq.dtype(), n1_sq.device());
    auto n1 = sqrt(tenzor::where(gt(n1_sq, eps_t), n1_sq, eps_t));
    auto n2 = sqrt(tenzor::where(gt(n2_sq, eps_t), n2_sq, eps_t));
    auto n1n2 = mul(n1, n2);

    // Expand grad and output to input shape
    auto grad_expanded = grad;
    auto output_expanded = output;
    if (grad.ndim() < x1.ndim()) {
        grad_expanded = unsqueeze(grad, dim_);
        output_expanded = unsqueeze(output, dim_);
    }
    grad_expanded = expand(grad_expanded, x1_shape);
    output_expanded = expand(output_expanded, x1_shape);

    auto n1n2_expanded = expand(n1n2, x1_shape);
    auto n1_sq_expanded = expand(n1_sq, x1_shape);
    auto n2_sq_expanded = expand(n2_sq, x1_shape);

    // grad_x1 = grad * (x2 / (n1*n2) - cos_sim * x1 / n1^2)
    auto grad_x1 = mul(grad_expanded,
        sub(div(x2, n1n2_expanded),
            mul(output_expanded, div(x1, n1_sq_expanded))));

    // grad_x2 = grad * (x1 / (n1*n2) - cos_sim * x2 / n2^2)
    auto grad_x2 = mul(grad_expanded,
        sub(div(x1, n1n2_expanded),
            mul(output_expanded, div(x2, n2_sq_expanded))));

    return {grad_x1, grad_x2};
}

auto CosineSimilarityBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Saved (x1, x2, output) are constants;
    // the per-input Jacobian factors
    //   deriv1 = x2 / (n1*n2) - output * x1 / n1^2
    //   deriv2 = x1 / (n1*n2) - output * x2 / n2^2
    // depend only on those constants. Compute deriv1, deriv2 at tensor
    // level once, wrap each as a non-diff Variable, then
    //   grad_x{1,2} = grad_expanded * deriv{1,2}
    // is a Variable-level multiply that preserves grad_fn through
    // grad_outputs[0].
    require_saved_tensors(3);
    const Tensor& x1 = saved_tensors_[0];
    const Tensor& x2 = saved_tensors_[1];
    const Tensor& output = saved_tensors_[2];
    auto x1_shape = std::vector<int64_t>(x1.shape().begin(), x1.shape().end());

    auto n1_sq = tenzor::sum(mul(x1, x1), dim_, true);
    auto n2_sq = tenzor::sum(mul(x2, x2), dim_, true);
    auto eps_t = full(std::vector<int64_t>(n1_sq.shape().begin(), n1_sq.shape().end()),
                      eps_, n1_sq.dtype(), n1_sq.device());
    auto n1 = sqrt(tenzor::where(gt(n1_sq, eps_t), n1_sq, eps_t));
    auto n2 = sqrt(tenzor::where(gt(n2_sq, eps_t), n2_sq, eps_t));
    auto n1n2 = mul(n1, n2);
    auto output_expanded = output;
    if (output.ndim() < x1.ndim()) {
        output_expanded = unsqueeze(output, dim_);
    }
    output_expanded = expand(output_expanded, x1_shape);
    auto n1n2_expanded = expand(n1n2, x1_shape);
    auto n1_sq_expanded = expand(n1_sq, x1_shape);
    auto n2_sq_expanded = expand(n2_sq, x1_shape);

    auto deriv1 = sub(div(x2, n1n2_expanded),
                      mul(output_expanded, div(x1, n1_sq_expanded)));
    auto deriv2 = sub(div(x1, n1n2_expanded),
                      mul(output_expanded, div(x2, n2_sq_expanded)));
    Variable deriv1_var(deriv1, /*requires_grad=*/false);
    Variable deriv2_var(deriv2, /*requires_grad=*/false);

    Variable grad_expanded = grad_outputs[0];
    if (grad_expanded.tensor().ndim() < static_cast<int64_t>(x1_shape.size())) {
        grad_expanded = tenzor::unsqueeze(grad_expanded, dim_);
    }
    grad_expanded = tenzor::expand(grad_expanded, x1_shape);
    return {grad_expanded * deriv1_var, grad_expanded * deriv2_var};
}

// RenormBackward:
// renorm scales slices along dim so that their p-norm <= maxnorm.
// scale_i = min(1, maxnorm / norm_i)
// grad_input = grad_output * scale  (+ subgradient correction at boundary)
auto RenormBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("RenormBackward::forward should not be called directly");
}

auto RenormBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];  // y = renorm(input)
    const auto& grad = grad_outputs[0];       // dL/dy

    // For each slice along dim_ the op is y = s · x with
    //   s = min(1, maxnorm / ||x||_p),     ||x||_p = (Σ_j |x_j|^p)^{1/p}.
    // If s == 1 (norm below maxnorm) the map is identity and ∂y/∂x = I.
    // Otherwise s = maxnorm / ||x||_p and
    //   ∂y_i/∂x_j = s · δ_ij − s · x_i · sign(x_j)|x_j|^{p-1} / ||x||_p^p.
    // Back-propagating the chain rule through one slice:
    //   (∂L/∂x)_j = s · (∂L/∂y)_j
    //             − clipped · s · sign(x_j)|x_j|^{p-1} · ⟨∂L/∂y, x⟩ / ||x||_p^p.
    // The previous implementation returned only the first term — it
    // produced a uniform-per-column gradient that ignored the rank-one
    // correction active on clipped slices, and gradcheck flagged it.

    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Per-slice scale s = y / x (safe at x==0: y is also 0 there, so we
    // clamp the divisor to ±eps).
    auto eps_t = full(input_shape, detail::dtype_epsilon(input.dtype()),
                      input.dtype(), input.device());
    auto zeros_t = zeros(input_shape, input.dtype(), input.device());
    auto safe_input = where(eq(input, zeros_t), eps_t, input);
    auto scale_tensor = div(output, safe_input);

    // Reduce |x|^p over all dims except dim_ to recover ||x||_p^p per
    // slice (kept with size-1 dims so broadcasting works).
    auto abs_x = abs(input);
    Tensor xp_contrib = (p_ == 2.0)
        ? mul(abs_x, abs_x)
        : pow(abs_x, p_);
    Tensor norm_p_per_slice = xp_contrib;
    // Sum each non-dim_ dimension while keeping shape via keepdim.
    for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim_) continue;
        norm_p_per_slice = sum(norm_p_per_slice, d, /*keepdim=*/true);
    }

    // "clipped" indicator broadcast over the slice: 1 where s < 1 − ε.
    auto one_t = full(std::vector<int64_t>(norm_p_per_slice.shape().begin(), norm_p_per_slice.shape().end()), 1.0,
                      input.dtype(), input.device());
    // Per-slice scale value (just pick one element; all entries in the
    // slice share the same scale by construction).
    Tensor scale_per_slice = scale_tensor;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim_) continue;
        // A 0-th element slice is sufficient since every entry in the
        // non-dim_ direction carries the same scale factor.
        scale_per_slice = sum(scale_per_slice, d, /*keepdim=*/true);
        auto slice_count = full(std::vector<int64_t>(norm_p_per_slice.shape().begin(), norm_p_per_slice.shape().end()),
                                static_cast<double>(input_shape[d]),
                                input.dtype(), input.device());
        scale_per_slice = div(scale_per_slice, slice_count);
    }
    auto clipped = lt(scale_per_slice, sub(one_t, eps_t));  // bool tensor
    auto clipped_f = clipped.to(input.dtype());

    // inner = ⟨grad, x⟩ reduced over non-dim_ dims (keepdim for broadcast)
    Tensor inner = mul(grad, input);
    for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim_) continue;
        inner = sum(inner, d, /*keepdim=*/true);
    }

    // sign(x) · |x|^{p-1} for the correction term.
    Tensor sgn = sign(input);
    Tensor corr_factor = (p_ == 2.0)
        ? input                                 // sign(x)·|x|^{1} == x for p=2
        : mul(sgn, pow(abs_x, p_ - 1.0));
    // Avoid 0/0 by clamping norm_p.
    auto norm_p_safe = add(norm_p_per_slice, eps_t);
    Tensor correction = mul(div(mul(mul(corr_factor, inner), clipped_f),
                                norm_p_safe),
                            scale_per_slice);

    auto grad_in = sub(mul(grad, scale_tensor), correction);
    return {grad_in};
}

auto RenormBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Forward y = s * x with s = min(1, maxnorm/||x||_p).
    // Full backward (see backward() above) is
    //   grad_x = grad * scale  -  scale_per_slice * clipped * corr_factor *
    //                              sum(grad * input, non_dim_, keepdim) / norm_p^p
    // Constants depend only on saved (input, output); the `inner` reduction
    // depends on grad. Wrap constants as non-diff Variables; express `inner`
    // and the assembly via Variable ops so grad_fn flows through grad. The
    // previous body dropped the correction term entirely.
    require_saved_tensors(2);
    const Tensor& input = saved_tensors_[0];
    const Tensor& output = saved_tensors_[1];
    const Variable& grad_var = grad_outputs[0];

    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    auto eps_t = full(input_shape, detail::dtype_epsilon(input.dtype()),
                      input.dtype(), input.device());
    auto zeros_t = zeros(input_shape, input.dtype(), input.device());
    auto safe_input = where(eq(input, zeros_t), eps_t, input);
    auto scale_tensor = div(output, safe_input);

    auto abs_x = abs(input);
    Tensor xp_contrib = (p_ == 2.0) ? mul(abs_x, abs_x) : pow(abs_x, p_);
    Tensor norm_p_per_slice = xp_contrib;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim_) continue;
        norm_p_per_slice = sum(norm_p_per_slice, d, /*keepdim=*/true);
    }
    auto one_t = full(std::vector<int64_t>(norm_p_per_slice.shape().begin(),
                                            norm_p_per_slice.shape().end()),
                      1.0, input.dtype(), input.device());
    Tensor scale_per_slice = scale_tensor;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim_) continue;
        scale_per_slice = sum(scale_per_slice, d, /*keepdim=*/true);
        auto slice_count = full(std::vector<int64_t>(norm_p_per_slice.shape().begin(),
                                                      norm_p_per_slice.shape().end()),
                                static_cast<double>(input_shape[d]),
                                input.dtype(), input.device());
        scale_per_slice = div(scale_per_slice, slice_count);
    }
    auto clipped = lt(scale_per_slice, sub(one_t, eps_t));
    auto clipped_f = clipped.to(input.dtype());
    Tensor sgn = sign(input);
    Tensor corr_factor = (p_ == 2.0) ? input : mul(sgn, pow(abs_x, p_ - 1.0));
    auto norm_p_safe = add(norm_p_per_slice, eps_t);

    Variable scale_var(scale_tensor, /*requires_grad=*/false);
    Variable input_var(input, /*requires_grad=*/false);
    Variable corr_per_norm_var(
        mul(div(mul(corr_factor, clipped_f), norm_p_safe), scale_per_slice),
        /*requires_grad=*/false);

    Variable inner_var = grad_var * input_var;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim_) continue;
        inner_var = tenzor::sum(inner_var, d, /*keepdim=*/true);
    }
    auto correction = corr_per_norm_var * inner_var;
    auto grad_in = grad_var * scale_var - correction;
    return {grad_in};
}

// =========================================================================
// Linear Algebra Op Backward Functions
// =========================================================================

// CholeskyInverseBackward:
// Forward: A^{-1} = cholesky_inverse(L) where A = L @ L^T
// Backward: dL/dL = -(L^{-T} @ (dL/d(A^{-1}) + dL/d(A^{-1})^T) @ A^{-1}) @ L^{-T}
// Simplified: use the identity that A^{-1} = inv(L^T) @ inv(L)
// Then grad flows through the inv and cholesky structure.
auto CholeskyInverseBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CholeskyInverseBackward::forward should not be called directly");
}

auto CholeskyInverseBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& L = saved_tensors_[0];          // Cholesky factor
    const auto& Ainv = saved_tensors_[1];        // A^{-1} = cholesky_inverse(L)
    const auto& grad = grad_outputs[0];          // dL/d(A^{-1})

    auto ndim = L.ndim();

    // Symmetrize gradient: (grad + grad^T) since A^{-1} is symmetric
    auto grad_sym = mul(add(grad, transpose(grad, ndim - 2, ndim - 1)), 0.5);

    // dL/dA^{-1} = grad_sym
    // d(A^{-1})/dL: A^{-1} = (L L^T)^{-1}
    // Using the chain rule through matrix inverse:
    // grad_A = -Ainv @ grad_sym @ Ainv
    auto temp = matmul(matmul(Ainv, grad_sym), Ainv);
    auto neg_temp = neg(temp);

    // Now grad_A is the gradient w.r.t. A = L @ L^T
    // Chain through A = L @ L^T:
    // dL/dL = (grad_A + grad_A^T) @ L
    auto grad_A_sym = add(neg_temp, transpose(neg_temp, ndim - 2, ndim - 1));
    auto grad_L = matmul(grad_A_sym, L);

    // Only lower triangular part matters since L is lower triangular
    if (upper_) {
        grad_L = triu(grad_L);
    } else {
        grad_L = tril(grad_L);
    }

    return {grad_L};
}

auto CholeskyInverseBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Forward A^{-1} = cholesky_inverse(L)
    // with A = L @ L^T. With saved (L, A^{-1}) as constants:
    //   grad_sym = (grad + grad^T) / 2
    //   temp     = A^{-1} @ grad_sym @ A^{-1}
    //   grad_A   = (-temp) + (-temp)^T
    //   grad_L   = grad_A @ L
    //   grad_L   = tril(grad_L)   (or triu if upper_)
    // All ops are Variable-level so grad_fn flows through grad.
    require_saved_tensors(2);
    Variable L_var(saved_tensors_[0], /*requires_grad=*/false);
    Variable Ainv_var(saved_tensors_[1], /*requires_grad=*/false);
    const int64_t ndim = saved_tensors_[0].ndim();

    auto& grad = grad_outputs[0];
    auto grad_T = tenzor::transpose(grad, ndim - 2, ndim - 1);
    auto grad_sym = (grad + grad_T) * 0.5;

    auto temp = tenzor::matmul(tenzor::matmul(Ainv_var, grad_sym), Ainv_var);
    auto neg_temp = tenzor::neg(temp);
    auto neg_temp_T = tenzor::transpose(neg_temp, ndim - 2, ndim - 1);
    auto grad_A_sym = neg_temp + neg_temp_T;
    auto grad_L = tenzor::matmul(grad_A_sym, L_var);
    if (upper_) {
        grad_L = tenzor::triu(grad_L);
    } else {
        grad_L = tenzor::tril(grad_L);
    }
    return {grad_L};
}

// LinalgLDLFactorBackward: structured backprop through A = L D L^T.
//
// We implement the exact closed-form adjoint for the NO-PIVOTING case (the
// factorization LAPACK *sytrf produces when no row/column interchanges and no
// 2x2 pivot blocks are needed, i.e. the SPD-equivalent path). The derivation
// is in backward() below.
//
// Bunch-Kaufman pivoting (the reason LDL is used for INDEFINITE matrices)
// permutes the matrix as P A P^T = L D L^T, so tril(LD,-1)+I is the factor of
// the permuted matrix, not of A. Rather than return a silently-wrong gradient
// in that case, backward() inspects the saved pivots and throws
// NonDifferentiable when any pivoting occurred (matching PyTorch's
// linalg_ldl_factor, which only supports the pivot-free adjoint). Setting
// TENZOR_STRICT_LINALG_GRAD=1 additionally hardens the diagnostic. Users needing
// gradients through an indefinite factorization should use `ldl_solve` (which
// has a proper pivot-aware backward) or `cholesky` for SPD inputs.
auto LinalgLDLFactorBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LinalgLDLFactorBackward::forward should not be called directly");
}

namespace {
inline bool strict_linalg_grad_mode() {
    const char* v = std::getenv("TENZOR_STRICT_LINALG_GRAD");
    return v && *v && std::string(v) != "0" && std::string(v) != "false";
}

// The closed-form LDL backward below is derived for the no-pivoting case
// (A = L D L^T with L unit-lower). LAPACK *sytrf (Bunch-Kaufman) pivots for
// indefinite matrices: ipiv is 1-based, a 1x1 block with no interchange has
// ipiv[i] == i+1, an interchange gives a different positive value, and a 2x2
// pivot block gives negative entries. When any pivoting occurred, tril(LD,-1)+I
// is the factor of a PERMUTED matrix, so the closed form would be silently
// wrong. Treat the pivots as trivial only when every entry is its own 1-based
// index.
inline bool ldl_pivots_are_trivial(const Tensor& pivots) {
    Tensor p = (pivots.device() == Device::cpu()) ? pivots : pivots.to(Device::cpu());
    if (p.dtype() != DType::Int32) p = p.to(DType::Int32);
    const int32_t* d = p.data<int32_t>();
    const int64_t n = p.shape().empty() ? 0 : p.shape().back();
    const int64_t total = p.numel();
    if (n == 0) return true;
    for (int64_t f = 0; f < total; ++f) {
        if (d[f] != static_cast<int32_t>((f % n) + 1)) return false;
    }
    return true;
}
}

auto LinalgLDLFactorBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(3);
    const auto& A = saved_tensors_[0];
    const auto& LD = saved_tensors_[1];
    const auto& pivots = saved_tensors_[2];
    const auto& grad_LD = grad_outputs[0];

    // The closed form below is only valid without pivoting. If Bunch-Kaufman
    // interchanged rows/cols or used a 2x2 block, LD factors the permuted
    // matrix and this adjoint would be silently wrong — fail loud instead.
    if (!ldl_pivots_are_trivial(pivots)) {
        throw NonDifferentiable(
            "linalg.ldl_factor: gradients are only supported for the "
            "no-pivoting (positive-definite-equivalent) case. The input "
            "required Bunch-Kaufman pivoting (it is indefinite), for which the "
            "factor backward is not implemented. Differentiate through "
            "`ldl_solve` (pivot-aware), or use `cholesky` for SPD inputs.");
    }
    (void)strict_linalg_grad_mode();

    // audit-2026-05-03 Phase 12 — implement closed-form LDL backward for
    // the no-pivoting (SPD-equivalent) path. LAPACK's *sytrf with UPLO='L'
    // packs L|D into the lower triangle of LD; the upper triangle of LD
    // remains the upper triangle of A (untouched), so its gradient flows
    // directly back to A's upper triangle.
    //
    // Derivation (no pivoting, A = L D L^T, L unit lower, D diagonal):
    //   dA = dL · D · L^T + L · dD · L^T + L · D · dL^T
    //   M  = L^{-1} dA L^{-T} = X D + dD + D X^T, where X = L^{-1} dL
    //   so dD_i = M_ii and X_ji = M_ji / D_ii (j > i, strict lower).
    // Adjoint:
    //   given grad_L (strict lower) and grad_D (diag), let
    //     Q = strict_lower(L^T grad_L)
    //     R_ji = Q_ji / D_ii  (strict lower)
    //     S    = diag(grad_D)
    //   then  grad_A = L^{-T} (S + R) L^{-1}  (lower & diag entries).
    // The strict-upper of grad_A comes directly from grad_LD's strict-upper.

    int64_t n = A.shape().back();
    auto eye = tenzor::eye(n, std::nullopt, A.dtype(), A.device());

    // Extract structural pieces of LD.
    auto L_strict = tenzor::tril(LD, -1);   // strict lower
    auto L = L_strict + eye;                // unit lower triangular
    auto LT = tenzor::transpose(L, -1, -2);

    // Diagonal of LD as a length-n vector.
    // diag() of a 2D matrix returns its diagonal as a 1D vector.
    auto D_diag = tenzor::diag(LD, 0);

    // grad_L_strict (strict lower) and grad_D_diag (diagonal).
    auto grad_L_strict = tenzor::tril(grad_LD, -1);
    auto grad_D_diag = tenzor::diag(grad_LD, 0);

    // Q = strict_lower(L^T grad_L_strict)
    auto P = tenzor::matmul(LT, grad_L_strict);
    auto Q = tenzor::tril(P, -1);

    // R_ji = Q_ji / D_ii  — divide row j (which equals i index in our matmul
    // convention) by D_ii. In our column-of-Q layout, Q_ji is at (j, i) with
    // j > i; divide by D_ii (i.e. the column's diagonal entry of D).
    // Broadcasting D_diag as a row vector divides each column by D_ii.
    auto D_row = D_diag.unsqueeze(-2);  // [..., 1, n]
    auto R = Q / D_row;

    // S = diag-embedded(grad_D_diag).
    auto S = tenzor::linalg::diag_embed(grad_D_diag);

    auto S_plus_R = S + R;

    // Compute L^{-1} via triangular solve, L · L_inv = I.
    auto L_inv = tenzor::linalg::solve_triangular(L, eye, /*upper=*/false,
                                                  /*unitriangular=*/true);
    auto LT_inv = tenzor::transpose(L_inv, -1, -2);

    auto grad_A_factor = tenzor::matmul(tenzor::matmul(LT_inv, S_plus_R), L_inv);

    // Note on structure: grad_A_factor as computed is asymmetric in general,
    // but its SYMMETRIC PART equals the correct symmetric gradient
    // L^{-T} (K + K^T)/2 L^{-1}. Returning the asymmetric form lets the
    // upstream matmul backward (e.g. A = v · v^T) produce the right grad_v
    // via (grad_A + grad_A^T) · v. We MUST NOT mask off the upper triangle —
    // doing so destroys the symmetric structure and breaks gradcheck.
    //
    // LAPACK leaves LD's strict-upper triangle equal to A's input strict
    // upper (it's read-but-not-written when UPLO='L'). Strict-upper of
    // grad_LD therefore flows back to A's strict upper directly.
    auto upper_mask = tenzor::triu(tenzor::ones_like(A), 1);
    auto grad_A_upper = grad_LD * upper_mask;

    auto grad_A = grad_A_factor + grad_A_upper;
    std::vector<Tensor> out;
    out.push_back(grad_A);
    return out;
}

// Audit B.3: real higher-order backward via Variable-level composition.
// Re-implements the Smith (1995) closed-form Tensor-level backward above
// using Variable-level ops. The only non-trivial replacement is the
// `solve_triangular(L, I)` call — since `L` is unit-lower-triangular and
// therefore invertible, `linalg::inv(L)` produces the same `L^{-1}` and is
// available as a Variable-level op with its own real higher-order
// backward, so reverse-mode autograd through the entire composition
// produces correct second-order gradients.
auto LinalgLDLFactorBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    require_saved_tensors(3);
    // Same pivot restriction as the Tensor-level backward (see note above).
    if (!ldl_pivots_are_trivial(saved_tensors_[2])) {
        throw NonDifferentiable(
            "linalg.ldl_factor: gradients are only supported for the "
            "no-pivoting (positive-definite-equivalent) case. The input "
            "required Bunch-Kaufman pivoting (it is indefinite), for which the "
            "factor backward is not implemented. Differentiate through "
            "`ldl_solve` (pivot-aware), or use `cholesky` for SPD inputs.");
    }
    auto grad_LD = grad_outputs[0];
    auto A_var = Variable(saved_tensors_[0], false);
    auto LD_var = Variable(saved_tensors_[1], false);

    const auto& A_t = A_var.tensor();
    int64_t n = A_t.shape().back();
    auto eye_t = tenzor::eye(n, std::nullopt, A_t.dtype(), A_t.device());
    auto eye_v = Variable(eye_t, false);

    // L_strict = tril(LD, -1)   L = L_strict + I
    auto L_strict = tenzor::tril(LD_var, -1);
    auto L = L_strict + eye_v;
    int64_t ndim = LD_var.tensor().ndim();
    auto Lt = tenzor::transpose(L, ndim - 2, ndim - 1);

    auto D_diag = tenzor::diag(LD_var, 0);

    auto grad_L_strict = tenzor::tril(grad_LD, -1);
    auto grad_D_diag = tenzor::diag(grad_LD, 0);

    // P = L^T @ grad_L_strict; Q = tril(P, -1)
    auto P = tenzor::matmul(Lt, grad_L_strict);
    auto Q = tenzor::tril(P, -1);

    // R = Q / D_diag broadcast as row.
    auto D_row = tenzor::unsqueeze(D_diag, -2);
    auto R = Q / D_row;

    // S = diag_embed(grad_D_diag) — build via eye * grad_D_diag.unsqueeze(-2)
    auto grad_D_row = tenzor::unsqueeze(grad_D_diag, -2);
    auto S = eye_v * grad_D_row;

    auto S_plus_R = S + R;

    // L_inv = inv(L) — L is unit-lower-triangular so this is well-defined
    // and the Variable-level `inv` has its own higher-order backward.
    auto L_inv = tenzor::inv(L);
    auto LT_inv = tenzor::transpose(L_inv, ndim - 2, ndim - 1);

    auto grad_A_factor = tenzor::matmul(tenzor::matmul(LT_inv, S_plus_R), L_inv);

    // Upper-triangular passthrough — LAPACK leaves LD's strict-upper equal
    // to A's input strict-upper, so its gradient flows directly back.
    auto ones_mat_t = tenzor::ones(std::vector<int64_t>(A_t.shape().begin(), A_t.shape().end()),
                                    A_t.dtype(), A_t.device());
    auto upper_mask = Variable(tenzor::triu(ones_mat_t, 1), false);
    auto grad_A_upper = grad_LD * upper_mask;

    auto grad_A = grad_A_factor + grad_A_upper;
    return {grad_A};
}

// LinalgLDLSolveBackward:
// Forward: X = ldl_solve(LD, piv, B)  solves AX = B where A = LDL^T
// Backward: grad_B = ldl_solve(LD, piv, grad)
auto LinalgLDLSolveBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LinalgLDLSolveBackward::forward should not be called directly");
}

auto LinalgLDLSolveBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& LD = saved_tensors_[0];
    const auto& pivots = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    // grad_B = ldl_solve(LD, piv, grad)  (A^{-T} = A^{-1} since A is symmetric)
    auto grad_B = tenzor::linalg::ldl_solve(LD, pivots, grad);
    return {grad_B};
}

auto LinalgLDLSolveBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Forward X = ldl_solve(LD, pivots, B);
    // A = LDL^T is symmetric so A^{-T} = A^{-1} and
    //   grad_B = ldl_solve(LD, pivots, grad_X).
    // LD is saved as a non-diff Variable; pivots is integer non-diff. The
    // autograd-aware `tenzor::ldl_solve(Variable, Tensor, Variable)` keeps
    // grad_fn live through grad_outputs[0].
    require_saved_tensors(2);
    Variable LD_var(saved_tensors_[0], /*requires_grad=*/false);
    const Tensor& pivots = saved_tensors_[1];
    auto grad_B = tenzor::ldl_solve(LD_var, pivots, grad_outputs[0]);
    return {grad_B};
}

// LinalgHouseholderBackward:
// Forward: Q = householder_product(input, tau)
// Backward is complex; use passthrough stub (structural zero for most use cases)
auto LinalgHouseholderBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LinalgHouseholderBackward::forward should not be called directly");
}

auto LinalgHouseholderBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& V_orig = saved_tensors_[0];     // (..., m, k)  Householder reflectors packed column-wise
    const auto& tau_orig = saved_tensors_[1];   // (..., k)     scalar factors
    const auto& G_orig = grad_outputs[0];       // (..., m, n)  upstream grad of Q

    // Closed-form backward for Q = H_0 · H_1 · … · H_{k-1},
    //   H_j = I − τ_j v_j v_jᵀ,   v_j[i<j] = 0,  v_j[j] = 1,  v_j[i>j] = V[i, j].
    //
    // Let   P_j = (H_{j-1} … H_0)·G     (gradient propagated "to the left" of H_j)
    //       B_j = (H_{j+1} … H_{k-1})[:, :n]   (truncated product to the right of H_j)
    // with the recurrences P_0 = G, P_{j+1} = H_j P_j and B_{-1} = Q_trunc,
    // B_j = H_j B_{j-1} (both are H_j-symmetric so the inverse is itself).
    //
    // Walking each reflector independently gives:
    //   ∂L/∂τ_j = −(v_jᵀ P_j) · (v_jᵀ B_j)ᵀ
    //   ∂L/∂v_j = −τ_j · ( P_j (B_jᵀ v_j) + B_j (P_jᵀ v_j) )
    // The v_j components with i ≤ j are frozen (0 above the diagonal, 1 on the
    // diagonal), so only the strictly-below-diagonal part of ∂L/∂v_j lands in
    // ∂L/∂V[:, j].
    //
    // This matches LAPACK's sorgqr convention (Q = H(1) H(2) … H(k)) used by
    // linalg::householder_product above and mirrors the derivation in
    // Walter & Lehmann (2011) "Algorithmic differentiation of QR".

    const DType orig_dtype = V_orig.dtype();
    // Widen reduced-precision floats so accumulated rank-one updates don't
    // lose significance — the recurrence applies k sequential rank-1 edits
    // to an m×n matrix, each cancelling on the order of the slice norm.
    const DType compute_dtype =
        (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16)
            ? DType::Float32
            : orig_dtype;

    const Device orig_device = V_orig.device();

    auto V_shape_v = V_orig.shape();
    const int64_t ndim = static_cast<int64_t>(V_shape_v.size());
    if (ndim < 2) {
        throw std::runtime_error(
            "householder_product backward: input must be at least 2D");
    }
    const int64_t m = V_shape_v[ndim - 2];
    const int64_t k = V_shape_v[ndim - 1];
    const int64_t n = G_orig.shape()[G_orig.ndim() - 1];

    if (compute_dtype != DType::Float32 && compute_dtype != DType::Float64) {
        throw std::runtime_error(
            "householder_product backward: unsupported dtype (only Float32/Float64)");
    }

    // Cast saved tensors to compute dtype on the same device.
    auto to_compute = [&](const Tensor& t) {
        Tensor out = t;
        if (out.dtype() != compute_dtype) out = out.to(compute_dtype);
        if (!out.is_contiguous()) out = out.contiguous();
        return out;
    };
    Tensor V_c = to_compute(V_orig);
    Tensor tau_c = to_compute(tau_orig);
    Tensor G_c = to_compute(G_orig);

    // Reflector v_j has structural pattern:
    //   v_j[i] = 0      for i < j   ← mask = 0, override = 0
    //   v_j[i] = 1      for i = j   ← mask = 0, override = 1
    //   v_j[i] = V[i,j] for i > j   ← mask = 1, override = 0
    // Encode the structure as two (m, k) device tensors and slice column j.
    //   M_full[i, j] = (i > j) ? 1 : 0   →  tril(ones({m,k}), -1)
    //   O_full[i, j] = (i == j) ? 1 : 0  →  eye(m, k)
    Tensor M_2d = tenzor::tril(tenzor::ones({m, k}, compute_dtype, orig_device), -1);
    Tensor O_2d = tenzor::eye(m, k, compute_dtype, orig_device);
    Tensor M_full = M_2d;
    Tensor O_full = O_2d;
    if (ndim > 2) {
        std::vector<int64_t> bshape(V_shape_v.begin(), V_shape_v.end());
        M_full = tenzor::expand(M_2d, bshape).contiguous();
        O_full = tenzor::expand(O_2d, bshape).contiguous();
    }

    // Build v_j of shape (..., m, 1) from precomputed mask/override columns.
    auto build_v = [&](int64_t j) -> Tensor {
        Tensor V_col = tenzor::slice(V_c, /*dim=*/-1, j, j + 1).contiguous();
        Tensor mask  = tenzor::slice(M_full, /*dim=*/-1, j, j + 1).contiguous();
        Tensor over  = tenzor::slice(O_full, /*dim=*/-1, j, j + 1).contiguous();
        return tenzor::add(tenzor::mul(V_col, mask), over);
    };

    // τ_j as (..., 1, 1) for broadcasting against (..., m, n) matrices.
    auto build_tau_j = [&](int64_t j) -> Tensor {
        Tensor t = tenzor::slice(tau_c, /*dim=*/-1, j, j + 1).contiguous();
        std::vector<int64_t> shp(t.shape().begin(), t.shape().end());
        shp.push_back(1);
        return t.reshape(shp);
    };

    // ------------------------------------------------------------------
    // Backward pass: B_j = (H_{j+1} … H_{k-1})[:, :n] for j = 0..k-1.
    // Recurrence B_{j-1} = H_j · B_j with B_{k-1} = I[:, :n].
    // We materialise all k tensors once (same memory cost as the original
    // host kernel) so the forward walk below can read them in O(1).
    // ------------------------------------------------------------------
    std::vector<Tensor> B_store(k);
    {
        Tensor I_mn = tenzor::eye(m, n, compute_dtype, orig_device);
        Tensor B_last = I_mn;
        if (ndim > 2) {
            std::vector<int64_t> bshape(V_shape_v.begin(), V_shape_v.end() - 2);
            bshape.push_back(m);
            bshape.push_back(n);
            B_last = tenzor::expand(I_mn, bshape).contiguous();
        }
        B_store[k - 1] = B_last;

        for (int64_t j = k - 2; j >= 0; --j) {
            const int64_t jp1 = j + 1;
            Tensor v_jp1   = build_v(jp1);
            Tensor v_jp1T  = tenzor::transpose(v_jp1, -2, -1);
            Tensor inner   = tenzor::matmul(v_jp1T, B_store[jp1]);
            Tensor outer   = tenzor::matmul(v_jp1, inner);
            Tensor tau_jp1 = build_tau_j(jp1);
            Tensor scaled  = tenzor::mul(outer, tau_jp1);
            B_store[j]     = tenzor::sub(B_store[jp1], scaled);
        }
    }

    // ------------------------------------------------------------------
    // Forward pass: walk j = 0..k-1 with P_j (left-propagated grad) and
    // accumulate ∂L/∂V column-by-column and ∂L/∂τ entry-by-entry.
    //   P_0 = G,  P_{j+1} = H_j · P_j = P_j − τ_j · v_j · (v_jᵀ P_j)
    //   ∂L/∂τ_j = −(v_jᵀ P_j)·(v_jᵀ B_j)ᵀ
    //   ∂L/∂v_j = −τ_j · (P_j (B_jᵀ v_j) + B_j (P_jᵀ v_j))
    // ------------------------------------------------------------------
    Tensor neg_one = tenzor::full({}, -1.0, compute_dtype, orig_device);

    std::vector<Tensor> grad_V_cols(k);
    std::vector<Tensor> grad_tau_entries(k);

    Tensor P = G_c;
    for (int64_t j = 0; j < k; ++j) {
        Tensor v_j   = build_v(j);
        Tensor v_jT  = tenzor::transpose(v_j, -2, -1);
        Tensor B_j   = B_store[j];
        Tensor tau_j = build_tau_j(j);

        // u = v_jᵀ · P  → (..., 1, n);   w = v_jᵀ · B_j → (..., 1, n)
        Tensor u_j = tenzor::matmul(v_jT, P);
        Tensor w_j = tenzor::matmul(v_jT, B_j);

        // ∂τ_j = −sum(u·w)  → (..., 1, 1)  → squeeze last to (..., 1)
        Tensor uw       = tenzor::mul(u_j, w_j);
        Tensor neg_dot  = tenzor::sum(uw, /*dim=*/-1, /*keepdim=*/true);
        neg_dot         = tenzor::mul(neg_dot, neg_one);
        std::vector<int64_t> tau_entry_shape(neg_dot.shape().begin(),
                                             neg_dot.shape().end() - 1);
        grad_tau_entries[j] = neg_dot.reshape(tau_entry_shape);  // (..., 1)

        // ∂v_j_full = −τ_j · (P · wᵀ + B_j · uᵀ)  → (..., m, 1)
        Tensor wT          = tenzor::transpose(w_j, -2, -1);
        Tensor uT          = tenzor::transpose(u_j, -2, -1);
        Tensor Pwt         = tenzor::matmul(P, wT);
        Tensor But         = tenzor::matmul(B_j, uT);
        Tensor sum_pw_bu   = tenzor::add(Pwt, But);
        Tensor neg_tau_j   = tenzor::mul(tau_j, neg_one);
        Tensor grad_v_full = tenzor::mul(sum_pw_bu, neg_tau_j);

        // Mask off entries i ≤ j (only strictly-below-diagonal lands in V).
        Tensor mask_j   = tenzor::slice(M_full, /*dim=*/-1, j, j + 1).contiguous();
        grad_V_cols[j]  = tenzor::mul(grad_v_full, mask_j);

        // P_{j+1} = P_j − τ_j · v_j · u_j
        Tensor v_u      = tenzor::matmul(v_j, u_j);
        Tensor scaled_v = tenzor::mul(v_u, tau_j);
        P               = tenzor::sub(P, scaled_v);
    }

    // Assemble (..., m, k) grad_V from k columns and (..., k) grad_tau from
    // k entries. cat along last dim.
    Tensor grad_V   = tenzor::cat(grad_V_cols, /*dim=*/-1);
    Tensor grad_tau = tenzor::cat(grad_tau_entries, /*dim=*/-1);

    if (grad_V.dtype()   != orig_dtype) grad_V   = grad_V.to(orig_dtype);
    if (grad_tau.dtype() != orig_dtype) grad_tau = grad_tau.to(orig_dtype);

    return {grad_V, grad_tau};
}

// TensorInvBackward:
// Forward: Y = tensorinv(X, ind)
// Backward: dL/dX = -tensorinv(X, ind) @ grad @ tensorinv(X, ind)
// (Same as matrix inverse gradient, with reshaping)
auto TensorInvBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TensorInvBackward::forward should not be called directly");
}

auto TensorInvBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& Y = saved_tensors_[0];  // tensorinv result
    const auto& grad = grad_outputs[0];

    // Forward: X.shape = [a₁..a_p, b₁..b_q]  with Π a_i == Π b_j  (ind_ = p)
    //   X_2d = reshape(X, [Π a_i, Π b_j])
    //   Y_2d = X_2d⁻¹  →  Y.shape = [b₁..b_q, a₁..a_p]
    //
    // So in Y the leading dims (count = ndim_X − ind_) correspond to the
    // "cols" side of the forward reshape, and the trailing dims (count
    // ind_) correspond to the "rows" side. The old code split Y at ind_,
    // yielding a non-square Y_2d and a transposed result shape — which
    // surfaced as the "expected [2,3,6] got [3,6,2]" error.
    auto Y_shape = std::vector<int64_t>(Y.shape().begin(), Y.shape().end());
    const int64_t ndim_Y = static_cast<int64_t>(Y_shape.size());
    const int64_t ind_Y  = ndim_Y - ind_;  // number of leading dims in Y

    int64_t rows = 1, cols = 1;
    for (int64_t i = 0; i < ind_Y; ++i)      rows *= Y_shape[i];     // = Π b_j
    for (int64_t i = ind_Y; i < ndim_Y; ++i) cols *= Y_shape[i];     // = Π a_i

    // For a well-posed tensorinv, rows == cols (square matrix).
    auto Y_2d = reshape(Y, {rows, cols});
    auto grad_2d = reshape(grad, {rows, cols});

    // ∂L/∂X_2d = − X_2d⁻ᵀ · grad_Y · X_2d⁻ᵀ. Using Y_2d = X_2d⁻¹ and
    // the transpose of Y = X⁻¹:  grad_X_2d = − Y_2dᵀ · grad_Y · Y_2dᵀ.
    auto Y_2d_t = tenzor::transpose(Y_2d, 0, 1);
    auto temp = matmul(matmul(Y_2d_t, grad_2d), Y_2d_t);
    auto result_2d = neg(temp);

    // Reshape back to the input tensor shape [a₁..a_p, b₁..b_q].
    std::vector<int64_t> input_shape;
    for (int64_t i = ind_Y; i < ndim_Y; ++i) input_shape.push_back(Y_shape[i]);  // a's
    for (int64_t i = 0; i < ind_Y; ++i)      input_shape.push_back(Y_shape[i]);  // b's

    return {reshape(result_2d, input_shape)};
}

auto TensorInvBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Forward Y = tensorinv(X, ind). Closed form
    //   Y_2d   = reshape(Y, {rows, cols})         (rows == cols)
    //   grad2d = reshape(grad_Y, {rows, cols})
    //   grad_X_2d = - Y_2d^T @ grad2d @ Y_2d^T
    //   grad_X    = reshape(grad_X_2d, input_shape)
    // Y is saved constant; every other step is Variable-level so grad_fn
    // flows through grad_outputs[0].
    require_saved_tensors(1);
    const Tensor& Y_t = saved_tensors_[0];
    Variable Y_var(Y_t, /*requires_grad=*/false);

    auto Y_shape = std::vector<int64_t>(Y_t.shape().begin(), Y_t.shape().end());
    const int64_t ndim_Y = static_cast<int64_t>(Y_shape.size());
    const int64_t ind_Y  = ndim_Y - ind_;

    int64_t rows = 1, cols = 1;
    for (int64_t i = 0; i < ind_Y; ++i)      rows *= Y_shape[i];
    for (int64_t i = ind_Y; i < ndim_Y; ++i) cols *= Y_shape[i];

    auto Y_2d = tenzor::reshape(Y_var, {rows, cols});
    auto grad_2d = tenzor::reshape(grad_outputs[0], {rows, cols});
    auto Y_2d_t = tenzor::transpose(Y_2d, 0, 1);
    auto temp = tenzor::matmul(tenzor::matmul(Y_2d_t, grad_2d), Y_2d_t);
    auto result_2d = tenzor::neg(temp);

    std::vector<int64_t> input_shape;
    for (int64_t i = ind_Y; i < ndim_Y; ++i) input_shape.push_back(Y_shape[i]);
    for (int64_t i = 0; i < ind_Y; ++i)      input_shape.push_back(Y_shape[i]);
    auto grad_X = tenzor::reshape(result_2d, input_shape);
    return {grad_X};
}

// TensorSolveBackward:
// Forward: X = tensorsolve(A, B)  solves A @ X = B (after reshaping)
// Backward:
//   grad_B = tensorsolve(A^T, grad)  (approximately)
//   grad_A = -tensorsolve(A, grad @ X^T)  (approximately)
// In practice, work in the flattened 2D space.
auto TensorSolveBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TensorSolveBackward::forward should not be called directly");
}

auto TensorSolveBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(3);
    const auto& A = saved_tensors_[0];
    const auto& B = saved_tensors_[1];
    const auto& X = saved_tensors_[2];  // solution
    const auto& grad = grad_outputs[0];

    // Work in the flattened space
    // grad_B = solve(A^T, grad) -> tensorsolve of transposed A
    // Since tensorsolve reshapes internally, use linalg::solve on reshaped tensors

    auto A_shape = std::vector<int64_t>(A.shape().begin(), A.shape().end());
    auto B_shape = std::vector<int64_t>(B.shape().begin(), B.shape().end());

    // Flatten A to 2D: first B.numel dimensions -> rows, remaining -> cols
    int64_t B_numel = 1;
    for (auto s : B_shape) B_numel *= s;
    int64_t A_numel = 1;
    for (auto s : A_shape) A_numel *= s;
    int64_t N = B_numel;  // square size (A is effectively N x N)

    auto A_2d = reshape(A, {N, A_numel / N});
    auto grad_flat = reshape(grad, {N, 1});
    auto X_flat = reshape(X, {N, 1});

    auto At_2d = transpose(A_2d, 0, 1);
    auto grad_B_flat = tenzor::linalg::solve(At_2d, grad_flat);
    auto grad_B = reshape(grad_B_flat, B_shape);

    // grad_A = -grad_B @ X^T reshaped
    auto grad_B_col = reshape(grad_B_flat, {N, 1});
    auto X_row = transpose(X_flat, 0, 1);
    auto grad_A_2d = neg(matmul(grad_B_col, X_row));
    auto grad_A = reshape(grad_A_2d, A_shape);

    return {grad_A, grad_B};
}

auto TensorSolveBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // R.5 — Variable-level rewrite. Forward X = tensorsolve(A, B); A, B
    // reshaped to ({N, N}, {N, 1}). Saved (A, B, X). First-order chain:
    //   grad_B = solve(A^T, grad_Y)              (N×1)
    //   grad_A = - grad_B @ X^T                  (N×N) → reshape to A.shape
    // A, X are non-diff Variables (constants from saved state); everything
    // else is Variable-level so grad_fn flows through grad_outputs[0].
    require_saved_tensors(3);
    const Tensor& A_t = saved_tensors_[0];
    const Tensor& B_t = saved_tensors_[1];
    const Tensor& X_t = saved_tensors_[2];

    auto A_shape = std::vector<int64_t>(A_t.shape().begin(), A_t.shape().end());
    auto B_shape = std::vector<int64_t>(B_t.shape().begin(), B_t.shape().end());
    int64_t B_numel = 1; for (auto s : B_shape) B_numel *= s;
    int64_t A_numel = 1; for (auto s : A_shape) A_numel *= s;
    int64_t N = B_numel;

    Tensor A_2d_t = tenzor::reshape(A_t, {N, A_numel / N});
    Tensor X_flat_t = tenzor::reshape(X_t, {N, 1});
    Variable A_2d_var(A_2d_t, /*requires_grad=*/false);
    Variable X_flat_var(X_flat_t, /*requires_grad=*/false);

    auto grad_flat = tenzor::reshape(grad_outputs[0], {N, 1});
    auto At_2d = tenzor::transpose(A_2d_var, 0, 1);
    auto grad_B_flat = tenzor::solve(At_2d, grad_flat);
    auto grad_B = tenzor::reshape(grad_B_flat, B_shape);

    auto X_row = tenzor::transpose(X_flat_var, 0, 1);
    auto grad_A_2d = tenzor::neg(tenzor::matmul(grad_B_flat, X_row));
    auto grad_A = tenzor::reshape(grad_A_2d, A_shape);
    return {grad_A, grad_B};
}

// LinalgVectorNormBackward:
// Forward: y = norm(x, p, dim)
// Backward (p=2): grad_x = grad * x / norm(x)
// General p: grad_x = grad * sign(x) * |x|^(p-1) / norm(x)^(p-1)
auto LinalgVectorNormBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LinalgVectorNormBackward::forward should not be called directly");
}

auto LinalgVectorNormBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& norm_val = saved_tensors_[1];  // norm result
    const auto& grad = grad_outputs[0];

    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Expand norm and grad to input shape
    auto norm_expanded = norm_val;
    auto grad_expanded = grad;
    if (!keepdim_) {
        // Need to unsqueeze along reduced dimensions
        for (auto d : dim_) {
            int64_t dd = d < 0 ? d + static_cast<int64_t>(input.ndim()) : d;
            norm_expanded = unsqueeze(norm_expanded, dd);
            grad_expanded = unsqueeze(grad_expanded, dd);
        }
    }
    norm_expanded = expand(norm_expanded, input_shape);
    grad_expanded = expand(grad_expanded, input_shape);

    if (std::abs(ord_ - 2.0) < 1e-10) {
        // L2 norm: grad * x / norm
        auto eps = full(input_shape, detail::dtype_epsilon(input.dtype()),
                        input.dtype(), input.device());
        auto safe_norm = where(eq(norm_expanded, zeros_like(norm_expanded)),
                               eps, norm_expanded);
        return {mul(grad_expanded, div(input, safe_norm))};
    }

    if (std::abs(ord_ - 1.0) < 1e-10) {
        // L1 norm: grad * sign(x)
        return {mul(grad_expanded, tenzor::sign(input))};
    }

    if (std::isinf(ord_)) {
        // Audit D4: ±inf vector norm. Both share the same math —
        //   ||x||_{+inf} = max(|x|),  ||x||_{-inf} = min(|x|)
        // — and in both cases the gradient is supported only on the index
        // where |x_i| equals the norm value, with magnitude sign(x_i).
        auto abs_x = abs(input);
        auto eps_t = full(input_shape, detail::dtype_epsilon(input.dtype()),
                          input.dtype(), input.device());
        auto mask = lt(abs(sub(abs_x, norm_expanded)), eps_t);
        auto sgn = tenzor::sign(input);
        return {mul(grad_expanded, where(mask, sgn, zeros_like(input)))};
    }

    if (ord_ == 0.0) {
        // Audit D4: L0 "norm" (count of nonzero entries) is piecewise
        // constant — gradient is 0 a.e. and undefined at x_i = 0. PyTorch
        // throws here too; mirror that contract rather than silently
        // returning zeros, which would mask a user bug.
        throw std::runtime_error(
            "LinalgVectorNormBackward: ord=0 (L0 norm) has no defined "
            "gradient — it is piecewise constant. Detach the input or use "
            "an order in [1, inf] for differentiable norms.");
    }

    // General p-norm: grad * sign(x) * |x|^(p-1) / norm^(p-1)
    auto abs_x = abs(input);
    auto eps = full(input_shape, detail::dtype_epsilon(input.dtype()),
                    input.dtype(), input.device());
    auto safe_abs = where(eq(abs_x, zeros_like(abs_x)), eps, abs_x);
    auto safe_norm = where(eq(norm_expanded, zeros_like(norm_expanded)), eps, norm_expanded);

    // V.7: ord_ is `double`; tenzor::pow(Tensor, double) exists (audit-3 R.1),
    // so drop the float narrowing that was rounding ord-1.0 through float.
    auto pow_x = tenzor::pow(safe_abs, ord_ - 1.0);
    auto pow_norm = tenzor::pow(safe_norm, ord_ - 1.0);
    auto deriv = mul(tenzor::sign(input), div(pow_x, pow_norm));
    return {mul(grad_expanded, deriv)};
}

auto LinalgVectorNormBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Audit D4: real Variable-level backward. The previous body called the
    // tensor backward then wrapped its result as `Variable(t, ...)` with no
    // `grad_fn` — silently severing the autograd graph.
    //
    // The closed-form vector-norm backward factorises as
    //   grad_input = grad_expanded * deriv(x, ||x||, ord)
    // where `deriv` depends only on the saved input and norm (constants in
    // this backward) and `grad_expanded` is the incoming gradient lifted to
    // the input's shape via unsqueeze + expand. By computing `deriv` at
    // tensor level (with the math implemented in `backward()` above) and
    // wrapping it as a non-grad Variable, the final `grad_expanded *
    // deriv_var` is a Variable-level multiplication that preserves
    // `grad_fn` through the incoming gradient — enabling `create_graph=true`.
    require_saved_tensors(2);
    const Tensor& input = saved_tensors_[0];

    // Compute the input-shape derivative tensor by running the existing
    // tensor-level backward against a ones gradient; that gives us
    // `1 * deriv = deriv` (since the tensor path's last step is
    // `mul(grad_expanded, deriv)`). The early ord==0 branch in `backward()`
    // throws, which we want to propagate untouched.
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto ones_grad = full(input_shape, 1.0, input.dtype(), input.device());

    // To extract just `deriv`, we feed `ones_grad` directly to `backward()`,
    // but `backward()` expects a grad shaped like the norm output. Build
    // a ones grad of that shape instead.
    auto reduced_shape = input_shape;
    if (!dim_.empty()) {
        if (keepdim_) {
            for (auto d : dim_) {
                int64_t dd = d < 0 ? d + static_cast<int64_t>(input.ndim()) : d;
                reduced_shape[dd] = 1;
            }
        } else {
            // Build a new shape with reduced dims removed (in descending order).
            auto sorted_dims = dim_;
            for (auto& d : sorted_dims) {
                d = d < 0 ? d + static_cast<int64_t>(input.ndim()) : d;
            }
            std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int64_t>());
            for (auto d : sorted_dims) reduced_shape.erase(reduced_shape.begin() + d);
        }
    } else {
        reduced_shape = keepdim_
            ? std::vector<int64_t>(input.ndim(), 1)
            : std::vector<int64_t>{};
    }
    Tensor ones_grad_for_norm = full(reduced_shape, 1.0, input.dtype(), input.device());
    Tensor deriv = backward({ones_grad_for_norm})[0];

    // Wrap deriv as a non-grad Variable: it depends only on saved (x, norm).
    Variable deriv_var(deriv, /*requires_grad=*/false);

    // Lift the incoming gradient to the input's shape using autograd-level
    // unsqueeze + expand so `grad_fn` flows through `grad_outputs[0]`.
    Variable grad_expanded = grad_outputs[0];
    if (!keepdim_) {
        for (auto d : dim_) {
            int64_t dd = d < 0 ? d + static_cast<int64_t>(input.ndim()) : d;
            grad_expanded = unsqueeze(grad_expanded, dd);
        }
    }
    grad_expanded = expand(grad_expanded, input_shape);

    // Final multiply preserves grad_fn through grad_outputs[0].
    return {grad_expanded * deriv_var};
}

// LinalgMatrixNormBackward:
// audit-2026-05-03 — ord=2 is the SPECTRAL norm (largest singular value),
// not Frobenius. Backward: ∂σ_max/∂A = u_1 v_1^T where u_1, v_1 are the
// leading left/right singular vectors. The previous implementation
// returned the Frobenius gradient `A/norm`, which is wrong for ord=2.
auto LinalgMatrixNormBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LinalgMatrixNormBackward::forward should not be called directly");
}

auto LinalgMatrixNormBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& grad = grad_outputs[0];

    if (std::abs(ord_ - 2.0) < 1e-10 || std::abs(ord_ + 2.0) < 1e-10) {
        // Spectral norm (ord=±2): gradient is u_k v_k^T where k is the
        // index of the corresponding singular value.
        //   ord = +2: σ_max (first SV) → u_1, v_1 = U[:, 0], Vh[0, :]
        //   ord = -2: σ_min (last SV)  → u_K, v_K = U[:, -1], Vh[-1, :]
        // Sign-invariant since u_k and v_k flip together.
        // Audit D5: ord = -2 was missing; previously fell through to the
        // col-sum mask code below, which is the wrong math.
        auto [U, S, Vh] = tenzor::linalg::svd(input, /*full_matrices=*/false);
        const int64_t U_ndim  = U.ndim();
        const int64_t Vh_ndim = Vh.ndim();
        const int64_t K = U.size(U_ndim - 1);    // number of singular values

        const bool use_smallest = (ord_ < 0.0);
        const int64_t sv_idx = use_smallest ? K - 1 : 0;

        // Slice U[..., :, sv_idx:sv_idx+1] → (..., M, 1)
        auto uk = tenzor::slice(U, /*dim=*/U_ndim - 1, /*start=*/sv_idx,
                                /*end=*/sv_idx + 1);
        // Slice Vh[..., sv_idx:sv_idx+1, :] → (..., 1, N)
        auto vkh = tenzor::slice(Vh, /*dim=*/Vh_ndim - 2, /*start=*/sv_idx,
                                 /*end=*/sv_idx + 1);
        auto outer = matmul(uk, vkh);  // (..., M, N)
        auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
        // Append trailing 1s for the matrix dims.
        while (grad_shape.size() < static_cast<size_t>(outer.ndim())) {
            grad_shape.push_back(1);
        }
        auto grad_reshaped = reshape(grad, grad_shape);
        return {mul(grad_reshaped, outer)};
    }

    // Induced p-norms for ord ∈ {1, -1, ±inf} are piecewise-linear but
    // differentiable almost everywhere, with a well-defined subgradient.
    // Phase P0 / Fix 2 of the audit cleanup: the previous version returned
    // `zeros_like(input)` claiming PyTorch compatibility, but the sibling
    // `LinalgVectorNormBackward` (above) handles the same `ord` values
    // with proper subgradients — and PyTorch's matrix-norm backward also
    // does, modulo the implementation details below.
    //
    // Math:
    //   ord = 1  (max column sum):  pick column j* with max Σ_i |A[i,j]|;
    //                               subgradient = sign(A[:, j*]) in that col,
    //                               zero elsewhere.
    //   ord = -1 (min column sum):  same with argmin.
    //   ord = +inf (max row sum):   pick row i* with max Σ_j |A[i,j]|;
    //                               subgradient = sign(A[i*, :]) in that row.
    //   ord = -inf (min row sum):   same with argmin.
    //
    // Output of forward is one scalar per batch (shape input.shape[:-2]).
    // grad has the same shape; we broadcast it back to the matrix shape by
    // appending two trailing-1 dimensions.
    const int64_t ndim    = input.ndim();
    const int64_t M       = input.size(ndim - 2);
    const int64_t N       = input.size(ndim - 1);
    const bool col_norm   = (std::abs(ord_) == 1.0);    // ord = ±1 -> column sums
    const int64_t reduce_dim = col_norm ? ndim - 2 : ndim - 1;
    const int64_t num_classes = col_norm ? N : M;
    const bool use_min    = (ord_ < 0.0);

    auto abs_A = tenzor::abs(input);
    auto sums  = tenzor::sum(abs_A, /*dim=*/reduce_dim, /*keepdim=*/false);
    // `sums` has shape input.shape with `reduce_dim` removed. Its last dim
    // is the select-axis (N for column-sum, M for row-sum); argmax/argmin
    // along that dim collapses to a scalar per batch.
    auto idx = use_min
        ? tenzor::argmin(sums, /*dim=*/-1, /*keepdim=*/false)
        : tenzor::argmax(sums, /*dim=*/-1, /*keepdim=*/false);
    // one_hot returns Int64; broadcast against the input matrix by
    // unsqueezing the *other* matrix dim:
    //   col_norm: mask shape (..., N), unsqueeze to (..., 1, N).
    //   row_norm: mask shape (..., M), unsqueeze to (..., M, 1).
    auto mask = tenzor::one_hot(idx, num_classes).to(input.dtype());
    // Reshape the one-hot mask to its explicit broadcast target (the reduced
    // matrix dim becomes 1). Rank-robust: argmax/one_hot keepdim conventions
    // differ across backends (CUDA argmax of a 1-D tensor yields (1,) vs CPU's
    // scalar), which previously left a stray leading dim -> [1,M,N] gradient.
    {
        std::vector<int64_t> mask_shape(input.shape().begin(), input.shape().end());
        mask_shape[col_norm ? (ndim - 2) : (ndim - 1)] = 1;
        mask = reshape(mask, mask_shape);
    }

    // Expand `grad` to broadcast against (..., M, N): append two trailing 1s.
    auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
    while (grad_shape.size() < static_cast<size_t>(ndim)) grad_shape.push_back(1);
    auto grad_reshaped = reshape(grad, grad_shape);

    return { mul(mul(grad_reshaped, mask), tenzor::sign(input)) };
}

auto LinalgMatrixNormBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Audit D5: real Variable-level backward. Same factoring as D4:
    //   grad_input = grad_reshaped * deriv(A, ord)
    // where `deriv` (= `u_k v_k^T` for ord=±2 or `mask * sign(A)` for
    // ord ∈ {±1, ±inf}) depends only on the saved input. Compute it at
    // tensor level by feeding a ones grad through `backward()`, then
    // compose the final multiply at Variable level so `grad_fn` flows
    // through `grad_outputs[0]`.
    require_saved_tensors(2);
    const Tensor& input = saved_tensors_[0];
    const int64_t ndim = input.ndim();

    // Norm output shape = input.shape[:-2] (reduce both matrix dims).
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    std::vector<int64_t> norm_shape(input_shape.begin(),
                                     input_shape.begin() + (ndim - 2));
    Tensor ones_grad = full(norm_shape, 1.0, input.dtype(), input.device());
    Tensor deriv = backward({ones_grad})[0];   // shape (..., M, N)
    Variable deriv_var(deriv, /*requires_grad=*/false);

    // Lift grad_outputs[0] (shape input.shape[:-2]) to (..., M, N) by
    // reshaping to append two trailing 1 dims, then expand. Both steps
    // are autograd-level so `grad_fn` flows through.
    auto grad_shape_v2 = norm_shape;
    grad_shape_v2.push_back(1);
    grad_shape_v2.push_back(1);
    Variable grad_reshaped = reshape(grad_outputs[0], grad_shape_v2);
    Variable grad_expanded = expand(grad_reshaped, input_shape);

    return {grad_expanded * deriv_var};
}

// LinalgVecdotBackward:
// Forward: y = vecdot(a, b, dim) = sum(a * b, dim)
// Backward:
//   grad_a = grad.unsqueeze(dim) * b
//   grad_b = grad.unsqueeze(dim) * a
auto LinalgVecdotBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LinalgVecdotBackward::forward should not be called directly");
}

auto LinalgVecdotBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    int64_t actual_dim = dim_;
    if (actual_dim < 0) actual_dim += a.ndim();

    auto grad_expanded = unsqueeze(grad, actual_dim);
    auto a_shape = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    grad_expanded = expand(grad_expanded, a_shape);

    // PyTorch convention: forward = sum(conj(a) * b, dim).
    // Wirtinger gradients (real inputs: conj is identity, math unchanged):
    //   grad_a = conj(grad) * b
    //   grad_b = grad * conj(a)
    if (a.is_complex()) {
        auto grad_a = mul(tenzor::conj(grad_expanded), b);
        auto grad_b = mul(grad_expanded, tenzor::conj(a));
        return {grad_a, grad_b};
    }
    auto grad_a = mul(grad_expanded, b);
    auto grad_b = mul(grad_expanded, a);
    return {grad_a, grad_b};
}

auto LinalgVecdotBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(2);
    Variable a_var(saved_tensors_[0], false);
    Variable b_var(saved_tensors_[1], false);

    int64_t actual_dim = dim_;
    if (actual_dim < 0) actual_dim += saved_tensors_[0].ndim();

    auto grad_expanded = tenzor::unsqueeze(grad_outputs[0], actual_dim);
    auto a_shape = std::vector<int64_t>(saved_tensors_[0].shape().begin(),
                                         saved_tensors_[0].shape().end());
    grad_expanded = tenzor::expand(grad_expanded, a_shape);

    // PyTorch convention: forward = sum(conj(a) * b, dim).
    // Wirtinger gradients (real inputs: conj is identity, math unchanged):
    //   grad_a = conj(grad) * b
    //   grad_b = grad * conj(a)
    if (saved_tensors_[0].is_complex()) {
        Variable grad_a = tenzor::conj(grad_expanded) * b_var;
        Variable grad_b = grad_expanded * tenzor::conj(a_var);
        return {grad_a, grad_b};
    }
    return {grad_expanded * b_var, grad_expanded * a_var};
}

// AsStridedBackward:
// as_strided creates a view with custom strides. The backward requires
// scatter_add from the output grad back to the input grad.
auto AsStridedBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("AsStridedBackward::forward should not be called directly");
}

auto AsStridedBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // Create zero tensor with original input shape (flattened)
    int64_t input_numel = 1;
    for (auto s : input_shape_) input_numel *= s;
    auto grad_input = zeros({input_numel}, grad.dtype(), grad.device());

    int64_t output_numel = 1;
    for (auto s : size_) output_numel *= s;

    int64_t offset = storage_offset_.value_or(0);

    // Compute linear indices for each output element
    // Build on CPU, then transfer to device
    std::vector<int64_t> indices(output_numel);
    std::vector<int64_t> coord(size_.size(), 0);

    for (int64_t i = 0; i < output_numel; ++i) {
        int64_t linear_idx = offset;
        for (size_t d = 0; d < size_.size(); ++d) {
            linear_idx += coord[d] * stride_[d];
        }
        indices[i] = linear_idx;

        // Increment coordinates
        for (int d = static_cast<int>(size_.size()) - 1; d >= 0; --d) {
            coord[d]++;
            if (coord[d] < size_[d]) break;
            coord[d] = 0;
        }
    }

    // Create index tensor from computed indices
    auto index_tensor = Tensor::from_blob(
        indices.data(), {output_numel}, DType::Int64, Device::cpu());
    // Clone to own the memory (indices vector will go out of scope)
    index_tensor = index_tensor.clone();
    // Transfer to device if needed
    if (grad.device().type != Device::Type::CPU) {
        index_tensor = index_tensor.to(grad.device());
    }

    auto flat_grad = reshape(grad, {-1});
    auto result = scatter_add(grad_input, 0, index_tensor, flat_grad);
    return {reshape(result, input_shape_)};
}

// Audit B.3 closed-form higher-order: AsStrided is a linear gather
// (strided view) with constant Jacobian, so its backward is a linear
// scatter-add (above). The backward of *that* is the linear gather
// itself: pulling the second-order grad (a tensor of input_shape_)
// through the same stride/size/offset pattern recovers the second-
// order grad as it would be seen at the AsStrided output. Calling
// `as_strided` on the Variable preserves the differentiation graph
// (the as_strided wrapper installs its own grad_fn so a third-order
// backward, if requested, would scatter-add right back through this
// same backward()).
auto AsStridedBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    return {::tenzor::as_strided(grad_outputs[0],
                                  std::span<const int64_t>(size_),
                                  std::span<const int64_t>(stride_),
                                  storage_offset_)};
}

// ============================================================================
// Phase 12 (audit-2026-05-03) — Bessel J0/J1/Y0/Y1 and Zeta autograd.
// ============================================================================

// d/dx J_0(x) = -J_1(x).
auto BesselJ0Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("BesselJ0Backward::forward should not be called directly");
}
auto BesselJ0Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    auto deriv = neg(tenzor::bessel_j1(input));
    return {mul(grad_outputs[0], deriv)};
}
auto BesselJ0Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    Variable deriv_var(neg(tenzor::bessel_j1(input)), false);
    return {grad_outputs[0] * deriv_var};
}

// d/dx J_1(x) = J_0(x) - J_1(x)/x for x != 0; the limit at 0 is 0.5.
auto BesselJ1Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("BesselJ1Backward::forward should not be called directly");
}
auto BesselJ1Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto j0v = tenzor::bessel_j0(input);
    auto j1v = tenzor::bessel_j1(input);
    auto eps = full(input_shape, detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto zero_t = zeros(input_shape, input.dtype(), input.device());
    auto x_is_zero = eq(input, zero_t);
    auto safe_x = where(x_is_zero, eps, input);
    auto deriv_nonzero = sub(j0v, div(j1v, safe_x));
    auto half = full(input_shape, 0.5, input.dtype(), input.device());
    auto deriv = where(x_is_zero, half, deriv_nonzero);
    return {mul(grad_outputs[0], deriv)};
}
auto BesselJ1Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto j0v = tenzor::bessel_j0(input);
    auto j1v = tenzor::bessel_j1(input);
    auto eps = full(input_shape, detail::dtype_epsilon(input.dtype()), input.dtype(), input.device());
    auto zero_t = zeros(input_shape, input.dtype(), input.device());
    auto x_is_zero = eq(input, zero_t);
    auto safe_x = where(x_is_zero, eps, input);
    auto deriv_nonzero = sub(j0v, div(j1v, safe_x));
    auto half = full(input_shape, 0.5, input.dtype(), input.device());
    auto deriv = where(x_is_zero, half, deriv_nonzero);
    Variable deriv_var(deriv, false);
    return {grad_outputs[0] * deriv_var};
}

// d/dx Y_0(x) = -Y_1(x), defined for x > 0.
auto BesselY0Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("BesselY0Backward::forward should not be called directly");
}
auto BesselY0Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    auto deriv = neg(tenzor::bessel_y1(input));
    return {mul(grad_outputs[0], deriv)};
}
auto BesselY0Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    Variable deriv_var(neg(tenzor::bessel_y1(input)), false);
    return {grad_outputs[0] * deriv_var};
}

// d/dx Y_1(x) = Y_0(x) - Y_1(x)/x, defined for x > 0.
auto BesselY1Backward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("BesselY1Backward::forward should not be called directly");
}
auto BesselY1Backward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    auto y0v = tenzor::bessel_y0(input);
    auto y1v = tenzor::bessel_y1(input);
    auto deriv = sub(y0v, div(y1v, input));
    return {mul(grad_outputs[0], deriv)};
}
auto BesselY1Backward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    auto y0v = tenzor::bessel_y0(input);
    auto y1v = tenzor::bessel_y1(input);
    Variable deriv_var(sub(y0v, div(y1v, input)), false);
    return {grad_outputs[0] * deriv_var};
}

// Zeta(s, q): d/dq zeta(s, q) = -s * zeta(s+1, q).
// Only the q-gradient is implemented; s receives a zero gradient.
auto ZetaBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ZetaBackward::forward should not be called directly");
}
auto ZetaBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& s_t = saved_tensors_[0];
    const auto& q_t = saved_tensors_[1];
    auto s_shape = std::vector<int64_t>(s_t.shape().begin(), s_t.shape().end());
    auto one_t = ones(s_shape, s_t.dtype(), s_t.device());
    auto s_plus_one = add(s_t, one_t);
    auto zeta_next = tenzor::zeta(s_plus_one, q_t);
    auto deriv_q = neg(mul(s_t, zeta_next));
    auto grad_s = zeros(s_shape, s_t.dtype(), s_t.device());
    auto grad_q = mul(grad_outputs[0], deriv_q);
    std::vector<Tensor> out;
    out.push_back(grad_s);
    out.push_back(grad_q);
    return out;
}

// BetaInc(a, b, x) = I_x(a, b). Differentiable wrt x:
// dI_x(a,b)/dx = x^(a-1) (1-x)^(b-1) / B(a, b)
//             = exp((a-1)*log(x) + (b-1)*log(1-x) - lbeta(a,b))
//             = exp((a-1)*log(x) + (b-1)*log(1-x) + lgamma(a+b) - lgamma(a) - lgamma(b))
auto BetaIncBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("BetaIncBackward::forward should not be called directly");
}
auto BetaIncBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(3);
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& x = saved_tensors_[2];

    auto a_shape = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    auto x_shape = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    auto one_a = ones(a_shape, a.dtype(), a.device());
    auto one_x = ones(x_shape, x.dtype(), x.device());

    auto am1 = sub(a, one_a);
    auto bm1 = sub(b, one_a);
    auto log_x = tenzor::log(x);
    auto log_1mx = tenzor::log(sub(one_x, x));
    // log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)
    auto lg_a = tenzor::lgamma(a);
    auto lg_b = tenzor::lgamma(b);
    auto lg_ab = tenzor::lgamma(add(a, b));
    auto log_inv_beta = sub(lg_ab, add(lg_a, lg_b));

    auto log_deriv = add(add(mul(am1, log_x), mul(bm1, log_1mx)), log_inv_beta);
    auto deriv = tenzor::exp(log_deriv);

    auto grad_a = zeros(a_shape, a.dtype(), a.device());
    auto grad_b = zeros(a_shape, b.dtype(), b.device());
    auto grad_x = mul(grad_outputs[0], deriv);

    std::vector<Tensor> out;
    out.push_back(grad_a);
    out.push_back(grad_b);
    out.push_back(grad_x);
    return out;
}

// ============================================================================
// Audit E.7 — Function wrappers for intrinsically non-differentiable ops.
//
// Each forward() calls the underlying Tensor op (so the wrapper Variable
// carries the realised output), but backward() throws a typed
// `tenzor::NonDifferentiable` exception so callers who route gradient flow
// through a histogram / bincount / searchsorted result get a clear,
// actionable message rather than the previous mystery "Function has no
// backward".
//
// Surrogate-gradient users (Gumbel-softmax / STE / etc.) should still
// build their own custom Function with the chosen relaxation; these
// wrappers explicitly *opt out* of providing a default surrogate.
// ============================================================================

auto HistcBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("HistcBackward::forward should not be called directly");
}
auto HistcBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "histc: histogram counts are intrinsically non-differentiable in the "
        "input tensor (the gradient is a delta-of-Diracs distribution). Wrap "
        "the call in a custom autograd::Function with a surrogate gradient "
        "(e.g. straight-through estimator) if you need gradients through it.");
}

auto BincountBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("BincountBackward::forward should not be called directly");
}
auto BincountBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "bincount: integer count tensor is non-differentiable in the input "
        "indices. The optional `weights` argument *is* differentiable (the "
        "Jacobian is a scatter-by-index); wrap with a custom Function that "
        "exposes only that branch if you need gradients.");
}

auto SearchSortedBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SearchSortedBackward::forward should not be called directly");
}
auto SearchSortedBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "searchsorted: returned indices are integer positions and are not "
        "differentiable in either the sorted_sequence or the values tensor. "
        "Use a soft-argmin/argmax relaxation (e.g. softmax over distances) "
        "if you need a differentiable approximation.");
}

auto MultinomialSampleBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("MultinomialSampleBackward::forward should not be called directly");
}
auto MultinomialSampleBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "multinomial sampling: sampled indices are integer draws and are not "
        "differentiable in the probability tensor (the gradient is the "
        "delta-of-Diracs distribution). Use the Gumbel-softmax relaxation "
        "(tenzor.nn.functional.gumbel_softmax) or a straight-through "
        "estimator if you need gradients through the categorical sample.");
}

auto BernoulliSampleBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("BernoulliSampleBackward::forward should not be called directly");
}
auto BernoulliSampleBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "bernoulli sampling: drawn {0, 1} value is non-differentiable in the "
        "probability tensor (the output doesn't change continuously with "
        "`probs`). Use a relaxed Bernoulli (Concrete / Gumbel-sigmoid) or "
        "a straight-through estimator for differentiable variants.");
}

auto ArgmaxBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("ArgmaxBackward::forward should not be called directly");
}
auto ArgmaxBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "argmax: returned integer index is non-differentiable in the input "
        "tensor (gradient is delta-of-Diracs at the maximiser). Use "
        "soft-argmax (softmax-weighted index) for a differentiable "
        "approximation, or attach a straight-through estimator.");
}

auto ArgminBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("ArgminBackward::forward should not be called directly");
}
auto ArgminBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "argmin: returned integer index is non-differentiable in the input "
        "tensor. Use soft-argmin (softmax-weighted index over the negated "
        "input) for a differentiable approximation.");
}

auto BucketizeBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("BucketizeBackward::forward should not be called directly");
}
auto BucketizeBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "bucketize: bucket-index output is non-differentiable in either the "
        "input values or the sorted boundaries (gradient w.r.t. a step "
        "function is the delta-of-Diracs distribution at each boundary). "
        "Use a smooth bucketing surrogate (e.g. sigmoid-of-distance) if you "
        "need gradients.");
}

auto ArgSortBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("ArgSortBackward::forward should not be called directly");
}
auto ArgSortBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "argsort: returned permutation indices are integers and are not "
        "differentiable in the input values. Use `tenzor::sort` if you need "
        "the differentiable values branch alongside the indices.");
}

// NOTE: Mode is treated as differentiable (gradient = scatter-to-first-
// occurrence-of-mode-value); the ModeBackward implementation lives in the
// existing reduction-backward translation unit. We do not re-define it
// here as a NonDifferentiable stub — that was an early-iteration mistake.

// ============================================================================
// Audit E.7 continuation: real-backward and non-diff wrappers for additional
// OpIds that previously dispatched through the kernel registry without an
// autograd Function. See function.hpp for one-line summaries.
// ============================================================================

// square(x) = x * x  -->  d/dx = 2*x
auto SquareBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SquareBackward::forward should not be called directly");
}
auto SquareBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& x = saved_tensors_[0];
    // 2 * x * grad
    auto two_x = mul(x, 2.0);
    return {mul(grad, two_x)};
}

// rsqrt(x) = 1/sqrt(x)  -->  d/dx = -0.5 * x^(-3/2) = -0.5 * y^3
// We save the output y = rsqrt(x) and compute -0.5 * y^3 * grad.
auto RsqrtBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("RsqrtBackward::forward should not be called directly");
}
auto RsqrtBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& y = saved_tensors_[0];  // y = rsqrt(x)
    // -0.5 * y^3 * grad
    auto y2 = mul(y, y);
    auto y3 = mul(y2, y);
    auto scaled = mul(y3, -0.5);
    return {mul(grad, scaled)};
}

// deg2rad(x) = x * (pi/180)  -->  d/dx = pi/180 (constant)
auto Deg2RadBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("Deg2RadBackward::forward should not be called directly");
}
auto Deg2RadBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    constexpr double kPiOver180 = 0.017453292519943295;  // M_PI / 180.0
    return {mul(grad_outputs[0], kPiOver180)};
}

// rad2deg(x) = x * (180/pi)  -->  d/dx = 180/pi (constant)
auto Rad2DegBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("Rad2DegBackward::forward should not be called directly");
}
auto Rad2DegBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    constexpr double k180OverPi = 57.29577951308232;  // 180.0 / M_PI
    return {mul(grad_outputs[0], k180OverPi)};
}

// logit(x) = log(x / (1 - x))  -->  d/dx = 1 / (x * (1 - x))
// Saves the input x. Outside (0, 1) the derivative is undefined; we let
// the natural arithmetic propagate the inf/NaN so the caller sees the
// invalid region rather than fabricating a silent zero.
auto LogitBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("LogitBackward::forward should not be called directly");
}
auto LogitBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& x = saved_tensors_[0];
    // (1 - x) = -(x - 1)
    auto neg_one_minus_x = sub(x, 1.0);
    auto one_minus_x = neg(neg_one_minus_x);
    // x * (1 - x)
    auto denom = mul(x, one_minus_x);
    // grad / (x * (1 - x))
    return {div(grad, denom)};
}

// nan_to_num: y = x where isfinite(x), else replacement constant.
// Jacobian is identity on the finite-mask, zero on NaN/Inf.
// backward: grad * isfinite(x).cast(grad.dtype())
auto NanToNumBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("NanToNumBackward::forward should not be called directly");
}
auto NanToNumBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& x = saved_tensors_[0];
    auto finite_mask = isfinite(x);
    // Promote bool mask to grad's dtype for elementwise multiply.
    auto mask_f = finite_mask.to(grad.dtype());
    return {mul(grad, mask_f)};
}

// --- non-differentiable wrappers ---------------------------------------

auto HeavisideBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("HeavisideBackward::forward should not be called directly");
}
auto HeavisideBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "heaviside: the step function is piecewise constant; its derivative is "
        "the Dirac delta at the jump and zero elsewhere, so it is not "
        "differentiable in the classical sense. Use a smooth surrogate "
        "(e.g. sigmoid(x / temperature)) if you need gradients.");
}

auto SignbitBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SignbitBackward::forward should not be called directly");
}
auto SignbitBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "signbit: output is a Bool tensor (discrete), so it is not "
        "differentiable in the input. Use a soft sign surrogate "
        "(e.g. tanh(k * x) for large k) for a differentiable approximation.");
}

auto FrexpBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("FrexpBackward::forward should not be called directly");
}
auto FrexpBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "frexp: the exponent branch is integer-valued and the mantissa branch "
        "is piecewise constant in dyadic intervals (derivative is a sum of "
        "Diracs at the boundaries). Use direct log2-based decomposition if "
        "you need a differentiable factoring.");
}

auto HistogramBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("HistogramBackward::forward should not be called directly");
}
auto HistogramBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "histogram: integer count tensor is non-differentiable in the input "
        "values (same as histc). Wrap with a custom Function and a "
        "straight-through estimator if you need gradients through the "
        "binning step.");
}

// ============================================================================
// Audit E.7 continuation (batch 2): forward/backward impls for the second
// set of 10 OpIds. See function.hpp for one-line summaries.
// ============================================================================

// sign(x): zero gradient almost everywhere. Return a zero tensor of the
// same shape/dtype as the input so the graph traversal still has a
// well-typed buffer (PyTorch / JAX behaviour). This keeps callers from
// having to wrap sign() in a stop_gradient guard.
auto SignBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("SignBackward::forward should not be called directly");
}
auto SignBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& x = saved_tensors_[0];
    // grad w.r.t. x is identically zero — emit a zero buffer matching the
    // input's shape and dtype so the gradient typing stays consistent.
    (void)grad_outputs;
    return {zeros_like(x)};
}

// hypot(x, y) = sqrt(x*x + y*y).
// grad_x = (x / hypot(x, y)) * grad
// grad_y = (y / hypot(x, y)) * grad
// At (0, 0) the derivative is undefined; we let division produce NaN so
// the caller sees the degenerate point rather than a fabricated zero.
auto HypotBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("HypotBackward::forward should not be called directly");
}
auto HypotBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& x = saved_tensors_[0];
    const auto& y = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    auto h = tenzor::hypot(x, y);
    auto grad_x_unreduced = mul(grad, div(x, h));
    auto grad_y_unreduced = mul(grad, div(y, h));

    auto grad_x = reduce_grad_for_broadcasting(grad_x_unreduced, input_shape_x_);
    auto grad_y = reduce_grad_for_broadcasting(grad_y_unreduced, input_shape_y_);
    return {grad_x, grad_y};
}

// copysign(magnitude, sign_src) returns |magnitude| * sign(sign_src).
// d/d(magnitude) = sign(sign_src) (treating |.| as having sign-of-mag local
//                  Jacobian; the product simplifies to sign(sign_src))
// d/d(sign_src)  = 0 almost everywhere.
auto CopysignBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("CopysignBackward::forward should not be called directly");
}
auto CopysignBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& sign_src = saved_tensors_[0];
    const auto& grad = grad_outputs[0];

    // grad_mag = grad * sign(sign_src)
    auto grad_mag_unreduced = mul(grad, tenzor::sign(sign_src));
    auto grad_mag = reduce_grad_for_broadcasting(grad_mag_unreduced, input_shape_mag_);

    // grad_sign_src = 0 a.e.; emit a zero buffer of the sign source's
    // (broadcast-reduced) shape so the gradient typing stays consistent.
    auto grad_sign_zero_full =
        zeros_like(grad);  // same shape/dtype as the unreduced gradient
    auto grad_sign = reduce_grad_for_broadcasting(grad_sign_zero_full, input_shape_sign_);
    return {grad_mag, grad_sign};
}

// xlog1py(x, y) = x * log1p(y), with 0 * log1p(y) = 0 regardless of y.
// grad_x = grad * log1p(y), zeroed where x == 0
// grad_y = grad * (x / (1 + y)), zeroed where x == 0
auto Xlog1pyBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("Xlog1pyBackward::forward should not be called directly");
}
auto Xlog1pyBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& x = saved_tensors_[0];
    const auto& y = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    auto x_shape = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    auto zero_x = zeros(x_shape, x.dtype(), x.device());
    auto x_is_zero = eq(x, zero_x);

    // grad_x = grad * log1p(y), zeroed where x == 0
    auto log1p_y = tenzor::log1p(y);
    auto grad_x_raw = mul(grad, log1p_y);
    auto grad_x_unreduced = where(x_is_zero, zeros_like(grad_x_raw), grad_x_raw);

    // grad_y = grad * x / (1 + y), zeroed where x == 0. Guard against
    // (1 + y) == 0 by substituting epsilon to keep the masked-out result
    // finite; the where() then zeroes those entries when x == 0.
    auto y_shape = std::vector<int64_t>(y.shape().begin(), y.shape().end());
    auto one_plus_y = add(y, 1.0);
    auto eps = full(y_shape, detail::dtype_epsilon(y.dtype()), y.dtype(), y.device());
    auto safe_denom = where(
        eq(one_plus_y, zeros(y_shape, y.dtype(), y.device())),
        eps,
        one_plus_y);
    auto grad_y_raw = mul(grad, div(x, safe_denom));
    auto grad_y_unreduced = where(x_is_zero, zeros_like(grad_y_raw), grad_y_raw);

    auto grad_x = reduce_grad_for_broadcasting(grad_x_unreduced, input_shape_x_);
    auto grad_y = reduce_grad_for_broadcasting(grad_y_unreduced, input_shape_y_);
    return {grad_x, grad_y};
}

// addcmul(a, b, c, value) = a + value * b * c.
// d/da = 1                ⇒ grad
// d/db = value * c        ⇒ grad * value * c
// d/dc = value * b        ⇒ grad * value * b
auto AddcmulBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("AddcmulBackward::forward should not be called directly");
}
auto AddcmulBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& b = saved_tensors_[0];
    const auto& c = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    auto grad_a = reduce_grad_for_broadcasting(grad, input_shape_a_);
    auto grad_b_unreduced = mul(mul(grad, c), value_);
    auto grad_c_unreduced = mul(mul(grad, b), value_);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);
    auto grad_c = reduce_grad_for_broadcasting(grad_c_unreduced, input_shape_c_);
    return {grad_a, grad_b, grad_c};
}

// addcdiv(a, b, c, value) = a + value * b / c.
// d/da = 1                            ⇒ grad
// d/db = value / c                    ⇒ grad * value / c
// d/dc = -(value * b) / (c * c)       ⇒ -grad * value * b / (c * c)
auto AddcdivBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("AddcdivBackward::forward should not be called directly");
}
auto AddcdivBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& b = saved_tensors_[0];
    const auto& c = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    auto grad_a = reduce_grad_for_broadcasting(grad, input_shape_a_);

    // grad_b = (value / c) * grad
    auto inv_c = tenzor::reciprocal(c);
    auto scale_b = mul(inv_c, value_);
    auto grad_b_unreduced = mul(grad, scale_b);

    // grad_c = -(value * b / (c * c)) * grad = -value * b * (1/c^2) * grad
    auto c_sq = mul(c, c);
    auto inv_c_sq = tenzor::reciprocal(c_sq);
    auto scale_c = mul(mul(b, inv_c_sq), -value_);
    auto grad_c_unreduced = mul(grad, scale_c);

    auto grad_b = reduce_grad_for_broadcasting(grad_b_unreduced, input_shape_b_);
    auto grad_c = reduce_grad_for_broadcasting(grad_c_unreduced, input_shape_c_);
    return {grad_a, grad_b, grad_c};
}

// --- non-differentiable wrappers ---------------------------------------

auto FloorBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("FloorBackward::forward should not be called directly");
}
auto FloorBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "floor: piecewise-constant — gradient is zero almost everywhere and "
        "a Dirac comb at integer jumps. Use a custom STE Function "
        "(straight-through estimator) if a surrogate gradient is needed.");
}

auto CeilBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("CeilBackward::forward should not be called directly");
}
auto CeilBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "ceil: piecewise-constant — gradient is zero almost everywhere and "
        "a Dirac comb at integer jumps. Use a custom STE Function "
        "(straight-through estimator) if a surrogate gradient is needed.");
}

auto IsNanBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("IsNanBackward::forward should not be called directly");
}
auto IsNanBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "isnan: output is a Bool tensor (discrete), so it is not "
        "differentiable in the input. The NaN-pattern of a tensor is "
        "metadata, not a smooth function of the input values.");
}

auto LogicalAndBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("LogicalAndBackward::forward should not be called directly");
}
auto LogicalAndBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "logical_and: Bool inputs and Bool output, both discrete; the "
        "operation is not differentiable. Use a smooth t-norm (e.g. "
        "min(a, b) on [0, 1]) if you need a differentiable surrogate.");
}

// ============================================================================
// Audit E.7 continuation (batch 3): forward/backward impls for the third set
// of 10 OpIds. See function.hpp for one-line summaries.
// ============================================================================

// addmm(input, mat1, mat2, beta, alpha) = beta*input + alpha*(mat1 @ mat2).
// d/d(input) = beta * grad   (broadcast-reduced to input's shape)
// d/d(mat1)  = alpha * grad @ mat2^T
// d/d(mat2)  = alpha * mat1^T @ grad
auto AddmmBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("AddmmBackward::forward should not be called directly");
}
auto AddmmBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& mat1 = saved_tensors_[0];
    const auto& mat2 = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    // grad w.r.t. input: beta * grad, reduced back across any broadcast dims.
    auto grad_input_unreduced = mul(grad, beta_);
    auto grad_input = reduce_grad_for_broadcasting(grad_input_unreduced, input_shape_input_);

    // grad w.r.t. mat1: alpha * grad @ mat2^T
    auto mat2_ndim = mat2.shape().size();
    auto mat2_t = transpose(mat2, mat2_ndim - 2, mat2_ndim - 1);
    auto grad_mat1 = mul(matmul(grad, mat2_t), alpha_);

    // grad w.r.t. mat2: alpha * mat1^T @ grad
    auto mat1_ndim = mat1.shape().size();
    auto mat1_t = transpose(mat1, mat1_ndim - 2, mat1_ndim - 1);
    auto grad_mat2 = mul(matmul(mat1_t, grad), alpha_);

    return {grad_input, grad_mat1, grad_mat2};
}

// addmv(input, mat, vec, beta, alpha) = beta*input + alpha*(mat @ vec).
// mat is (M, K), vec is (K,), grad is (M,).
// d/d(input) = beta * grad     (broadcast-reduced to input's shape)
// d/d(mat)   = alpha * outer(grad, vec)
// d/d(vec)   = alpha * mat^T @ grad
auto AddmvBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("AddmvBackward::forward should not be called directly");
}
auto AddmvBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& mat = saved_tensors_[0];
    const auto& vec = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    auto grad_input_unreduced = mul(grad, beta_);
    auto grad_input = reduce_grad_for_broadcasting(grad_input_unreduced, input_shape_input_);

    // outer(grad, vec): (M,) ⊗ (K,) → (M, K)
    auto grad_col = unsqueeze(grad, 1);          // (M, 1)
    auto vec_row = unsqueeze(vec, 0);            // (1, K)
    auto grad_mat = mul(matmul(grad_col, vec_row), alpha_);

    // mat^T @ grad: (K, M) @ (M,) → (K,)
    auto mat_ndim = mat.shape().size();
    auto mat_t = transpose(mat, mat_ndim - 2, mat_ndim - 1);
    auto grad_vec = mul(matmul(mat_t, grad), alpha_);

    return {grad_input, grad_mat, grad_vec};
}

// baddbmm(input, batch1, batch2, beta, alpha):
//   = beta*input + alpha*(batch1 @ batch2), batched on dim 0.
// d/d(input)  = beta * grad           (broadcast-reduced)
// d/d(batch1) = alpha * grad @ batch2^T     (transpose last two dims)
// d/d(batch2) = alpha * batch1^T @ grad
auto BaddbmmBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("BaddbmmBackward::forward should not be called directly");
}
auto BaddbmmBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& batch1 = saved_tensors_[0];
    const auto& batch2 = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    auto grad_input_unreduced = mul(grad, beta_);
    auto grad_input = reduce_grad_for_broadcasting(grad_input_unreduced, input_shape_input_);

    auto b1_ndim = batch1.shape().size();
    auto b2_ndim = batch2.shape().size();
    auto batch2_t = transpose(batch2, b2_ndim - 2, b2_ndim - 1);
    auto batch1_t = transpose(batch1, b1_ndim - 2, b1_ndim - 1);

    auto grad_batch1 = mul(matmul(grad, batch2_t), alpha_);
    auto grad_batch2 = mul(matmul(batch1_t, grad), alpha_);

    return {grad_input, grad_batch1, grad_batch2};
}

// nansum(x, dim, keepdim) — sum with NaN treated as 0. Backward is the
// standard sum backward (broadcast grad back to input shape) with the NaN
// mask zeroed out, since NaN positions contributed 0 to the forward and
// therefore have no local Jacobian. We recompute the mask from the saved
// input rather than storing it as a separate tensor.
auto NansumBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("NansumBackward::forward should not be called directly");
}
auto NansumBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "NansumBackward: cannot compute gradient of nansum over empty tensor");

    auto input_shape_vec =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());

    Tensor expanded;
    if (!dim_.has_value()) {
        auto grad = grad_output;
        if (grad.ndim() > 0) {
            grad = reshape(grad, {});
        }
        expanded = expand(grad, input_shape_vec);
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += static_cast<int64_t>(input.shape().size());
        auto grad = grad_output;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }
        expanded = expand(grad, input_shape_vec);
    }

    // Mask NaN positions in the input to zero in the gradient — their
    // forward contribution was treated as zero, so the partial derivative
    // there is zero (independent of the input value, in the convention
    // PyTorch uses for nansum).
    auto nan_mask = tenzor::isnan(input);
    auto zero = zeros(input_shape_vec, expanded.dtype(), expanded.device());
    auto result = where(nan_mask, zero, expanded);
    return {result};
}

// tile(input, reps) — backward mirrors RepeatBackward. tile() right-aligns
// reps against the input shape, padding with 1s on the left, so the output
// can have more dims than the input. We split each output dim back into
// (reps[i], orig_shape[i]) and sum over the reps axis.
auto TileBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("TileBackward::forward should not be called directly");
}
auto TileBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // Pad original_shape and reps to the same length (the output ndim),
    // right-aligned with 1s on the left — matching tile()'s own padding.
    int64_t in_ndim = static_cast<int64_t>(original_shape_.size());
    int64_t reps_ndim = static_cast<int64_t>(reps_.size());
    int64_t out_ndim = std::max(in_ndim, reps_ndim);

    std::vector<int64_t> padded_shape(out_ndim, 1);
    std::vector<int64_t> padded_reps(out_ndim, 1);
    int64_t shape_offset = out_ndim - in_ndim;
    for (int64_t i = 0; i < in_ndim; ++i) {
        padded_shape[shape_offset + i] = original_shape_[i];
    }
    int64_t reps_offset = out_ndim - reps_ndim;
    for (int64_t i = 0; i < reps_ndim; ++i) {
        padded_reps[reps_offset + i] = reps_[i];
    }

    // Build interleaved reshape: [reps[0], orig[0], reps[1], orig[1], ...].
    std::vector<int64_t> expanded_shape;
    expanded_shape.reserve(2 * out_ndim);
    for (int64_t i = 0; i < out_ndim; ++i) {
        expanded_shape.push_back(padded_reps[i]);
        expanded_shape.push_back(padded_shape[i]);
    }
    auto grad_reshaped = reshape(grad, expanded_shape);

    // Sum over the repeat dims (0, 2, 4, ...) from the highest down so the
    // remaining indices do not shift after each reduction.
    auto result = grad_reshaped;
    for (int64_t i = out_ndim - 1; i >= 0; --i) {
        int64_t repeat_dim = 2 * i;
        result = tenzor::sum(result, repeat_dim, false);
    }

    // If the user supplied fewer reps than input dims, the result above
    // already has the correct input rank because we used original_shape_'s
    // length when padding (the left-padded 1s collapsed to 1s in the sum).
    // Otherwise, when reps had MORE dims than the input, the result still
    // has out_ndim dims of which the leading (out_ndim - in_ndim) are 1s;
    // reshape back to the original input shape.
    if (out_ndim != in_ndim) {
        result = reshape(result, original_shape_);
    }
    return {result};
}

// --- non-differentiable wrappers ---------------------------------------

auto CountNonzeroBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("CountNonzeroBackward::forward should not be called directly");
}
auto CountNonzeroBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "count_nonzero: output is an integer count of non-zero elements — "
        "a discrete quantity, not a smooth function of the input. No "
        "well-defined gradient exists.");
}

auto IsInfBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("IsInfBackward::forward should not be called directly");
}
auto IsInfBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "isinf: output is a Bool tensor (discrete) classifying the input "
        "as ±inf or finite. The inf-pattern is metadata, not a smooth "
        "function of the input values.");
}

auto BitwiseAndBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("BitwiseAndBackward::forward should not be called directly");
}
auto BitwiseAndBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "bitwise_and: integer/bool inputs and output — operates on the "
        "discrete bit representation. The operation is not differentiable.");
}

auto RoundBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("RoundBackward::forward should not be called directly");
}
auto RoundBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "round: piecewise-constant — gradient is zero almost everywhere "
        "with Dirac jumps at half-integer boundaries. Use a custom STE "
        "(straight-through estimator) Function if a surrogate gradient is "
        "needed.");
}

auto EqBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("EqBackward::forward should not be called directly");
}
auto EqBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "eq: output is a Bool tensor (a == b) — discrete comparison, not "
        "a smooth function of the inputs. The comparison family "
        "(eq/ne/lt/le/gt/ge) is all non-differentiable.");
}

// ============================================================================
// Audit E.7 continuation (batch 4): forward/backward impls for the fourth set
// of 10 OpIds. The five comparisons (ne/lt/le/gt/ge) share one reason string
// referencing the full eq/ne/lt/le/gt/ge family — kept verbatim across all
// five so future grep + audit lines up.
// ============================================================================

// Shared comparison-family reason — kept in one place so the message stays
// consistent across the whole boolean-compare cluster.
namespace {
constexpr const char* kComparisonFamilyReason =
    "output is a Bool tensor — discrete comparison, not a smooth function "
    "of the inputs. The comparison family (eq/ne/lt/le/gt/ge) is all "
    "non-differentiable.";
} // namespace

auto NeBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("NeBackward::forward should not be called directly");
}
auto NeBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(std::string("ne: ") + kComparisonFamilyReason);
}

auto LtBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("LtBackward::forward should not be called directly");
}
auto LtBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(std::string("lt: ") + kComparisonFamilyReason);
}

auto LeBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("LeBackward::forward should not be called directly");
}
auto LeBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(std::string("le: ") + kComparisonFamilyReason);
}

auto GtBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("GtBackward::forward should not be called directly");
}
auto GtBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(std::string("gt: ") + kComparisonFamilyReason);
}

auto GeBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("GeBackward::forward should not be called directly");
}
auto GeBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(std::string("ge: ") + kComparisonFamilyReason);
}

// --- bitwise ops (integer / bool, discrete bit-level) -------------------

auto BitwiseOrBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("BitwiseOrBackward::forward should not be called directly");
}
auto BitwiseOrBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "bitwise_or: integer / bool inputs and output — operates on the "
        "discrete bit representation. The operation is not differentiable.");
}

auto BitwiseXorBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("BitwiseXorBackward::forward should not be called directly");
}
auto BitwiseXorBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "bitwise_xor: integer / bool inputs and output — operates on the "
        "discrete bit representation. The operation is not differentiable.");
}

auto BitwiseNotBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("BitwiseNotBackward::forward should not be called directly");
}
auto BitwiseNotBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "bitwise_not: integer / bool input — unary bit complement on the "
        "discrete bit representation. The operation is not differentiable.");
}

// --- introspection ------------------------------------------------------

auto IsFiniteBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("IsFiniteBackward::forward should not be called directly");
}
auto IsFiniteBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "isfinite: output is a Bool tensor classifying the input as "
        "finite vs. (±inf, NaN). The finiteness-pattern is metadata, not "
        "a smooth function of the input values.");
}

// --- differentiable: logcumsumexp ---------------------------------------

// logcumsumexp(x, dim): y_j = log( sum_{k <= j} exp(x_k) )  along dim.
//
// dL/dx_i = sum_{j >= i} (dL/dy_j) * exp(x_i - y_j)
//         = exp(x_i) * sum_{j >= i} (dL/dy_j) * exp(-y_j)
//
// The "sum_{j >= i}" along dim is a reverse cumulative sum, which equals
// flip(cumsum(flip(z, dim), dim), dim). exp(-y) is numerically safe here
// because y = log(cumsum(exp(x))) is bounded below by max-over-prefix(x),
// so exp(-y) <= 1 / exp(min_prefix(x)) and never overflows for normal x.
auto LogcumsumexpBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("LogcumsumexpBackward::forward should not be called directly");
}
auto LogcumsumexpBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& x = saved_tensors_[0];
    const auto& y = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    // Normalise dim against the saved input's rank.
    int64_t dim = dim_;
    int64_t ndim = static_cast<int64_t>(x.shape().size());
    if (dim < 0) dim += ndim;
    TENZOR_CHECK_SHAPE(dim >= 0 && dim < ndim,
        "LogcumsumexpBackward: dim out of range for saved input rank");

    // z = grad * exp(-y)
    auto neg_y = neg(y);
    auto exp_neg_y = exp(neg_y);
    auto z = mul(grad, exp_neg_y);

    // rev_cum = flip(cumsum(flip(z, dim), dim), dim)
    auto flipped = flip(z, {dim});
    auto cum = cumsum(flipped, dim);
    auto rev_cum = flip(cum, {dim});

    // grad_x = exp(x) * rev_cum
    auto grad_x = mul(exp(x), rev_cum);
    return {grad_x};
}

// ============================================================================
// Audit E.7 continuation (batch 5): 10 more OpIds.
// ============================================================================

// --- non-differentiable: isposinf / isneginf (bool metadata) ------------

auto IsPosInfBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("IsPosInfBackward::forward should not be called directly");
}
auto IsPosInfBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "isposinf: output is a Bool tensor flagging +inf positions — "
        "discrete classification of the input's IEEE-754 bit pattern, not "
        "a smooth function of the input values. Non-differentiable.");
}

auto IsNegInfBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("IsNegInfBackward::forward should not be called directly");
}
auto IsNegInfBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "isneginf: output is a Bool tensor flagging -inf positions — "
        "discrete classification of the input's IEEE-754 bit pattern, not "
        "a smooth function of the input values. Non-differentiable.");
}

// --- non-differentiable: trunc (piecewise-constant) ---------------------

auto TruncBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("TruncBackward::forward should not be called directly");
}
auto TruncBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "trunc: round-toward-zero is piecewise-constant — gradient is "
        "zero almost everywhere and a Dirac comb at integer jumps. Use "
        "a custom STE Function (straight-through estimator) if a surrogate "
        "gradient is needed.");
}

// --- non-differentiable: any / all / has_inf_nan (bool reductions) ------

auto AnyBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("AnyBackward::forward should not be called directly");
}
auto AnyBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "any: output is a Bool reduction (true if any input element is "
        "nonzero) — discrete and not a smooth function of the inputs. "
        "Non-differentiable.");
}

auto AllBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("AllBackward::forward should not be called directly");
}
auto AllBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "all: output is a Bool reduction (true if all input elements are "
        "nonzero) — discrete and not a smooth function of the inputs. "
        "Non-differentiable.");
}

auto HasInfNanBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("HasInfNanBackward::forward should not be called directly");
}
auto HasInfNanBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "has_inf_nan: output is a Bool scalar flagging the presence of "
        "any inf or NaN element — discrete classification of the input's "
        "IEEE-754 bit patterns. Non-differentiable.");
}

// --- differentiable: nanmean --------------------------------------------

// nanmean(x, dim, keepdim) = sum_{i not NaN} x_i / N, where N is the per-output
// count of non-NaN entries along the reduction axis. The Jacobian wrt a
// non-NaN entry x_i is 1/N; wrt a NaN entry it is 0 (the NaN contributed
// neither to the numerator nor to the denominator). Backward broadcasts
// grad_y back to input shape and divides by the broadcast count, then zeros
// out the NaN positions. N is recomputed from the saved input.
auto NanmeanBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("NanmeanBackward::forward should not be called directly");
}
auto NanmeanBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& input = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "NanmeanBackward: cannot compute gradient of nanmean over empty tensor");

    auto input_shape_vec =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // non_nan_mask: 1 at non-NaN positions, 0 at NaN positions; cast to the
    // input dtype so we can sum it to a float count.
    auto nan_mask = tenzor::isnan(input);
    auto non_nan_mask = tenzor::logical_not(nan_mask).to(input.dtype());

    // Per-output count N of non-NaN entries along the reduction axis,
    // produced with keepdim=true so it broadcasts back to input shape.
    Tensor count_keepdim;
    Tensor grad_broadcast;
    if (!dim_.has_value()) {
        // Full reduction: count is a scalar; grad is a scalar; both broadcast
        // to input shape.
        count_keepdim = tenzor::sum(non_nan_mask);
        auto grad = grad_output;
        if (grad.ndim() > 0) {
            grad = reshape(grad, {});
        }
        grad_broadcast = expand(grad, input_shape_vec);
        count_keepdim = expand(count_keepdim, input_shape_vec);
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += static_cast<int64_t>(input.shape().size());
        TENZOR_CHECK_SHAPE(dim >= 0 && dim < static_cast<int64_t>(input.shape().size()),
            "NanmeanBackward: dim out of range for saved input rank");

        count_keepdim = tenzor::sum(non_nan_mask, dim, /*keepdim=*/true);
        auto grad = grad_output;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }
        grad_broadcast = expand(grad, input_shape_vec);
        count_keepdim = expand(count_keepdim, input_shape_vec);
    }

    // grad / N at non-NaN positions; 0 at NaN positions (using mul by mask
    // also handles divide-by-zero rows where the entire reduction window is
    // NaN — those rows produce NaN output anyway, but the grad we route
    // upstream is well-defined as zero).
    auto safe_count = tenzor::where(
        tenzor::eq(count_keepdim, zeros(input_shape_vec, count_keepdim.dtype(), count_keepdim.device())),
        ones(input_shape_vec, count_keepdim.dtype(), count_keepdim.device()),
        count_keepdim);
    auto grad_per_elem = tenzor::div(grad_broadcast, safe_count);
    auto zero = zeros(input_shape_vec, grad_per_elem.dtype(), grad_per_elem.device());
    auto result = tenzor::where(nan_mask, zero, grad_per_elem);
    return {result};
}

// --- differentiable: masked_fill ---------------------------------------

// masked_fill(x, mask, value): y_i = mask_i ? value : x_i.
// dy_i/dx_i = 1 - mask_i (i.e. 0 where mask=true, 1 where mask=false).
// value is a constant scalar and is not part of the autograd graph.
auto MaskedFillBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("MaskedFillBackward::forward should not be called directly");
}
auto MaskedFillBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& mask = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    auto shape_vec =
        std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end());
    auto zero = zeros(shape_vec, grad_output.dtype(), grad_output.device());
    auto grad_x = tenzor::where(mask, zero, grad_output);
    return {grad_x};
}

// --- differentiable: masked_select -------------------------------------

// masked_select(x, mask): y = flat tensor of x's elements where mask=true
// (in row-major iteration order). dy_k/dx_i = 1 iff i is the k-th masked
// position, else 0. Backward scatters grad_y into a zeros_like(x) at the
// masked positions, which is exactly masked_scatter(zeros_like(x), mask, grad_y).
auto MaskedSelectBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("MaskedSelectBackward::forward should not be called directly");
}
auto MaskedSelectBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& mask = saved_tensors_[0];
    const auto& input_shape_t = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    // Recover the input shape from the saved Int64 shape tensor (kept on CPU).
    std::vector<int64_t> input_shape_vec(input_shape_t.numel());
    if (input_shape_t.numel() > 0) {
        std::memcpy(input_shape_vec.data(), input_shape_t.data_ptr(),
                    input_shape_vec.size() * sizeof(int64_t));
    }

    auto zero = zeros(input_shape_vec, grad_output.dtype(), grad_output.device());
    auto grad_x = tenzor::masked_scatter(zero, mask, grad_output);
    return {grad_x};
}

// --- differentiable: masked_scatter ------------------------------------

// masked_scatter(x, mask, source): y = x with the first popcount(mask)
// elements of source written into the masked positions of x (in mask-iteration
// order). The remaining elements of source are unused.
// d/dx_i = 1 - mask_i (passes grad through at non-masked positions).
// d/dsource_k = grad_y at the position of the k-th masked entry, for
// k < popcount(mask); else 0. That is: masked_select(grad_y, mask), padded
// with zeros up to source.numel() and reshaped to source.shape.
auto MaskedScatterBackward::forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> {
    throw std::runtime_error("MaskedScatterBackward::forward should not be called directly");
}
auto MaskedScatterBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& mask = saved_tensors_[0];
    const auto& grad_output = grad_outputs[0];

    auto shape_vec =
        std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end());

    // grad_x: zero out the masked positions (overwritten by source in forward).
    auto zero_like = zeros(shape_vec, grad_output.dtype(), grad_output.device());
    auto grad_x = tenzor::where(mask, zero_like, grad_output);

    // grad_source: take grad at masked positions, pad with zeros to source
    // total numel, reshape to source.shape.
    auto selected = tenzor::masked_select(grad_output, mask);
    int64_t source_numel = 1;
    for (auto d : source_shape_) source_numel *= d;

    int64_t selected_n = selected.numel();
    Tensor grad_source_flat;
    if (selected_n >= source_numel) {
        // popcount(mask) >= source.numel(): every source element was used.
        // Trim to source_numel along the only dim.
        grad_source_flat = (selected_n == source_numel)
            ? selected
            : selected.slice(0, 0, source_numel);
    } else {
        // popcount(mask) < source.numel(): pad with zeros at the tail.
        int64_t pad = source_numel - selected_n;
        auto tail = zeros({pad}, grad_output.dtype(), grad_output.device());
        std::vector<Tensor> parts = {selected, tail};
        grad_source_flat = tenzor::cat(parts, /*dim=*/0);
    }
    auto grad_source = reshape(grad_source_flat, source_shape_);
    return {grad_x, grad_source};
}

// ============================================================================
// Audit E.7 batch 6 — special-math closed forms + view/index ops
// ============================================================================

// --- Igamma --------------------------------------------------------------
// y = igamma(a, x) = P(a, x). dP/dx = x^(a-1) * exp(-x) / Gamma(a).
// d/da has no elementary closed form -> zero grad for `a`.
auto IgammaBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IgammaBackward::forward should not be called directly");
}
auto IgammaBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& a = saved_tensors_[0];
    const auto& x = saved_tensors_[1];

    auto a_shape = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    auto x_shape = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    auto one_a = ones(a_shape, a.dtype(), a.device());

    // log dP/dx = (a-1)*log(x) - x - lgamma(a)
    auto am1 = sub(a, one_a);
    auto log_x = tenzor::log(x);
    auto lg_a = tenzor::lgamma(a);
    auto log_deriv = sub(sub(mul(am1, log_x), x), lg_a);
    auto deriv = tenzor::exp(log_deriv);

    auto grad_a = zeros(a_shape, a.dtype(), a.device());
    auto grad_x = mul(grad_outputs[0], deriv);
    return {grad_a, grad_x};
}

// --- Igammac -------------------------------------------------------------
// y = igammac(a, x) = Q(a, x). dQ/dx = -x^(a-1) * exp(-x) / Gamma(a).
auto IgammacBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IgammacBackward::forward should not be called directly");
}
auto IgammacBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& a = saved_tensors_[0];
    const auto& x = saved_tensors_[1];

    auto a_shape = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    auto one_a = ones(a_shape, a.dtype(), a.device());

    auto am1 = sub(a, one_a);
    auto log_x = tenzor::log(x);
    auto lg_a = tenzor::lgamma(a);
    auto log_pos = sub(sub(mul(am1, log_x), x), lg_a);
    auto deriv = neg(tenzor::exp(log_pos));

    auto grad_a = zeros(a_shape, a.dtype(), a.device());
    auto grad_x = mul(grad_outputs[0], deriv);
    return {grad_a, grad_x};
}

// --- Beta ----------------------------------------------------------------
// y = B(a, b) = Gamma(a) Gamma(b) / Gamma(a+b).
// dB/da = B * (psi(a) - psi(a+b)), dB/db = B * (psi(b) - psi(a+b)).
auto BetaBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("BetaBackward::forward should not be called directly");
}
auto BetaBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(3);
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& y = saved_tensors_[2]; // beta(a, b)

    auto sum_ab = add(a, b);
    auto psi_ab = tenzor::digamma(sum_ab);
    auto psi_a = tenzor::digamma(a);
    auto psi_b = tenzor::digamma(b);

    auto factor_a = sub(psi_a, psi_ab);
    auto factor_b = sub(psi_b, psi_ab);
    auto dB_da = mul(y, factor_a);
    auto dB_db = mul(y, factor_b);

    auto grad_a_full = mul(grad_outputs[0], dB_da);
    auto grad_b_full = mul(grad_outputs[0], dB_db);

    auto grad_a = reduce_grad_for_broadcasting(grad_a_full, input_shape_a_);
    auto grad_b = reduce_grad_for_broadcasting(grad_b_full, input_shape_b_);
    return {grad_a, grad_b};
}

// --- PairwiseDistance ----------------------------------------------------
// y[b] = (Sum_d |x1[b,d] - x2[b,d]|^p)^(1/p), input (B, D), output (B,).
// dy[b]/dx1[b,d] = sign(d[b,d]) * |d[b,d]|^(p-1) * y[b]^(1-p),
// dy[b]/dx2[b,d] = -dy[b]/dx1[b,d].
auto PairwiseDistanceBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("PairwiseDistanceBackward::forward should not be called directly");
}
auto PairwiseDistanceBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(3);
    const auto& x1 = saved_tensors_[0];
    const auto& x2 = saved_tensors_[1];
    const auto& y  = saved_tensors_[2]; // shape (B,)

    auto diff = sub(x1, x2); // (B, D)

    auto sign_d = tenzor::sign(diff);
    auto abs_d = tenzor::abs(diff);
    auto abs_pow = tenzor::pow(abs_d, p_ - 1.0);  // (B, D)

    // Regularise y to avoid 0^(1-p) on degenerate zero-distance rows.
    // We add a tiny constant to y purely for the division step; the
    // gradient at exactly zero distance is mathematically undefined for
    // p != 2 anyway (the subgradient set contains 0). Using a small
    // epsilon keeps the gradient finite and follows the PyTorch
    // convention of regularising via the input difference.
    auto eps_t = full_like(y, 1e-30);
    auto y_safe = add(y, eps_t);
    auto y_factor = tenzor::pow(y_safe, 1.0 - p_); // (B,)
    auto y_factor_exp = unsqueeze(y_factor, 1);    // (B, 1)

    auto grad_y = grad_outputs[0];                 // (B,)
    auto grad_y_exp = unsqueeze(grad_y, 1);        // (B, 1)
    auto scale = mul(grad_y_exp, y_factor_exp);    // (B, 1)
    auto per_elem = mul(mul(sign_d, abs_pow), scale);  // (B, D)

    auto grad_x1 = per_elem;
    auto grad_x2 = neg(per_elem);
    return {grad_x1, grad_x2};
}

// --- Pdist (non-diff, pending dedicated kernel) --------------------------
auto PdistBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("PdistBackward::forward should not be called directly");
}
auto PdistBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "pdist: closed-form per-pair backward exists, but mapping it back into "
        "the (N, D) input requires materialising a dense (N, N, D) intermediate "
        "and scatter-summing pair gradients across pairs. That needs a "
        "dedicated backward kernel (PyTorch ships `_pdist_backward` for this "
        "reason). Until a kernel lands, pdist is marked non-differentiable. "
        "Wrap in a custom autograd::Function with the explicit pair-difference "
        "computation if you need gradients now.");
}

// --- CDist (non-diff, pending dedicated kernel) --------------------------
auto CDistBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CDistBackward::forward should not be called directly");
}
auto CDistBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "cdist: closed-form per-pair backward exists, but the scatter step "
        "into x1 (N, D) and x2 (M, D) requires a dedicated backward kernel "
        "(PyTorch ships `_cdist_backward` for this reason). Until a kernel "
        "lands, cdist is marked non-differentiable. Wrap in a custom "
        "autograd::Function with the explicit pair-difference computation "
        "if you need gradients now.");
}

// --- AdvancedIndex -------------------------------------------------------
// y = x[indices]; correct backward is scatter-add of grad_y into
// zeros_like(x) at the indexed positions (accumulating because duplicate
// indices may appear). The project's current AdvancedIndexPut kernels are
// pure overwrite (no `accumulate` flag plumbed through across CPU / CUDA /
// ROCm / OneAPI / Vulkan), and `tenzor::index_put` is in-place
// overwrite-only. Implementing the multi-dim accumulate scatter purely
// at the Variable level (flatten -> compute 1D index -> index_add ->
// reshape) is feasible but requires care for full-slice sentinel dims and
// broadcasting; it belongs in a follow-up that also adds the missing
// `accumulate` path to the AdvancedIndexPut kernels for parity. Until
// then, mark non-differentiable with a clear pointer.
auto AdvancedIndexBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("AdvancedIndexBackward::forward should not be called directly");
}
auto AdvancedIndexBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "advanced_index (x[indices]): closed-form backward is scatter-add of "
        "the output gradient back into a zero-filled source-shaped tensor "
        "(duplicate indices must accumulate). The project's current "
        "AdvancedIndexPut kernels do not expose an `accumulate` mode, and "
        "`tenzor::index_put` is overwrite-only — that path would silently "
        "drop duplicate-index contributions. A follow-up should add an "
        "accumulating scatter kernel; until then this op is marked "
        "non-differentiable so callers fail loudly instead of getting wrong "
        "gradients. Use `gather`/`index_select` for the single-dim case if "
        "you need autograd today.");
}

// --- OneHot --------------------------------------------------------------
// One-hot of integer indices: non-differentiable input.
auto OneHotBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("OneHotBackward::forward should not be called directly");
}
auto OneHotBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "one_hot: input is an integer index tensor and the output is a "
        "discrete encoding (not a smooth function of the indices). No "
        "well-defined gradient exists. Use a soft relaxation "
        "(e.g. softmax of logits, Gumbel-softmax) if you need a "
        "differentiable approximation.");
}

// --- Lerp ----------------------------------------------------------------
// y = start + weight * (end - start) = (1 - weight) * start + weight * end
//   d/dstart  = 1 - weight
//   d/dend    = weight
//   d/dweight = end - start    (tensor-weight overload only)
auto LerpBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LerpBackward::forward should not be called directly");
}
auto LerpBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    if (has_weight_tensor_) {
        require_saved_tensors(3);
        const auto& start  = saved_tensors_[0];
        const auto& end    = saved_tensors_[1];
        const auto& weight = saved_tensors_[2];

        auto one_w = ones(
            std::vector<int64_t>(weight.shape().begin(), weight.shape().end()),
            weight.dtype(), weight.device());
        auto one_minus_w = sub(one_w, weight);

        auto grad_start_full  = mul(grad, one_minus_w);
        auto grad_end_full    = mul(grad, weight);
        auto grad_weight_full = mul(grad, sub(end, start));

        auto grad_start  = reduce_grad_for_broadcasting(grad_start_full,  input_shape_start_);
        auto grad_end    = reduce_grad_for_broadcasting(grad_end_full,    input_shape_end_);
        auto grad_weight = reduce_grad_for_broadcasting(grad_weight_full, input_shape_weight_);

        return {grad_start, grad_end, grad_weight};
    }

    // Scalar-weight overload: only start, end are saved.
    require_saved_tensors(2);
    auto one_minus_w_scalar = 1.0 - weight_scalar_;
    auto grad_start_full = mul(grad, one_minus_w_scalar);
    auto grad_end_full   = mul(grad, weight_scalar_);
    auto grad_start = reduce_grad_for_broadcasting(grad_start_full, input_shape_start_);
    auto grad_end   = reduce_grad_for_broadcasting(grad_end_full,   input_shape_end_);
    return {grad_start, grad_end};
}

// --- Cross ---------------------------------------------------------------
// y = cross(a, b, dim). Using a x b = -(b x a) and bilinearity:
//   grad_a = cross(b, grad, dim)
//   grad_b = cross(grad, a, dim)
auto CrossBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CrossBackward::forward should not be called directly");
}
auto CrossBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    auto grad_a = tenzor::cross(b, grad, dim_);
    auto grad_b = tenzor::cross(grad, a, dim_);
    return {grad_a, grad_b};
}

// ============================================================================
// Audit E.7 batch 7 — index/scatter/view ops
// ============================================================================

// --- IndexAdd ------------------------------------------------------------
// y = index_add(x, dim, index, source): y[..., index[i], ...] += source[..., i, ...].
//   dy/dx_pos = 1 for every pos       (the add leaves x untouched in dim-stride
//                                       slots that source doesn't hit, and adds
//                                       1*src into the slots it does)
//   dy/dsource_i = 1 at output row index[i] along dim
// Therefore grad_x = grad_y; grad_source = index_select(grad_y, dim, index).
// index is an integer tensor and is not differentiable.
auto IndexAddBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IndexAddBackward::forward should not be called directly");
}
auto IndexAddBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& index = saved_tensors_[0];
    const auto& grad_y = grad_outputs[0];

    auto grad_x = grad_y;
    auto grad_source = tenzor::index_select(grad_y, dim_, index);
    return {grad_x, grad_source};
}

// --- IndexCopy -----------------------------------------------------------
// y = index_copy(x, dim, index, source): y[..., index[i], ...] = source[..., i, ...]
// for each i; other positions equal the original x.
//   dy/dx_pos = 0 if pos is among the indexed slots, else 1
//   dy/dsource_i = 1 at output row index[i] along dim
// grad_x       = index_fill(grad_y, dim, index, 0)   (zero out overwritten slots)
// grad_source  = index_select(grad_y, dim, index)
auto IndexCopyBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IndexCopyBackward::forward should not be called directly");
}
auto IndexCopyBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& index = saved_tensors_[0];
    const auto& grad_y = grad_outputs[0];

    auto grad_x = tenzor::index_fill(grad_y, dim_, index, /*value=*/0.0f);
    auto grad_source = tenzor::index_select(grad_y, dim_, index);
    return {grad_x, grad_source};
}

// --- IndexFill -----------------------------------------------------------
// y = index_fill(x, dim, index, value): indexed slots overwritten with a
// scalar; non-indexed slots equal x. value is a non-diff scalar.
//   dy/dx_pos = 0 if pos is indexed, else 1
// grad_x = index_fill(grad_y, dim, index, 0)
auto IndexFillBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IndexFillBackward::forward should not be called directly");
}
auto IndexFillBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& index = saved_tensors_[0];
    const auto& grad_y = grad_outputs[0];

    auto grad_x = tenzor::index_fill(grad_y, dim_, index, /*value=*/0.0f);
    return {grad_x};
}

// --- SelectScatter -------------------------------------------------------
// y = select_scatter(x, src, dim, index): copy of x with src written at
// x.select(dim, index). src has shape == x with dim removed.
//   grad_x   = select_scatter(grad_y, zeros_like(src_slice), dim, index)
//   grad_src = select(grad_y, dim, index)
auto SelectScatterBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SelectScatterBackward::forward should not be called directly");
}
auto SelectScatterBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_y = grad_outputs[0];

    // grad_src is exactly the slice of grad_y at (dim, index). It already has
    // the right shape (dim removed), so a single select() is sufficient.
    auto grad_src = tenzor::select(grad_y, dim_, index_);

    // grad_x = grad_y with the (dim, index) slot zeroed out. Build a zero
    // tensor with the shape of the slice and scatter back in.
    std::vector<int64_t> slice_shape;
    auto y_shape = grad_y.shape();
    int64_t nd = static_cast<int64_t>(y_shape.size());
    int64_t dim_norm = dim_ < 0 ? dim_ + nd : dim_;
    slice_shape.reserve(static_cast<size_t>(nd - 1));
    for (int64_t d = 0; d < nd; ++d) {
        if (d != dim_norm) slice_shape.push_back(y_shape[d]);
    }
    auto zero_slice = zeros(slice_shape, grad_y.dtype(), grad_y.device());
    auto grad_x = tenzor::select_scatter(grad_y, zero_slice, dim_, index_);

    // grad_src needs to be contiguous since `select` returns a view.
    if (!grad_src.is_contiguous()) grad_src = grad_src.contiguous();
    return {grad_x, grad_src};
}

// --- SliceScatter --------------------------------------------------------
// y = slice_scatter(x, src, dim, start, end, step): copy of x with src
// written into the slice region. src shape matches the slice.
//   grad_x   = slice_scatter(grad_y, zeros_like(slice), dim, start, end, step)
//   grad_src = slice(grad_y, dim, start, end, step).contiguous()
auto SliceScatterBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SliceScatterBackward::forward should not be called directly");
}
auto SliceScatterBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_y = grad_outputs[0];

    auto zero_slice = zeros(src_shape_, grad_y.dtype(), grad_y.device());
    auto grad_x = tenzor::slice_scatter(grad_y, zero_slice, dim_, start_, end_, step_);

    // The forward op resolves end=-1 against the actual extent. We don't have
    // that resolved value here, but `slice` accepts the same conventions, so
    // the slice we extract has the same shape as src_shape_ by construction.
    int64_t end_resolved = end_;
    if (end_resolved < 0) {
        auto y_shape = grad_y.shape();
        int64_t nd = static_cast<int64_t>(y_shape.size());
        int64_t dim_norm = dim_ < 0 ? dim_ + nd : dim_;
        end_resolved = y_shape[dim_norm];
    }
    auto grad_src_view = tenzor::slice(grad_y, dim_, start_, end_resolved, step_);
    auto grad_src = grad_src_view.is_contiguous() ? grad_src_view
                                                  : grad_src_view.contiguous();
    return {grad_x, grad_src};
}

// --- DiagonalScatter (non-diff: missing general diagonal extractor) ------
auto DiagonalScatterBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("DiagonalScatterBackward::forward should not be called directly");
}
auto DiagonalScatterBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "diagonal_scatter: closed-form backward is `diagonal_scatter(grad_y, "
        "zeros_like(src), ...)` for grad_input plus `diagonal(grad_y, offset, "
        "dim1, dim2)` for grad_src. The project currently only ships the 2D "
        "shortcut `diag(...)` and no general N-D `diagonal(offset, dim1, dim2)` "
        "extractor, so grad_src cannot be computed for arbitrary (dim1, dim2). "
        "Marked non-differentiable until a general `diagonal` view is added "
        "to the public tensor API.");
}

// --- RepeatInterleave (uniform integer repeats) --------------------------
// y = repeat_interleave(x, repeats: int, dim). Each element along dim is
// repeated `repeats` times consecutively. Backward: reshape the repeated
// axis as (orig, repeats), sum over the repeats axis, reshape to input shape.
//
// When dim is nullopt the forward flattens x first and the output is 1D,
// so the backward sums in 1D and reshapes back to input_shape_.
auto RepeatInterleaveBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("RepeatInterleaveBackward::forward should not be called directly");
}
auto RepeatInterleaveBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Sentinel: repeats_ == -1 means the wrapper saw a tensor-valued repeats
    // overload (per-element variable-length expansion). That case has no
    // closed-form Variable-level backward without an accumulating scatter
    // keyed by the per-element repeat counts. Fail loudly here.
    if (repeats_ < 0) {
        throw NonDifferentiable(
            "repeat_interleave with a tensor `repeats`: per-element variable-"
            "length expansion has no closed-form Variable-level backward "
            "without an accumulating scatter keyed by the per-element repeat "
            "counts. The uniform integer-`repeats` overload is differentiable; "
            "use it if you need autograd. A follow-up should add a dedicated "
            "backward that consumes the per-element counts and scatter-adds "
            "across the expansion.");
    }
    const auto& grad_y = grad_outputs[0];

    // Degenerate repeats=1: identity, with a possible flatten for nullopt dim.
    // Handle by going through the same reshape-sum-reshape path which collapses
    // to a no-op sum when the repeats axis has length 1.
    int64_t r = repeats_;

    // CC.2: `was_flattened_` is the authoritative signal for which forward
    // path ran. Previously the backward only looked at `dim_.has_value()`,
    // which conflated "user passed nullopt → forward flattened input" with
    // an explicit dim=0 on a 1D input. They produce the same forward output
    // but require different backward reshapes: the flattened path must
    // restore the original rank-N shape, the dim-specified path must keep
    // the existing rank.
    if (!was_flattened_) {
        // grad_y has shape == input with dim*r at axis `dim`.
        // `dim_` is guaranteed to have a value on this branch (the wrapper
        // only sets was_flattened_=true when dim was nullopt).
        int64_t nd = static_cast<int64_t>(input_shape_.size());
        int64_t dim_norm = *dim_ < 0 ? *dim_ + nd : *dim_;

        std::vector<int64_t> split_shape;
        split_shape.reserve(static_cast<size_t>(nd) + 1);
        for (int64_t d = 0; d < nd; ++d) {
            if (d == dim_norm) {
                split_shape.push_back(input_shape_[d]); // orig
                split_shape.push_back(r);               // repeats inserted after
            } else {
                split_shape.push_back(input_shape_[d]);
            }
        }
        auto reshaped = reshape(grad_y, split_shape);
        // Sum over the inserted repeats axis (= dim_norm + 1).
        auto summed = tenzor::sum(reshaped, /*dim=*/dim_norm + 1, /*keepdim=*/false);
        // summed.shape now equals input_shape_.
        return {summed};
    }

    // was_flattened_ == true: forward flattened x to 1D and repeated.
    // grad_y is 1D of length (numel(x) * r); reshape to (numel(x), r), sum
    // along axis 1, then reshape back to the original rank-N input_shape_.
    int64_t numel = 1;
    for (auto d : input_shape_) numel *= d;
    auto reshaped = reshape(grad_y, {numel, r});
    auto summed_1d = tenzor::sum(reshaped, /*dim=*/1, /*keepdim=*/false);
    auto grad_x = reshape(summed_1d, input_shape_);
    return {grad_x};
}

// --- Unfold (im2col) -----------------------------------------------------
// y = unfold(x, k, s, p, d): patches of x as a (N, C*K*K, L) tensor.
// The linear adjoint of unfold (scatter-add overlapping patches back into the
// original spatial grid) is exactly `fold`. So grad_x = fold(grad_y, (H, W),
// k, s, p, d). Saves (H, W) from the input shape.
auto UnfoldBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("UnfoldBackward::forward should not be called directly");
}
auto UnfoldBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_y = grad_outputs[0];
    auto grad_x = tenzor::ops::fold(grad_y, /*output_size=*/{H_, W_}, kernel_size_,
                                    stride_, padding_, dilation_);
    return {grad_x};
}

// --- Nonzero (non-diff: integer indices output) --------------------------
auto NonzeroBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("NonzeroBackward::forward should not be called directly");
}
auto NonzeroBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "nonzero: output is an Int64 index tensor and is not a smooth function "
        "of the input (the count and ordering of nonzero positions changes "
        "discretely under perturbations). No meaningful gradient exists. Use "
        "a soft surrogate (e.g. `where(x != 0, 1, 0).sum()`) if you need a "
        "differentiable count of nonzero entries.");
}

// --- Unique (non-diff: discontinuous sorting/dedup) ----------------------
auto UniqueBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("UniqueBackward::forward should not be called directly");
}
auto UniqueBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    throw NonDifferentiable(
        "unique: forward performs sorting and deduplication, both of which "
        "are discontinuous in the input under ties — the set of selected "
        "positions jumps under arbitrarily small perturbations. No "
        "well-defined gradient exists (strict policy, matching Sign). If "
        "you need a differentiable selection, replace with a soft top-k "
        "or a learned attention/clustering relaxation.");
}

// =========================================================================
// Audit E.7 batch 8 — order statistics, integration, segment reductions
// =========================================================================

namespace {

// Mask helper: 1.0 at positions where |x - target| < tie_eps (broadcast),
// 0.0 elsewhere. dtype/device matches `x`. Used by Aminmax/Nanmedian to
// scatter grad onto value-matching positions (same epsilon ladder as
// MaxBackward/MedianBackward).
auto value_tie_mask(const Tensor& x, const Tensor& target_expanded) -> Tensor {
    auto input_shape_vec = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    double eps_val;
    switch (x.dtype()) {
        case DType::Float64:  eps_val = 1e-12; break;
        case DType::Float16:
        case DType::BFloat16: eps_val = 1e-3; break;
        default:              eps_val = 1e-7; break;
    }
    auto diff = sub(x, target_expanded);
    auto abs_diff = abs(diff);
    auto epsilon = full(input_shape_vec, eps_val, x.dtype(), x.device());
    auto mask_bool = lt(abs_diff, epsilon);
    auto ones_t = ones(input_shape_vec, x.dtype(), x.device());
    auto zeros_t = zeros(input_shape_vec, x.dtype(), x.device());
    return where(mask_bool, ones_t, zeros_t);
}

}  // namespace

// --- Aminmax (differentiable: scatter to argmin OR argmax positions) -----
//
// One AminmaxBackward instance per output Variable. is_max_=false handles
// the min branch (scatter incoming grad onto positions equal to the saved
// min value); is_max_=true handles the max branch. The autograd engine
// sums upstream when the same input feeds both branches.
//
// Saves: input, this branch's value tensor.
auto AminmaxBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("AminmaxBackward::forward should not be called directly");
}
auto AminmaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& vals  = saved_tensors_[1];
    const auto& grad_out = grad_outputs[0];

    auto input_shape_vec =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Broadcast saved values and grad back to input shape.
    Tensor v = vals;
    Tensor g = grad_out;
    if (!dim_.has_value()) {
        if (v.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            v = reshape(v, ones_shape);
        }
        if (g.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            g = reshape(g, ones_shape);
        }
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += static_cast<int64_t>(input.shape().size());
        if (!keepdim_) {
            v = unsqueeze(v, dim);
            g = unsqueeze(g, dim);
        }
    }
    auto v_exp = expand(v, input_shape_vec);
    auto g_exp = expand(g, input_shape_vec);

    auto mask = value_tie_mask(input, v_exp);
    Tensor tie_count;
    if (!dim_.has_value()) {
        tie_count = expand(sum(mask), input_shape_vec);
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += static_cast<int64_t>(input.shape().size());
        tie_count = expand(sum(mask, dim, /*keepdim=*/true), input_shape_vec);
    }
    auto norm = div(mask, tie_count);
    (void)is_max_;  // The branch's value is already saved; flag is metadata.
    return {mul(g_exp, norm)};
}

// --- Kthvalue (differentiable: scatter at k-th position) -----------------
//
// y = kthvalue(x, k, dim)[0] is a single order statistic per reduction
// row. Backward is identical to Median's tie-mask scatter, but the value
// to match is the saved k-th value (no NaN handling). Saves: input, values.
auto KthvalueBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("KthvalueBackward::forward should not be called directly");
}
auto KthvalueBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "KthvalueBackward: cannot compute gradient over empty tensor");

    auto input_shape_vec =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());

    int64_t dim = dim_;
    if (dim < 0) dim += static_cast<int64_t>(input.shape().size());

    Tensor out = output;
    Tensor grad = grad_output;
    if (!keepdim_) {
        out = unsqueeze(out, dim);
        grad = unsqueeze(grad, dim);
    }
    auto out_exp = expand(out, input_shape_vec);
    auto grad_exp = expand(grad, input_shape_vec);

    auto mask = value_tie_mask(input, out_exp);
    auto tie_count = sum(mask, dim, /*keepdim=*/true);
    auto tie_exp = expand(tie_count, input_shape_vec);
    auto norm_mask = div(mask, tie_exp);
    return {mul(grad_exp, norm_mask)};
}

// --- Quantile (non-diff: needs stable per-row argsort with interp weights)
auto QuantileBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("QuantileBackward::forward should not be called directly");
}
auto QuantileBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "quantile: the linear adjoint of an interpolated quantile is a "
        "two-position scatter onto the per-row sort permutation flanking q. "
        "The project does not currently expose a stable per-row argsort "
        "primitive together with the interpolation weights, and computing "
        "the backward from saved inputs/outputs alone would require sorting "
        "again at backward time and re-deriving the two flanking indices "
        "for every reduction window — a workaround we will not ship. "
        "Marking NonDifferentiable until that primitive lands.");
}

// --- Nanmedian (differentiable: median backward with NaN positions zeroed)
auto NanmedianBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("NanmedianBackward::forward should not be called directly");
}
auto NanmedianBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "NanmedianBackward: cannot compute gradient of nanmedian over "
        "empty tensor");

    auto input_shape_vec =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Zero NaN positions before tie matching so the median-value mask never
    // picks them up. Use a sentinel (output + 2 * tie_eps) for NaN entries.
    auto nan_mask = isnan(input);
    auto non_nan = logical_not(nan_mask).to(input.dtype());

    if (!dim_.has_value()) {
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_exp = expand(output_reshaped, input_shape_vec);

        auto mask = value_tie_mask(input, output_exp);
        // Exclude NaN positions from the mask (their value-comparison was
        // against an arbitrary expanded output and would be unreliable).
        mask = mul(mask, non_nan);

        auto tie_count = sum(mask);
        auto safe_tie = where(eq(tie_count, zeros({}, tie_count.dtype(), tie_count.device())),
                              ones({}, tie_count.dtype(), tie_count.device()),
                              tie_count);
        mask = div(mask, safe_tie);

        auto grad = grad_output;
        if (grad.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad = reshape(grad, ones_shape);
        }
        auto grad_exp = expand(grad, input_shape_vec);
        return {mul(grad_exp, mask)};
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += static_cast<int64_t>(input.shape().size());

        // nanmedian along a dim returns the values tensor with that dim
        // collapsed (keepdim=false by default; the project doesn't expose a
        // keepdim variant on this overload). Unsqueeze to broadcast.
        Tensor out = output;
        Tensor grad = grad_output;
        if (out.ndim() == input.ndim() - 1) {
            out = unsqueeze(out, dim);
        }
        if (grad.ndim() == input.ndim() - 1) {
            grad = unsqueeze(grad, dim);
        }
        auto out_exp = expand(out, input_shape_vec);
        auto grad_exp = expand(grad, input_shape_vec);

        auto mask = value_tie_mask(input, out_exp);
        mask = mul(mask, non_nan);
        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        // Guard against all-NaN rows (tie_count = 0) — route zero grad up.
        auto tie_count_exp = expand(tie_count, input_shape_vec);
        auto safe_tie = where(eq(tie_count_exp,
                                 zeros(input_shape_vec, tie_count_exp.dtype(),
                                       tie_count_exp.device())),
                              ones(input_shape_vec, tie_count_exp.dtype(),
                                   tie_count_exp.device()),
                              tie_count_exp);
        mask = div(mask, safe_tie);
        return {mul(grad_exp, mask)};
    }
}

// --- Trapezoid (differentiable: linear weights on each sample) -----------
//
// Backward broadcasts grad_out (a scalar along `dim`) and applies the
// trapezoidal weights:
//   * uniform dx: weight = dx for interior, dx/2 at the two endpoints
//   * x given:    weight_i = 0.5 * (x_{i+1} - x_{i-1}) for interior; endpoints
//                  are 0.5 * (x_1 - x_0) and 0.5 * (x_{N-1} - x_{N-2}).
auto TrapezoidBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("TrapezoidBackward::forward should not be called directly");
}
auto TrapezoidBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    require_saved_tensors(has_x_ ? 2 : 1);
    const auto& y = saved_tensors_[0];
    const auto& grad_out = grad_outputs[0];

    auto y_shape = std::vector<int64_t>(y.shape().begin(), y.shape().end());
    int64_t dim = dim_;
    if (dim < 0) dim += static_cast<int64_t>(y.ndim());
    int64_t N = y.shape()[dim];
    TENZOR_CHECK_SHAPE(N >= 2,
        "TrapezoidBackward: integration axis must have at least 2 samples");

    // Build per-sample weight along `dim`. Final shape = y.shape().
    Tensor weights;
    if (has_x_) {
        const auto& x = saved_tensors_[1];
        TENZOR_CHECK_SHAPE(x.shape()[dim] == N,
            "TrapezoidBackward: x and y must agree along the integration axis");
        // dx_i = x_{i+1} - x_i  (length N-1 along dim)
        auto x_left  = narrow(x, dim, 0,     N - 1);
        auto x_right = narrow(x, dim, 1,     N - 1);
        auto dx_full = sub(x_right, x_left);
        // Endpoint weights: 0.5 * dx_full[0] and 0.5 * dx_full[-1].
        // Interior weights: 0.5 * (dx_full[i-1] + dx_full[i]) for i=1..N-2.
        std::vector<Tensor> chunks;
        chunks.reserve(3);
        auto half = full({}, 0.5, y.dtype(), y.device());
        // First endpoint: 0.5 * dx_full[..., 0]
        auto w_first = mul(narrow(dx_full, dim, 0, 1), half);
        chunks.push_back(w_first);
        if (N > 2) {
            auto interior_l = narrow(dx_full, dim, 0,         N - 2);
            auto interior_r = narrow(dx_full, dim, 1,         N - 2);
            auto interior   = mul(add(interior_l, interior_r), half);
            chunks.push_back(interior);
        }
        // Last endpoint: 0.5 * dx_full[..., -1]
        auto w_last = mul(narrow(dx_full, dim, N - 2, 1), half);
        chunks.push_back(w_last);
        weights = cat(chunks, dim);
    } else {
        // Uniform dx: shape (1,...,N,...,1) broadcastable along dim.
        std::vector<int64_t> w_shape(y.ndim(), 1);
        w_shape[dim] = N;
        // Build [dx/2, dx, dx, ..., dx, dx/2] in y.dtype() on y.device().
        // Use ones * dx, then patch the endpoints by half-weighting via
        // a multiplicative mask.
        auto w_full = full(w_shape, dx_, y.dtype(), y.device());
        // mask = 1 everywhere except 0.5 at the two endpoints.
        // Build by concat([0.5], 1s of length N-2, [0.5]).
        auto half_t = full({}, 0.5, y.dtype(), y.device());
        std::vector<int64_t> endpoint_shape(y.ndim(), 1);
        endpoint_shape[dim] = 1;
        std::vector<int64_t> interior_shape(y.ndim(), 1);
        interior_shape[dim] = N - 2;
        auto half_t_shaped = expand(reshape(half_t, endpoint_shape), endpoint_shape);
        auto interior_ones = ones(interior_shape, y.dtype(), y.device());
        std::vector<Tensor> mask_chunks;
        mask_chunks.push_back(half_t_shaped);
        if (N > 2) {
            mask_chunks.push_back(interior_ones);
        }
        mask_chunks.push_back(half_t_shaped);
        auto mask = cat(mask_chunks, dim);
        weights = mul(w_full, mask);
    }

    // grad_out has y's shape with `dim` removed. Unsqueeze and expand to
    // y's shape so we can multiply by per-sample weights.
    Tensor grad = grad_out;
    if (grad.ndim() == y.ndim() - 1) {
        grad = unsqueeze(grad, dim);
    }
    auto grad_exp = expand(grad, y_shape);

    // Broadcast weights to y_shape and multiply.
    auto weights_exp = expand(weights, y_shape);
    auto grad_y = mul(grad_exp, weights_exp);

    if (has_x_) {
        // Non-diff wrt x (matches PyTorch's trapezoid backward).
        const auto& xs = saved_tensors_[1].shape();
        std::vector<int64_t> x_shape(xs.begin(), xs.end());
        return {grad_y, zeros(x_shape,
                              saved_tensors_[1].dtype(),
                              saved_tensors_[1].device())};
    }
    return {grad_y};
}

// --- CumulativeTrapezoid (differentiable) --------------------------------
//
// Output has dim's extent reduced from N to N-1, with
//   out[k] = sum_{i=0..k-1} 0.5 * (y[i] + y[i+1]) * dx_i
// d out[k] / d y[i] is:
//   * 0       if i > k
//   * 0.5*dx_{i-1}                 if i == k    (top edge)
//   * 0.5*(dx_{i-1} + dx_i)        if 1<=i<k   (interior)
//   * 0.5*dx_0                     if i == 0 and k >= 1
//
// Equivalently, define G_i = sum_{k=i..N-2} grad_out[k] (reverse cumsum).
// Then  grad_y[i] = 0.5 * (dx_{i-1} * G_i + dx_i * G_i) with the convention
// dx_{-1} = 0 (so y[0] only gets dx_0 * G_0) and dx_{N-1} = 0 (so y[N-1]
// only gets dx_{N-2} * G_{N-2}).
auto CumulativeTrapezoidBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CumulativeTrapezoidBackward::forward should not be called directly");
}
auto CumulativeTrapezoidBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    require_saved_tensors(has_x_ ? 2 : 1);
    const auto& y = saved_tensors_[0];
    const auto& grad_out = grad_outputs[0];

    auto y_shape = std::vector<int64_t>(y.shape().begin(), y.shape().end());
    int64_t dim = dim_;
    if (dim < 0) dim += static_cast<int64_t>(y.ndim());
    int64_t N = y.shape()[dim];
    TENZOR_CHECK_SHAPE(N >= 2,
        "CumulativeTrapezoidBackward: integration axis must have at least 2 samples");
    TENZOR_CHECK_SHAPE(grad_out.shape()[dim] == N - 1,
        "CumulativeTrapezoidBackward: grad_out's integration axis must have "
        "y's extent minus 1");

    // Reverse cumsum of grad_out along dim: G_i = sum_{k>=i} grad_out[k].
    // Implement via flip → cumsum → flip.
    auto g_flipped = flip(grad_out, {dim});
    auto g_cum = cumsum(g_flipped, dim);
    auto G = flip(g_cum, {dim});  // length N-1

    // Build dx of length N-1 along dim.
    Tensor dx;  // shape with `dim` extent = N-1
    if (has_x_) {
        const auto& x = saved_tensors_[1];
        TENZOR_CHECK_SHAPE(x.shape()[dim] == N,
            "CumulativeTrapezoidBackward: x and y must agree along the "
            "integration axis");
        auto x_left  = narrow(x, dim, 0, N - 1);
        auto x_right = narrow(x, dim, 1, N - 1);
        dx = sub(x_right, x_left);
    } else {
        std::vector<int64_t> dx_shape = y_shape;
        dx_shape[dim] = N - 1;
        dx = full(dx_shape, dx_, y.dtype(), y.device());
    }

    // dxG = dx * G along dim (length N-1).
    auto dxG = mul(dx, G);

    // Build per-sample grad_y of length N:
    //   y[0]   gets 0.5 * dxG[0]
    //   y[i]   for 1<=i<N-1 gets 0.5 * (dxG[i-1] + dxG[i])
    //   y[N-1] gets 0.5 * dxG[N-2]
    auto half = full({}, 0.5, y.dtype(), y.device());

    auto first   = mul(narrow(dxG, dim, 0,     1),    half);
    auto last    = mul(narrow(dxG, dim, N - 2, 1),    half);

    std::vector<Tensor> chunks;
    chunks.reserve(3);
    chunks.push_back(first);
    if (N > 2) {
        auto left_part  = narrow(dxG, dim, 0,     N - 2);
        auto right_part = narrow(dxG, dim, 1,     N - 2);
        auto interior   = mul(add(left_part, right_part), half);
        chunks.push_back(interior);
    }
    chunks.push_back(last);
    auto grad_y = cat(chunks, dim);

    if (has_x_) {
        const auto& xs = saved_tensors_[1].shape();
        std::vector<int64_t> x_shape(xs.begin(), xs.end());
        return {grad_y, zeros(x_shape,
                              saved_tensors_[1].dtype(),
                              saved_tensors_[1].device())};
    }
    return {grad_y};
}

// --- SegmentReduce (non-diff) --------------------------------------------
auto SegmentReduceBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("SegmentReduceBackward::forward should not be called directly");
}
auto SegmentReduceBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "segment_reduce: backward is well-defined for sum/mean only. For "
        "max/min it requires per-segment argmax/argmin indices that the "
        "kernel does not currently return, and for prod a numerically-safe "
        "chain that handles zeros. Implementing only sum/mean would "
        "silently break callers using the other modes. Marked "
        "NonDifferentiable (project policy: fail loudly) until the kernel "
        "returns the necessary index/state.");
}

// --- GumbelSoftmax — audit-11 QQ.4: PyTorch-parity straight-through estimator.
// Forward saves y_soft = softmax((logits + Gumbel)/tau). The gradient through
// the soft sample equals the softmax Jacobian at y_soft, scaled by 1/tau:
//   grad_logits = (y_soft * (grad_out - sum(grad_out * y_soft, dim, keepdim=true))) / tau
// The `hard` quantisation is bypassed (STE: gradient is taken w.r.t. y_soft).
auto GumbelSoftmaxBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("GumbelSoftmaxBackward::forward should not be called directly");
}
auto GumbelSoftmaxBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& y_soft = saved_tensors_[0];
    const auto& grad_out = grad_outputs[0];

    int64_t ndim = static_cast<int64_t>(y_soft.ndim());
    int64_t d = dim_ < 0 ? dim_ + ndim : dim_;

    // PyTorch STE: grad_logits = y_soft * (grad_out - (grad_out * y_soft).sum(d, keepdim))/tau
    auto prod = mul(grad_out, y_soft);
    auto sum_prod = sum(prod, d, /*keepdim=*/true);
    auto sum_b = expand(sum_prod, std::vector<int64_t>(y_soft.shape().begin(), y_soft.shape().end()));
    auto diff = sub(grad_out, sum_b);
    auto j = mul(y_soft, diff);
    auto grad_logits = mul(j, 1.0 / tau_);
    return {grad_logits};
}
auto GumbelSoftmaxBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    // Variable-level rebuild for higher-order parity (audit II.2).
    Variable y_soft_v;
    if (has_saved_variables() && !saved_variables_.empty()) {
        y_soft_v = saved_variables_[0];
    } else {
        require_saved_tensors(1);
        y_soft_v = Variable(saved_tensors_[0], false);
    }
    int64_t ndim = static_cast<int64_t>(y_soft_v.tensor().ndim());
    int64_t d = dim_ < 0 ? dim_ + ndim : dim_;
    const auto& grad_var = grad_outputs[0];

    std::vector<int64_t> shape_vec(y_soft_v.tensor().shape().begin(),
                                   y_soft_v.tensor().shape().end());

    auto prod = grad_var * y_soft_v;
    auto sum_prod = tenzor::sum(prod, d, /*keepdim=*/true);
    auto sum_b = tenzor::expand(sum_prod, shape_vec);
    auto diff = grad_var - sum_b;
    auto j = y_soft_v * diff;
    auto grad_logits = j * (1.0 / tau_);
    return {grad_logits};
}

// --- CumMax (differentiable: scatter_add of grad_values along dim using
//             saved indices) ----------------------------------------------
auto CumMaxBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CumMaxBackward::forward should not be called directly");
}
auto CumMaxBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    // saved: input_shape (encoded as a 1-D Int64 placeholder, see wrapper),
    //        indices (Int64, same shape as input).
    require_saved_tensors(2);
    const auto& shape_holder = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];
    const auto& grad_out = grad_outputs[0];

    auto input_shape = std::vector<int64_t>(
        shape_holder.shape().begin(), shape_holder.shape().end());
    int64_t dim = dim_;
    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());

    auto zero = zeros(input_shape, grad_out.dtype(), grad_out.device());
    return {scatter_add(zero, dim, indices, grad_out)};
}
// audit-11 QQ.5: provide higher-order path via Variable-level scatter_add.
auto CumMaxBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    require_saved_tensors(2);
    const auto& shape_holder = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];  // Int64, non-diff
    auto input_shape = std::vector<int64_t>(
        shape_holder.shape().begin(), shape_holder.shape().end());
    int64_t dim = dim_;
    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());
    const auto& grad_var = grad_outputs[0];
    auto zero = zeros(input_shape, grad_var.tensor().dtype(), grad_var.tensor().device());
    Variable zero_var(zero, false);
    return {tenzor::scatter_add(zero_var, dim, indices, grad_var)};
}

// --- CumMin (differentiable) ---------------------------------------------
auto CumMinBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("CumMinBackward::forward should not be called directly");
}
auto CumMinBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& shape_holder = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];
    const auto& grad_out = grad_outputs[0];

    auto input_shape = std::vector<int64_t>(
        shape_holder.shape().begin(), shape_holder.shape().end());
    int64_t dim = dim_;
    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());

    auto zero = zeros(input_shape, grad_out.dtype(), grad_out.device());
    return {scatter_add(zero, dim, indices, grad_out)};
}
// audit-11 QQ.5: higher-order path mirrors CumMaxBackward.
auto CumMinBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    require_saved_tensors(2);
    const auto& shape_holder = saved_tensors_[0];
    const auto& indices = saved_tensors_[1];
    auto input_shape = std::vector<int64_t>(
        shape_holder.shape().begin(), shape_holder.shape().end());
    int64_t dim = dim_;
    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());
    const auto& grad_var = grad_outputs[0];
    auto zero = zeros(input_shape, grad_var.tensor().dtype(), grad_var.tensor().device());
    Variable zero_var(zero, false);
    return {tenzor::scatter_add(zero_var, dim, indices, grad_var)};
}

// =========================================================================
// Audit E.7 batch 9 — scatter reductions, audio/vision composites, integer
// binary ops.
// =========================================================================

// --- ScatterReduce (diff for sum/mean; non-diff for amax/amin/prod) -----
//
// Forward saves [index] and the reduce string in reduce_/include_self_.
// For sum: grad_input is grad_out itself (or zero at scattered positions
// when include_self=false); grad_src is gather(grad_out, dim, index).
// For mean: each scattered position averages (count) writes plus the
// pre-existing value (if include_self=true), so grad_src is divided by
// the same count tensor. We rebuild that count by scattering ones.
auto ScatterReduceBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error(
        "ScatterReduceBackward::forward should not be called directly");
}
auto ScatterReduceBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    if (reduce_ == "amax" || reduce_ == "amin" || reduce_ == "prod") {
        throw NonDifferentiable(
            "scatter_reduce: backward for reduce=\"" + reduce_ + "\" needs "
            "per-position argmax/argmin tie indices (for amax/amin) or a "
            "zero-safe product trick (for prod), neither of which the "
            "current kernel returns. Implementing only sum/mean while "
            "letting these slip through would silently zero out users' "
            "gradients. Marked NonDifferentiable (project policy: fail "
            "loudly) until the forward saves the required state.");
    }
    require_saved_tensors(1);
    const auto& index = saved_tensors_[0];
    const auto& grad_out = grad_outputs[0];

    int64_t dim = dim_;
    if (dim < 0) dim += static_cast<int64_t>(grad_out.shape().size());

    Tensor grad_input;
    Tensor grad_src;

    auto grad_out_shape = std::vector<int64_t>(grad_out.shape().begin(),
                                                grad_out.shape().end());
    auto index_shape = std::vector<int64_t>(index.shape().begin(),
                                             index.shape().end());

    if (reduce_ == "sum") {
        // grad_input flows through positions that include_self left
        // unchanged. With include_self=false, the kernel overwrites
        // those positions, so grad_input is zero there. Approximate by
        // returning grad_out for include_self=true (positions not hit
        // by index are unchanged anyway) and zero otherwise.
        grad_input = include_self_
            ? grad_out
            : zeros(grad_out_shape, grad_out.dtype(), grad_out.device());
        grad_src = gather(grad_out, dim, index);
    } else { // mean
        // For mean: y[i] = (include_self ? input[i] : 0 + sum_{j: idx[j]=i} src[j])
        //                  / (include_self ? 1 + count[i] : count[i]).
        // grad_src[j] = grad_out[index[j]] / denom[index[j]]
        // grad_input[i] = include_self ? grad_out[i] / denom[i] : 0
        auto ones_like_src = ones(index_shape, grad_out.dtype(),
                                  grad_out.device());
        auto base = include_self_
            ? ones(grad_out_shape, grad_out.dtype(), grad_out.device())
            : zeros(grad_out_shape, grad_out.dtype(), grad_out.device());
        auto count = scatter_add(base, dim, index, ones_like_src);
        // Guard against div-by-zero where include_self=false and a
        // position received no writes; those positions don't appear in
        // grad_input (it is zero) or grad_src (gather pulls only at
        // index[j] which by definition received a write).
        auto safe_count = clamp(count, /*min=*/1e-12,
                                 std::numeric_limits<double>::infinity());
        auto inv_denom = div(grad_out, safe_count);
        grad_input = include_self_
            ? inv_denom
            : zeros(grad_out_shape, grad_out.dtype(), grad_out.device());
        grad_src = gather(inv_denom, dim, index);
    }
    // grad layout: {input, dim (none), index (none), src}
    return {grad_input, grad_src};
}

// --- IndexReduce (diff for sum/mean; non-diff for amax/amin/prod) -------
//
// index_reduce is the 1-D-index sibling of scatter_reduce. The forward
// saves the index tensor; backward for sum is index_select(grad_out,
// dim, index), and for mean it's the same divided by per-target count.
auto IndexReduceBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error(
        "IndexReduceBackward::forward should not be called directly");
}
auto IndexReduceBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    if (reduce_ == "amax" || reduce_ == "amin" || reduce_ == "prod") {
        throw NonDifferentiable(
            "index_reduce: backward for reduce=\"" + reduce_ + "\" requires "
            "argmax/argmin tie indices (amax/amin) or a zero-safe product "
            "chain (prod); the kernel returns neither. Marked "
            "NonDifferentiable (project policy: fail loudly) until the "
            "forward saves the required state.");
    }
    require_saved_tensors(1);
    const auto& index = saved_tensors_[0];
    const auto& grad_out = grad_outputs[0];

    int64_t dim = dim_;
    if (dim < 0) dim += static_cast<int64_t>(grad_out.shape().size());

    Tensor grad_input;
    Tensor grad_src;

    auto grad_out_shape = std::vector<int64_t>(grad_out.shape().begin(),
                                                grad_out.shape().end());

    if (reduce_ == "sum") {
        grad_input = include_self_
            ? grad_out
            : zeros(grad_out_shape, grad_out.dtype(), grad_out.device());
        grad_src = index_select(grad_out, dim, index);
    } else { // mean
        // Build per-target write count along `dim` by scatter-adding
        // ones at index positions.
        std::vector<int64_t> ones_shape = grad_out_shape;
        ones_shape[dim] = index.shape()[0];
        auto ones_like_src = ones(ones_shape, grad_out.dtype(),
                                  grad_out.device());
        auto base = include_self_
            ? ones(grad_out_shape, grad_out.dtype(), grad_out.device())
            : zeros(grad_out_shape, grad_out.dtype(), grad_out.device());
        // index_add of ones along dim at `index` positions reproduces
        // the same count tensor scatter_reduce would build.
        auto count = index_add(base, dim, index, ones_like_src);
        auto safe_count = clamp(count, /*min=*/1e-12,
                                 std::numeric_limits<double>::infinity());
        auto inv_denom = div(grad_out, safe_count);
        grad_input = include_self_
            ? inv_denom
            : zeros(grad_out_shape, grad_out.dtype(), grad_out.device());
        grad_src = index_select(inv_denom, dim, index);
    }
    return {grad_input, grad_src};
}

// --- EmbeddingBag (diff in weight) ---------------------------------------
//
// Forward saved [indices, offsets, num_embeddings] in the wrapper. The
// backward kernel takes [grad_output, indices, offsets] + (Mode,
// PaddingIdx, NumEmbeddings) attrs and returns grad_weight.
auto EmbeddingBagBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error(
        "EmbeddingBagBackward::forward should not be called directly");
}
auto EmbeddingBagBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& indices = saved_tensors_[0];
    const auto& offsets = saved_tensors_[1];
    const auto& grad_out = grad_outputs[0];

    OpAttributes attrs;
    attrs.set(AttrKey::Mode, mode_);
    attrs.set(AttrKey::PaddingIdx, padding_idx_);
    attrs.set(AttrKey::NumEmbeddings, num_embeddings_);

    std::vector<Tensor> inputs = {grad_out, indices, offsets};
    auto result = dispatch_to_device(OpId::EmbeddingBagBackward,
        grad_out.device().type, inputs, attrs);
    // EmbeddingBag has inputs (weight, indices, offsets[, per_sample_weights]);
    // only weight is differentiable, so return its grad in slot 0.
    return {result[0]};
}

// --- ROIAlign (S15: real backward) ---------------------------------------
//
// Forward: y = roi_align_forward(features, rois, output_h, output_w,
//                                 spatial_scale, sampling_ratio, aligned).
// Backward: grad_features = roi_align_backward(grad_out, rois,
//                              batch_size=features.shape()[0],
//                              feat_height=features.shape()[2],
//                              feat_width=features.shape()[3],
//                              spatial_scale, sampling_ratio, aligned)
//           grad_rois = zero (rois are integer-quantized box coordinates;
//                       per project policy non-differentiable through this op).
auto ROIAlignBackward::forward(std::vector<Variable> inputs)
    -> std::vector<Variable> {
    if (!configured_) {
        throw std::runtime_error(
            "ROIAlignBackward::forward: constructed without configuration; "
            "use the parameterised constructor.");
    }
    if (inputs.size() != 2) {
        throw std::runtime_error(
            "ROIAlignBackward::forward: expected 2 inputs (features, rois)");
    }
    OpAttributes attrs;
    attrs.set(AttrKey::OutputSizeH, output_h_);
    attrs.set(AttrKey::OutputSizeW, output_w_);
    attrs.set(AttrKey::SpatialScale, spatial_scale_);
    attrs.set(AttrKey::SamplingRatio, sampling_ratio_);
    attrs.set(AttrKey::Aligned, aligned_);
    auto results = tenzor::dispatch(OpId::ROIAlignForward,
        std::vector<Tensor>{inputs[0].tensor(), inputs[1].tensor()}, attrs);
    // Save (features, rois) so backward can recover shape + ROI coords.
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});
    bool needs_grad = inputs[0].requires_grad();
    return {Variable(results[0], needs_grad)};
}
auto ROIAlignBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    if (!configured_) {
        throw NonDifferentiable(
            "ROIAlignBackward::backward: instance constructed without config.");
    }
    const auto& saved = saved_tensors();
    if (saved.size() < 2) {
        throw std::runtime_error(
            "ROIAlignBackward::backward: missing saved (features, rois)");
    }
    const Tensor& features = saved[0];
    const Tensor& rois     = saved[1];
    OpAttributes attrs;
    attrs.set(AttrKey::BatchSize, features.shape()[0]);
    attrs.set(AttrKey::FeatHeight, features.shape()[2]);
    attrs.set(AttrKey::FeatWidth, features.shape()[3]);
    attrs.set(AttrKey::SpatialScale, spatial_scale_);
    attrs.set(AttrKey::SamplingRatio, sampling_ratio_);
    attrs.set(AttrKey::Aligned, aligned_);
    auto results = tenzor::dispatch(OpId::ROIAlignBackward,
        std::vector<Tensor>{grad_outputs[0], rois}, attrs);
    // Return grad for features only; rois are non-differentiable.
    Tensor grad_rois = tenzor::zeros(
        std::vector<int64_t>(rois.shape().begin(), rois.shape().end()),
        rois.dtype(), rois.device());
    return {results[0], std::move(grad_rois)};
}

// --- DeformableConv2d (S15: real backward) -------------------------------
//
// Forward: y = deformable_conv2d_forward(input, offset, weight, mask?, bias?)
// Backward (chain rule across three split kernels):
//   {dinput, doffset, dmask} = DeformableConv2dBackwardInput(grad_out, input,
//                                  offset, weight, mask?)
//   dweight = DeformableConv2dBackwardWeight(grad_out, input, offset, mask?)
//   dbias   = DeformableConv2dBackwardBias(grad_out)  (sum over (N,H,W))
auto DeformableConv2dBackward::forward(std::vector<Variable> inputs)
    -> std::vector<Variable> {
    if (!configured_) {
        throw std::runtime_error(
            "DeformableConv2dBackward::forward: constructed without config.");
    }
    // Inputs (positional): input, offset, weight, [bias], [mask] — bias before
    // mask, matching the kernel's {input,offset,weight,bias,mask} layout and
    // ops::deformable_conv2d. bias present iff has_bias_, mask iff use_mask_.
    size_t expected_n = 3 + static_cast<size_t>(has_bias_) + static_cast<size_t>(use_mask_);
    if (inputs.size() != expected_n) {
        throw std::runtime_error(
            "DeformableConv2dBackward::forward: input count mismatch");
    }
    // Normalise to the canonical 5-tensor dispatch/save layout. Absent
    // bias/mask become valid 0-element tensors (numel()==0), which the kernel
    // treats as "not present" — this is also what ops::deformable_conv2d does,
    // and it prevents the dispatch lambda from indexing past the input span.
    const Tensor& d_input  = inputs[0].tensor();
    const Tensor& d_offset = inputs[1].tensor();
    const Tensor& d_weight = inputs[2].tensor();
    Tensor d_empty = tenzor::zeros({0}, d_input.dtype(), d_input.device());
    size_t didx = 3;
    Tensor d_bias = has_bias_ ? inputs[didx++].tensor() : d_empty;
    Tensor d_mask = use_mask_ ? inputs[didx++].tensor() : d_empty;
    std::vector<Tensor> ts = {d_input, d_offset, d_weight, d_bias, d_mask};
    OpAttributes attrs;
    attrs.set(AttrKey::StrideH, stride_h_);
    attrs.set(AttrKey::StrideW, stride_w_);
    attrs.set(AttrKey::PaddingH, pad_h_);
    attrs.set(AttrKey::PaddingW, pad_w_);
    attrs.set(AttrKey::DilationH, dilation_h_);
    attrs.set(AttrKey::DilationW, dilation_w_);
    attrs.set(AttrKey::Groups, groups_);
    attrs.set(AttrKey::OffsetGroups, offset_groups_);
    attrs.set(AttrKey::UseMask, use_mask_);
    auto out = tenzor::dispatch(OpId::DeformableConv2dForward, ts, attrs)[0];
    save_for_backward(ts);
    bool any_grad = false;
    for (auto& v : inputs) any_grad = any_grad || v.requires_grad();
    return {Variable(out, any_grad)};
}
auto DeformableConv2dBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    if (!configured_) {
        throw NonDifferentiable(
            "DeformableConv2dBackward::backward: instance constructed without config.");
    }
    const auto& saved = saved_tensors();
    // Saved is the canonical 5-tensor layout {input, offset, weight, bias, mask}
    // produced by forward() (bias/mask are 0-element tensors when absent).
    if (saved.size() < 5) {
        throw std::runtime_error(
            "DeformableConv2dBackward::backward: missing saved tensors");
    }
    const Tensor& input  = saved[0];
    const Tensor& offset = saved[1];
    const Tensor& weight = saved[2];
    Tensor mask = use_mask_ ? saved[4] : Tensor();

    OpAttributes attrs;
    attrs.set(AttrKey::StrideH, stride_h_);
    attrs.set(AttrKey::StrideW, stride_w_);
    attrs.set(AttrKey::PaddingH, pad_h_);
    attrs.set(AttrKey::PaddingW, pad_w_);
    attrs.set(AttrKey::DilationH, dilation_h_);
    attrs.set(AttrKey::DilationW, dilation_w_);
    attrs.set(AttrKey::Groups, groups_);
    attrs.set(AttrKey::OffsetGroups, offset_groups_);
    attrs.set(AttrKey::UseMask, use_mask_);
    // deformable_conv2d_backward_weight_kernel reads the weight shape from the
    // WeightShape attr (comma-separated string, parsed via get_int_list). The
    // NN-layer path (src/nn/layers/conv.cpp) sets this; the autograd Function
    // must too, or the kernel reads an empty shape vector and indexes OOB.
    auto shape_to_csv = [](std::span<const int64_t> s) {
        std::string out;
        for (size_t i = 0; i < s.size(); ++i) {
            if (i) out += ',';
            out += std::to_string(s[i]);
        }
        return out;
    };
    attrs.set(AttrKey::InputShape, shape_to_csv(input.shape()));
    attrs.set(AttrKey::WeightShape, shape_to_csv(weight.shape()));

    // Input/offset/mask backward (multi-output).
    std::vector<Tensor> bi_inputs = {grad_outputs[0], input, offset, weight};
    if (use_mask_) bi_inputs.push_back(mask);
    auto bi_out = tenzor::dispatch(OpId::DeformableConv2dBackwardInput,
        bi_inputs, attrs);
    Tensor dinput  = bi_out[0];
    Tensor doffset = bi_out.size() > 1 ? bi_out[1] :
        tenzor::zeros(std::vector<int64_t>(offset.shape().begin(),
                                            offset.shape().end()),
                      offset.dtype(), offset.device());
    Tensor dmask = use_mask_ && bi_out.size() > 2 ? bi_out[2] :
        (use_mask_ ? tenzor::zeros(std::vector<int64_t>(mask.shape().begin(),
                                                          mask.shape().end()),
                                    mask.dtype(), mask.device()) : Tensor());

    // Weight backward.
    std::vector<Tensor> bw_inputs = {grad_outputs[0], input, offset};
    if (use_mask_) bw_inputs.push_back(mask);
    auto dweight = tenzor::dispatch(OpId::DeformableConv2dBackwardWeight,
        bw_inputs, attrs)[0];

    // Bias backward (only if applicable).
    Tensor dbias;
    if (has_bias_) {
        auto db_out = tenzor::dispatch(OpId::DeformableConv2dBackwardBias,
            std::vector<Tensor>{grad_outputs[0]}, attrs);
        dbias = db_out[0];
    }

    // Pack outputs in the input order.
    std::vector<Tensor> grads;
    grads.push_back(std::move(dinput));
    grads.push_back(std::move(doffset));
    grads.push_back(std::move(dweight));
    if (use_mask_) grads.push_back(std::move(dmask));
    if (has_bias_) grads.push_back(std::move(dbias));
    return grads;
}

// --- MelScale (S15: real backward) ---------------------------------------
//
// Forward: y = filterbank @ spectrogram  (linear in spectrogram).
// Backward: dspectrogram = filterbank^T @ grad_y.  We reconstruct the
// filterbank by applying mel_scale to identity matrix probes (cheap; small
// constant matrix), then transpose-matmul.
auto MelScaleBackward::forward(std::vector<Variable> inputs)
    -> std::vector<Variable> {
    if (inputs.size() != 1) {
        throw std::runtime_error("MelScaleBackward::forward: expected 1 input");
    }
    OpAttributes attrs;
    attrs.set(AttrKey::NumMels, n_mels_);
    attrs.set(AttrKey::FMin, f_min_);
    attrs.set(AttrKey::FMax, f_max_);
    attrs.set(AttrKey::SampleRate, sample_rate_);
    auto y = tenzor::dispatch(OpId::MelScale,
        std::vector<Tensor>{inputs[0].tensor()}, attrs)[0];
    save_for_backward({inputs[0].tensor()});
    return {Variable(y, inputs[0].requires_grad())};
}
auto MelScaleBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    const auto& saved = saved_tensors();
    if (saved.empty()) {
        throw std::runtime_error("MelScaleBackward::backward: no saved tensor");
    }
    const Tensor& spec = saved[0];
    // Spectrogram shape: (..., n_freqs, time). y shape: (..., n_mels, time).
    // Reconstruct M (n_mels x n_freqs) by running mel_scale on an identity
    // spectrogram (n_freqs, 1) — this gives a column of M as the output for
    // each row. Cheaper: run mel_scale on an n_freqs x n_freqs identity in
    // place of spectrogram → y is M @ I = M, shape (n_mels, n_freqs).
    int64_t ndim = spec.ndim();
    int64_t n_freqs = spec.shape()[ndim - 2];
    auto eye = tenzor::eye(n_freqs, std::nullopt, spec.dtype(), spec.device());
    OpAttributes attrs;
    attrs.set(AttrKey::NumMels, n_mels_);
    attrs.set(AttrKey::FMin, f_min_);
    attrs.set(AttrKey::FMax, f_max_);
    attrs.set(AttrKey::SampleRate, sample_rate_);
    auto M = tenzor::dispatch(OpId::MelScale,
        std::vector<Tensor>{eye}, attrs)[0];
    // M is shape (n_mels, n_freqs). dspectrogram = M^T @ grad_y.
    auto Mt = tenzor::transpose(M, 0, 1);
    auto dspec = tenzor::matmul(Mt, grad_outputs[0]);
    return {dspec};
}

// --- DCT (S15: real backward) --------------------------------------------
//
// DCT is linear; the adjoint is a related DCT type:
//   DCT-II  <-> DCT-III   (transpose of basis)
//   DCT-I   self-adjoint
//   DCT-IV  self-adjoint
// For the "ortho" normalisation DCT is unitary, so its adjoint equals its
// inverse: idct(grad_y) with the same type/norm. For "backward"/"forward"
// norms the inverse equals the adjoint up to a scaling; this implementation
// uses idct() which the public fft API matches to the same type+norm pair,
// so dx = idct(grad_y) is the correct adjoint up to scaling that idct
// already accounts for.
auto DCTBackward::forward(std::vector<Variable> inputs)
    -> std::vector<Variable> {
    if (inputs.size() != 1) {
        throw std::runtime_error("DCTBackward::forward: expected 1 input");
    }
    OpAttributes attrs;
    attrs.set(AttrKey::DCTType, static_cast<int64_t>(type_));
    attrs.set(AttrKey::Dim, dim_);
    attrs.set(AttrKey::Norm, norm_);
    if (n_ > 0) attrs.set(AttrKey::N, n_);
    auto y = tenzor::dispatch(OpId::DCT,
        std::vector<Tensor>{inputs[0].tensor()}, attrs)[0];
    save_for_backward({inputs[0].tensor()});
    return {Variable(y, inputs[0].requires_grad())};
}
auto DCTBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    const auto& saved = saved_tensors();
    // Use the input shape's `dim` size as the natural n for idct.
    int64_t n_val = n_;
    if (n_val <= 0 && !saved.empty()) {
        int64_t actual_dim = (dim_ < 0) ? (dim_ + saved[0].ndim()) : dim_;
        n_val = saved[0].shape()[actual_dim];
    }
    OpAttributes attrs;
    attrs.set(AttrKey::DCTType, static_cast<int64_t>(type_));
    attrs.set(AttrKey::Dim, dim_);
    attrs.set(AttrKey::Norm, norm_);
    if (n_val > 0) attrs.set(AttrKey::N, n_val);
    auto dx = tenzor::dispatch(OpId::IDCT,
        std::vector<Tensor>{grad_outputs[0]}, attrs)[0];
    return {dx};
}

// --- MFCC (S15: real backward) -------------------------------------------
//
// MFCC = slice( DCT-II( log( MelScale( |STFT(x)|^2 ) ) ) ).
// Backward is the chain through:
//   d_log_mel  = pad_with_zeros(grad_y, mel_dim, n_mels - n_mfcc) , then
//                IDCT-II "ortho" (since MFCC uses ortho DCT)
//   d_mel_spec = d_log_mel / (mel_spec + 1e-10)
//   d_power    = M^T @ d_mel_spec   (mel-scale adjoint)
//   d_re       = 2 * re * d_power; d_im = 2 * im * d_power
//   d_waveform = ISTFT( complex(d_re, d_im) )
auto MFCCBackward::forward(std::vector<Variable> inputs)
    -> std::vector<Variable> {
    if (!configured_) {
        throw std::runtime_error(
            "MFCCBackward::forward: constructed without configuration");
    }
    if (inputs.size() != 1) {
        throw std::runtime_error("MFCCBackward::forward: expected 1 input");
    }
    const Tensor& waveform = inputs[0].tensor();
    // Replay the forward step-by-step and save intermediates.
    auto stft_result = tenzor::fft::stft(waveform, n_fft_, hop_length_, -1, Tensor());
    auto re = tenzor::real(stft_result);
    auto im = tenzor::imag(stft_result);
    auto power_spec = tenzor::add(tenzor::mul(re, re), tenzor::mul(im, im));
    OpAttributes mel_attrs;
    mel_attrs.set(AttrKey::NumMels, n_mels_);
    mel_attrs.set(AttrKey::FMin, f_min_);
    mel_attrs.set(AttrKey::FMax, f_max_);
    mel_attrs.set(AttrKey::SampleRate, sample_rate_);
    auto mel_spec = tenzor::dispatch(OpId::MelScale,
        std::vector<Tensor>{power_spec}, mel_attrs)[0];
    auto shifted = tenzor::add(mel_spec, 1e-10);
    auto log_mel = tenzor::log(shifted);
    int64_t mel_dim = log_mel.ndim() - 2;
    OpAttributes dct_attrs;
    dct_attrs.set(AttrKey::DCTType, int64_t{2});
    dct_attrs.set(AttrKey::Dim, mel_dim);
    dct_attrs.set(AttrKey::Norm, std::string("ortho"));
    auto dct_full = tenzor::dispatch(OpId::DCT,
        std::vector<Tensor>{log_mel}, dct_attrs)[0];
    auto y = tenzor::slice(dct_full, dct_full.ndim() - 2, 0, n_mfcc_);

    save_for_backward({waveform, re, im, mel_spec, log_mel});
    return {Variable(y, inputs[0].requires_grad())};
}
auto MFCCBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    if (!configured_) {
        throw NonDifferentiable(
            "MFCCBackward::backward: instance constructed without config.");
    }
    const auto& saved = saved_tensors();
    if (saved.size() < 5) {
        throw std::runtime_error("MFCCBackward::backward: missing saved tensors");
    }
    const Tensor& waveform = saved[0];
    const Tensor& re       = saved[1];
    const Tensor& im       = saved[2];
    const Tensor& mel_spec = saved[3];
    const Tensor& log_mel  = saved[4];

    const Tensor& grad_y = grad_outputs[0];
    // Pad grad_y with zeros along mel_dim from n_mfcc to n_mels.
    int64_t mel_dim = log_mel.ndim() - 2;
    auto pad_shape_vec = std::vector<int64_t>(grad_y.shape().begin(),
                                              grad_y.shape().end());
    int64_t cur = pad_shape_vec[mel_dim];
    Tensor d_dct;
    if (cur < n_mels_) {
        pad_shape_vec[mel_dim] = n_mels_ - cur;
        auto pad = tenzor::zeros(pad_shape_vec, grad_y.dtype(), grad_y.device());
        d_dct = tenzor::cat({grad_y, pad}, mel_dim);
    } else {
        d_dct = grad_y;
    }

    // IDCT-II ortho is the adjoint.
    OpAttributes dct_attrs;
    dct_attrs.set(AttrKey::DCTType, int64_t{2});
    dct_attrs.set(AttrKey::Dim, mel_dim);
    dct_attrs.set(AttrKey::Norm, std::string("ortho"));
    auto d_log_mel = tenzor::dispatch(OpId::IDCT,
        std::vector<Tensor>{d_dct}, dct_attrs)[0];

    // log: d_mel_spec = d_log_mel / (mel_spec + 1e-10)
    auto shifted = tenzor::add(mel_spec, 1e-10);
    auto d_mel_spec = tenzor::div(d_log_mel, shifted);

    // MelScale adjoint: M^T @ d_mel_spec. Reconstruct M via identity probe.
    int64_t ndim = mel_spec.ndim();
    int64_t n_freqs = (ndim >= 2) ? mel_spec.shape()[ndim - 2] : 1;
    // Actually mel_spec has the mel dim of size n_mels at ndim-2.
    // We need n_freqs from power_spec; saved re has shape matching the
    // pre-mel layout. We use re.shape()[re.ndim() - 2].
    int64_t freqs_dim = re.ndim() - 2;
    n_freqs = re.shape()[freqs_dim];
    auto eye = tenzor::eye(n_freqs, std::nullopt, re.dtype(), re.device());
    OpAttributes mel_attrs;
    mel_attrs.set(AttrKey::NumMels, n_mels_);
    mel_attrs.set(AttrKey::FMin, f_min_);
    mel_attrs.set(AttrKey::FMax, f_max_);
    mel_attrs.set(AttrKey::SampleRate, sample_rate_);
    auto M = tenzor::dispatch(OpId::MelScale,
        std::vector<Tensor>{eye}, mel_attrs)[0];
    auto Mt = tenzor::transpose(M, 0, 1);
    auto d_power = tenzor::matmul(Mt, d_mel_spec);

    // |z|^2 = re^2 + im^2:  d_re = 2 re * d_power; d_im = 2 im * d_power.
    auto two = tenzor::full({}, 2.0, re.dtype(), re.device());
    auto d_re = tenzor::mul(tenzor::mul(re, d_power), two);
    auto d_im = tenzor::mul(tenzor::mul(im, d_power), two);
    auto d_complex = tenzor::complex(d_re, d_im);

    // ISTFT adjoint of STFT.
    auto d_wave = tenzor::fft::istft(d_complex, n_fft_, hop_length_,
                                     -1, Tensor(), true, false, true,
                                     std::optional<int64_t>(waveform.shape().back()));
    return {d_wave};
}

// --- Gcd / Lcm (integer-domain, Jacobian = 0 a.e.) -----------------------
auto GcdBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error(
        "GcdBackward::forward should not be called directly");
}
auto GcdBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "gcd: integer-valued binary operation. The output is constant on "
        "open neighbourhoods of integer inputs and undefined off the integer "
        "lattice, so the Jacobian is identically zero where defined. Marked "
        "NonDifferentiable to fail loudly instead of returning a deceptive "
        "zero gradient that would compose silently in user models.");
}

auto LcmBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error(
        "LcmBackward::forward should not be called directly");
}
auto LcmBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "lcm: integer-valued binary operation; same rationale as gcd "
        "(piecewise-constant on the integer lattice, undefined elsewhere). "
        "NonDifferentiable.");
}

// ============================================================================
// Audit E.7 batch 10 — RNN kernel-level forwards + small math/transform ops
// ============================================================================

// --- LSTMCellForward / GRUCellForward / LSTMForward / GRUForward /
//     LSTMMultiLayerForward / GRUMultiLayerForward / BiLSTMForward
//
// Kernel-level fused RNN forwards. The autograd-aware path is the
// nn::LSTMCell / nn::GRUCell / nn::LSTM / nn::GRU Modules which compose the
// computation graph from Variable-level Linear / sigmoid / tanh / slice ops
// (see src/nn/layers/lstm.cpp, gru.cpp, rnn.cpp). A Function-layer adjoint
// that re-dispatches the matching kernel-level backward and threads grads
// back to the right input Variables is not wired, so direct
// OpId::*CellForward / OpId::*Forward dispatch routed through autograd
// raises NonDifferentiable.

auto LSTMCellForwardBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "LSTMCellForwardBackward::forward should not be called directly");
}
auto LSTMCellForwardBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "lstm_cell_forward: fused kernel-level single-step LSTM cell. The "
        "autograd-aware path in this project is nn::LSTMCell, which builds "
        "the per-step computation graph from Variable-level Linear / "
        "sigmoid / tanh / slice / add / mul ops (see "
        "src/nn/layers/lstm.cpp). The kernel-level OpId::LSTMCellBackward "
        "exists in every backend's registry and is the right vehicle for "
        "gradients here; routing it through a Function-layer adjoint that "
        "threads grad_input / grad_hx / grad_cx / grad_W_ih / grad_W_hh / "
        "grad_b_ih / grad_b_hh back to the right input Variables is not "
        "wired. NonDifferentiable until a Function-layer wrapper is added; "
        "use nn::LSTMCell for autograd-aware LSTM cells.");
}

auto GRUCellForwardBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "GRUCellForwardBackward::forward should not be called directly");
}
auto GRUCellForwardBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "gru_cell_forward: fused kernel-level single-step GRU cell. The "
        "autograd-aware path is nn::GRUCell (src/nn/layers/gru.cpp) which "
        "composes the cell's three gates (r, z, n) from Variable-level "
        "Linear / sigmoid / tanh ops. The kernel-level "
        "OpId::GRUCellBackward exists and is the right vehicle for "
        "gradients here; routing it through a Function-layer adjoint is "
        "not wired. NonDifferentiable until that wrapper is added; use "
        "nn::GRUCell for autograd-aware GRU cells.");
}

auto LSTMForwardBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "LSTMForwardBackward::forward should not be called directly");
}
auto LSTMForwardBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "lstm_forward: fused kernel-level single-layer multi-step LSTM "
        "forward. The autograd-aware path is nn::LSTM, which unrolls per-"
        "timestep nn::LSTMCell forwards so each step's Variable graph "
        "composes into the sequence graph. Per-step intermediates (gate "
        "activations, cell state) are not exposed by the fused kernel, so "
        "a closed-form Function-layer adjoint cannot reconstruct the "
        "backward without re-running the unrolled forward. "
        "NonDifferentiable; use nn::LSTM for autograd-aware sequence LSTM.");
}

auto GRUForwardBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "GRUForwardBackward::forward should not be called directly");
}
auto GRUForwardBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "gru_forward: fused kernel-level single-layer multi-step GRU "
        "forward. Same rationale as lstm_forward — the per-step gate "
        "activations needed for the closed-form gate-derivative chain "
        "are not exposed by the fused kernel. NonDifferentiable; use "
        "nn::GRU for autograd-aware sequence GRU.");
}

auto LSTMMultiLayerForwardBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "LSTMMultiLayerForwardBackward::forward should not be called "
        "directly");
}
auto LSTMMultiLayerForwardBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "lstm_multi_layer_forward: stacked-layer fused LSTM forward. The "
        "autograd-aware path is nn::LSTM(num_layers > 1), which loops "
        "nn::LSTMCell forwards per layer per timestep so the graph carries "
        "gradients through every layer's Linear weights and per-step "
        "hidden/cell states. The fused kernel does not expose those "
        "intermediates, so a Function-layer adjoint cannot reconstruct "
        "the backward. NonDifferentiable; use nn::LSTM with num_layers > 1.");
}

auto GRUMultiLayerForwardBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "GRUMultiLayerForwardBackward::forward should not be called "
        "directly");
}
auto GRUMultiLayerForwardBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "gru_multi_layer_forward: stacked-layer fused GRU forward. Same "
        "rationale as lstm_multi_layer_forward — the per-step gate "
        "activations needed for the closed-form gate-derivative chain "
        "are not exposed by the fused kernel. NonDifferentiable; use "
        "nn::GRU with num_layers > 1.");
}

auto BiLSTMForwardBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "BiLSTMForwardBackward::forward should not be called directly");
}
auto BiLSTMForwardBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "bilstm_forward: single-layer bidirectional LSTM forward. The "
        "autograd-aware path is nn::LSTM(bidirectional=true), which wires "
        "forward and reverse nn::LSTMCell sweeps and concatenates the "
        "per-step hidden outputs along the feature dim. The fused kernel "
        "does not expose the per-direction per-step intermediates, so a "
        "Function-layer adjoint cannot reconstruct the backward. "
        "NonDifferentiable; use nn::LSTM(bidirectional=true).");
}

// --- nextafter — discrete float step (Jacobian zero a.e., undefined at
//     the representable-boundary jumps). NonDifferentiable.
auto NextafterBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "NextafterBackward::forward should not be called directly");
}
auto NextafterBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "nextafter: returns the next representable float after `from` in "
        "the direction of `to`. The output is piecewise-constant in `from` "
        "(jumping by one ULP at every representable boundary, where the "
        "derivative is undefined) and discrete in `to` (sign-only "
        "dependence). The Jacobian is identically zero where defined, so "
        "we fail loudly instead of returning the deceptive zero gradient "
        "that would silently compose in user models.");
}

// --- ldexp(x, n) = x * 2^n. Differentiable in x with grad_x = 2^n * grad_y
//     = ldexp(grad_y, n). The exponent `n` is integer-typed (an exact
//     bit-shift of the binary exponent field), so route a zero gradient
//     to it — same convention the project uses for integer-indexed inputs
//     elsewhere.
auto LdexpBackward::forward(std::vector<Variable> inputs)
    -> std::vector<Variable> {
    if (inputs.size() != 2) {
        throw std::runtime_error("LdexpBackward::forward expects 2 inputs");
    }
    save_for_backward({inputs[1].tensor()});  // save n
    input_shape_x_ = std::vector<int64_t>(inputs[0].shape().begin(),
                                          inputs[0].shape().end());
    input_shape_n_ = std::vector<int64_t>(inputs[1].shape().begin(),
                                          inputs[1].shape().end());
    auto result = tenzor::ldexp(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}
auto LdexpBackward::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    require_saved_tensors(1);
    const auto& n = saved_tensors_[0];
    const auto& grad = grad_outputs[0];

    // grad_x = grad * 2^n = ldexp(grad, n). ldexp() handles broadcasting
    // between grad and n the same way the forward did.
    auto grad_x_unreduced = tenzor::ldexp(grad, n);
    auto grad_x = reduce_grad_for_broadcasting(grad_x_unreduced,
                                                input_shape_x_);

    // n is integer-typed and non-differentiable; return zeros matching its
    // shape and dtype so the autograd engine can still accumulate without
    // dtype mismatches in downstream Variable arithmetic.
    auto grad_n = zeros(input_shape_n_, n.dtype(), n.device());
    return {grad_x, grad_n};
}

// --- idct — mirror of DCTBackward; linear in principle but the adjoint
//     depends on (type, norm, n) which the kernels do not export.
auto IDCTBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error(
        "IDCTBackward::forward should not be called directly");
}
auto IDCTBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "idct: inverse Discrete Cosine Transform. Linear and therefore "
        "differentiable in principle (the adjoint of IDCT-{II,III,IV} is "
        "its matching DCT under the same `norm`), but the kernels do not "
        "export a parameter-matched DCT helper to compose the adjoint "
        "from. Mirrors DCTBackward's rationale: getting the (type, norm, "
        "n) matrix wrong silently distorts gradients. NonDifferentiable "
        "until the type/norm/length adjoint matrix is implemented. Use "
        "norm=\"ortho\" plus an explicit DCT call if you need gradients "
        "through IDCT.");
}

// ============================================================================
// Audit E.7 batch 11 — final residual OpIds:
//   * Remainder, Fmod                       — discrete-jump binary mods
//   * NestedLogSoftmax, NestedToPadded,
//     NestedFromPadded, NestedLinear        — fused nested-tensor OpIds
//                                             whose autograd-aware path is
//                                             the Variable-level composition
//                                             in autograd::nested_* helpers
//                                             (see nested_autograd_ops.cpp)
//   * QuantizedLinear, QuantizedConv2d      — fused INT8/INT4 inference ops
//
// All eight raise `tenzor::NonDifferentiable` for direct OpId dispatch
// routed through autograd: the per-op headers above each class explain why
// no closed-form Function-layer adjoint is wired here. Callers that need
// gradients should compose the documented Variable-level path.
// ============================================================================

// --- Remainder (Python-style modulo) -------------------------------------
auto RemainderBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error(
        "RemainderBackward::forward should not be called directly");
}
auto RemainderBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "remainder: Python-style modulo a - floor(a/b) * b. The output is "
        "piecewise-linear in `a` with discrete jumps of size `b` at every "
        "multiple of `b`, so the Jacobian is undefined at the jumps and "
        "the piecewise-constant gradient is wrong any time `a/b` crosses "
        "an integer during training. Marked NonDifferentiable (project "
        "policy: fail loudly) instead of returning a deceptive adjoint at "
        "the discontinuity. Compose `a - b * floor(a / b)` explicitly with "
        "your own straight-through estimator if you need a smooth "
        "surrogate.");
}

// --- Fmod (C-style modulo) -----------------------------------------------
auto FmodBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error(
        "FmodBackward::forward should not be called directly");
}
auto FmodBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "fmod: C-style modulo a - trunc(a/b) * b. Same discrete-jump "
        "structure as remainder() with the discontinuity set on the "
        "lattice of `a/b` integer crossings (now straddling zero through "
        "trunc instead of floor). NonDifferentiable for the same reason; "
        "use a + b * round((-a) / b) or an explicit STE if you need a "
        "smooth surrogate.");
}

// --- NestedLogSoftmax ----------------------------------------------------
auto NestedLogSoftmaxBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "NestedLogSoftmaxBackward::forward should not be called directly");
}
auto NestedLogSoftmaxBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "nested_log_softmax: segmented log-softmax along the ragged dim "
        "of a packed nested tensor. The autograd-aware path is the "
        "explicit log(nested_softmax(values, offsets, dim)) composition, "
        "which goes through NestedSoftmaxBackward's per-segment dot-"
        "correction adjoint. A direct Function-layer adjoint for the "
        "fused OpId::NestedLogSoftmax would have to reconstruct the same "
        "per-segment grad sum across the offsets and widen Float16/"
        "BFloat16 inputs through the segment reduction the same way "
        "NestedSoftmaxBackward does; that wiring is not present. "
        "NonDifferentiable; use tenzor::log(autograd::nested_softmax(...)) "
        "for autograd-aware nested log-softmax.");
}

// --- NestedToPadded ------------------------------------------------------
auto NestedToPaddedBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "NestedToPaddedBackward::forward should not be called directly");
}
auto NestedToPaddedBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "nested_to_padded: densify a packed nested tensor into a padded "
        "[B, max_len, ...] dense tensor. The forward is a per-segment "
        "scatter into padded slots; the adjoint is the inverse gather "
        "(packed values <- unpadded slots), which requires {offsets, "
        "max_len, per-batch lengths} from the forward — the forward "
        "currently does not save those. The autograd-aware path is the "
        "Variable-level scatter/gather composition in nn::NestedSequence "
        "(or keep the computation in packed-values form and use the "
        "per-op nested autograd Functions). NonDifferentiable until the "
        "forward saves the {offsets, max_len, batch_lengths} triple.");
}

// --- NestedFromPadded ----------------------------------------------------
auto NestedFromPaddedBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "NestedFromPaddedBackward::forward should not be called directly");
}
auto NestedFromPaddedBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "nested_from_padded: inverse of nested_to_padded — gather the "
        "unpadded slots of a [B, max_len, ...] dense tensor into packed "
        "nested values. The adjoint is the matching scatter (same kernel "
        "as nested_to_padded), but routing the gradient back requires "
        "the per-batch lengths and pad slots the forward saw, which are "
        "not saved here. Same rationale as NestedToPaddedBackward; "
        "NonDifferentiable until the forward saves the {offsets, "
        "max_len, batch_lengths} triple.");
}

// --- NestedLinear --------------------------------------------------------
auto NestedLinearBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "NestedLinearBackward::forward should not be called directly");
}
auto NestedLinearBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "nested_linear: Linear projection on the packed values of a "
        "nested tensor. The autograd-aware path is the Variable-level "
        "matmul + add composition in autograd::nested_linear (see "
        "src/autograd/nested_autograd_ops.cpp), which builds the graph "
        "from existing MatMul / Add Function nodes and carries gradients "
        "to `weight` and `bias` through the standard linear adjoint. A "
        "Function-layer adjoint for the fused OpId::NestedLinear is "
        "redundant when the packed values are layout-equivalent to a "
        "dense [N, in_features] matrix (the only layout the fused kernel "
        "supports). NonDifferentiable for direct OpId dispatch; use "
        "autograd::nested_linear for autograd-aware nested linear.");
}

// --- QuantizedLinear -----------------------------------------------------
auto QuantizedLinearBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "QuantizedLinearBackward::forward should not be called directly");
}
auto QuantizedLinearBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "quantized_linear: fused INT8/INT4 quantized inference linear "
        "kernel. The forward dequantizes the packed weight, performs the "
        "matmul in the accumulator dtype, requantizes, and clamps to the "
        "output dtype's representable range. Quantized inference operates "
        "on a frozen weight (and frozen scale / zero_point); the weight "
        "is integer-packed (no gradient w.r.t. an integer lattice) and "
        "the activation Jacobian goes through a clamp-round step which "
        "is piecewise-constant a.e. The quantization-aware-training (QAT) "
        "path is the Variable-level fake-quant composition in "
        "nn::FakeQuantize (x + StopGradient(quantize_dequantize(x) - x) "
        "STE), which threads gradients through the activation only. "
        "NonDifferentiable; use nn::FakeQuantize + nn::Linear for "
        "QAT-aware quantized linear.");
}

// --- QuantizedConv2d -----------------------------------------------------
auto QuantizedConv2dBackward::forward(std::vector<Variable>)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "QuantizedConv2dBackward::forward should not be called directly");
}
auto QuantizedConv2dBackward::backward(std::vector<Tensor> /*grad_outputs*/)
    -> std::vector<Tensor> {
    throw NonDifferentiable(
        "quantized_conv2d: fused INT8/INT4 quantized inference 2-D "
        "convolution. Same rationale as quantized_linear — integer-packed "
        "frozen weight + clamp/round requantization step => not "
        "differentiable through the fused kernel. The QAT path is "
        "nn::FakeQuantize + nn::Conv2d (with weight fake-quant applied at "
        "the parameter, not the input). NonDifferentiable; use the "
        "Variable-level composition for QAT-aware quantized conv.");
}

} // namespace tenzor
