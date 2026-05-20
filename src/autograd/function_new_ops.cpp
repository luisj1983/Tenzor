#include "tenzor/autograd/function.hpp"
#include "function_helpers.hpp"
#include <cassert>
#include <cstdlib>
#include <string>
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/advanced.hpp"
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
    // Use tensor-level computation for the Jacobian factor (constant)
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad()),
            Variable(result_tensors[1], grad_outputs[0].requires_grad())};
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
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];

    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto eps_t = full(input_shape, detail::dtype_epsilon(input.dtype()),
                      input.dtype(), input.device());
    auto safe_input = where(eq(input, zeros(input_shape, input.dtype(), input.device())),
                            eps_t, input);
    auto scale = div(output, safe_input);

    Variable scale_var(scale, false);
    return {grad_outputs[0] * scale_var};
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
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

// LinalgLDLFactorBackward: complex structured backprop
//
// Mathematical status: the closed-form backward through LDL = P L D L^T P^T
// (with pivoting from Bunch-Kaufman) is tractable but involves multiple
// triangular-projection operations; see Giles "Extended Matrix Backward"
// (2008) and Walter "Structured Matrix Differentiation" (2010). A correct
// implementation is multi-file work that needs its own dedicated test
// suite against SciPy/PyTorch references and is out of scope for this
// autograd completeness pass.
//
// Current behavior: returns zero gradient. Silent zeros are dangerous when
// users plug LDL into a loss that's downstream of other differentiable ops
// — they'll see "training proceeds" but gradients through the factorization
// are missing. Set `TENZOR_STRICT_LINALG_GRAD=1` to surface this as a
// runtime error instead, so CI catches accidental use of LDL in a gradient
// chain. Users needing gradients should use `ldl_solve` which has a proper
// backward, or compute through `cholesky` for symmetric positive definite
// matrices.
auto LinalgLDLFactorBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LinalgLDLFactorBackward::forward should not be called directly");
}

namespace {
inline bool strict_linalg_grad_mode() {
    const char* v = std::getenv("TENZOR_STRICT_LINALG_GRAD");
    return v && *v && std::string(v) != "0" && std::string(v) != "false";
}
}

auto LinalgLDLFactorBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& A = saved_tensors_[0];
    const auto& LD = saved_tensors_[1];
    const auto& grad_LD = grad_outputs[0];

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
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
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
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
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
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad()),
            Variable(result_tensors[1], grad_outputs[0].requires_grad())};
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

    auto pow_x = tenzor::pow(safe_abs, static_cast<float>(ord_ - 1.0));
    auto pow_norm = tenzor::pow(safe_norm, static_cast<float>(ord_ - 1.0));
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
    mask = mask.unsqueeze(col_norm ? ndim - 2 : ndim - 1);

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

} // namespace tenzor
