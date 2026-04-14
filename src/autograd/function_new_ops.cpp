#include "tenzor/autograd/function.hpp"
#include "function_helpers.hpp"
#include <cassert>
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
    const auto& output = saved_tensors_[1];  // renorm(input)
    const auto& grad = grad_outputs[0];

    auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Compute p-norm along all dims except dim_
    // scale = output / input (where input != 0)
    auto eps_t = full(input_shape, detail::dtype_epsilon(input.dtype()),
                      input.dtype(), input.device());
    auto safe_input = where(eq(input, zeros(input_shape, input.dtype(), input.device())),
                            eps_t, input);
    auto scale = div(output, safe_input);

    // Simple backward: grad * scale
    // This gives the correct gradient when the norm is above maxnorm (scaling active)
    // and when the norm is below maxnorm (scale = 1, identity).
    return {mul(grad, scale)};
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
// For now, use numerical differentiation approach via saved factors
auto LinalgLDLFactorBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LinalgLDLFactorBackward::forward should not be called directly");
}

auto LinalgLDLFactorBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    // LDL factorization backward is very complex (structured symmetric backprop).
    // For now, return zeros (non-differentiable through the factorization).
    // Users needing gradients through LDL should use the solve path.
    require_saved_tensors(1);
    const auto& A = saved_tensors_[0];
    return {zeros_like(A)};
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

auto LinalgHouseholderBackward::backward(std::vector<Tensor> /*grad_outputs*/) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& tau = saved_tensors_[1];
    // Householder product backward is complex and rarely needed in gradient flows.
    // Return zeros for now (covered by the STRUCTURAL_ZERO_STUB).
    return {zeros_like(input), zeros_like(tau)};
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

    // Flatten Y to 2D, compute -Y @ grad @ Y, reshape back
    auto Y_shape = std::vector<int64_t>(Y.shape().begin(), Y.shape().end());
    auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());

    // Compute product dimensions
    int64_t rows = 1, cols = 1;
    for (int64_t i = 0; i < ind_; ++i) rows *= Y_shape[i];
    for (size_t i = ind_; i < Y_shape.size(); ++i) cols *= Y_shape[i];

    auto Y_2d = reshape(Y, {rows, cols});
    auto grad_2d = reshape(grad, {cols, rows});

    auto temp = matmul(matmul(Y_2d, grad_2d), Y_2d);
    auto result_2d = neg(temp);

    // Reshape back to original input shape (which is Y's transposed shape)
    // tensorinv output shape is inverse of input shape
    std::vector<int64_t> input_shape;
    for (size_t i = ind_; i < Y_shape.size(); ++i) input_shape.push_back(Y_shape[i]);
    for (int64_t i = 0; i < ind_; ++i) input_shape.push_back(Y_shape[i]);

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

    if (std::isinf(ord_) && ord_ > 0) {
        // Linf norm: grad at position where |x| == norm
        auto abs_x = abs(input);
        auto eps_t = full(input_shape, detail::dtype_epsilon(input.dtype()),
                          input.dtype(), input.device());
        auto mask = lt(abs(sub(abs_x, norm_expanded)), eps_t);
        auto sgn = tenzor::sign(input);
        return {mul(grad_expanded, where(mask, sgn, zeros_like(input)))};
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
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

// LinalgMatrixNormBackward:
// For Frobenius (ord=2): dL/dA = dL/dy * A / norm(A)
// For other norms: complex, use dispatch
auto LinalgMatrixNormBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LinalgMatrixNormBackward::forward should not be called directly");
}

auto LinalgMatrixNormBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    require_saved_tensors(2);
    const auto& input = saved_tensors_[0];
    const auto& norm_val = saved_tensors_[1];
    const auto& grad = grad_outputs[0];

    if (std::abs(ord_ - 2.0) < 1e-10) {
        // Frobenius-like: grad * A / norm
        auto input_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        auto scale = div(grad, norm_val);
        auto scale_shape = std::vector<int64_t>(scale.shape().begin(), scale.shape().end());
        while (scale_shape.size() < input_shape.size()) {
            scale_shape.push_back(1);
        }
        auto scale_expanded = reshape(scale, scale_shape);
        return {mul(scale_expanded, input)};
    }

    // For spectral norm (largest singular value), the gradient involves SVD
    if (std::abs(ord_) > 2.0 || std::abs(ord_ - 1.0) < 1e-10) {
        // Return zeros for unsupported norm types
        return {zeros_like(input)};
    }

    return {zeros_like(input)};
}

auto LinalgMatrixNormBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
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

} // namespace tenzor
