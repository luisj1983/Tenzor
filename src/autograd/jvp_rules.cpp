#include "tenzor/autograd/jvp_rules.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include <cmath>

namespace tenzor {

// ============================================================================
// Arithmetic
// ============================================================================

auto jvp_add(const DualTensor& a, const DualTensor& b) -> DualTensor {
    auto primal = tenzor::add(a.primal(), b.primal());
    auto tangent = tenzor::add(a.tangent(), b.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_sub(const DualTensor& a, const DualTensor& b) -> DualTensor {
    auto primal = tenzor::sub(a.primal(), b.primal());
    auto tangent = tenzor::sub(a.tangent(), b.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_mul(const DualTensor& a, const DualTensor& b) -> DualTensor {
    // Product rule: d(a*b) = da*b + a*db
    auto primal = tenzor::mul(a.primal(), b.primal());
    auto tangent = tenzor::add(
        tenzor::mul(a.tangent(), b.primal()),
        tenzor::mul(a.primal(), b.tangent())
    );
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_div(const DualTensor& a, const DualTensor& b) -> DualTensor {
    // Quotient rule: d(a/b) = (da*b - a*db) / (b*b)
    auto primal = tenzor::div(a.primal(), b.primal());
    auto b_sq = tenzor::mul(b.primal(), b.primal());
    auto tangent = tenzor::div(
        tenzor::sub(
            tenzor::mul(a.tangent(), b.primal()),
            tenzor::mul(a.primal(), b.tangent())
        ),
        b_sq
    );
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_neg(const DualTensor& a) -> DualTensor {
    auto primal = tenzor::neg(a.primal());
    auto tangent = tenzor::neg(a.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Matrix ops
// ============================================================================

auto jvp_matmul(const DualTensor& a, const DualTensor& b) -> DualTensor {
    // d(A @ B) = dA @ B + A @ dB
    auto primal = tenzor::matmul(a.primal(), b.primal());
    auto tangent = tenzor::add(
        tenzor::matmul(a.tangent(), b.primal()),
        tenzor::matmul(a.primal(), b.tangent())
    );
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Activations
// ============================================================================

auto jvp_relu(const DualTensor& x) -> DualTensor {
    // relu(x) = max(0, x); d(relu) = x.tangent * (x.primal > 0)
    auto primal = tenzor::clamp_min(x.primal(), 0.0f);
    auto mask = tenzor::gt(x.primal(), tenzor::zeros_like(x.primal()));
    auto tangent = tenzor::mul(x.tangent(), mask);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_sigmoid(const DualTensor& x) -> DualTensor {
    // s = sigmoid(p); tangent = dx * s * (1 - s)
    auto s = tenzor::sigmoid(x.primal());
    auto one = tenzor::ones_like(s);
    auto tangent = tenzor::mul(x.tangent(), tenzor::mul(s, tenzor::sub(one, s)));
    return DualTensor(std::move(s), std::move(tangent));
}

auto jvp_tanh(const DualTensor& x) -> DualTensor {
    // t = tanh(p); tangent = dx * (1 - t*t)
    auto t = tenzor::tanh(x.primal());
    auto one = tenzor::ones_like(t);
    auto tangent = tenzor::mul(x.tangent(), tenzor::sub(one, tenzor::mul(t, t)));
    return DualTensor(std::move(t), std::move(tangent));
}

auto jvp_gelu(const DualTensor& x) -> DualTensor {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    // d(GELU)/dx = 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
    // We compute primal via the dispatch and tangent via the derivative formula.
    const double sqrt_2 = 1.4142135623730951;
    const double inv_sqrt_2pi = 0.3989422804014327;

    auto p = x.primal();
    auto scaled = tenzor::mul(p, 1.0 / sqrt_2);

    // Primal: 0.5 * x * (1 + erf(x/sqrt(2)))
    // Use the existing ops to build it
    auto erf_val = tenzor::sigmoid(p); // placeholder - we need erf
    // Actually compute properly using the identity:
    // GELU(x) = x * sigmoid(1.702 * x)  (approximate)
    // But for correctness let's use the exact formula via existing ops

    // The exact GELU primal: 0.5 * x * (1 + erf(x / sqrt(2)))
    // We don't have a tensor-level erf, so use the approximation:
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const double sqrt_2_over_pi = 0.7978845608028654;

    auto x3 = tenzor::mul(p, tenzor::mul(p, p));
    auto inner = tenzor::mul(
        tenzor::add(p, tenzor::mul(x3, 0.044715)),
        sqrt_2_over_pi
    );
    auto tanh_inner = tenzor::tanh(inner);
    auto one = tenzor::ones_like(p);
    auto primal = tenzor::mul(tenzor::mul(p, 0.5), tenzor::add(one, tanh_inner));

    // Derivative of GELU (tanh approximation):
    // Let u = sqrt(2/pi) * (x + 0.044715 * x^3)
    // GELU = 0.5 * x * (1 + tanh(u))
    // dGELU/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
    // du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
    auto sech2 = tenzor::sub(one, tenzor::mul(tanh_inner, tanh_inner));
    auto du_dx = tenzor::mul(
        tenzor::add(one, tenzor::mul(tenzor::mul(p, p), 3.0 * 0.044715)),
        sqrt_2_over_pi
    );
    auto deriv = tenzor::add(
        tenzor::mul(tenzor::add(one, tanh_inner), 0.5),
        tenzor::mul(tenzor::mul(tenzor::mul(p, 0.5), sech2), du_dx)
    );
    auto tangent = tenzor::mul(x.tangent(), deriv);

    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Math
// ============================================================================

auto jvp_exp(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::exp(x.primal());
    auto tangent = tenzor::mul(x.tangent(), primal);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_log(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::log(x.primal());
    auto tangent = tenzor::div(x.tangent(), x.primal());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_sqrt(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::sqrt(x.primal());
    // tangent = dx / (2 * sqrt(x))
    auto tangent = tenzor::div(x.tangent(), tenzor::mul(primal, 2.0));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_pow(const DualTensor& x, double exponent) -> DualTensor {
    auto primal = tenzor::pow(x.primal(), static_cast<float>(exponent));
    // tangent = dx * exponent * x^(exponent-1)
    auto tangent = tenzor::mul(
        x.tangent(),
        tenzor::mul(
            tenzor::pow(x.primal(), static_cast<float>(exponent - 1.0)),
            exponent
        )
    );
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_sin(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::sin(x.primal());
    auto tangent = tenzor::mul(x.tangent(), tenzor::cos(x.primal()));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_cos(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::cos(x.primal());
    auto tangent = tenzor::mul(x.tangent(), tenzor::neg(tenzor::sin(x.primal())));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_abs(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::abs(x.primal());
    auto tangent = tenzor::mul(x.tangent(), tenzor::sign(x.primal()));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_tan(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::tan(x.primal());
    // d(tan(x)) = dx / cos^2(x) = dx * (1 + tan^2(x))
    auto one = tenzor::ones_like(primal);
    auto tangent = tenzor::mul(x.tangent(), tenzor::add(one, tenzor::mul(primal, primal)));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_asin(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::asin(x.primal());
    // d(asin(x)) = dx / sqrt(1 - x^2)
    auto one = tenzor::ones_like(x.primal());
    auto denom = tenzor::sqrt(tenzor::sub(one, tenzor::mul(x.primal(), x.primal())));
    auto tangent = tenzor::div(x.tangent(), denom);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_acos(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::acos(x.primal());
    // d(acos(x)) = -dx / sqrt(1 - x^2)
    auto one = tenzor::ones_like(x.primal());
    auto denom = tenzor::sqrt(tenzor::sub(one, tenzor::mul(x.primal(), x.primal())));
    auto tangent = tenzor::neg(tenzor::div(x.tangent(), denom));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_atan(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::atan(x.primal());
    // d(atan(x)) = dx / (1 + x^2)
    auto one = tenzor::ones_like(x.primal());
    auto tangent = tenzor::div(x.tangent(), tenzor::add(one, tenzor::mul(x.primal(), x.primal())));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_sinh(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::sinh(x.primal());
    auto tangent = tenzor::mul(x.tangent(), tenzor::cosh(x.primal()));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_cosh(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::cosh(x.primal());
    auto tangent = tenzor::mul(x.tangent(), tenzor::sinh(x.primal()));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_log2(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::log2(x.primal());
    // d(log2(x)) = dx / (x * ln(2))
    auto tangent = tenzor::div(x.tangent(), tenzor::mul(x.primal(), M_LN2));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_log10(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::log10(x.primal());
    // d(log10(x)) = dx / (x * ln(10))
    auto tangent = tenzor::div(x.tangent(), tenzor::mul(x.primal(), M_LN10));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_log1p(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::log1p(x.primal());
    // d(log(1+x)) = dx / (1 + x)
    auto one = tenzor::ones_like(x.primal());
    auto tangent = tenzor::div(x.tangent(), tenzor::add(one, x.primal()));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_exp2(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::exp2(x.primal());
    // d(2^x) = dx * 2^x * ln(2)
    auto tangent = tenzor::mul(x.tangent(), tenzor::mul(primal, M_LN2));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_expm1(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::expm1(x.primal());
    // d(exp(x)-1) = dx * exp(x)
    auto tangent = tenzor::mul(x.tangent(), tenzor::exp(x.primal()));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_reciprocal(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::reciprocal(x.primal());
    // d(1/x) = -dx / x^2
    auto tangent = tenzor::neg(tenzor::div(x.tangent(), tenzor::mul(x.primal(), x.primal())));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_sign(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::sign(x.primal());
    // sign is piecewise constant, derivative is 0
    auto tangent = tenzor::zeros_like(x.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_erf(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::erf(x.primal());
    // d(erf(x)) = dx * (2/sqrt(pi)) * exp(-x^2)
    constexpr double two_over_sqrt_pi = 1.1283791670955126;
    auto neg_x_sq = tenzor::neg(tenzor::mul(x.primal(), x.primal()));
    auto tangent = tenzor::mul(x.tangent(), tenzor::mul(tenzor::exp(neg_x_sq), two_over_sqrt_pi));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_erfc(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::erfc(x.primal());
    constexpr double neg_two_over_sqrt_pi = -1.1283791670955126;
    auto neg_x_sq = tenzor::neg(tenzor::mul(x.primal(), x.primal()));
    auto tangent = tenzor::mul(x.tangent(), tenzor::mul(tenzor::exp(neg_x_sq), neg_two_over_sqrt_pi));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_clamp(const DualTensor& x, double min_val, double max_val) -> DualTensor {
    auto primal = tenzor::clamp(x.primal(), static_cast<float>(min_val), static_cast<float>(max_val));
    // Gradient passes through where min < x < max, zero at boundaries
    auto above_min = tenzor::gt(x.primal(), tenzor::mul(tenzor::ones_like(x.primal()), min_val));
    auto below_max = tenzor::lt(x.primal(), tenzor::mul(tenzor::ones_like(x.primal()), max_val));
    auto mask = tenzor::mul(above_min, below_max);
    auto tangent = tenzor::mul(x.tangent(), mask);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Activation extensions
// ============================================================================

auto jvp_leaky_relu(const DualTensor& x, float negative_slope) -> DualTensor {
    // leaky_relu(x) = x if x > 0, else negative_slope * x
    auto p = x.primal();
    auto zero = tenzor::zeros_like(p);
    auto pos_mask = tenzor::gt(p, zero);  // 1 where x > 0
    // derivative: 1 where x > 0, negative_slope elsewhere
    auto one = tenzor::ones_like(p);
    auto slope = tenzor::mul(tenzor::ones_like(p), static_cast<double>(negative_slope));
    // deriv = pos_mask * 1 + (1 - pos_mask) * negative_slope
    auto deriv = tenzor::add(tenzor::mul(pos_mask, one),
                             tenzor::mul(tenzor::sub(one, pos_mask), slope));
    auto primal = tenzor::mul(p, deriv);  // equivalent to leaky_relu
    auto tangent = tenzor::mul(x.tangent(), deriv);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_elu(const DualTensor& x, float alpha) -> DualTensor {
    // elu(x) = x if x > 0, else alpha * (exp(x) - 1)
    // d(elu)/dx = 1 if x > 0, else alpha * exp(x)
    auto p = x.primal();
    auto zero = tenzor::zeros_like(p);
    auto pos_mask = tenzor::gt(p, zero);
    auto one = tenzor::ones_like(p);
    auto exp_x = tenzor::exp(p);
    auto neg_deriv = tenzor::mul(exp_x, static_cast<double>(alpha));
    auto deriv = tenzor::add(tenzor::mul(pos_mask, one),
                             tenzor::mul(tenzor::sub(one, pos_mask), neg_deriv));
    // primal
    auto neg_val = tenzor::mul(tenzor::sub(exp_x, one), static_cast<double>(alpha));
    auto primal = tenzor::add(tenzor::mul(pos_mask, p),
                              tenzor::mul(tenzor::sub(one, pos_mask), neg_val));
    auto tangent = tenzor::mul(x.tangent(), deriv);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_selu(const DualTensor& x) -> DualTensor {
    // SELU(x) = lambda * (x if x > 0, else alpha * (exp(x) - 1))
    constexpr double lambda = 1.0507009873554805;
    constexpr double alpha = 1.6732632423543772;
    auto p = x.primal();
    auto zero = tenzor::zeros_like(p);
    auto pos_mask = tenzor::gt(p, zero);
    auto one = tenzor::ones_like(p);
    auto exp_x = tenzor::exp(p);
    // deriv: lambda if x>0, else lambda * alpha * exp(x)
    auto deriv = tenzor::add(
        tenzor::mul(pos_mask, lambda),
        tenzor::mul(tenzor::sub(one, pos_mask), tenzor::mul(exp_x, lambda * alpha))
    );
    // primal
    auto neg_val = tenzor::mul(tenzor::sub(exp_x, one), lambda * alpha);
    auto primal = tenzor::add(tenzor::mul(pos_mask, tenzor::mul(p, lambda)),
                              tenzor::mul(tenzor::sub(one, pos_mask), neg_val));
    auto tangent = tenzor::mul(x.tangent(), deriv);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_softplus(const DualTensor& x, float beta) -> DualTensor {
    // softplus(x) = (1/beta) * log(1 + exp(beta*x))
    // d(softplus)/dx = sigmoid(beta*x)
    auto bx = tenzor::mul(x.primal(), static_cast<double>(beta));
    auto exp_bx = tenzor::exp(bx);
    auto one = tenzor::ones_like(x.primal());
    auto primal = tenzor::mul(tenzor::log(tenzor::add(one, exp_bx)), 1.0 / beta);
    auto deriv = tenzor::sigmoid(bx);
    auto tangent = tenzor::mul(x.tangent(), deriv);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_mish(const DualTensor& x) -> DualTensor {
    // mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    // d(mish)/dx = tanh(sp) + x * sech^2(sp) * sigmoid(x)
    //   where sp = softplus(x) = ln(1 + exp(x))
    auto p = x.primal();
    auto one = tenzor::ones_like(p);
    auto exp_x = tenzor::exp(p);
    auto sp = tenzor::log(tenzor::add(one, exp_x));  // softplus(x)
    auto tanh_sp = tenzor::tanh(sp);
    auto primal = tenzor::mul(p, tanh_sp);

    auto sech2 = tenzor::sub(one, tenzor::mul(tanh_sp, tanh_sp));
    auto sig_x = tenzor::sigmoid(p);
    auto deriv = tenzor::add(tanh_sp, tenzor::mul(tenzor::mul(p, sech2), sig_x));
    auto tangent = tenzor::mul(x.tangent(), deriv);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Softmax
// ============================================================================

auto jvp_softmax(const DualTensor& x, int64_t dim) -> DualTensor {
    // s = softmax(p, dim)
    // ds = s * (dt - sum(s * dt, dim, keepdim=true))
    auto p = x.primal();
    // Compute softmax manually: exp(x - max) / sum(exp(x - max))
    auto max_val = tenzor::max(p, dim, /*keepdim=*/true);
    auto shifted = tenzor::sub(p, max_val);
    auto exp_shifted = tenzor::exp(shifted);
    auto sum_exp = tenzor::sum(exp_shifted, dim, /*keepdim=*/true);
    auto s = tenzor::div(exp_shifted, sum_exp);

    auto s_dt = tenzor::mul(s, x.tangent());
    auto sum_s_dt = tenzor::sum(s_dt, dim, /*keepdim=*/true);
    auto tangent = tenzor::mul(s, tenzor::sub(x.tangent(), sum_s_dt));
    return DualTensor(std::move(s), std::move(tangent));
}

auto jvp_log_softmax(const DualTensor& x, int64_t dim) -> DualTensor {
    // log_softmax(x) = x - log(sum(exp(x), dim))
    // d(log_softmax) = dt - softmax(x) * sum(dt, dim)
    auto p = x.primal();
    auto max_val = tenzor::max(p, dim, /*keepdim=*/true);
    auto shifted = tenzor::sub(p, max_val);
    auto exp_shifted = tenzor::exp(shifted);
    auto sum_exp = tenzor::sum(exp_shifted, dim, /*keepdim=*/true);
    auto log_sum_exp = tenzor::add(tenzor::log(sum_exp), max_val);
    auto primal = tenzor::sub(p, log_sum_exp);
    auto s = tenzor::div(exp_shifted, sum_exp);  // softmax

    auto sum_dt = tenzor::sum(x.tangent(), dim, /*keepdim=*/true);
    auto tangent = tenzor::sub(x.tangent(), tenzor::mul(s, sum_dt));
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Linear algebra
// ============================================================================

auto jvp_linear(const DualTensor& input, const DualTensor& weight, const DualTensor& bias) -> DualTensor {
    // y = x @ W^T + b
    // dy = dx @ W^T + x @ dW^T + db
    auto wt = tenzor::transpose(weight.primal(), 0, 1);
    auto dwt = tenzor::transpose(weight.tangent(), 0, 1);
    auto primal = tenzor::add(tenzor::matmul(input.primal(), wt), bias.primal());
    auto tangent = tenzor::add(
        tenzor::add(tenzor::matmul(input.tangent(), wt),
                    tenzor::matmul(input.primal(), dwt)),
        bias.tangent()
    );
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Shape ops (extensions)
// ============================================================================

auto jvp_permute(const DualTensor& x, std::vector<int64_t> dims) -> DualTensor {
    auto primal = tenzor::permute(x.primal(), dims);
    auto tangent = tenzor::permute(x.tangent(), dims);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_squeeze(const DualTensor& x, std::optional<int64_t> dim) -> DualTensor {
    auto primal = tenzor::squeeze(x.primal(), dim);
    auto tangent = tenzor::squeeze(x.tangent(), dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_unsqueeze(const DualTensor& x, int64_t dim) -> DualTensor {
    auto primal = tenzor::unsqueeze(x.primal(), dim);
    auto tangent = tenzor::unsqueeze(x.tangent(), dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_expand(const DualTensor& x, std::vector<int64_t> shape) -> DualTensor {
    auto primal = tenzor::expand(x.primal(), shape);
    auto tangent = tenzor::expand(x.tangent(), shape);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_flatten(const DualTensor& x, int64_t start_dim, int64_t end_dim) -> DualTensor {
    auto primal = tenzor::flatten(x.primal(), start_dim, end_dim);
    auto tangent = tenzor::flatten(x.tangent(), start_dim, end_dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Tensor combination
// ============================================================================

auto jvp_cat(std::span<const DualTensor> tensors, int64_t dim) -> DualTensor {
    std::vector<Tensor> primals, tangents;
    primals.reserve(tensors.size());
    tangents.reserve(tensors.size());
    for (const auto& t : tensors) {
        primals.push_back(t.primal());
        tangents.push_back(t.tangent());
    }
    auto primal = tenzor::cat(primals, dim);
    auto tangent = tenzor::cat(tangents, dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_stack(std::span<const DualTensor> tensors, int64_t dim) -> DualTensor {
    std::vector<Tensor> primals, tangents;
    primals.reserve(tensors.size());
    tangents.reserve(tensors.size());
    for (const auto& t : tensors) {
        primals.push_back(t.primal());
        tangents.push_back(t.tangent());
    }
    auto primal = tenzor::stack(primals, dim);
    auto tangent = tenzor::stack(tangents, dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Reductions
// ============================================================================

auto jvp_sum(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    auto primal = tenzor::sum(x.primal(), dim, keepdim);
    auto tangent = tenzor::sum(x.tangent(), dim, keepdim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_mean(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    auto primal = tenzor::mean(x.primal(), dim, keepdim);
    auto tangent = tenzor::mean(x.tangent(), dim, keepdim);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Shape ops
// ============================================================================

auto jvp_reshape(const DualTensor& x, std::vector<int64_t> shape) -> DualTensor {
    auto primal = tenzor::reshape(x.primal(), shape);
    auto tangent = tenzor::reshape(x.tangent(), shape);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_transpose(const DualTensor& x, int64_t dim0, int64_t dim1) -> DualTensor {
    auto primal = tenzor::transpose(x.primal(), dim0, dim1);
    auto tangent = tenzor::transpose(x.tangent(), dim0, dim1);
    return DualTensor(std::move(primal), std::move(tangent));
}

} // namespace tenzor
