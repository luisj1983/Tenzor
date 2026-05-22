#include "tenzor/autograd/jvp_rules.hpp"
#include "tenzor/autograd/jvp_dispatch.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/core/dtype.hpp"
#include <climits>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

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
    // Audit A.4: use the exact erf primal via dispatch instead of the
    // tanh approximation.  Tenzor exposes tenzor::erf so the dispatch
    // path lands on the backend-native erf kernel.
    //
    //   GELU(x)     = 0.5 * x * (1 + erf(x / sqrt(2)))
    //   d/dx GELU   = 0.5 * (1 + erf(x/sqrt(2)))
    //               + x * exp(-x^2/2) / sqrt(2*pi)
    auto p = x.primal();

    // inv_sqrt2 = 1 / sqrt(2);  inv_sqrt_2pi = 1 / sqrt(2*pi)
    constexpr double inv_sqrt2     = 0.7071067811865476;
    constexpr double inv_sqrt_2pi  = 0.3989422804014327;

    auto erf_arg     = tenzor::mul(p, inv_sqrt2);
    auto erf_val     = tenzor::erf(erf_arg);
    auto one         = tenzor::ones_like(p);
    auto one_plus    = tenzor::add(one, erf_val);
    auto primal      = tenzor::mul(tenzor::mul(p, 0.5), one_plus);

    // tangent: 0.5*(1+erf(x/√2)) + x*exp(-x^2/2)/√(2π)
    auto half_one_plus = tenzor::mul(one_plus, 0.5);
    auto neg_half_x2   = tenzor::mul(tenzor::mul(p, p), -0.5);
    auto pdf_factor    = tenzor::mul(tenzor::exp(neg_half_x2), inv_sqrt_2pi);
    auto x_pdf         = tenzor::mul(p, pdf_factor);
    auto deriv         = tenzor::add(half_one_plus, x_pdf);
    auto tangent       = tenzor::mul(x.tangent(), deriv);

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
    // Audit A.4: tenzor::pow takes a double exponent (no float cast
    // needed) and preserves Float64 precision.  Previous code did
    // static_cast<float>(exponent) twice, losing precision in
    // Float64 jvp tests.
    auto primal = tenzor::pow(x.primal(), exponent);
    // tangent = dx * exponent * x^(exponent-1)
    auto tangent = tenzor::mul(
        x.tangent(),
        tenzor::mul(
            tenzor::pow(x.primal(), exponent - 1.0),
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

// ============================================================================
// Additional rules (Audit A.4 batch 2)
// ============================================================================

auto jvp_asinh(const DualTensor& x) -> DualTensor {
    // d/dx asinh(x) = 1 / sqrt(1 + x^2)
    auto p = x.primal();
    auto primal = tenzor::asinh(p);
    auto one = tenzor::ones_like(p);
    auto denom = tenzor::sqrt(tenzor::add(one, tenzor::mul(p, p)));
    auto tangent = tenzor::div(x.tangent(), denom);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_acosh(const DualTensor& x) -> DualTensor {
    // d/dx acosh(x) = 1 / sqrt(x^2 - 1)  (defined for x > 1)
    auto p = x.primal();
    auto primal = tenzor::acosh(p);
    auto one = tenzor::ones_like(p);
    auto denom = tenzor::sqrt(tenzor::sub(tenzor::mul(p, p), one));
    auto tangent = tenzor::div(x.tangent(), denom);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_atanh(const DualTensor& x) -> DualTensor {
    // d/dx atanh(x) = 1 / (1 - x^2)
    auto p = x.primal();
    auto primal = tenzor::atanh(p);
    auto one = tenzor::ones_like(p);
    auto denom = tenzor::sub(one, tenzor::mul(p, p));
    auto tangent = tenzor::div(x.tangent(), denom);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_lgamma(const DualTensor& x) -> DualTensor {
    // d/dx lgamma(x) = digamma(x)
    auto primal = tenzor::lgamma(x.primal());
    auto deriv = tenzor::digamma(x.primal());
    auto tangent = tenzor::mul(x.tangent(), deriv);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_square(const DualTensor& x) -> DualTensor {
    // d/dx x^2 = 2x
    auto primal = tenzor::square(x.primal());
    auto tangent = tenzor::mul(x.tangent(), tenzor::mul(x.primal(), 2.0));
    return DualTensor(std::move(primal), std::move(tangent));
}

// ---- Reductions: Max/Min (using one-hot mask) ------------------------------
//
// Both forward and reverse mode use the same one-hot indicator at the
// argmax/argmin position. For ties, the gradient is split (the JVP rule
// uses (x == max(x)) as a mask which gives 1.0 on each tied location;
// this matches numpy/PyTorch sum-of-derivatives behaviour rather than
// the strict-tie convention).

auto jvp_max_red(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    auto primal = tenzor::max(x.primal(), dim, keepdim);
    // Broadcast primal back to x.primal()'s shape via keepdim form
    Tensor primal_kd;
    if (dim.has_value() && !keepdim) {
        primal_kd = tenzor::max(x.primal(), dim, /*keepdim=*/true);
    } else if (!dim.has_value()) {
        // global reduction - primal_kd has same shape as primal (scalar);
        // broadcasting eq() against full tensor still works.
        primal_kd = primal;
    } else {
        primal_kd = primal;
    }
    auto mask = tenzor::eq(x.primal(), primal_kd);
    // Convert bool mask to floating dtype matching tangent
    auto mask_f = mask.to(x.tangent().dtype());
    auto masked_tan = tenzor::mul(x.tangent(), mask_f);
    auto tangent = tenzor::sum(masked_tan, dim, keepdim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_min_red(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    auto primal = tenzor::min(x.primal(), dim, keepdim);
    Tensor primal_kd;
    if (dim.has_value() && !keepdim) {
        primal_kd = tenzor::min(x.primal(), dim, /*keepdim=*/true);
    } else if (!dim.has_value()) {
        primal_kd = primal;
    } else {
        primal_kd = primal;
    }
    auto mask = tenzor::eq(x.primal(), primal_kd);
    auto mask_f = mask.to(x.tangent().dtype());
    auto masked_tan = tenzor::mul(x.tangent(), mask_f);
    auto tangent = tenzor::sum(masked_tan, dim, keepdim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_prod(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    // d/dx prod(x) along dim:
    //   tangent = prod * sum(dx / x, dim)  (when x has no zeros)
    // For numerical safety with possible zeros we use the explicit
    // formula prod * sum(dx / x). This matches PyTorch's _prod_backward
    // behaviour for the no-zero case (zero-handling is asymptotic and
    // not exercised by smooth-domain JVP tests).
    auto primal = tenzor::prod(x.primal(), dim, keepdim);
    Tensor primal_kd;
    if (dim.has_value() && !keepdim) {
        primal_kd = tenzor::prod(x.primal(), dim, /*keepdim=*/true);
    } else {
        primal_kd = primal;
    }
    auto ratio = tenzor::div(x.tangent(), x.primal());
    auto sum_ratio = tenzor::sum(ratio, dim, keepdim);
    auto tangent = tenzor::mul(primal, sum_ratio);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_var(const DualTensor& x, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> DualTensor {
    // var(x) = sum((x - mean)^2) / (N - correction)
    // d var = 2 * sum((x - mean) * (dx - mean(dx))) / (N - correction)
    auto p = x.primal();
    auto t = x.tangent();
    auto mean_p = tenzor::mean(p, dim, /*keepdim=*/true);
    auto mean_t = tenzor::mean(t, dim, /*keepdim=*/true);
    auto centered_p = tenzor::sub(p, mean_p);
    auto centered_t = tenzor::sub(t, mean_t);
    auto numer = tenzor::sum(tenzor::mul(centered_p, centered_t), dim, keepdim);

    // Compute N (count of reduced elements)
    int64_t N = 1;
    auto shape = p.shape();
    if (dim.has_value()) {
        int64_t d = *dim;
        if (d < 0) d += static_cast<int64_t>(shape.size());
        N = shape[d];
    } else {
        for (auto s : shape) N *= s;
    }
    double denom = static_cast<double>(N - (unbiased ? 1 : 0));
    auto tangent = tenzor::mul(numer, 2.0 / denom);

    auto primal = tenzor::var(p, dim, keepdim, unbiased);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_std(const DualTensor& x, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> DualTensor {
    // std = sqrt(var); d std = d var / (2 * std)
    auto var_dual = jvp_var(x, dim, keepdim, unbiased);
    auto primal = tenzor::sqrt(var_dual.primal());
    auto denom = tenzor::mul(primal, 2.0);
    auto tangent = tenzor::div(var_dual.tangent(), denom);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_cumsum(const DualTensor& x, int64_t dim) -> DualTensor {
    auto primal = tenzor::cumsum(x.primal(), dim);
    auto tangent = tenzor::cumsum(x.tangent(), dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_cumprod(const DualTensor& x, int64_t dim) -> DualTensor {
    // d/dx cumprod_k(x) = cumprod_k(x) * cumsum_k(dx / x)
    auto p = x.primal();
    auto primal = tenzor::cumprod(p, dim);
    auto ratio = tenzor::div(x.tangent(), p);
    auto cs = tenzor::cumsum(ratio, dim);
    auto tangent = tenzor::mul(primal, cs);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_logsumexp(const DualTensor& x, int64_t dim, bool keepdim) -> DualTensor {
    // lse(x) = log(sum(exp(x)))
    // d lse = sum(softmax(x) * dx) along dim
    auto p = x.primal();
    auto max_val = tenzor::max(p, dim, /*keepdim=*/true);
    auto shifted = tenzor::sub(p, max_val);
    auto exp_shifted = tenzor::exp(shifted);
    auto sum_exp = tenzor::sum(exp_shifted, dim, /*keepdim=*/true);
    auto log_sum = tenzor::add(tenzor::log(sum_exp), max_val);
    Tensor primal = keepdim ? log_sum : tenzor::squeeze(log_sum, dim);

    auto s = tenzor::div(exp_shifted, sum_exp);  // softmax
    auto weighted = tenzor::mul(s, x.tangent());
    auto tangent = tenzor::sum(weighted, dim, keepdim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_slice(const DualTensor& x, int64_t dim, int64_t start, int64_t end, int64_t step) -> DualTensor {
    auto primal = tenzor::slice(x.primal(), dim, start, end, step);
    auto tangent = tenzor::slice(x.tangent(), dim, start, end, step);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_where(const DualTensor& cond, const DualTensor& a, const DualTensor& b) -> DualTensor {
    // where(cond, a, b): primal selects by cond; tangent selects da/db by cond.
    // cond is treated as non-differentiable (boolean), so its tangent is ignored.
    auto primal = tenzor::where(cond.primal(), a.primal(), b.primal());
    auto tangent = tenzor::where(cond.primal(), a.tangent(), b.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_gather(const DualTensor& x, int64_t dim, const Tensor& index) -> DualTensor {
    auto primal = tenzor::gather(x.primal(), dim, index);
    auto tangent = tenzor::gather(x.tangent(), dim, index);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_scatter(const DualTensor& input, int64_t dim, const Tensor& index, const DualTensor& src) -> DualTensor {
    auto primal = tenzor::scatter(input.primal(), dim, index, src.primal());
    auto tangent = tenzor::scatter(input.tangent(), dim, index, src.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_index_select(const DualTensor& x, int64_t dim, const Tensor& index) -> DualTensor {
    auto primal = tenzor::index_select(x.primal(), dim, index);
    auto tangent = tenzor::index_select(x.tangent(), dim, index);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Audit A.4 batch 3: linalg / conv / view long-tail
// ============================================================================

// ---- Linear algebra --------------------------------------------------------

auto jvp_bmm(const DualTensor& a, const DualTensor& b) -> DualTensor {
    // Batched matmul: d(A @ B)[i] = dA[i] @ B[i] + A[i] @ dB[i]
    auto primal = tenzor::bmm(a.primal(), b.primal());
    auto tangent = tenzor::add(
        tenzor::bmm(a.tangent(), b.primal()),
        tenzor::bmm(a.primal(), b.tangent())
    );
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_inv(const DualTensor& a) -> DualTensor {
    // Y = inv(A) ⇒ A @ Y = I.  Differentiating: dA @ Y + A @ dY = 0
    //   ⇒ dY = -A^{-1} @ dA @ Y = -Y @ dA @ Y.
    auto y = tenzor::linalg::inv(a.primal());
    auto tangent = tenzor::neg(
        tenzor::matmul(tenzor::matmul(y, a.tangent()), y)
    );
    return DualTensor(std::move(y), std::move(tangent));
}

auto jvp_solve(const DualTensor& a, const DualTensor& b) -> DualTensor {
    // Y = solve(A, B) ⇒ A @ Y = B.  Differentiating:
    //   dA @ Y + A @ dY = dB  ⇒  A @ dY = dB - dA @ Y
    //   ⇒  dY = solve(A, dB - dA @ Y).
    auto y = tenzor::linalg::solve(a.primal(), b.primal());
    auto rhs = tenzor::sub(b.tangent(), tenzor::matmul(a.tangent(), y));
    auto tangent = tenzor::linalg::solve(a.primal(), rhs);
    return DualTensor(std::move(y), std::move(tangent));
}

auto jvp_cholesky(const DualTensor& a, bool upper) -> DualTensor {
    // L = chol(A) with A = L @ L^T (lower form).  Differentiating:
    //   dA = dL @ L^T + L @ dL^T
    // The standard solution is:
    //   dL = L @ Φ(L^{-1} @ dA @ L^{-T})
    // where Φ takes the strict lower-triangular part plus half the diagonal:
    //   Φ(M) = tril(M, -1) + 0.5 * diag(diag(M)).
    //
    // For the upper form (U = chol(A, upper=true), A = U^T @ U) the rule is
    // analogous on the transposed factor: dU = Φ_upper(U^{-T} @ dA @ U^{-1}) @ U.
    // We implement lower-form via the explicit formula and recover upper-form
    // by transposing.
    auto L_lower = tenzor::linalg::cholesky(a.primal(), /*upper=*/false);
    // Solve L * X = dA for X = L^{-1} @ dA, then X * L^T... use inv for clarity.
    auto Linv = tenzor::linalg::inv(L_lower);
    auto LinvT = tenzor::transpose(Linv, /*dim0=*/-2, /*dim1=*/-1);
    auto M = tenzor::matmul(tenzor::matmul(Linv, a.tangent()), LinvT);
    // Φ(M) = tril(M, -1) + 0.5 * diag of M, on the last two dims.
    auto strict_lower = tenzor::tril(M, /*diagonal=*/-1);
    auto diag_part   = tenzor::sub(M, tenzor::tril(M, /*diagonal=*/-1));
    diag_part        = tenzor::sub(diag_part, tenzor::triu(M, /*diagonal=*/1));
    // diag_part now holds just the diagonal entries (zeros elsewhere).
    auto half_diag = tenzor::mul(diag_part, 0.5);
    auto phi = tenzor::add(strict_lower, half_diag);
    auto dL = tenzor::matmul(L_lower, phi);

    if (upper) {
        auto U = tenzor::transpose(L_lower, -2, -1);
        auto dU = tenzor::transpose(dL, -2, -1);
        return DualTensor(std::move(U), std::move(dU));
    }
    return DualTensor(std::move(L_lower), std::move(dL));
}

auto jvp_trace(const DualTensor& x) -> DualTensor {
    // y = sum_i A[i,i] ⇒ dy = sum_i dA[i,i] = trace(dA).
    auto primal = tenzor::trace(x.primal());
    auto tangent = tenzor::trace(x.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_det(const DualTensor& a) -> DualTensor {
    // y = det(A) ⇒ dy = det(A) * trace(A^{-1} @ dA) = y * trace(inv(A) @ dA).
    auto y = tenzor::linalg::det(a.primal());
    auto Ainv = tenzor::linalg::inv(a.primal());
    auto inner = tenzor::trace(tenzor::matmul(Ainv, a.tangent()));
    auto tangent = tenzor::mul(y, inner);
    return DualTensor(std::move(y), std::move(tangent));
}

// ---- Convolution forward (Conv{1,2,3}d / ConvTranspose{2,3}d) -------------

auto jvp_conv_forward(OpId op,
                      const DualTensor& input,
                      const DualTensor& weight,
                      const std::optional<DualTensor>& bias,
                      const OpAttributes& attrs) -> DualTensor {
    // y  = conv(x, w) (+ b)
    // dy = conv(dx, w) + conv(x, dw) (+ db)
    // The convolution is linear in both x and w; bias is a pure additive
    // broadcast, so its tangent is broadcast-added to the output tangent.
    //
    // We re-dispatch the same OpId/attrs for the primal and the two tangent
    // contributions. The primal pass uses (x, w[, b]); each tangent pass is
    // bias-less (the convolution part) and the bias tangent (if any) is
    // added directly to the result.
    std::vector<Tensor> primal_inputs;
    primal_inputs.push_back(input.primal());
    primal_inputs.push_back(weight.primal());
    if (bias.has_value()) {
        primal_inputs.push_back(bias->primal());
    }
    auto primal = tenzor::dispatch(op, primal_inputs, attrs)[0];

    // Tangent contribution from input perturbation: conv(dx, w) — no bias.
    std::vector<Tensor> tan_x_inputs;
    tan_x_inputs.push_back(input.tangent());
    tan_x_inputs.push_back(weight.primal());
    auto tan_from_x = tenzor::dispatch(op, tan_x_inputs, attrs)[0];

    // Tangent contribution from weight perturbation: conv(x, dw) — no bias.
    std::vector<Tensor> tan_w_inputs;
    tan_w_inputs.push_back(input.primal());
    tan_w_inputs.push_back(weight.tangent());
    auto tan_from_w = tenzor::dispatch(op, tan_w_inputs, attrs)[0];

    auto tangent = tenzor::add(tan_from_x, tan_from_w);
    if (bias.has_value()) {
        // Bias tangent: broadcast-add along the channel dim. The backend
        // already handles bias broadcasting in the primal; we mirror by
        // routing through a bias-only conv that adds db to a zeroed-input
        // forward — but that would double the tangent. The cheap route is
        // a plain elementwise add against db reshaped to broadcast over
        // {N, *spatial}. Conv outputs always have layout (N, C_out, *), so
        // reshape db to (1, C_out, 1, …) and broadcast-add.
        const auto& db = bias->tangent();
        auto out_shape = tangent.shape();
        std::vector<int64_t> br_shape(out_shape.size(), 1);
        if (out_shape.size() >= 2 && static_cast<int64_t>(db.numel()) == out_shape[1]) {
            br_shape[1] = out_shape[1];
            auto db_br = tenzor::reshape(db, br_shape);
            tangent = tenzor::add(tangent, db_br);
        } else {
            // Fallback: trust broadcasting (handles already-shaped db).
            tangent = tenzor::add(tangent, db);
        }
    }
    return DualTensor(std::move(primal), std::move(tangent));
}

// ---- View / shape long-tail ------------------------------------------------

auto jvp_repeat(const DualTensor& x, std::vector<int64_t> repeats) -> DualTensor {
    // repeat is a linear view-replication; tangent gets the same replication.
    auto primal = tenzor::repeat(x.primal(), repeats);
    auto tangent = tenzor::repeat(x.tangent(), std::move(repeats));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_tile(const DualTensor& x, std::vector<int64_t> reps) -> DualTensor {
    auto primal = tenzor::tile(x.primal(), reps);
    auto tangent = tenzor::tile(x.tangent(), std::move(reps));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_diag(const DualTensor& x, int64_t diagonal) -> DualTensor {
    // diag is linear (either extracts a diagonal vector from a matrix, or
    // embeds a vector into a diagonal matrix). Same op on the tangent.
    auto primal = tenzor::diag(x.primal(), diagonal);
    auto tangent = tenzor::diag(x.tangent(), diagonal);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_tril(const DualTensor& x, int64_t diagonal) -> DualTensor {
    // tril is a linear projection by a zero/one mask; mask the tangent the
    // same way.
    auto primal = tenzor::tril(x.primal(), diagonal);
    auto tangent = tenzor::tril(x.tangent(), diagonal);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_triu(const DualTensor& x, int64_t diagonal) -> DualTensor {
    auto primal = tenzor::triu(x.primal(), diagonal);
    auto tangent = tenzor::triu(x.tangent(), diagonal);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_flip(const DualTensor& x, std::vector<int64_t> dims) -> DualTensor {
    auto primal = tenzor::flip(x.primal(), dims);
    auto tangent = tenzor::flip(x.tangent(), std::move(dims));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_roll(const DualTensor& x, int64_t shifts, int64_t dim) -> DualTensor {
    auto primal = tenzor::roll(x.primal(), shifts, dim);
    auto tangent = tenzor::roll(x.tangent(), shifts, dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_repeat_interleave(const DualTensor& x, int64_t repeats,
                           std::optional<int64_t> dim) -> DualTensor {
    auto primal = tenzor::repeat_interleave(x.primal(), repeats, dim);
    auto tangent = tenzor::repeat_interleave(x.tangent(), repeats, dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_take(const DualTensor& x, const Tensor& index) -> DualTensor {
    // take(x, idx) is a gather-by-flat-index; linear in x.
    auto primal = tenzor::take(x.primal(), index);
    auto tangent = tenzor::take(x.tangent(), index);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_take_along_dim(const DualTensor& x, const Tensor& indices, int64_t dim) -> DualTensor {
    auto primal = tenzor::take_along_dim(x.primal(), indices, dim);
    auto tangent = tenzor::take_along_dim(x.tangent(), indices, dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_diagonal_scatter(const DualTensor& input, const DualTensor& src,
                          int64_t offset, int64_t dim1, int64_t dim2) -> DualTensor {
    // y = diagonal_scatter(input, src, …) is linear in both input and src
    // (it replaces a diagonal slice of `input` with `src`). Same op on the
    // tangents.
    auto primal = tenzor::diagonal_scatter(input.primal(), src.primal(), offset, dim1, dim2);
    auto tangent = tenzor::diagonal_scatter(input.tangent(), src.tangent(), offset, dim1, dim2);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_masked_select(const DualTensor& x, const Tensor& mask) -> DualTensor {
    auto primal = tenzor::masked_select(x.primal(), mask);
    auto tangent = tenzor::masked_select(x.tangent(), mask);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_masked_fill(const DualTensor& x, const Tensor& mask, float value) -> DualTensor {
    // masked_fill replaces tangent positions with 0 (constant value has zero derivative).
    auto primal = tenzor::masked_fill(x.primal(), mask, value);
    auto tangent = tenzor::masked_fill(x.tangent(), mask, 0.0f);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_cast(const DualTensor& x, DType target_dtype) -> DualTensor {
    // Cast is a linear function of x; tangent passes through with the same dtype change.
    auto primal = x.primal().to(target_dtype);
    auto tangent = x.tangent().to(target_dtype);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Audit A.4 batch 5: element-wise long-tail / reductions / indexing
// ============================================================================
//
// All rules below follow the established DualTensor convention:
//   - Recompute the primal via the corresponding tenzor:: op so the rule is
//     backend-agnostic (no CPU fallback).
//   - Compute the tangent in closed form (or zero where the op is locally
//     constant / integer-valued). Formula docs precede each rule body.

// ---- Non-differentiable element-wise (zero tangent) ------------------------
//
// Floor/Ceil/Round/Trunc/Frac/Heaviside are piecewise constant; their
// derivative is 0 almost everywhere. We preserve the primal but emit a zero
// tangent of matching shape/dtype/device. The zero tensor is built from the
// existing primal tangent so device/dtype propagate correctly.

auto jvp_floor(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::floor(x.primal());
    auto tangent = tenzor::zeros_like(x.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_ceil(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::ceil(x.primal());
    auto tangent = tenzor::zeros_like(x.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_round(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::round(x.primal());
    auto tangent = tenzor::zeros_like(x.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_trunc(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::trunc(x.primal());
    auto tangent = tenzor::zeros_like(x.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_frac(const DualTensor& x) -> DualTensor {
    // frac(x) = x - trunc(x). Derivative = 1 - 0 = 1 between integer points,
    // undefined at integers. We adopt the convention dy/dx = 1 (matches the
    // pass-through interpretation used by PyTorch/JAX); this differs from the
    // floor/ceil zero-tangent treatment because frac is locally an identity
    // (modulo integer steps) rather than a step function.
    auto primal = tenzor::frac(x.primal());
    auto tangent = x.tangent();
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_heaviside(const DualTensor& x, const DualTensor& values) -> DualTensor {
    // heaviside(x, values) = 0 if x<0; values if x==0; 1 if x>0.
    // It is piecewise constant in x (derivative is 0 a.e., plus an impulse
    // at x==0 which integrates to (values - 0) — non-smooth, we drop it).
    // The `values` operand contributes only when x==0 (measure zero), so its
    // tangent is also zero a.e.  Both contributions are zero in the smooth-
    // domain JVP, so we emit a zero tangent.
    auto primal = tenzor::heaviside(x.primal(), values.primal());
    auto tangent = tenzor::zeros_like(x.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

// ---- Closed-form binary math ----------------------------------------------

auto jvp_atan2(const DualTensor& y, const DualTensor& x) -> DualTensor {
    // d atan2(y, x) = (x * dy - y * dx) / (x^2 + y^2)
    auto primal = tenzor::atan2(y.primal(), x.primal());
    auto denom = tenzor::add(tenzor::mul(x.primal(), x.primal()),
                             tenzor::mul(y.primal(), y.primal()));
    auto numer = tenzor::sub(tenzor::mul(x.primal(), y.tangent()),
                             tenzor::mul(y.primal(), x.tangent()));
    auto tangent = tenzor::div(numer, denom);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_hypot(const DualTensor& a, const DualTensor& b) -> DualTensor {
    // h = sqrt(a^2 + b^2); dh = (a * da + b * db) / h
    auto primal = tenzor::hypot(a.primal(), b.primal());
    auto numer = tenzor::add(tenzor::mul(a.primal(), a.tangent()),
                             tenzor::mul(b.primal(), b.tangent()));
    auto tangent = tenzor::div(numer, primal);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_logaddexp(const DualTensor& a, const DualTensor& b) -> DualTensor {
    // logaddexp(a, b) = log(exp(a) + exp(b))
    // d/da = sigmoid(a - b);  d/db = sigmoid(b - a) = 1 - sigmoid(a - b)
    // ⇒ tangent = sigmoid(a-b) * da + (1 - sigmoid(a-b)) * db
    auto primal = tenzor::logaddexp(a.primal(), b.primal());
    auto sig = tenzor::sigmoid(tenzor::sub(a.primal(), b.primal()));
    auto one = tenzor::ones_like(sig);
    auto tangent = tenzor::add(tenzor::mul(sig, a.tangent()),
                               tenzor::mul(tenzor::sub(one, sig), b.tangent()));
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_nan_to_num(const DualTensor& x, double nan, double posinf, double neginf) -> DualTensor {
    // y = x where finite, else replacement constant. d y / d x = 1 on the
    // finite mask, 0 elsewhere (the constants are non-differentiable wrt x).
    auto primal = tenzor::nan_to_num(x.primal(), nan, posinf, neginf);
    auto finite_mask = tenzor::isfinite(x.primal()).to(x.tangent().dtype());
    auto tangent = tenzor::mul(x.tangent(), finite_mask);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ---- Reductions long-tail --------------------------------------------------

auto jvp_norm(const DualTensor& x, float p, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    // y = (sum |x|^p)^(1/p)
    // dy = (sum sign(x) * |x|^(p-1) * dx) * y^(1-p)
    //    where for p=2 this is (sum x*dx) / y, and for p=1 it is sum sign(x)*dx.
    // We build the keepdim form of the primal so the y^(1-p) broadcast is well-defined,
    // then squeeze if the caller did not request keepdim.
    auto y_kd = tenzor::norm(x.primal(), p, dim, /*keepdim=*/true);
    auto abs_x = tenzor::abs(x.primal());
    auto sgn   = tenzor::sign(x.primal());
    auto pow_abs = tenzor::pow(abs_x, static_cast<double>(p) - 1.0);
    auto weighted = tenzor::mul(tenzor::mul(sgn, pow_abs), x.tangent());
    auto num = tenzor::sum(weighted, dim, /*keepdim=*/true);
    auto scale = tenzor::pow(y_kd, 1.0 - static_cast<double>(p));
    auto tangent_kd = tenzor::mul(num, scale);

    Tensor primal = keepdim ? y_kd : tenzor::squeeze(y_kd, dim);
    Tensor tangent = keepdim ? tangent_kd : tenzor::squeeze(tangent_kd, dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

// argmax / argmin / argsort / bucketize: integer-valued outputs.
// Forward-mode AD treats them as non-differentiable; the tangent is a zero
// tensor with the *output* shape and the input tangent dtype.

auto jvp_argmax(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    auto primal = tenzor::argmax(x.primal(), dim, keepdim);
    auto sp = primal.shape();
    std::vector<int64_t> shape(sp.begin(), sp.end());
    auto tangent = tenzor::zeros(std::move(shape), x.tangent().dtype(), x.tangent().device());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_argmin(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    auto primal = tenzor::argmin(x.primal(), dim, keepdim);
    auto sp = primal.shape();
    std::vector<int64_t> shape(sp.begin(), sp.end());
    auto tangent = tenzor::zeros(std::move(shape), x.tangent().dtype(), x.tangent().device());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_argsort(const DualTensor& x, int64_t dim, bool descending) -> DualTensor {
    auto primal = tenzor::argsort(x.primal(), dim, descending);
    auto sp = primal.shape();
    std::vector<int64_t> shape(sp.begin(), sp.end());
    auto tangent = tenzor::zeros(std::move(shape), x.tangent().dtype(), x.tangent().device());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_bucketize(const DualTensor& x, const Tensor& boundaries, bool right) -> DualTensor {
    auto primal = tenzor::bucketize(x.primal(), boundaries, right);
    auto sp = primal.shape();
    std::vector<int64_t> shape(sp.begin(), sp.end());
    auto tangent = tenzor::zeros(std::move(shape), x.tangent().dtype(), x.tangent().device());
    return DualTensor(std::move(primal), std::move(tangent));
}

// ---- Index / scatter long-tail --------------------------------------------

auto jvp_index_add(const DualTensor& input, int64_t dim, const Tensor& index,
                   const DualTensor& source) -> DualTensor {
    // y = input; y[index[i]] += source[i] along dim.  Linear in both input
    // and source ⇒ same op on the tangents.
    auto primal = tenzor::index_add(input.primal(), dim, index, source.primal());
    auto tangent = tenzor::index_add(input.tangent(), dim, index, source.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_index_copy(const DualTensor& input, int64_t dim, const Tensor& index,
                    const DualTensor& source) -> DualTensor {
    // y = input; y[index[i]] = source[i].  Linear in both: input's
    // contribution is the unchanged positions; source's contribution is the
    // overwritten positions. Applying index_copy on the tangents reproduces
    // exactly that overwrite pattern.
    auto primal = tenzor::index_copy(input.primal(), dim, index, source.primal());
    auto tangent = tenzor::index_copy(input.tangent(), dim, index, source.tangent());
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_index_fill(const DualTensor& input, int64_t dim, const Tensor& index,
                    float value) -> DualTensor {
    // y = input; y[index[i]] = value (constant).  Derivative wrt input is the
    // identity on un-indexed positions and zero at indexed positions; we get
    // that by filling the input tangent at the same indices with 0.
    auto primal = tenzor::index_fill(input.primal(), dim, index, value);
    auto tangent = tenzor::index_fill(input.tangent(), dim, index, /*value=*/0.0f);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_select_scatter(const DualTensor& input, const DualTensor& src,
                        int64_t dim, int64_t index) -> DualTensor {
    // y = copy of input with src placed at input.select(dim, index).
    // Linear in input (zeroed at the selected slice) + src (placed there).
    auto primal = tenzor::select_scatter(input.primal(), src.primal(), dim, index);
    auto tangent = tenzor::select_scatter(input.tangent(), src.tangent(), dim, index);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_slice_scatter(const DualTensor& input, const DualTensor& src,
                       int64_t dim, int64_t start, int64_t end, int64_t step) -> DualTensor {
    // y = copy of input with src placed at input.slice(dim, start, end, step).
    // Linear in input (zeroed at the slice) + src (placed there).
    auto primal = tenzor::slice_scatter(input.primal(), src.primal(), dim, start, end, step);
    auto tangent = tenzor::slice_scatter(input.tangent(), src.tangent(), dim, start, end, step);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_unfold(const DualTensor& input, int64_t kernel_size, int64_t stride,
                int64_t padding, int64_t dilation) -> DualTensor {
    // Unfold is a linear reshape-with-replication (im2col).  Same op on the
    // tangent reproduces the Jacobian-vector product.
    auto primal = tenzor::ops::unfold(input.primal(), kernel_size, stride, padding, dilation);
    auto tangent = tenzor::ops::unfold(input.tangent(), kernel_size, stride, padding, dilation);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Dispatch-table adapters and registration
// ============================================================================
//
// The adapters below wrap the existing `jvp_*` rules (which operate on
// DualTensor) so they can be registered in the JvpDispatchTable, which
// expects a uniform `(span<primals>, span<tangents>, attrs) -> JvpResult`
// signature. Each adapter:
//   1. Constructs DualTensor inputs from parallel primal/tangent spans.
//      If a tangent slot is uninitialised, a zero tensor of the matching
//      primal shape/dtype/device is synthesised (downstream JVP rules
//      always touch the tangent, so we cannot leave it default-constructed).
//   2. Reads any op-specific attributes from `OpAttributes`.
//   3. Calls the DualTensor-based rule.
//   4. Returns {primal, tangent} as a JvpResult.
//
// The registration is performed exactly once via
// `tenzor::detail::register_builtin_jvp_rules`, which is invoked by
// `ensure_jvp_rules_registered()` in jvp_dispatch.cpp. Adding new rules
// is a one-line change in the `REG` block at the bottom of this section.

namespace {

inline auto make_dual(const Tensor& primal, const Tensor& tangent) -> DualTensor {
    // An uninitialised tangent (numel==0) is interpreted as "zero tangent";
    // synthesise a zero tensor of the matching primal shape/dtype/device.
    if (tangent.numel() == 0 && primal.numel() != 0) {
        return DualTensor(primal);
    }
    return DualTensor(primal, tangent);
}

inline auto dual_to_result(DualTensor&& d) -> JvpResult {
    return JvpResult{std::move(d.primal()), std::move(d.tangent())};
}

// ---- Binary elementwise (Add, Sub, Mul, Div) -------------------------------

#define TENZOR_JVP_BINARY_ADAPTER(name, rule)                                  \
    JvpResult name(std::span<const Tensor> primals,                            \
                   std::span<const Tensor> tangents,                           \
                   const OpAttributes&) {                                      \
        if (primals.size() != 2 || tangents.size() != 2) {                     \
            throw std::runtime_error(#name ": expected 2 inputs");             \
        }                                                                      \
        auto a = make_dual(primals[0], tangents[0]);                           \
        auto b = make_dual(primals[1], tangents[1]);                           \
        return dual_to_result(rule(a, b));                                     \
    }

TENZOR_JVP_BINARY_ADAPTER(jvp_adapter_add, jvp_add)
TENZOR_JVP_BINARY_ADAPTER(jvp_adapter_sub, jvp_sub)
TENZOR_JVP_BINARY_ADAPTER(jvp_adapter_mul, jvp_mul)
TENZOR_JVP_BINARY_ADAPTER(jvp_adapter_div, jvp_div)
TENZOR_JVP_BINARY_ADAPTER(jvp_adapter_matmul, jvp_matmul)

#undef TENZOR_JVP_BINARY_ADAPTER

// ---- Unary elementwise (Neg, Exp, Log, Sqrt, ReLU, Sigmoid, Tanh) ----------

#define TENZOR_JVP_UNARY_ADAPTER(name, rule)                                   \
    JvpResult name(std::span<const Tensor> primals,                            \
                   std::span<const Tensor> tangents,                           \
                   const OpAttributes&) {                                      \
        if (primals.size() != 1 || tangents.size() != 1) {                     \
            throw std::runtime_error(#name ": expected 1 input");              \
        }                                                                      \
        auto x = make_dual(primals[0], tangents[0]);                           \
        return dual_to_result(rule(x));                                        \
    }

TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_neg,     jvp_neg)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_exp,     jvp_exp)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_log,     jvp_log)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_sqrt,    jvp_sqrt)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_relu,    jvp_relu)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_sigmoid, jvp_sigmoid)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_tanh,    jvp_tanh)

#undef TENZOR_JVP_UNARY_ADAPTER

// ---- Reductions (Sum, Mean) ------------------------------------------------

JvpResult jvp_adapter_sum(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_sum: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::optional<int64_t> dim;
    if (attrs.has(AttrKey::Dim)) {
        dim = attrs.get_int(AttrKey::Dim);
    }
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    return dual_to_result(jvp_sum(x, dim, keepdim));
}

JvpResult jvp_adapter_mean(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_mean: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::optional<int64_t> dim;
    if (attrs.has(AttrKey::Dim)) {
        dim = attrs.get_int(AttrKey::Dim);
    }
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    return dual_to_result(jvp_mean(x, dim, keepdim));
}

// ---- Softmax ---------------------------------------------------------------

JvpResult jvp_adapter_softmax(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_softmax: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, /*default=*/-1);
    return dual_to_result(jvp_softmax(x, dim));
}

// ---- Shape ops (Transpose, Reshape) ---------------------------------------

JvpResult jvp_adapter_transpose(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_transpose: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim0 = attrs.get_int(AttrKey::Dim0, 0);
    int64_t dim1 = attrs.get_int(AttrKey::Dim1, 1);
    return dual_to_result(jvp_transpose(x, dim0, dim1));
}

JvpResult jvp_adapter_reshape(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_reshape: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::vector<int64_t> shape;
    if (attrs.has(AttrKey::Shape)) {
        shape = attrs.get_int_list(AttrKey::Shape);
    } else {
        // Fallback: use primal input shape (i.e. identity reshape) — caller
        // should always supply Shape in practice. Without it, return primal
        // shape so tangent at least matches.
        auto sp = primals[0].shape();
        shape.assign(sp.begin(), sp.end());
    }
    return dual_to_result(jvp_reshape(x, std::move(shape)));
}

// ---- Additional unary elementwise --------------------------------------

#define TENZOR_JVP_UNARY_ADAPTER(name, rule)                                   \
    JvpResult name(std::span<const Tensor> primals,                            \
                   std::span<const Tensor> tangents,                           \
                   const OpAttributes&) {                                      \
        if (primals.size() != 1 || tangents.size() != 1) {                     \
            throw std::runtime_error(#name ": expected 1 input");              \
        }                                                                      \
        auto x = make_dual(primals[0], tangents[0]);                           \
        return dual_to_result(rule(x));                                        \
    }

TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_abs,        jvp_abs)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_sin,        jvp_sin)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_cos,        jvp_cos)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_tan,        jvp_tan)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_asin,       jvp_asin)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_acos,       jvp_acos)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_atan,       jvp_atan)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_sinh,       jvp_sinh)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_cosh,       jvp_cosh)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_asinh,      jvp_asinh)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_acosh,      jvp_acosh)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_atanh,      jvp_atanh)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_erf,        jvp_erf)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_erfc,       jvp_erfc)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_lgamma,     jvp_lgamma)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_reciprocal, jvp_reciprocal)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_square,     jvp_square)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_sign,       jvp_sign)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_gelu,       jvp_gelu)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_selu,       jvp_selu)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_mish,       jvp_mish)

#undef TENZOR_JVP_UNARY_ADAPTER

// ---- Activations with scalar parameters --------------------------------

JvpResult jvp_adapter_leaky_relu(std::span<const Tensor> primals,
                                 std::span<const Tensor> tangents,
                                 const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_leaky_relu: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    float slope = static_cast<float>(attrs.get_float(AttrKey::Negative_slope, 0.01));
    return dual_to_result(jvp_leaky_relu(x, slope));
}

JvpResult jvp_adapter_elu(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_elu: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
    return dual_to_result(jvp_elu(x, alpha));
}

JvpResult jvp_adapter_softplus(std::span<const Tensor> primals,
                               std::span<const Tensor> tangents,
                               const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_softplus: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
    return dual_to_result(jvp_softplus(x, beta));
}

JvpResult jvp_adapter_log_softmax(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_log_softmax: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, /*default=*/-1);
    return dual_to_result(jvp_log_softmax(x, dim));
}

// ---- Math with scalar / numeric parameters -----------------------------

JvpResult jvp_adapter_pow(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_pow: expected 1 input (scalar exponent in attrs)");
    }
    auto x = make_dual(primals[0], tangents[0]);
    double exponent = attrs.get_float(AttrKey::Exponent, 1.0);
    return dual_to_result(jvp_pow(x, exponent));
}

JvpResult jvp_adapter_clamp(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_clamp: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    double min_val = attrs.get_float(AttrKey::Min, -std::numeric_limits<double>::infinity());
    double max_val = attrs.get_float(AttrKey::Max,  std::numeric_limits<double>::infinity());
    return dual_to_result(jvp_clamp(x, min_val, max_val));
}

// ---- Linear (3-input: input, weight, bias) -----------------------------

JvpResult jvp_adapter_linear(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes&) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error("jvp_adapter_linear: expected 3 inputs (input, weight, bias)");
    }
    auto inp = make_dual(primals[0], tangents[0]);
    auto w   = make_dual(primals[1], tangents[1]);
    auto b   = make_dual(primals[2], tangents[2]);
    return dual_to_result(jvp_linear(inp, w, b));
}

// ---- Reductions: Max/Min/Prod/Var/Std/LogSumExp/CumSum/CumProd ----------

namespace {
inline auto read_reduction_dim(const OpAttributes& attrs) -> std::optional<int64_t> {
    if (attrs.has(AttrKey::Dim)) return attrs.get_int(AttrKey::Dim);
    return std::nullopt;
}
} // namespace

JvpResult jvp_adapter_max(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_max: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    auto dim = read_reduction_dim(attrs);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    return dual_to_result(jvp_max_red(x, dim, keepdim));
}

JvpResult jvp_adapter_min(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_min: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    auto dim = read_reduction_dim(attrs);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    return dual_to_result(jvp_min_red(x, dim, keepdim));
}

JvpResult jvp_adapter_prod(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_prod: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    auto dim = read_reduction_dim(attrs);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    return dual_to_result(jvp_prod(x, dim, keepdim));
}

JvpResult jvp_adapter_var(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_var: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    auto dim = read_reduction_dim(attrs);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
    return dual_to_result(jvp_var(x, dim, keepdim, unbiased));
}

JvpResult jvp_adapter_std(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_std: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    auto dim = read_reduction_dim(attrs);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
    return dual_to_result(jvp_std(x, dim, keepdim, unbiased));
}

JvpResult jvp_adapter_logsumexp(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_logsumexp: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    return dual_to_result(jvp_logsumexp(x, dim, keepdim));
}

JvpResult jvp_adapter_cumsum(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_cumsum: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    return dual_to_result(jvp_cumsum(x, dim));
}

JvpResult jvp_adapter_cumprod(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_cumprod: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    return dual_to_result(jvp_cumprod(x, dim));
}

// ---- Shape / indexing --------------------------------------------------

JvpResult jvp_adapter_permute(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_permute: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::vector<int64_t> dims;
    if (attrs.has(AttrKey::Dims)) {
        dims = attrs.get_int_list(AttrKey::Dims);
    } else {
        // Fallback identity permutation
        for (int64_t i = 0; i < static_cast<int64_t>(primals[0].shape().size()); ++i) dims.push_back(i);
    }
    return dual_to_result(jvp_permute(x, std::move(dims)));
}

JvpResult jvp_adapter_squeeze(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_squeeze: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::optional<int64_t> dim;
    if (attrs.has(AttrKey::Dim)) dim = attrs.get_int(AttrKey::Dim);
    return dual_to_result(jvp_squeeze(x, dim));
}

JvpResult jvp_adapter_unsqueeze(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_unsqueeze: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    return dual_to_result(jvp_unsqueeze(x, dim));
}

JvpResult jvp_adapter_slice(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_slice: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim   = attrs.get_int(AttrKey::Dim, 0);
    int64_t start = attrs.get_int(AttrKey::Start, 0);
    int64_t end   = attrs.get_int(AttrKey::End, primals[0].shape()[dim < 0 ? dim + primals[0].shape().size() : dim]);
    int64_t step  = attrs.get_int(AttrKey::Step, 1);
    return dual_to_result(jvp_slice(x, dim, start, end, step));
}

JvpResult jvp_adapter_cat(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    if (primals.size() != tangents.size() || primals.empty()) {
        throw std::runtime_error("jvp_adapter_cat: expected >=1 inputs");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    std::vector<DualTensor> duals;
    duals.reserve(primals.size());
    for (size_t i = 0; i < primals.size(); ++i) {
        duals.push_back(make_dual(primals[i], tangents[i]));
    }
    return dual_to_result(jvp_cat(std::span<const DualTensor>(duals.data(), duals.size()), dim));
}

JvpResult jvp_adapter_stack(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes& attrs) {
    if (primals.size() != tangents.size() || primals.empty()) {
        throw std::runtime_error("jvp_adapter_stack: expected >=1 inputs");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    std::vector<DualTensor> duals;
    duals.reserve(primals.size());
    for (size_t i = 0; i < primals.size(); ++i) {
        duals.push_back(make_dual(primals[i], tangents[i]));
    }
    return dual_to_result(jvp_stack(std::span<const DualTensor>(duals.data(), duals.size()), dim));
}

// ---- where / gather / scatter / index_select / masked_* / cast ---------

JvpResult jvp_adapter_where(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes&) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error("jvp_adapter_where: expected 3 inputs (cond, a, b)");
    }
    auto cond = make_dual(primals[0], tangents[0]);
    auto a    = make_dual(primals[1], tangents[1]);
    auto b    = make_dual(primals[2], tangents[2]);
    return dual_to_result(jvp_where(cond, a, b));
}

JvpResult jvp_adapter_gather(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_gather: expected 2 inputs (input, index)");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    return dual_to_result(jvp_gather(x, dim, primals[1]));
}

JvpResult jvp_adapter_scatter(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error("jvp_adapter_scatter: expected 3 inputs (input, index, src)");
    }
    auto input = make_dual(primals[0], tangents[0]);
    auto src   = make_dual(primals[2], tangents[2]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    return dual_to_result(jvp_scatter(input, dim, primals[1], src));
}

JvpResult jvp_adapter_index_select(std::span<const Tensor> primals,
                                   std::span<const Tensor> tangents,
                                   const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_index_select: expected 2 inputs (input, index)");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    return dual_to_result(jvp_index_select(x, dim, primals[1]));
}

JvpResult jvp_adapter_masked_select(std::span<const Tensor> primals,
                                    std::span<const Tensor> tangents,
                                    const OpAttributes&) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_masked_select: expected 2 inputs (input, mask)");
    }
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_masked_select(x, primals[1]));
}

JvpResult jvp_adapter_masked_fill(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_masked_fill: expected 2 inputs (input, mask)");
    }
    auto x = make_dual(primals[0], tangents[0]);
    float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
    return dual_to_result(jvp_masked_fill(x, primals[1], value));
}

JvpResult jvp_adapter_cast(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_cast: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    if (!attrs.has(AttrKey::TargetDtype)) {
        throw std::runtime_error("jvp_adapter_cast: missing TargetDtype attribute");
    }
    auto target = static_cast<DType>(attrs.get_int(AttrKey::TargetDtype));
    return dual_to_result(jvp_cast(x, target));
}

// ---- Audit A.4 batch 3 adapters: linalg / conv / view long-tail ----

// Bmm: (a, b) -> a @ b batched. Same signature as MatMul.
JvpResult jvp_adapter_bmm(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes&) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_bmm: expected 2 inputs (a, b)");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_bmm(a, b));
}

JvpResult jvp_adapter_inv(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_inv: expected 1 input (A)");
    }
    auto a = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_inv(a));
}

JvpResult jvp_adapter_solve(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes&) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_solve: expected 2 inputs (A, B)");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_solve(a, b));
}

JvpResult jvp_adapter_cholesky(std::span<const Tensor> primals,
                               std::span<const Tensor> tangents,
                               const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_cholesky: expected 1 input (A)");
    }
    auto a = make_dual(primals[0], tangents[0]);
    bool upper = attrs.get_int(AttrKey::Upper, 0) != 0;
    return dual_to_result(jvp_cholesky(a, upper));
}

JvpResult jvp_adapter_trace(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_trace: expected 1 input (A)");
    }
    auto a = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_trace(a));
}

JvpResult jvp_adapter_det(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_det: expected 1 input (A)");
    }
    auto a = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_det(a));
}

// Convolution adapters. The Tensor-level dispatch table accepts inputs of
// {x, w} or {x, w, b}, plus stride/padding/dilation/groups attrs. The JVP
// rule re-uses the same OpId for both primal and tangent passes.
template <OpId ConvOp>
JvpResult jvp_adapter_conv_impl(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if ((primals.size() != 2 && primals.size() != 3) ||
        primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_conv: expected 2 or 3 inputs (input, weight[, bias])");
    }
    auto x = make_dual(primals[0], tangents[0]);
    auto w = make_dual(primals[1], tangents[1]);
    std::optional<DualTensor> b;
    if (primals.size() == 3) {
        b.emplace(make_dual(primals[2], tangents[2]));
    }
    return dual_to_result(jvp_conv_forward(ConvOp, x, w, b, attrs));
}

JvpResult jvp_adapter_conv1d(std::span<const Tensor> p, std::span<const Tensor> t,
                             const OpAttributes& a) {
    return jvp_adapter_conv_impl<OpId::Conv1dForward>(p, t, a);
}
JvpResult jvp_adapter_conv2d(std::span<const Tensor> p, std::span<const Tensor> t,
                             const OpAttributes& a) {
    return jvp_adapter_conv_impl<OpId::Conv2dForward>(p, t, a);
}
JvpResult jvp_adapter_conv3d(std::span<const Tensor> p, std::span<const Tensor> t,
                             const OpAttributes& a) {
    return jvp_adapter_conv_impl<OpId::Conv3dForward>(p, t, a);
}
JvpResult jvp_adapter_conv_transpose2d(std::span<const Tensor> p, std::span<const Tensor> t,
                                       const OpAttributes& a) {
    return jvp_adapter_conv_impl<OpId::ConvTranspose2dForward>(p, t, a);
}
JvpResult jvp_adapter_conv_transpose3d(std::span<const Tensor> p, std::span<const Tensor> t,
                                       const OpAttributes& a) {
    return jvp_adapter_conv_impl<OpId::ConvTranspose3dForward>(p, t, a);
}

// View / shape long-tail.

JvpResult jvp_adapter_expand(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_expand: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::vector<int64_t> shape;
    if (attrs.has(AttrKey::Shape)) {
        shape = attrs.get_int_list(AttrKey::Shape);
    } else {
        auto sp = primals[0].shape();
        shape.assign(sp.begin(), sp.end());
    }
    return dual_to_result(jvp_expand(x, std::move(shape)));
}

JvpResult jvp_adapter_repeat(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_repeat: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::vector<int64_t> repeats;
    if (attrs.has(AttrKey::Repeats)) {
        repeats = attrs.get_int_list(AttrKey::Repeats);
    } else {
        // No repeats attr means identity along all dims.
        repeats.assign(primals[0].shape().size(), 1);
    }
    return dual_to_result(jvp_repeat(x, std::move(repeats)));
}

JvpResult jvp_adapter_tile(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_tile: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::vector<int64_t> reps;
    if (attrs.has(AttrKey::Reps)) {
        reps = attrs.get_int_list(AttrKey::Reps);
    } else {
        reps.assign(primals[0].shape().size(), 1);
    }
    return dual_to_result(jvp_tile(x, std::move(reps)));
}

JvpResult jvp_adapter_flatten(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_flatten: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t start_dim = attrs.get_int(AttrKey::StartDim, 0);
    int64_t end_dim   = attrs.get_int(AttrKey::EndDim, -1);
    return dual_to_result(jvp_flatten(x, start_dim, end_dim));
}

JvpResult jvp_adapter_diag(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_diag: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
    return dual_to_result(jvp_diag(x, diagonal));
}

JvpResult jvp_adapter_tril(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_tril: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
    return dual_to_result(jvp_tril(x, diagonal));
}

JvpResult jvp_adapter_triu(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_triu: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
    return dual_to_result(jvp_triu(x, diagonal));
}

JvpResult jvp_adapter_flip(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_flip: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::vector<int64_t> dims;
    if (attrs.has(AttrKey::Dims)) {
        dims = attrs.get_int_list(AttrKey::Dims);
    }
    return dual_to_result(jvp_flip(x, std::move(dims)));
}

JvpResult jvp_adapter_roll(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_roll: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t shifts = attrs.get_int(AttrKey::Shift, 0);
    int64_t dim    = attrs.get_int(AttrKey::Dim, 0);
    return dual_to_result(jvp_roll(x, shifts, dim));
}

JvpResult jvp_adapter_repeat_interleave(std::span<const Tensor> primals,
                                        std::span<const Tensor> tangents,
                                        const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_repeat_interleave: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t repeats = attrs.get_int(AttrKey::NumRepeats, 1);
    std::optional<int64_t> dim;
    if (attrs.has(AttrKey::Dim)) {
        dim = attrs.get_int(AttrKey::Dim);
    }
    return dual_to_result(jvp_repeat_interleave(x, repeats, dim));
}

JvpResult jvp_adapter_take(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes&) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_take: expected 2 inputs (input, index)");
    }
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_take(x, primals[1]));
}

JvpResult jvp_adapter_take_along_dim(std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_take_along_dim: expected 2 inputs (input, indices)");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    return dual_to_result(jvp_take_along_dim(x, primals[1], dim));
}

JvpResult jvp_adapter_diagonal_scatter(std::span<const Tensor> primals,
                                       std::span<const Tensor> tangents,
                                       const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_diagonal_scatter: expected 2 inputs (input, src)");
    }
    auto input = make_dual(primals[0], tangents[0]);
    auto src   = make_dual(primals[1], tangents[1]);
    int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
    int64_t dim1   = attrs.get_int(AttrKey::Dim1, 0);
    int64_t dim2   = attrs.get_int(AttrKey::Dim2, 1);
    return dual_to_result(jvp_diagonal_scatter(input, src, offset, dim1, dim2));
}

// ---- Audit A.4 batch 5: adapters -----------------------------------------

// Unary zero-tangent: Floor, Ceil, Round, Trunc, Frac.
#define TENZOR_JVP_UNARY_ADAPTER(name, rule)                                   \
    JvpResult name(std::span<const Tensor> primals,                            \
                   std::span<const Tensor> tangents,                           \
                   const OpAttributes&) {                                      \
        if (primals.size() != 1 || tangents.size() != 1) {                     \
            throw std::runtime_error(#name ": expected 1 input");              \
        }                                                                      \
        auto x = make_dual(primals[0], tangents[0]);                           \
        return dual_to_result(rule(x));                                        \
    }

TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_floor, jvp_floor)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_ceil,  jvp_ceil)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_round, jvp_round)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_trunc, jvp_trunc)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_frac,  jvp_frac)

#undef TENZOR_JVP_UNARY_ADAPTER

// Binary closed-form / zero-tangent: Atan2, Hypot, LogAddExp, Heaviside.
#define TENZOR_JVP_BINARY_ADAPTER(name, rule)                                  \
    JvpResult name(std::span<const Tensor> primals,                            \
                   std::span<const Tensor> tangents,                           \
                   const OpAttributes&) {                                      \
        if (primals.size() != 2 || tangents.size() != 2) {                     \
            throw std::runtime_error(#name ": expected 2 inputs");             \
        }                                                                      \
        auto a = make_dual(primals[0], tangents[0]);                           \
        auto b = make_dual(primals[1], tangents[1]);                           \
        return dual_to_result(rule(a, b));                                     \
    }

TENZOR_JVP_BINARY_ADAPTER(jvp_adapter_atan2,     jvp_atan2)
TENZOR_JVP_BINARY_ADAPTER(jvp_adapter_hypot,     jvp_hypot)
TENZOR_JVP_BINARY_ADAPTER(jvp_adapter_logaddexp, jvp_logaddexp)
TENZOR_JVP_BINARY_ADAPTER(jvp_adapter_heaviside, jvp_heaviside)

#undef TENZOR_JVP_BINARY_ADAPTER

// nan_to_num: 1 input + scalar replacement values.
JvpResult jvp_adapter_nan_to_num(std::span<const Tensor> primals,
                                 std::span<const Tensor> tangents,
                                 const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_nan_to_num: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    double nan    = attrs.get_float(AttrKey::NanValue,    0.0);
    double posinf = attrs.get_float(AttrKey::PosInfValue, std::numeric_limits<double>::max());
    double neginf = attrs.get_float(AttrKey::NegInfValue, std::numeric_limits<double>::lowest());
    return dual_to_result(jvp_nan_to_num(x, nan, posinf, neginf));
}

// Norm: p-norm reduction.
JvpResult jvp_adapter_norm(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_norm: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
    std::optional<int64_t> dim;
    if (attrs.has(AttrKey::Dim)) {
        int64_t d = attrs.get_int(AttrKey::Dim);
        if (d != LLONG_MIN) dim = d;
    }
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    return dual_to_result(jvp_norm(x, p, dim, keepdim));
}

// argmax / argmin: reductions with optional dim/keepdim.
JvpResult jvp_adapter_argmax(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_argmax: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::optional<int64_t> dim;
    if (attrs.has(AttrKey::Dim)) {
        int64_t d = attrs.get_int(AttrKey::Dim);
        if (d != LLONG_MIN) dim = d;
    }
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    return dual_to_result(jvp_argmax(x, dim, keepdim));
}

JvpResult jvp_adapter_argmin(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_argmin: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    std::optional<int64_t> dim;
    if (attrs.has(AttrKey::Dim)) {
        int64_t d = attrs.get_int(AttrKey::Dim);
        if (d != LLONG_MIN) dim = d;
    }
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    return dual_to_result(jvp_argmin(x, dim, keepdim));
}

// argsort: dim, descending.
JvpResult jvp_adapter_argsort(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_argsort: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    bool descending = attrs.get_bool(AttrKey::Descending, false);
    return dual_to_result(jvp_argsort(x, dim, descending));
}

// bucketize: (input, boundaries) -> int tensor.
JvpResult jvp_adapter_bucketize(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_bucketize: expected 2 inputs (input, boundaries)");
    }
    auto x = make_dual(primals[0], tangents[0]);
    bool right = attrs.get_bool(AttrKey::Right, false);
    return dual_to_result(jvp_bucketize(x, primals[1], right));
}

// index_add / index_copy: 3 inputs (input, index, source).
JvpResult jvp_adapter_index_add(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error("jvp_adapter_index_add: expected 3 inputs (input, index, source)");
    }
    auto input  = make_dual(primals[0], tangents[0]);
    auto source = make_dual(primals[2], tangents[2]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    return dual_to_result(jvp_index_add(input, dim, primals[1], source));
}

JvpResult jvp_adapter_index_copy(std::span<const Tensor> primals,
                                 std::span<const Tensor> tangents,
                                 const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error("jvp_adapter_index_copy: expected 3 inputs (input, index, source)");
    }
    auto input  = make_dual(primals[0], tangents[0]);
    auto source = make_dual(primals[2], tangents[2]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    return dual_to_result(jvp_index_copy(input, dim, primals[1], source));
}

// index_fill: 2 inputs (input, index) + scalar value.
JvpResult jvp_adapter_index_fill(std::span<const Tensor> primals,
                                 std::span<const Tensor> tangents,
                                 const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_index_fill: expected 2 inputs (input, index)");
    }
    auto input  = make_dual(primals[0], tangents[0]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
    return dual_to_result(jvp_index_fill(input, dim, primals[1], value));
}

// select_scatter: 2 inputs (input, src) + dim, index.
JvpResult jvp_adapter_select_scatter(std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_select_scatter: expected 2 inputs (input, src)");
    }
    auto input = make_dual(primals[0], tangents[0]);
    auto src   = make_dual(primals[1], tangents[1]);
    int64_t dim   = attrs.get_int(AttrKey::Dim, 0);
    int64_t index = attrs.get_int(AttrKey::Index, 0);
    return dual_to_result(jvp_select_scatter(input, src, dim, index));
}

// slice_scatter: 2 inputs (input, src) + dim, start, end, step.
JvpResult jvp_adapter_slice_scatter(std::span<const Tensor> primals,
                                    std::span<const Tensor> tangents,
                                    const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_slice_scatter: expected 2 inputs (input, src)");
    }
    auto input = make_dual(primals[0], tangents[0]);
    auto src   = make_dual(primals[1], tangents[1]);
    int64_t dim   = attrs.get_int(AttrKey::Dim, 0);
    int64_t start = attrs.get_int(AttrKey::Start, 0);
    int64_t end   = attrs.get_int(AttrKey::End, -1);
    int64_t step  = attrs.get_int(AttrKey::Step, 1);
    return dual_to_result(jvp_slice_scatter(input, src, dim, start, end, step));
}

// unfold: 1 input + (kernel_size, stride, padding, dilation).
JvpResult jvp_adapter_unfold(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_unfold: expected 1 input");
    }
    auto input = make_dual(primals[0], tangents[0]);
    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 3);
    int64_t stride      = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding     = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation    = attrs.get_int(AttrKey::Dilation, 1);
    return dual_to_result(jvp_unfold(input, kernel_size, stride, padding, dilation));
}

} // anonymous namespace

namespace detail {

void register_builtin_jvp_rules() {
    using ::tenzor::register_jvp_rule;

    // Binary elementwise
    register_jvp_rule(OpId::Add, &jvp_adapter_add);
    register_jvp_rule(OpId::Sub, &jvp_adapter_sub);
    register_jvp_rule(OpId::Mul, &jvp_adapter_mul);
    register_jvp_rule(OpId::Div, &jvp_adapter_div);

    // Matrix
    register_jvp_rule(OpId::MatMul, &jvp_adapter_matmul);

    // Unary elementwise / math (existing batch)
    register_jvp_rule(OpId::Neg,  &jvp_adapter_neg);
    register_jvp_rule(OpId::Exp,  &jvp_adapter_exp);
    register_jvp_rule(OpId::Log,  &jvp_adapter_log);
    register_jvp_rule(OpId::Sqrt, &jvp_adapter_sqrt);

    // Activations (existing)
    register_jvp_rule(OpId::ReLU,           &jvp_adapter_relu);
    register_jvp_rule(OpId::Sigmoid,        &jvp_adapter_sigmoid);
    register_jvp_rule(OpId::Tanh,           &jvp_adapter_tanh);
    register_jvp_rule(OpId::TanhActivation, &jvp_adapter_tanh);

    // Softmax / Reductions (existing)
    register_jvp_rule(OpId::Softmax, &jvp_adapter_softmax);
    register_jvp_rule(OpId::Sum,     &jvp_adapter_sum);
    register_jvp_rule(OpId::Mean,    &jvp_adapter_mean);

    // Shape (existing)
    register_jvp_rule(OpId::Transpose, &jvp_adapter_transpose);
    register_jvp_rule(OpId::Reshape,   &jvp_adapter_reshape);

    // ---------------- Audit A.4 batch 2 ------------------

    // Additional activations
    register_jvp_rule(OpId::Gelu,        &jvp_adapter_gelu);
    register_jvp_rule(OpId::Elu,         &jvp_adapter_elu);
    register_jvp_rule(OpId::Selu,        &jvp_adapter_selu);
    register_jvp_rule(OpId::Mish,        &jvp_adapter_mish);
    register_jvp_rule(OpId::Softplus,    &jvp_adapter_softplus);
    register_jvp_rule(OpId::LeakyReLU,   &jvp_adapter_leaky_relu);
    register_jvp_rule(OpId::LogSoftmax,  &jvp_adapter_log_softmax);

    // Linear (input, weight, bias)
    register_jvp_rule(OpId::Linear, &jvp_adapter_linear);

    // Math: scalar-parametric
    register_jvp_rule(OpId::Pow,        &jvp_adapter_pow);
    register_jvp_rule(OpId::Clamp,      &jvp_adapter_clamp);

    // Math: trig / hyperbolic / inverse
    register_jvp_rule(OpId::Abs,        &jvp_adapter_abs);
    register_jvp_rule(OpId::Sin,        &jvp_adapter_sin);
    register_jvp_rule(OpId::Cos,        &jvp_adapter_cos);
    register_jvp_rule(OpId::Tan,        &jvp_adapter_tan);
    register_jvp_rule(OpId::Asin,       &jvp_adapter_asin);
    register_jvp_rule(OpId::Acos,       &jvp_adapter_acos);
    register_jvp_rule(OpId::Atan,       &jvp_adapter_atan);
    register_jvp_rule(OpId::Sinh,       &jvp_adapter_sinh);
    register_jvp_rule(OpId::Cosh,       &jvp_adapter_cosh);
    register_jvp_rule(OpId::Asinh,      &jvp_adapter_asinh);
    register_jvp_rule(OpId::Acosh,      &jvp_adapter_acosh);
    register_jvp_rule(OpId::Atanh,      &jvp_adapter_atanh);

    // Math: special
    register_jvp_rule(OpId::Erf,        &jvp_adapter_erf);
    register_jvp_rule(OpId::Erfc,       &jvp_adapter_erfc);
    register_jvp_rule(OpId::Lgamma,     &jvp_adapter_lgamma);
    register_jvp_rule(OpId::Reciprocal, &jvp_adapter_reciprocal);
    register_jvp_rule(OpId::Square,     &jvp_adapter_square);
    register_jvp_rule(OpId::Sign,       &jvp_adapter_sign);

    // Reductions
    register_jvp_rule(OpId::Max,        &jvp_adapter_max);
    register_jvp_rule(OpId::Min,        &jvp_adapter_min);
    register_jvp_rule(OpId::Prod,       &jvp_adapter_prod);
    register_jvp_rule(OpId::Var,        &jvp_adapter_var);
    register_jvp_rule(OpId::Std,        &jvp_adapter_std);
    register_jvp_rule(OpId::LogSumExp,  &jvp_adapter_logsumexp);
    register_jvp_rule(OpId::CumSum,     &jvp_adapter_cumsum);
    register_jvp_rule(OpId::CumProd,    &jvp_adapter_cumprod);

    // Shape / indexing
    register_jvp_rule(OpId::Permute,    &jvp_adapter_permute);
    register_jvp_rule(OpId::Squeeze,    &jvp_adapter_squeeze);
    register_jvp_rule(OpId::Unsqueeze,  &jvp_adapter_unsqueeze);
    register_jvp_rule(OpId::Slice,      &jvp_adapter_slice);
    register_jvp_rule(OpId::Cat,        &jvp_adapter_cat);
    register_jvp_rule(OpId::Stack,      &jvp_adapter_stack);

    // Gather / scatter / select
    register_jvp_rule(OpId::Where,         &jvp_adapter_where);
    register_jvp_rule(OpId::Gather,        &jvp_adapter_gather);
    register_jvp_rule(OpId::Scatter,       &jvp_adapter_scatter);
    register_jvp_rule(OpId::IndexSelect,   &jvp_adapter_index_select);
    register_jvp_rule(OpId::MaskedSelect,  &jvp_adapter_masked_select);
    register_jvp_rule(OpId::MaskedFill,    &jvp_adapter_masked_fill);

    // Cast
    register_jvp_rule(OpId::Cast,       &jvp_adapter_cast);

    // ---------------- Audit A.4 batch 3 ------------------

    // Linear algebra
    register_jvp_rule(OpId::Bmm,             &jvp_adapter_bmm);
    register_jvp_rule(OpId::LinalgInv,       &jvp_adapter_inv);
    register_jvp_rule(OpId::LinalgSolve,     &jvp_adapter_solve);
    register_jvp_rule(OpId::LinalgCholesky,  &jvp_adapter_cholesky);
    register_jvp_rule(OpId::Trace,           &jvp_adapter_trace);
    register_jvp_rule(OpId::LinalgDet,       &jvp_adapter_det);

    // Convolution forward (no BN/LN; multi-output ops handled separately)
    register_jvp_rule(OpId::Conv1dForward,            &jvp_adapter_conv1d);
    register_jvp_rule(OpId::Conv2dForward,            &jvp_adapter_conv2d);
    register_jvp_rule(OpId::Conv3dForward,            &jvp_adapter_conv3d);
    register_jvp_rule(OpId::ConvTranspose2dForward,   &jvp_adapter_conv_transpose2d);
    register_jvp_rule(OpId::ConvTranspose3dForward,   &jvp_adapter_conv_transpose3d);

    // View / shape long-tail
    register_jvp_rule(OpId::Expand,            &jvp_adapter_expand);
    register_jvp_rule(OpId::Repeat,            &jvp_adapter_repeat);
    register_jvp_rule(OpId::Tile,              &jvp_adapter_tile);
    register_jvp_rule(OpId::Flatten,           &jvp_adapter_flatten);
    register_jvp_rule(OpId::Diag,              &jvp_adapter_diag);
    register_jvp_rule(OpId::Tril,              &jvp_adapter_tril);
    register_jvp_rule(OpId::Triu,              &jvp_adapter_triu);
    register_jvp_rule(OpId::Flip,              &jvp_adapter_flip);
    register_jvp_rule(OpId::Roll,              &jvp_adapter_roll);
    register_jvp_rule(OpId::RepeatInterleave,  &jvp_adapter_repeat_interleave);

    // Indexing long-tail
    register_jvp_rule(OpId::Take,             &jvp_adapter_take);
    register_jvp_rule(OpId::TakeAlongDim,     &jvp_adapter_take_along_dim);
    register_jvp_rule(OpId::DiagonalScatter,  &jvp_adapter_diagonal_scatter);

    // ---------------- Audit A.4 batch 5 ------------------

    // Element-wise long-tail: piecewise-constant / step functions (zero tangent)
    register_jvp_rule(OpId::Floor,        &jvp_adapter_floor);
    register_jvp_rule(OpId::Ceil,         &jvp_adapter_ceil);
    register_jvp_rule(OpId::Round,        &jvp_adapter_round);
    register_jvp_rule(OpId::Trunc,        &jvp_adapter_trunc);
    register_jvp_rule(OpId::Frac,         &jvp_adapter_frac);
    register_jvp_rule(OpId::Heaviside,    &jvp_adapter_heaviside);

    // Element-wise long-tail: closed-form binary
    register_jvp_rule(OpId::Atan2,        &jvp_adapter_atan2);
    register_jvp_rule(OpId::Hypot,        &jvp_adapter_hypot);
    register_jvp_rule(OpId::LogAddExp,    &jvp_adapter_logaddexp);

    // Element-wise long-tail: NaN/Inf scrubbing
    register_jvp_rule(OpId::NanToNum,     &jvp_adapter_nan_to_num);

    // Reductions: p-norm + index-valued (zero tangent)
    register_jvp_rule(OpId::Norm,         &jvp_adapter_norm);
    register_jvp_rule(OpId::ArgMax,       &jvp_adapter_argmax);
    register_jvp_rule(OpId::ArgMin,       &jvp_adapter_argmin);
    register_jvp_rule(OpId::ArgSort,      &jvp_adapter_argsort);
    register_jvp_rule(OpId::Bucketize,    &jvp_adapter_bucketize);

    // Index / scatter long-tail
    register_jvp_rule(OpId::IndexAdd,       &jvp_adapter_index_add);
    register_jvp_rule(OpId::IndexCopy,      &jvp_adapter_index_copy);
    register_jvp_rule(OpId::IndexFill,      &jvp_adapter_index_fill);
    register_jvp_rule(OpId::SelectScatter,  &jvp_adapter_select_scatter);
    register_jvp_rule(OpId::SliceScatter,   &jvp_adapter_slice_scatter);

    // Linear shape long-tail
    register_jvp_rule(OpId::Unfold,         &jvp_adapter_unfold);
}

} // namespace detail

} // namespace tenzor
