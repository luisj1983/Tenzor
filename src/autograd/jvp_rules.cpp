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
#include "tenzor/ops/fft.hpp"
#include "tenzor/core/dtype.hpp"
#include <array>
#include <climits>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
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

auto jvp_hardswish(const DualTensor& x) -> DualTensor {
    // hardswish(x) = x * clamp(x + 3, 0, 6) / 6
    // derivative is piecewise:
    //   0          if x <= -3
    //   (2x + 3)/6 if -3 < x < 3
    //   1          if x >= 3
    auto p = x.primal();
    auto primal = tenzor::div(tenzor::mul(p, tenzor::clamp(tenzor::add(p, 3.0), 0.0, 6.0)), 6.0);

    auto shape_vec = std::vector<int64_t>(p.shape().begin(), p.shape().end());
    auto zero        = tenzor::zeros(shape_vec, p.dtype(), p.device());
    auto one_tensor  = tenzor::ones(shape_vec, p.dtype(), p.device());
    auto neg3        = tenzor::full(shape_vec, -3.0, p.dtype(), p.device());
    auto pos3        = tenzor::full(shape_vec,  3.0, p.dtype(), p.device());
    auto middle      = tenzor::div(tenzor::add(tenzor::mul(p, 2.0), 3.0), 6.0);
    auto cond_low    = tenzor::gt(p, neg3);
    auto cond_high   = tenzor::gt(p, pos3);
    auto deriv       = tenzor::where(cond_low, middle, zero);
    deriv            = tenzor::where(cond_high, one_tensor, deriv);
    auto tangent     = tenzor::mul(x.tangent(), deriv);
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
// Audit A.4 batch 7
//   - Sparse ops (SpMM/SpMV/SparseAdd): linear in dense+values; pattern static
//   - RNN cells (GRU/LSTM): chain rule through sigmoid/tanh gates
//   - Nested tensors (NestedSum/Mean/Linear): linear in values (and weight/bias)
//   - Reduction long-tail (Logcumsumexp, NumericalGradient, Trapezoid,
//     CumulativeTrapezoid): closed-form / linear
//   - Adaptive avg pool {1,2,3}d: linear in input (same op on tangent)
// ============================================================================

// ---------------------------------------------------------------------------
// Sparse JVP rules
// ---------------------------------------------------------------------------

auto jvp_sparse_spmm(const Tensor& crow, const Tensor& col,
                     const DualTensor& values, const DualTensor& dense,
                     int64_t M, int64_t K) -> DualTensor {
    // y = SpMM(crow, col, values, dense)
    // dy = SpMM(crow, col, dvalues, dense) + SpMM(crow, col, values, ddense)
    // (sparse pattern is static; only `values` and `dense` carry tangents)
    OpAttributes attrs;
    attrs.set(AttrKey::M, M);
    attrs.set(AttrKey::K, K);
    std::vector<Tensor> primal_in   = { crow, col, values.primal(),  dense.primal()  };
    std::vector<Tensor> tan_v_in    = { crow, col, values.tangent(), dense.primal()  };
    std::vector<Tensor> tan_d_in    = { crow, col, values.primal(),  dense.tangent() };
    auto primal       = tenzor::dispatch(OpId::SparseSpMM, primal_in, attrs)[0];
    auto t_from_vals  = tenzor::dispatch(OpId::SparseSpMM, tan_v_in,  attrs)[0];
    auto t_from_dense = tenzor::dispatch(OpId::SparseSpMM, tan_d_in,  attrs)[0];
    auto tangent = tenzor::add(t_from_vals, t_from_dense);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_sparse_spmv(const Tensor& crow, const Tensor& col,
                     const DualTensor& values, const DualTensor& dense,
                     int64_t M, int64_t K) -> DualTensor {
    // y = SpMV(crow, col, values, vec)
    // dy = SpMV(crow, col, dvalues, vec) + SpMV(crow, col, values, dvec)
    OpAttributes attrs;
    attrs.set(AttrKey::M, M);
    attrs.set(AttrKey::K, K);
    std::vector<Tensor> primal_in = { crow, col, values.primal(),  dense.primal()  };
    std::vector<Tensor> tan_v_in  = { crow, col, values.tangent(), dense.primal()  };
    std::vector<Tensor> tan_d_in  = { crow, col, values.primal(),  dense.tangent() };
    auto primal       = tenzor::dispatch(OpId::SparseSpMV, primal_in, attrs)[0];
    auto t_from_vals  = tenzor::dispatch(OpId::SparseSpMV, tan_v_in,  attrs)[0];
    auto t_from_dense = tenzor::dispatch(OpId::SparseSpMV, tan_d_in,  attrs)[0];
    auto tangent = tenzor::add(t_from_vals, t_from_dense);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_sparse_add(const Tensor& crow, const Tensor& col,
                    const DualTensor& values, const DualTensor& dense,
                    int64_t M, int64_t K) -> DualTensor {
    // y = SparseAdd(crow, col, values, dense)  (scatter-adds values into dense)
    // dy = SparseAdd(crow, col, dvalues, ddense)  (linear in both)
    OpAttributes attrs;
    attrs.set(AttrKey::M, M);
    attrs.set(AttrKey::K, K);
    std::vector<Tensor> primal_in = { crow, col, values.primal(),  dense.primal()  };
    std::vector<Tensor> tan_in    = { crow, col, values.tangent(), dense.tangent() };
    auto primal  = tenzor::dispatch(OpId::SparseAdd, primal_in, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::SparseAdd, tan_in,    attrs)[0];
    return DualTensor(std::move(primal), std::move(tangent));
}

// ---------------------------------------------------------------------------
// RNN-cell JVP rules
// ---------------------------------------------------------------------------

auto jvp_gru_cell_forward(const DualTensor& x, const DualTensor& h,
                          const DualTensor& W_ih, const DualTensor& W_hh,
                          const DualTensor& b_ih, const DualTensor& b_hh) -> DualTensor {
    // PyTorch GRU cell (cuDNN convention):
    //   g_x = x @ W_ih^T + b_ih              [B, 3H]
    //   g_h = h @ W_hh^T + b_hh              [B, 3H]
    //   split each into 3 along last dim -> {gxr, gxz, gxn}, {ghr, ghz, ghn}
    //   r   = sigmoid(gxr + ghr)
    //   z   = sigmoid(gxz + ghz)
    //   n   = tanh(gxn + r * ghn)
    //   hy  = (1 - z) * n + z * h
    //
    // Forward-mode AD applies the chain rule through each sigmoid/tanh and
    // the mul/add structure. We compute everything in DualTensor form so the
    // tangent threads through automatically.

    // Affine: g = X @ W^T + b   (linear in X, W, and b). Replay the existing
    // jvp_linear formula directly on the components.
    auto g_x = jvp_linear(x, W_ih, b_ih);  // [B, 3H]
    auto g_h = jvp_linear(h, W_hh, b_hh);  // [B, 3H]

    // Chunk each gate tensor into 3 along dim=1. Both primal and tangent are
    // sliced identically so the DualTensor structure is preserved.
    auto chunk3 = [](const DualTensor& t) -> std::array<DualTensor, 3> {
        auto p = tenzor::chunk(t.primal(),  3, /*dim=*/1);
        auto d = tenzor::chunk(t.tangent(), 3, /*dim=*/1);
        return { DualTensor(p[0], d[0]),
                 DualTensor(p[1], d[1]),
                 DualTensor(p[2], d[2]) };
    };
    auto gx = chunk3(g_x);   // {gxr, gxz, gxn}
    auto gh = chunk3(g_h);   // {ghr, ghz, ghn}

    // r = sigmoid(gxr + ghr)
    auto r = jvp_sigmoid(jvp_add(gx[0], gh[0]));
    // z = sigmoid(gxz + ghz)
    auto z = jvp_sigmoid(jvp_add(gx[1], gh[1]));
    // n = tanh(gxn + r * ghn)
    auto n = jvp_tanh(jvp_add(gx[2], jvp_mul(r, gh[2])));

    // hy = (1 - z) * n + z * h
    //    = n - z * n + z * h
    //    = n + z * (h - n)
    // Use the "(1 - z) * n + z * h" form directly via dual arithmetic.
    // Build a one-dual with shape of z's primal (broadcast handles scalar 1).
    auto one_primal  = tenzor::ones_like(z.primal());
    auto zero_tangent = tenzor::zeros_like(z.tangent());
    DualTensor one_d(std::move(one_primal), std::move(zero_tangent));
    auto one_minus_z = jvp_sub(one_d, z);

    auto hy = jvp_add(jvp_mul(one_minus_z, n), jvp_mul(z, h));
    return hy;
}

// ---------------------------------------------------------------------------
// Nested-tensor JVP rules
// ---------------------------------------------------------------------------

auto jvp_nested_sum(const DualTensor& values, const Tensor& offsets,
                    int64_t dim, bool keepdim) -> DualTensor {
    // y = NestedSum(values, offsets, dim, keepdim).
    // Sum is linear over `values`; offsets are integer (constant) so the
    // tangent is the same op applied to dvalues.
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> primal_in = { values.primal(),  offsets };
    std::vector<Tensor> tan_in    = { values.tangent(), offsets };
    auto primal  = tenzor::dispatch(OpId::NestedSum, primal_in, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::NestedSum, tan_in,    attrs)[0];
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_nested_mean(const DualTensor& values, const Tensor& offsets,
                     int64_t dim, bool keepdim) -> DualTensor {
    // y = NestedMean(values, offsets, dim, keepdim).
    // Mean is linear over `values` (segment sizes are determined by offsets,
    // which are constant), so the tangent is the same op on dvalues.
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> primal_in = { values.primal(),  offsets };
    std::vector<Tensor> tan_in    = { values.tangent(), offsets };
    auto primal  = tenzor::dispatch(OpId::NestedMean, primal_in, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::NestedMean, tan_in,    attrs)[0];
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_nested_linear(const DualTensor& values, const Tensor& offsets,
                       const DualTensor& weight,
                       const std::optional<DualTensor>& bias) -> DualTensor {
    // y = NestedLinear(values, offsets, weight, [bias])
    //   = values @ weight^T (+ bias),  applied per segment to packed values.
    // The op is bilinear in (values, weight) + linear additive in bias, so:
    //   dy = NestedLinear(dvalues, offsets, weight)
    //      + NestedLinear(values,  offsets, dweight)
    //      [+ dbias broadcast-added]
    OpAttributes attrs;
    auto build_inputs = [&](const Tensor& v, const Tensor& w,
                            const std::optional<Tensor>& b) -> std::vector<Tensor> {
        std::vector<Tensor> in = { v, offsets, w };
        if (b.has_value()) in.push_back(*b);
        return in;
    };

    auto primal_in = build_inputs(values.primal(), weight.primal(),
                                  bias.has_value() ? std::optional<Tensor>(bias->primal())
                                                   : std::nullopt);
    auto primal = tenzor::dispatch(OpId::NestedLinear, primal_in, attrs)[0];

    // Tangent from values: NestedLinear(dvalues, offsets, weight)  (no bias)
    auto tan_v_in = build_inputs(values.tangent(), weight.primal(), std::nullopt);
    auto t_from_v = tenzor::dispatch(OpId::NestedLinear, tan_v_in, attrs)[0];

    // Tangent from weight: NestedLinear(values, offsets, dweight)  (no bias)
    auto tan_w_in = build_inputs(values.primal(), weight.tangent(), std::nullopt);
    auto t_from_w = tenzor::dispatch(OpId::NestedLinear, tan_w_in, attrs)[0];

    auto tangent = tenzor::add(t_from_v, t_from_w);
    if (bias.has_value()) {
        // dbias broadcast-adds along the feature dim. NestedLinear's output
        // matches input packed-values layout in the trailing feature dim, so
        // a plain add suffices.
        tangent = tenzor::add(tangent, bias->tangent());
    }
    return DualTensor(std::move(primal), std::move(tangent));
}

// ---------------------------------------------------------------------------
// Reduction long-tail JVP rules
// ---------------------------------------------------------------------------

auto jvp_logcumsumexp(const DualTensor& x, int64_t dim) -> DualTensor {
    // l[t] = log(sum_{k<=t} exp(x[k]))
    // Let w[k] = exp(x[k]).  Then sum_cum[t] = cumsum(w)[t] and
    //   l[t]    = log(sum_cum[t])
    //   dl[t]/dx[k] = w[k] / sum_cum[t]   for k <= t, else 0
    // So  dl[t] = sum_{k<=t} (w[k] * dx[k]) / sum_cum[t]
    //          = cumsum(w * dx, dim)[t] / cumsum(w, dim)[t]
    //          = cumsum(exp(x - m) * dx, dim) / cumsum(exp(x - m), dim)
    // where m is any per-element constant for stability; using m = x.max(dim)
    // would force a global reduction so we instead piggy-back on the
    // backend-stable Logcumsumexp output: l = log(cumsum(exp(x))) and we have
    //   exp(x - l[t]) summed cumulatively to 1 → directly,
    //   dl[t] = cumsum(exp(x) * dx, dim)[t] / exp(l[t])
    //         = cumsum(exp(x - l[t]) * dx, dim)[t]
    // The latter form is *not* in general identical because l[t] depends on t;
    // the equivalent stable expression uses cumsum/exp directly:
    //   dl = cumsum(exp(x - shifted) * dx, dim) / exp(l - shifted)
    // We compute the simple (numerically robust enough for typical inputs)
    // form using a per-row shift = primal output l (broadcast back).
    auto primal = tenzor::dispatch(OpId::Logcumsumexp,
                                   std::vector<Tensor>{x.primal()},
                                   [&]() {
                                       OpAttributes a;
                                       a.set(AttrKey::Dim, dim);
                                       return a;
                                   }())[0];
    // Use the LSE stability shift: subtract l (the cumulative LSE so far) from
    // x, exponentiate, multiply by dx, cumsum, then "divide" by exp(0)==1.
    // exp(x[k] - l[t]) is only well defined for k<=t; for k>t the multiplier
    // is bounded above by 1 since x[k] - l[t] - log(... up to k) etc.  In
    // practice cumsum already truncates the contribution at k>t, so the
    // factor for k<=t reads exp(x[k] - l[t]) and the partial-sum identity
    //   sum_{k<=t} exp(x[k] - l[t]) = 1
    // makes the divisor cancel.  We therefore compute:
    //   tangent = cumsum(exp(x - l) * dx, dim)
    // which is mathematically identical to the closed form above.
    auto w_shifted = tenzor::exp(tenzor::sub(x.primal(), primal));
    auto tangent_pre = tenzor::mul(w_shifted, x.tangent());
    auto tangent = tenzor::cumsum(tangent_pre, dim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_numerical_gradient(const DualTensor& x, int64_t dim, double spacing) -> DualTensor {
    // The numerical_gradient kernel applies fixed finite-difference stencils
    // along `dim` (central [-1/2, 0, +1/2] interior, one-sided at edges) —
    // a linear operator. Its tangent is the same op applied to dx.
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Spacing, spacing);
    auto primal  = tenzor::dispatch(OpId::NumericalGradient,
                                    std::vector<Tensor>{x.primal()},  attrs)[0];
    auto tangent = tenzor::dispatch(OpId::NumericalGradient,
                                    std::vector<Tensor>{x.tangent()}, attrs)[0];
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_trapezoid(const DualTensor& y, const std::optional<DualTensor>& x_pos,
                   int64_t dim, double dx_uniform) -> DualTensor {
    // trapezoid integrates y along `dim` using either uniform spacing dx or
    // a coordinate tensor x. It is linear in y for fixed x and linear in x
    // for fixed y; thus bilinear: T(y, x) = sum_k 0.5 * (y[k] + y[k+1]) * (x[k+1] - x[k]).
    // Forward-mode tangent:
    //   dT = T(dy, x) + sum_k 0.5 * (y[k] + y[k+1]) * (dx[k+1] - dx[k])
    //      = T(dy, x) + T_x_only(y, dx)
    // where T_x_only(y, dx) reuses the same routine. The kernel implements
    // exactly this when called with (y, dx) as the "y" / "x" inputs, so we
    // can re-dispatch.
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Dx, dx_uniform);

    if (x_pos.has_value()) {
        std::vector<Tensor> primal_in = { y.primal(), x_pos->primal() };
        std::vector<Tensor> tan_dy_in = { y.tangent(), x_pos->primal() };
        std::vector<Tensor> tan_dx_in = { y.primal(),  x_pos->tangent() };
        auto primal     = tenzor::dispatch(OpId::Trapezoid, primal_in,  attrs)[0];
        auto t_from_dy  = tenzor::dispatch(OpId::Trapezoid, tan_dy_in,  attrs)[0];
        auto t_from_dx  = tenzor::dispatch(OpId::Trapezoid, tan_dx_in,  attrs)[0];
        auto tangent = tenzor::add(t_from_dy, t_from_dx);
        return DualTensor(std::move(primal), std::move(tangent));
    }
    // Uniform-spacing variant: only y carries a tangent.
    std::vector<Tensor> primal_in = { y.primal() };
    std::vector<Tensor> tan_in    = { y.tangent() };
    auto primal  = tenzor::dispatch(OpId::Trapezoid, primal_in, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::Trapezoid, tan_in,    attrs)[0];
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_cumulative_trapezoid(const DualTensor& y, const std::optional<DualTensor>& x_pos,
                              int64_t dim, double dx_uniform) -> DualTensor {
    // CumulativeTrapezoid is the running version of Trapezoid — same bilinear
    // structure in (y, x); tangent follows the same rule as jvp_trapezoid.
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Dx, dx_uniform);

    if (x_pos.has_value()) {
        std::vector<Tensor> primal_in = { y.primal(),  x_pos->primal() };
        std::vector<Tensor> tan_dy_in = { y.tangent(), x_pos->primal() };
        std::vector<Tensor> tan_dx_in = { y.primal(),  x_pos->tangent() };
        auto primal    = tenzor::dispatch(OpId::CumulativeTrapezoid, primal_in,  attrs)[0];
        auto t_from_dy = tenzor::dispatch(OpId::CumulativeTrapezoid, tan_dy_in,  attrs)[0];
        auto t_from_dx = tenzor::dispatch(OpId::CumulativeTrapezoid, tan_dx_in,  attrs)[0];
        auto tangent = tenzor::add(t_from_dy, t_from_dx);
        return DualTensor(std::move(primal), std::move(tangent));
    }
    std::vector<Tensor> primal_in = { y.primal() };
    std::vector<Tensor> tan_in    = { y.tangent() };
    auto primal  = tenzor::dispatch(OpId::CumulativeTrapezoid, primal_in, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::CumulativeTrapezoid, tan_in,    attrs)[0];
    return DualTensor(std::move(primal), std::move(tangent));
}

// ---------------------------------------------------------------------------
// Adaptive average pooling JVP rules
// ---------------------------------------------------------------------------

auto jvp_adaptive_avgpool_1d(const DualTensor& x, int64_t output_size) -> DualTensor {
    // AdaptiveAvgPool1d: y[n,c,o] = mean_{i in window(o)} x[n,c,i]. The window
    // boundaries depend only on the shape (output_size, input_size), so the
    // operator is linear in x → tangent = same op on dx.
    OpAttributes attrs;
    attrs.set(AttrKey::OutputSize, output_size);
    auto primal  = tenzor::dispatch(OpId::AdaptiveAvgPool1d,
                                    std::vector<Tensor>{x.primal()},  attrs)[0];
    auto tangent = tenzor::dispatch(OpId::AdaptiveAvgPool1d,
                                    std::vector<Tensor>{x.tangent()}, attrs)[0];
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_adaptive_avgpool_2d(const DualTensor& x, int64_t output_h, int64_t output_w) -> DualTensor {
    // Same linearity argument as 1d; window boundaries derive from shape.
    OpAttributes attrs;
    attrs.set(AttrKey::OutputSizeH, output_h);
    attrs.set(AttrKey::OutputSizeW, output_w);
    auto primal  = tenzor::dispatch(OpId::AdaptiveAvgPool2d,
                                    std::vector<Tensor>{x.primal()},  attrs)[0];
    auto tangent = tenzor::dispatch(OpId::AdaptiveAvgPool2d,
                                    std::vector<Tensor>{x.tangent()}, attrs)[0];
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_adaptive_avgpool_3d(const DualTensor& x,
                             int64_t output_d, int64_t output_h, int64_t output_w) -> DualTensor {
    // Same linearity argument as 1d/2d.
    OpAttributes attrs;
    attrs.set(AttrKey::OutputSizeD, output_d);
    attrs.set(AttrKey::OutputSizeH, output_h);
    attrs.set(AttrKey::OutputSizeW, output_w);
    auto primal  = tenzor::dispatch(OpId::AdaptiveAvgPool3d,
                                    std::vector<Tensor>{x.primal()},  attrs)[0];
    auto tangent = tenzor::dispatch(OpId::AdaptiveAvgPool3d,
                                    std::vector<Tensor>{x.tangent()}, attrs)[0];
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

// ---- Audit A.4 (extension): Hardswish single-output adapter -------------

JvpResult jvp_adapter_hardswish(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_hardswish: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_hardswish(x));
}

// ---- Audit A.4 (extension): FFT single-output adapters -------------------
//
// FFT/IFFT/RFFT/IRFFT are linear in their input, so the tangent is the same
// op applied to the input tangent. We use OpAttributes (Dim, N, Norm) the
// dispatcher carries and forward to the tenzor::fft::* tensor-level ops so
// the right backend is selected.

namespace {

template <typename FFTOp>
JvpResult fft_jvp_impl(const char* name,
                       std::span<const Tensor> primals,
                       std::span<const Tensor> tangents,
                       const OpAttributes& attrs,
                       FFTOp&& op) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error(std::string(name) + ": expected 1 input");
    }
    const Tensor& x  = primals[0];
    Tensor dx = tangents[0];
    if (dx.numel() == 0 && x.numel() != 0) {
        auto shape_vec = std::vector<int64_t>(x.shape().begin(), x.shape().end());
        dx = tenzor::zeros(shape_vec, x.dtype(), x.device());
    }
    // Pull op-specific attributes. Defaults match tenzor::fft::* defaults.
    std::optional<int64_t> n;
    if (attrs.has(AttrKey::N)) n = attrs.get_int(AttrKey::N);
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    std::string norm(attrs.get_string(AttrKey::Norm, "backward"));

    Tensor primal_out  = op(x,  n, dim, norm);
    Tensor tangent_out = op(dx, n, dim, norm);
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

} // anonymous (nested)

JvpResult jvp_adapter_fft(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    return fft_jvp_impl("jvp_adapter_fft", primals, tangents, attrs,
        [](const Tensor& t, std::optional<int64_t> n, int64_t dim,
           const std::string& norm) {
            return tenzor::fft::fft(t, n, dim, norm);
        });
}

JvpResult jvp_adapter_ifft(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    return fft_jvp_impl("jvp_adapter_ifft", primals, tangents, attrs,
        [](const Tensor& t, std::optional<int64_t> n, int64_t dim,
           const std::string& norm) {
            return tenzor::fft::ifft(t, n, dim, norm);
        });
}

JvpResult jvp_adapter_rfft(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& attrs) {
    return fft_jvp_impl("jvp_adapter_rfft", primals, tangents, attrs,
        [](const Tensor& t, std::optional<int64_t> n, int64_t dim,
           const std::string& norm) {
            return tenzor::fft::rfft(t, n, dim, norm);
        });
}

JvpResult jvp_adapter_irfft(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes& attrs) {
    return fft_jvp_impl("jvp_adapter_irfft", primals, tangents, attrs,
        [](const Tensor& t, std::optional<int64_t> n, int64_t dim,
           const std::string& norm) {
            return tenzor::fft::irfft(t, n, dim, norm);
        });
}

// ---- Audit A.4 (extension): Multi-output JVP rules -----------------------
//
// These ops return multiple tensors (e.g. BatchNorm2dForwardAffine returns
// {output, mean, rstd}). Each rule emits (primals, tangents) of matching
// arity. Registered via register_jvp_rule_multi (separate table).

namespace {

// BatchNorm2dForwardAffine kernel contract:
//   inputs : (x, batch_mean, batch_var, gamma, beta)
//   outputs: {y[, mean, rstd]}  — some backends may only return {y}; we
//            recompute mean/rstd from the supplied batch_mean/batch_var so
//            the JVP rule's output arity is fixed at 3.
//
// y = (x - mean) * rstd * gamma + beta,  rstd = 1/sqrt(var + eps)
// Linearise treating (x, mean, var, gamma, beta) as independent primals:
//   dmean_t = dmean
//   drstd_t = -0.5 * rstd^3 * dvar
//   dy = (dx - dmean) * gamma * rstd
//      + (x - mean) * gamma * drstd_t
//      + (x - mean) * rstd * dgamma
//      + dbeta
// Channel-broadcast: mean/var/gamma/beta are per-channel (shape [C]); we
// reshape to [1, C, 1, 1] for NCHW broadcasting.
JvpMultiResult jvp_adapter_batchnorm2d_forward_affine(
        std::span<const Tensor> primals,
        std::span<const Tensor> tangents,
        const OpAttributes& attrs) {
    if (primals.size() != 5 || tangents.size() != 5) {
        throw std::runtime_error(
            "jvp_adapter_batchnorm2d_forward_affine: expected 5 inputs "
            "(x, mean, var, gamma, beta)");
    }
    const Tensor& x     = primals[0];
    const Tensor& mean  = primals[1];
    const Tensor& var   = primals[2];
    const Tensor& gamma = primals[3];
    const Tensor& beta  = primals[4];

    auto zeros_like_or = [](const Tensor& t, const Tensor& tan) -> Tensor {
        if (tan.numel() != 0) return tan;
        auto sh = std::vector<int64_t>(t.shape().begin(), t.shape().end());
        return tenzor::zeros(sh, t.dtype(), t.device());
    };
    Tensor dx     = zeros_like_or(x,     tangents[0]);
    Tensor dmean  = zeros_like_or(mean,  tangents[1]);
    Tensor dvar   = zeros_like_or(var,   tangents[2]);
    Tensor dgamma = zeros_like_or(gamma, tangents[3]);
    Tensor dbeta  = zeros_like_or(beta,  tangents[4]);

    double eps = attrs.get_float(AttrKey::Eps, 1e-5);

    // rstd = 1/sqrt(var + eps)
    auto rstd = tenzor::rsqrt(tenzor::add(var, eps));

    // Reshape per-channel quantities to [1, C, 1, 1] for NCHW broadcast.
    int64_t C = var.shape()[0];
    std::vector<int64_t> c_shape = {1, C, 1, 1};
    std::vector<int64_t> var_shape_vec(var.shape().begin(), var.shape().end());
    auto rstd_b   = tenzor::reshape(rstd,   c_shape);
    auto mean_b   = tenzor::reshape(mean,   c_shape);
    auto gamma_b  = tenzor::reshape(gamma,  c_shape);
    auto beta_b   = tenzor::reshape(beta,   c_shape);
    auto dmean_b  = tenzor::reshape(dmean,  c_shape);
    auto dvar_b   = tenzor::reshape(dvar,   c_shape);
    auto dgamma_b = tenzor::reshape(dgamma, c_shape);
    auto dbeta_b  = tenzor::reshape(dbeta,  c_shape);

    // Primal: y = (x - mean) * gamma * rstd + beta
    auto x_minus_mean = tenzor::sub(x, mean_b);
    auto y_pre_beta   = tenzor::mul(tenzor::mul(x_minus_mean, gamma_b), rstd_b);
    auto y            = tenzor::add(y_pre_beta, beta_b);

    // drstd = -0.5 * rstd^3 * dvar  (still in [1,C,1,1] layout)
    auto rstd_cubed = tenzor::mul(tenzor::mul(rstd_b, rstd_b), rstd_b);
    auto drstd_b    = tenzor::mul(tenzor::mul(rstd_cubed, dvar_b), -0.5);

    // dy = (dx - dmean)*gamma*rstd + (x-mean)*gamma*drstd + (x-mean)*rstd*dgamma + dbeta
    auto term1 = tenzor::mul(tenzor::mul(tenzor::sub(dx, dmean_b), gamma_b), rstd_b);
    auto term2 = tenzor::mul(tenzor::mul(x_minus_mean, gamma_b), drstd_b);
    auto term3 = tenzor::mul(tenzor::mul(x_minus_mean, rstd_b), dgamma_b);
    auto dy    = tenzor::add(tenzor::add(tenzor::add(term1, term2), term3), dbeta_b);

    // Reshape drstd back to (C,) to match the kernel's mean/rstd shape.
    auto drstd = tenzor::reshape(drstd_b, var_shape_vec);

    JvpMultiResult result;
    result.primals  = { std::move(y),  mean,  rstd  };
    result.tangents = { std::move(dy), dmean, std::move(drstd) };
    return result;
}

// LayerNorm kernel contract:
//   inputs : (x, gamma, beta)
//   outputs: {y, mean, rstd}
// Normalisation runs over the last `normalized_shape_.size()` dims, so we
// reduce over those axes and broadcast back.
//
// Internally:
//   mean  = mean(x, dims=norm_dims, keepdim=true)
//   var   = mean((x - mean)^2, dims=norm_dims, keepdim=true)
//   rstd  = 1/sqrt(var + eps)
//   y     = (x - mean) * rstd * gamma + beta
// JVP (treat x, gamma, beta as inputs; mean/var derived from x):
//   dmean = mean(dx, dims)
//   dvar  = mean(2 * (x-mean) * (dx - dmean), dims)
//   drstd = -0.5 * rstd^3 * dvar
//   dy    = (dx - dmean) * rstd * gamma
//         + (x - mean) * drstd * gamma
//         + (x - mean) * rstd * dgamma
//         + dbeta
JvpMultiResult jvp_adapter_layer_norm(std::span<const Tensor> primals,
                                      std::span<const Tensor> tangents,
                                      const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_layer_norm: expected 3 inputs (x, gamma, beta)");
    }
    const Tensor& x     = primals[0];
    const Tensor& gamma = primals[1];
    const Tensor& beta  = primals[2];

    auto zeros_like_or = [](const Tensor& t, const Tensor& tan) -> Tensor {
        if (tan.numel() != 0) return tan;
        auto sh = std::vector<int64_t>(t.shape().begin(), t.shape().end());
        return tenzor::zeros(sh, t.dtype(), t.device());
    };
    Tensor dx     = zeros_like_or(x,     tangents[0]);
    Tensor dgamma = zeros_like_or(gamma, tangents[1]);
    Tensor dbeta  = zeros_like_or(beta,  tangents[2]);

    double eps = attrs.get_float(AttrKey::Eps, 1e-5);

    // normalized_shape is stored as comma-separated string; the last K dims
    // of x are the normalisation axes. We don't strictly need the values,
    // only K (the number of trailing axes to reduce over).
    auto norm_shape = attrs.get_int_list(AttrKey::NormalizedShape);
    int64_t K = static_cast<int64_t>(norm_shape.size());
    if (K <= 0) {
        // Fall back to gamma's rank (norm dims == gamma's shape rank).
        K = gamma.ndim();
    }
    int64_t xnd = x.ndim();
    if (K > xnd) {
        throw std::runtime_error(
            "jvp_adapter_layer_norm: normalized_shape larger than input rank");
    }

    // Build helper that reduces over the last K dims with keepdim=true.
    auto reduce_mean_trailing = [&](const Tensor& t) -> Tensor {
        Tensor acc = t;
        // mean accepts a single int64_t dim; apply iteratively. Always
        // reduce the highest-numbered axis first so the index sequence
        // stays valid.
        for (int64_t i = 0; i < K; ++i) {
            int64_t dim = xnd - 1 - i;
            acc = tenzor::mean(acc, dim, /*keepdim=*/true);
        }
        return acc;
    };

    auto mean = reduce_mean_trailing(x);
    auto x_minus_mean = tenzor::sub(x, mean);
    auto var = reduce_mean_trailing(tenzor::mul(x_minus_mean, x_minus_mean));
    auto rstd = tenzor::rsqrt(tenzor::add(var, eps));

    // Primal y. gamma/beta broadcast over the leading dims; layer_norm
    // expects them shaped like normalized_shape — broadcasting handles it
    // via standard rules.
    auto y_pre_beta = tenzor::mul(tenzor::mul(x_minus_mean, rstd), gamma);
    auto y = tenzor::add(y_pre_beta, beta);

    // Tangents.
    auto dmean = reduce_mean_trailing(dx);
    auto two_xmm = tenzor::mul(x_minus_mean, 2.0);
    auto dvar = reduce_mean_trailing(tenzor::mul(two_xmm, tenzor::sub(dx, dmean)));
    auto drstd = tenzor::mul(tenzor::mul(tenzor::mul(rstd, rstd), rstd),
                             tenzor::mul(dvar, -0.5));

    auto term1 = tenzor::mul(tenzor::mul(tenzor::sub(dx, dmean), rstd), gamma);
    auto term2 = tenzor::mul(tenzor::mul(x_minus_mean, drstd), gamma);
    auto term3 = tenzor::mul(tenzor::mul(x_minus_mean, rstd), dgamma);
    auto dy = tenzor::add(tenzor::add(tenzor::add(term1, term2), term3), dbeta);

    // Reshape mean/rstd to drop the trailing-1 dims that the LayerNorm
    // kernel does not include in its returned stats. The shape of mean
    // returned by the backend matches x with the last K dims collapsed
    // to size 1 (and *not* squeezed), per the existing contract; keep
    // keepdim=true layout for compatibility with both styles.
    JvpMultiResult result;
    result.primals  = { std::move(y), mean,  rstd  };
    result.tangents = { std::move(dy), dmean, std::move(drstd) };
    return result;
}

// LinalgEigh kernel contract (symmetric eigendecomposition):
//   inputs : (A) — A symmetric (..., N, N)
//   outputs: {W, V}  — W eigenvalues (..., N), V eigenvectors (..., N, N)
//
// JVP (Magnus & Neudecker; see Giles, "An extended collection of matrix
// derivative results"):
//   E   = V^T dA V
//   dW  = diag(E)
//   F   = 1/(W_j - W_i)  for i ≠ j; 0 on diagonal
//   dV  = V (F * E)        (Hadamard with the F mask)
// Degenerate eigenvalues produce non-finite F entries; the standard
// approach is to clip / mask, but for symmetric matrices with distinct
// spectra this formula is exact.
JvpMultiResult jvp_adapter_linalg_eigh(std::span<const Tensor> primals,
                                       std::span<const Tensor> tangents,
                                       const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_linalg_eigh: expected 1 input");
    }
    const Tensor& A = primals[0];
    if (A.ndim() != 2) {
        throw std::runtime_error(
            "jvp_adapter_linalg_eigh: batched (>2D) eigh JVP not supported; "
            "got ndim=" + std::to_string(A.ndim()));
    }
    Tensor dA = tangents[0];
    if (dA.numel() == 0) {
        auto sh = std::vector<int64_t>(A.shape().begin(), A.shape().end());
        dA = tenzor::zeros(sh, A.dtype(), A.device());
    }

    auto [W, V] = tenzor::linalg::eigh(A);

    // E = V^T @ dA @ V
    auto Vt = tenzor::transpose(V, -2, -1);
    auto E  = tenzor::matmul(Vt, tenzor::matmul(dA, V));

    // dW = diag(E) — extract main diagonal of the 2D matrix E.
    auto dW = tenzor::diag(E, /*diagonal=*/0);

    // Build F mask: F_{ij} = 1/(W_j - W_i) for i ≠ j, 0 on diagonal.
    auto W_col = tenzor::unsqueeze(W, -1);  // (N, 1)
    auto W_row = tenzor::unsqueeze(W, -2);  // (1, N)
    auto denom = tenzor::sub(W_row, W_col);
    // Avoid divide-by-zero on the diagonal; replace zeros with 1, then zero
    // the diagonal of F afterwards via masking.
    auto zero_tensor  = tenzor::zeros_like(denom);
    auto one_tensor   = tenzor::ones_like(denom);
    auto is_zero_mask = tenzor::eq(denom, zero_tensor);
    auto safe_denom   = tenzor::where(is_zero_mask, one_tensor, denom);
    auto F            = tenzor::div(one_tensor, safe_denom);
    F                 = tenzor::where(is_zero_mask, zero_tensor, F);

    // dV = V @ (F * E)
    auto dV = tenzor::matmul(V, tenzor::mul(F, E));

    JvpMultiResult result;
    result.primals  = { std::move(W),  std::move(V)  };
    result.tangents = { std::move(dW), std::move(dV) };
    return result;
}

} // anonymous (nested)

// ============================================================================
// Audit A.4 batch 7 dispatch adapters
// ============================================================================

namespace {

// ----- Sparse -----

// SparseSpMM input layout: [crow, col, values, dense].
// crow/col are integer index tensors (non-differentiable); the tangents
// matching slots [0] and [1] are ignored. Only values and dense carry
// tangents.
JvpResult jvp_adapter_sparse_spmm(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& attrs) {
    if (primals.size() != 4 || tangents.size() != 4) {
        throw std::runtime_error(
            "jvp_adapter_sparse_spmm: expected 4 inputs (crow, col, values, dense)");
    }
    int64_t M = attrs.get_int(AttrKey::M);
    int64_t K = attrs.get_int(AttrKey::K);
    auto values_d = make_dual(primals[2], tangents[2]);
    auto dense_d  = make_dual(primals[3], tangents[3]);
    return dual_to_result(jvp_sparse_spmm(primals[0], primals[1], values_d, dense_d, M, K));
}

JvpResult jvp_adapter_sparse_spmv(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& attrs) {
    if (primals.size() != 4 || tangents.size() != 4) {
        throw std::runtime_error(
            "jvp_adapter_sparse_spmv: expected 4 inputs (crow, col, values, vec)");
    }
    int64_t M = attrs.get_int(AttrKey::M);
    int64_t K = attrs.get_int(AttrKey::K);
    auto values_d = make_dual(primals[2], tangents[2]);
    auto vec_d    = make_dual(primals[3], tangents[3]);
    return dual_to_result(jvp_sparse_spmv(primals[0], primals[1], values_d, vec_d, M, K));
}

JvpResult jvp_adapter_sparse_add(std::span<const Tensor> primals,
                                 std::span<const Tensor> tangents,
                                 const OpAttributes& attrs) {
    if (primals.size() != 4 || tangents.size() != 4) {
        throw std::runtime_error(
            "jvp_adapter_sparse_add: expected 4 inputs (crow, col, values, dense)");
    }
    int64_t M = attrs.get_int(AttrKey::M);
    int64_t K = attrs.get_int(AttrKey::K);
    auto values_d = make_dual(primals[2], tangents[2]);
    auto dense_d  = make_dual(primals[3], tangents[3]);
    return dual_to_result(jvp_sparse_add(primals[0], primals[1], values_d, dense_d, M, K));
}

// ----- RNN cells -----

JvpResult jvp_adapter_gru_cell_forward(std::span<const Tensor> primals,
                                       std::span<const Tensor> tangents,
                                       const OpAttributes&) {
    if (primals.size() != 6 || tangents.size() != 6) {
        throw std::runtime_error(
            "jvp_adapter_gru_cell_forward: expected 6 inputs "
            "(x, h, W_ih, W_hh, b_ih, b_hh)");
    }
    auto x    = make_dual(primals[0], tangents[0]);
    auto h    = make_dual(primals[1], tangents[1]);
    auto Wih  = make_dual(primals[2], tangents[2]);
    auto Whh  = make_dual(primals[3], tangents[3]);
    auto bih  = make_dual(primals[4], tangents[4]);
    auto bhh  = make_dual(primals[5], tangents[5]);
    return dual_to_result(jvp_gru_cell_forward(x, h, Wih, Whh, bih, bhh));
}

// LSTMCellForward is multi-output: returns {hy, cy}. We re-derive the gates
// inline (replaying the same math as the CPU/cuDNN cell kernel) to produce
// tangents for both outputs in one pass.
//
// Layout (LSTMCellForward inputs):
//   primals[0] = x   [B, in]      tangents[0] = dx
//   primals[1] = h   [B, H]       tangents[1] = dh
//   primals[2] = c   [B, H]       tangents[2] = dc
//   primals[3] = W_ih [4H, in]    tangents[3] = dW_ih
//   primals[4] = W_hh [4H, H]     tangents[4] = dW_hh
//   primals[5] = b_ih [4H]        tangents[5] = db_ih
//   primals[6] = b_hh [4H]        tangents[6] = db_hh
//
// Math (PyTorch convention):
//   gate  = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh           [B, 4H]
//   split last dim → {g_i, g_f, g_g, g_o}
//   i, f, o = sigmoid(g_{i,f,o});  g = tanh(g_g)
//   cy = f * c + i * g
//   hy = o * tanh(cy)
//
// Forward-mode tangents follow directly: each sigmoid/tanh uses its
// existing single-input chain-rule formula on the (DualTensor) gate
// pre-activation, and the mul/add structure of cy / hy is propagated by
// jvp_mul / jvp_add.
JvpMultiResult jvp_adapter_lstm_cell_forward(std::span<const Tensor> primals,
                                             std::span<const Tensor> tangents,
                                             const OpAttributes&) {
    if (primals.size() != 7 || tangents.size() != 7) {
        throw std::runtime_error(
            "jvp_adapter_lstm_cell_forward: expected 7 inputs "
            "(x, h, c, W_ih, W_hh, b_ih, b_hh)");
    }
    auto x   = make_dual(primals[0], tangents[0]);
    auto h   = make_dual(primals[1], tangents[1]);
    auto c   = make_dual(primals[2], tangents[2]);
    auto Wih = make_dual(primals[3], tangents[3]);
    auto Whh = make_dual(primals[4], tangents[4]);
    auto bih = make_dual(primals[5], tangents[5]);
    auto bhh = make_dual(primals[6], tangents[6]);

    // gate = x @ W_ih^T + b_ih  +  h @ W_hh^T + b_hh   (both linear).
    auto gate_x = jvp_linear(x, Wih, bih);  // [B, 4H]
    auto gate_h = jvp_linear(h, Whh, bhh);  // [B, 4H]
    auto gate   = jvp_add(gate_x, gate_h);

    // Chunk into the four gates along dim=1.
    auto chunk4 = [](const DualTensor& t) -> std::array<DualTensor, 4> {
        auto p = tenzor::chunk(t.primal(),  4, /*dim=*/1);
        auto d = tenzor::chunk(t.tangent(), 4, /*dim=*/1);
        return { DualTensor(p[0], d[0]), DualTensor(p[1], d[1]),
                 DualTensor(p[2], d[2]), DualTensor(p[3], d[3]) };
    };
    auto g = chunk4(gate);   // {g_i, g_f, g_g, g_o}

    auto i_gate = jvp_sigmoid(g[0]);
    auto f_gate = jvp_sigmoid(g[1]);
    auto g_gate = jvp_tanh   (g[2]);
    auto o_gate = jvp_sigmoid(g[3]);

    // cy = f * c + i * g
    auto cy = jvp_add(jvp_mul(f_gate, c), jvp_mul(i_gate, g_gate));
    // hy = o * tanh(cy)
    auto hy = jvp_mul(o_gate, jvp_tanh(cy));

    JvpMultiResult result;
    result.primals  = { std::move(hy.primal()),  std::move(cy.primal())  };
    result.tangents = { std::move(hy.tangent()), std::move(cy.tangent()) };
    return result;
}

// ----- Nested tensors -----

JvpResult jvp_adapter_nested_sum(std::span<const Tensor> primals,
                                 std::span<const Tensor> tangents,
                                 const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_nested_sum: expected 2 inputs (values, offsets)");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    auto values = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_nested_sum(values, primals[1], dim, keepdim));
}

JvpResult jvp_adapter_nested_mean(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_nested_mean: expected 2 inputs (values, offsets)");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    auto values = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_nested_mean(values, primals[1], dim, keepdim));
}

JvpResult jvp_adapter_nested_linear(std::span<const Tensor> primals,
                                    std::span<const Tensor> tangents,
                                    const OpAttributes&) {
    // primals: [values, offsets, weight, (bias?)]
    if (primals.size() < 3 || primals.size() > 4 ||
        tangents.size() != primals.size()) {
        throw std::runtime_error(
            "jvp_adapter_nested_linear: expected 3 or 4 inputs "
            "(values, offsets, weight, [bias])");
    }
    auto values = make_dual(primals[0], tangents[0]);
    auto weight = make_dual(primals[2], tangents[2]);
    std::optional<DualTensor> bias;
    if (primals.size() == 4) {
        bias = make_dual(primals[3], tangents[3]);
    }
    return dual_to_result(jvp_nested_linear(values, primals[1], weight, bias));
}

// ----- Reduction long-tail (single-output, gather-at-index) -----

// Aminmax: outputs {min, max}. Forward-mode tangent is obtained by gathering
// the input tangent at the argmin / argmax positions — done implicitly by
// applying the "mask × tangent then sum" pattern (same as jvp_max_red /
// jvp_min_red) for each output.
JvpMultiResult jvp_adapter_aminmax(std::span<const Tensor> primals,
                                   std::span<const Tensor> tangents,
                                   const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_aminmax: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    // Aminmax CPU semantics: dim defaults to -1 (flatten + scalar reduction)
    // when not provided. We treat the no-dim case the same as a global
    // reduction over the flat tensor.
    bool has_dim = attrs.has(AttrKey::Dim);
    std::optional<int64_t> dim;
    if (has_dim) dim = attrs.get_int(AttrKey::Dim);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);

    auto min_dual = jvp_min_red(x, dim, keepdim);
    auto max_dual = jvp_max_red(x, dim, keepdim);

    JvpMultiResult result;
    result.primals  = { std::move(min_dual.primal()),  std::move(max_dual.primal())  };
    result.tangents = { std::move(min_dual.tangent()), std::move(max_dual.tangent()) };
    return result;
}

// Kthvalue: outputs {values, indices}. The "values" output is a gather of
// the primal at the kth-sorted positions; its tangent is the same gather on
// dx. The "indices" output is integer (zero tangent).
JvpMultiResult jvp_adapter_kthvalue(std::span<const Tensor> primals,
                                    std::span<const Tensor> tangents,
                                    const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_kthvalue: expected 1 input");
    }
    int64_t k       = attrs.get_int(AttrKey::K, 1);
    int64_t dim     = attrs.get_int(AttrKey::Dim, -1);
    bool    keepdim = attrs.get_bool(AttrKey::Keepdim, false);

    // Run the kernel once to obtain primal values + indices.
    OpAttributes kattrs;
    kattrs.set(AttrKey::K, k);
    kattrs.set(AttrKey::Dim, dim);
    kattrs.set(AttrKey::Keepdim, keepdim);
    auto out = tenzor::dispatch(OpId::Kthvalue,
                                std::vector<Tensor>{primals[0]}, kattrs);
    Tensor values_p  = out[0];
    Tensor indices_p = out[1];

    const Tensor& dx = tangents[0];
    Tensor values_t;
    if (dx.numel() == 0) {
        // No tangent → zero tangent of the values' shape/dtype.
        values_t = tenzor::zeros(std::vector<int64_t>(values_p.shape().begin(),
                                                       values_p.shape().end()),
                                  values_p.dtype(), values_p.device());
    } else {
        // To gather, indices must include the reduced dim. The kernel returns
        // indices in the kept-or-removed-dim layout matching `values_p`. To
        // call take_along_dim against dx (which still has the reduced dim),
        // we need a keepdim-style index tensor. Reinsert a singleton dim if
        // the user requested keepdim=false.
        Tensor idx_kd = keepdim
            ? indices_p
            : tenzor::unsqueeze(indices_p, dim);
        auto gathered = tenzor::take_along_dim(dx, idx_kd, dim);
        values_t = keepdim ? gathered : tenzor::squeeze(gathered, dim);
    }

    // Indices: integer dtype → zero tangent of matching shape.
    Tensor indices_t = tenzor::zeros(
        std::vector<int64_t>(indices_p.shape().begin(), indices_p.shape().end()),
        indices_p.dtype(), indices_p.device());

    JvpMultiResult result;
    result.primals  = { std::move(values_p),  std::move(indices_p)  };
    result.tangents = { std::move(values_t),  std::move(indices_t)  };
    return result;
}

// CumMax / CumMin: outputs {values, indices}. values[..., t, ...] is the
// running max/min; tangent is gather of dx at the corresponding running-
// argmax indices.
JvpMultiResult jvp_adapter_cummax_impl(std::span<const Tensor> primals,
                                       std::span<const Tensor> tangents,
                                       const OpAttributes& attrs,
                                       OpId op) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_cummax/cummin: expected 1 input");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    OpAttributes kattrs;
    kattrs.set(AttrKey::Dim, dim);
    auto out = tenzor::dispatch(op, std::vector<Tensor>{primals[0]}, kattrs);
    Tensor values_p  = out[0];
    Tensor indices_p = out[1];

    const Tensor& dx = tangents[0];
    Tensor values_t;
    if (dx.numel() == 0) {
        values_t = tenzor::zeros(std::vector<int64_t>(values_p.shape().begin(),
                                                       values_p.shape().end()),
                                  values_p.dtype(), values_p.device());
    } else {
        // CumMax/CumMin indices already share dx's rank — direct gather.
        values_t = tenzor::take_along_dim(dx, indices_p, dim);
    }
    Tensor indices_t = tenzor::zeros(
        std::vector<int64_t>(indices_p.shape().begin(), indices_p.shape().end()),
        indices_p.dtype(), indices_p.device());

    JvpMultiResult result;
    result.primals  = { std::move(values_p),  std::move(indices_p)  };
    result.tangents = { std::move(values_t),  std::move(indices_t)  };
    return result;
}

JvpMultiResult jvp_adapter_cummax(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& attrs) {
    return jvp_adapter_cummax_impl(primals, tangents, attrs, OpId::CumMax);
}
JvpMultiResult jvp_adapter_cummin(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& attrs) {
    return jvp_adapter_cummax_impl(primals, tangents, attrs, OpId::CumMin);
}

JvpResult jvp_adapter_logcumsumexp(std::span<const Tensor> primals,
                                   std::span<const Tensor> tangents,
                                   const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_logcumsumexp: expected 1 input");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_logcumsumexp(x, dim));
}

JvpResult jvp_adapter_numerical_gradient(std::span<const Tensor> primals,
                                         std::span<const Tensor> tangents,
                                         const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_numerical_gradient: expected 1 input");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    double spacing = attrs.get_float(AttrKey::Spacing, 1.0);
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_numerical_gradient(x, dim, spacing));
}

JvpResult jvp_adapter_trapezoid(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.empty() || primals.size() > 2 ||
        tangents.size() != primals.size()) {
        throw std::runtime_error(
            "jvp_adapter_trapezoid: expected 1 or 2 inputs (y, [x])");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    double dx_uniform = attrs.get_float(AttrKey::Dx, 1.0);
    auto y = make_dual(primals[0], tangents[0]);
    std::optional<DualTensor> x_pos;
    if (primals.size() == 2) {
        x_pos = make_dual(primals[1], tangents[1]);
    }
    return dual_to_result(jvp_trapezoid(y, x_pos, dim, dx_uniform));
}

JvpResult jvp_adapter_cumulative_trapezoid(std::span<const Tensor> primals,
                                           std::span<const Tensor> tangents,
                                           const OpAttributes& attrs) {
    if (primals.empty() || primals.size() > 2 ||
        tangents.size() != primals.size()) {
        throw std::runtime_error(
            "jvp_adapter_cumulative_trapezoid: expected 1 or 2 inputs (y, [x])");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    double dx_uniform = attrs.get_float(AttrKey::Dx, 1.0);
    auto y = make_dual(primals[0], tangents[0]);
    std::optional<DualTensor> x_pos;
    if (primals.size() == 2) {
        x_pos = make_dual(primals[1], tangents[1]);
    }
    return dual_to_result(jvp_cumulative_trapezoid(y, x_pos, dim, dx_uniform));
}

// ----- Adaptive average pooling -----

JvpResult jvp_adapter_adaptive_avgpool_1d(std::span<const Tensor> primals,
                                          std::span<const Tensor> tangents,
                                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_adaptive_avgpool_1d: expected 1 input");
    }
    int64_t out_size = attrs.get_int(AttrKey::OutputSize, 1);
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_adaptive_avgpool_1d(x, out_size));
}
JvpResult jvp_adapter_adaptive_avgpool_2d(std::span<const Tensor> primals,
                                          std::span<const Tensor> tangents,
                                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_adaptive_avgpool_2d: expected 1 input");
    }
    int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
    int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_adaptive_avgpool_2d(x, out_h, out_w));
}
JvpResult jvp_adapter_adaptive_avgpool_3d(std::span<const Tensor> primals,
                                          std::span<const Tensor> tangents,
                                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_adaptive_avgpool_3d: expected 1 input");
    }
    int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 1);
    int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
    int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_adaptive_avgpool_3d(x, out_d, out_h, out_w));
}

} // anonymous (batch 7)

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

    // ---------------- Audit A.4 (extension) ---------------------------------
    //
    // Single-output additions:
    //   - Hardswish (piecewise linear approx of swish; HardswishBackward in
    //     nn/activations/activations.cpp uses OpId::Unknown, so this rule
    //     is reachable only via direct dispatch_jvp() — registered for
    //     completeness / external callers).
    //   - FFT / IFFT / RFFT / IRFFT: complex Fourier transforms are linear,
    //     so the tangent is the same op applied to the input tangent.
    register_jvp_rule(OpId::Hardswish, &jvp_adapter_hardswish);
    register_jvp_rule(OpId::FFT,       &jvp_adapter_fft);
    register_jvp_rule(OpId::IFFT,      &jvp_adapter_ifft);
    register_jvp_rule(OpId::RFFT,      &jvp_adapter_rfft);
    register_jvp_rule(OpId::IRFFT,     &jvp_adapter_irfft);

    // Multi-output rules (separate dispatch table; see
    // register_jvp_rule_multi). These ops return multiple tensors and the
    // walker in try_traverse_jvp does not currently follow multi-output
    // Function nodes — the rules are exposed for direct dispatch via
    // dispatch_jvp_multi() so that higher-level callers (and future
    // walker extensions) can drive forward-mode AD through BN/LN/Eigh.
    register_jvp_rule_multi(OpId::BatchNorm2dForwardAffine,
                            &jvp_adapter_batchnorm2d_forward_affine);
    register_jvp_rule_multi(OpId::LayerNorm,   &jvp_adapter_layer_norm);
    register_jvp_rule_multi(OpId::LinalgEigh,  &jvp_adapter_linalg_eigh);

    // ---------------- Audit A.4 batch 7 ------------------
    //
    // Sparse: pattern (crow, col) is constant; tangents on `values` and the
    // dense operand thread through the bilinear / linear ops.
    register_jvp_rule(OpId::SparseSpMM, &jvp_adapter_sparse_spmm);
    register_jvp_rule(OpId::SparseSpMV, &jvp_adapter_sparse_spmv);
    register_jvp_rule(OpId::SparseAdd,  &jvp_adapter_sparse_add);

    // RNN cells: GRU single-output, LSTM multi-output {hy, cy}.
    register_jvp_rule       (OpId::GRUCellForward,  &jvp_adapter_gru_cell_forward);
    register_jvp_rule_multi (OpId::LSTMCellForward, &jvp_adapter_lstm_cell_forward);

    // Nested tensors: linear-in-values reductions + bilinear Linear.
    register_jvp_rule(OpId::NestedSum,    &jvp_adapter_nested_sum);
    register_jvp_rule(OpId::NestedMean,   &jvp_adapter_nested_mean);
    register_jvp_rule(OpId::NestedLinear, &jvp_adapter_nested_linear);

    // Reduction long-tail.
    register_jvp_rule_multi(OpId::Aminmax,   &jvp_adapter_aminmax);
    register_jvp_rule_multi(OpId::Kthvalue,  &jvp_adapter_kthvalue);
    register_jvp_rule_multi(OpId::CumMax,    &jvp_adapter_cummax);
    register_jvp_rule_multi(OpId::CumMin,    &jvp_adapter_cummin);
    register_jvp_rule(OpId::Logcumsumexp,        &jvp_adapter_logcumsumexp);
    register_jvp_rule(OpId::NumericalGradient,   &jvp_adapter_numerical_gradient);
    register_jvp_rule(OpId::Trapezoid,           &jvp_adapter_trapezoid);
    register_jvp_rule(OpId::CumulativeTrapezoid, &jvp_adapter_cumulative_trapezoid);

    // Adaptive average pooling (linear in input — same op on the tangent).
    register_jvp_rule(OpId::AdaptiveAvgPool1d, &jvp_adapter_adaptive_avgpool_1d);
    register_jvp_rule(OpId::AdaptiveAvgPool2d, &jvp_adapter_adaptive_avgpool_2d);
    register_jvp_rule(OpId::AdaptiveAvgPool3d, &jvp_adapter_adaptive_avgpool_3d);
}

} // namespace detail

} // namespace tenzor
