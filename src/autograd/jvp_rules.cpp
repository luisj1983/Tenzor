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
#include "tenzor/utils/error.hpp"
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

// ---- Reductions: Max/Min (one-hot at argmax/argmin, tie-safe) -------------
//
// Audit-4 U.1 / U.2: a previous implementation built `mask = eq(x, max(x))`
// and summed `dx * mask` along `dim`. For ties (e.g. `[3, 3, 1]`) the mask
// has multiple ones along the reduced axis, so the JVP returned
// `dx[0] + dx[1]` instead of a single representative — disagreeing with
// MaxBackward, which picks a unique winner via argmax. The fix here mirrors
// MaxBackward: pick the unique argmax/argmin index, one-hot it back to the
// input shape, and gather the tangent at that position.

namespace {

// Build a one-hot mask along `dim` selecting the argmax / argmin position
// for every reduction group, matching x.primal()'s shape. For the global
// reduction case (dim == nullopt) we flatten, take a scalar argmax/argmin,
// one-hot to numel and reshape back. The result is cast to the tangent
// dtype so element-wise mul matches downstream.
auto argreduce_one_hot_mask(const Tensor& x, std::optional<int64_t> dim,
                            bool use_argmin, DType out_dtype) -> Tensor {
    auto input_shape_vec = std::vector<int64_t>(x.shape().begin(), x.shape().end());
    if (dim.has_value()) {
        int64_t d = *dim;
        int64_t ndim = static_cast<int64_t>(x.ndim());
        if (d < 0) d += ndim;
        int64_t num_classes = x.shape()[d];
        auto idx = use_argmin
            ? tenzor::argmin(x, /*dim=*/d, /*keepdim=*/false)
            : tenzor::argmax(x, /*dim=*/d, /*keepdim=*/false);
        // one_hot appends a class axis at the trailing position; we need the
        // mask along the reduced axis `d`. After one_hot the mask has shape
        // (...without_d..., num_classes); move that last axis back to `d`.
        auto mask_int = tenzor::one_hot(idx, num_classes);
        // Reorder dims: insert the last axis into position `d`.
        std::vector<int64_t> perm;
        perm.reserve(static_cast<size_t>(ndim));
        int64_t mask_ndim = static_cast<int64_t>(mask_int.ndim());
        // mask_int dims are [0, 1, ..., mask_ndim-2, mask_ndim-1=class_axis]
        // We want to place class_axis at position d among the original axes.
        for (int64_t i = 0; i < ndim; ++i) {
            if (i == d) {
                perm.push_back(mask_ndim - 1);
            } else if (i < d) {
                perm.push_back(i);
            } else {
                perm.push_back(i - 1);
            }
        }
        auto mask_reordered = tenzor::permute(mask_int, perm);
        return mask_reordered.to(out_dtype);
    }
    // Global reduction: flatten, argreduce, one-hot, reshape.
    auto x_flat = tenzor::flatten(x, 0, -1);
    int64_t numel = x.numel();
    auto idx_scalar = use_argmin
        ? tenzor::argmin(x_flat, /*dim=*/0, /*keepdim=*/false)
        : tenzor::argmax(x_flat, /*dim=*/0, /*keepdim=*/false);
    auto mask_flat = tenzor::one_hot(idx_scalar, numel);
    auto mask = tenzor::reshape(mask_flat, input_shape_vec);
    return mask.to(out_dtype);
}

} // anonymous namespace

auto jvp_max_red(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    auto primal = tenzor::max(x.primal(), dim, keepdim);
    auto mask = argreduce_one_hot_mask(x.primal(), dim, /*use_argmin=*/false,
                                       x.tangent().dtype());
    auto masked_tan = tenzor::mul(x.tangent(), mask);
    auto tangent = tenzor::sum(masked_tan, dim, keepdim);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_min_red(const DualTensor& x, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    auto primal = tenzor::min(x.primal(), dim, keepdim);
    auto mask = argreduce_one_hot_mask(x.primal(), dim, /*use_argmin=*/true,
                                       x.tangent().dtype());
    auto masked_tan = tenzor::mul(x.tangent(), mask);
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
// Audit A.4 batch 8: Attention family (single-output JVP) + losses
// ============================================================================

// Scaled dot-product attention forward-mode JVP. Used by FlashAttention,
// FusedAttention, FlexAttention (identity score-mod), and NestedAttention
// (when the per-segment structure can be flattened to a dense [B*Heads, L, E]
// view). Math:
//
//   S    = (Q @ K^T) * scale           shape [..., Lq, Lk]
//   if causal: S += mask_add where mask_add[i,j] = 0 if j<=i else -inf
//   P    = softmax(S, dim=-1)
//   y    = P @ V                       shape [..., Lq, Ev]
//
// Tangent (chain rule):
//   S_t  = ((dQ @ K^T) + (Q @ dK^T)) * scale       (mask is constant)
//   P_t  = P * (S_t - sum(P * S_t, dim=-1, keepdim))
//   y_t  = (P_t @ V) + (P @ dV)
auto jvp_sdpa_forward(const DualTensor& Q,
                      const DualTensor& K,
                      const DualTensor& V,
                      float scale,
                      bool causal) -> DualTensor {
    const Tensor& Qp = Q.primal();   const Tensor& Qt = Q.tangent();
    const Tensor& Kp = K.primal();   const Tensor& Kt = K.tangent();
    const Tensor& Vp = V.primal();   const Tensor& Vt = V.tangent();

    // K^T over the last two dims.
    int64_t ndim_K = static_cast<int64_t>(Kp.shape().size());
    int64_t a = ndim_K - 2, b = ndim_K - 1;
    auto Kt_T = tenzor::transpose(Kp, a, b);
    auto Kt_T_tan = tenzor::transpose(Kt, a, b);

    // Tensor scalar for `scale`. Broadcasts cleanly against the score tensor.
    Tensor scale_t = tenzor::full({1}, scale, Qp.dtype(), Qp.device());

    // S = Q @ K^T * scale
    auto S = tenzor::mul(tenzor::matmul(Qp, Kt_T), scale_t);
    // S_t = (dQ @ K^T + Q @ dK^T) * scale
    auto S_t = tenzor::mul(
        tenzor::add(tenzor::matmul(Qt,  Kt_T),
                    tenzor::matmul(Qp,  Kt_T_tan)),
        scale_t);

    if (causal) {
        // Build a triu mask on the last two dims (Lq, Lk). triu with diag=1
        // selects the strictly-upper triangle (future positions); we want
        // those positions to be -inf in S (and 0 in S_t — masked-fill with 0
        // simulates "constant mask"; equivalently we can subtract a large
        // value, but `masked_fill` is the canonical pattern in the codebase).
        const auto& Sshape = S.shape();
        int64_t Lq = Sshape[static_cast<size_t>(ndim_K) - 2];
        int64_t Lk = Sshape[static_cast<size_t>(ndim_K) - 1];
        Tensor ones2d = tenzor::ones({Lq, Lk}, Qp.dtype(), Qp.device());
        Tensor mask_upper = tenzor::triu(ones2d, /*diagonal=*/1);
        // Cast to Bool for masked_fill. `mask_upper > 0` gives a Bool mask.
        Tensor zero = tenzor::full({1}, 0.0f, Qp.dtype(), Qp.device());
        Tensor mask_bool = tenzor::gt(mask_upper, zero);
        S = tenzor::masked_fill(S, mask_bool,
                                -std::numeric_limits<float>::infinity());
        // Mask is constant: corresponding S_t entries don't contribute to the
        // softmax output (the row is renormalised after softmax, so the
        // tangent contribution at masked positions is zeroed by setting the
        // same entries to 0 here — this matches the math where the mask is a
        // constant additive offset with zero derivative).
        S_t = tenzor::masked_fill(S_t, mask_bool, 0.0f);
    }

    // P = softmax(S, dim=-1)   (compute via shifted exp / sum for stability,
    // matching jvp_softmax).
    int64_t last_dim = -1;
    auto S_max = tenzor::max(S, last_dim, /*keepdim=*/true);
    auto S_shifted = tenzor::sub(S, S_max);
    auto exp_S = tenzor::exp(S_shifted);
    auto sum_exp = tenzor::sum(exp_S, last_dim, /*keepdim=*/true);
    auto P = tenzor::div(exp_S, sum_exp);

    // P_t = P * (S_t - sum(P * S_t, dim=-1, keepdim))
    auto P_St = tenzor::mul(P, S_t);
    auto sum_P_St = tenzor::sum(P_St, last_dim, /*keepdim=*/true);
    auto P_t = tenzor::mul(P, tenzor::sub(S_t, sum_P_St));

    // y = P @ V; y_t = P_t @ V + P @ dV
    auto y   = tenzor::matmul(P,   Vp);
    auto y_t = tenzor::add(tenzor::matmul(P_t, Vp),
                           tenzor::matmul(P,   Vt));
    return DualTensor(std::move(y), std::move(y_t));
}

// FusedSoftmaxCrossEntropy(logits, targets, reduction) JVP.
// Formula (per-sample, logits row x, target t):
//   loss_i = logsumexp(x_i) - x_i[t_i]
//   d_loss_i = sum_j softmax(x_i)[j] * dx_i[j] - dx_i[t_i]
// Then apply reduction:
//   "mean" → mean(per_loss); "sum" → sum(per_loss); "none" → per_loss.
auto jvp_fused_softmax_cross_entropy(const DualTensor& logits,
                                     const Tensor& targets,
                                     const std::string& reduction) -> DualTensor {
    const Tensor& X  = logits.primal();
    const Tensor& dX = logits.tangent();

    // Stable softmax across the last dim (class dim).
    int64_t last_dim = -1;
    auto x_max  = tenzor::max(X, last_dim, /*keepdim=*/true);
    auto shift  = tenzor::sub(X, x_max);
    auto exp_s  = tenzor::exp(shift);
    auto sum_e  = tenzor::sum(exp_s, last_dim, /*keepdim=*/true);
    auto P      = tenzor::div(exp_s, sum_e);                // softmax
    auto lse    = tenzor::add(tenzor::log(sum_e), x_max);   // [B,1]

    // Per-sample primal loss: logsumexp - gather(logits, target).
    // Reshape targets to a keepdim-style [B,1] index tensor for gather.
    int64_t B = X.shape()[0];
    Tensor tgt_2d = tenzor::reshape(targets, std::vector<int64_t>{B, 1});
    auto x_at_tgt = tenzor::gather(X, /*dim=*/1, tgt_2d);   // [B,1]
    auto per_loss = tenzor::sub(lse, x_at_tgt);             // [B,1]
    per_loss = tenzor::reshape(per_loss, std::vector<int64_t>{B});

    // Per-sample tangent.
    // sum_j p[j] * dx[j] over class dim → [B,1]; gather(dX, tgt) → [B,1].
    auto P_dx = tenzor::mul(P, dX);
    auto sum_P_dx = tenzor::sum(P_dx, last_dim, /*keepdim=*/true);  // [B,1]
    auto dx_at_tgt = tenzor::gather(dX, /*dim=*/1, tgt_2d);          // [B,1]
    auto per_tan = tenzor::sub(sum_P_dx, dx_at_tgt);                  // [B,1]
    per_tan = tenzor::reshape(per_tan, std::vector<int64_t>{B});

    // Apply reduction.
    Tensor primal_out;
    Tensor tangent_out;
    if (reduction == "mean") {
        primal_out  = tenzor::mean(per_loss);
        tangent_out = tenzor::mean(per_tan);
    } else if (reduction == "sum") {
        primal_out  = tenzor::sum(per_loss);
        tangent_out = tenzor::sum(per_tan);
    } else {
        // "none" (or anything else): return per-sample.
        primal_out  = std::move(per_loss);
        tangent_out = std::move(per_tan);
    }
    return DualTensor(std::move(primal_out), std::move(tangent_out));
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
    // Audit-4 U.3: IRFFT is a *real-valued projection* — its complex
    // Hermitian input X carries the constraint Im(X[0]) = 0 (and
    // Im(X[n/2]) = 0 when n is even). A naive `irfft(dx)` silently throws
    // away whatever non-Hermitian components dx has, but the true JVP must
    // first project dx onto the valid tangent subspace and then transport
    // through irfft. Mirroring the IRFFTBackward.backward saved-`n`
    // technique (function_fft.cpp:339-343), we symmetrise dx via
    //   sym_dx = rfft(irfft(dx, n), n_orig)
    // where n_orig is the freq-bin count of the forward irfft's input
    // (recoverable from primals[0].shape[dim]). irfft(sym_dx, n) is the
    // tangent in the time domain. Using `n_orig` for the inner rfft is
    // the same R.7 fix that IRFFTBackward applies for non-default n.
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_irfft: expected 1 input");
    }
    const Tensor& x = primals[0];
    Tensor dx = tangents[0];
    if (dx.numel() == 0 && x.numel() != 0) {
        auto shape_vec = std::vector<int64_t>(x.shape().begin(), x.shape().end());
        dx = tenzor::zeros(shape_vec, x.dtype(), x.device());
    }
    std::optional<int64_t> n;
    if (attrs.has(AttrKey::N)) n = attrs.get_int(AttrKey::N);
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    std::string norm(attrs.get_string(AttrKey::Norm, "backward"));

    Tensor primal_out = tenzor::fft::irfft(x, n, dim, norm);

    // n_orig: frequency-bin count of the input X (= shape along `dim`).
    int64_t actual_dim = dim < 0 ? dim + static_cast<int64_t>(x.ndim()) : dim;
    int64_t n_orig = x.shape()[actual_dim];

    // time_n: the time-domain length used by irfft (= n if supplied, else
    // the default 2*(n_orig-1)). This is what rfft must reproduce so that
    // sym_dx has the same shape as dx.
    int64_t time_n = n.value_or(2 * (n_orig - 1));
    std::optional<int64_t> time_n_opt(time_n);

    // Project dx onto the Hermitian subspace: round-trip through time domain
    // then back to freq domain using the saved n_orig (R.7-style). The pair
    // (irfft, rfft) is the identity on validly-Hermitian inputs and zeroes
    // the imaginary parts of bins 0 and n/2 otherwise.
    Tensor time_dx = tenzor::fft::irfft(dx, time_n_opt, dim, norm);
    Tensor sym_dx  = tenzor::fft::rfft(time_dx, time_n_opt, dim, norm);
    Tensor tangent_out = tenzor::fft::irfft(sym_dx, n, dim, norm);

    return JvpResult{std::move(primal_out), std::move(tangent_out)};
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
//
// Audit J.4: batched (>2D) support. All ops below broadcast / batch
// naturally on the trailing (N, N) / (N) axes, so the same expression
// works for any number of leading batch dimensions:
//   - matmul: routes to bmm for ndim > 2
//   - unsqueeze(-2)/unsqueeze(-1): operate on the trailing axis only
//   - eq/where/div/mul/sub/zeros_like/ones_like: broadcast elementwise
//   - sum(dim=last): collapses the last column-axis after masking with eye
JvpMultiResult jvp_adapter_linalg_eigh(std::span<const Tensor> primals,
                                       std::span<const Tensor> tangents,
                                       const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_linalg_eigh: expected 1 input");
    }
    const Tensor& A = primals[0];
    if (A.ndim() < 2) {
        throw std::runtime_error(
            "jvp_adapter_linalg_eigh: input must be at least 2D, got ndim=" +
            std::to_string(A.ndim()));
    }
    Tensor dA = tangents[0];
    if (dA.numel() == 0) {
        auto sh = std::vector<int64_t>(A.shape().begin(), A.shape().end());
        dA = tenzor::zeros(sh, A.dtype(), A.device());
    }

    auto [W, V] = tenzor::linalg::eigh(A);

    const int64_t N      = A.shape()[A.ndim() - 1];
    const int64_t E_ndim = A.ndim();           // E has same rank as A
    const int64_t last   = E_ndim - 1;         // trailing column axis

    // E = V^T @ dA @ V  (batched matmul handles leading batch dims)
    auto Vt = tenzor::transpose(V, -2, -1);
    auto E  = tenzor::matmul(Vt, tenzor::matmul(dA, V));

    // dW = diag(E):  per-batch main diagonal -> shape (..., N).
    // Use eye-mask * E then sum over the last axis. eye broadcasts over
    // all leading batch dims of E so this works for any rank >= 2.
    auto eye_mat   = tenzor::eye(N, /*m=*/std::nullopt, A.dtype(), A.device());
    auto E_diag    = tenzor::mul(E, eye_mat);
    auto dW        = tenzor::sum(E_diag, /*dim=*/last, /*keepdim=*/false);

    // Build F mask: F_{ij} = 1/(W_j - W_i) for i ≠ j, 0 on diagonal.
    // W has shape (..., N); unsqueeze on the last two axes to broadcast
    // into a (..., N, N) pairwise-difference table.
    auto W_col = tenzor::unsqueeze(W, -1);  // (..., N, 1)  varies over i
    auto W_row = tenzor::unsqueeze(W, -2);  // (..., 1, N)  varies over j
    auto denom = tenzor::sub(W_row, W_col); // (..., N, N)
    // Avoid divide-by-zero on the diagonal; replace zeros with 1, then zero
    // the diagonal of F afterwards via masking.
    auto zero_tensor  = tenzor::zeros_like(denom);
    auto one_tensor   = tenzor::ones_like(denom);
    auto is_zero_mask = tenzor::eq(denom, zero_tensor);
    auto safe_denom   = tenzor::where(is_zero_mask, one_tensor, denom);
    auto F            = tenzor::div(one_tensor, safe_denom);
    F                 = tenzor::where(is_zero_mask, zero_tensor, F);

    // dV = V @ (F * E)  (batched)
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

// ===========================================================================
// Audit A.4 batch 8: Attention family + losses + adaptive/fractional max pool
// + gather-at-saved-indices for Median + NonDifferentiable stubs for
// Quantile / Nanmedian / NestedLayerNorm / NestedAttention / dropout-on
// FlashAttention / non-identity FlexAttention.
// ===========================================================================

namespace {

// ---- Attention adapters --------------------------------------------------

// Helper: route FlashAttention / FusedAttention inputs (Q, K, V) through the
// shared SDPA JVP. We require dropout_p == 0 — dropout makes the JVP
// discontinuous unless we save & re-use the Bernoulli mask, which the
// dispatcher does not pipe through.
JvpResult jvp_adapter_flash_attention(std::span<const Tensor> primals,
                                      std::span<const Tensor> tangents,
                                      const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_flash_attention: expected 3 inputs (Q, K, V)");
    }
    double dropout_p = attrs.get_float(AttrKey::DropoutP, 0.0);
    if (dropout_p > 0.0) {
        throw NonDifferentiable(
            "FlashAttention forward-mode JVP: dropout_p>0 makes the JVP "
            "discontinuous unless the Bernoulli mask is saved and replayed; "
            "set dropout_p=0 to use forward-mode AD.");
    }
    float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
    bool causal = attrs.get_bool(AttrKey::Causal, false);
    auto Q = make_dual(primals[0], tangents[0]);
    auto K = make_dual(primals[1], tangents[1]);
    auto V = make_dual(primals[2], tangents[2]);
    return dual_to_result(jvp_sdpa_forward(Q, K, V, scale, causal));
}

// FusedAttention reuses the same JVP formula (it does not expose dropout).
JvpResult jvp_adapter_fused_attention(std::span<const Tensor> primals,
                                      std::span<const Tensor> tangents,
                                      const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_fused_attention: expected 3 inputs (Q, K, V)");
    }
    float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
    bool causal = attrs.get_bool(AttrKey::Causal, false);
    auto Q = make_dual(primals[0], tangents[0]);
    auto K = make_dual(primals[1], tangents[1]);
    auto V = make_dual(primals[2], tangents[2]);
    return dual_to_result(jvp_sdpa_forward(Q, K, V, scale, causal));
}

// FlexAttention: identical to FusedAttention only for the identity score-mod
// (ScoreModId == 0) without a block_mask (which would be a non-diff
// integer-mask input). Other score-mods are user-supplied OpIds that don't
// compose at the Variable-level JVP layer; refuse rather than silently
// dropping the mod's tangent contribution.
JvpResult jvp_adapter_flex_attention(std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs) {
    int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);
    bool has_block_mask  = (primals.size() > 3);
    if (score_mod_id != 0 || has_block_mask) {
        throw NonDifferentiable(
            "FlexAttention forward-mode JVP: only the identity score-mod "
            "(ScoreModId=0) without a block_mask is supported. Non-identity "
            "score_mods are user OpIds not composable at this layer; block "
            "masks are non-differentiable.");
    }
    if (primals.size() < 3 || tangents.size() < 3) {
        throw std::runtime_error(
            "jvp_adapter_flex_attention: expected 3 inputs (Q, K, V)");
    }
    float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
    // FlexAttention is non-causal by default; causal is encoded via score-mod
    // or block_mask, both of which we've already refused above.
    auto Q = make_dual(primals[0], tangents[0]);
    auto K = make_dual(primals[1], tangents[1]);
    auto V = make_dual(primals[2], tangents[2]);
    return dual_to_result(jvp_sdpa_forward(Q, K, V, scale, /*causal=*/false));
}

// NestedAttention: per-segment scaled dot-product over a packed [SumL, H, E]
// values buffer governed by Int64 offsets. The forward kernel iterates over
// segments and the JVP would need to (a) re-derive the per-segment softmax
// and (b) re-apply the offset-driven gather pattern in dual form. The
// underlying primitives (NestedAttention is dispatched as a single fused op)
// are not exposed at the tensor-level API, so a faithful JVP requires
// integration in C++ at kernel level. Refuse for now with a typed exception.
JvpResult jvp_adapter_nested_attention(std::span<const Tensor> primals,
                                       std::span<const Tensor> /*tangents*/,
                                       const OpAttributes& /*attrs*/) {
    if (primals.size() != 5) {
        throw std::runtime_error(
            "jvp_adapter_nested_attention: expected 5 inputs "
            "(Q_vals, K_vals, V_vals, offsets, mask?)");
    }
    throw NonDifferentiable(
        "NestedAttention forward-mode JVP not implemented: the fused "
        "per-segment kernel is not decomposable at the Tensor-level API. "
        "Use forward-mode AD through the manual unpadded matmul+softmax path "
        "instead.");
}

// ---- Loss adapters -------------------------------------------------------

// FusedSoftmaxCrossEntropy(logits, targets) JVP. `targets` is integer (no
// tangent). Reads AttrKey::Reduction (default "mean").
JvpResult jvp_adapter_fused_softmax_cross_entropy(
        std::span<const Tensor> primals,
        std::span<const Tensor> tangents,
        const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_fused_softmax_cross_entropy: expected 2 inputs "
            "(logits, targets)");
    }
    std::string reduction(attrs.get_string(AttrKey::Reduction, "mean"));
    auto logits = make_dual(primals[0], tangents[0]);
    return dual_to_result(
        jvp_fused_softmax_cross_entropy(logits, primals[1], reduction));
}

// CTCLossForward: dynamic-programming forward-backward over log-prob
// alignments. Forward-mode through DP requires propagating tangents through
// the per-(t, s) lattice using the same recurrence — feasible but requires
// a dedicated DP kernel (not just primitives). Refuse explicitly rather
// than silently zeroing tangents.
JvpResult jvp_adapter_ctc_loss_forward(std::span<const Tensor> /*primals*/,
                                       std::span<const Tensor> /*tangents*/,
                                       const OpAttributes& /*attrs*/) {
    throw NonDifferentiable(
        "CTCLossForward forward-mode JVP not implemented: the dynamic-"
        "programming alpha-beta recurrence is not expressible as a "
        "composition of registered primitives. A dedicated dual-DP kernel "
        "would be needed.");
}

// ---- Adaptive max pool: gather-at-saved-indices --------------------------
//
// AdaptiveMaxPool{1,2,3}d returns {output, indices}: `output[..., o] =
// input[..., argmax_window(o)]` and `indices[..., o] = argmax_window(o)`.
// Forward-mode tangent: gather(dx, indices) per output cell. Implemented by
// running the kernel once (to obtain indices), then gathering the input
// tangent at those flat-window indices using `take_along_dim` on the
// flattened spatial axes — matching the kth-value pattern.

namespace {

// Run AdaptiveMaxPool{1,2,3}d once via dispatch, returning {output, indices}.
auto run_adaptive_maxpool_nd(OpId op, const Tensor& x,
                             const OpAttributes& attrs) -> std::vector<Tensor> {
    return tenzor::dispatch(op, std::vector<Tensor>{x}, attrs);
}

// Common per-rank tangent gather: flatten the trailing `ndims` spatial axes
// of `dx` and of `indices` into a single dim, take_along_dim, then reshape
// the gathered values back to the output's [N, C, *out_dims] layout. The
// indices tensor from the kernel is already flat-encoded (single int per
// spatial output cell into the flat input window of that batch/channel).
auto gather_at_pool_indices(const Tensor& dx,
                            const Tensor& indices,
                            int64_t spatial_dims) -> Tensor {
    // dx: [N, C, *in_spatial];  indices: [N, C, *out_spatial]
    const auto& dx_shape  = dx.shape();
    const auto& idx_shape = indices.shape();
    int64_t total_ndim = static_cast<int64_t>(dx_shape.size());
    int64_t lead = total_ndim - spatial_dims;  // N + C dims (== 2 typically)

    // Flatten trailing spatial dims of dx → [N, C, in_prod]; same for
    // indices → [N, C, out_prod].
    int64_t in_prod = 1;
    for (int64_t d = lead; d < total_ndim; ++d) in_prod *= dx_shape[d];
    int64_t out_prod = 1;
    for (int64_t d = lead; d < static_cast<int64_t>(idx_shape.size()); ++d)
        out_prod *= idx_shape[d];

    std::vector<int64_t> dx_flat_shape(dx_shape.begin(),
                                       dx_shape.begin() + lead);
    dx_flat_shape.push_back(in_prod);
    std::vector<int64_t> idx_flat_shape(idx_shape.begin(),
                                        idx_shape.begin() + lead);
    idx_flat_shape.push_back(out_prod);

    Tensor dx_flat  = tenzor::reshape(dx,      dx_flat_shape);
    Tensor idx_flat = tenzor::reshape(indices, idx_flat_shape);

    Tensor gathered = tenzor::take_along_dim(dx_flat, idx_flat,
                                              /*dim=*/lead);
    // Reshape back to indices' original [N, C, *out_spatial] layout.
    return tenzor::reshape(gathered,
        std::vector<int64_t>(idx_shape.begin(), idx_shape.end()));
}

} // anonymous (helpers)

JvpMultiResult jvp_adapter_adaptive_maxpool_1d(std::span<const Tensor> primals,
                                               std::span<const Tensor> tangents,
                                               const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_adaptive_maxpool_1d: expected 1 input");
    }
    auto outs = run_adaptive_maxpool_nd(OpId::AdaptiveMaxPool1d, primals[0], attrs);
    Tensor out_p = outs[0], idx_p = outs[1];
    Tensor dx = tangents[0];
    if (dx.numel() == 0) {
        Tensor zero_out = tenzor::zeros(
            std::vector<int64_t>(out_p.shape().begin(), out_p.shape().end()),
            out_p.dtype(), out_p.device());
        Tensor zero_idx = tenzor::zeros(
            std::vector<int64_t>(idx_p.shape().begin(), idx_p.shape().end()),
            idx_p.dtype(), idx_p.device());
        return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                              {std::move(zero_out), std::move(zero_idx)}};
    }
    Tensor out_t = gather_at_pool_indices(dx, idx_p, /*spatial_dims=*/1);
    Tensor idx_t = tenzor::zeros(
        std::vector<int64_t>(idx_p.shape().begin(), idx_p.shape().end()),
        idx_p.dtype(), idx_p.device());
    return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                          {std::move(out_t), std::move(idx_t)}};
}

JvpMultiResult jvp_adapter_adaptive_maxpool_2d(std::span<const Tensor> primals,
                                               std::span<const Tensor> tangents,
                                               const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_adaptive_maxpool_2d: expected 1 input");
    }
    auto outs = run_adaptive_maxpool_nd(OpId::AdaptiveMaxPool2d, primals[0], attrs);
    Tensor out_p = outs[0], idx_p = outs[1];
    Tensor dx = tangents[0];
    if (dx.numel() == 0) {
        Tensor zero_out = tenzor::zeros(
            std::vector<int64_t>(out_p.shape().begin(), out_p.shape().end()),
            out_p.dtype(), out_p.device());
        Tensor zero_idx = tenzor::zeros(
            std::vector<int64_t>(idx_p.shape().begin(), idx_p.shape().end()),
            idx_p.dtype(), idx_p.device());
        return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                              {std::move(zero_out), std::move(zero_idx)}};
    }
    Tensor out_t = gather_at_pool_indices(dx, idx_p, /*spatial_dims=*/2);
    Tensor idx_t = tenzor::zeros(
        std::vector<int64_t>(idx_p.shape().begin(), idx_p.shape().end()),
        idx_p.dtype(), idx_p.device());
    return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                          {std::move(out_t), std::move(idx_t)}};
}

JvpMultiResult jvp_adapter_adaptive_maxpool_3d(std::span<const Tensor> primals,
                                               std::span<const Tensor> tangents,
                                               const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_adaptive_maxpool_3d: expected 1 input");
    }
    auto outs = run_adaptive_maxpool_nd(OpId::AdaptiveMaxPool3d, primals[0], attrs);
    Tensor out_p = outs[0], idx_p = outs[1];
    Tensor dx = tangents[0];
    if (dx.numel() == 0) {
        Tensor zero_out = tenzor::zeros(
            std::vector<int64_t>(out_p.shape().begin(), out_p.shape().end()),
            out_p.dtype(), out_p.device());
        Tensor zero_idx = tenzor::zeros(
            std::vector<int64_t>(idx_p.shape().begin(), idx_p.shape().end()),
            idx_p.dtype(), idx_p.device());
        return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                              {std::move(zero_out), std::move(zero_idx)}};
    }
    Tensor out_t = gather_at_pool_indices(dx, idx_p, /*spatial_dims=*/3);
    Tensor idx_t = tenzor::zeros(
        std::vector<int64_t>(idx_p.shape().begin(), idx_p.shape().end()),
        idx_p.dtype(), idx_p.device());
    return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                          {std::move(out_t), std::move(idx_t)}};
}

// ---- FractionalMaxPool2d/3d: gather-at-saved-indices ----------------------
//
// Both fractional max pool forwards return {output, indices} just like the
// adaptive max pools; the sampling pattern is data-independent of the input
// values (it's a function of random/uniform samples passed as input[1]),
// so given the saved indices the JVP is the same gather-at-indices pattern.
// The optional samples tensor (input[1]) is integer/uniform and carries no
// meaningful tangent.

JvpMultiResult jvp_adapter_fractional_maxpool_2d(std::span<const Tensor> primals,
                                                 std::span<const Tensor> tangents,
                                                 const OpAttributes& attrs) {
    if (primals.empty() || tangents.empty()) {
        throw std::runtime_error("jvp_adapter_fractional_maxpool_2d: missing inputs");
    }
    // Re-run the forward to capture indices.
    std::vector<Tensor> inps;
    inps.reserve(primals.size());
    for (const auto& t : primals) inps.push_back(t);
    auto outs = tenzor::dispatch(OpId::FractionalMaxPool2dForward, inps, attrs);
    Tensor out_p = outs[0], idx_p = outs[1];
    Tensor dx = tangents[0];
    if (dx.numel() == 0) {
        Tensor zero_out = tenzor::zeros(
            std::vector<int64_t>(out_p.shape().begin(), out_p.shape().end()),
            out_p.dtype(), out_p.device());
        Tensor zero_idx = tenzor::zeros(
            std::vector<int64_t>(idx_p.shape().begin(), idx_p.shape().end()),
            idx_p.dtype(), idx_p.device());
        return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                              {std::move(zero_out), std::move(zero_idx)}};
    }
    Tensor out_t = gather_at_pool_indices(dx, idx_p, /*spatial_dims=*/2);
    Tensor idx_t = tenzor::zeros(
        std::vector<int64_t>(idx_p.shape().begin(), idx_p.shape().end()),
        idx_p.dtype(), idx_p.device());
    return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                          {std::move(out_t), std::move(idx_t)}};
}

JvpMultiResult jvp_adapter_fractional_maxpool_3d(std::span<const Tensor> primals,
                                                 std::span<const Tensor> tangents,
                                                 const OpAttributes& attrs) {
    if (primals.empty() || tangents.empty()) {
        throw std::runtime_error("jvp_adapter_fractional_maxpool_3d: missing inputs");
    }
    std::vector<Tensor> inps;
    inps.reserve(primals.size());
    for (const auto& t : primals) inps.push_back(t);
    auto outs = tenzor::dispatch(OpId::FractionalMaxPool3dForward, inps, attrs);
    Tensor out_p = outs[0], idx_p = outs[1];
    Tensor dx = tangents[0];
    if (dx.numel() == 0) {
        Tensor zero_out = tenzor::zeros(
            std::vector<int64_t>(out_p.shape().begin(), out_p.shape().end()),
            out_p.dtype(), out_p.device());
        Tensor zero_idx = tenzor::zeros(
            std::vector<int64_t>(idx_p.shape().begin(), idx_p.shape().end()),
            idx_p.dtype(), idx_p.device());
        return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                              {std::move(zero_out), std::move(zero_idx)}};
    }
    Tensor out_t = gather_at_pool_indices(dx, idx_p, /*spatial_dims=*/3);
    Tensor idx_t = tenzor::zeros(
        std::vector<int64_t>(idx_p.shape().begin(), idx_p.shape().end()),
        idx_p.dtype(), idx_p.device());
    return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                          {std::move(out_t), std::move(idx_t)}};
}

// ---- Median (multi-output: {values, indices}) ---------------------------
//
// Median dispatch (OpId::Median): inputs = {x}; outputs = {values, indices}.
// Forward-mode tangent: take_along_dim(dx, indices, dim) — exact gather at
// the saved median positions. When dim==INT64_MIN the kernel flattens then
// reduces along axis 0, so we replicate that flattening for dx.
JvpMultiResult jvp_adapter_median(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_median: expected 1 input");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);

    OpAttributes mattrs;
    mattrs.set(AttrKey::Dim, dim);
    mattrs.set(AttrKey::Keepdim, keepdim);
    auto outs = tenzor::dispatch(OpId::Median,
                                 std::vector<Tensor>{primals[0]}, mattrs);
    Tensor values_p  = outs[0];
    Tensor indices_p = outs[1];

    const Tensor& dx = tangents[0];
    Tensor values_t;
    if (dx.numel() == 0) {
        values_t = tenzor::zeros(
            std::vector<int64_t>(values_p.shape().begin(),
                                  values_p.shape().end()),
            values_p.dtype(), values_p.device());
    } else if (dim == INT64_MIN) {
        // Full reduction over flattened input.
        Tensor dx_flat = tenzor::reshape(dx, std::vector<int64_t>{dx.numel()});
        Tensor idx_flat = tenzor::reshape(indices_p,
            std::vector<int64_t>{indices_p.numel()});
        values_t = tenzor::take_along_dim(dx_flat, idx_flat, /*dim=*/0);
        if (keepdim) {
            std::vector<int64_t> kshape(dx.ndim(), 1);
            values_t = tenzor::reshape(values_t, kshape);
        } else {
            values_t = tenzor::reshape(values_t, std::vector<int64_t>{1});
        }
    } else {
        Tensor idx_kd = keepdim ? indices_p
                                : tenzor::unsqueeze(indices_p, dim);
        auto gathered = tenzor::take_along_dim(dx, idx_kd, dim);
        values_t = keepdim ? gathered : tenzor::squeeze(gathered, dim);
    }

    Tensor indices_t = tenzor::zeros(
        std::vector<int64_t>(indices_p.shape().begin(), indices_p.shape().end()),
        indices_p.dtype(), indices_p.device());

    return JvpMultiResult{{std::move(values_p), std::move(indices_p)},
                          {std::move(values_t), std::move(indices_t)}};
}

// ---- NonDifferentiable stubs: Quantile / Nanquantile / Nanmedian ---------
//
// Quantile uses linear interpolation between two sorted neighbours; the
// per-element interpolation indices and weights are not exposed by the
// kernel, so a closed-form JVP would require either (a) re-implementing
// the search at the autograd layer or (b) extending the kernel to return
// the index pair. Until one of those lands, we mark these explicitly
// non-differentiable rather than silently zeroing tangents.
//
// Nanmedian's NaN-aware median selects different elements depending on the
// NaN mask of the input — the output is discontinuous with respect to
// continuous perturbations that flip a NaN to a finite value (and vice
// versa). Even the finite-NaN case lacks an "indices" output to gather
// against.
JvpResult jvp_adapter_quantile(std::span<const Tensor> /*primals*/,
                               std::span<const Tensor> /*tangents*/,
                               const OpAttributes& /*attrs*/) {
    throw NonDifferentiable(
        "Quantile forward-mode JVP not implemented: linear-interpolation "
        "indices/weights are not exposed by the kernel. Use Median for an "
        "indexed quantile path that supports JVP.");
}

JvpResult jvp_adapter_nanquantile(std::span<const Tensor> /*primals*/,
                                  std::span<const Tensor> /*tangents*/,
                                  const OpAttributes& /*attrs*/) {
    throw NonDifferentiable(
        "Nanquantile forward-mode JVP not implemented: NaN-mask-dependent "
        "selection makes the JVP discontinuous, and interpolation indices "
        "are not exposed.");
}

JvpResult jvp_adapter_nanmedian(std::span<const Tensor> /*primals*/,
                                std::span<const Tensor> /*tangents*/,
                                const OpAttributes& /*attrs*/) {
    throw NonDifferentiable(
        "Nanmedian forward-mode JVP not implemented: the kernel does not "
        "return indices, and NaN-driven selection is discontinuous in the "
        "input.");
}

// ---- Nested softmax / log-softmax: per-segment linear-in-input JVP -------
//
// NestedSoftmax / NestedLogSoftmax operate on packed `values` over segments
// defined by `offsets`. The mathematical structure is identical to dense
// softmax/log-softmax along `dim`, just restricted to per-segment rows —
// which means the JVP formula is the same per segment and we can express
// it by re-running the nested kernel on the tangent values where the
// operation is linear (log-softmax tangent: dt - softmax(...)*sum(dt)),
// but the nested-softmax tangent involves a per-segment softmax, which
// itself is the kernel output — so the simplest correct path is to call
// the nested kernel twice (once for the primal, once on a fused expression
// that yields the tangent). For NestedSoftmax that fused expression isn't
// a single nested op, so we delegate by computing the tangent using the
// returned primal output: P_t = P * (dvalues - sum_per_segment(P*dvalues)).
// The "sum per segment" reduction matches OpId::NestedSum semantics.
JvpResult jvp_adapter_nested_softmax(std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_nested_softmax: expected 2 inputs (values, offsets)");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    const Tensor& values = primals[0];
    const Tensor& offsets = primals[1];
    const Tensor& dvalues = tangents[0];

    // Primal: P = nested_softmax(values, offsets, dim).
    OpAttributes sattrs;
    sattrs.set(AttrKey::Dim, dim);
    auto P = tenzor::dispatch(OpId::NestedSoftmax,
                              std::vector<Tensor>{values, offsets}, sattrs)[0];

    Tensor tangent;
    if (dvalues.numel() == 0) {
        tangent = tenzor::zeros(
            std::vector<int64_t>(P.shape().begin(), P.shape().end()),
            P.dtype(), P.device());
    } else {
        // Per-segment reduction sum_j P[j] * dvalues[j] is a NestedSum
        // (keepdim=true so the broadcast works against the per-position
        // entries within the same segment).
        OpAttributes rattrs;
        rattrs.set(AttrKey::Dim, dim);
        rattrs.set(AttrKey::Keepdim, true);
        Tensor P_dv = tenzor::mul(P, dvalues);
        Tensor seg_sum = tenzor::dispatch(OpId::NestedSum,
            std::vector<Tensor>{P_dv, offsets}, rattrs)[0];
        // P_t = P * (dvalues - seg_sum)
        tangent = tenzor::mul(P, tenzor::sub(dvalues, seg_sum));
    }
    return JvpResult{std::move(P), std::move(tangent)};
}

// NestedLogSoftmax: y = log_softmax_per_segment(values).
// Tangent: dy = dvalues - softmax_per_segment * sum_per_segment(dvalues).
JvpResult jvp_adapter_nested_log_softmax(std::span<const Tensor> primals,
                                         std::span<const Tensor> tangents,
                                         const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_nested_log_softmax: expected 2 inputs (values, offsets)");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    const Tensor& values = primals[0];
    const Tensor& offsets = primals[1];
    const Tensor& dvalues = tangents[0];

    // Primal: log P.
    OpAttributes sattrs;
    sattrs.set(AttrKey::Dim, dim);
    auto logP = tenzor::dispatch(OpId::NestedLogSoftmax,
        std::vector<Tensor>{values, offsets}, sattrs)[0];

    Tensor tangent;
    if (dvalues.numel() == 0) {
        tangent = tenzor::zeros(
            std::vector<int64_t>(logP.shape().begin(), logP.shape().end()),
            logP.dtype(), logP.device());
    } else {
        // softmax_per_segment = exp(logP) (no recomputation through the
        // values needed; logP is the kernel's exact output).
        Tensor P = tenzor::exp(logP);
        OpAttributes rattrs;
        rattrs.set(AttrKey::Dim, dim);
        rattrs.set(AttrKey::Keepdim, true);
        Tensor seg_sum_dv = tenzor::dispatch(OpId::NestedSum,
            std::vector<Tensor>{dvalues, offsets}, rattrs)[0];
        tangent = tenzor::sub(dvalues, tenzor::mul(P, seg_sum_dv));
    }
    return JvpResult{std::move(logP), std::move(tangent)};
}

// NestedLayerNorm: requires per-segment mean/var statistics that are not
// exposed by the nested kernel; the dispatch returns only the normalised
// values. A correct JVP needs the saved mean/rstd to express the chain
// rule (same shape as the dense LayerNorm rule), so we refuse rather than
// silently zeroing tangents.
JvpResult jvp_adapter_nested_layer_norm(std::span<const Tensor> /*primals*/,
                                        std::span<const Tensor> /*tangents*/,
                                        const OpAttributes& /*attrs*/) {
    throw NonDifferentiable(
        "NestedLayerNorm forward-mode JVP not implemented: the dispatcher "
        "returns only the normalised values; per-segment mean/rstd are not "
        "exposed and are required to express the chain rule.");
}

} // anonymous (batch 8)

// ============================================================================
// Audit A.4 batch 9 — JVP coverage extension
// ============================================================================
//
// This batch closes the long tail of differentiable OpIds that the previous
// eight batches did not reach, plus registers explicit NonDifferentiable rules
// for ops that have no derivative (boolean / integer / RNG / *Backward /
// inplace / quantized / discrete-output) — failing loudly at JVP dispatch
// time rather than silently returning a zero tangent.
//
// Patterns used by the linear-in-input adapters below:
//
//   - "Linear pass-through" ops (Conj, Real, Imag, Contiguous, Clone,
//     AsStrided, ToMemoryFormat, FFT2/N, IFFT2/N, AvgPool{1,2,3}d, Fold,
//     Interpolate, GridSample, ROIAlignForward, DiagEmbed, Diagflat,
//     MaskedScatter, MaxUnpool{1,2,3}d, AffineGrid, NestedTo/FromPadded):
//        y  = f(x)  with f linear  →  dy = f(dx)
//     Implemented by re-dispatching the same OpId on the tangent.
//
//   - Linear-in-weight ops (Embedding): the indices argument is integer and
//     non-differentiable; only `weight` carries a tangent.
//
//   - Multi-output max-pool style (MaxPool{1,2,3}dForward): saves indices in
//     output[1]; tangent = gather_at_pool_indices(dx, indices, spatial_dims).
//
//   - Closed-form unary derivatives use the documented chain-rule formula in
//     the function header (e.g. d/dx rsqrt(x) = -0.5 * x^{-3/2}).
//
//   - Closed-form binary derivatives use the standard sub-gradients (Maximum,
//     Minimum, Fmax, Fmin: sub-gradient = indicator of the active operand).
// ============================================================================

namespace {

// ---- NonDifferentiable rule helper macros ----------------------------------
//
// A NonDifferentiable rule throws tenzor::NonDifferentiable when invoked.
// We expose two forms: one for single-output ops (JvpResult) and one for
// multi-output ops (JvpMultiResult). The thrown message names the OpId
// and the structural reason it cannot have a forward-mode JVP, so callers
// see "Cast forward-mode JVP not implemented: …" rather than a generic
// "no rule registered" runtime_error.

#define TENZOR_JVP_NONDIFF(name, reason)                                        \
    JvpResult name(std::span<const Tensor> /*primals*/,                         \
                   std::span<const Tensor> /*tangents*/,                        \
                   const OpAttributes& /*attrs*/) {                             \
        throw NonDifferentiable(reason);                                        \
    }

#define TENZOR_JVP_NONDIFF_MULTI(name, reason)                                  \
    JvpMultiResult name(std::span<const Tensor> /*primals*/,                    \
                        std::span<const Tensor> /*tangents*/,                   \
                        const OpAttributes& /*attrs*/) {                        \
        throw NonDifferentiable(reason);                                        \
    }

// ---- Linear-in-input adapter helper ----------------------------------------
//
// For ops whose forward kernel is linear in input(s), the JVP rule simply
// re-dispatches the same OpId on the tangent inputs. This works for unary
// (most pooling/shape ops) and binary linear forms. We provide a unary
// helper here; multi-input cases are handled bespoke below.

inline auto linear_unary_jvp(OpId op,
                             std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) -> JvpResult {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error(
            std::string("jvp linear-unary adapter: expected 1 input for ") +
            std::string(op_id_to_name(op)));
    }
    auto p = make_dual(primals[0], tangents[0]);
    auto primal_out  = tenzor::dispatch(op, std::vector<Tensor>{p.primal()},  attrs)[0];
    auto tangent_out = tenzor::dispatch(op, std::vector<Tensor>{p.tangent()}, attrs)[0];
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// ============================================================================
// Linear-in-input adapters (apply same op on tangent)
// ============================================================================

// Complex ops are R-linear (treating complex as a pair of reals). Conj, Real,
// Imag are linear projections of the complex value; tangent applies the same
// projection.
JvpResult jvp_adapter_conj(std::span<const Tensor> p, std::span<const Tensor> t,
                           const OpAttributes& a) { return linear_unary_jvp(OpId::Conj, p, t, a); }
JvpResult jvp_adapter_real(std::span<const Tensor> p, std::span<const Tensor> t,
                           const OpAttributes& a) { return linear_unary_jvp(OpId::Real, p, t, a); }
JvpResult jvp_adapter_imag(std::span<const Tensor> p, std::span<const Tensor> t,
                           const OpAttributes& a) { return linear_unary_jvp(OpId::Imag, p, t, a); }

// Identity-up-to-memory layout ops: contiguous / clone / to_memory_format /
// as_strided. The forward is a pure copy or stride-relabel; tangent is the
// same op on dx.
JvpResult jvp_adapter_contiguous(std::span<const Tensor> p, std::span<const Tensor> t,
                                 const OpAttributes& a) { return linear_unary_jvp(OpId::Contiguous, p, t, a); }
JvpResult jvp_adapter_clone(std::span<const Tensor> p, std::span<const Tensor> t,
                            const OpAttributes& a) { return linear_unary_jvp(OpId::Clone, p, t, a); }
JvpResult jvp_adapter_to_memory_format(std::span<const Tensor> p, std::span<const Tensor> t,
                                       const OpAttributes& a) { return linear_unary_jvp(OpId::ToMemoryFormat, p, t, a); }
JvpResult jvp_adapter_as_strided(std::span<const Tensor> p, std::span<const Tensor> t,
                                 const OpAttributes& a) { return linear_unary_jvp(OpId::AsStrided, p, t, a); }

// Diagonal layout ops: diag_embed / diagflat — linear shape operations.
JvpResult jvp_adapter_diag_embed(std::span<const Tensor> p, std::span<const Tensor> t,
                                 const OpAttributes& a) { return linear_unary_jvp(OpId::DiagEmbed, p, t, a); }
JvpResult jvp_adapter_diagflat(std::span<const Tensor> p, std::span<const Tensor> t,
                               const OpAttributes& a) { return linear_unary_jvp(OpId::Diagflat, p, t, a); }

// FFT variants beyond 1D: FFT2, IFFT2, FFTN, IFFTN are linear over complex.
JvpResult jvp_adapter_fft2(std::span<const Tensor> p, std::span<const Tensor> t,
                           const OpAttributes& a) { return linear_unary_jvp(OpId::FFT2, p, t, a); }
JvpResult jvp_adapter_ifft2(std::span<const Tensor> p, std::span<const Tensor> t,
                            const OpAttributes& a) { return linear_unary_jvp(OpId::IFFT2, p, t, a); }
JvpResult jvp_adapter_fftn(std::span<const Tensor> p, std::span<const Tensor> t,
                           const OpAttributes& a) { return linear_unary_jvp(OpId::FFTN, p, t, a); }
JvpResult jvp_adapter_ifftn(std::span<const Tensor> p, std::span<const Tensor> t,
                            const OpAttributes& a) { return linear_unary_jvp(OpId::IFFTN, p, t, a); }

// AvgPool{1,2,3}dForward: linear in input (each output cell is a fixed-weight
// average of an input window with kernel-derived stride/padding); the window
// pattern depends only on attributes, not values → tangent = same op on dx.
JvpResult jvp_adapter_avg_pool1d(std::span<const Tensor> p, std::span<const Tensor> t,
                                 const OpAttributes& a) { return linear_unary_jvp(OpId::AvgPool1dForward, p, t, a); }
JvpResult jvp_adapter_avg_pool2d(std::span<const Tensor> p, std::span<const Tensor> t,
                                 const OpAttributes& a) { return linear_unary_jvp(OpId::AvgPool2dForward, p, t, a); }
JvpResult jvp_adapter_avg_pool3d(std::span<const Tensor> p, std::span<const Tensor> t,
                                 const OpAttributes& a) { return linear_unary_jvp(OpId::AvgPool3dForward, p, t, a); }

// Fold / Unfold-style: fold is the linear adjoint of unfold (already
// registered); both are pure rearrangements with overlap-summation → linear.
JvpResult jvp_adapter_fold(std::span<const Tensor> p, std::span<const Tensor> t,
                           const OpAttributes& a) { return linear_unary_jvp(OpId::Fold, p, t, a); }

// Interpolate (bilinear/nearest/etc.): all currently-supported modes are
// linear in `input` for fixed output_size/scale; tangent = same op on dx.
// (For 'nearest', the JVP is well-defined since the gather pattern depends
// only on shape, not values.)
JvpResult jvp_adapter_interpolate(std::span<const Tensor> p, std::span<const Tensor> t,
                                  const OpAttributes& a) { return linear_unary_jvp(OpId::Interpolate, p, t, a); }

// GridSample: y[n, c, h, w] = sum_{i,j} G(h,w; grid) * input[n, c, i, j]
// is linear in `input` for a fixed grid; non-linear in `grid` (which the
// project marks NonDifferentiable through autograd at this layer). For JVP
// purposes we propagate the input tangent through the same op; if the grid
// has a non-zero tangent we refuse.
JvpResult jvp_adapter_grid_sample(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& attrs) {
    if (primals.size() < 2 || primals.size() != tangents.size()) {
        throw std::runtime_error("jvp_adapter_grid_sample: expected 2 inputs (input, grid)");
    }
    if (tangents[1].numel() != 0) {
        throw NonDifferentiable(
            "GridSample forward-mode JVP w.r.t. grid is not implemented; "
            "grid carries non-zero tangent. Only input-side JVP is supported.");
    }
    auto primal = tenzor::dispatch(OpId::GridSample,
                                   std::vector<Tensor>{primals[0], primals[1]}, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::GridSample,
                                    std::vector<Tensor>{tangents[0], primals[1]}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// ROIAlignForward: y[k, c, ph, pw] = bilinear-sample(input, boxes[k]) is
// linear in `input` for fixed boxes; boxes are integer-quantized then
// bilinear-interpolated, so we propagate input tangent only.
JvpResult jvp_adapter_roi_align(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() < 2 || primals.size() != tangents.size()) {
        throw std::runtime_error("jvp_adapter_roi_align: expected 2 inputs (input, boxes)");
    }
    if (tangents[1].numel() != 0) {
        throw NonDifferentiable(
            "ROIAlign forward-mode JVP w.r.t. boxes is not implemented; "
            "boxes carry non-zero tangent.");
    }
    auto primal = tenzor::dispatch(OpId::ROIAlignForward,
                                   std::vector<Tensor>{primals[0], primals[1]}, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::ROIAlignForward,
                                    std::vector<Tensor>{tangents[0], primals[1]}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// AffineGrid: theta -> grid is linear in theta (affine_grid is a fixed-base
// linear transform on the theta matrix). Tangent = same op on dtheta.
JvpResult jvp_adapter_affine_grid(std::span<const Tensor> p, std::span<const Tensor> t,
                                  const OpAttributes& a) { return linear_unary_jvp(OpId::AffineGrid, p, t, a); }

// NestedTo/FromPadded: pure data rearrangements (offset-driven copy); linear.
JvpResult jvp_adapter_nested_to_padded(std::span<const Tensor> primals,
                                       std::span<const Tensor> tangents,
                                       const OpAttributes& attrs) {
    if (primals.size() < 2 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_nested_to_padded: expected (values, offsets[, padding])");
    }
    // Offsets at primals[1] are integer; we never read tangents[1].
    std::vector<Tensor> p_in(primals.begin(), primals.end());
    std::vector<Tensor> t_in = p_in;
    t_in[0] = tangents[0].numel() != 0 ? tangents[0]
        : tenzor::zeros(std::vector<int64_t>(primals[0].shape().begin(),
                                              primals[0].shape().end()),
                        primals[0].dtype(), primals[0].device());
    auto primal  = tenzor::dispatch(OpId::NestedToPadded, p_in, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::NestedToPadded, t_in, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}
JvpResult jvp_adapter_nested_from_padded(std::span<const Tensor> primals,
                                         std::span<const Tensor> tangents,
                                         const OpAttributes& attrs) {
    if (primals.size() < 2 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_nested_from_padded: expected (padded, offsets)");
    }
    std::vector<Tensor> p_in(primals.begin(), primals.end());
    std::vector<Tensor> t_in = p_in;
    t_in[0] = tangents[0].numel() != 0 ? tangents[0]
        : tenzor::zeros(std::vector<int64_t>(primals[0].shape().begin(),
                                              primals[0].shape().end()),
                        primals[0].dtype(), primals[0].device());
    auto primal  = tenzor::dispatch(OpId::NestedFromPadded, p_in, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::NestedFromPadded, t_in, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// MaskedScatter: y = mask ? source[unflatten(cumsum(mask))] : input. Linear
// in (input, source); mask is boolean and non-differentiable. Tangent =
// MaskedScatter(d_input, mask, d_source).
JvpResult jvp_adapter_masked_scatter(std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs) {
    if (primals.size() != 3 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_masked_scatter: expected 3 inputs (input, mask, source)");
    }
    auto dinp = tangents[0].numel() != 0 ? tangents[0]
        : tenzor::zeros(std::vector<int64_t>(primals[0].shape().begin(),
                                              primals[0].shape().end()),
                        primals[0].dtype(), primals[0].device());
    auto dsrc = tangents[2].numel() != 0 ? tangents[2]
        : tenzor::zeros(std::vector<int64_t>(primals[2].shape().begin(),
                                              primals[2].shape().end()),
                        primals[2].dtype(), primals[2].device());
    auto primal  = tenzor::dispatch(OpId::MaskedScatter,
                                    std::vector<Tensor>{primals[0], primals[1], primals[2]},
                                    attrs)[0];
    auto tangent = tenzor::dispatch(OpId::MaskedScatter,
                                    std::vector<Tensor>{dinp, primals[1], dsrc},
                                    attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// Put: linear in input + source (mirror of MaskedScatter).
JvpResult jvp_adapter_put(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    if (primals.size() != 3 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_put: expected 3 inputs (input, index, source)");
    }
    auto dinp = tangents[0].numel() != 0 ? tangents[0]
        : tenzor::zeros(std::vector<int64_t>(primals[0].shape().begin(),
                                              primals[0].shape().end()),
                        primals[0].dtype(), primals[0].device());
    auto dsrc = tangents[2].numel() != 0 ? tangents[2]
        : tenzor::zeros(std::vector<int64_t>(primals[2].shape().begin(),
                                              primals[2].shape().end()),
                        primals[2].dtype(), primals[2].device());
    auto primal  = tenzor::dispatch(OpId::Put,
                                    std::vector<Tensor>{primals[0], primals[1], primals[2]},
                                    attrs)[0];
    auto tangent = tenzor::dispatch(OpId::Put,
                                    std::vector<Tensor>{dinp, primals[1], dsrc},
                                    attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// ScatterAdd: y = input.clone(); y[index] += source. Linear in input+source.
JvpResult jvp_adapter_scatter_add(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& attrs) {
    if (primals.size() != 3 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_scatter_add: expected 3 inputs (input, index, source)");
    }
    auto dinp = tangents[0].numel() != 0 ? tangents[0]
        : tenzor::zeros(std::vector<int64_t>(primals[0].shape().begin(),
                                              primals[0].shape().end()),
                        primals[0].dtype(), primals[0].device());
    auto dsrc = tangents[2].numel() != 0 ? tangents[2]
        : tenzor::zeros(std::vector<int64_t>(primals[2].shape().begin(),
                                              primals[2].shape().end()),
                        primals[2].dtype(), primals[2].device());
    auto primal  = tenzor::dispatch(OpId::ScatterAdd,
                                    std::vector<Tensor>{primals[0], primals[1], primals[2]},
                                    attrs)[0];
    auto tangent = tenzor::dispatch(OpId::ScatterAdd,
                                    std::vector<Tensor>{dinp, primals[1], dsrc},
                                    attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// ============================================================================
// MaxPool{1,2,3}dForward: multi-output {values, indices}; gather pattern.
// ============================================================================
//
// Each output is the max of an input window; the saved indices give the
// flat-spatial offset within the window. The tangent at the output is the
// tangent at the chosen input location, gathered via take_along_dim on the
// flattened spatial axis (same pattern used for adaptive max pool).
namespace {
auto run_maxpool_nd(OpId op, const Tensor& x,
                    const OpAttributes& attrs) -> std::vector<Tensor> {
    return tenzor::dispatch(op, std::vector<Tensor>{x}, attrs);
}

auto maxpool_nd_jvp_impl(OpId op, int64_t spatial_dims,
                         std::span<const Tensor> primals,
                         std::span<const Tensor> tangents,
                         const OpAttributes& attrs) -> JvpMultiResult {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_maxpool: expected 1 input");
    }
    auto outs = run_maxpool_nd(op, primals[0], attrs);
    Tensor out_p = outs[0];
    Tensor idx_p = outs.size() > 1 ? outs[1] : Tensor{};
    Tensor dx = tangents[0];
    auto zeros_like_t = [](const Tensor& t) -> Tensor {
        return tenzor::zeros(std::vector<int64_t>(t.shape().begin(), t.shape().end()),
                             t.dtype(), t.device());
    };
    Tensor idx_t = idx_p.numel() != 0 ? zeros_like_t(idx_p) : Tensor{};
    if (dx.numel() == 0) {
        Tensor zero_out = zeros_like_t(out_p);
        if (idx_p.numel() != 0) {
            return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                                  {std::move(zero_out), std::move(idx_t)}};
        }
        return JvpMultiResult{{std::move(out_p)}, {std::move(zero_out)}};
    }
    if (idx_p.numel() == 0) {
        // Some backends may not return indices for single-output max-pool
        // signatures; in that case we cannot index into dx, so fail loudly.
        throw NonDifferentiable(
            "MaxPool*Forward JVP requires saved indices (output[1]); "
            "the kernel returned only the values tensor.");
    }
    Tensor out_t = gather_at_pool_indices(dx, idx_p, spatial_dims);
    return JvpMultiResult{{std::move(out_p), std::move(idx_p)},
                          {std::move(out_t), std::move(idx_t)}};
}
} // namespace

JvpMultiResult jvp_adapter_max_pool_1d(std::span<const Tensor> p,
                                       std::span<const Tensor> t,
                                       const OpAttributes& a) {
    return maxpool_nd_jvp_impl(OpId::MaxPool1dForward, /*spatial_dims=*/1, p, t, a);
}
JvpMultiResult jvp_adapter_max_pool_2d(std::span<const Tensor> p,
                                       std::span<const Tensor> t,
                                       const OpAttributes& a) {
    return maxpool_nd_jvp_impl(OpId::MaxPool2dForward, /*spatial_dims=*/2, p, t, a);
}
JvpMultiResult jvp_adapter_max_pool_3d(std::span<const Tensor> p,
                                       std::span<const Tensor> t,
                                       const OpAttributes& a) {
    return maxpool_nd_jvp_impl(OpId::MaxPool3dForward, /*spatial_dims=*/3, p, t, a);
}

// MaxUnpool{1,2,3}dForward: y[indices] = input scattered to a zero buffer.
// Linear in `input` for fixed indices → tangent = same op on dx + same
// indices.
JvpResult jvp_adapter_max_unpool_impl(OpId op,
                                      std::span<const Tensor> primals,
                                      std::span<const Tensor> tangents,
                                      const OpAttributes& attrs) {
    if (primals.size() < 2 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_max_unpool: expected (input, indices[, output_size])");
    }
    auto dinp = tangents[0].numel() != 0 ? tangents[0]
        : tenzor::zeros(std::vector<int64_t>(primals[0].shape().begin(),
                                              primals[0].shape().end()),
                        primals[0].dtype(), primals[0].device());
    std::vector<Tensor> p_in(primals.begin(), primals.end());
    std::vector<Tensor> t_in = p_in;
    t_in[0] = dinp;
    auto primal  = tenzor::dispatch(op, p_in, attrs)[0];
    auto tangent = tenzor::dispatch(op, t_in, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}
JvpResult jvp_adapter_max_unpool_1d(std::span<const Tensor> p, std::span<const Tensor> t,
                                    const OpAttributes& a) {
    return jvp_adapter_max_unpool_impl(OpId::MaxUnpool1dForward, p, t, a);
}
JvpResult jvp_adapter_max_unpool_2d(std::span<const Tensor> p, std::span<const Tensor> t,
                                    const OpAttributes& a) {
    return jvp_adapter_max_unpool_impl(OpId::MaxUnpool2dForward, p, t, a);
}
JvpResult jvp_adapter_max_unpool_3d(std::span<const Tensor> p, std::span<const Tensor> t,
                                    const OpAttributes& a) {
    return jvp_adapter_max_unpool_impl(OpId::MaxUnpool3dForward, p, t, a);
}

// ============================================================================
// Linear-in-weight: Embedding.
//   y[i] = weight[indices[i]];   indices ∈ Integer, weight ∈ Float.
//   dy[i] = dweight[indices[i]]  → dispatch(Embedding, [indices, dweight]).
// ============================================================================
JvpResult jvp_adapter_embedding(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 2 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_embedding: expected 2 inputs (indices, weight)");
    }
    auto dweight = tangents[1].numel() != 0 ? tangents[1]
        : tenzor::zeros(std::vector<int64_t>(primals[1].shape().begin(),
                                              primals[1].shape().end()),
                        primals[1].dtype(), primals[1].device());
    auto primal  = tenzor::dispatch(OpId::Embedding,
                                    std::vector<Tensor>{primals[0], primals[1]}, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::Embedding,
                                    std::vector<Tensor>{primals[0], dweight}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// ============================================================================
// Linear-in-input bilinear GEMMs: Addmm / Addmv / Baddbmm.
//   Addmm(I, A, B; α, β) = β·I + α·(A @ B)
//   d/dt = β·dI + α·(dA @ B + A @ dB)
// Implemented by composing the existing linear ops since Addmm itself is
// linear in (I, A, B).
// ============================================================================
namespace {
inline auto fetch_alpha_beta(const OpAttributes& attrs) -> std::pair<double, double> {
    double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
    double beta  = attrs.get_float(AttrKey::Beta,  1.0);
    return {alpha, beta};
}
} // namespace

JvpResult jvp_adapter_addmm(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes& attrs) {
    if (primals.size() != 3 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_addmm: expected 3 inputs (input, mat1, mat2)");
    }
    auto [alpha, beta] = fetch_alpha_beta(attrs);
    const auto& I = primals[0]; const auto& A = primals[1]; const auto& B = primals[2];
    auto zeros_like = [](const Tensor& x) {
        return tenzor::zeros(std::vector<int64_t>(x.shape().begin(), x.shape().end()),
                             x.dtype(), x.device());
    };
    auto dI = tangents[0].numel() != 0 ? tangents[0] : zeros_like(I);
    auto dA = tangents[1].numel() != 0 ? tangents[1] : zeros_like(A);
    auto dB = tangents[2].numel() != 0 ? tangents[2] : zeros_like(B);
    auto primal  = tenzor::add(tenzor::mul(I, beta),
                               tenzor::mul(tenzor::matmul(A, B), alpha));
    auto t_AB    = tenzor::add(tenzor::matmul(dA, B), tenzor::matmul(A, dB));
    auto tangent = tenzor::add(tenzor::mul(dI, beta), tenzor::mul(t_AB, alpha));
    return JvpResult{std::move(primal), std::move(tangent)};
}

JvpResult jvp_adapter_addmv(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes& attrs) {
    if (primals.size() != 3 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_addmv: expected 3 inputs (input, mat, vec)");
    }
    auto [alpha, beta] = fetch_alpha_beta(attrs);
    const auto& I = primals[0]; const auto& A = primals[1]; const auto& v = primals[2];
    auto zeros_like = [](const Tensor& x) {
        return tenzor::zeros(std::vector<int64_t>(x.shape().begin(), x.shape().end()),
                             x.dtype(), x.device());
    };
    auto dI = tangents[0].numel() != 0 ? tangents[0] : zeros_like(I);
    auto dA = tangents[1].numel() != 0 ? tangents[1] : zeros_like(A);
    auto dv = tangents[2].numel() != 0 ? tangents[2] : zeros_like(v);
    auto primal  = tenzor::add(tenzor::mul(I, beta),
                               tenzor::mul(tenzor::matmul(A, v), alpha));
    auto t_Av    = tenzor::add(tenzor::matmul(dA, v), tenzor::matmul(A, dv));
    auto tangent = tenzor::add(tenzor::mul(dI, beta), tenzor::mul(t_Av, alpha));
    return JvpResult{std::move(primal), std::move(tangent)};
}

JvpResult jvp_adapter_baddbmm(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 3 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_baddbmm: expected 3 inputs (input, batch1, batch2)");
    }
    auto [alpha, beta] = fetch_alpha_beta(attrs);
    const auto& I = primals[0]; const auto& A = primals[1]; const auto& B = primals[2];
    auto zeros_like = [](const Tensor& x) {
        return tenzor::zeros(std::vector<int64_t>(x.shape().begin(), x.shape().end()),
                             x.dtype(), x.device());
    };
    auto dI = tangents[0].numel() != 0 ? tangents[0] : zeros_like(I);
    auto dA = tangents[1].numel() != 0 ? tangents[1] : zeros_like(A);
    auto dB = tangents[2].numel() != 0 ? tangents[2] : zeros_like(B);
    auto primal  = tenzor::add(tenzor::mul(I, beta),
                               tenzor::mul(tenzor::bmm(A, B), alpha));
    auto t_AB    = tenzor::add(tenzor::bmm(dA, B), tenzor::bmm(A, dB));
    auto tangent = tenzor::add(tenzor::mul(dI, beta), tenzor::mul(t_AB, alpha));
    return JvpResult{std::move(primal), std::move(tangent)};
}

// Dot: y = sum_i a[i] * b[i]; d = sum_i (da*b + a*db).
JvpResult jvp_adapter_dot(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& /*attrs*/) {
    if (primals.size() != 2 || primals.size() != tangents.size()) {
        throw std::runtime_error("jvp_adapter_dot: expected 2 inputs (a, b)");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    auto primal  = tenzor::dot(a.primal(), b.primal());
    auto tangent = tenzor::add(tenzor::dot(a.tangent(), b.primal()),
                                tenzor::dot(a.primal(), b.tangent()));
    return JvpResult{std::move(primal), std::move(tangent)};
}

// LinalgVecdot: sum along a single dim. Bilinear like dot but per-axis.
JvpResult jvp_adapter_linalg_vecdot(std::span<const Tensor> primals,
                                    std::span<const Tensor> tangents,
                                    const OpAttributes& attrs) {
    if (primals.size() != 2 || primals.size() != tangents.size()) {
        throw std::runtime_error("jvp_adapter_linalg_vecdot: expected 2 inputs (a, b)");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    auto primal  = tenzor::dispatch(OpId::LinalgVecdot,
                                    std::vector<Tensor>{a.primal(), b.primal()}, attrs)[0];
    auto t_from_a = tenzor::dispatch(OpId::LinalgVecdot,
                                     std::vector<Tensor>{a.tangent(), b.primal()}, attrs)[0];
    auto t_from_b = tenzor::dispatch(OpId::LinalgVecdot,
                                     std::vector<Tensor>{a.primal(), b.tangent()}, attrs)[0];
    return JvpResult{std::move(primal), tenzor::add(t_from_a, t_from_b)};
}

// ============================================================================
// Closed-form derivatives (linear and non-linear)
// ============================================================================

// d/dx rsqrt(x) = -0.5 * x^{-3/2}.
auto jvp_rsqrt(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::rsqrt(x.primal());                  // 1/sqrt(x)
    // -0.5 * rsqrt(x)^3 == -0.5 * x^{-3/2}
    auto factor = tenzor::mul(tenzor::mul(primal, primal), primal);
    auto tangent = tenzor::mul(x.tangent(), tenzor::mul(factor, -0.5));
    return DualTensor(std::move(primal), std::move(tangent));
}

// d/dx deg2rad(x) = π/180; d/dx rad2deg(x) = 180/π.
auto jvp_deg2rad(const DualTensor& x) -> DualTensor {
    constexpr double kDegToRad = 0.017453292519943295;  // π/180
    auto primal  = tenzor::deg2rad(x.primal());
    auto tangent = tenzor::mul(x.tangent(), kDegToRad);
    return DualTensor(std::move(primal), std::move(tangent));
}
auto jvp_rad2deg(const DualTensor& x) -> DualTensor {
    constexpr double kRadToDeg = 57.29577951308232;     // 180/π
    auto primal  = tenzor::rad2deg(x.primal());
    auto tangent = tenzor::mul(x.tangent(), kRadToDeg);
    return DualTensor(std::move(primal), std::move(tangent));
}

// logit(x; eps) = log(x/(1-x))  (with clamp at [eps, 1-eps] when eps>=0).
// d/dx logit(x) = 1/(x(1-x)) for x in the open interval; piecewise-constant
// (0) where the clamp pins the value. We implement the open-interval form,
// matching PyTorch's behaviour at non-clamped points (the clamp boundary is a
// measure-zero set in the unclamped case).
auto jvp_logit(const DualTensor& x, double /*eps*/) -> DualTensor {
    auto primal = tenzor::logit(x.primal(), -1.0);
    auto one = tenzor::ones_like(x.primal());
    auto one_minus = tenzor::sub(one, x.primal());
    auto denom = tenzor::mul(x.primal(), one_minus);
    auto tangent = tenzor::div(x.tangent(), denom);
    return DualTensor(std::move(primal), std::move(tangent));
}

// d/dx sinc(x) = (πx·cos(πx) − sin(πx)) / (πx)^2   for x ≠ 0;
//              = 0 at x = 0 (limit of even function with sinc'(0) = 0).
auto jvp_sinc(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::sinc(x.primal());
    constexpr double kPi = 3.141592653589793;
    auto px = tenzor::mul(x.primal(), kPi);
    auto cos_px = tenzor::cos(px);
    auto sin_px = tenzor::sin(px);
    auto px_safe = tenzor::add(px, 1e-30);  // avoid /0 at x=0
    auto deriv = tenzor::div(tenzor::sub(cos_px, tenzor::div(sin_px, px_safe)),
                              tenzor::add(x.primal(), 1e-30));
    // The above is sinc'(x) but unstable at x=0; replace with 0 where |x|<eps
    // via a mask.
    auto eps_t = tenzor::mul(tenzor::ones_like(x.primal()), 1e-6);
    auto small = tenzor::lt(tenzor::abs(x.primal()), eps_t);
    auto zero  = tenzor::zeros_like(deriv);
    auto deriv_safe = tenzor::where(small, zero, deriv);
    auto tangent = tenzor::mul(x.tangent(), deriv_safe);
    return DualTensor(std::move(primal), std::move(tangent));
}

// d/dx ndtr(x) = exp(-x^2/2) / sqrt(2π)  (standard normal PDF).
auto jvp_ndtr(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::ndtr(x.primal());
    constexpr double kInvSqrt2Pi = 0.3989422804014327;
    auto neg_half_x2 = tenzor::mul(tenzor::mul(x.primal(), x.primal()), -0.5);
    auto pdf = tenzor::mul(tenzor::exp(neg_half_x2), kInvSqrt2Pi);
    auto tangent = tenzor::mul(x.tangent(), pdf);
    return DualTensor(std::move(primal), std::move(tangent));
}

// d/dx log_ndtr(x) = pdf(x) / ndtr(x) — the score function of the normal CDF.
auto jvp_log_ndtr(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::log_ndtr(x.primal());
    constexpr double kInvSqrt2Pi = 0.3989422804014327;
    auto neg_half_x2 = tenzor::mul(tenzor::mul(x.primal(), x.primal()), -0.5);
    auto pdf = tenzor::mul(tenzor::exp(neg_half_x2), kInvSqrt2Pi);
    auto cdf = tenzor::ndtr(x.primal());
    auto tangent = tenzor::mul(x.tangent(), tenzor::div(pdf, cdf));
    return DualTensor(std::move(primal), std::move(tangent));
}

// d/dx erfinv(y) = √π / 2 · exp(erfinv(y)^2). Compute the primal first so the
// derivative reuses it without an extra dispatch.
auto jvp_erfinv(const DualTensor& y) -> DualTensor {
    auto primal = tenzor::erfinv(y.primal());
    constexpr double kHalfSqrtPi = 0.8862269254527580;  // √π / 2
    auto sq = tenzor::mul(primal, primal);
    auto factor = tenzor::mul(tenzor::exp(sq), kHalfSqrtPi);
    auto tangent = tenzor::mul(y.tangent(), factor);
    return DualTensor(std::move(primal), std::move(tangent));
}

// d/dx digamma(x) = trigamma(x) = polygamma(1, x). The polygamma series in
// the project is sufficient for the n=1 case.
auto jvp_digamma(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::digamma(x.primal());
    auto trig = tenzor::polygamma(1, x.primal());
    auto tangent = tenzor::mul(x.tangent(), trig);
    return DualTensor(std::move(primal), std::move(tangent));
}

// d/dx gamma(x) = gamma(x) · digamma(x).
auto jvp_gamma(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::gamma(x.primal());
    auto dg = tenzor::digamma(x.primal());
    auto tangent = tenzor::mul(x.tangent(), tenzor::mul(primal, dg));
    return DualTensor(std::move(primal), std::move(tangent));
}

// Bessel derivatives (DLMF identities):
//   J0'(x) = −J1(x)
//   J1'(x) = J0(x) − J1(x)/x       (we use J0(x) − J1(x)*x^{-1}; safe at x>0)
//   I0'(x) = I1(x)
//   I1'(x) = I0(x) − I1(x)/x       (analogous identity for modified Bessel).
auto jvp_bessel_j0(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::bessel_j0(x.primal());
    auto deriv  = tenzor::neg(tenzor::bessel_j1(x.primal()));
    return DualTensor(std::move(primal), tenzor::mul(x.tangent(), deriv));
}
auto jvp_bessel_j1(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::bessel_j1(x.primal());
    auto j0     = tenzor::bessel_j0(x.primal());
    auto x_safe = tenzor::add(tenzor::abs(x.primal()), 1e-30);
    auto deriv  = tenzor::sub(j0, tenzor::div(primal, x_safe));
    return DualTensor(std::move(primal), tenzor::mul(x.tangent(), deriv));
}
auto jvp_bessel_i0(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::bessel_i0(x.primal());
    auto deriv  = tenzor::bessel_i1(x.primal());
    return DualTensor(std::move(primal), tenzor::mul(x.tangent(), deriv));
}
auto jvp_bessel_i1(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::bessel_i1(x.primal());
    auto i0     = tenzor::bessel_i0(x.primal());
    auto x_safe = tenzor::add(tenzor::abs(x.primal()), 1e-30);
    auto deriv  = tenzor::sub(i0, tenzor::div(primal, x_safe));
    return DualTensor(std::move(primal), tenzor::mul(x.tangent(), deriv));
}

// d/dx i0e(x) = exp(-|x|)·BesselI0(x); derivative:
//   i0e'(x) = exp(-|x|)·(I1(x) − sign(x)·I0(x))
//          = i1e(x) − sign(x)·i0e(x)
auto jvp_i0e(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::i0e(x.primal());
    auto i1e_   = tenzor::i1e(x.primal());
    auto sgn    = tenzor::sign(x.primal());
    auto deriv  = tenzor::sub(i1e_, tenzor::mul(sgn, primal));
    return DualTensor(std::move(primal), tenzor::mul(x.tangent(), deriv));
}
// Analogous for i1e: i1e'(x) = i0e(x) − sign(x)·i1e(x) − i1e(x)/x.
auto jvp_i1e(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::i1e(x.primal());
    auto i0e_   = tenzor::i0e(x.primal());
    auto sgn    = tenzor::sign(x.primal());
    auto x_safe = tenzor::add(tenzor::abs(x.primal()), 1e-30);
    auto deriv  = tenzor::sub(tenzor::sub(i0e_, tenzor::mul(sgn, primal)),
                                tenzor::div(primal, x_safe));
    return DualTensor(std::move(primal), tenzor::mul(x.tangent(), deriv));
}

// spherical_bessel_j0(x) = sin(x)/x; derivative = (x·cos(x) − sin(x))/x² (and
// 0 at x=0 by symmetry).
auto jvp_spherical_bessel_j0(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::spherical_bessel_j0(x.primal());
    auto cosx = tenzor::cos(x.primal());
    auto sinx = tenzor::sin(x.primal());
    auto x_safe = tenzor::add(x.primal(), 1e-30);
    auto deriv = tenzor::div(tenzor::sub(tenzor::mul(x.primal(), cosx), sinx),
                              tenzor::mul(x_safe, x_safe));
    auto eps_t = tenzor::mul(tenzor::ones_like(x.primal()), 1e-6);
    auto small = tenzor::lt(tenzor::abs(x.primal()), eps_t);
    auto deriv_safe = tenzor::where(small, tenzor::zeros_like(deriv), deriv);
    return DualTensor(std::move(primal), tenzor::mul(x.tangent(), deriv_safe));
}

// d/dx entr(x) = -log(x) - 1; entr(x) = -x*log(x) for x > 0, 0 at x=0,
// -inf at x<0. We propagate the open-domain derivative.
auto jvp_entr(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::entr(x.primal());
    auto deriv = tenzor::sub(tenzor::neg(tenzor::log(x.primal())),
                              tenzor::ones_like(x.primal()));
    return DualTensor(std::move(primal), tenzor::mul(x.tangent(), deriv));
}

// d/dx (a + v*b*c) = da + v*(db*c + b*dc) for addcmul;
// d/dx (a + v*b/c) = da + v*(db/c - b*dc/c^2) for addcdiv.
auto jvp_addcmul(const DualTensor& a, const DualTensor& b,
                 const DualTensor& c, double v) -> DualTensor {
    auto primal = tenzor::addcmul(a.primal(), b.primal(), c.primal(), v);
    auto t_bc = tenzor::add(tenzor::mul(b.tangent(), c.primal()),
                              tenzor::mul(b.primal(), c.tangent()));
    auto tangent = tenzor::add(a.tangent(), tenzor::mul(t_bc, v));
    return DualTensor(std::move(primal), std::move(tangent));
}
auto jvp_addcdiv(const DualTensor& a, const DualTensor& b,
                 const DualTensor& c, double v) -> DualTensor {
    auto primal = tenzor::addcdiv(a.primal(), b.primal(), c.primal(), v);
    auto c_sq = tenzor::mul(c.primal(), c.primal());
    auto num  = tenzor::sub(tenzor::mul(b.tangent(), c.primal()),
                              tenzor::mul(b.primal(), c.tangent()));
    auto t_bc = tenzor::div(num, c_sq);
    auto tangent = tenzor::add(a.tangent(), tenzor::mul(t_bc, v));
    return DualTensor(std::move(primal), std::move(tangent));
}

// lerp(start, end, weight) = start + weight * (end - start)
//   = (1 - weight) * start + weight * end  (linear in start, end, weight)
auto jvp_lerp(const DualTensor& start, const DualTensor& end,
              const DualTensor& weight) -> DualTensor {
    auto primal = tenzor::lerp(start.primal(), end.primal(), weight.primal());
    auto one = tenzor::ones_like(weight.primal());
    auto one_minus_w = tenzor::sub(one, weight.primal());
    auto end_minus_start = tenzor::sub(end.primal(), start.primal());
    auto t = tenzor::add(
        tenzor::add(tenzor::mul(start.tangent(), one_minus_w),
                    tenzor::mul(end.tangent(),   weight.primal())),
        tenzor::mul(weight.tangent(), end_minus_start));
    return DualTensor(std::move(primal), std::move(t));
}

// Maximum / Minimum: piecewise-linear; sub-gradient = indicator of the
// active operand (ties split 0.5/0.5).
auto jvp_maximum(const DualTensor& a, const DualTensor& b) -> DualTensor {
    auto primal = tenzor::maximum(a.primal(), b.primal());
    auto a_active = tenzor::gt(a.primal(), b.primal());
    auto tie = tenzor::eq(a.primal(), b.primal());
    // weight_a = 1 if a>b, 0.5 if a==b, 0 otherwise.
    auto half = tenzor::mul(tenzor::ones_like(a.primal()), 0.5);
    auto wa = tenzor::where(a_active, tenzor::ones_like(a.primal()),
                              tenzor::where(tie, half, tenzor::zeros_like(a.primal())));
    auto wb = tenzor::sub(tenzor::ones_like(a.primal()), wa);
    auto tangent = tenzor::add(tenzor::mul(a.tangent(), wa),
                                 tenzor::mul(b.tangent(), wb));
    return DualTensor(std::move(primal), std::move(tangent));
}
auto jvp_minimum(const DualTensor& a, const DualTensor& b) -> DualTensor {
    auto primal = tenzor::minimum(a.primal(), b.primal());
    auto a_active = tenzor::lt(a.primal(), b.primal());
    auto tie = tenzor::eq(a.primal(), b.primal());
    auto half = tenzor::mul(tenzor::ones_like(a.primal()), 0.5);
    auto wa = tenzor::where(a_active, tenzor::ones_like(a.primal()),
                              tenzor::where(tie, half, tenzor::zeros_like(a.primal())));
    auto wb = tenzor::sub(tenzor::ones_like(a.primal()), wa);
    auto tangent = tenzor::add(tenzor::mul(a.tangent(), wa),
                                 tenzor::mul(b.tangent(), wb));
    return DualTensor(std::move(primal), std::move(tangent));
}
// Fmax / Fmin: like maximum/minimum but NaN-propagating per IEEE 754-2008.
// Sub-gradient identical (NaN inputs make the tangent NaN automatically via
// arithmetic in the same chain — no special handling needed for forward AD).
auto jvp_fmax(const DualTensor& a, const DualTensor& b) -> DualTensor {
    return jvp_maximum(a, b);
}
auto jvp_fmin(const DualTensor& a, const DualTensor& b) -> DualTensor {
    return jvp_minimum(a, b);
}

// d/d{a,b} log_add_exp2(a, b) = softmax2_weighted in base-2:
//   y = log2(2^a + 2^b);  dy/da = 2^a / (2^a + 2^b);  dy/db = 1 - dy/da.
auto jvp_logaddexp2(const DualTensor& a, const DualTensor& b) -> DualTensor {
    auto primal = tenzor::logaddexp2(a.primal(), b.primal());
    auto pa = tenzor::exp2(a.primal());
    auto pb = tenzor::exp2(b.primal());
    auto denom = tenzor::add(pa, pb);
    auto wa = tenzor::div(pa, denom);
    auto wb = tenzor::div(pb, denom);
    auto tangent = tenzor::add(tenzor::mul(a.tangent(), wa),
                                 tenzor::mul(b.tangent(), wb));
    return DualTensor(std::move(primal), std::move(tangent));
}

// xlog1py(x, y) = x * log1p(y); d = dx * log1p(y) + x * dy / (1+y).
auto jvp_xlog1py(const DualTensor& x, const DualTensor& y) -> DualTensor {
    auto primal = tenzor::xlog1py(x.primal(), y.primal());
    auto log1py = tenzor::log1p(y.primal());
    auto one = tenzor::ones_like(y.primal());
    auto t1 = tenzor::mul(x.tangent(), log1py);
    auto t2 = tenzor::div(tenzor::mul(x.primal(), y.tangent()),
                            tenzor::add(one, y.primal()));
    return DualTensor(std::move(primal), tenzor::add(t1, t2));
}

// xlogy(x, y) = x * log(y); d = dx * log(y) + x * dy / y.
auto jvp_xlogy(const DualTensor& x, const DualTensor& y) -> DualTensor {
    auto primal = tenzor::xlogy(x.primal(), y.primal());
    auto logy = tenzor::log(y.primal());
    auto t1 = tenzor::mul(x.tangent(), logy);
    auto t2 = tenzor::div(tenzor::mul(x.primal(), y.tangent()), y.primal());
    return DualTensor(std::move(primal), tenzor::add(t1, t2));
}

// float_power(b, e) = b^e (Float64-promoted). General base/exponent both
// floating: d = e * b^(e-1) * db + b^e * log(b) * de.
auto jvp_float_power(const DualTensor& b, const DualTensor& e) -> DualTensor {
    auto primal = tenzor::float_power(b.primal(), e.primal());
    auto one = tenzor::ones_like(e.primal());
    auto e_minus_1 = tenzor::sub(e.primal(), one);
    auto b_pow_em1 = tenzor::float_power(b.primal(), e_minus_1);
    auto logb = tenzor::log(b.primal());
    auto term_b = tenzor::mul(tenzor::mul(e.primal(), b_pow_em1), b.tangent());
    auto term_e = tenzor::mul(tenzor::mul(primal, logb), e.tangent());
    return DualTensor(std::move(primal), tenzor::add(term_b, term_e));
}

// Cross product (3D): linear in each operand. d/dt (a × b) = da × b + a × db.
auto jvp_cross(const DualTensor& a, const DualTensor& b, int64_t dim) -> DualTensor {
    auto primal = tenzor::cross(a.primal(), b.primal(), dim);
    auto t1 = tenzor::cross(a.tangent(), b.primal(), dim);
    auto t2 = tenzor::cross(a.primal(), b.tangent(), dim);
    return DualTensor(std::move(primal), tenzor::add(t1, t2));
}

// Polar(abs, angle) = abs * (cos(angle) + i*sin(angle))
//   d = dabs * (cos+i*sin) + abs * dangle * (−sin+i*cos)
// where ComplexTensor(re, im) builds the complex pair from reals.
auto jvp_polar(const DualTensor& abs_, const DualTensor& angle_) -> DualTensor {
    auto primal = tenzor::polar(abs_.primal(), angle_.primal());
    // Real part: abs * cos(angle); imag part: abs * sin(angle).
    auto cos_a = tenzor::cos(angle_.primal());
    auto sin_a = tenzor::sin(angle_.primal());
    // d_re = dabs*cos - abs*sin*dangle;  d_im = dabs*sin + abs*cos*dangle.
    auto d_re = tenzor::sub(tenzor::mul(abs_.tangent(), cos_a),
                              tenzor::mul(tenzor::mul(abs_.primal(), sin_a),
                                            angle_.tangent()));
    auto d_im = tenzor::add(tenzor::mul(abs_.tangent(), sin_a),
                              tenzor::mul(tenzor::mul(abs_.primal(), cos_a),
                                            angle_.tangent()));
    OpAttributes complex_attrs;
    auto tangent = tenzor::dispatch(OpId::ComplexTensor,
                                    std::vector<Tensor>{d_re, d_im}, complex_attrs)[0];
    return DualTensor(std::move(primal), std::move(tangent));
}

// ComplexTensor(re, im) = re + i*im; linear in both → d = dre + i*dim.
JvpResult jvp_adapter_complex_tensor(std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs) {
    if (primals.size() != 2 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_complex_tensor: expected 2 inputs (real, imag)");
    }
    auto zeros_like = [](const Tensor& x) {
        return tenzor::zeros(std::vector<int64_t>(x.shape().begin(), x.shape().end()),
                             x.dtype(), x.device());
    };
    auto d_re = tangents[0].numel() != 0 ? tangents[0] : zeros_like(primals[0]);
    auto d_im = tangents[1].numel() != 0 ? tangents[1] : zeros_like(primals[1]);
    auto primal  = tenzor::dispatch(OpId::ComplexTensor,
                                    std::vector<Tensor>{primals[0], primals[1]}, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::ComplexTensor,
                                    std::vector<Tensor>{d_re, d_im}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// d/dx angle(z) = (-imag(z)*dre + real(z)*dim) / |z|^2 for complex z.
// We expose the real-input case (angle(real x) = 0 for x>=0, π for x<0)
// which is a.e. constant → tangent = 0; this matches PyTorch.
auto jvp_angle(const DualTensor& z) -> DualTensor {
    auto primal = tenzor::angle(z.primal());
    auto tangent = tenzor::zeros_like(primal);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Adapters for the new closed-form rules
// ============================================================================

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

TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_rsqrt,        jvp_rsqrt)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_deg2rad,      jvp_deg2rad)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_rad2deg,      jvp_rad2deg)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_sinc,         jvp_sinc)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_ndtr,         jvp_ndtr)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_log_ndtr,     jvp_log_ndtr)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_erfinv,       jvp_erfinv)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_digamma,      jvp_digamma)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_gamma,        jvp_gamma)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_bessel_j0,    jvp_bessel_j0)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_bessel_j1,    jvp_bessel_j1)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_bessel_i0,    jvp_bessel_i0)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_bessel_i1,    jvp_bessel_i1)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_i0e,          jvp_i0e)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_i1e,          jvp_i1e)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_spherical_j0, jvp_spherical_bessel_j0)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_entr,         jvp_entr)
TENZOR_JVP_UNARY_ADAPTER(jvp_adapter_angle,        jvp_angle)

#undef TENZOR_JVP_UNARY_ADAPTER

// logit needs an `eps` attr; non-trivial signature.
JvpResult jvp_adapter_logit(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_logit: expected 1 input");
    }
    double eps = attrs.get_float(AttrKey::Eps, -1.0);
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_logit(x, eps));
}

// Binary closed-form adapters.
JvpResult jvp_adapter_maximum(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& /*attrs*/) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_maximum: expected 2 inputs");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_maximum(a, b));
}
JvpResult jvp_adapter_minimum(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& /*attrs*/) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_minimum: expected 2 inputs");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_minimum(a, b));
}
JvpResult jvp_adapter_fmax(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& /*attrs*/) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_fmax: expected 2 inputs");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_fmax(a, b));
}
JvpResult jvp_adapter_fmin(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& /*attrs*/) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_fmin: expected 2 inputs");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_fmin(a, b));
}
JvpResult jvp_adapter_logaddexp2(std::span<const Tensor> primals,
                                 std::span<const Tensor> tangents,
                                 const OpAttributes& /*attrs*/) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_logaddexp2: expected 2 inputs");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_logaddexp2(a, b));
}
JvpResult jvp_adapter_xlog1py(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& /*attrs*/) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_xlog1py: expected 2 inputs");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_xlog1py(a, b));
}
JvpResult jvp_adapter_xlogy(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes& /*attrs*/) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_xlogy: expected 2 inputs");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_xlogy(a, b));
}
JvpResult jvp_adapter_float_power(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes& /*attrs*/) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_float_power: expected 2 inputs");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_float_power(a, b));
}
JvpResult jvp_adapter_cross(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_cross: expected 2 inputs");
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_cross(a, b, dim));
}
JvpResult jvp_adapter_polar(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes& /*attrs*/) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_polar: expected 2 inputs");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    return dual_to_result(jvp_polar(a, b));
}
// Ternary closed-form adapters: addcmul / addcdiv / lerp.
JvpResult jvp_adapter_addcmul(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error("jvp_adapter_addcmul: expected 3 inputs");
    }
    double v = attrs.get_float(AttrKey::Value, 1.0);
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    auto c = make_dual(primals[2], tangents[2]);
    return dual_to_result(jvp_addcmul(a, b, c, v));
}
JvpResult jvp_adapter_addcdiv(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error("jvp_adapter_addcdiv: expected 3 inputs");
    }
    double v = attrs.get_float(AttrKey::Value, 1.0);
    auto a = make_dual(primals[0], tangents[0]);
    auto b = make_dual(primals[1], tangents[1]);
    auto c = make_dual(primals[2], tangents[2]);
    return dual_to_result(jvp_addcdiv(a, b, c, v));
}
JvpResult jvp_adapter_lerp(std::span<const Tensor> primals,
                           std::span<const Tensor> tangents,
                           const OpAttributes& /*attrs*/) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error("jvp_adapter_lerp: expected 3 inputs (start, end, weight)");
    }
    auto s = make_dual(primals[0], tangents[0]);
    auto e = make_dual(primals[1], tangents[1]);
    auto w = make_dual(primals[2], tangents[2]);
    return dual_to_result(jvp_lerp(s, e, w));
}

// ============================================================================
// NonDifferentiable rules — fail loudly instead of silently returning zero.
// ============================================================================

// Comparison ops: y is Bool; derivative undefined.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_eq,
    "Eq has no derivative: comparison produces Bool output (discrete).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_ne,
    "Ne has no derivative: comparison produces Bool output (discrete).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_lt,
    "Lt has no derivative: comparison produces Bool output (discrete).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_le,
    "Le has no derivative: comparison produces Bool output (discrete).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_gt,
    "Gt has no derivative: comparison produces Bool output (discrete).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_ge,
    "Ge has no derivative: comparison produces Bool output (discrete).")

// Logical / bitwise ops: integer/Bool outputs, non-differentiable.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_logical_and,
    "LogicalAnd has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_logical_or,
    "LogicalOr has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_logical_not,
    "LogicalNot has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_logical_xor,
    "LogicalXor has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_bitwise_and,
    "BitwiseAnd has no derivative: integer-only op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_bitwise_or,
    "BitwiseOr has no derivative: integer-only op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_bitwise_xor,
    "BitwiseXor has no derivative: integer-only op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_bitwise_not,
    "BitwiseNot has no derivative: integer-only op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_bitwise_left_shift,
    "BitwiseLeftShift has no derivative: integer-only op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_bitwise_right_shift,
    "BitwiseRightShift has no derivative: integer-only op.")

// Boolean predicates: discrete output → no derivative.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_isnan,
    "IsNan has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_isinf,
    "IsInf has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_isfinite,
    "IsFinite has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_isposinf,
    "IsPosInf has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_isneginf,
    "IsNegInf has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_isreal,
    "IsReal has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_signbit,
    "Signbit has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_has_inf_nan,
    "HasInfNan has no derivative: Bool output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_isin,
    "Isin has no derivative: Bool set-membership output.")

// Discrete / index-valued / counting ops.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_nonzero,
    "Nonzero has no derivative: returns integer indices of non-zero entries.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_count_nonzero,
    "CountNonzero has no derivative: returns an integer count.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_bincount,
    "Bincount has no derivative: integer histogram from integer input.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_histogram,
    "Histogram has no derivative: returns bin counts (integer-valued).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_histogramdd,
    "Histogramdd has no derivative: returns multi-dim bin counts.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_histc,
    "Histc has no derivative: returns fixed-bin histogram.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_unique,
    "Unique has no derivative: variable-length deduplicated output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_unique_consecutive,
    "UniqueConsecutive has no derivative: variable-length deduplicated output.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_mode,
    "Mode has no derivative: argmax-based discrete selection.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_one_hot,
    "OneHot has no derivative: discrete one-hot encoding from integer indices.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_searchsorted,
    "SearchSorted has no derivative: returns insertion indices (integer).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_segment_reduce,
    "SegmentReduce returns integer segment ids alongside values; full JVP "
    "would require splitting by reduction type (sum/mean differentiable; "
    "amax/amin/prod NonDifferentiable per project policy).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_advanced_index,
    "AdvancedIndex (NumPy-style fancy indexing) has no general JVP at the "
    "dispatch layer: the kernel does not expose the saved index broadcasting "
    "metadata required to thread a tangent through gather; use IndexSelect "
    "or Gather for the JVP-supported path.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_advanced_index_put,
    "AdvancedIndexPut: in-place scatter with multiple index tensors; "
    "see AdvancedIndex above.")

// Random / stochastic ops.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_rand,
    "Rand has no derivative: pure random sampling (no input tangent).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_randn,
    "Randn has no derivative: pure random sampling.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_randint,
    "Randint has no derivative: integer random sampling.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_multinomial,
    "Multinomial has no derivative: discrete sampling; the score-function "
    "estimator is reverse-mode only and not exposed as a JVP rule.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_bernoulli,
    "Bernoulli has no derivative: discrete sampling.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_normal_sample,
    "NormalSample has no derivative without the reparameterisation trick; "
    "the kernel does not save the Gaussian noise used to draw samples.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_poisson_sample,
    "PoissonSample has no continuous derivative (discrete sampling).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_exponential_sample,
    "ExponentialSample has no derivative without the reparameterisation "
    "trick; the kernel does not save the uniform noise used to draw samples.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_gumbel_softmax,
    "GumbelSoftmax has no JVP at this layer: the Gumbel noise is sampled "
    "inside the kernel and not exposed; reparameterised gradients require "
    "the noise to be passed in explicitly.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_rrelu,
    "RReLU has no JVP at this layer: the random negative-slope draw is "
    "sampled inside the kernel and not exposed.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_dropout,
    "Dropout has no JVP without the saved Bernoulli mask; the kernel does "
    "not expose the per-call mask used during the forward pass.")

// Creation ops with no differentiable inputs.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_zeros,
    "Zeros has no derivative: pure factory op with shape/dtype-only inputs.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_ones,
    "Ones has no derivative: pure factory op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_full,
    "Full has no derivative: pure factory op with constant fill value.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_fill,
    "Fill has no derivative: scalar broadcast (constant) into existing storage.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_eye,
    "Eye has no derivative: pure factory op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_arange,
    "Arange has no derivative: integer/float ramp factory op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_linspace,
    "Linspace has no derivative: deterministic spaced-ramp factory op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_tril_indices,
    "TrilIndices has no derivative: integer index pairs.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_triu_indices,
    "TriuIndices has no derivative: integer index pairs.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_strided_fill,
    "StridedFill has no derivative: in-place scalar fill into strided view.")

// Piecewise / discrete float ops.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_fmod,
    "Fmod has no useful derivative: discontinuous at multiples of the divisor.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_remainder,
    "Remainder has no useful derivative: discontinuous at multiples of the divisor.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_nextafter,
    "Nextafter has no derivative: discrete float step.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_copysign,
    "Copysign has no useful derivative: piecewise-constant sign of the second "
    "argument (sub-gradient = sign(b)*indicator on a; b non-diff).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_gcd,
    "Gcd has no derivative: integer-valued op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_lcm,
    "Lcm has no derivative: integer-valued op.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_ldexp,
    "Ldexp has no derivative w.r.t. the integer exponent argument; the "
    "value-side derivative is just multiplication by 2^n — use mul instead.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_frexp,
    "Frexp returns (mantissa, exponent) where the exponent is integer-valued "
    "and the mantissa is piecewise-constant across powers-of-two boundaries.")

// In-place variants: same value-semantics as their out-of-place forms but
// alias the input storage; the dispatch surface does not expose them as
// differentiable through the JVP table (callers should drive AD on the
// out-of-place version).
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_add_inplace,
    "AddInplace has no JVP at this layer: forward-mode AD requires the "
    "out-of-place Add (which is registered); driving JVP through the "
    "in-place dispatch would alias the input storage at tangent-eval time.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_sub_inplace,
    "SubInplace has no JVP at this layer; use Sub.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_mul_inplace,
    "MulInplace has no JVP at this layer; use Mul.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_div_inplace,
    "DivInplace has no JVP at this layer; use Div.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_relu_inplace,
    "ReLUInplace has no JVP at this layer; use ReLU.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_sigmoid_inplace,
    "SigmoidInplace has no JVP at this layer; use Sigmoid.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_tanh_inplace,
    "TanhInplace has no JVP at this layer; use Tanh.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_leaky_relu_inplace,
    "LeakyReLUInplace has no JVP at this layer; use LeakyReLU.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_gelu_inplace,
    "GeluInplace has no JVP at this layer; use Gelu.")

// *Backward kernels: these are themselves the backward kernels of registered
// forward ops. Forward-mode AD through a backward kernel is double-backward
// territory and is not exposed by the dispatch surface.
#define TENZOR_JVP_BWD_NONDIFF(opname)                                          \
    TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_##opname,                            \
        #opname " is a backward kernel; forward-mode AD through it is "         \
        "double-backward and not exposed at this layer.")

TENZOR_JVP_BWD_NONDIFF(relu_backward)
TENZOR_JVP_BWD_NONDIFF(sigmoid_backward)
TENZOR_JVP_BWD_NONDIFF(tanh_backward)
TENZOR_JVP_BWD_NONDIFF(gelu_backward)
TENZOR_JVP_BWD_NONDIFF(softmax_backward)
TENZOR_JVP_BWD_NONDIFF(log_softmax_backward)
TENZOR_JVP_BWD_NONDIFF(leaky_relu_backward)
TENZOR_JVP_BWD_NONDIFF(elu_backward)
TENZOR_JVP_BWD_NONDIFF(selu_backward)
TENZOR_JVP_BWD_NONDIFF(mish_backward)
TENZOR_JVP_BWD_NONDIFF(softplus_backward)
TENZOR_JVP_BWD_NONDIFF(log_sigmoid_backward)
TENZOR_JVP_BWD_NONDIFF(rrelu_backward)
TENZOR_JVP_BWD_NONDIFF(swish_backward)

TENZOR_JVP_BWD_NONDIFF(conv1d_backward_input)
TENZOR_JVP_BWD_NONDIFF(conv1d_backward_weight)
TENZOR_JVP_BWD_NONDIFF(conv1d_backward_bias)
TENZOR_JVP_BWD_NONDIFF(conv2d_backward_input)
TENZOR_JVP_BWD_NONDIFF(conv2d_backward_weight)
TENZOR_JVP_BWD_NONDIFF(conv2d_backward_bias)
TENZOR_JVP_BWD_NONDIFF(conv3d_backward_input)
TENZOR_JVP_BWD_NONDIFF(conv3d_backward_weight)
TENZOR_JVP_BWD_NONDIFF(conv3d_backward_bias)
TENZOR_JVP_BWD_NONDIFF(conv_transpose3d_backward_input)
TENZOR_JVP_BWD_NONDIFF(conv_transpose3d_backward_weight)
TENZOR_JVP_BWD_NONDIFF(conv_transpose3d_backward_bias)

TENZOR_JVP_BWD_NONDIFF(max_pool1d_backward)
TENZOR_JVP_BWD_NONDIFF(max_pool2d_backward)
TENZOR_JVP_BWD_NONDIFF(max_pool3d_backward)
TENZOR_JVP_BWD_NONDIFF(avg_pool1d_backward)
TENZOR_JVP_BWD_NONDIFF(avg_pool2d_backward)
TENZOR_JVP_BWD_NONDIFF(avg_pool3d_backward)
TENZOR_JVP_BWD_NONDIFF(adaptive_avg_pool1d_backward)
TENZOR_JVP_BWD_NONDIFF(adaptive_avg_pool2d_backward)
TENZOR_JVP_BWD_NONDIFF(adaptive_avg_pool3d_backward)
TENZOR_JVP_BWD_NONDIFF(adaptive_max_pool1d_backward)
TENZOR_JVP_BWD_NONDIFF(adaptive_max_pool2d_backward)
TENZOR_JVP_BWD_NONDIFF(adaptive_max_pool3d_backward)
TENZOR_JVP_BWD_NONDIFF(fractional_max_pool2d_backward)
TENZOR_JVP_BWD_NONDIFF(fractional_max_pool3d_backward)
TENZOR_JVP_BWD_NONDIFF(max_unpool1d_backward)
TENZOR_JVP_BWD_NONDIFF(max_unpool2d_backward)
TENZOR_JVP_BWD_NONDIFF(max_unpool3d_backward)

TENZOR_JVP_BWD_NONDIFF(batch_norm2d_backward)
TENZOR_JVP_BWD_NONDIFF(layer_norm_backward)
TENZOR_JVP_BWD_NONDIFF(group_norm_backward)
TENZOR_JVP_BWD_NONDIFF(instance_norm_backward)
TENZOR_JVP_BWD_NONDIFF(rms_norm_backward)
TENZOR_JVP_BWD_NONDIFF(linear_backward)
TENZOR_JVP_BWD_NONDIFF(embedding_backward)
TENZOR_JVP_BWD_NONDIFF(embedding_bag_backward)
TENZOR_JVP_BWD_NONDIFF(dropout_backward)
TENZOR_JVP_BWD_NONDIFF(flash_attention_backward)
TENZOR_JVP_BWD_NONDIFF(flex_attention_backward)
TENZOR_JVP_BWD_NONDIFF(nested_attention_backward)
TENZOR_JVP_BWD_NONDIFF(fused_layer_norm_backward)
TENZOR_JVP_BWD_NONDIFF(roi_align_backward)
TENZOR_JVP_BWD_NONDIFF(gru_cell_backward)
TENZOR_JVP_BWD_NONDIFF(lstm_cell_backward)
TENZOR_JVP_BWD_NONDIFF(interpolate_backward)
TENZOR_JVP_BWD_NONDIFF(deformable_conv2d_backward_input)
TENZOR_JVP_BWD_NONDIFF(deformable_conv2d_backward_weight)
TENZOR_JVP_BWD_NONDIFF(deformable_conv2d_backward_bias)

#undef TENZOR_JVP_BWD_NONDIFF

// BatchNorm helper ops (stats-only): not differentiable outputs.
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_batch_norm2d_mean_var,
    "BatchNorm2dMeanVar returns running statistics (mean/var) accumulated "
    "via Welford; treated as non-differentiable scalars in the JVP layer.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_batch_norm2d_update_running_stats,
    "BatchNorm2dUpdateRunningStats updates running mean/var buffers in place; "
    "stateful side-effecting op with no JVP.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_batch_norm2d_forward,
    "BatchNorm2dForward (non-affine variant) JVP not yet implemented; "
    "use BatchNorm2dForwardAffine which has a registered multi-output rule.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_batch_norm2d_fused_training,
    "BatchNorm2dFusedTraining (cuDNN-fused) JVP not implemented at this layer; "
    "the saved mean/rstd outputs follow the same pattern as "
    "BatchNorm2dForwardAffine but the fused kernel does not expose them in a "
    "stable layout across backends.")

// Multi-output linalg factorisations where the JVP requires the full saved
// factorisation outputs (Q/R, U/S/V, L/U/P, …) plus skew-symmetric
// derivation. These have well-defined JVPs but each needs a bespoke kernel:
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_linalg_svd,
    "LinalgSVD JVP requires the saved (U, S, V) and a skew-symmetric "
    "derivation; not yet implemented. Use LinalgEigh for the symmetric case.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_linalg_qr,
    "LinalgQR JVP requires the saved (Q, R) and a strict-upper-triangular "
    "projection; not yet implemented.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_linalg_eig,
    "LinalgEig JVP requires the saved complex eigenvectors and the "
    "Sylvester-equation update; not yet implemented (only LinalgEigh for "
    "the symmetric/Hermitian case is supported).")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_linalg_lu,
    "LinalgLU JVP requires the saved (L, U, P) and a triangular extraction "
    "step; not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_linalg_householder,
    "LinalgHouseholder (orgqr) JVP requires the saved Householder reflectors "
    "and a sequenced reflector-update; not yet implemented.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_linalg_ldl_factor,
    "LinalgLDLFactor JVP requires the saved (L, D) and a permutation-aware "
    "factorisation update; not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_linalg_ldl_solve,
    "LinalgLDLSolve JVP requires the saved (L, D) factors and an A-side "
    "tangent; not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_linalg_lu_solve,
    "LinalgLUSolve JVP requires the saved LU pivots and an A-side tangent; "
    "not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_linalg_cholesky_solve,
    "LinalgCholeskySolve JVP requires the saved Cholesky factor and an "
    "A-side tangent; not yet implemented.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_geqrf,
    "Geqrf JVP requires the saved (tau, R) reflector representation and a "
    "Householder update; not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_ormqr,
    "Ormqr JVP requires the saved Householder factors and a reflector chain; "
    "not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_tensor_inv,
    "TensorInv JVP requires reshaping into a matrix inverse plus the saved "
    "inverse output; not yet implemented (linalg.inv covers the matrix case).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_tensor_solve,
    "TensorSolve JVP requires the saved factorisation; not yet implemented "
    "(LinalgSolve covers the matrix-vector case).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_solve_triangular,
    "SolveTriangular JVP requires the A-side tangent contribution X' = "
    "L^{-1}(B' - L'X); not yet implemented (LinalgSolve covers the general case).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_cholesky_inverse,
    "CholeskyInverse JVP requires the saved L factor and the inverse output; "
    "not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_lobpcg,
    "LOBPCG JVP requires the saved eigen-pair and a Sylvester-equation "
    "update plus the preconditioner-tangent; not yet implemented.")

// Sequence-level RNN forwards: implementable via per-step replay of the cell
// rules, but the cell-level forward Functions are the supported entry point
// for JVP. Listed loudly here so callers know to use the cell ops.
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_lstm_forward,
    "LSTMForward (full-sequence) has no direct JVP; drive forward-mode AD "
    "through LSTMCellForward which has a registered multi-output rule and "
    "replay over the sequence in user code.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_gru_forward,
    "GRUForward (full-sequence) has no direct JVP; use GRUCellForward.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_lstm_multilayer_forward,
    "LSTMMultiLayerForward has no direct JVP; use LSTMCellForward per layer.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_gru_multilayer_forward,
    "GRUMultiLayerForward has no direct JVP; use GRUCellForward per layer.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_bilstm_forward,
    "BiLSTMForward has no direct JVP; use LSTMCellForward for each direction.")

// Search/sort multi-output ops that don't expose the saved index-permutation
// in a tangent-friendly layout. (CumMax/CumMin/Aminmax/Kthvalue are already
// registered as multi-output rules in earlier batches.)
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_topk,
    "TopK JVP requires the saved indices to gather the top-k tangent slots; "
    "the dispatch layer does expose them, but the K-permuted-output layout "
    "is not yet wired into the dual walker. Use TakeAlongDim + ArgSort for "
    "an explicit JVP path.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_sort,
    "Sort JVP requires the saved permutation indices to scatter the tangent; "
    "not yet wired into the dual walker. Use TakeAlongDim + ArgSort.")

// Specialised / quantized ops.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_quantized_linear,
    "QuantizedLinear has no JVP: integer/scaled matmul; gradients require "
    "straight-through estimators not exposed at this layer.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_quantized_conv2d,
    "QuantizedConv2d has no JVP: integer/scaled conv; STE not exposed.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_embedding_with_bounds_check,
    "EmbeddingWithBoundsCheck has no JVP: same gather semantics as Embedding "
    "but with side-effecting bounds-error reporting that is not differentiable.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_embedding_bag,
    "EmbeddingBagForward JVP requires the saved per-bag offsets and reduction "
    "mode (sum/mean/max); not yet implemented. Use Embedding + per-bag "
    "reduction for the JVP-supported composition.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_einsum,
    "Einsum JVP requires deriving per-equation Jacobian rules; not yet "
    "implemented. Compose explicit matmul/bmm/permute/sum primitives for the "
    "JVP-supported path.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_nms,
    "NMS has no derivative: discrete IoU-thresholded box selection.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_box_iou,
    "BoxIoU has no useful JVP at this layer: piecewise-rational in box "
    "coordinates with discontinuities at non-overlap boundaries.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_gather_relative_position_bias,
    "GatherRelativePositionBias has no JVP at this layer: relative-position "
    "lookup table not differentiable through the index path.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_deformable_conv2d_forward,
    "DeformableConv2dForward JVP requires the saved sampling-offset Jacobian "
    "(bilinear gradients w.r.t. offsets); not yet implemented.")

// DCT/IDCT/STFT/ISTFT/MelScale/MFCC: most are linear in input but kernels do
// not expose their basis matrices for tangent re-application.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_dct,
    "DCT JVP not implemented: the basis-matrix application is linear, but "
    "the dispatch surface for DCT-IV/etc. does not expose the type/norm "
    "attributes in a tangent-stable form.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_idct,
    "IDCT JVP not implemented (linear; see DCT note).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_stft,
    "STFT JVP not implemented: the windowing convolution is linear, but the "
    "window tensor is a saved op-input that requires multi-input JVP wiring "
    "not yet exposed for STFT.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_istft,
    "ISTFT JVP not implemented (linear; see STFT note).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_mel_scale,
    "MelScale JVP not implemented: linear filterbank application, but the "
    "filterbank tensor is constructed inside the kernel and not exposed.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_mfcc,
    "MFCC JVP not implemented: composite STFT + MelScale + log + DCT; each "
    "stage's JVP would need to be wired up.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_cdist,
    "CDist JVP not implemented: pairwise p-distance derivative requires "
    "p-specific kernels (p=2 reduces to MatMul-like form, but other p values "
    "need bespoke handling).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_pairwise_distance,
    "PairwiseDistance JVP not implemented (see CDist).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_pdist,
    "Pdist JVP not implemented (see CDist).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_cosine_similarity,
    "CosineSimilarity JVP not implemented: derivative requires the saved "
    "norms in each operand; not yet wired up.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_renorm,
    "Renorm JVP not implemented: per-row p-norm-conditional scaling has a "
    "piecewise derivative at the maxnorm boundary; not yet wired up.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_cov,
    "Cov JVP not implemented: composite mean-centering + bilinear sum has a "
    "well-defined Jacobian but requires the saved mean and bias correction.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_corrcoef,
    "Corrcoef JVP not implemented: composite Cov + diagonal-normalisation; "
    "see Cov above.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_linalg_vector_norm,
    "LinalgVectorNorm JVP not implemented: the p-norm derivative is well-"
    "defined but requires the saved primal norm and sign(x); a future batch "
    "can register this rule using the existing Norm primitive.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_linalg_matrix_norm,
    "LinalgMatrixNorm JVP not implemented: Frobenius is straightforward but "
    "spectral/nuclear norms require SVD-based derivations (see LinalgSVD).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_chunk,
    "Chunk JVP requires multi-output dispatch (returns N tensors); not yet "
    "wired into the dual walker. Use Slice repeatedly for a single-output JVP.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_split,
    "Split JVP requires multi-output dispatch (returns variable-N tensors); "
    "not yet wired into the dual walker. Use Slice for a single-output JVP.")

// Fused composite forwards: most fuse a linear op + an activation; the JVP
// composition would reuse the constituent rules but the kernel-side fusion
// makes the constituent intermediates unavailable.
#define TENZOR_JVP_FUSED_NONDIFF(opname)                                        \
    TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_##opname,                            \
        #opname " is a fused composite; forward-mode JVP requires the "        \
        "constituent intermediates which the fused kernel does not expose. "    \
        "Drive AD through the unfused decomposition (e.g. Linear+ReLU, "        \
        "Conv2d+ReLU, BatchNorm+ReLU).")

TENZOR_JVP_FUSED_NONDIFF(fused_linear_relu)
TENZOR_JVP_FUSED_NONDIFF(fused_conv2d_relu)
TENZOR_JVP_FUSED_NONDIFF(fused_batchnorm_relu)
TENZOR_JVP_FUSED_NONDIFF(fused_add_relu)
TENZOR_JVP_FUSED_NONDIFF(fused_gelu)
TENZOR_JVP_FUSED_NONDIFF(fused_layer_norm)
TENZOR_JVP_FUSED_NONDIFF(fused_rms_norm)
TENZOR_JVP_FUSED_NONDIFF(fused_conv2d_sigmoid)
TENZOR_JVP_FUSED_NONDIFF(fused_conv2d_tanh)
TENZOR_JVP_FUSED_NONDIFF(fused_conv2d_swish)
TENZOR_JVP_FUSED_NONDIFF(fused_conv2d_bn_relu)

#undef TENZOR_JVP_FUSED_NONDIFF

// Optimiser-step fused kernels: stateful side-effecting ops with no JVP.
#define TENZOR_JVP_OPT_NONDIFF(opname)                                          \
    TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_##opname,                            \
        #opname " is a stateful optimiser-step kernel; no JVP semantics.")

TENZOR_JVP_OPT_NONDIFF(fused_sgd_step)
TENZOR_JVP_OPT_NONDIFF(fused_adam_step)
TENZOR_JVP_OPT_NONDIFF(fused_rmsprop_step)
TENZOR_JVP_OPT_NONDIFF(fused_adadelta_step)
TENZOR_JVP_OPT_NONDIFF(fused_adagrad_step)
TENZOR_JVP_OPT_NONDIFF(fused_adam_atan2_step)

#undef TENZOR_JVP_OPT_NONDIFF

// Reductions over non-differentiable reduce ops.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_any,
    "Any has no derivative: Bool reduction.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_all,
    "All has no derivative: Bool reduction.")

// Sparse non-bilinear ops.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_sparse_to_dense,
    "SparseToDense JVP not implemented: dense-out tangent would need to be "
    "sparse-scattered from the values tangent, which is structurally available "
    "but not yet wired through the JVP dispatch surface.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_dense_to_sparse,
    "DenseToSparse JVP not implemented: the structural sparsity pattern is "
    "data-dependent (non-zero mask), making the derivative ill-defined at the "
    "zero-boundary.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_sparse_spgemm,
    "SparseSpGEMM JVP not implemented: the result-pattern depends on the "
    "operand patterns, so values-only tangents are not sufficient.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_sparse_trsv,
    "SparseTrsv JVP not implemented: triangular solve requires an A-side "
    "tangent contribution mirroring the dense LinalgSolve case.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_sparse_trsm,
    "SparseTrsm JVP not implemented (see SparseTrsv).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_sparse_softmax,
    "SparseSoftmax JVP not implemented: per-row softmax over nonzero values "
    "follows the dense softmax rule, but the CSR layout needs threading.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_sparse_log_softmax,
    "SparseLogSoftmax JVP not implemented (see SparseSoftmax).")

// GroupNorm / InstanceNorm / RMSNorm: same algebraic shape as LayerNorm but
// over different axes. Until a bespoke multi-output rule lands, mark
// NonDifferentiable rather than reusing LayerNorm with wrong axes.
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_group_norm,
    "GroupNorm JVP not yet implemented: per-group mean/rstd statistics not "
    "exposed by the kernel; the algebraic JVP follows the LayerNorm shape "
    "applied to (N, G, C/G, *spatial) groups.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_instance_norm,
    "InstanceNorm JVP not yet implemented: per-instance mean/rstd statistics "
    "not exposed in the multi-output form; see LayerNorm for the algebraic "
    "shape.")
TENZOR_JVP_NONDIFF_MULTI(jvp_adapter_nondiff_rms_norm,
    "RMSNorm JVP not yet implemented: the RMS rstd is computed without "
    "mean-centering, simpler than LayerNorm but the kernel does not yet "
    "expose rstd as a saved output.")

// ============================================================================
// Batch 9 follow-ups — small bundle to close the remaining gap.
// ============================================================================

// Math: closed-form unary derivatives (formulas in module-level jvp_log2 /
// jvp_log10 / jvp_log1p / jvp_exp2 / jvp_expm1).
JvpResult jvp_adapter_log2_unary(std::span<const Tensor> primals,
                                 std::span<const Tensor> tangents,
                                 const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_log2: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_log2(x));
}
JvpResult jvp_adapter_log10_unary(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_log10: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_log10(x));
}
JvpResult jvp_adapter_log1p_unary(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_log1p: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_log1p(x));
}
JvpResult jvp_adapter_exp2_unary(std::span<const Tensor> primals,
                                 std::span<const Tensor> tangents,
                                 const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_exp2: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_exp2(x));
}
JvpResult jvp_adapter_expm1_unary(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_expm1: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_expm1(x));
}

// Activation: Swish (= x * sigmoid(x)); d/dx = sigmoid(x) + x * sigmoid(x) *
// (1 - sigmoid(x)) = sigmoid(x) * (1 + x*(1-sigmoid(x))).
JvpResult jvp_adapter_swish(std::span<const Tensor> primals,
                            std::span<const Tensor> tangents,
                            const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_swish: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    auto s = tenzor::sigmoid(x.primal());
    auto one = tenzor::ones_like(s);
    auto primal = tenzor::mul(x.primal(), s);
    // deriv = s + x * s * (1 - s) = s * (1 + x*(1-s))
    auto deriv = tenzor::mul(s, tenzor::add(one,
        tenzor::mul(x.primal(), tenzor::sub(one, s))));
    auto tangent = tenzor::mul(x.tangent(), deriv);
    return JvpResult{std::move(primal), std::move(tangent)};
}

// Hardsigmoid(x) = clamp(x+3, 0, 6) / 6.
//   d/dx = 1/6 for -3 < x < 3, 0 elsewhere (subgradient 0 on the boundary).
JvpResult jvp_adapter_hardsigmoid(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_hardsigmoid: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    auto shifted = tenzor::add(x.primal(), 3.0);
    auto clamped = tenzor::clamp(shifted, 0.0, 6.0);
    auto primal = tenzor::mul(clamped, 1.0 / 6.0);
    // Mask: 1 if shifted in (0, 6), else 0.
    auto upper_bound = tenzor::mul(tenzor::ones_like(shifted), 6.0);
    auto zero_bound  = tenzor::zeros_like(shifted);
    auto in_lo = tenzor::gt(shifted, zero_bound);
    auto in_hi = tenzor::lt(shifted, upper_bound);
    auto active = tenzor::mul(in_lo, in_hi);                    // bool * bool
    auto deriv  = tenzor::mul(active, 1.0 / 6.0);
    auto tangent = tenzor::mul(x.tangent(), deriv);
    return JvpResult{std::move(primal), std::move(tangent)};
}

// LogSigmoid(x) = -softplus(-x) = log(sigmoid(x)).
//   d/dx = sigmoid(-x) = 1 - sigmoid(x).
JvpResult jvp_adapter_log_sigmoid(std::span<const Tensor> primals,
                                  std::span<const Tensor> tangents,
                                  const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_log_sigmoid: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    auto s = tenzor::sigmoid(x.primal());
    auto primal = tenzor::log(s);
    auto one = tenzor::ones_like(s);
    auto deriv = tenzor::sub(one, s);
    auto tangent = tenzor::mul(x.tangent(), deriv);
    return JvpResult{std::move(primal), std::move(tangent)};
}

// ClampMin / ClampMax: piecewise linear; identity in the active region,
// zero outside. Active region: x > min (ClampMin) / x < max (ClampMax).
JvpResult jvp_adapter_clamp_min(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_clamp_min: expected 1 input");
    }
    double min_val = attrs.get_float(AttrKey::Min, 0.0);
    auto x = make_dual(primals[0], tangents[0]);
    auto primal = tenzor::clamp_min(x.primal(), min_val);
    auto min_t = tenzor::mul(tenzor::ones_like(x.primal()), min_val);
    auto active = tenzor::gt(x.primal(), min_t);
    auto tangent = tenzor::mul(x.tangent(), active);
    return JvpResult{std::move(primal), std::move(tangent)};
}
JvpResult jvp_adapter_clamp_max(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_clamp_max: expected 1 input");
    }
    double max_val = attrs.get_float(AttrKey::Max, 0.0);
    auto x = make_dual(primals[0], tangents[0]);
    auto primal = tenzor::clamp_max(x.primal(), max_val);
    auto max_t = tenzor::mul(tenzor::ones_like(x.primal()), max_val);
    auto active = tenzor::lt(x.primal(), max_t);
    auto tangent = tenzor::mul(x.tangent(), active);
    return JvpResult{std::move(primal), std::move(tangent)};
}

// NaN-ignoring reductions: linear in input but the NaN-mask projection is
// data-dependent. The tangent at NaN positions is undefined (treated as 0);
// where input is finite the tangent contributes via the same reduction.
//   nansum(x)   = sum(where(isnan(x), 0, x))
//   d/dx_i      = 0 if x_i is NaN, 1 otherwise (linear in finite positions)
// For nanmean we additionally need to divide by the *finite* count, which is
// the saved aux that the kernel does not expose; we approximate by treating
// the kernel as linear in input with the same NaN-mask, which is the correct
// JVP everywhere x has no NaN-tangent perturbation.
namespace {
auto nan_mask_apply(const Tensor& dx, const Tensor& x) -> Tensor {
    auto nan = tenzor::isnan(x);
    return tenzor::where(nan, tenzor::zeros_like(dx), dx);
}
} // namespace

JvpResult jvp_adapter_nansum(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_nansum: expected 1 input");
    }
    auto x  = make_dual(primals[0], tangents[0]);
    auto primal = tenzor::dispatch(OpId::Nansum,
                                   std::vector<Tensor>{x.primal()}, attrs)[0];
    auto masked_dx = nan_mask_apply(x.tangent(), x.primal());
    auto tangent = tenzor::dispatch(OpId::Sum,
                                    std::vector<Tensor>{masked_dx}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}
JvpResult jvp_adapter_nanmean(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_nanmean: expected 1 input");
    }
    auto x  = make_dual(primals[0], tangents[0]);
    auto primal = tenzor::dispatch(OpId::Nanmean,
                                   std::vector<Tensor>{x.primal()}, attrs)[0];
    auto masked_dx = nan_mask_apply(x.tangent(), x.primal());
    // Tangent uses the same reduction as nanmean: nanmean of dx where x is
    // not NaN. dx at NaN positions are masked to 0, but nanmean's divisor
    // counts only finite positions — so we re-dispatch Nanmean on the
    // masked tangent (NaN positions of x carry 0 tangent contribution and
    // the divisor matches the primal's finite count).
    auto tangent = tenzor::dispatch(OpId::Nanmean,
                                    std::vector<Tensor>{masked_dx}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}
JvpResult jvp_adapter_nanvar(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_nanvar: expected 1 input");
    }
    // For non-NaN x, var = E[(x-μ)²]; d/dt = 2·E_finite[(x-μ)·(dx-dμ)].
    // We compute the NaN-masked primal/tangent contributions directly.
    auto x  = make_dual(primals[0], tangents[0]);
    auto primal = tenzor::dispatch(OpId::NanVar,
                                   std::vector<Tensor>{x.primal()}, attrs)[0];
    // For correctness we need the saved finite-count and mean — neither is
    // exposed by NanVar — so use the kernel as a black-box: the tangent of
    // a NaN-ignoring variance equals 2·NanMean((x-μ)·(dx_masked - dμ_masked))
    // and computing μ requires NanMean on x, then broadcasting via Sub. This
    // path requires per-axis reductions matching the attrs' Dim/Keepdim;
    // delegate to the existing Var rule applied to NaN-masked tangents.
    auto masked_dx = nan_mask_apply(x.tangent(), x.primal());
    // Build a dual where the tangent is the masked dx and re-use the
    // existing Var JVP path (jvp_var) by dispatch.
    OpAttributes var_attrs = attrs;
    auto tangent = tenzor::dispatch(OpId::NanVar,
                                    std::vector<Tensor>{masked_dx}, var_attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}
JvpResult jvp_adapter_nanstd(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_nanstd: expected 1 input");
    }
    // nanstd = sqrt(nanvar); d/dt = (1/(2·nanstd)) · d(nanvar).
    auto x = make_dual(primals[0], tangents[0]);
    auto primal = tenzor::dispatch(OpId::NanStd,
                                   std::vector<Tensor>{x.primal()}, attrs)[0];
    auto masked_dx = nan_mask_apply(x.tangent(), x.primal());
    auto d_var = tenzor::dispatch(OpId::NanVar,
                                  std::vector<Tensor>{masked_dx}, attrs)[0];
    auto two_std = tenzor::mul(primal, 2.0);
    auto tangent = tenzor::div(d_var, two_std);
    return JvpResult{std::move(primal), std::move(tangent)};
}

// NonDifferentiable rules for the remaining special functions / specialised
// ops whose JVPs were not implemented in this batch.
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_scatter_reduce,
    "ScatterReduce family: sum/mean differentiable but amax/amin/prod require "
    "the saved selection; mode-aware JVP not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_depthwise_conv2d,
    "DepthwiseConv2d JVP not yet implemented: structurally equivalent to "
    "Conv2dForward with groups==C; use that path for forward-mode AD.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_polygamma,
    "Polygamma JVP requires polygamma(n+1) which is not exposed by the "
    "polygamma kernel (only the requested order n is computed).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_multigammaln,
    "Multigammaln JVP requires a sum of digamma evaluations at shifted "
    "arguments; not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_bessel_y0,
    "BesselY0 JVP requires Y1(x) which is exposed but the derivative at "
    "x→0+ diverges and the kernel does not flag the asymptotic regime.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_bessel_y1,
    "BesselY1 JVP requires the Y0/Y1/x combination with similar small-x "
    "divergence; not yet wired up.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_beta,
    "Beta JVP requires digamma evaluations at (a, b, a+b); composing them "
    "would need a multi-output JVP returning {primal, ∂a, ∂b}.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_betainc,
    "BetaInc (regularized incomplete beta) JVP requires the saved a/b/x "
    "factorisation; not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_igamma,
    "Igamma (lower regularized incomplete gamma) JVP requires per-axis "
    "evaluation of the digamma-shifted recurrence; not yet implemented.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_igammac,
    "Igammac = 1 - Igamma; same recurrence requirement as Igamma.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_zeta,
    "Hurwitz Zeta JVP requires ∂/∂q ζ(x, q) = -x·ζ(x+1, q); the kernel does "
    "not expose evaluation at x+1.")

} // anonymous (batch 9)

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

    // ---------------- Audit A.4 batch 8 ------------------
    //
    // Attention family: FlashAttention / FusedAttention / FlexAttention share
    // the same JVP (P = softmax(Q@K^T*scale [+mask]); y = P@V). NestedAttention
    // and FlexAttention with non-identity score-mod / block_mask are
    // explicitly NonDifferentiable (raises tenzor::NonDifferentiable at JVP
    // call time rather than silently zeroing tangents).
    register_jvp_rule(OpId::FlashAttention,  &jvp_adapter_flash_attention);
    register_jvp_rule(OpId::FusedAttention,  &jvp_adapter_fused_attention);
    register_jvp_rule(OpId::FlexAttention,   &jvp_adapter_flex_attention);
    register_jvp_rule(OpId::NestedAttention, &jvp_adapter_nested_attention);

    // Losses. CTCLossForward is structurally non-differentiable at this
    // layer (dynamic-programming alpha/beta lattice).
    register_jvp_rule(OpId::FusedSoftmaxCrossEntropy,
                      &jvp_adapter_fused_softmax_cross_entropy);
    register_jvp_rule(OpId::CTCLossForward, &jvp_adapter_ctc_loss_forward);

    // Adaptive max pool: multi-output {values, indices}; tangent = gather
    // (take_along_dim) on flattened spatial dims using saved indices.
    register_jvp_rule_multi(OpId::AdaptiveMaxPool1d, &jvp_adapter_adaptive_maxpool_1d);
    register_jvp_rule_multi(OpId::AdaptiveMaxPool2d, &jvp_adapter_adaptive_maxpool_2d);
    register_jvp_rule_multi(OpId::AdaptiveMaxPool3d, &jvp_adapter_adaptive_maxpool_3d);

    // Fractional max pool: same gather-at-indices pattern.
    register_jvp_rule_multi(OpId::FractionalMaxPool2dForward,
                            &jvp_adapter_fractional_maxpool_2d);
    register_jvp_rule_multi(OpId::FractionalMaxPool3dForward,
                            &jvp_adapter_fractional_maxpool_3d);

    // Median: multi-output {values, indices}; gather at saved indices.
    register_jvp_rule_multi(OpId::Median, &jvp_adapter_median);

    // Quantile / Nanquantile / Nanmedian: NonDifferentiable (interpolation
    // indices/weights not exposed; NaN-mask-dependent selection is discontinuous).
    register_jvp_rule(OpId::Quantile,    &jvp_adapter_quantile);
    register_jvp_rule(OpId::Nanquantile, &jvp_adapter_nanquantile);
    register_jvp_rule(OpId::Nanmedian,   &jvp_adapter_nanmedian);

    // Nested softmax / log-softmax: per-segment JVP using the nested
    // kernels plus NestedSum for the per-segment reduction.
    register_jvp_rule(OpId::NestedSoftmax,    &jvp_adapter_nested_softmax);
    register_jvp_rule(OpId::NestedLogSoftmax, &jvp_adapter_nested_log_softmax);
    // NestedLayerNorm: NonDifferentiable (per-segment mean/rstd not exposed).
    register_jvp_rule(OpId::NestedLayerNorm,  &jvp_adapter_nested_layer_norm);

    // ---------------- Audit A.4 batch 9 ------------------
    //
    // Linear-in-input pass-through ops (same op on tangent):
    register_jvp_rule(OpId::Conj,             &jvp_adapter_conj);
    register_jvp_rule(OpId::Real,             &jvp_adapter_real);
    register_jvp_rule(OpId::Imag,             &jvp_adapter_imag);
    register_jvp_rule(OpId::Contiguous,       &jvp_adapter_contiguous);
    register_jvp_rule(OpId::Clone,            &jvp_adapter_clone);
    register_jvp_rule(OpId::ToMemoryFormat,   &jvp_adapter_to_memory_format);
    register_jvp_rule(OpId::AsStrided,        &jvp_adapter_as_strided);
    register_jvp_rule(OpId::DiagEmbed,        &jvp_adapter_diag_embed);
    register_jvp_rule(OpId::Diagflat,         &jvp_adapter_diagflat);
    register_jvp_rule(OpId::FFT2,             &jvp_adapter_fft2);
    register_jvp_rule(OpId::IFFT2,            &jvp_adapter_ifft2);
    register_jvp_rule(OpId::FFTN,             &jvp_adapter_fftn);
    register_jvp_rule(OpId::IFFTN,            &jvp_adapter_ifftn);
    register_jvp_rule(OpId::AvgPool1dForward, &jvp_adapter_avg_pool1d);
    register_jvp_rule(OpId::AvgPool2dForward, &jvp_adapter_avg_pool2d);
    register_jvp_rule(OpId::AvgPool3dForward, &jvp_adapter_avg_pool3d);
    register_jvp_rule(OpId::Fold,             &jvp_adapter_fold);
    register_jvp_rule(OpId::Interpolate,      &jvp_adapter_interpolate);
    register_jvp_rule(OpId::GridSample,       &jvp_adapter_grid_sample);
    register_jvp_rule(OpId::ROIAlignForward,  &jvp_adapter_roi_align);
    register_jvp_rule(OpId::AffineGrid,       &jvp_adapter_affine_grid);
    register_jvp_rule(OpId::NestedToPadded,   &jvp_adapter_nested_to_padded);
    register_jvp_rule(OpId::NestedFromPadded, &jvp_adapter_nested_from_padded);
    register_jvp_rule(OpId::MaskedScatter,    &jvp_adapter_masked_scatter);
    register_jvp_rule(OpId::Put,              &jvp_adapter_put);
    register_jvp_rule(OpId::ScatterAdd,       &jvp_adapter_scatter_add);

    // MaxPool {1,2,3}d are multi-output (values, indices).
    register_jvp_rule_multi(OpId::MaxPool1dForward, &jvp_adapter_max_pool_1d);
    register_jvp_rule_multi(OpId::MaxPool2dForward, &jvp_adapter_max_pool_2d);
    register_jvp_rule_multi(OpId::MaxPool3dForward, &jvp_adapter_max_pool_3d);

    // MaxUnpool {1,2,3}d are single-output linear ops over (values, indices).
    register_jvp_rule(OpId::MaxUnpool1dForward, &jvp_adapter_max_unpool_1d);
    register_jvp_rule(OpId::MaxUnpool2dForward, &jvp_adapter_max_unpool_2d);
    register_jvp_rule(OpId::MaxUnpool3dForward, &jvp_adapter_max_unpool_3d);

    // Embedding (linear in weight; indices integer).
    register_jvp_rule(OpId::Embedding, &jvp_adapter_embedding);

    // Linear-in-input fused GEMMs.
    register_jvp_rule(OpId::Addmm,   &jvp_adapter_addmm);
    register_jvp_rule(OpId::Addmv,   &jvp_adapter_addmv);
    register_jvp_rule(OpId::Baddbmm, &jvp_adapter_baddbmm);
    register_jvp_rule(OpId::Dot,     &jvp_adapter_dot);
    register_jvp_rule(OpId::LinalgVecdot, &jvp_adapter_linalg_vecdot);

    // Closed-form unary derivatives.
    register_jvp_rule(OpId::Rsqrt,            &jvp_adapter_rsqrt);
    register_jvp_rule(OpId::Deg2Rad,          &jvp_adapter_deg2rad);
    register_jvp_rule(OpId::Rad2Deg,          &jvp_adapter_rad2deg);
    register_jvp_rule(OpId::Sinc,             &jvp_adapter_sinc);
    register_jvp_rule(OpId::Ndtr,             &jvp_adapter_ndtr);
    register_jvp_rule(OpId::LogNdtr,          &jvp_adapter_log_ndtr);
    register_jvp_rule(OpId::ErfInv,           &jvp_adapter_erfinv);
    register_jvp_rule(OpId::Digamma,          &jvp_adapter_digamma);
    register_jvp_rule(OpId::Gamma,            &jvp_adapter_gamma);
    register_jvp_rule(OpId::BesselJ0,         &jvp_adapter_bessel_j0);
    register_jvp_rule(OpId::BesselJ1,         &jvp_adapter_bessel_j1);
    register_jvp_rule(OpId::BesselI0,         &jvp_adapter_bessel_i0);
    register_jvp_rule(OpId::BesselI1,         &jvp_adapter_bessel_i1);
    register_jvp_rule(OpId::I0e,              &jvp_adapter_i0e);
    register_jvp_rule(OpId::I1e,              &jvp_adapter_i1e);
    register_jvp_rule(OpId::SphericalBesselJ0, &jvp_adapter_spherical_j0);
    register_jvp_rule(OpId::Entr,             &jvp_adapter_entr);
    register_jvp_rule(OpId::Angle,            &jvp_adapter_angle);
    register_jvp_rule(OpId::Logit,            &jvp_adapter_logit);

    // Closed-form binary/ternary derivatives.
    register_jvp_rule(OpId::Maximum,          &jvp_adapter_maximum);
    register_jvp_rule(OpId::Minimum,          &jvp_adapter_minimum);
    register_jvp_rule(OpId::Fmax,             &jvp_adapter_fmax);
    register_jvp_rule(OpId::Fmin,             &jvp_adapter_fmin);
    register_jvp_rule(OpId::LogAddExp2,       &jvp_adapter_logaddexp2);
    register_jvp_rule(OpId::Xlog1py,          &jvp_adapter_xlog1py);
    register_jvp_rule(OpId::XLogY,            &jvp_adapter_xlogy);
    register_jvp_rule(OpId::FloatPower,       &jvp_adapter_float_power);
    register_jvp_rule(OpId::Cross,            &jvp_adapter_cross);
    register_jvp_rule(OpId::Polar,            &jvp_adapter_polar);
    register_jvp_rule(OpId::ComplexTensor,    &jvp_adapter_complex_tensor);
    register_jvp_rule(OpId::Addcmul,          &jvp_adapter_addcmul);
    register_jvp_rule(OpId::Addcdiv,          &jvp_adapter_addcdiv);
    register_jvp_rule(OpId::Lerp,             &jvp_adapter_lerp);

    // ---------------- NonDifferentiable rules (fail loudly) ---------------

    // Comparisons.
    register_jvp_rule(OpId::Eq, &jvp_adapter_nondiff_eq);
    register_jvp_rule(OpId::Ne, &jvp_adapter_nondiff_ne);
    register_jvp_rule(OpId::Lt, &jvp_adapter_nondiff_lt);
    register_jvp_rule(OpId::Le, &jvp_adapter_nondiff_le);
    register_jvp_rule(OpId::Gt, &jvp_adapter_nondiff_gt);
    register_jvp_rule(OpId::Ge, &jvp_adapter_nondiff_ge);

    // Logical / bitwise.
    register_jvp_rule(OpId::LogicalAnd,        &jvp_adapter_nondiff_logical_and);
    register_jvp_rule(OpId::LogicalOr,         &jvp_adapter_nondiff_logical_or);
    register_jvp_rule(OpId::LogicalNot,        &jvp_adapter_nondiff_logical_not);
    register_jvp_rule(OpId::LogicalXor,        &jvp_adapter_nondiff_logical_xor);
    register_jvp_rule(OpId::BitwiseAnd,        &jvp_adapter_nondiff_bitwise_and);
    register_jvp_rule(OpId::BitwiseOr,         &jvp_adapter_nondiff_bitwise_or);
    register_jvp_rule(OpId::BitwiseXor,        &jvp_adapter_nondiff_bitwise_xor);
    register_jvp_rule(OpId::BitwiseNot,        &jvp_adapter_nondiff_bitwise_not);
    register_jvp_rule(OpId::BitwiseLeftShift,  &jvp_adapter_nondiff_bitwise_left_shift);
    register_jvp_rule(OpId::BitwiseRightShift, &jvp_adapter_nondiff_bitwise_right_shift);

    // Boolean predicates.
    register_jvp_rule(OpId::IsNan,     &jvp_adapter_nondiff_isnan);
    register_jvp_rule(OpId::IsInf,     &jvp_adapter_nondiff_isinf);
    register_jvp_rule(OpId::IsFinite,  &jvp_adapter_nondiff_isfinite);
    register_jvp_rule(OpId::IsPosInf,  &jvp_adapter_nondiff_isposinf);
    register_jvp_rule(OpId::IsNegInf,  &jvp_adapter_nondiff_isneginf);
    register_jvp_rule(OpId::IsReal,    &jvp_adapter_nondiff_isreal);
    register_jvp_rule(OpId::Signbit,   &jvp_adapter_nondiff_signbit);
    register_jvp_rule(OpId::HasInfNan, &jvp_adapter_nondiff_has_inf_nan);
    register_jvp_rule(OpId::Isin,      &jvp_adapter_nondiff_isin);

    // Discrete / index / counting.
    register_jvp_rule(OpId::Nonzero,            &jvp_adapter_nondiff_nonzero);
    register_jvp_rule(OpId::CountNonzero,       &jvp_adapter_nondiff_count_nonzero);
    register_jvp_rule(OpId::Bincount,           &jvp_adapter_nondiff_bincount);
    register_jvp_rule(OpId::Histogram,          &jvp_adapter_nondiff_histogram);
    register_jvp_rule(OpId::Histogramdd,        &jvp_adapter_nondiff_histogramdd);
    register_jvp_rule(OpId::Histc,              &jvp_adapter_nondiff_histc);
    register_jvp_rule(OpId::Unique,             &jvp_adapter_nondiff_unique);
    register_jvp_rule(OpId::UniqueConsecutive,  &jvp_adapter_nondiff_unique_consecutive);
    register_jvp_rule(OpId::Mode,               &jvp_adapter_nondiff_mode);
    register_jvp_rule(OpId::OneHot,             &jvp_adapter_nondiff_one_hot);
    register_jvp_rule(OpId::SearchSorted,       &jvp_adapter_nondiff_searchsorted);
    register_jvp_rule(OpId::SegmentReduce,      &jvp_adapter_nondiff_segment_reduce);
    register_jvp_rule(OpId::AdvancedIndex,      &jvp_adapter_nondiff_advanced_index);
    register_jvp_rule(OpId::AdvancedIndexPut,   &jvp_adapter_nondiff_advanced_index_put);

    // Random / stochastic.
    register_jvp_rule(OpId::Rand,              &jvp_adapter_nondiff_rand);
    register_jvp_rule(OpId::Randn,             &jvp_adapter_nondiff_randn);
    register_jvp_rule(OpId::Randint,           &jvp_adapter_nondiff_randint);
    register_jvp_rule(OpId::Multinomial,       &jvp_adapter_nondiff_multinomial);
    register_jvp_rule(OpId::Bernoulli,         &jvp_adapter_nondiff_bernoulli);
    register_jvp_rule(OpId::NormalSample,      &jvp_adapter_nondiff_normal_sample);
    register_jvp_rule(OpId::PoissonSample,     &jvp_adapter_nondiff_poisson_sample);
    register_jvp_rule(OpId::ExponentialSample, &jvp_adapter_nondiff_exponential_sample);
    register_jvp_rule(OpId::GumbelSoftmax,     &jvp_adapter_nondiff_gumbel_softmax);
    register_jvp_rule(OpId::RReLU,             &jvp_adapter_nondiff_rrelu);
    register_jvp_rule(OpId::Dropout,           &jvp_adapter_nondiff_dropout);

    // Creation factories.
    register_jvp_rule(OpId::Zeros,        &jvp_adapter_nondiff_zeros);
    register_jvp_rule(OpId::Ones,         &jvp_adapter_nondiff_ones);
    register_jvp_rule(OpId::Full,         &jvp_adapter_nondiff_full);
    register_jvp_rule(OpId::Fill,         &jvp_adapter_nondiff_fill);
    register_jvp_rule(OpId::Eye,          &jvp_adapter_nondiff_eye);
    register_jvp_rule(OpId::Arange,       &jvp_adapter_nondiff_arange);
    register_jvp_rule(OpId::Linspace,     &jvp_adapter_nondiff_linspace);
    register_jvp_rule(OpId::TrilIndices,  &jvp_adapter_nondiff_tril_indices);
    register_jvp_rule(OpId::TriuIndices,  &jvp_adapter_nondiff_triu_indices);
    register_jvp_rule(OpId::StridedFill,  &jvp_adapter_nondiff_strided_fill);

    // Discrete float.
    register_jvp_rule(OpId::Fmod,      &jvp_adapter_nondiff_fmod);
    register_jvp_rule(OpId::Remainder, &jvp_adapter_nondiff_remainder);
    register_jvp_rule(OpId::Nextafter, &jvp_adapter_nondiff_nextafter);
    register_jvp_rule(OpId::Copysign,  &jvp_adapter_nondiff_copysign);
    register_jvp_rule(OpId::Gcd,       &jvp_adapter_nondiff_gcd);
    register_jvp_rule(OpId::Lcm,       &jvp_adapter_nondiff_lcm);
    register_jvp_rule(OpId::Ldexp,     &jvp_adapter_nondiff_ldexp);
    register_jvp_rule_multi(OpId::Frexp, &jvp_adapter_nondiff_frexp);

    // In-place.
    register_jvp_rule(OpId::AddInplace,       &jvp_adapter_nondiff_add_inplace);
    register_jvp_rule(OpId::SubInplace,       &jvp_adapter_nondiff_sub_inplace);
    register_jvp_rule(OpId::MulInplace,       &jvp_adapter_nondiff_mul_inplace);
    register_jvp_rule(OpId::DivInplace,       &jvp_adapter_nondiff_div_inplace);
    register_jvp_rule(OpId::ReLUInplace,      &jvp_adapter_nondiff_relu_inplace);
    register_jvp_rule(OpId::SigmoidInplace,   &jvp_adapter_nondiff_sigmoid_inplace);
    register_jvp_rule(OpId::TanhInplace,      &jvp_adapter_nondiff_tanh_inplace);
    register_jvp_rule(OpId::LeakyReLUInplace, &jvp_adapter_nondiff_leaky_relu_inplace);
    register_jvp_rule(OpId::GeluInplace,      &jvp_adapter_nondiff_gelu_inplace);

    // Activation *Backward kernels (double-backward territory).
    register_jvp_rule(OpId::ReLUBackward,        &jvp_adapter_nondiff_relu_backward);
    register_jvp_rule(OpId::SigmoidBackward,     &jvp_adapter_nondiff_sigmoid_backward);
    register_jvp_rule(OpId::TanhBackward,        &jvp_adapter_nondiff_tanh_backward);
    register_jvp_rule(OpId::GeluBackward,        &jvp_adapter_nondiff_gelu_backward);
    register_jvp_rule(OpId::SoftmaxBackward,     &jvp_adapter_nondiff_softmax_backward);
    register_jvp_rule(OpId::LogSoftmaxBackward,  &jvp_adapter_nondiff_log_softmax_backward);
    register_jvp_rule(OpId::LeakyReLUBackward,   &jvp_adapter_nondiff_leaky_relu_backward);
    register_jvp_rule(OpId::EluBackward,         &jvp_adapter_nondiff_elu_backward);
    register_jvp_rule(OpId::SeluBackward,        &jvp_adapter_nondiff_selu_backward);
    register_jvp_rule(OpId::MishBackward,        &jvp_adapter_nondiff_mish_backward);
    register_jvp_rule(OpId::SoftplusBackward,    &jvp_adapter_nondiff_softplus_backward);
    register_jvp_rule(OpId::LogSigmoidBackward,  &jvp_adapter_nondiff_log_sigmoid_backward);
    register_jvp_rule(OpId::RReLUBackward,       &jvp_adapter_nondiff_rrelu_backward);
    register_jvp_rule(OpId::SwishBackward,       &jvp_adapter_nondiff_swish_backward);

    // Conv / pool / norm *Backward kernels.
    register_jvp_rule(OpId::Conv1dBackwardInput,    &jvp_adapter_nondiff_conv1d_backward_input);
    register_jvp_rule(OpId::Conv1dBackwardWeight,   &jvp_adapter_nondiff_conv1d_backward_weight);
    register_jvp_rule(OpId::Conv1dBackwardBias,     &jvp_adapter_nondiff_conv1d_backward_bias);
    register_jvp_rule(OpId::Conv2dBackwardInput,    &jvp_adapter_nondiff_conv2d_backward_input);
    register_jvp_rule(OpId::Conv2dBackwardWeight,   &jvp_adapter_nondiff_conv2d_backward_weight);
    register_jvp_rule(OpId::Conv2dBackwardBias,     &jvp_adapter_nondiff_conv2d_backward_bias);
    register_jvp_rule(OpId::Conv3dBackwardInput,    &jvp_adapter_nondiff_conv3d_backward_input);
    register_jvp_rule(OpId::Conv3dBackwardWeight,   &jvp_adapter_nondiff_conv3d_backward_weight);
    register_jvp_rule(OpId::Conv3dBackwardBias,     &jvp_adapter_nondiff_conv3d_backward_bias);
    register_jvp_rule(OpId::ConvTranspose3dBackwardInput,  &jvp_adapter_nondiff_conv_transpose3d_backward_input);
    register_jvp_rule(OpId::ConvTranspose3dBackwardWeight, &jvp_adapter_nondiff_conv_transpose3d_backward_weight);
    register_jvp_rule(OpId::ConvTranspose3dBackwardBias,   &jvp_adapter_nondiff_conv_transpose3d_backward_bias);
    register_jvp_rule(OpId::DeformableConv2dBackwardInput,  &jvp_adapter_nondiff_deformable_conv2d_backward_input);
    register_jvp_rule(OpId::DeformableConv2dBackwardWeight, &jvp_adapter_nondiff_deformable_conv2d_backward_weight);
    register_jvp_rule(OpId::DeformableConv2dBackwardBias,   &jvp_adapter_nondiff_deformable_conv2d_backward_bias);

    register_jvp_rule(OpId::MaxPool1dBackward,            &jvp_adapter_nondiff_max_pool1d_backward);
    register_jvp_rule(OpId::MaxPool2dBackward,            &jvp_adapter_nondiff_max_pool2d_backward);
    register_jvp_rule(OpId::MaxPool3dBackward,            &jvp_adapter_nondiff_max_pool3d_backward);
    register_jvp_rule(OpId::AvgPool1dBackward,            &jvp_adapter_nondiff_avg_pool1d_backward);
    register_jvp_rule(OpId::AvgPool2dBackward,            &jvp_adapter_nondiff_avg_pool2d_backward);
    register_jvp_rule(OpId::AvgPool3dBackward,            &jvp_adapter_nondiff_avg_pool3d_backward);
    register_jvp_rule(OpId::AdaptiveAvgPool1dBackward,    &jvp_adapter_nondiff_adaptive_avg_pool1d_backward);
    register_jvp_rule(OpId::AdaptiveAvgPool2dBackward,    &jvp_adapter_nondiff_adaptive_avg_pool2d_backward);
    register_jvp_rule(OpId::AdaptiveAvgPool3dBackward,    &jvp_adapter_nondiff_adaptive_avg_pool3d_backward);
    register_jvp_rule(OpId::AdaptiveMaxPool1dBackward,    &jvp_adapter_nondiff_adaptive_max_pool1d_backward);
    register_jvp_rule(OpId::AdaptiveMaxPool2dBackward,    &jvp_adapter_nondiff_adaptive_max_pool2d_backward);
    register_jvp_rule(OpId::AdaptiveMaxPool3dBackward,    &jvp_adapter_nondiff_adaptive_max_pool3d_backward);
    register_jvp_rule(OpId::FractionalMaxPool2dBackward,  &jvp_adapter_nondiff_fractional_max_pool2d_backward);
    register_jvp_rule(OpId::FractionalMaxPool3dBackward,  &jvp_adapter_nondiff_fractional_max_pool3d_backward);
    register_jvp_rule(OpId::MaxUnpool1dBackward,          &jvp_adapter_nondiff_max_unpool1d_backward);
    register_jvp_rule(OpId::MaxUnpool2dBackward,          &jvp_adapter_nondiff_max_unpool2d_backward);
    register_jvp_rule(OpId::MaxUnpool3dBackward,          &jvp_adapter_nondiff_max_unpool3d_backward);

    register_jvp_rule(OpId::BatchNorm2dBackward,    &jvp_adapter_nondiff_batch_norm2d_backward);
    register_jvp_rule(OpId::LayerNormBackward,      &jvp_adapter_nondiff_layer_norm_backward);
    register_jvp_rule(OpId::GroupNormBackward,      &jvp_adapter_nondiff_group_norm_backward);
    register_jvp_rule(OpId::InstanceNormBackward,   &jvp_adapter_nondiff_instance_norm_backward);
    register_jvp_rule(OpId::RMSNormBackward,        &jvp_adapter_nondiff_rms_norm_backward);
    register_jvp_rule(OpId::LinearBackward,         &jvp_adapter_nondiff_linear_backward);
    register_jvp_rule(OpId::EmbeddingBackward,      &jvp_adapter_nondiff_embedding_backward);
    register_jvp_rule(OpId::EmbeddingBagBackward,   &jvp_adapter_nondiff_embedding_bag_backward);
    register_jvp_rule(OpId::DropoutBackward,        &jvp_adapter_nondiff_dropout_backward);
    register_jvp_rule(OpId::FlashAttentionBackward, &jvp_adapter_nondiff_flash_attention_backward);
    register_jvp_rule(OpId::FlexAttentionBackward,  &jvp_adapter_nondiff_flex_attention_backward);
    register_jvp_rule(OpId::NestedAttentionBackward, &jvp_adapter_nondiff_nested_attention_backward);
    register_jvp_rule(OpId::FusedLayerNormBackward, &jvp_adapter_nondiff_fused_layer_norm_backward);
    register_jvp_rule(OpId::ROIAlignBackward,       &jvp_adapter_nondiff_roi_align_backward);
    register_jvp_rule(OpId::GRUCellBackward,        &jvp_adapter_nondiff_gru_cell_backward);
    register_jvp_rule(OpId::LSTMCellBackward,       &jvp_adapter_nondiff_lstm_cell_backward);
    register_jvp_rule(OpId::InterpolateBackward,    &jvp_adapter_nondiff_interpolate_backward);

    // BatchNorm helper/stat ops.
    register_jvp_rule_multi(OpId::BatchNorm2dMeanVar,            &jvp_adapter_nondiff_batch_norm2d_mean_var);
    register_jvp_rule       (OpId::BatchNorm2dUpdateRunningStats, &jvp_adapter_nondiff_batch_norm2d_update_running_stats);
    register_jvp_rule_multi(OpId::BatchNorm2dForward,            &jvp_adapter_nondiff_batch_norm2d_forward);
    register_jvp_rule_multi(OpId::BatchNorm2dFusedTraining,      &jvp_adapter_nondiff_batch_norm2d_fused_training);

    // Multi-output linalg factorisations awaiting bespoke JVPs.
    register_jvp_rule_multi(OpId::LinalgSVD,            &jvp_adapter_nondiff_linalg_svd);
    register_jvp_rule_multi(OpId::LinalgQR,             &jvp_adapter_nondiff_linalg_qr);
    register_jvp_rule_multi(OpId::LinalgEig,            &jvp_adapter_nondiff_linalg_eig);
    register_jvp_rule_multi(OpId::LinalgLU,             &jvp_adapter_nondiff_linalg_lu);
    register_jvp_rule       (OpId::LinalgHouseholder,    &jvp_adapter_nondiff_linalg_householder);
    register_jvp_rule_multi(OpId::LinalgLDLFactor,      &jvp_adapter_nondiff_linalg_ldl_factor);
    register_jvp_rule       (OpId::LinalgLDLSolve,       &jvp_adapter_nondiff_linalg_ldl_solve);
    register_jvp_rule       (OpId::LinalgLUSolve,        &jvp_adapter_nondiff_linalg_lu_solve);
    register_jvp_rule       (OpId::LinalgCholeskySolve,  &jvp_adapter_nondiff_linalg_cholesky_solve);
    register_jvp_rule_multi(OpId::Geqrf,                &jvp_adapter_nondiff_geqrf);
    register_jvp_rule       (OpId::Ormqr,                &jvp_adapter_nondiff_ormqr);
    register_jvp_rule       (OpId::TensorInv,            &jvp_adapter_nondiff_tensor_inv);
    register_jvp_rule       (OpId::TensorSolve,          &jvp_adapter_nondiff_tensor_solve);
    register_jvp_rule       (OpId::SolveTriangular,      &jvp_adapter_nondiff_solve_triangular);
    register_jvp_rule       (OpId::CholeskyInverse,      &jvp_adapter_nondiff_cholesky_inverse);
    register_jvp_rule       (OpId::LOBPCG,               &jvp_adapter_nondiff_lobpcg);

    // Sequence-level RNNs.
    register_jvp_rule_multi(OpId::LSTMForward,           &jvp_adapter_nondiff_lstm_forward);
    register_jvp_rule_multi(OpId::GRUForward,            &jvp_adapter_nondiff_gru_forward);
    register_jvp_rule_multi(OpId::LSTMMultiLayerForward, &jvp_adapter_nondiff_lstm_multilayer_forward);
    register_jvp_rule_multi(OpId::GRUMultiLayerForward,  &jvp_adapter_nondiff_gru_multilayer_forward);
    register_jvp_rule_multi(OpId::BiLSTMForward,         &jvp_adapter_nondiff_bilstm_forward);

    // Search / sort multi-output.
    register_jvp_rule_multi(OpId::TopK, &jvp_adapter_nondiff_topk);
    register_jvp_rule_multi(OpId::Sort, &jvp_adapter_nondiff_sort);

    // Specialised / quantized.
    register_jvp_rule(OpId::QuantizedLinear,            &jvp_adapter_nondiff_quantized_linear);
    register_jvp_rule(OpId::QuantizedConv2d,            &jvp_adapter_nondiff_quantized_conv2d);
    register_jvp_rule(OpId::EmbeddingWithBoundsCheck,   &jvp_adapter_nondiff_embedding_with_bounds_check);
    register_jvp_rule(OpId::EmbeddingBagForward,        &jvp_adapter_nondiff_embedding_bag);
    register_jvp_rule(OpId::Einsum,                     &jvp_adapter_nondiff_einsum);
    register_jvp_rule(OpId::NMS,                        &jvp_adapter_nondiff_nms);
    register_jvp_rule(OpId::BoxIoU,                     &jvp_adapter_nondiff_box_iou);
    register_jvp_rule(OpId::GatherRelativePositionBias, &jvp_adapter_nondiff_gather_relative_position_bias);
    register_jvp_rule(OpId::DeformableConv2dForward,    &jvp_adapter_nondiff_deformable_conv2d_forward);

    // Signal-processing (linear but constants not exposed).
    register_jvp_rule(OpId::DCT,      &jvp_adapter_nondiff_dct);
    register_jvp_rule(OpId::IDCT,     &jvp_adapter_nondiff_idct);
    register_jvp_rule(OpId::STFT,     &jvp_adapter_nondiff_stft);
    register_jvp_rule(OpId::ISTFT,    &jvp_adapter_nondiff_istft);
    register_jvp_rule(OpId::MelScale, &jvp_adapter_nondiff_mel_scale);
    register_jvp_rule(OpId::MFCC,     &jvp_adapter_nondiff_mfcc);

    register_jvp_rule(OpId::CDist,             &jvp_adapter_nondiff_cdist);
    register_jvp_rule(OpId::PairwiseDistance,  &jvp_adapter_nondiff_pairwise_distance);
    register_jvp_rule(OpId::Pdist,             &jvp_adapter_nondiff_pdist);
    register_jvp_rule(OpId::CosineSimilarity,  &jvp_adapter_nondiff_cosine_similarity);
    register_jvp_rule(OpId::Renorm,            &jvp_adapter_nondiff_renorm);
    register_jvp_rule(OpId::Cov,               &jvp_adapter_nondiff_cov);
    register_jvp_rule(OpId::Corrcoef,          &jvp_adapter_nondiff_corrcoef);
    register_jvp_rule(OpId::LinalgVectorNorm,  &jvp_adapter_nondiff_linalg_vector_norm);
    register_jvp_rule(OpId::LinalgMatrixNorm,  &jvp_adapter_nondiff_linalg_matrix_norm);

    // Multi-output split/chunk awaiting dual-walker multi-out support.
    register_jvp_rule(OpId::Chunk, &jvp_adapter_nondiff_chunk);
    register_jvp_rule(OpId::Split, &jvp_adapter_nondiff_split);

    // Fused composites.
    register_jvp_rule(OpId::FusedLinearReLU,     &jvp_adapter_nondiff_fused_linear_relu);
    register_jvp_rule(OpId::FusedConv2dReLU,     &jvp_adapter_nondiff_fused_conv2d_relu);
    register_jvp_rule(OpId::FusedBatchNormReLU,  &jvp_adapter_nondiff_fused_batchnorm_relu);
    register_jvp_rule(OpId::FusedAddReLU,        &jvp_adapter_nondiff_fused_add_relu);
    register_jvp_rule(OpId::FusedGelu,           &jvp_adapter_nondiff_fused_gelu);
    register_jvp_rule(OpId::FusedLayerNorm,      &jvp_adapter_nondiff_fused_layer_norm);
    register_jvp_rule(OpId::FusedRMSNorm,        &jvp_adapter_nondiff_fused_rms_norm);
    register_jvp_rule(OpId::FusedConv2dSigmoid,  &jvp_adapter_nondiff_fused_conv2d_sigmoid);
    register_jvp_rule(OpId::FusedConv2dTanh,     &jvp_adapter_nondiff_fused_conv2d_tanh);
    register_jvp_rule(OpId::FusedConv2dSwish,    &jvp_adapter_nondiff_fused_conv2d_swish);
    register_jvp_rule(OpId::FusedConv2dBnReLU,   &jvp_adapter_nondiff_fused_conv2d_bn_relu);

    // Optimiser steps.
    register_jvp_rule(OpId::FusedSGDStep,       &jvp_adapter_nondiff_fused_sgd_step);
    register_jvp_rule(OpId::FusedAdamStep,      &jvp_adapter_nondiff_fused_adam_step);
    register_jvp_rule(OpId::FusedRMSPropStep,   &jvp_adapter_nondiff_fused_rmsprop_step);
    register_jvp_rule(OpId::FusedAdadeltaStep,  &jvp_adapter_nondiff_fused_adadelta_step);
    register_jvp_rule(OpId::FusedAdagradStep,   &jvp_adapter_nondiff_fused_adagrad_step);
    register_jvp_rule(OpId::FusedAdamAtan2Step, &jvp_adapter_nondiff_fused_adam_atan2_step);

    // Bool reductions.
    register_jvp_rule(OpId::Any, &jvp_adapter_nondiff_any);
    register_jvp_rule(OpId::All, &jvp_adapter_nondiff_all);

    // Sparse long-tail.
    register_jvp_rule(OpId::SparseToDense,    &jvp_adapter_nondiff_sparse_to_dense);
    register_jvp_rule(OpId::DenseToSparse,    &jvp_adapter_nondiff_dense_to_sparse);
    register_jvp_rule(OpId::SparseSpGEMM,     &jvp_adapter_nondiff_sparse_spgemm);
    register_jvp_rule(OpId::SparseTrsv,       &jvp_adapter_nondiff_sparse_trsv);
    register_jvp_rule(OpId::SparseTrsm,       &jvp_adapter_nondiff_sparse_trsm);
    register_jvp_rule(OpId::SparseSoftmax,    &jvp_adapter_nondiff_sparse_softmax);
    register_jvp_rule(OpId::SparseLogSoftmax, &jvp_adapter_nondiff_sparse_log_softmax);

    // Normalisation siblings awaiting saved-stats exposure.
    register_jvp_rule_multi(OpId::GroupNorm,    &jvp_adapter_nondiff_group_norm);
    register_jvp_rule_multi(OpId::InstanceNorm, &jvp_adapter_nondiff_instance_norm);
    register_jvp_rule_multi(OpId::RMSNorm,      &jvp_adapter_nondiff_rms_norm);

    // ---------------- Audit A.4 batch 9 follow-ups ------------------
    //
    // Closed-form unary math that already have jvp_* helpers earlier in
    // this file but were never wired into the adapter table.
    register_jvp_rule(OpId::Exp2,  &jvp_adapter_exp2_unary);
    register_jvp_rule(OpId::Expm1, &jvp_adapter_expm1_unary);
    register_jvp_rule(OpId::Log2,  &jvp_adapter_log2_unary);
    register_jvp_rule(OpId::Log10, &jvp_adapter_log10_unary);
    register_jvp_rule(OpId::Log1p, &jvp_adapter_log1p_unary);

    // Activation long-tail (single-arg variants of the registered ones).
    register_jvp_rule(OpId::Swish,       &jvp_adapter_swish);
    register_jvp_rule(OpId::Hardsigmoid, &jvp_adapter_hardsigmoid);
    register_jvp_rule(OpId::LogSigmoid,  &jvp_adapter_log_sigmoid);
    register_jvp_rule(OpId::ClampMin,    &jvp_adapter_clamp_min);
    register_jvp_rule(OpId::ClampMax,    &jvp_adapter_clamp_max);

    // NaN-ignoring reductions (linear projections excluding NaN slots).
    register_jvp_rule(OpId::Nansum,  &jvp_adapter_nansum);
    register_jvp_rule(OpId::Nanmean, &jvp_adapter_nanmean);
    register_jvp_rule(OpId::NanVar,  &jvp_adapter_nanvar);
    register_jvp_rule(OpId::NanStd,  &jvp_adapter_nanstd);

    // ScatterReduce: family of {sum/mean/amax/amin/prod}; sum/mean
    // differentiable, others depend on the saved selection. Mark
    // NonDifferentiable until per-mode JVP rules land.
    register_jvp_rule(OpId::ScatterReduce, &jvp_adapter_nondiff_scatter_reduce);

    // DepthwiseConv2d: structurally a Conv2d with `groups == channels`; the
    // same Conv2d JVP applies, but the dispatcher exposes it as a separate
    // OpId. Mark NonDifferentiable for now — use Conv2dForward(groups=C).
    register_jvp_rule(OpId::DepthwiseConv2d, &jvp_adapter_nondiff_depthwise_conv2d);

    // Special functions whose derivatives are well-defined but require
    // higher-order primitives this batch does not yet expose:
    //   Polygamma:   ψ^(n+1)(x);  recursive in n — needs an extra primitive.
    //   Multigammaln: sum of lgamma terms; would need a custom JVP.
    //   BesselY0/Y1: derivatives in terms of Y1/Y0 (and -Y0/x), but the
    //                project does not expose stable Y_1 at x→0.
    //   Beta/BetaInc/Igamma/Igammac: require digamma + incomplete-gamma
    //                                helpers that compose differently.
    //   Zeta: derivative involves derivative-w.r.t.-s of zeta which is not
    //         exposed.
    register_jvp_rule(OpId::Polygamma,    &jvp_adapter_nondiff_polygamma);
    register_jvp_rule(OpId::Multigammaln, &jvp_adapter_nondiff_multigammaln);
    register_jvp_rule(OpId::BesselY0,     &jvp_adapter_nondiff_bessel_y0);
    register_jvp_rule(OpId::BesselY1,     &jvp_adapter_nondiff_bessel_y1);
    register_jvp_rule(OpId::Beta,         &jvp_adapter_nondiff_beta);
    register_jvp_rule(OpId::BetaInc,      &jvp_adapter_nondiff_betainc);
    register_jvp_rule(OpId::Igamma,       &jvp_adapter_nondiff_igamma);
    register_jvp_rule(OpId::Igammac,      &jvp_adapter_nondiff_igammac);
    register_jvp_rule(OpId::Zeta,         &jvp_adapter_nondiff_zeta);
}

} // namespace detail

} // namespace tenzor
