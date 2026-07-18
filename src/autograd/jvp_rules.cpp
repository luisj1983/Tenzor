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
#include "tenzor/utils/safe_math.hpp"
#include <array>
#include <climits>
#include <cmath>
#include <complex>
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
    if (x.primal().is_complex()) {
        // |z| is real; d|z| = Re(conj(z)·dz) / |z|.  (J-03)
        // Natural div: at |z|=0 this yields 0/0 -> NaN, matching the way the
        // real branch's sign(0)=0 leaves the subgradient undefined at the cusp;
        // no epsilon convention exists for abs elsewhere in this file, so we do
        // not fabricate one.
        auto num = tenzor::real(tenzor::mul(tenzor::conj(x.primal()), x.tangent()));
        auto tangent = tenzor::div(num, primal);
        return DualTensor(std::move(primal), std::move(tangent));
    }
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
    auto primal = tenzor::clamp(x.primal(), min_val, max_val);
    // INCLUSIVE boundary mask, mirroring reverse-mode ClampBackward exactly so
    // forward-mode and reverse-mode AD agree at x == min and x == max:
    //     mask = 1 - |sign(x - clamp(x, min, max))|
    // which is 1 in the interior AND at the exact boundary (sign(0)==0), 0
    // outside. The previous strict gt/lt mask zeroed the tangent at points
    // sitting exactly on a clamp bound (e.g. clamp_min(0) on integer-valued
    // data), diverging from ClampBackward's pass-through there.
    auto clamped   = tenzor::clamp(x.primal(), min_val, max_val);
    auto diff      = tenzor::sub(x.primal(), clamped);
    auto mask      = tenzor::sub(tenzor::ones_like(x.primal()),
                                 tenzor::abs(tenzor::sign(diff)));
    auto tangent = tenzor::mul(x.tangent(), mask);
    return DualTensor(std::move(primal), std::move(tangent));
}

// ============================================================================
// Activation extensions
// ============================================================================

auto jvp_leaky_relu(const DualTensor& x, double negative_slope) -> DualTensor {
    // V.6: double end-to-end so Float64 callers don't silently narrow the
    // scalar parameter through float at the helper boundary.
    // leaky_relu(x) = x if x > 0, else negative_slope * x
    auto p = x.primal();
    auto zero = tenzor::zeros_like(p);
    auto pos_mask = tenzor::gt(p, zero);  // 1 where x > 0
    // derivative: 1 where x > 0, negative_slope elsewhere
    auto one = tenzor::ones_like(p);
    auto slope = tenzor::mul(tenzor::ones_like(p), negative_slope);
    // deriv = pos_mask * 1 + (1 - pos_mask) * negative_slope
    auto deriv = tenzor::add(tenzor::mul(pos_mask, one),
                             tenzor::mul(tenzor::sub(one, pos_mask), slope));
    auto primal = tenzor::mul(p, deriv);  // equivalent to leaky_relu
    auto tangent = tenzor::mul(x.tangent(), deriv);
    return DualTensor(std::move(primal), std::move(tangent));
}

auto jvp_elu(const DualTensor& x, double alpha) -> DualTensor {
    // V.6: double end-to-end.
    // elu(x) = x if x > 0, else alpha * (exp(x) - 1)
    // d(elu)/dx = 1 if x > 0, else alpha * exp(x)
    auto p = x.primal();
    auto zero = tenzor::zeros_like(p);
    auto pos_mask = tenzor::gt(p, zero);
    auto one = tenzor::ones_like(p);
    auto exp_x = tenzor::exp(p);
    auto neg_deriv = tenzor::mul(exp_x, alpha);
    auto deriv = tenzor::add(tenzor::mul(pos_mask, one),
                             tenzor::mul(tenzor::sub(one, pos_mask), neg_deriv));
    // primal
    auto neg_val = tenzor::mul(tenzor::sub(exp_x, one), alpha);
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

auto jvp_softplus(const DualTensor& x, double beta) -> DualTensor {
    // V.6: double end-to-end.
    // softplus(x) = (1/beta) * log(1 + exp(beta*x)); derivative = sigmoid(beta*x).
    // Compute the primal in the numerically stable form
    //   (1/beta) * (max(bx, 0) + log1p(exp(-|bx|)))
    // so that large beta*x does not overflow exp() to +inf (the naive
    // log(1+exp(bx)) does). This matches the backend Softplus kernel and the
    // max-shifted jvp_softmax/jvp_logsumexp in this file.
    auto bx = tenzor::mul(x.primal(), beta);
    auto pos = tenzor::maximum(bx, tenzor::zeros_like(bx));
    auto stable = tenzor::add(pos, tenzor::log1p(tenzor::exp(tenzor::neg(tenzor::abs(bx)))));
    auto primal = tenzor::mul(stable, 1.0 / beta);
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
    // Widen Float16/BFloat16 to Float32 for the reduction-heavy softmax chain
    // (catastrophic cancellation in the exp/sum), mirroring the reverse-mode
    // SoftmaxBackward's own widen-then-narrow and NestedSoftmaxBackward.
    const DType orig_dtype = x.primal().dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    auto p = widen ? x.primal().to(DType::Float32) : x.primal();
    auto dt = widen ? x.tangent().to(DType::Float32) : x.tangent();
    // Compute softmax manually: exp(x - max) / sum(exp(x - max))
    auto max_val = tenzor::max(p, dim, /*keepdim=*/true);
    auto shifted = tenzor::sub(p, max_val);
    auto exp_shifted = tenzor::exp(shifted);
    auto sum_exp = tenzor::sum(exp_shifted, dim, /*keepdim=*/true);
    auto s = tenzor::div(exp_shifted, sum_exp);

    auto s_dt = tenzor::mul(s, dt);
    auto sum_s_dt = tenzor::sum(s_dt, dim, /*keepdim=*/true);
    auto tangent = tenzor::mul(s, tenzor::sub(dt, sum_s_dt));
    if (widen) {
        s = s.to(orig_dtype);
        tangent = tangent.to(orig_dtype);
    }
    return DualTensor(std::move(s), std::move(tangent));
}

auto jvp_log_softmax(const DualTensor& x, int64_t dim) -> DualTensor {
    // log_softmax(x) = x - log(sum(exp(x), dim))
    // d(log_softmax)_i = dt_i - d(logsumexp) = dt_i - sum_j(softmax(x)_j * dt_j)
    // (the subtracted term is a per-row scalar, broadcast along dim)
    // Widen Float16/BFloat16 to Float32 — see jvp_softmax.
    const DType orig_dtype = x.primal().dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    auto p = widen ? x.primal().to(DType::Float32) : x.primal();
    auto dt = widen ? x.tangent().to(DType::Float32) : x.tangent();
    auto max_val = tenzor::max(p, dim, /*keepdim=*/true);
    auto shifted = tenzor::sub(p, max_val);
    auto exp_shifted = tenzor::exp(shifted);
    auto sum_exp = tenzor::sum(exp_shifted, dim, /*keepdim=*/true);
    auto log_sum_exp = tenzor::add(tenzor::log(sum_exp), max_val);
    auto primal = tenzor::sub(p, log_sum_exp);
    auto s = tenzor::div(exp_shifted, sum_exp);  // softmax

    auto s_dt = tenzor::mul(s, dt);
    auto sum_s_dt = tenzor::sum(s_dt, dim, /*keepdim=*/true);
    auto tangent = tenzor::sub(dt, sum_s_dt);
    if (widen) {
        primal = primal.to(orig_dtype);
        tangent = tangent.to(orig_dtype);
    }
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
    // M26: replace the naive `dx/x` ratio (NaN whenever ANY reduced element
    // is exactly zero) with the exact, zero-aware per-element factor —
    // mirrors ProdBackward's own compute_prod_backward_factor
    // (function_reduction_ext.cpp, anonymous-namespace / not directly
    // callable from here, so duplicated in this file's own self-contained
    // tensor-op style like every other JVP rule): for each position i,
    // factor_i = product of all OTHER elements in its reduction group,
    // which is finite even when x_i == 0 (or when exactly one OTHER element
    // in the group is 0), and exactly 0 when 2+ elements in the group are
    // zero. tangent = sum_i factor_i * dx_i is the forward-mode counterpart
    // of the VJP's grad_i = grad_output * factor_i.
    auto x_p = x.primal();
    auto primal = tenzor::prod(x_p, dim, keepdim);

    auto input_shape_vec = std::vector<int64_t>(x_p.shape().begin(), x_p.shape().end());
    auto zero_in = tenzor::zeros(input_shape_vec, x_p.dtype(), x_p.device());
    auto ones_in = tenzor::ones(input_shape_vec, x_p.dtype(), x_p.device());
    auto mask_zero = tenzor::eq(x_p, zero_in);
    auto safe_input = tenzor::where(mask_zero, ones_in, x_p);

    // Always reduce the intermediates with keepdim=true (same rank as the
    // input regardless of dim/full-reduction), so expand() is always a
    // simple same-rank broadcast — sidesteps unsqueeze+expand rank-mismatch
    // edge cases some backends' Expand kernel rejects for the dim=nullopt /
    // 1-D-input cases. Apply the CALLER's requested `keepdim` only at the
    // very end, on the final tangent reduction.
    Tensor prod_safe_kd = tenzor::prod(safe_input, dim, /*keepdim=*/true);
    Tensor prod_safe_expanded = tenzor::expand(prod_safe_kd, input_shape_vec);
    auto factor_raw = tenzor::div(prod_safe_expanded, safe_input);

    auto mask_zero_int = mask_zero.to(DType::Int64);
    Tensor zero_count_kd = tenzor::sum(mask_zero_int, dim, /*keepdim=*/true);
    Tensor zero_count_expanded = tenzor::expand(zero_count_kd, input_shape_vec);

    auto remaining = tenzor::sub(zero_count_expanded, mask_zero_int);
    auto zeros_int = tenzor::zeros(input_shape_vec, DType::Int64, x_p.device());
    auto keep = tenzor::eq(remaining, zeros_int);
    auto factor = tenzor::where(keep, factor_raw, zero_in);

    auto tangent = tenzor::sum(tenzor::mul(factor, x.tangent()), dim, keepdim);
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

namespace {
// H17: exact, zero-safe forward-mode (JVP) cumulative-product tangent.
//   y_k = prod_{j<=k} x_j
//   dy_k = sum_{i<=k} dx_i * prod_{j<=k, j!=i} x_j
// The inner product EXCLUDES the differentiated element x_i, so it never
// divides by x_i — correct for ANY number of zeros in a run, unlike the
// `cumsum(dx/x) * y` closed form this replaces: a single zero anywhere in
// the prefix poisons every subsequent output position via NaN propagation
// through the division and cumsum. Mirrors CumProdBackward's own
// already-fixed exact-prefix-product-excluding-self algorithm
// (function_fft.cpp cumprod_backward_exact_kernel) — same O(n^2)
// correctness-over-performance tradeoff, just gathering into output
// position k instead of scattering into input position i. Unlike the VJP,
// JVP is a directional derivative of a holomorphic function of x, so
// complex inputs are NOT conjugated here.
template <typename T>
void cumprod_jvp_exact_kernel(const T* x, const T* dx, T* dy, int64_t outer, int64_t n) {
    for (int64_t o = 0; o < outer; ++o) {
        const T* xr = x + o * n;
        const T* dxr = dx + o * n;
        T* dyr = dy + o * n;
        for (int64_t k = 0; k < n; ++k) dyr[k] = T(0);
        for (int64_t i = 0; i < n; ++i) {
            T prefix_excl = T(1);
            for (int64_t j = 0; j < i; ++j) prefix_excl *= xr[j];
            T running = prefix_excl;
            for (int64_t k = i; k < n; ++k) {
                if (k > i) running *= xr[k];
                dyr[k] += dxr[i] * running;
            }
        }
    }
}
} // namespace

auto jvp_cumprod(const DualTensor& x, int64_t dim) -> DualTensor {
    auto p = x.primal();
    auto primal = tenzor::cumprod(p, dim);

    const Device orig_device = p.device();
    const DType orig_dtype = p.dtype();
    const int64_t ndim = static_cast<int64_t>(p.shape().size());
    if (ndim == 0) {
        // Scalar: cumprod is the identity, tangent passes through unchanged.
        return DualTensor(std::move(primal), x.tangent());
    }
    int64_t d = dim;
    if (d < 0) d += ndim;

    Tensor xt = p;
    Tensor dxt = x.tangent();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    const DType work_dtype = widen ? DType::Float32 : orig_dtype;
    if (orig_device.type != Device::Type::CPU) {
        xt = xt.to(Device::cpu());
        dxt = dxt.to(Device::cpu());
    }
    if (xt.dtype() != work_dtype) xt = xt.to(work_dtype);
    if (dxt.dtype() != work_dtype) dxt = dxt.to(work_dtype);

    if (d != ndim - 1) {
        xt = tenzor::movedim(xt, {d}, {ndim - 1});
        dxt = tenzor::movedim(dxt, {d}, {ndim - 1});
    }
    xt = xt.contiguous();
    dxt = dxt.contiguous();

    const int64_t n = xt.shape().back();
    const int64_t total = xt.numel();
    const int64_t outer = (n == 0) ? 0 : total / n;

    Tensor dy = tenzor::empty(std::vector<int64_t>(xt.shape().begin(), xt.shape().end()),
                              work_dtype, Device::cpu());

    if (total > 0 && n > 0) {
        if (work_dtype == DType::Float64) {
            cumprod_jvp_exact_kernel<double>(xt.data<double>(), dxt.data<double>(),
                                             dy.data<double>(), outer, n);
        } else if (work_dtype == DType::Float32) {
            cumprod_jvp_exact_kernel<float>(xt.data<float>(), dxt.data<float>(),
                                            dy.data<float>(), outer, n);
        } else if (work_dtype == DType::Complex64) {
            cumprod_jvp_exact_kernel<std::complex<float>>(
                xt.data<std::complex<float>>(), dxt.data<std::complex<float>>(),
                dy.data<std::complex<float>>(), outer, n);
        } else if (work_dtype == DType::Complex128) {
            cumprod_jvp_exact_kernel<std::complex<double>>(
                xt.data<std::complex<double>>(), dxt.data<std::complex<double>>(),
                dy.data<std::complex<double>>(), outer, n);
        } else {
            throw std::runtime_error("jvp_cumprod: unsupported dtype");
        }
    }

    if (d != ndim - 1) {
        dy = tenzor::movedim(dy, {ndim - 1}, {d});
    }
    if (widen) {
        dy = dy.to(orig_dtype);
    }
    if (orig_device.type != Device::Type::CPU) {
        dy = dy.to(orig_device);
    }
    dy = dy.contiguous();

    return DualTensor(std::move(primal), std::move(dy));
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

// (sign, logabsdet) = slogdet(A). d(logabsdet) = trace(A^{-1} @ dA) — the
// det(A)-scale-free sibling of jvp_det's dy = det(A) * trace(A^{-1} @ dA)
// just above: SlogdetBackward::backward() (function_linalg.cpp) exploits
// exactly this simplification (dividing the det(A) factor back out of
// DetBackward's formula), and this reuses tenzor::linalg::inv(A) the same
// way that backward() (and jvp_det above) already do. sign is piecewise-
// constant almost everywhere; its tangent is structurally zero (the
// measure-zero non-differentiable points where sign flips are not
// specially handled, matching backward()'s own treatment).
auto jvp_slogdet(const DualTensor& a) -> std::pair<DualTensor, DualTensor> {
    auto [sign, logabsdet] = tenzor::linalg::slogdet(a.primal());
    auto Ainv = tenzor::linalg::inv(a.primal());
    auto tangent_logabsdet = tenzor::trace(tenzor::matmul(Ainv, a.tangent()));
    auto tangent_sign = tenzor::zeros_like(sign);
    return {DualTensor(std::move(sign), std::move(tangent_sign)),
            DualTensor(std::move(logabsdet), std::move(tangent_logabsdet))};
}

// y = ||A||_F (Frobenius norm of the WHOLE tensor flattened to a scalar,
// batch-agnostic — see linalg::norm() in ops/linalg.cpp; the string-ord
// "fro" case only, per NormBackward_Linalg's class comment in
// function.hpp). dy = <A, dA> / ||A||_F = sum(A * dA) / y. Mirrors
// NormBackward_Linalg::backward()'s "fro" branch exactly
// (function_linalg.cpp), including its norm==0 zero-subgradient
// convention (matches PyTorch's zero-at-origin convention rather than the
// unguarded Inf/NaN a raw /0 would produce).
auto jvp_norm_fro(const DualTensor& a) -> DualTensor {
    auto y = tenzor::linalg::norm(a.primal(), "fro");
    auto inner = tenzor::sum(tenzor::mul(a.primal(), a.tangent()));
    auto y_shape = std::vector<int64_t>(y.shape().begin(), y.shape().end());
    auto eps = tenzor::full(y_shape, detail::dtype_epsilon(y.dtype()), y.dtype(), y.device());
    auto safe_y = tenzor::where(tenzor::eq(y, tenzor::zeros_like(y)), eps, y);
    auto tangent = tenzor::div(inner, safe_y);
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

auto jvp_masked_fill(const DualTensor& x, const Tensor& mask, double value) -> DualTensor {
    // masked_fill replaces tangent positions with 0 (constant value has zero derivative).
    // `value` is `double` so Float64 callers don't lose precision before the public
    // `tenzor::masked_fill` overload widens it again (audit-5 Y.2 + Y.3).
    auto primal = tenzor::masked_fill(x.primal(), mask, value);
    auto tangent = tenzor::masked_fill(x.tangent(), mask, 0.0);
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

auto jvp_norm(const DualTensor& x, double p, std::optional<int64_t> dim, bool keepdim) -> DualTensor {
    // y = (sum |x|^p)^(1/p)
    // dy = (sum sign(x) * |x|^(p-1) * dx) * y^(1-p)
    //    where for p=2 this is (sum x*dx) / y, and for p=1 it is sum sign(x)*dx.
    // We build the keepdim form of the primal so the y^(1-p) broadcast is well-defined,
    // then squeeze if the caller did not request keepdim.
    // The public `tenzor::norm` API takes `float p` today; we keep the helper's
    // contract as `double` (audit-5 Y.1) so users on Float64 don't lose precision
    // in the `|x|^(p-1)` path which is computed at double width below.
    auto abs_x = tenzor::abs(x.primal());
    auto sgn   = tenzor::sign(x.primal());
    // Compute the PRIMAL norm via the same closed-form composition the tangent
    // uses (abs -> pow(p) -> sum -> pow(1/p)), so the exponent stays full-double
    // p. Routing the primal through `tenzor::norm(static_cast<float>(p), ...)`
    // would float-round the exponent (e.g. p=2.1) while the tangent terms below
    // use the exact double p, yielding inconsistent halves of the DualTensor
    // on Float64. `tenzor::pow` takes a double exponent, matching lines below.
    auto y_kd = tenzor::pow(
        tenzor::sum(tenzor::pow(abs_x, p), dim, /*keepdim=*/true),
        1.0 / p);
    auto pow_abs = tenzor::pow(abs_x, p - 1.0);
    // Guard against the p<1 singularity at x=0: |x|^(p-1) is +inf there, and
    // sign(0)=0 makes the product NaN. The contribution at exactly x=0 should
    // be zero (sign is zero, the subgradient is the convex-set [-1,1] which
    // averages to zero), so explicitly zero the weighted term where abs_x==0.
    auto zero_like_abs = tenzor::zeros_like(abs_x);
    auto zero_mask = tenzor::eq(abs_x, zero_like_abs);
    auto safe_pow_abs = tenzor::where(zero_mask, zero_like_abs, pow_abs);
    auto weighted = tenzor::mul(tenzor::mul(sgn, safe_pow_abs), x.tangent());
    auto num = tenzor::sum(weighted, dim, /*keepdim=*/true);
    auto scale = tenzor::pow(y_kd, 1.0 - p);
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
                    double value) -> DualTensor {
    // y = input; y[index[i]] = value (constant).  Derivative wrt input is the
    // identity on un-indexed positions and zero at indexed positions; we get
    // that by filling the input tangent at the same indices with 0.
    // `value` is `double` so Float64 callers don't lose precision before the
    // public `tenzor::index_fill` overload widens it again (audit-5 Y.2 + Y.3).
    auto primal = tenzor::index_fill(input.primal(), dim, index, value);
    auto tangent = tenzor::index_fill(input.tangent(), dim, index, /*value=*/0.0);
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
    //
    // CRITICAL: every numerator term for a given output position t shares the
    // *same* per-output denominator sum_cum[t] = exp(l[t]).  Dividing the whole
    // cumulative sum by exp(l[t]) (a per-output constant) is correct; dividing
    // each term k by its own exp(l[k]) is NOT — that is the bug we are fixing.
    //
    // For numerical stability we subtract a per-output constant m[t] (the
    // running max along `dim`) from x before exponentiating.  m cancels exactly:
    //   dl[t] = cumsum(exp(x - m) * dx, dim)[t] / cumsum(exp(x - m), dim)[t]
    // because cumsum(exp(x - m), dim)[t] = exp(l[t] - m[t]) and the m[t] factor
    // appears identically in numerator and denominator.  Using the cumulative
    // max (rather than a global reduction) keeps every entry of (x - m) <= 0.
    auto primal = tenzor::dispatch(OpId::Logcumsumexp,
                                   std::vector<Tensor>{x.primal()},
                                   [&]() {
                                       OpAttributes a;
                                       a.set(AttrKey::Dim, dim);
                                       return a;
                                   }())[0];
    // Per-output cumulative-max shift for stability.  m has the same shape as x
    // (cummax does not reduce the axis), so exp(x - m) keeps each term <= 1.
    auto m = tenzor::cummax(x.primal(), dim).first;
    auto w = tenzor::exp(tenzor::sub(x.primal(), m));            // exp(x - m), <= 1
    auto num = tenzor::cumsum(tenzor::mul(w, x.tangent()), dim);  // sum_{k<=t} w[k] dx[k]
    auto den = tenzor::cumsum(w, dim);                            // = exp(l[t] - m[t])
    auto tangent = tenzor::div(num, den);
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
                      double scale,
                      bool causal) -> DualTensor {
    // Widen Float16/BFloat16 to Float32 for the softmax reduction chain
    // (catastrophic cancellation), mirroring the reverse-mode
    // FlashAttentionBackward/NestedAttentionBackward's own widen-then-narrow.
    const DType orig_dtype = Q.primal().dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Tensor Qp = widen ? Q.primal().to(DType::Float32) : Q.primal();
    Tensor Qt = widen ? Q.tangent().to(DType::Float32) : Q.tangent();
    Tensor Kp = widen ? K.primal().to(DType::Float32) : K.primal();
    Tensor Kt = widen ? K.tangent().to(DType::Float32) : K.tangent();
    Tensor Vp = widen ? V.primal().to(DType::Float32) : V.primal();
    Tensor Vt = widen ? V.tangent().to(DType::Float32) : V.tangent();

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
    if (widen) {
        y   = y.to(orig_dtype);
        y_t = y_t.to(orig_dtype);
    }
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
    // Widen Float16/BFloat16 to Float32 for the softmax/logsumexp reduction
    // chain (catastrophic cancellation), mirroring jvp_softmax/jvp_log_softmax.
    const DType orig_dtype = logits.primal().dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Tensor X  = widen ? logits.primal().to(DType::Float32) : logits.primal();
    Tensor dX = widen ? logits.tangent().to(DType::Float32) : logits.tangent();

    // Stable softmax across the last dim (class dim). Mirrors the primal op's
    // rank>2 contract (fused_ops.cpp:213-293): logits is (D1,...,Dk,C) and
    // targets is (D1,...,Dk) — the class dim is ALWAYS the last logits dim,
    // for both the legacy rank-2 (N,C) case (k=1) and the seq2seq-style
    // rank>2 case. Gather accordingly along the last dim instead of assuming
    // dim=1 / a fixed (B,C) shape.
    int64_t last_dim = -1;
    auto x_max  = tenzor::max(X, last_dim, /*keepdim=*/true);
    auto shift  = tenzor::sub(X, x_max);
    auto exp_s  = tenzor::exp(shift);
    auto sum_e  = tenzor::sum(exp_s, last_dim, /*keepdim=*/true);
    auto P      = tenzor::div(exp_s, sum_e);                // softmax
    auto lse    = tenzor::add(tenzor::log(sum_e), x_max);   // (D1,...,Dk,1)

    // Per-sample primal loss: logsumexp - gather(logits, target) along the
    // last (class) dim. Reshape targets (D1,...,Dk) to (D1,...,Dk,1) — the
    // keepdim-style index shape gather() needs — instead of the old
    // rank-2-only {B,1} reshape (which threw for rank>2 targets since
    // targets.numel() == D1*...*Dk doesn't fit {B,1} unless k==1).
    int64_t logit_ndim = X.ndim();
    std::vector<int64_t> tgt_shape(targets.shape().begin(), targets.shape().end());
    tgt_shape.push_back(1);
    Tensor tgt_nd = tenzor::reshape(targets, tgt_shape);
    auto x_at_tgt = tenzor::gather(X, /*dim=*/logit_ndim - 1, tgt_nd);   // (D1,...,Dk,1)
    auto per_loss = tenzor::sub(lse, x_at_tgt);             // (D1,...,Dk,1)
    std::vector<int64_t> sample_shape(targets.shape().begin(), targets.shape().end());
    per_loss = tenzor::reshape(per_loss, sample_shape);

    // Per-sample tangent.
    // sum_j p[j] * dx[j] over class dim → (D1,...,Dk,1); gather(dX, tgt)
    // along the same last dim → (D1,...,Dk,1).
    auto P_dx = tenzor::mul(P, dX);
    auto sum_P_dx = tenzor::sum(P_dx, last_dim, /*keepdim=*/true);  // (D1,...,Dk,1)
    auto dx_at_tgt = tenzor::gather(dX, /*dim=*/logit_ndim - 1, tgt_nd);  // (D1,...,Dk,1)
    auto per_tan = tenzor::sub(sum_P_dx, dx_at_tgt);                  // (D1,...,Dk,1)
    per_tan = tenzor::reshape(per_tan, sample_shape);

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
    if (widen) {
        primal_out  = primal_out.to(orig_dtype);
        tangent_out = tangent_out.to(orig_dtype);
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
    // V.6: keep the scalar at double end-to-end (the underlying `jvp_leaky_relu`
    // now takes `double`; previously the static_cast<float> narrowed the value
    // read from the OpAttributes' double-typed slot).
    double slope = attrs.get_float(AttrKey::Negative_slope, 0.01);
    return dual_to_result(jvp_leaky_relu(x, slope));
}

JvpResult jvp_adapter_elu(std::span<const Tensor> primals,
                          std::span<const Tensor> tangents,
                          const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_elu: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    // V.6: double end-to-end (see jvp_adapter_leaky_relu for rationale).
    double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
    return dual_to_result(jvp_elu(x, alpha));
}

JvpResult jvp_adapter_softplus(std::span<const Tensor> primals,
                               std::span<const Tensor> tangents,
                               const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_softplus: expected 1 input");
    }
    auto x = make_dual(primals[0], tangents[0]);
    // V.6: double end-to-end (see jvp_adapter_leaky_relu for rationale).
    double beta = attrs.get_float(AttrKey::Beta, 1.0);
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
    double value = attrs.get_float(AttrKey::Value, 0.0);
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

// DeviceTransfer: Tensor::to(Device) is the identity on values — only
// residency changes. There is no dispatch()-backed kernel for this
// registry-only OpId (see its comment in op_id.hpp), so unlike
// linear_unary_jvp() below (which re-dispatches the SAME OpId on the
// tangent via tenzor::dispatch() — wrong here even setting aside the
// missing kernel, since re-dispatching a per-device kernel-table entry
// would run a same-device kernel, not a transfer) this adapter calls
// Tensor::to() directly for both primal and tangent, landing the tangent
// on the same target device as the (recomputed) primal output.
JvpResult jvp_adapter_device_transfer(std::span<const Tensor> primals,
                                      std::span<const Tensor> tangents,
                                      const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_device_transfer: expected 1 input");
    }
    Device target{
        static_cast<Device::Type>(attrs.get_int(AttrKey::DeviceType,
                                                  static_cast<int64_t>(Device::Type::CPU))),
        static_cast<int32_t>(attrs.get_int(AttrKey::DeviceIndex, 0))
    };
    auto primal_out  = primals[0].to(target);
    auto tangent_out = tangents[0].to(target);
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
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

// LinalgSlogdet: forward is (sign, logabsdet) = slogdet(A), a 2-output op
// (see OpId::LinalgSlogdet's comment in op_id.hpp for why this is NOT
// OpId::LinalgDet). Output-slot order: {d_logabsdet, d_sign} — see the
// long comment on SlogdetBackward::op_id() in function.hpp for why
// logabsdet (not sign) must be index 0: SlogdetBackward's saved_tensors_
// holds only A^{-1} (not the forward outputs), so the JVP walker's
// data_ptr root-matching can never resolve an output slot for this node
// and always falls through to index 0 of this result — which in practice
// is the only output ever queried, since slogdet(Variable) (ops.cpp)
// gives `sign` no grad_fn at all.
JvpMultiResult jvp_adapter_slogdet(std::span<const Tensor> primals,
                                   std::span<const Tensor> tangents,
                                   const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_slogdet: expected 1 input (A)");
    }
    auto a = make_dual(primals[0], tangents[0]);
    auto [sign_dual, logabsdet_dual] = jvp_slogdet(a);
    JvpMultiResult result;
    result.primals  = { logabsdet_dual.primal(),  sign_dual.primal()  };
    result.tangents = { logabsdet_dual.tangent(), sign_dual.tangent() };
    return result;
}

// LinalgNorm, "fro" case only: see NormBackward_Linalg's class comment in
// function.hpp for why the other 7 string-ord values ('nuc','1','-1','2',
// '-2','inf','-inf') stay on the finite-difference fallback. Reading
// AttrKey::NormOrd and throwing for anything but "fro" is what makes that
// fallback happen correctly (caught by the JVP walker in functional.cpp)
// instead of silently reusing the "fro" formula for an unsupported ord.
JvpResult jvp_adapter_linalg_norm_fro(std::span<const Tensor> primals,
                                      std::span<const Tensor> tangents,
                                      const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_linalg_norm_fro: expected 1 input (A)");
    }
    if (attrs.get_string(AttrKey::NormOrd, "fro") != "fro") {
        throw std::runtime_error(
            "jvp_adapter_linalg_norm_fro: only ord=='fro' is supported by this JVP rule");
    }
    auto a = make_dual(primals[0], tangents[0]);
    return dual_to_result(jvp_norm_fro(a));
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
    double p = attrs.get_float(AttrKey::P, 2.0);
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
    double value = attrs.get_float(AttrKey::Value, 0.0);
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
    // Widen Float16/BFloat16 to Float32 for the reduction-heavy mean/variance
    // chain (catastrophic cancellation in (x-mean)^2), mirroring
    // NestedLayerNormBackward's own widen-then-narrow and the reverse-mode
    // LayerNormBackward kernels.
    const DType orig_dtype = primals[0].dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Tensor x     = widen ? primals[0].to(DType::Float32) : primals[0];
    Tensor gamma = widen ? primals[1].to(DType::Float32) : primals[1];
    Tensor beta  = widen ? primals[2].to(DType::Float32) : primals[2];

    auto zeros_like_or = [](const Tensor& t, const Tensor& tan) -> Tensor {
        if (tan.numel() != 0) return tan;
        auto sh = std::vector<int64_t>(t.shape().begin(), t.shape().end());
        return tenzor::zeros(sh, t.dtype(), t.device());
    };
    Tensor dx     = zeros_like_or(primals[0], tangents[0]);
    Tensor dgamma = zeros_like_or(primals[1], tangents[1]);
    Tensor dbeta  = zeros_like_or(primals[2], tangents[2]);
    if (widen) {
        dx     = dx.to(DType::Float32);
        dgamma = dgamma.to(DType::Float32);
        dbeta  = dbeta.to(DType::Float32);
    }

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

    // H16: the real kernel (nn_kernels.cpp layer_norm_kernel_with_stats)
    // allocates mean/rstd as a FLAT {batch_size} tensor (batch_size =
    // numel(x)/norm_size, collapsed across ALL leading dims) — not x's shape
    // with the last K dims collapsed to 1 and kept. mean/rstd/dmean/drstd
    // above are keepdim=true (shape x with trailing K dims = 1), which has
    // the same total element count as {batch_size} — reshape down to match
    // the real contract.
    int64_t norm_size = 1;
    for (int64_t i = 0; i < K; ++i) {
        norm_size *= x.shape()[xnd - 1 - i];
    }
    int64_t batch_size = x.numel() / norm_size;
    mean  = tenzor::reshape(mean,  {batch_size});
    rstd  = tenzor::reshape(rstd,  {batch_size});
    dmean = tenzor::reshape(dmean, {batch_size});
    drstd = tenzor::reshape(drstd, {batch_size});

    if (widen) {
        y     = y.to(orig_dtype);
        dy    = dy.to(orig_dtype);
        mean  = mean.to(orig_dtype);
        rstd  = rstd.to(orig_dtype);
        dmean = dmean.to(orig_dtype);
        drstd = drstd.to(orig_dtype);
    }

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

    // M27: Lorentzian-broadened reciprocal of the eigenvalue gap (mirrors
    // the already-fixed EighBackward VJP, function_linalg.cpp): F_ij =
    // diff_ij / (diff_ij^2 + eps^2), diff_ij = w_j - w_i. Exact
    // 1/(w_j - w_i) for well-separated eigenvalues, BOUNDED by 1/(2 eps) at
    // (near-)degenerate roots, and NEVER hard-zeroed — unlike the previous
    // fixed-threshold mask this replaces, which clamped any pair under an
    // exact-equality test to 0 and thus dropped the eigenvector-rotation
    // tangent for closely-but-legitimately-separated spectra (e.g. w1=1.0,
    // w2=1.0+1e-7, common in near-repeated eigenvalues of covariance/PCA
    // matrices). The diagonal (diff == 0) yields F == 0 automatically (zero
    // numerator), so no explicit diagonal masking is needed.
    auto W_col = tenzor::unsqueeze(W, -1);  // (..., N, 1)  varies over i
    auto W_row = tenzor::unsqueeze(W, -2);  // (..., 1, N)  varies over j
    auto diffs = tenzor::sub(W_row, W_col); // diffs[..., i, j] = w_j - w_i
    double rel_eps;
    switch (W.dtype()) {
        case DType::Float16:
        case DType::BFloat16: rel_eps = 1e-2; break;
        case DType::Float64:  rel_eps = 1e-12; break;
        default:               rel_eps = 1e-6; break;  // Float32
    }
    auto max_scale = tenzor::max(tenzor::abs(W), W.ndim() - 1, /*keepdim=*/true);  // (..., 1)
    max_scale = tenzor::unsqueeze(max_scale, max_scale.ndim());  // (..., 1, 1)
    auto eps_tol = tenzor::add(tenzor::mul(max_scale, rel_eps), rel_eps);
    auto eps_sq  = tenzor::mul(eps_tol, eps_tol);
    auto denom   = tenzor::add(tenzor::mul(diffs, diffs), eps_sq);
    auto F       = tenzor::div(diffs, denom);

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
    double scale = attrs.get_float(AttrKey::Scale, 1.0);
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
    double scale = attrs.get_float(AttrKey::Scale, 1.0);
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
    double scale = attrs.get_float(AttrKey::Scale, 1.0);
    // FlexAttention is non-causal by default; causal is encoded via score-mod
    // or block_mask, both of which we've already refused above.
    auto Q = make_dual(primals[0], tangents[0]);
    auto K = make_dual(primals[1], tangents[1]);
    auto V = make_dual(primals[2], tangents[2]);
    return dual_to_result(jvp_sdpa_forward(Q, K, V, scale, /*causal=*/false));
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
            // Full-reduction non-keepdim returns a 0-dim scalar (shape {}),
            // matching the Median kernel's primal output; match it exactly
            // rather than hardcoding {1} so primal/tangent shapes agree.
            values_t = tenzor::reshape(
                values_t,
                std::vector<int64_t>(values_p.shape().begin(),
                                     values_p.shape().end()));
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
        // dy = dvalues - sum_per_segment(P * dvalues) — weight dvalues by P
        // FIRST, then sum (matches jvp_log_softmax's dense sibling
        // `sum_s_dt=sum(s*dt); tangent=dt-sum_s_dt` and the correct
        // jvp_adapter_nested_softmax above, which weights before summing).
        // The previous `dvalues - P*NestedSum(dvalues)` summed the
        // UNWEIGHTED dvalues, then applied P afterward — a different (wrong)
        // computation whenever P varies across positions within a segment.
        Tensor P = tenzor::exp(logP);
        OpAttributes rattrs;
        rattrs.set(AttrKey::Dim, dim);
        rattrs.set(AttrKey::Keepdim, true);
        Tensor P_dv = tenzor::mul(P, dvalues);
        Tensor seg_sum = tenzor::dispatch(OpId::NestedSum,
            std::vector<Tensor>{P_dv, offsets}, rattrs)[0];
        tangent = tenzor::sub(dvalues, seg_sum);
    }
    return JvpResult{std::move(logP), std::move(tangent)};
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

// view_as_real/view_as_complex are pure metadata reinterpretation (zero
// dispatch() calls — see ViewAsRealBackward/ViewAsComplexBackward's class
// comments in function.hpp) and R-linear, so the tangent gets the
// identical reinterpretation. Unlike Conj/Real/Imag just above, these
// can't go through linear_unary_jvp (which calls tenzor::dispatch(), and
// there is no kernel registered anywhere for these registry-only OpIds —
// see their comment in op_id.hpp) — call the ops-layer functions directly
// instead.
JvpResult jvp_adapter_view_as_real(std::span<const Tensor> primals,
                                   std::span<const Tensor> tangents,
                                   const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_view_as_real: expected 1 input");
    }
    auto primal_out  = tenzor::view_as_real(primals[0]);
    auto tangent_out = tenzor::view_as_real(tangents[0]);
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}
JvpResult jvp_adapter_view_as_complex(std::span<const Tensor> primals,
                                      std::span<const Tensor> tangents,
                                      const OpAttributes&) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_view_as_complex: expected 1 input");
    }
    auto primal_out  = tenzor::view_as_complex(primals[0]);
    auto tangent_out = tenzor::view_as_complex(tangents[0]);
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

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
    const Tensor& input = primals[0];
    const Tensor& grid  = primals[1];

    auto primal = tenzor::dispatch(OpId::GridSample,
                                   std::vector<Tensor>{input, grid}, attrs)[0];

    // Input-side tangent: grid_sample is linear in the input, so its JVP is the
    // same sampling applied to the input tangent.
    Tensor tangent;
    if (tangents[0].numel() != 0) {
        tangent = tenzor::dispatch(OpId::GridSample,
                                   std::vector<Tensor>{tangents[0], grid}, attrs)[0];
    } else {
        tangent = tenzor::zeros_like(primal);
    }

    // Grid-side tangent. The sampled output O(g) varies with the grid; its JVP
    // is sum_d (dO/dg_d) * tangent_grid_d. For bilinear interpolation dO/dfx
    // within a cell equals (column at x0+1) - (column at x0), obtained exactly
    // by re-sampling the input at the floor/ceil integer pixel columns with the
    // SAME interpolated y (and vice-versa for y). Reusing GridSample for the
    // shifted samples means padding mode and align_corners boundary handling are
    // inherited exactly. (Previously this threw NonDifferentiable.)
    if (tangents[1].numel() != 0) {
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        if (mode == "nearest") {
            // Nearest sampling is piecewise-constant in the grid → zero
            // grid-side contribution (a.e.). Nothing to add.
        } else if (mode == "bilinear" && input.ndim() == 4) {
            const bool align = attrs.get_bool(AttrKey::AlignCorners, false);
            const int64_t N = input.shape()[0];
            const int64_t H = input.shape()[2];
            const int64_t W = input.shape()[3];
            const int64_t Hg = grid.shape()[1];
            const int64_t Wg = grid.shape()[2];

            auto denorm = [&](const Tensor& g, int64_t size) -> Tensor {
                // align:    (g+1)*0.5*(size-1)
                // !align: ((g+1)*size - 1)*0.5
                return align
                    ? tenzor::mul(tenzor::add(g, 1.0), 0.5 * static_cast<double>(size - 1))
                    : tenzor::mul(tenzor::sub(tenzor::mul(tenzor::add(g, 1.0),
                                                          static_cast<double>(size)), 1.0), 0.5);
            };
            auto renorm = [&](const Tensor& ix, int64_t size) -> Tensor {
                // inverse of denorm
                return align
                    ? tenzor::sub(tenzor::mul(ix, 2.0 / static_cast<double>(size - 1)), 1.0)
                    : tenzor::sub(tenzor::mul(tenzor::add(tenzor::mul(ix, 2.0), 1.0),
                                              1.0 / static_cast<double>(size)), 1.0);
            };
            const double dscale_x = align ? 0.5 * static_cast<double>(W - 1)
                                          : 0.5 * static_cast<double>(W);
            const double dscale_y = align ? 0.5 * static_cast<double>(H - 1)
                                          : 0.5 * static_cast<double>(H);

            auto gx = tenzor::narrow(grid, 3, 0, 1);  // [N,Hg,Wg,1]
            auto gy = tenzor::narrow(grid, 3, 1, 1);

            // dO/dgx via integer-x columns (y kept at its interpolated value).
            auto ix0 = tenzor::floor(denorm(gx, W));
            auto gx0 = renorm(ix0, W);
            auto gx1 = renorm(tenzor::add(ix0, 1.0), W);
            auto col_l = tenzor::dispatch(OpId::GridSample,
                std::vector<Tensor>{input, tenzor::cat(std::vector<Tensor>{gx0, gy}, 3)}, attrs)[0];
            auto col_r = tenzor::dispatch(OpId::GridSample,
                std::vector<Tensor>{input, tenzor::cat(std::vector<Tensor>{gx1, gy}, 3)}, attrs)[0];
            auto dOdgx = tenzor::mul(tenzor::sub(col_r, col_l), dscale_x);  // [N,C,Hg,Wg]
            auto tgx = tenzor::reshape(tenzor::narrow(tangents[1], 3, 0, 1), {N, 1, Hg, Wg});
            auto contrib_x = tenzor::mul(dOdgx, tgx);  // broadcast over channels

            // dO/dgy via integer-y rows (x kept at its interpolated value).
            auto iy0 = tenzor::floor(denorm(gy, H));
            auto gy0 = renorm(iy0, H);
            auto gy1 = renorm(tenzor::add(iy0, 1.0), H);
            auto col_d = tenzor::dispatch(OpId::GridSample,
                std::vector<Tensor>{input, tenzor::cat(std::vector<Tensor>{gx, gy0}, 3)}, attrs)[0];
            auto col_u = tenzor::dispatch(OpId::GridSample,
                std::vector<Tensor>{input, tenzor::cat(std::vector<Tensor>{gx, gy1}, 3)}, attrs)[0];
            auto dOdgy = tenzor::mul(tenzor::sub(col_u, col_d), dscale_y);
            auto tgy = tenzor::reshape(tenzor::narrow(tangents[1], 3, 1, 1), {N, 1, Hg, Wg});
            auto contrib_y = tenzor::mul(dOdgy, tgy);

            tangent = tenzor::add(tangent, tenzor::add(contrib_x, contrib_y));
        } else {
            throw NonDifferentiable(
                "GridSample forward-mode JVP w.r.t. grid is implemented for 4D "
                "'bilinear' and 'nearest' modes; got mode '" + mode + "' with " +
                std::to_string(input.ndim()) + "D input.");
        }
    }

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
//   dy[i] = dweight[indices[i]]  → dispatch(Embedding, [dweight, indices]).
//
// H8: the universal OpId::Embedding calling convention (matching every real
// call site — cpu_kernel_registry.cpp:2506-2512 dispatches
// embedding_kernel(inputs[0]=weight, inputs[1]=indices);
// nn/layers/embedding.cpp and nn/functional.cpp both dispatch
// {weight, indices} in that order) is (weight, indices), NOT
// (indices, weight). This rule previously assumed the opposite order.
// ============================================================================
JvpResult jvp_adapter_embedding(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 2 || primals.size() != tangents.size()) {
        throw std::runtime_error(
            "jvp_adapter_embedding: expected 2 inputs (weight, indices)");
    }
    auto dweight = tangents[0].numel() != 0 ? tangents[0]
        : tenzor::zeros(std::vector<int64_t>(primals[0].shape().begin(),
                                              primals[0].shape().end()),
                        primals[0].dtype(), primals[0].device());
    auto primal  = tenzor::dispatch(OpId::Embedding,
                                    std::vector<Tensor>{primals[0], primals[1]}, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::Embedding,
                                    std::vector<Tensor>{dweight, primals[1]}, attrs)[0];
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
auto jvp_logit(const DualTensor& x, double eps) -> DualTensor {
    // Pass the real eps through so the primal matches the eager forward
    // tenzor::logit, which clamps x to [eps, 1-eps] only when eps > 0.  (J-02)
    auto primal = tenzor::logit(x.primal(), eps);
    auto one = tenzor::ones_like(x.primal());
    if (eps > 0.0) {
        // Denominator uses the clamped x (inside [eps,1-eps] the clamp is a
        // no-op, so this equals x(1-x) there); zero the tangent where the
        // ORIGINAL x is outside [eps,1-eps], because the eager forward pins the
        // primal to a constant there (slope 0).  We branch on eps > 0 (not the
        // spec's eps >= 0) to stay consistent with the primal above: at eps==0
        // the eager forward does not clamp, so the tangent must be 1/(x(1-x))
        // over all x, which the eps<=0 path below provides.
        auto xc = tenzor::clamp(x.primal(), eps, 1.0 - eps);
        auto denom = tenzor::mul(xc, tenzor::sub(one, xc));
        auto raw = tenzor::div(x.tangent(), denom);
        auto lo = tenzor::mul(tenzor::ones_like(x.primal()), eps);
        auto hi = tenzor::mul(tenzor::ones_like(x.primal()), 1.0 - eps);
        auto outside = tenzor::logical_or(tenzor::lt(x.primal(), lo),
                                          tenzor::gt(x.primal(), hi));
        auto zero = tenzor::zeros_like(raw);
        auto tangent = tenzor::where(outside, zero, raw);
        return DualTensor(std::move(primal), std::move(tangent));
    }
    // eps <= 0 (unset): open-interval derivative 1/(x(1-x)) over all x.
    auto denom = tenzor::mul(x.primal(), tenzor::sub(one, x.primal()));
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
//   J1'(x) = J0(x) − J1(x)/x
//   I0'(x) = I1(x)
//   I1'(x) = I0(x) − I1(x)/x       (analogous identity for modified Bessel).
// H24: the x^{-1} term must be SIGN-PRESERVING (x_safe = x + eps, matching
// jvp_spherical_bessel_j0/jvp_sinc's convention below) -- both J1 and I1 are
// odd functions of x, so dividing by abs(x) instead computes primal/|x| =
// -primal/x for x<0: the negation of the correct term, a hard sign flip
// across the entire negative domain, not merely a numerical-stability
// nicety. At the removable singularity x=0 itself, J1(0)=I1(0)=0 so the
// eps-nudge still yields the correct limit (0/eps -> 0) either way.
auto jvp_bessel_j0(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::bessel_j0(x.primal());
    auto deriv  = tenzor::neg(tenzor::bessel_j1(x.primal()));
    return DualTensor(std::move(primal), tenzor::mul(x.tangent(), deriv));
}
auto jvp_bessel_j1(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::bessel_j1(x.primal());
    auto j0     = tenzor::bessel_j0(x.primal());
    auto x_safe = tenzor::add(x.primal(), 1e-30);
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
    auto x_safe = tenzor::add(x.primal(), 1e-30);
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
// H24: x_safe must be sign-preserving (see jvp_bessel_j1/jvp_bessel_i1
// above) -- i1e is odd, so abs(x) flips the sign of this term for x<0.
auto jvp_i1e(const DualTensor& x) -> DualTensor {
    auto primal = tenzor::i1e(x.primal());
    auto i0e_   = tenzor::i0e(x.primal());
    auto sgn    = tenzor::sign(x.primal());
    auto x_safe = tenzor::add(x.primal(), 1e-30);
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
    // M7 (AUTOGRAD-R022): zero-safe denominator, mirroring AddcdivBackward's
    // VJP guard — addcdiv(a,b,c,value) has the same b/c and b/c^2 terms as
    // Div's a/b and a/b^2, so it needs DivBackward's same epsilon-substitution
    // guard. Only c_sq (the denominator) needs substitution; the numerator's
    // c*db term is a multiplication, not a division, so the TRUE c==0 there
    // is already safe (no NaN/Inf).
    auto c_p = c.primal();
    auto c_shape = std::vector<int64_t>(c_p.shape().begin(), c_p.shape().end());
    auto zero_c = tenzor::zeros(c_shape, c_p.dtype(), c_p.device());
    auto eps_c = tenzor::full(c_shape, detail::dtype_epsilon(c_p.dtype()), c_p.dtype(), c_p.device());
    auto safe_c = tenzor::where(tenzor::eq(c_p, zero_c), eps_c, c_p);
    auto c_sq = tenzor::mul(safe_c, safe_c);
    auto num  = tenzor::sub(tenzor::mul(b.tangent(), c_p),
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
// Fmax / Fmin: NaN-IGNORING per IEEE 754-2008 maxNum/minNum (confirmed at
// src/backends/cpu/kernels/advanced.cpp:1133-1161) — the opposite of
// maximum/minimum, which propagate NaN. H7: the primal must come from
// tenzor::fmax/fmin, not tenzor::maximum/tenzor::minimum, and the tangent
// weighting must special-case "exactly one operand is NaN" (the real op
// returns — and the tangent must come entirely from — the OTHER, non-NaN
// operand; gt/eq on a NaN operand are both false, so the plain
// maximum/minimum weighting silently gives that case weight 0).
auto jvp_fmax(const DualTensor& a, const DualTensor& b) -> DualTensor {
    auto primal = tenzor::fmax(a.primal(), b.primal());
    auto a_nan = tenzor::isnan(a.primal());
    auto b_nan = tenzor::isnan(b.primal());
    auto a_active = tenzor::gt(a.primal(), b.primal());
    auto tie = tenzor::eq(a.primal(), b.primal());
    auto ones = tenzor::ones_like(a.primal());
    auto zeros = tenzor::zeros_like(a.primal());
    auto half = tenzor::mul(ones, 0.5);
    auto wa_no_nan = tenzor::where(a_active, ones, tenzor::where(tie, half, zeros));
    // b NaN (a finite) -> wa=1; a NaN (b finite) -> wa=0; both NaN -> result
    // is NaN regardless, split 0.5/0.5.
    auto wa = tenzor::where(b_nan,
                             tenzor::where(a_nan, half, ones),
                             tenzor::where(a_nan, zeros, wa_no_nan));
    auto wb = tenzor::sub(ones, wa);
    auto tangent = tenzor::add(tenzor::mul(a.tangent(), wa),
                                 tenzor::mul(b.tangent(), wb));
    return DualTensor(std::move(primal), std::move(tangent));
}
auto jvp_fmin(const DualTensor& a, const DualTensor& b) -> DualTensor {
    auto primal = tenzor::fmin(a.primal(), b.primal());
    auto a_nan = tenzor::isnan(a.primal());
    auto b_nan = tenzor::isnan(b.primal());
    auto a_active = tenzor::lt(a.primal(), b.primal());
    auto tie = tenzor::eq(a.primal(), b.primal());
    auto ones = tenzor::ones_like(a.primal());
    auto zeros = tenzor::zeros_like(a.primal());
    auto half = tenzor::mul(ones, 0.5);
    auto wa_no_nan = tenzor::where(a_active, ones, tenzor::where(tie, half, zeros));
    auto wa = tenzor::where(b_nan,
                             tenzor::where(a_nan, half, ones),
                             tenzor::where(a_nan, zeros, wa_no_nan));
    auto wb = tenzor::sub(ones, wa);
    auto tangent = tenzor::add(tenzor::mul(a.tangent(), wa),
                                 tenzor::mul(b.tangent(), wb));
    return DualTensor(std::move(primal), std::move(tangent));
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
// M6: mask x==0 (xlog1py(0,y)=0 by convention, matching Xlog1pyBackward's
// VJP mask) and epsilon-guard the (1+y)==0 denominator, matching
// Xlog1pyBackward's safe_denom guard.
auto jvp_xlog1py(const DualTensor& x, const DualTensor& y) -> DualTensor {
    auto primal = tenzor::xlog1py(x.primal(), y.primal());
    auto x_p = x.primal();
    auto y_p = y.primal();
    auto x_shape = std::vector<int64_t>(x_p.shape().begin(), x_p.shape().end());
    auto x_is_zero = tenzor::eq(x_p, tenzor::zeros(x_shape, x_p.dtype(), x_p.device()));

    auto log1py = tenzor::log1p(y_p);
    auto t1_raw = tenzor::mul(x.tangent(), log1py);
    auto t1 = tenzor::where(x_is_zero, tenzor::zeros_like(t1_raw), t1_raw);

    auto y_shape = std::vector<int64_t>(y_p.shape().begin(), y_p.shape().end());
    auto one_plus_y = tenzor::add(y_p, 1.0);
    auto eps = tenzor::full(y_shape, detail::dtype_epsilon(y_p.dtype()), y_p.dtype(), y_p.device());
    auto denom_is_zero = tenzor::eq(one_plus_y, tenzor::zeros(y_shape, y_p.dtype(), y_p.device()));
    auto safe_denom = tenzor::where(denom_is_zero, eps, one_plus_y);
    auto t2_raw = tenzor::div(tenzor::mul(x_p, y.tangent()), safe_denom);
    auto t2 = tenzor::where(x_is_zero, tenzor::zeros_like(t2_raw), t2_raw);

    return DualTensor(std::move(primal), tenzor::add(t1, t2));
}

// xlogy(x, y) = x * log(y); d = dx * log(y) + x * dy / y.
// M6: mask x==0 (xlogy(0,y)=0 by convention, matching XLogYBackward's VJP
// mask) and epsilon-guard the y==0 denominator, matching XLogYBackward's
// safe_y guard.
auto jvp_xlogy(const DualTensor& x, const DualTensor& y) -> DualTensor {
    auto primal = tenzor::xlogy(x.primal(), y.primal());
    auto x_p = x.primal();
    auto y_p = y.primal();
    auto x_shape = std::vector<int64_t>(x_p.shape().begin(), x_p.shape().end());
    auto x_is_zero = tenzor::eq(x_p, tenzor::zeros(x_shape, x_p.dtype(), x_p.device()));

    auto logy = tenzor::log(y_p);
    auto t1_raw = tenzor::mul(x.tangent(), logy);
    auto t1 = tenzor::where(x_is_zero, tenzor::zeros_like(t1_raw), t1_raw);

    auto y_shape = std::vector<int64_t>(y_p.shape().begin(), y_p.shape().end());
    auto eps_y = tenzor::full(y_shape, detail::dtype_epsilon(y_p.dtype()), y_p.dtype(), y_p.device());
    auto y_is_zero = tenzor::eq(y_p, tenzor::zeros(y_shape, y_p.dtype(), y_p.device()));
    auto safe_y = tenzor::where(y_is_zero, eps_y, y_p);
    auto t2_raw = tenzor::div(tenzor::mul(x_p, y.tangent()), safe_y);
    auto t2 = tenzor::where(x_is_zero, tenzor::zeros_like(t2_raw), t2_raw);

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
    if (z.primal().is_complex()) {
        // angle(z) = atan2(Im z, Re z); d(angle) = Im(conj(z)·dz)/|z|^2.  (J-01)
        // imag() returns a real tensor, matching the real-valued primal.
        auto num = tenzor::imag(tenzor::mul(tenzor::conj(z.primal()), z.tangent()));
        auto re = tenzor::real(z.primal());
        auto im = tenzor::imag(z.primal());
        auto denom = tenzor::add(tenzor::mul(re, re), tenzor::mul(im, im));
        auto tangent = tenzor::div(num, denom);
        return DualTensor(std::move(primal), std::move(tangent));
    }
    // Real input: angle is a.e. constant (0 for x>=0, π for x<0) → tangent 0.
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
    // Same wrong-attr-key bug as jvp_adapter_addcdiv (discovered while
    // fixing M7): the real dispatcher (ops/math.cpp addcmul()) always sets
    // AttrKey::Alpha, matching every backend kernel.
    double v = attrs.get_float(AttrKey::Alpha, 1.0);
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
    // Discovered while fixing M7: the real dispatcher (ops/math.cpp
    // addcdiv()) always sets AttrKey::Alpha, matching every backend kernel
    // (CPU/CUDA/ROCm/OneAPI/Vulkan). This adapter read AttrKey::Value
    // instead, silently using the wrong scale factor (defaulting to 1.0)
    // whenever alpha != 1.0.
    double v = attrs.get_float(AttrKey::Alpha, 1.0);
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

// Multi-output linalg factorisations where the JVP requires the full saved
// factorisation outputs (Q/R, U/S/V, L/U/P, …) plus skew-symmetric
// derivation. These have well-defined JVPs but each needs a bespoke kernel:
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_lobpcg,
    "LOBPCG returns only the k extreme eigenpairs. The eigenvalue tangent "
    "dλ_i = v_iᵀ dA v_i is well defined, but the eigenvector tangent "
    "dv_i = Σ_j v_j (v_jᵀ dA v_i)/(λ_i−λ_j) sums over the FULL spectrum — the "
    "(k+1)-th eigenpair (absent from the k-subset output) contributes a "
    "dominant term — so a correct dV cannot be formed from the op's output. "
    "For a differentiable symmetric eigendecomposition use LinalgEigh.")

// Sequence-level RNN forwards: implementable via per-step replay of the cell
// rules, but the cell-level forward Functions are the supported entry point
// for JVP. Listed loudly here so callers know to use the cell ops.

// Search/sort multi-output ops that don't expose the saved index-permutation
// in a tangent-friendly layout. (CumMax/CumMin/Aminmax/Kthvalue are already
// registered as multi-output rules in earlier batches.)

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
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_dense_to_sparse,
    "DenseToSparse JVP not implemented: the structural sparsity pattern is "
    "data-dependent (non-zero mask), making the derivative ill-defined at the "
    "zero-boundary.")

// Ops whose former _s15 adapters built a fixed-eps (1e-3) central finite
// difference as the "tangent". A fixed-step FD is not a true JVP — it ignores
// the tangent magnitude, is only O(eps^2)-accurate, and for the piecewise-
// constant quantile/median family returns meaningless boundary values — so it
// fails gradcheck at tight tolerance. Registered NonDifferentiable until an
// exact closed-form JVP is derived (per-segment SDPA/LayerNorm differentials
// for the nested ops; argquantile gather for the quantile family).
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_nested_layer_norm,
    "NestedLayerNorm JVP not implemented: requires a per-segment LayerNorm "
    "differential; a fixed-eps finite-difference probe is not a valid JVP.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_nested_attention_fwd,
    "NestedAttention JVP not implemented: requires a per-segment SDPA "
    "differential; a fixed-eps finite-difference probe is not a valid JVP.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_ctc_loss_forward,
    "CTCLossForward JVP not implemented: the alpha-beta DP recurrence is not a "
    "composition of primitives; a fixed-eps finite-difference probe is not a "
    "valid JVP.")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_quantile,
    "Quantile JVP not implemented: piecewise-constant in x; a fixed-eps "
    "finite-difference probe is not a valid JVP (needs an argquantile gather).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_nanquantile,
    "Nanquantile JVP not implemented: piecewise-constant in x; a fixed-eps "
    "finite-difference probe is not a valid JVP (needs an argquantile gather).")
TENZOR_JVP_NONDIFF(jvp_adapter_nondiff_nanmedian,
    "Nanmedian JVP not implemented: piecewise-constant in x; a fixed-eps "
    "finite-difference probe is not a valid JVP (needs an argquantile gather).")

// GroupNorm / InstanceNorm / RMSNorm: same algebraic shape as LayerNorm but
// over different axes. Until a bespoke multi-output rule lands, mark
// NonDifferentiable rather than reusing LayerNorm with wrong axes.

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
    // d/dt nanmean(x) = sum_{finite}(dx) / finite_count(x). dx at NaN-x
    // positions contribute 0. Re-dispatching Nanmean on the masked tangent is
    // WRONG: the masked tangent has no NaNs, so Nanmean divides by the FULL
    // element count N rather than x's finite count, mis-scaling the tangent by
    // finite_count/N. Instead reduce the masked tangent with plain Sum and
    // divide by x's finite count over the same axes.
    std::optional<int64_t> dim;
    if (attrs.has(AttrKey::Dim)) {
        dim = attrs.get_int(AttrKey::Dim);
    }
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);

    auto is_nan = tenzor::isnan(x.primal());
    auto mask = tenzor::where(is_nan, tenzor::zeros_like(x.primal()),
                              tenzor::ones_like(x.primal()));
    auto dx0 = tenzor::where(is_nan, tenzor::zeros_like(x.tangent()), x.tangent());

    auto numer = tenzor::sum(dx0, dim, keepdim);
    auto nfin  = tenzor::sum(mask, dim, keepdim);   // x's finite count
    auto tangent = tenzor::div(numer, nfin);
    return JvpResult{std::move(primal), std::move(tangent)};
}
// Closed-form NaN-masked variance differential, shared by NanVar/NanStd.
//
// Variance is quadratic in x, so its JVP is the *linear* directional
// derivative (mirroring jvp_var), NaN-masked over the same axes the primal
// reduces and using the finite-count divisor:
//   mask = ~isnan(x)
//   nfin = sum(mask)                       (per reduced region)
//   mu   = sum(mask*x) / nfin              (= nanmean(x))
//   dmu  = sum(mask*dx) / nfin             (= nanmean(dx), NaN positions = 0)
//   dvar = 2 * sum(mask*(x-mu)*(dx-dmu)) / (nfin - correction)
// All NaN positions contribute 0 (masked) so no NaN leaks into the result.
// Returns dvar reduced with the caller's Dim/Keepdim.
inline auto nanvar_differential(const Tensor& x_in, const Tensor& dx_in,
                                const OpAttributes& attrs) -> Tensor {
    std::optional<int64_t> dim;
    if (attrs.has(AttrKey::Dim)) {
        dim = attrs.get_int(AttrKey::Dim);
    }
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    // nanvar/nanstd set BOTH AttrKey::Unbiased (bool) and AttrKey::Correction
    // (int). Prefer the explicit integer correction (covers ddof != {0,1});
    // fall back to the bool when only Unbiased is present.
    int64_t correction = attrs.has(AttrKey::Correction)
        ? attrs.get_int(AttrKey::Correction)
        : (attrs.get_bool(AttrKey::Unbiased, false) ? 1 : 0);

    // is_nan[i] = true where x is NaN; finite positions are its complement.
    auto is_nan = tenzor::isnan(x_in);
    // mask is a {0,1} tensor in x's dtype: 1 at finite positions, 0 at NaN.
    auto mask = tenzor::where(is_nan, tenzor::zeros_like(x_in),
                              tenzor::ones_like(x_in));
    // x / dx with NaN positions replaced by 0 so masked products are finite.
    auto x0 = tenzor::where(is_nan, tenzor::zeros_like(x_in), x_in);
    auto dx0 = tenzor::where(is_nan, tenzor::zeros_like(dx_in), dx_in);

    // Finite count per reduced region (keepdim for broadcasting).
    auto nfin = tenzor::sum(mask, dim, /*keepdim=*/true);          // >= 0
    auto mu  = tenzor::div(tenzor::sum(x0, dim, /*keepdim=*/true), nfin);
    auto dmu = tenzor::div(tenzor::sum(dx0, dim, /*keepdim=*/true), nfin);

    auto centered_p = tenzor::mul(mask, tenzor::sub(x0, mu));      // 0 at NaN
    auto centered_t = tenzor::mul(mask, tenzor::sub(dx0, dmu));    // 0 at NaN
    auto numer = tenzor::sum(tenzor::mul(centered_p, centered_t), dim, keepdim);

    // Divisor matches the primal nanvar: finite_count - correction.
    auto nfin_red = keepdim ? nfin : tenzor::sum(mask, dim, /*keepdim=*/false);
    Tensor denom = nfin_red;
    if (correction != 0) {
        auto corr = tenzor::full({1}, static_cast<double>(correction),
                                 x_in.dtype(), x_in.device());
        denom = tenzor::sub(nfin_red, corr);
    }
    auto tangent = tenzor::div(tenzor::mul(numer, 2.0), denom);
    return tangent;
}

JvpResult jvp_adapter_nanvar(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_nanvar: expected 1 input");
    }
    auto x  = make_dual(primals[0], tangents[0]);
    auto primal = tenzor::dispatch(OpId::NanVar,
                                   std::vector<Tensor>{x.primal()}, attrs)[0];
    // Linear variance differential (the previous NanVar(masked_dx) returned
    // the *variance of dx* — quadratic, always non-negative, and not the
    // directional derivative).
    auto tangent = nanvar_differential(x.primal(), x.tangent(), attrs);
    return JvpResult{std::move(primal), std::move(tangent)};
}
JvpResult jvp_adapter_nanstd(std::span<const Tensor> primals,
                             std::span<const Tensor> tangents,
                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_nanstd: expected 1 input");
    }
    // nanstd = sqrt(nanvar); d/dt = d(nanvar) / (2·nanstd).
    auto x = make_dual(primals[0], tangents[0]);
    auto primal = tenzor::dispatch(OpId::NanStd,
                                   std::vector<Tensor>{x.primal()}, attrs)[0];
    auto d_var = nanvar_differential(x.primal(), x.tangent(), attrs);
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

// ============================================================================
// Audit S15 batch — implement 26+ JVP rules previously marked NonDifferentiable
// ============================================================================
//
// Conventions:
//
//   - For linear forward ops (DCT, IDCT, STFT, ISTFT, MelScale, MFCC,
//     SparseToDense, DenseToSparse), the JVP is the same op applied to the
//     tangent. We re-dispatch the same OpId on dx.
//
//   - For nonlinear forwards (CDist, PairwiseDistance, Pdist, CosineSimilarity,
//     Renorm, Cov, Corrcoef, LinalgVectorNorm, LinalgMatrixNorm, SparseSoftmax,
//     SparseLogSoftmax, SparseSpGEMM, SparseTrsv, SparseTrsm,
//     BatchNorm2dFusedTraining, NestedLayerNorm, NestedAttention, CTCLoss,
//     Quantile / Nanquantile / Nanmedian), we derive a closed-form JVP using
//     compositions of primitives that themselves have registered JVP rules.
//
//   - The new adapters live in their own anonymous namespace to avoid clashing
//     with the existing `jvp_adapter_nondiff_*` symbols (which remain in the
//     file but become unreferenced after the registration switch).
//
// Each adapter follows the same template: shape/arity checks → resolve attrs
// → compute primal output (either by dispatching the same OpId or via a
// composition matching the kernel formula) → compute the tangent.

namespace {

// ---- Helper: full-zero tangent for a primal whose tangent slot is empty ---
inline auto zeros_like_tensor(const Tensor& ref) -> Tensor {
    auto shape_vec = std::vector<int64_t>(ref.shape().begin(), ref.shape().end());
    return tenzor::zeros(shape_vec, ref.dtype(), ref.device());
}

inline auto tangent_or_zeros(const Tensor& primal, const Tensor& tangent) -> Tensor {
    // Match make_dual's absent-tangent convention: a tangent is "absent" iff it
    // is the empty sentinel (numel()==0) for a non-empty primal. Using a numel
    // *equality* test instead silently dropped genuinely-supplied broadcastable
    // or scalar tangents (numel differs from primal) and mis-classified an
    // absent tangent on an empty primal as present.
    return (tangent.numel() == 0 && primal.numel() != 0)
        ? zeros_like_tensor(primal)
        : tangent;
}

// ---- Linear forward adapters (re-dispatch on tangent) ---------------------

// DCT: y = DCT(x) — linear; tangent = DCT(dx) with the same (type, n, dim, norm).
JvpResult jvp_adapter_dct_s15(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_dct_s15: expected 1 input");
    }
    auto primal  = tenzor::dispatch(OpId::DCT,
        std::vector<Tensor>{primals[0]}, attrs)[0];
    auto dx = tangent_or_zeros(primals[0], tangents[0]);
    auto tangent = tenzor::dispatch(OpId::DCT,
        std::vector<Tensor>{dx}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

JvpResult jvp_adapter_idct_s15(std::span<const Tensor> primals,
                               std::span<const Tensor> tangents,
                               const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_idct_s15: expected 1 input");
    }
    auto primal  = tenzor::dispatch(OpId::IDCT,
        std::vector<Tensor>{primals[0]}, attrs)[0];
    auto dx = tangent_or_zeros(primals[0], tangents[0]);
    auto tangent = tenzor::dispatch(OpId::IDCT,
        std::vector<Tensor>{dx}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// STFT(x, window) — linear in x for a fixed window; window's tangent is
// refused (treating window as a learned param would require multi-input
// linearisation we don't currently support).
JvpResult jvp_adapter_stft_s15(std::span<const Tensor> primals,
                               std::span<const Tensor> tangents,
                               const OpAttributes& attrs) {
    if (primals.size() < 1 || primals.size() != tangents.size()) {
        throw std::runtime_error("jvp_adapter_stft_s15: expected at least input");
    }
    if (primals.size() > 1 && tangents[1].numel() != 0) {
        throw NonDifferentiable(
            "STFT forward-mode JVP w.r.t. window is not implemented; "
            "window is treated as a fixed kernel.");
    }
    std::vector<Tensor> p_in(primals.begin(), primals.end());
    std::vector<Tensor> t_in = p_in;
    t_in[0] = tangent_or_zeros(primals[0], tangents[0]);
    auto primal  = tenzor::dispatch(OpId::STFT, p_in, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::STFT, t_in, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

JvpResult jvp_adapter_istft_s15(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() < 1 || primals.size() != tangents.size()) {
        throw std::runtime_error("jvp_adapter_istft_s15: expected at least input");
    }
    if (primals.size() > 1 && tangents[1].numel() != 0) {
        throw NonDifferentiable(
            "ISTFT forward-mode JVP w.r.t. window is not implemented.");
    }
    std::vector<Tensor> p_in(primals.begin(), primals.end());
    std::vector<Tensor> t_in = p_in;
    t_in[0] = tangent_or_zeros(primals[0], tangents[0]);
    auto primal  = tenzor::dispatch(OpId::ISTFT, p_in, attrs)[0];
    auto tangent = tenzor::dispatch(OpId::ISTFT, t_in, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// MelScale: y = filterbank @ x, linear in x.
JvpResult jvp_adapter_mel_scale_s15(std::span<const Tensor> primals,
                                    std::span<const Tensor> tangents,
                                    const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_mel_scale_s15: expected 1 input");
    }
    auto primal  = tenzor::dispatch(OpId::MelScale,
        std::vector<Tensor>{primals[0]}, attrs)[0];
    auto dx = tangent_or_zeros(primals[0], tangents[0]);
    auto tangent = tenzor::dispatch(OpId::MelScale,
        std::vector<Tensor>{dx}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// MFCC composite forward (STFT, |.|^2, mel_scale, log, DCT). The composite
// is NOT linear (the |.|^2 and log stages are nonlinear). The simplest
// correct JVP composes per-stage tangents using primitives whose JVP rules
// are already registered (fft, abs, square via mul, mel_scale, log, dct).
// We replay the forward of fft::mfcc with explicit primitives so that each
// step's JVP composes through DualTensors.
JvpResult jvp_adapter_mfcc_s15(std::span<const Tensor> primals,
                               std::span<const Tensor> tangents,
                               const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_mfcc_s15: expected 1 input");
    }
    int64_t sample_rate = attrs.get_int(AttrKey::SampleRate, 16000);
    int64_t n_mfcc = attrs.get_int(AttrKey::NumMFCC, 40);
    int64_t n_mels = attrs.get_int(AttrKey::NumMels, 128);
    int64_t n_fft = attrs.get_int(AttrKey::NFft, 400);
    int64_t hop_length = attrs.get_int(AttrKey::HopLength, 160);
    double f_min = attrs.get_float(AttrKey::FMin, 0.0);
    double f_max = attrs.get_float(AttrKey::FMax, 0.0);
    const Tensor& x = primals[0];
    auto dx = tangent_or_zeros(x, tangents[0]);

    // Stage 1: STFT (linear). y_stft is complex.
    auto y_stft  = tenzor::fft::stft(x,  n_fft, hop_length, -1, Tensor());
    auto dy_stft = tenzor::fft::stft(dx, n_fft, hop_length, -1, Tensor());

    // Stage 2: Power = re^2 + im^2 (nonlinear).
    auto re = tenzor::real(y_stft);
    auto im = tenzor::imag(y_stft);
    auto dre = tenzor::real(dy_stft);
    auto dim_ = tenzor::imag(dy_stft);
    auto power_spec = tenzor::add(tenzor::mul(re, re), tenzor::mul(im, im));
    auto two = tenzor::full({}, 2.0, x.dtype(), x.device());
    auto d_power = tenzor::add(
        tenzor::mul(tenzor::mul(re, dre), two),
        tenzor::mul(tenzor::mul(im, dim_), two));

    // Stage 3: MelScale (linear).
    OpAttributes mel_attrs;
    mel_attrs.set(AttrKey::NumMels, n_mels);
    mel_attrs.set(AttrKey::FMin, f_min);
    mel_attrs.set(AttrKey::FMax, f_max);
    mel_attrs.set(AttrKey::SampleRate, sample_rate);
    auto mel_spec  = tenzor::dispatch(OpId::MelScale,
        std::vector<Tensor>{power_spec}, mel_attrs)[0];
    auto d_mel_spec = tenzor::dispatch(OpId::MelScale,
        std::vector<Tensor>{d_power}, mel_attrs)[0];

    // Stage 4: log(mel + 1e-10) (nonlinear). d(log(z)) = dz / z.
    auto shifted = tenzor::add(mel_spec, 1e-10);
    auto log_mel = tenzor::log(shifted);
    auto d_log_mel = tenzor::div(d_mel_spec, shifted);

    // Stage 5: DCT-II ortho along dim -2 (linear).
    int64_t ndim = log_mel.ndim();
    int64_t mel_dim = ndim - 2;
    OpAttributes dct_attrs;
    dct_attrs.set(AttrKey::DCTType, int64_t{2});
    dct_attrs.set(AttrKey::Dim, mel_dim);
    dct_attrs.set(AttrKey::Norm, std::string("ortho"));
    auto dct_result   = tenzor::dispatch(OpId::DCT,
        std::vector<Tensor>{log_mel}, dct_attrs)[0];
    auto d_dct_result = tenzor::dispatch(OpId::DCT,
        std::vector<Tensor>{d_log_mel}, dct_attrs)[0];

    // Stage 6: slice to n_mfcc along dim mel_dim. (Linear; tangent slices too.)
    int64_t out_mel_dim = dct_result.ndim() - 2;
    auto primal_out  = tenzor::slice(dct_result,   out_mel_dim, 0, n_mfcc);
    auto tangent_out = tenzor::slice(d_dct_result, out_mel_dim, 0, n_mfcc);
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// SparseToDense: linear in `values`; indices are int (no tangent).
// Inputs: [crow, col, values], attrs M, K.
JvpResult jvp_adapter_sparse_to_dense_s15(std::span<const Tensor> primals,
                                          std::span<const Tensor> tangents,
                                          const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_sparse_to_dense_s15: expected 3 inputs (crow, col, values)");
    }
    auto primal = tenzor::dispatch(OpId::SparseToDense,
        std::vector<Tensor>{primals[0], primals[1], primals[2]}, attrs)[0];
    auto d_values = tangent_or_zeros(primals[2], tangents[2]);
    auto tangent = tenzor::dispatch(OpId::SparseToDense,
        std::vector<Tensor>{primals[0], primals[1], d_values}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// DenseToSparse: the differentiable values-tangent (dx gathered at the nonzero
// pattern) cannot be returned through the single-output JVP surface because
// outs[0] is the integer crow_indices structural tensor. It is registered as
// NonDifferentiable (jvp_adapter_nondiff_dense_to_sparse) rather than
// fabricating a zero crow_indices-shaped tangent. A multi-output JVP path would
// be required to surface the values tangent.

// ---- Nonlinear adapters via closed-form chain rule ------------------------

// LinalgVectorNorm: y = ||x||_p along dim, with optional keepdim. The JVP is
//   dy = ((|x|^(p-1) * sign(x)) · dx) / y^(p-1)
// for p < inf, and  dy = (sign(x_argmax) · dx) for p = inf.
// Special cases: p = 2 → dy = (x · dx) / y; p = 1 → dy = (sign(x) · dx).
JvpResult jvp_adapter_linalg_vector_norm_s15(std::span<const Tensor> primals,
                                             std::span<const Tensor> tangents,
                                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_linalg_vector_norm_s15: expected 1 input");
    }
    const Tensor& x = primals[0];
    auto dx = tangent_or_zeros(x, tangents[0]);
    double p   = attrs.get_float(AttrKey::P, 2.0);
    int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);

    auto primal_out = tenzor::dispatch(OpId::LinalgVectorNorm,
        std::vector<Tensor>{x}, attrs)[0];

    Tensor tangent_out;
    if (p == 2.0) {
        // y = sqrt(sum(x^2)); dy = sum(x*dx) / y
        auto num = tenzor::mul(x, dx);
        Tensor s;
        if (dim == INT64_MIN) {
            s = tenzor::sum(num);
        } else {
            s = tenzor::sum(num, dim, keepdim);
        }
        // Avoid divide-by-zero
        auto eps = tenzor::full({}, 1e-30, primal_out.dtype(), primal_out.device());
        tangent_out = tenzor::div(s, tenzor::add(primal_out, eps));
    } else if (p == 1.0) {
        // y = sum(|x|); dy = sum(sign(x) * dx)
        auto sgn = tenzor::sign(x);
        auto num = tenzor::mul(sgn, dx);
        if (dim == INT64_MIN) {
            tangent_out = tenzor::sum(num);
        } else {
            tangent_out = tenzor::sum(num, dim, keepdim);
        }
    } else {
        // General p: y = (sum(|x|^p))^(1/p);
        // dy = (sum(|x|^(p-1) * sign(x) * dx)) / y^(p-1)
        auto abs_x = tenzor::abs(x);
        auto sgn = tenzor::sign(x);
        auto pm1 = tenzor::pow(abs_x, p - 1.0);
        auto factor = tenzor::mul(tenzor::mul(pm1, sgn), dx);
        Tensor s;
        if (dim == INT64_MIN) {
            s = tenzor::sum(factor);
        } else {
            s = tenzor::sum(factor, dim, keepdim);
        }
        auto y_pm1 = tenzor::pow(primal_out, p - 1.0);
        auto eps = tenzor::full({}, 1e-30, primal_out.dtype(), primal_out.device());
        tangent_out = tenzor::div(s, tenzor::add(y_pm1, eps));
    }
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// LinalgMatrixNorm: ord = 0 (Frobenius), 1 (nuclear), 2 (spectral).
// Frobenius is a vector p=2 norm of the flattened matrix → same JVP shape.
// Nuclear and spectral are tied to SVD, which is NonDifferentiable in this
// codebase; we refuse those orders and supply Frobenius.
JvpResult jvp_adapter_linalg_matrix_norm_s15(std::span<const Tensor> primals,
                                             std::span<const Tensor> tangents,
                                             const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_linalg_matrix_norm_s15: expected 1 input");
    }
    int64_t ord = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
    if (ord != 0) {
        throw NonDifferentiable(
            "LinalgMatrixNorm JVP only implemented for Frobenius (ord=0); "
            "nuclear/spectral norms require SVD JVP which is not yet supported.");
    }
    const Tensor& x = primals[0];
    auto dx = tangent_or_zeros(x, tangents[0]);
    auto primal_out = tenzor::dispatch(OpId::LinalgMatrixNorm,
        std::vector<Tensor>{x}, attrs)[0];
    // Frobenius: y = sqrt(sum(x*x)); dy = sum(x*dx)/y.
    auto num = tenzor::sum(tenzor::mul(x, dx));
    auto eps = tenzor::full({}, 1e-30, primal_out.dtype(), primal_out.device());
    auto tangent_out = tenzor::div(num, tenzor::add(primal_out, eps));
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// CosineSimilarity: y = (x·y) / (||x|| ||y||)  along `dim`.
//
// Let dot = sum_d(x*y, dim), nx = ||x||_2, ny = ||y||_2 (with eps floor).
//   d(dot)  = sum_d(dx*y + x*dy, dim)
//   d(nx)   = sum_d(x*dx, dim) / nx
//   d(ny)   = sum_d(y*dy, dim) / ny
// y_cos = dot/(nx*ny); apply quotient rule:
//   dy_cos = d(dot)/(nx*ny) - y_cos*(d(nx)/nx + d(ny)/ny)
JvpResult jvp_adapter_cosine_similarity_s15(std::span<const Tensor> primals,
                                            std::span<const Tensor> tangents,
                                            const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_cosine_similarity_s15: expected 2 inputs (x, y)");
    }
    const Tensor& a = primals[0];
    const Tensor& b = primals[1];
    auto da = tangent_or_zeros(a, tangents[0]);
    auto db = tangent_or_zeros(b, tangents[1]);
    int64_t dim = attrs.get_int(AttrKey::Dim, 1);
    double eps  = attrs.get_float(AttrKey::Eps, 1e-8);

    auto primal_out = tenzor::dispatch(OpId::CosineSimilarity,
        std::vector<Tensor>{a, b}, attrs)[0];

    // Match the primal kernel's denominator exactly (math.cpp cosine kernel):
    //   D = sqrt(na2) * sqrt(nb2) + eps          (eps added once to the product)
    // The earlier form sqrt(na2+eps^2)*sqrt(nb2+eps^2) differentiates a
    // different function than the primal, breaking forward-mode gradcheck for
    // small-magnitude inputs. Here primal = dot / D and
    //   tangent = (d_dot - primal * dD) / D,
    //   dD = na' * sqrt(nb2) + sqrt(na2) * nb',
    //   na' = sum(a*da) / sqrt(na2), nb' = sum(b*db) / sqrt(nb2).
    // A tiny floor guards the divisions by the norms (the sqrt of a sum of
    // squares can be exactly zero for a zero input vector).
    auto dot = tenzor::sum(tenzor::mul(a, b), dim, false);
    auto na2 = tenzor::sum(tenzor::mul(a, a), dim, false);
    auto nb2 = tenzor::sum(tenzor::mul(b, b), dim, false);
    const double kNormFloor = 1e-30;
    auto norm_a = tenzor::sqrt(tenzor::add(na2,
        tenzor::full({}, kNormFloor, a.dtype(), a.device())));
    auto norm_b = tenzor::sqrt(tenzor::add(nb2,
        tenzor::full({}, kNormFloor, b.dtype(), b.device())));

    auto D = tenzor::add(tenzor::mul(norm_a, norm_b),
        tenzor::full({}, eps, a.dtype(), a.device()));

    auto d_dot = tenzor::sum(
        tenzor::add(tenzor::mul(da, b), tenzor::mul(a, db)), dim, false);
    auto d_na  = tenzor::div(tenzor::sum(tenzor::mul(a, da), dim, false), norm_a);
    auto d_nb  = tenzor::div(tenzor::sum(tenzor::mul(b, db), dim, false), norm_b);

    auto dD = tenzor::add(tenzor::mul(d_na, norm_b), tenzor::mul(norm_a, d_nb));

    auto tangent_out = tenzor::div(
        tenzor::sub(d_dot, tenzor::mul(primal_out, dD)), D);
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// CDist(x1, x2, p): pairwise p-distance.
//   y[i,j] = ||x1[i] - x2[j]||_p
//   dy[i,j] = (sum_d |x1[i,d]-x2[j,d]|^(p-1) * sign(x1[i,d]-x2[j,d])
//             * (dx1[i,d] - dx2[j,d])) / y[i,j]^(p-1)
JvpResult jvp_adapter_cdist_s15(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error("jvp_adapter_cdist_s15: expected 2 inputs (x1, x2)");
    }
    double p = attrs.get_float(AttrKey::DistP, 2.0);
    const Tensor& x1 = primals[0];
    const Tensor& x2 = primals[1];
    auto dx1 = tangent_or_zeros(x1, tangents[0]);
    auto dx2 = tangent_or_zeros(x2, tangents[1]);

    auto primal_out = tenzor::dispatch(OpId::CDist,
        std::vector<Tensor>{x1, x2}, attrs)[0];

    // Build broadcast shape: x1 [...M,D], x2 [...N,D] -> diff [...M,N,D]
    int64_t nd1 = x1.ndim();
    int64_t nd2 = x2.ndim();
    if (nd1 < 2 || nd2 < 2) {
        throw std::runtime_error("jvp_adapter_cdist_s15: inputs must be >= 2D");
    }
    auto x1_e  = tenzor::unsqueeze(x1, nd1 - 1);   // [...M,1,D]
    auto x2_e  = tenzor::unsqueeze(x2, nd2 - 2);   // [...1,N,D]
    auto dx1_e = tenzor::unsqueeze(dx1, nd1 - 1);
    auto dx2_e = tenzor::unsqueeze(dx2, nd2 - 2);

    auto diff   = tenzor::sub(x1_e, x2_e);
    auto d_diff = tenzor::sub(dx1_e, dx2_e);
    auto abs_d  = tenzor::abs(diff);
    auto sgn    = tenzor::sign(diff);
    int64_t reduce_dim = diff.ndim() - 1;

    Tensor tangent_out;
    if (p == 2.0) {
        auto num = tenzor::sum(tenzor::mul(diff, d_diff), reduce_dim, false);
        auto eps = tenzor::full({}, 1e-30, primal_out.dtype(), primal_out.device());
        tangent_out = tenzor::div(num, tenzor::add(primal_out, eps));
    } else if (p == 1.0) {
        auto num = tenzor::sum(tenzor::mul(sgn, d_diff), reduce_dim, false);
        tangent_out = num;
    } else {
        auto pm1 = tenzor::pow(abs_d, p - 1.0);
        auto factor = tenzor::mul(tenzor::mul(pm1, sgn), d_diff);
        auto num = tenzor::sum(factor, reduce_dim, false);
        auto y_pm1 = tenzor::pow(primal_out, p - 1.0);
        auto eps = tenzor::full({}, 1e-30, primal_out.dtype(), primal_out.device());
        tangent_out = tenzor::div(num, tenzor::add(y_pm1, eps));
    }
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// PairwiseDistance(x1, x2, p): ||x1 - x2||_p along last dim (single pair per row).
JvpResult jvp_adapter_pairwise_distance_s15(std::span<const Tensor> primals,
                                            std::span<const Tensor> tangents,
                                            const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_pairwise_distance_s15: expected 2 inputs (x1, x2)");
    }
    double p = attrs.get_float(AttrKey::DistP, 2.0);
    const Tensor& x1 = primals[0];
    const Tensor& x2 = primals[1];
    auto dx1 = tangent_or_zeros(x1, tangents[0]);
    auto dx2 = tangent_or_zeros(x2, tangents[1]);

    auto primal_out = tenzor::dispatch(OpId::PairwiseDistance,
        std::vector<Tensor>{x1, x2}, attrs)[0];

    auto diff   = tenzor::sub(x1, x2);
    auto d_diff = tenzor::sub(dx1, dx2);
    auto sgn    = tenzor::sign(diff);
    auto abs_d  = tenzor::abs(diff);
    int64_t reduce_dim = diff.ndim() - 1;

    Tensor tangent_out;
    if (p == 2.0) {
        auto num = tenzor::sum(tenzor::mul(diff, d_diff), reduce_dim, false);
        auto eps = tenzor::full({}, 1e-30, primal_out.dtype(), primal_out.device());
        tangent_out = tenzor::div(num, tenzor::add(primal_out, eps));
    } else if (p == 1.0) {
        tangent_out = tenzor::sum(tenzor::mul(sgn, d_diff), reduce_dim, false);
    } else {
        auto pm1 = tenzor::pow(abs_d, p - 1.0);
        auto factor = tenzor::mul(tenzor::mul(pm1, sgn), d_diff);
        auto num = tenzor::sum(factor, reduce_dim, false);
        auto y_pm1 = tenzor::pow(primal_out, p - 1.0);
        auto eps = tenzor::full({}, 1e-30, primal_out.dtype(), primal_out.device());
        tangent_out = tenzor::div(num, tenzor::add(y_pm1, eps));
    }
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// Pdist(x, p): pairwise distances between all rows of x. Output is the
// upper-triangular flattened distances. We can express it as a CDist on
// (x, x) and then a triu+flatten — the JVP follows the CDist closed form.
// Simpler approach: tangent of pdist is computed in the same triangular
// layout the kernel uses. Since we can't easily project to that layout from
// outside the kernel, we delegate to compute the JVP by composing:
//   y      = pdist(x, p)
//   dy_ij  = (sum_d ...formula...) but we'd still need to know which (i,j)
//            ordering the kernel uses.
// Pragmatic substitute: re-dispatch pdist on a synthesised input that
// captures the perturbation. We compute the FD JVP analytically by computing
// the cdist version (which gives a full M×M matrix) and extracting the
// upper triangle using the same convention as the Pdist kernel
// (row-major upper triangle without diagonal).
JvpResult jvp_adapter_pdist_s15(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_pdist_s15: expected 1 input");
    }
    double p = attrs.get_float(AttrKey::DistP, 2.0);
    const Tensor& x = primals[0];
    auto dx = tangent_or_zeros(x, tangents[0]);

    auto primal_out = tenzor::dispatch(OpId::Pdist,
        std::vector<Tensor>{x}, attrs)[0];

    // Compute full pairwise distance matrix using CDist semantics, then
    // take the upper triangle without the diagonal in row-major order.
    OpAttributes cd_attrs;
    cd_attrs.set(AttrKey::DistP, p);
    auto full = tenzor::dispatch(OpId::CDist,
        std::vector<Tensor>{x, x}, cd_attrs)[0];  // [M, M]

    // Build pairwise tangent matrix using the CDist closed form on (x, x).
    int64_t M = x.shape()[0];
    int64_t D = x.shape()[1];
    auto x_i  = tenzor::unsqueeze(x,  1);  // [M, 1, D]
    auto x_j  = tenzor::unsqueeze(x,  0);  // [1, M, D]
    auto dx_i = tenzor::unsqueeze(dx, 1);
    auto dx_j = tenzor::unsqueeze(dx, 0);
    auto diff   = tenzor::sub(x_i, x_j);     // [M, M, D]
    auto d_diff = tenzor::sub(dx_i, dx_j);
    auto sgn    = tenzor::sign(diff);
    auto abs_d  = tenzor::abs(diff);

    Tensor d_full;
    if (p == 2.0) {
        auto num = tenzor::sum(tenzor::mul(diff, d_diff), 2, false);
        auto eps = tenzor::full({}, 1e-30, full.dtype(), full.device());
        d_full = tenzor::div(num, tenzor::add(full, eps));
    } else if (p == 1.0) {
        d_full = tenzor::sum(tenzor::mul(sgn, d_diff), 2, false);
    } else {
        auto pm1 = tenzor::pow(abs_d, p - 1.0);
        auto factor = tenzor::mul(tenzor::mul(pm1, sgn), d_diff);
        auto num = tenzor::sum(factor, 2, false);
        auto y_pm1 = tenzor::pow(full, p - 1.0);
        auto eps = tenzor::full({}, 1e-30, full.dtype(), full.device());
        d_full = tenzor::div(num, tenzor::add(y_pm1, eps));
    }

    // Extract row-major upper triangle (i < j) into a flat tensor of length M*(M-1)/2.
    // We do this by gathering with explicit indices.
    int64_t out_len = M * (M - 1) / 2;
    std::vector<int64_t> ii; ii.reserve(out_len);
    std::vector<int64_t> jj; jj.reserve(out_len);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = i + 1; j < M; ++j) {
            ii.push_back(i);
            jj.push_back(j);
        }
    }
    auto ii_t = Tensor::from_blob(ii.data(), {out_len}, DType::Int64, Device::cpu()).to(x.device()).clone();
    auto jj_t = Tensor::from_blob(jj.data(), {out_len}, DType::Int64, Device::cpu()).to(x.device()).clone();
    // d_full is [M, M]. Linear index = i*M + j.
    auto lin = tenzor::add(tenzor::mul(ii_t, M), jj_t);
    auto flat = tenzor::reshape(d_full, std::vector<int64_t>{M * M});
    auto tangent_out = tenzor::index_select(flat, 0, lin);
    (void)D;
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// Cov(X, correction): X[N,M] -> [N,N] sample covariance.
// Cov = centered @ centered^T / (M-correction), centered = X - mean(X, dim=1, keepdim=true).
// dCov = (d_centered @ centered^T + centered @ d_centered^T) / (M-correction),
//   where d_centered = dX - mean(dX, dim=1, keepdim=true).
JvpResult jvp_adapter_cov_s15(std::span<const Tensor> primals,
                              std::span<const Tensor> tangents,
                              const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_cov_s15: expected 1 input");
    }
    int64_t correction = attrs.get_int(AttrKey::Correction, 1);
    const Tensor& X = primals[0];
    auto dX = tangent_or_zeros(X, tangents[0]);

    auto primal_out = tenzor::dispatch(OpId::Cov,
        std::vector<Tensor>{X}, attrs)[0];

    Tensor X2d = (X.ndim() == 1)
        ? tenzor::reshape(X, std::vector<int64_t>{1, X.shape()[0]})
        : X;
    Tensor dX2d = (dX.ndim() == 1)
        ? tenzor::reshape(dX, std::vector<int64_t>{1, dX.shape()[0]})
        : dX;
    int64_t M = X2d.shape()[1];
    double denom = static_cast<double>(M - correction);

    auto row_means_x  = tenzor::mean(X2d,  1, true);
    auto row_means_dx = tenzor::mean(dX2d, 1, true);
    auto centered    = tenzor::sub(X2d,  row_means_x);
    auto d_centered  = tenzor::sub(dX2d, row_means_dx);
    auto ct = tenzor::transpose(centered, 0, 1);
    auto dct = tenzor::transpose(d_centered, 0, 1);
    auto sum_xy = tenzor::add(
        tenzor::matmul(d_centered, ct),
        tenzor::matmul(centered, dct));
    auto tangent_out = tenzor::div(sum_xy,
        tenzor::full({}, denom, sum_xy.dtype(), sum_xy.device()));
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// Corrcoef(X) = D^{-1} Cov(X) D^{-1}, where D = diag(sqrt(diag(Cov))).
// Use quotient rule. Since the kernel is composed, compute directly via the
// composition.
JvpResult jvp_adapter_corrcoef_s15(std::span<const Tensor> primals,
                                   std::span<const Tensor> tangents,
                                   const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_corrcoef_s15: expected 1 input");
    }
    const Tensor& X = primals[0];
    auto dX = tangent_or_zeros(X, tangents[0]);

    auto primal_out = tenzor::dispatch(OpId::Corrcoef,
        std::vector<Tensor>{X}, attrs)[0];

    // Compute cov + d_cov first.
    OpAttributes cov_attrs;
    cov_attrs.set(AttrKey::Correction, int64_t{1});
    auto cov_mat = tenzor::dispatch(OpId::Cov,
        std::vector<Tensor>{X}, cov_attrs)[0];

    // d_cov via cov JVP (recursive composition).
    auto cov_jvp = jvp_adapter_cov_s15(primals, tangents, cov_attrs);
    Tensor d_cov = std::move(cov_jvp.tangent);

    // d_corr[i,j] = d_cov[i,j] / (sd[i]*sd[j])
    //              - 0.5 * cov[i,j] * (d_cov[i,i]/(cov[i,i]*sd[i]*sd[j])
    //                                 + d_cov[j,j]/(cov[j,j]*sd[i]*sd[j]))
    // Simpler form: let r = corr, c = cov, sd_i = sqrt(c[i,i]).
    //   r[i,j] = c[i,j] / (sd_i sd_j)
    //   dr[i,j] = dc[i,j]/(sd_i sd_j) - r[i,j]/2 * (dc[i,i]/c[i,i] + dc[j,j]/c[j,j])
    auto diag_idx = tenzor::arange(0.0,
        static_cast<double>(cov_mat.shape()[0]),
        1.0, DType::Int64, cov_mat.device());
    auto stride_scalar = tenzor::full({}, static_cast<double>(cov_mat.shape()[1]),
        DType::Int64, cov_mat.device());
    auto stride = tenzor::add(tenzor::mul(diag_idx, stride_scalar), diag_idx);
    auto flat_c  = tenzor::reshape(cov_mat,
        std::vector<int64_t>{cov_mat.shape()[0] * cov_mat.shape()[1]});
    auto flat_dc = tenzor::reshape(d_cov,
        std::vector<int64_t>{d_cov.shape()[0] * d_cov.shape()[1]});
    auto cii  = tenzor::index_select(flat_c,  0, stride);  // diagonal of cov
    auto dcii = tenzor::index_select(flat_dc, 0, stride);
    auto eps = tenzor::full({}, 1e-30, cii.dtype(), cii.device());
    auto sd  = tenzor::sqrt(tenzor::add(cii, eps));
    auto ratio = tenzor::div(dcii, tenzor::add(cii, eps));   // [N]
    auto ratio_i = tenzor::unsqueeze(ratio, 1);              // [N,1]
    auto ratio_j = tenzor::unsqueeze(ratio, 0);              // [1,N]
    auto sd_i = tenzor::unsqueeze(sd, 1);                    // [N,1]
    auto sd_j = tenzor::unsqueeze(sd, 0);
    auto sd_ij = tenzor::mul(sd_i, sd_j);                    // [N,N]
    auto half = tenzor::full({}, 0.5, primal_out.dtype(), primal_out.device());
    auto correction_term = tenzor::mul(
        tenzor::mul(primal_out, half),
        tenzor::add(ratio_i, ratio_j));
    auto tangent_out = tenzor::sub(
        tenzor::div(d_cov, sd_ij),
        correction_term);
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// Renorm(x, p, dim, maxnorm): each slice along `dim` is scaled if its
// p-norm exceeds maxnorm.
//   y = x * c, where c = min(1, maxnorm / (||x_slice||_p + eps))
// JVP: When ||x_slice||_p < maxnorm, c == 1 and dy = dx.
//   When ||x_slice||_p >= maxnorm, c = maxnorm/||x||_p:
//     dy = dx * c + x * dc
//     dc = -(maxnorm / ||x||_p^2) * d||x||_p
JvpResult jvp_adapter_renorm_s15(std::span<const Tensor> primals,
                                 std::span<const Tensor> tangents,
                                 const OpAttributes& attrs) {
    if (primals.size() != 1 || tangents.size() != 1) {
        throw std::runtime_error("jvp_adapter_renorm_s15: expected 1 input");
    }
    double p       = attrs.get_float(AttrKey::P, 2.0);
    int64_t dim    = attrs.get_int(AttrKey::Dim, 0);
    double maxnorm = attrs.get_float(AttrKey::MaxNorm, 1.0);
    const Tensor& x = primals[0];
    auto dx = tangent_or_zeros(x, tangents[0]);

    auto primal_out = tenzor::dispatch(OpId::Renorm,
        std::vector<Tensor>{x}, attrs)[0];

    // Compute per-slice p-norm. Renorm normalises slices along `dim`, which
    // means the slice index runs along `dim`; each slice is the full sub-
    // tensor at that index across all other dims. We need ||slice||_p with
    // shape broadcastable back to x's shape.
    // Reduce over all dims EXCEPT dim.
    int64_t ndim = x.ndim();
    int64_t norm_dim = (dim < 0) ? (dim + ndim) : dim;
    std::vector<int64_t> reduce_dims;
    for (int64_t d = 0; d < ndim; ++d) if (d != norm_dim) reduce_dims.push_back(d);

    Tensor norm_per_slice;
    {
        // Compute via abs.pow(p).sum(reduce_dims, keepdim=true).pow(1/p)
        auto abs_x = tenzor::abs(x);
        Tensor s = tenzor::pow(abs_x, p);
        for (int64_t i = static_cast<int64_t>(reduce_dims.size()) - 1; i >= 0; --i) {
            s = tenzor::sum(s, reduce_dims[i], true);
        }
        norm_per_slice = tenzor::pow(s, 1.0 / p);
    }
    auto eps = tenzor::full({}, 1e-30, x.dtype(), x.device());
    auto norm_plus = tenzor::add(norm_per_slice, eps);

    // Compute d(norm)/d(x) chain: same as LinalgVectorNorm with reduce_dims.
    Tensor d_norm;
    {
        auto abs_x = tenzor::abs(x);
        auto sgn   = tenzor::sign(x);
        auto pm1   = tenzor::pow(abs_x, p - 1.0);
        auto factor = tenzor::mul(tenzor::mul(pm1, sgn), dx);
        Tensor s = factor;
        for (int64_t i = static_cast<int64_t>(reduce_dims.size()) - 1; i >= 0; --i) {
            s = tenzor::sum(s, reduce_dims[i], true);
        }
        auto y_pm1 = tenzor::pow(norm_plus, p - 1.0);
        d_norm = tenzor::div(s, y_pm1);
    }

    // Compute scaling c (clamped):
    //   c = clamp(maxnorm / norm, 0, 1) when norm > maxnorm; else 1.
    // d_c = where(norm > maxnorm, -(maxnorm/norm^2)*d_norm, 0)
    auto maxnorm_t = tenzor::full({}, maxnorm, x.dtype(), x.device());
    // mask_exceeds: boolean tensor, but stay in float realm via comparison-to-float.
    // We approximate using: c = min(1, maxnorm/norm).
    auto ratio = tenzor::div(maxnorm_t, norm_plus);
    auto ones  = tenzor::full({}, 1.0, x.dtype(), x.device());
    // exceeds == 1.0 where norm > maxnorm, 0 otherwise (STRICT boundary,
    // matching renorm_kernel's `if (norm_val > maxnorm)` check in
    // src/backends/cpu/kernels/math.cpp and RenormBackward's
    // pass-through-at-equality convention in
    // src/autograd/function_new_ops.cpp). The previous
    // `0.5*sign(delta)+0.5` blend evaluated to exactly 0.5 at norm==maxnorm
    // — half of the "scaled" branch's derivative bleeding into a point both
    // the kernel and reverse-mode treat as pure identity. clamp(sign(delta),
    // 0, 1) instead gives a strict step: sign(delta) in {-1,0,1} maps to
    // {0,0,1} — 0 (pass-through) at delta<=0 INCLUDING exactly delta==0,
    // 1 (scaled) only strictly above. Mirrors jvp_clamp's own boundary
    // technique (an inclusive sign-based mask resolving to the
    // actually-executed branch, not a smooth blend) above.
    auto delta = tenzor::sub(norm_per_slice, maxnorm_t);
    auto exceeds = tenzor::clamp(tenzor::sign(delta), 0.0, 1.0);
    auto c_when_exceeds = ratio;
    auto c_when_not = ones;
    // c = exceeds * ratio + (1-exceeds) * 1
    auto one_minus = tenzor::sub(ones, exceeds);
    auto c = tenzor::add(
        tenzor::mul(exceeds, c_when_exceeds),
        tenzor::mul(one_minus, c_when_not));
    // dc_when_exceeds = -maxnorm/norm^2 * d_norm = -ratio/norm * d_norm
    auto dc_exc = tenzor::mul(
        tenzor::div(tenzor::mul(maxnorm_t, tenzor::full({}, -1.0, x.dtype(), x.device())),
                    tenzor::mul(norm_plus, norm_plus)),
        d_norm);
    auto d_c = tenzor::mul(exceeds, dc_exc);

    // dy = dx * c + x * dc
    auto tangent_out = tenzor::add(
        tenzor::mul(dx, c),
        tenzor::mul(x, d_c));
    (void)primal_out;
    return JvpResult{std::move(primal_out), std::move(tangent_out)};
}

// ---- Sparse ops on values tensor (sparsity pattern held fixed) ------------
//
// SparseSpGEMM(A_sp, B_sp): C_sp = A_sp * B_sp (CSR x CSR).
// Inputs: [A_crow, A_col, A_values, B_crow, B_col, B_values].
// Outputs: [C_crow, C_col, C_values] (audit H4: the single-output adapter
// this replaced returned outs[0] = C_crow — an integer structural tensor —
// as the "primal", not the values, and always returned a zero tangent even
// though the true values-tangent is well-defined and bilinear:
//   dC.values = SpGEMM(A_pattern, dA.values, B) + SpGEMM(A, B_pattern, dB.values)
// A CSR product's sparsity pattern is a function of (crow, col) alone, not of
// values, so substituting a tangent values buffer while keeping the SAME
// (crow, col) pattern lands the result on exactly C's own pattern — the two
// contributions sum position-for-position with C.values, no re-indexing
// needed. crow/col themselves are non-differentiable structural outputs
// (zero tangent, matching every other index/structural JVP output in this
// file, e.g. jvp_adapter_cummax_impl above).
JvpMultiResult jvp_adapter_sparse_spgemm_s15(std::span<const Tensor> primals,
                                             std::span<const Tensor> tangents,
                                             const OpAttributes& attrs) {
    if (primals.size() != 6 || tangents.size() != 6) {
        throw std::runtime_error(
            "jvp_adapter_sparse_spgemm_s15: expected 6 inputs "
            "(A_crow, A_col, A_values, B_crow, B_col, B_values)");
    }
    const Tensor& A_crow = primals[0];
    const Tensor& A_col = primals[1];
    const Tensor& A_values = primals[2];
    const Tensor& B_crow = primals[3];
    const Tensor& B_col = primals[4];
    const Tensor& B_values = primals[5];
    const Tensor& dA_values = tangents[2];
    const Tensor& dB_values = tangents[5];

    auto outs = tenzor::dispatch(OpId::SparseSpGEMM,
        std::vector<Tensor>{A_crow, A_col, A_values, B_crow, B_col, B_values}, attrs);
    Tensor C_crow = outs[0];
    Tensor C_col = outs[1];
    Tensor C_values = outs[2];

    int64_t M = attrs.get_int(AttrKey::M);
    int64_t N = attrs.get_int(AttrKey::N);

    // Each contribution below is computed via a fresh SpGEMM dispatch, which
    // prunes exact-zero accumulated entries from its OWN output pattern (see
    // cpu_spgemm_typed) — a contribution's nnz can legitimately be SMALLER
    // than C's own pattern (e.g. dB_values == 0 makes the whole B-side
    // contribution structurally empty). Summing two differently-pruned
    // values tensors positionally would misalign or crash. Route through the
    // dense M x N intermediate instead — exact, and sidesteps pattern
    // alignment entirely — then gather back onto C's own (row, col)
    // positions so the tangent is parallel to C_values.
    OpAttributes dense_attrs;
    dense_attrs.set(AttrKey::M, M);
    dense_attrs.set(AttrKey::K, N);  // SparseToDense's kernel names its 2nd dim "K"

    Tensor dense_tangent = tenzor::zeros({M, N}, C_values.dtype(), C_values.device());
    if (dA_values.numel() > 0) {
        auto outs_a = tenzor::dispatch(OpId::SparseSpGEMM,
            std::vector<Tensor>{A_crow, A_col, dA_values, B_crow, B_col, B_values}, attrs);
        auto dense_a = tenzor::dispatch(OpId::SparseToDense,
            std::vector<Tensor>{outs_a[0], outs_a[1], outs_a[2]}, dense_attrs)[0];
        dense_tangent = tenzor::add(dense_tangent, dense_a);
    }
    if (dB_values.numel() > 0) {
        auto outs_b = tenzor::dispatch(OpId::SparseSpGEMM,
            std::vector<Tensor>{A_crow, A_col, A_values, B_crow, B_col, dB_values}, attrs);
        auto dense_b = tenzor::dispatch(OpId::SparseToDense,
            std::vector<Tensor>{outs_b[0], outs_b[1], outs_b[2]}, dense_attrs)[0];
        dense_tangent = tenzor::add(dense_tangent, dense_b);
    }

    // Gather dense_tangent[row, col] for each of C's nnz positions, in C's
    // own CSR order. crow/col are tiny (O(nnz)) index metadata — expanding
    // crow to a per-nnz row index on the host is standard CSR practice and
    // mirrors what cpu_spgemm_typed itself does with raw index pointers.
    int64_t nnz_c = C_values.numel();
    Tensor values_tangent;
    if (nnz_c == 0) {
        values_tangent = tenzor::zeros_like(C_values);
    } else {
        Tensor crow_cpu = C_crow.to(Device::cpu()).contiguous();
        Tensor col_cpu = C_col.to(Device::cpu()).contiguous();
        Tensor flat_idx_cpu = Tensor({nnz_c}, DType::Int64, Device::cpu());
        {
            const int64_t* crow_ptr = crow_cpu.data<int64_t>();
            const int64_t* col_ptr = col_cpu.data<int64_t>();
            int64_t* flat_ptr = flat_idx_cpu.data<int64_t>();
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t p = crow_ptr[i]; p < crow_ptr[i + 1]; ++p) {
                    flat_ptr[p] = i * N + col_ptr[p];
                }
            }
        }
        Tensor flat_idx = flat_idx_cpu.to(dense_tangent.device());
        Tensor dense_flat = tenzor::reshape(dense_tangent, {M * N});
        values_tangent = tenzor::index_select(dense_flat, 0, flat_idx);
    }

    Tensor crow_tangent = tenzor::zeros_like(C_crow);
    Tensor col_tangent = tenzor::zeros_like(C_col);

    JvpMultiResult result;
    result.primals  = { std::move(C_crow), std::move(C_col), std::move(C_values) };
    result.tangents = { std::move(crow_tangent), std::move(col_tangent),
                         std::move(values_tangent) };
    return result;
}

// SparseTrsv: L @ x = b ; x = L^{-1} b (with L lower or upper triangular).
// Inputs: [L_crow, L_col, L_values, b]. Linear in b; for fixed L_values,
// dx_b = L^{-1} db.  For L_values held fixed (structural tangent zero on
// L), we propagate db only. We refuse L_values tangents (would require
// A-side trsv contribution X' = -L^{-1}(L' X)).
JvpResult jvp_adapter_sparse_trsv_s15(std::span<const Tensor> primals,
                                      std::span<const Tensor> tangents,
                                      const OpAttributes& attrs) {
    if (primals.size() != 4 || tangents.size() != 4) {
        throw std::runtime_error(
            "jvp_adapter_sparse_trsv_s15: expected 4 inputs (crow, col, values, b)");
    }
    if (tangents[2].numel() != 0) {
        throw NonDifferentiable(
            "SparseTrsv forward-mode JVP w.r.t. L_values is not yet implemented; "
            "only the b-side tangent is supported. Pass zero L_values tangent.");
    }
    auto primal = tenzor::dispatch(OpId::SparseTrsv,
        std::vector<Tensor>{primals[0], primals[1], primals[2], primals[3]}, attrs)[0];
    auto db = tangent_or_zeros(primals[3], tangents[3]);
    auto tangent = tenzor::dispatch(OpId::SparseTrsv,
        std::vector<Tensor>{primals[0], primals[1], primals[2], db}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

JvpResult jvp_adapter_sparse_trsm_s15(std::span<const Tensor> primals,
                                      std::span<const Tensor> tangents,
                                      const OpAttributes& attrs) {
    if (primals.size() != 4 || tangents.size() != 4) {
        throw std::runtime_error(
            "jvp_adapter_sparse_trsm_s15: expected 4 inputs (crow, col, values, B)");
    }
    if (tangents[2].numel() != 0) {
        throw NonDifferentiable(
            "SparseTrsm forward-mode JVP w.r.t. L_values is not yet implemented.");
    }
    auto primal = tenzor::dispatch(OpId::SparseTrsm,
        std::vector<Tensor>{primals[0], primals[1], primals[2], primals[3]}, attrs)[0];
    auto dB = tangent_or_zeros(primals[3], tangents[3]);
    auto tangent = tenzor::dispatch(OpId::SparseTrsm,
        std::vector<Tensor>{primals[0], primals[1], primals[2], dB}, attrs)[0];
    return JvpResult{std::move(primal), std::move(tangent)};
}

// SparseSoftmax / SparseLogSoftmax operate on `values` only (per-row softmax
// over nonzero values in CSR layout). The exact forward-mode tangents are:
//   softmax:     dy = P * (dv - rowsum(P * dv))
//   log-softmax: dy = dv - P * rowsum(dv)            (P = exp(log-softmax))
// where rowsum(x) is the sum of x over the nonzeros of each CSR row, broadcast
// back to every nonzero position in that row. The previous adapters dropped
// the rowsum correction entirely (softmax returned P*dv; log-softmax returned
// dv), which is wrong for any CSR row with more than one nonzero.
//
// We compute the per-row sum directly from crow_indices (the CSR row pointer)
// and scatter it back to each nonzero's position. The reduction is done on a
// CPU copy of the per-value tensor (Float32/Float64; Float16/BFloat16 widen to
// Float32, then narrow back to the values dtype), mirroring the sparse_softmax
// kernel's own CSR loop, and the result is moved back to the values device.
//
// csr_row_broadcast_sum(crow, x) -> y, where y[j] = sum over all nonzeros k in
// the same CSR row as j of x[k]. crow has length M+1 (M = number of rows);
// x and y are 1-D of length nnz.
inline auto csr_row_broadcast_sum(const Tensor& crow, const Tensor& x) -> Tensor {
    // Widen reduced dtypes to Float32 so the segment reduction stays exact for
    // the half-precision sparse paths; narrow the result back at the end.
    const DType orig_dtype = x.dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Tensor x_work = widen ? x.to(DType::Float32) : x;

    auto crow_cpu = (crow.device().type != Device::Type::CPU) ? crow.to(Device::cpu()) : crow;
    auto x_cpu = (x_work.device().type != Device::Type::CPU) ? x_work.to(Device::cpu()) : x_work;

    const int64_t nnz = x_cpu.numel();
    const int64_t M = crow_cpu.numel() - 1;
    Tensor out = tenzor::zeros({nnz}, x_cpu.dtype(), Device::cpu());
    const int64_t* row_ptr = crow_cpu.data<int64_t>();

    if (x_cpu.dtype() == DType::Float32) {
        const float* xv = x_cpu.data<float>();
        float* ov = out.data<float>();
        for (int64_t i = 0; i < M; ++i) {
            int64_t start = row_ptr[i];
            int64_t end = row_ptr[i + 1];
            if (start == end) continue;
            float s = 0.0f;
            for (int64_t j = start; j < end; ++j) s += xv[j];
            for (int64_t j = start; j < end; ++j) ov[j] = s;
        }
    } else if (x_cpu.dtype() == DType::Float64) {
        const double* xv = x_cpu.data<double>();
        double* ov = out.data<double>();
        for (int64_t i = 0; i < M; ++i) {
            int64_t start = row_ptr[i];
            int64_t end = row_ptr[i + 1];
            if (start == end) continue;
            double s = 0.0;
            for (int64_t j = start; j < end; ++j) s += xv[j];
            for (int64_t j = start; j < end; ++j) ov[j] = s;
        }
    } else {
        throw std::runtime_error(
            "csr_row_broadcast_sum: unsupported values dtype for sparse "
            "softmax/log-softmax JVP");
    }

    // Restore original device and dtype.
    if (x.device().type != Device::Type::CPU) {
        out = out.to(x.device());
    }
    if (widen) {
        out = out.to(orig_dtype);
    }
    return out;
}

// SparseSoftmax: dy = P * (dv - rowsum(P * dv)).
JvpResult jvp_adapter_sparse_softmax_s15(std::span<const Tensor> primals,
                                         std::span<const Tensor> tangents,
                                         const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_sparse_softmax_s15: expected 3 inputs (crow, col, values)");
    }
    auto primal = tenzor::dispatch(OpId::SparseSoftmax,
        std::vector<Tensor>{primals[0], primals[1], primals[2]}, attrs)[0];
    auto dvalues = tangent_or_zeros(primals[2], tangents[2]);
    // P = primal (softmax values). dy = P * (dv - rowsum(P * dv)).
    auto p_times_dv = tenzor::mul(primal, dvalues);
    auto rowsum = csr_row_broadcast_sum(primals[0], p_times_dv);
    auto tangent = tenzor::mul(primal, tenzor::sub(dvalues, rowsum));
    return JvpResult{std::move(primal), std::move(tangent)};
}

// SparseLogSoftmax: dy = dv - rowsum(P * dv), where P = exp(log-softmax).
JvpResult jvp_adapter_sparse_log_softmax_s15(std::span<const Tensor> primals,
                                             std::span<const Tensor> tangents,
                                             const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_sparse_log_softmax_s15: expected 3 inputs (crow, col, values)");
    }
    auto primal = tenzor::dispatch(OpId::SparseLogSoftmax,
        std::vector<Tensor>{primals[0], primals[1], primals[2]}, attrs)[0];
    auto dvalues = tangent_or_zeros(primals[2], tangents[2]);
    // P = exp(log-softmax). dy_i = dv_i - sum_k(P_k * dv_k) over the row.
    auto P = tenzor::exp(primal);
    auto rowsum = csr_row_broadcast_sum(primals[0], tenzor::mul(P, dvalues));
    auto tangent = tenzor::sub(dvalues, rowsum);
    return JvpResult{std::move(primal), std::move(tangent)};
}

// ---- BatchNorm2dFusedTraining multi-output ---------------------------------
// PyTorch's training-mode BN: y = (x - mean) / sqrt(var + eps), with mean
// and var computed per-channel over (N, H, W). The JVP through training-
// mode BN is:
//   rstd = 1/sqrt(var + eps)
//   y    = (x - mean) * rstd
//   dmean = mean(dx)
//   dvar  = mean((x-mean) * dx) * 2
//   drstd = -0.5 * dvar * rstd^3
//   dy    = (dx - dmean) * rstd + (x - mean) * drstd
JvpMultiResult jvp_adapter_batchnorm2d_fused_training_s15(
        std::span<const Tensor> primals,
        std::span<const Tensor> tangents,
        const OpAttributes& attrs) {
    if (primals.size() < 1) {
        throw std::runtime_error(
            "jvp_adapter_batchnorm2d_fused_training_s15: expected at least 1 input");
    }
    // We dispatch the kernel once to get the canonical primal outputs
    auto outs = tenzor::dispatch(OpId::BatchNorm2dFusedTraining,
        std::vector<Tensor>(primals.begin(), primals.end()), attrs);
    // Build a closed-form JVP via composition of primitive ops. Widen
    // Float16/BFloat16 to Float32 for the reduction-heavy mean/variance
    // chain (catastrophic cancellation in (x-mean)^2), mirroring the
    // reverse-mode BatchNorm backward kernels; only the composed tangent
    // (dy) needs narrowing back — the primal outputs come from the real
    // kernel dispatch above and are already in the native dtype.
    const DType orig_dtype = primals[0].dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Tensor x = widen ? primals[0].to(DType::Float32) : primals[0];
    auto dx = tangent_or_zeros(primals[0], tangents[0]);
    if (widen) dx = dx.to(DType::Float32);
    // weight / bias slots (if present) carry their own tangents.
    Tensor weight = (primals.size() > 1) ? primals[1] : Tensor();
    Tensor bias   = (primals.size() > 2) ? primals[2] : Tensor();
    Tensor dweight = (primals.size() > 1 && tangents.size() > 1) ?
        tangent_or_zeros(weight, tangents[1]) : Tensor();
    Tensor dbias = (primals.size() > 2 && tangents.size() > 2) ?
        tangent_or_zeros(bias, tangents[2]) : Tensor();
    if (widen) {
        if (weight.numel() != 0) weight = weight.to(DType::Float32);
        if (bias.numel() != 0) bias = bias.to(DType::Float32);
        if (dweight.numel() != 0) dweight = dweight.to(DType::Float32);
        if (dbias.numel() != 0) dbias = dbias.to(DType::Float32);
    }
    double eps = attrs.get_float(AttrKey::Eps, 1e-5);

    // x is [N, C, H, W]. Reduce over dims [0, 2, 3] for per-channel mean/var.
    auto mean_x = tenzor::mean(tenzor::mean(tenzor::mean(x, 3, true), 2, true), 0, true);
    auto centered = tenzor::sub(x, mean_x);
    auto var_x = tenzor::mean(tenzor::mean(tenzor::mean(
        tenzor::mul(centered, centered), 3, true), 2, true), 0, true);
    auto rstd = tenzor::pow(tenzor::add(var_x,
        tenzor::full({}, eps, x.dtype(), x.device())), -0.5);
    auto y_norm = tenzor::mul(centered, rstd);

    auto mean_dx = tenzor::mean(tenzor::mean(tenzor::mean(dx, 3, true), 2, true), 0, true);
    auto d_centered = tenzor::sub(dx, mean_dx);
    auto two = tenzor::full({}, 2.0, x.dtype(), x.device());
    auto d_var = tenzor::mul(two,
        tenzor::mean(tenzor::mean(tenzor::mean(
            tenzor::mul(centered, d_centered), 3, true), 2, true), 0, true));
    auto neg_half = tenzor::full({}, -0.5, x.dtype(), x.device());
    auto d_rstd = tenzor::mul(neg_half,
        tenzor::mul(d_var, tenzor::pow(rstd, 3.0)));

    auto dy_norm = tenzor::add(
        tenzor::mul(d_centered, rstd),
        tenzor::mul(centered, d_rstd));

    // Apply affine if weight/bias are present.
    Tensor dy = dy_norm;
    if (weight.numel() != 0) {
        // weight broadcast: [C] -> [1, C, 1, 1]
        auto w_view = tenzor::reshape(weight,
            std::vector<int64_t>{1, weight.shape()[0], 1, 1});
        auto dw_view = (dweight.numel() != 0) ? tenzor::reshape(dweight,
            std::vector<int64_t>{1, dweight.shape()[0], 1, 1}) : Tensor();
        dy = tenzor::mul(dy_norm, w_view);
        if (dweight.numel() != 0) {
            dy = tenzor::add(dy, tenzor::mul(y_norm, dw_view));
        }
        if (bias.numel() != 0 && dbias.numel() != 0) {
            auto db_view = tenzor::reshape(dbias,
                std::vector<int64_t>{1, dbias.shape()[0], 1, 1});
            dy = tenzor::add(dy, db_view);
        }
    }

    if (widen) {
        dy = dy.to(orig_dtype);
    }

    // outs is {y, save_mean, save_invstd, ...}. save_mean's and
    // save_invstd's tangents are NOT zero — they were already computed above
    // as mean_dx (d(mean_x)) and d_rstd (d(rstd)) for the dy_norm chain rule
    // and simply discarded here previously. Reshape them to match the
    // kernel's actual save_mean/save_invstd shape (computed above with
    // keepdim=true as [1,C,1,1]; the dispatched outputs may be [C]) and
    // narrow back from the widened Float32 compute dtype, mirroring dy's own
    // narrowing just above. Any FURTHER outputs (e.g. updated running
    // mean/var, which need momentum-aware EMA differentiation not
    // implemented here) remain zero.
    std::vector<Tensor> primal_outs = outs;
    std::vector<Tensor> tangent_outs;
    tangent_outs.reserve(primal_outs.size());
    tangent_outs.push_back(std::move(dy));
    if (primal_outs.size() > 1) {
        std::vector<int64_t> mean_shape(primal_outs[1].shape().begin(),
                                        primal_outs[1].shape().end());
        Tensor d_mean = tenzor::reshape(mean_dx, mean_shape);
        if (widen) d_mean = d_mean.to(orig_dtype);
        tangent_outs.push_back(std::move(d_mean));
    }
    if (primal_outs.size() > 2) {
        std::vector<int64_t> invstd_shape(primal_outs[2].shape().begin(),
                                          primal_outs[2].shape().end());
        Tensor d_invstd = tenzor::reshape(d_rstd, invstd_shape);
        if (widen) d_invstd = d_invstd.to(orig_dtype);
        tangent_outs.push_back(std::move(d_invstd));
    }
    for (size_t i = 3; i < primal_outs.size(); ++i) {
        tangent_outs.push_back(zeros_like_tensor(primal_outs[i]));
    }
    return JvpMultiResult{std::move(primal_outs), std::move(tangent_outs)};
}

} // anonymous (S15 batch)

// ============================================================================
// Wave-4 JVP: normalization layers (GroupNorm / InstanceNorm / RMSNorm) and
// Chunk / Split. These follow the LayerNorm template (jvp_adapter_layer_norm)
// adapted to each layer's reduction axes and output stat shapes.
// ============================================================================

namespace {

inline Tensor jvp_zeros_like_or(const Tensor& t, const Tensor& tan) {
    if (tan.numel() != 0) return tan;
    auto sh = std::vector<int64_t>(t.shape().begin(), t.shape().end());
    return tenzor::zeros(sh, t.dtype(), t.device());
}

// Build a [1, C, 1, ...] broadcast shape (xnd dims) for a per-channel vector.
inline std::vector<int64_t> jvp_channel_bcast_shape(int64_t xnd, int64_t C) {
    std::vector<int64_t> s(static_cast<size_t>(xnd), 1);
    if (xnd >= 2) s[1] = C;
    return s;
}

}  // namespace

// GroupNorm kernel contract:
//   inputs : (x, gamma[C], beta[C]); attr NumGroups|Groups, Eps
//   outputs: {y, mean[N,G], inv_std[N,G]}
// Normalisation runs per (n, g) over the (C/G) channels in the group and all
// spatial positions. We reshape x -> (N, G, M) with M = (C/G)*prod(spatial),
// reduce over the last axis, then map the normalised result back and apply the
// per-channel affine. Algebra mirrors LayerNorm with the group axis as the
// reduction axis.
JvpMultiResult jvp_adapter_group_norm_s15(std::span<const Tensor> primals,
                                          std::span<const Tensor> tangents,
                                          const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_group_norm_s15: expected 3 inputs (x, gamma, beta)");
    }
    // Widen Float16/BFloat16 to Float32 for the reduction-heavy mean/variance
    // chain (catastrophic cancellation in (x-mean)^2), mirroring
    // NestedLayerNormBackward's own widen-then-narrow and jvp_adapter_layer_norm.
    const DType orig_dtype = primals[0].dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Tensor x     = widen ? primals[0].to(DType::Float32) : primals[0];
    Tensor gamma = widen ? primals[1].to(DType::Float32) : primals[1];
    Tensor beta  = widen ? primals[2].to(DType::Float32) : primals[2];
    Tensor dx     = jvp_zeros_like_or(primals[0], tangents[0]);
    Tensor dgamma = jvp_zeros_like_or(primals[1], tangents[1]);
    Tensor dbeta  = jvp_zeros_like_or(primals[2], tangents[2]);
    if (widen) {
        dx     = dx.to(DType::Float32);
        dgamma = dgamma.to(DType::Float32);
        dbeta  = dbeta.to(DType::Float32);
    }

    double eps = attrs.get_float(AttrKey::Eps, 1e-5);
    int64_t G = attrs.get_int(AttrKey::NumGroups, attrs.get_int(AttrKey::Groups, 1));

    const auto xshape = x.shape();
    const int64_t xnd = static_cast<int64_t>(xshape.size());
    if (xnd < 2) {
        throw std::runtime_error("jvp_adapter_group_norm_s15: input rank must be >= 2");
    }
    const int64_t N = xshape[0];
    const int64_t C = xshape[1];
    if (G <= 0 || C % G != 0) {
        throw std::runtime_error("jvp_adapter_group_norm_s15: C not divisible by num_groups");
    }
    const int64_t M = x.numel() / (N * G);  // (C/G) * spatial

    const std::vector<int64_t> grp_shape = {N, G, M};
    auto orig_shape = std::vector<int64_t>(xshape.begin(), xshape.end());

    auto xg  = tenzor::reshape(x,  grp_shape);
    auto dxg = tenzor::reshape(dx, grp_shape);

    auto mean   = tenzor::mean(xg, /*dim=*/2, /*keepdim=*/true);     // (N,G,1)
    auto xmm    = tenzor::sub(xg, mean);
    auto var    = tenzor::mean(tenzor::mul(xmm, xmm), 2, true);      // (N,G,1)
    auto rstd   = tenzor::rsqrt(tenzor::add(var, eps));              // (N,G,1)

    auto dmean  = tenzor::mean(dxg, 2, true);
    auto two_xmm = tenzor::mul(xmm, 2.0);
    auto dvar   = tenzor::mean(tenzor::mul(two_xmm, tenzor::sub(dxg, dmean)), 2, true);
    auto drstd  = tenzor::mul(tenzor::mul(tenzor::mul(rstd, rstd), rstd),
                              tenzor::mul(dvar, -0.5));

    auto y_norm_g  = tenzor::mul(xmm, rstd);                          // (N,G,M)
    auto dy_norm_g = tenzor::add(tenzor::mul(tenzor::sub(dxg, dmean), rstd),
                                 tenzor::mul(xmm, drstd));

    auto y_norm  = tenzor::reshape(y_norm_g,  orig_shape);           // (N,C,*sp)
    auto dy_norm = tenzor::reshape(dy_norm_g, orig_shape);

    auto c_shape  = jvp_channel_bcast_shape(xnd, C);
    auto gamma_b  = tenzor::reshape(gamma,  c_shape);
    auto beta_b   = tenzor::reshape(beta,   c_shape);
    auto dgamma_b = tenzor::reshape(dgamma, c_shape);
    auto dbeta_b  = tenzor::reshape(dbeta,  c_shape);

    auto y  = tenzor::add(tenzor::mul(y_norm, gamma_b), beta_b);
    auto dy = tenzor::add(tenzor::add(tenzor::mul(dy_norm, gamma_b),
                                      tenzor::mul(y_norm, dgamma_b)),
                          dbeta_b);

    // Stats returned as (N, G).
    const std::vector<int64_t> ng_shape = {N, G};
    auto mean_ng  = tenzor::reshape(mean,  ng_shape);
    auto rstd_ng  = tenzor::reshape(rstd,  ng_shape);
    auto dmean_ng = tenzor::reshape(dmean, ng_shape);
    auto drstd_ng = tenzor::reshape(drstd, ng_shape);

    if (widen) {
        y        = y.to(orig_dtype);
        dy       = dy.to(orig_dtype);
        mean_ng  = mean_ng.to(orig_dtype);
        rstd_ng  = rstd_ng.to(orig_dtype);
        dmean_ng = dmean_ng.to(orig_dtype);
        drstd_ng = drstd_ng.to(orig_dtype);
    }

    JvpMultiResult result;
    result.primals  = { std::move(y),  std::move(mean_ng),  std::move(rstd_ng)  };
    result.tangents = { std::move(dy), std::move(dmean_ng), std::move(drstd_ng) };
    return result;
}

// InstanceNorm kernel contract:
//   inputs : (x, gamma[C], beta[C]); attr Eps
//   outputs: {y, mean[N,C], inv_std[N,C]}
// Identical to GroupNorm with one group per channel (G == C): each (n, c)
// normalises over spatial positions only.
JvpMultiResult jvp_adapter_instance_norm_s15(std::span<const Tensor> primals,
                                             std::span<const Tensor> tangents,
                                             const OpAttributes& attrs) {
    if (primals.size() != 3 || tangents.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_instance_norm_s15: expected 3 inputs (x, gamma, beta)");
    }
    const Tensor& x = primals[0];
    const auto xshape = x.shape();
    if (xshape.size() < 2) {
        throw std::runtime_error("jvp_adapter_instance_norm_s15: input rank must be >= 2");
    }
    // Reuse the GroupNorm derivation with G == C.
    OpAttributes a = attrs;
    a.set(AttrKey::NumGroups, xshape[1]);
    return jvp_adapter_group_norm_s15(primals, tangents, a);
}

// RMSNorm kernel contract:
//   inputs : (x, weight[norm_size]); attr Eps
//   outputs: {y, rrms[batch]}
// Normalises over the trailing `weight.numel()` elements, no mean-centering:
//   ms   = mean(x^2, last_dim)
//   rrms = 1/sqrt(ms + eps)
//   y    = x * rrms * weight
// JVP: dms = mean(2*x*dx, last_dim); drrms = -0.5*rrms^3*dms;
//      dy = (dx*rrms + x*drrms)*weight + x*rrms*dweight.
JvpMultiResult jvp_adapter_rms_norm_s15(std::span<const Tensor> primals,
                                        std::span<const Tensor> tangents,
                                        const OpAttributes& attrs) {
    if (primals.size() != 2 || tangents.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_rms_norm_s15: expected 2 inputs (x, weight)");
    }
    const Tensor& x = primals[0];
    const Tensor& w = primals[1];
    Tensor dx = jvp_zeros_like_or(x, tangents[0]);
    Tensor dw = jvp_zeros_like_or(w, tangents[1]);

    double eps = attrs.get_float(AttrKey::Eps, 1e-5);
    const int64_t xnd = x.ndim();
    if (xnd < 1) {
        throw std::runtime_error("jvp_adapter_rms_norm_s15: input rank must be >= 1");
    }
    const int64_t last = xnd - 1;
    const int64_t norm_size = w.numel();
    const int64_t batch = x.numel() / norm_size;

    auto ms    = tenzor::mean(tenzor::mul(x, x), last, /*keepdim=*/true);  // (...,1)
    auto rrms  = tenzor::rsqrt(tenzor::add(ms, eps));                      // (...,1)
    auto dms   = tenzor::mean(tenzor::mul(tenzor::mul(x, dx), 2.0), last, true);
    auto drrms = tenzor::mul(tenzor::mul(tenzor::mul(rrms, rrms), rrms),
                             tenzor::mul(dms, -0.5));

    // weight broadcasts over the trailing (norm) dim only; standard broadcast
    // rules align its [norm_size] shape against x's last axis.
    auto y  = tenzor::mul(tenzor::mul(x, rrms), w);
    auto dy = tenzor::add(
        tenzor::mul(tenzor::add(tenzor::mul(dx, rrms), tenzor::mul(x, drrms)), w),
        tenzor::mul(tenzor::mul(x, rrms), dw));

    // rrms is returned flattened to (batch,).
    const std::vector<int64_t> b_shape = {batch};
    auto rrms_b  = tenzor::reshape(rrms,  b_shape);
    auto drrms_b = tenzor::reshape(drrms, b_shape);

    JvpMultiResult result;
    result.primals  = { std::move(y),  std::move(rrms_b)  };
    result.tangents = { std::move(dy), std::move(drrms_b) };
    return result;
}

// Chunk / Split: both are linear in the input (they partition it), so the
// tangent of each output piece is the corresponding piece of the input tangent.
// Re-dispatching the op on the input tangent yields exactly that partition.
JvpMultiResult jvp_adapter_chunk_s15(std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_chunk_s15: expected 1 input");
    }
    Tensor dx = jvp_zeros_like_or(primals[0], tangents.empty() ? Tensor() : tangents[0]);
    auto prim = tenzor::dispatch(OpId::Chunk, std::vector<Tensor>{primals[0]}, attrs);
    auto tang = tenzor::dispatch(OpId::Chunk, std::vector<Tensor>{dx}, attrs);
    JvpMultiResult result;
    result.primals  = std::move(prim);
    result.tangents = std::move(tang);
    return result;
}

JvpMultiResult jvp_adapter_split_s15(std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_split_s15: expected 1 input");
    }
    Tensor dx = jvp_zeros_like_or(primals[0], tangents.empty() ? Tensor() : tangents[0]);
    auto prim = tenzor::dispatch(OpId::Split, std::vector<Tensor>{primals[0]}, attrs);
    auto tang = tenzor::dispatch(OpId::Split, std::vector<Tensor>{dx}, attrs);
    JvpMultiResult result;
    result.primals  = std::move(prim);
    result.tangents = std::move(tang);
    return result;
}

// ============================================================================
// Wave-4 JVP: TopK / Sort. Both select/permute elements of the input along a
// dim and return {values, indices}. The selection is fixed by the (saved)
// integer indices, so the values are a linear gather of the input: the tangent
// of `values` is the same gather applied to the input tangent, and the integer
// `indices` output is non-differentiable (zero tangent).
// ============================================================================
JvpMultiResult jvp_adapter_topk_s15(std::span<const Tensor> primals,
                                    std::span<const Tensor> tangents,
                                    const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_topk_s15: expected 1 input");
    }
    Tensor dx = jvp_zeros_like_or(primals[0], tangents.empty() ? Tensor() : tangents[0]);
    auto outs = tenzor::dispatch(OpId::TopK, std::vector<Tensor>{primals[0]}, attrs);
    const Tensor& values  = outs[0];
    const Tensor& indices = outs[1];

    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    OpAttributes gattr;
    gattr.set(AttrKey::Dim, dim);
    Tensor dvalues = tenzor::dispatch(OpId::Gather,
        std::vector<Tensor>{dx, indices}, gattr)[0];
    Tensor dindices = tenzor::zeros(
        std::vector<int64_t>(indices.shape().begin(), indices.shape().end()),
        indices.dtype(), indices.device());

    JvpMultiResult result;
    result.primals  = { values,  indices  };
    result.tangents = { std::move(dvalues), std::move(dindices) };
    return result;
}

JvpMultiResult jvp_adapter_sort_s15(std::span<const Tensor> primals,
                                    std::span<const Tensor> tangents,
                                    const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_sort_s15: expected 1 input");
    }
    Tensor dx = jvp_zeros_like_or(primals[0], tangents.empty() ? Tensor() : tangents[0]);
    auto outs = tenzor::dispatch(OpId::Sort, std::vector<Tensor>{primals[0]}, attrs);
    const Tensor& values  = outs[0];
    const Tensor& indices = outs[1];

    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    OpAttributes gattr;
    gattr.set(AttrKey::Dim, dim);
    Tensor dvalues = tenzor::dispatch(OpId::Gather,
        std::vector<Tensor>{dx, indices}, gattr)[0];
    Tensor dindices = tenzor::zeros(
        std::vector<int64_t>(indices.shape().begin(), indices.shape().end()),
        indices.dtype(), indices.device());

    JvpMultiResult result;
    result.primals  = { values,  indices  };
    result.tangents = { std::move(dvalues), std::move(dindices) };
    return result;
}

// ============================================================================
// Wave-4 JVP: reduced QR factorization (A = Q R, A is m x n with m >= n).
// Forward differential (standard reduced-QR JVP; mirrors QrBackward):
//   V     = dA R^{-1}                         (right triangular solve)
//   M     = Qᵀ V                              (n x n)
//   Omega = tril(M,-1) - tril(M,-1)ᵀ          (skew-symmetric = Qᵀ dQ)
//   U     = M - Omega                         (upper-triangular = dR R^{-1})
//   dR    = U R
//   dQ    = Q Omega + (V - Q M)               (= Q Omega + (I - Q Qᵀ) V)
// ============================================================================
JvpMultiResult jvp_adapter_linalg_qr_s15(std::span<const Tensor> primals,
                                         std::span<const Tensor> tangents,
                                         const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_linalg_qr_s15: expected 1 input (A)");
    }
    const Tensor& A = primals[0];
    Tensor dA = jvp_zeros_like_or(A, tangents.empty() ? Tensor() : tangents[0]);

    auto outs = tenzor::dispatch(OpId::LinalgQR, std::vector<Tensor>{A}, attrs);
    const Tensor& Q = outs[0];
    const Tensor& R = outs[1];

    // V = dA R^{-1}. solve_triangular(A, B, upper, unitri) solves A X = B (a
    // LEFT solve), so build the right-inverse via the transpose identity:
    //   V R = dA  <=>  Rᵀ Vᵀ = dAᵀ  (Rᵀ is lower-triangular)
    //   Vᵀ = solve_triangular(Rᵀ, dAᵀ, upper=false);  V = (Vᵀ)ᵀ
    Tensor Rt   = tenzor::transpose(R, -1, -2);
    Tensor dAt  = tenzor::transpose(dA, -1, -2);
    Tensor Vt   = tenzor::linalg::solve_triangular(Rt, dAt, /*upper=*/false,
                                                   /*unitriangular=*/false);
    Tensor V    = tenzor::transpose(Vt, -1, -2);

    Tensor Qt = tenzor::transpose(Q, -1, -2);
    Tensor M  = tenzor::matmul(Qt, V);
    Tensor Ml = tenzor::tril(M, -1);
    Tensor Omega = tenzor::sub(Ml, tenzor::transpose(Ml, -1, -2));  // Qᵀ dQ (skew)
    Tensor U  = tenzor::sub(M, Omega);                              // dR R^{-1} (upper)
    Tensor dR = tenzor::matmul(U, R);
    Tensor dQ = tenzor::add(tenzor::matmul(Q, Omega),
                            tenzor::sub(V, tenzor::matmul(Q, M)));

    JvpMultiResult result;
    result.primals  = { Q, R };
    result.tangents = { std::move(dQ), std::move(dR) };
    return result;
}

// ---------------- Tier E: triangular / Cholesky solve family ---------------
//
// These ops are linear-system solves with no gauge freedom: the tangent is
// obtained by differentiating the defining equation once and reapplying the
// same (factor-reusing) solve. All are gradcheck-clean against finite
// differences for generic well-conditioned inputs.

// SolveTriangular: X = A^{-1} B with A triangular (upper/lower, optional unit
// diagonal). A X = B  =>  A dX = dB - dA X  =>  dX = solve_triangular(A, rhs).
// Only the relevant triangle of dA participates (the solve ignores the rest),
// so dA is masked to the same triangle for a consistent tangent.
JvpResult jvp_adapter_solve_triangular_s15(std::span<const Tensor> primals,
                                           std::span<const Tensor> tangents,
                                           const OpAttributes& attrs) {
    if (primals.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_solve_triangular_s15: expected 2 inputs (A, B)");
    }
    const bool upper  = attrs.get_int(AttrKey::Upper, 1) != 0;
    const bool unitri = attrs.get_int(AttrKey::UnitTriangular, 0) != 0;
    const Tensor& A = primals[0];
    const Tensor& B = primals[1];
    Tensor dA = jvp_zeros_like_or(A, tangents.empty()    ? Tensor() : tangents[0]);
    Tensor dB = jvp_zeros_like_or(B, tangents.size() < 2 ? Tensor() : tangents[1]);

    Tensor X = tenzor::linalg::solve_triangular(A, B, upper, unitri);
    Tensor dA_tri = upper ? tenzor::triu(dA, unitri ? 1 : 0)
                          : tenzor::tril(dA, unitri ? -1 : 0);
    Tensor rhs = tenzor::sub(dB, tenzor::matmul(dA_tri, X));
    Tensor dX = tenzor::linalg::solve_triangular(A, rhs, upper, unitri);
    return JvpResult{ std::move(X), std::move(dX) };
}

// CholeskySolve: X = cholesky_solve(B, L) solves A X = B with A = L L^T (lower)
// or A = U^T U (upper).  A dX = dB - dA X, dA = dL L^T + L dL^T (lower form).
// dX = cholesky_solve(dB - dA X, L) reuses the same factor.
JvpResult jvp_adapter_cholesky_solve_s15(std::span<const Tensor> primals,
                                         std::span<const Tensor> tangents,
                                         const OpAttributes& attrs) {
    if (primals.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_cholesky_solve_s15: expected 2 inputs (B, L)");
    }
    const bool upper = attrs.get_int(AttrKey::Upper, 0) != 0;
    const Tensor& B = primals[0];
    const Tensor& L = primals[1];
    Tensor dB = jvp_zeros_like_or(B, tangents.empty()    ? Tensor() : tangents[0]);
    Tensor dL = jvp_zeros_like_or(L, tangents.size() < 2 ? Tensor() : tangents[1]);
    // cholesky_solve uses only the relevant triangle of the factor; mask the
    // factor tangent to the same triangle so the JVP matches the op exactly.
    dL = upper ? tenzor::triu(dL, 0) : tenzor::tril(dL, 0);

    Tensor X  = tenzor::linalg::cholesky_solve(B, L, upper);
    Tensor Lt  = tenzor::transpose(L, -2, -1);
    Tensor dLt = tenzor::transpose(dL, -2, -1);
    Tensor dA = upper
        ? tenzor::add(tenzor::matmul(dLt, L), tenzor::matmul(Lt, dL))   // dU^T U + U^T dU
        : tenzor::add(tenzor::matmul(dL, Lt), tenzor::matmul(L, dLt));  // dL L^T + L dL^T
    Tensor rhs = tenzor::sub(dB, tenzor::matmul(dA, X));
    Tensor dX = tenzor::linalg::cholesky_solve(rhs, L, upper);
    return JvpResult{ std::move(X), std::move(dX) };
}

// CholeskyInverse: Y = cholesky_inverse(L) = A^{-1}, A = L L^T (lower) or
// U^T U (upper).  dY = -Y dA Y, dA = dL L^T + L dL^T (lower form).
JvpResult jvp_adapter_cholesky_inverse_s15(std::span<const Tensor> primals,
                                           std::span<const Tensor> tangents,
                                           const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error(
            "jvp_adapter_cholesky_inverse_s15: expected 1 input (L)");
    }
    const bool upper = attrs.get_int(AttrKey::Upper, 0) != 0;
    const Tensor& L = primals[0];
    Tensor dL = jvp_zeros_like_or(L, tangents.empty() ? Tensor() : tangents[0]);
    // cholesky_inverse uses only the relevant triangle of the factor.
    dL = upper ? tenzor::triu(dL, 0) : tenzor::tril(dL, 0);

    Tensor Y  = tenzor::linalg::cholesky_inverse(L, upper);
    Tensor Lt  = tenzor::transpose(L, -2, -1);
    Tensor dLt = tenzor::transpose(dL, -2, -1);
    Tensor dA = upper
        ? tenzor::add(tenzor::matmul(dLt, L), tenzor::matmul(Lt, dL))
        : tenzor::add(tenzor::matmul(dL, Lt), tenzor::matmul(L, dLt));
    Tensor dY = tenzor::neg(tenzor::matmul(tenzor::matmul(Y, dA), Y));
    return JvpResult{ std::move(Y), std::move(dY) };
}

// TensorInv: generalized inverse. Reshape A to a square matrix M, invert,
// reshape back. Y = inv(M); dY = reshape(-Minv dM Minv).
JvpResult jvp_adapter_tensor_inv_s15(std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_tensor_inv_s15: expected 1 input (A)");
    }
    const int64_t ind = attrs.get_int(AttrKey::Ind, 2);
    const Tensor& A = primals[0];
    Tensor dA = jvp_zeros_like_or(A, tangents.empty() ? Tensor() : tangents[0]);
    auto shape = A.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t rows = 1; for (int64_t i = 0; i < ind; ++i)    rows *= shape[i];
    int64_t cols = 1; for (int64_t i = ind; i < ndim; ++i) cols *= shape[i];

    Tensor M    = tenzor::reshape(A,  {rows, cols});
    Tensor dM   = tenzor::reshape(dA, {rows, cols});
    Tensor Minv = tenzor::linalg::inv(M);
    Tensor dMinv = tenzor::neg(tenzor::matmul(tenzor::matmul(Minv, dM), Minv));

    std::vector<int64_t> out_shape;
    for (int64_t i = ind; i < ndim; ++i) out_shape.push_back(shape[i]);
    for (int64_t i = 0; i < ind; ++i)    out_shape.push_back(shape[i]);
    Tensor Y  = tenzor::reshape(Minv,  out_shape);
    Tensor dY = tenzor::reshape(dMinv, out_shape);
    return JvpResult{ std::move(Y), std::move(dY) };
}

// TensorSolve: A X = B with A reshaped to a square matrix. The differential is
// the ordinary linear-solve rule applied to the flattened system, then
// reshaped to the solution's tensor shape.
JvpResult jvp_adapter_tensor_solve_s15(std::span<const Tensor> primals,
                                       std::span<const Tensor> tangents,
                                       const OpAttributes&) {
    if (primals.size() != 2) {
        throw std::runtime_error("jvp_adapter_tensor_solve_s15: expected 2 inputs (A, B)");
    }
    const Tensor& A = primals[0];
    const Tensor& B = primals[1];
    Tensor dA = jvp_zeros_like_or(A, tangents.empty()    ? Tensor() : tangents[0]);
    Tensor dB = jvp_zeros_like_or(B, tangents.size() < 2 ? Tensor() : tangents[1]);

    auto a_shape = A.shape();
    auto b_shape = B.shape();
    const int64_t a_ndim = static_cast<int64_t>(a_shape.size());
    const int64_t b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t rhs_size = 1; for (int64_t i = 0; i < b_ndim; ++i) rhs_size *= b_shape[i];
    int64_t total = 1;    for (int64_t i = 0; i < a_ndim; ++i) total *= a_shape[i];
    int64_t lhs_size = total / rhs_size;

    Tensor A2d  = tenzor::reshape(A,  {rhs_size, lhs_size});
    Tensor dA2d = tenzor::reshape(dA, {rhs_size, lhs_size});
    Tensor Bcol  = tenzor::unsqueeze(tenzor::reshape(B,  {rhs_size}), -1);
    Tensor dBcol = tenzor::unsqueeze(tenzor::reshape(dB, {rhs_size}), -1);

    Tensor Xcol  = tenzor::linalg::solve(A2d, Bcol);
    Tensor rhs   = tenzor::sub(dBcol, tenzor::matmul(dA2d, Xcol));
    Tensor dXcol = tenzor::linalg::solve(A2d, rhs);

    Tensor Xflat  = tenzor::squeeze(Xcol,  -1);
    Tensor dXflat = tenzor::squeeze(dXcol, -1);
    std::vector<int64_t> x_shape;
    for (int64_t i = b_ndim; i < a_ndim; ++i) x_shape.push_back(a_shape[i]);
    if (x_shape.empty()) {
        return JvpResult{ std::move(Xflat), std::move(dXflat) };
    }
    Tensor Y  = tenzor::reshape(Xflat,  x_shape);
    Tensor dY = tenzor::reshape(dXflat, x_shape);
    return JvpResult{ std::move(Y), std::move(dY) };
}

// LinalgLUSolve: X = A^{-1} B from packed LU factors (A = P L U). The tangent
// w.r.t. the packed factors and B is, using A^{-1} = U^{-1} L^{-1} P^{-1} so
// the permutation cancels:
//   dX = A^{-1} dB - U^{-1} L^{-1} (dL U + L dU) X,
// where L = unit-lower(LU)+I, U = upper(LU), dL = strict-lower(dLU),
// dU = upper(dLU). A^{-1}(·) reuses lu_solve; the L/U solves are triangular.
// The integer pivot tensor carries no tangent.
JvpResult jvp_adapter_lu_solve_s15(std::span<const Tensor> primals,
                                   std::span<const Tensor> tangents,
                                   const OpAttributes&) {
    if (primals.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_lu_solve_s15: expected 3 inputs (LU_data, pivots, B)");
    }
    const Tensor& LU = primals[0];
    const Tensor& piv = primals[1];
    const Tensor& B   = primals[2];
    Tensor dLU = jvp_zeros_like_or(LU, tangents.empty()    ? Tensor() : tangents[0]);
    Tensor dB  = jvp_zeros_like_or(B,  tangents.size() < 3 ? Tensor() : tangents[2]);

    const int64_t n = LU.shape().back();
    Tensor I = tenzor::eye(n, std::nullopt, LU.dtype(), LU.device());
    Tensor L  = tenzor::add(tenzor::tril(LU, -1), I);
    Tensor U  = tenzor::triu(LU, 0);
    Tensor dL = tenzor::tril(dLU, -1);
    Tensor dU = tenzor::triu(dLU, 0);

    Tensor X = tenzor::linalg::lu_solve(LU, piv, B);
    // (dL U + L dU) X
    Tensor dAX = tenzor::add(tenzor::matmul(dL, tenzor::matmul(U, X)),
                             tenzor::matmul(L,  tenzor::matmul(dU, X)));
    // U^{-1} L^{-1} dAX  (L unit-lower, U upper).
    Tensor t = tenzor::linalg::solve_triangular(L, dAX, /*upper=*/false, /*unitri=*/true);
    t        = tenzor::linalg::solve_triangular(U, t,   /*upper=*/true,  /*unitri=*/false);
    Tensor dX = tenzor::sub(tenzor::linalg::lu_solve(LU, piv, dB), t);
    return JvpResult{ std::move(X), std::move(dX) };
}

// LinalgLU: A = P L U (L unit-lower, U upper, pivots in LAPACK 1-based
// sequential-swap convention).  With the pivot pattern locally constant,
//   dA = P (dL U + L dU)  =>  Pᵀ dA = dL U + L dU
//   M := L⁻¹ (Pᵀ dA) U⁻¹ = L⁻¹ dL + dU U⁻¹,
// where L⁻¹ dL is strictly lower and dU U⁻¹ is upper, so
//   dL = L · tril(M, -1),   dU = triu(M, 0) · U.
// Pᵀ is reconstructed from the pivot sequence as a permutation matrix
// (applied per batch element), which keeps the rule batch-correct.
JvpMultiResult jvp_adapter_linalg_lu_s15(std::span<const Tensor> primals,
                                         std::span<const Tensor> tangents,
                                         const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_linalg_lu_s15: expected 1 input (A)");
    }
    const Tensor& A = primals[0];
    Tensor dA = jvp_zeros_like_or(A, tangents.empty() ? Tensor() : tangents[0]);

    auto outs = tenzor::dispatch(OpId::LinalgLU, std::vector<Tensor>{A}, attrs);
    const Tensor& L = outs[0];
    const Tensor& U = outs[1];
    const Tensor& piv = outs[2];

    auto shape = A.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t n = shape[ndim - 1];
    int64_t batch = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch *= shape[i];

    // Build Pᵀ as a batched permutation matrix on the host (Float64), then cast
    // to A's dtype/device.  Applying the sequential getrf swaps to the identity
    // index yields the row map of Pᵀ (Pᵀ A = L U).
    Tensor piv_cpu = piv.to(Device::cpu()).contiguous();
    const int32_t* p = piv_cpu.data<int32_t>();
    Tensor Pt_cpu = tenzor::zeros({batch, n, n}, DType::Float64, Device::cpu());
    double* pt = Pt_cpu.data<double>();
    std::vector<int64_t> rowidx(static_cast<size_t>(n));
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < n; ++i) rowidx[i] = i;
        for (int64_t i = 0; i < n; ++i) {
            int64_t j = static_cast<int64_t>(p[b * n + i]) - 1;  // 1-based
            // The pivot vector is untrusted (LU may emit junk on a degenerate
            // factorization); a stray index would swap/scribble out of bounds.
            if (j < 0 || j >= n) {
                throw std::runtime_error(
                    "LU JVP: pivot index out of range [1, " + std::to_string(n) + "]");
            }
            std::swap(rowidx[i], rowidx[j]);
        }
        for (int64_t i = 0; i < n; ++i) {
            pt[(b * n + i) * n + rowidx[i]] = 1.0;
        }
    }
    std::vector<int64_t> pt_shape(shape.begin(), shape.end());  // (..., n, n)
    Tensor Pt = tenzor::reshape(Pt_cpu, {batch, n, n})
                    .to(A.dtype()).to(A.device());
    Pt = tenzor::reshape(Pt, pt_shape);

    Tensor W = tenzor::matmul(Pt, dA);                       // Pᵀ dA
    // M = L⁻¹ W U⁻¹.
    Tensor M = tenzor::linalg::solve_triangular(L, W, /*upper=*/false,
                                                /*unitri=*/true);   // L⁻¹ W
    // (L⁻¹ W) U⁻¹: solve X U = M  =>  Uᵀ Xᵀ = Mᵀ.
    Tensor Ut = tenzor::transpose(U, -1, -2);
    Tensor Mt = tenzor::transpose(M, -1, -2);
    Tensor Xt = tenzor::linalg::solve_triangular(Ut, Mt, /*upper=*/false,
                                                 /*unitri=*/false);
    M = tenzor::transpose(Xt, -1, -2);

    Tensor dL = tenzor::matmul(L, tenzor::tril(M, -1));
    Tensor dU = tenzor::matmul(tenzor::triu(M, 0), U);
    Tensor dPiv = tenzor::zeros(
        std::vector<int64_t>(piv.shape().begin(), piv.shape().end()),
        piv.dtype(), piv.device());

    JvpMultiResult result;
    result.primals  = { L, U, piv };
    result.tangents = { std::move(dL), std::move(dU), std::move(dPiv) };
    return result;
}

// ---------------- Tier E: sequence-level RNN forwards ----------------------
//
// The fused single-layer sequence kernels (LSTMForward / GRUForward) are just
// the per-step cell recurrence unrolled over time. Forward-mode AD composes
// the (already gradchecked) cell JVP at each timestep, threading the hidden /
// cell-state tangents through the recurrence and stacking the per-step output
// tangents. The weight/bias tangents are shared (constant) across steps.

// LSTMForward inputs: {x(T,B,I), W_ih(4H,I), W_hh(4H,H), b_ih, b_hh,
//                      h0(B,H), c0(B,H)}; outputs {y(T,B,H), hT(B,H), cT(B,H)}.
JvpMultiResult jvp_adapter_lstm_forward_s15(std::span<const Tensor> primals,
                                            std::span<const Tensor> tangents,
                                            const OpAttributes&) {
    if (primals.size() != 7) {
        throw std::runtime_error(
            "jvp_adapter_lstm_forward_s15: expected 7 inputs "
            "(x, W_ih, W_hh, b_ih, b_hh, h0, c0)");
    }
    auto td = [&](size_t i) { return i < tangents.size() ? tangents[i] : Tensor(); };
    const Tensor& X = primals[0];
    Tensor Wih = primals[1], Whh = primals[2];
    Tensor bih = primals[3], bhh = primals[4];
    Tensor h0 = primals[5], c0 = primals[6];

    Tensor dX   = jvp_zeros_like_or(X,   td(0));
    Tensor dWih = jvp_zeros_like_or(Wih, td(1));
    Tensor dWhh = jvp_zeros_like_or(Whh, td(2));
    Tensor cur_h  = h0,                       cur_c  = c0;
    Tensor cur_dh = jvp_zeros_like_or(h0, td(5));
    Tensor cur_dc = jvp_zeros_like_or(c0, td(6));

    const int64_t H = h0.shape()[1];
    const int64_t fourH = 4 * H;
    Tensor dbih, dbhh;
    if (bih.numel() == 0) { bih = tenzor::zeros({fourH}, Wih.dtype(), Wih.device());
                            dbih = tenzor::zeros({fourH}, Wih.dtype(), Wih.device()); }
    else                  { dbih = jvp_zeros_like_or(bih, td(3)); }
    if (bhh.numel() == 0) { bhh = tenzor::zeros({fourH}, Wih.dtype(), Wih.device());
                            dbhh = tenzor::zeros({fourH}, Wih.dtype(), Wih.device()); }
    else                  { dbhh = jvp_zeros_like_or(bhh, td(4)); }

    const int64_t T = X.shape()[0];
    const int64_t B = X.shape()[1];
    const int64_t I = X.shape()[2];
    std::vector<Tensor> out_p, out_d;
    out_p.reserve(static_cast<size_t>(T));
    out_d.reserve(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
        Tensor x_t  = X.slice(0, t, t + 1).reshape({B, I}).contiguous();
        Tensor dx_t = dX.slice(0, t, t + 1).reshape({B, I}).contiguous();
        std::array<Tensor, 7> cp = { x_t, cur_h, cur_c, Wih, Whh, bih, bhh };
        std::array<Tensor, 7> ct = { dx_t, cur_dh, cur_dc, dWih, dWhh, dbih, dbhh };
        auto r = jvp_adapter_lstm_cell_forward(cp, ct, OpAttributes{});
        cur_h  = r.primals[0];  cur_c  = r.primals[1];
        cur_dh = r.tangents[0]; cur_dc = r.tangents[1];
        out_p.push_back(cur_h.reshape({1, B, H}));
        out_d.push_back(cur_dh.reshape({1, B, H}));
    }
    JvpMultiResult result;
    result.primals  = { tenzor::cat(out_p, 0), cur_h, cur_c };
    result.tangents = { tenzor::cat(out_d, 0), cur_dh, cur_dc };
    return result;
}

// GRUForward inputs: {x(T,B,I), W_ih(3H,I), W_hh(3H,H), b_ih, h0(B,H), b_hh};
// outputs {y(T,B,H), hT(B,H)}.
JvpMultiResult jvp_adapter_gru_forward_s15(std::span<const Tensor> primals,
                                           std::span<const Tensor> tangents,
                                           const OpAttributes&) {
    if (primals.size() != 6) {
        throw std::runtime_error(
            "jvp_adapter_gru_forward_s15: expected 6 inputs "
            "(x, W_ih, W_hh, b_ih, h0, b_hh)");
    }
    auto td = [&](size_t i) { return i < tangents.size() ? tangents[i] : Tensor(); };
    const Tensor& X = primals[0];
    Tensor Wih = primals[1], Whh = primals[2];
    Tensor bih = primals[3], h0 = primals[4], bhh = primals[5];

    Tensor dX   = jvp_zeros_like_or(X,   td(0));
    Tensor dWih = jvp_zeros_like_or(Wih, td(1));
    Tensor dWhh = jvp_zeros_like_or(Whh, td(2));
    Tensor cur_h  = h0;
    Tensor cur_dh = jvp_zeros_like_or(h0, td(4));

    const int64_t H = h0.shape()[1];
    const int64_t threeH = 3 * H;
    Tensor dbih, dbhh;
    if (bih.numel() == 0) { bih = tenzor::zeros({threeH}, Wih.dtype(), Wih.device());
                            dbih = tenzor::zeros({threeH}, Wih.dtype(), Wih.device()); }
    else                  { dbih = jvp_zeros_like_or(bih, td(3)); }
    if (bhh.numel() == 0) { bhh = tenzor::zeros({threeH}, Wih.dtype(), Wih.device());
                            dbhh = tenzor::zeros({threeH}, Wih.dtype(), Wih.device()); }
    else                  { dbhh = jvp_zeros_like_or(bhh, td(5)); }

    const int64_t T = X.shape()[0];
    const int64_t B = X.shape()[1];
    const int64_t I = X.shape()[2];
    std::vector<Tensor> out_p, out_d;
    out_p.reserve(static_cast<size_t>(T));
    out_d.reserve(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
        Tensor x_t  = X.slice(0, t, t + 1).reshape({B, I}).contiguous();
        Tensor dx_t = dX.slice(0, t, t + 1).reshape({B, I}).contiguous();
        std::array<Tensor, 6> cp = { x_t, cur_h, Wih, Whh, bih, bhh };
        std::array<Tensor, 6> ct = { dx_t, cur_dh, dWih, dWhh, dbih, dbhh };
        auto r = jvp_adapter_gru_cell_forward(cp, ct, OpAttributes{});
        cur_h  = r.primal;
        cur_dh = r.tangent;
        out_p.push_back(cur_h.reshape({1, B, H}));
        out_d.push_back(cur_dh.reshape({1, B, H}));
    }
    JvpMultiResult result;
    result.primals  = { tenzor::cat(out_p, 0), cur_h };
    result.tangents = { tenzor::cat(out_d, 0), cur_dh };
    return result;
}

// LSTMMultiLayerForward inputs: {x, h0(L,B,H), c0(L,B,H), then per layer
// (W_ih, W_hh, bias)} with NumLayers attr; bias is the combined ih+hh term.
// Compose the single-layer sequence JVP layer by layer, feeding each layer's
// output sequence as the next layer's input and stacking the final states.
JvpMultiResult jvp_adapter_lstm_multilayer_forward_s15(std::span<const Tensor> primals,
                                                       std::span<const Tensor> tangents,
                                                       const OpAttributes& attrs) {
    const int64_t L = attrs.get_int(AttrKey::NumLayers, 1);
    if (static_cast<int64_t>(primals.size()) != 3 + 3 * L) {
        throw std::runtime_error("jvp_adapter_lstm_multilayer_forward_s15: bad input count");
    }
    auto td = [&](size_t i) { return i < tangents.size() ? tangents[i] : Tensor(); };
    const Tensor& h0 = primals[1];
    const Tensor& c0 = primals[2];
    const int64_t B = h0.shape()[1];
    const int64_t H = h0.shape()[2];

    Tensor cur_x  = primals[0];
    Tensor cur_dx = jvp_zeros_like_or(primals[0], td(0));
    Tensor dh0 = jvp_zeros_like_or(h0, td(1));
    Tensor dc0 = jvp_zeros_like_or(c0, td(2));

    std::vector<Tensor> hT_p, hT_d, cT_p, cT_d;
    for (int64_t l = 0; l < L; ++l) {
        const size_t base = static_cast<size_t>(3 + l * 3);
        Tensor h0_l  = h0.slice(0, l, l + 1).reshape({B, H}).contiguous();
        Tensor c0_l  = c0.slice(0, l, l + 1).reshape({B, H}).contiguous();
        Tensor dh0_l = dh0.slice(0, l, l + 1).reshape({B, H}).contiguous();
        Tensor dc0_l = dc0.slice(0, l, l + 1).reshape({B, H}).contiguous();
        std::array<Tensor, 7> cp = { cur_x, primals[base], primals[base + 1],
                                     primals[base + 2], Tensor(), h0_l, c0_l };
        std::array<Tensor, 7> ct = { cur_dx, td(base), td(base + 1),
                                     td(base + 2), Tensor(), dh0_l, dc0_l };
        auto r = jvp_adapter_lstm_forward_s15(cp, ct, OpAttributes{});
        cur_x  = r.primals[0];  cur_dx = r.tangents[0];
        hT_p.push_back(r.primals[1].reshape({1, B, H}));
        cT_p.push_back(r.primals[2].reshape({1, B, H}));
        hT_d.push_back(r.tangents[1].reshape({1, B, H}));
        cT_d.push_back(r.tangents[2].reshape({1, B, H}));
    }
    JvpMultiResult result;
    result.primals  = { cur_x,  tenzor::cat(hT_p, 0), tenzor::cat(cT_p, 0) };
    result.tangents = { cur_dx, tenzor::cat(hT_d, 0), tenzor::cat(cT_d, 0) };
    return result;
}

// GRUMultiLayerForward inputs: {x, h0(L,B,H), then per layer (W_ih, W_hh, bias)}.
JvpMultiResult jvp_adapter_gru_multilayer_forward_s15(std::span<const Tensor> primals,
                                                      std::span<const Tensor> tangents,
                                                      const OpAttributes& attrs) {
    const int64_t L = attrs.get_int(AttrKey::NumLayers, 1);
    if (static_cast<int64_t>(primals.size()) != 2 + 3 * L) {
        throw std::runtime_error("jvp_adapter_gru_multilayer_forward_s15: bad input count");
    }
    auto td = [&](size_t i) { return i < tangents.size() ? tangents[i] : Tensor(); };
    const Tensor& h0 = primals[1];
    const int64_t B = h0.shape()[1];
    const int64_t H = h0.shape()[2];

    Tensor cur_x  = primals[0];
    Tensor cur_dx = jvp_zeros_like_or(primals[0], td(0));
    Tensor dh0 = jvp_zeros_like_or(h0, td(1));

    std::vector<Tensor> hT_p, hT_d;
    for (int64_t l = 0; l < L; ++l) {
        const size_t base = static_cast<size_t>(2 + l * 3);
        Tensor h0_l  = h0.slice(0, l, l + 1).reshape({B, H}).contiguous();
        Tensor dh0_l = dh0.slice(0, l, l + 1).reshape({B, H}).contiguous();
        std::array<Tensor, 6> cp = { cur_x, primals[base], primals[base + 1],
                                     primals[base + 2], h0_l, Tensor() };
        std::array<Tensor, 6> ct = { cur_dx, td(base), td(base + 1),
                                     td(base + 2), dh0_l, Tensor() };
        auto r = jvp_adapter_gru_forward_s15(cp, ct, OpAttributes{});
        cur_x  = r.primals[0];  cur_dx = r.tangents[0];
        hT_p.push_back(r.primals[1].reshape({1, B, H}));
        hT_d.push_back(r.tangents[1].reshape({1, B, H}));
    }
    JvpMultiResult result;
    result.primals  = { cur_x,  tenzor::cat(hT_p, 0) };
    result.tangents = { cur_dx, tenzor::cat(hT_d, 0) };
    return result;
}

// BiLSTMForward inputs: {x, h0(2,B,H), c0(2,B,H), W_ih_f, W_hh_f, b_ih_f,
// b_hh_f, W_ih_b, W_hh_b, b_ih_b, b_hh_b}; outputs {y(T,B,2H), hT(2,B,H),
// cT(2,B,H)}. Forward direction is a plain LSTM sequence; backward runs the
// same on the time-reversed input and flips its output back. y concatenates
// the two along the hidden axis.
JvpMultiResult jvp_adapter_bilstm_forward_s15(std::span<const Tensor> primals,
                                              std::span<const Tensor> tangents,
                                              const OpAttributes&) {
    if (primals.size() != 11) {
        throw std::runtime_error("jvp_adapter_bilstm_forward_s15: expected 11 inputs");
    }
    auto td = [&](size_t i) { return i < tangents.size() ? tangents[i] : Tensor(); };
    const Tensor& x  = primals[0];
    const Tensor& h0 = primals[1];
    const Tensor& c0 = primals[2];
    const int64_t B = h0.shape()[1];
    const int64_t H = h0.shape()[2];

    Tensor dx  = jvp_zeros_like_or(x,  td(0));
    Tensor dh0 = jvp_zeros_like_or(h0, td(1));
    Tensor dc0 = jvp_zeros_like_or(c0, td(2));

    auto dir_state = [&](int64_t d, const Tensor& s) {
        return s.slice(0, d, d + 1).reshape({B, H}).contiguous();
    };

    // Forward direction.
    std::array<Tensor, 7> fp = { x, primals[3], primals[4], primals[5], primals[6],
                                 dir_state(0, h0), dir_state(0, c0) };
    std::array<Tensor, 7> ft = { dx, td(3), td(4), td(5), td(6),
                                 dir_state(0, dh0), dir_state(0, dc0) };
    auto rf = jvp_adapter_lstm_forward_s15(fp, ft, OpAttributes{});

    // Backward direction: reverse along time, run, then flip the output back.
    Tensor x_rev  = tenzor::flip(x,  std::vector<int64_t>{0});
    Tensor dx_rev = tenzor::flip(dx, std::vector<int64_t>{0});
    std::array<Tensor, 7> bp = { x_rev, primals[7], primals[8], primals[9], primals[10],
                                 dir_state(1, h0), dir_state(1, c0) };
    std::array<Tensor, 7> bt = { dx_rev, td(7), td(8), td(9), td(10),
                                 dir_state(1, dh0), dir_state(1, dc0) };
    auto rb = jvp_adapter_lstm_forward_s15(bp, bt, OpAttributes{});

    Tensor yb_p = tenzor::flip(rb.primals[0],  std::vector<int64_t>{0});
    Tensor yb_d = tenzor::flip(rb.tangents[0], std::vector<int64_t>{0});

    Tensor y_p = tenzor::cat(std::vector<Tensor>{ rf.primals[0],  yb_p }, 2);
    Tensor y_d = tenzor::cat(std::vector<Tensor>{ rf.tangents[0], yb_d }, 2);
    Tensor hN_p = tenzor::cat(std::vector<Tensor>{ rf.primals[1].reshape({1, B, H}),
                                                   rb.primals[1].reshape({1, B, H}) }, 0);
    Tensor cN_p = tenzor::cat(std::vector<Tensor>{ rf.primals[2].reshape({1, B, H}),
                                                   rb.primals[2].reshape({1, B, H}) }, 0);
    Tensor hN_d = tenzor::cat(std::vector<Tensor>{ rf.tangents[1].reshape({1, B, H}),
                                                   rb.tangents[1].reshape({1, B, H}) }, 0);
    Tensor cN_d = tenzor::cat(std::vector<Tensor>{ rf.tangents[2].reshape({1, B, H}),
                                                   rb.tangents[2].reshape({1, B, H}) }, 0);

    JvpMultiResult result;
    result.primals  = { std::move(y_p), std::move(hN_p), std::move(cN_p) };
    result.tangents = { std::move(y_d), std::move(hN_d), std::move(cN_d) };
    return result;
}

// ============================================================================
// Wave-4 JVP: thin SVD (A = U diag(S) Vh, V = Vhᵀ; U m×k, S k, Vh k×n,
// k = min(m,n)). Forward differential (Townsend 2016 / standard):
//   P  = Uᵀ dA V                                   (k×k)
//   dS = diag(P)
//   F_ij = 1/(s_j² - s_i²) (i≠j, Lorentzian-broadened), 0 on diagonal
//   C  = F ∘ (P diag(S) + diag(S) Pᵀ)              (antisymmetric = Uᵀ dU)
//   D  = F ∘ (diag(S) P + Pᵀ diag(S))              (antisymmetric = Vᵀ dV)
//   dU = U C + (dA V - U P) diag(1/S)              (2nd term: range-complement)
//   dV = V D + (dAᵀ U - V Pᵀ) diag(1/S)
//   dVh = dVᵀ
// Restricted to rank-2 inputs (the SVD kernel's domain for the autograd path).
// ============================================================================
JvpMultiResult jvp_adapter_linalg_svd_s15(std::span<const Tensor> primals,
                                          std::span<const Tensor> tangents,
                                          const OpAttributes& attrs) {
    // Thin-SVD forward differential (Townsend 2016 / standard). Gradcheck-clean
    // for distinct singular values; for repeated singular values the singular
    // vectors are only defined up to rotation within the degenerate subspace
    // and the per-vector tangent is not unique (dS and the reconstruction
    // U dS Vᵀ + ... remain well defined). Restricted to rank-2 inputs (the
    // SVD kernel's autograd domain).
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_linalg_svd_s15: expected 1 input (A)");
    }
    const Tensor& A = primals[0];
    if (A.ndim() != 2) {
        throw NonDifferentiable(
            "LinalgSVD forward-mode JVP is implemented for rank-2 inputs only.");
    }
    Tensor dA = jvp_zeros_like_or(A, tangents.empty() ? Tensor() : tangents[0]);

    auto outs = tenzor::dispatch(OpId::LinalgSVD, std::vector<Tensor>{A}, attrs);
    const Tensor& U  = outs[0];
    const Tensor& S  = outs[1];
    const Tensor& Vh = outs[2];
    const int64_t k = S.shape()[0];

    Tensor V  = tenzor::transpose(Vh, -1, -2);
    Tensor Ut = tenzor::transpose(U, -1, -2);
    Tensor dAV = tenzor::matmul(dA, V);
    Tensor P  = tenzor::matmul(Ut, dAV);
    Tensor Pt = tenzor::transpose(P, -1, -2);

    Tensor dS = tenzor::diag(P);  // diagonal of (k,k) -> (k,)

    Tensor s_row = tenzor::reshape(S, std::vector<int64_t>{k, 1});
    Tensor s_col = tenzor::reshape(S, std::vector<int64_t>{1, k});
    Tensor diff  = tenzor::sub(tenzor::mul(s_col, s_col), tenzor::mul(s_row, s_row));
    const double eps2 = 1e-24;
    Tensor F = tenzor::div(diff, tenzor::add(tenzor::mul(diff, diff), eps2));

    Tensor Sd   = tenzor::linalg::diag_embed(S, 0, -2, -1);
    // M3: regularize Sinv the same way F above is regularized (Lorentzian
    // broadening with the same eps2) instead of an unguarded reciprocal — a
    // zero (or near-zero) singular value otherwise makes Sinv blow up to
    // +Inf here, which can poison dU/dV with NaN even where the true JVP
    // contribution is finite.
    Tensor Sinv = tenzor::linalg::diag_embed(
        tenzor::div(S, tenzor::add(tenzor::mul(S, S), eps2)), 0, -2, -1);

    Tensor C = tenzor::mul(F, tenzor::add(tenzor::matmul(P, Sd), tenzor::matmul(Sd, Pt)));
    Tensor D = tenzor::mul(F, tenzor::add(tenzor::matmul(Sd, P), tenzor::matmul(Pt, Sd)));

    Tensor dU = tenzor::add(tenzor::matmul(U, C),
                            tenzor::matmul(tenzor::sub(dAV, tenzor::matmul(U, P)), Sinv));
    Tensor dAtU = tenzor::matmul(tenzor::transpose(dA, -1, -2), U);
    Tensor dV = tenzor::add(tenzor::matmul(V, D),
                            tenzor::matmul(tenzor::sub(dAtU, tenzor::matmul(V, Pt)), Sinv));
    Tensor dVh = tenzor::transpose(dV, -1, -2);

    JvpMultiResult result;
    result.primals  = { U, S, Vh };
    result.tangents = { std::move(dU), std::move(dS), std::move(dVh) };
    return result;
}

// LinalgEig (general, non-symmetric): A V = V diag(λ); outputs {Re(λ), Im(λ),
// V}. For a REAL spectrum (the differentiable real-arithmetic case) the
// standard differential is, with G = V^{-1} dA V:
//   dλ_i = G_ii
//   C_ji = G_ji / (λ_i - λ_j)   (j ≠ i)
//   C_ii fixed by the unit-norm gauge ‖v_i‖=1  =>  v_iᵀ dv_i = 0:
//          C_ii = -(Σ_{j≠i} C_ji N_ij) / N_ii,   N = Vᵀ V
//   dV = V C
// Complex spectra need complex eigenvector arithmetic the real op output does
// not expose, so this rule fails loudly when any eigenvalue is non-real.
JvpMultiResult jvp_adapter_linalg_eig_s15(std::span<const Tensor> primals,
                                          std::span<const Tensor> tangents,
                                          const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_linalg_eig_s15: expected 1 input (A)");
    }
    const Tensor& A = primals[0];
    if (A.ndim() != 2) {
        throw NonDifferentiable(
            "LinalgEig forward-mode JVP is implemented for rank-2 inputs only.");
    }
    Tensor dA = jvp_zeros_like_or(A, tangents.empty() ? Tensor() : tangents[0]);

    auto outs = tenzor::dispatch(OpId::LinalgEig, std::vector<Tensor>{A}, attrs);
    const Tensor& Re = outs[0];
    const Tensor& Im = outs[1];
    const Tensor& V  = outs[2];
    const int64_t n = Re.shape()[0];

    // Guard: a real-arithmetic forward-mode tangent only exists when every
    // eigenvalue is real (and the eigenvectors V are real).
    {
        Tensor im_cpu = Im.to(Device::cpu()).to(DType::Float64).contiguous();
        const double* im = im_cpu.data<double>();
        double max_im = 0.0;
        for (int64_t i = 0; i < n; ++i) max_im = std::max(max_im, std::abs(im[i]));
        if (max_im > 1e-7) {
            throw NonDifferentiable(
                "LinalgEig forward-mode JVP requires a real spectrum; the matrix "
                "has complex eigenvalues (complex eigenvector arithmetic is not "
                "exposed by the real {Re, Im, V} op output).");
        }
    }

    Tensor lam   = Re;                       // (n,)
    Tensor Vinv  = tenzor::linalg::inv(V);
    Tensor G     = tenzor::matmul(Vinv, tenzor::matmul(dA, V));  // V^{-1} dA V
    Tensor dRe   = tenzor::diag(G);
    Tensor dIm   = tenzor::zeros_like(Im);

    // F[j][i] = 1/(λ_i - λ_j) off-diagonal (Lorentzian-broadened), 0 on diag.
    Tensor lam_row = tenzor::reshape(lam, {n, 1});  // varies over j
    Tensor lam_col = tenzor::reshape(lam, {1, n});  // varies over i
    Tensor diff = tenzor::sub(lam_col, lam_row);    // [j][i] = λ_i - λ_j
    const double eps2 = 1e-24;
    Tensor F = tenzor::div(diff, tenzor::add(tenzor::mul(diff, diff), eps2));
    Tensor C = tenzor::mul(F, G);                   // off-diagonal part

    // Normalization-fixed diagonal: C_ii = -(Σ_j C_ji N_ij)/N_ii, N = Vᵀ V.
    Tensor N = tenzor::matmul(tenzor::transpose(V, -1, -2), V);
    Tensor s = tenzor::sum(tenzor::mul(C, N), 0, /*keepdim=*/false);  // Σ_j C[j][i] N[j][i]
    Tensor Ndiag = tenzor::diag(N);
    Tensor Cii = tenzor::neg(tenzor::div(s, Ndiag));
    C = tenzor::add(C, tenzor::linalg::diag_embed(Cii, 0, -2, -1));

    Tensor dV = tenzor::matmul(V, C);

    JvpMultiResult result;
    result.primals  = { Re, Im, V };
    result.tangents = { std::move(dRe), std::move(dIm), std::move(dV) };
    return result;
}

// Build the full (m×m) orthogonal factor Q = ∏_{i<k}(I − τ_i v_i v_iᵀ) from the
// LAPACK elementary-reflector representation together with its forward tangent.
// v_i is column i of (tril(reflectors,-1) + I) (unit diagonal, zeros above);
// reflectors' strict-upper triangle and diagonal are not part of the reflectors
// (orgqr/ormqr ignore them), so only tril(dR,-1) contributes to dv_i. Q is a
// smooth function of (reflectors, τ) away from degenerate reflectors.
inline std::pair<Tensor, Tensor> householder_q_and_dq(
        const Tensor& reflectors, const Tensor& dR,
        const Tensor& tau, const Tensor& dtau) {
    const int64_t m = reflectors.shape()[reflectors.ndim() - 2];
    const int64_t n = reflectors.shape()[reflectors.ndim() - 1];
    const int64_t k = tau.shape().back();
    const DType dt = reflectors.dtype();
    const Device dev = reflectors.device();

    Tensor Vmat  = tenzor::add(tenzor::tril(reflectors, -1), tenzor::eye(m, n, dt, dev));
    Tensor dVmat = tenzor::tril(dR, -1);

    Tensor Q  = tenzor::eye(m, std::nullopt, dt, dev);
    Tensor dQ = tenzor::zeros({m, m}, dt, dev);
    for (int64_t i = 0; i < k; ++i) {
        Tensor v   = Vmat.slice(1, i, i + 1);    // (m,1)
        Tensor dv  = dVmat.slice(1, i, i + 1);
        Tensor ti  = tau.slice(0, i, i + 1).reshape({1, 1});
        Tensor dti = dtau.slice(0, i, i + 1).reshape({1, 1});
        Tensor vt  = tenzor::transpose(v, -1, -2);
        Tensor dvt = tenzor::transpose(dv, -1, -2);

        Tensor Qv  = tenzor::matmul(Q, v);                            // (m,1)
        Tensor dQv = tenzor::add(tenzor::matmul(dQ, v), tenzor::matmul(Q, dv));
        Tensor QvVt = tenzor::matmul(Qv, vt);                         // (m,m)

        // Q ← Q − τ_i (Q v_i) v_iᵀ
        Tensor Q_new = tenzor::sub(Q, tenzor::mul(ti, QvVt));
        // dQ ← dQ − τ_i(dQv v_iᵀ + Q v_i dv_iᵀ) − dτ_i (Q v_i) v_iᵀ
        Tensor term = tenzor::add(tenzor::matmul(dQv, vt), tenzor::matmul(Qv, dvt));
        Tensor dQ_new = tenzor::sub(dQ, tenzor::mul(ti, term));
        dQ_new = tenzor::sub(dQ_new, tenzor::mul(dti, QvVt));
        Q = std::move(Q_new); dQ = std::move(dQ_new);
    }
    return { std::move(Q), std::move(dQ) };
}

// LinalgHouseholder (orgqr): Y = Q (first n columns). Smooth in (reflectors, τ).
JvpResult jvp_adapter_linalg_householder_s15(std::span<const Tensor> primals,
                                             std::span<const Tensor> tangents,
                                             const OpAttributes&) {
    if (primals.size() != 2) {
        throw std::runtime_error(
            "jvp_adapter_linalg_householder_s15: expected 2 inputs (reflectors, tau)");
    }
    const Tensor& reflectors = primals[0];
    const Tensor& tau = primals[1];
    Tensor dR   = jvp_zeros_like_or(reflectors, tangents.empty()    ? Tensor() : tangents[0]);
    Tensor dtau = jvp_zeros_like_or(tau,        tangents.size() < 2 ? Tensor() : tangents[1]);

    // L3: the primal comes from the real dispatched op (LAPACK orgqr via
    // OpId::LinalgHouseholder), matching every sibling S15 rule that needs a
    // primal (SVD, Eig, LDLFactor, LDLSolve, Geqrf) instead of the naive
    // sequential-reflector reconstruction below. householder_q_and_dq's own
    // Q is discarded — it's only still computed here because dQ's recurrence
    // needs the intermediate Q_i at each step — but dQ itself remains a
    // valid tangent for the dispatched Q: it's an exact differentiation of
    // the SAME mathematical Q(reflectors,tau) LAPACK computes, so the two
    // primals agree up to rounding and dQ carries over unchanged.
    auto [Q_naive, dQ] = householder_q_and_dq(reflectors, dR, tau, dtau);
    (void)Q_naive;
    auto outs = tenzor::dispatch(OpId::LinalgHouseholder, std::vector<Tensor>{reflectors, tau}, OpAttributes{});
    const int64_t n = reflectors.shape()[reflectors.ndim() - 1];
    return JvpResult{ outs[0], dQ.slice(1, 0, n) };
}

// Ormqr: Y = op(Q) B (left) or B op(Q) (right), op = transpose? Qᵀ : Q.
// dY follows the product rule with dQ from the reflector representation.
JvpResult jvp_adapter_ormqr_s15(std::span<const Tensor> primals,
                                std::span<const Tensor> tangents,
                                const OpAttributes& attrs) {
    if (primals.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_ormqr_s15: expected 3 inputs (reflectors, tau, B)");
    }
    const bool left  = attrs.get_int(AttrKey::Left, 1) != 0;
    const bool trans = attrs.get_int(AttrKey::TransposeQ, 0) != 0;
    const Tensor& reflectors = primals[0];
    const Tensor& tau = primals[1];
    const Tensor& B   = primals[2];
    Tensor dR   = jvp_zeros_like_or(reflectors, tangents.empty()    ? Tensor() : tangents[0]);
    Tensor dtau = jvp_zeros_like_or(tau,        tangents.size() < 2 ? Tensor() : tangents[1]);
    Tensor dB   = jvp_zeros_like_or(B,          tangents.size() < 3 ? Tensor() : tangents[2]);

    // L3: get the real op(Q) by dispatching OpId::Ormqr itself against an
    // identity "other" operand (op(Q) @ I = op(Q), or I @ op(Q) = op(Q) on
    // the right) — matching every sibling S15 rule that needs a primal.
    // OpId::LinalgHouseholder (orgqr) isn't usable here: its (m,n) economy
    // output isn't guaranteed to match Ormqr's implicit (order,order) Q when
    // reflectors' column count differs from `order` (Ormqr only requires
    // cols(reflectors) >= k, not == order). Using Ormqr's own kernel with
    // B=I sidesteps that shape mismatch entirely, and folds `trans` in for
    // free (the dispatch attrs already carry TransposeQ). The tangent dQ
    // still comes from the naive sequential-reflector recurrence, which is
    // an exact differentiation of the SAME mathematical Q(reflectors,tau)
    // Ormqr computes, so it remains a valid tangent for the real primal.
    auto [Q_naive, dQ] = householder_q_and_dq(reflectors, dR, tau, dtau);
    (void)Q_naive;
    std::vector<int64_t> B_shape(B.shape().begin(), B.shape().end());
    const int64_t bndim = static_cast<int64_t>(B_shape.size());
    const int64_t order = left ? B_shape[bndim - 2] : B_shape[bndim - 1];
    Tensor I_order = tenzor::eye(order, std::nullopt, B.dtype(), B.device());
    if (bndim > 2) {
        std::vector<int64_t> full_shape(B_shape.begin(), B_shape.end() - 2);
        full_shape.push_back(order);
        full_shape.push_back(order);
        for (int64_t i = 0; i < bndim - 2; ++i) {
            I_order = tenzor::unsqueeze(I_order, 0);
        }
        I_order = tenzor::expand(I_order, full_shape);
    }
    auto ormqr_outs = tenzor::dispatch(OpId::Ormqr,
        std::vector<Tensor>{reflectors, tau, I_order}, attrs);
    Tensor opQ  = ormqr_outs[0];
    Tensor dopQ = trans ? tenzor::transpose(dQ, -1, -2) : dQ;

    Tensor Y, dY;
    if (left) {
        Y  = tenzor::matmul(opQ, B);
        dY = tenzor::add(tenzor::matmul(dopQ, B), tenzor::matmul(opQ, dB));
    } else {
        Y  = tenzor::matmul(B, opQ);
        dY = tenzor::add(tenzor::matmul(dB, opQ), tenzor::matmul(B, dopQ));
    }
    return JvpResult{ std::move(Y), std::move(dY) };
}

// Helper: true iff the LAPACK sytrf pivot vector encodes the trivial case —
// every pivot is a 1×1 block with no interchange (ipiv[i] == i+1, 1-based).
// Interchanges or 2×2 blocks (negative ipiv) make the packed (L,D) layout
// permutation/block dependent, which this rule does not handle.
inline bool ldl_pivots_are_identity(const Tensor& pivots, int64_t n) {
    Tensor pc = pivots.to(Device::cpu()).contiguous();
    const int32_t* p = pc.data<int32_t>();
    for (int64_t i = 0; i < n; ++i) if (p[i] != static_cast<int32_t>(i + 1)) return false;
    return true;
}

// LinalgLDLFactor: symmetric A = L D Lᵀ (sytrf 'L', 1×1 pivots, no interchange).
// sytrf reads only the lower triangle, so the effective input perturbation is
// dA_eff = tril(dA) + tril(dA,-1)ᵀ. With M = L⁻¹ dA_eff L⁻ᵀ:
//   dD = diag(M),  dL = L · (tril(M,-1) diag(1/D))   (strictly lower)
// packed dLD = strict-lower(dL) + diag(dD); pivots carry no tangent.
JvpMultiResult jvp_adapter_linalg_ldl_factor_s15(std::span<const Tensor> primals,
                                                 std::span<const Tensor> tangents,
                                                 const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_linalg_ldl_factor_s15: expected 1 input (A)");
    }
    const Tensor& A = primals[0];
    if (A.ndim() != 2) {
        throw NonDifferentiable("LinalgLDLFactor JVP is implemented for rank-2 inputs only.");
    }
    Tensor dA = jvp_zeros_like_or(A, tangents.empty() ? Tensor() : tangents[0]);

    auto outs = tenzor::dispatch(OpId::LinalgLDLFactor, std::vector<Tensor>{A}, attrs);
    const Tensor& LD = outs[0];
    const Tensor& piv = outs[1];
    const int64_t n = LD.shape().back();
    if (!ldl_pivots_are_identity(piv, n)) {
        throw NonDifferentiable(
            "LinalgLDLFactor JVP supports only the no-interchange, 1x1-pivot case "
            "(diagonally dominant symmetric input); the Bunch-Kaufman pivoting/2x2 "
            "blocking for this matrix is not differentiably reconstructable.");
    }

    Tensor I = tenzor::eye(n, std::nullopt, A.dtype(), A.device());
    Tensor L = tenzor::add(tenzor::tril(LD, -1), I);
    Tensor Dvec = tenzor::diag(LD);                              // (n,)
    // sytrf 'L' factors from the lower triangle (L = tril(LD,-1)+I, D=diag(LD),
    // verified by the storage probe), so the effective symmetric input is built
    // from the lower triangle: dA_eff = tril(dA) + tril(dA,-1)ᵀ.
    Tensor dA_eff = tenzor::add(tenzor::tril(dA, 0),
                                tenzor::transpose(tenzor::tril(dA, -1), -1, -2));
    Tensor Linv = tenzor::linalg::inv(L);
    Tensor M = tenzor::matmul(tenzor::matmul(Linv, dA_eff),
                              tenzor::transpose(Linv, -1, -2));
    Tensor dDvec = tenzor::diag(M);
    Tensor invD = tenzor::reciprocal(Dvec);
    Tensor dL = tenzor::matmul(L, tenzor::mul(tenzor::tril(M, -1),
                                              tenzor::reshape(invD, {1, n})));
    // The op leaves LD's strict-upper triangle equal to the input's strict-upper
    // triangle (unreferenced by sytrf), so that part of the tangent passes
    // through as strict-upper(dA).
    Tensor dLD = tenzor::add(
        tenzor::add(tenzor::tril(dL, -1),
                    tenzor::linalg::diag_embed(dDvec, 0, -2, -1)),
        tenzor::triu(dA, 1));
    Tensor dPiv = tenzor::zeros(
        std::vector<int64_t>(piv.shape().begin(), piv.shape().end()),
        piv.dtype(), piv.device());

    JvpMultiResult result;
    result.primals  = { LD, piv };
    result.tangents = { std::move(dLD), std::move(dPiv) };
    return result;
}

// LinalgLDLSolve: X = A⁻¹ B, A = L D Lᵀ from packed LD (1×1 pivots, no
// interchange). A⁻¹(·) reuses ldl_solve; dA from the packed-factor tangent.
JvpResult jvp_adapter_linalg_ldl_solve_s15(std::span<const Tensor> primals,
                                           std::span<const Tensor> tangents,
                                           const OpAttributes&) {
    if (primals.size() != 3) {
        throw std::runtime_error(
            "jvp_adapter_linalg_ldl_solve_s15: expected 3 inputs (LD, pivots, B)");
    }
    const Tensor& LD = primals[0];
    const Tensor& piv = primals[1];
    const Tensor& B   = primals[2];
    const int64_t n = LD.shape().back();
    if (!ldl_pivots_are_identity(piv, n)) {
        throw NonDifferentiable(
            "LinalgLDLSolve JVP supports only the no-interchange, 1x1-pivot case; "
            "the Bunch-Kaufman pivoting/2x2 blocking for this factor is not "
            "differentiably reconstructable.");
    }
    Tensor dLD = jvp_zeros_like_or(LD, tangents.empty()    ? Tensor() : tangents[0]);
    Tensor dB  = jvp_zeros_like_or(B,  tangents.size() < 3 ? Tensor() : tangents[2]);

    Tensor I = tenzor::eye(n, std::nullopt, LD.dtype(), LD.device());
    Tensor L  = tenzor::add(tenzor::tril(LD, -1), I);
    Tensor dL = tenzor::tril(dLD, -1);
    Tensor D  = tenzor::linalg::diag_embed(tenzor::diag(LD),  0, -2, -1);
    Tensor dD = tenzor::linalg::diag_embed(tenzor::diag(dLD), 0, -2, -1);
    Tensor Lt = tenzor::transpose(L, -1, -2);
    Tensor dLt = tenzor::transpose(dL, -1, -2);
    // dA = dL D Lᵀ + L dD Lᵀ + L D dLᵀ
    Tensor dA = tenzor::add(
        tenzor::matmul(tenzor::matmul(dL, D), Lt),
        tenzor::add(tenzor::matmul(tenzor::matmul(L, dD), Lt),
                    tenzor::matmul(tenzor::matmul(L, D), dLt)));

    Tensor X = tenzor::linalg::ldl_solve(LD, piv, B);
    Tensor rhs = tenzor::sub(dB, tenzor::matmul(dA, X));
    Tensor dX = tenzor::linalg::ldl_solve(LD, piv, rhs);
    return JvpResult{ std::move(X), std::move(dX) };
}

// Geqrf: A → (packed[R above/on diag, reflectors below], tau). The Householder
// QR is inherently sequential (each reflector depends on the previous trailing
// update), so the forward tangent is obtained by dual-propagating the standard
// LAPACK dlarfg recurrence (β=−sign(x₀)‖x‖, τ=(β−x₀)/β, vᵢ=xᵢ/(x₀−β)) on host
// buffers — consistent with geqrf's forward being a host/LAPACK routine and
// with the LU adapter's host-side pivot handling. Primals come from the op;
// only the tangent is computed here. Restricted to rank-2 inputs.
JvpMultiResult jvp_adapter_geqrf_s15(std::span<const Tensor> primals,
                                     std::span<const Tensor> tangents,
                                     const OpAttributes& attrs) {
    if (primals.empty()) {
        throw std::runtime_error("jvp_adapter_geqrf_s15: expected 1 input (A)");
    }
    const Tensor& A = primals[0];
    if (A.ndim() != 2) {
        throw NonDifferentiable("Geqrf forward-mode JVP is implemented for rank-2 inputs only.");
    }
    Tensor dA = jvp_zeros_like_or(A, tangents.empty() ? Tensor() : tangents[0]);

    const int64_t m = A.shape()[0];
    const int64_t n = A.shape()[1];
    const int64_t k = std::min(m, n);

    Tensor Acpu  = A.to(Device::cpu()).to(DType::Float64).contiguous();
    Tensor dAcpu = dA.to(Device::cpu()).to(DType::Float64).contiguous();
    std::vector<double> Ap(Acpu.data<double>(),  Acpu.data<double>()  + m * n);
    std::vector<double> Ad(dAcpu.data<double>(), dAcpu.data<double>() + m * n);
    std::vector<double> dtau(static_cast<size_t>(k), 0.0);

    std::vector<double> v(static_cast<size_t>(m)), dv(static_cast<size_t>(m));
    for (int64_t j = 0; j < k; ++j) {
        double nrm2 = 0.0, dnrm2 = 0.0;
        for (int64_t i = j; i < m; ++i) {
            double a = Ap[i * n + j], da = Ad[i * n + j];
            nrm2 += a * a; dnrm2 += 2.0 * a * da;
        }
        double nrm = std::sqrt(nrm2);
        if (nrm == 0.0) { dtau[j] = 0.0; continue; }
        double dnrm = dnrm2 / (2.0 * nrm);
        double x0 = Ap[j * n + j], dx0 = Ad[j * n + j];
        double s = (x0 >= 0.0) ? 1.0 : -1.0;
        double beta = -s * nrm, dbeta = -s * dnrm;
        double tau_j = (beta - x0) / beta;
        // d[(β−x₀)/β] = (dβ−dx₀)/β − (β−x₀)dβ/β²
        dtau[j] = (dbeta - dx0) / beta - (beta - x0) * dbeta / (beta * beta);
        double denom = x0 - beta, ddenom = dx0 - dbeta;

        for (int64_t i = 0; i < m; ++i) { v[i] = 0.0; dv[i] = 0.0; }
        v[j] = 1.0;
        for (int64_t i = j + 1; i < m; ++i) {
            double a = Ap[i * n + j], da = Ad[i * n + j];
            v[i]  = a / denom;
            dv[i] = (da * denom - a * ddenom) / (denom * denom);
        }
        // Apply H_j = I − τ v vᵀ to trailing columns c > j (rows j..m-1).
        for (int64_t c = j + 1; c < n; ++c) {
            double w = 0.0, dw = 0.0;
            for (int64_t i = j; i < m; ++i) {
                w  += v[i] * Ap[i * n + c];
                dw += dv[i] * Ap[i * n + c] + v[i] * Ad[i * n + c];
            }
            for (int64_t i = j; i < m; ++i) {
                double old = Ap[i * n + c], dold = Ad[i * n + c];
                Ap[i * n + c] = old - tau_j * v[i] * w;
                Ad[i * n + c] = dold - (dtau[j] * v[i] * w + tau_j * dv[i] * w + tau_j * v[i] * dw);
            }
        }
        // Column j: R diagonal = β, reflectors v_i (i>j) below.
        Ap[j * n + j] = beta;  Ad[j * n + j] = dbeta;
        for (int64_t i = j + 1; i < m; ++i) { Ap[i * n + j] = v[i]; Ad[i * n + j] = dv[i]; }
    }

    Tensor packed_tan_cpu(std::vector<int64_t>{m, n}, DType::Float64, Device::cpu());
    std::copy(Ad.begin(), Ad.end(), packed_tan_cpu.data<double>());
    Tensor tau_tan_cpu(std::vector<int64_t>{k}, DType::Float64, Device::cpu());
    std::copy(dtau.begin(), dtau.end(), tau_tan_cpu.data<double>());

    auto outs = tenzor::dispatch(OpId::Geqrf, std::vector<Tensor>{A}, attrs);
    Tensor packed_tan = packed_tan_cpu.to(A.dtype()).to(A.device());
    Tensor tau_tan    = tau_tan_cpu.to(outs[1].dtype()).to(outs[1].device());

    JvpMultiResult result;
    result.primals  = { outs[0], outs[1] };
    result.tangents = { std::move(packed_tan), std::move(tau_tan) };
    return result;
}

// ============================================================================
// Wave-4 JVP: BatchNorm2dForward (single-output kernel; inputs (x, mean, var),
// attr Eps; output y = (x - mean) * rstd, rstd = 1/sqrt(var + eps)). mean/var
// are supplied per-channel (C,) constants — NOT batch statistics derived from x
// — so they are treated as independent inputs. Per-channel quantities broadcast
// as [1, C, 1, 1] for NCHW. JVP:
//   rstd  = rsqrt(var + eps)
//   drstd = -0.5 * rstd^3 * dvar
//   dy    = (dx - dmean) * rstd + (x - mean) * drstd
// ============================================================================
JvpResult jvp_adapter_batch_norm2d_forward_single_s15(std::span<const Tensor> primals,
                                                      std::span<const Tensor> tangents,
                                                      const OpAttributes& attrs) {
    if (primals.size() < 3) {
        throw std::runtime_error(
            "jvp_adapter_batch_norm2d_forward_single_s15: expected 3 inputs (x, mean, var)");
    }
    const Tensor& x    = primals[0];
    const Tensor& mean = primals[1];
    const Tensor& var  = primals[2];
    Tensor dx    = jvp_zeros_like_or(x,    tangents.size() > 0 ? tangents[0] : Tensor());
    Tensor dmean = jvp_zeros_like_or(mean, tangents.size() > 1 ? tangents[1] : Tensor());
    Tensor dvar  = jvp_zeros_like_or(var,  tangents.size() > 2 ? tangents[2] : Tensor());

    double eps = attrs.get_float(AttrKey::Eps, 1e-5);
    const auto xshape = x.shape();
    if (xshape.size() != 4) {
        throw std::runtime_error(
            "jvp_adapter_batch_norm2d_forward_single_s15: expected NCHW (rank-4) input");
    }
    const int64_t C = xshape[1];
    const std::vector<int64_t> c_shape = {1, C, 1, 1};

    auto rstd  = tenzor::rsqrt(tenzor::add(var, eps));
    auto rstd_b  = tenzor::reshape(rstd,  c_shape);
    auto mean_b  = tenzor::reshape(mean,  c_shape);
    auto dmean_b = tenzor::reshape(dmean, c_shape);
    auto dvar_b  = tenzor::reshape(dvar,  c_shape);

    auto x_minus_mean = tenzor::sub(x, mean_b);
    auto y = tenzor::mul(x_minus_mean, rstd_b);

    auto drstd_b = tenzor::mul(tenzor::mul(tenzor::mul(rstd_b, rstd_b), rstd_b),
                               tenzor::mul(dvar_b, -0.5));
    auto dy = tenzor::add(tenzor::mul(tenzor::sub(dx, dmean_b), rstd_b),
                          tenzor::mul(x_minus_mean, drstd_b));

    return JvpResult{ std::move(y), std::move(dy) };
}

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
    register_jvp_rule(OpId::DeviceTransfer, &jvp_adapter_device_transfer);

    // ---------------- Audit A.4 batch 3 ------------------

    // Linear algebra
    register_jvp_rule(OpId::Bmm,             &jvp_adapter_bmm);
    register_jvp_rule(OpId::LinalgInv,       &jvp_adapter_inv);
    register_jvp_rule(OpId::LinalgSolve,     &jvp_adapter_solve);
    register_jvp_rule(OpId::LinalgCholesky,  &jvp_adapter_cholesky);
    register_jvp_rule(OpId::Trace,           &jvp_adapter_trace);
    register_jvp_rule(OpId::LinalgDet,       &jvp_adapter_det);
    register_jvp_rule_multi(OpId::LinalgSlogdet, &jvp_adapter_slogdet);
    register_jvp_rule(OpId::LinalgNorm,      &jvp_adapter_linalg_norm_fro);

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
    register_jvp_rule(OpId::NestedAttention, &jvp_adapter_nondiff_nested_attention_fwd);

    // Losses. CTCLossForward is structurally non-differentiable at this
    // layer (dynamic-programming alpha/beta lattice).
    register_jvp_rule(OpId::FusedSoftmaxCrossEntropy,
                      &jvp_adapter_fused_softmax_cross_entropy);
    register_jvp_rule(OpId::CTCLossForward, &jvp_adapter_nondiff_ctc_loss_forward);

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
    register_jvp_rule(OpId::Quantile,    &jvp_adapter_nondiff_quantile);
    register_jvp_rule(OpId::Nanquantile, &jvp_adapter_nondiff_nanquantile);
    register_jvp_rule(OpId::Nanmedian,   &jvp_adapter_nondiff_nanmedian);

    // Nested softmax / log-softmax: per-segment JVP using the nested
    // kernels plus NestedSum for the per-segment reduction.
    register_jvp_rule(OpId::NestedSoftmax,    &jvp_adapter_nested_softmax);
    register_jvp_rule(OpId::NestedLogSoftmax, &jvp_adapter_nested_log_softmax);
    // NestedLayerNorm: NonDifferentiable (per-segment mean/rstd not exposed).
    register_jvp_rule(OpId::NestedLayerNorm,  &jvp_adapter_nondiff_nested_layer_norm);

    // ---------------- Audit A.4 batch 9 ------------------
    //
    // Linear-in-input pass-through ops (same op on tangent):
    register_jvp_rule(OpId::Conj,             &jvp_adapter_conj);
    register_jvp_rule(OpId::Real,             &jvp_adapter_real);
    register_jvp_rule(OpId::Imag,             &jvp_adapter_imag);
    register_jvp_rule(OpId::ViewAsReal,       &jvp_adapter_view_as_real);
    register_jvp_rule(OpId::ViewAsComplex,    &jvp_adapter_view_as_complex);
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
    register_jvp_rule(OpId::BatchNorm2dForward,                 &jvp_adapter_batch_norm2d_forward_single_s15);
    // S15: replaced NonDifferentiable stub with closed-form chain-rule JVP.
    register_jvp_rule_multi(OpId::BatchNorm2dFusedTraining,      &jvp_adapter_batchnorm2d_fused_training_s15);

    // Multi-output linalg factorisations awaiting bespoke JVPs.
    register_jvp_rule_multi(OpId::LinalgSVD,            &jvp_adapter_linalg_svd_s15);
    register_jvp_rule_multi(OpId::LinalgQR,             &jvp_adapter_linalg_qr_s15);
    register_jvp_rule_multi(OpId::LinalgEig,            &jvp_adapter_linalg_eig_s15);
    register_jvp_rule_multi(OpId::LinalgLU,             &jvp_adapter_linalg_lu_s15);
    register_jvp_rule       (OpId::LinalgHouseholder,    &jvp_adapter_linalg_householder_s15);
    register_jvp_rule_multi(OpId::LinalgLDLFactor,      &jvp_adapter_linalg_ldl_factor_s15);
    register_jvp_rule       (OpId::LinalgLDLSolve,       &jvp_adapter_linalg_ldl_solve_s15);
    // S-final: solve / inverse family — closed-form JVP, gradchecked.
    register_jvp_rule       (OpId::LinalgLUSolve,        &jvp_adapter_lu_solve_s15);
    register_jvp_rule       (OpId::LinalgCholeskySolve,  &jvp_adapter_cholesky_solve_s15);
    register_jvp_rule_multi(OpId::Geqrf,                &jvp_adapter_geqrf_s15);
    register_jvp_rule       (OpId::Ormqr,                &jvp_adapter_ormqr_s15);
    register_jvp_rule       (OpId::TensorInv,            &jvp_adapter_tensor_inv_s15);
    register_jvp_rule       (OpId::TensorSolve,          &jvp_adapter_tensor_solve_s15);
    register_jvp_rule       (OpId::SolveTriangular,      &jvp_adapter_solve_triangular_s15);
    register_jvp_rule       (OpId::CholeskyInverse,      &jvp_adapter_cholesky_inverse_s15);
    register_jvp_rule       (OpId::LOBPCG,               &jvp_adapter_nondiff_lobpcg);

    // Sequence-level RNNs. Single-layer forwards compose the gradchecked cell
    // JVP over the time axis (S-final).
    register_jvp_rule_multi(OpId::LSTMForward,           &jvp_adapter_lstm_forward_s15);
    register_jvp_rule_multi(OpId::GRUForward,            &jvp_adapter_gru_forward_s15);
    register_jvp_rule_multi(OpId::LSTMMultiLayerForward, &jvp_adapter_lstm_multilayer_forward_s15);
    register_jvp_rule_multi(OpId::GRUMultiLayerForward,  &jvp_adapter_gru_multilayer_forward_s15);
    register_jvp_rule_multi(OpId::BiLSTMForward,         &jvp_adapter_bilstm_forward_s15);

    // Search / sort multi-output.
    register_jvp_rule_multi(OpId::TopK, &jvp_adapter_topk_s15);
    register_jvp_rule_multi(OpId::Sort, &jvp_adapter_sort_s15);

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
    // S15: replaced NonDifferentiable stubs with proper JVP rules.
    register_jvp_rule(OpId::DCT,      &jvp_adapter_dct_s15);
    register_jvp_rule(OpId::IDCT,     &jvp_adapter_idct_s15);
    register_jvp_rule(OpId::STFT,     &jvp_adapter_stft_s15);
    register_jvp_rule(OpId::ISTFT,    &jvp_adapter_istft_s15);
    register_jvp_rule(OpId::MelScale, &jvp_adapter_mel_scale_s15);
    register_jvp_rule(OpId::MFCC,     &jvp_adapter_mfcc_s15);

    // S15: replaced NonDifferentiable stubs with closed-form chain-rule JVPs.
    register_jvp_rule(OpId::CDist,             &jvp_adapter_cdist_s15);
    register_jvp_rule(OpId::PairwiseDistance,  &jvp_adapter_pairwise_distance_s15);
    register_jvp_rule(OpId::Pdist,             &jvp_adapter_pdist_s15);
    register_jvp_rule(OpId::CosineSimilarity,  &jvp_adapter_cosine_similarity_s15);
    register_jvp_rule(OpId::Renorm,            &jvp_adapter_renorm_s15);
    register_jvp_rule(OpId::Cov,               &jvp_adapter_cov_s15);
    register_jvp_rule(OpId::Corrcoef,          &jvp_adapter_corrcoef_s15);
    register_jvp_rule(OpId::LinalgVectorNorm,  &jvp_adapter_linalg_vector_norm_s15);
    register_jvp_rule(OpId::LinalgMatrixNorm,  &jvp_adapter_linalg_matrix_norm_s15);

    // Multi-output split/chunk awaiting dual-walker multi-out support.
    register_jvp_rule_multi(OpId::Chunk, &jvp_adapter_chunk_s15);
    register_jvp_rule_multi(OpId::Split, &jvp_adapter_split_s15);

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
    // S15: replaced NonDifferentiable stubs with JVP rules
    // (sparsity pattern held fixed; tangent on values only).
    register_jvp_rule(OpId::SparseToDense,    &jvp_adapter_sparse_to_dense_s15);
    // DenseToSparse cannot carry a meaningful values-tangent through the
    // single-output JVP surface (outs[0] is the integer crow_indices). The
    // former _s15 adapter fabricated a zero tangent of crow_indices shape,
    // silently claiming differentiability while dropping the real values
    // tangent. Register as NonDifferentiable until a multi-output JVP path
    // exists to return the values tangent (gather of dx at nonzero positions).
    register_jvp_rule(OpId::DenseToSparse,    &jvp_adapter_nondiff_dense_to_sparse);
    register_jvp_rule_multi(OpId::SparseSpGEMM, &jvp_adapter_sparse_spgemm_s15);
    register_jvp_rule(OpId::SparseTrsv,       &jvp_adapter_sparse_trsv_s15);
    register_jvp_rule(OpId::SparseTrsm,       &jvp_adapter_sparse_trsm_s15);
    register_jvp_rule(OpId::SparseSoftmax,    &jvp_adapter_sparse_softmax_s15);
    register_jvp_rule(OpId::SparseLogSoftmax, &jvp_adapter_sparse_log_softmax_s15);

    // Normalisation siblings awaiting saved-stats exposure.
    register_jvp_rule_multi(OpId::GroupNorm,    &jvp_adapter_group_norm_s15);
    register_jvp_rule_multi(OpId::InstanceNorm, &jvp_adapter_instance_norm_s15);
    register_jvp_rule_multi(OpId::RMSNorm,      &jvp_adapter_rms_norm_s15);

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
