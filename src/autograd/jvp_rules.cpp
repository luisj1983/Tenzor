#include "tenzor/autograd/jvp_rules.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"

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
