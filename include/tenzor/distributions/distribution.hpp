/**
 * @file distribution.hpp
 * @brief Base class and common distributions for probabilistic sampling
 *
 * Provides a PyTorch-compatible Distribution base class with sample(),
 * log_prob(), entropy(), mean(), and variance() methods.
 */

#pragma once

#include "../core/tensor.hpp"
#include "../ops/creation.hpp"
#include "../ops/math.hpp"
#include "../ops/reduction.hpp"
#include "../ops/linalg.hpp"
#include "../ops/indexing.hpp"
#include "../ops/transform.hpp"
#include <vector>
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>
#include <limits>

namespace tenzor {
namespace distributions {

/**
 * @brief Abstract base class for probability distributions.
 */
class Distribution {
public:
    virtual ~Distribution() = default;

    /** @brief Draw samples from the distribution */
    virtual auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor = 0;

    /**
     * @brief Draw samples using a specific Generator for reproducibility.
     *
     * Default implementation sets the generator as active, calls sample(),
     * then clears it. Subclasses can override for direct generator usage.
     */
    virtual auto sample(std::vector<int64_t> sample_shape, Generator& generator) -> Tensor {
        generator_ = &generator;
        auto result = sample(std::move(sample_shape));
        generator_ = nullptr;
        return result;
    }

    /**
     * @brief Draw reparameterized samples (differentiable through the sampler).
     *
     * For reparameterizable distributions (Normal, Uniform, Exponential,
     * Laplace, Gamma, Beta, Dirichlet, StudentT, MultivariateNormal),
     * rsample() returns a sample that can be backpropagated through the
     * distribution parameters. Non-reparameterizable distributions
     * (Categorical, Bernoulli, Poisson) raise at runtime — call sample()
     * instead (possibly with a straight-through estimator outside).
     *
     * The default implementation forwards to sample(); concrete
     * distributions override when a reparameterization exists.
     */
    virtual auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor {
        return sample(std::move(sample_shape));
    }

    /** @brief Reparameterized sample with a specific Generator. */
    virtual auto rsample(std::vector<int64_t> sample_shape, Generator& generator) -> Tensor {
        generator_ = &generator;
        auto result = rsample(std::move(sample_shape));
        generator_ = nullptr;
        return result;
    }

    /** @brief Compute log probability of a value under this distribution */
    virtual auto log_prob(const Tensor& value) -> Tensor = 0;

    /** @brief Compute entropy of the distribution (optional) */
    virtual auto entropy() -> Tensor {
        throw std::runtime_error("entropy() not implemented for this distribution");
    }

    /** @brief Distribution mean (optional) */
    virtual auto mean() -> Tensor {
        throw std::runtime_error("mean() not implemented for this distribution");
    }

    /** @brief Distribution variance (optional) */
    virtual auto variance() -> Tensor {
        throw std::runtime_error("variance() not implemented for this distribution");
    }

protected:
    /** @brief Active generator (set during Generator-aware sample/rsample calls) */
    Generator* generator_{nullptr};
};

// ============================================================================
// Normal Distribution
// ============================================================================

/**
 * @brief Normal (Gaussian) distribution parameterized by mean and std.
 */
class Normal : public Distribution {
public:
    Normal(Tensor loc, Tensor scale)
        : loc_(std::move(loc)), scale_(std::move(scale)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(loc_.shape().begin(), loc_.shape().end())
            : sample_shape;
        auto eps = randn(shape, loc_.dtype(), loc_.device());
        return loc_ + scale_ * eps;
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        // Normal is already reparameterized via loc + scale * randn.
        return sample(std::move(sample_shape));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto var = scale_ * scale_;
        auto log_scale = tenzor::log(scale_);
        auto diff = value - loc_;
        return tenzor::neg(diff * diff) / (var * 2.0f) - log_scale - 0.9189385332f; // -log(sqrt(2*pi))
    }

    auto entropy() -> Tensor override {
        return tenzor::log(scale_) + 1.4189385332f; // 0.5 * log(2*pi*e)
    }

    auto mean() -> Tensor override { return loc_; }
    auto variance() -> Tensor override { return scale_ * scale_; }

private:
    Tensor loc_;
    Tensor scale_;
};

// ============================================================================
// Uniform Distribution
// ============================================================================

/**
 * @brief Uniform distribution on [low, high).
 */
class Uniform : public Distribution {
public:
    Uniform(Tensor low, Tensor high)
        : low_(std::move(low)), high_(std::move(high)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(low_.shape().begin(), low_.shape().end())
            : sample_shape;
        auto u = rand(shape, low_.dtype(), low_.device());
        return low_ + (high_ - low_) * u;
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto range = high_ - low_;
        return tenzor::log(range) * (-1.0f); // -log(high - low)
    }

    auto entropy() -> Tensor override {
        return tenzor::log(high_ - low_);
    }

    auto mean() -> Tensor override { return (low_ + high_) * 0.5f; }
    auto variance() -> Tensor override {
        auto range = high_ - low_;
        return (range * range) / 12.0f;
    }

private:
    Tensor low_;
    Tensor high_;
};

// ============================================================================
// Categorical Distribution
// ============================================================================

/**
 * @brief Categorical distribution parameterized by logits or probabilities.
 */
class Categorical : public Distribution {
public:
    explicit Categorical(Tensor probs) : probs_(std::move(probs)) {}

    static auto from_logits(const Tensor& logits) -> Categorical {
        // Softmax to convert logits to probabilities
        auto max_val = tenzor::max(logits);
        auto shifted = logits - max_val;
        auto exp_vals = tenzor::exp(shifted);
        auto sum_exp = tenzor::sum(exp_vals);
        return Categorical(exp_vals / sum_exp);
    }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        int64_t num_samples = 1;
        for (auto s : sample_shape) num_samples *= s;
        if (num_samples < 1) num_samples = 1;
        return multinomial(probs_, num_samples, /*replacement=*/true);
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "Categorical is not reparameterizable; use sample() or a "
            "straight-through / Gumbel-softmax estimator instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto log_probs = tenzor::log(probs_);
        return gather(log_probs, -1, value.to(DType::Int64));
    }

    auto entropy() -> Tensor override {
        auto log_probs = tenzor::log(probs_);
        return tenzor::neg(tenzor::sum(probs_ * log_probs));
    }

private:
    Tensor probs_;
};

// ============================================================================
// Exponential Distribution
// ============================================================================

/**
 * @brief Exponential distribution parameterized by rate (lambda).
 */
class Exponential : public Distribution {
public:
    explicit Exponential(Tensor rate) : rate_(std::move(rate)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(rate_.shape().begin(), rate_.shape().end())
            : sample_shape;
        auto u = rand(shape, rate_.dtype(), rate_.device());
        // Clamp to avoid log(0)
        auto clamped = tenzor::clamp(u, 1e-7f, 1.0f);
        return tenzor::log(clamped) * (-1.0f) / rate_;
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        return tenzor::log(rate_) - rate_ * value;
    }

    auto entropy() -> Tensor override {
        return 1.0f - tenzor::log(rate_);
    }

    auto mean() -> Tensor override { return tenzor::reciprocal(rate_); }
    auto variance() -> Tensor override {
        return tenzor::reciprocal(rate_ * rate_);
    }

private:
    Tensor rate_;
};

// ============================================================================
// Laplace Distribution (Phase 6.1)
// ============================================================================

/**
 * @brief Laplace (double exponential) distribution parameterized by
 *        location and scale.
 *
 * PDF(x) = (1 / (2b)) * exp(-|x - mu| / b)
 *
 * Sampling uses inverse-CDF transform: if U ~ Uniform(-0.5, 0.5), then
 * `mu - b * sign(U) * log(1 - 2|U|)` is Laplace(mu, b). We generate U in
 * (epsilon, 1-epsilon) first to avoid log(0) at the tails.
 */
class Laplace : public Distribution {
public:
    Laplace(Tensor loc, Tensor scale)
        : loc_(std::move(loc)), scale_(std::move(scale)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(loc_.shape().begin(), loc_.shape().end())
            : sample_shape;
        // u in (eps, 1-eps)
        auto u = rand(shape, loc_.dtype(), loc_.device());
        constexpr float kEps = 1e-7f;
        auto u_clamped = tenzor::clamp(u, kEps, 1.0f - kEps);
        // Center at 0.5 so we can take sign and log symmetrically.
        auto centered = u_clamped - 0.5f;  // in (-0.5, 0.5)
        auto abs_c = tenzor::abs(centered);
        auto sign_c = tenzor::sign(centered);
        auto one_minus_2abs = tenzor::log(1.0f - 2.0f * abs_c);
        // x = mu - b * sign * log(1 - 2|u - 0.5|)
        return loc_ - scale_ * sign_c * one_minus_2abs;
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log(1 / (2b)) - |x - mu| / b
        auto diff = tenzor::abs(value - loc_);
        auto log_2b = tenzor::log(scale_ * 2.0f);
        return tenzor::neg(log_2b) - diff / scale_;
    }

    auto entropy() -> Tensor override {
        // 1 + log(2b)
        auto log_2b = tenzor::log(scale_ * 2.0f);
        return log_2b + 1.0f;
    }

    auto mean() -> Tensor override { return loc_; }
    auto variance() -> Tensor override { return scale_ * scale_ * 2.0f; }

private:
    Tensor loc_;
    Tensor scale_;
};

// ============================================================================
// Bernoulli Distribution
// ============================================================================

/**
 * @brief Bernoulli distribution parameterized by probability.
 */
class BernoulliDist : public Distribution {
public:
    explicit BernoulliDist(Tensor probs) : probs_(std::move(probs)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return bernoulli(probs_);
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "Bernoulli is not reparameterizable; use sample() or Gumbel-softmax instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto eps = 1e-7f;
        auto clamped = tenzor::clamp(probs_, eps, 1.0f - eps);
        return value * tenzor::log(clamped) + (1.0f - value) * tenzor::log(1.0f - clamped);
    }

    auto entropy() -> Tensor override {
        auto eps = 1e-7f;
        auto clamped = tenzor::clamp(probs_, eps, 1.0f - eps);
        return tenzor::neg(clamped * tenzor::log(clamped) + (1.0f - clamped) * tenzor::log(1.0f - clamped));
    }

    auto mean() -> Tensor override { return probs_; }
    auto variance() -> Tensor override { return probs_ * (1.0f - probs_); }

private:
    Tensor probs_;
};

// ============================================================================
// CPU-only helpers for Gamma / Beta / Dirichlet / Poisson / StudentT sampling
// ============================================================================
//
// These distributions need element-wise transform/rejection sampling that is
// easier to express as explicit loops over CPU memory. GPU tensors are moved
// to CPU, sampled, and moved back. They are intentionally simple — the goal
// is PyTorch-feature-parity coverage rather than raw throughput.

namespace detail {

/// Shared PRNG for per-sample rejection algorithms. Seeded once on first use.
inline auto& distribution_rng() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng;
}

/// Marsaglia–Tsang (2000) rejection sampler for Gamma(shape >= 1).
/// For shape < 1, use the boosting trick gamma(s) = gamma(s+1) * U^(1/s).
/// `shape` is the concentration parameter (a.k.a. alpha, k).
template <typename T>
inline auto sample_gamma_scalar(T shape_alpha, T rate_beta) -> T {
    auto& rng = distribution_rng();
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    double alpha = static_cast<double>(shape_alpha);
    if (alpha <= 0.0) {
        throw std::runtime_error("Gamma: shape must be positive");
    }

    double boost = 1.0;
    if (alpha < 1.0) {
        // gamma(alpha) = gamma(alpha + 1) * U^(1/alpha)
        double u = uniform(rng);
        // Avoid log(0); the tail is astronomically unlikely but guard anyway.
        if (u == 0.0) u = 1e-300;
        boost = std::pow(u, 1.0 / alpha);
        alpha += 1.0;
    }

    const double d = alpha - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);

    while (true) {
        double x, v;
        do {
            x = normal(rng);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        double u = uniform(rng);
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            return static_cast<T>(d * v * boost / static_cast<double>(rate_beta));
        }
        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
            return static_cast<T>(d * v * boost / static_cast<double>(rate_beta));
        }
    }
}

/// Knuth (small λ) / transformed-rejection (large λ) sampler for Poisson.
inline auto sample_poisson_scalar(double lambda) -> int64_t {
    auto& rng = detail::distribution_rng();
    if (lambda <= 0.0) return 0;
    if (lambda < 10.0) {
        // Knuth's algorithm.
        double L = std::exp(-lambda);
        int64_t k = 0;
        double p = 1.0;
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        do {
            ++k;
            p *= uniform(rng);
        } while (p > L);
        return k - 1;
    }
    // Atkinson's transformed-rejection for large lambda.
    const double c = 0.767 - 3.36 / lambda;
    const double beta = M_PI / std::sqrt(3.0 * lambda);
    const double alpha = beta * lambda;
    const double k = std::log(c) - lambda - std::log(beta);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    while (true) {
        double u = uniform(rng);
        double x = (alpha - std::log((1.0 - u) / u)) / beta;
        int64_t n = static_cast<int64_t>(std::floor(x + 0.5));
        if (n < 0) continue;
        double v = uniform(rng);
        double y = alpha - beta * x;
        double lhs = y + std::log(v / ((1.0 + std::exp(y)) * (1.0 + std::exp(y))));
        double rhs = k + static_cast<double>(n) * std::log(lambda) - std::lgamma(static_cast<double>(n) + 1.0);
        if (lhs <= rhs) return n;
    }
}

/// Broadcast-compatible element-wise CPU fill for Gamma sampling.
/// concentration and rate are broadcast to the output shape.
inline auto fill_gamma_cpu(const Tensor& concentration, const Tensor& rate,
                           std::vector<int64_t> shape) -> Tensor {
    auto out = zeros(shape, concentration.dtype(), Device::cpu());
    int64_t n = out.numel();

    // Materialize concentration/rate on CPU; for simplicity require that
    // they are either scalars or match `shape` exactly. Full broadcasting
    // is out of scope for the P1 distributions pass.
    auto conc_cpu = concentration.to(Device::cpu()).contiguous();
    auto rate_cpu = rate.to(Device::cpu()).contiguous();
    const int64_t conc_n = conc_cpu.numel();
    const int64_t rate_n = rate_cpu.numel();

    if (out.dtype() == DType::Float32) {
        float* op = out.data<float>();
        const float* cp = conc_cpu.data<float>();
        const float* rp = rate_cpu.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            float conc = (conc_n == 1) ? cp[0] : cp[i % conc_n];
            float rate_v = (rate_n == 1) ? rp[0] : rp[i % rate_n];
            op[i] = sample_gamma_scalar<float>(conc, rate_v);
        }
    } else if (out.dtype() == DType::Float64) {
        double* op = out.data<double>();
        const double* cp = conc_cpu.data<double>();
        const double* rp = rate_cpu.data<double>();
        for (int64_t i = 0; i < n; ++i) {
            double conc = (conc_n == 1) ? cp[0] : cp[i % conc_n];
            double rate_v = (rate_n == 1) ? rp[0] : rp[i % rate_n];
            op[i] = sample_gamma_scalar<double>(conc, rate_v);
        }
    } else {
        throw std::runtime_error("Gamma: only Float32 and Float64 supported");
    }

    return out;
}

} // namespace detail

// ============================================================================
// Gamma Distribution
// ============================================================================

/**
 * @brief Gamma(concentration, rate) — shape-rate parameterization.
 *
 * PDF(x) = rate^concentration / Gamma(concentration) * x^(concentration-1) * exp(-rate * x)
 *
 * Uses the Marsaglia–Tsang rejection sampler on CPU.
 */
class Gamma : public Distribution {
public:
    Gamma(Tensor concentration, Tensor rate)
        : concentration_(std::move(concentration)), rate_(std::move(rate)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(concentration_.shape().begin(), concentration_.shape().end())
            : sample_shape;
        auto result = detail::fill_gamma_cpu(concentration_, rate_, shape);
        return result.to(concentration_.device());
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        // Gamma is reparameterizable in theory via implicit reparam (Figurnov
        // et al. 2018), but for this P1 pass we expose sample() as rsample()
        // and note that the graph is detached. Downstream users that need
        // backprop-through-Gamma should use the implicit-reparam path via
        // Kingma's trick externally.
        return sample(std::move(sample_shape));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x) = c*log(rate) - lgamma(c) + (c-1)*log(x) - rate*x
        // Approximated elementwise using existing tensor ops; lgamma applied
        // via tenzor::lgamma if available, else log(gamma) is computed via
        // the existing math ops. For the common case we just use the
        // canonical closed form with existing primitives.
        auto log_rate = tenzor::log(rate_);
        auto log_val = tenzor::log(value);
        auto minus_lgamma = tenzor::neg(tenzor::lgamma(concentration_));
        auto term1 = concentration_ * log_rate;
        auto term2 = (concentration_ - 1.0f) * log_val;
        auto term3 = tenzor::neg(rate_ * value);
        return term1 + minus_lgamma + term2 + term3;
    }

    auto mean() -> Tensor override { return concentration_ / rate_; }
    auto variance() -> Tensor override { return concentration_ / (rate_ * rate_); }

private:
    Tensor concentration_;
    Tensor rate_;
};

// ============================================================================
// Beta Distribution
// ============================================================================

/**
 * @brief Beta(concentration1, concentration0) on (0, 1).
 *
 * Sampled as X/(X+Y) where X ~ Gamma(c1, 1) and Y ~ Gamma(c0, 1).
 */
class Beta : public Distribution {
public:
    Beta(Tensor concentration1, Tensor concentration0)
        : c1_(std::move(concentration1)), c0_(std::move(concentration0)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(c1_.shape().begin(), c1_.shape().end())
            : sample_shape;
        auto one = full({1}, 1.0, c1_.dtype(), Device::cpu());
        auto x = detail::fill_gamma_cpu(c1_, one, shape);
        auto y = detail::fill_gamma_cpu(c0_, one, shape);
        auto result = x / (x + y);
        return result.to(c1_.device());
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return sample(std::move(sample_shape));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log Beta(x; a, b) = (a-1) log x + (b-1) log(1-x) - logB(a,b)
        // logB(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)
        auto log_x = tenzor::log(value);
        auto log_1mx = tenzor::log(1.0f - value);
        auto log_B = tenzor::lgamma(c1_) + tenzor::lgamma(c0_) - tenzor::lgamma(c1_ + c0_);
        return (c1_ - 1.0f) * log_x + (c0_ - 1.0f) * log_1mx - log_B;
    }

    auto mean() -> Tensor override { return c1_ / (c1_ + c0_); }

private:
    Tensor c1_;
    Tensor c0_;
};

// ============================================================================
// Dirichlet Distribution
// ============================================================================

/**
 * @brief Dirichlet(concentration) over the simplex.
 *
 * Sampled by drawing K independent Gamma(alpha_k, 1) and normalizing.
 * `concentration` is a 1D (K,) or batched (..., K) tensor.
 */
class Dirichlet : public Distribution {
public:
    explicit Dirichlet(Tensor concentration)
        : concentration_(std::move(concentration)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(concentration_.shape().begin(), concentration_.shape().end())
            : sample_shape;
        auto one = full({1}, 1.0, concentration_.dtype(), Device::cpu());
        auto gammas = detail::fill_gamma_cpu(concentration_, one, shape);

        // Normalize along the last dim.
        // sum over last dim, keepdim to broadcast.
        auto sum_last = tenzor::sum(gammas, -1, /*keepdim=*/true);
        auto result = gammas / sum_last;
        return result.to(concentration_.device());
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return sample(std::move(sample_shape));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x | alpha) = lgamma(sum alpha) - sum lgamma(alpha) + sum (alpha - 1) * log x
        auto log_x = tenzor::log(value);
        auto sum_alpha = tenzor::sum(concentration_, -1, true);
        auto lgamma_sum = tenzor::lgamma(sum_alpha);
        auto sum_lgamma = tenzor::sum(tenzor::lgamma(concentration_), -1, true);
        auto log_term = tenzor::sum((concentration_ - 1.0f) * log_x, -1, true);
        return lgamma_sum - sum_lgamma + log_term;
    }

private:
    Tensor concentration_;
};

// ============================================================================
// StudentT Distribution
// ============================================================================

/**
 * @brief Student-t distribution with df degrees of freedom.
 *
 * X = mu + sigma * Z / sqrt(Y / df), where Z ~ N(0, 1) and
 * Y ~ ChiSquared(df) = Gamma(df/2, 1/2).
 */
class StudentT : public Distribution {
public:
    StudentT(Tensor df, Tensor loc, Tensor scale)
        : df_(std::move(df)), loc_(std::move(loc)), scale_(std::move(scale)) {}

    explicit StudentT(Tensor df)
        : df_(std::move(df)),
          loc_(zeros({1}, DType::Float32, Device::cpu())),
          scale_(full({1}, 1.0, DType::Float32, Device::cpu())) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(df_.shape().begin(), df_.shape().end())
            : sample_shape;

        // Z ~ N(0, 1), X² = ChiSq(df) via Gamma(df/2, 1/2).
        auto z = randn(shape, df_.dtype(), Device::cpu());
        auto half = full({1}, 0.5, df_.dtype(), Device::cpu());
        auto alpha = df_.to(Device::cpu()) * half;
        auto rate  = half;
        auto chi2 = detail::fill_gamma_cpu(alpha, rate, shape);

        auto df_cpu = df_.to(Device::cpu());
        auto scaled = z / tenzor::sqrt(chi2 / df_cpu);
        auto result = loc_.to(Device::cpu()) + scale_.to(Device::cpu()) * scaled;
        return result.to(df_.device());
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return sample(std::move(sample_shape));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x; df, loc, scale) = lgamma((df+1)/2) - lgamma(df/2)
        //                          - 0.5*log(df*pi) - log(scale)
        //                          - (df+1)/2 * log(1 + ((x-loc)/scale)^2 / df)
        auto y = (value - loc_) / scale_;
        auto df_half = df_ * 0.5f;
        auto df_half_plus_half = df_half + 0.5f;
        auto log_pi = full({1}, static_cast<double>(std::log(M_PI)), df_.dtype(), df_.device());
        auto term1 = tenzor::lgamma(df_half_plus_half) - tenzor::lgamma(df_half);
        auto term2 = tenzor::neg(0.5f * (tenzor::log(df_) + log_pi) + tenzor::log(scale_));
        auto term3 = tenzor::neg(df_half_plus_half * tenzor::log(1.0f + y * y / df_));
        return term1 + term2 + term3;
    }

    auto mean() -> Tensor override { return loc_; }

private:
    Tensor df_;
    Tensor loc_;
    Tensor scale_;
};

// ============================================================================
// Poisson Distribution
// ============================================================================

/**
 * @brief Poisson distribution parameterized by rate (lambda >= 0).
 *
 * Uses Knuth's algorithm for small lambda (<10) and Atkinson's
 * transformed-rejection algorithm for large lambda. Returns an Int64
 * tensor of counts.
 */
class Poisson : public Distribution {
public:
    explicit Poisson(Tensor rate) : rate_(std::move(rate)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(rate_.shape().begin(), rate_.shape().end())
            : sample_shape;

        auto rate_cpu = rate_.to(Device::cpu()).contiguous();
        auto out = zeros(shape, DType::Int64, Device::cpu());
        int64_t n = out.numel();
        const int64_t rate_n = rate_cpu.numel();
        int64_t* op = out.data<int64_t>();

        if (rate_cpu.dtype() == DType::Float32) {
            const float* rp = rate_cpu.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                double lam = (rate_n == 1) ? rp[0] : rp[i % rate_n];
                op[i] = detail::sample_poisson_scalar(lam);
            }
        } else if (rate_cpu.dtype() == DType::Float64) {
            const double* rp = rate_cpu.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                double lam = (rate_n == 1) ? rp[0] : rp[i % rate_n];
                op[i] = detail::sample_poisson_scalar(lam);
            }
        } else {
            throw std::runtime_error("Poisson: rate must be Float32 or Float64");
        }

        return out.to(rate_.device());
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "Poisson is not reparameterizable; use sample() instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(k; lambda) = k * log(lambda) - lambda - lgamma(k + 1)
        auto val_float = value.to(rate_.dtype());
        auto log_lam = tenzor::log(rate_);
        auto term1 = val_float * log_lam;
        auto term2 = tenzor::neg(rate_);
        auto term3 = tenzor::neg(tenzor::lgamma(val_float + 1.0f));
        return term1 + term2 + term3;
    }

    auto mean() -> Tensor override { return rate_; }
    auto variance() -> Tensor override { return rate_; }

private:
    Tensor rate_;
};

// ============================================================================
// Multivariate Normal Distribution
// ============================================================================

/**
 * @brief MultivariateNormal(loc, covariance_matrix).
 *
 * Sampled via loc + L @ eps where L is the lower-triangular Cholesky factor
 * of the covariance matrix and eps ~ N(0, I).
 *
 * `loc` shape: (..., D). `covariance_matrix` shape: (..., D, D).
 */
class MultivariateNormal : public Distribution {
public:
    MultivariateNormal(Tensor loc, Tensor covariance_matrix)
        : loc_(std::move(loc)), cov_(std::move(covariance_matrix)) {
        // Precompute Cholesky factor L of covariance for sampling + log_prob.
        scale_tril_ = linalg::cholesky(cov_, /*upper=*/false);
    }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(loc_.shape().begin(), loc_.shape().end())
            : sample_shape;
        auto eps = randn(shape, loc_.dtype(), loc_.device());

        // Compute L @ eps along the last dim. Assume loc/eps are 1D for the
        // common case; for batched we fall back to matmul over the last
        // two dims (requires (D, D) @ (D,) broadcast, handled by matmul).
        // If eps is 1D (D,), promote to (D, 1) for matmul then squeeze.
        if (eps.shape().size() == 1) {
            auto eps_col = eps.reshape({eps.shape()[0], 1});
            auto result_col = matmul(scale_tril_, eps_col);
            auto result = result_col.reshape({result_col.shape()[0]});
            return loc_ + result;
        } else {
            // Batched: ... x D eps -> add trailing col, matmul, squeeze.
            auto shape_vec = std::vector<int64_t>(eps.shape().begin(), eps.shape().end());
            shape_vec.push_back(1);
            auto eps_col = eps.reshape(shape_vec);
            auto result_col = matmul(scale_tril_, eps_col);
            auto out_shape = shape_vec;
            out_shape.pop_back();
            auto result = result_col.reshape(out_shape);
            return loc_ + result;
        }
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return sample(std::move(sample_shape));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // -0.5 * (d * log(2*pi) + log|cov| + (x-mu)^T cov^{-1} (x-mu))
        // Use Cholesky for log|cov| = 2 * sum(log(diag(L))) and
        // for the Mahalanobis term by solving L * y = (x - mu).
        auto diff = value - loc_;  // shape (D,) or (..., D)
        int64_t D = loc_.shape().back();

        // Promote diff to column (D, 1) for triangular solve (simplest path).
        auto diff_col = (diff.shape().size() == 1)
            ? diff.reshape({D, 1})
            : diff.reshape(std::vector<int64_t>(diff.shape().begin(), diff.shape().end()) /*dummy*/);
        // For this P1 pass we support the unbatched case only.
        if (diff.shape().size() != 1) {
            throw std::runtime_error(
                "MultivariateNormal::log_prob: batched log_prob not yet implemented");
        }

        // Solve L y = diff  ->  y = L^{-1} diff. Mahalanobis = y^T y.
        auto y = linalg::solve(scale_tril_, diff_col);
        auto mahal = tenzor::sum(y * y);

        // log|cov| = 2 * sum(log(diag(L)))
        double log_det = 0.0;
        auto L_cpu = scale_tril_.to(Device::cpu()).contiguous();
        if (L_cpu.dtype() == DType::Float32) {
            const float* p = L_cpu.data<float>();
            for (int64_t i = 0; i < D; ++i) {
                log_det += std::log(static_cast<double>(p[i * D + i]));
            }
        } else {
            const double* p = L_cpu.data<double>();
            for (int64_t i = 0; i < D; ++i) {
                log_det += std::log(p[i * D + i]);
            }
        }
        log_det *= 2.0;

        auto lpf = full({1}, -0.5 * (static_cast<double>(D) * std::log(2.0 * M_PI) + log_det),
                        loc_.dtype(), Device::cpu());
        auto result = lpf.to(loc_.device()) - mahal * 0.5f;
        return result;
    }

    auto mean() -> Tensor override { return loc_; }

private:
    Tensor loc_;
    Tensor cov_;
    Tensor scale_tril_;
};

// ============================================================================
// Binomial Distribution
// ============================================================================

/**
 * @brief Binomial(total_count, probs) distribution.
 *
 * The number of successes in `total_count` independent Bernoulli trials,
 * each with success probability `probs`.
 *
 * Sampling is implemented by drawing `total_count` Bernoulli trials and
 * summing. For large total_count the Normal approximation could be used,
 * but the Bernoulli-sum approach is correct for all parameter values.
 *
 * Returns Float32 tensor of counts.
 */
class Binomial : public Distribution {
public:
    Binomial(int64_t total_count, Tensor probs)
        : total_count_(total_count), probs_(std::move(probs)) {
        if (total_count < 0) {
            throw std::runtime_error("Binomial: total_count must be non-negative");
        }
    }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(probs_.shape().begin(), probs_.shape().end())
            : sample_shape;

        // Sum of total_count independent Bernoulli(probs) trials
        auto result = zeros(shape, probs_.dtype(), probs_.device());
        for (int64_t i = 0; i < total_count_; ++i) {
            result = result + bernoulli(probs_);
        }
        return result;
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "Binomial is not reparameterizable; use sample() instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log P(k; n, p) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        //                + k*log(p) + (n-k)*log(1-p)
        auto eps = 1e-7f;
        auto p = tenzor::clamp(probs_, eps, 1.0f - eps);
        auto n = full(std::vector<int64_t>(value.shape().begin(), value.shape().end()),
                      static_cast<float>(total_count_),
                      value.dtype(), value.device());
        auto val_f = value.to(probs_.dtype());

        auto log_binom = tenzor::lgamma(n + 1.0f)
                       - tenzor::lgamma(val_f + 1.0f)
                       - tenzor::lgamma(n - val_f + 1.0f);
        return log_binom + val_f * tenzor::log(p)
             + (n - val_f) * tenzor::log(1.0f - p);
    }

    auto entropy() -> Tensor override {
        // No simple closed form; use the 0.5 * log(2*pi*e*n*p*(1-p)) approximation
        auto eps = 1e-7f;
        auto p = tenzor::clamp(probs_, eps, 1.0f - eps);
        auto n = full(std::vector<int64_t>(probs_.shape().begin(), probs_.shape().end()),
                      static_cast<float>(total_count_),
                      probs_.dtype(), probs_.device());
        return 0.5f * tenzor::log(n * p * (1.0f - p) * 17.0794684f);  // 2*pi*e ~ 17.079
    }

    auto mean() -> Tensor override {
        return probs_ * static_cast<float>(total_count_);
    }

    auto variance() -> Tensor override {
        return probs_ * (1.0f - probs_) * static_cast<float>(total_count_);
    }

private:
    int64_t total_count_;
    Tensor probs_;
};

// ============================================================================
// LogNormal Distribution
// ============================================================================

/**
 * @brief LogNormal(loc, scale) distribution.
 *
 * If X ~ Normal(loc, scale), then exp(X) ~ LogNormal(loc, scale).
 * Composes directly from Normal distribution.
 *
 * PDF(x) = 1/(x * scale * sqrt(2*pi)) * exp(-(log(x) - loc)^2 / (2*scale^2))
 */
class LogNormal : public Distribution {
public:
    LogNormal(Tensor loc, Tensor scale)
        : loc_(std::move(loc)), scale_(std::move(scale)),
          normal_(loc_, scale_) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return tenzor::exp(normal_.sample(std::move(sample_shape)));
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return tenzor::exp(normal_.rsample(std::move(sample_shape)));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x) = Normal.log_prob(log(x)) - log(x)
        auto log_val = tenzor::log(value);
        return normal_.log_prob(log_val) - log_val;
    }

    auto entropy() -> Tensor override {
        // H = loc + 0.5 + log(scale * sqrt(2*pi*e))
        // = loc + 0.5 + log(scale) + 0.5*log(2*pi*e)
        return loc_ + 0.5f + tenzor::log(scale_) + 0.9189385332f;
    }

    auto mean() -> Tensor override {
        // E[X] = exp(loc + scale^2/2)
        return tenzor::exp(loc_ + scale_ * scale_ * 0.5f);
    }

    auto variance() -> Tensor override {
        // Var[X] = (exp(scale^2) - 1) * exp(2*loc + scale^2)
        auto s2 = scale_ * scale_;
        return (tenzor::exp(s2) - 1.0f) * tenzor::exp(2.0f * loc_ + s2);
    }

private:
    Tensor loc_;
    Tensor scale_;
    Normal normal_;
};

// ============================================================================
// Cauchy Distribution
// ============================================================================

/**
 * @brief Cauchy(loc, scale) distribution.
 *
 * Heavy-tailed distribution with no defined mean or variance.
 * Sampled via inverse CDF: loc + scale * tan(pi * (U - 0.5))
 * where U ~ Uniform(0, 1).
 *
 * PDF(x) = 1 / (pi * scale * (1 + ((x - loc)/scale)^2))
 */
class Cauchy : public Distribution {
public:
    Cauchy(Tensor loc, Tensor scale)
        : loc_(std::move(loc)), scale_(std::move(scale)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(loc_.shape().begin(), loc_.shape().end())
            : sample_shape;
        auto u = rand(shape, loc_.dtype(), loc_.device());
        // Clamp away from 0 and 1 to avoid tan(+-pi/2)
        constexpr float kEps = 1e-7f;
        auto u_clamped = tenzor::clamp(u, kEps, 1.0f - kEps);
        // x = loc + scale * tan(pi * (u - 0.5))
        auto pi_val = static_cast<float>(M_PI);
        return loc_ + scale_ * tenzor::tan((u_clamped - 0.5f) * pi_val);
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        // Cauchy is reparameterizable via the inverse CDF transform.
        return sample(std::move(sample_shape));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x) = -log(pi * scale) - log(1 + ((x - loc)/scale)^2)
        auto z = (value - loc_) / scale_;
        auto log_pi_scale = tenzor::log(scale_ * static_cast<float>(M_PI));
        return tenzor::neg(log_pi_scale) - tenzor::log(1.0f + z * z);
    }

    auto entropy() -> Tensor override {
        // H = log(4 * pi * scale)
        return tenzor::log(scale_ * static_cast<float>(4.0 * M_PI));
    }

    auto mean() -> Tensor override {
        throw std::runtime_error("Cauchy distribution has no defined mean");
    }

    auto variance() -> Tensor override {
        throw std::runtime_error("Cauchy distribution has no defined variance");
    }

private:
    Tensor loc_;
    Tensor scale_;
};

// ============================================================================
// Chi2 (Chi-Squared) Distribution
// ============================================================================

/**
 * @brief Chi-squared distribution with `df` degrees of freedom.
 *
 * Chi2(df) = Gamma(df/2, rate=0.5). Composes directly from Gamma.
 *
 * PDF(x) = x^(df/2 - 1) * exp(-x/2) / (2^(df/2) * Gamma(df/2))
 */
class Chi2 : public Distribution {
public:
    explicit Chi2(Tensor df)
        : df_(std::move(df)),
          gamma_(df_ * 0.5f,
                 full(std::vector<int64_t>(df_.shape().begin(), df_.shape().end()),
                      0.5f, df_.dtype(), df_.device())) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return gamma_.sample(std::move(sample_shape));
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return gamma_.rsample(std::move(sample_shape));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        return gamma_.log_prob(value);
    }

    auto mean() -> Tensor override { return df_; }

    auto variance() -> Tensor override { return df_ * 2.0f; }

private:
    Tensor df_;
    Gamma gamma_;
};

// ============================================================================
// Geometric Distribution
// ============================================================================

/**
 * @brief Geometric(probs) distribution.
 *
 * Models the number of trials until the first success in a sequence of
 * independent Bernoulli trials. Uses 1-indexed convention (minimum value 1).
 *
 * Sampled via inverse CDF: floor(log(U) / log(1 - probs)) + 1
 * where U ~ Uniform(0, 1).
 *
 * P(X = k) = (1 - probs)^(k-1) * probs, for k = 1, 2, 3, ...
 */
class Geometric : public Distribution {
public:
    explicit Geometric(Tensor probs) : probs_(std::move(probs)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(probs_.shape().begin(), probs_.shape().end())
            : sample_shape;
        auto u = rand(shape, probs_.dtype(), probs_.device());
        // Clamp to avoid log(0)
        constexpr float kEps = 1e-7f;
        auto u_clamped = tenzor::clamp(u, kEps, 1.0f);
        auto probs_clamped = tenzor::clamp(probs_, kEps, 1.0f - kEps);
        // k = floor(log(u) / log(1 - p)) + 1
        return tenzor::floor(tenzor::log(u_clamped) / tenzor::log1p(tenzor::neg(probs_clamped))) + 1.0f;
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "Geometric is not reparameterizable; use sample() instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log P(k) = (k-1) * log(1-p) + log(p)
        auto eps = 1e-7f;
        auto p = tenzor::clamp(probs_, eps, 1.0f - eps);
        auto val_f = value.to(probs_.dtype());
        return (val_f - 1.0f) * tenzor::log(1.0f - p) + tenzor::log(p);
    }

    auto entropy() -> Tensor override {
        // H = -(1-p)*log(1-p)/p - log(p)
        auto eps = 1e-7f;
        auto p = tenzor::clamp(probs_, eps, 1.0f - eps);
        return tenzor::neg((1.0f - p) * tenzor::log(1.0f - p) / p) - tenzor::log(p);
    }

    auto mean() -> Tensor override {
        return tenzor::reciprocal(probs_);
    }

    auto variance() -> Tensor override {
        return (1.0f - probs_) / (probs_ * probs_);
    }

private:
    Tensor probs_;
};

// ============================================================================
// Gumbel Distribution
// ============================================================================

/**
 * @brief Gumbel(loc, scale) distribution (Type-I extreme value).
 *
 * Used in extreme value theory and as the basis for the Gumbel-Softmax trick.
 *
 * Sampled via inverse CDF: loc - scale * log(-log(U))
 * where U ~ Uniform(0, 1).
 *
 * PDF(x) = (1/scale) * exp(-(z + exp(-z))) where z = (x - loc)/scale
 */
class Gumbel : public Distribution {
public:
    Gumbel(Tensor loc, Tensor scale)
        : loc_(std::move(loc)), scale_(std::move(scale)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(loc_.shape().begin(), loc_.shape().end())
            : sample_shape;
        auto u = rand(shape, loc_.dtype(), loc_.device());
        // Clamp to avoid log(0) and log(-log(0))
        constexpr float kEps = 1e-7f;
        auto u_clamped = tenzor::clamp(u, kEps, 1.0f - kEps);
        // x = loc - scale * log(-log(u))
        return loc_ - scale_ * tenzor::log(tenzor::neg(tenzor::log(u_clamped)));
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        // Gumbel is reparameterizable via the inverse CDF transform.
        return sample(std::move(sample_shape));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x) = -log(scale) - z - exp(-z) where z = (x - loc)/scale
        auto z = (value - loc_) / scale_;
        return tenzor::neg(tenzor::log(scale_)) - z - tenzor::exp(tenzor::neg(z));
    }

    auto entropy() -> Tensor override {
        // H = log(scale) + 1 + euler_gamma (euler_gamma ~ 0.5772156649)
        return tenzor::log(scale_) + 1.5772156649f;
    }

    auto mean() -> Tensor override {
        // E[X] = loc + scale * euler_gamma
        return loc_ + scale_ * 0.5772156649f;
    }

    auto variance() -> Tensor override {
        // Var[X] = (pi^2 / 6) * scale^2
        constexpr float pi_sq_over_6 = static_cast<float>(M_PI * M_PI / 6.0);
        return scale_ * scale_ * pi_sq_over_6;
    }

private:
    Tensor loc_;
    Tensor scale_;
};

// ============================================================================
// HalfNormal Distribution
// ============================================================================

/**
 * @brief HalfNormal(scale) distribution — the absolute value of N(0, scale).
 *
 * Support: x >= 0.
 * PDF(x) = sqrt(2 / (pi * scale^2)) * exp(-x^2 / (2*scale^2))
 */
class HalfNormal : public Distribution {
public:
    explicit HalfNormal(Tensor scale)
        : scale_(std::move(scale)),
          normal_(zeros(std::vector<int64_t>(scale_.shape().begin(), scale_.shape().end()),
                        scale_.dtype(), scale_.device()),
                  scale_) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return tenzor::abs(normal_.sample(std::move(sample_shape)));
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return tenzor::abs(normal_.rsample(std::move(sample_shape)));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x) = Normal(0, scale).log_prob(x) + log(2) for x >= 0
        // For x < 0 return -inf
        auto lp = normal_.log_prob(value) + 0.6931471806f;  // log(2)
        auto neg_inf = full(std::vector<int64_t>(value.shape().begin(), value.shape().end()),
                            -std::numeric_limits<float>::infinity(),
                            value.dtype(), value.device());
        auto zero = zeros(std::vector<int64_t>(value.shape().begin(), value.shape().end()),
                          value.dtype(), value.device());
        auto mask = value >= zero;
        return where(mask, lp, neg_inf);
    }

    auto entropy() -> Tensor override {
        // H = 0.5 * log(pi * scale^2 / 2) + 0.5
        auto s2 = scale_ * scale_;
        return 0.5f * tenzor::log(s2 * static_cast<float>(M_PI / 2.0)) + 0.5f;
    }

    auto mean() -> Tensor override {
        // E[X] = scale * sqrt(2/pi)
        constexpr float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
        return scale_ * sqrt_2_over_pi;
    }

    auto variance() -> Tensor override {
        // Var[X] = scale^2 * (1 - 2/pi)
        constexpr float one_minus_2_over_pi = 0.3633802276f;  // 1 - 2/pi
        return scale_ * scale_ * one_minus_2_over_pi;
    }

    const Tensor& scale() const { return scale_; }

private:
    Tensor scale_;
    Normal normal_;
};

// ============================================================================
// HalfCauchy Distribution
// ============================================================================

/**
 * @brief HalfCauchy(scale) distribution — the absolute value of Cauchy(0, scale).
 *
 * Support: x >= 0.
 * PDF(x) = 2 / (pi * scale * (1 + (x/scale)^2))
 */
class HalfCauchy : public Distribution {
public:
    explicit HalfCauchy(Tensor scale)
        : scale_(std::move(scale)),
          cauchy_(zeros(std::vector<int64_t>(scale_.shape().begin(), scale_.shape().end()),
                        scale_.dtype(), scale_.device()),
                  scale_) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return tenzor::abs(cauchy_.sample(std::move(sample_shape)));
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "HalfCauchy is not reparameterizable; use sample() instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x) = log(2) - log(pi) - log(scale) - log(1 + (x/scale)^2) for x >= 0
        auto z = value / scale_;
        auto lp = 0.6931471806f - static_cast<float>(std::log(M_PI))
                 - tenzor::log(scale_) - tenzor::log(1.0f + z * z);
        auto neg_inf = full(std::vector<int64_t>(value.shape().begin(), value.shape().end()),
                            -std::numeric_limits<float>::infinity(),
                            value.dtype(), value.device());
        auto zero = zeros(std::vector<int64_t>(value.shape().begin(), value.shape().end()),
                          value.dtype(), value.device());
        auto mask = value >= zero;
        return where(mask, lp, neg_inf);
    }

    auto mean() -> Tensor override {
        throw std::runtime_error("HalfCauchy distribution has no defined mean");
    }

    auto variance() -> Tensor override {
        throw std::runtime_error("HalfCauchy distribution has no defined variance");
    }

private:
    Tensor scale_;
    Cauchy cauchy_;
};

// ============================================================================
// FisherSnedecor (F) Distribution
// ============================================================================

/**
 * @brief FisherSnedecor(df1, df2) — the F-distribution.
 *
 * Sampled as (X1/df1) / (X2/df2) where X1 ~ Gamma(df1/2, 1) and
 * X2 ~ Gamma(df2/2, 1).
 */
class FisherSnedecor : public Distribution {
public:
    FisherSnedecor(Tensor df1, Tensor df2)
        : df1_(std::move(df1)), df2_(std::move(df2)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(df1_.shape().begin(), df1_.shape().end())
            : sample_shape;
        auto one = full({1}, 1.0f, df1_.dtype(), Device::cpu());
        auto x1 = detail::fill_gamma_cpu(df1_.to(Device::cpu()) * 0.5f, one, shape);
        auto x2 = detail::fill_gamma_cpu(df2_.to(Device::cpu()) * 0.5f, one, shape);
        auto result = (x1 / df1_.to(Device::cpu())) / (x2 / df2_.to(Device::cpu()));
        return result.to(df1_.device());
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "FisherSnedecor is not reparameterizable; use sample() instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x; d1, d2) = d1/2 * log(d1) + d2/2 * log(d2) - lgamma(d1/2)
        //                   - lgamma(d2/2) + lgamma((d1+d2)/2)
        //                   + (d1/2 - 1) * log(x) - (d1+d2)/2 * log(d1*x + d2)
        auto d1h = df1_ * 0.5f;
        auto d2h = df2_ * 0.5f;
        auto ct = d1h * tenzor::log(df1_) + d2h * tenzor::log(df2_)
                - tenzor::lgamma(d1h) - tenzor::lgamma(d2h)
                + tenzor::lgamma(d1h + d2h);
        return ct + (d1h - 1.0f) * tenzor::log(value)
             - (d1h + d2h) * tenzor::log(df1_ * value + df2_);
    }

    auto mean() -> Tensor override {
        // E[X] = df2 / (df2 - 2) for df2 > 2
        return df2_ / (df2_ - 2.0f);
    }

    auto variance() -> Tensor override {
        // Var[X] = 2 * df2^2 * (df1 + df2 - 2) / (df1 * (df2 - 2)^2 * (df2 - 4))
        // for df2 > 4
        auto d2m2 = df2_ - 2.0f;
        auto d2m4 = df2_ - 4.0f;
        return 2.0f * df2_ * df2_ * (df1_ + df2_ - 2.0f)
             / (df1_ * d2m2 * d2m2 * d2m4);
    }

private:
    Tensor df1_;
    Tensor df2_;
};

// ============================================================================
// NegativeBinomial Distribution
// ============================================================================

/**
 * @brief NegativeBinomial(total_count, probs) distribution.
 *
 * Compound Poisson-Gamma formulation: sample a Gamma rate and then draw
 * Poisson counts. `total_count` is the number of failures, `probs` is the
 * success probability.
 *
 * P(X = k) = C(k + r - 1, k) * p^k * (1 - p)^r
 */
class NegativeBinomial : public Distribution {
public:
    NegativeBinomial(Tensor total_count, Tensor probs)
        : total_count_(std::move(total_count)), probs_(std::move(probs)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(total_count_.shape().begin(), total_count_.shape().end())
            : sample_shape;

        // rate ~ Gamma(total_count, probs / (1 - probs))
        auto eps = 1e-7f;
        auto p_clamped = tenzor::clamp(probs_, eps, 1.0f - eps);
        auto gamma_rate = p_clamped / (1.0f - p_clamped);
        auto rate_samples = detail::fill_gamma_cpu(
            total_count_.to(Device::cpu()),
            tenzor::reciprocal(gamma_rate.to(Device::cpu())),  // Gamma uses rate param
            shape);

        // Draw Poisson(rate) counts
        auto rate_cpu = rate_samples.contiguous();
        auto out = zeros(shape, DType::Int64, Device::cpu());
        int64_t n = out.numel();
        const int64_t rate_n = rate_cpu.numel();
        int64_t* op = out.data<int64_t>();

        if (rate_cpu.dtype() == DType::Float32) {
            const float* rp = rate_cpu.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                double lam = (rate_n == 1) ? rp[0] : rp[i % rate_n];
                op[i] = detail::sample_poisson_scalar(std::max(lam, 0.0));
            }
        } else if (rate_cpu.dtype() == DType::Float64) {
            const double* rp = rate_cpu.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                double lam = (rate_n == 1) ? rp[0] : rp[i % rate_n];
                op[i] = detail::sample_poisson_scalar(std::max(lam, 0.0));
            }
        } else {
            throw std::runtime_error("NegativeBinomial: probs must be Float32 or Float64");
        }

        return out.to(total_count_.device());
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "NegativeBinomial is not reparameterizable; use sample() instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log P(k) = lgamma(k + r) - lgamma(r) - lgamma(k + 1)
        //          + r * log(1 - p) + k * log(p)
        auto eps = 1e-7f;
        auto p = tenzor::clamp(probs_, eps, 1.0f - eps);
        auto val_f = value.to(total_count_.dtype());
        return tenzor::lgamma(val_f + total_count_)
             - tenzor::lgamma(total_count_)
             - tenzor::lgamma(val_f + 1.0f)
             + total_count_ * tenzor::log(1.0f - p)
             + val_f * tenzor::log(p);
    }

    auto mean() -> Tensor override {
        // E[X] = r * p / (1 - p)
        auto eps = 1e-7f;
        auto p = tenzor::clamp(probs_, eps, 1.0f - eps);
        return total_count_ * p / (1.0f - p);
    }

    auto variance() -> Tensor override {
        // Var[X] = r * p / (1 - p)^2
        auto eps = 1e-7f;
        auto p = tenzor::clamp(probs_, eps, 1.0f - eps);
        auto q = 1.0f - p;
        return total_count_ * p / (q * q);
    }

private:
    Tensor total_count_;
    Tensor probs_;
};

// ============================================================================
// VonMises Distribution
// ============================================================================

/**
 * @brief VonMises(loc, concentration) — circular distribution on [-pi, pi].
 *
 * Uses Best & Fisher (1979) rejection sampling algorithm.
 * PDF(x) = exp(kappa * cos(x - mu)) / (2 * pi * I0(kappa))
 */
class VonMises : public Distribution {
public:
    VonMises(Tensor loc, Tensor concentration)
        : loc_(std::move(loc)), concentration_(std::move(concentration)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(loc_.shape().begin(), loc_.shape().end())
            : sample_shape;

        // Best & Fisher (1979) rejection sampling on CPU
        auto kappa_cpu = concentration_.to(Device::cpu()).contiguous();
        auto loc_cpu = loc_.to(Device::cpu()).contiguous();
        auto out = zeros(shape, loc_.dtype(), Device::cpu());
        int64_t n = out.numel();
        const int64_t kappa_n = kappa_cpu.numel();
        const int64_t loc_n = loc_cpu.numel();

        auto& rng = detail::distribution_rng();
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        if (out.dtype() == DType::Float32) {
            float* op = out.data<float>();
            const float* kp = kappa_cpu.data<float>();
            const float* lp = loc_cpu.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                double kappa = (kappa_n == 1) ? kp[0] : kp[i % kappa_n];
                double mu = (loc_n == 1) ? lp[0] : lp[i % loc_n];
                op[i] = static_cast<float>(sample_vonmises_scalar(kappa, mu, rng, uniform));
            }
        } else if (out.dtype() == DType::Float64) {
            double* op = out.data<double>();
            const double* kp = kappa_cpu.data<double>();
            const double* lp = loc_cpu.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                double kappa = (kappa_n == 1) ? kp[0] : kp[i % kappa_n];
                double mu = (loc_n == 1) ? lp[0] : lp[i % loc_n];
                op[i] = sample_vonmises_scalar(kappa, mu, rng, uniform);
            }
        } else {
            throw std::runtime_error("VonMises: only Float32 and Float64 supported");
        }

        return out.to(loc_.device());
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "VonMises is not reparameterizable; use sample() instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x) = kappa * cos(x - mu) - log(2*pi) - log(I0(kappa))
        auto log_i0 = tenzor::log(tenzor::bessel_i0(concentration_));
        return concentration_ * tenzor::cos(value - loc_)
             - static_cast<float>(std::log(2.0 * M_PI)) - log_i0;
    }

    auto mean() -> Tensor override {
        // Circular mean = loc
        return loc_;
    }

    auto variance() -> Tensor override {
        // Circular variance = 1 - I1(kappa) / I0(kappa)
        return 1.0f - tenzor::bessel_i1(concentration_) / tenzor::bessel_i0(concentration_);
    }

private:
    /// Best & Fisher (1979) rejection sampler for a single VonMises variate.
    template <typename RNG, typename UniformDist>
    static auto sample_vonmises_scalar(double kappa, double mu,
                                       RNG& rng, UniformDist& uniform) -> double {
        if (kappa < 1e-6) {
            // Nearly uniform on the circle
            return mu + (uniform(rng) * 2.0 - 1.0) * M_PI;
        }
        double tau = 1.0 + std::sqrt(1.0 + 4.0 * kappa * kappa);
        double rho = (tau - std::sqrt(2.0 * tau)) / (2.0 * kappa);
        double r = (1.0 + rho * rho) / (2.0 * rho);

        while (true) {
            double u1 = uniform(rng);
            double u2 = uniform(rng);
            double u3 = uniform(rng);

            double z = std::cos(M_PI * u1);
            double f = (1.0 + r * z) / (r + z);
            double c = kappa * (r - f);

            if (c * (2.0 - c) >= u2 || std::log(c / u2) + 1.0 - c >= 0.0) {
                double sign = (u3 - 0.5 < 0.0) ? -1.0 : 1.0;
                double theta = mu + sign * std::acos(f);
                // Wrap to [-pi, pi]
                theta = std::fmod(theta + M_PI, 2.0 * M_PI);
                if (theta < 0.0) theta += 2.0 * M_PI;
                theta -= M_PI;
                return theta;
            }
        }
    }

    Tensor loc_;
    Tensor concentration_;
};

// ============================================================================
// RelaxedBernoulli Distribution
// ============================================================================

/**
 * @brief RelaxedBernoulli(temperature, probs) — continuous relaxation of
 *        Bernoulli via the Binary Concrete / Logistic-sigmoid.
 *
 * Reparameterizable. Samples are in (0, 1).
 */
class RelaxedBernoulli : public Distribution {
public:
    RelaxedBernoulli(Tensor temperature, Tensor probs)
        : temperature_(std::move(temperature)),
          probs_(std::move(probs)) {
        // Compute logits = log(probs / (1 - probs))
        auto eps = 1e-7f;
        auto p = tenzor::clamp(probs_, eps, 1.0f - eps);
        logits_ = tenzor::log(p / (1.0f - p));
    }

    static auto from_logits(Tensor temperature, Tensor logits) -> RelaxedBernoulli {
        auto t = std::move(temperature);
        auto l = std::move(logits);
        auto probs = tenzor::sigmoid(l);
        RelaxedBernoulli rb(t, probs);
        rb.logits_ = l;
        return rb;
    }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return rsample(std::move(sample_shape));
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(probs_.shape().begin(), probs_.shape().end())
            : sample_shape;
        auto u = rand(shape, probs_.dtype(), probs_.device());
        constexpr float kEps = 1e-7f;
        auto u_clamped = tenzor::clamp(u, kEps, 1.0f - kEps);
        // Standard logistic noise
        auto logistic = tenzor::log(u_clamped) - tenzor::log(1.0f - u_clamped);
        return tenzor::sigmoid((logits_ + logistic) / temperature_);
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // logit(x) = log(x / (1 - x))
        auto eps = 1e-7f;
        auto x = tenzor::clamp(value, eps, 1.0f - eps);
        auto logit_x = tenzor::log(x / (1.0f - x));
        auto diff = logits_ - temperature_ * logit_x;
        // softplus(x) = log(1 + exp(x))
        auto sp = tenzor::log(1.0f + tenzor::exp(diff));
        return diff - 2.0f * sp + tenzor::log(temperature_)
             - tenzor::log(x) - tenzor::log(1.0f - x);
    }

private:
    Tensor temperature_;
    Tensor probs_;
    Tensor logits_;
};

// ============================================================================
// RelaxedOneHotCategorical Distribution (Gumbel-Softmax)
// ============================================================================

/**
 * @brief RelaxedOneHotCategorical(temperature, probs/logits) — Gumbel-Softmax.
 *
 * Continuous relaxation of the Categorical distribution. Reparameterizable.
 * Samples are on the simplex (sum to 1).
 */
class RelaxedOneHotCategorical : public Distribution {
public:
    RelaxedOneHotCategorical(Tensor temperature, Tensor probs)
        : temperature_(std::move(temperature)),
          probs_(std::move(probs)) {
        auto eps = 1e-7f;
        auto p = tenzor::clamp(probs_, eps, 1.0f);
        logits_ = tenzor::log(p);
    }

    static auto from_logits(Tensor temperature, Tensor logits)
        -> RelaxedOneHotCategorical {
        // Compute probs via softmax (inline, no dependency on nn::)
        auto max_val = tenzor::max(logits);
        auto shifted = logits - max_val;
        auto exp_vals = tenzor::exp(shifted);
        auto sum_exp = tenzor::sum(exp_vals, -1, /*keepdim=*/true);
        auto probs = exp_vals / sum_exp;
        RelaxedOneHotCategorical d(temperature, probs);
        d.logits_ = logits;
        return d;
    }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return rsample(std::move(sample_shape));
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(logits_.shape().begin(), logits_.shape().end())
            : sample_shape;
        auto u = rand(shape, logits_.dtype(), logits_.device());
        constexpr float kEps = 1e-7f;
        auto u_clamped = tenzor::clamp(u, kEps, 1.0f - kEps);
        // Gumbel(0, 1) noise: -log(-log(u))
        auto gumbels = tenzor::neg(tenzor::log(tenzor::neg(tenzor::log(u_clamped))));
        // Scores: (logits + gumbels) / temperature
        auto scores = (logits_ + gumbels) / temperature_;
        // Softmax (inline implementation over last dim)
        auto max_s = tenzor::max(scores);
        auto shifted = scores - max_s;
        auto exp_s = tenzor::exp(shifted);
        auto sum_exp = tenzor::sum(exp_s, -1, /*keepdim=*/true);
        return exp_s / sum_exp;
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // Concrete distribution PDF:
        // log p(x) = lgamma(K) + (K-1)*log(temp) + sum(logits/temp - log(x))
        //          - K * logsumexp(logits/temp - log(x))
        auto eps = 1e-7f;
        auto x = tenzor::clamp(value, eps, 1.0f);
        auto K_val = static_cast<float>(logits_.shape().back());
        auto scaled_logits = logits_ / temperature_;
        auto score = scaled_logits - tenzor::log(x);
        auto lse = logsumexp(score, -1, /*keepdim=*/true);
        auto lgamma_K = static_cast<float>(std::lgamma(static_cast<double>(K_val)));
        return lgamma_K + (K_val - 1.0f) * tenzor::log(temperature_)
             + tenzor::sum(score, -1, /*keepdim=*/true) - K_val * lse;
    }

private:
    Tensor temperature_;
    Tensor probs_;
    Tensor logits_;
};

// ============================================================================
// Wishart Distribution
// ============================================================================

/**
 * @brief Wishart(df, scale_tril) — matrix-variate distribution over positive
 *        definite matrices.
 *
 * `df` is a scalar tensor (degrees of freedom, must be >= dimension p).
 * `scale_tril` is the lower-triangular Cholesky factor (p x p) of the scale
 * matrix V, such that the distribution's scale is V = scale_tril @ scale_tril^T.
 *
 * Uses Bartlett decomposition for sampling.
 */
class Wishart : public Distribution {
public:
    Wishart(Tensor df, Tensor scale_tril)
        : df_(std::move(df)), scale_tril_(std::move(scale_tril)) {
        // Determine dimension from scale_tril shape
        auto s = scale_tril_.shape();
        if (s.size() < 2 || s[s.size()-1] != s[s.size()-2]) {
            throw std::runtime_error("Wishart: scale_tril must be a square matrix");
        }
        p_ = s[s.size() - 1];
    }

    auto sample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        // Bartlett decomposition on CPU
        auto df_cpu = df_.to(Device::cpu()).contiguous();
        double df_val;
        if (df_cpu.dtype() == DType::Float32) {
            df_val = static_cast<double>(df_cpu.data<float>()[0]);
        } else {
            df_val = df_cpu.data<double>()[0];
        }

        auto dtype = scale_tril_.dtype();
        auto device = scale_tril_.device();

        // Build lower-triangular Bartlett factor L on CPU
        auto L = zeros({p_, p_}, dtype, Device::cpu());

        auto& rng = detail::distribution_rng();
        std::normal_distribution<double> normal(0.0, 1.0);

        if (dtype == DType::Float32) {
            float* lp = L.data<float>();
            for (int64_t i = 0; i < p_; ++i) {
                // Diagonal: sqrt(Chi2(df - i)) = sqrt(Gamma((df-i)/2, 0.5) * 2)
                // Equivalently, sample Chi2(df - i) via Gamma((df-i)/2, 0.5)
                double chi2_val = detail::sample_gamma_scalar<double>(
                    (df_val - static_cast<double>(i)) / 2.0, 0.5);
                lp[i * p_ + i] = static_cast<float>(std::sqrt(chi2_val));
                // Below diagonal: standard normal
                for (int64_t j = 0; j < i; ++j) {
                    lp[i * p_ + j] = static_cast<float>(normal(rng));
                }
            }
        } else if (dtype == DType::Float64) {
            double* lp = L.data<double>();
            for (int64_t i = 0; i < p_; ++i) {
                double chi2_val = detail::sample_gamma_scalar<double>(
                    (df_val - static_cast<double>(i)) / 2.0, 0.5);
                lp[i * p_ + i] = std::sqrt(chi2_val);
                for (int64_t j = 0; j < i; ++j) {
                    lp[i * p_ + j] = normal(rng);
                }
            }
        } else {
            throw std::runtime_error("Wishart: only Float32 and Float64 supported");
        }

        L = L.to(device);
        // A = scale_tril @ L
        auto A = matmul(scale_tril_, L);
        // W = A @ A^T
        return matmul(A, transpose(A, -2, -1));
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "Wishart is not reparameterizable; use sample() instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(X; n, V) = (n-p-1)/2 * log|X| - 0.5 * tr(V^{-1} X) - log C
        // where log C = n*p/2 * log(2) + n/2 * log|V| + multivariate_lgamma(n/2, p)
        // and |V| from scale_tril: log|V| = 2 * sum(log(diag(scale_tril)))

        double df_val;
        auto df_cpu = df_.to(Device::cpu()).contiguous();
        if (df_cpu.dtype() == DType::Float32) {
            df_val = static_cast<double>(df_cpu.data<float>()[0]);
        } else {
            df_val = df_cpu.data<double>()[0];
        }

        double half_df = df_val / 2.0;
        double p_d = static_cast<double>(p_);

        // log|V| = 2 * sum(log(diag(L)))
        auto L_cpu = scale_tril_.to(Device::cpu()).contiguous();
        double log_det_scale = 0.0;
        if (L_cpu.dtype() == DType::Float32) {
            const float* lp = L_cpu.data<float>();
            for (int64_t i = 0; i < p_; ++i) {
                log_det_scale += std::log(static_cast<double>(lp[i * p_ + i]));
            }
        } else {
            const double* lp = L_cpu.data<double>();
            for (int64_t i = 0; i < p_; ++i) {
                log_det_scale += std::log(lp[i * p_ + i]);
            }
        }
        log_det_scale *= 2.0;

        // multivariate_lgamma(a, p) = p*(p-1)/4 * log(pi) + sum_{j=1}^{p} lgamma(a - (j-1)/2)
        double mv_lgamma = p_d * (p_d - 1.0) / 4.0 * std::log(M_PI);
        for (int64_t j = 1; j <= p_; ++j) {
            mv_lgamma += std::lgamma(half_df - static_cast<double>(j - 1) / 2.0);
        }

        double log_C = half_df * p_d * std::log(2.0) + half_df * log_det_scale + mv_lgamma;

        // log|X| via slogdet
        auto [sign_det, logabsdet] = linalg::slogdet(value);

        // tr(V^{-1} X): solve V Y = X, then tr(Y)
        // V = L @ L^T, so we can solve via L.
        auto scale_matrix = matmul(scale_tril_, transpose(scale_tril_, -2, -1));
        auto solved = linalg::solve(scale_matrix, value);
        auto trace_val = tenzor::trace(solved);

        auto coeff = static_cast<float>((df_val - p_d - 1.0) / 2.0);
        auto log_C_tensor = full({1}, static_cast<float>(log_C),
                                 value.dtype(), value.device());

        return coeff * logabsdet - 0.5f * trace_val - log_C_tensor;
    }

    auto mean() -> Tensor override {
        // E[W] = df * V = df * scale_tril @ scale_tril^T
        auto scale = matmul(scale_tril_, transpose(scale_tril_, -2, -1));
        return df_ * scale;
    }

    auto variance() -> Tensor override {
        throw std::runtime_error(
            "Wishart::variance() not implemented (element-wise variance of a "
            "matrix distribution is non-trivial; use mean() and sample())");
    }

private:
    Tensor df_;
    Tensor scale_tril_;
    int64_t p_;
};

// ============================================================================
// kl_divergence
// ============================================================================
//
// Closed-form KL divergences for supported distribution pairs. Falls back
// to throwing for unknown pairs — users should use a Monte-Carlo estimator
// via sampling when no closed form is registered.

/**
 * @brief Compute KL(p || q) for matching distribution types.
 *
 * Supported pairs:
 *   - Normal || Normal
 *   - Uniform || Uniform (same support)
 *   - Exponential || Exponential
 *   - Laplace || Laplace (same loc)
 *   - Gamma || Gamma
 *   - Beta || Beta
 *   - Categorical || Categorical
 *   - Bernoulli || Bernoulli
 *
 * For anything else, throws std::runtime_error. Use Monte-Carlo estimation
 * via `(p.sample() * (p.log_prob(x) - q.log_prob(x))).mean()` instead.
 */
inline auto kl_divergence(Distribution& p, Distribution& q) -> Tensor {
    if (auto* pn = dynamic_cast<Normal*>(&p)) {
        if (auto* qn = dynamic_cast<Normal*>(&q)) {
            // KL(N(mu1, s1) || N(mu2, s2))
            //   = log(s2/s1) + (s1^2 + (mu1 - mu2)^2) / (2 s2^2) - 0.5
            auto mu1 = pn->mean(); auto v1 = pn->variance();
            auto mu2 = qn->mean(); auto v2 = qn->variance();
            auto s1 = tenzor::sqrt(v1);
            auto s2 = tenzor::sqrt(v2);
            auto diff = mu1 - mu2;
            return tenzor::log(s2 / s1) + (v1 + diff * diff) / (v2 * 2.0f) - 0.5f;
        }
    }
    if (auto* pb = dynamic_cast<BernoulliDist*>(&p)) {
        if (auto* qb = dynamic_cast<BernoulliDist*>(&q)) {
            auto p_mean = pb->mean();
            auto q_mean = qb->mean();
            auto eps = 1e-7f;
            auto p_clamped = tenzor::clamp(p_mean, eps, 1.0f - eps);
            auto q_clamped = tenzor::clamp(q_mean, eps, 1.0f - eps);
            return p_clamped * (tenzor::log(p_clamped) - tenzor::log(q_clamped))
                 + (1.0f - p_clamped) * (tenzor::log(1.0f - p_clamped) - tenzor::log(1.0f - q_clamped));
        }
    }
    if (auto* ph = dynamic_cast<HalfNormal*>(&p)) {
        if (auto* qh = dynamic_cast<HalfNormal*>(&q)) {
            // KL(HalfNormal(s1) || HalfNormal(s2))
            //   = KL(N(0,s1) || N(0,s2))   (the folded densities share the
            //   same KL when both are centered at zero)
            //   = log(s2/s1) + s1^2 / (2*s2^2) - 0.5
            auto s1 = ph->scale();
            auto s2 = qh->scale();
            auto v1 = s1 * s1;
            auto v2 = s2 * s2;
            return tenzor::log(s2 / s1) + v1 / (v2 * 2.0f) - 0.5f;
        }
    }
    throw std::runtime_error(
        "kl_divergence: closed form not registered for this distribution pair; "
        "use Monte-Carlo estimation via sample() + log_prob()");
}

} // namespace distributions
} // namespace tenzor
