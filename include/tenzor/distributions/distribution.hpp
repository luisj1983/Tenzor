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
#include "../utils/error.hpp"
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

    /**
     * @brief Compute entropy of the distribution.
     *
     * Pure-virtual: every concrete distribution must supply a closed-form
     * or Monte-Carlo implementation. Use the `detail::mc_entropy(...)`
     * helper when no closed form exists (the MC estimator IS the correct
     * mathematical definition).
     */
    virtual auto entropy() -> Tensor = 0;

    /**
     * @brief Distribution mean E[X].
     *
     * Pure-virtual; every concrete distribution must implement.
     */
    virtual auto mean() -> Tensor = 0;

    /**
     * @brief Distribution variance Var[X].
     *
     * Pure-virtual; every concrete distribution must implement.
     */
    virtual auto variance() -> Tensor = 0;

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
        // Full-precision (double) additive constants. A float literal (`f`
        // suffix) would silently cap accuracy at ~1e-7 for Float64 inputs even
        // though the rest of the computation is double — the classic
        // float32-constant-in-multi-dtype-path regression.
        constexpr double kLogSqrt2Pi = 0.91893853320467274178; // log(sqrt(2*pi))
        auto var = scale_ * scale_;
        auto log_scale = tenzor::log(scale_);
        auto diff = value - loc_;
        return tenzor::neg(diff * diff) / (var * 2.0) - log_scale - kLogSqrt2Pi;
    }

    auto entropy() -> Tensor override {
        constexpr double kHalfLog2PiE = 1.41893853320467274178; // 0.5*log(2*pi*e)
        return tenzor::log(scale_) + kHalfLog2PiE;
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

    auto low() const -> const Tensor& { return low_; }
    auto high() const -> const Tensor& { return high_; }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(low_.shape().begin(), low_.shape().end())
            : sample_shape;
        auto u = rand(shape, low_.dtype(), low_.device());
        return low_ + (high_ - low_) * u;
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // Density is 1/(high-low) on the support [low, high) and 0 (log_prob
        // -inf) outside it, matching torch.distributions.Uniform.
        auto shape = std::vector<int64_t>(value.shape().begin(), value.shape().end());
        auto lp = tenzor::neg(tenzor::log(high_ - low_)); // -log(high - low)
        auto lp_b = lp + zeros(shape, value.dtype(), value.device()); // broadcast to value shape
        auto neg_inf = full(shape, -std::numeric_limits<float>::infinity(),
                            value.dtype(), value.device());
        auto ge_low = value >= low_;
        auto lt_high = value < high_;
        return where(ge_low, where(lt_high, lp_b, neg_inf), neg_inf);
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

    auto probs() const -> const Tensor& { return probs_; }

    static auto from_logits(const Tensor& logits) -> Categorical {
        // Per-row softmax over the last (category) dim. Using full-tensor
        // scalar reductions would normalize across the whole batch, so for
        // batched (B, K) logits the per-row probs would not sum to 1. Reduce
        // over the last dim with keepdim so broadcasting subtracts/divides
        // per row.
        auto max_val = tenzor::max(logits, /*dim=*/-1, /*keepdim=*/true);
        auto shifted = logits - max_val;
        auto exp_vals = tenzor::exp(shifted);
        auto sum_exp = tenzor::sum(exp_vals, /*dim=*/-1, /*keepdim=*/true);
        return Categorical(exp_vals / sum_exp);
    }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        // PyTorch Categorical semantics: sample(sample_shape) returns a tensor
        // of shape sample_shape + batch_shape, where batch_shape = probs[:-1].
        // multinomial draws num_samples = prod(sample_shape) indices per row:
        //   1D probs (K,)   -> (num_samples,)         ; batch_shape = ()
        //   2D probs (B, K) -> (B, num_samples)       ; batch_shape = (B,)
        // We then permute the sample axis to the front and reshape to the
        // requested sample_shape + batch_shape layout.
        int64_t num_samples = 1;
        for (auto s : sample_shape) {
            if (s < 0) {
                throw std::runtime_error("Categorical: sample_shape dims must be non-negative");
            }
            num_samples *= s;
        }
        if (num_samples < 1) num_samples = 1;

        auto flat = multinomial(probs_, num_samples, /*replacement=*/true);

        // batch_shape = probs_.shape[:-1]
        auto pshape = probs_.shape();
        std::vector<int64_t> batch_shape(pshape.begin(),
                                         pshape.end() > pshape.begin()
                                             ? pshape.end() - 1
                                             : pshape.end());

        // Bring the sample axis to the front so reshape produces
        // sample_shape + batch_shape (sample dims vary slowest).
        Tensor samples_first;
        if (batch_shape.empty()) {
            // 1D probs: multinomial result is already (num_samples,).
            samples_first = flat;
        } else {
            // 2D probs: result is (B, num_samples) -> transpose to
            // (num_samples, B) so the sample dim leads.
            samples_first = tenzor::transpose(flat, 0, 1).contiguous();
        }

        std::vector<int64_t> out_shape(sample_shape.begin(), sample_shape.end());
        out_shape.insert(out_shape.end(), batch_shape.begin(), batch_shape.end());
        if (out_shape.empty()) {
            // sample_shape == {} and scalar batch: a single draw. Keep the
            // 1-element vector shape produced by multinomial(num_samples=1).
            return samples_first;
        }
        return samples_first.reshape(out_shape);
    }

    auto rsample(std::vector<int64_t> /*sample_shape*/ = {}) -> Tensor override {
        throw std::runtime_error(
            "Categorical is not reparameterizable; use sample() or a "
            "straight-through / Gumbel-softmax estimator instead");
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto log_probs = tenzor::log(probs_);  // (batch..., K)
        const auto lp_shape = log_probs.shape();  // rank >= 1
        const int64_t K = lp_shape.empty() ? 1 : lp_shape.back();
        const int64_t batch_rank =
            static_cast<int64_t>(lp_shape.size()) - 1;  // dims before K

        auto idx_i64 = value.to(DType::Int64);
        const auto val_shape = idx_i64.shape();
        const int64_t val_rank = static_cast<int64_t>(val_shape.size());

        // PyTorch Categorical.log_prob(value): for probs of batch_shape B and
        // event size K, `value` may be ANY shape whose trailing dims broadcast
        // against B. gather() requires input and index to share rank, so we
        // must align log_probs (B..., K) with the query layout — NOT just
        // assume value has exactly the batch dims (which breaks for unbatched
        // probs (K,) queried with an (N,) index, the previous bug:
        // "gather: input and index must have same number of dimensions").
        //
        // General handling:
        //   - Unbatched probs (batch_rank == 0): every query value indexes the
        //     same K-vector. Broadcast log_probs to value_shape + (K,), make
        //     idx value_shape + (1,), gather along -1, squeeze.
        //   - Batched probs whose batch dims match value's trailing dims: the
        //     classic path (value rank == batch_rank), unsqueeze + gather.
        if (batch_rank == 0) {
            // Reshape log_probs (K,) -> (1,1,...,1,K) with val_rank leading 1s,
            // then expand to value_shape + (K,).
            std::vector<int64_t> lp_expand_shape;
            lp_expand_shape.reserve(static_cast<size_t>(val_rank) + 1);
            for (int64_t i = 0; i < val_rank; ++i) lp_expand_shape.push_back(1);
            lp_expand_shape.push_back(K);
            auto lp_reshaped = log_probs.reshape(lp_expand_shape);

            std::vector<int64_t> target_shape(val_shape.begin(), val_shape.end());
            target_shape.push_back(K);
            auto lp_b = tenzor::expand(lp_reshaped, target_shape).contiguous();

            auto idx = tenzor::unsqueeze(idx_i64, -1);  // value_shape + (1,)
            return tenzor::squeeze(gather(lp_b, -1, idx), -1);
        }

        // Batched: keep the original equal-rank contract.
        auto idx = tenzor::unsqueeze(idx_i64, -1);
        return tenzor::squeeze(gather(log_probs, -1, idx), -1);
    }

    auto entropy() -> Tensor override {
        auto log_probs = tenzor::log(probs_);
        // Reduce only over the category axis so batched probs (B, K) yield (B,).
        return tenzor::neg(tenzor::sum(probs_ * log_probs, /*dim=*/-1, /*keepdim=*/false));
    }

    /**
     * @brief Mean of the categorical distribution: E[X] = sum_k k * p_k.
     *
     * Treats class indices as scalar values 0..K-1 (PyTorch convention).
     * Result has shape probs_.shape[:-1].
     */
    auto mean() -> Tensor override {
        const auto K = probs_.shape().back();
        // arange(0..K) in probs_'s dtype/device; broadcasts against probs_.
        auto idx = arange(0.0, static_cast<double>(K), 1.0,
                          probs_.dtype(), probs_.device());
        return tenzor::sum(probs_ * idx, /*dim=*/-1, /*keepdim=*/false);
    }

    /**
     * @brief Variance of the categorical distribution: E[(X - E[X])^2].
     */
    auto variance() -> Tensor override {
        const auto K = probs_.shape().back();
        auto idx = arange(0.0, static_cast<double>(K), 1.0,
                          probs_.dtype(), probs_.device());
        auto m = tenzor::sum(probs_ * idx, /*dim=*/-1, /*keepdim=*/true);
        auto diff = idx - m;
        return tenzor::sum(probs_ * diff * diff, /*dim=*/-1, /*keepdim=*/false);
    }

    /**
     * @brief cdf/icdf are not defined for Categorical (audit E.5).
     *
     * Categorical has no canonical scalar order on its support (class
     * labels are nominal), so no cumulative distribution exists in the
     * usual sense. Raise a typed ``DistributionMethodUndefined`` rather
     * than a generic ``runtime_error``.
     */
    [[noreturn]] auto cdf(const Tensor& /*value*/) -> Tensor {
        throw ::tenzor::error::DistributionMethodUndefined(
            "Categorical::cdf is undefined: class labels are nominal, "
            "no canonical scalar order on the support");
    }

    [[noreturn]] auto icdf(const Tensor& /*q*/) -> Tensor {
        throw ::tenzor::error::DistributionMethodUndefined(
            "Categorical::icdf is undefined: class labels are nominal, "
            "no canonical scalar order on the support");
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

    auto rate() const -> const Tensor& { return rate_; }

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

    auto loc() const -> const Tensor& { return loc_; }
    auto scale() const -> const Tensor& { return scale_; }

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

    auto probs() const -> const Tensor& { return probs_; }

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

    /**
     * @brief CDF of Bernoulli (audit E.5).
     *
     *   F(k) = 0       for k < 0
     *        = 1 - p   for 0 <= k < 1
     *        = 1       for k >= 1
     */
    auto cdf(const Tensor& value) -> Tensor {
        auto val_f = value.to(probs_.dtype());
        auto zeros_t = tenzor::zeros_like(probs_ * val_f);
        auto ones_t  = zeros_t + 1.0f;
        // mask_below: value < 0
        auto mask_at_or_above = val_f >= ones_t;        // value >= 1
        auto mask_negative    = val_f < zeros_t;        // value < 0
        // Default (0 <= value < 1): 1 - p
        auto base = 1.0f - probs_;
        auto with_top    = tenzor::where(mask_at_or_above, ones_t, base);
        auto with_bottom = tenzor::where(mask_negative, zeros_t, with_top);
        return with_bottom;
    }

    /**
     * @brief Inverse CDF of Bernoulli (audit E.5).
     *
     * Q(q) = 0  if q <= 1 - p
     *      = 1  if q >  1 - p
     */
    auto icdf(const Tensor& q) -> Tensor {
        auto qf = q.to(probs_.dtype());
        auto threshold = 1.0f - probs_;
        auto ones_t  = tenzor::ones_like(qf * probs_);
        auto zeros_t = tenzor::zeros_like(ones_t);
        return tenzor::where(qf > threshold, ones_t, zeros_t);
    }

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

/// Shared fallback PRNG for per-sample rejection algorithms, used only when no
/// Generator is threaded through. Seeded once on first use.
inline auto& distribution_rng() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng;
}

/// Resolve the engine to draw from: the supplied Generator's engine when one is
/// active (so seeding a Generator makes Gamma/Beta/Dirichlet/StudentT/Poisson
/// reproducible), otherwise the thread-local fallback.
inline auto rng_for(Generator* gen) -> std::mt19937_64& {
    return gen ? gen->engine() : distribution_rng();
}

/// Marsaglia–Tsang (2000) rejection sampler for Gamma(shape >= 1).
/// For shape < 1, use the boosting trick gamma(s) = gamma(s+1) * U^(1/s).
/// `shape` is the concentration parameter (a.k.a. alpha, k).
/// Draws from the supplied engine for reproducibility.
template <typename T>
inline auto sample_gamma_scalar(T shape_alpha, T rate_beta, std::mt19937_64& rng) -> T {
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
/// Draws from the supplied engine for reproducibility.
inline auto sample_poisson_scalar(double lambda, std::mt19937_64& rng) -> int64_t {
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

/// Draw a single Binomial(n, p) variate via the geometric (waiting-time)
/// inversion method. Successive inter-success gaps G ~ Geometric(p) are drawn
/// and counted until they exceed n trials. Expected iterations are
/// n·min(p, 1-p) (using the p>0.5 symmetry flip), so this is far cheaper than
/// the n full-tensor Bernoulli sums it replaces while remaining exact for all
/// parameter values.
inline auto sample_binomial_scalar(int64_t n, double p, std::mt19937_64& rng) -> int64_t {
    if (n <= 0) return 0;
    if (p <= 0.0) return 0;
    if (p >= 1.0) return n;

    // Symmetry: sampling successes with prob p is the same as n minus the
    // number of "failures" with prob 1-p. Always iterate over the rarer event
    // so the expected loop count is n·min(p, 1-p).
    const bool flip = (p > 0.5);
    const double q = flip ? (1.0 - p) : p;

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double log_one_minus_q = std::log(1.0 - q);

    int64_t count = 0;
    int64_t trial = 0;  // index of the next trial to consider (0-based)
    while (true) {
        // Geometric gap: number of failures before the next success.
        double u = uniform(rng);
        // Guard against u == 0 producing -inf; log(1) == 0 advances by 0.
        double gap = std::floor(std::log(1.0 - u) / log_one_minus_q);
        if (gap < 0.0) gap = 0.0;
        trial += static_cast<int64_t>(gap);
        if (trial >= n) break;
        ++count;        // a success lands on trial `trial`
        ++trial;        // move past it
        if (trial >= n) break;
    }

    return flip ? (n - count) : count;
}

/// Broadcast-compatible element-wise CPU fill for Gamma sampling.
/// concentration and rate are broadcast to the output shape.
inline auto fill_gamma_cpu(const Tensor& concentration, const Tensor& rate,
                           std::vector<int64_t> shape, std::mt19937_64& rng) -> Tensor {
    // Float16/BFloat16: sample in Float32 (FP16 lacks the range/precision
    // for the Marsaglia+Tsang squeeze-and-trickle; tail Gammas overflow to
    // inf at scale > 65504). Same widen-narrow pattern used by the CPU
    // pointwise kernels.
    if (concentration.dtype() == DType::Float16 || concentration.dtype() == DType::BFloat16) {
        auto orig = concentration.dtype();
        auto sampled = fill_gamma_cpu(concentration.to(DType::Float32),
                                      rate.to(DType::Float32),
                                      shape, rng);
        return sampled.to(orig);
    }

    auto out = zeros(shape, concentration.dtype(), Device::cpu());
    int64_t n = out.numel();

    // Materialize concentration/rate on CPU and broadcast each to the output
    // shape so that element i of the output is paired with the geometrically
    // correct parameter element. Scalar params (numel==1) and params whose
    // shape already equals `shape` are the trivial cases; numpy-style
    // right-aligned broadcasting (e.g. sample_shape prepending leading dims,
    // or differing non-scalar concentration/rate shapes) is handled by
    // broadcast_to, which throws a clear error for genuinely incompatible
    // shapes instead of silently wrapping with a modulo.
    auto conc_cpu = ::tenzor::broadcast_to(
                        concentration.to(Device::cpu()), shape)
                        .contiguous();
    auto rate_cpu = ::tenzor::broadcast_to(
                        rate.to(Device::cpu()), shape)
                        .contiguous();

    if (out.dtype() == DType::Float32) {
        float* op = out.data<float>();
        const float* cp = conc_cpu.data<float>();
        const float* rp = rate_cpu.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            op[i] = sample_gamma_scalar<float>(cp[i], rp[i], rng);
        }
    } else if (out.dtype() == DType::Float64) {
        double* op = out.data<double>();
        const double* cp = conc_cpu.data<double>();
        const double* rp = rate_cpu.data<double>();
        for (int64_t i = 0; i < n; ++i) {
            op[i] = sample_gamma_scalar<double>(cp[i], rp[i], rng);
        }
    } else {
        throw std::runtime_error("Gamma: only Float32 and Float64 supported");
    }

    return out;
}

/**
 * @brief Monte-Carlo estimator for distribution moments / entropy.
 *
 * Used by distributions whose entropy / mean / variance has no closed form
 * (RelaxedBernoulli, RelaxedOneHotCategorical, LogisticNormal, LKJCholesky,
 * MixtureSameFamily, TransformedDistribution, ...). The MC estimator IS
 * the correct mathematical definition for those quantities; this is not a
 * fallback. Default sample count is 4096 which gives stable estimates with
 * standard error ~1/√N for bounded random variables.
 */
template <typename SampleFn>
inline auto mc_mean(SampleFn&& sample_fn, int N = 4096) -> Tensor {
    std::vector<Tensor> samples; samples.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) samples.push_back(sample_fn());
    auto stacked = ::tenzor::stack(
        std::span<const Tensor>(samples.data(), samples.size()), /*dim=*/0);
    return ::tenzor::mean(stacked, /*dim=*/0, /*keepdim=*/false);
}

template <typename SampleFn>
inline auto mc_variance(SampleFn&& sample_fn, int N = 4096) -> Tensor {
    std::vector<Tensor> samples; samples.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) samples.push_back(sample_fn());
    auto stacked = ::tenzor::stack(
        std::span<const Tensor>(samples.data(), samples.size()), /*dim=*/0);
    return ::tenzor::var(stacked, /*dim=*/0, /*keepdim=*/false,
                         /*unbiased=*/true);
}

/**
 * @brief Monte-Carlo entropy: H = -E[log p(X)] estimated as
 *   H ≈ -mean_i log_prob(sample_i)
 */
template <typename SampleFn, typename LogProbFn>
inline auto mc_entropy(SampleFn&& sample_fn, LogProbFn&& lp_fn, int N = 4096)
    -> Tensor
{
    std::vector<Tensor> lps; lps.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        auto s  = sample_fn();
        auto lp = lp_fn(s);
        lps.push_back(lp);
    }
    auto stacked = ::tenzor::stack(
        std::span<const Tensor>(lps.data(), lps.size()), /*dim=*/0);
    return ::tenzor::neg(::tenzor::mean(stacked, /*dim=*/0, /*keepdim=*/false));
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

    auto concentration() const -> const Tensor& { return concentration_; }
    auto rate() const -> const Tensor& { return rate_; }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(concentration_.shape().begin(), concentration_.shape().end())
            : sample_shape;
        auto result = detail::fill_gamma_cpu(concentration_, rate_, shape, detail::rng_for(generator_));
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

    /**
     * @brief CDF of Gamma(α, β) (audit E.5).
     *
     * F(x; α, β) = P(α, β·x) = igamma(α, β·x), where igamma is the
     * regularised lower incomplete gamma function (tenzor::igamma).
     */
    auto cdf(const Tensor& value) -> Tensor {
        return tenzor::igamma(concentration_, rate_ * value);
    }

    /**
     * @brief Inverse CDF of Gamma(α, β) via Newton iteration on the CDF
     *        (audit E.5).
     *
     * Newton step: x_{n+1} = x_n - (F(x_n) - q) / f(x_n), where
     *   f(x; α, β) = β^α · x^(α-1) · exp(-β·x) / Γ(α)
     * The Wilson–Hilferty cube-root approximation seeds the iteration
     * with a good first guess so convergence is reached in <10 steps
     * across the typical (α, q) range. We clamp x > 0 to stay inside
     * the support after each step.
     */
    auto icdf(const Tensor& q) -> Tensor {
        // Wilson–Hilferty seed: x0 ≈ α · (1 - 1/(9α) + Φ⁻¹(q)·√(1/(9α)))^3 / β
        auto qf = q.to(concentration_.dtype());
        auto a  = concentration_;
        auto b  = rate_;
        auto inv9a = 1.0f / (9.0f * a);
        auto z = tenzor::ndtri(qf);                    // standard-normal quantile
        auto base = 1.0f - inv9a + z * tenzor::sqrt(inv9a);
        auto x = a * (base * base * base) / b;
        x = tenzor::clamp(x, 1e-12f, std::numeric_limits<float>::infinity());

        // Newton refinement: x ← x - (F(x) - q) / pdf(x).
        constexpr int kIters = 10;
        auto log_norm = a * tenzor::log(b) - tenzor::lgamma(a); // log(β^α / Γ(α))
        for (int i = 0; i < kIters; ++i) {
            auto F  = tenzor::igamma(a, b * x);
            auto log_pdf = log_norm + (a - 1.0f) * tenzor::log(x) - b * x;
            auto pdf = tenzor::exp(log_pdf);
            auto pdf_safe = tenzor::clamp(pdf, 1e-30f,
                                          std::numeric_limits<float>::infinity());
            x = x - (F - qf) / pdf_safe;
            x = tenzor::clamp(x, 1e-12f, std::numeric_limits<float>::infinity());
        }
        return x;
    }

    // Wave Inf-B7: entropy = α - log(β) + lgamma(α) + (1-α)·ψ(α)
    // Matches torch.distributions.Gamma.entropy().
    auto entropy() -> Tensor override {
        auto one = tenzor::full(std::vector<int64_t>(concentration_.shape().begin(),
                                                       concentration_.shape().end()),
                                  1.0, concentration_.dtype(), concentration_.device());
        return concentration_
             - tenzor::log(rate_)
             + tenzor::lgamma(concentration_)
             + (one - concentration_) * tenzor::digamma(concentration_);
    }

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

    auto concentration1() const -> const Tensor& { return c1_; }
    auto concentration0() const -> const Tensor& { return c0_; }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(c1_.shape().begin(), c1_.shape().end())
            : sample_shape;
        auto one = full({1}, 1.0, c1_.dtype(), Device::cpu());
        auto x = detail::fill_gamma_cpu(c1_, one, shape, detail::rng_for(generator_));
        auto y = detail::fill_gamma_cpu(c0_, one, shape, detail::rng_for(generator_));
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

    /**
     * @brief Variance: Var[X] = (a * b) / ((a + b)^2 * (a + b + 1)).
     */
    auto variance() -> Tensor override {
        auto sum = c1_ + c0_;
        return (c1_ * c0_) / (sum * sum * (sum + 1.0f));
    }

    /**
     * @brief Entropy of Beta(α, β):
     * H = lgamma(α) + lgamma(β) - lgamma(α+β)
     *     - (α-1)·ψ(α) - (β-1)·ψ(β) + (α+β-2)·ψ(α+β)
     */
    auto entropy() -> Tensor override {
        auto sum = c1_ + c0_;
        auto log_B = tenzor::lgamma(c1_) + tenzor::lgamma(c0_) - tenzor::lgamma(sum);
        auto term = (c1_ - 1.0f) * tenzor::digamma(c1_)
                  + (c0_ - 1.0f) * tenzor::digamma(c0_)
                  - (sum - 2.0f) * tenzor::digamma(sum);
        return log_B - term;
    }

    /**
     * @brief CDF of Beta(α, β) (audit E.5).
     *
     * F(x; α, β) = I_x(α, β), the regularised incomplete beta function.
     */
    auto cdf(const Tensor& value) -> Tensor {
        auto v = tenzor::clamp(value, 0.0f, 1.0f);
        return tenzor::betainc(c1_, c0_, v);
    }

    /**
     * @brief Inverse CDF of Beta(α, β) via Newton iteration on betainc
     *        (audit E.5).
     *
     * Newton step:  x ← x - (F(x) - q) / pdf(x), with
     *   pdf(x; α, β) = x^(α-1) (1-x)^(β-1) / B(α, β).
     * Seed with the distribution mean α / (α + β); clamp to (0, 1) after
     * each step.
     */
    auto icdf(const Tensor& q) -> Tensor {
        auto qf = q.to(c1_.dtype());
        auto x = c1_ / (c1_ + c0_);                 // seed at the mean
        x = tenzor::clamp(x, 1e-6f, 1.0f - 1e-6f);

        // log B(α, β) = lgamma(α) + lgamma(β) - lgamma(α+β)
        auto log_B = tenzor::lgamma(c1_) + tenzor::lgamma(c0_)
                   - tenzor::lgamma(c1_ + c0_);

        constexpr int kIters = 25;
        for (int i = 0; i < kIters; ++i) {
            auto F = tenzor::betainc(c1_, c0_, x);
            auto log_pdf = (c1_ - 1.0f) * tenzor::log(x)
                         + (c0_ - 1.0f) * tenzor::log(1.0f - x)
                         - log_B;
            auto pdf = tenzor::exp(log_pdf);
            auto pdf_safe = tenzor::clamp(pdf, 1e-30f,
                                          std::numeric_limits<float>::infinity());
            x = x - (F - qf) / pdf_safe;
            x = tenzor::clamp(x, 1e-6f, 1.0f - 1e-6f);
        }
        return x;
    }

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

    auto concentration() const -> const Tensor& { return concentration_; }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(concentration_.shape().begin(), concentration_.shape().end())
            : sample_shape;
        auto one = full({1}, 1.0, concentration_.dtype(), Device::cpu());
        auto gammas = detail::fill_gamma_cpu(concentration_, one, shape, detail::rng_for(generator_));

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

    // Wave Inf-B1: Dirichlet closed forms.
    // mean = α / α₀  where α₀ = sum(α, -1, keepdim).
    auto mean() -> Tensor override {
        auto sum_alpha = tenzor::sum(concentration_, -1, /*keepdim=*/true);
        return concentration_ / sum_alpha;
    }

    // variance = α (α₀ - α) / (α₀² (α₀ + 1))
    auto variance() -> Tensor override {
        auto sum_alpha = tenzor::sum(concentration_, -1, /*keepdim=*/true);
        auto num = concentration_ * (sum_alpha - concentration_);
        auto den = sum_alpha * sum_alpha * (sum_alpha + 1.0f);
        return num / den;
    }

    // entropy = lbeta(α) + (α₀ - K)·ψ(α₀) − sum((αᵢ - 1)·ψ(αᵢ), -1)
    //   where lbeta(α) = sum(lgamma(α), -1) − lgamma(α₀) and K = α.shape[-1].
    // Matches torch.distributions.Dirichlet.entropy().
    auto entropy() -> Tensor override {
        auto sum_alpha = tenzor::sum(concentration_, -1, /*keepdim=*/false);
        auto lgamma_sum = tenzor::lgamma(sum_alpha);
        auto sum_lgamma = tenzor::sum(tenzor::lgamma(concentration_), -1, /*keepdim=*/false);
        // K = number of categories along the last dim.
        auto cshape = concentration_.shape();
        float K = static_cast<float>(cshape[cshape.size() - 1]);
        auto digamma_sum = tenzor::digamma(sum_alpha);
        auto sum_term = tenzor::sum(
            (concentration_ - 1.0f) * tenzor::digamma(concentration_), -1, /*keepdim=*/false);
        // lbeta(α) = sum_lgamma - lgamma_sum
        auto lbeta = sum_lgamma - lgamma_sum;
        return lbeta + (sum_alpha - K) * digamma_sum - sum_term;
    }

    /**
     * @brief cdf/icdf are not defined for Dirichlet (audit E.5).
     *
     * Dirichlet is supported on the (K-1)-simplex; there is no canonical
     * total order on the support, so no scalar cumulative distribution.
     */
    [[noreturn]] auto cdf(const Tensor& /*value*/) -> Tensor {
        throw ::tenzor::error::DistributionMethodUndefined(
            "Dirichlet::cdf is undefined: simplex-valued support has no "
            "canonical scalar order");
    }

    [[noreturn]] auto icdf(const Tensor& /*q*/) -> Tensor {
        throw ::tenzor::error::DistributionMethodUndefined(
            "Dirichlet::icdf is undefined: simplex-valued support has no "
            "canonical scalar order");
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
        auto chi2 = detail::fill_gamma_cpu(alpha, rate, shape, detail::rng_for(generator_));

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

    /**
     * @brief Variance of StudentT(df, loc, scale):
     *   Var = scale^2 * df / (df - 2)  for df > 2
     * For df <= 2 the variance is infinite or undefined; PyTorch returns
     * positive infinity for 1 < df <= 2 and NaN for df <= 1. We follow the
     * arithmetic naturally — when df <= 2 the formula yields +inf / -nan and
     * propagates through tensor ops.
     */
    auto variance() -> Tensor override {
        return scale_ * scale_ * df_ / (df_ - 2.0f);
    }

    /**
     * @brief Differential entropy:
     *   H = 0.5*(df+1)*(ψ((df+1)/2) - ψ(df/2))
     *     + 0.5*log(df*π) + lgamma(df/2) - lgamma((df+1)/2)
     *     + log(scale)
     */
    auto entropy() -> Tensor override {
        auto df_half = df_ * 0.5f;
        auto df_half_plus_half = df_half + 0.5f;
        auto log_pi = full({1}, static_cast<double>(std::log(M_PI)), df_.dtype(), df_.device());

        auto digamma_term = 0.5f * (df_ + 1.0f)
                          * (tenzor::digamma(df_half_plus_half) - tenzor::digamma(df_half));
        auto log_term     = 0.5f * (tenzor::log(df_) + log_pi);
        auto lgamma_term  = tenzor::lgamma(df_half) - tenzor::lgamma(df_half_plus_half);
        auto scale_term   = tenzor::log(scale_);
        return digamma_term + log_term + lgamma_term + scale_term;
    }

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

    auto rate() const -> const Tensor& { return rate_; }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(rate_.shape().begin(), rate_.shape().end())
            : sample_shape;

        auto rate_cpu = rate_.to(Device::cpu()).contiguous();
        // Widen Float16/BFloat16 to Float32 — sample_poisson_scalar takes
        // double; Float16 rate has insufficient range for typical λ values.
        if (rate_cpu.dtype() == DType::Float16 || rate_cpu.dtype() == DType::BFloat16) {
            rate_cpu = rate_cpu.to(DType::Float32);
        }
        auto out = zeros(shape, DType::Int64, Device::cpu());
        int64_t n = out.numel();
        const int64_t rate_n = rate_cpu.numel();
        int64_t* op = out.data<int64_t>();
        auto& rng = detail::rng_for(generator_);

        if (rate_cpu.dtype() == DType::Float32) {
            const float* rp = rate_cpu.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                double lam = (rate_n == 1) ? rp[0] : rp[i % rate_n];
                op[i] = detail::sample_poisson_scalar(lam, rng);
            }
        } else if (rate_cpu.dtype() == DType::Float64) {
            const double* rp = rate_cpu.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                double lam = (rate_n == 1) ? rp[0] : rp[i % rate_n];
                op[i] = detail::sample_poisson_scalar(lam, rng);
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

    /**
     * @brief CDF of Poisson(λ) (audit E.5).
     *
     * F(k; λ) = P(X ≤ floor(k)) = Q(floor(k) + 1, λ), where Q is the
     * regularised upper incomplete gamma. ``tenzor::igammac`` implements Q.
     * For k < 0 the CDF is 0 by convention.
     */
    auto cdf(const Tensor& value) -> Tensor {
        auto v = tenzor::floor(value.to(rate_.dtype()));
        auto out = tenzor::igammac(v + 1.0f, rate_);
        auto zeros_t = tenzor::zeros_like(out);
        // For k < 0, CDF is 0.
        return tenzor::where(v < zeros_t, zeros_t, out);
    }

    /**
     * @brief Inverse CDF of Poisson(λ) (audit E.5).
     *
     * Discrete distribution: scan k = 0, 1, ... per element on CPU, then
     * return the first k with F(k) >= q. We widen Float16/BFloat16 to
     * Float32 for the scan and narrow back at the end. The search cutoff
     * K = λ + 30·√λ + 50 covers the upper tail to well below float
     * precision.
     */
    auto icdf(const Tensor& q) -> Tensor {
        auto orig_dtype = rate_.dtype();
        auto orig_device = rate_.device();
        const bool widen_for_f16 = (orig_dtype == DType::Float16 ||
                                    orig_dtype == DType::BFloat16);

        auto rate_cpu = rate_.to(Device::cpu()).contiguous();
        auto q_cpu    = q.to(Device::cpu()).contiguous();
        if (widen_for_f16) {
            rate_cpu = rate_cpu.to(DType::Float32);
            q_cpu    = q_cpu.to(DType::Float32);
        }

        const int64_t rn = rate_cpu.numel();
        const int64_t qn = q_cpu.numel();
        const int64_t n  = std::max(rn, qn);
        auto out_dtype = widen_for_f16 ? DType::Float32 : orig_dtype;
        auto out = zeros({n}, out_dtype, Device::cpu());

        auto compute = [&](auto* op, const auto* rp, const auto* qp) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(op)>>;
            for (int64_t i = 0; i < n; ++i) {
                double lam  = static_cast<double>(rp[i % rn]);
                double qval = static_cast<double>(qp[i % qn]);
                if (qval <= 0.0) { op[i] = T{0}; continue; }
                if (qval >= 1.0) { op[i] = static_cast<T>(
                    std::numeric_limits<double>::infinity()); continue; }
                const int64_t K = static_cast<int64_t>(
                    std::max<double>(50.0, lam + 30.0 * std::sqrt(std::max(lam, 1.0))));
                // Streaming sum: P(X=k) = P(X=k-1) * lam / k.
                double pk = std::exp(-lam);   // P(X=0)
                double cum = pk;
                int64_t found = K;
                for (int64_t k = 0; k <= K; ++k) {
                    if (cum >= qval) { found = k; break; }
                    pk *= lam / static_cast<double>(k + 1);
                    cum += pk;
                }
                op[i] = static_cast<T>(found);
            }
        };

        if (out_dtype == DType::Float32) {
            compute(out.data<float>(), rate_cpu.data<float>(), q_cpu.data<float>());
        } else if (out_dtype == DType::Float64) {
            compute(out.data<double>(), rate_cpu.data<double>(), q_cpu.data<double>());
        } else {
            throw std::runtime_error("Poisson::icdf: rate must be Float32 or Float64");
        }

        // Reshape to broadcast shape (use larger of rate/q shapes).
        auto target_shape = (rn >= qn ? rate_ : q).shape();
        auto reshaped = out.reshape(
            std::vector<int64_t>(target_shape.begin(), target_shape.end()));
        if (widen_for_f16) {
            reshaped = reshaped.to(orig_dtype);
        }
        return reshaped.to(orig_device);
    }

    // Inf-B9 (deferred → landed): Poisson entropy has no closed form.
    // Use Stirling expansion for large λ (≥ 10):
    //   H(λ) ≈ 0.5 · log(2πeλ) - 1/(12λ) - 1/(24λ²) - 19/(360λ³)
    // For small λ use truncated -Σ p(k) log p(k) with cutoff
    // K = max(20, λ + 15·√λ) so tail probability < 1e-12.
    // Reference: torch/distributions/poisson.py::entropy (matches within 1e-6).
    auto entropy() -> Tensor override {
        auto r_cpu = rate_.to(Device::cpu()).to(DType::Float64).contiguous();
        const int64_t n = r_cpu.numel();
        auto out = zeros({n}, DType::Float32, Device::cpu());
        const double* rp = r_cpu.data<double>();
        float* op = out.data<float>();
        constexpr double kStirlingCutoff = 10.0;
        for (int64_t i = 0; i < n; ++i) {
            const double lam = std::max(rp[i], 0.0);
            if (lam == 0.0) { op[i] = 0.0f; continue; }
            double h = 0.0;
            if (lam >= kStirlingCutoff) {
                // Stirling asymptotic series.
                constexpr double kLog2Pi = 1.8378770664093454835606594728112352798;
                h = 0.5 * (kLog2Pi + 1.0 + std::log(lam))
                  - 1.0 / (12.0 * lam)
                  - 1.0 / (24.0 * lam * lam)
                  - 19.0 / (360.0 * lam * lam * lam);
            } else {
                // Truncated sum.
                const double log_lam = std::log(lam);
                const int64_t K = static_cast<int64_t>(
                    std::max<double>(20.0, lam + 15.0 * std::sqrt(lam)));
                for (int64_t k = 0; k <= K; ++k) {
                    const double log_pk = static_cast<double>(k) * log_lam
                                        - lam
                                        - std::lgamma(static_cast<double>(k + 1));
                    const double pk = std::exp(log_pk);
                    if (pk > 0.0) h -= pk * log_pk;
                }
            }
            op[i] = static_cast<float>(h);
        }
        auto target_shape = rate_.shape();
        return out.reshape(std::vector<int64_t>(target_shape.begin(), target_shape.end()))
                  .to(rate_.device());
    }

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

    auto loc() const -> const Tensor& { return loc_; }
    auto covariance_matrix() const -> const Tensor& { return cov_; }
    auto scale_tril() const -> const Tensor& { return scale_tril_; }

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
        // log_prob(x) = -0.5 * (D * log(2*pi) + log|cov| + (x-mu)^T cov^{-1} (x-mu))
        //
        // We Cholesky-factor cov = L L^T at construction (scale_tril_, lower).
        // Then:
        //   * log|cov| = 2 * sum(log(diag(L)))
        //   * (x-mu)^T cov^{-1} (x-mu) = ||L^{-1} (x-mu)||_2^2
        //
        // Supports unbatched value (shape (D,)) and arbitrarily batched value
        // (shape (..., D)). scale_tril_ stays unbatched (D, D); the batched
        // triangular solve broadcasts it across the leading dims.
        const int64_t D = loc_.shape().back();
        auto diff = value - loc_;                              // (..., D) or (D,)

        // Promote to column (..., D, 1) for triangular solve.
        auto diff_col = unsqueeze(diff, -1);                   // (..., D, 1)

        // Solve L y = diff (lower triangular).  scale_tril_ is (D, D), diff_col
        // is (..., D, 1); solve_triangular broadcasts the system over the
        // leading batch dims.
        auto y_col = linalg::solve_triangular(scale_tril_, diff_col,
                                              /*upper=*/false);  // (..., D, 1)
        // Drop the trailing 1 axis: (..., D, 1) → (..., D).
        const auto& diff_shape = diff.shape();
        std::vector<int64_t> y_shape(diff_shape.begin(), diff_shape.end());
        auto y = y_col.reshape(y_shape);

        // Mahalanobis term = sum over last axis of y^2 → shape (...,).
        auto mahal = sum(y * y, /*dim=*/static_cast<int64_t>(-1));

        // log|cov| = 2 * sum(log(diag(L))). scale_tril_ is 2D so diag() returns
        // a 1D vector of the diagonal elements — kept on-device throughout.
        auto diag_L      = diag(scale_tril_);                       // (D,)
        auto log_det_scl = sum(log(diag_L)) * full_like(sum(log(diag_L)), 2.0);

        // Constant term: -0.5 * (D * log(2π) + log_det)
        auto two_pi  = full_like(log_det_scl,
                                 static_cast<double>(D) * std::log(2.0 * M_PI));
        auto const_t = (two_pi + log_det_scl) * full_like(log_det_scl, -0.5);

        // Broadcast the scalar constant against the (...,) Mahalanobis tensor.
        auto half_mahal = mahal * full_like(mahal, 0.5);
        return const_t - half_mahal;
    }

    auto mean() -> Tensor override { return loc_; }

    /**
     * @brief Variance: diagonal of the covariance matrix.
     *
     * For Σ shaped (..., D, D), returns the (..., D) tensor of per-coordinate
     * marginal variances Var[X_i] = Σ_ii.
     */
    auto variance() -> Tensor override {
        // Diagonal of covariance Σ; for batched cov_ ((..., D, D)) we'd need a
        // batched diagonal helper. tenzor::diag is 1D/2D; gate accordingly.
        if (cov_.ndim() != 2) {
            throw std::runtime_error(
                "MultivariateNormal::variance: batched covariance not supported");
        }
        return tenzor::diag(cov_);
    }

    /**
     * @brief Differential entropy:
     *   H = 0.5 * (D * (1 + log(2π)) + log|Σ|)
     * where log|Σ| = 2 * Σ_i log(L_ii) via the Cholesky factor.
     */
    auto entropy() -> Tensor override {
        if (scale_tril_.ndim() != 2) {
            throw std::runtime_error(
                "MultivariateNormal::entropy: batched covariance not supported");
        }
        const int64_t D = loc_.shape().back();
        auto diag_L  = tenzor::diag(scale_tril_);   // (D,)
        auto log_det = tenzor::sum(tenzor::log(diag_L)) * 2.0f;
        const double base = 0.5 * static_cast<double>(D) * (1.0 + std::log(2.0 * M_PI));
        auto base_t = full({1}, base, loc_.dtype(), loc_.device());
        return base_t + log_det * 0.5f;
    }

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

        // Draw each Binomial(total_count, p) count directly with an exact
        // per-element scalar sampler (geometric-gap inversion). This replaces
        // the previous O(total_count) loop of full-tensor Bernoulli sums —
        // which allocated two tensors per trial — with a single output buffer
        // and O(total_count·min(p,1-p)) expected scalar work per element.
        auto orig_dtype = probs_.dtype();
        auto orig_device = probs_.device();
        const bool widen_for_f16 = (orig_dtype == DType::Float16 ||
                                    orig_dtype == DType::BFloat16);
        auto work_dtype = widen_for_f16 ? DType::Float32 : orig_dtype;

        // Broadcast probs to the requested output shape on CPU so each output
        // element is paired with the correct probability.
        auto probs_cpu = ::tenzor::broadcast_to(
                             probs_.to(Device::cpu()), shape)
                             .to(work_dtype)
                             .contiguous();

        auto out = zeros(shape, work_dtype, Device::cpu());
        const int64_t n = out.numel();
        auto& rng = detail::rng_for(generator_);

        if (work_dtype == DType::Float32) {
            float* op = out.data<float>();
            const float* pp = probs_cpu.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                op[i] = static_cast<float>(detail::sample_binomial_scalar(
                    total_count_, static_cast<double>(pp[i]), rng));
            }
        } else if (work_dtype == DType::Float64) {
            double* op = out.data<double>();
            const double* pp = probs_cpu.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                op[i] = static_cast<double>(detail::sample_binomial_scalar(
                    total_count_, pp[i], rng));
            }
        } else {
            throw std::runtime_error("Binomial: only Float32 and Float64 supported");
        }

        if (widen_for_f16) {
            out = out.to(orig_dtype);
        }
        return out.to(orig_device);
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

    /**
     * @brief CDF of Binomial(n, p) (audit E.5).
     *
     * F(k) = P(X ≤ floor(k)) = I_{1-p}(n - floor(k), floor(k) + 1)
     * where I_x(a, b) is the regularised incomplete beta function.
     * For k < 0, F = 0; for k >= n, F = 1.
     */
    auto cdf(const Tensor& value) -> Tensor {
        auto v_raw = value.to(probs_.dtype());
        auto v = tenzor::floor(v_raw);
        auto n_scalar = full(std::vector<int64_t>(probs_.shape().begin(),
                                                   probs_.shape().end()),
                              static_cast<double>(total_count_),
                              probs_.dtype(), probs_.device());
        // Clamp arguments to keep betainc inside its domain.
        auto a = tenzor::clamp(n_scalar - v, 1e-6f,
                                static_cast<float>(total_count_) + 1.0f);
        auto b = v + 1.0f;
        auto x = 1.0f - probs_;
        auto out = tenzor::betainc(a, b, x);
        // k < 0  → 0;  k >= n → 1.
        auto zeros_t = tenzor::zeros_like(out);
        auto ones_t  = zeros_t + 1.0f;
        out = tenzor::where(v < zeros_t, zeros_t, out);
        auto n_t  = tenzor::full_like(out, static_cast<double>(total_count_));
        out = tenzor::where(v >= n_t, ones_t, out);
        return out;
    }

    /**
     * @brief Inverse CDF of Binomial(n, p) (audit E.5).
     *
     * Discrete distribution: scan k = 0..n per element on CPU, return the
     * first k with F(k) >= q. The element count is bounded by n, so this is
     * O(n) per element and matches the discrete-Newton fallbacks used in
     * SciPy / torch.distributions.
     */
    auto icdf(const Tensor& q) -> Tensor {
        auto orig_dtype = probs_.dtype();
        auto orig_device = probs_.device();
        const bool widen_for_f16 = (orig_dtype == DType::Float16 ||
                                    orig_dtype == DType::BFloat16);

        auto p_cpu = probs_.to(Device::cpu()).contiguous();
        auto q_cpu = q.to(Device::cpu()).contiguous();
        if (widen_for_f16) {
            p_cpu = p_cpu.to(DType::Float32);
            q_cpu = q_cpu.to(DType::Float32);
        }

        const int64_t pn = p_cpu.numel();
        const int64_t qn = q_cpu.numel();
        const int64_t n  = std::max(pn, qn);
        auto out_dtype = widen_for_f16 ? DType::Float32 : orig_dtype;
        auto out = zeros({n}, out_dtype, Device::cpu());

        auto compute = [&](auto* op, const auto* pp, const auto* qp) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(op)>>;
            const int64_t N = total_count_;
            for (int64_t i = 0; i < n; ++i) {
                double pv   = std::clamp(static_cast<double>(pp[i % pn]), 0.0, 1.0);
                double qval = static_cast<double>(qp[i % qn]);
                if (qval <= 0.0) { op[i] = T{0}; continue; }
                if (qval >= 1.0) { op[i] = static_cast<T>(N); continue; }
                // P(X=0) = (1-p)^N; streaming P(X=k) = P(X=k-1) * (N-k+1)/k * p/(1-p).
                double log_pmf0 = static_cast<double>(N) * std::log(1.0 - pv);
                double pk = std::exp(log_pmf0);
                double cum = pk;
                int64_t found = N;
                const double ratio_p = pv / std::max(1.0 - pv, 1e-300);
                for (int64_t k = 0; k <= N; ++k) {
                    if (cum >= qval) { found = k; break; }
                    pk *= ratio_p * static_cast<double>(N - k)
                                  / static_cast<double>(k + 1);
                    cum += pk;
                }
                op[i] = static_cast<T>(found);
            }
        };

        if (out_dtype == DType::Float32) {
            compute(out.data<float>(), p_cpu.data<float>(), q_cpu.data<float>());
        } else if (out_dtype == DType::Float64) {
            compute(out.data<double>(), p_cpu.data<double>(), q_cpu.data<double>());
        } else {
            throw std::runtime_error("Binomial::icdf: probs must be Float32 or Float64");
        }

        auto target_shape = (pn >= qn ? probs_ : q).shape();
        auto reshaped = out.reshape(
            std::vector<int64_t>(target_shape.begin(), target_shape.end()));
        if (widen_for_f16) {
            reshaped = reshaped.to(orig_dtype);
        }
        return reshaped.to(orig_device);
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

    auto loc() const -> const Tensor& { return loc_; }
    auto scale() const -> const Tensor& { return scale_; }

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
        // Use double literals so Float64 LogNormal keeps full precision
        // (0.5*log(2*pi) = 0.91893853320467274178...); a float literal capped it
        // at ~1e-7.
        return loc_ + 0.5 + tenzor::log(scale_) + 0.91893853320467274178;
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

    /**
     * @brief CDF of LogNormal(μ, σ) (audit E.5).
     *
     * F(x; μ, σ) = Φ((log x - μ) / σ) = 0.5 · [1 + erf((log x - μ) / (σ√2))].
     */
    auto cdf(const Tensor& value) -> Tensor {
        auto z = (tenzor::log(value) - loc_) / scale_;
        constexpr float kInvSqrt2 = 0.70710678118654752440f;
        return 0.5f * (1.0f + tenzor::erf(z * kInvSqrt2));
    }

    /**
     * @brief Inverse CDF of LogNormal(μ, σ) (audit E.5).
     *
     * icdf(q) = exp(μ + σ · Φ⁻¹(q)) — probit composition via tenzor::ndtri.
     */
    auto icdf(const Tensor& q) -> Tensor {
        auto qf = q.to(loc_.dtype());
        return tenzor::exp(loc_ + scale_ * tenzor::ndtri(qf));
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

    // Wave Inf-B8: Chi2(df) = Gamma(df/2, 1/2), so entropy delegates.
    auto entropy() -> Tensor override { return gamma_.entropy(); }

    /**
     * @brief CDF / ICDF of Chi2(df) (audit E.5).
     *
     * Chi2(df) = Gamma(df/2, 1/2). Delegate to Gamma which uses
     * tenzor::igamma (CDF) and Newton-iterated inverse igamma (ICDF).
     */
    auto cdf(const Tensor& value) -> Tensor { return gamma_.cdf(value); }
    auto icdf(const Tensor& q)     -> Tensor { return gamma_.icdf(q); }

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

    /**
     * @brief CDF of Geometric(p) (audit E.5).
     *
     * Tenzor's Geometric uses the 1-indexed convention (X >= 1). For integer
     * k >= 1:
     *   F(k; p) = 1 - (1 - p)^floor(k)
     * Returns 0 for k < 1.
     */
    auto cdf(const Tensor& value) -> Tensor {
        auto v = tenzor::floor(value.to(probs_.dtype()));
        auto q = 1.0f - tenzor::clamp(probs_, 0.0f, 1.0f);
        // 1 - q^k = 1 - exp(k * log(q)). For p=1 (q=0) this is 1 for k>=1.
        auto cdf_pos = 1.0f - tenzor::exp(v * tenzor::log(tenzor::clamp(
            q, 1e-30f, 1.0f)));
        auto zeros_t = tenzor::zeros_like(cdf_pos);
        auto ones_t  = zeros_t + 1.0f;
        return tenzor::where(v < ones_t, zeros_t, cdf_pos);
    }

    /**
     * @brief Inverse CDF of Geometric(p) (audit E.5).
     *
     * icdf(q) = ceil(log(1 - q) / log(1 - p)) for q in (0, 1).
     */
    auto icdf(const Tensor& q) -> Tensor {
        auto qf = q.to(probs_.dtype());
        constexpr float kEps = 1e-7f;
        auto q_clamped = tenzor::clamp(qf, kEps, 1.0f - kEps);
        auto p_clamped = tenzor::clamp(probs_, kEps, 1.0f - kEps);
        auto val = tenzor::log(1.0f - q_clamped)
                 / tenzor::log(1.0f - p_clamped);
        return tenzor::ceil(val);
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

    // Wave Inf-B10: differential entropy of HalfCauchy = log(2π·scale).
    // (Half-Cauchy has finite entropy despite undefined mean/variance.)
    // Source: Wolfram MathWorld § Half-Cauchy distribution.
    auto entropy() -> Tensor override {
        // entropy = log(2π) + log(scale) = log(scale) + log(2*M_PI)
        return tenzor::log(scale_) +
               static_cast<float>(std::log(2.0 * M_PI));
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
        // Broadcast the degrees-of-freedom tensors to the full output shape so
        // that both the chi-square gamma draws and the subsequent divisions
        // pair each sampled element with the correct df value. Without this,
        // a sample_shape that differs from the df shape mispairs elements or
        // raises a broadcast error in the divisions below.
        auto df1_cpu = ::tenzor::broadcast_to(df1_.to(Device::cpu()), shape);
        auto df2_cpu = ::tenzor::broadcast_to(df2_.to(Device::cpu()), shape);
        auto x1 = detail::fill_gamma_cpu(df1_cpu * 0.5f, one, shape, detail::rng_for(generator_));
        auto x2 = detail::fill_gamma_cpu(df2_cpu * 0.5f, one, shape, detail::rng_for(generator_));
        auto result = (x1 / df1_cpu) / (x2 / df2_cpu);
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

    // Inf-B11 (deferred → landed): closed-form differential entropy.
    // H[F(d1, d2)] = log Beta(d1/2, d2/2) + (1 - d1/2) ψ(d1/2)
    //              - (1 + d2/2) ψ(d2/2) + (d1+d2)/2 · ψ((d1+d2)/2)
    //              + log(d2/d1)
    // Reference: scipy.stats.f.entropy, matches PyTorch's
    // FisherSnedecor.entropy() in scipy/scipy/stats/_continuous_distns.py.
    auto entropy() -> Tensor override {
        auto d1h = df1_ * 0.5f;
        auto d2h = df2_ * 0.5f;
        auto sum_h = d1h + d2h;
        // log Beta(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)
        auto log_beta = tenzor::lgamma(d1h) + tenzor::lgamma(d2h)
                      - tenzor::lgamma(sum_h);
        return log_beta
             + (1.0f - d1h) * tenzor::digamma(d1h)
             - (1.0f + d2h) * tenzor::digamma(d2h)
             + sum_h * tenzor::digamma(sum_h)
             + tenzor::log(df2_ / df1_);
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
        auto& rng = detail::rng_for(generator_);
        auto rate_samples = detail::fill_gamma_cpu(
            total_count_.to(Device::cpu()),
            tenzor::reciprocal(gamma_rate.to(Device::cpu())),  // Gamma uses rate param
            shape, rng);

        // Draw Poisson(rate) counts
        auto rate_cpu = rate_samples.contiguous();
        // Widen Float16/BFloat16 — sample_poisson_scalar takes double.
        if (rate_cpu.dtype() == DType::Float16 || rate_cpu.dtype() == DType::BFloat16) {
            rate_cpu = rate_cpu.to(DType::Float32);
        }
        auto out = zeros(shape, DType::Int64, Device::cpu());
        int64_t n = out.numel();
        const int64_t rate_n = rate_cpu.numel();
        int64_t* op = out.data<int64_t>();

        if (rate_cpu.dtype() == DType::Float32) {
            const float* rp = rate_cpu.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                double lam = (rate_n == 1) ? rp[0] : rp[i % rate_n];
                op[i] = detail::sample_poisson_scalar(std::max(lam, 0.0), rng);
            }
        } else if (rate_cpu.dtype() == DType::Float64) {
            const double* rp = rate_cpu.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                double lam = (rate_n == 1) ? rp[0] : rp[i % rate_n];
                op[i] = detail::sample_poisson_scalar(std::max(lam, 0.0), rng);
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

    /**
     * @brief CDF of NegativeBinomial(r, p) (audit E.5).
     *
     * F(k; r, p) = I_{1-p}(r, floor(k) + 1), where I_x(a, b) is the
     * regularised incomplete beta function. Returns 0 for k < 0.
     */
    auto cdf(const Tensor& value) -> Tensor {
        auto v = tenzor::floor(value.to(probs_.dtype()));
        auto p_safe = tenzor::clamp(probs_, 1e-7f, 1.0f - 1e-7f);
        auto x = 1.0f - p_safe;
        auto a = total_count_;
        auto b = v + 1.0f;
        auto out = tenzor::betainc(a, b, x);
        auto zeros_t = tenzor::zeros_like(out);
        return tenzor::where(v < zeros_t, zeros_t, out);
    }

    /**
     * @brief Inverse CDF of NegativeBinomial(r, p) (audit E.5).
     *
     * Discrete distribution: scan k = 0, 1, ... per element on CPU until the
     * cumulative PMF exceeds q. The streaming PMF recursion
     *   P(X=k+1) = P(X=k) * (k + r) / (k + 1) * p
     * is stable across the whole tail; cutoff K = mean + 30·std + 50
     * bounds the truncation error below float precision.
     */
    auto icdf(const Tensor& q) -> Tensor {
        auto r_cpu = total_count_.to(Device::cpu()).contiguous();
        auto p_cpu = probs_.to(Device::cpu()).contiguous();
        auto q_cpu = q.to(Device::cpu()).contiguous();
        auto orig_dtype = probs_.dtype();
        auto orig_device = probs_.device();
        const bool widen_for_f16 = (orig_dtype == DType::Float16 ||
                                    orig_dtype == DType::BFloat16);
        if (widen_for_f16) {
            r_cpu = r_cpu.to(DType::Float32);
            p_cpu = p_cpu.to(DType::Float32);
            q_cpu = q_cpu.to(DType::Float32);
        }

        const int64_t rn = r_cpu.numel();
        const int64_t pn = p_cpu.numel();
        const int64_t qn = q_cpu.numel();
        const int64_t n  = std::max({rn, pn, qn});
        auto out_dtype = widen_for_f16 ? DType::Float32 : orig_dtype;
        auto out = zeros({n}, out_dtype, Device::cpu());

        auto compute = [&](auto* op, const auto* rp, const auto* pp,
                           const auto* qp) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(op)>>;
            for (int64_t i = 0; i < n; ++i) {
                double r    = static_cast<double>(rp[i % rn]);
                double pv   = std::clamp(static_cast<double>(pp[i % pn]),
                                          1e-12, 1.0 - 1e-12);
                double qval = static_cast<double>(qp[i % qn]);
                if (qval <= 0.0) { op[i] = T{0}; continue; }
                if (qval >= 1.0) { op[i] = static_cast<T>(
                    std::numeric_limits<double>::infinity()); continue; }
                double q_ = 1.0 - pv;
                double mean = r * pv / q_;
                double var  = r * pv / (q_ * q_);
                int64_t K = static_cast<int64_t>(
                    std::max<double>(50.0, mean + 30.0 * std::sqrt(var)));
                // P(X=0) = (1-p)^r = q^r.
                double pk = std::exp(r * std::log(q_));
                double cum = pk;
                int64_t found = K;
                for (int64_t k = 0; k <= K; ++k) {
                    if (cum >= qval) { found = k; break; }
                    pk *= (static_cast<double>(k) + r)
                        / static_cast<double>(k + 1) * pv;
                    cum += pk;
                }
                op[i] = static_cast<T>(found);
            }
        };

        if (out_dtype == DType::Float32) {
            compute(out.data<float>(), r_cpu.data<float>(),
                    p_cpu.data<float>(), q_cpu.data<float>());
        } else if (out_dtype == DType::Float64) {
            compute(out.data<double>(), r_cpu.data<double>(),
                    p_cpu.data<double>(), q_cpu.data<double>());
        } else {
            throw std::runtime_error(
                "NegativeBinomial::icdf: probs must be Float32 or Float64");
        }

        auto target_shape = (pn >= qn ? probs_ : q).shape();
        if (rn > std::max(pn, qn)) target_shape = total_count_.shape();
        auto reshaped = out.reshape(
            std::vector<int64_t>(target_shape.begin(), target_shape.end()));
        if (widen_for_f16) {
            reshaped = reshaped.to(orig_dtype);
        }
        return reshaped.to(orig_device);
    }

    // Inf-B12 (deferred → landed): closed-form entropy is not available
    // for NegativeBinomial. Compute via truncated -Σ P(k) log P(k); the
    // distribution's tail decays exponentially, so a fixed cutoff
    // K = max(50, mean + 20·std) bounds the truncation error well below
    // float precision. CPU-only, element-wise.
    auto entropy() -> Tensor override {
        auto r_cpu = total_count_.to(Device::cpu()).to(DType::Float64).contiguous();
        auto p_cpu = probs_.to(Device::cpu()).to(DType::Float64).contiguous();
        const int64_t n = std::max(r_cpu.numel(), p_cpu.numel());
        const int64_t rn = r_cpu.numel();
        const int64_t pn = p_cpu.numel();
        auto out = zeros({n}, DType::Float32, Device::cpu());
        const double* rp = r_cpu.data<double>();
        const double* pp = p_cpu.data<double>();
        float* op = out.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            const double r = rp[i % rn];
            const double p = std::clamp(pp[i % pn], 1e-7, 1.0 - 1e-7);
            const double q = 1.0 - p;
            const double log_p = std::log(p);
            const double log_q = std::log(q);
            // Adaptive cutoff: mean + 20·std + 50.
            const double mean = r * p / q;
            const double var = r * p / (q * q);
            const double std_ = std::sqrt(var);
            const int64_t K = static_cast<int64_t>(
                std::max<double>(50.0, mean + 20.0 * std_));
            const double lgamma_r = std::lgamma(r);
            double h = 0.0;
            for (int64_t k = 0; k <= K; ++k) {
                // log P(k) = lgamma(k + r) - lgamma(r) - lgamma(k+1) + r·log(q) + k·log(p)
                const double log_pk = std::lgamma(static_cast<double>(k) + r)
                                    - lgamma_r
                                    - std::lgamma(static_cast<double>(k + 1))
                                    + r * log_q + static_cast<double>(k) * log_p;
                const double pk = std::exp(log_pk);
                if (pk > 0.0) h -= pk * log_pk;
            }
            op[i] = static_cast<float>(h);
        }
        // Reshape to broadcast shape (use larger of r/p shapes).
        auto target_shape = (rn >= pn ? total_count_ : probs_).shape();
        return out.reshape(std::vector<int64_t>(target_shape.begin(), target_shape.end()))
                  .to(probs_.device());
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

        // Best & Fisher (1979) rejection sampling on CPU.
        // Widen Float16/BFloat16: rejection sampling needs double precision
        // for the K = 1 + sqrt(1 + 4·kappa²) tau computation; Float16 loses
        // it in the kappa² overflow at moderate concentration values.
        auto orig_dtype = loc_.dtype();
        const bool widen_for_f16 = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
        auto kappa_cpu = concentration_.to(Device::cpu()).contiguous();
        auto loc_cpu = loc_.to(Device::cpu()).contiguous();
        if (widen_for_f16) {
            kappa_cpu = kappa_cpu.to(DType::Float32);
            loc_cpu = loc_cpu.to(DType::Float32);
        }
        auto out_dtype = widen_for_f16 ? DType::Float32 : orig_dtype;
        auto out = zeros(shape, out_dtype, Device::cpu());
        int64_t n = out.numel();
        const int64_t kappa_n = kappa_cpu.numel();
        const int64_t loc_n = loc_cpu.numel();

        auto& rng = detail::rng_for(generator_);
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

        // Narrow back to original dtype if we widened for Float16/BFloat16.
        if (widen_for_f16) {
            out = out.to(orig_dtype);
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

    // Wave Inf-B13: differential entropy of VonMises:
    //   H = log(2π·I₀(κ)) − κ · I₁(κ) / I₀(κ)
    // Reuses existing Bessel-I helpers.
    auto entropy() -> Tensor override {
        auto i0 = tenzor::bessel_i0(concentration_);
        auto i1 = tenzor::bessel_i1(concentration_);
        return tenzor::log(i0) + static_cast<float>(std::log(2.0 * M_PI))
             - concentration_ * (i1 / i0);
    }

    /**
     * @brief CDF of VonMises(μ, κ) on [-π, π] (audit E.5).
     *
     * Uses Marsaglia (1965) Fourier-series expansion of the wrapped CDF:
     *   F(x; μ, κ) = (x - μ + π) / (2π)
     *              + (1 / (π I₀(κ))) · Σ_{k=1..N} I_k(κ) · sin(k (x - μ)) / k
     * The series converges in O(κ) terms; we use N = 50 which is well past
     * convergence for κ ≤ 50 (precision ≈ 1e-7 for moderate κ). Evaluated
     * scalar-wise on CPU; broadcasts across batched (loc, kappa, value)
     * inputs via the standard element-modulo loop pattern used elsewhere
     * in this header.
     */
    auto cdf(const Tensor& value) -> Tensor {
        auto orig_dtype = loc_.dtype();
        auto orig_device = loc_.device();
        const bool widen_for_f16 = (orig_dtype == DType::Float16 ||
                                    orig_dtype == DType::BFloat16);
        auto loc_cpu = loc_.to(Device::cpu()).contiguous();
        auto kappa_cpu = concentration_.to(Device::cpu()).contiguous();
        auto val_cpu = value.to(Device::cpu()).contiguous();
        if (widen_for_f16) {
            loc_cpu = loc_cpu.to(DType::Float32);
            kappa_cpu = kappa_cpu.to(DType::Float32);
            val_cpu = val_cpu.to(DType::Float32);
        }

        const int64_t ln = loc_cpu.numel();
        const int64_t kn = kappa_cpu.numel();
        const int64_t vn = val_cpu.numel();
        const int64_t n  = std::max({ln, kn, vn});
        auto out_dtype = widen_for_f16 ? DType::Float32 : orig_dtype;
        auto out = zeros({n}, out_dtype, Device::cpu());

        auto compute = [&](auto* op, const auto* lp, const auto* kp,
                           const auto* vp) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(op)>>;
            constexpr int kMaxTerms = 50;
            const int M = kMaxTerms + 20;
            // Hoist the Miller-recurrence scratch buffer out of the per-element
            // loop; it is overwritten each iteration so a single allocation is
            // reused across all n elements instead of churning the heap.
            std::vector<double> r(M + 1, 0.0);
            // Cache the I_k/I_0 ratios keyed on κ: consecutive elements often
            // share the same concentration (kn == 1 is the common case), so we
            // skip recomputing the full recurrence when κ is unchanged.
            double cached_kappa = std::numeric_limits<double>::quiet_NaN();
            for (int64_t i = 0; i < n; ++i) {
                double mu    = static_cast<double>(lp[i % ln]);
                double kappa = static_cast<double>(kp[i % kn]);
                double x     = static_cast<double>(vp[i % vn]);
                double dx = std::remainder(x - mu, 2.0 * M_PI);   // wrap to (-π, π]
                // Series: only meaningful for κ > 0; uniform limit for κ→0.
                double base = (dx + M_PI) / (2.0 * M_PI);
                if (kappa < 1e-6) {
                    op[i] = static_cast<T>(std::clamp(base, 0.0, 1.0));
                    continue;
                }
                // Compute I_k(κ) / I_0(κ) ratios via the recurrence
                //   I_{k+1}(κ) = I_{k-1}(κ) - 2k/κ · I_k(κ).
                // Start from a stable downward Miller recurrence; ratio
                // formulation avoids overflow.
                // We compute r_k = I_k(κ) / I_0(κ) by Miller's algorithm:
                //   start at high index with r_M = 0, r_{M-1} = 1 (unnormalised),
                //   recurse down, then normalise against the resulting r_0.
                // The ratios depend only on κ, so we recompute (into the reused
                // buffer) only when κ changes.
                if (kappa != cached_kappa) {
                    // The recurrence below writes every index 0..M (r[M] and
                    // r[M-1] explicitly, then r[k-1] for k = M-1..1), so no
                    // pre-zeroing of the reused buffer is required.
                    r[M] = 0.0;
                    r[M - 1] = 1.0;
                    for (int k = M - 1; k > 0; --k) {
                        r[k - 1] = r[k + 1] + 2.0 * k / kappa * r[k];
                    }
                    const double norm = r[0];
                    for (int k = 1; k <= kMaxTerms; ++k) {
                        r[k] /= norm;                 // store I_k(κ) / I_0(κ)
                    }
                    cached_kappa = kappa;
                }
                double series = 0.0;
                for (int k = 1; k <= kMaxTerms; ++k) {
                    series += r[k] * std::sin(k * dx) / static_cast<double>(k);
                }
                double F = base + series / M_PI;
                F = std::clamp(F, 0.0, 1.0);
                op[i] = static_cast<T>(F);
            }
        };

        if (out_dtype == DType::Float32) {
            compute(out.data<float>(), loc_cpu.data<float>(),
                    kappa_cpu.data<float>(), val_cpu.data<float>());
        } else if (out_dtype == DType::Float64) {
            compute(out.data<double>(), loc_cpu.data<double>(),
                    kappa_cpu.data<double>(), val_cpu.data<double>());
        } else {
            throw std::runtime_error("VonMises::cdf: only Float32/Float64 supported");
        }

        std::vector<int64_t> target_shape;
        if (vn >= ln && vn >= kn) {
            target_shape.assign(value.shape().begin(), value.shape().end());
        } else if (ln >= kn) {
            target_shape.assign(loc_.shape().begin(), loc_.shape().end());
        } else {
            target_shape.assign(concentration_.shape().begin(),
                                concentration_.shape().end());
        }
        auto reshaped = out.reshape(target_shape);
        if (widen_for_f16) {
            reshaped = reshaped.to(orig_dtype);
        }
        return reshaped.to(orig_device);
    }

    /**
     * @brief Inverse CDF of VonMises(μ, κ) via per-element bisection on the
     *        Marsaglia CDF (audit E.5).
     *
     * No closed form exists. Bisection over [μ - π, μ + π] converges in
     * ~52 iterations to float64 precision and is robust for all κ.
     */
    auto icdf(const Tensor& q) -> Tensor {
        auto orig_dtype = loc_.dtype();
        auto orig_device = loc_.device();
        const bool widen_for_f16 = (orig_dtype == DType::Float16 ||
                                    orig_dtype == DType::BFloat16);
        auto loc_cpu = loc_.to(Device::cpu()).contiguous();
        auto kappa_cpu = concentration_.to(Device::cpu()).contiguous();
        auto q_cpu = q.to(Device::cpu()).contiguous();
        if (widen_for_f16) {
            loc_cpu = loc_cpu.to(DType::Float32);
            kappa_cpu = kappa_cpu.to(DType::Float32);
            q_cpu = q_cpu.to(DType::Float32);
        }

        const int64_t ln = loc_cpu.numel();
        const int64_t kn = kappa_cpu.numel();
        const int64_t qn = q_cpu.numel();
        const int64_t n  = std::max({ln, kn, qn});
        auto out_dtype = widen_for_f16 ? DType::Float32 : orig_dtype;
        auto out = zeros({n}, out_dtype, Device::cpu());

        constexpr int kMaxTerms = 50;
        const int kMillerM = kMaxTerms + 20;

        // Scalar Marsaglia CDF used inside the bisection. The I_k/I_0 ratios
        // depend only on κ (not on x), so the caller precomputes them once per
        // element via fill_bessel_ratios() and passes the ratio buffer here.
        // This collapses the 60 bisection steps from O(M) recurrence work each
        // to a single O(kMaxTerms) sin-series, with zero per-call allocation.
        auto scalar_cdf = [&](double x, double mu, double kappa,
                              const std::vector<double>& ratios) -> double {
            double dx = std::remainder(x - mu, 2.0 * M_PI);
            double base = (dx + M_PI) / (2.0 * M_PI);
            if (kappa < 1e-6) return std::clamp(base, 0.0, 1.0);
            double series = 0.0;
            for (int k = 1; k <= kMaxTerms; ++k) {
                series += ratios[k] * std::sin(k * dx) / static_cast<double>(k);
            }
            double F = base + series / M_PI;
            return std::clamp(F, 0.0, 1.0);
        };

        // Fill `ratios[k] = I_k(κ)/I_0(κ)` for k = 1..kMaxTerms into the reused
        // buffer using the downward Miller recurrence.
        auto fill_bessel_ratios = [&](double kappa, std::vector<double>& r) {
            r[kMillerM] = 0.0;
            r[kMillerM - 1] = 1.0;
            for (int k = kMillerM - 1; k > 0; --k) {
                r[k - 1] = r[k + 1] + 2.0 * k / kappa * r[k];
            }
            const double norm = r[0];
            for (int k = 1; k <= kMaxTerms; ++k) {
                r[k] /= norm;
            }
        };

        auto compute = [&](auto* op, const auto* lp, const auto* kp,
                           const auto* qp) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(op)>>;
            // Reused buffers/cache across elements; the Bessel ratios are
            // recomputed only when κ changes (kn == 1 is the common case).
            std::vector<double> ratios(kMillerM + 1, 0.0);
            double cached_kappa = std::numeric_limits<double>::quiet_NaN();
            for (int64_t i = 0; i < n; ++i) {
                double mu    = static_cast<double>(lp[i % ln]);
                double kappa = static_cast<double>(kp[i % kn]);
                double qval  = static_cast<double>(qp[i % qn]);
                qval = std::clamp(qval, 0.0, 1.0);
                if (kappa >= 1e-6 && kappa != cached_kappa) {
                    fill_bessel_ratios(kappa, ratios);
                    cached_kappa = kappa;
                }
                double lo = mu - M_PI;
                double hi = mu + M_PI;
                for (int it = 0; it < 60; ++it) {
                    double mid = 0.5 * (lo + hi);
                    double F = scalar_cdf(mid, mu, kappa, ratios);
                    if (F < qval) lo = mid; else hi = mid;
                }
                double result = 0.5 * (lo + hi);
                // Wrap to (-π, π].
                result = std::remainder(result, 2.0 * M_PI);
                op[i] = static_cast<T>(result);
            }
        };

        if (out_dtype == DType::Float32) {
            compute(out.data<float>(), loc_cpu.data<float>(),
                    kappa_cpu.data<float>(), q_cpu.data<float>());
        } else if (out_dtype == DType::Float64) {
            compute(out.data<double>(), loc_cpu.data<double>(),
                    kappa_cpu.data<double>(), q_cpu.data<double>());
        } else {
            throw std::runtime_error("VonMises::icdf: only Float32/Float64 supported");
        }

        std::vector<int64_t> target_shape;
        if (qn >= ln && qn >= kn) {
            target_shape.assign(q.shape().begin(), q.shape().end());
        } else if (ln >= kn) {
            target_shape.assign(loc_.shape().begin(), loc_.shape().end());
        } else {
            target_shape.assign(concentration_.shape().begin(),
                                concentration_.shape().end());
        }
        auto reshaped = out.reshape(target_shape);
        if (widen_for_f16) {
            reshaped = reshaped.to(orig_dtype);
        }
        return reshaped.to(orig_device);
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


    /**
     * @brief Mean of the relaxed sample. No simple closed form (the sigmoid
     * of a scaled logistic is not analytically integrable); Monte-Carlo.
     */
    auto mean() -> Tensor override {
        return detail::mc_mean([this]() { return this->rsample({}); });
    }

    /** @brief Element-wise variance via Monte-Carlo. */
    auto variance() -> Tensor override {
        return detail::mc_variance([this]() { return this->rsample({}); });
    }

    /** @brief Differential entropy via Monte-Carlo: H = -E[log p(X)]. */
    auto entropy() -> Tensor override {
        return detail::mc_entropy(
            [this]() { return this->rsample({}); },
            [this](const Tensor& s) { return this->log_prob(s); });
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
        // Compute probs via softmax (inline, no dependency on nn::). Subtract the
        // PER-ROW max over the category axis, not a global scalar max: with a
        // global max a row whose logits are all far below it underflows
        // exp(.)→0, making sum_exp 0 and probs 0/0 = NaN.
        auto max_val = tenzor::max(logits, std::optional<int64_t>{-1}, /*keepdim=*/true);
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
        // Softmax (inline implementation over last dim). Per-row max over the
        // category axis (a global scalar max underflows rows that sit far below
        // it to all-zero, giving sum_exp 0 and a 0/0 = NaN sample).
        auto max_s = tenzor::max(scores, std::optional<int64_t>{-1}, /*keepdim=*/true);
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


    /**
     * @brief Mean of the relaxed one-hot sample. Concrete-distribution
     * moments have no closed form; Monte-Carlo.
     */
    auto mean() -> Tensor override {
        return detail::mc_mean([this]() { return this->rsample({}); });
    }

    /** @brief Element-wise variance via Monte-Carlo. */
    auto variance() -> Tensor override {
        return detail::mc_variance([this]() { return this->rsample({}); });
    }

    /** @brief Differential entropy via Monte-Carlo. */
    auto entropy() -> Tensor override {
        return detail::mc_entropy(
            [this]() { return this->rsample({}); },
            [this](const Tensor& s) { return this->log_prob(s); });
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

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        // Bartlett decomposition on CPU.
        // Widen Float16/BFloat16 to Float32 for the gamma + matmul chain;
        // narrow back at the end. Float16 lacks the dynamic range for
        // chi2 samples and propagates corruption through the trsm path.
        auto orig_dtype = scale_tril_.dtype();
        const bool widen_for_f16 = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);

        auto df_cpu = df_.to(Device::cpu()).contiguous();
        if (df_cpu.dtype() == DType::Float16 || df_cpu.dtype() == DType::BFloat16) {
            df_cpu = df_cpu.to(DType::Float32);
        }
        double df_val;
        if (df_cpu.dtype() == DType::Float32) {
            df_val = static_cast<double>(df_cpu.data<float>()[0]);
        } else {
            df_val = df_cpu.data<double>()[0];
        }

        auto orig_device = scale_tril_.device();
        auto dtype = widen_for_f16 ? DType::Float32 : orig_dtype;
        auto device = orig_device;
        Tensor scale_tril_widened = widen_for_f16
            ? scale_tril_.to(DType::Float32)
            : scale_tril_;

        // sample_shape gives the leading batch dimensions; the returned tensor
        // has shape sample_shape + [p_, p_]. An empty sample_shape yields a
        // single (p_, p_) matrix. Compute how many independent Bartlett
        // factors we need to draw.
        int64_t batch_count = 1;
        for (auto d : sample_shape) {
            if (d < 0) {
                throw std::runtime_error("Wishart: sample_shape dims must be non-negative");
            }
            batch_count *= d;
        }

        // Build a batch of lower-triangular Bartlett factors L on CPU with
        // shape [batch_count, p_, p_]. Each [p_, p_] slice is an independent
        // draw so that batched Wishart sampling produces distinct matrices.
        auto L = zeros({batch_count, p_, p_}, dtype, Device::cpu());

        auto& rng = detail::rng_for(generator_);
        std::normal_distribution<double> normal(0.0, 1.0);

        const int64_t mat_stride = p_ * p_;
        if (dtype == DType::Float32) {
            float* lp = L.data<float>();
            for (int64_t b = 0; b < batch_count; ++b) {
                float* mp = lp + b * mat_stride;
                for (int64_t i = 0; i < p_; ++i) {
                    // Diagonal: sqrt(Chi2(df - i)) = sqrt(Gamma((df-i)/2, 0.5) * 2)
                    // Equivalently, sample Chi2(df - i) via Gamma((df-i)/2, 0.5)
                    double chi2_val = detail::sample_gamma_scalar<double>(
                        (df_val - static_cast<double>(i)) / 2.0, 0.5, rng);
                    mp[i * p_ + i] = static_cast<float>(std::sqrt(chi2_val));
                    // Below diagonal: standard normal
                    for (int64_t j = 0; j < i; ++j) {
                        mp[i * p_ + j] = static_cast<float>(normal(rng));
                    }
                }
            }
        } else if (dtype == DType::Float64) {
            double* lp = L.data<double>();
            for (int64_t b = 0; b < batch_count; ++b) {
                double* mp = lp + b * mat_stride;
                for (int64_t i = 0; i < p_; ++i) {
                    double chi2_val = detail::sample_gamma_scalar<double>(
                        (df_val - static_cast<double>(i)) / 2.0, 0.5, rng);
                    mp[i * p_ + i] = std::sqrt(chi2_val);
                    for (int64_t j = 0; j < i; ++j) {
                        mp[i * p_ + j] = normal(rng);
                    }
                }
            }
        } else {
            throw std::runtime_error("Wishart: only Float32 and Float64 supported");
        }

        L = L.to(device);

        // A = scale_tril @ L (use the widened scale_tril if we upcast).
        // Broadcast scale_tril [p_, p_] across the batch so each Bartlett
        // factor is transformed by the same Cholesky factor.
        auto scale_b = ::tenzor::broadcast_to(
            scale_tril_widened.reshape({1, p_, p_}), {batch_count, p_, p_});
        auto A = matmul(scale_b, L);
        // W = A @ A^T  (batched over the leading dimension)
        auto W = matmul(A, transpose(A, -2, -1));
        if (widen_for_f16) {
            W = W.to(orig_dtype);
        }

        // Reshape from [batch_count, p_, p_] to sample_shape + [p_, p_]. With
        // an empty sample_shape this collapses to the single (p_, p_) matrix.
        std::vector<int64_t> out_shape(sample_shape.begin(), sample_shape.end());
        out_shape.push_back(p_);
        out_shape.push_back(p_);
        return W.reshape(out_shape);
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

    // Inf-B14 (deferred → landed): element-wise variance closed form.
    // For W ~ Wishart(V, df), Eaton (1983) Theorem 8.2 gives
    //   Var(W_ij) = df · (V_ii · V_jj + V_ij²)
    // where V = scale_tril @ scale_tril^T is the p×p scale matrix.
    // Reference: torch.distributions.Wishart.variance().
    auto variance() -> Tensor override {
        auto V = matmul(scale_tril_, transpose(scale_tril_, -2, -1));
        // diag(V) is the p-vector V_ii. Build the outer product
        // diag(V)[i] · diag(V)[j] = V_ii · V_jj via unsqueeze+broadcast.
        auto diag_V = tenzor::diag(V);                            // (..., p)
        auto vii = diag_V.unsqueeze(-1);                          // (..., p, 1)
        auto vjj = diag_V.unsqueeze(-2);                          // (..., 1, p)
        auto outer = vii * vjj;                                   // (..., p, p)
        auto V_sq = V * V;
        return df_ * (outer + V_sq);
    }


    /**
     * @brief Differential entropy of Wishart(ν, V).
     *
     * Closed form (Wikipedia / Eaton 1983):
     *   H = log Γ_p(ν/2) + p(p+1)/2 · log(2) + (p+1)/2 · log|V| + νp/2
     *       - (ν - p - 1)/2 · Σ_{j=1..p} ψ((ν + 1 - j)/2)
     *
     * where Γ_p is the multivariate gamma function and log|V| is obtained
     * from the Cholesky factor as 2·Σ log(diag(scale_tril)).
     *
     * Computed scalar-wise on CPU for the unbatched case (matches the
     * sample/log_prob restrictions above).
     */
    auto entropy() -> Tensor override {
        // Extract scalar df.
        auto df_cpu = df_.to(Device::cpu()).contiguous();
        double df_val = (df_cpu.dtype() == DType::Float32)
            ? static_cast<double>(df_cpu.data<float>()[0])
            : df_cpu.data<double>()[0];
        const double half_df = df_val / 2.0;
        const double p_d = static_cast<double>(p_);

        // log|V| from scale_tril: V = L Lᵀ, log|V| = 2·Σ log(diag(L)).
        auto L_cpu = scale_tril_.to(Device::cpu()).contiguous();
        double log_det_V = 0.0;
        if (L_cpu.dtype() == DType::Float32) {
            const float* lp = L_cpu.data<float>();
            for (int64_t i = 0; i < p_; ++i) {
                log_det_V += std::log(static_cast<double>(lp[i * p_ + i]));
            }
        } else {
            const double* lp = L_cpu.data<double>();
            for (int64_t i = 0; i < p_; ++i) {
                log_det_V += std::log(lp[i * p_ + i]);
            }
        }
        log_det_V *= 2.0;

        // log Γ_p(ν/2) = p(p-1)/4 · log(π) + Σ_{j=1..p} lgamma(ν/2 + (1-j)/2)
        double mv_lgamma = p_d * (p_d - 1.0) / 4.0 * std::log(M_PI);
        for (int64_t j = 1; j <= p_; ++j) {
            mv_lgamma += std::lgamma(half_df + static_cast<double>(1 - j) / 2.0);
        }

        // Σ_{j=1..p} ψ((ν + 1 - j)/2)
        // Use std::digamma if available; otherwise compute via series.
        // Tenzor's ops::digamma works on tensors; we just need scalars here.
        auto digamma_scalar = [](double x) -> double {
            // Boost-style series; accurate to 1e-12 for x > 6, recurse for smaller.
            double result = 0.0;
            while (x < 6.0) { result -= 1.0 / x; x += 1.0; }
            double inv = 1.0 / x;
            double inv2 = inv * inv;
            result += std::log(x) - 0.5 * inv
                    - inv2 * ( 1.0 / 12.0
                              - inv2 * ( 1.0 / 120.0
                                        - inv2 * ( 1.0 / 252.0 )));
            return result;
        };
        double digamma_sum = 0.0;
        for (int64_t j = 1; j <= p_; ++j) {
            digamma_sum += digamma_scalar(half_df + static_cast<double>(1 - j) / 2.0);
        }

        double H = mv_lgamma
                 + p_d * (p_d + 1.0) / 2.0 * std::log(2.0)
                 + (p_d + 1.0) / 2.0 * log_det_V
                 + df_val * p_d / 2.0
                 - (df_val - p_d - 1.0) / 2.0 * digamma_sum;

        return full({1}, H, scale_tril_.dtype(), scale_tril_.device());
    }

private:
    Tensor df_;
    Tensor scale_tril_;
    int64_t p_;
};


// ============================================================================
// Pareto Distribution
// ============================================================================

/**
 * @brief Pareto (Type I) distribution parameterized by scale and alpha.
 *
 * f(x) = alpha * scale^alpha / x^(alpha+1),  x >= scale
 */
class Pareto : public Distribution {
public:
    Pareto(Tensor scale, Tensor alpha)
        : scale_(std::move(scale)), alpha_(std::move(alpha)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(scale_.shape().begin(), scale_.shape().end())
            : sample_shape;
        // Inverse CDF: scale / U^(1/alpha) = scale * exp(-log(U)/alpha)
        auto u = rand(shape, scale_.dtype(), scale_.device());
        u = tenzor::clamp(u, 1e-7f, 1.0f);
        return scale_ * tenzor::exp(tenzor::neg(tenzor::log(u)) / alpha_);
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return sample(std::move(sample_shape));  // Inverse CDF is differentiable
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        return tenzor::log(alpha_) + alpha_ * tenzor::log(scale_)
             - (alpha_ + 1.0f) * tenzor::log(value);
    }

    auto cdf(const Tensor& value) -> Tensor {
        // CDF = 1 - (scale/x)^alpha = 1 - exp(alpha * log(scale/x))
        return 1.0f - tenzor::exp(alpha_ * tenzor::log(scale_ / value));
    }

    auto entropy() -> Tensor override {
        return tenzor::log(scale_ / alpha_) + tenzor::reciprocal(alpha_) + 1.0f;
    }

    auto mean() -> Tensor override {
        return alpha_ * scale_ / (alpha_ - 1.0f);
    }

    auto variance() -> Tensor override {
        auto am1 = alpha_ - 1.0f;
        return scale_ * scale_ * alpha_ / (am1 * am1 * (alpha_ - 2.0f));
    }

private:
    Tensor scale_, alpha_;
};

// ============================================================================
// Weibull Distribution
// ============================================================================

/**
 * @brief Weibull distribution parameterized by scale (lambda) and concentration (k).
 *
 * f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)
 */
class Weibull : public Distribution {
public:
    Weibull(Tensor scale, Tensor concentration)
        : scale_(std::move(scale)), concentration_(std::move(concentration)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(scale_.shape().begin(), scale_.shape().end())
            : sample_shape;
        auto u = rand(shape, scale_.dtype(), scale_.device());
        u = tenzor::clamp(u, 1e-7f, 1.0f - 1e-7f);
        auto neg_log = tenzor::neg(tenzor::log(1.0f - u));
        return scale_ * tenzor::exp(tenzor::log(neg_log) / concentration_);
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return sample(std::move(sample_shape));  // Inverse CDF is differentiable
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto x_over_l = value / scale_;
        auto log_x_over_l = tenzor::log(x_over_l);
        return tenzor::log(concentration_ / scale_)
             + (concentration_ - 1.0f) * log_x_over_l
             - tenzor::exp(concentration_ * log_x_over_l);
    }

    auto cdf(const Tensor& value) -> Tensor {
        // CDF = 1 - exp(-(x/lambda)^k)
        auto log_x_over_l = tenzor::log(value / scale_);
        return 1.0f - tenzor::exp(tenzor::neg(tenzor::exp(concentration_ * log_x_over_l)));
    }

    auto entropy() -> Tensor override {
        // H = gamma_const * (1 - 1/k) + log(lambda/k) + 1
        // where gamma_const = Euler-Mascheroni constant ~ 0.5772
        constexpr float euler_mascheroni = 0.5772156649f;
        return euler_mascheroni * (1.0f - tenzor::reciprocal(concentration_))
             + tenzor::log(scale_ / concentration_) + 1.0f;
    }

    auto mean() -> Tensor override {
        return scale_ * tenzor::gamma(1.0f + tenzor::reciprocal(concentration_));
    }

    auto variance() -> Tensor override {
        auto inv_k = tenzor::reciprocal(concentration_);
        auto g1 = tenzor::gamma(1.0f + inv_k);
        auto g2 = tenzor::gamma(1.0f + 2.0f * inv_k);
        return scale_ * scale_ * (g2 - g1 * g1);
    }

private:
    Tensor scale_, concentration_;
};

// ============================================================================
// Kumaraswamy Distribution
// ============================================================================

/**
 * @brief Kumaraswamy distribution on (0, 1).
 *
 * f(x) = a * b * x^(a-1) * (1 - x^a)^(b-1)
 */
class Kumaraswamy : public Distribution {
public:
    Kumaraswamy(Tensor concentration1, Tensor concentration0)
        : a_(std::move(concentration1)), b_(std::move(concentration0)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(a_.shape().begin(), a_.shape().end())
            : sample_shape;
        // Inverse CDF: (1 - (1 - U)^(1/b))^(1/a)
        // = exp(log(1 - exp(log(1-U)/b)) / a)
        auto u = rand(shape, a_.dtype(), a_.device());
        u = tenzor::clamp(u, 1e-7f, 1.0f - 1e-7f);
        auto inner = 1.0f - tenzor::exp(tenzor::log(1.0f - u) / b_);
        inner = tenzor::clamp(inner, 1e-7f, 1.0f);
        return tenzor::exp(tenzor::log(inner) / a_);
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return sample(std::move(sample_shape));  // Inverse CDF is differentiable
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto log_x = tenzor::log(tenzor::clamp(value, 1e-7f, 1.0f));
        auto x_a = tenzor::exp(a_ * log_x);
        return tenzor::log(a_) + tenzor::log(b_)
             + (a_ - 1.0f) * log_x
             + (b_ - 1.0f) * tenzor::log(tenzor::clamp(1.0f - x_a, 1e-7f, 1.0f));
    }

    auto cdf(const Tensor& value) -> Tensor {
        // CDF = 1 - (1 - x^a)^b
        auto log_x = tenzor::log(tenzor::clamp(value, 1e-7f, 1.0f));
        auto x_a = tenzor::exp(a_ * log_x);
        return 1.0f - tenzor::exp(b_ * tenzor::log(tenzor::clamp(1.0f - x_a, 1e-7f, 1.0f)));
    }

    // Wave Inf-B4: Kumaraswamy closed forms.
    // mean = b · B(1 + 1/a, b) = b · exp(lgamma(1+1/a) + lgamma(b) - lgamma(1+1/a+b))
    auto mean() -> Tensor override {
        auto one = tenzor::full(std::vector<int64_t>(a_.shape().begin(), a_.shape().end()),
                                  1.0, a_.dtype(), a_.device());
        auto one_plus_inv_a = one + tenzor::reciprocal(a_);
        auto log_beta = tenzor::lgamma(one_plus_inv_a) + tenzor::lgamma(b_)
                      - tenzor::lgamma(one_plus_inv_a + b_);
        return b_ * tenzor::exp(log_beta);
    }

    // variance = m₂ - mean²; m₂ = b · B(1 + 2/a, b).
    auto variance() -> Tensor override {
        auto one = tenzor::full(std::vector<int64_t>(a_.shape().begin(), a_.shape().end()),
                                  1.0, a_.dtype(), a_.device());
        auto inv_a = tenzor::reciprocal(a_);
        auto one_plus_2inv_a = one + 2.0f * inv_a;
        auto log_beta2 = tenzor::lgamma(one_plus_2inv_a) + tenzor::lgamma(b_)
                       - tenzor::lgamma(one_plus_2inv_a + b_);
        auto m2 = b_ * tenzor::exp(log_beta2);
        auto m1 = mean();
        return m2 - m1 * m1;
    }

    // entropy = (1 - 1/b) + (1 - 1/a)·(ψ(b+1) + γ_EM) + log(b/a)
    // where γ_EM = Euler-Mascheroni ≈ 0.5772156649.
    auto entropy() -> Tensor override {
        constexpr float euler_mascheroni = 0.5772156649f;
        auto one = tenzor::full(std::vector<int64_t>(a_.shape().begin(), a_.shape().end()),
                                  1.0, a_.dtype(), a_.device());
        auto inv_a = tenzor::reciprocal(a_);
        auto inv_b = tenzor::reciprocal(b_);
        auto term1 = one - inv_b;
        auto term2 = (one - inv_a) * (tenzor::digamma(b_ + one) + euler_mascheroni);
        auto term3 = tenzor::log(b_ / a_);
        return term1 + term2 + term3;
    }

private:
    Tensor a_, b_;
};

// ============================================================================
// ContinuousBernoulli Distribution
// ============================================================================

/**
 * @brief Continuous Bernoulli distribution on (0, 1).
 *
 * f(x) = C(lambda) * lambda^x * (1 - lambda)^(1-x)
 * where C(lambda) is the normalizing constant.
 */
class ContinuousBernoulli : public Distribution {
public:
    explicit ContinuousBernoulli(Tensor probs)
        : probs_(std::move(probs)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(probs_.shape().begin(), probs_.shape().end())
            : sample_shape;
        // Sample via rejection or direct inverse CDF.
        // For lambda != 0.5: icdf(u) = log1p((2*lambda - 1) * u / (1 - lambda))
        //                              / log(lambda / (1 - lambda))
        auto u = rand(shape, probs_.dtype(), probs_.device());
        u = tenzor::clamp(u, 1e-6f, 1.0f - 1e-6f);
        auto p = tenzor::clamp(probs_, 1e-6f, 1.0f - 1e-6f);
        auto logits = tenzor::log(p) - tenzor::log(1.0f - p);
        // Stable inverse CDF: x = log1p(u * (exp(logits) - 1)) / logits
        auto exp_logits = tenzor::exp(logits);
        auto x = tenzor::log1p(u * (exp_logits - 1.0f)) / logits;
        return tenzor::clamp(x, 0.0f, 1.0f);
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return sample(std::move(sample_shape));  // Inverse CDF is differentiable
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto p = tenzor::clamp(probs_, 1e-6f, 1.0f - 1e-6f);
        auto log_c = log_normalizing_constant(p);
        return log_c + value * tenzor::log(p) + (1.0f - value) * tenzor::log(1.0f - p);
    }

    auto entropy() -> Tensor override {
        // H = -log(C) + (2*lambda - 1) / (2*lambda*(1 - lambda)) * mean
        //   - where mean is the distribution mean
        auto p = tenzor::clamp(probs_, 1e-6f, 1.0f - 1e-6f);
        auto log_c = log_normalizing_constant(p);
        auto m = this->mean();
        auto logits = tenzor::log(p) - tenzor::log(1.0f - p);
        return tenzor::neg(log_c) - logits * m;
    }

    auto mean() -> Tensor override {
        auto p = tenzor::clamp(probs_, 1e-6f, 1.0f - 1e-6f);
        auto two_p_minus_1 = 2.0f * p - 1.0f;
        return p / two_p_minus_1 + 1.0f / (2.0f * tenzor::atanh(1.0f - 2.0f * p));
    }

    auto variance() -> Tensor override {
        auto p = tenzor::clamp(probs_, 1e-6f, 1.0f - 1e-6f);
        auto m = this->mean();
        auto logits = tenzor::log(p) - tenzor::log(1.0f - p);
        // var = mean * (1 - mean) / (1 + logits * (1 - 2*mean))
        // Simplified: use second moment
        return (m - m * m) / (1.0f + logits * (1.0f - 2.0f * m));
    }

private:
    static auto log_normalizing_constant(const Tensor& p) -> Tensor {
        // C(lambda) = 2 * atanh(1 - 2*lambda) / (1 - 2*lambda)
        auto two_p_minus_1 = 2.0f * p - 1.0f;
        auto abs_diff = tenzor::abs(two_p_minus_1);
        return tenzor::log(2.0f * tenzor::atanh(abs_diff) / abs_diff);
    }

    Tensor probs_;
};

// ============================================================================
// OneHotCategorical Distribution
// ============================================================================

/**
 * @brief One-hot categorical distribution.
 *
 * Samples are one-hot vectors drawn from a categorical distribution.
 */
class OneHotCategorical : public Distribution {
public:
    explicit OneHotCategorical(Tensor probs)
        : probs_(std::move(probs)),
          categorical_(probs_) {}

    auto probs() const -> const Tensor& { return probs_; }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto indices = categorical_.sample(sample_shape);
        auto num_classes = probs_.shape().back();
        return one_hot(indices, num_classes).to(probs_.dtype());
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log_prob = sum(value * log(probs), dim=-1)
        auto log_probs = tenzor::log(tenzor::clamp(probs_, 1e-7f, 1.0f));
        return tenzor::sum(value * log_probs, -1);
    }

    auto entropy() -> Tensor override {
        return categorical_.entropy();
    }

    auto mean() -> Tensor override {
        return probs_;
    }

    auto variance() -> Tensor override {
        return probs_ * (1.0f - probs_);
    }

private:
    Tensor probs_;
    Categorical categorical_;
};

// ============================================================================
// LogisticNormal Distribution
// ============================================================================

/**
 * @brief Logistic-normal distribution (softmax of a multivariate normal).
 *
 * Samples lie on the simplex: X = softmax(Y), Y ~ Normal(loc, scale).
 */
class LogisticNormal : public Distribution {
public:
    LogisticNormal(Tensor loc, Tensor scale)
        : loc_(std::move(loc)), scale_(std::move(scale)),
          normal_(loc_, scale_) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        // Combine sample_shape with event shape
        auto event_shape = std::vector<int64_t>(loc_.shape().begin(), loc_.shape().end());
        auto full_shape = sample_shape;
        full_shape.insert(full_shape.end(), event_shape.begin(), event_shape.end());

        auto y = normal_.sample(full_shape);
        // softmax: exp(y_i) / sum(exp(y_j))
        auto exp_y = tenzor::exp(y);
        auto sum_exp = tenzor::sum(exp_y, -1, /*keepdim=*/true);
        return exp_y / sum_exp;
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return sample(std::move(sample_shape));  // Normal rsample + softmax is differentiable
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto log_x = tenzor::log(tenzor::clamp(value, 1e-7f, 1.0f));
        auto normal_lp = normal_.log_prob(log_x);
        return tenzor::sum(normal_lp, -1) - tenzor::sum(log_x, -1);
    }


    /**
     * @brief Mean of the logistic-normal sample. Softmax(Normal) has no
     * closed-form moments; Monte-Carlo estimator.
     */
    auto mean() -> Tensor override {
        return detail::mc_mean([this]() { return this->rsample({}); });
    }

    /** @brief Element-wise variance via Monte-Carlo. */
    auto variance() -> Tensor override {
        return detail::mc_variance([this]() { return this->rsample({}); });
    }

    /** @brief Differential entropy via Monte-Carlo: H = -E[log p(X)]. */
    auto entropy() -> Tensor override {
        return detail::mc_entropy(
            [this]() { return this->rsample({}); },
            [this](const Tensor& s) { return this->log_prob(s); });
    }

private:
    Tensor loc_, scale_;
    Normal normal_;
};

// ============================================================================
// LowRankMultivariateNormal Distribution
// ============================================================================

/**
 * @brief Multivariate normal with low-rank plus diagonal covariance.
 *
 * Covariance = W @ W^T + diag(D), where W is (p, rank) and D is (p,).
 * Efficient for high-dimensional distributions with low-rank structure.
 */
class LowRankMultivariateNormal : public Distribution {
public:
    LowRankMultivariateNormal(Tensor loc, Tensor cov_factor, Tensor cov_diag)
        : loc_(std::move(loc)), cov_factor_(std::move(cov_factor)),
          cov_diag_(std::move(cov_diag)) {}

    auto loc() const -> const Tensor& { return loc_; }
    auto cov_factor() const -> const Tensor& { return cov_factor_; }
    auto cov_diag() const -> const Tensor& { return cov_diag_; }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        auto p = loc_.shape().back();
        auto rank = cov_factor_.shape().back();
        auto shape = sample_shape.empty()
            ? std::vector<int64_t>(loc_.shape().begin(), loc_.shape().end())
            : sample_shape;

        // Sample z ~ N(0, I_rank) and eps ~ N(0, I_p)
        auto z_shape = shape;
        z_shape.back() = rank;
        auto z = randn(z_shape, loc_.dtype(), loc_.device());
        auto eps = randn(shape, loc_.dtype(), loc_.device());

        // X = loc + W @ z + sqrt(D) * eps
        auto wz = matmul(z, transpose(cov_factor_, -2, -1));
        // W is (p, rank), z is (..., rank), wz needs to be (..., p)
        // Actually z is (..., rank) and W^T is (rank, p), so matmul(z, W^T) = (..., p)
        return loc_ + wz + tenzor::sqrt(cov_diag_) * eps;
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto diff = value - loc_;
        auto p = static_cast<double>(loc_.shape().back());

        // Use Woodbury identity for efficient log-det and solve
        // Sigma = D + W W^T
        // Sigma^-1 = D^-1 - D^-1 W (I + W^T D^-1 W)^-1 W^T D^-1
        auto d_inv = tenzor::reciprocal(cov_diag_);
        auto d_inv_w = d_inv.unsqueeze(-1) * cov_factor_;
        auto capacitance = tenzor::eye(cov_factor_.shape().back(), std::nullopt,
                                      loc_.dtype(), loc_.device())
                         + matmul(transpose(cov_factor_, -2, -1), d_inv_w);
        auto cap_chol = tenzor::linalg::cholesky(capacitance);

        // log|Sigma| = log|D| + log|capacitance|
        auto log_det_d = tenzor::sum(tenzor::log(cov_diag_), -1);
        auto log_det_cap = 2.0f * tenzor::sum(
            tenzor::log(tenzor::abs(tenzor::diag(cap_chol))), -1);
        auto log_det = log_det_d + log_det_cap;

        // Mahalanobis: diff^T Sigma^-1 diff
        auto d_inv_diff = d_inv * diff;
        auto w_d_inv_diff = matmul(transpose(cov_factor_, -2, -1),
                                    d_inv_diff.unsqueeze(-1));
        auto solved = tenzor::linalg::solve(capacitance, w_d_inv_diff);
        auto quad_term = tenzor::sum(diff * d_inv_diff, -1)
                       - tenzor::sum(w_d_inv_diff.squeeze(-1)
                                   * solved.squeeze(-1), -1);

        return -0.5 * (p * std::log(2.0 * M_PI) + log_det + quad_term);
    }

    auto mean() -> Tensor override { return loc_; }

    // Inf-B15 (deferred → landed): variance = diagonal of Σ = D + WW^T.
    // The diagonal of WW^T at row i is Σ_j W[i,j]² — i.e. row-wise L2².
    auto variance() -> Tensor override {
        auto ww_diag = tenzor::sum(cov_factor_ * cov_factor_, -1);
        return cov_diag_ + ww_diag;
    }

    // Inf-B15 (deferred → landed): closed-form entropy via Woodbury.
    // For X ~ N(μ, Σ) with Σ = D + WW^T (D diagonal, W of shape (..., p, r)):
    //   H[X] = 0.5 · (p · log(2πe) + log|Σ|)
    // log|Σ| via the matrix-determinant lemma:
    //   log|D + WW^T| = log|D| + log|I_r + W^T D^-1 W|
    // The capacitance matrix I_r + W^T D^-1 W is r×r — cheap to Cholesky.
    auto entropy() -> Tensor override {
        const double p = static_cast<double>(loc_.shape().back());
        auto d_inv = tenzor::reciprocal(cov_diag_);
        auto d_inv_w = d_inv.unsqueeze(-1) * cov_factor_;
        auto capacitance = tenzor::eye(cov_factor_.shape().back(), std::nullopt,
                                      loc_.dtype(), loc_.device())
                         + matmul(transpose(cov_factor_, -2, -1), d_inv_w);
        auto cap_chol = tenzor::linalg::cholesky(capacitance);
        auto log_det_d = tenzor::sum(tenzor::log(cov_diag_), -1);
        auto log_det_cap = 2.0f * tenzor::sum(
            tenzor::log(tenzor::abs(tenzor::diag(cap_chol))), -1);
        auto log_det_sigma = log_det_d + log_det_cap;
        // 0.5 · (p · log(2πe) + log|Σ|).
        constexpr double kLog2PiE = 2.8378770664093454835606594728112352798;  // log(2πe)
        return 0.5f * (static_cast<float>(p * kLog2PiE) + log_det_sigma);
    }

private:
    Tensor loc_, cov_factor_, cov_diag_;
};

// ============================================================================
// LKJCholesky Distribution
// ============================================================================

/**
 * @brief LKJ distribution over Cholesky factors of correlation matrices.
 *
 * Parameterized by dimension d and concentration eta.
 * eta > 0: higher values concentrate around the identity.
 */
class LKJCholesky : public Distribution {
public:
    LKJCholesky(int64_t dim, Tensor concentration)
        : dim_(dim), concentration_(std::move(concentration)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        // Generate via Bartlett decomposition of Wishart(df, I) then normalize
        // to produce Cholesky factor of a correlation matrix.
        // df = 2*eta + d - 1 maps LKJ concentration to Wishart degrees of freedom.
        auto dtype = concentration_.dtype();
        auto device = concentration_.device();

        auto batch_shape = sample_shape.empty()
            ? std::vector<int64_t>(concentration_.shape().begin(),
                                    concentration_.shape().end())
            : sample_shape;

        // audit-2026-05-03 bug #7: item<double>() on a Float32 concentration
        // tensor throws a dtype-mismatch error. Dispatch on the actual dtype
        // before reading the scalar.
        auto df = 2.0 * concentration_as_double() + dim_ - 1.0;
        auto eye_mat = tenzor::eye(dim_, std::nullopt, dtype, device);
        auto wishart = Wishart(tenzor::full({1}, df, dtype, device), eye_mat);
        auto W = wishart.sample(batch_shape);

        // Cholesky of the Wishart sample
        auto chol_W = tenzor::linalg::cholesky(W);
        // Normalize each row by its diagonal to get Cholesky of correlation matrix
        auto diag_chol = tenzor::diag(chol_W);
        auto inv_diag = tenzor::reciprocal(diag_chol);
        return chol_W * inv_diag.unsqueeze(-1);
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(L) = sum_{k=1}^{d-1} (d - k - 1 + 2*eta - 2) * log(L_{k+1,k+1})
        // The first diagonal element L[0,0] = 1 always, so it's excluded.
        auto diag_vals = tenzor::diag(value);
        auto log_diag = tenzor::log(tenzor::abs(diag_vals));

        auto eta = concentration_as_double();
        // Build weight tensor: weight[k] = (d - k - 1 + 2*eta - 2) for k = 1..d-1
        // The weight buffer is built on CPU Float32 (the math fits comfortably
        // in Float32) and cast/transferred to the diag_vals dtype/device — same
        // pattern used elsewhere in the file for backend-agnostic mask buffers.
        auto cpu_w = tenzor::Tensor({dim_}, DType::Float32, Device::cpu());
        auto* w_ptr = cpu_w.data<float>();
        w_ptr[0] = 0.0f;  // First diagonal excluded
        for (int64_t k = 1; k < dim_; ++k) {
            w_ptr[k] = static_cast<float>(dim_ - k - 1.0 + 2.0 * eta - 2.0);
        }
        auto weights = cpu_w.to(diag_vals.dtype()).to(diag_vals.device());
        return tenzor::sum(weights * log_diag, -1);
    }

    // Inf-B6 (deferred → landed): LKJCholesky mean.
    // For LKJ(eta) on the Cholesky factor of a correlation matrix, the
    // expected value is the identity matrix's Cholesky factor: the
    // p×p identity I_p. (Each off-diagonal entry has E = 0 by the
    // distribution's symmetry around uncorrelated factors; each diagonal
    // entry has E = 1 because the factor is normalized.) Broadcast to
    // any batch dimensions implied by concentration_.
    auto mean() -> Tensor override {
        auto eye_mat = tenzor::eye(dim_, std::nullopt,
                                   concentration_.dtype(),
                                   concentration_.device());
        // If concentration has a batch shape, broadcast eye_mat by an
        // unsqueeze + expand. Without explicit broadcasting, just return
        // the single eye matrix — callers needing a batch broadcast can
        // expand themselves (mirrors torch's behavior for Distribution
        // shape-handling).
        return eye_mat;
    }


    /**
     * @brief Variance of the LKJ-Cholesky factor entries (Monte-Carlo).
     *
     * No simple closed-form exists for the variance of entries of the
     * Cholesky factor under LKJ(η). Monte-Carlo estimation is the standard
     * definition; result shape matches a single sample (dim_, dim_).
     */
    auto variance() -> Tensor override {
        return detail::mc_variance([this]() { return this->sample({}); });
    }

    /**
     * @brief Differential entropy of LKJCholesky(η) — Monte-Carlo:
     *   H = -E[log p(L)] ≈ -mean_i log_prob(L_i)
     */
    auto entropy() -> Tensor override {
        return detail::mc_entropy(
            [this]() { return this->sample({}); },
            [this](const Tensor& s) { return this->log_prob(s); });
    }

private:
    // Read concentration as a double, regardless of its tensor dtype.
    // item<T>() requires the requested type match the tensor dtype exactly,
    // so we must dispatch.
    auto concentration_as_double() const -> double {
        switch (concentration_.dtype()) {
            case DType::Float32:
                return static_cast<double>(concentration_.item<float>());
            case DType::Float64:
                return concentration_.item<double>();
            case DType::Float16:
            case DType::BFloat16:
                // Half-precision: convert to a Float32 scalar tensor first,
                // then read via item<float>(). cpu() ensures item<> works.
                return static_cast<double>(
                    concentration_.cpu().to(DType::Float32).item<float>());
            default:
                throw std::runtime_error(
                    "LKJCholesky: unsupported concentration dtype");
        }
    }

private:
    int64_t dim_;
    Tensor concentration_;
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
 *   - Bernoulli || Bernoulli
 *   - HalfNormal || HalfNormal
 *   - Uniform || Uniform
 *   - Exponential || Exponential
 *   - Laplace || Laplace
 *   - Gamma || Gamma
 *   - Beta || Beta
 *   - Categorical || Categorical
 *   - Dirichlet || Dirichlet
 *   - Poisson || Poisson
 *   - LogNormal || LogNormal
 *
 * For anything else, throws std::runtime_error. Use Monte-Carlo estimation
 * via `(p.sample() * (p.log_prob(x) - q.log_prob(x))).mean()` instead.
 */
inline auto kl_divergence(Distribution& p, Distribution& q) -> Tensor {
    // --- Normal || Normal ---
    if (auto* pn = dynamic_cast<Normal*>(&p)) {
        if (auto* qn = dynamic_cast<Normal*>(&q)) {
            auto mu1 = pn->mean(); auto v1 = pn->variance();
            auto mu2 = qn->mean(); auto v2 = qn->variance();
            auto s1 = tenzor::sqrt(v1);
            auto s2 = tenzor::sqrt(v2);
            auto diff = mu1 - mu2;
            return tenzor::log(s2 / s1) + (v1 + diff * diff) / (v2 * 2.0f) - 0.5f;
        }
    }
    // --- Bernoulli || Bernoulli ---
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
    // --- Bernoulli || Categorical (audit E.6) ---
    // A Bernoulli(p) is equivalent to a Categorical with probs [1-p, p]
    // over two classes. Closed form when q has exactly 2 classes.
    if (auto* pb = dynamic_cast<BernoulliDist*>(&p)) {
        if (auto* qc = dynamic_cast<Categorical*>(&q)) {
            const auto& q_probs_full = qc->probs();
            if (q_probs_full.shape().back() != 2) {
                throw std::runtime_error(
                    "kl_divergence(Bernoulli || Categorical): Categorical must "
                    "have exactly 2 classes to be equivalent to Bernoulli");
            }
            auto eps = 1e-7f;
            auto p_mean = pb->mean();
            auto p_clamped = tenzor::clamp(p_mean, eps, 1.0f - eps);
            auto q_clamped = tenzor::clamp(q_probs_full, eps, 1.0f);
            // KL = (1-p)*(log(1-p) - log q0) + p*(log p - log q1)
            // q0 = q_probs[..., 0], q1 = q_probs[..., 1]; use gather/slicing-free
            // approach: split along last dim via narrow.
            auto q0 = tenzor::narrow(q_clamped, /*dim=*/-1, /*start=*/0, /*length=*/1)
                          .reshape(std::vector<int64_t>(p_clamped.shape().begin(),
                                                        p_clamped.shape().end()));
            auto q1 = tenzor::narrow(q_clamped, /*dim=*/-1, /*start=*/1, /*length=*/1)
                          .reshape(std::vector<int64_t>(p_clamped.shape().begin(),
                                                        p_clamped.shape().end()));
            return (1.0f - p_clamped) * (tenzor::log(1.0f - p_clamped) - tenzor::log(q0))
                 + p_clamped * (tenzor::log(p_clamped) - tenzor::log(q1));
        }
    }
    // --- HalfNormal || HalfNormal ---
    if (auto* ph = dynamic_cast<HalfNormal*>(&p)) {
        if (auto* qh = dynamic_cast<HalfNormal*>(&q)) {
            auto s1 = ph->scale();
            auto s2 = qh->scale();
            auto v1 = s1 * s1;
            auto v2 = s2 * s2;
            return tenzor::log(s2 / s1) + v1 / (v2 * 2.0f) - 0.5f;
        }
    }
    // --- Uniform || Uniform ---
    if (auto* pu = dynamic_cast<Uniform*>(&p)) {
        if (auto* qu = dynamic_cast<Uniform*>(&q)) {
            // KL = log((b2 - a2) / (b1 - a1))
            // Only valid when support(p) ⊆ support(q)
            auto range_p = pu->high() - pu->low();
            auto range_q = qu->high() - qu->low();
            return tenzor::log(range_q / range_p);
        }
    }
    // --- Exponential || Exponential ---
    if (auto* pe = dynamic_cast<Exponential*>(&p)) {
        if (auto* qe = dynamic_cast<Exponential*>(&q)) {
            // KL = log(λ2/λ1) + λ1/λ2 - 1
            auto r1 = pe->rate();
            auto r2 = qe->rate();
            return tenzor::log(r2 / r1) + r1 / r2 - 1.0f;
        }
    }
    // --- Laplace || Laplace ---
    if (auto* pl = dynamic_cast<Laplace*>(&p)) {
        if (auto* ql = dynamic_cast<Laplace*>(&q)) {
            // KL = |μ1 - μ2|/b2 + b1/b2 * exp(-|μ1 - μ2|/b1) + log(b2/b1) - 1
            auto b1 = pl->scale();
            auto b2 = ql->scale();
            auto abs_diff = tenzor::abs(pl->loc() - ql->loc());
            return abs_diff / b2
                 + (b1 / b2) * tenzor::exp(tenzor::neg(abs_diff / b1))
                 + tenzor::log(b2 / b1) - 1.0f;
        }
    }
    // --- Gamma || Exponential (audit E.6) ---
    // Exponential(λ) = Gamma(1, λ); use the Gamma||Gamma formula specialised
    // to α2=1, β2=λ:
    //   KL(Gamma(a,b) || Exp(λ)) = (a-1)·ψ(a) - lgamma(a) + log(b)
    //                              + log(λ) · (-1) ... actually derived:
    //   Plug a2=1, b2=λ into Gamma||Gamma:
    //     (a-1)ψ(a) - lgamma(a) + lgamma(1) + 1·log(b/λ) + a·(λ/b - 1)
    //   = (a-1)ψ(a) - lgamma(a) + log(b) - log(λ) + a·λ/b - a
    if (auto* pg = dynamic_cast<Gamma*>(&p)) {
        if (auto* qe = dynamic_cast<Exponential*>(&q)) {
            auto a  = pg->concentration();
            auto b  = pg->rate();
            auto lam = qe->rate();
            return (a - 1.0f) * tenzor::digamma(a)
                 - tenzor::lgamma(a)
                 + tenzor::log(b) - tenzor::log(lam)
                 + a * lam / b
                 - a;
        }
    }
    // --- Gamma || Gamma ---
    if (auto* pg = dynamic_cast<Gamma*>(&p)) {
        if (auto* qg = dynamic_cast<Gamma*>(&q)) {
            // KL = (α1-α2)ψ(α1) - lgamma(α1) + lgamma(α2)
            //      + α2·log(β1/β2) + α1(β2/β1 - 1)
            auto a1 = pg->concentration();
            auto a2 = qg->concentration();
            auto b1 = pg->rate();
            auto b2 = qg->rate();
            return (a1 - a2) * tenzor::digamma(a1)
                 - tenzor::lgamma(a1) + tenzor::lgamma(a2)
                 + a2 * tenzor::log(b1 / b2)
                 + a1 * (b2 / b1 - 1.0f);
        }
    }
    // --- Beta || Beta ---
    if (auto* pb = dynamic_cast<Beta*>(&p)) {
        if (auto* qb = dynamic_cast<Beta*>(&q)) {
            // KL = lgamma(a1) + lgamma(b1) - lgamma(a1+b1)
            //    - lgamma(a2) - lgamma(b2) + lgamma(a2+b2)
            //    + (a1-a2)ψ(a1) + (b1-b2)ψ(b1)
            //    - (a1+b1-a2-b2)ψ(a1+b1)
            //  (note: sign flipped vs some references because
            //   we compute lgamma(q) - lgamma(p) terms)
            auto a1 = pb->concentration1();
            auto b1 = pb->concentration0();
            auto a2 = qb->concentration1();
            auto b2 = qb->concentration0();
            auto sum1 = a1 + b1;
            auto sum2 = a2 + b2;
            return tenzor::lgamma(a2) + tenzor::lgamma(b2) - tenzor::lgamma(sum2)
                 - tenzor::lgamma(a1) - tenzor::lgamma(b1) + tenzor::lgamma(sum1)
                 + (a1 - a2) * tenzor::digamma(a1)
                 + (b1 - b2) * tenzor::digamma(b1)
                 - (sum1 - sum2) * tenzor::digamma(sum1);
        }
    }
    // --- Categorical || Categorical ---
    if (auto* pc = dynamic_cast<Categorical*>(&p)) {
        if (auto* qc = dynamic_cast<Categorical*>(&q)) {
            // KL = Σ p_i (log p_i - log q_i)
            auto eps = 1e-7f;
            auto p_probs = tenzor::clamp(pc->probs(), eps, 1.0f);
            auto q_probs = tenzor::clamp(qc->probs(), eps, 1.0f);
            auto t = p_probs * (tenzor::log(p_probs) - tenzor::log(q_probs));
            // Reduce only over the category axis -> per-batch KL of shape (B,).
            return tenzor::sum(t, /*dim=*/-1);
        }
    }
    // --- Categorical || OneHotCategorical (audit E.6) ---
    // OneHotCategorical is the same distribution as Categorical (just a
    // different sample parameterisation); KL collapses to Categorical||Categorical
    // over the underlying probs vectors.
    if (auto* pc = dynamic_cast<Categorical*>(&p)) {
        if (auto* qoh = dynamic_cast<OneHotCategorical*>(&q)) {
            auto eps = 1e-7f;
            auto p_probs = tenzor::clamp(pc->probs(), eps, 1.0f);
            auto q_probs = tenzor::clamp(qoh->probs(), eps, 1.0f);
            auto t = p_probs * (tenzor::log(p_probs) - tenzor::log(q_probs));
            // Reduce only over the category axis -> per-batch KL of shape (B,).
            return tenzor::sum(t, /*dim=*/-1);
        }
    }
    // --- OneHotCategorical || Categorical (symmetric helper) ---
    if (auto* poh = dynamic_cast<OneHotCategorical*>(&p)) {
        if (auto* qc = dynamic_cast<Categorical*>(&q)) {
            auto eps = 1e-7f;
            auto p_probs = tenzor::clamp(poh->probs(), eps, 1.0f);
            auto q_probs = tenzor::clamp(qc->probs(), eps, 1.0f);
            auto t = p_probs * (tenzor::log(p_probs) - tenzor::log(q_probs));
            // Reduce only over the category axis -> per-batch KL of shape (B,).
            return tenzor::sum(t, /*dim=*/-1);
        }
    }
    // --- OneHotCategorical || OneHotCategorical (audit E.6) ---
    if (auto* poh = dynamic_cast<OneHotCategorical*>(&p)) {
        if (auto* qoh = dynamic_cast<OneHotCategorical*>(&q)) {
            auto eps = 1e-7f;
            auto p_probs = tenzor::clamp(poh->probs(), eps, 1.0f);
            auto q_probs = tenzor::clamp(qoh->probs(), eps, 1.0f);
            auto t = p_probs * (tenzor::log(p_probs) - tenzor::log(q_probs));
            // Reduce only over the category axis -> per-batch KL of shape (B,).
            return tenzor::sum(t, /*dim=*/-1);
        }
    }
    // --- Dirichlet || Dirichlet ---
    if (auto* pd = dynamic_cast<Dirichlet*>(&p)) {
        if (auto* qd = dynamic_cast<Dirichlet*>(&q)) {
            // KL = lgamma(Σα1) - lgamma(Σα2) - Σlgamma(α1) + Σlgamma(α2)
            //      + Σ(α1-α2)(ψ(α1) - ψ(Σα1))
            auto a1 = pd->concentration();
            auto a2 = qd->concentration();
            // Reduce only over the last (event/simplex) axis so batched
            // concentration (B, K) yields per-batch KL of shape (B, 1).
            auto sum_a1 = tenzor::sum(a1, -1, /*keepdim=*/true);
            auto sum_a2 = tenzor::sum(a2, -1, /*keepdim=*/true);
            return tenzor::lgamma(sum_a1) - tenzor::lgamma(sum_a2)
                 - tenzor::sum(tenzor::lgamma(a1), -1, /*keepdim=*/true)
                 + tenzor::sum(tenzor::lgamma(a2), -1, /*keepdim=*/true)
                 + tenzor::sum((a1 - a2) * (tenzor::digamma(a1) - tenzor::digamma(sum_a1)),
                               -1, /*keepdim=*/true);
        }
    }
    // --- Poisson || Poisson ---
    if (auto* pp = dynamic_cast<Poisson*>(&p)) {
        if (auto* qp = dynamic_cast<Poisson*>(&q)) {
            // KL = λ1(log λ1 - log λ2) - (λ1 - λ2)
            auto r1 = pp->rate();
            auto r2 = qp->rate();
            return r1 * (tenzor::log(r1) - tenzor::log(r2)) - (r1 - r2);
        }
    }
    // --- LogNormal || LogNormal ---
    if (auto* pln = dynamic_cast<LogNormal*>(&p)) {
        if (auto* qln = dynamic_cast<LogNormal*>(&q)) {
            // KL(LogNormal(μ1,σ1) || LogNormal(μ2,σ2))
            //   = KL(Normal(μ1,σ1) || Normal(μ2,σ2))
            auto mu1 = pln->loc(); auto s1 = pln->scale();
            auto mu2 = qln->loc(); auto s2 = qln->scale();
            auto v1 = s1 * s1;
            auto v2 = s2 * s2;
            auto diff = mu1 - mu2;
            return tenzor::log(s2 / s1) + (v1 + diff * diff) / (v2 * 2.0f) - 0.5f;
        }
    }
    // --- MultivariateNormal || MultivariateNormal (audit E.6) ---
    // General closed form:
    //   KL(N(μ1,Σ1) || N(μ2,Σ2))
    //     = 0.5 · [ tr(Σ2⁻¹ Σ1) + (μ2-μ1)ᵀ Σ2⁻¹ (μ2-μ1) - k + log(|Σ2|/|Σ1|) ]
    // Implementation notes:
    //   * tr(Σ2⁻¹ Σ1) = sum over diagonal of solve(Σ2, Σ1).
    //   * Mahalanobis term via the q-side scale_tril cached at construction.
    //   * log|Σ| = 2·Σ log(diag(L)) using each distribution's cached Cholesky.
    if (auto* pm = dynamic_cast<MultivariateNormal*>(&p)) {
        if (auto* qm = dynamic_cast<MultivariateNormal*>(&q)) {
            const auto& mu1 = pm->loc();
            const auto& mu2 = qm->loc();
            const auto& s1  = pm->scale_tril();
            const auto& s2  = qm->scale_tril();
            const auto& cov1 = pm->covariance_matrix();
            const int64_t k = mu1.shape().back();

            // log|Σ| = 2 · Σ log(diag(L))
            auto log_det1 = 2.0f * tenzor::sum(tenzor::log(tenzor::diag(s1)));
            auto log_det2 = 2.0f * tenzor::sum(tenzor::log(tenzor::diag(s2)));

            // tr(Σ2⁻¹ Σ1) via Σ2⁻¹ Σ1 = cholesky_solve(Σ1, L2):
            auto sigma2_inv_sigma1 = tenzor::linalg::cholesky_solve(cov1, s2, /*upper=*/false);
            auto trace_term = tenzor::sum(tenzor::diag(sigma2_inv_sigma1));

            // Mahalanobis term: (μ2-μ1)ᵀ Σ2⁻¹ (μ2-μ1) = ||L2⁻¹ (μ2-μ1)||².
            auto diff = mu2 - mu1;
            auto diff_col = tenzor::unsqueeze(diff, -1);
            auto y_col = tenzor::linalg::solve_triangular(s2, diff_col, /*upper=*/false);
            const auto& diff_shape = diff.shape();
            std::vector<int64_t> y_shape(diff_shape.begin(), diff_shape.end());
            auto y = y_col.reshape(y_shape);
            auto mahal = tenzor::sum(y * y);

            return 0.5f * (trace_term + mahal - static_cast<float>(k)
                           + log_det2 - log_det1);
        }
    }
    // --- LowRankMultivariateNormal || MultivariateNormal (audit E.6) ---
    // Reconstruct Σ1 = W Wᵀ + diag(D) and plug into the general MVN formula.
    if (auto* pl = dynamic_cast<LowRankMultivariateNormal*>(&p)) {
        if (auto* qm = dynamic_cast<MultivariateNormal*>(&q)) {
            const auto& mu1 = pl->loc();
            const auto& mu2 = qm->loc();
            const auto& W   = pl->cov_factor();      // (..., k, r)
            const auto& D   = pl->cov_diag();        // (..., k)
            const auto& s2  = qm->scale_tril();
            const int64_t k = mu1.shape().back();

            // Σ1 = W Wᵀ + diag(D)
            auto Wt = tenzor::transpose(W, -2, -1);
            auto cov1 = tenzor::matmul(W, Wt)
                      + tenzor::diag(D);  // diag promotes (k,) -> (k,k)

            // log|Σ1| via matrix-determinant lemma:
            //   log|D + W Wᵀ| = log|D| + log|I_r + Wᵀ D⁻¹ W|
            auto d_inv = tenzor::reciprocal(D);
            auto d_inv_w = tenzor::unsqueeze(d_inv, -1) * W;
            auto capacitance = tenzor::eye(W.shape().back(), std::nullopt,
                                           mu1.dtype(), mu1.device())
                             + tenzor::matmul(Wt, d_inv_w);
            auto cap_chol = tenzor::linalg::cholesky(capacitance);
            auto log_det_d   = tenzor::sum(tenzor::log(D));
            auto log_det_cap = 2.0f * tenzor::sum(
                tenzor::log(tenzor::abs(tenzor::diag(cap_chol))));
            auto log_det1 = log_det_d + log_det_cap;

            // log|Σ2| = 2 · Σ log(diag(L2))
            auto log_det2 = 2.0f * tenzor::sum(tenzor::log(tenzor::diag(s2)));

            // tr(Σ2⁻¹ Σ1)
            auto sigma2_inv_sigma1 = tenzor::linalg::cholesky_solve(cov1, s2, /*upper=*/false);
            auto trace_term = tenzor::sum(tenzor::diag(sigma2_inv_sigma1));

            // Mahalanobis
            auto diff = mu2 - mu1;
            auto diff_col = tenzor::unsqueeze(diff, -1);
            auto y_col = tenzor::linalg::solve_triangular(s2, diff_col, /*upper=*/false);
            const auto& diff_shape = diff.shape();
            std::vector<int64_t> y_shape(diff_shape.begin(), diff_shape.end());
            auto y = y_col.reshape(y_shape);
            auto mahal = tenzor::sum(y * y);

            return 0.5f * (trace_term + mahal - static_cast<float>(k)
                           + log_det2 - log_det1);
        }
    }
    // Audit item E.6: provide a Monte-Carlo fallback for pairs without a
    // registered closed form (e.g. Normal||Cauchy, Normal||LogNormal).
    //
    // Standard estimator:
    //   KL(p || q) ≈ (1/N) Σ_i [ log_prob_p(x_i) - log_prob_q(x_i) ],  x_i ~ p
    //
    // We use N=1024 samples — accurate to ~2% for well-behaved supports.
    // Users wanting tighter bounds can stack multiple calls.
    //
    // The fallback uses rsample() when available (preserves gradients through
    // the sampler for reparameterizable p); otherwise sample() (no grad).
    // When supports don't overlap, log_q is -inf on some samples and the
    // mean becomes NaN/+inf — we detect this via has_inf_nan() and raise
    // a clear error instead of returning silent garbage.
    constexpr int kMcSamples = 1024;
    Tensor samples;
    try {
        samples = p.rsample({kMcSamples});
    } catch (const std::exception&) {
        samples = p.sample({kMcSamples});
    }
    auto log_p = p.log_prob(samples);
    auto log_q = q.log_prob(samples);
    auto diff = log_p - log_q;
    // Mean over the leading sample axis.
    auto result = tenzor::mean(diff, /*dim=*/0, /*keepdim=*/false);
    // Hard fail (no silent fallback) if supports don't overlap or the
    // estimator otherwise blows up — caller needs to know.
    if (tenzor::has_inf_nan(result).data<bool>()[0]) {
        throw std::runtime_error(
            "kl_divergence: no closed form is registered for this distribution "
            "pair, and the Monte-Carlo fallback produced NaN/Inf (likely the "
            "supports do not overlap or log_q(x_i) = -inf for some sample). "
            "Provide a closed-form KL or ensure supp(p) ⊆ supp(q).");
    }
    return result;
}
} // namespace distributions
} // namespace tenzor
