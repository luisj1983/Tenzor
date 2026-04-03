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
#include <vector>
#include <cmath>
#include <optional>
#include <stdexcept>

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

    auto log_prob(const Tensor& value) -> Tensor override {
        auto var = scale_ * scale_;
        auto log_scale = tenzor::log(scale_);
        auto diff = value - loc_;
        return -(diff * diff) / (var * 2.0f) - log_scale - 0.9189385332f; // -log(sqrt(2*pi))
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

    auto log_prob(const Tensor& value) -> Tensor override {
        auto log_probs = tenzor::log(probs_);
        return gather(log_probs, -1, value.to(DType::Int64));
    }

    auto entropy() -> Tensor override {
        auto log_probs = tenzor::log(probs_);
        return -(probs_ * log_probs).sum();
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

    auto log_prob(const Tensor& value) -> Tensor override {
        auto eps = 1e-7f;
        auto clamped = tenzor::clamp(probs_, eps, 1.0f - eps);
        return value * tenzor::log(clamped) + (1.0f - value) * tenzor::log(1.0f - clamped);
    }

    auto entropy() -> Tensor override {
        auto eps = 1e-7f;
        auto clamped = tenzor::clamp(probs_, eps, 1.0f - eps);
        return -(clamped * tenzor::log(clamped) + (1.0f - clamped) * tenzor::log(1.0f - clamped));
    }

    auto mean() -> Tensor override { return probs_; }
    auto variance() -> Tensor override { return probs_ * (1.0f - probs_); }

private:
    Tensor probs_;
};

} // namespace distributions
} // namespace tenzor
