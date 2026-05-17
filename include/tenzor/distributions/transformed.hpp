/**
 * @file transformed.hpp
 * @brief TransformedDistribution — applies bijective transforms to a base distribution
 *
 * Equivalent to PyTorch's torch.distributions.TransformedDistribution.
 * Used for normalizing flows and reparameterized distributions.
 */

#pragma once

#include "distribution.hpp"
#include "transforms.hpp"
#include "../ops/reduction.hpp"
#include <memory>
#include <vector>

namespace tenzor {
namespace distributions {

/**
 * @brief Distribution formed by applying a chain of transforms to a base distribution.
 *
 * Given a base distribution p(x) and transforms f_1, f_2, ..., f_n:
 *   y = f_n(f_{n-1}(...f_1(x)))
 *   log p(y) = log p(f^{-1}(y)) - log|det(J)|
 *
 * @code
 * auto base = std::make_shared<Normal>(zeros({2}), ones({2}));
 * auto transforms = std::vector<std::shared_ptr<Transform>>{
 *     std::make_shared<ExpTransform>()
 * };
 * TransformedDistribution lognorm(base, transforms);
 * // lognorm is equivalent to LogNormal(0, 1)
 * @endcode
 */
class TransformedDistribution : public Distribution {
public:
    TransformedDistribution(std::shared_ptr<Distribution> base,
                           std::vector<std::shared_ptr<Transform>> transforms)
        : base_(std::move(base)), transforms_(std::move(transforms)) {}

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        Tensor x = base_->sample(std::move(sample_shape));
        for (auto& t : transforms_) {
            x = t->call(x);
        }
        return x;
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        Tensor x = base_->rsample(std::move(sample_shape));
        for (auto& t : transforms_) {
            x = t->call(x);
        }
        return x;
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // Invert transforms to get back to base space
        Tensor x = value;
        Tensor log_det = tenzor::zeros(
            std::vector<int64_t>(value.shape().begin(), value.shape().end()),
            value.dtype(), value.device());

        for (auto it = transforms_.rbegin(); it != transforms_.rend(); ++it) {
            Tensor prev = (*it)->inv(x);
            log_det = log_det + (*it)->log_abs_det_jacobian(prev, x);
            x = prev;
        }

        return base_->log_prob(x) - log_det;
    }


    /**
     * @brief Mean via Monte-Carlo (general transforms have no closed form).
     *
     * For bijective transforms the MC estimator IS the correct definition:
     *   E[Y] = E[T(X)] ≈ (1/N) Σ T(X_i), X_i ~ base.
     */
    auto mean() -> Tensor override {
        return distributions::detail::mc_mean(
            [this]() { return this->sample({}); });
    }

    /** @brief Element-wise variance via Monte-Carlo. */
    auto variance() -> Tensor override {
        return distributions::detail::mc_variance(
            [this]() { return this->sample({}); });
    }

    /**
     * @brief Differential entropy.
     *
     * For a bijective transform Y = T(X):
     *   H(Y) = H(X) + E[log|det J_T(X)|]
     *
     * We compute the second term by Monte-Carlo and add the base entropy.
     * If the base distribution doesn't supply a closed-form entropy, we
     * fall through to a pure-MC estimator on Y (H = -E[log p(Y)]).
     */
    auto entropy() -> Tensor override {
        try {
            Tensor base_H = base_->entropy();

            // E[ Σ_t log|det J_t| ] via Monte-Carlo over the chain.
            const int N = 4096;
            std::vector<Tensor> log_det_terms;
            log_det_terms.reserve(static_cast<size_t>(N));
            for (int i = 0; i < N; ++i) {
                Tensor x = base_->rsample({});
                Tensor accum = tenzor::zeros(
                    std::vector<int64_t>(x.shape().begin(), x.shape().end()),
                    x.dtype(), x.device());
                for (auto& t : transforms_) {
                    Tensor y = t->call(x);
                    accum = accum + t->log_abs_det_jacobian(x, y);
                    x = std::move(y);
                }
                log_det_terms.push_back(accum);
            }
            auto stacked = tenzor::stack(
                std::span<const Tensor>(log_det_terms.data(),
                                        log_det_terms.size()), /*dim=*/0);
            auto E_logdet = tenzor::mean(stacked, /*dim=*/0, /*keepdim=*/false);
            return base_H + E_logdet;
        } catch (const std::runtime_error&) {
            // Base distribution has no closed-form entropy; fall back to MC
            // directly on the transformed distribution.
            return distributions::detail::mc_entropy(
                [this]() { return this->sample({}); },
                [this](const Tensor& s) { return this->log_prob(s); });
        }
    }

    /** @brief Access the base distribution */
    auto base_dist() const -> const Distribution& { return *base_; }

    /** @brief Access the transforms */
    auto transforms() const -> const std::vector<std::shared_ptr<Transform>>& {
        return transforms_;
    }

private:
    std::shared_ptr<Distribution> base_;
    std::vector<std::shared_ptr<Transform>> transforms_;
};

} // namespace distributions
} // namespace tenzor
