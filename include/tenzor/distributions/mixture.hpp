/**
 * @file mixture.hpp
 * @brief MixtureSameFamily — mixture model with same-family components
 *
 * Equivalent to PyTorch's torch.distributions.MixtureSameFamily.
 * Used for Gaussian mixture models and other mixture distributions.
 */

#pragma once

#include "distribution.hpp"
#include "../ops/creation.hpp"
#include "../ops/math.hpp"
#include "../ops/reduction.hpp"
#include "../ops/indexing.hpp"
#include <memory>

namespace tenzor {
namespace distributions {

/**
 * @brief Mixture distribution where all components are from the same family.
 *
 * Models p(x) = sum_k w_k * p_k(x) where w_k are mixture weights from
 * a Categorical distribution and p_k are component distributions of the
 * same type (e.g., all Normal).
 *
 * @code
 * // Gaussian mixture with 3 components
 * auto mix = std::make_shared<Categorical>(tensor({0.3f, 0.5f, 0.2f}));
 * auto comp = std::make_shared<Normal>(
 *     tensor({-1.0f, 0.0f, 1.0f}),   // means
 *     tensor({0.5f, 0.3f, 0.5f})     // stds
 * );
 * MixtureSameFamily gmm(mix, comp);
 * @endcode
 */
class MixtureSameFamily : public Distribution {
public:
    /**
     * @param mixture_distribution Categorical distribution over K components
     * @param component_distribution Distribution with rightmost batch dim K
     * @param mixture_logits Log-probabilities of each mixture component (pre-normalized)
     */
    MixtureSameFamily(std::shared_ptr<Distribution> mixture_distribution,
                     std::shared_ptr<Distribution> component_distribution,
                     Tensor mixture_logits)
        : mixture_(std::move(mixture_distribution))
        , component_(std::move(component_distribution))
        , log_weights_(std::move(mixture_logits))
    {}

    /**
     * @brief Convenience constructor from unnormalized weights.
     * @param weights Tensor of mixture weights (will be normalized)
     * @param component_distribution Distribution with rightmost batch dim K
     */
    MixtureSameFamily(const Tensor& weights,
                     std::shared_ptr<Distribution> component_distribution)
        : mixture_(std::make_shared<Categorical>(weights))
        , component_(std::move(component_distribution))
    {
        // Normalize and take log
        auto sum_w = tenzor::sum(weights, -1);
        auto normed = weights / tenzor::unsqueeze(sum_w, -1);
        log_weights_ = tenzor::log(normed);
    }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        // Sample mixture indices
        auto mix_sample = mixture_->sample(sample_shape);  // shape: sample_shape

        // Sample from all components
        auto comp_samples = component_->sample(sample_shape);  // shape: sample_shape + (..., K, event_shape)

        // Gather the selected component's sample
        // mix_sample has indices into the K components
        return gather_components(comp_samples, mix_sample);
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        // log p(x) = logsumexp(log w_k + log p_k(x))
        // Get mixture log probabilities (log weights)
        auto mix_lp = mixture_log_probs();  // shape: (..., K)

        // Expand value for all K components and get component log probs
        auto comp_lp = component_->log_prob(
            expand_for_components(value));  // shape: (..., K)

        // logsumexp over the component dimension
        auto combined = mix_lp + comp_lp;
        return logsumexp_last_dim(combined);
    }

    /** @brief Access the mixture distribution */
    auto mixture_dist() const -> const Distribution& { return *mixture_; }

    /** @brief Access the component distribution */
    auto component_dist() const -> const Distribution& { return *component_; }

private:
    std::shared_ptr<Distribution> mixture_;
    std::shared_ptr<Distribution> component_;
    Tensor log_weights_;

    auto mixture_log_probs() -> Tensor {
        return log_weights_;
    }

    static auto gather_components(const Tensor& comp_samples,
                                  const Tensor& indices) -> Tensor {
        // comp_samples: (..., K, event_dims...)
        // indices: (...) with values in [0, K)
        // result: (..., event_dims...)
        auto idx = tenzor::unsqueeze(indices, -1);  // (..., 1)

        // For each sample, select the component at the index
        // Use gather along the component dimension
        int64_t comp_dim = comp_samples.ndim() - indices.ndim() - 1;
        if (comp_dim < 0) comp_dim = indices.ndim();

        // Expand idx to match event dims
        auto target_shape = std::vector<int64_t>(
            comp_samples.shape().begin(), comp_samples.shape().end());
        target_shape[comp_dim] = 1;

        auto idx_expanded = tenzor::expand(idx, target_shape);
        auto gathered = tenzor::gather(comp_samples, comp_dim, idx_expanded);
        return tenzor::squeeze(gathered, comp_dim);
    }

    static auto expand_for_components(const Tensor& value) -> Tensor {
        // Add a component dimension and expand
        return tenzor::unsqueeze(value, -1);
    }

    static auto logsumexp_last_dim(const Tensor& x) -> Tensor {
        // logsumexp along the last dimension
        auto max_val = tenzor::max(x, -1);
        auto shifted = x - tenzor::unsqueeze(max_val, -1);
        auto log_sum = tenzor::log(tenzor::sum(tenzor::exp(shifted), -1));
        return max_val + log_sum;
    }
};

} // namespace distributions
} // namespace tenzor
