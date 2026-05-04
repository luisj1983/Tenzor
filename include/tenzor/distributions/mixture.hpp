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
        // Tenzor convention: sample_shape is the FULL desired output shape.
        // To produce a mixture sample of that shape we need:
        //   1. one mixture index per output position  (mix_sample shape == sample_shape)
        //   2. one component value per (output position, component k) pair
        //      (comp_samples shape == sample_shape + (K,))
        // Then gather_components selects the right component per position.
        //
        // audit-2026-05-03 bug #6: previously this passed sample_shape
        // unchanged to component_, which failed for non-empty sample_shape
        // because the component's batch_shape (which already encodes the K
        // dim) doesn't broadcast with the user's requested output shape.
        const int64_t K = log_weights_.shape().back();

        std::vector<int64_t> comp_sample_shape;
        if (!sample_shape.empty()) {
            comp_sample_shape = sample_shape;
            comp_sample_shape.push_back(K);
        }
        auto comp_samples = component_->sample(comp_sample_shape);
        auto mix_sample = mixture_->sample(sample_shape);

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
        // comp_samples shape: prefix_shape + (K,) + event_dims
        //   where prefix_shape = sample_shape + batch_shape
        // indices shape:      prefix_shape (one int per prefix point)
        // result shape:       prefix_shape + event_dims
        //
        // audit-2026-05-03 bug #6: the previous heuristic computed
        //   comp_dim = comp_samples.ndim() - indices.ndim() - 1
        // and fell back to `indices.ndim()` if negative. Categorical's
        // `sample({})` (Tenzor convention) returns shape (1,) rather than
        // (), so indices.ndim() ended up larger than comp_samples.ndim()
        // and the `target_shape[comp_dim] = 1;` write was OOB on the
        // shape vector, tripping a libstdc++ debug assertion.
        //
        // The only fully-supported case today is event_dims = () (scalar
        // event), which is what every distribution exposed via the
        // (weights, base) ctor produces. Under that assumption, the K
        // dim is always the LAST dim of comp_samples.
        const int64_t comp_dim = comp_samples.ndim() - 1;

        // Align indices to have ndim = comp_dim by squeezing trailing
        // length-1 dims (handles Categorical's (1,) "single sample" dim
        // when no batch shape is requested) or by inserting leading
        // length-1 dims when indices is too low-rank.
        auto aligned_idx = indices;
        while (aligned_idx.ndim() > comp_dim) {
            bool squeezed = false;
            for (int64_t d = aligned_idx.ndim() - 1; d >= 0; --d) {
                if (aligned_idx.shape()[d] == 1) {
                    aligned_idx = tenzor::squeeze(aligned_idx, d);
                    squeezed = true;
                    break;
                }
            }
            if (!squeezed) break;
        }
        while (aligned_idx.ndim() < comp_dim) {
            aligned_idx = tenzor::unsqueeze(aligned_idx, 0);
        }

        // Re-introduce the K position as a length-1 dim, then expand to
        // comp_samples' shape (with that K position pinned to 1).
        auto idx_with_K = tenzor::unsqueeze(aligned_idx, comp_dim);
        auto target_shape = std::vector<int64_t>(
            comp_samples.shape().begin(), comp_samples.shape().end());
        target_shape[comp_dim] = 1;
        auto idx_expanded = tenzor::expand(idx_with_K, target_shape);

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
