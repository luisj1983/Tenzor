/**
 * @file independent.hpp
 * @brief Independent distribution — reinterprets batch dims as event dims
 *
 * Equivalent to PyTorch's torch.distributions.Independent.
 * Useful for creating multivariate distributions from batches of
 * univariate distributions (e.g., diagonal Normal for VAE latent).
 */

#pragma once

#include "distribution.hpp"
#include "../ops/reduction.hpp"
#include <memory>

namespace tenzor {
namespace distributions {

/**
 * @brief Reinterprets trailing batch dimensions as event dimensions.
 *
 * This changes the result of log_prob() by summing over the
 * reinterpreted dimensions, effectively treating independent
 * univariate distributions as a single multivariate distribution.
 *
 * @code
 * // Create diagonal normal: 10-dim with independent components
 * auto base = std::make_shared<Normal>(zeros({10}), ones({10}));
 * Independent diag_normal(base, 1);
 * // base.log_prob(x) has shape {10}
 * // diag_normal.log_prob(x) has shape {} (scalar, summed over 10 dims)
 * @endcode
 */
class Independent : public Distribution {
public:
    /**
     * @param base_distribution The base distribution
     * @param reinterpreted_batch_ndims Number of trailing batch dims to
     *        reinterpret as event dims
     */
    Independent(std::shared_ptr<Distribution> base_distribution,
               int64_t reinterpreted_batch_ndims)
        : base_(std::move(base_distribution))
        , reinterpreted_batch_ndims_(reinterpreted_batch_ndims)
    {
        if (reinterpreted_batch_ndims < 0) {
            throw std::invalid_argument(
                "Independent: reinterpreted_batch_ndims must be >= 0");
        }
    }

    auto sample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return base_->sample(std::move(sample_shape));
    }

    auto rsample(std::vector<int64_t> sample_shape = {}) -> Tensor override {
        return base_->rsample(std::move(sample_shape));
    }

    auto log_prob(const Tensor& value) -> Tensor override {
        auto lp = base_->log_prob(value);
        // Sum over the last reinterpreted_batch_ndims_ dims
        for (int64_t i = 0; i < reinterpreted_batch_ndims_; ++i) {
            lp = tenzor::sum(lp, -1);
        }
        return lp;
    }

    auto entropy() -> Tensor override {
        auto ent = base_->entropy();
        for (int64_t i = 0; i < reinterpreted_batch_ndims_; ++i) {
            ent = tenzor::sum(ent, -1);
        }
        return ent;
    }

    auto mean() -> Tensor override {
        return base_->mean();
    }

    auto variance() -> Tensor override {
        return base_->variance();
    }

    /** @brief Access the base distribution */
    auto base_dist() const -> const Distribution& { return *base_; }

    /** @brief Number of reinterpreted batch dimensions */
    auto reinterpreted_batch_ndims() const -> int64_t {
        return reinterpreted_batch_ndims_;
    }

private:
    std::shared_ptr<Distribution> base_;
    int64_t reinterpreted_batch_ndims_;
};

} // namespace distributions
} // namespace tenzor
