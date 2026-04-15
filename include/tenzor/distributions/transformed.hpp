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
