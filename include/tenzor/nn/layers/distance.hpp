/**
 * @file distance.hpp
 * @brief Distance computation modules (CosineSimilarity, PairwiseDistance)
 */

#pragma once

#include "../../autograd/ops.hpp"
#include "../../ops/creation.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Cosine similarity module.
 *
 * Computes cosine similarity between two input tensors along a dimension:
 * \f[
 * \text{similarity} = \frac{x_1 \cdot x_2}{\max(\|x_1\|_2, \epsilon) \cdot \max(\|x_2\|_2, \epsilon)}
 * \f]
 *
 * @param dim Dimension along which to compute similarity (default: 1)
 * @param eps Small value to avoid division by zero (default: 1e-8)
 */
class CosineSimilarity {
public:
    explicit CosineSimilarity(int64_t dim = 1, double eps = 1e-8)
        : dim_(dim), eps_(eps) {}

    auto forward(const Variable& x1, const Variable& x2) -> Variable {
        return tenzor::cosine_similarity(x1, x2, dim_, eps_);
    }

    auto operator()(const Variable& x1, const Variable& x2) -> Variable {
        return forward(x1, x2);
    }

private:
    int64_t dim_;
    double eps_;
};

/**
 * @brief Pairwise distance module.
 *
 * Computes the p-norm distance between every pair of row vectors:
 * \f[
 * \text{dist}(x_1, x_2) = \|x_1 - x_2 + \epsilon\|_p
 * \f]
 *
 * @param p Norm degree (default: 2.0)
 * @param eps Small value to avoid division by zero (default: 1e-6)
 * @param keepdim Whether to keep the dimension (default: false)
 */
class PairwiseDistance {
public:
    explicit PairwiseDistance(double p = 2.0, double eps = 1e-6, bool keepdim = false)
        : p_(p), eps_(eps), keepdim_(keepdim) {}

    auto forward(const Variable& x1, const Variable& x2) -> Variable {
        // Compute ||x1 - x2||_p via tensor ops with autograd
        using namespace tenzor;
        auto diff = x1 - x2;
        // Add eps to the signed difference before taking the norm, matching the
        // documented formula ||x1 - x2 + eps||_p (PyTorch semantics).
        auto shape = diff.tensor().shape();
        auto eps_var = Variable(tenzor::full({shape.begin(), shape.end()}, static_cast<float>(eps_),
                                             diff.tensor().dtype(), diff.tensor().device()), false);
        diff = diff + eps_var;
        auto abs_diff = abs(diff);

        if (p_ == 1.0) {
            return sum(abs_diff, -1, keepdim_);
        } else if (p_ == 2.0) {
            auto sq = abs_diff * abs_diff;
            auto s = sum(sq, -1, keepdim_);
            return sqrt(s);
        } else {
            auto powered = pow(abs_diff, static_cast<float>(p_));
            auto s = sum(powered, -1, keepdim_);
            return pow(s, static_cast<float>(1.0 / p_));
        }
    }

    auto operator()(const Variable& x1, const Variable& x2) -> Variable {
        return forward(x1, x2);
    }

private:
    double p_;
    double eps_;
    bool keepdim_;
};

} // namespace nn
} // namespace tenzor
