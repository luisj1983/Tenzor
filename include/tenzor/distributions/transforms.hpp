/**
 * @file transforms.hpp
 * @brief Bijective transforms for use with TransformedDistribution
 *
 * Provides a Transform base class and common built-in transforms
 * (Exp, Affine, Sigmoid, Tanh, Softmax, Compose) for building
 * normalizing flows and reparameterized distributions.
 */

#pragma once

#include "../core/tensor.hpp"
#include "../ops/creation.hpp"
#include "../ops/math.hpp"
#include "../ops/reduction.hpp"
#include "../utils/error.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace distributions {

/**
 * @brief Abstract base class for invertible transforms.
 *
 * Each transform must implement forward (call), inverse (inv),
 * and the log absolute determinant of the Jacobian.
 */
class Transform {
public:
    virtual ~Transform() = default;

    /** @brief Apply the transform: y = f(x) */
    virtual auto call(const Tensor& x) -> Tensor = 0;

    /** @brief Apply the inverse transform: x = f^{-1}(y) */
    virtual auto inv(const Tensor& y) -> Tensor = 0;

    /**
     * @brief Log absolute determinant of the Jacobian df/dx.
     * @param x Input value
     * @param y Output value (y = f(x)), may be used for efficiency
     */
    virtual auto log_abs_det_jacobian(const Tensor& x, const Tensor& y) -> Tensor = 0;

    /**
     * @brief Number of trailing dimensions the transform treats as a single
     *        event (mirrors PyTorch's Transform.event_dim).
     *
     * For element-wise transforms (Exp, Affine, Sigmoid, Tanh) this is 0:
     * log_abs_det_jacobian already returns one value per element. A transform
     * that couples a trailing block of dims into one event returns the size of
     * that block, and TransformedDistribution sums the per-element log-det over
     * those trailing dims so its rank matches a reduced base log_prob.
     */
    virtual auto event_dim() const -> int64_t { return 0; }
};

/**
 * @brief Exponential transform: y = exp(x)
 */
class ExpTransform : public Transform {
public:
    auto call(const Tensor& x) -> Tensor override {
        return tenzor::exp(x);
    }

    auto inv(const Tensor& y) -> Tensor override {
        return tenzor::log(y);
    }

    auto log_abs_det_jacobian(const Tensor& x, const Tensor& /*y*/) -> Tensor override {
        return x;  // d/dx exp(x) = exp(x), log|exp(x)| = x
    }
};

/**
 * @brief Affine transform: y = loc + scale * x
 */
class AffineTransform : public Transform {
public:
    AffineTransform(Tensor loc, Tensor scale)
        : loc_(std::move(loc)), scale_(std::move(scale)) {}

    auto call(const Tensor& x) -> Tensor override {
        return loc_ + scale_ * x;
    }

    auto inv(const Tensor& y) -> Tensor override {
        return (y - loc_) / scale_;
    }

    auto log_abs_det_jacobian(const Tensor& /*x*/, const Tensor& /*y*/) -> Tensor override {
        return tenzor::log(tenzor::abs(scale_));
    }

private:
    Tensor loc_;
    Tensor scale_;
};

/**
 * @brief Sigmoid transform: y = 1 / (1 + exp(-x))
 */
class SigmoidTransform : public Transform {
public:
    auto call(const Tensor& x) -> Tensor override {
        return tenzor::sigmoid(x);
    }

    auto inv(const Tensor& y) -> Tensor override {
        // logit: x = log(y / (1 - y)). Sampled sigmoid outputs reach the open
        // endpoints {0,1} in finite precision, where log(0) = -inf poisons
        // log_prob/entropy with NaN. Clamp into (eps, 1-eps) first, matching the
        // domain clamping used by Exponential/Laplace/Bernoulli.
        constexpr double kEps = 1e-7;
        auto yc = tenzor::clamp(y, kEps, 1.0 - kEps);
        return tenzor::log(yc) - tenzor::log(1.0f - yc);
    }

    auto log_abs_det_jacobian(const Tensor& x, const Tensor& /*y*/) -> Tensor override {
        // log|sigmoid'(x)| = log(sigmoid(x) * (1 - sigmoid(x)))
        //                   = -softplus(-x) - softplus(x)
        auto sp = tenzor::log(1.0f + tenzor::exp(x));          // softplus(x)
        auto sm = tenzor::log(1.0f + tenzor::exp(tenzor::neg(x))); // softplus(-x)
        return tenzor::neg(sp) - sm;
    }
};

/**
 * @brief Tanh transform: y = tanh(x)
 */
class TanhTransform : public Transform {
public:
    auto call(const Tensor& x) -> Tensor override {
        return tenzor::tanh(x);
    }

    auto inv(const Tensor& y) -> Tensor override {
        // atanh(y) = 0.5 * log((1+y)/(1-y)). tanh outputs reach the open
        // endpoints {-1,1} in finite precision, where log(0) = -inf -> NaN.
        // Clamp into (-1+eps, 1-eps) first (common with tanh-squashed RL
        // policies), matching the clamping used elsewhere in this file.
        constexpr double kEps = 1e-7;
        auto yc = tenzor::clamp(y, -1.0 + kEps, 1.0 - kEps);
        return 0.5f * (tenzor::log(1.0f + yc) - tenzor::log(1.0f - yc));
    }

    auto log_abs_det_jacobian(const Tensor& /*x*/, const Tensor& y) -> Tensor override {
        // log(1 - tanh(x)^2) = log(1 - y^2). Clamp y away from {-1,1} so
        // 1 - y*y never underflows to 0, which would give log(0) = -inf.
        constexpr double kEps = 1e-7;
        auto yc = tenzor::clamp(y, -1.0 + kEps, 1.0 - kEps);
        return tenzor::log(1.0f - yc * yc);
    }
};

/**
 * @brief Softmax transform: y_i = exp(x_i) / sum(exp(x_j)) along a dimension
 */
class SoftmaxTransform : public Transform {
public:
    explicit SoftmaxTransform(int64_t dim = -1) : dim_(dim) {}

    auto call(const Tensor& x) -> Tensor override {
        auto max_val = tenzor::max(x, dim_);
        auto shifted = x - tenzor::unsqueeze(max_val, dim_);
        auto exp_vals = tenzor::exp(shifted);
        auto sum_exp = tenzor::sum(exp_vals, dim_);
        return exp_vals / tenzor::unsqueeze(sum_exp, dim_);
    }

    [[noreturn]] auto inv(const Tensor& /*y*/) -> Tensor override {
        // Softmax is NOT injective: log(softmax(x)) = x - logsumexp(x), which
        // differs from x by an arbitrary additive constant along `dim_`, so
        // log(y) is not a genuine inverse. Returning it silently corrupts
        // TransformedDistribution::log_prob (the base-space point is wrong by
        // that constant). Until a true simplex bijection (the K-1 dimensional
        // map PyTorch uses) is implemented, refuse rather than mislead.
        throw ::tenzor::error::DistributionMethodUndefined(
            "SoftmaxTransform::inv is not implemented: softmax is not injective, "
            "so log(y) is not its inverse and would yield an incorrect "
            "TransformedDistribution::log_prob. Use a bijective transform "
            "(e.g. StickBreaking) for flows over the simplex.");
    }

    [[noreturn]] auto log_abs_det_jacobian(const Tensor& /*x*/, const Tensor& /*y*/)
        -> Tensor override {
        // The previous implementation returned a zero tensor as an
        // "approximation"; that omits the change-of-variables correction in
        // TransformedDistribution::log_prob, silently producing an
        // unnormalized density. Throw rather than return a wrong value.
        throw ::tenzor::error::DistributionMethodUndefined(
            "SoftmaxTransform::log_abs_det_jacobian is not implemented: the exact "
            "simplex log-det Jacobian is required for a correct density. Use a "
            "bijective transform (e.g. StickBreaking) for flows over the simplex.");
    }

private:
    int64_t dim_;
};

/**
 * @brief Composition of multiple transforms: y = f_n(f_{n-1}(...f_1(x)))
 */
class ComposeTransform : public Transform {
public:
    explicit ComposeTransform(std::vector<std::shared_ptr<Transform>> transforms)
        : transforms_(std::move(transforms)) {}

    auto call(const Tensor& x) -> Tensor override {
        Tensor y = x;
        for (auto& t : transforms_) {
            y = t->call(y);
        }
        return y;
    }

    auto inv(const Tensor& y) -> Tensor override {
        Tensor x = y;
        for (auto it = transforms_.rbegin(); it != transforms_.rend(); ++it) {
            x = (*it)->inv(x);
        }
        return x;
    }

    auto log_abs_det_jacobian(const Tensor& x, const Tensor& y) -> Tensor override {
        // Chain rule: sum of log|det(J)| at each step
        Tensor total = tenzor::zeros(
            std::vector<int64_t>(x.shape().begin(), x.shape().end()),
            x.dtype(), x.device());
        Tensor current = x;
        for (auto& t : transforms_) {
            Tensor next = t->call(current);
            auto ladj = t->log_abs_det_jacobian(current, next);
            // Sum over event dimensions (all dims except batch)
            // For element-wise transforms, ladj has same shape as input
            total = total + ladj;
            current = next;
        }
        return total;
    }

private:
    std::vector<std::shared_ptr<Transform>> transforms_;
};

} // namespace distributions
} // namespace tenzor
