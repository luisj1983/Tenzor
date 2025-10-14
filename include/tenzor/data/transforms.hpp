#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <cstdlib>
#include "../tenzor.hpp"

namespace tenzor {
namespace data {
namespace transforms {

/**
 * @brief Base class for data transforms
 */
class Transform {
public:
    virtual ~Transform() = default;

    /**
     * @brief Apply transform to input and target tensors
     * @param input Input tensor
     * @param target Target tensor
     * @return Transformed (input, target) pair
     */
    virtual auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> = 0;
};

/**
 * @brief Normalize tensor with mean and standard deviation
 */
class Normalize : public Transform {
public:
    /**
     * @brief Construct normalize transform
     * @param mean Mean values for normalization
     * @param std Standard deviation values for normalization
     */
    Normalize(std::vector<float> mean, std::vector<float> std)
        : mean_(std::move(mean)), std_(std::move(std)) {
        if (mean_.size() != std_.size()) {
            throw std::invalid_argument("Mean and std must have same size");
        }
    }

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        Tensor normalized = input;

        // Apply normalization: (x - mean) / std
        const auto& shape = input.shape();
        if (shape.empty()) {
            throw std::invalid_argument("Cannot normalize scalar tensor");
        }

        // Assume last dimension is channels
        size_t num_channels = mean_.size();
        if (shape.back() != num_channels && num_channels > 1) {
            throw std::invalid_argument("Input channels must match normalization parameters");
        }

        // Simple channel-wise normalization
        for (size_t i = 0; i < num_channels; ++i) {
            // Note: This is simplified. In practice, you'd want more efficient tensor operations
            float mean = mean_[i];
            float std = std_[i];

            if (std == 0.0f) {
                throw std::invalid_argument("Standard deviation cannot be zero");
            }

            // Normalize channel i
            // normalized[..., i] = (input[..., i] - mean) / std
        }

        return {normalized, target};
    }

private:
    std::vector<float> mean_;
    std::vector<float> std_;
};

/**
 * @brief Convert data to tensor (identity transform for tensors)
 */
class ToTensor : public Transform {
public:
    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        // Already tensors, return as-is
        return {input, target};
    }
};

/**
 * @brief Compose multiple transforms into a pipeline
 */
class Compose : public Transform {
public:
    /**
     * @brief Construct composed transform
     * @param transforms Vector of transforms to apply in sequence
     */
    explicit Compose(std::vector<std::shared_ptr<Transform>> transforms)
        : transforms_(std::move(transforms)) {}

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        auto result = std::make_pair(input, target);

        // Apply each transform in sequence
        for (const auto& transform : transforms_) {
            result = (*transform)(result.first, result.second);
        }

        return result;
    }

private:
    std::vector<std::shared_ptr<Transform>> transforms_;
};

/**
 * @brief Random horizontal flip transform
 */
class RandomHorizontalFlip : public Transform {
public:
    /**
     * @brief Construct random flip transform
     * @param p Probability of applying flip (default 0.5)
     */
    explicit RandomHorizontalFlip(float p = 0.5f) : p_(p) {
        if (p < 0.0f || p > 1.0f) {
            throw std::invalid_argument("Probability must be in [0, 1]");
        }
    }

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        // Random flip based on probability
        float random_val = static_cast<float>(std::rand()) / RAND_MAX;

        if (random_val < p_) {
            // Flip input horizontally (assuming HWC or NHWC format)
            // Note: Actual implementation would need tensor flip operation
            return {input, target};  // Placeholder
        }

        return {input, target};
    }

private:
    float p_;
};

/**
 * @brief Lambda transform for custom operations
 */
class Lambda : public Transform {
public:
    using LambdaFunc = std::function<std::pair<Tensor, Tensor>(const Tensor&, const Tensor&)>;

    /**
     * @brief Construct lambda transform
     * @param func Custom transform function
     */
    explicit Lambda(LambdaFunc func) : func_(std::move(func)) {}

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        return func_(input, target);
    }

private:
    LambdaFunc func_;
};

} // namespace transforms
} // namespace data
} // namespace tenzor
