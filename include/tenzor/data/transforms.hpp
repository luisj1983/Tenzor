#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <random>
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
 *
 * Applies channel-wise normalization: output[c] = (input[c] - mean[c]) / std[c]
 * Assumes the last dimension is the channel dimension.
 */
class Normalize : public Transform {
public:
    Normalize(std::vector<float> mean, std::vector<float> std)
        : mean_(std::move(mean)), std_(std::move(std)) {
        if (mean_.size() != std_.size()) {
            throw std::invalid_argument("Mean and std must have same size");
        }
        for (auto s : std_) {
            if (s == 0.0f) {
                throw std::invalid_argument("Standard deviation cannot be zero");
            }
        }
    }

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        const auto& shape = input.shape();
        if (shape.empty()) {
            throw std::invalid_argument("Cannot normalize scalar tensor");
        }

        size_t num_channels = mean_.size();
        if (static_cast<size_t>(shape.back()) != num_channels && num_channels > 1) {
            throw std::invalid_argument("Input channels must match normalization parameters");
        }

        // Build mean and std tensors with shape [1, 1, ..., C] for broadcasting
        std::vector<int64_t> param_shape(shape.size(), 1);
        param_shape.back() = static_cast<int64_t>(num_channels);

        // Create tensors from vectors
        auto mean_tensor = full(param_shape, 0.0f, input.dtype(), input.device());
        auto std_tensor = full(param_shape, 0.0f, input.dtype(), input.device());

        // Fill mean/std values
        for (size_t c = 0; c < num_channels; ++c) {
            auto mean_val = full({1}, mean_[c], input.dtype(), input.device());
            auto std_val = full({1}, std_[c], input.dtype(), input.device());
            // Use slice assignment via scatter-like approach
            // Simple approach: build flat then reshape
        }

        // Simpler approach: create 1D tensors and reshape for broadcasting
        auto mean_flat = zeros({static_cast<int64_t>(num_channels)}, input.dtype(), input.device());
        auto std_flat = zeros({static_cast<int64_t>(num_channels)}, input.dtype(), input.device());
        for (size_t c = 0; c < num_channels; ++c) {
            auto idx = full({1}, static_cast<float>(c), DType::Int64, Device::cpu());
            // Direct approach: use full tensors
        }

        // Most practical approach: compute (input - mean) / std using scalar ops per channel
        // For single-channel (num_channels == 1): straightforward scalar math
        if (num_channels == 1) {
            auto scale = full(std::vector<int64_t>(shape.begin(), shape.end()),
                             1.0f / std_[0], input.dtype(), input.device());
            auto offset = full(std::vector<int64_t>(shape.begin(), shape.end()),
                              mean_[0], input.dtype(), input.device());
            auto normalized = mul(sub(input, offset), scale);
            return {normalized, target};
        }

        // Multi-channel: build broadcast-compatible mean/std tensors
        // Create a [C] tensor, then reshape to [1,...,1,C]
        std::vector<float> mean_data = mean_;
        std::vector<float> std_data = std_;

        // Allocate and fill mean/std tensors on CPU, then move to device
        auto mean_1d = zeros({static_cast<int64_t>(num_channels)}, DType::Float32, Device::cpu());
        auto std_1d = zeros({static_cast<int64_t>(num_channels)}, DType::Float32, Device::cpu());
        auto mean_ptr = mean_1d.data<float>();
        auto std_ptr = std_1d.data<float>();
        for (size_t c = 0; c < num_channels; ++c) {
            mean_ptr[c] = mean_data[c];
            std_ptr[c] = std_data[c];
        }

        // Move to input device and dtype, reshape for broadcasting
        mean_1d = mean_1d.to(input.dtype()).to(input.device());
        std_1d = std_1d.to(input.dtype()).to(input.device());
        auto mean_bc = reshape(mean_1d, param_shape);
        auto std_bc = reshape(std_1d, param_shape);

        auto normalized = div(sub(input, mean_bc), std_bc);
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
        return {input, target};
    }
};

/**
 * @brief Compose multiple transforms into a pipeline
 */
class Compose : public Transform {
public:
    explicit Compose(std::vector<std::shared_ptr<Transform>> transforms)
        : transforms_(std::move(transforms)) {}

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        auto result = std::make_pair(input, target);
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
 *
 * Flips the input tensor along the last spatial dimension (width).
 * Assumes HWC or CHW format where width is the last dimension.
 */
class RandomHorizontalFlip : public Transform {
public:
    explicit RandomHorizontalFlip(float p = 0.5f) : p_(p) {
        if (p < 0.0f || p > 1.0f) {
            throw std::invalid_argument("Probability must be in [0, 1]");
        }
    }

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        if (dist(gen) < p_) {
            // Flip along last dimension (width)
            int64_t flip_dim = static_cast<int64_t>(input.ndim()) - 1;
            auto flipped = flip(input, {flip_dim});
            return {flipped, target};
        }
        return {input, target};
    }

private:
    float p_;
};

/**
 * @brief Random vertical flip transform
 *
 * Flips the input tensor along the second-to-last spatial dimension (height).
 */
class RandomVerticalFlip : public Transform {
public:
    explicit RandomVerticalFlip(float p = 0.5f) : p_(p) {
        if (p < 0.0f || p > 1.0f) {
            throw std::invalid_argument("Probability must be in [0, 1]");
        }
    }

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        if (input.ndim() < 2) {
            throw std::invalid_argument("RandomVerticalFlip requires at least 2D input");
        }

        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        if (dist(gen) < p_) {
            int64_t flip_dim = static_cast<int64_t>(input.ndim()) - 2;
            auto flipped = flip(input, {flip_dim});
            return {flipped, target};
        }
        return {input, target};
    }

private:
    float p_;
};

/**
 * @brief Center crop transform
 *
 * Crops the center of a tensor to the given size.
 * Assumes spatial dimensions are the last two dimensions.
 */
class CenterCrop : public Transform {
public:
    CenterCrop(int64_t height, int64_t width) : height_(height), width_(width) {}

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        if (input.ndim() < 2) {
            throw std::invalid_argument("CenterCrop requires at least 2D input");
        }

        auto ndim = input.ndim();
        int64_t h = input.shape()[ndim - 2];
        int64_t w = input.shape()[ndim - 1];

        if (height_ > h || width_ > w) {
            throw std::invalid_argument("Crop size must not exceed input size");
        }

        int64_t y_start = (h - height_) / 2;
        int64_t x_start = (w - width_) / 2;

        // Slice height then width
        auto cropped = slice(input, static_cast<int64_t>(ndim - 2), y_start, y_start + height_);
        cropped = slice(cropped, static_cast<int64_t>(ndim - 1), x_start, x_start + width_);

        return {cropped, target};
    }

private:
    int64_t height_;
    int64_t width_;
};

/**
 * @brief Random crop transform
 *
 * Randomly crops a tensor to the given size.
 * Assumes spatial dimensions are the last two dimensions.
 */
class RandomCrop : public Transform {
public:
    RandomCrop(int64_t height, int64_t width) : height_(height), width_(width) {}

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override {
        if (input.ndim() < 2) {
            throw std::invalid_argument("RandomCrop requires at least 2D input");
        }

        auto ndim = input.ndim();
        int64_t h = input.shape()[ndim - 2];
        int64_t w = input.shape()[ndim - 1];

        if (height_ > h || width_ > w) {
            throw std::invalid_argument("Crop size must not exceed input size");
        }

        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int64_t> y_dist(0, h - height_);
        std::uniform_int_distribution<int64_t> x_dist(0, w - width_);

        int64_t y_start = y_dist(gen);
        int64_t x_start = x_dist(gen);

        auto cropped = slice(input, static_cast<int64_t>(ndim - 2), y_start, y_start + height_);
        cropped = slice(cropped, static_cast<int64_t>(ndim - 1), x_start, x_start + width_);

        return {cropped, target};
    }

private:
    int64_t height_;
    int64_t width_;
};

/**
 * @brief Lambda transform for custom operations
 */
class Lambda : public Transform {
public:
    using LambdaFunc = std::function<std::pair<Tensor, Tensor>(const Tensor&, const Tensor&)>;

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
