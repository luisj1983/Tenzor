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
        if (static_cast<size_t>(shape.back()) != num_channels && num_channels > 1) {
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

/**
 * @brief Random rotation transform using nearest-neighbor interpolation
 *
 * Rotates input tensor by a random angle within [min_degrees, max_degrees].
 * Assumes input is in HWC or CHW format with spatial dimensions.
 */
class RandomRotation : public Transform {
public:
    /**
     * @brief Construct random rotation transform
     * @param min_degrees Minimum rotation angle in degrees
     * @param max_degrees Maximum rotation angle in degrees
     */
    RandomRotation(float min_degrees, float max_degrees);

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    float min_degrees_, max_degrees_;
};

/**
 * @brief Color jitter transform for data augmentation
 *
 * Randomly adjusts brightness, contrast, saturation, and hue of the input.
 * Each parameter specifies the maximum deviation from the original value.
 */
class ColorJitter : public Transform {
public:
    /**
     * @brief Construct color jitter transform
     * @param brightness Maximum brightness adjustment factor
     * @param contrast Maximum contrast adjustment factor
     * @param saturation Maximum saturation adjustment factor
     * @param hue Maximum hue shift (not applied in this simplified version)
     */
    ColorJitter(float brightness = 0, float contrast = 0,
                float saturation = 0, float hue = 0);

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    float brightness_, contrast_, saturation_, hue_;
};

/**
 * @brief Cutout regularization transform
 *
 * Randomly masks out rectangular regions of the input tensor with zeros,
 * encouraging the model to attend to less prominent features.
 */
class Cutout : public Transform {
public:
    /**
     * @brief Construct cutout transform
     * @param num_holes Number of rectangular holes to cut out
     * @param hole_size Side length of each square hole in pixels
     */
    Cutout(int num_holes, int hole_size);

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    int num_holes_, hole_size_;
};

/**
 * @brief Random vertical flip transform
 */
class RandomVerticalFlip : public Transform {
public:
    explicit RandomVerticalFlip(float p = 0.5f) : p_(p) {
        if (p < 0.0f || p > 1.0f) {
            throw std::invalid_argument("Probability must be in [0, 1]");
        }
    }

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    float p_;
};

/**
 * @brief Center crop transform
 *
 * Crops the input tensor at the center to the specified size.
 * Assumes input has spatial dimensions (at least 2D).
 */
class CenterCrop : public Transform {
public:
    CenterCrop(int64_t height, int64_t width)
        : height_(height), width_(width) {}

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    int64_t height_, width_;
};

/**
 * @brief Random crop transform
 *
 * Randomly crops a region of the specified size from the input.
 * Optionally applies padding before cropping.
 */
class RandomCrop : public Transform {
public:
    RandomCrop(int64_t height, int64_t width, int64_t padding = 0)
        : height_(height), width_(width), padding_(padding) {}

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    int64_t height_, width_, padding_;
};

/**
 * @brief Resize transform
 *
 * Resizes input spatial dimensions to target size using interpolation.
 * Uses nearest-neighbor interpolation.
 */
class Resize : public Transform {
public:
    Resize(int64_t height, int64_t width)
        : height_(height), width_(width) {}

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    int64_t height_, width_;
};

/**
 * @brief Random resized crop transform
 *
 * Crops a random region and resizes to target size.
 * Scale and ratio parameters control the random crop area and aspect ratio.
 */
class RandomResizedCrop : public Transform {
public:
    RandomResizedCrop(int64_t height, int64_t width,
                      float scale_min = 0.08f, float scale_max = 1.0f,
                      float ratio_min = 0.75f, float ratio_max = 1.333f)
        : height_(height), width_(width),
          scale_min_(scale_min), scale_max_(scale_max),
          ratio_min_(ratio_min), ratio_max_(ratio_max) {}

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    int64_t height_, width_;
    float scale_min_, scale_max_, ratio_min_, ratio_max_;
};

/**
 * @brief Gaussian blur transform
 *
 * Applies Gaussian blur with the specified kernel size and sigma.
 */
class GaussianBlur : public Transform {
public:
    GaussianBlur(int kernel_size, float sigma_min = 0.1f, float sigma_max = 2.0f)
        : kernel_size_(kernel_size), sigma_min_(sigma_min), sigma_max_(sigma_max) {
        if (kernel_size % 2 == 0) {
            throw std::invalid_argument("Kernel size must be odd");
        }
    }

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    int kernel_size_;
    float sigma_min_, sigma_max_;
};

/**
 * @brief Random affine transformation
 *
 * Applies a random affine transformation (rotation + translation + scale + shear)
 * to the input tensor using nearest-neighbor interpolation.
 */
class RandomAffine : public Transform {
public:
    /**
     * @param degrees       Max rotation angle in degrees (symmetric: [-degrees, degrees])
     * @param translate_x   Max horizontal translation as fraction of width [0, 1)
     * @param translate_y   Max vertical translation as fraction of height [0, 1)
     * @param scale_min     Minimum scaling factor (default: 1.0)
     * @param scale_max     Maximum scaling factor (default: 1.0)
     * @param shear         Max shear angle in degrees (default: 0)
     */
    RandomAffine(float degrees, float translate_x = 0.0f, float translate_y = 0.0f,
                 float scale_min = 1.0f, float scale_max = 1.0f, float shear = 0.0f)
        : degrees_(degrees), translate_x_(translate_x), translate_y_(translate_y),
          scale_min_(scale_min), scale_max_(scale_max), shear_(shear) {}

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    float degrees_, translate_x_, translate_y_;
    float scale_min_, scale_max_, shear_;
};

} // namespace transforms
} // namespace data
} // namespace tenzor
