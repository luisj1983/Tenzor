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

        // Validate no zero std
        for (size_t i = 0; i < num_channels; ++i) {
            if (std_[i] == 0.0f) {
                throw std::invalid_argument("Standard deviation cannot be zero");
            }
        }

        // Create output tensor and apply channel-wise normalization
        Tensor normalized = zeros(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype());
        const float* src = static_cast<const float*>(input.data_ptr());
        float* dst = static_cast<float*>(normalized.data_ptr());
        int64_t total = 1;
        for (auto s : shape) total *= s;
        int64_t C = static_cast<int64_t>(num_channels);

        for (int64_t i = 0; i < total; ++i) {
            int64_t c = i % C;
            dst[i] = (src[i] - mean_[c]) / std_[c];
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
        -> std::pair<Tensor, Tensor> override;

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

/**
 * @brief Random erasing augmentation (Zhong et al., 2020)
 *
 * With probability p, erases a random rectangular region of the input
 * and fills it with a constant value. Assumes input is 3D (C,H,W).
 */
class RandomErasing : public Transform {
public:
    /**
     * @param p Probability of applying erasing
     * @param scale_min Minimum fraction of image area to erase
     * @param scale_max Maximum fraction of image area to erase
     * @param ratio_min Minimum aspect ratio of erased region
     * @param ratio_max Maximum aspect ratio of erased region
     * @param value Fill value for erased region
     */
    explicit RandomErasing(float p = 0.5f, float scale_min = 0.02f,
                           float scale_max = 0.33f, float ratio_min = 0.3f,
                           float ratio_max = 3.3f, float value = 0.0f);

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    float p_, scale_min_, scale_max_, ratio_min_, ratio_max_, value_;
};

/**
 * @brief Random perspective warp augmentation
 *
 * With probability p, applies a random perspective transformation
 * using bilinear interpolation. Assumes input is 3D (C,H,W).
 */
class RandomPerspective : public Transform {
public:
    /**
     * @param distortion_scale Controls the magnitude of corner displacements
     * @param p Probability of applying the transform
     */
    explicit RandomPerspective(float distortion_scale = 0.5f, float p = 0.5f);

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    float distortion_scale_, p_;
};

/**
 * @brief Elastic deformation augmentation (Simard et al., 2003)
 *
 * With probability p, applies random elastic distortion by generating
 * displacement fields smoothed with a Gaussian kernel. Assumes input is 3D (C,H,W).
 */
class ElasticTransform : public Transform {
public:
    /**
     * @param alpha Displacement magnitude scaling factor
     * @param sigma Gaussian smoothing sigma for displacement fields
     * @param p Probability of applying the transform
     */
    explicit ElasticTransform(float alpha = 50.0f, float sigma = 5.0f, float p = 0.5f);

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    float alpha_, sigma_, p_;
};

/**
 * @brief MixUp data augmentation (Zhang et al., 2018)
 *
 * Blends two input-target pairs using a mixing coefficient sampled from
 * Beta(alpha, alpha). This is a standalone class, not a Transform subclass,
 * because it operates on pairs of samples.
 */
class MixUp {
public:
    /**
     * @param alpha Beta distribution parameter controlling mixing strength
     */
    explicit MixUp(float alpha = 0.2f);

    /**
     * @brief Mix two input-target pairs
     * @return Mixed (input, target) pair where mixed = lambda*first + (1-lambda)*second
     */
    auto operator()(const Tensor& input1, const Tensor& target1,
                    const Tensor& input2, const Tensor& target2)
        -> std::pair<Tensor, Tensor>;

private:
    float alpha_;
};

/**
 * @brief CutMix data augmentation (Yun et al., 2019)
 *
 * Cuts a rectangular region from one image and pastes it onto another,
 * with proportional target mixing. Standalone class (not a Transform subclass).
 */
class CutMix {
public:
    /**
     * @param alpha Beta distribution parameter controlling cut size
     */
    explicit CutMix(float alpha = 1.0f);

    /**
     * @brief Apply CutMix to two input-target pairs
     * @return Mixed (input, target) pair
     */
    auto operator()(const Tensor& input1, const Tensor& target1,
                    const Tensor& input2, const Tensor& target2)
        -> std::pair<Tensor, Tensor>;

private:
    float alpha_;
};

/**
 * @brief RandAugment automatic augmentation policy (Cubuk et al., 2020)
 *
 * Randomly selects and applies num_ops transforms from a predefined pool,
 * each at a strength controlled by magnitude. Assumes input is 3D (C,H,W).
 */
class RandAugment : public Transform {
public:
    /**
     * @param num_ops Number of transforms to apply per invocation
     * @param magnitude Magnitude level (0-30) controlling transform strength
     * @param magnitude_std Standard deviation for per-application magnitude randomization
     */
    explicit RandAugment(int num_ops = 2, int magnitude = 9, float magnitude_std = 0.5f);

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    int num_ops_, magnitude_;
    float magnitude_std_;
};

/**
 * @brief TrivialAugmentWide augmentation policy (Muller & Hutter, 2021)
 *
 * Applies a single randomly selected transform at a uniformly sampled magnitude.
 * Simpler than RandAugment with no hyperparameter tuning needed.
 * Assumes input is 3D (C,H,W).
 */
class TrivialAugmentWide : public Transform {
public:
    /**
     * @param num_magnitude_bins Number of discrete magnitude levels
     */
    explicit TrivialAugmentWide(int num_magnitude_bins = 31);

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    int num_magnitude_bins_;
};

/**
 * @brief AugMix augmentation strategy (Hendrycks et al., 2020)
 *
 * Creates multiple parallel augmentation chains mixed via Dirichlet weights,
 * then blended with the original image using a Beta-distributed coefficient.
 * Assumes input is 3D (C,H,W).
 */
class AugMix : public Transform {
public:
    /**
     * @param width Number of parallel augmentation chains
     * @param depth Number of transforms per chain (-1 for random 1-3)
     * @param severity Augmentation severity (0-10)
     * @param alpha Dirichlet/Beta mixing parameter
     */
    explicit AugMix(int width = 3, int depth = -1, float severity = 3.0f,
                    float alpha = 1.0f);

    auto operator()(const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> override;

private:
    int width_, depth_;
    float severity_, alpha_;
};

} // namespace transforms
} // namespace data
} // namespace tenzor
