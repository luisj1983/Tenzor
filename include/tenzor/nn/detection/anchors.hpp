/**
 * @file anchors.hpp
 * @brief Anchor box generation for object detection
 *
 * Implements anchor box generation for Faster R-CNN, Mask R-CNN, and similar
 * detection architectures. Supports multiple scales and aspect ratios.
 */

#pragma once

#include <vector>
#include <cstdint>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"

namespace tenzor {
namespace nn {
namespace detection {

/**
 * @brief Generate anchor boxes at multiple scales and aspect ratios.
 *
 * Creates anchor boxes for each position in a feature map. Anchors are
 * defined by their size (scale) and aspect ratio (width/height).
 *
 * Example:
 * @code
 * AnchorGenerator anchors({32.0f, 64.0f, 128.0f}, {0.5f, 1.0f, 2.0f});
 * auto boxes = anchors.generate(38, 38, 16);  // 38x38 feature map, stride 16
 * // Returns: (38*38*9, 4) tensor with (x1, y1, x2, y2) coordinates
 * @endcode
 */
class AnchorGenerator {
public:
    /**
     * @brief Construct anchor generator.
     *
     * @param sizes Anchor base sizes in pixels (e.g., {32, 64, 128, 256, 512})
     * @param aspect_ratios Aspect ratios as width/height (e.g., {0.5, 1.0, 2.0})
     */
    AnchorGenerator(std::vector<float> sizes, std::vector<float> aspect_ratios);

    /**
     * @brief Generate anchors for a feature map.
     *
     * Creates anchors at each spatial position of the feature map. The number
     * of anchors per location equals sizes.size() * aspect_ratios.size().
     *
     * @param feat_height Feature map height
     * @param feat_width Feature map width
     * @param stride Feature map stride relative to input image (e.g., 16 for 1/16 resolution)
     * @param device Device to create anchors on (default: CPU)
     * @return Tensor of shape (H*W*K, 4) where K=num_anchors_per_location
     *         Format: (x1, y1, x2, y2) in image coordinates
     */
    auto generate(int64_t feat_height, int64_t feat_width, int64_t stride,
                  Device device = Device::cpu()) const -> Tensor;

    /**
     * @brief Get number of anchors per spatial location.
     *
     * @return Number of anchor boxes at each position (sizes * aspect_ratios)
     */
    auto num_anchors_per_location() const -> int64_t {
        return static_cast<int64_t>(sizes_.size() * aspect_ratios_.size());
    }

    /**
     * @brief Get anchor sizes.
     */
    auto sizes() const -> const std::vector<float>& { return sizes_; }

    /**
     * @brief Get anchor aspect ratios.
     */
    auto aspect_ratios() const -> const std::vector<float>& { return aspect_ratios_; }

private:
    std::vector<float> sizes_;
    std::vector<float> aspect_ratios_;
};

} // namespace detection
} // namespace nn
} // namespace tenzor
