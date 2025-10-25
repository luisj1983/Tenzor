/**
 * @file anchors.cpp
 * @brief Anchor box generation implementation
 */

#include "tenzor/nn/detection/anchors.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>

namespace tenzor {
namespace nn {
namespace detection {

AnchorGenerator::AnchorGenerator(std::vector<float> sizes,
                                 std::vector<float> aspect_ratios)
    : sizes_(std::move(sizes)), aspect_ratios_(std::move(aspect_ratios)) {
    if (sizes_.empty()) {
        throw std::invalid_argument("Anchor sizes cannot be empty");
    }
    if (aspect_ratios_.empty()) {
        throw std::invalid_argument("Aspect ratios cannot be empty");
    }
}

auto AnchorGenerator::generate(int64_t feat_height, int64_t feat_width,
                                int64_t stride, Device device) const -> Tensor {
    const int64_t num_sizes = static_cast<int64_t>(sizes_.size());
    const int64_t num_ratios = static_cast<int64_t>(aspect_ratios_.size());
    const int64_t num_anchors_per_loc = num_sizes * num_ratios;
    const int64_t total_anchors = feat_height * feat_width * num_anchors_per_loc;

    // Pre-allocate anchor buffer
    std::vector<float> anchor_data(total_anchors * 4);

    // Generate anchors for each spatial location
    int64_t anchor_idx = 0;
    for (int64_t y = 0; y < feat_height; ++y) {
        for (int64_t x = 0; x < feat_width; ++x) {
            // Center of anchor in image coordinates
            float cx = (static_cast<float>(x) + 0.5f) * static_cast<float>(stride);
            float cy = (static_cast<float>(y) + 0.5f) * static_cast<float>(stride);

            // Generate anchors at this location
            for (int64_t size_idx = 0; size_idx < num_sizes; ++size_idx) {
                float size = sizes_[size_idx];

                for (int64_t ratio_idx = 0; ratio_idx < num_ratios; ++ratio_idx) {
                    float ratio = aspect_ratios_[ratio_idx];

                    // Compute width and height
                    // w = size * sqrt(ratio), h = size / sqrt(ratio)
                    float sqrt_ratio = std::sqrt(ratio);
                    float w = size * sqrt_ratio;
                    float h = size / sqrt_ratio;

                    // Compute box coordinates (x1, y1, x2, y2)
                    int64_t base_idx = anchor_idx * 4;
                    anchor_data[base_idx + 0] = cx - w * 0.5f;  // x1
                    anchor_data[base_idx + 1] = cy - h * 0.5f;  // y1
                    anchor_data[base_idx + 2] = cx + w * 0.5f;  // x2
                    anchor_data[base_idx + 3] = cy + h * 0.5f;  // y2

                    ++anchor_idx;
                }
            }
        }
    }

    // Create tensor and copy data to avoid dangling pointer
    // (anchor_data vector would be destroyed, leaving tensor pointing to freed memory)
    auto anchors = zeros({total_anchors, 4}, DType::Float32, Device::cpu());
    float* anchors_ptr = anchors.data<float>();
    std::copy(anchor_data.begin(), anchor_data.end(), anchors_ptr);

    // Move to target device if needed
    if (device != Device::cpu()) {
        return anchors.to(device);
    }

    return anchors;
}

} // namespace detection
} // namespace nn
} // namespace tenzor
