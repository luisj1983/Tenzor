/**
 * @file upsample.hpp
 * @brief Upsample layer for neural networks
 *
 * Wraps the interpolate operation to provide a stateful module
 * for upsampling tensors using nearest, bilinear, or trilinear interpolation.
 */

#pragma once

#include "../module.hpp"
#include <optional>
#include <string>
#include <vector>

namespace tenzor {
namespace nn {

/**
 * @brief Upsample layer that resizes input tensors.
 *
 * Upsamples (or downsamples) an input tensor to a given size or by a given
 * scale factor, using specified interpolation mode.
 *
 * Exactly one of `size` or `scale_factor` must be provided.
 *
 * @code
 * // Fixed output size
 * Upsample up1({64, 64});
 * // Input: (N, C, H, W) -> Output: (N, C, 64, 64)
 *
 * // Scale factor 2x
 * Upsample up2(std::nullopt, 2.0, "bilinear");
 * // Input: (N, C, 32, 32) -> Output: (N, C, 64, 64)
 *
 * // 3D trilinear
 * Upsample up3({16, 32, 32}, std::nullopt, "trilinear");
 * // Input: (N, C, D, H, W) -> Output: (N, C, 16, 32, 32)
 * @endcode
 *
 * **Modes:**
 * - "nearest": Nearest neighbor interpolation (default)
 * - "bilinear": Bilinear interpolation (4D only)
 * - "trilinear": Trilinear interpolation (5D only)
 */
class Upsample : public Module {
public:
    /**
     * @brief Construct Upsample layer with target size.
     *
     * @param size Target output spatial size (e.g., {H, W} for 4D)
     * @param scale_factor Scale factor for spatial dimensions (ignored if size is given)
     * @param mode Interpolation mode: "nearest", "bilinear", "trilinear" (default: "nearest")
     * @param align_corners If true, align corner pixels (default: false). Only used with bilinear/trilinear.
     */
    explicit Upsample(std::optional<std::vector<int64_t>> size = std::nullopt,
                      std::optional<double> scale_factor = std::nullopt,
                      const std::string& mode = "nearest",
                      bool align_corners = false);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::optional<std::vector<int64_t>> size_;
    std::optional<double> scale_factor_;
    std::string mode_;
    bool align_corners_;
};

} // namespace nn
} // namespace tenzor
