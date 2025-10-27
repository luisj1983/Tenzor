/**
 * @file roi_ops.hpp
 * @brief ROI (Region of Interest) operations for detection models
 *
 * Provides ROI Align for extracting fixed-size feature maps from regions
 * of interest. Essential for Mask R-CNN and other detection architectures.
 */

#pragma once

#include <cstdint>
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/nn/module.hpp"

namespace tenzor {
namespace nn {
namespace detection {

/**
 * @brief ROI Align layer with bilinear interpolation.
 *
 * Extracts fixed-size feature maps from regions of interest using bilinear
 * interpolation. Improves upon ROI Pooling by avoiding quantization, which
 * leads to better mask quality in Mask R-CNN.
 *
 * Algorithm:
 * 1. For each ROI, divide it into output_h x output_w bins
 * 2. In each bin, sample points at regular grid (determined by sampling_ratio)
 * 3. Bilinearly interpolate feature map at each sample point
 * 4. Average all samples in the bin
 *
 * @code
 * ROIAlign roi_align(7, 7, 1.0/16.0, 2);  // 7x7 output, 1/16 scale, 2 samples
 *
 * auto features = randn({2, 256, 50, 50});  // Batch of feature maps
 * auto rois = randn({100, 5});              // 100 ROIs with batch indices
 * // ROI format: (batch_idx, x1, y1, x2, y2)
 *
 * auto aligned = roi_align.forward(features, rois);  // Shape: (100, 256, 7, 7)
 * @endcode
 */
class ROIAlign : public Module {
public:
    /**
     * @brief Construct ROI Align layer.
     *
     * @param output_h Output height for each ROI
     * @param output_w Output width for each ROI
     * @param spatial_scale Scale factor from input image to feature map
     *                       (e.g., 1/16 if feature map is 16x downsampled)
     * @param sampling_ratio Number of sampling points per bin dimension
     *                       (0 = adaptive based on bin size)
     * @param aligned Use aligned coordinates: (x2-x1)/(output_w-1) vs (x2-x1)/output_w
     *                (default: true, matches PyTorch aligned=True)
     */
    ROIAlign(int64_t output_h, int64_t output_w, double spatial_scale,
             int64_t sampling_ratio = 2, bool aligned = true);

    /**
     * @brief Forward pass with features and ROIs.
     *
     * @param features Input feature maps (N, C, H, W)
     * @param rois Regions of interest (num_rois, 5)
     *             Format: (batch_index, x1, y1, x2, y2) where coordinates
     *             are in the original image space (before spatial_scale)
     * @return Aligned features (num_rois, C, output_h, output_w)
     */
    auto forward(const Variable& features, const Tensor& rois) -> Variable;

    /**
     * @brief Module forward (not used, ROIAlign requires both features and ROIs).
     */
    auto forward(const Variable& /* input */) -> Variable override {
        throw std::runtime_error("ROIAlign requires both features and rois. "
                                 "Use forward(features, rois) instead.");
    }

private:
    int64_t output_h_;
    int64_t output_w_;
    double spatial_scale_;
    int64_t sampling_ratio_;
    bool aligned_;
};

/**
 * @brief Utility class for ROI Align operations.
 *
 * Provides static methods for ROI Align computation with bilinear interpolation.
 * This is a pure utility class (not an autograd::Function). The ROIAlign module
 * class handles autograd integration through its own backward function.
 *
 * Design Note: Named "Op" instead of "Function" to avoid confusion with
 * autograd::Function base class and prevent any virtual function hiding concerns.
 */
class ROIAlignOp {
public:
    /**
     * @brief Compute ROI Align forward pass.
     *
     * Extracts fixed-size feature maps from ROIs using bilinear interpolation.
     *
     * @param features Input feature maps (N, C, H, W)
     * @param rois ROI coordinates (num_rois, 5) - format: (batch_idx, x1, y1, x2, y2)
     * @param output_h Output height for each ROI
     * @param output_w Output width for each ROI
     * @param spatial_scale Scale factor from input image to feature map
     * @param sampling_ratio Number of samples per output bin (0=adaptive)
     * @param aligned Use aligned coordinates (matches PyTorch aligned=True)
     * @return Aligned features (num_rois, C, output_h, output_w)
     */
    static auto apply(const Tensor& features, const Tensor& rois,
                      int64_t output_h, int64_t output_w,
                      double spatial_scale, int64_t sampling_ratio,
                      bool aligned) -> Tensor;

    /**
     * @brief Compute ROI Align gradient with respect to input features.
     *
     * Distributes output gradients back to input feature map using bilinear
     * interpolation weights.
     *
     * @param grad_output Gradient w.r.t output (num_rois, C, output_h, output_w)
     * @param features Original input features (for shape information)
     * @param rois Original ROI coordinates
     * @param spatial_scale Scale factor from input image to feature map
     * @param sampling_ratio Number of samples per output bin
     * @param aligned Use aligned coordinates
     * @return Gradient w.r.t features (N, C, H, W)
     */
    static auto apply_backward(const Tensor& grad_output, const Tensor& features,
                               const Tensor& rois, double spatial_scale,
                               int64_t sampling_ratio, bool aligned) -> Tensor;
};

} // namespace detection
} // namespace nn
} // namespace tenzor
