/**
 * @file roi_ops.cpp
 * @brief ROI operations implementation (CPU reference + GPU dispatch)
 */

#include "tenzor/nn/detection/roi_ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace tenzor {
namespace nn {
namespace detection {

// Forward implementation: routes all devices through dispatch<OpId::ROIAlignForward>,
// which lands on the registered backend kernel (CPU kernel is dtype-preserving
// and Float64-correct, matching the GPU backends).
auto ROIAlignOp::apply(const Tensor& features, const Tensor& rois,
                       int64_t output_h, int64_t output_w,
                       double spatial_scale, int64_t sampling_ratio,
                       bool aligned) -> Tensor {
    if (features.ndim() != 4) {
        throw std::invalid_argument("Features must be 4D tensor (N, C, H, W)");
    }
    if (rois.ndim() != 2 || rois.shape()[1] != 5) {
        throw std::invalid_argument("ROIs must be (num_rois, 5) tensor");
    }

    // Route every device (CPU included) through the registered backend kernel
    // via dispatch. The CPU kernel (cpu::roi_align_forward_kernel) keeps Float32
    // inputs in Float32 and Float64 inputs in Float64, matching the
    // double-precision GPU backends. The previous inline CPU loop forced all
    // sampling/interpolation through Float32, diverging from the GPU path and
    // from its own registered kernel — a second, lower-precision implementation
    // of the same op. There is now a single source of truth.
    OpAttributes attrs;
    attrs.set(AttrKey::OutputSizeH, output_h);
    attrs.set(AttrKey::OutputSizeW, output_w);
    attrs.set(AttrKey::SpatialScale, spatial_scale);
    attrs.set(AttrKey::SamplingRatio, sampling_ratio);
    attrs.set(AttrKey::Aligned, aligned);

    std::array<Tensor, 2> inputs = {features, rois};
    auto results = dispatch<OpId::ROIAlignForward>(inputs, attrs);
    return results[0];
}

// Backward implementation: routes all devices through dispatch<OpId::ROIAlignBackward>,
// which lands on the registered backend kernel (CPU kernel is dtype-preserving
// and Float64-correct, matching the GPU backends).
auto ROIAlignOp::apply_backward(const Tensor& grad_output, const Tensor& features,
                                const Tensor& rois, double spatial_scale,
                                int64_t sampling_ratio, bool aligned) -> Tensor {
    // Route every device (CPU included) through the registered backend kernel
    // via dispatch. cpu::roi_align_backward_kernel preserves the input dtype
    // (Float64 stays Float64), matching the double-precision GPU backends. The
    // previous inline CPU loop computed the scatter in Float32 only, diverging
    // from both the GPU path and its own registered kernel. Single source of
    // truth now.
    OpAttributes attrs;
    attrs.set(AttrKey::BatchSize, features.shape()[0]);
    attrs.set(AttrKey::FeatHeight, features.shape()[2]);
    attrs.set(AttrKey::FeatWidth, features.shape()[3]);
    attrs.set(AttrKey::SpatialScale, spatial_scale);
    attrs.set(AttrKey::SamplingRatio, sampling_ratio);
    attrs.set(AttrKey::Aligned, aligned);

    std::array<Tensor, 2> inputs = {grad_output, rois};
    auto results = dispatch<OpId::ROIAlignBackward>(inputs, attrs);
    return results[0];
}

// ROIAlign module implementation
ROIAlign::ROIAlign(int64_t output_h, int64_t output_w, double spatial_scale,
                   int64_t sampling_ratio, bool aligned)
    : output_h_(output_h),
      output_w_(output_w),
      spatial_scale_(spatial_scale),
      sampling_ratio_(sampling_ratio),
      aligned_(aligned) {}

auto ROIAlign::forward(const Variable& features, const Tensor& rois) -> Variable {
    // Forward pass through ROIAlignOp utility
    auto output_tensor = ROIAlignOp::apply(
        features.tensor(), rois, output_h_, output_w_, spatial_scale_,
        sampling_ratio_, aligned_);

    // Create Variable with gradient function
    Variable output(output_tensor, features.requires_grad());

    if (features.requires_grad() && is_grad_enabled()) {
        // Create custom backward function
        struct ROIAlignBackward : public Function {
            Tensor features_;
            Tensor rois_;
            double spatial_scale_;
            int64_t sampling_ratio_;
            bool aligned_;

            ROIAlignBackward(const Tensor& features, const Tensor& rois,
                             double spatial_scale, int64_t sampling_ratio, bool aligned)
                : features_(features),
                  rois_(rois),
                  spatial_scale_(spatial_scale),
                  sampling_ratio_(sampling_ratio),
                  aligned_(aligned) {}

            auto forward(std::vector<Variable> /* inputs */) -> std::vector<Variable> override {
                throw std::runtime_error("ROIAlignBackward::forward should not be called");
            }

            auto backward(std::vector<Tensor> grad_outputs)
                -> std::vector<Tensor> override {
                auto grad_features = ROIAlignOp::apply_backward(
                    grad_outputs[0], features_, rois_, spatial_scale_,
                    sampling_ratio_, aligned_);
                return {grad_features};
            }
        };

        auto grad_fn = std::make_shared<ROIAlignBackward>(
            features.tensor(), rois, spatial_scale_, sampling_ratio_, aligned_);

        // Connect to features' gradient function to maintain gradient chain
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(features.grad_fn());  // nullptr if features is leaf
        grad_fn->set_next_functions(next_funcs);

        // Track input variable for gradient accumulation (critical for leaf variables)
        std::vector<Variable> input_vars;
        if (features.requires_grad()) {
            input_vars.push_back(features);
        }
        grad_fn->set_input_variables(input_vars);

        output.set_grad_fn(grad_fn);
    }

    return output;
}

} // namespace detection
} // namespace nn
} // namespace tenzor
