/**
 * @file roi_ops.cpp
 * @brief ROI operations implementation (CPU reference)
 */

#include "tenzor/nn/detection/roi_ops.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>
#include <algorithm>

namespace tenzor {
namespace nn {
namespace detection {

// Helper function: bilinear interpolation at (y, x)
static inline float bilinear_interpolate(const float* data, int64_t height,
                                          int64_t width, float y, float x) {
    // Handle out of bounds
    if (y < -1.0f || y > height || x < -1.0f || x > width) {
        return 0.0f;
    }

    // Clamp to valid range
    y = std::max(0.0f, std::min(y, static_cast<float>(height - 1)));
    x = std::max(0.0f, std::min(x, static_cast<float>(width - 1)));

    // Integer coordinates
    int64_t y_low = static_cast<int64_t>(std::floor(y));
    int64_t x_low = static_cast<int64_t>(std::floor(x));
    int64_t y_high = std::min(y_low + 1, height - 1);
    int64_t x_high = std::min(x_low + 1, width - 1);

    // Interpolation weights
    float ly = y - static_cast<float>(y_low);
    float lx = x - static_cast<float>(x_low);
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    // Get values at four corners
    float v1 = data[y_low * width + x_low];
    float v2 = data[y_low * width + x_high];
    float v3 = data[y_high * width + x_low];
    float v4 = data[y_high * width + x_high];

    // Bilinear interpolation
    float w1 = hy * hx;
    float w2 = hy * lx;
    float w3 = ly * hx;
    float w4 = ly * lx;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

// CPU forward implementation
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

    const int64_t num_rois = rois.shape()[0];
    const int64_t channels = features.shape()[1];
    const int64_t feat_height = features.shape()[2];
    const int64_t feat_width = features.shape()[3];

    // Remember original device and dtype
    Device original_device = features.device();
    DType original_dtype = features.dtype();

    // Process on CPU in Float32 for numerical stability
    auto features_cpu = features.to(Device::cpu()).to(DType::Float32);
    auto rois_cpu = rois.to(Device::cpu()).to(DType::Float32);

    // Create output tensor on CPU for processing (in Float32)
    auto output = tenzor::zeros({num_rois, channels, output_h, output_w},
                                 DType::Float32, Device::cpu());

    const float* features_data = features_cpu.data<float>();
    const float* rois_data = rois_cpu.data<float>();
    float* output_data = output.data<float>();

    const float spatial_scale_f = static_cast<float>(spatial_scale);

    // Process each ROI
    for (int64_t roi_idx = 0; roi_idx < num_rois; ++roi_idx) {
        const float* roi = rois_data + roi_idx * 5;
        const int64_t batch_idx = static_cast<int64_t>(roi[0]);

        // Scale ROI coordinates
        float roi_x1 = roi[1] * spatial_scale_f;
        float roi_y1 = roi[2] * spatial_scale_f;
        float roi_x2 = roi[3] * spatial_scale_f;
        float roi_y2 = roi[4] * spatial_scale_f;

        // Handle aligned coordinates
        if (aligned) {
            // Aligned mode: shift by 0.5 pixel
            roi_x1 -= 0.5f;
            roi_y1 -= 0.5f;
            roi_x2 -= 0.5f;
            roi_y2 -= 0.5f;
        }

        // ROI dimensions
        float roi_width = roi_x2 - roi_x1;
        float roi_height = roi_y2 - roi_y1;

        // Bin dimensions
        float bin_size_h = roi_height / static_cast<float>(output_h);
        float bin_size_w = roi_width / static_cast<float>(output_w);

        // Determine sampling grid
        int64_t roi_bin_grid_h = (sampling_ratio > 0)
                                      ? sampling_ratio
                                      : static_cast<int64_t>(std::ceil(bin_size_h));
        int64_t roi_bin_grid_w = (sampling_ratio > 0)
                                      ? sampling_ratio
                                      : static_cast<int64_t>(std::ceil(bin_size_w));

        const int64_t count = roi_bin_grid_h * roi_bin_grid_w;

        // Get feature map for this batch
        const float* batch_features = features_data +
                                       batch_idx * channels * feat_height * feat_width;

        // Process each output bin
        for (int64_t ph = 0; ph < output_h; ++ph) {
            for (int64_t pw = 0; pw < output_w; ++pw) {
                // Process each channel
                for (int64_t c = 0; c < channels; ++c) {
                    float sum = 0.0f;

                    // Sample points in this bin
                    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
                        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                            // Compute sample position
                            float y = roi_y1 + static_cast<float>(ph) * bin_size_h +
                                      (static_cast<float>(iy) + 0.5f) * bin_size_h /
                                          static_cast<float>(roi_bin_grid_h);
                            float x = roi_x1 + static_cast<float>(pw) * bin_size_w +
                                      (static_cast<float>(ix) + 0.5f) * bin_size_w /
                                          static_cast<float>(roi_bin_grid_w);

                            // Bilinear interpolation
                            const float* channel_data =
                                batch_features + c * feat_height * feat_width;
                            sum += bilinear_interpolate(channel_data, feat_height,
                                                         feat_width, y, x);
                        }
                    }

                    // Average over samples
                    int64_t output_idx = roi_idx * channels * output_h * output_w +
                                         c * output_h * output_w + ph * output_w + pw;
                    output_data[output_idx] = sum / static_cast<float>(count);
                }
            }
        }
    }

    // Move output back to original dtype and device
    return output.to(original_dtype).to(original_device);
}

// CPU backward implementation
auto ROIAlignOp::apply_backward(const Tensor& grad_output, const Tensor& features,
                                const Tensor& rois, double spatial_scale,
                                int64_t sampling_ratio, bool aligned) -> Tensor {
    const int64_t num_rois = rois.shape()[0];
    const int64_t channels = features.shape()[1];
    const int64_t feat_height = features.shape()[2];
    const int64_t feat_width = features.shape()[3];
    const int64_t output_h = grad_output.shape()[2];
    const int64_t output_w = grad_output.shape()[3];

    // Remember original device and dtype
    Device original_device = features.device();
    DType original_dtype = features.dtype();

    // Move to CPU and convert to Float32 for processing
    auto grad_output_cpu = grad_output.to(Device::cpu()).to(DType::Float32);
    auto rois_cpu = rois.to(Device::cpu()).to(DType::Float32);

    // Create gradient tensor on CPU in Float32 (same shape as features)
    auto grad_features = tenzor::zeros({features.shape()[0], channels, feat_height, feat_width},
                                       DType::Float32, Device::cpu());

    const float* grad_output_data = grad_output_cpu.data<float>();
    const float* rois_data = rois_cpu.data<float>();
    float* grad_features_data = grad_features.data<float>();

    const float spatial_scale_f = static_cast<float>(spatial_scale);

    // Process each ROI
    for (int64_t roi_idx = 0; roi_idx < num_rois; ++roi_idx) {
        const float* roi = rois_data + roi_idx * 5;
        const int64_t batch_idx = static_cast<int64_t>(roi[0]);

        // Scale ROI coordinates
        float roi_x1 = roi[1] * spatial_scale_f;
        float roi_y1 = roi[2] * spatial_scale_f;
        float roi_x2 = roi[3] * spatial_scale_f;
        float roi_y2 = roi[4] * spatial_scale_f;

        if (aligned) {
            roi_x1 -= 0.5f;
            roi_y1 -= 0.5f;
            roi_x2 -= 0.5f;
            roi_y2 -= 0.5f;
        }

        float roi_width = roi_x2 - roi_x1;
        float roi_height = roi_y2 - roi_y1;

        float bin_size_h = roi_height / static_cast<float>(output_h);
        float bin_size_w = roi_width / static_cast<float>(output_w);

        int64_t roi_bin_grid_h = (sampling_ratio > 0)
                                      ? sampling_ratio
                                      : static_cast<int64_t>(std::ceil(bin_size_h));
        int64_t roi_bin_grid_w = (sampling_ratio > 0)
                                      ? sampling_ratio
                                      : static_cast<int64_t>(std::ceil(bin_size_w));

        const int64_t count = roi_bin_grid_h * roi_bin_grid_w;
        const float grad_scale = 1.0f / static_cast<float>(count);

        // Get gradient for this batch
        float* batch_grad = grad_features_data +
                            batch_idx * channels * feat_height * feat_width;

        // Distribute gradient from each output bin
        for (int64_t ph = 0; ph < output_h; ++ph) {
            for (int64_t pw = 0; pw < output_w; ++pw) {
                for (int64_t c = 0; c < channels; ++c) {
                    int64_t grad_idx = roi_idx * channels * output_h * output_w +
                                       c * output_h * output_w + ph * output_w + pw;
                    float grad_val = grad_output_data[grad_idx] * grad_scale;

                    // Distribute to sampled points
                    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
                        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                            float y = roi_y1 + static_cast<float>(ph) * bin_size_h +
                                      (static_cast<float>(iy) + 0.5f) * bin_size_h /
                                          static_cast<float>(roi_bin_grid_h);
                            float x = roi_x1 + static_cast<float>(pw) * bin_size_w +
                                      (static_cast<float>(ix) + 0.5f) * bin_size_w /
                                          static_cast<float>(roi_bin_grid_w);

                            // Distribute gradient via bilinear weights
                            if (y < -1.0f || y > feat_height || x < -1.0f ||
                                x > feat_width) {
                                continue;
                            }

                            y = std::max(0.0f, std::min(y, static_cast<float>(feat_height - 1)));
                            x = std::max(0.0f, std::min(x, static_cast<float>(feat_width - 1)));

                            int64_t y_low = static_cast<int64_t>(std::floor(y));
                            int64_t x_low = static_cast<int64_t>(std::floor(x));
                            int64_t y_high = std::min(y_low + 1, feat_height - 1);
                            int64_t x_high = std::min(x_low + 1, feat_width - 1);

                            float ly = y - static_cast<float>(y_low);
                            float lx = x - static_cast<float>(x_low);
                            float hy = 1.0f - ly;
                            float hx = 1.0f - lx;

                            float* channel_grad = batch_grad + c * feat_height * feat_width;

                            // Atomic-like accumulation (single-threaded here)
                            channel_grad[y_low * feat_width + x_low] += grad_val * hy * hx;
                            channel_grad[y_low * feat_width + x_high] += grad_val * hy * lx;
                            channel_grad[y_high * feat_width + x_low] += grad_val * ly * hx;
                            channel_grad[y_high * feat_width + x_high] += grad_val * ly * lx;
                        }
                    }
                }
            }
        }
    }

    // Move gradient back to original dtype and device
    return grad_features.to(original_dtype).to(original_device);
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
