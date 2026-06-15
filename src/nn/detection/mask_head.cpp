/**
 * @file mask_head.cpp
 * @brief Implementation of mask prediction head for Mask R-CNN
 */

#include "tenzor/nn/detection/mask_head.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/autograd/ops.hpp"
#include <stdexcept>

namespace tenzor {
namespace nn {
namespace detection {

MaskHead::MaskHead(int64_t in_channels,
                   int64_t num_classes,
                   int64_t conv_dim,
                   int64_t num_conv,
                   int64_t mask_size)
    : num_classes_(num_classes)
    , conv_dim_(conv_dim)
    , mask_size_(mask_size) {

    // Create convolutional layers for feature extraction
    // First layer takes in_channels, rest take conv_dim
    for (int64_t i = 0; i < num_conv; ++i) {
        int64_t in_ch = (i == 0) ? in_channels : conv_dim;

        auto conv = std::make_shared<Conv2d>(
            in_ch,
            conv_dim,
            3,  // kernel_size
            1,  // stride
            1,  // padding
            1,  // dilation
            1,  // groups
            false  // no bias (using batchnorm)
        );
        conv_layers_.push_back(conv);
        register_module("conv" + std::to_string(i + 1), conv);

        auto bn = std::make_shared<BatchNorm2d>(conv_dim);
        bn_layers_.push_back(bn);
        register_module("bn" + std::to_string(i + 1), bn);
    }

    // Deconvolution for 2× upsampling
    // Input: (N, conv_dim, H, W) -> Output: (N, conv_dim, 2H, 2W)
    deconv_ = std::make_shared<ConvTranspose2d>(
        conv_dim,
        conv_dim,
        2,  // kernel_size
        2,  // stride (for 2× upsampling)
        0,  // padding
        0,  // output_padding
        1,  // groups
        false  // no bias (using batchnorm)
    );
    register_module("deconv", deconv_);

    deconv_bn_ = std::make_shared<BatchNorm2d>(conv_dim);
    register_module("deconv_bn", deconv_bn_);

    // Final 1×1 convolution for per-class mask prediction
    // Output: (N, num_classes, H, W)
    mask_pred_ = std::make_shared<Conv2d>(
        conv_dim,
        num_classes,
        1,  // kernel_size
        1,  // stride
        0,  // padding
        1,  // dilation
        1,  // groups
        true  // bias
    );
    register_module("mask_pred", mask_pred_);
}

auto MaskHead::forward_impl(const Variable& roi_features) -> Variable {
    // roi_features: (num_rois, in_channels, H, W)
    // Expected: (num_rois, 256, 14, 14) for 28×28 output masks

    auto x = roi_features;

    // Apply convolutional layers with ReLU activation
    for (size_t i = 0; i < conv_layers_.size(); ++i) {
        x = conv_layers_[i]->forward(x);
        x = bn_layers_[i]->forward(x);
        x = relu(x);
    }

    // Deconvolution for 2× upsampling
    // (num_rois, conv_dim, H, W) -> (num_rois, conv_dim, 2H, 2W)
    x = deconv_->forward(x);
    x = deconv_bn_->forward(x);
    x = relu(x);

    // Final 1×1 conv for mask prediction
    // (num_rois, conv_dim, 2H, 2W) -> (num_rois, num_classes, 2H, 2W)
    auto mask_logits = mask_pred_->forward(x);

    return mask_logits;
}

auto mask_loss(const Variable& mask_logits,
               const Tensor& mask_targets,
               const Tensor& class_labels) -> Variable {
    // mask_logits: (num_rois, num_classes, H, W)
    // mask_targets: (num_rois, H, W) in [0, 1]
    // class_labels: (num_rois,) class indices

    auto num_rois = mask_logits.tensor().shape()[0];
    auto num_classes = mask_logits.tensor().shape()[1];
    auto H = mask_logits.tensor().shape()[2];
    auto W = mask_logits.tensor().shape()[3];

    if (num_rois == 0) {
        // No ROIs, return zero loss
        auto zero_loss = Variable(
            Tensor({}, mask_logits.tensor().dtype(), mask_logits.tensor().device()).fill_(0.0),
            false
        );
        return zero_loss;
    }

    // Select, for each ROI, the mask channel of its ground-truth class. This MUST
    // stay on the autograd graph: selecting on the raw tensor and rewrapping in a
    // fresh Variable severs grad_fn, so no gradient would reach the mask head or
    // backbone. We build a (num_rois, 1, H, W) gather index and use the
    // autograd-aware gather, whose backward scatters gradients back to
    // mask_logits at the selected channels.
    // Move class labels to CPU and ensure Int64 for data access. Without the
    // dtype conversion a Float32/Int32 label tensor would be reinterpreted as
    // int64_t (garbage class indices). GPU data_ptr() access can segfault, hence
    // the device move too.
    auto class_labels_cpu = class_labels.to(Device::cpu()).to(DType::Int64);
    const auto* class_labels_data =
        static_cast<const int64_t*>(class_labels_cpu.data_ptr());

    Tensor gather_index({num_rois, 1, H, W}, DType::Int64, Device::cpu());
    auto* gather_index_data = static_cast<int64_t*>(gather_index.data_ptr());
    const int64_t plane = H * W;
    for (int64_t i = 0; i < num_rois; ++i) {
        int64_t class_idx = class_labels_data[i];

        // Validate class index
        if (class_idx < 0 || class_idx >= num_classes) {
            throw std::runtime_error(
                "mask_loss: class label " + std::to_string(class_idx) +
                " out of range [0, " + std::to_string(num_classes) + ")"
            );
        }

        for (int64_t p = 0; p < plane; ++p) {
            gather_index_data[i * plane + p] = class_idx;
        }
    }
    if (mask_logits.tensor().device().type != Device::Type::CPU) {
        gather_index = gather_index.to(mask_logits.tensor().device());
    }

    // (num_rois, num_classes, H, W) -> gather class -> (num_rois, 1, H, W)
    // -> squeeze -> (num_rois, H, W); grad_fn preserved throughout.
    auto selected_logits = tenzor::squeeze(tenzor::gather(mask_logits, 1, gather_index), 1);

    // Reshape for BCE loss computation
    // (num_rois, H, W) -> (num_rois * H * W,)
    auto logits_flat = selected_logits.reshape(std::vector<int64_t>{num_rois * H * W});
    auto targets_flat = Variable(
        mask_targets.reshape(std::vector<int64_t>{num_rois * H * W}),
        false
    );

    // Compute binary cross-entropy loss with logits
    // This applies sigmoid internally
    BCEWithLogitsLoss bce_loss;
    auto loss = bce_loss.forward(logits_flat, targets_flat);

    return loss;
}

auto process_masks(const Tensor& mask_logits,
                   const Tensor& boxes,
                   const Tensor& class_labels,
                   int64_t image_height,
                   int64_t image_width,
                   double threshold) -> Tensor {
    // mask_logits: (num_detections, num_classes, mask_h, mask_w)
    // boxes: (num_detections, 4) as (x1, y1, x2, y2)
    // class_labels: (num_detections,)

    auto num_detections = mask_logits.shape()[0];

    if (num_detections == 0) {
        // No detections, return empty 4D masks: [0, 1, H, W]
        return Tensor(
            std::vector<int64_t>{0, 1, image_height, image_width},
            mask_logits.dtype(),
            mask_logits.device()
        );
    }

    // Keep track of original device for final output
    auto original_device = mask_logits.device();

    // Create the output mask buffer on CPU and write every detection into it,
    // then upload once after the loop. Previously the [num_detections,1,H,W]
    // buffer was downloaded and re-uploaded on every iteration (O(N^2*H*W)
    // transfer on GPU).
    auto full_masks_cpu = Tensor(
        std::vector<int64_t>{num_detections, 1, image_height, image_width},
        DType::UInt8,
        Device::cpu()
    );
    full_masks_cpu.fill_(0);
    auto* full_data = full_masks_cpu.data<uint8_t>();

    // Move class_labels and boxes to CPU for data access. class_labels must be
    // Int64 before pointer reinterpretation (a Float32/Int32 tensor would yield
    // garbage class indices).
    auto class_labels_cpu = class_labels.to(Device::cpu()).to(DType::Int64);
    auto boxes_cpu = boxes.to(DType::Float32).to(Device::cpu());
    auto* class_labels_data = class_labels_cpu.data<int64_t>();
    auto* boxes_data = boxes_cpu.data<float>();

    for (int64_t i = 0; i < num_detections; ++i) {
        // Get class label and select corresponding mask
        int64_t class_idx = class_labels_data[i];

        // Extract mask logits for this class: (mask_h, mask_w)
        auto class_mask_logits = tenzor::select(tenzor::select(mask_logits, 0, i), 0, class_idx);

        // Apply sigmoid to get probabilities: 1 / (1 + exp(-x))
        auto neg_logits = class_mask_logits * (-1.0f);
        auto exp_neg = tenzor::exp(neg_logits);
        auto one = tenzor::ones_like(exp_neg);
        auto mask_probs = one / (one + exp_neg);

        // Get ROI coordinates (boxes are flattened row-major: [x1, y1, x2, y2] per row)
        float x1 = boxes_data[i * 4 + 0];
        float y1 = boxes_data[i * 4 + 1];
        float x2 = boxes_data[i * 4 + 2];
        float y2 = boxes_data[i * 4 + 3];

        // Clip to image boundaries
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(image_width - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(image_height - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(image_width - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(image_height - 1)));

        // Ensure valid box
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        int64_t roi_w = static_cast<int64_t>(x2 - x1 + 1);
        int64_t roi_h = static_cast<int64_t>(y2 - y1 + 1);
        int64_t roi_x = static_cast<int64_t>(x1);
        int64_t roi_y = static_cast<int64_t>(y1);

        // Resize mask from (mask_h, mask_w) to (roi_h, roi_w)
        // For simplicity, we use bilinear interpolation
        // In production, this would use ops::interpolate
        auto resized_mask = ops::interpolate(
            mask_probs.unsqueeze(0).unsqueeze(0),  // (1, 1, mask_h, mask_w)
            std::vector<int64_t>{roi_h, roi_w},
            "bilinear",
            false  // align_corners
        );
        resized_mask = resized_mask.squeeze(0).squeeze(0);  // (roi_h, roi_w)

        // Threshold to get binary mask
        auto threshold_tensor = tenzor::ones_like(resized_mask) * static_cast<float>(threshold);
        auto binary_mask = (resized_mask > threshold_tensor).to(DType::UInt8);

        // Move only this detection's binary mask to CPU; write directly into
        // the persistent CPU output buffer (no per-iteration buffer download).
        auto binary_mask_cpu = binary_mask.to(Device::cpu());
        auto* binary_data = binary_mask_cpu.data<uint8_t>();

        // Paste into full image at ROI location
        for (int64_t h = 0; h < roi_h && (roi_y + h) < image_height; ++h) {
            for (int64_t w = 0; w < roi_w && (roi_x + w) < image_width; ++w) {
                int64_t full_idx = i * image_height * image_width +
                                   (roi_y + h) * image_width +
                                   (roi_x + w);
                int64_t binary_idx = h * roi_w + w;
                full_data[full_idx] = binary_data[binary_idx];
            }
        }
    }

    // Upload the completed mask buffer to the original device once.
    return (original_device == Device::cpu())
        ? full_masks_cpu
        : full_masks_cpu.to(original_device);
}

} // namespace detection
} // namespace nn
} // namespace tenzor
