/**
 * @file yolo.hpp
 * @brief YOLO (You Only Look Once) single-stage object detection models
 *
 * Implements YOLOv3 and YOLOv5 architectures for real-time object detection.
 * YOLOv3: Darknet53 backbone with multi-scale predictions
 * YOLOv5: CSPDarknet backbone with PANet and improved training
 *
 * References:
 * - YOLOv3: "YOLOv3: An Incremental Improvement" (Redmon & Farhadi, 2018)
 * - YOLOv5: https://github.com/ultralytics/yolov5
 */

#pragma once

#include <memory>
#include <vector>
#include <tuple>
#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/batchnorm.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/pooling.hpp"
#include "../nn/activations/activations.hpp"
#include "../nn/detection/anchors.hpp"
#include "../core/tensor.hpp"

namespace tenzor {
namespace models {

// ============================================================================
// YOLOv3 Components
// ============================================================================

/**
 * @brief Residual block for Darknet architectures.
 *
 * Darknet residual block:
 * ```
 * x -> [Conv1x1 -> BN -> LeakyReLU -> Conv3x3 -> BN -> LeakyReLU] -> + -> out
 *  |_________________________________________________________|
 * ```
 */
class DarknetResidualBlock : public nn::Module {
public:
    /**
     * @brief Construct residual block.
     *
     * @param channels Number of input/output channels
     */
    explicit DarknetResidualBlock(int64_t channels);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::Conv2d> conv1_;  // 1x1 conv (reduce channels)
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::Conv2d> conv2_;  // 3x3 conv (restore channels)
    std::shared_ptr<nn::BatchNorm2d> bn2_;
    nn::LeakyReLU act_;
};

/**
 * @brief Darknet53 backbone for YOLOv3.
 *
 * 53-layer convolutional network with residual connections.
 * Extracts features at 3 scales: 1/8, 1/16, 1/32 of input size.
 *
 * Architecture:
 * - Conv 32 (stride 1)
 * - Conv 64 (stride 2) + 1x Residual
 * - Conv 128 (stride 2) + 2x Residual
 * - Conv 256 (stride 2) + 8x Residual  -> Output 1 (1/8)
 * - Conv 512 (stride 2) + 8x Residual  -> Output 2 (1/16)
 * - Conv 1024 (stride 2) + 4x Residual -> Output 3 (1/32)
 */
class Darknet53 : public nn::Module {
public:
    /**
     * @brief Construct Darknet53 backbone.
     *
     * @param in_channels Number of input channels (default: 3 for RGB)
     */
    explicit Darknet53(int64_t in_channels = 3);

    /**
     * @brief Forward pass extracting multi-scale features.
     *
     * @param input Input images (N, 3, H, W)
     * @return Tuple of feature maps at scales 1/8, 1/16, 1/32
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get multi-scale feature maps.
     *
     * Returns features at three scales for FPN.
     *
     * @param input Input images (N, 3, H, W)
     * @return Vector of [feat_small (1/8), feat_medium (1/16), feat_large (1/32)]
     */
    auto forward_multiscale(const Variable& input) -> std::vector<Variable>;

private:
    auto make_layer(int64_t in_channels, int64_t out_channels, int64_t num_blocks)
        -> std::vector<std::shared_ptr<nn::Module>>;

    std::shared_ptr<nn::Conv2d> conv1_;  // Initial conv
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    nn::LeakyReLU act_;

    // 5 residual layer groups
    std::vector<std::shared_ptr<nn::Module>> layer1_;  // 64 channels, 1 block
    std::vector<std::shared_ptr<nn::Module>> layer2_;  // 128 channels, 2 blocks
    std::vector<std::shared_ptr<nn::Module>> layer3_;  // 256 channels, 8 blocks (1/8)
    std::vector<std::shared_ptr<nn::Module>> layer4_;  // 512 channels, 8 blocks (1/16)
    std::vector<std::shared_ptr<nn::Module>> layer5_;  // 1024 channels, 4 blocks (1/32)
};

/**
 * @brief YOLO detection head for multi-scale predictions.
 *
 * Predicts bounding boxes, objectness, and class probabilities.
 * Each grid cell predicts 3 boxes using different anchor sizes.
 *
 * Output format per scale:
 * Shape: (N, num_anchors, grid_h, grid_w, 5 + num_classes)
 * Where: [tx, ty, tw, th, objectness, class_probs...]
 */
class YOLOv3Head : public nn::Module {
public:
    /**
     * @brief Construct YOLOv3 detection head.
     *
     * @param in_channels Input feature channels
     * @param num_classes Number of object classes
     * @param num_anchors Number of anchors per grid cell (default: 3)
     */
    YOLOv3Head(int64_t in_channels, int64_t num_classes, int64_t num_anchors = 3);

    /**
     * @brief Forward pass.
     *
     * @param input Feature map (N, C, H, W)
     * @return Predictions (N, num_anchors, H, W, 5 + num_classes)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t num_anchors_;
    int64_t num_classes_;

    std::shared_ptr<nn::Conv2d> conv1_;
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::Conv2d> detect_;  // Final 1x1 conv
    nn::LeakyReLU act_;
};

/**
 * @brief YOLOv3 complete model.
 *
 * Single-stage object detector with:
 * - Darknet53 backbone
 * - Feature Pyramid Network (FPN)
 * - Multi-scale detection at 3 levels
 * - Grid-based prediction
 *
 * Input: (N, 3, 416, 416) or (N, 3, 608, 608)
 * Output: List of detections [(boxes, scores, class_ids), ...]
 */
class YOLOv3 : public nn::Module {
public:
    /**
     * @brief Construct YOLOv3 model.
     *
     * @param num_classes Number of object classes (default: 80 for COCO)
     * @param pretrained Load pretrained weights (default: false)
     * @param conf_threshold Confidence threshold for filtering (default: 0.25)
     * @param nms_threshold NMS IoU threshold (default: 0.45)
     */
    explicit YOLOv3(int64_t num_classes = 80,
                    bool pretrained = false,
                    double conf_threshold = 0.25,
                    double nms_threshold = 0.45);

    /**
     * @brief Forward pass with post-processing.
     *
     * Returns filtered detections after applying confidence threshold and NMS.
     *
     * @param input Input images (N, 3, H, W)
     * @return Tuple of (boxes, scores, class_ids) for each image
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get raw predictions before post-processing.
     *
     * @param input Input images (N, 3, H, W)
     * @return Vector of predictions at 3 scales
     */
    auto forward_raw(const Variable& input) -> std::vector<Variable>;

    /**
     * @brief Decode predictions to bounding boxes.
     *
     * Converts grid-based predictions to absolute box coordinates.
     *
     * @param predictions Raw predictions from forward_raw
     * @param img_size Input image size
     * @return Tensor of boxes (N, num_predictions, 4) in (x1, y1, x2, y2) format
     */
    auto decode_predictions(const std::vector<Variable>& predictions, int64_t img_size)
        -> Tensor;

    /**
     * @brief Post-process predictions with NMS.
     *
     * Applies confidence filtering and NMS to get final detections.
     *
     * @param boxes Decoded boxes (N, num_boxes, 4)
     * @param scores Objectness * class_probs (N, num_boxes, num_classes)
     * @return Vector of (boxes, scores, labels) per image
     */
    auto postprocess(const Tensor& boxes, const Tensor& scores)
        -> std::vector<std::tuple<Tensor, Tensor, Tensor>>;

    /**
     * @brief Load pretrained weights.
     *
     * @param path Path to weights file
     */
    auto load_pretrained(const std::string& path) -> void;

private:
    auto build_fpn_neck() -> void;

    int64_t num_classes_;
    double conf_threshold_;
    double nms_threshold_;

    // Backbone
    std::shared_ptr<Darknet53> backbone_;

    // FPN neck (top-down pathway)
    std::shared_ptr<nn::Conv2d> fpn_upsample1_;
    std::shared_ptr<nn::Conv2d> fpn_upsample2_;
    std::shared_ptr<nn::Conv2d> fpn_lateral1_;
    std::shared_ptr<nn::Conv2d> fpn_lateral2_;

    // Detection heads at 3 scales
    std::shared_ptr<YOLOv3Head> head_large_;   // 13x13 for large objects
    std::shared_ptr<YOLOv3Head> head_medium_;  // 26x26 for medium objects
    std::shared_ptr<YOLOv3Head> head_small_;   // 52x52 for small objects

    // Anchors for each scale (COCO dataset defaults)
    std::vector<std::pair<float, float>> anchors_large_;   // Large objects
    std::vector<std::pair<float, float>> anchors_medium_;  // Medium objects
    std::vector<std::pair<float, float>> anchors_small_;   // Small objects
};

// ============================================================================
// YOLOv5 Components
// ============================================================================

/**
 * @brief Cross Stage Partial (CSP) bottleneck for YOLOv5.
 *
 * CSP splits feature map into two parts, processes one part through
 * bottleneck layers, then concatenates. Reduces computation while
 * maintaining accuracy.
 */
class CSPBottleneck : public nn::Module {
public:
    /**
     * @brief Construct CSP bottleneck.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param num_blocks Number of bottleneck blocks
     * @param shortcut Use skip connections (default: true)
     */
    CSPBottleneck(int64_t in_channels,
                  int64_t out_channels,
                  int64_t num_blocks = 1,
                  bool shortcut = true);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::Conv2d> conv1_;  // Initial conv
    std::shared_ptr<nn::Conv2d> conv2_;  // Split branch conv
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::BatchNorm2d> bn2_;

    // Bottleneck blocks
    std::vector<std::shared_ptr<nn::Module>> bottlenecks_;

    std::shared_ptr<nn::Conv2d> conv_merge_;  // Merge after concat
    std::shared_ptr<nn::BatchNorm2d> bn_merge_;
    // Note: SiLU (Swish) is created locally in forward()
};

/**
 * @brief CSPDarknet backbone for YOLOv5.
 *
 * Improved version of Darknet53 with CSP connections.
 * More efficient and accurate than original Darknet.
 */
class CSPDarknet : public nn::Module {
public:
    /**
     * @brief Construct CSPDarknet.
     *
     * @param depth_multiple Depth scaling factor (0.33, 0.67, 1.0, 1.33)
     * @param width_multiple Width scaling factor (0.25, 0.5, 0.75, 1.0, 1.25)
     */
    CSPDarknet(double depth_multiple = 1.0, double width_multiple = 1.0);

    auto forward_impl(const Variable& input) -> Variable override;
    auto forward_multiscale(const Variable& input) -> std::vector<Variable>;

private:
    auto make_divisible(int64_t x, int64_t divisor = 8) -> int64_t;

    double depth_multiple_;
    double width_multiple_;

    std::shared_ptr<nn::Conv2d> stem_;  // Focus layer or conv
    std::shared_ptr<nn::BatchNorm2d> bn_stem_;
    // Note: SiLU (Swish) is created locally in forward()

    // CSP stages
    std::vector<std::shared_ptr<nn::Module>> stage1_;  // 1/2
    std::vector<std::shared_ptr<nn::Module>> stage2_;  // 1/4
    std::vector<std::shared_ptr<nn::Module>> stage3_;  // 1/8 (P3)
    std::vector<std::shared_ptr<nn::Module>> stage4_;  // 1/16 (P4)
    std::vector<std::shared_ptr<nn::Module>> stage5_;  // 1/32 (P5)
};

/**
 * @brief PANet (Path Aggregation Network) neck for YOLOv5.
 *
 * Combines top-down FPN with bottom-up path aggregation for better
 * feature fusion across scales.
 */
class PANet : public nn::Module {
public:
    /**
     * @brief Construct PANet.
     *
     * @param channels Vector of channel numbers for each scale
     */
    explicit PANet(const std::vector<int64_t>& channels);

    /**
     * @brief Forward pass (single input - not used directly).
     *
     * @param input Placeholder input
     * @return Placeholder output
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Forward pass with multi-scale features.
     *
     * @param features Input features from backbone [P3, P4, P5]
     * @return Enhanced features [P3_out, P4_out, P5_out]
     */
    auto forward_multi(const std::vector<Variable>& features) -> std::vector<Variable>;

private:
    // Top-down pathway
    std::shared_ptr<nn::Conv2d> td_conv1_;
    std::shared_ptr<nn::Conv2d> td_conv2_;

    // Bottom-up pathway
    std::shared_ptr<nn::Conv2d> bu_conv1_;
    std::shared_ptr<nn::Conv2d> bu_conv2_;

    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::BatchNorm2d> bn2_;
    std::shared_ptr<nn::BatchNorm2d> bn3_;
    std::shared_ptr<nn::BatchNorm2d> bn4_;
    // Note: SiLU (Swish) is created locally in forward()
};

/**
 * @brief YOLOv5 detection head.
 *
 * Improved detection head with anchor-free option and better loss functions.
 */
class YOLOv5Head : public nn::Module {
public:
    YOLOv5Head(int64_t in_channels, int64_t num_classes, int64_t num_anchors = 3);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t num_anchors_;
    int64_t num_classes_;

    std::shared_ptr<nn::Conv2d> conv1_;
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::Conv2d> detect_;
    // Note: SiLU (Swish) is created locally in forward()
};

/**
 * @brief YOLOv5 complete model.
 *
 * State-of-the-art single-stage detector with:
 * - CSPDarknet backbone
 * - PANet neck
 * - Improved training techniques
 * - Better anchor assignment
 * - CIoU loss
 *
 * Variants: YOLOv5n, YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x
 */
class YOLOv5 : public nn::Module {
public:
    /**
     * @brief YOLOv5 model size variants.
     */
    enum class Size {
        Nano,    // n: depth=0.33, width=0.25
        Small,   // s: depth=0.33, width=0.50
        Medium,  // m: depth=0.67, width=0.75
        Large,   // l: depth=1.00, width=1.00
        XLarge   // x: depth=1.33, width=1.25
    };

    /**
     * @brief Construct YOLOv5 model.
     *
     * @param size Model size variant
     * @param num_classes Number of object classes (default: 80)
     * @param pretrained Load pretrained weights
     * @param conf_threshold Confidence threshold
     * @param nms_threshold NMS threshold
     */
    explicit YOLOv5(Size size = Size::Small,
                    int64_t num_classes = 80,
                    bool pretrained = false,
                    double conf_threshold = 0.25,
                    double nms_threshold = 0.45);

    auto forward_impl(const Variable& input) -> Variable override;
    auto forward_raw(const Variable& input) -> std::vector<Variable>;
    auto decode_predictions(const std::vector<Variable>& predictions, int64_t img_size)
        -> Tensor;
    auto postprocess(const Tensor& boxes, const Tensor& scores)
        -> std::vector<std::tuple<Tensor, Tensor, Tensor>>;
    auto load_pretrained(const std::string& path) -> void;

private:
    auto get_size_params(Size size) -> std::pair<double, double>;

    Size size_;
    int64_t num_classes_;
    double conf_threshold_;
    double nms_threshold_;

    std::shared_ptr<CSPDarknet> backbone_;
    std::shared_ptr<PANet> neck_;

    std::shared_ptr<YOLOv5Head> head_p3_;  // 80x80
    std::shared_ptr<YOLOv5Head> head_p4_;  // 40x40
    std::shared_ptr<YOLOv5Head> head_p5_;  // 20x20

    // Auto-learned anchors (COCO defaults)
    std::vector<std::pair<float, float>> anchors_p3_;
    std::vector<std::pair<float, float>> anchors_p4_;
    std::vector<std::pair<float, float>> anchors_p5_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create YOLOv3 model.
 *
 * @param num_classes Number of classes (default: 80 for COCO)
 * @param pretrained Load pretrained weights
 * @return Shared pointer to YOLOv3 model
 */
auto yolov3(int64_t num_classes = 80, bool pretrained = false)
    -> std::shared_ptr<YOLOv3>;

/**
 * @brief Create YOLOv5 Nano model (smallest, fastest).
 *
 * Parameters: ~1.9M
 * Speed: ~45 FPS on V100
 */
auto yolov5n(int64_t num_classes = 80, bool pretrained = false)
    -> std::shared_ptr<YOLOv5>;

/**
 * @brief Create YOLOv5 Small model.
 *
 * Parameters: ~7.2M
 * Speed: ~35 FPS on V100
 */
auto yolov5s(int64_t num_classes = 80, bool pretrained = false)
    -> std::shared_ptr<YOLOv5>;

/**
 * @brief Create YOLOv5 Medium model.
 *
 * Parameters: ~21.2M
 * Speed: ~25 FPS on V100
 */
auto yolov5m(int64_t num_classes = 80, bool pretrained = false)
    -> std::shared_ptr<YOLOv5>;

/**
 * @brief Create YOLOv5 Large model.
 *
 * Parameters: ~46.5M
 * Speed: ~18 FPS on V100
 */
auto yolov5l(int64_t num_classes = 80, bool pretrained = false)
    -> std::shared_ptr<YOLOv5>;

/**
 * @brief Create YOLOv5 XLarge model (largest, most accurate).
 *
 * Parameters: ~86.7M
 * Speed: ~12 FPS on V100
 */
auto yolov5x(int64_t num_classes = 80, bool pretrained = false)
    -> std::shared_ptr<YOLOv5>;

} // namespace models
} // namespace tenzor
