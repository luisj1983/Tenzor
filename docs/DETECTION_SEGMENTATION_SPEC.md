# Detection and Segmentation Models - Architecture Specification

**Document Version:** 1.0
**Date:** 2025-10-18
**Purpose:** Complete architectural specifications for implementing object detection and semantic segmentation models in Tenzor

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Faster R-CNN](#1-faster-r-cnn)
3. [YOLO Variants (v3, v5)](#2-yolo-variants-v3-v5)
4. [Mask R-CNN](#3-mask-r-cnn)
5. [U-Net](#4-u-net)
6. [DeepLab v3+](#5-deeplab-v3)
7. [New Components Required](#6-new-components-required-for-tenzor)
8. [Implementation Priorities](#7-implementation-priorities)
9. [References](#8-references)

---

## Executive Summary

This document provides comprehensive architectural specifications for implementing five major object detection and semantic segmentation models in Tenzor. Based on analysis of existing Tenzor components, the following NEW operations must be implemented:

**Critical New Components:**
1. **ROI Operations**: ROI Pooling, ROI Align (with bilinear interpolation)
2. **Post-processing**: Non-Maximum Suppression (NMS), IoU computation
3. **Anchor Generation**: Multi-scale anchor box generation for RPN/YOLO
4. **Specialized Convolutions**: Atrous/Dilated convolution (for DeepLab)
5. **Upsampling**: Bilinear/Nearest neighbor interpolation for segmentation
6. **Loss Functions**: IoU-based losses (CIoU, GIoU), combined losses

**Existing Tenzor Components (Can Reuse):**
- Conv2d, Conv1d, ConvTranspose2d (transposed convolution)
- MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- Linear layers
- BatchNorm
- Loss functions: BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, DiceLoss, SmoothL1Loss, FocalLoss
- Basic tensor operations: reshape, transpose, concatenate, split

---

## 1. Faster R-CNN

### 1.1 Architecture Overview

**Two-Stage Detection Pipeline:**
```
Input Image (H×W×3)
    ↓
Backbone CNN (e.g., ResNet-50)
    ↓
Feature Maps (H/16 × W/16 × 2048)
    ↓
    ├─→ Region Proposal Network (RPN)
    │       ↓
    │   Proposals (2000 boxes)
    │       ↓
    └─→ ROI Pooling/Align
            ↓
        Detection Head
            ↓
        ├─→ Classification (N classes)
        └─→ Bounding Box Regression
```

### 1.2 Region Proposal Network (RPN)

**Architecture:**
```cpp
// RPN slides a small network over feature map
// Input: Feature map (H×W×C)
// Output: Object proposals with scores

class RegionProposalNetwork : public Module {
public:
    // 3×3 conv for spatial context
    Conv2d conv(in_channels, 512, 3, 1, 1);

    // Classification: object vs background (2×k anchors)
    Conv2d cls_logits(512, num_anchors * 2, 1);

    // Regression: box coordinates (4×k anchors)
    Conv2d bbox_pred(512, num_anchors * 4, 1);

private:
    int num_anchors = 9;  // 3 scales × 3 aspect ratios
};
```

**Forward Pass:**
```cpp
Variable forward(const Variable& features) {
    // 1. Apply 3×3 conv
    auto x = relu(conv(features));  // [N, 512, H, W]

    // 2. Classification scores
    auto cls = cls_logits(x);  // [N, 2*k, H, W]

    // 3. Bounding box deltas
    auto bbox = bbox_pred(x);  // [N, 4*k, H, W]

    // 4. Reshape to [N, H*W*k, 2] and [N, H*W*k, 4]
    cls = permute(cls, {0, 2, 3, 1}).reshape({N, -1, 2});
    bbox = permute(bbox, {0, 2, 3, 1}).reshape({N, -1, 4});

    return {cls, bbox};
}
```

### 1.3 Anchor Generation

**Specifications:**
- **Base Anchors:** Generated at each position on feature map
- **Scales:** {128², 256², 512²} pixels (or {32², 64², 128²} for smaller inputs)
- **Aspect Ratios:** {1:2, 1:1, 2:1}
- **Total Anchors:** 9 per feature map position
- **Stride:** 16 (for typical VGG/ResNet backbones)

**Implementation:**
```cpp
struct AnchorGenerator {
    std::vector<float> scales = {128.0f, 256.0f, 512.0f};
    std::vector<float> aspect_ratios = {0.5f, 1.0f, 2.0f};
    int stride = 16;

    // Generate anchors for feature map of size (H, W)
    Tensor generate_anchors(int feat_h, int feat_w) {
        // Output: [H*W*9, 4] tensor of (x1, y1, x2, y2)
        std::vector<float> anchors;

        for (int y = 0; y < feat_h; ++y) {
            for (int x = 0; x < feat_w; ++x) {
                float cx = (x + 0.5f) * stride;
                float cy = (y + 0.5f) * stride;

                for (float scale : scales) {
                    for (float ratio : aspect_ratios) {
                        float w = scale * sqrt(ratio);
                        float h = scale / sqrt(ratio);

                        anchors.push_back(cx - w/2);  // x1
                        anchors.push_back(cy - h/2);  // y1
                        anchors.push_back(cx + w/2);  // x2
                        anchors.push_back(cy + h/2);  // y2
                    }
                }
            }
        }
        return Tensor(anchors, {feat_h * feat_w * 9, 4});
    }
};
```

### 1.4 ROI Pooling

**Purpose:** Convert variable-sized ROIs to fixed-size feature maps

**Algorithm:**
```cpp
class ROIPooling : public Module {
public:
    ROIPooling(int output_h, int output_w, float spatial_scale)
        : output_h_(output_h), output_w_(output_w),
          spatial_scale_(spatial_scale) {}

    // Input: features [N, C, H, W], rois [K, 5] where roi = [batch_idx, x1, y1, x2, y2]
    // Output: [K, C, output_h, output_w]
    Tensor forward(const Tensor& features, const Tensor& rois) {
        int K = rois.size(0);
        int C = features.size(1);

        Tensor output({K, C, output_h_, output_w_});

        for (int k = 0; k < K; ++k) {
            int batch_idx = rois[k][0];
            float x1 = rois[k][1] * spatial_scale_;
            float y1 = rois[k][2] * spatial_scale_;
            float x2 = rois[k][3] * spatial_scale_;
            float y2 = rois[k][4] * spatial_scale_;

            // Round to integer coordinates
            int ix1 = floor(x1), iy1 = floor(y1);
            int ix2 = ceil(x2), iy2 = ceil(y2);

            float roi_h = max(iy2 - iy1, 1);
            float roi_w = max(ix2 - ix1, 1);

            // Divide ROI into grid
            float bin_h = roi_h / output_h_;
            float bin_w = roi_w / output_w_;

            for (int ph = 0; ph < output_h_; ++ph) {
                for (int pw = 0; pw < output_w_; ++pw) {
                    int hstart = floor(iy1 + ph * bin_h);
                    int wstart = floor(ix1 + pw * bin_w);
                    int hend = ceil(iy1 + (ph + 1) * bin_h);
                    int wend = ceil(ix1 + (pw + 1) * bin_w);

                    // Max pooling over the bin
                    for (int c = 0; c < C; ++c) {
                        float max_val = -INFINITY;
                        for (int h = hstart; h < hend; ++h) {
                            for (int w = wstart; w < wend; ++w) {
                                max_val = max(max_val, features[batch_idx][c][h][w]);
                            }
                        }
                        output[k][c][ph][pw] = max_val;
                    }
                }
            }
        }
        return output;
    }

private:
    int output_h_, output_w_;
    float spatial_scale_;  // Feature map stride (e.g., 1/16)
};
```

### 1.5 ROI Align (Improved Version)

**Key Improvement:** Uses bilinear interpolation instead of quantization

**Algorithm:**
```cpp
class ROIAlign : public Module {
public:
    ROIAlign(int output_h, int output_w, float spatial_scale, int sampling_ratio = 2)
        : output_h_(output_h), output_w_(output_w),
          spatial_scale_(spatial_scale), sampling_ratio_(sampling_ratio) {}

    Tensor forward(const Tensor& features, const Tensor& rois) {
        int K = rois.size(0);
        int C = features.size(1);
        int H = features.size(2);
        int W = features.size(3);

        Tensor output({K, C, output_h_, output_w_});

        for (int k = 0; k < K; ++k) {
            int batch_idx = rois[k][0];
            float x1 = rois[k][1] * spatial_scale_;
            float y1 = rois[k][2] * spatial_scale_;
            float x2 = rois[k][3] * spatial_scale_;
            float y2 = rois[k][4] * spatial_scale_;

            float roi_h = y2 - y1;
            float roi_w = x2 - x1;

            float bin_h = roi_h / output_h_;
            float bin_w = roi_w / output_w_;

            // Sample points in each bin
            int roi_bin_grid_h = (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_h / output_h_);
            int roi_bin_grid_w = (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_w / output_w_);

            int count = roi_bin_grid_h * roi_bin_grid_w;

            for (int ph = 0; ph < output_h_; ++ph) {
                for (int pw = 0; pw < output_w_; ++pw) {
                    for (int c = 0; c < C; ++c) {
                        float sum = 0.0f;

                        // Sample within the bin
                        for (int iy = 0; iy < roi_bin_grid_h; ++iy) {
                            for (int ix = 0; ix < roi_bin_grid_w; ++ix) {
                                float y = y1 + ph * bin_h + (iy + 0.5f) * bin_h / roi_bin_grid_h;
                                float x = x1 + pw * bin_w + (ix + 0.5f) * bin_w / roi_bin_grid_w;

                                // Bilinear interpolation
                                sum += bilinear_interpolate(features[batch_idx][c], H, W, y, x);
                            }
                        }
                        output[k][c][ph][pw] = sum / count;
                    }
                }
            }
        }
        return output;
    }

private:
    // Bilinear interpolation at (y, x) in 2D tensor
    float bilinear_interpolate(const Tensor& data, int H, int W, float y, float x) {
        if (y < -1.0 || y > H || x < -1.0 || x > W) return 0.0f;

        y = max(0.0f, min(y, H - 1.0f));
        x = max(0.0f, min(x, W - 1.0f));

        int y_low = floor(y);
        int x_low = floor(x);
        int y_high = min(y_low + 1, H - 1);
        int x_high = min(x_low + 1, W - 1);

        float ly = y - y_low;
        float lx = x - x_low;
        float hy = 1.0f - ly;
        float hx = 1.0f - lx;

        float v1 = data[y_low][x_low];
        float v2 = data[y_low][x_high];
        float v3 = data[y_high][x_low];
        float v4 = data[y_high][x_high];

        float w1 = hy * hx;
        float w2 = hy * lx;
        float w3 = ly * hx;
        float w4 = ly * lx;

        return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
    }

    int output_h_, output_w_, sampling_ratio_;
    float spatial_scale_;
};
```

### 1.6 Non-Maximum Suppression (NMS)

**Purpose:** Remove duplicate detections

**Algorithm:**
```cpp
struct NMS {
    // boxes: [N, 4] (x1, y1, x2, y2)
    // scores: [N]
    // iou_threshold: typically 0.5-0.7
    static std::vector<int> nms(const Tensor& boxes, const Tensor& scores, float iou_threshold) {
        int N = boxes.size(0);

        // Sort by scores descending
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&scores](int i, int j) { return scores[i] > scores[j]; });

        std::vector<bool> suppressed(N, false);
        std::vector<int> keep;

        for (int i = 0; i < N; ++i) {
            int idx = indices[i];
            if (suppressed[idx]) continue;

            keep.push_back(idx);

            // Suppress boxes with high IoU
            for (int j = i + 1; j < N; ++j) {
                int idx2 = indices[j];
                if (suppressed[idx2]) continue;

                float iou = compute_iou(boxes[idx], boxes[idx2]);
                if (iou > iou_threshold) {
                    suppressed[idx2] = true;
                }
            }
        }
        return keep;
    }

    static float compute_iou(const Tensor& box1, const Tensor& box2) {
        float x1 = max(box1[0].item<float>(), box2[0].item<float>());
        float y1 = max(box1[1].item<float>(), box2[1].item<float>());
        float x2 = min(box1[2].item<float>(), box2[2].item<float>());
        float y2 = min(box1[3].item<float>(), box2[3].item<float>());

        float inter_area = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);

        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
        float union_area = area1 + area2 - inter_area;

        return inter_area / (union_area + 1e-6f);
    }
};
```

### 1.7 Loss Functions

**RPN Losses:**
```cpp
class RPNLoss {
public:
    Variable forward(const Variable& cls_logits, const Variable& bbox_pred,
                    const Tensor& cls_targets, const Variable& bbox_targets,
                    const Tensor& bbox_weights) {
        // Classification: Binary cross entropy
        auto cls_loss = BCEWithLogitsLoss()(cls_logits, cls_targets);

        // Regression: Smooth L1 (only for positive anchors)
        auto reg_loss = SmoothL1Loss()(bbox_pred * bbox_weights, bbox_targets * bbox_weights);

        return cls_loss + lambda_ * reg_loss;
    }

private:
    float lambda_ = 1.0f;  // Balance factor
};
```

**Detection Losses:**
```cpp
class DetectionLoss {
public:
    Variable forward(const Variable& cls_scores, const Variable& bbox_pred,
                    const Tensor& cls_targets, const Variable& bbox_targets) {
        // Classification: Cross entropy for N classes
        auto cls_loss = CrossEntropyLoss()(cls_scores, cls_targets);

        // Regression: Smooth L1 for bounding boxes
        auto reg_loss = SmoothL1Loss()(bbox_pred, bbox_targets);

        return cls_loss + lambda_ * reg_loss;
    }

private:
    float lambda_ = 1.0f;
};
```

### 1.8 Input/Output Formats

**Input:**
- Image: `[N, 3, H, W]` (typically H=W=800 or 1024)
- Preprocessing: Normalize to ImageNet mean/std

**Output:**
- Boxes: `[K, 4]` (x1, y1, x2, y2) in pixel coordinates
- Scores: `[K, num_classes]` class probabilities
- Labels: `[K]` predicted class indices

**Post-Processing:**
1. Apply NMS per class
2. Filter by confidence threshold (e.g., 0.05)
3. Keep top-K detections (e.g., K=100)

---

## 2. YOLO Variants (v3, v5)

### 2.1 Architecture Overview

**Single-Stage Detection:**
```
Input Image (416×416×3)
    ↓
Backbone (Darknet-53 / CSPDarknet)
    ↓
Feature Pyramid Network (FPN)
    ↓
Multi-Scale Predictions
    ├─→ Small Objects (52×52 grid)
    ├─→ Medium Objects (26×26 grid)
    └─→ Large Objects (13×13 grid)
```

### 2.2 YOLOv3 Specifications

**Backbone: Darknet-53**
- 53 convolutional layers
- Residual connections
- No pooling (uses stride-2 convolutions for downsampling)
- Output stride: 32 (for 13×13 grid from 416×416 input)

**Detection Heads:**
```cpp
class YOLOv3Head : public Module {
public:
    YOLOv3Head(int num_classes, std::vector<int> anchors_per_scale)
        : num_classes_(num_classes) {
        // Each grid cell predicts 3 bounding boxes
        // Each box: (tx, ty, tw, th, confidence, class_scores)
        int output_channels = anchors_per_scale * (5 + num_classes);

        // Three detection layers at different scales
        detect_large = Conv2d(1024, output_channels, 1);   // 13×13
        detect_medium = Conv2d(512, output_channels, 1);   // 26×26
        detect_small = Conv2d(256, output_channels, 1);    // 52×52
    }

    std::vector<Variable> forward(const std::vector<Variable>& features) {
        // features[0]: 13×13, features[1]: 26×26, features[2]: 52×52
        auto large = detect_large(features[0]);
        auto medium = detect_medium(features[1]);
        auto small = detect_small(features[2]);

        return {large, medium, small};
    }

private:
    int num_classes_;
};
```

### 2.3 Grid-Based Prediction

**Bounding Box Encoding:**
```cpp
struct YOLODecoder {
    // Decode predictions to bounding boxes
    // pred: [N, 3, S, S, 5+C] where S is grid size, 3 is anchors per cell
    Tensor decode_boxes(const Tensor& pred, const std::vector<std::pair<float, float>>& anchors,
                        int stride, int img_size) {
        int N = pred.size(0);
        int S = pred.size(2);
        int num_anchors = anchors.size();

        Tensor boxes({N, num_anchors * S * S, 4});

        for (int n = 0; n < N; ++n) {
            for (int a = 0; a < num_anchors; ++a) {
                for (int i = 0; i < S; ++i) {
                    for (int j = 0; j < S; ++j) {
                        // Get predictions
                        float tx = sigmoid(pred[n][a][i][j][0]);
                        float ty = sigmoid(pred[n][a][i][j][1]);
                        float tw = pred[n][a][i][j][2];
                        float th = pred[n][a][i][j][3];

                        // Decode center coordinates
                        float bx = (j + tx) * stride;
                        float by = (i + ty) * stride;

                        // Decode width and height
                        float bw = anchors[a].first * exp(tw);
                        float bh = anchors[a].second * exp(th);

                        // Convert to x1, y1, x2, y2
                        int idx = a * S * S + i * S + j;
                        boxes[n][idx][0] = bx - bw / 2;
                        boxes[n][idx][1] = by - bh / 2;
                        boxes[n][idx][2] = bx + bw / 2;
                        boxes[n][idx][3] = by + bh / 2;
                    }
                }
            }
        }
        return boxes;
    }
};
```

### 2.4 Anchor Boxes

**YOLOv3 Anchors (COCO Dataset):**
```cpp
struct YOLOv3Anchors {
    // Anchors for 416×416 input
    // Format: (width, height) in pixels

    // Small objects (52×52 grid, stride=8)
    std::vector<std::pair<float, float>> small = {
        {10, 13}, {16, 30}, {33, 23}
    };

    // Medium objects (26×26 grid, stride=16)
    std::vector<std::pair<float, float>> medium = {
        {30, 61}, {62, 45}, {59, 119}
    };

    // Large objects (13×13 grid, stride=32)
    std::vector<std::pair<float, float>> large = {
        {116, 90}, {156, 198}, {373, 326}
    };
};
```

**Anchor Generation (K-Means Clustering):**
```python
# Pseudocode for generating custom anchors
def generate_anchors(dataset, num_clusters=9):
    # Extract all ground truth box dimensions
    widths = []
    heights = []
    for image, boxes in dataset:
        for box in boxes:
            widths.append(box.w)
            heights.append(box.h)

    # K-means clustering on (w, h) pairs
    data = np.column_stack([widths, heights])
    centroids = kmeans(data, n_clusters=num_clusters, distance=iou_distance)

    # Sort by area and split into 3 scales
    centroids = sorted(centroids, key=lambda x: x[0] * x[1])
    return {
        'small': centroids[0:3],
        'medium': centroids[3:6],
        'large': centroids[6:9]
    }
```

### 2.5 YOLOv5 Improvements

**Key Changes:**
1. **CSPDarknet Backbone:** Cross Stage Partial connections
2. **PANet:** Path Aggregation Network for feature fusion
3. **Auto-Anchor Learning:** Automatic anchor optimization
4. **Mosaic Augmentation:** 4-image mixing during training
5. **Adaptive Loss Weights:** Dynamic balancing of loss components

**Loss Function (YOLOv5):**
```cpp
class YOLOv5Loss {
public:
    Variable forward(const Variable& predictions, const Tensor& targets) {
        // predictions: [N, num_anchors, S, S, 5+C]
        // targets: [N, max_objects, 6] (class, x, y, w, h, conf)

        Variable box_loss = compute_ciou_loss(predictions, targets);
        Variable obj_loss = compute_objectness_loss(predictions, targets);
        Variable cls_loss = compute_classification_loss(predictions, targets);

        // Weighted sum with different weights for each scale
        // P3 (small): 4.0, P4 (medium): 1.0, P5 (large): 0.4
        return box_loss * box_weight_ +
               obj_loss * obj_weight_ +
               cls_loss * cls_weight_;
    }

private:
    // CIoU Loss for bounding boxes
    Variable compute_ciou_loss(const Variable& pred, const Tensor& target) {
        // Complete IoU: IoU + distance + aspect ratio
        Variable iou = compute_iou(pred, target);
        Variable distance_term = compute_distance(pred, target);
        Variable aspect_term = compute_aspect_ratio(pred, target);

        return 1.0f - iou + distance_term + aspect_term;
    }

    // BCE for objectness
    Variable compute_objectness_loss(const Variable& pred, const Tensor& target) {
        auto obj_pred = sigmoid(pred.select(-1, 4));  // Confidence
        return BCELoss()(obj_pred, target);
    }

    // BCE for classification
    Variable compute_classification_loss(const Variable& pred, const Tensor& target) {
        auto cls_pred = sigmoid(pred.slice(-1, 5, -1));  // Class scores
        return BCELoss()(cls_pred, target);
    }

    float box_weight_ = 0.05f;
    float obj_weight_ = 1.0f;
    float cls_weight_ = 0.5f;
};
```

### 2.6 Multi-Scale Predictions

**Feature Pyramid Implementation:**
```cpp
class YOLOFPNNeck : public Module {
public:
    YOLOFPNNeck(std::vector<int> channels) {
        // Bottom-up: already done by backbone

        // Top-down pathway
        upsample1 = nn::Upsample(2, "nearest");
        conv1 = Conv2d(channels[2], channels[1], 1);

        upsample2 = nn::Upsample(2, "nearest");
        conv2 = Conv2d(channels[1], channels[0], 1);

        // Lateral connections
        lateral1 = Conv2d(channels[1], channels[1], 1);
        lateral2 = Conv2d(channels[0], channels[0], 1);
    }

    std::vector<Variable> forward(const std::vector<Variable>& features) {
        // features: [C5 (large), C4 (medium), C3 (small)]

        // Top-down
        auto p5 = features[0];
        auto p4 = lateral1(features[1]) + upsample1(conv1(p5));
        auto p3 = lateral2(features[2]) + upsample2(conv2(p4));

        return {p3, p4, p5};  // Small, medium, large
    }
};
```

### 2.7 Input/Output Formats

**Input:**
- Image: `[N, 3, 416, 416]` or `[N, 3, 640, 640]` (YOLOv5)
- Preprocessing: Normalize to [0, 1], letterbox resize

**Output:**
- Predictions: `[N, num_predictions, 5+C]`
  - `num_predictions = (52×52 + 26×26 + 13×13) × 3 = 10647` for 416×416
  - Format: `[x, y, w, h, confidence, class_scores...]`

**Post-Processing:**
1. Filter by confidence threshold (e.g., 0.25)
2. Apply NMS per class (IoU threshold 0.45)
3. Keep top-K detections

---

## 3. Mask R-CNN

### 3.1 Architecture Overview

**Extension of Faster R-CNN:**
```
Input Image
    ↓
Backbone (ResNet + FPN)
    ↓
RPN → Proposals
    ↓
ROI Align
    ↓
    ├─→ Box Head → Classification + BBox Regression
    └─→ Mask Head → Pixel-wise Segmentation Masks
```

### 3.2 Mask Prediction Branch

**Architecture:**
```cpp
class MaskHead : public Module {
public:
    MaskHead(int in_channels, int num_classes, int mask_size = 28)
        : num_classes_(num_classes), mask_size_(mask_size) {
        // 4 conv layers for feature extraction
        conv1 = Conv2d(in_channels, 256, 3, 1, 1);
        conv2 = Conv2d(256, 256, 3, 1, 1);
        conv3 = Conv2d(256, 256, 3, 1, 1);
        conv4 = Conv2d(256, 256, 3, 1, 1);

        // Transposed conv for upsampling
        deconv = ConvTranspose2d(256, 256, 2, 2);  // 2x upsampling

        // 1×1 conv for class-specific masks
        mask_pred = Conv2d(256, num_classes, 1);
    }

    Variable forward(const Variable& roi_features) {
        // roi_features: [K, 256, 14, 14] from ROI Align

        auto x = relu(conv1(roi_features));  // [K, 256, 14, 14]
        x = relu(conv2(x));
        x = relu(conv3(x));
        x = relu(conv4(x));
        x = relu(deconv(x));  // [K, 256, 28, 28]
        auto masks = mask_pred(x);  // [K, num_classes, 28, 28]

        return masks;
    }

private:
    int num_classes_, mask_size_;
};
```

### 3.3 ROI Align for Masks

**Configuration:**
- Output size: 14×14 (before mask head) or 7×7 (for box head)
- Sampling ratio: 2 (4 sample points per bin)
- Spatial scale: 1/4, 1/8, 1/16, 1/32 (depending on FPN level)

**Feature Pyramid Selection:**
```cpp
struct FPNROIAlign {
    // Assign ROIs to FPN levels based on size
    int get_fpn_level(float roi_w, float roi_h, int min_level = 2, int max_level = 5) {
        float roi_area = roi_w * roi_h;
        float target_level = 4.0f + log2(sqrt(roi_area) / 224.0f);
        return std::clamp(int(target_level), min_level, max_level);
    }

    Tensor forward(const std::vector<Tensor>& fpn_features, const Tensor& rois) {
        // Group ROIs by FPN level
        std::map<int, std::vector<int>> level_to_rois;
        for (int i = 0; i < rois.size(0); ++i) {
            float w = rois[i][3] - rois[i][1];
            float h = rois[i][4] - rois[i][2];
            int level = get_fpn_level(w, h);
            level_to_rois[level].push_back(i);
        }

        // Apply ROI Align at each level
        std::vector<Tensor> aligned;
        for (auto& [level, roi_indices] : level_to_rois) {
            auto level_rois = rois.index_select(0, roi_indices);
            auto level_features = fpn_features[level - 2];
            aligned.push_back(roi_align(level_features, level_rois, 14, 14, 1.0f / (4 << level)));
        }

        return cat(aligned, 0);
    }
};
```

### 3.4 Mask Loss

**Binary Cross Entropy per Pixel:**
```cpp
class MaskLoss {
public:
    Variable forward(const Variable& mask_logits, const Tensor& mask_targets,
                    const Tensor& roi_labels) {
        // mask_logits: [K, num_classes, 28, 28]
        // mask_targets: [K, 28, 28]
        // roi_labels: [K] class indices

        int K = mask_logits.size(0);
        int H = mask_logits.size(2);
        int W = mask_logits.size(3);

        // Select mask for predicted class
        Variable selected_masks({K, H, W});
        for (int k = 0; k < K; ++k) {
            int label = roi_labels[k].item<int>();
            selected_masks[k] = mask_logits[k][label];
        }

        // Binary cross entropy
        return BCEWithLogitsLoss()(selected_masks, mask_targets);
    }
};
```

### 3.5 Binary Mask Representation

**Format:**
- Training: 28×28 binary mask per ROI
- Inference: Resize to ROI size, then paste into full image
- Threshold: 0.5 (after sigmoid)

**Mask Post-Processing:**
```cpp
struct MaskPostProcessor {
    Tensor process_masks(const Tensor& mask_logits, const Tensor& boxes,
                        int img_h, int img_w) {
        // mask_logits: [K, 28, 28] (class-selected)
        // boxes: [K, 4]

        Tensor full_masks({K, img_h, img_w});

        for (int k = 0; k < K; ++k) {
            // Get ROI coordinates
            int x1 = boxes[k][0], y1 = boxes[k][1];
            int x2 = boxes[k][2], y2 = boxes[k][3];
            int roi_w = x2 - x1, roi_h = y2 - y1;

            // Resize 28×28 mask to ROI size using bilinear interpolation
            auto resized_mask = resize_bilinear(mask_logits[k], roi_h, roi_w);

            // Apply threshold
            auto binary_mask = (sigmoid(resized_mask) > 0.5).to(DType::UInt8);

            // Paste into full image
            full_masks[k].slice(0, y1, y2).slice(1, x1, x2) = binary_mask;
        }

        return full_masks;
    }
};
```

### 3.6 Training Strategy

**Two-Stage Training:**
1. **Stage 1:** Train RPN and box head (freeze mask head)
2. **Stage 2:** Fine-tune all components including mask head

**Loss Combination:**
```cpp
class MaskRCNNLoss {
public:
    Variable forward(const Variable& rpn_cls, const Variable& rpn_bbox,
                    const Variable& det_cls, const Variable& det_bbox,
                    const Variable& mask_logits, const Targets& targets) {
        auto l_rpn_cls = rpn_classification_loss(rpn_cls, targets);
        auto l_rpn_bbox = rpn_regression_loss(rpn_bbox, targets);
        auto l_det_cls = detection_classification_loss(det_cls, targets);
        auto l_det_bbox = detection_regression_loss(det_bbox, targets);
        auto l_mask = mask_loss(mask_logits, targets);

        return l_rpn_cls + l_rpn_bbox + l_det_cls + l_det_bbox + l_mask;
    }
};
```

### 3.7 Input/Output Formats

**Input:**
- Image: `[N, 3, H, W]` (variable size, typically 800-1333)
- Preprocessing: Same as Faster R-CNN

**Output:**
- Boxes: `[K, 4]`
- Labels: `[K]`
- Scores: `[K]`
- Masks: `[K, H, W]` binary masks (0 or 1)

---

## 4. U-Net

### 4.1 Architecture Overview

**Encoder-Decoder with Skip Connections:**
```
Input (572×572×1)
    ↓
Encoder (Contracting Path)
    ├─→ Conv-Conv-MaxPool → 1/2 size
    ├─→ Conv-Conv-MaxPool → 1/4 size
    ├─→ Conv-Conv-MaxPool → 1/8 size
    └─→ Conv-Conv-MaxPool → 1/16 size
            ↓
        Bottleneck
            ↓
Decoder (Expanding Path)
    ├─→ UpConv-Concat-Conv-Conv → 1/8 size
    ├─→ UpConv-Concat-Conv-Conv → 1/4 size
    ├─→ UpConv-Concat-Conv-Conv → 1/2 size
    └─→ UpConv-Concat-Conv-Conv → 1/1 size
            ↓
        1×1 Conv → Output
```

### 4.2 Encoder (Contracting Path)

**Repeated Pattern:**
```cpp
class EncoderBlock : public Module {
public:
    EncoderBlock(int in_channels, int out_channels, bool apply_pooling = true)
        : apply_pooling_(apply_pooling) {
        // Two 3×3 convolutions
        conv1 = Conv2d(in_channels, out_channels, 3, 1, 1);
        bn1 = BatchNorm2d(out_channels);

        conv2 = Conv2d(out_channels, out_channels, 3, 1, 1);
        bn2 = BatchNorm2d(out_channels);

        if (apply_pooling) {
            pool = MaxPool2d(2, 2);  // 2×2 max pooling
        }
    }

    std::pair<Variable, Variable> forward(const Variable& x) {
        auto c1 = relu(bn1(conv1(x)));
        auto c2 = relu(bn2(conv2(c1)));  // Skip connection

        if (apply_pooling_) {
            auto pooled = pool(c2);
            return {pooled, c2};  // Return pooled and skip
        }
        return {c2, c2};
    }

private:
    bool apply_pooling_;
};
```

**Full Encoder:**
```cpp
class UNetEncoder : public Module {
public:
    UNetEncoder(int in_channels = 1) {
        enc1 = EncoderBlock(in_channels, 64);
        enc2 = EncoderBlock(64, 128);
        enc3 = EncoderBlock(128, 256);
        enc4 = EncoderBlock(256, 512);
        bottleneck = EncoderBlock(512, 1024, false);  // No pooling
    }

    std::vector<Variable> forward(const Variable& x) {
        auto [e1, skip1] = enc1.forward(x);       // 64 channels
        auto [e2, skip2] = enc2.forward(e1);      // 128 channels
        auto [e3, skip3] = enc3.forward(e2);      // 256 channels
        auto [e4, skip4] = enc4.forward(e3);      // 512 channels
        auto [b, _] = bottleneck.forward(e4);     // 1024 channels

        return {b, skip4, skip3, skip2, skip1};
    }
};
```

### 4.3 Decoder (Expanding Path)

**Upsampling + Skip Connection + Convolutions:**
```cpp
class DecoderBlock : public Module {
public:
    DecoderBlock(int in_channels, int out_channels) {
        // Transposed convolution for upsampling
        upconv = ConvTranspose2d(in_channels, out_channels, 2, 2);

        // Two 3×3 convolutions after concatenation
        conv1 = Conv2d(in_channels, out_channels, 3, 1, 1);  // in_channels because of concat
        bn1 = BatchNorm2d(out_channels);

        conv2 = Conv2d(out_channels, out_channels, 3, 1, 1);
        bn2 = BatchNorm2d(out_channels);
    }

    Variable forward(const Variable& x, const Variable& skip) {
        // Upsample
        auto upsampled = upconv(x);

        // Concatenate with skip connection
        // Handle size mismatch if necessary
        auto cropped_skip = center_crop(skip, upsampled.size(2), upsampled.size(3));
        auto concat = cat({upsampled, cropped_skip}, 1);  // Concat on channel dimension

        // Convolutions
        auto c1 = relu(bn1(conv1(concat)));
        auto c2 = relu(bn2(conv2(c1)));

        return c2;
    }

private:
    // Center crop skip to match upsampled size
    Variable center_crop(const Variable& x, int target_h, int target_w) {
        int h = x.size(2), w = x.size(3);
        int start_h = (h - target_h) / 2;
        int start_w = (w - target_w) / 2;
        return x.slice(2, start_h, start_h + target_h)
                .slice(3, start_w, start_w + target_w);
    }
};
```

**Full Decoder:**
```cpp
class UNetDecoder : public Module {
public:
    UNetDecoder() {
        dec1 = DecoderBlock(1024, 512);
        dec2 = DecoderBlock(512, 256);
        dec3 = DecoderBlock(256, 128);
        dec4 = DecoderBlock(128, 64);
    }

    Variable forward(const Variable& bottleneck,
                    const std::vector<Variable>& skips) {
        auto d1 = dec1.forward(bottleneck, skips[0]);  // 512 channels
        auto d2 = dec2.forward(d1, skips[1]);          // 256 channels
        auto d3 = dec3.forward(d2, skips[2]);          // 128 channels
        auto d4 = dec4.forward(d3, skips[3]);          // 64 channels

        return d4;
    }
};
```

### 4.4 Complete U-Net Model

```cpp
class UNet : public Module {
public:
    UNet(int in_channels = 1, int num_classes = 2) {
        encoder = UNetEncoder(in_channels);
        decoder = UNetDecoder();

        // Final 1×1 convolution for classification
        final_conv = Conv2d(64, num_classes, 1);
    }

    Variable forward(const Variable& x) {
        // Encoder
        auto encoder_outputs = encoder.forward(x);
        auto bottleneck = encoder_outputs[0];
        std::vector<Variable> skips(encoder_outputs.begin() + 1, encoder_outputs.end());

        // Decoder
        auto decoded = decoder.forward(bottleneck, skips);

        // Final classification
        auto output = final_conv(decoded);

        return output;  // [N, num_classes, H, W]
    }
};
```

### 4.5 Skip Connections

**Purpose:** Preserve spatial information lost during downsampling

**Implementation Details:**
- Concatenate encoder features with decoder features
- May require cropping if sizes don't match (due to valid padding)
- Alternative: Use padding='same' to maintain sizes

### 4.6 Loss Functions

**Binary Segmentation:**
```cpp
class UNetBinaryLoss {
public:
    Variable forward(const Variable& predictions, const Variable& targets) {
        // Combine BCE and Dice loss
        auto bce = BCEWithLogitsLoss()(predictions, targets);
        auto dice = DiceLoss()(sigmoid(predictions), targets);

        return alpha_ * bce + (1 - alpha_) * dice;
    }

private:
    float alpha_ = 0.5f;  // Balance between BCE and Dice
};
```

**Multi-Class Segmentation:**
```cpp
class UNetMultiClassLoss {
public:
    Variable forward(const Variable& predictions, const Tensor& targets) {
        // predictions: [N, C, H, W] logits
        // targets: [N, H, W] class indices

        // Reshape to [N*H*W, C] and [N*H*W]
        int N = predictions.size(0), C = predictions.size(1);
        int H = predictions.size(2), W = predictions.size(3);

        auto pred_flat = predictions.permute({0, 2, 3, 1}).reshape({N * H * W, C});
        auto target_flat = targets.reshape({N * H * W});

        return CrossEntropyLoss()(pred_flat, target_flat);
    }
};
```

### 4.7 Input/Output Formats

**Input:**
- Image: `[N, 1, H, W]` (grayscale) or `[N, 3, H, W]` (RGB)
- Typical sizes: 256×256, 512×512, or 572×572 (original paper)

**Output:**
- Binary: `[N, 1, H, W]` probabilities (after sigmoid)
- Multi-class: `[N, C, H, W]` logits or probabilities

**Post-Processing:**
- Threshold at 0.5 for binary
- Argmax over channel dimension for multi-class

---

## 5. DeepLab v3+

### 5.1 Architecture Overview

**Encoder-Decoder with Atrous Convolution:**
```
Input Image
    ↓
Encoder (ResNet/Xception + ASPP)
    ├─→ Backbone (stride=16)
    ├─→ Atrous Spatial Pyramid Pooling (ASPP)
    └─→ Encoded Features
            ↓
Decoder
    ├─→ Upsample 4× (bilinear)
    ├─→ Concatenate with low-level features
    ├─→ Refine with 3×3 convolutions
    └─→ Upsample 4× (bilinear)
            ↓
        Segmentation Map
```

### 5.2 Atrous (Dilated) Convolution

**Purpose:** Increase receptive field without reducing resolution

**Implementation:**
```cpp
class AtrousConv2d : public Module {
public:
    AtrousConv2d(int in_channels, int out_channels, int kernel_size,
                 int dilation = 1, int padding = 0) {
        // Standard conv2d with dilation parameter
        // Effective receptive field = kernel_size + (kernel_size - 1) * (dilation - 1)
        conv = Conv2d(in_channels, out_channels, kernel_size, 1, padding, dilation);
    }

    Variable forward(const Variable& x) {
        return conv(x);
    }
};
```

**Dilation Mechanics:**
```cpp
// For standard 3×3 convolution (dilation=1):
// Receptive field: 3×3
// Weight positions: (0,0), (0,1), (0,2), (1,0), ..., (2,2)

// For dilated 3×3 convolution (dilation=2):
// Receptive field: 5×5
// Weight positions: (0,0), (0,2), (0,4), (2,0), ..., (4,4)
// Gaps of 1 pixel between weight positions

// For dilation=3:
// Receptive field: 7×7
// Gaps of 2 pixels between weights
```

**Padding Calculation:**
```cpp
int calculate_atrous_padding(int kernel_size, int dilation) {
    // To maintain spatial dimensions with stride=1
    return dilation * (kernel_size - 1) / 2;
}

// Example: kernel=3, dilation=6
// padding = 6 * (3 - 1) / 2 = 6
```

### 5.3 Atrous Spatial Pyramid Pooling (ASPP)

**Multi-Scale Context Aggregation:**
```cpp
class ASPP : public Module {
public:
    ASPP(int in_channels, int out_channels = 256) {
        // 1×1 convolution
        conv1x1 = nn::Sequential(
            Conv2d(in_channels, out_channels, 1),
            BatchNorm2d(out_channels),
            nn::ReLU()
        );

        // 3×3 atrous convolutions with different rates
        atrous_conv1 = nn::Sequential(
            AtrousConv2d(in_channels, out_channels, 3, 6, 6),
            BatchNorm2d(out_channels),
            nn::ReLU()
        );

        atrous_conv2 = nn::Sequential(
            AtrousConv2d(in_channels, out_channels, 3, 12, 12),
            BatchNorm2d(out_channels),
            nn::ReLU()
        );

        atrous_conv3 = nn::Sequential(
            AtrousConv2d(in_channels, out_channels, 3, 18, 18),
            BatchNorm2d(out_channels),
            nn::ReLU()
        );

        // Global average pooling branch
        global_pool = AdaptiveAvgPool2d(1);
        global_conv = nn::Sequential(
            Conv2d(in_channels, out_channels, 1),
            BatchNorm2d(out_channels),
            nn::ReLU()
        );

        // Fusion: 1×1 conv to reduce concatenated features
        project = nn::Sequential(
            Conv2d(out_channels * 5, out_channels, 1),
            BatchNorm2d(out_channels),
            nn::ReLU(),
            nn::Dropout(0.5)
        );
    }

    Variable forward(const Variable& x) {
        int h = x.size(2), w = x.size(3);

        // Five parallel branches
        auto feat1 = conv1x1(x);                    // [N, 256, H, W]
        auto feat2 = atrous_conv1(x);               // [N, 256, H, W]
        auto feat3 = atrous_conv2(x);               // [N, 256, H, W]
        auto feat4 = atrous_conv3(x);               // [N, 256, H, W]

        auto feat5 = global_pool(x);                // [N, C, 1, 1]
        feat5 = global_conv(feat5);                 // [N, 256, 1, 1]
        feat5 = upsample(feat5, h, w, "bilinear"); // [N, 256, H, W]

        // Concatenate all branches
        auto concat = cat({feat1, feat2, feat3, feat4, feat5}, 1);  // [N, 1280, H, W]

        // Project to output channels
        return project(concat);  // [N, 256, H, W]
    }
};
```

### 5.4 DeepLab v3+ Encoder

```cpp
class DeepLabV3PlusEncoder : public Module {
public:
    DeepLabV3PlusEncoder(int in_channels = 3) {
        // Backbone: ResNet-101 or Xception (output stride = 16)
        // Modified to use atrous convolutions in later layers
        backbone = ResNet101(output_stride=16);

        // ASPP on top of backbone
        aspp = ASPP(2048, 256);
    }

    std::pair<Variable, Variable> forward(const Variable& x) {
        // Get features from backbone
        auto low_level_feat = backbone.layer1(x);  // 1/4 resolution, 256 channels
        auto high_level_feat = backbone.forward_to_end(low_level_feat);  // 1/16 resolution, 2048 channels

        // Apply ASPP
        auto aspp_feat = aspp(high_level_feat);  // [N, 256, H/16, W/16]

        return {aspp_feat, low_level_feat};
    }
};
```

### 5.5 DeepLab v3+ Decoder

```cpp
class DeepLabV3PlusDecoder : public Module {
public:
    DeepLabV3PlusDecoder(int num_classes, int low_level_channels = 256) {
        // Reduce low-level feature channels
        low_level_reduce = nn::Sequential(
            Conv2d(low_level_channels, 48, 1),
            BatchNorm2d(48),
            nn::ReLU()
        );

        // Refinement convolutions after concatenation
        refine_conv = nn::Sequential(
            Conv2d(256 + 48, 256, 3, 1, 1),
            BatchNorm2d(256),
            nn::ReLU(),
            Conv2d(256, 256, 3, 1, 1),
            BatchNorm2d(256),
            nn::ReLU()
        );

        // Final classification layer
        classifier = Conv2d(256, num_classes, 1);
    }

    Variable forward(const Variable& aspp_feat, const Variable& low_level_feat) {
        // Upsample ASPP features by 4×
        int low_h = low_level_feat.size(2);
        int low_w = low_level_feat.size(3);
        auto upsampled = upsample(aspp_feat, low_h, low_w, "bilinear");  // [N, 256, H/4, W/4]

        // Reduce low-level feature channels
        auto low_reduced = low_level_reduce(low_level_feat);  // [N, 48, H/4, W/4]

        // Concatenate
        auto concat = cat({upsampled, low_reduced}, 1);  // [N, 304, H/4, W/4]

        // Refine
        auto refined = refine_conv(concat);  // [N, 256, H/4, W/4]

        // Classify
        auto logits = classifier(refined);  // [N, num_classes, H/4, W/4]

        // Final upsample to input resolution
        int out_h = low_h * 4;
        int out_w = low_w * 4;
        auto output = upsample(logits, out_h, out_w, "bilinear");

        return output;
    }
};
```

### 5.6 Complete DeepLab v3+ Model

```cpp
class DeepLabV3Plus : public Module {
public:
    DeepLabV3Plus(int num_classes = 21, int in_channels = 3) {
        encoder = DeepLabV3PlusEncoder(in_channels);
        decoder = DeepLabV3PlusDecoder(num_classes);
    }

    Variable forward(const Variable& x) {
        auto [aspp_feat, low_level_feat] = encoder.forward(x);
        auto output = decoder.forward(aspp_feat, low_level_feat);
        return output;  // [N, num_classes, H, W]
    }
};
```

### 5.7 Atrous Separable Convolution (Optimization)

**Depthwise Separable + Atrous:**
```cpp
class AtrousSeparableConv2d : public Module {
public:
    AtrousSeparableConv2d(int in_channels, int out_channels, int kernel_size,
                         int dilation = 1, int padding = 0) {
        // Depthwise atrous convolution
        depthwise = Conv2d(in_channels, in_channels, kernel_size, 1, padding,
                          dilation, in_channels);  // groups = in_channels

        // Pointwise convolution
        pointwise = Conv2d(in_channels, out_channels, 1);

        bn = BatchNorm2d(out_channels);
    }

    Variable forward(const Variable& x) {
        auto dw = depthwise(x);
        auto pw = pointwise(dw);
        return relu(bn(pw));
    }
};
```

### 5.8 Multi-Scale Context

**ASPP Dilation Rates:**
- Rate 1: Standard 1×1 convolution
- Rate 6: 3×3 conv with dilation=6 (receptive field: 13×13)
- Rate 12: 3×3 conv with dilation=12 (receptive field: 25×25)
- Rate 18: 3×3 conv with dilation=18 (receptive field: 37×37)
- Global: Adaptive average pooling (entire feature map)

### 5.9 Input/Output Formats

**Input:**
- Image: `[N, 3, H, W]` (typically 513×513 or 1024×2048)
- Preprocessing: Normalize to ImageNet mean/std

**Output:**
- Segmentation: `[N, num_classes, H, W]` logits
- Apply softmax for probabilities
- Argmax for class prediction

**Loss Function:**
```cpp
class DeepLabLoss {
public:
    Variable forward(const Variable& predictions, const Tensor& targets) {
        // Cross entropy with optional class weighting
        return CrossEntropyLoss()(predictions, targets);
    }
};
```

---

## 6. New Components Required for Tenzor

Based on analysis of existing Tenzor code in `/home/lee/Projects/Tenzor/include/tenzor/`, the following components need to be implemented:

### 6.1 ROI Operations

**STATUS:** ❌ Not implemented

**Required:**
```cpp
// File: include/tenzor/ops/roi_ops.hpp

namespace tenzor {
namespace nn {

class ROIPooling : public Module {
public:
    ROIPooling(int output_h, int output_w, float spatial_scale);
    Variable forward(const Variable& features, const Tensor& rois) override;
};

class ROIAlign : public Module {
public:
    ROIAlign(int output_h, int output_w, float spatial_scale, int sampling_ratio = 2);
    Variable forward(const Variable& features, const Tensor& rois) override;
};

} // namespace nn
} // namespace tenzor
```

**Backend Support Needed:**
- CPU implementation (reference)
- CUDA kernel for GPU acceleration
- Backward pass for both operations

### 6.2 NMS and IoU Operations

**STATUS:** ❌ Not implemented

**Required:**
```cpp
// File: include/tenzor/ops/detection_ops.hpp

namespace tenzor {

// Non-Maximum Suppression
auto nms(const Tensor& boxes, const Tensor& scores, float iou_threshold)
    -> Tensor;  // Returns indices of kept boxes

// IoU computation
auto compute_iou(const Tensor& boxes1, const Tensor& boxes2)
    -> Tensor;  // Returns [N, M] IoU matrix

// GIoU for loss
auto compute_giou(const Tensor& boxes1, const Tensor& boxes2)
    -> Tensor;

// CIoU for YOLO
auto compute_ciou(const Tensor& boxes1, const Tensor& boxes2)
    -> Tensor;

} // namespace tenzor
```

### 6.3 Anchor Generation

**STATUS:** ❌ Not implemented

**Required:**
```cpp
// File: include/tenzor/ops/anchor_generator.hpp

namespace tenzor {
namespace nn {

class AnchorGenerator {
public:
    AnchorGenerator(std::vector<float> scales, std::vector<float> aspect_ratios, int stride = 16);

    // Generate anchors for feature map of size (H, W)
    Tensor generate(int feat_height, int feat_width);

    // Generate multi-level anchors for FPN
    std::vector<Tensor> generate_fpn(const std::vector<std::pair<int, int>>& sizes,
                                     const std::vector<int>& strides);
};

} // namespace nn
} // namespace tenzor
```

### 6.4 Atrous/Dilated Convolution

**STATUS:** ⚠️ Partially implemented (Conv2d has dilation parameter in header but needs verification)

**Action Required:**
- Verify Conv2d dilation is fully implemented in all backends
- Ensure dilation works correctly for rates > 1
- Add specialized AtrousSeparableConv2d layer

**Check in:**
- `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/conv.hpp` (line 59: dilation parameter exists)
- Backend implementations in `src/backends/*/kernels/conv*.cpp`

### 6.5 Upsampling and Interpolation

**STATUS:** ❌ Not fully implemented

**Existing:**
- ConvTranspose2d ✓ (in conv.hpp)

**Missing:**
```cpp
// File: include/tenzor/ops/interpolate.hpp

namespace tenzor {

// Bilinear interpolation
auto upsample_bilinear(const Tensor& input, int output_h, int output_w)
    -> Tensor;

// Nearest neighbor interpolation
auto upsample_nearest(const Tensor& input, int output_h, int output_w)
    -> Tensor;

// Bicubic interpolation (optional)
auto upsample_bicubic(const Tensor& input, int output_h, int output_w)
    -> Tensor;

// General interpolate with mode selection
auto interpolate(const Tensor& input,
                std::optional<std::pair<int, int>> size,
                std::optional<std::pair<float, float>> scale_factor,
                const std::string& mode = "bilinear",  // "nearest", "bilinear", "bicubic"
                bool align_corners = false)
    -> Tensor;

} // namespace tenzor
```

### 6.6 Loss Functions

**STATUS:** ✓ Mostly complete

**Existing in `/home/lee/Projects/Tenzor/include/tenzor/nn/loss/losses.hpp`:**
- BCELoss ✓
- BCEWithLogitsLoss ✓
- CrossEntropyLoss ✓
- SmoothL1Loss ✓
- DiceLoss ✓
- FocalLoss ✓

**Missing:**
```cpp
// File: include/tenzor/nn/loss/detection_losses.hpp

namespace tenzor {
namespace nn {

// IoU-based losses for object detection
class IoULoss {
public:
    enum class Type { IoU, GIoU, DIoU, CIoU };
    IoULoss(Type type = Type::CIoU);
    Variable forward(const Variable& pred_boxes, const Variable& target_boxes);
};

} // namespace nn
} // namespace tenzor
```

### 6.7 Specialized Layers

**Required for Model Implementations:**

```cpp
// File: include/tenzor/nn/layers/detection_layers.hpp

namespace tenzor {
namespace nn {

// ASPP module for DeepLab
class ASPP : public Module {
public:
    ASPP(int in_channels, int out_channels = 256,
         std::vector<int> atrous_rates = {6, 12, 18});
    Variable forward(const Variable& x) override;
};

// FPN (Feature Pyramid Network)
class FPN : public Module {
public:
    FPN(std::vector<int> in_channels, int out_channels = 256);
    std::vector<Variable> forward(const std::vector<Variable>& features) override;
};

} // namespace nn
} // namespace tenzor
```

### 6.8 Tensor Operations

**Check Existing in `/home/lee/Projects/Tenzor/include/tenzor/ops/`:**

**Available:**
- reshape, view ✓ (transform.hpp)
- transpose, permute ✓ (transform.hpp)
- cat, stack ✓ (transform.hpp)
- split, chunk ✓ (transform.hpp)
- expand ✓ (transform.hpp)
- topk, sort ✓ (advanced.hpp)

**May Need:**
- Efficient batch indexing for ROI operations
- Gather/scatter operations for detection post-processing

### 6.9 Summary Table

| Component | Status | Priority | Complexity |
|-----------|--------|----------|------------|
| ROI Pooling | ❌ Missing | High | Medium |
| ROI Align | ❌ Missing | High | High |
| NMS | ❌ Missing | High | Medium |
| IoU Computation | ❌ Missing | High | Low |
| Anchor Generation | ❌ Missing | High | Low |
| Atrous Conv | ⚠️ Check | High | Low |
| Bilinear Upsample | ❌ Missing | High | Medium |
| IoU-based Losses | ❌ Missing | Medium | Low |
| ASPP Layer | ❌ Missing | Medium | Low |
| FPN Layer | ❌ Missing | Medium | Medium |

---

## 7. Implementation Priorities

### 7.1 Phase 1: Foundation (Week 1-2)

**Core Operations:**
1. **IoU Computation** - Fundamental for all detection models
2. **NMS** - Essential post-processing
3. **Bilinear Interpolation** - Needed for upsampling and ROI Align
4. **Verify Atrous Convolution** - Check existing implementation

**Deliverables:**
- `include/tenzor/ops/detection_ops.hpp`
- `include/tenzor/ops/interpolate.hpp`
- CPU implementations
- Unit tests

### 7.2 Phase 2: ROI Operations (Week 3-4)

**Implementation Order:**
1. **ROI Pooling** - Simpler, good starting point
2. **ROI Align** - More complex, builds on bilinear interpolation
3. **Anchor Generation** - Utility for training

**Deliverables:**
- `include/tenzor/ops/roi_ops.hpp`
- `include/tenzor/ops/anchor_generator.hpp`
- CUDA kernels for GPU acceleration
- Backward passes for training

### 7.3 Phase 3: Model-Specific Layers (Week 5-6)

**High-Level Components:**
1. **ASPP** - For DeepLab
2. **FPN** - For Faster R-CNN, Mask R-CNN
3. **IoU-based Losses** - For YOLO training

**Deliverables:**
- `include/tenzor/nn/layers/detection_layers.hpp`
- `include/tenzor/nn/loss/detection_losses.hpp`

### 7.4 Phase 4: Model Implementations (Week 7-10)

**Order:**
1. **U-Net** - Simplest, no ROI operations
2. **DeepLab v3+** - Tests ASPP and atrous convolutions
3. **Faster R-CNN** - Tests ROI operations and NMS
4. **YOLO v3** - Tests anchor generation and multi-scale detection
5. **Mask R-CNN** - Most complex, combines everything

**Deliverables:**
- `include/tenzor/models/unet.hpp`
- `include/tenzor/models/deeplabv3plus.hpp`
- `include/tenzor/models/faster_rcnn.hpp`
- `include/tenzor/models/yolov3.hpp`
- `include/tenzor/models/mask_rcnn.hpp`

### 7.5 Phase 5: Testing and Optimization (Week 11-12)

**Tasks:**
1. Comprehensive unit tests
2. Integration tests with pretrained weights
3. Benchmark performance (CPU/GPU)
4. Optimize critical paths (CUDA kernels)
5. Documentation and examples

---

## 8. References

### 8.1 Papers

1. **Faster R-CNN:** Ren et al. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (NIPS 2015)
2. **Mask R-CNN:** He et al. "Mask R-CNN" (ICCV 2017)
3. **YOLOv3:** Redmon & Farhadi "YOLOv3: An Incremental Improvement" (arXiv 2018)
4. **YOLOv5:** Ultralytics (https://github.com/ultralytics/yolov5)
5. **U-Net:** Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
6. **DeepLab v3+:** Chen et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (ECCV 2018)

### 8.2 Implementation References

- **Detectron2** (Facebook): https://github.com/facebookresearch/detectron2
- **MMDetection** (OpenMMLab): https://github.com/open-mmlab/mmdetection
- **Ultralytics YOLOv5**: https://github.com/ultralytics/yolov5
- **PyTorch Vision**: https://github.com/pytorch/vision (torchvision.ops)
- **TensorFlow Object Detection API**: https://github.com/tensorflow/models/tree/master/research/object_detection

### 8.3 Key Specifications

**Faster R-CNN:**
- Anchors: 9 per position (3 scales × 3 ratios)
- ROI size: 7×7 for classification
- NMS threshold: 0.7 (proposals), 0.3 (detections)
- Training: 2-stage (RPN then detector)

**YOLO v3:**
- Input: 416×416 (multiples of 32)
- Grids: 13×13, 26×26, 52×52
- Anchors: 9 total (3 per scale)
- Loss: MSE + BCE + BCE

**YOLOv5:**
- Input: 640×640
- Anchors: Auto-learned
- Loss: CIoU + BCE + BCE
- Backbone: CSPDarknet

**Mask R-CNN:**
- Extends Faster R-CNN
- Mask size: 28×28
- ROI Align: 14×14 for masks
- Loss: All Faster R-CNN losses + mask BCE

**U-Net:**
- Encoder: 4 downsample blocks
- Decoder: 4 upsample blocks
- Skip connections: Concatenation
- Loss: BCE + Dice

**DeepLab v3+:**
- Backbone: ResNet-101
- Output stride: 16
- ASPP rates: 1, 6, 12, 18
- Decoder: Simple 4× upsample
- Loss: Cross entropy

---

## Appendix A: Component Dependencies

```
Detection Models Component Dependencies:

U-Net
├── ConvTranspose2d ✓
├── Bilinear Upsample ❌
└── Dice Loss ✓

DeepLab v3+
├── Atrous Convolution ⚠️
├── ASPP ❌
├── Bilinear Upsample ❌
└── Cross Entropy ✓

Faster R-CNN
├── Anchor Generator ❌
├── RPN ❌
├── ROI Pooling ❌
├── NMS ❌
├── IoU ❌
└── Smooth L1 Loss ✓

YOLO v3/v5
├── Anchor Generator ❌
├── Multi-scale Detection ❌
├── NMS ❌
├── IoU/CIoU ❌
└── BCE Loss ✓

Mask R-CNN
├── All Faster R-CNN components
├── ROI Align ❌
├── Mask Head ❌
└── Dice Loss ✓
```

---

**End of Specification Document**
