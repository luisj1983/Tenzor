# Detection Components Quick Reference

**Quick guide for using Tenzor's detection operations**

---

## 📦 Include Headers

```cpp
#include "tenzor/nn/detection/anchors.hpp"    // AnchorGenerator
#include "tenzor/nn/detection/roi_ops.hpp"    // ROIAlign
#include "tenzor/ops/detection.hpp"           // IoU, NMS, encoding
```

---

## 🎯 Anchor Generation

```cpp
using namespace tenzor::nn::detection;

// Create generator
AnchorGenerator anchors(
    {32.0f, 64.0f, 128.0f, 256.0f, 512.0f},  // Sizes in pixels
    {0.5f, 1.0f, 2.0f}                        // Aspect ratios (w/h)
);

// Generate for feature map
auto boxes = anchors.generate(
    38,   // Feature height
    38,   // Feature width
    16    // Stride (image_size / feature_size)
);
// Returns: (38*38*15, 4) tensor with (x1, y1, x2, y2)

// Number of anchors per location
int64_t k = anchors.num_anchors_per_location();  // 15 (5 sizes × 3 ratios)
```

---

## 📐 Box IoU Computation

```cpp
using namespace tenzor::ops;

auto boxes1 = randn({100, 4});  // (x1, y1, x2, y2) format
auto boxes2 = randn({50, 4});

// Standard IoU
auto iou = box_iou(boxes1, boxes2);  // (100, 50) matrix

// Advanced IoU variants
auto giou = box_iou(boxes1, boxes2, IoUType::GIoU);  // Generalized IoU
auto diou = box_iou(boxes1, boxes2, IoUType::DIoU);  // Distance IoU
auto ciou = box_iou(boxes1, boxes2, IoUType::CIoU);  // Complete IoU (best for YOLO)
```

---

## 🔄 Box Encoding/Decoding

```cpp
// Training: Encode ground truth relative to anchors
auto deltas = encode_boxes(
    gt_boxes,                           // Ground truth (N, 4)
    anchor_boxes,                       // Anchors (N, 4)
    {1.0, 1.0, 1.0, 1.0}               // Weights (optional)
);
// Returns: (N, 4) deltas (dx, dy, dw, dh)

// Inference: Decode predictions to boxes
auto pred_boxes = decode_boxes(
    predicted_deltas,                   // Model output (N, 4)
    anchor_boxes,                       // Same anchors (N, 4)
    {1.0, 1.0, 1.0, 1.0}               // Same weights
);
// Returns: (N, 4) boxes (x1, y1, x2, y2)
```

---

## 🎭 Non-Maximum Suppression

### Single-Class NMS
```cpp
auto keep_indices = nms(
    boxes,        // (N, 4) tensor
    scores,       // (N,) tensor
    0.5           // IoU threshold (0.5-0.7 typical)
);
// Returns: Indices of kept boxes (sorted by score)

// Get filtered boxes
auto filtered_boxes = boxes.index_select(0, keep_indices);
auto filtered_scores = scores.index_select(0, keep_indices);
```

### Multi-Class NMS
```cpp
auto [kept_boxes, kept_scores, kept_labels] = batched_nms(
    boxes,        // (N, 4) all boxes
    scores,       // (N, num_classes) scores per class
    0.5,          // IoU threshold
    0.05,         // Score threshold (filter low confidence)
    100           // Max boxes per class
);
// Returns: Filtered (boxes, scores, labels) tensors
```

---

## 🎯 ROI Align

### Basic Usage
```cpp
using namespace tenzor::nn::detection;

ROIAlign roi_align(
    7,              // Output height
    7,              // Output width
    1.0 / 16.0,    // Spatial scale (feature_size / image_size)
    2,              // Sampling ratio (0 = adaptive)
    true            // Aligned mode (PyTorch compatible)
);

auto features = randn({2, 256, 50, 50});  // Batch of feature maps
auto rois = tensor({                       // ROIs with batch indices
    {0.0f, 10.0f, 10.0f, 110.0f, 110.0f},  // batch_idx, x1, y1, x2, y2
    {1.0f, 20.0f, 20.0f, 120.0f, 120.0f}
});

auto aligned = roi_align.forward(
    Variable(features, true),  // Enable gradients
    rois
);
// Returns: (num_rois, 256, 7, 7) aligned features
```

### With Gradient
```cpp
// Forward pass
auto features_var = Variable(features, true);
auto aligned = roi_align.forward(features_var, rois);

// Backward pass
aligned.backward(grad_output);

// Get gradients
auto grad_features = features_var.grad();  // Same shape as features
```

---

## 🛠️ Utility Functions

### Clip Boxes to Image
```cpp
auto clipped = clip_boxes_to_image(
    boxes,      // (N, 4) tensor
    height,     // Image height
    width       // Image width
);
// Ensures all coords in [0, width) × [0, height)
```

### Remove Small Boxes
```cpp
auto keep = remove_small_boxes(
    boxes,      // (N, 4) tensor
    scores,     // (N,) tensor
    5.0         // Minimum width/height in pixels
);
// Returns: Indices of boxes to keep
```

---

## 🏗️ Faster R-CNN Example

```cpp
using namespace tenzor;
using namespace tenzor::nn::detection;
using namespace tenzor::ops;

// 1. Generate anchors
AnchorGenerator anchors({32, 64, 128, 256, 512}, {0.5, 1.0, 2.0});
auto anchor_boxes = anchors.generate(feat_h, feat_w, 16);

// 2. RPN predictions (your model)
auto rpn_scores = rpn_head.forward(features);  // Objectness
auto rpn_deltas = rpn_bbox_head.forward(features);  // Box deltas

// 3. Decode proposals
auto proposals = decode_boxes(rpn_deltas, anchor_boxes);

// 4. Clip to image
proposals = clip_boxes_to_image(proposals, img_h, img_w);

// 5. Remove small boxes
auto keep = remove_small_boxes(proposals, rpn_scores, 16.0);
proposals = proposals.index_select(0, keep);
rpn_scores = rpn_scores.index_select(0, keep);

// 6. NMS on proposals
auto keep_nms = nms(proposals, rpn_scores, 0.7);
proposals = proposals.index_select(0, keep_nms);

// 7. ROI Align
ROIAlign roi_align(7, 7, 1.0/16.0, 2);
auto roi_features = roi_align.forward(features, proposals);

// 8. Detection head (your model)
auto det_scores = det_head.forward(roi_features);
auto det_deltas = det_bbox_head.forward(roi_features);

// 9. Final predictions
auto final_boxes = decode_boxes(det_deltas, proposals);
auto [boxes, scores, labels] = batched_nms(
    final_boxes, det_scores, 0.5, 0.05, 100
);
```

---

## 🎨 Mask R-CNN Extension

```cpp
// Same as Faster R-CNN, plus:

// 10. Mask ROI Align (higher resolution)
ROIAlign mask_roi_align(14, 14, 1.0/16.0, 2);
auto mask_features = mask_roi_align.forward(features, final_boxes);

// 11. Mask head (your model)
auto mask_logits = mask_head.forward(mask_features);  // (K, num_classes, 28, 28)

// 12. Select masks for predicted classes
auto masks = select_class_specific_masks(mask_logits, labels);
```

---

## ⚡ YOLO Example

```cpp
// 1. YOLO anchors (per grid cell)
AnchorGenerator yolo_anchors(
    {10, 13, 16, 30, 33, 23},  // COCO anchors for 416×416 input
    {1.0}                       // No aspect ratio variation
);

// 2. Grid predictions (your model)
auto grid_preds = yolo_head.forward(features);  // (N, 3, 13, 13, 85)

// 3. Decode grid to boxes
auto boxes = decode_yolo_grid(grid_preds, yolo_anchors, stride=32);

// 4. Get scores and classes
auto scores = grid_preds.slice(-1, 4, 5).sigmoid();  // Objectness
auto class_scores = grid_preds.slice(-1, 5, 85).sigmoid();  // Classes

// 5. Filter by confidence
auto mask = scores > 0.25;
boxes = boxes.masked_select(mask);
scores = scores.masked_select(mask);
class_scores = class_scores.masked_select(mask);

// 6. Multi-class NMS
auto [final_boxes, final_scores, final_labels] = batched_nms(
    boxes, class_scores, 0.45, 0.25, 100
);
```

---

## 🧪 Testing Your Code

```cpp
#include <gtest/gtest.h>

TEST(MyDetectionTest, AnchorTest) {
    AnchorGenerator anchors({32.0f}, {1.0f});
    auto boxes = anchors.generate(2, 2, 16);

    EXPECT_EQ(boxes.size(0), 4);   // 2×2×1 = 4 anchors
    EXPECT_EQ(boxes.size(1), 4);   // (x1, y1, x2, y2)
}

TEST(MyDetectionTest, NMSTest) {
    auto boxes = tensor({
        {0.0f, 0.0f, 10.0f, 10.0f},
        {1.0f, 1.0f, 11.0f, 11.0f}
    });
    auto scores = tensor({0.9f, 0.8f});

    auto keep = nms(boxes, scores, 0.5);
    EXPECT_EQ(keep.size(0), 1);  // Should keep only highest score
}
```

---

## 💡 Tips & Best Practices

### Anchor Generation
- **Cache anchors:** Generate once per feature map size, reuse across batches
- **Multi-scale:** Use FPN with different anchor sizes per level
- **Dataset-specific:** Use K-means clustering on your dataset for optimal anchors

### NMS
- **IoU threshold:** 0.5-0.7 for proposals, 0.3-0.5 for final detections
- **Score threshold:** 0.05 typical, adjust based on precision/recall needs
- **Max detections:** 100-300 typical, more for dense scenes

### ROI Align
- **Sampling ratio:** 2 is good balance of speed/quality, 0 for adaptive
- **Aligned mode:** Use `aligned=true` for consistency with PyTorch
- **Output size:** 7×7 for classification, 14×14 for masks

### Box Encoding
- **Weights:** Start with [1, 1, 1, 1], adjust if boxes don't converge
- **Numerical stability:** epsilon=1e-7 prevents log(0) and division by zero

---

## 🐛 Common Issues

### Issue: NMS removes all boxes
**Solution:** Lower IoU threshold or check if boxes actually overlap

### Issue: ROIAlign returns NaN
**Solution:** Ensure ROI coordinates are valid (x2 > x1, y2 > y1)

### Issue: Anchors outside image
**Solution:** Clip anchors or adjust anchor sizes for your input resolution

### Issue: Gradients not flowing through ROIAlign
**Solution:** Ensure features have `requires_grad=true`

---

## 📚 API Reference

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `box_iou` | (N,4), (M,4) | (N,M) | Pairwise IoU |
| `nms` | (N,4), (N,) | (K,) | Filter overlaps |
| `encode_boxes` | (N,4), (N,4) | (N,4) | Boxes → deltas |
| `decode_boxes` | (N,4), (N,4) | (N,4) | Deltas → boxes |
| `batched_nms` | (N,4), (N,C) | 3 tensors | Multi-class NMS |
| `roi_align.forward` | (B,C,H,W), (K,5) | (K,C,h,w) | Extract ROIs |

---

**For full details, see:** `/home/lee/Projects/Tenzor/docs/DETECTION_COMPONENTS_IMPLEMENTATION.md`
