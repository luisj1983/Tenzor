# Mask R-CNN Quick Reference

## Installation

```cpp
#include "tenzor/models/mask_rcnn.hpp"
#include "tenzor/nn/detection/mask_head.hpp"
```

---

## Creating Models

### ResNet-50-FPN (Standard)
```cpp
auto model = models::mask_rcnn_resnet50_fpn(80, false);  // 80 classes, no pretrained
```

### ResNet-101-FPN (High Accuracy)
```cpp
auto model = models::mask_rcnn_resnet101_fpn(80, true);  // with pretrained weights
```

---

## Inference

### Basic Usage
```cpp
// Load and prepare image
auto image = load_image("image.jpg");  // (1, 3, H, W)

// Set to eval mode
model->eval();

// Run inference
auto [boxes, labels, scores, masks] = model->forward_test(image);

// boxes: (num_detections, 4) as (x1, y1, x2, y2)
// labels: (num_detections,) class indices
// scores: (num_detections,) confidence [0, 1]
// masks: (num_detections, H, W) binary {0, 1}
```

### Process Results
```cpp
for (int i = 0; i < boxes.size(0); ++i) {
    auto x1 = boxes.at<float>(i, 0);
    auto y1 = boxes.at<float>(i, 1);
    auto x2 = boxes.at<float>(i, 2);
    auto y2 = boxes.at<float>(i, 3);
    auto label = labels.at<int64_t>(i);
    auto score = scores.at<float>(i);
    auto mask = masks[i];  // (H, W) binary mask

    // Use detection...
}
```

---

## Training

### Setup
```cpp
#include "tenzor/nn/optim/sgd.hpp"

// Create model
auto model = models::mask_rcnn_resnet50_fpn(num_classes, false);
model->train();

// Optimizer
auto optimizer = std::make_shared<nn::optim::SGD>(
    model->parameters(),
    0.02,    // learning rate
    0.9,     // momentum
    0.0001   // weight decay
);
```

### Training Loop
```cpp
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& [images, gt_boxes, gt_labels, gt_masks] : dataloader) {
        // Forward pass
        auto [rpn_cls, rpn_bbox, roi_cls, roi_bbox, mask_loss] =
            model->forward_train(images, gt_boxes, gt_labels, gt_masks);

        // Total loss
        auto loss = rpn_cls + rpn_bbox + roi_cls + roi_bbox + mask_loss;

        // Backward pass
        optimizer->zero_grad();
        loss.backward();
        optimizer->step();
    }
}
```

---

## MaskHead Usage

### Standalone Mask Head
```cpp
#include "tenzor/nn/detection/mask_head.hpp"

// Create mask head
nn::detection::MaskHead mask_head(
    256,  // in_channels
    80,   // num_classes
    256,  // conv_dim (optional, default: 256)
    4,    // num_conv (optional, default: 4)
    28    // mask_size (optional, default: 28)
);

// Forward pass
auto roi_features = randn({100, 256, 14, 14});  // From ROI Align
auto mask_logits = mask_head.forward(roi_features);  // (100, 80, 28, 28)

// Apply sigmoid for probabilities
auto mask_probs = ops::sigmoid(mask_logits.tensor());
```

### Compute Mask Loss
```cpp
#include "tenzor/nn/detection/mask_head.hpp"

auto mask_logits = mask_head.forward(roi_features);  // (N, C, 28, 28)
auto mask_targets = get_ground_truth_masks();        // (N, 28, 28) in [0, 1]
auto class_labels = get_class_labels();              // (N,) class indices

auto loss = nn::detection::mask_loss(
    mask_logits,
    mask_targets,
    class_labels
);

loss.backward();
```

### Post-Process Masks
```cpp
auto masks_full_res = nn::detection::process_masks(
    mask_logits.tensor(),  // (num_detections, num_classes, 28, 28)
    boxes,                 // (num_detections, 4)
    class_labels,          // (num_detections,)
    image_height,          // original image height
    image_width,           // original image width
    0.5                    // threshold (optional, default: 0.5)
);
// Returns: (num_detections, image_height, image_width) binary masks
```

---

## Configuration

### Model Hyperparameters

```cpp
auto model = std::make_shared<MaskRCNN>(
    backbone,
    num_classes,
    800,    // min_size: minimum image dimension
    1333,   // max_size: maximum image dimension
    2000,   // rpn_pre_nms_top_n_train
    1000,   // rpn_pre_nms_top_n_test
    2000,   // rpn_post_nms_top_n_train
    1000,   // rpn_post_nms_top_n_test
    0.7,    // rpn_nms_thresh
    0.05,   // box_score_thresh
    0.5,    // box_nms_thresh
    100     // box_detections_per_img
);
```

### Default Values (COCO Settings)

| Parameter | Train | Test |
|-----------|-------|------|
| Image Size | min=800, max=1333 | same |
| RPN Proposals (pre-NMS) | 2000 | 1000 |
| RPN Proposals (post-NMS) | 2000 | 1000 |
| RPN NMS Threshold | 0.7 | 0.7 |
| Score Threshold | - | 0.05 |
| Detection NMS Threshold | - | 0.5 |
| Max Detections per Image | - | 100 |

---

## Input/Output Formats

### Training Input
```cpp
// Images: (batch_size, 3, H, W) in [0, 1], normalized
auto images = randn({2, 3, 800, 1200});

// Ground truth boxes: (batch_size, max_objects, 4) as (x1, y1, x2, y2)
auto gt_boxes = randn({2, 50, 4});

// Ground truth labels: (batch_size, max_objects) class indices [0, num_classes)
auto gt_labels = randint(0, 80, {2, 50});

// Ground truth masks: (batch_size, max_objects, H, W) binary {0, 1}
auto gt_masks = randint(0, 2, {2, 50, 800, 1200}).to(DType::Float32);
```

### Training Output
```cpp
auto [rpn_cls_loss, rpn_bbox_loss, roi_cls_loss, roi_bbox_loss, mask_loss] =
    model->forward_train(images, gt_boxes, gt_labels, gt_masks);

// All losses are scalars (Variable with shape [])
auto total_loss = rpn_cls_loss + rpn_bbox_loss +
                  roi_cls_loss + roi_bbox_loss + mask_loss;
```

### Inference Input
```cpp
// Images: (batch_size, 3, H, W) in [0, 1], normalized
auto images = randn({1, 3, 800, 1200});
```

### Inference Output
```cpp
auto [boxes, labels, scores, masks] = model->forward_test(images);

// boxes: (num_detections, 4) Float32, (x1, y1, x2, y2)
// labels: (num_detections,) Int64, class indices
// scores: (num_detections,) Float32, confidence in [0, 1]
// masks: (num_detections, H, W) UInt8, binary {0, 1}
```

---

## Common Patterns

### Filtering by Score
```cpp
auto [boxes, labels, scores, masks] = model->forward_test(image);

// Filter detections by score threshold
float threshold = 0.7;
std::vector<int64_t> keep_indices;

for (int64_t i = 0; i < scores.size(0); ++i) {
    if (scores.at<float>(i) >= threshold) {
        keep_indices.push_back(i);
    }
}

auto filtered_boxes = boxes.index_select(0, Tensor(keep_indices));
auto filtered_labels = labels.index_select(0, Tensor(keep_indices));
auto filtered_scores = scores.index_select(0, Tensor(keep_indices));
auto filtered_masks = masks.index_select(0, Tensor(keep_indices));
```

### Filtering by Class
```cpp
// Get all detections of class 0 (person in COCO)
int64_t target_class = 0;
std::vector<int64_t> class_indices;

for (int64_t i = 0; i < labels.size(0); ++i) {
    if (labels.at<int64_t>(i) == target_class) {
        class_indices.push_back(i);
    }
}

auto person_boxes = boxes.index_select(0, Tensor(class_indices));
auto person_masks = masks.index_select(0, Tensor(class_indices));
```

### Visualization
```cpp
#include "tenzor/ops/vision.hpp"

// Draw boxes and masks on image
for (int i = 0; i < boxes.size(0); ++i) {
    // Draw bounding box
    draw_box(image, boxes[i], labels[i], scores[i]);

    // Overlay mask with transparency
    auto mask = masks[i];  // (H, W)
    auto color = get_class_color(labels.at<int64_t>(i));
    overlay_mask(image, mask, color, alpha=0.5);
}

save_image("result.jpg", image);
```

---

## Performance Tips

### Inference Speed
1. **Use GPU:** `.to(Device::cuda())`
2. **Batch Size = 1:** Faster than batching for variable-size images
3. **Smaller Images:** Resize to 800×600 instead of 800×1200
4. **Fewer Proposals:** Reduce `rpn_post_nms_top_n_test` to 500

### Memory Usage
1. **Smaller Batch:** Use batch_size=1 or 2
2. **Gradient Checkpointing:** For very deep backbones
3. **Mixed Precision:** FP16 training (when available)
4. **Smaller Backbone:** Use ResNet-50 instead of ResNet-101

### Accuracy
1. **Larger Images:** Increase max_size to 1600
2. **More Proposals:** Increase `rpn_post_nms_top_n_test` to 2000
3. **Lower Threshold:** Decrease `box_score_thresh` to 0.01
4. **Multi-Scale Testing:** Test at multiple image scales

---

## Debugging

### Check Output Shapes
```cpp
auto [boxes, labels, scores, masks] = model->forward_test(image);

std::cout << "Boxes shape: " << boxes.shape() << "\n";      // (N, 4)
std::cout << "Labels shape: " << labels.shape() << "\n";    // (N,)
std::cout << "Scores shape: " << scores.shape() << "\n";    // (N,)
std::cout << "Masks shape: " << masks.shape() << "\n";      // (N, H, W)
std::cout << "Num detections: " << boxes.size(0) << "\n";
```

### Validate Loss Values
```cpp
auto [rpn_cls, rpn_bbox, roi_cls, roi_bbox, mask_loss] =
    model->forward_train(images, gt_boxes, gt_labels, gt_masks);

std::cout << "RPN cls loss: " << rpn_cls.item<float>() << "\n";
std::cout << "RPN bbox loss: " << rpn_bbox.item<float>() << "\n";
std::cout << "ROI cls loss: " << roi_cls.item<float>() << "\n";
std::cout << "ROI bbox loss: " << roi_bbox.item<float>() << "\n";
std::cout << "Mask loss: " << mask_loss.item<float>() << "\n";

// Expected ranges (approximate):
// RPN cls: 0.1 - 1.0
// RPN bbox: 0.01 - 0.1
// ROI cls: 0.1 - 2.0
// ROI bbox: 0.05 - 0.5
// Mask: 0.1 - 1.0
```

### Check Predictions
```cpp
auto [boxes, labels, scores, masks] = model->forward_test(image);

// Print first few detections
for (int i = 0; i < std::min(5, int(boxes.size(0))); ++i) {
    std::cout << "Detection " << i << ":\n"
              << "  Box: (" << boxes.at<float>(i, 0) << ", "
              << boxes.at<float>(i, 1) << ", "
              << boxes.at<float>(i, 2) << ", "
              << boxes.at<float>(i, 3) << ")\n"
              << "  Label: " << labels.at<int64_t>(i) << "\n"
              << "  Score: " << scores.at<float>(i) << "\n";
}
```

---

## Error Messages

### Common Issues

**"ROIAlign requires both features and rois"**
- Solution: Use `roi_align->forward(features, rois)` not `roi_align->forward(features)`

**"mask_loss: class label X out of range"**
- Solution: Ensure class labels are in [0, num_classes)

**"Cannot reshape tensor of size X to size Y"**
- Solution: Check input shapes match expected dimensions

**"Expected 4D tensor for Conv2d input"**
- Solution: Ensure ROI features have shape (N, C, H, W)

---

## COCO Class Names (0-79)

```cpp
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// Usage
auto label_id = labels.at<int64_t>(i);
std::cout << "Detected: " << COCO_CLASSES[label_id] << "\n";
```

---

## Further Reading

- **Full Documentation:** `/home/lee/Projects/Tenzor/docs/MASK_RCNN_IMPLEMENTATION.md`
- **Specification:** `/home/lee/Projects/Tenzor/docs/DETECTION_SEGMENTATION_SPEC.md`
- **Paper:** "Mask R-CNN" https://arxiv.org/abs/1703.06870
- **Detectron2:** https://github.com/facebookresearch/detectron2
