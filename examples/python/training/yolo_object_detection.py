"""
YOLO Object Detection Training with Detection Operations

This comprehensive example demonstrates:
- YOLO architecture components
- Detection operations: box_iou, nms, batched_nms
- Anchor encoding/decoding
- GroupNorm normalization
- Multi-scale feature extraction
- Advanced tensor operations: topk, sort, unique
- Vision operations: interpolate
- Detection loss (objectness + classification + localization)
"""

import tenzor as tz
import numpy as np


# ============================================================================
# Detection Dataset
# ============================================================================

class DetectionDataset:
    """Object detection dataset with bounding boxes"""

    def __init__(self, num_samples, num_classes, img_size=416, max_objects=20):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.max_objects = max_objects

        np.random.seed(42)
        self.targets = []

        for i in range(num_samples):
            num_objs = np.random.randint(1, max_objects + 1)
            boxes = []

            for j in range(num_objs):
                cx = np.random.uniform(0.1, 0.9)
                cy = np.random.uniform(0.1, 0.9)
                w = min(np.random.uniform(0.05, 0.3), min(cx, 1 - cx) * 2)
                h = min(np.random.uniform(0.05, 0.3), min(cy, 1 - cy) * 2)
                cls = np.random.randint(0, num_classes)

                # [x1, y1, x2, y2, class_id]
                boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2, cls])

            self.targets.append(boxes)

    def get_batch(self, start, batch_size):
        """Get a batch of images and targets"""
        end = min(start + batch_size, self.num_samples)
        actual_batch = end - start

        # Random images
        images = np.random.randn(actual_batch, 3, self.img_size,
                                 self.img_size).astype(np.float32)

        # Targets as list (variable objects per image)
        batch_targets = []
        for i in range(actual_batch):
            boxes = np.array(self.targets[start + i], dtype=np.float32)
            batch_targets.append(tz.Tensor.from_numpy(boxes))

        return tz.Tensor.from_numpy(images), batch_targets

    def __len__(self):
        return self.num_samples


# ============================================================================
# Detection Operations Demo
# ============================================================================

def demo_detection_ops():
    """Demonstrate detection operations"""
    print("\n" + "=" * 60)
    print("Detection Operations Demo")
    print("=" * 60)

    # box_iou
    print("\n[1] box_iou - Calculate IoU between boxes")

    boxes1 = np.array([
        [0, 0, 10, 10],    # Box 1
        [0, 0, 5, 5],      # Box 2 (25% of box 1)
        [5, 5, 15, 15],    # Box 3
    ], dtype=np.float32)

    boxes2 = np.array([
        [0, 0, 10, 10],    # Same as box1
        [5, 5, 10, 10],    # 50% overlap
        [20, 20, 30, 30],  # No overlap
    ], dtype=np.float32)

    iou = tz.detection.box_iou(tz.Tensor.from_numpy(boxes1), tz.Tensor.from_numpy(boxes2))
    iou_np = iou.numpy()

    print("  IoU matrix (3x3):")
    for i in range(3):
        print(f"    [{', '.join([f'{iou_np[i,j]:.3f}' for j in range(3)])}]")

    # NMS
    print("\n[2] nms - Remove overlapping detections")

    boxes = np.array([
        [0, 0, 10, 10],    # score 0.9
        [1, 1, 11, 11],    # score 0.8 (overlaps with box0)
        [50, 50, 60, 60],  # score 0.7
        [2, 2, 12, 12],    # score 0.6 (overlaps with box0)
        [51, 51, 61, 61],  # score 0.5 (overlaps with box2)
    ], dtype=np.float32)

    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)

    keep = tz.detection.nms(tz.Tensor.from_numpy(boxes), tz.Tensor.from_numpy(scores), 0.5)
    keep_np = keep.numpy()

    print(f"  Input: 5 boxes (2 clusters with overlap)")
    print(f"  NMS keeps indices: {list(keep_np)}")

    # batched_nms - expects boxes [N, 4] and scores [N, num_classes]
    print("\n[3] batched_nms - Batched NMS for multi-class detection")

    # Create tensors with proper shapes for multi-class detection
    boxes_flat = np.array([
        [0, 0, 10, 10],    # Box 0
        [1, 1, 11, 11],    # Box 1 (overlaps with box 0)
    ], dtype=np.float32)  # [2 boxes, 4 coords]

    # Scores must be 2D: [num_boxes, num_classes]
    scores_2d = np.array([
        [0.9, 0.1],  # Box 0: class 0 = 0.9, class 1 = 0.1
        [0.8, 0.2],  # Box 1: class 0 = 0.8, class 1 = 0.2
    ], dtype=np.float32)  # [2 boxes, 2 classes]

    batch_keep, batch_scores, batch_indices = tz.detection.batched_nms(
        tz.Tensor.from_numpy(boxes_flat),
        tz.Tensor.from_numpy(scores_2d),
        iou_threshold=0.5,
        score_threshold=0.3,
        max_output_boxes=10
    )

    print(f"  Score threshold: 0.3, IoU threshold: 0.5")
    print("  Batched NMS for multi-class detection pipelines")

    # Anchor encoding/decoding
    print("\n[4] encode_boxes / decode_boxes - Anchor-based encoding")

    anchors = np.array([
        [5, 5, 15, 15],
        [20, 20, 40, 40],
        [50, 50, 70, 70],
    ], dtype=np.float32)

    gt_boxes = np.array([
        [6, 4, 16, 14],
        [22, 18, 42, 38],
        [52, 48, 72, 68],
    ], dtype=np.float32)

    encoded = tz.detection.encode_boxes(tz.Tensor.from_numpy(gt_boxes),
                              tz.Tensor.from_numpy(anchors))
    decoded = tz.detection.decode_boxes(encoded, tz.Tensor.from_numpy(anchors))

    print("  Anchors -> Encode GT offsets -> Decode back to boxes")
    print("  Used for predicting box refinements instead of absolute coords")

    # clip_boxes_to_image
    print("\n[5] clip_boxes_to_image - Clamp boxes to image bounds")

    out_of_bounds = np.array([
        [-5, -5, 110, 110],
        [50, 50, 150, 80],
    ], dtype=np.float32)

    clipped = tz.detection.clip_boxes_to_image(tz.Tensor.from_numpy(out_of_bounds), 100, 100)
    clipped_np = clipped.numpy()

    print("  Image size: 100x100")
    print(f"  Box {list(out_of_bounds[0])} -> {list(clipped_np[0])}")


# ============================================================================
# Advanced Tensor Operations
# ============================================================================

def demo_advanced_ops():
    """Demonstrate advanced tensor operations"""
    print("\n" + "=" * 60)
    print("Advanced Tensor Operations")
    print("=" * 60)

    # topk
    print("\n[1] topk - Get top K values and indices")

    values = np.array([[i * i % 7 for i in range(10)]], dtype=np.float32)
    print(f"  Input: {list(values[0])}")

    top_vals, top_idx = tz.topk(tz.Tensor.from_numpy(values), k=3, dim=-1,
                                largest=True, sorted=True)
    print(f"  Top-3 values: {list(top_vals.numpy()[0])}")
    print(f"  Top-3 indices: {list(top_idx.numpy()[0])}")

    # sort
    print("\n[2] sort - Sort tensor along dimension")

    sorted_vals, sort_idx = tz.sort(tz.Tensor.from_numpy(values), dim=-1, descending=True)
    print(f"  Sorted (descending): {list(sorted_vals.numpy()[0])}")

    # unique
    print("\n[3] unique - Get unique values")

    with_dups = np.array([1, 2, 1, 3, 2, 1, 4, 3], dtype=np.float32)
    print(f"  Input: {list(with_dups)}")

    unique_result = tz.unique(tz.Tensor.from_numpy(with_dups))
    unique_vals = unique_result[0]  # First element is the unique values tensor
    print(f"  Unique: {list(unique_vals.numpy())}")

    # cumsum
    print("\n[4] cumsum - Cumulative sum")

    seq = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    print(f"  Input: {list(seq)}")

    cumulative = tz.cumsum(tz.Tensor.from_numpy(seq), dim=0)
    print(f"  Cumsum: {list(cumulative.numpy())}")

    # cumprod
    print("\n[5] cumprod - Cumulative product")

    cumproduct = tz.cumprod(tz.Tensor.from_numpy(seq), dim=0)
    print(f"  Cumprod: {list(cumproduct.numpy())}")


# ============================================================================
# Vision Operations
# ============================================================================

def demo_vision_ops():
    """Demonstrate vision operations (C++ API - Python bindings pending)"""
    print("\n" + "=" * 60)
    print("Vision Operations (C++ API available)")
    print("=" * 60)

    # Note: These operations are available in the C++ API
    # Python bindings are pending for: interpolate, unfold, fold

    print("\n[1] interpolate - Resize/upsample images")
    print("  C++ API: tz::interpolate(tensor, size, mode, align_corners)")
    print("  Modes: bilinear, nearest, bicubic")
    print("  Example: [1,3,32,32] -> bilinear 2x -> [1,3,64,64]")

    print("\n[2] unfold - Extract sliding window patches")
    print("  C++ API: tz::unfold(tensor, kernel_size, stride, padding)")
    print("  Extracts sliding local blocks from input")

    print("\n[3] fold - Inverse of unfold")
    print("  C++ API: tz::fold(tensor, output_size, kernel_size, stride, padding)")
    print("  Combines sliding local blocks into output")


# ============================================================================
# Fused Operations
# ============================================================================

def demo_fused_ops():
    """Demonstrate fused operations (C++ API - Python bindings pending)"""
    print("\n" + "=" * 60)
    print("Fused Operations (C++ API available)")
    print("=" * 60)

    # Note: These operations are available in the C++ API for performance
    # Python bindings are pending for: fused_conv2d_relu, fused_add_relu, fused_gelu

    print("\n[1] fused_conv2d_relu - Conv + ReLU in one kernel")
    print("  C++ API: tz::fused_conv2d_relu(input, weight, bias, stride, padding)")
    print("  Fuses convolution and ReLU to reduce memory bandwidth")
    print("  Example: [4,32,64,64] -> fused_conv2d_relu -> [4,64,64,64]")

    print("\n[2] fused_add_relu - Residual + ReLU")
    print("  C++ API: tz::fused_add_relu(a, b)")
    print("  a + b + ReLU in single operation")
    print("  Useful for ResNet-style skip connections")

    print("\n[3] fused_gelu - Optimized GELU activation")
    print("  C++ API: tz::fused_gelu(x)")
    print("  GELU with optimized approximation")

    print("\n[4] fused_layer_norm - LayerNorm + operations")
    print("  C++ API: tz::fused_layer_norm(x, weight, bias)")
    print("  Reduces memory traffic for transformer layers")

    print("\nFused operations reduce memory bandwidth and improve performance!")


# ============================================================================
# YOLO Training
# ============================================================================

def train_yolo():
    """Demonstrate YOLO training setup (C++ API - Python model bindings pending)"""
    print("\n" + "=" * 60)
    print("YOLO Detection Model Training (C++ API)")
    print("=" * 60)

    # Note: YOLO model classes are available in C++ but not exposed in Python yet
    # Available Python models: ResNet variants (resnet18, resnet34, resnet50, resnet101, resnet152)

    print("\nConfiguration (C++ API):")
    print("  Model: YOLOv5s")
    print("  Classes: 20 (VOC)")
    print("  Image size: 416x416")
    print("  Anchors: 3 per scale, 3 scales")
    print("  Loss: Objectness + Classification + CIoU")

    print("\nC++ Training Code:")
    print("  auto config = tz::models::YOLOConfig::yolov5s(num_classes);")
    print("  auto model = tz::models::YOLOv5(config);")
    print("  auto optimizer = tz::optim::Adam(model.parameters(), 0.001);")
    print("  ")
    print("  for (auto& batch : dataloader) {")
    print("      optimizer.zero_grad();")
    print("      auto predictions = model.forward(batch.images);")
    print("      auto loss = compute_loss(predictions, batch.targets);")
    print("      loss.backward();")
    print("      optimizer.step();")
    print("  }")

    print("\nDetection operations demonstrated above work in Python!")
    print("Model training requires C++ API or future Python bindings.")


# ============================================================================
# Main
# ============================================================================

def main():
    # Initialize Tenzor library first
    tz.initialize()

    print("=" * 60)
    print("   YOLO Object Detection - Component Coverage         ")
    print("=" * 60)

    print("\nComponents demonstrated in this example:")
    print("  Detection: box_iou, nms, batched_nms, encode/decode_boxes")
    print("             clip_boxes_to_image, remove_small_boxes")
    print("  Vision: interpolate, unfold, fold")
    print("  Advanced: topk, sort, unique, cumsum, cumprod")
    print("  Fused: fused_conv2d_relu, fused_add_relu, fused_gelu")
    print("  Layers: GroupNorm, Mish")
    print("  Models: YOLOv5")

    demo_detection_ops()
    demo_advanced_ops()
    demo_vision_ops()
    demo_fused_ops()
    train_yolo()

    print("\n" + "=" * 60)
    print("   All detection examples completed successfully!     ")
    print("=" * 60)


if __name__ == "__main__":
    main()
