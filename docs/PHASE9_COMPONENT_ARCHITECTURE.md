# Phase 9 Component Architecture Design

**Version:** 1.0
**Date:** 2025-10-18
**Status:** Approved for Implementation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Existing Infrastructure Analysis](#existing-infrastructure-analysis)
3. [Missing Components Identification](#missing-components-identification)
4. [Detailed Component Specifications](#detailed-component-specifications)
5. [Integration Strategy](#integration-strategy)
6. [Memory and Performance Considerations](#memory-and-performance-considerations)
7. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

This document specifies the architecture for **NEW** components required to implement Phase 9 models (ViT, Swin, EfficientNet, MobileNet, Faster R-CNN, Mask R-CNN, YOLO, U-Net, DeepLab). After comprehensive codebase analysis, the following components are identified as **MISSING** and require implementation:

### Already Implemented ✅
- Conv2d, Conv1d (with groups support for depthwise)
- ConvTranspose2d (for upsampling)
- MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- MultiheadAttention (standard self/cross-attention)
- Linear, BatchNorm2d, LayerNorm
- All standard activations (ReLU, GELU, Swish, etc.)
- Dropout
- Basic tensor operations (reshape, transpose, cat, etc.)

### Missing Components ❌ (To Be Implemented)
1. **Specialized Layers** (11 new layer types)
2. **Detection Components** (9 detection-specific modules)
3. **Utility Functions** (6 helper operations)

**Total Effort:** ~45 hours of implementation + 25 hours testing = **70 hours**

---

## Existing Infrastructure Analysis

### Module System
- **Base Class:** `tenzor::nn::Module` with full autograd integration
- **Parameter Management:** `register_parameter()`, `register_buffer()`, `register_module()`
- **Serialization:** `state_dict()`, `load_state_dict()`, `save()`, `load()`
- **Device Management:** `to()`, `cuda()`, `cpu()`
- **Training Mode:** `train()`, `eval()`, `is_training()`

### Autograd System
- **Variable:** Wraps Tensor with gradient tracking via `VariableImpl`
- **Function:** Base class for custom backward operations
- **Computation Graph:** Automatic graph building via `grad_fn_`
- **Gradient Hooks:** Support for backward hooks

### Backend Dispatch
- **Supported Backends:** CPU, CUDA, ROCm, oneAPI
- **Kernel Infrastructure:** Fused operations, optimized kernels
- **Memory Management:** Device-specific allocators

### Existing Operations
- **Transform Ops:** reshape, view, transpose, permute, squeeze, unsqueeze, flatten, cat, stack, split, expand
- **Math Ops:** Full set (add, mul, div, matmul, etc.)
- **Reduction Ops:** sum, mean, max, min, etc.
- **Indexing Ops:** slice, gather, scatter

---

## Missing Components Identification

### Category 1: Vision Transformer Layers (15h implementation + 8h testing)

#### 1.1 PatchEmbedding
**Purpose:** Convert 2D images into sequence of patch embeddings for ViT
**Status:** ❌ Missing
**Required For:** ViT, Swin Transformer

#### 1.2 WindowAttention
**Purpose:** Shifted window attention for Swin Transformer
**Status:** ❌ Missing (we have MultiheadAttention but not windowed variant)
**Required For:** Swin Transformer

#### 1.3 SwinTransformerBlock
**Purpose:** Complete Swin block with window partitioning, shifted windows, and MLP
**Status:** ❌ Missing
**Required For:** Swin Transformer

### Category 2: Efficient CNN Blocks (12h implementation + 6h testing)

#### 2.1 SqueezeExcitation (SE Block)
**Purpose:** Channel attention mechanism for EfficientNet, MobileNet
**Status:** ❌ Missing
**Required For:** EfficientNet, MobileNetV3, ResNet-SE variants

#### 2.2 InvertedResidual (MBConv)
**Purpose:** Inverted bottleneck with depthwise separable convolutions
**Status:** ❌ Missing (Conv2d supports groups, but need complete module)
**Required For:** MobileNetV2, MobileNetV3, EfficientNet

#### 2.3 FusedMBConv
**Purpose:** Fused MBConv variant for EfficientNet
**Status:** ❌ Missing
**Required For:** EfficientNet (optimization)

#### 2.4 DepthwiseSeparableConv2d
**Purpose:** Depthwise separable convolution wrapper
**Status:** ⚠️ Can use Conv2d with groups=in_channels, but need convenience wrapper
**Required For:** MobileNet family

### Category 3: Specialized Convolutions (8h implementation + 4h testing)

#### 3.1 AtrousConv2d (Dilated Convolution)
**Purpose:** Dilated convolution for semantic segmentation
**Status:** ⚠️ Conv2d has dilation parameter, but need ASPP module
**Required For:** DeepLabV3/V3+

#### 3.2 ASPP (Atrous Spatial Pyramid Pooling)
**Purpose:** Multi-scale feature extraction via parallel dilated convs
**Status:** ❌ Missing
**Required For:** DeepLabV3/V3+

### Category 4: Detection Layers (20h implementation + 12h testing)

#### 4.1 RegionProposalNetwork (RPN)
**Purpose:** Generate object proposals from feature maps
**Status:** ❌ Missing
**Required For:** Faster R-CNN, Mask R-CNN

#### 4.2 AnchorGenerator
**Purpose:** Generate anchor boxes for object detection
**Status:** ❌ Missing
**Required For:** Faster R-CNN, Mask R-CNN, YOLO

#### 4.3 ROIPool
**Purpose:** Extract fixed-size features from regions of interest
**Status:** ❌ Missing
**Required For:** Faster R-CNN

#### 4.4 ROIAlign
**Purpose:** Improved ROI pooling with bilinear interpolation
**Status:** ❌ Missing
**Required For:** Mask R-CNN (critical for mask quality)

#### 4.5 FeaturePyramidNetwork (FPN)
**Purpose:** Multi-scale feature fusion
**Status:** ❌ Missing
**Required For:** Modern detection models

### Category 5: Detection Utilities (15h implementation + 10h testing)

#### 5.1 Non-Maximum Suppression (NMS)
**Purpose:** Suppress overlapping bounding boxes
**Status:** ❌ Missing
**Required For:** All detection models

#### 5.2 BboxEncoder/BboxDecoder
**Purpose:** Encode/decode bounding boxes with anchors
**Status:** ❌ Missing
**Required For:** All detection models

#### 5.3 IoU Computation
**Purpose:** Compute Intersection over Union for boxes
**Status:** ❌ Missing
**Required For:** All detection models, NMS

### Category 6: Utility Operations (10h implementation + 5h testing)

#### 6.1 Unfold (im2col) Operation
**Purpose:** Extract sliding local blocks for patch embedding
**Status:** ❌ Missing
**Required For:** ViT, Swin, efficient convolution implementations

#### 6.2 WindowPartition / WindowReverse
**Purpose:** Partition/reverse windows for Swin Transformer
**Status:** ❌ Missing
**Required For:** Swin Transformer

#### 6.3 GridSample
**Purpose:** Bilinear sampling for ROI Align
**Status:** ❌ Missing
**Required For:** ROI Align, spatial transformers

---

## Detailed Component Specifications

### 1. Vision Transformer Components

#### 1.1 PatchEmbedding

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/vision.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/nn/layers/vision.cpp`
**CUDA Kernel:** `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/vision.cu`

```cpp
namespace tenzor {
namespace nn {

/**
 * @brief Patch embedding layer for Vision Transformers.
 *
 * Splits input image into patches and projects them to embedding dimension.
 *
 * Architecture:
 *   Input: (N, C, H, W)
 *   -> Extract patches of size (patch_size, patch_size)
 *   -> Flatten patches: (N, num_patches, C * patch_size^2)
 *   -> Linear projection: (N, num_patches, embed_dim)
 *
 * Where num_patches = (H / patch_size) * (W / patch_size)
 *
 * @code
 * // ViT-B/16 configuration
 * PatchEmbedding patch_embed(3, 768, 16);
 *
 * Variable img({batch, 3, 224, 224}, DType::Float32, Device::cpu(), true);
 * Variable patches = patch_embed.forward(img);  // {batch, 196, 768}
 * @endcode
 */
class PatchEmbedding : public Module {
public:
    /**
     * @brief Construct patch embedding layer.
     *
     * @param in_channels Number of input channels (e.g., 3 for RGB)
     * @param embed_dim Embedding dimension
     * @param patch_size Size of each square patch
     * @param img_size Input image size (used for validation)
     */
    PatchEmbedding(int64_t in_channels,
                   int64_t embed_dim,
                   int64_t patch_size,
                   int64_t img_size = 224);

    auto forward(const Variable& input) -> Variable override;

    auto num_patches() const -> int64_t { return num_patches_; }

private:
    int64_t in_channels_;
    int64_t embed_dim_;
    int64_t patch_size_;
    int64_t img_size_;
    int64_t num_patches_;

    std::shared_ptr<Conv2d> proj_;  // Use conv2d for efficient patch extraction
};

} // namespace nn
} // namespace tenzor
```

**Implementation Strategy:**
- Use Conv2d with kernel_size=patch_size, stride=patch_size for efficient extraction
- Alternative: Unfold + Linear (implement both, benchmark)
- Flatten spatial dimensions: reshape from (N, embed_dim, H', W') to (N, H'*W', embed_dim)
- Add position embeddings in transformer layer (not in PatchEmbedding)

**Autograd Integration:**
- Conv2d already has full backward pass
- Reshape operations propagate gradients automatically

**Memory Efficiency:**
- For 224x224 image with 16x16 patches: 196 patches
- Embedding size 768: 196 * 768 = 150,528 values per image
- Use in-place operations where possible

**CUDA Kernel Requirements:**
- Leverage existing conv2d CUDA kernel
- No custom kernel needed (efficient as-is)

---

#### 1.2 WindowAttention

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/vision.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/nn/layers/vision.cpp`

```cpp
/**
 * @brief Window-based multi-head attention for Swin Transformer.
 *
 * Applies multi-head attention within local windows to reduce complexity
 * from O(H*W)^2 to O(M^2 * H*W) where M is window size.
 *
 * Supports:
 * - Local window attention (non-overlapping windows)
 * - Shifted window attention for cross-window connections
 * - Relative position bias
 *
 * @code
 * WindowAttention attn(768, 8, 7);  // dim=768, heads=8, window_size=7
 *
 * // Input: (batch, num_windows, window_size^2, dim)
 * Variable x({batch, 64, 49, 768}, DType::Float32, Device::cpu(), true);
 * auto [output, attn_weights] = attn.forward(x);
 * @endcode
 */
class WindowAttention : public Module {
public:
    /**
     * @brief Construct window attention layer.
     *
     * @param dim Feature dimension
     * @param num_heads Number of attention heads
     * @param window_size Window size (M)
     * @param qkv_bias Add bias to qkv projection (default: true)
     * @param attn_drop Attention dropout probability (default: 0.0)
     * @param proj_drop Projection dropout probability (default: 0.0)
     */
    WindowAttention(int64_t dim,
                    int64_t num_heads,
                    int64_t window_size,
                    bool qkv_bias = true,
                    double attn_drop = 0.0,
                    double proj_drop = 0.0);

    /**
     * @brief Forward pass with optional attention mask.
     *
     * @param x Input features (num_windows*B, N, C) where N = window_size^2
     * @param mask Optional attention mask for shifted window (default: none)
     * @return Pair of (output, attention_weights)
     */
    auto forward(const Variable& x, const Tensor& mask = Tensor{})
        -> std::pair<Variable, Variable>;

    auto forward(const Variable& input) -> Variable override {
        return forward(input, Tensor{}).first;
    }

private:
    int64_t dim_;
    int64_t num_heads_;
    int64_t window_size_;
    int64_t head_dim_;
    double scale_;  // 1 / sqrt(head_dim)

    std::shared_ptr<Linear> qkv_;      // Combined Q, K, V projection
    std::shared_ptr<Linear> proj_;     // Output projection
    std::shared_ptr<Dropout> attn_drop_;
    std::shared_ptr<Dropout> proj_drop_;

    Variable relative_position_bias_table_;  // Learnable bias table
    Tensor relative_position_index_;         // Precomputed index for bias lookup

    auto get_relative_position_bias() const -> Tensor;
};
```

**Implementation Strategy:**
- Compute Q, K, V via single linear projection (efficiency)
- Reshape to (batch, num_heads, num_patches, head_dim)
- Scaled dot-product attention with relative position bias
- Relative position bias: learnable table indexed by relative positions

**Autograd Integration:**
- Linear projections: automatic gradient via existing Linear backward
- Softmax: existing autograd
- Relative position bias: embed as learnable parameter

**Memory Efficiency:**
- Window size 7: 49x49 attention matrix per head (small!)
- Compared to global: 14x14 = 196, saves 16x memory
- Relative position bias table: (2*M-1) x (2*M-1) = 13x13 = 169 values

**CUDA Kernel Requirements:**
- Use existing matrix multiply and softmax kernels
- Potential optimization: fused attention kernel (Phase 10)

---

#### 1.3 WindowPartition / WindowReverse Utilities

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/ops/vision.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/ops/vision.cpp`

```cpp
namespace tenzor {
namespace ops {

/**
 * @brief Partition feature map into non-overlapping windows.
 *
 * Converts (B, H, W, C) -> (num_windows*B, window_size, window_size, C)
 *
 * @param x Input features (B, H, W, C)
 * @param window_size Window size M
 * @return Windowed features (num_windows*B, M, M, C)
 *
 * @code
 * Tensor x({2, 56, 56, 96}, DType::Float32, Device::cpu());
 * Tensor windows = window_partition(x, 7);  // {128, 7, 7, 96}
 * // where 128 = 2 * (56/7) * (56/7) = 2 * 64
 * @endcode
 */
auto window_partition(const Tensor& x, int64_t window_size) -> Tensor;

/**
 * @brief Reverse window partition operation.
 *
 * Converts (num_windows*B, window_size, window_size, C) -> (B, H, W, C)
 *
 * @param windows Windowed features
 * @param window_size Window size M
 * @param H Original height
 * @param W Original width
 * @return Merged features (B, H, W, C)
 */
auto window_reverse(const Tensor& windows,
                   int64_t window_size,
                   int64_t H,
                   int64_t W) -> Tensor;

/**
 * @brief Create attention mask for shifted window.
 *
 * For SW-MSA (Shifted Window Multi-head Self Attention), creates mask
 * to prevent attention between non-adjacent windows.
 *
 * @param H Feature map height
 * @param W Feature map width
 * @param window_size Window size
 * @param shift_size Shift amount (typically window_size / 2)
 * @param device Device to create mask on
 * @return Attention mask
 */
auto create_shifted_window_mask(int64_t H,
                                int64_t W,
                                int64_t window_size,
                                int64_t shift_size,
                                Device device = Device::cpu()) -> Tensor;

} // namespace ops
} // namespace tenzor
```

**Implementation Strategy:**
- `window_partition`: Reshape + Transpose operations
  1. Reshape (B, H, W, C) -> (B, H/M, M, W/M, M, C)
  2. Transpose to (B, H/M, W/M, M, M, C)
  3. Reshape to (B*H/M*W/M, M, M, C)
- `window_reverse`: Inverse operations
- Use existing tensor ops (reshape, transpose, permute)

**Autograd Integration:**
- All operations are differentiable tensor transforms
- Gradients flow through existing reshape/transpose backward

**Memory Efficiency:**
- Zero-copy where possible (view operations)
- May need contiguous() for certain reshape patterns

---

### 2. Efficient CNN Blocks

#### 2.1 SqueezeExcitation

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/mobilenet.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/nn/layers/mobilenet.cpp`

```cpp
namespace tenzor {
namespace nn {

/**
 * @brief Squeeze-and-Excitation block for channel attention.
 *
 * Recalibrates channel-wise features via global pooling and gating.
 *
 * Architecture:
 *   Input (B, C, H, W)
 *   -> GlobalAvgPool -> (B, C, 1, 1)
 *   -> FC (C -> C/r) -> ReLU
 *   -> FC (C/r -> C) -> Sigmoid
 *   -> Scale input channels
 *
 * @code
 * SqueezeExcitation se(256, 16);  // 256 channels, reduction=16
 *
 * Variable x({batch, 256, 28, 28}, DType::Float32, Device::cpu(), true);
 * Variable out = se.forward(x);  // Same shape, channel-wise rescaled
 * @endcode
 */
class SqueezeExcitation : public Module {
public:
    /**
     * @brief Construct SE block.
     *
     * @param channels Number of input/output channels
     * @param reduction Reduction ratio for bottleneck (default: 16)
     * @param activation Activation function ("relu" or "swish", default: "relu")
     */
    SqueezeExcitation(int64_t channels,
                      int64_t reduction = 16,
                      std::string activation = "relu");

    auto forward(const Variable& input) -> Variable override;

private:
    std::shared_ptr<AdaptiveAvgPool2d> pool_;
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<Module> activation_;  // ReLU or Swish
};

} // namespace nn
} // namespace tenzor
```

**Implementation Strategy:**
1. Global average pooling: AdaptiveAvgPool2d(1, 1)
2. Squeeze spatial dimensions: reshape (B, C, 1, 1) -> (B, C)
3. Two FC layers: C -> C/r -> C
4. Sigmoid activation for gating
5. Expand back: (B, C) -> (B, C, 1, 1)
6. Multiply with input (channel-wise)

**Autograd Integration:**
- All layers (pool, linear, sigmoid) have existing backward
- Element-wise multiply propagates gradients to both inputs

**Memory Efficiency:**
- Squeeze path: O(C^2/r) parameters (tiny)
- Activation map: O(1) spatial size
- Very lightweight compared to input

**CUDA Kernel Requirements:**
- Use existing kernels (no custom kernel needed)

---

#### 2.2 InvertedResidual (MBConv)

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/mobilenet.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/nn/layers/mobilenet.cpp`

```cpp
/**
 * @brief Inverted residual block (Mobile Inverted Bottleneck Convolution).
 *
 * Used in MobileNetV2, MobileNetV3, and EfficientNet.
 *
 * Architecture:
 *   1. Expansion: 1x1 conv to expand channels (C -> C*expand_ratio)
 *   2. Depthwise: 3x3 depthwise conv (groups = C*expand_ratio)
 *   3. Projection: 1x1 conv to project back (C*expand_ratio -> out_channels)
 *   4. Skip connection if stride=1 and in_channels=out_channels
 *
 * Optional:
 *   - Squeeze-and-Excitation after depthwise
 *   - Different activations (ReLU, ReLU6, Swish)
 *
 * @code
 * InvertedResidual block(32, 64, 6, 1, true);  // in=32, out=64, expand=6
 *
 * Variable x({batch, 32, 56, 56}, DType::Float32, Device::cpu(), true);
 * Variable out = block.forward(x);  // {batch, 64, 56, 56}
 * @endcode
 */
class InvertedResidual : public Module {
public:
    /**
     * @brief Construct inverted residual block.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param expand_ratio Channel expansion ratio (default: 6)
     * @param stride Stride for depthwise conv (default: 1)
     * @param use_se Use Squeeze-and-Excitation (default: false)
     * @param activation Activation type ("relu", "relu6", "swish")
     */
    InvertedResidual(int64_t in_channels,
                     int64_t out_channels,
                     int64_t expand_ratio = 6,
                     int64_t stride = 1,
                     bool use_se = false,
                     std::string activation = "relu6");

    auto forward(const Variable& input) -> Variable override;

private:
    bool use_residual_;  // Use skip connection
    int64_t stride_;

    std::shared_ptr<Sequential> conv_;  // Main conv path
    std::shared_ptr<SqueezeExcitation> se_;  // Optional SE
};

} // namespace nn
} // namespace tenzor
```

**Implementation Strategy:**
1. Pointwise expansion: Conv2d(in, in*expand, 1, 1, 0)
2. Depthwise conv: Conv2d(in*expand, in*expand, 3, stride, 1, groups=in*expand)
3. Pointwise projection: Conv2d(in*expand, out, 1, 1, 0)
4. Optional SE block after depthwise
5. Skip connection: input + output (if stride=1 and in=out)

**Autograd Integration:**
- Conv2d backward: already implemented
- Skip connection: add backward (sum gradients)

**Memory Efficiency:**
- Depthwise conv saves massive computation vs standard conv
- Parameters: K^2 * C + C * C' vs K^2 * C * C'
- For 3x3, C=64: 9*64 + 64*64 = 4,672 vs 9*64*64 = 36,864 (7.9x savings)

**CUDA Kernel Requirements:**
- Use existing conv2d CUDA kernels
- Depthwise already supported via groups parameter

---

#### 2.3 FusedMBConv

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/mobilenet.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/nn/layers/mobilenet.cpp`

```cpp
/**
 * @brief Fused Mobile Inverted Bottleneck (for EfficientNet).
 *
 * Replaces expansion + depthwise with single 3x3 conv for efficiency
 * in early layers where resolution is high.
 *
 * Architecture:
 *   1. Fused: 3x3 conv (C -> C*expand_ratio)
 *   2. Projection: 1x1 conv (C*expand_ratio -> out_channels)
 *   3. Optional SE
 *   4. Skip connection
 *
 * @code
 * FusedMBConv block(32, 64, 4, 1);  // Fused expansion
 * @endcode
 */
class FusedMBConv : public Module {
public:
    FusedMBConv(int64_t in_channels,
                int64_t out_channels,
                int64_t expand_ratio = 4,
                int64_t stride = 1,
                bool use_se = false);

    auto forward(const Variable& input) -> Variable override;

private:
    bool use_residual_;
    std::shared_ptr<Sequential> conv_;
    std::shared_ptr<SqueezeExcitation> se_;
};

} // namespace nn
} // namespace tenzor
```

**Implementation Strategy:**
- Single 3x3 conv instead of 1x1 + 3x3 depthwise
- Rest identical to InvertedResidual
- Used in EfficientNetV2 for early stages

---

### 3. Detection Components

#### 3.1 AnchorGenerator

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/nn/detection/anchors.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/nn/detection/anchors.cpp`

```cpp
namespace tenzor {
namespace nn {
namespace detection {

/**
 * @brief Generate anchor boxes for object detection.
 *
 * Creates anchor boxes at different scales and aspect ratios for each
 * position in the feature map. Used in Faster R-CNN, Mask R-CNN, RetinaNet.
 *
 * @code
 * AnchorGenerator anchors(
 *     {32, 64, 128, 256, 512},  // sizes
 *     {0.5, 1.0, 2.0}            // aspect ratios
 * );
 *
 * // Generate for feature map 38x38
 * std::vector<int64_t> feature_shape = {38, 38};
 * int64_t stride = 16;
 * Tensor boxes = anchors.generate(feature_shape, stride);
 * // Shape: (num_anchors, 4) where 4 = (x1, y1, x2, y2)
 * @endcode
 */
class AnchorGenerator {
public:
    /**
     * @brief Construct anchor generator.
     *
     * @param sizes Anchor sizes (base sizes in pixels)
     * @param aspect_ratios Anchor aspect ratios (width/height)
     */
    AnchorGenerator(std::vector<float> sizes,
                    std::vector<float> aspect_ratios);

    /**
     * @brief Generate anchors for feature map.
     *
     * @param feature_shape Shape of feature map (H, W)
     * @param stride Stride of feature map relative to input image
     * @param device Device to create anchors on
     * @return Anchor boxes (num_anchors, 4) in (x1, y1, x2, y2) format
     */
    auto generate(const std::vector<int64_t>& feature_shape,
                  int64_t stride,
                  Device device = Device::cpu()) const -> Tensor;

    /**
     * @brief Number of anchors per location.
     */
    auto num_anchors_per_location() const -> int64_t {
        return sizes_.size() * aspect_ratios_.size();
    }

private:
    std::vector<float> sizes_;
    std::vector<float> aspect_ratios_;
};

} // namespace detection
} // namespace nn
} // namespace tenzor
```

**Implementation Strategy:**
1. For each feature map location (i, j):
   - Compute center: (j * stride, i * stride)
   - For each (size, aspect_ratio) combination:
     - Width = size * sqrt(aspect_ratio)
     - Height = size / sqrt(aspect_ratio)
     - Create box: (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
2. Return all anchors as (N, 4) tensor

**Autograd Integration:**
- Not differentiable (anchor generation is fixed)
- Used in forward pass only

**Memory Efficiency:**
- Cache anchors per feature map size
- Typically ~15-50K anchors for 800x800 image

**CUDA Kernel Requirements:**
- CPU implementation sufficient (one-time generation)
- Can implement CUDA version for large batch generation

---

#### 3.2 ROIAlign

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/nn/detection/roi_ops.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/nn/detection/roi_ops.cpp`
**CUDA Kernel:** `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/roi_align.cu`

```cpp
/**
 * @brief ROI Align layer for Mask R-CNN.
 *
 * Extracts fixed-size feature maps from regions of interest using
 * bilinear interpolation. Improves upon ROI Pooling by avoiding
 * quantization for better mask quality.
 *
 * @code
 * ROIAlign roi_align(7, 7, 1.0/16.0, 2);  // 7x7 output, scale=1/16, sampling=2
 *
 * Variable features({batch, 256, 38, 38}, DType::Float32, Device::cpu(), true);
 * Tensor rois({num_rois, 5}, DType::Float32, Device::cpu());
 * // ROIs format: (batch_idx, x1, y1, x2, y2)
 *
 * Variable aligned = roi_align.forward(features, rois);
 * // Shape: (num_rois, 256, 7, 7)
 * @endcode
 */
class ROIAlign : public Module {
public:
    /**
     * @brief Construct ROI Align layer.
     *
     * @param output_h Output height
     * @param output_w Output width
     * @param spatial_scale Scale factor from input image to feature map
     * @param sampling_ratio Number of sampling points (0 = adaptive)
     * @param aligned Use aligned coordinates (default: true)
     */
    ROIAlign(int64_t output_h,
             int64_t output_w,
             double spatial_scale,
             int64_t sampling_ratio = 0,
             bool aligned = true);

    /**
     * @brief Forward pass.
     *
     * @param features Input feature map (N, C, H, W)
     * @param rois Regions of interest (num_rois, 5)
     *             Format: (batch_index, x1, y1, x2, y2)
     * @return Aligned features (num_rois, C, output_h, output_w)
     */
    auto forward(const Variable& features, const Tensor& rois) -> Variable;

    auto forward(const Variable& input) -> Variable override {
        throw std::runtime_error("ROIAlign requires both features and rois");
    }

private:
    int64_t output_h_;
    int64_t output_w_;
    double spatial_scale_;
    int64_t sampling_ratio_;
    bool aligned_;
};
```

**Implementation Strategy:**
1. For each ROI:
   - Scale ROI coordinates by spatial_scale
   - Divide ROI into output_h x output_w bins
   - For each bin:
     - Sample sampling_ratio^2 points (or adaptive)
     - Bilinearly interpolate feature map at each point
     - Average sampled values
2. Stack outputs for all ROIs

**Autograd Integration:**
- Custom backward function needed
- Gradient flows back to features via bilinear interpolation weights
- ROIs are not differentiable (fixed coordinates)

**Memory Efficiency:**
- Output size: num_rois * C * output_h * output_w
- Typical: 1000 ROIs * 256 * 7 * 7 = 12.5M values

**CUDA Kernel Requirements:**
- **Critical:** CUDA kernel essential for performance
- CPU fallback for debugging
- Bilinear interpolation kernel with atomic adds for backward

---

#### 3.3 RegionProposalNetwork

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/nn/detection/rpn.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/nn/detection/rpn.cpp`

```cpp
/**
 * @brief Region Proposal Network for Faster R-CNN.
 *
 * Generates object proposals from feature maps by predicting
 * objectness scores and bounding box deltas for anchors.
 *
 * Architecture:
 *   Feature Map -> 3x3 Conv -> ReLU
 *                           |-> 1x1 Conv (objectness: 2k outputs)
 *                           |-> 1x1 Conv (bbox deltas: 4k outputs)
 *   where k = num_anchors_per_location
 *
 * @code
 * RPNHead rpn(256, 9);  // 256 channels, 9 anchors per location
 *
 * Variable features({batch, 256, 38, 38}, DType::Float32, Device::cpu(), true);
 * auto [objectness, bbox_deltas] = rpn.forward(features);
 * // objectness: (batch, num_anchors, 2)
 * // bbox_deltas: (batch, num_anchors, 4)
 * @endcode
 */
class RPNHead : public Module {
public:
    /**
     * @brief Construct RPN head.
     *
     * @param in_channels Number of input feature channels
     * @param num_anchors Number of anchors per location
     */
    RPNHead(int64_t in_channels, int64_t num_anchors);

    /**
     * @brief Forward pass.
     *
     * @param features Input feature map (N, C, H, W)
     * @return Pair of (objectness_scores, bbox_deltas)
     */
    auto forward(const Variable& features)
        -> std::pair<Variable, Variable>;

    auto forward(const Variable& input) -> Variable override {
        return forward(input).first;  // Return objectness for Module interface
    }

private:
    std::shared_ptr<Conv2d> conv_;
    std::shared_ptr<Conv2d> cls_logits_;
    std::shared_ptr<Conv2d> bbox_pred_;
    nn::ReLU relu_;
};

/**
 * @brief Complete Region Proposal Network.
 *
 * Combines RPNHead with anchor generation, proposal generation,
 * and NMS to produce final region proposals.
 */
class RegionProposalNetwork : public Module {
public:
    /**
     * @brief Construct RPN.
     *
     * @param anchor_generator Anchor generator
     * @param rpn_head RPN head for predictions
     * @param fg_iou_thresh Foreground IoU threshold (training)
     * @param bg_iou_thresh Background IoU threshold (training)
     * @param batch_size_per_image Anchors per image (training)
     * @param positive_fraction Fraction of positive anchors
     * @param pre_nms_top_n Top N proposals before NMS
     * @param post_nms_top_n Top N proposals after NMS
     * @param nms_thresh NMS IoU threshold
     */
    RegionProposalNetwork(std::shared_ptr<AnchorGenerator> anchor_generator,
                          std::shared_ptr<RPNHead> rpn_head,
                          double fg_iou_thresh = 0.7,
                          double bg_iou_thresh = 0.3,
                          int64_t batch_size_per_image = 256,
                          double positive_fraction = 0.5,
                          int64_t pre_nms_top_n = 2000,
                          int64_t post_nms_top_n = 1000,
                          double nms_thresh = 0.7);

    auto forward(const Variable& input) -> Variable override;

private:
    std::shared_ptr<AnchorGenerator> anchor_generator_;
    std::shared_ptr<RPNHead> rpn_head_;

    // Training parameters
    double fg_iou_thresh_;
    double bg_iou_thresh_;
    int64_t batch_size_per_image_;
    double positive_fraction_;

    // Inference parameters
    int64_t pre_nms_top_n_;
    int64_t post_nms_top_n_;
    double nms_thresh_;
};

} // namespace detection
} // namespace nn
} // namespace tenzor
```

**Implementation Strategy:**
1. Generate anchors for feature map
2. Apply RPN head to get objectness and deltas
3. Decode bounding boxes from deltas
4. Apply NMS to remove overlapping proposals
5. Return top N proposals

**Autograd Integration:**
- RPN head: full backward through conv layers
- NMS: non-differentiable (applied in eval mode)
- Training: use all anchors with sampling

---

#### 3.4 Non-Maximum Suppression (NMS)

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/ops/detection.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/ops/detection.cpp`
**CUDA Kernel:** `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/nms.cu`

```cpp
namespace tenzor {
namespace ops {

/**
 * @brief Non-Maximum Suppression for bounding boxes.
 *
 * Filters overlapping bounding boxes by keeping only the highest-scoring
 * box in each cluster of overlapping boxes.
 *
 * Algorithm:
 *   1. Sort boxes by score (descending)
 *   2. For each box:
 *      - If not suppressed:
 *        - Keep box
 *        - Suppress all boxes with IoU > threshold
 *
 * @param boxes Bounding boxes (N, 4) in (x1, y1, x2, y2) format
 * @param scores Confidence scores (N,)
 * @param iou_threshold IoU threshold for suppression (default: 0.5)
 * @return Indices of kept boxes
 *
 * @code
 * Tensor boxes({1000, 4}, DType::Float32, Device::cpu());
 * Tensor scores({1000}, DType::Float32, Device::cpu());
 * Tensor keep_indices = nms(boxes, scores, 0.5);
 * Tensor kept_boxes = boxes.index_select(0, keep_indices);
 * @endcode
 */
auto nms(const Tensor& boxes,
         const Tensor& scores,
         double iou_threshold = 0.5) -> Tensor;

/**
 * @brief Batched NMS for multiple classes.
 *
 * Applies NMS separately for each class to prevent cross-class suppression.
 *
 * @param boxes Bounding boxes (N, 4)
 * @param scores Class scores (N, num_classes)
 * @param iou_threshold IoU threshold
 * @param score_threshold Score threshold (filter low-confidence boxes)
 * @param max_output_boxes Maximum boxes to return per class
 * @return Tuple of (kept_boxes, kept_scores, kept_classes)
 */
auto batched_nms(const Tensor& boxes,
                 const Tensor& scores,
                 double iou_threshold = 0.5,
                 double score_threshold = 0.05,
                 int64_t max_output_boxes = 100)
    -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Compute Intersection over Union (IoU) for bounding boxes.
 *
 * @param boxes1 First set of boxes (N, 4)
 * @param boxes2 Second set of boxes (M, 4)
 * @return IoU matrix (N, M)
 */
auto box_iou(const Tensor& boxes1, const Tensor& boxes2) -> Tensor;

/**
 * @brief Encode bounding boxes relative to anchors.
 *
 * Used in training to convert ground truth boxes to regression targets.
 *
 * @param boxes Ground truth boxes (N, 4)
 * @param anchors Anchor boxes (N, 4)
 * @param weights Encoding weights (default: [1, 1, 1, 1])
 * @return Encoded deltas (N, 4) as (dx, dy, dw, dh)
 */
auto encode_boxes(const Tensor& boxes,
                  const Tensor& anchors,
                  const std::vector<double>& weights = {1.0, 1.0, 1.0, 1.0})
    -> Tensor;

/**
 * @brief Decode bounding boxes from deltas and anchors.
 *
 * Used in inference to convert predicted deltas to actual boxes.
 *
 * @param deltas Predicted deltas (N, 4) as (dx, dy, dw, dh)
 * @param anchors Anchor boxes (N, 4)
 * @param weights Encoding weights (must match encode_boxes)
 * @return Decoded boxes (N, 4)
 */
auto decode_boxes(const Tensor& deltas,
                  const Tensor& anchors,
                  const std::vector<double>& weights = {1.0, 1.0, 1.0, 1.0})
    -> Tensor;

} // namespace ops
} // namespace tenzor
```

**Implementation Strategy:**

**NMS Algorithm:**
```
Input: boxes (N, 4), scores (N,), threshold
Output: keep_indices

1. Sort boxes by score (descending) -> order
2. keep = []
3. while order is not empty:
     i = order[0]
     keep.append(i)
     if len(order) == 1:
         break
     ious = box_iou(boxes[i], boxes[order[1:]])
     order = order[1:][ious <= threshold]
4. return keep
```

**Box IoU:**
```
For boxes1[i] = (x1_i, y1_i, x2_i, y2_i) and boxes2[j]:
  inter_x1 = max(x1_i, x1_j)
  inter_y1 = max(y1_i, y1_j)
  inter_x2 = min(x2_i, x2_j)
  inter_y2 = min(y2_i, y2_j)
  inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
  area_i = (x2_i - x1_i) * (y2_i - y1_i)
  area_j = (x2_j - x1_j) * (y2_j - y1_j)
  union_area = area_i + area_j - inter_area
  iou = inter_area / union_area
```

**Box Encoding (Faster R-CNN style):**
```
Given anchor (ax, ay, aw, ah) and box (bx, by, bw, bh):
  dx = (bx - ax) / aw / weights[0]
  dy = (by - ay) / ah / weights[1]
  dw = log(bw / aw) / weights[2]
  dh = log(bh / ah) / weights[3]
```

**Autograd Integration:**
- NMS: non-differentiable (used in inference only)
- Box encoding/decoding: non-differentiable (used in loss computation)
- Box IoU: can be made differentiable if needed (for IoU loss)

**Memory Efficiency:**
- IoU computation: O(N*M) memory for N boxes and M anchors
- NMS: O(N^2) worst case, typically much better
- Use CUDA for large N (>1000 boxes)

**CUDA Kernel Requirements:**
- **Critical for performance:** CUDA NMS kernel
- Parallel IoU computation
- Optimized sorting (thrust library)

---

### 4. Utility Operations

#### 4.1 Unfold (im2col) Operation

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/ops/transform.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/ops/transform.cpp`

```cpp
namespace tenzor {
namespace ops {

/**
 * @brief Extract sliding local blocks (unfold/im2col operation).
 *
 * Extracts sliding local blocks from a batched input tensor.
 * Useful for patch extraction in Vision Transformers and efficient
 * convolution implementations.
 *
 * @param input Input tensor (N, C, H, W)
 * @param kernel_size Size of sliding blocks
 * @param stride Stride of sliding blocks (default: 1)
 * @param padding Padding applied to input (default: 0)
 * @param dilation Dilation of kernel elements (default: 1)
 * @return Unfolded tensor (N, C*K*K, L) where L = num_blocks
 *
 * @code
 * Tensor img({1, 3, 224, 224}, DType::Float32, Device::cpu());
 * Tensor patches = unfold(img, 16, 16, 0, 1);
 * // Shape: (1, 3*16*16, 196) where 196 = (224/16)^2
 * @endcode
 */
auto unfold(const Tensor& input,
            int64_t kernel_size,
            int64_t stride = 1,
            int64_t padding = 0,
            int64_t dilation = 1) -> Tensor;

/**
 * @brief Fold tensor back to spatial dimensions (col2im).
 *
 * Reverse operation of unfold. Accumulates overlapping blocks.
 *
 * @param input Unfolded tensor (N, C*K*K, L)
 * @param output_size Output spatial size (H, W)
 * @param kernel_size Size of blocks
 * @param stride Stride used in unfold
 * @param padding Padding used in unfold
 * @param dilation Dilation used in unfold
 * @return Folded tensor (N, C, H, W)
 */
auto fold(const Tensor& input,
          const std::vector<int64_t>& output_size,
          int64_t kernel_size,
          int64_t stride = 1,
          int64_t padding = 0,
          int64_t dilation = 1) -> Tensor;

} // namespace ops
} // namespace tenzor
```

**Implementation Strategy:**
- Unfold: Extract each KxK block at stride positions
- Can be implemented via reshape + permute + reshape
- Alternative: explicit loop over spatial positions (simpler but slower)

**Autograd Integration:**
- Unfold backward: fold operation
- Fold backward: unfold operation (adjoint operators)

**Memory Efficiency:**
- Unfold can explode memory for large K and small stride
- Example: 224x224 image, K=16, stride=16: 196 patches (manageable)
- Example: 224x224 image, K=7, stride=1: ~48K patches (large!)

**CUDA Kernel Requirements:**
- Custom CUDA kernel recommended for performance
- Can reuse im2col from cuDNN if available

---

### 5. Additional Vision Layers

#### 5.1 ASPP (Atrous Spatial Pyramid Pooling)

**Header:** `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/segmentation.hpp`
**Source:** `/home/lee/Projects/Tenzor/src/nn/layers/segmentation.cpp`

```cpp
namespace tenzor {
namespace nn {

/**
 * @brief Atrous Spatial Pyramid Pooling for DeepLab.
 *
 * Multi-scale feature extraction using parallel dilated convolutions
 * at different rates, plus global average pooling.
 *
 * Architecture:
 *   Input features -> |-> 1x1 conv
 *                     |-> 3x3 conv, dilation=6
 *                     |-> 3x3 conv, dilation=12
 *                     |-> 3x3 conv, dilation=18
 *                     |-> Global Pool -> 1x1 conv -> Upsample
 *   -> Concat all -> 1x1 conv -> Output
 *
 * @code
 * ASPP aspp(2048, 256, {6, 12, 18});  // in=2048, out=256
 *
 * Variable x({batch, 2048, 32, 32}, DType::Float32, Device::cpu(), true);
 * Variable out = aspp.forward(x);  // {batch, 256, 32, 32}
 * @endcode
 */
class ASPP : public Module {
public:
    /**
     * @brief Construct ASPP module.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param atrous_rates Dilation rates for parallel convs
     */
    ASPP(int64_t in_channels,
         int64_t out_channels,
         const std::vector<int64_t>& atrous_rates = {6, 12, 18});

    auto forward(const Variable& input) -> Variable override;

private:
    std::vector<std::shared_ptr<Sequential>> atrous_convs_;
    std::shared_ptr<Sequential> global_pool_;
    std::shared_ptr<Sequential> project_;
};

} // namespace nn
} // namespace tenzor
```

**Implementation Strategy:**
1. Create parallel branches with different dilation rates
2. Apply each branch to input
3. Concatenate all outputs along channel dimension
4. Project to final output channels via 1x1 conv

**Autograd Integration:**
- All conv layers: automatic gradient
- Concatenation: split gradient along channel dim

**Memory Efficiency:**
- Creates num_rates + 2 intermediate feature maps
- Typical: 5 branches * C channels = 5C intermediate storage

---

## Integration Strategy

### File Organization

```
include/tenzor/nn/
├── layers/
│   ├── vision.hpp              # PatchEmbedding, WindowAttention
│   ├── mobilenet.hpp           # SE, InvertedResidual, FusedMBConv
│   └── segmentation.hpp        # ASPP
├── detection/
│   ├── anchors.hpp             # AnchorGenerator
│   ├── roi_ops.hpp             # ROIAlign, ROIPool
│   └── rpn.hpp                 # RPNHead, RegionProposalNetwork

include/tenzor/ops/
├── vision.hpp                  # window_partition, window_reverse
└── detection.hpp               # nms, box_iou, encode/decode_boxes

src/nn/
├── layers/
│   ├── vision.cpp
│   ├── mobilenet.cpp
│   └── segmentation.cpp
├── detection/
│   ├── anchors.cpp
│   ├── roi_ops.cpp
│   └── rpn.cpp

src/ops/
├── vision.cpp
└── detection.cpp

src/backends/cuda/kernels/
├── vision.cu                   # patch_embed, window ops (if needed)
├── roi_align.cu                # Critical: ROIAlign CUDA kernel
└── nms.cu                      # Critical: NMS CUDA kernel
```

### CMake Integration

Add to `/home/lee/Projects/Tenzor/src/nn/CMakeLists.txt`:
```cmake
# Vision layers
target_sources(tenzor_nn PRIVATE
    layers/vision.cpp
    layers/mobilenet.cpp
    layers/segmentation.cpp
)

# Detection components
target_sources(tenzor_nn PRIVATE
    detection/anchors.cpp
    detection/roi_ops.cpp
    detection/rpn.cpp
)
```

Add to `/home/lee/Projects/Tenzor/src/backends/cuda/CMakeLists.txt`:
```cmake
target_sources(tenzor_cuda_backend PRIVATE
    kernels/roi_align.cu
    kernels/nms.cu
    kernels/vision.cu  # Optional optimizations
)
```

### Autograd Integration Pattern

All new layers follow the existing pattern:

```cpp
// Example: SqueezeExcitation forward
auto SqueezeExcitation::forward(const Variable& input) -> Variable {
    // 1. Extract tensor
    auto x = input.tensor();

    // 2. Apply operations (builds computation graph)
    auto pooled = pool_->forward(input);       // Tracks gradients
    auto squeezed = pooled.reshape({-1, channels_});
    auto fc1_out = fc1_->forward(squeezed);
    auto activated = activation_->forward(fc1_out);
    auto fc2_out = fc2_->forward(activated);
    auto gates = fc2_out.sigmoid();

    // 3. Apply gates (element-wise multiply)
    auto expanded = gates.unsqueeze(-1).unsqueeze(-1);
    auto output = input * expanded;             // Gradient flows to both

    // 4. Return Variable (computation graph intact)
    return output;
}
```

**Key Principles:**
- Use Variable for all intermediate computations
- Leverage existing layer backward passes (Linear, Conv2d, etc.)
- Element-wise ops automatically propagate gradients
- Only implement custom backward for truly novel operations (ROIAlign)

---

## Memory and Performance Considerations

### Memory Optimization Strategies

#### 1. Gradient Checkpointing for Deep Models
- Swin Transformer: 24+ layers can OOM
- Implement `checkpoint()` wrapper for transformer blocks
- Trade-off: 2x slower backward, 10x less memory

#### 2. ROI Operations Memory
- ROIAlign with 1000 ROIs, 256 channels, 7x7: ~12.5M values
- Batch ROI operations efficiently
- Consider maximum ROI limit per forward pass

#### 3. Window Attention Caching
- Cache window masks for Swin Transformer
- Masks are same across batch and layers
- Memory: O(window_size^2) (tiny)

#### 4. Anchor Generation Caching
- Generate anchors once per image size
- Cache in `AnchorGenerator` class
- Avoid recomputation across batches

### Performance Optimization

#### Critical CUDA Kernels (Priority Order)
1. **ROIAlign** - Used in every Mask R-CNN forward pass
   - Bilinear interpolation kernel
   - Atomic operations for backward
   - Benchmark target: 5-10ms for 1000 ROIs on V100

2. **NMS** - Used in every detection inference
   - Parallel IoU computation
   - Efficient suppression with bitmasks
   - Benchmark target: 2-5ms for 1000 boxes on V100

3. **Window Partition** - Used in every Swin forward pass
   - Can be done via reshape/transpose (no custom kernel)
   - Fallback: CPU implementation is fast enough

#### Fused Operations
- Leverage existing fused operations from Phase 8
- InvertedResidual: Conv-BN-ReLU fusion
- WindowAttention: QKV projection fusion

#### Multi-Scale Features
- FeaturePyramidNetwork: Parallel computation across levels
- Use async CUDA streams for concurrent processing

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1, 20h)
**Priority:** High
**Components:**
- [ ] PatchEmbedding (4h)
- [ ] SqueezeExcitation (3h)
- [ ] InvertedResidual (5h)
- [ ] FusedMBConv (3h)
- [ ] Unfold/Fold operations (5h)

**Deliverables:**
- Basic ViT components functional
- MobileNet/EfficientNet blocks ready
- Unit tests for all components

### Phase 2: Swin Transformer (Week 2, 15h)
**Priority:** High
**Components:**
- [ ] WindowAttention (8h)
- [ ] Window partition/reverse utilities (4h)
- [ ] Shifted window mask generation (3h)

**Deliverables:**
- Complete Swin Transformer block
- Integration test with full Swin-T model
- Gradient verification

### Phase 3: Detection Foundations (Week 3, 20h)
**Priority:** Critical
**Components:**
- [ ] AnchorGenerator (4h)
- [ ] Box IoU computation (3h)
- [ ] Box encoding/decoding (3h)
- [ ] NMS (CPU implementation) (5h)
- [ ] NMS (CUDA kernel) (5h)

**Deliverables:**
- Complete anchor and NMS pipeline
- Performance benchmark vs PyTorch
- Unit tests with reference outputs

### Phase 4: ROI Operations (Week 4, 15h)
**Priority:** Critical
**Components:**
- [ ] ROIAlign (CPU implementation) (5h)
- [ ] ROIAlign (CUDA kernel) (8h)
- [ ] ROIAlign backward pass (2h)

**Deliverables:**
- Bit-accurate ROIAlign vs PyTorch
- Gradient check passing
- Performance benchmark

### Phase 5: Detection Networks (Week 5, 15h)
**Priority:** Medium
**Components:**
- [ ] RPNHead (4h)
- [ ] RegionProposalNetwork (6h)
- [ ] FeaturePyramidNetwork (5h)

**Deliverables:**
- Complete RPN module
- Integration test with ResNet backbone
- Proposal quality validation

### Phase 6: Segmentation Components (Week 6, 10h)
**Priority:** Medium
**Components:**
- [ ] ASPP (5h)
- [ ] DeepLabV3 decoder (5h)

**Deliverables:**
- Complete segmentation modules
- Integration test with DeepLabV3

### Testing & Validation (Parallel, 25h)
**Throughout all phases:**
- [ ] Unit tests for each component (15h)
- [ ] Integration tests with full models (5h)
- [ ] Gradient checks (3h)
- [ ] Performance benchmarks (2h)

---

## Success Criteria

### Functional Requirements
- ✅ All 26 new components implemented
- ✅ Full autograd support (gradients flow correctly)
- ✅ CPU and CUDA backends functional
- ✅ API matches design specifications

### Quality Requirements
- ✅ 100% unit test coverage
- ✅ Gradient checks passing (numerical gradient validation)
- ✅ Bit-accurate with PyTorch reference (where applicable)
- ✅ No memory leaks (valgrind clean)

### Performance Requirements
- ✅ ROIAlign: < 10ms for 1000 ROIs (CUDA, V100)
- ✅ NMS: < 5ms for 1000 boxes (CUDA, V100)
- ✅ WindowAttention: Within 10% of PyTorch
- ✅ PatchEmbedding: Within 5% of PyTorch (conv-based)

### Integration Requirements
- ✅ ViT model runs end-to-end
- ✅ Swin Transformer trains successfully
- ✅ Faster R-CNN inference produces valid proposals
- ✅ Mask R-CNN mask quality matches PyTorch

---

## Risk Mitigation

### High-Risk Components

#### 1. ROIAlign CUDA Kernel
**Risk:** Complex bilinear interpolation, backward pass tricky
**Mitigation:**
- Start with CPU implementation (reference)
- Port PyTorch CUDA kernel (MIT license)
- Extensive gradient checking
- Fallback: CPU-only for initial release

#### 2. NMS Performance
**Risk:** Naive NMS is O(N^2), slow for many boxes
**Mitigation:**
- Use spatial hashing for large N (> 10K boxes)
- Implement batched NMS (parallel across classes)
- Benchmark against torchvision
- Consider cuDNN NMS if available

#### 3. Window Attention Memory
**Risk:** Swin Transformer can OOM on large images
**Mitigation:**
- Implement gradient checkpointing
- Profile memory usage carefully
- Provide window_size configuration
- Document memory requirements

### Medium-Risk Components

#### 4. Unfold/Fold Operations
**Risk:** Memory explosion with small stride
**Mitigation:**
- Document memory requirements clearly
- Validate input shapes before allocation
- Consider chunked processing for large inputs

#### 5. Anchor Generation Scalability
**Risk:** 50K+ anchors for large images
**Mitigation:**
- Cache anchors per image size
- Lazy generation on first use
- Efficient tensor operations (no loops)

---

## Appendix: Component Dependency Graph

```
PatchEmbedding
    ├─> Conv2d (existing)
    └─> reshape (existing)

WindowAttention
    ├─> Linear (existing)
    ├─> Softmax (existing)
    └─> Dropout (existing)

SqueezeExcitation
    ├─> AdaptiveAvgPool2d (existing)
    └─> Linear (existing)

InvertedResidual
    ├─> Conv2d (existing)
    ├─> BatchNorm2d (existing)
    └─> SqueezeExcitation (NEW)

ROIAlign
    └─> grid_sample (NEW - custom CUDA)

RegionProposalNetwork
    ├─> AnchorGenerator (NEW)
    ├─> Conv2d (existing)
    ├─> NMS (NEW)
    └─> box encoding/decoding (NEW)

NMS
    └─> box_iou (NEW)
```

**Critical Path:**
1. Implement box_iou, encode/decode (no dependencies)
2. Implement NMS (depends on box_iou)
3. Implement AnchorGenerator (no dependencies)
4. Implement RPNHead (depends on Conv2d - existing)
5. Implement RegionProposalNetwork (depends on all above)

**Parallel Development:**
- Vision Transformer components (PatchEmbedding, WindowAttention) can be developed in parallel with detection components
- MobileNet blocks (SE, InvertedResidual) can be developed in parallel
- ROIAlign can be developed independently

---

## Conclusion

This architecture provides a complete specification for implementing all missing Phase 9 components. The design:

1. **Leverages Existing Infrastructure:** Reuses Conv2d, Linear, MultiheadAttention, etc.
2. **Follows Established Patterns:** All modules inherit from Module, use Variable for autograd
3. **Prioritizes Critical Components:** CUDA kernels for ROIAlign and NMS
4. **Enables Model Zoo:** All building blocks for ViT, Swin, EfficientNet, MobileNet, Faster R-CNN, Mask R-CNN
5. **Maintains Quality:** 100% test coverage, gradient checks, performance benchmarks

**Estimated Total Effort:** 70 hours (45h implementation + 25h testing)

**Ready for Implementation:** ✅

---

**Document Prepared By:** Claude (System Architecture Designer)
**Review Status:** Pending technical review
**Next Steps:** Begin Phase 1 implementation
