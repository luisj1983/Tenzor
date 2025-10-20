# Swin Transformer Implementation

**Date:** 2025-10-18
**Status:** Complete - Full Implementation
**Location:**
- Headers: `/include/tenzor/models/swin_transformer.hpp`, `/include/tenzor/nn/layers/vision.hpp`
- Source: `/src/models/swin_transformer.cpp`, `/src/nn/layers/vision.cpp`

## Overview

Complete implementation of Swin Transformer (Hierarchical Vision Transformer using Shifted Windows) as specified in the ICCV 2021 Best Paper by Liu et al.

## Architecture Components

### 1. WindowAttention (`vision.hpp/cpp`)

**Purpose:** Efficient window-based multi-head self-attention with linear complexity O(M²·H·W).

**Key Features:**
- Local 7×7 window attention (configurable)
- Learnable relative position bias table
- Pre-computed relative position indices
- Support for attention masks (SW-MSA)

**Parameters:**
- `dim`: Number of input channels
- `window_size`: Window size M (default: 7)
- `num_heads`: Number of attention heads
- `qkv_bias`: Add bias to QKV projections
- `qk_scale`: Override default QK scale
- `attn_drop`: Attention dropout rate
- `proj_drop`: Projection dropout rate

**Implementation Details:**
```cpp
// Relative position bias table shape: (2M-1, 2M-1, num_heads)
// For M=7: (13, 13, num_heads)
// Range: [-6, +6] for each dimension

// Pre-computed index: (M*M, M*M) -> maps each (query, key) pair to bias table
```

**Critical Functions:**
- `compute_relative_position_index()`: Pre-computes position indices once during initialization
- `get_relative_position_bias()`: Gathers bias from table using indices
- `forward()`: Applies window attention with optional masking

### 2. Helper Functions

**window_partition():**
- Input: (B, H, W, C)
- Output: (B·num_windows, M·M, C)
- Partitions feature map into non-overlapping M×M windows

**window_reverse():**
- Input: (B·num_windows, M·M, C)
- Output: (B, H, W, C)
- Merges windows back into feature map

**create_shifted_window_mask():**
- Creates attention mask for SW-MSA
- Prevents attention between non-adjacent regions after cyclic shift
- Mask shape: (num_windows, M·M, M·M)
- Masked positions: -100.0

### 3. SwinTransformerBlock

**Architecture:**
```
Input (B, H·W, C)
  ↓
LayerNorm
  ↓
Reshape to (B, H, W, C)
  ↓
Cyclic Shift (if SW-MSA)
  ↓
Window Partition
  ↓
Window Attention
  ↓
Window Reverse
  ↓
Reverse Cyclic Shift (if SW-MSA)
  ↓
Reshape to (B, H·W, C)
  ↓
Residual Connection
  ↓
LayerNorm
  ↓
MLP (expansion ratio 4.0)
  ↓
Residual Connection
  ↓
Output
```

**W-MSA vs SW-MSA:**
- Even blocks (0, 2, 4...): W-MSA with `shift_size=0`
- Odd blocks (1, 3, 5...): SW-MSA with `shift_size=window_size/2`

### 4. PatchMerging

**Downsampling Layer:**
```
Input: (B, H, W, C)
  ↓
Concatenate 2×2 neighbors: (B, H/2, W/2, 4C)
  ↓
LayerNorm
  ↓
Linear: 4C → 2C
  ↓
Output: (B, H/2, W/2, 2C)
```

**Extracts 4 patches:**
- x0: [0::2, 0::2] (top-left)
- x1: [1::2, 0::2] (bottom-left)
- x2: [0::2, 1::2] (top-right)
- x3: [1::2, 1::2] (bottom-right)

### 5. BasicLayer (Stage)

**Composition:**
- Multiple SwinTransformerBlock instances
- Alternating W-MSA and SW-MSA blocks
- Optional PatchMerging for downsampling

**Stochastic Depth:**
- Linearly scaled drop_path rates across depth
- Provides regularization for deep models

### 6. PatchEmbed

**Initial Patch Embedding:**
```
Image (B, 3, 224, 224)
  ↓
Conv2d 4×4, stride=4
  ↓
(B, embed_dim, 56, 56)
  ↓
Permute to (B, 56, 56, embed_dim)
  ↓
Optional LayerNorm
  ↓
Output: (B, num_patches, embed_dim)
```

### 7. SwinTransformer (Main Model)

**Overall Architecture:**
```
Input: 224×224×3
  ↓
Patch Embed (4×4) → 56×56×C
  ↓
Stage 1: 56×56×C,    depth[0] blocks, heads[0]
  ↓ Patch Merge
Stage 2: 28×28×2C,   depth[1] blocks, heads[1]
  ↓ Patch Merge
Stage 3: 14×14×4C,   depth[2] blocks, heads[2]
  ↓ Patch Merge
Stage 4: 7×7×8C,     depth[3] blocks, heads[3]
  ↓
LayerNorm + Global Average Pool
  ↓
Linear: num_classes
```

## Model Variants

### Swin-Tiny
```cpp
auto model = swin_tiny(1000);
```
- **Embed Dim:** 96
- **Depths:** [2, 2, 6, 2]
- **Num Heads:** [3, 6, 12, 24]
- **Parameters:** ~29M
- **FLOPs @ 224×224:** 4.5G
- **ImageNet Top-1:** ~81.3%

### Swin-Small
```cpp
auto model = swin_small(1000);
```
- **Embed Dim:** 96
- **Depths:** [2, 2, 18, 2]
- **Num Heads:** [3, 6, 12, 24]
- **Parameters:** ~50M
- **FLOPs @ 224×224:** 8.7G
- **ImageNet Top-1:** ~83.0%

### Swin-Base
```cpp
auto model = swin_base(1000);
```
- **Embed Dim:** 128
- **Depths:** [2, 2, 18, 2]
- **Num Heads:** [4, 8, 16, 32]
- **Parameters:** ~88M
- **FLOPs @ 224×224:** 15.4G
- **ImageNet Top-1:** ~83.5%

### Swin-Large
```cpp
auto model = swin_large(1000);
```
- **Embed Dim:** 192
- **Depths:** [2, 2, 18, 2]
- **Num Heads:** [6, 12, 24, 48]
- **Parameters:** ~197M
- **FLOPs @ 224×224:** 34.5G
- **ImageNet Top-1:** ~84.5%

## Key Implementation Details

### 1. Cyclic Shift

**Implementation:**
```cpp
// Forward shift: roll by (-shift_size, -shift_size)
x_tensor = x_tensor.roll(-shift_size_, 1);  // height
x_tensor = x_tensor.roll(-shift_size_, 2);  // width

// Reverse shift: roll by (+shift_size, +shift_size)
x_tensor = x_tensor.roll(shift_size_, 1);
x_tensor = x_tensor.roll(shift_size_, 2);
```

### 2. Relative Position Bias

**Pre-computation:**
- Index table computed once in constructor
- Maps each (query_pos, key_pos) to table index
- Range: [0, (2M-1)²-1]

**Runtime:**
- Simple table lookup using pre-computed indices
- O(1) complexity per position pair
- Bias shape: (num_heads, M·M, M·M)

### 3. Attention Mask for SW-MSA

**Mask Creation:**
1. Partition image into 9 regions after shift
2. Assign region ID to each pixel
3. Partition into windows
4. Mask positions where region IDs differ

**Mask Values:**
- Same region: 0.0
- Different region: -100.0 (effectively -∞ after softmax)

### 4. Hierarchical Features

**forward_features() method:**
Returns features from all 4 stages:
- Stage 1: (B, H/4, W/4, C)
- Stage 2: (B, H/8, W/8, 2C)
- Stage 3: (B, H/16, W/16, 4C)
- Stage 4: (B, H/32, W/32, 8C)

Useful for:
- Object detection (FPN backbone)
- Semantic segmentation
- Dense prediction tasks

## Usage Examples

### Basic Classification

```cpp
#include "tenzor/models/swin_transformer.hpp"

// Create Swin-Tiny model
auto model = tenzor::models::swin_tiny(1000);

// Forward pass
Variable input({1, 3, 224, 224});
Variable output = model->forward(input);
// output shape: {1, 1000}
```

### Feature Extraction

```cpp
// Create model
auto model = tenzor::models::swin_base(1000);

// Extract hierarchical features
Variable input({1, 3, 224, 224});
auto features = model->forward_features(input);

// features[0]: Stage 1 output (1, 56, 56, 128)
// features[1]: Stage 2 output (1, 28, 28, 256)
// features[2]: Stage 3 output (1, 14, 14, 512)
// features[3]: Stage 4 output (1, 7, 7, 1024)
```

### Custom Configuration

```cpp
// Create custom Swin model
auto model = std::make_shared<tenzor::models::SwinTransformer>(
    224,                                    // img_size
    4,                                      // patch_size
    3,                                      // in_chans
    1000,                                   // num_classes
    96,                                     // embed_dim
    std::vector<int64_t>{2, 2, 6, 2},      // depths
    std::vector<int64_t>{3, 6, 12, 24},    // num_heads
    7,                                      // window_size
    4.0,                                    // mlp_ratio
    true,                                   // qkv_bias
    0.0,                                    // qk_scale
    0.1,                                    // drop_rate
    0.1,                                    // attn_drop_rate
    0.2,                                    // drop_path_rate
    true                                    // norm_layer
);
```

## Computational Complexity

### Standard Attention (ViT)
- Complexity: O(N²·C) where N = H·W
- Memory: O(N²) for attention scores

### Window Attention (Swin)
- Complexity: O(M²·N·C) where M = window_size
- Memory: O(M²) per window
- **Reduction:** From O((H·W)²) to O(M²·H·W) = Linear!

### Example (224×224 image):
- **ViT:** N = 14×14 = 196, Complexity ∝ 196² = 38,416
- **Swin:** M = 7, num_windows = 8×8 = 64, Complexity ∝ 7² × 64 = 3,136
- **Speedup:** ~12× reduction in attention complexity

## Optimizations

### 1. Efficient Window Operations
- Use tensor reshaping and permutation (no data copying)
- Batch all windows together for parallel attention

### 2. Pre-computed Indices
- Relative position indices computed once
- O(1) lookup during forward pass

### 3. Cached Attention Masks
- Masks computed once per block
- Reused across all forward passes

### 4. Fused Operations
- QKV computed in single matrix multiplication
- Drop path combined with residual connection

## Testing Recommendations

### Unit Tests

```cpp
// Test 1: Window partition/reverse
TEST(SwinTest, WindowPartitionReverse) {
    Variable input({2, 56, 56, 96});
    int64_t window_size = 7;

    auto windows = window_partition(input, window_size);
    // Expected shape: (128, 49, 96)
    // 128 = 2 * 8 * 8 (batch * num_windows)

    auto reversed = window_reverse(windows, window_size, 56, 56);
    // Should match input shape
    EXPECT_EQ(reversed.shape(), input.shape());
}

// Test 2: Window attention
TEST(SwinTest, WindowAttention) {
    WindowAttention attn(96, 7, 3);
    Variable input({128, 49, 96});  // num_windows*batch, M*M, C

    auto output = attn.forward(input);
    EXPECT_EQ(output.shape(), input.shape());
}

// Test 3: Swin block
TEST(SwinTest, SwinTransformerBlock) {
    SwinTransformerBlock block(96, {56, 56}, 3, 7, 0);
    Variable input({2, 3136, 96});  // B, H*W, C

    auto output = block.forward(input);
    EXPECT_EQ(output.shape(), input.shape());
}

// Test 4: Full model
TEST(SwinTest, SwinTiny) {
    auto model = swin_tiny(1000);
    Variable input({1, 3, 224, 224});

    auto output = model->forward(input);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1000);
}
```

### Integration Tests

```cpp
// Test gradient flow
TEST(SwinTest, GradientFlow) {
    auto model = swin_tiny(10);
    Variable input({2, 3, 224, 224}, true);

    auto output = model->forward(input);
    auto loss = output.sum();
    loss.backward();

    // Check gradients exist
    EXPECT_TRUE(input.grad().has_value());
}

// Test hierarchical features
TEST(SwinTest, HierarchicalFeatures) {
    auto model = swin_base(1000);
    Variable input({1, 3, 224, 224});

    auto features = model->forward_features(input);
    EXPECT_EQ(features.size(), 4);
    EXPECT_EQ(features[0].shape(), std::vector<int64_t>({1, 56, 56, 128}));
    EXPECT_EQ(features[3].shape(), std::vector<int64_t>({1, 7, 7, 1024}));
}
```

## Comparison with Specification

| Component | Specification | Implementation | Status |
|-----------|--------------|----------------|--------|
| Window Size | 7×7 | Configurable, default 7 | ✅ Complete |
| Relative Position Bias | Learnable table | Pre-computed indices + table | ✅ Complete |
| Cyclic Shift | (-M/2, -M/2) | Roll operation | ✅ Complete |
| W-MSA/SW-MSA Alternation | Even/odd blocks | Implemented | ✅ Complete |
| Patch Merging | 2×2 downsample | Concatenate + Linear | ✅ Complete |
| Hierarchical Stages | 4 stages | BasicLayer | ✅ Complete |
| Swin-Tiny | [2,2,6,2] blocks | Factory function | ✅ Complete |
| Swin-Small | [2,2,18,2] blocks | Factory function | ✅ Complete |
| Swin-Base | [2,2,18,2] blocks | Factory function | ✅ Complete |
| Swin-Large | [2,2,18,2] blocks | Factory function | ✅ Complete |

## Files Created/Modified

### New Headers
- `/include/tenzor/models/swin_transformer.hpp` - Main model interface
- `/include/tenzor/nn/layers/vision.hpp` - WindowAttention layer (updated)

### New Source Files
- `/src/models/swin_transformer.cpp` - Complete implementation

### Modified Files
- `/src/nn/layers/vision.cpp` - WindowAttention already implemented

## Dependencies

### Required Headers
```cpp
#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/normalization.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/layers/vision.hpp"
#include "../nn/activations/activations.hpp"
```

### Required Operations
- `reshape`, `permute`, `transpose`
- `slice`, `squeeze`, `unsqueeze`
- `matmul`, `softmax`
- `roll` (for cyclic shift)
- `cat` (for concatenation)

## Notes

1. **NO STUBS:** All functions are fully implemented
2. **Cyclic Shift:** Uses tensor roll operation
3. **Attention Mask:** Pre-computed for shifted windows
4. **Relative Position Bias:** Learnable parameter with efficient indexing
5. **Head Dimension:** Fixed at 32 for all variants (dim / num_heads)
6. **Drop Path:** Stochastic depth with linear scaling

## Future Enhancements

1. **Pretrained Weights:** Add weight loading from official checkpoints
2. **Flash Window Attention:** Optimize for memory efficiency
3. **Mixed Precision:** FP16 support for faster inference
4. **Backbone Mode:** Export for detection/segmentation frameworks
5. **ONNX Export:** For deployment on various platforms

## Reference

**Paper:** "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
**Authors:** Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
**Conference:** ICCV 2021 (Best Paper Award)
**arXiv:** https://arxiv.org/abs/2103.14030
**Official Code:** https://github.com/microsoft/Swin-Transformer
